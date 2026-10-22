/* plugin/spechub/spechub.cpp -- the two coupling ops' schemas, shapes and argument validation.
 *
 * Shape hooks and geometry only: the arithmetic is in abi/rad_spechub.h (inline, host and device) and
 * the launches are in spechub.hip and ref_spechub.cpp. This file is host C++ and compiles with no
 * card, which is what lets a card-less machine check the plugin's schema and shapes.
 */
#include "spechub.h"

#include <climits>
#include <cstddef>

namespace {

#define PI(k) { k, RAD_P_INT, RAD_REQUIRED, 0 }
const RadParamSpec kProposeParams[] = {
    PI("M"), PI("n_vocab"), PI("round"), PI("max_rounds"),
};
const RadParamSpec kVerifyParams[] = { PI("M"), PI("n_cand"), PI("n_vocab") };
#undef PI

/* `idx` is INOUT: the round's proposal IS the head's column 0, and a sampled row overwrites it. */
const RadOperandSpec kProposeOperands[] = {
    { "idx",     RAD_OPD_INOUT, 0 },
    { "val",     RAD_OPD_IN,    0 },
    { "seq_ids", RAD_OPD_IN,    0 },
    { "draw",    RAD_OPD_IN,    0 },
    { "state",   RAD_OPD_INOUT, 0 },
};

/* The verify runs after the ordinary pick over the SAME final candidate plane, so its operands are
 * the sampler's own: candidates, parameter rows, coupled state rows, and the token plane the pick
 * already wrote. */
const RadOperandSpec kVerifyOperands[] = {
    { "cand_idx", RAD_OPD_IN,    0 },
    { "cand_val", RAD_OPD_IN,    0 },
    { "params",   RAD_OPD_IN,    0 },
    { "state",    RAD_OPD_IN,    0 },
    { "token",    RAD_OPD_INOUT, 0 },
};

const RadOpSchema kSchemas[] = {
    { "mtp_couple_propose", kProposeParams, 4, kProposeOperands, 5,
      "Linear MTP round `round`: x~Q, Q = FP64 T1 softmax of the final top8 scores; x into idx "
      "column 0 and the state into state[round-1][row]. mode0 rows keep column 0 (status 0); bad "
      "ids/scores or seq_ids != draw.card keep it too (status -1)." },
    { "mtp_couple_verify", kVerifyParams, 3, kVerifyOperands, 5,
      "Maximal coupling of a bound mtp_couple state after sample_pick over all surviving target "
      "candidates, FP64 P and 53-bit ACCEPT/RESID uniforms at params.pos; residual (P-Q)+ with "
      "r(x)=0. Rows whose state is not status 1, not bound to params seed/pos, malformed, or whose "
      "(P,Q) admit no coupling keep the ordinary token." },
};

}  /* namespace */

int rad_spechub_schema_count() { return 2; }
const RadOpSchema* rad_spechub_schema_at(int i) {
    return i >= 0 && i < 2 ? &kSchemas[i] : nullptr;
}

namespace {

/* The geometry both hooks and the arg readers agree on. `op` 0 proposes, 1 verifies. A RANGE `M`
 * reads as its high bound, which is what a shape hook must do: it sizes buffers before selection has
 * collapsed anything, so it describes the whole band. */
int couple_geometry(const RadParam* p, int n, int op, bool range, int64_t* M, int64_t* vocab,
                    int64_t* n_cand) {
    if (!p) return RAD_E_INVAL;
    *M = range ? rad_param_getdim(p, n, "M", 0) : rad_param_geti(p, n, "M", 0);
    *vocab  = rad_param_geti(p, n, "n_vocab", 0);
    *n_cand = op ? rad_param_geti(p, n, "n_cand", 0) : 8;
    if (*M < 1 || *M > INT32_MAX || *vocab < 1 || *vocab > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (*n_cand < 1 || *n_cand > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (!op) {
        const int64_t round  = rad_param_geti(p, n, "round", 0);
        const int64_t rounds = rad_param_geti(p, n, "max_rounds", 0);
        if (rounds < 1 || rounds > RAD_MTP_COUPLE_MAX_ROUNDS || round < 1 || round > rounds)
            return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* Does one operand's extent fit the shape its hook declares? A row-owning kernel reads a whole row,
 * so the last axis is structural and every leading one is a row count. */
int span(const RadTensor& t, const RadOpdDesc& d, uint64_t* bytes) {
    if (!t.data || (uintptr_t)t.data % 4) return RAD_E_INVAL;
    if (t.dtype != d.dtype) return RAD_E_DTYPE;
    if (t.rank != d.rank) return RAD_E_SHAPE;
    uint64_t words = 1;
    for (int j = (int)d.rank - 1; j >= 0; --j) {
        if (t.shape[j] < 1 || t.shape[j] > INT32_MAX || (d.shape[j] > 0 && t.shape[j] != d.shape[j]))
            return RAD_E_SHAPE;
        if (t.stride[j] < 1 || (j == (int)d.rank - 1 && t.stride[j] != 1) ||
            (j < (int)d.rank - 1 && (uint64_t)t.stride[j] < words))
            return RAD_E_STRIDE;
        const uint64_t extent = (uint64_t)(t.shape[j] - 1);
        if (extent && (uint64_t)t.stride[j] > (INT64_MAX / 4 - words) / extent) return RAD_E_SHAPE;
        words += extent * (uint64_t)t.stride[j];
    }
    *bytes = words * 4;
    if ((uintptr_t)t.data > UINTPTR_MAX - *bytes) return RAD_E_INVAL;
    return RAD_OK;
}

}  /* namespace */

int rad_mtp_couple_propose_shape(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (!out || i < 0 || i >= 5) return RAD_E_INVAL;
    int64_t M, v, nc;
    int rc = couple_geometry(p, n, 0, true, &M, &v, &nc);
    if (rc != RAD_OK) return rc;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.dtype = RAD_I32;
    d.rank = 2;
    d.shape[0] = M;
    d.shape[1] = 8;
    if (i == 0) { d.fill = RAD_FILL_INDEX; d.idx_max = (int32_t)v; }
    if (i == 1) d.dtype = RAD_F32;
    if (i == 2) { d.rank = 1; d.fill = RAD_FILL_INDEX; d.idx_max = (int32_t)M; }
    if (i == 3) { d.dtype = RAD_U32; d.shape[1] = RAD_MTP_COUPLE_DRAW_WORDS; }
    if (i == 4) {
        /* Rounds are the state plane's leading axis, and `max_rounds` is exactly the bound the caller
         * sized it at. */
        const int64_t rounds = rad_param_geti(p, n, "max_rounds", 1);
        d.dtype = RAD_U32;
        d.rank = 3;
        d.shape[0] = rounds;
        d.shape[1] = M;
        d.shape[2] = RAD_MTP_COUPLE_STATE_WORDS;
    }
    *out = d;
    return RAD_OK;
}

int rad_mtp_couple_verify_shape(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (!out || i < 0 || i >= 5) return RAD_E_INVAL;
    int64_t M, v, nc;
    int rc = couple_geometry(p, n, 1, true, &M, &v, &nc);
    if (rc != RAD_OK) return rc;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.dtype = RAD_I32;
    d.rank = 2;
    d.shape[0] = M;
    d.shape[1] = nc;
    if (i == 1) d.dtype = RAD_F32;
    if (i == 2 || i == 3) {
        d.dtype = RAD_U32;
        d.shape[1] = i == 2 ? 24 : RAD_MTP_COUPLE_STATE_WORDS;
    }
    if (i == 4) d.rank = 1;
    *out = d;
    return RAD_OK;
}

/* The resolved `M` is the arena band's ceiling; this issue may carry fewer live rows, and both
 * kernels take the row count from operand 0. Dim 0 is narrowed to the live rows before each
 * operand's extent is checked, the wire planes must be eight-byte aligned with an even row stride,
 * and separate storage is part of the contract: an aliased pair would let a write land on a read. */
int rad_couple_validate(const RadArgs* a, int op, int64_t* M, int64_t* vocab, int64_t* n_cand) {
    if (!a || !a->t || a->n_t != 5) return RAD_E_INVAL;
    int rc = couple_geometry(a->p, a->n_p, op, false, M, vocab, n_cand);
    if (rc != RAD_OK) return rc;
    const int64_t live = a->t[0].rank ? a->t[0].shape[0] : 0;
    if (live < 1 || live > *M) return RAD_E_SHAPE;
    *M = live;
    const int64_t rounds = op ? 1 : rad_param_geti(a->p, a->n_p, "max_rounds", 1);
    uint64_t bytes[5] = {};
    for (int i = 0; i < 5; ++i) {
        RadOpdDesc d{};
        rc = op ? rad_mtp_couple_verify_shape(a->p, a->n_p, i, &d)
                : rad_mtp_couple_propose_shape(a->p, a->n_p, i, &d);
        if (rc != RAD_OK) return rc;
        d.shape[0] = *M;
        /* `seq_ids` covers the whole batch and the state plane is [rounds, rows]; both are checked
         * below rather than against the live row count. */
        if (!op && i == 2) d.shape[0] = 0;
        if (!op && i == 4) { d.shape[0] = rounds; d.shape[1] = 0; }
        /* The sampler exposes a flat token plane and its production [M, 1] one. */
        if (op && i == 4 && a->t[i].rank == 2) { d.rank = 2; d.shape[1] = 1; }
        if ((rc = span(a->t[i], d, &bytes[i])) != RAD_OK) return rc;
        if (!op && i == 2 && a->t[i].shape[0] < *M) return RAD_E_SHAPE;
        if (!op && i == 4 && a->t[i].shape[1] < *M) return RAD_E_SHAPE;
        const bool wire = op ? (i == 2 || i == 3) : (i == 3 || i == 4);
        if (wire && ((uintptr_t)a->t[i].data % 8 || a->t[i].stride[0] % 2 ||
                     (a->t[i].rank == 3 && a->t[i].stride[1] % 2)))
            return RAD_E_INVAL;
    }
    /* An even ROW stride keeps every shortlist row 8-byte aligned; the innermost stride is 1. */
    if (!op && a->t[0].stride[0] % 2) return RAD_E_STRIDE;
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j) {
            const uintptr_t x = (uintptr_t)a->t[i].data, y = (uintptr_t)a->t[j].data;
            if (x < y + bytes[j] && y < x + bytes[i]) return RAD_E_INVAL;
        }
    return RAD_OK;
}

int rad_mtp_couple_propose_args(const RadArgs* a, RadMtpCoupleProposeCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M, v, nc;
    const int rc = rad_couple_validate(a, 0, &M, &v, &nc);
    if (rc != RAD_OK) return rc;
    RadMtpCoupleProposeCall c{};
    c.M = M;
    c.vocab = v;
    c.round = (int)rad_param_geti(a->p, a->n_p, "round", 0);
    c.idx   = (int32_t*)a->t[0].data;
    c.val   = (const float*)a->t[1].data;
    c.card  = (const int32_t*)a->t[2].data;
    c.draw  = (const uint32_t*)a->t[3].data;
    c.state = (uint32_t*)a->t[4].data;
    c.ld_idx   = a->t[0].stride[0];
    c.ld_val   = a->t[1].stride[0];
    c.ld_draw  = a->t[3].stride[0];
    c.st_round = a->t[4].stride[0];
    c.st_row   = a->t[4].stride[1];
    *out = c;
    return RAD_OK;
}

int rad_mtp_couple_verify_args(const RadArgs* a, RadMtpCoupleVerifyCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M, v, nc;
    const int rc = rad_couple_validate(a, 1, &M, &v, &nc);
    if (rc != RAD_OK) return rc;
    RadMtpCoupleVerifyCall c{};
    c.M = M;
    c.vocab = v;
    c.n_cand = nc;
    c.idx    = (const int32_t*)a->t[0].data;
    c.val    = (const float*)a->t[1].data;
    c.params = (const uint32_t*)a->t[2].data;
    c.state  = (const uint32_t*)a->t[3].data;
    c.token  = (int32_t*)a->t[4].data;
    for (int i = 0; i < 5; ++i) c.ld[i] = a->t[i].stride[0];
    *out = c;
    return RAD_OK;
}

RadKernelInfo rad_spechub_row(int i, RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = i ? "spechub_mtp_couple_verify" : "spechub_mtp_couple_propose";
    r.op = i ? "mtp_couple_verify" : "mtp_couple_propose";
    r.family = "draft";
    r.computes = i ? "maximal coupling over the surviving target candidates, FP64 P"
                   : "x~Q over the final top8, FP64 T1 softmax";
    r.shape = i ? "M,n_cand,n_vocab" : "M,n_vocab,round,max_rounds<=8";
    r.dtypes = "f32,i32,u32";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.launch = launch;
    r.opd_shape = i ? rad_mtp_couple_verify_shape : rad_mtp_couple_propose_shape;
    return r;
}
