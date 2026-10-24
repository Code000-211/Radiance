/* dflash_tree_select.cpp -- schema, operand description and call validation for DFlash2's bounded
 * DarTree selector. The definition of the op is mine/dflash_tree_select.h; the arithmetic lives in
 * mine/dflash_tree_select.hip (device) and mine/ref_dflash_tree.cpp (host oracle).
 *
 * THE SCHEMA IS `dflash_select`'s, PLUS THE TREE'S OWN BOUNDS. Parameters `M`, `steps`, `top_k`,
 * `rank`, `n_vocab`, `anchor_stride`, `dtype` are spelled exactly as the linear selector spells
 * them, because this op is selected where that one is and a caller that has the geometry already
 * has it in this form. `tree_budget`, `tree_width` and `depth_penalty` are the three the tree adds.
 *
 * THE OPERANDS ARE `dflash_select`'s FIRST SIX, IN ITS ORDER, AND THE TREE REPLACES `tokens`:
 * cand, unary, hp?, anchor, pred, succ -> packed_tree. Keeping the six positions means the caller
 * does not re-describe the candidate planes, and `hp` stays OPTIONAL AT INDEX 2 because a graph that
 * does not use the hidden projection still passes an empty operand there (the absent-operand rule
 * in abi/rad_plugin.h: absent is a null `data`, never a shorter array).
 *
 * ONE SCHEMA, TWO DOMAINS. `mine` and `mine_ref` publish the same schema and different rows, which
 * is the ordinary host-oracle convention in this tree: `mine_ref`'s copy is the reference, and the
 * comparators select a row by name rather than by plugin priority.
 *
 * WHAT THIS FILE DOES NOT VALIDATE. Per-row data -- an anchor outside the vocabulary, a non-finite
 * unary term, an id that is not a codebook row -- is decided by the kernel and the oracle, because
 * it is read from device memory at launch time and the contract says such a row produces
 * `count = 0` rather than a refusal. This file refuses only what a call's METADATA can prove
 * unserviceable, which is the same split every other row here makes. */
#include "dflash_tree_select.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kParams = 10;

/* `dtype` is the operand encoding, not a mode: cand/anchor are I32 by the op's definition, unary
 * is F32 (the head's widened logits, which is what row_topk emits) and the codebooks are BF16. So
 * `dtype` names the CODEBOOK and candidate encoding family and "bf16" is the only value that
 * exists. Anything else is a refusal rather than a conversion. */
#define P_INT(k) { k, RAD_P_INT, RAD_REQUIRED, 0 }
#define P_F64(k) { k, RAD_P_F64, RAD_REQUIRED, 0 }
#define P_STR(k) { k, RAD_P_STR, RAD_REQUIRED, 0 }
#define IN(n)    { n, RAD_OPD_IN, 0 }
#define WGT(n)   { n, RAD_OPD_WEIGHT, 0 }
#define OUT(n)   { n, RAD_OPD_OUT, 0 }

static const RadParamSpec kParamSpecs[] = {
    P_INT("M"), P_INT("steps"), P_INT("top_k"), P_INT("rank"), P_INT("n_vocab"),
    P_INT("anchor_stride"), P_INT("tree_budget"), P_INT("tree_width"), P_F64("depth_penalty"),
    P_STR("dtype"),
};

static const RadOperandSpec kOperandSpecs[] = {
    IN("cand"), IN("unary"), { "hp", RAD_OPD_IN, 1 }, IN("anchor"), WGT("pred"), WGT("succ"),
    OUT("packed_tree"),
};

static const RadOpSchema kSchema = {
    "dflash_tree_select", kParamSpecs, kParams, kOperandSpecs, RAD_DFLASH_TREE_OPERANDS,
    "A bounded DarTree over DFlash2's candidate lattice: every parent at level l scores every "
    "candidate at level l+1 with the linear selector's own FP32 edge "
    "unary + sum_r pred[pid][r]*hp[row][r]*succ[cand][r], each parent's set is normalised into a "
    "log probability, at most tree_width successors a level and tree_budget nodes survive, and "
    "the survivors are packed root-inclusive into [count, token[16], parent[16]] with the anchor "
    "as node 0. Pruning keeps every ancestor of every kept node. steps <= 3, budget <= 15."
};
#undef P_INT
#undef P_F64
#undef P_STR
#undef IN
#undef WGT
#undef OUT

/* The geometry every entry point agrees on. Reading it once keeps the shape hook, the validator
 * and the row's constraints from drifting apart, which for four independent readers of the same
 * seven names is the failure this tree has already paid for. */
struct Geom {
    int64_t M, steps, K, R, V, astr, budget, width;
    float beta;
};

/* Returns RAD_OK or the refusal. `wide` selects rad_param_getdim (a shape hook, which reads a
 * band's high bound) over rad_param_geti (a launch, where a surviving range is a selection bug).
 *
 * `M` IS TREATED DIFFERENTLY IN THE TWO DOMAINS, and deliberately. A shape hook must be told the
 * band's ceiling -- that is what a tool sizes a buffer from. A LAUNCH reads the sequence count from
 * the output tensor's rows, exactly as `dflash_select` reads its own `M` from `tokens`, because
 * RadArgs collapses a range to a value and a call that issued one sequence must not be refused for
 * not owning `max_seqs * steps` candidate rows. So a missing or non-positive `M` at launch is not
 * fatal; the band it would have supplied is simply absent. */
int geometry(const RadParam* p, int n_p, Geom* g, bool wide) {
    if (!p || !g) return RAD_E_INVAL;
    auto dim = [&](const char* k, int64_t dflt) {
        return wide ? rad_param_getdim(p, n_p, k, dflt) : rad_param_geti(p, n_p, k, dflt);
    };
    g->M      = dim("M", 0);
    g->steps  = dim("steps", 0);
    g->K      = dim("top_k", 0);
    g->R      = dim("rank", 0);
    g->V      = dim("n_vocab", 0);
    g->astr   = dim("anchor_stride", 1);
    g->budget = dim("tree_budget", 0);
    g->width  = dim("tree_width", 0);
    if (g->M <= 0 && wide) return RAD_E_UNSUPPORTED;
    if (g->M < 0) g->M = 0;
    if (g->steps <= 0 || g->K <= 0 || g->R <= 0 || g->V <= 0 || g->astr <= 0 ||
        g->budget <= 0 || g->width <= 0)
        return RAD_E_UNSUPPORTED;
    /* THE DECLARED BAND, REFUSED BY NAME. More levels would be out of distribution for a drafter
     * trained at block 4, more than 15 draft nodes cannot fit the verifier, and a width above the
     * candidate count is a request for candidates that do not exist. */
    if (g->steps > RAD_DFLASH_TREE_MAX_STEPS || g->K > RAD_DFLASH_TREE_MAX_K ||
        g->budget > RAD_DFLASH_TREE_MAX_BUDGET || g->width > g->K ||
        g->width > RAD_DFLASH_TREE_MAX_WIDTH || g->K > g->V)
        return RAD_E_UNSUPPORTED;
    /* `depth_penalty` must be finite and non-positive: the tree's prefix-closure argument is that
     * a child never outranks its parent, and a positive penalty would invert it. */
    g->beta = (float)rad_param_getf(p, n_p, "depth_penalty", 0.0);
    if (!std::isfinite(g->beta) || g->beta > 0.0f) return RAD_E_INVAL;
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_E_DTYPE;
    /* Every derived extent is a product a caller may allocate from, so the overflow is refused
     * here rather than described and then refused at issue. */
    if (g->M > INT64_MAX / g->steps || g->M * g->steps > INT64_MAX / g->K ||
        g->V > INT64_MAX / g->R || (int64_t)g->M * g->astr > INT64_MAX ||
        g->M * g->steps > INT64_MAX / g->astr)
        return RAD_E_UNSUPPORTED;
    if (g->M * g->steps > INT32_MAX || g->V > INT32_MAX || g->R > INT32_MAX) return RAD_E_SHAPE;
    return RAD_OK;
}

/* A dense rank-2 plane whose row pitch is at least the width the op reads. */
int64_t plane(const RadTensor& t, uint32_t dtype, int64_t rows, int64_t width) {
    if (!t.data || t.dtype != dtype || t.rank != 2 || t.shape[0] < rows || t.shape[1] < width)
        return -1;
    if (t.stride[1] != 1 || t.stride[0] < t.shape[1]) return -1;
    return t.stride[0];
}

/* A dense rank-1 plane. */
bool vector(const RadTensor& t, uint32_t dtype, int64_t n) {
    return t.data && t.dtype == dtype && t.rank == 1 && t.stride[0] == 1 && t.shape[0] >= n;
}

bool overlap(const RadTensor& a, const RadTensor& b) {
    if (!a.data || !b.data) return false;
    auto bytes = [](const RadTensor& t) {
        int64_t n = 1;
        for (uint32_t i = 0; i < t.rank; ++i) n += (t.shape[i] - 1) * t.stride[i];
        return (uint64_t)n * (t.dtype == RAD_BF16 ? 2u : 4u);
    };
    const uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
    const uint64_t xa = bytes(a), xb = bytes(b);
    return x < y + xb && y < x + xa;
}

}  /* namespace */

int rad_dflash_tree_select_schema_count(void) { return 1; }

const RadOpSchema* rad_dflash_tree_select_schema_at(int i) { return i == 0 ? &kSchema : nullptr; }

/* ---- operand description --------------------------------------------------------------------- */
/* `M` is the band's ceiling; the operands name the CALL. That is the same reading `dflash_select`'s
 * hook takes and the reason the launch reads rows from the tensors instead of from the parameter. */
int rad_dflash_tree_select_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= RAD_DFLASH_TREE_OPERANDS) return RAD_E_INVAL;
    Geom g{};
    const int rc = geometry(p, n_p, &g, true);
    if (rc != RAD_OK) return rc;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.fill = (uint32_t)RAD_FILL_NORMAL;
    const int64_t rows = g.M * g.steps;
    switch (operand) {
    case RAD_DFLASH_TREE_CAND:
        d.dtype = RAD_I32; d.rank = 2; d.shape[0] = rows; d.shape[1] = g.K;
        d.fill = (uint32_t)RAD_FILL_INDEX; d.idx_max = g.V;
        break;
    case RAD_DFLASH_TREE_UNARY:
        d.dtype = RAD_F32; d.rank = 2; d.shape[0] = rows; d.shape[1] = g.K;
        break;
    case RAD_DFLASH_TREE_HP:
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = rows; d.shape[1] = g.R;
        break;
    case RAD_DFLASH_TREE_ANCHOR:
        d.dtype = RAD_I32; d.rank = 1; d.shape[0] = g.M * g.astr;
        d.fill = (uint32_t)RAD_FILL_INDEX; d.idx_max = g.V;
        break;
    case RAD_DFLASH_TREE_PRED:
    case RAD_DFLASH_TREE_SUCC:
        d.dtype = RAD_BF16; d.rank = 2; d.shape[0] = g.V; d.shape[1] = g.R;
        break;
    default:
        /* THE PACKED TREE, root-inclusive: count, 16 tokens, 16 parents. Its width is the
         * verifier's own constant and not `budget`, because a tool sizing from this must get a
         * buffer the engine's tree declaration accepts whatever the budget of the moment is. */
        d.dtype = RAD_I32; d.rank = 2; d.shape[0] = g.M; d.shape[1] = RAD_DFLASH_TREE_OUT_COLS;
        d.fill = (uint32_t)RAD_FILL_INDEX; d.idx_max = g.V;
        break;
    }
    *out = d;
    return RAD_OK;
}

/* ---- the call ---------------------------------------------------------------------------------- */
int rad_dflash_tree_select_args(const RadArgs* a, RadDflashTreeSelectCall* call) {
    if (!a || !call || !a->t || a->n_t != RAD_DFLASH_TREE_OPERANDS) return RAD_E_INVAL;
    Geom g{};
    int rc = geometry(a->p, a->n_p, &g, false);
    if (rc != RAD_OK) return rc;

    const RadTensor& cand = a->t[RAD_DFLASH_TREE_CAND];
    const RadTensor& un   = a->t[RAD_DFLASH_TREE_UNARY];
    const RadTensor& hp   = a->t[RAD_DFLASH_TREE_HP];
    const RadTensor& anc  = a->t[RAD_DFLASH_TREE_ANCHOR];
    const RadTensor& pr   = a->t[RAD_DFLASH_TREE_PRED];
    const RadTensor& sc   = a->t[RAD_DFLASH_TREE_SUCC];
    const RadTensor& out  = a->t[RAD_DFLASH_TREE_OUT];
    if (!cand.data || !un.data || !anc.data || !pr.data || !sc.data || !out.data)
        return RAD_E_INVAL;

    /* THE OUTPUT'S ROWS ARE THE COUNT, exactly as `dflash_select` reads its own `M`: RadArgs carries
     * the geometry with its ranges collapsed, so `M` here is the band's ceiling (max_seqs) and not
     * the sequences this pass issued. Reading the parameter would demand max_seqs*steps candidate
     * rows on a one-sequence step and refuse every launch. */
    if (out.rank != 2 || out.shape[1] < RAD_DFLASH_TREE_OUT_COLS) return RAD_E_SHAPE;
    const int64_t M = out.shape[0];
    /* `M` is required by the schema, so a band is always declared; a call that issues MORE rows
     * than its band is a disagreement, and one that issues fewer is a normal partial step. */
    if (M <= 0 || M > g.M) return RAD_E_SHAPE;
    const int64_t rows = M * g.steps;
    if (rows > INT32_MAX) return RAD_E_SHAPE;

    RadDflashTreeSelectCall v{};
    v.M = M;
    v.rows = rows;
    v.steps = g.steps;
    v.K = g.K;
    v.R = g.R;
    v.V = g.V;
    v.anchor_stride = g.astr;
    v.budget = g.budget;
    v.width = g.width;
    v.beta = g.beta;

    v.cand_ld  = plane(cand, RAD_I32, rows, g.K);
    v.unary_ld = plane(un, RAD_F32, rows, g.K);
    /* The output is a [M, 33] plane too; its own pitch is what the kernel writes through, so it is
     * checked the same way rather than assumed contiguous. */
    if (out.stride[1] != 1 || out.stride[0] < RAD_DFLASH_TREE_OUT_COLS) return RAD_E_STRIDE;
    v.out_ld   = (int64_t)out.stride[0];
    v.pred_ld  = plane(pr, RAD_BF16, g.V, g.R);
    v.succ_ld  = plane(sc, RAD_BF16, g.V, g.R);
    /* ABSENT IS A NULL `data`, NOT A MISSING OPERAND -- see the header. Both spellings of "no hidden
     * projection" arrive here and both mean the bigram form. */
    const bool has_hp = hp.data != nullptr;
    v.hp_ld = has_hp ? plane(hp, RAD_BF16, rows, g.R) : 0;
    if (v.cand_ld < 0 || v.unary_ld < 0 || v.hp_ld < 0 || v.pred_ld < 0 || v.succ_ld < 0)
        return RAD_E_SHAPE;
    if (!vector(anc, RAD_I32, (M - 1) * g.astr + 1)) return RAD_E_SHAPE;
    /* One pitch for both candidate planes, because the kernel indexes them with one expression and
     * a call whose two planes disagree would read a row of the other. */
    if (v.cand_ld != v.unary_ld) return RAD_E_STRIDE;
    /* The launch narrows every pitch to `int`; refusing above it here beats a silent truncation. */
    if (v.cand_ld > INT32_MAX || v.unary_ld > INT32_MAX || v.hp_ld > INT32_MAX ||
        v.pred_ld > INT32_MAX || v.succ_ld > INT32_MAX || v.out_ld > INT32_MAX ||
        v.M > INT32_MAX || v.R > INT32_MAX || v.K > INT32_MAX || v.V > INT32_MAX)
        return RAD_E_SHAPE;

    /* THE PACKED OUTPUT MAY NOT SHARE BYTES WITH ANY INPUT: the kernel reads all three planes of a
     * level before it writes, but a row's write would land in a plane the NEXT level still reads. */
    if (overlap(out, cand) || overlap(out, un) || overlap(out, anc) ||
        (has_hp && overlap(out, hp)) || overlap(out, pr) || overlap(out, sc))
        return RAD_E_INVAL;

    v.cand = cand.data;
    v.unary = un.data;
    v.hp = has_hp ? hp.data : nullptr;
    v.anchor = anc.data;
    v.pred = pr.data;
    v.succ = sc.data;
    v.out = out.data;
    *call = v;
    return RAD_OK;
}

/* The row's necessary conditions on a GEOMETRY. Strides, dtypes and alignment are properties of a
 * call and stay in the entry point above, as spec §2.1 requires. */
static const RadConstraint kConstraints[] = {
    RAD_CLE("steps", RAD_DFLASH_TREE_MAX_STEPS),
    RAD_CLE("top_k", RAD_DFLASH_TREE_MAX_K),
    RAD_CLE("tree_budget", RAD_DFLASH_TREE_MAX_BUDGET),
    RAD_CLE("tree_width", RAD_DFLASH_TREE_MAX_WIDTH),
    RAD_CGE("steps", 1),
    RAD_CGE("tree_budget", 1),
    RAD_CGE("tree_width", 1),
    RAD_CIN("dtype", "bf16"),
};

RadKernelInfo rad_dflash_tree_select_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "mine_dflash_tree_select";
    r.op = kSchema.op;
    r.family = "draft";
    r.computes = kSchema.doc;
    r.shape = "steps <= 3, top_k <= 16, tree_budget <= 15, tree_width <= top_k; candidate planes "
              "are (seq*steps+level) rows; output is the root-inclusive [M, 33] packed tree";
    r.dtypes = "i32 candidates and anchors, f32 unary, bf16 codebooks and hp, i32 tree out";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.constraints = kConstraints;
    r.n_constraints = (int)(sizeof(kConstraints) / sizeof(kConstraints[0]));
    r.launch = launch;
    r.opd_shape = rad_dflash_tree_select_shape;
    /* NO `replaces`. This op is not a fusion of conventional ops: it is a DIFFERENT selector with
     * the same edge arithmetic as `dflash_select`, and a chain claim would say the two agree on
     * every call, which is false for every budget above one path. */
    return r;
}
