#include "mtp_tree_accum.h"

#include <climits>
#include <cstddef>
#include <cstdint>

namespace {
#define PI(k) {k, RAD_P_INT, RAD_REQUIRED, 0}
const RadParamSpec params[] = {PI("M"), PI("pitch"), PI("capacity"), PI("max_rounds"),
                               PI("n_vocab"), PI("round")};
#undef PI
const RadOperandSpec operands[] = {{"cand", RAD_OPD_IN, 0}, {"token_ids", RAD_OPD_IN, 1},
    {"draft_out_ids", RAD_OPD_IN, 1}, {"history", RAD_OPD_INOUT, 0}, {"packed", RAD_OPD_INOUT, 0}};
const RadOpSchema schema = {"mtp_tree_accum", params, 6, operands, 5,
    "Accumulate up to max_rounds (3 or 8) top-8 MTP rounds on device; root-inclusive bounded tree, "
    "spine first then sibling-deduplicated alternates. Round one resets live rows."};
const RadParamSpec merge_params[] = {
    {"M", RAD_P_INT, RAD_REQUIRED, 0}, {"pitch", RAD_P_INT, RAD_REQUIRED, 0},
    {"capacity", RAD_P_INT, RAD_REQUIRED, 0}, {"max_rounds", RAD_P_INT, RAD_REQUIRED, 0},
    {"n_vocab", RAD_P_INT, RAD_REQUIRED, 0}, {"round", RAD_P_INT, RAD_REQUIRED, 0},
    {"world_size", RAD_P_INT, RAD_REQUIRED, 0}};
const RadOperandSpec merge_operands[] = {
    {"gathered", RAD_OPD_IN, 0}, {"idx", RAD_OPD_OUT, 0}, {"val", RAD_OPD_OUT, 0},
    {"token_ids", RAD_OPD_IN, 1}, {"draft_out_ids", RAD_OPD_IN, 1},
    {"history", RAD_OPD_INOUT, 0}, {"packed", RAD_OPD_INOUT, 0}};
const RadOpSchema merge_schema = {"mtp_merge_tree_accum", merge_params, 7, merge_operands, 7,
    "TP2 top-8 merge with libr4d bit-key ordering, then MTP history and packed-tree accumulation; "
    "all eight merged ids and value bits remain observable."};
struct Geometry { int64_t M, cap, vocab, round, rounds; };
/* max_rounds=3: frozen legacy geometry (capacity<=16). max_rounds=8: fixed MTP8/64
 * geometry, 8 spine + 56 alternates + root = capacity<=65. */
int geometry(const RadParam* p, int n, Geometry& g, bool wide) {
    if (!p) return RAD_E_INVAL;
    g.M = wide ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
    g.cap = rad_param_geti(p,n,"capacity",0);
    g.vocab = rad_param_geti(p,n,"n_vocab",0);
    g.round = rad_param_geti(p,n,"round",0);
    g.rounds = rad_param_geti(p,n,"max_rounds",0);
    if (g.rounds != 3 && g.rounds != 8) return RAD_E_UNSUPPORTED;
    if (g.M < 1 || g.M > INT32_MAX || g.cap < 2 || g.cap > (g.rounds == 3 ? 16 : 65) ||
        g.vocab < 1 || g.vocab > INT32_MAX || g.round < 1 || g.round > g.rounds ||
        rad_param_geti(p,n,"pitch",0) != 8)
        return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
/* Validate span arithmetic before overlap or kernel pointer arithmetic. Padding is supported,
 * inner planes remain dense, and no product can overflow the signed byte address domain. */
int tensor(const RadTensor& t, int rank, const int64_t* minimum, uint64_t& bytes,
           uint32_t dtype = RAD_I32) {
    if (!t.data) return RAD_E_INVAL;
    if (reinterpret_cast<uintptr_t>(t.data) % alignof(int32_t)) return RAD_E_INVAL;
    if (t.dtype != dtype) return RAD_E_DTYPE;
    if (t.rank != static_cast<uint32_t>(rank)) return RAD_E_SHAPE;
    uint64_t span = 1;
    for (int j = rank - 1; j >= 0; --j) {
        if (t.shape[j] < minimum[j] || t.shape[j] > INT32_MAX) return RAD_E_SHAPE;
        if (t.stride[j] < 1 || (j < rank-1 && static_cast<uint64_t>(t.stride[j]) < span))
            return RAD_E_STRIDE;
        const uint64_t extent = static_cast<uint64_t>(t.shape[j]-1);
        if (extent && static_cast<uint64_t>(t.stride[j]) > (INT64_MAX/4-span)/extent)
            return RAD_E_SHAPE;
        span += extent * static_cast<uint64_t>(t.stride[j]);
    }
    bytes = span * 4;
    if (reinterpret_cast<uintptr_t>(t.data) > UINTPTR_MAX-bytes) return RAD_E_INVAL;
    return RAD_OK;
}
bool overlap(const RadTensor& a, uint64_t an, const RadTensor& b, uint64_t bn) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a.data), y = reinterpret_cast<uintptr_t>(b.data);
    return an && bn && x < y+bn && y < x+an;
}
}

int rad_mtp_tree_accum_schema_count(void) { return 2; }
const RadOpSchema* rad_mtp_tree_accum_schema_at(int i) {
    return i == 0 ? &schema : i == 1 ? &merge_schema : nullptr;
}
int rad_mtp_tree_accum_shape(const RadParam* p, int n, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= 5) return RAD_E_INVAL;
    Geometry g{};
    const int rc = geometry(p,n,g,true);
    if (rc != RAD_OK) return rc;
    RadOpdDesc d{};
    d.dtype = RAD_I32;
    d.idx_const = -1;
    d.fill = RAD_FILL_INDEX;
    d.idx_max = g.vocab;
    d.rank = 2; d.shape[0] = g.M; d.shape[1] = 8;
    if (operand == 1 || operand == 2) {
        d.rank = 1;
        if (operand == 2) d.idx_max = g.M;
        if (g.round > 1) d.flags = RAD_OPD_F_ABSENT;
    } else if (operand == 3) {
        d.rank = 3; d.shape[0] = g.rounds; d.shape[1] = g.M; d.shape[2] = 8;
    } else if (operand == 4) {
        d.shape[1] = 1+2*g.cap;
    }
    *out = d;
    return RAD_OK;
}

int rad_mtp_tree_accum_args(const RadArgs* a, RadMtpTreeAccumCall* out) {
    if (!a || !out || !a->t || a->n_t != 5) return RAD_E_INVAL;
    Geometry g{};
    int rc = geometry(a->p,a->n_p,g,false);
    if (rc != RAD_OK) return rc;
    const auto& cand = a->t[0]; const auto& tok = a->t[1]; const auto& ids = a->t[2];
    const auto& hist = a->t[3]; const auto& packed = a->t[4];
    if (packed.rank != 2 || packed.shape[0] < 1 || packed.shape[0] > g.M) return RAD_E_SHAPE;
    const int64_t M = packed.shape[0];
    const int64_t cs[] = {M,8}, hs[] = {g.rounds,M,8}, os[] = {M,1+2*g.cap}, ts[] = {1}, is[] = {M};
    uint64_t bytes[5]{};
    if ((rc=tensor(cand,2,cs,bytes[0])) != RAD_OK ||
        (rc=tensor(hist,3,hs,bytes[3])) != RAD_OK ||
        (rc=tensor(packed,2,os,bytes[4])) != RAD_OK) return rc;
    if (cand.stride[1] != 1 || hist.stride[2] != 1 || packed.stride[1] != 1)
        return RAD_E_STRIDE;
    if (hist.shape[0] != g.rounds || hist.shape[2] != 8) return RAD_E_SHAPE;
    if (g.round == 1) {
        if ((rc=tensor(tok,1,ts,bytes[1])) != RAD_OK ||
            (rc=tensor(ids,1,is,bytes[2])) != RAD_OK) return rc;
    }
    for (int i=0;i<5;++i) for (int j=i+1;j<5;++j)
        if ((i >= 3 || j >= 3) && overlap(a->t[i],bytes[i],a->t[j],bytes[j]))
            return RAD_E_INVAL;
    *out = {static_cast<const int32_t*>(cand.data), static_cast<const int32_t*>(tok.data),
        static_cast<const int32_t*>(ids.data), static_cast<int32_t*>(hist.data),
        static_cast<int32_t*>(packed.data), M, g.round == 1 ? tok.shape[0] : 0,
        cand.stride[0], tok.stride[0], ids.stride[0], hist.stride[0], hist.stride[1],
        packed.stride[0], static_cast<int>(g.cap), static_cast<int>(g.round), static_cast<int>(g.vocab),
        static_cast<int>(g.rounds)};
    return RAD_OK;
}

RadKernelInfo rad_mtp_tree_accum_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "mine_mtp_tree_accum"; r.op = schema.op; r.family = "draft";
    r.computes = schema.doc;
    r.shape = "pitch=8, max_rounds=3 (capacity=2..16) or 8 (capacity=2..65), round=1..max_rounds";
    r.dtypes = "i32"; r.domain = RAD_DOMAIN_DEVICE; r.priority = 100;
    r.launch = launch; r.opd_shape = rad_mtp_tree_accum_shape;
    return r;
}

int rad_mtp_merge_tree_accum_shape(const RadParam* p, int n, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= 7) return RAD_E_INVAL;
    Geometry g{};
    const int rc = geometry(p,n,g,true);
    if (rc != RAD_OK) return rc;
    if (rad_param_geti(p,n,"world_size",0) != 2) return RAD_E_UNSUPPORTED;
    if (operand >= 3)
        return rad_mtp_tree_accum_shape(p,n,operand-2,out);
    RadOpdDesc d{};
    d.dtype = operand == 1 ? RAD_I32 : RAD_F32;
    d.idx_const = -1;
    d.rank = 2; d.shape[0] = g.M; d.shape[1] = 8;
    if (operand == 0) {
        d.rank = 3; d.shape[1] = 16; d.shape[2] = 2;
    }
    *out = d;
    return RAD_OK;
}

int rad_mtp_merge_tree_accum_args(const RadArgs* a, RadMtpMergeTreeAccumCall* out) {
    if (!a || !out || !a->p || !a->t || a->n_t != 7) return RAD_E_INVAL;
    if (rad_param_geti(a->p,a->n_p,"world_size",0) != 2) return RAD_E_UNSUPPORTED;
    RadTensor tree_t[] = {a->t[1],a->t[3],a->t[4],a->t[5],a->t[6]};
    RadArgs tree_a = *a; tree_a.t = tree_t; tree_a.n_t = 5;
    RadMtpMergeTreeAccumCall c{};
    int rc = rad_mtp_tree_accum_args(&tree_a,&c.tree);
    if (rc != RAD_OK) return rc;
    const int64_t M = c.tree.M;
    const int64_t gs[] = {M,16,2}, cs[] = {M,8}, ts[] = {1}, is[] = {M};
    const int64_t hs[] = {c.tree.max_rounds,M,8}, os[] = {M,1+2*c.tree.capacity};
    const int ranks[] = {3,2,2,1,1,3,2};
    const int64_t* shapes[] = {gs,cs,cs,ts,is,hs,os};
    uint64_t bytes[7]{};
    for (int i=0;i<7;++i) {
        if (c.tree.round > 1 && (i == 3 || i == 4)) continue;
        rc = tensor(a->t[i],ranks[i],shapes[i],bytes[i],
                    i == 0 || i == 2 ? RAD_F32 : RAD_I32);
        if (rc != RAD_OK) return rc;
    }
    const auto& gathered = a->t[0]; const auto& idx = a->t[1]; const auto& val = a->t[2];
    if (gathered.shape[0] != M || gathered.shape[1] != 16 || gathered.shape[2] != 2 ||
        idx.shape[0] != M || idx.shape[1] != 8 || val.shape[0] != M || val.shape[1] != 8 ||
        a->t[6].shape[1] != 1+2*c.tree.capacity) return RAD_E_SHAPE;
    if (gathered.stride[2] != 1 || gathered.stride[1] != 2 || val.stride[1] != 1)
        return RAD_E_STRIDE;
    for (int i=0;i<7;++i) for (int j=i+1;j<7;++j) {
        const bool writes_i = i == 1 || i == 2 || i >= 5;
        const bool writes_j = j == 1 || j == 2 || j >= 5;
        if ((writes_i || writes_j) && overlap(a->t[i],bytes[i],a->t[j],bytes[j]))
            return RAD_E_INVAL;
    }
    c.gathered = static_cast<const uint32_t*>(gathered.data);
    c.idx = static_cast<int32_t*>(idx.data);
    c.val = static_cast<uint32_t*>(val.data);
    c.gathered_ld = gathered.stride[0]; c.val_ld = val.stride[0];
    *out = c;
    return RAD_OK;
}

RadKernelInfo rad_mtp_merge_tree_accum_row(RadLaunchFn launch) {
    RadKernelInfo r = rad_mtp_tree_accum_row(launch);
    r.name = "mine_mtp_merge_tree_accum"; r.op = merge_schema.op;
    r.computes = merge_schema.doc;
    r.shape = "world_size=2, pitch=8, max_rounds=3 (capacity=2..16) or 8 (capacity=2..65), "
              "round=1..max_rounds";
    r.dtypes = "f32 pairs, i32 ids/history/tree, f32 values";
    r.opd_shape = rad_mtp_merge_tree_accum_shape;
    return r;
}
