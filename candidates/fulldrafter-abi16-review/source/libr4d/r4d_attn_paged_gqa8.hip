// R4D attention dispatch at EIGHT queries per KV head, for head_dim 256 and head_dim 128.
//
// It is the same device code as the 27B target's h256/gqa6 dispatch -- the kernels are templates
// on head_dim and gqa, so this file is a set of instantiations and not a set of kernels -- and it
// exists because two models in scope are gqa 8 and the constraint tables gate on the exact value:
//
//   Qwen3.6-35B-A3B   16 query heads over 2 KV heads, head_dim 256   (the routed sibling of the
//                     27B, which is 24 over 4 at the same head_dim)
//   MiniCPM5-2B       16 query heads over 2 KV heads, head_dim 128
//
// WHAT CHANGES WITH GQA, AND IT IS ONLY THE TWO BOUNDS. A decode workgroup carries `q_len * gqa`
// rows and holds 64 of them, so the verify width falls from 10 at gqa 6 to 8 at gqa 8 -- and that
// bound is the constraint value the selector bands on, so it is stated in the row and nowhere
// else. A prefill block carries `NWARPS * 16 / gqa` query positions, 48 here against 64 there.
//
// WHAT CHANGES WITH HEAD_DIM IS ONE OPTION BIT. `O_KWIDE` fetches 16 K elements a thread and its
// thread-to-(key, channel) map is written for 16 channels of 16 -- which is head_dim 256 exactly.
// At 128 it would stage past the tile, so the h128 prefill runs the same option set without it
// (3430923 against 3955211) and pays one more pass over a tile that is half the size. Everything
// else -- the lazy reference-max softmax, f16 operands, the folded query scale, contiguous-k
// fragments, the SGPR block table, the V prefetch, the LDS-scoped barrier -- is unchanged, and
// each bit is defined and justified in r4d_attn_prefill_h256_gqa6.hip.
//
// NO `r4d_attn_dims` HERE. That entry point and the `check_dims_once` assertion built on it are
// the h256/gqa6 unit's statement about ITSELF, and a second definition would not link. A geometry
// gate is what the row table's constraints are for; this file states its own in the launchers.
#include "r4d.h"
#include "r4d_attn_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
// The decode kernel shares its query conversion and fragment helpers with the prefill one, so the
// prefill header comes first here for the same reason it does in the other two dispatches.
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK8() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* THE QUERIES PER KV HEAD, AND THE SYMBOL NAMES BUILT FROM IT.
 *
 * This unit is a set of INSTANTIATIONS of templates that already take `gqa`, so a second ratio is
 * a second set of macro values and not a second kernel -- and a second COPY of this file would be
 * the drift the header warns about, in a dispatch two shipped models run. So the geometry comes
 * from macros with this unit's own values as the defaults, the entry points paste the ratio into
 * their names, and another ratio is a four-line file that defines the macros and includes this
 * one (r4d_attn_paged_h256_gqa12.hip).
 *
 * WHAT CHANGES WITH THE RATIO IS THE TWO BOUNDS AND NOTHING ELSE, exactly as the header says: the
 * decode verify width is 64/gqa (8 here, 5 at gqa 12) and a prefill block carries 384/gqa query
 * positions (48 here, 32 at gqa 12). Both must divide, and both are stated in the ROW so the
 * selector bands on them. */
#ifndef R4D_ATTN_GQA
#define R4D_ATTN_GQA  8
#define R4D_ATTN_H128 1      /* this ratio also serves head_dim 128 -- MiniCPM5-2B is 16 over 2 */
#endif
#define GQ_JOIN_(a, b, c) a##b##c
#define GQ_JOIN(a, b, c)  GQ_JOIN_(a, b, c)
#define GQ_SYM(pre, suf)  GQ_JOIN(pre, R4D_ATTN_GQA, suf)

#define G8_GQA         R4D_ATTN_GQA
/* The KV page, in tokens. Overridable by the including translation unit because the BODY IS
 * GENERIC IN IT -- `bt[kk / BS]` picks the page, `kk % BS` the offset inside it, and
 * `NB = TILE / BS` is how many block-table entries the O_BTS prefetch stages a tile. 16 is what
 * every geometry here has wanted so far; Qwen4-Exp's QSA wants 4, because its indexer selects
 * 4-token blocks and a page of 4 makes the selection a block table rather than a new kernel
 * (docs/QSA.md). */
#ifndef G8_BLOCK_SIZE
#define G8_BLOCK_SIZE 16
#endif
#define G8_MAX_DECODE_ROWS 64       // q_len * gqa a single decode workgroup can hold: q_len <= 8

#ifndef G8_P_NWARPS
#define G8_P_NWARPS 24
#endif
#ifndef G8_P_TILE
#define G8_P_TILE   48
#endif
#ifndef G8_P_OPT_256
#define G8_P_OPT_256 3955211
#endif
#ifndef G8_P_OPT_128
#define G8_P_OPT_128 3430923        // the same set without O_KWIDE; see the header
#endif
#define G8_D_TILE   16
#define G8_D_OPT    3429947
#define G8_D_JPB     2
#define G8_D_VECMIN 256

// ---- prefill -----------------------------------------------------------------------------------
template <int HEAD_DIM, int OPT, int KVP>
static int g8_prefill_launch(const R4DArgs* a, hipStream_t s) {
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    constexpr int BQ = (G8_P_NWARPS * 16) / G8_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<G8_P_NWARPS, G8_P_TILE, HEAD_DIM, G8_GQA,
                                                G8_BLOCK_SIZE, KVP, OPT>),
                       grid, dim3(G8_P_NWARPS * 32), 0, s, *a);
    CHK8();
    return 0;
}

// ---- decode ------------------------------------------------------------------------------------
// The h256/gqa6 split law verbatim -- as many workgroups as fit under 128, at least two tiles a
// segment. It reads kv_heads, num_seqs and max_ctx and nothing about gqa or head_dim, so there is
// nothing in it to re-derive for this geometry; copying it rather than sharing it is the cost of
// the three dispatches being three translation units, and the target count is the h256 leg's.
static int g8_decode_splits(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + G8_D_TILE - 1) / G8_D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    int s = 128 / units;
    const int cap = tiles / 2;
    if (s > cap) s = cap;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

extern "C" long GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(const R4DArgs* a) {
    const int  splits = g8_decode_splits(a);
    const long rows   = (long)r4d_q_total(*a) * a->q_heads;
    return (G8_D_OPT & 32) ? rows * splits * (long)(a->head_dim * 2 + 8)
                           : rows * splits * (long)(a->head_dim + 2) * 4L;
}

template <int HEAD_DIM, int KVP>
/* THE DECODE PATH TAKES A SINK, so the same geometry serves the launch and the description.
 * Two dispatches: the attention kernel and the split-KV combine that follows it. The kernel takes
 * `R4DArgs` BY VALUE -- 168 bytes -- which the descriptor's blob carries like any other argument,
 * because a kernel parameter is a kernel parameter whatever its size.
 *
 * `gq` is attn_paged_gate_quant's epilogue: the same two dispatches, with the combine taking the
 * output gate and the fp8 quantiser on (r4d_attn_decode_h256_gqa6.hip, R4DGateQuant). */
static int g8_decode_launch(const R4DArgs* a, const R4dSink& sk,
                            const R4DGateQuant* gq = nullptr) {
    int emit_rc = RAD_OK;
    if (a->head_dim != HEAD_DIM || a->block_size != G8_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != G8_GQA) return -2;
    const int rows = a->q_len * G8_GQA;
    if (rows > G8_MAX_DECODE_ROWS) return -4;
    const int nw = (rows + 15) / 16;
    const int splits = g8_decode_splits(a);

#define G8_LAUNCH(NW)                                                                          \
    emit_rc = R4D_EMIT(sk, (r4d_attn_decode_kernel<NW, G8_D_TILE, HEAD_DIM, G8_GQA,            \
                                                   G8_BLOCK_SIZE, KVP, G8_D_OPT>),             \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, *a, splits)
    /* One wave a workgroup stages V as its codes (O_V8): the one wave converts each element once
     * either way, and the tile is half the LDS. */
    if      (nw == 1)
        emit_rc = R4D_EMIT(sk, (r4d_attn_decode_kernel<1, G8_D_TILE, HEAD_DIM, G8_GQA,
                                                       G8_BLOCK_SIZE, KVP, G8_D_OPT | O_V8>),
                           dim3(splits, a->kv_heads, a->num_seqs), dim3(32), 0, *a, splits);
    else if (nw == 2) G8_LAUNCH(2);
    else if (nw == 3) G8_LAUNCH(3);
    else              G8_LAUNCH(4);
#undef G8_LAUNCH
    if (emit_rc != RAD_OK) return -5;
    if (!sk.desc) CHK8();

    // The same two merge forms and the same rule for choosing between them as the other two
    // dispatches: a wave a row while there are enough rows to fill the part and the segment count
    // fits a wave, the general LDS-tree form otherwise. Both produce the same bytes.
    const int jobs3 = r4d_q_total(*a) * a->q_heads;
    if (gq) {
        /* GQ 2 is the int8 codes' epilogue, 1 the E4M3 one; see R4DGateQuant. */
#define G8_GQ_VEC(PF, GQV)                                                                     \
        R4D_EMIT(sk, (r4d_attn_splitkv_combine_vec_gq_kernel<HEAD_DIM, 4, PF, G8_D_JPB, 0, GQV>), \
                 dim3((jobs3 + G8_D_JPB - 1) / G8_D_JPB), dim3(32 * G8_D_JPB), 0, *a, splits,   \
                 G8_D_TILE, *gq)
#define G8_GQ_GEN(PF, GQV)                                                                     \
        R4D_EMIT(sk, (r4d_attn_splitkv_combine_gq_kernel<HEAD_DIM, 4, PF, 0, GQV>), dim3(jobs3),  \
                 dim3(HEAD_DIM), splits * sizeof(float), *a, splits, G8_D_TILE, *gq)
        const bool i8 = gq->i8 != 0;
        if (splits <= 64 && jobs3 >= G8_D_VECMIN) {
            if (G8_D_OPT & 32) emit_rc = i8 ? G8_GQ_VEC(1, 2) : G8_GQ_VEC(1, 1);
            else               emit_rc = i8 ? G8_GQ_VEC(0, 2) : G8_GQ_VEC(0, 1);
        } else if (G8_D_OPT & 32) {
            emit_rc = i8 ? G8_GQ_GEN(1, 2) : G8_GQ_GEN(1, 1);
        } else {
            emit_rc = i8 ? G8_GQ_GEN(0, 2) : G8_GQ_GEN(0, 1);
        }
#undef G8_GQ_VEC
#undef G8_GQ_GEN
    } else if (splits <= 64 && jobs3 >= G8_D_VECMIN) {
        const int blocks = (jobs3 + G8_D_JPB - 1) / G8_D_JPB;
        if (G8_D_OPT & 32)
            emit_rc = R4D_EMIT(sk, (r4d_attn_splitkv_combine_vec_kernel<HEAD_DIM, 4, 1, G8_D_JPB>),
                               dim3(blocks), dim3(32 * G8_D_JPB), 0, *a, splits, G8_D_TILE);
        else
            emit_rc = R4D_EMIT(sk, (r4d_attn_splitkv_combine_vec_kernel<HEAD_DIM, 4, 0, G8_D_JPB>),
                               dim3(blocks), dim3(32 * G8_D_JPB), 0, *a, splits, G8_D_TILE);
    } else if (G8_D_OPT & 32)
        /* THE BLOCK IS HEAD_DIM THREADS AND NOT 256: the general combine is
         * __launch_bounds__(HEAD_DIM), one thread a channel, so a 128-wide head launched at 256
         * is refused by the runtime. A dispatch that serves only head_dim 256 can write 256 here;
         * this one serves both widths and cannot. */
        emit_rc = R4D_EMIT(sk, (r4d_attn_splitkv_combine_kernel<HEAD_DIM, 4, 1>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), *a, splits, G8_D_TILE);
    else
        emit_rc = R4D_EMIT(sk, (r4d_attn_splitkv_combine_kernel<HEAD_DIM, 4, 0>), dim3(jobs3),
                           dim3(HEAD_DIM), splits * sizeof(float), *a, splits, G8_D_TILE);
    if (emit_rc != RAD_OK) return -5;
    if (!sk.desc) CHK8();
    return 0;
}

// ============================================================== the plugin ABI entry points

extern "C" int GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _fp8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    /* Absent descales mean 1.0, exactly as on the gqa6 leg: the kernels spell it that way and a
     * cache written at the default scale needs no plane to allocate or keep in step. */
    return g8_prefill_launch<256, G8_P_OPT_256, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<256, G8_P_OPT_256, 1>(&r, (hipStream_t)s);
}
#if R4D_ATTN_H128
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h128_gqa, _fp8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<128, G8_P_OPT_128, 0>(&r, (hipStream_t)s);
}
extern "C" int GQ_SYM(r4d_rad_attn_prefill_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = fill_paged(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_prefill_launch<128, G8_P_OPT_128, 1>(&r, (hipStream_t)s);
}
#endif

/* The split-KV partial buffer, sized at declare from the band's upper bound. IT IS CALLED WITH
 * PARAMETERS AND NO TENSORS -- that is what declare is -- so the parameters are the primary source
 * and the tensors the fallback, which is the trap the h256/gqa6 dispatch documents at length.
 *
 * The size is the largest over EVERY sequence count up to max_seqs, not the size at max_seqs:
 * S * floor(128 / (kv_heads * S)) is not monotone in S, which the h256/gqa6 dispatch also
 * spells out. */
static int64_t g8_decode_scratch(const RadArgs* a, int head_dim) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);

    R4DArgs r;
    const long long n_head   = r4d_geti_or(a, "n_head", 0);
    const long long max_seqs = r4d_geti_or(a, "max_seqs", 0);
    if (n_head > 0 && max_seqs > 0) {
        std::memset(&r, 0, sizeof(r));
        r.head_dim   = head_dim;
        r.q_heads    = (int)n_head;
        const long long gqa = r4d_geti_or(a, "gqa", 1);
        r.kv_heads   = (int)(gqa > 0 ? n_head / gqa : n_head);
        r.num_seqs   = (int)max_seqs;
        r.q_len      = (int)r4d_geti_or(a, "q_len", 1);
        r.max_ctx    = (int)r4d_geti_or(a, "max_ctx", 0);
        r.block_size = (int)r4d_geti_or(a, "block_size", 16);
        r.splits     = sp;
        if (r.max_ctx <= 0) return R4D_NO(RAD_E_INVAL);
        int64_t worst = 0;
        for (long long n = 1; n <= max_seqs; ++n) {
            r.num_seqs = (int)n;
            const int64_t need = GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
            if (need > worst) worst = need;
        }
        return worst;
    }
    const int rc = fill_paged(a, &r, head_dim);
    if (rc != RAD_OK) return rc;
    r.splits = sp;
    return GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(&r);
}

extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h256_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 256);
}
#if R4D_ATTN_H128
extern "C" int64_t GQ_SYM(r4d_rad_attn_decode_h128_gqa, _scratch)(const RadArgs* a) {
    return g8_decode_scratch(a, 128);
}
#endif

static int g8_decode_common(const RadArgs* a, R4DArgs* r, int head_dim) {
    const int sp = split_of(a);
    if (sp < 0) return R4D_NO(RAD_E_INVAL);
    const int rc = fill_paged(a, r, head_dim);
    if (rc != RAD_OK) return rc;
    r->splits = sp;
    const int64_t need = GQ_SYM(r4d_attn_decode_gqa, _scratch_bytes)(r);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    return RAD_OK;
}

static int GQ_SYM(rad_attn_decode_h256_gqa, _fp8_body)(const RadArgs* a, const R4dSink& sk) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<256, 0>(&r, sk);
}
/* R4D_ENTRY CANNOT BE USED HERE. It pastes `_describe` onto its first argument, and this
 * name is itself a macro -- the paste would land on the GQ_SYM invocation text rather than on
 * the symbol it expands to. Both entries are spelled instead, through the same join. */
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _fp8)(const RadArgs* a, RadStream s) {
    R4dSink sk; sk.stream = s;
    return GQ_SYM(rad_attn_decode_h256_gqa, _fp8_body)(a, sk);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _fp8_describe)(
        const RadArgs* a, RadStream s, RadLaunchDesc* d, int cap, int* n) {
    if (!n) return RAD_E_INVAL;
    *n = 0;
    R4dSink sk; sk.stream = s; sk.desc = d; sk.cap = cap; sk.n = n;
    return GQ_SYM(rad_attn_decode_h256_gqa, _fp8_body)(a, sk);
}
/* ---- attn_paged_gate_quant: attn_paged's eight operands, then the gate, the codes and the
 * scales. The gate is gate_quant_fp8's: a [T, n_head, head_dim] column view of the [q|gate]
 * projection, whose (token, head) rows are one pitch apart only because the view narrowed the last
 * axis and nothing above it -- the same test gate_quant_fp8's shim applies (tq_pitch), repeated
 * here because this unit cannot link that one. */
enum { AP_GQ_G = AP_OUT + 1, AP_GQ_Q, AP_GQ_S };

static long long gq_row_pitch(const RadTensor* t, long long width) {
    if (!t || t->rank < 1) return -1;
    if (t->shape[t->rank - 1] != width) return r4d_contig(t) ? width : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    if (t->rank == 1) return width;
    for (int i = (int)t->rank - 2; i >= 1; --i)
        if (t->stride[i - 1] != t->shape[i] * t->stride[i]) return -1;
    return t->stride[t->rank - 2];
}

static int g8_gate_quant_args(const RadArgs* a, const R4DArgs& r, R4DGateQuant* gq) {
    const RadTensor* g = r4d_opt(a, AP_GQ_G);
    const RadTensor* q = r4d_opt(a, AP_GQ_Q);
    const RadTensor* s = r4d_opt(a, AP_GQ_S);
    if (!g || !q || !s) return R4D_NO(RAD_E_INVAL);
    /* The codes' dtype picks the quantiser: E4M3, or int8 for an int8 out projection. */
    if (g->dtype != RAD_BF16 || (q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) ||
        s->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    const char* act = r4d_gets(a, "act");
    if (!act || std::strcmp(act, "sigmoid")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (r4d_geti_or(a, "group", kGqGroup) != kGqGroup) return R4D_NO(RAD_E_SHAPE);
    const long long n = r.head_dim;
    if (n % kGqGroup) return R4D_NO(RAD_E_SHAPE);
    const long long nkb  = n / kGqGroup;
    const long long rows = (long long)r4d_q_total(r) * r.q_heads;
    if (r4d_numel(g) < rows * n || r4d_numel(q) < rows * n || r4d_numel(s) < rows * nkb)
        return R4D_NO(RAD_E_SHAPE);
    const long long gp = gq_row_pitch(g, n), qp = gq_row_pitch(q, n), sp = gq_row_pitch(s, nkb);
    if (gp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    /* The vector merge reads a lane's gate as one 16-byte access and stores its codes as one
     * 8-byte one. */
    if ((r4d_addr(g) & 15) || (gp % 8) || (r4d_addr(q) & 7) || (qp % 8))
        return R4D_NO(RAD_E_STRIDE);
    gq->g    = (const uint16_t*)r4d_addr(g);
    gq->q    = (uint8_t*)r4d_addr(q);
    gq->s    = (float*)r4d_addr(s);
    gq->g_ld = (unsigned)gp;
    gq->q_ld = (unsigned)qp;
    gq->s_ld = (unsigned)sp;
    gq->i8   = q->dtype == RAD_I8 ? 1 : 0;
    return RAD_OK;
}

static int GQ_SYM(rad_attn_decode_gq_h256_gqa, _fp8_body)(const RadArgs* a, const R4dSink& sk) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 0>(&r, sk, &gq);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _fp8)(const RadArgs* a, RadStream s) {
    R4dSink sk; sk.stream = s;
    return GQ_SYM(rad_attn_decode_gq_h256_gqa, _fp8_body)(a, sk);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _fp8_describe)(
        const RadArgs* a, RadStream s, RadLaunchDesc* d, int cap, int* n) {
    if (!n) return RAD_E_INVAL;
    *n = 0;
    R4dSink sk; sk.stream = s; sk.desc = d; sk.cap = cap; sk.n = n;
    return GQ_SYM(rad_attn_decode_gq_h256_gqa, _fp8_body)(a, sk);
}

/* The bf16 cache's decode, plain and with the gate and the quantiser in the merge: the fp8 pair's
 * dispatches at KVP 1, described the same way so either cache replays as a graph. */
static int GQ_SYM(rad_attn_decode_h256_gqa, _bf16_body)(const RadArgs* a, const R4dSink& sk) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<256, 1>(&r, sk);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4dSink sk; sk.stream = s;
    return GQ_SYM(rad_attn_decode_h256_gqa, _bf16_body)(a, sk);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_h256_gqa, _bf16_describe)(
        const RadArgs* a, RadStream s, RadLaunchDesc* d, int cap, int* n) {
    if (!n) return RAD_E_INVAL;
    *n = 0;
    R4dSink sk; sk.stream = s; sk.desc = d; sk.cap = cap; sk.n = n;
    return GQ_SYM(rad_attn_decode_h256_gqa, _bf16_body)(a, sk);
}
static int GQ_SYM(rad_attn_decode_gq_h256_gqa, _bf16_body)(const RadArgs* a, const R4dSink& sk) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 256);
    if (rc != RAD_OK) return rc;
    R4DGateQuant gq{};
    const int gc = g8_gate_quant_args(a, r, &gq);
    if (gc != RAD_OK) return gc;
    return g8_decode_launch<256, 1>(&r, sk, &gq);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4dSink sk; sk.stream = s;
    return GQ_SYM(rad_attn_decode_gq_h256_gqa, _bf16_body)(a, sk);
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_gq_h256_gqa, _bf16_describe)(
        const RadArgs* a, RadStream s, RadLaunchDesc* d, int cap, int* n) {
    if (!n) return RAD_E_INVAL;
    *n = 0;
    R4dSink sk; sk.stream = s; sk.desc = d; sk.cap = cap; sk.n = n;
    return GQ_SYM(rad_attn_decode_gq_h256_gqa, _bf16_body)(a, sk);
}
#if R4D_ATTN_H128
extern "C" int GQ_SYM(r4d_rad_attn_decode_h128_gqa, _fp8)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 0>(&r, r4d_stream_sink(s));
}
extern "C" int GQ_SYM(r4d_rad_attn_decode_h128_gqa, _bf16)(const RadArgs* a, RadStream s) {
    R4DArgs r;
    const int rc = g8_decode_common(a, &r, 128);
    if (rc != RAD_OK) return rc;
    return g8_decode_launch<128, 1>(&r, r4d_stream_sink(s));
}
#endif
