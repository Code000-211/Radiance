/* r4d_shapes.cpp -- the extent and content domain of each operand of an op at a geometry.
 *
 * This is libr4d's side of the `RadShapeFn` hook: the component that reads an operand is the one
 * that knows its extent. The hand-written table in `tools/` cannot reach ops that exist only here
 * (every fp8 fusion, both all-reduce folds, both DFlash2 ops, `attn_dense`, the sampler stages), so
 * without this hook a tool skips them.
 *
 * The authority for every extent is the shim, not the prose or the row table. Each `rad_*` entry
 * point unpacks `RadArgs.t[i]` into extents and pitches; if a shim passes `x->shape[1]` as `K`,
 * that is what the operand is. Each group names its shim, and a change to a shim's contract must
 * be mirrored here.
 *
 * Content matters as much as size. A token id gathers a row of a codebook, a KV slot indexes
 * n_blocks * block_size positions, a block-table entry indexes n_blocks, a uniform is in [0, 1).
 * Drawn from the wrong range, such an operand tests the kernel's out-of-range handling instead of
 * its arithmetic. `RAD_FILL_INDEX` with an explicit `idx_max` states each range, and
 * `RAD_OPD_F_IDX_UNIQUE` marks scatters whose duplicate indices would make the store a race.
 *
 * ============================================================ rows with no description
 *
 * A decline is a real answer: the caller skips and counts the case, where an invented extent
 * yields a meaningless pass. These rows carry a null `opd_shape`:
 *
 *   gdn_conv_prep, gdn_conv_update, gdn_kkt_solve, gdn_chunk_scan, gdn_recurrent_update
 *       The head counts are not in the geometry. `head_k` and `head_v` are head widths (128); the
 *       kernels read the number of key and value heads off the operands, so
 *       `conv_dim = 2*Hk*hk + Hv*hv`, the leading extent of nearly every operand, is not derivable
 *       from `p`. tools/opshapes.h reaches them only through the container metadata rad-kbench
 *       passes in `ShapeCtx`, which this hook does not receive. (`gdn_gated_rmsnorm` and
 *       `gdn_gated_norm_had_quant_i8` are described: their rows are (token, head) pairs and their
 *       widths are parameters.)
 *
 *   attn_dense
 *       No head count either: the parameters are M, head_dim, gqa, causal and dtype, the vision
 *       kernel reads the head count off `q->shape[1]`, and gqa is constrained to 1. ref describes
 *       it with a single head (`gqa` query heads against one KV head), and correctness checks use
 *       ref's description anyway.
 *
 *   sample_merge_topk
 *       `gathered` packs (u32 global id, f32 logit) pairs into one word-typed buffer, so no fill
 *       domain puts a sensible float in the value lane, and `world_size` is not a parameter. ref
 *       declines it too. Details at the sampler section below.
 *
 *   gemm_nt_q at every grid but fp8
 *       The weight is a plugin-private fragment stream (R4D_DT_W4_FRAG and friends) whose byte
 *       count comes from `RadLayout::bytes`; RadOpdDesc has only a dtype and a shape, so a
 *       description would be a second source for a number r4d_layout.cpp owns. R4D_DT_W4_SZ, a
 *       (f16 scale, f16 -(1024+zero)) dword, has no public dtype either. The fp8 family is
 *       described: its weight is the checkpoint's E4M3 [N][K] against a bf16
 *       [ceil(N/128)][ceil(K/128)] plane, and fragment order is a permutation of the same count
 *       (r4d_fp8_frag.h: `frag_bytes(N,K) == N*K`).
 *
 *   logits_gemm at fp8
 *       The declared shape and the stored plane differ. The weight is declared [n_vocab][n_embd]
 *       E4M3 but the kernel strides it by r4d_fp8::lm_row_bytes(n_embd), since each row carries
 *       its bf16 scales in its tail (r4d_fp8_layout.cpp). A buffer of the declared shape would be
 *       read 1.6% past its end. Describable once RadOpdDesc can say "declared this, stored that".
 *
 * No tuned axis is read here: this plugin's axes (tile shapes, split counts, wave counts,
 * all-reduce launch geometry) change how a kernel walks its operands, not their extents. An axis
 * that did would be read from `p`, where the core appends the chosen point.
 *
 * Pure host code (no r4d.h, no HIP), like r4d_rows.cpp and r4d_layout.cpp, so the tables compile
 * and can be read without ROCm.
 */

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "r4d_plugin.h"

#include "rad_sample.h"

namespace {

/* ==================================================================== building a description
 *
 * Each op builds its whole operand list in schema order and hands out the one asked for, so the
 * count can be checked by eye against the schema in r4d_rows.cpp (the tool also checks it against
 * `RadOpSchema::n_operands`).
 */

/* A plain operand of up to four extents. A trailing 0 ends the shape; zero is never a legal
 * extent (`pick` refuses one), so it cannot be mistaken for a real one. */
RadOpdDesc opd(uint32_t dt, int64_t d0, int64_t d1 = 0, int64_t d2 = 0, int64_t d3 = 0) {
    RadOpdDesc d{};
    d.dtype     = dt;
    d.fill      = RAD_FILL_NORMAL;
    /* -1 is "draw"; set explicitly though the caller pre-initialises it. */
    d.idx_const = -1;
    const int64_t e[4] = { d0, d1, d2, d3 };
    for (int i = 0; i < 4 && e[i] != 0; ++i) d.shape[d.rank++] = e[i];
    return d;
}

/* An index operand. `idx_max` is the exclusive upper bound of the draw; 0 means the previous
 * operand's leading extent (the ABI default, right for a gather from the operand before it).
 * `konst >= 0` fills every element with that value. Always i32: every index libr4d reads is signed
 * 32-bit, and the sampler's u32 spellings (`dt_i32` accepts either) are the same bytes. */
RadOpdDesc opd_idx(int64_t d0, int64_t d1, int64_t idx_max, int64_t konst = -1,
                   uint32_t flags = 0) {
    RadOpdDesc d = opd((uint32_t)RAD_I32, d0, d1);
    d.fill      = RAD_FILL_INDEX;
    d.idx_max   = idx_max;
    d.idx_const = konst;
    d.flags     = flags;
    return d;
}

/* An optional operand this description does not pass. One entry per schema operand is still
 * emitted, so absence needs a spelling; to the ABI it means a null RadTensor. */
RadOpdDesc opd_none() {
    RadOpdDesc d{};
    d.idx_const = -1;
    d.flags     = RAD_OPD_F_ABSENT;
    return d;
}

/* Restricts the fill to (0, 1); `RAD_FILL_SIGMOID` is the ABI's only such draw. */
RadOpdDesc in_unit(RadOpdDesc d) { d.fill = RAD_FILL_SIGMOID; return d; }

/* Hands out operand `operand` of a list after checking the whole list, so a geometry missing a
 * parameter declines every operand rather than describing a few and declining the rest. A
 * non-positive extent means a needed parameter was absent or zero: RAD_E_UNSUPPORTED, not
 * RAD_E_SHAPE, which would read as a tool bug. An index outside the list is a bug here or in the
 * schema, reported as RAD_E_SHAPE. */
int pick(RadOpdDesc* out, int operand, const RadOpdDesc* list, int n) {
    if (!out || !list || n <= 0) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (list[i].flags & RAD_OPD_F_ABSENT) continue;
        if (list[i].rank == 0 || list[i].rank > RAD_MAX_RANK) return RAD_E_UNSUPPORTED;
        for (uint32_t d = 0; d < list[i].rank; ++d)
            if (list[i].shape[d] <= 0) return RAD_E_UNSUPPORTED;
    }
    if (operand < 0 || operand >= n) return RAD_E_SHAPE;
    *out = list[operand];
    return RAD_OK;
}

#define R4D_PICK(list) return pick(out, operand, (list), (int)(sizeof(list) / sizeof((list)[0])))

/* ==================================================================== parameters
 *
 * `M` and other ranged parameters read as the band's bound, not any one step's row count (as in
 * every shim here; see the dflash_select shim in r4d_dflash_select_bf16.hip). Buffers must hold the whole band
 * the kernel was selected for.
 */
/* Not r4d_pgeti_or(): that reads RAD_P_INT only, which suits a launch (every range has collapsed),
 * but this hook can see ranges still open. rad_param_getdim (ABI, shared by both plugins) reads a
 * RAD_P_RANGE as its high bound. Unlike ref's reader it rejects a string spelling of an integer,
 * because this plugin's shims read RAD_P_INT only and describing a geometry the launch refuses
 * would turn a skip into a failure. */
int64_t pi(const RadParam* p, int n_p, const char* k, int64_t dflt = 0) {
    return (int64_t)rad_param_getdim(p, n_p, k, (long long)dflt);
}

/* The ABI's parser (static inline in rad_types.h), so the accepted spellings match libref's; a
 * local subset would refuse what libref accepts and show up as a libr4d gap. It also carries the
 * `all_reduce` synonyms (arch blocks declare `bf16`/`fp32`, core/sample/sampler.cpp `f32`). */
uint32_t dt_of(const char* s) { return rad_dtype_parse(s); }

uint32_t dt_par(const RadParam* p, int n_p, const char* key) {
    return dt_of(r4d_pgets(p, n_p, key));
}

/* The activation element type. An unknown `dtype` is usually a quantisation scheme ("fp8a8",
 * "w4a8") rather than an element width, and activations are bf16 unless stated, so that is the
 * fallback; an invalid dtype would size a zero-byte buffer. `cast` uses dt_par instead, because
 * there an unknown dtype is a pair the row cannot serve. */
uint32_t dt_act(const RadParam* p, int n_p, const char* key) {
    const uint32_t dt = dt_par(p, n_p, key);
    return dt == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;
}

/* ==================================================================== the sampler's params row
 *
 * `params` is an array of RadSampleParams viewed through a 32-bit dtype, one row per sampled
 * position (abi/rad_sample.h). The extent is exact: 96 bytes per row, as r4d_sample_stages_f32.hip's
 * `params_ok` checks. The content is a struct, and no ABI fill domain describes "a plausible
 * request".
 *
 * It is described as f32, matching ref_shapes.cpp (both implementations receive the same bytes,
 * so the two files must agree). The struct is eleven floats and nine int32s. As f32 normals the
 * float fields (temp, top_p, min_p, typical_p, the penalties, the XTC values) land in plausible
 * ranges, so some rows make their stage do work and the rest are the identity, as in a real batch.
 *
 * Limits: the integer fields are bit patterns of normals, so `hist_off`, `hist_len` and `mask_row`
 * are huge. Every stage clamps them, which keeps this safe but weak:
 *
 *     sample_penalties   ho is clamped to hmax, hl to hmax - ho, and `hl <= 0` returns
 *     sample_dry         the same clamp, then `hl <= allowed` returns
 *     sample_mask        `mask_row >= mask_rows` returns before a word of the plane is read
 *
 * So a pass on those three checks the plumbing, not the arithmetic. `flags` is a bit pattern too,
 * so about half the rows carry RAD_SP_GREEDY, which every candidate stage treats as having no
 * candidates; the comparison covers the other half. A word-typed or all-zero draw would be worse:
 * denormal or NaN floats with `1 / temp` infinite, or every stage the identity.
 *
 * Word-typed operands here are never u32: tools/opshapes.cpp's `rad_widen` and `rad_narrow` have
 * no RAD_U32 case, so a u32 operand is filled and compared as zeroes, a vacuous pass. i32 is the
 * same four bytes. */
RadOpdDesc opd_sample_params(int64_t M) {
    return opd((uint32_t)RAD_F32, M, (int64_t)(sizeof(RadSampleParams) / 4));
}

}  /* namespace */

/* ==========================================================================================
 * The small model ops (shims in r4d_model_bf16.hip and its neighbours).
 *
 * Each takes `n` as the row width and reads its row count off the operand, so descriptions size
 * for the band's `M`. A norm's gain is the one operand that is not bf16: every architecture plugin
 * declares norm gains RAD_F32 (a few thousand elements, and a norm is where a rounded gain shows),
 * and the shim reads the width off the operand.
 * ========================================================================================= */

extern "C" int r4d_shape_embed_lookup(const RadParam* p, int n_p, int operand,
                                      RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_embd");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t vo = pi(p, n_p, "vocab_offset", 0);
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* A token id indexes the table, which is the next operand, so the ABI default (the
         * previous operand's extent) is wrong and the bound is stated. Under a vocab shard ids
         * are global, so the bound is vo + nv; drawing only within the slice would never exercise
         * the zero-row path the following all_reduce depends on. */
        opd_idx(M, 0, vo + nv),
        opd(D, nv, ne),
        opd(D, M, ne),
    };
    R4D_PICK(l);
}

/* The quantised gather, as libref describes it (ref_embed_lookup_q_shape): an E4M3 table and a
 * one-element scale in the activation dtype. `ahead` is not passed: it holds ids for a reader that
 * fetches rows from a file, and the kernel ignores it. */
extern "C" int r4d_shape_embed_lookup_q(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_embd");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t vo = pi(p, n_p, "vocab_offset", 0);
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd_idx(M, 0, vo + nv),
        opd((uint32_t)RAD_F8E4M3, nv, ne),
        opd(D, 1),
        opd(D, M, ne),
        opd_none(),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gather_rows(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* Four times the output rows, so the gather is not the identity (with an M-row source,
         * idx[i] == i is plausible and passes a broken kernel). */
        opd(D, 4 * M, n),
        opd_idx(M, 0, 4 * M),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* The byte-row pair. `n` counts bytes, constrained by the registry to a multiple of 4 so the
 * vectorised copy is the only path. */
extern "C" int r4d_shape_gather_rows_u8(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_U8, 4 * M, n),
        opd_idx(M, 0, 4 * M),
        opd((uint32_t)RAD_U8, M, n),
    };
    R4D_PICK(l);
}

/* The mirror. The destination is four times taller: a scatter of M distinct indices onto M rows
 * is a permutation, so a kernel ignoring `idx` would still touch every row. With a taller
 * destination, unnamed rows must come back unchanged. */
extern "C" int r4d_shape_scatter_rows_u8(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_U8, M, n),
        opd_idx(M, 0, 4 * M, -1, RAD_OPD_F_IDX_UNIQUE),
        opd((uint32_t)RAD_U8, 4 * M, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_scatter_rows_bf16(const RadParam* p, int n_p, int operand,
                                           RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, n),
        opd_idx(M, 0, 4 * M, -1, RAD_OPD_F_IDX_UNIQUE),
        opd((uint32_t)RAD_BF16, 4 * M, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_cast(const RadParam* p, int n_p, int operand,
                              RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    /* The rows serve bf16 -> bf16 and f32 -> f32 (strided row copies), bf16 -> f32, bf16 -> f16
     * and f32 -> bf16, and the shims refuse any other pair, so another pair is declined here. */
    const uint32_t f = dt_par(p, n_p, "from"), t = dt_par(p, n_p, "to");
    const uint32_t bf = (uint32_t)RAD_BF16, f32 = (uint32_t)RAD_F32, f16 = (uint32_t)RAD_F16;
    if (!(f == bf && (t == bf || t == f32 || t == f16)) && !(f == f32 && (t == bf || t == f32)))
        return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = { opd(f, M, n), opd(t, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rmsnorm(const RadParam* p, int n_p, int operand,
                                 RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_F32, n),      /* the gain; see the family note above */
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* add_rmsnorm -- rmsnorm's operands with the addend after x. */
extern "C" int r4d_shape_add_rmsnorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd(D, M, n),
        opd((uint32_t)RAD_F32, n),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* ---- the vision tower's (r4d_vit_bf16.hip). Every optional operand is passed: the norm's bias and
 * the GEMM's residual are the same code with a branch taken, so this covers both forms. */
extern "C" int r4d_shape_layernorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, n), opd(D, n), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_grid_embed(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), side = pi(p, n_p, "side");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* Coordinates drawn over the table's side, so some rows fall past their grid's last row and
     * exercise the clamped taps as well as the interior ones. */
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, side * side, n), opd_idx(4, M, side) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gemm_nt_bias(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, K), opd(D, N, K), opd(D, N), opd(D, M, N), opd(D, M, N) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_moe_mxfp4(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const int64_t N1 = rad_param_geti(p, n_p, "N_odd", N), K1 = rad_param_geti(p, n_p, "K_odd", K);
    const int64_t ne = pi(p, n_p, "n_expert"), top = pi(p, n_p, "top_k");
    const char* order = rad_param_gets(p, n_p, "a_order", "");
    const int64_t ar = !std::strcmp(order, "sorted") ? M * top : M;
    const int64_t nk = std::max(K, K1), nn = std::max(N, N1);
    RadOpdDesc s0 = opd((uint32_t)RAD_U8, (ne + 1) / 2, N, K / 32);
    RadOpdDesc s1 = opd((uint32_t)RAD_U8, ne / 2, N1, K1 / 32);
    s0.fill = s1.fill = RAD_FILL_INDEX;
    s0.idx_const = s1.idx_const = 127;
    const RadOpdDesc l[] = { opd((uint32_t)RAD_F8E4M3, ar, nk),
                             opd((uint32_t)RAD_F32, ar, nk / 128),
                             opd((uint32_t)RAD_FP4E2M1, (ne + 1) / 2, N, K),
                             s0,
                             opd_idx(M * top, 0, M * top),
                             opd_idx(ne + 1, 0, M * top + 1),
                             opd((uint32_t)RAD_FP4E2M1, ne / 2, N1, K1),
                             s1,
                             opd((uint32_t)RAD_BF16, M * top, nn) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_kda_update(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const char* form = rad_param_gets(p, n_p, "state_form", "");
    if (!std::strcmp(form, "anchor")) return RAD_E_UNSUPPORTED;
    const int64_t M = pi(p, n_p, "M"), H = pi(p, n_p, "heads");
    const int64_t K = pi(p, n_p, "head_k"), V = pi(p, n_p, "head_v");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const char* mode = rad_param_gets(p, n_p, "mode", "");
    const int64_t history = !std::strcmp(mode, "verify") ? M : 1;
    const RadOpdDesc l[] = {
        opd(D, M, H, K), opd(D, M, H, K), opd(D, M, H, V), opd(D, M, H, K), opd(D, M, H),
        opd((uint32_t)RAD_F32, H), opd((uint32_t)RAD_F32, H * K),
        opd((uint32_t)RAD_F32, history + 1, H, V, K),
        opd_idx(1, history, history + 1, -1, RAD_OPD_F_IDX_UNIQUE),
        opd_idx(2, 0, M, -1, RAD_OPD_F_IDX_CU),
        opd_idx(1, 0, 0, 1), opd_idx(1, 0, 0, 1), opd(D, M, H, V),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_mhc_pre(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, hc * n), opd((uint32_t)RAD_F32, hc * (hc + 2), hc * n),
        opd((uint32_t)RAD_F32, 3), opd((uint32_t)RAD_F32, hc * (hc + 2)), opd(D, M, n),
        opd((uint32_t)RAD_F32, M, hc), opd((uint32_t)RAD_F32, M, hc * hc) };
    R4D_PICK(l);
}
/* The partial matrix: per chunk the hc*(hc+2) coefficient partials and the sum of squares, padded to
 * a multiple of four floats (docs/OPS.md). */
static int64_t mhc_part_stride(int64_t hc) { return (hc * (hc + 2) + 1 + 3) / 4 * 4; }
extern "C" int r4d_shape_mhc_pre_partial(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const int64_t chunks = pi(p, n_p, "chunks");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, hc * n), opd((uint32_t)RAD_F32, hc * (hc + 2), hc * n),
        opd((uint32_t)RAD_F32, M, chunks * mhc_part_stride(hc)) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_mhc_pre_finish(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const int64_t chunk = pi(p, n_p, "chunk");
    const int64_t chunks = chunk > 0 ? hc * n / chunk : 0;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd((uint32_t)RAD_F32, M, chunks * mhc_part_stride(hc)),
        opd(D, M, hc * n), opd((uint32_t)RAD_F32, 3), opd((uint32_t)RAD_F32, hc * (hc + 2)),
        opd(D, M, n), opd((uint32_t)RAD_F32, M, hc), opd((uint32_t)RAD_F32, M, hc * hc) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_mhc_post(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M, hc * n), opd((uint32_t)RAD_F32, M, hc),
        opd((uint32_t)RAD_F32, M, hc * hc), opd(D, M, hc * n) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_mhc_contract(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, hc * n), opd(D, M, n) };
    R4D_PICK(l);
}

/* ---- the gated residual (r4d_hc_bf16.hip's three shims). `inject` and `group` default on, so a
 * query naming neither describes what every trunk connection runs; 0 for either describes one of
 * the two exceptions: the final mixer, and a read whose consumer wants bf16. */
extern "C" int r4d_shape_hc_enter(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M, hc * n) };
    R4D_PICK(l);
}

/* The MTP head's entry. Both gains are F32, as hc_read's is; the two fc matrices are the
 * checkpoint's `mtp.fc_hidden` / `mtp.fc_embedding`, square, in the activation dtype. */
extern "C" int r4d_shape_mtp_enter(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, hc * n), opd(D, M, n),
                             opd((uint32_t)RAD_F32, hc * n), opd((uint32_t)RAD_F32, n),
                             opd(D, n, n), opd(D, n, n), opd(D, M, hc * n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_hc_write(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* `inj` is a gain in (0, 2), twice a sigmoid, so it is drawn in (0, 1) rather than as a
     * normal: the op's gains are small and data-dependent. */
    const RadOpdDesc l[] = { opd(D, M, n), in_unit(opd(D, M, hc)), opd(D, M, hc * n) };
    R4D_PICK(l);
}

/* hc_project's three: the bf16 stream (in place), the layer's unit f32 direction, and the int32
 * per-row switch. The switch is an index domain: 0 or 1. */
extern "C" int r4d_shape_hc_project(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, hc * n),                                  /* h, in place */
        opd((uint32_t) RAD_F32, n),                         /* v */
        opd_idx(M, 1, 1, 0),                                /* on: 0 or 1 */
    };
    R4D_PICK(l);
}

/* hc_write's operands with the all-reduce in front; `y` is also the message. */
extern "C" int r4d_shape_ar_hc_write(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return r4d_shape_hc_write(p, n_p, operand, out);
}

/* moe_gather's five and ar_hc_write's three, in that order; the gather's `y` and the reduction's are
 * one operand. */
extern "C" int r4d_shape_ar_gather_hc_write(const RadParam* p, int n_p, int operand,
                                            RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const int64_t k = pi(p, n_p, "top_k");
    if (M <= 0 || n <= 0 || k <= 0 || hc <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M * k, n),
        opd(D, M, k),
        opd_idx(M * k, 0, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE),
        opd(D, M, n),
        opd(D, M),
        opd(D, M, n),
        in_unit(opd(D, M, hc)),
        opd(D, M, hc * n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_hc_read(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const int64_t lr = pi(p, n_p, "lowrank");
    const int64_t inject = pi(p, n_p, "inject", 1);
    const int64_t group  = pi(p, n_p, "group", 0);
    const int64_t rotate = pi(p, n_p, "rotate", 0);
    const uint32_t D = dt_act(p, n_p, "dtype");
    const int64_t HN = hc * n;
    const RadOpdDesc l[] = {
        opd(D, M, HN),                                        /* h */
        opd((uint32_t)RAD_F32, HN),                           /* the gain, f32 as every norm's */
        opd(D, lr, HN),                                       /* mix_down */
        opd(D, HN, lr),                                       /* mix_up */
        inject ? opd(D, hc, HN) : opd_none(),                 /* inject_w */
        opd(D, M, n),                                         /* x */
        inject ? opd(D, M, hc) : opd_none(),                  /* inj */
        group > 0 ? opd((uint32_t)RAD_F8E4M3, M, n) : opd_none(),
        group > 0 ? opd((uint32_t)RAD_F32, M, n / (group > 0 ? group : 1)) : opd_none(),
        group > 0 && rotate ? opd((uint32_t)RAD_F8E4M3, M, n) : opd_none(),           /* rq */
        group > 0 && rotate ? opd((uint32_t)RAD_F32, M, n / group) : opd_none(),      /* rscale */
    };
    R4D_PICK(l);
}

/* add and mul share an operand list. Only the elementwise form is described, the one every caller
 * issues; the broadcast form (`b` of exactly n elements against M rows) is a second geometry. */
extern "C" int r4d_shape_binary(const RadParam* p, int n_p, int operand,
                                RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M, n), opd(D, M, n) };
    R4D_PICK(l);
}

/* mul_weight: the one-row form of the weight (a per-channel gain), or with `scalar` the
 * one-element form. */
extern "C" int r4d_shape_mul_weight(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const int64_t wn = pi(p, n_p, "scalar") ? 1 : n;
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, 1, wn), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_fill(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_unary(const RadParam* p, int n_p, int operand,
                               RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_silu_mul(const RadParam* p, int n_p, int operand,
                                  RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* The gate is the first half of a [M, 2n] row, the order rad-convert fuses gate_proj and
     * up_proj in. */
    const RadOpdDesc l[] = { opd(D, M, 2 * n), opd(D, M, n) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_router_sigmoid_topk(const RadParam* p, int n_p, int operand,
                                             RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), ne = pi(p, n_p, "n_expert"), k = pi(p, n_p, "top_k");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, ne), opd((uint32_t)RAD_F32, ne), opd_idx(M, k, ne),
                             opd(D, M, k) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_router_sigmoid_topk_scatter(const RadParam* p, int n_p, int operand,
                                                     RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), ne = pi(p, n_p, "n_expert"), k = pi(p, n_p, "top_k");
    const int64_t nl = pi(p, n_p, "n_local", ne);
    if (ne <= 0 || ne > 512 || k <= 0 || k > ne || nl <= 0 || nl > ne || M * k > 1024)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, ne),
                             opd((uint32_t)RAD_F32, ne),
                             opd_idx(M, k, ne),
                             opd(D, M, k),
                             opd((uint32_t)RAD_I32, M * k),
                             opd((uint32_t)RAD_I32, nl + 1),
                             opd((uint32_t)RAD_I32, nl),
                             opd((uint32_t)RAD_BF16, M, 1),
                             opd((uint32_t)RAD_I32, M * k) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_dsa_pool(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), ratio = pi(p, n_p, "ratio");
    const RadOpdDesc l[] = { opd((uint32_t)RAD_BF16, M * ratio, 2 * n),
                             opd((uint32_t)RAD_F32, ratio, n),
                             opd((uint32_t)RAD_F8E4M3, M, n),
                             opd((uint32_t)RAD_F32, M, 1),
                             opd_none(),
                             opd_none() };
    R4D_PICK(l);
}
extern "C" int r4d_shape_dsa_query(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    const RadOpdDesc l[] = { opd((uint32_t)RAD_BF16, M, n), opd((uint32_t)RAD_F8E4M3, M, n),
                             opd((uint32_t)RAD_F32, M, 1) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_mla_project(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), h = pi(p, n_p, "heads"), d = pi(p, n_p, "head_dim"),
                  l = pi(p, n_p, "latent_dim");
    const char* mode = r4d_pgets(p, n_p, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"),
               expand = mode && !std::strcmp(mode, "expand");
    if (!absorb && !expand) return RAD_E_UNSUPPORTED;
    const RadOpdDesc list[] = { opd((uint32_t)RAD_BF16, M, h, absorb ? d : l),
                                opd((uint32_t)RAD_BF16, h * 2 * d, l),
                                opd((uint32_t)RAD_BF16, M, h, absorb ? l : d) };
    R4D_PICK(list);
}
extern "C" int r4d_shape_dsa_score(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), H = pi(p, n_p, "heads");
    const int64_t B = pi(p, n_p, "blocks"), S = pi(p, n_p, "seqs"), pages = S * B;
    const RadOpdDesc l[] = { opd((uint32_t)RAD_F8E4M3, M, H, n),
                             opd((uint32_t)RAD_F32, M, H),
                             opd((uint32_t)RAD_F32, M, H),
                             opd((uint32_t)RAD_F8E4M3, pages, n),
                             opd((uint32_t)RAD_F32, pages),
                             opd_idx(S, B, pages),
                             opd_idx(S + 1, 0, M, -1, RAD_OPD_F_IDX_CU),
                             opd_idx(S, 0, 0, B),
                             opd((uint32_t)RAD_F32, M, B) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_mla_attn(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), D = pi(p, n_p, "latent_dim"), H = pi(p, n_p, "heads");
    const int64_t B = pi(p, n_p, "blocks"), bs = pi(p, n_p, "block_size"),
                  pages = std::max((int64_t)2, B * M);
    const uint32_t dt = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(dt, M, H, D), opd(dt, pages, 1, bs, D), opd_idx(M, B, pages),
                             opd_idx(M, 0, 0, B * bs), opd(dt, M, H, D) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rope(const RadParam* p, int n_p, int operand,
                              RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t nq  = pi(p, n_p, "n_head");
    const int64_t nkv = pi(p, n_p, "n_head_kv");
    /* The fused projection row: n_head queries, then n_head_kv each of k and v (v untouched), so
     * the width includes the V heads. `n_head_kv` 0 is the legal q-only form the gated attention
     * block issues, so only the width must be positive. */
    const int64_t row = (nq + 2 * nkv) * hd;
    /* A multi-component mode with sections takes one position per component, component-major:
     * three for the interleaved form, one per section otherwise. */
    const char* mode = rad_param_gets(p, n_p, "mode", "neox");
    const char* sec = rad_param_gets(p, n_p, "sections", nullptr);
    int64_t nc = 0;
    for (const char* q = sec; q && *q;) {
        char* e = nullptr;
        (void)std::strtoll(q, &e, 10);
        if (e == q) break;
        ++nc;
        q = e;
    }
    const bool multi = nc > 0 && (!std::strcmp(mode, "mrope") || !std::strcmp(mode, "imrope") ||
                                  !std::strcmp(mode, "axial"));
    if (multi && !std::strcmp(mode, "imrope")) nc = 3;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, row),
        /* Not a lookup: the kernel computes sincos from the value, so any non-negative integer is
         * legal. M keeps the draw in the range a step presents. */
        multi ? opd_idx(nc, M, M) : opd_idx(M, 0, M),
    };
    R4D_PICK(l);
}

/* kv_store, both rows: they differ only in the cache dtype, the `kv_dtype` parameter.
 *
 * The paged layout (libref/ref_ops.h, checked by the shim) is
 * kv_cache[n_blocks, kv_heads, block_size, 2 * head_dim], K at [.., d] and V at [.., head_dim + d].
 * The block count belongs to the deployment's cache, not to any parameter, so any count large
 * enough for the slots will do. */
extern "C" int r4d_shape_kv_store(const RadParam* p, int n_p, int operand,
                                  RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t kvh = pi(p, n_p, "n_head_kv");
    const int64_t bs  = pi(p, n_p, "block_size");
    const uint32_t KV = dt_act(p, n_p, "kv_dtype");
    if (M <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    /* Two blocks of slack, so the slot mapping is not a dense prefix and the block index is
     * actually tested. */
    const int64_t nb = (M + bs - 1) / bs + 2;
    const RadOpdDesc l[] = {
        /* [tokens, kv_heads, head_dim], as ref spells it. The shim folds trailing extents into
         * one row pitch, so a flattened [M, kv_heads * head_dim] is the same bytes and pitch. */
        opd((uint32_t)RAD_BF16, M, kvh, hd),
        opd((uint32_t)RAD_BF16, M, kvh, hd),
        /* A scatter destination, so no slot may repeat (two tokens on one slot race). It indexes
         * positions, n_blocks * block_size, which is no operand's leading extent. */
        opd_idx(M, 0, nb * bs, -1, RAD_OPD_F_IDX_UNIQUE),
        opd(KV, nb, kvh, bs, 2 * hd),
    };
    R4D_PICK(l);
}

/* logits_gemm at bf16. The fp8 row is not described; see the file header. */
extern "C" int r4d_shape_logits_gemm(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t ne = pi(p, n_p, "n_embd");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, ne),
        opd((uint32_t)RAD_BF16, nv, ne),
        /* F32 logits, checked by the shim: the bf16 GEMM would write bf16 pairs into the f32
         * plane, giving plausible magnitudes and wrong logits. */
        opd((uint32_t)RAD_F32, M, nv),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * The decomposed sampler chain (shims in r4d_sample_stages_f32.hip). `sample_merge_topk` is declined (below).
 * See opd_sample_params on the limits: `sample_penalties`, `sample_dry` and `sample_mask` take
 * their early-out branch, so a pass on those three checks the plumbing only.
 * ========================================================================================= */

/* logits + params, and one auxiliary plane that differs per stage. */
extern "C" int r4d_shape_sample_penalties(const RadParam* p, int n_p, int operand,
                                          RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    /* `history` is [M, max_hist], and the shim takes max_hist from the buffer (numel / M); no
     * parameter names it, since a deployment sizes it by its longest live sequence. 64 is a
     * plausible window; the kernel clamps `hist_off` and `hist_len` to it. */
    const int64_t hist = 64;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        /* Token ids, drawn from the vocabulary. */
        opd_idx(M, hist, nv),
    };
    R4D_PICK(l);
}

/* sample_dry: the penalties' three operands, and the optional breaker plane, not passed (absent
 * means no token is exempt; the params row takes the early out before either plane is read). As
 * ref describes it. */
extern "C" int r4d_shape_sample_dry(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t hist = 64;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        opd_idx(M, hist, nv),
        opd_none(),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_temp(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const RadOpdDesc l[] = { opd((uint32_t)RAD_F32, M, nv), opd_sample_params(M) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_mask(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    if (nv <= 0) return RAD_E_UNSUPPORTED;
    /* The plane is [rows, ceil(n_vocab/32)], shorter than M in a real batch (unconstrained
     * sequences have `mask_row` -1). One row per position is the fully constrained case. */
    const int64_t words = (nv + 31) / 32;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        /* A set bit means allowed: word w bit b covers token 32w + b; a cleared bit sets
         * -infinity. i32, not u32 (see opd_sample_params). Unreached with this params row, whose
         * `mask_row` is out of range. */
        opd((uint32_t)RAD_I32, M, words),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_topk(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    /* The candidate width is the buffer's, the deployment bound every later stage reads. No
     * parameter carries it on this op (`n_cand` is the narrowing stages' key), so 64 is used,
     * well inside the 4096 the LDS bitonic sort covers. */
    const int64_t n_cand = 64;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        opd((uint32_t)RAD_I32, M, n_cand),
        opd((uint32_t)RAD_F32, M, n_cand),   /* cand_val: logits, not probabilities */
        /* `pairs`, the vocab-parallel output: (u32 global id, f32 logit) with a hole marker for an
         * unfilled slot. As an output it needs no fill, and comparing it checks the packing; with
         * `vocab_off` zero on one rank the ids are local. As ref describes it. */
        opd((uint32_t)RAD_I32, M, 2 * n_cand),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_argmax(const RadParam* p, int n_p, int operand,
                                       RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        opd((uint32_t)RAD_I32, M),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_logprobs(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t k  = pi(p, n_p, "k");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        opd((uint32_t)RAD_F32, M, 2 + 2 * k),
    };
    R4D_PICK(l);
}

/* top_p, min_p, typical and xtc: the same three operands, narrowed in place. */
extern "C" int r4d_shape_sample_narrow(const RadParam* p, int n_p, int operand,
                                       RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nc = pi(p, n_p, "n_cand");
    const RadOpdDesc l[] = {
        /* Every drawn id is >= 0, so every candidate arrives live (an empty set would be
         * vacuous). The range is the candidate width: these kernels only reorder and drop ids,
         * and `n_vocab` is not a parameter here. Stated explicitly because operand 0 has no
         * previous operand for the ABI default. */
        opd_idx(M, nc, nc),
        opd((uint32_t)RAD_F32, M, nc),
        opd_sample_params(M),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_pick(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nc = pi(p, n_p, "n_cand");
    const RadOpdDesc l[] = {
        opd_idx(M, nc, nc),                  /* see the narrowing stages on the range */
        opd((uint32_t)RAD_F32, M, nc),
        opd_sample_params(M),                /* the draw's seed and position live here */
        opd((uint32_t)RAD_I32, M),
    };
    R4D_PICK(l);
}

/* `sample_merge_topk` is declined, as in ref. `gathered` is world_size * n_cand (u32 global token
 * id, f32 logit) pairs in one word-typed buffer (i32 or u32 in both plugins: one buffer, one
 * dtype). The logit lane is read by bit pattern, and an integer draw makes every logit a denormal;
 * a device that flushes denormals then orders candidates differently from a host that does not.
 * r4d_selftest's two-rank all-gather row checks exactly that. `world_size` is not a parameter
 * either, so the gathered width is unknown.
 *
 * `row_topk_merge` has the same shape and is described, because its plane is f32: the value is a
 * real float and the id is the bit pattern. */

/* The fused sampler (r4d_sample_chain_f32.hip): a separate op, not a composition of the stages above. Its
 * three outputs are what a rejection-sampling verify needs from one pass. */
extern "C" int r4d_shape_sample_chain(const RadParam* p, int n_p, int operand,
                                      RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        /* The drafted token per position, drawn from the vocabulary. Passed because without it
         * the kernel skips `p_query`, the accept probability. */
        opd_idx(M, 0, nv),
        /* Two uniforms per position (draw and exclusive draw), in (0, 1): the inverse-CDF walk
         * falls through to its last live candidate above the mass, so normals would only test
         * that fallback. */
        in_unit(opd((uint32_t)RAD_F32, 2 * M)),
        /* [M] x R4DSpecDraw = {u32 tok, u32 tok_excl, f32 p_query}: three dwords per position.
         * The shim checks only numel >= M (structs, not words), so [M] would let the kernel write
         * past the buffer. i32, not u32 (see opd_sample_params). */
        opd((uint32_t)RAD_I32, M, 3),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Attention (shims in the r4d_attn_*.hip units).
 * ========================================================================================= */

/* attn_paged, all six rows. `q_len` is the ranged parameter (one row at decode, thousands in a
 * prefill chunk), and `kv_dtype` separates the fp8kv rows from the bf16kv ones.
 *
 * One sequence attending causally over its own prefix, as a prefill chunk does; the context
 * defaults to the query length (see `ctx` below) and the block count follows from it. */
extern "C" int r4d_shape_attn_paged(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t q_len = pi(p, n_p, "q_len");
    const int64_t hd    = pi(p, n_p, "head_dim");
    const int64_t gqa   = pi(p, n_p, "gqa");
    const int64_t bs    = pi(p, n_p, "block_size");
    /* n_head is optional on this op (the split-KV scratch hook needs it at declare). Without it
     * there is no head count, so this declines; tools/opshapes.h falls back to one kv head. */
    const int64_t nh    = pi(p, n_p, "n_head");
    if (q_len <= 0 || hd <= 0 || gqa <= 0 || bs <= 0 || nh <= 0 || nh % gqa)
        return RAD_E_UNSUPPORTED;
    const int64_t kvh = nh / gqa;
    /* The context depth: the query length unless `ctx` overrides it. No op declares `ctx` (the
     * kernel reads the context off `seqused` and the block table); it lets a caller sweep depth.
     * It must be at least q_len, or the causal mask reads a missing prefix. */
    int64_t ctx = pi(p, n_p, "ctx", 0);
    if (ctx < q_len) ctx = q_len;
    const int64_t nblk = (ctx + bs - 1) / bs;
    const uint32_t Q  = dt_act(p, n_p, "q_dtype");
    const uint32_t KV = dt_act(p, n_p, "kv_dtype");
    const RadOpdDesc l[] = {
        opd(Q, q_len, nh, hd),
        opd(KV, nblk, kvh, bs, 2 * hd),
        /* A block-table entry indexes n_blocks; it is read-side, so it may repeat. */
        opd_idx(1, nblk, nblk),
        /* Fixed at ctx, not drawn: a shorter context would make causal attention read a missing
         * prefix. */
        opd_idx(1, 0, 0, ctx),
        /* Absent descales mean 1.0, vLLM's default and what the fp8kv kernels assume. They are
         * one number per (sequence, head) and cannot be derived from the tokens. */
        opd_none(),
        opd_none(),
        /* Absent cu_seqlens: one q_len for the whole batch, exact for one sequence. */
        opd_none(),
        opd(Q, q_len, nh, hd),
    };
    R4D_PICK(l);
}

/* attn_paged_gate_quant -- attn_paged's operands, then gate_quant_fp8's gate, codes and scales,
 * flat as gate_quant_fp8 describes them: one row per (token, head) pair, `head_dim` columns. */
extern "C" int r4d_shape_attn_paged_gate_quant(const RadParam* p, int n_p, int operand,
                                               RadOpdDesc* out) {
    if (operand < 8) return r4d_shape_attn_paged(p, n_p, operand, out);
    const int64_t q_len = pi(p, n_p, "q_len");
    const int64_t hd    = pi(p, n_p, "head_dim");
    const int64_t nh    = pi(p, n_p, "n_head");
    const int64_t g     = pi(p, n_p, "group", R4D_FP8_BLOCK);
    if (q_len <= 0 || hd <= 0 || nh <= 0 || g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const uint32_t Q = dt_act(p, n_p, "q_dtype");
    const int64_t  M = q_len * nh;
    const RadOpdDesc l[] = {
        opd_none(), opd_none(), opd_none(), opd_none(), opd_none(), opd_none(), opd_none(),
        opd_none(),
        opd(Q, M, hd),
        opd((uint32_t)RAD_F8E4M3, M, hd),
        opd((uint32_t)RAD_F32, M, (hd + g - 1) / g),
    };
    R4D_PICK(l);
}

/* qk_norm_rope_gate -- the gated-attention QKV preamble (shim in r4d_qk_norm_rope_gate.hip).
 *
 * Not docs/OPS.md's `qk_norm_rope`: it splits a [q|gate] projection per head, writes q, k and the
 * gate to three destinations, and takes a precomputed cos/sin table (from `rope_table`) instead of
 * a theta. */
extern "C" int r4d_shape_qk_norm_rope_gate(const RadParam* p, int n_p, int operand,
                                           RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t nq  = pi(p, n_p, "n_head");
    const int64_t nkv = pi(p, n_p, "n_head_kv");
    const int64_t rot = pi(p, n_p, "rot");
    const uint32_t Q  = dt_act(p, n_p, "q_dtype");
    /* The kernel reads the cos/sin and position widths off the operands; the optional `cs_f32` /
     * `pos_i64` keys declare the same, so they decide what is allocated here. The issuing arch
     * passes cs_f32 = 1. */
    const uint32_t CS = pi(p, n_p, "cs_f32", 0) ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16;
    const RadOpdDesc l[] = {
        /* The whole projection, q and gate interleaved per head; the kernel splits it. Flat: the
         * shim folds trailing extents into one row pitch. Absent in the k-only form (`n_head` 0);
         * the shim checks for no tensor, not a zero-width one. */
        nq ? opd(Q, M, nq * 2 * hd) : opd_none(),
        opd(Q, M, nkv * hd),
        /* The [max_ctx, rot] plane, one row per position of this band. Unlike `rope`, a position
         * is a lookup into it, so the table's row count bounds the draw. */
        opd(CS, M, rot),
        opd_idx(M, 0, M),
        /* The QK norm gains, one per head channel: f32 in every architecture plugin here (vLLM's
         * are bf16). The shim requires both to agree, since one flag serves both. */
        nq ? opd((uint32_t)RAD_F32, hd) : opd_none(),
        opd((uint32_t)RAD_F32, hd),
        nq ? opd(Q, M, nq * hd) : opd_none(),
        opd(Q, M, nkv * hd),
        /* `gate_out` absent, as the fp8 gated attention block passes it: `gate_quant_fp8` reads
         * the gate in place inside the projection. */
        opd_none(),
    };
    R4D_PICK(l);
}

/* qk_norm_rope_gate_kv_store -- the prologue's operands, then kv_store's v, slot map and cache at
 * kv_store's own shapes (see r4d_shape_kv_store for the slack and the uniqueness). */
extern "C" int r4d_shape_qk_norm_rope_gate_kv_store(const RadParam* p, int n_p, int operand,
                                                    RadOpdDesc* out) {
    if (operand < 9) return r4d_shape_qk_norm_rope_gate(p, n_p, operand, out);
    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t kvh = pi(p, n_p, "n_head_kv");
    const int64_t bs  = pi(p, n_p, "block_size");
    const uint32_t KV = dt_act(p, n_p, "kv_dtype");
    if (M <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    const int64_t nb = (M + bs - 1) / bs + 2;
    const RadOpdDesc l[] = {
        opd_none(), opd_none(), opd_none(), opd_none(), opd_none(), opd_none(), opd_none(),
        opd_none(), opd_none(),
        opd((uint32_t)RAD_BF16, M, kvh, hd),
        opd_idx(M, 0, nb * bs, -1, RAD_OPD_F_IDX_UNIQUE),
        opd(KV, nb, kvh, bs, 2 * hd),
    };
    R4D_PICK(l);
}

/* qkv_norm_rope -- separate q [M, n_head * head_dim], k and v [M, n_head_kv * head_dim] (dense
 * here; the shim also takes each as a slice of a wider row), the positions, and the q and k gains,
 * f32 as every architecture plugin declares them (shim in r4d_qkv_norm_rope.hip). As for `rope`,
 * a position is not a lookup: the kernel takes sincos of position x frequency, so it is drawn over
 * a context's range, where the angles a decode step sees are, not over the band's rows. */
extern "C" int r4d_shape_qkv_norm_rope(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t nq  = pi(p, n_p, "n_head");
    const int64_t nkv = pi(p, n_p, "n_head_kv");
    const RadOpdDesc l[] = {
        /* at n_head 0 the kernel reads no q; it is passed as k's buffer, so drawn like k */
        opd((uint32_t)RAD_BF16, M, (nq ? nq : nkv) * hd),
        opd((uint32_t)RAD_BF16, M, nkv * hd),
        opd((uint32_t)RAD_BF16, M, nkv * hd),
        opd_idx(M, 0, 32768),
        opd((uint32_t)RAD_F32, hd),
        opd((uint32_t)RAD_F32, hd),
    };
    R4D_PICK(l);
}

/* qkv_norm_rope_kv_store -- qkv_norm_rope's six, then kv_store's slot mapping (unique slots, as
 * the engine hands them) and its paged cache, sized as kv_store's description sizes it. */
extern "C" int r4d_shape_qkv_norm_rope_kv_store(const RadParam* p, int n_p, int operand,
                                                RadOpdDesc* out) {
    if (operand < 6) return r4d_shape_qkv_norm_rope(p, n_p, operand, out);
    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t kvh = pi(p, n_p, "n_head_kv");
    const int64_t bs  = pi(p, n_p, "block_size");
    const uint32_t KV = dt_act(p, n_p, "kv_dtype");
    if (M <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    const int64_t nb = (M + bs - 1) / bs + 2;
    const RadOpdDesc l[] = {
        opd_none(), opd_none(), opd_none(), opd_none(), opd_none(), opd_none(),
        opd_idx(M, 0, nb * bs, -1, RAD_OPD_F_IDX_UNIQUE),
        opd(KV, nb, kvh, bs, 2 * hd),
    };
    R4D_PICK(l);
}

/* rope_table -- the cos/sin plane the fused prologue reads (shim in r4d_qk_norm_rope_gate.hip). */
extern "C" int r4d_shape_rope_table(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t rot = pi(p, n_p, "rot");
    const uint32_t D  = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* Only the rows `positions` names are written, each at its own row, so the plane's row
         * count bounds the draw. The real table is [max_ctx, rot]; M rows holds every row this
         * band can name. */
        opd_idx(M, 0, M),
        opd(D, M, rot),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Gated delta net -- the one row of the family whose widths are all parameters; the other five are
 * declined (file header). `gdn_gated_norm_had_quant_i8` is with the fusions below.
 * ========================================================================================= */

extern "C" int r4d_shape_gdn_gated_rmsnorm(const RadParam* p, int n_p, int operand,
                                           RadOpdDesc* out) {

    /* One row per (token, head): `M` is tokens x heads and `channels` is head_v. */
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ch = pi(p, n_p, "channels");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, ch),
        opd((uint32_t)RAD_BF16, M, ch),
        opd((uint32_t)RAD_F32,  ch),      /* "bf16 x / z / out, fp32 weight" -- checked */
        opd((uint32_t)RAD_BF16, M, ch),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Collectives (shims in r4d_ar_entry.hip).
 *
 * These need a real peer over IPC, so a single-process tool gets RAD_E_STATE from init. They are
 * described for a two-process caller.
 * ========================================================================================= */

extern "C" int r4d_shape_all_reduce(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    /* `numel` is the largest message the instance carries (init sizes the peer scratch from it;
     * the launch reads the real count off the tensor), so it is the buffer extent. */
    const int64_t numel = pi(p, n_p, "numel");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, numel),
        /* The optional second destination; absent means in place. Described so a caller that
         * passes one is checked against the same extent. */
        opd(D, numel),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_all_gather(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t numel = pi(p, n_p, "numel");
    const int64_t world = pi(p, n_p, "world_size");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* `row` absent: the message is one row and `y` the plain concatenation, rank r's slice at
     * r * numel. The row-interleaved form has the same extent, so this covers both. */
    const RadOpdDesc l[] = { opd(D, numel), opd(D, world * numel) };
    R4D_PICK(l);
}

/* The two fused all-reduce + norm folds. Their operands match the standalone norms they replace,
 * in order, with `x` also the message. */

extern "C" int r4d_shape_ar_ln_had_quant_i8(const RadParam* p, int n_p, int operand,
                                            RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, n),
        opd((uint32_t)RAD_BF16, M, n),      /* residual: the fused_add_rms_norm form */
        /* The gain is bf16 on the int8 side: r4d_ar_ln_had_quant_i8 takes `wgain` as
         * __hip_bfloat16 with no flag (the fp8 twin below reads the width off the operand). An f32
         * gain would silently be read as two wrong bf16 values. */
        opd((uint32_t)RAD_BF16, n),
        opd((uint32_t)RAD_I8, M, n),
        /* One scale per row, viable because of the rotation; it also carries the 1/sqrt(had)
         * normalisation. This family has no scale group. */
        opd((uint32_t)RAD_F32, M, 1),
        opd((uint32_t)RAD_BF16, M, n),      /* out_bf16, for a bf16 consumer of the post-norm row */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_ar_rmsnorm_quant_fp8(const RadParam* p, int n_p, int operand,
                                              RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    /* The block is the checkpoint's scale grid, not a knob; the shim refuses any other. */
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, n),      /* inout: the message and the input */
        opd((uint32_t)RAD_BF16, M, n),
        opd((uint32_t)RAD_F32,  n),         /* f32 or bf16; the shim reads the operand's own */
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd((uint32_t)RAD_BF16, M, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_ar_rmsnorm_had_quant_i8g(const RadParam* p, int n_p, int operand,
                                                  RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, n),      /* inout: the message and the input */
        opd((uint32_t)RAD_BF16, M, n),
        opd((uint32_t)RAD_F32,  n),         /* f32 or bf16; the shim reads the operand's own */
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd((uint32_t)RAD_BF16, M, n),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * GEMM, dequant and the row top-R (shims in the r4d_gemm_*.hip units, r4d_dequant_w4_bf16.hip and
 * r4d_rowtopk_bf16.hip).
 * ========================================================================================= */

extern "C" int r4d_shape_gemm_nt_f32(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const RadOpdDesc l[] = { opd((uint32_t)RAD_BF16, M, K), opd((uint32_t)RAD_F32, N, K),
                             opd((uint32_t)RAD_F32, M, N) };
    R4D_PICK(l);
}
extern "C" int r4d_shape_gemm_nt(const RadParam* p, int n_p, int operand,
                                 RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, K), opd(D, N, K), opd(D, M, N) };
    R4D_PICK(l);
}

/* gemm_nt_q, the block-scaled fp8 family only; the other five grids are declined (file header).
 * The checkpoint fixes the grid: E4M3 [N][K] against a bf16 [ceil(N/128)][ceil(K/128)] plane,
 * `group` carrying the 128. The fragment order the narrow and tiled rows read is a permutation of
 * the same bytes (frag_bytes(N, K) == N * K), so one description sizes every row. */
extern "C" int r4d_shape_gemm_nt_q(const RadParam* p, int n_p, int operand,
                                   RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const char* q = r4d_pgets(p, n_p, "dtype");
    const bool a8  = q && !std::strcmp(q, "fp8a8");
    const bool a16 = q && !std::strcmp(q, "fp8a16");
    if ((!a8 && !a16) || g != R4D_FP8_BLOCK || K <= 0 || N <= 0) return RAD_E_UNSUPPORTED;
    const int64_t kb = (K + g - 1) / g, nb = (N + g - 1) / g;
    const RadOpdDesc l[] = {
        /* a, a_scale. The matvec reads the bf16 activation directly, and its shim refuses an
         * activation scale rather than silently ignoring it. */
        a8 ? opd((uint32_t)RAD_F8E4M3, M, K) : opd((uint32_t)RAD_BF16, M, K),
        a8 ? opd((uint32_t)RAD_F32, M, kb)   : opd_none(),
        opd((uint32_t)RAD_F8E4M3, N, K),     /* the checkpoint's own bytes */
        opd((uint32_t)RAD_BF16, nb, kb),     /* weight_scale_inv, one per 128x128 tile */
        opd((uint32_t)RAD_BF16, M, N),
        /* `a_sum` belongs to the asymmetric integer grids and `b_ref` to mxfp4; fp8 has neither. */
        opd_none(),
        opd_none(),
    };
    R4D_PICK(l);
}

/* gemm_nt_q_bias at fp8a8: gemm_nt_q's fp8 operands, a bf16 bias per output column, then y. The
 * libref description matches it (ref_gemm_nt_q_bias_shape). */
extern "C" int r4d_shape_gemm_nt_q_bias(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M");
    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const char* q = r4d_pgets(p, n_p, "dtype");
    if (!q || std::strcmp(q, "fp8a8") || g != R4D_FP8_BLOCK || K <= 0 || N <= 0)
        return RAD_E_UNSUPPORTED;
    const int64_t kb = (K + g - 1) / g, nb = (N + g - 1) / g;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F8E4M3, M, K),
        opd((uint32_t)RAD_F32, M, kb),
        opd((uint32_t)RAD_F8E4M3, N, K),
        opd((uint32_t)RAD_BF16, nb, kb),
        opd((uint32_t)RAD_BF16, N),
        opd((uint32_t)RAD_BF16, M, N),
    };
    R4D_PICK(l);
}

/* The same GEMM with the SwiGLU and the quantiser in its epilogue. `N` is the weight's rows, gate
 * then up, so the quantiser's outputs are N/2 wide (hence a schema of its own). */
extern "C" int r4d_shape_gemm_nt_q_gated(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const char* q = r4d_pgets(p, n_p, "dtype");
    if (!q || std::strcmp(q, "fp8a8") || g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    if (N <= 0 || N % 2 || K <= 0) return RAD_E_UNSUPPORTED;
    const int64_t nout = N / 2;
    const int64_t kb = (K + g - 1) / g;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F8E4M3, M, K),
        opd((uint32_t)RAD_F32, M, kb),
        opd((uint32_t)RAD_F8E4M3, N, K),
        opd((uint32_t)RAD_BF16, (N + g - 1) / g, kb),
        opd((uint32_t)RAD_F8E4M3, M, nout),
        opd((uint32_t)RAD_F32, M, (nout + g - 1) / g),
        opd((uint32_t)RAD_BF16, M, nout),
    };
    R4D_PICK(l);
}

/* dequant_w4 -- the packed 4-bit weight back to row-major bf16 (shim in r4d_dequant_w4_bf16.hip).
 *
 * The one packed weight a public dtype sizes exactly: RAD_I4 is four bits per element, so [N, K]
 * is the N*K/2 bytes r4d_rad_layout_w4 declares. The scale plane is one (f16 scale,
 * f16 -(1024+zero)) dword per (row, group), tile-major [N/16][K/group][16], read at
 * [(t * ng + k/group) * 16 + r].
 *
 * Neither stream's content is expressible (bit layouts, not numbers), so the extents are exact for
 * timing but a correctness comparison here means nothing. */
extern "C" int r4d_shape_dequant_w4(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const int64_t g = pi(p, n_p, "group", R4D_W4_GROUP);
    if (N <= 0 || K <= 0 || g <= 0 || N % R4D_NTILE || K % g) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_I4, N, K),
        /* i32, not u32: a u32 operand is filled with zeroes (see opd_sample_params). */
        opd((uint32_t)RAD_I32, N / R4D_NTILE, (K / g) * R4D_NTILE),
        opd((uint32_t)RAD_BF16, N, K),
    };
    R4D_PICK(l);
}

/* ---- the routed mixture of experts (shims in r4d_moe.hip) --------------------------------------
 *
 * `expert_offset` is moe_scatter's output, a non-decreasing prefix array ending at the live row
 * count, searched to find a sorted row's expert. No ABI fill domain produces that, so a random draw
 * is a malformed histogram and two implementations may search it differently without either being
 * wrong. The extents are right for a benchmark; a correctness comparison over an op that reads it
 * proves little. libref describes the operand the same way. */
extern "C" int r4d_shape_scale_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M), opd(D, M, n), opd(D, M, n) };
    R4D_PICK(l);
}

/* The per-tensor scalar: one [1] weight `s` (the middle operand) broadcast over [M, n]. Its dtype
 * is the stream's, bf16 or f32. */
extern "C" int r4d_shape_scale(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, 1), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_router_topk(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    if (ne <= 0 || ne > 256 || k <= 0 || k > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, ne),
        opd((uint32_t)RAD_I32, M, k),
        opd(D, M, k),
    };
    R4D_PICK(l);
}

/* router_topk's and moe_scatter's lists concatenated, with `expert_ids` named once (here an
 * output). */
extern "C" int r4d_shape_router_topk_scatter(const RadParam* p, int n_p, int operand,
                                             RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    if (ne <= 0 || ne > 256 || k <= 0 || k > ne) return RAD_E_UNSUPPORTED;
    /* The sort's single-workgroup bound (`kScatterSmall`), which the shim enforces. */
    if (M * k > 1024) return RAD_E_UNSUPPORTED;
    /* The sort's three outputs are sized by this rank's slice of experts, not by the router. */
    const int64_t nl = pi(p, n_p, "n_local", ne);
    if (nl <= 0 || nl > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, ne),
        opd((uint32_t)RAD_I32, M, k),
        opd(D, M, k),
        opd((uint32_t)RAD_I32, M * k),
        opd((uint32_t)RAD_I32, nl + 1),
        opd((uint32_t)RAD_I32, nl),
    };
    R4D_PICK(l);
}
/* The router's gemm_nt before router_topk_scatter, plus the shared gate's gemm_nt at N = 1: the
 * three lists, with `logits` named once as the GEMM's output. */
extern "C" int r4d_shape_router_gemm_topk_scatter(const RadParam* p, int n_p, int operand,
                                                  RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t K  = pi(p, n_p, "K");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    /* At most 512 experts: the selection keeps sixteen logits per lane (kMaxExpert in
     * r4d_moe.hip), the shim's bound and the row's constraint. */
    if (ne <= 0 || ne > 512 || k <= 0 || k > ne || K <= 0) return RAD_E_UNSUPPORTED;
    if (M > 64 || M * k > 1024) return RAD_E_UNSUPPORTED;
    const int64_t nl = pi(p, n_p, "n_local", ne);
    if (nl <= 0 || nl > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, K),
        opd(D, ne, K),
        opd(D, 1, K),
        opd(D, M, ne),
        opd((uint32_t)RAD_I32, M, k),
        opd(D, M, k),
        opd((uint32_t)RAD_I32, M * k),
        opd((uint32_t)RAD_I32, nl + 1),
        opd((uint32_t)RAD_I32, nl),
        opd(D, M, 1),
    };
    R4D_PICK(l);
}
extern "C" int r4d_shape_moe_scatter(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    if (ne <= 0 || ne > 256 || k <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        /* Expert ids index experts, no operand's dim 0, so the range is stated. */
        opd_idx(M, k, ne, -1),
        opd((uint32_t)RAD_I32, M * k),
        opd((uint32_t)RAD_I32, ne + 1),
        opd((uint32_t)RAD_I32, ne),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_moe_scatter_inv(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    if (ne <= 0 || ne > 512 || k <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd_idx(M, k, ne, -1),
        opd((uint32_t)RAD_I32, M * k),
        opd((uint32_t)RAD_I32, ne + 1),
        opd((uint32_t)RAD_I32, ne),
        opd((uint32_t)RAD_I32, M * k),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_moe_gemm_q(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t N0 = pi(p, n_p, "N"), K0 = pi(p, n_p, "K");
    const int64_t N1 = pi(p, n_p, "N_odd", N0), K1 = pi(p, n_p, "K_odd", K0);
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    int64_t g = pi(p, n_p, "group");
    if (ne <= 0 || k <= 0 || N0 <= 0 || K0 <= 0 || N1 <= 0 || K1 <= 0) return RAD_E_UNSUPPORTED;
    if (g <= 0) g = 128;
    /* The output is the wider class's width and the activation the longer class's K. */
    const int64_t N = N0 > N1 ? N0 : N1, K = K0 > K1 ? K0 : K1;
    const int64_t ne0 = (ne + 1) / 2, ne1 = ne / 2;
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* `a_order` sets the activation's row count: one per token, or one per (token, slot) already
     * in scatter order ("sorted"). */
    const char* ord = rad_param_gets(p, n_p, "a_order", nullptr);
    const int64_t AR = (ord && !std::strcmp(ord, "sorted")) ? M * k : M;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F8E4M3, AR, K),                 /* a -- see `a_order` */
        opd((uint32_t)RAD_F32, AR, (K + g - 1) / g),      /* a_scale */
        /* The weight operands are tables when issued, described here as one stacked plane; a
         * fixture cannot hold a pointer array, and the kernel reads both forms (r4d_moe.hip,
         * wtab_at). */
        opd((uint32_t)RAD_F8E4M3, ne0, N0, K0),           /* w -- the even experts */
        opd((uint32_t)RAD_BF16, ne0, (N0 + g - 1) / g, (K0 + g - 1) / g),   /* w_scale */
        opd_idx(M * k, 0, M * k, -1),                     /* sorted_tok */
        opd_idx(ne + 1, 0, M * k + 1, -1),                /* expert_offset -- see the note above */
        opd((uint32_t)RAD_F8E4M3, ne1, N1, K1),           /* w_odd -- the odd experts */
        opd((uint32_t)RAD_BF16, ne1, (N1 + g - 1) / g, (K1 + g - 1) / g),   /* w_odd_scale */
        opd(D, M * k, N),                                 /* y, in sorted order */
    };
    R4D_PICK(l);
}

/* The four-class form: class q holds the experts e with e % 4 == q at [N_q, K_q], defaulting to
 * class 0's, ceil((n - q) / 4) of them; `y` follows the last class. */
extern "C" int r4d_shape_moe_gemm_q_mod4(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    int64_t Nc[4], Kc[4];
    Nc[0] = pi(p, n_p, "N");
    Kc[0] = pi(p, n_p, "K");
    Nc[1] = pi(p, n_p, "N_1", Nc[0]); Kc[1] = pi(p, n_p, "K_1", Kc[0]);
    Nc[2] = pi(p, n_p, "N_2", Nc[0]); Kc[2] = pi(p, n_p, "K_2", Kc[0]);
    Nc[3] = pi(p, n_p, "N_3", Nc[0]); Kc[3] = pi(p, n_p, "K_3", Kc[0]);
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    int64_t g = pi(p, n_p, "group");
    if (ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    if (g <= 0) g = 128;
    int64_t N = 0, K = 0;
    for (int q = 0; q < 4; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0) return RAD_E_UNSUPPORTED;
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    int64_t nq[4];
    for (int q = 0; q < 4; ++q) nq[q] = (ne + 3 - q) / 4;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const char* ord = rad_param_gets(p, n_p, "a_order", nullptr);
    const int64_t AR = (ord && !std::strcmp(ord, "sorted")) ? M * k : M;
    auto wq = [&](int q) { return opd((uint32_t)RAD_F8E4M3, nq[q], Nc[q], Kc[q]); };
    auto sq = [&](int q) {
        return opd((uint32_t)RAD_BF16, nq[q], (Nc[q] + g - 1) / g, (Kc[q] + g - 1) / g);
    };
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F8E4M3, AR, K),                 /* a -- see `a_order` */
        opd((uint32_t)RAD_F32, AR, (K + g - 1) / g),      /* a_scale */
        wq(0), sq(0),                                     /* w, w_scale -- class 0 */
        opd_idx(M * k, 0, M * k, -1),                     /* sorted_tok */
        opd_idx(ne + 1, 0, M * k + 1, -1),                /* expert_offset */
        wq(1), sq(1), wq(2), sq(2), wq(3), sq(3),         /* classes 1, 2, 3 */
        opd(D, M * k, N),                                 /* y -- in SORTED order */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_moe_gather(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    const int64_t k = pi(p, n_p, "top_k");
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M * k, n),
        opd(D, M, k),
        /* Distinct: gather rebuilds the inverse permutation of sorted_tok, and a repeated entry
         * would let two sorted rows race for one (token, slot). */
        opd_idx(M * k, 0, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE),
        /* shared and shared_gate: optional, described so a caller exercises the folded form. */
        opd(D, M, n),
        opd(D, M),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* moe_gather's list with the slot table in place of `sorted_tok`: the same draw, since a valid
 * table, like a valid sort, holds distinct rows of `y_expert` (and -1 for a dropped slot). */
extern "C" int r4d_shape_moe_gather_inv(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return r4d_shape_moe_gather(p, n_p, operand, out);
}

/* row_topk -- exact per-row top-R of a wide bf16 matrix (shim in r4d_rowtopk_bf16.hip). R is at
 * most 32, stage 2's PT. */
extern "C" int r4d_shape_row_topk(const RadParam* p, int n_p, int operand,
                                  RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t N = pi(p, n_p, "N");
    const int64_t R = pi(p, n_p, "R");
    const uint32_t D = dt_act(p, n_p, "dtype");
    if (R > 32) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, N),
        opd((uint32_t)RAD_I32, M, R),
        opd((uint32_t)RAD_F32, M, R),
        /* `pairs`, the vocab-parallel output (optional; absent with `vocab_off` zero is the
         * single-rank launch). Described, as in ref, because comparing it checks the packing: an
         * int32 id memcpy'd into the first float slot beside its value, id -1 for an unfilled
         * slot. f32 because the id travels as a bit pattern while the value must be a float. */
        opd((uint32_t)RAD_F32, M, R, 2),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_row_topk_merge(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t R = pi(p, n_p, "R");
    const int64_t W = pi(p, n_p, "world_size");
    if (R > 32) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        /* [M, world_size * R, 2] f32 pairs, rank r's at [r*R, (r+1)*R), an int32 id in the first
         * slot of each. Describable, unlike `sample_merge_topk`, because the plane is f32: a normal
         * fill puts a real float in the value lane the merge orders on, and the id lane reads the
         * same bits as an int32. A negative id is a pad the kernel drops, a positive one a large
         * legal id, so about half the candidates are pads and the rest are ordered without ties,
         * exercising both rules. */
        opd((uint32_t)RAD_F32, M, W * R, 2),
        opd((uint32_t)RAD_I32, M, R),
        opd((uint32_t)RAD_F32, M, R),
    };
    R4D_PICK(l);
}

/* pair_argmax (r4d_pair_argmax_f32.hip): row_topk_merge at R = 1 over `n` gathered pairs, with
 * the same f32 plane and fill reading, and one id and one value out per row. */
extern "C" int r4d_shape_pair_argmax(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, n, 2),
        opd((uint32_t)RAD_I32, M),
        opd((uint32_t)RAD_F32, M),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Quantisation, the norms and the fusions (shims beside their kernels).
 *
 * Two families, separated by the scale. The int8 side carries one f32 scale per row; its `group`
 * is the rotation width of the rotated forms (the block Walsh-Hadamard makes one scale viable) and
 * the width of quant_act_i8's code sums, never a scale grid. The fp8 side is not rotated and
 * carries an f32 scale per (row, 128 columns), the block the checkpoint's weight scales use.
 * ========================================================================================= */

extern "C" int r4d_shape_quant_act_i8(const RadParam* p, int n_p, int operand,
                                      RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),
        /* The `_asum` form's group-major sums: whole 16-row tiles, as the asymmetric GEMMs read
         * them. */
        g > 0 && n % g == 0 ? opd((uint32_t)RAD_F32, n / g, (M + 15) / 16 * 16) : opd_none(),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_had_quant_act_i8(const RadParam* p, int n_p, int operand,
                                          RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rmsnorm_had_quant_i8(const RadParam* p, int n_p, int operand,
                                              RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd(D, M, n),                       /* residual, the fused_add_rms_norm form */
        opd((uint32_t)RAD_BF16, n),         /* bf16 gain; see r4d_shape_ar_ln_had_quant_i8 */
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),       /* one scale per row */
        opd(D, M, n),                       /* out_bf16, unrotated, for a bf16 consumer */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gated_had_quant_i8(const RadParam* p, int n_p, int operand,
                                            RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* The packed gate_up form: `b` absent means one [M, 2n] tensor with `b` at `a + n`, vLLM's
         * layout and what the MLP produces. The two-buffer form differs only in pitch. */
        opd(D, M, 2 * n),
        opd_none(),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),       /* one scale per row */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gdn_gated_norm_had_quant_i8(const RadParam* p, int n_p, int operand,
                                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t n  = pi(p, n_p, "n");
    const int64_t hd = pi(p, n_p, "head_dim");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* The RMS is per head while the rotation and the int8 scale are per token (so no unfused
     * equivalent exists), and the gain is head_dim floats, not n; the shim checks this.
     *
     * `zgrp` / `zblk` describe an interleaved qkvz projection, element i of the flattened row at
     * (i / zblk) * zgrp + (i % zblk). Only the contiguous gate is described; the interleaved one
     * is a column view of a plane whose width no parameter carries. */
    if (pi(p, n_p, "zgrp", 0) != 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd(D, M, n),
        opd((uint32_t)RAD_F32, hd),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),       /* one scale per row */
    };
    R4D_PICK(l);
}

/* ---- the fp8 side. `group` is the scale grid here, 128 from the checkpoint. ---- */

extern "C" int r4d_shape_quant_act_fp8(const RadParam* p, int n_p, int operand,
                                       RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");   /* the input's width; the output is E4M3 */
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
    };
    R4D_PICK(l);
}

/* The QSA work list. `M` is the token count and `work` the block bound: two different extents,
 * hence two parameters. */
extern "C" int r4d_shape_qsa_work(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t r = pi(p, n_p, "ratio", 4);
    const int64_t g = pi(p, n_p, "ring", r);
    const int64_t s = pi(p, n_p, "seqs", 1);
    const int64_t W = pi(p, n_p, "work", 1);
    if (n <= 0 || r <= 0 || g < r || s <= 0 || W <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd_idx(M, 1, M * r),          /* positions */
        opd_idx(s + 1, 1, M + 1),      /* cu_seqlens; a drawn one is not ascending, see below */
        /* The tail state as the group binds it: [slots, heads, ring, n], heads 1. One slot per
         * sequence, the narrowest case in which `sidx` means something. */
        opd(D, s, 1, g, n),
        opd_idx(s, 1, s),              /* the state row; one column, values inside the pool */
        opd_idx(s, W, W),              /* the block-key block table */
        opd_none(),                    /* rope: the one-component form (every text model) */
        opd(D, W * r, n),
        opd((uint32_t)RAD_I32, W),
        opd((uint32_t)RAD_I32, W),
        opd((uint32_t)RAD_I32, s),
    };
    R4D_PICK(l);
}

/* The QSA tail store. `cu` is `seqs + 1` long; the shim takes the count from the operand at run
 * time, so `seqs` here only sizes the buffer. */
extern "C" int r4d_shape_qsa_tail_store(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t r = pi(p, n_p, "ratio", 4);
    const int64_t g = pi(p, n_p, "ring", r);
    const int64_t s = pi(p, n_p, "seqs", 1);
    if (n <= 0 || r <= 0 || g < r || s <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd_idx(M, 1, M * r),          /* positions; bounded by what the tail can index */
        opd_idx(s + 1, 1, M + 1),      /* cu_seqlens should ascend; a drawn one does not, so a
                                        * synthesised case tests the scan, not the copy */
        opd(D, s, 1, g, n),            /* the tail state: [slots, heads, ring, n], heads 1 */
        opd_idx(s, 1, s),              /* the state row; one column, values inside the pool */
    };
    R4D_PICK(l);
}

/* The QSA selection. `sel` is `topk + 1` wide: the extra slot is the partial page, without which
 * the newest tokens would be dropped. `selb` is the same shape holding block indices, which a table
 * shared by a tile of query rows needs to sort the rows' entries (a page id does not sort by
 * block). */
extern "C" int r4d_shape_qsa_select(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nb = pi(p, n_p, "blocks");
    const int64_t k  = pi(p, n_p, "topk");
    if (nb <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    const int64_t r = pi(p, n_p, "ratio", 4);
    const int64_t s = pi(p, n_p, "seqs", 1);
    if (s <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nb),
        opd_idx(s, nb, nb),                   /* page ids, and they index nothing wider */
        opd_idx(s, 1, nb + 1),                /* complete blocks per sequence, at most nb */
        opd_idx(M, 1, nb * r),                /* the query's own position */
        opd_idx(s + 1, 1, M + 1, -1, RAD_OPD_F_IDX_CU),
        opd((uint32_t)RAD_I32, M, k + 1),    /* sel: the selected pages */
        opd((uint32_t)RAD_I32, M),
        opd((uint32_t)RAD_I32, M),
        opd((uint32_t)RAD_I32, M, k + 1),    /* selb: the same entries as block indices */
    };
    R4D_PICK(l);
}

/* The QSA block score. The block-key store's height is the KV pool's page count, which the
 * runtime decides, so `blocks` gives the widest a table entry can address: a bound, not the plan.
 * Table entries are page ids. */
extern "C" int r4d_shape_qsa_score(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t n  = pi(p, n_p, "n");
    const int64_t h  = pi(p, n_p, "heads");
    const int64_t nb = pi(p, n_p, "blocks");
    const int64_t s  = pi(p, n_p, "seqs", 1);
    if (n <= 0 || h <= 0 || nb <= 0 || s <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, h * n),
        opd(D, nb, n),                 /* the store, bounded by the blocks one query can address */
        /* Page ids, which must index the store: the ABI default (the previous operand) fits. */
        opd_idx(s, nb, 0),
        opd_idx(s + 1, 1, M + 1, -1, RAD_OPD_F_IDX_CU),
        opd_idx(s, 1, nb + 1),         /* complete blocks per sequence, at most nb */
        opd((uint32_t)RAD_F32, M, nb),
    };
    R4D_PICK(l);
}

/* The QSA compressed key. `M` counts blocks, and the key input is `ratio` times taller. */
extern "C" int r4d_shape_qsa_block_key(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t r = pi(p, n_p, "ratio", 4);
    if (n <= 0 || r <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M * r, n),
        opd((uint32_t)RAD_F32, n),
        opd_idx(M, 1, M),          /* page ids; a synthesised one addresses the store below */
        opd_idx(M, 1, M * r),      /* each block's first-token position */
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* H += X^T X. The accumulator is [n][n] (K on both axes), not [M][n] as elsewhere; a row-shaped
 * `h` is refused. */
extern "C" int r4d_shape_gram_accum(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    /* dtype bf16 is a plain activation with no scale operand; fp8a8 is E4M3 codes and a dequant
     * scale per (row, group). */
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    if (dt && !std::strcmp(dt, "bf16")) {
        const RadOpdDesc l[] = {
            opd((uint32_t)RAD_BF16, M, n),
            opd_none(),
            opd((uint32_t)RAD_F32, n, n),
        };
        R4D_PICK(l);
    }
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd((uint32_t)RAD_F32, n, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rmsnorm_quant_fp8(const RadParam* p, int n_p, int operand,
                                           RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, n),
        /* The residual is second, as in `rmsnorm_had_quant_i8`, so the two ops differ only by
         * the rotation. Passed, as every caller issues it. The engine may alias it with
         * out_bf16; separate buffers are the safe case. */
        opd(D, M, n),
        /* f32 or bf16: the shim reads the operand's width and passes the kernel a flag. Every
         * architecture plugin declares norm gains f32. */
        opd((uint32_t)RAD_F32, n),
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gated_quant_fp8(const RadParam* p, int n_p, int operand,
                                         RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, 2 * n),                   /* gate then up in one buffer */
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* The `_had_quant_i8g` twins: the same operands with int8 codes (had_quant_act_i8g's). */
extern "C" int r4d_shape_rmsnorm_had_quant_i8g(const RadParam* p, int n_p, int operand,
                                               RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd(D, M, n),
        opd((uint32_t)RAD_F32, n),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gated_had_quant_i8g(const RadParam* p, int n_p, int operand,
                                             RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, 2 * n),                   /* gate then up in one buffer */
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gate_quant_fp8(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    /* A row is a (token, head) pair: `M` is tokens * n_head and `n` is head_dim, so the gate is a
     * fixed stride inside the [q|gate] projection and the scale plane lands where the out
     * projection reads it. In the engine the gate is a column slice; described flat. */
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd(D, M, n),
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * DFlash2 (shims in r4d_dflash_conv_t2_g16_bf16.hip and r4d_dflash_select_bf16.hip). Neither op has an unfused equivalent or a ref implementation.
 * ========================================================================================= */

extern "C" int r4d_shape_dflash_conv(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t T    = pi(p, n_p, "T");
    const int64_t H    = pi(p, n_p, "hidden_size");
    const int64_t taps = pi(p, n_p, "taps");
    const int64_t NG   = pi(p, n_p, "NG");
    const int64_t grp  = pi(p, n_p, "group_size");
    const uint32_t D   = dt_act(p, n_p, "dtype");
    /* The entry point's conditions: NG is H/group and a thread owns eight channels. */
    if (H <= 0 || grp <= 0 || NG <= 0 || taps <= 0) return RAD_E_UNSUPPORTED;
    if (H % (grp * 8) || NG != H / grp) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, T, H),
        /* delta is a slice of the [T, 2, taps, NG] kernel projection, row pitch 2*taps*NG; a
         * packed pitch would silently read the other side's coefficients. Declared at the
         * projection's rank so the shape carries the pitch. */
        opd(D, T, 2, taps, NG),
        opd(D, taps, H),                    /* base_kernel: a weight, contiguous [taps, H] */
        opd(D, T, H),                       /* out; must not alias x */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_dflash_select(const RadParam* p, int n_p, int operand,
                                       RadOpdDesc* out) {

    const int64_t M     = pi(p, n_p, "M");
    const int64_t steps = pi(p, n_p, "steps");
    const int64_t K     = pi(p, n_p, "top_k");
    const int64_t R     = pi(p, n_p, "rank");
    const int64_t V     = pi(p, n_p, "n_vocab");
    const int64_t astr  = pi(p, n_p, "anchor_stride", 1);
    const uint32_t D    = dt_act(p, n_p, "dtype");
    if (M <= 0 || astr < 1) return RAD_E_UNSUPPORTED;
    /* The three input planes are read at row s*steps+l, so each holds M*steps rows; a short one
     * would read another sequence's candidates. */
    const int64_t rows = M * steps;
    const RadOpdDesc l[] = {
        /* Candidate token ids, used as gather rows: the walk reads pred[id] and succ[id] from two
         * [n_vocab, rank] codebooks (operands 4 and 5), so the range is n_vocab, not the ABI
         * default. */
        opd_idx(rows, K, V),
        opd((uint32_t)RAD_F32, rows, K),
        opd(D, rows, R),
        /* `anchor` is usually the step's token_ids, `block` ids per sequence with the anchor
         * first, hence the stride. Gathered like the candidates, so the same range. The buffer
         * holds (M-1)*stride + 1. */
        opd_idx((M - 1) * astr + 1, 0, V),
        opd(D, V, R),
        opd(D, V, R),
        opd((uint32_t)RAD_I32, M, steps),
    };
    R4D_PICK(l);
}
