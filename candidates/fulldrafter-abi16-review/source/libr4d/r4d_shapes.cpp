/* r4d_shapes.cpp -- how big is operand i of this op at this geometry, and what may be in it.
 *
 * THE COMPONENT THAT KNOWS AN OPERAND'S EXTENT IS THE ONE THAT READS IT, which is what the
 * `RadShapeFn` hook is for. A tool's other source is the hand-written table in `tools/`, and that
 * table cannot reach the ops that exist only here -- every fp8 fusion, both all-reduce folds, both
 * DFlash2 ops, `attn_dense` and all twelve sampler stages -- so without this hook those are
 * SKIPPED AND COUNTED for want of a recipe. This file is libr4d's side of the hook.
 *
 * THE AUTHORITY FOR EVERY EXTENT BELOW IS THE SHIM, not the prose and not the row table. Each
 * `rad_*` entry point in the libr4d shims unpacks `RadArgs.t[i]` and hands libr4d the
 * extents and pitches; where a shim reads `x->shape[1]` and passes it as `K`, that is what the
 * operand is. A comment naming the shim is on every group, so a reader can check one against
 * the other -- and when a shim's contract changes, that is the file whose neighbour has to move.
 *
 * WHAT IS IN AN OPERAND MATTERS AS MUCH AS HOW BIG IT IS. A token id gathers a row of a codebook,
 * a KV slot indexes n_blocks * block_size positions, a block-table entry indexes n_blocks, and a
 * uniform is in [0, 1); draw any of them from the wrong range and the case tests the kernel's
 * out-of-range handling while the report calls it correctness. `RAD_FILL_INDEX` with an explicit
 * `idx_max` is how each of those is said here, and `RAD_OPD_F_IDX_UNIQUE` marks the one scatter
 * (kv_store's slot mapping) whose duplicates would make the store a race and the comparison a
 * measurement of the scheduler.
 *
 * WHAT THIS FILE DOES NOT DESCRIBE, AND WHY. A decline is a real answer: the caller skips the case
 * and counts it, which is correct, where an invented extent produces a PASS that means nothing.
 * The rows below carry a null `opd_shape` deliberately.
 *
 *   gdn_conv_prep, gdn_conv_update, gdn_kkt_solve, gdn_chunk_scan, gdn_recurrent_update
 *       THE HEAD COUNTS ARE NOT IN THE GEOMETRY. `head_k` and `head_v` are head WIDTHS (128 each);
 *       the number of key heads and value heads is read off the operands by the kernels and named
 *       by no parameter of any of these ops, so `conv_dim = 2*Hk*hk + Hv*hv` -- the leading extent
 *       of nearly every operand -- is not derivable from `p`. tools/opshapes.h reaches them only
 *       because rad-kbench hands it the container's metadata through `ShapeCtx`, which this hook's
 *       signature does not carry. A GDN case run at an invented head ratio is a check of a model
 *       nobody is running. (`gdn_gated_rmsnorm` and `gdn_gated_norm_had_quant_i8` ARE described:
 *       their rows are (token, head) pairs and their widths are parameters.)
 *
 *   attn_dense
 *       No head count either. Its parameters are M, head_dim, gqa, causal and dtype; the vision
 *       kernel reads the head count off `q->shape[1]` and gqa is constrained to 1, so nothing in
 *       the geometry says how many heads a token carries. Picking one would be picking a model.
 *       ref describes it -- it chooses `gqa` query heads against one KV head, which at this row's
 *       gqa == 1 is a single head -- and ref's is the description a correctness check uses anyway,
 *       so the gap this leaves is a benchmark of the vision kernel at a head count nobody runs.
 *
 *   sample_merge_topk
 *       The extents are known and a description would still be a lie. `gathered` is
 *       world_size * n_cand (u32 global id, f32 logit) pairs packed into one WORD-TYPED buffer, so
 *       the value lane is read by BIT PATTERN and no fill domain puts a sensible float there: an
 *       integer draw makes every logit a denormal, and a device that flushes denormals against a
 *       host that does not orders the candidates differently -- which is the failure r4d_selftest's
 *       two-rank all-gather row exists to catch. `world_size` is not in the parameter list either.
 *       ref declines it for the same two reasons, and the two files agreeing on a decline is the
 *       point: a description here would supply the bytes ref refused to. The full argument is at
 *       the gap this leaves in the sampler section below.
 *
 *   gemm_nt_q at every grid but fp8
 *       The weight is a plugin-private fragment stream (R4D_DT_W4_FRAG and friends) whose byte
 *       count comes from `RadLayout::bytes`, and RadOpdDesc has only a dtype and a shape -- so a
 *       description here would be a SECOND source for a number r4d_layout.cpp already owns, and
 *       the two would drift silently the first time a packing changed. The scale planes are worse:
 *       R4D_DT_W4_SZ is a (f16 scale, f16 -(1024+zero)) dword whose meaning no public dtype
 *       carries. The fp8 family is different and IS described -- its weight is the checkpoint's
 *       own E4M3 [N][K] bytes against a bf16 [ceil(N/128)][ceil(K/128)] plane, and the fragment
 *       reordering is a permutation of the same count (r4d_fp8_frag.h: `frag_bytes(N,K) == N*K`).
 *
 *   logits_gemm at fp8
 *       THE DECLARED SHAPE AND THE STORED PLANE DIFFER, and only the container knows. The weight
 *       arrives declared [n_vocab][n_embd] E4M3 while the kernel strides it by
 *       r4d_fp8::lm_row_bytes(n_embd) --
 *       the row's own bf16 scales ride in its tail (r4d_fp8_layout.cpp says why: lm_head is ROW
 *       sharded and a shard has to stay a plain byte range). A tool that allocated the declared
 *       shape would hand the kernel a buffer it reads 1.6% past the end of, on every row. The
 *       bf16 row of the same op is described; this one cannot be, until RadOpdDesc can say
 *       "declared this, stored that".
 *
 * NO TUNED AXIS IS READ HERE, and none needs to be. Every axis this plugin declares -- tile
 * shapes, split counts, wave counts, all-reduce launch geometry -- says how a kernel WALKS its
 * operands, not what it is HANDED. An axis that did change an extent would be read from `p`
 * like any other parameter, because the core appends the chosen point to the same list.
 *
 * Pure host code: no r4d.h, no HIP. It is in the CMake `SOURCES` list beside r4d_rows.cpp and
 * r4d_layout.cpp for the reason that split exists -- the tables a transcription error hides in
 * should compile and be readable on a box with no ROCm.
 */

#include <cstdlib>
#include <cstring>
#include "r4d_plugin.h"

#include "rad_sample.h"

namespace {

/* ==================================================================== building a description
 *
 * Each op below builds its whole operand list, in schema order, and then hands out the one the
 * caller asked for. Building the list rather than switching on the index is what makes the COUNT
 * checkable by eye against the schema in r4d_rows.cpp -- the tool cross-checks it against
 * `RadOpSchema::n_operands` as well, and a hook that has drifted from its schema is caught there,
 * but a list a reader can count is the half that stops the drift happening.
 */

/* A plain operand of up to four extents. A trailing 0 means "this rank is not present": zero is
 * never a legal extent -- `pick` refuses one below -- so it cannot be confused with a real one,
 * and it lets a two-dimensional and a four-dimensional operand be written the same way. */
RadOpdDesc opd(uint32_t dt, int64_t d0, int64_t d1 = 0, int64_t d2 = 0, int64_t d3 = 0) {
    RadOpdDesc d{};
    d.dtype     = dt;
    d.fill      = RAD_FILL_NORMAL;
    /* Explicit, though the caller pre-initialises it: -1 is "draw", and a description that meant
     * "every element is zero" and got it by leaving a field alone is one edit away from being a
     * description that means nothing. */
    d.idx_const = -1;
    const int64_t e[4] = { d0, d1, d2, d3 };
    for (int i = 0; i < 4 && e[i] != 0; ++i) d.shape[d.rank++] = e[i];
    return d;
}

/* An INDEX operand. `idx_max` is the EXCLUSIVE upper bound of the draw and 0 means "the leading
 * extent of the previous operand", which is the ABI's default and is right for a gather whose
 * source is the operand before it. `konst >= 0` makes every element that value instead of a draw.
 * i32 throughout: every index libr4d reads is a signed 32-bit one, and the two u32 spellings the
 * sampler accepts (`dt_i32` takes either) are the same bytes. */
RadOpdDesc opd_idx(int64_t d0, int64_t d1, int64_t idx_max, int64_t konst = -1,
                   uint32_t flags = 0) {
    RadOpdDesc d = opd((uint32_t)RAD_I32, d0, d1);
    d.fill      = RAD_FILL_INDEX;
    d.idx_max   = idx_max;
    d.idx_const = konst;
    d.flags     = flags;
    return d;
}

/* An OPTIONAL operand this description deliberately does not pass. One entry per schema operand
 * is still emitted -- that count is what catches a hook that has drifted -- so absence needs a
 * spelling, and a null RadTensor is what the ABI already means by it. */
RadOpdDesc opd_none() {
    RadOpdDesc d{};
    d.idx_const = -1;
    d.flags     = RAD_OPD_F_ABSENT;
    return d;
}

/* A domain the fill has to land in. `RAD_FILL_SIGMOID` is the ABI's only (0, 1) draw. */
RadOpdDesc in_unit(RadOpdDesc d) { d.fill = RAD_FILL_SIGMOID; return d; }

/* Hand out operand `operand` of a list, having checked the WHOLE list first.
 *
 * The whole list, because a geometry missing one parameter would otherwise describe its first few
 * operands and decline the rest, and a caller that allocated the first few is a caller that has
 * done work for a case it is about to skip. A non-positive extent means a parameter this
 * description needed was absent or zero, which is RAD_E_UNSUPPORTED -- the geometry does not carry
 * what the description needs -- and not RAD_E_SHAPE, which would read as a bug in the tool.
 *
 * An operand index outside the list IS a bug, in this file or in the schema it was written
 * against, and says so with a different code. */
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
 * `p` is the RESOLVED geometry, which is to say the geometry with its RANGES COLLAPSED: `M` here
 * is the band's bound, not the row count of any particular step (the same reading every shim in
 * this plugin takes, and the trap r4d_gdn.cpp's dflash_select shim spells out). That is the right
 * value for a description -- the buffers a tool allocates have to hold the band the kernel was
 * selected for -- and it is why nothing below tries to be cleverer about it.
 */
/* NOT r4d_pgeti_or(), and the difference is load-bearing: that reader takes RAD_P_INT only, which
 * is right for a LAUNCH -- by then the core has collapsed every range to the band's value -- and
 * wrong here, because this hook can be called with a geometry whose ranges are still ranges. A
 * RAD_P_RANGE reads as its HIGH bound: the buffers a tool allocates have to hold the whole band the
 * kernel was selected for, and the low end of a band fits inside them.
 *
 * That reading is rad_param_getdim's, in the ABI, and is shared by both plugins: a shape hook
 * cannot be written without it.
 *
 * A string spelling of an integer parameter is deliberately NOT accepted, where ref's reader takes
 * one: this plugin's shims read integers with r4d_geti_or, which is RAD_P_INT only, so describing a
 * geometry the launch would then refuse would turn a skip into a failure. rad_param_getdim takes
 * the strict reading for exactly that reason; ref layers its leniency on top. */
int64_t pi(const RadParam* p, int n_p, const char* k, int64_t dflt = 0) {
    return (int64_t)rad_param_getdim(p, n_p, k, (long long)dflt);
}

/* THE ABI's PARSER, and not an if-ladder of this file's own. rad_dtype_parse is static inline in
 * rad_types.h, which rad_abi.h includes, which this plugin compiles against, so it is reachable
 * from a plugin.
 *
 * Using it is what keeps the accepted spellings identical to libref's. A local subset -- one
 * missing u8, i16, i64, u32, bool, w4, w2 or mxfp4, say -- refuses here what libref accepts, and
 * rad-kbench reads that divergence as a libr4d hole rather than as a parser gap. rad_dtype_parse
 * also carries the `all_reduce` synonyms (the arch blocks declare `bf16`/`fp32` and
 * core/sample/sampler.cpp declares `f32`). */
uint32_t dt_of(const char* s) { return rad_dtype_parse(s); }

uint32_t dt_par(const RadParam* p, int n_p, const char* key) {
    return dt_of(r4d_pgets(p, n_p, key));
}

/* THE ACTIVATION WIDTH, with the fallback that matters. A `dtype` this table does not know is
 * usually a quantisation SCHEME rather than an element width -- "fp8a8", "w4a8" -- and bf16 is
 * what every activation in this project is when nothing says otherwise. Falling back to an invalid
 * dtype instead would size a zero-byte buffer and the case would be skipped for a reason that
 * looks like an allocation failure. `cast` uses the raw reader above, because there a dtype it
 * does not know is a pair this row genuinely cannot serve. */
uint32_t dt_act(const RadParam* p, int n_p, const char* key) {
    const uint32_t dt = dt_par(p, n_p, key);
    return dt == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;
}

/* ==================================================================== the sampler's params row
 *
 * `params` is an ARRAY OF RadSampleParams viewed through a 32-bit dtype -- one row per sampled
 * position, uploaded once a step (abi/rad_sample.h). Its EXTENT is exact: 96 bytes a row,
 * which is what r4d_sample.cpp's `params_ok` checks in words. Its CONTENT is a struct, and the
 * ABI's fill domains describe a buffer -- normals, a sigmoid, a gate cumsum, a triangular mask, an
 * index draw -- so there is no spelling for "fill this like a request".
 *
 * DESCRIBED AS f32, WHICH IS A CHOICE ABOUT CONTENT AND NOT ABOUT SIZE, and it is the same choice
 * ref_shapes.cpp makes for the same operand -- the two files have to agree here, because whichever
 * of them a tool asks, BOTH implementations are then handed those bytes. The struct is eleven
 * floats and nine int32s. Drawn as f32 normals every float field -- temp, top_p, min_p,
 * typical_p, the penalties, the two XTC values -- lands in a plausible range, so a third to a half
 * of the rows fall inside the interval that makes their stage do work and the rest are the
 * identity: which is the mix a real batch has.
 *
 * WHAT IT CANNOT REACH, said plainly rather than left for someone to discover from a PASS. The
 * INTEGER fields come out as the bit patterns of those same normals, so `hist_off`, `hist_len` and
 * `mask_row` are enormous -- and every stage clamps them, which is why this is safe on a device
 * and also why it is weak:
 *
 *     sample_penalties   ho is clamped to hmax, hl to hmax - ho, and `hl <= 0` returns
 *     sample_dry         the same clamp, then `hl <= allowed` returns
 *     sample_mask        `mask_row >= mask_rows` returns before a word of the plane is read
 *
 * `flags` is a bit pattern too, so roughly half the rows carry RAD_SP_GREEDY, and every candidate
 * stage -- top-k, the narrowing stages, the merge, the pick -- treats those as having no
 * candidates. Both implementations do, so the comparison holds; it covers the other half.
 *
 * A pass on those three says the plumbing agrees, not that the arithmetic does. The alternative --
 * a word-typed draw, or the all-zero row -- is worse in both directions: it makes every float
 * field a denormal or a NaN and `1 / temp` an infinity, or it makes every stage the identity.
 *
 * u32 IS NEVER USED for this or any other word-typed operand here, and that is not taste:
 * tools/opshapes.cpp's `rad_widen` and `rad_narrow` have no RAD_U32 case, so a u32 operand is
 * FILLED as zeroes and COMPARED as zeroes -- a vacuous pass, which is worse than a skip. i32 is
 * the same four bytes and is read. */
RadOpdDesc opd_sample_params(int64_t M) {
    return opd((uint32_t)RAD_F32, M, (int64_t)(sizeof(RadSampleParams) / 4));
}

}  /* namespace */

/* ==========================================================================================
 * The small model ops (shims in r4d_model.cpp).
 *
 * Every one of these takes `n` as the row width and reads its row COUNT off the operand, so the
 * band's `M` is what a description has to size for. The gain of a norm is the one operand that is
 * not bf16: every architecture plugin in this tree declares its norm gains RAD_F32 -- a few
 * thousand elements against gigabytes of weights, and a norm is where a rounded gain shows -- and
 * the shim reads the width off the operand rather than assuming it.
 * ========================================================================================= */

extern "C" int r4d_shape_embed_lookup(const RadParam* p, int n_p, int operand,
                                      RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_embd");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t vo = pi(p, n_p, "vocab_offset", 0);
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* A TOKEN ID INDEXES THE TABLE, and the table is the operand AFTER it, so the ABI's
         * default -- the previous operand's leading extent -- is the wrong one here and n_vocab is
         * said outright. A draw from anywhere else tests the row's negative-id path and calls it
         * an embedding.
         *
         * Under a vocab shard the bound is vo + nv, because ids are GLOBAL: drawing only within
         * the slice never exercises the zero-row path the following all_reduce depends on. */
        opd_idx(M, 0, vo + nv),
        opd(D, nv, ne),
        opd(D, M, ne),
    };
    R4D_PICK(l);
}

/* The quantised gather: libref's description (ref_embed_lookup_q_shape), an E4M3 table and a
 * one-element scale in the activation dtype. `ahead` is not passed: it is ids for a reader that
 * fetches rows from a file, and the kernel does not read it. */
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
        /* FOUR TIMES THE ROWS THE GATHER PRODUCES, so the gather is not the identity: a source
         * exactly M rows tall makes idx[i] == i a plausible answer, and the identity is the one
         * case that passes while the kernel is broken. */
        opd(D, 4 * M, n),
        opd_idx(M, 0, 4 * M),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* The byte-row pair. `n` is a count of BYTES here, and the registry constrains it to a multiple
 * of 4 so the vectorised copy is the only path -- a description that drew an odd `n` would be
 * describing a case the kernel correctly refuses. */
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

/* And the mirror. The destination is four times the height for the same reason, plus one of its
 * own: a scatter onto exactly M rows under M DISTINCT indices is a permutation, so a kernel that
 * ignored `idx` would still touch every row and could still pass. Wider than the write, and the
 * rows nobody names have to come back unchanged. */
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
    /* `from` and `to` are separate keys because the whole point of the op is that the two ends
     * differ. This row is compiled for bf16 -> bf16 only -- it is a strided row copy -- and the
     * shim refuses anything else on dtype, so a geometry that names another pair is declined here
     * rather than described into a refusal. */
    const uint32_t f = dt_par(p, n_p, "from"), t = dt_par(p, n_p, "to");
    if (f != (uint32_t)RAD_BF16 || t != (uint32_t)RAD_BF16) return RAD_E_UNSUPPORTED;
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

/* ---- the vision tower's (r4d_vit_bf16.hip). Every optional operand is PASSED: the bias of the
 * norm and the residual of the GEMM are the same code with a branch taken, so describing them
 * covers both forms. */
extern "C" int r4d_shape_layernorm(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, n), opd(D, n), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_grid_embed(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), side = pi(p, n_p, "side");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* Coordinates drawn over the table's side: some rows sit past their grid's last row and take
     * the clamped taps as well as the interior ones. */
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, side * side, n), opd_idx(4, M, side) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gemm_nt_bias(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), N = pi(p, n_p, "N"), K = pi(p, n_p, "K");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, K), opd(D, N, K), opd(D, N), opd(D, M, N), opd(D, M, N) };
    R4D_PICK(l);
}

/* ---- the gated residual (r4d_hc_bf16.hip's three shims). `inject` and `group` DEFAULT ON, so a
 * query naming neither describes what every trunk connection runs; 0 for either describes one of
 * the two exceptions -- the final mixer, and a read whose consumer wants bf16. */
extern "C" int r4d_shape_hc_enter(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M, hc * n) };
    R4D_PICK(l);
}

/* The MTP head's entry. Both gains are F32, as hc_read's is, and the two fc matrices are the
 * activation dtype -- the checkpoint's `mtp.fc_hidden` / `mtp.fc_embedding`, square and bf16. */
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
    /* `inj` is a gain in (0, 2) -- twice a sigmoid -- and a normal draw would make hc_write a
     * different test: the point of the op is that the four gains are small and data-dependent. */
    const RadOpdDesc l[] = { opd(D, M, n), in_unit(opd(D, M, hc)), opd(D, M, hc * n) };
    R4D_PICK(l);
}

/* hc_write's operands with the all-reduce in front: the same three, `y` now the message as well. */
extern "C" int r4d_shape_ar_hc_write(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n"), hc = pi(p, n_p, "hc");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), in_unit(opd(D, M, hc)), opd(D, M, hc * n) };
    R4D_PICK(l);
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

/* add and mul share an operand list. THE BROADCAST FORM IS NOT DESCRIBED: `b` carrying exactly n
 * elements against an output of M rows broadcasts along rows, and that is a second geometry of the
 * same op -- one description can only be one of them, and the elementwise form is the one every
 * caller in this tree issues. */
extern "C" int r4d_shape_binary(const RadParam* p, int n_p, int operand,
                                RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M, n), opd(D, M, n) };
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
    /* The gate is the FIRST half of a [M, 2n] row, which is the order rad-convert fuses gate_proj
     * and up_proj in. */
    const RadOpdDesc l[] = { opd(D, M, 2 * n), opd(D, M, n) };
    R4D_PICK(l);
}

extern "C" int r4d_shape_rope(const RadParam* p, int n_p, int operand,
                              RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t nq  = pi(p, n_p, "n_head");
    const int64_t nkv = pi(p, n_p, "n_head_kv");
    /* The fused projection row: n_head queries, then n_head_kv each of k and v. The V heads are
     * untouched, which is the whole reason this op takes the fused row rather than two tensors --
     * and why the row width is not just (n_head + n_head_kv) * head_dim. `n_head_kv` 0 is the
     * legal "q only" form the gated attention block issues, so it is not required to be positive
     * and the width is what has to be. */
    const int64_t row = (nq + 2 * nkv) * hd;
    /* A multi-component mode with its sections takes a position per component, component-major:
     * three for the interleaved form, one a section otherwise. */
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
        /* A POSITION IS NOT A LOOKUP HERE: this kernel computes sincos from the value, so any
         * non-negative integer is legal and none of them is out of range. M keeps the draw in the
         * range a step actually presents. */
        multi ? opd_idx(nc, M, M) : opd_idx(M, 0, M),
    };
    R4D_PICK(l);
}

/* kv_store, both rows. The cache dtype is the one difference between them and it is a PARAMETER,
 * so one description serves both: `kv_dtype` is what the constraint table selects on.
 *
 * The paged layout is stated once for the whole project in libref/ref_ops.h and checked by
 * this plugin's shim:  kv_cache[n_blocks, kv_heads, block_size, 2 * head_dim], K at [.., d] and V
 * at [.., head_dim + d]. The block COUNT is not a parameter of anything -- it is a property of the
 * deployment's cache -- so it is this description's to choose, and any cache big enough for the
 * slots to land in will do. */
extern "C" int r4d_shape_kv_store(const RadParam* p, int n_p, int operand,
                                  RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t kvh = pi(p, n_p, "n_head_kv");
    const int64_t bs  = pi(p, n_p, "block_size");
    const uint32_t KV = dt_act(p, n_p, "kv_dtype");
    if (M <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    /* Two blocks of slack, so the slot mapping is not a dense prefix: a scatter that only ever
     * writes block 0 agrees with anything at all about the block index. */
    const int64_t nb = (M + bs - 1) / bs + 2;
    const RadOpdDesc l[] = {
        /* Spelled [tokens, kv_heads, head_dim]: the shim folds every trailing extent into one row
         * pitch, so the flattened [M, kv_heads * head_dim] an architecture may declare is the same
         * bytes and the same pitch -- and this is the spelling ref's description of the op uses,
         * which is the one that has to match when the two are compared against each other. */
        opd((uint32_t)RAD_BF16, M, kvh, hd),
        opd((uint32_t)RAD_BF16, M, kvh, hd),
        /* A SCATTER DESTINATION, so no slot may repeat -- two tokens on one slot make the store a
         * race and the two implementations resolve it differently. It indexes POSITIONS, which is
         * n_blocks * block_size and is no operand's leading extent. */
        opd_idx(M, 0, nb * bs, -1, RAD_OPD_F_IDX_UNIQUE),
        opd(KV, nb, kvh, bs, 2 * hd),
    };
    R4D_PICK(l);
}

/* logits_gemm at bf16. The fp8 row of the same op is deliberately not described; see the file
 * header on why its declared shape and its stored plane differ. */
extern "C" int r4d_shape_logits_gemm(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const int64_t ne = pi(p, n_p, "n_embd");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, ne),
        opd((uint32_t)RAD_BF16, nv, ne),
        /* F32 LOGITS OUT, and checked by the shim rather than assumed: dispatching this op to the
         * bf16 GEMM instead writes bf16 pairs into an f32 plane -- every logit wrong, the
         * magnitudes roughly right, fluent nonsense out of the sampler. */
        opd((uint32_t)RAD_F32, M, nv),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * The decomposed sampler chain (shims in r4d_sample.cpp). Eleven of the twelve; `sample_merge_topk`
 * is declined and the argument is with the others in the file header.
 *
 * Read the `opd_sample_params` note above before trusting a PASS from any of these: the row's float
 * fields are real and its INTEGER fields are the bit patterns of the same normals, so
 * `sample_penalties`, `sample_dry` and `sample_mask` take their "nothing to do" branch and a pass
 * on those three says the plumbing agrees rather than the arithmetic.
 * ========================================================================================= */

/* logits + params, and one auxiliary plane that differs per stage. */
extern "C" int r4d_shape_sample_penalties(const RadParam* p, int n_p, int operand,
                                          RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    /* THE HISTORY WINDOW IS THIS DESCRIPTION'S CHOICE. `history` is [M, max_hist] and the shim
     * takes max_hist from the buffer (numel / M); no parameter names it, because a deployment's
     * history plane is sized by its longest live sequence. 64 is a plausible window, and the
     * kernel clamps `hist_off` and `hist_len` to it before reading a word -- which is what makes
     * the params row above safe and this stage's comparison weak. */
    const int64_t hist = 64;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        /* Token ids, so they are drawn from the vocabulary and not from the plane's own extent. */
        opd_idx(M, hist, nv),
    };
    R4D_PICK(l);
}

/* sample_dry: the penalties' three operands and the optional breaker plane, which is not passed --
 * absent means no token is exempt, and the params row above reaches the row's early out before a
 * word of either plane is read. The same reading as ref's description of this op. */
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
    /* The plane is [rows, ceil(n_vocab/32)] and is SHORTER than M in a real batch -- a constrained
     * sequence and an unconstrained one share it and `mask_row` is -1 for the second. One row per
     * position is the fully-constrained case and the only one a description can pick. */
    const int64_t words = (nv + 31) / 32;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        /* A SET BIT MEANS ALLOWED: word w bit b covers token 32w + b, and a cleared bit sets
         * -infinity. i32 and not u32 for the reason opd_sample_params gives -- a u32 operand is
         * filled and compared as zeroes -- and a normal draw narrowed to i32 is a sparse mask
         * rather than a plane that forbids everything. The kernel does not reach it at this params
         * row anyway: `mask_row` is enormous and the row returns first. */
        opd((uint32_t)RAD_I32, M, words),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_sample_topk(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    /* THE CANDIDATE WIDTH IS THE BUFFER'S, not the request's: it is the deployment bound the arena
     * was sized for and every later stage reads it. No parameter carries it on this op -- `n_cand`
     * is the narrowing stages' key -- so 64 is chosen here, well inside the 4096 the LDS bitonic
     * sort covers. */
    const int64_t n_cand = 64;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        opd_sample_params(M),
        opd((uint32_t)RAD_I32, M, n_cand),
        opd((uint32_t)RAD_F32, M, n_cand),   /* cand_val -- LOGITS, not probabilities */
        /* `pairs` is the VOCAB-PARALLEL path, and it is described rather than skipped because it is
         * an OUTPUT: nothing has to fill it, comparing it is what checks the packing -- (u32 global
         * id, f32 logit) with a hole marker for a slot the row could not fill -- and `vocab_off` is
         * zero on a single-rank graph, which makes the ids the local ones and the check exactly as
         * strict. Same reading as ref's description of this operand. */
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

/* top_p, min_p, typical and xtc: the same three operands, narrowed in place. */
extern "C" int r4d_shape_sample_narrow(const RadParam* p, int n_p, int operand,
                                       RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nc = pi(p, n_p, "n_cand");
    const RadOpdDesc l[] = {
        /* EVERY DRAWN ID IS >= 0, so every candidate arrives LIVE -- a stage that dropped nothing
         * because the set was already empty is the vacuous case. The range is the candidate width
         * and not the vocabulary: nothing DEREFERENCES an id here, these kernels only reorder and
         * drop, and `n_vocab` is not a parameter of a narrowing stage anyway. Stated rather than
         * left at the ABI's default, which for operand ZERO has no previous operand to take a
         * leading extent from. */
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

/* `sample_merge_topk` IS DECLINED, and ref declines it too -- deliberately the same answer, because
 * a description here would supply the very bytes ref refused to.
 *
 * `gathered` is world_size * n_cand (u32 GLOBAL token id, f32 logit) pairs packed into ONE
 * WORD-TYPED buffer: both plugins require i32 or u32 for it, because the two halves have different
 * types and a buffer has one dtype. The value lane is therefore read by BIT PATTERN, and no fill
 * domain in this ABI puts a sensible float there -- an integer draw makes every logit a DENORMAL,
 * and a device that flushes denormals against a host reference that does not then order the
 * candidates differently. That is not hypothetical: it is the failure r4d_selftest's two-rank
 * all-gather row exists to catch ("a token id is a small integer, and its bit pattern read as f32
 * is a denormal"). And `world_size` is not in this op's parameter list, so the gathered width would
 * have to be invented on top of that. Either reason is sufficient.
 *
 * `row_topk_merge` is the same shape of operand and is NOT declined, because its plane is f32:
 * there the value lane is a real float and the ID is the bit pattern, which is the way round that
 * works. */

/* The FUSED sampler, which is a different op from the eleven above and not their composition
 * (r4d_fp8.cpp). Its three outputs are what a rejection-sampling verify needs from one pass. */
extern "C" int r4d_shape_sample_chain(const RadParam* p, int n_p, int operand,
                                      RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t nv = pi(p, n_p, "n_vocab");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F32, M, nv),
        /* The drafted token per position. Passed rather than left absent: `p_query` -- the accept
         * probability -- is the output this op exists for, and with `query` absent the kernel
         * computes two of its three answers. A token id, so it is drawn from the vocabulary. */
        opd_idx(M, 0, nv),
        /* TWO UNIFORMS A POSITION, one for the draw and one for the exclusive draw, and they have
         * to be IN [0, 1): the inverse-CDF walk falls through to its last live candidate for
         * anything above the mass, so a normal draw would make every position take the fallback
         * and the case would check the fallback. RAD_FILL_SIGMOID is the ABI's (0, 1) domain. */
        in_unit(opd((uint32_t)RAD_F32, 2 * M)),
        /* [M] x R4DSpecDraw = {u32 tok, u32 tok_excl, f32 p_query}: THREE dwords a position. The
         * shim only checks numel >= M, which is the count of STRUCTS and not of words, so a
         * description that said [M] would leave the kernel writing two thirds of its output past
         * the end of the buffer. i32 rather than u32, for the reason opd_sample_params gives: a u32
         * operand is compared as zeroes and every case would pass vacuously. */
        opd((uint32_t)RAD_I32, M, 3),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Attention (shims in r4d_attn.cpp).
 * ========================================================================================= */

/* attn_paged, all six rows. `q_len` is the ranged parameter -- one query row at decode, thousands
 * in a prefill chunk -- and the KV cache's dtype comes from `kv_dtype`, which is what separates the
 * fp8kv rows from the bf16kv ones, so one description serves the family.
 *
 * ONE SEQUENCE WHOSE CONTEXT IS EXACTLY ITS QUERY: a causal attention over its own prefix, which is
 * what a prefill chunk runs and what a decode step's last row is. The block count follows from
 * that, and the alternative -- a context longer than the query -- would need a number no parameter
 * carries. */
extern "C" int r4d_shape_attn_paged(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t q_len = pi(p, n_p, "q_len");
    const int64_t hd    = pi(p, n_p, "head_dim");
    const int64_t gqa   = pi(p, n_p, "gqa");
    const int64_t bs    = pi(p, n_p, "block_size");
    /* n_head is OPTIONAL on this op -- it is there because the split-KV scratch hook needs it at
     * declare -- and without it there is no head count. tools/opshapes.h falls back to `gqa`,
     * which makes one kv head; that is a different model and this declines instead. */
    const int64_t nh    = pi(p, n_p, "n_head");
    if (q_len <= 0 || hd <= 0 || gqa <= 0 || bs <= 0 || nh <= 0 || nh % gqa)
        return RAD_E_UNSUPPORTED;
    const int64_t kvh = nh / gqa;
    /* THE CONTEXT DEPTH. Defaults to the query length -- one sequence attending its own prefix,
     * which is the prefill chunk -- and `ctx` overrides it. No op declares `ctx`: the kernel
     * reads its context off `seqused` and the block table, so only a DESCRIPTION of the operands
     * needs to be told, and a harness sweeping depth is the only thing that ever sets it. Must be
     * at least q_len or the causal mask reads a prefix that is not there. */
    int64_t ctx = pi(p, n_p, "ctx", 0);
    if (ctx < q_len) ctx = q_len;
    const int64_t nblk = (ctx + bs - 1) / bs;
    const uint32_t Q  = dt_act(p, n_p, "q_dtype");
    const uint32_t KV = dt_act(p, n_p, "kv_dtype");
    const RadOpdDesc l[] = {
        opd(Q, q_len, nh, hd),
        opd(KV, nblk, kvh, bs, 2 * hd),
        /* A block-table entry indexes n_blocks and is a READ-side index, so it may repeat. */
        opd_idx(1, nblk, nblk),
        /* NOT A DRAW. A context shorter than the query makes a causal attention read a prefix that
         * does not exist, and both implementations would agree about the garbage while neither
         * computed what the model asks for. */
        opd_idx(1, 0, 0, ctx),
        /* The descales are absent and that means 1.0, which is what vLLM defaults them to and what
         * the fp8kv kernels assume: they are ONE number per (sequence, head) on the read side and
         * cannot be derived from the tokens being written. */
        opd_none(),
        opd_none(),
        /* cu_seqlens absent = one q_len for the whole batch, which is exact at one sequence. */
        opd_none(),
        opd(Q, q_len, nh, hd),
    };
    R4D_PICK(l);
}

/* attn_paged_gate_quant -- attn_paged's operands, then gate_quant_fp8's gate, codes and scales
 * over its (token, head) rows: the gate described flat as gate_quant_fp8 describes it, one row a
 * (token, head) pair of `head_dim` columns. */
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

/* qk_norm_rope_gate -- the gated-attention QKV preamble (shim in r4d_quant.cpp).
 *
 * NOT docs/OPS.md's `qk_norm_rope`: it splits a [q|gate] projection per head, writes q, k and the
 * gate to three destinations, and takes a PRECOMPUTED cos/sin table rather than a theta. The table
 * is why `rope_table` exists at all. */
extern "C" int r4d_shape_qk_norm_rope_gate(const RadParam* p, int n_p, int operand,
                                           RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t hd  = pi(p, n_p, "head_dim");
    const int64_t nq  = pi(p, n_p, "n_head");
    const int64_t nkv = pi(p, n_p, "n_head_kv");
    const int64_t rot = pi(p, n_p, "rot");
    const uint32_t Q  = dt_act(p, n_p, "q_dtype");
    /* The two widths the kernel reads off the operands rather than off a parameter: a cos/sin
     * table built in f32 and a positions vector in int64 are what the tensors SAY they are. The
     * optional `cs_f32` / `pos_i64` keys are the caller's declaration of the same thing, so they
     * decide what to allocate here. The arch that issues this passes cs_f32 = 1. */
    const uint32_t CS = pi(p, n_p, "cs_f32", 0) ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16;
    const RadOpdDesc l[] = {
        /* THE PROJECTION WHOLE, q and gate interleaved per head -- the kernel splits it itself,
         * which is what the fusion replaces. Declared flat: the shim folds every trailing extent
         * into one row pitch, so [M, n_head * 2 * head_dim] and [M, n_head, 2 * head_dim] are the
         * same bytes and the same pitch. */
        /* ABSENT IN THE K-ONLY FORM (`n_head` 0), where there is no gate to split and every head
         * the op serves is read off `k`. A zero-width operand would be a different statement --
         * "a tensor of no columns" rather than "no tensor" -- and the shim checks for the second. */
        nq ? opd(Q, M, nq * 2 * hd) : opd_none(),
        opd(Q, M, nkv * hd),
        /* The [max_ctx, rot] plane, at one row per position of this band. A position indexes a ROW
         * of it, so the table's row count is the draw's bound -- and unlike `rope`, this one is a
         * LOOKUP: a position past the end reads another layer's angles or nothing at all. */
        opd(CS, M, rot),
        opd_idx(M, 0, M),
        /* The QK norm gains, one per head channel. f32 in every architecture plugin here where
         * vLLM's are bf16, and the shim requires the two to agree because one flag serves both. */
        nq ? opd((uint32_t)RAD_F32, hd) : opd_none(),
        opd((uint32_t)RAD_F32, hd),
        nq ? opd(Q, M, nq * hd) : opd_none(),
        opd(Q, M, nkv * hd),
        /* `gate_out` absent, which is what the fp8 gated attention block passes: `gate_quant_fp8`
         * reads the gate at its column offset inside the projection, so a contiguous copy would be
         * q_dim of bf16 a layer written for nothing. */
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

/* rope_table -- the cos/sin plane the fused prologue reads (shim in r4d_quant.cpp). */
extern "C" int r4d_shape_rope_table(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t M   = pi(p, n_p, "M");
    const int64_t rot = pi(p, n_p, "rot");
    const uint32_t D  = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* ONLY THE ROWS `positions` NAMES ARE WRITTEN, at each position's OWN row of the plane, so
         * the draw's bound is the plane's row count and a position outside it would scribble past
         * the table. The real table is [max_ctx, rot]; M rows is the smallest one that holds every
         * row this band can name. */
        opd_idx(M, 0, M),
        opd(D, M, rot),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Gated delta net -- the one row of the family whose widths are all parameters. The other five are
 * declined; see the file header on the head counts. (`gdn_gated_norm_had_quant_i8` is describable
 * for the same reason and is with the fusions below, where its siblings are.)
 * ========================================================================================= */

extern "C" int r4d_shape_gdn_gated_rmsnorm(const RadParam* p, int n_p, int operand,
                                           RadOpdDesc* out) {

    /* ONE ROW IS ONE (token, head): `M` is already tokens x heads and `channels` is head_v, which
     * is what makes this op describable where the rest of the family is not. */
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
 * Collectives (shims in r4d_ar.cpp).
 *
 * These need a real peer over IPC, so a single-process tool gets RAD_E_STATE from init long before
 * it gets to the buffers. Described anyway: the description is not what decides that, and a
 * two-process harness needs it the moment one exists.
 * ========================================================================================= */

extern "C" int r4d_shape_all_reduce(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    /* `numel` is the LARGEST message this instance will ever carry -- init sizes the peer scratch
     * from it once and the launch reads the real count off the tensor -- so it is exactly the
     * extent a buffer has to have. */
    const int64_t numel = pi(p, n_p, "numel");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, numel),
        /* The optional SECOND DESTINATION. Absent means in place, which every row of the op
         * serves; described so a caller that passes one is checked against the same extent. */
        opd(D, numel),
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_all_gather(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t numel = pi(p, n_p, "numel");
    const int64_t world = pi(p, n_p, "world_size");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* `row` absent means the whole message is one row and `y` is the plain concatenation, rank r's
     * slice at r * numel. The row-INTERLEAVED form is the same extent at a different placement, so
     * this description covers both. */
    const RadOpdDesc l[] = { opd(D, numel), opd(D, world * numel) };
    R4D_PICK(l);
}

/* The two fused all-reduce + norm folds. Same operands in the same order as the standalone norms
 * they replace, which is the property that makes them substitutable -- so these two descriptions
 * are the standalone ones with `x` reread as the message. */

extern "C" int r4d_shape_ar_ln_had_quant_i8(const RadParam* p, int n_p, int operand,
                                            RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, n),
        opd((uint32_t)RAD_BF16, M, n),      /* residual: the fused_add_rms_norm form */
        /* THE GAIN IS BF16 ON THE INT8 SIDE, unlike the fp8 twin below whose shim reads the width
         * off the operand: r4d_ar_ln_had_quant_i8 takes `wgain` as __hip_bfloat16 and has no flag
         * for anything else. An f32 gain here would be read as two bf16 halves of the wrong
         * number, which is the silent kind of wrong. */
        opd((uint32_t)RAD_BF16, n),
        opd((uint32_t)RAD_I8, M, n),
        /* ONE SCALE PER ROW -- the rotation is what makes that viable, and it carries the
         * 1/sqrt(had) normalisation as well. Not a per-group plane: this family has no group. */
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
    /* The block is a property of the checkpoint's scale grid and not a knob: a geometry that
     * declares a different one is asking for a format the GEMM's fold does not read, and the shim
     * refuses it by name rather than substituting 128. */
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_BF16, M, n),      /* INOUT: the message AND the input */
        opd((uint32_t)RAD_BF16, M, n),
        opd((uint32_t)RAD_F32,  n),         /* f32 or bf16; the shim reads the operand's own */
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
        opd((uint32_t)RAD_BF16, M, n),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * GEMM, dequant and the row top-R (shims in r4d_gemm.cpp and r4d_fp8.cpp).
 * ========================================================================================= */

extern "C" int r4d_shape_gemm_nt(const RadParam* p, int n_p, int operand,
                                 RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, K), opd(D, N, K), opd(D, M, N) };
    R4D_PICK(l);
}

/* gemm_nt_q, THE BLOCK-SCALED FP8 FAMILY ONLY. The other five grids are declined; the file header
 * says why, and it is the same argument tools/opshapes.h makes from the other side.
 *
 * The grid here is stated by the CHECKPOINT -- E4M3 [N][K] against a bf16
 * [ceil(N/128)][ceil(K/128)] plane, with `group` carrying the 128 -- so there is nothing left to
 * guess. The fragment-ordered arrangement the narrow and tiled rows read is a permutation of the
 * same bytes (frag_bytes(N, K) == N * K), so one description sizes every row of the family. */
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
        /* a, a_scale. The matvec reads the bf16 activation DIRECTLY -- there is no quantise launch
         * in front of it -- and its shim refuses an activation scale outright, because running
         * anyway would silently drop a scale the caller believes was applied. */
        a8 ? opd((uint32_t)RAD_F8E4M3, M, K) : opd((uint32_t)RAD_BF16, M, K),
        a8 ? opd((uint32_t)RAD_F32, M, kb)   : opd_none(),
        opd((uint32_t)RAD_F8E4M3, N, K),     /* the checkpoint's own bytes */
        opd((uint32_t)RAD_BF16, nb, kb),     /* weight_scale_inv, one per 128x128 tile */
        opd((uint32_t)RAD_BF16, M, N),
        /* `a_sum` belongs to the asymmetric integer grids and `b_ref` to mxfp4. An fp8 grid is
         * symmetric by construction and has neither. */
        opd_none(),
        opd_none(),
    };
    R4D_PICK(l);
}

/* The same GEMM with the SwiGLU and the quantiser folded into its epilogue. `N` is the WEIGHT's
 * rows -- twice the fused width, gate then up -- so the quantiser's outputs are N/2 wide, and that
 * factor is the whole reason this op has a schema of its own rather than a flag on gemm_nt_q. */
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

/* dequant_w4 -- the packed 4-bit weight back to row-major bf16 (shim in r4d_gemm.cpp).
 *
 * The one weight in this plugin whose packed stream a PUBLIC dtype sizes exactly: RAD_I4 is four
 * bits an element, so [N, K] is the N*K/2 bytes r4d_rad_layout_w4 declares. The scale plane is one
 * (f16 scale, f16 -(1024+zero)) dword per (row, group), stored tile-major as [N/16][K/group][16] --
 * which is N * (K/group) dwords however it is grouped, and the kernel reads it at
 * [(t * ng + k/group) * 16 + r].
 *
 * NEITHER STREAM'S CONTENT IS EXPRESSIBLE and that is worth knowing before reading a result: a
 * packed nibble plane and a scale dword are bit layouts, not numbers, so the caller's fill leaves
 * them zeroed or filled with small integers. The extents are exact, which is what a timing harness
 * needs; a correctness comparison would be comparing two dequantisations of the same nonsense. */
extern "C" int r4d_shape_dequant_w4(const RadParam* p, int n_p, int operand,
                                    RadOpdDesc* out) {

    const int64_t N = pi(p, n_p, "N");
    const int64_t K = pi(p, n_p, "K");
    const int64_t g = pi(p, n_p, "group", R4D_W4_GROUP);
    if (N <= 0 || K <= 0 || g <= 0 || N % R4D_NTILE || K % g) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_I4, N, K),
        opd((uint32_t)RAD_U32, N / R4D_NTILE, (K / g) * R4D_NTILE),
        opd((uint32_t)RAD_BF16, N, K),
    };
    R4D_PICK(l);
}

/* ---- the routed mixture of experts (shims in r4d_moe.hip) --------------------------------------
 *
 * `expert_offset` IS NOT A FREE DRAW and no fill domain in the ABI says so: it is moe_scatter's
 * output, a non-decreasing prefix array ending at the live row count, and the expert that owns a
 * sorted row is found by searching it. Drawn at random it is a malformed histogram, so two
 * implementations that search it differently can pick different experts and disagree without
 * either being wrong. The EXTENTS are exactly right, which is what a benchmark needs; a
 * correctness comparison over an op that reads it proves less than it looks like. libref's
 * description of the same operand says the same thing, which is the point -- the two plugins have
 * to agree about what a case means before they can be compared. */
extern "C" int r4d_shape_scale_rows(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = pi(p, n_p, "M"), n = pi(p, n_p, "n");
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = { opd(D, M, n), opd(D, M), opd(D, M, n), opd(D, M, n) };
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

/* The two shape functions concatenated, with `expert_ids` named ONCE -- it is an output here, not
 * an input, which is the only difference between this list and the pair's two. */
extern "C" int r4d_shape_router_topk_scatter(const RadParam* p, int n_p, int operand,
                                             RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    if (ne <= 0 || ne > 512 || k <= 0 || k > ne) return RAD_E_UNSUPPORTED;
    /* The sort's three outputs are sized by THIS RANK'S SLICE, not by the router. */
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
extern "C" int r4d_shape_moe_scatter(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = pi(p, n_p, "M");
    const int64_t ne = pi(p, n_p, "n_expert");
    const int64_t k  = pi(p, n_p, "top_k");
    if (ne <= 0 || ne > 256 || k <= 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        /* Expert ids index EXPERTS, which is no operand's dim 0, so the range is stated. */
        opd_idx(M, k, ne, -1),
        opd((uint32_t)RAD_I32, M * k),
        opd((uint32_t)RAD_I32, ne + 1),
        opd((uint32_t)RAD_I32, ne),
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
    /* `a_order` decides the activation's row count: one per token, or one per (token, slot)
     * already in the scatter's order. A description that took the wrong reading would build the
     * operand at the other extent, and nothing about the case would look like a shape error. */
    const char* ord = rad_param_gets(p, n_p, "a_order", nullptr);
    const int64_t AR = (ord && !std::strcmp(ord, "sorted")) ? M * k : M;
    const RadOpdDesc l[] = {
        opd((uint32_t)RAD_F8E4M3, AR, K),                 /* a -- see `a_order` */
        opd((uint32_t)RAD_F32, AR, (K + g - 1) / g),      /* a_scale */
        /* THE WEIGHT OPERANDS ARE TABLES at issue and are described here as the STACKED plane they
         * would be if they were one tensor. A fixture has nowhere to put a pointer array, and the
         * kernel reads both forms (r4d_moe.hip, wtab_at) precisely so that a described case runs
         * against the same bytes the engine would hand it. */
        opd((uint32_t)RAD_F8E4M3, ne0, N0, K0),           /* w -- the even experts */
        opd((uint32_t)RAD_BF16, ne0, (N0 + g - 1) / g, (K0 + g - 1) / g),   /* w_scale */
        opd_idx(M * k, 0, M * k, -1),                     /* sorted_tok */
        opd_idx(ne + 1, 0, M * k + 1, -1),                /* expert_offset -- see the note above */
        opd((uint32_t)RAD_F8E4M3, ne1, N1, K1),           /* w_odd -- the odd experts */
        opd((uint32_t)RAD_BF16, ne1, (N1 + g - 1) / g, (K1 + g - 1) / g),   /* w_odd_scale */
        opd(D, M * k, N),                                 /* y -- in SORTED order */
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
        /* DISTINCT, and it has to be: gather rebuilds the INVERSE permutation of sorted_tok, so a
         * repeated entry makes two sorted rows claim one (token, slot) and the last writer wins --
         * by whichever workgroup lands last, which is a measurement of the scheduler. */
        opd_idx(M * k, 0, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE),
        /* shared and shared_gate -- optional, described anyway so a cold caller allocates them
         * and exercises the folded form. */
        opd(D, M, n),
        opd(D, M),
        opd(D, M, n),
    };
    R4D_PICK(l);
}

/* row_topk -- exact per-row top-R of a wide bf16 matrix (shim in r4d_gemm.cpp).
 *
 * R is capped at 32 by the entry point (stage 2's PT, "a real limit, not a tuning choice") even
 * though libr4d's own row says 1024; the constraint is transcribed as written and the shim refuses
 * above 32, so a band naming a wider R is declined here rather than described into a refusal. */
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
        /* `pairs` is the VOCAB-PARALLEL form and both halves of it are optional -- absent, with
         * `vocab_off` zero, is the single-rank launch bit for bit. It is described anyway, and for
         * the reason ref's description of the same operand gives: it is an OUTPUT, so nothing has
         * to fill it, and comparing it is what checks the packing convention -- an int32 id
         * memcpy'd into the first float slot beside its value, with id -1 for a slot the row could
         * not fill. f32 for exactly that reason: the ID is carried by BIT PATTERN, and a
         * word-typed plane could not hold the value. */
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
        /* [M, world_size * R, 2] f32 pairs with rank r's at [r*R, (r+1)*R), an int32 id in the
         * first slot of each. THE PLANE IS f32 AND THAT IS WHY THIS ONE IS DESCRIBABLE where
         * `sample_merge_topk`'s is not: a normal fill puts a REAL float in the value lane, which is
         * the lane the merge orders on, and the id lane is that same float's bit pattern read as an
         * int32. A negative float is then a negative id, which the kernel drops as a pad
         * (`if (id < 0) continue`), and a positive one is a large but perfectly legal id that is
         * only ever compared against the others and copied out. So about half the candidates are
         * pads and the rest are ordered exactly -- ties are impossible between distinct normals --
         * which exercises the pad rule and the ordering in one case. */
        opd((uint32_t)RAD_F32, M, W * R, 2),
        opd((uint32_t)RAD_I32, M, R),
        opd((uint32_t)RAD_F32, M, R),
    };
    R4D_PICK(l);
}

/* ==========================================================================================
 * Quantisation, the norms and the fusions (shims in r4d_quant.cpp and r4d_fp8.cpp).
 *
 * TWO FAMILIES, AND THE SCALE IS WHAT SEPARATES THEM. The int8 side is ROTATED and carries ONE f32
 * scale per row -- the block Walsh-Hadamard is what makes a single scale per row viable at all --
 * and its `group` parameter is the ROTATION WIDTH, not a scale grid.
 * The fp8 side has no rotation and carries an f32 scale per (row, 128 columns), because 128
 * is the block the CHECKPOINT's weight scales are taken over. A description that swapped the two
 * would be off by a factor of n/128 on the scale plane and would still look plausible.
 * ========================================================================================= */

extern "C" int r4d_shape_quant_act_i8(const RadParam* p, int n_p, int operand,
                                      RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_I8, M, n),
        /* ONE SCALE PER ROW, AND THIS IS WHERE THE TWO PLUGINS DISAGREE ABOUT `group`.
         * r4d_quant_act_i8 takes no group argument at all: its kernel writes `S[row] = amax/127`
         * and nothing else, whatever `group` says -- the key reaches this op only to size the
         * `_asum` form's per-group sums. ref's `quant_act_i8` reads `group` as the number of
         * elements sharing one scale and writes ceil(n/group) of them a row, so at the
         * declaration the MTP block makes (n = n_embd, group = 128) the two produce different
         * planes and only libr4d's is what the w2a8 GEMM consumes. This description says what THIS
         * kernel writes; the disagreement is real and is reported rather than papered over by
         * describing a plane libr4d never fills. */
        opd((uint32_t)RAD_F32, M, 1),
        /* THE THIRD OUTPUT SELECTS THE `_asum` FORM, whose [K/group][ceil(M/16)*16] group-major
         * sums the ASYMMETRIC weight grids subtract. Absent here: it is the form the engine
         * issues, and libref implements no asymmetric grid, so passing the buffer would
         * compare a zeroed reference against a real one and report a gap in the ORACLE as a fault
         * in the kernel. */
        opd_none(),
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
        opd((uint32_t)RAD_F32, M, 1),       /* one scale per ROW */
        opd(D, M, n),                       /* out_bf16, UNROTATED, for a bf16 consumer */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gated_had_quant_i8(const RadParam* p, int n_p, int operand,
                                            RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const uint32_t D = dt_act(p, n_p, "dtype");
    const RadOpdDesc l[] = {
        /* THE PACKED gate_up FORM: `b` absent means the two halves are one [M, 2n] tensor and `b`
         * is `a + n`, which is vLLM's layout and the one the MLP actually produces. The two-buffer
         * form is the same op at a different pitch, and only one of the two can be described. */
        opd(D, M, 2 * n),
        opd_none(),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),       /* one scale per ROW */
    };
    R4D_PICK(l);
}

extern "C" int r4d_shape_gdn_gated_norm_had_quant_i8(const RadParam* p, int n_p, int operand,
                                                     RadOpdDesc* out) {

    const int64_t M  = pi(p, n_p, "M");
    const int64_t n  = pi(p, n_p, "n");
    const int64_t hd = pi(p, n_p, "head_dim");
    const uint32_t D = dt_act(p, n_p, "dtype");
    /* The RMS is per HEAD while the rotation and the int8 scale are per TOKEN, which is why this
     * fusion has no unfused equivalent in the vocabulary -- and why the gain is head_dim floats
     * and not n of them. The shim checks exactly that (`wgain is 128 floats, not K`).
     *
     * `zgrp` / `zblk` describe an INTERLEAVED qkvz projection, where element i of the flattened
     * row sits at (i / zblk) * zgrp + (i % zblk). Absent means a contiguous gate, which is the
     * shape this describes; the interleaved one is a column view of a wider plane whose width no
     * parameter of this op carries. */
    if (pi(p, n_p, "zgrp", 0) != 0) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd(D, M, n),
        opd((uint32_t)RAD_F32, hd),
        opd((uint32_t)RAD_I8, M, n),
        opd((uint32_t)RAD_F32, M, 1),       /* one scale per ROW */
    };
    R4D_PICK(l);
}

/* ---- the fp8 side. `group` is the SCALE GRID here and is 128 by the checkpoint. ---- */

extern "C" int r4d_shape_quant_act_fp8(const RadParam* p, int n_p, int operand,
                                       RadOpdDesc* out) {

    const int64_t M = pi(p, n_p, "M");
    const int64_t n = pi(p, n_p, "n");
    const int64_t g = pi(p, n_p, "group", R4D_FP8_BLOCK);
    const uint32_t D = dt_act(p, n_p, "dtype");   /* the INPUT's width; the output is E4M3 */
    if (g != R4D_FP8_BLOCK) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, M, n),
        opd((uint32_t)RAD_F8E4M3, M, n),
        opd((uint32_t)RAD_F32, M, (n + g - 1) / g),
    };
    R4D_PICK(l);
}

/* H += X^T X. `n` is K on BOTH axes of the accumulator, which is what makes this op's shape hook
 * worth reading: a caller that passed a [M][n] `h` by analogy with every other op here would be
 * refused by name rather than silently accumulating a row block. */
/* The QSA compressed key. `M` is BLOCKS and the key operand is `ratio` times taller, which is the
 * one thing a caller sizing buffers off this hook has to read rather than assume. */
/* The QSA block score. `pages` is the block-key store's height and is NOT derivable from this
 * op's parameters -- it is however many pages the KV pool has, which the runtime decides -- so the
 * hook reports the widest thing a table entry can address and a caller sizing a store off it is
 * getting a bound, not the plan. `bt` entries are PAGE IDS. */
/* The QSA selection. `sel` is `topk + 1` wide and the extra slot is the PARTIAL PAGE, not spare:
 * sized at `topk` it would drop the newest tokens on three queries in four. */
/* The work list. `M` is the TOKEN count and `work` the block bound, and they are different
 * numbers with different meanings -- one op, two extents, which is why both are parameters. */
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
        opd_idx(s + 1, 1, M + 1),      /* cu_seqlens; a drawn one is not ascending -- see below */
        /* THE TAIL STATE, SHAPED AS THE GROUP BINDS IT: [slots, heads, ring, n] with heads 1.
         * One slot a sequence is the narrowest case that still makes `sidx` mean something. */
        opd(D, s, 1, g, n),
        opd_idx(s, 1, s),              /* the state row; one column, values inside the pool */
        opd_idx(s, W, W),              /* the block-key block table */
        opd_none(),                    /* rope: the one-component form, which is every text model */
        opd(D, W * r, n),
        opd((uint32_t)RAD_I32, W),
        opd((uint32_t)RAD_I32, W),
        opd((uint32_t)RAD_I32, s),
    };
    R4D_PICK(l);
}

/* The tail. `seqs` is a parameter AND `cu` is `seqs + 1` long; the shim takes the count from the
 * operand at run time, and this hook only has to size a buffer. */
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
        opd_idx(s + 1, 1, M + 1),      /* cu_seqlens, ASCENDING -- a drawn one is not, and a
                                        * synthesised case here tests the scan and not the copy */
        opd(D, s, 1, g, n),            /* the tail state: [slots, heads, ring, n], heads 1 */
        opd_idx(s, 1, s),              /* the state row; one column, values inside the pool */
    };
    R4D_PICK(l);
}

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
        opd_idx(s, 1, nb + 1),                /* complete blocks a SEQUENCE, so at most nb */
        opd_idx(M, 1, nb * r),                /* the query's own position */
        opd_idx(s + 1, 1, M + 1, -1, RAD_OPD_F_IDX_CU),
        opd((uint32_t)RAD_I32, M, k + 1),
        opd((uint32_t)RAD_I32, M),
        opd((uint32_t)RAD_I32, M),
    };
    R4D_PICK(l);
}

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
        /* The table holds PAGE IDS, so a synthesised one must index the store and nothing wider.
         * Drawn from the operand before it by the ABI's default, which is exactly right here. */
        opd_idx(s, nb, 0),
        opd_idx(s + 1, 1, M + 1, -1, RAD_OPD_F_IDX_CU),
        opd_idx(s, 1, nb + 1),         /* complete blocks a sequence, so at most nb */
        opd((uint32_t)RAD_F32, M, nb),
    };
    R4D_PICK(l);
}

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
        /* THE RESIDUAL IS SECOND, where `rmsnorm_had_quant_i8` puts it, so the two ops differ by
         * the rotation and nothing else. Passed: it is what turns `add` + `rmsnorm` into one
         * launch and it is what every caller in this tree issues. It may ALIAS out_bf16 in the
         * engine; a tool allocating two buffers is the safe case of that. */
        opd(D, M, n),
        /* f32 or bf16 -- the shim reads the operand's own width and hands the kernel a flag.
         * Every architecture plugin here declares its norm gains f32. */
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
        opd(D, M, 2 * n),                   /* gate then up in one buffer; splitting is a copy */
        opd((uint32_t)RAD_F8E4M3, M, n),
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
    /* THE ROW IS A (token, head) PAIR here, not a token: `M` is tokens * n_head and `n` is
     * head_dim, which is what makes the gate a fixed stride inside the [q|gate] projection and
     * leaves the scale plane byte-for-byte where the out projection reads it. The gate is a COLUMN
     * SLICE in the engine; described flat, because a slice is a pitch and the extents are these. */
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
 * DFlash2 (shims in r4d_gdn.cpp). Neither op has an unfused equivalent in the vocabulary and
 * neither has a ref implementation, so a tool's only description of them is this one.
 * ========================================================================================= */

extern "C" int r4d_shape_dflash_conv(const RadParam* p, int n_p, int operand,
                                     RadOpdDesc* out) {

    const int64_t T    = pi(p, n_p, "T");
    const int64_t H    = pi(p, n_p, "hidden_size");
    const int64_t taps = pi(p, n_p, "taps");
    const int64_t NG   = pi(p, n_p, "NG");
    const int64_t grp  = pi(p, n_p, "group_size");
    const uint32_t D   = dt_act(p, n_p, "dtype");
    /* The entry point's own conditions: NG is H/group and a thread owns eight channels. Checked
     * here so a geometry that cannot be served declines rather than being described into -1. */
    if (H <= 0 || grp <= 0 || NG <= 0 || taps <= 0) return RAD_E_UNSUPPORTED;
    if (H % grp || H % 8 || NG != H / grp) return RAD_E_UNSUPPORTED;
    const RadOpdDesc l[] = {
        opd(D, T, H),
        /* delta IS A SLICE of the [T, 2, taps, NG] kernel projection and its row pitch is
         * 2*taps*NG, not taps*NG -- passing the packed pitch silently reads the other side's
         * coefficients and is not a shape error. Declared with the projection's own rank so the
         * pitch falls out of the shape rather than being asserted. */
        opd(D, T, 2, taps, NG),
        opd(D, taps, H),                    /* base_kernel: a WEIGHT, contiguous [taps, H] */
        opd(D, T, H),                       /* out -- must not alias x, which two buffers are */
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
    /* The three input planes are read at row s*steps+l, so between them they must hold M*steps
     * rows -- a short one is a walk that reads another sequence's candidates and proposes a
     * perfectly plausible draft belonging to the wrong request. */
    const int64_t rows = M * steps;
    const RadOpdDesc l[] = {
        /* CANDIDATE TOKEN IDS, AND THEY ARE GATHER ROWS: the walk scores an edge by reading
         * pred[id] and succ[id] out of two [n_vocab, rank] codebooks, so a candidate outside the
         * vocabulary reads off the end of a 1.2 GiB weight. The codebooks are the operands four
         * and five, not the one before this, so the ABI's default range is the wrong one and
         * n_vocab is said outright. */
        opd_idx(rows, K, V),
        opd((uint32_t)RAD_F32, rows, K),
        opd(D, rows, R),
        /* `anchor` is usually the step's own token_ids, which carries `block` ids a sequence with
         * the anchor first -- hence the stride. It is gathered from the codebooks exactly as the
         * candidates are, so it has the same range. The buffer has to hold (M-1)*stride + 1. */
        opd_idx((M - 1) * astr + 1, 0, V),
        opd(D, V, R),
        opd(D, V, R),
        opd((uint32_t)RAD_I32, M, steps),
    };
    R4D_PICK(l);
}
