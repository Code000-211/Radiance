#ifndef RAD_MINE_DFLASH_GATED_BF16_H
#define RAD_MINE_DFLASH_GATED_BF16_H

#include "rad_plugin.h"
#include <stdint.h>

/* dflash_gated_bf16 -- DFlash2's BF16 gate/up projection with the SiLU-gated product fused into
 * the GEMM's epilogue: one launch where the bf16 graph runs `gemm_nt` then `silu_mul`.
 *
 * ============================== WHY THIS IS NOT THE FP8 FOLD ==============================
 *
 * libr4d already has a gated projection, `gemm_nt_q_gated` (r4d_gemm_fp8a8.hip:1560-1660), and it
 * is not usable here for the reason rad_block_dflash2.h:831 states: its whole purpose is to write
 * the QUANTISED product, so its epilogue ends in a quantiser and its operands are the E4M3 pair.
 * A bf16 down projection reads a bf16 [T, n_ff] plane. The two ops share a shape and nothing else.
 *
 * ============================== THE NUMERICS, WHICH ARE THE CONTRACT ==============================
 *
 * The unfused pair this stands in for, at the drafter's geometry:
 *
 *   gemm_nt   C[M, 2n] = A[M, K] @ W[2n, K]^T, fp32 accumulate, store `(__hip_bfloat16)v`
 *   silu_mul  y[m, i] = f32_to_bf16((g / (1 + expf(-g))) * u),
 *             g = bf2f(C[m, i]), u = bf2f(C[m, n + i])
 *
 * so the gate and the up arm are EACH ROUNDED TO BF16 before the activation touches them. The
 * fold keeps both roundings: `narrow_bf16` in the .hip is exactly the GEMM store's own
 * `(__hip_bfloat16)v` followed by the exact widening back (every bf16 is a float, so that half is
 * a shift), and only then is the SiLU evaluated -- in f32, with the same
 * `g / (1.0f + expf(-g))` spelling and the same `f32_to_bf16` store as silu_mul_kernel
 * (r4d_model_bf16.hip:498-508). Removing either rounding would be a different function, not a
 * faster one.
 *
 * ============================== WHY THE ACCUMULATION ORDER IS UNCHANGED ============================
 *
 * The BF16 GEMM dispatch is read off the tree, not assumed, and it is THREE rows for one op:
 *
 *   M <= 16        gemm_bf16_nt_m16   (r4d_rows.cpp:4099) -- a forwarding row
 *   17 <= M <= 64  gemm_bf16_nt_m64   (r4d_rows.cpp:4108)
 *   M >= 65        gemm_bf16_nt_tiled (r4d_rows.cpp:4117)
 *
 * `m16` forwards to `m64` (r4d_gemm_bf16_nt_m16.hip:19-21), and `tiled` is documented and built
 * to be BYTE-IDENTICAL to `m64` -- same WMMA sequence, same split count, slices summed in the same
 * order from 0.f (r4d_gemm_bf16_nt_tiled.hip:16-27, r4d_rows.cpp:4120-4122). So ONE accumulation
 * law covers the whole band IN ITS UNTUNED FORM, and it is: for each output element, K is cut into
 * SK slices; within a slice the 16-wide k steps run ASCENDING into the WMMA accumulator; the
 * slices are then summed ASCENDING from 0.f.
 *
 * THE FUSION DOES NOT TOUCH THAT LAW. An output element's sum depends only on its own row, column
 * and the K loop -- never on which wave or block computes it, nor on how N is tiled. What this
 * kernel changes is WHICH COLUMNS a wave owns: it owns gate column-window t and up column-window t
 * together (the two are `n_ff/16` tiles apart in the stored [2*n_ff, K] gate_up weight), so the
 * product closes in registers instead of through a [M, 2*n_ff] global plane. No reduction tree,
 * no slice partition and no rounding boundary moves.
 *
 * ============================== SK == 1, THE SHAPE'S OWN ANSWER ==============================
 *
 * The split law is `pick_sk_for(n_tiles, K, unit, wv, bound)` (r4d_gemm_args.h:150-159): the
 * SMALLEST split whose `n_tiles * sk` reaches kFlatTileTarget = 128. This op's N is the gate_up
 * width, 2 * n_ff, which for this checkpoint (intermediate_size 7680) is 15360 at TP1, 7680 at TP2
 * and 3840 at TP4 -- 960, 480 and 240 N tiles, all far past the target -- so the law answers
 * SK = 1. That is the same answer the unfused `gemm_nt` gets at the same shape, which is what
 * keeps the two byte-identical.
 *
 * At SK = 1 the m64 kernel says it outright: "there is nothing to reduce: lane already holds the
 * finished value" (r4d_gemm_bf16_nt_m64.hip:136). This kernel serves exactly that. The row's
 * `N >= 2048` constraint makes SK = 1 unreachable-otherwise inside the declared band (2048/16 =
 * 128 tiles, which already meets the target at sk = 1 for any K divisible by 16).
 *
 * ============================== THE SUPPORTED REFERENCE BASELINE ==============================
 *
 * `UNTUNED SK == 1 ONLY`, and this is the one place the fusion is narrower than the op it stands
 * in for. The baseline reads a tuned `sk` when the tune cache has one for THIS kernel at this
 * geometry (`r4d_gemm_args.h:272-275`: `c->sk > 0 ? c->sk : ...`), and `r4d_tune_valid_gemm`
 * admits any legal split (`r4d_rows.cpp:2978-2986`). A tuned `sk >= 2` sums its slices in a
 * different tree, so a candidate fixed at SK = 1 is not byte-identical to THAT baseline.
 *
 * THE REFUSAL IS DEFENSIVE, NOT A FALLBACK. `rad_dflash_gated_bf16_args` and the launcher reject a
 * pinned `sk > 1` with `RAD_E_UNSUPPORTED`, but a refusal at issue is only a failed step: it does
 * NOT un-resolve the op and does NOT emit the unfused pair. Fusion here is a DECLARE-time
 * selection (spec §2.3): the architecture asks for the fused op, and if nothing serves it, it
 * emits the pair itself. Also, the baseline's `gemm_nt` tune-cache row is keyed on the KERNEL,
 * the PLUGIN and the geometry, so it is NOT inherited by this op's key and a `sk` from the
 * baseline row never reaches this launcher automatically.
 *
 * SO A FUTURE GRAPH INTEGRATION MUST ESTABLISH, BEFORE ISSUE, ONE OF TWO THINGS:
 *   (a) the unfused `gemm_nt` at this geometry actually selects `sk == 1` (untuned, or an
 *       explicit `sk=1` pin in the reference deployment's cache); or
 *   (b) the original unfused pair is kept for that deployment.
 * The issue-time refusal is the last guard against a mismatch, not the mechanism that produces
 * the fallback. `wv`, `mb` and `nt` are fine to pin: they move which block owns an element or the
 * load hint, never what it adds (`r4d_gemm_args.h:203-205`, and BR_FLAT makes `bound` ignore MB so
 * the split law answers the same). This row declares no tunable axes of its own.
 *
 * CONSEQUENCE FOR THE COMPARATOR: the reference arm must be run UNTUNED, or with `sk=1` pinned
 * explicitly. A reference run against a cache holding `sk>=2` at this shape is a different
 * baseline and must be reported as such, not as a candidate failure.
 *
 * ============================== WHAT IT DOES NOT DO ==============================
 *
 * No quantiser, no scale plane, no int8, no head, no all-reduce, no allocation, no synchronisation
 * and no persistent state. The output is [M, n] bf16 with n = N/2, written straight from
 * registers. Nothing is in place: `y` may not alias `a`.
 *
 * The declared chain (`gemm_nt` then `silu_mul`) is a DESCRIPTION. It is not executable by stock
 * rad-kbench, whose chain resolver looks every step up inside the fused row's own plugin
 * (tools/rad_kbench.cpp:2682-2686) and `mine` publishes neither `gemm_nt` nor `silu_mul`.
 * Parity has to be established by a comparator that names this row and each unfused libr4d
 * reference row explicitly. */
enum {
    RAD_DFLASH_GATED_A = 0,
    RAD_DFLASH_GATED_B = 1,
    RAD_DFLASH_GATED_Y = 2,
    RAD_DFLASH_GATED_OPERANDS = 3
};

/* Validated, device-free view of one call. Shape, dtype and alias checks only: nothing here
 * dereferences a pointer, because at issue every operand is device memory. */
struct RadDflashGatedCall {
    const void* a;
    const void* w;
    void*       y;
    int64_t     M;      /* the activation's rows, not the M band ceiling */
    int64_t     K;      /* n_embd */
    int64_t     N;      /* the gate_up width, 2 * n_ff */
    int64_t     n;      /* N / 2, the output width */
};

int                 rad_dflash_gated_bf16_schema_count(void);
const RadOpSchema*  rad_dflash_gated_bf16_schema_at(int);
int                 rad_dflash_gated_bf16_shape(const RadParam*, int, int, RadOpdDesc*);
/* RAD_OK, or the reason this call is not the op: RAD_E_UNSUPPORTED outside the declared geometry,
 * RAD_E_SHAPE for extents that disagree with the parameters, RAD_E_DTYPE for anything but bf16,
 * RAD_E_STRIDE for a non-contiguous operand, RAD_E_ALIGN for a pointer the fragment loads cannot
 * address, and RAD_E_INVAL for a required operand absent or an output aliasing an input. */
int                 rad_dflash_gated_bf16_args(const RadArgs*, RadDflashGatedCall*);
/* The declared chain: `gemm_nt` into a temporary, then `silu_mul` over it at half the width. */
int                 rad_dflash_gated_bf16_fuse(const RadParam*, int, uint64_t, int, RadFuseStep*);
RadKernelInfo       rad_dflash_gated_bf16_row(RadLaunchFn);
int                 rad_mine_dflash_gated_bf16_count(void);
const RadKernelInfo* rad_mine_dflash_gated_bf16_at(int);
/* The launch entry point. Declared here so the .hip can define the row from it and so the
 * registration site has one name to read; it is not part of what `rows.cpp` wire-up needs. */
int                 rad_mine_dflash_gated_bf16_launch(const RadArgs*, RadStream);

#endif /* RAD_MINE_DFLASH_GATED_BF16_H */
