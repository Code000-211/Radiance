/* dflash_gated_bf16.hip -- DFlash2's BF16 gate/up projection with the SiLU-gated product in the
 * epilogue, on the device. One launch where the bf16 graph runs `gemm_nt` then `silu_mul`.
 *
 * ============================== THE KERNEL IT MIRRORS ==============================
 *
 * libr4d's `gemm_bf16_nt_m64` (r4d_gemm_bf16_nt_m64.hip:71-205) at SK == 1. Every arithmetic
 * decision here is copied from it rather than re-derived, because the output must be the same
 * bytes:
 *
 *   - the fragment reads: one 16-column N tile a wave, every k step's fragments read straight from
 *     global memory with 32-bit element offsets, clamped rows (`rowGo`), KU = 8 unrolled, the
 *     address-space-1 pointer so the load is a global_load_b64 rather than a flat_load;
 *   - the accumulation: one WMMA per 16-wide k step, ASCENDING in k, into a v8f accumulator that
 *     starts at 0.f -- and at SK == 1 there is nothing after it, because "lane already holds the
 *     finished value" there (r4d_gemm_bf16_nt_m64.hip:136);
 *   - the store rounding: the language's own `(__hip_bfloat16)v`, deliberately not the header's
 *     round-half-away `f2bf` (r4d_gemm_bf16_nt_m64.hip:196-199);
 *   - the wave geometry: WV waves a block, `wave = tid / 32`, `col_local = wave / SK` which at
 *     SK == 1 is `wave`, and MT = 1 because the decode band's MB resolves to 1
 *     (r4d_gemm_args.h:196-206). The default `wv` is 1 (r4d_rows.cpp:2946,2960), so the baseline
 *     runs 32-thread blocks here too.
 *
 * THE ONLY STRUCTURAL CHANGE IS THAT A WAVE OWNS TWO N TILES INSTEAD OF ONE. The gate tile is at
 * weight row n0 and the up tile at n0 + n (the two halves of the stored [2n, K] gate_up plane), so
 * the product closes in registers rather than through a [M, 2n] global plane. Because SK == 1, and
 * because neither WV nor MB changes what an element's sum is -- only which block holds it
 * (r4d_gemm_args.h:203-205) -- the per-element addition sequence is the unfused GEMM's own.
 *
 * The wave count is therefore HALF the unfused GEMM's: n/16 windows instead of 2n/16 tiles, each
 * doing two WMMA chains that are independent of one another. Whether that is a throughput win, a
 * wash or a loss is a measurement this file does not make; the plan's occupancy budget is a
 * build-time check against the code object.
 *
 * ============================== THE TWO ROUNDINGS ==============================
 *
 * `silu_mul_kernel` (r4d_model_bf16.hip:498-508) reads bf16, computes in f32 and stores with
 * `f32_to_bf16`. The GEMM stored bf16 before it. So the fused epilogue does exactly:
 *
 *   g = widen(narrow_bf16(acc_gate))     <- gemm_nt's store, widened back (a shift)
 *   u = widen(narrow_bf16(acc_up))       <- gemm_nt's store, widened back
 *   y = f32_to_bf16((g / (1.0f + expf(-g))) * u)
 *
 * The narrow-then-widen is not a no-op and must not be optimised into a bare float use: it is the
 * bf16 rounding boundary the unfused pair has, and removing it would change the product. Both
 * conversions come from libr4d's own headers, so neither is a second spelling of a rounding rule:
 * `__hip_bfloat16(float)` for the intermediate (what m64's store executes) and `f32_to_bf16`
 * (r4d_common.h:58) for the final store (what silu_mul executes).
 *
 * The SiLU expression is character for character silu_mul's: `g / (1.0f + expf(-g))` in f32, then
 * multiplied by the up arm, then stored. Both plugins build -ffp-contract=off and neither turns on
 * fast-math, so the division, the expf and the multiply keep their own roundings.
 *
 * ============================== WHY THE GATE IS THE FIRST HALF ==============================
 *
 * `silu_mul` takes the gate as the FIRST half of its [M, 2n] operand (r4d_model_bf16.hip:495-496),
 * and rad-convert fuses gate_proj before up_proj into the stored weight in that same order
 * (rad_block_dflash2.h:608-612 declares the row parts {n_ff, n_ff} over
 * `...mlp.gate_proj.weight` then `...mlp.up_proj.weight`). So weight rows [0, n) are the gate and
 * [n, 2n) are the up arm, and the up fragment's element offset is the gate's plus n*K. That is
 * stated rather than inferred, because getting it backwards produces fluent wrong text.
 *
 * ============================== GEOMETRY ==============================
 *
 * `rad_dflash_gated_bf16_args` has already refused every geometry this kernel was not given --
 * including N < 2048, which is what makes SK == 1 by construction, and N % 32, which makes the
 * output width a whole number of 16-column windows -- so this file carries no second validation
 * and no slow path. Nothing is allocated and nothing synchronises: one launch, one argument struct
 * by value, no LDS at all (there is no cross-wave reduction because there is no split).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "dflash_gated_bf16.h"
#include "../libr4d/r4d_common.h"     /* f32_to_bf16: silu_mul's store rounding, spelled once */
#include "../libr4d/r4d_gdn_wmma.h"   /* v8bf/v8f, mma(), fragGo/rowGo -- the fragment layout */

namespace {

constexpr int kWave32 = 32;
/* K steps a loop iteration loads ahead, which is m64's KU: it is what turns the fragment address
 * into the 13-bit immediate of global_load_b64 instead of a rebuilt 64-bit pointer. */
constexpr int kKU = 8;

/* m64's fragment reader, verbatim in shape: the pointer is explicitly in the global address space
 * so the load is a global_load_b64 rather than a flat_load (which bumps dscnt on gfx12). */
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;

__device__ __forceinline__ v8bf fragU(r4d_gu2 p, int u) {
    union { unsigned long long w[2]; v8bf f; } x;
    x.w[0] = p[4 * u];
    x.w[1] = p[4 * u + 2];
    return x.f;
}

/* The GEMM store's rounding, widened back: exactly `(__hip_bfloat16)v` followed by the exact
 * widening (every bf16 is a float, so that half loses nothing). This is the bf16 boundary between
 * the projection and the activation, and it exists in the fused form because it exists in the
 * unfused one. The expression is m64's own `(__hip_bfloat16)v` character for character, so it is
 * the same rounding by construction rather than by argument. */
__device__ __forceinline__ float narrow_bf16(float v) {
    const __hip_bfloat16 h = (__hip_bfloat16)v;
    return (float)h;
}

struct Args {
    const unsigned short* __restrict__ A;
    const unsigned short* __restrict__ W;
    unsigned short* __restrict__ Y;
    int M, K, n;
};

__global__ __launch_bounds__(kWave32) void gated_kernel(Args a) {
    const int lane = threadIdx.x;
    /* m64's `col_local` with SK == 1 folded out and WV at 1: the wave owns one 16-wide N window
     * and the block is one wave. */
    const int n0 = blockIdx.x * 16;
    const int mt0 = blockIdx.y * 16;          /* MT = 1: one 16-row tile a block */

    const int K = a.K, n = a.n;

    /* The gate tile's row offset, and the up twin's: the up half of the weight starts n rows later,
     * so its element offset is the gate's plus n*K. That derivation is exact ONLY because the gate
     * window never clamps -- the grid is exactly n/16 windows and n % 16 == 0, so every lane's row
     * n0 + (lane & 15) is inside [0, n) and `rowGo`'s clamp never fires. The clamp is kept anyway
     * so the expression is m64's own. Both offsets fit unsigned 32-bit; the launcher refuses a
     * shape where they would not. */
    const unsigned woff_g = rowGo((unsigned)K, n0, n - 1, lane);
    const unsigned woff_u = woff_g + (unsigned)n * (unsigned)K;
    const unsigned aoff = rowGo((unsigned)K, mt0, a.M - 1, lane);

    v8f acc_g = (v8f)(0.f), acc_u = (v8f)(0.f);

    /* The loop body is m64's: KU unrolled steps carrying a constant offset, then the remainder. */
    const int ksteps = K / 16;
    r4d_gu2 pg = (r4d_gu2)(const unsigned long long*)(a.W + woff_g);
    r4d_gu2 pu = (r4d_gu2)(const unsigned long long*)(a.W + woff_u);
    r4d_gu2 pa = (r4d_gu2)(const unsigned long long*)(a.A + aoff);
    int s = 0;
    for (; s + kKU <= ksteps; s += kKU) {
#pragma unroll
        for (int u = 0; u < kKU; ++u) {
            const v8bf av = fragU(pa, u);
            acc_g = mma(av, fragU(pg, u), acc_g);
            acc_u = mma(av, fragU(pu, u), acc_u);
        }
        pg += kKU * 4;
        pu += kKU * 4;
        pa += kKU * 4;
    }
    for (; s < ksteps; ++s) {
        const v8bf av = fragU(pa, 0);
        acc_g = mma(av, fragU(pg, 0), acc_g);
        acc_u = mma(av, fragU(pu, 0), acc_u);
        pg += 4;
        pu += 4;
        pa += 4;
    }

    /* AT SK == 1 THERE IS NO REDUCTION: the lane holds the finished value, exactly as m64 says.
     * The epilogue is silu_mul's over the two halves, with both bf16 boundaries intact. */
    const int col = n0 + (lane & 15);
    const int mb = mt0 + 8 * (lane >> 4);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int m = mb + e;
        if (col < n && m < a.M) {
            const float g = narrow_bf16(acc_g[e]);
            const float u = narrow_bf16(acc_u[e]);
            a.Y[(size_t)m * n + col] = f32_to_bf16((g / (1.0f + expf(-g))) * u);
        }
    }
}

}  /* namespace */

/* The launch. `rad_dflash_gated_bf16_args` has already refused every geometry this row was not
 * given, so either the call is the op or it returned a reason.
 *
 * ONE WAVE A BLOCK, which is m64's own default `wv` of 1 (r4d_rows.cpp:2946,2960) and therefore
 * what an untuned baseline runs at this shape. Widening the block would move which block owns a
 * window and nothing about the arithmetic, but it buys nothing here: the grid over `n/16` windows
 * is already 120+ blocks at this checkpoint's smallest TP shard. */
int rad_mine_dflash_gated_bf16_launch(const RadArgs* a, RadStream stream) {
    RadDflashGatedCall call{};
    const int rc = rad_dflash_gated_bf16_args(a, &call);
    if (rc != RAD_OK) return rc;

    /* NO SEPARATE OFFSET CHECK HERE. `rad_dflash_gated_bf16_args` already decided
     * `product_fits(N, K)` and `product_fits(rows, K)`, which is a stronger bound than the
     * kernel's 32-bit element offsets need, so by the time this runs `call.M * call.K` and
     * `call.N * call.K` are both <= INT32_MAX. A second check here would be dead code, and a
     * refusal at issue is not a fallback in any case -- see the header's baseline section. */

    /* A PINNED SPLIT IS REFUSED UNLESS IT IS 1. This kernel has no cross-slice reduction: it
     * reproduces `gemm_bf16_nt_m64` at SK == 1, where "lane already holds the finished value".
     * DEFENSIVE ONLY: it fails this step, and does not un-resolve the op. The graph must have
     * established the `sk == 1` baseline before issue. */
    for (int i = 0; a->p && i < a->n_p; ++i) {
        if (!a->p[i].key || std::strcmp(a->p[i].key, "sk") != 0) continue;
        if (a->p[i].kind == RAD_P_INT && a->p[i].ival > 1) return RAD_E_UNSUPPORTED;
        break;
    }

    Args args{};
    args.A = static_cast<const unsigned short*>(call.a);
    args.W = static_cast<const unsigned short*>(call.w);
    args.Y = static_cast<unsigned short*>(call.y);
    args.M = (int)call.M;
    args.K = (int)call.K;
    args.n = (int)call.n;

    const dim3 grid((unsigned)(call.n / 16), (unsigned)((call.M + 15) / 16), 1);
    void* kargs[] = { &args };
    const hipError_t e = hipLaunchKernel((const void*)gated_kernel, grid, dim3(kWave32), kargs, 0,
                                         (hipStream_t)stream);
    return e == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int rad_mine_dflash_gated_bf16_count(void) { return 1; }

const RadKernelInfo* rad_mine_dflash_gated_bf16_at(int i) {
    static const RadKernelInfo row = rad_dflash_gated_bf16_row(rad_mine_dflash_gated_bf16_launch);
    return i == 0 ? &row : nullptr;
}
