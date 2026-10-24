// The shared expert's DOWN projection at the speculative-verify width (M=1..4), promoted from the
// private probe reports/betterbench-1200-campaign/shared-i8a8-probe/shared_i8a8_static_m.hip with
// ONE change: the row count is derived from the OPERANDS instead of the band's `M` parameter, and
// the geometry is pinned to the served shape. Arithmetic, layout, dot order, K split, FP32 fold,
// precision and the operand list are the probe's.
//
// WHY THIS ROW EXISTS. The shared expert is `gemm_nt_q` at dtype i8a8 (arch/common/rad_fp8.h,
// LinearFP8::declare's `i8` branch: i8 codes [N][K], a bf16 scale per row per 128 of K, an i8
// activation pair with an f32 scale per (row, 128)). At M<=4 the frozen row is
// libr4d's `gemm_i8a8_nt_m16`, a 16-row WMMA tile with split-K; this row gives one 32-lane wave a
// 32-column strip and reuses each packed weight word for every live token. It is registered only
// behind RADIANCE_SHARED_I8A8_DOWN=1, so with the variable unset `rad_kernel_count` does not grow
// and every selection is the one the unchanged binary made.
//
// NOT REGISTERED: gate/up (N=768, K=2560). It was measured and loses; the constraints and the
// body both refuse it, so no wider row can pick it up by accident.
//
// NUMERICS: verlustfrei, NICHT bitgenau DECLARED (candidates/shared-i8a8-down/gate.json says
// "reordered"); see the qualifier at the end of this paragraph. The INT32 dot per 128-K group and
// every encoded byte are the frozen path's, and the per-group product is associated the same way:
// the frozen fold is `out += acc_val * (wsr * s)` (libr4d/r4d_gemm_fp8a8.hip, the `AW && I8` branch),
// which is this row's `out[m] += (float)sum[m] * (as[...] * scale)` -- same two FP32 factors, and
// multiplication is commutative, so the product is bit-identical. The epilogue rounds with the same
// `f32_to_bf16` (RNE) the frozen split-K finisher uses.
//
// THE ORDER OF THE GROUP FOLD is the only thing that could differ, and at this shape it does not:
// the frozen narrow row splits K only when `K >= 512` and the grid is short of `kSplitTarget`
// (libr4d/r4d_gemm_fp8a8.hip, `narrow_ksplit`), and K = 384 is below that bound, so the frozen row
// runs unsplit and folds the three 128-K groups ascending -- the order this row folds them in. The
// arithmetic is therefore EXPECTED to be bit-identical here. That expectation is not claimed as a
// result: the declaration stays "reordered" because the frozen row's own row text states that its
// split COUNT follows M (libr4d/r4d_rows.cpp, `gemm_fp8a8_nt_m16`) and no byte comparison has been
// run on this row yet. The parent's cold comparison is what turns it into a bitexact claim or a
// finding. No lower-precision accumulator, no approximation, no requantisation, no copy.
//
// WEIGHT LAYOUT. The stored planes are the frozen row's: `r4d.i8.w.frag64` and `r4d.i8.s.t16`
// (libr4d/r4d_fp8_layout.cpp, i8_enc_layout). Publishing the SAME hooks is not optional: a weight
// is stored once, in the arrangement the claiming kernel asked for, and the same weight is read
// by libr4d's row in the bands above M=4 -- so the tag and the bytes have to be identical or the
// wider path reads a permutation it did not expect. The hooks are compiled into THIS plugin once,
// by hc_up_wmma.hip's inclusion of libr4d/r4d_fp8_layout.cpp; they are declared here, hidden, and
// not re-included (a second definition of an extern "C" symbol would not link).
#ifdef __FAST_MATH__
#error "the reordered FP32 fold requires precise floating-point compilation"
#endif
#include <hip/hip_runtime.h>
#include "rad_abi.h"
#include "../libr4d/r4d_emit.h"
#include "../libr4d/r4d_common.h"

#include <cstdlib>
#include <cstring>

/* The frozen i8 layout hooks, defined once in this target by hc_up_wmma.hip (see the note above).
 * Declared hidden, matching those definitions, so the calls bind inside the DSO and this plugin
 * never interposes on the frozen library's copies. */
#define MINE_I8_HOOK __attribute__((visibility("hidden")))
extern "C" MINE_I8_HOOK int r4d_rad_layout_i8_block(
    const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, RadLayout*);
extern "C" MINE_I8_HOOK int r4d_rad_relayout_i8_block(
    const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, void*,
    int64_t);
extern "C" MINE_I8_HOOK int r4d_rad_unrelayout_i8_block(
    const RadParam*, int, int, const RadEncoding*, const int*, const void*, int64_t,
    const RadTensor*, int);
#undef MINE_I8_HOOK

namespace mine_shared_i8a8_down {
constexpr int Group = 128;
constexpr int N = 2560;     /* n_embd, this rank's down_proj output width */
constexpr int K = 384;      /* n_ff_shared, this rank's down_proj input width */
constexpr int RowCap = 4;   /* speculative-verify width; M >= 5 is a different row's job */

// One lane owns one output column; each packed weight word serves every live token.
template <int Rows>
__global__ __launch_bounds__(32) void dot(const int8_t* __restrict__ w,
    const uint16_t* __restrict__ ws, const int8_t* __restrict__ a,
    const float* __restrict__ as, uint16_t* __restrict__ y, int n, int k) {
    static_assert(Rows >= 1 && Rows <= RowCap, "only the verify widths are specialized");
    const int col = blockIdx.x * 32 + threadIdx.x;   /* n is a multiple of 32: 2560 / 32 = 80 */
    const int groups = k / Group;
    float out[Rows] = {};
    for (int g = 0; g < groups; ++g) {
        int32_t sum[Rows] = {};
        for (int super = 0; super < 2; ++super) {
            /* r4d_fp8_frag.h: tile (col/16, g*2+super) is 1024 bytes; lane l's four fragments are
             * at l*32 + 0/8/16/24, and fragment t covers k = g*128 + super*64 + t*16 + (l>>4)*8. */
            const int tile = ((col / 16) * (k / 64) + g * 2 + super) * 1024;
            for (int frag = 0; frag < 4; ++frag) {
                for (int half = 0; half < 2; ++half) {
                    const int kb = g * Group + super * 64 + frag * 16 + half * 8;
                    const int lane = (col & 15) + half * 16;
                    const uint32_t* wp = (const uint32_t*)(w + tile + lane * 32 + frag * 8);
                    const uint32_t w0 = wp[0], w1 = wp[1];
#pragma unroll
                    for (int m = 0; m < Rows; ++m) {
                        const uint32_t* ap = (const uint32_t*)(a + m * k + kb);
                        sum[m] = __builtin_amdgcn_sudot4(true, w0, true, ap[0], sum[m], false);
                        sum[m] = __builtin_amdgcn_sudot4(true, w1, true, ap[1], sum[m], false);
                    }
                }
            }
        }
        /* One 128-K group is at most 2,097,152 for signed i8 extrema, so no INT32 overflow. The
         * bf16 scale is widened by bit placement; both factors and the fold stay FP32. */
        const uint32_t bits = (uint32_t)ws[((col / 16) * groups + g) * 16 + col % 16] << 16;
        const float scale = __builtin_bit_cast(float, bits);
#pragma unroll
        for (int m = 0; m < Rows; ++m) out[m] += (float)sum[m] * (as[m * groups + g] * scale);
    }
#pragma unroll
    for (int m = 0; m < Rows; ++m) y[m * n + col] = f32_to_bf16(out[m]);
}

template <int Rows>
int emit(const R4dSink& sink, const int8_t* w, const uint16_t* ws, const int8_t* a,
         const float* as, uint16_t* y, int n, int k) {
    return R4D_EMIT(sink, dot<Rows>, dim3((unsigned)(n / 32)), dim3(32), 0, w, ws, a, as, y, n, k);
}

int body(const RadArgs* args, const R4dSink& sink) {
    if (!args || args->n_t < 5 || !args->t || !args->p) return RAD_E_INVAL;
    const auto* ta  = r4d_opt(args, 0);   /* activation codes      [M][K]      i8   */
    const auto* tas = r4d_opt(args, 1);   /* activation scales     [M][K/128]  f32  */
    const auto* tw  = r4d_opt(args, 2);   /* weight codes          [N][K]      i8   */
    const auto* tws = r4d_opt(args, 3);   /* weight scales         [N][K/128]  bf16 */
    const auto* ty  = r4d_opt(args, 4);   /* output                [M][N]      bf16 */
    if (!ta || !tas || !tw || !tws || !ty) return RAD_E_INVAL;
    if (ta->dtype != RAD_I8 || tas->dtype != RAD_F32 || tw->dtype != RAD_I8 ||
        tws->dtype != RAD_BF16 || ty->dtype != RAD_BF16) return RAD_E_DTYPE;

    /* THE ROW COUNT IS THE OPERANDS', NOT THE BAND'S `M`. The op is declared with a RANGE on M
     * (arch/common/rad_fp8.h, LinearFP8::declare), and core/plugin/bands.cpp resolves each band at
     * its UPPER bound -- so the `M` parameter says 4 while a decode step issues one row. The issue
     * path narrows every operand's dim 0 to the live row count (core/runtime/issue.cpp), which is
     * also where the frozen family reads M (libr4d/r4d_fp8_args.h, fp8_ops: `g->M = g->y->shape[0]`).
     * Dispatching on the parameter would run a 4-row template over a 1-row pass and write past the
     * output. Reading it here means the emitted function, and therefore the recorded dispatch, is
     * one function of the row count that is actually issued -- including on a tape replay, which
     * re-issues this body with the same live tensors. */
    /* Rank is `>= 2` for the activation and the output, exactly as the frozen family reads them
     * (libr4d/r4d_fp8_args.h, fp8_ops: `if (g->a->rank < 2)` and ROWS ARE DIM 0), so a projection
     * declared [tokens, heads, width] is not refused here for a reason the serving path never
     * hits. The weight and its scale are the declared [N][K] and [N][K/128] planes. */
    if (ty->rank < 2 || tw->rank != 2 || tws->rank != 2 || ta->rank < 2 || tas->rank != 2)
        return RAD_E_SHAPE;
    const long long M = ty->shape[0];
    const long long k = tw->shape[1];
    if (M < 1 || M > RowCap || k != K || r4d_numel(ty) % M) return RAD_E_SHAPE;
    const long long n = r4d_numel(ty) / M;
    if (n != N) return RAD_E_SHAPE;
    if (tw->shape[0] != n || tws->shape[0] != n || tws->shape[1] != k / Group ||
        ta->shape[0] != M || r4d_numel(ta) != M * k || tas->shape[0] != M ||
        tas->shape[1] != k / Group) return RAD_E_SHAPE;
    if (r4d_geti_or(args, "group", 0) != Group) return RAD_E_SHAPE;
    const char* dtype = r4d_gets(args, "dtype");
    if (!dtype || std::strcmp(dtype, "i8a8")) return RAD_E_DTYPE;

    /* Only the exact packed/tight geometry this row is written for. THESE ARE DIAGNOSTICS, NOT
     * FALLBACKS: selection already happened at declare, so a refusal fails the step and names why
     * (core/runtime/issue.cpp, `abort_step`). What actually keeps a shape this row cannot read off
     * it is the constraint table above -- M > 4 or another (N, K) never resolves to this row. */
    if (r4d_row_pitch(ta) != k || r4d_row_pitch(tas) != k / Group || r4d_row_pitch(tw) != k ||
        r4d_row_pitch(tws) != k / Group || r4d_row_pitch(ty) != n) return RAD_E_STRIDE;
    if (!r4d_aligned(ta->data, 4) || !r4d_aligned(tw->data, 4) || !r4d_aligned(tas->data, 4) ||
        !r4d_aligned(tws->data, 2) || !r4d_aligned(ty->data, 2)) return RAD_E_ALIGN;

    const int8_t* ap = (const int8_t*)ta->data;
    const float* asp = (const float*)tas->data;
    const int8_t* wp = (const int8_t*)tw->data;
    const uint16_t* wsp = (const uint16_t*)tws->data;
    uint16_t* yp = (uint16_t*)ty->data;
    const int rows = (int)M, nn = (int)n, kk = (int)k;
    switch (rows) {
        case 1: return emit<1>(sink, wp, wsp, ap, asp, yp, nn, kk);
        case 2: return emit<2>(sink, wp, wsp, ap, asp, yp, nn, kk);
        case 3: return emit<3>(sink, wp, wsp, ap, asp, yp, nn, kk);
        default: return emit<4>(sink, wp, wsp, ap, asp, yp, nn, kk);
    }
}

/* The launch and its description from one body; see libr4d/r4d_emit.h. */
R4D_ENTRY(rad_mine_shared_i8a8_down, body)

const RadKernelInfo* row() {
    static const RadConstraint constraints[] = {
        RAD_CGE("M", 1), RAD_CLE("M", RowCap), RAD_CEQ("N", N),
        RAD_CEQ("K", K), RAD_CEQ("group", Group), RAD_CIN("dtype", "i8a8"),
    };
    static const RadKernelInfo value = [] {
        RadKernelInfo r{};
        r.name = "mine_shared_i8a8_down_m1_m4"; r.op = "gemm_nt_q"; r.family = "gemm";
        r.computes = "signed i8 packed dot, int32 per 128 K, FP32 scale/fold, BF16 output";
        r.shape = "M=1..4 N=2560 K=384 group=128; tight rows, packed frag64 weights";
        r.dtypes = "I8 A and blocked W; F32 A scales; BF16 W scales and Y";
        r.domain = RAD_DOMAIN_DEVICE;
        /* Within THIS plugin only (core/plugin/select.cpp): no other `mine` row claims gemm_nt_q. */
        r.priority = 100;
        r.constraints = constraints;
        r.n_constraints = sizeof(constraints) / sizeof(constraints[0]);
        r.launch = rad_mine_shared_i8a8_down;
        r.describe = rad_mine_shared_i8a8_down_describe;
        /* The frozen row's hooks, byte for byte; see the layout note at the top of this file. */
        r.layout = r4d_rad_layout_i8_block;
        r.relayout = r4d_rad_relayout_i8_block;
        r.unrelayout = r4d_rad_unrelayout_i8_block;
        /* No scratch: one launch, nothing staged. `opd_shape` stays null exactly as the frozen row
         * publishes it (libr4d/r4d_rows.cpp, gemm_i8a8_nt_m16: layout, relayout, null, unrelayout).
         * The i8 weight is a plugin-private fragment stream whose byte count only RadLayout knows,
         * and r4d_shape_gemm_nt_q answers RAD_E_UNSUPPORTED for every non-fp8 dtype, so a hook
         * here would either duplicate that number or merely decline. */
        return r;
    }();
    return &value;
}
} // namespace mine_shared_i8a8_down

/* Default off: with the variable unset this plugin publishes exactly the rows it published
 * before, so the unchanged binary stays the baseline. */
int rad_mine_shared_i8a8_count() {
    static const bool enabled = [] {
        const char* v = std::getenv("RADIANCE_SHARED_I8A8_DOWN");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled ? 1 : 0;
}
const RadKernelInfo* rad_mine_shared_i8a8_at(int i) {
    return i == 0 && rad_mine_shared_i8a8_count() ? mine_shared_i8a8_down::row() : nullptr;
}
