/* shared_i8a8_down.hip -- the shared expert's DOWN projection at the speculative-verify width.
 *
 * Ported from mine/shared_i8a8_down.hip (ABI 16 tree) to the target ABI: the launch body, the
 * arithmetic, the geometry and the refusals are unchanged; the registration no longer carries a
 * `describe` entry point or a RadLaunchDesc sink, which target ABI 26 does not have. A kernel row
 * here is the launch alone.
 *
 * WHY THIS ROW EXISTS. The shared expert's down projection is `gemm_nt_q` at dtype i8a8
 * (arch/common/rad_fp8.h, LinearFP8::declare's `i8` branch): i8 codes [N][K], a bf16 scale per row
 * per 128 of K, an i8 activation with an f32 scale per (row, 128). At M <= 3 the frozen row is
 * libr4d's `gemm_i8a8_nt_m16`, a 16-row WMMA tile; this row gives one 32-lane wave a 32-column
 * strip and reuses each packed weight word for every live token.
 *
 * SCOPE. Exactly one geometry: N=2560 (n_embd), K=384 (n_ff_shared per rank), group 128, M=1..3.
 * M=4 IS EXCLUDED ON A MEASUREMENT: at four live rows this row's Rows=4 path is slower than the
 * frozen tile, 5.35 us against 5.23 us per launch (36/36 rounds, cold 256 MiB weights, MAIN-42),
 * while M=1..3 win by 6-8 % in the same protocol. The band boundary therefore sits at 3 and a
 * four-token verify step goes to libr4d's wider row, which is where it is faster.
 * The gate/up geometry (N=768, K=2560) was measured and LOSES; its constraints and its body both
 * refuse it, so no wider row can pick it up. This row covers K=384 only: the rank-1 K=256 shape of
 * a different model is a different instantiation and is not served here.
 *
 * NUMERICS: DECLARED "reordered". The INT32 dot per 128-K group, every encoded byte, the F32
 * activation scale, the BF16 tile-major weight scale and the BF16 output are the frozen path's,
 * and the per-group product keeps the frozen association: the frozen fold is
 * `out += acc_val * (wsr * s)` (libr4d/r4d_gemm_fp8a8.hip, the I8 branch), which is this row's
 * `out[m] += (float)sum[m] * (as[...] * scale)` -- the same two FP32 factors. The epilogue rounds
 * with the same `f32_to_bf16` (RNE) the frozen row uses.
 *
 * What can differ is the ORDER of the FP32 fold across 128-K groups. At this shape it does not:
 * the frozen narrow row splits K only when `K >= 512` and the grid is short of `kSplitTarget`
 * (libr4d/r4d_gemm_fp8a8.hip, `narrow_ksplit`), and K=384 is below that bound, so the frozen row
 * runs unsplit and folds the three groups ascending -- this row's order. The declaration stays
 * "reordered" because that is an argument, not a byte comparison on this tree: the byte check runs
 * in the parent's cold comparison, not here.
 *
 * WEIGHT LAYOUT. The stored planes are the frozen row's: `r4d.i8.w.frag64` codes and the
 * `r4d.i8.s.t16` tile-major scale plane. The hooks are libr4d's own (libr4d/r4d_fp8_layout.cpp),
 * compiled into THIS DSO with hidden visibility (CMakeLists.txt) rather than left as unresolved
 * references into a library loaded RTLD_LOCAL. The tag and the bytes therefore match what
 * libr4d's wider row reads for the same weight in the bands above M=4.
 */
#ifdef __FAST_MATH__
#error "the reordered FP32 fold requires precise floating-point compilation"
#endif

#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_plugin.h"          /* rad_param_getdim, for the operand description            */
#include "r4d_args.h"            /* r4d_opt / r4d_row_pitch / r4d_aligned: libr4d's own readings */
#include "r4d_common.h"          /* f32_to_bf16: the frozen epilogue's rounding, one definition */

#include "shared_i8a8_down.h"

#include <cstring>
#include <tuple>

namespace {

constexpr int Group  = 128;
constexpr int N      = 2560;   /* n_embd: this rank's down output width              */
constexpr int K      = 384;    /* n_ff_shared: this rank's down input width           */
constexpr int RowCap = 3;      /* measured band; M >= 4 is libr4d's wider row's job */

/* One launch path, typed: the kernel arrives as `void (*)(P...)`, so every argument is converted
 * to the parameter the kernel declares before it is packed. `hipLaunchKernel` copies raw bytes and
 * would otherwise accept a width the kernel does not read. This is libr4d's r4d_emit without the
 * dispatch log it feeds from its own row table. */
template <class... P, class... A>
int emit(RadStream stream, void (*kernel)(P...), dim3 grid, dim3 block, uint32_t lds,
         const A&... args) {
    static_assert(sizeof...(P) == sizeof...(A),
                  "a kernel argument list must match the kernel's parameter list");
    std::tuple<P...> vals(static_cast<P>(args)...);
    void* params[sizeof...(P) + 1];
    std::apply([&](P&... v) {
        int i = 0;
        ((params[i++] = (void*)&v), ...);
    }, vals);
    params[sizeof...(P)] = nullptr;
    return hipLaunchKernel((const void*)kernel, grid, block, params, lds,
                           (hipStream_t)stream) == hipSuccess
               ? RAD_OK
               : RAD_E_DEVICE;
}

/* One lane owns one output column; each packed weight word serves every live token. */
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
                    /* V_DOT4_I32_IU8: two packed 4-D i8 operands dot into an i32 accumulator, one
                     * NEG bit per source selecting signed or unsigned. Both are 1 here, i.e. i8 x
                     * i8 sign-extended; the sixth argument is CLAMP, 0. Source: "RDNA4"
                     * Instruction Set Architecture: Reference Guide, §16.10 VOP3P Instructions,
                     * V_DOT4_I32_IU8, p. 393 of 697 (PDF p. 403), revision 2025-03-27,
                     * gpuopen.com/download/rdna4-instruction-set-architecture.pdf. The four byte
                     * loads per dword are the instruction's, not this kernel's: a and w are read as
                     * 4-byte-aligned dwords, which is all a global dword load requires. */
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
int launch(RadStream stream, const int8_t* w, const uint16_t* ws, const int8_t* a,
           const float* as, uint16_t* y, int n, int k) {
    return emit(stream, dot<Rows>, dim3((unsigned)(n / 32)), dim3(32), 0, w, ws, a, as, y, n, k);
}

int body(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 5 || !args->t || !args->p) return RAD_E_INVAL;
    const auto* ta  = r4d_opt(args, 0);   /* activation codes      [M][K]      i8   */
    const auto* tas = r4d_opt(args, 1);   /* activation scales     [M][K/128]  f32  */
    const auto* tw  = r4d_opt(args, 2);   /* weight codes          [N][K]      i8   */
    const auto* tws = r4d_opt(args, 3);   /* weight scales         [N][K/128]  bf16 */
    const auto* ty  = r4d_opt(args, 4);   /* output                [M][N]      bf16 */
    if (!ta || !tas || !tw || !tws || !ty) return RAD_E_INVAL;
    if (ta->dtype != RAD_I8 || tas->dtype != RAD_F32 || tw->dtype != RAD_I8 ||
        tws->dtype != RAD_BF16 || ty->dtype != RAD_BF16) return RAD_E_DTYPE;
    /* This row reads five operands. The schema's trailing pair belongs to the asymmetric integer
     * grids and to mxfp4; a call that passes either is a different format, and silently ignoring
     * it would compute a product the caller did not ask for. */
    if (args->n_t > 5 && (r4d_opt(args, 5) || r4d_opt(args, 6)))
        return RAD_E_UNSUPPORTED;

    /* THE ROW COUNT IS THE OPERANDS', NOT THE BAND'S `M`. The op is declared with a RANGE on M
     * (arch/common/rad_fp8.h, LinearFP8::declare), and core/plugin/bands.cpp resolves each band at
     * its UPPER bound -- so `M` says 4 while a decode step issues one row. The issue path narrows
     * every operand's dim 0 to the live row count, which is where the frozen family reads M too.
     * Dispatching on the parameter would run a 4-row template over a 1-row pass and write past the
     * output. */
    /* Rank is `>= 2` for the activation and the output, exactly as the frozen family reads them
     * (libr4d/r4d_fp8_args.h): a projection declared [tokens, heads, width] is not refused here
     * for a reason the serving path never hits. The weight and its scale are the declared [N][K]
     * and [N][K/128] planes. */
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
    /* The geometry the caller declared is the geometry its operands must have. Only refused when
     * present: a caller that omits them gets the operands' own numbers. */
    const long long pn = r4d_geti_or(args, "N", n), pk = r4d_geti_or(args, "K", k);
    if (pn != n || pk != k) return RAD_E_SHAPE;
    if (r4d_geti_or(args, "group", 0) != Group) return RAD_E_SHAPE;
    const char* dtype = r4d_gets(args, "dtype");
    if (!dtype || std::strcmp(dtype, "i8a8")) return RAD_E_DTYPE;
    /* `split_n` names the stored width a K split is taken from. This row never splits K, and at
     * K=384 the frozen row's `narrow_ksplit` returns 1 before it looks at split_n (its first test
     * is `K < 512`), so accepting it here computes the frozen row's own unsplit sum. Anything
     * else -- a nonzero pb1/pb2 -- is a partitioned weight this row does not read. */
    if (r4d_geti_or(args, "pb1", 0) > 0 || r4d_geti_or(args, "pb2", 0) > 0)
        return RAD_E_UNSUPPORTED;

    /* Only the exact packed/tight geometry this row is written for. THESE ARE DIAGNOSTICS, NOT
     * FALLBACKS: selection already happened at declare, so a refusal fails the step and names why.
     * What keeps a shape this row cannot read off it is the constraint table above. */
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
        case 1: return launch<1>(stream, wp, wsp, ap, asp, yp, nn, kk);
        case 2: return launch<2>(stream, wp, wsp, ap, asp, yp, nn, kk);
        default: return launch<3>(stream, wp, wsp, ap, asp, yp, nn, kk);
    }
}

}  // namespace

/* The ABI's launch entry point: one body, reached through the row's `launch` pointer. */
extern "C" SI8_LOCAL int rad_shared_i8a8_down(const RadArgs* a, RadStream stream) {
    return body(a, stream);
}

/* ==================================================================== operand description
 *
 * This is the description libref publishes for gemm_nt_q at i8a8 (libref/ref_shapes.cpp,
 * gemm_q_shape): i8 codes with one f32 scale per (row, 128 of K) against i8 codes with one bf16
 * scale per (row, 128 of K), a bf16 output, and neither trailing aux operand. Stating it lets a
 * tool call this row cold and compare it with the oracle.
 */
extern "C" SI8_LOCAL int rad_shared_i8a8_down_shape(const RadParam* p, int n_p, int operand,
                                                    RadOpdDesc* out) {
    if (!out) return RAD_E_INVAL;
    const long long m = rad_param_getdim(p, n_p, "M", 0);
    const long long n = rad_param_getdim(p, n_p, "N", 0);
    const long long k = rad_param_getdim(p, n_p, "K", 0);
    const long long g = rad_param_getdim(p, n_p, "group", 0);
    /* Bounded on M as the row is: a description is only offered for a geometry this row would
     * actually run, so a tool asking about M=5 skips (and counts) rather than sizes buffers for a
     * launch that then refuses. N, K and the group are equalities. */
    if (m < 1 || m > RowCap || n != N || k != K || g != Group) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand > 6) return RAD_E_SHAPE;
    const long long kb = k / g;
    std::memset(out, 0, sizeof *out);
    out->fill = RAD_FILL_NORMAL;
    out->idx_const = -1;
    switch (operand) {
        case 0: out->dtype = RAD_I8;   out->rank = 2; out->shape[0] = m; out->shape[1] = k;  break;
        /* An activation scale is drawn in (0, 1): a quantiser's is never negative. */
        case 1: out->dtype = RAD_F32;  out->rank = 2; out->shape[0] = m; out->shape[1] = kb;
                out->fill = RAD_FILL_SIGMOID; break;
        case 2: out->dtype = RAD_I8;   out->rank = 2; out->shape[0] = n; out->shape[1] = k;  break;
        case 3: out->dtype = RAD_BF16; out->rank = 2; out->shape[0] = n; out->shape[1] = kb; break;
        case 4: out->dtype = RAD_BF16; out->rank = 2; out->shape[0] = m; out->shape[1] = n;  break;
        /* `a_sum` and `b_ref` are absent for this dtype. */
        default: out->flags = RAD_OPD_F_ABSENT; break;
    }
    return RAD_OK;
}
