// Per-(row, 128-column) symmetric E4M3 quantisation of a bf16 activation -- the gate on the fp8a8
// GEMMs. `quantization_config.activation_scheme` on the target checkpoint is "dynamic", which means
// the file ships no activation scale and one is computed every step; and a GEMM cannot compute its
// own without a reduction across every M block, so it does not try.
//
// UNLIKE r4d_quant_act_i8, THE OUTPUT IS PLAIN ROW-MAJOR. That kernel writes the byte order
// gemm_w4a8_nt_m64's A fragment reads, because that GEMM loads A straight to registers. The fp8a8
// GEMMs stage BOTH operands through the same `stage8` byte copy and read both through the same XOR
// swizzle, so the permutation would have to be undone. Row-major is also what makes the activation
// a legal B operand of the narrow instantiation, where it is the sixteen-wide side.
//
// THE GROUP IS 128 AND IS NOT A BUILD FLAG. It is the weight scale's block width -- a property of
// the checkpoint on disk, not a choice this library's quantiser makes, unlike R4D_GEMM_W4A8_GROUP.
// A different group would make the GEMM's fold read the wrong scale for part of a block.
//
// NO LDS. Every activation in this library is already row-major per token, the layout the codes
// are written in, so there is no transpose to do and staging through LDS would buy nothing.
//
// THE THREE THINGS THAT ARE NOT OBVIOUS:
//
//   448 is E4M3's largest finite value, and the scale is `amax / 448` -- a DEQUANT multiplier, the
//   same sense as the checkpoint's `weight_scale_inv`, so the GEMM's fold multiplies both scales
//   and never divides.
//
//   A ZERO BLOCK gets a scale of 1 and quantises to zeros, because `amax / 448` would be zero and
//   its reciprocal infinite.
//
//   THE CLAMP IS NOT DEFENSIVE. `amax * (1/scale)` is 448 in exact arithmetic but can land an ulp
//   above it in float, and v_cvt_pk_fp8_f32 answers an out-of-range input with NaN rather than
//   saturating -- so without the clamp EVERY block containing its own maximum produces a NaN. One
//   missing fminf is a whole-model failure.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_fp8_args.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"

namespace {

constexpr uint32_t kWave = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves = kThreads / kWave;
constexpr uint32_t kGroup = 128;

__device__ __forceinline__ float wave_max(float v) {
    v = r4d_wave_max(v);
    return v;
}

// One block a row; a wave owns one 128-column group, a lane four consecutive columns of it. Four is
// what makes the store a single dword and the wave's store 128 contiguous bytes, and it is also
// exactly the two v_cvt_pk_fp8_f32 the packing needs.
//
// ROT MAKES IT THE ROTATED QUANTISER AND NEEDS NO EXTRA MACHINERY, because that mapping -- one
// wave a 128-wide group, four consecutive values a lane -- is exactly the one r4d_fwht_wave<128> wants
// (VPL = 128/32 = 4). The transform runs in registers between the load and the absmax, with no
// LDS and no barrier, and the absmax then sees the ROTATED values -- which is the entire point,
// since the whole benefit of a rotation is that the maximum the grid is scaled by gets smaller.
//
// NOTHING NORMALISES sqrt(128) HERE. The unnormalised transform gives sqrt(128) times the
// orthonormal values, the scale is amax/448 over exactly those, and <Hw, Ha> = 128 <w, a> -- so
// the whole residual factor is one number, and r4d_layout.cpp folds it into the STORED WEIGHT
// SCALE at convert, where it is an exponent shift and costs nothing. A pass here to divide by
// sqrt(128) would be a second touch of every element for a constant the other side carries free.
// I8 IS THE SAME QUANTISER AT INT8 CODES (quant_act_i8g, had_quant_act_i8g): the scale is amax/127
// and a code is the value over it rounded to nearest-even, -127..127, one byte -- for the int8 trunk
// GEMMs, which read the same plain row-major bytes and per-(row, 128) scales the fp8 ones do.
template <bool ROT, bool I8 = false>
__global__ __launch_bounds__(kThreads) void quant_act_fp8_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s,
    uint32_t K, uint32_t nkb, uint32_t x_ld, uint32_t q_ld, uint32_t s_ld) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const size_t xrow = (size_t)row * x_ld;
    const size_t qrow = (size_t)row * q_ld;

    for (uint32_t kb = wave; kb < nkb; kb += kWaves) {
        const uint32_t c0 = kb * kGroup + lane * 4u;
        float v[4];
        float m = 0.0f;
        if (__builtin_expect(c0 + 4 <= K, 1)) {
            uint64_t raw;
            __builtin_memcpy(&raw, x + xrow + c0, sizeof(raw));
            v[0] = __uint_as_float(((uint32_t)(uint16_t)raw) << 16);
            v[1] = __uint_as_float(((uint32_t)(uint16_t)(raw >> 16)) << 16);
            v[2] = __uint_as_float(((uint32_t)(uint16_t)(raw >> 32)) << 16);
            v[3] = __uint_as_float(((uint32_t)(uint16_t)(raw >> 48)) << 16);
        } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t c = c0 + (uint32_t)j;
                v[j] = c < K ? __uint_as_float((uint32_t)x[xrow + c] << 16) : 0.0f;
            }
        }
        // The whole wave has to be inside the row for this, which is why the rotated launcher
        // refuses a K that is not a multiple of the group: a partial block would rotate real
        // values against padding zeros, and the weight side has no matching block to undo it.
        if constexpr (ROT) r4d_fwht_wave<(int)kGroup>(v, (int)lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = wave_max(m);
        constexpr float kTop = I8 ? 127.0f : 448.0f;
        const float sc = m > 0.0f ? m * (1.0f / kTop) : 1.0f;
        if (lane == 0) s[(size_t)row * s_ld + kb] = sc;
        const float inv = 1.0f / sc;
        const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
        // A ragged last group is written zero-padded rather than partially: the padding lanes hold
        // zeros, which quantise to zeros, and the GEMM's own bounds guard is what stops them
        // reaching an accumulator. A partial store would be a second code path for a case no dense
        // tensor in the target has -- but the caller must own K rounded up to 4 bytes of `q`.
        if (c0 < K) *(uint32_t*)(q + qrow + c0) = (uint32_t)pk;
    }
}

}  // namespace

extern "C" int r4d_quant_act_fp8_group(void) { return (int)kGroup; }

//   a  [M][K]                bf16, `x_ld` elements a row (0 = K)
//   q  [M][K]                E4M3, `q_ld` bytes a row (0 = K), rounded up to 4
//   s  [M][ceil(K/128)]      f32,  `s_ld` elements a row (0 = ceil(K/128))
//
// `rot` runs the 128-wide Hadamard first. It is a STRICTER call, not a flag on the same one: a
// ragged last group is legal unrotated (it is written zero-padded and the GEMM's bounds guard
// stops the padding) and is not legal rotated, because a block that mixes real values with
// padding has no matching weight block on the other side.
static int quant_act_fp8_launch(long a, long q, long s, int M, int K,
                                int x_ld, int q_ld, int s_ld, const R4dSink& sk, bool rot,
                                bool i8 = false) {
    int emit_rc = RAD_OK;
    if (!a || !q || !s) return -1;
    if (M <= 0 || K <= 0) return -2;
    if (K % 4) return -3;              // a lane owns four columns; see the kernel
    if (rot && (K % (int)kGroup)) return -5;
    if (!x_ld) x_ld = K;
    if (!q_ld) q_ld = K;
    const int nkb = (K + (int)kGroup - 1) / (int)kGroup;
    if (!s_ld) s_ld = nkb;
#define R4D_QA(ROT_, I8_)                                                                     \
    R4D_EMIT(sk, (quant_act_fp8_kernel<ROT_, I8_>), dim3((uint32_t)M), dim3(kThreads), 0,       \
             (const uint16_t*)a, (uint8_t*)q, (float*)s, (uint32_t)K, (uint32_t)nkb,             \
             (uint32_t)x_ld, (uint32_t)q_ld, (uint32_t)s_ld)
    if (i8)
        emit_rc = rot ? R4D_QA(true, true) : R4D_QA(false, true);
    else if (rot)
        emit_rc = R4D_EMIT(sk, quant_act_fp8_kernel<true>, dim3((uint32_t)M), dim3(kThreads), 0,
                           (const uint16_t*)a, (uint8_t*)q, (float*)s,
                           (uint32_t)K, (uint32_t)nkb, (uint32_t)x_ld, (uint32_t)q_ld,
                           (uint32_t)s_ld);
    else
        emit_rc = R4D_EMIT(sk, quant_act_fp8_kernel<false>, dim3((uint32_t)M), dim3(kThreads), 0,
                           (const uint16_t*)a, (uint8_t*)q, (float*)s,
                           (uint32_t)K, (uint32_t)nkb, (uint32_t)x_ld, (uint32_t)q_ld,
                           (uint32_t)s_ld);
#undef R4D_QA
    if (emit_rc != RAD_OK) return -4;
    return (!sk.desc && hipGetLastError() != hipSuccess) ? -4 : 0;
}

extern "C" int r4d_quant_act_fp8(long a, long q, long s, int M, int K,
                                 int x_ld, int q_ld, int s_ld, long stream) {
    return quant_act_fp8_launch(a, q, s, M, K, x_ld, q_ld, s_ld,
                                r4d_stream_sink((RadStream)(uintptr_t)stream), false);
}

extern "C" int r4d_had_quant_act_fp8(long a, long q, long s, int M, int K,
                                     int x_ld, int q_ld, int s_ld, long stream) {
    return quant_act_fp8_launch(a, q, s, M, K, x_ld, q_ld, s_ld,
                                r4d_stream_sink((RadStream)(uintptr_t)stream), true);
}

// ============================================================== the plugin ABI entry points
//
// The shared operand reader is r4d_fp8_args.h. See libr4d/r4d_args.h for why there is no adapter
// directory between the core and these launches.
/* ================================================================== the fp8 activation quantiser */

static int rad_quant_act_fp8(const RadArgs* a, const R4dSink& sk, bool rot, bool i8 = false) {
    const RadTensor* x  = r4d_opt(a, QF_X);
    const RadTensor* q  = r4d_opt(a, QF_Q);
    const RadTensor* sc = r4d_opt(a, QF_S);
    if (!x || !q || !sc) return R4D_NO(RAD_E_INVAL);
    /* ROWS ARE DIM 0 AND EVERYTHING AFTER IT IS THE ROW, which is the same reading r4d_row_pitch
     * already takes -- it folds every trailing extent into one pitch and returns stride[0]. So a
     * rank-3 activation is accepted and is one row per token: a GDN block's out-projection input is
     * declared [tokens, heads, width] because libr4d's scan kernels require that rank, and its K is
     * heads*width. Flattening it here rather than making the caller declare a second buffer is the
     * only way to quantise it at all, and reading it as one row per HEAD would put a scale on a
     * grid the consuming GEMM does not read.
     *
     * `q` and `sc` stay rank 2 because this kernel WRITES them and they exist for the GEMM. */
    if (x->rank < 2 || q->rank != 2 || sc->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long M = x->shape[0], K = r4d_numel(x) / (x->shape[0] > 0 ? x->shape[0] : 1);
    const long long grp = r4d_quant_act_fp8_group();
    const long long nkb = (K + grp - 1) / grp;
    /* The group is a property of the checkpoint's scale grid, not a knob. A geometry that declares
     * a different one is asking for a format the GEMM's fold does not read, and the answer is a
     * refusal by name rather than a silent 128. */
    if (r4d_geti_or(a, "group", grp) != grp) return R4D_NO(RAD_E_SHAPE);
    if (q->shape[0] != M || q->shape[1] != K) return R4D_NO(RAD_E_SHAPE);
    if (sc->shape[0] != M || sc->shape[1] != nkb) return R4D_NO(RAD_E_SHAPE);
    const long long xp = r4d_row_pitch(x), qp = r4d_row_pitch(q), sp = r4d_row_pitch(sc);
    if (xp < 0 || qp < 0 || sp < 0) return R4D_NO(RAD_E_STRIDE);
    const int e = quant_act_fp8_launch(r4d_addr(x), r4d_addr(q), r4d_addr(sc), (int)M, (int)K,
                                       (int)xp, (int)qp, (int)sp, sk, rot, i8);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

static int rad_quant_act_fp8_body(const RadArgs* a, const R4dSink& sk) {
    return rad_quant_act_fp8(a, sk, false);
}
R4D_ENTRY(r4d_rad_quant_act_fp8, rad_quant_act_fp8_body)

/* THE ROTATED FORM, and it is a DIFFERENT OP rather than a parameter on that one, for the reason
 * quant_act_i8 and had_quant_act_i8 are two ops: the number it writes is not the same number, so
 * a graph in which one was substituted for the other would be a different model that still ran.
 * The op name is what says which, and it is checked against the weight's layout tag by the pair
 * being declared together in arch/common/rad_block_moe_fp8.h -- nothing here can see the weight.
 *
 * It pairs with the `w4a8h` expert plane and with NOTHING ELSE: any consumer of this activation
 * that was not rotated at convert reads it as noise. */
static int rad_had_quant_act_fp8_body(const RadArgs* a, const R4dSink& sk) {
    return rad_quant_act_fp8(a, sk, true);
}
R4D_ENTRY(r4d_rad_had_quant_act_fp8, rad_had_quant_act_fp8_body)


/* THE INT8 PAIR: the same rows at int8 codes, for the int8 trunk GEMMs (gemm_i8a8_*). Ops of their
 * own for the reason the rotated form is one: they write other numbers into a buffer of the same
 * shape, and the name is what says which. */
static int rad_quant_act_i8g_body(const RadArgs* a, const R4dSink& sk) {
    return rad_quant_act_fp8(a, sk, false, true);
}
R4D_ENTRY(r4d_rad_quant_act_i8g, rad_quant_act_i8g_body)

static int rad_had_quant_act_i8g_body(const RadArgs* a, const R4dSink& sk) {
    return rad_quant_act_fp8(a, sk, true, true);
}
R4D_ENTRY(r4d_rad_had_quant_act_i8g, rad_had_quant_act_i8g_body)
