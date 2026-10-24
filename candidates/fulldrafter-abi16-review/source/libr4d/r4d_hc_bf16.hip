// r4d_hc_bf16.hip -- the GATED RESIDUAL: hc_enter, hc_read and hc_write.
//
// Qwen4-Exp carries a residual stream `hc` times the model width and has no pre-norm and no
// residual add; every block reads a data-dependent mix of the hc sub-streams and writes its output
// back into all of them with hc data-dependent gains. docs/OPS.md's three rows hold the schema and
// arch/common/rad_block_hc.h holds the composition. This file is what runs it.
//
// ============================== WHY hc_read IS TWO KERNELS AND NOT ONE ==============================
//
// The read is
//
//     N   = grouped_rmsnorm(h)                     [HN]   HN = hc * n
//     d   = silu(W_down @ N / hc)                  [LR]   LR = lowrank
//     x   = mean_c( sigmoid(W_up @ d)[c] * N[c] )  [n]
//     inj = 2 * sigmoid(W_inj @ N / hc)            [hc]
//
// and `x` depends on ALL of `d`, so no single workgroup can produce a channel of `x` without every
// column of `d` first. A one-kernel form would need a grid-wide barrier; two kernels with the
// kernel boundary as the barrier is the same thing without cooperative launch.
//
// WHAT DECIDES THE DECOMPOSITION IS THE WEIGHT TRAFFIC, NOT THE ARITHMETIC. At Qwen3.8-Flash-Next's
// shape (n 2560, hc 4, LR 320) the two mixing matrices are 6.55 MB each, and there are 96
// connections -- 1.27 GB a step, which at ~600 GB/s is 2.1 ms and is the floor for ANY
// implementation. The arithmetic is 6.6M MACs a token, which is nothing. So the only thing that
// matters is that each weight byte is read ONCE per step, and that is what the grids below do:
//
//   kernel A   grid (ceil(LR/64), T). A workgroup owns 64 columns of W_down and reads all of h
//              once for the norm -- so h is read ceil(LR/64) times a row, 100 KB against the
//              1.3 MB of W_down the same workgroup reads, and W_down is read exactly once.
//   kernel B   grid (n/128, T). A workgroup owns 128 CHANNELS, which is hc*128 rows of W_up, so
//              W_up is read exactly once -- and `u` is never materialised, because the workgroup
//              holding all hc of a channel's rows can do the mean itself.
//
// The fp8 twin of `x` falls out of kernel B for free: 128 channels is exactly one 128-column scale
// group, so a workgroup owns whole groups and quantises its own with no second pass and no second
// launch. That is the whole reason the read emits it -- a separate `quant_act_fp8` at each of 96
// connections is ~0.43 ms of pure launch overhead at this tree's ~4.5 us floor.
//
// THE TILED PATH KEEPS THAT PROPERTY, BUT ONLY WITH THE RIGHT TILE. Above ~8 rows the per-row
// kernels re-read both mixing matrices once a ROW, so the products run TILED -- and a tile over the
// FLATTENED channel index, 64 consecutive `j`, is 64 channels of ONE sub-stream. The four terms of
// a channel's mean then live in four different workgroups, so `u` has to be materialised into
// `ubuf` and a third kernel (`hc_mix`) has to read it back: an extra launch a connection and
// 1.28 MB of pure round trip with it.
//
// `hc_up_mix_tiled_kernel` tiles over (sub-stream, channel) PAIRS instead. Same bytes of W_up, in
// hc runs of NT/hc rows rather than one run of NT, and all hc of a channel land in one workgroup
// again. What it gives up is the free fp8 twin -- a workgroup owns NT/hc = 16 channels where a
// scale group is 128 -- and an arrival counter buys that back: the last of the eight workgroups
// covering a group quantises it. Worth 1-2% of the step at multi-sequence decode, for identical
// output. At the 128-row prefill tile the same pairing is `hc_up_mix_wide_kernel`, which holds a
// channel's sub-streams in one lane's accumulators instead of in LDS.
//
// ============================== WHAT MUST MATCH THE REFERENCE ==============================
//
// libref/ref_hc.cpp is the oracle and three of its roundings are deliberate rather than incidental
// -- torch's grouped norm returns `.type_as(x)`, so the REFERENCE MODEL reads a bf16 N in both
// matrix products and in the final multiply. So this narrows N, narrows `d`, and narrows
// sigmoid(u) before the multiply, in the same places. The sums of squares are accumulated in f32
// here and f64 there, which is the one difference, and it is inside any tolerance a bf16 operand
// carries.
#include <hip/hip_runtime.h>

#include <atomic>
#include <type_traits>
#include <cstdlib>
#include <mutex>
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_gdn_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include "r4d_fwht.h"
#include "r4d_hc_fp8.h"
#include "r4d_q8.h"

namespace {

constexpr int kWave    = 32;
constexpr int kThreads = 256;
constexpr int kWaves   = kThreads / kWave;
constexpr int kHcMax   = 8;      // streams; the reference refuses hc <= 1 and nothing needs > 8
constexpr int kDownTAMax = 64;   // columns of W_down a workgroup owns, at most
constexpr int kUpRows  = 64;     // rows of W_up a workgroup owns
constexpr int kWantWG  = 64;     // workgroups worth launching, when the CU count cannot be read

/* WHAT THIS CALL RETURNS IS NOT THE CU COUNT. `hipDeviceProp_t::multiProcessorCount` reports WGPs
 * on RDNA, and a WGP is two CUs -- the R9700 has 64 CUs in 32 WGPs, which rocminfo confirms as
 * `Compute Unit: 64` -- so every grid sized from this number targets half the part.
 *
 * It is left reporting WGPs deliberately, because the number it feeds is a FLOOR on how many
 * workgroups are worth launching, and doubling that floor here would be a guess. On this kernel a
 * grid of 40, 80 and 160 all measure the same, so the block count is not what binds, and halving
 * `ta` only doubles the redundant work: each workgroup re-reads all of `h` and all of the norm
 * weight (60 KiB at this model's shape) before touching its own columns of W_down. At ta=4 that is
 * 4.8 MB of re-read against 6.55 MB of weight; at ta=8, 2.4 MB.
 *
 * A kernel that IS short of blocks on this part should double the number at its own call site
 * rather than re-titling it here. */
static inline int want_wg() {
    static const int v = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return kWantWG;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return kWantWG;
        return pr.multiProcessorCount > 0 ? pr.multiProcessorCount : kWantWG;
    }();
    return v;
}

constexpr int kUpTB    = 128;    // CHANNELS a workgroup owns -- and one fp8 scale group
constexpr int kFp8Grp  = 128;
#ifndef R4D_HC_DOT_U
#define R4D_HC_DOT_U 4      // 512-byte loads a wave keeps in flight over a long weight row
#endif
constexpr int kDotU    = R4D_HC_DOT_U;

__device__ __forceinline__ float sigmoidf_(float v) { return 1.0f / (1.0f + __expf(-v)); }
__device__ __forceinline__ float siluf_(float v)    { return v * sigmoidf_(v); }
__device__ __forceinline__ float nbf(float v)       { return bf2f(f2bf(v)); }

/* ---- THE E4M3 MIXING MATRICES (r4d_hc_fp8.h) ------------------------------------------------
 *
 * Every kernel below that reads mix_down or mix_up has an F8 form, selected by the stored dtype:
 * a row is its codes and then its f32 group scales, `ld` bytes from the next. The per-row kernels
 * multiply codes in f32 and apply a group's scale to that group's partial sum; the tiled ones
 * decode into their bf16 staging tile, scale applied, and keep the bf16 WMMA. Nothing else about a
 * kernel changes, and the bf16 form of each is compiled from the same text with F8 false. */

/* Eight E4M3 codes, two dwords of them, to f32 in order: v_cvt_pk_f32_fp8, two a time. */
__device__ __forceinline__ void e4m3x8(uint32_t lo, uint32_t hi, float* f) {
    const v2f a = __builtin_amdgcn_cvt_pk_f32_fp8((int)lo, false);
    const v2f b = __builtin_amdgcn_cvt_pk_f32_fp8((int)lo, true);
    const v2f c = __builtin_amdgcn_cvt_pk_f32_fp8((int)hi, false);
    const v2f d = __builtin_amdgcn_cvt_pk_f32_fp8((int)hi, true);
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
    f[4] = c.x; f[5] = c.y; f[6] = d.x; f[7] = d.y;
}

/* Eight bf16 of a uint4 to f32, in order. */
__device__ __forceinline__ void bf16x8(const uint4& v, float* f) {
    const uint32_t w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i]     = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

/* Eight codes of an E4M3 row, decoded and scaled into the bf16 a tiled kernel stages. */
__device__ __forceinline__ uint4 e4m3x8_bf16(const unsigned char* __restrict__ codes, float sc) {
    const uint2 c = *(const uint2*)codes;
    float f[8];
    e4m3x8(c.x, c.y, f);
    union { uint4 v; unsigned short s[8]; } o;
#pragma unroll
    for (int e = 0; e < 8; ++e) o.s[e] = f2bf(f[e] * sc);
    return o.v;
}

/* A WAVE'S DOT OF THE NATURAL-ORDER LDS VECTOR WITH ONE E4M3 ROW OF mix_down. A lane takes
 * SIXTEEN consecutive codes of every 512-element step -- one 16-byte load, so a wave's request is
 * 512 bytes as the bf16 dots' are; at eight codes a lane it would be 256, twice the requests for
 * the same bytes. Sixteen share one 128-group, whose scale multiplies the lane's
 * partial sum for the step. kU steps are loaded before the first is summed. K is a multiple of
 * 512. */
constexpr long kHc8Step = kWave * 16;

__device__ __forceinline__ float hc8_dot16(const unsigned short* __restrict__ sv, uint4 c) {
    const uint4 s0 = *(const uint4*)sv;
    const uint4 s1 = *(const uint4*)(sv + 8);

    const v2f a0 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.x, false);
    const v2f a1 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.x, true);
    const v2f a2 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.y, false);
    const v2f a3 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.y, true);

    const v2f a4 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.z, false);
    const v2f a5 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.z, true);
    const v2f a6 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.w, false);
    const v2f a7 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c.w, true);

    float p = 0.0f;
    p = fmaf(a0.x, __uint_as_float(s0.x << 16), p);
    p = fmaf(a0.y, __uint_as_float(s0.x & 0xffff0000u), p);
    p = fmaf(a1.x, __uint_as_float(s0.y << 16), p);
    p = fmaf(a1.y, __uint_as_float(s0.y & 0xffff0000u), p);
    p = fmaf(a2.x, __uint_as_float(s0.z << 16), p);
    p = fmaf(a2.y, __uint_as_float(s0.z & 0xffff0000u), p);
    p = fmaf(a3.x, __uint_as_float(s0.w << 16), p);
    p = fmaf(a3.y, __uint_as_float(s0.w & 0xffff0000u), p);

    p = fmaf(a4.x, __uint_as_float(s1.x << 16), p);
    p = fmaf(a4.y, __uint_as_float(s1.x & 0xffff0000u), p);
    p = fmaf(a5.x, __uint_as_float(s1.y << 16), p);
    p = fmaf(a5.y, __uint_as_float(s1.y & 0xffff0000u), p);
    p = fmaf(a6.x, __uint_as_float(s1.z << 16), p);
    p = fmaf(a6.y, __uint_as_float(s1.z & 0xffff0000u), p);
    p = fmaf(a7.x, __uint_as_float(s1.w << 16), p);
    p = fmaf(a7.y, __uint_as_float(s1.w & 0xffff0000u), p);

    return p;
}

template <int kU>
__device__ __forceinline__ float hc8_dot_down(const unsigned short* __restrict__ sv,
                                              const unsigned char* __restrict__ wr, long K,
                                              int lane) {
    const float* __restrict__ sr = (const float*)(wr + K);
    float acc = 0.0f;
    long  j = 0;
    for (; j + kHc8Step * kU <= K; j += kHc8Step * kU) {
        uint4 c[kU];
        float s[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            c[u] = *(const uint4*)(wr + e);
            s[u] = sr[(size_t)e >> 7];
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            acc = fmaf(hc8_dot16(sv + e, c[u]), s[u], acc);
        }
    }
    for (; j < K; j += kHc8Step) {
        const long e = j + lane * 16;
        acc = fmaf(hc8_dot16(sv + e, *(const uint4*)(wr + e)), sr[(size_t)e >> 7], acc);
    }
    return acc;
}

/* ONE THREAD'S QUARTER OF AN E4M3 ROW OF mix_up against RG rows of `d` in LDS. The quarter is the
 * row's scale group (r4d_hc8::up_group), so its partial sums are complete before the scale is
 * applied and four threads' sums make the row. All of a quarter's sixteen-code loads are issued
 * before the first is used: at lowrank 320 that is five of them, and kMaxQ bounds it at 512. */
constexpr int kUpMaxQ = 8;
template <int RG>
__device__ __forceinline__ void hc8_dot_up(const float* __restrict__ sd, long lr,
                                           const unsigned char* __restrict__ wr, int q4,
                                           float* acc) {
    const long G = lr / 4, k0 = (long)q4 * G;
    const int  nq = (int)(G / 16);
    uint4 c[kUpMaxQ];
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i)
        if (i < nq) c[i] = *(const uint4*)(wr + k0 + 16 * i);
#pragma unroll
    for (int r = 0; r < RG; ++r) acc[r] = 0.0f;
#pragma unroll
    for (int i = 0; i < kUpMaxQ; ++i) {
        if (i >= nq) break;
        const v2f a0 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].x, false);
        const v2f a1 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].x, true);
        const v2f a2 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].y, false);
        const v2f a3 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].y, true);
        const v2f a4 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].z, false);
        const v2f a5 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].z, true);
        const v2f a6 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].w, false);
        const v2f a7 = __builtin_amdgcn_cvt_pk_f32_fp8((int)c[i].w, true);
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const float* dr = sd + (long)r * lr + k0 + 16 * i;
            float p = acc[r];
            p = fmaf(a0.x, dr[0], p);
            p = fmaf(a0.y, dr[1], p);
            p = fmaf(a1.x, dr[2], p);
            p = fmaf(a1.y, dr[3], p);
            p = fmaf(a2.x, dr[4], p);
            p = fmaf(a2.y, dr[5], p);
            p = fmaf(a3.x, dr[6], p);
            p = fmaf(a3.y, dr[7], p);
            p = fmaf(a4.x, dr[8], p);
            p = fmaf(a4.y, dr[9], p);
            p = fmaf(a5.x, dr[10], p);
            p = fmaf(a5.y, dr[11], p);
            p = fmaf(a6.x, dr[12], p);
            p = fmaf(a6.y, dr[13], p);
            p = fmaf(a7.x, dr[14], p);
            p = fmaf(a7.y, dr[15], p);
            acc[r] = p;
        }
    }
    const float sc = ((const float*)(wr + lr))[q4];
#pragma unroll
    for (int r = 0; r < RG; ++r) {
        float a = acc[r] * sc;
        a += __shfl_xor(a, 1);
        a += __shfl_xor(a, 2);
        acc[r] = a;
    }
}

/* ON THE VALU, NOT THE LDS CROSSBAR -- r4d_common.h has the measurement. It matters here because
 * the norm prologue reduces `hc + 1` accumulators back to back and `hc` is a runtime bound, so the
 * five exchanges of each are one dependency chain of five times `hc + 1`. Bit-identical: the same
 * lanes in the same order. */
__device__ __forceinline__ float wave_sum(float v) { return r4d_wave_sum(v); }

/* ============================================================== hc_enter
 *
 * h[c][i] = x[i], for every stream. One launch a step, and it is its own op because the embedding
 * lookup's weight is [vocab, n] -- a wide output would be a different weight shape. */
__global__ __launch_bounds__(kThreads)
void hc_enter_kernel(const unsigned short* __restrict__ x, unsigned short* __restrict__ h,
                     long rows, long n, int hc, long xrow, long hrow) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    const unsigned short* xp = x + r * xrow;
    unsigned short*       hp = h + r * hrow;
    for (long i = (long)blockIdx.x * kThreads + threadIdx.x; i < n; i += (long)gridDim.x * kThreads) {
        const unsigned short v = xp[i];
        for (int c = 0; c < hc; ++c) hp[(long)c * n + i] = v;
    }
}


/* ============================================================== hc_write
 *
 * h[c][i] += inj[c] * y[i]. In place, on the UNNORMED stream -- the read's normalisation is
 * consumed by the block input and the gates and never enters the residual. */
__global__ __launch_bounds__(kThreads)
void hc_write_kernel(const unsigned short* __restrict__ y, const unsigned short* __restrict__ inj,
                     unsigned short* __restrict__ h,
                     long rows, long n, int hc, long yrow, long irow, long hrow) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    /* THROUGH LDS AND NOT A REGISTER ARRAY. `hc` is a runtime value, so `g[c]` in the inner loop
     * is a dynamic index into private memory -- which the compiler spills to scratch, turning a
     * four-element broadcast into a per-element memory round trip. */
    __shared__ float s_g[kHcMax];
    if ((int)threadIdx.x < hc) s_g[threadIdx.x] = bf2f(inj[r * irow + threadIdx.x]);
    __syncthreads();
    const unsigned short* yp = y + r * yrow;
    unsigned short*       hp = h + r * hrow;
    for (long i = (long)blockIdx.x * kThreads + threadIdx.x; i < n; i += (long)gridDim.x * kThreads) {
        const float v = bf2f(yp[i]);
        for (int c = 0; c < hc; ++c) {
            unsigned short* p = hp + (long)c * n + i;
            *p = f2bf(bf2f(*p) + s_g[c] * v);
        }
    }
}

/* EIGHT bf16 A LANE, the same element update. hc_write_kernel moves one u16 a lane, 64 B a wave
 * instruction, and reaches ~81% of bandwidth at a prefill chunk. Here a lane owns 8 contiguous
 * columns: one 16-byte load of y, then every stream's 16-byte load of h before any of them is
 * updated and stored back. Each element is still `f2bf(bf2f(h) + s_g[c] * v)`, so the same bits.
 * The launcher takes it only when n, both pitches and both bases are 16-byte aligned. */
__device__ __forceinline__ unsigned short hcw_u16(const uint4& v, int k) {
    const unsigned w = k < 2 ? v.x : k < 4 ? v.y : k < 6 ? v.z : v.w;
    return (unsigned short)((k & 1) ? (w >> 16) : (w & 0xFFFFu));
}

__global__ __launch_bounds__(kThreads)
void hc_write_vec_kernel(const unsigned short* __restrict__ y, const unsigned short* __restrict__ inj,
                         unsigned short* __restrict__ h,
                         long rows, long n, int hc, long yrow, long irow, long hrow) {
    const long r = blockIdx.y;
    if (r >= rows) return;
    __shared__ float s_g[kHcMax];
    if ((int)threadIdx.x < hc) s_g[threadIdx.x] = bf2f(inj[r * irow + threadIdx.x]);
    __syncthreads();
    const unsigned short* yp = y + r * yrow;
    unsigned short*       hp = h + r * hrow;
    const long n8 = n / 8;
    for (long i8 = (long)blockIdx.x * blockDim.x + threadIdx.x; i8 < n8;
         i8 += (long)gridDim.x * blockDim.x) {
        const long i = i8 * 8;
        const uint4 yv = *(const uint4*)(yp + i);
        uint4 hv[kHcMax];
#pragma unroll
        for (int c = 0; c < kHcMax; ++c)
            if (c < hc) hv[c] = *(const uint4*)(hp + (long)c * n + i);
#pragma unroll
        for (int c = 0; c < kHcMax; ++c) {
            if (c >= hc) break;
            unsigned short o[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float v = bf2f(hcw_u16(yv, k));
                o[k] = f2bf(bf2f(hcw_u16(hv[c], k)) + s_g[c] * v);
            }
            *(uint4*)(hp + (long)c * n + i) =
                make_uint4((unsigned)o[0] | ((unsigned)o[1] << 16), (unsigned)o[2] | ((unsigned)o[3] << 16),
                           (unsigned)o[4] | ((unsigned)o[5] << 16), (unsigned)o[6] | ((unsigned)o[7] << 16));
        }
    }
}

/* A WAVE'S DOT PRODUCT OF THE LDS VECTOR WITH A CONTIGUOUS bf16 ROW, at 512 bytes a load and in
 * the ORDER THE SCALAR FORM SUMMED IN.
 *
 * `for (j = lane; j < HN; j += kWave)` is one 64-BYTE load a wave instruction -- a quarter of what
 * the memory system wants -- and it reaches 219 GB/s on a shape where the lm head gets 631 out of
 * the same part. Eight CONTIGUOUS bf16 a lane is 512 bytes and reaches 554, but it changes which
 * j's a lane owns, and therefore the last bit of every output the kernel writes.
 *
 * So the wide load goes THROUGH LDS and each lane reads back exactly the strided subset it would
 * have loaded itself, ascending: lane L still sums j = L, L+32, L+64, ... The transpose costs
 * nothing and in fact wins -- 8.9 us, against 9.0 for simply reordering which j a lane owns and
 * 29.9 for the scalar form -- because the reordered form's lanes fight over the same LDS banks
 * for `sN` where these do not.
 * `stage` is this wave's own 256-entry slot, so the round trip needs no barrier: a wave's LDS
 * operations complete in order and no other wave reads it.
 *
 * The unaligned row falls back to the scalar loop rather than being refused -- the fast path needs
 * the row 16-byte aligned, which is the base and `HN % 8` together. */
__device__ __forceinline__ float sv_at(const unsigned short* p, long i) { return bf2f(p[i]); }
__device__ __forceinline__ float sv_at(const float* p, long i) { return p[i]; }

/* `kU` IS HOW MANY LOADS A WAVE HAS IN FLIGHT, and it is the whole performance of kernel A.
 *
 * The staged form above issues ONE 512-byte load and then immediately reads the staged bytes back
 * out of LDS, so the global latency is exposed in full on every iteration -- a wave is never
 * waiting on more than one load. Kernel A runs 40 workgroups of 8 waves, so that is 320 loads of
 * 512 bytes in flight against a part that needs some hundreds of KiB to stay fed, which is
 * 25.4 us for 6.55 MB, or 258 GB/s. Kernel B moves the SAME 6.55 MB in 13.0 us (504 GB/s) purely
 * because its waves each own eight rows and so have eight loads outstanding.
 *
 * So the lever is not a wider load, a bigger grid or fewer workgroups -- every block count from 40
 * to 160 lands within 0.2 us of the others. It is depth: issue kU loads into registers FIRST, where
 * they have no dependency on each other and the compiler emits kU back-to-back global_load_dwordx4,
 * and only then walk them through LDS one at a time.
 *
 * BIT-IDENTICAL, and that is why the staging slot is not replicated kU times. Lane L still sums
 * j = L, L+32, ... within a 256-element chunk and still takes the chunks in ascending order; only
 * when the bytes ARRIVE moves. One slot reused kU times costs kU ordered LDS round trips, ~100
 * cycles each, which is what the one global latency already cost -- and replicating the slot would
 * take kernel A's LDS from 25 KiB to 37 KiB and halve the workgroups a CU can hold. */
template <int kU, typename SV>
__device__ __forceinline__ float wave_dot_lds(const SV* __restrict__ sv,
                                              const unsigned short* __restrict__ wr, long HN,
                                              int lane, unsigned short* stage) {
    constexpr long kPer = kWave * 8;
    float acc = 0.0f;
    long  j = 0;
    if (((uintptr_t)wr & 15u) == 0u) {
        if (kU > 1) {
            for (; j + kPer * kU <= HN; j += kPer * kU) {
                uint4 v[kU > 1 ? kU : 1];
#pragma unroll
                for (int u = 0; u < kU; ++u)
                    v[u] = *(const uint4*)(wr + j + (long)u * kPer + lane * 8);
#pragma unroll
                for (int u = 0; u < kU; ++u) {
                    *(uint4*)(stage + lane * 8) = v[u];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const int jj = lane + t * kWave;
                        acc += sv_at(sv, j + (long)u * kPer + jj) * bf2f(stage[jj]);
                    }
                }
            }
        }
        for (; j + kPer <= HN; j += kPer) {
            *(uint4*)(stage + lane * 8) = *(const uint4*)(wr + j + lane * 8);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int jj = lane + t * kWave;
                acc += sv_at(sv, j + jj) * bf2f(stage[jj]);
            }
        }
    }
    for (long i = j + lane; i < HN; i += kWave) acc += sv_at(sv, i) * bf2f(wr[i]);
    return acc;
}

/* N IS HELD IN LDS TRANSPOSED WITHIN EACH 256-ELEMENT CHUNK, AND THAT IS WORTH A THIRD OF KERNEL A.
 *
 * `wave_dot_lds` above reads the STAGED WEIGHT out of LDS strided -- lane L takes jj = L, L+32,
 * ... -- because that is the summation order and it is not negotiable. It reads `sv`, the LDS copy
 * of N, at the SAME strided addresses, and that one is free to move: N is written by this kernel,
 * so the layout is ours to pick.
 *
 * A strided 16-bit LDS read is the worst access this part has. `ds_load_u16 v, v6 offset:64*t`
 * gives lane L the address 2L + 64t: thirty-two lanes land in sixteen of the thirty-two four-byte
 * banks, two lanes a bank, so the instruction takes two LDS cycles and returns 64 of the 128 bytes
 * a cycle is worth. Eight of them a chunk, for N, and eight more for the weight. Disassembled at
 * this model's shape the inner loop is 228 instructions for 32 lane-FMAs -- 64 ds_load_u16, 42
 * v_lshlrev_b32 (which is what bf2f is) and 60 VALU -- and at 84 blocks that is ~24k cycles a
 * compute unit, which IS the 18.7 us the kernel traces. THE KERNEL IS ISSUE-BOUND, NOT BANDWIDTH-
 * BOUND, and the evidence is that M=1 reading 6.55 MB and M=4 reading 26.2 MB both take 18.7 us.
 *
 * So store N so that the eight elements ONE LANE WANTS ARE CONTIGUOUS. Inside a chunk of kPer =
 * 256, logical element p goes to physical (p % 32) * 8 + (p / 32); lane L's eight then sit at
 * physical L*8 .. L*8+7, which is one 16-byte `ds_load_b128` at a 16-byte-aligned address, no
 * conflict, four cycles for the whole 512-byte chunk. Eight ds_load_u16 become one ds_load_b128.
 *
 * BIT-IDENTICAL, and that is the point: lane L still multiplies N[j + L + 32t] by W[j + L + 32t]
 * for t ascending, still sums into its own scalar, and the chunks still ascend. ONLY THE ADDRESS
 * THE VALUE IS FETCHED FROM MOVES. The chunk is 256 and never straddles a sub-stream because
 * n % 256 == 0 is a condition of the path; when it does not hold the kernel keeps the plain layout
 * and the plain loop, which is what the `swz` flag below selects.
 *
 * THE COST LANDS ON WHOEVER WRITES N, and it is why there is no raw copy of `h` into LDS. Writing
 * the transpose out of a plain staged copy would be eight 2-byte LDS stores at stride 16, an
 * EIGHT-way conflict -- worse than what it saves. So the staging and the rescale are ONE pass:
 * a thread owns one (chunk, lane) group, reads its eight `h` and eight `w` straight from global
 * where both are still coalesced across the wave, and writes the group with a single
 * ds_store_b128. That is fewer instructions than a copy-then-rescale, and it costs one extra pass
 * over `h` (20 KiB a block, against the 320 KiB of W_down the same block reads) because the sum of
 * squares has no LDS copy to read and takes `h` from global instead.
 * The reduction's order is untouched: thread tid still accumulates i = tid, tid + 256, ... */
template <int kU>
/* AND THE WEIGHT IS TRANSPOSED BY THE LOAD, WHICH RETIRES THE STAGING SLOT ALTOGETHER.
 *
 * The bank-conflict argument above rules out doing the weight's transpose IN LDS, and nothing
 * more. `global_load_tr_b128` does it in the LOAD instead: the eight lanes of a group each name an
 * address and lane j receives element j from all eight of them (r4d_common.h, and three attention
 * kernels already read V that way).
 *
 * THE ADDRESS THAT MAKES IT THE RIGHT TRANSPOSE. Lane L must own the chunk's elements L, L+32,
 * ... L+224 -- that is the summation order and it is not negotiable. Write L = 8g + i; if lane L
 * names `wr + j + 8g + 32i`, then lane 8g+j receives { wr[j + 8g + j' + 32i] : i = 0..7 } for its
 * own j' = L % 8, which is exactly elements L, L+32, ... in ascending order. Every address is
 * 16-byte aligned because j is a multiple of kPer, 8g and 32i are multiples of 8, and `swz`
 * already requires `down` itself to be.
 *
 * WHAT IT REMOVES: one 16-byte LDS store and EIGHT `ds_load_u16` a chunk, the eight being a
 * two-way bank conflict (lane L reads byte 2L + 64t, so thirty-two lanes land in sixteen banks) --
 * about sixteen of the twenty LDS cycles a chunk that remain once N is one `ds_load_b128`. This
 * path takes no `stage` slot at all.
 *
 * BIT-IDENTICAL: lane L multiplies the same N element by the same weight element in the same
 * order into the same scalar. Only where the weight arrives from moves. */
__device__ __forceinline__ float wave_dot_swz(const unsigned short* __restrict__ sv,
                                              const unsigned short* __restrict__ wr, long HN,
                                              int lane) {
    constexpr long kPer = kWave * 8;
    const long woff = 8 * (long)(lane / 8) + 32 * (long)(lane % 8);
    float acc = 0.0f;
    long  j = 0;
    for (; j + kPer * kU <= HN; j += kPer * kU) {
        uint4 v[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u)
            v[u] = __builtin_bit_cast(uint4, load_tr_b128(wr + j + (long)u * kPer + woff));
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            const uint4 g = *(const uint4*)(sv + j + (long)u * kPer + lane * 8);
            const unsigned short* gp = (const unsigned short*)&g;
            const unsigned short* wp = (const unsigned short*)&v[u];
#pragma unroll
            for (int t = 0; t < 8; ++t) acc += bf2f(gp[t]) * bf2f(wp[t]);
        }
    }
    for (; j + kPer <= HN; j += kPer) {
        const uint4 g = *(const uint4*)(sv + j + lane * 8);
        const unsigned short* gp = (const unsigned short*)&g;
        const uint4 w = __builtin_bit_cast(uint4, load_tr_b128(wr + j + woff));
        const unsigned short* wp = (const unsigned short*)&w;
#pragma unroll
        for (int t = 0; t < 8; ++t) acc += bf2f(gp[t]) * bf2f(wp[t]);
    }
    return acc;
}

/* ============================================================== mtp_enter
 *
 * THE MULTI-TOKEN-PREDICTION HEAD'S ENTRY, and it is in this file because it is the same shape of
 * problem as hc_read: a wide stream, a grouped norm over it, and a matvec whose weight is the only
 * traffic that matters.
 *
 *     ne[j]    = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
 *     fe[i]    = fc_e[i] . ne
 *     nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)          rs[k] over sub-stream k
 *     x[k*n+i] = fc_h[i] . nh[k] + fe[i]                         for each of the hc sub-streams
 *
 * WHAT DECIDES THE DECOMPOSITION IS THAT fc_h[i] IS READ ONCE AND USED hc TIMES. The two square
 * matrices are 13.1 MB at this model's shape and a draft round issues this once, so the arithmetic
 * (hc + 1 dots of 2560 per output channel) is free and the weight bytes are the whole cost. A
 * workgroup therefore owns kOutTB output CHANNELS, stages `nh` and `ne` in LDS, and walks each
 * weight row ONCE with hc accumulators live -- the alternative, hc calls to wave_dot_lds over the
 * same row, would read fc_h four times and quadruple the only traffic there is.
 *
 * The norms are recomputed by every workgroup, which is 25 KB of `h` and `e` against the 327 KB of
 * weights the same workgroup reads -- 7.6%, and the alternative is a second launch and a scratch
 * buffer for five scalars. */
constexpr int kMtpOutTB = 32;    // output channels a workgroup owns: 80 blocks at n = 2560
/* Above this many rows the tiled form below wins, and at one row it cannot -- the per-row kernel
 * already reads the weight once there and is the floor. From two rows up the tiled form is never
 * worse in step time and the kernel itself is up to 2.7x faster a call; at high concurrency the
 * step is flat only because the device is not fully busy there. */
constexpr long kMtpTileMinRows = 2;
/* `nh` + `ne` in LDS, in elements. 64 KiB less the staging slots and the reduction planes. */
constexpr long kMtpLdsMaxN = 28672;

/* One pass over a weight row with hc accumulators live, staged through LDS exactly as
 * wave_dot_lds does and for the same reason -- a 16-byte lane load reorders which j a lane owns,
 * and that changes the last bit of the sum. Lane L still sums j = L, L+32, L+64, ... */
template <int kU>
__device__ __forceinline__ void wave_dot_hc(const unsigned short* __restrict__ sv, long svstride,
                                            int hc, const unsigned short* __restrict__ wr, long n,
                                            int lane, unsigned short* stage, float* acc) {
    constexpr long kPer = kWave * 8;
    long j = 0;
    if (((uintptr_t)wr & 15u) == 0u && (n % 8) == 0) {
        if (kU > 1) {
            for (; j + kPer * kU <= n; j += kPer * kU) {
                uint4 v[kU > 1 ? kU : 1];
#pragma unroll
                for (int u = 0; u < kU; ++u)
                    v[u] = *(const uint4*)(wr + j + (long)u * kPer + lane * 8);
#pragma unroll
                for (int u = 0; u < kU; ++u) {
                    *(uint4*)(stage + lane * 8) = v[u];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const int  jj = lane + t * kWave;
                        const long jg = j + (long)u * kPer + jj;
                        const float wv = bf2f(stage[jj]);
                        for (int c = 0; c < hc; ++c)
                            acc[c] += bf2f(sv[(long)c * svstride + jg]) * wv;
                    }
                }
            }
        }
        for (; j + kPer <= n; j += kPer) {
            *(uint4*)(stage + lane * 8) = *(const uint4*)(wr + j + lane * 8);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int  jj = lane + t * kWave;
                const long jg = j + jj;
                const float wv = bf2f(stage[jj]);
                for (int c = 0; c < hc; ++c)
                    acc[c] += bf2f(sv[(long)c * svstride + jg]) * wv;
            }
        }
    }
    for (long i = j + lane; i < n; i += kWave) {
        const float wv = bf2f(wr[i]);
        for (int c = 0; c < hc; ++c) acc[c] += bf2f(sv[(long)c * svstride + i]) * wv;
    }
}

/* BOTH WEIGHT ROWS OF ONE OUTPUT CHANNEL IN ONE PASS, over `nh` and `ne` held TRANSPOSED within
 * each 256-element chunk as wave_dot_swz reads N: a lane's eight elements of a chunk are one
 * ds_load_b128 a stream, and each weight row arrives transposed by global_load_tr_b128. The fc_h
 * row feeds the hc stream accumulators off one load and the fc_e row feeds `ae`, and all 2 kU loads
 * of an iteration are issued before the first is read.
 *
 * wave_dot_hc and wave_dot_lds, the plain-layout pair, stage each weight load through LDS and read
 * it and every stream back with strided 16-bit LDS loads -- two-way bank conflicts, 41 LDS
 * instructions a chunk for 40 lane-FMAs, and an accumulator array indexed by the runtime `hc`,
 * which is private memory. Here a chunk is hc + 1 conflict-free LDS loads, and the accumulators
 * are unrolled over kHcMax with the stream count as a guard, so they stay in registers.
 *
 * BIT-IDENTICAL to that pair: each accumulator still sums elements L, L + 32, ... L + 224 of each
 * chunk, chunks ascending, each product the stream value times the weight. That the fc_e dot
 * interleaves with the fc_h ones changes no accumulator's own sequence. */
template <int kU>
__device__ __forceinline__ void wave_dot_mtp_swz(const unsigned short* __restrict__ snh,
                                                 const unsigned short* __restrict__ sne,
                                                 const unsigned short* __restrict__ wh_row,
                                                 const unsigned short* __restrict__ we_row,
                                                 long n, int hc, int lane, float* acc, float& ae) {
    constexpr long kPer = kWave * 8;
    const long woff = 8 * (long)(lane / 8) + 32 * (long)(lane % 8);
    auto chunk = [&](long jc, const uint4& vh, const uint4& ve) {
        const unsigned short* wp = (const unsigned short*)&vh;
#pragma unroll
        for (int c = 0; c < kHcMax; ++c) {
            if (c < hc) {
                const uint4 g = *(const uint4*)(snh + (long)c * n + jc);
                const unsigned short* gp = (const unsigned short*)&g;
#pragma unroll
                for (int t = 0; t < 8; ++t) acc[c] += bf2f(gp[t]) * bf2f(wp[t]);
            }
        }
        const uint4 g = *(const uint4*)(sne + jc);
        const unsigned short* gp = (const unsigned short*)&g;
        const unsigned short* ep = (const unsigned short*)&ve;
#pragma unroll
        for (int t = 0; t < 8; ++t) ae += bf2f(gp[t]) * bf2f(ep[t]);
    };
    long j = 0;
    for (; j + kPer * kU <= n; j += kPer * kU) {
        uint4 vh[kU], ve[kU];
#pragma unroll
        for (int u = 0; u < kU; ++u) {
            vh[u] = __builtin_bit_cast(uint4, load_tr_b128(wh_row + j + (long)u * kPer + woff));
            ve[u] = __builtin_bit_cast(uint4, load_tr_b128(we_row + j + (long)u * kPer + woff));
        }
#pragma unroll
        for (int u = 0; u < kU; ++u) chunk(j + (long)u * kPer + lane * 8, vh[u], ve[u]);
    }
    for (; j + kPer <= n; j += kPer) {
        const uint4 vh = __builtin_bit_cast(uint4, load_tr_b128(wh_row + j + woff));
        const uint4 ve = __builtin_bit_cast(uint4, load_tr_b128(we_row + j + woff));
        chunk(j + lane * 8, vh, ve);
    }
}

__global__ __launch_bounds__(kThreads)
void mtp_enter_kernel(const unsigned short* __restrict__ h, const unsigned short* __restrict__ e,
                      const float* __restrict__ wh, const float* __restrict__ we,
                      const unsigned short* __restrict__ fch,
                      const unsigned short* __restrict__ fce,
                      unsigned short* __restrict__ x,
                      long rows, long n, int hc, int flat, float eps, float wadd,
                      long hrow, long erow, long xrow) {
    extern __shared__ unsigned short sM[];        /* [hc * n] nh, then [n] ne */
    __shared__ float s_part[kWaves][kHcMax];
    __shared__ float s_pe[kWaves];
    __shared__ float s_scale[kHcMax];
    __shared__ float s_rse;
    __shared__ unsigned short s_stage[kWaves][kWave * 8];

    const long r = blockIdx.y;
    if (r >= rows) return;
    const long HN = (long)hc * n;
    unsigned short* s_nh = sM;
    unsigned short* s_ne = sM + HN;

    const unsigned short* hp = h + r * hrow;
    const unsigned short* ep = e + r * erow;
    unsigned short*       xp = x + r * xrow;

    const int tid  = (int)threadIdx.x;
    const int wave = tid / kWave;
    const int lane = tid % kWave;

    /* ---- the sums of squares. One partial a sub-stream plus the embedding's, in one pass over
     * both rows -- and the FLAT reading is the same partials added together, which is why `flat`
     * costs a branch here and nothing anywhere else. */
    float part[kHcMax];
#pragma unroll
    for (int c = 0; c < kHcMax; ++c) part[c] = 0.0f;
    float pe = 0.0f;
    /* kSsIter ITERATIONS' LOADS ARE ISSUED BEFORE THE FIRST IS SUMMED. One iteration at a time,
     * each of a thread's ten iterations waits out a load latency before the next is asked for;
     * this way two do. Same elements, same order into the same accumulators. */
    constexpr int kSsIter = 5;
    long i = tid;
    for (; i + (long)(kSsIter - 1) * kThreads < n; i += (long)kSsIter * kThreads) {
        unsigned short ev[kSsIter], hv[kSsIter][kHcMax];
#pragma unroll
        for (int q = 0; q < kSsIter; ++q) {
            const long iq = i + (long)q * kThreads;
            ev[q] = ep[iq];
#pragma unroll
            for (int c = 0; c < kHcMax; ++c)
                if (c < hc) hv[q][c] = hp[(long)c * n + iq];
        }
#pragma unroll
        for (int q = 0; q < kSsIter; ++q) {
            const float evf = bf2f(ev[q]);
            pe += evf * evf;
#pragma unroll
            for (int c = 0; c < kHcMax; ++c)
                if (c < hc) { const float v = bf2f(hv[q][c]); part[c] += v * v; }
        }
    }
    for (; i < n; i += kThreads) {
        const float ev = bf2f(ep[i]);
        pe += ev * ev;
        for (int c = 0; c < hc; ++c) {
            const float v = bf2f(hp[(long)c * n + i]);
            part[c] += v * v;
        }
    }
    for (int c = 0; c < hc; ++c) part[c] = wave_sum(part[c]);
    pe = wave_sum(pe);
    if (lane == 0) {
        for (int c = 0; c < hc; ++c) s_part[wave][c] = part[c];
        s_pe[wave] = pe;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int c = 0; c < hc; ++c) {
            float v = 0.0f;
            for (int w = 0; w < kWaves; ++w) v += s_part[w][c];
            s_scale[c] = v;
            tot += v;
        }
        if (flat) {
            const float sc = 1.0f / sqrtf(tot / (float)HN + eps);
            for (int c = 0; c < hc; ++c) s_scale[c] = sc;
        } else {
            for (int c = 0; c < hc; ++c)
                s_scale[c] = 1.0f / sqrtf(s_scale[c] / (float)n + eps);
        }
        float ve = 0.0f;
        for (int w = 0; w < kWaves; ++w) ve += s_pe[w];
        s_rse = 1.0f / sqrtf(ve / (float)n + eps);
    }
    __syncthreads();

    /* THE TRANSPOSED LAYOUT, WHEN THE SHAPE ALLOWS IT -- the same condition hc_read_a states:
     * every 256-element chunk inside one stream, and every weight row 16-byte aligned for the
     * transposing load. Otherwise the plain layout and the plain loop; both write the same values. */
    constexpr long kPer = kWave * 8;
    const bool swz = (n % kPer == 0) && (((uintptr_t)fch & 15u) == 0u)
                  && (((uintptr_t)fce & 15u) == 0u);

    /* ---- the normed vectors, staged bf16. Both are read hc + 1 times below and are what makes
     * one pass over each weight row enough. */
    {
        const float rse = s_rse;
        if (swz) {
            /* A thread owns one (chunk, lane) group: logical elements lane + 32t of its chunk,
             * written as one 16-byte store at physical lane * 8. The reads stay coalesced across
             * the wave; see hc_read_a's build for why the transpose is done here and not in LDS. */
            const int lane8 = tid & (kWave - 1);
            for (int c = -1; c < hc; ++c) {
                const unsigned short* src = c < 0 ? ep : hp + (long)c * n;
                const float* wv = c < 0 ? we : wh + (long)c * n;
                const float sc  = c < 0 ? rse : s_scale[c];
                unsigned short* dst = c < 0 ? s_ne : s_nh + (long)c * n;
                for (long base = 0; base < n; base += (long)kThreads * 8) {
                    const long q = base + (long)tid * 8;
                    if (q >= n) break;
                    const long ch = q & ~(kPer - 1);
                    unsigned short out[8];
#pragma unroll
                    for (int t = 0; t < 8; ++t) {
                        const long li = ch + lane8 + (long)t * kWave;
                        out[t] = f2bf(bf2f(src[li]) * sc * (wv[li] + wadd));
                    }
                    *(uint4*)(dst + q) = *(const uint4*)out;
                }
            }
        } else {
            for (long i = tid; i < n; i += kThreads)
                s_ne[i] = f2bf(bf2f(ep[i]) * rse * (we[i] + wadd));
            for (int c = 0; c < hc; ++c) {
                const float sc  = s_scale[c];
                const long  base = (long)c * n;
                for (long i = tid; i < n; i += kThreads)
                    s_nh[base + i] = f2bf(bf2f(hp[base + i]) * sc * (wh[base + i] + wadd));
            }
        }
    }
    __syncthreads();

    /* ---- the output channels this workgroup owns: one weight row read, hc + 1 dots off it. */
    const long i0 = (long)blockIdx.x * kMtpOutTB;
    for (int t = wave; t < kMtpOutTB; t += kWaves) {
        const long i = i0 + t;
        if (i >= n) break;
        float acc[kHcMax];
#pragma unroll
        for (int c = 0; c < kHcMax; ++c) acc[c] = 0.0f;
        float ae = 0.0f;
        if (swz) {
            wave_dot_mtp_swz<kDotU>(s_nh, s_ne, fch + i * n, fce + i * n, n, hc, lane, acc, ae);
        } else {
            wave_dot_hc<kDotU>(s_nh, n, hc, fch + i * n, n, lane, s_stage[wave], acc);
            ae = wave_dot_lds<kDotU>(s_ne, fce + i * n, n, lane, s_stage[wave]);
        }
        ae = wave_sum(ae);
        for (int c = 0; c < hc; ++c) {
            const float v = wave_sum(acc[c]);
            if (lane == 0) xp[(long)c * n + i] = f2bf(v + ae);
        }
    }
}

/* THE STATIC LDS kernel A declares beside N: the reduction planes and the staging slots. The
 * dynamic allocation for N has to fit in what is left of the workgroup's 64 KiB, so the launcher's
 * bound on hc * n is taken from this number, and the kernel asserts it matches its declarations. */
constexpr long kHcReadAStaticLds = (long)(kWaves * kHcMax + kHcMax) * (long)sizeof(float)
                                 + (long)kWaves * kWave * 8 * (long)sizeof(unsigned short);

/* ============================================================== hc_read, kernel A
 *
 * The grouped norm, then `d = silu(W_down @ N / hc)` for this workgroup's `ta` columns. Workgroup
 * 0 also produces the hc inject gains and publishes the per-stream norm scales for kernel B, which
 * cannot recompute them: kernel B reads only its own 128 channels of `h` and a stream's scale is a
 * reduction over all n of them.
 *
 * LDS holds N as bf16 -- HN * 2 bytes, 20 KiB at this model's shape -- which is what lets the
 * columns share one pass over `h` and one grouped reduction. It is held TRANSPOSED inside each
 * 256-element chunk; wave_dot_swz above is the whole argument for that, and it is the difference
 * between eight conflicted 16-bit LDS reads a chunk and one ds_load_b128. */
template <bool F8>
__global__ __launch_bounds__(kThreads)
void hc_read_a_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const void* __restrict__ downv, long dld,
                      const unsigned short* __restrict__ injw,
                      unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                      float* __restrict__ scaleout,
                      long rows, long n, int hc, long lr, long ta, long hrow, long irow,
                      float eps, float wadd) {
    extern __shared__ unsigned short sN[];          /* [hc * n], transposed per 256-element chunk */
    __shared__ float s_part[kWaves][kHcMax];
    __shared__ float s_scale[kHcMax];
    /* One 256-entry staging slot a wave, for the dot above. 4 KiB, and no barrier: a wave writes
     * and reads only its own. */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];
    static_assert(sizeof(s_part) + sizeof(s_scale) + sizeof(s_stage) == kHcReadAStaticLds,
                  "the launcher sizes N's allocation against this kernel's static LDS");

    constexpr long kPer = kWave * 8;

    const long r  = blockIdx.y;
    if (r >= rows) return;
    const long HN = (long)hc * n;
    /* THE INJECT GAINS GET A BLOCK OF THEIR OWN, AND THAT IS A LOAD-BALANCE REQUIREMENT, NOT
     * TIDINESS.
     *
     * Putting them on block 0, which also owns `ta` columns of W_down, is what a naive layout
     * does. With ta at kWaves the column loop gives every wave exactly one dot product over HN --
     * except on that block, where waves 0..hc-1 would do a SECOND one for the gains. A kernel
     * takes as long as its slowest block, so one block at 2x sets the duration of all of them, and
     * the kernel runs over three times the matvec's own cost at this shape.
     *
     * The grid is one wider instead and the last block owns no columns, so every wave in the
     * kernel does one dot. The output does not move -- the gains and the published scales are the
     * same arithmetic on the same bytes, and only which block does it changes. */
    const bool pub = (blockIdx.x == gridDim.x - 1);
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const unsigned short* hp = h + r * hrow;
    const unsigned short* __restrict__ down = (const unsigned short*)downv;

    /* THE TRANSPOSED LAYOUT IS A PROPERTY OF THE SHAPE, NOT A SWITCH. It needs the chunk to sit
     * inside one sub-stream (n % 256) and every 16-byte group to be aligned in LDS and in `h`. A
     * model that does not satisfy it keeps the plain layout and the plain loop; nothing else in
     * the kernel changes, and both write the same N. */
    /* The E4M3 dot reads N in its natural order, eight consecutive elements a lane, so the
     * transposed layout is a bf16 form's alone. */
    const bool swz = !F8 && (n % kPer == 0) && (((uintptr_t)hp   & 15u) == 0u)
                  && (((uintptr_t)down & 15u) == 0u)
                  && (!injw || ((uintptr_t)injw & 15u) == 0u);

    /* ---- the sums of squares, one stream at a time, straight from `h`. */
    /* ONE STREAM AT A TIME, which keeps the accumulator a SCALAR. Summing all hc at once needs an
     * accumulator indexed by `j / n`, and `hc` is a runtime value -- so that array lives in
     * scratch memory and every element of the row pays a private-memory round trip. */
    /* THE PREFETCH DEPTH IS THE POINT OF THE LOOP, and it is worth more than its byte count
     * suggests. This moves 20 KiB against the 320 KiB of W_down the same block reads -- and yet
     * the norm is 6.7 us of the kernel's 18.7 at the decode grid. The loop is not moving bytes, it
     * is WAITING: a runtime trip count of ten, a global load feeding an FMA, and nothing else
     * resident on the compute unit to cover the latency. Issuing eight iterations' loads into
     * registers first turns eight serial latencies into one. Bit-identical: the same element is
     * computed from the same inputs in the same order, and only when the load is issued moves.
     *
     * THE PROLOGUE IS LATENCY-BOUND, NOT TRAFFIC-BOUND, and three separate traffic levers are
     * worth nothing here:
     *
     *   LOAD WIDTH. The N-build loop below reads `h` 32 consecutive bf16 at a time and `w` 32
     *   consecutive f32 -- 64- and 128-byte requests, where wave_dot_lds measures 64 bytes at
     *   219 GB/s against 554 for the same bytes taken eight contiguous to a lane. Putting both
     *   operands through the same per-wave LDS staging turns sixteen narrow loads into three
     *   global_load_b128, bit-identically, and changes the step time by nothing. THE WIDTH
     *   ARGUMENT IS A DRAM ARGUMENT AND THESE BYTES ARE NOT COLD: `w` is one 40 KiB array every
     *   block of the grid reads and `h` is 20 KiB read by the blocks of one row, so the norm phase
     *   moves 6.7 MB across the grid in 6.7 us -- about 1 TB/s, above what this part's DRAM can
     *   deliver, so it is served out of the 64 MB last level where a narrow request costs a small
     *   fraction of what it costs at the DRAM boundary. Load width is a lever ONLY where the bytes
     *   are cold, which is why it is worth 2.5x on the dot's streamed weight and nothing here.
     *
     *   BYTE COUNT. The prologue reads `h` TWICE -- once for the sums of squares and again to
     *   build N -- on top of 40 KiB of f32 norm weight. The second read is removable: sN holds
     *   nothing until the scale pass writes it, so the first pass can park the RAW bytes there and
     *   the second read them back, bit-identically (the transposed store is a permutation inside
     *   one 256-element chunk and a wave owns a whole chunk, so it needs no barrier beyond the
     *   wave's own lockstep). That takes 80 KiB a block to 60 and is worth zero.
     *
     *   DEEPER STAGING. Prefetching the four sub-streams together, 32 loads outstanding instead of
     *   8, is also within noise.
     *
     * What is left is the LATENCY CHAIN and the two barriers: ten dependent iterations, a
     * block-wide reduction over kWaves partials, and 5.25 waves a SIMD at the decode grid with
     * nothing else resident to hide any of it. A gain here has to come from concurrency or from
     * the chain, not from traffic. */
    constexpr int kPreSs = 8;      // n / kThreads is 10 a stream at this shape, and 12 spills
    for (int c = 0; c < hc; ++c) {
        const long cb = (long)c * n;
        float acc = 0.0f;
        long  i = tid;
        for (; i + (long)(kPreSs - 1) * kThreads < n; i += (long)kPreSs * kThreads) {
            unsigned short hv[kPreSs];
#pragma unroll
            for (int p = 0; p < kPreSs; ++p) hv[p] = hp[cb + i + (long)p * kThreads];
#pragma unroll
            for (int p = 0; p < kPreSs; ++p) { const float v = bf2f(hv[p]); acc += v * v; }
        }
        for (; i < n; i += kThreads) { const float v = bf2f(hp[cb + i]); acc += v * v; }
        acc = wave_sum(acc);
        if (lane == 0) s_part[wave][c] = acc;
    }
    __syncthreads();
    if (tid < hc) {
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += s_part[k][tid];
        s_scale[tid] = __frsqrt_rn(ss / (float)n + eps);
        /* One workgroup publishes; the rest recompute the identical value from the identical
         * bytes, so which one wins is not a question. */
        if (pub && scaleout) scaleout[r * kHcMax + tid] = s_scale[tid];
    }
    __syncthreads();

    /* ---- N into LDS, narrowed to bf16 exactly where the reference narrows it. */
    /* PER STREAM, AND NOT `j / n`. `j` and `n` are both long, so `j / n` is a 64-BIT INTEGER
     * DIVISION per element -- a software sequence on this part -- to recover a value the loop
     * structure already knows. The map is elementwise, so the scale is a loop-invariant scalar
     * and every element is computed from the same inputs in the same way: bit-identical. */
    for (int c = 0; c < hc; ++c) {
        const long  cb = (long)c * n;
        const float sc = s_scale[c];
        if (swz) {
            /* A THREAD OWNS ONE (chunk, lane) GROUP, which is the transpose: its eight logical
             * elements are `lane + 32t`, and they are one 16-byte store at physical `q`. A wave's
             * 32 threads cover exactly one chunk, so `ch` is wave-uniform and the eight `h` and
             * `w` reads are each 32 consecutive elements -- still coalesced, which the transposed
             * store is not asking them to give up. */
            for (long base = 0; base < n; base += (long)kThreads * 8) {
                const long q = base + (long)tid * 8;
                if (q >= n) break;
                const long ch = cb + (q & ~(kPer - 1));
                unsigned short out[8];
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const long li = ch + lane + (long)t * kWave;
                    out[t] = f2bf(bf2f(hp[li]) * sc * (w[li] + wadd));
                }
                *(uint4*)(sN + cb + q) = *(const uint4*)out;
            }
        } else {
            constexpr int kPreScale = 8;
            long i = tid;
            for (; i + (long)(kPreScale - 1) * kThreads < n; i += (long)kPreScale * kThreads) {
                float          wv[kPreScale];
                unsigned short hv[kPreScale];
#pragma unroll
                for (int p = 0; p < kPreScale; ++p) {
                    wv[p] = w[cb + i + (long)p * kThreads];
                    hv[p] = hp[cb + i + (long)p * kThreads];
                }
#pragma unroll
                for (int p = 0; p < kPreScale; ++p)
                    sN[cb + i + (long)p * kThreads] = f2bf(bf2f(hv[p]) * sc * (wv[p] + wadd));
            }
            for (; i < n; i += kThreads)
                sN[cb + i] = f2bf(bf2f(hp[cb + i]) * sc * (w[cb + i] + wadd));
        }
    }
    __syncthreads();

    /* ---- this workgroup's columns of W_down. One wave a column, a lane striding by 32. */
    const float inv_hc = 1.0f / (float)hc;
    const long  k0 = (long)blockIdx.x * ta;
    for (long k = k0 + wave; k < k0 + ta && k < lr; k += kWaves) {
        float acc;
        if constexpr (F8) {
            acc = hc8_dot_down<kDotU>(sN, (const unsigned char*)downv + k * dld, HN, lane);
        } else {
            const unsigned short* wr = down + k * HN;
            acc = swz ? wave_dot_swz<kDotU>(sN, wr, HN, lane)
                      : wave_dot_lds<kDotU>(sN, wr, HN, lane, s_stage[wave]);
        }
        acc = wave_sum(acc);
        if (lane == 0) dout[r * lr + k] = f2bf(siluf_(acc * inv_hc));
    }

    /* ---- the inject gains, on the block that owns no columns and already holds N. */
    if (pub && injw && inj) {
        for (int c = wave; c < hc; c += kWaves) {
            const unsigned short* wr = injw + (long)c * HN;
            float acc = swz ? wave_dot_swz<kDotU>(sN, wr, HN, lane)
                            : wave_dot_lds<kDotU>(sN, wr, HN, lane, s_stage[wave]);
            acc = wave_sum(acc);
            if (lane == 0) inj[r * irow + c] = f2bf(2.0f * sigmoidf_(acc * inv_hc));
        }
    }
}

/* ============================================================== hc_read, kernel B1
 *
 * `u = sigmoid(W_up @ d)`, one wave a row of W_up. SEPARATE FROM THE MIX, and the reason is
 * occupancy rather than tidiness: the mix owns 128 CHANNELS, which is hc*128 rows of W_up, so a
 * 2560-wide model gives it twenty workgroups against sixty-four compute units and the matvec runs
 * at a quarter of the machine. Split, this one is HN/64 = 160 workgroups on the same bytes.
 *
 * `u` costs a round trip through scratch -- 20 KiB a row written and read -- against 6.55 MiB of
 * weight, which is the trade.
 *
 * ---- RG: THE ROWS OF A STEP SHARE ONE BLOCK, NOT ONE BLOCK EACH -------------------------------
 *
 * Putting the row on `blockIdx.y` reads W_up ONCE A ROW. At decode that is the floor and the
 * file's header says so -- but a depth-3 MTP verify is FOUR rows, and four blocks each streaming
 * the same 6.55 MB is 26.2 MB for 6.55 MB of weight. At M=4 that form runs 640 blocks at 1120 GB/s
 * aggregate -- above what the part's DRAM can deliver, so three of the four reads are already
 * being served out of the 64 MB last level. That is why the cost is 1.4x and not 4x, and also why
 * it is worth removing rather than free: the cache is doing work that need not exist, and every
 * byte of it evicts something else.
 *
 * RG rows share ONE pass over the weight. `d` is 320 floats a row -- 1.25 KiB -- so all of them
 * fit in LDS beside the staging slots, and a wave that has the weight row in registers can take
 * RG dot products off it instead of one. The weight traffic drops to 6.55 MB whatever M is, worth
 * about 3% of a multi-row step for identical output, and RG=1 is the one-row case.
 *
 * ---- THIS KERNEL IS AT 440 GB/s, AND FIVE LEVERS ON THAT ARE SLOWER ---------------------------
 *
 * It moves 6.55 MB of W_up in 14.9 us at the decode grid -- 440 GB/s against a part that reads
 * 621 cold. A wave owns eight rows of 640 bytes and takes them ONE at a time, so it has one
 * 512-byte load outstanding, and 160 blocks of 8 waves on 64 compute units is ~10 KiB in flight a
 * CU. That reads exactly like the depth argument wave_dot_lds makes, and it is not. Each of the
 * following is SLOWER, with hc_read_a flat in the same traces as a control:
 *
 *   DEPTH. Four weight rows in flight instead of one (the eight rows are independent, so four
 *   uint4 of register buys four times the bytes outstanding): 17.5 us, worse by 16%.
 *
 *   GRID, IN BOTH DIRECTIONS. kUpRows 64 -> 16 is four times the blocks at the same depth a wave:
 *   15.9 us here, and 112 -> 131 us at prefill. kUpRows 64 -> 128 is HALF the blocks -- 80 of
 *   them, which is where hc_read_a's faster 84 sits -- and is worse again.
 *
 *   LOAD WIDTH AT THE TAIL. `lr` is 320 and a chunk is 256, so wave_dot_lds_rows takes 256
 *   elements as one 512-byte staged load and the last 64 as TWO 64-byte ones. Eight lanes bringing
 *   a uint4 each is the same 128 bytes in ONE request and is 4.6% WORSE. The 219-against-554
 *   figure elsewhere in this file does not reach this tail, because that is a REQUEST-RATE effect
 *   across a 20 KiB row, where the narrow form pays the per-request cost forty times over. A 64-element tail is TWO requests and both land in the SAME 128-byte line -- a
 *   640-byte row is five lines, so `wr + 256` is line-aligned and the second load is an L1 hit.
 *   There is no transaction to save, and the wide form costs one: the scalar pair issues two loads
 *   and waits once, where the staged form issues one load and then waits on a dependent LDS round
 *   trip that eight lane-FMAs cannot hide.
 *
 *   PER-WAVE SEQUENTIALITY. `for (t = wave; t < kUpRows; t += kWaves)` gives a wave rows 5120
 *   bytes apart, where hc_read_a -- the kernel at 546 GB/s -- gives each of its waves ONE 20 KiB
 *   row to walk end to end. Giving each wave eight CONSECUTIVE rows instead, same block and same
 *   bytes, is 9.9% worse. SO THE LOCALITY THAT MATTERS IS ACROSS THE EIGHT WAVES OF A BLOCK AT ONE
 *   INSTANT, NOT ALONG ONE WAVE'S OWN SEQUENCE: strided, the block's eight waves work on eight
 *   ADJACENT rows at the same time and their loads arrive as one contiguous 5 KiB burst;
 *   consecutive, the same eight waves sit 5 KiB apart and the block asks the memory system for
 *   eight scattered pieces. The strided form in the code is the one the part wants.
 *
 *   THE TAIL'S STALL COUNT. At lr = 320 the tail is TWO iterations of a rolled loop the compiler
 *   cannot unroll, because HN is a runtime value, and each is `global_load_u16`,
 *   `s_wait_loadcnt 0x0`, four VALU -- so a row costs THREE dependent global stalls. Issuing the
 *   tail into registers first (an unrolled `tv[8]` with `(i < HN)` masking, one waitcnt serving
 *   all of them) does turn three stalls into two, and is 13.1% worse: the compiler turns the
 *   masked unrolled load into EIGHT conditional basic blocks and sinks the single
 *   `s_wait_loadcnt 0x0` to just after the last of them, BEFORE the chunk's load is issued. The
 *   tail's latency stops being something the chunk's FMAs retire under and becomes a naked stall
 *   at the front, plus six branches a row that do nothing. FEWER STALLS ON A LATENCY-BOUND KERNEL
 *   IS NOT THE SAME AS LESS STALLING.
 *
 * NOTHING THAT CHANGES WHAT ONE WAVE ASKS FOR MOVES THIS KERNEL. A sixth lever would have to be
 * about what the BLOCK asks for, and the arithmetic is at least expressible -- kUpRows * lr * 2
 * is 40960 bytes, exactly ten block-wide passes of 256 threads by 16 bytes, so a cooperative
 * stage of the whole span is one clean loop. It is also 40 KiB of LDS against the 9 KiB this
 * kernel holds, which is 1 block a CU instead of 7, at an occupancy cost that is not measured.
 *
 * BIT-IDENTICAL, and for the same reason wave_dot_lds is: lane L still sums j = L, L+32, ...
 * within a 256-element chunk, still takes the chunks in ascending order, and still accumulates
 * into a per-row scalar. Only the LOOP NEST changes -- rows move from the grid to the innermost
 * position -- and `acc[]` is indexed by an unrolled constant so it stays in registers. A runtime
 * `rows` bound there would put it in scratch, which is the trap hc_write's header records. */

/* RG rows of the LDS vector against ONE contiguous bf16 weight row, the weight read once.
 * `sv_ld` is the row pitch of `sv`; `live` masks the tail group of a row count RG does not
 * divide. */
template <int RG>
__device__ __forceinline__ void wave_dot_lds_rows(const float* __restrict__ sv, long sv_ld,
                                                  const unsigned short* __restrict__ wr, long HN,
                                                  int lane, unsigned short* stage,
                                                  float* __restrict__ acc) {
    constexpr long kPer = kWave * 8;
#pragma unroll
    for (int r = 0; r < RG; ++r) acc[r] = 0.0f;
    long j = 0;
    if (((uintptr_t)wr & 15u) == 0u) {
        for (; j + kPer <= HN; j += kPer) {
            *(uint4*)(stage + lane * 8) = *(const uint4*)(wr + j + lane * 8);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int   jj = lane + t * kWave;
                const float wv = bf2f(stage[jj]);
#pragma unroll
                for (int r = 0; r < RG; ++r) acc[r] += sv_at(sv, (long)r * sv_ld + j + jj) * wv;
            }
        }
    }
    for (long i = j + lane; i < HN; i += kWave) {
        const float wv = bf2f(wr[i]);
#pragma unroll
        for (int r = 0; r < RG; ++r) acc[r] += sv_at(sv, (long)r * sv_ld + i) * wv;
    }
}

template <int RG, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_up_kernel(const void* __restrict__ upv, long uld, const unsigned short* __restrict__ dbuf,
                  unsigned short* __restrict__ ubuf, long rows, long hn, long lr) {
    const unsigned short* __restrict__ up = (const unsigned short*)upv;
    extern __shared__ float sd[];                   /* [RG * lr] */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];   /* wave_dot_lds, one slot a wave */
    const long r0 = (long)blockIdx.y * RG;
    if (r0 >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;

    /* A dead row of the tail group reads nothing and is zero-filled, so its dot is defined and
     * its store is dropped below. */
    for (long k = tid; k < (long)RG * lr; k += kThreads) {
        const long r = k / lr;
        sd[k] = (r0 + r < rows) ? bf2f(dbuf[(r0 + r) * lr + (k - r * lr)]) : 0.0f;
    }
    __syncthreads();

    const long j0 = (long)blockIdx.x * kUpRows;
    /* THE E4M3 FORM TAKES A ROW FOUR THREADS AT A TIME, a quarter each -- the quarter is the row's
     * scale group -- so the block's 256 threads cover its 64 rows in one pass. */
    if constexpr (F8) {
        static_assert(kUpRows * 4 == kThreads, "four threads a row of W_up");
        const int  tr = tid >> 2, q4 = tid & 3;
        const long j  = j0 + tr;
        float acc[RG];
        hc8_dot_up<RG>(sd, lr, (const unsigned char*)upv + (j < hn ? j : hn - 1) * uld, q4, acc);
#pragma unroll
        for (int r = 0; r < RG; ++r)
            if (q4 == 0 && j < hn && r0 + r < rows) ubuf[(r0 + r) * hn + j] = f2bf(sigmoidf_(acc[r]));
        return;
    }
    for (long t = wave; t < kUpRows; t += kWaves) {
        const long j = j0 + t;
        if (j >= hn) break;
        float acc[RG];
        wave_dot_lds_rows<RG>(sd, lr, up + j * lr, lr, lane, s_stage[wave], acc);
#pragma unroll
        for (int r = 0; r < RG; ++r) {
            const float a = wave_sum(acc[r]);
            /* Narrowed where the reference narrows it: torch's sigmoid output is bf16 before the
             * multiply, so an f32 `u` here would disagree in the last bit of every channel. */
            if (lane == 0 && r0 + r < rows) ubuf[(r0 + r) * hn + j] = f2bf(sigmoidf_(a));
        }
    }
}

/* THE ROTATED TWIN OF ONE 128-CHANNEL GROUP, for a consumer whose weight carries a 128-wide
 * Hadamard (the w4a8h expert plane). A wave holds the group four consecutive values a lane -- the
 * layout r4d_fwht_wave<128> transforms in registers -- loaded by the caller from the bf16 the read
 * stored, and quantises the rotated values exactly as the rotated quant_act_fp8 kernel does: the
 * same transform, absmax, scale, clamp and packing, so its consumer cannot tell which op wrote
 * the codes and no separate launch over `x` is needed. */
__device__ __forceinline__ void hc_rot_group(float* v, int lane, unsigned char* qg, float* sg) {
    r4d_fwht_wave<kUpTB>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
    m = r4d_wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f;
    if (lane == 0) *sg = sc;
    const float inv = 1.0f / sc;
    int pk = 0;
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[0] * inv, -448.0f), 448.0f),
                                         fminf(fmaxf(v[1] * inv, -448.0f), 448.0f), pk, false);
    pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(v[2] * inv, -448.0f), 448.0f),
                                         fminf(fmaxf(v[3] * inv, -448.0f), 448.0f), pk, true);
    *(unsigned*)(qg + lane * 4) = (unsigned)pk;
}

/* ============================================================== hc_read, kernels B and B2 as one
 *
 * hc_up tiles the FLATTENED index, so a workgroup's 64 rows of W_up are 64 channels of ONE
 * sub-stream, the hc terms of a channel's mean land in hc different workgroups, and `u` has to
 * round-trip through `ubuf` into hc_mix. That second launch is a whole dependent dispatch a
 * connection -- on this part a gap longer than either kernel's own work at one row.
 *
 * This workgroup owns 64 / hc CHANNELS in all hc sub-streams instead: the same 64 rows of W_up and
 * the same grid, taken as hc runs of 64 / hc rows, so every term of its channels' means is in its
 * own LDS and `u` never leaves it. What it gives up is the whole 128-channel scale group, and an
 * arrival counter buys that back exactly as hc_up_mix_tiled does: the last of the workgroups
 * covering a group of a row group quantises it, from the bf16 `x` the others wrote.
 *
 * BIT-IDENTICAL to the pair. Each row's dot is wave_dot_lds_rows over the same weight row,
 * narrowed where hc_up narrows it; the mean runs over c ASCENDING in hc_mix's expression; and the
 * codes are taken from the same bf16 values against the same amax, which a max reaches in any
 * order. */
template <int RG, bool F8, bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_kernel(const void* __restrict__ upv, long uld,
                      const unsigned short* __restrict__ dbuf,
                      const unsigned short* __restrict__ h, const float* __restrict__ w,
                      const float* __restrict__ scalein,
                      unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                      float* __restrict__ qs, unsigned char* __restrict__ rq,
                      float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                      long rows, long n, int hc, long lr, long hrow, long xrow,
                      long qrow, long srow, long rqrow, long rsrow, float wadd) {
    extern __shared__ float sd[];                   /* [RG * lr] */
    __shared__ unsigned short s_stage[kWaves][kWave * 8];   /* wave_dot_lds, one slot a wave */
    __shared__ unsigned short s_u[RG * kUpRows];
    const unsigned short* __restrict__ up = (const unsigned short*)upv;
    const long r0 = (long)blockIdx.y * RG;
    if (r0 >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const int  ipb = kUpRows / hc;                  /* channels this workgroup owns */
    const long i0  = (long)blockIdx.x * ipb;

    for (long k = tid; k < (long)RG * lr; k += kThreads) {
        const long r = k / lr;
        sd[k] = (r0 + r < rows) ? bf2f(dbuf[(r0 + r) * lr + (k - r * lr)]) : 0.0f;
    }
    __syncthreads();

    /* Row t of this workgroup's slice of W_up is sub-stream t / ipb of channel i0 + t % ipb. The
     * E4M3 form takes each row four threads at a time, as hc_up_kernel's does. */
    if constexpr (F8) {
        static_assert(kUpRows * 4 == kThreads, "four threads a row of W_up");
        const int  t = tid >> 2, q4 = tid & 3;
        const long j = (long)(t / ipb) * n + i0 + t % ipb;
        float acc[RG];
        hc8_dot_up<RG>(sd, lr, (const unsigned char*)upv + j * uld, q4, acc);
#pragma unroll
        for (int r = 0; r < RG; ++r)
            if (q4 == 0) s_u[r * kUpRows + t] = f2bf(sigmoidf_(acc[r]));
    } else {
        for (int t = wave; t < kUpRows; t += kWaves) {
            const long j = (long)(t / ipb) * n + i0 + t % ipb;
            float acc[RG];
            wave_dot_lds_rows<RG>(sd, lr, up + j * lr, lr, lane, s_stage[wave], acc);
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                const float a = wave_sum(acc[r]);
                if (lane == 0) s_u[r * kUpRows + t] = f2bf(sigmoidf_(a));
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    for (int idx = tid; idx < RG * ipb; idx += kThreads) {
        const int  rl = idx / ipb, ii = idx - rl * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const long i = i0 + ii;
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) {
            const long j = (long)c * n + i;
            const float Nv = nbf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c]
                                 * (w[j] + wadd));
            acc += bf2f(s_u[rl * kUpRows + c * ipb + ii]) * Nv;
        }
        acc *= inv_hc;
        x[r * xrow + i] = f2bf(acc);
    }
    if (!q || !ctr) return;

    /* The group's last workgroup quantises it: kUpTB / ipb of them cover one group of a row group. */
    const int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    const unsigned slot = blockIdx.y * (unsigned)(n / kUpTB) + (unsigned)(i0 / kUpTB);
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != (unsigned)nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* One wave a row: RG is at most the wave count, and a wave's 32 lanes take the group's 128
     * channels four apiece, which is also one packed dword of codes a lane. */
    if (wave >= RG) return;
    const long r = r0 + wave;
    if (r >= rows) return;
    const long base = (i0 / kUpTB) * kUpTB;
    const unsigned short* const xr = x + r * xrow + base;
    float m = 0.0f;
    for (int t = lane; t < kUpTB; t += kWave) m = fmaxf(m, fabsf(bf2f(xr[t])));
    m = r4d_wave_max(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * srow + base / kUpTB] = sc;
    for (int d = lane; d < kUpTB / 4; d += kWave) {
        const int c4 = d * 4;
        const uint32_t pk = q8_pack4<I8>(bf2f(xr[c4 + 0]), bf2f(xr[c4 + 1]), bf2f(xr[c4 + 2]),
                                         bf2f(xr[c4 + 3]), inv);
        *(unsigned*)(q + r * qrow + base + c4) = (unsigned)pk;
    }
    if (rq) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[lane * 4 + j]);
        hc_rot_group(v, lane, rq + r * rqrow + base, rqs + r * rsrow + base / kUpTB);
    }
}

/* THE GROUP SIZE IS THE ROW COUNT, ROUNDED DOWN TO AN INSTANTIATION. RG is a template parameter
 * because `acc[RG]` has to be registers; a runtime bound makes it scratch, and one private-memory
 * round trip an element would cost more than the weight re-read it is here to remove.
 *
 * The LDS cap is the real bound: RG rows of `d` are RG*lr floats beside the 4 KiB of staging
 * slots, so a model with a wide low rank gets a smaller group rather than a launch failure. */

static inline long hc_up_rg(long long rows, long long lr) {
    constexpr long kLdsCap = 60 * 1024;          /* leave the 4 KiB of staging slots room */
    long rg = (rows >= 8) ? 8 : (rows >= 4) ? 4 : (rows >= 2) ? 2 : 1;
    while (rg > 1 && (long long)rg * lr * (long long)sizeof(float) > kLdsCap) rg >>= 1;
    return rg;
}

static inline int hc_up_launch(const RadTensor* up, long uld, bool f8, const unsigned short* dbuf,
                               unsigned short* ubuf, long long rows, long long HN, long long lr,
                               const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const long rg = hc_up_rg(rows, lr);
    const dim3 grid((unsigned)((HN + kUpRows - 1) / kUpRows),
                    (unsigned)((rows + rg - 1) / rg));
    const unsigned lds = (unsigned)(rg * lr * sizeof(float));
    const void* upp = up->data;
#define HC_UP1_CASE(RG_, F8_)                                                                    \
    emit_rc = R4D_EMIT(sk, (hc_up_kernel<RG_, F8_>), grid, dim3(kThreads), lds, upp, uld, dbuf,   \
                       ubuf, (long)rows, (long)HN, (long)lr)
    switch (rg) {
        case 8:  if (f8) HC_UP1_CASE(8, true); else HC_UP1_CASE(8, false); break;
        case 4:  if (f8) HC_UP1_CASE(4, true); else HC_UP1_CASE(4, false); break;
        case 2:  if (f8) HC_UP1_CASE(2, true); else HC_UP1_CASE(2, false); break;
        default: if (f8) HC_UP1_CASE(1, true); else HC_UP1_CASE(1, false); break;
    }
#undef HC_UP1_CASE
    return emit_rc;
}

/* hc_up_mix_kernel's launch: hc_up's grid and row group, which the counter slots are laid over --
 * (rows / rg) row groups of n / kUpTB scale groups each. */
static inline int hc_up_mix_launch(const RadTensor* up, long uld, bool f8, bool i8,
                                   const unsigned short* dbuf,
                                   const unsigned short* h, const float* w, const float* scale,
                                   unsigned short* x, unsigned char* q, float* qs,
                                   unsigned char* rq, float* rqs,
                                   unsigned int* ctr, long long rows, long long n, int hc,
                                   long long lr, long long hrow, long long xrow, long long qrow,
                                   long long srow, long long rqrow, long long rsrow, float wadd,
                                   const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const long rg = hc_up_rg(rows, lr);
    const dim3 grid((unsigned)(n / (kUpRows / hc)), (unsigned)((rows + rg - 1) / rg));
    const unsigned lds = (unsigned)(rg * lr * sizeof(float));
    const void* upp = up->data;
#define HC_UPMIX_CASE3(RG_, F8_, I8_)                                                            \
    emit_rc = R4D_EMIT(sk, (hc_up_mix_kernel<RG_, F8_, I8_>), grid, dim3(kThreads), lds, upp, uld,\
                       dbuf, h, w, scale, x, q, qs, rq, rqs, ctr, (long)rows, (long)n, hc,        \
                       (long)lr, (long)hrow, (long)xrow, (long)qrow, (long)srow, (long)rqrow,     \
                       (long)rsrow, wadd)
#define HC_UPMIX_CASE(RG_, F8_)                                                                  \
    do { if (i8) HC_UPMIX_CASE3(RG_, F8_, true); else HC_UPMIX_CASE3(RG_, F8_, false); } while (0)
    switch (rg) {
        case 8:  if (f8) HC_UPMIX_CASE(8, true); else HC_UPMIX_CASE(8, false); break;
        case 4:  if (f8) HC_UPMIX_CASE(4, true); else HC_UPMIX_CASE(4, false); break;
        case 2:  if (f8) HC_UPMIX_CASE(2, true); else HC_UPMIX_CASE(2, false); break;
        default: if (f8) HC_UPMIX_CASE(1, true); else HC_UPMIX_CASE(1, false); break;
    }
#undef HC_UPMIX_CASE
#undef HC_UPMIX_CASE3
    return emit_rc;
}

/* ============================================================== hc_read, kernel B2
 *
 * `x = mean_c( u[c] * N[c] )` for this workgroup's 128 channels, and their fp8 codes -- 128 is one
 * scale group, so the amax is this workgroup's alone and there is no second pass over the row.
 *
 * N is rebuilt from `h` and the scales kernel A published rather than carried: 128 channels of it
 * is 512 elements against the 20 KiB the whole row would cost to move. */
template <bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_mix_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                   const unsigned short* __restrict__ ubuf, const float* __restrict__ scalein,
                   unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                   float* __restrict__ qs, unsigned char* __restrict__ rq,
                   float* __restrict__ rqs,
                   long rows, long n, int hc, long hrow, long xrow,
                   long qrow, long srow, long rqrow, long rsrow, float wadd) {
    __shared__ float s_x[kUpTB];
    __shared__ float s_amax[kWaves];

    const long r = blockIdx.y;
    if (r >= rows) return;
    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long i0  = (long)blockIdx.x * kUpTB;
    const long hn  = (long)hc * n;

    const float inv_hc = 1.0f / (float)hc;
    if (tid < kUpTB) {
        const long i = i0 + tid;
        float acc = 0.0f;
        if (i < n) {
            for (int c = 0; c < hc; ++c) {
                const long j = (long)c * n + i;
                const float Nv = nbf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c]
                                     * (w[j] + wadd));
                acc += bf2f(ubuf[r * hn + j]) * Nv;
            }
            acc *= inv_hc;
            x[r * xrow + i] = f2bf(acc);
        }
        s_x[tid] = acc;
    }
    if (!q) return;
    __syncthreads();

    /* IT QUANTISES WHAT WAS WRITTEN: `s_x` is narrowed to bf16 first, because that is what the
     * reference reads back and a fused kernel using its f32 accumulator would disagree in the last
     * bit of every code. */
    float m = 0.0f;
    for (int t = tid; t < kUpTB; t += kThreads) m = fmaxf(m, fabsf(nbf(s_x[t])));
    m = r4d_wave_max(m);
    if (lane == 0) s_amax[wave] = m;
    __syncthreads();
    float amax = 0.0f;
    for (int k = 0; k < kWaves; ++k) amax = fmaxf(amax, s_amax[k]);
    const float sc  = amax > 0.0f ? amax * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (tid == 0) qs[r * srow + blockIdx.x] = sc;
    /* Four codes a lane, packed into one dword, exactly as r4d_quant_act_fp8 does it. */
    if (tid < kUpTB / 4) {
        const long c0 = i0 + (long)tid * 4;
        float v[4];
        for (int j = 0; j < 4; ++j) v[j] = (c0 + j) < n ? nbf(s_x[tid * 4 + j]) : 0.0f;
        const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
        if (c0 < n) *(unsigned*)(q + r * qrow + c0) = (unsigned)pk;
    }
    /* The rotated twin from the same narrowed values; the launch refuses a channel count that is
     * not whole groups, so the group is always full. */
    if (rq && wave == 0) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = nbf(s_x[lane * 4 + j]);
        hc_rot_group(v, lane, rq + r * rqrow + i0, rqs + r * rsrow + blockIdx.x);
    }
}

/* ==================================================== hc_read, the PREFILL path
 *
 * The three kernels above PUT ROWS ON THE GRID: one workgroup a token, and each of them reads the
 * whole of its slice of the mixing matrix. At decode that is exactly right -- there is one row, the
 * weight is read once, and the header's 2.1 ms is the floor. At a prefill chunk it is a disaster:
 * EVERY TOKEN re-reads both 6.55 MB matrices, so 2048 tokens times 96 connections times 13.1 MB is
 * 2.5 TB of weight traffic for a step whose arithmetic is 6.6M MACs a token. On the full Flash-Next
 * that is 56% of a prefill step, nearly twice the MoE.
 *
 * Above a row count the two products run as TILED GEMMs instead. A workgroup owns 128 rows and 64
 * output columns; the weight tile lands in LDS once and is reused by all 128 rows, so the weight is
 * read ceil(rows/128) times a step rather than `rows` times -- 16 passes at a 2048-token chunk
 * instead of 2048. The arithmetic is the same two products; only the ORDER of the sum changes, from
 * a lane-strided partial sum to WMMA's 16-wide fragments, which is the same class of reordering the
 * per-row form already does against the reference's sequential f32 sum.
 *
 * THE NORM MOVES OUT rather than being carried. The per-row kernel A folds the grouped rmsnorm, the
 * inject gains and the W_down product into one pass because it already holds N in LDS. A tile
 * cannot: N for 128 rows is 2.5 MB. So the scales and the gains get a kernel of their own (one pass
 * over `h`, and injw is 80 KB that stays in L2), and the GEMM rebuilds N as it stages each A tile --
 * which costs nothing, because the tile has to be staged either way and `h` would be read anyway.
 *
 * THE COLUMN COUNT IS WHY THIS IS TWO GEMMS AND NOT ONE. `d` is [rows, 320] and `u` is [rows,
 * 10240]; a workgroup that owned a whole 128-channel scale group of the mix would own hc*128 = 512
 * rows of W_up for only its 16 affordable rows of accumulator, and the weight would then be read
 * rows/16 times. What keeps `u` out of memory instead is a workgroup owning sixteen channels of
 * every sub-stream and an arrival counter for the scale group: hc_up_mix_tiled_kernel at the decode
 * tiles, hc_up_mix_wide_kernel at the prefill one.
 */
constexpr int kTNT = 64;            // output columns a workgroup owns: four n tiles
/* AND IT IS FITTED AT DECODE SHAPES, WHICH COSTS SOMETHING AT PREFILL. The tiled down product's
 * grid is (ncol/kTNT, rows/rt, HN/kc), so `h` is re-read once per COLUMN tile and the weight once
 * per ROW tile. At decode h is 20 KiB against 1.3 MiB of weight and six reads of it are free; at a
 * 2048-row chunk h is 41.9 MB and six reads are 251 MB against 105 MB of weight. The trade
 * inverts and nothing here notices.
 *
 * So the prefill tile does not use it: hc_down_wide_kernel takes kWideNT columns a workgroup.
 * The K-split stays hc_down_kchunk's, counted at kTNT columns: at 2048 rows and 112 columns, splits
 * of 1, 3, 4, 5 and 8 are all slower than the rule's 2 (844 / 797 / 774 / 765 / 766 us a call
 * against 760), and keeping it keeps the bytes. No split at all loses because the grid is then 48
 * workgroups on 64 CUs. */
constexpr int kTKT = 64;            // K staged per pass
constexpr int kTAP = kTKT + 4;      // LDS row pitch in shorts. 136 B: 8-byte aligned, and the 16
                                    // lanes of a fragment land on 16 distinct bank pairs (AP 68,
                                    // the same pitch the kkt solve uses).
constexpr long kTileMinRows = 64;   // at or above this the 128-row tile fills and the down product
                                    // needs no K-split: the prefill form.
constexpr long kDecMinRows  = 8;    // ...and below THIS the per-row kernels still win: there is not
                                    // enough re-read to pay for hc_scale_inj and the partials.
                                    // Eight rows is the crossover -- the two forms are level there
                                    // -- and above it the tiled form pulls away, reaching ~25%
                                    // of the step by eight concurrent sequences.

/* ==================================================== the norm scales, alone
 *
 * THE SCALES GET A GRID, NOT A WORKGROUP. The inject gains ride along as four more columns of the
 * down product (see hc_down_tiled_kernel), so nothing here needs a shared LDS copy of N -- a form
 * that computed the per-stream rsqrt AND the gains off one such copy would have to stage the whole
 * 20 KiB row and walk it a second time for four dot products.
 *
 * With the gains gone the streams are independent, so the grid is (hc, rows) rather than
 * (1, rows): one rsqrt per (row, stream), four times the workgroups, and no LDS staging in front
 * of them. The fused form's 75 VGPRs and 4384 B of LDS are 9 and 32 here.
 *
 * BIT-IDENTICAL to the fused form: the same thread stride over the same values, the same
 * wave_sum, and the same ascending sum over the eight wave partials.
 *
 * IT IS WORTH TENS OF MICROSECONDS A STEP AND NOT MORE. A kernel this short is mostly the ~20%
 * launch ramp this part charges anything under ~13 us, so deleting its WORK can never buy much.
 * Only deleting the LAUNCH would, and that needs the scales computed by something already running
 * -- which the op's private scratch forbids, since the producer of `h` is hc_write, a different op
 * with a different arena.
 *
 * THE LOADS GO IN BATCHES OF kPre, then the tail one at a time: at this model's n = 2560 that is
 * all ten of a thread's loads in flight at once instead of ten dependent trips. The same element
 * in the same order, so the same scale (hc-read-tp4's PRE form, one batch wide enough for n). */
__global__ __launch_bounds__(kThreads)
void hc_norm_scale_kernel(const unsigned short* __restrict__ h, float* __restrict__ scaleout,
                          long rows, long n, long hrow, float eps) {
    __shared__ float s_part[kWaves];
    const long r = blockIdx.y, c = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const unsigned short* hp = h + r * hrow + c * n;
    constexpr int kPre = 10;
    float acc = 0.0f;
    long  i = tid;
    for (; i + (long)(kPre - 1) * kThreads < n; i += (long)kPre * kThreads) {
        unsigned short hv[kPre];
#pragma unroll
        for (int p = 0; p < kPre; ++p) hv[p] = hp[i + (long)p * kThreads];
#pragma unroll
        for (int p = 0; p < kPre; ++p) { const float v = bf2f(hv[p]); acc += v * v; }
    }
    for (; i < n; i += kThreads) { const float v = bf2f(hp[i]); acc += v * v; }
    acc = wave_sum(acc);
    if (lane == 0) s_part[wave] = acc;
    __syncthreads();
    if (tid == 0) {
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += s_part[k];
        scaleout[r * kHcMax + c] = __frsqrt_rn(ss / (float)n + eps);
    }
}

/* ==================================================== THE TILE SHAPE IS A FUNCTION OF THE ROWS
 *
 * One tile shape cannot serve a 2048-token prefill chunk and a 32-row decode batch. RT is what
 * varies: 128 rows a workgroup is right when there are thousands, and at 32 it stages 128 rows
 * of A to use 32 of them AND leaves the down product's grid at ceil(320/64) = FIVE workgroups on
 * a 64-CU part. Forcing the 128-row tile on at decode roughly doubles the step at one sequence
 * and costs ~18% at eight -- which is what `kTileMinRows` protects against, and why simply
 * lowering that bound is not the fix.
 *
 * WHICH WAVE OWNS WHICH TILE FOLLOWS FROM RT. With 8 m-tiles there is one a wave and each wave
 * walks every n-tile, which is the 128-row form's mapping. With 2 m-tiles the same rule would idle
 * six waves, so the waves spread over N instead: WM = min(MT, waves) waves across the m-tiles and
 * WN = waves/WM across the n-tiles. RT 128 gives exactly the one-m-tile-a-wave mapping (WM 8,
 * WN 1).
 *
 * AND THE DOWN PRODUCT NEEDS THE K-SPLIT THAT THE UP PRODUCT DOES NOT. Its output is [rows, 320]
 * -- five 64-column tiles, one workgroup each -- while the up product's is [rows, 10240] and
 * already covers the part at 160. So the down product splits K over blockIdx.z, writes f32
 * partials, and a reduce pass sums them in ASCENDING z and applies the silu. The partial buffer
 * is ksplit * rows * lr floats, 655 KB at the decode band, and the reduce reads it once.
 */
constexpr int kTKSPLIT_WANT = 128;   // workgroups the down product aims to cover the part with

/* ==================================================== d = silu(W_down @ N / hc), tiled
 *
 * C[RT rows, NT cols] over K = hc*n. The A tile is N, rebuilt from `h` as it is staged: the launch
 * refuses `n % kTKT`, so a K slice never straddles a stream and the scale index is one divide a
 * tile rather than one an element -- the same reason the per-row kernel walks streams outermost.
 *
 * `part` non-null is the split-K form: this block owns K in [z*kc, min(HN, (z+1)*kc)) and writes
 * its raw f32 accumulator; the silu and the /hc move to the reduce. With `part` null, z is 0, the
 * block owns all of K and the epilogue applies them itself.
 *
 * THE INJECT GAINS RIDE ALONG AS FOUR MORE COLUMNS. `inj[c] = 2 sigmoid(W_inj[c] . N / hc)` is a
 * dot of the SAME A matrix this GEMM already has staged, over the same K, and W_inj is [hc, HN] --
 * four more rows of a weight whose other 320 rows are W_down. Computed on its own it costs a whole
 * kernel -- a (1, rows) grid in which four of eight waves walk a 20 KiB row each, ~18 us a call at
 * eight sequences for 3.2 MB. Folded here it is 4/324 of a kernel that is already at bandwidth,
 * and the norm kernel is left with one pass over `h`. `ncol` is lr + hc when there are gains and
 * lr when there are not; columns at or past lr read injw and take the sigmoid epilogue. */
template <int RT, int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_down_tiled_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                          const void* __restrict__ downv, long dld,
                          const float* __restrict__ scalein,
                          const unsigned short* __restrict__ injw,
                          unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                          float* __restrict__ part,
                          long rows, long n, int hc, long lr, long ncol, long hrow, long irow,
                          long kc, float wadd) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long HN = (long)hc * n;
    const long r0 = (long)blockIdx.y * RT;
    const long c0 = (long)blockIdx.x * NT;
    const long ka = (long)blockIdx.z * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || c0 >= ncol || ka >= HN) return;

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (long k0 = ka; k0 < kb; k0 += kTKT) {
        const int c = (int)(k0 / n);
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const float sc = scalein[r * kHcMax + c];
            const float* wp = w + k0 + k8;
            union { uint4 v; unsigned short s[8]; } in, out;
            in.v = *(const uint4*)(h + r * hrow + k0 + k8);
#pragma unroll
            for (int e = 0; e < 8; ++e) out.s[e] = f2bf(bf2f(in.s[e]) * sc * (wp[e] + wadd));
            *(uint4*)(sA + rr * kTAP + k8) = out.v;
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long col = c0 + tt; if (col >= ncol) col = ncol - 1;
            if (F8 && col < lr) {
                const unsigned char* row = (const unsigned char*)downv + col * dld;
                const long k = k0 + k8;
                *(uint4*)(sB + tt * kTAP + k8) =
                    e4m3x8_bf16(row + k, ((const float*)(row + HN))[k / r4d_hc8::kDownGroup]);
            } else {
                const unsigned short* src = (col < lr) ? (const unsigned short*)downv + col * HN
                                                       : injw + (col - lr) * HN;
                *(uint4*)(sB + tt * kTAP + k8) = *(const uint4*)(src + k0 + k8);
            }
        }
        __syncthreads();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    }

    const float inv_hc = 1.0f / (float)hc;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const long col = c0 + nt * 16 + nn;
            if (col >= ncol) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r >= rows) continue;
                const float v = acc[im][in][e];
                if (part)          part[((long)blockIdx.z * rows + r) * ncol + col] = v;
                else if (col < lr) dout[r * lr + col] = f2bf(siluf_(v * inv_hc));
                else               inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
        }
    }
}

/* The split-K reduce: ksplit partials summed in ASCENDING z -- a fixed function of the shape, so
 * the result is reproducible -- then the epilogue the tiled form would have applied, which is the
 * silu for a W_down column and the doubled sigmoid for an inject column. */
__global__ __launch_bounds__(kThreads)
void hc_down_reduce_kernel(const float* __restrict__ part, unsigned short* __restrict__ dout,
                           unsigned short* __restrict__ inj, long rows, long lr, long ncol,
                           long irow, int ksplit, float inv_hc) {
    const long i   = (long)blockIdx.x * kThreads + threadIdx.x;
    const long tot = rows * ncol;
    if (i >= tot) return;
    /* EIGHT LOADS BEFORE THE ADDS, which is the whole performance of this kernel and none of its
     * arithmetic: written as one `s +=` an iteration each load sits behind the previous add, and
     * a 23-way split is a 23-deep dependent chain of DRAM latency on a 41-workgroup grid. The
     * batching does not touch the ORDER of the adds, so the answer is the same bytes.
     * gemm_fp8a8's splitk_reduce_kernel says the same thing at more length. */
    float s = 0.0f;
    int   z = 0;
    for (; z + 8 <= ksplit; z += 8) {
        float v[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) v[t] = part[(long)(z + t) * tot + i];
#pragma unroll
        for (int t = 0; t < 8; ++t) s += v[t];
    }
    for (; z < ksplit; ++z) s += part[(long)z * tot + i];
    const long r = i / ncol, col = i - r * ncol;
    if (col < lr) dout[r * lr + col] = f2bf(siluf_(s * inv_hc));
    else          inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(s * inv_hc));
}

/* THE PREFILL TILE: its rows, the down product's columns a workgroup and the up product's
 * channels a workgroup. 112 columns is three tiles over 320 + 4, the fewest that hold a
 * 128-row A tile and the B tile in LDS twice a CU; measured against 64 and 128 at 129 to 4096
 * rows, it is the fastest at every one. */
constexpr int kWideRT = 128, kWideNT = 112, kWideCH = 16;

/* THE PREFILL TILE'S FORM, pinned for a test: 0 is the launcher's rule, 1 the decode tiles'
 * kernels at RT 128 (hc_down_tiled, then hc_up_tiled into `ubuf` and hc_mix), 2 the wide kernels
 * (which then refuse a shape they are not built for). */
static int g_hc_prefill_form = 0;

/* ==================================================== d = silu(W_down @ N / hc) at the PREFILL tile
 *
 * hc_down_tiled_kernel at RT 128 pays three things at a 2048-row chunk that its decode tiles do not
 * notice:
 *
 *   THE INJECT TILE. 320 + 4 columns in 64-column tiles is six tiles, and the sixth stages the
 *   whole A tile -- N rebuilt from h, the costliest part of a slice -- and runs every WMMA of the
 *   tile for four real columns.
 *   THE A TILE ONCE PER COLUMN TILE. Every column tile rebuilds the same N; the fewer the tiles,
 *   the fewer the rebuilds.
 *   NO PREFETCH. A slice's global loads are issued after the previous slice's WMMAs and waited on
 *   before the next, so a workgroup's loads and its matrix work never overlap.
 *
 * So the prefill tile has its own kernel: NT columns a workgroup, the n-tiles past the last real
 * column skipped, and the next slice's global loads issued into registers before this slice's
 * WMMAs. A thread stages one fixed 8-wide column of four A rows, so the norm weight is loaded once
 * a slice rather than once a row.
 *
 * THE ARITHMETIC IS hc_down_tiled_kernel's: the same N, the same decode of W_down, the same 16x16
 * tiles over the same K slices in the same order, and the launch keeps hc_down_kchunk's split --
 * so it is the same bytes, which r4d_selftest checks. hc_read at Qwen3.8-Flash-Next's shape, with
 * hc_up_mix_wide_kernel, against the decode tiles' kernels at RT 128: 767 us a call against 1134
 * at 2048 rows, 396 against 560 at 1024, 106 against 128 at 129. */
template <int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_down_wide_kernel(const unsigned short* __restrict__ h, const float* __restrict__ w,
                         const void* __restrict__ downv, long dld,
                         const float* __restrict__ scalein,
                         const unsigned short* __restrict__ injw,
                         unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
                         float* __restrict__ part,
                         long rows, long n, int hc, long lr, long ncol, long hrow, long irow,
                         long kc, float wadd) {
    constexpr int RT = kWideRT, NT16 = NT / 16;
    constexpr int kKV = kTKT / 8;                              /* 16-byte vectors a staged row */
    constexpr int kRS = kThreads / kKV;                        /* rows between a thread's A rows */
    constexpr int kAV = RT / kRS;
    constexpr int kBV = (NT * kKV + kThreads - 1) / kThreads;
    /* THE WAVES TILE THE BLOCK TWO WAYS, four over the rows and two over the columns, each holding
     * two m-tiles and up to four n-tiles. One m-tile a wave across all seven n-tiles read eight
     * fragments out of LDS for seven WMMA; two by four read six for eight, and the block's LDS
     * reads a k step fall from 64 fragments to 44. The column halves are four n-tiles and three,
     * and a wave's half is `wave / kDWM`, so waves w and w + 4 -- one of each -- share a SIMD. */
    constexpr int kDWM = 4, kDTM = RT / 16 / kDWM, kDTN = (NT16 + 1) / 2;
    static_assert(kDWM * 2 == kWaves && kDTM * 16 * kDWM == RT && RT % kRS == 0,
                  "the waves cover the tile, whole A rows a pass");
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long HN = (long)hc * n;
    const long r0 = (long)blockIdx.y * RT;
    const long c0 = (long)blockIdx.x * NT;
    const long ka = (long)blockIdx.z * kc;
    const long kb = (ka + kc < HN) ? ka + kc : HN;
    if (r0 >= rows || c0 >= ncol || ka >= HN) return;
    const int nlive = (ncol - c0 >= NT) ? NT16 : (int)((ncol - c0 + 15) / 16);

    const int k8a = (tid % kKV) * 8, ar = tid / kKV;
    long arow[kAV];
#pragma unroll
    for (int i = 0; i < kAV; ++i) {
        const long r = r0 + ar + i * kRS;
        arow[i] = r < rows ? r : rows - 1;
    }

    uint4 pa[kAV];
    float psc[kAV], pw[8];
    uint4 pb[kBV];
    float pbs[kBV];
    auto fetch = [&](long k0) {
        const int c = (int)(k0 / n);
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            pa[i]  = *(const uint4*)(h + arow[i] * hrow + k0 + k8a);
            psc[i] = scalein[arow[i] * kHcMax + c];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) pw[e] = w[k0 + k8a + e];
#pragma unroll
        for (int b = 0; b < kBV; ++b) {
            const int idx = tid + b * kThreads, tt = idx / kKV, k8 = (idx % kKV) * 8;
            if (idx >= NT * kKV || tt >= nlive * 16) continue;
            long col = c0 + tt; if (col >= ncol) col = ncol - 1;
            const long k = k0 + k8;
            if (F8 && col < lr) {
                const unsigned char* row = (const unsigned char*)downv + col * dld;
                const uint2 cc = *(const uint2*)(row + k);
                pb[b]  = make_uint4(cc.x, cc.y, 0u, 0u);
                pbs[b] = ((const float*)(row + HN))[k / r4d_hc8::kDownGroup];
            } else {
                const unsigned short* src = (col < lr) ? (const unsigned short*)downv + col * HN
                                                       : injw + (col - lr) * HN;
                pb[b] = *(const uint4*)(src + k);
            }
        }
    };
    auto stage = [&]() {
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            union { uint4 v; unsigned short s[8]; } in, out;
            in.v = pa[i];
#pragma unroll
            for (int e = 0; e < 8; ++e) out.s[e] = f2bf(bf2f(in.s[e]) * psc[i] * (pw[e] + wadd));
            *(uint4*)(sA + (ar + i * kRS) * kTAP + k8a) = out.v;
        }
#pragma unroll
        for (int b = 0; b < kBV; ++b) {
            const int idx = tid + b * kThreads, tt = idx / kKV, k8 = (idx % kKV) * 8;
            if (idx >= NT * kKV || tt >= nlive * 16) continue;
            long col = c0 + tt; if (col >= ncol) col = ncol - 1;
            uint4 v = pb[b];
            if (F8 && col < lr) {
                float f[8];
                e4m3x8(v.x, v.y, f);
                union { uint4 v; unsigned short s[8]; } o;
#pragma unroll
                for (int e = 0; e < 8; ++e) o.s[e] = f2bf(f[e] * pbs[b]);
                v = o.v;
            }
            *(uint4*)(sB + tt * kTAP + k8) = v;
        }
    };

    const int wm = wave % kDWM, nb = (wave / kDWM) * kDTN;     /* first m-tile / n-tile */
    v8f acc[kDTM][kDTN];
#pragma unroll
    for (int i = 0; i < kDTM; ++i)
#pragma unroll
        for (int j = 0; j < kDTN; ++j) acc[i][j] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    fetch(ka);
    for (long k0 = ka; k0 < kb; k0 += kTKT) {
        __syncthreads();
        stage();
        __syncthreads();
        if (k0 + kTKT < kb) fetch(k0 + kTKT);
        /* Each fragment is read once a k step and taken to every tile of the wave that uses it;
         * each accumulator still sees its k steps in ascending order. */
#pragma unroll
        for (int ks = 0; ks < kTKT / 16; ++ks) {
            v8bf af[kDTM];
#pragma unroll
            for (int i = 0; i < kDTM; ++i)
                af[i] = frag(sA, kTAP, (wm * kDTM + i) * 16, ks * 16, lane);
#pragma unroll
            for (int j = 0; j < kDTN; ++j) {
                if (nb + j >= nlive) continue;
                const v8bf bf = frag(sB, kTAP, (nb + j) * 16, ks * 16, lane);
#pragma unroll
                for (int i = 0; i < kDTM; ++i) acc[i][j] = mma(af[i], bf, acc[i][j]);
            }
        }
    }

    const float inv_hc = 1.0f / (float)hc;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int j = 0; j < kDTN; ++j) {
        if (nb + j >= nlive) continue;
        const long col = c0 + (nb + j) * 16 + nn;
        if (col >= ncol) continue;
#pragma unroll
        for (int i = 0; i < kDTM; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + (wm * kDTM + i) * 16 + e0 + e;
                if (r >= rows) continue;
                const float v = acc[i][j][e];
                if (part)          part[((long)blockIdx.z * rows + r) * ncol + col] = v;
                else if (col < lr) dout[r * lr + col] = f2bf(siluf_(v * inv_hc));
                else               inj[r * irow + (col - lr)] = f2bf(2.0f * sigmoidf_(v * inv_hc));
            }
    }
}

/* Eight elements of row j of W_up at k, as the bf16 a tiled kernel stages: the stored bf16 itself,
 * or eight E4M3 codes decoded against their group's scale. A group is a multiple of sixteen codes,
 * so eight from a multiple of eight never straddle two. */
template <bool F8>
__device__ __forceinline__ uint4 hc_up_stage(const void* __restrict__ upv, long uld, long j, long k,
                                            long lr) {
    if constexpr (F8) {
        const unsigned char* row = (const unsigned char*)upv + j * uld;
        return e4m3x8_bf16(row + k, ((const float*)(row + lr))[k / r4d_hc8::up_group(lr)]);
    } else {
        return *(const uint4*)((const unsigned short*)upv + j * lr + k);
    }
}

/* ==================================================== u = sigmoid(W_up @ d), tiled
 *
 * C[RT rows, NT channels] over K = lowrank. Same tiling; the A tile is `d` straight out of scratch
 * and the K loop is five passes rather than 160. No K-split: hn/NT is already 160 workgroups. */
template <int RT, int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_up_tiled_kernel(const void* __restrict__ upv, long uld,
                        const unsigned short* __restrict__ dbuf,
                        unsigned short* __restrict__ ubuf, long rows, long hn, long lr) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long r0 = (long)blockIdx.y * RT;
    const long j0 = (long)blockIdx.x * NT;
    if (r0 >= rows || j0 >= hn) return;

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = *(const uint4*)(dbuf + r * lr + k);
            *(uint4*)(sA + rr * kTAP + k8) = v;
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long j = j0 + tt; if (j >= hn) j = hn - 1;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = hc_up_stage<F8>(upv, uld, j, k, lr);
            *(uint4*)(sB + tt * kTAP + k8) = v;
        }
        __syncthreads();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    }

    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const long j = j0 + nt * 16 + nn;
            if (j >= hn) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r < rows) ubuf[r * hn + j] = f2bf(sigmoidf_(acc[im][in][e]));
            }
        }
    }
}


/* ==================================== hc_read, kernels B and B2 in ONE launch, tiled
 *
 * `x = mean_c( u[c] * N[c] )` with `u` NEVER MATERIALISED -- which is what this file's header says
 * kernel B is for, and what a tile over the FLATTENED channel index cannot give. A tile of 64
 * consecutive `j` is 64 channels of ONE sub-stream, so the four terms of a channel's mean live in
 * four different workgroups and `ubuf` has to carry them; a tile of (stream, channel) PAIRS puts
 * all hc of a channel in the same workgroup and the mean is local.
 *
 * WHAT IT COSTS AND WHAT IT BUYS. The B tile stops being 64 contiguous rows of W_up and becomes hc
 * groups of NT/hc -- same bytes, four runs of 10 KiB instead of one of 40 -- and the fp8 twin
 * cannot fall out of one workgroup, because 128 contiguous channels is one scale group and a
 * workgroup owns NT/hc = 16 of them. That is the arrival counter below: the last of the eight
 * workgroups covering a group quantises it, reading back the 128 bf16 it and its siblings just
 * wrote. The tail is 8 KiB on the same 20 workgroups `hc_mix` would use, NOT on one, which is why
 * it costs less than the launch it saves.
 *
 * BIT-IDENTICAL TO THE UNFUSED PAIR, and every rounding is placed to keep it that way: `u` is
 * narrowed to bf16 before the multiply because hc_mix reads it back out of a bf16 `ubuf`, `N` is
 * narrowed by `nbf` in the same place, both go through LDS as bf16 so the reducer sees exactly the
 * bytes the unfused form loads, and the sum over c runs ASCENDING. */
template <int RT, int NT, bool F8, bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_tiled_kernel(const void* __restrict__ upv, long uld,
                            const unsigned short* __restrict__ dbuf,
                            const unsigned short* __restrict__ h, const float* __restrict__ w,
                            const float* __restrict__ scalein,
                            unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                            float* __restrict__ qs, unsigned char* __restrict__ rq,
                            float* __restrict__ rqs, unsigned int* __restrict__ ctr,
                            long rows, long n, int hc, long lr, long hrow, long xrow,
                            long qrow, long srow, long rqrow, long rsrow, float wadd) {
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];
    __shared__ unsigned short s_u[RT * NT];
    __shared__ unsigned short s_N[RT * NT];
    __shared__ long           s_j[NT];

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const int  ipb = NT / hc;                       /* channels this workgroup owns */
    const long r0  = (long)blockIdx.y * RT;
    const long i0  = (long)blockIdx.x * ipb;
    if (r0 >= rows) return;

    /* The column map, once: tile column t is sub-stream t % hc of channel i0 + t / hc. */
    if (tid < NT) s_j[tid] = (long)(tid % hc) * n + i0 + tid / hc;
    __syncthreads();

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
        for (int idx = tid; idx < RT * (kTKT / 8); idx += kThreads) {
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long r = r0 + rr; if (r >= rows) r = rows - 1;
            const long ka = k0 + k8;
            uint4 va = make_uint4(0u, 0u, 0u, 0u);
            if (ka + 8 <= lr) va = *(const uint4*)(dbuf + r * lr + ka);
            *(uint4*)(sA + rr * kTAP + k8) = va;
        }
        for (int idx = tid; idx < NT * (kTKT / 8); idx += kThreads) {
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            const long k = k0 + k8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k + 8 <= lr) v = hc_up_stage<F8>(upv, uld, s_j[tt], k, lr);
            *(uint4*)(sB + tt * kTAP + k8) = v;
        }
        __syncthreads();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sA, kTAP, mt, sB, kTAP, nt, lane, acc[im][in]);
            }
        }
    }
    /* N, IN A COALESCED PASS OF ITS OWN. The (stream, channel) tile is what makes the mean local
     * and it is also what would wreck this read: in the wmma epilogue consecutive lanes hold
     * consecutive `t`, which is consecutive SUB-STREAMS, so `h[r * hrow + j]` walks n elements a
     * lane and every load is its own transaction. Here consecutive threads take consecutive
     * channels of one sub-stream, exactly as `hc_mix` did. */
    for (int idx = tid; idx < RT * NT; idx += kThreads) {
        const int  rl = idx / NT, rem = idx - rl * NT;
        const int  c  = rem / ipb, ii = rem - c * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const long j = (long)c * n + i0 + ii;
        s_N[rl * NT + ii * hc + c] =
            f2bf(bf2f(h[r * hrow + j]) * scalein[r * kHcMax + c] * (w[j] + wadd));
    }

    /* u out of the accumulators, where it is. */
    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const int  t = nt * 16 + nn;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const long r = r0 + mt * 16 + e0 + e;
                if (r >= rows) continue;
                s_u[(r - r0) * NT + t] = f2bf(sigmoidf_(acc[im][in][e]));
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    for (int idx = tid; idx < RT * ipb; idx += kThreads) {
        const int  rl = idx / ipb, ii = idx - rl * ipb;
        const long r  = r0 + rl;
        if (r >= rows) continue;
        const unsigned short* const up_ = s_u + rl * NT + ii * hc;
        const unsigned short* const np_ = s_N + rl * NT + ii * hc;
        float a = 0.0f;
        for (int c = 0; c < hc; ++c) a += bf2f(up_[c]) * bf2f(np_[c]);
        x[r * xrow + i0 + ii] = f2bf(a * inv_hc);
    }
    if (!q || !ctr) return;

    /* The group's last workgroup quantises it. `nb` is how many cover one 128-channel group. */
    const int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    const unsigned slot = blockIdx.y * (gridDim.x / nb) + (unsigned)(i0 / kUpTB);
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != (unsigned)nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* The rotated twin wants a whole wave a row, where the codes below take kThreads / RT threads
     * a row -- so it walks the tile's rows a wave at a time, before those threads split off. */
    if (rq) {
        const long gb = (i0 / kUpTB) * kUpTB;
        for (int rl = wave; rl < RT; rl += kWaves) {
            const long r = r0 + rl;
            if (r >= rows) break;
            const unsigned short* const xr = x + r * xrow + gb;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[lane * 4 + j]);
            hc_rot_group(v, lane, rq + r * rqrow + gb, rqs + r * rsrow + gb / kUpTB);
        }
    }

    constexpr int kTPR = kThreads / RT;             /* threads a row; RT divides kThreads */
    const int  rl = tid / kTPR, lr_ = tid % kTPR;
    const long r  = r0 + rl;
    if (r >= rows) return;
    const long base = (i0 / kUpTB) * kUpTB;
    const unsigned short* const xr = x + r * xrow + base;
    float m = 0.0f;
    for (int t = lr_; t < kUpTB; t += kTPR) m = fmaxf(m, fabsf(bf2f(xr[t])));
    m = r4d_xor_max<kTPR>(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lr_ == 0) qs[r * srow + base / kUpTB] = sc;
    for (int d = lr_; d < kUpTB / 4; d += kTPR) {
        const int c4 = d * 4;
        const uint32_t pk = q8_pack4<I8>(bf2f(xr[c4 + 0]), bf2f(xr[c4 + 1]), bf2f(xr[c4 + 2]),
                                         bf2f(xr[c4 + 3]), inv);
        *(unsigned*)(q + r * qrow + base + c4) = (unsigned)pk;
    }
}

/* ==================================== hc_read, kernels B and B2 in ONE launch at the PREFILL tile
 *
 * AT THE 128-ROW TILE THE PAIRED FORM DOES NOT FIT. hc_up_mix_tiled_kernel keeps `u` out of memory
 * at the decode tiles, but at RT 128 its two staging planes on top of the A tile are 58 KiB of LDS
 * and one workgroup a CU, and the unfused pair -- hc_up_tiled into `ubuf`, hc_mix reading it back
 * -- costs 42 MB written and 42 MB read again a call at a 2048-row chunk, the largest stream the op
 * moves, and a third launch.
 *
 * NEITHER PLANE IS NEEDED when a LANE holds every sub-stream of a channel. A WMMA accumulator gives
 * lane `l` column `l & 15` of each 16-column n-tile, so with n-tile `c` holding sub-stream c of the
 * workgroup's sixteen channels, the hc terms of a channel's mean are the same lane's `acc[c]` and
 * the mean is register-local. N is staged once, coalesced, into the A tile's LDS after the K loop is
 * done with it. The workgroup's LDS is the unfused up product's.
 *
 * BIT-IDENTICAL TO THE UNFUSED PAIR: `u` is narrowed to bf16 before the multiply as hc_mix reads it
 * out of `ubuf`, N the same way, and the sum over c runs ascending in the same `a += u * N` form. A
 * 16x16 output tile is computed the same way whichever workgroup computes it.
 *
 * THE FP8 TWIN is the arrival counter's, as in hc_up_mix_tiled_kernel: a workgroup owns 16 of a
 * scale group's 128 channels, and the last of the eight covering a group quantises it for the
 * tile's rows. */
template <int HC, bool F8>
__global__ __launch_bounds__(kThreads)
void hc_up_mix_wide_kernel(const void* __restrict__ upv, long uld,
                           const unsigned short* __restrict__ dbuf,
                           const unsigned short* __restrict__ h, const float* __restrict__ w,
                           const float* __restrict__ scalein,
                           unsigned short* __restrict__ x,
                           long rows, long n, long lr, long hrow, long xrow, float wadd) {
    constexpr int RT = kWideRT, CH = kWideCH, NT = HC * CH;
    static_assert(RT / 16 == kWaves, "one m-tile a wave");
    static_assert(RT * NT <= RT * kTAP, "N must fit in the A tile's LDS");
    __shared__ unsigned short sA[RT * kTAP];
    __shared__ unsigned short sB[NT * kTAP];

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long r0 = (long)blockIdx.y * RT;
    const long i0 = (long)blockIdx.x * CH;
    if (r0 >= rows) return;

    /* EVERY GLOBAL LOAD IS ISSUED A STEP BEFORE IT IS NEEDED. The mix's `h` -- 42 MB a call at a
     * 2048-row chunk, the op's largest read, and needed only after the product -- is loaded into
     * registers before the K loop, so its latency runs under the matrix work instead of after it;
     * and the loop holds the next slice's A rows and weight codes in registers while this slice's
     * WMMA runs, as hc_down_wide_kernel does. Only WHEN a load is issued moves: the same bytes reach
     * the same LDS and the same arithmetic, so the output is the same bytes.
     *
     * A thread's slots are fixed: A and B rows `tid / 8 + 32 i` at column `(tid % 8) * 8` of a
     * slice, and for N the (row, sub-stream, eight channels) the idx `tid + 256 i` names. */
    constexpr int kKV = kTKT / 8;                               /* 16-byte vectors a staged row */
    constexpr int kRS = kThreads / kKV;                         /* rows between a thread's rows */
    constexpr int kAV = RT / kRS, kBV = NT / kRS;
    constexpr int kNV = RT * HC * (CH / 8) / kThreads;
    static_assert(RT % kRS == 0 && NT % kRS == 0 && (RT * HC * (CH / 8)) % kThreads == 0,
                  "every thread stages whole vectors");
    const int k8a = (tid % kKV) * 8, tr = tid / kKV;

    uint4 pn[kNV];
    float pns[kNV];
#pragma unroll
    for (int i = 0; i < kNV; ++i) {
        const int  idx = tid + i * kThreads;
        const int  rl = idx / (HC * CH / 8), rem = idx % (HC * CH / 8);
        const int  c  = rem / (CH / 8), i8 = (rem % (CH / 8)) * 8;
        const long r  = r0 + rl;
        pn[i] = make_uint4(0u, 0u, 0u, 0u);
        pns[i] = 0.0f;
        if (r < rows) {
            pn[i]  = *(const uint4*)(h + r * hrow + (long)c * n + i0 + i8);
            pns[i] = scalein[r * kHcMax + c];
        }
    }

    uint4 pa[kAV], pb[kBV];
    float pbs[kBV];
    auto fetch = [&](long k0) {
        const long k = k0 + k8a;
        const bool in = k + 8 <= lr;
#pragma unroll
        for (int i = 0; i < kAV; ++i) {
            long r = r0 + tr + i * kRS; if (r >= rows) r = rows - 1;
            pa[i] = in ? *(const uint4*)(dbuf + r * lr + k) : make_uint4(0u, 0u, 0u, 0u);
        }
        /* Tile row tt is sub-stream tt / CH of channel i0 + tt % CH. */
#pragma unroll
        for (int i = 0; i < kBV; ++i) {
            const int  tt = tr + i * kRS;
            const long j  = (long)(tt / CH) * n + i0 + tt % CH;
            pb[i] = make_uint4(0u, 0u, 0u, 0u);
            pbs[i] = 0.0f;
            if (!in) continue;
            if constexpr (F8) {
                const unsigned char* row = (const unsigned char*)upv + j * uld;
                const uint2 cc = *(const uint2*)(row + k);
                pb[i]  = make_uint4(cc.x, cc.y, 0u, 0u);
                pbs[i] = ((const float*)(row + lr))[k / r4d_hc8::up_group(lr)];
            } else {
                pb[i] = *(const uint4*)((const unsigned short*)upv + j * lr + k);
            }
        }
    };

    v8f acc[HC];
#pragma unroll
    for (int c = 0; c < HC; ++c) acc[c] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    fetch(0);
    for (long k0 = 0; k0 < lr; k0 += kTKT) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kAV; ++i) *(uint4*)(sA + (tr + i * kRS) * kTAP + k8a) = pa[i];
#pragma unroll
        for (int i = 0; i < kBV; ++i) {
            uint4 v = pb[i];
            if constexpr (F8) {
                float f[8];
                e4m3x8(v.x, v.y, f);
                union { uint4 v; unsigned short s[8]; } o;
#pragma unroll
                for (int e = 0; e < 8; ++e) o.s[e] = f2bf(f[e] * pbs[i]);
                v = k0 + k8a + 8 <= lr ? o.v : make_uint4(0u, 0u, 0u, 0u);
            }
            *(uint4*)(sB + (tr + i * kRS) * kTAP + k8a) = v;
        }
        __syncthreads();
        if (k0 + kTKT < lr) fetch(k0 + kTKT);
        /* A's fragment is read once a k step and taken to every n-tile. Each accumulator still
         * sees its k steps in ascending order, which is all its bytes depend on. */
#pragma unroll
        for (int ks = 0; ks < kTKT / 16; ++ks) {
            const v8bf af = frag(sA, kTAP, wave * 16, ks * 16, lane);
#pragma unroll
            for (int c = 0; c < HC; ++c)
                acc[c] = mma(af, frag(sB, kTAP, c * 16, ks * 16, lane), acc[c]);
        }
    }
    __syncthreads();

    /* N for the tile, [RT][HC][CH] bf16 in the A tile's LDS, from the `h` loaded before the loop. */
    unsigned short* const sN = sA;
#pragma unroll
    for (int i = 0; i < kNV; ++i) {
        const int  idx = tid + i * kThreads;
        const int  rl = idx / (HC * CH / 8), rem = idx % (HC * CH / 8);
        const int  c  = rem / (CH / 8), i8 = (rem % (CH / 8)) * 8;
        if (r0 + rl >= rows) continue;
        const long j  = (long)c * n + i0 + i8;
        union { uint4 v; unsigned short s[8]; } in, out;
        in.v = pn[i];
#pragma unroll
        for (int e = 0; e < 8; ++e) out.s[e] = f2bf(bf2f(in.s[e]) * pns[i] * (w[j + e] + wadd));
        *(uint4*)(sN + (rl * HC + c) * CH + i8) = out.v;
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)HC;
    const int   e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int  rl = wave * 16 + e0 + e;
        const long r  = r0 + rl;
        float a = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c)
            a += bf2f(f2bf(sigmoidf_(acc[c][e]))) * bf2f(sN[(rl * HC + c) * CH + nn]);
        if (r < rows) x[r * xrow + i0 + nn] = f2bf(a * inv_hc);
    }
}

/* THE FP8 CODES OF `x` AT THE PREFILL TILE, and the rotated twin: one wave a (row, 128-channel
 * scale group), after hc_up_mix_wide_kernel. That kernel's workgroup owns sixteen of a group's
 * channels, so quantising inside it means whichever of its eight workgroups arrives last, behind
 * an arrival counter and a device-scope fence in every one of them, walking the group's 128 rows
 * by itself, a row at a time -- 82 of that kernel's 356 us at a 2048-row chunk. This launch is a
 * stream over `x` at memory speed.
 *
 * The same bytes: the amax is an fmaxf reduction, exact in any order, and every code is the same
 * conversion of the same bf16 value by the same scale; the rotated twin is hc_rot_group itself. */
template <bool I8 = false>
__global__ __launch_bounds__(kThreads)
void hc_quant_rows_kernel(const unsigned short* __restrict__ x, unsigned char* __restrict__ q,
                          float* __restrict__ qs, unsigned char* __restrict__ rq,
                          float* __restrict__ rqs, long rows, long ngrp, long xrow, long qrow,
                          long srow, long rqrow, long rsrow) {
    const int  lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const long item = (long)blockIdx.x * kWaves + wave;
    if (item >= rows * ngrp) return;
    const long r = item / ngrp, g = item % ngrp, base = g * kUpTB;
    const uint2 raw = *(const uint2*)(x + r * xrow + base + lane * 4);
    const unsigned short* xs = reinterpret_cast<const unsigned short*>(&raw);
    float v[4], t[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = bf2f(xs[j]); t[j] = v[j]; }

    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
    m = r4d_wave_max(m);
    const float sc  = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * srow + g] = sc;
    const uint32_t pk = q8_pack4<I8>(v[0], v[1], v[2], v[3], inv);
    *(unsigned*)(q + r * qrow + base + lane * 4) = (unsigned)pk;

    if (rq) hc_rot_group(t, lane, rq + r * rqrow + base, rqs + r * rsrow + g);
}

/* The mix finisher's arrival counters, one block a STREAM and not a device -- the same rule
 * gemm_fp8a8's split_ctr states, for the same reason: a self-resetting counter is sound only
 * while two launches of its kernel cannot overlap, and one stream is what guarantees that.
 * Failure to allocate is not an error; the caller runs the unfused pair instead. */
constexpr unsigned kMaxMixCtr = 2048;

unsigned int* hc_mix_ctr(RadStream s) {
    static std::mutex m;
    struct Slot { std::atomic<RadStream> key{nullptr}; std::atomic<unsigned int*> ctr{nullptr}; };
    static Slot slot[16][4];
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return nullptr;
    for (int i = 0; i < 4; ++i)
        if (slot[dev][i].key.load(std::memory_order_acquire) == s)
            return slot[dev][i].ctr.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lk(m);
    for (int i = 0; i < 4; ++i)
        if (slot[dev][i].key.load(std::memory_order_relaxed) == s)
            return slot[dev][i].ctr.load(std::memory_order_relaxed);
    int fi = -1;
    for (int i = 0; i < 4; ++i)
        if (slot[dev][i].ctr.load(std::memory_order_relaxed) == nullptr) { fi = i; break; }
    if (fi < 0) return nullptr;
    void* p = nullptr;
    if (hipMalloc(&p, (size_t)kMaxMixCtr * sizeof(unsigned int)) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, (size_t)kMaxMixCtr * sizeof(unsigned int)) != hipSuccess) {
        (void)hipFree(p);
        return nullptr;
    }
    slot[dev][fi].ctr.store((unsigned int*)p, std::memory_order_relaxed);
    slot[dev][fi].key.store(s, std::memory_order_release);
    return (unsigned int*)p;
}

static inline void hc_up_mix_tiled_launch(int rt, bool f8, bool i8, const void* up, long uld,
                                          const unsigned short* dbuf, const unsigned short* h,
                                          const float* w, const float* scale, unsigned short* x,
                                          unsigned char* q, float* qs, unsigned char* rq,
                                          float* rqs, unsigned int* ctr,
                                          long rows, long n, int hc, long lr, long hrow,
                                          long xrow, long qrow, long srow, long rqrow, long rsrow,
                                          float wadd, RadStream s) {
    const dim3 gm((unsigned)(n / (kTNT / hc)), (unsigned)((rows + rt - 1) / rt));
#define HC_UM_CASE3(RT_, F8_, I8_)                                                               \
    hipLaunchKernelGGL((hc_up_mix_tiled_kernel<RT_, kTNT, F8_, I8_>), gm, dim3(kThreads), 0,      \
                       (hipStream_t)s, up, uld, dbuf, h, w, scale, x, q, qs, rq, rqs, ctr, rows, \
                       n, hc, lr, hrow, xrow, qrow, srow, rqrow, rsrow, wadd)
#define HC_UM_CASE(RT_, F8_)                                                                     \
    do { if (i8) HC_UM_CASE3(RT_, F8_, true); else HC_UM_CASE3(RT_, F8_, false); } while (0)
    if (f8) { if (rt == 32) HC_UM_CASE(32, true);  else HC_UM_CASE(64, true); }
    else    { if (rt == 32) HC_UM_CASE(32, false); else HC_UM_CASE(64, false); }
#undef HC_UM_CASE
#undef HC_UM_CASE3
}

static inline void hc_up_mix_wide_launch(bool f8, bool i8, const void* up, long uld,
                                         const unsigned short* dbuf, const unsigned short* h,
                                         const float* w, const float* scale, unsigned short* x,
                                         unsigned char* q, float* qs, unsigned char* rq,
                                         float* rqs,
                                         long rows, long n, long lr, long hrow, long xrow,
                                         long qrow, long srow, long rqrow, long rsrow, float wadd,
                                         RadStream s) {
    const dim3 gm((unsigned)(n / kWideCH), (unsigned)((rows + kWideRT - 1) / kWideRT));
#define HC_UMW_CASE(F8_)                                                                         \
    hipLaunchKernelGGL((hc_up_mix_wide_kernel<4, F8_>), gm, dim3(kThreads), 0, (hipStream_t)s,   \
                       up, uld, dbuf, h, w, scale, x, rows, n, lr, hrow, xrow, wadd)
    if (f8) HC_UMW_CASE(true); else HC_UMW_CASE(false);
#undef HC_UMW_CASE
    if (!q) return;
    const long items = rows * (n / kUpTB);
    if (i8)
        hipLaunchKernelGGL(hc_quant_rows_kernel<true>, dim3((unsigned)((items + kWaves - 1) / kWaves)),
                           dim3(kThreads), 0, (hipStream_t)s, x, q, qs, rq, rqs, rows, n / kUpTB,
                           xrow, qrow, srow, rqrow, rsrow);
    else
        hipLaunchKernelGGL(hc_quant_rows_kernel<false>, dim3((unsigned)((items + kWaves - 1) / kWaves)),
                           dim3(kThreads), 0, (hipStream_t)s, x, q, qs, rq, rqs, rows, n / kUpTB,
                           xrow, qrow, srow, rqrow, rsrow);
}


/* K per down-product workgroup: enough z-blocks that lr/kTNT * z covers the part, rounded to a
 * whole number of kTKT slices so a slice never straddles a sub-stream. Returns HN when one block
 * already covers K, which is the prefill form and takes the un-split epilogue. */

/* ==================================================== mtp_enter, tiled
 *
 * THE PER-ROW FORM READS BOTH SQUARE MATRICES ONCE A ROW, and a draft round at eight concurrent
 * sequences has eight. Its grid is (n/kMtpOutTB, rows) and a workgroup owns 32 output channels of
 * ONE row, so the 26.2 MB of fc_h and fc_e is read `rows` times -- 210 MB a call at eight rows,
 * four times the cost of one. Same shape of problem as the gated residual's, and the same fix.
 *
 * AS A GEMM IT IS TWO PRODUCTS SHARING ONE ACCUMULATOR. Number the output rows (r, k) -- one per
 * sub-stream of each token -- and
 *
 *     x[(r,k)][i] = nh[(r,k)] . fc_h[i]  +  ne[r] . fc_e[i]
 *
 * is C[R, n] over K = n with R = rows*hc, where the second term's A operand is the same `ne[r]`
 * for all hc of a token's rows. So one tile, two A tiles, two B tiles, one accumulator: the fc_e
 * product is computed hc times instead of once, which costs WMMA slots that are idle anyway and
 * buys the weight being read ONCE.
 *
 * THE NORM SCALES ARE RECOMPUTED PER BLOCK rather than published, because this op has no scratch
 * hook and adding one to the ABI for five floats a row is the wrong trade: `h` and `e` are 25 KiB
 * a row and stay in cache across the 40 blocks, against the 655 KiB of weight each block reads.
 * The reduction order is per output row here and per thread-stride there, so the scales differ in
 * their last bit from the per-row kernel's -- the same class of reassociation the tiled gated
 * residual already carries, on a draft head whose output is a proposal. */
/* THE COLUMN TILE FOLLOWS THE ROW TILE. The product has no split over K -- its two products share
 * an accumulator, and splitting them would need a reduce launch and a scratch the op does not have
 * -- so the grid is n / NT workgroups a row tile, and each of them sums the squares of every row
 * of its tile before its first weight byte. A narrow tile buys workgroups for the weight stream
 * and pays for them in that prologue, once each; the prologue grows with the row tile. Measured
 * at this model's n = 2560, kbench, us a call:
 *
 *     columns   M=2   M=4   M=8   M=16   M=32
 *       16       68    70    72    124    280
 *       32       48    50    55     92    303
 *       64       60    63    69    103    189
 *
 * so 32 up to the 64-row tile and 64 at the 128-row one. Which workgroup computes a 16x16 output
 * tile changes nothing about how it is computed. */
constexpr int kMtpTNTNarrow = 32, kMtpTNTWide = 64;

template <int RT, int NT, bool F8>
__global__ __launch_bounds__(kThreads)
void mtp_enter_tiled_kernel(const unsigned short* __restrict__ h, const unsigned short* __restrict__ e,
                            const float* __restrict__ wh, const float* __restrict__ we,
                            const void* __restrict__ fchv, const void* __restrict__ fcev, long fld,
                            unsigned short* __restrict__ x,
                            long rows, long n, int hc, int flat, float eps, float wadd,
                            long hrow, long erow, long xrow) {
    __shared__ unsigned short sAh[RT * kTAP];
    __shared__ unsigned short sAe[RT * kTAP];
    __shared__ unsigned short sBh[NT * kTAP];
    __shared__ unsigned short sBe[NT * kTAP];
    __shared__ float s_rs[RT];                 /* one per OUTPUT row (r,k) */
    __shared__ float s_re[RT];                 /* ne's scale, replicated per output row */
    __shared__ float s_et[RT + 1];             /* ne's scale, one per token of the tile */

    constexpr int MT = RT / 16, NT16 = NT / 16;
    constexpr int WM = (MT < kWaves) ? MT : kWaves, WN = kWaves / WM;
    constexpr int AM = (MT + WM - 1) / WM, AN = (NT16 + WN - 1) / WN;

    const int  tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const long R    = rows * hc;
    const long r0   = (long)blockIdx.y * RT;
    const long j0   = (long)blockIdx.x * NT;
    if (r0 >= R || j0 >= n) return;

    /* ---- the scales, one output row a wave. `flat` sums the whole wide row instead of one
     * sub-stream, which is the same partials added together.
     *
     * THE SUMS OF SQUARES ARE LATENCY CHAINS, AND UNHIDDEN THEY ARE THE KERNEL'S TIME. A lane's
     * share of a row is 80 loads at this shape, each feeding an add; written as one load an
     * iteration, every link waits out a full load latency, for every row of the tile, in every
     * workgroup, before the first weight byte is used -- ~170 us a call at two, four and sixteen
     * rows alike, where the weight stream is ~50. So a wave keeps its loads in flight: kSsRows rows
     * at a time, and kSsPre of each row's loads issued before the first is summed.
     *
     * AND THE EMBEDDING'S SCALE IS A TOKEN'S, NOT A ROW'S: the hc output rows of one token share
     * `e`, so its sum is taken once a token of the tile rather than hc times.
     *
     * Bit-identical: each row is still summed by one wave, a lane still takes j = lane, lane + 32,
     * ... ascending into one accumulator, and the wave reduction is the same. */
    constexpr int kSsPre = 8, kSsRows = 4;
    auto sumsq = [&](const unsigned short* const* p, int nr, long len, float* out) {
        float acc[kSsRows];
#pragma unroll
        for (int k = 0; k < kSsRows; ++k) acc[k] = 0.0f;
        long j = lane;
        for (; j + (long)(kSsPre - 1) * kWave < len; j += (long)kSsPre * kWave) {
            unsigned short v[kSsRows][kSsPre];
#pragma unroll
            for (int k = 0; k < kSsRows; ++k)
                if (k < nr) {
#pragma unroll
                    for (int q = 0; q < kSsPre; ++q) v[k][q] = p[k][j + (long)q * kWave];
                }
#pragma unroll
            for (int k = 0; k < kSsRows; ++k)
                if (k < nr) {
#pragma unroll
                    for (int q = 0; q < kSsPre; ++q) { const float f = bf2f(v[k][q]); acc[k] += f * f; }
                }
        }
        for (; j < len; j += kWave)
#pragma unroll
            for (int k = 0; k < kSsRows; ++k)
                if (k < nr) { const float f = bf2f(p[k][j]); acc[k] += f * f; }
#pragma unroll
        for (int k = 0; k < kSsRows; ++k)
            if (k < nr) out[k] = wave_sum(acc[k]);
    };
    const long hlen = flat ? (long)hc * n : n;
    for (int rb = wave; rb < RT; rb += kWaves * kSsRows) {
        const unsigned short* p[kSsRows];
        int nr = 0;
        for (int rr = rb; rr < RT && nr < kSsRows; rr += kWaves) {
            const long Ri = r0 + rr;
            if (Ri >= R) break;
            const long rt = Ri / hc, kt = Ri % hc;
            p[nr] = h + rt * hrow + (flat ? 0 : kt * n);
            ++nr;
        }
        if (nr > 0) {
            float ss[kSsRows];
            sumsq(p, nr, hlen, ss);
            if (lane == 0)
                for (int k = 0; k < nr; ++k)
                    s_rs[rb + k * kWaves] = __frsqrt_rn(ss[k] / (float)hlen + eps);
        }
    }
    /* The tile's tokens: those of its real rows, and `rows - 1` for the padding rows past R,
     * which is the last real row's token -- so the tile's tokens are [t0, t1]. */
    const long t0 = r0 / hc;
    const long t1 = ((r0 + RT < R) ? (r0 + RT - 1) : (R - 1)) / hc;
    for (long tb = t0 + wave; tb <= t1; tb += kWaves * kSsRows) {
        const unsigned short* p[kSsRows];
        int nr = 0;
        for (long tt = tb; tt <= t1 && nr < kSsRows; tt += kWaves, ++nr) p[nr] = e + tt * erow;
        float ss[kSsRows];
        sumsq(p, nr, n, ss);
        if (lane == 0)
            for (int k = 0; k < nr; ++k)
                s_et[tb - t0 + k * kWaves] = __frsqrt_rn(ss[k] / (float)n + eps);
    }
    __syncthreads();
    for (int rr = tid; rr < RT; rr += kThreads) {
        const long Ri = r0 + rr;
        s_re[rr] = (Ri < R) ? s_et[Ri / hc - t0] : 0.0f;
    }
    __syncthreads();

    v8f acc[AM][AN];
#pragma unroll
    for (int im = 0; im < AM; ++im)
#pragma unroll
        for (int in = 0; in < AN; ++in) acc[im][in] = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    /* THE NEXT K SLICE IS LOADED INTO REGISTERS WHILE THIS ONE COMPUTES. Staged straight through,
     * every one of the n / kTKT slices exposes a global latency in full -- nothing else in the
     * block is runnable while its loads are out, and at decode the grid is a few workgroups a CU.
     * So a thread's share of slice k + 1 -- the raw `h`, `e` and norm-weight values of its A items
     * and the weight rows of its B items -- is issued right after slice k is written to LDS, and
     * waited for at the next write. Only WHEN a byte is loaded moves: the values written to LDS
     * and every product are the ones the straight form computes. */
    constexpr int kAI = (RT * (kTKT / 8) + kThreads - 1) / kThreads;
    constexpr int kBI = (NT * (kTKT / 8) + kThreads - 1) / kThreads;
    struct APre { uint4 h, e; float4 wh0, wh1, we0, we1; };
    APre ap[kAI];
    /* A weight item is eight bf16 of a row, or at E4M3 its eight codes and their group's scale. */
    struct BPre16 { uint4 v; };
    struct BPre8  { uint2 c; float s; };
    using BPre = typename std::conditional<F8, BPre8, BPre16>::type;
    BPre bph[kBI], bpe[kBI];
    auto bload = [&](const void* base, long j, long k, BPre& o) {
        if constexpr (F8) {
            const unsigned char* row = (const unsigned char*)base + j * fld;
            o.c = *(const uint2*)(row + k);
            o.s = ((const float*)(row + n))[k / r4d_hc8::kDownGroup];
        } else {
            o.v = *(const uint4*)((const unsigned short*)base + j * n + k);
        }
    };
    auto bval = [&](const BPre& o) -> uint4 {
        if constexpr (F8) {
            float f[8];
            e4m3x8(o.c.x, o.c.y, f);
            union { uint4 v; unsigned short s[8]; } u;
#pragma unroll
            for (int q = 0; q < 8; ++q) u.s[q] = f2bf(f[q] * o.s);
            return u.v;
        } else {
            return o.v;
        }
    };
    auto issue = [&](long k0) {
#pragma unroll
        for (int a = 0; a < kAI; ++a) {
            const int idx = tid + a * kThreads;
            if (idx >= RT * (kTKT / 8)) break;
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            const long Ri = r0 + rr;
            if (Ri < R) {
                const long rt = Ri / hc, kt = Ri % hc;
                ap[a].h   = *(const uint4*)(h + rt * hrow + kt * n + k0 + k8);
                ap[a].e   = *(const uint4*)(e + rt * erow + k0 + k8);
                ap[a].wh0 = *(const float4*)(wh + kt * n + k0 + k8);
                ap[a].wh1 = *(const float4*)(wh + kt * n + k0 + k8 + 4);
                ap[a].we0 = *(const float4*)(we + k0 + k8);
                ap[a].we1 = *(const float4*)(we + k0 + k8 + 4);
            }
        }
#pragma unroll
        for (int b = 0; b < kBI; ++b) {
            const int idx = tid + b * kThreads;
            if (idx >= NT * (kTKT / 8)) break;
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            long j = j0 + tt; if (j >= n) j = n - 1;
            bload(fchv, j, k0 + k8, bph[b]);
            bload(fcev, j, k0 + k8, bpe[b]);
        }
    };
    issue(0);

    for (long k0 = 0; k0 < n; k0 += kTKT) {
        __syncthreads();
#pragma unroll
        for (int a = 0; a < kAI; ++a) {
            const int idx = tid + a * kThreads;
            if (idx >= RT * (kTKT / 8)) break;
            const int rr = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            const long Ri = r0 + rr;
            if (Ri < R) {
                const float sc = s_rs[rr], se = s_re[rr];
                union { uint4 v; unsigned short s[8]; } in, out;
                const float wp[8] = { ap[a].wh0.x, ap[a].wh0.y, ap[a].wh0.z, ap[a].wh0.w,
                                      ap[a].wh1.x, ap[a].wh1.y, ap[a].wh1.z, ap[a].wh1.w };
                const float wq[8] = { ap[a].we0.x, ap[a].we0.y, ap[a].we0.z, ap[a].we0.w,
                                      ap[a].we1.x, ap[a].we1.y, ap[a].we1.z, ap[a].we1.w };
                in.v = ap[a].h;
#pragma unroll
                for (int q = 0; q < 8; ++q) out.s[q] = f2bf(bf2f(in.s[q]) * sc * (wp[q] + wadd));
                *(uint4*)(sAh + rr * kTAP + k8) = out.v;
                in.v = ap[a].e;
#pragma unroll
                for (int q = 0; q < 8; ++q) out.s[q] = f2bf(bf2f(in.s[q]) * se * (wq[q] + wadd));
                *(uint4*)(sAe + rr * kTAP + k8) = out.v;
            } else {
                *(uint4*)(sAh + rr * kTAP + k8) = make_uint4(0, 0, 0, 0);
                *(uint4*)(sAe + rr * kTAP + k8) = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int b = 0; b < kBI; ++b) {
            const int idx = tid + b * kThreads;
            if (idx >= NT * (kTKT / 8)) break;
            const int tt = idx / (kTKT / 8), k8 = (idx % (kTKT / 8)) * 8;
            *(uint4*)(sBh + tt * kTAP + k8) = bval(bph[b]);
            *(uint4*)(sBe + tt * kTAP + k8) = bval(bpe[b]);
        }
        if (k0 + kTKT < n) issue(k0 + kTKT);
        __syncthreads();
#pragma unroll
        for (int im = 0; im < AM; ++im) {
            const int mt = wave / WN + im * WM;
            if (mt >= MT) continue;
#pragma unroll
            for (int in = 0; in < AN; ++in) {
                const int nt = wave % WN + in * WN;
                if (nt >= NT16) continue;
                acc[im][in] = gemm<kTKT / 16>(sAh, kTAP, mt, sBh, kTAP, nt, lane, acc[im][in]);
                acc[im][in] = gemm<kTKT / 16>(sAe, kTAP, mt, sBe, kTAP, nt, lane, acc[im][in]);
            }
        }
    }

    const int e0 = 8 * (lane >> 4), nn = lane & 15;
#pragma unroll
    for (int im = 0; im < AM; ++im) {
        const int mt = wave / WN + im * WM;
        if (mt >= MT) continue;
#pragma unroll
        for (int in = 0; in < AN; ++in) {
            const int nt = wave % WN + in * WN;
            if (nt >= NT16) continue;
            const long j = j0 + nt * 16 + nn;
            if (j >= n) continue;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const long Ri = r0 + mt * 16 + e0 + q;
                if (Ri >= R) continue;
                x[(Ri / hc) * xrow + (Ri % hc) * n + j] = f2bf(acc[im][in][q]);
            }
        }
    }
}
/* ==================================================== mtp_enter from 32 tokens up
 *
 * THE TILED KERNEL ABOVE IS THE SHAPE FOR A FEW ROWS -- a draft round -- and past that it pays
 * three costs in proportion to the rows. It computes the embedding term `ne[r] . fc_e[i]` once for each
 * of a token's hc output rows, because the two products share its accumulator; every one of its
 * 40 column blocks sums the squares of its 128 rows again before its first weight byte; and each
 * of those blocks re-stages and re-normalises the same A rows. At 2048 tokens that is 6.2 ms a
 * chunk, ~22 TF/s of useful work.
 *
 * HERE THE TWO PRODUCTS ARE SEPARATE GEMMS. The scales are one pass, a wave a row
 * (mtp_prefill_scale_kernel). The embedding term is a [rows, n] GEMM over the tokens, kept in f32
 * in scratch; the stream term is the [rows * hc, n] GEMM, and its epilogue adds the token's
 * embedding term before the one rounding to bf16. Both are mtp_prefill_gemm_kernel: 128 x 128
 * tiles, eight waves each holding 64 x 32, the next K tile in registers while this one computes,
 * the A tile normalised and the weight rows decoded on the way into LDS.
 *
 * THE BYTES DIFFER FROM THE TILED KERNEL'S. That one adds both products into one accumulator a K
 * step at a time; this one sums each over all of K and adds the two once. Same terms, different
 * rounding, on a head whose output is a draft the trunk verifies -- the tiled gated residual's
 * scales already differ from its per-row form's in the same way. */
/* THE FORM CHANGES AT 32 TOKENS. Below it the grid is n/128 = 20 blocks and the three launches
 * cost ~100 us whatever the rows; the tiled kernel is under that there and over it above.
 * rad-kbench on the Flash-Next container, us a call (tiled / this form):
 *
 *      tokens    8        16       32        64        128       512        2048
 *              52 / 98  88 / 105  182 / 117  267 / 158  433 / 213  1540 / 627  6180 / 2430 */
constexpr long kMtpPrefillRows = 32;

/* One wave a row: rows [0, R) are the output rows (t, k) over `h` -- sub-stream k of token t, or
 * the whole wide row when `flat` -- and rows [R, R + rows) are the tokens over `e`. A lane sums
 * j = lane, lane + 32, ... in order, eight loads ahead of the adds. */
__global__ __launch_bounds__(kThreads)
void mtp_prefill_scale_kernel(const unsigned short* __restrict__ h,
                              const unsigned short* __restrict__ e,
                              float* __restrict__ rs, float* __restrict__ re,
                              long rows, long n, int hc, int flat, float eps, long hrow,
                              long erow) {
    const int  lane = threadIdx.x & (kWave - 1);
    const long row  = (long)blockIdx.x * kWaves + threadIdx.x / kWave;
    const long R    = rows * hc;
    if (row >= R + rows) return;
    const bool tok = row >= R;
    const long len = tok ? n : (flat ? (long)hc * n : n);
    const unsigned short* p = tok ? e + (row - R) * erow
                                  : h + (row / hc) * hrow + (flat ? 0 : (row % hc) * n);
    constexpr int kPre = 8;
    float acc = 0.0f;
    long j = lane;
    for (; j + (long)(kPre - 1) * kWave < len; j += (long)kPre * kWave) {
        unsigned short v[kPre];
#pragma unroll
        for (int q = 0; q < kPre; ++q) v[q] = p[j + (long)q * kWave];
#pragma unroll
        for (int q = 0; q < kPre; ++q) { const float f = bf2f(v[q]); acc += f * f; }
    }
    for (; j < len; j += kWave) { const float f = bf2f(p[j]); acc += f * f; }
    acc = wave_sum(acc);
    if (lane == 0) {
        const float s = __frsqrt_rn(acc / (float)len + eps);
        if (tok) re[row - R] = s; else rs[row] = s;
    }
}

#define MP_BK    64                  /* K a staged tile: four WMMA k steps */
#define MP_PITCH (MP_BK + 8)         /* 144-byte LDS rows, as r4d_gemm_bf16_nt_tiled's */
#define MP_TM    4                   /* WMMA tiles a wave over M ... */
#define MP_TN    2                   /* ... and over N: 64 x 32 a wave, 128 x 128 a block */

/* C = normalise(A) @ fc^T for one of the two products. EPASS: A is `e`, one row a token, and the
 * f32 result goes to `eterm`. Otherwise A is `h` read as rows (t, k), and the result plus the
 * token's `eterm` row is rounded once into `x`. The weight is bf16 rows, or at F8 E4M3 rows with
 * their f32 128-column group scales in each row's tail (r4d_hc_fp8.h), `fld` bytes apart. */
template <bool F8, bool EPASS>
__global__ __launch_bounds__(kThreads)
void mtp_prefill_gemm_kernel(const unsigned short* __restrict__ src, long srow,
                             const float* __restrict__ gain, const float* __restrict__ rscale,
                             const void* __restrict__ fcv, long fld,
                             float* __restrict__ eterm, unsigned short* __restrict__ x,
                             long rows, long n, int hc, float wadd, long xrow) {
    constexpr int BM = 2 * MP_TM * 16, BN = 4 * MP_TN * 16;
    constexpr int CPR = MP_BK / 8;                      /* 16-byte chunks a staged row */
    constexpr int AL = BM * CPR / kThreads, WL = BN * CPR / kThreads;
    static_assert(BM * CPR % kThreads == 0 && BN * CPR % kThreads == 0, "whole chunks");
    __shared__ __attribute__((aligned(16))) unsigned short sA[BM * MP_PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short sW[BN * MP_PITCH];

    const int  tid = threadIdx.x, wave = tid / kWave, lane = tid & (kWave - 1);
    const int  wm = wave / 4, wn = wave % 4;
    const long Mr = EPASS ? rows : rows * (long)hc;     /* A rows of this product */
    const long m0 = (long)blockIdx.y * BM, n0 = (long)blockIdx.x * BN;

    /* A thread's staging slots: chunk c = tid + 256 i is tile row c / CPR, K offset (c % CPR) * 8.
     * Per slot the row's source, its gain offset and its scale, fixed for the whole K loop. */
    const unsigned short* asrc[AL];
    const float* again[AL];
    float asc[AL];
    unsigned short* adst[AL];
#pragma unroll
    for (int i = 0; i < AL; ++i) {
        const int  c = tid + kThreads * i, r = c / CPR, k8 = (c % CPR) * 8;
        const long gr = m0 + r < Mr ? m0 + r : Mr - 1;
        const long t = EPASS ? gr : gr / hc, kk = EPASS ? 0 : gr % hc;
        asrc[i]  = src + t * srow + kk * n + k8;
        again[i] = gain + kk * n + k8;
        asc[i]   = rscale[gr];
        adst[i]  = sA + r * MP_PITCH + k8;
    }
    const unsigned char* wrow[WL];
    const unsigned short* wsrc[WL];
    unsigned short* wdst[WL];
    int wk8[WL];
#pragma unroll
    for (int i = 0; i < WL; ++i) {
        const int  c = tid + kThreads * i, r = c / CPR, k8 = (c % CPR) * 8;
        const long gj = n0 + r < n ? n0 + r : n - 1;
        wrow[i] = (const unsigned char*)fcv + gj * fld;
        wsrc[i] = (const unsigned short*)fcv + gj * n + k8;
        wk8[i]  = k8;
        wdst[i] = sW + r * MP_PITCH + k8;
    }

    const unsigned short* fa = sA + (wm * MP_TM * 16 + (lane & 15)) * MP_PITCH + 4 * (lane >> 4);
    const unsigned short* fw = sW + (wn * MP_TN * 16 + (lane & 15)) * MP_PITCH + 4 * (lane >> 4);

    v8f acc[MP_TM][MP_TN];
#pragma unroll
    for (int i = 0; i < MP_TM; ++i)
#pragma unroll
        for (int j = 0; j < MP_TN; ++j) acc[i][j] = (v8f)(0.f);

    /* The next K tile's raw bytes, in registers while this one computes. */
    uint4  ra[AL];
    float4 rg0[AL], rg1[AL];
    uint2  rc[WL];
    float  rsc[WL];
    uint4  rw[WL];
    auto fetch = [&](long k0) {
#pragma unroll
        for (int i = 0; i < AL; ++i) {
            ra[i]  = *(const uint4*)(asrc[i] + k0);
            rg0[i] = *(const float4*)(again[i] + k0);
            rg1[i] = *(const float4*)(again[i] + k0 + 4);
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            if constexpr (F8) {
                const long k = k0 + wk8[i];
                rc[i]  = *(const uint2*)(wrow[i] + k);
                rsc[i] = ((const float*)(wrow[i] + n))[k / r4d_hc8::kDownGroup];
            } else {
                rw[i] = *(const uint4*)(wsrc[i] + k0);
            }
        }
    };
    fetch(0);

    const long nk = n / MP_BK;
    for (long kt = 0; kt < nk; ++kt) {
        __syncthreads();                                /* the previous tile's fragments are read */
#pragma unroll
        for (int i = 0; i < AL; ++i) {
            union { uint4 v; unsigned short s[8]; } in, out;
            in.v = ra[i];
            const float g[8] = { rg0[i].x, rg0[i].y, rg0[i].z, rg0[i].w,
                                 rg1[i].x, rg1[i].y, rg1[i].z, rg1[i].w };
#pragma unroll
            for (int q = 0; q < 8; ++q) out.s[q] = f2bf(bf2f(in.s[q]) * asc[i] * (g[q] + wadd));
            *(uint4*)adst[i] = out.v;
        }
#pragma unroll
        for (int i = 0; i < WL; ++i) {
            if constexpr (F8) {
                float f[8];
                e4m3x8(rc[i].x, rc[i].y, f);
                union { uint4 v; unsigned short s[8]; } o;
#pragma unroll
                for (int q = 0; q < 8; ++q) o.s[q] = f2bf(f[q] * rsc[i]);
                *(uint4*)wdst[i] = o.v;
            } else {
                *(uint4*)wdst[i] = rw[i];
            }
        }
        __syncthreads();
        if (kt + 1 < nk) fetch((kt + 1) * MP_BK);
#pragma unroll
        for (int s = 0; s < MP_BK / 16; ++s) {
            v8bf af[MP_TM], bf[MP_TN];
#pragma unroll
            for (int i = 0; i < MP_TM; ++i) af[i] = fragP(fa + i * 16 * MP_PITCH + s * 16);
#pragma unroll
            for (int j = 0; j < MP_TN; ++j) bf[j] = fragP(fw + j * 16 * MP_PITCH + s * 16);
#pragma unroll
            for (int i = 0; i < MP_TM; ++i)
#pragma unroll
                for (int j = 0; j < MP_TN; ++j) acc[i][j] = mma(af[i], bf[j], acc[i][j]);
        }
    }

    /* Lane l holds column l & 15 of a tile and rows 8 (l >> 4) + e. */
#pragma unroll
    for (int i = 0; i < MP_TM; ++i)
#pragma unroll
        for (int j = 0; j < MP_TN; ++j) {
            const long col = n0 + (wn * MP_TN + j) * 16 + (lane & 15);
            const long mb  = m0 + (wm * MP_TM + i) * 16 + 8 * (lane >> 4);
            if (col >= n) continue;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const long m = mb + q;
                if (m >= Mr) continue;
                if constexpr (EPASS) {
                    eterm[m * n + col] = acc[i][j][q];
                } else {
                    const long t = m / hc, kk = m % hc;
                    x[t * xrow + kk * n + col] = f2bf(acc[i][j][q] + eterm[t * n + col]);
                }
            }
        }
}
#undef MP_BK
#undef MP_PITCH
#undef MP_TM
#undef MP_TN

/* The prefill form's scratch: the stream rows' scales, the tokens' scales, and the embedding term
 * in f32, each region 256-byte aligned. */
static inline long long mtp_prefill_bytes(long long rows, long long n, long long hc) {
    auto al = [](long long b) { return (b + 255) & ~255LL; };
    return al(rows * hc * 4) + al(rows * 4) + al(rows * n * 4);
}

/* ==================================================== HOW MANY WORKGROUPS THE DOWN PRODUCT GETS
 *
 * THE SPLIT HAS TO COUNT THE ROW TILES, NOT JUST THE COLUMN ONES. Taking `nb` as ceil(ncol/kTNT)
 * alone -- five or six workgroups -- and choosing `ks` to bring that to kTKSPLIT_WANT is right at
 * one row tile and wrong at any other; disabling the split outright at rt = 128, on the grounds
 * that the prefill form does not need it, makes it worse still.
 *
 * WHAT THAT COSTS: rt is 128 at or above kTileMinRows = 64 rows, and a decode step has
 * C * (1 + n_spec) rows -- so SIXTEEN concurrent sequences at depth 3 is exactly 64 rows and falls
 * onto the prefill tile with no split. Six workgroups on a 64-CU part. That is a throughput CLIFF
 * of about 25% at one particular concurrency, invisible from any smaller one, because the usual
 * shape is eight slots and 32 rows.
 *
 * So `nrt` enters here and there is no rt = 128 special case: the split is a function of the WHOLE
 * grid, it turns itself off exactly when the grid is already wide enough (about 2800 rows at this
 * model's ncol), and no row count has a cliff in it. */
static inline long hc_down_kchunk(long long lr, long long HN, long long nrt) {
    const long long nb     = ((lr + kTNT - 1) / kTNT) * (nrt > 0 ? nrt : 1);
    const long long chunks = (HN + kTKT - 1) / kTKT;
    long long want = (kTKSPLIT_WANT + nb - 1) / nb;
    if (want < 1) want = 1;
    if (want > chunks) want = chunks;
    return (long)(((chunks + want - 1) / want) * kTKT);
}
/* THE TILE A ROW COUNT PICKS. One place, because the planner and the call site have to agree.
 *
 * STRICTLY GREATER, and the character matters. A 128-row tile at EXACTLY 64 rows stages 128 rows
 * of A to use 64 -- the staging loop clamps the overshoot to `rows - 1`, so half the LDS traffic
 * and half the WMMA slots go to duplicate rows -- and its 26112 B of LDS puts two workgroups on a
 * WGP where the 64-row tile puts three. It buys nothing in return: the weight is re-read
 * ceil(rows/rt) times and ceil(64/64) is ceil(64/128) is one. Above 64 rows it does halve that
 * re-read and is the right tile. A `>=` here costs several milliseconds a step at exactly the row
 * count where the two tiles are equivalent on traffic. */
static inline int hc_down_rt(long long rows) {
    return (rows > kTileMinRows) ? 128 : (rows > 32 ? 64 : 32);
}

/* THE PARTIAL BUFFER FOR ANY ROW COUNT UP TO THE BAND'S M, in bytes.
 *
 * The planner does not know the step's rows and `ks * rows` is NOT monotone in rows: a smaller row
 * count takes a coarser tile and therefore a DEEPER split, so an endpoint is not the worst case.
 * This walks the band instead. Past the point where the row tiles alone cover the part the split
 * is off and the buffer grows linearly, so the walk stops there and takes M as the last candidate.
 *
 * The rule is: derive the split from `lr` on BOTH sides, and size the buffer for the widest `ncol`
 * a gain fold can produce. Breaking it overruns the partial buffer, which the kbench red zone
 * catches. */
static inline long long hc_part_bytes(long long M, long long lr, long long HN) {
    const long long ncol = lr + (long long)kHcMax;
    const long long hi   = (M < 4096) ? M : 4096;
    long long worst = 0;
    for (long long r = 1; r <= hi; ++r) {
        const long long nrt = (r + hc_down_rt(r) - 1) / hc_down_rt(r);
        const long long ks  = (HN + hc_down_kchunk(lr, HN, nrt) - 1) / hc_down_kchunk(lr, HN, nrt);
        const long long b   = ks * r * ncol * 4;
        if (b > worst) worst = b;
    }
    if (M > hi) {
        const long long nrt = (M + hc_down_rt(M) - 1) / hc_down_rt(M);
        const long long ks  = (HN + hc_down_kchunk(lr, HN, nrt) - 1) / hc_down_kchunk(lr, HN, nrt);
        const long long b   = ks * M * ncol * 4;
        if (b > worst) worst = b;
    }
    return worst;
}

static inline void hc_down_tiled_launch(int rt, bool f8, const unsigned short* h, const float* w,
                                        const void* down, long dld, const float* scale,
                                        const unsigned short* injw,
                                        unsigned short* dbuf, unsigned short* inj, float* part,
                                        long rows, long n, int hc, long lr, long ncol, long hrow,
                                        long irow, long kc, float wadd, RadStream s) {
    const long HN = (long)hc * n;
    const dim3 gd((unsigned)((ncol + kTNT - 1) / kTNT), (unsigned)((rows + rt - 1) / rt),
                  (unsigned)((HN + kc - 1) / kc));
#define HC_DOWN_CASE(RT_, F8_)                                                                   \
    hipLaunchKernelGGL((hc_down_tiled_kernel<RT_, kTNT, F8_>), gd, dim3(kThreads), 0,             \
                       (hipStream_t)s, h, w, down, dld, scale, injw, dbuf, inj, part, rows, n,  \
                       hc, lr, ncol, hrow, irow, kc, wadd)
    switch (rt) {
        case 32:  if (f8) HC_DOWN_CASE(32, true);  else HC_DOWN_CASE(32, false);  break;
        case 64:  if (f8) HC_DOWN_CASE(64, true);  else HC_DOWN_CASE(64, false);  break;
        default:  if (f8) HC_DOWN_CASE(128, true); else HC_DOWN_CASE(128, false); break;
    }
#undef HC_DOWN_CASE
}

static inline void hc_down_wide_launch(bool f8, const unsigned short* h, const float* w,
                                       const void* down, long dld, const float* scale,
                                       const unsigned short* injw,
                                       unsigned short* dbuf, unsigned short* inj, float* part,
                                       long rows, long n, int hc, long lr, long ncol, long hrow,
                                       long irow, long kc, float wadd, RadStream s) {
    const long HN = (long)hc * n;
    const dim3 gd((unsigned)((ncol + kWideNT - 1) / kWideNT),
                  (unsigned)((rows + kWideRT - 1) / kWideRT), (unsigned)((HN + kc - 1) / kc));
#define HC_DW_CASE(F8_)                                                                          \
    hipLaunchKernelGGL((hc_down_wide_kernel<kWideNT, F8_>), gd, dim3(kThreads), 0,               \
                       (hipStream_t)s, h, w, down, dld, scale, injw, dbuf, inj, part, rows, n,  \
                       hc, lr, ncol, hrow, irow, kc, wadd)
    if (f8) HC_DW_CASE(true); else HC_DW_CASE(false);
#undef HC_DW_CASE
}

static inline void hc_up_tiled_launch(int rt, bool f8, const void* up, long uld,
                                      const unsigned short* dbuf, unsigned short* ubuf, long rows,
                                      long hn, long lr, RadStream s) {
    const dim3 gu((unsigned)((hn + kTNT - 1) / kTNT), (unsigned)((rows + rt - 1) / rt));
#define HC_UP_CASE(RT_, F8_)                                                                     \
    hipLaunchKernelGGL((hc_up_tiled_kernel<RT_, kTNT, F8_>), gu, dim3(kThreads), 0,               \
                       (hipStream_t)s, up, uld, dbuf, ubuf, rows, hn, lr)
    switch (rt) {
        case 32:  if (f8) HC_UP_CASE(32, true);  else HC_UP_CASE(32, false);  break;
        case 64:  if (f8) HC_UP_CASE(64, true);  else HC_UP_CASE(64, false);  break;
        default:  if (f8) HC_UP_CASE(128, true); else HC_UP_CASE(128, false); break;
    }
#undef HC_UP_CASE
}
}  // namespace

// ============================================================== the plugin ABI entry points

enum { HCE_X = 0, HCE_H };
enum { HCW_Y = 0, HCW_INJ, HCW_H };
enum { HCR_H = 0, HCR_W, HCR_DOWN, HCR_UP, HCR_INJW, HCR_X, HCR_INJ, HCR_Q, HCR_S, HCR_RQ, HCR_RS };

/* The LDS kernel A holds N in, as bf16, beside its own static planes. gfx1201 gives a workgroup
 * 64 KiB, so N may have 64 KiB less kHcReadAStaticLds -- 30576 elements. The row table claims
 * `n <= 7552` and `hc <= 4`, which is 30208 of them, and the shim refuses the rest by name rather
 * than launching a kernel whose shared allocation fails. */
static const long kHcLdsMaxHN = (65536 - kHcReadAStaticLds) / (long)sizeof(unsigned short);

static int hc_common(const RadArgs* a, const RadTensor* h, long long* rows, long long* n,
                     long long* hc, long long* hrow) {
    *n  = r4d_geti_or(a, "n", 0);
    *hc = r4d_geti_or(a, "hc", 0);
    if (*n <= 0 || *hc <= 1 || *hc > kHcMax) return R4D_NO(RAD_E_SHAPE);
    const long long HN = *n * *hc;
    *hrow = r4d_rows_pitch(h, HN, rows);
    if (*hrow < 0 || *rows <= 0) return R4D_NO(RAD_E_STRIDE);
    return RAD_OK;
}

extern "C" int r4d_rad_hc_enter(const RadArgs* a, RadStream s) {
    const RadTensor* x = r4d_opt(a, HCE_X);
    const RadTensor* h = r4d_opt(a, HCE_H);
    if (!x || !h) return R4D_NO(RAD_E_INVAL);
    R4D_DT(x, RAD_BF16); R4D_DT(h, RAD_BF16);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long xrows = 0;
    const long long xrow = r4d_rows_pitch(x, n, &xrows);
    if (xrow < 0 || xrows < rows) return R4D_NO(RAD_E_STRIDE);

    const unsigned blocks = (unsigned)((n + kThreads - 1) / kThreads);
    hipLaunchKernelGGL(hc_enter_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), 0,
                       (hipStream_t)s, (const unsigned short*)x->data, (unsigned short*)h->data,
                       (long)rows, (long)n, (int)hc, (long)xrow, (long)hrow);
    return (int)hipGetLastError();
}

static int rad_hc_write_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* y   = r4d_opt(a, HCW_Y);
    const RadTensor* inj = r4d_opt(a, HCW_INJ);
    const RadTensor* h   = r4d_opt(a, HCW_H);
    if (!y || !inj || !h) return R4D_NO(RAD_E_INVAL);
    R4D_DT(y, RAD_BF16); R4D_DT(inj, RAD_BF16); R4D_DT(h, RAD_BF16);
    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    long long yrows = 0, irows = 0;
    const long long yrow = r4d_rows_pitch(y, n, &yrows);
    const long long irow = r4d_rows_pitch(inj, hc, &irows);
    if (yrow < 0 || irow < 0 || yrows < rows || irows < rows) return R4D_NO(RAD_E_STRIDE);

    /* Eight bf16 a lane when every row is 16-byte aligned (hc_write_vec_kernel, the same bits).
     * The block is the fewest whole waves that split a row evenly: n = 2560 is 2 x 160. */
    if (n % 8 == 0 && yrow % 8 == 0 && hrow % 8 == 0 && r4d_aligned(y->data, 16) &&
        r4d_aligned(h->data, 16)) {
        const long long n8 = n / 8;
        const unsigned vb = (unsigned)((n8 + kThreads - 1) / kThreads);
        const unsigned nt = (unsigned)(((n8 + vb - 1) / vb + kWave - 1) / kWave * kWave);
        return R4D_EMIT(sk, hc_write_vec_kernel, dim3(vb, (unsigned)rows), dim3(nt), 0,
                        (const unsigned short*)y->data,
                        (const unsigned short*)inj->data, (unsigned short*)h->data,
                        (long)rows, (long)n, (int)hc, (long)yrow, (long)irow, (long)hrow);
    }
    const unsigned blocks = (unsigned)((n + kThreads - 1) / kThreads);
    return R4D_EMIT(sk, hc_write_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), 0,
                    (const unsigned short*)y->data,
                    (const unsigned short*)inj->data, (unsigned short*)h->data,
                    (long)rows, (long)n, (int)hc, (long)yrow, (long)irow, (long)hrow);
}
R4D_ENTRY(r4d_rad_hc_write, rad_hc_write_body)


enum { MTE_H = 0, MTE_E, MTE_WH, MTE_WE, MTE_FCH, MTE_FCE, MTE_X };

static int rad_mtp_enter_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const RadTensor* h   = r4d_opt(a, MTE_H);
    const RadTensor* e   = r4d_opt(a, MTE_E);
    const RadTensor* wh  = r4d_opt(a, MTE_WH);
    const RadTensor* we  = r4d_opt(a, MTE_WE);
    const RadTensor* fch = r4d_opt(a, MTE_FCH);
    const RadTensor* fce = r4d_opt(a, MTE_FCE);
    const RadTensor* x   = r4d_opt(a, MTE_X);
    if (!h || !e || !wh || !we || !fch || !fce || !x) return R4D_NO(RAD_E_INVAL);
    R4D_DT(h, RAD_BF16); R4D_DT(e, RAD_BF16); R4D_DT(x, RAD_BF16);
    R4D_DT(wh, RAD_F32); R4D_DT(we, RAD_F32);
    /* BOTH fc MATRICES bf16 OR BOTH E4M3 rows with f32 128-column group scales in each row's tail
     * (r4d_hc_fp8.h's geometry at K = n), as the stored dtype says. */
    const bool f8 = fch->dtype == RAD_F8E4M3;
    if (f8) { R4D_DT(fce, RAD_F8E4M3); } else { R4D_DT(fch, RAD_BF16); R4D_DT(fce, RAD_BF16); }

    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    const long long HN = n * hc;
    /* `nh` and `ne` both live in LDS for the whole of the second phase -- that is what makes one
     * pass over a weight row enough -- so the wide stream plus one narrow row has to fit beside
     * the per-wave staging slots. 64 KiB less the 4 KiB of `s_stage` and the reduction planes. */
    if ((HN + n) > kMtpLdsMaxN) return R4D_NO(RAD_E_SHAPE);

    long long erows = 0, xrows = 0;
    const long long erow = r4d_rows_pitch(e, n, &erows);
    const long long xrow = r4d_rows_pitch(x, HN, &xrows);
    if (erow < 0 || xrow < 0 || erows < rows || xrows < rows) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(wh) < HN || r4d_numel(we) < n) return R4D_NO(RAD_E_SHAPE);
    /* A stored row's pitch, in the stored dtype's elements. At E4M3 the DECLARED shape arrives
     * ([n, n], the stored dtype), so the pitch is derived by the layout's rule -- see hc_read. */
    long long fld = n;
    if (f8) {
        if (!r4d_hc8::group_ok(n, r4d_hc8::kDownGroup)) return R4D_NO(RAD_E_SHAPE);
        if (fch->rank != 2 || fch->shape[0] != n || fch->shape[1] != n ||
            fce->rank != 2 || fce->shape[0] != n || fce->shape[1] != n)
            return R4D_NO(RAD_E_SHAPE);
        fld = r4d_hc8::row_bytes(n, r4d_hc8::kDownGroup);
    } else if (r4d_numel(fch) < n * n || r4d_numel(fce) < n * n) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (!r4d_contig(wh) || !r4d_contig(we) || !r4d_contig(fch) || !r4d_contig(fce))
        return R4D_NO(RAD_E_STRIDE);

    /* `ngroup` == 1 is the FLAT reading -- one rsqrt over the whole wide row instead of one per
     * sub-stream. The checkpoint fits both and only the acceptance rate can tell them apart, so
     * the choice travels as a parameter rather than being compiled in. Default is the grouped
     * form, which is what every gated residual in this architecture does. */
    const long long ngroup = r4d_geti_or(a, "ngroup", hc);
    if (ngroup != 1 && ngroup != hc) return R4D_NO(RAD_E_SHAPE);
    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);

    const unsigned blocks = (unsigned)((n + kMtpOutTB - 1) / kMtpOutTB);
    const size_t   lds    = (size_t)(HN + n) * sizeof(unsigned short);

    /* THE ROW COUNT PICKS THE SHAPE, as it does for the gated residual. One row is the floor and
     * the per-row kernel is at it; a draft round at eight sequences carries eight rows and pays
     * eight times the weight. */
    const long R = rows * hc;
    const bool tileable = (n % kTKT) == 0 && (n % 8) == 0 && (hrow % 8) == 0
        && (((uintptr_t)h->data & 15u) == 0u) && (((uintptr_t)e->data & 15u) == 0u)
        && (((uintptr_t)fch->data & 15u) == 0u) && (((uintptr_t)fce->data & 15u) == 0u)
        && (((uintptr_t)wh->data & 15u) == 0u) && (((uintptr_t)we->data & 15u) == 0u)
        && (erow % 8) == 0;
    /* E4M3 fc rows are read by the TILED kernel at every row count, one row included: at one row
     * it is 46 us against the per-row bf16 kernel's 54, where a per-row E4M3 dot is 57 --
     * the per-row kernel's win is its transposed LDS layout, which the codes cannot use. So the
     * per-row kernel stays bf16's alone, and a shape the tile cannot take is refused. */
    if (f8 && !tileable) return R4D_NO(RAD_E_SHAPE);
    /* THE PREFILL FORM from kMtpPrefillRows tokens, on its scratch: a shape the tile takes is one
     * this form takes, so past the threshold there is no second path. */
    if (rows >= kMtpPrefillRows && tileable) {
        if (!a->scratch || a->scratch_bytes < mtp_prefill_bytes(rows, n, hc))
            return R4D_NO(RAD_E_SCRATCH);
        /* Three launches over a private scratch, which the descriptor form does not cover; it is
         * submitted individually (ABI v10), as hc_read's tiled path is. */
        if (sk.desc) return RAD_E_UNSUPPORTED;
        auto al = [](long long b) { return (b + 255) & ~255LL; };
        float* rs = (float*)a->scratch;
        float* re = (float*)((unsigned char*)a->scratch + al(R * 4));
        float* et = (float*)((unsigned char*)a->scratch + al(R * 4) + al(rows * 4));
        const unsigned sb = (unsigned)((R + rows + kWaves - 1) / kWaves);
        hipLaunchKernelGGL(mtp_prefill_scale_kernel, dim3(sb), dim3(kThreads), 0,
                           (hipStream_t)sk.stream,
                           (const unsigned short*)h->data, (const unsigned short*)e->data, rs, re,
                           (long)rows, (long)n, (int)hc, ngroup == 1 ? 1 : 0, eps, (long)hrow,
                           (long)erow);
        const dim3 ge((unsigned)((n + 127) / 128), (unsigned)((rows + 127) / 128));
        const dim3 gh((unsigned)((n + 127) / 128), (unsigned)((R + 127) / 128));
#define MTP_PF(F8_, EP_, G_, SRC_, SROW_, GAIN_, RSC_, FC_)                                      \
        hipLaunchKernelGGL((mtp_prefill_gemm_kernel<F8_, EP_>), G_, dim3(kThreads), 0,           \
                           (hipStream_t)sk.stream, (const unsigned short*)SRC_, (long)SROW_,    \
                           (const float*)GAIN_, RSC_, (const void*)FC_, (long)fld, et,           \
                           (unsigned short*)x->data, (long)rows, (long)n, (int)hc, wadd,         \
                           (long)xrow)
        if (f8) {
            MTP_PF(true,  true,  ge, e->data, erow, we->data, re, fce->data);
            MTP_PF(true,  false, gh, h->data, hrow, wh->data, rs, fch->data);
        } else {
            MTP_PF(false, true,  ge, e->data, erow, we->data, re, fce->data);
            MTP_PF(false, false, gh, h->data, hrow, wh->data, rs, fch->data);
        }
#undef MTP_PF
        return hipGetLastError() == hipSuccess ? RAD_OK : R4D_NO(RAD_E_DEVICE);
    }
    if ((f8 || rows >= kMtpTileMinRows) && tileable) {
        const int rt = (R > 64) ? 128 : (R > 32) ? 64 : 32;
        const int nt = rt >= 128 ? kMtpTNTWide : kMtpTNTNarrow;
        const dim3 gm((unsigned)((n + nt - 1) / nt), (unsigned)((R + rt - 1) / rt));
#define MTP_TILE_CASE(RT_, NT_, F8_)                                                             \
        emit_rc = R4D_EMIT(sk, (mtp_enter_tiled_kernel<RT_, NT_, F8_>), gm, dim3(kThreads), 0,     \
                           (const unsigned short*)h->data,                                       \
                           (const unsigned short*)e->data, (const float*)wh->data,               \
                           (const float*)we->data, (const void*)fch->data,                       \
                           (const void*)fce->data, (long)fld, (unsigned short*)x->data,          \
                           (long)rows, (long)n, (int)hc, ngroup == 1 ? 1 : 0, eps, wadd,         \
                           (long)hrow, (long)erow, (long)xrow)
        switch (rt) {
            case 32:  if (f8) MTP_TILE_CASE(32, kMtpTNTNarrow, true);
                      else    MTP_TILE_CASE(32, kMtpTNTNarrow, false);  break;
            case 64:  if (f8) MTP_TILE_CASE(64, kMtpTNTNarrow, true);
                      else    MTP_TILE_CASE(64, kMtpTNTNarrow, false);  break;
            default:  if (f8) MTP_TILE_CASE(128, kMtpTNTWide, true);
                      else    MTP_TILE_CASE(128, kMtpTNTWide, false);   break;
        }
#undef MTP_TILE_CASE
        if (emit_rc != RAD_OK) return emit_rc;
        return sk.desc ? RAD_OK : (int)hipGetLastError();
    }

    emit_rc = R4D_EMIT(sk, mtp_enter_kernel, dim3(blocks, (unsigned)rows), dim3(kThreads), lds,
                       (const unsigned short*)h->data,
                       (const unsigned short*)e->data, (const float*)wh->data,
                       (const float*)we->data, (const unsigned short*)fch->data,
                       (const unsigned short*)fce->data, (unsigned short*)x->data,
                       (long)rows, (long)n, (int)hc, ngroup == 1 ? 1 : 0, eps, wadd,
                       (long)hrow, (long)erow, (long)xrow);
    if (emit_rc != RAD_OK) return emit_rc;
    return sk.desc ? RAD_OK : (int)hipGetLastError();
}
R4D_ENTRY(r4d_rad_mtp_enter, rad_mtp_enter_body)

/* The prefill form's scratch, for a band that reaches it. `M` is the band's ceiling and the form
 * is chosen per call, so a band reaching kMtpPrefillRows is sized at its ceiling. */
extern "C" int64_t r4d_rad_mtp_enter_scratch(const RadArgs* a) {
    const long long M  = r4d_geti_or(a, "M", 0);
    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M <= 0 || n <= 0 || hc <= 0) return R4D_NO(RAD_E_SHAPE);
    return M >= kMtpPrefillRows ? mtp_prefill_bytes(M, n, hc) : 0;
}
/* `d` between the two kernels, and the per-stream norm scales beside it. Sized from the BAND's `M`
 * rather than from the operand, because the arena is sized once for the whole band and the launch
 * below checks against the same number -- a scratch hook that answered for the step's actual rows
 * would size the arena for whatever step happened to be asked first. */
extern "C" int64_t r4d_rad_hc_read_scratch(const RadArgs* a) {
    const long long M  = r4d_geti_or(a, "M", 0);
    const long long lr = r4d_geti_or(a, "lowrank", 0);
    const long long n  = r4d_geti_or(a, "n", 0);
    const long long hc = r4d_geti_or(a, "hc", 0);
    if (M <= 0 || lr <= 0 || n <= 0 || hc <= 0) return R4D_NO(RAD_E_SHAPE);
    /* `d` [M, lowrank] bf16, the per-stream norm scales [M, kHcMax] f32, and `u` [M, hc*n] bf16 --
     * 20 KiB a row at this model's shape, which is what buys the up-matvec its own grid.
     *
     * PLUS THE SPLIT-K PARTIALS. Every row count below a few thousand takes a split, so this is
     * not a decode-only buffer and is not bounded by kTileMinRows -- hc_part_bytes
     * walks the band because ks * rows is not monotone in rows. */
    const long long HN   = hc * n;
    /* THE SPLIT MUST BE COMPUTED FROM THE SAME NUMBER IN BOTH PLACES. The call site knows whether
     * this connection has inject weights and the planner does not, so both derive `kc` from `lr`
     * alone and the buffer is sized for the widest ncol a fold can produce. Deriving it from ncol
     * here and from lr there gives two different split counts for the same shape, and the partial
     * buffer then overruns. */
    return M * ((long long)lr * 2 + (long long)kHcMax * 4 + hc * n * 2)
           + hc_part_bytes(M, lr, HN);
}

static int rad_hc_read_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    /* The tiled path below launches directly -- it declines to describe -- and takes the stream
     * as `s`. */
    const RadStream s = sk.stream;
    (void)s;
    const RadTensor* h    = r4d_opt(a, HCR_H);
    const RadTensor* w    = r4d_opt(a, HCR_W);
    const RadTensor* down = r4d_opt(a, HCR_DOWN);
    const RadTensor* up   = r4d_opt(a, HCR_UP);
    const RadTensor* injw = r4d_opt(a, HCR_INJW);
    const RadTensor* x    = r4d_opt(a, HCR_X);
    const RadTensor* inj  = r4d_opt(a, HCR_INJ);
    const RadTensor* q    = r4d_opt(a, HCR_Q);
    const RadTensor* qs   = r4d_opt(a, HCR_S);
    const RadTensor* rq   = r4d_opt(a, HCR_RQ);
    const RadTensor* rqs  = r4d_opt(a, HCR_RS);
    if (!h || !w || !down || !up || !x) return R4D_NO(RAD_E_INVAL);
    R4D_DT(h, RAD_BF16); R4D_DT(w, RAD_F32);
    /* THE MIXING MATRICES ARE BOTH bf16 OR BOTH E4M3 rows (r4d_hc_fp8.h), and which one is the
     * stored dtype's to say: the layout hook that wrote them is the one this row resolved to. */
    const bool f8 = down->dtype == RAD_F8E4M3;
    if (f8) { R4D_DT(up, RAD_F8E4M3); } else { R4D_DT(down, RAD_BF16); R4D_DT(up, RAD_BF16); }
    R4D_DT(x, RAD_BF16);
    /* An inject WEIGHT with no `inj` output is a connection whose two halves disagree about
     * whether there is a write; refused rather than silently dropped. */
    if ((injw != nullptr) != (inj != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (injw) { R4D_DT(injw, RAD_BF16); R4D_DT(inj, RAD_BF16); }
    if ((q != nullptr) != (qs != nullptr)) return R4D_NO(RAD_E_INVAL);
    /* THE CODES' DTYPE IS THE FORMAT: E4M3 for an fp8 trunk, int8 for an int8 one (gemm_i8a8_*),
     * the same scale grid either way (r4d_q8.h). The rotated twin below stays E4M3 -- it is the
     * routed experts' operand, whatever the trunk is. */
    if (q && q->dtype != RAD_F8E4M3 && q->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    if (q) R4D_DT(qs, RAD_F32);
    const bool i8 = q && q->dtype == RAD_I8;
    /* The rotated twin rides in the plain codes' finisher, so it is refused without them. */
    if ((rq != nullptr) != (rqs != nullptr) || (rq && !q)) return R4D_NO(RAD_E_INVAL);
    if (rq) { R4D_DT(rq, RAD_F8E4M3); R4D_DT(rqs, RAD_F32); }

    long long rows = 0, n = 0, hc = 0, hrow = 0;
    const int rc = hc_common(a, h, &rows, &n, &hc, &hrow);
    if (rc != RAD_OK) return rc;
    const long long HN = n * hc;
    if (HN > kHcLdsMaxHN) return R4D_NO(RAD_E_SHAPE);
    /* One workgroup owns 128 channels AND the 128-column scale group over them, so a channel count
     * that is not a whole number of them would split a group across two workgroups -- two amaxes
     * for one scale. */
    if (n % kUpTB) return R4D_NO(RAD_E_SHAPE);

    const long long lr = r4d_geti_or(a, "lowrank", 0);
    if (lr <= 0) return R4D_NO(RAD_E_SHAPE);
    /* A row's pitch in ELEMENTS of the stored dtype: HN or lr for bf16, the stored row's bytes for
     * E4M3, whose dot needs K a multiple of its 256-element step and whose up rows split in four
     * sixteen-code quarters of at most kUpMaxQ loads. */
    long long dld = HN, uld = lr;
    if (f8) {
        const long long gd = r4d_hc8::down_group(HN), gu = r4d_hc8::up_group(lr);
        if (!r4d_hc8::group_ok(HN, gd) || !r4d_hc8::group_ok(lr, gu) || HN % kHc8Step ||
            gu / 16 > kUpMaxQ)
            return R4D_NO(RAD_E_SHAPE);
        /* THE DECLARED SHAPE ARRIVES, NOT THE STORED ONE -- a weight's RadTensor is built from
         * `decl.shape` with the layout's stored dtype, as the fp8 lm_head's is (its launcher says
         * so at length). So the row pitch is DERIVED here by the layout hook's own rule, and what
         * guarantees the plane has it is the engine's layout-tag check, run before a byte moves. */
        if (down->rank != 2 || down->shape[0] != lr || down->shape[1] != HN ||
            up->rank != 2 || up->shape[0] != HN || up->shape[1] != lr)
            return R4D_NO(RAD_E_SHAPE);
        dld = r4d_hc8::row_bytes(HN, gd);
        uld = r4d_hc8::row_bytes(lr, gu);
    } else if (r4d_numel(down) < lr * HN || r4d_numel(up) < HN * lr) {
        return R4D_NO(RAD_E_SHAPE);
    }
    if (r4d_numel(w) < HN) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(w) || !r4d_contig(down) || !r4d_contig(up)) return R4D_NO(RAD_E_STRIDE);
    if (injw && (!r4d_contig(injw) || r4d_numel(injw) < hc * HN)) return R4D_NO(RAD_E_STRIDE);

    long long xrows = 0, irows = 0;
    const long long xrow = r4d_rows_pitch(x, n, &xrows);
    if (xrow < 0 || xrows < rows) return R4D_NO(RAD_E_STRIDE);
    long long irow = hc;
    if (inj) {
        irow = r4d_rows_pitch(inj, hc, &irows);
        if (irow < 0 || irows < rows) return R4D_NO(RAD_E_STRIDE);
    }

    long long qrow = 0, srow = 0;
    if (q) {
        const long long group = r4d_geti_or(a, "group", 0);
        if (group != kFp8Grp) return R4D_NO(RAD_E_SHAPE);
        long long qrows = 0, srows = 0;
        qrow = r4d_rows_pitch(q, n, &qrows);
        srow = r4d_rows_pitch(qs, n / kFp8Grp, &srows);
        if (qrow < 0 || srow < 0 || qrows < rows || srows < rows) return R4D_NO(RAD_E_STRIDE);
    }
    long long rqrow = 0, rsrow = 0;
    if (rq) {
        long long qrows = 0, srows = 0;
        rqrow = r4d_rows_pitch(rq, n, &qrows);
        rsrow = r4d_rows_pitch(rqs, n / kFp8Grp, &srows);
        if (rqrow < 0 || rsrow < 0 || qrows < rows || srows < rows) return R4D_NO(RAD_E_STRIDE);
        /* A lane stores its four codes as one dword. */
        if ((((uintptr_t)rq->data | (uintptr_t)rqrow) & 3u) != 0u) return R4D_NO(RAD_E_STRIDE);
    }

    const float eps  = (float)r4d_getf(a, "eps", 1e-6);
    const float wadd = (float)r4d_getf(a, "wadd", 0.0);

    /* THE CHECK IS AGAINST WHAT THIS LAUNCH TOUCHES, NOT AGAINST THE SCRATCH HOOK. The hook walks
     * the band's row counts to find the worst split -- right once at planning, and a large share
     * of the issuing thread's time when repeated on every launch. `d`, the scales and `u` sit at
     * offsets fixed by the band's M; the split-K partials behind them are checked where the
     * split is known, in the tiled branch below. */
    const long long Mband = r4d_geti_or(a, "M", 0);
    if (Mband <= 0) return R4D_NO(RAD_E_SHAPE);
    /* `d`, the scales and `u` are all sized from the BAND, so a step wider than the band it was
     * planned for does not run slowly -- it writes past three buffers. */
    if (rows > Mband) return R4D_NO(RAD_E_SHAPE);
    const long long fixed = Mband * (lr * 2 + (long long)kHcMax * 4 + HN * 2);
    if (!a->scratch || a->scratch_bytes < fixed) return R4D_NO(RAD_E_SCRATCH);
    unsigned short* dbuf  = (unsigned short*)a->scratch;
    float*          scale = (float*)((unsigned char*)a->scratch + Mband * lr * 2);
    unsigned short* ubuf  = (unsigned short*)((unsigned char*)a->scratch
                                              + Mband * (lr * 2 + kHcMax * 4));

    /* THE COLUMN TILE IS CHOSEN FOR THE GRID AND NOT FOR THE CACHE. A workgroup owning 64 of the
     * 320 columns is five workgroups at decode, on a part with sixty-four compute units -- which
     * costs this op an order of magnitude over what its weight traffic says it should. Narrowing
     * the tile until the grid covers the machine costs one extra pass over `h` per workgroup
     * (20 KiB against the 1.3 MiB of weight the same workgroup reads at the widest tile), and at
     * decode that is a megabyte and a half against six and a half of weight.
     *
     * IT ONLY EVER NARROWS: rows past kDecMinRows leave this path entirely for the tiled one
     * below, so the widest tile is never the right answer here and `rows` in the condition is only
     * ever a one-to-seven-row speculative window. */
    const bool can_tile = (n % kTKT) == 0 && (HN % 8) == 0 && (lr % 8) == 0 && (hrow % 8) == 0
                          && (((uintptr_t)h->data & 15u) == 0u)
                          && (((uintptr_t)down->data & 15u) == 0u)
                          && (((uintptr_t)up->data & 15u) == 0u);
    /* THE ROW COUNT PICKS THE SHAPE OF THE OP, and it is not a tuning knob. The per-row kernels
     * read both mixing matrices once A ROW: one row is the floor, four rows is a 4x re-read that
     * an issue-bound kernel hides, and THIRTY-TWO -- eight sequences at depth 3 -- is 210 MB a
     * connection that nothing hides. Left per-row, these two kernels alone account for roughly
     * 45% of the whole cost of going from one sequence to eight.
     *
     * So above kDecMinRows the two products run TILED, and which tile is a function of the rows:
     * 128 when there are enough to fill it (the prefill form), 32 or 64 below that
     * with the down product split over K. Below kDecMinRows the per-row kernels still win --
     * there is not enough re-read to pay for the extra launch and the partials. */
    bool fused = false;
    const bool tiled  = can_tile && rows >= kDecMinRows;
    /* THE TILED PATH DECLINES TO DESCRIBE, and declining is a first-class answer: ABI v10
     * says a kernel that cannot describe a configuration is submitted individually, which is
     * correct and merely slower. This path dispatches through three more launch helpers and takes
     * a second per-stream counter, none of which the descriptor form covers. Declining keeps
     * hc_read describable on the path decode takes without pretending about the one it does not. */
    if (tiled && sk.desc) return RAD_E_UNSUPPORTED;
    if (tiled) {
        const int  rt   = hc_down_rt(rows);
        /* The inject gains are four more columns of the same GEMM when there are any. */
        const bool fold = injw != nullptr && inj != nullptr;
        const long ncol = fold ? lr + hc : lr;
        const long nrt  = (rows + rt - 1) / rt;
        const bool wdown = rt == kWideRT && g_hc_prefill_form != 1;
        const long kc   = hc_down_kchunk(lr, HN, nrt);
        const int  ks   = (int)((HN + kc - 1) / kc);
        if (ks > 1 && a->scratch_bytes < fixed + (long long)ks * rows * ncol * 4)
            return R4D_NO(RAD_E_SCRATCH);
        float*     part = (ks > 1) ? (float*)((unsigned char*)a->scratch + fixed) : nullptr;

        /* The per-stream norm scales, on a grid: one workgroup a (row, stream). */
        hipLaunchKernelGGL(hc_norm_scale_kernel,
                           dim3((unsigned)hc, (unsigned)rows), dim3(kThreads), 0, (hipStream_t)s,
                           (const unsigned short*)h->data, scale,
                           (long)rows, (long)n, (long)hrow, eps);
        if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

        if (wdown)
            hc_down_wide_launch(f8, (const unsigned short*)h->data, (const float*)w->data,
                                down->data, (long)dld, scale,
                                fold ? (const unsigned short*)injw->data : nullptr,
                                dbuf, fold ? (unsigned short*)inj->data : nullptr, part,
                                (long)rows, (long)n, (int)hc, (long)lr, ncol, (long)hrow,
                                (long)irow, kc, wadd, s);
        else
            hc_down_tiled_launch(rt, f8, (const unsigned short*)h->data, (const float*)w->data,
                                 down->data, (long)dld, scale,
                                 fold ? (const unsigned short*)injw->data : nullptr,
                                 dbuf, fold ? (unsigned short*)inj->data : nullptr, part,
                                 (long)rows, (long)n, (int)hc, (long)lr, ncol, (long)hrow,
                                 (long)irow, kc, wadd, s);
        if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

        if (ks > 1) {
            const long tot = (long)rows * ncol;
            hipLaunchKernelGGL(hc_down_reduce_kernel,
                               dim3((unsigned)((tot + kThreads - 1) / kThreads)), dim3(kThreads),
                               0, (hipStream_t)s,
                               part, dbuf, fold ? (unsigned short*)inj->data : nullptr,
                               (long)rows, (long)lr, ncol, (long)irow, ks, 1.0f / (float)hc);
            if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
        }

        /* THE UP PRODUCT AND THE MIX IN ONE LAUNCH, when the tile divides into (stream, channel)
         * pairs and the finisher's counters fit. At the 32- and 64-row tiles that is
         * hc_up_mix_tiled_kernel. The prefill tile takes hc_up_mix_wide_kernel instead, whose
         * lanes hold a channel's sub-streams in registers: the pair form's two staging planes on
         * top of a 128-row A tile would be 58 KiB of LDS and one workgroup a CU. */
        const int ipb = (hc > 0 && (kTNT % hc) == 0) ? kTNT / hc : 0;
        const long ntile = (long)(n / kUpTB) * ((rows + rt - 1) / rt);
        unsigned int* mctr = nullptr;
        const bool wide = rt == kWideRT && hc == 4 && g_hc_prefill_form != 1;
        if (g_hc_prefill_form == 2 && !wide) return R4D_NO(RAD_E_SHAPE);
        fused = wide || (rt <= 64 && ipb > 0 && (kUpTB % ipb) == 0 && (n % ipb) == 0
                         && ntile <= (long)kMaxMixCtr);
        if (fused && !wide && q) { mctr = hc_mix_ctr(s); fused = (mctr != nullptr); }
        if (fused && wide)
            hc_up_mix_wide_launch(f8, i8, up->data, (long)uld, dbuf, (const unsigned short*)h->data,
                                  (const float*)w->data, scale, (unsigned short*)x->data,
                                  q ? (unsigned char*)q->data : nullptr,
                                  q ? (float*)qs->data : nullptr,
                                  rq ? (unsigned char*)rq->data : nullptr,
                                  rq ? (float*)rqs->data : nullptr,
                                  (long)rows, (long)n, (long)lr, (long)hrow, (long)xrow,
                                  (long)qrow, (long)srow, (long)rqrow, (long)rsrow, wadd, s);
        else if (fused)
            hc_up_mix_tiled_launch(rt, f8, i8, up->data, (long)uld, dbuf,
                                   (const unsigned short*)h->data, (const float*)w->data, scale,
                                   (unsigned short*)x->data,
                                   q ? (unsigned char*)q->data : nullptr,
                                   q ? (float*)qs->data : nullptr,
                                   rq ? (unsigned char*)rq->data : nullptr,
                                   rq ? (float*)rqs->data : nullptr, mctr,
                                   (long)rows, (long)n, (int)hc, (long)lr, (long)hrow,
                                   (long)xrow, (long)qrow, (long)srow, (long)rqrow, (long)rsrow,
                                   wadd, s);
        else
            hc_up_tiled_launch(rt, f8, up->data, (long)uld, dbuf, ubuf,
                               (long)rows, (long)HN, (long)lr, s);
        if (hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    } else {
    long long ta = kDownTAMax;
    /* TWO PER WGP, WHICH IS ONE PER CU -- and this is the kernel want_wg()'s note anticipates.
     * `multiProcessorCount` reports WGPs on RDNA and this part has 64 CUs in 32 of them, so a grid
     * sized to want_wg() alone covers half the machine. At a depth-3 verify that is four rows at
     * ta = 32, or 44 blocks, which a one-WGP target accepts and a one-CU target does not. Widening
     * the target is worth ~1.3% of the step at that shape and is inside the noise at one row;
     * below about 64 blocks the step falls off hard either way, which is the half-idle part
     * showing up directly.
     *
     * THE LADDER HALVES ta, so it only ever lands on 44 / 84 / 164 blocks. Across every value of ta
     * at ONE row -- where the median launch of a decode step lives, and where this rule already
     * runs at its floor of 4 -- the floor IS the optimum, and it is a SHARP one: ta 4 (81 blocks)
     * is 18.6 us, ta 5 (65 blocks) is +50% and ta 3 (108 blocks) is +76%, while ta 11, 12 and 24
     * (31, 28 and 15 blocks) all sit within 1.7 us of each other. Whatever ta 4 does right is NOT a
     * smooth function of the block count -- 65 blocks is worse than both 81 and 31 -- so do not
     * reason about this grid from CU occupancy or from a balance argument. Leave it at 4.
     *
     * BLOCK COUNT IS RULED OUT AS A DIRECTION HERE. With the prologue's traffic worth zero (above)
     * and the dot already at 546 GB/s, this kernel has no lever left that is not a restructuring. */
    while (ta > 4 && ((lr + ta - 1) / ta) * rows < 2 * want_wg()) ta >>= 1;

#define HC_READ_A(F8_)                                                                           \
    R4D_EMIT(sk, hc_read_a_kernel<F8_>, dim3((unsigned)((lr + ta - 1) / ta) + 1u, (unsigned)rows), \
             dim3(kThreads), (unsigned)(HN * sizeof(unsigned short)),                             \
             (const unsigned short*)h->data, (const float*)w->data, (const void*)down->data,       \
             (long)dld, injw ? (const unsigned short*)injw->data : nullptr, dbuf,                 \
             inj ? (unsigned short*)inj->data : nullptr, scale, (long)rows, (long)n, (int)hc,     \
             (long)lr, (long)ta, (long)hrow, (long)irow, eps, wadd)
    emit_rc = f8 ? HC_READ_A(true) : HC_READ_A(false);
#undef HC_READ_A
    if (emit_rc != RAD_OK) return R4D_NO(RAD_E_DEVICE);
    if (!sk.desc && hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);

    /* THE UP PRODUCT AND THE MIX IN ONE LAUNCH whenever a channel's hc rows divide a workgroup's
     * 64 and the finisher's counters fit -- which at this model's hc of 4 is every row count this
     * path takes. A counter makes the launch indescribable, as on the tiled path, so a describe
     * that would need one declines. */
    const long long ipb   = (hc > 0 && (kUpRows % hc) == 0) ? kUpRows / hc : 0;
    const long long slots = ((rows + hc_up_rg(rows, lr) - 1) / hc_up_rg(rows, lr)) * (n / kUpTB);
    unsigned int* mctr = nullptr;
    fused = ipb > 0 && (kUpTB % ipb) == 0 && (n % ipb) == 0 && slots <= (long long)kMaxMixCtr;
    if (fused && q) {
        if (sk.desc) return RAD_E_UNSUPPORTED;
        mctr = hc_mix_ctr(s);
        fused = (mctr != nullptr);
    }
    if (fused)
        emit_rc = hc_up_mix_launch(up, (long)uld, f8, i8, dbuf, (const unsigned short*)h->data,
                                   (const float*)w->data,
                                   scale, (unsigned short*)x->data,
                                   q ? (unsigned char*)q->data : nullptr,
                                   q ? (float*)qs->data : nullptr,
                                   rq ? (unsigned char*)rq->data : nullptr,
                                   rq ? (float*)rqs->data : nullptr, mctr, rows, n, (int)hc, lr,
                                   hrow, xrow, qrow, srow, rqrow, rsrow, wadd, sk);
    else
        emit_rc = hc_up_launch(up, (long)uld, f8, dbuf, ubuf, rows, HN, lr, sk);
    if (emit_rc != RAD_OK) return R4D_NO(RAD_E_DEVICE);
    if (!sk.desc && hipGetLastError() != hipSuccess) return R4D_NO(RAD_E_DEVICE);
    }

    if (!fused) {
#define HC_MIX_EMIT(I8_)                                                                         \
        R4D_EMIT(sk, hc_mix_kernel<I8_>,                                                         \
                 dim3((unsigned)(n / kUpTB), (unsigned)rows),                                    \
                 dim3(kThreads), 0,                                                              \
                 (const unsigned short*)h->data, (const float*)w->data, ubuf, scale,             \
                 (unsigned short*)x->data, q ? (unsigned char*)q->data : nullptr,                \
                 q ? (float*)qs->data : nullptr,                                                 \
                 rq ? (unsigned char*)rq->data : nullptr,                                        \
                 rq ? (float*)rqs->data : nullptr,                                               \
                 (long)rows, (long)n, (int)hc, (long)hrow, (long)xrow,                           \
                 (long)qrow, (long)srow, (long)rqrow, (long)rsrow, wadd)
        emit_rc = i8 ? HC_MIX_EMIT(true) : HC_MIX_EMIT(false);
#undef HC_MIX_EMIT
    }

    if (emit_rc != RAD_OK) return emit_rc;
    return sk.desc ? RAD_OK : (int)hipGetLastError();
}
R4D_ENTRY(r4d_rad_hc_read, rad_hc_read_body)

extern "C" void r4d_hc_read_pin_prefill(int form) { g_hc_prefill_form = form; }
