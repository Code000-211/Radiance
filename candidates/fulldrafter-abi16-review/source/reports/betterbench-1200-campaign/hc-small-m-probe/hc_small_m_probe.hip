// Standalone experiment, NOT a plugin or an engine replacement. Seven selectable variants:
//
// --variant wmma (default, the recorded three-launch evidence):
// numerik: verlustfrei-nicht-bitgenau. At M<8 only down/inject reduction changes.
// M=8 is a crossover control: the frozen row may also use its tiled up path,
// whereas this probe retains the original per-row FP32-scaled up GEMV.
// E4M3 codes enter WMMA as exactly representable BF16; their scales stay FP32.
// Never use upstream e4m3x8_bf16(code, scale): that would narrow scaled weights.
// --variant wmma-fusednorm: two launches. Each down CTA repeats the unchanged
// 256-thread RMS reduction; only CTA (0,0,0) publishes scales for up_mix.
// WMMA partials and every output must match the three-launch variant bytewise.
//
// --variant rowpair: numerik: bitgenau. Two launches like the frozen M<8 path:
// at M=2..4 rowpair_a<2> (hc_read_a_kernel<true> with two rows a CTA sharing only W_down/W_inj/
// norm-weight loads; per-row norm, lane order, FMA chain, prefetch depths and BF16 roundings
// unchanged; M=3's ghost row re-reads row r0 and publishes nothing). At M=1 the UNCHANGED
// upstream hc_read_a_kernel<true> runs with the frozen launch arguments: an unchanged control,
// not an optimized path. Both are followed by the unchanged upstream hc_up_mix_launch. M=1..4
// only: at M>=kDecMinRows (8) the frozen row dispatches its tiled path, so M=8 is skipped.
// Comparator is byte-exact for x, inject, q, qs, rq, rqs and for d vs frozen dbuf.
// --variant up-wmma: original down/inject/norm, then unscaled-BF16 WMMA up/mix.
// M=1..4 only. Each 80-K dot is reordered; FP32 scales, the four-group sum tree,
// sigmoid BF16, normalized-stream BF16 and output BF16/quantization stay explicit.
// --variant up-wmma-group: one CTA owns a complete quantization group, with no counters.
// Stage 16 K columns at a time to keep its larger channel tile within the LDS limit.
// --variant up-wmma-direct: the same counter-free group, but decode B into WMMA registers.
// --variant up-wmma-split: 32-channel up/mix, then stateless quantization; no counters.
// Prior rowpair attempts, both 36/36 byte-exact and slower than frozen at every M:
//  1) one rowpair_a with in-kernel NR=1/NR=2 branch, prefetch 4/8: 256 VGPRs, 532 B spill.
//  2) rowpair_a<1>/<2>, prefetch 2/4: NR1 256 VGPRs 480 B spill, NR2 98 VGPRs 0 spill.
//
// From the worktree root (no upstream build or runtime mutation):
// TMPDIR=/tmp /opt/rocm/bin/hipcc -O3 -std=c++17 -fno-fast-math \
//   -ffp-contract=off --offload-arch=gfx1201 \
//   -Iabi -Ilibr4d -ffunction-sections -fdata-sections \
//   reports/betterbench-1200-campaign/hc-small-m-probe/hc_small_m_probe.hip \
//   -Wl,--gc-sections -ldl -pthread -o /tmp/hc-small-m-probe
// (append -Rpass-analysis=kernel-resource-usage for compiler VGPR/LDS/occupancy remarks)
// /tmp/hc-small-m-probe --device 0 --warmup 100 --iters 200 --repeats 9
// /tmp/hc-small-m-probe --variant rowpair --device 0 --warmup 100 --iters 200 --repeats 9
// Defaults: M=1,2,3,4,8 (rowpair: 1,2,3,4); three seeds; none/FP8+rotated/INT8+rotated outputs.
// --m restricts to one supported M value; --check-only disables all timing.
// HIP events measure complete ordinary-stream submissions, including submission gaps.
// HC_PROBE_AQL selects engine-owned AQL streams/events when linked with rad_core
// and its exported HIP interposition symbols. This measures live submissions,
// not PassTape replay; ordinary HIP timing remains the default build.

#ifdef __FAST_MATH__
#error "This experiment must not use fast math."
#endif

#include "r4d_emit.h"
// Include immutable upstream helpers, not a copy of thousands of lines. Do not
// emit its plugin entries into this executable or interpose on the frozen DSO.
#undef R4D_ENTRY
#define R4D_ENTRY(sym, body)
#define r4d_rad_hc_enter probe_unused_hc_enter
#define r4d_rad_mtp_enter_scratch probe_unused_mtp_scratch
#define r4d_rad_hc_read_scratch probe_unused_hc_scratch
#define r4d_hc_read_pin_prefill probe_unused_pin_prefill
#include "../../../libr4d/r4d_hc_bf16.hip"
#undef r4d_rad_hc_enter
#undef r4d_rad_mtp_enter_scratch
#undef r4d_rad_hc_read_scratch
#undef r4d_hc_read_pin_prefill
#undef R4D_ENTRY
#ifdef HC_PROBE_AQL
#include "rad_device.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <initializer_list>
#include <limits.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace probe {
constexpr int N = 2560, HC = 4, LR = 320, HN = N * HC;
constexpr int NC = LR + HC, GROUP = 128, SPLITS = 8;
constexpr int RT = 16, NT = 32, DT = 64, PITCH = GROUP + 8;
constexpr int NG = HN / GROUP, GS = NG / SPLITS;
constexpr long DLD = r4d_hc8::row_bytes(HN, r4d_hc8::down_group(HN));
constexpr long ULD = r4d_hc8::row_bytes(LR, r4d_hc8::up_group(LR));
static_assert(N % GROUP == 0 && NG % SPLITS == 0 && NT == 2 * 16);
static_assert(DLD == 10560 && ULD == 336 && PITCH % 8 == 0);
constexpr const char* ORACLE =
    "/tmp/radiance-mtp8-control-88ohufxq/radiance_home/kernels/libr4d.so";

// 88 CTAs: 10 down tiles plus one inject-tail tile, times 8 K splits.
// Each split owns 10 complete 128-K quantization groups (1280 K).
// A/B pitches are in BF16 elements; vector stores are 16-byte aligned.
// A/B LDS = (16+32)*136*2 = 13056 bytes; two waves compute WMMA.
// Fused norm uses eight waves for normalization/staging and adds shared norm storage.
template<bool FusedNorm>
__global__ __launch_bounds__(FusedNorm ? kThreads : DT)
void down(const unsigned short* __restrict__ h, const float* __restrict__ w,
          const unsigned char* __restrict__ dw, const unsigned short* __restrict__ iw,
          float* __restrict__ scale, float* __restrict__ part, int rows, float wadd) {
    __shared__ __attribute__((aligned(16))) unsigned short a[RT * PITCH];
    __shared__ __attribute__((aligned(16))) unsigned short b[NT * PITCH];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
    const int c0 = blockIdx.x * NT, z = blockIdx.z;
    v8f total = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    constexpr int Threads = FusedNorm ? kThreads : DT;
    __shared__ float local_scale[RT * kHcMax];
    __shared__ float norm_part[kWaves];
    const float* norms = scale;
    if constexpr (FusedNorm) {
        constexpr int Pre = 10;
        static_assert(N == Pre * kThreads);
        for (int r = 0; r < rows; ++r) for (int c = 0; c < HC; ++c) {
            unsigned short hv[Pre];
#pragma unroll
            for (int p = 0; p < Pre; ++p)
                hv[p] = h[r * HN + c * N + tid + p * kThreads];
            float acc = 0.0f;
#pragma unroll
            for (int p = 0; p < Pre; ++p) {
                const float v = bf2f(hv[p]);
                acc += v * v;
            }
            acc = wave_sum(acc);
            if (lane == 0) norm_part[wave] = acc;
            __syncthreads();
            if (tid == 0) {
                float ss = 0.0f;
                for (int k = 0; k < kWaves; ++k) ss += norm_part[k];
                const float norm = __frsqrt_rn(ss / (float)N + 1e-6f);
                local_scale[r * kHcMax + c] = norm;
                if (blockIdx.x == 0 && blockIdx.z == 0)
                    scale[r * kHcMax + c] = norm;
            }
            __syncthreads();
        }
        norms = local_scale;
    }
    for (int g = z * GS; g < (z + 1) * GS; ++g) {
        const int k0 = g * GROUP, stream = k0 / N;
        __syncthreads();
        for (int t = tid; t < RT * (GROUP / 8); t += Threads) {
            const int r = t / (GROUP / 8), k8 = (t % (GROUP / 8)) * 8;
            union { uint4 v; unsigned short s[8]; } out;
            out.v = make_uint4(0, 0, 0, 0);
            if (r < rows) {
                union { uint4 v; unsigned short s[8]; } in;
                in.v = *(const uint4*)(h + r * HN + k0 + k8);
                const float sc = norms[r * kHcMax + stream];
#pragma unroll
                for (int e = 0; e < 8; ++e)
                    out.s[e] = f2bf(bf2f(in.s[e]) * sc * (w[k0 + k8 + e] + wadd));
            }
            *(uint4*)(a + r * PITCH + k8) = out.v;
        }
        for (int t = tid; t < NT * (GROUP / 8); t += Threads) {
            const int col = c0 + t / (GROUP / 8), k8 = (t % (GROUP / 8)) * 8;
            union { uint4 v; unsigned short s[8]; } out;
            out.v = make_uint4(0, 0, 0, 0);
            if (col < LR) {
                const uint2 codes = *(const uint2*)(dw + col * DLD + k0 + k8);
                float decoded[8];
                e4m3x8(codes.x, codes.y, decoded);
#pragma unroll
                for (int e = 0; e < 8; ++e) out.s[e] = f2bf(decoded[e]);
            } else if (col < NC) {
                out.v = *(const uint4*)(iw + (col - LR) * HN + k0 + k8);
            }
            *(uint4*)(b + (t / (GROUP / 8)) * PITCH + k8) = out.v;
        }
        __syncthreads();
        if (!FusedNorm || wave < NT / 16) {
            v8f raw = (v8f){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            raw = gemm<GROUP / 16>(a, PITCH, 0, b, PITCH, wave, lane, raw);
            const int col = c0 + wave * 16 + (lane & 15);
            const float sc = col < LR ? ((const float*)(dw + col * DLD + HN))[g] : 1.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) total[e] = fmaf(raw[e], sc, total[e]);
        }
    }
    const int col = c0 + wave * 16 + (lane & 15), r0 = 8 * (lane >> 4);
    if ((!FusedNorm || wave < NT / 16) && col < NC) {
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (r0 + e < rows) part[(z * rows + r0 + e) * NC + col] = total[e];
    }
}

// Same ascending-z, eight-load-then-eight-add order as hc_down_reduce_kernel.
__device__ __forceinline__ float reduce(const float* part, int rows, int r, int col) {
    float v[SPLITS];
#pragma unroll
    for (int z = 0; z < SPLITS; ++z) v[z] = part[(z * rows + r) * NC + col];
    float s = 0.0f;
#pragma unroll
    for (int z = 0; z < SPLITS; ++z) s += v[z];
    return s;
}

// Adapted from hc_up_mix_kernel<RG,true,I8> (1273..1383). The only
// arithmetic replacement is its dbuf load: ordered reduction + original
// /hc, silu, f2bf, bf2f. Up GEMV, sigmoid, mixing and quantizers stay upstream's.
// No global dbuf/ubuf and no extra quantization launch. One CTA per row group
// also publishes inject; no kernel consumes inject until this op completes.
template <int RG, bool I8>
__global__ __launch_bounds__(kThreads)
void up_mix(const unsigned char* __restrict__ up, const float* __restrict__ part,
            const unsigned short* __restrict__ h, const float* __restrict__ w,
            const float* __restrict__ scalein, unsigned short* __restrict__ x,
            unsigned short* __restrict__ inject, unsigned char* __restrict__ q,
            float* __restrict__ qs, unsigned char* __restrict__ rq,
            float* __restrict__ rqs, unsigned int* __restrict__ ctr, int rows, float wadd) {
    extern __shared__ float sd[];
    __shared__ unsigned short s_u[RG * kUpRows];
    const int r0 = blockIdx.y * RG;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
    constexpr int ipb = kUpRows / HC;
    const int i0 = blockIdx.x * ipb;
    for (int k = tid; k < RG * LR; k += kThreads) {
        const int r = k / LR, col = k % LR;
        sd[k] = r0 + r < rows
                    ? bf2f(f2bf(siluf_(reduce(part, rows, r0 + r, col) * (1.0f / HC))))
                    : 0.0f;
    }
    if (blockIdx.x == 0) {
        for (int k = tid; k < RG * HC; k += kThreads) {
            const int r = r0 + k / HC, c = k % HC;
            if (r < rows)
                inject[r * HC + c] = f2bf(2.0f * sigmoidf_(reduce(part, rows, r, LR + c)
                                                         * (1.0f / HC)));
        }
    }
    __syncthreads();
    const int t = tid >> 2, q4 = tid & 3;
    const long j = (long)(t / ipb) * N + i0 + t % ipb;
    float acc[RG];
    hc8_dot_up<RG>(sd, LR, up + j * ULD, q4, acc);
#pragma unroll
    for (int r = 0; r < RG; ++r)
        if (q4 == 0) s_u[r * kUpRows + t] = f2bf(sigmoidf_(acc[r]));
    __syncthreads();
    for (int idx = tid; idx < RG * ipb; idx += kThreads) {
        const int rl = idx / ipb, ii = idx % ipb, r = r0 + rl;
        if (r >= rows) continue;
        const int i = i0 + ii;
        float sum = 0.0f;
        for (int c = 0; c < HC; ++c) {
            const int hj = c * N + i;
            const float nv = nbf(bf2f(h[r * HN + hj]) * scalein[r * kHcMax + c]
                                 * (w[hj] + wadd));
            sum += bf2f(s_u[rl * kUpRows + c * ipb + ii]) * nv;
        }
        sum *= 1.0f / HC;
        x[r * N + i] = f2bf(sum);
    }
    if (!q || !ctr) return;
    constexpr int nb = kUpTB / ipb;
    __threadfence();
    __syncthreads();
    __shared__ unsigned int arrived;
    const unsigned slot = blockIdx.y * (N / kUpTB) + i0 / kUpTB;
    if (tid == 0)
        arrived = __hip_atomic_fetch_add(&ctr[slot], 1u, __ATOMIC_ACQ_REL,
                                         __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (arrived + 1u != nb) return;
    if (tid == 0)
        __hip_atomic_store(&ctr[slot], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();
    if (wave >= RG) return;
    const int r = r0 + wave;
    if (r >= rows) return;
    const int base = (i0 / kUpTB) * kUpTB;
    const unsigned short* xr = x + r * N + base;
    float m = 0.0f;
    for (int t0 = lane; t0 < kUpTB; t0 += kWave) m = fmaxf(m, fabsf(bf2f(xr[t0])));
    m = r4d_wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * (N / kUpTB) + base / kUpTB] = sc;
    for (int d = lane; d < kUpTB / 4; d += kWave) {
        const int c4 = d * 4;
        const uint32_t pk = q8_pack4<I8>(bf2f(xr[c4]), bf2f(xr[c4 + 1]),
                                        bf2f(xr[c4 + 2]), bf2f(xr[c4 + 3]), inv);
        *(unsigned*)(q + r * N + base + c4) = pk;
    }
    if (rq) {
        float v[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) v[k] = bf2f(xr[lane * 4 + k]);
        hc_rot_group(v, lane, rq + r * N + base, rqs + r * (N / kUpTB) + base / kUpTB);
    }
}

// ---- --variant rowpair: hc_read_a_kernel<true> (r4d_hc_bf16.hip:879..1070), two rows a CTA.
// rowpair_a<2> serves M=2..4. A ragged last pair (M=3) reads row r0 again for its ghost row and
// publishes nothing for it; there is no third launch. M=1 is NOT row-shared: it launches the
// unchanged upstream hc_read_a_kernel<true> with the frozen launch arguments (unchanged control).
// Per row the arithmetic is upstream's: the kPreSs=8 norm loop, wave_sum, the ascending s_part
// sum, __frsqrt_rn, the non-swizzled N narrowing, hc8_dot_down<kDotU>'s lane/step/FMA order,
// wave_dot_lds<kDotU>'s staged lane order and the f2bf epilogues; prefetch depths equal upstream's.
// Shared across the rows: only E4M3 code, FP32 group-scale, W_inj bf16 and norm-weight `w` loads.
// LDS: 2*HN bf16 dynamic + static; register/occupancy cost is the compiler's, printed at runtime.
// All Buffers are hipMalloc + GUARD, so upstream's 16-byte-aligned W_inj path is the one taken.
constexpr int kRpU = kDotU, kRpPre = 8;  // upstream kDotU and kPreSs/kPreScale
static_assert(HN % (kHc8Step * kRpU) == 0 && HN % (kWave * 8 * kRpU) == 0,
              "dot tails must be dead at this shape");

template <int NR>
__device__ __forceinline__ void rp_dot_down(const unsigned short* __restrict__ sN,
                                            const unsigned char* __restrict__ wr, int lane,
                                            float* acc) {
    const float* __restrict__ sr = (const float*)(wr + HN);
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    for (long j = 0; j < HN; j += kHc8Step * kRpU) {
        uint4 c[kRpU];
        float s[kRpU];
#pragma unroll
        for (int u = 0; u < kRpU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
            c[u] = *(const uint4*)(wr + e);
            s[u] = sr[(size_t)e >> 7];
        }
#pragma unroll
        for (int u = 0; u < kRpU; ++u) {
            const long e = j + (long)u * kHc8Step + lane * 16;
#pragma unroll
            for (int r = 0; r < NR; ++r)
                acc[r] = fmaf(hc8_dot16(sN + (long)r * HN + e, c[u]), s[u], acc[r]);
        }
    }
}

template <int NR>
__device__ __forceinline__ void rp_dot_inj(const unsigned short* __restrict__ sN,
                                           const unsigned short* __restrict__ wr, int lane,
                                           unsigned short* stage, float* acc) {
    constexpr long kPer = kWave * 8;
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
    for (long j = 0; j < HN; j += kPer * kRpU) {
        uint4 v[kRpU];
#pragma unroll
        for (int u = 0; u < kRpU; ++u) v[u] = *(const uint4*)(wr + j + (long)u * kPer + lane * 8);
#pragma unroll
        for (int u = 0; u < kRpU; ++u) {
            *(uint4*)(stage + lane * 8) = v[u];
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                const int jj = lane + t * kWave;
                const float wv = bf2f(stage[jj]);
#pragma unroll
                for (int r = 0; r < NR; ++r)
                    acc[r] += bf2f(sN[(long)r * HN + j + (long)u * kPer + jj]) * wv;
            }
        }
    }
}

// Grid: (ceil(LR / ta) + 1 publishing block, ceil(M / 2)); dynamic LDS 2 * HN * 2 bytes.
// Only rowpair_a<2> is instantiated; NR is a template parameter so the row loops stay unrolled.
template <int NR>
__global__ __launch_bounds__(kThreads)
void rowpair_a(const unsigned short* __restrict__ h, const float* __restrict__ w,
               const unsigned char* __restrict__ dw, const unsigned short* __restrict__ iw,
               unsigned short* __restrict__ dout, unsigned short* __restrict__ inj,
               float* __restrict__ scaleout, int rows, long ta, float eps, float wadd) {
    static_assert(NR == 2, "M=1 runs upstream hc_read_a_kernel<true>, not a one-row instance");
    extern __shared__ __attribute__((aligned(16))) unsigned short sN[];  // [NR][HN]
    __shared__ float s_part[NR][kWaves][kHcMax];
    __shared__ float s_scale[NR][kHcMax];
    __shared__ unsigned short s_stage[kWaves][kWave * 8];
    const int r0 = NR * (int)blockIdx.y;
    const bool pub = blockIdx.x == gridDim.x - 1;
    const int tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    bool live[NR];
    const unsigned short* hp[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        live[r] = r0 + r < rows;  // r = 0 is always live: gridDim.y = ceil(rows / NR)
        hp[r] = h + (long)(live[r] ? r0 + r : r0) * HN;
    }
    for (int c = 0; c < HC; ++c) {
        const long cb = (long)c * N;
        float acc[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) acc[r] = 0.0f;
        long i = tid;
        for (; i + (long)(kRpPre - 1) * kThreads < N; i += (long)kRpPre * kThreads) {
            unsigned short hv[NR][kRpPre];
#pragma unroll
            for (int r = 0; r < NR; ++r)
#pragma unroll
                for (int p = 0; p < kRpPre; ++p) hv[r][p] = hp[r][cb + i + (long)p * kThreads];
#pragma unroll
            for (int r = 0; r < NR; ++r)
#pragma unroll
                for (int p = 0; p < kRpPre; ++p) { const float v = bf2f(hv[r][p]); acc[r] += v * v; }
        }
        for (; i < N; i += kThreads)
#pragma unroll
            for (int r = 0; r < NR; ++r) { const float v = bf2f(hp[r][cb + i]); acc[r] += v * v; }
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            acc[r] = wave_sum(acc[r]);
            if (lane == 0) s_part[r][wave][c] = acc[r];
        }
    }
    __syncthreads();
    if (tid < NR * HC) {
        const int r = tid / HC, c = tid % HC;
        float ss = 0.0f;
        for (int k = 0; k < kWaves; ++k) ss += s_part[r][k][c];
        s_scale[r][c] = __frsqrt_rn(ss / (float)N + eps);
        if (pub && r0 + r < rows) scaleout[(long)(r0 + r) * kHcMax + c] = s_scale[r][c];
    }
    __syncthreads();
    for (int c = 0; c < HC; ++c) {
        const long cb = (long)c * N;
        float sc[NR];
#pragma unroll
        for (int r = 0; r < NR; ++r) sc[r] = s_scale[r][c];
        long i = tid;
        for (; i + (long)(kRpPre - 1) * kThreads < N; i += (long)kRpPre * kThreads) {
            float wv[kRpPre];
            unsigned short hv[NR][kRpPre];
#pragma unroll
            for (int p = 0; p < kRpPre; ++p) {
                wv[p] = w[cb + i + (long)p * kThreads];
#pragma unroll
                for (int r = 0; r < NR; ++r) hv[r][p] = hp[r][cb + i + (long)p * kThreads];
            }
#pragma unroll
            for (int r = 0; r < NR; ++r)
#pragma unroll
                for (int p = 0; p < kRpPre; ++p)
                    sN[(long)r * HN + cb + i + (long)p * kThreads] =
                        f2bf(bf2f(hv[r][p]) * sc[r] * (wv[p] + wadd));
        }
        for (; i < N; i += kThreads) {
            const float wv = w[cb + i];
#pragma unroll
            for (int r = 0; r < NR; ++r)
                sN[(long)r * HN + cb + i] = f2bf(bf2f(hp[r][cb + i]) * sc[r] * (wv + wadd));
        }
    }
    __syncthreads();
    const float inv_hc = 1.0f / (float)HC;
    const long k0 = (long)blockIdx.x * ta;
    for (long k = k0 + wave; k < k0 + ta && k < LR; k += kWaves) {
        float acc[NR];
        rp_dot_down<NR>(sN, dw + k * DLD, lane, acc);
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float a = wave_sum(acc[r]);
            if (lane == 0 && live[r]) dout[(long)(r0 + r) * LR + k] = f2bf(siluf_(a * inv_hc));
        }
    }
    if (!pub) return;
    for (int c = wave; c < HC; c += kWaves) {
        float acc[NR];
        rp_dot_inj<NR>(sN, iw + (long)c * HN, lane, s_stage[wave], acc);
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float a = wave_sum(acc[r]);
            if (lane == 0 && live[r]) inj[(long)(r0 + r) * HC + c] = f2bf(2.0f * sigmoidf_(a * inv_hc));
        }
    }
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
void hip_check(hipError_t error, const char* call) {
    if (error != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(error));
}
#define HIP(call) ::probe::hip_check((call), #call)
constexpr size_t GUARD = 256;
using Bytes = std::vector<unsigned char>;
struct Buffer {
    unsigned char* raw = nullptr;
    size_t size;
    explicit Buffer(size_t bytes) : size(bytes) {
        require(size > 0, "zero-sized buffer");
        HIP(hipMalloc((void**)&raw, size + 2 * GUARD));
        reset();
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() {
        const auto rc = hipFree(raw);
        if (rc != hipSuccess) std::fprintf(stderr, "cleanup hipFree: %s\n", hipGetErrorString(rc));
    }
    void* data() const { return raw + GUARD; }
    template <class T> T* as() const { return (T*)data(); }
    void reset() const {
        HIP(hipMemset(raw, 0xa5, size + 2 * GUARD));
        HIP(hipMemset(data(), 0xcd, size));
    }
    void put(const void* src) const { HIP(hipMemcpy(data(), src, size, hipMemcpyHostToDevice)); }
    static bool guards(const Bytes& b, size_t size) {
        return b.size() == size + 2 * GUARD
            && std::all_of(b.begin(), b.begin() + GUARD, [](unsigned char x) { return x == 0xa5; })
            && std::all_of(b.end() - GUARD, b.end(), [](unsigned char x) { return x == 0xa5; });
    }
    Bytes get() const {
        Bytes b(size + 2 * GUARD);
        HIP(hipMemcpy(b.data(), raw, b.size(), hipMemcpyDeviceToHost));
        require(guards(b, size), "GPU buffer guard changed (bytes=" + std::to_string(size) + ")");
        return Bytes(b.begin() + GUARD, b.end() - GUARD);
    }
};
float host_bf(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}
uint16_t host_to_bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
template <class T> T load(const Bytes& b, size_t i) {
    T v;
    std::memcpy(&v, b.data() + i * sizeof(T), sizeof(v));
    return v;
}
float fp8(unsigned char b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
    const float v = e ? std::ldexp(1.0f + m * 0.125f, e - 7) : m * (1.0f / 512.0f);
    return (b & 128) ? -v : v;
}
unsigned char encode_fp8(float v) {
    const unsigned sign = std::signbit(v) ? 128u : 0u;
    const float a = std::min(std::fabs(v), 448.0f);
    unsigned best = 0;
    float distance = std::numeric_limits<float>::infinity();
    for (unsigned c = 0; c < 127; ++c) {
        const float d = std::fabs(a - fp8((unsigned char)c));
        if (d < distance || (d == distance && !(c & 1))) { distance = d; best = c; }
    }
    return (unsigned char)(best | sign);
}
RadTensor tensor(void* data, uint32_t dtype, std::initializer_list<int64_t> shape) {
    RadTensor t{};
    t.data = data; t.dtype = dtype; t.rank = (uint32_t)shape.size();
    std::copy(shape.begin(), shape.end(), t.shape);
    rad_tensor_pack(&t);
    return t;
}
RadParam pi(const char* key, long long value) {
    RadParam p{}; p.key = key; p.kind = RAD_P_INT; p.ival = value; return p;
}
RadParam pf(const char* key, double value) {
    RadParam p{}; p.key = key; p.kind = RAD_P_F64; p.dval = value; return p;
}
RadParam ps(const char* key, const char* value) {
    RadParam p{}; p.key = key; p.kind = RAD_P_STR; p.sval = value; return p;
}
struct Inputs {
    Buffer h, w, dw, up, iw;
    std::array<Bytes, 5> original;
    Inputs(int rows, uint64_t seed)
        : h(rows * HN * 2), w(HN * 4), dw(LR * DLD), up(HN * ULD), iw(HC * HN * 2) {
        uint64_t rs = 0x9e3779b97f4a7c15ull ^ seed ^ (uint64_t)rows;
        auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; };
        std::vector<uint16_t> hh(rows * HN), ii(HC * HN);
        std::vector<float> ww(HN);
        // Same operand distribution as r4d_selftest.cpp:5099..5121, not benign power-of-two scales.
        for (auto& v : hh) v = host_to_bf((float)((int)(rnd() % 2001) - 1000) / 400.0f);
        for (auto& v : ww) v = (float)((int)(rnd() % 201) - 100) / 1000.0f;
        for (auto& v : ii) v = host_to_bf((float)((int)(rnd() % 201) - 100) / 50000.0f);
        auto plane = [&](int nr, int k, int group, int ld) {
            Bytes b((size_t)nr * ld, 0);
            for (int r = 0; r < nr; ++r) {
                unsigned char* row = b.data() + (size_t)r * ld;
                for (int j = 0; j < k; ++j)
                    row[j] = (unsigned char)((8 + rnd() % 111) | ((rnd() & 1) ? 128u : 0u));
                for (int g = 0; g < k / group; ++g) {
                    const float sc = 0.002f + 0.004f * (float)(rnd() % 1000) / 1000.0f;
                    std::memcpy(row + k + g * 4, &sc, 4);
                }
            }
            return b;
        };
        Bytes dd = plane(LR, HN, GROUP, DLD), uu = plane(HN, LR, LR / 4, ULD);
        h.put(hh.data()); w.put(ww.data()); dw.put(dd.data()); up.put(uu.data()); iw.put(ii.data());
        original = {h.get(), w.get(), dw.get(), up.get(), iw.get()};
    }
    void check() const {
        require(original == std::array<Bytes, 5>{h.get(), w.get(), dw.get(), up.get(), iw.get()},
                "an input, weight, scale or row padding changed");
    }
};
using Snapshot = std::array<Bytes, 6>;
constexpr const char* names[] = {"x", "inject", "q", "qs", "rq", "rqs"};
struct Outputs {
    std::array<std::unique_ptr<Buffer>, 6> b;
    explicit Outputs(int rows) {
        const size_t sizes[] = {(size_t)rows * N * 2, (size_t)rows * HC * 2,
            (size_t)rows * N, (size_t)rows * (N / GROUP) * 4,
            (size_t)rows * N, (size_t)rows * (N / GROUP) * 4};
        for (int i = 0; i < 6; ++i) b[i] = std::make_unique<Buffer>(sizes[i]);
    }
    Snapshot get() const {
        Snapshot s;
        for (int i = 0; i < 6; ++i) s[i] = b[i]->get();
        return s;
    }
};
// part: WMMA split-K f32 partials [z][M][324]; d: row-pair bf16 d [M][320], frozen dbuf layout.
// Each variant must fill its own and leave the other's 0xcd fill: evidence of the path reached.
struct Work {
    const bool rowpair;
    Buffer scale, part, d, ctr;
    Work(int rows, bool rp) : rowpair(rp), scale(rows * kHcMax * 4), part(SPLITS * rows * NC * 4),
        d(rows * LR * 2), ctr(((rows + hc_up_rg(rows, LR) - 1) / hc_up_rg(rows, LR)) * (N / GROUP) * 4) {
        HIP(hipMemset(ctr.data(), 0, ctr.size));
    }
    void check(int rows) const {
        const Bytes s = scale.get(), p = part.get(), dd = d.get(), c = ctr.get();
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < HC; ++j)
                require(std::isfinite(load<float>(s, r * kHcMax + j)), "unwritten norm scale");
            for (int j = HC * 4; j < kHcMax * 4; ++j)
                require(s[r * kHcMax * 4 + j] == 0xcd, "norm scale tail changed");
        }
        const auto untouched = [](const Bytes& b) {
            return std::all_of(b.begin(), b.end(), [](unsigned char v) { return v == 0xcd; });
        };
        if (rowpair) {
            require(untouched(p), "row-pair variant wrote WMMA partials");
            for (size_t i = 0; i < dd.size() / 2; ++i)
                require(std::isfinite(host_bf(load<uint16_t>(dd, i))) && load<uint16_t>(dd, i) != 0xcdcdu,
                        "unwritten/nonfinite row-pair d");
        } else {
            for (size_t i = 0; i < p.size() / 4; ++i)
                require(std::isfinite(load<float>(p, i)) && load<uint32_t>(p, i) != 0xcdcdcdcdu,
                        "unwritten/nonfinite down partial");
            require(untouched(dd), "WMMA variant wrote row-pair d");
        }
        require(std::all_of(c.begin(), c.end(), [](unsigned char v) { return v == 0; }),
                "quantization arrival counters did not reset");
    }
};

template<bool I8>
__device__ __forceinline__
void quant_group(const unsigned short* x, unsigned char* q, float* qs,
                 unsigned char* rq, float* rqs, int r, int group, int lane) {
    const int base = group * GROUP;
    const unsigned short* xr = x + r * N + base;
    float m = 0.0f;
    for (int t = lane; t < GROUP; t += 32) m = fmaxf(m, fabsf(bf2f(xr[t])));
    m = r4d_wave_max(m);
    const float sc = m > 0.0f ? m * (1.0f / q8_top<I8>()) : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) qs[r * (N / GROUP) + group] = sc;
    const int c4 = lane * 4;
    *(unsigned*)(q + r * N + base + c4) =
        q8_pack4<I8>(bf2f(xr[c4]), bf2f(xr[c4 + 1]), bf2f(xr[c4 + 2]), bf2f(xr[c4 + 3]), inv);
    if (rq) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = bf2f(xr[c4 + j]);
        hc_rot_group(v, lane, rq + r * N + base, rqs + r * (N / GROUP) + group);
    }
}

template<bool I8>
__global__ __launch_bounds__(128)
void quant_x(const unsigned short* x, unsigned char* q, float* qs,
             unsigned char* rq, float* rqs, int rows) {
    const int wave = threadIdx.x / 32;
    if (wave < rows) quant_group<I8>(x, q, qs, rq, rqs, wave, blockIdx.x, threadIdx.x & 31);
}

// Unscaled E4M3 codes are exact BF16 operands. Keep each 80-K scale group
// separate, then retain the original (group0+group1)+(group2+group3) tree.
// The only reordered arithmetic is the WMMA dot within a scale group.
template<bool I8, int Channels = 32, bool DirectB = false>
__global__ __launch_bounds__(Channels == GROUP ? 256 : 128)
void up_wmma(const unsigned char* __restrict__ up, const unsigned short* __restrict__ d,
             const unsigned short* __restrict__ h, const float* __restrict__ w,
             const float* __restrict__ scale, unsigned short* __restrict__ x,
             unsigned char* __restrict__ q, float* __restrict__ qs,
             unsigned char* __restrict__ rq, float* __restrict__ rqs,
             unsigned int* __restrict__ ctr, int rows) {
    constexpr int Cols = HC * Channels, K = LR / 4;
    constexpr int Threads = Channels == GROUP ? 256 : 128;
    constexpr int TileK = Channels == GROUP && !DirectB ? 16 : K, Pitch = TileK + 8;
    constexpr int Waves = Threads / 32, Tiles = Cols / (16 * Waves);
    static_assert(K == 80 && (Channels == 32 || Channels == GROUP));
    __shared__ __attribute__((aligned(16))) unsigned short a[16 * Pitch], b[DirectB ? 1 : Cols * Pitch];
    __shared__ unsigned short u[4 * Cols];
    __shared__ unsigned int arrived;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid / 32;
    const int i0 = blockIdx.x * Channels;
    v8f total[Tiles] = {}, pair[Tiles] = {};
    for (int g = 0; g < 4; ++g) {
        v8f raw[Tiles] = {};
        for (int kb = 0; kb < K; kb += TileK) {
            __syncthreads();
            for (int t = tid; t < 16 * (TileK / 8); t += Threads) {
                const int r = t / (TileK / 8), k8 = (t % (TileK / 8)) * 8;
                const uint4 v = r < rows ? *(const uint4*)(d + r * LR + g * K + kb + k8)
                                        : make_uint4(0, 0, 0, 0);
                *(uint4*)(a + r * Pitch + k8) = v;
            }
            if constexpr (!DirectB) {
                for (int t = tid; t < Cols * (TileK / 8); t += Threads) {
                    const int col = t / (TileK / 8), k8 = (t % (TileK / 8)) * 8;
                    const int j = (col / Channels) * N + i0 + col % Channels;
                    const uint2 codes = *(const uint2*)(up + j * ULD + g * K + kb + k8);
                    float decoded[8];
                    e4m3x8(codes.x, codes.y, decoded);
                    union { uint4 v; unsigned short s[8]; } out;
#pragma unroll
                    for (int e = 0; e < 8; ++e) out.s[e] = f2bf(decoded[e]);
                    *(uint4*)(b + col * Pitch + k8) = out.v;
                }
            }
            __syncthreads();
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
                if constexpr (DirectB) {
                    const int col = (wave + tile * Waves) * 16 + (lane & 15);
                    const int j = (col / Channels) * N + i0 + col % Channels;
                    const unsigned char* row = up + j * ULD + g * K + 4 * (lane >> 4);
#pragma unroll
                    for (int k = 0; k < K; k += 16) {
                        // WMMA's fragment is two runs of four, eight codes apart.
                        float decoded[8];
                        e4m3x8(*(const unsigned*)(row + k), *(const unsigned*)(row + k + 8), decoded);
                        union { unsigned short h[8]; v8bf v; } bf;
#pragma unroll
                        for (int e = 0; e < 8; ++e) bf.h[e] = f2bf(decoded[e]);
                        raw[tile] = mma(frag(a, Pitch, 0, k, lane), bf.v, raw[tile]);
                    }
                } else {
                    raw[tile] = gemm<TileK / 16>(a, Pitch, 0, b, Pitch, wave + tile * Waves, lane, raw[tile]);
                }
            }
        }
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
            const int col = (wave + tile * Waves) * 16 + (lane & 15);
            const int j = (col / Channels) * N + i0 + col % Channels;
            const float sc = ((const float*)(up + j * ULD + LR))[g];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = raw[tile][e] * sc;
                if ((g & 1) == 0) pair[tile][e] = v;
                else {
                    pair[tile][e] += v;
                    if (g == 1) total[tile][e] = pair[tile][e];
                    else total[tile][e] += pair[tile][e];
                }
            }
        }
    }
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const int col = (wave + tile * Waves) * 16 + (lane & 15), r0 = 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (r0 + e < rows) u[(r0 + e) * Cols + col] = f2bf(sigmoidf_(total[tile][e]));
    }
    __syncthreads();
    for (int idx = tid; idx < rows * Channels; idx += Threads) {
        const int r = idx / Channels, ii = idx % Channels, i = i0 + ii;
        float acc = 0.0f;
        for (int c = 0; c < HC; ++c) {
            const int j = c * N + i;
            const float nv = nbf(bf2f(h[r * HN + j]) * scale[r * kHcMax + c] * (w[j] + 1.0f));
            acc += bf2f(u[r * Cols + c * Channels + ii]) * nv;
        }
        x[r * N + i] = f2bf(acc * (1.0f / HC));
    }
    if (!q) return;
    const int group = i0 / GROUP;
    if constexpr (Channels == GROUP) {
        __syncthreads();
    } else {
        __threadfence();
        __syncthreads();
        if (tid == 0)
            arrived = __hip_atomic_fetch_add(ctr + group, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        __syncthreads();
        if (arrived + 1u != GROUP / Channels) return;
        if (tid == 0)
            __hip_atomic_store(ctr + group, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();
    }
    if (wave >= rows) return;
    quant_group<I8>(x, q, qs, rq, rqs, wave, group, lane);
}

void launch(Inputs& in, Outputs& out, Work& work, int rows, int mode, hipStream_t stream,
            bool fused_norm) {
    // Two launches with fused norm, otherwise the original three-launch control.
    if (!fused_norm) {
        hipLaunchKernelGGL(hc_norm_scale_kernel, dim3(HC, rows), dim3(kThreads), 0, stream,
            in.h.as<unsigned short>(), work.scale.as<float>(), (long)rows, (long)N, (long)HN, 1e-6f);
        HIP(hipGetLastError());
    }
#define PROBE_DOWN(FUSED_) \
    hipLaunchKernelGGL((down<FUSED_>), dim3((NC + NT - 1) / NT, 1, SPLITS), \
        dim3(FUSED_ ? kThreads : DT), 0, stream, in.h.as<unsigned short>(), in.w.as<float>(), \
        in.dw.as<unsigned char>(), in.iw.as<unsigned short>(), work.scale.as<float>(), \
        work.part.as<float>(), rows, 1.0f)
    if (fused_norm) { PROBE_DOWN(true); } else { PROBE_DOWN(false); }
#undef PROBE_DOWN
    HIP(hipGetLastError());
    const int rg = (int)hc_up_rg(rows, LR);
    const dim3 grid(N / (kUpRows / HC), (rows + rg - 1) / rg);
#define PROBE_UP(RG_, I8_) \
    hipLaunchKernelGGL((up_mix<RG_, I8_>), grid, dim3(kThreads), RG_ * LR * sizeof(float), stream, \
        in.up.as<unsigned char>(), work.part.as<float>(), in.h.as<unsigned short>(), \
        in.w.as<float>(), work.scale.as<float>(), out.b[0]->as<unsigned short>(), \
        out.b[1]->as<unsigned short>(), mode ? out.b[2]->as<unsigned char>() : nullptr, \
        mode ? out.b[3]->as<float>() : nullptr, mode ? out.b[4]->as<unsigned char>() : nullptr, \
        mode ? out.b[5]->as<float>() : nullptr, work.ctr.as<unsigned int>(), rows, 1.0f)
#define PROBE_RG(RG_) do { if (mode == 2) { PROBE_UP(RG_, true); } else { PROBE_UP(RG_, false); } } while (0)
    switch (rg) {
        case 8: PROBE_RG(8); break;
        case 4: PROBE_RG(4); break;
        case 2: PROBE_RG(2); break;
        default: PROBE_RG(1); break;
    }
#undef PROBE_RG
#undef PROBE_UP
    HIP(hipGetLastError());
}

// r4d_hc_bf16.hip:3725's ta ladder over the grid's y extent: rows for the frozen hc_read_a,
// row groups for rowpair_a.
long hc_read_ta(long grid_rows) {
    long ta = kDownTAMax;
    while (ta > 4 && ((LR + ta - 1) / ta) * grid_rows < 2 * want_wg()) ta >>= 1;
    return ta;
}
// M=1 is the unchanged control: upstream hc_read_a_kernel<true>. M=2..4: rowpair_a<2>.
bool rowpair_shared(int rows) { return rows >= 2; }
int rowpair_groups(int rows) { return rowpair_shared(rows) ? (rows + 1) / 2 : rows; }

// Two launches, as the frozen M<8 path: kernel A, then upstream's own hc_up_mix_launch with
// the frozen call's arguments (r4d_hc_bf16.hip:3753); its counters only when q is present.
void launch_rowpair(Inputs& in, Outputs& out, Work& work, int rows, int mode, hipStream_t stream) {
    const int groups = rowpair_groups(rows);
    const long ta = hc_read_ta(groups);
    const dim3 grid((LR + ta - 1) / ta + 1, groups);
    if (rowpair_shared(rows)) {
        hipLaunchKernelGGL(rowpair_a<2>, grid, dim3(kThreads), 2 * HN * sizeof(unsigned short),
            stream, in.h.as<unsigned short>(), in.w.as<float>(), in.dw.as<unsigned char>(),
            in.iw.as<unsigned short>(), work.d.as<unsigned short>(), out.b[1]->as<unsigned short>(),
            work.scale.as<float>(), rows, ta, 1e-6f, 1.0f);
    } else {
        // r4d_hc_bf16.hip:3727..3733's HC_READ_A(true), argument for argument.
        const int rc = R4D_EMIT(r4d_stream_sink((RadStream)stream), hc_read_a_kernel<true>, grid,
            dim3(kThreads), (unsigned)(HN * sizeof(unsigned short)), in.h.as<unsigned short>(),
            in.w.as<float>(), (const void*)in.dw.data(), (long)DLD, in.iw.as<unsigned short>(),
            work.d.as<unsigned short>(), out.b[1]->as<unsigned short>(), work.scale.as<float>(),
            (long)rows, (long)N, (int)HC, (long)LR, ta, (long)HN, (long)HC, 1e-6f, 1.0f);
        require(rc == RAD_OK, "upstream hc_read_a_kernel emit refused: rc=" + std::to_string(rc));
    }
    HIP(hipGetLastError());
    RadTensor up{};
    up.data = in.up.data();  // hc_up_mix_launch reads only up->data
    const int rc = hc_up_mix_launch(&up, ULD, true, mode == 2, work.d.as<unsigned short>(),
        in.h.as<unsigned short>(), in.w.as<float>(), work.scale.as<float>(),
        out.b[0]->as<unsigned short>(), mode ? out.b[2]->as<unsigned char>() : nullptr,
        mode ? out.b[3]->as<float>() : nullptr, mode ? out.b[4]->as<unsigned char>() : nullptr,
        mode ? out.b[5]->as<float>() : nullptr, mode ? work.ctr.as<unsigned int>() : nullptr,
        rows, N, HC, LR, HN, N, N, N / GROUP, N, N / GROUP, 1.0f,
        r4d_stream_sink((RadStream)stream));
    require(rc == RAD_OK, "upstream hc_up_mix_launch refused: rc=" + std::to_string(rc));
    HIP(hipGetLastError());
}

void launch_up_wmma(Inputs& in, Outputs& out, Work& work, int rows, int mode, hipStream_t stream,
                    bool whole_group, bool direct_b, bool split_quant) {
    require(rows >= 1 && rows <= 4, "up-wmma supports M=1..4");
    const long ta = hc_read_ta(rows);
    const int rc = R4D_EMIT(r4d_stream_sink((RadStream)stream), hc_read_a_kernel<true>,
        dim3((LR + ta - 1) / ta + 1, rows), dim3(kThreads), (unsigned)(HN * sizeof(unsigned short)),
        in.h.as<unsigned short>(), in.w.as<float>(), (const void*)in.dw.data(), (long)DLD,
        in.iw.as<unsigned short>(), work.d.as<unsigned short>(), out.b[1]->as<unsigned short>(),
        work.scale.as<float>(), (long)rows, (long)N, (int)HC, (long)LR, ta,
        (long)HN, (long)HC, 1e-6f, 1.0f);
    require(rc == RAD_OK, "up-wmma original down launch refused");
    HIP(hipGetLastError());
#define PROBE_UP_WMMA(I8_, C_, D_) \
    hipLaunchKernelGGL((up_wmma<I8_, C_, D_>), dim3(N / C_), dim3(C_ == GROUP ? 256 : 128), 0, stream, \
        in.up.as<unsigned char>(), work.d.as<unsigned short>(), in.h.as<unsigned short>(), \
        in.w.as<float>(), work.scale.as<float>(), out.b[0]->as<unsigned short>(), \
        mode && !split_quant ? out.b[2]->as<unsigned char>() : nullptr, \
        mode && !split_quant ? out.b[3]->as<float>() : nullptr, \
        mode && !split_quant ? out.b[4]->as<unsigned char>() : nullptr, \
        mode && !split_quant ? out.b[5]->as<float>() : nullptr, \
        whole_group || split_quant ? nullptr : work.ctr.as<unsigned int>(), rows)
    if (direct_b) {
        if (mode == 2) { PROBE_UP_WMMA(true, GROUP, true); } else { PROBE_UP_WMMA(false, GROUP, true); }
    } else if (whole_group) {
        if (mode == 2) { PROBE_UP_WMMA(true, GROUP, false); } else { PROBE_UP_WMMA(false, GROUP, false); }
    } else {
        if (mode == 2) { PROBE_UP_WMMA(true, 32, false); } else { PROBE_UP_WMMA(false, 32, false); }
    }
#undef PROBE_UP_WMMA
    HIP(hipGetLastError());
    if (split_quant && mode) {
#define PROBE_QUANT(I8_) \
        hipLaunchKernelGGL((quant_x<I8_>), dim3(N / GROUP), dim3(128), 0, stream, \
            out.b[0]->as<unsigned short>(), out.b[2]->as<unsigned char>(), out.b[3]->as<float>(), \
            out.b[4]->as<unsigned char>(), out.b[5]->as<float>(), rows)
        if (mode == 2) { PROBE_QUANT(true); } else { PROBE_QUANT(false); }
#undef PROBE_QUANT
        HIP(hipGetLastError());
    }
}

void kernel_attr(const char* name, const void* f) {
    hipFuncAttributes a{};
    HIP(hipFuncGetAttributes(&a, f));
    std::printf("kernel_attr name=%s numRegs=%d localSizeBytes=%zu sharedSizeBytes=%zu "
                "maxThreadsPerBlock=%d (hipFuncGetAttributes)\n",
                name, a.numRegs, a.localSizeBytes, a.sharedSizeBytes, a.maxThreadsPerBlock);
}

struct Oracle {
    void* handle = nullptr;
    const RadKernelInfo* row = nullptr;
    int index = -1;
    template <class T> T symbol(const char* name) {
        dlerror();
        void* address = dlsym(handle, name);
        const char* error = dlerror();
        require(!error && address, std::string("oracle symbol ") + name + ": " + (error ? error : "null"));
        T f;
        static_assert(sizeof(f) == sizeof(address));
        std::memcpy(&f, &address, sizeof(f));
        return f;
    }
    explicit Oracle(const char* library = ORACLE, const char* row_name = "hc_read_e4m3") {
        handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
        if (!handle) { const char* e = dlerror(); throw std::runtime_error(e ? e : "oracle dlopen"); }
        const auto abi = symbol<uint32_t (*)()>("rad_plugin_abi_version");
        require(abi() == RAD_ABI_VERSION, "frozen oracle ABI differs from headers");
        const auto count = symbol<int (*)()>("rad_kernel_count");
        const auto at = symbol<const RadKernelInfo* (*)(int)>("rad_kernel_at");
        for (int i = 0; i < count(); ++i) {
            const auto* r = at(i);
            if (r && r->name && std::strcmp(r->name, row_name) == 0) {
                require(!row, "duplicate requested HC row"); row = r; index = i;
            }
        }
        require(row && row->launch && row->scratch && row->op
                && std::strcmp(row->op, "hc_read") == 0 && row->domain == RAD_DOMAIN_DEVICE,
                "frozen HC device row/launch/scratch absent");
        require(!row->init && !row->fini, "unexpected HC instance lifecycle");
        Dl_info info{};
        require(dladdr((const void*)row->launch, &info) != 0 && info.dli_fname, "oracle launch origin unknown");
        char expected[PATH_MAX], actual[PATH_MAX];
        require(realpath(library, expected) && realpath(info.dli_fname, actual)
                && std::strcmp(expected, actual) == 0, "HC launch resolves outside its own plugin");
        std::printf("oracle=%s abi=%u actual_row_index=%d name=%s op=%s domain=%d launch_origin=%s\n",
                    library, abi(), index, row->name, row->op, row->domain, actual);
        std::printf("row_shape=%s\nrow_dtypes=%s\n", row->shape, row->dtypes);
        if (std::strcmp(row_name, "hc_read_e4m3") == 0) {
            const auto pin = symbol<void (*)(int)>("r4d_hc_read_pin_prefill");
            pin(0); // Frozen launcher's normal shape-dependent choice.
        }
    }
    ~Oracle() { if (handle && dlclose(handle)) std::fprintf(stderr, "cleanup dlclose: %s\n", dlerror()); }
};
struct OracleCall {
    std::vector<RadTensor> tensors;
    std::vector<RadParam> params;
    RadArgs args{};
    std::unique_ptr<Buffer> scratch;
    OracleCall(const Oracle& oracle, Inputs& in, Outputs& out, int rows, int mode) {
        tensors = {tensor(in.h.data(), RAD_BF16, {rows, HN}), tensor(in.w.data(), RAD_F32, {HN}),
            tensor(in.dw.data(), RAD_F8E4M3, {LR, HN}), tensor(in.up.data(), RAD_F8E4M3, {HN, LR}),
            tensor(in.iw.data(), RAD_BF16, {HC, HN}), tensor(out.b[0]->data(), RAD_BF16, {rows, N}),
            tensor(out.b[1]->data(), RAD_BF16, {rows, HC}),
            tensor(out.b[2]->data(), mode == 2 ? RAD_I8 : RAD_F8E4M3, {rows, N}),
            tensor(out.b[3]->data(), RAD_F32, {rows, N / GROUP}),
            tensor(out.b[4]->data(), RAD_F8E4M3, {rows, N}),
            tensor(out.b[5]->data(), RAD_F32, {rows, N / GROUP})};
        if (!mode) for (int i = 7; i < 11; ++i) tensors[i] = RadTensor{};
        params = {pi("M", rows), pi("n", N), pi("hc", HC), pi("lowrank", LR), pf("eps", 1e-6),
            ps("dtype", "bf16"), pf("wadd", 1.0), pi("inject", 1), pi("group", GROUP),
            pi("rotate", mode ? 1 : 0), ps("mix", "e4m3")};
        args.t = tensors.data(); args.n_t = (int)tensors.size();
        args.p = params.data(); args.n_p = (int)params.size(); args.world_size = 1;
        const int64_t need = oracle.row->scratch(&args);
        require(need > 0, "oracle scratch hook refused geometry");
        scratch = std::make_unique<Buffer>((size_t)need);
        args.scratch = scratch->data(); args.scratch_bytes = need;
    }
    void launch(const Oracle& oracle, hipStream_t stream) {
        const int rc = oracle.row->launch(&args, (RadStream)stream);
        if (rc != RAD_OK)
            throw std::runtime_error("frozen HC launch refused: rc=" + std::to_string(rc));
        HIP(hipGetLastError());
    }
};

void check_scratch(const Work& work, const OracleCall& oracle, int rows) {
    const Bytes own = work.scale.get(), frozen = oracle.scratch->get();
    const size_t offset = (size_t)rows * LR * sizeof(uint16_t);
    require(offset + own.size() <= frozen.size(), "frozen norm scratch layout is unsupported");
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < HC; ++c) {
            const size_t i = r * kHcMax + c;
            require(load<uint32_t>(own, i) == load<uint32_t>(frozen, offset / 4 + i),
                    "norm scales differ bitwise from frozen HC; timing prohibited");
        }
    // Frozen dbuf is the scratch's first M*LR bf16 (r4d_hc_bf16.hip:3585, band M = rows).
    if (work.rowpair) {
        const Bytes d = work.d.get();
        require(d.size() == offset && std::equal(d.begin(), d.end(), frozen.begin()),
                "row-pair d differs bytewise from frozen dbuf; timing prohibited");
    }
}

float value(const Bytes& b, int field, size_t i, int mode) {
    if (field < 2) return host_bf(load<uint16_t>(b, i));
    if (field == 3 || field == 5) return load<float>(b, i);
    return field == 2 && mode == 2 ? (float)load<int8_t>(b, i) : fp8(b[i]);
}
bool compare(const Snapshot& ref, const Snapshot& got, int mode, bool exact, bool print) {
    bool pass = true;
    for (int f = 0; f < 6; ++f) {
        if (!mode && f >= 2) {
            const bool untouched = ref[f] == got[f]
                && std::all_of(got[f].begin(), got[f].end(), [](unsigned char v) { return v == 0xcd; });
            pass = pass && untouched;
            if (print) std::printf("  %s absent_untouched=%d\n", names[f], untouched);
            continue;
        }
        if (ref[f].size() != got[f].size()) return false;
        const size_t width = f < 2 ? 2 : (f == 3 || f == 5 ? 4 : 1);
        if (exact) {
            size_t diff = 0, first = 0, nonfinite = 0;
            for (size_t i = 0; i < ref[f].size(); ++i)
                if (ref[f][i] != got[f][i]) { if (!diff) first = i; ++diff; }
            for (size_t i = 0; i < got[f].size() / width; ++i)
                nonfinite += !std::isfinite(value(got[f], f, i, mode));
            const bool good = diff == 0 && nonfinite == 0;
            pass = pass && good;
            if (print) std::printf("  %s byte_exact=%d bytes=%zu mismatched_bytes=%zu first_mismatch_byte=%ld "
                                   "nonfinite=%zu pass=%d\n", names[f], diff == 0, ref[f].size(), diff,
                                   diff ? (long)first : -1L, nonfinite, good);
            continue;
        }
        double num = 0, den = 0, mine = 0, max_abs = 0;
        bool finite = true;
        for (size_t i = 0; i < ref[f].size() / width; ++i) {
            const double a = value(got[f], f, i, mode), b = value(ref[f], f, i, mode);
            if (!std::isfinite(a) || !std::isfinite(b)) { finite = false; continue; }
            const double d = a - b;
            num += d * d; den += b * b; mine += a * a;
            max_abs = std::max(max_abs, std::fabs(d));
        }
        const double l2 = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
        const double norm = den > 0 ? std::fabs(std::sqrt(mine) - std::sqrt(den)) / std::sqrt(den)
                                    : (mine > 0 ? 1.0 : 0.0);
        const double error = std::max(l2, norm);
        // tools/rad_kbench.cpp:120..138: bf16 default 6e-3, hc_read q/rq 3e-2.
        // Compare every element, not the kbench sample subset. No CLI tolerance override.
        const double tolerance = f == 2 || f == 4 ? 3e-2 : 6e-3;
        const bool good = finite && error >= 0 && error <= tolerance;
        pass = pass && good;
        if (print) std::printf("  %s rel_l2=%.9g norm_error=%.9g max_abs=%.9g tol=%.9g pass=%d\n",
                               names[f], l2, norm, max_abs, tolerance, good);
    }
    return pass;
}
void quant_contract(const Snapshot& s, int rows, int mode) {
    if (!mode) return;
    size_t misses = 0, nonzero = 0;
    for (int r = 0; r < rows; ++r) for (int g = 0; g < N / GROUP; ++g) {
        float v[GROUP], rotated[GROUP];
        float amax = 0;
        for (int j = 0; j < GROUP; ++j) {
            v[j] = host_bf(load<uint16_t>(s[0], r * N + g * GROUP + j));
            rotated[j] = v[j]; amax = std::max(amax, std::fabs(v[j]));
        }
        const float sc = amax > 0 ? amax * (1.0f / (mode == 2 ? 127.0f : 448.0f)) : 1.0f;
        misses += load<float>(s[3], r * (N / GROUP) + g) != sc;
        const float inv = 1.0f / sc;
        for (int j = 0; j < GROUP; ++j) {
            const size_t idx = r * N + g * GROUP + j;
            if (mode == 2) {
                const float z = std::max(-127.0f, std::min(127.0f, v[j] * inv));
                misses += load<int8_t>(s[2], idx) != (int)std::nearbyint(z);
            } else misses += s[2][idx] != encode_fp8(v[j] * inv);
            nonzero += (s[2][idx] & 127) != 0;
        }
        // Same ascending strides as the register-local then cross-lane FWHT.
        for (int stride = 1; stride < GROUP; stride *= 2)
            for (int i = 0; i < GROUP; i += 2 * stride)
                for (int j = 0; j < stride; ++j) {
                    const float a = rotated[i + j], b = rotated[i + j + stride];
                    rotated[i + j] = a + b; rotated[i + j + stride] = a - b;
                }
        float rm = 0;
        for (float x : rotated) rm = std::max(rm, std::fabs(x));
        const float rs = rm > 0 ? rm * (1.0f / 448.0f) : 1.0f;
        misses += load<float>(s[5], r * (N / GROUP) + g) != rs;
        const float ri = 1.0f / rs;
        for (int j = 0; j < GROUP; ++j)
            misses += s[4][r * N + g * GROUP + j] != encode_fp8(rotated[j] * ri);
    }
    require(misses == 0, "quantizer differs from BF16-output host contract: " + std::to_string(misses));
    require(nonzero > (size_t)rows * N / 2, "degenerate/vacuous quantized output");
}
void controls(const Snapshot& ref, const Snapshot& got, int rows, int mode, bool exact) {
    require(compare(ref, ref, mode, exact, false), "positive comparator control failed");
    for (int f = 0; f < (mode ? 6 : 2); ++f) {
        Snapshot bad = got;
        std::fill(bad[f].begin(), bad[f].end(), 0);
        require(!compare(ref, bad, mode, exact, false), std::string("negative comparator control accepted: ") + names[f]);
        if (!exact) continue;
        // One flipped bit is inside every WMMA tolerance; the byte-exact comparator must refuse it.
        bad = got;
        bad[f].back() ^= 1;
        require(!compare(ref, bad, mode, exact, false), std::string("one-bit negative control accepted: ") + names[f]);
    }
    if (rows == 3) {
        Snapshot bad = got;
        std::fill(bad[0].begin() + 2 * N * 2, bad[0].end(), 0);
        require(!compare(ref, bad, mode, exact, false), "M=3 final-row negative control accepted");
    }
    Bytes guard(GUARD * 2 + 32, 0xa5);
    require(Buffer::guards(guard, 32), "positive guard control failed");
    guard.back() ^= 1;
    require(!Buffer::guards(guard, 32), "negative guard control accepted");
    std::printf("  controls=PASS positive,each-present-output-zeroed%s,guards%s\n",
                exact ? ",each-present-output-last-byte-bit0-flipped" : "",
                rows == 3 ? ",M3-final-row-zeroed" : "");
}

struct Options {
    int device = 0, m = 0, warmup = 100, iters = 200, repeats = 9;
    bool check_only = false, rowpair = false, fused_norm = false, up_wmma = false, up_group = false;
    bool up_direct = false, up_split = false;
    std::string candidate_plugin;
};
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--check-only") { o.check_only = true; continue; }
        require(i + 1 < argc, "missing option value: " + a);
        if (a == "--candidate-plugin") { o.candidate_plugin = argv[++i]; continue; }
        if (a == "--variant") {
            const std::string v = argv[++i];
            require(v == "wmma" || v == "rowpair" || v == "wmma-fusednorm" ||
                    v == "up-wmma" || v == "up-wmma-group" || v == "up-wmma-direct" || v == "up-wmma-split",
                    "unknown variant (wmma|rowpair|wmma-fusednorm|up-wmma|up-wmma-group|up-wmma-direct|up-wmma-split): " + v);
            o.rowpair = v == "rowpair";
            o.fused_norm = v == "wmma-fusednorm";
            o.up_direct = v == "up-wmma-direct";
            o.up_group = v == "up-wmma-group" || o.up_direct;
            o.up_split = v == "up-wmma-split";
            o.up_wmma = v == "up-wmma" || o.up_group || o.up_split;
            continue;
        }
        char* end = nullptr;
        const long v = std::strtol(argv[++i], &end, 10);
        require(end && *end == '\0' && v >= 0 && v <= 1000000, "invalid option value");
        if (a == "--device") o.device = (int)v;
        else if (a == "--m") o.m = (int)v;
        else if (a == "--warmup") o.warmup = (int)v;
        else if (a == "--iters") o.iters = (int)v;
        else if (a == "--repeats") o.repeats = (int)v;
        else require(false, "unknown option: " + a);
    }
    if (!o.candidate_plugin.empty()) {
        require(!o.rowpair && !o.fused_norm && !o.up_group && !o.up_direct,
                "--candidate-plugin cannot select an experimental prototype");
        require(!o.m || o.m == 3 || o.m == 4, "candidate plugin supports M=3/4");
        o.up_wmma = o.up_split = true;
    }
    require(!o.m || o.m == 1 || o.m == 2 || o.m == 3 || o.m == 4 || o.m == 8, "unsupported M");
    require(!(o.rowpair || o.up_wmma) || o.m < kDecMinRows,
            "rowpair and up-wmma support M=1,2,3,4 only: frozen M>=8 dispatches its tiled path");
    require(o.warmup > 0 && o.iters > 0 && o.repeats >= 3 && o.repeats <= 101,
            "warmup/iters must be positive; repeats must be 3..101");
    return o;
}
struct Stream {
    hipStream_t value = nullptr;
    Stream() {
#ifdef HC_PROBE_AQL
        RadStream native = nullptr;
        require(rad_stream_create(&native, 0) == RAD_OK, "AQL stream creation failed");
        value = (hipStream_t)native;
#else
        HIP(hipStreamCreate(&value));
#endif
    }
    void sync() {
#ifdef HC_PROBE_AQL
        require(rad_stream_sync((RadStream)value) == RAD_OK, "AQL stream sync failed");
#else
        HIP(hipStreamSynchronize(value));
#endif
    }
    ~Stream() {
#ifdef HC_PROBE_AQL
        rad_stream_destroy((RadStream)value);
#else
        const auto e = hipStreamDestroy(value);
        if (e != hipSuccess) std::fprintf(stderr, "stream cleanup: %s\n", hipGetErrorString(e));
#endif
    }
};
struct Events {
#ifdef HC_PROBE_AQL
    RadEvent start = nullptr, stop = nullptr;
    Events() {
        require(rad_event_create(&start) == RAD_OK, "AQL start event creation failed");
        require(rad_event_create(&stop) == RAD_OK, "AQL stop event creation failed");
    }
    ~Events() { rad_event_destroy(start); rad_event_destroy(stop); }
#else
    hipEvent_t start = nullptr, stop = nullptr;
    Events() { HIP(hipEventCreate(&start)); HIP(hipEventCreate(&stop)); }
    ~Events() {
        const auto a = hipEventDestroy(start), b = hipEventDestroy(stop);
        if (a != hipSuccess || b != hipSuccess) std::fprintf(stderr, "event cleanup failed\n");
    }
#endif
};
template <class F> double measure(F&& f, hipStream_t stream, Events& e, int iters) {
    float ms = 0;
#ifdef HC_PROBE_AQL
    require(rad_event_record(e.start, (RadStream)stream) == RAD_OK, "AQL start record failed");
    for (int i = 0; i < iters; ++i) f();
    require(rad_event_record(e.stop, (RadStream)stream) == RAD_OK, "AQL stop record failed");
    require(rad_event_sync(e.stop) == RAD_OK, "AQL stop sync failed");
    require(rad_event_elapsed_ms(e.start, e.stop, &ms) == RAD_OK, "AQL event timing failed");
#else
    HIP(hipEventRecord(e.start, stream));
    for (int i = 0; i < iters; ++i) f();
    HIP(hipEventRecord(e.stop, stream));
    HIP(hipEventSynchronize(e.stop));
    HIP(hipEventElapsedTime(&ms, e.start, e.stop));
#endif
    require(ms > 0 && std::isfinite(ms), "invalid GPU event duration");
    return (double)ms * 1000.0 / iters;
}
void spreads(const char* path, std::vector<double> samples) {
    std::printf("  timing path=%s samples_us=", path);
    for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", samples[i]);
    std::sort(samples.begin(), samples.end());
    auto q = [&](double p) {
        const double pos = p * (samples.size() - 1);
        const size_t lo = (size_t)pos, hi = std::min(lo + 1, samples.size() - 1);
        return samples[lo] + (pos - lo) * (samples[hi] - samples[lo]);
    };
    std::printf(" min_us=%.6f p10_us=%.6f median_us=%.6f p90_us=%.6f max_us=%.6f\n",
                samples.front(), q(.1), q(.5), q(.9), samples.back());
}

int run(int argc, char** argv) {
    const Options o = options(argc, argv);
    const char* variant = !o.candidate_plugin.empty() ? "mine-plugin" : o.rowpair ? "rowpair"
                         : o.up_split ? "up-wmma-split" : o.up_direct ? "up-wmma-direct"
                         : o.up_group ? "up-wmma-group" : o.up_wmma ? "up-wmma"
                         : o.fused_norm ? "wmma-fusednorm" : "wmma";
    HIP(hipSetDevice(o.device));
#ifdef HC_PROBE_AQL
    require(rad_dev_set(o.device) == RAD_OK, "AQL device selection failed");
    RadDeviceProps native_props{};
    require(rad_dev_props(o.device, &native_props) == RAD_OK && !native_props.is_host_backend,
            "AQL probe requires the accelerator backend");
    constexpr const char* timing = "native_AQL_live_submission_events";
#else
    constexpr const char* timing = "ordinary_HIP_stream_events";
#endif
    hipDeviceProp_t properties{};
    HIP(hipGetDeviceProperties(&properties, o.device));
    require(std::strncmp(properties.gcnArchName, "gfx1201", 7) == 0, "this probe requires gfx1201");
    require(properties.warpSize == 32 && properties.sharedMemPerBlock >= 65536, "unsupported wave/LDS geometry");
    std::printf("device=%d name=%s arch=%s wave=%d reported_multiprocessors=%d\n",
                o.device, properties.name, properties.gcnArchName, properties.warpSize, properties.multiProcessorCount);
    if (!o.candidate_plugin.empty()) {
        std::printf("variant=mine-plugin supported_M=3,4 candidate=%s warmup=%d iters=%d repeats=%d\n",
                    o.candidate_plugin.c_str(), o.warmup, o.iters, o.repeats);
    } else if (o.rowpair) {
        std::printf("variant=rowpair supported_M=1,2,3,4 skipped_M=8(frozen_M>=kDecMinRows=%ld_is_tiled) "
                    "n=%d hc=%d lr=%d dld_bytes=%ld uld_bytes=%ld "
                    "kernels=M1:upstream_hc_read_a_kernel<true>(unchanged_control,not_optimized),"
                    "M2..4:rowpair_a<2>(M3:ghost_row)+upstream_hc_up_mix_launch "
                    "dot_prefetch=%d(upstream_kDotU=%d) norm_prefetch=%d(upstream=8) "
                    "ta_rule=r4d_hc_bf16.hip:3725_over_grid_rows numerik=bitgenau\n",
                    kDecMinRows, N, HC, LR, DLD, ULD, kRpU, kDotU, kRpPre);
        std::printf("comparator=byte_exact outputs=x,inject,q,qs,rq,rqs d_vs_frozen_dbuf=byte_exact "
                    "norm_scales=bitwise finite_required=1 quantizer_host_contract=exact warmup=%d iters=%d "
                    "repeats=%d\n", o.warmup, o.iters, o.repeats);
        // Attributes of the kernel A this run's M set actually launches.
        if (!o.m || rowpair_shared(o.m))
            kernel_attr("rowpair_a<2>(selected_M=2,3,4,dyn_lds=40960)", (const void*)rowpair_a<2>);
        kernel_attr(!o.m || !rowpair_shared(o.m)
                        ? "hc_read_a_kernel<true>(selected_M=1_unchanged_control,dyn_lds=20480)"
                        : "hc_read_a_kernel<true>(reference_only,dyn_lds=20480)",
                    (const void*)hc_read_a_kernel<true>);
    } else if (o.up_wmma) {
        std::printf("variant=%s supported_M=1,2,3,4 n=%d hc=%d lr=%d "
                    "up_grid=%d threads=%d channels_per_CTA=%d up_group=80 "
                    "numerik=verlustfrei-nicht-bitgenau down_d_and_norm=byte_exact "
                    "warmup=%d iters=%d repeats=%d counter_argument=%s\n",
                    variant, N, HC, LR, N / (o.up_group ? GROUP : 32), o.up_group ? 256 : 128,
                    o.up_group ? GROUP : 32, o.warmup, o.iters, o.repeats,
                    o.up_group || o.up_split ? "null" : "initialized");
        if (o.up_direct) {
            kernel_attr("up_wmma<false,128,true>", (const void*)up_wmma<false, GROUP, true>);
            kernel_attr("up_wmma<true,128,true>", (const void*)up_wmma<true, GROUP, true>);
        } else if (o.up_group) {
            kernel_attr("up_wmma<false,128>", (const void*)up_wmma<false, GROUP>);
            kernel_attr("up_wmma<true,128>", (const void*)up_wmma<true, GROUP>);
        } else {
            kernel_attr("up_wmma<false,32>", (const void*)up_wmma<false>);
            kernel_attr("up_wmma<true,32>", (const void*)up_wmma<true>);
        }
    } else {
        std::printf("variant=%s supported_M=1,2,3,4,8\n", variant);
        if (o.fused_norm)
            kernel_attr("down<true>(norm_fused,256_threads)", (const void*)down<true>);
        std::printf("n=%d hc=%d lr=%d dld_bytes=%ld down_scale_offset=%d down_group=%d "
                    "uld_bytes=%ld up_scale_offset=%d up_group=%d splitK=%d splitK_chunk=%d "
                    "down_grid=%dx1x%d RT=%d NT=%d partial_layout=[z][M][324] numerik=verlustfrei-nicht-bitgenau\n",
                    N, HC, LR, DLD, HN, GROUP, ULD, LR, LR / 4, SPLITS, GS * GROUP,
                    (NC + NT - 1) / NT, SPLITS, RT, NT);
        std::printf("budget_source=tools/rad_kbench.cpp:120-138 bf16_and_scales_rel_l2=0.006 q_rq_rel_l2=0.03 "
                    "all_elements=1 finite_required=1 quantizer_host_contract=exact warmup=%d iters=%d repeats=%d\n",
                    o.warmup, o.iters, o.repeats);
    }
    Oracle oracle;
    std::unique_ptr<Oracle> plugin;
    if (!o.candidate_plugin.empty())
        plugin = std::make_unique<Oracle>(o.candidate_plugin.c_str(), "mine_hc_read_up_wmma_m3_m4");
    Stream stream;
    Events events;
    int cases = 0;
    for (int rows : {1, 2, 3, 4, 8}) {
        if (o.m && rows != o.m) continue;
        if (plugin && rows < 3) continue;
        if ((o.rowpair || o.up_wmma) && rows >= kDecMinRows) {
            std::printf("skip M=%d variant=%s reason=frozen_M>=kDecMinRows_dispatches_tiled_path\n", rows, variant);
            continue;
        }
        for (uint64_t seed : {uint64_t(1), uint64_t(17), uint64_t(101)}) {
            Inputs in(rows, seed);
            std::array<Snapshot, 3> candidate_outputs;
            for (int mode = 0; mode < 3; ++mode) {
                Outputs baseline(rows), candidate(rows);
                std::unique_ptr<Work> work;
                std::unique_ptr<OracleCall> candidate_call;
                if (plugin) candidate_call = std::make_unique<OracleCall>(*plugin, in, candidate, rows, mode);
                else work = std::make_unique<Work>(rows, o.rowpair || o.up_wmma);
                OracleCall call(oracle, in, baseline, rows, mode);
                auto base = [&]() { call.launch(oracle, stream.value); };
                auto test = [&]() {
                    if (plugin) candidate_call->launch(*plugin, stream.value);
                    else if (o.rowpair) launch_rowpair(in, candidate, *work, rows, mode, stream.value);
                    else if (o.up_wmma) launch_up_wmma(in, candidate, *work, rows, mode, stream.value,
                                                     o.up_group, o.up_direct, o.up_split);
                    else launch(in, candidate, *work, rows, mode, stream.value, o.fused_norm);
                };
                auto check_work = [&]() {
                    in.check();
                    if (!plugin) { work->check(rows); check_scratch(*work, call, rows); return; }
                    const Bytes own = candidate_call->scratch->get(), frozen = call.scratch->get();
                    const size_t dbytes = (size_t)rows * LR * 2;
                    require(own.size() >= dbytes + rows * kHcMax * 4 &&
                            std::equal(own.begin(), own.begin() + dbytes, frozen.begin()),
                            "plugin down output differs bytewise");
                    for (int r = 0; r < rows; ++r)
                        for (int c = 0; c < HC; ++c) {
                            const size_t i = dbytes / 4 + r * kHcMax + c;
                            require(load<uint32_t>(own, i) == load<uint32_t>(frozen, i),
                                    "plugin normalization differs bytewise");
                        }
                };
                const long rg = hc_up_rg(rows, LR);
                std::printf("case M=%d seed=%llu mode=%s variant=%s RG=%ld oracle_row=%s index=%d "
                            "oracle_scratch_bytes=%zu %s=%zu\n", rows,
                            (unsigned long long)seed, mode == 0 ? "none" : mode == 1 ? "fp8+rot" : "int8+rot",
                            variant, rg, oracle.row->name, oracle.index,
                            call.scratch->size, plugin ? "plugin_scratch_bytes" : work->rowpair ? "d_bytes" : "partial_bytes",
                            plugin ? candidate_call->scratch->size : work->rowpair ? work->d.size : work->part.size);
                const bool shared = o.rowpair && rowpair_shared(rows);
                const char* path = !o.rowpair ? variant : shared ? "rowpair_a<2>" : "upstream_hc_read_a_control";
                if (o.rowpair) {
                    const int groups = rowpair_groups(rows);
                    const long ta = hc_read_ta(groups), fta = hc_read_ta(rows);
                    std::printf("  launches=%s(grid=%ldx%d,ta=%ld,ghost_rows=%d)+hc_up_mix_kernel<RG=%ld,"
                                "F8=1,I8=%d>(grid=%dx%ld) frozen_ladder=hc_read_a_kernel(grid=%ldx%d,ta=%ld)\n",
                                shared ? "rowpair_a<2>" : "hc_read_a_kernel<true>", (LR + ta - 1) / ta + 1,
                                groups, ta, shared ? 2 * groups - rows : 0, rg, mode == 2,
                                N / (kUpRows / HC), (rows + rg - 1) / rg, (LR + fta - 1) / fta + 1, rows, fta);
                }
                base(); test();
                stream.sync();
                const Snapshot ref = baseline.get(), got = candidate.get();
                require(compare(ref, got, mode, o.rowpair, true), "frozen oracle comparison failed; timing prohibited");
                if (o.up_wmma) require(ref[1] == got[1], "unchanged injection output differs bytewise");
                quant_contract(ref, rows, mode); quant_contract(got, rows, mode);
                controls(ref, got, rows, mode, o.rowpair);
                check_work();
                if (o.fused_norm) {
                    Outputs unfused(rows);
                    Work unfused_work(rows, false);
                    launch(in, unfused, unfused_work, rows, mode, stream.value, false);
                    stream.sync();
                    require(unfused.get() == got && unfused_work.scale.get() == work->scale.get() &&
                            unfused_work.part.get() == work->part.get(),
                            "fused norm changed the three-launch WMMA outputs or intermediates");
                    unfused_work.check(rows);
                }
                std::printf("  reached=%s evidence=%s\n", path,
                            plugin ? "plugin_launch_origin,dirty_scratch,d_eq_frozen_dbuf"
                            : work->rowpair ? "d_written,d_eq_frozen_dbuf,wmma_part_untouched"
                                           : "part_written,rowpair_d_untouched");
                // A second operation without resetting counters must produce identical bytes.
                base(); test();
                stream.sync();
                require(baseline.get() == ref && candidate.get() == got, "repeated operation changed output");
                check_work();
                if (plugin) {
                    auto& pc = *candidate_call;
                    int refused = 0;
                    auto refuse = [&](int expected) {
                        require(plugin->row->launch(&pc.args, (RadStream)stream.value) == expected,
                                "plugin accepted or misclassified an invalid call");
                        ++refused;
                    };
                    const RadParam band = pc.params[0];
                    for (int bad : {1, 2, 5, 8}) {
                        pc.params[0] = pi("M", bad);
                        refuse(RAD_E_SHAPE);
                    }
                    pc.params[0] = band;
                    --pc.args.scratch_bytes; refuse(RAD_E_SCRATCH); ++pc.args.scratch_bytes;
                    const RadTensor up = pc.tensors[3], h = pc.tensors[0], x = pc.tensors[5];
                    pc.tensors[3].dtype = RAD_BF16; refuse(RAD_E_DTYPE); pc.tensors[3] = up;
                    pc.tensors[3].data = (unsigned char*)up.data + 1;
                    refuse(RAD_E_STRIDE); pc.tensors[3] = up;
                    pc.tensors[0].shape[0] = 2; refuse(RAD_E_SHAPE); pc.tensors[0] = h;
                    pc.tensors[0].stride[0] += 8; refuse(RAD_E_SHAPE); pc.tensors[0] = h;
                    --pc.tensors[5].shape[0]; refuse(RAD_E_STRIDE); pc.tensors[5] = x;
                    if (mode) {
                        const RadTensor qs = pc.tensors[8], rqs = pc.tensors[10];
                        pc.tensors[8].data = nullptr; refuse(RAD_E_INVAL); pc.tensors[8] = qs;
                        pc.tensors[10].data = nullptr; refuse(RAD_E_INVAL); pc.tensors[10] = rqs;
                    }
                    if (rows == 4) {
                        pc.params[0] = pi("M", 3); refuse(RAD_E_SHAPE); pc.params[0] = band;
                    }
                    if (rows == 3) {
                        pc.params[0] = pi("M", 4);
                        Buffer wider((size_t)plugin->row->scratch(&pc.args));
                        pc.args.scratch = wider.data(); pc.args.scratch_bytes = wider.size;
                        for (auto& b : candidate.b) b->reset();
                        test(); stream.sync();
                        require(candidate.get() == got, "M=3 inside band=4 changed output");
                        const Bytes wide = wider.get(), frozen = call.scratch->get();
                        const size_t used_d = rows * LR * 2, band_d = 4 * LR * 2;
                        require(std::equal(wide.begin(), wide.begin() + used_d, frozen.begin()),
                                "wider-band down output differs");
                        require(std::all_of(wide.begin() + used_d, wide.begin() + band_d,
                                            [](unsigned char v) { return v == 0xcd; }),
                                "wider band computed a nonexistent fourth row");
                        for (int r = 0; r < rows; ++r)
                            for (int c = 0; c < HC; ++c)
                                require(load<uint32_t>(wide, band_d / 4 + r * kHcMax + c) ==
                                        load<uint32_t>(frozen, used_d / 4 + r * kHcMax + c),
                                        "wider-band norm offset differs");
                        pc.params[0] = band;
                        pc.args.scratch = pc.scratch->data(); pc.args.scratch_bytes = pc.scratch->size;
                    }
                    require(plugin->row->describe, "stateless plugin has no replay description");
                    for (auto& b : candidate.b) b->reset();
                    pc.scratch->reset();
                    const Snapshot dirty = candidate.get();
                    const Bytes scratch_dirty = pc.scratch->get();
                    RadLaunchDesc desc[3]{};
                    int ndesc = -1;
                    require(plugin->row->describe(&pc.args, (RadStream)stream.value, desc, 1, &ndesc) == RAD_E_FULL,
                            "short descriptor capacity was accepted");
                    require(plugin->row->describe(&pc.args, (RadStream)stream.value, desc, 3, &ndesc) == RAD_OK &&
                            ndesc == (mode ? 3 : 2), "plugin replay description failed");
                    require(candidate.get() == dirty && pc.scratch->get() == scratch_dirty,
                            "describe touched device outputs or scratch");
                    if (seed == 1 && mode == 2)
                        for (int i = 0; i < ndesc; ++i)
                            kernel_attr(i == 0 ? "plugin_down" : i == 1 ? "plugin_up" : "plugin_quant", desc[i].func);
                    for (int i = 0; i < ndesc; ++i)
                        HIP(hipLaunchKernel(desc[i].func,
                            dim3(desc[i].grid[0], desc[i].grid[1], desc[i].grid[2]),
                            dim3(desc[i].block[0], desc[i].block[1], desc[i].block[2]),
                            desc[i].params, desc[i].shared_bytes, stream.value));
                    stream.sync();
                    require(candidate.get() == got, "described replay differs from live launch");
                    check_work();
                    std::printf("  plugin_ABI=PASS refusals=%d dirty_scratch=PASS band_boundary=PASS "
                                "describe_no_device_effect=PASS replay_descriptors=%d replay=byte_exact\n",
                                refused, ndesc);
                }
                candidate_outputs[mode] = got;
                if (mode) {
                    require(got[0] == candidate_outputs[0][0] && got[1] == candidate_outputs[0][1],
                            "quantized output mode changed BF16 x/inject");
                    if (mode == 2)
                        require(got[4] == candidate_outputs[1][4] && got[5] == candidate_outputs[1][5],
                                "INT8 mode changed rotated FP8 output");
                }
                std::printf("  correctness=PASS %s,all_guards,immutable_inputs,repeat,cross_mode\n",
                            o.rowpair ? "byte_exact_all_outputs,d_eq_frozen_dbuf"
                                      : o.up_wmma ? "all_outputs,byte_exact_d_and_inject" : "all_outputs");
                if (!o.check_only) {
                    for (int i = 0; i < o.warmup; ++i) { base(); test(); }
                    stream.sync();
                    std::vector<double> a, b;
                    for (int repeat = 0; repeat < o.repeats; ++repeat) {
                        if (repeat & 1) {
                            b.push_back(measure(test, stream.value, events, o.iters));
                            a.push_back(measure(base, stream.value, events, o.iters));
                        } else {
                            a.push_back(measure(base, stream.value, events, o.iters));
                            b.push_back(measure(test, stream.value, events, o.iters));
                        }
                    }
                    require(baseline.get() == ref && candidate.get() == got, "timed operation changed output");
                    check_work();
                    spreads("frozen_libr4d", a);
                    spreads(plugin ? "mine_plugin_native" : o.up_split ? "stateless_up_wmma_probe"
                            : o.up_direct ? "two_launch_up_wmma_direct_probe"
                            : o.up_group ? "two_launch_up_wmma_group_probe" : o.up_wmma ? "two_launch_up_wmma_probe"
                            : o.fused_norm ? "two_launch_fusednorm_probe" : !o.rowpair ? "three_launch_probe"
                            : rowpair_shared(rows) ? "rowpair_probe" : "rowpair_M1_upstream_control", b);
                }
                ++cases;
            }
        }
    }
    std::printf("PASS cases=%d timing=%s variant=%s; standalone evidence only, not an engine/gate result\n",
                cases, o.check_only ? "disabled" : timing, variant);
    return 0;
}
} // namespace probe

int main(int argc, char** argv) {
    try { return probe::run(argc, argv); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
