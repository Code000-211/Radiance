// r4d_gemm_bf16_nt_m64.hip - r4d_gemm_bf16_nt_m64: skinny bf16 GEMM at any M, gfx1201.
//
// THE NAME SAYS 64 AND THE KERNEL IS GENERAL IN M. MT row tiles live in a block and `grid.y`
// covers the rest -- `mt0 = blockIdx.y * MT`, fragment rows clamped to M-1, the store guarded by
// `m < M` -- so any M up to R4D_GEMM64_MAX_M is served. The 64 is the registry band this row is
// listed under, not a limit of the kernel; the M <= 16 row forwards here as well.
//
// C[M,N] = A[M,K] @ W[N,K]^T, bf16 in and out, fp32 accumulate.
//
// THE SCALAR ALTERNATIVE is a wave per output COLUMN holding M accumulators for it, so every
// 16-wide step of K costs one weight load and M activation loads. That ratio is the whole cost at
// these shapes: at M=16, N=48, K=5120 the activation is re-read sixteen times per weight element,
// 7.9 MB against a 480 KiB weight, so the kernel is bound by traffic it did not need to move.
// Above M=16 that form cannot run at all -- 64 accumulators per column leaves nothing to prefetch
// with.
//
// WMMA fixes the ratio by construction. One wmma_f32_16x16x16 consumes a 16x16 activation fragment
// and a 16x16 weight fragment and produces a whole 16(M) x 16(N) tile, so a step of 16 K costs
// ceil(M/16) activation fragments and ONE weight fragment however large M is, and the accumulator
// for a tile is 8 VGPRs per lane instead of 16 scalars per output element. At M=64 that is four
// fragments and four wmma per weight fragment, against sixty-four loads before.
//
// Parallel decomposition: a block is WV column-tiles x SK k-splits
// waves, each wave owning one 16-wide N tile over one contiguous slice of K, and the k-splits are
// reduced through LDS in wave order so the result does not depend on how the waves are scheduled.
// N=48 (a TP=2 shard of the gated-delta-net b/a projection) is three tiles; three waves would leave
// the machine idle, so SK is what fills it.
//
// Fragment rows are CLAMPED, not masked: a tile whose rows run past M or N re-reads the last valid
// row, computes a result nobody stores, and never touches memory it does not own.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM64_MAX_M 65536
#define WAVE32 32

// A fragment from a base pointer plus a COMPILE-TIME k step. The header's fragGo takes a 32-bit
// element offset, which is correct but leaves LLVM re-deriving a 64-bit address per fragment per
// step -- a b64 shift and a carry-in add pair, three instructions each, twenty per step around four
// WMMA. Holding one pointer per fragment and stepping it with a constant turns those into the
// 13-bit immediate field of global_load_b64: a k step is 32 bytes, so eight of them fit.
//
// The pointer is explicitly in the global address space. Held as a plain `const uint2*` it is a
// GENERIC pointer once it lives in an array across loop iterations -- LLVM's address-space
// inference does not see through that -- and every load becomes flat_load_b64, which on gfx12
// increments dscnt as well as loadcnt, so any later LDS wait also waits on HBM.
// The element type is a scalar rather than uint2: HIP's vector types are structs whose assignment
// operator does not accept an address-space-qualified source, and an 8-byte scalar carries the same
// alignment the paired load needs.
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;

__device__ __forceinline__ v8bf fragU(r4d_gu2 p, int u) {
  union { unsigned long long w[2]; v8bf f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// MT = row tiles per BLOCK, held as MT v8f accumulators (8 VGPRs each); grid.y covers the rest.
// More row tiles per block amortises the weight fragment over more WMMA, fewer of them put more
// blocks on the machine -- which one wins is a per-shape question, so MT is tunable, not assumed.
// KU is how many k steps the loop unrolls; see the loop for why it exists.
#ifndef KU
#define KU 8
#endif
template <int MT>
__global__ __launch_bounds__(1024) void r4d_gemm_bf16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned short* __restrict__ W,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];              // WV*SK*256 floats: one 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt = blockIdx.x * WV + col_local;
  const int n0 = nt * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, ksteps = Ks / 16, kbase = ksplit * Ks;

  v8f acc[MT];
#pragma unroll
  for (int i = 0; i < MT; i++) acc[i] = (v8f)(0.f);

  // Row offsets as 32-bit ELEMENT offsets: computing them as pointers instead makes the size_t row
  // stride a 64-bit multiply-add and the row clamp exec-mask work, none of which LLVM keeps live
  // across the loop. They are turned into pointers ONCE, below.
  unsigned woff = rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
  unsigned aoff[MT];
#pragma unroll
  for (int i = 0; i < MT; i++)
    aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;

  // Unrolled so the k step is a load IMMEDIATE rather than an address update. Rolled, LLVM widens
  // each offset to a 64-bit induction variable and rebuilds the address every step: a b64 shift and
  // a carry-in add pair per fragment, twenty instructions per step around four WMMA. Unrolled, the
  // sub-step offset is a constant that folds into the 13-bit field of global_load_b64, and the base
  // offsets advance once per block of KU.
  if (n0 < N) {
    r4d_gu2 pw = (r4d_gu2)(const unsigned long long*)(W + woff);
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);
    int s = 0;
    for (; s + KU <= ksteps; s += KU) {
#pragma unroll
      for (int u = 0; u < KU; u++) {
        const v8bf b = fragU(pw, u);
#pragma unroll
        for (int i = 0; i < MT; i++) acc[i] = mma(fragU(pa[i], u), b, acc[i]);
      }
      pw += KU * 4;
#pragma unroll
      for (int i = 0; i < MT; i++) pa[i] += KU * 4;
    }
    for (; s < ksteps; s++) {
      const v8bf b = fragU(pw, 0);
      pw += 4;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        acc[i] = mma(fragU(pa[i], 0), b, acc[i]);
        pa[i] += 4;
      }
    }
  }

  // Reduce one row tile at a time: SK partial tiles is WV*SK KiB of LDS whatever MT is, where
  // holding all MT at once would be four times that at M=64. The sum is spread over every thread
  // in the block rather than left to the k-split-0 waves, which at SK=8 is one output element per
  // thread instead of sixty-four; the order over splits is fixed, so the result does not depend on
  // how the waves were scheduled.
  const int slot = (col_local * SK + ksplit) * 256;
  const int nthreads = WV * SK * WAVE32;

  // AT SK == 1 THERE IS NOTHING TO REDUCE: lane already holds the finished value.
  if (SK == 1) {
    if (n0 >= N) return;
    const int n = nt * 16 + (lane & 15);
    if (n < N) {
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int mb = (mt0 + i) * 16 + 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int m = mb + e;
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)acc[i][e];
        }
      }
    }
    return;
  }

  // AT N == 1: compact [split][m] epilog (1.25 KiB at SK=20). Only lanes with
  // (lane & 15) == 0 store their 8 accum values. Finisher reduces only live M rows.
  if (N == 1) {
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const int row_base = (mt0 + i) * 16;
      if (row_base >= M) break;
      const int live_m = (M - row_base < 16) ? (M - row_base) : 16;
      __syncthreads();
      if (col_local == 0 && (lane & 15) == 0) {
        const int m_base = 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          red[ksplit * 16 + m_base + e] = acc[i][e];
        }
      }
      __syncthreads();
      if (tid < live_m) {
        float v = 0.f;
#pragma unroll
        for (int s = 0; s < SK; s++) {
          v += red[s * 16 + tid];
        }
        C[(size_t)(row_base + tid) * N + 0] = (__hip_bfloat16)v;
      }
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < MT; i++) {
    const int row_base = (mt0 + i) * 16;
    if (row_base >= M) break;
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[i][e];   // conflict-free
    __syncthreads();
    for (int idx = tid; idx < WV * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      const int l = off & 31;
      const int n = (blockIdx.x * WV + col) * 16 + (l & 15);
      const int m = row_base + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) {
        float v = 0.f;
        for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
        // The language's own float-to-bf16 conversion, deliberately and not the header's f2bf,
        // which rounds half away from zero.
        C[(size_t)m * N + n] = (__hip_bfloat16)v;
      }
    }
  }
}

// grid = ceil(ceil(N/16)/WV) blocks; block = WV*SK waves. MT is compile-time so the accumulators
// and the fragment loop are exactly the size M needs.
void r4d_gemm_bf16_nt_m64(long a, long w, long c, int M, int K, int N, int WV, int SK, int MB,
                          long stream) {
  if (M < 1 || M > R4D_GEMM64_MAX_M) throw std::runtime_error("r4d gemm_bf16_nt_m64: M out of range");
  if (K % (SK * 16)) throw std::runtime_error("r4d gemm_bf16_nt_m64: K must be divisible by SK*16");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_bf16_nt_m64: threads/block > 1024");
  const int ntiles = (N + 15) / 16, mtiles = (M + 15) / 16;
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_bf16_nt_m64: MB must be 1..4");
  dim3 grid((ntiles + WV - 1) / WV, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)WV * SK * (N == 1 ? 16 : 256) * sizeof(float);
  auto Ap = (const unsigned short*)a; auto Wp = (const unsigned short*)w;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT) hipLaunchKernelGGL(r4d_gemm_bf16_nt_m64_kernel<MT>, grid, block, smem, st, \
                                      Ap, Wp, Cp, M, K, N, WV, SK)
  switch (MB) {
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    case 4: LAUNCH(4); break;
    default: throw std::runtime_error("r4d gemm_bf16_nt_m64: MB unsupported");
  }
#undef LAUNCH
}

int r4d_gemm_bf16_nt_m64_max_m(void) { return R4D_GEMM64_MAX_M; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
extern "C" int r4d_rad_gemm_bf16_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 0);
    if (rc != RAD_OK) return rc;
    if (g.M > r4d_gemm_bf16_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    Cfg c;
    rc = cfg_for(a, g.K, g.M, 16, BR_FLAT, &c, (g.N + 15) / 16);  /* "K divisible by SK*16" */
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_bf16_nt_m64",
        r4d_gemm_bf16_nt_m64(r4d_addr(g.a), r4d_addr(g.b), r4d_addr(g.y),
                             (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb,
                             (long)(uintptr_t)s));
    return RAD_OK;
}

