// W4A8 GEMM FOR PREFILL: the same packed weight and the same rotated int8 activation as
// r4d_gemm_w4a8_tiled, laid out for the regime where the matrix pipe is the answer.
//
// WHY A SECOND KERNEL.  At a decode batch this GEMM is a weight stream and every question is
// bandwidth: `gate_up_proj` at M = 9 moves 47 MB and runs at the DRAM ceiling, so the inner loop's
// instruction mix does not matter because the loop is waiting on memory.  At M = 4096
// the same call is 730 GFLOP against 44.6 MB, so the loop is waiting on nothing and every
// instruction that is not a matrix instruction is a tax.  The two want different code.
//
// THE CEILING THIS AIMS AT.  `v_wmma_i32_16x16x16_iu8` occupies its SIMD for EIGHT cycles on
// gfx1201 -- half what f16/bf16 take, at 16.0-16.2 cycles -- so the int8 matrix peak is ~375 TOPS,
// not the ~187 an f16 roofline would say.  gfx1201 has no wider iu8 form: `llvm-mc` rejects
// `v_wmma_i32_16x16x32_iu8`, and `v_wmma_i32_16x16x32_iu4` is exactly twice the rate but needs
// BOTH operands 4-bit, which the activation is not.  So the budget is 8 cycles of matrix work per
// matrix instruction plus one cycle per other VALU instruction, and the job is to make the second
// term small.  In a register-only replica of this inner loop, the matrix instructions on their own
// reach most of that peak and the weight unpack costs little on top; the GROUP FOLD is the largest
// non-matrix item in the loop, and it costs materially more as a `v_fma_mix_f32` than as a plain
// `v_fmac_f32`.
//
// The reason is dual issue.  `v_fma_mix_f32` is VOP3P, which can never dual-issue;
// `v_dual_fmac_f32` is the ONLY VOPD form gfx1201 offers that this loop can use
// (`v_dual_and_b32` and `v_dual_lshlrev_b32` do not exist here, so the unpack cannot be packed at
// all).  This kernel therefore forces the f16 scale to materialise and folds with plain FMACs.
//
// THE OTHER HALF IS THE ACTIVATION FRAGMENT READ.  The tiled kernel stores A in LDS k-major and
// reads one `ds_load_b64` per (row tile, k step): 32 reads and ~30 `s_wait_dscnt` per scale group.
// Here the row's 64 bytes are stored HALF-MAJOR -- [row][lane half][k step][2] instead of
// [row][k step][lane half][2] -- which puts a lane's two consecutive k steps side by side, so one
// `ds_load_b128` fetches them both and a whole 64-k block of A is TM*2 reads instead of TM*4.
// The permutation is inside the LDS copy only: the packed weight is untouched and the activation
// in memory is untouched, because a lane still consumes exactly the k values the weight names.
//
// The 4-dword XOR swizzle keeps that conflict-free.  A row is 16 dwords and LDS has 32 banks, so
// the row index contributes only its low bit; the mask has to separate the rest and it has to
// permute whole 4-dword BLOCKS or a `ds_load_b128` lands astride two of them.  ((r >> 1) & 3) << 2
// gives lanes 0..7 the start banks {0,16,4,20,8,24,12,28} -- eight distinct starts, four dwords
// each, all 32 banks once.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_PF_BK 64

typedef int      v2i_t __attribute__((ext_vector_type(2)));
typedef int      v8i_t __attribute__((ext_vector_type(8)));
typedef float    v8f_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v_t __attribute__((ext_vector_type(4)));
typedef float    v4f_t __attribute__((ext_vector_type(4)));

// The unpacked byte is 16*nibble, read SIGNED, and the 16 comes back out in the store.  Three
// instructions for two dwords is the floor here: gfx1201 has no `v_lshl_and_b32` and no nibble
// permute, so each output dword costs at least its own mask.
__device__ __forceinline__ v2i_t pf_dq8(unsigned w) {
  v2i_t r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

// OPT bits
//   1  grid swizzle: walk GSW m-tiles per n-tile so the resident workgroups form a square and
//      both the A row block and the W column block are shared by several of them
//   2  fold with plain `v_fmac_f32` (VOPD) instead of letting the f16 convert contract into a
//      `v_fma_mix_f32`
//   4  split barrier: signal after the commit, do the k block's matrix work, wait at the top
//   8  read the A fragments one `ds_load_b64` per k step (the tiled kernel's shape), for comparison
template <int WM, int WN, int TM, int TN, int OPT, int GSW>
__global__ __launch_bounds__(WM * WN * 32, 8) void r4d_gemm_w4a8_prefill_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N) {
  constexpr int BK = R4D_PF_BK;
  constexpr int GRP = R4D_GEMM_W4A8_GROUP;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AD = BM * 16;                        // A dwords per buffer (BM rows x 16)
  constexpr int WD = NTILE * 128;                    // W dwords per buffer
  constexpr int NT_ = WM * WN * 32;
  constexpr int UNR = GRP / BK;                      // k blocks per scale group
  constexpr bool VOPD = (OPT & 2) != 0;
  constexpr bool SPLITB = (OPT & 4) != 0;
  constexpr bool NARROWA = (OPT & 8) != 0;
  // ABLATIONS.  Both compute wrong answers on purpose and both keep every instruction that is not
  // the thing being priced, so what they measure is one item and not a deleted pipeline.
  //   16  every fragment read comes from buffer 0 at k = 0: the same LDS reads, the same count,
  //       the same banks, but no dependency on the staging that filled it
  //   32  the A fragment never leaves registers: prices the A LDS read itself
  constexpr bool ABSTALE = (OPT & 16) != 0;
  constexpr bool ABNOA = (OPT & 32) != 0;
  //   64  store one output element in eight, with the other seven folded into it so nothing in the
  //       k loop goes dead: prices the store epilogue, which is 64 predicated bf16 stores, 64 f32
  //       multiplies, 64 software bf16 roundings and 32 `Ascale` loads per lane
  constexpr bool ABSTORE = (OPT & 64) != 0;
  //  128  load the eight `Ascale` values of a row tile as two b128 instead of eight scalars: the
  //       rows a lane owns (m0 + i*16 + 8*half + e, e = 0..7) are CONTIGUOUS
  constexpr bool ASCVEC = (OPT & 128) != 0;
  //  256  an unpredicated store path for the blocks that are wholly inside the matrix, which at a
  //       prefill M is all but the last row block: 64 compares and 64 cndmasks per lane go away
  //  512  non-temporal stores for C, which is written once and never read by this kernel
  constexpr bool STFAST = (OPT & 256) != 0;
  constexpr bool STNT = (OPT & 512) != 0;

  // Both stagings move 16 bytes per thread per pass, with a predicate on the last one so a block
  // shape does not have to divide the block width.  AD/4 = BM*4 and WD/4 = (BN/16)*32, so at 512
  // threads that is exact only for BM = 128 and BN = 256; every other tile in the sweep is ragged.
  constexpr int AU = AD / 4, WU = WD / 4;
  constexpr int APT = (AU + NT_ - 1) / NT_, WPT = (WU + NT_ - 1) / NT_;
  constexpr bool APAD = (AU % NT_) != 0, WPAD = (WU % NT_) != 0;

  __shared__ unsigned lds_a[2][AD];
  __shared__ unsigned lds_w[2][WD];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int half = lane >> 4;                        // which 8-byte half of the 16-k step

  int bx = blockIdx.x, by = blockIdx.y;
  if (OPT & 1) {                                     // square-ish walk over the grid
    const int nt = gridDim.x, mt = gridDim.y;
    const int lin = by * nt + bx, per = GSW * nt;
    const int grp = lin / per, rem = lin - grp * per;
    const int gm = (mt - grp * GSW) < GSW ? (mt - grp * GSW) : GSW;
    by = grp * GSW + rem % gm;
    bx = rem / gm;
  }
  const int m0 = by * BM, n0 = bx * BN;
  const int kpb = K / BK, nsz = K / GRP;

  // A LDS position of (row, lane half, k step). Bank-conflict-free swizzling:
  // For 64-bit loads (NARROWA), 3-bit swizzle (((r >> 1) & 7) << 1) achieves 2-way bank access
  // across all 32 banks. For 128-bit loads (b128), 4-dword block swizzle (((r >> 1) & 3) << 2)
  // maintains 16-byte alignment while evenly spreading 32 lanes across 8 4-bank groups (4-way).
#define PF_APOS(r, h, u) ((r) * 16 + ((((h) * 8 + (u) * 2)) ^ (NARROWA ? ((((r) >> 1) & 7) << 1) : ((((r) >> 1) & 3) << 2))))

  v8f_t acc[TN][TM];
  v8i_t g[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f_t)(0.f);
  float csum[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) csum[j] = 0.f;
  unsigned scv[TN];

  u4v_t ra[APT], rw[WPT];
  auto issue = [&](int kk) {
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t >> 2, q = t & 3;
      if (APAD && t >= AU) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = *(const u4v_t*)(A + (unsigned)(mm * K + kk + q * 16));
    }
#pragma unroll
    for (int x = 0; x < WPT; x++) {
      const int t = tid + x * NT_, tl = t >> 5, o4 = t & 31;
      if (WPAD && t >= WU) continue;
      rw[x] = *(const u4v_t*)(Wq + (unsigned)(((n0 / 16 + tl) * kpb + kk / BK) * 128 + o4 * 4));
    }
  };
  auto commit = [&](int buf) {
#pragma unroll
    for (int x = 0; x < APT; x++) {
      const int t = tid + x * NT_, r = t >> 2, q = t & 3;
      if (APAD && t >= AU) continue;
      // the 16 bytes are k = 16q .. 16q+15: dwords 0,1 are the low half, 2,3 the high half
      v2i_t lo, hi;
      lo[0] = (int)ra[x][0]; lo[1] = (int)ra[x][1];
      hi[0] = (int)ra[x][2]; hi[1] = (int)ra[x][3];
      *(v2i_t*)(lds_a[buf] + PF_APOS(r, 0, q)) = lo;
      *(v2i_t*)(lds_a[buf] + PF_APOS(r, 1, q)) = hi;
    }
#pragma unroll
    for (int x = 0; x < WPT; x++) {
      const int t = tid + x * NT_;
      if (WPAD && t >= WU) continue;
      *(u4v_t*)(lds_w[buf] + t * 4) = rw[x];
    }
  };
  auto bar = [&]() {
    // THE LDS STORES IN `commit` HAVE TO LAND BEFORE THE SIGNAL, which is what the
    // release fence is for.  `s_barrier_signal` is not `__syncthreads()`: it emits
    // the barrier alone, with no wait on the outstanding `ds_store`s.  Without the
    // fence a wave announces writes that are still in flight and its neighbours read
    // whatever LDS held from the previous k block -- the generated code carries
    // `s_wait_loadcnt` for the global loads feeding the stores and no `s_wait_dscnt`
    // at all, and the kernel returns a different answer on every call.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
  };

  issue(0);
  commit(0);
  bar();

  for (int kg = 0; kg < K; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++) {
      const int k0 = kg + h * BK;
      const int buf = h & 1;
      if (k0 + BK < K) issue(k0 + BK);
      if ((k0 % GRP) == 0) {                         // the group scale, one whole k block early
#pragma unroll
        for (int j = 0; j < TN; j++)
          scv[j] = Ws[(unsigned)(((n0 / 16 + wn * TN + j) * nsz + k0 / GRP) * 16 + (lane & 15))];
      }

      // ---- fragments: the whole 64-k block, up front ----
      v2i_t af[TM][BK / 16];
#pragma unroll
      for (int step = 0; step < TM; step++) {
        const int i = (step + wn) % TM;
        const int row = (wm * TM + i) * 16 + (lane & 15);
        if (ABNOA) {
#pragma unroll
          for (int u = 0; u < BK / 16; u++) {
            af[i][u][0] = (int)(0x11111111u + i + u);
            af[i][u][1] = (int)(0x22222222u + i);
          }
          continue;
        }
        if (NARROWA) {
#pragma unroll
          for (int u = 0; u < BK / 16; u++) {
            const unsigned* p = lds_a[ABSTALE ? 0 : buf] + PF_APOS(row, half, u);
            af[i][u][0] = (int)p[0];
            af[i][u][1] = (int)p[1];
          }
        } else {
#pragma unroll
          for (int q = 0; q < BK / 32; q++) {        // one b128 = two k steps
            const u4v_t v =
                *(const u4v_t*)(lds_a[ABSTALE ? 0 : buf] + PF_APOS(row, half, 2 * q));
            af[i][2 * q][0] = (int)v[0];
            af[i][2 * q][1] = (int)v[1];
            af[i][2 * q + 1][0] = (int)v[2];
            af[i][2 * q + 1][1] = (int)v[3];
          }
        }
      }
      u4v_t wv[TN];
#pragma unroll
      for (int j = 0; j < TN; j++)
        wv[j] = *(const u4v_t*)(lds_w[ABSTALE ? 0 : buf] + (size_t)(wn * TN + j) * 128 + lane * 4);

      if (SPLITB) {                                  // the next block's LDS write, then arrive
        if (k0 + BK < K) commit(buf ^ 1);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");   // see bar()
        __builtin_amdgcn_s_barrier_signal(-1);
      }

      // ---- the matrix work ----
#pragma unroll
      for (int u = 0; u < BK / 16; u++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const v2i_t bf = pf_dq8(wv[j][u]);
#pragma unroll
          for (int i = 0; i < TM; i++)
            g[j][i] = ((k0 % GRP) == 0 && u == 0)
                          ? __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, af[i][u], true, bf, (v8i_t)(0x4B400000), false)
                          : __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                                true, af[i][u], true, bf, g[j][i], false);
        }
      }

      if ((k0 + BK) % GRP == 0) {                    // fold the closed group into the f32 acc
        float sc[TN];
#pragma unroll
        for (int j = 0; j < TN; j++) {
          sc[j] = __half2float(__ushort_as_half((unsigned short)(scv[j] & 0xFFFFu)));
          // Materialise the convert.  Left alone the compiler folds it into `v_fma_mix_f32`, which
          // reads the f16 for free and is VOP3P -- and VOP3P never dual-issues, so the fold stays
          // at one instruction per accumulator element instead of one per two.
          if (VOPD) asm("" : "+v"(sc[j]));
          csum[j] += sc[j];
        }
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int i = 0; i < TM; i++)
#pragma unroll
            for (int e = 0; e < 8; e++)
              acc[j][i][e] = fmaf(__uint_as_float((unsigned)g[j][i][e]), sc[j], acc[j][i][e]);
      }

      if (SPLITB) {
        __builtin_amdgcn_s_barrier_wait(-1);
      } else {
        if (k0 + BK < K) commit(buf ^ 1);
        bar();
      }
    }
  }
#undef PF_APOS

#pragma unroll
  for (int j = 0; j < TN; j++) {
    const float d = -12582912.0f * csum[j];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[j][i][e] += d;
  }

  if (ABSTORE) {
#pragma unroll
    for (int j = 0; j < TN; j++)
#pragma unroll
      for (int i = 0; i < TM; i++) {
        float v = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) v += acc[j][i][e];
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
        if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * 0.0625f);
      }
    return;
  }
  const bool inside = STFAST && (m0 + BM <= M);
#pragma unroll
  for (int i = 0; i < TM; i++) {
    float as[8];
    const int mb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
    if (ASCVEC) {
      // The vector reads Ascale[mc .. mc+7], which lies inside [0, M) only when M >= 8. Below that
      // it would read past the last row's scale -- past the operand, when the plane is exactly M
      // long -- so a short M takes the clamped scalar read for every lane instead. M is uniform
      // over the grid, so the branch costs one compare.
      const int mc = mb < M - 8 ? mb : (M > 8 ? M - 8 : 0);
      if (M >= 8) {
        const v4f_t* pv = (const v4f_t*)(Ascale + mc);
        const v4f_t s0 = pv[0], s1 = pv[1];
#pragma unroll
        for (int e = 0; e < 4; e++) { as[e] = s0[e]; as[e + 4] = s1[e]; }
      }
      if (M < 8 || mb != mc) {
#pragma unroll
        for (int e = 0; e < 8; e++) as[e] = Ascale[mb + e < M ? mb + e : M - 1];
      }
    }
#pragma unroll
    for (int j = 0; j < TN; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = mb + e;
        const __hip_bfloat16 v =
            (__hip_bfloat16)(acc[j][i][e] * (ASCVEC ? as[e] : Ascale[m]) * 0.0625f);
        if (inside || (n < N && m < M)) {
          // the non-temporal builtin does not take a __hip_bfloat16*, so go through its bits
          if (STNT)
            __builtin_nontemporal_store(*(const unsigned short*)&v,
                                        (unsigned short*)(C + (size_t)m * N + n));
          else
            C[(size_t)m * N + n] = v;
        }
      }
  }
}

template <int WM, int WN, int TM, int TN, int OPT, int GSW>
static void pf_launch(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                      long stream) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  if (N % BN) throw std::runtime_error("r4d gemm_w4a8_prefill: N must be a multiple of BN");
  if (K % R4D_GEMM_W4A8_GROUP) throw std::runtime_error("r4d gemm_w4a8_prefill: K % group");
  dim3 grid(N / BN, (M + BM - 1) / BM);
  hipLaunchKernelGGL((r4d_gemm_w4a8_prefill_kernel<WM, WN, TM, TN, OPT, GSW>), grid,
                     dim3(WM * WN * 32), 0, (hipStream_t)stream, (const signed char*)a,
                     (const float*)ascale, (const unsigned*)wq, (const unsigned*)ws,
                     (__hip_bfloat16*)c, M, K, N);
}

void r4d_gemm_w4a8_prefill(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                           long stream, int variant) {
#define PF(...) pf_launch<__VA_ARGS__>(a, ascale, wq, ws, c, M, K, N, stream)
  switch (variant) {
    case 1:  PF(2, 8, 4, 2, 0, 1); break;                 // no VOPD fold, b128 A
    case 2:  PF(2, 8, 4, 2, 8, 1); break;                 // b64 A, no VOPD fold
    case 3:  PF(2, 8, 4, 2, 10, 1); break;                // b64 A, VOPD fold
    case 4:  PF(2, 8, 4, 2, 6, 1); break;                 // + split barrier
    case 5:  PF(2, 8, 4, 2, 3, 4); break;                 // + grid swizzle, GSW 4
    case 6:  PF(2, 8, 4, 2, 3, 8); break;                 // + grid swizzle, GSW 8
    case 7:  PF(2, 8, 4, 2, 7, 8); break;                 // swizzle + split barrier
    case 8:  PF(2, 8, 2, 2, 2, 1); break;                 // BM 64,  BN 256
    case 9:  PF(2, 8, 3, 2, 2, 1); break;                 // BM 96,  BN 256
    case 10: PF(2, 8, 4, 1, 2, 1); break;                 // BM 128, BN 128
    case 11: PF(4, 8, 4, 2, 2, 1); break;                 // BM 256, BN 256, 1024 threads
    case 12: PF(2, 16, 4, 2, 2, 1); break;                // BM 128, BN 512, 1024 threads
    case 13: PF(1, 8, 4, 2, 2, 1); break;                 // BM 64,  BN 256, 256 threads
    case 14: PF(2, 8, 8, 1, 2, 1); break;                 // BM 256, BN 128
    case 15: PF(4, 4, 4, 2, 2, 1); break;                 // BM 256, BN 128, 512 threads
    case 20: PF(2, 8, 4, 2, 2 | 16, 1); break;            // abl: fragments from a stale buffer
    case 21: PF(2, 8, 4, 2, 2 | 32, 1); break;            // abl: A fragment never touches LDS
    case 22: PF(2, 8, 4, 2, 2 | 16 | 32, 1); break;       // abl: both
    case 23: PF(2, 8, 4, 2, 8 | 16, 1); break;            // abl: stale, b64 A reads
    case 24: PF(2, 8, 4, 2, 0, 1); break;                 // b128 A, fma_mix fold
    case 25: PF(2, 8, 4, 2, 2 | 64, 1); break;            // abl: one store in eight
    case 26: PF(2, 8, 4, 2, 8 | 64, 1); break;            // abl: one store in eight, b64 A
    case 27: PF(2, 8, 4, 2, 2 | 128, 1); break;           // vectorised Ascale
    case 28: PF(2, 8, 4, 2, 8 | 128, 1); break;           // vectorised Ascale, b64 A
    case 29: PF(2, 8, 4, 2, 8 | 128 | 256, 1); break;     // + unpredicated store
    case 30: PF(2, 8, 4, 2, 8 | 128 | 256 | 512, 1); break;  // + non-temporal
    case 31: PF(2, 8, 4, 2, 2 | 128 | 256, 1); break;     // b128 A + both
    case 32: PF(2, 8, 4, 2, 8 | 128 | 256 | 1, 4); break; // + grid swizzle 4
    case 33: PF(2, 8, 4, 2, 8 | 128 | 256 | 1, 8); break; // + grid swizzle 8
    case 34: PF(2, 8, 4, 2, 2 | 128 | 256 | 1, 4); break; // b128 A + VOPD + swizzle 4
    case 35: PF(2, 8, 4, 2, 2 | 128 | 256 | 1, 8); break; // b128 A + VOPD + swizzle 8
    default: PF(2, 8, 4, 2, 2, 1); break;                 // b128 A + VOPD fold
  }
#undef PF
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
extern "C" int r4d_rad_gemm_w4a8_prefill(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    if (is_asym(a)) return R4D_NO(RAD_E_UNSUPPORTED);   /* no asymmetric prefill form */
    if (g.K % R4D_GEMM_W4A8_GROUP) return R4D_NO(RAD_E_SHAPE);
    R4dTileCfg tv;
    r4d_cfg_prefill(a, &tv);
    const R4dTileCfg* t = &tv;
    rc = tile_ok(t, g.N);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_w4a8_prefill",
        r4d_gemm_w4a8_prefill(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                              r4d_addr(g.bscale), r4d_addr(g.y),
                              (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s, t->variant));
    return RAD_OK;
}

