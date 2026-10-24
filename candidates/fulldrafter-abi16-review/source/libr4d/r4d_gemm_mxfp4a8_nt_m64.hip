// r4d_gemm_mxfp4a8_nt_m64.hip - r4d_gemm_mxfp4a8_nt_m64: OCP-MXFP4 weight, fp8 activation skinny
// GEMM for M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, A quantised per row to
// e4m3 and W stored as OCP micro-scaling fp4: an e2m1 element per nibble and one E8M0 exponent
// per 32 elements along K.
//
// WHY THIS EXISTS ALONGSIDE r4d_gemm_w4a8_nt_m64.  That kernel's 4-bit code is an int4 the library
// derives at load; this one's is a checkpoint format it does not get to choose.  An MXFP4
// checkpoint arrives with its own per-32 E8M0 exponents, and a serving stack that re-derives them
// as int4 has quantised twice.  The two also want different matrix instructions -- iu8 there, fp8
// here -- and on this part both run at the same rate, so the format is the whole decision:
//
//     v_wmma_f32_16x16x16_f16       207 TF/s      5.95 cyc/instr
//     v_wmma_f32_16x16x16_fp8_fp8   412           2.99
//     v_wmma_i32_16x16x16_iu8       407           3.03
//
// THE PER-32 EXPONENT IS FOLDED INTO THE ELEMENT, WHICH IS WHY THE UNPACK IS CHEAP.  The naive
// MXFP4 dequant multiplies each element by 2^(E8M0-127), which is a per-32-block rescale inside
// the inner loop and puts a multiply on every weight.  Instead, take a per-ROW reference exponent
// Wref[n] = max_k E8M0[n][k] once, offline, and fold the difference d = Wref[n] - E8M0[n][blk]
// into the e4m3 VALUE: the eight e2m1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by 2^-d are
// themselves e4m3 numbers, so a 16-entry table of packed e4m3 bytes indexed by d replaces the
// multiply with a v_perm_b32.  What is left is one factor per row, 2^(Wref[n]-127), applied once
// in the epilogue next to the activation's own per-row scale.
//
// The table runs into e4m3 SUBNORMALS on purpose.  Stopping at the smallest normal (2^-6) flushes
// every element whose block exponent is more than 5 binades below its row reference; on this
// checkpoint that is 6.7% of the weights on 36 output channels, for per-channel errors up to
// 18.7%.  Extending to 2^-9 makes every d <= 8 exact.  That is only sound because the gfx12 fp8
// WMMA HONOURS e4m3 subnormals rather than flushing them: the product is exact for every m*2^-9,
// m = 1..7.  d >= 13 rounds to zero, which is correct: those elements are below the
// representable range of the row.
//
// UNPACK IS FOUR v_perm_b32 PER EIGHT WEIGHTS.  A lane's fragment is eight consecutive k, which is
// four packed bytes.  Splitting the low and high nibbles gives the even and odd elements; each is
// mapped through the folded-magnitude table by a v_perm_b32 that indexes it with the 3-bit
// magnitude, the sign rides along as bit 3 shifted into the e4m3 sign position, and two more
// v_perm_b32 interleave the halves back into k order.  No multiply, no exponent arithmetic in the
// loop, and the scale lookup is once per fragment rather than once per element: eight consecutive
// k always lie inside one 32-element block, because the fragment's k offset is a multiple of 8.
//
// SPLIT-K IS IN THE WORKGROUP, AS EVERYWHERE ELSE IN THIS LIBRARY.  With one 16-row M fragment the
// independent N fragments number N/16 -- 320 waves at N=5120 -- which does not fill the part, so
// the K range has to be split.  Doing it across workgroups needs an fp32 partial buffer in HBM and
// a counter to decide who reduces; doing it across waves of one workgroup reduces in LDS and needs
// neither.  The LDS form is also what keeps this kernel cudagraph-safe with no scratch to register:
// a caller can replay it without having handed the library any memory.
//
// THE WEIGHT IS PRE-PERMUTED INTO FRAGMENT ORDER, AS IT IS FOR w4a16/w4a8.  A B fragment maps lane
// l onto N row l&15, so in checkpoint order the sixteen lanes of a half-wave read sixteen rows K/2
// bytes apart -- a fully strided read on the operand that dominates traffic.  Reordering the weight
// once at load fixes it and is what lets this kernel skip LDS entirely.
//
// The layout is one 32-lane slot per (n-tile, k-step):
//
//     Wp[((n/16) * (K/16) + (k/16)) * 32 + lane]      one uint32 = 4 packed bytes = 8 elements
//
// where lane's row is 16*(n/16) + (lane&15) and its eight elements start at k + 8*(lane>>4).  A
// wave then reads 128 contiguous bytes per step.  In checkpoint terms, taking W as
// [N/16][16][K/16][8] bytes, slot l of tile (nt, ks) is W[nt][l&15][ks][4*(l>>4) .. +4].
// permute_w() in mxfp4_layout.py is the reference implementation.
//
// The permute is what makes the LDS-free form worth having.  Reading checkpoint order directly,
// this kernel is slower than an LDS-staged one on every shape; with the permute it is faster than
// the LDS-staged form on every shape, at both ends of the M range.  It also moves the tuning: the
// fastest configs are the ones with deeper split-K (SK 4-8) and NPW 2, which only pay once the
// weight read coalesces.
//
// The E8M0 scales are left in [K/32][N] order -- a half-wave reads sixteen consecutive bytes of
// those already.
//
// TRAP FOR ANYONE MEASURING THIS KERNEL: a weight of tens of megabytes streamed in a loop is
// Infinity-Cache resident on this part, so a bandwidth figure taken that way can read above the
// DRAM roofline.  It ranks kernels reading the same bytes; it is not achieved DRAM bandwidth.
//
// A IS NOT STAGED IN LDS.  A lane's A fragment is eight consecutive bytes of one row, so the
// global load already has the shape the fragment wants; staging it would cost LDS, a barrier and a
// second read for nothing.  Rows past M are CLAMPED rather than masked -- a tile past M recomputes
// the last valid row and the store discards it -- which keeps the inner loop branch-free.
#include "r4d_gemm_args.h"
#include "r4d_common.h"

#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_MXFP4_MAX_M 64
#define R4D_GEMM_MXFP4_GROUP 32       // K per E8M0 exponent, fixed by the OCP MX format
#define WAVE32 32

typedef int      v2i    __attribute__((ext_vector_type(2)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

// Folded magnitudes: kMag[d] is the eight e2m1 magnitudes scaled by 2^-d, as packed e4m3 bytes,
// low table then high table, indexed by the 3-bit magnitude through v_perm_b32.
// Every entry is the round-to-nearest-EVEN e4m3 of that product, which is exact for d <= 8 and
// a rounding for d >= 9 -- three entries there (d=9 mag 1.5, d=10 mag 3, d=11 mag 6) are the
// same exact tie 1.5*2^-9 and take the even neighbour 2^-8, not the smaller one. They matter
// little (such an element is 2.9e-3 of its row's reference) but the table is generated, not
// hand-rounded, so it should agree with the hardware's own conversion.
__device__ __constant__ unsigned int kMag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u}, {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u}, {0x02010000u, 0x06040302u},
  {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

// Eight e2m1 codes (four packed bytes) -> eight e4m3 bytes in k order, with the block's exponent
// difference already folded in.
__device__ __forceinline__ v2i r4d_mxfp4_unpack8(unsigned int wv, int dsh) {
  const unsigned int t0 = kMag[dsh][0], t1 = kMag[dsh][1];
  const unsigned int ev = wv & 0x0F0F0F0Fu;          // low nibble of each byte  -> even elements
  const unsigned int od = (wv >> 4) & 0x0F0F0F0Fu;   // high nibble             -> odd elements
  const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
  const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
  v2i out;
  out[0] = (int)__builtin_amdgcn_perm(bo, be, 0x05010400u);   // elements 0..3
  out[1] = (int)__builtin_amdgcn_perm(bo, be, 0x07030602u);   // elements 4..7
  return out;
}

template <int MT, int NPW>
__global__ void r4d_gemm_mxfp4a8_nt_m64_kernel(
    const unsigned char* __restrict__ A, const float* __restrict__ As,
    const unsigned char* __restrict__ W, const unsigned char* __restrict__ Ws,
    const unsigned char* __restrict__ Wref, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK, kbase = ksplit * Ks;
  const int nlane = lane & 15, mhalf = (lane >> 4) * 8;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  // Column tiles this wave owns, CLAMPED to the last real tile. grid.x rounds N/16 up to a
  // multiple of ncols = WV*NPW, so the tail block addresses tiles that do not exist -- at N=48
  // with ncols=4 it reaches tile 3 of 3. Clamping the TILE (not just the row) matters because the
  // weight is indexed by tile: an unclamped tile index walks off the end of the buffer, which is a
  // GPU memory fault rather than a wrong answer. The epilogue still drops these columns.
  const int ntmax = N / 16 - 1;
  int wtile[NPW], wrow[NPW];
  unsigned char wref[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int t = nt0 + j;
    wtile[j] = t < ntmax ? t : ntmax;
    wrow[j] = wtile[j] * 16 + nlane;
    wref[j] = Wref[wrow[j]];
  }

  const unsigned int* __restrict__ Wp = (const unsigned int*)W;
  const int ksteps = K / 16;

  for (int k = 0; k < Ks; k += 16) {
    const int kg = kbase + k;                       // fragment base, both lane halves
    const int kstep = kg / 16;
    const int blk = kg / R4D_GEMM_MXFP4_GROUP;      // one E8M0 block covers this whole step
    v2i af[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + nlane;
      r = r < M - 1 ? r : M - 1;
      const unsigned int* pa = (const unsigned int*)(A + (size_t)r * K + kg + mhalf);
      af[i][0] = (int)pa[0];
      af[i][1] = (int)pa[1];
    }
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = wrow[j];
      // Fragment-order weight: lane l's four bytes for this (n-tile, k-step) sit at slot l, so the
      // wave reads 128 contiguous bytes. In checkpoint order these would be sixteen rows K/2 bytes
      // apart -- a fully strided read on the operand that dominates traffic.
      const unsigned int wv = Wp[((size_t)wtile[j] * ksteps + kstep) * WAVE32 + lane];
      int dsh = (int)wref[j] - (int)Ws[(size_t)blk * N + n];
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      const v2i wf = r4d_mxfp4_unpack8(wv, dsh);
#pragma unroll
      for (int i = 0; i < MT; i++)
        acc[j][i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[j][i]);
    }
  }

  // Reduce the SK partial sums in LDS, one M tile at a time so the buffer stays WV*NPW*SK*256.
  const int ncols = WV * NPW;

  // AT SK == 1 THERE IS NOTHING TO REDUCE: lane already holds the finished value.
  if (SK == 1) {
    if (nt0 * 16 >= N) return;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = (nt0 + j) * 16 + (lane & 15);
      if (n >= N) continue;
#pragma unroll
      for (int i = 0; i < MT; i++) {
        const int mb = (mt0 + i) * 16 + 8 * (lane >> 4);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int m = mb + e;
          if (m < M) C[(size_t)m * N + n] =
              (__hip_bfloat16)(acc[j][i][e] * __int_as_float((int)Wref[n] << 23) * As[m]);
        }
      }
    }
    return;
  }

#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + lane * 8 + e] = acc[j][i][e];
    }
    __syncthreads();
    // One pass over the block's 256-element tiles; every thread takes a stride.
    for (int t = tid; t < ncols * 256; t += blockDim.x) {
      const int col = t >> 8, off = t & 255;
      // off = lane*8 + e, so the lane is off>>3 -- but a lane's N index is lane&15, and its
      // eight accumulator elements are the M values 8*(lane>>4)+e. Using the whole lane id for N
      // writes the upper half-wave into the NEXT column tile; it is invisible for M <= 8 because
      // those rows are then out of range and dropped by the guard below.
      const int n = (blockIdx.x * ncols + col) * 16 + ((off >> 3) & 15);
      const int m = (mt0 + i) * 16 + ((off >> 7) << 3) + (off & 7);
      if (n >= N || m >= M) continue;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      C[(size_t)m * N + n] =
          (__hip_bfloat16)(v * __int_as_float((int)Wref[n] << 23) * As[m]);
    }
  }
}

extern "C" int r4d_gemm_mxfp4a8_nt_m64_max_m(void) { return R4D_GEMM_MXFP4_MAX_M; }
extern "C" int r4d_gemm_mxfp4a8_nt_m64_group(void) { return R4D_GEMM_MXFP4_GROUP; }

extern "C" void r4d_gemm_mxfp4a8_nt_m64(long a, long ascale, long wq, long ws, long wref, long c,
                                        int M, int K, int N, int WV, int SK, int MB, int NPW,
                                        long stream) {
  if (M < 1 || M > R4D_GEMM_MXFP4_MAX_M)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_MXFP4_GROUP))
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: K must be divisible by SK*32");
  if (WV * SK * WAVE32 > 1024)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024)
    throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned char*)wq;
  auto Zp = (const unsigned char*)ws;
  auto Rp = (const unsigned char*)wref;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MTv, NPv) hipLaunchKernelGGL((r4d_gemm_mxfp4a8_nt_m64_kernel<MTv, NPv>), grid, \
                                            block, smem, st, Ap, Sp, Wp, Zp, Rp, Cp, M, K, N, WV, SK)
#define LAUNCH_MB(NPv) switch (MB) { \
    case 1: LAUNCH(1, NPv); break; \
    case 2: LAUNCH(2, NPv); break; \
    case 3: LAUNCH(3, NPv); break; \
    case 4: LAUNCH(4, NPv); break; \
    default: throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: MB unsupported"); }
  switch (NPW) {
    case 1: LAUNCH_MB(1); break;
    case 2: LAUNCH_MB(2); break;
    case 4: LAUNCH_MB(4); break;
    case 8: LAUNCH_MB(8); break;
    default: throw std::runtime_error("r4d gemm_mxfp4a8_nt_m64: NPW unsupported");
  }
#undef LAUNCH_MB
#undef LAUNCH
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
extern "C" int r4d_rad_gemm_mxfp4a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_mxfp4a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    /* The per-row reference exponent is a weight aux stream and RadArgs has no way to deliver one,
     * so gemm_nt_q carries it as a trailing optional weight operand. Without it the fold has no
     * reference and every element would be scaled against zero. */
    if (!g.bref) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_mxfp4a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_FLAT, &c);
    if (rc != RAD_OK) return rc;
    R4D_TRY("gemm_mxfp4a8_nt_m64",
        r4d_gemm_mxfp4a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                r4d_addr(g.bscale), r4d_addr(g.bref), r4d_addr(g.y),
                                (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw,
                                (long)(uintptr_t)s));
    return RAD_OK;
}

