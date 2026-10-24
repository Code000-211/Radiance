// r4d_gemm_w4a8_nt_m64.hip - r4d_gemm_w4a8_nt_m64: 4-bit-weight, 8-bit-activation skinny GEMM for
// M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with A quantised per row.
//
// WHY THIS EXISTS ALONGSIDE r4d_gemm_w4a16_nt_m64.  The f16 kernel is at the DRAM roofline at M=8
// (620-648 GB/s) and there is nothing left to win there.  At M=64 -- the drafter's batch at eight
// concurrent requests -- it is not near memory at all: 300-350 GB/s and 72-78 TF/s, against a
// v_wmma_f32_16x16x16_f16 issue ceiling of 207 TF/s, and removing the unpack entirely barely
// changes that.  The wall is the f16 matrix instruction, and on this part it is the slow one:
//
//     v_wmma_f32_16x16x16_f16       207 TF/s      5.95 cyc/instr
//     v_wmma_f32_16x16x16_fp8_fp8   412           2.99
//     v_wmma_i32_16x16x16_iu8       407           3.03
//     v_wmma_i32_16x16x32_iu4       830           2.97
//
// So the widening that made the f16 kernel cheap is what caps it: every 8-bit form runs at twice
// the rate. This kernel takes the int8 one, which needs no exponent games at all.
//
// THE UNPACK IS THREE INSTRUCTIONS AND NEEDS NO ZERO POINT.  Store the weight as a 4-bit TWO'S
// COMPLEMENT code and put the nibble in the HIGH half of its byte: for q in -8..7 the byte 16*q is
// exactly the signed int8 of 16*q, so
//
//     lo = (w << 4) & 0xF0F0F0F0     bytes <- nibbles 0,2,4,6      (elements 0..3)
//     hi =  w       & 0xF0F0F0F0     bytes <- nibbles 1,3,5,7      (elements 4..7)
//
// is the whole dequant: two v_and_b32 and one v_lshlrev_b32 per eight weights, against eleven for
// the f16 route, and the factor of 16 folds into the group scale offline.  A symmetric grid also
// deletes the zero point, which in an integer kernel would otherwise cost a per-group row-sum of A
// and a second correction term.  A symmetric 4-bit grid is the best of the 4-bit options on
// quality anyway, so this costs nothing to pick.
//
// The nibble ORDER is the one r4d_gemm_w4a16_nt_m64 already uses: its f16 unpack pairs nibble 2b
// into element b and nibble 2b+1 into element 4+b, which is exactly what the two masks above
// produce.  One packed layout, two kernels.
//
// IT SHARES ITS WEIGHT WITH r4d_gemm_w4a16_nt_m64, BYTE FOR BYTE.  The two kernels want the same
// nibbles in the same order, and a signed 4-bit code is what the f16 kernel computes anyway when
// its integer zero point is the constant 8: it builds 1024+n and adds -(1024+8), leaving n-8.  So
// one quantised weight feeds both, and the scale array is the f16 kernel's (scale, -(1024+zero))
// dword with only the low half read here.  The stored scale multiplies q-8 while the WMMA here
// sees 16*(q-8), so the factor of 1/16 rides along with the per-row activation scale at the store,
// where it costs nothing.  That matters because the dispatch is per call: below M=16 the f16
// kernel is already at the DRAM roofline and there is nothing for an int8 one to win, while the
// activation quantiser costs a launch -- so the small-M path stays on f16 and the same bytes serve
// both.
//
// THE ACTIVATION IS PRE-SHUFFLED TOO, BUT ONLY INSIDE A K STEP.  A fragment element e of lane l is
// A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], so read out of a row-major A a lane's eight bytes are
// two dwords eight bytes apart: 64 global_load_b32 against 64 WMMA for one group, and 32
// s_wait_loadcnt with them.  The activation layout is this library's to choose -- the quantiser
// writes it -- so it is written with the sixteen bytes of each k step reordered to
// [half 0's eight][half 1's eight].  A lane's fragment is then one global_load_b64 and the group
// costs 32 loads instead of 64.
//
// DO NOT LAY A OUT AS TWO PLANES.  [2][M][K/2] makes a lane's four k steps contiguous and the
// group eight global_load_b128, and it is SLOWER than the shuffle below.  Sixteen lanes of a wave
// then address one plane and sixteen the other, Mp*K/2 bytes away, so a single instruction spans
// two distant regions and doubles the cache lines it touches.  The shuffle keeps every lane of a
// wave inside the same sixteen rows the row-major version touches, and only makes each lane's own
// eight bytes adjacent.
//
// THE ACTIVATION SCALE COMES OUT OF THE GROUP LOOP.  Per-row symmetric quantisation makes the row
// scale a constant in n and in the group index, so sum_g s_g * <a_row, w_g> * ar = ar * sum_g ...:
// it is applied once at the store and never inside the k loop.  What does stay in the loop is one
// v_cvt_f32_i32 per accumulator element per group, because group scales differ -- eight of them
// against eight WMMA.  That argued for a WIDE group, and this kernel alone would take 256; the
// group is 128 because the f16 kernel at M=8 needs the split-K that a wider group takes away
// (`K % (SK*GROUP) == 0` halves the legal split count), and one artifact feeds every W4 kernel.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

// M is not structural here -- `mt0 = blockIdx.y * MT` tiles it and the row index is clamped into
// 0..M-1 -- so this is a guard against a garbage shape, not the band the kernel is good at.  It is
// set wide enough to reach a prefill chunk as well as the decode band, where at eight concurrent
// sequences with spec width 8 the target verifies 8 x 9 = 72 rows.  Whether the tuned geometry
// above M = 64 is still the right one is a separate question and is NOT answered by this bound.
#define R4D_GEMM_W4A8_MAX_M 8192
// R4D_GEMM_W4A8_GROUP comes from r4d_common.h -- see the note there.
#define R4D_GEMM_W4A8_KPB   64       // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32

typedef int      v8i   __attribute__((ext_vector_type(8)));
typedef float    v4f   __attribute__((ext_vector_type(4)));
typedef int      v2i   __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(1))) const uint4v*   r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned* r4d_gu1;
typedef __attribute__((address_space(1))) const unsigned* r4d_gh1;
typedef __attribute__((address_space(1))) const unsigned short* r4d_gh2;
typedef unsigned uint2v __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(1))) const uint2v*   r4d_ga2;

// A fragment out of the pre-permuted plane: eight contiguous bytes per k step, so a uint4 is two
// steps and the whole group is eight global_load_b128 per row tile.
#ifndef R4D_W4A8_APERM
#define R4D_W4A8_APERM 1        // 0 reads A row-major, for a harness with an unpermuted buffer
#endif

// THE SCALE DWORD IS HALF DEAD ON A SYMMETRIC GRID, AND AT THE DECODE BATCH THAT IS PURE COST.
// The shared (f16 scale, f16 of -(1024 + zero)) dword exists for r4d_gemm_w4a16_nt_m64, which
// builds its value as 1024 + n and needs the second half to subtract.  A symz grid only ever
// writes the constant 8 there, and THIS kernel never reads it: the ISA shows one
// `global_load_b32` per group whose high sixteen bits are dropped by the `op_sel_hi` on the
// v_fma_mix that consumes it.
//
// At group 128 the scale array is 4 bytes per 16 columns per group against 1024 bytes of nibbles,
// so the dead half is 32 of every 1088 bytes -- 2.94% of everything the kernel reads.  That is not
// an instruction count, it is DRAM traffic, and at M <= 16 this kernel runs at 97-98% of the part's
// 639 GB/s read ceiling, where traffic is the ONLY thing that sets the time.  WS16 reads
// an f16-only scale plane instead.  Sixteen lanes then take 32 contiguous bytes and consecutive
// groups are adjacent, so the 64-byte line is still fully used -- the halving is real and not an
// artefact of counting bytes the cache would have fetched anyway.
//
// The asymmetric form keeps the dword: its whole reason for existing is the zero point.
#ifndef R4D_GEMM_W4A8_WS16
#define R4D_GEMM_W4A8_WS16 0
#endif

// COST ABLATIONS.  Wrong answers by construction; they exist to say what the kernel's time is made
// of.  WS16 above deletes 2.94% of the bytes read and does not move the time, so byte counts alone
// do not explain it.
//   1  no scale load        (sc = 1)
//   2  no activation load   (the A fragment is a constant)
//   4  no weight load       (the B fragment is a constant; leaves the A stream and the WMMA)
#ifndef R4D_W4A8_ABL
#define R4D_W4A8_ABL 0
#endif

// SOFTWARE-PIPELINE THE WEIGHT.  The ablations above say the weight stream IS the kernel at a
// decode M: deleting the scale load or the activation loads barely moves the time, while deleting
// the weight load removes most of it, and that stream runs at 587 GB/s against the part's
// 639.6 GB/s ceiling.  The loop as written issues its two `global_load_b128`, waits on them, then
// spends ~80 cycles in VALU and WMMA during which the wave issues no memory at all.  PF holds the
// NEXT group's weight in registers so the load is issued a whole group ahead of the wait.
// Costs bpg*NPW*4 more VGPRs (16 instead of 8 at NPW=1, against 58 used).
#ifndef R4D_W4A8_PF
#define R4D_W4A8_PF 0
#endif

// STAGGER THE GROUP ORDER PER WAVE, so the block's waves stop marching in lockstep across the
// memory channels.
//
// Wave (col_local, ksplit) reads n-tile plane `nt0+col_local` starting at byte offset
// `ksplit * (Ks/KPB) * 512` and advances 512 bytes per k-block.  Every plane pitch here is
// kpb*512 -- 40960 at K=5120, 69632 at 8704, 24576 at 3072 -- and every one of those is an exact
// multiple of 4096.  A 256-bit GDDR6 part interleaves 16 channels at 256 B, so the address period
// that sweeps every channel is 4096 B, which means at time t EVERY wave in the grid is at the same
// address modulo 4096 and therefore aimed at the same channel.  The block issues 20 concurrent
// streams and they all camp one channel group at a time.
//
// The fix is hipBLASLt's `StaggerU`: start each wave at a different group inside its OWN k-slice
// and wrap.  The slice, the split-K partition and the bytes read are all unchanged -- only the
// order is -- so the only cost is that `pa` must be computed from g instead of incremented.
//
// NOTE this reorders an fp32 accumulation, so results move by ~1 ULP against a non-staggered build.
#ifndef R4D_W4A8_STAGGER
#define R4D_W4A8_STAGGER 0
#endif

// Under STAGGER the group loop no longer visits groups in order, so the activation pointer is
// indexed rather than bumped; without it the bump stays and this is zero.
#if R4D_W4A8_STAGGER
#if R4D_W4A8_APERM
#define AOFF(g) ((g) * (R4D_GEMM_W4A8_GROUP / 8))
#else
#define AOFF(g) ((g) * (R4D_GEMM_W4A8_GROUP / 4))
#endif
#else
#define AOFF(g) 0
#endif

__device__ __forceinline__ v2i fragA8(r4d_ga2 p, int u) {
  const uint2v w = p[2 * u];
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}

__device__ __forceinline__ v2i fragA8row(r4d_gu1 p, int u) {
  v2i r;
  r[0] = p[4 * u];
  r[1] = p[4 * u + 2];
  return r;
}

// The whole dequant: two ands and a shift, no zero point. It is a few percent of the kernel at
// M=64 and under a fifth of a percent at M=8, so there is nothing here worth a knob.
__device__ __forceinline__ v2i dequant8(unsigned w) {
  v2i r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

// BSIGNED=false reads the unpacked byte as 16*nibble (0..240) rather than 16*twos(nibble)
// (-128..112).  That is what makes an ASYMMETRIC grid expressible: 16*nibble is affine in the
// nibble, so sum_k a_k (nibble_k - z) = sum_k a_k nibble_k - z sum_k a_k, whereas twos() is affine
// on each half of the range and not across it.  The flag's meaning on gfx1201: a B byte of 0x81
// gives -2032 signed against 2064 unsigned, a ratio of -127/129 exactly.
template <bool BSIGNED>
__device__ __forceinline__ v8i mmai(v2i a, v2i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, BSIGNED, b, c, false);
}

// __launch_bounds__ IS PER NPW, AND A FLAT 1024 COSTS THE WIDE ONES 192 REGISTERS.
// A block cannot exceed 1024 threads, but it also cannot exceed 64 KB of LDS, and the split-K
// buffer is WV*NPW*SK*1024 bytes -- so WV*NPW*SK <= 64 is structural and threads = 32*WV*SK is
// therefore capped at 32*64/NPW.  Telling the compiler 1024 for every NPW makes it budget for 32
// waves on one workgroup, which on this part is 1536/8 = 192 VGPRs per wave, and NPW=4 and NPW=8
// then SPILL against that cap (up to 1875 bytes at MT=4, NPW=8).  Those configs can never run more
// than 8 or 16 waves, so that cap is imaginary and the spill is not.
#ifndef R4D_GEMM_W4A8_LB
#define R4D_GEMM_W4A8_LB 1
#endif
#if R4D_GEMM_W4A8_LB
#define R4D_W4A8_BOUNDS(NPW) ((NPW) >= 8 ? 256 : ((NPW) >= 4 ? 512 : 1024))
#else
#define R4D_W4A8_BOUNDS(NPW) 1024
#endif

template <int MT, int NPW, bool NT, bool ASYM>
__global__ __launch_bounds__(R4D_W4A8_BOUNDS(NPW)) void r4d_gemm_w4a8_nt_m64_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W4A8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  // TWO THINGS THAT DO NOT WORK ON THE ASYMMETRIC CORRECTION'S ACTIVATION SUMS.
  //
  // Loading the sums just before the epilogue instead of at the top of the group, to cut
  // asum[MT][8]'s 32-VGPR live range at MT=4, is SLOWER.  At the top of the group the load is
  // already hidden behind the WMMA block, and the register pressure costs less than exposing that
  // latency does.
  //
  // Staging the sums in LDS is also SLOWER, at both ends of the M range.  Every wave of the block
  // re-reads a group's sums, and the staging can alias the split-K buffer for free since `red` is
  // only touched in the store loop and that loop opens with a barrier -- but the staging prologue
  // is (K/GROUP)*(MT*16) loads plus a __syncthreads() on a kernel that runs a few microseconds at a
  // decode M, and the sums are not the latency problem.

  if (n0 < N) {
    // Rows clamped, never masked, as in the f16 kernel: a tile past M recomputes the last valid
    // row and the store discards it.  A is int8 so the row stride is K bytes, not 2K.
    // Plane (lane >> 4), row clamped into 0..M-1, and the group's byte offset is K/2 per row.
#if R4D_W4A8_APERM
    r4d_ga2 pa[MT];
#else
    r4d_gu1 pa[MT];
#endif
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + (lane & 15);
      r = r < M - 1 ? r : M - 1;
#if R4D_W4A8_APERM
      const size_t off = (size_t)r * K + kbase + 8 * (lane >> 4);
      pa[i] = (r4d_ga2)(const uint2v*)(A + off);
#else
      const size_t off = (size_t)r * K + kbase + 4 * (lane >> 4);
      pa[i] = (r4d_gu1)(const unsigned*)(A + off);
#endif
    }

    const int kpb = K / R4D_GEMM_W4A8_KPB;
    const int nsz = K / R4D_GEMM_W4A8_GROUP;
    r4d_gw4 pw[NPW];
#if R4D_GEMM_W4A8_WS16
    r4d_gh2 ps[NPW];
#else
    r4d_gh1 ps[NPW];
#endif
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4A8_KPB) * WAVE32 + lane;
#if R4D_GEMM_W4A8_WS16
      ps[j] = (r4d_gh2)(const unsigned short*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W4A8_GROUP) * 16 + (lane & 15);
#else
      ps[j] = (r4d_gh1)(const unsigned*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W4A8_GROUP) * 16 + (lane & 15);
#endif
    }

    const int bpg = R4D_GEMM_W4A8_GROUP / R4D_GEMM_W4A8_KPB;

#if R4D_W4A8_PF
    uint4v nx[NPW][bpg];
#pragma unroll
    for (int j = 0; j < NPW; j++)
#pragma unroll
      for (int b = 0; b < bpg; b++)
        nx[j][b] = NT ? __builtin_nontemporal_load(&pw[j][(size_t)b * WAVE32])
                      : pw[j][(size_t)b * WAVE32];
#endif

#if R4D_W4A8_STAGGER
    const int gstag = groups > 1 ? (wave * R4D_W4A8_STAGGER) % groups : 0;
#endif
    for (int gi = 0; gi < groups; gi++) {
#if R4D_W4A8_STAGGER
      const int g = gi + gstag < groups ? gi + gstag : gi + gstag - groups;
#else
      const int g = gi;
#endif
      float sc[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#if R4D_W4A8_ABL & 1
        sc[j] = 1.f;
#elif R4D_GEMM_W4A8_WS16
        sc[j] = __half2float(__ushort_as_half(ps[j][(size_t)g * 16]));
#else
        sc[j] = __half2float(__ushort_as_half(
            (unsigned short)(ps[j][(size_t)g * 16] & 0xFFFFu)));
#endif

      // The zero point lives in the SAME dword's high half, as the f16 of -(1024 + zero) -- the
      // encoding the f16 kernel already needs, and exact because f16 has exactly 1024 steps between
      // 1024 and 2048.  The symz grid only ever writes the constant 8 there.
      float z16[NPW];
      float asum[MT][8];
      if (ASYM) {
#if R4D_W4A8_ASUM_ABLATE
        // Cost ablation only -- same arithmetic, no global loads.  Wrong answers by construction;
        // it exists to say whether the asymmetric path costs its FMAs or its memory.
#pragma unroll
        for (int j = 0; j < NPW; j++) z16[j] = 16.f;
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) asum[i][e] = 1.f;
#else
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const float nz = __half2float(__ushort_as_half(
              (unsigned short)(ps[j][(size_t)g * 16] >> 16)));
          z16[j] = 16.f * (-nz - 1024.f);
        }
        // Element e of tile i at this lane is output row (mt0+i)*16 + 8*(lane>>4) + e, so the eight
        // rows a lane needs are CONTIGUOUS at a fixed group -- which is why Asum is [G][Mpad] and
        // not [M][G].
        //
        // TWO v4f LOADS, NOT EIGHT SCALARS.  These loads, and not the correction's arithmetic, are
        // most of what an asymmetric grid costs over a symmetric one.
        // They vectorise only if the base is 32-byte aligned, which is why the row stride is padded
        // to a multiple of 16 (also making the last tile's rb+7 in bounds, since rb reaches
        // (mtiles-1)*16 + 8).
        const int Mpad = ((M + 15) / 16) * 16;
        const size_t gi = (size_t)(kbase / R4D_GEMM_W4A8_GROUP + g) * Mpad;
#pragma unroll
        for (int i = 0; i < MT; i++) {
          // CLAMPED, and it has to be.  When MB exceeds the number of M tiles -- MB=4 with M=9 is a
          // legal config the tuner will pick -- this loop still runs MT tiles, and rb reaches
          // (MT-1)*16+8 = 56 against an Mpad of 16.  The symz path survives that because its A
          // pointer clamps and the store discards rows past M; an unclamped Asum read is a page
          // fault.  Rows past M produce garbage that is discarded, so clamping the base is enough,
          // and it stays a multiple of 8 so the v4f loads stay aligned.
          int rb = (mt0 + i) * 16 + 8 * (lane >> 4);
          rb = rb < Mpad - 8 ? rb : Mpad - 8;
          const v4f* pv = (const v4f*)(Asum + gi + rb);
          const v4f v0 = pv[0], v1 = pv[1];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            asum[i][e] = v0[e];
            asum[i][e + 4] = v1[e];
          }
        }
#endif
      }

      v8i g_acc[NPW][MT];
#if R4D_W4A8_PF
      // Take this group's weight out of the registers the last iteration filled, and issue the
      // next group's loads BEFORE any of it is consumed, so the wait sits a full group away.
      uint4v cw[NPW][bpg];
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int b = 0; b < bpg; b++) cw[j][b] = nx[j][b];
      if (g + 1 < groups) {
#pragma unroll
        for (int j = 0; j < NPW; j++)
#pragma unroll
          for (int b = 0; b < bpg; b++) {
            const size_t o = (size_t)((g + 1) * bpg + b) * WAVE32;
            nx[j][b] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          }
      }
#endif
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
#if R4D_W4A8_PF
          wv[j] = cw[j][b];
#else
          const size_t o = (size_t)(g * bpg + b) * WAVE32;
#if R4D_W4A8_ABL & 4
          (void)o;
          wv[j] = (uint4v)(0x21212121u);
#else
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
#endif
#endif
        }
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int u = b * 4 + s;
          v2i afrag[MT];
#pragma unroll
#if R4D_W4A8_ABL & 2
          for (int i = 0; i < MT; i++) { afrag[i][0] = 0x01010101; afrag[i][1] = 0x01010101; }
#elif R4D_W4A8_APERM
          for (int i = 0; i < MT; i++) afrag[i] = fragA8(pa[i] + AOFF(g), u);
#else
          for (int i = 0; i < MT; i++) afrag[i] = fragA8row(pa[i] + AOFF(g), u);
#endif
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v2i bfrag = dequant8(wv[j][s]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = (u == 0) ? mmai<!ASYM>(afrag[i], bfrag, (v8i)(0))
                                     : mmai<!ASYM>(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++) {
            float t = (float)g_acc[j][i][e];
            if (ASYM) t -= z16[j] * asum[i][e];
            acc[j][i][e] = fmaf(t, sc[j], acc[j][i][e]);
          }
#if !R4D_W4A8_STAGGER
#pragma unroll
#if R4D_W4A8_APERM
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W4A8_GROUP / 8;
#else
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W4A8_GROUP / 4;
#endif
#endif
    }
  }

  // Split-K reduction and store, one row tile at a time.  The per-row activation scale is applied
  // here and only here: it is constant over n and over the group index, so it never entered the
  // k loop.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;

  // AT SK == 1 THERE IS NOTHING TO REDUCE: lane already holds the finished value.
  if (SK == 1) {
    if (n0 >= N) return;
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
          if (m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(acc[j][i][e] * Ascale[m] * 0.0625f);
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
      for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      // 0.0625 is the 1/16 the unpack introduced by putting the nibble in the byte's high half.
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * 0.0625f);
    }
  }
}

static void r4d_gemm_w4a8_impl(long a, long ascale, long wq, long ws, long asum, long c,
                               int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                               bool asym, long stream) {
  if (M < 1 || M > R4D_GEMM_W4A8_MAX_M)
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: M out of range");
  if (N % 16) throw std::runtime_error("r4d gemm_w4a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W4A8_GROUP))
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: K must be divisible by SK*group");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w4a8_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w4a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w4a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Zp = (const unsigned*)ws;
  auto Up = (const float*)asum;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT, AS) hipLaunchKernelGGL((r4d_gemm_w4a8_nt_m64_kernel<MT, NP, TT, AS>), \
    grid, block, smem, st, Ap, Sp, Wp, Zp, Up, Cp, M, K, N, WV, SK)
#define LAUNCH_MB(NP, TT, AS) switch (MB) { \
    case 1: LAUNCH(1, NP, TT, AS); break; \
    case 2: LAUNCH(2, NP, TT, AS); break; \
    case 3: LAUNCH(3, NP, TT, AS); break; \
    case 4: LAUNCH(4, NP, TT, AS); break; \
    default: throw std::runtime_error("r4d gemm_w4a8_nt_m64: MB unsupported"); }
#define LAUNCH_NPW(TT, AS) switch (NPW) { \
    case 1: LAUNCH_MB(1, TT, AS); break; \
    case 2: LAUNCH_MB(2, TT, AS); break; \
    case 8: LAUNCH_MB(8, TT, AS); break; \
    default: LAUNCH_MB(4, TT, AS); break; }
  if (asym) {
    if (NT) { LAUNCH_NPW(true, true); } else { LAUNCH_NPW(false, true); }
  } else {
    if (NT) { LAUNCH_NPW(true, false); } else { LAUNCH_NPW(false, false); }
  }
#undef LAUNCH_NPW
#undef LAUNCH_MB
#undef LAUNCH
}

void r4d_gemm_w4a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream) {
  r4d_gemm_w4a8_impl(a, ascale, wq, ws, 0, c, M, K, N, WV, SK, MB, NPW, NT, false, stream);
}

// Asymmetric grid: the weight nibble is read UNSIGNED and the stored zero point is corrected out
// with a per-row per-group sum of the activation, which r4d_quant_act_i8_asum emits.  `asum` is
// [K/group][Mpad] float with Mpad = ceil(M/16)*16 -- group-major, because a lane's eight
// accumulator elements are eight CONTIGUOUS output rows at one group, and padded so that base is
// 32-byte aligned for a v4f load and the last tile stays in bounds.  Allocate it ZEROED.
void r4d_gemm_w4a8_asym_nt_m64(long a, long ascale, long wq, long ws, long asum, long c,
                               int M, int K, int N, int WV, int SK, int MB, int NPW, int NT,
                               long stream) {
  if (!asum) throw std::runtime_error("r4d gemm_w4a8_asym_nt_m64: asum is required");
#if R4D_GEMM_W4A8_WS16
  // The zero point is the only thing in the dword's high half, and WS16 is the build that does not
  // store it.  The two are mutually exclusive by construction, so say so instead of reading zero.
  throw std::runtime_error("r4d gemm_w4a8_asym_nt_m64: built with WS16, which has no zero point");
#endif
  r4d_gemm_w4a8_impl(a, ascale, wq, ws, asum, c, M, K, N, WV, SK, MB, NPW, NT, true, stream);
}

int r4d_gemm_w4a8_nt_m64_max_m(void) { return R4D_GEMM_W4A8_MAX_M; }
int r4d_gemm_w4a8_nt_m64_group(void) { return R4D_GEMM_W4A8_GROUP; }
int r4d_gemm_w4a8_nt_m64_aperm(void) { return R4D_W4A8_APERM; }
int r4d_gemm_w4a8_nt_m64_ws16(void) { return R4D_GEMM_W4A8_WS16; }

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
/* ------------------------------------------------------------------ int8-activation grids */

/* The asymmetric grid is a different WEIGHT FORMAT, not a different call: the nibble is read
 * UNSIGNED and the stored per-group zero is corrected out with the activation's per-group row
 * sums. So it is selected by `dtype`, which is what the weight file was written for, and never by
 * whether a_sum happens to be present -- guessing from an operand would make a caller that forgot
 * to pass the sums silently compute the symmetric answer against asymmetric bytes. */

extern "C" int r4d_rad_gemm_w4a8_m64(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (g.N % 16) return R4D_NO(RAD_E_SHAPE);
    if (g.M > r4d_gemm_w4a8_nt_m64_max_m()) return R4D_NO(RAD_E_SHAPE);
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    const int group = r4d_gemm_w4a8_nt_m64_group();
    Cfg c;
    rc = cfg_for(a, g.K, g.M, group, BR_NPW, &c);
    if (rc != RAD_OK) return rc;

    if (is_asym(a)) {
        if (!g.asum) return R4D_NO(RAD_E_INVAL);      /* the entry point throws "asum is required" */
        R4D_TRY("gemm_w4a8_asym_nt_m64",
            r4d_gemm_w4a8_asym_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                      r4d_addr(g.bscale), r4d_addr(g.asum), r4d_addr(g.y),
                                      (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw,
                                      c.nt, (long)(uintptr_t)s));
        return RAD_OK;
    }
    R4D_TRY("gemm_w4a8_nt_m64",
        r4d_gemm_w4a8_nt_m64(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                             r4d_addr(g.bscale), r4d_addr(g.y),
                             (int)g.M, (int)g.K, (int)g.N, c.wv, c.sk, c.mb, c.npw, c.nt,
                             (long)(uintptr_t)s));
    return RAD_OK;
}

