// r4d_gemm_w4a8_tiled.hip - r4d_gemm_w4a8_tiled: the W4A8 GEMM for PREFILL batches, where the skinny
// kernel is structurally wrong.
//
// WHY.  r4d_gemm_w4a8_nt_m64 gives a block at most 64 rows, so at M=4096 there are 64 M-blocks and
// each re-reads the whole weight slice: 22.3 MB x 64 = 1.43 GB per call for gate_proj, which is why
// it holds a FLAT ~105 TF/s at every M and loses to a plain bf16 matmul.  The bar is the block-fp8
// prefill GEMM, which runs at ~204 TF/s on the same shapes; without a tiled W4 kernel both W4
// routes -- the skinny one and dequantise-to-bf16 -- are around twice that time, so prefill
// regresses badly.  Nothing slower than the bar ships.
//
// THE SHAPE FOLLOWS FROM TRAFFIC.  Block tile BM x BN reads (M/BM) copies of the weight and (N/BN)
// copies of the activation.  For gate_proj at M=4096, against ~600 GB/s of DRAM and ~1049 from the
// 64 MB MALL (both operands fit, so re-reads are cached):
//
//     64 x 128 (skinny)   4352 blocks   2.85 GB   2.7 ms
//     128 x 256           1088 blocks   1.43 GB   1.4 ms
//     256 x 256            544 blocks   1.07 GB   1.0 ms
//
// 256x256 wants TM=TN=4 per wave, and that does not fit: the group loop needs the int32 accumulator
// AND the float one live at once (16 + 16 vector registers of 8 = 256 VGPRs).  TM*TN <= 8 is the
// budget, so this is 128 x 256 -- 1.4 ms projected, still under the 1786 us bar.
//
// BK IS HALF THE GROUP, AND DOUBLE-BUFFERED.  BK = the group with a SINGLE buffer makes every
// iteration load -> barrier -> compute -> barrier, with 40 global-load latencies fully exposed: that
// form is SLOWER than the skinny kernel while moving HALF the traffic.  The loop is latency-bound,
// not bandwidth-bound, which is what a traffic argument alone cannot tell you.
//
// Double buffering needs the working set to stay under 32 KB so two workgroups still fit a CU; at
// BK = the group that would be 64 KB and cost the occupancy.  At BK = 64 it is 8 KB of A and 8 KB of
// W per buffer, 32 KB in total.  The scale group then spans two iterations, which costs nothing --
// the int32 accumulator lives in registers and a barrier does not disturb it.
//
// (Padding A's row stride to dodge LDS bank conflicts is a LOSS here: it helps a little at M = 512
// and costs far more at M = 4096, because 16 KB + 16 KB is exactly 32 KB and one extra dword per
// row pushes a CU from two workgroups to one.  The swizzle below gets the same effect for free.)
//
// LAYOUTS ARE THE SKINNY KERNEL'S, UNCHANGED.  The packed weight is already in WMMA fragment order --
// lane L of a (tile, k-block) owns dwords [4L, 4L+3] of a 128-dword run -- so staging is a contiguous
// copy and the LDS read repeats the global pattern.  The activation is pre-shuffled inside each k
// step by the quantiser, so a lane's eight bytes are contiguous there too.  One weight, one
// activation, both kernels.
#include "r4d.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <mutex>
#include <stdexcept>

#define R4D_TILED_KPB 64

typedef int      v2i_t __attribute__((ext_vector_type(2)));
typedef int      v8i_t __attribute__((ext_vector_type(8)));
typedef float    v8f_t __attribute__((ext_vector_type(8)));
typedef unsigned u4v_t __attribute__((ext_vector_type(4)));
typedef unsigned u2v_t __attribute__((ext_vector_type(2)));   // 8 bytes = four bf16
typedef float    v4f_t __attribute__((ext_vector_type(4)));

// BSIGNED=false reads the unpacked byte as 16*nibble (0..240) instead of 16*twos(nibble): the
// asymmetric grid, whose per-group zero is then corrected out with a row-sum of the activation.
template <bool BSIGNED>
__device__ __forceinline__ v8i_t tmmai(v2i_t a, v2i_t b, v8i_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, BSIGNED, b, c, false);
}

__device__ __forceinline__ v2i_t tdequant8(unsigned w) {
  v2i_t r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

__device__ __forceinline__ unsigned tpack2bf(float a, float b) {
  unsigned ua = __float_as_uint(a), ub = __float_as_uint(b);
  ua += 0x7FFFu + ((ua >> 16) & 1u);          // round to nearest even, no NaN branch
  ub += 0x7FFFu + ((ub >> 16) & 1u);
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);   // {ub.hi16, ua.hi16}
}

// WM x WN waves, each owning TM x TN WMMA tiles.  BM = WM*TM*16, BN = WN*TN*16, BK = the scale group.
//
// BIAS KILLS THE int32 -> float CONVERSION IN THE GROUP EPILOGUE.  Every group closes with
// `acc += (float)g * s` over TM*TN*8 = 64 accumulator elements, which is 128 VALU against 64 WMMA
// instructions -- 2 VALU per WMMA, the single largest overhead in the loop.  Half of it is
// `v_cvt_f32_i32`, and that half can be deleted outright: for |x| < 2^22,
//
//     float(x) == __uint_as_float(0x4B400000 + x) - 12582912.0f
//
// because 0x4B400000 is 1.5*2^23, whose mantissa field sits exactly at 2^22, so adding x lands x in
// the mantissa unrounded.  The group's int32 sum is at most 128 * 127 * 128 = 2,080,768, half the
// budget.  BIAS=1 adds the constant with `v_add_nc_u32` (which, unlike v_cvt, is VOPD-eligible and
// can dual-issue with the fmac next to it); BIAS=2 seeds the WMMA's C operand with it instead, so
// the accumulator comes OUT biased and the conversion costs nothing at all -- at the price of eight
// VGPRs held for the constant vector, and 192 is exactly the budget for two workgroups per CU.
// The -12582912*s per group is folded into a per-column running sum and subtracted once at the end.
//
// SKT: A BALANCED LAST ROUND.  The part runs 32 workgroups at a time (one 512-thread workgroup per
// WGP), so a grid of T tiles takes ceil(T/32) rounds and the last one runs T mod 32 workgroups on a
// 32-slot machine.  `gate_up_proj` is N = 17408 = 2^10 x 17, which is 136 tiles at BN = 128: four
// full rounds and a round of eight, and that round standalone costs well over twice what the same
// work does spread over 32 workgroups at a quarter of K each.
//
// SKT != 0 splits ONLY the tail tiles along K.  The first `nfull` tiles are ordinary data-parallel
// blocks that store straight to C; the rest are `r` tiles x `s` K-pieces, which write fp32 partials
// to a workspace and let the LAST piece to arrive reduce them.  "Last to arrive" is an atomic
// counter, so there is no spin and no assumption about dispatch order; the reduction reads the s
// slots in INDEX order, so the sum is deterministic run to run even though the arrival order is not.
// It is not bit-identical to the unsplit kernel -- the fp32 adds re-associate at the piece boundary
// -- and it is not meant to be; it is a different, equally valid summation order.  It agrees with
// the unsplit kernel to at most one bf16 ULP across every shape, batch and tile it was checked on,
// and against an fp32 reference it is equal or very slightly better.
//
// AND IT DOES NOT PAY at any batch the decode dispatch reaches.  The tail round is cheaper split,
// but a full grid run whole is cheaper than running the two halves back to back: the part is
// ALREADY overlapping most of that tail with the round in front of it, and the workspace round trip
// plus the reduction's arrival barrier cost more than the remainder.  It is a loss at the narrow
// batches and a wash to a small loss everywhere the dispatch table picks.  The two instantiations
// are kept because the path is correct and cheap to re-check, NOT because anything should call it.
template <int WM, int WN, int TM, int TN, int BIAS, int ABL = 0, int OPT = 0, int WPE = 8,
          int ASYM = 0, int UNR = 1, int SKT = 0>
__global__ __launch_bounds__(WM * WN * 32, WPE) void r4d_gemm_w4a8_tiled_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    const float* __restrict__ Asum, __hip_bfloat16* __restrict__ C, int M, int K, int N,
    float* __restrict__ WSP, unsigned* __restrict__ CNT, int nfull, int nrag, int nspl) {
  constexpr int BK = 64;                            // half a scale group
  constexpr int GRP = R4D_GEMM_W4A8_GROUP;
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  constexpr int NTILE = BN / 16;
  constexpr int AW = BM * BK / 4;                   // A dwords per buffer
  // OPT BIT 256: UNPACK THE WEIGHT ONCE, AT STAGING, INSTEAD OF WM TIMES ON READ.
  // The fragment read gives wave (wm, wn) the n-tiles of `wn`, so every wave sharing a `wn` --
  // WM of them -- unpacks the SAME dwords.  Counted per block per k step: the read path does
  // WM*WN waves x 32 lanes x TN*4 dwords x 3 VALU, the staging path does WW dwords once, and the
  // ratio is exactly WM.  With the 16-row swizzle in place the DCE-proof ablation puts the unpack
  // at about a quarter of this kernel at a decode M, and the W LDS read at ~0 (deleting it measures
  // the same as deleting the unpack alone), so trading half the unpack for twice the W LDS traffic
  // is the right side of that trade.  It costs LDS: the W buffer holds bytes rather than nibbles.
  constexpr bool WU = (OPT & 256) != 0;
  // OPT BIT 512: THE iu4 PATH -- no weight dequant at all.
  // `v_wmma_i32_16x16x32_iu4` eats PACKED nibbles, and the DCE-proof ablation puts the unpack at
  // about a quarter of this kernel at a decode M, with a no-unpack build landing at 1.09x the
  // one-pass memory floor.  The activation is the SAME int8, split exactly as
  // `a = 16*(a>>4) + (a&15)` with the high half read signed and the low half unsigned, so two iu4
  // replace the two iu8 that covered the same K.  Integer accumulation makes the re-association
  // exact, and the split holds over all 256 x 16 (a, q) pairs on hardware -- this is bit-identical
  // arithmetic, not an approximation.
  //
  // THE WEIGHT IS NOT RE-PERMUTED.  A GEMM is invariant under a permutation of k applied to BOTH
  // operands, and the instruction pairs A slot p with B slot p exclusively -- a perfect 16x16
  // diagonal on hardware.  So the packed artifact stays byte-identical
  // and only the ACTIVATION is laid out to name the k the weight packing already names:
  //   byte c*32 + P*16 + h*8 + d*4 + i/2, nibble i&1  <-  plane P of a[m][64*blk + c*32 + d*16 +
  //   KIDX[h][i]],  KIDX[0] = 0,8,1,9,2,10,3,11   KIDX[1] = 4,12,5,13,6,14,7,15
  // which is the same bytes and the same `ds_load_b64` count as the int8 form.
  constexpr bool IU4 = (OPT & 512) != 0;
  // OPT BIT 65536: NON-TEMPORAL WEIGHT LOADS.
  // At M=64 this kernel is 20-27% over its weight roofline where at M=8 it is 5%, and the inner
  // loop is not the reason: per 512 bytes streamed at the full 634 GB/s this part
  // gives away 32 iu8 matrix instructions, 96+ VALU and 96+ LDS reads for FREE, and the shipped
  // tile spends 20, 12 and 20 of them.  What scales with M is the ACTIVATION:
  // every column block stages the whole M x K of A, so gate_up's 136 blocks read 44.6 MB of A
  // against 47.35 MB of weight, and A is 328 KB -- it should live in L2 and be nearly free.  The
  // weight stream is what evicts it: 636 MB walked once, never re-read, filling every level on the
  // way past.  A non-temporal hint on the weight says so, and leaves A where it was.
  constexpr bool WNT = (OPT & 65536) != 0;
  // OPT BIT 131072: THE WEIGHT NEVER GOES THROUGH LDS.
  // The staged weight buys sharing between the WM waves that hold the same `wn`, and every tile a
  // decode step picks has WM = 1 -- so it buys nothing and costs a `ds_store_b128` per thread per
  // k step, half the LDS budget, and a barrier that has to wait for the weight's global load as
  // well as the activation's.  The fragment a wave wants is already one contiguous 512-byte block
  // read `lane*4` dwords at a time, which is EXACTLY the shape the staging loads it in, so the
  // register the prefetch put it in can be read directly.  Double buffering moves from LDS to a
  // second register set, indexed by the same compile-time `it & 1` the unrolled k loop already has.
  constexpr bool WNOLDS = (OPT & 131072) != 0;
  // OPT BIT 262144: FETCH THE GROUP SCALE A WHOLE GROUP EARLIER STILL.
  // Of everything inside this loop, the group scale is the ONLY item that prices above the control
  // spread at the tiles a decode step picks: replacing the load with a constant is worth several
  // percent on `gate_up`, where the unpack, the weight LDS read, the activation LDS read and the
  // group fold are each inside it.  Its bandwidth share is 5.9%, so about a quarter is latency
  // the existing one-group prefetch does not cover -- the load lands at the start of the group it
  // is used at the end of, and every wave in every workgroup reaches that boundary together.
  // Rotate two registers instead: fetch group g+1 at the start of group g.
  constexpr bool SCPF2 = (OPT & 262144) != 0;
  // OPT BIT 524288: THE SCALE TABLE IS f16, NOT A DWORD.
  // The dword per (column, group) carries the f16 scale in its low half and the f16 of -(1024+zero)
  // in its high half, and the SYMMETRIC path -- which is what ships -- reads `raw & 0xFFFF` and
  // never touches the other two bytes.  They are 5.9% of everything `gate_up` streams and the only
  // item in this loop that prices above the control spread, and pricing it as latency is wrong:
  // a prefetch a whole group deeper buys nothing where deleting the load buys several percent.  So
  // halve the table.  The arithmetic is untouched -- the same f16, the same fold, same output --
  // and the asymmetric path keeps the dword because it needs the other half.
  constexpr bool SC16 = (OPT & 524288) != 0;
  static_assert(!SC16 || !ASYM, "the asymmetric path needs the zero in the scale dword's high half");

  // AN ABLATION THAT ALSO DELETES THE STAGING IS TWO ABLATIONS.  The variants below 10 are guarded
  // by `ABL == 0` at the prefetch and the commit, so every one of them ALSO runs with no global
  // loads and no double buffer -- fine for ranking two variants against each other, useless for
  // pricing one item against the shipped kernel.  ABL >= 10 repeats one of those variants with the
  // staging left in, which is what a prefill cost breakdown needs; the variants below 10 keep their
  // own meanings.
  constexpr int ABX = ABL == 10 ? 5 : ABL == 11 ? 6 : ABL == 12 ? 7 : ABL == 13 ? 3
                    : ABL == 14 ? 4 : ABL == 15 ? 9 : ABL == 16 ? 2 : ABL == 17 ? 8 : ABL;
  constexpr bool TWOACC = IU4 || ABX == 8;
  // THE MAGIC BIAS AND iu4 DO NOT MIX, and the reason is worth stating because the arithmetic
  // itself is exact.  BIAS=2 seeds the int32 accumulator with 0x4B400000 so the int->float
  // conversion is a reinterpretation; the float fold then rounds `(12582912 + G) * s` once.  The
  // iu4 path multiplies the RAW nibble rather than 16*q, so its integer G is SIXTEEN TIMES smaller
  // for the same answer -- the bias is unchanged but the signal under it is not, and the rounding
  // that was 1/16 ULP of the signal becomes one.  The result is 7.22e-3 against the iu8 path's
  // 2.37e-3, three times the 4-bit quantisation error, which is not a rounding difference worth
  // arguing about.
  //
  // Restoring it would mean holding `bias + 16*G` in the accumulator, which costs a second shift
  // per element per group.  Plain `v_cvt_f32_i32` (BIAS=0) costs a fraction of a percent against
  // the biased build here and is MORE accurate than the shipped path (2.33e-3 against 2.37e-3),
  // because the product it rounds has no bias in it at all.  So iu4 simply does not take the bias.
  static_assert(!IU4 || BIAS == 0, "the iu4 path must be built with BIAS = 0");
  //
  // REJECTED, and the arithmetic says why in one line.  Per scale group this path
  //   SAVES  the unpack, 3 VALU per weight dword    = 24*TN
  //   COSTS  the hi/lo combine, one per element     =  8*TM*TN
  //   COSTS  the cvt that the magic bias deletes    =  8*TM*TN
  // so the net is 8*TN*(2*TM - 3): a win only at TM = 1, which means BM = 32 and no batch a serve
  // makes wants a row tile that small.  It also costs 30 VGPRs (85 -> 115 at BM=96/BN=128), which
  // crosses the 96-VGPR step from 16 waves per SIMD to 12, and at BM=128 it spills 65 bytes.
  // Against the same tile on iu8 it is bit-identical in output and ranges from a small win to a
  // large loss across the tiles these batches pick, with BM=128 more than twice the time.  The
  // ABL 8 variant looks better than the real path only because it keeps the magic bias, which the
  // real path cannot have.  One instantiation is kept because it is correct and bit-identical, and
  // the only thing that would change the verdict is a tile with TM = 1.
  constexpr int WW = NTILE * 128 * (WU ? 2 : 1);    // W dwords per buffer (BK == KPB, so one run)

  __shared__ unsigned lds_a[2][AW];
  __shared__ unsigned lds_w[2][WNOLDS ? 4 : WW];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  int tilex = blockIdx.x, kbeg = 0, kend = K, slot = -1;
  if (SKT && (int)blockIdx.x >= nfull) {
    const int jj = (int)blockIdx.x - nfull, kper = K / nspl;
    tilex = nfull + jj / nspl;
    kbeg = (jj % nspl) * kper;
    kend = kbeg + kper;
    slot = (int)blockIdx.y * (nrag * nspl) + jj;
  }
  const int m0 = blockIdx.y * BM, n0 = tilex * BN;
  const int kpb = K / R4D_TILED_KPB, nsz = K / GRP;
  constexpr int NT_ = WM * WN * 32;

  v8f_t acc[TN][TM];
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++) acc[j][i] = (v8f_t)(0.f);
  v8i_t g[TN][TM];
  // ABL 8 prices the iu4 ROUTE.  `v_wmma_i32_16x16x32_iu4` consumes PACKED nibbles, so a kernel
  // built on it has no weight unpack at all -- and the DCE-proof ablation puts the unpack at 24.7%
  // of this kernel at M=72, with a no-unpack build landing at 1.09x the one-pass memory floor.
  // The route is `A_i8 = 16*A_hi + A_lo` with A_hi signed and A_lo unsigned int4, which is exact,
  // so ONE iu4 instruction covers K=32 and two of them (hi and lo, sharing the B fragment) replace
  // the two iu8 instructions that covered the same K.  Instruction count is unchanged; what goes
  // away is the unpack, and what it costs is a second int32 accumulator per output tile.
  // This variant computes the WRONG ANSWER on purpose -- it feeds whatever bits are to hand,
  // because the real thing needs an offline weight re-permutation and a 4-bit-plane activation
  // format. It exists to say whether that pipeline is worth building.
  v8i_t g2[TWOACC ? TN : 1][TWOACC ? TM : 1];
  float csum[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) csum[j] = 0.f;
  unsigned scv[TN], scn[SCPF2 ? TN : 1];
  float asp[ASYM ? TM : 1][8];

  // XOR SWIZZLE instead of padding.  A row is BK/4 = 16 dwords and LDS has 32 banks, so a half-wave's
  // sixteen lanes -- differing only in the row -- would collide.  Padding by one dword fixes that and
  // costs more than it saves: A+W is exactly 32 KB and one extra dword per row drops a CU from two
  // workgroups to one, a small gain at M=512 and a large loss at M=4096.  The mask must stay INSIDE the row:
  // (row & 15) << 1 reaches 30, walks into the next row and gives a 0.386 relative error.
// With 128-bit staging (OPT bit 3) the mask has to permute whole 4-dword BLOCKS, or a `ds_store_b128`
// of dwords 0-3 lands at swizzled position 2 and writes {2,3,4,5} instead of {2,3,0,1}: silently
// wrong data (relative error 0.79) and a 16-byte store at an 8-byte-aligned address.  (r&3)<<2 is
// 0/4/8/12, which still separates rows two apart -- the LDS bank period at a 16-dword row stride --
// and still lands inside the row, since 15 ^ 12 = 3.
// OPT BIT 128: A SWIZZLE THAT SEPARATES SIXTEEN ROWS, NOT EIGHT.
// The fragment read has sixteen lanes differing ONLY in the row -- `row = ... + (lane & 15)` -- at a
// fixed column, and the row stride is BK/4 = 16 dwords.  Bank = (r*16 + (c ^ mask)) % 32, so the
// row contributes only its low bit and the mask has to separate the other four.  `(r & 7) << 1`
// separates eight, so rows r and r+8 land on the same bank: a 2-WAY CONFLICT on every A read.
// `((r >> 1) & 7) << 1` gives the eight EVEN rows eight distinct masks and the eight odd rows the
// same eight in the upper bank half, which is conflict-free at the read.  It stays inside the row
// (both operands are 4-bit, so the XOR is) and stays EVEN, which the `ds_load_b64` alignment needs.
// The store side is unaffected either way: there sixteen consecutive lanes share a row and differ
// in c, so they were already spread across all sixteen banks of the row.
//
// This matters because the A LDS read is the LARGEST single item in this inner loop -- the
// DCE-proof ablation puts it at about a third of the kernel at the tiles a decode step picks.
#define ASW(r, c)                                                                      \
  ((r) * (BK / 4) + ((c) ^ ((OPT & 8) ? (((OPT & 128) ? ((r) >> 1) : (r)) & 3) << 2     \
                                      : (((OPT & 128) ? ((r) >> 1) : (r)) & 7) << 1)))

  // A PREFETCH THAT WRITES LDS DIRECTLY DOES NOT OVERLAP, and it is worth more than a factor of two.
  // The LDS store waits on the global load and sits before the compute in program order, so the
  // compute waits too.  Load into REGISTERS, compute off the other buffer, then commit, so the wait
  // lands after the WMMA work.  Same buffers, same barriers, same traffic; only the s_waitcnt moves.
  // A RAGGED A STAGING, so BM does not have to be a multiple of the workgroup width.
  // AW = BM*BK/4 dwords, so 512 threads need BM % 32 == 0 and the tile ladder had to skip 80 and
  // 112 -- exactly the sizes that fit a decode M without rounding it up.  One predicate on the
  // last partial pass costs nothing when it divides (the guard folds away) and unlocks the rest.
  constexpr int APT = (AW + NT_ - 1) / NT_, WPT = WW / NT_;
  constexpr bool APAD = (AW % NT_) != 0;
  unsigned ra[APT > 0 ? APT : 1], rw[WPT > 0 ? WPT : 1];
  // 128-BIT STAGING (OPT bit 3).  Each thread otherwise moves one dword per array per iteration:
  // four global loads, four LDS stores and four addresses for 32 bytes.  A row is BK/4 = 16 dwords
  // and the block is 512 threads, so four threads cover a row exactly -- one `global_load_b128` and
  // one `ds_store_b128` each instead of four of each.  The wave touches the same segments either
  // way: 32 lanes x 16 B over 8 rows is the same 64-byte runs as 32 lanes x 4 B over 2 rows.
  // A 128-BIT STAGING PATH THAT DOES NOT TILE LOADS NOTHING AT ALL, SILENTLY.
  // Variant 9 (<2,8,4,1>, BM 128 BN 128) is 512 threads against WW = NTILE*128 = 1024 weight
  // dwords, so (WW/4)/NT_ is 0, the `OPT & 16` loop has an empty trip count, the dword path is
  // switched off by the same bit, and the kernel multiplies whatever was in LDS -- it returns a
  // 1.6 relative error rather than raising.  Choose the width from whether it actually divides,
  // not from the OPT bit alone, so a new tile shape degrades to dword staging instead of breaking.
  constexpr int WWP = NTILE * 128;                  // packed W dwords, whatever the store width
  constexpr int WPTU = WU ? (WWP / NT_ > 0 ? WWP / NT_ : 1) : 1;
  unsigned rwu[WPTU];
  // OPT BIT 32768: LET THE 128-BIT A STAGING BE RAGGED, the way OPT-less dword staging already is.
  // A128 needs (AW/4) % NT_ == 0, and AW/4 = BM*4, so at 512 threads only BM % 128 == 0 qualifies:
  // BM = 160 and 192 silently fall back to SIX global_load_b32 + six ds_store_b32 per thread per k
  // step instead of two b128, which is ~10 extra instructions per k step and is most of why the
  // tall tiles lose at prefill.  One predicate on the last pass fixes it.  Gated on its own bit so
  // that no shape the decode table already picked changes: the tiles where AW/4 < NT_ (BM <= 96 at
  // 512 threads, BM <= 80 at 256) keep A128 OFF exactly as before.
  constexpr bool A128R = (OPT & 32768) != 0;
  constexpr int APT4 = A128R ? ((AW / 4 + NT_ - 1) / NT_) : ((AW / 4) / NT_);
  constexpr int WPT4 = WU ? 0 : (WW / 4) / NT_;
  constexpr bool A4PAD = A128R && ((AW / 4) % NT_) != 0;
  constexpr bool A128 =
      (OPT & 8) && APT4 > 0 && (A128R ? (AW / 4) >= NT_ : (AW / 4) % NT_ == 0);
  constexpr bool W128 = !WU && (OPT & 16) && WPT4 > 0 && (WW / 4) % NT_ == 0;
  u4v_t ra4[APT4 > 0 ? APT4 : 1], rw4[WPT4 > 0 ? WPT4 : 1];
  u4v_t rwg[WNOLDS ? 2 : 1][WNOLDS ? TN : 1];       // the weight double buffer, in registers

  // OPT BIT 32: DO NOT STAGE ROWS PAST M.
  // The tiled kernel's cost at these batches is proportional to the PADDED row count, not to M and
  // not to the weight traffic: on `gate_up_proj` the BM=96 tile costs exactly 96/64 of what the
  // BM=64 tile does at the same M, while the weight read is identical.  What scales
  // with BM is the activation staging -- APT = (BM*BK/4)/threads dwords per thread per k step,
  // 2 for BM=64 at 512 threads and 3 for BM=96 -- and the 256-thread variants, which triple it
  // again, cost 2-3x.
  //
  // At M=72 in a BM=96 tile, 24 of every 96 staged rows are `mm = M-1` clamped duplicates whose
  // accumulators the store epilogue throws away.  A quarter of the staging is for rows that do not
  // exist.  Skipping the global load AND the LDS store for them leaves stale LDS under those rows,
  // which is exactly as harmless as the duplicate was: the WMMA consumes it, and the epilogue
  // discards every m >= M.
  constexpr bool MSKIP = (OPT & 32) != 0;
  // ABL 9 PRICES TAKING THE ACTIVATION OUT OF LDS ENTIRELY.
  // The A fragment read is the largest single item in this loop -- the ABL 7 variant shows halving
  // its BYTES is worth close to a fifth of the kernel -- and it exists only because A arrives
  // row-major: a lane's eight bytes are one row at a fixed k, so sixteen lanes would each need
  // their own 8-byte segment and a direct global load would be sixteen-way scattered.  LDS is what
  // transposes it.  But the quantiser writes the activation, so it could write it FRAGMENT-MAJOR --
  // [row tile][k step][lane][8] -- and then the fragment is one coalesced `global_load_b64` per
  // (i, u) and there is no A staging, no `ds_store`, no `ds_load` and no A half of the LDS budget
  // at all.  This variant loads exactly that address, so the instruction mix and the traffic are
  // the real thing's; the ANSWER is wrong because the activation in memory is still row-major.
  constexpr bool ANOLDS = (ABX == 9);
  // OPT BIT 1024: A BARRIER THAT FENCES LDS ONLY.
  // `__syncthreads()` is a full-memory fence, and on gfx12 that means the barrier carries a
  // `global_inv` -- the ISA shows two of them per scale group here.  Everything this loop
  // synchronises is in LDS: the staging writes `lds_a`/`lds_w` and the fragment reads read them
  // back.  The global loads are into REGISTERS and are ordered by their own load counter, not by
  // the barrier.  So the release/acquire pair only has to cover the `local` address space, which
  // drops the cache invalidate and leaves the barrier itself.
  //
  // IT IS NEUTRAL, though the variant is not inert: `global_inv` really does go 2 -> 0 per scale
  // group, traded for two more `s_wait_dscnt`, at an identical 236 instructions.  The time
  // difference sits inside the run-to-run spread of a control variant that duplicates the baseline,
  // so the cache invalidate is free here.  The variant stays because it is correct and the
  // comparison needs it.
  auto ldsbar = [&]() {
    if (OPT & 1024) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
      __syncthreads();
    }
  };
  // One index, two containers: the dword table the asymmetric path needs and the f16 table the
  // symmetric path is the only reader of.  Written as a macro rather than a lambda because it is
  // read from inside the epilogue's `#pragma unroll` bodies where a capture would be re-formed.
#define R4D_WS(g_, j_)                                                                            \
  (SC16 ? (unsigned)((const unsigned short*)Ws)[(unsigned)(((n0 / 16 + wn * TN + (j_)) * nsz +    \
                                                            (g_)) * 16 + (lane & 15))]           \
        : Ws[(unsigned)(((n0 / 16 + wn * TN + (j_)) * nsz + (g_)) * 16 + (lane & 15))])
  auto issue = [&](int kk, int bufn) {
    if constexpr (WNOLDS) {
#pragma unroll
      for (int j = 0; j < TN; j++) {
        const u4v_t* wp4 = (const u4v_t*)(Wq + (unsigned)(((n0 / 16 + wn * TN + j) * kpb +
                                                          kk / R4D_TILED_KPB) * 128 + lane * 4));
        rwg[bufn][j] = WNT ? __builtin_nontemporal_load(wp4) : *wp4;
      }
    }
    if (A128 && !ANOLDS) {
#pragma unroll
      for (int x = 0; x < APT4; x++) {
        const int t = tid + x * NT_, r = t / (BK / 16), c4 = t % (BK / 16);
        if (A4PAD && t >= AW / 4) continue;
        if (MSKIP && m0 + r >= M) continue;
        const int mm = m0 + r < M ? m0 + r : M - 1;
        ra4[x] = *(const u4v_t*)(A + (unsigned)(mm * K + kk + c4 * 16));
      }
    }
    if (W128 && !WNOLDS) {
#pragma unroll
      for (int x = 0; x < WPT4; x++) {
        const int t = tid + x * NT_, tl = t / 32, off4 = t % 32;
        const u4v_t* wp4 = (const u4v_t*)(Wq + (unsigned)(((n0 / 16 + tl) * kpb +
                                                          kk / R4D_TILED_KPB) * 128 + off4 * 4));
        rw4[x] = WNT ? __builtin_nontemporal_load(wp4) : *wp4;
      }
    }
    if (WU && !WNOLDS) {
      // One dword of nibbles per thread, kept packed in a register until commit; the unpack itself
      // rides on the store so the global load is unchanged.
#pragma unroll
      for (int x = 0; x < WPTU; x++) {
        const int t = tid + x * NT_, tl = t / 128, off = t % 128;
        const unsigned* wpu = Wq + (unsigned)(((n0 / 16 + tl) * kpb + kk / R4D_TILED_KPB) * 128
                                              + off);
        rwu[x] = WNT ? __builtin_nontemporal_load(wpu) : *wpu;
      }
    }
#pragma unroll
    for (int x = 0; x < APT && !A128 && !ANOLDS; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      if (MSKIP && m0 + r >= M) continue;
      const int mm = m0 + r < M ? m0 + r : M - 1;
      ra[x] = (OPT & 2) ? *(const unsigned*)(A + (unsigned)(mm * K + kk + c * 4))
                        : *(const unsigned*)(A + (size_t)mm * K + kk + c * 4);
    }
#pragma unroll
    for (int x = 0; x < WPT && !W128 && !WU && !WNOLDS; x++) {
      const int t = tid + x * NT_, tl = t / 128, off = t % 128;
      const unsigned* wp = (OPT & 2)
              ? Wq + (unsigned)(((n0 / 16 + tl) * kpb + kk / R4D_TILED_KPB) * 128 + off)
              : Wq + ((size_t)((n0 / 16 + tl) * kpb + kk / R4D_TILED_KPB) * 32) * 4 + off;
      rw[x] = WNT ? __builtin_nontemporal_load(wp) : *wp;
    }
  };
  auto commit = [&](int buf) {
    if (A128 && !ANOLDS) {
#pragma unroll
      for (int x = 0; x < APT4; x++) {
        const int t = tid + x * NT_, r = t / (BK / 16), c4 = t % (BK / 16);
        if (A4PAD && t >= AW / 4) continue;
        if (MSKIP && m0 + r >= M) continue;
        *(u4v_t*)(lds_a[buf] + ASW(r, c4 * 4)) = ra4[x];
      }
    }
    if (W128 && !WNOLDS) {
#pragma unroll
      for (int x = 0; x < WPT4; x++) *(u4v_t*)(lds_w[buf] + (tid + x * NT_) * 4) = rw4[x];
    }
    if (WU && !WNOLDS) {
#pragma unroll
      for (int x = 0; x < WPTU; x++) {
        const int t = tid + x * NT_, tl = t / 128, off = t % 128;
        const int wl = off / 4, wu_ = off % 4;      // the read's lane and k step
        const v2i_t d = tdequant8(rwu[x]);
        // [tile][lane][u][2], so a wave's fragment for one k step is one ds_load_b64 and its four
        // k steps are 32 contiguous bytes -- two b128 where the packed form needed one.
        *(v2i_t*)(lds_w[buf] + (size_t)tl * 256 + wl * 8 + wu_ * 2) = d;
      }
    }
#pragma unroll
    for (int x = 0; x < APT && !A128 && !ANOLDS; x++) {
      const int t = tid + x * NT_, r = t / (BK / 4), c = t % (BK / 4);
      if (APAD && t >= AW) continue;
      if (MSKIP && m0 + r >= M) continue;
      lds_a[buf][ASW(r, c)] = ra[x];
    }
#pragma unroll
    for (int x = 0; x < WPT && !W128 && !WU && !WNOLDS; x++) lds_w[buf][tid + x * NT_] = rw[x];
  };
  // Neither staging width can move the buffer if it does not divide the block, and the loops above
  // would then silently move nothing. Both paths are compile-time constants, so say so at compile
  // time rather than returning a wrong answer.
  static_assert(ANOLDS || A128 || APT > 0, "no A staging path tiles this block shape");
  static_assert(WNOLDS || WU || W128 || (WPT > 0 && WW % NT_ == 0),
                "no W staging path tiles this block shape");
  // Every variant that reads `lds_w` with its own addressing -- the unpack-at-staging path, the iu4
  // route and the two DCE-proof ablations -- would read a four-dword stub under WNOLDS.  None of
  // them is a shipping path, so refuse the combination rather than measure a wrong answer.
  static_assert(!WNOLDS || (ABL == 0 && !WU && !IU4),
                "WNOLDS does not compose with the weight-LDS ablations");
  static_assert(!WU || WWP % NT_ == 0, "unpacked W staging does not tile this block shape");

  if constexpr (SCPF2) {
#pragma unroll
    for (int j = 0; j < TN; j++)
      scn[j] = R4D_WS(kbeg / GRP, j);
  }
  issue(kbeg, 0);
  commit(0);          // a BUFFER index, not a k offset -- the two only coincide at kbeg = 0
  ldsbar();

  // UNROLL THE K LOOP OVER THE GROUP.  At BK = 64 and a 128-wide scale group the group boundary is
  // a DYNAMIC branch -- `(k0 % GRP) == 0` alternates -- and the epilogue sits inside it, which stops
  // the scheduler from interleaving it with the WMMA work on either side.  Unrolling by GRP/BK makes
  // both conditions compile-time constants per copy.  Worth several percent, and it is what brings
  // the kernel from a few points over the block-fp8 bar to PARITY.  The tell that the dynamic
  // branch is there at all: a group-64 build, where GRP == BK makes the conditions constant for
  // free, runs the SAME as group 128 despite doing twice the epilogue arithmetic.
  for (int kg = kbeg, it = 0; kg < kend; kg += BK * UNR) {
#pragma unroll
    for (int h = 0; h < UNR; h++, it++) {
    const int k0 = kg + h * BK;
    const int buf = it & 1;
    if ((ABL == 0 || ABL >= 10) && k0 + BK < kend) issue(k0 + BK, buf ^ 1);  // loads in flight across the WMMA work below

    constexpr int AGRP = ABX == 3 ? GRP * 8 : GRP;  // ABL 3 prices the group epilogue at 1/8
    const bool first_half = (k0 % AGRP) == 0;
    // THE GROUP SCALE IS A GLOBAL LOAD THAT NOTHING HIDES.  Every wave in every workgroup reaches
    // the group boundary on the same k0, so when the epilogue loads Ws right where it uses it, all
    // of them stall on the same latency at the same moment and there is no other wave to run.  That
    // is why the epilogue costs about four times what issuing its 5120 VALU would: most of it is
    // the wait.  Fetch it at the START of the group instead -- one whole 64-k iteration of WMMA
    // work, plus the second iteration's, stands between the load and its use.
    // Keep the RAW dword.  Converting it here consumes the load here, and the compiler puts the
    // s_waitcnt in front of the first USE: calling __half2float() at the prefetch site lands the
    // wait immediately after the load, ahead of every WMMA -- a prefetch that prefetches nothing.
    // Convert at the group close instead.
    if ((OPT & 1) && (k0 % GRP) == 0) {
      if constexpr (SCPF2) {
        // The rotate has to come first: `scn` is the group this iteration will fold with, and the
        // load below is for the NEXT one.  The group index is clamped rather than branched -- the
        // value fetched past the last group is never folded, and a uniform branch here would sit
        // on the boundary every wave arrives at together.
#pragma unroll
        for (int j = 0; j < TN; j++) scv[j] = scn[j];
        const int gnx = (k0 / GRP + 1 < nsz) ? k0 / GRP + 1 : nsz - 1;
#pragma unroll
        for (int j = 0; j < TN; j++)
          scn[j] = R4D_WS(gnx, j);
      } else {
#pragma unroll
        for (int j = 0; j < TN; j++)
          scv[j] = R4D_WS(k0 / GRP, j);
      }
    }
    // The activation row-sums have exactly the scale's problem, and worse: every wave in every
    // workgroup hits the group boundary on the same k0, so eight `v4f` loads issued where they are
    // used stall the whole machine at once.  An ablation that keeps the correction's arithmetic and
    // drops only the loads shows the loads cost about twice what the FMAs do.
    // OPT bit 6 issues them a full 64-k iteration ahead, at the cost of TM*8 live VGPRs.
    if (ASYM && (OPT & 64) && (k0 % GRP) == 0) {
      const int Mpadp = ((M + 15) / 16) * 16;
#pragma unroll
      for (int i = 0; i < TM; i++) {
        int rb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
        rb = rb < Mpadp - 8 ? rb : Mpadp - 8;
        const v4f_t* pv = (const v4f_t*)(Asum + (size_t)(k0 / GRP) * Mpadp + rb);
        const v4f_t v0 = pv[0], v1 = pv[1];
#pragma unroll
        for (int e = 0; e < 4; e++) { asp[i][e] = v0[e]; asp[i][e + 4] = v1[e]; }
      }
    }
    if (IU4) {
#pragma unroll
      for (int c = 0; c < BK / 32; c++) {
        // ONE PLANE AT A TIME.  Holding ah[TM] and al[TM] together costs TM more v2i of live range
        // on a kernel that already pays 8*TM*TN for the second accumulator: 115 VGPRs against the
        // iu8 path's 85, straight through the 96-VGPR step and from 16 waves per SIMD down to 12.
        // Doing the high plane's WMMAs and then the low plane's reuses the registers; the cost is
        // re-reading the weight fragment once per plane, and the
        // DCE-proof ablation already showed the W LDS read is free once the unpack is gone.
#pragma unroll
        for (int P = 0; P < 2; P++) {
          v2i_t af4[TM];
#pragma unroll
          for (int i = 0; i < TM; i++) {
            const int row = (wm * TM + i) * 16 + (lane & 15);
            // dword offset in the row: c*8 for the K=32 chunk, +4 for the low plane, +2 per half
            const unsigned* pa4 = lds_a[buf] + ASW(row, c * 8 + P * 4 + 2 * (lane >> 4));
            af4[i][0] = (int)pa4[0];
            af4[i][1] = (int)pa4[1];
          }
#pragma unroll
          for (int j = 0; j < TN; j++) {
            const u4v_t wv = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + lane * 4);
            v2i_t bq;                                // two k steps = one K=32 iu4 B fragment
            bq[0] = (int)wv[2 * c];
            bq[1] = (int)wv[2 * c + 1];
#pragma unroll
            for (int i = 0; i < TM; i++) {
              const bool f = (first_half && c == 0);
              // P = 0 is the signed high half, P = 1 the unsigned low half.
              if (P == 0)
                g2[j][i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    true, af4[i], true, bq, f ? (v8i_t)(0) : g2[j][i], false);
              else
                g[j][i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, af4[i], true, bq, f ? (v8i_t)(0) : g[j][i], false);
            }
          }
        }
      }
    } else {
    // OPT BIT 16384: PREFETCH THE A FRAGMENTS ONE K STEP AHEAD.
    // The ISA of the shipped tile issues a k step's `ds_load_2addr_stride64_b64` and then waits on
    // it one instruction later -- `s_wait_dscnt 0x1` sits between the load and the first WMMA that
    // consumes it, and `s_wait_dscnt 0x0` sits between the second and third WMMA of every triple.
    // The loads have essentially no cover.  Rotating a second register set moves each load a whole
    // WMMA triple plus an unpack earlier, at the cost of TM*2 VGPRs.
    v2i_t afp[TM];
    auto aload = [&](int uu, v2i_t* dst) {
#pragma unroll
      for (int i = 0; i < TM; i++) {
        const int row = (wm * TM + i) * 16 + (lane & 15);
        if (ANOLDS) {
          const int rt0 = m0 / 16 + wm * TM + i, rtm = (M - 1) / 16;
          const int rt = rt0 < rtm ? rt0 : rtm;
          dst[i] = *(const v2i_t*)(A + ((size_t)rt * (K / 16) + (k0 / 16 + uu)) * 256 + lane * 8);
          continue;
        }
        const unsigned* pa = lds_a[buf] + ASW(row, uu * 4 + 2 * (lane >> 4));
        dst[i][0] = (int)pa[0];
        dst[i][1] = (ABX == 7) ? (int)pa[0] : (int)pa[1];
      }
    };
    if (OPT & 16384) aload(0, afp);
#pragma unroll
    for (int u = 0; u < BK / 16; u++) {
      v2i_t af[TM];
      if (OPT & 16384) {
#pragma unroll
        for (int i = 0; i < TM; i++) af[i] = afp[i];
        if (u + 1 < BK / 16) aload(u + 1, afp);
      } else
#pragma unroll
      for (int i = 0; i < TM; i++) {
        const int row = (wm * TM + i) * 16 + (lane & 15);
        if (ANOLDS) {
          // [row tile][k step][lane][8 bytes]: lane-contiguous, so one coalesced 8-byte load per
          // (i, u).  Clamped to the last real row tile only because the CURRENT artifact is not
          // padded to a multiple of 16 rows; a real one would be, and the clamp would not exist.
          const int rt0 = m0 / 16 + wm * TM + i, rtm = (M - 1) / 16;
          const int rt = rt0 < rtm ? rt0 : rtm;
          const v2i_t* pg =
              (const v2i_t*)(A + ((size_t)rt * (K / 16) + (k0 / 16 + u)) * 256 + lane * 8);
          af[i] = *pg;
          continue;
        }
        const unsigned* pa = lds_a[buf] + ASW(row, u * 4 + 2 * (lane >> 4));
        // ABL 7 prices the SECOND activation LDS dword only -- halving the A read while keeping
        // the staging path live, which is what a wider A permutation would buy.  A version that
        // dropped both made lds_a dead and deleted the staging, which is not a measurement.
        af[i][0] = (int)pa[0];
        af[i][1] = (ABX == 7) ? (int)pa[0] : (int)pa[1];
      }
      if (ABX == 8) {
        // THE INSTRUCTION MIX HAS TO MATCH THE REAL KERNEL OR THIS PRICES NOTHING.
        // Doing both iu4 calls on even u only leaves the odd u's `af` dead -- the compiler deletes
        // those `ds_load_b64` and the variant then does HALF the activation reads the real thing
        // would. For K=64 the real kernel loads A_hi and A_lo for two K=32 chunks: four
        // `ds_load_b64` per (i, BK), exactly what the iu8 path loads.  So run every u and do ONE
        // iu4 per (i, j, u), alternating the accumulator -- four A loads, four matrix instructions,
        // one b128 of B, no dequant.
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const u4v_t wv4 = *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + lane * 4);
          v2i_t bq;
          bq[0] = (int)wv4[u & ~1];
          bq[1] = (int)wv4[(u & ~1) + 1];
#pragma unroll
          for (int i = 0; i < TM; i++) {
            // The signedness argument must be a literal, so the two halves are two call sites
            // rather than one with a computed flag.
            const bool f = (first_half && u < 2);
            if (u & 1)
              g2[j][i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                  false, af[i], true, bq, f ? (v8i_t)(0) : g2[j][i], false);
            else
              g[j][i] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                  true, af[i], true, bq, f ? (v8i_t)(0) : g[j][i], false);
          }
        }
      } else
#pragma unroll
      for (int j = 0; j < TN; j++) {
        // ABL 5 prices the UNPACK: same LDS read, no dequant VALU.  ABL 6 prices the weight LDS
        // read as well.  Both compute wrong answers by construction; they exist to say what the gap
        // between this kernel's rate and the 407 TOPS matrix ceiling is made of.
        u4v_t wv;
        if constexpr (WNOLDS)
          wv = rwg[buf][j];
        else
          wv = (WU || ABX == 6)
                   ? (u4v_t)(0x21212121u)
                   : *(const u4v_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 128 + lane * 4);
        v2i_t bf;
        if (WU) bf = *(const v2i_t*)(lds_w[buf] + (size_t)(wn * TN + j) * 256 + lane * 8 + u * 2);
        else
        // ABL 5 KEEPS THE DATA DEPENDENCY, and it has to.  Assigning constants here would make the
        // LDS read dead and let the compiler delete the whole weight staging path, which prices
        // something else entirely.  Forwarding the raw dword prices the three unpack instructions
        // and nothing else: same load, same live range, no masks and no shift.
             if (ABX == 5) { bf[0] = (int)wv[u]; bf[1] = (int)wv[u]; }
        else if (ABX == 6) { bf[0] = 0x10101010; bf[1] = 0x20202020; }
        else bf = tdequant8(wv[u]);
#pragma unroll
        for (int i = 0; i < TM; i++)
          g[j][i] = (first_half && u == 0)
                        ? tmmai<!ASYM>(af[i], bf, (v8i_t)(BIAS == 2 ? 0x4B400000 : 0))
                        : tmmai<!ASYM>(af[i], bf, g[j][i]);
      }
    }
    }

    // OPT BIT 2048: SCHEDULE THE K-BLOCK BY HAND.
    // Left alone, LLVM interleaves the fragment reads with the matrix work and drains LDS between
    // them -- the ISA reads `WW d W RR d WW d W RR`, an `s_wait_dscnt` inside the WMMA run.  The
    // same pathology appears in the R4D attention kernel and the same fix applies: name the
    // pipeline, DS reads then unpack then matrix, and let the staging fill the gaps.  Counts are
    // per k-block and are upper bounds -- an unmatched instruction is simply scheduled freely, and
    // the A reads in particular collapse into `ds_load_2addr_stride64_b64` pairs, so asking for
    // more DS reads than exist is harmless.
    if (OPT & 2048) {
      constexpr int NW = TM * TN * (BK / 16);       // matrix instructions
      constexpr int NR = TM * (BK / 16) + TN;       // A fragments + the weight b128
      constexpr int NV = 3 * TN * (BK / 16);        // the unpack
      if (OPT & 4096) {                             // finer: one stage per k step
#pragma unroll
        for (int q = 0; q < BK / 16; q++) {
          __builtin_amdgcn_sched_group_barrier(0x100, NR / (BK / 16) + 1, 0);
          __builtin_amdgcn_sched_group_barrier(0x002, NV / (BK / 16), 0);
          __builtin_amdgcn_sched_group_barrier(0x008, NW / (BK / 16), 0);
        }
      } else {                                      // coarse: the whole k-block in three stages
        __builtin_amdgcn_sched_group_barrier(0x100, NR, 0);
        __builtin_amdgcn_sched_group_barrier(0x002, NV, 0);
        __builtin_amdgcn_sched_group_barrier(0x008, NW, 0);
      }
    }

    if ((k0 + BK) % AGRP == 0) {                    // the group closed: fold it into the float acc
      if (TWOACC) {
        // The hi/lo combine belongs at the GROUP close, not every k step -- both halves accumulate
        // in int32 across the whole group and only the float fold needs them summed.  Doing it per
        // k step costs 2x at GRP=128/BK=64.  One `v_lshl_add_u32` per element per group.
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int i = 0; i < TM; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) g[j][i][e] += g2[j][i][e] << 4;
      }
      float sc[TN];
#pragma unroll
      for (int j = 0; j < TN; j++)
      {
        // 32-BIT INDEX.  With size_t the whole address goes 64-bit and every group pays
        // v_ashrrev + v_lshlrev_b64 + two v_add_co per scale; the element count here is at most
        // (N/16)*(K/GRP)*16, a few hundred thousand.
        const unsigned raw =
            ABX == 4  ? 0x3C00u                    /* ABL 4: no scale LOAD, same arithmetic */
            : (OPT & 1) ? scv[j] : R4D_WS(k0 / GRP, j);
        sc[j] = __half2float(__ushort_as_half((unsigned short)(raw & 0xFFFFu)));
      // OPT BIT 8192: STOP THE f16 CONVERSION FOLDING INTO THE FOLD.
      // Left alone the compiler contracts `(float)h * x + acc` into `v_fma_mix_f32`, which reads
      // the f16 for free -- and is VOP3P, so it can never dual-issue.  `v_fmac_f32` IS on the VOPD
      // list (`v_dual_fmac_f32 :: v_dual_fmac_f32` assembles for gfx1201), and the fold is
      // TM*TN*8 = 24 of the tile's 221 instructions per scale group.  One `asm` per group forces
      // the convert to materialise and leaves 24 plain FMACs for the VOPD packer.
      if (OPT & 8192) asm("" : "+v"(sc[j]));
      }
      // THE ASYMMETRIC GRID.  With the nibble read unsigned the dequantised weight is
      // (q - z) * s, so the group contributes s * (sum_k a_k q_k - z * sum_k a_k) and the second
      // term needs a per-row per-group sum of the ACTIVATION -- `r4d_quant_act_i8_asum`'s output,
      // laid out [G][Mpad] because element e of tile i at this lane is output row
      // (m0 + (wm*TM+i)*16 + 8*(lane>>4) + e), so a lane's eight rows are CONTIGUOUS at a fixed
      // group and load as two `v4f`.  The zero rides in the scale dword's high half as the f16 of
      // -(1024 + z), which is exact because f16 has exactly 1024 steps between 1024 and 2048.
      //
      // It costs ONE extra fma per element and nothing else: symz needs `cvt` then `fma`, asym
      // needs `fma` then `fma`, and the magic bias has already deleted the cvt.  The loop is nested
      // i-outer so only ONE tile's eight asum values are live -- asum[TM][8] at TM=4 is 32 VGPRs
      // against a budget of 192 that is already exactly full.
      if (BIAS) {
#pragma unroll
        for (int j = 0; j < TN; j++) csum[j] += sc[j];
      }
      float z16[TN];
      if (ASYM) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
          const unsigned raw =
              (OPT & 1) ? scv[j]
                        : Ws[(unsigned)(((n0 / 16 + wn * TN + j) * nsz + k0 / GRP) * 16 +
                                        (lane & 15))];
          const float nz = __half2float(__ushort_as_half((unsigned short)(raw >> 16)));
          z16[j] = 16.f * (-nz - 1024.f);
        }
      }
      const int Mpad = ((M + 15) / 16) * 16;
      // ONE TILE AHEAD (OPT bit 7).  Written as load-then-use per tile, the four tiles' asum loads
      // serialise: each one's latency is fully exposed because the sixteen fmas that could hide it
      // come after it, not before.  Holding all TM*8 values instead (the skinny kernel's shape) is
      // 32 VGPRs on a kernel that has 6 spare and spills 164 bytes.  Two v4f sets -- current and
      // next -- is 16, and only the first load's latency is left uncovered.
#define R4D_ASUM_LD(ii, a, b)                                                                     \
  do {                                                                                            \
    int rb_ = m0 + (wm * TM + (ii)) * 16 + 8 * (lane >> 4);                                        \
    rb_ = rb_ < Mpad - 8 ? rb_ : Mpad - 8;                                                         \
    const v4f_t* pv_ = (const v4f_t*)(Asum + (size_t)(k0 / GRP) * Mpad + rb_);                     \
    (a) = pv_[0];                                                                                 \
    (b) = pv_[1];                                                                                 \
  } while (0)
      v4f_t c0, c1;
      if (ASYM && (OPT & 128)) R4D_ASUM_LD(0, c0, c1);
#pragma unroll
      for (int i = 0; i < TM; i++) {
        constexpr int NE = ABX == 5 ? 1 : 8;             /* ABL 5: 1/8 of the arithmetic */
        float as[8];
        if (ASYM) {
          if (OPT & 32) {                                /* prices the fmas without their loads */
#pragma unroll
            for (int e = 0; e < 8; e++) as[e] = 1.f;
          } else if (OPT & 64) {
#pragma unroll
            for (int e = 0; e < 8; e++) as[e] = asp[i][e];
          } else if (OPT & 128) {
            const v4f_t u0 = c0, u1 = c1;
            if (i + 1 < TM) R4D_ASUM_LD(i + 1, c0, c1);  // in flight across THIS tile's fmas
#pragma unroll
            for (int e = 0; e < 4; e++) { as[e] = u0[e]; as[e + 4] = u1[e]; }
          } else {
            int rb = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4);
            rb = rb < Mpad - 8 ? rb : Mpad - 8;          // rows past M are stored away, not read
            const v4f_t* pv = (const v4f_t*)(Asum + (size_t)(k0 / GRP) * Mpad + rb);
            const v4f_t v0 = pv[0], v1 = pv[1];
#pragma unroll
            for (int e = 0; e < 4; e++) { as[e] = v0[e]; as[e + 4] = v1[e]; }
          }
        }
#pragma unroll
        for (int j = 0; j < TN; j++)
#pragma unroll
          for (int e = 0; e < NE; e++) {
            float v = BIAS == 2 ? __uint_as_float((unsigned)g[j][i][e])
                    : BIAS == 1 ? __uint_as_float((unsigned)g[j][i][e] + 0x4B400000u)
                                : (float)g[j][i][e];
            if (ASYM) v = fmaf(-z16[j], as[e], v);
            acc[j][i][e] = fmaf(v, sc[j], acc[j][i][e]);
          }
      }
#undef R4D_ASUM_LD
    }
    if ((ABL == 0 || ABL >= 10) && k0 + BK < kend) commit(buf ^ 1);  // the wait for those loads lands HERE
    if (ABX != 2) ldsbar();
    }
  }
#undef ASW

  if (BIAS) {
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const float d = -12582912.0f * csum[j];
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][i][e] += d;
    }
  }

  // ---- the split tail's partial, and its reduction ----
  //
  // The partial is the accumulator AFTER the magic-bias correction and BEFORE `Ascale[m]`, because
  // the row scale is common to every piece and multiplying it in once at the end is both cheaper
  // and exactly what the unsplit path does.  Rows past M are never written and never read: at
  // M = 72 in a BM = 96 tile that is a quarter of the workspace traffic saved for free.
  //
  // The counter lives in ONE dword of `lds_a`, not in a `__shared__` of its own: A and W are
  // exactly 32 KB together and four more bytes would drop a CU from two workgroups to one.  The
  // k loop is finished by here, so the buffer is dead.
  if (SKT && slot >= 0) {
    float* wsp = WSP + (size_t)slot * BM * BN;
#pragma unroll
    for (int j = 0; j < TN; j++)
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int nl = (wn * TN + j) * 16 + (lane & 15);
          const int ml = (wm * TM + i) * 16 + 8 * (lane >> 4) + e;
          if (m0 + ml < M) wsp[(size_t)ml * BN + nl] = acc[j][i][e];
        }
    // THE FENCE HAS TO BE ON EVERY THREAD, not just the one that bumps the counter.  A fence
    // orders the stores of the thread that executes it; `__syncthreads()` is a workgroup-scope
    // release and says nothing about when the other 511 threads' stores become visible to another
    // CU.  With the fence on thread 0 alone this kernel returns a DIFFERENT wrong answer on every
    // call -- the reducer reads whatever happens to have drained.
    __threadfence();
    __syncthreads();
    const int cidx = (int)blockIdx.y * nrag + (tilex - nfull);
    if (tid == 0)
      lds_a[0][0] = __hip_atomic_fetch_add(&CNT[cidx], 1u, __ATOMIC_ACQ_REL,
                                           __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (lds_a[0][0] != (unsigned)(nspl - 1)) return;
    if (tid == 0) CNT[cidx] = 0u;          // left clean for the next call on this stream
    __threadfence();                       // acquire, before reading the other pieces
    const float* base =
        WSP + (size_t)((int)blockIdx.y * (nrag * nspl) + (tilex - nfull) * nspl) * BM * BN;
#pragma unroll
    for (int j = 0; j < TN; j++)
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const int nl = (wn * TN + j) * 16 + (lane & 15);
          const int ml = (wm * TM + i) * 16 + 8 * (lane >> 4) + e;
          const int n = n0 + nl, m = m0 + ml;
          if (n < N && m < M) {
            float v = 0.f;
            // SKT == 2 is a DEBUG variant: reduce only the first piece, so the output is piece 0's
            // partial and can be scored against the unsplit kernel run on the K slice [0, K/nspl).
            for (int q = 0; q < nspl; q++)
              v += base[(size_t)q * BM * BN + (size_t)ml * BN + nl];
            C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * (IU4 ? 1.0f : 0.0625f));
          }
        }
    return;
  }

  // ---- store ----
  //
  // FIVE FORMS OF THIS EPILOGUE WERE COMPARED; THE FLAT PREDICATED ONE WINS EVERY TIME.
  //
  //   flat predicated store (this)
  //   transpose through LDS as bf16
  //   transpose through LDS as shuffle-packed dwords
  //   reorder to hoist Ascale, early-out on m
  //   reorder to hoist Ascale + the 64-bit m*N, pure predication
  //
  // The epilogue is NOT bandwidth-bound: folding every address down to 64 rows, so the whole output
  // is cache-resident while the instruction count is untouched, takes the same time.  That is why
  // both transposes lose -- they add instructions to fix a bandwidth problem that does not exist.
  // (An ablation that simply deletes the stores makes them look like 40% of the kernel, but it also
  // deletes the accumulator's only consumer, so most of that is dead-code elimination; the
  // transposed-accumulator note below says what happens when the stores really do get cheaper.)
  //
  // And reducing the count by hand does not help either.  Hoisting the Ascale load and the 64-bit
  // row multiply out of the j loop is a real reduction -- 16 of each instead of 64 -- and is slower
  // whether the guard is a branch or a predicate.  Putting j innermost is what does it: the
  // accumulator is acc[j][i][e], so j-innermost strides the register file, and the four stores then
  // share one guard instead of being independently predicated.  The compiler schedules the flat form
  // better than any of the four rewrites.
#pragma unroll
  for (int j = 0; j < TN; j++)
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int n = n0 + (wn * TN + j) * 16 + (lane & 15);
        const int m = m0 + (wm * TM + i) * 16 + 8 * (lane >> 4) + e;
        if (n < N && m < M)
          // 0.0625 undoes the 16 the iu8 unpack puts in the byte's high nibble.  The iu4 path
          // multiplies the RAW nibble, so there is no 16 to undo.
          C[(size_t)m * N + n] =
              (__hip_bfloat16)(acc[j][i][e] * Ascale[m] * (IU4 ? 1.0f : 0.0625f));
      }
}


// THE TRANSPOSED ACCUMULATOR IS NEUTRAL.  The code is not here; this note is why not.
//
// gfx12's WMMA output is column-distributed -- lanes index the column, the eight accumulator
// registers index the row -- so with the activation as the A operand a lane owns eight DIFFERENT
// ROWS of one column and each of its 64 outputs is an independent 2-byte store.  Swapping the A and
// B operands transposes D (AMD's own RDNA4 WMMA guide documents the swap for exactly this), a lane
// then owns eight CONSECUTIVE COLUMNS of one row, and the epilogue collapses to eight
// `global_store_b128` with `Ascale[m]` loaded twice instead of 64 times.  Coalescing is unchanged:
// 128 segments for the wave's 4096 bytes either way.
//
// It is EXACTLY NEUTRAL on `gate_proj`, inside the run-to-run spread, and correct to the same
// 2.32e-3.  The store epilogue is not the cost: the ablation that appears to make it 40% of the
// kernel deletes the accumulator's only consumer, so most of that 40% is dead-code elimination.
// The transposed form also costs 193 VGPRs unless it is written carefully, which is a fifth of the
// runtime on its own.

// THE BLOCK TILE IS NOT FIXED, because 128 x 256 is the wrong shape for a mid-sized M.  At
// M = 128-256 that tile gives one or two M-blocks and almost no M-parallelism, and that is exactly
// the band where the kernel falls behind the block-fp8 bar while beating it at M = 64 and matching
// it at M = 4096.  The grid therefore comes from the template arguments rather than a constant.
// THE TAIL WORKSPACE is a cached allocation, grown on demand and never freed, ONE A (DEVICE,
// STREAM).  It is only ever touched by the tail workgroups of a single launch, and two launches on
// one stream cannot overlap on it.  The counters are zeroed once here and left at zero by the
// kernel, which is what makes a graph replay safe: nothing per-call runs on the host.
//
// Both halves of that rest on ONE STREAM, which is why the cache is keyed the way gemm_fp8a8's
// split_ctr is.  A tensor-parallel process launches on one device a rank, so a single allocation
// would hand one device's pointer to the other's kernel; and `rad_lane` runs a second stream beside
// the first, where two launches sharing one workspace would reduce into each other's pieces and
// each see the other's counter arrivals.  A rank runs a compute stream and at most one lane, so the
// table is small and fixed, and a stream past it is refused rather than shared.
constexpr int kSktStreams = 4;
struct SktSlot {
  bool used = false;
  hipStream_t key = nullptr;
  float* ws = nullptr;
  size_t ws_bytes = 0;
  unsigned* cnt = nullptr;
  size_t cnt_n = 0;
};

// AND IT IS THE LAST ALLOCATION ON A LAUNCH PATH IN THIS LIBRARY, which the fp8a8 header calls out
// as the thing not to do ("Nothing on an R4D launch path allocates, so the partial plane is an
// argument and r4d_gemm_fp8a8_nt_m16_scratch() sizes it").  It is bounded -- the guard above caps
// mblk*nrag*nspl at 64 and a piece is BM*BN floats, so a couple of MiB at the shapes that take the
// split at all -- and it is taken once and kept, so it costs one hipMalloc a (device, stream) in
// the life of a process rather than one a step.
//
// It is still invisible to the engine's VRAM budget.  core/mem/vram_budget.cpp sizes the pools
// down to --gpu-headroom-mib from terms it can compute, and a plugin's own hipMalloc is not one of
// them: core/device/dispatch.cpp's ledger counts what goes through rad_dev_alloc and nothing that
// goes around it.  Engine::start reads the card back afterwards for exactly this reason, so the
// bytes show up as "settled at N free against the M --gpu-headroom-mib asked to keep" rather than
// as a number nobody can name.
//
// THE FIX IS THE SCRATCH HOOK, NOT A BIGGER HEADROOM -- with one obstacle worth writing down
// before somebody tries it.  The workspace itself would move to a->scratch cleanly: it is written
// and read inside one launch.  The COUNTERS cannot, because the contract that makes a graph replay
// safe is that they are zero at kernel entry and left at zero on exit, and the arena's scratch
// region is shared by every op on the stream -- the next kernel to use it would leave them
// anything at all.  They need a persistent declared buffer, which the buffer plan would then place
// and the budget would then charge.
static void r4d_skt_reserve(hipStream_t stream, size_t ws_bytes, size_t cnt_n, float** ws,
                            unsigned** cnt) {
  static std::mutex mu;
  static SktSlot slot[16][kSktStreams];
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16)
    throw std::runtime_error("r4d gemm_w4a8_tiled: no current device for the split-tail workspace");
  std::lock_guard<std::mutex> lk(mu);
  SktSlot* e = nullptr;
  for (SktSlot& x : slot[dev])
    if (x.used && x.key == stream) { e = &x; break; }
  if (!e) {
    for (SktSlot& x : slot[dev])
      if (!x.used) { e = &x; break; }
    if (!e) throw std::runtime_error("r4d gemm_w4a8_tiled: more streams than split-tail workspaces");
    e->used = true;
    e->key = stream;
  }
  if (ws_bytes > e->ws_bytes) {
    if (e->ws) (void)hipFree(e->ws);
    e->ws = nullptr;
    e->ws_bytes = 0;
    if (hipMalloc((void**)&e->ws, ws_bytes) != hipSuccess)
      throw std::runtime_error("r4d gemm_w4a8_tiled: split-tail workspace alloc failed");
    e->ws_bytes = ws_bytes;
  }
  if (cnt_n > e->cnt_n) {
    if (e->cnt) (void)hipFree(e->cnt);
    e->cnt = nullptr;
    e->cnt_n = 0;
    if (hipMalloc((void**)&e->cnt, cnt_n * sizeof(unsigned)) != hipSuccess)
      throw std::runtime_error("r4d gemm_w4a8_tiled: split-tail counter alloc failed");
    (void)hipMemset(e->cnt, 0, cnt_n * sizeof(unsigned));
    e->cnt_n = cnt_n;
  }
  *ws = e->ws;
  *cnt = e->cnt;
}

// HOW MANY WAYS TO SPLIT THE TAIL.  Enough pieces to fill the 32 slots, but the piece boundary has
// to land on a scale group or the group epilogue reads the wrong scale, so `nspl` must divide
// K/GROUP.  Capped at 8 because the tail cost turns back up past that point: the pieces become too
// short to hide their own prologue.
static int r4d_skt_split(int nrag, int K) {
  const int ngrp = K / R4D_GEMM_W4A8_GROUP;
  const int want = 32 / (nrag > 0 ? nrag : 32);
  int best = 1;
  for (int t = 2; t <= 8 && t <= want; t++)
    if (ngrp % t == 0) best = t;
  return best;
}

template <int WM, int WN, int TM, int TN, int BIAS, int ABL, int OPT, int WPE, int ASYM, int UNR,
          int SKT = 0>
static void r4d_tiled_launch(long a, long ascale, long wq, long ws, long asum, long c, int M, int K,
                             int N, long stream) {
  constexpr int BM = WM * TM * 16, BN = WN * TN * 16;
  if (N % BN)
    throw std::runtime_error("r4d gemm_w4a8_tiled: N is not a multiple of this variant's block");
  const int tiles = N / BN, mblk = (M + BM - 1) / BM;
  int nfull = tiles, nrag = 0, nspl = 1, gx = tiles;
  float* skt_ws = nullptr;
  unsigned* skt_cnt = nullptr;
  if (SKT) {
    nrag = tiles & 31;
    nfull = tiles - nrag;
    nspl = nrag ? r4d_skt_split(nrag, K) : 1;
    // A tail worth splitting, and a workspace worth keeping: at decode mblk is 1 and this is a
    // couple of MiB.  At prefill mblk is large, the tail is a rounding error against the M blocks,
    // and the workspace would not be -- so the split is simply off there.
    if (nrag == 0 || nspl < 2 || (size_t)mblk * nrag * nspl > 64) {
      nfull = tiles; nrag = 0; nspl = 1;
    } else {
      gx = nfull + nrag * nspl;
      r4d_skt_reserve((hipStream_t)stream, (size_t)mblk * nrag * nspl * BM * BN * sizeof(float),
                      (size_t)mblk * nrag, &skt_ws, &skt_cnt);
    }
  }
  dim3 grid(gx, mblk), block(WM * WN * 32);
  hipLaunchKernelGGL(
      (r4d_gemm_w4a8_tiled_kernel<WM, WN, TM, TN, BIAS, ABL, OPT, WPE, ASYM, UNR, SKT>), grid,
      block, 0, (hipStream_t)stream, (const signed char*)a, (const float*)ascale,
      (const unsigned*)wq, (const unsigned*)ws, (const float*)asum, (__hip_bfloat16*)c, M, K, N,
      skt_ws, skt_cnt, nfull, nrag, nspl);
}

static void r4d_gemm_w4a8_tiled_impl(long a, long ascale, long wq, long ws, long asum, long c,
                                     int M, int K, int N, long stream, int variant, bool asym) {
  if (N % 256) throw std::runtime_error("r4d gemm_w4a8_tiled: N must be a multiple of 256");
  if (K % R4D_GEMM_W4A8_GROUP)
    throw std::runtime_error("r4d gemm_w4a8_tiled: K must be a multiple of the scale group");
  if (M < 1) throw std::runtime_error("r4d gemm_w4a8_tiled: M must be positive");
  // THE UNROLL MUST COVER EXACTLY ONE SCALE GROUP -- no more.  GRP/BK makes the group boundary a
  // compile-time constant, which is the whole point; going further just doubles live ranges and
  // wrecks the register allocation.  At group 128, UNR = GRP/BK sits at the fp8 bar, UNR = 1 is
  // about a tenth over it, and UNR = 4 is more than twice it; a group-64 build forced to two groups
  // a trip is four times it.
  constexpr int UNRG = R4D_GEMM_W4A8_GROUP / R4D_TILED_KPB > 1
                           ? R4D_GEMM_W4A8_GROUP / R4D_TILED_KPB
                           : 1;
  const bool ufull = (K % (UNRG * R4D_TILED_KPB)) == 0;
#define R4D_TILED_LAUNCH(...) \
  r4d_tiled_launch<__VA_ARGS__>(a, ascale, wq, ws, asum, c, M, K, N, stream)
  // The variant argument selects an alternative tile and exists for the bench; 0 is what ships.
  // A variant that lost and has nothing left to say is NOT kept as an instantiation -- each one is
  // a kernel to compile.
  if (asym) {
    switch (variant) {
      // EVERY asym instantiation with the magic bias SPILLS -- z16[TN], as[8] and csum[TN] land on
      // top of 128 accumulator registers -- so the asym variants drop it.  BIAS=0 at <2,8,4,2> is
      // 186 VGPRs with no scratch; BIAS=2 is 192 with 24 bytes.
      case 1: R4D_TILED_LAUNCH(2, 8, 4, 2, 0, 0, 51, 8, 1, 1); break;  // no asum LOADS
      case 2: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 19, 8, 1, 1); break;  // magic bias
      // The unroll is a LOSS on the asymmetric path -- 186 VGPRs plus doubled live ranges -- so the
      // asym default keeps UNR=1 where symz takes UNR=2.
      case 3: R4D_TILED_LAUNCH(2, 8, 4, 2, 0, 0, 19, 8, 1, UNRG); break; // unrolled
      // Losers, not instantiated: holding all TM*8 asum values (OPT 64) spills 164 bytes and costs
      // half again; pipelining them one tile ahead (OPT 128) is neutral; TM=2 is a few percent
      // worse.  The code for all three is still in the kernel because each is one branch and
      // together they document the wall.
      default: R4D_TILED_LAUNCH(2, 8, 4, 2, 0, 0, 19, 8, 1, 1); break;
    }
  } else {
    switch (variant) {
      case 1: R4D_TILED_LAUNCH(4, 4, 2, 4, 0, 0, 0, 8, 0, 1); break;   // 4x4 waves, 2x4 tiles, plain cvt
      case 2: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 19, 8, 0, 1); break;  // UNR=1
      case 3: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 3, 8, 0, UNRG); break; // no wide W
      case 4: R4D_TILED_LAUNCH(2, 8, 4, 2, 0, 0, 19, 8, 0, UNRG); break; // no bias
      case 5: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 19, 8, 0, 2 * UNRG); break; // 2x
      // BM=64 for the mid-M band.  Same 256-wide N, half the M tile, so M=128 is two blocks instead
      // of one and M=256 is four.  TM*TN drops 8 -> 4, which also halves the accumulator registers.
      case 6: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 0, 147, 8, 0, UNRG); break;  // BM 64,  BN 256, 512 thr
      case 7: R4D_TILED_LAUNCH(1, 16, 4, 1, 2, 0, 19, 8, 0, UNRG); break;  // BM 64, 1 wave deep
      case 8: R4D_TILED_LAUNCH(2, 4, 2, 4, 2, 0, 19, 8, 0, UNRG); break;   // BM 64, 256 thr
      case 9: R4D_TILED_LAUNCH(2, 8, 4, 1, 2, 0, 19, 8, 0, UNRG); break;   // BM 128, BN 128
      // BM 96 AND BM 80, FOR THE SPECULATIVE-DECODE BATCH.  A serve at eight concurrent requests
      // with verify width 9 runs M=72, and 72 is the worst M either kernel has: the skinny one
      // needs five row tiles so it walks the weight five times, and BM=64 needs two blocks whose
      // second holds 8 of 64 rows.  BM=96 covers 72 in ONE block, so the weight is read once and
      // nothing is half empty.  BM=80 is the tightest fit and the widest tile per wave.
      case 10: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 147, 8, 0, UNRG); break; // BM 96,  BN 256, 512 thr
      case 12: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 147, 8, 0, UNRG); break; // BM 96,  BN 128, 512 thr
      // The staging tilings are not free choices: W needs WM | 4*TN and A needs WN | 8*TM, or the
      // static_asserts above fire.  That leaves WM*TM = 6 as the only route to BM 96, i.e. <2,_,3,_>
      // or <1,_,6,_>, and it is why there is no <3,8,2,2>.  Dropped: BM 80 in either width, BM 96
      // at 256 threads, BN 512 anywhere, and BM 64 at 1024 threads -- everything that raises A
      // dwords-per-thread costs 2-3x.  Also dropped: skipping the staging of rows past M
      // (OPT bit 32), which is WORSE -- the divergent branch costs more than the duplicate rows it
      // saves.
      // Cost ablations on the two tiles that win at these batches. Wrong answers by construction.
      case 24: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 5, 19, 8, 0, UNRG); break;  // BM64 no unpack
      case 25: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 6, 19, 8, 0, UNRG); break;  // BM64 no W LDS read
      case 26: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 7, 19, 8, 0, UNRG); break;  // BM64 no A LDS read
      case 27: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 5, 19, 8, 0, UNRG); break;  // BM96 no unpack
      case 28: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 6, 19, 8, 0, UNRG); break;  // BM96 no W LDS read
      case 29: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 7, 19, 8, 0, UNRG); break;  // BM96 no A LDS read
      // The same three against the SWIZZLED baseline, which is what ships; the unswizzled variants
      // above overstate the A read, because it is 2-way conflicted there.
      case 34: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 5, 147, 8, 0, UNRG); break; // BM96 swz, no unpack
      case 35: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 6, 147, 8, 0, UNRG); break; // BM96 swz, no W lds
      case 36: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 7, 147, 8, 0, UNRG); break; // BM96 swz, half A lds
      case 37: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 5, 147, 8, 0, UNRG); break; // BM64 swz, no unpack
      case 38: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 7, 147, 8, 0, UNRG); break; // BM64 swz, half A lds
      // OPT bit 256: unpack the weight once at staging.  147|256 = 403.
      case 40: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 403, 8, 0, UNRG); break; // BM96 BN128 swz + WU
      case 41: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 0, 403, 8, 0, UNRG); break; // BM64 BN256 swz + WU
      case 42: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 403, 8, 0, UNRG); break; // BM96 BN256 swz + WU
      case 43: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 403, 8, 0, UNRG); break; // BM128 BN256 swz + WU
      // TALLER ROW TILES AT BN=128, so that M = 97..256 still reads the weight exactly ONCE.
      // A second M-block doubles the weight traffic, and this kernel is weight-bandwidth-bound at
      // decode M; BM=64 with two blocks reads 2x45 MiB at M=128 where BM=192 reads 45.
      case 56: R4D_TILED_LAUNCH(2, 8, 6, 1, 2, 0, 147, 8, 0, UNRG); break; // BM 192, BN 128, 512 thr
      case 57: R4D_TILED_LAUNCH(2, 8, 5, 1, 2, 0, 147, 8, 0, UNRG); break; // BM 160, BN 128, 512 thr
      case 58: R4D_TILED_LAUNCH(2, 8, 8, 1, 2, 0, 147, 8, 0, UNRG); break; // BM 256, BN 128, 512 thr
      case 59: R4D_TILED_LAUNCH(2, 8, 4, 1, 2, 0, 147, 8, 0, UNRG); break; // BM 128, BN 128, 512 thr
      // SAME BLOCK TILE, DIFFERENT WAVE SPLIT.  Per WMMA the loop pays 3/TM unpack VALU and 1/TN
      // activation `ds_load_b64`, so BM 96 x BN 128 costs 2.0 as <2,8,3,1> and 1.5 either as
      // <2,4,3,2> or as <1,8,6,1> -- at half the waves per block.
      case 52: R4D_TILED_LAUNCH(2, 4, 3, 2, 2, 0, 147, 8, 0, UNRG); break; // BM 96,  BN 128, 256 thr
      case 53: R4D_TILED_LAUNCH(1, 8, 6, 1, 2, 0, 147, 8, 0, UNRG); break; // BM 96,  BN 128, 256 thr
      // OPT bit 1024: the LDS-only barrier (147|1024 = 1171).
      case 60: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 1171, 8, 0, UNRG); break; // BM 96,  BN 128
      case 61: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 0, 1171, 8, 0, UNRG); break; // BM 64,  BN 256
      case 62: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 1171, 8, 0, UNRG); break; // BM 96,  BN 256
      case 63: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 1171, 8, 0, UNRG); break; // BM 128, BN 256
      case 64: R4D_TILED_LAUNCH(2, 8, 5, 1, 2, 0, 1171, 8, 0, UNRG); break; // BM 160, BN 128
      // OPT bit 2048 = hand-scheduled k-block (147|2048 = 2195); |4096 = the finer variant (6291).
      case 70: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 2195, 8, 0, UNRG); break; // BM 96,  BN 128 sched
      case 71: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 6291, 8, 0, UNRG); break; // BM 96,  BN 128 sched/fine
      case 72: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 0, 2195, 8, 0, UNRG); break; // BM 64,  BN 256 sched
      case 73: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 2195, 8, 0, UNRG); break; // BM 96,  BN 256 sched
      case 74: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 2195, 8, 0, UNRG); break; // BM 128, BN 256 sched
      case 75: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 6291, 8, 0, UNRG); break; // BM 128, BN 256 sched/fine
      // ABL 9: the A fragment straight from global in a fragment-major layout, no A LDS at all.
      case 46: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 9, 147, 8, 0, UNRG); break; // BM96 BN128 abl:A global
      case 47: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 9, 147, 8, 0, UNRG); break; // BM64 BN256 abl:A global
      case 48: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 9, 147, 8, 0, UNRG); break; // BM96 BN256 abl:A global
      case 49: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 9, 147, 8, 0, UNRG); break; // BM128 BN256 abl:A global
      case 44: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 8, 147, 8, 0, UNRG); break; // BM96 BN128 swz, iu4 abl
      case 45: R4D_TILED_LAUNCH(2, 8, 2, 2, 2, 8, 147, 8, 0, UNRG); break; // BM64 BN256 swz, iu4 abl
      // The iu4 path (147|512 = 659, BIAS 0), and its like-for-like iu8 twin.  Rejected on the
      // arithmetic above; kept as one pair so the result is reproducible.  Takes the i4x2
      // activation (`w4a8pack.quant_act_i4x2`), NOT the int8 one -- same bytes, two 4-bit planes.
      case 50: R4D_TILED_LAUNCH(2, 8, 3, 2, 0, 0, 659, 8, 0, UNRG); break; // BM 96, BN 256, iu4
      case 51: R4D_TILED_LAUNCH(2, 8, 3, 2, 0, 0, 147, 8, 0, UNRG); break; // the same tile on iu8
      // THE BALANCED TAIL (SKT = 1).  Same tiles, same inner loop; the only difference is that the
      // `tiles mod 32` blocks the grid would otherwise run alone in a last round are split along K
      // and reduced through a workspace.  One instantiation per tile the dispatch table actually
      // picks at a decode M.
      case 90: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 147, 8, 0, UNRG, 1); break; // BM 96, BN 128
      case 91: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 147, 8, 0, UNRG, 1); break; // BM 96, BN 256
      // The inner loop, from the ISA.
      // 8192 = the f32 fold (VOPD-eligible FMAC), 16384 = the A-fragment prefetch, 147|... = both.
      case 110: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 111: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 147 | 16384, 8, 0, UNRG); break;
      case 112: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 147 | 8192 | 16384, 8, 0, UNRG); break;
      // BM = 80 IS THE SMALLEST TILE THAT COVERS M = 72.  The shipped BM = 96 tile does a sixth of
      // its WMMA, its fold and its A reads on m-tile 5, which is entirely past M at eight
      // sequences.  `WM*TM=5` is prime, so one wave deep in M is the only split: TM=5 also drops
      // the per-WMMA unpack from 3/TM = 1.0 to 0.6. 512 threads cannot stage BM = 80: AW = 1280
      // dwords and 1280 % 512 != 0, so the only legal width is 256, which triples the staging per
      // thread.
      case 114: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147, 8, 0, UNRG); break;   // BM 80, BN 128, 256 thr
      case 116: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8192 | 16384, 8, 0, UNRG); break;
      case 117: R4D_TILED_LAUNCH(2, 4, 3, 2, 2, 0, 147 | 8192 | 16384, 8, 0, UNRG); break; // BM96 BN256 256thr
      case 118: R4D_TILED_LAUNCH(1, 8, 6, 1, 2, 0, 147 | 8192 | 16384, 8, 0, UNRG); break; // BM96 BN128 256thr
      case 119: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8192, 8, 0, UNRG); break;  // BM 80, BN 128, f32 fold
      case 120: R4D_TILED_LAUNCH(1, 8, 5, 2, 2, 0, 147, 8, 0, UNRG); break;         // BM 80, BN 256, 256 thr
      case 121: R4D_TILED_LAUNCH(1, 4, 5, 2, 2, 0, 147, 4, 0, UNRG); break;         // BM 80, BN 128, 128 thr
      case 122: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 80, BN 128, WPE 4
      case 123: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 16384, 8, 0, UNRG); break; // BM 80, A prefetch
      case 124: R4D_TILED_LAUNCH(1, 8, 4, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 64, BN 128, 256 thr
      case 125: R4D_TILED_LAUNCH(1, 8, 7, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 112, BN 128, 256 thr
      case 126: R4D_TILED_LAUNCH(1, 8, 10, 1, 2, 0, 147, 8, 0, UNRG); break;        // BM 160, BN 128, 256 thr
      // THE BM FAMILY.  512 threads need BM % 32 == 0, which is why the shipped ladder skips 80 and
      // 112; at 256 threads `<1, 8, TM, 1>` gives BM = 16*TM for every TM, so the block can be the
      // SMALLEST multiple of 16 that covers M instead of the nearest multiple of 32.
      case 127: R4D_TILED_LAUNCH(1, 8, 1, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 16
      case 128: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 32
      // OPT bit 65536 = non-temporal weight load, on the three tiles a decode step picks.
      case 300: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 65536, 8, 0, UNRG); break;  // 114 + wnt
      case 301: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 65536, 8, 0, UNRG); break;  // 128 + wnt
      case 302: R4D_TILED_LAUNCH(1, 8, 3, 1, 2, 0, 147 | 65536, 8, 0, UNRG); break;  // 129 + wnt
      case 303: R4D_TILED_LAUNCH(1, 8, 4, 1, 2, 0, 147 | 65536, 4, 0, UNRG); break;  // 130 + wnt
      case 304: R4D_TILED_LAUNCH(1, 8, 6, 1, 2, 0, 147 | 65536, 4, 0, UNRG); break;  // 131 + wnt
      case 305: R4D_TILED_LAUNCH(2, 8, 5, 1, 2, 0, 147 | 65536, 8, 0, UNRG); break;  // 57  + wnt
      case 306: R4D_TILED_LAUNCH(2, 8, 3, 1, 2, 0, 147 | 65536, 8, 0, UNRG); break;  // BM96 512thr wnt
      // BN=256 HALVES THE COLUMN BLOCKS AND SO THE ACTIVATION RE-READ, which is the term that
      // scales with M (see OPT bit 65536 above).  These are 62/63/117 with the hint.
      case 307: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 1171 | 65536, 8, 0, UNRG); break; // BM96  BN256
      case 308: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 1171 | 65536, 8, 0, UNRG); break; // BM128 BN256
      case 309: R4D_TILED_LAUNCH(2, 4, 3, 2, 2, 0, 147 | 8192 | 16384 | 65536, 8, 0, UNRG); break;
      case 310: R4D_TILED_LAUNCH(1, 8, 2, 2, 2, 0, 147 | 65536, 8, 0, UNRG); break;  // BM32  BN256
      case 311: R4D_TILED_LAUNCH(1, 8, 2, 2, 2, 0, 147, 8, 0, UNRG); break;          // BM32  BN256 no hint
      // The shipped tiles stage the ACTIVATION a dword at a time -- BM=80 needs
      // AW/4 = 320 dwords over 256 threads, which A128 refuses and A128R allows -- and stage the
      // WEIGHT through LDS for a sharing that WM=1 does not have.  320-329 cross the three fixes.
      case 320: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8 | 32768 | 65536, 8, 0, UNRG); break;
      case 321: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 32 | 65536, 8, 0, UNRG); break;
      case 322: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8 | 32 | 32768 | 65536, 8, 0, UNRG); break;
      case 323: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 65536 | 131072, 8, 0, UNRG); break;
      case 324: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8 | 32768 | 65536 | 131072, 8, 0, UNRG); break;
      case 325: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8 | 32 | 32768 | 65536 | 131072, 8, 0, UNRG); break;
      case 326: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 65536 | 131072, 8, 0, UNRG); break;
      case 327: R4D_TILED_LAUNCH(1, 8, 4, 1, 2, 0, 147 | 8 | 65536, 8, 0, UNRG); break;
      case 328: R4D_TILED_LAUNCH(1, 8, 4, 1, 2, 0, 147 | 8 | 65536 | 131072, 8, 0, UNRG); break;
      case 329: R4D_TILED_LAUNCH(1, 8, 3, 1, 2, 0, 147 | 65536 | 131072, 8, 0, UNRG); break;
      // THE COST BREAKDOWN OF THE TILES A DECODE STEP ACTUALLY PICKS.  The ablations above are
      // taken at BM=96/M=72, which is a prefill shape; 330-341 repeat the same variants on <1,8,5>
      // and <1,8,2> at the shipped OPT.  The 3xx variants compute wrong answers by construction --
      // they are timings, not results.
      case 330: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 10, 147 | 65536, 8, 0, UNRG); break;  // no unpack
      case 331: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 11, 147 | 65536, 8, 0, UNRG); break;  // + no W LDS read
      case 332: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 12, 147 | 65536, 8, 0, UNRG); break;  // A read halved
      case 333: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 15, 147 | 65536, 8, 0, UNRG); break;  // A from global
      case 334: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 13, 147 | 65536, 8, 0, UNRG); break;  // epilogue /8
      case 335: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 14, 147 | 65536, 8, 0, UNRG); break;  // no scale load
      case 336: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 12, 147 | 65536, 8, 0, UNRG); break;  // BM32 A read halved
      case 337: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 15, 147 | 65536, 8, 0, UNRG); break;  // BM32 A from global
      case 338: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 1024 | 65536, 8, 0, UNRG); break;
      case 339: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 16384 | 65536, 8, 0, UNRG); break;
      case 340: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 2048 | 65536, 8, 0, UNRG); break;
      case 341: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 8192 | 65536, 8, 0, UNRG); break;
      case 342: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 16384 | 65536 | 131072, 8, 0, UNRG); break;
      case 343: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 2048 | 65536 | 131072, 8, 0, UNRG); break;
      case 344: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 16, 147 | 65536, 8, 0, UNRG); break;  // no barrier
      case 345: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 16, 147 | 65536, 8, 0, UNRG); break;  // BM32 no barrier
      case 346: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 262144 | 65536, 8, 0, UNRG); break;
      case 347: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 262144 | 65536 | 131072, 8, 0, UNRG); break;
      case 348: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 262144 | 65536, 8, 0, UNRG); break;
      case 349: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 14, 147 | 65536, 8, 0, UNRG); break;  // BM32 no scale
      case 350: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 524288 | 65536, 8, 0, UNRG); break;
      case 351: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 524288 | 65536 | 131072, 8, 0, UNRG); break;
      case 352: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147 | 524288 | 65536, 8, 0, UNRG); break;
      case 129: R4D_TILED_LAUNCH(1, 8, 3, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 48
      case 130: R4D_TILED_LAUNCH(1, 8, 4, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 64,  WPE 4
      case 131: R4D_TILED_LAUNCH(1, 8, 6, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 96,  WPE 4
      case 132: R4D_TILED_LAUNCH(1, 8, 7, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 112, WPE 4
      case 133: R4D_TILED_LAUNCH(1, 8, 9, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 144, WPE 4
      case 134: R4D_TILED_LAUNCH(1, 8, 3, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 48,  WPE 4
      case 135: R4D_TILED_LAUNCH(1, 8, 2, 1, 2, 0, 147, 4, 0, UNRG); break;         // BM 32,  WPE 4
      // The same odd BM at 512 threads, which the ragged staging allows.  BN is forced to 256
      // there: WM*WN = 16 and WM*TM = 5 leaves WM = 1, so WN = 16.
      case 136: R4D_TILED_LAUNCH(1, 16, 5, 1, 2, 0, 147, 8, 0, UNRG); break;        // BM 80,  BN 256, 512 thr
      case 137: R4D_TILED_LAUNCH(1, 16, 5, 1, 2, 0, 147, 4, 0, UNRG); break;        // BM 80,  BN 256, WPE 4
      case 138: R4D_TILED_LAUNCH(1, 16, 3, 1, 2, 0, 147, 8, 0, UNRG); break;        // BM 48,  BN 256, 512 thr
      case 139: R4D_TILED_LAUNCH(1, 16, 7, 1, 2, 0, 147, 8, 0, UNRG); break;        // BM 112, BN 256, 512 thr
      // TM = 5 changes the balance the scheduling variants above are aimed at: the loop is 6.30
      // instructions per WMMA instead of 9.21, and five matrix instructions per k step cover the
      // LDS reads that three do not.
      case 140: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 2048, 8, 0, UNRG); break;  // BM 80 + sched
      case 141: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 6144, 8, 0, UNRG); break;  // BM 80 + sched/fine
      case 142: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 1024, 8, 0, UNRG); break;  // BM 80 + LDS-only barrier
      // NARROW N: `out_proj`/`down_proj` are N = 5120, which is 40 tiles at BN = 128 -- 1.25 rounds
      // per M block.  BN = 64 doubles the tile count for the same work.
      case 143: R4D_TILED_LAUNCH(1, 4, 5, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 80, BN 64, 128 thr
      case 144: R4D_TILED_LAUNCH(1, 4, 3, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 48, BN 64, 128 thr
      case 145: R4D_TILED_LAUNCH(2, 4, 5, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 160, BN 64, 256 thr
      // BM = 144 covers M = 144 in one block, which BM 128 and BM 160 both fail to do well.
      case 146: R4D_TILED_LAUNCH(1, 8, 9, 1, 2, 0, 147, 8, 0, UNRG); break;         // BM 144, BN 128, 256 thr
      case 147: R4D_TILED_LAUNCH(1, 16, 9, 1, 2, 0, 147, 8, 0, UNRG); break;        // BM 144, BN 256, 512 thr
      // OPT sweep on the BM=80 tile.  32 = do not stage rows past M (8 of 80 at M=72), 1024 = the
      // LDS-only barrier, 8|16 = 128-bit staging, 64 = the asum prefetch (inert at ASYM=0).
      case 150: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 32, 8, 0, UNRG); break;    // BM 80 + MSKIP
      case 151: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 32 | 1024, 8, 0, UNRG); break;
      case 152: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 155, 8, 0, UNRG); break;         // BM 80 + A 128-bit
      case 153: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 & ~16, 8, 0, UNRG); break;   // BM 80, dword W staging
      case 154: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 256, 8, 0, UNRG); break;   // BM 80 + unpack at staging
      case 155: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147, 8, 0, 1); break;            // BM 80, UNR 1
      case 156: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147, 8, 0, 2 * UNRG); break;     // BM 80, UNR 4
      case 157: R4D_TILED_LAUNCH(1, 8, 5, 1, 1, 0, 147, 8, 0, UNRG); break;         // BM 80, BIAS 1
      case 158: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 0, 147 | 1024 | 32 | 8192, 8, 0, UNRG); break;
      // DCE-PROOF ABLATIONS ON THE BM=80 TILE.  Each computes a WRONG answer on purpose and keeps
      // the data dependency that would otherwise let the compiler delete the path being priced.
      case 160: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 5, 147, 8, 0, UNRG); break;  // no unpack
      case 161: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 6, 147, 8, 0, UNRG); break;  // no unpack, no W LDS read
      case 162: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 7, 147, 8, 0, UNRG); break;  // half the A LDS read
      case 163: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 3, 147, 8, 0, UNRG); break;  // group epilogue at 1/8
      case 164: R4D_TILED_LAUNCH(1, 8, 5, 1, 2, 2, 147, 8, 0, UNRG); break;  // no LDS barrier at all
      // THE iu4 ROUTE ON THE BM=80 TILE.  The DCE-proof ablation puts the unpack at 24.2% here --
      // three instructions per weight dword, sitting between the LDS read and the matrix
      // instruction that consumes it.  `v_wmma_i32_16x16x32_iu4` eats the packed nibbles, so the
      // unpack goes away entirely; what it costs is a SECOND int32 accumulator per output tile,
      // and at TM=5 that is 80 int VGPRs on top of 40 float.  Takes the i4x2 activation.
      // ---------------- PREFILL VARIANTS ----------------
      // At M >= 1024 this kernel is not a weight stream: gate_up_proj is 730 GFLOP against 44.6 MB
      // of weight, so the answer is the matrix pipe.  A `v_wmma_i32_16x16x16_iu8` occupies its SIMD
      // for 8 cycles (NOT the 16 that f16/bf16 take), so the ceiling is 8/(8+r)
      // with r the non-matrix instructions per matrix instruction.  The shipped tile's inner loop
      // is 361 instructions for 64 WMMA, r = 4.64, a 63% ceiling against 53% achieved.
      //
      // The three biggest items in that loop, per scale group: the fold (64 v_fma_mix_f32), the
      // unpack (48 VALU) and the LDS waits (30 s_wait_dscnt).  Bit 8192 is the fold: v_fma_mix_f32
      // is VOP3P and can never dual-issue, and `v_dual_fmac_f32` is the ONLY VOPD form gfx1201
      // offers that this loop can use (v_dual_and_b32 and v_dual_lshlrev_b32 do not exist here,
      // so the unpack cannot be packed at all).
      case 200: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 201: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 8192 | 1024, 8, 0, UNRG); break;
      case 202: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 1024, 8, 0, 2 * UNRG); break;
      case 203: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 8192 | 1024, 8, 0, 2 * UNRG); break;
      case 204: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 8192 | 2048, 8, 0, UNRG); break;
      case 205: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 8192 | 2048 | 4096, 8, 0, UNRG); break;
      // Taller tiles: the unpack is 3 VALU per (n tile, k step) shared over TM matrix instructions,
      // so it is 3/TM per WMMA and TM is the only lever on it.  Bit 32768 is what makes them worth
      // trying -- without ragged 128-bit staging BM = 160 and 192 stage A one dword at a time.
      case 206: R4D_TILED_LAUNCH(2, 8, 5, 2, 2, 0, 147 | 8192 | 32768, 8, 0, UNRG); break;
      case 207: R4D_TILED_LAUNCH(2, 8, 6, 1, 2, 0, 147 | 8192 | 32768, 8, 0, UNRG); break;
      case 208: R4D_TILED_LAUNCH(2, 8, 5, 1, 2, 0, 147 | 8192 | 32768, 8, 0, UNRG); break;
      case 209: R4D_TILED_LAUNCH(2, 4, 6, 2, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 210: R4D_TILED_LAUNCH(4, 4, 4, 2, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 211: R4D_TILED_LAUNCH(2, 16, 4, 2, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 212: R4D_TILED_LAUNCH(2, 8, 3, 2, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 213: R4D_TILED_LAUNCH(1, 8, 4, 2, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 214: R4D_TILED_LAUNCH(2, 8, 8, 1, 2, 0, 147 | 8192, 8, 0, UNRG); break;
      case 215: R4D_TILED_LAUNCH(2, 8, 4, 2, 0, 0, 147 | 8192, 8, 0, UNRG); break;
      // THE SHAPE IS ALREADY RIGHT AT PREFILL: every taller, wider, narrower and more-unrolled tile
      // loses, and the VOPD fold is a wash on K=5120 and a small win on K=3072.  What is left is
      // the inner loop, priced at M=4096 with the DCE-proof ablations: the VALU model says
      // 8*64 + 146 = 658 cycles per 64 matrix instructions, a 78% ceiling, against 55% achieved.
      // These variants say what the missing 23 points are made of.  All of them compute wrong
      // answers by construction.
      case 220: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 10, 147, 8, 0, UNRG); break;  // no unpack
      case 221: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 11, 147, 8, 0, UNRG); break;  // + no W LDS read
      case 222: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 12, 147, 8, 0, UNRG); break;  // half the A LDS read
      case 223: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 13, 147, 8, 0, UNRG); break;  // fold at 1/8
      case 224: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 16, 147, 8, 0, UNRG); break;  // no LDS barrier
      case 225: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 14, 147, 8, 0, UNRG); break;  // no scale load
      case 226: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 15, 147, 8, 0, UNRG); break;  // A from global
      case 227: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 16384, 8, 0, UNRG); break;  // A prefetch
      case 228: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147 | 8192 | 16384, 8, 0, UNRG); break;
      case 229: R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 17, 147, 8, 0, UNRG); break;  // iu4 route, staged
      case 165: R4D_TILED_LAUNCH(1, 8, 5, 1, 0, 0, 659, 8, 0, UNRG); break;  // BM 80, iu4
      case 166: R4D_TILED_LAUNCH(1, 8, 5, 1, 0, 0, 147, 8, 0, UNRG); break;  // the same tile on iu8
      case 167: R4D_TILED_LAUNCH(1, 8, 4, 1, 0, 0, 659, 8, 0, UNRG); break;  // BM 64, iu4
      case 168: R4D_TILED_LAUNCH(1, 8, 3, 1, 0, 0, 659, 8, 0, UNRG); break;  // BM 48, iu4
      // THE DEFAULT: 2x8 waves of 4x2 tiles, the magic-bias accumulator, the raw scale prefetch,
      // 32-bit global indexing and 128-bit staging FOR THE WEIGHT ONLY, at eight waves per EU.
      // At M = 4096 it lands within about ten points of the block-fp8 bar on gate_proj, down_proj
      // and in_proj_qkv, where the 4x4-wave, 2x4-tile, plain-cvt variant (variant 1) is 25-30% over
      // it.
      default:
        if (ufull) R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147, 8, 0, UNRG);
        else R4D_TILED_LAUNCH(2, 8, 4, 2, 2, 0, 147, 8, 0, 1);
        break;
    }
  }
#undef R4D_TILED_LAUNCH
}

void r4d_gemm_w4a8_tiled(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                         long stream, int variant) {
  r4d_gemm_w4a8_tiled_impl(a, ascale, wq, ws, 0, c, M, K, N, stream, variant, false);
}

// Asymmetric grid: the nibble is read UNSIGNED and the stored
// per-group zero is corrected out with a per-row per-group sum of the activation, which
// r4d_quant_act_i8_asum emits as [K/group][Mpad] with Mpad = ceil(M/16)*16.
void r4d_gemm_w4a8_asym_tiled(long a, long ascale, long wq, long ws, long asum, long c, int M,
                              int K, int N, long stream, int variant) {
  if (!asum) throw std::runtime_error("r4d gemm_w4a8_asym_tiled: asum is required");
  r4d_gemm_w4a8_tiled_impl(a, ascale, wq, ws, asum, c, M, K, N, stream, variant, true);
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are in r4d_gemm_args.h. See libr4d/r4d_args.h for why there
// is no adapter directory between the core and these launches.
/* ------------------------------------------------------------------ the large-M kernels
 *
 * These take a single opaque tile VARIANT into libr4d's own tile switch rather than five knobs.
 * The block width BN is a property of the variant, and N must be a multiple of it -- which the ROW
 * cannot express, because BN varies by variant. So the row constrains N % 256 (the default
 * variant's BN) and this is where a variant whose BN does not divide N is refused. */

extern "C" int r4d_rad_gemm_w4a8_tiled(const RadArgs* a, RadStream s) {
    GemmOps g;
    int rc = gemm_ops(a, &g, 1);
    if (rc != RAD_OK) return rc;
    if (!g.ascale) return R4D_NO(RAD_E_INVAL);
    if (g.N % 256) return R4D_NO(RAD_E_SHAPE);                       /* the entry point throws on this */
    if (g.K % R4D_GEMM_W4A8_GROUP) return R4D_NO(RAD_E_SHAPE);
    R4dTileCfg tv;
    r4d_cfg_tiled(a, &tv);
    const R4dTileCfg* t = &tv;
    rc = tile_ok(t, g.N);
    if (rc != RAD_OK) return rc;

    if (is_asym(a)) {
        if (!g.asum) return R4D_NO(RAD_E_INVAL);
        R4D_TRY("gemm_w4a8_asym_tiled",
            r4d_gemm_w4a8_asym_tiled(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b),
                                     r4d_addr(g.bscale), r4d_addr(g.asum), r4d_addr(g.y),
                                     (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s,
                                     t->variant));
        return RAD_OK;
    }
    R4D_TRY("gemm_w4a8_tiled",
        r4d_gemm_w4a8_tiled(r4d_addr(g.a), r4d_addr(g.ascale), r4d_addr(g.b), r4d_addr(g.bscale),
                            r4d_addr(g.y), (int)g.M, (int)g.K, (int)g.N, (long)(uintptr_t)s,
                            t->variant));
    return RAD_OK;
}

/* libr4d defines r4d_gemm_w8a8_asym_tiled but does NOT declare it in r4d.h, so this plugin cannot
 * reach it and `w8a8_asym` is deliberately absent from the row's dtype set. */
