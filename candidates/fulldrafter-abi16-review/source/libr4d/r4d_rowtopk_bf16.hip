// r4d_rowtopk_bf16.hip - r4d_rowtopk_bf16: exact per-row top-R of a wide bf16 matrix.
//
// WHAT IT IS FOR.  A quantised draft head produces coarse logits over a whole vocabulary shard and
// only the best few dozen of them are ever rescored exactly.  Selecting those is not a GEMM
// problem, but doing it with a general top-k costs more than the GEMM saves: torch.topk over a
// 15520-wide candidate array measures 132 us at M=64 in three kernels (digit counts, within-K
// counts, gather), and over a full 124160-wide row it is several times that.
//
// THE SHAPE IS TWO STAGES BECAUSE THE SCAN AND THE SELECTION WANT OPPOSITE THINGS.  Reading the
// matrix wants thousands of waves in flight; the selection is a chain of dependent cross-lane
// steps, ~300 cycles a round, and running that chain thousands of times costs more than the read.
// Two simpler shapes are both far off that on a 15.9 MB matrix whose roofline is 25 us:
//
//     256-thread blocks, R=32 rounds, two barriers a round     75.6 us   (chain, plus barriers)
//     one wave per chunk, R=32 rounds, no barrier              68.5 us   (chain, or too few waves)
//
// Neither is bandwidth-bound.  So stage 1 runs MANY waves and a SHORT chain -- one wave per (row,
// chunk), keeping only the chunk's top-RC -- and stage 2 runs one wave per row over the nch*RC
// survivors with the full R-round chain, where there are only M chains in the whole launch.
//
//   * A LANE KEEPS ITS OWN TOP-PT IN REGISTERS, SORTED.  Insertion is guarded by a single compare
//     against the tail, so the common element costs one instruction; only an element that actually
//     places pays the branch-free shift.  There is no LDS array of candidates.
//   * A ROUND ONLY EVER REMOVES A LIST HEAD.  The round winner is the head of exactly one lane's
//     sorted list, so "remove it" is that lane shifting its own registers.  A mask-and-rescan over
//     a shared candidate array -- the obvious form -- re-reads every candidate each round; this
//     reads none.
//   * ONE WAVE PER BLOCK, so a round is five xor-butterfly steps and NO barrier.
//
// THE KEY IS ONE 64-BIT INTEGER, which is what makes a round a plain max.  The high 32 bits are the
// float's order-preserving image (invert everything if negative, set the sign bit if not), the low
// 32 are ~index, so a larger key means a larger value and, on an exact tie, the LOWER column --
// the tie rule a dense top-k over the row would give.  The index in the low bits also makes every
// key distinct, which is what lets a round identify its winner by equality instead of carrying a
// position alongside the value.
//
// WHERE PT HAS TO BE BIG ENOUGH, AND WHERE IT BIT.  A lane's top-PT filter loses a candidate when
// more than PT of the ones that matter land in that lane's slice, so PT is a statement about how
// the slice is chosen:
//
//   * Stage 1 gives a lane every 32nd eight-column group, and PT must cover a whole VECTOR LOAD --
//     eight bf16 arrive at once, so PT=4 can lose a candidate to a single load.  That is a real
//     failure and not a conservative bound: PT=4 disagrees with torch.topk on smooth data while
//     passing on tied data.  PT=8 makes one load placeable in full, and beyond that a chunk would
//     need nine of its own top-RC in one of 32 slices.
//   * Stage 2's slice is NOT random.  Stage 1 writes chunk-major, so at stride 32 with RC=32 lane
//     0 would receive the maximum of EVERY chunk -- exactly where the row's top-R concentrates --
//     and PT=8 then drops all but eight of them.  That is the second way the filter can lose.  Stage 2
//     therefore carries PT = 32 >= R: a lane can hold the entire answer, so no distribution of the
//     candidates over lanes can lose one.  It is exact by construction, not by argument.
//
// RC IS THE ONE PLACE THE ANSWER CAN BE APPROXIMATE, and it is the caller's choice.  A chunk hands
// on its own top-RC, so the row top-R survives unless MORE THAN RC of it falls in a single chunk.
// RC=R is exact for any data; smaller RC shortens stage 1's chain, which is what stage 1 costs, at
// a risk that falls off fast in the chunk count -- at R=32, RC=8 and 64 chunks it takes nine of
// the thirty-two in one sixty-fourth of the row.  The caller that rescores its candidates exactly
// loses a candidate, not correctness, when that happens.
#include "r4d.h"
#include "r4d_emit.h"
#include "r4d_emit.h"
#include "r4d_gemm_args.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fp8_lm.h"
#include <hip/hip_bf16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_TOPK_WAVE  32
#ifndef R4D_TOPK_PT1
#define R4D_TOPK_PT1   8       // stage 1: >= the eight bf16 a lane loads in one step
#endif
#define R4D_TOPK_PT2   32      // stage 2: >= R, so a lane can hold the whole answer

typedef unsigned u4v_tk __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const u4v_tk* r4d_tk_gv4;

#ifndef R4D_TOPK_WPB
#define R4D_TOPK_WPB 8        // independent waves per block: one chunk each, no barrier between
#endif
#ifndef R4D_TOPK_UNROLL
#define R4D_TOPK_UNROLL 4     // vector loads issued before any of them is consumed
#endif

__device__ __forceinline__ unsigned long long tk_key(unsigned short bf, unsigned idx) {
  unsigned u = (unsigned)bf << 16;
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - idx);
}

__device__ __forceinline__ float tk_val(unsigned long long k) {
  unsigned u = (unsigned)(k >> 32);
  u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
  return __uint_as_float(u);
}

__device__ __forceinline__ int tk_idx(unsigned long long k) {
  return (int)(0xFFFFFFFFu - (unsigned)(k & 0xFFFFFFFFull));
}

// The larger of this lane's key and its partner's at lane ^ N, the partner read through DPP or
// permlanex16 rather than `__shfl_xor` -- see r4d_xor_lane.
template <int N>
__device__ __forceinline__ unsigned long long tk_xor_max(unsigned long long m) {
  const unsigned long long p = r4d_xor_lane_u64<N>(m);
  return p > m ? p : m;
}

// THE ONLY GUARD IN FRONT OF THE INSERT IS ITS OWN, ON THE KEY.  A float guard -- one v_cmp_f32
// against the list tail's value -- is a loss in stage 1, where making it a branch breaks the
// unrolled straight-line body that eight elements per vector load produce and costs about 18% at
// M=64 even on tied data.  And it is wrong anywhere a NaN can arrive: the key orders a positive NaN
// above +inf and a negative one below -inf, while a float compare is false against everything, so
// a float guard drops what the key would keep and, once a NaN is the tail, drops every candidate
// after it.  Every stage here and the merge rank on the key alone, so they agree on one order.
template <int PT>
__device__ __forceinline__ void tk_insert(unsigned long long* t, unsigned long long k) {
  if (k > t[PT - 1]) {
    unsigned long long c = k;
#pragma unroll
    for (int q = 0; q < PT; q++) {
      const unsigned long long hi = t[q] > c ? t[q] : c;
      c = t[q] > c ? c : t[q];
      t[q] = hi;
    }
  }
}

// One reduction round over the wave.  Every lane gets the winner; the one lane that owns it drops
// its head.  No LDS and no barrier -- that is what one wave per block buys.
template <int PT>
__device__ __forceinline__ unsigned long long tk_round(unsigned long long* t) {
  unsigned long long m = t[0];
  m = tk_xor_max<16>(m);
  m = tk_xor_max<8>(m);
  m = tk_xor_max<4>(m);
  m = tk_xor_max<2>(m);
  m = tk_xor_max<1>(m);
  if (m && t[0] == m) {
#pragma unroll
    for (int q = 0; q < PT - 1; q++) t[q] = t[q + 1];
    t[PT - 1] = 0ull;
  }
  return m;
}

template <int PT>
__global__ __launch_bounds__(R4D_TOPK_WAVE * R4D_TOPK_WPB) void r4d_rowtopk_stage1_kernel(
    const unsigned short* __restrict__ Y, unsigned long long* __restrict__ P,
    int N, int RC, int NCH, int CH) {
  // WPB independent waves in one block, each on its own chunk.  They share nothing -- no LDS, no
  // barrier -- and the only reason to group them is that a block of 32 threads spends the CU's
  // workgroup slots, not its wave slots, and runs out of the former first.
  const int m = blockIdx.y;
  const int c = blockIdx.x * R4D_TOPK_WPB + (int)(threadIdx.x >> 5);
  if (c >= NCH) return;
  const int lane = threadIdx.x & 31;
  const int c0 = c * CH;
  int c1 = c0 + CH;
  if (c1 > N) c1 = N;

  unsigned long long t[PT];
#pragma unroll
  for (int q = 0; q < PT; q++) t[q] = 0ull;

  const unsigned short* row = Y + (size_t)m * N;
  // Eight bf16 a lane a step: the wave then moves 512 contiguous bytes per issue, where a scalar
  // ushort load would move 64 for the same instruction.  UNROLL of them are issued before the
  // first is consumed -- the insert chain is a register dependency and would otherwise let one
  // memory latency stand between every load and the next.
  const int v0 = c0 >> 3, v1 = c1 >> 3;
  const int step = R4D_TOPK_WAVE * R4D_TOPK_UNROLL;
  int v = v0 + lane;
  for (; v + step - R4D_TOPK_WAVE < v1; v += step) {
    u4v_tk p[R4D_TOPK_UNROLL];
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++)
      p[u] = *(r4d_tk_gv4)(row + (size_t)(v + u * R4D_TOPK_WAVE) * 8);
#pragma unroll
    for (int u = 0; u < R4D_TOPK_UNROLL; u++) {
      const unsigned base = (unsigned)((v + u * R4D_TOPK_WAVE) * 8);
#pragma unroll
      for (int e = 0; e < 4; e++) {
        tk_insert<PT>(t, tk_key((unsigned short)(p[u][e] & 0xFFFFu), base + 2 * e));
        tk_insert<PT>(t, tk_key((unsigned short)(p[u][e] >> 16), base + 2 * e + 1));
      }
    }
  }
  for (; v < v1; v += R4D_TOPK_WAVE) {
    const u4v_tk p = *(r4d_tk_gv4)(row + (size_t)v * 8);
    const unsigned base = (unsigned)(v * 8);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      tk_insert<PT>(t, tk_key((unsigned short)(p[e] & 0xFFFFu), base + 2 * e));
      tk_insert<PT>(t, tk_key((unsigned short)(p[e] >> 16), base + 2 * e + 1));
    }
  }
  for (int i = (v1 << 3) + lane; i < c1; i += R4D_TOPK_WAVE) {
    tk_insert<PT>(t, tk_key(row[i], (unsigned)i));
  }

  unsigned long long* out = P + ((size_t)m * NCH + c) * RC;
  for (int r = 0; r < RC; r++) {
    const unsigned long long w = tk_round<PT>(t);
    if (lane == 0) out[r] = w;
  }
}

// `off` is the first GLOBAL column this rank's plane stands for, and `OP` is the optional
// (id, value) pair plane an all-gather moves -- eight bytes an entry, an int32 id riding in the
// first float slot, which is the layout core/sample/sampler.h holds as buf_pairs_ and the
// same one the sampler's own vocab-parallel path gathers.  Both are inert at off = 0 with OP null,
// and that is the single-rank launch unchanged to the bit.
//
// AN EMPTY SLOT IS NOT TOKEN `off`.  A chunk that ran out of candidates leaves a zero key; OI has
// always written column 0 for it and keeps doing so, because adding the offset there would make a
// pad indistinguishable from this rank's first real token.  The pair plane says -1 instead, which
// is the sentinel the merge drops -- and a sentinel rather than -INFINITY, because a masked logit
// legitimately takes that value.
template <int PT2>
__global__ __launch_bounds__(R4D_TOPK_WAVE) void r4d_rowtopk_stage2_kernel(
    const unsigned long long* __restrict__ P, float* __restrict__ OV, int* __restrict__ OI,
    float* __restrict__ OP, int off, int R, int n) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x;

  unsigned long long t[PT2];
#pragma unroll
  for (int q = 0; q < PT2; q++) t[q] = 0ull;

  // THE KEY DECIDES, NOT THE FLOAT, as in stage 1 -- see tk_insert.  A chunk that handed on a NaN
  // among its top-RC handed on one fewer real candidate, so dropping the NaN here can leave the row
  // short of R, and the pad that fills the gap is what the merge reports as id -1.  An empty slot's
  // key is 0 and never places.
  const unsigned long long* p = P + (size_t)m * n;
  for (int i = lane; i < n; i += R4D_TOPK_WAVE) tk_insert<PT2>(t, p[i]);

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<PT2>(t);
    if (lane == 0) {
      // A chunk that ran out of candidates wrote a zero key, and a zero key must not read back as
      // a real column: it is an empty slot and says so.
      const float v = w ? tk_val(w) : -INFINITY;
      OV[(size_t)m * R + r] = v;
      OI[(size_t)m * R + r] = w ? (int)tk_idx(w) + off : 0;
      if (OP) {
        float* const q = OP + ((size_t)m * R + r) * 2;
        const int id = w ? (int)tk_idx(w) + off : -1;
        __builtin_memcpy(q, &id, sizeof(int));
        q[1] = v;
      }
    }
  }
}

// Chunks are a multiple of eight columns so stage 1 stays on the vector load, and the last one is
// clamped to N rather than padded -- a padded tail would need a mask on every element.  The caller
// sizes `partial` from this, so the rounding lives in one place.
int r4d_rowtopk_bf16_chunks(int N, int NCH) {
  int CH = ((N + NCH - 1) / NCH + 7) & ~7;
  if (CH < 8) CH = 8;
  return (N + CH - 1) / CH;
}

void r4d_rowtopk_bf16(long y, long partial, long outv, long outi,
                      int M, int N, int R, int RC, int NCH, int PT1, long stream) {
  r4d_rowtopk_bf16_sharded(y, partial, outv, outi, 0, M, N, R, RC, NCH, PT1, 0, stream);
}

// The same top-R over a plane that is one rank's SLICE of a wider one: `off` is the first global
// column the slice stands for, and `outp` -- optional, 0 for none -- is the M*R (int32 id, float
// value) pair plane an all-gather moves so the ranks can agree on a global top-R without ever
// gathering the plane itself.  It is a separate entry point, so the unsharded one keeps its
// signature.
/* The body takes a sink and RETURNS; the exported thrower keeps its signature. */
static int rowtopk_bf16_sharded_emit(long y, long partial, long outv, long outi, long outp,
                                     int M, int N, int R, int RC, int NCH, int PT1, int off,
                                     const R4dSink& sk) {
  int emit_rc = RAD_OK;
  if (M < 1 || N < 1) return -1;
  if (N % 8) return -2;
  // R <= PT2 is what makes stage 2 exact whatever the candidates look like; it is a real limit,
  // not a tuning choice, so it is checked rather than clamped.
  if (R < 1 || R > R4D_TOPK_PT2)
    return -3;
  if (RC < 1 || RC > PT1 * R4D_TOPK_WAVE)
    return -4;
  if (NCH < 1) return -5;
  int CH = ((N + NCH - 1) / NCH + 7) & ~7;
  if (CH < 8) CH = 8;
  const int nch = (N + CH - 1) / CH;
#define R4D_TK_S1(PT) emit_rc = R4D_EMIT(sk, (r4d_rowtopk_stage1_kernel<PT>),               \
                        dim3((nch + R4D_TOPK_WPB - 1) / R4D_TOPK_WPB, M),                        \
                        dim3(R4D_TOPK_WAVE * R4D_TOPK_WPB), 0,                                   \
                        (const unsigned short*)y, (unsigned long long*)partial, N, RC, nch, CH)
  switch (PT1) {
    case 2: R4D_TK_S1(2); break;
    case 4: R4D_TK_S1(4); break;
    case 8: R4D_TK_S1(8); break;
    case 16: R4D_TK_S1(16); break;
    default: return -6;
  }
#undef R4D_TK_S1
  if (emit_rc != RAD_OK) return emit_rc;
#define R4D_TK_S2(PT2) emit_rc = R4D_EMIT(sk, (r4d_rowtopk_stage2_kernel<PT2>), dim3(M), dim3(R4D_TOPK_WAVE), 0, \
                                          (const unsigned long long*)partial, (float*)outv, (int*)outi,          \
                                          (float*)outp, off, R, nch * RC)
  if (R <= 1) {
    R4D_TK_S2(1);
  } else if (R <= 2) {
    R4D_TK_S2(2);
  } else if (R <= 4) {
    R4D_TK_S2(4);
  } else if (R <= 8) {
    R4D_TK_S2(8);
  } else if (R <= 16) {
    R4D_TK_S2(16);
  } else {
    R4D_TK_S2(32);
  }
#undef R4D_TK_S2
  return emit_rc;
}

void r4d_rowtopk_bf16_sharded(long y, long partial, long outv, long outi, long outp,
                              int M, int N, int R, int RC, int NCH, int PT1, int off,
                              long stream) {
  const int rc = rowtopk_bf16_sharded_emit(y, partial, outv, outi, outp, M, N, R, RC, NCH, PT1,
                                           off, r4d_stream_sink((RadStream)(uintptr_t)stream));
  if (rc != RAD_OK) throw std::runtime_error("r4d rowtopk_bf16: refused or failed to launch");
}

// ============================================================ the vocab-parallel merge
//
// The other half of a sharded top-R.  Each rank ran the two stages above over ITS OWN slice of the
// plane and wrote an [M, R, 2] pair plane -- an int32 global id beside a float value; an all-gather
// concatenated those into [M, W*R, 2] with rank r's at [r*R, (r+1)*R).  This picks the R largest of
// the W*R and writes them in the same (idx, val) form stage 2 writes, so nothing downstream can
// tell which path produced them.
//
// THE KEY IS THE SAME 64-BIT INTEGER, built from the float's bits rather than a bf16's, so a round
// is still a plain max and ties still break on the LOWER id -- which is what makes this equal to a
// top-R over the unsharded plane rather than merely close to it.  A pad carries id -1 and is
// skipped: a sentinel, because -INFINITY is a value a masked logit legitimately takes.
//
// One wave a row, W*R entries -- 32 at world 2 and R 16 -- so this is a launch with a rounding
// error attached, and its only real requirement is that it be exact.
__device__ __forceinline__ unsigned long long tk_key_f32(float v, unsigned id) {
  unsigned u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - id);
}

__global__ __launch_bounds__(R4D_TOPK_WAVE) void r4d_rowtopk_merge_kernel(
    const float* __restrict__ G, float* __restrict__ OV, int* __restrict__ OI, int R, int n,
    int ld = 0) {
  const int m = blockIdx.x;
  const int lane = threadIdx.x;
  const int out_stride = (ld > 0) ? ld : R;

  unsigned long long t[R4D_TOPK_PT2];
#pragma unroll
  for (int q = 0; q < R4D_TOPK_PT2; q++) t[q] = 0ull;

  const float* const row = G + (size_t)m * n * 2;
  for (int i = lane; i < n; i += R4D_TOPK_WAVE) {
    int id;
    __builtin_memcpy(&id, row + (size_t)i * 2, sizeof(int));
    if (id < 0) continue;
    tk_insert<R4D_TOPK_PT2>(t, tk_key_f32(row[(size_t)i * 2 + 1], (unsigned)id));
  }

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<R4D_TOPK_PT2>(t);
    if (lane == 0) {
      OV[(size_t)m * out_stride + r] = w ? tk_val(w) : -INFINITY;
      OI[(size_t)m * out_stride + r] = w ? tk_idx(w) : -1;
    }
  }
}

void r4d_rowtopk_merge(long gathered, long outv, long outi, int M, int R, int W, long stream,
                       int ld) {
  if (M < 1 || R < 1 || W < 1) throw std::runtime_error("r4d rowtopk_merge: M, R and W must be >= 1");
  // R <= PT2 is what makes this exact whatever the candidates look like -- a lane can hold the
  // whole answer -- so it is checked rather than clamped, the same as stage 2's.
  if (R > R4D_TOPK_PT2) throw std::runtime_error("r4d rowtopk_merge: R must be 1..32");
  hipLaunchKernelGGL(r4d_rowtopk_merge_kernel, dim3(M), dim3(R4D_TOPK_WAVE), 0,
                     (hipStream_t)stream, (const float*)gathered, (float*)outv, (int*)outi,
                     R, W * R, ld);
}

// ============================================================== logit_rerank
//
// THE EXACT SECOND HALF OF A COARSE TOP-R, which is what this file's header already says the top-R
// is for: "only the best few dozen of them are ever rescored exactly".  The coarse head proposes R
// candidates a row and this scores those R -- and only those R -- against the FULL-PRECISION head,
// then re-sorts on the exact values.  A draft round reads R rows of the weight instead of all of
// it: 40 KB a row against 318 MB.  On Qwen3.8-Flash-Next the 2-bit head accepts 4 to 8 points
// below the exact one, and that gap is what this closes.
//
// ONE BLOCK A ROW, THE WAVES SPREAD OVER THE CANDIDATES.  A wave walks K with a lane stride and
// folds with the same five xor-butterfly steps the selection uses, so the reduction order is a fixed
// function of the shape and the score is reproducible.  Then wave 0 runs the ORDINARY top-R chain
// over the R scores -- one lane a candidate, R <= PT2 -- so the tie rule is tk_key_f32's and is the
// same rule row_topk and row_topk_merge break on.  A tie inside a shard and a tie between shards
// therefore resolve alike, which is the property that makes the merged answer exact.
//
// A CANDIDATE OUTSIDE THIS RANK'S ROWS IS A PAD.  `voff` is the shard's first column and the ids
// are GLOBAL, because that is what row_topk reports; under vocab-parallel drafting every id a rank
// is handed came from its own plane, and the guard is for a caller that mixes them.
//
// I8 READS THE INT8 HEAD logits_gemm_i8 stores (r4d.i8lm.r128): the same row -- K codes, then the
// row's own bf16 scale per 128 columns in its tail -- with each code an int8 value rather than an
// E4M3 one. The drafter's rescoring reads whichever head the trunk's logits GEMM reads, so a model
// with an int8 head needs this form or its lm_head has no device reader here.
#define R4D_RERANK_WPB 8       // waves a block: 256 threads, and R <= 32 so a wave takes <= 4

__device__ __forceinline__ float tk_bf2f(unsigned short b) {
  return __uint_as_float((unsigned)b << 16);
}

template <bool I8>
__global__ __launch_bounds__(R4D_TOPK_WAVE * R4D_RERANK_WPB) void r4d_logit_rerank_kernel(
    const unsigned short* __restrict__ X, const int* __restrict__ IN,
    const unsigned char* __restrict__ W, int* __restrict__ OI, float* __restrict__ OV,
    float* __restrict__ OP, int R, long K, long voff, long nv, long w_ld, int R_pairs = 0) {
  __shared__ float s_sc[R4D_TOPK_PT2];
  __shared__ int   s_id[R4D_TOPK_PT2];

  const int  m    = blockIdx.x;
  const int  lane = threadIdx.x & (R4D_TOPK_WAVE - 1);
  const int  wv   = threadIdx.x / R4D_TOPK_WAVE;
  const unsigned short* const xr = X + (size_t)m * K;
  const int  np   = (R_pairs > 0) ? R_pairs : R;

  for (int r = wv; r < R; r += R4D_RERANK_WPB) {
    const int  gid = IN[(size_t)m * R + r];
    const long loc = (long)gid - voff;
    const bool own = gid >= 0 && loc >= 0 && loc < nv;
    float acc = 0.0f;
    if (own) {
      /* THE STORED ROW, exactly as logits_gemm_fp8 reads it (r4d_fp8_lm.h): K E4M3 codes then
       * K/128 bf16 block scales in the row's own tail. A lane owns a strided subset of the
       * columns and applies each column's own block scale locally, so the fold below is a plain
       * sum and the order is fixed by the lane count alone. */
      const unsigned char* const wr = W + (size_t)loc * w_ld;
      const unsigned short* const sr = (const unsigned short*)(wr + K);
      const int n_blocks = (int)(K / 128);
      for (int b = 0; b < n_blocks; ++b) {
        const float s = r4d_fp8::bscale(sr[b]);
        const long col = (long)b * 128 + lane * 4;
        const uint32_t w4 = *(const uint32_t*)(wr + col);
        const uint2 x4 = *(const uint2*)(xr + col);
        const float x0 = tk_bf2f((unsigned short)x4.x);
        const float x1 = __uint_as_float(x4.x & 0xFFFF0000u);
        const float x2 = tk_bf2f((unsigned short)x4.y);
        const float x3 = __uint_as_float(x4.y & 0xFFFF0000u);
        float dot;
        if (I8) {
          const int8_t c0 = (int8_t)(w4 & 0xFFu);
          const int8_t c1 = (int8_t)((w4 >> 8) & 0xFFu);
          const int8_t c2 = (int8_t)((w4 >> 16) & 0xFFu);
          const int8_t c3 = (int8_t)(w4 >> 24);
          dot = (float)c0 * x0 + (float)c1 * x1 + (float)c2 * x2 + (float)c3 * x3;
        } else {
          const v2f w_lo = r4d_fp8::e4m3_pk<false>(w4);
          const v2f w_hi = r4d_fp8::e4m3_pk<true>(w4);
          dot = w_lo.x * x0 + w_lo.y * x1 + w_hi.x * x2 + w_hi.y * x3;
        }
        acc = fmaf(dot, s, acc);
      }
      acc = r4d_wave_sum(acc);
    }
    if (lane == 0) { s_sc[r] = own ? acc : -INFINITY; s_id[r] = own ? gid : -1; }
  }
  __syncthreads();
  if (wv != 0) return;

  unsigned long long t[R4D_TOPK_PT2];
#pragma unroll
  for (int q = 0; q < R4D_TOPK_PT2; q++) t[q] = 0ull;
  if (lane < R && s_id[lane] >= 0)
    tk_insert<R4D_TOPK_PT2>(t, tk_key_f32(s_sc[lane], (unsigned)s_id[lane]));

  for (int r = 0; r < R; r++) {
    const unsigned long long w = tk_round<R4D_TOPK_PT2>(t);
    if (lane == 0) {
      const float v = w ? tk_val(w) : -INFINITY;
      const int   i = w ? tk_idx(w) : -1;
      OV[(size_t)m * R + r] = v;
      OI[(size_t)m * R + r] = i;
      if (OP && r < np) {
        float* const p = OP + ((size_t)m * np + r) * 2;
        __builtin_memcpy(p, &i, sizeof i);
        p[1] = v;
      }
    }
  }
}

static int logit_rerank_emit(long x, long idx_in, long head, long outi, long outv, long outp,
                             int M, int R, long K, long voff, long nv, const R4dSink& sk,
                             bool i8 = false, int R_pairs = 0) {
  const long w_ld = r4d_fp8::lm_row_bytes(K);
  if (M < 1 || R < 1 || K < 1 || nv < 1) return -1;
  // R <= PT2 is what makes the re-sort exact whatever the scores look like -- one lane holds the
  // whole answer -- so it is checked rather than clamped, the same as the merge's.
  if (R > R4D_TOPK_PT2) return -2;
#define R4D_RERANK_EMIT(I8)                                                                     \
  R4D_EMIT(sk, r4d_logit_rerank_kernel<I8>, dim3(M), dim3(R4D_TOPK_WAVE * R4D_RERANK_WPB), 0,    \
           (const unsigned short*)x, (const int*)idx_in, (const unsigned char*)head, (int*)outi,  \
           (float*)outv, (float*)outp, R, K, voff, nv, w_ld, R_pairs)
  const int rc = i8 ? R4D_RERANK_EMIT(true) : R4D_RERANK_EMIT(false);
#undef R4D_RERANK_EMIT
  return rc;
}

void r4d_logit_rerank(long x, long idx_in, long head, long outi, long outv, long outp,
                      int M, int R, long K, long voff, long nv, long stream, int R_pairs) {
  if (logit_rerank_emit(x, idx_in, head, outi, outv, outp, M, R, K, voff, nv,
                        r4d_stream_sink((RadStream)(uintptr_t)stream), false, R_pairs) != RAD_OK)
    throw std::runtime_error("r4d logit_rerank: refused or failed to launch");
}

// ============================================================== the plugin ABI entry points
//
// The operand reader and the split law are r4d_gemm_args.h. See libr4d/r4d_args.h for why there is
// no adapter directory between the core and these launches.
/* ------------------------------------------------------------------ row top-k */

enum { T_X = 0, T_IDX, T_VAL, T_PAIRS };

/* R > 32 is refused here, not by a constraint: libr4d's row says R <= 1024 in both its predicate
 * and its prose, and the entry point throws above R4D_TOPK_PT2 == 32, calling that "a real limit,
 * not a tuning choice". The constraint table transcribes what libr4d wrote and this is the
 * per-call half that makes the pair sound. */
static int topk_params(const RadArgs* a,
                       const RadTensor** x, const RadTensor** idx, const RadTensor** val,
                       long long* M, long long* N, long long* R, int* RC, int* NCH, int* PT1) {
    *x   = r4d_opt(a, T_X);
    *idx = r4d_opt(a, T_IDX);
    *val = r4d_opt(a, T_VAL);
    /* THE SCRATCH SIZER RUNS AT DECLARE, WHERE THERE ARE NO TENSORS. It is the same entry point as
     * the launch, so reading the geometry out of `x` alone would make sizing the partial plane
     * ahead of a step fail with "invalid argument" -- which reads as a refusal by the kernel and
     * is actually an absent operand. `M` and `N` are declared parameters of this op for exactly
     * this reason (spec §2.1), so they are the fallback; when the tensor IS present it still wins,
     * because a band ceiling is an upper bound and the tensor is the row count of the call. */
    if (*x) {
        if ((*x)->rank != 2 || !r4d_contig(*x)) return R4D_NO(RAD_E_STRIDE);
        *M = (*x)->shape[0];
        *N = (*x)->shape[1];
    } else {
        *M = r4d_geti_or(a, "M", 0);
        *N = r4d_geti_or(a, "N", 0);
    }
    *R = r4d_geti_or(a, "R", 0);
    if (*M < 1 || *N < 1) return R4D_NO(RAD_E_SHAPE);
    if (*N % 8) return R4D_NO(RAD_E_SHAPE);
    if (*R < 1 || *R > 32) return R4D_NO(RAD_E_SHAPE);

    R4dTopkCfg tk;
    r4d_cfg_topk(a, &tk);
    const R4dTopkCfg* c = &tk;
    *PT1 = c->pt1;
    if (*PT1 != 2 && *PT1 != 4 && *PT1 != 8 && *PT1 != 16) return R4D_NO(RAD_E_INVAL);
    *RC  = c->rc > 0 ? c->rc : (int)*R;
    if (*RC < (int)*R) return R4D_NO(RAD_E_INVAL);          /* a chunk must keep at least R */
    if (*RC > *PT1 * 32) return R4D_NO(RAD_E_SHAPE);
    /* NCH is a REQUEST for a chunk COUNT. 0 means "one chunk per 2048 columns, at least one",
     * which is the shape the MoE router this kernel serves actually runs. */
    *NCH = c->nch > 0 ? c->nch : (int)((*N + 2047) / 2048);
    if (*NCH < 1) *NCH = 1;
    return RAD_OK;
}

extern "C" int64_t r4d_rad_rowtopk_scratch(const RadArgs* a) {
    const RadTensor *x, *idx, *val;
    long long M, N, R;
    int RC, NCH, PT1;
    int rc = topk_params(a, &x, &idx, &val, &M, &N, &R, &RC, &NCH, &PT1);
    if (rc != RAD_OK) return rc;
    /* The chunk width is rounded up to a multiple of eight columns inside libr4d, so the count is
     * asked for rather than recomputed -- "the caller sizes `partial` from this, so the rounding
     * lives in one place". */
    const int nch = r4d_rowtopk_bf16_chunks((int)N, NCH);
    return (int64_t)M * nch * RC * (int64_t)sizeof(unsigned long long);
}

static int rad_rowtopk_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor *x, *idx, *val;
    long long M, N, R;
    int RC, NCH, PT1;
    int rc = topk_params(a, &x, &idx, &val, &M, &N, &R, &RC, &NCH, &PT1);
    if (rc != RAD_OK) return rc;
    if (!x || !idx || !val) return R4D_NO(RAD_E_INVAL);
    if (!r4d_contig(idx) || !r4d_contig(val)) return R4D_NO(RAD_E_STRIDE);

    const int nch = r4d_rowtopk_bf16_chunks((int)N, NCH);
    const int64_t need = (int64_t)M * nch * RC * (int64_t)sizeof(unsigned long long);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);

    /* The vocab-parallel form, and both halves of it are optional: `vocab_off` shifts the columns
     * reported so a rank's slice reports global ids, and `pairs` is the [M, R, 2] plane an
     * all_gather moves. Absent and zero is the single-rank launch, bit for bit. */
    const RadTensor* pairs = r4d_opt(a, T_PAIRS);
    const long long off = r4d_geti_or(a, "vocab_off", 0);
    if (off < 0) return R4D_NO(RAD_E_SHAPE);
    if (pairs) {
        if (pairs->dtype != RAD_F32 || !r4d_contig(pairs)) return R4D_NO(RAD_E_STRIDE);
        if (r4d_numel(pairs) < M * R * 2) return R4D_NO(RAD_E_SHAPE);
    }

    if (rowtopk_bf16_sharded_emit(r4d_addr(x), (long)(uintptr_t)a->scratch, r4d_addr(val),
                                  r4d_addr(idx), pairs ? r4d_addr(pairs) : 0,
                                  (int)M, (int)N, (int)R, RC, NCH, PT1, (int)off, sk) != RAD_OK)
        return R4D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
R4D_ENTRY(r4d_rad_rowtopk, rad_rowtopk_body)

/* ------------------------------------------------------------------ row_topk_merge
 *
 * The R largest of the all-gathered per-rank candidate sets. No variants and no scratch: it is one
 * wave a row over world_size * R pairs -- 32 of them at world 2 and R 16 -- so what it costs is its
 * launch, and the only thing that matters about it is that it be exact. */
extern "C" int r4d_rad_rowtopk_merge(const RadArgs* a, RadStream s) {
    const RadTensor* g   = r4d_opt(a, 0);
    const RadTensor* idx = r4d_opt(a, 1);
    const RadTensor* val = r4d_opt(a, 2);
    if (!g || !idx || !val) return R4D_NO(RAD_E_INVAL);
    if (g->dtype != RAD_F32 || !r4d_contig(g)) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(idx) || !r4d_contig(val)) return R4D_NO(RAD_E_STRIDE);

    /* M COMES OFF THE OPERAND, not off the parameter. `M` is declared as a RANGE -- a band ceiling
     * -- and what the parameter table carries is that ceiling, not the row count of this call; the
     * gathered plane is row-narrowed to the rows the step actually has. Reading the parameter
     * sized the checks below at the band and refused every real issue, which is the same trap
     * `topk_params` above documents for the top-R's own scratch hook. The parameter is still the
     * fallback, for a sizing query made before there are tensors. */
    const long long M = (g->rank >= 1) ? g->shape[0] : r4d_geti_or(a, "M", 0);
    const long long R = r4d_geti_or(a, "R", 0);
    const long long W = r4d_geti_or(a, "world_size", 0);
    const int ld = (idx && idx->rank >= 2 && idx->shape[1] > R) ? (int)idx->shape[1] : (int)R;
    if (M < 1 || R < 1 || W < 1) return R4D_NO(RAD_E_SHAPE);
    if (R > 32) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(g) < M * W * R * 2) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(idx) < M * ld || r4d_numel(val) < M * ld) return R4D_NO(RAD_E_SHAPE);

    R4D_TRY("rowtopk_merge",
        r4d_rowtopk_merge(r4d_addr(g), r4d_addr(val), r4d_addr(idx), (int)M, (int)R, (int)W,
                          (long)(uintptr_t)s, ld));
    return RAD_OK;
}

/* `M` and the candidate count come off the OPERANDS for the reason rowtopk_merge's do: `M` is a
 * band ceiling and a step narrows it. `n_vocab` is THIS RANK'S rows of the head, which is what
 * makes the ownership guard a shard test rather than a vocabulary one. */
static int rad_logit_rerank(const RadArgs* a, const R4dSink& sk, bool i8) {
    const RadTensor* x   = r4d_opt(a, 0);
    const RadTensor* w   = r4d_opt(a, 1);
    const RadTensor* in  = r4d_opt(a, 2);
    const RadTensor* idx = r4d_opt(a, 3);
    const RadTensor* val = r4d_opt(a, 4);
    const RadTensor* pr  = r4d_opt(a, 5);
    if (!x || !in || !w || !idx || !val) return R4D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    /* THE DECLARED DTYPE IS WHAT ARRIVES, and for this weight it is the bf16 the trunk
     * declared -- the E4M3 plane is the STORED form. logits_gemm_fp8 says the same thing
     * about the stored WIDTH: it cannot be checked here and is derived by the same rule
     * the layout hook used, and what guarantees the two agree is the engine's layout-tag
     * check, which refuses a container laid out for any other row before a byte of it
     * moves. So the head is accepted at either spelling and read as r4d.fp8lm.b128 -- or, for
     * the int8 form, as r4d.i8lm.r128. */
    if (w->dtype != (i8 ? RAD_I8 : RAD_F8E4M3) && w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (in->dtype != RAD_I32 || idx->dtype != RAD_I32 || val->dtype != RAD_F32)
        return R4D_NO(RAD_E_DTYPE);
    if (pr && pr->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(x) || !r4d_contig(in) || !r4d_contig(w) || !r4d_contig(idx) ||
        !r4d_contig(val) || (pr && !r4d_contig(pr))) return R4D_NO(RAD_E_STRIDE);

    const long long K    = r4d_geti_or(a, "n_embd", 0);
    const long long NV   = r4d_geti_or(a, "n_vocab", 0);
    const long long R    = r4d_geti_or(a, "R", 0);
    const long long voff = r4d_geti_or(a, "vocab_off", 0);
    const int R_pairs    = (pr && pr->rank >= 2 && pr->shape[1] > 0) ? (int)pr->shape[1] : (int)R;
    if (K < 1 || NV < 1 || R < 1) return R4D_NO(RAD_E_SHAPE);
    if (R > 32) return R4D_NO(RAD_E_SHAPE);
    const long long M = (x->rank >= 1) ? x->shape[0] : r4d_geti_or(a, "M", 0);
    if (M < 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(x) < M * K || r4d_numel(in) < M * R) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(idx) < M * R || r4d_numel(val) < M * R) return R4D_NO(RAD_E_SHAPE);
    if (pr && r4d_numel(pr) < M * R_pairs * 2) return R4D_NO(RAD_E_SHAPE);
    if (K % 128) return R4D_NO(RAD_E_SHAPE);   /* the scale block; lm_row_bytes assumes it */

    if (logit_rerank_emit(r4d_addr(x), r4d_addr(in), r4d_addr(w), r4d_addr(idx), r4d_addr(val),
                          pr ? r4d_addr(pr) : 0, (int)M, (int)R, (long)K, (long)voff, (long)NV,
                          sk, i8, R_pairs) != RAD_OK)
        return R4D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
static int rad_logit_rerank_body(const RadArgs* a, const R4dSink& sk) {
    return rad_logit_rerank(a, sk, false);
}
R4D_ENTRY(r4d_rad_logit_rerank, rad_logit_rerank_body)
static int rad_logit_rerank_i8_body(const RadArgs* a, const R4dSink& sk) {
    return rad_logit_rerank(a, sk, true);
}
R4D_ENTRY(r4d_rad_logit_rerank_i8, rad_logit_rerank_i8_body)

