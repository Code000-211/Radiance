// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip -- the gated-delta-net recurrent state update, which is what
// decode runs where prefill runs the chunked scan. One kernel for the whole step:
//
//     g = -exp(A_log) . softplus(a + dt_bias)      beta = sigmoid(b)      q, k l2-normalised
//     S  *= e^g
//     u   = beta . (v - S k)                       S += u k^T              o = S q
//
// per token, against a PAGED state cache: the initial state is read from the slot the last
// accepted token wrote, and every candidate token writes its own slot, because which of them
// survives verification is not known until after this layer has run. It replaces FLA's
// fused_sigmoid_gating_delta_rule_update, and its non-speculative sibling falls out as the
// width-1 case.
//
// THE STATE TRAFFIC IS THE WHOLE COST. One [V,K] state is 64 KB, and a speculative step writes
// one per candidate token per head: at width 9 that is 113 MB per layer per rank, at the
// bandwidth limit of this part. So this kernel is a port that removes a Triton dependency and
// a kernel launch, not an arithmetic win, and it is written to keep the state resident rather
// than to be clever: one thread owns one v row and half its k range, so
//
//   * S k and S q are in-lane dot products closed by a single lane<->lane^1 exchange, not a
//     tree reduction;
//   * the outer-product update is 64 independent FMAs with no cross-lane traffic at all;
//   * a thread's 64 state elements are contiguous in k, so the load and every store is four
//     16-byte accesses over one 256-byte run.
//
// The state cache is fp32 -- this model's HF config sets mamba_ssm_dtype=float32, which vLLM
// honours over the `auto` mamba dtype, so the temporal state is fp32 while the conv state is
// bf16. That is also what makes the state traffic the cost: one [V,K] state is 64 KB, not 32.
//
// The slot and head strides are ARGUMENTS, not H*V*K: vLLM pads the mamba page so it matches the
// attention page size (here 421888 elements per slot against the 393216 the shape implies), and a
// kernel that derives the stride from the shape reads every slot but the first at the wrong
// offset -- which looks like a correct kernel producing noise.
#ifdef __FAST_MATH__
#error "This unit must not use fast math: the shared-norm bit-identity argument needs -ffp-contract=off."
#endif

#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_args.h"
#include "r4d_gdn_wmma.h"
#include "r4d_common.h"
#include "r4d_gdn_conv_tap.h"
#include "r4d_q8.h"         // the 8-bit codes the folded quantiser writes: E4M3 or int8
#include "r4d_emit.h"      // R4D_EMIT: one body, both a dispatch and a description of one
#include <cstdlib>
#include <cstring>

// The plugin ABI exports are the only visible symbols of this DSO. Everything the arithmetic
// body pulls in from r4d_common.h / r4d_gdn_wmma.h is a device template or an inline helper and
// stays internal, so the loader sees exactly the six entry points rad_plugin.h defines and the
// linker's version script can name them.
#pragma GCC visibility push(hidden)

#define RU_K   128              // head_k
#define RU_V   128              // head_v
#define RU_HK   64              // k elements per thread (RU_K / 2)

// THE STATE TRAFFIC IS DEAD ON ARRIVAL AND IT TAXES THE NEXT GEMM.  A speculative step writes one
// full [128,128] fp32 state per candidate token per head -- 12 MB a layer, ~660 MB a step -- and
// exactly one state per head is read again, a whole step later.  None of it can live in a 64 MB
// last level, so the dirty lines are written back while the layer's out_proj is streaming its
// weight, and that GEMM pays for it: behind a matched 12 MB write flood the same (5120, 3072) GEMM
// goes from 14.9 us clean to 20.4 us, and to 18.4 us when the flood is marked non-temporal.  The
// hint changes no byte of the result -- the line stays coherent, it is only evicted first -- and
// whether it pays depends on the flood size, so it is a runtime switch (see RU_NT_MB below).  It
// applies to the state LOAD as well: the initial state is read once per layer per step and never
// again.
typedef float ru_f4 __attribute__((ext_vector_type(4)));
typedef _Float16 ru_h8 __attribute__((ext_vector_type(8)));

// SDT is the STATE CACHE's element type -- 0 = fp32, 1 = fp16 -- and nothing else: every value
// this kernel computes with is fp32 either way.  It exists because the state read and the state
// write ARE this kernel.  At the shape a decode step runs (eight sequences, eight candidate
// tokens) a call is ~68 us, of which the token loop is ~3.1 us a token and the rest -- ~45 us --
// is 24 MB of state moving at 88% of what a launch that size can reach.  Nothing else in the
// kernel is worth a percent, and halving the state width is worth a third.
//
// ---- THE COST MODEL.  Draft depth is the one axis that moves the token count without moving the
// state traffic: the anchor writes ONE state a sequence whatever the block width.  Fitting a call
// at eight sequences and 16k of context across two depths separates the two terms cleanly:
//
//     ~3.2 us per token walked, on top of ~57 us fixed per layer
//
// so a speculative step is simply the fixed cost plus about eleven token iterations.
//
// AND THE TWO HALVES OVERLAP.  Halving the state bytes (the fp16 cache below) is worth about a
// fifth of what the byte count alone predicts, so the state traffic hides under the token loop
// rather than running before it, while the depth split says the loop IS on the critical path.
// Shrinking either half alone therefore buys little, and the loop is the delta rule itself: four
// passes over 64 registers, not three.
//
// Resource use of the shipped instantiation: 207 VGPRs, 7 waves/SIMD, ZERO spill, 384 B LDS.  At
// seven waves a dependent FMA chain is hidden, so the serial `dot +=` and `on +=` accumulations are
// throughput and not latency -- splitting them into partial sums would move the summation order,
// and the text, for nothing.
//
// fp16 and not bf16 because the state is O(1) with no range problem and eleven mantissa bits are
// four more than bf16's: the elements are sums of beta.(v - Sk) k^T over an effective horizon of
// order 1/(1-e^g) with |k| l2-normalised, which puts them around 0.1 and the round-trip error at
// 5e-4 relative.  vLLM allocates the cache from --mamba-ssm-cache-dtype, which already accepts
// float16, so this is a serve flag rather than a private format the prefill path has to invent.
template <int SDT> struct ru_state { using T = float; };
template <> struct ru_state<1>     { using T = _Float16; };

// Eight elements a time: two 16-byte accesses in fp32 and ONE in fp16, so the fp16 cache halves
// the access count as well as the bytes.
template <int NT, int SDT>
__device__ __forceinline__ void ru_load8(float* d, const typename ru_state<SDT>::T* p) {
  if constexpr (SDT) {
    const ru_h8 t = NT ? __builtin_nontemporal_load((const ru_h8*)p) : *(const ru_h8*)p;
#pragma unroll
    for (int e = 0; e < 8; ++e) d[e] = (float)t[e];
  } else {
    const ru_f4* q = (const ru_f4*)p;
    *(ru_f4*)d       = NT ? __builtin_nontemporal_load(q)     : *q;
    *(ru_f4*)(d + 4) = NT ? __builtin_nontemporal_load(q + 1) : *(q + 1);
  }
}

template <int NT, int SDT>
__device__ __forceinline__ void ru_store8(typename ru_state<SDT>::T* p, const float* d) {
  if constexpr (SDT) {
    ru_h8 t;
#pragma unroll
    for (int e = 0; e < 8; ++e) t[e] = (_Float16)d[e];
    if (NT) __builtin_nontemporal_store(t, (ru_h8*)p);
    else    *(ru_h8*)p = t;
  } else {
    ru_f4* q = (ru_f4*)p;
    if (NT) {
      __builtin_nontemporal_store(*(const ru_f4*)d,       q);
      __builtin_nontemporal_store(*(const ru_f4*)(d + 4), q + 1);
    } else {
      *q       = *(const ru_f4*)d;
      *(q + 1) = *(const ru_f4*)(d + 4);
    }
  }
}
// The body still carries the VS axis -- VS > 1 splits a head's v rows across workgroups for the
// unfoldable shapes -- but this row instantiates VS == 1 only, which is what the fused norm and
// the folded convolution both require. No launch here picks another value.
__device__ __forceinline__ float ru_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

// a and b are a chunk() of the ba projection: bf16, strided by 2*H. See r4d_gdn_conv_w4_h128_bf16.hip.
template <int ABF16>
__device__ __forceinline__ float ru_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

// The layer's gated RMS norm, folded into the epilogue. A workgroup here owns one (sequence,
// head) and therefore the WHOLE 128-channel row of the output, which is exactly what an RMS needs
// -- so on this path the norm costs one workgroup reduction per token and no kernel launch, no
// second pass over the output, and no read of it back from HBM. The prefill path cannot do this
// (its workgroup owns half a row) and calls r4d_gdn_gated_rmsnorm_h128_bf16 instead.
__device__ __forceinline__ float ru_act(float z, int kind) {
  const float s = 1.0f / (1.0f + __expf(-z));
  return kind ? s : z * s;
}

// One token's replay factors, inside the head's block of state slot sidx[n][1]:
//
//     [0    .. 4*K) normalised k, FOUR copies -- one per fixed 32-row group
//     [4*K  .. 4*K+V) the v row, one float per v row
//     [4*K+V .. +8)   decay and beta, one pair per row group
//
// The k copies are the price of not needing a barrier ACROSS workgroups. The launcher splits a
// head's v rows over up to four workgroups, and the split changes with the batch size, so a single
// shared copy would be read by one workgroup while another was overwriting it. Fixed 32-row groups
// make every byte belong to exactly one workgroup at every split the launcher can pick, and the
// read-before-write inside a workgroup is ordered by a barrier.
#define RU_FSTR (4 * RU_K + RU_V + 8)

// RU_TMAX bounds the candidate block STG can stage: two floats a token of LDS, plus four
// partials while they are being combined. The launcher checks the caller's width against it.
#define RU_TMAX 16

// RU_ANC_DUP=0 drops the two neighbouring-page copies of the anchor and of the replay factors and
// keeps only the one the read column points at.
//
// WITHOUT THE MARKER IT IS NOT CORRECT and only measures the cost of the duplication -- a
// sequence whose page index moves reads a slot nothing wrote.  WITH the marker it is the whole
// point: the marker already says which column to read, so the anchor is written once (RU_ANC_MARK
// does that on its own) and the factor block is written once too (this switch).  The factor block
// is the bigger of the two, RU_TMAX * RU_FSTR against RU_V * RU_K of a head's slot, so
// `-DRU_ANC_MARK=1 -DRU_ANC_DUP=0` is the shape to build.
#ifndef RU_ANC_DUP
#define RU_ANC_DUP 1
#endif

// SHIFT MARKER.  Three copies of the anchor is the price of not knowing which way the page index
// moved -- and the factor block, which has to be written in three copies anyway, can just SAY.
// Each copy carries the column it was written to, so the reader finds `m` at its own fixed factor
// column, learns that the page index moved by `m - 1`, and reads the anchor at the one column that
// follows from it.  The anchor is then written ONCE: at eight sequences that is 6.3 MB a layer
// instead of 18.9, and the whole discovery costs one element.
//
// The marker sits past everything the factor data can reach -- RU_TMAX tokens of RU_FSTR -- and
// inside one head's slice of a slot, so the three copies never touch each other's and no workgroup
// reads a marker another one is writing (different `hv`, different address).  Its stored values
// are 1, 2 and 3 (one-based, see the read), exact in fp16 as well as fp32.
//
// WRITE THE ANCHOR AT COLUMN 2, NOT COLUMN 1.  vLLM MIGRATES the mamba state across a page boundary
// itself: `precopy_mamba_align_fused_kernel` copies state[bt[src_col + num_accepted - 1]] into
// state[bt[dst_col]] before the forward, and dst_col is the page the new window starts on -- this
// window's column 0.  Writing the anchor at column 1 and reading it at 2 - m lands on column 0 for
// a +1 shift, which the migration has already overwritten; the damage is intermittent, because a
// generation that never crosses with the wrong count is unaffected while one that does degenerates
// mid-word.  Column 2 / 3 - m keeps the read in {1, 2, 3} and never lands on 0.
//
// The single-write anchor is worth roughly a fifth of the layer's time on its own, and about 14%
// of the whole decode step.
#ifndef RU_ANC_MARK
#define RU_ANC_MARK 1
#endif
#define RU_MARK_OFF (RU_TMAX * RU_FSTR)

// RU_ANC_FRD=1 is a diagnostic build: it takes the displaced factor read (column 5 - m) WITHOUT
// dropping the two spare factor copies.  With all three written every step the read column cannot
// change what is read, so it separates the column algebra and `m` itself from the write side:
// degenerate output under it means the marker cannot steer a read reliably, while clean output
// under it and degenerate output under RU_ANC_DUP=0 place the fault on the WRITE side -- a step
// that writes no factors at all, whose stale page the three-copy form makes harmless.
#ifndef RU_ANC_FRD
#define RU_ANC_FRD 0
#endif

// THE DECODE CONVOLUTION, FOLDED IN (CV).  gdn_conv_update's only reader at decode is this kernel,
// and as two kernels the pair pays a launch and a dependent-dispatch gap for a few microseconds of
// work.  Under CV
// the workgroup computes the convolution its own token loop reads -- the 128 q and 128 k channels
// of its key head and the 128 v channels of its value head -- straight from the projection's
// output rows into LDS, and q, k and v never reach memory.
//
// THE ARITHMETIC IS gdn_conv_update's, channel by channel: the same taps in the same order, the
// same cp_silu (r4d_gdn_conv_tap.h), rounded to bf16 where that kernel stores, so the fold is
// byte-identical to the pair.
//
// THE CONV STATE IS REWRITTEN IN PLACE, and the key head's channels are read by every value head
// of its group -- H / Hg workgroups -- so no one of them may rewrite those channels until all of
// them have read the history.  Each workgroup arrives on a counter once its history is in hand,
// and the LAST to arrive rewrites the key head's q and k channels at its end; nothing waits.  A
// value head's own v channels have one reader, so it rewrites them itself. The last arriver
// resets its counter; the row clears the live counters itself before every dispatch, so a
// reused scratch slice cannot stall the protocol.
struct RuConv {
  const unsigned short* x; long xpitch;   // the projection's [q | k | v] columns, row pitch
  const unsigned short* w;                // [conv_dim, R4D_GDN_CONV_W]
  const unsigned short* b;                // [conv_dim] or null
  unsigned short* cs;                     // the conv state cache
  long cs_seq, cs_dim, cs_tok;
  const int* cidx; long ci_stride;        // conv state slot per sequence; negative is padding
  unsigned int* ctr;                      // one arrival counter per (sequence, key head)
};
// Tokens a sequence may bring to the folded convolution: the history and the step's inputs are
// preloaded into registers and indexed at compile time, as in the standalone update.
#define RU_CVT 16

// The token loop stages at most RU_CVT tokens and rewrites the conv state with the same W - 2
// split gdn_conv_update computes (state_len_max - max_query_len), so the stores are that
// kernel's. The cache's DEPTH is the deployment's window, not this step's token count, and the
// entry point bounds it separately.

// The CV dispatch takes separate pointer arguments so native graph replay can rebind each moving
// operand, plus one 72-byte trivially copyable scalar config: 20 arguments and 224 packed bytes,
// compile-time asserted below against RAD_MAX_KPARAMS and RAD_KPARAM_BLOB.
struct RuCompact {
  int ab_stride, st_slot, st_head, oq_pitch, os_ld, sidx_stride, zg_pitch;
  int xpitch, cs_seq, cs_dim, cs_tok, ci_stride;
  float neps, scale, sp_thr;
  int nact, H, Hg;
};
static_assert(sizeof(RuCompact) == 72 && alignof(RuCompact) == 4);

template <int ABF16, int NORM, int VS, int NT, int ANC, int STG, int SDT, int CV>
__device__ __forceinline__
void ru_update_body(const unsigned short* __restrict__ q,
                    const unsigned short* __restrict__ k,
                    const unsigned short* __restrict__ v,
                    const void* __restrict__ av, const void* __restrict__ bv, long ab_stride,
                    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                    typename ru_state<SDT>::T* __restrict__ state, long st_slot, long st_head,
                    unsigned short* __restrict__ o, unsigned char* __restrict__ oq,
                    float* __restrict__ os, long oq_pitch, long os_ld, const int* __restrict__ cu,
                    const int* __restrict__ sidx, long sidx_stride,
                    const int* __restrict__ naccept, const unsigned short* __restrict__ zg,
                    long zg_pitch, const float* __restrict__ nw, float neps, int nact,
                    int H, int Hg, float scale, float sp_thr, const RuConv cv)
{
  using ST = typename ru_state<SDT>::T;
  constexpr int NR = RU_V / VS;         // v rows this workgroup owns
  constexpr int THR = NR * 2;
  static_assert(!(NORM && VS > 1), "the fused row norm needs the whole 128-wide row");
  /* AG01: Separate reduction buffers for RMS norm and Amax eliminate 2 reuse barriers per token */
  __shared__ float wsum_rms[THR / 32];
  __shared__ float wsum_amax[THR / 32];
  __shared__ float spart[STG ? RU_TMAX * 4 : 1];   // per (token, half): sum q^2, sum k^2
  __shared__ float sscl [STG || CV ? RU_TMAX * 2 : 1]; // per token: two closed scales
  const int n = blockIdx.x, hv = blockIdx.y;
  const int bos = cu[n], T = cu[n + 1] - bos;
  // A sequence with no tokens still has an anchor to advance: ANCHOR mode is called with T == 0
  // on purpose, to fold the previous step's accepted prefix in before something else reads the
  // state cache. Without ANCHOR there is nothing to do and the workgroup leaves.
  const bool empty = (T <= 0);
  if (empty && !ANC) return;

  const int tid = threadIdx.x;
  const int row = (tid >> 1) + (VS > 1 ? blockIdx.z * NR : 0), half = tid % 2, k0 = half * RU_HK;
  const int hk = hv / (H / Hg);
  /* The head's two constants, read here so their latency runs under the prologue's loads rather
   * than behind the barriers in front of the token loop. */
  const float a_log_h = A_log[hv], dtb = dt_bias[hv];

  // ---- CV: the convolution's inputs, issued first so the history and this step's tokens are in
  // flight while the anchor replay below runs; they are consumed after it. Thread `tid` owns one
  // q|k channel of the key head (q below RU_K, k above) and, below RU_V, one v channel of the
  // value head.
  static_assert(!CV || (NORM && VS == 1 && STG == 0), "the folded convolution is the decode shape");
  static_assert(RU_K == RU_V, "one LDS row pitch serves the q, k and v taps");
  constexpr int CW = R4D_GDN_CONV_W, CS = R4D_GDN_CONV_W - 1;
  __shared__ unsigned short cvq[CV ? RU_CVT * RU_K : 1], cvk[CV ? RU_CVT * RU_K : 1];
  __shared__ unsigned short cvv[CV ? RU_CVT * RU_V : 1];
  __shared__ int cvlast;
  /* THE TOKEN LOOP'S OWN GLOBAL READS, STAGED THE SAME WAY. Per token the loop reads `a` and `b`
   * (the decay and beta inputs, one scalar each) and the output gate's row, and at decode there
   * is well under a wave a SIMD to hide a load behind -- read in the loop, each is a full global
   * latency on every token. They are issued here with the history, parked in LDS at the
   * convolution's barrier and read from there: `ca_b` is one token's a (tid < T) or b
   * (T <= tid < 2T), `czg` four channels of the step's gate rows, thread-strided so the loads
   * stay coalesced. The values are the loaded ones, so every product downstream is unchanged. */
  constexpr int ZPT = (RU_CVT * RU_V + THR - 1) / THR;  // gate values a thread stages
  __shared__ float sab[CV ? 2 * RU_CVT : 1];
  __shared__ float sg [CV ? RU_CVT : 1];
  __shared__ float sbt[CV ? RU_CVT : 1];
  __shared__ float seg[CV ? RU_CVT : 1];
  __shared__ unsigned short szg[CV ? RU_CVT * RU_V : 1];
  if constexpr (CV) {
    if (!empty) {
      const int cslot = cv.cidx[(size_t)n * cv.ci_stride];
      if (cslot < 0) return;
    }
  }

  // Without ANCHOR the initial state is the one the last accepted token of the previous step
  // wrote; with it there is only ever one state per sequence and the accepted prefix is replayed
  // onto it below.
  const int nacc = naccept ? naccept[n] : 1;
  // PAGED ANCHOR.  vLLM addresses the mamba state by PAGE and re-derives the page every step as
  // (seq_lens - 1) / block_size, so a sequence whose page index moves finds column 0 where some
  // other column used to be.  A scheme that keeps one state in a fixed column therefore reads its
  // anchor out of the wrong slot whenever the index moves.
  //
  // THE INDEX MOVES BOTH WAYS.  `seq_lens` is the context PLUS this step's query, and the query is
  // the draft depth, which the dynamic controller changes from step to step.  So
  //     start' - start = ((ctx + acc + q') - 1)/bs - ((ctx + q) - 1)/bs
  // is +1 when the committed context crosses, 0 almost always, and -1 when a step drafts fewer
  // tokens than the one before it did -- the whole row slides BACK a column.  Duplicating into
  // column 1 alone covers +1 and 0 and is wrong on -1, where a long greedy generation diverges
  // from the per-candidate path at the second page.
  //
  // Three copies is the minimum that needs no extra metadata.  With the shift in {-1, 0, +1}, one
  // FIXED read column works iff the three pages it can land on were all written, so the anchor is
  // written to columns 0, 1, 2 and read from column 1, and the replay factors are written to
  // columns 3, 4, 5 and read from column 4:
  //     shift -1 -> next column 1 is this column 0     (and next column 4 is this column 3)
  //     shift  0 -> next column 1 is this column 1     (and next column 4 is this column 4)
  //     shift +1 -> next column 1 is this column 2     (and next column 4 is this column 5)
  // The six columns are real, distinct, writable slots: vLLM's own Triton path writes one state
  // per candidate to exactly these indices, so `indices_stride` of them exist whenever a spec step
  // runs, and `indices_stride` here is num_spec + 1 = 8 whatever depth the controller picked.
  //
  // AND vLLM MIGRATES THE STATE ITSELF, WHICH IS THE OTHER HALF OF WHY THIS LAYOUT IS THE ONE.
  // `precopy_mamba_align_fused_kernel` runs before the forward on any step that crosses a boundary
  // and copies state[bt[src_col + num_accepted - 1]] into state[bt[dst_col]] -- it assumes the
  // per-candidate layout, so under the anchor its source is usually a factor block and its
  // destination, page start_prev + 1, is garbage afterwards.  Reading the anchor at column 1
  // resolves to page start_prev + 2 and the factors at column 4 to start_prev + 5, both of which
  // the migration never writes, so it does not matter what it put there.  Any future scheme that
  // moves these columns has to be re-derived against that kernel, not only against
  // (seq_lens - 1) / block_size.
  //
  // NOTHING ELSE MAY WRITE THIS CACHE.  A path that advances column 0 alone leaves the other two
  // copies stale, so every step shape a generation can take is covered by the fused path --
  // including the non-speculative decode the controller emits when it declines to draft
  // (`kind == "decode1"` in radiance_gdn.py) -- and a step that still falls out folds the accepted
  // prefix back into a single state first.
  const int sa0 = ANC ? sidx[(size_t)n * sidx_stride + 0] : 0;   // anchor, page before
  const int sa1 = ANC ? sidx[(size_t)n * sidx_stride + 1] : 0;   // anchor, written here
  const int sa2 = ANC ? sidx[(size_t)n * sidx_stride + 2] : 0;   // anchor, page after
  const int sf0 = ANC ? sidx[(size_t)n * sidx_stride + 3] : 0;   // factors, marker 0
  const int sf  = ANC ? sidx[(size_t)n * sidx_stride + 4] : 0;   // factors, marker 1, read here
  const int sf2 = ANC ? sidx[(size_t)n * sidx_stride + 5] : 0;   // factors, marker 2
  // A NEGATIVE slot is the pad and ZERO IS A REAL SLOT, the same rule the per-candidate read
  // below follows. Testing <= 0 here would drop every update for whoever holds slot
  // 0, silently: its state would simply never advance.
  if (ANC && (sa0 < 0 || sa1 < 0 || sa2 < 0 || sf0 < 0 || sf < 0 || sf2 < 0)) return;
  ST* const fb  = ANC ? (state + (size_t)sf  * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb0 = ANC ? (state + (size_t)sf0 * st_slot + (size_t)hv * st_head) : nullptr;
  ST* const fb2 = ANC ? (state + (size_t)sf2 * st_slot + (size_t)hv * st_head) : nullptr;
  // Which column the anchor is in.  Without markers it is column 1 and all three copies were
  // written; with them the factor block this step reads says how far the index moved.
  // COLUMN 0 IS THE MIGRATION'S TARGET, so the anchor is written at column 2 and read at 3 - m,
  // which is 1, 2 or 3 -- never 0.  `precopy_mamba_align_fused_kernel` derives its destination from
  // num_computed_tokens, i.e. the page of the COMMITTED context, which is this window's column 0
  // (or below it when the query itself crossed).  Writing the anchor at column 1 and reading at
  // 2 - m resolves to column 0 on a +1 shift, where the migration has already put
  // state[bt[src + num_accepted - 1]].  The failure is intermittent: a generation that never
  // crosses with the wrong count is unaffected, one that does degenerates mid-word.
  // A CALLER WHOSE STATE DOES NOT PAGE NAMES ONE SLOT IN EVERY COLUMN, and then every duplicate
  // below is the same bytes written again. The copies exist because vLLM's mamba cache is
  // addressed by page and the page index moves; radiance's linear pool gives a sequence one slot
  // for its life. Comparing the indices costs one uniform branch and saves two thirds of the
  // factor traffic -- 8.2 MB a layer at eight sequences -- for the caller that does not need it.
  const bool dup_a = ANC && RU_ANC_DUP && (sa0 != sa2 || sa1 != sa2);
  const bool dup_f = ANC && RU_ANC_DUP && (sf0 != sf || sf2 != sf);
  int acol = RU_ANC_MARK ? 2 : 1;
  int mrk = 1;
  if (ANC && RU_ANC_MARK) {
    // THE MARKER IS ONE-BASED so that a ZEROED factor block reads as absent rather than as
    // shift 0. A freshly handed out slot is zeroed by the engine, and a zero-based marker would
    // steer the state read to column 3 -- a FACTOR column -- which on the first step after a
    // prefill is those zeros: the sequence would start from an empty state and then write that
    // over the state the prefill had just built. Values 1, 2, 3 read back as shifts 0, 1, 2, so a
    // paged caller sees no difference; only the unwritten case moves, and it moves to the default
    // column, which is an anchor.
    const int m = (int)((float)fb[RU_MARK_OFF] + 0.5f) - 1;
    mrk = (m >= 0 && m <= 2) ? m : 1;
    acol = 3 - mrk;
  }
  // AND THE FACTORS FOLLOW THE SAME RULE.  Once the marker says how far the index moved there is
  // no reason to write the factor block three times either: it is written at column 4 and read at
  // 5 - m, which is 3, 4 or 5.  That is the larger of the two saves -- a head's factors are
  // RU_TMAX * RU_FSTR of its RU_V * RU_K slot, so three copies of them outweigh three copies of
  // the state.  The MARKER still lives at the fixed column 4 that produced m, and is still stamped
  // into all three: it is one element, and it is what makes the rest of this work.
  const ST* const fr = (ANC && RU_ANC_MARK && (RU_ANC_FRD || !RU_ANC_DUP))
      ? (const ST*)(state + (size_t)(mrk == 0 ? sf2 : (mrk == 1 ? sf : sf0)) * st_slot
                    + (size_t)hv * st_head)
      : (const ST*)fb;
  const int si = sidx[(size_t)n * sidx_stride + (ANC ? acol : (nacc - 1))];
  // A NEGATIVE slot is the pad and ZERO IS A REAL SLOT -- see the same rule in
  // r4d_gdn_conv_w4_h128_bf16.hip. A `<= 0` test here loses the first sequence's state entirely.
  if (si < 0) return;
  const int grp = row >> 5;                             // fixed 32-row group, see RU_FSTR

  if constexpr (CV) {
    if (!empty) {
      const int G = H / Hg;
      const int cslot = cv.cidx[(size_t)n * cv.ci_stride];
      const int off = naccept ? naccept[n] - 1 : 0;
      const int ca = tid < RU_K ? hk * RU_K + tid : (Hg + hk) * RU_K + (tid - RU_K);
      const int cb = (2 * Hg + hv) * RU_V + tid;
      const unsigned short* xr = cv.x + (size_t)bos * cv.xpitch;
      float cwa[CW], cba = 0.0f;
      unsigned short cha[CS];
      unsigned short* spa = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)ca * cv.cs_dim;
      {
        const ushort4 w4 = *(const ushort4*)(cv.w + (size_t)ca * CW);
        cwa[0] = bf2f(w4.x); cwa[1] = bf2f(w4.y); cwa[2] = bf2f(w4.z); cwa[3] = bf2f(w4.w);
        if (cv.b) cba = bf2f(cv.b[ca]);
      }
#pragma unroll
      for (int i = 0; i < CS; ++i) cha[i] = spa[(size_t)(off + i) * cv.cs_tok];
      auto conv_stream = [&](const float* wt, float bs, const unsigned short* hist,
                             const unsigned short* xcol, unsigned short* dst) {
        float win[CW];
#pragma unroll
        for (int i = 0; i < CS; ++i) win[i] = bf2f(hist[i]);
        for (int t = 0; t < T; ++t) {
          win[CW - 1] = bf2f(xcol[(size_t)t * cv.xpitch]);
          float acc = bs;
#pragma unroll
          for (int i = 0; i < CW; ++i) acc += wt[i] * win[i];
          dst[t * RU_K] = f2bf(cp_silu(acc));
#pragma unroll
          for (int i = 0; i < CW - 1; ++i) win[i] = win[i + 1];
        }
      };
      auto rewrite_stream = [&](unsigned short* sp, const unsigned short* hist,
                                const unsigned short* xcol) {
        constexpr int NH = R4D_GDN_CONV_W - 2;
#pragma unroll
        for (int i = 0; i < NH; ++i) sp[(size_t)i * cv.cs_tok] = hist[i + 1];
        for (int t = 0; t < T; ++t) sp[(size_t)(t + NH) * cv.cs_tok] = xcol[(size_t)t * cv.xpitch];
      };
      conv_stream(cwa, cba, cha, xr + ca, (tid < RU_K ? cvq : cvk) + (tid & (RU_K - 1)));
      if (tid < RU_V) {
        float cwb[CW], cbb = 0.0f;
        unsigned short chb[CS];
        const ushort4 w4 = *(const ushort4*)(cv.w + (size_t)cb * CW);
        cwb[0] = bf2f(w4.x); cwb[1] = bf2f(w4.y); cwb[2] = bf2f(w4.z); cwb[3] = bf2f(w4.w);
        if (cv.b) cbb = bf2f(cv.b[cb]);
        unsigned short* spb = cv.cs + (size_t)cslot * cv.cs_seq + (size_t)cb * cv.cs_dim;
#pragma unroll
        for (int i = 0; i < CS; ++i) chb[i] = spb[(size_t)(off + i) * cv.cs_tok];
        conv_stream(cwb, cbb, chb, xr + cb, cvv + tid);
        rewrite_stream(spb, chb, xr + cb);
      }
      if (tid < 2 * T)
        sab[tid] = tid < T ? ru_ab<ABF16>(av, (size_t)(bos + tid) * ab_stride + hv)
                           : ru_ab<ABF16>(bv, (size_t)(bos + tid - T) * ab_stride + hv);
      for (int i = 0; i < ZPT; ++i) {
        const int zi = tid + i * THR;
        if (zi < T * RU_V)
          szg[zi] = zg[(size_t)(bos + zi / RU_V) * zg_pitch + (size_t)hv * RU_V + (zi % RU_V)];
      }
      __builtin_amdgcn_wave_barrier();
      if (tid < T) {
        const float ra = sab[tid];
        const float rb = sab[T + tid];
        const float alog_t = __expf(a_log_h);
        const float g  = -alog_t * ru_softplus(ra + dtb, sp_thr);
        const float bt = 1.0f / (1.0f + __expf(-rb));
        const float eg = __expf(g);
        sg[tid]  = g;
        sbt[tid] = bt;
        seg[tid] = eg;
      }
      __syncthreads();
      if (G > 1) {
        if (tid == 0) {
          const unsigned int arrived = __hip_atomic_fetch_add(
              cv.ctr + (size_t)n * Hg + hk, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
          cvlast = arrived == (unsigned)(G - 1);
          if (cvlast)
            __hip_atomic_store(cv.ctr + (size_t)n * Hg + hk, 0u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        if (cvlast) rewrite_stream(spa, cha, xr + ca);
      } else {
        rewrite_stream(spa, cha, xr + ca);
      }
    }
  }

  float h[RU_HK];
  {
    const ST* p = state + (size_t)si * st_slot + (size_t)hv * st_head
                + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) ru_load8<NT, SDT>(&h[j], p + j);
  }

  if (ANC) {
    for (int t = 0; t < nacc; ++t) {
      const ST* f = fr + (size_t)t * RU_FSTR;
      float kk[RU_HK];
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) ru_load8<0, SDT>(&kk[j], f + grp * RU_K + k0 + j);
      const float vv = (float)f[4 * RU_K + row];
      const float eg = __expf((float)f[4 * RU_K + RU_V + 2 * grp]);
      const float bt = (float)f[4 * RU_K + RU_V + 2 * grp + 1];
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
      dot += r4d_xor_lane<1>(dot);          // a DPP modifier, not an LDS round trip
      const float u = (vv - dot) * bt;
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) h[j] += u * kk[j];
    }
    // Publish the advanced anchor to BOTH pages, then fence the factor block: every replay read
    // above must happen before the loop below overwrites it with this step's factors.
    const size_t rowoff = (size_t)hv * st_head + (size_t)row * RU_K + k0;
    ST* p  = state + (size_t)(RU_ANC_MARK ? sa2 : sa1) * st_slot + rowoff;
    ST* p0 = state + (size_t)sa0 * st_slot + rowoff;
    ST* p2 = state + (size_t)sa2 * st_slot + rowoff;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
    if (dup_a && !RU_ANC_MARK) {
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p0 + j, &h[j]);
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p2 + j, &h[j]);
    }
    __syncthreads();
    // THE STAMP COMES AFTER THE BARRIER. Every thread read `fb`'s marker at the top to pick its
    // anchor and factor columns, and a wave can reach that read after thread 0 has finished its
    // own replay: stamped before the barrier, the new marker could steer a late wave to a
    // different column from the rest of the workgroup. The replay reads are done by now and the
    // marker lies past everything the factor writes below can reach.
    if (RU_ANC_MARK && tid == 0) {                      // the three copies say where they are
      fb0[RU_MARK_OFF] = (ST)1.0f;
      fb[RU_MARK_OFF]  = (ST)2.0f;
      fb2[RU_MARK_OFF] = (ST)3.0f;
    }
  }
  if (empty) return;




  const float alog = CV ? 0.0f : __expf(a_log_h);

  if (STG) {
    // One lane per (token, half) does exactly the sum a v-row takes for itself on the STG=0 path.
    if (tid < 2 * T) {
      const int ut = tid >> 1, uk0 = (tid % 2) * RU_HK;
      const unsigned short* qp = q + ((size_t)(bos + ut) * Hg + hk) * RU_K + uk0;
      const unsigned short* kp = k + ((size_t)(bos + ut) * Hg + hk) * RU_K + uk0;
      float ssq = 0.0f, ssk = 0.0f;
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) {
        const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
        const unsigned short* aq = (const unsigned short*)&wq;
        const unsigned short* ak = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float fq = bf2f(aq[e]), fk = bf2f(ak[e]);
          ssq += fq * fq; ssk += fk * fk;
        }
      }
      spart[tid * 2] = ssq;
      spart[tid * 2 + 1] = ssk;
    }
    __syncthreads();
    if (tid < T) {
      // Same order the shfl closed them in: own half plus the other half's partial.
      const float ssq = spart[tid * 4]     + spart[tid * 4 + 2];
      const float ssk = spart[tid * 4 + 1] + spart[tid * 4 + 3];
      sscl[tid * 2]     = __frsqrt_rn(ssq + 1e-6f) * scale;
      sscl[tid * 2 + 1] = __frsqrt_rn(ssk + 1e-6f);
    }
    __syncthreads();
  }

  for (int t = 0; t < T; ++t) {
    const int tok = bos + t;
    const unsigned short* qp = CV ? cvq + t * RU_K + k0 : q + ((size_t)tok * Hg + hk) * RU_K + k0;
    const unsigned short* kp = CV ? cvk + t * RU_K + k0 : k + ((size_t)tok * Hg + hk) * RU_K + k0;

    float qq[RU_HK], kk[RU_HK];
    if constexpr (!CV) {
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) {
        const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
        const unsigned short* sq = (const unsigned short*)&wq;
        const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) { qq[j + e] = bf2f(sq[e]); kk[j + e] = bf2f(sk[e]); }
      }
    }

    // tid 0/1 have row 0's two halves (k0 = 0/64); every value row addresses
    // the same cvq/cvk[t * RU_K + k0 + j] for its key head. Keep the original
    // ascending 64-term sums and lane-1 exchange. All lanes execute the DPP
    // exchange, so both producer lanes are active even though only they sum.
    float qs, ks;
    if constexpr (CV) {
      float sq = 0.0f, sk = 0.0f;
      if (tid < 2) {
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) {
          const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
          const unsigned short* aq = (const unsigned short*)&wq;
          const unsigned short* ak = (const unsigned short*)&wk;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float fq = bf2f(aq[e]), fk = bf2f(ak[e]);
            sq += fq * fq; sk += fk * fk;
          }
        }
      }
      sq += r4d_xor_lane<1>(sq);
      sk += r4d_xor_lane<1>(sk);
      if (tid == 0) {
        sscl[t * 2] = __frsqrt_rn(sq + 1e-6f) * scale;
        sscl[t * 2 + 1] = __frsqrt_rn(sk + 1e-6f);
      }
      // Distinct slots per token: later iterations cannot overwrite this token's scales.
      __syncthreads();
      qs = sscl[t * 2]; ks = sscl[t * 2 + 1];
    } else if (STG) {
      qs = sscl[t * 2]; ks = sscl[t * 2 + 1];
    } else {
      float sq = 0.0f, sk = 0.0f;
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { sq += qq[j] * qq[j]; sk += kk[j] * kk[j]; }
      sq += r4d_xor_lane<1>(sq);
      sk += r4d_xor_lane<1>(sk);
      qs = __frsqrt_rn(sq + 1e-6f) * scale; ks = __frsqrt_rn(sk + 1e-6f);
    }
    // STG 2 leaves qq/kk RAW and carries the scales into the reductions below instead.
    //
    // =========================== the query LDS reload =========================
    // UNDER CV ONLY, q IS NOT SCALED IN PLACE. `qs` is carried into the output accumulation
    // below, which reads the same LDS bytes again and multiplies them by the same qs -- the
    // same two fp32 operands and the same single multiply, so the product is bit-identical
    // (this unit is compiled -ffp-contract=off, so the multiply cannot fuse into anything).
    //
    // CV only reads raw q in the first thread pair for the shared norm; the
    // output accumulation still reloads cvq at its original width. The norm
    // uses the same BF16 bytes and ordered FP32 sum as every former row pair.
    // No full qq array needs to remain live under CV. The resource effect of
    // the removed redundant reads and the new block barrier is unmeasured.
    // Every BF16 intermediate, anchor/state semantic and int8 epilogue stays.
    //
    // CV IMPLIES STG == 0 (the static_assert at the top of the kernel), so the `STG == 2`
    // arm below is unreachable here and the late read cannot be scaled twice.
    // =====================================================================================
    //
    // ============================ the key LDS reload ===========================
    // UNDER CV, k IS NOT SCALED IN PLACE EITHER. Only the first pair reads its
    // raw k for the shared norm; the three post-norm consumers -- the delta-rule inner product,
    // anchor factor store and the `h` update -- re-reads the immutable BF16 conv output `cvk` at
    // the same 16-byte width and multiplies by the same `ks`:
    //
    //     bf2f(cvk[t*RU_K + k0 + j]) * ks   ==   kk[j] * ks
    //
    // Both sides are one bf2f widening of the same 16 bits followed by one fp32 multiply, and
    // this unit is compiled -ffp-contract=off, so neither multiply can fuse into the add or the
    // FMA after it. No intermediate format changes: kk is fp32, the reloaded value is fp32, and
    // the factor store still rounds once through ru_store8 into the block's element type.
    //
    // cvk is written once, in this kernel's prologue, by the conv_stream call that fills it; the
    // token loop contains no store to cvk, and the prologue's __syncthreads() already orders
    // those writes against the first read. The reload adds no barrier and no new lifetime.
    // =====================================================================================
    if (STG != 2) {
      // Under CV, q and k are reloaded by their existing consumers, unchanged.
      if constexpr (!CV) {
#pragma unroll
        for (int j = 0; j < RU_HK; ++j) qq[j] *= qs;
#pragma unroll
        for (int j = 0; j < RU_HK; ++j) kk[j] *= ks;
      }
    }

    float g, bt, eg;
    if constexpr (CV) {
      g  = sg[t];
      bt = sbt[t];
      eg = seg[t];
    } else {
      const float ra = ru_ab<ABF16>(av, (size_t)tok * ab_stride + hv);
      const float rb = ru_ab<ABF16>(bv, (size_t)tok * ab_stride + hv);
      g  = -alog * ru_softplus(ra + dtb, sp_thr);
      bt = 1.0f / (1.0f + __expf(-rb));
      eg = __expf(g);
    }

    float dot = 0.0f;
    if constexpr (CV) {
      // THE KEY RELOAD: the inner product's k comes back from LDS, so `kk` need not span
      // this loop. `h[j] *= eg` and `dot +=` still walk j ascending, as the loop below does.
      const unsigned short* kr = cvk + t * RU_K + k0;
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) {
        const uint4 wk = *(const uint4*)(kr + j);
        const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float kv = bf2f(sk[e]) * ks;
          h[j + e] *= eg;
          dot += h[j + e] * kv;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
    }
    dot += r4d_xor_lane<1>(dot);
    if (STG == 2) dot *= ks;

    const float vraw = bf2f(CV ? cvv[t * RU_V + row] : v[((size_t)tok * H + hv) * RU_V + row]);
    const float u = (vraw - dot) * bt;
    const float uk = (STG == 2) ? u * ks : u;

    if (ANC) {
      // This step's factors, for the next step's replay. One thread pair per row group writes the
      // group's k copy and its decay/beta; every row writes its own v.
      ST* f  = fb  + (size_t)t * RU_FSTR;
      ST* f0 = fb0 + (size_t)t * RU_FSTR;
      ST* f2 = fb2 + (size_t)t * RU_FSTR;
      if ((row & 31) == 0) {
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) {
          float kn[8];
          if constexpr (CV) {
            // THE KEY RELOAD: the stored copy is re-read from cvk too. Only the 1-in-32
            // rows that own a group reach this, so it is the cheapest of the three reloads.
            const uint4 wk = *(const uint4*)(cvk + t * RU_K + k0 + j);
            const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
            for (int e = 0; e < 8; ++e) kn[e] = bf2f(sk[e]) * ks;
          } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) kn[e] = (STG == 2) ? kk[j + e] * ks : kk[j + e];
          }
          ru_store8<0, SDT>(f + grp * RU_K + k0 + j, kn);
          if (dup_f) {
            ru_store8<0, SDT>(f0 + grp * RU_K + k0 + j, kn);
            ru_store8<0, SDT>(f2 + grp * RU_K + k0 + j, kn);
          }
        }
        if (half == 0) {
          f[4 * RU_K + RU_V + 2 * grp]      = (ST)g;
          f[4 * RU_K + RU_V + 2 * grp + 1]  = (ST)bt;
          if (dup_f) {
            f0[4 * RU_K + RU_V + 2 * grp]     = (ST)g;
            f0[4 * RU_K + RU_V + 2 * grp + 1] = (ST)bt;
            f2[4 * RU_K + RU_V + 2 * grp]     = (ST)g;
            f2[4 * RU_K + RU_V + 2 * grp + 1] = (ST)bt;
          }
        }
      }
      if (half == 0) {
        f[4 * RU_K + row] = (ST)vraw;
        if (dup_f) { f0[4 * RU_K + row] = (ST)vraw; f2[4 * RU_K + row] = (ST)vraw; }
      }
    }

    // =========================== the query LDS reload =========================
    // CV READS q BACK FROM LDS HERE, INSTEAD OF HOLDING IT ACROSS THE UPDATE. `qr` is the same
    // cvq row `qp` pointed at, read at the same 16-byte width the original load used, so every
    // element is bf2f() of the same 16 bits and `qv` is the product the deferred scale above
    // would have written into qq -- the same two fp32 operands, one multiply, `-ffp-contract=off`
    // so it stays a plain v_mul_f32. `on +=` still walks j in ascending order over the same
    // `h[j]`, so the accumulation sequence is unchanged.
    //
    // cvq is written once, in this kernel's prologue, by the conv_stream call that fills it, and
    // is not touched again before this point: the token loop contains no store to cvq, and the
    // prologue's __syncthreads() already orders those writes against the first read. The reload
    // adds no barrier and no new lifetime.
    // =====================================================================================
    //
    // ============================ the key LDS reload ===========================
    // THE `h` UPDATE READS `cvk` HERE FOR THE SAME REASON AND WITH THE SAME ARGUMENT: `kv` is
    // bf2f(cvk) * ks, the value the in-place scale would have put in kk[j], so `uk * kv` is the
    // same two fp32 operands as `uk * kk[j]` and still a plain multiply followed by an add. `on`
    // still accumulates ascending over the same post-update `h[j]`, so the row's sum is unchanged.
    // =====================================================================================
    float on = 0.0f;
    if constexpr (CV) {
      const unsigned short* qr = cvq + t * RU_K + k0;
      const unsigned short* kr = cvk + t * RU_K + k0;
#pragma unroll
      for (int j = 0; j < RU_HK; j += 8) {
        const uint4 wq = *(const uint4*)(qr + j);
        const uint4 wk = *(const uint4*)(kr + j);
        const unsigned short* sq = (const unsigned short*)&wq;
        const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float qv = bf2f(sq[e]) * qs;
          const float kv = bf2f(sk[e]) * ks;
          h[j + e] += uk * kv;
          on += h[j + e] * qv;
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < RU_HK; ++j) { h[j] += uk * kk[j]; on += h[j] * qq[j]; }
    }
    on += r4d_xor_lane<1>(on);
    if (STG == 2) on *= qs;

    if (NORM) {
      // sum of squares over the row: each pair holds one channel, so one wave covers 16 of them
      float sq = (half == 0) ? on * on : 0.0f;
      // unrolled so each mask is a compile-time constant: 16 takes v_permlanex16 and the
      // rest are DPP row_xmask.  Same order, same value.
      sq += r4d_xor_lane<16>(sq); sq += r4d_xor_lane<8>(sq); sq += r4d_xor_lane<4>(sq);
      sq += r4d_xor_lane<2>(sq);  sq += r4d_xor_lane<1>(sq);
      if ((threadIdx.x & 31) == 0) wsum_rms[threadIdx.x >> 5] = sq;
      __syncthreads();
      float tot = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) tot += wsum_rms[i];
      const float sc = __frsqrt_rn(tot * (1.0f / RU_V) + neps);
      const unsigned short zr = CV ? szg[t * RU_V + row]
                                   : zg[(size_t)tok * zg_pitch + (size_t)hv * RU_V + row];
      on *= sc * nw[row] * ru_act(bf2f(zr), nact);
      /* AG01: Barrier removed: wsum_amax is a distinct LDS buffer from wsum_rms */
    }

    const unsigned short ob = f2bf(on);
    if (half == 0) o[((size_t)tok * H + hv) * RU_V + row] = ob;

    // THE OUT PROJECTION'S QUANTISER, FOLDED IN.  `o` is read by exactly one thing -- the fp8
    // out projection -- and a standalone quant_act_fp8 between them is 48 launches a decode step
    // for 3.1 us of work each, which on this part is less than the 3.2 us gap that precedes each
    // one.  The fold is free because THE GROUP IS THE ROW: r4d_quant_act_fp8_group() is
    // 128, V is 128, and NORM already requires one workgroup per (token, head), so the group's
    // amax is a reduction this workgroup can already do and the scale plane index is (tok, hv).
    //
    // BIT-IDENTICAL TO THE STANDALONE QUANTISER, and deliberately: `ob` is the bf16 this kernel
    // stores, and quantising `bf2f(ob)` rather than `on` is what makes it so -- the standalone
    // quantiser reads these bytes, not the f32 that produced them.  Everything after is
    // the same arithmetic as r4d_fused_quant_fp8.hip's quant_group, including the fmin that keeps
    // the group's own maximum from converting to a NaN.
    //
    // NORM == 2 WRITES INT8 CODES instead, for an int8 out projection (gemm_i8a8_*): the scale
    // amax / 127 and the value over it rounded to nearest-even -- quant_act_i8g's arithmetic over
    // the same bf16 bytes, so the codes are the ones that quantiser would have written.
    if (NORM && oq) {
      const float ov = bf2f(ob);
      float mv = (half == 0) ? fabsf(ov) : 0.0f;
      mv = fmaxf(mv, r4d_xor_lane<16>(mv)); mv = fmaxf(mv, r4d_xor_lane<8>(mv));
      mv = fmaxf(mv, r4d_xor_lane<4>(mv));  mv = fmaxf(mv, r4d_xor_lane<2>(mv));
      mv = fmaxf(mv, r4d_xor_lane<1>(mv));
      if ((threadIdx.x & 31) == 0) wsum_amax[threadIdx.x >> 5] = mv;
      __syncthreads();
      float amax = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) amax = fmaxf(amax, wsum_amax[i]);
      const float qs = amax > 0.0f ? amax * (1.0f / q8_top<NORM == 2>()) : 1.0f;
      if (tid == 0) os[(size_t)tok * os_ld + hv] = qs;
      if (half == 0) {
        unsigned char code;
        if constexpr (NORM == 2) {
          constexpr float kMagic = 12582912.0f;
          const float cv = fminf(fmaxf(ov * (1.0f / qs), -127.0f), 127.0f);
          code = (unsigned char)(__float_as_uint(cv + kMagic) & 0xff);
        } else {
          const float cv = fminf(fmaxf(ov * (1.0f / qs), -448.0f), 448.0f);
          code = (unsigned char)(__builtin_amdgcn_cvt_pk_fp8_f32(cv, cv, 0, false) & 0xff);
        }
        oq[(size_t)tok * oq_pitch + (size_t)hv * RU_V + row] = code;
      }
      /* AG01: Barrier removed: wsum_rms in next token iteration does not overwrite wsum_amax */
    }

    if (!ANC) {
      const int so = sidx[(size_t)n * sidx_stride + t];
      // `>= 0`, NOT `> 0`.  A NEGATIVE slot is the pad and ZERO IS A REAL SLOT -- the same rule
      // the READ above follows, and the same one r4d_gdn_conv_w4_h128_bf16.hip follows.  With `> 0`
      // a single-sequence decode, whose only slot IS zero, never writes the state back at all:
      // every step then recurs from the state prefill left behind, the model re-reads its own
      // prompt and the generation degenerates into repeating it.  A correct read side hides this
      // one -- the kernel still produces a correct `o` for the first decoded token and a stale one
      // for every token after it.
      if (so >= 0) {
        ST* p = state + (size_t)so * st_slot + (size_t)hv * st_head
              + (size_t)row * RU_K + k0;
#pragma unroll
        for (int j = 0; j < RU_HK; j += 8) ru_store8<NT, SDT>(p + j, &h[j]);
      }
    }
  }


}
template <int NORM>
__global__ __launch_bounds__(RU_V * 2)
void ru_compact_cv(const void* av, const void* bv, const float* A_log,
                   const float* dt_bias, float* state, unsigned short* o,
                   unsigned char* oq, float* os, const int* cu, const int* sidx,
                   const int* naccept, const unsigned short* zg, const float* nw,
                   const unsigned short* x, const unsigned short* w,
                   const unsigned short* b, unsigned short* cs, const int* cidx,
                   unsigned int* ctr, RuCompact cfg) {
  RuConv cv{};
  cv.x = x; cv.xpitch = (long)cfg.xpitch;
  cv.w = w; cv.b = b; cv.cs = cs;
  cv.cs_seq = (long)cfg.cs_seq; cv.cs_dim = (long)cfg.cs_dim;
  cv.cs_tok = (long)cfg.cs_tok; cv.cidx = cidx;
  cv.ci_stride = (long)cfg.ci_stride; cv.ctr = ctr;
  ru_update_body<1, NORM, 1, 1, 1, 0, 0, 1>(
      nullptr, nullptr, nullptr, av, bv, (long)cfg.ab_stride,
      A_log, dt_bias, state, (long)cfg.st_slot, (long)cfg.st_head, o, oq, os,
      (long)cfg.oq_pitch, (long)cfg.os_ld, cu, sidx, (long)cfg.sidx_stride,
      naccept, zg, (long)cfg.zg_pitch, nw, cfg.neps, cfg.nact,
      cfg.H, cfg.Hg, cfg.scale, cfg.sp_thr, cv);
}

// THE NON-TEMPORAL HINT on the state load and store (th:TH_STORE_NT / TH_LOAD_NT).
// It changes no byte of the result -- the line stays coherent, it is only evicted first -- and it
// is chosen per launch because its cost and its saving land in different kernels and the trade is
// only visible in a whole step.  ISOLATED the hint is 43.8% SLOWER (20.76 us against 14.44 at
// 12 MB a call): non-temporal, this kernel is DRAM-write bound, where without the hint the writes
// stay in the last level and a later kernel pays for the write-back.  In a step there is no later
// escape -- 48 layers write ~660 MB and a 64 MB last level holds none of it -- so what the hint
// really moves is WHO pays.  With a matched 12 MB write flood between the decode GEMMs, the same
// (5120, 3072) GEMM is 14.9 us clean, 20.4 us behind the flood and 18.4 us behind a non-temporal
// flood, and the whole 64-layer sequence is 110 us faster with the hint.  Under anchor state the
// balance is about -140 us on the tiled GEMM against +25 us here.
//
// ANCHOR STATE: one state per sequence, with the accepted prefix replayed onto it.  It needs the
// caller's cooperation -- the prefill path must zero state slot sidx[n][1] for every sequence it
// writes -- so the kernel and radiance_gdn have to agree, and they do so by both being built this
// way rather than by reading a switch.
// It is a RUNTIME switch, not a build-time one, because the anchor scheme is NOT PAGED (see the
// block comment in radiance_gdn.py): vLLM addresses the mamba state by page, and the anchor keeps
// one state per sequence in the page holding the last token, so a sequence crossing a page reads
// its anchor out of the previous step's replay factors.  Off, every candidate writes its own slot
// -- which is exactly the indexing vLLM's own Triton path uses, so it is correct by construction
// and stays clean across consecutive page crossings.  radiance_gdn.py reads THE SAME variable, so
// the kernel and its caller cannot disagree about what the state cache means.
static bool ru_anchor() {
  const char* e = getenv("RADIANCE_GDN_ANCHOR");
  return e && e[0] == '1';
}


// A non-temporal store on the state write.  THE FLOOD SIZE DECIDES THE SIGN.  Under anchor state a
// layer floods 12.6 MB of state plus 4.1 MB of replay factors, and at that size the hint is worth
// about -140 us a step on the tiled W4A8 GEMM against +25 us on this kernel: net about -100 us.
// WITH ONE STATE PER CANDIDATE THE SIGN FLIPS: the flood is eight times larger and the hint is
// DRAM-write bound for all of it, where without it this kernel runs about 20% faster than FLA's
// Triton kernel. It stays a per-launch choice because the hint moves cost onto the GEMMs rather
// than deleting it. This row is compiled with the hint ON (NT = 1), which is the instantiation the
// folded decode shape was measured at.
// IT IS THE FLOOD SIZE THAT DECIDES, NOT THE SHAPE.  Non-temporal costs this kernel time and saves
// the GEMMs behind it time, so it wins exactly when the state a layer writes would otherwise have
// stayed in the 64 MB last level and been handed to the next GEMM as write-back.  At one sequence
// a layer floods 6.3 MB and the hint is worth about 3% of a whole step; at eight it floods 50 MB,
// none of it survives, and the hint costs about 2%.
//
// The threshold is the last level with room left for the weights streaming through it, not the
// whole 64 MB.
#ifndef RU_NT_MB
#define RU_NT_MB 16
#endif

// ============================================================== the folded convolution
constexpr unsigned kMaxConvCtr = 4096;

/* THE ARRIVAL COUNTERS LIVE IN THE ENGINE'S SCRATCH, AND THE ENGINE MAKES NO FIRST-USE ZERO
 * PROMISE. The recurrent body's last-arriving value head resets its own counter, so a scratch
 * slice that starts zeroed is stable across launches -- but the arena reuses a region and the
 * previous occupant's bytes are whatever it left. A counter that starts at a value the arrival
 * protocol never reaches makes the LAST arriver never fire, and the key head's conv channels are
 * never rewritten: the state is silently one step stale for every head of that group.
 *
 * So the row clears the live counters itself, on the caller's stream, before the folded dispatch.
 * This is a real dispatch of a real kernel (not hipMemset): both `launch` and `describe` emit it,
 * so a described sequence replays the same two dispatches. Only N * Hg counters are touched, not
 * the whole slice, and the clear never reads the counters.
 *
 * The counter is a plain store: the main kernel's arrival is a release/acquire atomic at agent
 * scope, and this dispatch is ordered before it on one stream, so stream order is the whole
 * happens-before argument. No sync, no memset, no allocation. */
__global__ void ru_scratch_clear(unsigned int* ctr, int count) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (unsigned int)count) ctr[i] = 0;
}


/* ------------------------------------------------------------------ conv + recurrent update */

/* gdn_conv_update's inputs, then gdn_recurrent_update's minus q, k and v, which are the pair's
 * intermediate and never reach memory here. Inputs first, outputs last -- docs/OPS.md. */
enum { OPD_CR_X = 0, OPD_CR_CW, OPD_CR_CB, OPD_CR_CSTATE, OPD_CR_CIDX, OPD_CR_A, OPD_CR_B,
       OPD_CR_ALOG, OPD_CR_DTB, OPD_CR_STATE, OPD_CR_SIDX, OPD_CR_NACC, OPD_CR_CU, OPD_CR_Z,
       OPD_CR_NORMW, OPD_CR_O, OPD_CR_OQ, OPD_CR_OS };

static int ru_compact_entry(const RadArgs* a, const R4dSink& sk) {
    if (gdn_dims_ok() != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    const RadTensor* x     = r4d_opt(a, OPD_CR_X);
    const RadTensor* cw    = r4d_opt(a, OPD_CR_CW);
    const RadTensor* cb    = r4d_opt(a, OPD_CR_CB);
    const RadTensor* cst   = r4d_opt(a, OPD_CR_CSTATE);
    const RadTensor* cidx  = r4d_opt(a, OPD_CR_CIDX);
    const RadTensor* ta    = r4d_opt(a, OPD_CR_A);
    const RadTensor* tb    = r4d_opt(a, OPD_CR_B);
    const RadTensor* alog  = r4d_opt(a, OPD_CR_ALOG);
    const RadTensor* dtb   = r4d_opt(a, OPD_CR_DTB);
    const RadTensor* state = r4d_opt(a, OPD_CR_STATE);
    const RadTensor* sidx  = r4d_opt(a, OPD_CR_SIDX);
    const RadTensor* nacc  = r4d_opt(a, OPD_CR_NACC);
    const RadTensor* cu    = r4d_opt(a, OPD_CR_CU);
    const RadTensor* z     = r4d_opt(a, OPD_CR_Z);
    const RadTensor* nw    = r4d_opt(a, OPD_CR_NORMW);
    const RadTensor* o     = r4d_opt(a, OPD_CR_O);
    const RadTensor* oq    = r4d_opt(a, OPD_CR_OQ);
    const RadTensor* os    = r4d_opt(a, OPD_CR_OS);
    /* THE ROW NORM IS REQUIRED: the fold is the decode shape, one workgroup per (sequence, value
     * head), and that is the shape the norm already forces on the unfolded kernel. */
    if (!x || !cw || !cst || !cidx || !ta || !tb || !alog || !dtb || !state || !sidx ||
        !nacc || !cu || !z || !nw || !o || !oq || !os)
        return R4D_NO(RAD_E_INVAL);
    if (x->rank != 2 || cst->rank != 3 || o->rank != 3 || ta->rank != 2 || tb->rank != 2 ||
        state->rank < 2 || sidx->rank != 2 || cidx->rank != 1)
        return R4D_NO(RAD_E_SHAPE);

    int N = 0;
    int rc = seqs_of(cu, &N);
    if (rc != RAD_OK) return rc;
    if (sidx->shape[0] != N) return R4D_NO(RAD_E_SHAPE);

    /* THE HEAD GEOMETRY COMES FROM `o` AND `x`: `o` is [tokens, H, V] and `x` is the projection's
     * [q | k | v] columns, 2 Hg K + H V wide. There is no q, k or v operand to read it off. */
    const long long T  = o->shape[0];
    const long long H  = o->shape[1];
    const long long V  = o->shape[2];
    const long long K  = r4d_geti_or(a, "head_k", 0);
    if (K != RU_K || V != RU_V || r4d_geti_or(a, "head_v", 0) != RU_V) return R4D_NO(RAD_E_SHAPE);
    if (x->shape[0] != T) return R4D_NO(RAD_E_SHAPE);
    const long long conv_dim = x->shape[1];
    if (H <= 0 || H > INT32_MAX || conv_dim <= H * V ||
        (conv_dim - H * V) % (2 * K)) return R4D_NO(RAD_E_SHAPE);
    const long long Hg = (conv_dim - H * V) / (2 * K);
    if (H % Hg) return R4D_NO(RAD_E_SHAPE);
    const long long nhv = r4d_geti_or(a, "n_head_v", 0);
    if (nhv > 0 && nhv != H) return R4D_NO(RAD_E_SHAPE);
    if ((long long)N * Hg > (long long)kMaxConvCtr) return R4D_NO(RAD_E_SHAPE);
    const long long xpitch = r4d_row_pitch(x);
    if (xpitch < 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(cw) || !r4d_contig(o)) return R4D_NO(RAD_E_STRIDE);
    const long long ab_stride = r4d_row_pitch(ta);
    if (ab_stride < 0 || r4d_row_pitch(tb) != ab_stride) return R4D_NO(RAD_E_STRIDE);

    R4D_DT(x, RAD_BF16); R4D_DT(cw, RAD_BF16); R4D_DT(cb, RAD_BF16); R4D_DT(cst, RAD_BF16);
    R4D_DT(ta, RAD_BF16); R4D_DT(tb, RAD_BF16); R4D_DT(o, RAD_BF16); R4D_DT(z, RAD_BF16);
    R4D_DT(alog, RAD_F32); R4D_DT(dtb, RAD_F32); R4D_DT(nw, RAD_F32);
    if (state->dtype != RAD_F32 || oq->dtype != RAD_I8) return R4D_NO(RAD_E_DTYPE);
    R4D_DT(os, RAD_F32);
    if (!r4d_contig(oq)) return R4D_NO(RAD_E_STRIDE);

    /* Convolution cache depth is the deployment window, not this step's token count.
     * The folded token loop stages at most RU_CVT tokens, while the cache may be deeper.
     * Check the declared q_len band separately from the cache's history depth. */
    if (r4d_geti_or(a, "conv_width", 0) != R4D_GDN_CONV_W) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long q_len = r4d_geti_or(a, "q_len", 0);
    if (q_len < 1 || q_len > RU_CVT) return R4D_NO(RAD_E_SHAPE);
    if (cst->shape[2] < R4D_GDN_CONV_W - 1) return R4D_NO(RAD_E_SHAPE);

    /* The recurrent half's contract checks, as r4d_rad_gdn_recurrent_update makes them. */
    if (const char* sf = r4d_gets(a, "state_form")) {
        const int want_anc = std::strcmp(sf, "anchor") == 0;
        if (!want_anc && std::strcmp(sf, "per_candidate") != 0) return R4D_NO(RAD_E_UNSUPPORTED);
        if (want_anc != (ru_anchor() ? 1 : 0)) return R4D_NO(RAD_E_UNSUPPORTED);
    }
    const float scale = (float)r4d_getf(a, "scale", 1.0);
    const float thr   = (float)r4d_getf(a, "softplus_thr", 20.0);
    const float neps  = (float)r4d_getf(a, "eps", 0.0);
    const int   nact  = r4d_act_code(a, "act", "norm_act", 0);
    if (nact < 0) return R4D_NO(RAD_E_INVAL);
    if ((float)r4d_getf(a, "l2_eps", 1e-6) != 1e-6f) return R4D_NO(RAD_E_UNSUPPORTED);
    const long long z_row = gn_row_pitch(z, H * V);
    if (z_row < 0) return R4D_NO(RAD_E_STRIDE);

    if (!ru_anchor()) return R4D_NO(RAD_E_UNSUPPORTED);
    auto fits = [](int64_t v) { return v >= 0 && v <= INT32_MAX; };
    const int64_t oq_row = r4d_row_pitch(oq), os_row = r4d_row_pitch(os);
    if (!fits(xpitch) || !fits(ab_stride) || !fits(state->stride[0]) ||
        !fits(state->stride[1]) || !fits(oq_row) || !fits(os_row) ||
        !fits(sidx->stride[0]) || !fits(z_row) || !fits(cst->stride[0]) ||
        !fits(cst->stride[1]) || !fits(cst->stride[2]) || !fits(cidx->stride[0]))
        return R4D_NO(RAD_E_STRIDE);
    if (sidx->stride[0] < 6 || sidx->stride[0] > INT32_MAX / RU_FSTR ||
        sidx->stride[0] * RU_FSTR > state->stride[1])
        return R4D_NO(RAD_E_SHAPE);
    if ((long long)N * state->stride[0] * 4 > (long long)RU_NT_MB * 1024 * 1024)
        return R4D_NO(RAD_E_UNSUPPORTED);  // this row is the reached NT=1 instantiation
    const int64_t counter_bytes = (int64_t)N * Hg * sizeof(unsigned int);
    if (!a->scratch || a->scratch_bytes < counter_bytes ||
        (reinterpret_cast<uintptr_t>(a->scratch) & (alignof(unsigned int) - 1)))
        return R4D_NO(RAD_E_SCRATCH);
    RuCompact cfg{(int)ab_stride, (int)state->stride[0], (int)state->stride[1],
                  (int)oq_row, (int)os_row, (int)sidx->stride[0], (int)z_row,
                  (int)xpitch, (int)cst->stride[0], (int)cst->stride[1],
                  (int)cst->stride[2], (int)cidx->stride[0],
                  neps, scale, thr, nact, (int)H, (int)Hg};
    static_assert(R4dParams<decltype(ru_compact_cv<2>)>::count == 20);
    static_assert(R4dParams<decltype(ru_compact_cv<2>)>::bytes() == 224);
    static_assert(R4dParams<decltype(ru_compact_cv<2>)>::count <= RAD_MAX_KPARAMS &&
                  R4dParams<decltype(ru_compact_cv<2>)>::bytes() <= RAD_KPARAM_BLOB);
    /* TWO REAL DISPATCHES, and the counter clear is the first of them. Emitting it through the
     * same sink is what makes `launch` and `describe` unable to disagree: a described sequence
     * carries both, so a graph replay clears the counters exactly as the direct path does. The
     * clear is ordered before the folded kernel on one stream, and the counters it writes are the
     * ones this dispatch's arrival protocol will read. */
    const int count = (int)((long long)N * Hg);
    const int clear_rc = R4D_EMIT(sk, ru_scratch_clear,
                                  dim3(((unsigned)count + 255u) / 256u), dim3(256u), 0,
                                  (unsigned int*)a->scratch, count);
    if (clear_rc != RAD_OK) return clear_rc;
    return R4D_EMIT(sk, ru_compact_cv<2>, dim3(N, (unsigned)H, 1), dim3(RU_V * 2), 0,
                    ta->data, tb->data, (const float*)alog->data, (const float*)dtb->data,
                    (float*)state->data, (unsigned short*)o->data,
                    (unsigned char*)oq->data, (float*)os->data, (const int*)cu->data,
                    (const int*)sidx->data, (const int*)nacc->data,
                    (const unsigned short*)z->data, (const float*)nw->data,
                    (const unsigned short*)x->data, (const unsigned short*)cw->data,
                    cb ? (const unsigned short*)cb->data : nullptr,
                    (unsigned short*)cst->data, (const int*)cidx->data,
                    (unsigned int*)a->scratch, cfg);
}

/* libr4d's compiled-in geometry, restated here. The arithmetic body calls r4d_gdn_dims to check
 * the geometry it was written for; libr4d defines that in the chunk scan's translation unit, which
 * this DSO does not build. These are the three numbers the body bakes in (RU_K, RU_V and the chunk
 * scan's 64), so this DSO cannot silently disagree with what it forked. It keeps the visibility
 * r4d.h declared it with -- the version script, not this attribute, keeps it out of the dynamic
 * table. */
extern "C" void r4d_gdn_dims(int* head_k, int* head_v, int* chunk) {
  if (head_k) *head_k = RU_K;
  if (head_v) *head_v = RU_V;
  if (chunk)  *chunk  = 64;
}

// The internal helpers above -- the device templates of r4d_common.h and r4d_gdn_wmma.h, the
// arithmetic body, the scratch clear and the geometry restatement -- are hidden. Everything below
// is what the loader may resolve; the version script narrows the dynamic symbol table to the six
// ABI exports.
#pragma GCC visibility pop

// Both entry points from one body: `gdn_query_reload_launch` performs the two dispatches and
// `_describe` appends the same two to a descriptor array. See libr4d/r4d_emit.h for why one body
// is the only way the two cannot drift.
R4D_ENTRY(gdn_query_reload_launch, ru_compact_entry)

namespace gqreload {

/* THE SHAPE THIS ROW IS WRITTEN FOR, as a selection predicate. Every key here is one the caller
 * already put in the query -- the geometry the op schema declares -- so a match is a statement
 * about the call and not about what happens to be in the tensors. */
static const RadConstraint constraints[] = {
    RAD_CEQ("head_k", 128), RAD_CEQ("head_v", 128), RAD_CEQ("conv_width", 4),
    RAD_CGE("q_len", 1), RAD_CLE("q_len", 16),
};

/* THE SCRATCH HOOK SEES PARAMETERS ONLY. A declare call has no tensors, so N and Hg are not
 * available here; the runtime enforces the same N * Hg bound the entry point checks, and what
 * this hook states is the ceiling. The constant maximum is kMaxConvCtr counters -- 16384 B --
 * which is exactly the bound the entry point refuses past. The engine reuses a scratch region and
 * does not zero it, so the entry point clears the live counters itself with a real dispatch before
 * every main dispatch; nothing here allocates, resets with HIP, or synchronizes. */
static int64_t scratch(const RadArgs*) { return (int64_t)kMaxConvCtr * sizeof(unsigned int); }

}  // namespace gqreload

/* THE SCHEMA IS LIBR4D'S. Declaring it here would make this DSO the schema owner for an op
 * libr4d already declares, and the first plugin to declare an op fixes its vocabulary -- a
 * re-declaration that drifted by one operand would be refused against libref for a reason that
 * has nothing to do with this row. The installed libr4d owns the op; this row implements it. */
extern "C" int rad_kernel_schema_count() { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }

/* ONE ROW. With RADIANCE_KERNELS=mine this DSO is not on the hierarchy and `rad_kernel_count` is
 * never asked; when the parent puts it there for an A/B, the hierarchy position is what selects it
 * ahead of the frozen row. The priority is intra-plugin only and there is one row, so it competes
 * with nothing. */
extern "C" int rad_kernel_count() { return 1; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    static const RadKernelInfo row = [] {
        RadKernelInfo r{};
        r.name    = "gdn_query_reload_conv_recurrent_k128_v128_bf16";
        r.op      = "gdn_conv_recurrent_update";
        r.family  = "gdn";
        r.computes = "the folded gdn decode: an emitted counter clear, then the conv in the "
                     "recurrent kernel's prologue and gating, qk l2norm closed once per token on "
                     "the first thread pair, the delta-rule update, gated rms norm and the folded "
                     "int8 quantiser; query and key are re-read from LDS by their consumers "
                     "instead of being held across the update";
        r.shape   = "head_k/head_v 128, conv width 4, 1..16 tokens per sequence, anchor state, "
                    "fp32 state cache, one workgroup per (sequence, value head), N * Hg <= 4096";
        r.dtypes  = "bf16 projection/conv/a/b/gate/output, f32 recurrent state, int8 quant output "
                    "and f32 scale; optional bf16 conv bias";
        r.domain  = RAD_DOMAIN_DEVICE;
        r.priority = 60;
        r.constraints   = gqreload::constraints;
        r.n_constraints = (int)(sizeof(gqreload::constraints) / sizeof(gqreload::constraints[0]));
        r.launch   = gdn_query_reload_launch;
        r.describe = gdn_query_reload_launch_describe;
        r.scratch  = gqreload::scratch;
        return r;
    }();
    return i == 0 ? &row : nullptr;
}

extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info() {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "gdn_query_reload", "1.0.0",
        "folded gdn decode conv+recurrent update with shared l2 scales, query/key LDS reload and "
        "an emitted scratch counter clear; RDNA4 (gfx1201)", "gfx1201"
    };
    return &info;
}
