// docs/OPS.md's DECOMPOSED sampler chain, on device.  Not r4d_sample_chain_f32 -- that is one
// workgroup doing temperature, softmax, top_k, top_p, min_p and the draw in vLLM's fixed order, and
// it is the right kernel when the order is fixed.  It is the wrong kernel for this engine: the
// chain the core issues is eleven ops it composes per request (core/sample/sampler.h), with
// penalties, DRY, a grammar bitmask, locally-typical and XTC stages the fused one does not have and
// with stages a request may skip.  "Fusion is a selection, not a compiler pass" (spec §2.3) cuts
// both ways: a library that ships only the fused form forces the fusion.
//
// So both exist.  This unit is what makes a served request's sampler run on the card at all;
// r4d_sample_chain_f32 stays for the speculative-verify path, which needs its three outputs.
//
// ============================== EVERY SCALAR IS PER ROW ==============================
//
// `params` is an array of RadSampleParams, one row per sampled position, uploaded once a step.  It
// is abi/rad_sample.h's struct, INCLUDED rather than restated, because it is a WIRE FORMAT between
// the core and this library and a silent disagreement is a sampler reading `top_p` out of `min_p`.
//
// ============================== WHAT IS EXACT AND WHY ==============================
//
// libref is the oracle (spec §17) and a sampler is the one place where "close" is worthless:
// two implementations that disagree on one candidate's survival produce different TOKENS, and the
// difference compounds.  So:
//
//   * THE TOP-K IS EXACT, and it is not ref's O(k*n_vocab) repeated-maximum scan either.  Eight
//     radix passes over a monotone 32-bit key: four to find the k-th largest logit, four more over
//     the token id to break a tie at that logit by the LOWER id -- which is ref's tie rule -- and
//     then one bitonic sort of the k survivors, because every stage after this one assumes
//     descending order.  Exact, deterministic, and nine reads of the row instead of a thousand.
//
//   * THE SEQUENTIAL STAGES STAY SEQUENTIAL.  `top_p` accumulates a cumulative probability in f32
//     in candidate order and `sample_pick` walks an inverse CDF the same way.  A parallel prefix
//     sum would be a different rounding, so a candidate near the cut would survive in one
//     implementation and not the other.  Both run on one thread over at most a couple of thousand
//     candidates, which costs nothing next to the vocabulary passes around them.
//
//   * THE DRAW IS THE SPECIFIED ONE.  splitmix64 over (seed * golden + pos), top 24 bits as a float
//     in [0,1) -- rad_sample.h fixes it so that a host reference, a device kernel and a rerun
//     produce the same stream.  Restated here, bit for bit.
//
//   * expf / logf / fabsf ARE THE ACCURATE FORMS, not __expf.  Ref takes libm and these are
//     compared against it.
//
// A DROPPED CANDIDATE IS index -1, value -infinity, and the buffers never shrink.  Kept candidates
// stay where they are, in descending order, so a later stage may stop at the first hole or skip it;
// these skip.  `cand_val` carries LOGITS and each narrowing stage takes its own softmax over the
// live set, which is what keeps the stages composable in any order.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"

#include <cfloat>

// THE PARAMS ROW IS THE ENGINE'S STRUCT, not a restatement of it. Two independent declarations of
// one wire format drift, and the drift is silent: a sampler reading `top_p` out of `min_p`
// produces text, not an error. A kernel library that can include the ABI header has no reason to
// keep a second declaration and a field-by-field static_assert to police it.
#include "rad_sample.h"

namespace {

constexpr uint32_t kWave    = 32;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kWaves   = kThreads / kWave;

// THE VOCABULARY SCAN GETS ITS OWN WIDTH, and it is the only stage that needs one.  Every kernel
// below `topk_kernel` reads a CANDIDATE set -- a thousand or two words -- where 256 threads is
// already more than the work; `topk_kernel` reads the whole row, 124160 floats a row on this
// model, and at 256 threads that is 117.6 us a step for 3.97 MB -- 34 GB/s on a
// part that streams 620.  It is not bandwidth: one block a row is eight blocks on a 64-CU card,
// and 256 threads is two waves a SIMD, which cannot keep enough loads in flight to cover the
// latency of even one CU.  1024 threads is eight waves a SIMD on the same one block a row.
//
// IT DOES NOT CHANGE A BIT, and that is a property of this kernel rather than of the file.  The
// scan reduces with max over a packed (key, ~id) word and the select counts histogram bins -- both
// commutative and associative, so the answer does not depend on how the row was cut up.  The
// narrowing stages below are NOT like this: `sum_block` folds lds[0..kWaves) in wave order and a
// float sum is not associative, so widening THOSE would move the last bits of top_p and min_p.
// That is why this is a second constant and not a larger `kThreads`.
/* Four times kThreads, for the reason argmax_kernel's header gives: the grid is one workgroup a
 * row whatever happens, so the only knob left is how many loads that workgroup has in flight. */
constexpr uint32_t kArgmaxThreads = 1024;
constexpr uint32_t kTopkThreads = 1024;
constexpr uint32_t kTopkWaves   = kTopkThreads / kWave;

// The largest candidate width this unit sorts.  The bitonic network runs in LDS over a
// power-of-two padding, so the bound is LDS: 4096 x 8 bytes = 32 KiB, half the 64 KiB a workgroup
// may have.  The engine declares 1024 (core/sample/sampler.h sizes it from max_num_seqs), so this
// is four doublings of headroom; the shim refuses anything wider rather than truncating a set.
constexpr uint32_t kMaxCand = 4096;

// ---------------------------------------------------------------------------- the params row
// One declaration, the engine's. See the note at the include.
using SP = RadSampleParams;

enum {
    SP_GREEDY  = 1 << 0,
    SP_PENALTY = 1 << 1,
    SP_MASK    = 1 << 2,
    SP_XTC     = 1 << 3,
    SP_DRY     = 1 << 4
};

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ __forceinline__ float uniform01(uint64_t seed, uint64_t pos) {
    const uint64_t z = splitmix64(seed * 0x9E3779B97F4A7C15ull + pos);
    return (float)(uint32_t)(z >> 40) * (1.0f / 16777216.0f);
}

// A float's ORDER-PRESERVING 32-bit key.  Flipping the sign bit of a positive and every bit of a
// negative makes unsigned comparison agree with float comparison, which is what lets a radix select
// work on logits at all.  -infinity maps to 0, so a dropped candidate sorts last without a branch.
__device__ __forceinline__ uint32_t fkey(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float unfkey(uint32_t key) {
    const uint32_t u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
    return __builtin_bit_cast(float, u);
}

// The wave's largest (key, ~index), by an xor butterfly on r4d_xor_lane's exchange. A maximum of
// integers has one answer in any order.
__device__ __forceinline__ unsigned long long wave_max_u64(unsigned long long b) {
    unsigned long long o;
    o = r4d_xor_lane_u64<16>(b); if (o > b) b = o;
    o = r4d_xor_lane_u64<8>(b);  if (o > b) b = o;
    o = r4d_xor_lane_u64<4>(b);  if (o > b) b = o;
    o = r4d_xor_lane_u64<2>(b);  if (o > b) b = o;
    o = r4d_xor_lane_u64<1>(b);  if (o > b) b = o;
    return b;
}

__device__ __forceinline__ float block_sum(float v, float* lds) {
    v = r4d_wave_sum(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (uint32_t w = 0; w < kWaves; ++w) s += lds[w];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float v, float* lds) {
    v = r4d_wave_max(v);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = v;
    __syncthreads();
    float s = -INFINITY;
#pragma unroll
    for (uint32_t w = 0; w < kWaves; ++w) s = fmaxf(s, lds[w]);
    __syncthreads();
    return s;
}

// ---------------------------------------------------------------------------- sample_temp
// ref_sample_temp.  temp <= 0 leaves the row alone: greedy is `sample_argmax`, and a temperature of
// zero quietly meaning something else is how a sampler chain becomes unreadable.
__global__ __launch_bounds__(kThreads) void temp_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, uint32_t n_vocab, uint32_t lg_ld) {
    const uint32_t m = blockIdx.x;
    const float t = sp[m].temp;
    if (!(t > 0.0f) || t == 1.0f) return;
    const float inv = 1.0f / t;
    float* __restrict__ row = lg + (size_t)m * lg_ld;
    for (uint32_t i = threadIdx.x + blockIdx.y * kThreads; i < n_vocab; i += kThreads * gridDim.y)
        row[i] *= inv;
}

// ---------------------------------------------------------------------------- sample_mask
// ref_sample_mask.  A SET BIT MEANS ALLOWED -- a grammar produces the admissible set, and the mask
// is that set rather than its complement.  Word w bit b covers token 32w + b, least significant
// first.  `mask_row` is -1 for an unconstrained sequence, which is why the plane is shared and
// shorter than M.
__global__ __launch_bounds__(kThreads) void mask_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const uint32_t* __restrict__ bm,
    uint32_t n_vocab, uint32_t lg_ld, uint32_t words, int32_t mask_rows) {
    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    if (!(p.flags & SP_MASK)) return;
    if (p.mask_row < 0 || p.mask_row >= mask_rows) return;
    float* __restrict__ row = lg + (size_t)m * lg_ld;
    const uint32_t* __restrict__ mrow = bm + (size_t)p.mask_row * words;
    for (uint32_t i = threadIdx.x + blockIdx.y * kThreads; i < n_vocab; i += kThreads * gridDim.y)
        if (!((mrow[i >> 5] >> (i & 31)) & 1u)) row[i] = -INFINITY;
}

// ---------------------------------------------------------------------------- sample_penalties
// ref_sample_penalties, in llama.cpp's order:
//
//     repetition   logit = logit > 0 ? logit / rep : logit * rep    (once per DISTINCT token)
//     frequency    logit -= count * freq
//     presence     logit -= pres                                     (once per distinct token)
//
// The repetition penalty being multiplicative and sign-dependent is the part everyone gets wrong:
// dividing a negative logit would make it larger.
//
// PARALLEL OVER THE HISTORY AND STILL RACE-FREE, which is worth stating because it looks like a
// scatter.  Only the FIRST occurrence of a token updates it, so every distinct token has exactly
// one writing thread; two threads writing different tokens do not collide.  That is the same
// property that makes ref's `first` test necessary, used for a second purpose.
//
// History ids are GLOBAL and the row is this rank's shard: a token's column is its id less
// `voff`, and a token outside the shard is another rank's to penalise.

// The window both history stages read, ref's hist_window: the row's own row of the plane, clamped,
// then the TAIL `last_n` long.  last_n 0 turns the stage off for the row (false); negative keeps
// the whole span.
__device__ __forceinline__ bool hist_window(const SP& p, int32_t last_n, uint32_t hmax,
                                            int32_t* ho_out, int32_t* hl_out) {
    if (last_n == 0) return false;
    int32_t hl = p.hist_len;
    int32_t ho = p.hist_off < 0 ? 0 : p.hist_off;
    if (ho > (int32_t)hmax) ho = (int32_t)hmax;
    if (hl > (int32_t)hmax - ho) hl = (int32_t)hmax - ho;
    if (last_n > 0 && hl > last_n) {
        ho += hl - last_n;
        hl = last_n;
    }
    *ho_out = ho;
    *hl_out = hl;
    return hl > 0;
}

__global__ __launch_bounds__(kThreads) void penalties_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const int32_t* __restrict__ hist,
    uint32_t n_vocab, uint32_t lg_ld, uint32_t hmax, int32_t voff) {
    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    if (!(p.flags & SP_PENALTY)) return;
    const float rep = p.rep_penalty, freq = p.freq_penalty, pres = p.pres_penalty;
    if (rep == 1.0f && freq == 0.0f && pres == 0.0f) return;

    int32_t ho = 0, hl = 0;
    if (!hist_window(p, p.penalty_last_n, hmax, &ho, &hl)) return;

    const int32_t* __restrict__ h = hist + (size_t)m * hmax + ho;
    float* __restrict__ row = lg + (size_t)m * lg_ld;

    for (int32_t i = (int32_t)threadIdx.x; i < hl; i += (int32_t)kThreads) {
        const int32_t tok = h[i];
        const int64_t col = (int64_t)tok - voff;
        if (tok < 0 || col < 0 || col >= (int64_t)n_vocab) continue;
        bool  first = true;
        int32_t count = 0;
        for (int32_t j = 0; j < hl; ++j) {
            if (h[j] != tok) continue;
            if (j < i) { first = false; break; }
            ++count;
        }
        if (!first) continue;
        float v = row[col];
        v = v > 0.0f ? v / rep : v * rep;
        v -= (float)count * freq;
        v -= pres;
        row[col] = v;
    }
}

// ---------------------------------------------------------------------------- sample_dry
// ref_sample_dry, which is llama.cpp's DRY: every token that would EXTEND a suffix of the window
// that has occurred before is penalised ONCE, by multiplier * base^(L - allowed) for the LONGEST
// repeat L it would extend, repeats capped at the row's dry_rep_limit, and a position the optional
// `brk` plane marks -- a whole sequence breaker -- is never penalised.
//
// ONE BLOCK A ROW, TWO PHASES, AND NO ATOMICS.  Phase one is parallel over the earlier positions j:
// each thread measures how long a suffix of the window also ends at j and leaves it in LDS, zero
// where the continuation after j is exempt or another rank's.  Phase two picks, for every
// continuation token, ONE representative position -- its longest repeat, the earliest on a tie --
// and only that thread writes the token's logit.  A token therefore has exactly one writer and one
// subtraction, which is both llama.cpp's rule and what keeps the answer independent of thread
// order; summing a penalty per position, or an atomic sum, would give neither.
//
// The cost is quadratic only in a history that really is repetitive: phase one's match loop exits
// on the first mismatch, and phase two scans only for positions whose repeat already qualifies.
// The penalty is formed by repeated multiplication stopping short of overflow, exactly as ref
// forms it, so the two agree to the bit.
__device__ __forceinline__ float dry_penalty(float multiplier, float base, int32_t e) {
    float pen = multiplier;
    if (base == 1.0f) return pen;
    const float lim = FLT_MAX / base;
    for (int32_t i = 0; i < e; ++i) {
        if (pen > lim) break;
        pen *= base;
    }
    return pen;
}

__global__ __launch_bounds__(kThreads) void dry_kernel(
    float* __restrict__ lg, const SP* __restrict__ sp, const int32_t* __restrict__ hist,
    const uint8_t* __restrict__ brk, uint32_t n_vocab, uint32_t lg_ld, uint32_t hmax,
    int32_t voff) {
    extern __shared__ uint16_t len[];
    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    // Every test below reads only the row, so the whole block leaves together.
    if (!(p.flags & SP_DRY) || !(p.dry_multiplier > 0.0f)) return;
    const int32_t allowed = p.dry_allowed_length > 0 ? p.dry_allowed_length : 2;

    int32_t ho = 0, hl = 0;
    if (!hist_window(p, p.dry_penalty_last_n, hmax, &ho, &hl)) return;
    if (hl <= allowed) return;
    const int32_t cap = p.dry_rep_limit > 0 && p.dry_rep_limit < hl ? p.dry_rep_limit : hl;
    if (cap < allowed) return;

    const int32_t* __restrict__ h = hist + (size_t)m * hmax + ho;
    const uint8_t* __restrict__ f = brk ? brk + (size_t)m * hmax + ho : nullptr;
    float* __restrict__ row = lg + (size_t)m * lg_ld;

    for (int32_t j = (int32_t)threadIdx.x; j + 1 < hl; j += (int32_t)kThreads) {
        const int32_t lim = j + 1 < cap ? j + 1 : cap;
        int32_t n = 0;
        while (n < lim && h[j - n] == h[hl - 1 - n]) ++n;
        const int64_t col = (int64_t)h[j + 1] - voff;
        const bool skip = (f && f[j + 1]) || h[j + 1] < 0 || col < 0 || col >= (int64_t)n_vocab;
        len[j] = skip ? (uint16_t)0 : (uint16_t)n;
    }
    __syncthreads();

    const float base = p.dry_base > 0.0f ? p.dry_base : 1.75f;
    for (int32_t j = (int32_t)threadIdx.x; j + 1 < hl; j += (int32_t)kThreads) {
        const int32_t n = len[j];
        if (n < allowed) continue;
        const int32_t next = h[j + 1];
        bool rep = true;
        for (int32_t j2 = 0; j2 + 1 < hl; ++j2) {
            if (j2 == j || h[j2 + 1] != next) continue;
            const int32_t n2 = len[j2];
            if (n2 > n || (n2 == n && j2 < j)) { rep = false; break; }
        }
        if (!rep) continue;
        const int64_t col = (int64_t)next - voff;
        row[col] -= dry_penalty(p.dry_multiplier, base, n - allowed);
    }
}

// ---------------------------------------------------------------------------- sample_argmax
// ref_sample_argmax.  Greedy, TIES TO THE LOWER TOKEN ID -- the same rule the top-k selection uses,
// so a greedy step and a top-1 step cannot disagree.  Packing (key, ~id) into one 64-bit word makes
// that tie rule fall out of a plain unsigned max reduction.
//
// ONE WORKGROUP A ROW, AND IT STAYS THAT WAY -- so the row has to be read WIDE.  A greedy step
// samples one row, so the grid is one workgroup and the other 31 CUs are idle no matter what.  At
// Qwen3.8-Flash-Next's 248320-entry vocabulary that one workgroup reads 993 KB, and at 256 threads
// of scalar loads it took 170 us to do it -- 5.8 GB/s, which is not a bandwidth number, it is a
// not-enough-loads-in-flight number.  Four times the threads and a float4 load each is 15.5 us at
// 64 GB/s, an eleven-fold cut for the same single workgroup and the same one dispatch.
//
// SPLITTING THE ROW ACROSS WORKGROUPS IS FASTER AND STILL NOT WORTH IT.  A two-stage form over 128
// partials measures 7.3 us, but it needs a scratch plane through the ABI and a second launch, and
// this tree's per-dispatch floor is ~4.5 us -- so the split lands at ~12 us against this 15.5 and
// costs an operand for the difference.
//
// The vector path needs the ROW aligned, which is the base and the pitch together; `VEC` is chosen
// by the launcher and the scalar body is the fallback, not a legacy path.
template <uint32_t TH, bool VEC>
__global__ __launch_bounds__(TH) void argmax_kernel(
    const float* __restrict__ lg, int32_t* __restrict__ tok, uint32_t n_vocab, uint32_t lg_ld) {
    __shared__ unsigned long long lds[TH / kWave];
    const uint32_t m = blockIdx.x;
    const float* __restrict__ row = lg + (size_t)m * lg_ld;

    unsigned long long best = 0;
    if (VEC) {
        const uint32_t  n4 = n_vocab >> 2;
        const float4* __restrict__ r4 = (const float4*)row;
        for (uint32_t j = threadIdx.x; j < n4; j += TH) {
            const float4   v = r4[j];
            const uint32_t i = j << 2;
            unsigned long long e;
            e = ((unsigned long long)fkey(v.x) << 32) | (unsigned long long)(~(i + 0u));
            if (e > best) best = e;
            e = ((unsigned long long)fkey(v.y) << 32) | (unsigned long long)(~(i + 1u));
            if (e > best) best = e;
            e = ((unsigned long long)fkey(v.z) << 32) | (unsigned long long)(~(i + 2u));
            if (e > best) best = e;
            e = ((unsigned long long)fkey(v.w) << 32) | (unsigned long long)(~(i + 3u));
            if (e > best) best = e;
        }
        /* The <4 entries a vocabulary that is not a multiple of four leaves over. */
        for (uint32_t i = (n4 << 2) + threadIdx.x; i < n_vocab; i += TH) {
            const unsigned long long e =
                ((unsigned long long)fkey(row[i]) << 32) | (unsigned long long)(~i);
            if (e > best) best = e;
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n_vocab; i += TH) {
            const unsigned long long e =
                ((unsigned long long)fkey(row[i]) << 32) | (unsigned long long)(~i);
            if (e > best) best = e;
        }
    }
    best = wave_max_u64(best);
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    if (lane == 0) lds[wave] = best;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (uint32_t w = 1; w < TH / kWave; ++w) if (lds[w] > lds[0]) lds[0] = lds[w];
    tok[m] = (int32_t)(~(uint32_t)(lds[0] & 0xffffffffull));
}

// ---------------------------------------------------------------------------- sample_topk
//
// THE RADIX SELECT.  Four passes over the row, most significant byte first, narrowing a prefix of
// the monotone key until the k-th largest is pinned exactly.  `remaining` is how many of the k are
// still unaccounted for once every strictly-larger prefix has been counted; at the end it is how
// many of the elements EQUAL to the threshold belong in the set.
//
// Then, ONLY WHEN THE TIE GROUP IS BIGGER THAN THE SET IS OWED, four more passes over the token
// id, restricted to that group, to find which of them --
// the LOWEST ids, ref's rule.  Ids are unique, so `idx <= I` takes exactly the right number and
// there is no second tie to break.
// DESC selects on the LOGIT KEY and walks the bins from 255 down; ASC selects on the TOKEN ID and
// walks them from 0 up.  `gate` is the tie group the ASC pass is restricted to -- only elements
// whose key equals it take part -- and is ignored by the DESC pass, which sees the whole row.
//
// FOUR AT A TIME, AND THE DIGIT FROM A SCAN RATHER THAN A WALK.  The row is one workgroup's to read,
// so how many bytes each load instruction carries is what the pass runs at; a float4 is four times
// what a float is for the same wait, and the guard is the shape's, as in the top-1 path below.
// And a digit chosen by ONE thread walking the 256 bins keeps 1023 waiting at the barrier, four
// times a select: the bins' running sums in the walk's direction (a suffix sum for DESC, a
// prefix sum for ASC) are a DPP scan a wave and one add across the eight waves, and the bin the walk
// stops at is the one whose running sum first reaches `remaining` -- exactly one, because the sums
// are monotone -- with `cum` the sum before it. A walk that never stops ends on the last bin with
// every bin counted, and so does this. Counts are integers, so it is the same digit and the same
// `cum`. `scan` holds 256 words and `wt` eight.
template <bool DESC>
__device__ uint32_t radix_select(const float* __restrict__ row, uint32_t n, uint32_t gate,
                                 uint32_t rank, uint32_t* hist, uint32_t* scan, uint32_t* wt,
                                 uint32_t* remaining_out, uint32_t* tie_out) {
    const uint32_t t = threadIdx.x;
    const int      lane = (int)(t & (kWave - 1)), wave = (int)(t / kWave);
    const bool     v4 = ((uintptr_t)row & 15u) == 0 && (n & 3u) == 0;
    uint32_t prefix = 0, mask = 0, remaining = rank;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint32_t b = t; b < 256; b += kTopkThreads) hist[b] = 0;
        __syncthreads();
        auto count = [&](uint32_t key, uint32_t i) {
            if (!DESC && key != gate) return;
            const uint32_t v = DESC ? key : i;
            if ((v & mask) != prefix) return;
            atomicAdd(&hist[(v >> shift) & 0xffu], 1u);
        };
        if (v4) {
            const float4* __restrict__ r4 = (const float4*)row;
            for (uint32_t j = t; j < (n >> 2); j += kTopkThreads) {
                const float4 f = r4[j];
                count(fkey(f.x), 4 * j);
                count(fkey(f.y), 4 * j + 1);
                count(fkey(f.z), 4 * j + 2);
                count(fkey(f.w), 4 * j + 3);
            }
        } else {
            for (uint32_t i = t; i < n; i += kTopkThreads) count(fkey(row[i]), i);
        }
        __syncthreads();
        uint32_t x = 0;
        if (t < 256) {
            x = DESC ? r4d_wave_suffix_u32(hist[t], lane) : r4d_wave_prefix_u32(hist[t], lane);
            if (lane == (DESC ? 0 : kWave - 1)) wt[wave] = x;
        }
        __syncthreads();
        if (t < 256) {
            uint32_t add = 0;
            if (DESC) for (int w = wave + 1; w < 256 / (int)kWave; ++w) add += wt[w];
            else      for (int w = 0; w < wave; ++w) add += wt[w];
            scan[t] = x + add;
        }
        __syncthreads();
        if (t < 256) {
            const uint32_t before = DESC ? (t < 255 ? scan[t + 1] : 0u) : (t > 0 ? scan[t - 1] : 0u);
            if (scan[t] >= remaining && before < remaining) { hist[256] = t; hist[257] = before; }
        }
        if (t == 0) {
            const uint32_t total = DESC ? scan[0] : scan[255];
            if (total < remaining) { hist[256] = DESC ? 0u : 255u; hist[257] = total; }
        }
        __syncthreads();
        const uint32_t b = hist[256], cum = hist[257];
        if (tie_out) *tie_out = hist[b];
        remaining -= cum;
        prefix |= b << shift;
        mask |= 0xffu << shift;
        __syncthreads();
    }
    *remaining_out = remaining;
    return prefix;
}

template <bool PAIR_ONLY = false>
__global__ __launch_bounds__(kTopkThreads) void topk_kernel(
    const float* __restrict__ lg, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv, uint32_t* __restrict__ pr,
    uint32_t n_vocab, uint32_t n_cand, uint32_t lg_ld, uint32_t ci_ld, uint32_t cv_ld,
    uint32_t pr_ld, int32_t vocab_off, uint32_t pad) {
    extern __shared__ unsigned long long shared_cand[];
    __shared__ uint32_t hist[258];
    __shared__ uint32_t scan[256], wt[256 / kWave];
    __shared__ uint32_t fill;
    __shared__ unsigned long long lds[kTopkWaves];

    const uint32_t m = blockIdx.x;
    const float* __restrict__ row = lg + (size_t)m * lg_ld;
    int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    // A GREEDY ROW HAS NO CANDIDATES: its token is sample_argmax's, and in a mixed batch every
    // stage after this one skips it.  Holes, written without reading the row -- the vocabulary
    // pass a greedy row would otherwise cost is the whole price of this kernel.  The test reads
    // only the row's params, so the whole block leaves together.
    if (sp[m].flags & SP_GREEDY) {
        uint32_t* __restrict__ gp = pr ? pr + (size_t)m * pr_ld : nullptr;
        for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) {
            if (!PAIR_ONLY) {
                oi[r] = -1;
                ov[r] = -INFINITY;
            }
            if (gp) {
                gp[2 * r]     = 0xffffffffu;
                gp[2 * r + 1] = __builtin_bit_cast(uint32_t, -INFINITY);
            }
        }
        return;
    }

    // A row's own top_k, clamped to the buffer AND to the vocabulary.  0 or negative means the
    // whole candidate width, which is what a request that set no top_k asks for.
    uint32_t k = (uint32_t)(sp[m].top_k > 0 ? sp[m].top_k : (int32_t)n_cand);
    if (k > n_cand) k = n_cand;
    if (k > n_vocab) k = n_vocab;

    // TOP-1 IS AN ARGMAX AND DOES NOT NEED THE SELECT.  A greedy row over a split vocabulary
    // arrives here at k = 1 and not at sample_argmax -- core/sample/sampler.cpp stages it that way
    // so the merge can turn this rank's local index into a global one -- and the general path then
    // spends EIGHT passes over the vocabulary and a bitonic sort of the whole candidate width to
    // find a single element -- on the 27B at tp 2 that is 896 us a step, 2.7% of decode, for one
    // maximum.  The packed (key, ~id) word below is the one argmax_kernel reduces, so the two
    // agree by construction, and its tie rule IS the select's tie rule -- see both comments above.
    uint32_t got;
    if (k == 1) {
        unsigned long long best = 0;
        // FOUR AT A TIME while the row and the pointer allow it.  A thread that loads one float an
        // iteration issues one 4-byte request and then waits on it; the scan is latency-bound (see
        // kTopkThreads), so what it needs is more bytes in flight per instruction, and a float4 is
        // four times as many for the same wait.  The guard is the shape's, not the loop's: `row`
        // moves by lg_ld between rows, so a row is 16-byte aligned only when the base and the
        // stride both are, and n_vocab need not be a multiple of four.  When either fails this
        // falls back to the scalar form, which is what a vocabulary that is not 4-aligned gets.
        const bool v4 = ((uintptr_t)row & 15u) == 0 && (n_vocab & 3u) == 0;
        if (v4) {
            const float4* __restrict__ r4 = (const float4*)row;
            const uint32_t n4 = n_vocab >> 2;
            for (uint32_t j = threadIdx.x; j < n4; j += kTopkThreads) {
                const float4 f = r4[j];
                const uint32_t i0 = j << 2;
                const float fs[4] = { f.x, f.y, f.z, f.w };
#pragma unroll
                for (uint32_t c = 0; c < 4; ++c) {
                    const unsigned long long e = ((unsigned long long)fkey(fs[c]) << 32) |
                                                 (unsigned long long)(~(i0 + c));
                    if (e > best) best = e;
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < n_vocab; i += kTopkThreads) {
                const unsigned long long e =
                    ((unsigned long long)fkey(row[i]) << 32) | (unsigned long long)(~i);
                if (e > best) best = e;
            }
        }
        best = wave_max_u64(best);
        const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
        if (lane == 0) lds[wave] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            for (uint32_t w = 1; w < kTopkWaves; ++w) if (lds[w] > lds[0]) lds[0] = lds[w];
            shared_cand[0] = lds[0];
        }
        __syncthreads();
        got = 1;
    } else {
        // `n_eq` is how many of the k are still owed once every strictly-larger logit has been
        // counted: exactly the number to take from the tie group at the threshold, and always at
        // least one.  `n_tie` is how big that group is.
        uint32_t n_eq = 0, dummy = 0, n_tie = 0;
        const uint32_t T = radix_select<true>(row, n_vocab, 0, k, hist, scan, wt, &n_eq, &n_tie);

        // THE TIE-BREAK PASS ONLY RUNS WHEN THERE IS A TIE TO BREAK.  `n_tie` is how many elements
        // carry the threshold key exactly and `n_eq` how many of them the set is owed; when they
        // are equal the whole group is in and there is nothing for the ASC select to choose
        // between, so the four passes it would cost are four passes over 124160 logits for an
        // answer already known.  Distinct logits are the ordinary case, and `i <= 0xffffffff` is
        // then true for every member -- the same set, without the search.
        const uint32_t idx_cut = (n_tie == n_eq)
                                     ? 0xffffffffu
                                     : radix_select<false>(row, n_vocab, T, n_eq, hist, scan, wt,
                                                           &dummy, &n_tie);

        if (threadIdx.x == 0) fill = 0;
        __syncthreads();
        /* Four at a time, for the reason radix_select gives. The slots are claimed in whatever
         * order the atomics land, and the sort below puts the set in (key, ~id) order, so the
         * candidates come out the same whatever that order was. */
        auto collect = [&](float f, uint32_t i) {
            const uint32_t key = fkey(f);
            const bool take = key > T || (key == T && i <= idx_cut);
            if (!take) return;
            const uint32_t slot = atomicAdd(&fill, 1u);
            if (slot < kMaxCand)
                shared_cand[slot] = ((unsigned long long)key << 32) | (unsigned long long)(~i);
        };
        if (((uintptr_t)row & 15u) == 0 && (n_vocab & 3u) == 0) {
            const float4* __restrict__ r4 = (const float4*)row;
            for (uint32_t j = threadIdx.x; j < (n_vocab >> 2); j += kTopkThreads) {
                const float4 f = r4[j];
                collect(f.x, 4 * j);
                collect(f.y, 4 * j + 1);
                collect(f.z, 4 * j + 2);
                collect(f.w, 4 * j + 3);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < n_vocab; i += kTopkThreads) collect(row[i], i);
        }
        __syncthreads();
        got = fill < kMaxCand ? fill : kMaxCand;

        // SORTED OVER THE LIVE SET, NOT THE CANDIDATE WIDTH.  The select puts exactly `got`
        // elements in the low slots and the output loop reads none above them, so a request at
        // top_k = 20 sorts 32 words rather than the 4096 the buffer is wide.
        uint32_t sn = 1;
        while (sn < got) sn <<= 1;
        for (uint32_t i = got + threadIdx.x; i < sn; i += kTopkThreads) shared_cand[i] = 0;
        __syncthreads();

        // Bitonic sort, descending.  The packed word is (key, ~id), so a plain descending sort puts
        // the largest logit first and, among equal logits, the LOWER id first -- ref's tie rule,
        // again for free.  Padding words are 0 and sort to the tail, where they are written as
        // holes.
        for (uint32_t sz = 2; sz <= sn; sz <<= 1) {
            for (uint32_t st = sz >> 1; st > 0; st >>= 1) {
                for (uint32_t t = threadIdx.x; t < sn; t += kTopkThreads) {
                    const uint32_t partner = t ^ st;
                    if (partner <= t) continue;
                    const bool up = (t & sz) == 0;                 // `up` here means DESCENDING
                    const unsigned long long a = shared_cand[t], b = shared_cand[partner];
                    if (up ? (a < b) : (a > b)) { shared_cand[t] = b; shared_cand[partner] = a; }
                }
                __syncthreads();
            }
        }
    }
    (void)pad;

    // THE CANDIDATE INDEX IS LOCAL AND THE PAIR PLANE IS GLOBAL, deliberately. ci/cv index `row`,
    // which is this rank's slice of the vocabulary, and every stage after top-k reads them that
    // way. `pr` is the vocab-parallel path's payload instead: (u32 GLOBAL id, f32 logit) pairs, in
    // the layout sample_merge_topk reads, with vocab_off already added -- so the all-gather in
    // between moves bytes it does not have to understand, and the single-rank path is unchanged
    // because it passes pr = null.
    uint32_t* __restrict__ op = pr ? pr + (size_t)m * pr_ld : nullptr;
    for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) {
        const bool live = r < got && r < k;
        const unsigned long long e = live ? shared_cand[r] : 0ull;
        const uint32_t loc = (uint32_t)(~(uint32_t)(e & 0xffffffffull));
        const float    val = live ? unfkey((uint32_t)(e >> 32)) : -INFINITY;
        if (!PAIR_ONLY) {
            oi[r] = live ? (int32_t)loc : -1;
            ov[r] = val;
        }
        if (op) {
            // 0xffffffff is the hole: the merge tests the id as a SIGNED int and drops it when
            // negative, which is the same convention ci carries as -1.
            op[2 * r]     = live ? (uint32_t)((int32_t)loc + vocab_off) : 0xffffffffu;
            op[2 * r + 1] = __builtin_bit_cast(uint32_t, val);
        }
    }
}

// ---------------------------------------------------------------------------- the narrowing stages
//
// All four read a fixed-width candidate set in place: index -1 and value -infinity mark a hole, and
// the live set keeps its descending order.  n_cand is a thousand or two, so one block a row with a
// couple of block reductions is the whole cost.

enum { NR_TOPP = 0, NR_MINP, NR_TYPICAL, NR_XTC };

__global__ __launch_bounds__(kThreads) void narrow_kernel(
    int32_t* __restrict__ ci, float* __restrict__ cv, const SP* __restrict__ sp,
    uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld, int which) {
    extern __shared__ float lds_exp[];
    __shared__ float red[kWaves];
    __shared__ float sh_mx, sh_inv, sh_H;
    __shared__ int   sh_cut_bits, sh_have_cut, sh_last, sh_above;

    const uint32_t m = blockIdx.x;
    const SP p = sp[m];
    // A greedy row has no candidates to narrow; the whole block leaves together.
    if (p.flags & SP_GREEDY) return;
    int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (oi[i] >= 0) mx = fmaxf(mx, ov[i]);
    mx = block_max(mx, red);
    if (mx == -INFINITY) return;
    if (threadIdx.x == 0) sh_mx = mx;
    __syncthreads();

    if (which == NR_MINP) {
        // Keep everything within a factor min_p of the most likely candidate.  Comparing
        // exp(v - max) against p is the same test as prob >= p * max_prob without forming the
        // normaliser: one pass fewer and one rounding fewer.
        if (!(p.min_p > 0.0f)) return;
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (oi[i] < 0) continue;
            const float v = ov[i];
            if (v == mx) continue;                       // the max always survives
            if (expf(v - mx) < p.min_p) { oi[i] = -1; ov[i] = -INFINITY; }
        }
        return;
    }

    // The other three need the normalised probabilities.
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        if (oi[i] >= 0) {
            const float e = expf(ov[i] - mx);
            if (which == NR_TOPP) lds_exp[i] = e;
            sum += e;
        } else {
            if (which == NR_TOPP) lds_exp[i] = 0.0f;
        }
    }
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) sh_inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();
    const float inv = sh_inv;

    if (which == NR_TOPP) {
        // ONE THREAD FINDS THE CUT, and deliberately: it is the shortest prefix whose cumulative
        // probability reaches top_p, accumulated in f32 in candidate order.  A parallel prefix sum
        // sums in a different order and a candidate at the boundary would survive here and not in
        // ref.
        //
        // BUT ONLY THE CUT IS SERIAL, and the removal after it is not. The scan stops at the
        // candidate that reaches top_p; every live candidate past it goes, which the whole block
        // does. Walking the rest on the one thread costs a dependent global load per candidate --
        // ~50 us a row at 1024 candidates, on every step of a greedy request over a split
        // vocabulary, whose top-1 row reaches any top_p on its first candidate.
        if (!(p.top_p < 1.0f) || !(p.top_p > 0.0f)) return;
        __shared__ uint32_t sh_cut;
        if (threadIdx.x == 0) {
            float cum = 0.0f;
            uint32_t cut = n_cand;
            for (uint32_t i = 0; i < n_cand; ++i) {
                if (oi[i] < 0) continue;
                cum += lds_exp[i] * inv;
                if (cum >= p.top_p) { cut = i + 1; break; }   // always keep at least one
            }
            sh_cut = cut;
        }
        __syncthreads();
        for (uint32_t i = sh_cut + threadIdx.x; i < n_cand; i += kThreads)
            if (oi[i] >= 0) { oi[i] = -1; ov[i] = -INFINITY; }
        return;
    }

    if (which == NR_TYPICAL) {
        // Locally typical: keep the candidates whose surprisal is closest to the distribution's
        // entropy until their mass reaches typical_p.  NOT a prefix -- the candidate order is by
        // probability and this order is by |surprisal - H| -- so it is a threshold on the distance,
        // ties kept, which can overshoot p slightly.  The alternative is an order that depends on
        // the candidate index, which is worse.
        if (!(p.typical_p < 1.0f) || !(p.typical_p > 0.0f)) return;
        float H = 0.0f;
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (oi[i] < 0) continue;
            const float q = expf(ov[i] - mx) * inv;
            if (q > 0.0f) H -= q * logf(q);
        }
        H = block_sum(H, red);
        if (threadIdx.x == 0) {
            sh_H = H;
            /* A distance is non-negative, so its f32 bit pattern is a non-negative int and orders
             * the same way -- which is what lets the minimum be an integer atomicMin. */
            sh_cut_bits = __float_as_int(INFINITY);
            sh_have_cut = 0;
        }
        __syncthreads();

        // The smallest distance threshold whose mass reaches p, over the distances that actually
        // occur.  Quadratic in n_cand and parallel over the outer index; the reduction is a
        // minimum, so it is order-independent and reproduces.
        float local_cut = INFINITY;
        bool  local_have = false;
        for (uint32_t c = threadIdx.x; c < n_cand; c += kThreads) {
            if (oi[c] < 0) continue;
            const float qc = expf(ov[c] - sh_mx) * inv;
            if (!(qc > 0.0f)) continue;
            const float dc = fabsf(-logf(qc) - sh_H);
            float mass = 0.0f;
            for (uint32_t i = 0; i < n_cand; ++i) {
                if (oi[i] < 0) continue;
                const float q = expf(ov[i] - sh_mx) * inv;
                if (!(q > 0.0f)) continue;
                if (fabsf(-logf(q) - sh_H) <= dc) mass += q;
            }
            if (mass >= p.typical_p && dc < local_cut) { local_cut = dc; local_have = true; }
        }
        if (local_have) {
            atomicMin(&sh_cut_bits, __float_as_int(local_cut));
            atomicOr(&sh_have_cut, 1);
        }
        __syncthreads();
        if (!sh_have_cut) return;
        const float cut = __int_as_float(sh_cut_bits);
        for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
            if (oi[i] < 0) continue;
            const float q = expf(ov[i] - sh_mx) * inv;
            if (!(q > 0.0f) || fabsf(-logf(q) - sh_H) > cut) { oi[i] = -1; ov[i] = -INFINITY; }
        }
        return;
    }

    // XTC -- "exclude top choices".  With probability xtc_probability, DROP every candidate at or
    // above xtc_threshold EXCEPT the least likely of them.  The opposite of every other stage here
    // and that is the point: it removes the obvious continuations and leaves the interesting one.
    // The draw is per row from the row's own stream, so it reproduces.
    if (!(p.xtc_probability > 0.0f) || !(p.xtc_threshold > 0.0f)) return;
    if (uniform01(p.seed ^ RAD_SAMPLE_KEY_XTC, p.pos) >= p.xtc_probability) return;

    if (threadIdx.x == 0) { sh_last = -1; sh_above = 0; }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        if (oi[i] < 0) continue;
        if (expf(ov[i] - mx) * inv >= p.xtc_threshold) {
            atomicAdd(&sh_above, 1);
            atomicMax(&sh_last, (int)i);
        }
    }
    __syncthreads();
    // Nothing to do unless at least two clear the threshold: dropping the only one would leave the
    // row without its own choice.
    if (sh_above < 2) return;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        if (oi[i] < 0 || (int)i == sh_last) continue;
        if (expf(ov[i] - mx) * inv >= p.xtc_threshold) { oi[i] = -1; ov[i] = -INFINITY; }
    }
}

// ---------------------------------------------------------------------------- sample_pick
// ref_sample_pick: the inverse CDF over the candidate set, walked in candidate order with the sum
// accumulated in f32.  ONE THREAD for the same reason top_p is one thread -- the walk's rounding is
// the contract.  The fallback is the last candidate ABOVE -INFINITY: `u` is strictly below 1 and
// the probabilities sum to 1 in exact arithmetic but not in f32, and a draw that falls through the
// last comparison must still produce a token the distribution allows.  A live candidate at
// -infinity is a grammar-masked logit top-k took to fill its width, and falling through to it emits
// a token the grammar ruled out.  It adds nothing to `cum`, so the comparison never chooses one.
//
// A GREEDY ROW IS NOT TOUCHED: its token is the argmax's, written before the chain ran.
__global__ __launch_bounds__(kThreads) void pick_kernel(
    const int32_t* __restrict__ ci, const float* __restrict__ cv, const SP* __restrict__ sp,
    int32_t* __restrict__ tok, uint32_t n_cand, uint32_t ci_ld, uint32_t cv_ld) {
    extern __shared__ float lds_exp[];
    __shared__ float red[kWaves];
    __shared__ float sh_inv;

    const uint32_t m = blockIdx.x;
    if (sp[m].flags & SP_GREEDY) return;
    const int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    const float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    float mx = -INFINITY;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads)
        if (oi[i] >= 0) mx = fmaxf(mx, ov[i]);
    mx = block_max(mx, red);
    if (mx == -INFINITY) { if (threadIdx.x == 0) tok[m] = -1; return; }

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_cand; i += kThreads) {
        if (oi[i] >= 0) {
            const float e = expf(ov[i] - mx);
            lds_exp[i] = e;
            sum += e;
        } else {
            lds_exp[i] = 0.0f;
        }
    }
    sum = block_sum(sum, red);
    if (threadIdx.x == 0) sh_inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();
    if (threadIdx.x != 0) return;

    const float u = uniform01(sp[m].seed ^ RAD_SAMPLE_KEY_PICK, sp[m].pos);
    float cum = 0.0f;
    int32_t chosen = -1, last = -1;
    for (uint32_t i = 0; i < n_cand; ++i) {
        const int32_t id = oi[i];
        if (id < 0) continue;
        const float v = ov[i];
        if (v > -INFINITY) last = id;
        cum += lds_exp[i] * sh_inv;
        if (u < cum) { chosen = id; break; }
    }
    tok[m] = chosen >= 0 ? chosen : last;
}

// ---------------------------------------------------------------------------- sample_merge_topk
// THE VOCAB-PARALLEL MERGE.  Each rank runs top-k over its own lm_head shard and the candidate sets
// are all-gathered -- k * world_size pairs, kilobytes, against the tens of megabytes a full-logits
// gather would move.  `gathered` is (u32 global id, f32 logit) pairs: each rank adds its own vocab
// offset before the gather, so this needs no shard layout.
//
// One block a row, and the same packed (key, ~id) word as the top-k, so the descending order and
// the lower-id tie rule are the same code path.  n_in is world_size * n_cand, a few thousand.
__global__ __launch_bounds__(kTopkThreads) void merge_topk_kernel(
    const uint32_t* __restrict__ gt, const SP* __restrict__ sp,
    int32_t* __restrict__ ci, float* __restrict__ cv,
    uint32_t n_in, uint32_t n_cand, uint32_t gt_ld, uint32_t ci_ld, uint32_t cv_ld, uint32_t pad) {
    extern __shared__ unsigned long long shared_cand[];
    const uint32_t m = blockIdx.x;
    const uint32_t* __restrict__ g = gt + (size_t)m * gt_ld;
    int32_t* __restrict__ oi = ci + (size_t)m * ci_ld;
    float*   __restrict__ ov = cv + (size_t)m * cv_ld;

    // A greedy row has no candidates, as in topk_kernel; the whole block leaves together.
    if (sp[m].flags & SP_GREEDY) {
        for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) { oi[r] = -1; ov[r] = -INFINITY; }
        return;
    }

    /* THE INPUT IS ALREADY world_size SORTED RUNS, and the sort starts where they leave off. Each
     * rank's top-k wrote its n_cand pairs in descending (key, ~id) order with its holes at the
     * tail, and a rank's vocab offset is one constant added to every id, so the run is in THIS
     * kernel's key order too. When the runs are a power of two long and fill the padded width,
     * that is exactly the state the bitonic network is in after its stages up to sz = n_cand --
     * provided the odd runs are loaded reversed, since those stages leave them ascending -- and
     * only the log2(world_size) merge stages are left: 11 passes over two runs of 1024 where the
     * whole network is 66. Any other shape starts from the first stage. The keys are distinct but
     * for the holes, which are all zero, so the order that comes out is the only one there is. */
    const bool     runs  = (n_cand & (n_cand - 1)) == 0 && pad == n_in;
    const uint32_t first = runs ? 2 * n_cand : 2;
    for (uint32_t j = threadIdx.x; j < pad; j += kTopkThreads) {
        if (j >= n_in) { shared_cand[j] = 0ull; continue; }
        const uint32_t run = runs ? j / n_cand : 0;
        const uint32_t src = (run & 1u) ? run * n_cand + (n_cand - 1 - (j - run * n_cand)) : j;
        const uint32_t id = g[2 * src];
        const float    v  = __builtin_bit_cast(float, g[2 * src + 1]);
        shared_cand[j] = (int32_t)id < 0
                             ? 0ull
                             : (((unsigned long long)fkey(v) << 32) | (unsigned long long)(~id));
    }
    __syncthreads();

    const bool is_tp2_runs = runs && (n_in == 2 * n_cand) && (pad == n_in);
    for (uint32_t sz = first; sz <= pad; sz <<= 1) {
        for (uint32_t st = sz >> 1; st > 0; st >>= 1) {
            const uint32_t limit = (is_tp2_runs && st < n_cand) ? n_cand : pad;
            for (uint32_t t = threadIdx.x; t < limit; t += kTopkThreads) {
                const uint32_t partner = t ^ st;
                if (partner <= t) continue;
                const bool up = (t & sz) == 0;
                const unsigned long long a = shared_cand[t], b = shared_cand[partner];
                if (up ? (a < b) : (a > b)) { shared_cand[t] = b; shared_cand[partner] = a; }
            }
            __syncthreads();
        }
    }

    // THE ROW'S OWN top_k, RE-APPLIED. Each rank ran its own top-k before the gather, so the
    // merged set holds up to world_size * k candidates where the request asked for k -- and
    // nothing downstream narrows again. Leaving them live is not a rounding error in the
    // sampling support, it is a different distribution, and at k = 1 it is a different ANSWER:
    // a greedy row is staged at top_k = 1 precisely so the chain returns the argmax, and with
    // world_size candidates alive `sample_pick` draws among them against the request's seed
    // instead. On two ranks that makes the same greedy prompt return either the true argmax or the
    // OTHER rank's local best, depending on nothing but an unset seed, about a quarter of the time.
    // 0 or negative means the whole candidate width, as everywhere else.
    const uint32_t keep = (uint32_t)(sp[m].top_k > 0 ? sp[m].top_k : (int32_t)n_cand);
    for (uint32_t r = threadIdx.x; r < n_cand; r += kTopkThreads) {
        const unsigned long long e = (r < pad && r < keep) ? shared_cand[r] : 0ull;
        if (e == 0ull) { oi[r] = -1; ov[r] = -INFINITY; continue; }
        const uint32_t key = (uint32_t)(e >> 32);
        oi[r] = (int32_t)(~(uint32_t)(e & 0xffffffffull));
        ov[r] = __builtin_bit_cast(float, (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
    }
}

}  // namespace

// ================================================================================ entry points
//
// Every pitch is in ELEMENTS and 0 means tight.  `lg` is f32 -- the sampler runs in f32 whatever
// the model's activation dtype is, because a bf16 logit has 8 bits of mantissa and a nucleus cut is
// a comparison of neighbouring probabilities.

static uint32_t host_pow2(uint32_t v) { uint32_t p = 1; while (p < v) p <<= 1; return p; }

static int r4d_sample_max_cand(void) { return (int)kMaxCand; }

static int r4d_sample_temp_f32(long lg, long params, int M, int n_vocab, int lg_ld,
                                   long stream) {
    if (!lg || !params) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    const uint32_t gy = ((uint32_t)n_vocab + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(temp_kernel, dim3((uint32_t)M, gy > 256u ? 256u : gy), dim3(kThreads), 0,
                       (hipStream_t)stream, (float*)lg, (const SP*)params,
                       (uint32_t)n_vocab, (uint32_t)lg_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int r4d_sample_mask_f32(long lg, long params, long bitmask, int M, int n_vocab,
                                   int mask_rows, int lg_ld, long stream) {
    if (!lg || !params || !bitmask) return -1;
    if (M <= 0 || n_vocab <= 0 || mask_rows <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    const uint32_t words = ((uint32_t)n_vocab + 31u) / 32u;
    const uint32_t gy = ((uint32_t)n_vocab + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(mask_kernel, dim3((uint32_t)M, gy > 256u ? 256u : gy), dim3(kThreads), 0,
                       (hipStream_t)stream, (float*)lg, (const SP*)params,
                       (const uint32_t*)bitmask, (uint32_t)n_vocab, (uint32_t)lg_ld, words,
                       (int32_t)mask_rows);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int r4d_sample_penalties_f32(long lg, long params, long hist, int M, int n_vocab,
                                        int hmax, int lg_ld, int vocab_off, long stream) {
    if (!lg || !params || !hist) return -1;
    if (M <= 0 || n_vocab <= 0 || hmax <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    hipLaunchKernelGGL(penalties_kernel, dim3((uint32_t)M), dim3(kThreads), 0, (hipStream_t)stream,
                       (float*)lg, (const SP*)params, (const int32_t*)hist,
                       (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hmax, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// The widest history row DRY takes: one u16 repeat length a position lives in LDS, and 16384 of
// them is 32 KiB, half of what a workgroup may have.  The engine declares 2048.
constexpr uint32_t kDryMaxHist = 16384;

static int r4d_sample_dry_f32(long lg, long params, long hist, long brk, int M, int n_vocab,
                                  int hmax, int lg_ld, int vocab_off, long stream) {
    if (!lg || !params || !hist) return -1;
    if (M <= 0 || n_vocab <= 0 || hmax <= 0) return -2;
    if ((uint32_t)hmax > kDryMaxHist) return -3;
    if (!lg_ld) lg_ld = n_vocab;
    hipLaunchKernelGGL(dry_kernel, dim3((uint32_t)M), dim3(kThreads),
                       (size_t)hmax * sizeof(uint16_t), (hipStream_t)stream,
                       (float*)lg, (const SP*)params, (const int32_t*)hist, (const uint8_t*)brk,
                       (uint32_t)n_vocab, (uint32_t)lg_ld, (uint32_t)hmax, (int32_t)vocab_off);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int r4d_sample_argmax_f32(long lg, long tok, int M, int n_vocab, int lg_ld,
                                     long stream) {
    if (!lg || !tok) return -1;
    if (M <= 0 || n_vocab <= 0) return -2;
    if (!lg_ld) lg_ld = n_vocab;
    /* THE PITCH IS PART OF THE ALIGNMENT, not just the base: row m starts at `m * lg_ld` floats, so
     * a pitch that is not a multiple of four misaligns every row after the first. */
    const bool vec = ((uintptr_t)lg % (4 * sizeof(float)) == 0) && ((uint32_t)lg_ld % 4u == 0u);
    if (vec)
        hipLaunchKernelGGL((argmax_kernel<kArgmaxThreads, true>), dim3((uint32_t)M),
                           dim3(kArgmaxThreads), 0, (hipStream_t)stream,
                           (const float*)lg, (int32_t*)tok, (uint32_t)n_vocab, (uint32_t)lg_ld);
    else
        hipLaunchKernelGGL((argmax_kernel<kArgmaxThreads, false>), dim3((uint32_t)M),
                           dim3(kArgmaxThreads), 0, (hipStream_t)stream,
                           (const float*)lg, (int32_t*)tok, (uint32_t)n_vocab, (uint32_t)lg_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int r4d_sample_topk_f32(long lg, long params, long cand_idx, long cand_val, long pairs,
                                   int M, int n_vocab, int n_cand,
                                   int lg_ld, int ci_ld, int cv_ld, int pr_ld, int vocab_off,
                                   long stream) {
    if (!lg || !params || !cand_idx || !cand_val) return -1;
    if (M <= 0 || n_vocab <= 0 || n_cand <= 0) return -2;
    if ((uint32_t)n_cand > kMaxCand) return -3;
    if (!lg_ld) lg_ld = n_vocab;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    if (!pr_ld) pr_ld = 2 * n_cand;
    const uint32_t pad = host_pow2((uint32_t)n_cand);
    if (pairs) {
        hipLaunchKernelGGL((topk_kernel<true>), dim3((uint32_t)M), dim3(kTopkThreads),
                           pad * sizeof(unsigned long long), (hipStream_t)stream,
                           (const float*)lg, (const SP*)params, (int32_t*)cand_idx, (float*)cand_val,
                           (uint32_t*)pairs,
                           (uint32_t)n_vocab, (uint32_t)n_cand, (uint32_t)lg_ld, (uint32_t)ci_ld,
                           (uint32_t)cv_ld, (uint32_t)pr_ld, (int32_t)vocab_off, pad);
    } else {
        hipLaunchKernelGGL((topk_kernel<false>), dim3((uint32_t)M), dim3(kTopkThreads),
                           pad * sizeof(unsigned long long), (hipStream_t)stream,
                           (const float*)lg, (const SP*)params, (int32_t*)cand_idx, (float*)cand_val,
                           (uint32_t*)pairs,
                           (uint32_t)n_vocab, (uint32_t)n_cand, (uint32_t)lg_ld, (uint32_t)ci_ld,
                           (uint32_t)cv_ld, (uint32_t)pr_ld, (int32_t)vocab_off, pad);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// `which` is 0 top_p, 1 min_p, 2 typical, 3 xtc -- one entry point because the four differ by a
// branch after the same two reductions, and four rows in the table because the SELECTOR names ops,
// not entry points.
static int r4d_sample_narrow_f32(long cand_idx, long cand_val, long params, int M, int n_cand,
                                     int which, int ci_ld, int cv_ld, long stream) {
    if (!cand_idx || !cand_val || !params) return -1;
    if (M <= 0 || n_cand <= 0) return -2;
    if (which < 0 || which > 3) return -3;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    const size_t narrow_lds = (which == NR_TOPP) ? (size_t)n_cand * sizeof(float) : 0;
    hipLaunchKernelGGL(narrow_kernel, dim3((uint32_t)M), dim3(kThreads), narrow_lds, (hipStream_t)stream,
                       (int32_t*)cand_idx, (float*)cand_val, (const SP*)params,
                       (uint32_t)n_cand, (uint32_t)ci_ld, (uint32_t)cv_ld, which);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int r4d_sample_pick_f32(long cand_idx, long cand_val, long params, long tok,
                                   int M, int n_cand, int ci_ld, int cv_ld, long stream) {
    if (!cand_idx || !cand_val || !params || !tok) return -1;
    if (M <= 0 || n_cand <= 0) return -2;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    const size_t pick_lds = (size_t)n_cand * sizeof(float);
    hipLaunchKernelGGL(pick_kernel, dim3((uint32_t)M), dim3(kThreads), pick_lds, (hipStream_t)stream,
                       (const int32_t*)cand_idx, (const float*)cand_val, (const SP*)params,
                       (int32_t*)tok, (uint32_t)n_cand, (uint32_t)ci_ld, (uint32_t)cv_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int r4d_sample_merge_topk_f32(long gathered, long params, long cand_idx, long cand_val,
                                         int M, int n_in, int n_cand,
                                         int gt_ld, int ci_ld, int cv_ld, long stream) {
    if (!gathered || !params || !cand_idx || !cand_val) return -1;
    if (M <= 0 || n_in <= 0 || n_cand <= 0) return -2;
    if ((uint32_t)n_in > kMaxCand) return -3;
    if (!gt_ld) gt_ld = 2 * n_in;
    if (!ci_ld) ci_ld = n_cand;
    if (!cv_ld) cv_ld = n_cand;
    const uint32_t pad = host_pow2((uint32_t)n_in);
    hipLaunchKernelGGL(merge_topk_kernel, dim3((uint32_t)M), dim3(kTopkThreads),
                       pad * sizeof(unsigned long long), (hipStream_t)stream,
                       (const uint32_t*)gathered, (const SP*)params,
                       (int32_t*)cand_idx, (float*)cand_val,
                       (uint32_t)n_in, (uint32_t)n_cand, (uint32_t)gt_ld, (uint32_t)ci_ld,
                       (uint32_t)cv_ld, pad);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

// ============================================================== the plugin ABI entry points
//
// Unpacking RadArgs and re-deriving the pitches IS the kernel's own business -- which operand is
// the candidate index plane, whether a row pitch is tight -- so it lives beside the kernel rather
// than in a translation layer an out-of-tree author cannot write. The flattened entry points above
// are private to this unit.
//
// WHY THE CHAIN IS ELEVEN OPS AND NOT r4d_sample_chain_f32's ONE. The fused kernel does
// temperature, softmax, top_k, top_p, min_p and the draw in vLLM's fixed order; the core
// composes its chain per request out of stages a request may skip, and it has stages the fused
// kernel does not -- the penalties, DRY, a grammar bitmask, locally-typical, XTC. "Fusion is a
// selection, not a compiler pass" (spec 2.3) cuts both ways: a library that shipped only the
// fused form would be choosing the fusion on the plugin's behalf. Both rows exist; this is the
// one the served path resolves.
//
// EVERY SCALAR IS PER ROW. `top_p` as a parameter would make two requests with different top_p
// two different resolutions of one op, so a deployment would have exactly one top_p
// (rad_sample.h says this at length). The ops take `M` and `n_vocab` / `n_cand` as parameters
// and everything else as `params` data.

#define R4D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R4D_NO(_s); } while (0)

static inline int dt_f32(const RadTensor* t) {
    return (t && t->dtype == RAD_F32) ? RAD_OK : RAD_E_DTYPE;
}
static inline int dt_i32(const RadTensor* t) {
    return (t && (t->dtype == RAD_I32 || t->dtype == RAD_U32)) ? RAD_OK : RAD_E_DTYPE;
}

/* `params` is a BYTE BLOB viewed through a 32-bit dtype: the builder can only say "this many
 * words a row", so the extent is checked in words and the dtype is required to be a 32-bit one --
 * rad_dtype_bytes is core's and a kernel plugin links only the ABI. */
static int params_ok(const RadTensor* t, long long M) {
    if (!t || !t->data) return R4D_NO(RAD_E_INVAL);
    if (t->dtype != RAD_U32 && t->dtype != RAD_I32 && t->dtype != RAD_F32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(t)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(t) * 4 < M * (long long)sizeof(RadSampleParams)) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

/* [rows, n] with a row pitch. The row count comes from the OPERAND: `M` is ranged, so the value in
 * the args is the band's upper bound and a loop bounded by it walks off the arena. */
static long long rows_pitch(const RadTensor* t, long long n, long long* rows) {
    if (!t || t->rank < 1 || n <= 0) return -1;
    const long long nel = r4d_numel(t);
    if (nel % n) return -1;
    *rows = nel / n;
    if (t->rank == 1) return n;
    if (t->shape[t->rank - 1] != n) return r4d_contig(t) ? n : -1;
    if (t->stride[t->rank - 1] != 1) return -1;
    return t->rank >= 2 ? t->stride[t->rank - 2] : n;
}

#define R4D_LAUNCH(call) return ((call) == 0 ? RAD_OK : RAD_E_DEVICE)

/* ------------------------------------------------------------------ the vocabulary-wide stages */

enum { V_LOGITS = 0, V_PARAMS, V_AUX };

/* logits [M, n_vocab] f32, and the M that matters. */
static int logits_of(const RadArgs* a, const RadTensor** lg, long long* M, long long* n_vocab,
                     long long* ld) {
    *lg = r4d_opt(a, V_LOGITS);
    if (!*lg) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_f32(*lg));
    *n_vocab = r4d_geti_or(a, "n_vocab", r4d_dim(*lg, (int)(*lg)->rank - 1));
    if (*n_vocab <= 0) return R4D_NO(RAD_E_SHAPE);
    *ld = rows_pitch(*lg, *n_vocab, M);
    if (*ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (*M <= 0) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int r4d_rad_sample_temp(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    R4D_NEED(params_ok(pr, M));
    R4D_LAUNCH(r4d_sample_temp_f32(r4d_addr(lg), r4d_addr(pr), (int)M, (int)nv, (int)ld,
                                   (long)(uintptr_t)s));
}

extern "C" int r4d_rad_sample_mask(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    const RadTensor* bm = r4d_opt(a, V_AUX);
    R4D_NEED(params_ok(pr, M));
    if (!bm) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(bm));
    if (!r4d_contig(bm)) return R4D_NO(RAD_E_STRIDE);
    /* The plane is [rows, ceil(n_vocab/32)] and is SHORTER than M: a constrained sequence and an
     * unconstrained one share it, and `mask_row` is -1 for the second. */
    const long long words = (nv + 31) / 32;
    const long long rows = r4d_numel(bm) / (words > 0 ? words : 1);
    if (rows <= 0) return R4D_NO(RAD_E_SHAPE);
    R4D_LAUNCH(r4d_sample_mask_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(bm), (int)M, (int)nv,
                                   (int)rows, (int)ld, (long)(uintptr_t)s));
}

/* `history` is [M, max_hist] int32 of GLOBAL token ids, row m's window addressed by row m's
 * hist_off / hist_len within row m. `vocab_off` is this rank's first vocabulary row, 0 when the
 * vocabulary is whole. DRY's optional fourth operand, `breakers`, is a u8 plane of the same shape:
 * nonzero where that position's token is a whole sequence breaker. */
static int penal(const RadArgs* a, int dry, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, V_PARAMS);
    const RadTensor* h  = r4d_opt(a, V_AUX);
    R4D_NEED(params_ok(pr, M));
    if (!h) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(h));
    if (!r4d_contig(h)) return R4D_NO(RAD_E_STRIDE);
    const long long hmax = r4d_numel(h) / M;
    if (hmax <= 0) return R4D_NO(RAD_E_SHAPE);
    const long long voff = r4d_pgeti_or(a->p, a->n_p, "vocab_off", 0);
    if (dry) {
        const RadTensor* b = r4d_opt(a, V_AUX + 1);
        if (b) {
            if (b->dtype != RAD_U8 && b->dtype != RAD_I8 && b->dtype != RAD_BOOL)
                return R4D_NO(RAD_E_DTYPE);
            if (!r4d_contig(b)) return R4D_NO(RAD_E_STRIDE);
            if (r4d_numel(b) < M * hmax) return R4D_NO(RAD_E_SHAPE);
        }
        if (hmax > (long long)kDryMaxHist) return R4D_NO(RAD_E_SHAPE);
        R4D_LAUNCH(r4d_sample_dry_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(h),
                                      b ? r4d_addr(b) : 0, (int)M, (int)nv, (int)hmax, (int)ld,
                                      (int)voff, (long)(uintptr_t)s));
    }
    R4D_LAUNCH(r4d_sample_penalties_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(h), (int)M, (int)nv,
                                        (int)hmax, (int)ld, (int)voff, (long)(uintptr_t)s));
}

extern "C" int r4d_rad_sample_penalties(const RadArgs* a, RadStream s) {
    return penal(a, 0, s);
}
extern "C" int r4d_rad_sample_dry(const RadArgs* a, RadStream s) { return penal(a, 1, s); }

extern "C" int r4d_rad_sample_argmax(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* tok = r4d_opt(a, 2);
    if (!tok) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(tok));
    if (!r4d_contig(tok) || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);
    R4D_LAUNCH(r4d_sample_argmax_f32(r4d_addr(lg), r4d_addr(tok), (int)M, (int)nv, (int)ld,
                                     (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ top-k */

enum { TK_LOGITS = 0, TK_PARAMS, TK_CI, TK_CV, TK_PAIRS };

extern "C" int r4d_rad_sample_topk(const RadArgs* a, RadStream s) {
    const RadTensor* lg;
    long long M = 0, nv = 0, ld = 0;
    R4D_NEED(logits_of(a, &lg, &M, &nv, &ld));
    const RadTensor* pr = r4d_opt(a, TK_PARAMS);
    const RadTensor* ci = r4d_opt(a, TK_CI);
    const RadTensor* cv = r4d_opt(a, TK_CV);
    R4D_NEED(params_ok(pr, M));
    if (!ci || !cv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));

    /* The candidate WIDTH is the buffer's and not the request's: it is the deployment bound the
     * arena was sized for, and every later stage reads it. A row's own top_k may be narrower and
     * leaves holes, which is the kernel's business. */
    const long long n_cand = ci->rank >= 2 ? ci->shape[ci->rank - 1] : r4d_numel(ci) / M;
    if (n_cand <= 0 || n_cand > r4d_sample_max_cand()) return R4D_NO(RAD_E_SHAPE);
    long long Mi = 0, Mv = 0;
    const long long ci_ld = rows_pitch(ci, n_cand, &Mi);
    const long long cv_ld = rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (Mi < M || Mv < M) return R4D_NO(RAD_E_SHAPE);

    /* THE VOCAB-PARALLEL PAIR PLANE, optional. Present only when lm_head is row-sharded: the
     * candidates this rank found, as (u32 global id, f32 logit) pairs, which is what the
     * all-gather moves and what sample_merge_topk reads. `vocab_off` is this rank's first
     * vocabulary row and the kernel adds it, so the gather never has to know the shard layout. */
    const RadTensor* pp = r4d_opt(a, TK_PAIRS);
    long long pr_ld = 0, Mp = 0;
    if (pp) {
        R4D_NEED(dt_i32(pp));
        pr_ld = rows_pitch(pp, 2 * n_cand, &Mp);
        if (pr_ld < 0) return R4D_NO(RAD_E_STRIDE);
        if (Mp < M) return R4D_NO(RAD_E_SHAPE);
    }
    const long long voff = r4d_pgeti_or(a->p, a->n_p, "vocab_off", 0);

    R4D_LAUNCH(r4d_sample_topk_f32(r4d_addr(lg), r4d_addr(pr), r4d_addr(ci), r4d_addr(cv),
                                   pp ? r4d_addr(pp) : 0,
                                   (int)M, (int)nv, (int)n_cand, (int)ld, (int)ci_ld, (int)cv_ld,
                                   (int)pr_ld, (int)voff, (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ the narrowing stages */

enum { NR_CI = 0, NR_CV, NR_PARAMS };

static int narrow(const RadArgs* a, int which, RadStream s) {
    const RadTensor* ci = r4d_opt(a, NR_CI);
    const RadTensor* cv = r4d_opt(a, NR_CV);
    const RadTensor* pr = r4d_opt(a, NR_PARAMS);
    if (!ci || !cv) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));

    const long long n_cand = r4d_geti_or(a, "n_cand", r4d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long ci_ld = rows_pitch(ci, n_cand, &M);
    const long long cv_ld = rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mv < M) return R4D_NO(RAD_E_SHAPE);
    R4D_NEED(params_ok(pr, M));

    R4D_LAUNCH(r4d_sample_narrow_f32(r4d_addr(ci), r4d_addr(cv), r4d_addr(pr), (int)M,
                                     (int)n_cand, which, (int)ci_ld, (int)cv_ld,
                                     (long)(uintptr_t)s));
}

extern "C" int r4d_rad_sample_topp(const RadArgs* a, RadStream s)    { return narrow(a, 0, s); }
extern "C" int r4d_rad_sample_minp(const RadArgs* a, RadStream s)    { return narrow(a, 1, s); }
extern "C" int r4d_rad_sample_typical(const RadArgs* a, RadStream s) { return narrow(a, 2, s); }
extern "C" int r4d_rad_sample_xtc(const RadArgs* a, RadStream s)     { return narrow(a, 3, s); }

/* ------------------------------------------------------------------ pick */

enum { PK_CI = 0, PK_CV, PK_PARAMS, PK_TOK };

extern "C" int r4d_rad_sample_pick(const RadArgs* a, RadStream s) {
    const RadTensor* ci  = r4d_opt(a, PK_CI);
    const RadTensor* cv  = r4d_opt(a, PK_CV);
    const RadTensor* pr  = r4d_opt(a, PK_PARAMS);
    const RadTensor* tok = r4d_opt(a, PK_TOK);
    if (!ci || !cv || !tok) return R4D_NO(RAD_E_INVAL);
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));
    R4D_NEED(dt_i32(tok));
    if (!r4d_contig(tok)) return R4D_NO(RAD_E_STRIDE);

    const long long n_cand = r4d_geti_or(a, "n_cand", r4d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long ci_ld = rows_pitch(ci, n_cand, &M);
    const long long cv_ld = rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mv < M || r4d_numel(tok) < M) return R4D_NO(RAD_E_SHAPE);
    R4D_NEED(params_ok(pr, M));

    R4D_LAUNCH(r4d_sample_pick_f32(r4d_addr(ci), r4d_addr(cv), r4d_addr(pr), r4d_addr(tok),
                                   (int)M, (int)n_cand, (int)ci_ld, (int)cv_ld,
                                   (long)(uintptr_t)s));
}

/* ------------------------------------------------------------------ merge_topk */

enum { MG_GATHERED = 0, MG_PARAMS, MG_CI, MG_CV };

extern "C" int r4d_rad_sample_merge_topk(const RadArgs* a, RadStream s) {
    const RadTensor* gt = r4d_opt(a, MG_GATHERED);
    const RadTensor* pr = r4d_opt(a, MG_PARAMS);
    const RadTensor* ci = r4d_opt(a, MG_CI);
    const RadTensor* cv = r4d_opt(a, MG_CV);
    /* `params` carries the row's top_k, and the merge is the one place that can re-apply it:
     * each rank narrowed to k before the gather, so the merged set holds world_size * k
     * candidates and no later stage narrows again. At k = 1 -- which is what
     * a greedy row is staged at, precisely so the chain returns the argmax -- leaving two alive
     * makes `sample_pick` draw between them against the request's seed. */
    if (!gt || !pr || !ci || !cv) return R4D_NO(RAD_E_INVAL);
    /* Each gathered entry is a (u32 id, f32 logit) PAIR IN ONE BUFFER, so the two halves have
     * different types and the buffer's dtype can only be one of them. It is a word type and the
     * logit is read by BIT PATTERN. */
    R4D_NEED(dt_i32(gt));
    R4D_NEED(dt_i32(ci));
    R4D_NEED(dt_f32(cv));
    if (!r4d_contig(gt)) return R4D_NO(RAD_E_STRIDE);

    const long long n_cand = r4d_geti_or(a, "n_cand", r4d_dim(ci, (int)ci->rank - 1));
    if (n_cand <= 0) return R4D_NO(RAD_E_SHAPE);
    long long M = 0, Mv = 0;
    const long long ci_ld = rows_pitch(ci, n_cand, &M);
    const long long cv_ld = rows_pitch(cv, n_cand, &Mv);
    if (ci_ld < 0 || cv_ld < 0) return R4D_NO(RAD_E_STRIDE);
    if (M <= 0 || Mv < M) return R4D_NO(RAD_E_SHAPE);

    const long long per_row = r4d_numel(gt) / M;
    const long long n_in = per_row / 2;
    if (n_in <= 0 || n_in > r4d_sample_max_cand()) return R4D_NO(RAD_E_SHAPE);

    R4D_LAUNCH(r4d_sample_merge_topk_f32(r4d_addr(gt), r4d_addr(pr), r4d_addr(ci), r4d_addr(cv),
                                         (int)M, (int)n_in, (int)n_cand, (int)per_row, (int)ci_ld,
                                         (int)cv_ld, (long)(uintptr_t)s));
}
