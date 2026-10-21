// The QSA indexer's SELECTION: the `topk` highest-scoring blocks of a query's key history, as a
// page-id table the paged attention can read directly.
//
// This is the last step of the indexer and the one that makes the attention constant-cost: below
// it the scorer has produced one number per complete 4-token block of the sequence, and above it
// the attention reads at most `topk + 1` pages however long the context is. docs/QSA.md is the
// decomposition.
//
// THE OUTPUT IS A PAGE-ID TABLE AND NOT A LIST OF RANKS, which is the whole reason the KV group
// is paged at the compress ratio: a selected block IS a page, so what comes out of here is
// literally what the attention's block table wants. Nothing is remapped.
//
// THE PARTIAL PAGE IS APPENDED LAST AND IS NOT A CANDIDATE. The reference attends the
// `(pos+1) % ratio` tokens after the last complete block UNCONDITIONALLY -- they are the newest
// tokens and the query is among them -- so they are not scored and cannot be dropped. Last is
// load-bearing and not cosmetic: every selected COMPLETE page is entirely in the query's past, so
// the only causal masking left is inside the final page, and putting it at the end of the table
// makes the attention's own `kk <= q` bound do exactly that with no change to the kernel. The
// effective query position the caller must then pass is `(nsel - 1) * ratio + pos % ratio`.
//
// SELECTION IS A FOUR-DIGIT RADIX SELECT, not a sort and not a threshold search. A sort of 65536
// scores a query a layer a step is not affordable and is not needed -- attention is a softmax over
// a SET, so the order within the table does not matter. What does matter is that the set is
// EXACT and DETERMINISTIC: ties break to the lower block index through a prefix count, never
// through the order atomics happen to complete in, because this tree compares generated text
// across builds byte for byte.
//
// Returns a negative code rather than throwing, for the reason given in r4d_gemm_fp8a8.hip.

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_emit.h"
#include "r4d_common.h"   // the DPP wave scans

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int kThreads = 256;
constexpr int kWave    = 32;
constexpr int kWaves   = kThreads / kWave;
constexpr int kBins    = 256;
/* ONE THREAD A BIN, which is what lets the threshold scan below be parallel at all. */
static_assert(kThreads == kBins, "the bin scan gives one bin to one thread");

/* THE KEY CACHE, AND WHY IT IS GATED ON THE LAUNCH WIDTH RATHER THAN ALWAYS ON.
 *
 * Five passes read every key -- four radix digits and the emit -- and on the same workgroup, so
 * at decode this kernel is one workgroup on the whole device making the same 5277 dependent
 * loads five times. Staging the keys in LDS once makes four of those five passes an LDS read,
 * worth about 25% of the kernel at n = 5277, topk = 512.
 *
 * IT IS A LOSS AT PREFILL AND THE REASON IS OCCUPANCY, NOT THE COPY. 32 KiB a workgroup is half
 * the per-workgroup LDS budget, so a prefill chunk's 2048 workgroups stop fitting side by side:
 * at M=2048 the cache costs about 7%, and M=128 is already the wrong
 * side. The crossover sits between 96 and 128 workgroups on a 64-CU part -- about where a
 * workgroup stops having a CU to itself -- so the gate is the launch width and 64 leaves margin.
 * Decode is M = n_seq and prefill is M = the chunk; nothing real lands in between, and a MIXED
 * step is wide and correctly takes the uncached path. */
constexpr int kLdsCap    = 8192;   /* keys, so 32 KiB of dynamic LDS */
constexpr int kCacheMaxM = 64;     /* launch width above which the cache costs more than it saves */

/* Clamped to what the device will actually grant a workgroup, minus the static shared above. A
 * part with a smaller LDS gets a smaller cache and deeper queries simply take the global path. */
static inline int lds_cap_of() {
    static const int v = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return 0;
        hipDeviceProp_t pr{};
        if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return 0;
        const size_t stat = 3072;                      /* hist + wsum + the scalars */
        if (pr.sharedMemPerBlock <= stat) return 0;
        const size_t room = (pr.sharedMemPerBlock - stat) / sizeof(uint32_t);
        return (int)(room < (size_t)kLdsCap ? room : (size_t)kLdsCap);
    }();
    return v;
}

/* The gate is the launch width, for the occupancy reason in the note above. */
static inline int cache_max_m() { return kCacheMaxM; }

/* THE ORDER KEY. A score is a sum of relus over a positive constant, so it is >= 0 or -inf, and
 * for non-negative floats the IEEE bit pattern is monotonic as an unsigned. `+1` keeps a live
 * score of exactly zero -- every head vetoed the block -- above the 0 that means "no block here",
 * which is the distinction the scorer's -inf padding exists to make. */
__device__ __forceinline__ uint32_t keyof(float v, int page) {
    if (page < 0 || !(v > -1e30f)) return 0u;
    return __float_as_uint(fmaxf(v, 0.0f)) + 1u;
}

/* THE SCANS RUN IN REGISTERS, ON DPP, rather than as LDS round trips between barriers.
 *
 * In LDS the bin suffix scan is eight step-doubling rounds with two barriers each, and the
 * per-wave count prefix a serial walk over up to 32 LDS words, both once per radix digit or once
 * per 256 candidates, on workgroups of up to 1024 threads, where every barrier waits for
 * thirty-two waves. The merge is one workgroup a query, so those barriers and walks would be most
 * of what it does. A wave's 32 values scan in five DPP steps -- row_shr or row_shl within each row
 * of sixteen, then one readlane across the two rows (r4d_common.h) -- and only the waves' totals
 * need LDS at all. The sums are of integers, so either form gives the same answer. */
/* radix_step selects the digit without materialising the suffix plane in LDS.
 * Each wave suffix-scans its own 32 bins in registers, then publishes its total in wt.
 * Thread t computes its complete suffix sum s and after = s - hist[t].
 * When want > 0 and s >= want && after < want, that unique bin publishes the digit and new need.
 * If s < want (or want == 0), thread 0 publishes the fallback.
 * One barrier publishes the wave sums; one barrier publishes sh_T and sh_need. */
template <int NT>
__device__ __forceinline__ void radix_step(const uint32_t* hist, uint32_t* wt,
                                           uint32_t shift, uint32_t fixed, uint32_t want,
                                           uint32_t* sh_T, uint32_t* sh_need) {
    static_assert(NT >= kBins && kBins % kWave == 0, "the bins are whole waves of the workgroup");
    const int t = (int)threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;
    uint32_t h = 0, x = 0;
    if (t < kBins) {
        h = hist[t];
        x = r4d_wave_suffix_u32(h, lane);
        if (lane == 0) wt[wave] = x;
    }
    __syncthreads();
    if (t < kBins) {
        uint32_t above = 0;
        for (int w = wave + 1; w < kBins / kWave; ++w) above += wt[w];
        const uint32_t s = x + above;
        const uint32_t after = s - h;
        if (want > 0 && s >= want && after < want) {
            *sh_T = fixed | ((uint32_t)t << shift);
            *sh_need = want - after;
        }
        if (t == 0 && (s < want || want == 0)) {
            *sh_T = fixed;
            *sh_need = want - s;
        }
    }
    __syncthreads();
}

/* How many of the workgroup's NW wave counts come before this wave, and all of them together, from
 * `wsum` after the barrier that published them: every wave scans the NW words in its own lanes. */
template <int NW>
__device__ __forceinline__ void counts_before(const uint32_t* wsum, int lane, int wave,
                                              uint32_t* before, uint32_t* total) {
    static_assert(NW <= kWave, "a wave count a lane");
    const uint32_t c   = lane < NW ? wsum[lane] : 0u;
    const uint32_t inc = r4d_wave_prefix_u32(c, lane);
    const int      w   = __builtin_amdgcn_readfirstlane(wave);
    *before = (uint32_t)__builtin_amdgcn_readlane((int)(inc - c), w);
    *total  = (uint32_t)__builtin_amdgcn_readlane((int)inc, NW - 1);
}


/* `m` IS A QUERY AND NOT A SEQUENCE -- see the same note in r4d_qsa_score_bf16.hip. The block
 * table and the complete-block count belong to the SEQUENCE, `cu` says which one, and the
 * per-query bound is `(pos + 1) / ratio`: block `b` is entirely in this query's past exactly when
 * `b * ratio + ratio - 1 <= pos`. At decode the two agree by construction -- a sequence's complete
 * count IS its single query's -- so the per-query bound changes nothing there, to the bit. Any
 * sparse-against-dense comparison below the exactness bound therefore tests the SELECTION and not
 * the arithmetic it sits on. */
__global__ __launch_bounds__(kThreads) void qsa_select_kernel(
    const float* __restrict__ score, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    int32_t* __restrict__ sel, int32_t* __restrict__ nsel, int32_t* __restrict__ seqused,
    int nb, int topk, int ratio, int nseq, int s_ld, int t_ld, int o_ld, int cap,
    int m_offset) {

    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t wsum[kWaves];
    __shared__ uint32_t sh_T, sh_need, sh_out;
    __shared__ int      sh_seq;
    extern __shared__ uint32_t sh_key[];   /* `cap` keys, or nothing when the gate said no */

    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) sh_seq = -1;
    __syncthreads();
    const int gm = m + m_offset;
    for (int i = t; i < nseq; i += kThreads)
        if (gm >= cu[i] && gm < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    if (seq < 0) {
        if (t == 0) { nsel[m] = 0; seqused[m] = 0; }
        return;
    }
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;            /* blocks entirely in THIS query's past */
    int n = nc[seq] < nb ? nc[seq] : nb;        /* complete blocks; the rest is at most a partial */
    if (own < n) n = own;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt    + (size_t)seq * t_ld;
    int32_t*       ou = sel   + (size_t)m * o_ld;

    /* ---- the key cache. One global pass with nothing dependent in it, in place of four. */
    const bool cached = n <= cap;
    if (cached) {
        for (int b0 = 0; b0 < n; b0 += kThreads) {
            const int b = b0 + t;
            if (b < n) sh_key[b] = keyof(sc[b], tb[b]);
        }
        __syncthreads();
    }

    /* ---- the threshold. `T` is the exact key of the topk-th largest and `need` how many entries
     * hold it, which together define the set without ever ordering it. */
    uint32_t T = 0, need = 0;
    if (n > topk) {
        uint32_t fixed = 0, want = (uint32_t)topk;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = t; i < kBins; i += kThreads) hist[i] = 0;
            __syncthreads();
            const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
            for (int b = t; b < n; b += kThreads) {
                const uint32_t k = cached ? sh_key[b] : keyof(sc[b], tb[b]);
                if (k != 0u && (k & mask) == (fixed & mask))
                    atomicAdd(&hist[(k >> shift) & 255u], 1u);
            }
            __syncthreads();
            /* ---- the digit, from a SUFFIX SCAN over the bins and not a walk down them.
             *
             * The answer is the largest bin whose suffix sum still reaches `want`; a serial
             * form walks from 255 until it does. Real scores are a sum of four relus over a
             * positive constant, so they occupy a handful of adjacent exponents and that walk
             * spends two hundred of its iterations on empty bins ABOVE them, one thread, four
             * times a query. Eight parallel steps instead is worth about a third of the kernel at
             * decode, and it wins at PREFILL too (M=2048: 158 us against 289), which the key cache
             * above does not.
             *
             * Bit-identical to the walk by construction. `suf` is non-increasing in the index, so
             * `suf[t] >= want && suf[t+1] < want` holds for exactly one bin and it is the one the
             * walk stops at, with `run` its `suf[t+1]`. The fallback covers the case where the
             * walk reaches the bottom without ever arriving: fewer live keys than `want`, which
             * happens whenever enough of the table is padding. */
            radix_step<kThreads>(hist, wsum, (uint32_t)shift, fixed, want, &sh_T, &sh_need);
            fixed = sh_T;
            want  = sh_need;
        }
        T = fixed; need = want;
    }
    __syncthreads();
    if (t == 0) sh_out = 0;
    __syncthreads();

    /* ---- the emit. Two predicates: strictly above the threshold (always in) and exactly at it
     * (in while slots remain, LOWEST BLOCK INDEX FIRST). Both are placed by a workgroup prefix
     * count over chunks, so a block's slot depends only on how many qualifying blocks precede it
     * -- never on the order the hardware got there. */
    uint32_t taken_eq = 0;
    for (int b0 = 0; b0 < n; b0 += kThreads) {
        const int b = b0 + t;
        uint32_t k = 0;
        if (b < n) k = cached ? sh_key[b] : keyof(sc[b], tb[b]);
        const bool hi = k != 0u && (n <= topk || k > T);
        const bool eq = k != 0u && n > topk && k == T;

        /* Two scans in one pass: the high entries take their slots from the running output
         * count, the tie entries from the count of ties already taken.
         * AG05: Both counts are packed into a single uint32 (hi in low 16 bits, eq in high 16 bits),
         * reducing 4 CTA barriers down to 2 and halving wave scan dispatches with zero cross-lane carry. */
        const unsigned long long mh = __ballot(hi), me = __ballot(eq);
        const uint32_t rh = (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
        const uint32_t re = (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
        if (lane == 0) wsum[wave] = (uint32_t)__popcll(mh) | ((uint32_t)__popcll(me) << 16);
        __syncthreads();
        uint32_t before_pk, tot_pk;
        counts_before<kWaves>(wsum, lane, wave, &before_pk, &tot_pk);
        const uint32_t base_h = sh_out + (before_pk & 0xffffu);
        const uint32_t base_e = taken_eq + (before_pk >> 16);
        const uint32_t tot_h  = tot_pk & 0xffffu;
        const uint32_t tot_e  = tot_pk >> 16;

        if (hi) {
            const uint32_t slot = base_h + rh;
            if (slot < (uint32_t)topk) ou[slot] = tb[b];
        }
        if (eq) {
            const uint32_t rank = base_e + re;
            if (rank < need) {
                const uint32_t slot = (uint32_t)topk - need + rank;
                ou[slot] = tb[b];
            }
        }
        __syncthreads();
        if (t == 0) sh_out += tot_h;
        taken_eq += tot_e;
        __syncthreads();
    }

    /* ---- the partial page, appended last. `bt[n]` is the page the query itself lives in, and
     * there is one exactly when the query does NOT sit on the last token of a block.
     *
     * THE `(p + 1) % ratio` TEST IS WHAT KEEPS PREFILL CAUSAL, and at decode it is invisible. A
     * decode query on a block boundary has `n == nc`, and `bt[nc]` is the page after the
     * sequence's last, which the KV manager leaves -1 -- so a bare `part >= 0` test is right at
     * decode and only there. Mid-CHUNK the same query has later tokens of the
     * same chunk already stored in `bt[n]`, which is a LIVE page entirely in its future, and
     * `seqused` would then cover all `ratio` of them. The condition is the query's position, not
     * whether a page happens to be allocated. */
    if (t == 0) {
        uint32_t out = n <= topk ? sh_out : (uint32_t)topk;
        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? tb[n] : -1;
        if (part >= 0) ou[out++] = part;
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[m] = (int32_t)out;
        /* THE ATTENTION'S CAUSAL BOUND, AND IT IS WHY THIS OP EXISTS IN THE SHAPE IT DOES.
         *
         * `attn_paged` takes the block table and `seqused_k` as ORDINARY OPERANDS -- the
         * architecture already reads them off the KV group batch and passes them -- so QSA needs
         * no new attention kernel and no engine change: the same op is issued with `sel` in place
         * of the sequence's table and this number in place of its length.
         *
         * The flat KV the attention then walks is `out` pages of `ratio`, and the query sits at
         * flat index `(out - 1) * ratio + pos % ratio` because the partial page is LAST. One
         * formula covers both cases: with a tail the query is inside that last page, and without
         * one `pos % ratio` is `ratio - 1` and the expression collapses to `out * ratio`, which
         * is every selected page fully visible -- correct, because a COMPLETE page is entirely in
         * the query's past by construction. */
        const int32_t su = (int32_t)((out == 0) ? 0 : (out - 1) * (uint32_t)ratio +
                                                      (uint32_t)(p % ratio) + 1u);
        if (seqused) seqused[m] = su;
    }
}


/* ===================================================== the WIDE-LAUNCH path, for a prefill chunk
 *
 * AT A PREFILL CHUNK THE KERNEL ABOVE READS EVERY KEY FROM GLOBAL MEMORY FIVE TIMES -- four radix
 * digits and the emit, a score and a page id a block each time; the key cache is off for the
 * occupancy reason at kCacheMaxM -- and its first digit is the key's top EIGHT bits, which for a
 * score is its exponent. Real scores sit in a handful of adjacent exponents, so that pass puts
 * nearly every key into a few LDS bins one atomic at a time and settles almost nothing. At 80K of
 * context, 20K blocks a query, that form costs 1.43 ms a call and 5% of a prefill chunk, and it
 * grows with the context.
 *
 * So a wide launch takes the top TWELVE bits first -- the exponent and four bits of mantissa, 4096
 * bins -- which spreads the atomics and leaves one bin holding the threshold. A second pass copies
 * that bin's keys into LDS, over the bins the first digit no longer needs, and the other three
 * digits run on them there; the emit is the third and last pass over global memory -- or, when
 * the bin is small enough to keep its blocks too (kMergeCand), part of the second, which is then
 * the last. A bin larger than the LDS list runs those digits over global memory instead, as the
 * kernel above does.
 *
 * THE SAME TABLE, BIT FOR BIT. The threshold is a number, not a procedure -- the topk-th largest
 * key, and how many entries equal to it to take -- so any digit split finds the same one; and the
 * emit is the kernel above's, line for line, so the same entries land in the same slots.
 * r4d_selftest compares the two launches' tables byte for byte. */
constexpr int kWideBits    = 12;                     /* the first digit */
constexpr int kWideShift   = 31 - kWideBits;         /* a key's bit 31 is clear: see keyof */
constexpr int kWideBins    = 1 << kWideBits;
constexpr int kWideCand    = kWideBins - kBins;      /* keys the second pass may collect */
constexpr int kWidePer     = kWideBins / kThreads;   /* first-digit bins a thread scans */
static_assert(kWideBins % kThreads == 0 && kWideCand > 0, "the first digit's bins tile the block");

/* THE SECOND PASS IS ALSO THE EMIT when the threshold's bin is small enough to hold with its block
 * indices (at 150K of context and a 2048-query chunk, 1179 us a call against 1800): the pass over
 * global memory compacts, in block order, every key ABOVE that bin (all of them are taken, and
 * there are fewer than topk) and every key IN it with its block, so once the later digits have
 * fixed the threshold the table is a merge of the two lists by block index and the third pass
 * over global memory is not needed. Same entries in the same slots: the taken keys above the
 * threshold in block order, the ties after them in block order. The three lists share the first
 * digit's bins, so the LDS and the occupancy are the three-pass form's; a larger bin, or a wider
 * topk, takes the three-pass form. */
constexpr int kMergeAbove  = 512;                    /* keys above the threshold's bin */
constexpr int kMergeCand   = (kWideCand - kMergeAbove) / 2;   /* keys, and blocks, in it */
static_assert(kMergeCand > 0, "room for the bin's keys and their blocks");

__device__ __forceinline__ uint32_t lower_bound_u32(const uint32_t* v, uint32_t n, uint32_t x) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (v[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* THE FIRST PASS IS A SAMPLE when the history is long (at 150K of context a query has 37.5K keys:
 * 0.67 ms a layer this way, 1.2 ms as two passes over them): kSampRuns runs of kSampRun
 * consecutive keys spread over the history bound the threshold from below, one pass collects every
 * key at or above that bound with its block in block order, and the exact select and the emit run
 * on that list in LDS. EXACT BY CONSTRUCTION, not by estimate: when at least topk keys clear the
 * bound, the threshold is at or above it, so every taken key -- ties included -- is in the list, in
 * block order, and lands in the slot the full passes give it. When fewer clear it, or more than the
 * list holds, the kernel takes the two-pass form below; the sample only decides which.
 *
 * Runs rather than a stride, because a strided sample touches every cache line of the scores and
 * saves nothing. The bound is the sample's (2 topk S / n + 8)-th largest, about twice the
 * expected count, so the list is ~1000 of kSampCap and short of topk rarely. */
constexpr int kSampRun   = 32;
constexpr int kSampRuns  = 64;
constexpr int kSamp      = kSampRun * kSampRuns;             /* 2048 keys */
/* Below it the two passes win: at a 2048-query chunk 174 against 196 us at 5000 blocks, and the
 * sample 221 against 243 at 7500. */
constexpr int kSampMinN  = 3 * kSamp;
constexpr int kSampCap   = (kWideBins - kBins) / 2;          /* list entries: a key and a block */
static_assert(kSamp + kBins <= kWideBins, "the sample and its histogram share the bins");

/* The `want`-th largest of `cnt` keys in LDS, and how many entries equal to it to take: the
 * single-workgroup kernel's four eight-bit digits over a list. Fewer than `want` live keys leave
 * the threshold at 0, which every live key exceeds. */
__device__ __forceinline__ void lds_kth(const uint32_t* keys, int cnt, uint32_t want,
                                        uint32_t* hist, uint32_t* wsum,
                                        uint32_t* sh_T, uint32_t* sh_need, uint32_t* T,
                                        uint32_t* need) {
    const int t = (int)threadIdx.x;
    uint32_t fixed = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < kBins; i += kThreads) hist[i] = 0;
        __syncthreads();
        const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int i = t; i < cnt; i += kThreads) {
            const uint32_t k = keys[i];
            if (k != 0u && (k & mask) == (fixed & mask)) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        radix_step<kThreads>(hist, wsum, (uint32_t)shift, fixed, want, sh_T, sh_need);
        fixed = *sh_T;
        want  = *sh_need;
    }
    *T = fixed;
    *need = want;
}

__global__ __launch_bounds__(kThreads) void qsa_select_wide_kernel(
    const float* __restrict__ score, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    int32_t* __restrict__ sel, int32_t* __restrict__ nsel, int32_t* __restrict__ seqused,
    int nb, int topk, int ratio, int nseq, int s_ld, int t_ld, int o_ld,
    int m_offset) {

    /* The first digit's 4096 bins; after it, bins [0, kBins) are the later digits' histogram and
     * the rest hold the collected keys. */
    __shared__ uint32_t wide[kWideBins];
    __shared__ uint32_t wsum[kWaves];
    __shared__ uint32_t sh_T, sh_need, sh_out, sh_cnt, sh_bin;
    __shared__ int      sh_seq;
    uint32_t* const hist = wide;
    uint32_t* const cand = wide + kBins;
    /* The merged form's lists, over the same words: the bin's keys, their blocks, and the blocks
     * above the bin. */
    uint32_t* const mkey = wide + kBins;
    uint32_t* const mblk = mkey + kMergeCand;
    uint32_t* const mabv = mblk + kMergeCand;
    bool merged = false;

    const int m = blockIdx.x;
    const int t = threadIdx.x;
    const int lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) sh_seq = -1;
    __syncthreads();
    const int gm = m + m_offset;
    for (int i = t; i < nseq; i += kThreads)
        if (gm >= cu[i] && gm < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    if (seq < 0) {
        if (t == 0) { nsel[m] = 0; seqused[m] = 0; }
        return;
    }
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;
    int n = nc[seq] < nb ? nc[seq] : nb;
    if (own < n) n = own;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt    + (size_t)seq * t_ld;
    int32_t*       ou = sel   + (size_t)m * o_ld;

    uint32_t T = 0, need = 0;
    bool sampled = false;
    if (n > topk && n >= kSampMinN && topk <= kMergeAbove) {
        /* ---- the sample, and the bound from it. */
        uint32_t* const samp = wide;
        for (int i = t; i < kSamp; i += kThreads) {
            const int  r = i / kSampRun, j = i % kSampRun;
            const long b = (long)r * (n - kSampRun) / (kSampRuns - 1) + j;
            samp[i] = keyof(sc[b], tb[b]);
        }
        __syncthreads();
        uint32_t m = (uint32_t)(((uint64_t)topk * 2u * kSamp + (uint64_t)n - 1u) / (uint64_t)n) + 8u;
        if (m > (uint32_t)kSamp) m = (uint32_t)kSamp;
        uint32_t lo, lo_need;
        lds_kth(samp, kSamp, m, wide + kSamp, wsum, &sh_T, &sh_need, &lo, &lo_need);

        /* ---- one pass: every key at or above the bound, with its block, in block order. */
        uint32_t* const lkey = wide + kBins;
        uint32_t* const lblk = lkey + kSampCap;
        /* EIGHT CONSECUTIVE KEYS A THREAD, so a pass is a vector load stream with one workgroup
         * prefix -- two barriers -- every 2048 keys rather than every 256. Block order holds: a
         * thread's keys are consecutive and the threads are in order. */
        constexpr int kPer = 8;
        const bool vec = (((uintptr_t)sc | (uintptr_t)tb) & 15u) == 0u;
        uint32_t cnt = 0;
        for (int b0 = 0; b0 < n; b0 += kThreads * kPer) {
            const int bb = b0 + t * kPer;
            uint32_t kk[kPer];
            if (vec && bb + kPer <= n) {
                const float4 s0 = *(const float4*)(sc + bb), s1 = *(const float4*)(sc + bb + 4);
                const int4   p0 = *(const int4*)(tb + bb),   p1 = *(const int4*)(tb + bb + 4);
                kk[0] = keyof(s0.x, p0.x); kk[1] = keyof(s0.y, p0.y);
                kk[2] = keyof(s0.z, p0.z); kk[3] = keyof(s0.w, p0.w);
                kk[4] = keyof(s1.x, p1.x); kk[5] = keyof(s1.y, p1.y);
                kk[6] = keyof(s1.z, p1.z); kk[7] = keyof(s1.w, p1.w);
            } else {
#pragma unroll
                for (int e = 0; e < kPer; ++e)
                    kk[e] = bb + e < n ? keyof(sc[bb + e], tb[bb + e]) : 0u;
            }
            uint32_t mine = 0;
#pragma unroll
            for (int e = 0; e < kPer; ++e) mine += (kk[e] != 0u && kk[e] >= lo) ? 1u : 0u;
            const uint32_t inc = r4d_wave_prefix_u32(mine, lane);
            if (lane == kWave - 1) wsum[wave] = inc;
            __syncthreads();
            uint32_t before, tot;
            counts_before<kWaves>(wsum, lane, wave, &before, &tot);
            uint32_t j = cnt + before + inc - mine;
#pragma unroll
            for (int e = 0; e < kPer; ++e) {
                if (kk[e] == 0u || kk[e] < lo) continue;
                if (j < (uint32_t)kSampCap) { lkey[j] = kk[e]; lblk[j] = (uint32_t)(bb + e); }
                ++j;
            }
            cnt += tot;
            __syncthreads();
        }

        /* ---- the table from the list, when the list holds the answer. */
        if (cnt >= (uint32_t)topk && cnt <= (uint32_t)kSampCap) {
            lds_kth(lkey, (int)cnt, (uint32_t)topk, hist, wsum, &sh_T, &sh_need, &T, &need);
            uint32_t hbase = 0, ebase = 0;
            for (uint32_t i0 = 0; i0 < cnt; i0 += kThreads) {
                const uint32_t i = i0 + (uint32_t)t;
                const uint32_t k = i < cnt ? lkey[i] : 0u;
                const bool hi = i < cnt && k > T;
                const bool eq = i < cnt && k == T;
                const unsigned long long mh = __ballot(hi), me = __ballot(eq);
                if (lane == 0) wsum[wave] = (uint32_t)__popcll(mh) | ((uint32_t)__popcll(me) << 16);
                __syncthreads();
                uint32_t before_pk, tot_pk;
                counts_before<kWaves>(wsum, lane, wave, &before_pk, &tot_pk);
                const uint32_t hp = hbase + (before_pk & 0xffffu) +
                                    (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
                const uint32_t ep = ebase + (before_pk >> 16) +
                                    (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
                if (hi) ou[hp] = tb[lblk[i]];
                if (eq && ep < need) ou[(uint32_t)topk - need + ep] = tb[lblk[i]];
                hbase += (tot_pk & 0xffffu);
                ebase += (tot_pk >> 16);
                __syncthreads();
            }
            sampled = true;
        }
        __syncthreads();
    }
    if (n > topk && !sampled) {
        /* ---- the first digit, over global memory. */
        for (int i = t; i < kWideBins; i += kThreads) wide[i] = 0;
        __syncthreads();
        for (int b = t; b < n; b += kThreads) {
            const uint32_t k = keyof(sc[b], tb[b]);
            if (k != 0u) atomicAdd(&wide[k >> kWideShift], 1u);
        }
        __syncthreads();

        /* The bin holding the topk-th largest key: the largest bin whose suffix count reaches
         * `topk`. A thread owns kWidePer consecutive bins; the counts above its range come from a
         * suffix scan over the threads, and it walks its own bins down from there. None qualifies
         * when fewer than topk keys are live, and then every live key is in: T stays 0 and `need`
         * is what is left, as in the kernel above. */
        uint32_t loc[kWidePer], mine = 0;
#pragma unroll
        for (int j = 0; j < kWidePer; ++j) { loc[j] = wide[t * kWidePer + j]; mine += loc[j]; }
        const uint32_t x = r4d_wave_suffix_u32(mine, lane);
        if (lane == 0) wsum[wave] = x;
        __syncthreads();
        uint32_t above = 0;
        for (int w = wave + 1; w < kWaves; ++w) above += wsum[w];
        uint32_t run = x - mine + above;
        if (t == 0) { sh_bin = 0xFFFFFFFFu; sh_T = 0; sh_need = (uint32_t)topk - (run + mine); }
        __syncthreads();
#pragma unroll
        for (int j = kWidePer - 1; j >= 0; --j) {
            if (run < (uint32_t)topk && run + loc[j] >= (uint32_t)topk) {
                sh_bin  = (uint32_t)(t * kWidePer + j);
                sh_need = (uint32_t)topk - run;
                sh_cnt  = loc[j];
            }
            run += loc[j];
        }
        __syncthreads();
        const uint32_t B = sh_bin;

        if (B != 0xFFFFFFFFu) {
            uint32_t fixed = B << kWideShift, want = sh_need;
            /* ---- the threshold's bin, into LDS when it fits: with its blocks and the keys above
             * it, in block order, when those fit too (the merged form); else its keys alone. */
            merged = sh_cnt <= (uint32_t)kMergeCand && topk <= kMergeAbove;
            const bool inlds = sh_cnt <= (uint32_t)kWideCand;
            __syncthreads();
            if (t == 0) sh_cnt = 0;
            __syncthreads();
            uint32_t n_abv = 0;
            if (merged) {
                uint32_t n_in = 0;
                for (int b0 = 0; b0 < n; b0 += kThreads) {
                    const int b = b0 + t;
                    const uint32_t k = b < n ? keyof(sc[b], tb[b]) : 0u;
                    const bool abv = k != 0u && (k >> kWideShift) > B;
                    const bool inb = k != 0u && (k >> kWideShift) == B;
                    const unsigned long long ma = __ballot(abv), mi = __ballot(inb);
                    if (lane == 0) wsum[wave] = (uint32_t)__popcll(ma) | ((uint32_t)__popcll(mi) << 16);
                    __syncthreads();
                    uint32_t before_pk, tot_pk;
                    counts_before<kWaves>(wsum, lane, wave, &before_pk, &tot_pk);
                    if (abv)
                        mabv[n_abv + (before_pk & 0xffffu) + (uint32_t)__popcll(ma & ((1ull << lane) - 1ull))] =
                            (uint32_t)b;
                    if (inb) {
                        const uint32_t j = n_in + (before_pk >> 16) +
                                           (uint32_t)__popcll(mi & ((1ull << lane) - 1ull));
                        mkey[j] = k;
                        mblk[j] = (uint32_t)b;
                    }
                    n_abv += (tot_pk & 0xffffu);
                    n_in  += (tot_pk >> 16);
                    __syncthreads();
                }
                if (t == 0) sh_cnt = n_in;
                __syncthreads();
            } else if (inlds) {
                for (int b = t; b < n; b += kThreads) {
                    const uint32_t k = keyof(sc[b], tb[b]);
                    if (k != 0u && (k >> kWideShift) == B) cand[atomicAdd(&sh_cnt, 1u)] = k;
                }
                __syncthreads();
            }
            const int ncand = (merged || inlds) ? (int)sh_cnt : n;
            const uint32_t* const keys = merged ? mkey : cand;

            /* ---- the other 19 bits: digits of 8, 8 and 3. */
            const int shifts[3] = { kWideShift - 8, kWideShift - 16, 0 };
            const int widths[3] = { 8, 8, kWideShift - 16 };
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                const int      shift = shifts[d];
                const uint32_t dm    = (1u << widths[d]) - 1u;
                const uint32_t hm    = 0xFFFFFFFFu << (shift + widths[d]);
                for (int i = t; i < kBins; i += kThreads) hist[i] = 0;
                __syncthreads();
                for (int i = t; i < ncand; i += kThreads) {
                    const uint32_t k = (merged || inlds) ? keys[i] : keyof(sc[i], tb[i]);
                    if (k != 0u && (k & hm) == (fixed & hm)) atomicAdd(&hist[(k >> shift) & dm], 1u);
                }
                __syncthreads();
                radix_step<kThreads>(hist, wsum, (uint32_t)shift, fixed, want, &sh_T, &sh_need);
                fixed = sh_T;
                want  = sh_need;
            }
            T = fixed; need = want;

            if (merged) {
                /* ---- the emit, from the lists. A key of the bin above the threshold takes the
                 * slot after every taken key before its block: the bin's own before it, then the
                 * above-bin keys before its block, found by search. Its running count replaces
                 * its key, and an above-bin key finds its slot the same way from the other side.
                 * The ties fill from topk - need, in block order. */
                const uint32_t nin = (uint32_t)ncand;
                uint32_t hbase = 0, ebase = 0;
                for (uint32_t i0 = 0; i0 < nin; i0 += kThreads) {
                    const uint32_t i = i0 + (uint32_t)t;
                    const uint32_t k = i < nin ? mkey[i] : 0u;
                    const bool hi = i < nin && k > T;
                    const bool eq = i < nin && k == T;
                    const unsigned long long mh = __ballot(hi), me = __ballot(eq);
                    if (lane == 0) wsum[wave] = (uint32_t)__popcll(mh) | ((uint32_t)__popcll(me) << 16);
                    __syncthreads();
                    uint32_t before_pk, tot_pk;
                    counts_before<kWaves>(wsum, lane, wave, &before_pk, &tot_pk);
                    const uint32_t hp = hbase + (before_pk & 0xffffu) +
                                        (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
                    const uint32_t ep = ebase + (before_pk >> 16) +
                                        (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
                    if (i < nin) {
                        const uint32_t b = mblk[i];
                        if (hi) ou[hp + lower_bound_u32(mabv, n_abv, b)] = tb[b];
                        if (eq && ep < need) ou[(uint32_t)topk - need + ep] = tb[b];
                        mkey[i] = hp;
                    }
                    hbase += (tot_pk & 0xffffu);
                    ebase += (tot_pk >> 16);
                    __syncthreads();
                }
                for (uint32_t j = (uint32_t)t; j < n_abv; j += kThreads) {
                    const uint32_t b = mabv[j];
                    const uint32_t p = lower_bound_u32(mblk, nin, b);
                    ou[j + (p < nin ? mkey[p] : hbase)] = tb[b];
                }
            }
        } else {
            T = sh_T; need = sh_need;
        }
    }
    __syncthreads();
    if (t == 0) sh_out = 0;
    __syncthreads();

    /* ---- the emit, the kernel above's line for line. */
    uint32_t taken_eq = 0;
    for (int b0 = 0; b0 < ((merged || sampled) ? 0 : n); b0 += kThreads) {
        const int b = b0 + t;
        uint32_t k = 0;
        if (b < n) k = keyof(sc[b], tb[b]);
        const bool hi = k != 0u && (n <= topk || k > T);
        const bool eq = k != 0u && n > topk && k == T;

        const unsigned long long mh = __ballot(hi), me = __ballot(eq);
        const uint32_t rh = (uint32_t)__popcll(mh & ((1ull << lane) - 1ull));
        const uint32_t re = (uint32_t)__popcll(me & ((1ull << lane) - 1ull));
        if (lane == 0) wsum[wave] = (uint32_t)__popcll(mh) | ((uint32_t)__popcll(me) << 16);
        __syncthreads();
        uint32_t before_pk, tot_pk;
        counts_before<kWaves>(wsum, lane, wave, &before_pk, &tot_pk);
        const uint32_t base_h = sh_out + (before_pk & 0xffffu);
        const uint32_t base_e = taken_eq + (before_pk >> 16);
        const uint32_t tot_h  = tot_pk & 0xffffu;
        const uint32_t tot_e  = tot_pk >> 16;

        if (hi) {
            const uint32_t slot = base_h + rh;
            if (slot < (uint32_t)topk) ou[slot] = tb[b];
        }
        if (eq) {
            const uint32_t rank = base_e + re;
            if (rank < need) {
                const uint32_t slot = (uint32_t)topk - need + rank;
                ou[slot] = tb[b];
            }
        }
        __syncthreads();
        if (t == 0) sh_out += tot_h;
        taken_eq += tot_e;
        __syncthreads();
    }

    /* ---- the partial page, appended last: the kernel above's rule. */
    if (t == 0) {
        uint32_t out = n <= topk ? sh_out : (uint32_t)topk;
        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? tb[n] : -1;
        if (part >= 0) ou[out++] = part;
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[m] = (int32_t)out;
        const int32_t su = (int32_t)((out == 0) ? 0 : (out - 1) * (uint32_t)ratio +
                                                      (uint32_t)(p % ratio) + 1u);
        if (seqused) seqused[m] = su;
    }
}

/* THE SELECTION'S FORM, pinned for a test: 0 the launcher's rule, 1 the single-workgroup kernel,
 * 2 the wide-launch kernel. The split path is its own rule and neither pin touches it. */
static int g_qsa_select_form = 0;

/* ===================================================== the split path, for a DECODE-shaped launch
 *
 * ONE WORKGROUP A QUERY IS SIX PERCENT OF THE DEVICE, and at length that is the whole kernel. The
 * grid above is `M` and nothing else, so a decode step runs `1 + n_spec` workgroups -- four at the
 * shipped depth -- each walking the entire block list five times. At the deployment's geometry
 * (topk 512, ratio 4) its cost barely moves from M=1 to M=8, around 255-285 us either way, which
 * is the signature of a kernel that is one workgroup deep however wide the batch is. At 200K of
 * context that is ~257 us a layer a step, and sixteen such layers make it 4.1 ms.
 *
 * A CHUNK'S LOCAL TOP-K CONTAINS EVERY GLOBAL TOP-K MEMBER OF THAT CHUNK, so the block list splits
 * with nothing shared and the answer is still EXACT. Suppose a block is in the global top-k but not
 * in its own chunk's: then k blocks of that chunk outrank it, and a block that outranks it in the
 * chunk outranks it globally -- so k blocks outrank it globally and it was not in the global top-k.
 * The order has to be the same TOTAL order on both sides for that to hold, and it is: key
 * descending, then block index ascending, which is what the emit below builds by prefix count.
 *
 * So: `kParW` chunks a query find their own top-k in LDS, and one workgroup merges at most
 * `kParW * topk` candidates -- 6656 at the deployment's shape against 50000 -- and emits. The
 * merge's input is chunk-ordered and each chunk's list is block-ordered, so the concatenation is
 * block-ordered and the tie rule needs no sort.
 *
 * WHY NOT ALSO SPLIT THE MERGE: it reads a candidate list that is already small enough to stage,
 * and a second split would need a global prefix over chunk counts to keep the tie rule exact --
 * two more kernels and two more dispatch gaps (2.8 us each on this part) for a pass that costs
 * about 4% of what the unsplit kernel does.
 *
 * The path is DECODE-SHAPED and says so: above `kParM` query rows the grid is wide enough on its
 * own and the single-workgroup kernel above is what runs. A prefill chunk is 512 queries. */
constexpr int kParM    = 32;      /* query rows the split path serves; a prefill chunk is wider */
constexpr int kParWMax = 16;      /* chunks a query, and what the scratch is sized for */
constexpr int kParCap  = 8192;    /* keys a chunk may stage: 32 KiB, the same budget as above */

/* THE MERGE IS WIDER THAN EVERYTHING ELSE, because it is the one kernel here with no blocks to
 * spare. It runs one workgroup a query -- four at the deployment's decode shape -- so it occupies
 * four CUs of sixty-four and the only parallelism available to it is INSIDE a workgroup. At 256
 * threads that is eight waves over a CU's four SIMDs, two apiece, which is not enough to hide an
 * LDS round trip let alone a global one; at 1024 it is eight apiece. VGPR is 24, so occupancy is
 * not the limit -- the workgroup size is.
 *
 * The bin scan does not widen with it (256 bins, one a thread) and `par_scan`'s count prefix is
 * one wave's scan at any width, so neither of those is what the width buys; the wave count is. */
constexpr int kMergeThreads = 1024;
/* The local pass has forty-eight workgroups at this shape and a CU each, so it is not starved of
 * waves the way the merge is and takes a narrower block. */
constexpr int kLocalThreads = 512;

/* How many chunks a block list of `nb` is cut into, and the cut is the same on both sides of the
 * pair -- the merge indexes the scratch at a fixed `kParWMax` stride whatever this returns, so the
 * two kernels agree by construction rather than by both recomputing it the same way. */
static inline int par_chunks(long long nb) {
    long long w = (nb + 4095) / 4096;
    if (w < 1) w = 1;
    if (w > kParWMax) w = kParWMax;
    return (int)w;
}

/* THE THRESHOLD, over keys already staged in LDS. Digit for digit the same four-pass radix and the
 * same suffix scan the single-workgroup kernel above runs, so a split selection and an unsplit one
 * are the same bits -- which is the property the whole path rests on. */
/* TEMPLATED ON THE WORKGROUP WIDTH, because the merge wants to be wider than the bin count.
 *
 * The bin scan is still one bin to one thread and always will be -- it is a 256-entry suffix scan
 * -- so above 256 threads the surplus has to sit the scan out. The guards are around the WORK and
 * never around a `__syncthreads()`: a barrier reached by some threads of a block and not others is
 * undefined, and the whole point of widening is that some threads have nothing to do here. */
template <int NT>
__device__ __forceinline__ void radix_thresh(const uint32_t* __restrict__ key, int n,
                                             uint32_t want, uint32_t* hist,
                                             uint32_t* wt, uint32_t* sh_T, uint32_t* sh_need) {
    const int t = (int)threadIdx.x;
    uint32_t fixed = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < kBins; i += NT) hist[i] = 0;
        __syncthreads();
        const uint32_t mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (int b = t; b < n; b += NT) {
            const uint32_t k = key[b];
            if (k != 0u && (k & mask) == (fixed & mask))
                atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        radix_step<NT>(hist, wt, (uint32_t)shift, fixed, want, sh_T, sh_need);
        fixed = *sh_T;
        want  = *sh_need;
    }
}

/* THE QUALIFYING SET, BY PREFIX COUNT, IN TWO LAYOUTS -- and the second one is not a nicety.
 *
 * The single-workgroup kernel puts the entries strictly above the threshold at the FRONT of the
 * output table and the ties at the BACK, because a selected set has no order. The attention that
 * reads the table disagrees: it accumulates a softmax over the pages in table order, and float
 * addition is not associative, so a permuted table is different BITS. This tree compares generated
 * text across builds byte for byte, so the split path has to reproduce that layout exactly.
 *
 * The chunk pass wants the other one: its output is the merge's input and the merge's tie rule
 * reads it in block order, so there a tie has to land in its block's place. Hence both prefixes
 * are computed and the caller picks. Every count is a workgroup prefix over chunks of `kThreads`,
 * so a candidate's slot depends only on how many candidates precede it, never on scheduling. */
struct ParEmit {
    uint32_t* wsum;                    /* [kWaves] scratch for the wave prefix */
    uint32_t* sh_hi;                   /* running count of entries above the threshold */
    uint32_t* sh_ord;                  /* running count of entries taken, in block order */
    uint32_t  taken_eq;                /* ties taken so far; per thread, and uniform by induction */
    uint32_t  T, need;
    bool      all;                     /* below the top-k: everything live is in */
};
struct ParTake {
    bool     hi, take;
    uint32_t slot_hi, slot_ord, rank_eq;
};

/* One ballot, one wave prefix, one workgroup prefix -- the shape the loop below repeats three
 * times. `*run` is the running total and is advanced by thread 0. */
template <int NT>
__device__ __forceinline__ uint32_t par_scan(ParEmit& e, bool p, int lane, int wave,
                                             uint32_t* run, uint32_t base0, uint32_t* total) {
    constexpr int NW = NT / kWave;
    const unsigned long long mk = __ballot(p);
    const uint32_t rk = (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
    if (lane == 0) e.wsum[wave] = (uint32_t)__popcll(mk);
    __syncthreads();
    uint32_t before, tot;
    counts_before<NW>(e.wsum, lane, wave, &before, &tot);
    const uint32_t base = base0 + before;
    __syncthreads();
    if (total) *total = tot;
    if (run && threadIdx.x == 0) *run += tot;
    return base + rk;
}

/* WHICH SLOTS THIS CALLER ACTUALLY READS, as template parameters, because neither caller reads
 * both and without them each pays for the other's.
 *
 * `par_scan` is not cheap: a ballot, two `__syncthreads()` and a scan of the wave counts. The emit
 * calls it once per predicate per iteration of a loop that runs `tot/kThreads` times, so a scan
 * nobody reads is a third of the emit.
 *
 * And a third of them go unread: the MERGE uses `slot_hi` and `rank_eq` and never touches
 * `slot_ord`, while the LOCAL pass uses `slot_ord` and never touches `slot_hi`. Computing both
 * unconditionally is expensive -- the emit is over half of the merge's time, and the threshold
 * scan most of the rest.
 *
 * The branches are on template parameters, so they are resolved at compile time and every thread
 * of the block takes the same path -- which is what makes a `__syncthreads()` inside them legal.
 * The running counter a skipped scan would have advanced is the one its caller does not read.
 *
 * Bit-identical: the scans are independent of each other except that `take` needs `rank_eq`, and
 * that ordering is kept. */
template <bool WantHi, bool WantOrd, int NT>
__device__ __forceinline__ ParTake par_take(ParEmit& e, uint32_t k, int lane, int wave) {
    ParTake r{};
    r.hi        = k != 0u && (e.all || k > e.T);
    const bool eq = k != 0u && !e.all && k == e.T;

    uint32_t tot_e = 0;
    r.rank_eq = par_scan<NT>(e, eq, lane, wave, nullptr, e.taken_eq, &tot_e);
    r.take    = r.hi || (eq && r.rank_eq < e.need);
    if (WantHi)  r.slot_hi  = par_scan<NT>(e, r.hi,   lane, wave, e.sh_hi,  *e.sh_hi,  nullptr);
    if (WantOrd) r.slot_ord = par_scan<NT>(e, r.take, lane, wave, e.sh_ord, *e.sh_ord, nullptr);
    e.taken_eq += tot_e;
    __syncthreads();
    return r;
}

/* ---- pass one: every chunk's own top-k, as (key, block index) in block order. */
__global__ __launch_bounds__(kLocalThreads) void qsa_local_kernel(
    const float* __restrict__ score, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    uint32_t* __restrict__ ck, int32_t* __restrict__ ci, int32_t* __restrict__ cc,
    int nb, int topk, int ratio, int nseq, int s_ld, int t_ld, int chunk,
    int m_offset) {

    constexpr int LT = kLocalThreads;
    __shared__ uint32_t hist[kBins], wsum[LT / kWave];
    __shared__ uint32_t sh_T, sh_need, sh_hi, sh_ord;
    __shared__ int      sh_seq;
    extern __shared__ uint32_t sh_key[];

    const int m = (int)blockIdx.y, c = (int)blockIdx.x;
    const int t = (int)threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) { sh_seq = -1; cc[(size_t)m * kParWMax + c] = 0; }
    __syncthreads();
    const int gm = m + m_offset;
    for (int i = t; i < nseq; i += LT)
        if (gm >= cu[i] && gm < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    if (seq < 0) return;  /* The merge pass zeros the padded row's outputs. */
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;
    int n = nc[seq] < nb ? nc[seq] : nb;
    if (own < n) n = own;

    const int lo = c * chunk;
    if (lo >= n) return;
    const int hi = (lo + chunk) < n ? (lo + chunk) : n;
    const int cn = hi - lo;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt    + (size_t)seq * t_ld;
    for (int i = t; i < cn; i += LT) sh_key[i] = keyof(sc[lo + i], tb[lo + i]);
    __syncthreads();

    ParEmit e{ wsum, &sh_hi, &sh_ord, 0u, 0u, 0u, cn <= topk };
    if (!e.all) {
        radix_thresh<LT>(sh_key, cn, (uint32_t)topk, hist, wsum, &sh_T, &sh_need);
        e.T = sh_T; e.need = sh_need;
    }
    if (t == 0) { sh_hi = 0; sh_ord = 0; }
    __syncthreads();

    /* IN BLOCK ORDER, because this list is the merge's input and the merge breaks ties by block
     * index. The output table's own layout is the merge's business. */
    uint32_t* dk = ck + (size_t)(m * kParWMax + c) * topk;
    int32_t*  di = ci + (size_t)(m * kParWMax + c) * topk;
    for (int i0 = 0; i0 < cn; i0 += LT) {
        const int i = i0 + t;
        const uint32_t k = i < cn ? sh_key[i] : 0u;
        const ParTake r = par_take<false, true, LT>(e, k, lane, wave);
        if (r.take && r.slot_ord < (uint32_t)topk) {
            dk[r.slot_ord] = k;
            /* THE PAGE, NOT THE BLOCK INDEX, and the difference is which kernel pays for the
             * lookup. Leaving it to the merge means resolving `tb[ci[...]]` there -- two DEPENDENT
             * global loads for every candidate it takes -- on four workgroups, one a query, with
             * nothing to hide the second round trip behind. Resolving it here costs the same load
             * on FORTY-EIGHT workgroups, where there is. Nothing downstream wants the index: the
             * merge's tie rule breaks on a candidate's position in the concatenated list, which is
             * block-ordered by construction, and the index is never compared. A taken candidate
             * has `k != 0`, which `keyof` only returns for a live page, so no dead page arrives
             * here. */
            di[r.slot_ord] = tb[lo + i];
        }
    }
    if (t == 0) cc[(size_t)m * kParWMax + c] =
        (int32_t)(sh_ord < (uint32_t)topk ? sh_ord : (uint32_t)topk);
}

/* ---- pass two: merge the chunk lists and emit the page-id table. */
__global__ __launch_bounds__(kMergeThreads) void qsa_merge_kernel(
    const uint32_t* __restrict__ ck, const int32_t* __restrict__ ci,
    const int32_t* __restrict__ cc, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    int32_t* __restrict__ sel, int32_t* __restrict__ nsel, int32_t* __restrict__ seqused,
    int nb, int topk, int ratio, int nseq, int t_ld, int o_ld, int nchunk,
    int m_offset) {

    constexpr int NT = kMergeThreads;
    __shared__ uint32_t hist[kBins], wsum[NT / kWave];
    __shared__ uint32_t sh_T, sh_need, sh_hi, sh_ord, sh_tot;
    __shared__ int      sh_seq, sh_off[kParWMax + 1];
    extern __shared__ uint32_t sh_key[];

    const int m = (int)blockIdx.x;
    const int t = (int)threadIdx.x, lane = t & (kWave - 1), wave = t >> 5;

    if (t == 0) sh_seq = -1;
    __syncthreads();
    const int gm = m + m_offset;
    for (int i = t; i < nseq; i += NT)
        if (gm >= cu[i] && gm < cu[i + 1]) sh_seq = i;
    __syncthreads();
    const int seq = sh_seq;
    if (seq < 0) {
        if (t == 0) { nsel[m] = 0; seqused[m] = 0; }
        return;
    }
    const int p   = pos ? pos[m] : 0;
    const int own = (p + 1) / ratio;
    int n = nc[seq] < nb ? nc[seq] : nb;
    if (own < n) n = own;

    /* The chunk offsets, serially on one thread: `nchunk` is at most sixteen. */
    if (t == 0) {
        int s = 0;
        for (int c = 0; c < nchunk; ++c) { sh_off[c] = s; s += cc[(size_t)m * kParWMax + c]; }
        sh_off[nchunk] = s;
        sh_tot = (uint32_t)s;
    }
    __syncthreads();
    const int tot = (int)sh_tot;

    for (int c = 0; c < nchunk; ++c) {
        const int base = sh_off[c], cnt = sh_off[c + 1] - base;
        const uint32_t* src = ck + (size_t)(m * kParWMax + c) * topk;
        for (int i = t; i < cnt; i += NT) sh_key[base + i] = src[i];
    }
    __syncthreads();

    ParEmit e{ wsum, &sh_hi, &sh_ord, 0u, 0u, 0u, tot <= topk };
    if (!e.all) {
        radix_thresh<NT>(sh_key, tot, (uint32_t)topk, hist, wsum, &sh_T, &sh_need);
        e.T = sh_T; e.need = sh_need;
    }
    if (t == 0) { sh_hi = 0; sh_ord = 0; }
    __syncthreads();

    const int32_t* tb = bt  + (size_t)seq * t_ld;
    int32_t*       ou = sel + (size_t)m * o_ld;
    for (int i0 = 0; i0 < tot; i0 += NT) {
        const int i = i0 + t;
        const uint32_t k = i < tot ? sh_key[i] : 0u;
        const ParTake r = par_take<true, false, NT>(e, k, lane, wave);
        if (r.take) {
            /* The chunk a candidate came from, from the offsets; `nchunk` is at most sixteen. */
            int c = nchunk - 1;
            while (c > 0 && sh_off[c] > i) --c;
            /* Already a page: the local pass resolved it where there were blocks to hide
             * the load behind. `tb` is still needed below, for the partial page. */
            const int32_t page = ci[(size_t)(m * kParWMax + c) * topk + (i - sh_off[c])];
            /* THE SINGLE-WORKGROUP KERNEL'S LAYOUT, slot for slot: above the threshold at the
             * front, ties at the back. The attention sums the pages in table order. */
            if (r.hi) { if (r.slot_hi < (uint32_t)topk) ou[r.slot_hi] = page; }
            else       { ou[(uint32_t)topk - e.need + r.rank_eq] = page; }
        }
    }

    /* The partial page, appended last -- the same contract the single-workgroup kernel documents
     * at length: `bt[n]` is the page the query itself lives in, and there is one exactly when the
     * query does not sit on the last token of a block. */
    if (t == 0) {
        /* `n` AND NOT THE CANDIDATE COUNT, slot for slot with the single-workgroup kernel: past
         * the top-k the table is full to `topk` because the ties fill it, and below it the live
         * count is the answer. The merged candidate count can fall short of `topk` where the
         * single-workgroup kernel's table is full, and the padding loop below would then write -1
         * over the ties. */
        uint32_t out = n <= topk ? sh_hi : (uint32_t)topk;
        const int32_t part = (n < nb && (p + 1) % ratio != 0) ? tb[n] : -1;
        if (part >= 0) ou[out++] = part;
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[m] = (int32_t)out;
        const int32_t su = (int32_t)((out == 0) ? 0
                                   : (out - 1) * (uint32_t)ratio + (uint32_t)(p % ratio) + 1u);
        seqused[m] = su;
    }
}
}  // namespace

/* WHAT THE SPLIT PATH NEEDS, sized from `topk` alone so the answer is the same at declare -- when
 * the op has parameters and no tensors -- as it is at issue. Two candidate planes and a count per
 * (query, chunk), at the FIXED `kParM x kParWMax` stride both kernels index: 2.1 MiB at the
 * deployment's topk of 512, out of an arena that is shared with every other op's worst step. */
static inline int64_t par_scratch_bytes(long long topk) {
    if (topk <= 0) return 0;
    return (int64_t)kParM * kParWMax * topk * 8 + (int64_t)kParM * kParWMax * 4;
}

//   score [M][nb]        f32, `s_ld` elements a row (0 = nb)             -- M is QUERIES
//   bt    [nseq][nb]     i32, `t_ld` elements a row (0 = nb); page ids, -1 = absent
//   nc    [nseq]         i32, complete blocks a SEQUENCE
//   pos   [M]            i32, each query's absolute position
//   cu    [nseq+1]       i32, cumulative query counts: which sequence a query row belongs to
//   sel   [M][topk+1]    i32, `o_ld` elements a row (0 = topk+1); -1 padded
//   nsel  [M]            i32
static int r4d_qsa_select_bf16_emit(long score, long bt, long nc, long pos, long cu, long sel,
                                   long nsel, long seqused, int M, int nb, int topk, int ratio,
                                   int nseq, int s_ld, int t_ld, int o_ld,
                                   void* scratch, long long scratch_bytes, const R4dSink& sk,
                                   int m_offset = 0) {
    int emit_rc = RAD_OK;
    if (!score || !bt || !nc || !cu || !sel || !nsel) return -1;
    if (M <= 0 || nb <= 0 || topk <= 0 || ratio <= 0 || nseq <= 0) return -2;
    if (!s_ld) s_ld = nb;
    if (!t_ld) t_ld = nb;
    if (!o_ld) o_ld = topk + 1;

    /* THE SPLIT PATH WHEN THE LAUNCH IS DECODE-SHAPED AND EVERYTHING IT NEEDS IS THERE. Every
     * clause is a capacity: a wider launch already fills the device, a chunk or a candidate list
     * that will not stage is not a chunk, and a caller that gave no scratch gets the kernel that
     * needs none. None of them changes the answer -- see the proof above the split. */
    const int   nchunk = par_chunks(nb);
    const int   chunk  = (int)((nb + nchunk - 1) / nchunk);
    const int   lds    = lds_cap_of();
    const bool  split  = M <= kParM && nchunk > 1 && scratch != nullptr &&
                         scratch_bytes >= par_scratch_bytes(topk) &&
                         chunk <= kParCap && chunk <= lds && nchunk * topk <= lds;
    if (split) {
        uint32_t* ck = (uint32_t*)scratch;
        int32_t*  ci = (int32_t*)(ck + (size_t)kParM * kParWMax * topk);
        int32_t*  cc = ci + (size_t)kParM * kParWMax * topk;
        emit_rc = R4D_EMIT(sk, qsa_local_kernel, dim3((unsigned)nchunk, (unsigned)M), dim3(kLocalThreads),
                           (size_t)chunk * sizeof(uint32_t), (const float*)score, (const int32_t*)bt,
                           (const int32_t*)nc, (const int32_t*)pos, (const int32_t*)cu,
                           ck, ci, cc, nb, topk, ratio, nseq, s_ld, t_ld, chunk, m_offset);
        emit_rc = R4D_EMIT(sk, qsa_merge_kernel, dim3((unsigned)M), dim3(kMergeThreads),
                           (size_t)nchunk * topk * sizeof(uint32_t), (const uint32_t*)ck, (const int32_t*)ci,
                           (const int32_t*)cc, (const int32_t*)bt, (const int32_t*)nc,
                           (const int32_t*)pos, (const int32_t*)cu,
                           (int32_t*)sel, (int32_t*)nsel, (int32_t*)seqused,
                           nb, topk, ratio, nseq, t_ld, o_ld, nchunk, m_offset);
        return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
    }

    /* A WIDE LAUNCH -- a prefill chunk, where the key cache is off -- takes the three-pass form. */
    if (g_qsa_select_form == 2 || (g_qsa_select_form == 0 && M > cache_max_m())) {
        emit_rc = R4D_EMIT(sk, qsa_select_wide_kernel, dim3((unsigned)M), dim3(kThreads), 0,
                           (const float*)score, (const int32_t*)bt, (const int32_t*)nc,
                           (const int32_t*)pos, (const int32_t*)cu, (int32_t*)sel,
                           (int32_t*)nsel, (int32_t*)seqused, nb, topk, ratio, nseq, s_ld, t_ld,
                           o_ld, m_offset);
        return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
    }

    /* Never wider than the table itself: a short context asks for less LDS, not for the cap. */
    int cap = (M <= cache_max_m()) ? lds_cap_of() : 0;
    if (cap > nb) cap = nb;
    emit_rc = R4D_EMIT(sk, qsa_select_kernel, dim3((unsigned)M), dim3(kThreads),
                       (size_t)cap * sizeof(uint32_t), (const float*)score, (const int32_t*)bt,
                       (const int32_t*)nc, (const int32_t*)pos, (const int32_t*)cu,
                       (int32_t*)sel, (int32_t*)nsel,
                       (int32_t*)seqused, nb, topk, ratio, nseq, s_ld, t_ld, o_ld, cap, m_offset);
    return emit_rc == RAD_OK && (sk.desc || hipGetLastError() == hipSuccess) ? 0 : -4;
}

extern "C" int r4d_qsa_select_bf16_emit(long score, long bt, long nc, long pos, long cu, long sel,
                                   long nsel, long seqused, int M, int nb, int topk, int ratio,
                                   int nseq, int s_ld, int t_ld, int o_ld,
                                   void* scratch, long long scratch_bytes, long stream) {
    R4dSink sk; sk.stream = (RadStream)(uintptr_t)stream;
    return r4d_qsa_select_bf16_emit(score, bt, nc, pos, cu, sel, nsel, seqused, M, nb, topk, ratio, nseq, s_ld, t_ld, o_ld, scratch, scratch_bytes, sk);
}

/* The arena the split path asks for. Called at DECLARE, where the op has parameters and no
 * tensors -- so it reads `topk` and nothing shape-derived, and the fixed stride above is what
 * makes that legal. */
extern "C" void r4d_qsa_select_pin_form(int form) { g_qsa_select_form = form; }

extern "C" int64_t r4d_rad_qsa_select_scratch(const RadArgs* a) {
    return par_scratch_bytes(r4d_geti_or(a, "topk", 0));
}

// ============================================================== the plugin ABI entry point

enum { QL_SCORE = 0, QL_BT = 1, QL_NC = 2, QL_POS = 3, QL_CU = 4, QL_SEL = 5, QL_NSEL = 6,
       QL_SEQ = 7 };

static int r4d_rad_qsa_select_body(const RadArgs* a, const R4dSink& sk) {
    const RadTensor* sc = r4d_opt(a, QL_SCORE);
    const RadTensor* bt = r4d_opt(a, QL_BT);
    const RadTensor* nc = r4d_opt(a, QL_NC);
    const RadTensor* po = r4d_opt(a, QL_POS);
    const RadTensor* cu = r4d_opt(a, QL_CU);
    const RadTensor* se = r4d_opt(a, QL_SEL);
    const RadTensor* ns = r4d_opt(a, QL_NSEL);
    const RadTensor* su = r4d_opt(a, QL_SEQ);
    if (!sc || !bt || !nc || !po || !cu || !se || !ns || !su) return R4D_NO(RAD_E_INVAL);
    if (sc->rank != 2 || bt->rank != 2 || se->rank != 2) return R4D_NO(RAD_E_SHAPE);
    const long long topk  = r4d_geti_or(a, "topk", 0);
    const long long ratio = r4d_geti_or(a, "ratio", 0);
    if (topk <= 0 || ratio <= 0) return R4D_NO(RAD_E_INVAL);
    /* THE SMALLER OF THE TWO WIDTHS, for the reason r4d_qsa_score_bf16.hip gives at the same
     * check: the score plane is the architecture's buffer and the table's pitch is the KV
     * manager's, and requiring them equal refuses a configuration where they merely round
     * differently.
     * A query's live block count is bounded by max_ctx/ratio and both are sized for that, so the
     * minimum is never below it -- and `nc` bounds the range actually read regardless. */
    const long long M = sc->shape[0];
    const long long nb = bt->shape[1] < sc->shape[1] ? bt->shape[1] : sc->shape[1];
    if (M < 1 || nb < 1) return R4D_NO(RAD_E_SHAPE);
    /* THE SEQUENCE COUNT COMES OFF `cu`, not off the table's height, which is max_seqs. */
    if (cu->rank != 1 || cu->shape[0] < 2) return R4D_NO(RAD_E_SHAPE);
    const long long nseq = cu->shape[0] - 1;
    if (bt->shape[0] < nseq || r4d_numel(nc) < nseq) return R4D_NO(RAD_E_SHAPE);
    /* THE OUTPUT IS topk + 1 WIDE AND THE EXTRA SLOT IS NOT SPARE: it is the partial page, which
     * is appended unconditionally and is what the query itself lives in. A caller that sized it
     * at `topk` would drop the newest tokens on every query whose position is not a multiple of
     * the ratio -- which is three queries in four. */
    if (se->shape[0] != M || se->shape[1] != topk + 1) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(ns) < M) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(po) < M || r4d_numel(su) < M) return R4D_NO(RAD_E_SHAPE);
    const long long sp = r4d_row_pitch(sc), tp = r4d_row_pitch(bt), op = r4d_row_pitch(se);
    long long m_offset = r4d_geti_or(a, "m_offset", 0);
    const long long tp_part = r4d_geti_or(a, "tp_partition", 0);
    if (tp_part && a->world_size == 2 && a->rank == 1) {
        m_offset = M;
    }
    /* Scratch is OPTIONAL here and the launcher says so: the split path takes it when it is
     * there and big enough, and the single-workgroup kernel runs when it is not. */
    const int e = r4d_qsa_select_bf16_emit(r4d_addr(sc), r4d_addr(bt), r4d_addr(nc), r4d_addr(po),
                                      r4d_addr(cu), r4d_addr(se), r4d_addr(ns), r4d_addr(su),
                                      (int)M, (int)nb, (int)topk, (int)ratio, (int)nseq,
                                      (int)sp, (int)tp, (int)op, a->scratch, a->scratch_bytes, sk, (int)m_offset);
    return e == 0 ? RAD_OK : RAD_E_SHAPE;
}

R4D_ENTRY(r4d_rad_qsa_select, r4d_rad_qsa_select_body)
