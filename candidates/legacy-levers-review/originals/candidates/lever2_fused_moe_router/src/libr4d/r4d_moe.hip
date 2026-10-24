// R4D mixture of experts: the router, the scatter/gather pair, and the grouped block-scaled FP8
// GEMM. One translation unit, because the four are one graph and three of them exist only to feed
// the fourth.
//
// ---- WHAT THE GEMM IS SHAPED BY ---------------------------------------------------------------
//
// Qwen3.6-35B-A3B: 256 experts, 8 of them per token, 512 columns of intermediate each. So a decode
// step at eight requests routes 64 (token, slot) rows across up to 64 distinct experts, and reads
// ONE expert's gate_up [1024, 2048] plus its down [2048, 512] for each -- 3 MB an expert, 24 MB a
// layer, and that read IS the step. A grouped GEMM on this model is a bandwidth problem wearing a
// matmul's clothes, and everything below follows from that:
//
//   * A BLOCK OWNS A CONTIGUOUS RANGE OF SORTED ROWS AND LOOPS OVER THE EXPERTS IN IT. No block
//     map, no padded sort. The cost of a block that straddles two experts is that it stages two
//     weight tiles instead of one, and the number of (block, expert) incidences -- which is what
//     the weight traffic is proportional to -- is at most `n_expert + T/FM`. At prefill that is
//     384 against an optimum of 256, at decode it is within one of the number of distinct experts.
//
//     AT PREFILL A BLOCK MAP IS BUILT ALL THE SAME. See the AL note beside moe_blockmap_kernel:
//     at the shipped 2048 chunk the uniform grid is 414 incidences against 254, and a duplicate
//     staging of a POOLED expert is another pass of its shard over PCIe rather than an L2 hit. One
//     run a block removes every straddle. It is worth 3.6% of prefill end to end, bit-identical --
//     less than the incidence count suggests, because most duplicate stagings are served from
//     cache, so a straddle never costs the full link price of its bytes.
//
//     At DECODE the map is not built: the (tile, run) grid already gives one run a block, and the
//     incidence count is within one of the number of distinct experts.
//
//   * FM IS BANDED. A decode step's runs are a row or two, so it gets 16-row tiles spread one run a
//     block; a prefill chunk's are tens of rows, so it gets 32-row tiles one run a block, and a wave
//     skips the 16-row M tiles its run does not reach. See r4d_moe_decode_band for the crossover and
//     the kernel's prefill-tile note for why 32.
//
//   * WITHIN A 16-ROW TILE, DEAD ROWS ARE MASKED, NOT SKIPPED. A run of one row still runs a
//     16-row WMMA tile. On a kernel whose wall is the weight stream that is free, and it keeps the
//     K loop and its barriers wave-uniform, which a dynamic tile range would not.
//
// ---- WHY THE WEIGHT IS A POINTER TABLE --------------------------------------------------------
//
// `Wtab[e]` and not `W + e * stride`. A layer's experts are separate movement units (spec §4.1):
// after placement one is in VRAM, the next is streamed over the link and the third is still on
// disk, so there is no stride that describes them. RAD_OPK_WTAB is how the core says that; see
// abi/rad_runtime.h. The STACKED form -- an ordinary rank-3 [n_expert, N, K] tensor -- is accepted
// too, because a caller with no engine behind it (rad-kbench replaying a fixture) has nowhere to
// put a pointer array, and the two are told apart by the leading stride being zero.
//
// ---- DETERMINISM ------------------------------------------------------------------------------
//
// The scatter is a STABLE counting sort and it is stable without atomics ordering anything: a
// slot's position is `expert_offset[e] + blocks_before[b][e] + rank_within_block`, all three of
// which are computed by reduction rather than by whoever got there first. An atomic cursor would
// have been four lines and would have made `sorted_tok` -- and therefore every row of the grouped
// GEMM's output -- depend on the scheduler. This engine's reproducibility claims are byte-level
// (spec §17) and a non-deterministic sort forfeits all of them.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_fp8_wmma.h"
#include "r4d_fwht.h"
#include "r4d_q8.h"
#include "r4d_plugin.h"
#include "r4d_emit.h"
#include "r4d_gemm_args.h"

#include <hip/hip_bf16.h>

#include <cstdlib>

#include <algorithm>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

namespace {

constexpr uint32_t kWave  = 32;
/* THE EXPERT-COUNT CEILING, and it is a real limit of these kernels rather than a policy.
 * Three of them size something by it: the router keeps a lane's own logits in registers
 * (kMaxOwn below), and both scatter forms keep a histogram of n_expert + 1 counters in LDS.
 * 512 is what Qwen3.8-Flash-Next asks for -- 512 experts of 640 at top-10 -- and it costs
 * 4 KiB of LDS in the scatter and, in the router, a second instantiation rather than sixteen
 * registers on every model. */
constexpr uint32_t kMaxExpert = 512;
constexpr uint32_t kBlock = 128;          // the fp8 scale block, along K and along N
/* A RUN-MAP ENTRY THAT OWNS NO EXPERT: the rows above `off[n_expert]` are dropped slots whose
 * output must be zeroed and which no run covers. Declared here because the GEMM reads it and the
 * map that writes it is built further down, beside the launcher it belongs to. */
constexpr uint32_t kMapTail = 0xffffffffu;
constexpr uint32_t kFK    = 128;          // K bytes staged per pass; one scale block exactly

using r4d_fp8::i2v;
using r4d_fp8::f8v;
using r4d_fp8::mma;
using r4d_fp8::acc_zero;
using r4d_fp8::acc_row;
using r4d_fp8::frag_c;
using r4d_fp8::frag;
using r4d_fp8::bscale;
using r4d_fp8::Stage8;
using r4d_fp8::stage8_load;
using r4d_fp8::stage8_store;
using r4d_fp8::lds_pitch;
using r4d_fp8::r4d_f8_ntu4;

#define R4D_MOE_CHK() do { hipError_t e_ = hipGetLastError();                       \
                           if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

/* An entry of a weight-table operand. `tab` is the pointer array when `stride0` is 0, and the
 * base of a stacked plane otherwise. Resolved on the device rather than the host because the
 * table lives in device memory -- reading it host-side would be a synchronisation a step cannot
 * afford, and the mover may have rewritten it since the last one. */
__device__ __forceinline__ const uint8_t* wtab_at(const void* const* tab, size_t stride0,
                                                  uint32_t e) {
    if (stride0 == 0) return (const uint8_t*)tab[e];
    return (const uint8_t*)tab + (size_t)e * stride0;
}

/* ONE SORTABLE INTEGER FOR (value, expert id), WHICH IS WHAT TAKES THE BRANCHES OUT OF THE
 * SELECTION -- and it is free only because the logits are BF16.
 *
 * The selection wants "largest value, ties to the lower id". Written as a comparison that is what
 * it says -- `v > bv || (v == bv && id < bid)` over two registers -- and gfx12 compiles it, plus
 * the `continue` that skips an already-chosen entry, into exec-mask branches: 40 s_and_saveexec,
 * 80 s_cbranch_execz and 306 s_wait_alu in a 2137-instruction kernel that runs on ONE workgroup.
 * On four waves of one CU there is nothing to hide any of that behind.
 *
 * A bf16 has sixteen bits, so the ORDER-PRESERVING map of the value fits in the top half of a
 * u32 and the id fits under it. The map is the standard one -- flip the sign bit for a positive
 * float, invert every bit for a negative -- and it is monotone across the whole range including
 * -inf. `~id` in the low half makes a LOWER id compare HIGHER at equal value, which is the tie
 * rule exactly. The selection is then `v_max_u32`, the wave reduction carries one register
 * instead of two (half the ds_bpermute), and an entry is removed by setting its key to zero.
 *
 * IT IS THE SAME ANSWER, and the two places it would not be are handled rather than assumed:
 *
 *   - NEGATIVE ZERO. -0.0 == +0.0 in float, so the comparison form breaks that tie on the id; as
 *     bit patterns they differ and the key would not. Folded to +0.0 when the key is built.
 *   - NaN. A NaN loses every float comparison, so the comparison form can never select one; a
 *     positive NaN's bit pattern sorts ABOVE +inf and the key would select it first. Given key
 *     zero, which is the same "never selected" and leaves the all-NaN row hitting the same
 *     `nothing left` exit.
 *
 * `bv` is recovered by inverting the map, which returns the ORIGINAL bf16 pattern and therefore
 * the identical f32 -- the softmax that reads it is untouched.
 *
 * WHAT IT BUYS is not instruction count but the DEPENDENCY CHAIN. Against the two-register
 * comparison the object is 2014 instructions and 78 basic blocks with 30 ds_bpermute and 214
 * s_wait_alu, where that form is 2137, 96, 40 and 306. This kernel runs on ONE workgroup at
 * decode, so a chain with exec-mask branches in it is what it costs.
 *
 * DO NOT CHASE rad-kbench's 6.354e-05 ON THE M=512 ROUTER CASE. The selection there is exact
 * (expert_ids rel_l2 0.000e+00); what differs by about one bf16 ulp is the WEIGHT, and it is not
 * this key form's doing. Every case from M=1 to M=256, decode included, is exact. */
__device__ __forceinline__ uint32_t topk_key(uint16_t raw, uint32_t e) {
    const bool nan = ((raw & 0x7F80u) == 0x7F80u) && (raw & 0x7Fu);
    uint16_t   u   = (raw == 0x8000u) ? (uint16_t)0u : raw;          /* -0.0 folds to +0.0 */
    const uint16_t k = (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
    return nan ? 0u : (((uint32_t)k << 16) | (~e & 0xFFFFu));
}
__device__ __forceinline__ uint16_t topk_key_raw(uint32_t key) {
    const uint16_t k = (uint16_t)(key >> 16);
    return (k & 0x8000u) ? (uint16_t)(k & 0x7FFFu) : (uint16_t)~k;
}

/* THE SELECTION'S WAVE MINIMUM, ON THE DPP AND PERMLANE EXCHANGE r4d_xor_lane provides rather than
 * `__shfl_xor`, which gfx1201 lowers to `ds_bpermute_b32`: an LDS round trip a stage. A row runs
 * this top_k times, five stages each, and nothing else is in flight on the wave to cover it, so
 * those round trips would be most of the selection. The minimum of unsigned integers has one answer
 * whatever order the lanes combine in. */
__device__ __forceinline__ uint32_t wave_min_u32(uint32_t b) {
    uint32_t o;
    o = r4d_xor_lane_u32<16>(b); b = b < o ? b : o;
    o = r4d_xor_lane_u32<8>(b);  b = b < o ? b : o;
    o = r4d_xor_lane_u32<4>(b);  b = b < o ? b : o;
    o = r4d_xor_lane_u32<2>(b);  b = b < o ? b : o;
    o = r4d_xor_lane_u32<1>(b);  b = b < o ? b : o;
    return b;
}

/* ================================================================== router_topk
 *
 * Softmax over all n_expert logits, then the top_k of them by value, then -- if `norm` -- the kept
 * weights renormalised to sum to one.
 *
 * ONE WAVE A ROW, and the selection is k passes of a wave reduction rather than a sort. At
 * n_expert 256 and top_k 8 that is 8 passes over 8 values a lane, which is less work than any sort
 * and, more to the point, is EXACTLY the reference's selection: the largest remaining value, ties
 * broken on the lower expert id. A partial sort with a different tie rule is a different router on
 * exactly the rows where two experts score the same, and those rows are not rare in a trained
 * model's tail.
 *
 * The selection runs on the LOGITS, not on the probabilities: softmax is monotone, so the order is
 * identical and the comparison is exact rather than through an exp. libref says the same. */
/* MAXOWN is how many of the row's logits ONE LANE holds, so n_expert <= MAXOWN * kWave. It is a
 * template parameter and not a constant because those are REGISTERS: a 512-expert model needs 16
 * of them a lane and a 256-expert one needs 8, and paying the larger everywhere would cost
 * occupancy on every model in the tree to serve one. Each entry is one u32 key -- see topk_key --
 * so MAXOWN is bounded by registers and by nothing else. */
/* ONE ROW, ONE WAVE, AND NOTHING OUTSIDE THE WAVE. Every reduction below is a butterfly over the
 * row's own 32 lanes and every store is addressed by `m`, so WHICH wave of WHICH workgroup
 * runs a row cannot change its answer. That is what lets `router_topk_scatter_kernel` walk the
 * same rows from a different grid and still be byte-identical -- it is the same code, not a copy
 * of it, which is the only form of that claim that cannot drift. */
/* WHERE A ROW ALSO HANDS ITS SLOTS TO THE SORT, when it runs inside the fused kernel: each slot's
 * expert relative to the slice, or -1 when it went to another rank, into `e`. The standalone
 * selection passes a null `e` and the branch folds away. */
/* THE PROTECTED EXPERTS OF A SLICE, SORTED LAST. A layer may keep a few of its experts at a higher
 * precision than the rest -- the ones whose output dwarfs the others' -- and one grouped GEMM takes
 * one weight format. So the sort places those experts' runs after every other expert's, in the
 * order listed, and two GEMMs serve the two segments: the regular experts are [0, n - p) and the
 * protected ones [n - p, n) of the offsets. Local indices, strictly ascending; p <= kMaxProtect.
 * With none listed the place of an expert is the expert, which is every layer without them. */
constexpr uint32_t kMaxProtect = 8;
struct Protect {
    uint32_t n;
    uint16_t e[kMaxProtect];
};

/* Local expert `e`'s place in the sort: the protected ones last, the rest closed up in order. */
__device__ __forceinline__ int32_t sort_place(int32_t e, uint32_t n_local, const Protect& p) {
    int32_t below = 0;
    for (uint32_t j = 0; j < p.n; ++j) {
        if ((int32_t)p.e[j] == e) return (int32_t)(n_local - p.n + j);
        below += (int32_t)p.e[j] < e;
    }
    return e - below;
}

struct SlotSink {
    int32_t* e       = nullptr;
    int32_t  base    = 0;
    uint32_t n_local = 0;
    Protect  prot{};
    int32_t* s_h     = nullptr;
    uint32_t s_0     = 0;
    uint32_t s_1     = 0;
};

template <uint32_t MAXOWN>
__device__ __forceinline__ void router_topk_row(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    uint32_t m, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm, uint32_t lane,
    SlotSink snk) {

    const uint16_t* row = lg + (size_t)m * lg_ld;
    /* Up to MAXOWN experts a lane: n_expert <= MAXOWN * kWave is a launch precondition, checked
     * on the host, which is also what picks the instantiation.
     *
     * THE TRIP COUNT IS MAXOWN AND NOT `n_own`, AND THAT IS THE WHOLE COST OF THIS KERNEL.
     *
     * `for (e = lane; e < n_expert; e += kWave) v[n_own++] = ...` makes `n_own` a value the
     * compiler derives from `lane`, so every later `v[i]` is a register index it must assume is
     * DIVERGENT -- and a divergent VGPR index on gfx12 is not an addressing mode, it is a
     * waterfall: `v_movrels_b32` inside an exec-masked loop that re-runs for each distinct index
     * in the wave. The top-k loop reads `v[i]` top_k * MAXOWN times, so written that way the kernel
     * disassembles to four such loops and takes tens of microseconds to rank 512 numbers -- tens of
     * thousands of cycles, on one wave, for about a kilobyte of input.
     *
     * Unrolled to the compile-time bound every index is a constant, the array stays in registers
     * and there are no waterfalls. Slots past `n_expert` hold -INFINITY and start out marked in
     * `taken`, which is what keeps the result BIT-IDENTICAL: fmaxf against -inf is the identity,
     * __expf(-inf - mx) is exactly zero and lands at the tail of the same ascending sum, and the
     * selection loop skips a marked slot exactly as it skips a chosen one. At this model's 512
     * experts every lane owns all 16 anyway, so nothing is even masked. */
    /* ALL MAXOWN LOADS ARE ISSUED BEFORE ANY IS WAITED ON. Guarded as `if (ok) raw = row[e]`, each
     * load sits in its own exec-masked branch with its use straight after it, so the compiler
     * waits for every one before issuing the next: MAXOWN dependent round trips to L2, most of a
     * row's time. So a slot past `n_expert` reads the row's first logit instead -- always in
     * bounds -- and the select below discards it. */
    uint16_t raw[MAXOWN];
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kWave;
        raw[i] = row[e < n_expert ? e : 0u];
    }
    float    v[MAXOWN];
    uint32_t key[MAXOWN];    /* (order-preserving value, ~id); 0 = absent or already chosen */
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t e  = lane + i * kWave;
        const bool     ok = e < n_expert;
        v[i]   = ok ? bf16_to_f32(raw[i]) : -INFINITY;
        key[i] = topk_key(raw[i], e) & (0u - (uint32_t)ok);
    }

    /* R4-05: Round 0 of integer top-k finds the row maximum, eliminating the separate float max. */
    uint32_t P     = 0xFFFFFFFFu;     /* round 0 may take any key */
    uint32_t mykey = 0u;
    uint32_t nsel  = 0;

    uint32_t d0 = 0xFFFFFFFFu;
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) {
        const uint32_t t = P - key[i];
        d0 = d0 < t ? d0 : t;
    }
    const uint32_t best0 = P - wave_min_u32(d0);
    float mx;
    if ((best0 & 0xFFFFu) == 0u) {
        mx = -INFINITY;
    } else {
        if (lane == 0) mykey = best0;
        P = best0 - 1u;
        nsel = 1;
        const uint16_t raw_mx = topk_key_raw(best0);
        mx = bf16_to_f32(raw_mx);
    }

    float sum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < MAXOWN; ++i) sum += __expf(v[i] - mx);
    sum = r4d_wave_sum(sum);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    if (nsel > 0) {
        for (uint32_t r = 1; r < top_k; ++r) {
            uint32_t d = 0xFFFFFFFFu;
#pragma unroll
            for (uint32_t i = 0; i < MAXOWN; ++i) {
                const uint32_t t = P - key[i];
                d = d < t ? d : t;
            }
            const uint32_t best = P - wave_min_u32(d);
            if ((best & 0xFFFFu) == 0u) break;    /* fewer live entries than top_k: nothing left */
            if (lane == r) mykey = best;
            P = best - 1u;
            ++nsel;
        }
    }
    const float    p  = __expf(bf16_to_f32(topk_key_raw(mykey)) - mx) * inv;
    const uint16_t pb = f32_to_bf16(p);
    float wsum = 0.0f;
    for (uint32_t r = 0; r < nsel; ++r)
        wsum += __builtin_bit_cast(float,
                    __builtin_amdgcn_readlane(__builtin_bit_cast(int, p), (int)r));
    if (lane < nsel) {
        const uint32_t s = m * top_k + lane;
        const bool own = (snk.s_1 == 0) || (s >= snk.s_0 && s < snk.s_1);
        if (own) {
            ids[(size_t)s] = (int32_t)(~mykey & 0xFFFFu);
            w[(size_t)s]   = (norm && wsum > 0.0f)
                           ? f32_to_bf16(bf16_to_f32(pb) * (1.0f / wsum)) : pb;
            if (snk.s_h) {
                const int32_t e = (int32_t)(~mykey & 0xFFFFu) - snk.base;
                if (e >= 0 && e < (int32_t)snk.n_local) {
                    atomicAdd(&snk.s_h[sort_place(e, snk.n_local, snk.prot)], 1);
                }
            }
        }
    }
    if (snk.e && lane < top_k) {
        const int32_t e  = (int32_t)(~mykey & 0xFFFFu) - snk.base;
        const bool    ok = lane < nsel && e >= 0 && e < (int32_t)snk.n_local;
        snk.e[m * top_k + lane] = ok ? sort_place(e, snk.n_local, snk.prot) : -1;
    }
}

template <uint32_t NW, uint32_t MAXOWN>
__global__ __launch_bounds__(NW * kWave) void router_topk_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    uint32_t M, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, int norm) {
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    const uint32_t m = blockIdx.x * NW + wave;
    if (m >= M) return;
    router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane,
                            SlotSink{nullptr, 0, 0, Protect{}});
}

/* ================================================================== moe_scatter
 *
 * The small form: one workgroup does the whole sort. T <= kScatterSmall, which covers every decode
 * and verify step (a verify step at eight sequences and depth 3 is 32 rows at top_k 10, T = 320),
 * and it is one launch against the three the general form needs. The histogram is in LDS, the
 * offsets are an exclusive prefix over the n_expert <= kMaxExpert counts, and each slot finds its
 * own place from its rank inside its expert's run.
 *
 * ON ONE WORKGROUP THE COST IS THE CRITICAL PATH, NOT THE WORK, and it goes to barriers and to
 * serial chains of LDS round trips. Phase by phase, a Hillis-Steele scan costs 0.72 us (sixteen
 * barriers), prefixing a per-wave cursor table 0.45 us, and placing with one lane a wave walking
 * its chunk of 40 slots -- each step an LDS read, a read-modify-write of its cursor and a store --
 * 2.24 us at T = 320. So:
 *
 *   - THE PREFIX is a DPP scan inside each wave and one exchange of the wave totals through LDS:
 *     a barrier a tile of 32 counts a wave, and n_expert <= 512 is at most two tiles at eight
 *     waves and one at sixteen or more.
 *   - THE PLACEMENT GROUPS FIRST AND ORDERS AFTER. The histogram's atomic hands each slot a ticket
 *     inside its expert's run; tickets come in arrival order, so the slots are first gathered
 *     into their runs in that order, and then each slot's place in the run is the number of the
 *     run's slots with a lower index. That count is as long as the run -- at most one slot a row,
 *     since a row picks an expert once -- where counting over every earlier slot is T/4 LDS
 *     reads long and slower than the walk.
 *
 * STABLE AND BYTE-IDENTICAL: the place in the run is the position among the same expert's slots
 * in ascending i, which is the order the reference produces, whatever order the tickets were
 * handed out in; and an exclusive prefix of non-negative integers
 * has one answer. Every barrier here orders LDS alone, so none of them waits on or invalidates
 * anything in the vector memory path. */
constexpr uint32_t kScatterSmall = 1024;
constexpr uint32_t kScatWaves    = 8;          /* the standalone sort: 256 threads */

/* THE SMALL SORT'S LDS, NAMED ONCE so the standalone kernel and the fused one cannot declare it
 * differently. About 16 KiB. NW is the workgroup's wave count, which sizes the scan's exchange. */
template <uint32_t NW>
struct ScatterLds {
    static constexpr uint32_t kTiles = (kMaxExpert + NW * kWave - 1) / (NW * kWave);
    int32_t e[kScatterSmall];                    /* the slot list: expert in the slice, or -1 */
    int32_t tk[kScatterSmall];                   /* the slot's ticket inside its expert's run */
    int32_t run[kScatterSmall];                  /* slots grouped by expert, tickets' order */
    int32_t cnt[kMaxExpert + 1];
    int32_t off[kMaxExpert + 1];
    int32_t wt[kTiles][NW];                      /* each wave's total, a slot per tile */
};

/* INCLUSIVE PREFIX SUM OVER A WAVE32, on the DPP row shifts: four steps inside each row of
 * sixteen, then the upper row adds the lower row's total. A lane shifted in from outside its row
 * reads zero (bound_ctrl), which is what a prefix sum wants there. */
__device__ __forceinline__ int32_t wave_incl_scan_i32(int32_t x, uint32_t lane) {
    x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xF, 0xF, true);    /* row_shr:1 */
    x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xF, 0xF, true);    /* row_shr:2 */
    x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xF, 0xF, true);    /* row_shr:4 */
    x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xF, 0xF, true);    /* row_shr:8 */
    const int32_t lo = __builtin_amdgcn_readlane(x, 15);
    return lane >= 16u ? x + lo : x;
}

/* The sort, from a filled slot list on: histogram, offsets and counts, placement. The caller has
 * written `sl.e[0, T)` and zeroed `sl.cnt[0, n_expert)`, and a barrier has passed since. Extracted
 * so the standalone sort and the fused kernel run THIS code and not a transcription of it. The
 * workgroup is exactly NW waves. */
template <uint32_t NW>
__device__ __forceinline__ void scatter_small_place(
    int32_t* __restrict__ sorted, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t T, uint32_t n_expert, ScatterLds<NW>& sl) {
    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;

    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e >= 0) sl.tk[i] = atomicAdd(&sl.cnt[e], 1);
    }
    lds_barrier();

    /* Every thread sums all NW wave totals, so `carry` -- and at the end the number of live
     * slots -- is the same in every thread without another read. */
    int32_t carry = 0;
    for (uint32_t b = 0, t = 0; b < n_expert; b += NW * kWave, ++t) {
        const uint32_t e = b + tid;
        const int32_t  v = (e < n_expert) ? sl.cnt[e] : 0;
        const int32_t  x = wave_incl_scan_i32(v, lane);
        if (lane == kWave - 1) sl.wt[t][wave] = x;
        lds_barrier();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const int32_t c = sl.wt[t][k];
            below += (k < wave) ? c : 0;
            tot   += c;
        }
        if (e < n_expert) {
            const int32_t o = carry + below + x - v;      /* inclusive -> exclusive */
            sl.off[e] = o;
            off[e]    = o;
            cnt[e]    = v;
        }
        carry += tot;
    }
    if (tid == 0) off[n_expert] = carry;
    lds_barrier();

    for (uint32_t i = (uint32_t)carry + tid; i < T; i += blockDim.x) sorted[i] = -1;
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e >= 0) sl.run[sl.off[e] + sl.tk[i]] = (int32_t)i;
    }
    lds_barrier();
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = sl.e[i];
        if (e < 0) continue;
        const int32_t o = sl.off[e], n = sl.cnt[e];
        int32_t rank = 0;
        for (int32_t k = 0; k < n; ++k) rank += sl.run[o + k] < (int32_t)i;
        sorted[o + rank] = (int32_t)i;
    }
}

__global__ __launch_bounds__(kScatWaves * kWave) void scatter_small_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ sorted, int32_t* __restrict__ off,
    int32_t* __restrict__ cnt, uint32_t T, uint32_t n_expert, int32_t base, Protect prot) {
    __shared__ ScatterLds<kScatWaves> sl;
    const uint32_t tid = threadIdx.x;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) sl.cnt[e] = 0;
    for (uint32_t i = tid; i < T; i += blockDim.x) {
        const int32_t e = ids[i] - base;
        sl.e[i] = (e >= 0 && e < (int32_t)n_expert) ? sort_place(e, n_expert, prot) : -1;
    }
    lds_barrier();
    scatter_small_place(sorted, off, cnt, T, n_expert, sl);
}

/* ================================================================== router_topk + moe_scatter
 *
 * ONE LAUNCH FOR THE PAIR, and the pair is worth fusing because NEITHER HALF IS WORK. At the
 * decode shape each is a single 256-thread workgroup -- `router_topk` is a wave a row and four rows
 * is one workgroup; the small sort is one workgroup by construction -- moving about a kilobyte,
 * so each launch is its dispatch floor plus a critical path of barriers and round trips. With one
 * of each per MoE layer per rank per step, a separate launch is a real share of the decode step.
 *
 * THE SLOTS REACH THE SORT THROUGH LDS, NOT THROUGH `expert_ids`. Each row's lanes write their
 * slot's expert (relative to the slice, or -1) into the sort's slot list as they finish, so the
 * one barrier between the halves orders LDS only: the sort neither re-reads what the selection
 * just stored nor waits for those stores to land. Nothing comes BEFORE the rows either -- the
 * histogram is zeroed by each thread on its way in and counted after that barrier -- because a
 * barrier at the top waits for the workgroup's last wave to be launched, about half a
 * microsecond that otherwise passes under the first rows' selection.
 *
 * WHY IT IS BYTE-IDENTICAL: the selection is the same device function the standalone kernel
 * calls, wave-local in every reduction and addressing every store by `m`, so running the rows
 * from a different number of waves cannot move a bit; the placement is the same function the
 * standalone sort calls, and it is stable without an atomic ordering anything, so it does not
 * depend on which wave got there first. The one input on which the two paths differ is a row
 * with fewer live experts than top_k -- NaN logits -- whose unselected slots the standalone sort
 * reads from whatever `expert_ids` held and this one records as dropped.
 *
 * A LOOP AND NOT AN EARLY RETURN over the rows -- every thread has to reach the barrier, and
 * `router_topk_kernel`'s `if (m >= M) return;` is exactly what would make that fail on the tail
 * workgroup. The row bound is the caller's: T = M * top_k must fit `kScatterSmall`, which the shim
 * checks and the architecture asks about before it declares the op at all.
 */
/* TWO EXPERT COUNTS AND THEY ARE NOT THE SAME NUMBER. The selection runs over the WHOLE router --
 * it is replicated, and every rank picks from every expert -- while the sort runs over THIS
 * RANK'S SLICE and drops what went elsewhere. On this model at two ranks that is 512 against 254.
 * Passing one count to both is the bug this signature exists to make impossible: the sort would
 * accept ids outside the slice and index its histogram out of range, and the selection would rank
 * half the experts. `n_expert` is the router's, `n_local` the slice's, exactly as the two
 * unfused ops declare them. */
template <uint32_t MAXOWN, uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void router_topk_scatter_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    int32_t* __restrict__ sorted, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t M, uint32_t n_expert, uint32_t n_local, uint32_t top_k, uint32_t lg_ld, int norm,
    int32_t base, Protect prot) {
    __shared__ ScatterLds<NW> sl;
    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
    for (uint32_t e = threadIdx.x; e < n_local; e += blockDim.x) sl.cnt[e] = 0;
    const SlotSink snk{sl.e, base, n_local, prot};
    for (uint32_t m = wave; m < M; m += NW)
        router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane, snk);
    lds_barrier();
    scatter_small_place(sorted, off, cnt, M * top_k, n_local, sl);
}

/* The general form, in three passes, and the three exist to keep it stable without an atomic
 * deciding anything. `bcnt` is [n_tile, n_expert] in scratch. */
constexpr uint32_t kScatterTile = 256;

/* R4-01: Prefill router-top-k and histogram fusion for M >= 128 (T > kScatterSmall).
 * Each CTA processes a histogram tile of 256 slots (the tokens falling into that slot range).
 * It calculates the Top-K selection for its assigned tokens, writes IDs and weights strictly
 * according to its ownership mask to prevent write collisions, and immediately increments the
 * LDS histogram counter for the winning experts. */
template <uint32_t MAXOWN>
__global__ __launch_bounds__(kScatterTile) void router_topk_prefill_hist_kernel(
    const uint16_t* __restrict__ lg, int32_t* __restrict__ ids, uint16_t* __restrict__ w,
    int32_t* __restrict__ bcnt,
    uint32_t M, uint32_t n_expert, uint32_t n_local, uint32_t top_k, uint32_t lg_ld, int norm,
    int32_t base, Protect prot) {
    extern __shared__ int32_t s_h[];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    for (uint32_t e = tid; e < n_local; e += blockDim.x) s_h[e] = 0;
    __syncthreads();

    const uint32_t T = M * top_k;
    const uint32_t s_0 = b * kScatterTile;
    if (s_0 >= T) return;
    const uint32_t s_1 = (s_0 + kScatterTile < T) ? (s_0 + kScatterTile) : T;

    const uint32_t m_start = s_0 / top_k;
    const uint32_t m_end   = (s_1 - 1) / top_k;

    const uint32_t wave = tid / kWave;
    const uint32_t lane = tid % kWave;
    constexpr uint32_t NW = kScatterTile / kWave;

    const SlotSink snk{nullptr, base, n_local, prot, s_h, s_0, s_1};

    for (uint32_t m = m_start + wave; m <= m_end; m += NW) {
        router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane, snk);
    }
    __syncthreads();

    for (uint32_t e = tid; e < n_local; e += blockDim.x)
        bcnt[(size_t)b * n_local + e] = s_h[e];
}

__global__ __launch_bounds__(kScatterTile) void scatter_hist_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ bcnt, uint32_t T, uint32_t n_expert,
    int32_t base, Protect prot) {
    extern __shared__ int32_t s_h[];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) s_h[e] = 0;
    __syncthreads();
    const uint32_t i = b * kScatterTile + tid;
    if (i < T) {
        const int32_t e = ids[i] - base;
        if (e >= 0 && e < (int32_t)n_expert) atomicAdd(&s_h[sort_place(e, n_expert, prot)], 1);
    }
    __syncthreads();
    for (uint32_t e = tid; e < n_expert; e += blockDim.x)
        bcnt[(size_t)b * n_expert + e] = s_h[e];
}

/* One workgroup: totals per expert, the exclusive scan across experts, and the exclusive scan
 * across tiles within each expert -- the last one in place, so `bcnt[b][e]` becomes "how many of
 * expert e are in tiles before b". One thread an expert, which at n_expert 256 is one wave-group
 * and no cross-thread communication at all. */
__global__ __launch_bounds__(256) void scatter_scan_kernel(
    int32_t* __restrict__ bcnt, int32_t* __restrict__ off, int32_t* __restrict__ cnt,
    uint32_t n_tile, uint32_t n_expert) {
    __shared__ int32_t s_wt[2][8];
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid / kWave;
    const uint32_t lane = tid % kWave;

    int32_t v1 = 0;
    if (tid < n_expert) {
        int32_t run = 0;
        for (uint32_t b = 0; b < n_tile; ++b) {
            const int32_t v = bcnt[(size_t)b * n_expert + tid];
            bcnt[(size_t)b * n_expert + tid] = run;
            run += v;
        }
        v1 = run;
        cnt[tid] = run;
    }

    int32_t v2 = 0;
    if (tid + 256 < n_expert) {
        const uint32_t e2 = tid + 256;
        int32_t run = 0;
        for (uint32_t b = 0; b < n_tile; ++b) {
            const int32_t v = bcnt[(size_t)b * n_expert + e2];
            bcnt[(size_t)b * n_expert + e2] = run;
            run += v;
        }
        v2 = run;
        cnt[e2] = run;
    }

    const int32_t x1 = wave_incl_scan_i32(v1, lane);
    const int32_t x2 = wave_incl_scan_i32(v2, lane);

    if (lane == kWave - 1) {
        s_wt[0][wave] = x1;
        s_wt[1][wave] = x2;
    }
    __syncthreads();

    int32_t below0 = 0, tot0 = 0;
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        const int32_t c = s_wt[0][k];
        below0 += (k < wave) ? c : 0;
        tot0   += c;
    }

    int32_t below1 = 0, tot1 = 0;
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k) {
        const int32_t c = s_wt[1][k];
        below1 += (k < wave) ? c : 0;
        tot1   += c;
    }

    if (tid < n_expert) {
        off[tid] = below0 + x1 - v1;
    }
    if (tid + 256 < n_expert) {
        off[tid + 256] = tot0 + below1 + x2 - v2;
    }
    if (tid == 0) {
        off[n_expert] = tot0 + tot1;
    }
}

__global__ __launch_bounds__(kScatterTile) void scatter_place_kernel(
    const int32_t* __restrict__ ids, const int32_t* __restrict__ bcnt,
    const int32_t* __restrict__ off, int32_t* __restrict__ sorted,
    uint32_t T, uint32_t n_expert, int32_t base, Protect prot) {
    constexpr uint32_t NW = kScatterTile / kWave;  /* 8 waves */
    __shared__ uint32_t s_ballot[NW][10];
    const uint32_t tid = threadIdx.x, b = blockIdx.x;
    const uint32_t i = b * kScatterTile + tid;
    const uint32_t live = (uint32_t)off[n_expert];
    if (i >= live && i < T) sorted[i] = -1;
    /* THE PLACE IN THE SORT throughout -- the local expert index, with any protected experts moved
     * last: s_id, the range test and the rank scan all work in the indices the offsets and the
     * weight tables are built from. */
    const int32_t  l = (i < T) ? ids[i] - base : -1;
    const int32_t  e = (l >= 0 && l < (int32_t)n_expert) ? sort_place(l, n_expert, prot) : -1;
    const bool valid = (e >= 0);

    const uint32_t wave = tid / kWave;
    const uint32_t lane = tid % kWave;

    uint32_t b_planes[9];
#pragma unroll
    for (int bit = 0; bit < 9; ++bit) {
        b_planes[bit] = __builtin_amdgcn_ballot_w32(valid && ((e >> bit) & 1));
    }
    const uint32_t bv = __builtin_amdgcn_ballot_w32(valid);

    if (lane < 9) {
        s_ballot[wave][lane] = b_planes[lane];
    } else if (lane == 9) {
        s_ballot[wave][9] = bv;
    }
    __syncthreads();

    if (!valid) return;

    int32_t rank = 0;
#pragma unroll
    for (uint32_t w = 0; w < 8; ++w) {
        if (w < wave) {
            uint32_t m = s_ballot[w][9];
#pragma unroll
            for (int bit = 0; bit < 9; ++bit) {
                m &= ((e >> bit) & 1) ? s_ballot[w][bit] : ~s_ballot[w][bit];
            }
            rank += __builtin_popcount(m);
        }
    }
    {
        uint32_t m = bv & ((1u << lane) - 1u);
#pragma unroll
        for (int bit = 0; bit < 9; ++bit) {
            m &= ((e >> bit) & 1) ? b_planes[bit] : ~b_planes[bit];
        }
        rank += __builtin_popcount(m);
    }

    sorted[off[e] + bcnt[(size_t)b * n_expert + e] + rank] = (int32_t)i;
}


/* ================================================================== the 4-bit expert weight
 *
 * SIXTEEN PACKED BYTES TO THE THIRTY-TWO E4M3 BYTES OF A Stage8, and nothing downstream of it
 * changes: the padded LDS pitch, frag(), the WMMA and the store are the fp8 kernel's, untouched.
 * r4d_layout.cpp's moe w4 section states the stored plane; this is the decode it was chosen for.
 *
 * Byte b of a packed dword holds element b in its low nibble and element b+4 in its high one, so
 * the two masks below produce elements 0..3 and 4..7 already in order and nothing has to be
 * interleaved afterwards. It is the same argument the dense w4a8 kernel's header makes for the
 * same pair of masks.
 *
 * Fifteen VALU per eight weights. At every shape this kernel runs it is DRAM-bound by better than
 * an order of magnitude -- a prefill layer reads about 3 GiB of expert weight for 134 GFLOP, which
 * is 25 FLOP a byte against a part that wants 400 -- so the dequant is free and the halved weight
 * stream is the whole of the win.
 */
__device__ __forceinline__ uint32_t w4_e4m3(uint32_t nib, const uint32_t (&tab)[4]) {
    /* SIXTEEN LEVELS OUT OF AN EIGHT-BYTE PERMUTE, which is the whole trick. V_PERM_B32 selects
     * from a 64-bit source, so one table holds eight entries and no more -- and a nibble needs
     * sixteen. Two tables and a selector that CHOOSES BETWEEN THEM gets there:
     *
     *   m   = nib & 7                      the index within a half, the same for both
     *   lo  = perm(tab[1], tab[0], m)      E4M3 of entries 0..7   (codes 0..7)
     *   hi  = perm(tab[3], tab[2], m)      E4M3 of entries 8..15  (codes 8..15)
     *   sel = 0x03020100 + ((nib & 8) >> 1)   byte i picks lo[i], or hi[i] four bytes further on
     *   out = perm(hi, lo, sel)
     *
     * The add cannot carry: every selector byte is 0..3 before it and 4..7 after. Fifteen VALU per
     * eight weights, and the same fifteen whatever the table holds -- on a kernel that is
     * DRAM-bound by better than an order of magnitude.
     *
     * THE TABLE IS AN ARGUMENT, one of two (r4d_args.h): the integers -8..7 of a two's complement
     * nibble (w4a8, w4a8h) or the w4nl levels (w4nla8h). Every entry of both is EXACT IN E4M3, so a
     * code expands losslessly and the WMMA's product is entry * activation with no rounding the fp8
     * path did not already have. That is what lets this weight ride the existing
     * f32_16x16x16_fp8_fp8 inner loop untouched: only the STAGING changes. */
    const uint32_t m   = nib & 0x07070707u;
    const uint32_t lo  = __builtin_amdgcn_perm(tab[1], tab[0], m);
    const uint32_t hi  = __builtin_amdgcn_perm(tab[3], tab[2], m);
    const uint32_t sel = 0x03020100u + ((nib & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, sel);
}

/* KM01: w4nl levels are 8 positive magnitudes in tab[0..1] and their exact negatives with
 * the E4M3 sign bit (0x80) set in tab[2..3]. We unpack the magnitude with one permute out of
 * tab[1], tab[0], and set the sign bit directly from nibble bit 3 with ((nib & 0x08080808) << 4).
 * Eliminates 2 permutes, 1 add, and 1 shift per 4 weights (16 permutes per 32 weights). */
__device__ __forceinline__ uint32_t w4nl_e4m3(uint32_t nib, const uint32_t (&tab)[4]) {
    const uint32_t m  = nib & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(tab[1], tab[0], m);
    return lo | ((nib & 0x08080808u) << 4);
}

__device__ __forceinline__ Stage8 w4nl_expand(uint4 p, const uint32_t (&tab)[4]) {
    const uint32_t d[4] = { p.x, p.y, p.z, p.w };
    uint32_t o[8];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        o[2 * i]     = w4nl_e4m3(d[i] & 0x0F0F0F0Fu, tab);
        o[2 * i + 1] = w4nl_e4m3((d[i] >> 4) & 0x0F0F0F0Fu, tab);
    }
    Stage8 r;
    r.v = make_uint4(o[0], o[1], o[2], o[3]);
    r.w = make_uint4(o[4], o[5], o[6], o[7]);
    return r;
}

__device__ __forceinline__ Stage8 w4_expand(uint4 p, const uint32_t (&tab)[4]) {
    const uint32_t d[4] = { p.x, p.y, p.z, p.w };
    uint32_t o[8];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        o[2 * i]     = w4_e4m3(d[i] & 0x0F0F0F0Fu, tab);
        o[2 * i + 1] = w4_e4m3((d[i] >> 4) & 0x0F0F0F0Fu, tab);
    }
    Stage8 r;
    r.v = make_uint4(o[0], o[1], o[2], o[3]);
    r.w = make_uint4(o[4], o[5], o[6], o[7]);
    return r;
}

/* w5nl64a8h's 32 codes: the magnitudes as w4_expand expands any nibble -- `tab` is the sixteen
 * magnitudes -- and then each byte's sign bit from `sg`. Output dword m holds elements 4m..4m+3,
 * and the relayout stores element 4m+b's sign at bit 7-m of byte b of `sg` (r4d_layout.cpp), so a
 * shift by m puts dword m's four signs at bit 7 of their bytes -- E4M3's sign -- and the mask keeps
 * just those: two VALU per four weights, one where m is 0. */
__device__ __forceinline__ Stage8 w5_expand(uint4 p, uint32_t sg, const uint32_t (&tab)[4]) {
    Stage8 r = w4_expand(p, tab);
    uint32_t o[8] = { r.v.x, r.v.y, r.v.z, r.v.w, r.w.x, r.w.y, r.w.z, r.w.w };
#pragma unroll
    for (uint32_t m = 0; m < 8; ++m)
        o[m] |= (sg << m) & 0x80808080u;
    r.v = make_uint4(o[0], o[1], o[2], o[3]);
    r.w = make_uint4(o[4], o[5], o[6], o[7]);
    return r;
}

/* The packed half of a Stage8's worth of K, or zeros for a row this block does not own. A dead
 * lane's zeros expand to entry 0, which is E4M3 zero under -8..7 and 1.125 under w4nl; either way
 * the column contributes nothing, because the staging gives a column this block does not own a
 * weight scale of zero. */
template <bool NT = false>
__device__ __forceinline__ uint4 w4_load(const uint8_t* __restrict__ src, bool live) {
    if (!live) return make_uint4(0, 0, 0, 0);
    if constexpr (NT) {
        return __builtin_bit_cast(uint4,
                   __builtin_nontemporal_load((const r4d_f8_ntu4*)src));
    } else {
        return *(const uint4*)src;
    }
}

/* ================================================================== moe_gemm_q
 *
 * y[i, :] = a[row(i), :] @ w[e]^T, in SORTED order, both operands E4M3 with a scale per 128 along
 * K (and, for the weight, per 128 along N).
 *
 * `a_sorted` PICKS row(i), and it is an argument because the two grouped GEMMs of one routed block
 * disagree about it. The gate_up projection reads the block INPUT, one row per token, so row(i) is
 * `sorted[i] / top_k`. The down projection reads gate_up's OUTPUT, which is already one row per
 * (token, slot) in this very order, so row(i) is i -- gathering that a second time reads another
 * slot's activation, which is a model that still runs and answers wrongly. Only the resolution of
 * `s_tok` below changes; everything downstream reads a row index and does not care where it came
 * from.
 *
 * MT   M tiles a block, 16 rows each; the banded knob.
 * NTL  N tiles a block, 16 columns each.
 * WGM  waves along M; WGN waves along N. WGM * WGN waves a block, a wave owning MT/WGM M tiles and
 *      NTL/WGN N tiles.
 *
 * SK   skip, per wave, the M tiles that start at or past the end of the block's run.
 *
 * ---- THE PREFILL TILE IS 32 ROWS BY 64 COLUMNS, AND A WAVE SKIPS WHAT ITS RUN DOES NOT REACH ----
 *
 * (And by 128 columns over eight waves where the weight's parts allow it: see kPrefillCols.)
 *
 * At a prefill chunk a block serves one run (AL, below), and a run is `T / n_expert` rows on
 * average -- forty at Qwen3.8-Flash-Next's 2048 chunk, from none to a few hundred. A tile of 128
 * rows by 64 columns, eight waves of one 16-row M tile each, is therefore mostly rows past the end
 * of the run, and every wave runs its WMMA over them anyway: dead rows are masked at the load and
 * the store but not at the arithmetic. Reshaping that tile -- WGN 2, FN 128, or FM 64 at WGN 2 --
 * only moves the staging depth, and each is slower; none of them removes the dead work.
 *
 * Two things do, and both are bit-identical -- every live element is still one wave's
 * accumulation over the same K in the same order:
 *
 *   SK     a wave skips the fragment reads, the WMMA and the scale fold of any M tile that starts
 *          at or past the end of its run. The test is wave-uniform, and the wave still stages and
 *          meets every barrier. On the 128-row tile alone it is -18% on both projections.
 *   FM 32  the tile is what a run fills. A longer run is two or more blocks, and the second
 *          staging of its weight is an L2 hit.
 *
 * r4d_selftest --perfmoe, rank 0's tp2 shape over a skewed routing, us a call (gate_up / down):
 *
 *      rows a run (mean)          20             40             64 (256 experts, top 8)
 *      128 x 64, 8 waves          3138 / 1915    3346 / 2081    1920 / 1198
 *      32 x 64, 4 waves, SK       1067 / 667     1552 / 987     1115 / 717
 *
 * At 128 columns a 16-row tile is slower again (1908 / 1193 at forty) and a 64-row one slower
 * still (1843 / 1206). A 128-column tile at 32 rows is within ~3% at forty rows and slower at
 * twenty, so the tile keeps 64 columns -- and the op's "N a multiple of 64" with them.
 *
 * AT PREFILL IT IS NOT THE LINK. Across a prefill, each layer's GEMM time is uncorrelated with the
 * number of that layer's experts that are pooled rather than resident (61 to 78 of 512): the
 * pooled experts' reads overlap the resident ones', and the kernel's own time is the cost.
 *
 * THE DEPLOYMENT RUNS A 2048 CHUNK: a run fills more of its tile, and the whole expert plane is
 * swept once for four times the tokens of a 512 chunk. End to end on prompts of 8k and 24k, with
 * the 128-row tile:
 *
 *      chunk asked for      512      1024     2048     4096
 *      chunk actually run   512      1024     2048     2048
 *      tok/s               1750      2200     2706     2620
 *
 * 4096 IS NOT A 4096 CHUNK. `plan_chunk` clamps to the checkpoint interval BEFORE the token
 * budget, and the interval is 2048, so asking for 4096 runs 2048 chunks and pays for 4096: the
 * activation arena grows with the budget and is charged to the VRAM budget ahead of the
 * expert/KV division, so the expert slab loses residency for a chunk it never got -- 9458
 * resident units at 512, 9219 at 2048, 8980 at 4096. That is the whole of the 4096 column, and
 * it is why the number falls. AND A REAL 4096 CHUNK DOES NOT HELP EITHER: with
 * --checkpoint-interval 4096 the engine runs 4096-token chunks, and prefill is 2610 tok/s against
 * 2748. The arena takes another 239 resident experts and the extra amortisation does not pay for
 * them.
 *
 * AND IT DOES NOT COST CO-RUNNING DECODERS. A longer chunk holds the device longer per step, so
 * every decode step queued behind one waits longer -- but the prefill those steps are queued
 * behind is a third shorter, and that is the larger term. Four 1500-token decoders with a ~24k
 * prefill landing beside them:
 *
 *      chunk    decoders alone   beside the prefill   the prefill
 *      512          19.12 s           31.86 s          1840 tok/s
 *      2048         19.13 s           29.52 s          2434 tok/s
 *
 * Both halves improve.
 *
 * ES   one RUN a block instead of all of a tile's runs -- see below.
 *
 * ---- ES: WHY A DECODE STEP NEEDS THE RUNS SPREAD OVER BLOCKS -----------------------------------
 *
 * A block owning a contiguous row range and looping over the experts in it is a workable shape at
 * prefill, where a tile holds a handful of runs and there are enough tiles to fill the device. At
 * DECODE it is the whole problem: one token routes to `top_k` experts, so T is about eleven rows
 * -- ONE 16-row tile -- and the grid collapses to (1, N/FN). That is 20 workgroups for a
 * 1280-column gate_up on a 32-CU part, 80 waves against the ~1000 the device holds, and the
 * eleven experts are then walked SERIALLY inside those 20 blocks -- around 89 GB/s on a kernel
 * that is a bandwidth problem.
 *
 * The runs are independent: run `j` reads expert `e(j)`'s weight and writes rows [rs, re), and
 * those row ranges partition the tile. So `blockIdx.x` carries (tile, run) instead of tile alone
 * and each block does one run. The weight traffic is UNCHANGED -- every expert's N slice is still
 * read exactly once -- and so is every accumulation order, so the output is bit-identical; only
 * which workgroup does it moves. A tile has at most FM runs because a run owns at least one row,
 * so FM blocks a tile always suffices and the ones past the run count exit at the loop head.
 *
 * The split is unconditional inside the decode band; r4d_moe_decode_band says where that band
 * ends and why.
 *
 * ---- AT DECODE WHAT BINDS IT IS THE LINK, NOT THIS KERNEL ---------------------------------------
 *
 * This instantiation is the largest single item in a decode trace -- around a fifth of two-rank
 * device time -- so it is worth saying precisely what it is waiting for, because it is not anything
 * this file can change.
 *
 * A gate/up dispatch reads about 19.7 MB of expert weight. Its FASTEST dispatches run at 576 GB/s,
 * which IS this part's cold-stream peak, so the best case is already perfect and there is no
 * compute or bandwidth inefficiency left to find here. The MEAN is 270 GB/s, and the entire gap is
 * dispatches that routed to an expert the slab could not hold, whose shard is then streamed over
 * the link inside the kernel at ~26 GB/s: one pooled expert is 1.64 MB and about 63 us, which is
 * why the median sits near the resident cost and the upper quantiles near one and two misses.
 *
 * SO THE LEVER IS THE MISS RATE AND NOTHING ELSE. Residency is 82% of 256 experts a layer and it
 * is not raisable here: --expert-vs-cache-ratio already gives the experts everything the KV pool
 * does not need, and the KV pool is sized by --max-num-seqs and --max-model-len, which are the
 * deployment's capability rather than a tuning knob. If every routed expert were resident this
 * kernel would run at its resident rate and its share of device time would roughly halve. That is
 * the size of the prize for anything that raises capacity or sharpens the ranking, and it is why
 * the kernel's own arithmetic is not the lever.
 */
/* ONE CLASS OF EXPERTS: the even-numbered or the odd-numbered ones, whose weight tables are
 * separate operands because every entry of one table has one geometry. Expert e is entry e / 2 of
 * class e % 2. The classes differ when a rank holds an uneven slice of each expert's width -- three
 * 128-column blocks of one parity and two of the other -- and are the same geometry otherwise.
 * `moe_gemm_q_mod4` has four, by the expert's number mod 4: expert e is entry e / 4 of class e % 4,
 * which is what lets four ranks take the extra block of a five-block expert in turn.
 *
 * `n` is the weight's rows and `k` its K; `w_ld` the packed byte pitch of a weight row and `s_ld`
 * the scale plane's row pitch in elements. */
struct MoeCls {
    const void* const* w;
    const void* const* s;
    uint32_t w_s0, s_s0;   /* bytes an entry of a stacked plane; 0 marks a pointer table */
    uint32_t n, k, w_ld, s_ld;
};
/* W4: the code table, r4d_args.h's kW4IntPerm, kW4nlPerm or kW5nlMagPerm -- one for every class,
 * since every class is one format.
 *
 * FOUR CLASSES AND THE TABLE MUST FIT RAD_KPARAM_BLOB (256 bytes) BESIDE EVERYTHING ELSE, because
 * a kernel's arguments are packed into one descriptor whether it is launched or described: four
 * classes at 40 bytes, the five operand pointers and the map, eight words and this -- 160 + 48 +
 * 32 + 16, exactly 256, with no padding since a class is 8-aligned. That is why a class carries
 * 32-bit entry strides and no table of its own, and why `mode` packs three small values into one
 * word. The sizes are asserted here and the whole list after the kernel. */
struct MoeTab { uint32_t v[4]; };
static_assert(sizeof(MoeCls) == 40 && alignof(MoeCls) == 8,
              "a class is two pointers and six words: the kernel's argument budget counts 40");
static_assert(sizeof(MoeTab) == 16, "the code table is four words");

/* W4 IS THE WEIGHT'S FORM: 0 block E4M3, 1 four-bit with one bf16 scale a (row, 128 of K), 2 the
 * same codes with one E4M3 scale a (row, 64 of K) times the fixed kW4G64Scale (w4nl64a8h), 3 the
 * same at one a (row, 32 of K) (w4nl32a8h), 4 five-bit codes -- 2's magnitudes and scales, and a
 * sign bit a code from the plane after a row's nibbles (w5nl64a8h). At 2, 3 and 4 a staged
 * 128-wide tile holds two or four scale groups, each summed in its own accumulator; a tile's E4M3
 * scales are one sixteen- or thirty-two-bit load. */
template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, int W4 = 0,
          bool ES = false, bool AL = false, bool SK = false>
__device__ __forceinline__ void moe_gemm_q_body(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y, const uint32_t* __restrict__ map,
    uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab) {
    /* `mode`: bit 0 `a_sorted`, bits 1-2 the class shift (1 for two classes, 2 for four), bits
     * 8 up `parts`. */
    const uint32_t a_sorted = mode & 1u, cshift = (mode >> 1) & 3u, parts = mode >> 8;
    constexpr uint32_t FM = MT * 16, FN = NTL * 16;
    constexpr uint32_t NTHR = WGM * WGN * kWave;
    constexpr uint32_t WM = MT / WGM, WN = NTL / WGN;
    static_assert(MT % WGM == 0 && NTL % WGN == 0, "the wave grid must divide the tile");
    static_assert(kBlock % FN == 0, "a block's N rows must sit in one 128-row weight scale row");
    /* Scale groups a staged tile: one, or at W4 2, 3 and 4 the 64- or 32-column ones. */
    constexpr uint32_t kSub = W4 == 3 ? 4u : (W4 == 2 || W4 == 4) ? 2u : 1u;
    static_assert(kFK % (kSub * r4d_fp8::kK) == 0, "a scale group is whole WMMA K steps");
    static_assert(FM + FN <= NTHR, "a thread stages each row's and each column's scale: see "
                                   "THE SCALES RIDE WITH THE TILE");

    __shared__ uint8_t s_a[FM * lds_pitch<kFK>()];
    __shared__ uint8_t s_b[FN * lds_pitch<kFK>()];
    __shared__ int32_t s_tok[FM];
    /* The staged K block's scales: one an activation row, and one a weight column (w4) or one for
     * the whole block (fp8). See THE SCALES RIDE WITH THE TILE, at the K loop. */
    __shared__ float   s_as[FM];
    __shared__ float   s_ws[kSub * FN];

    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    const uint32_t wgm = wave / WGN, wgn = wave % WGN;
    /* With ES the x axis is (tile, run) and the run index is the minor half, so a tile's blocks
     * stay adjacent and land on neighbouring CUs -- they share the activation rows. */
    const uint32_t jrun = ES ? (blockIdx.x % FM) : 0u;
    /* UNDER AL THE BLOCK IS NAMED BY THE MAP, not by arithmetic on blockIdx: entry i is
     * (expert, first row) and the block serves that run and no other. The grid is the upper bound
     * `n_expert + T/FM`, so the blocks past the real count exit here. */
    uint32_t r0, rN, e_al = 0;
    bool tail = false;
    if constexpr (AL) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        e_al = ent[0];
        r0   = ent[1];
        tail = (e_al == kMapTail);
        const uint32_t lim = tail ? T : (uint32_t)off[e_al + 1];
        rN = (r0 + FM < lim) ? (r0 + FM) : lim;
    } else {
        r0 = (ES ? (blockIdx.x / FM) : blockIdx.x) * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    }
    const uint32_t n0 = blockIdx.y * FN;

    /* The token each of the block's sorted rows reads its activation from, resolved once. -1 marks
     * a dropped slot, whose output row is zero. */
    for (uint32_t r = tid; r < FM; r += NTHR) {
        const uint32_t i = r0 + r;
        int32_t t = -1;
        if (i < rN) {
            const int32_t f = sorted[i];
            t = f < 0 ? -1 : (a_sorted ? (int32_t)i : (top_k == 10 ? (int32_t)(((uint64_t)f * 0xCCCCCCCDULL) >> 35) : f / (int32_t)top_k));
        }
        s_tok[r] = t;
    }
    __syncthreads();

    const uint32_t live = (uint32_t)off[n_expert];
    /* Rows past the live count are dropped slots: zeroed here, and then the run loop below has
     * nothing left to do for them. No run owns them, so under ES exactly one of the tile's blocks
     * must -- every block writing the same zeros would be a benign race and is still a race. */
    if (!ES || jrun == 0)
    for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
        for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
            Y[(size_t)r * y_ld + n0 + c] = 0;

    /* A TAIL BLOCK HAS NOW DONE ALL ITS WORK: it exists to zero rows no expert owns and holds no
     * run for the loop below. Block-uniform, so no barrier is skipped by some threads only. */
    if constexpr (AL) { if (tail) return; }

    /* THE EXPERT OF THE BLOCK'S FIRST RUN. The row-tile grid finds the expert of its first row by
     * binary search over the offsets and walks the runs from there, which the loop below does.
     *
     * UNDER ES THAT WALK WOULD BE THE KERNEL. A decode step routes a few dozen slots over hundreds
     * of experts, so between two runs sit a dozen experts with no rows, and the walk to run `jrun`
     * steps over every one of them a dependent load at a time -- a block serving a tile's last run
     * walks most of the tile's expert range before it does anything, and a block past the tile's
     * last run walks all of it to find it has nothing to do. At hundreds of experts a rank and a
     * down projection a few K blocks long, that walk is a large part of the kernel.
     *
     * So the block finds its tile's runs in one parallel pass: every thread tests one expert for
     * "has rows, and they meet this tile", a ballot and a count of the waves before it place the
     * hits in expert order -- which is row order -- and run `jrun` is entry `jrun`. Chunks of
     * experts wholly before or after the tile are skipped on one broadcast load, and the pass
     * stops once it has found the block's run. The same rows reach the same expert: bit-identical.
     */
    uint32_t e = 0;
    if constexpr (AL) {
        e = tail ? 0u : e_al;          /* the map already named it; a tail block owns no expert */
    } else if constexpr (ES) {
        if (T <= 16) {
            __shared__ uint32_t s_es_e;
            __shared__ uint32_t s_es_nrun;
            if (wave == 0) {
                uint32_t e_lane = 0;
                const uint32_t r = r0 + lane;
                const bool valid = (lane < FM) && (r < rN) && (r < live);
                if (valid) {
                    uint32_t lo = 0, hi = n_expert;
                    while (lo + 1 < hi) {
                        const uint32_t mid = (lo + hi) >> 1;
                        if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid;
                    }
                    e_lane = lo;
                }
                const uint32_t prev_e = __shfl_up(e_lane, 1, kWave);
                const bool is_new_run = valid && (lane == 0 || e_lane != prev_e);
                const unsigned long long mk = __ballot(is_new_run);
                const uint32_t nrun = (uint32_t)__popcll(mk);
                const uint32_t pos = (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
                if (is_new_run && pos == jrun) s_es_e = e_lane;
                if (lane == 0) s_es_nrun = nrun;
            }
            __syncthreads();
            if (jrun >= s_es_nrun) return;
            e = s_es_e;
        } else {
            __shared__ int32_t  s_run[FM];
            __shared__ uint32_t s_wc[NTHR / kWave];
            uint32_t nrun = 0;
            for (uint32_t c = 0; c < n_expert && nrun <= jrun; c += NTHR) {
                const uint32_t ce = c + NTHR < n_expert ? c + NTHR : n_expert;
                if ((uint32_t)off[ce] <= r0) continue;          /* every expert here ends before it */
                if ((uint32_t)off[c] >= rN) break;              /* and every one from here after it */
                const uint32_t x = c + tid;
                bool p = false;
                if (x < ce) {
                    const uint32_t a = (uint32_t)off[x], b = (uint32_t)off[x + 1];
                    p = a < b && a < rN && b > r0;
                }
                const unsigned long long mk = __ballot(p);
                if (lane == 0) s_wc[wave] = (uint32_t)__popcll(mk);
                __syncthreads();
                uint32_t before = nrun, tot = 0;
                for (uint32_t w = 0; w < NTHR / kWave; ++w) {
                    if (w < wave) before += s_wc[w];
                    tot += s_wc[w];
                }
                const uint32_t pos = before + (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
                if (p && pos < FM) s_run[pos] = (int32_t)x;
                __syncthreads();
                nrun += tot;
            }
            if (jrun >= nrun) return;       /* the tile has fewer runs than this block's index */
            e = (uint32_t)s_run[jrun];
        }
    } else {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
        }
        e = lo;
    }

    /* THE OUTPUT IS `parts` STACKED PARTS OF N / parts COLUMNS, and so is every weight, each part
     * of it `n / parts` rows. A weight part narrower than the output's fills the part's leading
     * columns and the rest are ZERO: that is a rank's two-block slice of a gate/up pair written
     * into the three-block layout the gated quantiser reads, and the zeros are what make the
     * quantiser's output there zero rather than whatever the arena held. With one part and one
     * geometry this is the whole weight at column n0. */
    const uint32_t fy = N / parts;
    const uint32_t part = n0 / fy, j0 = n0 - part * fy;

    constexpr uint32_t kSeg = 32;                     /* bytes a thread stages, contiguous */
    constexpr uint32_t kTPR = kFK / kSeg;             /* threads covering one staged row */
    constexpr uint32_t kRPP = NTHR / kTPR;            /* rows one staging pass covers */
    constexpr uint32_t kAP  = (FM + kRPP - 1) / kRPP;
    constexpr uint32_t kBP  = (FN + kRPP - 1) / kRPP;
    const uint32_t sn = tid / kTPR, sk = (tid % kTPR) * kSeg;

    /* Under ES the block's run was found above, so the loop starts at it: its first row in the
     * tile, and the run count already at `jrun`. */
    uint32_t rs = r0, run = 0;
    if constexpr (ES) {
        const uint32_t a = (uint32_t)off[e];
        rs = a > r0 ? a : r0;
        run = jrun;
    }
    while (rs < rN && rs < live) {
        /* This run: rows [rs, re) all belong to expert `e`. The advance is at the TOP so `e` can
         * never be walked past the last expert -- `off[e + 1]` is read only for e < n_expert, and
         * off[n_expert] is the live row count, which is the bound the loop already respects. */
        while (e < n_expert && (uint32_t)off[e + 1] <= rs) ++e;
        if (e >= n_expert) break;
        const uint32_t ehi = (uint32_t)off[e + 1];
        const uint32_t re  = rN < live ? (rN < ehi ? rN : ehi) : (live < ehi ? live : ehi);
        if (re <= rs) break;
        /* Not this block's run: walk past it. Block-uniform, so no barrier is skipped by some
         * threads and not others. */
        if (ES && run != jrun) { rs = re; ++run; continue; }

        /* The expert's class, and where this block's columns sit in its weight. Block-uniform: a
         * pad tile zeroes the run's rows and moves on without touching LDS or a barrier. `cshift`
         * is 1 for two classes and 2 for four; c_2 and c_3 are never read at two. */
        const uint32_t cls = e & ((1u << cshift) - 1u);
        const MoeCls& C = cls == 0u ? c_even : cls == 1u ? c_odd : cls == 2u ? c_2 : c_3;
        const uint32_t fc = C.n / parts;
        if (j0 >= fc) {
            const uint32_t nr = re - rs;
            for (uint32_t x = tid; x < nr * FN; x += NTHR) {
                const uint32_t c = x % FN;
                if (n0 + c < N) Y[(size_t)(rs + x / FN) * y_ld + n0 + c] = 0;
            }
            rs = re;
            ++run;
            if (ES) break;
            continue;
        }
        const uint32_t wn0 = part * fc + j0;          /* the weight row of the block's column 0 */
        const uint32_t K = C.k, w_ld = C.w_ld, ws_ld = C.s_ld;
        const uint32_t kb_n = wn0 / kBlock;           /* the weight scale's N row for this block */
        if constexpr (W4) (void)kb_n;                 /* w4 scales vary per ROW, not per N block */

        const uint8_t*  Wp  = wtab_at(C.w, C.w_s0, e >> cshift);
        const uint16_t* Wsp = (const uint16_t*)wtab_at(C.s, C.s_s0, e >> cshift);

        /* SK: A WAVE WHOSE M TILES ARE ALL PAST THE RUN has nothing to accumulate. Under AL the
         * run is the block's rows [rs, re) from its first row, so that is a wave-uniform test;
         * such a wave still stages and meets every barrier, and only its fragment reads, its
         * WMMA and its scale fold are skipped. */
        const bool wlive = !SK || (r0 + wgm * WM * 16 < re);
        f8v raw[WM][WN], out[WM][WN];
        /* The other scale groups' accumulators at W4 2 and 3 (`raw` is the first's); one unused
         * element otherwise. Each starts fresh in its group's first WMMA, so none is zeroed. */
        f8v rx[kSub > 1 ? kSub - 1 : 1][kSub > 1 ? WM : 1][kSub > 1 ? WN : 1];
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j) { raw[i][j] = acc_zero(); out[i][j] = acc_zero(); }

        /* THE K LOOP IS SOFTWARE PIPELINED, and without that it spends most of its time waiting.
         *
         * Staged straight through -- issue the loads, wait for them, store to LDS, barrier,
         * compute -- the global latency of every one of K/kFK = 20 iterations is exposed in full,
         * because nothing else in the block is runnable while the loads are outstanding. A decode
         * layer puts about 110 of these blocks on 64 CUs, which is under two a CU: there is no
         * second resident block to cover the stall either. Staged that way the kernel runs at
         * about 245 GB/s on gate_up's 9.0 MB, against the 486 this part reaches on a cold stream,
         * and its K loop disassembles to 432 instructions for 8 WMMA -- so it is not issue-bound
         * and not short of blocks. It is waiting.
         *
         * So the tile is loaded into REGISTERS one iteration ahead. The loads for tile i+1 are
         * issued right after tile i's bytes have been handed to LDS, and the `s_waitcnt` that
         * collects them does not land until the next iteration's LDS store -- one barrier and the
         * whole WMMA block later. Sixteen VGPRs a thread buys that; nothing else changes.
         *
         * BIT-IDENTICAL: the same bytes reach the same LDS addresses in the same order and the
         * accumulation is untouched. Only WHEN the load is issued moves.
         *
         * ONE TILE AHEAD IS THE WHOLE OF IT -- do not go deeper. A two-slot ring is WORSE both ways
         * it can be written: indexing the ring with a running `slot` puts the register arrays in
         * SCRATCH, and unrolling the K loop by the depth to make the index a constant instead takes
         * VGPRs 96 -> 112 and doubles the body. Neither is the reason it cannot win, though. This
         * kernel is not waiting on DRAM at all: at eight concurrent sequences it serves eight times
         * the tokens and several times the distinct experts in the SAME time it takes at one, and
         * its duration barely moves with the number of experts routed. At decode it is
         * critical-path bound -- 400 waves on 256 SIMDs, each stalled about 88% of cycles on its
         * own dependencies -- and the only axis
         * that shortens the 20-step serial K chain is a split-K, which is not bit-identical and so
         * is not a decision this file gets to make. */
        /* THE SCALES RIDE WITH THE TILE. The fold at the end of a K block needs the activation
         * scale of eight rows a lane and the weight scale of a lane's columns; loaded there, from
         * global memory, after the block's WMMA, they would be ten scattered loads a lane, issued
         * behind the matrix work and waited on before the next block could start. They are
         * one value a row and one a column, so the thread that stages a row or a column loads its
         * scale with the tile, one K block ahead, and the fold reads them from LDS. The same
         * values reach the same multiplies: bit-identical. */
        Stage8 apre[kAP];
        std::conditional_t<W4 != 0, uint4, Stage8> bpre[kBP];
        uint32_t spre[W4 == 4 ? kBP : 1];   /* W4 4: the same 32 elements' sign bits */
        float    pas = 0.0f;    /* thread tid < FM: its row's activation scale */
        /* thread FM + c: column c's weight scale (w4), or the block's (fp8); at W4 3 its four */
        std::conditional_t<W4 == 3, uint32_t, uint16_t> pws = 0;
        auto issue = [&](uint32_t cc) {
            const uint32_t kbk = cc / kBlock;
            if (tid < FM) {
                const int32_t tok = s_tok[tid];
                const bool    ok  = tok >= 0 && (r0 + tid) >= rs && (r0 + tid) < re;
                pas = ok ? As[(size_t)tok * as_ld + kbk] : 0.0f;
            } else if (tid < FM + FN) {
                const uint32_t nn = tid - FM;
                if constexpr (W4 == 3)
                    pws = j0 + nn < fc ? ((const uint32_t*)Wsp)[(size_t)(wn0 + nn) * ws_ld + kbk]
                                       : 0u;
                else if constexpr (W4)
                    pws = j0 + nn < fc ? Wsp[(size_t)(wn0 + nn) * ws_ld + kbk] : (uint16_t)0;
                else if (nn == 0)
                    pws = Wsp[(size_t)kb_n * ws_ld + kbk];
            }
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FM) continue;
                const int32_t tok = s_tok[r];
                const bool    ok  = tok >= 0 && (r0 + r) >= rs && (r0 + r) < re;
                apre[p] = stage8_load(Aq + (size_t)(ok ? tok : 0) * a_ld + cc + sk, ok);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FN) continue;
                const bool ok = (j0 + r) < fc;
                /* W4: half the bytes off the same row, expanded on the way in. `w_ld` is the
                 * PACKED byte pitch there, which is why it is not divided here. At W4 4 the row's
                 * sign plane follows its K/2 bytes of nibbles, a bit an element. */
                if constexpr (W4) {
                    const uint8_t* wr = Wp + (size_t)(wn0 + r) * w_ld;
                    bpre[p] = w4_load(wr + ((cc + sk) >> 1), ok);
                    if constexpr (W4 == 4)
                        spre[p] = ok ? *(const uint32_t*)(wr + (K >> 1) + ((cc + sk) >> 3)) : 0u;
                } else
                    bpre[p] = stage8_load(Wp + (size_t)(wn0 + r) * w_ld + cc + sk, ok);
            }
        };
        /* `s_tok` is settled by the barrier above the run loop and no run rewrites it, so priming
         * ahead of the loop's own barrier reads nothing another thread is still writing. */
        issue(0);

        for (uint32_t c0 = 0; c0 < K; c0 += kFK) {
            __syncthreads();                       /* the previous run's reads are done */
#pragma unroll
            for (uint32_t p = 0; p < kAP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FM) continue;
                stage8_store<kFK>(s_a, r, sk, apre[p]);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBP; ++p) {
                const uint32_t r = sn + p * kRPP;
                if (r >= FN) continue;
                if constexpr (W4 == 4)
                    stage8_store<kFK>(s_b, r, sk, w5_expand(bpre[p], spre[p], tab.v));
                else if constexpr (W4)
                    stage8_store<kFK>(s_b, r, sk, w4_expand(bpre[p], tab.v));
                else
                    stage8_store<kFK>(s_b, r, sk, bpre[p]);
            }
            if (tid < FM) s_as[tid] = pas;
            else if (tid < FM + FN) {
                if constexpr (kSub == 2) {
                    const v2f sc = r4d_fp8::e4m3_pk<false>((uint32_t)pws);
                    s_ws[0 * FN + tid - FM] = sc.x * kW4G64Scale;
                    s_ws[1 * FN + tid - FM] = sc.y * kW4G64Scale;
                } else if constexpr (kSub > 1) {
                    /* The tile's scale groups in column order, the first in the low byte. */
#pragma unroll
                    for (uint32_t q = 0; q < kSub; ++q)
                        s_ws[q * FN + tid - FM] =
                            r4d_fp8::e4m3((uint32_t)((pws >> (8 * q)) & 0xFFu)) * kW4G64Scale;
                } else {
                    s_ws[tid - FM] = (W4 || tid == FM) ? bscale(pws) : 0.0f;
                }
            }
            if (c0 + kFK < K) issue(c0 + kFK);
            __syncthreads();

            if (!wlive) continue;
            /* One K step of the tile into the accumulator `acc`; `fresh` starts it from zero
             * rather than from what it holds -- exactly the same sum, and no reset to pay. */
            auto kstep = [&](uint32_t ks, f8v (&acc)[WM][WN], bool fresh = false) {
                i2v af[WM], bf[WN];
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
                    af[i] = frag<kFK>(s_a, (wgm * WM + i) * 16, ks, lane);
                }
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j)
                    bf[j] = frag<kFK>(s_b, (wgn * WN + j) * 16, ks, lane);
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j)
                        acc[i][j] = mma(af[i], bf[j], fresh ? acc_zero() : acc[i][j]);
                }
            };
            /* AT W4 2 AND 3 EACH SCALE GROUP OF THE TILE HAS ITS OWN ACCUMULATOR, because each
             * has its own scale: folding a group as it ends would make the next group's WMMA wait
             * on its results, and at a decode step -- a serial chain of tiles -- that wait is most
             * of what the finer groups would cost. They fold once, at the end of the tile, and
             * each group's first step starts its accumulator, so the fold resets none. */
            if constexpr (kSub > 1) {
                constexpr uint32_t kG = kFK / kSub;
#pragma unroll
                for (uint32_t ks = 0; ks < kG; ks += r4d_fp8::kK) kstep(ks, raw, ks == 0);
#pragma unroll
                for (uint32_t q = 1; q < kSub; ++q)
#pragma unroll
                    for (uint32_t ks = q * kG; ks < (q + 1) * kG; ks += r4d_fp8::kK)
                        kstep(ks, rx[q - 1], ks == q * kG);
            } else {
#pragma unroll
                for (uint32_t ks = 0; ks < kFK; ks += r4d_fp8::kK) kstep(ks, raw);
            }
            {
                /* THE GRIDS DIFFER IN WHAT THE WEIGHT SCALE VARIES OVER, and that is the only
                 * arithmetic difference between the instantiations.
                 *
                 * fp8:  one bf16 per 128x128 tile, so it is uniform over the whole block and folds
                 *       into the activation scale before the accumulate -- one multiply.
                 * w4:   one bf16 per (ROW, group of 128 K), which a 4-bit grid needs: a 128x128
                 *       block scale would spend all sixteen levels on the largest row in it. A
                 *       fragment's lanes hold sixteen consecutive output COLUMNS, so the scale is
                 *       a per-lane constant per N tile and comes out as a third factor.
                 * W4 2-4: groups of 64 or 32 K, two or four a tile, each group's sum under its
                 *       own scale.
                 *
                 * The fp8 form is written out separately rather than folded into a common
                 * expression because a * (b * c) and (a * b) * c differ in the last bit, and that
                 * kernel's output is byte-checked. */
                if constexpr (kSub > 1) {
                    float wq[kSub][WN];
#pragma unroll
                    for (uint32_t q = 0; q < kSub; ++q)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
                            wq[q][j] = s_ws[q * FN + (wgn * WN + j) * 16 + frag_c(lane)];
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                float v = raw[i][j][t] * wq[0][j];
#pragma unroll
                                for (uint32_t q = 1; q < kSub; ++q) v += rx[q - 1][i][j][t] * wq[q][j];
                                const float as_val = s_as[(wgm * WM + i) * 16 + acc_row(lane, t)];
                                out[i][j][t] += v * as_val;
                            }
                } else if constexpr (W4) {
                    float wsj[WN];
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j)
                        wsj[j] = s_ws[(wgn * WN + j) * 16 + frag_c(lane)];
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                const float as_val = s_as[(wgm * WM + i) * 16 + acc_row(lane, t)];
                                out[i][j][t] += raw[i][j][t] * as_val * wsj[j];
                                raw[i][j][t] = 0.0f;
                            }
                } else {
                    const float ws = s_ws[0];
#pragma unroll
                    for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
                        for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                            for (uint32_t t = 0; t < 8; ++t) {
                                const float as_val = s_as[(wgm * WM + i) * 16 + acc_row(lane, t)] * ws;
                                out[i][j][t] += raw[i][j][t] * as_val;
                                raw[i][j][t] = 0.0f;
                            }
                }
            }
        }

        /* The store. For a fixed accumulator slot the wave's sixteen lanes hold sixteen
         * consecutive output columns, so this is coalesced across the wave even though a lane's
         * own eight values are eight rows apart. */
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = (wgm * WM + i) * 16 + acc_row(lane, t);
                    const uint32_t c = (wgn * WN + j) * 16 + frag_c(lane);
                    const uint32_t row = r0 + r;
                    if (r >= FM || row < rs || row >= re || n0 + c >= N) continue;
                    Y[(size_t)row * y_ld + n0 + c] = f32_to_bf16(out[i][j][t]);
                }

        rs = re;
        ++run;
        if (ES) break;                     /* one run a block, and it is done */
    }
}

template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, int W4 = 0,
          bool ES = false, bool AL = false, bool SK = false>
__global__ __launch_bounds__(WGM * WGN * kWave) void moe_gemm_q_kernel(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y, const uint32_t* __restrict__ map,
    uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab) {
    moe_gemm_q_body<MT, NTL, WGM, WGN, W4, ES, AL, SK>(
        Aq, As, c_even, c_odd, c_2, c_3, sorted, off, Y, map, T, N, n_expert, top_k,
        a_ld, as_ld, y_ld, mode, tab);
}

template <int W4 = 0>
__attribute__((amdgpu_num_vgpr(84)))
__global__ __launch_bounds__(4 * kWave) void moe_gemm_q_decode_kernel(
    const uint8_t* __restrict__ Aq, const float* __restrict__ As,
    const MoeCls c_even, const MoeCls c_odd, const MoeCls c_2, const MoeCls c_3,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y, const uint32_t* __restrict__ map,
    uint32_t T, uint32_t N, uint32_t n_expert, uint32_t top_k,
    uint32_t a_ld, uint32_t as_ld, uint32_t y_ld, uint32_t mode, const MoeTab tab) {
    moe_gemm_q_body<1, 4, 1, 4, W4, true, false, false>(
        Aq, As, c_even, c_odd, c_2, c_3, sorted, off, Y, map, T, N, n_expert, top_k,
        a_ld, as_ld, y_ld, mode, tab);
}
/* The whole argument list against the descriptor (see MoeTab): every instantiation takes the same
 * parameters, so the decode form stands for all of them. */
using MoeQArgs = R4dParams<decltype(moe_gemm_q_kernel<1, 4, 1, 4, 0, true>)>;
static_assert(MoeQArgs::bytes() <= RAD_KPARAM_BLOB,
              "moe_gemm_q_kernel's arguments overflow RAD_KPARAM_BLOB");
static_assert(MoeQArgs::count <= RAD_MAX_KPARAMS,
              "moe_gemm_q_kernel has more arguments than RAD_MAX_KPARAMS");

/* ================================================================== moe_gemm, bf16
 *
 * y[i, :] = a[row(i), :] @ w[e]^T in SORTED order, both operands bf16 and one f32 accumulator over
 * the whole of K -- the expert GEMM of a model served from its own bf16 checkpoint. That is the
 * reference deployment, and it is the forward pass a calibration run records its Hessians through,
 * where the activations reaching each layer have to be the ones the unquantised weights produce.
 *
 * THE GROUPING IS moe_gemm_q_kernel's AND ONLY THE INNER LOOP DIFFERS. A block serves one run of
 * the run map at prefill (AL) and one run of a 16-row tile at decode (ES), a wave skips the M tiles
 * its run does not reach (SK), and `a_sorted` picks the activation row a sorted row reads -- every
 * argument made for those beside the fp8 kernel holds here unchanged. So does the reason the forms
 * are byte-identical to each other: every output element is one wave's accumulation over the whole
 * of K, in one order, whichever block computes it.
 *
 * WHAT DIFFERS:
 *   - ONE TABLE, one geometry. The parity classes and `parts` exist because a 640-column expert
 *     does not halve into 128-column scale blocks; a bf16 plane has no block to respect, so a rank
 *     holds whole experts or every expert at one width and every entry has one shape.
 *   - THE PLANE IS THE CHECKPOINT'S ROW-MAJOR [N, K] AS IT STANDS -- no layout hook, no relayout.
 *     The bytes the kernel reads are the bytes in the safetensors file, which is what lets a
 *     container's expert go from disk into a staging buffer and straight to this kernel.
 *   - K IS STAGED kBK16 = 64 ELEMENTS A PASS, 128 bytes a row like the fp8 kernel's 128 codes,
 *     through r4d_gemm_bf16_nt_tiled.hip's LDS pitch of 72 elements: a fragment read's sixteen rows
 *     then land on distinct bank pairs. The next pass is loaded into registers while the current
 *     one is computed, the one-ahead prefetch the fp8 kernel's K-loop note argues for.
 *   - NOTHING TO FOLD: the WMMA accumulates into the output tile across every pass of K.
 */
constexpr uint32_t kBK16 = 64;            // bf16 K elements staged per pass
constexpr uint32_t kP16  = kBK16 + 12;    // the LDS row pitch, in elements (76 elements = 38 dwords, gcd(38, 32)=2)

typedef __bf16 mb_v8bf __attribute__((ext_vector_type(8)));
typedef float  mb_v8f  __attribute__((ext_vector_type(8)));

__device__ __forceinline__ mb_v8f mb_mma(mb_v8bf a, mb_v8bf b, mb_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

/* A 16x16 bf16 operand fragment out of an LDS tile whose row 0 is `tile`, already advanced to the k
 * step. Lane l reads row l % 16 at k = 4 * (l / 16) .. +3 and eight past that -- the wave32 bf16
 * WMMA operand layout, two runs of four (r4d_gdn_wmma.h probed it). Two 8-byte reads 16 bytes apart
 * off one base, which the assembler folds into one ds_load_2addr_b64. */
__device__ __forceinline__ mb_v8bf mb_frag(const uint16_t* __restrict__ tile, uint32_t lane) {
    const uint2* pw = (const uint2*)(tile + (lane & 15u) * kP16 + 4u * (lane >> 4));
    union { uint2 w[2]; mb_v8bf f; } u;
    u.w[0] = pw[0];
    u.w[1] = pw[2];
    return u.f;
}

template <uint32_t MT, uint32_t NTL, uint32_t WGM, uint32_t WGN, bool ES = false, bool AL = false,
          bool SK = false>
__global__ __launch_bounds__(WGM * WGN * kWave) void moe_gemm_bf16_kernel(
    const uint16_t* __restrict__ A, const void* const* __restrict__ Wt, size_t w_s0,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    uint16_t* __restrict__ Y,
    uint32_t T, uint32_t N, uint32_t K, uint32_t n_expert, uint32_t top_k, uint32_t a_sorted,
    uint32_t a_ld, uint32_t w_ld, uint32_t y_ld,
    const uint32_t* __restrict__ map) {
    constexpr uint32_t FM = MT * 16, FN = NTL * 16;
    constexpr uint32_t NTHR = WGM * WGN * kWave;
    constexpr uint32_t WM = MT / WGM, WN = NTL / WGN;
    /* Sixteen-byte chunks: eight of them a staged row, and how many a thread carries per tile. */
    constexpr uint32_t kCPR = kBK16 / 8;
    constexpr uint32_t kAC = (FM * kCPR + NTHR - 1) / NTHR;
    constexpr uint32_t kBC = (FN * kCPR + NTHR - 1) / NTHR;
    static_assert(MT % WGM == 0 && NTL % WGN == 0, "the wave grid must divide the tile");

    __shared__ __attribute__((aligned(16))) uint16_t s_a[FM * kP16];
    __shared__ __attribute__((aligned(16))) uint16_t s_b[FN * kP16];
    __shared__ int32_t s_tok[FM];

    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;
    const uint32_t wgm = wave / WGN, wgn = wave % WGN;
    /* The block's rows and its run, named exactly as moe_gemm_q_kernel names them -- see there. */
    const uint32_t jrun = ES ? (blockIdx.x % FM) : 0u;
    uint32_t r0, rN, e_al = 0;
    bool tail = false;
    if constexpr (AL) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t* ent = map + 1 + 2 * blockIdx.x;
        e_al = ent[0];
        r0   = ent[1];
        tail = (e_al == kMapTail);
        const uint32_t lim = tail ? T : (uint32_t)off[e_al + 1];
        rN = (r0 + FM < lim) ? (r0 + FM) : lim;
    } else {
        r0 = (ES ? (blockIdx.x / FM) : blockIdx.x) * FM;
        if (r0 >= T) return;
        rN = (r0 + FM < T) ? (r0 + FM) : T;
    }
    const uint32_t n0 = blockIdx.y * FN;

    for (uint32_t r = tid; r < FM; r += NTHR) {
        const uint32_t i = r0 + r;
        int32_t t = -1;
        if (i < rN) {
            const int32_t f = sorted[i];
            t = f < 0 ? -1 : (a_sorted ? (int32_t)i : (top_k == 10 ? (int32_t)(((uint64_t)f * 0xCCCCCCCDULL) >> 35) : f / (int32_t)top_k));
        }
        s_tok[r] = t;
    }
    __syncthreads();

    /* Dropped slots past the live count, zeroed by exactly one of a tile's blocks. */
    const uint32_t live = (uint32_t)off[n_expert];
    if (!ES || jrun == 0)
    for (uint32_t r = (r0 > live ? r0 : live); r < rN; ++r)
        for (uint32_t c = tid; c < FN && n0 + c < N; c += NTHR)
            Y[(size_t)r * y_ld + n0 + c] = 0;
    if constexpr (AL) { if (tail) return; }

    /* The expert of the block's first run, found the three ways moe_gemm_q_kernel finds it. */
    uint32_t e = 0;
    if constexpr (AL) {
        e = e_al;
    } else if constexpr (ES) {
        if (T <= 16) {
            __shared__ uint32_t s_es_e;
            __shared__ uint32_t s_es_nrun;
            if (wave == 0) {
                uint32_t e_lane = 0;
                const uint32_t r = r0 + lane;
                const bool valid = (lane < FM) && (r < rN) && (r < live);
                if (valid) {
                    uint32_t lo = 0, hi = n_expert;
                    while (lo + 1 < hi) {
                        const uint32_t mid = (lo + hi) >> 1;
                        if ((uint32_t)off[mid] <= r) lo = mid; else hi = mid;
                    }
                    e_lane = lo;
                }
                const uint32_t prev_e = __shfl_up(e_lane, 1, kWave);
                const bool is_new_run = valid && (lane == 0 || e_lane != prev_e);
                const unsigned long long mk = __ballot(is_new_run);
                const uint32_t nrun = (uint32_t)__popcll(mk);
                const uint32_t pos = (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
                if (is_new_run && pos == jrun) s_es_e = e_lane;
                if (lane == 0) s_es_nrun = nrun;
            }
            __syncthreads();
            if (jrun >= s_es_nrun) return;
            e = s_es_e;
        } else {
            __shared__ int32_t  s_run[FM];
            __shared__ uint32_t s_wc[NTHR / kWave];
            uint32_t nrun = 0;
            for (uint32_t c = 0; c < n_expert && nrun <= jrun; c += NTHR) {
                const uint32_t ce = c + NTHR < n_expert ? c + NTHR : n_expert;
                if ((uint32_t)off[ce] <= r0) continue;          /* every expert here ends before it */
                if ((uint32_t)off[c] >= rN) break;              /* and every one from here after it */
                const uint32_t x = c + tid;
                bool p = false;
                if (x < ce) {
                    const uint32_t a = (uint32_t)off[x], b = (uint32_t)off[x + 1];
                    p = a < b && a < rN && b > r0;
                }
                const unsigned long long mk = __ballot(p);
                if (lane == 0) s_wc[wave] = (uint32_t)__popcll(mk);
                __syncthreads();
                uint32_t before = nrun, tot = 0;
                for (uint32_t w = 0; w < NTHR / kWave; ++w) {
                    if (w < wave) before += s_wc[w];
                    tot += s_wc[w];
                }
                const uint32_t pos = before + (uint32_t)__popcll(mk & ((1ull << lane) - 1ull));
                if (p && pos < FM) s_run[pos] = (int32_t)x;
                __syncthreads();
                nrun += tot;
            }
            if (jrun >= nrun) return;       /* the tile has fewer runs than this block's index */
            e = (uint32_t)s_run[jrun];
        }
    } else {
        uint32_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if ((uint32_t)off[mid] <= r0) lo = mid; else hi = mid;
        }
        e = lo;
    }

    uint32_t rs = r0, run = 0;
    if constexpr (ES) {
        const uint32_t s0 = (uint32_t)off[e];
        rs = s0 > r0 ? s0 : r0;
        run = jrun;
    } else {
        /* A CALL MAY SERVE A TRAILING SEGMENT of the sort -- a layer's protected experts, after
         * the four-bit GEMM has served the rest -- and then its offsets begin at off[0] > 0 and the
         * rows below are another call's output. The block map never names them and the run search
         * above never finds one there; the plain row grid has to be told. */
        const uint32_t s0 = (uint32_t)off[0];
        rs = s0 > r0 ? s0 : r0;
    }
    while (rs < rN && rs < live) {
        while (e < n_expert && (uint32_t)off[e + 1] <= rs) ++e;
        if (e >= n_expert) break;
        const uint32_t ehi = (uint32_t)off[e + 1];
        const uint32_t re  = rN < live ? (rN < ehi ? rN : ehi) : (live < ehi ? live : ehi);
        if (re <= rs) break;
        if (ES && run != jrun) { rs = re; ++run; continue; }

        const uint16_t* Wp = (const uint16_t*)wtab_at(Wt, w_s0, e);
        const bool wlive = !SK || (r0 + wgm * WM * 16 < re);
        mb_v8f acc[WM][WN];
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j) acc[i][j] = (mb_v8f)(0.0f);

        /* Chunk c of a tile is row c / kCPR, elements (c % kCPR) * 8 of it; a row outside the run
         * or a dropped slot stages zeros and issues no load, so the bytes read are the live rows. */
        uint4 apre[kAC], bpre[kBC];
        auto issue = [&](uint32_t k0) {
#pragma unroll
            for (uint32_t p = 0; p < kAC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FM * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                const int32_t tok = s_tok[r];
                const bool ok = tok >= 0 && (r0 + r) >= rs && (r0 + r) < re;
                apre[p] = ok ? *(const uint4*)(A + (size_t)tok * a_ld + k0 + kk)
                             : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FN * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                bpre[p] = (n0 + r) < N ? *(const uint4*)(Wp + (size_t)(n0 + r) * w_ld + k0 + kk)
                                       : make_uint4(0, 0, 0, 0);
            }
        };
        issue(0);

        for (uint32_t k0 = 0; k0 < K; k0 += kBK16) {
            __syncthreads();                       /* the previous pass's fragments are read */
#pragma unroll
            for (uint32_t p = 0; p < kAC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FM * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                *(uint2*)(s_a + r * kP16 + kk)     = make_uint2(apre[p].x, apre[p].y);
                *(uint2*)(s_a + r * kP16 + kk + 4) = make_uint2(apre[p].z, apre[p].w);
            }
#pragma unroll
            for (uint32_t p = 0; p < kBC; ++p) {
                const uint32_t c = tid + p * NTHR;
                if (c >= FN * kCPR) continue;
                const uint32_t r = c / kCPR, kk = (c % kCPR) * 8;
                *(uint2*)(s_b + r * kP16 + kk)     = make_uint2(bpre[p].x, bpre[p].y);
                *(uint2*)(s_b + r * kP16 + kk + 4) = make_uint2(bpre[p].z, bpre[p].w);
            }
            if (k0 + kBK16 < K) issue(k0 + kBK16);
            __syncthreads();

            if (!wlive) continue;
#pragma unroll
            for (uint32_t ks = 0; ks < kBK16; ks += 16) {
                mb_v8bf af[WM], bf[WN];
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
                    af[i] = mb_frag(s_a + (wgm * WM + i) * 16 * kP16 + ks, lane);
                }
#pragma unroll
                for (uint32_t j = 0; j < WN; ++j)
                    bf[j] = mb_frag(s_b + (wgn * WN + j) * 16 * kP16 + ks, lane);
#pragma unroll
                for (uint32_t i = 0; i < WM; ++i) {
                    if (SK && r0 + (wgm * WM + i) * 16 >= re) continue;
#pragma unroll
                    for (uint32_t j = 0; j < WN; ++j) acc[i][j] = mb_mma(af[i], bf[j], acc[i][j]);
                }
            }
        }

        /* Lane l holds column l % 16 of a tile and rows 8 * (l / 16) + t: the accumulator layout
         * the fp8 kernel's acc_row / frag_c also describe. Rounded to nearest-even. */
#pragma unroll
        for (uint32_t i = 0; i < WM; ++i)
#pragma unroll
            for (uint32_t j = 0; j < WN; ++j)
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    const uint32_t r = (wgm * WM + i) * 16 + 8 * (lane >> 4) + t;
                    const uint32_t c = (wgn * WN + j) * 16 + (lane & 15u);
                    const uint32_t row = r0 + r;
                    if (r >= FM || row < rs || row >= re || n0 + c >= N) continue;
                    Y[(size_t)row * y_ld + n0 + c] = f32_to_bf16(acc[i][j][t]);
                }

        rs = re;
        ++run;
        if (ES) break;
    }
}

/* ================================================================== moe_gather
 *
 * y[t, :] = sum over a token's slots of expert_w[t, j] * y_expert[position of (t, j)], and the sum
 * runs in ASCENDING SLOT ORDER because that is what makes it reproducible -- walking the
 * expert-sorted rows instead would sum a token's contributions in expert-id order and the two
 * differ in the last bits.
 *
 * The inverse permutation is built by a pass of its own rather than searched per token: `sorted` is
 * a permutation of the live slots, so `inv[sorted[i]] = i` is one scattered store per row. A slot
 * nothing claims keeps -1 and contributes nothing, which is how a dropped slot is expressed. */
__global__ void gather_inv_kernel(int32_t* __restrict__ inv, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) inv[i] = -1;
}

__global__ void gather_scatter_kernel(const int32_t* __restrict__ sorted,
                                      int32_t* __restrict__ inv, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    const int32_t f = sorted[i];
    if (f >= 0 && f < (int32_t)T) inv[f] = (int32_t)i;
}

/* THE SLOT TABLE IS HOISTED, AND THAT IS THE WHOLE OF THIS KERNEL'S COST.
 *
 * `inv[t * top_k + j]` and `ew[t * ew_ld + j]` do not depend on the column. Read inside the column
 * loop -- `inv` IN FRONT OF the `ye` load that uses it -- every one of the top_k loads is a
 * dependent round trip: at a decode step's one token, 2560 columns and eleven slots, that is 110
 * serialised memory latencies, tens of microseconds a layer. Read once into LDS, the `ye` loads
 * are independent and pipeline.
 *
 * The column range is also split over blocks, because one token is one block otherwise and a
 * decode step has one token. Neither the hoist nor the split touches the sum: a column's slots are
 * still summed in ASCENDING SLOT ORDER, which is what the note above says is load-bearing, and no
 * column's sum is split across blocks. */
constexpr uint32_t kGatherSlots = 64;
/* THE INVERSE PERMUTATION IS BUILT IN THIS KERNEL WHEN IT IS SMALL ENOUGH TO BE FREE, and the two
 * prepass kernels above are then not launched at all.
 *
 * A block owns one token, so it needs the inverse for `top_k` slots and not for all T of them:
 * scanning `sorted` for the entries that land in [t*top_k, (t+1)*top_k) gives exactly the same
 * table `inv` would have held, built from LDS instead of a device round trip. Every block repeats
 * the scan, which is why the launcher caps it -- at a decode step T is M*top_k = 40 and the scan
 * is one pass of a 256-thread block, against two dispatches and 2.9 us of kernel time for two
 * single-block kernels that between them touch 40 integers. See r4d_rad_moe_gather for the bound.
 *
 * WHAT IT SAVES IS DISPATCHES RATHER THAN KERNEL TIME, and the arithmetic is unchanged -- `inv`
 * holds exactly what the prepass would have put there. The two kernels it removes are together a
 * fraction of a millisecond a rank a step of device time across 96 dispatches, and at the ~2.1 us
 * this part charges for a dependent launch the dispatches are the larger half.
 *
 * `sorted == nullptr` is the prepass path, the one a prefill chunk takes. */
constexpr uint32_t kGatherSelfT = 512;

template <uint32_t NTHR>
__global__ __launch_bounds__(NTHR) void gather_kernel(
    const uint16_t* __restrict__ ye, const uint16_t* __restrict__ ew,
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,
    const int32_t* __restrict__ sorted, uint32_t T,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg,
    uint32_t sh_ld, uint32_t sg_ld, int sig) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    const uint32_t t = blockIdx.x;
    if (t >= M) return;
    const bool cached = top_k <= kGatherSlots;
    /* The shared arm's scalar, read once a block exactly as scale_rows reads it once a row. */
    float gate = 0.0f;
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
    if (sorted) {
        /* A slot no sorted row claims must read -1, which is why the fill and the scan are two
         * phases with a barrier between them -- the same reason the prepass is two kernels. Here
         * the barrier is __syncthreads() and costs nothing. */
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR) s_i[j] = -1;
        __syncthreads();
        const int32_t lo = (int32_t)(t * top_k), hi = lo + (int32_t)top_k;
        for (uint32_t i = threadIdx.x; i < T; i += NTHR) {
            const int32_t f = sorted[i];
            if (f >= lo && f < hi) s_i[f - lo] = (int32_t)i;
        }
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR)
            s_w[j] = bf16_to_f32(ew[(size_t)t * ew_ld + j]);
        __syncthreads();
    } else if (cached) {
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR) {
            s_i[j] = inv[t * top_k + j];
            s_w[j] = bf16_to_f32(ew[(size_t)t * ew_ld + j]);
        }
        __syncthreads();
    }
    for (uint32_t c = blockIdx.y * NTHR + threadIdx.x; c < n; c += NTHR * gridDim.y) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < top_k; ++j) {
            const int32_t i = cached ? s_i[j] : inv[t * top_k + j];
            if (i < 0) continue;
            const float wj = cached ? s_w[j] : bf16_to_f32(ew[(size_t)t * ew_ld + j]);
            acc += wj * bf16_to_f32(ye[(size_t)i * ye_ld + c]);
        }
        const uint16_t g = f32_to_bf16(acc);
        if (!sh) { y[(size_t)t * y_ld + c] = g; continue; }
        /* THE SHARED ARM, FOLDED. Both narrows are the unfused pair's: `scale_rows` reads this
         * gather out of a bf16 buffer and narrows its own product before adding, so carrying
         * either in f32 here would move the last bit of every channel. */
        const uint16_t p = f32_to_bf16(bf16_to_f32(sh[(size_t)t * sh_ld + c]) * gate);
        y[(size_t)t * y_ld + c] = f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(g));
    }
}

/* EIGHT COLUMNS A LANE, the same sums. gather_kernel gives a lane one column, so every ye load is
 * one u16 a lane and 64 bytes a wave instruction, and its runtime-trip slot loop issues them one
 * behind each accumulate: 254 GB/s at a prefill chunk. Here a lane owns 8 contiguous bf16 (one
 * 16-byte load a slot, 512 B a wave instruction) and a batch of kGatherBatch slots is loaded
 * before any of it is summed.
 *
 * BIT-IDENTICAL: each column still starts at 0, adds `wj * x` for ascending j over the live slots
 * only, in the same expression (so the same contraction), and narrows through the same epilogue.
 * The launcher takes this form only for top_k <= kGatherSlots (the LDS slot table) and n, every
 * pitch and every base 16-byte aligned; anything else is gather_kernel, unchanged. */
constexpr uint32_t kGatherBatch = 16;

__device__ __forceinline__ uint16_t gather_lane_u16(const uint4& v, uint32_t k) {
    const uint32_t w = k < 2 ? v.x : k < 4 ? v.y : k < 6 ? v.z : v.w;
    return (uint16_t)((k & 1) ? (w >> 16) : (w & 0xFFFFu));
}

template <uint32_t NTHR>
__global__ __launch_bounds__(NTHR) void gather_vec_kernel(
    const uint16_t* __restrict__ ye, const uint16_t* __restrict__ ew,
    const int32_t* __restrict__ inv, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t top_k, uint32_t ye_ld, uint32_t ew_ld, uint32_t y_ld,
    const int32_t* __restrict__ sorted, uint32_t T,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg,
    uint32_t sh_ld, uint32_t sg_ld, int sig) {
    __shared__ int32_t s_i[kGatherSlots];
    __shared__ float   s_w[kGatherSlots];
    const uint32_t t = blockIdx.x;
    if (t >= M) return;
    float gate = 0.0f;
    if (sh) {
        const float v0 = bf16_to_f32(sg[(size_t)t * sg_ld]);
        gate = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
    /* The slot table, filled exactly as gather_kernel's `cached` forms fill it. */
    if (sorted) {
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR) s_i[j] = -1;
        __syncthreads();
        const int32_t lo = (int32_t)(t * top_k), hi = lo + (int32_t)top_k;
        for (uint32_t i = threadIdx.x; i < T; i += NTHR) {
            const int32_t f = sorted[i];
            if (f >= lo && f < hi) s_i[f - lo] = (int32_t)i;
        }
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR)
            s_w[j] = bf16_to_f32(ew[(size_t)t * ew_ld + j]);
    } else {
        for (uint32_t j = threadIdx.x; j < top_k; j += NTHR) {
            s_i[j] = inv[t * top_k + j];
            s_w[j] = bf16_to_f32(ew[(size_t)t * ew_ld + j]);
        }
    }
    __syncthreads();
    const uint32_t n8 = n / 8;
    for (uint32_t c8 = blockIdx.y * NTHR + threadIdx.x; c8 < n8; c8 += NTHR * gridDim.y) {
        const uint32_t c = c8 * 8;
        float acc[8];
#pragma unroll
        for (uint32_t k = 0; k < 8; ++k) acc[k] = 0.0f;
        for (uint32_t j0 = 0; j0 < top_k; j0 += kGatherBatch) {
            /* Every load of the batch first; `i` is uniform across the block, so the predicate is
             * a scalar branch and a dead slot issues nothing. */
            int32_t iv[kGatherBatch];
            uint4   xv[kGatherBatch];
#pragma unroll
            for (uint32_t p = 0; p < kGatherBatch; ++p) {
                iv[p] = (j0 + p < top_k) ? s_i[j0 + p] : -1;
                xv[p] = iv[p] >= 0 ? *(const uint4*)(ye + (size_t)iv[p] * ye_ld + c)
                                   : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (uint32_t p = 0; p < kGatherBatch; ++p) {
                if (iv[p] < 0) continue;
                const float wj = s_w[j0 + p];
#pragma unroll
                for (uint32_t k = 0; k < 8; ++k)
                    acc[k] += wj * bf16_to_f32(gather_lane_u16(xv[p], k));
            }
        }
        uint16_t o[8];
        if (!sh) {
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) o[k] = f32_to_bf16(acc[k]);
        } else {
            /* The folded shared arm, both narrows as in gather_kernel. */
            const uint4 sv = *(const uint4*)(sh + (size_t)t * sh_ld + c);
#pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const uint16_t g = f32_to_bf16(acc[k]);
                const uint16_t p = f32_to_bf16(bf16_to_f32(gather_lane_u16(sv, k)) * gate);
                o[k] = f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(g));
            }
        }
        *(uint4*)(y + (size_t)t * y_ld + c) =
            make_uint4((uint32_t)o[0] | ((uint32_t)o[1] << 16), (uint32_t)o[2] | ((uint32_t)o[3] << 16),
                       (uint32_t)o[4] | ((uint32_t)o[5] << 16), (uint32_t)o[6] | ((uint32_t)o[7] << 16));
    }
}

/* ================================================================== scale_rows
 *
 * y[m, i] = x[m, i] * act(s[m]). One workgroup a row, grid-strided along the row, and the scalar
 * read once into a register rather than once per element. It is the shared expert's gate and
 * nothing else in this tree uses it; see docs/OPS.md for why `mul` cannot serve it. */
/* `ad` PRESENT IS THE FUSED GATE-AND-ADD FORM, `y = bf16(x * act(s)) + ad`, and the NARROW in the
 * middle is load-bearing: the pair this stands in for writes the product to a bf16 buffer and
 * then adds that buffer, so the product is rounded before the sum sees it. Carrying the f32
 * product into the add would differ in the last bit of every channel, and the emitted text with
 * it. */
template <uint32_t NTHR>
__global__ __launch_bounds__(NTHR) void scale_rows_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ sc,
    const uint16_t* __restrict__ ad, uint16_t* __restrict__ y,
    uint32_t M, uint32_t n, uint32_t x_ld, uint32_t s_ld, uint32_t a_ld, uint32_t y_ld, int sig) {
    const uint32_t m = blockIdx.x;
    if (m >= M) return;
    const float v0 = bf16_to_f32(sc[(size_t)m * s_ld]);
    const float v  = sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    for (uint32_t i = threadIdx.x; i < n; i += NTHR) {
        const uint16_t p = f32_to_bf16(bf16_to_f32(x[(size_t)m * x_ld + i]) * v);
        y[(size_t)m * y_ld + i] =
            ad ? f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(ad[(size_t)m * a_ld + i])) : p;
    }
}

}  /* namespace */

/* ============================================================== the plugin ABI entry points */

extern "C" int r4d_rad_router_topk(const RadArgs* a, RadStream s) {
    const RadTensor* lg  = r4d_opt(a, 0);
    const RadTensor* ids = r4d_opt(a, 1);
    const RadTensor* w   = r4d_opt(a, 2);
    if (!lg || !ids || !w || !lg->data || !ids->data || !w->data) return R4D_NO(RAD_E_INVAL);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return R4D_NO(RAD_E_SHAPE);
    /* The key packs `~e` into sixteen bits, so an expert id has to fit there: kMaxExpert is 512
     * and the refusal above is what holds it. top_k is bounded by the wave, because round r parks
     * its winner on lane r. */
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long lg_ld = r4d_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(ids) || !r4d_contig(w)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ids) < M * top_k || r4d_numel(w) < M * top_k) return R4D_NO(RAD_E_SHAPE);

    constexpr uint32_t NW = 4;
    const dim3 grid((unsigned)((M + NW - 1) / NW)), blk(NW * kWave);
    /* THE NARROWEST INSTANTIATION THAT COVERS n_expert, because the wider one costs eight
     * registers a lane for nothing on a 256-expert model. */
    if (n_expert <= (long long)(8 * kWave))
        hipLaunchKernelGGL((router_topk_kernel<NW, 8>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld,
                           (int)norm);
    else
        hipLaunchKernelGGL((router_topk_kernel<NW, 16>), grid, blk, 0, (hipStream_t)s,
                           (const uint16_t*)lg->data, (int32_t*)ids->data, (uint16_t*)w->data,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)lg_ld,
                           (int)norm);
    R4D_MOE_CHK();
    return RAD_OK;
}

/* `protect`, a comma list of local expert indices: strictly ascending, inside the slice, at most
 * kMaxProtect. Absent or empty is none. */
static int protect_of(const RadArgs* a, long long n_local, Protect* p) {
    *p = Protect{};
    const char* s = r4d_gets(a, "protect");
    if (!s || !*s) return RAD_OK;

    struct ProtectCacheEntry {
        const char* str = nullptr;
        long long n_local = 0;
        Protect prot{};
        int rc = 0;
    };
    static ProtectCacheEntry s_cache[64];
    static int s_cache_len = 0;
    for (int i = 0; i < s_cache_len; ++i) {
        if (s_cache[i].str == s && s_cache[i].n_local == n_local) {
            *p = s_cache[i].prot;
            return s_cache[i].rc;
        }
    }

    long long prev = -1;
    const char* cur = s;
    while (*cur) {
        char* end = nullptr;
        const long long v = std::strtoll(cur, &end, 10);
        if (end == cur || v <= prev || v >= n_local || p->n >= kMaxProtect)
            return R4D_NO(RAD_E_SHAPE);
        p->e[p->n++] = (uint16_t)v;
        prev = v;
        cur = end;
        if (*cur == ',') ++cur;
        else if (*cur) return R4D_NO(RAD_E_SHAPE);
    }
    if (s_cache_len < 64) {
        s_cache[s_cache_len++] = { s, n_local, *p, RAD_OK };
    }
    return RAD_OK;
}

/* THE PAIR IN ONE LAUNCH. Operands are the two ops' own, concatenated with the selection's first:
 * logits -> expert_ids, expert_w, sorted_tok, expert_offset, expert_count. Every check below is
 * one of the two shims' checks; nothing new is asserted, because nothing new is computed.
 *
 * IT REFUSES rather than falls back when the sort would not fit one workgroup. A silent fallback
 * would make the op's cost depend on a bound the caller cannot see, and the architecture already
 * asks the same question before declaring it (`moe_fuse_rows`). */
static int rad_router_topk_scatter_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const RadTensor* lg     = r4d_opt(a, 0);
    const RadTensor* ids    = r4d_opt(a, 1);
    const RadTensor* w      = r4d_opt(a, 2);
    const RadTensor* sorted = r4d_opt(a, 3);
    const RadTensor* off    = r4d_opt(a, 4);
    const RadTensor* cnt    = r4d_opt(a, 5);
    if (!lg || !ids || !w || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if (!lg->data || !ids->data || !w->data || !sorted->data || !off->data || !cnt->data)
        return R4D_NO(RAD_E_INVAL);

    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    /* Absent means "no expert parallelism", where the slice IS the router. */
    const long long n_local  = r4d_geti_or(a, "n_local", n_expert);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    const long long base     = r4d_geti_or(a, "expert_base", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return R4D_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (n_local <= 0 || n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    if (base < 0 || base + n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_local, &prot);
    if (prc != RAD_OK) return prc;
    if (lg->dtype != RAD_BF16 || w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long lg_ld = r4d_rows_pitch(lg, n_expert, &M);
    if (lg_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    if (!r4d_contig(ids) || !r4d_contig(w) || !r4d_contig(sorted) || !r4d_contig(off) ||
        !r4d_contig(cnt)) return R4D_NO(RAD_E_STRIDE);

    const long long T = M * top_k;
    if (r4d_numel(ids) < T || r4d_numel(w) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(off) < n_local + 1 || r4d_numel(cnt) < n_local) return R4D_NO(RAD_E_SHAPE);

    if (T <= (long long)kScatterSmall) {
#define R4D_RTS_L(OWN, NWV)                                                                     \
        emit_rc = R4D_EMIT(sk, (router_topk_scatter_kernel<OWN, NWV>), dim3(1),                 \
                           dim3((NWV) * kWave), 0,                                              \
                           (const uint16_t*)lg->data, (int32_t*)ids->data,                      \
                           (uint16_t*)w->data, (int32_t*)sorted->data, (int32_t*)off->data,     \
                           (int32_t*)cnt->data, (uint32_t)M, (uint32_t)n_expert,                \
                           (uint32_t)n_local, (uint32_t)top_k,                                  \
                           (uint32_t)lg_ld, (int)norm, (int32_t)base, prot)
        /* The same narrowest-instantiation rule the standalone selection uses, and for the same
         * reason: MAXOWN is registers a lane.
         *
         * ONE WAVE A ROW, as a power of two between the sort's own eight waves and the workgroup's
         * thirty-two. A wave selects its rows one after another, so at eight waves a 32-row verify
         * step runs four selections back to back: 8.6 us against 5.4 at thirty-two waves. Fewer rows do
         * not pay for more waves -- a launch of thirty-two costs about 0.4 us more than one of eight at
         * four rows. Above 32 rows the waves wrap. */
        const int nw = M <= 8 ? 8 : M <= 16 ? 16 : 32;
        if (n_expert <= (long long)(8 * kWave)) {
            if      (nw == 8)  R4D_RTS_L(8, 8);
            else if (nw == 16) R4D_RTS_L(8, 16);
            else               R4D_RTS_L(8, 32);
        } else {
            if      (nw == 8)  R4D_RTS_L(16, 8);
            else if (nw == 16) R4D_RTS_L(16, 16);
            else               R4D_RTS_L(16, 32);
        }
#undef R4D_RTS_L
        if (emit_rc != RAD_OK) return emit_rc;
        if (!sk.desc) R4D_MOE_CHK();
        return RAD_OK;
    }

    /* R4-01: Prefill Router-Top-K and Histogram Fusion (T > kScatterSmall, M >= 128).
     * Eliminates standalone scatter_hist_kernel across prefill:
     * Launch 1: Fused router_topk_prefill_hist_kernel calculates Top-K and populates bcnt.
     * Launch 2: scatter_scan_kernel scans bcnt, building offsets and expert counts.
     * Launch 3: scatter_place_kernel places slots into sorted order. */
    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    const int64_t need = n_tile * n_local * (long long)sizeof(int32_t);
    if (!sk.desc && (!a->scratch || a->scratch_bytes < need)) return R4D_NO(RAD_E_SCRATCH);
    int32_t* bcnt = (int32_t*)a->scratch;

    const unsigned lds_bytes = (unsigned)(n_local * sizeof(int32_t));
    if (n_expert <= (long long)(8 * kWave)) {
        emit_rc = R4D_EMIT(sk, (router_topk_prefill_hist_kernel<8>),
                           dim3((unsigned)n_tile), dim3(kScatterTile), lds_bytes,
                           (const uint16_t*)lg->data, (int32_t*)ids->data,
                           (uint16_t*)w->data, bcnt,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)n_local,
                           (uint32_t)top_k, (uint32_t)lg_ld, (int)norm,
                           (int32_t)base, prot);
    } else {
        emit_rc = R4D_EMIT(sk, (router_topk_prefill_hist_kernel<16>),
                           dim3((unsigned)n_tile), dim3(kScatterTile), lds_bytes,
                           (const uint16_t*)lg->data, (int32_t*)ids->data,
                           (uint16_t*)w->data, bcnt,
                           (uint32_t)M, (uint32_t)n_expert, (uint32_t)n_local,
                           (uint32_t)top_k, (uint32_t)lg_ld, (int)norm,
                           (int32_t)base, prot);
    }
    if (emit_rc != RAD_OK) return emit_rc;

    emit_rc = R4D_EMIT(sk, scatter_scan_kernel, dim3(1), dim3(256), 0,
                       bcnt, (int32_t*)off->data, (int32_t*)cnt->data,
                       (uint32_t)n_tile, (uint32_t)n_local);
    if (emit_rc != RAD_OK) return emit_rc;

    emit_rc = R4D_EMIT(sk, scatter_place_kernel,
                       dim3((unsigned)n_tile), dim3(kScatterTile), 0,
                       (const int32_t*)ids->data, bcnt,
                       (const int32_t*)off->data, (int32_t*)sorted->data,
                       (uint32_t)T, (uint32_t)n_local, (int32_t)base, prot);
    if (emit_rc != RAD_OK) return emit_rc;
    if (!sk.desc) R4D_MOE_CHK();
    return RAD_OK;
}
R4D_ENTRY(r4d_rad_router_topk_scatter, rad_router_topk_scatter_body)

extern "C" int64_t r4d_rad_router_topk_scatter_scratch(const RadArgs* a) {
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long n_local  = r4d_geti_or(a, "n_local", n_expert);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    if (n_expert <= 0 || n_local <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);

    long long M = r4d_geti_or(a, "M", 0);
    if (M <= 0) {
        const RadTensor* lg = r4d_opt(a, 0);
        if (lg && lg->data) {
            const long long lg_ld = r4d_rows_pitch(lg, n_expert, &M);
            if (lg_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
        } else {
            const RadTensor* ids = r4d_opt(a, 1);
            if (ids && ids->data) M = r4d_numel(ids) / top_k;
        }
    }
    if (M <= 0) return R4D_NO(RAD_E_SHAPE);

    const long long T = M * top_k;
    if (T <= (long long)kScatterSmall) return 0;
    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    return n_tile * n_local * (long long)sizeof(int32_t);
}

/* Scratch for the general scatter: the per-tile histogram plane. The small form needs none, and
 * the hook answers for the shape it is asked about rather than for the worst case, because the
 * band the op resolves in is what sizes the arena. */
static int scatter_dims(const RadArgs* a, long long* T, long long* n_expert) {
    const RadTensor* ids = r4d_opt(a, 0);
    *n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 1);
    if (*n_expert <= 0 || *n_expert > (long long)kMaxExpert || top_k <= 0)
        return R4D_NO(RAD_E_SHAPE);
    *T = ids && ids->data ? r4d_numel(ids) : r4d_geti_or(a, "M", 0) * top_k;
    if (*T <= 0) return R4D_NO(RAD_E_SHAPE);
    return RAD_OK;
}

extern "C" int64_t r4d_rad_moe_scatter_scratch(const RadArgs* a) {
    long long T = 0, n_expert = 0;
    const int rc = scatter_dims(a, &T, &n_expert);
    if (rc != RAD_OK) return rc;
    if (T <= (long long)kScatterSmall) return 0;
    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    return n_tile * n_expert * (long long)sizeof(int32_t);
}

extern "C" int r4d_rad_moe_scatter(const RadArgs* a, RadStream s) {
    const RadTensor* ids    = r4d_opt(a, 0);
    const RadTensor* sorted = r4d_opt(a, 1);
    const RadTensor* off    = r4d_opt(a, 2);
    const RadTensor* cnt    = r4d_opt(a, 3);
    if (!ids || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if (!ids->data || !sorted->data || !off->data || !cnt->data) return R4D_NO(RAD_E_INVAL);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);
    if (!r4d_contig(ids) || !r4d_contig(sorted) || !r4d_contig(off) || !r4d_contig(cnt))
        return R4D_NO(RAD_E_STRIDE);

    long long T = 0, n_expert = 0;
    const int rc = scatter_dims(a, &T, &n_expert);
    if (rc != RAD_OK) return rc;
    if (r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(off) < n_expert + 1 || r4d_numel(cnt) < n_expert) return R4D_NO(RAD_E_SHAPE);
    /* THIS RANK'S FIRST EXPERT, subtracted from every routed id so that everything downstream --
     * the offsets, the counts, the weight table -- is indexed locally. 0 is the whole-model case,
     * without expert parallelism. */
    const long long base = r4d_geti_or(a, "expert_base", 0);
    if (base < 0) return R4D_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_expert, &prot);
    if (prc != RAD_OK) return prc;

    const int32_t* idp = (const int32_t*)ids->data;
    int32_t* sp = (int32_t*)sorted->data;
    int32_t* op = (int32_t*)off->data;
    int32_t* cp = (int32_t*)cnt->data;

    if (T <= (long long)kScatterSmall) {
        hipLaunchKernelGGL(scatter_small_kernel, dim3(1), dim3(256), 0, (hipStream_t)s,
                           idp, sp, op, cp, (uint32_t)T, (uint32_t)n_expert, (int32_t)base,
                           prot);
        R4D_MOE_CHK();
        return RAD_OK;
    }

    const long long n_tile = (T + kScatterTile - 1) / kScatterTile;
    const int64_t need = n_tile * n_expert * (long long)sizeof(int32_t);
    if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
    int32_t* bcnt = (int32_t*)a->scratch;

    hipLaunchKernelGGL(scatter_hist_kernel, dim3((unsigned)n_tile), dim3(kScatterTile),
                       (unsigned)(n_expert * sizeof(int32_t)), (hipStream_t)s,
                       idp, bcnt, (uint32_t)T, (uint32_t)n_expert, (int32_t)base, prot);
    R4D_MOE_CHK();
    hipLaunchKernelGGL(scatter_scan_kernel, dim3(1), dim3(256), 0, (hipStream_t)s,
                       bcnt, op, cp, (uint32_t)n_tile, (uint32_t)n_expert);
    R4D_MOE_CHK();
    hipLaunchKernelGGL(scatter_place_kernel, dim3((unsigned)n_tile), dim3(kScatterTile), 0,
                       (hipStream_t)s, idp, bcnt, op, sp, (uint32_t)T, (uint32_t)n_expert,
                       (int32_t)base, prot);
    R4D_MOE_CHK();
    return RAD_OK;
}

/* The weight-table operand, read off the tensor the core built. A zero leading stride is the
 * marker for a pointer array (abi/rad_runtime.h, RAD_WTAB); anything else is a stacked plane and
 * the stride is in ELEMENTS of the weight's dtype, which for E4M3 is bytes. */
static int wtab_of(const RadTensor* t, uint32_t dtype, int bits, long long n_expert,
                   long long rows, long long cols, const void* const** tab, size_t* stride0,
                   long long* ld) {
    if (!t || !t->data || t->rank != 3) return R4D_NO(RAD_E_SHAPE);
    if (t->dtype != dtype) return R4D_NO(RAD_E_DTYPE);
    if (t->shape[0] < n_expert || t->shape[1] < rows || t->shape[2] < cols)
        return R4D_NO(RAD_E_SHAPE);
    if (t->stride[2] != 1) return R4D_NO(RAD_E_STRIDE);
    *tab     = (const void* const*)t->data;
    /* THE ELEMENT WIDTH IS PASSED IN, not looked up. A plugin-private dtype -- which the 4-bit
     * expert plane is -- is not in the core's table, so rad_dtype_bits() answers 0 for it and a
     * stacked stride would collapse to the very zero that marks a POINTER ARRAY. The bits are
     * multiplied before the divide so a four-bit plane keeps its half-byte. */
    *stride0 = (size_t)((t->stride[0] * (int64_t)bits) / 8);
    *ld      = t->stride[1];
    return RAD_OK;
}

/* ---- AL: ONE RUN A BLOCK, SO A STREAMED EXPERT'S WEIGHT CROSSES THE LINK ONCE -----------------
 *
 * The uniform row grid above puts a block on rows [i*FM, i*FM + FM) whatever expert those rows
 * belong to, so a run that straddles a tile boundary is served by TWO blocks and its weight is
 * staged twice. The kernel's tile note counts that as `n_expert + T/FM` incidences against an
 * optimum of `n_expert`, and for a resident expert a duplicated tile is not worth a prefix-sum
 * pass and an extra launch a layer.
 *
 * THAT IS TRUE OF A RESIDENT EXPERT AND FALSE OF A POOLED ONE. A second staging of a weight that
 * is in VRAM is an L2 or MALL hit. A second staging of one that is not is another pass of its
 * whole shard over PCIe, and the two differ by more than an order of magnitude: on this part the
 * streamed read runs at about 26 GB/s against VRAM's 576. At the shipped 2048 chunk the
 * grid is 160 tiles over 254 experts, so the incidence count is 414 against 254 -- a THIRD of the
 * link traffic is a re-read of weight a neighbouring block already pulled.
 *
 * So a block map is built here, and it is one run a block:
 * entry i names (expert, first row), the block serves rows [first, min(first + FM, off[e+1])), and
 * no expert is ever staged by two blocks unless its own run is longer than FM.
 *
 * WHAT IT DOES NOT COST:
 *
 *   * NOT ACTIVATION TRAFFIC. An aligned block still stages FM rows, but `stage8_load` takes a
 *     live flag and issues no load at all for a row outside the run -- so the bytes read are the
 *     LIVE rows either way, and the block count does not enter it.
 *   * NOT ARITHMETIC. The uniform grid already runs the whole FM tile once per run it covers, with
 *     the rows of the other runs masked. At the shipped chunk that is 160 tiles x ~1.6 runs = 256
 *     full-tile passes; aligned it is 254, one a run. The dead-row work is the same work.
 *   * NOT DETERMINISM. Every output row is still produced by one block, from one expert, with the
 *     same K order -- only which workgroup does it moves, exactly as the ES split above. The
 *     result is bit-identical.
 *
 * THE GRID IS AN UPPER BOUND AND THE COUNT IS ON THE DEVICE. `sum_e ceil(run_e / FM)` cannot be
 * known on the host without reading the offsets back, which is a synchronisation a step does not
 * have. It is bounded by `n_expert + T/FM` -- every expert contributes one block plus at most one
 * more per FM rows -- so the launcher uses that, the map's first slot carries the real count, and
 * blocks past it exit at the head. Those are a few hundred empty workgroups on a grid of tens of
 * thousands.
 *
 * ROWS NO EXPERT OWNS still have to be zeroed: `off[n_expert]` is the live row count and anything
 * above it is a dropped slot whose output row must not be whatever the arena left there. The map
 * carries them as entries with expert `kMapTail`, and such a block zeroes and returns. */


/* THE PREFILL TILE'S ROWS -- see "THE PREFILL TILE IS 32 ROWS BY 64 COLUMNS" at the kernel. */
constexpr long long kPrefillRows = 32;
/* ...AND 128 COLUMNS OVER EIGHT WAVES where a column tile cannot straddle two stacked parts of the
 * weight. A wave's share is what the 64-column tile's four waves each had -- one 16-row tile by two
 * 16-column ones -- so the arithmetic of every element is unchanged; what the wider block buys is
 * the rows' activation staged once for twice the columns, and half the blocks. r4d_selftest
 * --perfmoe, rank 0's tp2 shape, 512 experts, top 10, us a call (gate_up / down), same bytes:
 *
 *      tokens               512          1024         2048          4096
 *      32 x 64, 4 waves     822 / 491    898 / 620    1305 / 884    2255 / 1477
 *      32 x 128, 8 waves    780 / 468    868 / 582    1242 / 845    2133 / 1450
 *
 * Four waves over 128 columns (two a wave across 64) are within ~3% of the 64-column tile at forty
 * rows a run and slower at twenty: the width is only worth it with the waves to go with it. */
constexpr long long kPrefillCols = 128;

/* WHERE THE (tile, run) DECODE GRID ENDS AND THE RUN-ALIGNED PREFILL GRID BEGINS: at an average of
 * one and a half routed rows an expert. At file scope because the scratch hook and the launcher
 * must answer for the SAME band -- a hook that sized a map for a shape the launcher then ran the
 * other form on would be arena charged to nothing, and the reverse would be a launcher finding
 * scratch it was not given.
 *
 * THE CROSSOVER IS A RUN LENGTH, NOT A TOKEN COUNT. The decode form gives every run its own block
 * and a 16-row tile; the prefill form gives it a 32-row tile. Where runs are a row or two the
 * narrower tile wins, and once they are several rows the wider one does. --perfmoe at 512
 * experts, gate_up + down in us:
 *
 *      T (rows)         640    800    1000   1280   2560   5120
 *      decode form      804    900    1006   1113   1541   2224
 *      prefill form     826    900     981   1053   1247   1413
 *
 * So the band is T <= 1.5 * n_expert: 768 rows here. The decode envelope is
 * `max_num_seqs * (1 + n_spec) * top_k` -- 320 rows at eight sequences, depth 3 and top 10 -- so
 * a decode step stays on the decode form with room; a server with a few dozen concurrent
 * sequences crosses onto the prefill form exactly where that form is the faster one. */
static inline bool r4d_moe_decode_band(long long T, long long n_expert) {
    return 2 * T <= 3 * n_expert;
}
/* THE MAP'S UPPER BOUND, stated once because the scratch hook and the launcher must agree about
 * it exactly: a hook that answers for a different shape than the launch uses is how a kernel ends
 * up writing past the arena it was given. Every expert contributes one block, plus one more for
 * each FM rows its run exceeds; the dropped-slot tail contributes at most T/FM and one more. */
static inline long long r4d_blockmap_cap(long long T, long long n_expert, long long FM) {
    return n_expert + (T + FM - 1) / FM + 1;
}
/* One count slot, then two uint32 an entry. */
static inline long long r4d_blockmap_bytes(long long cap) {
    return (1 + 2 * cap) * (long long)sizeof(uint32_t);
}


/* ONE WORKGROUP, because the whole of it is a scan over at most kMaxExpert counters and a second
 * launch to split that across the device would cost more than the scan does. The offsets are read
 * into LDS once so the scan is not 512 dependent global loads. */
__global__ __launch_bounds__(256) void moe_blockmap_kernel(
    const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
    uint32_t* __restrict__ map) {
    __shared__ uint32_t s_cnt[kMaxExpert + 1];
    __shared__ int32_t  s_off[kMaxExpert + 1];
    const uint32_t tid = threadIdx.x;

    for (uint32_t e = tid; e <= n_expert; e += blockDim.x) s_off[e] = off[e];
    __syncthreads();
    /* Blocks this expert needs. A zero-row run needs none, and asking for one would put an empty
     * block in the map that the GEMM would then have to recognise. */
    for (uint32_t e = tid; e < n_expert; e += blockDim.x) {
        const uint32_t run = (uint32_t)(s_off[e + 1] - s_off[e]);
        s_cnt[e] = (run + FM - 1) / FM;
    }
    /* The tail of dropped slots, carried as one more "expert" so the scan places it too. */
    const uint32_t live = (uint32_t)s_off[n_expert];
    if (tid == 0) s_cnt[n_expert] = T > live ? (T - live + FM - 1) / FM : 0u;
    __syncthreads();

    /* Exclusive scan, one thread. kMaxExpert + 1 adds is nothing beside the launch, and a parallel
     * scan here would be three barriers to save a microsecond on a kernel that runs once a layer. */
    if (tid == 0) {
        uint32_t acc = 0;
        for (uint32_t e = 0; e <= n_expert; ++e) { const uint32_t c = s_cnt[e]; s_cnt[e] = acc; acc += c; }
        map[0] = acc < cap ? acc : cap;      /* the count, clamped to what the grid can serve */
    }
    __syncthreads();

    for (uint32_t e = tid; e < n_expert; e += blockDim.x) {
        const uint32_t run = (uint32_t)(s_off[e + 1] - s_off[e]);
        uint32_t o = s_cnt[e];
        for (uint32_t r = 0; r < run; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = e;
            map[1 + 2 * o + 1] = (uint32_t)s_off[e] + r;
        }
    }
    if (tid == 0) {
        uint32_t o = s_cnt[n_expert];
        for (uint32_t r = live; r < T; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = kMapTail;
            map[1 + 2 * o + 1] = r;
        }
    }
}

/* WHICH FORM A LAUNCH TAKES: 0 is the rule; for a test, 1 pins the run-aligned prefill form, 2 the
 * (tile, run) decode form and 3 the uniform prefill grid, whatever the band. r4d_selftest --perfmoe
 * times each on one input and compares it with form 1: all three must match byte for byte. */
static int g_moe_prefill_form = 0;

/* Both instantiations, from one body. The only things that differ are the weight table's dtype and
 * its two pitches, so the fp8 launcher and the w4 one are the same forty lines with three values
 * swapped -- and writing them twice is how a shape check drifts between two paths that must agree.
 *
 * `ncls` is the op's class count: 2 for `moe_gemm_q`, whose odd class is operands 6 and 7 at
 * N_odd/K_odd, and 4 for `moe_gemm_q_mod4`, whose classes 1..3 are operands 6/7, 8/9 and 10/11 at
 * N_1/K_1, N_2/K_2 and N_3/K_3, with `y` after them at 12. Everything else is one body. */
template <int W4>
static int moe_gemm_launch(const RadArgs* a, const R4dSink& sk, int ncls = 2) {
    int emit_rc = RAD_OK;
    if (ncls != 2 && ncls != 4) return R4D_NO(RAD_E_INVAL);
    const RadTensor* Aq     = r4d_opt(a, 0);
    const RadTensor* As     = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 4);
    const RadTensor* off    = r4d_opt(a, 5);
    const RadTensor* Wc[4]  = { r4d_opt(a, 2), r4d_opt(a, 6),
                                ncls == 4 ? r4d_opt(a, 8) : nullptr,
                                ncls == 4 ? r4d_opt(a, 10) : nullptr };
    const RadTensor* Wsc[4] = { r4d_opt(a, 3), r4d_opt(a, 7),
                                ncls == 4 ? r4d_opt(a, 9) : nullptr,
                                ncls == 4 ? r4d_opt(a, 11) : nullptr };
    const RadTensor* Y      = r4d_opt(a, ncls == 4 ? 12 : 8);
    if (!Aq || !As || !sorted || !off || !Y) return R4D_NO(RAD_E_INVAL);
    for (int q = 0; q < ncls; ++q)
        if (!Wc[q] || !Wsc[q]) return R4D_NO(RAD_E_INVAL);
    if (!Aq->data || !As->data || !sorted->data || !off->data || !Y->data)
        return R4D_NO(RAD_E_INVAL);

    /* Each class's weight geometry, the others defaulting to class 0's; the output is as wide as
     * the widest class and the activation as long as the longest K. */
    static const char* const kNk[4][2] = { { "N", "K" }, { "N_1", "K_1" }, { "N_2", "K_2" },
                                           { "N_3", "K_3" } };
    long long Nc[4] = { 0, 0, 0, 0 }, Kc[4] = { 0, 0, 0, 0 };
    Nc[0] = r4d_geti_or(a, "N", 0);
    Kc[0] = r4d_geti_or(a, "K", 0);
    for (int q = 1; q < ncls; ++q) {
        Nc[q] = r4d_geti_or(a, ncls == 4 ? kNk[q][0] : "N_odd", Nc[0]);
        Kc[q] = r4d_geti_or(a, ncls == 4 ? kNk[q][1] : "K_odd", Kc[0]);
    }
    const long long parts = r4d_geti_or(a, "parts", 1);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    const long long group = r4d_geti_or(a, "group", kBlock);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    if (group != (long long)kBlock) return R4D_NO(RAD_E_SHAPE);
    long long N = 0, K = 0;
    for (int q = 0; q < ncls; ++q) {
        if (Nc[q] <= 0 || Kc[q] <= 0) return R4D_NO(RAD_E_SHAPE);
        /* A part of every class is whole 64-column tiles, so a tile is all weight or all padding. */
        if (Kc[q] % kBlock || Nc[q] % (64 * parts)) return R4D_NO(RAD_E_SHAPE);
        if (Nc[q] > N) N = Nc[q];
        if (Kc[q] > K) K = Kc[q];
    }
    if (Aq->dtype != RAD_F8E4M3 || As->dtype != RAD_F32 || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = r4d_rows_pitch(Aq, K, &M);
    if (a_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Ms = 0;
    const long long as_ld = r4d_rows_pitch(As, K / kBlock, &Ms);
    if (as_ld < 0 || Ms < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);

    /* WHICH ROWS THE ACTIVATION HAS. Checked, not inferred: an operand a row short is the caller
     * having handed the down projection the gate_up projection's buffer, and refusing it by name
     * here is the only place that reads as what it is. */
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);

    /* WHICH TABLE THE CODES INDEX, read off the dtype the layout hook accepted the weight under:
     * w4nla8h's levels, w5nl64a8h's magnitudes, or two's complement for w4a8 and w4a8h. The schema
     * admits no other. */
    const char* wdt = r4d_gets(a, "dtype");
    const uint32_t* tab = W4 == 4 ? kW5nlMagPerm
                        : (W4 && wdt && (!std::strcmp(wdt, "w4nla8h") ||
                                         !std::strcmp(wdt, "w4nl64a8h") ||
                                         !std::strcmp(wdt, "w4nl32a8h"))) ? kW4nlPerm : kW4IntPerm;

    /* One class's tables. THE WEIGHT PITCH THE KERNEL WANTS IS IN BYTES, and for a 4-bit plane
     * that is half the element count, for the 5-bit one five eighths. `wtab_of` reports the
     * declared element stride, so the scaling happens here, once, where the forms are side by
     * side. */
    auto cls_of = [&](const RadTensor* w, const RadTensor* ws, long long cnt, long long n,
                      long long k, MoeCls* c) -> int {
        long long w_ld = 0, ws_ld = 0;
        size_t w_s0 = 0, s_s0 = 0;
        int rc = wtab_of(w, W4 == 4 ? (uint32_t)R4D_DT_MOEW5
                            : W4 ? (uint32_t)R4D_DT_MOEW4 : (uint32_t)RAD_F8E4M3,
                         W4 == 4 ? 5 : W4 ? 4 : 8, cnt, n, k, &c->w, &w_s0, &w_ld);
        if (rc != RAD_OK) return rc;
        if constexpr (W4 >= 2) {
            /* One E4M3 byte per (row, group of 64 or 32 K), and the stored row is exactly those
             * bytes. THE PITCH IS DERIVED, NOT READ: the scale view selects two planes, so the
             * engine declares it at the weight's logical [n, k] and hands it over at that shape --
             * its row stride is k, the stored row k/64 or k/32 (r4d_layout.cpp). A tile's scales
             * are one sixteen- or thirty-two-bit word, so the kernel's pitch is k/128 words. */
            rc = wtab_of(ws, RAD_F8E4M3, 8, cnt, n, k / (kBlock / (W4 == 3 ? 4 : 2)), &c->s,
                         &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
            /* A 5-bit row is its nibbles then its sign bits, so its pitch is whole bytes of both;
             * the kernel finds the signs K/2 bytes in. */
            if (W4 == 4 ? (w_ld % 8 || k % 32) : (w_ld % 2)) return R4D_NO(RAD_E_STRIDE);
            w_ld = W4 == 4 ? w_ld / 8 * 5 : w_ld / 2;
            ws_ld = k / kBlock;
        } else if constexpr (W4) {
            /* The scale plane is one bf16 per (row, group of 128 K) -- not per 128x128 tile. */
            rc = wtab_of(ws, RAD_BF16, 16, cnt, n, k / kBlock, &c->s, &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
            if (w_ld % 2) return R4D_NO(RAD_E_STRIDE);
            w_ld /= 2;
        } else {
            rc = wtab_of(ws, RAD_BF16, 16, cnt, (n + kBlock - 1) / kBlock, k / kBlock, &c->s,
                         &s_s0, &ws_ld);
            if (rc != RAD_OK) return rc;
        }
        /* A stacked entry is one expert's plane, far below 4 GiB; a pointer table is 0. */
        if (w_s0 > 0xFFFFFFFFull || s_s0 > 0xFFFFFFFFull) return R4D_NO(RAD_E_STRIDE);
        c->w_s0 = (uint32_t)w_s0;
        c->s_s0 = (uint32_t)s_s0;
        c->n = (uint32_t)n;
        c->k = (uint32_t)k;
        c->w_ld = (uint32_t)w_ld;
        c->s_ld = (uint32_t)ws_ld;
        return RAD_OK;
    };
    /* Class q holds the experts e with e % ncls == q, entry e / ncls: ceil((n - q) / ncls) of
     * them. At two classes the last two are never read and stay zero. */
    MoeCls cl[4]{};
    for (int q = 0; q < ncls; ++q) {
        const int rc = cls_of(Wc[q], Wsc[q], (n_expert + ncls - 1 - q) / ncls, Nc[q], Kc[q],
                              &cl[q]);
        if (rc != RAD_OK) return rc;
    }
    const MoeCls& ce = cl[0];
    const MoeCls& co = cl[1];
    const MoeCls& c2 = cl[2];
    const MoeCls& c3 = cl[3];
    const uint32_t cshift = ncls == 4 ? 2u : 1u;
    if (parts > 0xFFFFFF) return R4D_NO(RAD_E_SHAPE);
    const uint32_t mode = (a_sorted ? 1u : 0u) | (cshift << 1) | ((uint32_t)parts << 8);
    MoeTab tabv{};
    for (int i = 0; i < 4; ++i) tabv.v[i] = tab[i];

    /* THE BAND: a decode step takes the (tile, run) form -- one 16-row tile of runs spread over
     * blocks, see the kernel's ES note -- and anything with runs of several rows takes the aligned
     * 32-row form. r4d_moe_decode_band has the crossover. */
    const uint32_t ny = (uint32_t)(N / 64);
    const int form = g_moe_prefill_form;
    if ((form == 0 && r4d_moe_decode_band(T, n_expert)) || form == 2) {
        /* THE DECODE GRID IS (tile, run) WIDE -- see the kernel's ES note. One tile of rows is all
         * a decode step has, so without this the whole layer runs on N/64 workgroups. */
        const unsigned tiles = (unsigned)((T + 15) / 16);
        const unsigned gx = (tiles <= 1) ? (unsigned)T : (tiles * 16u);
        emit_rc = R4D_EMIT(sk, (moe_gemm_q_decode_kernel<W4>),
                           dim3(gx, ny), dim3(4 * kWave), 0,
                           (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                           (const int32_t*)sorted->data, (const int32_t*)off->data,
                           (uint16_t*)Y->data, nullptr,
                           (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                           (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
    } else {
        /* THE PREFILL FORM IS ALIGNED TO RUNS when the scratch for a map is there, and uniform
         * when it is not. A caller that never asked the scratch hook -- a tool replaying a
         * recording, a plugin issuing this op by hand -- gets the uniform grid and the same
         * answer: a run that straddles a tile is staged by both, and nothing else differs.
         *
         * The map's builder scans the offsets in LDS tables of kMaxExpert + 1 entries, so a wider
         * expert count takes the uniform grid whatever scratch it was handed -- the GEMM itself
         * reads `off` from global memory and serves any count. */
        const long long cap = r4d_blockmap_cap(T, n_expert, kPrefillRows);
        uint32_t* map = (form != 3 && n_expert <= (long long)kMaxExpert && a->scratch &&
                         a->scratch_bytes >= r4d_blockmap_bytes(cap))
                            ? (uint32_t*)a->scratch : nullptr;
        /* The 128-column tile where no column tile straddles two parts of the weight. */
        const bool wide_n = (N / parts) % kPrefillCols == 0;
        const unsigned nyw = (unsigned)(N / kPrefillCols);
        if (map) {
            emit_rc = R4D_EMIT(sk, moe_blockmap_kernel, dim3(1), dim3(256), 0,
                               (const int32_t*)off->data, (uint32_t)n_expert, (uint32_t)T,
                               (uint32_t)kPrefillRows, (uint32_t)cap, map);
            if (emit_rc != RAD_OK) return emit_rc;
            if (wide_n)
                emit_rc = R4D_EMIT(sk, (moe_gemm_q_kernel<kPrefillRows / 16, kPrefillCols / 16, 2, 4, W4, false, true, true>),
                                   dim3((unsigned)cap, nyw), dim3(8 * kWave), 0,
                                   (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                                   (const int32_t*)sorted->data, (const int32_t*)off->data,
                                   (uint16_t*)Y->data, (const uint32_t*)map,
                                   (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                                   (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
            else
            emit_rc = R4D_EMIT(sk, (moe_gemm_q_kernel<kPrefillRows / 16, 4, 2, 2, W4, false, true, true>),
                               dim3((unsigned)cap, ny), dim3(4 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, (const uint32_t*)map,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
        } else if (wide_n) {
            emit_rc = R4D_EMIT(sk, (moe_gemm_q_kernel<kPrefillRows / 16, kPrefillCols / 16, 2, 4, W4, false, false, true>),
                               dim3((unsigned)((T + kPrefillRows - 1) / kPrefillRows), nyw),
                               dim3(8 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, nullptr,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
        } else {
            emit_rc = R4D_EMIT(sk, (moe_gemm_q_kernel<kPrefillRows / 16, 4, 2, 2, W4, false, false, true>),
                               dim3((unsigned)((T + kPrefillRows - 1) / kPrefillRows), ny),
                               dim3(4 * kWave), 0,
                               (const uint8_t*)Aq->data, (const float*)As->data, ce, co, c2, c3,
                               (const int32_t*)sorted->data, (const int32_t*)off->data,
                               (uint16_t*)Y->data, nullptr,
                               (uint32_t)T, (uint32_t)N, (uint32_t)n_expert, (uint32_t)top_k,
                               (uint32_t)a_ld, (uint32_t)as_ld, (uint32_t)y_ld, mode, tabv);
        }
    }
    if (emit_rc != RAD_OK) return emit_rc;
    if (!sk.desc) R4D_MOE_CHK();
    return RAD_OK;
}

/* Bytes for the run map, and zero everywhere it is not used.
 *
 * T IS COMPUTED FROM THE PARAMETERS, not from an operand, because a scratch hook is called at
 * DECLARE with the band's shape and the tensors need not be bound yet. `M * top_k` is the routed
 * row count by definition -- the same quantity the launcher reads off Y's row pitch -- so the two
 * agree at every shape that reaches the aligned form.
 *
 * IF THEY EVER DISAGREE THE FAILURE IS SAFE BY CONSTRUCTION, and that is deliberate rather than
 * lucky: the launcher checks the scratch it was actually handed against the size the map needs and
 * runs the uniform grid when it comes up short. A hook that underestimates costs the optimisation,
 * not the arena.
 */
static int64_t moe_gemm_map_scratch(const RadArgs* a) {
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long M        = r4d_geti_or(a, "M", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || M <= 0) return 0;
    const long long T = M * top_k;
    if (r4d_moe_decode_band(T, n_expert)) return 0;      /* the (tile, run) grid needs no map */
    return r4d_blockmap_bytes(r4d_blockmap_cap(T, n_expert, kPrefillRows));
}

extern "C" int64_t r4d_rad_moe_gemm_q_scratch(const RadArgs* a)    { return moe_gemm_map_scratch(a); }
extern "C" int64_t r4d_rad_moe_gemm_w4a8_scratch(const RadArgs* a) { return moe_gemm_map_scratch(a); }

extern "C" void r4d_moe_gemm_pin_prefill(int form) { g_moe_prefill_form = form; }

extern "C" int r4d_rad_moe_gemm_q(const RadArgs* a, RadStream s) {
    return moe_gemm_launch<0>(a, r4d_stream_sink(s));
}

/* The four-bit forms -- w4nl64a8h's and w4nl32a8h's E4M3 scale a 64 or 32 of K, or the others'
 * bf16 a 128 -- and the five-bit w5nl64a8h. */
static int rad_moe_gemm_w4a8_body(const RadArgs* a, const R4dSink& sk) {
    const char* wdt = r4d_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8h")) return moe_gemm_launch<2>(a, sk);
    if (wdt && !std::strcmp(wdt, "w4nl32a8h")) return moe_gemm_launch<3>(a, sk);
    if (wdt && !std::strcmp(wdt, "w5nl64a8h")) return moe_gemm_launch<4>(a, sk);
    return moe_gemm_launch<1>(a, sk);
}
R4D_ENTRY(r4d_rad_moe_gemm_w4a8, rad_moe_gemm_w4a8_body)

/* `moe_gemm_q_mod4`: the same launches over four classes, by the expert's number mod 4. The run
 * map and its scratch are the two-class op's -- they depend on the offsets and nothing else. */
extern "C" int r4d_rad_moe_gemm_q_mod4(const RadArgs* a, RadStream s) {
    return moe_gemm_launch<0>(a, r4d_stream_sink(s), 4);
}
static int rad_moe_gemm_w4a8_mod4_body(const RadArgs* a, const R4dSink& sk) {
    const char* wdt = r4d_gets(a, "dtype");
    if (wdt && !std::strcmp(wdt, "w4nl64a8h")) return moe_gemm_launch<2>(a, sk, 4);
    if (wdt && !std::strcmp(wdt, "w4nl32a8h")) return moe_gemm_launch<3>(a, sk, 4);
    if (wdt && !std::strcmp(wdt, "w5nl64a8h")) return moe_gemm_launch<4>(a, sk, 4);
    return moe_gemm_launch<1>(a, sk, 4);
}
R4D_ENTRY(r4d_rad_moe_gemm_w4a8_mod4, rad_moe_gemm_w4a8_mod4_body)

/* `moe_gemm` at bf16: operands a, w (one table), sorted_tok, expert_offset -> y. The band and the
 * three forms are moe_gemm_launch's and are chosen by the same rule from the same arguments, so the
 * scratch hook that sizes the run map is the same function too.
 *
 * EVERY LOAD IS SIXTEEN BYTES, and that is what the alignment checks below are for: a row pitch
 * that is not a multiple of eight elements, or a stacked plane whose experts do not start on a
 * sixteen-byte boundary, would put a chunk across two lines. A pointer table's entries cannot be
 * checked from here -- they live in device memory -- and every allocator that fills one hands out
 * movement-unit-aligned addresses. */
static int moe_gemm_bf16_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const RadTensor* A      = r4d_opt(a, 0);
    const RadTensor* W      = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 2);
    const RadTensor* off    = r4d_opt(a, 3);
    const RadTensor* Y      = r4d_opt(a, 4);
    if (!A || !W || !sorted || !off || !Y) return R4D_NO(RAD_E_INVAL);
    if (!A->data || !sorted->data || !off->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    const char* dt = r4d_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "bf16")) return R4D_NO(RAD_E_DTYPE);

    const long long N = r4d_geti_or(a, "N", 0);
    const long long K = r4d_geti_or(a, "K", 0);
    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (N <= 0 || K <= 0 || n_expert <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    /* A column tile is 64 wide and every tile is all weight; a K pass is 64 deep. */
    if (N % 64 || K % (long long)kBK16) return R4D_NO(RAD_E_SHAPE);
    if (A->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || off->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    long long T = 0;
    const long long y_ld = r4d_rows_pitch(Y, N, &T);
    if (y_ld < 0 || T <= 0) return R4D_NO(RAD_E_STRIDE);
    long long M = 0;
    const long long a_ld = r4d_rows_pitch(A, K, &M);
    if (a_ld < 0 || M <= 0 || a_ld % 8 || ((uintptr_t)A->data % 16)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sorted) < T || r4d_numel(off) < n_expert + 1) return R4D_NO(RAD_E_SHAPE);

    /* WHICH ROWS THE ACTIVATION HAS, checked rather than inferred -- see moe_gemm_launch. */
    const char* ord = r4d_gets(a, "a_order");
    const bool a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !a_sorted && std::strcmp(ord, "token")) return R4D_NO(RAD_E_UNSUPPORTED);
    if (M < (a_sorted ? T : (T + top_k - 1) / top_k)) return R4D_NO(RAD_E_SHAPE);

    const void* const* tab = nullptr;
    size_t w_s0 = 0;
    long long w_ld = 0;
    int rc = wtab_of(W, RAD_BF16, 16, n_expert, N, K, &tab, &w_s0, &w_ld);
    if (rc != RAD_OK) return rc;
    if (w_ld % 8 || w_s0 % 16) return R4D_NO(RAD_E_STRIDE);

    const uint32_t ny = (uint32_t)(N / 64);
    const int form = g_moe_prefill_form;
    if ((form == 0 && r4d_moe_decode_band(T, n_expert)) || form == 2) {
        const unsigned tiles = (unsigned)((T + 15) / 16);
        const unsigned gx = (tiles <= 1) ? (unsigned)T : (tiles * 16u);
        emit_rc = R4D_EMIT(sk, (moe_gemm_bf16_kernel<1, 4, 1, 4, true>),
                           dim3(gx, ny), dim3(4 * kWave), 0,
                           (const uint16_t*)A->data, tab, w_s0,
                           (const int32_t*)sorted->data, (const int32_t*)off->data,
                           (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)K,
                           (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)(a_sorted ? 1 : 0),
                           (uint32_t)a_ld, (uint32_t)w_ld, (uint32_t)y_ld, nullptr);
    } else {
        const long long cap = r4d_blockmap_cap(T, n_expert, kPrefillRows);
        uint32_t* map = (form != 3 && n_expert <= (long long)kMaxExpert && a->scratch &&
                         a->scratch_bytes >= r4d_blockmap_bytes(cap))
                            ? (uint32_t*)a->scratch : nullptr;
        /* The 128-column, eight-wave tile wherever the width allows it, as the fp8 forms do. */
        const bool wide_n = N % kPrefillCols == 0;
        const unsigned nyw = (unsigned)(N / kPrefillCols);
        const unsigned gx = map ? (unsigned)cap : (unsigned)((T + kPrefillRows - 1) / kPrefillRows);
        if (map) {
            emit_rc = R4D_EMIT(sk, moe_blockmap_kernel, dim3(1), dim3(256), 0,
                               (const int32_t*)off->data, (uint32_t)n_expert, (uint32_t)T,
                               (uint32_t)kPrefillRows, (uint32_t)cap, map);
            if (emit_rc != RAD_OK) return emit_rc;
        }
#define R4D_MOE16(AL_, NTL_, WGN_, NYG_)                                                         \
        R4D_EMIT(sk, (moe_gemm_bf16_kernel<kPrefillRows / 16, NTL_, 2, WGN_, false, AL_, true>),   \
                 dim3(gx, NYG_), dim3(2 * WGN_ * kWave), 0,                                      \
                 (const uint16_t*)A->data, tab, w_s0,                                            \
                 (const int32_t*)sorted->data, (const int32_t*)off->data,                        \
                 (uint16_t*)Y->data, (uint32_t)T, (uint32_t)N, (uint32_t)K,                      \
                 (uint32_t)n_expert, (uint32_t)top_k, (uint32_t)(a_sorted ? 1 : 0),              \
                 (uint32_t)a_ld, (uint32_t)w_ld, (uint32_t)y_ld, (const uint32_t*)map)
        if (map)  emit_rc = wide_n ? R4D_MOE16(true, kPrefillCols / 16, 4, nyw)
                                   : R4D_MOE16(true, 4, 2, ny);
        else      emit_rc = wide_n ? R4D_MOE16(false, kPrefillCols / 16, 4, nyw)
                                   : R4D_MOE16(false, 4, 2, ny);
#undef R4D_MOE16
    }
    if (emit_rc != RAD_OK) return emit_rc;
    if (!sk.desc) R4D_MOE_CHK();
    return RAD_OK;
}
R4D_ENTRY(r4d_rad_moe_gemm_bf16, moe_gemm_bf16_body)

extern "C" int64_t r4d_rad_moe_gemm_bf16_scratch(const RadArgs* a) {
    return moe_gemm_map_scratch(a);
}

extern "C" int64_t r4d_rad_moe_gather_scratch(const RadArgs* a) {
    const long long M = r4d_geti_or(a, "M", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 1);
    if (M <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    return M * top_k * (long long)sizeof(int32_t);
}

extern "C" int r4d_rad_moe_gather(const RadArgs* a, RadStream s) {
    const RadTensor* ye     = r4d_opt(a, 0);
    const RadTensor* ew     = r4d_opt(a, 1);
    const RadTensor* sorted = r4d_opt(a, 2);
    const RadTensor* sh     = r4d_opt(a, 3);
    const RadTensor* sg     = r4d_opt(a, 4);
    const RadTensor* Y      = r4d_opt(a, 5);
    if (!ye || !ew || !sorted || !Y) return R4D_NO(RAD_E_INVAL);
    if (!ye->data || !ew->data || !sorted->data || !Y->data) return R4D_NO(RAD_E_INVAL);
    const long long n = r4d_geti_or(a, "n", 0);
    const long long top_k = r4d_geti_or(a, "top_k", 0);
    if (n <= 0 || top_k <= 0) return R4D_NO(RAD_E_SHAPE);
    if (ye->dtype != RAD_BF16 || ew->dtype != RAD_BF16 || Y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (sorted->dtype != RAD_I32 || !r4d_contig(sorted)) return R4D_NO(RAD_E_STRIDE);
    /* The shared arm is a PAIR: the plane and its per-row gate, or neither. */
    if ((sh != nullptr) != (sg != nullptr)) return R4D_NO(RAD_E_INVAL);
    if (sh && (sh->dtype != RAD_BF16 || sg->dtype != RAD_BF16)) return R4D_NO(RAD_E_DTYPE);

    long long M = 0;
    const long long y_ld = r4d_rows_pitch(Y, n, &M);
    if (y_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Te = 0;
    const long long ye_ld = r4d_rows_pitch(ye, n, &Te);
    if (ye_ld < 0) return R4D_NO(RAD_E_STRIDE);
    long long Mw = 0;
    const long long ew_ld = r4d_rows_pitch(ew, top_k, &Mw);
    if (ew_ld < 0 || Mw < M) return R4D_NO(RAD_E_STRIDE);
    const long long T = M * top_k;
    if (Te < T || r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);


    /* SELF: small enough that the gather builds its own slot table and the two prepass kernels
     * are not launched at all -- see gather_kernel. The bound is on T because every one of the
     * M*gy blocks repeats the scan: at a decode step that is 4*10 blocks over 40 entries, and at
     * the cap it is still under half a megabyte of L2 traffic, against two dispatches saved on
     * every MoE layer of every step. A prefill chunk is far past it and takes the prepass. */
    const bool self = T <= (long long)kGatherSelfT && top_k <= (long long)kGatherSlots;

    const int64_t need = self ? 0 : T * (long long)sizeof(int32_t);
    int32_t* inv = nullptr;
    if (!self) {
        if (!a->scratch || a->scratch_bytes < need) return R4D_NO(RAD_E_SCRATCH);
        inv = (int32_t*)a->scratch;
        const unsigned blocks = (unsigned)((T + 255) / 256);
        /* TWO PASSES AND NOT ONE: a slot no sorted row claims must read -1, and the clear cannot
         * be fused into the scatter because a thread does not know which entries nobody will
         * write. Both are T/256 workgroups of trivial work; the launch is the cost. */
        hipLaunchKernelGGL(gather_inv_kernel, dim3(blocks), dim3(256), 0, (hipStream_t)s,
                           inv, (uint32_t)T);
        R4D_MOE_CHK();
        hipLaunchKernelGGL(gather_scatter_kernel, dim3(blocks), dim3(256), 0, (hipStream_t)s,
                           (const int32_t*)sorted->data, inv, (uint32_t)T);
        R4D_MOE_CHK();
    }
    long long shrows = 0, sgrows = 0, sh_ld = 0, sg_ld = 0;
    int sig = 0;
    if (sh) {
        sh_ld = r4d_rows_pitch(sh, n, &shrows);
        sg_ld = r4d_rows_pitch(sg, 1, &sgrows);
        if (sh_ld < 0 || sg_ld < 0 || shrows < M || sgrows < M) return R4D_NO(RAD_E_STRIDE);
        const char* const act = r4d_gets(a, "act");
        if (act && !std::strcmp(act, "sigmoid"))   sig = 1;
        else if (act && std::strcmp(act, "none"))  return R4D_NO(RAD_E_UNSUPPORTED);
    }

    /* EIGHT COLUMNS A LANE when every row is 16-byte aligned and the slots fit the LDS table
     * (gather_vec_kernel; the same bits). 160 threads tile this model's n = 2560 in two blocks a
     * token; a small M keeps 64 so a decode step still spreads over enough workgroups. */
    const bool vec = top_k <= (long long)kGatherSlots && n % 8 == 0 && ye_ld % 8 == 0 &&
                     y_ld % 8 == 0 && r4d_aligned(ye->data, 16) && r4d_aligned(Y->data, 16) &&
                     (!sh || (sh_ld % 8 == 0 && r4d_aligned(sh->data, 16)));
    if (vec) {
        const long long n8 = n / 8;
        const bool wide = M >= 256 && n8 % 160 == 0;
        const unsigned nt = wide ? 160u : 64u;
        const unsigned gv = (unsigned)((n8 + nt - 1) / nt);
        if (wide)
            hipLaunchKernelGGL((gather_vec_kernel<160>), dim3((unsigned)M, gv), dim3(160), 0,
                               (hipStream_t)s,
                               (const uint16_t*)ye->data, (const uint16_t*)ew->data, inv,
                               (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,
                               (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,
                               self ? (const int32_t*)sorted->data : nullptr, (uint32_t)T,
                               sh ? (const uint16_t*)sh->data : nullptr,
                               sh ? (const uint16_t*)sg->data : nullptr,
                               (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
        else
            hipLaunchKernelGGL((gather_vec_kernel<64>), dim3((unsigned)M, gv), dim3(64), 0,
                               (hipStream_t)s,
                               (const uint16_t*)ye->data, (const uint16_t*)ew->data, inv,
                               (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,
                               (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,
                               self ? (const int32_t*)sorted->data : nullptr, (uint32_t)T,
                               sh ? (const uint16_t*)sh->data : nullptr,
                               sh ? (const uint16_t*)sg->data : nullptr,
                               (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
        R4D_MOE_CHK();
        return RAD_OK;
    }
    const unsigned gy = (unsigned)((n + 255) / 256);
    hipLaunchKernelGGL((gather_kernel<256>), dim3((unsigned)M, gy), dim3(256), 0, (hipStream_t)s,
                       (const uint16_t*)ye->data, (const uint16_t*)ew->data, inv,
                       (uint16_t*)Y->data, (uint32_t)M, (uint32_t)n, (uint32_t)top_k,
                       (uint32_t)ye_ld, (uint32_t)ew_ld, (uint32_t)y_ld,
                       self ? (const int32_t*)sorted->data : nullptr, (uint32_t)T,
                       sh ? (const uint16_t*)sh->data : nullptr,
                       sh ? (const uint16_t*)sg->data : nullptr,
                       (uint32_t)sh_ld, (uint32_t)sg_ld, sig);
    R4D_MOE_CHK();
    return RAD_OK;
}

extern "C" int r4d_rad_scale_rows(const RadArgs* a, RadStream s) {
    const RadTensor* x  = r4d_opt(a, 0);
    const RadTensor* sc = r4d_opt(a, 1);
    const RadTensor* ad = r4d_opt(a, 2);          /* optional: the fused gate-and-add form */
    const RadTensor* y  = r4d_opt(a, 3);
    if (!x || !sc || !y || !x->data || !sc->data || !y->data) return R4D_NO(RAD_E_INVAL);
    if (ad && !ad->data) ad = nullptr;
    if (x->dtype != RAD_BF16 || sc->dtype != RAD_BF16 || y->dtype != RAD_BF16)
        return R4D_NO(RAD_E_DTYPE);
    if (ad && ad->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    const long long n = r4d_geti_or(a, "n", 0);
    if (n <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Read as a NAME, not through r4d_act_code: that helper's vocabulary is silu/sigmoid and this
     * op's is sigmoid/none, so "none" would come back as "not in the list" and be indistinguishable
     * from a typo. An unknown name is refused rather than run as the other activation. */
    const char* act = r4d_gets(a, "act");
    const int sig = (act && !std::strcmp(act, "sigmoid")) ? 1 : 0;
    if (act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return R4D_NO(RAD_E_INVAL);

    long long M = 0;
    const long long y_ld = r4d_rows_pitch(y, n, &M);
    if (y_ld < 0 || M <= 0) return R4D_NO(RAD_E_STRIDE);
    long long Mx = 0;
    const long long x_ld = r4d_rows_pitch(x, n, &Mx);
    if (x_ld < 0 || Mx < M) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(sc) < M) return R4D_NO(RAD_E_SHAPE);
    long long a_ld = 0;
    if (ad) {
        long long Ma = 0;
        a_ld = r4d_rows_pitch(ad, n, &Ma);
        if (a_ld < 0 || Ma < M) return R4D_NO(RAD_E_STRIDE);
    }
    /* One scalar a row. The stream arrives either as [M] rank 1 or as [M, 1] rank 2, and the row
     * stride is dim 0's in both -- a rank-1 tensor's stride[0] is its element stride. */
    const long long s_ld = r4d_stride(sc, 0) > 0 ? r4d_stride(sc, 0) : 1;

    hipLaunchKernelGGL((scale_rows_kernel<256>), dim3((unsigned)M), dim3(256), 0, (hipStream_t)s,
                       (const uint16_t*)x->data, (const uint16_t*)sc->data,
                       ad ? (const uint16_t*)ad->data : nullptr, (uint16_t*)y->data,
                       (uint32_t)M, (uint32_t)n, (uint32_t)x_ld, (uint32_t)(s_ld ? s_ld : 1),
                       (uint32_t)a_ld, (uint32_t)y_ld, sig);
    R4D_MOE_CHK();
    return RAD_OK;
}

/* ================================================================== router GEMM + router_topk_scatter
 *
 * THE ROUTER, ITS SELECTION AND ITS SORT IN ONE LAUNCH -- and, where the layer has a shared expert,
 * that expert's one-column gate beside them. rad_block_moe_fp8.h declares it under
 * R4D_ROUTER_FUSED; this is the kernel and its shim.
 *
 * At a decode step the router is a [n_expert, n_embd] bf16 GEMM over a handful of rows, and what
 * follows it -- `router_topk_scatter` -- is one workgroup with almost no work in it. They are two
 * DEPENDENT dispatches on every routed layer, and the boundary between them is a command-processor
 * gap plus a second launch, on the critical path. The selection needs every logit of a row, so it
 * cannot run inside a column tile; it runs in whichever workgroup finishes LAST, which an arrival
 * counter names. That is the device-wide pattern gemm_fp8a8's split-K finisher and hc_up_mix's
 * quantiser already use, and its rule is theirs: one counter a (stream, output), self-resetting.
 *
 * The shared expert's gate is a gemm_nt at N = 1 over the SAME input, a launch of its own for one
 * output column. It rides along as one more workgroup of this grid (`skg` > 0), so the launch the
 * routed gate_up was overlapping disappears from the queue.
 *
 * BYTE-IDENTICAL TO THE LAUNCHES IT STANDS IN FOR, and the claim rests on two things:
 *
 *   - THE GEMM IS r4d_gemm_bf16_nt_m64_kernel'S ARITHMETIC, ELEMENT BY ELEMENT. An output element of
 *     that kernel is SK partial tiles, each ONE wave's WMMA chain over one contiguous K slice in
 *     ascending K, summed in ascending split order from 0.f and narrowed by the language's own
 *     conversion. Which block owns the tile, how many tiles a block holds (WV, MB) and how many
 *     waves a block has do not enter it -- only SK does. So the split counts are asked of the law
 *     the unfused launches ask (cfg_for with the same N-tile count: a 512-wide router and a
 *     one-column gate get different ones), and a wave here may run several (tile, split) items
 *     one after another: each item is the same chain from zero into an LDS slot of its own, so the
 *     sum sees exactly the operands it saw before.
 *   - THE SELECTION AND THE SORT ARE router_topk_scatter_kernel'S BODY: the same two device
 *     functions over the same logits, wave-local per row and stable without an atomic ordering
 *     anything, so the number of waves the last block happens to have moves no bit.
 *
 * A TUNED POINT for the unfused router would break the first of those: this row is NOTUNE and
 * always takes the law's split. No shipped configuration tunes these GEMMs.
 *
 * DECODE ONLY: at most kRgMaxM rows (four 16-row tiles), which is the skinny GEMM's decode band;
 * past it the router is the tiled kernel and the architecture issues the pair. */
constexpr long long kRgMaxM = 64;

namespace {

constexpr int kRgKU = 8;                  /* r4d_gemm_bf16_nt_m64's KU: the unroll, not the order */

typedef __bf16 rg_v8bf __attribute__((ext_vector_type(8)));
typedef float  rg_v8f  __attribute__((ext_vector_type(8)));
/* Global-address-space pointer for the fragment loads, as r4d_gemm_bf16_nt_m64.hip holds it. */
typedef __attribute__((address_space(1))) const unsigned long long* rg_gu2;

__device__ __forceinline__ rg_v8f rg_mma(rg_v8bf a, rg_v8bf b, rg_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
/* r4d_gemm_bf16_nt_m64.hip's fragU: k step `u` of a K-contiguous row, two runs of four. */
__device__ __forceinline__ rg_v8bf rg_fragU(rg_gu2 p, int u) {
    union { unsigned long long w[2]; rg_v8bf f; } x;
    x.w[0] = p[4 * u];
    x.w[1] = p[4 * u + 2];
    return x.f;
}
/* r4d_gdn_wmma.h's rowGo: this lane's row, clamped to `lim`, as a 32-bit element offset. */
__device__ __forceinline__ unsigned rg_rowGo(unsigned rowstride, int row0, int lim, int lane) {
    int r = row0 + (lane & 15);
    r = r < lim ? r : lim;
    return (unsigned)r * rowstride + 4u * (unsigned)(lane >> 4);
}

/* ONE (tile, split) ITEM: the 16 x 16 partial tile r4d_gemm_bf16_nt_m64_kernel's wave (n0, ks)
 * accumulates at MT = 1 -- the same loads, the same WMMA chain in the same order from zero. */
__device__ __forceinline__ rg_v8f rg_item(const uint16_t* __restrict__ A,
                                          const uint16_t* __restrict__ W, int M, int K, int N,
                                          int n0, int m0, int ks, int SK, int lane) {
    const int Ks = K / SK, ksteps = Ks / 16, kbase = ks * Ks;
    rg_v8f acc = (rg_v8f)(0.f);
    const unsigned woff = rg_rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
    const unsigned aoff = rg_rowGo((unsigned)K, m0, M - 1, lane) + (unsigned)kbase;
    if (n0 < N) {
        rg_gu2 pw = (rg_gu2)(const unsigned long long*)(W + woff);
        rg_gu2 pa = (rg_gu2)(const unsigned long long*)(A + aoff);
        int s = 0;
        for (; s + kRgKU <= ksteps; s += kRgKU) {
#pragma unroll
            for (int u = 0; u < kRgKU; u++) {
                const rg_v8bf b = rg_fragU(pw, u);
                acc = rg_mma(rg_fragU(pa, u), b, acc);
            }
            pw += kRgKU * 4;
            pa += kRgKU * 4;
        }
        for (; s < ksteps; s++) {
            const rg_v8bf b = rg_fragU(pw, 0);
            pw += 4;
            acc = rg_mma(rg_fragU(pa, 0), b, acc);
            pa += 4;
        }
    }
    return acc;
}

/* The launch geometry, one argument: `sk` the router's split and `tpb` its 16-column tiles a
 * block, `nrb` the router's blocks along x, `skg` the gate's split (0: no gate; its block is
 * x = nrb). Rows are tiled along y, one 16-row tile a block. */
struct RgGeom {
    uint32_t M, K, n_expert, n_local, top_k, lg_ld;
    int32_t  norm, base;
    uint32_t sk, tpb, nrb, skg;
};

/* `lg` is neither const nor __restrict__: every block writes it and the last one reads it back. */
template <uint32_t MAXOWN, uint32_t NW>
__global__ __launch_bounds__(NW * kWave) void router_gemm_topk_scatter_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ Wg, uint16_t* lg, uint16_t* __restrict__ gate,
    int32_t* __restrict__ ids, uint16_t* __restrict__ w, int32_t* __restrict__ sorted,
    int32_t* __restrict__ off, int32_t* __restrict__ cnt, unsigned int* __restrict__ ctr,
    RgGeom g, Protect prot) {
    /* The partial tiles of the GEMM, then -- in the last block only, after two barriers -- the
     * sort's LDS. The shim sizes it for the larger of the two. */
    extern __shared__ __align__(16) unsigned char rg_smem[];
    float* const red = (float*)rg_smem;
    const uint32_t tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;

    const bool      gb  = g.skg > 0 && blockIdx.x == g.nrb;
    const int       SK  = (int)(gb ? g.skg : g.sk);
    const uint32_t  nti = gb ? 1u : g.tpb;
    const int       N   = gb ? 1 : (int)g.n_expert;
    const uint32_t  nt0 = gb ? 0u : blockIdx.x * g.tpb;
    const uint16_t* Wb  = gb ? Wg : W;
    const int       m0  = (int)(blockIdx.y * 16u);

    const uint32_t items = nti * (uint32_t)SK;
    for (uint32_t it = wave; it < items; it += NW) {
        const uint32_t j = it / (uint32_t)SK, ks = it % (uint32_t)SK;
        const rg_v8f acc = rg_item(A, Wb, (int)g.M, (int)g.K, N, (int)((nt0 + j) * 16u), m0,
                                   (int)ks, SK, (int)lane);
        float* const slot = red + (size_t)it * 256;           /* it == j * SK + ks */
#pragma unroll
        for (int e = 0; e < 8; e++) slot[e * (int)kWave + (int)lane] = acc[e];
    }
    __syncthreads();
    /* THE REDUCTION, r4d_gemm_bf16_nt_m64_kernel's: the splits in ascending order from 0.f, and
     * the language's float-to-bf16 conversion -- not f32_to_bf16, which is the same only away
     * from NaN. */
    for (uint32_t idx = tid; idx < nti * 256u; idx += NW * kWave) {
        const uint32_t j = idx >> 8, o = idx & 255u;
        float v = 0.f;
        for (int s = 0; s < SK; s++) v += red[((size_t)j * (size_t)SK + (size_t)s) * 256 + o];
        const uint32_t l = o & 31u;
        const int n = (int)((nt0 + j) * 16u + (l & 15u));
        const int m = m0 + (int)(8u * (l >> 4) + (o >> 5));
        if (n < N && m < (int)g.M) {
            if (gb) ((__hip_bfloat16*)gate)[m] = (__hip_bfloat16)v;
            else    ((__hip_bfloat16*)lg)[(size_t)m * g.lg_ld + (size_t)n] = (__hip_bfloat16)v;
        }
    }

    /* ---- THE LAST BLOCK RANKS. Every store above is made visible device-wide before the
     * arrival is published, and the block that completes the count resets it for the next
     * launch and acquires the other blocks' logits. */
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_arrived;
    if (tid == 0)
        s_arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (s_arrived + 1u != gridDim.x * gridDim.y) return;
    if (tid == 0) __hip_atomic_store(ctr, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __threadfence();

    /* router_topk_scatter_kernel, from here to the end. */
    ScatterLds<NW>& sl = *reinterpret_cast<ScatterLds<NW>*>(rg_smem);
    for (uint32_t e = tid; e < g.n_local; e += blockDim.x) sl.cnt[e] = 0;
    const SlotSink snk{sl.e, g.base, g.n_local, prot};
    for (uint32_t m = wave; m < g.M; m += NW)
        router_topk_row<MAXOWN>(lg, ids, w, m, g.n_expert, g.top_k, g.lg_ld, g.norm, lane, snk);
    lds_barrier();
    scatter_small_place(sorted, off, cnt, g.M * g.top_k, g.n_local, sl);
}

}  /* namespace */

/* THE ARRIVAL COUNTER, one a (stream, logits buffer): ctx.h's rule TWO, and gemm_fp8a8's split_ctr
 * says why the output is part of the key. Launches that share a logits buffer on one stream are
 * ordered through it -- every routed layer writes the same plane -- so they reuse one counter.
 * Zeroed once and reset by the last block of every launch. Null when it cannot be allocated; the
 * op then refuses rather than computing without it. */
static unsigned int* router_fused_ctr(RadStream s, const void* out) {
    static std::mutex m;
    static std::map<std::pair<RadStream, const void*>, unsigned int*> blocks;
    std::lock_guard<std::mutex> lk(m);
    const auto key = std::make_pair(s, out);
    if (auto it = blocks.find(key); it != blocks.end()) return it->second;
    void* p = nullptr;
    if (hipMalloc(&p, 64) != hipSuccess) return nullptr;
    if (hipMemset(p, 0, 64) != hipSuccess) {
        (void)hipFree(p);
        return nullptr;
    }
    blocks.emplace(key, (unsigned int*)p);
    return (unsigned int*)p;
}

/* Operands: a, b (the router), gate_b? -> logits, expert_ids, expert_w, sorted_tok,
 * expert_offset, expert_count, gate?. The two optional ones come together or not at all. Every
 * check of the two shims it stands in for is made here too; nothing new is asserted beyond the
 * row bound. */
static int rad_router_gemm_topk_scatter_body(const RadArgs* a, const R4dSink& sk) {
    int emit_rc = RAD_OK;
    const RadTensor* x      = r4d_opt(a, 0);
    const RadTensor* wr     = r4d_opt(a, 1);
    const RadTensor* wg     = r4d_opt(a, 2);
    const RadTensor* lg     = r4d_opt(a, 3);
    const RadTensor* ids    = r4d_opt(a, 4);
    const RadTensor* w      = r4d_opt(a, 5);
    const RadTensor* sorted = r4d_opt(a, 6);
    const RadTensor* off    = r4d_opt(a, 7);
    const RadTensor* cnt    = r4d_opt(a, 8);
    const RadTensor* gate   = r4d_opt(a, 9);
    if (!x || !wr || !lg || !ids || !w || !sorted || !off || !cnt) return R4D_NO(RAD_E_INVAL);
    if ((wg != nullptr) != (gate != nullptr)) return R4D_NO(RAD_E_INVAL);

    const long long n_expert = r4d_geti_or(a, "n_expert", 0);
    const long long n_local  = r4d_geti_or(a, "n_local", n_expert);
    const long long top_k    = r4d_geti_or(a, "top_k", 0);
    const long long norm     = r4d_geti_or(a, "norm", 0);
    const long long base     = r4d_geti_or(a, "expert_base", 0);
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || top_k > n_expert)
        return R4D_NO(RAD_E_SHAPE);
    if (top_k > (long long)kWave) return R4D_NO(RAD_E_SHAPE);
    if (n_local <= 0 || n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    if (base < 0 || base + n_local > n_expert) return R4D_NO(RAD_E_SHAPE);
    Protect prot;
    const int prc = protect_of(a, n_local, &prot);
    if (prc != RAD_OK) return prc;
    if (x->dtype != RAD_BF16 || wr->dtype != RAD_BF16 || lg->dtype != RAD_BF16 ||
        w->dtype != RAD_BF16) return R4D_NO(RAD_E_DTYPE);
    if (wg && (wg->dtype != RAD_BF16 || gate->dtype != RAD_BF16)) return R4D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || sorted->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32) return R4D_NO(RAD_E_DTYPE);

    /* The GEMM's operands, as gemm_ops reads them: A [M, K] contiguous, the output contiguous,
     * A and the weights 16-byte aligned. */
    long long M = 0, K = 0;
    if (r4d_rows_of(x, &M, &K) != RAD_OK) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(x)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_geti_or(a, "K", K) != K) return R4D_NO(RAD_E_SHAPE);
    if (M > kRgMaxM) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_aligned(x->data, 16) || !r4d_aligned(wr->data, 16) ||
        (wg && !r4d_aligned(wg->data, 16))) return R4D_NO(RAD_E_ALIGN);
    if (r4d_numel(wr) < n_expert * K || (wg && r4d_numel(wg) < K)) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(lg) || r4d_numel(lg) < M * n_expert) return R4D_NO(RAD_E_STRIDE);
    if (gate && (!r4d_contig(gate) || r4d_numel(gate) < M)) return R4D_NO(RAD_E_STRIDE);

    const long long T = M * top_k;
    if (T > (long long)kScatterSmall) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(ids) || !r4d_contig(w) || !r4d_contig(sorted) || !r4d_contig(off) ||
        !r4d_contig(cnt)) return R4D_NO(RAD_E_STRIDE);
    if (r4d_numel(ids) < T || r4d_numel(w) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(sorted) < T) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(off) < n_local + 1 || r4d_numel(cnt) < n_local) return R4D_NO(RAD_E_SHAPE);

    /* THE SPLITS THE UNFUSED LAUNCHES TAKE: gemm_bf16_nt_m16/_m64 ask cfg_for with ceil(N/16)
     * tiles, so the router asks with its own width and the gate with one tile. */
    const long long ntiles = (n_expert + 15) / 16;
    Cfg cr{}, cg{};
    int rc = cfg_for(a, K, M, 16, BR_FLAT, &cr, ntiles);
    if (rc != RAD_OK) return rc;
    if (wg) {
        rc = cfg_for(a, K, M, 16, BR_FLAT, &cg, 1);
        if (rc != RAD_OK) return rc;
    }
    unsigned int* const ctr = router_fused_ctr(sk.stream, lg->data);
    if (!ctr) return R4D_NO(RAD_E_DEVICE);

    /* THE WAVE COUNT IS THE SELECTION'S, router_topk_scatter's rule: a wave a row up to 32. The
     * GEMM then takes as many 16-column tiles a block as fills those waves at its split. */
    const uint32_t nw = M <= 8 ? 8u : M <= 16 ? 16u : 32u;
    long long tpb = (long long)nw / cr.sk;
    if (tpb < 1) tpb = 1;
    if (tpb > ntiles) tpb = ntiles;
    const long long nrb = (ntiles + tpb - 1) / tpb;
    size_t lds = (size_t)(tpb * cr.sk) * 256 * sizeof(float);
    if (wg) lds = std::max(lds, (size_t)cg.sk * 256 * sizeof(float));
    const size_t sort_lds = nw == 8 ? sizeof(ScatterLds<8>)
                          : nw == 16 ? sizeof(ScatterLds<16>) : sizeof(ScatterLds<32>);
    lds = std::max(lds, sort_lds);

    RgGeom g{};
    g.M = (uint32_t)M;               g.K = (uint32_t)K;
    g.n_expert = (uint32_t)n_expert; g.n_local = (uint32_t)n_local;
    g.top_k = (uint32_t)top_k;       g.lg_ld = (uint32_t)n_expert;
    g.norm = (int32_t)norm;          g.base = (int32_t)base;
    g.sk = (uint32_t)cr.sk;          g.tpb = (uint32_t)tpb;
    g.nrb = (uint32_t)nrb;           g.skg = wg ? (uint32_t)cg.sk : 0u;
    const dim3 grid((unsigned)(nrb + (wg ? 1 : 0)), (unsigned)((M + 15) / 16));

#define R4D_RG_L(OWN, NWV)                                                                      \
    emit_rc = R4D_EMIT(sk, (router_gemm_topk_scatter_kernel<OWN, NWV>), grid,                   \
                       dim3((NWV) * kWave), lds,                                                \
                       (const uint16_t*)x->data, (const uint16_t*)wr->data,                     \
                       wg ? (const uint16_t*)wg->data : (const uint16_t*)nullptr,               \
                       (uint16_t*)lg->data, gate ? (uint16_t*)gate->data : (uint16_t*)nullptr,  \
                       (int32_t*)ids->data, (uint16_t*)w->data, (int32_t*)sorted->data,         \
                       (int32_t*)off->data, (int32_t*)cnt->data, ctr, g, prot)
    if (n_expert <= (long long)(8 * kWave)) {
        if      (nw == 8)  R4D_RG_L(8, 8);
        else if (nw == 16) R4D_RG_L(8, 16);
        else               R4D_RG_L(8, 32);
    } else {
        if      (nw == 8)  R4D_RG_L(16, 8);
        else if (nw == 16) R4D_RG_L(16, 16);
        else               R4D_RG_L(16, 32);
    }
#undef R4D_RG_L
    if (emit_rc != RAD_OK) return emit_rc;
    if (!sk.desc) R4D_MOE_CHK();
    return RAD_OK;
}
R4D_ENTRY(r4d_rad_router_gemm_topk_scatter, rad_router_gemm_topk_scatter_body)
