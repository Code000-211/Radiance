/* heat.h -- the heat engine: which conditional weights should be in VRAM, decided from the
 * router's observed choices.
 *
 * Layer offload gets an exact prefetch schedule, because declare enumerates the access sequence.
 * Routing (RAD_ACCESS_CONDITIONAL) is data-dependent, so expert offload needs a policy instead.
 * This file is that policy: from a tier and a heat per unit and a dispatch count, it proposes
 * swaps. It is pure host code with no device calls and no knowledge of slabs, streams or events
 * (that is mover.cpp), so the decision half, hysteresis included, is testable without a card.
 *
 * ---- one step behind ---------------------------------------------------------------------------
 *
 * Routing runs on device and never blocks the compute stream. The histogram is copied back
 * asynchronously and consumed on the next step (spec §5.5), so every decision is one step stale.
 * Ranking is unaffected, since a hot expert stays hot across many steps. But an expert that is not
 * resident this step cannot be fetched in time: it is served from where it lies, streamed from the
 * pinned pool or staged off the file tier. A last-moment fetch would have to fit between the
 * router's output and the expert GEMM's first block, which is microseconds. Hit rate is therefore
 * a property of the placement policy, and the run phase has no device round-trip.
 *
 * ---- why LFU with decay ------------------------------------------------------------------------
 *
 * In routing-trace replays, runtime placement beats static profile placement by a wide margin, and
 * the move rate bounds it, not the ranking: at the same swaps a step, an oracle that knows every
 * future routing scores no better (78.0% vs 78.4% at 64 swaps a step, Qwen3.8-Flash-Next TP1, 40%
 * of experts resident). Unrestricted Belady reaches 92% only by admitting every miss at once, which
 * here would copy again bytes the GEMM just streamed. The levers are the move budget
 * (RAD_PLACE_MOVES_PER_STEP), the slots behind it, and keeping a prompt's routing from swamping the
 * answer's (HeatParams::credit_cap), not the decay constant or the guards.
 */
#pragma once
#include "planner.h"

#include <string>
#include <vector>

namespace rad {

struct HeatParams {
    /* Applied to a layer's whole plane each time its histogram is credited. Every layer is
     * credited once a step, so planes decay in lockstep and heats stay comparable across layers,
     * which the global ranking needs. 0.95 is a half-life of about 14 steps. In replay at 64 swaps
     * a step, hit rate is flat from 0.8 to 0.95 and falls above it (77.1% at 0.98, 74.2% at 0.99):
     * a longer memory ranks experts the conversation has moved past above the current ones. */
    float decay = 0.95f;

    /* Promote only when the candidate beats the victim by this ratio, so the two ends of the
     * ranking do not trade places on noise. */
    float hysteresis = 1.25f;

    /* The additive half of the guard, in activations. It stops a swap against a zero-heat victim,
     * where any ratio passes; right after warmup that is most of the resident set.
     *
     * Do not raise it to save link traffic: traffic is U-shaped in this threshold. A declined
     * promotion comes back as a miss, streamed over the same link, so past the minimum a tighter
     * bar loses hit rate and saves no bytes. A swap costs its promotion: the link is near full
     * duplex (R9700: 28.5 GB/s up and 25.0 down at once, 28.8 up alone), so a demotion rides the
     * idle direction while a promotion shares the upstream with every miss. At the default budget,
     * lower values replay within 0.2 points of hit rate and promote a fifth more. */
    float min_gain = 1.5f;

    /* Swaps step() may propose per slab class each step, after every layer is credited. The value
     * and its reason are at RAD_PLACE_MOVES_PER_STEP, which also sizes the free slots a swap lands
     * in (shadow and turnover slots); a budget raised here without them is spent on
     * `starved_slab`. Promotions share the link with streamed misses, but each swap saves many
     * times its own bytes in misses.
     *
     * Per class because a rank holding an uneven slice of every expert has two classes of about
     * half an expert each; a shared budget would make residency converge half as fast. */
    int moves_per_step = RAD_PLACE_MOVES_PER_STEP;

    /* The most one histogram credits, in selections; a larger histogram is scaled to this total.
     * Decode routes a few rows a sequence and stays under it. A prefill chunk routes thousands
     * and would otherwise make the resident set the prompt's for dozens of steps, while prefill
     * gains little from residency (a chunk reads nearly every expert; the stager copies the rest).
     * Mixed-workload replay, Qwen3.8-Flash-Next TP1: decode hit rate 77.3% uncapped, 78.9% at
     * 1024, 79.0% at 512 or 256. */
    float credit_cap = 512.0f;

    /* Dispatches to watch before moving anything. 0, because min_gain already requires
     * `hot >= cold*hysteresis + 1.5`, which against a zero-heat victim is two tokens agreeing in
     * one dispatch; a warmup only delays convergence. */
    int warmup_dispatches = 0;
};

/* A promotion and its matching demotion. The mover issues two transfers, but they are decided
 * together: a promotion without its demotion would grow the resident set past the slab. */
struct Swap {
    int32_t promote = -1;   /* unit index */
    int32_t demote = -1;
};

/* One movable unit as step() ranks it: its heat, its layer's resident count, its index. */
struct HeatRanked { float h; int32_t rc; int32_t unit; };

struct HeatStats {
    uint64_t dispatches = 0;
    uint64_t proposed = 0, promotions = 0, demotions = 0;
    /* A subset of `promotions`: the ones Mover::fill issued into lent spare slots. A fill
     * materialises the plan's start placement -- it evicts nothing and only uses memory the
     * plan left free -- so the cost profile's drift evidence leaves it out until the start
     * placement is resolved (Engine::residency_moves). A swap's promotion is not one. */
    uint64_t fill_promotions = 0;
    /* Of those, the ones issued before this rank's first observation. Heat is what orders a
     * fill, and at zero dispatches every heat is zero, so only this fill is the plan's own free
     * slots by layer -- a function of the plan and the loan. A fill after an observation picks
     * its units from what the run has seen and is never the start placement. */
    uint64_t initial_fill_promotions = 0;

    /* The placement moves the drift evidence counts: every promotion and demotion but the
     * `fill_exempt` promotions that are the start placement materialising. */
    int64_t moves(int64_t fill_exempt) const {
        return (int64_t)promotions - fill_exempt + (int64_t)demotions;
    }
    /* Why a dispatch moved nothing. `refused` is the guard declining, the healthy steady state.
     * `no_candidate` is an empty end of the ranking (everything movable is resident, or nothing
     * is), a budget question rather than a policy one. */
    uint64_t refused = 0, no_candidate = 0;
    /* Progress or thrash. `distinct_promoted` counts first-time promotions (still discovering the
     * working set); `readmits` counts promotions of units demoted earlier, link traffic spent to
     * return to where it started. Readmits say whether the guards are set right. */
    uint64_t distinct_promoted = 0, readmits = 0;
    /* Routed experts and how many were resident, counted on the histogram pass observe() already
     * makes. A miss does not stall: the GEMM reads the expert from its tier, which for a
     * host-pinned unit crosses the link on the step's critical path. `stream_bytes` is that
     * traffic, the cost promotions exist to avoid. Weighted by the histogram's `weight`, since a
     * draft head's histogram stands for several routing passes; the hit rate is unaffected. */
    uint64_t routed = 0, routed_resident = 0, stream_bytes = 0;
};

class HeatEngine {
public:
    /* Binds to the plan. A static plan is a mode, not a failure: init returns RAD_OK and
     * enabled() is false. Under Config::deterministic the plan is static and step() proposes
     * nothing for the whole run (spec §17). */
    int init(const Plan& plan, const HeatParams& params);
    bool enabled() const { return enabled_; }

    /* ---- input: last step's routing histogram ------------------------------------------------
     *
     * `counts[e]` is how many of the step's tokens routed to expert e of this layer. The credit
     * is the count, not 1: a dispatch reads an expert once however many tokens chose it, but an
     * expert several tokens chose is far likelier to be chosen again soon.
     *
     * `weight` is how many of the layer's routing passes this histogram stands for (see
     * Ctx::last_histogram_reports): 1 for a non-speculative step, n_spec for a draft head. It
     * scales the credit, not the decay, because every layer still decays exactly once a step. */
    void observe(int32_t layer, const int32_t* counts, int32_t n_expert, float weight = 1.0f);
    /* The same, sparse: distinct expert ids and their counts. `counts` may be null for one each. */
    void observe_ids(int32_t layer, const int32_t* experts, const int32_t* counts, int32_t n);
    /* Starts fetching a histogram observe() will read. A row the card copied into host memory
     * misses on every line, so a caller that knows its next rows asks early. */
    static void prefetch(const int32_t* counts, int32_t n_expert) {
        for (int32_t j = 0; j < n_expert; j += 64 / (int32_t)sizeof(int32_t))
            __builtin_prefetch(counts + j);
    }

    /* ---- tier map, written by the mover when a move publishes, not when it is issued ---------
     *
     * Until publish both copies are valid and dispatch reads the old one, so the tier here must
     * match dispatch's view. `set_busy` is what keeps an in-flight unit from being picked again.
     * `fill` marks the promotion as the fill's (HeatStats::fill_promotions), `initial` as one
     * issued before the first observation (HeatStats::initial_fill_promotions). */
    void set_tier(int32_t unit, Tier t, bool fill = false, bool initial = false);
    void set_busy(int32_t unit, bool busy);
    Tier tier(int32_t unit) const { return tier_[(size_t)unit]; }
    bool busy(int32_t unit) const { return busy_[(size_t)unit] != 0; }

    /* The best swap available, or false when the guards refuse. */
    bool pick(Swap* out) const;

    /* One step's proposals, at most `moves_per_step` per class.
     *
     * Every unit in the result is marked busy. The caller must either issue the transfer
     * (set_tier() on publish, set_busy(false) after the quarantine) or abandon() the pair. A
     * dropped proposal leaves both units busy forever, with no counter to show it. */
    int step(std::vector<Swap>* out);
    void abandon(const Swap& s);

    float heat(int32_t unit) const { return heat_[(size_t)pos_[(size_t)unit]]; }
    /* Resident units in this unit's layer: the tie-break every residency decision needs (see
     * `res_count_`). `pick` uses it on an exact heat tie, and so must anything else that chooses. */
    int32_t layer_residents(int32_t unit) const {
        const int32_t l = (size_t)unit < layer_of_.size() ? layer_of_[(size_t)unit] : -1;
        return l >= 0 && (size_t)l < res_count_.size() ? res_count_[(size_t)l] : 0;
    }
    const HeatStats& stats() const { return stats_; }

    /* Exit banner, ending with the per-layer resident counts, which show whether every layer kept
     * its share of slots. */
    std::string report() const;

private:
    HeatParams params_;
    bool       enabled_ = false;
    int32_t    n_layer_ = 0;
    HeatStats  stats_;

    /* Heat by position, not by unit. Each layer owns a run of positions whose front is indexed by
     * expert: expert e of layer l is at slot_lo_[l] + e, whether or not the plan has a unit for it
     * or that unit may move. Then come the layer's other movable units, then padding to a multiple
     * of kLanes. After the layers: the movable units with no layer, padded to a screening block,
     * then everything else.
     *
     * This lines a histogram up with the positions it credits, so observe() walks counts[e] and
     * the heat side by side in fixed-width vector blocks, with no unit lookup or per-expert branch.
     * A position with no unit is a hole: never a candidate, never priced. `pos_` and `unit_at_`
     * translate (-1 for a hole); everything outside the position arrays is indexed by MoveUnit
     * index, as in the plan. */
    std::vector<float>   heat_;               /* by position */
    std::vector<int32_t> pos_;                /* unit -> position */
    std::vector<int32_t> unit_at_;            /* position -> unit, -1 for a hole */
    /* Per position, what step() ranks it as: 0 not a candidate (fixed, in flight, or a hole), 1 a
     * candidate off the device, 2 a candidate on it. Kept by set_tier and set_busy. */
    std::vector<uint8_t> rank_as_;
    /* Per position, what a routed read of it costs: `routed_as_` is 0 for a hole, 1 off the device
     * and 2 in VRAM, and `miss_bytes_` is the unit's bytes when it is off the device and 0 when it
     * is not. Kept by set_tier, so pricing a histogram is arithmetic on the pass that credits it. */
    std::vector<uint8_t> routed_as_;
    std::vector<int64_t> miss_bytes_;
    /* Slot s's positions are [slot_lo_[s], slot_lo_[s + 1]): layer s, and slot n_layer_ the
     * movable units with no layer. Every movable unit is below slot_lo_[n_layer_ + 1], which is a
     * whole number of screening blocks. The first n_ex_[l] positions of layer l are its experts. */
    std::vector<int32_t> slot_lo_;
    std::vector<int32_t> n_ex_;
    void set_place(int32_t unit);
    /* Unit bytes from the plan, so a miss is priced where it is detected. */
    std::vector<int64_t> bytes_;
    std::vector<Tier>    tier_;
    std::vector<uint8_t> busy_;
    std::vector<uint8_t> movable_;
    std::vector<int32_t> layer_of_;
    /* Bit 0: promoted before. Bit 1: demoted before. Feeds the readmit counters. */
    std::vector<uint8_t> ever_;

    /* Resident units per layer. A layer starved of slots misses on nearly every token however
     * good the ranking inside it is. Most resident units are cold, so exact `heat == 0.0f` ties
     * are common at both ends and the tie-break is effectively the policy; breaking it by index
     * would strip low-numbered layers first. Ties take the victim from the layer holding the most
     * and promote into the layer holding the fewest. */
    std::vector<int32_t> res_count_;
    /* A swap stays inside one slab class: each half uses a slot of its own class, so a pair across
     * classes would drain one class's spare slots into the other's until promotions starve. A rank
     * has one expert class when every expert is the same size, two when it holds an uneven slice
     * of each expert's width (three 128-column blocks of one parity, two of the other). `cls_` is
     * the dense class index, -1 for a unit that cannot move. */
    std::vector<int32_t> cls_;
    int32_t              n_cls_ = 1;
    /* step()'s two ends of the ranking, `moves_per_step` entries a class, kept between calls so a
     * step allocates nothing. */
    std::vector<HeatRanked> rank_hot_, rank_cold_;
    std::vector<int32_t>    n_hot_, n_cold_, m_of_;
    /* Per slot, its lowest movable unit index: the bound step()'s layer screen needs on a tie. */
    std::vector<int32_t> min_unit_;

    /* Each slot's two ends, over movable units not in flight: the hottest off the device and the
     * coldest on it, and how many of each. A dispatch that proposes nothing (nearly all of them
     * once residency settles) is then decided from one entry a layer, not a scan of every unit.
     * Slot n_layer_ is the movable units with no layer.
     *
     * observe() updates a layer's ends on the pass that credits it. A tier or busy change
     * invalidates the unit's slot, which step() rebuilds from its positions on its next read. */
    std::vector<float>   end_hot_, end_cold_;
    std::vector<int32_t> end_n_off_, end_n_on_;
    std::vector<int32_t> end_n_cold_;   /* units on the device at exactly the cold end's heat */
    std::vector<uint8_t> end_ok_;
    void end_rebuild(size_t slot);
    void end_invalidate(int32_t unit) {
        if (end_ok_.empty()) return;
        const int32_t l = (size_t)unit < layer_of_.size() ? layer_of_[(size_t)unit] : -1;
        end_ok_[l >= 0 ? (size_t)l : (size_t)n_layer_] = 0;
    }

    bool warm() const { return (int64_t)stats_.dispatches >= params_.warmup_dispatches; }
};

}  /* namespace rad */
