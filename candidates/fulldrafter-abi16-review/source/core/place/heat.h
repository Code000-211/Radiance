/* heat.h -- the heat engine: which conditional weight should be in VRAM, decided from what the
 * router actually did.
 *
 * EXPERT OFFLOAD IS A PREDICTION PROBLEM. Layer offload is not -- declare enumerated the whole
 * access sequence and the planner emits an exact prefetch schedule off it. Routing is
 * data-dependent, so for RAD_ACCESS_CONDITIONAL weights there is no schedule to emit and the only
 * thing left is a policy. This file is that policy and nothing else: it sees a tier per unit, a
 * heat per unit and a dispatch count, and it answers one question -- "swap these two."
 *
 * Deliberately pure host code with no device call in it and no knowledge of slabs, streams or
 * events. Everything about how the bytes get there is mover.cpp. That split is what lets the
 * entire decision half be tested on a machine with no card, which is the only reason the
 * hysteresis behaviour below is checkable at all.
 *
 * ---- WHAT IT COSTS TO BE ONE STEP BEHIND, HONESTLY -------------------------------------------
 *
 * Routing is computed on device and never blocks the compute stream. The histogram is copied back
 * asynchronously and this engine consumes it on the NEXT step (spec §5.5), so every placement
 * decision is one step stale.
 *
 * For heat-based promotion that is irrelevant: a hot expert is hot across many steps, and a
 * ranking built from the step before ranks almost exactly the same as one built from this step.
 * The real cost is different and it is worth being blunt about, because it is what people expect
 * this file to do and it does not:
 *
 *   AN EXPERT THAT IS NOT RESIDENT THIS STEP CANNOT BE FETCHED IN TIME. It is served from
 *   wherever it is -- streamed out of the pinned pool over the link, or staged off the file tier.
 *   There is no last-moment rescue and there is no point adding one: the fetch would have to
 *   finish inside the gap between the router's output and the expert GEMM's first block, which
 *   is microseconds.
 *
 * Hit rate is therefore a property of the PLACEMENT POLICY, not of a rescue path. That is the
 * trade, and what it buys is a run phase with no device round-trip in it.
 *
 * ---- WHY LFU-WITH-DECAY AND NOT SOMETHING CLEVERER --------------------------------------------
 *
 * Replaying real routing traces against cache policies settles the shape of this. Runtime
 * placement beats static profile placement by a wide margin, so it is worth building; LFU-with-
 * decay and LRU score the same, so the decay constant is not where the wins are and this does not
 * need to be subtle. Belady is a few points of hit rate above both -- but an oracle restricted to
 * a disjoint sample of the future keeps most of that gap, so the remainder is knowledge of the
 * realised future and is not reachable. Of what is left, no statistic of PAST USE captures any of
 * it: an expert never selected has count zero, no age and no history, and most of the space is in
 * that state. The router's own gate value does separate those cases, so a scorer built on it is
 * the one direction here that measurement has not refuted.
 */
#pragma once
#include "planner.h"

#include <string>
#include <vector>

namespace rad {

struct HeatParams {
    /* Applied to a layer's whole plane each time that layer is dispatched. Every layer is
     * dispatched exactly once per forward, so all planes decay in lockstep and heats stay
     * comparable ACROSS layers -- which the global ranking below depends on.
     *
     * 0.98 is a half-life of a few dozen dispatches: long enough to survive a paragraph, short
     * enough to follow a change of subject. Nearby values are not distinguishable from each other
     * above the run-to-run spread, which is what the offline replay predicts. */
    float decay = 0.98f;

    /* Promote only when the candidate beats the victim by this ratio. Without it the two ends of
     * the ranking trade places on noise and the engine spends the whole link on churn that
     * changes nothing. */
    float hysteresis = 1.25f;

    /* ...and the additive half of the same guard, in units of one activation. It is what stops a
     * swap when the victim's heat is ZERO, where any ratio test passes trivially -- and right
     * after warmup that describes most of the resident set, so without this the first dispatch
     * would try to swap the entire cache.
     *
     * LINK TRAFFIC IS U-SHAPED IN THIS THRESHOLD. PAST THE BOTTOM OF THE CURVE, RAISING IT MOVES
     * MORE BYTES, NOT FEWER -- so do not "fix" it by tightening the bar to save traffic. The
     * economics look one-directional -- a promotion costs an H2D plus a D2H and only pays if the
     * expert is selected again -- and they are not, because A PROMOTION DECLINED COMES BACK AS A
     * MISS, and a miss is a shard streamed over the same link on the same budget. A tighter bar
     * then gives up hit rate and buys no traffic back.
     *
     * The exchange rate behind the value: the two link directions share one budget, so
     * a demotion costs about what a promotion does, and a swap therefore has to repay about 1.4
     * future hits rather than the 0.74 a half-price demotion would imply. */
    float min_gain = 1.5f;

    /* A BOUNDED MOVE BUDGET PER LAYER DISPATCH. The apparent risk is that a high budget makes the
     * mover compete with the tier it is trying to shrink -- a promotion and a streamed miss are
     * the same bytes over the same link. At this scale that does not bind: a larger budget pays
     * even though it moves considerably more bytes, because converging sooner is worth more than
     * the traffic costs. The benefit then SATURATES, and this sits where the knob stops mattering
     * rather than at the largest value that still helps. Past that point the guard, not the
     * budget, is what declines the extra opportunities -- so raising this alone is worth nothing
     * until min_gain moves with it.
     *
     * IT IS A BUDGET PER SLAB CLASS. A rank holding an uneven slice of every expert has two expert
     * classes -- the wider slice of one parity's experts and the narrower slice of the other's --
     * each about half a whole expert, and a budget shared between them would move half the bytes
     * a dispatch that whole experts do: residency would converge half as fast, and a fresh server
     * would run about 2% slower for its first minute. Per class, the two together move about as
     * many whole experts as one class of whole experts does. */
    int moves_per_dispatch = 8;

    /* Dispatches to watch before moving anything, and it is 0 because THE GUARD ALREADY DOES THIS
     * JOB. min_gain means a promotion needs `hot >= cold*hysteresis + 1.5`, and against a
     * zero-heat victim that is two tokens agreeing in one dispatch -- evidence, not noise,
     * whatever the dispatch count is. A warmup only postpones convergence. */
    int warmup_dispatches = 0;

    /* Minimum residency hold-down lease: a newly promoted unit cannot be evicted until at least
     * this many dispatches have elapsed since its promotion. Slashes wasted promotions by >5x.
     * Default to 0 for baseline pure-RLFU, configured to 30 in production engine. */
    int min_residency_dispatches = 0;

    /* Extra gain threshold required to re-admit an expert that was previously demoted, preventing
     * borderline experts from ping-ponging across the link. Default to 0.0f for baseline pure-RLFU. */
    float readmit_penalty = 0.0f;
};

/* A swap is returned as a PAIR even though the mover issues two independent transfers, because
 * the DECISION is a comparison between them: promoting without the matching demotion would grow
 * the resident set past the slab. */
struct Swap {
    int32_t promote = -1;   /* unit index */
    int32_t demote = -1;
};

/* One movable unit as step() ranks it: its heat, its layer's resident count, its index. */
struct HeatRanked { float h; int32_t rc; int32_t unit; };

struct HeatStats {
    uint64_t dispatches = 0;
    uint64_t proposed = 0, promotions = 0, demotions = 0;
    /* Why a dispatch did NOT move anything, which is the diagnostic that matters once the engine
     * looks idle. `refused` is the guard declining -- the healthy steady state, and the number
     * that should dominate once the hot set has settled. */
    uint64_t refused = 0, no_candidate = 0;
    /* Is the churn PROGRESS or THRASH? The same total move count means two opposite things. If
     * every promotion is of a unit never promoted before, the engine is still discovering the
     * working set and the traffic is an investment. If it keeps re-admitting units it demoted a
     * moment ago, the traffic is pure loss: both directions of the link spent to arrive back
     * where it started. `readmits` is the second case counted directly, and it is the number that
     * decides whether the guards are set right. */
    uint64_t distinct_promoted = 0, readmits = 0;
    uint64_t rapid_pingpong = 0; // Demoted units re-admitted within <= 5 dispatches
    /* WHAT THE ROUTER ASKED FOR, AND WHETHER IT WAS THERE. Counted on the histogram this engine
     * already walks, so it is one compare per routed expert and no second pass.
     *
     * A miss is not an error and nothing stalls on it: the GEMM reads that expert's weights from
     * whatever tier holds them, which for a host-pinned unit means pulling them across the link
     * on the critical path of the step. So `stream_bytes` is the OTHER traffic the mover's
     * promotions were hiding -- a promotion is bytes spent once to stop paying this, and without
     * both numbers beside each other there is no way to say whether the policy is winning.
     *
     * Weighted by the histogram's `weight`, because a draft head's histogram stands for several
     * routing passes and each of them really did read its experts. The hit RATE is unaffected --
     * both counters scale together -- and the byte figure is right.
     */
    uint64_t routed = 0, routed_resident = 0, stream_bytes = 0;
};

class HeatEngine {
public:
    /* Binds to the plan. Returns RAD_OK even when the plan is static -- that is a mode, not a
     * failure, and every caller would otherwise have to special-case it. `enabled()` says which.
     * Under Config::deterministic the plan is static, this stays off, and step() proposes
     * nothing for the whole run (spec §17). */
    int init(const Plan& plan, const HeatParams& params);
    bool enabled() const { return enabled_; }

    /* ---- the input: last step's routing histogram ------------------------------------------
     *
     * `counts[e]` is how many of the step's tokens routed to expert e of this layer. The credit
     * is that count and not a flat 1.0: cost is per DISTINCT expert -- a dispatch reads an
     * expert's weights once however many tokens chose it -- but the probability of needing it
     * again is not. An expert several tokens chose is far likelier to be chosen again soon than
     * one a single token chose, and most selections are single-token. Weighting by it moves no
     * extra bytes; it only ranks better. */
    /* `weight` is how many of that layer's routing passes this histogram stands for -- see
     * Ctx::last_histogram_reports. It is 1 for every layer of a non-speculative step and n_spec
     * for a draft head, and it belongs on the CREDIT rather than the decay: the caller sweeps
     * every layer exactly once a step, so the planes decay in lockstep and heats stay comparable
     * across layers. Without it a single histogram silently stands for several passes' worth of
     * activations and under-credits the experts a draft head routed to. */
    void observe(int32_t layer, const int32_t* counts, int32_t n_expert, float weight = 1.0f);
    /* The same, sparse: distinct expert ids and their counts. `counts` may be null for one each. */
    void observe_ids(int32_t layer, const int32_t* experts, const int32_t* counts, int32_t n);
    /* Starts the fetch of a histogram observe() will read: a row the card copied into host
     * memory is a miss on every line, so a caller that knows its next rows asks for them early. */
    static void prefetch(const int32_t* counts, int32_t n_expert) {
        for (int32_t j = 0; j < n_expert; j += 64 / (int32_t)sizeof(int32_t))
            __builtin_prefetch(counts + j);
    }

    /* ---- the tier map, written by the MOVER when a move PUBLISHES, not when it is issued ----
     *
     * Between issue and publish both copies of the bytes are valid and the dispatch still reads
     * the old one, so this engine must agree with dispatch about where a unit is or it will pick
     * the same one twice. `set_busy` is what excludes an in-flight unit instead. */
    void set_tier(int32_t unit, Tier t);
    void set_busy(int32_t unit, bool busy);
    Tier tier(int32_t unit) const { return tier_[(size_t)unit]; }
    bool busy(int32_t unit) const { return busy_[(size_t)unit] != 0; }

    /* The best swap available, or false when the guards refuse. */
    bool pick(Swap* out) const;

    /* One layer dispatch's worth of proposals, at most `moves_per_dispatch` of them a class.
     *
     * CONTRACT: every unit named in the result is marked busy on the way out. The caller must
     * either issue the transfer -- in which case set_tier() on publish and set_busy(false) after
     * the quarantine -- or call abandon() for both halves. A proposal that is silently dropped
     * leaves the pair busy forever and the engine goes quiet with no counter to say why. */
    int step(std::vector<Swap>* out);
    void abandon(const Swap& s);

    void boost_unit(int32_t unit, float amount) {
        if (!enabled_ || unit < 0 || (size_t)unit >= pos_.size()) return;
        const int32_t p = pos_[(size_t)unit];
        if (p < 0 || (size_t)p >= heat_.size()) return;
        heat_[(size_t)p] += amount;
        end_invalidate(unit);
    }
    void set_burst_moves(int n) { burst_moves_ = std::max(burst_moves_, n); }
    std::vector<int32_t> active_units(float min_heat = 0.5f, size_t max_units = 512) const;

    float heat(int32_t unit) const { return heat_[(size_t)pos_[(size_t)unit]]; }
    /* How many units of THIS unit's layer are resident. The tiebreak every residency decision
     * needs: a global ranking with no layer term leaves some layers holding far more than others,
     * and a layer starved of slots misses on nearly every token however good the ranking inside
     * it is. `pick` uses it on an exact heat tie and so must anything else that chooses. */
    int32_t layer_residents(int32_t unit) const {
        const int32_t l = (size_t)unit < layer_of_.size() ? layer_of_[(size_t)unit] : -1;
        return l >= 0 && (size_t)l < res_count_.size() ? res_count_[(size_t)l] : 0;
    }
    const HeatStats& stats() const { return stats_; }

    /* Hold-down lease query: whether unit has an active minimum residency lease. */
    bool is_leased(int32_t unit) const {
        if (!enabled_ || unit < 0 || (size_t)unit >= promoted_at_.size()) return false;
        if (tier_[(size_t)unit] != Tier::VRAM) return false;
        if (params_.min_residency_dispatches <= 0 || promoted_at_[(size_t)unit] == UINT64_MAX) return false;
        return stats_.dispatches < promoted_at_[(size_t)unit] + (uint64_t)params_.min_residency_dispatches;
    }

    uint64_t lease_remaining(int32_t unit) const {
        if (!is_leased(unit)) return 0;
        uint64_t exp = promoted_at_[(size_t)unit] + (uint64_t)params_.min_residency_dispatches;
        return exp > stats_.dispatches ? (exp - stats_.dispatches) : 0;
    }

    int min_residency_dispatches() const { return params_.min_residency_dispatches; }
    void set_min_residency_dispatches(int d) { params_.min_residency_dispatches = d; }
    uint64_t dispatches() const { return stats_.dispatches; }
    void advance_dispatch(uint64_t n = 1) { stats_.dispatches += n; }
    uint64_t promoted_at(int32_t unit) const {
        return (unit >= 0 && (size_t)unit < promoted_at_.size()) ? promoted_at_[(size_t)unit] : UINT64_MAX;
    }

    /* Finds a resident eviction candidate in slab_class whose hold-down lease has expired.
     * Returns unit index, or -1 if no eligible candidate exists (e.g. all slots are leased). */
    int32_t find_eviction_candidate(int32_t slab_class) const;
    int32_t find_eviction_candidate(size_t slab_class) const {
        return find_eviction_candidate(static_cast<int32_t>(slab_class));
    }

    /* Translates a plan slab_class to dense cls_ index, or returns -1 */
    int32_t slab_to_dense(int32_t slab_class) const;
    int32_t unit_klass(int32_t unit) const {
        return (unit >= 0 && (size_t)unit < cls_.size()) ? cls_[(size_t)unit] : -1;
    }

    /* Registers a prefetched unit as placed in VRAM and stamps its promotion dispatch for lease protection. */
    void register_prefetch(int32_t unit) { register_speculative_prefetch(unit, params_.min_residency_dispatches); }
    void register_speculative_prefetch(int32_t unit, int transient_lease);

    /* The exit banner. The per-layer residency histogram at the end of it is the one number that
     * says whether the engine did the thing it exists for: a global profile ranking leaves layers
     * with wildly different resident counts, and a layer starved of slots misses on nearly every
     * token however good the ranking inside it is. */
    std::string report() const;

private:
    HeatParams params_;
    bool       enabled_ = false;
    int32_t    n_layer_ = 0;
    int        burst_moves_ = 0;
    HeatStats  stats_;

    /* THE HEAT, BY POSITION RATHER THAN BY UNIT. Each layer owns one run of positions, and the
     * front of that run is indexed by EXPERT: expert e of layer l sits at slot_lo_[l] + e, whether
     * or not the plan has a unit for it and whether or not that unit may move. Then the layer's
     * other movable units, then padding to a whole number of lanes. After the layers come the
     * movable units with no layer, padded to a whole screening block, and after them everything
     * else.
     *
     * That lines a histogram up with the positions it credits. observe() walks counts[e] and the
     * heat at slot_lo_[l] + e side by side, in fixed-width blocks the compiler turns into vector
     * code, with no unit number looked up and no branch taken per expert. A position with no unit
     * behind it is a hole: never a candidate, never priced, and its heat is read by nothing.
     * `pos_` and `unit_at_` translate (-1 for a hole); everything outside the position arrays
     * stays indexed by MoveUnit index so this file and the plan cannot disagree about what a
     * number means. */
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
    /* Unit bytes, from the plan. Here so a miss can be priced where it is detected rather than
     * joined back to the plan by every reader. */
    std::vector<int64_t> bytes_;
    std::vector<Tier>    tier_;
    std::vector<uint8_t> busy_;
    std::vector<uint8_t> movable_;
    std::vector<int32_t> layer_of_;
    /* Bit 0: promoted before. Bit 1: demoted before. Two bits a unit, so the thrash question
     * costs nothing on the dispatch path. */
    std::vector<uint8_t> ever_;
    /* Dispatch index at which the unit was promoted to VRAM, for hold-down lease. */
    std::vector<uint64_t> promoted_at_;
    std::vector<uint64_t> lease_expire_at_;

    /* How many of each layer's units are resident. THE TIE-BREAK IS NOT COSMETIC: most resident
     * units are cold, so `heat == 0.0f` exactly is the common case at both ends of the ranking,
     * and whatever breaks that tie IS the policy for most of a run. Breaking it by index order
     * strips the low-numbered layers first, which replaces the skew this engine exists to fix
     * with a different one. So ties go to the layer that can most afford it: take the victim from
     * the layer holding the MOST, promote into the one holding the FEWEST. */
    std::vector<int32_t> res_count_;
    /* A SWAP STAYS INSIDE ONE SLAB CLASS. The promoted unit takes a slot of its own class and the
     * demoted one frees a slot of its own, so a pair across two classes would drain one class's
     * spare slots into the other's until promotions starve. A model has one expert class when
     * every expert is the same size on a rank, and two when a rank holds an uneven slice of each
     * expert's width (three 128-column blocks of one parity, two of the other). `cls_` is the
     * unit's class as a dense index, -1 for a unit that cannot move. */
    std::vector<int32_t> cls_;
    std::vector<int32_t> dense_classes_;
    int32_t              n_cls_ = 1;
    /* step()'s two ends of the ranking, `moves_per_dispatch` entries a class, kept between calls
     * so a dispatch allocates nothing. */
    std::vector<HeatRanked> rank_hot_, rank_cold_;
    std::vector<int32_t>    n_hot_, n_cold_, m_of_;
    /* Per slot, its lowest movable unit index: the bound step()'s layer screen needs on a tie. */
    std::vector<int32_t> min_unit_;

    /* EACH LAYER'S TWO ENDS, so a dispatch that proposes nothing -- the guard declining the best
     * pair, which is nearly every dispatch once the resident set has settled -- is decided from
     * one entry a layer rather than a scan of every unit. Over the movable units not in flight:
     * the hottest one off the device and the coldest one on it, and how many there are of each.
     * Slot n_layer_ is the movable units with no layer.
     *
     * observe() recomputes a layer's ends on the pass that decays and credits it, where each is
     * one more compare on a value already in a register. A change of tier or of being in flight
     * invalidates the unit's layer, and an invalid entry is rebuilt from its positions when step()
     * next reads it. */
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
