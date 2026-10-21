/* heat.cpp -- the heat engine's mechanics. See heat.h for why any of it is shaped this way. */
#include "heat.h"

#include <algorithm>
#include <cmath>

namespace rad {

namespace {
/* The width of one observe() block: a layer's run of positions is a whole number of these, so
 * every block has the same trip count and the compiler vectorizes it with no scalar remainder. */
constexpr int32_t kLanes = 16;
/* What the movable units with no layer are padded to. */
constexpr int32_t kBlock = 64;
}  /* namespace */

int HeatEngine::init(const Plan& plan, const HeatParams& params) {
    params_ = params;
    stats_ = HeatStats{};

    const size_t n = plan.units.size();
    tier_.assign(n, Tier::VRAM);
    busy_.assign(n, 0u);
    movable_.assign(n, 0u);
    layer_of_.assign(n, -1);
    bytes_.assign(n, 0);
    ever_.assign(n, 0u);
    promoted_at_.assign(n, UINT64_MAX);

    /* The movable units' slab classes, as dense indices -- see heat.h's `cls_`. */
    cls_.assign(n, -1);
    dense_classes_.clear();
    for (size_t u = 0; u < n; ++u) {
        const MoveUnit& mu = plan.units[u];
        if (!mu.movable) continue;
        const int32_t k = mu.slab_class;
        auto it = std::find(dense_classes_.begin(), dense_classes_.end(), k);
        if (it == dense_classes_.end()) { dense_classes_.push_back(k); it = dense_classes_.end() - 1; }
        cls_[u] = (int32_t)(it - dense_classes_.begin());
    }
    n_cls_ = dense_classes_.empty() ? 1 : (int32_t)dense_classes_.size();
    n_hot_.assign((size_t)n_cls_, 0);
    n_cold_.assign((size_t)n_cls_, 0);
    m_of_.assign((size_t)n_cls_, 0);

    n_layer_ = 0;
    for (const MoveUnit& mu : plan.units) n_layer_ = std::max(n_layer_, mu.layer + 1);
    if (n_layer_ < 0) n_layer_ = 0;
    res_count_.assign((size_t)n_layer_, 0);

    /* Each layer's units by expert id, and each slot's other movable units in plan order: a
     * layer's, then (slot n_layer_) those with no layer. Two units naming one (layer, expert)
     * leave the later one indexed by the expert -- the one a histogram credits -- and the
     * earlier, if it can move, among the layer's others. */
    std::vector<std::vector<int32_t>> by_expert((size_t)n_layer_);
    std::vector<std::vector<int32_t>> other((size_t)n_layer_ + 1);
    int64_t movable = 0;
    for (size_t u = 0; u < n; ++u) {
        const MoveUnit& mu = plan.units[u];
        tier_[u] = mu.tier;
        layer_of_[u] = mu.layer;
        bytes_[u] = mu.bytes;
        movable_[u] = mu.movable ? 1u : 0u;
        if (mu.movable) ++movable;
        if (mu.layer < 0) {
            if (mu.movable) other[(size_t)n_layer_].push_back((int32_t)u);
            continue;
        }
        if (mu.tier == Tier::VRAM) ++res_count_[(size_t)mu.layer];
        if (mu.expert >= 0) {
            auto& row = by_expert[(size_t)mu.layer];
            if ((size_t)mu.expert >= row.size()) row.resize((size_t)mu.expert + 1, -1);
            const int32_t was = row[(size_t)mu.expert];
            if (was >= 0 && movable_[(size_t)was]) other[(size_t)mu.layer].push_back(was);
            row[(size_t)mu.expert] = (int32_t)u;
        } else if (mu.movable) {
            other[(size_t)mu.layer].push_back((int32_t)u);
        }
    }

    pos_.assign(n, -1);
    unit_at_.clear();
    unit_at_.reserve(n + (size_t)(n_layer_ + 1) * kLanes + kBlock);
    n_ex_.assign((size_t)n_layer_, 0);
    slot_lo_.assign((size_t)n_layer_ + 2, 0);
    auto place = [&](int32_t u) {
        if (u >= 0) pos_[(size_t)u] = (int32_t)unit_at_.size();
        unit_at_.push_back(u);
    };
    for (size_t s = 0; s <= (size_t)n_layer_; ++s) {
        slot_lo_[s] = (int32_t)unit_at_.size();
        if (s < (size_t)n_layer_) {
            for (int32_t u : by_expert[s]) place(u);
            n_ex_[s] = (int32_t)by_expert[s].size();
        }
        for (int32_t u : other[s]) place(u);
        const int32_t to = s < (size_t)n_layer_ ? kLanes : kBlock;
        while (unit_at_.size() % (size_t)to) place(-1);
    }
    slot_lo_[(size_t)n_layer_ + 1] = (int32_t)unit_at_.size();
    for (size_t u = 0; u < n; ++u)
        if (pos_[u] < 0) place((int32_t)u);

    const size_t np = unit_at_.size();
    heat_.assign(np, 0.0f);
    rank_as_.assign(np, 0u);
    routed_as_.assign(np, 0u);
    miss_bytes_.assign(np, 0);
    for (size_t u = 0; u < n; ++u) set_place((int32_t)u);

    /* Each slot's lowest movable unit index, for step()'s tie test. */
    min_unit_.assign((size_t)n_layer_ + 1, INT32_MAX);
    for (size_t sl = 0; sl <= (size_t)n_layer_; ++sl)
        for (int32_t p = slot_lo_[sl]; p < slot_lo_[sl + 1]; ++p) {
            const int32_t u = unit_at_[(size_t)p];
            if (u >= 0 && movable_[(size_t)u] && u < min_unit_[sl]) min_unit_[sl] = u;
        }

    end_hot_.assign((size_t)n_layer_ + 1, 0.0f);
    end_cold_.assign((size_t)n_layer_ + 1, 0.0f);
    end_n_off_.assign((size_t)n_layer_ + 1, 0);
    end_n_on_.assign((size_t)n_layer_ + 1, 0);
    end_n_cold_.assign((size_t)n_layer_ + 1, 0);
    end_ok_.assign((size_t)n_layer_ + 1, 0);
    lease_expire_at_.assign((size_t)n_layer_ + 1, UINT64_MAX);

    /* Two ways to be off, and they are different facts worth different lines. A static plan is a
     * mode the operator asked for; a plan with nothing movable is a model that does not need this
     * engine -- the all-in-VRAM degenerate case, where the mover never runs either. */
    if (!plan.dynamic) {
        enabled_ = false;
        RAD_INFO("placement is static (--deterministic): the heat engine will not run, so "
                 "residency -- and therefore the summation order -- is fixed for the run");
        return RAD_OK;
    }
    if (movable == 0) {
        enabled_ = false;
        RAD_DEBUG("heat engine: nothing movable in this plan; every weight is statically placed");
        return RAD_OK;
    }
    enabled_ = true;
    RAD_INFO("heat engine: %lld movable units, %d move%s a dispatch, "
             "decay %.3f hysteresis %.2f min-gain %.2f",
             (long long)movable, params_.moves_per_dispatch,
             params_.moves_per_dispatch == 1 ? "" : "s",
             (double)params_.decay, (double)params_.hysteresis, (double)params_.min_gain);
    return RAD_OK;
}


void HeatEngine::observe_ids(int32_t layer, const int32_t* experts, const int32_t* counts,
                             int32_t n) {
    if (!enabled_ || layer < 0 || layer >= n_layer_) return;
    /* The histogram these name, dense, so both entry points are the one pass. */
    const int32_t n_ex = n_ex_[(size_t)layer];
    std::vector<int32_t> dense((size_t)n_ex, 0);
    for (int32_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        if (e >= 0 && e < n_ex) dense[(size_t)e] += counts ? counts[i] : 1;
    }
    observe(layer, dense.data(), n_ex, 1.0f);
}

/* DECAY EVERY POSITION, THEN CREDIT THE ROUTED ONES. Decay first, then credit: the other order
 * gives THIS dispatch's experts a decayed count, which makes min_gain = 1.0 mean 1/decay
 * activations rather than one -- the kind of off-by-a-constant that only ever shows up as a
 * threshold that does not do what it says.
 *
 * THE DECAY IS THE ONLY WORK EVERY POSITION NEEDS, so it is the only dense pass: one multiply a
 * float, vector code at any instruction set. A dispatch routes to a few dozen of a layer's
 * hundreds of experts, and everything else -- the credit, the pricing, the layer's ends -- is a
 * property of those, so it is done for those alone. `h * d` and then `+ c * w` round exactly as
 * the one expression `h * d + c * w` does, so the heats are the same bits either way.
 *
 * A ROUTED EXPERT IS PRICED HERE TOO. A unit not in VRAM is not a stall -- the GEMM reads it
 * where it lies -- but for every tier above VRAM that read crosses the link, and those are the
 * bytes the mover exists to stop paying. Pricing them on the pass that credits the heat puts the
 * hit rate and the streamed bytes beside the policy's decisions.
 *
 * THE LAYER'S ENDS FOLLOW FROM THE CREDITED EXPERTS WHEN THEY WERE VALID. A positive scale keeps
 * every order, and rounding is monotone, so the hottest unit off the device decays to exactly
 * `hot * d` and nothing uncredited passes it: the new hot end is that or a credited unit, whichever
 * is larger. The coldest units on the device decay to exactly `cold * d` likewise, and a credit
 * only raises a unit, so the cold end stands while any unit is left at it. The ends count the
 * units AT the cold value -- equal heats decay to equal heats -- because the common cold end is
 * a crowd: every resident unit no token has routed to yet sits at exactly zero. Only when the
 * last of them is credited is the next coldest unknown here, and the layer is rebuilt from its
 * positions. A tie decay creates is not counted, which can only rebuild a layer that did not need
 * it. */
void HeatEngine::observe(int32_t layer, const int32_t* counts, int32_t n_expert, float weight) {
    if (!enabled_ || layer < 0 || layer >= n_layer_ || !counts) return;
    if (!(weight > 0.0f)) weight = 1.0f;
    /* Passes this histogram stands for -- see observe()'s contract. At least one, so a caller
     * that does not set it still prices its reads. */
    const uint64_t passes = weight >= 1.0f ? (uint64_t)(weight + 0.5f) : 1u;

    const size_t l = (size_t)layer;
    const int32_t lo = slot_lo_[l], len = slot_lo_[l + 1] - lo;
    /* Past the histogram's last expert, and past the layer's, a position is credited nothing. */
    const int32_t n = std::max(0, std::min(n_expert, n_ex_[l]));
    const float d = params_.decay;

    /* THE HISTOGRAM ARRIVES BY DMA, so none of it is in the cache: every line is a miss, and a
     * scan that meets them one at a time pays one latency a line. Asked for all at once they
     * overlap each other and the decay below -- and a caller sweeping layers asks earlier still
     * (prefetch()), which makes this a no-op. */
    prefetch(counts, n);

    float* __restrict h = heat_.data() + lo;
    for (int32_t j = 0; j < len; ++j) h[j] *= d;

    bool ends = end_ok_[l] != 0;
    float hot = ends && end_n_off_[l] > 0 ? end_hot_[l] * d : -INFINITY;
    const float cold = ends && end_n_on_[l] > 0 ? end_cold_[l] * d : INFINITY;
    int32_t n_cold = ends ? end_n_cold_[l] : 0;
    int32_t routed = 0, resident = 0;
    int64_t miss = 0;
    auto credit = [&](int32_t j) {
        const int32_t c = counts[j];
        const size_t q = (size_t)(lo + j);
        const float was = h[j];
        const float x = was + (float)c * weight;
        h[j] = x;
        routed += routed_as_[q] != 0;
        resident += routed_as_[q] == 2;
        miss += miss_bytes_[q];
        if (rank_as_[q] == 1) hot = std::max(hot, x);
        else if (rank_as_[q] == 2 && was <= cold && --n_cold <= 0) ends = false;
    };
    /* A dispatch routes to a few dozen of the layer's experts, so the credited ones are found a
     * block at a time -- a mask of the positive counts is vector code -- and only its set bits are
     * visited, in increasing order as a scan would meet them. */
    int32_t j0 = 0;
    for (; j0 + kLanes <= n; j0 += kLanes) {
        uint32_t m = 0;
        for (int32_t j = 0; j < kLanes; ++j) m |= (uint32_t)(counts[j0 + j] > 0) << j;
        for (; m; m &= m - 1) credit(j0 + __builtin_ctz(m));
    }
    for (int32_t j = j0; j < n; ++j)
        if (counts[j] > 0) credit(j);

    if (ends) {
        end_hot_[l] = end_n_off_[l] ? hot : 0.0f;
        end_cold_[l] = end_n_on_[l] ? cold : 0.0f;
        end_n_cold_[l] = n_cold;
    } else {
        end_rebuild(l);
    }

    stats_.routed += (uint64_t)routed * passes;
    stats_.routed_resident += (uint64_t)resident * passes;
    stats_.stream_bytes += (uint64_t)miss * passes;
    ++stats_.dispatches;
}

void HeatEngine::set_tier(int32_t unit, Tier t) {
    if (unit < 0 || (size_t)unit >= tier_.size()) return;
    const int32_t l = layer_of_[(size_t)unit];
    const Tier was = tier_[(size_t)unit];
    if (l >= 0) {
        if (was == Tier::VRAM && t != Tier::VRAM) --res_count_[(size_t)l];
        else if (was != Tier::VRAM && t == Tier::VRAM) ++res_count_[(size_t)l];
    }
    tier_[(size_t)unit] = t;
    set_place(unit);
    end_invalidate(unit);

    if (was != Tier::VRAM && t == Tier::VRAM) {
        ++stats_.promotions;
        promoted_at_[(size_t)unit] = stats_.dispatches;
        if (ever_[(size_t)unit] & 2u) ++stats_.readmits;
        if (!(ever_[(size_t)unit] & 1u)) { ++stats_.distinct_promoted; ever_[(size_t)unit] |= 1u; }
    } else if (was == Tier::VRAM && t != Tier::VRAM) {
        ++stats_.demotions;
        ever_[(size_t)unit] |= 2u;
        promoted_at_[(size_t)unit] = UINT64_MAX;
    }
}

void HeatEngine::set_busy(int32_t unit, bool busy) {
    if (unit < 0 || (size_t)unit >= busy_.size()) return;
    busy_[(size_t)unit] = busy ? 1u : 0u;
    set_place(unit);
    end_invalidate(unit);
}

void HeatEngine::set_place(int32_t unit) {
    const size_t u = (size_t)unit, p = (size_t)pos_[u];
    const bool vram = tier_[u] == Tier::VRAM;
    rank_as_[p] = (!movable_[u] || busy_[u]) ? 0u : vram ? 2u : 1u;
    routed_as_[p] = vram ? 2u : 1u;
    miss_bytes_[p] = vram ? 0 : bytes_[u];
}

void HeatEngine::end_rebuild(size_t slot) {
    float hot = 0.0f, cold = 0.0f;
    int32_t n_off = 0, n_on = 0, n_cold = 0;
    uint64_t next_lease = UINT64_MAX;
    for (int32_t p = slot_lo_[slot], e = slot_lo_[slot + 1]; p < e; ++p) {
        const uint8_t r = rank_as_[(size_t)p];
        if (!r) continue;
        const float h = heat_[(size_t)p];
        if (r == 2) {
            const int32_t u = unit_at_[(size_t)p];
            if (u >= 0 && params_.min_residency_dispatches > 0 && promoted_at_[(size_t)u] != UINT64_MAX) {
                uint64_t exp = promoted_at_[(size_t)u] + (uint64_t)params_.min_residency_dispatches;
                if (stats_.dispatches < exp) {
                    next_lease = std::min(next_lease, exp);
                    continue;
                }
            }
            if (n_on++ == 0 || h < cold) { cold = h; n_cold = 1; }
            else if (h == cold) ++n_cold;
        } else {
            if (n_off++ == 0 || h > hot) hot = h;
        }
    }
    lease_expire_at_[slot] = next_lease;
    end_hot_[slot] = hot;
    end_cold_[slot] = cold;
    end_n_off_[slot] = n_off;
    end_n_on_[slot] = n_on;
    end_n_cold_[slot] = n_cold;
    end_ok_[slot] = 1;
}

void HeatEngine::abandon(const Swap& s) {
    set_busy(s.promote, false);
    set_busy(s.demote, false);
}

/* THE ONE-PAIR PICK, kept as the definition of the ranking: the hottest movable unit off the device
 * against the coldest one on it, ties broken toward the layer that most needs the slot, and the
 * first unit in index order on a full tie. step() proposes exactly the pairs repeated calls to
 * this would, from one pass. */
bool HeatEngine::pick(Swap* out) const {
    if (!enabled_ || !warm() || !out) return false;

    /* Each class's hottest unit off the device and coldest one on it: a swap stays in its class
     * (heat.h's `cls_`). */
    std::vector<int32_t> hot((size_t)n_cls_, -1), cold((size_t)n_cls_, -1);
    std::vector<float>   hot_h((size_t)n_cls_, 0.0f), cold_h((size_t)n_cls_, 0.0f);
    std::vector<int32_t> hot_res((size_t)n_cls_, 0), cold_res((size_t)n_cls_, 0);

    for (size_t i = 0; i < movable_.size(); ++i) {
        if (!movable_[i] || busy_[i]) continue;
        const size_t c = (size_t)cls_[i];
        const int32_t l = layer_of_[i];
        const int32_t rc = l >= 0 ? res_count_[(size_t)l] : 0;
        const float h = heat_[(size_t)pos_[i]];
        if (tier_[i] == Tier::VRAM) {
            /* Hold-down lease: a unit promoted within min_residency_dispatches cannot be demoted */
            if (params_.min_residency_dispatches > 0 &&
                promoted_at_[i] != UINT64_MAX &&
                stats_.dispatches < promoted_at_[i] + (uint64_t)params_.min_residency_dispatches)
                continue;
            /* Colder wins; on an exact tie take from the layer holding the MOST resident units,
             * which is the layer that can spare it. */
            if (cold[c] < 0 || h < cold_h[c] || (h == cold_h[c] && rc > cold_res[c])) {
                cold[c] = (int32_t)i; cold_h[c] = h; cold_res[c] = rc;
            }
        } else {
            /* Hotter wins; on an exact tie the layer with the FEWEST resident units wins, because
             * that is the layer whose tokens are missing most often. Both halves of this are the
             * same statement about where capacity should sit.
             *
             * Tier::SSD is a candidate here and Tier::HostPinned is too, deliberately: this
             * engine promotes UP from wherever the unit is, and the mover knows which source that
             * means. What it never does is push a unit DOWN to the file tier -- banishing
             * something to disk is a bet that it will not be needed, paid at a full read every
             * time the bet is wrong, and a frequency counter has no forward-looking confidence to
             * justify one. Demotion targets the host pool, and only the planner assigns SSD. */
            if (hot[c] < 0 || h > hot_h[c] || (h == hot_h[c] && rc < hot_res[c])) {
                hot[c] = (int32_t)i; hot_h[c] = h; hot_res[c] = rc;
            }
        }
    }

    /* BOTH guards, and they answer different failures. The ratio stops a swap that trades one
     * warm unit for another; the additive floor stops the degenerate case where the victim's heat
     * is zero, which every ratio passes trivially and which describes most of the resident set
     * right after warmup. Of the classes whose pair passes, the one with the hotter candidate. */
    int32_t best = -1;
    for (int32_t c = 0; c < n_cls_; ++c) {
        if (hot[(size_t)c] < 0 || cold[(size_t)c] < 0) continue;
        const float penalty = (ever_[(size_t)hot[(size_t)c]] & 2u) ? params_.readmit_penalty : 0.0f;
        if (hot_h[(size_t)c] < cold_h[(size_t)c] * params_.hysteresis + params_.min_gain + penalty) continue;
        if (best < 0 || hot_h[(size_t)c] > hot_h[(size_t)best] ||
            (hot_h[(size_t)c] == hot_h[(size_t)best] &&
             (hot_res[(size_t)c] < hot_res[(size_t)best] ||
              (hot_res[(size_t)c] == hot_res[(size_t)best] && hot[(size_t)c] < hot[(size_t)best]))))
            best = c;
    }
    if (best < 0) return false;

    out->promote = hot[(size_t)best];
    out->demote = cold[(size_t)best];
    return true;
}

/* THE WHOLE DISPATCH'S PROPOSALS FROM ONE PASS. Marking a pair busy changes neither a unit's heat
 * nor any layer's resident count, so the m-th pair repeated pick() calls would return is simply the
 * m-th entry at each end of the same ranking. One pass keeps the `moves_per_dispatch` best of each
 * end; the proposals, and the counters that say why the list stopped, are the ones the repeated
 * picks produce. This runs before every pass a rank issues -- draft passes included, where the
 * device waits for the issue -- so a scan per proposed move would be most of its cost. */
namespace {
using Ranked = HeatRanked;
/* Hotter first, then the layer with fewer residents, then the lower index. */
inline bool hotter(const Ranked& a, const Ranked& b) {
    if (a.h != b.h) return a.h > b.h;
    if (a.rc != b.rc) return a.rc < b.rc;
    return a.unit < b.unit;
}
/* Colder first, then the layer with more residents, then the lower index. */
inline bool colder(const Ranked& a, const Ranked& b) {
    if (a.h != b.h) return a.h < b.h;
    if (a.rc != b.rc) return a.rc > b.rc;
    return a.unit < b.unit;
}
/* Insert into a sorted list of at most `k`, dropping what falls off the end. */
template <typename Less>
inline void keep_best(Ranked* v, int* n, int k, const Ranked& x, Less less) {
    if (*n == k && !less(x, v[k - 1])) return;
    int i = *n < k ? (*n)++ : k - 1;
    while (i > 0 && less(x, v[i - 1])) { v[i] = v[i - 1]; --i; }
    v[i] = x;
}
}  /* namespace */

int HeatEngine::step(std::vector<Swap>* out) {
    if (!out) return RAD_E_INVAL;
    out->clear();
    if (!enabled_) return RAD_OK;
    const int k = burst_moves_ > 0 ? burst_moves_ : params_.moves_per_dispatch;
    burst_moves_ = 0;
    if (k <= 0) return RAD_OK;

    /* THE BEST PAIR FROM THE LAYERS' ENDS FIRST. If it fails the guard every pair does -- the
     * ranking only gets colder at one end and warmer at the other -- and the dispatch proposes
     * nothing, which is the verdict the full ranking below reaches, from one entry a layer. The
     * ends also count the candidates at each end, which is all the stop reasons below need. */
    int64_t n_off = 0, n_on = 0;
    float bar = 0.0f;
    {
        float hot_h = 0.0f, cold_h = 0.0f;
        for (size_t l = 0; l < end_ok_.size(); ++l) {
            if (!end_ok_[l] || (params_.min_residency_dispatches > 0 && stats_.dispatches >= lease_expire_at_[l]))
                end_rebuild(l);
            if (end_n_off_[l] > 0 && (n_off == 0 || end_hot_[l] > hot_h)) hot_h = end_hot_[l];
            if (end_n_on_[l] > 0 && (n_on == 0 || end_cold_[l] < cold_h)) cold_h = end_cold_[l];
            n_off += end_n_off_[l];
            n_on += end_n_on_[l];
        }
        if (!warm() || n_off == 0 || n_on == 0 ||
            hot_h < cold_h * params_.hysteresis + params_.min_gain) {
            if (n_off > 0 && n_on > 0) ++stats_.refused; else ++stats_.no_candidate;
            return RAD_OK;
        }
        bar = cold_h * params_.hysteresis + params_.min_gain;
    }

    /* THE RANKING, over every movable position. A unit can only enter a full list by reaching
     * its last entry's heat, so each block of positions is first screened against the two lists'
     * last heats -- one compare a position, over contiguous floats -- and only what passes the
     * screen meets the exact comparison, ties and all. The screen passes a tie, so it never drops
     * a unit the comparison would keep; and the lists are a strict order's best k, so meeting the
     * units in position order rather than unit order ends in the same lists.
     *
     * The hot end is screened against the guard as well. A unit colder than the coldest resident
     * unit's bar fails the guard against every victim the cold list can hold, so it cannot be
     * proposed, and the list it is kept out of only ever ends at the pair that fails. A layer's
     * run is a whole number of kLanes blocks, so every block has the same trip count and the
     * screen is vector code.
     *
     * A WHOLE LAYER IS SCREENED FIRST, BY ITS ENDS. Most resident units sit at exactly the cold
     * end's heat -- zero, for every one no token has routed to lately -- and the position screen
     * passes a tie, so it would send nearly every resident unit to the exact comparison only for
     * the tie break to refuse it. The ends are that layer's hottest candidate off the device and
     * coldest on it, so a layer whose hot end is below the hot list's bar and whose cold end
     * cannot beat the cold list's last entry holds no unit either list would keep. On a tie the
     * comparison falls to the layer's resident count, which every unit of the layer shares, and
     * then to the unit index, which the layer's lowest bounds. */
    /* ONE PAIR OF LISTS A CLASS, because a swap stays inside its class (heat.h's `cls_`). The
     * screens below take the loosest of the classes' thresholds, so a unit any class's list would
     * keep still reaches the exact comparison; with one class they are that class's thresholds
     * and the screen is exact. */
    const int nc = n_cls_;
    rank_hot_.resize((size_t)(k * nc));
    rank_cold_.resize((size_t)(k * nc));
    Ranked* hot = rank_hot_.data();
    Ranked* cold = rank_cold_.data();
    int32_t* n_hot = n_hot_.data();
    int32_t* n_cold = n_cold_.data();
    for (int c = 0; c < nc; ++c) n_hot[c] = n_cold[c] = 0;
    const int64_t n_pool = n_off, n_vram = n_on;
    const float*   hp = heat_.data();
    const uint8_t* ra = rank_as_.data();
    /* A hot unit reaches a list at the lowest last entry of a full list, and none below the guard;
     * a cold one at the highest last entry, and any heat while some list is not full. */
    auto hot_bar = [&]() {
        float t = INFINITY;
        for (int c = 0; c < nc; ++c) t = std::min(t, n_hot[c] == k ? hot[c * k + k - 1].h : -INFINITY);
        return std::max(t, bar);
    };
    auto cold_bar = [&]() {
        float t = -INFINITY;
        for (int c = 0; c < nc; ++c) t = std::max(t, n_cold[c] == k ? cold[c * k + k - 1].h : INFINITY);
        return t;
    };
    uint8_t pass[kLanes];
    for (int32_t sl = 0; sl <= n_layer_; ++sl) {
        const int32_t lo = slot_lo_[(size_t)sl], hi = slot_lo_[(size_t)sl + 1];
        if (lo >= hi) continue;
        const float th0 = hot_bar();
        const bool some_hot = end_n_off_[(size_t)sl] > 0 && end_hot_[(size_t)sl] >= th0;
        bool some_cold = end_n_on_[(size_t)sl] > 0;
        if (some_cold && nc == 1 && n_cold[0] == k) {
            const Ranked& last = cold[k - 1];
            const float c = end_cold_[(size_t)sl];
            const int32_t rc = sl < n_layer_ ? res_count_[(size_t)sl] : 0;
            some_cold = c < last.h ||
                        (c == last.h && (rc > last.rc ||
                                         (rc == last.rc && min_unit_[(size_t)sl] < last.unit)));
        } else if (some_cold) {
            some_cold = end_cold_[(size_t)sl] <= cold_bar();
        }
        if (!some_hot && !some_cold) continue;
        for (int32_t b = lo; b < hi; b += kLanes) {
            const float th = hot_bar();
            const float tc = cold_bar();
            uint8_t any = 0;
            for (int32_t j = 0; j < kLanes; ++j) {
                const float h = hp[b + j];
                const uint8_t r = ra[b + j];
                pass[j] = (uint8_t)(((r == 1) & (h >= th)) | (((r == 2) & (h <= tc)) << 1));
                any |= pass[j];
            }
            if (!any) continue;
            for (int32_t j = 0; j < kLanes; ++j) {
                if (!pass[j]) continue;
                const int32_t u = unit_at_[(size_t)(b + j)];
                const int32_t l = layer_of_[(size_t)u];
                const int32_t c = cls_[(size_t)u];
                const Ranked r{ hp[b + j], l >= 0 ? res_count_[(size_t)l] : 0, u };
                if (pass[j] & 2) {
                    if (params_.min_residency_dispatches == 0 ||
                        promoted_at_[(size_t)u] == UINT64_MAX ||
                        stats_.dispatches >= promoted_at_[(size_t)u] + (uint64_t)params_.min_residency_dispatches) {
                        keep_best(cold + c * k, &n_cold[c], k, r, colder);
                    }
                } else {
                    keep_best(hot + c * k, &n_hot[c], k, r, hotter);
                }
            }
        }
    }

    /* The proposals: each class's m-th hottest against its m-th coldest, for as long as that
     * passes the guard -- the ranking only gets colder at one end and warmer at the other, so the
     * first pair that fails ends the class -- and at most `k` of them a class, taken across the
     * classes hottest first, which is the order repeated pick() calls would return them in. */
    int32_t* m_of = m_of_.data();
    for (int c = 0; c < nc; ++c) m_of[c] = 0;
    int made = 0;
    bool short_of_budget = false;
    for (;;) {
        int best = -1;
        for (int c = 0; c < nc; ++c) {
            const int32_t m = m_of[c];
            if (m >= k) continue;
            if (m >= n_hot[c] || m >= n_cold[c]) { short_of_budget = true; continue; }
            const Ranked& h = hot[c * k + m];
            const float penalty = (ever_[(size_t)h.unit] & 2u) ? params_.readmit_penalty : 0.0f;
            if (h.h < cold[c * k + m].h * params_.hysteresis + params_.min_gain + penalty) {
                short_of_budget = true;
                continue;
            }
            if (best < 0 || hotter(h, hot[best * k + m_of[best]])) best = c;
        }
        if (best < 0) break;
        Swap s;
        s.promote = hot[best * k + m_of[best]].unit;
        s.demote = cold[best * k + m_of[best]].unit;
        ++m_of[best];
        /* Marked busy on the way out: the caller owns the pair until it publishes or abandons. */
        set_busy(s.promote, true);
        set_busy(s.demote, true);
        out->push_back(s);
        ++stats_.proposed;
        ++made;
    }
    if (short_of_budget) {
        /* Two ways to propose nothing more and they are different diagnoses. `refused` is the
         * guard declining -- the healthy steady state. `no_candidate` is one end of the ranking
         * being empty, which means either everything movable is resident or nothing is, and is a
         * budget question rather than a policy one. The pairs already proposed are busy by now,
         * so they are not counted at either end. */
        if (n_pool - made > 0 && n_vram - made > 0) ++stats_.refused; else ++stats_.no_candidate;
    }
    return RAD_OK;
}


std::string HeatEngine::report() const {
    if (!enabled_)
        return "heat engine: off (static placement); residency and therefore the summation "
               "order are fixed for the run\n";

    std::string s = fmt("heat engine: %llu dispatches, %llu swaps proposed, %llu promotions, "
                        "%llu demotions\n",
                        (unsigned long long)stats_.dispatches,
                        (unsigned long long)stats_.proposed,
                        (unsigned long long)stats_.promotions,
                        (unsigned long long)stats_.demotions);
    s += fmt("  %llu dispatches declined on hysteresis, %llu found no candidate at one end\n",
             (unsigned long long)stats_.refused, (unsigned long long)stats_.no_candidate);
    /* The one line that says whether the traffic above was worth paying. A readmit is a unit this
     * engine demoted and then fetched back: both directions of the link spent to return to where
     * it started, on the link the zero-copy tier needs. High here means the guards are too
     * loose -- and note that TIGHTENING min_gain to fix it costs more bytes, not fewer. */
    s += fmt("  %llu distinct units promoted, %llu readmits (%.0f%% of promotions were of "
             "something already demoted)\n",
             (unsigned long long)stats_.distinct_promoted, (unsigned long long)stats_.readmits,
             stats_.promotions ? 100.0 * (double)stats_.readmits / (double)stats_.promotions : 0.0);

    /* The histogram this engine exists for. A global profile ranking leaves the per-layer
     * resident count wherever it happens to fall; a layer starved of slots misses on nearly every
     * token however good the ranking inside it is, which is how a hit rate lands below what
     * choosing at random would give. Whether the engine repaired that is a question about this
     * DISTRIBUTION, not about the mean. */
    if (!res_count_.empty()) {
        int32_t lo = res_count_[0], hi = res_count_[0];
        double sum = 0;
        for (int32_t v : res_count_) { lo = std::min(lo, v); hi = std::max(hi, v); sum += v; }
        const double mean = sum / (double)res_count_.size();
        double var = 0;
        for (int32_t v : res_count_) var += (v - mean) * (v - mean);
        s += fmt("  resident units a layer: min %d max %d mean %.1f sd %.1f\n", (int)lo, (int)hi,
                 mean, std::sqrt(var / (double)res_count_.size()));
        /* AND WHICH LAYERS, because min/max/sd cannot be joined to anything. A kernel trace can
         * say which layers run longer than their neighbours; only this line can say whether those
         * are the starved ones, and without it the two measurements sit in different files saying
         * nothing to each other. One line, one number a layer, printed once at exit. */
        s += "  per layer:";
        for (size_t i = 0; i < res_count_.size(); ++i) {
            if (i % 12 == 0) s += fmt("\n   L%-3zu ", i);
            s += fmt("%5d", (int)res_count_[i]);
        }
        s += "\n";
    }
    return s;
}

std::vector<int32_t> HeatEngine::active_units(float min_heat, size_t max_units) const {
    std::vector<std::pair<float, int32_t>> cands;
    if (!enabled_) return {};
    for (size_t u = 0; u < movable_.size(); ++u) {
        if (!movable_[u]) continue;
        const int32_t p = pos_[u];
        if (p >= 0 && (size_t)p < heat_.size() && heat_[(size_t)p] >= min_heat) {
            cands.emplace_back(heat_[(size_t)p], (int32_t)u);
        }
    }
    if (cands.size() > max_units) {
        std::partial_sort(cands.begin(), cands.begin() + max_units, cands.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        cands.resize(max_units);
    }
    std::vector<int32_t> res;
    res.reserve(cands.size());
    for (const auto& c : cands) res.push_back(c.second);
    return res;
}

int32_t HeatEngine::slab_to_dense(int32_t slab_class) const {
    auto it = std::find(dense_classes_.begin(), dense_classes_.end(), slab_class);
    return it != dense_classes_.end() ? static_cast<int32_t>(it - dense_classes_.begin()) : -1;
}

int32_t HeatEngine::find_eviction_candidate(int32_t slab_class) const {
    if (!enabled_) return -1;
    const int32_t dense = slab_to_dense(slab_class);
    if (dense < 0 || dense >= n_cls_) return -1;

    int32_t cold_u = -1;
    float cold_h = 0.0f;
    int32_t cold_res = 0;

    for (size_t i = 0; i < movable_.size(); ++i) {
        if (!movable_[i] || busy_[i]) continue;
        if (cls_[i] != dense) continue;
        if (tier_[i] != Tier::VRAM) continue;

        /* Hold-down lease: a unit promoted within min_residency_dispatches cannot be evicted */
        if (params_.min_residency_dispatches > 0 &&
            promoted_at_[i] != UINT64_MAX &&
            stats_.dispatches < promoted_at_[i] + (uint64_t)params_.min_residency_dispatches) {
            continue;
        }

        const int32_t l = layer_of_[i];
        const int32_t rc = l >= 0 ? res_count_[(size_t)l] : 0;
        const float h = heat_[(size_t)pos_[i]];

        if (cold_u < 0 || h < cold_h || (h == cold_h && rc > cold_res)) {
            cold_u = (int32_t)i;
            cold_h = h;
            cold_res = rc;
        }
    }
    return cold_u;
}

void HeatEngine::register_speculative_prefetch(int32_t unit, int transient_lease) {
    if (unit < 0 || (size_t)unit >= tier_.size()) return;
    if (tier_[(size_t)unit] != Tier::VRAM) {
        set_tier(unit, Tier::VRAM);
    }
    // Set lease expiration to transient_lease dispatches instead of full min_residency_dispatches
    const uint64_t offset = (uint64_t)std::max(0, params_.min_residency_dispatches - transient_lease);
    promoted_at_[(size_t)unit] = stats_.dispatches >= offset ? stats_.dispatches - offset : 0;
    end_invalidate(unit);
}

}  /* namespace rad */

