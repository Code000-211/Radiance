/* heat.cpp -- the heat engine's mechanics. See heat.h for the policy and data layout. */
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

    /* The movable units' slab classes as dense indices (heat.h: `cls_`). */
    cls_.assign(n, -1);
    {
        std::vector<int32_t> dense;
        for (size_t u = 0; u < n; ++u) {
            const MoveUnit& mu = plan.units[u];
            if (!mu.movable) continue;
            const int32_t k = mu.slab_class;
            auto it = std::find(dense.begin(), dense.end(), k);
            if (it == dense.end()) { dense.push_back(k); it = dense.end() - 1; }
            cls_[u] = (int32_t)(it - dense.begin());
        }
        n_cls_ = dense.empty() ? 1 : (int32_t)dense.size();
    }
    n_hot_.assign((size_t)n_cls_, 0);
    n_cold_.assign((size_t)n_cls_, 0);
    m_of_.assign((size_t)n_cls_, 0);

    n_layer_ = 0;
    for (const MoveUnit& mu : plan.units) n_layer_ = std::max(n_layer_, mu.layer + 1);
    res_count_.assign((size_t)n_layer_, 0);

    /* Each layer's units by expert id, and each slot's other movable units in plan order (slot
     * n_layer_ holds those with no layer). If two units name one (layer, expert), the later one
     * is indexed by the expert (and credited by histograms); the earlier, if movable, goes among
     * the layer's others. */
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

    /* Two distinct ways to be off: a static plan (an operator choice), or nothing movable (every
     * weight fits in VRAM; the mover never runs either). */
    if (!plan.dynamic) {
        enabled_ = false;
        RAD_INFO("placement is static (--deterministic): residency and summation order are "
                 "fixed for the run");
        return RAD_OK;
    }
    if (movable == 0) {
        enabled_ = false;
        RAD_DEBUG("heat engine: nothing movable in this plan; every weight is statically placed");
        return RAD_OK;
    }
    enabled_ = true;
    RAD_INFO("heat engine: %lld movable units, %d move%s a step, "
             "decay %.3f hysteresis %.2f min-gain %.2f credit cap %.0f",
             (long long)movable, params_.moves_per_step,
             params_.moves_per_step == 1 ? "" : "s",
             (double)params_.decay, (double)params_.hysteresis, (double)params_.min_gain,
             (double)params_.credit_cap);
    return RAD_OK;
}


void HeatEngine::observe_ids(int32_t layer, const int32_t* experts, const int32_t* counts,
                             int32_t n) {
    if (!enabled_ || layer < 0 || layer >= n_layer_) return;
    /* Densify so both entry points share observe(). */
    const int32_t n_ex = n_ex_[(size_t)layer];
    std::vector<int32_t> dense((size_t)n_ex, 0);
    for (int32_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        if (e >= 0 && e < n_ex) dense[(size_t)e] += counts ? counts[i] : 1;
    }
    observe(layer, dense.data(), n_ex, 1.0f);
}

/* Decay every position, then credit the routed ones. Decay comes first so this dispatch's
 * credit is not decayed; otherwise min_gain = 1.0 would mean 1/decay activations.
 *
 * The decay is the only dense pass (one vector multiply per float). A dispatch routes to a few
 * dozen of a layer's hundreds of experts, so crediting, pricing and the layer's ends touch only
 * those. `h * d` then `+ c * w` rounds exactly as `h * d + c * w`, so heats are bit-identical.
 *
 * Routed experts are priced here too: a unit off the device is read in place across the link,
 * and those are the bytes the mover exists to save. Pricing on the crediting pass puts hit rate
 * and streamed bytes beside the policy's decisions.
 *
 * The layer's ends are updated from the credited experts when they were valid. Decay by a positive
 * scale with monotone rounding keeps every order, so the hot end becomes `hot * d` or a credited
 * unit, whichever is larger; the cold end becomes `cold * d` and stands while any unit remains at
 * it, since credit only raises. The cold end counts units at exactly that value, because it is
 * usually a crowd (every resident unit not routed to lately sits at zero). When the last is
 * credited the layer is rebuilt from its positions. Ties created by decay are not counted, which
 * can only cause an unneeded rebuild. */
void HeatEngine::observe(int32_t layer, const int32_t* counts, int32_t n_expert, float weight) {
    if (!enabled_ || layer < 0 || layer >= n_layer_ || !counts) return;
    if (!(weight > 0.0f)) weight = 1.0f;
    /* Routing passes this histogram stands for (see heat.h: observe); at least one. */
    const uint64_t passes = weight >= 1.0f ? (uint64_t)(weight + 0.5f) : 1u;

    const size_t l = (size_t)layer;
    const int32_t lo = slot_lo_[l], len = slot_lo_[l + 1] - lo;
    /* Positions past the histogram's or the layer's last expert get no credit. */
    const int32_t n = std::max(0, std::min(n_expert, n_ex_[l]));
    const float d = params_.decay;

    /* The histogram arrives by DMA and is not in cache. Prefetching every line at once overlaps
     * the misses with each other and with the decay; a caller sweeping layers may prefetch
     * earlier still. */
    prefetch(counts, n);

    /* The credit cap (HeatParams::credit_cap) scales the whole histogram, preserving order.
     * Pricing ignores the cap. */
    int64_t total = 0;
    for (int32_t j = 0; j < n; ++j) total += counts[j] > 0 ? counts[j] : 0;
    const float cap = params_.credit_cap;
    const float w = cap > 0.0f && (float)total > cap ? weight * cap / (float)total : weight;

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
        const float x = was + (float)c * w;
        h[j] = x;
        routed += routed_as_[q] != 0;
        resident += routed_as_[q] == 2;
        miss += miss_bytes_[q];
        if (rank_as_[q] == 1) hot = std::max(hot, x);
        else if (rank_as_[q] == 2 && was <= cold && --n_cold <= 0) ends = false;
    };
    /* Find credited experts a block at a time via a vectorisable mask of positive counts, then
     * visit only the set bits, in increasing order. */
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

void HeatEngine::set_tier(int32_t unit, Tier t, bool fill, bool initial) {
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
        if (fill) {
            ++stats_.fill_promotions;
            if (initial) ++stats_.initial_fill_promotions;
        }
        if (ever_[(size_t)unit] & 2u) ++stats_.readmits;
        if (!(ever_[(size_t)unit] & 1u)) { ++stats_.distinct_promoted; ever_[(size_t)unit] |= 1u; }
    } else if (was == Tier::VRAM && t != Tier::VRAM) {
        ++stats_.demotions;
        ever_[(size_t)unit] |= 2u;
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
    for (int32_t p = slot_lo_[slot], e = slot_lo_[slot + 1]; p < e; ++p) {
        const uint8_t r = rank_as_[(size_t)p];
        if (!r) continue;
        const float h = heat_[(size_t)p];
        if (r == 2) {
            if (n_on++ == 0 || h < cold) { cold = h; n_cold = 1; }
            else if (h == cold) ++n_cold;
        } else {
            if (n_off++ == 0 || h > hot) hot = h;
        }
    }
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

/* The one-pair pick, kept as the reference definition of the ranking: the hottest movable unit
 * off the device against the coldest on it, ties broken toward the layer that most needs the
 * slot, then by lowest index. step() proposes exactly the pairs repeated calls to this would. */
bool HeatEngine::pick(Swap* out) const {
    if (!enabled_ || !warm() || !out) return false;

    /* Per class, since a swap stays in its class (heat.h: `cls_`). */
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
            /* Colder wins; on a tie, take from the layer holding the most resident units. */
            if (cold[c] < 0 || h < cold_h[c] || (h == cold_h[c] && rc > cold_res[c])) {
                cold[c] = (int32_t)i; cold_h[c] = h; cold_res[c] = rc;
            }
        } else {
            /* Hotter wins; on a tie, the layer with the fewest resident units, whose tokens miss
             * most often.
             *
             * Both Tier::SSD and Tier::HostPinned units are candidates: promotion is from wherever
             * the unit is, and the mover knows the source. Demotion never targets the file tier,
             * since a frequency counter cannot justify a bet that a unit will not be needed; it
             * targets the host pool, and only the planner assigns SSD. */
            if (hot[c] < 0 || h > hot_h[c] || (h == hot_h[c] && rc < hot_res[c])) {
                hot[c] = (int32_t)i; hot_h[c] = h; hot_res[c] = rc;
            }
        }
    }

    /* Both guards: the ratio stops trading one warm unit for another, and the additive floor
     * stops swaps against a zero-heat victim, which any ratio passes. Of the classes whose pair
     * passes, take the one with the hotter candidate. */
    int32_t best = -1;
    for (int32_t c = 0; c < n_cls_; ++c) {
        if (hot[(size_t)c] < 0 || cold[(size_t)c] < 0) continue;
        if (hot_h[(size_t)c] < cold_h[(size_t)c] * params_.hysteresis + params_.min_gain) continue;
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

/* A whole step's proposals from one pass. Marking a pair busy changes no heat or resident
 * count, so the m-th pair repeated pick() calls would return is the m-th entry at each end of the
 * same ranking. One pass keeps the `moves_per_step` best of each end and yields the same
 * proposals and stop counters as repeated picks, without a scan per move on the issuing thread. */
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
    const int k = params_.moves_per_step;
    if (k <= 0) return RAD_OK;

    /* Test the best pair from the layers' ends first. If it fails the guard, every pair does,
     * and the dispatch proposes nothing after one entry per layer. The ends' counts also give the
     * stop reasons. */
    int64_t n_off = 0, n_on = 0;
    float bar = 0.0f;
    {
        float hot_h = 0.0f, cold_h = 0.0f;
        for (size_t l = 0; l < end_ok_.size(); ++l) {
            if (!end_ok_[l]) end_rebuild(l);
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

    /* The ranking over every movable position. Each kLanes block is screened against the two
     * lists' last heats (one vector compare per position); only what passes meets the exact
     * comparison. The screen passes ties, so it never drops a unit the comparison would keep, and
     * since the lists hold a strict order's best k, visiting in position order gives the same
     * lists. The hot end is also screened against the guard bar: a unit below it cannot be
     * proposed against any victim.
     *
     * Each layer is first screened by its ends. Most resident units sit exactly at the cold end
     * (zero), which the position screen passes, so without this nearly every resident unit would
     * reach the exact comparison. A layer whose hot end is below the hot bar and whose cold end
     * cannot beat the cold list's last entry is skipped; on a heat tie the comparison falls to the
     * layer's shared resident count, then to the unit index, bounded by the layer's lowest.
     *
     * One pair of lists per class (heat.h: `cls_`). The screens use the loosest threshold across
     * classes, so they stay conservative; with one class they are exact. */
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
    /* Hot threshold: the lowest last entry of any full list, never below the guard. Cold
     * threshold: the highest last entry, or any heat while some list is not full. */
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
                if (pass[j] & 2) keep_best(cold + c * k, &n_cold[c], k, r, colder);
                else             keep_best(hot + c * k, &n_hot[c], k, r, hotter);
            }
        }
    }

    /* Each class's m-th hottest against its m-th coldest while that passes the guard (the first
     * failure ends the class), at most `k` per class, taken across classes hottest first: the
     * order repeated pick() calls would return. */
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
            if (h.h < cold[c * k + m].h * params_.hysteresis + params_.min_gain) {
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
        /* The caller owns the pair until it publishes or abandons it. */
        set_busy(s.promote, true);
        set_busy(s.demote, true);
        out->push_back(s);
        ++stats_.proposed;
        ++made;
    }
    if (short_of_budget) {
        /* See HeatStats for `refused` vs `no_candidate`. Pairs already proposed are busy and not
         * counted at either end. */
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
    /* A readmit is a unit demoted and then fetched back: link traffic both ways for nothing. A high
     * rate means the guards are too loose, though raising min_gain costs more bytes (heat.h). */
    s += fmt("  %llu distinct units promoted, %llu readmits (%.0f%% of promotions were of "
             "something already demoted)\n",
             (unsigned long long)stats_.distinct_promoted, (unsigned long long)stats_.readmits,
             stats_.promotions ? 100.0 * (double)stats_.readmits / (double)stats_.promotions : 0.0);

    /* The per-layer resident distribution: a layer starved of slots misses on nearly every token
     * however good its internal ranking, so the spread matters more than the mean. */
    if (!res_count_.empty()) {
        int32_t lo = res_count_[0], hi = res_count_[0];
        double sum = 0;
        for (int32_t v : res_count_) { lo = std::min(lo, v); hi = std::max(hi, v); sum += v; }
        const double mean = sum / (double)res_count_.size();
        double var = 0;
        for (int32_t v : res_count_) var += (v - mean) * (v - mean);
        s += fmt("  resident units a layer: min %d max %d mean %.1f sd %.1f\n", (int)lo, (int)hi,
                 mean, std::sqrt(var / (double)res_count_.size()));
        /* Per-layer counts, so slow layers in a kernel trace can be matched to starved ones. */
        s += "  per layer:";
        for (size_t i = 0; i < res_count_.size(); ++i) {
            if (i % 12 == 0) s += fmt("\n   L%-3zu ", i);
            s += fmt("%5d", (int)res_count_[i]);
        }
        s += "\n";
    }
    return s;
}

}  /* namespace rad */
