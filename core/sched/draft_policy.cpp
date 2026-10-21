/* draft_policy.cpp -- the cost-based draft-depth policy. See draft_policy.h.
 *
 * The objective is the issue's: expected output tokens of the whole batch over its expected
 * batch time. A row that does not draft still emits its one token, so it adds 1 to the yield and
 * costs a verify row at K 0.
 */
#include "draft_policy.h"

#include <cmath>
#include <cstdint>

namespace rad {

namespace {

/* A request's own cap: no draft past its remaining output (the verified token takes one of
 * them), no draft past the declared ceiling, none at all when it may not draft. */
int32_t cap_of(const DraftSeqView& v, int32_t ceiling) {
    if (v.eligibility != DraftEligibility::Eligible) return 0;
    int32_t cap = v.remaining - 1;
    if (cap < 0) cap = 0;
    if (cap > ceiling) cap = ceiling;
    if (cap > RAD_SCHED_MAX_SPEC) cap = RAD_SCHED_MAX_SPEC;
    return cap;
}

/* The shape a candidate assignment verifies, with every k clamped to `clamp_to` when that is
 * >= 0. n_rows is every decode row; verify_tokens counts 1 + k_i for each of them, the
 * non-drafting rows included. */
void shape_of(const DraftBatchView& view, const int32_t* k, int32_t clamp_to,
              DraftVerifyShape* s) {
    *s = DraftVerifyShape{};
    s->n_rows = view.n_seq;
    s->prefill_tokens = view.prefill_tokens;
    bool sampled = false;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const int32_t want = k[i];
        const int32_t ki = clamp_to >= 0 && want > clamp_to ? clamp_to : want;
        if (ki > s->verify_k) s->verify_k = ki;
        s->verify_tokens += 1 + ki;
        for (int32_t j = 1; j <= ki; ++j) ++s->rows[j];
        if (view.seq[i].ctx_class > s->ctx_class) s->ctx_class = view.seq[i].ctx_class;
        if (view.seq[i].sampling == DraftSampling::Sampled) sampled = true;
    }
    s->sampling = sampled ? DraftSampling::Sampled : DraftSampling::Greedy;
}

}  /* namespace */

int DraftPlanner::configure(const DraftPolicyConfig& c) {
    if (c.k_ceiling < 0 || c.k_ceiling > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft policy: ceiling %d is outside 0..%d", (int)c.k_ceiling,
                (int)RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    /* The tier list is the cost choice's input; a legacy plan never reads it, so a deployment
     * whose window is shallower than the default tiers is not refused for them. */
    if (c.mode == DraftPolicyMode::Cost) {
        if (c.n_tiers < 1 || c.n_tiers > RAD_DRAFT_MAX_TIERS) {
            RAD_ERR("draft policy: %d tiers, outside 1..%d", (int)c.n_tiers,
                    (int)RAD_DRAFT_MAX_TIERS);
            return RAD_E_INVAL;
        }
        for (int32_t i = 0; i < c.n_tiers; ++i) {
            const int32_t t = c.tiers[i];
            if (t < 0 || t > c.k_ceiling) {
                RAD_ERR("draft policy: tier %d is outside 0..%d", (int)t, (int)c.k_ceiling);
                return RAD_E_INVAL;
            }
            if (i > 0 && t <= c.tiers[i - 1]) {
                RAD_ERR("draft policy: tiers must be ascending and unique (tier %d is %d after "
                        "%d)", (int)i, (int)t, (int)c.tiers[i - 1]);
                return RAD_E_INVAL;
            }
        }
        if (c.fallback_k < 0 || c.fallback_k > c.k_ceiling) return RAD_E_INVAL;
    }
    if (c.passes < 0 || c.passes > 8 || c.max_evals < 1) return RAD_E_INVAL;
    if (!std::isfinite(c.hyst) || c.hyst < 0.0 || c.hold < 0) return RAD_E_INVAL;
    c_ = c;
    reset();
    if (c_.mode == DraftPolicyMode::Cost)
        RAD_INFO("draft policy: cost, ceiling %d, tiers %d..%d (%d), a raise needs %.0f%% and %d "
                 "cycles", (int)c_.k_ceiling, (int)c_.tiers[0], (int)c_.tiers[c_.n_tiers - 1],
                 (int)c_.n_tiers, 100.0 * c_.hyst, (int)c_.hold);
    return RAD_OK;
}

void DraftPlanner::reset() {
    last_k_ = 0;
    held_   = 0;
    chosen_ = false;
}

double DraftPlanner::expect(const DraftPredictions& pred, int32_t i, int32_t cap, int32_t k) {
    if (k > cap) k = cap;
    const DraftPrediction& p = pred.seq[i];
    double e = 1.0, run = 1.0;
    for (int32_t d = 1; d <= k; ++d) {
        run *= (double)p.a[d - 1];
        e += run;
    }
    return e;
}

int32_t DraftPlanner::total_k(const int32_t* k, int32_t n, int32_t* max_k) {
    int32_t sum = 0, top = 0;
    for (int32_t i = 0; i < n; ++i) {
        sum += k[i];
        if (k[i] > top) top = k[i];
    }
    if (max_k) *max_k = top;
    return sum;
}

int DraftPlanner::price(const DraftBatchView& view, const DraftPredictions& pred,
                        const DraftCostModel& costs, const int32_t* k, int32_t clamp_to,
                        int* evals, Priced* out) const {
    *out = Priced{};
    if (*evals >= c_.max_evals) return RAD_OK;
    ++*evals;

    DraftVerifyShape shape;
    shape_of(view, k, clamp_to, &shape);

    DraftCostEstimate est;
    const int rc = costs.estimate(costs.ctx, shape, &est);
    if (rc < 0) return rc;
    if (!est.valid || !(est.ms > 0.0) || !std::isfinite(est.ms)) return RAD_OK;   /* not priced */

    double y = 0.0;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        int32_t ki = k[i];
        if (clamp_to >= 0 && ki > clamp_to) ki = clamp_to;
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        y += expect(pred, i, c, ki);
    }
    out->yield = y;
    out->ms    = est.ms;
    out->unc   = est.unc_ms > 0.0 && std::isfinite(est.unc_ms) ? est.unc_ms : 0.0;
    out->valid = true;
    return RAD_OK;
}

int DraftPlanner::finish(const DraftBatchView& view, int32_t max_k, DraftPlanSource src,
                         DraftFallback fb, DraftPlan* out) {
    out->cycle_id       = view.cycle_id;
    out->policy_version = view.policy_version ? view.policy_version : RAD_DRAFT_POLICY_VERSION;
    out->n_seq          = view.n_seq;
    out->k_ceiling      = view.k_ceiling;
    out->max_k          = max_k;
    out->source         = src;
    out->fallback       = fb;
    out->probe          = DraftProbe::None;
    if (out->cap < view.n_seq) return RAD_E_INVAL;
    if (draft_plan_rows(*out) < 0) return RAD_E_INVAL;
    return draft_plan_valid(*out);
}

int DraftPlanner::choose(const DraftBatchView& view, const DraftPredictions& pred,
                         const DraftCostModel& costs, DraftPlan* out) {
    if (!out || !out->k || !out->rows || !out->req_id || !out->slot || !out->slot_gen ||
        !out->why)
        return RAD_E_INVAL;
    if (view.n_seq < 0 || view.n_seq > out->cap) return RAD_E_INVAL;
    if (pred.n_seq < view.n_seq || pred.cap < view.n_seq || !pred.seq) return RAD_E_INVAL;

    /* Identity first: a plan names its requests even where it drafts nothing. */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        out->req_id[i]   = view.seq[i].req_id;
        out->slot[i]     = view.seq[i].slot;
        out->slot_gen[i] = view.seq[i].slot_gen;
        out->why[i]      = DraftFallback::None;
        out->k[i]        = 0;
    }
    /* verify_k is the window the paired step verifies: the previous cycle's choice. */
    out->verify_k = last_k_;

    if (c_.mode == DraftPolicyMode::Legacy) {
        const int32_t want = c_.legacy_k > 0 ? c_.legacy_k : c_.k_ceiling;
        int32_t top = 0;
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = want < c ? want : c;
            /* An eligible request bounded below the legacy depth was cut, by its remaining
             * output or the ceiling; an ineligible one drafts nothing by its own state. */
            if (view.seq[i].eligibility == DraftEligibility::Eligible && c < want)
                out->why[i] = DraftFallback::Resources;
            if (out->k[i] > top) top = out->k[i];
        }
        return finish(view, top, DraftPlanSource::Legacy, DraftFallback::None, out);
    }

    if (!costs.estimate) return RAD_E_UNSUPPORTED;

    int evals = 0;

    /* Stage 1: the homogeneous baselines, one estimate call each. */
    Priced  best_p;
    int32_t best_tier = 0;
    bool    any = false;
    for (int32_t t = 0; t < c_.n_tiers; ++t) {
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = c_.tiers[t] < c ? c_.tiers[t] : c;
        }
        Priced p;
        const int rc = price(view, pred, costs, out->k, -1, &evals, &p);
        if (rc < 0) return rc;
        if (p.valid && (!any || p.yield / p.ms > best_p.yield / best_p.ms)) {
            best_p    = p;
            best_tier = c_.tiers[t];
            any       = true;
        }
    }

    if (!any) {
        /* No shape could be priced: a named baseline, not an invented price. */
        int32_t top = 0;
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
            out->k[i] = c_.fallback_k < c ? c_.fallback_k : c;
            if (out->k[i] > top) top = out->k[i];
        }
        return finish(view, top, DraftPlanSource::Fallback, DraftFallback::UnknownShape, out);
    }

    /* Take the winning baseline as the search's starting assignment. */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const int32_t c = cap_of(view.seq[i], c_.k_ceiling);
        out->k[i] = best_tier < c ? best_tier : c;
    }
    double best_score = best_p.yield / best_p.ms;

    /* Stage 2: deterministic coordinate passes. A pass walks the requests in index order and
     * keeps a tier only when it improves the whole batch's tokens per millisecond. */
    for (int32_t pass = 0; pass < c_.passes; ++pass) {
        bool moved = false;
        for (int32_t i = 0; i < view.n_seq && evals < c_.max_evals; ++i) {
            const int32_t cap_i = cap_of(view.seq[i], c_.k_ceiling);
            if (cap_i <= 0) continue;
            const int32_t keep = out->k[i];
            int32_t accepted = keep;
            for (int32_t t = 0; t < c_.n_tiers && evals < c_.max_evals; ++t) {
                const int32_t cand = c_.tiers[t] < cap_i ? c_.tiers[t] : cap_i;
                if (cand == accepted) continue;
                out->k[i] = cand;
                Priced p;
                const int rc = price(view, pred, costs, out->k, -1, &evals, &p);
                if (rc < 0) { out->k[i] = keep; return rc; }
                if (p.valid && p.yield / p.ms > best_score) {
                    best_score = p.yield / p.ms;
                    best_p     = p;
                    accepted   = cand;
                    moved      = true;
                }
            }
            out->k[i] = accepted;
        }
        if (!moved || evals >= c_.max_evals) break;
    }

    /* Budgets: a positive token or KV budget caps the summed drafts. Scaling every K down is
     * the resource cut that may happen at once. */
    int32_t budget = -1;
    if (view.token_budget > 0) budget = view.token_budget;
    if (view.kv_budget > 0 && (budget < 0 || view.kv_budget < budget)) budget = view.kv_budget;
    if (budget >= 0) {
        int32_t top = 0;
        const int32_t sum = total_k(out->k, view.n_seq, &top);
        if (sum > budget && sum > 0) {
            for (int32_t i = 0; i < view.n_seq; ++i) {
                if (out->k[i] <= 0) continue;
                const int32_t scaled = (int32_t)((int64_t)out->k[i] * budget / sum);
                if (scaled < out->k[i]) {
                    out->k[i] = scaled;
                    out->why[i] = DraftFallback::Resources;
                }
            }
        }
    }

    int32_t top = 0;
    total_k(out->k, view.n_seq, &top);

    /* held_ counts the cycles that have run at last_k_; the plan issued here is not one of them
     * yet. */
    if (chosen_) { if (held_ < INT32_MAX) ++held_; } else held_ = 0;

    /* Hysteresis and hold: a raise waits, a reduction is immediate. A raise is judged
     * pessimistically (`ms + unc`), the incumbent at its known cost; with no price for the
     * incumbent it is refused. */
    if (chosen_ && top > last_k_) {
        Priced incumbent;
        const int rc = price(view, pred, costs, out->k, last_k_, &evals, &incumbent);
        if (rc < 0) return rc;
        const double raised = best_p.yield / (best_p.ms + best_p.unc);
        if (held_ < c_.hold || !incumbent.valid ||
            !(raised > incumbent.yield / incumbent.ms * (1.0 + c_.hyst))) {
            for (int32_t i = 0; i < view.n_seq; ++i)
                if (out->k[i] > last_k_) out->k[i] = last_k_;
            top = last_k_;
        }
    }
    if (top != last_k_) held_ = 0;   /* a new depth starts a new hold */
    last_k_ = top;
    chosen_ = true;

    return finish(view, top, DraftPlanSource::Cost, DraftFallback::None, out);
}

}  /* namespace rad */
