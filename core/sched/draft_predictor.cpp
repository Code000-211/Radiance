/* draft_predictor.cpp -- B's position-wise acceptance model. See draft_predictor.h. */
#include "draft_predictor.h"

#include <cmath>
#include <cstdint>

namespace rad {

namespace {

/* The (n+1)-th label's weight: a running mean over prior and evidence until 1 / (n + k + 1) falls
 * below the moving average's step at the half-life, and that average after. The same arithmetic as
 * DraftController::adapt_observe (draft.cpp), so a sequence keeps its meaning across the seam. */
double settle_weight(uint16_t n, double prior_weight, double alpha) {
    const double mean = 1.0 / ((double)n + prior_weight + 1.0);
    return mean > alpha ? mean : alpha;
}

/* The effective sample size after one more label of weight w. Past labels keep their relative
 * weights, so mixing a new observation with a past distribution of ESS `e` gives
 * 1 / (w^2 + (1-w)^2 / e); with a constant w = alpha it converges to (2 - alpha) / alpha. */
double settle_ess(double e, double w) {
    if (e <= 0.0) return 1.0;
    const double q = 1.0 - w;
    return 1.0 / (w * w + q * q / e);
}

}  /* namespace */

int DraftPredictModel::configure(const Params& p, DraftPredictState* states, int32_t cap) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft predictor: %d positions are outside 0..%d", (int)p.max_depth,
                RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    /* Written as negated comparisons so a NaN parameter is refused, not accepted. */
    if (!(p.prior >= 0.0f && p.prior <= 1.0f)) {
        RAD_ERR("draft predictor: prior %g is not a probability", (double)p.prior);
        return RAD_E_INVAL;
    }
    if (!(p.half_life > 0.0f) || !(p.prior_weight >= 0.0f) || !(p.unc_z >= 0.0f)) {
        RAD_ERR("draft predictor: half_life %g, prior_weight %g, unc_z %g are not usable",
                (double)p.half_life, (double)p.prior_weight, (double)p.unc_z);
        return RAD_E_INVAL;
    }
    if (cap < 0 || (cap > 0 && !states)) {
        RAD_ERR("draft predictor: %d slots but no state table", (int)cap);
        return RAD_E_INVAL;
    }
    /* A new table, or a change to what shapes the estimate, starts over. A version bump alone does
     * not: policies are provenance, and a live cycle planned under the old one must still teach. */
    const bool reset = states_ != states || cap_ != cap || p.max_depth != p_.max_depth ||
                       p.prior != p_.prior || p.half_life != p_.half_life ||
                       p.prior_weight != p_.prior_weight;
    p_ = p;
    states_ = states;
    cap_ = cap;
    alpha_ = 1.0 - std::exp2(-1.0 / (double)p_.half_life);
    if (reset) {
        for (int32_t i = 0; i < cap_; ++i) start(states_[i], 0, 0);
    }
    return RAD_OK;
}

void DraftPredictModel::start(DraftPredictState& s, uint64_t req_id, uint32_t slot_gen) const {
    s.req_id = req_id;
    s.slot_gen = slot_gen;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
        s.a[j] = p_.prior;
        s.n[j] = 0;
        s.ess[j] = p_.prior_weight;
    }
}

float DraftPredictModel::unc_at(const DraftPredictState& s, int j) const {
    double p = (double)s.a[j];
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    /* A Beta-like half-width over the effective evidence, not the raw count: an EMA with a
     * half-life cannot carry more than (2 - alpha) / alpha labels, and a policy reading n must
     * see that ceiling rather than a number that grows forever. */
    const double denom = (double)s.ess[j] + 1.0;
    double u = (double)p_.unc_z * std::sqrt(p * (1.0 - p) / denom);
    if (u > 1.0) u = 1.0;
    return (float)u;
}

int DraftPredictModel::predict(const DraftBatchView& view, DraftPredictions* out) const {
    if (!out || !out->seq || !out->req_id) return RAD_E_INVAL;
    if (view.n_seq < 0 || view.cap < 0 || view.n_seq > view.cap) return RAD_E_INVAL;
    if (view.n_seq > 0 && !view.seq) return RAD_E_INVAL;
    if (view.n_seq > out->cap) {
        RAD_ERR("draft predictor: %d requests but predictions hold %d", (int)view.n_seq,
                (int)out->cap);
        return RAD_E_INVAL;
    }
    /* A prediction is for the policy that asks for it; a plan made under another version is not
     * this prediction's. Observations are judged by their plan instead -- see observe(). */
    if (view.policy_version != p_.version) {
        RAD_ERR("draft predictor: view is policy version %u, this model %u",
                (unsigned)view.policy_version, (unsigned)p_.version);
        return RAD_E_INVAL;
    }
    out->n_seq = view.n_seq;
    out->version = p_.version;
    const int D = p_.max_depth;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const DraftSeqView& v = view.seq[i];
        if (v.slot < 0 || v.slot >= cap_) {
            RAD_ERR("draft predictor: request %llu names slot %d outside 0..%d",
                    (unsigned long long)v.req_id, (int)v.slot, (int)cap_ - 1);
            return RAD_E_INVAL;
        }
        DraftPredictState& s = states_[v.slot];
        if (s.req_id != v.req_id || s.slot_gen != v.slot_gen) start(s, v.req_id, v.slot_gen);

        out->req_id[i] = v.req_id;
        DraftPrediction& pr = out->seq[i];
        uint64_t valid = 0;
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
            const bool in = j < D;
            pr.a[j] = in ? s.a[j] : 0.0f;
            pr.unc[j] = in ? unc_at(s, j) : 0.0f;
            double data = in ? (double)s.ess[j] - (double)p_.prior_weight : 0.0;
            if (data < 0.0) data = 0.0;
            const long count = std::lround(data);
            pr.n[j] = (uint16_t)(count > 65535 ? 65535 : count);
            if (in && s.n[j] > 0) valid |= 1ull << j;
        }
        pr.valid = valid;
    }
    return RAD_OK;
}

int DraftPredictModel::predict_hook(void* self, const DraftBatchView& view,
                                     DraftPredictions* out) {
    if (!self) return RAD_E_INVAL;
    return static_cast<DraftPredictModel*>(self)->predict(view, out);
}

void DraftPredictModel::learn(DraftPredictState& s, const DraftSeqObservation& o,
                              DraftDrafter src) const {
    /* Another drafter's matches are not this head's acceptance, and a K0 row measured nothing. */
    if (src != DraftDrafter::Serial || o.generated == 0 || p_.max_depth <= 0) return;

    int accepted = (int)o.accepted;
    if (accepted > (int)o.checked) accepted = (int)o.checked;  /* observe() refuses this already */
    /* draft_label()'s mask: checked ends at the first rejection, and nothing past it was measured.
     * A censor is a reason, not a position -- a rejection checked before it stays a label. */
    int labeled = (int)o.checked;
    if (labeled > accepted + 1) labeled = accepted + 1;
    const int D = p_.max_depth;
    if (labeled > D) labeled = D;
    for (int j = 0; j < labeled; ++j) {
        const double w = settle_weight(s.n[j], (double)p_.prior_weight, alpha_);
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        s.ess[j] = (float)settle_ess((double)s.ess[j], w);
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* Positions past the window the verify computed drift toward its deepest position, the same
     * basis and step as DraftController::adapt_observe. No count, no valid bit. */
    int verified = (int)o.verified;
    if (verified > D) verified = D;
    if (verified <= 0) return;
    const double edge = (double)s.a[verified - 1];
    for (int j = verified; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));
}

int DraftPredictModel::observe(const DraftObservation& obs) const {
    if (!obs.complete) {
        RAD_ERR("draft predictor: cycle %llu is not complete; no label is taken from it",
                (unsigned long long)obs.cycle_id);
        return RAD_E_INVAL;
    }
    if (obs.n_seq < 0 || obs.n_seq > obs.cap) return RAD_E_INVAL;
    if (obs.n_seq > 0 && (!obs.req_id || !obs.slot || !obs.slot_gen || !obs.seq))
        return RAD_E_INVAL;
    if (obs.n_seq > 0 && !states_) {
        RAD_ERR("draft predictor: no state table configured");
        return RAD_E_INVAL;
    }
    /* Origin, cycle and identity are part of the immutable cycle contract. Without the plan a row
     * has no provenance, and a missing array must never be read as "the head". */
    if (obs.n_seq > 0 && !obs.plan) {
        RAD_ERR("draft predictor: cycle %llu names no plan; its origin is unknown",
                (unsigned long long)obs.cycle_id);
        return RAD_E_INVAL;
    }
    if (obs.plan) {
        const DraftPlan& plan = *obs.plan;
        /* draft_plan_identifies() reads the plan's [cap] rows; its n_seq must fit that cap. */
        if (obs.n_seq > plan.cap) {
            RAD_ERR("draft predictor: cycle %llu has %d rows but its plan holds %d",
                    (unsigned long long)obs.cycle_id, (int)obs.n_seq, (int)plan.cap);
            return RAD_E_INVAL;
        }
        if (!draft_plan_identifies(plan, obs)) {
            RAD_ERR("draft predictor: observation %llu does not match its plan",
                    (unsigned long long)obs.cycle_id);
            return RAD_E_INVAL;
        }
        if (obs.n_seq > 0 && !plan.fill) {
            RAD_ERR("draft predictor: cycle %llu does not say who drafted its rows",
                    (unsigned long long)obs.cycle_id);
            return RAD_E_INVAL;
        }
    }
    /* Validate every row before any state moves: a half-applied cycle is not a smaller update,
     * it is a mix of two cycles. */
    for (int32_t i = 0; i < obs.n_seq; ++i) {
        const int32_t slot = obs.slot[i];
        if (slot < 0 || slot >= cap_) {
            RAD_ERR("draft predictor: observed slot %d is outside 0..%d", (int)slot,
                    (int)cap_ - 1);
            return RAD_E_INVAL;
        }
        const DraftSeqObservation& o = obs.seq[i];
        if (o.accepted > o.checked || o.checked > o.verified) {
            RAD_ERR("draft predictor: row %d has accepted %u, checked %u, verified %u",
                    (int)i, (unsigned)o.accepted, (unsigned)o.checked, (unsigned)o.verified);
            return RAD_E_INVAL;
        }
        const DraftPredictState& s = states_[slot];
        if (s.req_id != obs.req_id[i] || s.slot_gen != obs.slot_gen[i]) {
            RAD_ERR("draft predictor: slot %d holds request %llu/%u, not %llu/%u", (int)slot,
                    (unsigned long long)s.req_id, (unsigned)s.slot_gen,
                    (unsigned long long)obs.req_id[i], (unsigned)obs.slot_gen[i]);
            return RAD_E_INVAL;
        }
    }
    for (int32_t i = 0; i < obs.n_seq; ++i)
        learn(states_[obs.slot[i]], obs.seq[i], obs.plan ? obs.plan->fill[i] : DraftDrafter::None);
    return RAD_OK;
}

}  /* namespace rad */
