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
    p_ = p;
    states_ = states;
    cap_ = cap;
    alpha_ = 1.0 - std::exp2(-1.0 / (double)p_.half_life);
    for (int32_t i = 0; i < cap_; ++i) start(states_[i], 0, 0);
    return RAD_OK;
}

void DraftPredictModel::start(DraftPredictState& s, uint64_t req_id, uint32_t slot_gen) const {
    s.req_id = req_id;
    s.slot_gen = slot_gen;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
        s.a[j] = p_.prior;
        s.n[j] = 0;
    }
}

float DraftPredictModel::unc_at(const DraftPredictState& s, int j) const {
    double p = (double)s.a[j];
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    /* A Beta-like half-width: the prior's pseudo-observations and the real ones both narrow it,
     * and the +1 keeps a run of identical labels from claiming zero uncertainty. */
    const double denom = (double)s.n[j] + (double)p_.prior_weight + 1.0;
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
            pr.n[j] = in ? s.n[j] : 0;
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
                              DraftProposalSource src) const {
    /* Another drafter's matches are not this head's acceptance, and a K0 row measured nothing. */
    if (src != DraftProposalSource::Head || o.generated == 0 || p_.max_depth <= 0) return;

    int accepted = (int)o.accepted;
    if (accepted > (int)o.checked) accepted = (int)o.checked;  /* the prefix cannot outrun the labels */
    /* A censor ends the cycle at the accepted prefix: the position after it was never a real
     * rejection. Without a censor the first rejection is the only zero label; `checked` past it --
     * a caller's mistake -- buys no more evidence. */
    int labeled = (int)o.checked;
    if (labeled > accepted + 1) labeled = accepted + 1;
    if (o.censor != DraftCensor::None) labeled = accepted;
    const int D = p_.max_depth;
    const int upto = labeled < D ? labeled : D;
    for (int j = 0; j < upto; ++j) {
        const double w = settle_weight(s.n[j], (double)p_.prior_weight, alpha_);
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* Positions past the deepest label drift toward it. No count, no valid bit: this is a prior
     * moving with the text, not an observation. */
    const int edge_at = upto - 1;
    if (edge_at < 0) return;
    const double edge = (double)s.a[edge_at];
    for (int j = upto; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));
}

int DraftPredictModel::observe(const DraftObservation& obs,
                               const DraftProposalSource* source) const {
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
    if (obs.policy_version != p_.version) {
        RAD_ERR("draft predictor: cycle %llu was planned under policy version %u, not %u",
                (unsigned long long)obs.cycle_id, (unsigned)obs.policy_version,
                (unsigned)p_.version);
        return RAD_E_INVAL;
    }
    /* A plan stays alive while issue_ahead() may name it; an observation that names another
     * cycle's plan, or another generation's row, is stale and teaches nothing. */
    if (obs.plan) {
        const DraftPlan& plan = *obs.plan;
        if (plan.cycle_id != obs.cycle_id || plan.policy_version != obs.policy_version ||
            plan.n_seq != obs.n_seq ||
            (obs.n_seq > 0 && (!plan.req_id || !plan.slot || !plan.slot_gen))) {
            RAD_ERR("draft predictor: observation %llu does not match its plan",
                    (unsigned long long)obs.cycle_id);
            return RAD_E_INVAL;
        }
        for (int32_t i = 0; i < obs.n_seq; ++i) {
            if (plan.req_id[i] != obs.req_id[i] || plan.slot[i] != obs.slot[i] ||
                plan.slot_gen[i] != obs.slot_gen[i]) {
                RAD_ERR("draft predictor: observation row %d differs from its plan", (int)i);
                return RAD_E_INVAL;
            }
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
        const DraftPredictState& s = states_[slot];
        if (s.req_id != obs.req_id[i] || s.slot_gen != obs.slot_gen[i]) {
            RAD_ERR("draft predictor: slot %d holds request %llu/%u, not %llu/%u", (int)slot,
                    (unsigned long long)s.req_id, (unsigned)s.slot_gen,
                    (unsigned long long)obs.req_id[i], (unsigned)obs.slot_gen[i]);
            return RAD_E_INVAL;
        }
    }
    for (int32_t i = 0; i < obs.n_seq; ++i) {
        learn(states_[obs.slot[i]], obs.seq[i],
              source ? source[i] : DraftProposalSource::Head);
    }
    return RAD_OK;
}

}  /* namespace rad */
