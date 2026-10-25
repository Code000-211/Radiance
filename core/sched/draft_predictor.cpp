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
        /* The workload's estimate is part of what a reset means: a new table starts from the prior,
         * so a slot entering it has nothing else to start from. */
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
            pop_a_[j] = p_.prior;
            pop_ess_[j] = p_.prior_weight;
            pop_n_[j] = 0;
        }
        for (int32_t i = 0; i < cap_; ++i) start(states_[i], 0, 0);
    }
    return RAD_OK;
}

int DraftPredictModel::grow(DraftPredictState* states, int32_t cap) {
    if (cap < cap_ || (cap > 0 && !states)) {
        RAD_ERR("draft predictor: cannot grow %d slots to %d", (int)cap_, (int)cap);
        return RAD_E_INVAL;
    }
    for (int32_t i = cap_; i < cap; ++i) start(states[i], 0, 0);
    states_ = states;
    cap_ = cap;
    return RAD_OK;
}

void DraftPredictModel::start(DraftPredictState& s, uint64_t req_id, uint32_t slot_gen) const {
    s.req_id = req_id;
    s.slot_gen = slot_gen;
    /* A slot with no labels of its own starts where the workload was measured, not at the flat
     * prior: acceptance is a property of what the request writes, and E(K) has no other way to see
     * it before this request produced a label (the class visibility, see the header). With no
     * workload evidence this is the prior unchanged, and its own labels move it from there. */
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
        s.a[j] = pop_a_[j];
        s.n[j] = 0;
        s.ess[j] = pop_ess_[j];
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
    /* Validate every row before anything moves: a view refused at its last row must not have reset
     * a live slot on the way there -- that would make an older cycle unobservable. */
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const DraftSeqView& v = view.seq[i];
        if (v.slot < 0 || v.slot >= cap_) {
            RAD_ERR("draft predictor: request %llu names slot %d outside 0..%d",
                    (unsigned long long)v.req_id, (int)v.slot, (int)cap_ - 1);
            return RAD_E_INVAL;
        }
    }
    out->n_seq = view.n_seq;
    /* The cycle is part of the set's identity, not only its rows: draft_predictions_current
     * compares it first, so a set from an earlier cycle is not current even when every request
     * still matches. */
    out->cycle_id = view.cycle_id;
    out->version = p_.version;
    const int D = p_.max_depth;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const DraftSeqView& v = view.seq[i];
        DraftPredictState& s = states_[v.slot];
        if (s.req_id != v.req_id || s.slot_gen != v.slot_gen) start(s, v.req_id, v.slot_gen);

        out->req_id[i] = v.req_id;
        DraftPrediction& pr = out->seq[i];
        uint64_t valid = 0;
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
            const bool in = j < D;
            pr.a[j] = in ? s.a[j] : 0.0f;
            pr.unc[j] = in ? unc_at(s, j) : 0.0f;
            /* The contract's n counts real observations; the effective sample size is this
             * model's own and only shapes unc[] above. */
            pr.n[j] = in ? s.n[j] : (uint16_t)0;
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

bool DraftPredictModel::learn(DraftPredictState& s, const DraftSeqObservation& o) const {
    /* Another drafter's matches are not this head's acceptance, and a row nobody drafted measured
     * nothing. The gate is the provenance of the window the verify consumed (`label_origin`), not
     * the plan this observation names: with pipeline_depth > 0 the labels belong to the previous
     * cycle's publication, and a row that changed drafter between the two would otherwise teach
     * the head from a lookup's matches (or drop the head's). The plan's fill[] is the permission
     * (draft_origin_fits_plan); this is the fact. */
    if (o.label_origin != DraftOrigin::Head || o.label_generated == 0 || p_.max_depth <= 0)
        return false;

    /* The shared mask, not a second one: 1 on the accepted prefix, 0 at a real rejection, and no
     * label at a matched-but-uncommitted stop. A censor past the rejection does not erase it. */
    bool applied = false;
    const int D = p_.max_depth;
    for (int j = 0; j < (int)o.checked && j < D; ++j) {
        float x = 0.0f;
        if (!draft_label(o, j, &x)) break;
        const double w = settle_weight(s.n[j], (double)p_.prior_weight, alpha_);
        s.a[j] = (float)((double)s.a[j] + w * ((double)x - (double)s.a[j]));
        s.ess[j] = (float)settle_ess((double)s.ess[j], w);
        if (s.n[j] < UINT16_MAX) ++s.n[j];
        /* The same label at the workload's level: the level a slot with no labels starts from. Its
         * own n keeps this the workload's mean, not this request's. */
        const double pw = settle_weight(pop_n_[j], (double)p_.prior_weight, alpha_);
        pop_a_[j] = (float)((double)pop_a_[j] + pw * ((double)x - (double)pop_a_[j]));
        pop_ess_[j] = (float)settle_ess((double)pop_ess_[j], pw);
        if (pop_n_[j] < UINT16_MAX) ++pop_n_[j];
        applied = true;
    }
    /* Positions past the window the verify computed drift toward its deepest position, the same
     * basis and step as DraftController::adapt_observe. No count, no valid bit. */
    int verified = (int)o.verified;
    if (verified > D) verified = D;
    if (verified <= 0) return applied;
    const double edge = (double)s.a[verified - 1];
    for (int j = verified; j < D; ++j) {
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));
        applied = true;
    }
    return applied;
}

int DraftPredictModel::observe(const DraftObservation& obs) const {
    if (obs.n_seq < 0 || obs.n_seq > obs.cap) return RAD_E_INVAL;
    if (obs.n_seq > 0 && (!obs.req_id || !obs.slot || !obs.slot_gen || !obs.seq))
        return RAD_E_INVAL;
    if (obs.n_seq > 0 && !states_) {
        RAD_ERR("draft predictor: no state table configured");
        return RAD_E_INVAL;
    }
    /* The plan is the cycle's immutable record and passed the shared issuance gate, so validate it
     * with that same gate before draft_plan_identifies() reads its rows: an over-wide n_seq would
     * otherwise walk off the plan's arrays. No local bound stands in for the contract here. */
    if (obs.plan && draft_plan_valid(*obs.plan) != RAD_OK) {
        RAD_ERR("draft predictor: cycle %llu names a plan that is not valid",
                (unsigned long long)obs.cycle_id);
        return RAD_E_INVAL;
    }
    /* The contract's one lifetime gate: complete, plan-identified, valid rows, origin fits the
     * plan. It never consults the live slot, so a cycle that ran before a recycle stays
     * deliverable; a row that does not name a plan has no provenance and is refused. */
    if (!draft_observation_deliverable(obs)) {
        RAD_ERR("draft predictor: cycle %llu is not a deliverable observation",
                (unsigned long long)obs.cycle_id);
        return RAD_E_INVAL;
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
    }
    for (int32_t i = 0; i < obs.n_seq; ++i) {
        const int32_t slot = obs.slot[i];
        DraftPredictState& s = states_[slot];
        ++counts_.rows_attempted;
        /* A recycled slot belongs to another request now. The cycle stays deliverable, but its
         * rows must never merge into the new generation's estimate: skip them. */
        if (s.req_id != obs.req_id[i] || s.slot_gen != obs.slot_gen[i]) {
            ++counts_.rows_skipped;
            continue;
        }
        if (learn(s, obs.seq[i])) ++counts_.rows_applied;
        else ++counts_.rows_skipped;
    }
    return RAD_OK;
}

}  /* namespace rad */
