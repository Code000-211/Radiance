/* draft_fakes.h -- compiling fakes for the P00 contracts (core/sched/draft_types.h).
 *
 * Each hook is the smallest honest implementation of its seam, so a test can exercise the whole
 * cycle -- predict, estimate, choose, observe -- without a card, a scheduler or a policy. The
 * fakes are deliberately not smart: the policy prices its fixed baseline to prove the cost seam
 * is wired, and the cost model reports an invalid estimate until a test supplies measured
 * milliseconds. That keeps "no profile" a visible fallback rather than a fabricated price.
 *
 * A and B replace their fakes with core/sched/draft_cost.cpp and draft_predictor.cpp; tests that
 * need a real model use those once they exist. Fixture holds the caller-provided storage the
 * contracts require, sized for a handful of requests.
 */
#pragma once
#include "rad_test.h"

#include "sched/draft_types.h"

#include <cstring>

namespace draft_fake {

using namespace rad;

/* ================================================================== B: predictor */
/* A flat conditional acceptance per position. `valid` stays 0: nothing has been observed, so no
 * position claims to be learned. */
struct Predictor {
    float a = 0.8f;
    float unc = 0.05f;
    int   calls = 0;
};

inline int predict(void* ctx, const DraftBatchView& view, DraftPredictions* out) {
    auto* s = static_cast<Predictor*>(ctx);
    ++s->calls;
    if (!out || !out->seq || !out->req_id || out->cap < view.n_seq) return RAD_E_INVAL;
    out->n_seq = view.n_seq;
    out->version = 1;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        out->req_id[i] = view.seq[i].req_id;
        DraftPrediction& p = out->seq[i];
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
            p.a[j] = s->a;
            p.unc[j] = s->unc;
            p.n[j] = 0;
        }
        p.valid = 0;
    }
    return RAD_OK;
}

/* ================================================================== A: cost model */
/* `ms[k]` is a measured mean per deepest window once `measured` is set. Otherwise the estimate is
 * invalid: an unmeasured prior is not a duration and is never reported as milliseconds. */
struct CostModel {
    double ms[RAD_SCHED_MAX_SPEC + 1] = {};
    bool   measured = false;
    int    calls = 0;
};

inline int estimate(void* ctx, const DraftVerifyShape& shape, DraftCostEstimate* out) {
    auto* s = static_cast<CostModel*>(ctx);
    ++s->calls;
    if (!out) return RAD_E_INVAL;
    *out = DraftCostEstimate{};
    const int k = shape.verify_k;
    if (k < 0 || k > RAD_SCHED_MAX_SPEC) return RAD_E_INVAL;
    if (!s->measured || !(s->ms[k] > 0.0)) {
        out->source = DraftCostSource::ConservativeFallback;
        out->valid = false;   /* nothing measured, nothing priced */
        return RAD_OK;
    }
    out->ms = s->ms[k];
    out->unc_ms = 0.05 * s->ms[k];
    out->coverage = 4;
    out->source = DraftCostSource::Measured;
    out->valid = true;
    return RAD_OK;
}

/* ================================================================== C: policy */
/* Legacy: every eligible request drafts `legacy_k`, clamped by the ceiling and its remaining
 * output. Cost: asks the model to price that baseline; an invalid estimate leaves the baseline in
 * place and marks the plan Fallback/NoProfile. The coordinate search over requests is C's work,
 * not the fake's; the point here is the seam and the honest fallback. */
struct Policy {
    enum class Mode { Legacy, Cost };
    Mode mode = Mode::Legacy;
    int  legacy_k = 3;
    int  calls = 0;
};

inline int choose(void* ctx, const DraftBatchView& view, const DraftPredictions& pred,
                  const DraftCostModel& costs, DraftPlan* plan) {
    auto* s = static_cast<Policy*>(ctx);
    ++s->calls;
    (void)pred;   /* the fake does not use position estimates; cost and legacy share one K */
    if (!plan || !plan->k || !plan->rows || !plan->fill || plan->cap < view.n_seq ||
        !plan->req_id || !plan->slot || !plan->slot_gen)
        return RAD_E_INVAL;

    int want = s->legacy_k;
    if (want < 0) want = 0;
    if (want > view.k_ceiling) want = view.k_ceiling;

    plan->cycle_id = view.cycle_id;
    plan->policy_version = view.policy_version;
    plan->n_seq = view.n_seq;
    plan->k_ceiling = view.k_ceiling;
    plan->verify_k = 0;
    plan->source = s->mode == Policy::Mode::Cost ? DraftPlanSource::Cost : DraftPlanSource::Legacy;
    plan->fallback = DraftFallback::None;
    plan->probe = DraftProbe::None;

    int32_t eligible = 0;
    int32_t top = 0;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        const DraftSeqView& v = view.seq[i];
        plan->req_id[i] = v.req_id;
        plan->slot[i] = v.slot;
        plan->slot_gen[i] = v.slot_gen;
        plan->why[i] = DraftFallback::None;
        int k = 0;
        if (v.eligibility == DraftEligibility::Eligible) {
            ++eligible;
            k = want < v.remaining ? want : v.remaining;
            if (k < want) plan->why[i] = DraftFallback::Resources;
        }
        plan->k[i] = k;
        /* Who may fill the budget: the head when there is one to spend, nobody at K 0. The fake
         * never chooses the n-gram fallback; a test that wants that state sets it by hand. */
        plan->fill[i] = k > 0 ? DraftDrafter::Serial : DraftDrafter::None;
        if (k > top) top = k;
    }
    plan->max_k = top;

    if (s->mode == Policy::Mode::Cost) {
        if (!costs.estimate) return RAD_E_UNSUPPORTED;   /* no model wired: say so */
        DraftVerifyShape shape;
        shape.n_rows = eligible;
        shape.verify_k = top;
        shape.ctx_class = 0;
        for (int32_t i = 0; i < view.n_seq; ++i) {
            const int k = plan->k[i];
            shape.verify_tokens += 1 + k;
            for (int j = 1; j <= k; ++j) ++shape.rows[j];
        }
        DraftCostEstimate e;
        const int rc = costs.estimate(costs.ctx, shape, &e);
        if (rc < 0) return rc;
        if (!e.valid) {
            /* The baseline stays, visibly, instead of a price invented for the shape. */
            plan->source = DraftPlanSource::Fallback;
            plan->fallback = DraftFallback::NoProfile;
        }
    }

    if (draft_plan_rows(*plan) < 0) return RAD_E_INVAL;
    return draft_plan_valid(*plan);
}

/* ================================================================== the observer */
/* Records what it was handed and refuses a cycle the caller has not declared complete. */
struct Observer {
    int calls = 0;
    uint64_t cycle_id = 0;
    int32_t  accepted = 0;
    double   total_ms = 0.0;
};

inline int observe(void* ctx, const DraftObservation& obs, const DraftCostSample& sample) {
    auto* s = static_cast<Observer*>(ctx);
    if (!obs.complete) return RAD_E_INVAL;   /* only after a safe completion */
    ++s->calls;
    s->cycle_id = obs.cycle_id;
    s->accepted = 0;
    for (int32_t i = 0; i < obs.n_seq; ++i) s->accepted += obs.seq[i].accepted;
    s->total_ms = sample.total_ms;
    return RAD_OK;
}

/* ================================================================== storage */
/* The caller-provided arrays a plan, a prediction set and a small batch need. The scheduler owns
 * a ring of these; a test owns one on the stack. */
struct Fixture {
    static constexpr int32_t kMaxSeq = 8;

    DraftSeqView     seq[kMaxSeq];
    DraftBatchView   view;
    DraftPrediction  pred_seq[kMaxSeq];
    uint64_t         pred_id[kMaxSeq];
    DraftPredictions pred;
    uint64_t         req_id[kMaxSeq];
    int32_t          slot[kMaxSeq];
    uint32_t         slot_gen[kMaxSeq];
    int32_t          k[kMaxSeq];
    DraftDrafter     fill[kMaxSeq];
    DraftFallback    why[kMaxSeq];
    int32_t          rows[RAD_SCHED_MAX_SPEC + 1];
    DraftPlan        plan;

    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    Fixture() {
        for (DraftSeqView& v : seq) v = DraftSeqView{};
        view.seq = seq;
        view.cap = kMaxSeq;
        view.cycle_id = 1;
        view.k_ceiling = 6;
        pred.seq = pred_seq;
        pred.req_id = pred_id;
        pred.cap = kMaxSeq;
        plan.cap = kMaxSeq;
        plan.req_id = req_id;
        plan.slot = slot;
        plan.slot_gen = slot_gen;
        plan.k = k;
        plan.fill = fill;
        plan.why = why;
        plan.rows = rows;
    }

    /* Request i: identity `id`, `remaining` output tokens, and optionally ineligible. */
    DraftSeqView& add(int32_t i, uint64_t id, int32_t remaining = 64,
                      DraftEligibility e = DraftEligibility::Eligible) {
        DraftSeqView& v = seq[i];
        v = DraftSeqView{};
        v.req_id = id;
        v.slot = i;
        v.slot_gen = 1;
        v.ctx_len = 1024;
        v.ctx_class = draft_ctx_class(v.ctx_len);
        v.remaining = remaining;
        v.eligibility = e;
        view.n_seq = i + 1;
        return v;
    }
};

}  /* namespace draft_fake */
