/* draft.cpp -- the static draft-depth policy and the adaptive one (`--draft-window adaptive`).
 * See draft.h.
 */
#include "draft.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace rad {

namespace {

/* Step-cost prior, a shape rather than a measurement: each draft adds 6% to a step (one serial
 * head round plus one verify row on a weight-bound step). Once anything is measured only its
 * ratios are used: an unmeasured depth is the nearest measured one scaled by prior(d) / prior(d'). */
constexpr double kCostSlope = 0.06;
double cost_prior(int d) { return 1.0 + kCostSlope * (double)d; }

/* A step-time mean is a running mean for the first steps, then a moving average with a half-life
 * of 32 steps: long enough to average out one-off costs (a tier copy, a recording, a prefix-cache
 * trim), short enough to follow a context growing by a few thousand tokens over a reply. */
constexpr double kCostHalfLife = 32.0;
/* Once a key has 4 samples, one step counts for at most 4x its mean. A step that waited on
 * something else (a first recording, an allocation, a host stall) says nothing about its depth's
 * cost. Below 4 samples the mean is not a trustworthy reference to clip against. */
constexpr double kCostOutlier = 4.0;
constexpr int64_t kCostClipAfter = 4;

/* The prior weighs as much as this many verifies. The (n+1)-th verify moves an estimate by
 * 1 / (n + k + 1), a running mean over prior and evidence, until that falls below the half-life's
 * step and the moving average takes over. Without it a sequence would spend its first ~16
 * verifies at the prior's window, most of a short answer. */
constexpr double kPriorWeight = 4.0;

/* The weight of the (n+1)-th sample in an estimate that is a running mean until 1 / (n + 1)
 * falls below a moving average's step at half-life `h` samples, and that average after. */
double settle_weight(double h, int64_t n) {
    return std::fmax(1.0 - std::exp2(-1.0 / h), 1.0 / (double)(n + 1));
}

}  /* namespace */

int DraftController::configure(const Params& p) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft depth %d is outside 0..%d -- the verify kernel band is %d query rows",
                p.max_depth, RAD_SCHED_MAX_SPEC, RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    p_ = p;
    tot_drafted_ = tot_accepted_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) { reach_[j] = 0; hit_[j] = 0; }
    seen_ = 0;

    adapt_ = p.adapt && p.max_depth > 0;
    alpha_ = 1.0 - std::exp2(-1.0 / kHalfLife);
    step_d_ = 0;
    step_w_ = 0;
    step_held_ = 0;
    prior_n_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) prior_[j] = 0.0f;
    for (int d = 0; d <= RAD_SCHED_MAX_SPEC; ++d) {
        by_depth_[d] = 0;
        for (int c = 0; c < kCostClasses; ++c) { cost_ms_[d][c] = 0.0; cost_n_[d][c] = 0; }
    }
    if (adapt_)
        RAD_INFO("draft depth: adaptive, %d..%d; a sequence stops at the first draft emitted less "
                 "than %.0f%% of the time, and the depth moves at +%.0f%% at most every %d steps",
                 kMinDepth, p_.max_depth, 100.0 * kDraftMinChance, 100.0 * kHyst, kHold);

    /* The cost policy's seam. It stays off unless a mode was asked for, and it is refused --
     * not silently mapped to legacy -- when either hook it chooses from is missing. */
    DraftPolicyConfig pc = p.policy;
    pc.k_ceiling = p.max_depth;
    pc.max_seqs = p.max_seqs;
    const bool want_cost = pc.mode == DraftPolicyMode::Cost;
    if (want_cost && (!p.predictor.predict || !p.costs.estimate)) {
        RAD_ERR("--draft-policy cost needs a predictor and a cost model: this build has %s%s",
                p.predictor.predict ? "" : "no predictor", p.costs.estimate ? "" : "no cost model");
        p_.policy.mode = DraftPolicyMode::Legacy;
        planner_ = DraftPlanner{};
        return RAD_E_UNSUPPORTED;
    }
    const int rc = planner_.configure(pc);
    if (rc < 0) return rc;
    plan_cur_ = -1;
    cost_failed_ = false;
    const size_t cap = (size_t)(p.max_seqs > 0 ? p.max_seqs : 0);
    for (PlanSlot& s : plan_slots_) {
        s.id.assign(cap, 0);
        s.slot.assign(cap, -1);
        s.gen.assign(cap, 0);
        s.k.assign(cap, 0);
        s.why.assign(cap, DraftFallback::None);
        s.rows.assign(RAD_SCHED_MAX_SPEC + 1, 0);
        s.ctx_class.assign(cap, 0);
        s.history_tokens.assign(cap, 0);
        s.fill.assign(cap, DraftDrafter::None);
        s.plan          = DraftPlan{};
        s.plan.cap      = (int)cap;
        s.plan.req_id   = s.id.data();
        s.plan.slot     = s.slot.data();
        s.plan.slot_gen = s.gen.data();
        s.plan.k        = s.k.data();
        s.plan.why      = s.why.data();
        s.plan.rows     = s.rows.data();
        s.plan.ctx_class      = s.ctx_class.data();
        s.plan.history_tokens = s.history_tokens.data();
        s.plan.fill           = s.fill.data();
    }
    return RAD_OK;
}

int DraftController::open_cycle(const DraftBatchView& view, DraftPredictions* preds) {
    plan_cur_ = -1;
    if (!cost_policy() || !preds) return RAD_OK;
    /* The next cycle writes the other set, so the plan handed out last cycle keeps its rows. */
    const int next = plan_cur_prev_ ^ 1;
    PlanSlot& slot = plan_slots_[next];
    if (view.n_seq > slot.plan.cap) return RAD_E_INVAL;
    if (preds->cap < view.n_seq || !preds->seq || !preds->req_id) return RAD_E_INVAL;

    int rc = p_.predictor.predict(p_.predictor.ctx, view, preds);
    if (rc < 0) return rc;
    /* A prediction set describes the view it was computed for: a stale version or a row list
     * that is not this view's would let another cycle's numbers choose this plan. */
    if (!draft_predictions_current(view, *preds)) return RAD_E_INVAL;
    rc = planner_.choose(view, *preds, p_.costs, &slot.plan);
    if (rc < 0) return rc;
    plan_cur_      = next;
    plan_cur_prev_ = next;
    return RAD_OK;
}

int DraftController::planned_k(uint64_t req_id, int32_t slot, uint32_t slot_gen) const {
    const DraftPlan* p = plan();
    if (!p) return -1;
    for (int32_t i = 0; i < p->n_seq; ++i)
        if (p->req_id[i] == req_id && p->slot[i] == slot && p->slot_gen[i] == slot_gen)
            return p->k[i];
    return -1;
}

void DraftController::observe(int drafted, int accepted, int raw_match) {
    if (drafted <= 0) return;
    if (drafted > RAD_SCHED_MAX_SPEC) drafted = RAD_SCHED_MAX_SPEC;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    /* The raw tally is the uncensored comparison: a censored stop leaves it above `accepted`, and
     * it can never sit below what was committed. */
    if (raw_match < accepted) raw_match = accepted;
    if (raw_match > drafted) raw_match = drafted;
    tot_drafted_  += drafted;
    tot_accepted_ += accepted;
    tot_raw_      += raw_match;

    /* Position j is evaluated only if every earlier position was accepted: 0..accepted-1 are
     * evaluated and accepted, position `accepted` (if drafted) evaluated and rejected, and nothing
     * past it is reached. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        ++reach_[j];
        if (j < accepted) ++hit_[j];
    }
    if (drafted > seen_) seen_ = drafted;
    if (adapt_) ++by_depth_[drafted];
}

/* ------------------------------------------------------------------ the step's cost */
int DraftController::cost_class(int n_rows) {
    int c = 0;
    while (c + 1 < kCostClasses && n_rows >= (2 << c)) ++c;
    return c;
}

void DraftController::step_time(int depth, int n_rows, double ms) {
    if (!adapt_ || depth < 1 || depth > p_.max_depth || n_rows < 1 || !(ms > 0.0) ||
        !std::isfinite(ms))
        return;
    const int c = cost_class(n_rows);
    double&  m = cost_ms_[depth][c];
    int64_t& n = cost_n_[depth][c];
    if (n >= kCostClipAfter && ms > kCostOutlier * m) ms = kCostOutlier * m;
    const double w = settle_weight(kCostHalfLife, n);
    m = n ? m + w * (ms - m) : ms;
    ++n;
}

int64_t DraftController::cost_samples(int depth, int n_rows) const {
    if (depth < 1 || depth > RAD_SCHED_MAX_SPEC || n_rows < 1) return 0;
    return cost_n_[depth][cost_class(n_rows)];
}

double DraftController::step_cost(int depth, int n_rows) const {
    if (depth < 1 || depth > RAD_SCHED_MAX_SPEC) return 0.0;
    const int c = n_rows >= 1 ? cost_class(n_rows) : 0;
    if (cost_n_[depth][c] > 0) return cost_ms_[depth][c];
    /* Nearest measured key: this class first (nearest depth, shallower on a tie), then the
     * nearest class. Only its ratio to the prior is borrowed, so D's comparison across depths in
     * one class follows the prior's shape until those depths are measured. */
    for (int dc = 0; dc < kCostClasses; ++dc) {
        for (int sgn = 0; sgn < 2; ++sgn) {
            const int k = sgn ? c + dc : c - dc;
            if (k < 0 || k >= kCostClasses || (dc == 0 && sgn)) continue;
            int best = -1;
            for (int d = 1; d <= p_.max_depth; ++d) {
                if (cost_n_[d][k] <= 0) continue;
                if (best < 0 || std::abs(d - depth) < std::abs(best - depth)) best = d;
            }
            if (best > 0) return cost_ms_[best][k] * cost_prior(depth) / cost_prior(best);
        }
    }
    return cost_prior(depth);
}

/* ------------------------------------------------------------------ one sequence */
int DraftController::stop_of(const float* a) const {
    double p = 1.0;
    int d = 0;
    while (d < p_.max_depth) {
        p *= (double)a[d];
        if (p < kDraftMinChance) break;
        ++d;
    }
    return d < kMinDepth ? kMinDepth : d;
}

double DraftController::prior_at(int j) const {
    if (j < 0 || j >= RAD_SCHED_MAX_SPEC) return 0.0;
    return prior_n_ >= kPriorMinSeqs ? (double)prior_[j] : kPrior;
}

void DraftController::start(DraftAdapt& s) const {
    /* From finished sequences once there are enough (adapt_retire), else the fixed kPrior. */
    for (int j = 0; j < p_.max_depth; ++j) {
        s.a[j] = (float)prior_at(j);
        s.n[j] = 0;
    }
    s.depth = stop_of(s.a);
    s.held  = 0;
}

void DraftController::adapt_retire(const DraftAdapt& s) {
    if (!adapt_ || s.depth <= 0 || (double)s.n[0] < kHalfLife) return;
    const double w = settle_weight(kPriorHalfLifeSeqs, prior_n_);
    for (int j = 0; j < p_.max_depth; ++j)
        prior_[j] = prior_n_ ? (float)((double)prior_[j] + w * ((double)s.a[j] - (double)prior_[j]))
                             : s.a[j];
    ++prior_n_;
}

double DraftController::yield(const DraftAdapt& s, int d) const {
    if (d > p_.max_depth) d = p_.max_depth;
    double e = 1.0, run = 1.0;
    for (int k = 1; k <= d; ++k) { run *= (double)s.a[k - 1]; e += run; }
    return e;
}

int DraftController::window_of(const DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) return 0;
    const int D = step_d_ > 0 ? step_d_ : p_.max_depth;
    return s.depth < D ? s.depth : D;
}

int DraftController::window_for(DraftAdapt& s) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    return window_of(s);
}

int DraftController::adapt_observe(DraftAdapt& s, int drafted, int accepted) const {
    if (!adapt_) return p_.max_depth;
    if (s.depth <= 0) start(s);
    if (drafted <= 0) return s.depth;
    const int D = p_.max_depth;
    if (drafted > D) drafted = D;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;

    /* The positions this verify evaluated, by observe()'s rule. Positions past the first
     * rejection were never evaluated and are not updated here. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        const double w = std::fmax(alpha_, 1.0 / ((double)s.n[j] + kPriorWeight + 1.0));
        const double x = j < accepted ? 1.0 : 0.0;
        s.a[j] = (float)((double)s.a[j] + w * (x - (double)s.a[j]));
        if (s.n[j] < UINT16_MAX) ++s.n[j];
    }
    /* Positions past the window drift toward its deepest one. Otherwise a sequence held shallow
     * by prose that turns into code would keep its stale deep estimates and never draft deep
     * enough to update them. Assumes acceptance decays smoothly with position; when wrong it costs
     * a move out and back. */
    const double edge = (double)s.a[drafted - 1];
    for (int j = drafted; j < D; ++j)
        s.a[j] = (float)((double)s.a[j] + alpha_ * (edge - (double)s.a[j]));

    /* The stop moves only past the hold: it is this sequence's share of the step's shape. */
    if (s.held < INT32_MAX) ++s.held;
    if (s.held < kHold) return s.depth;
    const int d = stop_of(s.a);
    if (d != s.depth) {
        s.depth = d;
        s.held  = 0;
    }
    return s.depth;
}

/* ------------------------------------------------------------------ the step */
int DraftController::plan_step(DraftAdapt* const* seq, int n_seq, int n_rows) {
    if (!adapt_ || n_seq <= 0) return step_d_;
    if (n_rows < n_seq) n_rows = n_seq;
    const int M = p_.max_depth;

    /* Expected tokens at each head depth: sum_i E_i(min(D, stop_i)), plus one per non-drafting
     * row. Candidates end at the deepest stop, since a deeper D changes nothing. */
    double tok[RAD_SCHED_MAX_SPEC + 1] = {};
    int top = kMinDepth;
    for (int i = 0; i < n_seq; ++i) {
        DraftAdapt& s = *seq[i];
        if (s.depth <= 0) start(s);
        if (s.depth > top) top = s.depth;
        double e = 1.0, run = 1.0;
        for (int d = 1; d <= M; ++d) {
            if (d <= s.depth) { run *= (double)s.a[d - 1]; e += run; }
            tok[d] += e;
        }
    }
    const double plain = (double)(n_rows - n_seq);
    if (top > M) top = M;

    int best = kMinDepth;
    double best_g = -1.0;
    for (int d = kMinDepth; d <= top; ++d) {
        const double g = (tok[d] + plain) / step_cost(d, n_rows);
        if (g > best_g) { best_g = g; best = d; }
    }

    /* Move only after kHold steps and only for kHyst more yield than the current depth. The
     * first drafting step takes the best outright. */
    if (step_d_ <= 0) {
        step_d_ = best;
        step_held_ = 0;
    } else {
        if (step_held_ < INT32_MAX) ++step_held_;
        const int cur = step_d_ < top ? step_d_ : top;
        const double cur_g = (tok[cur] + plain) / step_cost(cur, n_rows);
        if (best != cur && step_held_ >= kHold && best_g > cur_g * (1.0 + kHyst)) {
            RAD_DEBUG("draft window: head depth %d -> %d over %d drafting of %d rows (%.3f -> "
                      "%.3f tokens per unit of step cost)", step_d_, best, n_seq, n_rows, cur_g,
                      best_g);
            step_d_ = best;
            step_held_ = 0;
        }
    }
    step_w_ = step_d_ < top ? step_d_ : top;
    return step_d_;
}

}  /* namespace rad */
