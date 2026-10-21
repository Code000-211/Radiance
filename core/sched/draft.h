/* draft.h -- the speculative draft-depth controller.
 *
 * Static by default: the depth is `--num-speculative-tokens`. A dynamic policy that gates on the
 * drafter's own top-1 confidences cannot work here, because the scheduler picks the depth before
 * the drafter runs; measured acceptance arrives a step late and pooled across sequences. With a
 * draft head whose cost is nearly flat in depth (one extra verify row on a weight-bound step), the
 * best depth is a property of the model pair and is known up front, so it is a flag. The
 * controller still keeps acceptance accounting, a reported metric (spec §14).
 *
 * `--draft-window adaptive` is for when that premise fails: with streamed weights (a MoE reading
 * experts per verify row, a serial head running one pass per round) each draft costs a head round
 * and a verify row, and the best depth depends on the text (deep on code and JSON, shallow on
 * prose). The adaptive policy has two halves:
 *
 *   Per sequence (DraftAdapt): a smoothed estimate of each draft position's conditional
 *   acceptance, fed by the verify outcomes, and the sequence's stop -- the deepest draft whose
 *   chance of being emitted still pays for its verify row (kDraftMinChance).
 *
 *   Per step (plan_step): the head depth D. A serial head runs as many rounds as the deepest
 *   window in the step, so D maximises sum_i E_i(min(D, stop_i)) / T(D, n), with
 *   E(d) = 1 + sum_k prod_{j<=k} a_j and T measured online (step_time). Each sequence then
 *   drafts min(D, stop_i).
 *
 * Both move only past a hold, and D also past a hysteresis, because a new window is a new pass
 * shape that must be recorded before it plays (runtime/ctx.h). `max_depth` stays the ceiling and
 * everything is sized for it.
 *
 * The policy is for serial drafters only (Scheduler::init): a block drafter proposes its block in
 * one pass and prompt lookup runs none, so neither pays per-round cost.
 *
 * It changes only how many drafts a step verifies. Acceptance follows core/sample/accept.h, so
 * every emitted token is the target's own; at temperature 0 the text matches a plain run as far
 * as the logits do, and a verify row's logits are computed at an M that moves with the window
 * (see accept.h).
 */
#pragma once
#include "draft_policy.h"
#include "draft_types.h"

#include <vector>

namespace rad {

/* One sequence's half of the adaptive policy: the drafter's record on this sequence's text and the
 * resulting stop. Stored in the scheduler's per-request state; the controller does the arithmetic.
 *
 * `depth` 0 means not started: the first ask fills the estimate from the prior and sets the stop.
 * It belongs to the request, not its cache: preemption keeps it, a recycled slot resets it. */
struct DraftAdapt {
    int32_t  depth = 0;                       /* the sequence's stop; 0 = not started */
    int32_t  held  = 0;                       /* verifies since `depth` last moved */
    float    a[RAD_SCHED_MAX_SPEC] = {};      /* conditional acceptance at each draft position */
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};      /* verifies that evaluated each one, saturating */
};

class DraftController {
public:
    struct Params {
        int  max_depth = 0;       /* the deployment's window; 0 disables speculation entirely */
        /* The adaptive policy: `--draft-window adaptive` with a serial drafter, decided by
         * Scheduler::init. Off, the depth is max_depth. */
        bool adapt = false;
        /* The cost policy (--draft-policy cost, Welle 2). Off, nothing here changes. */
        DraftPolicyConfig policy;
        /* Requests a plan's caller-provided arrays hold: cfg.max_seqs. */
        int max_seqs = 0;
        /* A/B's hooks. The cost policy refuses to run without both (RAD_E_UNSUPPORTED): the
         * choice is their output, and neither a prediction nor a price is invented here. */
        DraftPredictor  predictor;
        DraftCostModel  costs;
    };

    /* ---------------------------------------------------------- the policy's constants
     * Not configurable: each is a property of the policy, not of a deployment. The one
     * deployment property, step cost, is measured (step_time). */

    /* A window moves only for 2% more yield. One draft more or less moves E/T by several percent,
     * while a smoothed estimate wobbles about 1% per verify; a lower threshold would flip D
     * between near-equal neighbours and pay a recording each time. */
    static constexpr double kHyst = 0.02;
    /* And not within 8 verifies (steps, for D) of the last move. A pass shape is recorded on its
     * second sighting and played from the third (runtime/ctx.h), so a shape held 8 steps mostly
     * plays from the recording; 8 verifies at kHalfLife move an estimate about a third of the way
     * to a new rate, a trend rather than one rejection. */
    static constexpr int kHold = 8;
    /* Each position's estimate has a half-life of 16 verifies that evaluate it (about 50 tokens of
     * text): short enough to follow a reply from prose into code and back, long enough that one
     * rejection moves an estimate by about 4%. */
    static constexpr double kHalfLife = 16.0;
    /* An unmeasured position starts at 0.8, optimistic on purpose: a window that starts deep
     * learns every position, one that starts shallow never learns the deep ones. 0.8 lies between
     * typical first-position acceptance on prose and on code, so a wrong start corrects within a
     * hold. */
    static constexpr double kPrior = 0.8;
    /* Once 4 sequences have finished, new ones start from the mean of their final estimates
     * (adapt_retire): a running mean, then a moving average with a half-life of 32 sequences, so
     * the prior follows a change of traffic within a few dozen requests. A sequence counts only
     * after kHalfLife verifies; before that its estimate is mostly the prior itself. */
    static constexpr int    kPriorMinSeqs = 4;
    static constexpr double kPriorHalfLifeSeqs = 32.0;
    /* Never fewer than one draft: at zero no position is evaluated, so the estimate could never
     * recover. */
    static constexpr int kMinDepth = 1;
    /* A sequence stops at the first draft whose chance of being emitted, prod_{j<=d} a_j, is
     * under 1 in 10. Once the head runs D rounds for the deepest neighbour, a sequence's only
     * added cost is its verify row. At one sequence the step-cost prior (1 + 0.06 d) breaks even
     * near this chance (a draft pays while p_d / E exceeds the ~6% it adds, with E around 2), and
     * a row alone costs less than a round plus a row, so this is never stricter than D's own
     * trade. A constant because a row's share cannot be separated from the head's rounds in step
     * times keyed by depth. */
    static constexpr double kDraftMinChance = 0.1;
    /* Step times are kept per running-row class: 1, 2-3, 4-7, 8-15, 16+. Cost moves with how many
     * rows share the weight reads; power-of-two classes keep every key fed. */
    static constexpr int kCostClasses = 5;

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* The draft depth for the next step; 0 means do not speculate. Constant: under the adaptive
     * policy it is the ceiling, and window_for() is the window. */
    int  next_depth() const { return p_.max_depth; }

    /* ---------------------------------------------------------- the adaptive policy */
    bool adaptive() const { return adapt_; }
    /* The window this sequence may draft into: min(D, its stop), starting its estimate on the
     * first ask; max_depth when the policy is off. Before any step chooses D, D is the ceiling. */
    int  window_for(DraftAdapt& s) const;
    /* As window_for() but never starts an estimate; 0 if not started. For reporting. */
    int  window_of(const DraftAdapt& s) const;
    /* One verify of this sequence: `drafted` proposed, a prefix of `accepted` taken. Updates the
     * estimate, and the stop once past the hold. Returns the stop. */
    int  adapt_observe(DraftAdapt& s, int drafted, int accepted) const;
    /* A finished sequence: its estimate joins the prior that new sequences start from.
     *
     * Per sequence, not per verify. Under this policy, pooled accepted(j) / reached(j) is biased:
     * only well-predicted texts get windows deep enough to measure deep positions. Here each
     * finished sequence counts once at every position, using its edge extrapolation where it never
     * reached (adapt_observe), so shallow texts pull deep positions down as much as deep ones hold
     * them up. */
    void adapt_retire(const DraftAdapt& s);
    /* The prior a sequence starting now is given at position `j`. For tests and logs. */
    double prior_at(int j) const;

    /* D for the step being opened. `seq` holds the estimates of the `n_seq` drafting sequences
     * (started here if needed), `n_rows` all decode rows (the rest emit one token and draft
     * nothing). Moves D past the hold and hysteresis; with nothing to draft, D stays. Returns D. */
    int  plan_step(DraftAdapt* const* seq, int n_seq, int n_rows);
    /* The step depth in force: 0 before the first step chose one. */
    int  step_depth() const { return step_d_; }
    /* The deepest window a step at depth D carries: D capped at the deepest stop it was chosen
     * over; the ceiling before the first choice or with the policy off. Proposals from outside
     * the head (prompt lookup) are held to it, so they add no deeper pass shape. */
    int  step_window() const { return adapt_ && step_w_ > 0 ? step_w_ : p_.max_depth; }

    /* T(depth, n): a decode step with deepest window `depth` and `n_rows` decode rows took `ms`
     * since the end of the previous step. Fed by Scheduler::step_finished; adaptive policy only. */
    void step_time(int depth, int n_rows, double ms);
    /* Expected cost of a step at `depth` with `n_rows` rows: the key's mean (ms) if it has
     * samples; else the nearest measured key scaled by the prior's ratio (1 + 0.06 d); else the
     * unitless prior itself. */
    double step_cost(int depth, int n_rows) const;
    /* Samples behind a key: for tests and logs. */
    int64_t cost_samples(int depth, int n_rows) const;

    /* E(d) for `s`'s current estimate: expected tokens emitted by a verify of d drafts, the
     * verified token included. For tests and logs. */
    double yield(const DraftAdapt& s, int d) const;
    /* Histogram of verifies by window, counted only under the adaptive policy. */
    int64_t verified_at(int d) const {
        return (d >= 0 && d <= RAD_SCHED_MAX_SPEC) ? by_depth_[d] : 0;
    }

    /* ---------------------------------------------------------- the cost policy
     *
     * One plan per cycle, written once by open_cycle() and read by every site that caps a
     * proposal. It is replaced only when the next cycle opens, so a proposal drafted and a step
     * verified inside a cycle read the same plan. A plan whose identity names another cycle is
     * not this cycle's; the scheduler falls back to next_depth() instead of acting on it. */
    bool cost_policy() const { return planner_.config().mode == DraftPolicyMode::Cost; }
    /* Fill `preds` through the predictor and choose the cycle's plan. Returns RAD_OK, or the
     * failing hook's error; on any error no plan is in force and the cycle runs legacy. */
    int  open_cycle(const DraftBatchView& view, DraftPredictions* preds);
    /* The plan in force, or null. */
    const DraftPlan* plan() const { return have_plan_ ? &plan_ : nullptr; }
    /* This cycle's K for the request, or -1 when the plan does not cover it. Identity, never an
     * index: a recycled slot with another generation is another request. */
    int  planned_k(uint64_t req_id, int32_t slot, uint32_t slot_gen) const;
    DraftPlanner&      planner() { return planner_; }
    const DraftPlanner& planner() const { return planner_; }

    /* One verify's outcome: `drafted` proposed, a prefix of `accepted` taken. Metrics only. */
    void observe(int drafted, int accepted);

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }

    /* Per-position acceptance, a diagnostic rather than a policy input. The pooled ratio cannot
     * tell a drafter that decays gradually from one that fails at a specific position.
     *
     * `reached`: every earlier position was accepted, so this one was evaluated. `hits`: and it
     * was accepted. Their ratio is the conditional acceptance. Unreached positions are not
     * charged, so a rejection at position 0 does not count against later positions. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has reached; bounds the meaningful part of both arrays. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;

    /* The adaptive policy as configure() resolved it. */
    bool    adapt_ = false;
    double  alpha_ = 0.0;                         /* the smoothing step from kHalfLife */
    int64_t by_depth_[RAD_SCHED_MAX_SPEC + 1] = {};
    int32_t step_d_ = 0;                          /* D; 0 until a step chose one */
    int32_t step_w_ = 0;                          /* min(D, the deepest stop it saw) */
    int32_t step_held_ = 0;                       /* steps since D last moved */
    /* The prior from finished sequences (adapt_retire), and how many are behind it. */
    float   prior_[RAD_SCHED_MAX_SPEC] = {};
    int64_t prior_n_ = 0;
    /* T(d, class): the mean step time of each key, and the steps behind it. */
    double  cost_ms_[RAD_SCHED_MAX_SPEC + 1][kCostClasses] = {};
    int64_t cost_n_ [RAD_SCHED_MAX_SPEC + 1][kCostClasses] = {};

    /* The cost policy's plan and its storage (reserved at configure, outside the step path). */
    DraftPlanner planner_;
    DraftPlan    plan_;
    std::vector<uint64_t>       plan_id_;
    std::vector<int32_t>        plan_slot_;
    std::vector<uint32_t>       plan_gen_;
    std::vector<int32_t>        plan_k_;
    std::vector<DraftFallback>  plan_why_;
    std::vector<int32_t>        plan_rows_;
    bool have_plan_   = false;
    bool cost_failed_ = false;   /* the open_cycle error was logged */

    /* Fill a sequence's estimate from the prior and set its first stop. */
    void    start(DraftAdapt& s) const;
    /* The deepest draft whose chance of being emitted is at least kDraftMinChance. */
    int     stop_of(const float* a) const;
    static int cost_class(int n_rows);
};

}  /* namespace rad */
