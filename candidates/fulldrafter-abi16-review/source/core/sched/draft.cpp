/* draft.cpp -- the static draft-depth policy, and the opt-in measured one.
 *
 * See draft.h for why the default stays a constant and what the bandit is allowed to decide.
 */
#include "draft.h"

#include "rad_internal.h"

#include <climits>

namespace rad {

/* ------------------------------------------------------------------ constants */
namespace {

/* THE COLD SWEEP gives every arm this many COMPLETED INTERVALS before any of them is ranked, and
 * the count is of intervals rather than of credited samples on purpose: a workload of short
 * generations, a context bound or a clamped window can make every step uncreditable, and holding
 * the sweep on credit alone would then leave the controller exploring forever. An interval that
 * could not be credited still spent real time on the arm. */
constexpr int32_t kColdAttempts = 3;

/* HOW LONG A CHOSEN ARM IS HELD before it is ranked against the others, also in intervals. The
 * step-to-step noise of one batch's tokens over nanoseconds is large -- a single step is one host
 * interval and one acceptance draw -- so a ranking taken every step would chase it. */
constexpr int32_t kDwellIntervals = 8;

/* ONE BOUNDED PROBE every this many RANKINGS: the second-best measured arm is sampled so a change
 * in the workload can be noticed. The clock is the ranking and not the interval count, because
 * intervals advance during the cold sweep too and a modulo on them would land on a value the sweep
 * had already passed and never fire again. A ranking is one dwell, so four of them is a probe
 * about every 32 intervals in steady state. */
constexpr int32_t kProbeEvery = 4;

/* HOW MANY INTERVALS A PROBE IS HELD, and two is the minimum that can be measured: the interval
 * that follows a change verifies the PREVIOUS arm's publication while paying for the new one's, so
 * only the second interval has both halves at the probed depth. */
constexpr int32_t kProbeIntervals = 2;

/* THE REWARD'S EMA WEIGHT. One sample moves an arm by a fifth: fast enough to follow a workload
 * change within a dwell, slow enough that one lucky step does not become the policy. */
constexpr double kRewardAlpha = 0.2;

}  /* namespace */

int DraftController::configure(const Params& p) {
    if (p.max_depth < 0 || p.max_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("draft depth %d is outside 0..%d -- the verify kernel band is %d query rows",
                p.max_depth, RAD_SCHED_MAX_SPEC, RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    if (p.drafter_depth < 0 || p.drafter_depth > RAD_SCHED_MAX_SPEC) {
        RAD_ERR("the declared drafter depth %d is outside 0..%d", p.drafter_depth,
                RAD_SCHED_MAX_SPEC);
        return RAD_E_INVAL;
    }
    /* THE BANDIT'S ADMISSION TEST, and every refusal names the reason: a controller that reported
     * itself active while another path chose the depth would publish a policy nobody is running.
     *
     * The two-stage wide branch is refused rather than carried over. `next_depth` returns 8 for a
     * window of six or more on the unarmed path because the second stage of that branch is gated
     * on device state after the drafter has run; an armed controller cannot attribute such a step
     * to an arm, so the two cannot share a deployment. */
    if (p.bandit) {
        if (p.max_depth <= 0) {
            RAD_ERR("RADIANCE_ADAPTIVE_BANDIT: speculation is off (window 0), so there is no depth "
                    "to choose");
            return RAD_E_UNSUPPORTED;
        }
        if (p.adaptive) {
            RAD_ERR("RADIANCE_ADAPTIVE_BANDIT and RADIANCE_ADAPTIVE_SPEC (or --adaptive-spec) both "
                    "claim the draft depth; use one");
            return RAD_E_UNSUPPORTED;
        }
        if (p.tree_state) {
            RAD_ERR("RADIANCE_ADAPTIVE_BANDIT requires linear verification; this program declares "
                    "a tree verifier");
            return RAD_E_UNSUPPORTED;
        }
        if (p.max_depth >= 6) {
            RAD_ERR("RADIANCE_ADAPTIVE_BANDIT requires a window below 6; at %d the two-stage wide "
                    "branch chooses its depth from device state and no step could be attributed "
                    "to an arm", p.max_depth);
            return RAD_E_UNSUPPORTED;
        }
    }

    p_ = p;
    tot_drafted_ = tot_accepted_ = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) { reach_[j] = 0; hit_[j] = 0; }
    seen_ = 0;
    ema_accept_len_ = 2.0f;

    b_ = Bandit{};
    if (!p.bandit) return RAD_OK;
    /* ARMS 1..min(window, the drafter's own depth, RAD_BANDIT_MAX_ARMS). A depth the head cannot
     * propose, or deeper than the ranking can tell apart, is not an arm; and no arm is ever past
     * the configured window. */
    int arms = std::min(p.max_depth, (int)RAD_BANDIT_MAX_ARMS);
    if (p.drafter_depth > 0) arms = std::min(arms, p.drafter_depth);
    if (arms <= 0) {
        RAD_ERR("RADIANCE_ADAPTIVE_BANDIT: the drafter declared depth 0, so no arm can be executed");
        return RAD_E_UNSUPPORTED;
    }
    b_.active = true;
    b_.arms = arms;
    b_.depth = 1;                 /* the cheapest arm while the cold sweep is under way */
    b_.cold_arm = 0;
    b_.cold_left = kColdAttempts;
    /* THE FIRST ARM IS HELD FROM THE START, so the very first step is already an arm's step. */
    b_.epoch = 1;
    RAD_INFO("adaptive bandit: arms 1..%d (window %d, drafter depth %d), %d cold interval(s) an arm, "
             "dwell %d, probe every %d", arms, p.max_depth, p.drafter_depth, (int)kColdAttempts,
             (int)kDwellIntervals, (int)kProbeEvery);
    return RAD_OK;
}

void DraftController::observe(int drafted, int accepted, int active_seqs) {
    if (drafted <= 0) return;
    if (drafted > RAD_SCHED_MAX_SPEC) drafted = RAD_SCHED_MAX_SPEC;
    if (accepted < 0) accepted = 0;
    if (accepted > drafted) accepted = drafted;
    tot_drafted_  += drafted;
    tot_accepted_ += accepted;

    /* Scaled EMA update based on active concurrency:
     * When active_seqs > 1, each sequence contributes alpha / active_seqs so that
     * the total batch update per engine step has effective weight alpha = 0.2 */
    const float eff_alpha = 0.2f / (float)std::max(1, active_seqs);
    ema_accept_len_ = (1.0f - eff_alpha) * ema_accept_len_ + eff_alpha * (float)accepted;

    /* Position j was EVALUATED only if every position before it was accepted, so this verify
     * produces exactly: 0..accepted-1 evaluated and accepted, and -- when the prefix was not the
     * whole draft -- position `accepted` evaluated and rejected. Nothing past that was reached. */
    const int last = accepted < drafted ? accepted : drafted - 1;
    for (int j = 0; j <= last; ++j) {
        ++reach_[j];
        if (j < accepted) ++hit_[j];
    }
    if (drafted > seen_) seen_ = drafted;
}

/* ------------------------------------------------------------------ the bandit */
int DraftController::bandit_best() const {
    int best = 1;
    double best_r = -1.0;
    for (int k = 1; k <= b_.arms; ++k) {
        const BanditArm& a = b_.a[k - 1];
        /* AN ARM NOBODY HAS CREDITED IS NOT A CANDIDATE. Its ema is 0, which is not "measured
         * bad", it is "not measured"; ranking it would let an unvisited arm win by being absent. */
        if (a.pulls <= 0) continue;
        if (a.ema > best_r) { best_r = a.ema; best = k; }
    }
    return best;
}

void DraftController::bandit_hold(int depth) {
    if (depth < 1) depth = 1;
    if (depth > b_.arms) depth = b_.arms;
    if (depth == b_.depth) return;
    b_.depth = depth;
    ++b_.epoch;   /* the step in flight, if any, is now a transition and not evidence */
}

bool DraftController::is_attempt(const DraftController::BanditStep& s, int epoch) const {
    /* `step_depth > 0 || out_depth > 0` is what makes the word "ran" true: an interval where
     * neither the verify nor the draft passes carried a single proposal spent no arm's cost, so
     * counting it would let a workload with no drafts at all walk the cold sweep and then report
     * an arm nothing ever executed. */
    return s.step_epoch == epoch && s.step_chosen > 0 && s.emitted > 0 &&
           (s.step_depth > 0 || s.out_depth > 0);
}

bool DraftController::is_credit(const DraftController::BanditStep& s, int epoch) const {
    return is_attempt(s, epoch) && s.step_depth == s.step_chosen &&
           s.out_depth == s.step_chosen && s.prefill_tok == 0;
}

void DraftController::observe_step(const DraftController::BanditStep& s) {
    if (!b_.active) return;
    const bool attempted = is_attempt(s, b_.epoch);
    const bool credit = is_credit(s, b_.epoch);
    /* A PROBE IS BOUNDED AND MEASURABLE, and it expires on the INTERVAL, not on the attempt: an
     * interval the probe arm could not even be measured on still spent the probe, so counting only
     * attempts would leave the policy on a probed arm for as long as the workload never drafts.
     * Two intervals, never one: the interval that follows a change verifies the PREVIOUS arm's
     * publication while paying for the new one's, so only the second has both halves at the probed
     * depth (see kProbeIntervals). */
    if (b_.probe_left > 0 && --b_.probe_left == 0) {
        b_.dwell = 0;
        bandit_hold(bandit_best());
    }
    if (attempted) {
        ++b_.attempts;
        ++b_.dwell;
        if (b_.cold_left > 0) {
            /* THE COLD SWEEP: every arm gets its intervals, in order, before anything is ranked.
             * Bounded by INTERVALS and not by credit -- see kColdAttempts -- so a workload that can
             * never be measured still reaches the ranking. */
            if (--b_.cold_left == 0) {
                ++b_.cold_arm;
                if (b_.cold_arm < b_.arms) {
                    b_.cold_left = kColdAttempts;
                    bandit_hold(b_.cold_arm + 1);
                } else {
                    b_.dwell = 0;               /* swept: from here the best measured arm holds */
                    bandit_hold(bandit_best());
                }
            }
        }
    }

    if (credit && s.rows > 0 && s.ns > 0) {
        /* THE REWARD: real emitted decode tokens per decode row, over the real interval the step
         * took. Emitted tokens and not accepted drafts, so a step that verifies more rows for
         * nothing is charged for them; per row, so a batch that carried two sequences is not
         * credited twice for one arm. */
        const double reward = (double)s.emitted / ((double)s.rows * (double)s.ns);
        BanditArm& a = b_.a[s.step_depth - 1];
        a.ema = a.pulls > 0 ? (1.0 - kRewardAlpha) * a.ema + kRewardAlpha * reward : reward;
        if (a.pulls < INT32_MAX) ++a.pulls;
        ++b_.credited;
        b_.tokens += s.emitted;
        b_.ns += s.ns;
    }

    if (b_.cold_left > 0 || b_.probe_left > 0) return;

    /* THE RANKING, at a dwell boundary only: hold an arm long enough for its own evidence to be
     * about the arm and not about one interval. */
    if (b_.dwell >= kDwellIntervals) {
        b_.dwell = 0;
        ++b_.dwells;
        /* A bounded probe: every kProbeEvery rankings, sample the second-best measured arm once, so
         * a change in the workload is noticed within one dwell instead of never. The clock is the
         * RANKING and not `attempts % k`: attempts advance during the cold sweep too, and a modulo
         * on them would land on a value the sweep had already passed and never fire again. */
        if ((b_.dwells % kProbeEvery) == 0 && b_.arms > 1) {
            int second = 0;
            double second_r = -1.0;
            for (int k = 1; k <= b_.arms; ++k) {
                const BanditArm& a = b_.a[k - 1];
                if (k == b_.depth || a.pulls <= 0) continue;
                if (a.ema > second_r) { second_r = a.ema; second = k; }
            }
            if (second > 0) {
                b_.probe_left = kProbeIntervals;
                bandit_hold(second);
                return;
            }
        }
        bandit_hold(bandit_best());
    }
}

}  /* namespace rad */
