/* draft.h -- the speculative draft-depth controller.
 *
 * IT IS STATIC. The depth is `--num-speculative-tokens` and nothing moves it.
 *
 * No dynamic policy sits here, and the reason is structural rather than a tuning failure. The
 * dynamic policy this engine is otherwise modelled on -- a confidence-product gate against a
 * decayed per-position acceptance estimate, plus a batch-size ceiling schedule -- gates on the
 * DRAFTER's own top-1 confidences, per slot, on device, after the drafter has run. The scheduler
 * decides the depth BEFORE the drafter runs and cannot see them; the only substitute available
 * here is measured acceptance, which is the right quantity in expectation but arrives one step
 * late and pooled across sequences. Against a draft head whose cost is nearly flat in depth (one
 * extra verify row on a step that is weight-bound, not compute-bound) that estimate has nothing to
 * buy: the best depth is a property of the model pair, it is known before the serve starts, and it
 * is a flag. A controller that can only pick a worse depth than the constant contributes only the
 * risk that it picks one.
 *
 * What survives is the accounting, because the acceptance rate is a reported metric (spec §14)
 * and because a verify that accepts nothing has to be visible.
 */
#pragma once
#include "rad_core.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows precisely because a verify step falls in that
 * band (spec §10), so 64 is the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

class DraftController {
public:
    struct Params {
        int max_depth = 0;        /* the deployment's window; 0 disables speculation entirely */
        bool adaptive = false;    /* dynamic depth adaptation */
    };

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* The draft depth ceiling for the next step.
     * When adaptive is false: static depth is preserved (tests and fixed profiles).
     * When adaptive is true: returns max_depth ceiling so SchedReq::suggest_depth
     * can dynamically allocate speculation depth per request via the LCB Bandit. */
    int  next_depth(int active_seqs = 1) const {
        (void)active_seqs;
        if (p_.max_depth <= 0) return 0;
        return p_.max_depth;
    }

    /* One verify's outcome: `drafted` tokens were proposed and a prefix of `accepted` of them was
     * taken. Feeds the reported metrics and adaptive depth controller. */
    void observe(int drafted, int accepted, int active_seqs = 1);

    /* Reset adaptive state (e.g. on new request arrival or prefill completion) */
    void reset_adaptive() {
        ema_accept_len_ = 2.0f;
    }

    /* Metrics (spec §14). Cumulative and exact. */
    int64_t drafted()  const { return tot_drafted_; }
    int64_t accepted() const { return tot_accepted_; }
    float   acceptance() const {
        return tot_drafted_ ? (float)((double)tot_accepted_ / (double)tot_drafted_) : 0.0f;
    }
    float   ema_accept_len() const { return ema_accept_len_; }

    /* PER-POSITION ACCEPTANCE, and it is a diagnostic rather than a policy input.
     *
     * The pooled ratio above cannot tell a drafter that decays from one that falls off a cliff,
     * and the difference is the difference between "this is what the head is worth" and "round
     * three is broken". A drafter whose accepted-tokens-per-step stops rising past some depth has
     * a hazard that goes to zero there; pooled over the whole window that reads as a uniformly
     * mediocre drafter, which is a different diagnosis and a different fix.
     *
     * `reached` is "every earlier position was accepted, so this one was evaluated"; `hits` is
     * "and it was accepted too". Their ratio is the conditional hazard. A position the draft
     * never reached is charged nothing, which is why a rejection at position 0 does not libel
     * every position behind it. */
    int64_t reached(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? reach_[j] : 0;
    }
    int64_t hits(int j) const {
        return (j >= 0 && j < RAD_SCHED_MAX_SPEC) ? hit_[j] : 0;
    }
    /* The deepest position any draft has ever reached; how much of the two arrays is meaningful. */
    int     depth_seen() const { return seen_; }

private:
    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;
    float   ema_accept_len_ = 2.0f;
};

}  /* namespace rad */
