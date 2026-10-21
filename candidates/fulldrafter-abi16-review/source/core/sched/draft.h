/* draft.h -- the speculative draft-depth controller.
 *
 * Tree-state windows retain up to 64 proposal nodes for one sequence, four with concurrency.
 *
 * TWO THINGS LIVE HERE, and the second is opt-in.
 *
 *   - THE ACCOUNTING (spec §14): drafted, accepted, per-position reach and hit, the acceptance EMA.
 *     Always on, exact, and what the reported metrics divide.
 *   - RADIANCE_ADAPTIVE_BANDIT=1: a MEASURED depth choice for the linear serial path only. The arms
 *     are depths 1..min(the configured window, the drafter's declared depth, 8). A bounded cold
 *     sweep visits every arm, then the arm with the best measured emitted-tokens per unit of
 *     completed step time holds, with a bounded number of probes of a neighbour sampled less.
 *
 * WHY THE DEFAULT IS STILL A CONSTANT, and not a controller. The dynamic policy this engine is
 * otherwise modelled on -- a confidence-product gate against a decayed per-position acceptance
 * estimate, plus a batch-size ceiling schedule -- gates on the DRAFTER's own top-1 confidences, per
 * slot, on device, after the drafter has run. The scheduler decides the depth BEFORE the drafter
 * runs and cannot see them; the only substitute available here is measured acceptance, which is
 * the right quantity in expectation but arrives one step late and pooled across sequences. Against
 * a draft head whose cost is nearly flat in depth that estimate has little to buy: the best depth
 * is then close to a property of the model pair, known before the serve starts, and it is a flag.
 * A controller that can only pick a worse depth than the constant contributes only the risk.
 *
 * WHY THE BANDIT EXISTS ANYWAY. "Close to a property of the model pair" is an expectation, not a
 * measurement, and the campaign has to know which of depths 1..3 actually pays on real traffic.
 * So the measured controller ships behind a flag, and it is the ONLY thing that makes a depth
 * decision from a reward: emitted decode tokens over the step time that produced them, both
 * counted by the engine's own clock and commit path. It is not acceptance-driven, it does not
 * touch the sampler, and it never raises the window.
 *
 * WHAT IT MAY NOT DO. It is refused unless the program is the true linear serial path: no tree
 * proposal, no tree verifier, no pruned rounds, no SpecHub, no DLoop full-context drafter, no block
 * drafter, no KL mode, and no second controller (RADIANCE_ADAPTIVE_SPEC). The two-stage wide
 * branch (next_depth returning 8 for a window of six or more) is preserved untouched for the
 * unarmed path and is a REFUSAL for the armed one: that branch chooses its depth from device state
 * after the drafter has run, so a step it produced could not be attributed to an arm at all.
 * Refusing is the only honest answer; pretending to be active would credit the wrong depth.
 */
#pragma once
#include <algorithm>
#include <cstdint>

#include "../../arch/common/rad_tree_candidate.h"

namespace rad {

/* libr4d's decode kernel serves up to 64 query rows precisely because a verify step falls in that
 * band (spec §10), so 64 is the largest window any batch can carry. */
enum { RAD_SCHED_MAX_SPEC = 64 };

/* THE MOST ARMS THE BANDIT RANKS, and a policy bound rather than a hardware one: a controller that
 * cannot tell eight depths apart on the evidence one step gives it cannot tell eighty apart
 * either. It is also the depth of every fixed per-arm array below, so nothing grows at run time. */
enum { RAD_BANDIT_MAX_ARMS = 8 };

/* The root-inclusive host tree must hold a full linear draft window too. */
static_assert(RAD_SCHED_MAX_SPEC + 1 <= radiance::RAD_TREE_MAX_NODES,
              "a full linear window's accepted path must fit one staging row");

inline int tree_draft_depth(int max_depth, bool multi_seq) {
    return std::min(max_depth, multi_seq ? 4 :
        static_cast<int>(radiance::RAD_TREE_MAX_NODES) - 1);
}

class DraftController {
public:
    struct Params {
        int max_depth = 0;        /* the deployment's window; 0 disables speculation entirely */
        bool adaptive = false;    /* dynamic depth adaptation */
        bool tree_state = false;  /* parent-aware verifier; not FullDrafter's 4/8 policy */
        /* RADIANCE_ADAPTIVE_BANDIT: the measured controller described above. Off by default. */
        bool bandit = false;
        /* THE DRAFTER'S OWN DEPTH, as the program declared it: an arm deeper than the head can
         * propose is a depth no step can execute, so it is not an arm. 0 means the program
         * declared no depth and the window alone bounds the arms. */
        int  drafter_depth = 0;
    };

    /* ONE COMPLETED INTERVAL, as the scheduler observed it. Every field is a fact the engine
     * already counts; the classification below is the policy's, in one place, so a test can hold
     * it to the rule without a device.
     *
     *   step_epoch  the policy's epoch when THIS step was planned. Any change of arm bumps the
     *               epoch, so a step that straddles one cannot be evidence about either arm.
     *   step_chosen the arm the policy held then.
     *   step_depth  the depth every one of its DECODE rows verified, or 0 when the rows disagreed
     *               or none of them verified anything.
     *   out_depth   the depth of the drafts this interval published for the NEXT step, or
     *               negative when no single depth was published (none, or rows disagreed).
     *   rows        decode rows it carried;  emitted  decode tokens it committed;
     *   prefill_tok prompt tokens it computed;      ns  its real interval.
     *
     * The engine issues the next step's drafts BEFORE this one commits, so `out_depth` is what
     * makes an interval's cost attributable: the verify and the draft passes it paid for must be
     * the same arm. */
    struct BanditStep {
        int32_t step_epoch = 0;
        int32_t step_chosen = 0;
        int32_t step_depth = 0;
        int32_t out_depth = -1;
        int32_t rows = 0;
        int32_t emitted = 0;
        int32_t prefill_tok = 0;
        int64_t ns = 0;
    };

    int  configure(const Params& p);
    const Params& params() const { return p_; }

    /* Tree-state verification uses its full configured node budget, capped by
     * the tree's row limit and the concurrent row budget. Other programs keep
     * their legacy ceilings and per-request SchedReq::suggest_depth policy.
     *
     * A PURE GETTER, including when the bandit is armed: the arm is chosen once per completed step
     * (observe_step), so a metrics scrape -- which calls this -- cannot move the policy. */
    int  next_depth(int active_seqs = 1) const {
        if (p_.max_depth <= 0) return 0;
        /* The arm the policy holds. Arms never exceed the window: see configure(). */
        if (b_.active) return b_.depth;
        if (p_.tree_state) return tree_draft_depth(p_.max_depth, active_seqs > 1);
        if (p_.max_depth >= 6) return 8;
        if (!p_.adaptive) return p_.max_depth;
        if (p_.max_depth < 3) return p_.max_depth;
        if (active_seqs > 1) return std::min(p_.max_depth, 4);
        return p_.max_depth;
    }

    /* One verify's outcome: `drafted` tokens were proposed and a prefix of `accepted` of them was
     * taken. Feeds the reported metrics and the acceptance EMA. */
    void observe(int drafted, int accepted, int active_seqs = 1);

    /* Reset adaptive state (e.g. on new request arrival or prefill completion).
     *
     * THE BANDIT IS NOT RESET HERE. Admission, an idle spell and a phase boundary say nothing
     * about which depth pays; wiping the arms there would restart the cold sweep on every request
     * and leave the controller permanently exploring. Only configure() -- a new deployment -- is
     * a new controller. */
    void reset_adaptive() {
        ema_accept_len_ = 2.0f;
    }

    /* ------------------------------------------------------------------ the bandit */

    /* THE POLICY'S ONLY ADVANCE, called once per completed interval by Scheduler::step_finished
     * with the real interval. Classifies the interval, folds what it measured into its arm, and
     * chooses the next arm when the current one has had its dwell. */
    void observe_step(const BanditStep& s);

    /* HOW ONE INTERVAL IS CLASSIFIED, exposed so the rule can be tested without a device.
     *
     *   attempted  the interval ran one arm end to end: the epoch it was planned under is still
     *              the policy's, a real arm was held, and it committed decode tokens. Deliberately
     *              independent of whether the executed depth matched -- a window the context bound
     *              shortened still ran the arm's plan, and requiring cleanliness here would let a
     *              permanently clipping workload stall the cold sweep forever.
     *   credited   the interval is a clean measurement of the arm: attempted, every decode row
     *              verified exactly the chosen arm, the drafts it paid for were that arm's too, and
     *              no prompt tokens shared the step. */
    bool is_attempt(const BanditStep& s, int epoch) const;
    bool is_credit(const BanditStep& s, int epoch) const;

    bool    bandit_active()   const { return b_.active; }
    int     bandit_arms()     const { return b_.active ? b_.arms : 0; }
    int     bandit_arm()      const { return b_.active ? b_.depth : 0; }
    int     bandit_epoch()    const { return b_.epoch; }
    int64_t bandit_attempts() const { return b_.attempts; }
    int64_t bandit_credited() const { return b_.credited; }
    int64_t bandit_tokens()   const { return b_.tokens; }
    int64_t bandit_ns()       const { return b_.ns; }
    /* Emitted tokens per row per nanosecond this arm has averaged, and the samples behind it. */
    double  bandit_reward(int arm) const {
        return (arm >= 1 && arm <= b_.arms) ? b_.a[arm - 1].ema : 0.0;
    }
    int32_t bandit_pulls(int arm) const {
        return (arm >= 1 && arm <= b_.arms) ? b_.a[arm - 1].pulls : 0;
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
    /* ONE ARM'S EVIDENCE. Fixed size, no allocation, and no per-request copy of it: the arm is a
     * property of the deployment, not of a conversation. */
    struct BanditArm {
        double  ema = 0.0;     /* emitted tokens per row per nanosecond, exponentially averaged */
        int32_t pulls = 0;    /* credited samples folded into ema */
    };
    struct Bandit {
        bool    active = false;
        int32_t arms = 0;
        int32_t depth = 0;    /* the arm held right now */
        int32_t epoch = 0;    /* bumped whenever `depth` changes; a step must not straddle one */
        int32_t cold_arm = 0; /* 0-based index of the arm the cold sweep is on */
        int32_t cold_left = 0;/* attempt intervals that arm still gets */
        int32_t dwell = 0;    /* attempt intervals the current arm has had since the last ranking */
        int32_t dwells = 0;   /* rankings performed; the probe's own clock */
        int32_t probe_left = 0;
        int64_t attempts = 0;
        int64_t credited = 0;
        int64_t tokens = 0;
        int64_t ns = 0;
        BanditArm a[RAD_BANDIT_MAX_ARMS];
    };

    /* The best-measured arm, or 1 when nothing has been credited: the cheapest depth is the one to
     * fall back to, not a deeper one nobody has evidence for. Ties go to the shallower arm. */
    int  bandit_best() const;
    /* Hold `depth`; a change starts a new epoch, which is what makes the step in flight a
     * transition rather than evidence. */
    void bandit_hold(int depth);

    Params  p_;
    int64_t tot_drafted_ = 0, tot_accepted_ = 0;
    int64_t reach_[RAD_SCHED_MAX_SPEC] = {};
    int64_t hit_  [RAD_SCHED_MAX_SPEC] = {};
    int     seen_ = 0;
    float   ema_accept_len_ = 2.0f;
    Bandit  b_;
};

}  /* namespace rad */
