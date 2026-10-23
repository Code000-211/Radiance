/* draft_policy.h -- the cost-based draft-depth policy (strand C).
 *
 * `DraftController` stays the entry point and owns one `DraftPlanner` plus the plan's storage, so
 * the scheduler reads one frozen plan per cycle. This is the `choose()` hook of the P00 seam
 * (draft_types.h): the predictor says what a draft position is worth, the cost model says what
 * the shape it produces costs, and this file picks each request's K.
 *
 * Welle 1 is deliberately ragged: one cycle verifies every row at its own K (which
 * StepEntry::n_spec already carries). The scheduler bounds the serial rounds to the rows that
 * still draft (Scheduler::serial_begin); the engine's rounds loop is not yet wired to that bound.
 * `verify_k` in the plan is the window the frozen previous plan issued; `max_k` is this cycle's.
 *
 * The search is a heuristic, not an optimum: a few homogeneous baselines, then at most
 * `passes` deterministic coordinate passes over the requests, each change judged by the new
 * total batch cost. Never the 7^B enumeration, which is only used by a test on a small batch.
 *
 * Budgets: a `token_budget` or `kv_budget` that is <= 0 is not a constraint (the view's
 * default); a positive one caps the summed draft tokens of the plan. When it does, every K is
 * scaled down proportionally and marked `DraftFallback::Resources` -- the plan never silently
 * spends more than it was given. A reduced plan that cannot itself be priced is issued as a
 * `Resources` fallback, not as a `Cost` plan: the price of the uncut plan is not evidence for a
 * distribution it does not describe, and a raise is refused rather than argued from it.
 *
 * Two windows, not one. The plan's `k[]`/`max_k`/`rows[]` are the cycle's draft window; the pass
 * the paired step verifies is the previous, frozen plan's distribution, and under issue_ahead()
 * the two are different. The planner keeps that frozen distribution keyed by
 * (req_id, slot, slot_gen) -- a recycled slot with another generation is another request -- and
 * stamps the plan's `verify_k` from the rows that are still present. A `DraftVerifyShape` can
 * carry one distribution, so the search still prices every candidate on the shape it verifies;
 * when the two windows differ the draft half of that shape is not in its own estimate, and a
 * depth change under issue_ahead() is priced on the verify window until the shared contract
 * carries both (see the cost request in the issue).
 *
 * The priced shape is written down, not left at its defaults. `history_tokens` is the history the
 * paired step owes per row (one row for the verified token plus one per draft it verifies, the
 * serial head's rule), `prefill_tokens` comes from the view, `drafter` names what is loaded. The
 * replay dimension comes from the view, which carries what the deployment's ranks proved
 * (draft_replay_evidence): a runtime that cannot record runs every pass live and unkeyed, so the
 * shape may say `None`. When the view has no such evidence (`Unknown`) the shape marks it
 * (kDraftShapeReplayUnknown) instead of claiming `None`: no model may price it, and the search
 * falls back to the named baseline. The plan carries the same per-row numbers
 * (`ctx_class`, `history_tokens`) and who may spend each `k[]` (`fill`): a K0 plan with history is
 * "policy K0 with history", and the n-gram fallback is a drafter too, barred where `k[]` is 0.
 *
 * No allocation per candidate: every candidate is written into the caller's `out->k`, and every
 * evaluation is one call to the cost model's `estimate`. The frozen window is reserved at
 * configure(), or grown once on the first cycle when the caller did not size it.
 */
#pragma once
#include "draft_types.h"

#include <cstddef>
#include <vector>

namespace rad {

/* --draft-policy. `Cost` needs a predictor and a cost model; without either it is refused, not
 * silently mapped to `legacy` (docs/ADAPTIVE-MTP.md). */
enum class DraftPolicyMode : uint8_t { Legacy = 0, Cost = 1 };

enum { RAD_DRAFT_MAX_TIERS = kDraftMaxTiers };
/* The plan's version, stamped into every plan so a reader can tell which policy chose it. The
 * integrator's DraftHooks::version wins when it is wired; this is the default. */
enum { RAD_DRAFT_POLICY_VERSION = 1 };

struct DraftPolicyConfig {
    DraftPolicyMode mode = DraftPolicyMode::Legacy;
    int32_t k_ceiling = 0;    /* --num-speculative-tokens; 0 = no drafter process */
    int32_t max_seqs = 0;     /* cfg.max_seqs: the frozen window's storage; 0 = grow once */
    int32_t legacy_k = 0;     /* what a legacy plan drafts; 0 = the ceiling */
    int32_t fallback_k = 0;   /* the known fixed baseline when no shape can be priced */
    /* --draft-depth-tiers, ascending and unique, 0 first when K0 is a legal choice. */
    int32_t tiers[RAD_DRAFT_MAX_TIERS] = { 0, 2, 3, 4, 6 };
    int32_t n_tiers = 5;
    /* A raise needs this much more yield and this many cycles since the last move. Reductions
     * happen at once. Both are starting values (the issue's 2%/eight steps). The hold is on the
     * plan's deepest K: a reallocation of drafts below it is judged by the model's price for the
     * whole shape, and a per-request hold would need identity-keyed state in the policy. */
    double  hyst = 0.02;
    int32_t hold = 8;
    /* Deterministic coordinate passes over the requests, and the whole cycle's estimate call
     * budget. The budget bounds a large batch's per-cycle host time. */
    int32_t passes = 2;
    int32_t max_evals = 136;
    /* --draft-calib-k: a fixed sequence of depths, one per non-empty cycle, taken in the order
     * given (duplicates allowed, unsorted) and never priced. Only a calibration run sets it
     * (Engine::begin_draft_calibration refuses it everywhere else), and configure() checks every
     * value against the effective ladder -- the trimmed, deduplicated list the serving policy
     * offers -- so no sample is ever measured at a depth the ladder never offers. */
    int32_t calib_k[RAD_DRAFT_MAX_TIERS] = {};
    int32_t n_calib_k = 0;
};

class DraftPlanner {
public:
    int  configure(const DraftPolicyConfig& c);
    const DraftPolicyConfig& config() const { return c_; }
    /* Forget the previous cycle: the next choose() is a first choice. */
    void reset();

    /* Fill `out` once and return RAD_OK, or a negative RAD_E_* (the caller then issues nothing).
     * RAD_E_UNSUPPORTED without a cost model. Allocation-free; `out->k` is the scratch. */
    int  choose(const DraftBatchView& view, const DraftPredictions& pred,
                const DraftCostModel& costs, DraftPlan* out);

    /* The seam's free-function form. */
    static int choose_hook(void* ctx, const DraftBatchView& view, const DraftPredictions& pred,
                           const DraftCostModel& costs, DraftPlan* out) {
        return static_cast<DraftPlanner*>(ctx)->choose(view, pred, costs, out);
    }

    /* The last chosen `max_k`; for tests and logs. */
    int32_t last_k() const { return last_k_; }

    /* How many of the ladder's depths the last choose() found a price for: the deepest offered
     * tier plus every shallower one the model priced, whether or not it won (0 when the deepest
     * was unpriced). This is what makes "both alternatives were compared" checkable from the
     * ledger -- a Cost plan at the deep tier alone does not show whether the shallow one lost or
     * was never priced. Reset to 0 by reset() and by a plan that is not a Cost decision. */
    int32_t last_tiers_priced() const { return last_tiers_priced_; }

    /* This choose() call was refused by the explicit-ladder coverage gap, not by the hook: the
     * cycle has no plan for a reason no later cycle can fix. Cleared at every entry. */
    bool coverage_gap_refused() const { return gap_refused_; }

private:
    /* One candidate assignment's expected tokens and price. `valid` false: nothing to compare. */
    struct Priced {
        double yield = 0.0;
        double ms = 0.0;
        double unc = 0.0;
        bool   valid = false;
    };

    /* Price the assignment in `k[]`, with every k clamped to `clamp_to` first when that is >= 0.
     * Returns RAD_OK, or the model's negative error. */
    int  price(const DraftBatchView& view, const DraftPredictions& pred,
               const DraftCostModel& costs, const int32_t* k, int32_t clamp_to, int* evals,
               Priced* out) const;
    /* E(K) for request i: 1 + the sum of prefix survival probabilities, capped by its remaining
     * output. */
    static double expect(const DraftPredictions& pred, int32_t i, int32_t cap, int32_t k);
    /* Summed drafts of the plan, and its deepest K. */
    static int32_t total_k(const int32_t* k, int32_t n, int32_t* max_k);
    /* The plan's rows and validity; the failure is the caller's to act on. Remembers the issued
     * k[] as the next cycle's verify window. */
    int    finish(const DraftBatchView& view, int32_t max_k, DraftPlanSource src,
                  DraftFallback fb, DraftPlan* out);

    /* The deepest K the frozen plan issued among the rows this view still holds; 0 when it names
     * none of them. Identity, never an index. */
    int32_t frozen_verify_k(const DraftBatchView& view) const;
    /* The frozen plan's K for the view's row i, 0 when it names none. */
    int32_t frozen_k_of(const DraftBatchView& view, int32_t i) const;
    /* The shape assignment `k[]` verifies (clamped to `clamp_to` when that is >= 0), with the
     * history the paired step owes and the drafter that is loaded. */
    void    shape_of(const DraftBatchView& view, const int32_t* k, int32_t clamp_to,
                     DraftVerifyShape* s) const;
    void    remember(const DraftBatchView& view, const int32_t* k);

    DraftPolicyConfig c_{};
    int32_t last_k_ = 0;     /* the previous cycle's depth; 0 until one was chosen */
    int32_t held_   = 0;     /* cycles since it last moved */
    bool    chosen_ = false;
    /* The calibration sequence's position. It advances only on a cycle that has rows: an empty
     * view has no depth to measure, and stepping on it would put every later depth one place off
     * the rotation. reset() and configure() start it at 0. */
    int32_t calib_pos_ = 0;
    /* Tiers the last Cost search found priced (see last_tiers_priced()). */
    int32_t last_tiers_priced_ = 0;
    /* The explicit-ladder coverage gap was diagnosed; once per priced/unpriced transition. */
    bool    gap_reported_ = false;
    /* This choose() call returned RAD_E_UNSUPPORTED from the coverage gap. */
    bool    gap_refused_ = false;
    /* The last issued plan's K per request: the window the paired step verifies. `fr_n_` counts the
     * entries in use; reset() forgets them without giving the storage back. */
    std::vector<uint64_t> fr_id_;
    std::vector<int32_t>  fr_slot_;
    std::vector<uint32_t> fr_gen_;
    std::vector<int32_t>  fr_k_;
    size_t                fr_n_ = 0;
};

}  /* namespace rad */
