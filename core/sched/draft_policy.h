/* draft_policy.h -- the cost-based draft-depth policy (strand C).
 *
 * `DraftController` stays the entry point and owns one `DraftPlanner` plus the plan's storage, so
 * the scheduler reads one frozen plan per cycle. This is the `choose()` hook of the P00 seam
 * (draft_types.h): the predictor says what a draft position is worth, the cost model says what
 * the shape it produces costs, and this file picks each request's K.
 *
 * Welle 1 is deliberately ragged: one cycle verifies every row at its own K (which
 * StepEntry::n_spec already carries), and E2's shrinking rounds are not wired yet. `verify_k` in
 * the plan is the previous cycle's `max_k`; `max_k` is this cycle's.
 *
 * The search is a heuristic, not an optimum: a few homogeneous baselines, then at most
 * `passes` deterministic coordinate passes over the requests, each change judged by the new
 * total batch cost. Never the 7^B enumeration, which is only used by a test on a small batch.
 *
 * Budgets: a `token_budget` or `kv_budget` that is <= 0 is not a constraint (the view's
 * default); a positive one caps the summed draft tokens of the plan. When it does, every K is
 * scaled down proportionally and marked `DraftFallback::Resources` -- the plan never silently
 * spends more than it was given.
 *
 * No allocation: every candidate is written into the caller's `out->k`, and every evaluation is
 * one call to the cost model's `estimate`.
 */
#pragma once
#include "draft_types.h"

namespace rad {

/* --draft-policy. `Cost` needs a predictor and a cost model; without either it is refused, not
 * silently mapped to `legacy` (docs/ADAPTIVE-MTP.md). */
enum class DraftPolicyMode : uint8_t { Legacy = 0, Cost = 1 };

enum { RAD_DRAFT_MAX_TIERS = 8 };
/* The plan's version, stamped into every plan so a reader can tell which policy chose it. The
 * integrator's DraftHooks::version wins when it is wired; this is the default. */
enum { RAD_DRAFT_POLICY_VERSION = 1 };

struct DraftPolicyConfig {
    DraftPolicyMode mode = DraftPolicyMode::Legacy;
    int32_t k_ceiling = 0;    /* --num-speculative-tokens; 0 = no drafter process */
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
    /* The plan's rows and validity; the failure is the caller's to act on. */
    int    finish(const DraftBatchView& view, int32_t max_k, DraftPlanSource src,
                  DraftFallback fb, DraftPlan* out);

    DraftPolicyConfig c_{};
    int32_t last_k_ = 0;     /* the previous cycle's depth; 0 until one was chosen */
    int32_t held_   = 0;     /* cycles since it last moved */
    bool    chosen_ = false;
};

}  /* namespace rad */
