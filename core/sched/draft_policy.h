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
 * `verify_k` in the plan is the window the paired step runs, per row: this cycle's own K at
 * pipeline_depth 0, the frozen previous plan's K ahead, and on the direct step() path that K
 * re-capped to the candidate's own K (Scheduler::draft_cap). `max_k` is this cycle's draft depth.
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
 * stamps the plan's `verify_k` from the rows that are still present. The view names which step path
 * opened the cycle (`DraftBatchView::direct_step`): a step issued ahead verifies the frozen window
 * as declared, while the step() path builds its step after the plan and re-caps each row to that
 * plan's own K (Scheduler::draft_cap), so only there the priced verify is the capped window. A
 * `DraftVerifyShape` can carry one distribution, so the search still prices every candidate on the
 * shape its paired step verifies; when the draft and verify windows differ the draft half of that
 * shape is not in its own estimate, and a depth change under issue_ahead() is priced on the verify
 * window until the shared contract carries both (see the cost request in the issue).
 *
 * Execution price and horizon score are two numbers. The execution price is the shape above --
 * candidate drafts, frozen verify -- and is what the plan reports as its own (last_price().ms). The
 * candidate's yield E(K) is not earned in that pass: it is verified by the next one. Ranking
 * E(K) against the frozen window's cost let an inferior depth hold itself (each cycle's price
 * carried the window it froze), so candidates are ranked by a horizon score: E(K) over the price of
 * the stationary shape that verifies K -- the same shape with every row's verify window set to its
 * own k (one cycle ahead, assuming the next cycle keeps K; prefill and the other dimensions as
 * this cycle's). At pipeline_depth 0, and wherever the candidate equals the frozen window, the two
 * shapes are one and one estimate is asked. A candidate is scored only when both shapes have their
 * own exact price: a missing stationary price is never stood in for by the execution price (nor
 * the other way round), whether the verify window steps up or down, so such a candidate is not
 * scored -- and as the deepest gate that is the named baseline. No price is invented for a shape
 * the step will not run, nor for one the ranking divides by.
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
 * evaluation is one call to the cost model's `estimate`. A candidate whose horizon shape differs
 * from its execution shape needs two of `max_evals` and is not priced at all when fewer remain.
 * The search past the deepest gate leaves three such prices of `max_evals` to the calls after it
 * (the cut's re-price, the incumbent, the issued plan's own price); see `max_evals`. The frozen
 * window is reserved at configure(), or grown once on the first cycle when the caller did not
 * size it.
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
    /* The switch rule's two inputs, not its justification. A depth change is taken when the move's
     * own rate over `hold` cycles (at least two: the move's own cycle plus the first that earns the
     * new depth's yield) -- the dwell the new depth is assumed to run -- beats staying:
     * the first of those cycles is priced, not assumed free (it runs the candidate's shape while the
     * tokens it commits are the frozen window's), the rest run at the candidate's own price and
     * yield. `hyst` is the return that net rate must show over the rate of not moving. Neither is a
     * threshold on its own: no fixed percentage of the incumbent's score and no separate hold
     * counter. Both are starting values (the issue's 2% and eight cycles). The incumbent depth is
     * the deepest K the policy last chose, not one a row's own cap forced: a reallocation below it
     * is judged by the model's price for the whole shape, and a per-request dwell would need
     * identity-keyed state. */
    double  hyst = 0.02;
    int32_t hold = 8;
    /* --draft-hysteresis both: a reduction is judged by the same priced switch rule, against the
     * incumbent's own assignment (every row at min(incumbent, its cap)), as the adaptive window
     * does both ways. A reduction that pays for itself is still taken at once -- a shallow window
     * whose throughput covers the yield it gives up saves more than one transition cycle costs. Not
     * held up either way: a reduction the rows' own caps force, the resource cut, and a reduction
     * from an incumbent with no price of its own. Off: reductions happen at once. */
    bool    hold_reductions = false;
    /* Deterministic coordinate passes over the requests, and the whole cycle's estimate call
     * budget. The budget bounds a large batch's per-cycle host time. The deepest gate may spend all
     * of it; the other alternatives stop 6 calls short (3 at pipeline_depth 0), kept for the
     * cut's re-price, the incumbent and the issued price. Below 2 + 6 a pipelined cycle compares
     * no alternative to the deepest tier; below 2 (or 1 at depth 0) it prices nothing and takes
     * the named baseline. A call the budget cannot cover is unpriced: the cut plan is a Resources
     * fallback, a raise is refused, last_price() is invalid. */
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

/* One depth's price as the last Cost search actually obtained it, for the diagnostic trace. A
 * depth the model could not price keeps `valid` false and its numbers meaningless: "the shallow
 * alternative lost" and "it was never measured" are different facts, and the ledger's count alone
 * cannot tell them apart. `k` names the offered tier, not the per-row clamped depth the batch ran. */
struct DraftTierPrice {
    int32_t k         = 0;
    bool    valid     = false;
    double  ms        = 0.0;   /* the execution price: the shape this cycle's step runs */
    double  unc_ms    = 0.0;
    double  horizon_ms = 0.0;  /* the cost the ranking divided E(K) by (see the header) */
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

    /* The hysteresis incumbent: the last chosen `max_k`, kept across a cycle that only the rows'
     * own caps reduced. For tests and logs; the issued depth is the plan's `max_k`. */
    int32_t last_k() const { return last_k_; }

    /* How many of the ladder's depths the last choose() found a price for: the deepest offered
     * tier plus every shallower one the model priced, whether or not it won (0 when the deepest
     * was unpriced). This is what makes "both alternatives were compared" checkable from the
     * ledger -- a Cost plan at the deep tier alone does not show whether the shallow one lost or
     * was never priced. A search metric, not an output one: a resource cut or the hysteresis can
     * change the issued depth afterwards without changing how many depths the search weighed, so
     * read the issued depth from the plan's `max_k` and its own price from last_price(). Reset to 0 by
     * reset() and by a plan that is not a Cost decision. */
    int32_t last_tiers_priced() const { return last_tiers_priced_; }

    /* The prices the last choose() actually asked for, in ladder order: validity and the estimate
     * for the homogeneous plan at that depth, whether or not it won. A depth enters the list only
     * once the search asked about it -- the whole ladder when the deepest tier priced, the deepest
     * gate alone when it did not (the shallower alternatives are then not compared, and a trace
     * that listed them as invalid would claim they were examined). A calibration's forced plan
     * prices nothing, so the list is empty and `last_price_known()` is false -- never an invalid
     * price, which would read as "the search ran and found none". */
    const DraftTierPrice* last_tier_prices() const { return last_tier_price_; }
    int32_t last_n_tier_prices() const { return last_n_tier_price_; }
    /* A Cost search ran this cycle (whether or not it produced a price). */
    bool    last_price_known() const { return last_price_known_; }
    /* The estimate of the plan that was issued -- after the budget cut and the hysteresis, the
     * assignment the step will really run -- with `valid` false when it could not be priced. */
    const DraftTierPrice& last_price() const { return last_price_; }

    /* This choose() call was refused by the explicit-ladder coverage gap, not by the hook: the
     * cycle has no plan for a reason no later cycle can fix. Cleared at every entry. */
    bool coverage_gap_refused() const { return gap_refused_; }

    /* One row's verify window under a cycle that got no plan, so the next plan declares and prices
     * the step that runs instead of the plan before the failure. Called where that window is made:
     * ahead, after the build, with each row's own n_spec -- a step issued under a refused cycle
     * verifies that many rounds even for a row the head publishes nothing for (a state today's
     * sampler contract excludes; Scheduler::draft_freeze_ahead); direct, with the count
     * set_draft() accepted, 0 included (Scheduler::draft_freeze_published). Not a plan and not a
     * measurement -- no price is issued, no depth or sequence position moves, and no ring record
     * opens. */
    void remember_published(uint64_t req_id, int32_t slot, uint32_t slot_gen, int32_t k);

private:
    /* One candidate assignment's expected tokens, execution price (`ms`, `unc`) and horizon cost
     * (`hms`); the score is yield / hms. `valid` false: nothing to compare. */
    struct Priced {
        double yield = 0.0;
        double ms = 0.0;
        double unc = 0.0;
        double hms = 0.0;
        bool   valid = false;
    };

    /* Price the assignment in `k[]`, with every k clamped to `clamp_to` first when that is >= 0:
     * its stationary horizon shape (when it differs), then its execution shape. Both must fit
     * under `limit` evaluations or neither is asked. Returns RAD_OK, or the model's negative error. */
    int  price(const DraftBatchView& view, const DraftPredictions& pred,
               const DraftCostModel& costs, const int32_t* k, int32_t clamp_to, int limit,
               int* evals, Priced* out) const;
    /* E(K) for request i: 1 + the sum of prefix survival probabilities, capped by its remaining
     * output. */
    static double expect(const DraftPredictions& pred, int32_t i, int32_t cap, int32_t k);
    /* The priced switch rule: whether the yield `cand` gains over `hold` cycles pays for the one
     * cycle the move spends (both `Priced`s are valid and price per-cycle stationary shapes). */
    bool   switch_pays(const Priced& cand, const Priced& inc) const;
    /* Summed drafts of the plan, and its deepest K. */
    static int32_t total_k(const int32_t* k, int32_t n, int32_t* max_k);
    /* The plan's rows and validity; the failure is the caller's to act on. Remembers the issued
     * k[] as the next cycle's verify window. */
    int    finish(const DraftBatchView& view, int32_t max_k, DraftPlanSource src,
                  DraftFallback fb, DraftPlan* out);

    /* The verify window row `i` runs at this plan's own depth `ki`: its own K at pipeline_depth 0,
     * the frozen plan's K ahead, and that K re-capped to `ki` on the direct step() path, whose
     * paired step is built after the plan (Scheduler::draft_cap). */
    int32_t verify_k_of(const DraftBatchView& view, int32_t i, int32_t ki) const;
    /* The frozen plan's K for the view's row i, 0 when it names none. */
    int32_t frozen_k_of(const DraftBatchView& view, int32_t i) const;
    /* The shape assignment `k[]` verifies (clamped to `clamp_to` when that is >= 0), with the
     * history the paired step owes and the drafter that is loaded. A row's verify is
     * verify_k_of(): the frozen plan's window, re-capped to this candidate's K on the direct
     * path only. `horizon` instead verifies every row at its own k: the stationary shape. */
    void    shape_of(const DraftBatchView& view, const int32_t* k, int32_t clamp_to,
                     DraftVerifyShape* s, bool horizon = false) const;
    void    remember(const DraftBatchView& view, const int32_t* k);
    /* This view's rows' identity in the frozen storage, growing it only when configure() did not
     * size it; every caller then fills fr_k_ itself. */
    void    fit_frozen(const DraftBatchView& view);

    DraftPolicyConfig c_{};
    int32_t last_k_ = 0;     /* the switch's incumbent; 0 until one was chosen */
    bool    chosen_ = false;
    /* The calibration sequence's position. It advances only on a cycle that has rows: an empty
     * view has no depth to measure, and stepping on it would put every later depth one place off
     * the rotation. reset() and configure() start it at 0. */
    int32_t calib_pos_ = 0;
    /* Tiers the last Cost search found priced (see last_tiers_priced()). */
    int32_t last_tiers_priced_ = 0;
    /* The last search's per-depth prices and the issued plan's own (see last_tier_prices()). */
    DraftTierPrice last_tier_price_[RAD_DRAFT_MAX_TIERS] = {};
    int32_t        last_n_tier_price_ = 0;
    bool           last_price_known_ = false;
    DraftTierPrice last_price_;
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
    /* The incumbent's own assignment a held reduction is priced against (hold_reductions);
     * reserved with the frozen window, grown on first use otherwise. */
    std::vector<int32_t>  held_k_;
};

}  /* namespace rad */
