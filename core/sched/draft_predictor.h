/* draft_predictor.h -- B: each request's conditional draft acceptance, position by position.
 *
 * One model per scheduler. The scheduler asks it once per cycle, before a window is chosen
 * (DraftHooks::predictor), and hands it a completed cycle's observations afterwards (observe).
 * It is the per-request half of the adaptive controller in draft.h -- the same position-wise EMA
 * and the same drift -- but with labels, censoring, origin and evidence kept apart, so the policy
 * can price a position by what the verify measured rather than by what the window assumed.
 *
 * What counts as evidence:
 *   - The label mask is the shared one, draft_label(): a label exists at position j only while
 *     j < checked and j <= accepted. `checked` ends at the first rejection, so a censored cycle
 *     keeps the rejection it actually checked -- a censor cuts the cycle earlier, it does not
 *     erase a measured 0. A cycle where the draft hit the EOG but the token was not committed has
 *     checked == accepted and therefore no false rejection. (DraftController infers
 *     checked = accepted + 1 from (drafted, accepted); a censored row can carry fewer labels, so
 *     here the contract's checked is authoritative.)
 *   - Positions past the verified window (`verified`, the controller's `drafted`) drift toward
 *     a[verified - 1], exactly as DraftController::adapt_observe does. The drift is not a sample:
 *     it does not touch n[] or ess[] and never sets a valid bit, so a prior cannot turn into
 *     calibration evidence.
 *   - Only the head's own proposals teach this model. Origin is per row and comes from the cycle's
 *     immutable plan (DraftPlan::fill): Serial is the head, Ngram is prompt lookup with another
 *     length distribution. A row of any other origin is skipped, never averaged in. An observation
 *     that names no plan has no origin and no identity, so it is refused -- null never means Head.
 *   - A K0 cycle (generated == 0, fill == None) leaves the estimate untouched: the request keeps
 *     its history and recovers at the next drafted cycle.
 *
 * Versions: predict serves only the current policy and refuses a view of another version. observe
 * accepts any cycle its plan identifies, whatever version planned it: the plan and observation
 * agree on version, cycle and row identity, so a cycle that was live across a policy bump still
 * teaches. configure() keeps a table that already holds state when only the version changed; a
 * version is provenance, not a reset.
 *
 * Evidence: a[j] is the EMA over at most the last (2 - alpha)/alpha labels, so the label count
 * stops being the evidence after the running-mean phase. DraftPrediction.n reports the effective
 * sample size (prior pseudo-labels excluded), and unc[] is computed from it: a policy can price a
 * position by how much it actually knows, not by how long the slot has lived.
 *
 * Memory: DraftPredictState is one per request slot, owned by the scheduler, and nothing here
 * allocates. A slot whose (req_id, slot_gen) differs from its state is recycled and starts at the
 * prior; an unchanged identity (preemption) keeps the estimate. Recycling a slot while a cycle
 * that names its old identity is still unobserved drops that cycle: observe() refuses it rather
 * than mixing it into the new request, so the scheduler must not reuse a slot with a live plan.
 * A request that moves to another slot needs its state copied by the scheduler; a fresh slot
 * starts over.
 *
 * Sampling: the model has no sampling dimension. Labels from a request that changes class land in
 * the same EMA; a[] is a head estimate for the row it was predicted for and must not price a
 * lookup row, and no RNG or device property is claimed here.
 */
#pragma once
#include "draft_types.h"

#include <cstdint>

namespace rad {

/* One request's estimate. Identity is the view's (req_id, slot_gen); `a[j]` is the chance position
 * j+1 is accepted given that 1..j were. `n[j]` is the raw label count that drives the EMA step
 * (the controller's schedule); `ess[j]` is the effective sample size, prior pseudo-labels
 * included, that predict() reports and unc_at() uses. */
struct DraftPredictState {
    uint64_t req_id = 0;
    uint32_t slot_gen = 0;
    float    a[RAD_SCHED_MAX_SPEC] = {};
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};
    float    ess[RAD_SCHED_MAX_SPEC] = {};
};

class DraftPredictModel {
public:
    struct Params {
        int32_t  max_depth = 0;      /* positions to estimate; 0 = no drafter loaded */
        float    prior = 0.8f;       /* draft.h's kPrior: optimistic on purpose */
        float    half_life = 16.0f; /* draft.h's kHalfLife, in labelled verifies */
        float    prior_weight = 4.0f; /* pseudo-observations behind the prior (kPriorWeight) */
        float    unc_z = 2.0f;       /* half-width in standard deviations */
        uint32_t version = 1;       /* stamped into predictions; the policy version predict()
                                     * serves. A changed version keeps the learned table. */
    };

    /* `states` is the caller's per-slot table, [cap]. A fresh table starts every slot at the
     * prior; re-configuring the same table with the same estimate-shaping parameters keeps what
     * it learned, so a policy-version bump does not erase history or drop live cycles. */
    int configure(const Params& p, DraftPredictState* states, int32_t cap);
    const Params& params() const { return p_; }

    /* DraftPredictFn: fills out->seq[] in view order, starting a recycled slot at the prior.
     * Refuses a view of another policy version. */
    int predict(const DraftBatchView& view, DraftPredictions* out) const;
    /* The trampoline an integrator puts in DraftHooks::predictor. */
    static int predict_hook(void* self, const DraftBatchView& view, DraftPredictions* out);

    /* DraftObserveFn's B half, composed by the integrator after a cycle the caller declared
     * complete. The observation must name the plan it came from (draft_plan_identifies); origin,
     * cycle and identity are read from it. Refuses an incomplete cycle, a malformed row
     * (accepted > checked > verified), a plan row set outside the plan's cap, and a slot whose
     * current generation is not the one observed. */
    int observe(const DraftObservation& obs) const;

private:
    void  start(DraftPredictState& s, uint64_t req_id, uint32_t slot_gen) const;
    void  learn(DraftPredictState& s, const DraftSeqObservation& o, DraftDrafter src) const;
    float unc_at(const DraftPredictState& s, int j) const;

    Params p_;
    DraftPredictState* states_ = nullptr;
    int32_t cap_ = 0;
    double  alpha_ = 0.0;   /* the EMA step at half_life */
};

}  /* namespace rad */
