/* draft_predictor.h -- B: each request's conditional draft acceptance, position by position.
 *
 * One model per scheduler. The scheduler asks it once per cycle, before a window is chosen
 * (DraftHooks::predictor), and hands it a completed cycle's observations afterwards (observe).
 * It is the per-request half of the adaptive controller in draft.h -- the same position-wise EMA
 * and the same drift -- but with labels, censoring, origin and evidence kept apart, so the policy
 * can price a position by what the verify measured rather than by what the window assumed.
 *
 * What counts as evidence:
 *   - The label mask is the shared one, draft_label(): 1 on the accepted prefix, 0 only at
 *     DraftSeqEnd::Rejected, and no label at DraftSeqEnd::Stopped (the draft hit the token the
 *     target chose, but it was not committed) -- never a fabricated reject. A censor is the
 *     reason a cycle ended, not a position: a rejection measured before it keeps its 0. A row with
 *     DraftSeqEnd::Unset is refused by the contract's gate, not guessed at.
 *   - Positions past the verified window (`verified`, the controller's `drafted`) drift toward
 *     a[verified - 1], exactly as DraftController::adapt_observe does. The drift is not a sample:
 *     it does not touch n[] or ess[] and never sets a valid bit, so a prior cannot turn into
 *     calibration evidence.
 *   - Only the head's own proposals teach this model. The teaching origin is the one of the
 *     window the verify consumed (DraftSeqObservation::label_origin), not of the plan the row is
 *     named after: with pipeline_depth > 0 the labels measured the PREVIOUS cycle's publication,
 *     and a row that changed drafter between the two plans must be judged by what it actually
 *     verified. The plan's fill[] is only the permission the contract checks with
 *     draft_origin_fits_plan for the row's own drafting half. A row whose label window was of any
 *     other origin is skipped, never averaged in; an observation that names no plan is refused by
 *     the gate -- null never means Head.
 *   - A K0 cycle (label_generated == 0, label_origin None) leaves the estimate untouched: the
 *     request keeps its history and recovers at the next drafted cycle.
 *
 * Versions: predict serves only the current policy and refuses a view of another version. observe
 * accepts any cycle its plan identifies, whatever version planned it: the plan and observation
 * agree on version, cycle and row identity, so a cycle that was live across a policy bump still
 * teaches. configure() keeps a table that already holds state when only the version changed; a
 * version is provenance, not a reset.
 *
 * Evidence: a[j] is the EMA over at most the last (2 - alpha)/alpha labels, so the label count
 * stops being the evidence after the running-mean phase. DraftPrediction.n is the contract's raw
 * observation count; the effective sample size is internal and only unc[] is computed from it: a
 * policy can price a position by how much it actually knows, not by how long the slot has lived.
 *
 * Memory: DraftPredictState is one per request slot, owned by the scheduler, and nothing here
 * allocates. A slot whose (req_id, slot_gen) differs from its state is recycled and starts at the
 * prior; an unchanged identity (preemption) keeps the estimate. Lifetime belongs to the cycle
 * ring: draft_observation_deliverable() decides whether a cycle may be observed and never consults
 * the live slot. When a recycled slot owns the state an old row names, the row is skipped -- the
 * cycle is deliverable, but its measurement must not merge into the new generation. A request that
 * moves to another slot needs its state copied by the scheduler; a fresh slot starts over.
 *
 * Sampling: the model has no sampling dimension. Labels from a request that changes class land in
 * the same EMA; a[] is a head estimate for the row it was predicted for and must not price a
 * lookup row, and no RNG or device property is claimed here.
 *
 * Class visibility, on the yield side: a request's acceptance differs with what it writes -- the
 * measured 72% for json, 55% for code and 47% for prose at a similar K are the same head, the same
 * window and the same row shape. That difference is a property of the text, not of the shape, so it
 * has no place in the cost key (draft_cost.h prices a shape in milliseconds; a class-keyed row cost
 * would mix a yield into a price). It enters the plan through E(K) instead: the model keeps the
 * workload's own position-wise acceptance next to each request's, with the same EMA and the same
 * prior weight, and a slot with no labels of its own starts from it rather than from the flat
 * prior. So a cycle that opens a prose request begins at the acceptance prose was measured at, and
 * a code request at its own -- no text classifier (nothing here reads the text) and no per-token
 * query (one number per position per cycle, learned from labels the verify already produced).
 * The workload estimate is not a prior over the class: it is evidence, so a recycled slot's a[] is
 * the measured level of acceptance, and its own labels move it from there. Where the classes are
 * interleaved the workload estimate is their running mix; a new request's first ~half-life of
 * labels is then centred between them, which is the best a model without a class key can say.
 */
#pragma once
#include "draft_types.h"

#include <cstdint>

namespace rad {

/* One request's estimate. Identity is the view's (req_id, slot_gen); `a[j]` is the chance position
 * j+1 is accepted given that 1..j were. `n[j]` is the raw label count that drives the EMA step
 * (the controller's schedule); `ess[j]` is the effective sample size, prior pseudo-labels
 * included, that unc_at() uses. */
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
    /* Rebinds to a larger table whose first old-cap entries the caller carried over (a vector
     * resize does), so every learned history stays; only the new tail starts at the prior. A
     * request slot indexes the scheduler's request table, which can outgrow max_seqs. */
    int grow(DraftPredictState* states, int32_t cap);
    const Params& params() const { return p_; }

    /* DraftPredictFn: fills out->seq[] in view order, starting a recycled slot at the workload's
     * measured acceptance (the prior while there is none), and stamps the view's cycle into the
     * set: draft_predictions_current compares (cycle, version, rows), so a set computed for an
     * earlier cycle is not current while its history still is.
     * Refuses a view of another policy version. */
    int predict(const DraftBatchView& view, DraftPredictions* out) const;

    /* The workload's position-wise acceptance, [0..max_depth), for a diagnostic or a test: what a
     * slot with no labels of its own starts from. */
    float workload_a(int j) const { return j >= 0 && j < RAD_SCHED_MAX_SPEC ? pop_a_[j] : 0.0f; }

    /* The trampoline an integrator puts in DraftHooks::predictor. */
    static int predict_hook(void* self, const DraftBatchView& view, DraftPredictions* out);

    /* observe()'s delivery ledger, cumulative. RAD_OK says the cycle was deliverable; it does not
     * say a row reached the estimate -- every row of a cycle can be skipped by a recycled
     * (req_id, slot_gen) and the call still returns OK. The scheduler publishes these as
     * SchedMetrics::draft_obs_*. */
    struct Counts {
        uint64_t rows_attempted = 0;  /* request/sequence rows handed to the loop: one per row,
                                       * not per token position of a wider window */
        uint64_t rows_applied = 0;    /* rows that applied a learning update -- at least one
                                       * position's state moved; the numeric estimate need not
                                       * change */
        uint64_t rows_skipped = 0;    /* rows that taught nothing: recycled identity, other
                                       * origin, or K0 */
    };
    const Counts& counts() const { return counts_; }

    /* DraftObserveFn's B half, composed by the integrator after a cycle the caller declared
     * complete. The cycle must pass draft_observation_deliverable: it names its plan, its rows are
     * complete and valid, and each row's origin fits the plan's permission. Refuses a malformed
     * observation, an unclassified cut, a plan that fails the shared issuance gate
     * (draft_plan_valid bounds its rows), and a slot outside the caller's table. A row whose slot
     * was recycled to another identity is skipped, not merged. */
    int observe(const DraftObservation& obs) const;

private:
    void  start(DraftPredictState& s, uint64_t req_id, uint32_t slot_gen) const;
    bool  learn(DraftPredictState& s, const DraftSeqObservation& o) const;
    float unc_at(const DraftPredictState& s, int j) const;

    Params p_;
    DraftPredictState* states_ = nullptr;
    int32_t cap_ = 0;
    double  alpha_ = 0.0;   /* the EMA step at half_life */
    /* The workload's position-wise acceptance: the same EMA over every labelled position of every
     * request, used as the starting point of a slot with no labels of its own (class visibility on
     * the yield side, see the header). Not per request and not a sample count of any one slot. */
    mutable float    pop_a_[RAD_SCHED_MAX_SPEC] = {};
    mutable float    pop_ess_[RAD_SCHED_MAX_SPEC] = {};
    mutable uint16_t pop_n_[RAD_SCHED_MAX_SPEC] = {};
    mutable Counts counts_;
};

}  /* namespace rad */
