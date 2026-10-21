/* draft_predictor.h -- B: each request's conditional draft acceptance, position by position.
 *
 * One model per scheduler. The scheduler asks it once per cycle, before a window is chosen
 * (DraftHooks::predictor), and hands it a completed cycle's observations afterwards (observe).
 * It is the per-request half of the adaptive controller in draft.h -- the same position-wise EMA
 * -- but with labels, censoring and evidence kept apart, so the policy can price a position by
 * what the verify measured rather than by what the window assumed.
 *
 * What counts as evidence:
 *   - A label exists only where the target checked the position: the accepted prefix is 1, the
 *     first rejection 0, and nothing past either. A censoring cycle (EOG, max_tokens, cancel,
 *     context end) has no rejection label -- its accepted prefix is measured, the rest is not.
 *   - Positions a shallow window did not reach drift toward the deepest measured position, so a
 *     sequence held shallow by prose can move back to code. The drift is not a sample: it does not
 *     touch n[] and never sets a valid bit, so a prior cannot turn into calibration evidence.
 *   - Only the head's own proposals teach this model. A prompt-lookup draft has another length
 *     distribution; the caller marks those rows and they are skipped, never averaged in.
 *   - A K0 cycle generates nothing and leaves the estimate untouched: the request keeps its
 *     history and recovers at the next drafted cycle.
 *
 * Memory: DraftPredictState is one per request slot, owned by the scheduler, and nothing here
 * allocates. A slot whose (req_id, slot_gen) differs from its state is recycled and starts at the
 * prior; preemption keeps the identity and therefore the estimate. A request that moves to another
 * slot needs its state copied by the scheduler; a fresh slot starts over.
 */
#pragma once
#include "draft_types.h"

#include <cstdint>

namespace rad {

/* Which drafter produced an observation's proposals. `Lookup` is prompt lookup, not the head: the
 * n-gram gate is a separate drafter with its own label distribution (docs/ADAPTIVE-MTP.md), so its
 * rows never update this model. */
enum class DraftProposalSource : uint8_t {
    Head = 0,
    Lookup = 1,
};

/* One request's estimate. Identity is the view's (req_id, slot_gen); `a[j]` is the chance position
 * j+1 is accepted given that 1..j were, and `n[j]` counts the verifies that labelled it. */
struct DraftPredictState {
    uint64_t req_id = 0;
    uint32_t slot_gen = 0;
    float    a[RAD_SCHED_MAX_SPEC] = {};
    uint16_t n[RAD_SCHED_MAX_SPEC] = {};
};

class DraftPredictModel {
public:
    struct Params {
        int32_t  max_depth = 0;      /* positions to estimate; 0 = no drafter loaded */
        float    prior = 0.8f;       /* draft.h's kPrior: optimistic on purpose */
        float    half_life = 16.0f; /* draft.h's kHalfLife, in labelled verifies */
        float    prior_weight = 4.0f; /* pseudo-observations behind the prior (kPriorWeight) */
        float    unc_z = 2.0f;       /* half-width in standard deviations */
        uint32_t version = 1;       /* stamped into predictions; observations of another policy
                                     * version are refused, never learned from */
    };

    /* `states` is the caller's per-slot table, [cap]. Everything is reset to the prior. */
    int configure(const Params& p, DraftPredictState* states, int32_t cap);
    const Params& params() const { return p_; }

    /* DraftPredictFn: fills out->seq[] in view order, starting a recycled slot at the prior. */
    int predict(const DraftBatchView& view, DraftPredictions* out) const;
    /* The trampoline an integrator puts in DraftHooks::predictor. */
    static int predict_hook(void* self, const DraftBatchView& view, DraftPredictions* out);

    /* DraftObserveFn's B half, composed by the integrator after a cycle the caller declared
     * complete. `source` is [obs.cap], row-aligned with the observation; null means all Head.
     * Refuses an incomplete cycle, a plan whose identity differs from the observation's, and a
     * slot whose current generation is not the one observed. */
    int observe(const DraftObservation& obs, const DraftProposalSource* source = nullptr) const;

private:
    void  start(DraftPredictState& s, uint64_t req_id, uint32_t slot_gen) const;
    void  learn(DraftPredictState& s, const DraftSeqObservation& o,
                DraftProposalSource src) const;
    float unc_at(const DraftPredictState& s, int j) const;

    Params p_;
    DraftPredictState* states_ = nullptr;
    int32_t cap_ = 0;
    double  alpha_ = 0.0;   /* the EMA step at half_life */
};

}  /* namespace rad */
