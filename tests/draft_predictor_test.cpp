/* draft_predictor_test.cpp -- B's position-wise acceptance model (core/sched/draft_predictor.h).
 *
 * What is checked here is what the policy may rely on: a label exists only where draft_label()
 * measured one -- the accepted prefix and, only when the cut is Rejected, the position past it;
 * a Stopped (matched EOG, not committed) row carries no label, and an Unset cut is refused. The
 * origin is the observation's own field and must fit the plan's permission, so nothing becomes
 * "head" by default. Lookups and K0 rows never enter the head's estimate. An observation that
 * names no plan is refused by the contract's gate; a cycle from an older policy stays observable
 * while it is alive; a recycled slot is skipped, never merged into the new generation.
 * Extrapolation moves a prior without inflating evidence; the reported n is the raw label count
 * while the effective sample size only shapes unc; and an offline replay of code -> prose -> code
 * gives the same answer whatever the batch order.
 *
 * No card, no scheduler, no RNG: the observations are the ones E will publish, and nothing here
 * claims a target-sampling or device property.
 */
#include "rad_test.h"

#include "sched/draft.h"
#include "sched/draft_predictor.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace rad;

static_assert(RAD_SCHED_MAX_SPEC == 64, "the verify kernel band is the array bound");

namespace {

constexpr int32_t kCap = 8;

/* The scheduler-side storage: one state per slot, the view it fills, the predictions it receives,
 * the plan a cycle was frozen under, and the observation it hands back. */
struct Rig {
    DraftPredictModel::Params params;
    DraftPredictState      state[kCap];
    DraftPredictModel      model;
    DraftSeqView           seq[kCap];
    DraftBatchView         view;
    DraftPrediction        pseq[kCap];
    uint64_t               pid[kCap];
    DraftPredictions       pred;
    DraftObservation       obs;
    DraftSeqObservation    sobs[kCap];
    uint64_t               oid[kCap];
    int32_t                oslot[kCap];
    uint32_t               ogen[kCap];
    DraftPlan              plan;
    int32_t                plan_k[kCap];
    DraftFallback          plan_why[kCap];
    int32_t                plan_rows[RAD_SCHED_MAX_SPEC + 1];
    uint64_t               plan_id[kCap];
    int32_t                plan_slot[kCap];
    uint32_t               plan_gen[kCap];
    DraftDrafter           plan_fill[kCap];
    int32_t                plan_ctx[kCap];
    int32_t                plan_hist[kCap];
    uint64_t               cycle = 0;

    explicit Rig(int depth = 6, uint32_t version = 1) {
        params.max_depth = depth;
        params.version = version;
        CHECK_OK(model.configure(params, state, kCap));
        for (DraftSeqView& v : seq) v = DraftSeqView{};
        view.seq = seq;
        view.cap = kCap;
        view.cycle_id = 1;
        view.k_ceiling = depth;
        view.policy_version = version;
        pred.seq = pseq;
        pred.req_id = pid;
        pred.cap = kCap;
        for (DraftSeqObservation& o : sobs) o = DraftSeqObservation{};
        obs.cap = kCap;
        obs.req_id = oid;
        obs.slot = oslot;
        obs.slot_gen = ogen;
        obs.seq = sobs;
        obs.policy_version = version;
        obs.complete = true;
        plan.cap = kCap;
        plan.policy_version = version;
        plan.k_ceiling = depth;
        plan.k = plan_k;
        plan.why = plan_why;
        plan.rows = plan_rows;
        plan.req_id = plan_id;
        plan.slot = plan_slot;
        plan.slot_gen = plan_gen;
        plan.fill = plan_fill;
        plan.ctx_class = plan_ctx;
        plan.history_tokens = plan_hist;
    }

    /* Request i of the next batch: identity, slot and generation. */
    DraftSeqView& add(int32_t i, uint64_t id, int32_t slot, uint32_t gen = 1) {
        DraftSeqView& v = seq[i];
        v = DraftSeqView{};
        v.req_id = id;
        v.slot = slot;
        v.slot_gen = gen;
        v.ctx_len = 1024;
        v.ctx_class = draft_ctx_class(v.ctx_len);
        v.remaining = 64;
        view.n_seq = i + 1;
        return v;
    }

    int predict() { return model.predict(view, &pred); }

    /* A policy bump on the same table: what predict serves changes, the learned state does not.
     * The observation and plan of a cycle already in flight keep their version. */
    void bump(uint32_t version) {
        params.version = version;
        CHECK_OK(model.configure(params, state, kCap));
        view.policy_version = version;
    }

    /* One head observation for the identity last predicted at `slot`, with the plan it came from.
     * `drafted` is the verified window; the plan's k[] and fill[] say who drafted it. `end` and
     * `origin` are the observation's own fields: -1 derives the honest default (Rejected when the
     * draft missed, AllAccepted when it did not; Head for a row with drafts, None for K0), and an
     * explicit value -- including Unset -- is passed through so the refusals can be tested. */
    void arm(uint64_t id, int drafted, int accepted, DraftCensor censor = DraftCensor::None,
             int checked = -1, int generated = -1, DraftDrafter fill = DraftDrafter::Serial,
             int32_t slot = 0, uint32_t gen = 1, int end = -1, int origin = -1) {
        obs.cycle_id = ++cycle;
        view.cycle_id = obs.cycle_id;
        obs.n_seq = 1;
        oid[0] = id;
        oslot[0] = slot;
        ogen[0] = gen;
        sobs[0] = DraftSeqObservation{};
        const DraftSeqEnd e =
            end < 0 ? (accepted < drafted ? DraftSeqEnd::Rejected : DraftSeqEnd::AllAccepted)
                    : (DraftSeqEnd)end;
        sobs[0].end = e;
        sobs[0].origin = origin < 0 ? (drafted == 0 ? DraftOrigin::None : DraftOrigin::Head)
                                    : (DraftOrigin)origin;
        sobs[0].accepted = (uint8_t)accepted;
        sobs[0].checked = (uint8_t)(checked < 0
                                        ? accepted + (e == DraftSeqEnd::Rejected ? 1 : 0)
                                        : checked);
        sobs[0].verified = (uint8_t)drafted;
        sobs[0].generated = (uint8_t)(generated < 0 ? drafted : generated);
        sobs[0].published = sobs[0].generated;
        sobs[0].raw_match = (uint64_t)(e == DraftSeqEnd::Stopped ? accepted + 1 : accepted);
        sobs[0].censor = censor;
        /* A single-window fixture by default: the plan named here published the window the verify
         * measured. Tests that need two plans set the label fields after arm(). */
        sobs[0].label_origin = sobs[0].origin;
        sobs[0].label_generated = sobs[0].generated;

        plan.cycle_id = obs.cycle_id;
        plan.cap = kCap;
        plan.n_seq = 1;
        plan.max_k = drafted > 0 ? drafted : 0;
        plan.verify_k = plan.max_k;
        plan_id[0] = id;
        plan_slot[0] = slot;
        plan_gen[0] = gen;
        plan_k[0] = plan.max_k;
        plan_why[0] = DraftFallback::None;
        plan_fill[0] = plan.max_k == 0 ? DraftDrafter::None : fill;
        plan_ctx[0] = 0;
        plan_hist[0] = 0;
        CHECK_OK(draft_plan_rows(plan));
        obs.plan = &plan;
    }

    int observe(uint64_t id, int drafted, int accepted, DraftCensor censor = DraftCensor::None,
                int checked = -1, int generated = -1,
                DraftDrafter fill = DraftDrafter::Serial, int32_t slot = 0, uint32_t gen = 1,
                int end = -1, int origin = -1) {
        arm(id, drafted, accepted, censor, checked, generated, fill, slot, gen, end, origin);
        return model.observe(obs);
    }
};

/* The prediction for a one-request batch: fill it and return it. */
DraftPrediction predicted(Rig& r, uint64_t id, int32_t slot, uint32_t gen = 1) {
    r.add(0, id, slot, gen);
    CHECK_OK(r.predict());
    return r.pseq[0];
}

}  /* namespace */

TEST(predict_starts_every_position_at_the_prior_with_no_evidence) {
    Rig r(4);
    r.add(0, 10, 0);
    DraftPredictor hook{&r.model, &DraftPredictModel::predict_hook};
    REQUIRE_EQ(hook.predict(hook.ctx, r.view, &r.pred), RAD_OK);   /* the contract's signature */

    CHECK_EQ(r.pred.n_seq, 1);
    CHECK_EQ(r.pred.req_id[0], 10u);
    for (int j = 0; j < 4; ++j) {
        CHECK_EQ(r.pseq[0].a[j], 0.8f);
        CHECK_EQ(r.pseq[0].n[j], 0);
        CHECK_NEAR(r.pseq[0].unc[j], 0.3577709, 1e-5);  /* 2 * sqrt(.16 / (4 + 1)) */
    }
    CHECK_EQ(r.pseq[0].valid, 0u);
    /* Past the configured depth there is nothing to predict. */
    CHECK_EQ(r.pseq[0].a[4], 0.0f);
    CHECK_EQ(r.pseq[0].n[4], 0);
}

TEST(labels_are_the_accepted_prefix_and_the_first_rejection) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    CHECK_OK(r.observe(10, 6, 2));   /* checked = 3, verified = 6 */

    /* Prior 0.8, first weight 1 / (0 + 4 + 1) = 0.2: two accepts and one rejection. */
    CHECK_NEAR(r.state[0].a[0], 0.84, 1e-6);
    CHECK_NEAR(r.state[0].a[1], 0.84, 1e-6);
    CHECK_NEAR(r.state[0].a[2], 0.64, 1e-6);
    CHECK_EQ(r.state[0].n[0], 1);
    CHECK_EQ(r.state[0].n[1], 1);
    CHECK_EQ(r.state[0].n[2], 1);
    CHECK_EQ(r.state[0].n[3], 0);

    const DraftPrediction p = predicted(r, 10, 0);
    CHECK_EQ(p.valid, 0x7ull);
    CHECK_EQ(p.n[3], 0);
    /* The window reached six, so the drift starts at six: nothing in 3..5 moves on this cycle,
     * the same basis as DraftController::adapt_observe. */
    CHECK_EQ(p.a[3], 0.8f);
}

/* The seam with the half of the policy that already exists: on every cycle the controller can
 * express (no censor, so checked = accepted + 1), B's estimate and its drift move to the same
 * numbers as DraftController::adapt_observe -- same basis, same first weight, same edge and step.
 * Two arithmetics at this seam would price a position differently depending on which half saw the
 * verify, so this is checked rather than asserted in prose. */
TEST(the_estimate_and_drift_are_the_controllers_arithmetic) {
    DraftController ctrl;
    DraftController::Params cp;
    cp.max_depth = 6;
    cp.adapt = true;
    REQUIRE_EQ(ctrl.configure(cp), RAD_OK);

    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    DraftAdapt adapt;
    REQUIRE_EQ(ctrl.window_for(adapt), 6);   /* starts the estimate at kPrior, as Rig did */

    /* Full window with a rejection, full window with none, then a shallow one whose tail drifts. */
    const int drafted[3]  = {6, 6, 3};
    const int accepted[3] = {2, 0, 2};
    for (int i = 0; i < 3; ++i) {
        CHECK_OK(r.observe(10, drafted[i], accepted[i]));
        ctrl.adapt_observe(adapt, drafted[i], accepted[i]);
        for (int j = 0; j < 6; ++j) {
            CHECK_EQ(r.state[0].a[j], adapt.a[j]);
            CHECK_EQ(r.state[0].n[j], adapt.n[j]);
        }
    }
    /* And the shallow cycle moved the tail, so the loop above is not comparing a frozen array. */
    CHECK(r.state[0].a[5] < 0.8f);
}

TEST(a_row_that_labels_past_the_first_rejection_is_refused) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    /* The reviewer's probe, now a contract case: accepted=1, checked=4. The shared mask would
     * stop at position 1, but the row itself is invalid (Rejected means checked == accepted + 1),
     * so the whole cycle is refused before any state moves -- a caller that sets
     * checked = verified buys no labels and teaches nothing. */
    const std::string refused = stderr_of(
        [&] { CHECK_EQ(r.observe(10, 4, 1, DraftCensor::None, 4), RAD_E_INVAL); });
    CHECK(has(refused, "not a deliverable observation"));
    for (int j = 0; j < 6; ++j) {
        CHECK_EQ(r.state[0].n[j], 0);
        CHECK_EQ(r.state[0].a[j], 0.8f);
    }
    /* The same counts with the mask's own cut (checked = accepted + 1) teach one rejection. */
    CHECK_OK(r.observe(10, 4, 1, DraftCensor::None, 2));
    CHECK_EQ(r.state[0].n[1], 1);
    CHECK_EQ(r.state[0].n[2], 0);
    CHECK_NEAR(r.state[0].a[1], 0.64, 1e-6);
}

TEST(a_censored_cycle_keeps_the_rejection_it_checked) {
    {
        /* The draft missed the EOG: position 1 is a real, checked rejection, and the censor that
         * ended the cycle afterwards does not erase it (end = Rejected). */
        Rig r(6);
        REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
        CHECK_OK(r.observe(10, 2, 1, DraftCensor::Eog));
        CHECK_EQ(r.state[0].n[1], 1);
        CHECK_NEAR(r.state[0].a[1], 0.64, 1e-6);
        CHECK_EQ(predicted(r, 10, 0).valid, 0x3ull);
    }
    {
        /* The draft hit the EOG but the token was not committed: end = Stopped, checked ends at
         * the accepted prefix, so the position gets no false rejection. */
        Rig r(6);
        REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
        CHECK_OK(r.observe(10, 3, 2, DraftCensor::Eog, 2, -1, DraftDrafter::Serial, 0, 1,
                           (int)DraftSeqEnd::Stopped));
        CHECK_EQ(r.state[0].n[2], 0);
        CHECK_EQ(r.state[0].a[2], 0.8f);
        CHECK_EQ(predicted(r, 10, 0).valid, 0x3ull);
    }
    {
        /* Censored before anything was checked: nothing to learn. */
        Rig r(6);
        REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
        CHECK_OK(r.observe(10, 1, 0, DraftCensor::Eog, 0, -1, DraftDrafter::Serial, 0, 1,
                           (int)DraftSeqEnd::AllAccepted));
        for (int j = 0; j < 6; ++j) CHECK_EQ(r.state[0].n[j], 0);
    }
    {
        /* The same numbers without a censor label the rejection. */
        Rig r(6);
        REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
        CHECK_OK(r.observe(10, 3, 2, DraftCensor::None, 3));
        CHECK_EQ(r.state[0].n[2], 1);
        CHECK_NEAR(r.state[0].a[2], 0.64, 1e-6);
    }
    {
        /* An unclassified cut is refused, never guessed into a reject. */
        Rig r(6);
        REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
        CHECK_EQ(r.observe(10, 3, 2, DraftCensor::None, 2, -1, DraftDrafter::Serial, 0, 1,
                           (int)DraftSeqEnd::Unset),
                 RAD_E_INVAL);
        for (int j = 0; j < 6; ++j) CHECK_EQ(r.state[0].n[j], 0);
    }
}

TEST(the_origin_is_the_observations_and_the_plan_is_only_the_permission) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);

    /* A lookup row behind a head budget is refused by the contract: whoever wires the pipeline
     * cannot relabel it as the head's work. */
    r.arm(10, 6, 6, DraftCensor::None, -1, -1, DraftDrafter::Serial, 0, 1, -1,
          (int)DraftOrigin::Lookup);
    CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL);
    CHECK_EQ(r.state[0].n[0], 0);

    /* And a head row behind a lookup budget is not the lookup's either. */
    r.arm(10, 6, 6, DraftCensor::None, -1, -1, DraftDrafter::Ngram, 0, 1, -1,
          (int)DraftOrigin::Head);
    CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL);
    CHECK_EQ(r.state[0].n[0], 0);

    /* The fit pair teaches: the plan permits Serial and the observation says the head produced
     * it. */
    CHECK_OK(r.observe(10, 6, 6));
    CHECK_EQ(r.state[0].n[0], 1);

    /* A row that produced nothing is None whatever its budget was. */
    Rig k0(6);
    REQUIRE_EQ(predicted(k0, 20, 0).valid, 0u);
    CHECK_OK(k0.observe(20, 0, 0, DraftCensor::None, 0, 0, DraftDrafter::None, 0, 1, -1,
                        (int)DraftOrigin::None));
    CHECK_EQ(k0.state[0].n[0], 0);
}

/* Review 2, finding 2: with pipeline_depth 1 the labels belong to the window the PREVIOUS cycle
 * published, so the teaching origin is the label's, not the named plan's. */
TEST(the_labels_teach_by_the_window_they_measured_not_by_the_plans_origin) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);

    /* Head -> Lookup: plan N's row is a lookup row (fill Ngram, origin None), but its paired
     * verify measured the head window plan N-1 published. The head's labels teach. */
    r.arm(10, 6, 6, DraftCensor::None, -1, /*generated=*/0, DraftDrafter::Ngram, 0, 1, -1,
          (int)DraftOrigin::None);
    r.sobs[0].label_origin    = DraftOrigin::Head;
    r.sobs[0].label_generated = 6;
    CHECK_OK(r.model.observe(r.obs));
    CHECK_EQ(r.state[0].n[0], 1);
    CHECK_EQ(r.model.counts().rows_applied, 1u);
    CHECK_EQ(r.model.counts().rows_skipped, 0u);
    const float learned = r.state[0].a[0];

    /* Lookup -> Head: plan N's row is the head's, but the window it verified was the previous
     * cycle's lookup fill, which never published. Nothing to learn, and the estimate is intact.
     * A row whose label window drafted nothing has verified == 0 -- that is what the None label
     * means (draft_observation_deliverable), not a window that computed positions it cannot
     * name. */
    r.arm(10, 6, 0, DraftCensor::None, -1, -1, DraftDrafter::Serial, 0, 1,
          (int)DraftSeqEnd::AllAccepted);
    r.sobs[0].verified        = 0;
    r.sobs[0].label_origin    = DraftOrigin::None;
    r.sobs[0].label_generated = 0;
    CHECK_OK(r.model.observe(r.obs));
    CHECK_EQ(r.state[0].n[0], 1);
    CHECK_EQ(r.state[0].a[0], learned);
    CHECK_EQ(r.model.counts().rows_applied, 1u);
    CHECK_EQ(r.model.counts().rows_skipped, 1u);

    /* K0/changing K: a row whose label window drafted nothing teaches nothing, whatever this
     * plan's own K is; the next drafted window learns again, per position. */
    r.arm(10, 3, 0, DraftCensor::None, -1, -1, DraftDrafter::Serial, 0, 1,
          (int)DraftSeqEnd::AllAccepted);
    r.sobs[0].verified        = 0;
    r.sobs[0].label_origin    = DraftOrigin::None;
    r.sobs[0].label_generated = 0;
    CHECK_OK(r.model.observe(r.obs));
    CHECK_EQ(r.state[0].n[0], 1);

    r.arm(10, 3, 2);
    r.sobs[0].label_origin    = DraftOrigin::Head;
    r.sobs[0].label_generated = 3;
    CHECK_OK(r.model.observe(r.obs));
    CHECK_EQ(r.state[0].n[0], 2);
    CHECK_EQ(r.state[0].n[1], 2);
    CHECK_EQ(r.state[0].n[2], 2);
    CHECK_EQ(r.model.counts().rows_applied, 2u);
    CHECK_EQ(r.model.counts().rows_skipped, 2u);
}

/* Review 2, finding 3: RAD_OK is not application. A deliverable cycle whose every row was
 * recycled away applies nothing, and the ledger has to say so. */
TEST(a_deliverable_cycle_whose_rows_were_all_skipped_still_counts_them) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    /* The state at slot 0 belongs to another generation now. */
    CHECK_OK(r.observe(10, 6, 6, DraftCensor::None, -1, -1, DraftDrafter::Serial, 0, 2));
    CHECK_EQ(r.state[0].n[0], 0);
    CHECK_EQ(r.model.counts().rows_attempted, 1u);
    CHECK_EQ(r.model.counts().rows_applied, 0u);
    CHECK_EQ(r.model.counts().rows_skipped, 1u);
}

TEST(lookups_and_k0_rows_never_enter_the_head_estimate) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);

    CHECK_OK(r.observe(10, 6, 6, DraftCensor::None, -1, -1, DraftDrafter::Ngram, 0, 1, -1,
                       (int)DraftOrigin::Lookup));
    for (int j = 0; j < 6; ++j) {
        CHECK_EQ(r.state[0].n[j], 0);
        CHECK_EQ(r.state[0].a[j], 0.8f);
    }

    /* K0 with the head still loaded: no drafts generated, no labels, state untouched. A stretch
     * of K0 cycles -- a probe target for C -- must not erode the estimate it will come back to. */
    for (int i = 0; i < 8; ++i) CHECK_OK(r.observe(10, 0, 0, DraftCensor::None, 0, 0));
    for (int j = 0; j < 6; ++j) {
        CHECK_EQ(r.state[0].n[j], 0);
        CHECK_EQ(r.state[0].a[j], 0.8f);
    }

    CHECK_OK(r.observe(10, 3, 3));   /* back at depth 3: it learns from the preserved prior */
    CHECK_EQ(r.state[0].n[0], 1);
    CHECK_NEAR(r.state[0].a[0], 0.84, 1e-6);

    CHECK_OK(r.observe(10, 3, 1, DraftCensor::None, 2));
    CHECK_EQ(r.state[0].n[0], 2);
    CHECK_EQ(r.state[0].n[1], 2);
    CHECK_EQ(r.state[0].n[2], 1);
}

TEST(an_observation_without_its_plan_is_refused) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    r.arm(10, 3, 3);
    r.obs.plan = nullptr;
    const std::string refused = stderr_of([&] { CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL); });
    CHECK(has(refused, "not a deliverable observation"));
    /* A missing side list must not mean "the head". */
    CHECK_EQ(r.state[0].n[0], 0);
}

TEST(a_recycled_slot_starts_over_and_an_unchanged_identity_keeps_the_estimate) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    CHECK_OK(r.observe(10, 3, 3));
    const float learned = r.state[0].a[1];
    CHECK(learned > 0.8f);

    /* Same request, same slot, same generation: preemption does not reset it. */
    r.add(0, 10, 0, 1);
    CHECK_OK(r.predict());
    CHECK_NEAR(r.state[0].a[1], learned, 0.0);

    /* The slot is recycled for another generation: the new request starts at the prior. */
    const DraftPrediction fresh = predicted(r, 11, 0, 2);
    CHECK_EQ(r.state[0].req_id, 11u);
    CHECK_EQ(r.state[0].slot_gen, 2u);
    CHECK_EQ(fresh.valid, 0u);
    CHECK_EQ(fresh.a[0], 0.8f);

    /* A cycle of the old identity is still deliverable (the contract never consults the live
     * slot), but request 11 owns this table slot now: the old rows are skipped, never mixed into
     * the new generation's estimate. */
    CHECK_OK(r.observe(10, 3, 3, DraftCensor::None, -1, -1, DraftDrafter::Serial, 0, 1));
    CHECK_EQ(r.state[0].req_id, 11u);
    CHECK_EQ(r.state[0].n[0], 0);
    CHECK_EQ(r.state[0].a[0], 0.8f);

    /* And the old request on another slot starts over too (the scheduler moves state itself). */
    r.add(0, 10, 1, 1);
    CHECK_OK(r.predict());
    CHECK_EQ(r.pseq[0].a[0], 0.8f);
    CHECK_EQ(r.pseq[0].n[0], 0);

    /* Identity is the pair, not the request: a cycle of an older generation is deliverable after
     * a preempt -> recycle -> resume, but it teaches nothing to the resumed generation. */
    Rig p(6);
    REQUIRE_EQ(predicted(p, 11, 0, 2).valid, 0u);
    p.arm(11, 3, 3, DraftCensor::None, -1, -1, DraftDrafter::Serial, 0, 2);
    const DraftPrediction resumed = predicted(p, 11, 0, 4);
    CHECK_EQ(resumed.valid, 0u);
    CHECK_OK(p.model.observe(p.obs));
    CHECK_EQ(p.state[0].req_id, 11u);
    CHECK_EQ(p.state[0].slot_gen, 4u);
    CHECK_EQ(p.state[0].n[0], 0);
    CHECK_EQ(p.state[0].a[0], 0.8f);
}

TEST(uncertainty_is_computed_from_the_effective_sample_size) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    float previous = r.pseq[0].unc[0];
    CHECK(previous > 0.0f);
    for (int i = 1; i <= 32; ++i) {
        CHECK_OK(r.observe(10, 6, 6));
        const DraftPrediction p = predicted(r, 10, 0);
        CHECK(p.unc[0] < previous);
        CHECK(p.unc[0] >= 0.0f);
        previous = p.unc[0];
    }
    CHECK_NEAR(r.state[0].a[0], 1.0, 0.05);
    CHECK_EQ(r.state[0].n[0], 32);         /* the EMA's own count keeps growing ... */
    const DraftPrediction saturated = predicted(r, 10, 0);
    CHECK_EQ(saturated.n[0], 32);          /* ... and n reports it raw, not the EMA's reach */

    /* 1400 alternating labels: the estimate still rests on ~42 effective labels, so the reported
     * uncertainty stays near 2*sqrt(.25/47) instead of collapsing with the cycle count. */
    for (int i = 0; i < 1400; ++i)
        CHECK_OK(r.observe(10, 6, i % 2 == 0 ? 6 : 0));
    const DraftPrediction p = predicted(r, 10, 0);
    CHECK_EQ(r.state[0].n[0], 1432);
    CHECK_EQ(p.n[0], 1432);                /* raw count; the ~42 effective labels shape unc only */
    CHECK(p.unc[0] > 0.12f);
    CHECK(p.unc[0] < 0.18f);
    CHECK(p.unc[0] >= 0.0f && p.unc[0] <= 1.0f);
}

TEST(an_observation_must_name_a_complete_cycle_its_plan_and_a_known_slot) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);

    r.obs.complete = false;
    const std::string incomplete = stderr_of([&] { CHECK_EQ(r.observe(10, 3, 3), RAD_E_INVAL); });
    CHECK(has(incomplete, "not a deliverable observation"));
    CHECK_EQ(r.state[0].n[0], 0);
    r.obs.complete = true;

    /* A plan for another cycle is stale; its rows are not this cycle's. */
    r.arm(10, 3, 3);
    r.plan.cycle_id = 99;
    const std::string stale = stderr_of([&] { CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL); });
    CHECK(has(stale, "not a deliverable observation"));
    CHECK_EQ(r.state[0].n[0], 0);

    /* A plan whose row names another request is not this observation's plan. */
    r.arm(10, 3, 3);
    r.plan_id[0] = 11;
    CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL);

    /* An over-wide plan is refused by the shared issuance gate before a row is read. */
    r.arm(10, 3, 3);
    r.plan.cap = 0;
    const std::string oob = stderr_of([&] { CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL); });
    CHECK(has(oob, "names a plan that is not valid"));

    /* A row that contradicts itself teaches nothing. */
    r.arm(10, 3, 3, DraftCensor::None, 4);   /* checked > verified */
    CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL);
    r.arm(10, 3, 3, DraftCensor::None, 2);   /* accepted > checked */
    CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL);
    CHECK_EQ(r.state[0].n[0], 0);

    /* The same plan, the same cycle: the labels land. */
    CHECK_OK(r.observe(10, 3, 3));
    CHECK_EQ(r.state[0].n[0], 1);

    /* Recycle the slot while that cycle is still in the ring. The cycle stays deliverable -- the
     * gate never consults the live slot -- but its rows must not merge into the new generation. */
    const DraftPrediction fresh = predicted(r, 11, 0, 3);
    CHECK_EQ(fresh.valid, 0u);
    CHECK_OK(r.model.observe(r.obs));   /* r.obs still names request 10 of generation 1 */
    CHECK_EQ(r.state[0].req_id, 11u);
    CHECK_EQ(r.state[0].n[0], 0);
    CHECK_EQ(r.state[0].a[0], 0.8f);
}

TEST(an_extrapolated_position_carries_no_evidence) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    /* A shallow window held by prose for a long stretch: position 5 never got a label. */
    for (int i = 0; i < 64; ++i) CHECK_OK(r.observe(10, 2, 1, DraftCensor::None, 2));
    CHECK_EQ(r.state[0].n[5], 0);
    CHECK(r.state[0].a[5] < 0.8f);
    const DraftPrediction p = predicted(r, 10, 0);
    CHECK_EQ(p.valid & (1ull << 5), 0ull);
    CHECK_EQ(p.n[5], 0);
}

TEST(a_replay_is_order_independent_and_recovers_after_prose) {
    /* A recorded code -> prose -> code run: 12 all-accept cycles, 10 that reject at 2, then 12
     * that accept again. Replayed once alone and once interleaved with a second request whose
     * labels arrive in the opposite batch order. */
    struct Step { int drafted; int accepted; DraftCensor censor; };
    struct Script { Step step[34]; int n = 0; };

    Script code_then_prose_then_code;
    auto push = [](Script& s, int drafted, int accepted) {
        s.step[s.n++] = Step{ drafted, accepted, DraftCensor::None };
    };
    for (int i = 0; i < 12; ++i) push(code_then_prose_then_code, 6, 6);
    for (int i = 0; i < 10; ++i) push(code_then_prose_then_code, 6, 2);
    for (int i = 0; i < 12; ++i) push(code_then_prose_then_code, 6, 6);

    Rig solo(6);
    REQUIRE_EQ(predicted(solo, 10, 0).valid, 0u);
    float end_of_prose = 1.0f;
    for (int i = 0; i < code_then_prose_then_code.n; ++i) {
        const Step& s = code_then_prose_then_code.step[i];
        CHECK_OK(solo.observe(10, s.drafted, s.accepted, s.censor));
        if (i == 21) end_of_prose = solo.state[0].a[2];   /* after the prose stretch */
    }
    const DraftPrediction alone = predicted(solo, 10, 0);

    /* The same script, with a second request interleaved in the other order. Request 20 always
     * accepts at 4, so it cannot influence request 10. */
    Rig mixed(6);
    mixed.view.n_seq = 2;
    mixed.add(0, 10, 0);
    mixed.add(1, 20, 1, 7);
    CHECK_OK(mixed.predict());
    mixed.view.n_seq = 1;   /* only request 10 is observed per cycle; 20 is predicted separately */
    for (int i = 0; i < code_then_prose_then_code.n; ++i) {
        const Step& s = code_then_prose_then_code.step[i];
        CHECK_OK(mixed.observe(10, s.drafted, s.accepted, s.censor));
        mixed.add(0, 20, 1, 7);
        CHECK_OK(mixed.predict());
        CHECK_OK(mixed.observe(20, 5, 4, DraftCensor::None, -1, -1, DraftDrafter::Serial, 1, 7));
    }
    const DraftPrediction in_batch = predicted(mixed, 10, 0);

    for (int j = 0; j < 6; ++j) {
        CHECK_EQ(alone.a[j], in_batch.a[j]);
        CHECK_EQ(alone.n[j], in_batch.n[j]);
        CHECK_EQ(alone.unc[j], in_batch.unc[j]);
    }
    CHECK_EQ(alone.valid, in_batch.valid);
    /* Prose pulled position 2 down; code brought it back, without the replay ever seeing a
     * deeper label than 2 during the prose stretch. */
    CHECK(end_of_prose < 0.7f);
    CHECK(alone.a[2] > end_of_prose);
    CHECK_EQ(solo.state[0].n[2], 34u);    /* the raw label count */
    CHECK(alone.n[2] <= 43);             /* what the policy sees: the effective one */
}

TEST(requests_do_not_couple_across_slots) {
    /* Each slot's estimate is its own. This says nothing about sampling classes: the model has no
     * sampling dimension, so a request that changes class lands in one EMA. */
    Rig both(6);
    both.add(0, 10, 0);
    both.add(1, 20, 1);
    CHECK_OK(both.predict());

    Rig first(6);
    REQUIRE_EQ(predicted(first, 10, 0).valid, 0u);
    Rig second(6);
    REQUIRE_EQ(predicted(second, 20, 1).valid, 0u);

    for (int i = 0; i < 8; ++i) {
        both.view.n_seq = 2;
        CHECK_OK(both.predict());
        CHECK_OK(first.observe(10, 6, 6));
        CHECK_OK(second.observe(20, 6, 2, DraftCensor::None, -1, -1,
                                DraftDrafter::Serial, 1));
        CHECK_OK(both.observe(10, 6, 6));
        CHECK_OK(both.observe(20, 6, 2, DraftCensor::None, -1, -1,
                              DraftDrafter::Serial, 1));
    }

    for (int j = 0; j < 6; ++j) {
        CHECK_EQ(both.state[0].a[j], first.state[0].a[j]);
        CHECK_EQ(both.state[1].a[j], second.state[1].a[j]);
        CHECK_EQ(both.state[0].n[j], first.state[0].n[j]);
        CHECK_EQ(both.state[1].n[j], second.state[1].n[j]);
    }
}

TEST(an_older_consistent_cycle_still_teaches_after_a_version_bump) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    CHECK_OK(r.observe(10, 3, 3));
    const float learned = r.state[0].a[0];

    /* A second cycle is predicted under v1 and is still in flight when the policy moves to v2. */
    r.add(0, 10, 0);
    CHECK_OK(r.predict());
    r.bump(2);
    CHECK_NEAR(r.state[0].a[0], learned, 0.0);   /* the bump is provenance, not a reset */
    CHECK_OK(r.predict());                        /* v2 predictions are served */

    /* Its observation still names its v1 plan, and plan and observation agree: it must teach. */
    CHECK_OK(r.observe(10, 3, 3));
    CHECK_EQ(r.state[0].n[0], 2);
    CHECK_NEAR(r.state[0].a[0], 0.8666667, 1e-6);   /* 0.84 + 1/6 * (1 - 0.84) */

    /* An observation may not claim the current version while naming the older plan. */
    r.arm(10, 3, 3);
    r.obs.policy_version = 2;
    CHECK_EQ(r.model.observe(r.obs), RAD_E_INVAL);
    CHECK_EQ(r.state[0].n[0], 2);

    /* A prediction for the old version, however, is not this model's to make. */
    r.view.policy_version = 1;
    CHECK_EQ(r.predict(), RAD_E_INVAL);
}

TEST(a_prediction_set_is_bound_to_its_cycle_while_history_outlives_it) {
    /* Same requests, same policy version, another cycle: the set computed for the old cycle is
     * not current. The view's plan parameters belong to the cycle, and only the cycle names
     * them; the request set alone would let the previous cycle's set pass as this one's. */
    Rig c(6);
    c.add(0, 10, 0);
    c.view.cycle_id = 100;
    REQUIRE_EQ(c.model.predict(c.view, &c.pred), RAD_OK);
    CHECK_EQ(c.pred.cycle_id, 100u);
    CHECK(draft_predictions_current(c.view, c.pred));
    c.view.cycle_id = 101;
    CHECK(!draft_predictions_current(c.view, c.pred));
    REQUIRE_EQ(c.model.predict(c.view, &c.pred), RAD_OK);
    CHECK_EQ(c.pred.cycle_id, 101u);
    CHECK(draft_predictions_current(c.view, c.pred));

    /* A new cycle is not a new request: the history survives the stamp. */
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    CHECK_OK(r.observe(10, 3, 3));
    const uint16_t n = r.state[0].n[1];
    const float learned = r.state[0].a[1];
    CHECK(n > 0);
    r.add(0, 10, 0, 1);
    r.view.cycle_id = 7;
    CHECK_OK(r.predict());
    CHECK_EQ(r.pred.cycle_id, 7u);
    CHECK_EQ(r.state[0].req_id, 10u);
    CHECK_EQ(r.state[0].n[1], n);
    CHECK_NEAR(r.state[0].a[1], learned, 0.0);
}

TEST(predict_refuses_a_batch_it_cannot_fill) {
    Rig r(6);
    r.add(0, 10, 0);
    r.view.policy_version = 99;
    CHECK_EQ(r.model.predict(r.view, &r.pred), RAD_E_INVAL);
    r.view.policy_version = 1;
    CHECK_EQ(r.model.predict(r.view, nullptr), RAD_E_INVAL);
    r.pred.cap = 0;
    CHECK_EQ(r.model.predict(r.view, &r.pred), RAD_E_INVAL);
    r.pred.cap = kCap;
    r.seq[0].slot = kCap;
    CHECK_EQ(r.model.predict(r.view, &r.pred), RAD_E_INVAL);
    r.seq[0].slot = 0;
    r.view.n_seq = kCap + 1;
    CHECK_EQ(r.model.predict(r.view, &r.pred), RAD_E_INVAL);

    DraftPredictModel::Params bad;
    bad.max_depth = 6;
    bad.prior = 1.5f;
    CHECK_EQ(r.model.configure(bad, r.state, kCap), RAD_E_INVAL);
    bad.prior = 0.8f;
    bad.half_life = -1.0f;
    CHECK_EQ(r.model.configure(bad, r.state, kCap), RAD_E_INVAL);
}

TEST(a_refused_view_does_not_reset_a_live_slot_on_its_way_to_the_bad_row) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    CHECK_OK(r.observe(10, 3, 3));
    const float learned = r.state[0].a[0];
    CHECK_EQ(r.state[0].n[0], 1);

    /* A new identity in row 0 would reset the slot; row 1 is outside the table. The view is
     * refused as a whole, so the reset must not have happened on the way. */
    r.add(0, 12, 0, 2);
    r.add(1, 13, kCap, 3);
    CHECK_EQ(r.model.predict(r.view, &r.pred), RAD_E_INVAL);
    CHECK_EQ(r.state[0].req_id, 10u);
    CHECK_EQ(r.state[0].slot_gen, 1u);
    CHECK_EQ(r.state[0].n[0], 1);
    CHECK_NEAR(r.state[0].a[0], learned, 0.0);
}

RAD_TEST_MAIN()
