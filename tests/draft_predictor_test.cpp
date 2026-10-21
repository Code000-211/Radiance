/* draft_predictor_test.cpp -- B's position-wise acceptance model (core/sched/draft_predictor.h).
 *
 * What is checked here is what the policy may rely on: a label exists only where the target
 * checked the position, and a censoring cycle has no rejection label; lookups and K0 rows never
 * enter the head's estimate; extrapolation moves a prior without inflating evidence; a recycled
 * slot starts over while preemption keeps its history; uncertainty narrows with labels; and an
 * offline replay of code -> prose -> code gives the same answer whatever the batch order.
 *
 * No card, no scheduler: the observations are the ones E will publish.
 */
#include "rad_test.h"

#include "sched/draft_predictor.h"

#include <cmath>
#include <cstring>
#include <string>

using namespace rad;

static_assert(RAD_SCHED_MAX_SPEC == 64, "the verify kernel band is the array bound");

namespace {

constexpr int32_t kCap = 8;

/* The scheduler-side storage: one state per slot, the view it fills, the predictions it receives,
 * and the observation it hands back. */
struct Rig {
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
    DraftProposalSource    src[kCap];
    const DraftProposalSource* srcs = nullptr;
    uint64_t               cycle = 0;

    explicit Rig(int depth = 6) {
        DraftPredictModel::Params p;
        p.max_depth = depth;
        CHECK_OK(model.configure(p, state, kCap));
        for (DraftSeqView& v : seq) v = DraftSeqView{};
        view.seq = seq;
        view.cap = kCap;
        view.cycle_id = 1;
        view.k_ceiling = depth;
        view.policy_version = p.version;
        pred.seq = pseq;
        pred.req_id = pid;
        pred.cap = kCap;
        for (DraftSeqObservation& o : sobs) o = DraftSeqObservation{};
        obs.cap = kCap;
        obs.req_id = oid;
        obs.slot = oslot;
        obs.slot_gen = ogen;
        obs.seq = sobs;
        obs.policy_version = p.version;
        obs.complete = true;
        for (DraftProposalSource& s : src) s = DraftProposalSource::Head;
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

    /* One head observation for the identity last predicted at `slot`. */
    int observe(uint64_t id, int drafted, int accepted, DraftCensor censor = DraftCensor::None,
                int checked = -1, int generated = -1,
                DraftProposalSource source = DraftProposalSource::Head, int32_t slot = 0,
                uint32_t gen = 1) {
        obs.cycle_id = ++cycle;
        view.cycle_id = obs.cycle_id;
        obs.n_seq = 1;
        oid[0] = id;
        oslot[0] = slot;
        ogen[0] = gen;
        sobs[0] = DraftSeqObservation{};
        sobs[0].accepted = (uint8_t)accepted;
        sobs[0].checked = (uint8_t)(checked < 0
                                        ? accepted + (accepted < drafted ? 1 : 0)
                                        : checked);
        sobs[0].verified = (uint8_t)(drafted < 0 ? sobs[0].checked : drafted);
        sobs[0].generated = (uint8_t)(generated < 0 ? drafted : generated);
        sobs[0].published = sobs[0].generated;
        sobs[0].censor = censor;
        src[0] = source;
        return model.observe(obs, src);
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
        CHECK_NEAR(r.pseq[0].unc[j], 0.3577709, 1e-5);  /* 2 * sqrt(.16 / 5) */
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
    /* The unreached positions drifted toward the rejection but carry no evidence. */
    CHECK(p.a[3] < 0.8f);
    CHECK_NEAR(p.a[3], 0.8f + (1.0 - std::exp2(-1.0 / 16.0)) * (0.64f - 0.8f), 1e-6);
}

TEST(checked_past_the_first_rejection_does_not_buy_more_labels) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    /* accepted=1, checked=4: verified positions 2 and 3 were not checked for acceptance; a
     * caller that set checked = verified must not pull the estimate down twice. */
    CHECK_OK(r.observe(10, 4, 1, DraftCensor::None, 4));

    CHECK_EQ(r.state[0].n[0], 1);
    CHECK_EQ(r.state[0].n[1], 1);
    CHECK_EQ(r.state[0].n[2], 0);
    CHECK_EQ(r.state[0].n[3], 0);
    CHECK_NEAR(r.state[0].a[1], 0.64, 1e-6);
    CHECK_EQ(predicted(r, 10, 0).valid, 0x3ull);
}

TEST(a_censoring_cycle_has_no_rejection_label) {
    const DraftCensor censors[] = { DraftCensor::Eog, DraftCensor::MaxTokens,
                                    DraftCensor::Cancelled, DraftCensor::ContextEnd };
    for (DraftCensor c : censors) {
        Rig r(6);
        REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
        /* accepted=2 but checked=3: the naive count would turn EOG into a rejection at 2. */
        CHECK_OK(r.observe(10, 3, 2, c, 3));
        CHECK_EQ(r.state[0].n[0], 1);
        CHECK_EQ(r.state[0].n[1], 1);
        CHECK_EQ(r.state[0].n[2], 0);
        CHECK_NEAR(r.state[0].a[2], 0.8f + (1.0 - std::exp2(-1.0 / 16.0)) * (0.84f - 0.8f), 1e-6);
        CHECK_EQ(predicted(r, 10, 0).valid, 0x3ull);
    }
    /* The same numbers without a censor do label the rejection. */
    Rig plain(6);
    REQUIRE_EQ(predicted(plain, 10, 0).valid, 0u);
    CHECK_OK(plain.observe(10, 3, 2, DraftCensor::None, 3));
    CHECK_EQ(plain.state[0].n[2], 1);
    CHECK_NEAR(plain.state[0].a[2], 0.64, 1e-6);
}

TEST(lookups_and_k0_rows_never_enter_the_head_estimate) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);

    CHECK_OK(r.observe(10, 6, 6, DraftCensor::None, -1, -1, DraftProposalSource::Lookup));
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

TEST(a_recycled_slot_starts_over_and_preemption_keeps_the_estimate) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);
    CHECK_OK(r.observe(10, 3, 3));
    const float learned = r.state[0].a[1];
    CHECK(learned > 0.8f);

    /* Same request, same slot: preemption does not reset it. */
    r.add(0, 10, 0, 1);
    CHECK_OK(r.predict());
    CHECK_NEAR(r.state[0].a[1], learned, 0.0);

    /* The slot is recycled for another generation: the new request starts at the prior. */
    const DraftPrediction fresh = predicted(r, 11, 0, 2);
    CHECK_EQ(r.state[0].req_id, 11u);
    CHECK_EQ(r.state[0].slot_gen, 2u);
    CHECK_EQ(fresh.valid, 0u);
    CHECK_EQ(fresh.a[0], 0.8f);

    /* And the old request on another slot starts over too (the scheduler moves state itself). */
    r.add(0, 10, 1, 1);
    CHECK_OK(r.predict());
    CHECK_EQ(r.pseq[0].a[0], 0.8f);
    CHECK_EQ(r.pseq[0].n[0], 0);
}

TEST(uncertainty_narrows_with_evidence_and_stays_a_bound) {
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
        if (i % 8 == 0) CHECK_EQ(p.n[0], i);
    }
    CHECK_NEAR(r.state[0].a[0], 1.0, 0.05);

    /* A rejection run narrows it again around the other answer, never past the bound. */
    for (int i = 0; i < 96; ++i) CHECK_OK(r.observe(10, 6, 0, DraftCensor::None, 1));
    const DraftPrediction p = predicted(r, 10, 0);
    CHECK_NEAR(r.state[0].a[0], 0.0, 0.05);
    CHECK(p.unc[0] >= 0.0f && p.unc[0] <= 1.0f);
}

TEST(an_observation_must_name_a_complete_cycle_and_its_own_generation) {
    Rig r(6);
    REQUIRE_EQ(predicted(r, 10, 0).valid, 0u);

    r.obs.complete = false;
    const std::string incomplete = stderr_of([&] { CHECK_EQ(r.observe(10, 3, 3), RAD_E_INVAL); });
    CHECK(has(incomplete, "not complete"));
    CHECK_EQ(r.state[0].n[0], 0);
    r.obs.complete = true;

    /* A plan for another cycle is stale; its rows are not this cycle's. */
    int32_t k[kCap] = { 2 };
    DraftFallback why[kCap] = {};
    int32_t rows[RAD_SCHED_MAX_SPEC + 1] = {};
    uint64_t plan_id[kCap] = { 10 };
    int32_t plan_slot[kCap] = { 0 };
    uint32_t plan_gen[kCap] = { 1 };
    DraftPlan plan;
    plan.cycle_id = 99;
    plan.policy_version = r.obs.policy_version;
    plan.n_seq = 1;
    plan.cap = kCap;
    plan.k_ceiling = 6;
    plan.max_k = 2;
    plan.req_id = plan_id;
    plan.slot = plan_slot;
    plan.slot_gen = plan_gen;
    plan.k = k;
    plan.why = why;
    plan.rows = rows;
    REQUIRE_EQ(draft_plan_rows(plan), RAD_OK);
    r.obs.plan = &plan;
    const std::string stale = stderr_of([&] { CHECK_EQ(r.observe(10, 3, 3), RAD_E_INVAL); });
    CHECK(has(stale, "does not match its plan"));
    CHECK_EQ(r.state[0].n[0], 0);

    /* The same plan, the same cycle: the labels land. */
    plan.cycle_id = r.cycle + 1;
    CHECK_OK(r.observe(10, 3, 3));
    CHECK_EQ(r.state[0].n[0], 1);
    r.obs.plan = nullptr;

    /* Another generation in the same slot is refused rather than mixed. */
    r.ogen[0] = 3;
    CHECK_EQ(r.model.observe(r.obs, nullptr), RAD_E_INVAL);
    CHECK_EQ(r.state[0].n[0], 1);
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
        CHECK_OK(mixed.observe(20, 5, 4, DraftCensor::None, -1, -1,
                               DraftProposalSource::Head, 1, 7));
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
    CHECK(alone.n[2] == 34u);
}

TEST(sampling_classes_do_not_couple_across_requests) {
    /* Greedy and sampled requests have their own RNG coordinates; nothing in B may average one
     * request's labels into another's estimate. */
    Rig both(6);
    both.add(0, 10, 0);
    both.add(1, 20, 1);
    CHECK_OK(both.predict());
    both.seq[0].sampling = DraftSampling::Greedy;
    both.seq[1].sampling = DraftSampling::Sampled;

    Rig greedy(6);
    REQUIRE_EQ(predicted(greedy, 10, 0).valid, 0u);
    Rig sampled(6);
    REQUIRE_EQ(predicted(sampled, 20, 1).valid, 0u);

    for (int i = 0; i < 8; ++i) {
        both.view.n_seq = 2;
        CHECK_OK(both.predict());
        CHECK_OK(greedy.observe(10, 6, 6));
        CHECK_OK(sampled.observe(20, 6, 2, DraftCensor::None, -1, -1,
                                 DraftProposalSource::Head, 1));
        CHECK_OK(both.observe(10, 6, 6));
        CHECK_OK(both.observe(20, 6, 2, DraftCensor::None, -1, -1,
                              DraftProposalSource::Head, 1));
    }

    for (int j = 0; j < 6; ++j) {
        CHECK_EQ(both.state[0].a[j], greedy.state[0].a[j]);
        CHECK_EQ(both.state[1].a[j], sampled.state[1].a[j]);
        CHECK_EQ(both.state[0].n[j], greedy.state[0].n[j]);
        CHECK_EQ(both.state[1].n[j], sampled.state[1].n[j]);
    }
}

TEST(predict_refuses_a_batch_it_cannot_fill) {
    Rig r(6);
    r.add(0, 10, 0);
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

RAD_TEST_MAIN()
