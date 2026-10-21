/* draft_types_test.cpp -- the P00 contracts (core/sched/draft_types.h), against the compiling
 * fakes in draft_fakes.h.
 *
 * What is checked here is what the six strands may assume: a plan's rows are the requests that
 * reach each round, a plan is refused before issue when its arrays disagree, only checked
 * positions carry a label, an unpriced shape is a visible fallback rather than a fabricated
 * number, and observing a completed cycle does not write into the plan it names.
 *
 * No card, no scheduler, no cost model: the point is that the seam is usable before any of those
 * exist.
 */
#include "rad_test.h"

#include "draft_fakes.h"

#include <cstring>
#include <string>

using namespace rad;

static_assert(RAD_SCHED_MAX_SPEC == 64, "the verify kernel band is the array bound");
static_assert(RAD_DRAFT_CTX_CLASSES == 4, "the profile's context classes");

TEST(plan_rows_count_the_requests_that_reach_each_round) {
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 100, 64);   /* K 6 */
    f.add(1, 101, 2);    /* K 2, capped by remaining output */
    f.add(2, 102, 0);     /* K 0 */
    f.add(3, 103, 3);    /* K 3 */

    draft_fake::Policy policy;
    policy.legacy_k = 6;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    CHECK_EQ(f.plan.max_k, 6);
    CHECK_EQ(f.plan.k[0], 6);
    CHECK_EQ(f.plan.k[1], 2);
    CHECK_EQ(f.plan.k[2], 0);
    CHECK_EQ(f.plan.k[3], 3);
    /* N_j: 6, 2 and 3 reach round 1; 6 and 3 reach 2; only 6 reaches 4..6. */
    const int32_t want[7] = { 0, 3, 3, 2, 1, 1, 1 };
    for (int j = 0; j <= 6; ++j) CHECK_EQ(f.plan.rows[j], want[j]);
    CHECK(f.plan.why[1] == DraftFallback::Resources);
    CHECK(f.plan.why[2] == DraftFallback::Resources);
    CHECK(f.plan.why[0] == DraftFallback::None);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(an_ineligible_request_keeps_its_identity_and_drafts_nothing) {
    draft_fake::Fixture f;
    f.add(0, 7);
    f.add(1, 8, 64, DraftEligibility::Grammar);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    CHECK_EQ(f.plan.k[1], 0);
    CHECK_EQ(f.plan.req_id[1], 8u);
    CHECK_EQ(f.plan.max_k, 3);
    /* A grammar-capped request is the view's eligibility, not the policy's fallback. */
    CHECK(f.plan.why[1] == DraftFallback::None);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(plan_validation_refuses_inconsistent_rows_and_k) {
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2, 2);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    const int32_t saved = f.plan.rows[2];
    f.plan.rows[2] += 1;
    const std::string log = stderr_of([&] { CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL); });
    CHECK(has(log, "k[] implies"));
    f.plan.rows[2] = saved;

    f.plan.k[0] = f.plan.max_k + 1;
    const std::string log2 = stderr_of([&] { CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL); });
    CHECK(has(log2, "outside"));
    f.plan.k[0] = 3;

    f.plan.max_k = f.plan.k_ceiling + 1;
    const std::string log3 = stderr_of([&] { CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL); });
    CHECK(has(log3, "ceiling"));
}

TEST(plan_rows_refuses_a_k_past_the_plans_maximum) {
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    f.plan.k[0] = f.plan.max_k + 1;
    const std::string log = stderr_of([&] { CHECK_EQ(draft_plan_rows(f.plan), RAD_E_INVAL); });
    CHECK(has(log, "outside"));
    CHECK_EQ(f.plan.rows[0], 0);
}

TEST(a_k0_plan_bars_every_drafter_including_the_ngram_fallback) {
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;          /* the head is loaded: K0 is a temporary state, not a process */
    f.view.ngram_allowed = true;   /* the view allows the fallback; the plan still decides */
    f.add(0, 1);
    f.add(1, 2);

    draft_fake::Policy policy;
    policy.legacy_k = 0;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 0);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
    for (int32_t i = 0; i < 2; ++i) {
        CHECK(f.plan.fill[i] == DraftDrafter::None);
        CHECK(draft_plan_bars_fill(f.plan, i));
    }

    /* A drafting plan names its filler; the fallback still may not take the request over. */
    f.plan.k[0] = 2;
    f.plan.max_k = 2;
    f.plan.fill[0] = DraftDrafter::Serial;
    CHECK_EQ(draft_plan_rows(f.plan), RAD_OK);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
    CHECK(!draft_plan_bars_fill(f.plan, 0));

    /* K 0 with a filler claimed, and K > 0 with nobody to fill it, are both refused. */
    f.plan.fill[1] = DraftDrafter::Ngram;
    const std::string log = stderr_of([&] { CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL); });
    CHECK(has(log, "fill"));
    f.plan.fill[1] = DraftDrafter::None;
    f.plan.fill[0] = DraftDrafter::None;
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
}

TEST(a_k0_plan_says_which_shape_it_was_priced_on) {
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.legacy_k = 0;
    policy.history_tokens = 8;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 0);
    /* "Policy K0 with history", not "no work": the head is loaded and its history still runs. */
    CHECK_EQ(f.plan.history_tokens[0], 8);
    CHECK_EQ(f.plan.ctx_class[0], 0);
    CHECK_EQ(f.plan.prefill_tokens, 0);
    CHECK(f.plan.timing_valid);
    CHECK(f.plan.replay == DraftReplay::None);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    /* The fields are part of the checked plan, not decoration. */
    f.plan.history_tokens[0] = -1;
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
    f.plan.history_tokens[0] = 0;
    f.plan.ctx_class[0] = RAD_DRAFT_CTX_CLASSES;   /* a class the profile never carries */
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
    f.plan.ctx_class[0] = 0;
    f.plan.prefill_tokens = -1;
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
}

TEST(labels_stop_at_the_first_rejection_and_at_a_censor) {
    DraftSeqObservation o;
    o.accepted = 1;
    o.checked = 2;   /* accepted prefix plus the first rejection */
    o.verified = 4;  /* the target computed more than it may label */

    float label = -1.0f;
    REQUIRE(draft_label(o, 0, &label));
    CHECK_EQ(label, 1.0f);
    REQUIRE(draft_label(o, 1, &label));
    CHECK_EQ(label, 0.0f);
    CHECK(!draft_label(o, 2, &label));
    CHECK(!draft_label(o, 3, &label));

    DraftSeqObservation all;
    all.accepted = 3;
    all.checked = 3;   /* every draft matched: no zero label */
    all.verified = 3;
    CHECK(draft_label(all, 2, &label));
    CHECK_EQ(label, 1.0f);
    CHECK(!draft_label(all, 3, &label));

    /* The reviewer's repro: checked covers four positions but the first rejection is at 1. */
    DraftSeqObservation long_tail;
    long_tail.accepted = 1;
    long_tail.checked = 4;
    long_tail.verified = 6;
    REQUIRE(draft_label(long_tail, 0, &label));
    CHECK_EQ(label, 1.0f);
    REQUIRE(draft_label(long_tail, 1, &label));
    CHECK_EQ(label, 0.0f);
    CHECK(!draft_label(long_tail, 2, &label));
    CHECK(!draft_label(long_tail, 3, &label));

    DraftSeqObservation cut;
    cut.accepted = 1;
    cut.checked = 1;   /* an EOG cut after the committed prefix */
    cut.verified = 3;
    cut.censor = DraftCensor::Eog;
    CHECK(draft_label(cut, 0, &label));
    CHECK(!draft_label(cut, 1, &label));
    CHECK(!draft_label(cut, 2, &label));
}

TEST(only_cleanly_attributed_samples_may_update_a_measured_bucket) {
    DraftCostSample s;
    s.total_ms = 10.0;
    CHECK(!draft_sample_attributable(s));   /* not marked valid */

    s.flags = kDraftSampleValid;
    CHECK(draft_sample_attributable(s));
    s.flags |= kDraftSampleAhead;           /* flagged for the model, not disqualified */
    CHECK(draft_sample_attributable(s));

    for (uint16_t flag : { kDraftSampleWarmup, kDraftSampleRecord, kDraftSampleAudit,
                           kDraftSampleAdmission, kDraftSamplePlacement }) {
        DraftCostSample dirty = s;
        dirty.flags |= flag;
        CHECK(!draft_sample_attributable(dirty));
    }

    DraftCostSample untimed;
    untimed.flags = kDraftSampleValid;
    CHECK(!draft_sample_attributable(untimed));   /* no positive bracket */
}

TEST(a_missing_profile_falls_back_visibly_instead_of_inventing_a_price) {
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2);

    draft_fake::Predictor predictor;
    draft_fake::CostModel model;   /* nothing measured */
    draft_fake::Policy policy;
    policy.mode = draft_fake::Policy::Mode::Cost;
    policy.legacy_k = 3;
    draft_fake::Observer observer;

    DraftHooks hooks;
    hooks.predictor = { &predictor, draft_fake::predict };
    hooks.costs     = { &model, draft_fake::estimate };
    hooks.policy    = { &policy, draft_fake::choose };
    hooks.observer  = { &observer, draft_fake::observe };

    CHECK_EQ(hooks.predictor.predict(hooks.predictor.ctx, f.view, &f.pred), RAD_OK);
    CHECK_EQ(predictor.calls, 1);

    CHECK_EQ(hooks.policy.choose(hooks.policy.ctx, f.view, f.pred, hooks.costs, &f.plan), RAD_OK);
    CHECK_EQ(model.calls, 1);
    CHECK(f.plan.source == DraftPlanSource::Fallback);
    CHECK(f.plan.fallback == DraftFallback::NoProfile);
    CHECK_EQ(f.plan.max_k, 3);   /* the fixed baseline, not a price */
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    /* With a measured baseline the same seam reports a measured cost plan. */
    model.measured = true;
    model.ms[3] = 12.0;
    CHECK_EQ(hooks.policy.choose(hooks.policy.ctx, f.view, f.pred, hooks.costs, &f.plan), RAD_OK);
    CHECK(f.plan.source == DraftPlanSource::Cost);
    CHECK(f.plan.fallback == DraftFallback::None);
    CHECK_EQ(model.calls, 2);
}

TEST(observing_a_completed_cycle_does_not_write_into_its_plan) {
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2);

    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    int32_t k_before[draft_fake::Fixture::kMaxSeq];
    int32_t rows_before[RAD_SCHED_MAX_SPEC + 1];
    std::memcpy(k_before, f.plan.k, sizeof k_before);
    std::memcpy(rows_before, f.plan.rows, sizeof rows_before);

    DraftSeqObservation rows[2];
    rows[0].accepted = 2; rows[0].checked = 2; rows[0].verified = 3;
    rows[1].accepted = 1; rows[1].checked = 2; rows[1].verified = 3;

    DraftObservation obs;
    obs.cycle_id = f.view.cycle_id;
    obs.policy_version = f.plan.policy_version;
    obs.n_seq = 2;
    obs.cap = 2;
    obs.plan = &f.plan;
    obs.req_id = f.req_id;
    obs.slot = f.slot;
    obs.slot_gen = f.slot_gen;
    obs.seq = rows;
    obs.complete = true;

    DraftCostSample sample;
    sample.cycle_id = f.view.cycle_id;
    sample.total_ms = 12.0;
    sample.flags = kDraftSampleValid;

    draft_fake::Observer observer;
    CHECK_EQ(draft_fake::observe(&observer, obs, sample), RAD_OK);
    CHECK_EQ(observer.calls, 1);
    CHECK_EQ(observer.accepted, 3);
    CHECK_EQ(observer.total_ms, 12.0);

    for (int i = 0; i < draft_fake::Fixture::kMaxSeq; ++i) CHECK_EQ(f.plan.k[i], k_before[i]);
    for (int j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) CHECK_EQ(f.plan.rows[j], rows_before[j]);

    /* The observation must name this plan's cycle and identities, not just any complete one. */
    CHECK(draft_plan_identifies(f.plan, obs));
    DraftObservation next_cycle = obs;
    next_cycle.cycle_id = obs.cycle_id + 1;
    CHECK(!draft_plan_identifies(f.plan, next_cycle));
    CHECK_EQ(draft_fake::observe(&observer, next_cycle, sample), RAD_E_INVAL);

    uint64_t recycled[2] = { f.req_id[0], f.req_id[1] + 1 };   /* slot reused, new generation */
    DraftObservation other_req = obs;
    other_req.req_id = recycled;
    CHECK(!draft_plan_identifies(f.plan, other_req));
    CHECK_EQ(draft_fake::observe(&observer, other_req, sample), RAD_E_INVAL);
    CHECK_EQ(observer.calls, 1);

    DraftObservation unnamed = obs;
    unnamed.plan = nullptr;
    CHECK(!draft_plan_identifies(f.plan, unnamed));
    CHECK_EQ(draft_fake::observe(&observer, unnamed, sample), RAD_E_INVAL);

    /* A cycle the caller has not declared complete is refused, not observed. */
    obs.complete = false;
    CHECK_EQ(draft_fake::observe(&observer, obs, sample), RAD_E_INVAL);
    CHECK_EQ(observer.calls, 1);
}

TEST(a_cost_plan_without_a_cost_model_is_unsupported) {
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.mode = draft_fake::Policy::Mode::Cost;

    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan),
             RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()
