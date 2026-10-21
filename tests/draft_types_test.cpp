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
#include <limits>
#include <string>

using namespace rad;

static_assert(RAD_SCHED_MAX_SPEC == 64, "the verify kernel band is the array bound");
static_assert(RAD_DRAFT_CTX_CLASSES == 4, "the profile's context classes");
static_assert(RAD_DRAFT_PROFILE_VERSION == 2, "the key grew: a version-1 profile is refused");

TEST(plan_rows_count_the_requests_that_reach_each_round) {
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 100, 64);   /* K 6 */
    f.add(1, 101, 2);    /* remaining 2: the bonus token leaves room for one draft */
    f.add(2, 102, 0);     /* K 0 */
    f.add(3, 103, 4);    /* K 3: remaining 4 leaves room for three drafts plus the bonus */

    draft_fake::Policy policy;
    policy.legacy_k = 6;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    CHECK_EQ(f.plan.max_k, 6);
    CHECK_EQ(f.plan.k[0], 6);
    CHECK_EQ(f.plan.k[1], 1);
    CHECK_EQ(f.plan.k[2], 0);
    CHECK_EQ(f.plan.k[3], 3);
    /* N_j: 6, 1 and 3 reach round 1; 6 and 3 reach 2; 6 and 3 reach 3; only 6 reaches 4..6. */
    const int32_t want[7] = { 0, 3, 2, 2, 1, 1, 1 };
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

TEST(plan_rows_refuses_more_requests_than_the_caller_sized) {
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    const int32_t saved = f.plan.n_seq;
    f.plan.n_seq = f.plan.cap + 1;   /* would read k[] past the caller's array */
    const std::string log = stderr_of([&] { CHECK_EQ(draft_plan_rows(f.plan), RAD_E_INVAL); });
    CHECK(has(log, "exceeds"));
    CHECK_EQ(f.plan.rows[1], 1);     /* nothing was counted from the out-of-range request */
    f.plan.n_seq = saved;
}

TEST(plan_validation_refuses_a_max_k_above_the_deepest_k) {
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2);
    draft_fake::Policy policy;
    policy.legacy_k = 3;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    f.plan.k[0] = 2;
    f.plan.k[1] = 0;
    f.plan.fill[1] = DraftDrafter::None;   /* K 0 and nobody to fill it */
    f.plan.max_k = 5;
    /* draft_plan_rows fills rows[] for what k[] says and then refuses max_k != deepest. Those
     * rows are consistent, so only the max_k rule is left to catch the plan. */
    const std::string rows_log = stderr_of([&] { CHECK_EQ(draft_plan_rows(f.plan), RAD_E_INVAL); });
    CHECK(has(rows_log, "deepest"));
    CHECK_EQ(f.plan.rows[2], 1);
    CHECK_EQ(f.plan.rows[5], 0);
    const std::string log = stderr_of([&] { CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL); });
    CHECK(has(log, "no request drafts that far"));
}

TEST(labels_stop_at_the_first_rejection_and_at_a_censor) {
    DraftSeqObservation o;
    o.generated = 4;
    o.published = 4;
    o.verified = 4;   /* the target computed more than it may label */
    o.accepted = 1;
    o.checked = 2;    /* accepted prefix plus the first rejection */
    o.raw_match = 1;
    o.end = DraftSeqEnd::Rejected;

    float label = -1.0f;
    REQUIRE(draft_label(o, 0, &label));
    CHECK_EQ(label, 1.0f);
    REQUIRE(draft_label(o, 1, &label));
    CHECK_EQ(label, 0.0f);
    CHECK(!draft_label(o, 2, &label));
    CHECK(!draft_label(o, 3, &label));
    CHECK(draft_seq_observation_valid(o));

    DraftSeqObservation all;
    all.generated = 3;
    all.published = 3;
    all.verified = 3;
    all.accepted = 3;
    all.checked = 3;   /* every draft matched: no zero label */
    all.raw_match = 3;
    all.end = DraftSeqEnd::AllAccepted;
    CHECK(draft_label(all, 2, &label));
    CHECK_EQ(label, 1.0f);
    CHECK(!draft_label(all, 3, &label));
    CHECK(draft_seq_observation_valid(all));

    /* The reviewer's repro: checked covers four positions but the first rejection is at 1. The
     * mask still stops at the rejection; the row itself is not a valid observation. */
    DraftSeqObservation long_tail;
    long_tail.generated = 4;
    long_tail.published = 4;
    long_tail.verified = 4;
    long_tail.accepted = 1;
    long_tail.checked = 4;
    long_tail.raw_match = 1;
    long_tail.end = DraftSeqEnd::Rejected;
    REQUIRE(draft_label(long_tail, 0, &label));
    CHECK_EQ(label, 1.0f);
    REQUIRE(draft_label(long_tail, 1, &label));
    CHECK_EQ(label, 0.0f);
    CHECK(!draft_label(long_tail, 2, &label));
    CHECK(!draft_label(long_tail, 3, &label));
    CHECK(!draft_seq_observation_valid(long_tail));

    /* A censor after the committed prefix that compared nothing carries no label. */
    DraftSeqObservation cut;
    cut.generated = 1;
    cut.published = 1;
    cut.verified = 1;
    cut.accepted = 1;
    cut.checked = 1;
    cut.raw_match = 1;
    cut.end = DraftSeqEnd::Stopped;
    cut.censor = DraftCensor::Eog;
    CHECK(draft_label(cut, 0, &label));
    CHECK_EQ(label, 1.0f);
    CHECK(!draft_label(cut, 1, &label));
    CHECK(!draft_label(cut, 2, &label));
    CHECK(draft_seq_observation_valid(cut));
}

TEST(a_rejection_measured_before_a_censor_keeps_its_zero) {
    /* B's repro: accepted = 1, checked = 2, censor = Eog. Position 1 was compared and lost; the
     * censor is the reason the cycle ended, not a reason to erase the measured rejection. */
    DraftSeqObservation o;
    o.generated = 2;
    o.published = 2;
    o.verified = 2;
    o.accepted = 1;
    o.checked = 2;
    o.raw_match = 1;
    o.end = DraftSeqEnd::Rejected;
    o.censor = DraftCensor::Eog;
    float label = -1.0f;
    REQUIRE(draft_label(o, 1, &label));
    CHECK_EQ(label, 0.0f);
    CHECK(draft_seq_observation_valid(o));
    CHECK(draft_label(o, 0, &label));
    CHECK_EQ(label, 1.0f);
}

TEST(a_correctly_hit_but_uncommitted_eog_is_not_a_reject) {
    /* F's case: the draft proposed the EOG the target chose (raw_match covers it), but the token
     * is not committed, so `accepted` stays at the committed prefix. The position was compared
     * and matched; `Stopped` says so, and the mask must not turn it into a 0. */
    DraftSeqObservation o;
    o.generated = 2;
    o.published = 2;
    o.verified = 2;
    o.accepted = 1;
    o.checked = 1;
    o.raw_match = 2;
    o.end = DraftSeqEnd::Stopped;
    o.censor = DraftCensor::Eog;
    CHECK(draft_seq_observation_valid(o));
    float label = -1.0f;
    REQUIRE(draft_label(o, 0, &label));
    CHECK_EQ(label, 1.0f);
    CHECK(!draft_label(o, 1, &label));   /* compared, matched, not committed: no label */
    CHECK(o.raw_match == o.accepted + 1); /* the binding assertion F can make */

    /* The same counts without a classification get no label at the cut, never a guessed reject. */
    DraftSeqObservation unclassified = o;
    unclassified.end = DraftSeqEnd::Unset;
    CHECK(!draft_seq_observation_valid(unclassified));
    CHECK(!draft_label(unclassified, 1, &label));
    REQUIRE(draft_label(unclassified, 0, &label));
    CHECK_EQ(label, 1.0f);
}

TEST(only_cleanly_attributed_samples_may_update_a_measured_bucket) {
    DraftCostSample s;
    s.cycle_id = 7;
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
    untimed.cycle_id = 7;
    untimed.flags = kDraftSampleValid;
    CHECK(!draft_sample_attributable(untimed));   /* no positive bracket */

    /* A sample that names no cycle, or does not name a duration, is not attributable. */
    DraftCostSample unnamed = s;
    unnamed.cycle_id = 0;
    CHECK(!draft_sample_attributable(unnamed));
    DraftCostSample infinite = s;
    infinite.total_ms = std::numeric_limits<double>::infinity();
    CHECK(!draft_sample_attributable(infinite));
    CHECK(!draft_sample_storable(infinite));

    /* Record and audit brackets have their own path: stored, never folded into a play bucket. */
    for (uint16_t flag : { kDraftSampleRecord, kDraftSampleAudit }) {
        DraftCostSample nonplay = s;
        nonplay.flags = kDraftSampleValid | flag;
        CHECK(draft_sample_storable(nonplay));
        CHECK(!draft_sample_attributable(nonplay));
    }
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
    rows[0].generated = 3; rows[0].published = 3; rows[0].verified = 3;
    rows[0].accepted = 2; rows[0].checked = 3; rows[0].raw_match = 2;
    rows[0].end = DraftSeqEnd::Rejected; rows[0].origin = DraftOrigin::Head;
    rows[1].generated = 3; rows[1].published = 3; rows[1].verified = 3;
    rows[1].accepted = 1; rows[1].checked = 2; rows[1].raw_match = 1;
    rows[1].end = DraftSeqEnd::Rejected; rows[1].origin = DraftOrigin::Head;

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
    sample.policy_version = f.plan.policy_version;
    sample.total_ms = 12.0;
    sample.flags = kDraftSampleValid;
    CHECK(draft_sample_attributable(obs, sample));

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

    DraftCostSample other_cycle_sample = sample;
    other_cycle_sample.cycle_id = obs.cycle_id + 1;
    CHECK(!draft_sample_attributable(obs, other_cycle_sample));   /* not this cycle's timing */

    /* A cycle the caller has not declared complete is refused, not observed. */
    obs.complete = false;
    CHECK_EQ(draft_fake::observe(&observer, obs, sample), RAD_E_INVAL);
    CHECK_EQ(observer.calls, 1);
}

TEST(the_draft_budget_reserves_the_bonus_token) {
    /* The verify step emits the target's bonus token after the accepted prefix, so `remaining`
     * tokens allow at most remaining - 1 drafts; with one token left, drafting would overrun. */
    CHECK_EQ(draft_budget_k(0, 6), 0);
    CHECK_EQ(draft_budget_k(1, 6), 0);
    CHECK_EQ(draft_budget_k(2, 6), 1);
    CHECK_EQ(draft_budget_k(3, 6), 2);
    CHECK_EQ(draft_budget_k(64, 6), 6);
    CHECK_EQ(draft_budget_k(64, 0), 0);

    draft_fake::Fixture f;
    f.add(0, 1, 2);   /* two tokens left, one draft plus the bonus */
    f.add(1, 2, 1);   /* one token left: the bonus only, no draft */
    draft_fake::Policy policy;
    policy.legacy_k = 3;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.k[0], 1);
    CHECK_EQ(f.plan.k[1], 0);
    CHECK(f.plan.why[0] == DraftFallback::Resources);
    CHECK(f.plan.why[1] == DraftFallback::Resources);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(a_cycle_stays_deliverable_across_preemption_recycle_and_resume) {
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    /* The ring owns the cycle's identity: copies, not pointers into the live slot state. */
    uint64_t p_req[2]; int32_t p_slot[2]; uint32_t p_gen[2];
    std::memcpy(p_req, f.req_id, sizeof p_req);
    std::memcpy(p_slot, f.slot, sizeof p_slot);
    std::memcpy(p_gen, f.slot_gen, sizeof p_gen);
    DraftPlan ring_plan = f.plan;
    ring_plan.req_id = p_req; ring_plan.slot = p_slot; ring_plan.slot_gen = p_gen;

    DraftSeqObservation rows[2];
    rows[0].generated = 2; rows[0].published = 2; rows[0].verified = 2;
    rows[0].accepted = 2; rows[0].checked = 2; rows[0].raw_match = 2;
    rows[0].end = DraftSeqEnd::AllAccepted; rows[0].origin = DraftOrigin::Head;
    rows[1].generated = 1; rows[1].published = 1; rows[1].verified = 1;
    rows[1].accepted = 0; rows[1].checked = 1; rows[1].raw_match = 0;
    rows[1].end = DraftSeqEnd::Rejected; rows[1].origin = DraftOrigin::Head;

    DraftObservation obs;
    obs.cycle_id = ring_plan.cycle_id;
    obs.policy_version = ring_plan.policy_version;
    obs.n_seq = 2;
    obs.cap = 2;
    obs.plan = &ring_plan;
    obs.req_id = p_req;
    obs.slot = p_slot;
    obs.slot_gen = p_gen;
    obs.seq = rows;
    obs.complete = true;
    CHECK(draft_observation_deliverable(obs));

    /* Preempt and recycle the live slot: another request generation now owns it. The cycle that
     * already ran is untouched in the ring and stays deliverable -- its identity is its own. */
    f.seq[0].eligibility = DraftEligibility::Preempted;
    f.seq[0].req_id = 99;
    f.seq[0].slot_gen = 2;
    f.req_id[0] = 99;
    f.slot_gen[0] = 2;

    DraftCostSample sample;
    sample.cycle_id = obs.cycle_id;
    sample.policy_version = obs.policy_version;
    sample.total_ms = 9.0;
    sample.flags = kDraftSampleValid;
    draft_fake::Observer observer;
    CHECK(draft_observation_deliverable(obs));
    CHECK_EQ(draft_fake::observe(&observer, obs, sample), RAD_OK);
    CHECK_EQ(observer.calls, 1);

    /* Resume: a new cycle and a new generation. The old rows never merge into it. */
    uint32_t resumed_gen[2] = { 2, 2 };
    DraftPlan resumed = ring_plan;
    resumed.cycle_id = ring_plan.cycle_id + 1;
    resumed.policy_version = ring_plan.policy_version + 1;
    resumed.slot_gen = resumed_gen;
    DraftObservation merged = obs;
    merged.plan = &resumed;
    merged.cycle_id = resumed.cycle_id;
    merged.policy_version = resumed.policy_version;
    CHECK(!draft_observation_deliverable(merged));
    CHECK_EQ(draft_fake::observe(&observer, merged, sample), RAD_E_INVAL);
    CHECK_EQ(observer.calls, 1);
}

TEST(a_plan_from_an_older_policy_stays_observable_while_it_is_alive) {
    draft_fake::Fixture f;
    f.view.policy_version = 3;
    f.add(0, 1);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    DraftSeqObservation row;
    row.generated = 2; row.published = 2; row.verified = 2;
    row.accepted = 1; row.checked = 2; row.raw_match = 1;
    row.end = DraftSeqEnd::Rejected; row.origin = DraftOrigin::Head;
    DraftObservation obs;
    obs.cycle_id = f.plan.cycle_id;
    obs.policy_version = f.plan.policy_version;
    obs.n_seq = 1;
    obs.cap = 1;
    obs.plan = &f.plan;
    obs.req_id = f.req_id;
    obs.slot = f.slot;
    obs.slot_gen = f.slot_gen;
    obs.seq = &row;
    obs.complete = true;

    /* The hooks move on to the next policy; the plan issued under 3 is still the one this
     * observation names, and the mask accepts it. */
    f.view.policy_version = 4;
    CHECK(draft_observation_deliverable(obs));

    /* An observation may not claim the current version while naming the older plan. */
    obs.policy_version = 4;
    CHECK(!draft_observation_deliverable(obs));
}

TEST(a_prediction_set_is_bound_to_the_view_it_describes) {
    draft_fake::Fixture f;
    f.view.policy_version = 7;
    f.add(0, 1);
    draft_fake::Predictor predictor;
    CHECK_EQ(draft_fake::predict(&predictor, f.view, &f.pred), RAD_OK);
    CHECK(draft_predictions_current(f.view, f.pred));
    CHECK_EQ(f.pred.version, 7u);

    /* The hooks advance: the set computed for version 7 is stale, not "current". */
    f.view.policy_version = 8;
    CHECK(!draft_predictions_current(f.view, f.pred));

    /* Same version, another request's row: still not this view's prediction. */
    f.view.policy_version = 7;
    f.seq[0].req_id = 2;
    CHECK(!draft_predictions_current(f.view, f.pred));
}

TEST(a_rows_origin_is_the_plans_permission_not_a_side_list) {
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2);
    draft_fake::Policy policy;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan), RAD_OK);

    DraftSeqObservation rows[2];
    for (DraftSeqObservation& r : rows) {
        r.generated = 2; r.published = 2; r.verified = 2;
        r.accepted = 1; r.checked = 2; r.raw_match = 1;
        r.end = DraftSeqEnd::Rejected; r.origin = DraftOrigin::Head;
    }
    CHECK(draft_origin_fits_plan(f.plan, 0, rows[0]));
    CHECK(draft_origin_fits_plan(f.plan, 1, rows[1]));

    /* A lookup row may not hide behind a head budget; the origin is the fact, fill[] the
     * permission. */
    rows[1].origin = DraftOrigin::Lookup;
    CHECK(!draft_origin_fits_plan(f.plan, 1, rows[1]));

    DraftObservation obs;
    obs.cycle_id = f.plan.cycle_id;
    obs.policy_version = f.plan.policy_version;
    obs.n_seq = 2;
    obs.cap = 2;
    obs.plan = &f.plan;
    obs.req_id = f.req_id;
    obs.slot = f.slot;
    obs.slot_gen = f.slot_gen;
    obs.seq = rows;
    obs.complete = true;
    CHECK(!draft_observation_deliverable(obs));   /* the leaked origin refuses the cycle */

    rows[1].origin = DraftOrigin::None;
    rows[1].generated = 0; rows[1].published = 0; rows[1].verified = 0;
    rows[1].accepted = 0; rows[1].checked = 0; rows[1].raw_match = 0;
    rows[1].end = DraftSeqEnd::AllAccepted;
    CHECK(draft_origin_fits_plan(f.plan, 1, rows[1]));
    CHECK(draft_observation_deliverable(obs));

    /* A plan that bars filling (K 0, fill None) refuses any row claiming proposals. */
    f.plan.k[1] = 0;
    f.plan.fill[1] = DraftDrafter::None;
    rows[1].origin = DraftOrigin::Head;
    rows[1].generated = 1;
    CHECK(!draft_origin_fits_plan(f.plan, 1, rows[1]));
}

TEST(the_shape_keys_verify_and_draft_distributions_apart_and_refuses_unknown) {
    draft_fake::Fixture f;
    f.view.pipeline_depth = 1;
    f.add(0, 1);
    f.add(1, 2);
    f.add(2, 3);
    draft_fake::Policy policy;
    policy.mode = draft_fake::Policy::Mode::Cost;
    draft_fake::CostModel model;
    model.measured = true;
    model.ms[3] = 10.0;
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, {&model, draft_fake::estimate}, &f.plan),
             RAD_OK);

    const DraftVerifyShape& s = model.last;
    CHECK(draft_shape_priceable(s));
    CHECK(draft_shape_matches_plan(s, f.plan));
    CHECK_EQ(s.draft_k, 3);
    CHECK_EQ(s.verify_k, 3);
    CHECK_EQ(s.pipeline_depth, 1);
    CHECK_EQ(s.n_rows, 3);
    CHECK_EQ(s.verify_tokens, 12);   /* three rows, each verifying 1 + 3 tokens */

    /* Same row count, same deepest round, same total draft rows -- a different shrinking
     * distribution. `rounds` plus a total would have called these the same key. */
    DraftVerifyShape a;
    a.n_rows = 3;
    a.verify_k = 3;
    a.verify_tokens = 10;
    a.draft_k = 3;
    a.rows[1] = 3; a.rows[2] = 2; a.rows[3] = 2;
    a.draft_rows[1] = 3; a.draft_rows[2] = 2; a.draft_rows[3] = 2;
    CHECK(draft_shape_priceable(a));
    DraftVerifyShape b = a;
    b.rows[2] = 3; b.rows[3] = 1;
    b.draft_rows[2] = 3; b.draft_rows[3] = 1;
    CHECK(draft_shape_priceable(b));
    CHECK_EQ(a.draft_rows[1] + a.draft_rows[2] + a.draft_rows[3],
             b.draft_rows[1] + b.draft_rows[2] + b.draft_rows[3]);
    CHECK(a.draft_rows[2] != b.draft_rows[2]);

    /* Unknown is not a measured zero: history, prefill, replay and the plan distribution each
     * have an explicit refusal. */
    for (uint32_t flag : { (uint32_t)kDraftShapeHistoryUnknown, (uint32_t)kDraftShapePrefillUnknown,
                           (uint32_t)kDraftShapeReplayUnknown, (uint32_t)kDraftShapeDraftUnknown }) {
        DraftVerifyShape u = s;
        u.unknown = flag;
        CHECK(!draft_shape_priceable(u));
    }

    /* A stale entry past the deepest round is not a key, and verify_tokens must match the rows. */
    DraftVerifyShape stale = s;
    stale.rows[s.verify_k + 1] = 1;
    CHECK(!draft_shape_priceable(stale));
    DraftVerifyShape wrong = s;
    wrong.verify_tokens += 1;
    CHECK(!draft_shape_priceable(wrong));

    /* A draft distribution assembled beside the plan is not the cycle's own. */
    DraftVerifyShape beside = s;
    beside.draft_rows[2] -= 1;
    CHECK(!draft_shape_matches_plan(beside, f.plan));
}

TEST(the_verify_shape_names_the_loaded_drafter_so_the_two_k0_arms_differ) {
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.mode = draft_fake::Policy::Mode::Cost;
    policy.legacy_k = 0;
    draft_fake::CostModel model;
    model.measured = true;
    model.ms[0] = 4.0;

    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, {&model, draft_fake::estimate}, &f.plan),
             RAD_OK);
    CHECK_EQ(f.plan.max_k, 0);
    CHECK(model.last.drafter == DraftDrafter::Serial);   /* K0 with the head loaded */

    f.view.k_ceiling = 0;   /* a process with no drafter at all */
    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, {&model, draft_fake::estimate}, &f.plan),
             RAD_OK);
    CHECK(model.last.drafter == DraftDrafter::None);
}

TEST(the_fake_prices_the_shape_the_plan_names) {
    draft_fake::Fixture f;
    f.add(0, 1);                                  /* eligible, drafts three */
    f.add(1, 2, 64, DraftEligibility::Grammar);   /* ineligible: a plain decode row */
    draft_fake::Policy policy;
    policy.mode = draft_fake::Policy::Mode::Cost;
    policy.legacy_k = 3;
    draft_fake::CostModel model;
    model.measured = true;
    model.ms[3] = 9.0;

    CHECK_EQ(draft_fake::choose(&policy, f.view, f.pred, {&model, draft_fake::estimate}, &f.plan),
             RAD_OK);
    /* The plan's verify_k is the shape's, not its max_k read twice. */
    CHECK_EQ(f.plan.max_k, 3);
    CHECK_EQ(f.plan.verify_k, 3);
    CHECK_EQ(model.last.verify_k, f.plan.verify_k);
    /* Both rows verify their committed token; only one of them drafts, so N_j and the token
     * count agree: 2 rows, 2 + 3 tokens. */
    CHECK_EQ(model.last.n_rows, 2);
    CHECK_EQ(model.last.verify_tokens, 2 + 3);
    CHECK_EQ(model.last.rows[1], 1);
    CHECK(model.last.drafter == DraftDrafter::Serial);
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
