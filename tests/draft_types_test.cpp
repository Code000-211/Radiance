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

    /* A drafting plan names its filler; the fallback still may not take the request over. At
     * depth 0 it also verifies the window it drafts, so verify_k is the new max_k. */
    f.plan.k[0] = 2;
    f.plan.max_k = 2;
    f.plan.verify_k = 2;
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
    s.replay = DraftReplay::None;
    s.shape.replay = DraftReplay::None;   /* the run established the state the key names */
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

    /* Replay state is part of the bucket key, so an unevidenced state fills nothing -- and the key
     * must be the state the run established, not the one the plan declared. */
    DraftCostSample unknown = s;
    unknown.replay = DraftReplay::Unknown;
    unknown.shape.replay = DraftReplay::None;
    CHECK(draft_sample_storable(unknown));       /* still a bracket */
    CHECK(!draft_sample_attributable(unknown));  /* but not a measurement of any play bucket */
    DraftCostSample declared = s;
    declared.replay = DraftReplay::Play;
    declared.shape.replay = DraftReplay::None;   /* key from the plan, run says playback */
    CHECK(!draft_sample_attributable(declared));
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
    rows[0].label_origin = DraftOrigin::Head; rows[0].label_generated = 3;
    rows[1].generated = 3; rows[1].published = 3; rows[1].verified = 3;
    rows[1].accepted = 1; rows[1].checked = 2; rows[1].raw_match = 1;
    rows[1].end = DraftSeqEnd::Rejected; rows[1].origin = DraftOrigin::Head;
    rows[1].label_origin = DraftOrigin::Head; rows[1].label_generated = 3;

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
    sample.replay = DraftReplay::None;
    sample.shape.replay = DraftReplay::None;
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
    rows[0].label_origin = DraftOrigin::Head; rows[0].label_generated = 2;
    rows[1].generated = 1; rows[1].published = 1; rows[1].verified = 1;
    rows[1].accepted = 0; rows[1].checked = 1; rows[1].raw_match = 0;
    rows[1].end = DraftSeqEnd::Rejected; rows[1].origin = DraftOrigin::Head;
    rows[1].label_origin = DraftOrigin::Head; rows[1].label_generated = 1;

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
    row.label_origin = DraftOrigin::Head; row.label_generated = 2;
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

    /* Same requests, next cycle: the set carries the previous cycle's plan parameters. */
    f.seq[0].req_id = 1;
    f.view.cycle_id += 1;
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
        r.label_origin = DraftOrigin::Head; r.label_generated = 2;
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
    rows[1].label_origin = DraftOrigin::None; rows[1].label_generated = 0;
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

TEST(an_identifying_plan_is_bounded_by_the_rows_it_was_sized_for) {
    /* ASan repro: cap 1, n_seq 2. The plan's own capacity is checked before the identity loop, or
     * the check that is meant to stop a stale row is the read past the caller's storage. The
     * identity arrays here really are one row, so a regression is an ASan failure, not a diff. */
    draft_fake::Fixture f;
    f.add(0, 7);
    f.add(1, 8);
    draft_fake::Policy policy;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);

    uint64_t req[1] = { 7 };
    int32_t slot[1] = { 0 };
    uint32_t gen[1] = { 1 };
    f.plan.cap = 1;   /* the storage is one row; the plan claims two */
    f.plan.req_id = req;
    f.plan.slot = slot;
    f.plan.slot_gen = gen;

    uint64_t o_req[2] = { 7, 8 };
    int32_t o_slot[2] = { 0, 1 };
    uint32_t o_gen[2] = { 1, 1 };
    DraftSeqObservation rows[2] = {};
    DraftObservation obs;
    obs.cycle_id = f.plan.cycle_id;
    obs.policy_version = f.plan.policy_version;
    obs.n_seq = 2;
    obs.cap = 2;
    obs.plan = &f.plan;
    obs.req_id = o_req;
    obs.slot = o_slot;
    obs.slot_gen = o_gen;
    obs.seq = rows;
    obs.complete = true;

    CHECK(!draft_plan_identifies(f.plan, obs));
    CHECK(!draft_observation_deliverable(obs));
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
}

TEST(the_delivery_and_attribution_gates_refuse_a_plan_that_fails_its_own_check) {
    /* A plan whose max_k escapes its ceiling is not a cycle to learn from, however cleanly the
     * observation names it. Identity says "these rows are this plan's"; the plan's own gate says
     * "this plan could have been issued". Both are needed. */
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.legacy_k = 3;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);

    DraftSeqObservation row;
    row.generated = 3; row.published = 3; row.verified = 3;
    row.accepted = 2; row.checked = 3; row.raw_match = 2;
    row.end = DraftSeqEnd::Rejected; row.origin = DraftOrigin::Head;
    row.label_origin = DraftOrigin::Head; row.label_generated = 3;
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

    DraftCostSample sample;
    sample.cycle_id = f.plan.cycle_id;
    sample.policy_version = f.plan.policy_version;
    sample.total_ms = 4.0;
    sample.flags = kDraftSampleValid;
    sample.replay = DraftReplay::None;
    sample.shape.replay = DraftReplay::None;

    CHECK(draft_observation_deliverable(obs));
    CHECK(draft_sample_attributable(obs, sample));

    f.plan.k_ceiling = 1;   /* max_k 3 is now outside the ceiling it declared */
    CHECK(draft_plan_identifies(f.plan, obs));
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
    CHECK(!draft_observation_deliverable(obs));
    CHECK(!draft_sample_attributable(obs, sample));

    /* An unfinished cycle has no bracket to fold into a bucket either. */
    f.plan.k_ceiling = 6;
    obs.complete = false;
    CHECK(!draft_sample_attributable(obs, sample));
}

TEST(an_origin_is_bounded_by_the_draft_budget_it_claims) {
    /* `k[]` is the budget for every drafter, so the amount is part of the fit, not only the
     * type: a head row that reports more proposals than its row was allowed is not this plan's
     * work, and neither is one that claims the head while reporting none. */
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.legacy_k = 1;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);
    CHECK_EQ(f.plan.k[0], 1);

    DraftSeqObservation row;
    row.generated = 1; row.published = 1; row.verified = 1;
    row.accepted = 1; row.checked = 1; row.raw_match = 1;
    row.end = DraftSeqEnd::AllAccepted; row.origin = DraftOrigin::Head;
    CHECK(draft_seq_observation_valid(row));
    CHECK(draft_origin_fits_plan(f.plan, 0, row));

    row.generated = 2;   /* two proposals on a budget of one */
    CHECK(draft_seq_observation_valid(row));
    CHECK(!draft_origin_fits_plan(f.plan, 0, row));

    row.generated = 0; row.published = 0; row.verified = 0;
    row.accepted = 0; row.checked = 0; row.raw_match = 0;
    CHECK(draft_seq_observation_valid(row));
    CHECK(!draft_origin_fits_plan(f.plan, 0, row));   /* still Head: a row that produced nothing is None */
    row.origin = DraftOrigin::None;
    CHECK(draft_origin_fits_plan(f.plan, 0, row));

    /* A count past the verify band is not a count, whatever the budget says. */
    row.origin = DraftOrigin::Head;
    row.generated = RAD_SCHED_MAX_SPEC + 1;
    row.published = RAD_SCHED_MAX_SPEC + 1;
    CHECK(!draft_seq_observation_valid(row));
}

TEST(the_labels_are_bounded_by_the_window_their_own_origin_verified) {
    /* The bound is per row and per provenance, not per union of the two candidate windows: at
     * depth > 0 the head's labels come from the frozen verify_k while the lookup fallback fills in
     * the step that verifies it, so a union would admit a lookup row at the frozen window and a
     * head row at the deepest row's K. */
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.legacy_k = 3;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);
    CHECK_EQ(f.plan.k[0], 3);

    uint64_t req[1] = { 1 };
    int32_t slot[1] = { 0 };
    uint32_t gen[1] = { 1 };
    DraftSeqObservation row;
    row.generated = 3; row.published = 3; row.verified = 3;
    row.accepted = 3; row.checked = 3; row.raw_match = 3;
    row.end = DraftSeqEnd::AllAccepted; row.origin = DraftOrigin::Head;
    row.label_origin = DraftOrigin::Head; row.label_generated = 3;
    DraftObservation obs;
    obs.cycle_id = f.plan.cycle_id;
    obs.policy_version = f.plan.policy_version;
    obs.n_seq = 1; obs.cap = 1;
    obs.plan = &f.plan;
    obs.req_id = req; obs.slot = slot; obs.slot_gen = gen;
    obs.seq = &row; obs.complete = true;

    /* Depth 0: the plan drafts and verifies the same K, so that is the head's window. */
    CHECK(draft_observation_deliverable(obs));
    row.verified = 4;
    CHECK(!draft_observation_deliverable(obs));   /* one past the window it verified */
    row.verified = 3;
    row.label_generated = 4;
    CHECK(!draft_observation_deliverable(obs));   /* labels from wider than that window */
    row.label_generated = 3;

    /* A row the verified window did not draft consumed no window at all, whatever its plan did. */
    row.label_origin = DraftOrigin::None; row.label_generated = 0;
    CHECK(!draft_observation_deliverable(obs));
    row.verified = 0; row.checked = 0; row.accepted = 0; row.raw_match = 0;
    row.end = DraftSeqEnd::Stopped;
    CHECK(draft_observation_deliverable(obs));

    /* Depth 1: the head's labels were verified by the paired step over the frozen verify_k, which
     * an ahead plan may set apart from what it drafts. */
    f.plan.pipeline_depth = 1;
    f.plan.verify_k = 2;
    row.label_origin = DraftOrigin::Head; row.label_generated = 2;
    row.verified = 2; row.checked = 2; row.accepted = 2; row.raw_match = 2;
    row.end = DraftSeqEnd::AllAccepted;
    CHECK(draft_observation_deliverable(obs));
    row.verified = 3; row.checked = 3; row.accepted = 3; row.raw_match = 3;
    CHECK(!draft_observation_deliverable(obs));   /* past the frozen window, not past k[i] */

    /* A deeper frozen window is legal ahead, and the head row follows it -- but no further. */
    f.plan.verify_k = 5;
    CHECK(draft_observation_deliverable(obs));
    row.label_generated = 6;
    CHECK(!draft_observation_deliverable(obs));

    /* The lookup fallback is not published ahead: it fills in the step that verifies it, so its
     * window stays k[i] even when the frozen verify_k is deeper. */
    f.plan.fill[0] = DraftDrafter::Ngram;
    row.origin = DraftOrigin::Lookup; row.label_origin = DraftOrigin::Lookup;
    row.label_generated = 3;
    CHECK(draft_observation_deliverable(obs));
    row.verified = 4;
    CHECK(!draft_observation_deliverable(obs));   /* the frozen window is not the lookup's */
    f.plan.pipeline_depth = 0;
    f.plan.verify_k = 3;
    row.verified = 3;
    CHECK(draft_observation_deliverable(obs));   /* depth 0: k[i] is the window either way */
}

TEST(the_end_of_the_checked_span_and_the_raw_match_are_one_statement) {
    DraftSeqObservation o;
    o.generated = 3; o.published = 3; o.verified = 3;
    o.accepted = 1; o.checked = 2; o.raw_match = 1;
    o.end = DraftSeqEnd::Rejected;
    CHECK(draft_seq_observation_valid(o));
    float label = -1.0f;
    REQUIRE(draft_label(o, 1, &label));
    CHECK_EQ(label, 0.0f);

    /* The reviewer's repro: a rejection that says the draft matched. Accepting it would write a 0
     * for a token the drafter got right; the classification and the tally must agree. */
    DraftSeqObservation wrong_reject = o;
    wrong_reject.raw_match = 2;
    CHECK(!draft_seq_observation_valid(wrong_reject));

    /* A matched stop is the one position past the prefix, and the censor -- not the tally --
     * decides where commitment stopped. `DraftController::observe` takes the committed count from
     * the card, so a budget or cancellation censor can stop it below the compared span: a taller
     * raw match is a real row and must keep its labels, and it still gets none at `accepted`. */
    DraftSeqObservation stopped;
    stopped.generated = 3; stopped.published = 3; stopped.verified = 3;
    stopped.accepted = 1; stopped.checked = 1; stopped.raw_match = 1;
    stopped.end = DraftSeqEnd::Stopped;
    CHECK(draft_seq_observation_valid(stopped));   /* censored before anything was compared */
    stopped.raw_match = 2;
    CHECK(draft_seq_observation_valid(stopped));   /* the EOG the target chose */
    CHECK(!draft_label(stopped, 1, nullptr));       /* matched, not committed: no label */
    stopped.raw_match = 3;
    CHECK(draft_seq_observation_valid(stopped));   /* a censor stopped commitment below the span */
    CHECK(!draft_label(stopped, 1, nullptr));
    CHECK(draft_label(stopped, 0, nullptr));       /* the committed prefix keeps its 1 */

    /* The tally still may not invent matches the drafter never produced. */
    stopped.raw_match = 4;
    CHECK(!draft_seq_observation_valid(stopped));

    /* The tally is bounded by the window the paired step computed, not by the drafter's own count:
     * shrinking `verified` below it must refuse the row even though `generated` still covers it. */
    stopped.verified = 2;
    CHECK(!draft_seq_observation_valid(stopped));
    stopped.verified = 3;

    /* raw_match counts matches among what was compared, not an independent tally. */
    DraftSeqObservation over;
    over.generated = 1; over.published = 1; over.verified = 1;
    over.accepted = 1; over.checked = 1; over.raw_match = 64;
    over.end = DraftSeqEnd::AllAccepted;
    CHECK(!draft_seq_observation_valid(over));

    /* The two windows are separate, in both directions. A verify window above this plan's draft
     * window computes positions this plan never published, and one below it stops short of them;
     * chaining `verified` to `published` would refuse the first and misread the second. */
    DraftSeqObservation wider;   /* verify-K above draft-K */
    wider.generated = 2; wider.published = 2; wider.verified = 4;
    wider.accepted = 2; wider.checked = 2; wider.raw_match = 2;
    wider.end = DraftSeqEnd::AllAccepted;
    CHECK(draft_seq_observation_valid(wider));
    wider.raw_match = 3;         /* a match past the drafts that were published */
    CHECK(!draft_seq_observation_valid(wider));
    wider.raw_match = 2;

    DraftSeqObservation narrower;   /* verify-K below draft-K */
    narrower.generated = 4; narrower.published = 4; narrower.verified = 2;
    narrower.accepted = 2; narrower.checked = 2; narrower.raw_match = 2;
    narrower.end = DraftSeqEnd::AllAccepted;
    CHECK(draft_seq_observation_valid(narrower));
    narrower.raw_match = 3;         /* more matches than the verify window compared */
    CHECK(!draft_seq_observation_valid(narrower));

    /* A current K0 plan in front of the verify window a previously published plan produced. Nothing
     * is drafted now, yet the step commits drafts from that window, so `accepted` and the tally are
     * positive; `accepted <= published` would refuse the row and lose a measured cycle. */
    DraftSeqObservation k0_after;
    k0_after.generated = 0; k0_after.published = 0; k0_after.verified = 4;
    k0_after.accepted = 3; k0_after.checked = 3; k0_after.raw_match = 3;
    k0_after.end = DraftSeqEnd::AllAccepted;
    CHECK(draft_seq_observation_valid(k0_after));
    k0_after.end = DraftSeqEnd::Stopped;          /* a censor stopped commitment below the span */
    k0_after.raw_match = 4;
    CHECK(draft_seq_observation_valid(k0_after));
    k0_after.raw_match = 5;                      /* past the window the paired verify computed */
    CHECK(!draft_seq_observation_valid(k0_after));
    k0_after.raw_match = 2;                      /* fewer matches than were committed */
    CHECK(!draft_seq_observation_valid(k0_after));

    /* K0 with the paired verify still pending: nothing drafted, nothing compared, nothing
     * committed. A zero verified is a measured zero here, and a tally of one would be a match the
     * target never had a draft to compare. */
    DraftSeqObservation k0;
    k0.generated = 0; k0.published = 0; k0.verified = 0;
    k0.accepted = 0; k0.checked = 0; k0.raw_match = 0;
    k0.end = DraftSeqEnd::AllAccepted;
    CHECK(draft_seq_observation_valid(k0));
    k0.raw_match = 1;
    CHECK(!draft_seq_observation_valid(k0));
}

TEST(the_verify_distribution_is_the_plans_own_when_nothing_is_issued_ahead) {
    /* Nothing issued ahead: the plan drafts and verifies the same window, so its verify_k is its
     * max_k and the shape it was priced on is the plan's own rows. */
    draft_fake::Fixture f;
    f.add(0, 1);
    f.add(1, 2, 2);   /* two tokens left: one draft plus the bonus */
    draft_fake::Policy policy;
    policy.legacy_k = 2;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);
    CHECK_EQ(f.plan.pipeline_depth, 0);
    CHECK_EQ(f.plan.max_k, 2);
    CHECK_EQ(f.plan.verify_k, 2);
    CHECK_EQ(f.plan.rows[1], 2);
    CHECK_EQ(f.plan.rows[2], 1);

    DraftVerifyShape ok;
    ok.n_rows = 2;
    ok.verify_k = 2;
    ok.rows[1] = 2; ok.rows[2] = 1;
    ok.verify_tokens = 2 + 2 + 1;
    ok.draft_k = 2;
    ok.draft_rows[1] = 2; ok.draft_rows[2] = 1;
    CHECK(draft_shape_priceable(ok));
    CHECK(draft_shape_matches_plan(ok, f.plan));

    /* The reviewer's repro: three empty verify rounds are not a measured zero, they are a missing
     * shape. The token count is consistent, so only the deepest-round rule refuses it. */
    DraftVerifyShape all_zero = ok;
    all_zero.verify_k = 3;
    all_zero.verify_tokens = 2;
    CHECK(!draft_shape_priceable(all_zero));

    /* A verify distribution from another K is priced, but it is not this plan's verify pass. */
    DraftVerifyShape other = ok;
    other.verify_k = 1;
    other.rows[2] = 0;
    other.verify_tokens = 2 + 2;
    CHECK(draft_shape_priceable(other));
    CHECK(!draft_shape_matches_plan(other, f.plan));

    /* The plan gate holds the same binding: no plan verifies another K than it drafts here. */
    f.plan.verify_k = 1;
    const std::string log = stderr_of([&] { CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL); });
    CHECK(has(log, "verifies"));
    f.plan.verify_k = 2;
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(an_ahead_plan_may_verify_a_different_k_than_it_drafts) {
    /* Issued ahead, the verify pass belongs to the cycle the paired step verifies: verifying a
     * different K than this plan drafts is the point, and both the plan gate and the shape check
     * must leave it alone. */
    draft_fake::Fixture f;
    f.view.pipeline_depth = 1;
    f.add(0, 1);
    f.add(1, 2, 2);
    draft_fake::Policy policy;
    policy.legacy_k = 2;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);
    CHECK_EQ(f.plan.pipeline_depth, 1);
    CHECK_EQ(f.plan.max_k, 2);

    f.plan.verify_k = 1;   /* the previous cycle verified one draft per row, not two */
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
    f.plan.verify_k = 3;   /* or one deeper than this plan drafts */
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    DraftVerifyShape s;
    s.n_rows = 2;
    s.verify_k = 1;
    s.rows[1] = 2;
    s.verify_tokens = 2 + 2;
    s.draft_k = 2;
    s.draft_rows[1] = 2; s.draft_rows[2] = 1;
    s.pipeline_depth = 1;
    CHECK(s.verify_k != s.draft_k);
    CHECK(draft_shape_priceable(s));
    CHECK(draft_shape_matches_plan(s, f.plan));

    /* The other direction: a verify window deeper than this plan's own draft distribution. */
    DraftVerifyShape deep = s;
    deep.verify_k = 3;
    deep.rows[2] = 2; deep.rows[3] = 1;
    deep.verify_tokens = 2 + 2 + 2 + 1;
    CHECK(draft_shape_priceable(deep));
    CHECK(draft_shape_matches_plan(deep, f.plan));
}

TEST(a_current_k0_plan_still_carries_the_pending_verify_window) {
    /* K 0 this cycle and the verify window still outstanding: the plan drafts nothing, the paired
     * step verifies what a previously published plan produced. Both the gate and the shape keep
     * the two apart, so neither zeroing verify_k to fit max_k nor refusing the plan is needed. */
    draft_fake::Fixture f;
    f.view.pipeline_depth = 1;
    f.add(0, 1);
    draft_fake::Policy policy;
    policy.legacy_k = 0;
    REQUIRE(draft_fake::choose(&policy, f.view, f.pred, DraftCostModel{}, &f.plan) == RAD_OK);
    CHECK_EQ(f.plan.pipeline_depth, 1);
    CHECK_EQ(f.plan.max_k, 0);
    CHECK_EQ(f.plan.k[0], 0);
    CHECK(f.plan.fill[0] == DraftDrafter::None);
    f.plan.verify_k = 3;   /* the previous cycle published three drafts for this row */
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);

    DraftVerifyShape s;
    s.n_rows = 1;
    s.verify_k = 3;
    s.rows[1] = 1; s.rows[2] = 1; s.rows[3] = 1;
    s.verify_tokens = 1 + 1 + 1 + 1;
    s.draft_k = 0;
    s.pipeline_depth = 1;
    CHECK(draft_shape_priceable(s));
    CHECK(draft_shape_matches_plan(s, f.plan));

    /* At depth 0 nothing is pending and the plan verifies the window it drafts, so a K0 plan
     * naming a verify window is not this gate's to accept. */
    f.plan.pipeline_depth = 0;
    CHECK_EQ(draft_plan_valid(f.plan), RAD_E_INVAL);
}

RAD_TEST_MAIN()
