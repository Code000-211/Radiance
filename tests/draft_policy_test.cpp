/* draft_policy_test.cpp -- the cost-based draft-depth policy (draft_policy.h) against A's and
 * B's fakes (draft_fakes.h).
 *
 * What is checked here is what strand C promises before a real model exists: a legacy plan keeps
 * its meaning, a cost plan prices the shape it will verify, an unpriced shape is a named fallback
 * rather than an invented price, K0 is a decision, a raise waits for its hold and its margin while
 * a reduction happens at once, budgets cut the plan visibly, and the controller freezes exactly
 * one plan per cycle, keyed by (req_id, slot, slot_gen).
 *
 * Not here: the scheduler's wiring (the n-gram gate, shrinking rounds, issue_ahead). Those need
 * the engine's fake program and belong to the integrated matrix (strand F).
 */
#include "rad_test.h"

#include "draft_fakes.h"
#include "sched/draft.h"
#include "sched/draft_policy.h"

#include <cmath>

using namespace rad;

namespace {

/* The issue's starting configuration. */
DraftPolicyConfig cost_config(int ceiling = 6) {
    DraftPolicyConfig c;
    c.mode = DraftPolicyMode::Cost;
    c.k_ceiling = ceiling;
    c.fallback_k = 3;
    return c;
}

/* A fixture that re-runs the fake predictor whenever a request is added, as the controller does
 * at the start of a cycle. Tests may overwrite single positions afterwards. */
struct Cycle {
    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    draft_fake::CostModel costs;
    DraftPlanner planner;

    explicit Cycle(int ceiling = 6, DraftPolicyConfig cfg = cost_config()) {
        f.view.k_ceiling = ceiling;
        cfg.k_ceiling = ceiling;   /* the controller copies max_depth in, so the test does too */
        CHECK_EQ(planner.configure(cfg), RAD_OK);
    }
    void add(int32_t i, uint64_t id, int32_t remaining = 64,
             DraftEligibility e = DraftEligibility::Eligible) {
        f.add(i, id, remaining, e);
        draft_fake::predict(&pred, f.view, &f.pred);
    }
    int choose() {
        return planner.choose(f.view, f.pred, DraftCostModel{ &costs, draft_fake::estimate },
                              &f.plan);
    }
    /* A model that has measured every window: `ms[k]` grows as ms0 + step_ms * k. */
    void measure(double ms0, double step_ms) {
        costs.measured = true;
        for (int k = 0; k <= RAD_SCHED_MAX_SPEC; ++k) costs.ms[k] = ms0 + step_ms * k;
    }
};

/* A model whose cost grows with the tokens the pass computes instead of only with the deepest
 * window: the shape in which a mixed plan can beat every homogeneous one. At pipeline_depth 0 the
 * verify pass computes this cycle's drafts, so verify_tokens alone carries the candidate. */
struct TokenPriced {
    double base = 1.0;
    double per_token = 0.1;
};
int token_estimate(void* ctx, const DraftVerifyShape& shape, DraftCostEstimate* out) {
    auto* m = static_cast<TokenPriced*>(ctx);
    *out = DraftCostEstimate{};
    out->ms = m->base + m->per_token * shape.verify_tokens;
    out->coverage = 8;
    out->source = DraftCostSource::Measured;
    out->valid = true;
    return RAD_OK;
}

/* The two C8 points core/rad_core.h documents -- depth 5: 43.0 ms / 23.3 tokens, depth 3:
 * 36.3 ms / 22.3 tokens -- and nothing else. A shape the document does not cover comes back
 * invalid, so the policy cannot lean on a price this test invented. */
struct C8Model {
    int calls[RAD_SCHED_MAX_SPEC + 1] = {};
};
int c8_estimate(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
    auto* m = static_cast<C8Model*>(ctx);
    *out = DraftCostEstimate{};
    if (s.verify_k >= 0 && s.verify_k <= RAD_SCHED_MAX_SPEC) ++m->calls[s.verify_k];
    /* A uniform depth-k C8 pass: 8 rows, 8 * (1 + k) tokens. */
    if (s.n_rows == 8 && s.verify_k == 3 && s.verify_tokens == 8 * 4)      out->ms = 36.3;
    else if (s.n_rows == 8 && s.verify_k == 5 && s.verify_tokens == 8 * 6) out->ms = 43.0;
    else return RAD_OK;   /* not one of the documented points: nothing to price */
    out->coverage = 1;
    out->source = DraftCostSource::Measured;
    out->valid = true;
    return RAD_OK;
}

/* Per-row acceptance: the controller re-runs the predictor at every open_cycle, so a table the
 * test changes between cycles is what the plan is chosen from. */
struct RowPredictor {
    float a[draft_fake::Fixture::kMaxSeq] = {};
};
int row_predict(void* ctx, const DraftBatchView& view, DraftPredictions* out) {
    auto* s = static_cast<RowPredictor*>(ctx);
    if (!out || !out->seq || !out->req_id || out->cap < view.n_seq) return RAD_E_INVAL;
    out->n_seq = view.n_seq;
    out->version = view.policy_version;
    for (int32_t i = 0; i < view.n_seq; ++i) {
        out->req_id[i] = view.seq[i].req_id;
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) {
            out->seq[i].a[j] = s->a[i];
            out->seq[i].unc[j] = 0.05f;
            out->seq[i].n[j] = 0;
        }
        out->seq[i].valid = 0;
    }
    return RAD_OK;
}

/* Records every shape the policy asks about. */
struct ShapeLog {
    static constexpr int kMax = 64;
    DraftVerifyShape shape[kMax];
    int n = 0;
};
int log_estimate(void* ctx, const DraftVerifyShape& shape, DraftCostEstimate* out) {
    auto* s = static_cast<ShapeLog*>(ctx);
    if (s->n < ShapeLog::kMax) s->shape[s->n++] = shape;
    *out = DraftCostEstimate{};
    out->ms = 1.0 + 0.2 * shape.verify_k;
    out->coverage = 8;
    out->source = DraftCostSource::Measured;
    out->valid = true;
    return RAD_OK;
}

}  /* namespace */

TEST(a_legacy_plan_drafts_the_configured_depth_and_keeps_the_ceiling) {
    Cycle c(4, DraftPolicyConfig{});   /* legacy, legacy_k 0 = the ceiling */
    c.add(0, 10, 64);
    c.add(1, 11, 2);
    c.add(2, 12, 64, DraftEligibility::Grammar);

    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.k[0], 4);
    CHECK_EQ(c.f.plan.k[1], 1);   /* remaining 2: one verified token, one draft */
    CHECK_EQ(c.f.plan.k[2], 0);
    CHECK_EQ(c.f.plan.max_k, 4);
    CHECK(c.f.plan.source == DraftPlanSource::Legacy);
    CHECK(c.f.plan.why[1] == DraftFallback::Resources);
    CHECK(c.f.plan.why[2] == DraftFallback::None);   /* its own state, not the policy's cut */
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
}

TEST(a_legacy_config_with_a_shallow_ceiling_is_not_refused_for_the_default_tiers) {
    DraftPolicyConfig cfg;
    cfg.k_ceiling = 3;   /* the default tiers reach 6 */
    DraftPlanner p;
    CHECK_EQ(p.configure(cfg), RAD_OK);
}

TEST(cost_without_a_model_or_a_predictor_is_refused_by_name) {
    DraftPlanner p;
    CHECK_EQ(p.configure(cost_config()), RAD_OK);
    draft_fake::Fixture f;
    f.add(0, 1);
    draft_fake::Predictor pred;
    draft_fake::predict(&pred, f.view, &f.pred);
    /* No estimate function: unsupported, never a fabricated plan. */
    CHECK_EQ(p.choose(f.view, f.pred, DraftCostModel{}, &f.plan), RAD_E_UNSUPPORTED);

    /* And the controller refuses the mode rather than mapping it to legacy. */
    DraftController::Params dp;
    dp.max_depth = 6;
    dp.max_seqs = 8;
    dp.policy = cost_config();
    DraftController dc;
    CHECK_EQ(dc.configure(dp), RAD_E_UNSUPPORTED);
    CHECK(!dc.cost_policy());
}

TEST(an_unpriced_shape_is_a_named_fallback_with_the_fixed_baseline) {
    Cycle c;
    /* The fake model is invalid until measured: the plan keeps the configured fallback depth. */
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Fallback);
    CHECK(c.f.plan.fallback == DraftFallback::UnknownShape);
    CHECK_EQ(c.f.plan.n_seq, 0);

    c.add(0, 1);
    c.add(1, 2);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.k[0], 3);   /* cfg.fallback_k */
    CHECK_EQ(c.f.plan.k[1], 3);
    CHECK(c.f.plan.fallback == DraftFallback::UnknownShape);
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
}

TEST(a_deep_window_is_chosen_where_drafts_keep_matching_and_a_shallow_one_where_they_do_not) {
    /* High acceptance: E(K) keeps growing, so the deepest tier wins. */
    Cycle deep;
    deep.pred.a = 0.95f;
    deep.add(0, 1);
    deep.add(1, 2);
    deep.measure(1.0, 0.2);   /* each round costs real time */
    CHECK_EQ(deep.choose(), RAD_OK);
    CHECK_EQ(deep.f.plan.k[0], 6);

    /* Low acceptance: the second draft is nearly always thrown away, so a shallow window wins. */
    Cycle shallow;
    shallow.pred.a = 0.05f;
    shallow.add(0, 1);
    shallow.add(1, 2);
    shallow.measure(1.0, 0.2);
    CHECK_EQ(shallow.choose(), RAD_OK);
    CHECK(shallow.f.plan.max_k < 6);
    CHECK(shallow.f.plan.k[0] <= 2);
}

TEST(the_batch_verifies_each_request_at_its_own_k_so_one_cycle_mixes_k0_and_k6) {
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 1);   /* text that keeps matching */
    f.add(1, 2);   /* prose */
    f.add(2, 3);   /* text that does not repeat at all */
    const float a[3] = { 0.95f, 0.3f, 0.1f };
    draft_fake::Predictor pred;
    draft_fake::predict(&pred, f.view, &f.pred);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) f.pred.seq[i].a[j] = a[i];

    DraftPlanner p;
    CHECK_EQ(p.configure(cost_config()), RAD_OK);
    TokenPriced model;
    CHECK_EQ(p.choose(f.view, f.pred, DraftCostModel{ &model, token_estimate }, &f.plan), RAD_OK);

    /* The high-acceptance request drafts deep while a low-acceptance one drafts nothing in the
     * same cycle: K belongs to the request, and the batch's rounds shrink to the requests that
     * reach them. */
    CHECK(f.plan.k[0] >= 4);
    CHECK_EQ(f.plan.k[2], 0);
    CHECK_EQ(f.plan.max_k, f.plan.k[0]);
    for (int j = 1; j <= f.plan.max_k; ++j) {
        int32_t want = 0;
        for (int i = 0; i < 3; ++i) want += f.plan.k[i] >= j ? 1 : 0;
        CHECK_EQ(f.plan.rows[j], want);
    }
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(k0_is_a_plan_and_not_a_vacancy) {
    Cycle c;
    c.pred.a = 0.0f;   /* every draft rejected: drafting is pure cost */
    c.add(0, 1);
    c.measure(1.0, 0.5);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.k[0], 0);
    CHECK_EQ(c.f.plan.max_k, 0);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);   /* a priced decision, not a fallback */
    for (int j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) CHECK_EQ(c.f.plan.rows[j], 0);
}

TEST(a_positive_budget_cuts_the_plan_and_says_so) {
    Cycle c;
    c.pred.a = 0.95f;
    for (int i = 0; i < 4; ++i) c.add(i, (uint64_t)(i + 1));
    c.measure(1.0, 0.2);
    CHECK_EQ(c.choose(), RAD_OK);
    const int32_t before = c.f.plan.max_k;
    CHECK_EQ(before, 6);

    /* The same cycle under a draft-token budget. */
    c.f.view.token_budget = 4;
    CHECK_EQ(c.choose(), RAD_OK);
    int32_t sum = 0;
    for (int i = 0; i < 4; ++i) sum += c.f.plan.k[i];
    CHECK(sum <= c.f.view.token_budget);
    CHECK(c.f.plan.k[0] < before);
    CHECK(c.f.plan.why[0] == DraftFallback::Resources);
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
}

TEST(a_cut_plan_its_own_shape_cannot_price_is_issued_unpriced) {
    Cycle c;
    c.pred.a = 0.95f;
    for (int i = 0; i < 4; ++i) c.add(i, (uint64_t)(i + 1));
    c.measure(1.0, 0.2);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 6);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK(c.f.plan.fallback == DraftFallback::None);

    /* Every window is priced except the shallow one the cut will leave: the plan is priceable
     * before the cut and not after it. */
    c.costs.ms[1] = 0.0;
    c.f.view.token_budget = 4;
    CHECK_EQ(c.choose(), RAD_OK);
    int32_t sum = 0;
    for (int i = 0; i < 4; ++i) sum += c.f.plan.k[i];
    CHECK(sum <= c.f.view.token_budget);          /* the resource limit stands */
    CHECK(c.f.plan.max_k < 6);
    /* Issued unpriced, not as a Cost plan carrying the uncut plan's price. */
    CHECK(c.f.plan.source == DraftPlanSource::Fallback);
    CHECK(c.f.plan.fallback == DraftFallback::Resources);
    CHECK(c.f.plan.why[0] == DraftFallback::Resources);
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
}

TEST(a_raise_waits_for_the_hold_and_a_reduction_does_not) {
    Cycle c;
    c.pred.a = 0.95f;
    c.add(0, 1);
    c.measure(1.0, 0.2);

    /* A first choice takes the best depth outright. */
    CHECK_EQ(c.choose(), RAD_OK);
    const int32_t deep = c.f.plan.k[0];
    CHECK_EQ(deep, 6);

    /* The text turns to prose: the estimate says shallow, and the reduction is applied at once,
     * before any hold has elapsed. */
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.2f;
    CHECK_EQ(c.choose(), RAD_OK);
    const int32_t shallow = c.f.plan.k[0];
    CHECK(shallow < deep);

    /* And back: the raise needs the hold as well as 2% more yield. Within the hold it is refused
     * even though the deep window now prices better. */
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.95f;
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.k[0], shallow);
}

TEST(a_raise_is_taken_once_the_hold_has_passed) {
    DraftPolicyConfig cfg = cost_config();
    cfg.hold = 2;
    Cycle c(6, cfg);
    c.pred.a = 0.95f;
    c.add(0, 1);
    c.measure(1.0, 0.2);

    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.k[0], 6);

    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.2f;
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK(c.f.plan.k[0] < 6);
    const int32_t shallow = c.f.plan.k[0];

    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.95f;
    CHECK_EQ(c.choose(), RAD_OK);   /* first cycle of the hold: the raise is refused */
    CHECK_EQ(c.f.plan.k[0], shallow);
    CHECK_EQ(c.choose(), RAD_OK);   /* hold spent, the deep window pays again */
    CHECK_EQ(c.f.plan.k[0], 6);
}

TEST(the_controller_freezes_one_plan_per_cycle_and_keys_it_by_identity) {
    draft_fake::Fixture f;
    f.view.cycle_id = 41;
    f.add(0, 100, 32);
    f.add(1, 101, 32);

    draft_fake::Predictor pred;
    pred.a = 0.9f;
    draft_fake::CostModel costs;
    costs.measured = true;
    for (int k = 0; k <= RAD_SCHED_MAX_SPEC; ++k) costs.ms[k] = 1.0 + 0.2 * k;

    DraftController::Params dp;
    dp.max_depth = 6;
    dp.max_seqs = draft_fake::Fixture::kMaxSeq;
    dp.policy = cost_config(6);
    dp.predictor = DraftPredictor{ &pred, draft_fake::predict };
    dp.costs = DraftCostModel{ &costs, draft_fake::estimate };
    DraftController dc;
    CHECK_EQ(dc.configure(dp), RAD_OK);
    CHECK(dc.cost_policy());

    CHECK_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* plan = dc.plan();
    CHECK(plan != nullptr);
    CHECK_EQ(plan->cycle_id, 41u);
    CHECK_EQ(plan->n_seq, 2);
    const int32_t k0 = dc.planned_k(100, 0, 1);
    CHECK_EQ(k0, plan->k[0]);

    /* Another generation in the same slot, another slot, another request: no plan. */
    CHECK_EQ(dc.planned_k(100, 0, 2), -1);
    CHECK_EQ(dc.planned_k(100, 1, 1), -1);
    CHECK_EQ(dc.planned_k(999, 0, 1), -1);

    /* The plan a cycle issued does not move while the cycle runs. */
    pred.a = 0.0f;
    CHECK_EQ(dc.planned_k(100, 0, 1), k0);

    /* A new cycle is a new plan. */
    f.view.cycle_id = 42;
    CHECK_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    CHECK_EQ(dc.plan()->cycle_id, 42u);
}

TEST(choose_is_deterministic_and_does_not_write_the_predictions) {
    Cycle a;
    Cycle b;
    a.pred.a = 0.7f;
    b.pred.a = 0.7f;
    for (int i = 0; i < 3; ++i) { a.add(i, i + 1); b.add(i, i + 1); }
    a.measure(1.0, 0.2);
    b.measure(1.0, 0.2);

    const float before = a.f.pred.seq[0].a[0];
    CHECK_EQ(a.choose(), RAD_OK);
    CHECK_EQ(b.choose(), RAD_OK);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(a.f.plan.k[i], b.f.plan.k[i]);
        CHECK_EQ(a.f.plan.req_id[i], b.f.plan.req_id[i]);
    }
    CHECK_EQ(a.f.pred.seq[0].a[0], before);
}

TEST(the_choice_is_never_worse_than_the_best_homogeneous_baseline) {
    /* Three requests with different text. The plan's yield per millisecond must be at least that
     * of every fixed depth, which is the guarantee the heuristic makes. */
    Cycle c;
    c.add(0, 1);
    c.add(1, 2);
    c.add(2, 3);
    const float a[3] = { 0.95f, 0.6f, 0.1f };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[i].a[j] = a[i];
    c.measure(1.0, 0.2);
    CHECK_EQ(c.choose(), RAD_OK);

    /* The fake prices by the deepest window, exactly as the lambda below does. */
    auto score_of = [&](const int32_t* k) {
        double y = 0.0;
        int32_t top = 0;
        for (int i = 0; i < 3; ++i) {
            double run = 1.0;
            y += 1.0;
            for (int d = 1; d <= k[i]; ++d) { run *= a[i]; y += run; }
            if (k[i] > top) top = k[i];
        }
        return y / (1.0 + 0.2 * top);
    };
    const double got = score_of(c.f.plan.k);
    const int32_t tiers[5] = { 0, 2, 3, 4, 6 };
    for (int t = 0; t < 5; ++t) {
        const int32_t k[3] = { tiers[t], tiers[t], tiers[t] };
        CHECK(got + 1e-9 >= score_of(k));
    }
}

TEST(the_cost_model_is_shown_the_shape_the_plan_verifies) {
    ShapeLog log;
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.view.prefill_tokens = 128;
    f.add(0, 1);
    f.add(1, 2, 64, DraftEligibility::OutputExhausted);
    f.seq[0].ctx_len = 40000;   /* the deepest class in the batch */
    f.seq[0].ctx_class = draft_ctx_class(f.seq[0].ctx_len);
    draft_fake::Predictor pred;
    draft_fake::predict(&pred, f.view, &f.pred);

    DraftPlanner p;
    CHECK_EQ(p.configure(cost_config()), RAD_OK);
    CHECK_EQ(p.choose(f.view, f.pred, DraftCostModel{ &log, log_estimate }, &f.plan), RAD_OK);

    CHECK(log.n >= 5);   /* one per homogeneous baseline at least */
    for (int i = 0; i < log.n; ++i) {
        const DraftVerifyShape& s = log.shape[i];
        CHECK_EQ(s.n_rows, 2);                       /* every decode row, drafting or not */
        CHECK_EQ(s.prefill_tokens, 128);
        CHECK_EQ(s.ctx_class, f.seq[0].ctx_class);
        /* verify_tokens is sum_i (1 + k_i), which the N_j round counts must agree with. */
        int32_t from_rows = s.n_rows;
        for (int j = 1; j <= RAD_SCHED_MAX_SPEC; ++j) from_rows += s.rows[j];
        CHECK_EQ(s.verify_tokens, from_rows);
        /* N_j never grows with j. */
        for (int j = 2; j <= RAD_SCHED_MAX_SPEC; ++j)
            CHECK(s.rows[j] <= s.rows[j - 1]);
        /* verify_k is the deepest round with a row in it (0 when nothing drafts). */
        int32_t top = 0;
        for (int j = 0; j <= RAD_SCHED_MAX_SPEC; ++j) if (s.rows[j] > 0) top = j;
        CHECK_EQ(s.verify_k, top);
    }
}

TEST(config_validation_refuses_a_tier_list_it_cannot_use) {
    DraftPlanner p;
    DraftPolicyConfig bad = cost_config(6);
    bad.tiers[2] = 2;   /* not ascending */
    CHECK_EQ(p.configure(bad), RAD_E_INVAL);

    DraftPolicyConfig over = cost_config(6);
    over.tiers[4] = 9;  /* past the ceiling */
    CHECK_EQ(p.configure(over), RAD_E_INVAL);

    DraftPolicyConfig dup = cost_config(6);
    dup.tiers[0] = 1; dup.tiers[1] = 1;   /* not unique */
    CHECK_EQ(p.configure(dup), RAD_E_INVAL);
}

TEST(the_documented_c8_pair_prices_depth_three_ahead_of_depth_five) {
    /* core/rad_core.h documents C8 at depth 5: 43.0 ms, 23.3 tokens; at depth 3: 36.3 ms,
     * 22.3 tokens -- 0.542 against 0.614 tokens per millisecond. The acceptance below is chosen
     * so that 8 rows reproduce exactly those two token counts; the model prices exactly those two
     * shapes. No third measurement is claimed, and none is needed: both candidates are the
     * document's. */
    draft_fake::Fixture f;
    f.view.cycle_id = 7;
    f.view.k_ceiling = 6;
    for (int i = 0; i < 8; ++i) f.add(i, (uint64_t)(100 + i));
    draft_fake::Predictor pred;
    draft_fake::predict(&pred, f.view, &f.pred);
    const float a[5] = { 0.95f, 0.6f, 0.469298f, 0.35f, 0.335114f };
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 5; ++j) f.pred.seq[i].a[j] = a[j];

    /* The same arithmetic the policy maximises, spelled out: 8 rows, E(3) and E(5). */
    double run = 1.0, e3 = 1.0;
    for (int j = 0; j < 3; ++j) { run *= a[j]; e3 += run; }
    double e5 = e3;
    for (int j = 3; j < 5; ++j) { run *= a[j]; e5 += run; }
    CHECK_NEAR(8.0 * e3, 22.3, 1e-3);
    CHECK_NEAR(8.0 * e5, 23.3, 1e-3);
    CHECK(8.0 * e3 / 36.3 > 8.0 * e5 / 43.0);

    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 3; cfg.tiers[2] = 5; cfg.n_tiers = 3;
    C8Model model;
    DraftPlanner p;
    CHECK_EQ(p.configure(cfg), RAD_OK);
    CHECK_EQ(p.choose(f.view, f.pred, DraftCostModel{ &model, c8_estimate }, &f.plan), RAD_OK);

    CHECK_EQ(f.plan.max_k, 3);
    CHECK(f.plan.source == DraftPlanSource::Cost);
    for (int i = 0; i < 8; ++i) CHECK_EQ(f.plan.k[i], 3);
    CHECK(model.calls[3] > 0);   /* both documented depths were priced ... */
    CHECK(model.calls[5] > 0);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
}

TEST(the_verify_depth_of_a_cycle_comes_from_the_frozen_plan_not_from_a_scalar) {
    /* Two rows; the deep one leaves before the next cycle. The paired step then verifies only the
     * K0 row that is still there, so the new plan's verify_k is 0 -- the previous plan's max of 6
     * is not the window that runs. */
    Cycle c;
    c.f.view.pipeline_depth = 1;   /* issued ahead: the paired step verifies the frozen plan */
    c.pred.a = 0.95f;
    c.add(0, 11);
    c.add(1, 12, 64, DraftEligibility::OutputExhausted);
    c.measure(1.0, 0.2);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 6);
    CHECK_EQ(c.f.plan.verify_k, 0);   /* the first cycle had nothing issued ahead of it */
    CHECK_EQ(c.planner.last_k(), 6);

    /* Request 11 is gone; 12 stays in its slot with its generation. */
    c.f.seq[0] = c.f.seq[1];
    c.f.view.n_seq = 1;
    draft_fake::predict(&c.pred, c.f.view, &c.f.pred);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 0);
    CHECK_EQ(c.f.plan.verify_k, 0);   /* the frozen plan drafted 0 for the surviving row */
}

TEST(alternating_plans_are_read_per_request_and_never_mixed) {
    /* The [6,2,0,0] shape: four rows, two of which stop after their second draft. Each cycle's
     * plan is rebuilt from that cycle's view, and every lookup answers with the plan in force
     * under the identity it was asked about -- never a value from the previous plan. */
    draft_fake::Fixture f;
    f.view.k_ceiling = 6;
    f.add(0, 100);
    f.add(1, 101);
    f.add(2, 102);
    f.add(3, 103);
    RowPredictor rows;
    rows.a[0] = 0.95f;
    rows.a[1] = 0.7f;
    rows.a[2] = 0.0f;
    rows.a[3] = 0.0f;

    TokenPriced model;
    DraftController::Params dp;
    dp.max_depth = 6;
    dp.max_seqs = draft_fake::Fixture::kMaxSeq;
    dp.policy = cost_config(6);
    dp.predictor = DraftPredictor{ &rows, row_predict };
    dp.costs = DraftCostModel{ &model, token_estimate };
    DraftController dc;
    CHECK_EQ(dc.configure(dp), RAD_OK);

    f.view.cycle_id = 1;
    CHECK_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* p1 = dc.plan();
    CHECK(p1 != nullptr);
    CHECK_EQ(p1->k[0], 6);
    CHECK_EQ(p1->k[1], 2);
    CHECK_EQ(p1->k[2], 0);
    CHECK_EQ(p1->k[3], 0);
    CHECK_EQ(p1->max_k, 6);
    CHECK_EQ(p1->rows[1], 2);
    CHECK_EQ(p1->rows[2], 2);
    for (int j = 3; j <= 6; ++j) CHECK_EQ(p1->rows[j], 1);
    CHECK_EQ(draft_plan_valid(*p1), RAD_OK);
    for (int i = 0; i < 4; ++i) CHECK_EQ(dc.planned_k(f.seq[i].req_id, f.seq[i].slot, 1), p1->k[i]);

    /* The next cycle has another batch: 100 finished, 104 admitted into a free slot. The plan is
     * rebuilt from that view and every lookup answers with it. */
    f.seq[0] = f.seq[1];
    f.seq[1] = f.seq[2];
    f.seq[2] = f.seq[3];
    f.add(3, 104);
    f.seq[3].slot = 7;
    rows.a[0] = 0.7f;
    rows.a[1] = 0.0f;
    rows.a[2] = 0.0f;
    rows.a[3] = 0.95f;
    f.view.cycle_id = 2;
    CHECK_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* p2 = dc.plan();
    CHECK(p2 != nullptr);
    CHECK_EQ(p2->cycle_id, 2u);
    CHECK_EQ(p2->n_seq, 4);
    CHECK_EQ(dc.planned_k(100, 0, 1), -1);   /* not in this cycle's plan at all */
    for (int i = 0; i < 4; ++i) CHECK_EQ(dc.planned_k(f.seq[i].req_id, f.seq[i].slot, 1), p2->k[i]);
    CHECK_EQ(dc.planned_k(104, 7, 1), p2->k[3]);
    CHECK_EQ(dc.planned_k(104, 7, 2), -1);   /* another generation of the slot is another request */
    CHECK_EQ(draft_plan_valid(*p2), RAD_OK);
}

RAD_TEST_MAIN()
