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
#include "sched/draft_cost.h"
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
    out->cycle_id = view.cycle_id;        /* the set describes this cycle, as draft_predictions_current asks */
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

/* The four-prompt calibration's shape: every row reaches every round, the serial head loaded
 * (history = verify tokens), the cycle played. `draft_k` is what this plan drafts, `verify_k`
 * what the paired step verifies -- under issue_ahead() the two differ. */
DraftVerifyShape key_shape(int32_t n_rows, int32_t draft_k, int32_t verify_k, int32_t depth) {
    DraftVerifyShape s;
    s.n_rows = n_rows;
    s.ctx_class = draft_ctx_class(1024);
    s.sampling = DraftSampling::Greedy;
    s.replay = DraftReplay::Play;
    s.drafter = DraftDrafter::Serial;
    s.pipeline_depth = depth;
    s.draft_k = draft_k;
    s.verify_k = verify_k;
    for (int32_t j = 1; j <= draft_k; ++j) s.draft_rows[j] = n_rows;
    s.verify_tokens = n_rows;
    for (int32_t j = 1; j <= verify_k; ++j) {
        s.rows[j] = n_rows;
        s.verify_tokens += n_rows;
    }
    s.history_tokens = s.verify_tokens;   /* one history row per verify row, no shared head */
    return s;
}

/* A four-prompt batch at `depth`, played, with the serial head loaded. */
void four_rows(draft_fake::Fixture* f, draft_fake::Predictor* pred, int32_t depth) {
    f->view.k_ceiling = 6;
    f->view.pipeline_depth = depth;
    f->view.replay = DraftReplay::Play;
    for (int32_t i = 0; i < 4; ++i) f->add(i, (uint64_t)(i + 1));
    draft_fake::predict(pred, f->view, &f->pred);
}

/* A loaded no-loan reference: the real DraftCostTable, exact buckets only. There is no
 * interpolation and no heuristic fallback behind it, so a shape that is not in this table is not
 * priceable at all -- which is what makes the lookup key itself testable. */
struct ReferenceTable {
    DraftCostProfile p;
    DraftCostTable t;

    void add(const DraftVerifyShape& key, double ms) {
        DraftCostBucket b;
        b.shape = key;
        b.rounds = key.draft_k;
        b.draft_rows_total = 0;
        for (int32_t j = 1; j <= key.draft_k; ++j) {
            b.draft_round_rows[j] = key.draft_rows[j];
            b.draft_rows_total += key.draft_rows[j];
        }
        b.mean_ms = ms;
        b.unc_ms = 0.5;
        b.n = 3;
        p.buckets.push_back(b);
    }
    void build() {
        p.execution = DraftCostExecution::Rectangular;
        p.fingerprint.basis = DraftCostBasis::NoLoan;
        DraftCostRuntime rt;
        rt.execution = DraftCostExecution::Rectangular;
        rt.fingerprint.basis = DraftCostBasis::NoLoan;
        CHECK_EQ(t.configure(p, rt), RAD_OK);
        CHECK(t.active());
        CHECK(t.reference());
    }
};

/* The reference table asked through a log: every key the policy really requested, priced by the
 * real table. "The lookup asked for the executable shape" is then an assertion about the query
 * itself, not about the plan that happened to come out. */
struct SpiedTable {
    const DraftCostTable* t = nullptr;
    ShapeLog log;

    void reset(const DraftCostTable* table) { t = table; log.n = 0; }
    DraftCostModel model() { return DraftCostModel{this, &priced}; }
    static int priced(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
        auto* self = static_cast<SpiedTable*>(ctx);
        if (self->log.n < ShapeLog::kMax) self->log.shape[self->log.n++] = s;
        return self->t->estimate(s, out);
    }
    const DraftVerifyShape* find(int32_t draft_k, int32_t verify_k) const {
        for (int i = 0; i < log.n; ++i)
            if (log.shape[i].draft_k == draft_k && log.shape[i].verify_k == verify_k)
                return &log.shape[i];
        return nullptr;
    }
};

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

TEST(a_raise_and_a_reduction_are_both_priced_by_the_switch) {
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

    /* And back: the raise is priced, not counted. The deep window's yield pays for the one cycle
     * the switch spends before the dwell's worth of its gain has accrued, so it is taken on the
     * very next cycle -- no hold to wait out. */
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.95f;
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.k[0], deep);
    CHECK_EQ(c.planner.last_k(), deep);
}

/* `hold` is the dwell the switch is priced over, not a counter that delays it: a short dwell makes
 * the one transition cycle weigh more (it is 1 of 2 cycles instead of 1 of 8), it does not add a
 * cycle of waiting. The raise below pays under either. */
TEST(hold_prices_the_dwell_and_does_not_delay_the_switch) {
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

    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.95f;
    CHECK_EQ(c.choose(), RAD_OK);   /* the deep window pays its own transition cycle */
    CHECK_EQ(c.f.plan.k[0], 6);
}

/* A reference's single-row play buckets, (k,k) at context class 0, as a TP2 no-loan calibration
 * left them: means and the bucket's `unc`, the largest deviation one sample showed. The well
 * measured buckets carry the widest `unc` (n is 1034..2003 behind each of these). */
namespace {
const double kRefMs[7]  = { 15.701, 21.057, 25.713, 30.348, 34.376, 39.616, 44.415 };
const double kRefUnc[7] = { 3.566, 5.929, 6.640, 29.420, 8.914, 158.719, 12.013 };
int reference_estimate(void*, const DraftVerifyShape& s, DraftCostEstimate* out) {
    *out = DraftCostEstimate{};
    if (s.n_rows != 1 || s.verify_k < 0 || s.verify_k > 6) return RAD_OK;
    out->ms = kRefMs[s.verify_k];
    out->unc_ms = kRefUnc[s.verify_k];
    out->coverage = 1000;
    out->source = DraftCostSource::Measured;
    out->valid = true;
    return RAD_OK;
}
}  /* namespace */

/* A reduction is not permanent. Prose-like acceptance takes the plan from K4 to K2 at once; when
 * code-like acceptance returns, the priced switch takes K4 back on the very next cycle. Scoring
 * the raise on `ms + unc` refused it forever: 3.935 / (34.376 + 8.914) is under K2's
 * 2.654 / 25.713 * 1.02 at any acceptance this text has, and that is how a BetterBench run stayed
 * shallow after its first category. Pipeline depth 0, so the executed and the stationary shape are
 * the same (k,k). */
TEST(a_reduction_is_not_permanent_when_the_deeper_bucket_is_well_measured) {
    DraftPolicyConfig cfg = cost_config();
    for (int32_t t = 0; t <= 6; ++t) cfg.tiers[t] = t;
    cfg.n_tiers = 7;
    cfg.fallback_k = 6;
    Cycle c(6, cfg);
    c.add(0, 1, 10000);
    const DraftCostModel model{ nullptr, reference_estimate };
    const auto accept = [&](float a) {
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = a;
    };

    accept(0.88f);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 4);   /* 3.935 / 34.376 beats K5's 4.463 / 39.616 */

    accept(0.68f);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 2);   /* a reduction is immediate */
    CHECK(c.f.plan.source == DraftPlanSource::Cost);

    /* Code-like again: the raise is priced and taken at once -- K4 yields 48% more tokens for 34%
     * more milliseconds, which pays for the one cycle the switch spends at the deeper price. */
    accept(0.88f);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 4);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(c.planner.last_k(), 4);

    /* The switch has to pay, and at a = .93 it does not: K6 is the best depth by 1.3%, but its
     * cycle costs 29% more than K4's, so the dwell's first cycle at (44.4 ms, K4's yield) eats the
     * gain no matter how long the plan would stay. It stays at K4 ... */
    accept(0.93f);
    for (int n = 0; n < 3 * cfg.hold; ++n) {
        REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        CHECK_EQ(c.f.plan.max_k, 4);
    }
    /* ... and at a = .95, where the deep rung is 3.2% ahead, it is still a wash: 44.4 ms for one
     * cycle of K4's tokens against eight cycles 3.2% ahead. */
    accept(0.95f);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 4);
}

/* The other side of the same rule: the climb is one priced decision, not one rung per `hold`. From
 * K2 the raise goes straight to K6 at a = .95 -- K6 yields 108% more tokens than K2 for 73% more
 * milliseconds, which pays the dwell's one transition cycle -- and a plan already at K6 stays
 * there. */
TEST(a_priced_raise_climbs_in_one_cycle_and_not_a_rung_at_a_time) {
    DraftPolicyConfig cfg = cost_config();
    for (int32_t t = 0; t <= 6; ++t) cfg.tiers[t] = t;
    cfg.n_tiers = 7;
    cfg.fallback_k = 6;
    Cycle c(6, cfg);
    c.add(0, 1, 10000);
    const DraftCostModel model{ nullptr, reference_estimate };
    const auto accept = [&](float a) {
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = a;
    };

    accept(0.68f);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 2);   /* the free first choice: 2.142 / 25.713 */

    accept(0.95f);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 6);   /* one cycle, all four rungs, priced once */
    CHECK_EQ(c.planner.last_k(), 6);
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 6);   /* and the incumbent now holds the best rung */
}

/* The same reference, with one depth's buckets taken away on request. */
namespace {
struct ReferenceWithout {
    int32_t missing = -1;   /* the verify depth that has no bucket; -1: none */
};
int reference_without(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
    if (s.verify_k == static_cast<const ReferenceWithout*>(ctx)->missing) {
        *out = DraftCostEstimate{};
        return RAD_OK;
    }
    return reference_estimate(nullptr, s, out);
}
DraftPolicyConfig held_reductions_config() {
    DraftPolicyConfig cfg = cost_config();
    for (int32_t t = 0; t <= 6; ++t) cfg.tiers[t] = t;
    cfg.n_tiers = 7;
    cfg.fallback_k = 6;
    cfg.hold_reductions = true;
    return cfg;
}
}  /* namespace */

/* --draft-hysteresis both: a reduction is judged by the same priced switch rule a raise is, so a
 * dip of the estimate can no longer take the depth down at once without paying for it -- and one
 * that pays is still taken at once, no hold to wait out. */
TEST(with_held_reductions_a_reduction_is_taken_when_it_pays_and_refused_when_it_does_not) {
    const DraftPolicyConfig cfg = held_reductions_config();
    Cycle c(6, cfg);
    c.add(0, 1, 10000);
    const DraftCostModel model{ nullptr, reference_estimate };
    const auto accept = [&](float a) {
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = a;
    };
    const auto choose = [&]() {
        CHECK_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        return c.f.plan.max_k;
    };

    accept(0.88f);
    CHECK_EQ(choose(), 4);   /* the first choice is free */

    /* Prose-like: K2 is 7.2% ahead of K4 and 8.7 ms cheaper, so the dwell's first cycle still
     * commits K4's tokens at that price -- the move pays at once. */
    accept(0.68f);
    CHECK_EQ(choose(), 2);
    CHECK_EQ(c.planner.last_k(), 2);

    /* The raise back is priced the same way and taken at once: 3.935 / 34.376 against
     * 2.654 / 25.713, one transition cycle at K4's price included. */
    accept(0.88f);
    CHECK_EQ(choose(), 4);

    /* A reduction that does not pay is refused however long the plan would hold it: at a = .78 K3
     * is the best depth, 0.3% ahead of K4 (2.863 / 30.348 against 3.233 / 34.376) -- under the
     * rule's 2% return. */
    accept(0.78f);
    for (int n = 0; n < 3 * cfg.hold; ++n) CHECK_EQ(choose(), 4);
    /* At a = .70 it pays (2.19 / 25.713 against 2.773 / 34.376, and the first cycle keeps the deep
     * window's tokens), so it is taken. */
    accept(0.70f);
    CHECK_EQ(choose(), 2);
}

/* What a held reduction does not hold up: a row its own remaining output caps, the resource cut,
 * and an incumbent the model can no longer price. */
TEST(with_held_reductions_caps_cuts_and_an_unpriced_incumbent_still_act_at_once) {
    const DraftPolicyConfig cfg = held_reductions_config();
    ReferenceWithout ref;
    const DraftCostModel model{ &ref, reference_without };

    Cycle c(6, cfg);
    const auto accept = [&](float a) {
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = a;
    };
    const auto choose = [&]() {
        CHECK_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        return c.f.plan.max_k;
    };
    c.add(0, 1, 10000);
    accept(0.88f);
    CHECK_EQ(choose(), 4);

    /* Two tokens left: the row's cap is one draft, within the hold, and the incumbent stays. */
    c.add(0, 1, 2);
    accept(0.88f);
    CHECK_EQ(choose(), 1);
    CHECK_EQ(c.planner.last_k(), 4);

    /* A token budget of two drafts cuts the plan at once, within the hold. */
    c.add(0, 2, 10000);
    accept(0.88f);
    c.f.view.token_budget = 2;
    CHECK_EQ(choose(), 2);
    CHECK(c.f.plan.why[0] == DraftFallback::Resources);
    c.f.view.token_budget = 0;

    /* An incumbent the model cannot price is no reason to stay: with K4's buckets gone, K2 is
     * taken at once although the hold has not passed. */
    Cycle u(6, cfg);
    u.add(0, 1, 10000);
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) u.f.pred.seq[0].a[j] = 0.88f;
    REQUIRE_EQ(u.planner.choose(u.f.view, u.f.pred, model, &u.f.plan), RAD_OK);
    CHECK_EQ(u.f.plan.max_k, 4);
    ref.missing = 4;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) u.f.pred.seq[0].a[j] = 0.70f;
    REQUIRE_EQ(u.planner.choose(u.f.view, u.f.pred, model, &u.f.plan), RAD_OK);
    CHECK_EQ(u.f.plan.max_k, 2);
    CHECK(u.f.plan.source == DraftPlanSource::Cost);
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

    REQUIRE_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* plan = dc.plan();
    REQUIRE(plan != nullptr);   /* no plan means the rest of the case indexes nothing */
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
    REQUIRE_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* next = dc.plan();
    REQUIRE(next != nullptr);
    CHECK_EQ(next->cycle_id, 42u);
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

/* The heuristic against the enumeration it stands in for, on the small batch the enumeration is
 * affordable on (three requests, five tiers, 125 plans). The policy never runs this -- 7^B is not a
 * hotpath -- but on a batch this size the coordinate passes must land on the enumerated optimum,
 * not merely beat the homogeneous baselines. */
TEST(the_coordinate_search_matches_the_enumeration_on_a_small_batch) {
    Cycle c;
    for (int i = 0; i < 3; ++i) c.add(i, (uint64_t)i + 1);
    const float a[3] = { 0.9f, 0.6f, 0.3f };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[i].a[j] = a[i];
    c.measure(1.0, 0.2);
    CHECK_EQ(c.choose(), RAD_OK);

    /* The fake prices a plan by its deepest window, as the enumeration below does. */
    const auto score_of = [&](const int32_t* k) {
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
    const int32_t tiers[5] = { 0, 2, 3, 4, 6 };
    double best = 0.0;
    int32_t best_k[3] = { 0, 0, 0 };
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
            for (int z = 0; z < 5; ++z) {
                const int32_t k[3] = { tiers[x], tiers[y], tiers[z] };
                const double v = score_of(k);
                if (v > best) { best = v; for (int i = 0; i < 3; ++i) best_k[i] = k[i]; }
            }

    const double got = score_of(c.f.plan.k);
    CHECK(got + 1e-9 >= best);
    /* And it is a plan, not a lucky tie: the deep window goes to the request that keeps matching. */
    CHECK_EQ(c.f.plan.k[0], best_k[0]);
    CHECK_EQ(c.f.plan.k[2], best_k[2]);
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

/* A calibration sequence is not a ladder: order and duplicates are kept (`3,2,2,3` is the point),
 * but every value must be one of the depths the deployment actually offers -- the effective
 * ladder after the trim/dedup -- so no sample is filed at a depth serving never offers. */
TEST(config_validation_refuses_a_calibration_sequence_it_cannot_use) {
    DraftPlanner p;
    DraftPolicyConfig cfg = cost_config(6);
    cfg.calib_k[0] = 3; cfg.calib_k[1] = 2; cfg.calib_k[2] = 2; cfg.calib_k[3] = 3;
    cfg.n_calib_k = 4;
    CHECK_EQ(p.configure(cfg), RAD_OK);   /* duplicates, unsorted: accepted */

    DraftPolicyConfig shallow = cost_config(3);
    shallow.tiers[0] = 0; shallow.tiers[1] = 2; shallow.tiers[2] = 3; shallow.n_tiers = 3;
    shallow.calib_k[0] = 3; shallow.calib_k[1] = 4;   /* 4 is in no offered tier */
    shallow.n_calib_k = 2;
    CHECK_EQ(p.configure(shallow), RAD_E_INVAL);

    DraftPolicyConfig legacy;   /* a sequence has no meaning without the cost policy */
    legacy.k_ceiling = 6;
    legacy.calib_k[0] = 3; legacy.calib_k[1] = 2;
    legacy.n_calib_k = 2;
    CHECK_EQ(p.configure(legacy), RAD_E_INVAL);

    DraftPolicyConfig single = cfg;
    single.n_calib_k = 1;   /* one depth is not a sequence */
    CHECK_EQ(p.configure(single), RAD_E_INVAL);
}

/* `3,2,2,3` over five non-empty cycles: A (3->3), B (3->2), D (2->2), C (2->3), A. The verify
 * depth of each plan is the frozen previous plan's -- the declared window. What the paired step
 * really verifies is the executed shape (sched_draft_test.cpp); on the direct step() path
 * `draft_cap` cuts a fall below that window, on the ahead path it does not. The branch never
 * prices and never consults the predictor. */
TEST(a_calibration_sequence_forces_its_depths_and_only_rows_advance_it) {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.calib_k[0] = 3; cfg.calib_k[1] = 2; cfg.calib_k[2] = 2; cfg.calib_k[3] = 3;
    cfg.n_calib_k = 4;
    Cycle c(6, cfg);
    for (int i = 0; i < 4; ++i) c.add(i, (uint64_t)(i + 1));
    c.f.view.pipeline_depth = 1;   /* the scheduler issues ahead: the step verifies the last plan */

    const int32_t want[5]   = { 3, 2, 2, 3, 3 };
    const int32_t verify[5] = { 0, 3, 2, 2, 3 };
    for (int n = 0; n < 5; ++n) {
        c.f.view.cycle_id = (uint64_t)(n + 1);
        CHECK_EQ(c.choose(), RAD_OK);
        CHECK(c.f.plan.source == DraftPlanSource::Calibration);
        CHECK(c.f.plan.fallback == DraftFallback::None);
        CHECK_EQ(c.f.plan.max_k, want[n]);
        CHECK_EQ(c.f.plan.verify_k, verify[n]);
        for (int i = 0; i < 4; ++i) CHECK_EQ(c.f.plan.k[i], want[n]);
        CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
        /* Forced, not chosen: no tier was priced and this cycle is not a cost decision. */
        CHECK_EQ(c.planner.last_tiers_priced(), 0);
    }
    CHECK_EQ(c.costs.calls, 0);   /* the branch never asks for a price */

    /* reset() starts the rotation over, as configure() does. */
    c.planner.reset();
    c.f.view.cycle_id = 6;
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 3);
    CHECK_EQ(c.f.plan.verify_k, 0);   /* reset also forgets the frozen window */
}

/* An empty view -- the prefill cycle before the first decode, or an idle planning site -- is
 * forced too, but it is no depth measurement: it must not step the sequence, or every later depth
 * lands one place off the rotation. */
TEST(an_empty_cycle_does_not_advance_the_calibration_sequence) {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.calib_k[0] = 3; cfg.calib_k[1] = 2; cfg.calib_k[2] = 2; cfg.calib_k[3] = 3;
    cfg.n_calib_k = 4;
    Cycle c(6, cfg);
    for (int i = 0; i < 4; ++i) c.add(i, (uint64_t)(i + 1));
    c.f.view.pipeline_depth = 1;

    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 3);

    /* The prefill step: nothing runs yet. */
    c.f.view.cycle_id = 2;
    c.f.view.n_seq = 0;
    draft_fake::predict(&c.pred, c.f.view, &c.f.pred);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 0);
    CHECK(c.f.plan.source == DraftPlanSource::Calibration);

    /* The first decode cycle is still the sequence's second depth, not its third. The empty
     * cycle cleared the frozen window (existing remember() semantics), so it verifies 0. */
    c.f.view.cycle_id = 3;
    c.f.view.n_seq = 4;
    draft_fake::predict(&c.pred, c.f.view, &c.f.pred);
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 2);
    CHECK_EQ(c.f.plan.verify_k, 0);
}

/* The same budget cut the cost path applies, applied to a forced depth: the plan never asks for
 * more drafts than the step was given, and says on the row why. */
TEST(a_budget_cuts_a_forced_depth_and_marks_the_row) {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.calib_k[0] = 3; cfg.calib_k[1] = 2;
    cfg.n_calib_k = 2;
    Cycle c(6, cfg);
    for (int i = 0; i < 4; ++i) c.add(i, (uint64_t)(i + 1));
    c.f.view.token_budget = 8;   /* 4 rows at K3 want 12 */

    CHECK_EQ(c.choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Calibration);
    CHECK_EQ(c.f.plan.max_k, 2);
    for (int i = 0; i < 4; ++i) {
        CHECK_EQ(c.f.plan.k[i], 2);
        CHECK(c.f.plan.why[i] == DraftFallback::Resources);
    }
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
}

/* K2: "both depths were priced" has to be readable, not assumed. Stage 1 issues a Cost plan at
 * the deepest tier even when only that tier had a price, so the plan's source alone cannot tell
 * an alternative that lost from one that was never measured. */
TEST(the_ledger_shows_how_many_depths_the_cost_search_priced) {
    DraftPolicyConfig cfg = cost_config(3);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;

    Cycle deep_only(3, cfg);
    deep_only.pred.a = 0.95f;
    deep_only.add(0, 1);
    deep_only.costs.measured = true;
    deep_only.costs.ms[3] = 3.0;
    CHECK_EQ(deep_only.choose(), RAD_OK);
    CHECK(deep_only.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(deep_only.planner.last_tiers_priced(), 1);

    Cycle both(3, cfg);
    both.pred.a = 0.95f;
    both.add(0, 1);
    both.costs.measured = true;
    both.costs.ms[3] = 3.0;
    both.costs.ms[2] = 2.9;   /* priced, and still not enough to beat the deep window */
    CHECK_EQ(both.choose(), RAD_OK);
    CHECK(both.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(both.f.plan.max_k, 3);
    CHECK_EQ(both.planner.last_tiers_priced(), 2);
}

/* The same evidence per depth, which the ledger's count cannot show: every offered tier records
 * whether it was priced and for how much, and the issued plan carries its own estimate -- the
 * number a diagnostic compares against the alternatives. A depth the model could not price keeps
 * `valid` false instead of a zero, and a plan that priced nothing (a calibration) says so by
 * `last_price_known() == false`, not by an invalid price. */
TEST(every_offered_tier_records_its_price_and_the_issued_plan_its_own) {
    DraftPolicyConfig cfg = cost_config(3);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;

    Cycle c(3, cfg);
    c.pred.a = 0.95f;
    c.add(0, 1);
    c.costs.measured = true;
    c.costs.ms[2] = 2.9;
    c.costs.ms[3] = 3.0;   /* K0 is asked and has no price */
    CHECK_EQ(c.choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    REQUIRE(c.planner.last_price_known());
    REQUIRE_EQ(c.planner.last_n_tier_prices(), 3);
    const DraftTierPrice* tp = c.planner.last_tier_prices();
    CHECK_EQ(tp[0].k, 0);
    CHECK(!tp[0].valid);               /* asked, but this model has no K0 measurement */
    CHECK_EQ(tp[1].k, 2);
    CHECK(tp[1].valid);
    CHECK_NEAR(tp[1].ms, 2.9, 1e-9);
    CHECK_NEAR(tp[1].unc_ms, 0.05 * 2.9, 1e-9);
    CHECK_EQ(tp[2].k, 3);
    CHECK(tp[2].valid);
    CHECK_NEAR(tp[2].ms, 3.0, 1e-9);
    /* The issued plan is the deep one, and its own estimate is exactly that tier's. */
    CHECK_EQ(c.f.plan.max_k, 3);
    CHECK_EQ(c.planner.last_price().k, 3);
    CHECK(c.planner.last_price().valid);
    CHECK_NEAR(c.planner.last_price().ms, 3.0, 1e-9);

    /* An unpriced deepest tier no longer stops the search: every tier is asked, the priced K2 is
     * the Cost plan, and the trace keeps the deepest as invalid rather than zero. */
    Cycle u(3, cfg);
    u.pred.a = 0.95f;
    u.add(0, 1);
    u.costs.measured = true;
    u.costs.ms[2] = 2.9;   /* the only priced tier */
    CHECK_EQ(u.choose(), RAD_OK);
    CHECK(u.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(u.f.plan.max_k, 2);
    CHECK(u.planner.last_price_known());
    REQUIRE_EQ(u.planner.last_n_tier_prices(), 3);
    CHECK_EQ(u.planner.last_tier_prices()[2].k, 3);
    CHECK(!u.planner.last_tier_prices()[2].valid);
    CHECK(u.planner.last_tier_prices()[1].valid);
    CHECK_EQ(u.planner.last_tiers_priced(), 1);
    CHECK(u.planner.last_price().valid);
    CHECK_NEAR(u.planner.last_price().ms, 2.9, 1e-9);

    /* A calibration prices nothing at all: empty list, and "no search ran" is a different fact
     * from "the search found no price". */
    DraftPolicyConfig cal = cfg;
    cal.calib_k[0] = 2;
    cal.calib_k[1] = 3;
    cal.n_calib_k  = 2;
    Cycle k(3, cal);
    k.add(0, 1);
    CHECK_EQ(k.choose(), RAD_OK);
    CHECK(k.f.plan.source == DraftPlanSource::Calibration);
    CHECK(!k.planner.last_price_known());
    CHECK_EQ(k.planner.last_n_tier_prices(), 0);
}

/* The price lookup asks for the window the plan declares: the frozen previous cycle's K. A K2
 * candidate after a K3 plan therefore asks `(2,3)` -- four rows at verify 3 and 16 verify/history
 * tokens. This view is the ahead path (promote_next()), whose step was built under the publishing
 * plan and verifies the frozen window as declared; the direct step() path prices the window its
 * step will really run, which the_direct_step_prices_the_window_the_step_will_re_cap covers. The
 * reference table is exact, so what it does not hold nothing prices: the query itself is what the
 * test sees. */
TEST(the_price_lookup_asks_for_the_frozen_verify_window_and_the_cut_reprices_it) {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.fallback_k = 3;

    ReferenceTable a_d;   /* A (3,3), B (2,3) and the cold (3,0) of the first cycle */
    a_d.add(key_shape(4, 3, 0, 1), 9.0);
    a_d.add(key_shape(4, 3, 3, 1), 12.0);
    a_d.add(key_shape(4, 2, 3, 1), 11.0);   /* measured, and does not beat the deep window */
    a_d.add(key_shape(4, 2, 2, 1), 11.0);   /* K2's own stationary horizon: 2.44 / 11 < 2.952 / 12 */
    a_d.build();
    ReferenceTable a_only;
    a_only.add(key_shape(4, 3, 3, 1), 12.0);
    a_only.build();

    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    four_rows(&f, &pred, /*depth=*/1);
    DraftPlanner planner;
    REQUIRE_EQ(planner.configure(cfg), RAD_OK);
    SpiedTable spy;

    /* Cycle 1: no frozen predecessor, so the deepest gate asks (3,0) and issues K3. */
    spy.reset(&a_d.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK(f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(f.plan.max_k, 3);
    CHECK(spy.find(3, 0) != nullptr);

    /* Cycle 2 against A-only: K2 after K3 asks the declared (2,3) -- four rows, 16 verify and 16
     * history tokens -- and nothing is priced for it, so the search says so instead of inventing
     * a number. */
    spy.reset(&a_only.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    const DraftVerifyShape* b = spy.find(2, 3);
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_rows, 4);
    CHECK_EQ(b->verify_tokens, 16);
    CHECK_EQ(b->history_tokens, 16);
    CHECK_EQ(b->rows[1], 4);
    CHECK_EQ(b->rows[3], 4);
    CHECK_EQ(b->pipeline_depth, 1);
    CHECK_EQ(planner.last_tiers_priced(), 1);
    CHECK(!planner.last_tier_prices()[1].valid);

    /* Cycle 3 against A+B: the same query hits the measured bucket. Both alternatives are
     * priceable, and the deep one still wins. */
    spy.reset(&a_d.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(planner.last_tiers_priced(), 2);
    CHECK_EQ(planner.last_tier_prices()[1].k, 2);
    CHECK(planner.last_tier_prices()[1].valid);
    CHECK_NEAR(planner.last_tier_prices()[1].ms, 11.0, 1e-9);
    CHECK_EQ(planner.last_tier_prices()[2].k, 3);
    CHECK(planner.last_tier_prices()[2].valid);
    CHECK_EQ(f.plan.max_k, 3);
    CHECK(f.plan.source == DraftPlanSource::Cost);

    /* Cycle 4: the token budget cuts K3 to K2 before the plan is issued. The re-price asks the
     * same window again and finds it, so the cut plan stays a Cost plan -- and `tiers_priced`
     * keeps the search's count (two depths weighed), not the issued depth (last_k). */
    f.view.token_budget = 8;   /* four rows at K3 sum to 12 */
    spy.reset(&a_d.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 2);
    CHECK(f.plan.why[0] == DraftFallback::Resources);
    CHECK(f.plan.source == DraftPlanSource::Cost);
    CHECK(planner.last_price().valid);
    CHECK_EQ(planner.last_price().k, 2);
    CHECK_NEAR(planner.last_price().ms, 11.0, 1e-9);
    CHECK_EQ(planner.last_k(), 2);
    CHECK_EQ(planner.last_tiers_priced(), 2);
}

/* An upshift and a pipeline-0 cycle are not affected by the two-window rule: a candidate at or
 * above the frozen depth verifies the frozen window, and at pipeline_depth 0 the candidate's own K
 * is the window. */
TEST(an_upshift_and_a_pipeline_zero_cycle_keep_the_verify_window_they_had) {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.fallback_k = 2;

    /* Upshift: a K2 plan is issued first (the named baseline, the table is empty), so the next
     * cycle weighs K3 against a frozen 2. */
    ReferenceTable none;
    none.build();
    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    four_rows(&f, &pred, /*depth=*/1);
    DraftPlanner planner;
    REQUIRE_EQ(planner.configure(cfg), RAD_OK);
    SpiedTable spy;
    spy.reset(&none.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK(f.plan.source == DraftPlanSource::Fallback);
    CHECK_EQ(f.plan.max_k, 2);

    spy.reset(&none.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    const DraftVerifyShape* up = spy.find(3, 2);
    REQUIRE(up != nullptr);      /* frozen 2, candidate 3: the frozen window is still the bound */
    CHECK_EQ(up->verify_tokens, 12);
    CHECK_EQ(up->history_tokens, 12);
    /* (3,3) is asked only as K3's stationary horizon, never as the window this step verifies. */
    CHECK(spy.find(2, 3) == nullptr);
    CHECK_EQ(f.plan.verify_k, 2);   /* the declared window is the frozen one, unchanged */

    /* Pipeline 0: the paired step verifies what this cycle drafts, so the candidate's own K is
     * the key. */
    ReferenceTable in_step;
    in_step.add(key_shape(4, 3, 3, 0), 12.0);
    in_step.add(key_shape(4, 2, 2, 0), 11.0);
    in_step.build();
    draft_fake::Fixture g;
    draft_fake::Predictor pred2;
    four_rows(&g, &pred2, /*depth=*/0);
    DraftPlanner p0;
    REQUIRE_EQ(p0.configure(cfg), RAD_OK);
    spy.reset(&in_step.t);
    CHECK_EQ(p0.choose(g.view, g.pred, spy.model(), &g.plan), RAD_OK);
    const DraftVerifyShape* own = spy.find(2, 2);
    REQUIRE(own != nullptr);
    CHECK_EQ(own->verify_tokens, 12);
    CHECK_EQ(p0.last_tiers_priced(), 2);
    CHECK_EQ(g.plan.max_k, 3);
    CHECK(spy.find(2, 3) == nullptr);
}

TEST(the_direct_step_prices_the_window_the_step_will_re_cap) {
    /* A step() cycle builds its step after the plan, so Scheduler::draft_cap re-caps each row's
     * frozen window to the candidate's own K. The lookup follows the caller: a K2 candidate after
     * a frozen K3 asks `(2,2)`, never the ahead path's declared `(2,3)` -- and the resource cut
     * re-prices the same window it will run. */
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.fallback_k = 3;

    ReferenceTable a_d;   /* the cold (3,0) gate, the re-capped (2,2), the K3 window (3,3) */
    a_d.add(key_shape(4, 3, 0, 1), 9.0);
    a_d.add(key_shape(4, 3, 3, 1), 12.0);
    a_d.add(key_shape(4, 2, 2, 1), 11.0);
    a_d.build();

    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    four_rows(&f, &pred, /*depth=*/1);
    f.view.direct_step = true;
    DraftPlanner planner;
    REQUIRE_EQ(planner.configure(cfg), RAD_OK);
    SpiedTable spy;

    /* Cycle 1: nothing frozen, so the gate asks (3,0) and issues K3 on either path. */
    spy.reset(&a_d.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(f.plan.source, DraftPlanSource::Cost);
    CHECK_EQ(f.plan.max_k, 3);
    CHECK(spy.find(3, 0) != nullptr);

    /* Cycle 2: K2 after a frozen K3 asks the re-capped (2,2) -- four rows, 12 verify and 12
     * history tokens -- and the ahead path's (2,3) is never asked. */
    spy.reset(&a_d.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    const DraftVerifyShape* d2 = spy.find(2, 2);
    REQUIRE(d2 != nullptr);
    CHECK_EQ(d2->n_rows, 4);
    CHECK_EQ(d2->verify_tokens, 12);
    CHECK_EQ(d2->history_tokens, 12);
    CHECK(spy.find(2, 3) == nullptr);
    CHECK_EQ(planner.last_tiers_priced(), 2);
    CHECK_EQ(f.plan.max_k, 3);

    /* Cycle 3: the token budget cuts K3 to K2. The re-price asks the same re-capped window and
     * finds its own bucket: the issued plan is priced as its step will run. */
    f.view.token_budget = 8;   /* four rows at K3 sum to 12 */
    spy.reset(&a_d.t);
    CHECK_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 2);
    CHECK(f.plan.why[0] == DraftFallback::Resources);
    CHECK_EQ(f.plan.verify_k, 2);
    CHECK_EQ(f.plan.history_tokens[0], 3);
    CHECK(spy.find(2, 2) != nullptr);
    CHECK(spy.find(2, 3) == nullptr);
    CHECK(planner.last_price().valid);
    CHECK_EQ(planner.last_price().k, 2);
    CHECK_NEAR(planner.last_price().ms, 11.0, 1e-9);
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
    REQUIRE_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* p1 = dc.plan();
    REQUIRE(p1 != nullptr);
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
    REQUIRE_EQ(dc.open_cycle(f.view, &f.pred), RAD_OK);
    const DraftPlan* p2 = dc.plan();
    REQUIRE(p2 != nullptr);
    CHECK_EQ(p2->cycle_id, 2u);
    CHECK_EQ(p2->n_seq, 4);
    CHECK_EQ(dc.planned_k(100, 0, 1), -1);   /* not in this cycle's plan at all */
    for (int i = 0; i < 4; ++i) CHECK_EQ(dc.planned_k(f.seq[i].req_id, f.seq[i].slot, 1), p2->k[i]);
    CHECK_EQ(dc.planned_k(104, 7, 1), p2->k[3]);
    CHECK_EQ(dc.planned_k(104, 7, 2), -1);   /* another generation of the slot is another request */
    CHECK_EQ(draft_plan_valid(*p2), RAD_OK);
}

/* A request that ends at max_tokens is capped to K1/K0 by its own remaining output. That is not a
 * policy reduction: the next request must be planned at the incumbent depth again. Before the fix
 * the cap became the hysteresis incumbent, and a fresh request's raise was priced against (K1,
 * frozen 0) -- a shape no calibration runs -- so it was refused and every later request stayed
 * shallow (BetterBench single-stream, all requests finishing on `length`). */
TEST(a_request_capped_by_its_remaining_output_does_not_hold_the_next_request_shallow) {
    DraftPolicyConfig cfg = cost_config(3);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    cfg.fallback_k = 3;

    ReferenceTable t;   /* the single-row play buckets a K3,2,2,3 calibration leaves */
    t.add(key_shape(1, 3, 0, 1), 17.75);
    t.add(key_shape(1, 3, 3, 1), 15.5);
    t.add(key_shape(1, 2, 3, 1), 14.0);
    t.add(key_shape(1, 1, 3, 1), 13.0);
    t.add(key_shape(1, 0, 2, 1), 11.0);
    t.build();

    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    pred.a = 0.8f;
    f.view.k_ceiling = 3;
    f.view.pipeline_depth = 1;
    f.view.replay = DraftReplay::Play;
    f.add(0, 1, 64);
    draft_fake::predict(&pred, f.view, &f.pred);
    DraftPlanner planner;
    REQUIRE_EQ(planner.configure(cfg), RAD_OK);
    const DraftCostModel model{ &t.t, &DraftCostTable::estimate_thunk };

    CHECK_EQ(planner.choose(f.view, f.pred, model, &f.plan), RAD_OK);   /* frozen 0: (3,0) */
    CHECK_EQ(f.plan.max_k, 3);
    CHECK_EQ(planner.choose(f.view, f.pred, model, &f.plan), RAD_OK);   /* frozen 3: (3,3) */
    CHECK_EQ(f.plan.max_k, 3);

    /* Two tokens left: the row's cap is one draft, and the plan says so at once. */
    f.add(0, 1, 2);
    draft_fake::predict(&pred, f.view, &f.pred);
    CHECK_EQ(planner.choose(f.view, f.pred, model, &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 1);
    CHECK_EQ(planner.last_k(), 3);   /* the cap is not the incumbent */

    /* The next request starts with no frozen window and drafts at the incumbent again. */
    f.add(0, 2, 64);
    draft_fake::predict(&pred, f.view, &f.pred);
    for (int cycle = 0; cycle < 4; ++cycle) {
        CHECK_EQ(planner.choose(f.view, f.pred, model, &f.plan), RAD_OK);
        CHECK_EQ(f.plan.max_k, 3);
        CHECK(f.plan.source == DraftPlanSource::Cost);
    }
}

/* The horizon-score regression. Single-row ahead path, acceptance .5, ladder {3,6}, exact prices:
 * K3 stationary (3,3) 153.61, K3 after a frozen K6 (3,6) 191.84, any K6 192.09. K3 is better at
 * steady state (1.875/153.61 > 1.984375/192.09), but its first step down still pays the frozen K6
 * verify. Ranking E(3) against that execution price kept K6 forever; the horizon score ranks it
 * against the pass that verifies its own drafts. Prices the model lacks stay invalid. */
namespace {
int horizon_estimate(void*, const DraftVerifyShape& s, DraftCostEstimate* out) {
    *out = DraftCostEstimate{};
    if (s.draft_k == 3 && s.verify_k == 3) out->ms = 153.61;
    else if (s.draft_k == 3 && s.verify_k == 6) out->ms = 191.84;
    else if (s.draft_k == 6) out->ms = 192.09;
    else return RAD_OK;
    out->valid = true;
    out->coverage = 20;
    out->source = DraftCostSource::Measured;
    return RAD_OK;
}
DraftPolicyConfig horizon_config() {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 3; cfg.tiers[1] = 6; cfg.n_tiers = 2;
    cfg.fallback_k = 6;
    return cfg;
}
}  /* namespace */

TEST(the_pipeline_score_does_not_lock_into_an_inferior_stationary_depth) {
    Cycle c(6, horizon_config());
    c.pred.a = 0.5f;
    c.f.view.pipeline_depth = 1;
    c.add(0, 1, 10000);
    const DraftCostModel model{ nullptr, horizon_estimate };
    CHECK(1.875 / 153.61 > 1.984375 / 192.09);

    /* Cycle 1: nothing frozen, only K6 has an execution price (3,0 is not measured). */
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 6);
    /* Cycle 2: K3 runs (3,6) now and is ranked on (3,3); the plan still declares the frozen 6 and
     * reports the execution price it will run. */
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    CHECK_EQ(c.f.plan.max_k, 3);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(c.f.plan.verify_k, 6);
    CHECK_EQ(c.f.plan.history_tokens[0], 7);
    CHECK_NEAR(c.planner.last_price().ms, 191.84, 1e-9);
    CHECK_NEAR(c.planner.last_price().horizon_ms, 153.61, 1e-9);
    CHECK_NEAR(c.planner.last_tier_prices()[1].ms, 192.09, 1e-9);
    for (int n = 0; n < 30; ++n) {
        REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        CHECK_EQ(c.f.plan.max_k, 3);
        CHECK_EQ(c.f.plan.verify_k, 3);
    }
}

/* Pipeline 0: the step verifies what this cycle drafts, so the execution shape is the horizon
 * shape and exactly one estimate is asked per candidate. */
TEST(at_pipeline_zero_the_horizon_is_the_execution_shape) {
    Cycle c(6, horizon_config());
    c.pred.a = 0.5f;
    c.add(0, 1, 10000);
    ShapeLog log;
    const DraftCostModel model{ &log, log_estimate };
    for (int n = 0; n < 3; ++n) {
        log.n = 0;
        REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        for (int i = 0; i < log.n; ++i) CHECK_EQ(log.shape[i].verify_k, log.shape[i].draft_k);
        CHECK_NEAR(c.planner.last_price().horizon_ms, c.planner.last_price().ms, 1e-12);
    }
}

/* A missing stationary price is never substituted. A step down after a frozen K3 runs (2,3), and
 * that execution price is no stand-in for the missing (2,2) -- substituted, K2 would win here
 * (5.56 / 9.5 against 5.67 / 12.0). So K2 is not scored and K3 stays. A step up without its own
 * (3,3) is not scored either, and as the deepest gate that is the named baseline. */
TEST(a_missing_stationary_price_is_never_substituted) {
    DraftPolicyConfig cfg = cost_config(6);
    cfg.tiers[0] = 2; cfg.tiers[1] = 3; cfg.n_tiers = 2;
    cfg.fallback_k = 3;

    ReferenceTable t;   /* no (2,2) stationary bucket for the step down */
    t.add(key_shape(4, 3, 0, 1), 9.0);
    t.add(key_shape(4, 3, 3, 1), 12.0);
    t.add(key_shape(4, 2, 3, 1), 9.5);
    t.build();
    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    pred.a = 0.3f;   /* K2 would beat K3 on the execution price */
    four_rows(&f, &pred, /*depth=*/1);
    DraftPlanner planner;
    REQUIRE_EQ(planner.configure(cfg), RAD_OK);
    SpiedTable spy;
    CHECK(4 * (1 + .3 + .09) / 9.5 > 4 * (1 + .3 + .09 + .027) / 12.0);

    spy.reset(&t.t);
    REQUIRE_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 3);
    CHECK(spy.find(3, 3) != nullptr);   /* (3,0) runs, (3,3) scores it */

    for (int n = 0; n < 3; ++n) {
        spy.reset(&t.t);
        REQUIRE_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
        CHECK(spy.find(2, 2) != nullptr);   /* asked, and not measured */
        CHECK(spy.find(2, 3) != nullptr);   /* measured, and not a score */
        CHECK(!planner.last_tier_prices()[0].valid);
        CHECK_EQ(f.plan.max_k, 3);
        CHECK(f.plan.source == DraftPlanSource::Cost);
        CHECK_NEAR(planner.last_price().ms, 12.0, 1e-9);
        CHECK_NEAR(planner.last_price().horizon_ms, 12.0, 1e-9);
    }

    /* With (3,2) measured but (3,3) gone, the execution price alone is not a score for a raise. */
    ReferenceTable up;
    up.add(key_shape(4, 3, 2, 1), 10.0);
    up.build();
    DraftPlanner p2;
    REQUIRE_EQ(p2.configure(cfg), RAD_OK);
    p2.remember_published(1, 0, 1, 2);
    p2.remember_published(2, 1, 1, 2);
    p2.remember_published(3, 2, 1, 2);
    p2.remember_published(4, 3, 1, 2);
    spy.reset(&up.t);
    REQUIRE_EQ(p2.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK(spy.find(3, 2) != nullptr);
    CHECK(spy.find(3, 3) != nullptr);
    CHECK(f.plan.source == DraftPlanSource::Fallback);
    CHECK(!p2.last_tier_prices()[0].valid);
}

/* A row capped by its remaining output: both shapes use the capped K, so the horizon of a K1 row
 * after a frozen K3 is (1,1), and its yield is the capped one. */
TEST(a_short_row_is_scored_on_its_capped_stationary_shape) {
    DraftPolicyConfig cfg = cost_config(3);
    cfg.tiers[0] = 0; cfg.tiers[1] = 2; cfg.tiers[2] = 3; cfg.n_tiers = 3;
    ReferenceTable t;
    t.add(key_shape(1, 3, 0, 1), 17.75);
    t.add(key_shape(1, 3, 3, 1), 15.5);
    t.add(key_shape(1, 1, 3, 1), 13.0);
    t.add(key_shape(1, 1, 1, 1), 11.0);
    t.build();
    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    f.view.k_ceiling = 3;
    f.view.pipeline_depth = 1;
    f.view.replay = DraftReplay::Play;
    f.add(0, 1, 64);
    draft_fake::predict(&pred, f.view, &f.pred);
    DraftPlanner planner;
    REQUIRE_EQ(planner.configure(cfg), RAD_OK);
    SpiedTable spy;
    spy.reset(&t.t);
    REQUIRE_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 3);

    f.add(0, 1, 2);   /* cap 1 */
    draft_fake::predict(&pred, f.view, &f.pred);
    spy.reset(&t.t);
    REQUIRE_EQ(planner.choose(f.view, f.pred, spy.model(), &f.plan), RAD_OK);
    CHECK_EQ(f.plan.max_k, 1);
    CHECK_EQ(f.plan.verify_k, 3);
    CHECK(spy.find(1, 1) != nullptr);
    CHECK(spy.find(3, 3) == nullptr);   /* never the uncapped stationary shape */
    CHECK_NEAR(planner.last_price().ms, 13.0, 1e-9);
    CHECK_NEAR(planner.last_price().horizon_ms, 11.0, 1e-9);
}

/* Mixed rows: an old row frozen at K6 and a new one with no frozen window. The horizon shape is
 * per row -- each verifies its own K -- while the plan keeps the executed (6, 0) window. A model
 * that has no stationary price for the pair leaves the candidate unscored, so the gate falls back
 * instead of scoring the cheaper execution pass. */
namespace {
struct MixedModel {
    ShapeLog log;
    bool stationary = true;   /* false: every shape whose rows all verify their drafts is missing */
};
int mixed_estimate(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
    auto* m = static_cast<MixedModel*>(ctx);
    if (m->log.n < ShapeLog::kMax) m->log.shape[m->log.n++] = s;
    *out = DraftCostEstimate{};
    if (!m->stationary && s.n_rows == 2 && s.rows[1] == 2 && s.draft_k == s.verify_k)
        return RAD_OK;
    out->ms = 10.0 + 0.5 * s.verify_tokens + 0.1 * s.draft_k;
    out->valid = true;
    out->coverage = 8;
    out->source = DraftCostSource::Measured;
    return RAD_OK;
}
}  /* namespace */

TEST(mixed_rows_are_scored_on_each_rows_own_stationary_window) {
    for (int stationary = 1; stationary >= 0; --stationary) {
        Cycle c(6, horizon_config());
        c.pred.a = 0.9f;
        c.f.view.pipeline_depth = 1;
        c.add(0, 1, 10000);
        MixedModel m;
        const DraftCostModel model{ &m, mixed_estimate };
        REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        REQUIRE_EQ(c.f.plan.max_k, 6);

        c.add(1, 2, 10000);   /* a new request beside the K6 one */
        m.stationary = stationary != 0;
        m.log.n = 0;
        REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
        bool exec_seen = false, horizon_seen = false;
        for (int i = 0; i < m.log.n; ++i) {
            const DraftVerifyShape& s = m.log.shape[i];
            if (s.draft_k != 6 || s.draft_rows[6] != 2) continue;
            if (s.verify_k == 6 && s.rows[1] == 1 && s.verify_tokens == 8) exec_seen = true;
            if (s.verify_k == 6 && s.rows[1] == 2 && s.rows[6] == 2 && s.verify_tokens == 14 &&
                s.history_tokens == 14)
                horizon_seen = true;
        }
        CHECK(exec_seen);
        CHECK(horizon_seen);
        CHECK_EQ(c.f.plan.verify_k, 6);
        CHECK_EQ(c.f.plan.history_tokens[0], 7);
        CHECK_EQ(c.f.plan.history_tokens[1], 1);
        if (stationary) {
            CHECK(c.f.plan.source == DraftPlanSource::Cost);
            CHECK(c.planner.last_price().horizon_ms > c.planner.last_price().ms);
        } else {
            CHECK(c.f.plan.source == DraftPlanSource::Fallback);
            CHECK(!c.planner.last_tier_prices()[0].valid);
        }
    }
}

/* Mixed rows on a small budget. Every candidate's shapes differ (the new row has no frozen
 * window), so each needs two lookups: it gets both or none, and the search leaves the incumbent
 * and the issued price their lookups. Before, 10 calls went to the gate, K3 and both rows' K3
 * moves, the incumbent took the last two and the issued plan was left unpriced. */
namespace {
bool stationary(const DraftVerifyShape& s) {
    if (s.verify_k != s.draft_k) return false;
    for (int32_t j = 1; j <= s.draft_k; ++j)
        if (s.rows[j] != s.draft_rows[j]) return false;
    return true;
}
/* Every lookup is a pair: the horizon, then the execution shape of the same drafts. */
void check_pairs(const ShapeLog& log) {
    CHECK_EQ(log.n % 2, 0);
    for (int i = 0; i + 1 < log.n; i += 2) {
        CHECK(stationary(log.shape[i]));
        CHECK(!stationary(log.shape[i + 1]));
        CHECK_EQ(log.shape[i].draft_k, log.shape[i + 1].draft_k);
        for (int32_t j = 1; j <= log.shape[i].draft_k; ++j)
            CHECK_EQ(log.shape[i].draft_rows[j], log.shape[i + 1].draft_rows[j]);
    }
}
}  /* namespace */

TEST(a_small_budget_prices_mixed_rows_whole_and_keeps_the_post_search_calls) {
    DraftPolicyConfig cfg = horizon_config();
    cfg.max_evals = 10;
    cfg.hold = 0;
    Cycle c(6, cfg);
    c.pred.a = 0.3f;
    c.f.view.pipeline_depth = 1;
    c.add(0, 1, 10000);
    MixedModel m;
    const DraftCostModel model{ &m, mixed_estimate };

    /* One row, nothing frozen: the gate and K3 (4 lookups), then the issued K3 (2). */
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    REQUIRE_EQ(c.f.plan.max_k, 3);
    CHECK_EQ(m.log.n, 6);
    check_pairs(m.log);

    /* A new row beside it, and drafts that now pay: K6 wins and is raised over the K3 incumbent. */
    c.pred.a = 0.9f;
    c.add(1, 2, 10000);
    m.log.n = 0;
    REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
    check_pairs(m.log);
    CHECK_EQ(m.log.n, 8);   /* gate, K3, incumbent, issued: the search stopped at 4 of 10 */
    CHECK_EQ(c.f.plan.max_k, 6);   /* the incumbent was priced, so the raise could be judged */
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(c.planner.last_tiers_priced(), 2);
    REQUIRE(c.planner.last_price().valid);
    /* (6,(3,0)) runs: 10 + .5 * (4 + 1) + .6; (6,6)x2 scores it: 10 + .5 * 14 + .6. */
    CHECK_NEAR(c.planner.last_price().ms, 13.1, 1e-9);
    CHECK_NEAR(c.planner.last_price().horizon_ms, 17.6, 1e-9);
    const DraftVerifyShape& last = m.log.shape[m.log.n - 1];   /* the issued execution shape */
    CHECK_EQ(last.verify_k, 3);
    CHECK_EQ(last.verify_tokens, 5);
    CHECK_EQ(last.draft_k, 6);

    /* Below one pair nothing is priced: the named baseline, and no lookup was half-spent. */
    DraftPolicyConfig tiny = horizon_config();
    tiny.max_evals = 1;
    Cycle t(6, tiny);
    t.f.view.pipeline_depth = 1;
    t.add(0, 1, 10000);
    MixedModel tm;
    const DraftCostModel tmodel{ &tm, mixed_estimate };
    REQUIRE_EQ(t.planner.choose(t.f.view, t.f.pred, tmodel, &t.f.plan), RAD_OK);
    CHECK_EQ(tm.log.n, 0);
    CHECK(t.f.plan.source == DraftPlanSource::Fallback);
    CHECK(t.f.plan.fallback == DraftFallback::UnknownShape);

    /* At pipeline depth 0 one lookup is a whole price: the gate decides, the issued price starves. */
    Cycle z(6, tiny);
    z.add(0, 1, 10000);
    tm.log.n = 0;
    REQUIRE_EQ(z.planner.choose(z.f.view, z.f.pred, tmodel, &z.f.plan), RAD_OK);
    CHECK_EQ(tm.log.n, 1);
    CHECK(z.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(z.f.plan.max_k, 6);
    CHECK(!z.planner.last_price().valid);
}

/* Partial coverage at the deepest tier. The c8 trace ran a K0..K6 ladder whose stationary K6
 * (6,6) bucket was missing for the cycle's shape (`tiers=[6:invalid]`): the old gate then issued
 * the named K6 baseline although K0..K5 were priced. A missing deepest price is not evidence
 * about the cheaper tiers, so the priced alternative is chosen as a Cost plan; only a cycle with
 * no priced tier takes the baseline (still K6 here), and a priced K6 still wins on its score.
 * Horizon prices are the trace's; an execution shape whose windows differ costs 10 ms more. */
namespace {
const double kTraceH[7] = { 58.722, 98.737, 129.659, 153.613, 162.620, 180.832, 192.095 };
struct LadderModel {
    bool deep = true;   /* false: the stationary (6,6) shape is missing */
    bool any  = true;   /* false: nothing is priced */
};
int ladder_estimate(void* ctx, const DraftVerifyShape& s, DraftCostEstimate* out) {
    const auto* m = static_cast<const LadderModel*>(ctx);
    *out = DraftCostEstimate{};
    if (!m->any || (!m->deep && s.draft_k == 6 && s.verify_k == 6)) return RAD_OK;
    const int32_t hi = s.draft_k > s.verify_k ? s.draft_k : s.verify_k;
    out->ms = kTraceH[hi] + (s.draft_k != s.verify_k ? 10.0 : 0.0);
    out->valid = true;
    out->coverage = 20;
    out->source = DraftCostSource::Measured;
    return RAD_OK;
}
DraftPolicyConfig ladder_config() {
    DraftPolicyConfig cfg = cost_config(6);
    for (int32_t t = 0; t <= 6; ++t) cfg.tiers[t] = t;
    cfg.n_tiers = 7;
    cfg.fallback_k = 6;   /* the scheduler's exec_depth baseline */
    return cfg;
}
}  /* namespace */

TEST(a_missing_deepest_price_does_not_discard_the_priced_cheaper_tiers) {
    for (int32_t depth = 0; depth <= 1; ++depth) {
        /* a=.9: K6 (5.217/192.095) beats K5 (4.686/180.832), which beats every shallower tier. */
        CHECK(5.217031 / 192.095 > 4.68559 / 180.832);
        CHECK(4.68559 / 180.832 > 4.0951 / 162.620);

        LadderModel m;
        m.deep = false;
        const DraftCostModel model{ &m, ladder_estimate };
        Cycle c(6, ladder_config());
        c.pred.a = 0.9f;
        c.f.view.pipeline_depth = depth;
        c.add(0, 1, 10000);
        for (int n = 0; n < 4; ++n) {
            REQUIRE_EQ(c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan), RAD_OK);
            CHECK(c.f.plan.source == DraftPlanSource::Cost);
            CHECK(c.f.plan.fallback == DraftFallback::None);
            CHECK_EQ(c.f.plan.max_k, 5);
            CHECK_EQ(c.f.plan.k[0], 5);
            CHECK_EQ(c.f.plan.verify_k, depth == 0 || n > 0 ? 5 : 0);
            REQUIRE_EQ(c.planner.last_n_tier_prices(), 7);
            CHECK(!c.planner.last_tier_prices()[6].valid);   /* asked, and not priced */
            CHECK(c.planner.last_tier_prices()[5].valid);
            CHECK_EQ(c.planner.last_tiers_priced(), 6);
            REQUIRE(c.planner.last_price().valid);
            CHECK_NEAR(c.planner.last_price().horizon_ms, 180.832, 1e-9);
        }

        /* The same partial coverage under a token budget: the cut still applies to the chosen
         * tier and says so on the row; its own shape is priced, so it is still a Cost plan. */
        Cycle b(6, ladder_config());
        b.pred.a = 0.9f;
        b.f.view.pipeline_depth = depth;
        b.add(0, 1, 10000);
        b.f.view.token_budget = 4;
        REQUIRE_EQ(b.planner.choose(b.f.view, b.f.pred, model, &b.f.plan), RAD_OK);
        CHECK_EQ(b.f.plan.k[0], 4);
        CHECK(b.f.plan.why[0] == DraftFallback::Resources);
        CHECK(b.f.plan.source == DraftPlanSource::Cost);

        /* High acceptance with K6 priced: the deepest tier still wins. */
        LadderModel full;
        const DraftCostModel full_model{ &full, ladder_estimate };
        Cycle h(6, ladder_config());
        h.pred.a = 0.9f;
        h.f.view.pipeline_depth = depth;
        h.add(0, 1, 10000);
        REQUIRE_EQ(h.planner.choose(h.f.view, h.f.pred, full_model, &h.f.plan), RAD_OK);
        CHECK(h.f.plan.source == DraftPlanSource::Cost);
        CHECK_EQ(h.f.plan.max_k, 6);
        CHECK_EQ(h.planner.last_tiers_priced(), 7);

        /* Nothing priced: the named baseline at fallback_k, every tier listed as unpriced. */
        LadderModel none;
        none.any = false;
        const DraftCostModel none_model{ &none, ladder_estimate };
        Cycle u(6, ladder_config());
        u.pred.a = 0.9f;
        u.f.view.pipeline_depth = depth;
        u.add(0, 1, 10000);
        REQUIRE_EQ(u.planner.choose(u.f.view, u.f.pred, none_model, &u.f.plan), RAD_OK);
        CHECK(u.f.plan.source == DraftPlanSource::Fallback);
        CHECK(u.f.plan.fallback == DraftFallback::UnknownShape);
        CHECK_EQ(u.f.plan.k[0], 6);
        CHECK_EQ(u.planner.last_tiers_priced(), 0);
        REQUIRE_EQ(u.planner.last_n_tier_prices(), 7);
        for (int32_t t = 0; t < 7; ++t) CHECK(!u.planner.last_tier_prices()[t].valid);
        CHECK(!u.planner.last_price().valid);
    }
}

/* The switch priced ahead of its step. Every test above runs at pipeline_depth 0, where the pass
 * that verifies a candidate's drafts is the candidate's own shape: `ms` and `hms` are one number
 * and the rule reduces to a ratio of two prices. Ahead of the step they are two: the cycle the move
 * spends runs the *frozen* window's shape while committing the candidate's tokens, so the switch
 * must price the transition at what that shape costs and the rest of the dwell at the candidate's
 * own. Both shapes are asked here, which is the branch this test exists for. */
TEST(the_priced_switch_prices_the_frozen_window_its_transition_cycle_runs) {
    ShapeLog log;
    const DraftCostModel model{ &log, log_estimate };
    Cycle c;
    c.f.view.pipeline_depth = 1;
    c.pred.a = 0.95f;
    c.add(0, 1);
    auto choose = [&] {
        return c.planner.choose(c.f.view, c.f.pred, model, &c.f.plan);
    };

    /* Nothing is frozen yet, so the first cycle verifies nothing and every candidate is priced on
     * its own stationary shape. */
    REQUIRE_EQ(choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(c.f.plan.k[0], 6);
    bool saw_first_cycle = false;
    for (int i = 0; i < log.n; ++i)
        if (log.shape[i].draft_k == 6 && log.shape[i].verify_k == 0) saw_first_cycle = true;
    CHECK(saw_first_cycle);

    /* Lower acceptance: a reduction is immediate, and its depth is frozen for the next cycle. Not
     * prose (a = 0.2): at pipeline_depth 1 a K0 plan spends no verify at all, so K0 wins there and
     * the window the raise would be priced against is empty. */
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.8f;
    REQUIRE_EQ(choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    const int32_t shallow = c.f.plan.k[0];
    REQUIRE(shallow > 0);
    CHECK(shallow < 6);

    /* Back to code-like text: the raise is priced on both windows -- the frozen one this cycle
     * runs (draft six, verify the shallow window) and the stationary one it is raised to. */
    log.n = 0;
    for (int j = 0; j < RAD_SCHED_MAX_SPEC; ++j) c.f.pred.seq[0].a[j] = 0.95f;
    REQUIRE_EQ(choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK_EQ(c.f.plan.k[0], 6);
    bool transition = false, stationary = false;
    for (int i = 0; i < log.n; ++i) {
        if (log.shape[i].draft_k != 6) continue;
        if (log.shape[i].verify_k == shallow) transition = true;
        if (log.shape[i].verify_k == 6) stationary = true;
    }
    CHECK(transition);   /* the one cycle the move spends, at the window it really runs */
    CHECK(stationary);   /* and the dwell's steady shape, which is what the depth is worth */
}

RAD_TEST_MAIN()
