/* draft_reference_test.cpp -- the no-loan reference's contract, at the two seams that carry it:
 * the depth rule R in DraftPlanner::choose and the measurement-only lookup in DraftCostTable.
 *
 * From Review 1's contract (MAIN-6, K1-K5 and R):
 *
 *  - Coverage is not a tie-break. The deepest candidate a batch offers has to be priced; if it is
 *    not, the run takes its named baseline. A shallower tier that happens to be measured must not
 *    inherit the choice, or a missing measurement silently buys a flatter run and the plan reports
 *    a decision the data did not make. The first case is that rule, and on the pre-R code it fails:
 *    the policy picks the deepest *priced* tier and calls the plan Cost.
 *  - A one-depth profile is a plan, not a comparison. With a single measured depth the run executes
 *    exactly that depth and reports Cost; the reference prices the execution, it does not choose
 *    among alternatives it never saw. The second case pins that R does not turn a valid single-depth
 *    profile into a fallback.
 *  - Measured-only. A query planning deeper than any bucket ran has no price -- no interpolation,
 *    no depth extrapolation -- so the policy cannot be handed a number for work nothing measured.
 *  - The loaded profile is copied, never written back. The table learns into its own storage; the
 *    caller's DraftCostProfile stays exactly as it was read.
 *
 * The other half is the table's own contract under a no-loan basis: the loaded bucket is immutable
 * (a bracket is counted, never folded into mean/unc/n), the lookup is Measured-only (a live table's
 * interpolation step and heuristic fallback are not measurements), and the same content under a
 * live basis keeps learning -- the basis, not the numbers, selects the behaviour.
 *
 * The file includes no engine.h, so tests/CMakeLists.txt's glob builds it as a plain rad_core test
 * with no list entry.
 */
#include "rad_test.h"

#include "draft_fakes.h"
#include "sched/draft.h"
#include "sched/draft_cost.h"
#include "sched/draft_policy.h"

#include <nlohmann/json.hpp>

#include <string>

using namespace rad;

namespace {

constexpr int32_t kMax = RAD_SCHED_MAX_SPEC;

/* ---------------------------------------------------------------- policy: rule R */

struct Cycle {
    draft_fake::Fixture f;
    draft_fake::Predictor pred;
    draft_fake::CostModel costs;
    DraftPlanner planner;

    explicit Cycle(DraftPolicyConfig cfg) {
        f.view.k_ceiling = cfg.k_ceiling;
        CHECK_EQ(planner.configure(cfg), RAD_OK);
    }
    void add(int32_t i, uint64_t id) {
        f.add(i, id);
        draft_fake::predict(&pred, f.view, &f.pred);
    }
    int choose() {
        return planner.choose(f.view, f.pred, DraftCostModel{ &costs, draft_fake::estimate },
                              &f.plan);
    }
    /* Price exactly the depths in `priced`; every other depth has no measurement. */
    void measure_only(std::initializer_list<int32_t> priced) {
        costs.measured = true;
        for (int32_t k : priced) costs.ms[k] = 1.0 + 0.2 * k;
    }
};

DraftPolicyConfig cost_config(int n_tiers, std::initializer_list<int32_t> tiers) {
    DraftPolicyConfig c;
    c.mode = DraftPolicyMode::Cost;
    c.k_ceiling = 6;
    c.fallback_k = 1;               /* deliberately not one of the priced tiers */
    c.n_tiers = n_tiers;
    int32_t i = 0;
    for (int32_t t : tiers) c.tiers[(size_t)i++] = t;
    return c;
}

/* R: the deepest offered candidate is `tiers[n_tiers - 1]` (the ceiling here allows it). Tiers 0, 2
 * and 3 are measured, 6 is not. The missing deepest price is not evidence about the measured
 * tiers, so they are still compared and the best of them is a Cost plan; only a cycle that prices
 * no tier at all runs the named baseline. */
TEST(the_deepest_offered_candidate_unpriced_leaves_the_priced_tiers_compared) {
    Cycle c(cost_config(5, { 0, 2, 3, 4, 6 }));
    c.pred.a = 0.95f;
    c.add(0, 1);
    c.add(1, 2);
    c.measure_only({ 0, 2, 3 });

    CHECK_EQ(c.choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK(c.f.plan.fallback == DraftFallback::None);
    CHECK_EQ(c.f.plan.k[0], 3);   /* the measured tier 3, not cfg.fallback_k */
    CHECK_EQ(c.f.plan.k[1], 3);
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);

    Cycle none(cost_config(5, { 0, 2, 3, 4, 6 }));
    none.pred.a = 0.95f;
    none.add(0, 1);
    none.measure_only({ 1 });     /* priced, but not an offered tier */
    CHECK_EQ(none.choose(), RAD_OK);
    CHECK(none.f.plan.source == DraftPlanSource::Fallback);
    CHECK(none.f.plan.fallback == DraftFallback::UnknownShape);
    CHECK_EQ(none.f.plan.k[0], 1);   /* cfg.fallback_k */
}

/* The other side of R, and the state a reference calibration actually produces: one measured
 * depth. It runs, as a Cost plan at exactly that depth -- R must not read "no comparison was
 * possible" as "unpriced". */
TEST(a_one_tier_profile_runs_its_measured_depth) {
    Cycle c(cost_config(1, { 4 }));
    c.pred.a = 0.95f;
    c.add(0, 1);
    c.add(1, 2);
    c.measure_only({ 4 });

    CHECK_EQ(c.choose(), RAD_OK);
    CHECK(c.f.plan.source == DraftPlanSource::Cost);
    CHECK(c.f.plan.fallback == DraftFallback::None);
    CHECK_EQ(c.f.plan.k[0], 4);
    CHECK_EQ(c.f.plan.k[1], 4);
    CHECK_EQ(c.f.plan.max_k, 4);
    CHECK_EQ(draft_plan_valid(c.f.plan), RAD_OK);
    /* The one depth that was priced is the one the model saw, and nothing deeper was invented. */
    CHECK_EQ(c.costs.last.verify_k, 4);
}

/* ---------------------------------------------------------------- lookup: measured-only */

using nlohmann::json;

json fingerprint_json(const char* basis = "live") {
    json f{ { "model", "ref" }, { "quant", "q4" }, { "tp", 1 }, { "tp_wire", "w" },
            { "gpus", "g" }, { "kernels", "k" }, { "kv_format", "bf16" },
            { "placement", "plan" }, { "basis", basis }, { "arena", "none" } };
    /* Only a live basis names a loan; a no-loan reference has none to state. */
    if (std::string(basis) == "live") f["live_loan"] = "flex=0 lend=0";
    return f;
}

DraftCostRuntime runtime() {
    DraftCostRuntime r;
    r.fingerprint.model = "ref";
    r.fingerprint.quant = "q4";
    r.fingerprint.tp = 1;
    r.fingerprint.tp_wire = "w";
    r.fingerprint.gpus = "g";
    r.fingerprint.kernels = "k";
    r.fingerprint.kv_format = "bf16";
    r.fingerprint.placement = "plan";
    r.fingerprint.basis = DraftCostBasis::Live;
    r.fingerprint.arena = "none";
    r.fingerprint.live_loan = "flex=0 lend=0";
    return r;
}

/* One rectangular bucket at depth `k`, rows all through round k. */
json bucket_json(int32_t k, double mean_ms) {
    json rows = json::array();
    rows.push_back(0);                    /* index 0 is unused in both distributions */
    for (int j = 1; j <= k; ++j) rows.push_back(2);
    json origins = json::array();
    for (int32_t i = 0; i < 2; ++i) origins.push_back("head");   /* one per row */
    return json{
        { "ctx_class", 0 },
        { "shape", { { "n_rows", 2 }, { "verify_k", k },
                     { "verify_tokens", 2 * (k + 1) }, { "history_tokens", 0 },
                     { "prefill_tokens", 0 }, { "pipeline_depth", 0 },
                     { "rows", rows }, { "draft_k", k }, { "draft_rows", rows },
                     { "origins", origins }, { "sampling", "greedy" }, { "replay", "play" },
                     { "drafter", "serial" } } },
        { "extent", { { "rounds", k }, { "draft_rows_total", 2 * k },
                      { "draft_round_rows", rows } } },
        { "time_ms", { { "mean", mean_ms }, { "unc", 0.5 }, { "n", 5 } } },
        { "flags", json::array() } };
}

/* A query at the plan's own distribution: the same rows, the same depth. */
DraftVerifyShape shape_at(int32_t k) {
    DraftVerifyShape s;
    s.ctx_class = 0;
    s.n_rows = 2;
    s.verify_k = k;
    for (int j = 1; j <= k; ++j) s.rows[j] = 2;
    s.verify_tokens = 2 * (k + 1);
    s.replay = DraftReplay::Play;
    s.drafter = DraftDrafter::Serial;
    s.pipeline_depth = 0;
    s.draft_k = k;
    for (int j = 1; j <= k; ++j) s.draft_rows[j] = 2;
    return s;
}

std::string profile_text(const json& bucket, const char* basis = "live") {
    return json{ { "format", "rad-draft-profile/4" }, { "fingerprint", fingerprint_json(basis) },
                 { "execution", "rectangular" }, { "buckets", json::array({ bucket }) } }
        .dump();
}

DraftCostProfile profile_at(int32_t k, double mean_ms, const char* basis = "live") {
    DraftCostProfile p;
    std::string err;
    const int rc =
        draft_cost_profile_parse(profile_text(bucket_json(k, mean_ms), basis), &p, &err);
    CHECK_EQ(rc, RAD_OK);
    if (rc < 0) fprintf(stderr, "  parse error: %s\n", err.c_str());
    return p;
}

/* The same context class and the same executed draft distribution, but a different verify window:
 * a live table borrows a scaled bound from the measured bucket, which a reference must not do. */
DraftVerifyShape wider_verify_shape() {
    DraftVerifyShape s = shape_at(2);
    s.n_rows = 3;
    s.verify_tokens = 3 + 2 + 2;
    return s;
}

/* One completed cycle at the single bucket's own shape (two rows, K=2), `total_ms` long. */
int observe_once(DraftCostTable& t, double total_ms) {
    DraftPlan plan;
    std::vector<int32_t> k(2, 2), ctx(2, 0), hist(2, 0);
    std::vector<DraftDrafter> fill(2, DraftDrafter::Serial);
    std::vector<uint64_t> req{ 1, 2 };
    std::vector<int32_t> slot{ 0, 1 };
    std::vector<uint32_t> gen{ 1, 1 };
    int32_t rows[kMax + 1] = {};
    rows[1] = 2;
    rows[2] = 2;
    plan.cycle_id = 1;
    plan.policy_version = 1;
    plan.n_seq = 2;
    plan.cap = 2;
    plan.k_ceiling = 6;
    plan.max_k = 2;
    plan.verify_k = 2;
    plan.replay = DraftReplay::Play;
    plan.timing_valid = true;
    plan.req_id = req.data();
    plan.slot = slot.data();
    plan.slot_gen = gen.data();
    plan.k = k.data();
    plan.fill = fill.data();
    plan.ctx_class = ctx.data();
    plan.history_tokens = hist.data();
    plan.rows = rows;
    if (draft_plan_valid(plan) != RAD_OK) return RAD_E_INVAL;

    DraftObservation o;
    o.cycle_id = 1;
    o.policy_version = 1;
    o.n_seq = 2;
    o.cap = 2;
    o.plan = &plan;
    o.req_id = plan.req_id;
    o.slot = plan.slot;
    o.slot_gen = plan.slot_gen;
    o.complete = true;

    DraftCostSample s;
    s.cycle_id = 1;
    s.policy_version = 1;
    s.shape = shape_at(2);
    s.replay = DraftReplay::Play;
    s.total_ms = total_ms;
    s.flags = kDraftSampleValid;
    s.rounds = 2;
    for (int j = 1; j <= 2; ++j) s.draft_round_rows[j] = 2;
    s.draft_rows_total = 4;
    return t.observe(o, s);
}

}  /* namespace */

/* The reference measures one execution; a deeper plan is not that execution. Handing it the
 * shallower number would price draft rounds nothing ran, and the policy would then have a price for
 * a candidate the reference does not cover. */
TEST(a_query_deeper_than_any_bucket_ran_has_no_price) {
    DraftCostProfile prof = profile_at(2, 4.0);
    DraftCostTable t;
    CHECK_EQ(t.configure(prof, runtime()), RAD_OK);
    REQUIRE(t.active());

    DraftCostEstimate exact;
    CHECK_EQ(t.estimate(shape_at(2), &exact), RAD_OK);
    CHECK(exact.valid);
    CHECK(exact.source == DraftCostSource::Measured);
    CHECK_EQ(exact.coverage, 5);

    DraftCostEstimate deeper;
    CHECK_EQ(t.estimate(shape_at(4), &deeper), RAD_OK);
    CHECK(!deeper.valid);   /* no depth extrapolation, not even a named fallback */
}

/* The profile is copied at configure(): learning lands in the table's own bucket, and the object the
 * caller read is untouched. A reference that could be rewritten by the run it prices is not a
 * reference. */
TEST(the_loaded_profile_is_copied_and_never_written_back) {
    DraftCostProfile prof = profile_at(2, 4.0);
    REQUIRE_EQ(prof.buckets.size(), (size_t)1);
    const double mean = prof.buckets[0].mean_ms;
    const double unc = prof.buckets[0].unc_ms;
    const int32_t n = prof.buckets[0].n;

    DraftCostTable t;
    CHECK_EQ(t.configure(prof, runtime()), RAD_OK);
    REQUIRE(t.active());

    /* One completed cycle at the bucket's own shape, twice as slow. */
    CHECK_EQ(observe_once(t, 8.0), RAD_OK);

    /* The table learned into its copy ... */
    REQUIRE_EQ(t.buckets().size(), (size_t)1);
    CHECK(t.buckets()[0].mean_ms > mean);
    CHECK_EQ(t.buckets()[0].n, n + 1);
    /* ... and the profile this test still holds is exactly as it was read. */
    CHECK(prof.buckets[0].mean_ms == mean);
    CHECK(prof.buckets[0].unc_ms == unc);
    CHECK_EQ(prof.buckets[0].n, n);
}

/* A reference is not a starting point for online learning: the file's bucket is what it measured,
 * and a run it prices is not evidence about it. The bracket here is a valid observation of the
 * bucket's own shape -- the same sample the live case below folds in -- so only the basis stops it.
 * n and the width are checked with the mean: a "learning" that counted a sample without moving the
 * mean would still misstate the reference's coverage and uncertainty. */
TEST(a_reference_observation_leaves_the_loaded_bucket_untouched) {
    DraftCostProfile prof = profile_at(2, 4.0, "no-loan");
    REQUIRE_EQ(prof.buckets.size(), (size_t)1);
    const double mean = prof.buckets[0].mean_ms;
    const double unc = prof.buckets[0].unc_ms;
    const int32_t n = prof.buckets[0].n;

    DraftCostTable t;
    CHECK_EQ(t.configure(prof, runtime()), RAD_OK);
    REQUIRE(t.active());
    REQUIRE(t.reference());
    CHECK_EQ(t.reference_skipped(), 0);

    CHECK_EQ(observe_once(t, 8.0), RAD_OK);

    REQUIRE_EQ(t.buckets().size(), (size_t)1);
    CHECK(t.buckets()[0].mean_ms == mean);
    CHECK(t.buckets()[0].unc_ms == unc);
    CHECK_EQ(t.buckets()[0].n, n);
    CHECK_EQ(t.reference_skipped(), 1);
}

/* The same bucket content and the same bracket under a live basis: it learns. The basis is the only
 * difference between this table and the one above, so a reference that learned (or a live profile
 * that did not) would fail here rather than pass as a plausible default. */
TEST(the_same_bucket_under_a_live_basis_keeps_learning) {
    DraftCostProfile prof = profile_at(2, 4.0);
    REQUIRE_EQ(prof.buckets.size(), (size_t)1);
    const double mean = prof.buckets[0].mean_ms;
    const int32_t n = prof.buckets[0].n;

    DraftCostTable t;
    CHECK_EQ(t.configure(prof, runtime()), RAD_OK);
    REQUIRE(t.active());
    CHECK(!t.reference());
    CHECK_EQ(t.reference_skipped(), 0);

    CHECK_EQ(observe_once(t, 8.0), RAD_OK);

    REQUIRE_EQ(t.buckets().size(), (size_t)1);
    CHECK(t.buckets()[0].mean_ms > mean);
    CHECK_EQ(t.buckets()[0].n, n + 1);
}

/* Measured-only, at the lookup seam: the bucket's own shape is priced, every other shape is not.
 * The two refused queries are chosen to be ones the live table prices -- the neighbouring context
 * class (interpolation) and an anchor of the same class and extent with a wider verify window
 * (heuristic fallback) -- so the check is about the basis and not about a shape the reader could
 * not match anyway. Both tables are built from identical bucket content. */
TEST(a_reference_prices_only_the_shape_it_measured) {
    DraftCostProfile ref_prof = profile_at(2, 4.0, "no-loan");
    DraftCostProfile live_prof = profile_at(2, 4.0);

    DraftCostTable ref;
    DraftCostTable live;
    CHECK_EQ(ref.configure(ref_prof, runtime()), RAD_OK);
    CHECK_EQ(live.configure(live_prof, runtime()), RAD_OK);
    REQUIRE(ref.active());
    REQUIRE(live.active());
    REQUIRE(ref.reference());
    REQUIRE(!live.reference());

    DraftVerifyShape other_ctx = shape_at(2);
    other_ctx.ctx_class = 1;

    /* The measured shape is a price in both. */
    DraftCostEstimate ref_exact, live_exact;
    CHECK_EQ(ref.estimate(shape_at(2), &ref_exact), RAD_OK);
    CHECK_EQ(live.estimate(shape_at(2), &live_exact), RAD_OK);
    CHECK(ref_exact.valid);
    CHECK(live_exact.valid);
    CHECK(ref_exact.source == DraftCostSource::Measured);
    CHECK(live_exact.source == DraftCostSource::Measured);
    CHECK(ref_exact.ms == live_exact.ms);

    /* A live table interpolates the neighbouring class; a reference has no measurement to report. */
    DraftCostEstimate ref_near, live_near;
    CHECK_EQ(ref.estimate(other_ctx, &ref_near), RAD_OK);
    CHECK_EQ(live.estimate(other_ctx, &live_near), RAD_OK);
    CHECK(!ref_near.valid);
    CHECK(live_near.valid);
    CHECK(live_near.source == DraftCostSource::Interpolated);

    /* A live table scales an anchor across the wider verify window; a reference does not, even
     * though the anchor ran the same executed draft distribution. */
    DraftCostEstimate ref_scaled, live_scaled;
    CHECK_EQ(ref.estimate(wider_verify_shape(), &ref_scaled), RAD_OK);
    CHECK_EQ(live.estimate(wider_verify_shape(), &live_scaled), RAD_OK);
    CHECK(!ref_scaled.valid);
    CHECK(live_scaled.valid);
    CHECK(live_scaled.source == DraftCostSource::ConservativeFallback);
    CHECK(live_scaled.ms > live_exact.ms);   /* the ratio is never below 1 */
}

RAD_TEST_MAIN()
