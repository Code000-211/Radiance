/* draft_cost_test.cpp -- strand A: the profile reader, the lookup and the sample replay
 * (core/sched/draft_cost.h), on synthetic profiles only.
 *
 * What is checked is what the policy may rely on: a profile is refused by name when it is not
 * this format or not this deployment, an exact bucket is Measured, a missing shape is one
 * documented interpolation or a named conservative bound -- never a unitless prior labelled
 * milliseconds -- and only an attributable sample moves a measured bucket. The shapes cover K0,
 * a depth change under overlap, and the C8 windows whose N_j differ at the same depth.
 */
#include "rad_test.h"

#include "sched/draft_cost.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

using namespace rad;
using nlohmann::json;

namespace {

json fingerprint_json() {
    return json{ { "model", "Qwen3.8-Flash-Next" }, { "quant", "gguf-q4_k_m" }, { "tp", 2 },
                 { "tp_wire", "wht6" }, { "gpus", "2x Radeon AI PRO R9700" },
                 { "kernels", "libr4d test+gfx1201" }, { "kv_format", "bf16" },
                 { "placement", "all_vram" } };
}

DraftCostFingerprint runtime_fingerprint() {
    DraftCostFingerprint f;
    f.model = "Qwen3.8-Flash-Next";
    f.quant = "gguf-q4_k_m";
    f.tp = 2;
    f.tp_wire = "wht6";
    f.gpus = "2x Radeon AI PRO R9700";
    f.kernels = "libr4d test+gfx1201";
    f.kv_format = "bf16";
    f.placement = "all_vram";
    return f;
}

DraftCostRuntime runtime() {
    DraftCostRuntime r;
    r.fingerprint = runtime_fingerprint();
    return r;
}

/* verify_tokens follows from the other two counts, so the helper derives it. */
int32_t token_count(int32_t n_rows, const std::vector<int32_t>& rows) {
    int32_t t = n_rows;
    for (size_t j = 1; j < rows.size(); ++j) t += rows[j];
    return t;
}

json bucket_json(int32_t ctx, int32_t n_rows, int32_t verify_k, std::vector<int32_t> rows,
                 double mean_ms, double unc_ms = 0.5, int32_t n = 5,
                 std::vector<std::string> flags = {}, const char* replay = "play",
                 int32_t history = 0) {
    json r = json::array();
    for (int32_t v : rows) r.push_back(v);
    json b{ { "ctx_class", ctx },
            { "shape", { { "n_rows", n_rows }, { "verify_k", verify_k },
                         { "verify_tokens", token_count(n_rows, rows) },
                         { "history_tokens", history }, { "prefill_tokens", 0 },
                         { "rows", r }, { "sampling", "greedy" }, { "replay", replay } } },
            { "extent", { { "rounds", verify_k }, { "draft_rows", token_count(n_rows, rows) - n_rows } } },
            { "time_ms", { { "mean", mean_ms }, { "unc", unc_ms }, { "n", n } } } };
    if (!flags.empty()) b["flags"] = flags;
    return b;
}

std::string profile_text(const std::vector<json>& buckets) {
    json j{ { "format", "rad-draft-profile/1" }, { "fingerprint", fingerprint_json() },
            { "execution", "rectangular" }, { "buckets", json::array() } };
    for (const json& b : buckets) j["buckets"].push_back(b);
    return j.dump();
}

DraftCostProfile parse_or_fail(const std::string& text) {
    DraftCostProfile p;
    std::string err;
    const int rc = draft_cost_profile_parse(text, &p, &err);
    CHECK_EQ(rc, RAD_OK);
    if (rc < 0) fprintf(stderr, "  parse error: %s\n", err.c_str());
    return p;
}

DraftVerifyShape query(int32_t ctx, int32_t n_rows, int32_t verify_k, std::vector<int32_t> rows,
                       int32_t history = 0) {
    DraftVerifyShape s;
    s.ctx_class = ctx;
    s.n_rows = n_rows;
    s.verify_k = verify_k;
    for (size_t j = 0; j < rows.size() && j <= (size_t)verify_k; ++j) s.rows[j] = rows[j];
    s.verify_tokens = token_count(n_rows, rows);
    s.history_tokens = history;
    s.replay = DraftReplay::Play;
    return s;
}

DraftCostSample sample_of(const DraftVerifyShape& s, double ms,
                          uint16_t flags = kDraftSampleValid) {
    DraftCostSample c;
    c.cycle_id = 1;
    c.policy_version = 1;
    c.shape = s;
    c.total_ms = ms;
    c.flags = flags;
    c.rounds = s.verify_k;
    c.draft_rows = s.verify_tokens - s.n_rows;
    return c;
}

/* The plan the observation names. max_k defaults to the verify window; an issue_ahead cycle
 * passes its own draft depth, which is what makes verify_k != max_k legal. */
DraftPlan plan_for(const DraftVerifyShape& s, uint64_t cycle = 1, uint32_t version = 1,
                   int32_t max_k = -1) {
    DraftPlan p;
    p.cycle_id = cycle;
    p.policy_version = version;
    p.n_seq = s.n_rows;
    p.cap = s.n_rows;
    p.k_ceiling = s.verify_k;
    p.verify_k = s.verify_k;
    p.max_k = max_k < 0 ? s.verify_k : max_k;
    return p;
}

DraftObservation completed_cycle(const DraftPlan& p) {
    DraftObservation o;
    o.cycle_id = p.cycle_id;
    o.policy_version = p.policy_version;
    o.n_seq = p.n_seq;
    o.plan = &p;
    o.complete = true;
    return o;
}

}  /* namespace */

TEST(the_reader_accepts_the_documented_profile) {
    /* The format example from docs/ADAPTIVE-MTP.md, verbatim, to check field handling. Its 43.0 ms
     * is the example's own value -- a format sample, not the measured C8 K3 figure; the documented
     * C8 numbers are replayed in the_documented_c8_gradient_is_reproduced_without_a_new_measurement
     * and the two must not be conflated. */
    const std::string text = R"({
      "format": "rad-draft-profile/1",
      "fingerprint": {
        "model": "Qwen3.8-Flash-Next", "quant": "gguf-q4_k_m", "tp": 2, "tp_wire": "wht6",
        "gpus": "2x Radeon AI PRO R9700", "kernels": "libr4d test+gfx1201",
        "kv_format": "bf16", "placement": "all_vram"
      },
      "execution": "rectangular",
      "buckets": [{
        "ctx_class": 0,
        "shape": { "n_rows": 8, "verify_k": 3, "verify_tokens": 32, "history_tokens": 0,
                   "prefill_tokens": 0, "rows": [0, 8, 8, 8], "sampling": "greedy",
                   "replay": "play" },
        "extent": { "rounds": 3, "draft_rows": 24 },
        "time_ms": { "mean": 43.0, "unc": 0.6, "n": 5 },
        "components_ms": { "target": 25.9, "draft": 12.1, "host": 4.2, "valid": true },
        "flags": ["ahead"]
      }]
    })";
    const DraftCostProfile p = parse_or_fail(text);
    CHECK_EQ((int)p.buckets.size(), 1);
    CHECK(p.fingerprint.tp == 2);
    CHECK(p.execution == DraftCostExecution::Rectangular);
    const DraftCostBucket& b = p.buckets[0];
    CHECK_EQ(b.shape.verify_k, 3);
    CHECK_EQ(b.shape.verify_tokens, 32);
    CHECK_EQ(b.mean_ms, 43.0);
    CHECK_EQ(b.n, 5);
    CHECK_EQ(b.rounds, 3);
    CHECK_EQ(b.draft_rows, 24);
    CHECK((b.flags & kDraftSampleAhead) != 0);
    /* The format carries the ahead flag, not how many samples were ahead: a loaded bucket claims
     * no online count, and the flag alone keeps it visible. */
    CHECK_EQ(b.n_ahead, 0);
    CHECK(b.target_ms > 0.0 && b.draft_ms > 0.0 && b.host_ms > 0.0);
}

TEST(the_reader_refuses_a_foreign_format_and_inconsistent_counts) {
    std::string err;
    DraftCostProfile p;

    const std::string v2 = R"({"format":"rad-draft-profile/2","fingerprint":{},"execution":"rectangular","buckets":[]})";
    CHECK_EQ(draft_cost_profile_parse(v2, &p, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "rad-draft-profile/2"));

    /* verify_tokens disagreeing with n_rows + sum(rows) is not a measurement of this shape. */
    json bad = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    bad["shape"]["verify_tokens"] = 31;
    CHECK_EQ(draft_cost_profile_parse(profile_text({ bad }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "verify_tokens"));

    /* A bucket with no samples is not a measurement. */
    json none = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0, 0.5, 0);
    CHECK_EQ(draft_cost_profile_parse(profile_text({ none }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "time_ms.n"));
}

TEST(the_reader_refuses_nan_infinity_and_duplicate_buckets) {
    std::string err;
    DraftCostProfile p;

    std::string inf = profile_text({ bucket_json(0, 4, 1, { 0, 4 }, 1.0) });
    inf.replace(inf.find("\"mean\":1.0"), 10, "\"mean\":1e999");
    CHECK_EQ(draft_cost_profile_parse(inf, &p, &err), RAD_E_INVAL);
    CHECK(has(err, "finite") || has(err, "JSON"));   /* a number out of double's range */

    const json a = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    CHECK_EQ(draft_cost_profile_parse(profile_text({ a, a }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "duplicates"));
}

TEST(a_fingerprint_or_execution_mismatch_disables_the_profile) {
    const DraftCostProfile p = parse_or_fail(
        profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));

    DraftCostRuntime other = runtime();
    other.fingerprint.kernels = "libr4d deadbeef+gfx1201";
    DraftCostTable t;
    CHECK_EQ(t.configure(p, other), RAD_OK);   /* refused, not an error */
    CHECK(!t.active());
    CHECK(has(t.reason(), "kernels"));
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK(e.source == DraftCostSource::ConservativeFallback);
    CHECK_EQ(e.coverage, 0);

    DraftCostRuntime shrinking = runtime();
    shrinking.execution = DraftCostExecution::Shrinking;
    CHECK_EQ(t.configure(p, shrinking), RAD_OK);
    CHECK(!t.active());
    CHECK(has(t.reason(), "shrinking"));

    CHECK_EQ(t.configure(p, runtime()), RAD_OK);
    CHECK(t.active());
}

TEST(an_exact_bucket_is_measured_and_carries_its_coverage) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 },
                                                              43.0, 0.6, 7) })),
                         runtime()),
             RAD_OK);
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(e.valid);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 43.0);
    CHECK_EQ(e.unc_ms, 0.6);
    CHECK_EQ(e.coverage, 7);
}

TEST(a_missing_bucket_is_one_interpolation_or_a_named_bound) {
    /* One bucket: C8 rectangular K3 at context class 0. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 },
                                                                43.0, 0.6, 7) })),
                         runtime()),
             RAD_OK);

    /* The neighbouring context class, same shape: Interpolated, uncertain by the class step. */
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(1, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(e.valid);
    CHECK(e.source == DraftCostSource::Interpolated);
    CHECK_EQ(e.ms, 43.0);
    CHECK(std::fabs(e.unc_ms - std::sqrt(0.6 * 0.6 + 4.3 * 4.3)) < 1e-9);
    CHECK_EQ(e.coverage, 7);

    /* A deeper shape at the same class: a bound above the anchor, no coverage claimed. */
    CHECK_EQ(t.estimate(query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }), &e), RAD_OK);
    CHECK(e.valid);
    CHECK(e.source == DraftCostSource::ConservativeFallback);
    CHECK_EQ(e.coverage, 0);
    CHECK(e.ms > 43.0);
    CHECK(e.unc_ms > 0.0);

    /* Two classes away with nothing measured between: no anchor, no number. */
    CHECK_EQ(t.estimate(query(3, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.ms, 0.0);
}

TEST(k0_is_a_plain_decode_step_with_its_own_bucket) {
    const json k0 = bucket_json(0, 8, 0, { 0 }, 11.0, 0.2, 9);
    const json k3 = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0, 0.6, 7);
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ k0, k3 })), runtime()), RAD_OK);

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 8, 0, { 0 }), &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 11.0);

    /* K0 with the drafter loaded carries history; it is a different measurement from K0 in a
     * process without one, so it must not be priced by the plain K0 bucket. */
    const json warm = bucket_json(0, 8, 0, { 0 }, 14.0, 0.3, 4, {}, "play", 512);
    DraftCostTable h;
    CHECK_EQ(h.configure(parse_or_fail(profile_text({ k0, warm })), runtime()), RAD_OK);
    CHECK_EQ(h.estimate(query(0, 8, 0, { 0 }, 512), &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 14.0);
    CHECK_EQ(h.estimate(query(0, 8, 0, { 0 }, 0), &e), RAD_OK);
    CHECK_EQ(e.ms, 11.0);
}

TEST(a_c8_mix_is_priced_by_its_rows_not_only_by_its_depth) {
    /* Same n_rows and verify_k, different N_j: 4 requests at K=6 versus 8 requests, 6 of them at
     * K=6 and 2 at K=2. A key of (rows, max K) would price these the same. */
    const json rect = bucket_json(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }, 61.0);
    const json mix  = bucket_json(0, 8, 6, { 0, 8, 8, 2, 2, 2, 2 }, 48.0);
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ rect, mix })), runtime()), RAD_OK);

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }), &e), RAD_OK);
    CHECK_EQ(e.ms, 61.0);
    CHECK_EQ(t.estimate(query(0, 8, 6, { 0, 8, 8, 2, 2, 2, 2 }), &e), RAD_OK);
    CHECK_EQ(e.ms, 48.0);
}

TEST(only_an_attributable_sample_updates_a_measured_bucket) {
    DraftCostProfile empty = parse_or_fail(profile_text({}));
    DraftCostTable t;
    CHECK_EQ(t.configure(empty, runtime()), RAD_OK);
    CHECK(t.active());

    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const DraftPlan p3 = plan_for(k3);
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 40.0)), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);

    /* A recording bracket is kept, flagged, and never folded into the measured bucket. */
    CHECK_EQ(t.observe(completed_cycle(p3),
                       sample_of(k3, 200.0, kDraftSampleValid | kDraftSampleRecord)),
             RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 2);
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK_EQ(e.ms, 40.0);
    CHECK_EQ(e.coverage, 1);
    bool saw_record = false;
    for (const DraftCostBucket& b : t.buckets())
        if ((b.flags & kDraftSampleRecord) && b.flags == kDraftSampleRecord) saw_record = true;
    CHECK(saw_record);

    /* A sample that did not time its shape teaches nothing. */
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 90.0, kDraftSampleAhead)), RAD_OK);
    CHECK_EQ(t.ignored(), 1);
    CHECK_EQ((int)t.buckets().size(), 2);
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK_EQ(e.ms, 40.0);

    /* An ahead sample whose bracket the caller checked may update; it stays visible. */
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 44.0, kDraftSampleValid | kDraftSampleAhead)),
             RAD_OK);
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK_EQ(e.ms, 42.0);
    CHECK_EQ(e.coverage, 2);
    CHECK_EQ((int)t.buckets().size(), 2);
    int32_t ahead = 0;
    for (const DraftCostBucket& b : t.buckets()) ahead += b.n_ahead;
    CHECK_EQ(ahead, 1);
}

TEST(a_depth_change_under_overlap_lands_in_its_own_bucket) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);

    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const DraftVerifyShape k6 = query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 });
    const DraftPlan p3 = plan_for(k3);
    const DraftPlan p6 = plan_for(k6);
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 30.0)), RAD_OK);
    CHECK_EQ(t.observe(completed_cycle(p6), sample_of(k6, 60.0, kDraftSampleValid | kDraftSampleAhead)),
             RAD_OK);

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK_EQ(e.ms, 30.0);
    CHECK_EQ(e.coverage, 1);   /* the K6 sample did not move it */
    CHECK_EQ(t.estimate(k6, &e), RAD_OK);
    CHECK_EQ(e.ms, 60.0);
    CHECK_EQ(e.coverage, 1);
}

TEST(an_incomplete_or_inconsistent_observation_is_refused) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const DraftPlan p3 = plan_for(k3);

    DraftObservation incomplete = completed_cycle(p3);
    incomplete.complete = false;
    CHECK_EQ(t.observe(incomplete, sample_of(k3, 40.0)), RAD_E_INVAL);

    /* A NaN or a non-positive bracket must not enter the mean. */
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, std::nan(""))), RAD_E_INVAL);
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, -1.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());
}

TEST(an_unattributed_identity_is_refused) {
    /* A sample with no cycle or policy identity cannot be traced to a measurement; folding it in
     * would mix cycles the model cannot tell apart. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });

    const DraftPlan p3 = plan_for(k3);
    DraftCostSample zero_cycles = sample_of(k3, 40.0);
    zero_cycles.cycle_id = 0;
    CHECK_EQ(t.observe(completed_cycle(p3), zero_cycles), RAD_E_INVAL);
    DraftCostSample zero_version = sample_of(k3, 40.0);
    zero_version.policy_version = 0;
    CHECK_EQ(t.observe(completed_cycle(p3), zero_version), RAD_E_INVAL);

    /* An observation at 5/4 and a sample at 0/0 do not describe the same measurement either. */
    const DraftPlan other = plan_for(k3, 5, 4);
    DraftObservation o5 = completed_cycle(other);
    CHECK_EQ(t.observe(o5, zero_cycles), RAD_E_INVAL);
    /* And an identity that disagrees with the observation is a wiring error. */
    CHECK_EQ(t.observe(o5, sample_of(k3, 40.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());
}

TEST(a_sample_must_match_the_plan_that_produced_it) {
    /* Under issue_ahead the verify window belongs to the paired step, not to this cycle's draft
     * depth. A shape filled from the current plan (verify_k 3, max_k 6) is self-consistent and
     * false: the plan says the verified window was 3, the sample claims 6. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const DraftVerifyShape k6 = query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 });

    const DraftPlan ahead_plan = plan_for(k3, 1, 1, 6);   /* verifies 3, drafts 6 */
    CHECK_EQ(t.observe(completed_cycle(ahead_plan), sample_of(k6, 60.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());

    /* A cycle that did draft 6 rounds while verifying 3 is the legal shape of the same cycle. */
    DraftCostSample ahead = sample_of(k3, 40.0);
    ahead.rounds = 6;
    ahead.draft_rows = 48;
    CHECK_EQ(t.observe(completed_cycle(ahead_plan), ahead), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);
    CHECK_EQ(t.buckets()[0].shape.verify_k, 3);
    CHECK_EQ(t.buckets()[0].rounds, 6);

    /* The row count is part of the same identity: a consistent 4-row shape from an 8-row plan. */
    DraftCostTable h;
    CHECK_EQ(h.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape four = query(0, 4, 3, { 0, 4, 4, 4 });
    CHECK_EQ(h.observe(completed_cycle(plan_for(k3)), sample_of(four, 40.0)), RAD_E_INVAL);
    /* A draft extent deeper than the plan's own max_k cannot have run. */
    const DraftPlan shallow = plan_for(k3, 1, 1, 2);
    CHECK_EQ(h.observe(completed_cycle(shallow), sample_of(k3, 40.0)), RAD_E_INVAL);
    CHECK(h.buckets().empty());
}

TEST(an_impossible_or_inconsistent_shape_is_refused) {
    /* rows[0] unused and the deepest verified position must have run a row: [0,0,0,0] is not a
     * K3 window, it is a K0 window wearing verify_k = 3. */
    DraftCostProfile p;
    std::string err;
    json degenerate = bucket_json(0, 8, 3, { 0, 0, 0, 0 }, 40.0);
    degenerate["extent"] = { { "rounds", 1 }, { "draft_rows", 3 } };
    CHECK_EQ(draft_cost_profile_parse(profile_text({ degenerate }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "deepest verified position"));

    /* estimate() holds a query to the same invariants, not just the reader. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    DraftVerifyShape bad = query(0, 8, 3, { 0, 8, 8, 8 });
    bad.rows[3] = 0;
    bad.verify_tokens = 24;   /* n_rows + rows[1] + rows[2] */
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(bad, &e), RAD_E_INVAL);
    bad = query(0, 8, 3, { 0, 8, 8, 8 });
    bad.verify_tokens = 31;   /* disagrees with n_rows + sum(rows) */
    CHECK_EQ(t.estimate(bad, &e), RAD_E_INVAL);

    /* A bucket extent that cannot have run: no round means no row, a round means at least one. */
    json no_rows = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    no_rows["extent"] = { { "rounds", 3 }, { "draft_rows", 0 } };
    CHECK_EQ(draft_cost_profile_parse(profile_text({ no_rows }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "extent"));
}

TEST(the_same_verify_shape_at_another_extent_is_another_bucket) {
    /* R1: one verify window (K6, 8 rows), two cycles that drafted a different number of rounds.
     * total_ms is an amortised batch time at one pipeline depth, so the two must not be averaged:
     * a K6 candidate plan is priced by the K6 cycle, never by (60 + 30) / 2. */
    const DraftVerifyShape k6 = query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 });
    const json deep = bucket_json(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }, 60.0, 0.6, 5);
    json shallow = bucket_json(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }, 30.0, 0.4, 3);
    shallow["extent"] = { { "rounds", 3 }, { "draft_rows", 24 } };

    /* Same shape, different extent: both buckets are legal, they are not duplicates. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ deep, shallow })), runtime()), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 2);
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(k6, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 60.0);
    CHECK_EQ(e.coverage, 5);

    /* Online: the same rule keeps two cycles apart instead of averaging them. */
    DraftCostTable h;
    CHECK_EQ(h.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftPlan p6 = plan_for(k6);
    CHECK_EQ(h.observe(completed_cycle(p6), sample_of(k6, 60.0)), RAD_OK);
    DraftCostSample shallow_cycle = sample_of(k6, 30.0);
    shallow_cycle.cycle_id = 2;
    shallow_cycle.rounds = 3;
    shallow_cycle.draft_rows = 24;
    CHECK_EQ(h.observe(completed_cycle(plan_for(k6, 2, 1, 6)), shallow_cycle), RAD_OK);
    CHECK_EQ((int)h.buckets().size(), 2);
    CHECK_EQ(h.estimate(k6, &e), RAD_OK);
    CHECK_EQ(e.ms, 60.0);
    CHECK_EQ(e.coverage, 1);
}

TEST(a_full_table_drops_and_counts_instead_of_growing) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime(), 2), RAD_OK);
    for (int k = 0; k <= 2; ++k) {
        /* rows is exactly verify_k + 1 long: index 0 unused, N_j = 4 above it. */
        std::vector<int32_t> rows((size_t)k + 1, 4);
        rows[0] = 0;
        const DraftVerifyShape s = query(0, 4, k, rows);
        const DraftPlan p = plan_for(s);
        CHECK_EQ(t.observe(completed_cycle(p), sample_of(s, 10.0 + k)), RAD_OK);
    }
    CHECK_EQ((int)t.buckets().size(), 2);
    CHECK_EQ(t.dropped(), 1);
}

TEST(an_interpolation_needs_the_whole_key_but_the_context_class) {
    /* R3: the neighbouring context class borrows a measurement only when everything else is the
     * same shape. Same depth, different N_j is a different kernel and must not be borrowed: with
     * no same-class anchor the answer is "no number", not a silently wrong one. */
    const json rect = bucket_json(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }, 61.0, 0.6, 6);
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ rect })), runtime()), RAD_OK);

    DraftCostEstimate e;
    const DraftVerifyShape mix = query(1, 8, 6, { 0, 8, 8, 2, 2, 2, 2 });
    CHECK_EQ(t.estimate(mix, &e), RAD_OK);
    CHECK(!e.valid);
    CHECK(e.source == DraftCostSource::ConservativeFallback);

    /* The same N_j at the neighbouring class is the documented one-step interpolation. */
    CHECK_EQ(t.estimate(query(1, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }), &e), RAD_OK);
    CHECK(e.valid);
    CHECK(e.source == DraftCostSource::Interpolated);
    CHECK_EQ(e.ms, 61.0);
}

TEST(the_documented_c8_gradient_is_reproduced_without_a_new_measurement) {
    /* The C8 case from the plan: depth 3 costs 36.3 ms for 22.3 emitted tokens per step, depth 5
     * costs 43.0 ms for 23.3. These are the documented figures replayed as profile buckets -- no
     * new measurement is claimed here -- and the model must price K3 strictly below K5, with K5
     * worse per emitted token. The verify window's verify_tokens is the pass's token count
     * (8 + sum N_j), not the emitted count. */
    const double k3_ms = 36.3, k3_tokens = 22.3;
    const double k5_ms = 43.0, k5_tokens = 23.3;
    CHECK(k3_ms < k5_ms);
    CHECK(k5_ms / k5_tokens > k3_ms / k3_tokens);

    const json c8_k3 = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, k3_ms, 0.5, 5);
    const json c8_k5 = bucket_json(0, 8, 5, { 0, 8, 8, 8, 8, 8 }, k5_ms, 0.7, 5);
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ c8_k3, c8_k5 })), runtime()), RAD_OK);

    DraftCostEstimate a, b;
    CHECK_EQ(t.estimate(query(0, 8, 3, { 0, 8, 8, 8 }), &a), RAD_OK);
    CHECK_EQ(t.estimate(query(0, 8, 5, { 0, 8, 8, 8, 8, 8 }), &b), RAD_OK);
    CHECK(a.source == DraftCostSource::Measured && b.source == DraftCostSource::Measured);
    CHECK_EQ(a.ms, k3_ms);
    CHECK_EQ(b.ms, k5_ms);
    CHECK(a.ms < b.ms);   /* the policy's comparison: K3 wins */
}

RAD_TEST_MAIN()
