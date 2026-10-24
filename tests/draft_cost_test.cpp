/* draft_cost_test.cpp -- strand A: the profile reader, the lookup and the sample replay
 * (core/sched/draft_cost.h), on synthetic profiles only.
 *
 * What is checked is what the policy may rely on: a profile is refused by name when it is not
 * this format or not this deployment, an exact bucket is Measured, a missing shape is one
 * documented interpolation or a named conservative bound -- never a unitless prior labelled
 * milliseconds -- and only an attributable sample moves a measured bucket. The key is the verify
 * shape -- the verify distribution, the pipeline depth and the loaded drafter -- paired with the
 * draft distribution that actually ran, so the same round count and the same row total with a
 * different shrinking distribution are two buckets, and a plan's original distribution never makes
 * its own truncated run unreachable.
 */
#include "rad_test.h"

#include "sched/draft_cost.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace rad;
using nlohmann::json;

namespace {

constexpr int32_t kMax = RAD_SCHED_MAX_SPEC;

json fingerprint_json() {
    return json{ { "model", "Qwen3.8-Flash-Next" }, { "quant", "gguf-q4_k_m" }, { "tp", 2 },
                 { "tp_wire", "wht6" }, { "gpus", "2x Radeon AI PRO R9700" },
                 { "kernels", "libr4d test+gfx1201" }, { "kv_format", "bf16" },
                 { "placement", "all_vram" }, { "basis", "live" }, { "arena", "none" },
                 { "live_loan", "flex=0 lend=0;flex=0 lend=0" } };
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
    f.basis = DraftCostBasis::Live;
    f.arena = "none";
    f.live_loan = "flex=0 lend=0;flex=0 lend=0";
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

json to_json_array(const std::vector<int32_t>& v) {
    json a = json::array();
    for (int32_t x : v) a.push_back(x);
    return a;
}

/* A bucket whose draft rounds plan exactly the verify window (pipeline_depth 0), which is the
 * synchronous case: one distribution, two names. */
json bucket_json(int32_t ctx, int32_t n_rows, int32_t k, std::vector<int32_t> rows,
                 double mean_ms, double unc_ms = 0.5, int32_t n = 5,
                 std::vector<std::string> flags = {}, const char* replay = "play",
                 int32_t history = 0, const char* drafter = "serial") {
    json origins = json::array();
    for (int32_t i = 0; i < n_rows; ++i)
        origins.push_back(k > 0 ? "head" : "none");
    return json{
        { "ctx_class", ctx },
        { "shape", { { "n_rows", n_rows }, { "verify_k", k }, { "verify_tokens", token_count(n_rows, rows) },
                     { "history_tokens", history }, { "prefill_tokens", 0 }, { "pipeline_depth", 0 },
                     { "rows", to_json_array(rows) }, { "draft_k", k },
                     { "draft_rows", to_json_array(rows) }, { "origins", origins },
                     { "sampling", "greedy" }, { "replay", replay }, { "drafter", drafter } } },
        { "extent", { { "rounds", k }, { "draft_rows_total", token_count(n_rows, rows) - n_rows },
                      { "draft_round_rows", to_json_array(rows) } } },
        { "time_ms", { { "mean", mean_ms }, { "unc", unc_ms }, { "n", n } } },
        { "flags", flags } };
}

std::string profile_text(const std::vector<json>& buckets) {
    json j{ { "format", "rad-draft-profile/4" }, { "fingerprint", fingerprint_json() },
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

DraftVerifyShape make_shape(int32_t ctx, int32_t n_rows, std::vector<int32_t> rows,
                            int32_t history = 0, int32_t draft_k = -1,
                            std::vector<int32_t> draft_rows = {}, int32_t depth = 0,
                            DraftDrafter drafter = DraftDrafter::Serial) {
    DraftVerifyShape s;
    s.ctx_class = ctx;
    s.n_rows = n_rows;
    s.verify_k = (int32_t)rows.size() - 1;
    for (size_t j = 0; j < rows.size() && j <= (size_t)s.verify_k; ++j) s.rows[j] = rows[j];
    s.verify_tokens = token_count(n_rows, rows);
    s.history_tokens = history;
    s.replay = DraftReplay::Play;
    s.drafter = drafter;
    s.pipeline_depth = depth;
    s.draft_k = draft_k < 0 ? s.verify_k : draft_k;
    if (draft_rows.empty()) {
        for (int j = 0; j <= s.draft_k; ++j) s.draft_rows[j] = s.rows[j];
    } else {
        for (size_t j = 0; j < draft_rows.size() && j <= (size_t)s.draft_k; ++j)
            s.draft_rows[j] = draft_rows[j];
    }
    return s;
}

/* The synchronous case: the plan's draft distribution is the verify window. */
DraftVerifyShape query(int32_t ctx, int32_t n_rows, int32_t k, std::vector<int32_t> rows,
                       int32_t history = 0) {
    return make_shape(ctx, n_rows, rows, history);
}

/* A plan over `s`, with the request arrays a real plan carries. k[] follows rows[] and fill[] is
 * None exactly where k is 0, which is what draft_plan_valid requires. */
struct PlanFixture {
    DraftPlan plan;
    std::vector<uint64_t> req_id;
    std::vector<int32_t> slot;
    std::vector<uint32_t> slot_gen;
    std::vector<int32_t> k;
    std::vector<DraftDrafter> fill;
    std::vector<int32_t> ctx;
    std::vector<int32_t> hist;
    int32_t rows[kMax + 1] = {};
};

void fill_plan(PlanFixture* f, const DraftVerifyShape& s, uint64_t cycle, uint32_t version) {
    const int32_t n = s.n_rows;
    f->req_id.assign((size_t)n, 0);
    f->slot.assign((size_t)n, 0);
    f->slot_gen.assign((size_t)n, 0);
    /* Real identities: a ring slot is recycled and the generation is what tells the rows apart. */
    for (int32_t i = 0; i < n; ++i) {
        f->req_id[(size_t)i] = (uint64_t)(11 + i);
        f->slot[(size_t)i] = i;
        f->slot_gen[(size_t)i] = 1;
    }
    f->k.assign((size_t)n, 0);
    f->fill.assign((size_t)n, DraftDrafter::None);
    f->ctx.assign((size_t)n, s.ctx_class);
    f->hist.assign((size_t)n, s.history_tokens);
    for (int j = 0; j <= kMax; ++j) f->rows[j] = 0;

    /* rows[j] is N_j, the count of requests whose k reaches round j. */
    for (int32_t i = 0; i < n; ++i) {
        int32_t depth = 0;
        for (int j = 1; j <= s.draft_k; ++j)
            if (i < s.draft_rows[j]) depth = j;
        f->k[(size_t)i] = depth;
        f->fill[(size_t)i] = depth > 0 ? (s.drafter == DraftDrafter::Ngram ? DraftDrafter::Ngram
                                                                          : DraftDrafter::Serial)
                                       : DraftDrafter::None;
        for (int j = 1; j <= depth; ++j) ++f->rows[j];
    }

    f->plan.cycle_id = cycle;
    f->plan.policy_version = version;
    f->plan.n_seq = n;
    f->plan.cap = n;
    f->plan.k_ceiling = s.drafter == DraftDrafter::None ? 0 : std::max(1, s.draft_k);
    f->plan.max_k = s.draft_k;
    f->plan.verify_k = s.verify_k;
    f->plan.pipeline_depth = s.pipeline_depth;
    f->plan.replay = s.replay;
    f->plan.timing_valid = true;
    f->plan.prefill_tokens = s.prefill_tokens;
    f->plan.req_id = f->req_id.data();
    f->plan.slot = f->slot.data();
    f->plan.slot_gen = f->slot_gen.data();
    f->plan.k = f->k.data();
    f->plan.fill = f->fill.data();
    f->plan.ctx_class = f->ctx.data();
    f->plan.history_tokens = f->hist.data();
    f->plan.rows = f->rows;
}

PlanFixture plan_for(const DraftVerifyShape& s, uint64_t cycle = 1, uint32_t version = 1) {
    PlanFixture f;
    fill_plan(&f, s, cycle, version);
    CHECK_EQ(draft_plan_valid(f.plan), RAD_OK);
    return f;
}

DraftObservation completed_cycle(const PlanFixture& f) {
    DraftObservation o;
    o.cycle_id = f.plan.cycle_id;
    o.policy_version = f.plan.policy_version;
    o.n_seq = f.plan.n_seq;
    o.cap = f.plan.n_seq;
    o.plan = &f.plan;
    o.req_id = f.plan.req_id;
    o.slot = f.plan.slot;
    o.slot_gen = f.plan.slot_gen;
    o.complete = true;
    return o;
}

/* A sample for `s` as planned, at `ms`. The replay state is the one the shape carries -- what the
 * executing path established for this cycle -- and is copied into the sample's own evidence field
 * rather than defaulted: `Unknown` never becomes a bucket key here, and neither does a state the
 * shape does not name. */
DraftCostSample sample_of(const DraftVerifyShape& s, double ms,
                          uint16_t flags = kDraftSampleValid) {
    DraftCostSample c;
    c.cycle_id = 1;
    c.policy_version = 1;
    c.shape = s;
    c.replay = s.replay;
    c.total_ms = ms;
    c.flags = flags;
    c.rounds = s.draft_k;
    for (int j = 0; j <= s.draft_k; ++j) {
        c.draft_round_rows[j] = s.draft_rows[j];
        c.draft_rows_total += s.draft_rows[j];
    }
    return c;
}

/* A sample of `s` whose cycle stopped after `rounds` of the planned rounds and kept the full verify
 * window: the pass verifies the planned window, only the drafting stopped. */
DraftCostSample truncated(const DraftVerifyShape& s, int32_t rounds, double ms) {
    DraftCostSample c = sample_of(s, ms);
    c.rounds = rounds;
    c.flags |= kDraftSampleShortened;   /* the shared contract's name for the same fact */
    for (int j = rounds + 1; j <= s.draft_k; ++j) c.draft_round_rows[j] = 0;
    c.draft_rows_total = 0;
    for (int j = 0; j <= rounds; ++j) c.draft_rows_total += c.draft_round_rows[j];
    return c;
}

}  /* namespace */

/* The bucket stores the set of origins its samples verified each row from, as a bit per
 * DraftOrigin; the file writes it as a '+' join. */
uint8_t ob(DraftOrigin o) { return (uint8_t)(1u << (unsigned)o); }

TEST(the_reader_accepts_the_documented_profile) {
    /* The format example from docs/ADAPTIVE-MTP.md, verbatim, to check field handling. Its 43.0 ms
     * is the example's own value -- a format sample, not the measured C8 K3 figure; the documented
     * C8 numbers are replayed in the_documented_c8_gradient_is_reproduced_without_a_new_measurement
     * and the two must not be conflated. */
    const std::string text = R"({
      "format": "rad-draft-profile/4",
      "fingerprint": {
        "model": "Qwen3.8-Flash-Next", "quant": "gguf-q4_k_m", "tp": 2, "tp_wire": "wht6",
        "gpus": "2x Radeon AI PRO R9700", "kernels": "libr4d test+gfx1201",
        "kv_format": "bf16", "placement": "all_vram", "basis": "live", "arena": "none",
        "live_loan": "flex=0 lend=0;flex=0 lend=0"
      },
      "execution": "rectangular",
      "buckets": [{
        "ctx_class": 0,
        "shape": { "n_rows": 8, "verify_k": 3, "verify_tokens": 32, "history_tokens": 0,
                   "prefill_tokens": 0, "pipeline_depth": 0,
                   "rows": [0, 8, 8, 8], "draft_k": 3, "draft_rows": [0, 8, 8, 8],
                   "origins": ["head", "head", "head", "head", "head", "head", "head", "none"],
                   "sampling": "greedy", "replay": "play", "drafter": "serial" },
        "extent": { "rounds": 3, "draft_rows_total": 24, "draft_round_rows": [0, 8, 8, 8] },
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
    CHECK_EQ(b.shape.draft_k, 3);
    CHECK_EQ(b.shape.pipeline_depth, 0);
    CHECK(b.shape.drafter == DraftDrafter::Serial);
    CHECK_EQ(b.mean_ms, 43.0);
    CHECK_EQ(b.n, 5);
    CHECK_EQ(b.rounds, 3);
    CHECK_EQ(b.draft_rows_total, 24);
    CHECK_EQ(b.draft_round_rows[3], 8);
    CHECK((b.flags & kDraftSampleAhead) != 0);
    /* The format carries the ahead flag, not how many samples were ahead: a loaded bucket claims
     * no online count, and the flag alone keeps it visible. */
    CHECK_EQ(b.n_ahead, 0);
    CHECK(b.target_ms > 0.0 && b.draft_ms > 0.0 && b.host_ms > 0.0);
}

TEST(an_older_profile_is_refused_by_name) {
    /* v1 has no draft distribution, no pipeline depth and no origin mix; v2 has no resolved
     * placement or device equipment in its fingerprint; v3 has neither a measurement basis nor an
     * arena-level identity, so a v3 file cannot say whether its numbers were measured with the
     * lenders off. None can be mapped onto the current key by assuming values: the reader refuses
     * the id and names the version it will read. It never rescales. */
    std::string err;
    DraftCostProfile p;

    json v1 = json::parse(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));
    v1["format"] = "rad-draft-profile/1";
    CHECK_EQ(draft_cost_profile_parse(v1.dump(), &p, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "rad-draft-profile/1"));
    CHECK(has(err, "rad-draft-profile/4"));

    json v2 = json::parse(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));
    v2["format"] = "rad-draft-profile/2";
    CHECK_EQ(draft_cost_profile_parse(v2.dump(), &p, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "rad-draft-profile/2"));
    CHECK(has(err, "placement"));

    json v3 = json::parse(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));
    v3["format"] = "rad-draft-profile/3";
    CHECK_EQ(draft_cost_profile_parse(v3.dump(), &p, &err), RAD_E_UNSUPPORTED);
    CHECK(has(err, "rad-draft-profile/3"));
    CHECK(has(err, "basis"));

    /* The basis is required, not defaulted: a file that does not say how its numbers were
     * measured would be read under a rule it never claimed. */
    json no_basis = json::parse(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));
    no_basis["fingerprint"].erase("basis");
    CHECK_EQ(draft_cost_profile_parse(no_basis.dump(), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "basis"));

    /* A bucket that lost a current field is malformed, not an older file to be guessed at. */
    json missing = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    missing["shape"].erase("draft_rows");
    CHECK_EQ(draft_cost_profile_parse(profile_text({ missing }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "draft_rows"));
}

TEST(the_reader_refuses_inconsistent_counts_and_unknown_dimensions) {
    std::string err;
    DraftCostProfile p;

    /* verify_tokens disagreeing with n_rows + sum(rows) is not a measurement of this shape. */
    json bad = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    bad["shape"]["verify_tokens"] = 31;
    CHECK_EQ(draft_cost_profile_parse(profile_text({ bad }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "verify_tokens") || has(err, "priceable"));

    /* A bucket with no samples is not a measurement. */
    json none = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0, 0.5, 0);
    CHECK_EQ(draft_cost_profile_parse(profile_text({ none }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "time_ms.n"));

    /* At pipeline_depth 0 the plan's distribution and the verify window are one execution; a file
     * that lets them disagree was not produced by a measurement. */
    json split = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    split["shape"]["draft_rows"] = { 0, 8, 4, 4 };
    split["shape"]["draft_k"] = 3;
    CHECK_EQ(draft_cost_profile_parse(profile_text({ split }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "disagrees"));

    /* An origin that does not fit the bucket's own drafter: a head row cannot appear under a
     * lookup-only drafter. */
    json mix = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0, 0.5, 5, {}, "play", 0, "ngram");
    CHECK_EQ(draft_cost_profile_parse(profile_text({ mix }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "origins"));

    json no_origin = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    no_origin["shape"].erase("origins");
    CHECK_EQ(draft_cost_profile_parse(profile_text({ no_origin }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "origins"));
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

/* `arena` is part of the placement identity for every basis: two starts with different level rows
 * lend at different step sizes, so a file measured under one does not price the other. The refusal
 * names the field, like the other identity mismatches. */
TEST(an_arena_level_mismatch_disables_the_profile) {
    const DraftCostProfile p = parse_or_fail(
        profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));

    DraftCostRuntime other = runtime();
    other.fingerprint.arena = "8192,2048,512";
    DraftCostTable t;
    CHECK_EQ(t.configure(p, other), RAD_OK);   /* refused, not an error */
    CHECK(!t.active());
    CHECK(has(t.reason(), "arena"));
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.coverage, 0);
}

/* A live profile compares the loan it was measured under; a different flex/lend identity is a
 * different deployment. (A no-loan reference ignores the field by design; that rule is covered
 * where the reference is.) */
TEST(a_live_loan_mismatch_disables_a_live_profile) {
    const DraftCostProfile p = parse_or_fail(
        profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) }));

    DraftCostRuntime other = runtime();
    other.fingerprint.live_loan = "flex=19922944 lend=0;flex=0 lend=0";
    DraftCostTable t;
    CHECK_EQ(t.configure(p, other), RAD_OK);   /* refused, not an error */
    CHECK(!t.active());
    CHECK(has(t.reason(), "live_loan"));
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.coverage, 0);
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

    /* A deeper draft distribution at the same class: the anchor ran three rounds, the query plans
     * six. A verify-token ratio cannot stand in for three unmeasured draft rounds, so there is no
     * number here -- the named baseline -- and the deeper tier does not win by being free. */
    CHECK_EQ(t.estimate(query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 }), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.ms, 0.0);

    /* More history than the anchor ran is not an anchor: no measured ratio covers history work, so
     * there is no number to borrow -- the named baseline. */
    CHECK_EQ(t.estimate(query(0, 8, 3, { 0, 8, 8, 8 }, 512), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.ms, 0.0);

    /* The same for prefill work sharing the step. */
    DraftVerifyShape prefill = query(0, 8, 3, { 0, 8, 8, 8 });
    prefill.prefill_tokens = 1;
    CHECK_EQ(t.estimate(prefill, &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.ms, 0.0);

    /* Two classes away with nothing measured between: no anchor, no number. */
    CHECK_EQ(t.estimate(query(3, 8, 3, { 0, 8, 8, 8 }), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(e.ms, 0.0);

    /* Another loaded drafter is another process, not a bound to borrow. */
    const DraftVerifyShape ngram = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, {}, 0,
                                              DraftDrafter::Ngram);
    CHECK_EQ(t.estimate(ngram, &e), RAD_OK);
    CHECK(!e.valid);
}

TEST(k0_is_a_plain_decode_step_with_its_own_bucket) {
    /* K0 with no drafter loaded is its own process, not "policy K0" with a head standing by. */
    const json k0 = bucket_json(0, 8, 0, { 0 }, 11.0, 0.2, 9, {}, "play", 0, "none");
    const json k3 = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0, 0.6, 7);
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ k0, k3 })), runtime()), RAD_OK);

    DraftCostEstimate e;
    const DraftVerifyShape plain = make_shape(0, 8, { 0 }, 0, -1, {}, 0, DraftDrafter::None);
    CHECK_EQ(t.estimate(plain, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 11.0);

    /* K0 with the drafter loaded carries history; it is a different measurement from K0 in a
     * process without one, so it must not be priced by the plain K0 bucket. */
    const json warm = bucket_json(0, 8, 0, { 0 }, 14.0, 0.3, 4, {}, "play", 512);
    DraftCostTable h;
    CHECK_EQ(h.configure(parse_or_fail(profile_text({ k0, warm })), runtime()), RAD_OK);
    const DraftVerifyShape warm_q = make_shape(0, 8, { 0 }, 512);
    CHECK_EQ(h.estimate(warm_q, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 14.0);
    CHECK_EQ(h.estimate(plain, &e), RAD_OK);
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

TEST(a_shrinking_distribution_is_its_own_bucket_at_the_same_rounds_and_total) {
    /* The review case: [3,1,1] and [2,2,1] have the same round count (3) and the same row total
     * (5), and they are different kernels. A key of rounds plus total would price one by the
     * other; the full distribution keeps them apart -- in the file and online. */
    const std::vector<int32_t> a_rows{ 0, 3, 1, 1 };
    const std::vector<int32_t> b_rows{ 0, 2, 2, 1 };
    CHECK_EQ(token_count(4, a_rows), token_count(4, b_rows));   /* same total, on purpose */

    const json a = bucket_json(0, 4, 3, a_rows, 20.0, 0.2, 3);
    const json b = bucket_json(0, 4, 3, b_rows, 26.0, 0.2, 3);
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ a, b })), runtime()), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 2);   /* not duplicates */

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 4, 3, a_rows), &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 20.0);
    CHECK_EQ(t.estimate(query(0, 4, 3, b_rows), &e), RAD_OK);
    CHECK_EQ(e.ms, 26.0);

    /* Online, the same two cycles land in two buckets and are never averaged. */
    DraftCostTable h;
    CHECK_EQ(h.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape sa = query(0, 4, 3, a_rows);
    const DraftVerifyShape sb = query(0, 4, 3, b_rows);
    const PlanFixture pa = plan_for(sa);
    const PlanFixture pb = plan_for(sb, 2);
    CHECK_EQ(h.observe(completed_cycle(pa), sample_of(sa, 20.0)), RAD_OK);
    DraftCostSample second = sample_of(sb, 26.0);
    second.cycle_id = 2;
    CHECK_EQ(h.observe(completed_cycle(pb), second), RAD_OK);
    CHECK_EQ((int)h.buckets().size(), 2);
    CHECK_EQ(h.estimate(sa, &e), RAD_OK);
    CHECK_EQ(e.ms, 20.0);

    /* One context class up, the same shape: each distribution interpolates from its own bucket. */
    CHECK_EQ(t.estimate(query(1, 4, 3, a_rows), &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Interpolated);
    CHECK_EQ(e.ms, 20.0);
    CHECK_EQ(t.estimate(query(1, 4, 3, b_rows), &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Interpolated);
    CHECK_EQ(e.ms, 26.0);

    /* With only [3,1,1] measured, the other distribution must never borrow it -- not as a
     * measurement and not as a bound: at the same class the two distributions are two kernels, and
     * the fallback scales verify work only, so there is no number at either class. That is what
     * keeps an unmeasured deeper tier from being priced at a shallower one's cost. */
    DraftCostTable only_a;
    CHECK_EQ(only_a.configure(parse_or_fail(profile_text({ a })), runtime()), RAD_OK);
    CHECK_EQ(only_a.estimate(query(0, 4, 3, b_rows), &e), RAD_OK);
    CHECK(!e.valid);
    CHECK_EQ(only_a.estimate(query(1, 4, 3, b_rows), &e), RAD_OK);
    CHECK(!e.valid);
}

TEST(a_pipeline_depth_is_part_of_the_key) {
    /* The same verify window and the same planned distribution, issued at another pipeline depth,
     * is another overlap structure: a depth-0 measurement may not price a depth-1 plan. */
    const DraftVerifyShape sync = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, {}, 0);
    const DraftVerifyShape deep = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, {}, 1);
    const json b = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0, 0.6, 5);

    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ b })), runtime()), RAD_OK);
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(sync, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(t.estimate(deep, &e), RAD_OK);
    CHECK(!e.valid);   /* not interpolated, not borrowed across the depth */
}

TEST(only_an_attributable_sample_updates_a_measured_bucket) {
    DraftCostProfile empty = parse_or_fail(profile_text({}));
    DraftCostTable t;
    CHECK_EQ(t.configure(empty, runtime()), RAD_OK);
    CHECK(t.active());

    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);
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

    /* A disturbing flag without Valid did not time the shape: it is not storable, so it is neither
     * a measured bucket nor a diagnostic record (draft_sample_storable). */
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 210.0, kDraftSampleRecord)), RAD_OK);
    CHECK_EQ(t.ignored(), 2);
    CHECK_EQ((int)t.buckets().size(), 2);
}

/* A calibration measures and never prices: the table files what the run executes, and no estimate
 * comes out of it -- not even from a bucket it has just recorded. Otherwise the run's second cycle
 * would decide with the first cycle's measurement and the file would describe a steered run. */
TEST(a_record_only_table_learns_without_pricing_anything) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    t.record_only();
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);

    /* The recording still works: a clean sample becomes a bucket the export may write. */
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 40.0)), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);
    CHECK(t.has_usable_bucket());

    /* And nothing is priced, not even the shape just measured. */
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(!e.valid);

    /* A cleared table does not inherit the mode. */
    t.clear();
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 40.0)), RAD_OK);
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(e.valid);
}

/* The start placement is not materialised yet (Engine::start_placement_settled): the table prices
 * nothing, so the planner runs its named baseline instead of a measurement of a residency the run
 * has not reached. Unlike drift, this is not a deactivation -- the profile is still the one this
 * deployment will run on, and once the fill has landed the same bucket prices again. */
TEST(held_prices_answer_nothing_without_disabling_the_profile) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 },
                                                                  43.0, 0.6, 7) })),
                         runtime()),
             RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(e.valid);
    CHECK_EQ(e.ms, 43.0);

    t.hold_prices(true);
    CHECK(t.prices_held());
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(!e.valid);                 /* not priced, not refused */
    CHECK_EQ(e.ms, 0.0);
    CHECK(t.active());               /* still this deployment's profile */
    CHECK(t.reason().empty());      /* no drift reason: nothing diverged */

    t.hold_prices(false);
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(e.valid);
    CHECK_EQ(e.ms, 43.0);

    /* A cleared table does not inherit the hold, like the record-only mode. */
    t.hold_prices(true);
    t.clear();
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 },
                                                                  43.0, 0.6, 7) })),
                         runtime()),
             RAD_OK);
    CHECK(!t.prices_held());
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(e.valid);
}

TEST(a_sample_without_replay_evidence_is_never_learned_from) {
    /* The bucket key is the replay state the run established, so the sample has to carry one and it
     * has to be the state its shape names. Neither case may fall through as "no replay" -- a None
     * written for gate conformity would price a shape with a fact nobody measured. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);

    /* The executing path could not establish the state: the bracket exists, it is not evidence. */
    DraftCostSample unknown = sample_of(k3, 40.0);
    unknown.replay = DraftReplay::Unknown;
    CHECK_EQ(t.observe(completed_cycle(p3), unknown), RAD_OK);
    CHECK_EQ(t.ignored(), 1);
    CHECK_EQ((int)t.buckets().size(), 0);

    /* A sample that disagrees with the shape it files itself under is a broken record, not a
     * measurement under another key: refused, and nothing is stored. */
    DraftCostSample contradicted = sample_of(k3, 40.0);
    contradicted.replay = DraftReplay::Record;
    CHECK_EQ(t.observe(completed_cycle(p3), contradicted), RAD_E_INVAL);
    CHECK_EQ((int)t.buckets().size(), 0);
    CHECK_EQ(t.ignored(), 1);

    /* The evidence the run did establish is learned from as before. */
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k3, 41.0)), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(k3, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 41.0);
}

/* And the deployment's proof is the whole deployment's: the pass spans the ranks, so `None` needs
 * every one of them to keep no tape. One recording rank -- or no rank asked at all -- leaves the
 * cycle's state Unknown, which no bucket may be keyed on. */
TEST(the_replay_evidence_is_the_whole_deployments) {
    CHECK(draft_replay_evidence(1, 0) == DraftReplay::None);
    CHECK(draft_replay_evidence(2, 0) == DraftReplay::None);
    CHECK(draft_replay_evidence(2, 1) == DraftReplay::Unknown);   /* mixed evidence */
    CHECK(draft_replay_evidence(2, 2) == DraftReplay::Unknown);
    CHECK(draft_replay_evidence(0, 0) == DraftReplay::Unknown);   /* nobody asked: no evidence */
}

TEST(a_depth_change_under_overlap_lands_in_its_own_bucket) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);

    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const DraftVerifyShape k6 = query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);
    const PlanFixture p6 = plan_for(k6);
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

TEST(an_ahead_bucket_is_found_again_as_measured_through_its_plan) {
    /* R2: a paired step verifies 3 rounds while the cycle that ran ahead drafted 6 rounds of 8
     * rows. The stored extent is the plan's own draft distribution, not (verify_k, verify tokens -
     * rows), so the same shape must find its own measurement -- as Measured, not as a 25 % bound
     * scaled from itself, and not as a bucket no query can express. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);

    const DraftVerifyShape ahead = make_shape(0, 8, { 0, 8, 8, 8 }, 0, 6,
                                              { 0, 8, 8, 8, 8, 8, 8 }, 6);
    const PlanFixture p = plan_for(ahead);
    CHECK_EQ(t.observe(completed_cycle(p), sample_of(ahead, 40.0, kDraftSampleValid | kDraftSampleAhead)),
             RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);
    CHECK_EQ(t.buckets()[0].rounds, 6);
    CHECK_EQ(t.buckets()[0].draft_rows_total, 48);

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(ahead, &e), RAD_OK);
    CHECK(e.valid);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 40.0);
    CHECK_EQ(e.coverage, 1);

    /* The k6 verify window is another shape key and must not be priced by the k3 sample. It is
     * priced as the documented heuristic instead: the same executed distribution and the same
     * history/prefill, scaled by the verify-token ratio 56/32 plus the named margin -- the one
     * dimension the bound may scale, and a rule of thumb, not a measurement. */
    const DraftVerifyShape k6 = make_shape(0, 8, { 0, 8, 8, 8, 8, 8, 8 }, 0, 6,
                                           { 0, 8, 8, 8, 8, 8, 8 }, 6);
    CHECK_EQ(t.estimate(k6, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::ConservativeFallback);
    CHECK_EQ(e.coverage, 0);
    CHECK(std::fabs(e.ms - 40.0 * (56.0 / 32.0) * 1.25) < 1e-9);
    CHECK(std::fabs(e.unc_ms - 40.0 * (56.0 / 32.0) * 0.25) < 1e-9);
}

TEST(a_short_execution_is_not_the_price_of_a_short_plan) {
    /* A cycle may legally stop before its plan's deepest round. Its bracket then carries the cost
     * of stopping -- drain, rows in flight, discarded slots -- which a plan deliberately drawn at
     * that extent would not pay. So it prices neither the full plan nor the shorter one: it is a
     * diagnostic in both directions, not just when no round ran. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape full = make_shape(0, 8, { 0, 8, 8, 8, 8 }, 0, -1, {}, 0);
    const PlanFixture p = plan_for(full);

    DraftCostSample stopped = sample_of(full, 18.0);
    stopped.rounds = 2;   /* stopped after round 2 of 4, both rounds full */
    stopped.draft_round_rows[3] = 0;
    stopped.draft_round_rows[4] = 0;
    stopped.draft_rows_total = 16;
    CHECK_EQ(t.observe(completed_cycle(p), stopped), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);
    CHECK_EQ(t.buckets()[0].rounds, 2);
    CHECK_EQ(t.buckets()[0].shape.draft_k, 4);   /* the plan stays as provenance and upper bound */
    CHECK((t.buckets()[0].flags & kDraftBucketUnpriced) != 0);

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(full, &e), RAD_OK);
    CHECK(!e.valid);   /* nobody measured the full plan */

    /* And the run is not sold as the price of a two-round plan either: the abandoned tail is work
     * this bracket paid and that plan would not. */
    const DraftVerifyShape short_plan = make_shape(0, 8, { 0, 8, 8, 8, 8 }, 0, 2, { 0, 8, 8 }, 0);
    CHECK_EQ(t.estimate(short_plan, &e), RAD_OK);
    CHECK(!e.valid);

    /* A cycle that ran every planned round is priced as usual -- shrinking included, because the
     * plan's deepest round did run. */
    DraftCostSample full_run = sample_of(full, 24.0);
    full_run.cycle_id = 2;
    CHECK_EQ(t.observe(completed_cycle(plan_for(full, 2)), full_run), RAD_OK);
    CHECK_EQ(t.estimate(full, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 24.0);
    CHECK_EQ(e.coverage, 1);
}

TEST(a_truncation_and_a_shrinking_run_are_told_apart) {
    /* Q2-Q4 of the review, with the stop distinguished from the shrinking execution. One planned
     * K3 shape, two cycles: one stopped after two full rounds (31.0 ms), one ran all three rounds
     * shrinking 8,5,2 (38.0 ms). The shrinking run finished its plan, so it is the measurement of
     * its own distribution; the stopped run is a diagnostic and prices nothing. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape full = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, {}, 0);
    const PlanFixture p = plan_for(full);

    /* Q2: two of three rounds, both full -- cut short, so it carries abort cost. */
    CHECK_EQ(t.observe(completed_cycle(p), truncated(full, 2, 31.0)), RAD_OK);

    DraftCostEstimate e;
    CHECK_EQ(t.estimate(full, &e), RAD_OK);
    CHECK(!e.valid);   /* not the plan's price, and no anchor to scale from */
    const DraftVerifyShape two_rounds = make_shape(0, 8, { 0, 8, 8, 8 }, 0, 2, { 0, 8, 8 }, 0);
    CHECK_EQ(t.estimate(two_rounds, &e), RAD_OK);
    CHECK(!e.valid);   /* nor is it the price of a two-round plan */

    /* Q3/Q4: the same shape, a third cycle that shrank 8,5,2 in three rounds -- its plan's deepest
     * round ran, so this is a real measurement of that distribution. */
    DraftCostSample shrinking = sample_of(full, 38.0);
    shrinking.cycle_id = 2;
    shrinking.draft_round_rows[2] = 5;
    shrinking.draft_round_rows[3] = 2;
    shrinking.draft_rows_total = 15;
    const PlanFixture p2 = plan_for(full, 2);
    CHECK_EQ(t.observe(completed_cycle(p2), shrinking), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 2);   /* two executions, neither folded into the other */

    const DraftVerifyShape shrunk = make_shape(0, 8, { 0, 8, 8, 8 }, 0, 3, { 0, 8, 5, 2 }, 0);
    CHECK_EQ(t.estimate(shrunk, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 38.0);
    CHECK_EQ(t.estimate(full, &e), RAD_OK);
    CHECK(e.source != DraftCostSource::Measured);   /* nobody measured the full plan */
}

TEST(a_shorter_verify_window_is_not_sold_the_longer_runs_bracket) {
    /* The limit of the executed-distribution key: the draft half is comparable, the verify half is
     * not. A query that drafts the same two rounds but verifies only [0,8,8] -- one window fewer --
     * pays less verify work, so the stored bracket is neither its measurement nor its bound. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape full = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, {}, 0);
    const PlanFixture p = plan_for(full);
    CHECK_EQ(t.observe(completed_cycle(p), truncated(full, 2, 31.0)), RAD_OK);

    DraftCostEstimate e;
    const DraftVerifyShape cheap = make_shape(0, 8, { 0, 8, 8 }, 0, 2, { 0, 8, 8 }, 0);
    CHECK_EQ(t.estimate(cheap, &e), RAD_OK);
    CHECK(!e.valid);   /* the only run of that extent was cut short, so it is not an anchor */
}

TEST(a_truncation_is_kept_as_a_diagnostic_and_prices_nothing) {
    /* Drafts were planned and none ran: the bracket is a verify window plus an abandoned draft
     * phase. A K0 query plans no draft work, so recording this as a measured K0 bucket would
     * overprice it; the sample is kept flagged and prices nothing. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);
    CHECK_EQ(t.observe(completed_cycle(p3), truncated(k3, 0, 20.0)), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 1);
    CHECK((t.buckets()[0].flags & kDraftBucketUnpriced) != 0);

    DraftCostEstimate e;
    const DraftVerifyShape k0 = make_shape(0, 8, { 0 }, 0, 0, {}, 0);
    CHECK_EQ(t.estimate(k0, &e), RAD_OK);
    CHECK(!e.valid);   /* a diagnostic is not a price, and it is not an anchor either */

    /* A genuine K0 cycle (nothing planned, nothing drafted) is still measurable. */
    const PlanFixture p0 = plan_for(k0, 2);
    DraftCostSample idle = sample_of(k0, 12.0);
    idle.cycle_id = 2;
    CHECK_EQ(t.observe(completed_cycle(p0), idle), RAD_OK);
    CHECK_EQ((int)t.buckets().size(), 2);
    CHECK_EQ(t.estimate(k0, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    CHECK_EQ(e.ms, 12.0);
}

TEST(a_plan_that_disowns_its_bracket_teaches_nothing) {
    /* DraftPlan::timing_valid is the plan's own statement that total_ms brackets it. A sample of a
     * plan that says it does not is no price for any shape the plan names. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    PlanFixture p = plan_for(k3);
    p.plan.timing_valid = false;
    CHECK_EQ(t.observe(completed_cycle(p), sample_of(k3, 40.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());
}

TEST(an_incomplete_or_inconsistent_observation_is_refused) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);

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

    const PlanFixture p3 = plan_for(k3);
    DraftCostSample zero_cycles = sample_of(k3, 40.0);
    zero_cycles.cycle_id = 0;
    CHECK_EQ(t.observe(completed_cycle(p3), zero_cycles), RAD_E_INVAL);
    DraftCostSample zero_version = sample_of(k3, 40.0);
    zero_version.policy_version = 0;
    CHECK_EQ(t.observe(completed_cycle(p3), zero_version), RAD_E_INVAL);

    /* An observation at 5/4 and a sample at 1/1 do not describe the same measurement either. */
    const PlanFixture other = plan_for(k3, 5, 4);
    DraftObservation o5 = completed_cycle(other);
    CHECK_EQ(t.observe(o5, zero_cycles), RAD_E_INVAL);
    /* And an identity that disagrees with the observation is a wiring error. */
    CHECK_EQ(t.observe(o5, sample_of(k3, 40.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());
}

TEST(a_recycled_slot_or_a_short_observation_is_refused) {
    /* R3: draft_plan_identifies is the stale-plan guard, per row: cycle, policy version and
     * (req_id, slot, slot_gen). Same slots one generation later, or another request in one row, is
     * another cycle's work and may not teach this plan. The observation must also count the plan's
     * rows -- its n_seq is the length of the identity it carries. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p = plan_for(k3);

    std::vector<uint32_t> next_gen(p.slot_gen.size(), 2);
    DraftObservation recycled = completed_cycle(p);
    recycled.slot_gen = next_gen.data();
    CHECK_EQ(t.observe(recycled, sample_of(k3, 40.0)), RAD_E_INVAL);

    std::vector<uint64_t> other_req(p.req_id);
    other_req[3] += 100;
    DraftObservation swapped = completed_cycle(p);
    swapped.req_id = other_req.data();
    CHECK_EQ(t.observe(swapped, sample_of(k3, 40.0)), RAD_E_INVAL);

    DraftObservation short_obs = completed_cycle(p);
    short_obs.n_seq = 3;   /* the plan issued 8 rows */
    CHECK_EQ(t.observe(short_obs, sample_of(k3, 40.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());

    /* The unmodified observation of the same plan is still accepted. */
    CHECK_EQ(t.observe(completed_cycle(p), sample_of(k3, 40.0)), RAD_OK);
}

TEST(a_sample_must_match_the_plan_that_produced_it) {
    /* The shape must be the plan's own: same rows, same deepest round, same pipeline depth and the
     * same planned distribution. A self-consistent shape from another cycle is refused. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const DraftVerifyShape k6 = query(0, 8, 6, { 0, 8, 8, 8, 8, 8, 8 });

    const PlanFixture p3 = plan_for(k3);
    CHECK_EQ(t.observe(completed_cycle(p3), sample_of(k6, 60.0)), RAD_E_INVAL);
    CHECK(t.buckets().empty());

    /* Another planned distribution than the one the plan asked for. */
    DraftCostSample wrong_rows = sample_of(k3, 40.0);
    wrong_rows.shape = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, { 0, 8, 4, 4 });
    CHECK_EQ(t.observe(completed_cycle(p3), wrong_rows), RAD_E_INVAL);

    /* Another pipeline depth than the plan's. */
    DraftCostSample wrong_depth = sample_of(k3, 40.0);
    wrong_depth.shape = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, {}, 1);
    CHECK_EQ(t.observe(completed_cycle(p3), wrong_depth), RAD_E_INVAL);

    /* The row count is part of the same identity: a consistent 4-row shape from an 8-row plan. */
    DraftCostTable h;
    CHECK_EQ(h.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape four = query(0, 4, 3, { 0, 4, 4, 4 });
    CHECK_EQ(h.observe(completed_cycle(p3), sample_of(four, 40.0)), RAD_E_INVAL);

    /* An executed extent deeper than the plan's own max_k cannot have run. */
    const PlanFixture shallow = plan_for(query(0, 8, 2, { 0, 8, 8 }));
    CHECK_EQ(h.observe(completed_cycle(shallow), sample_of(k3, 40.0)), RAD_E_INVAL);
    CHECK(h.buckets().empty());
}

TEST(an_executed_distribution_that_could_not_have_run_is_refused) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime()), RAD_OK);
    const DraftVerifyShape k3 = query(0, 8, 3, { 0, 8, 8, 8 });
    const PlanFixture p3 = plan_for(k3);

    /* The total must be the sum of the rounds. */
    DraftCostSample bad_total = sample_of(k3, 40.0);
    bad_total.draft_rows_total = 23;
    CHECK_EQ(t.observe(completed_cycle(p3), bad_total), RAD_E_INVAL);

    /* N_j never rises with the round. */
    DraftCostSample rising = sample_of(k3, 40.0);
    rising.draft_round_rows[1] = 4;
    rising.draft_round_rows[2] = 8;
    rising.draft_rows_total = 20;
    CHECK_EQ(t.observe(completed_cycle(p3), rising), RAD_E_INVAL);

    /* More rounds than the plan asked for. */
    DraftCostSample deep = sample_of(k3, 40.0);
    deep.rounds = 4;
    CHECK_EQ(t.observe(completed_cycle(p3), deep), RAD_E_INVAL);

    /* One round cannot draft more rows than the plan asked of it: this plan commits 8 requests to
     * round 1 and only 5 to rounds 2 and 3, so an executed [0,8,8,8] is impossible even though it
     * is monotone, complete and sums to its own total. That is the plan's cap * max_k = 24 bound,
     * per round rather than as one number. Under overlap, so the verify window is not the plan's
     * own and the per-round rule is the one that fires. */
    const DraftVerifyShape mixed = make_shape(0, 8, { 0, 8, 8, 8 }, 0, -1, { 0, 8, 5, 5 }, 1);
    const PlanFixture pm = plan_for(mixed);
    CHECK_EQ(pm.plan.rows[2], 5);
    DraftCostSample ran_deeper = sample_of(mixed, 40.0);
    ran_deeper.draft_round_rows[2] = 8;
    ran_deeper.draft_round_rows[3] = 8;
    ran_deeper.draft_rows_total = 24;
    CHECK_EQ(t.observe(completed_cycle(pm), ran_deeper), RAD_E_INVAL);
    CHECK(t.buckets().empty());

    /* The reader refuses the same impossibility in a file. */
    DraftCostProfile p;
    std::string err;
    json no_rows = bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0);
    no_rows["extent"] = { { "rounds", 3 }, { "draft_rows_total", 0 },
                          { "draft_round_rows", { 0, 0, 0, 0 } } };
    CHECK_EQ(draft_cost_profile_parse(profile_text({ no_rows }), &p, &err), RAD_E_INVAL);
    CHECK(has(err, "extent"));
}

TEST(an_unknown_dimension_is_refused_not_priced_as_zero) {
    /* `unknown` is the explicit "not measured" marker: a shape that lacks history, prefill, replay
     * or the plan's distribution is refused, never priced as if the zeros had been measured. */
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({ bucket_json(0, 8, 3, { 0, 8, 8, 8 }, 43.0) })),
                         runtime()),
             RAD_OK);

    DraftVerifyShape s = query(0, 8, 3, { 0, 8, 8, 8 });
    s.unknown = kDraftShapeHistoryUnknown;
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(s, &e), RAD_E_INVAL);
    CHECK(!e.valid);

    /* A bucket for the shape exists, so a zero-history shape would have been Measured; the flag
     * must not turn a missing measurement into that bucket's price. */
    DraftVerifyShape zero = query(0, 8, 3, { 0, 8, 8, 8 });
    CHECK_EQ(t.estimate(zero, &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
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

TEST(a_full_table_drops_and_counts_instead_of_growing) {
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(profile_text({})), runtime(), 2), RAD_OK);
    for (int k = 0; k <= 2; ++k) {
        /* rows is exactly verify_k + 1 long: index 0 unused, N_j = 4 above it. */
        std::vector<int32_t> rows((size_t)k + 1, 4);
        rows[0] = 0;
        const DraftVerifyShape s = query(0, 4, k, rows);
        const PlanFixture p = plan_for(s);
        CHECK_EQ(t.observe(completed_cycle(p), sample_of(s, 10.0 + k)), RAD_OK);
    }
    CHECK_EQ((int)t.buckets().size(), 2);
    CHECK_EQ(t.dropped(), 1);
}

TEST(the_documented_c8_gradient_is_reproduced_without_a_new_measurement) {
    /* The C8 case from the plan: depth 3 costs 36.3 ms, depth 5 costs 43.0 ms. These are the
     * documented figures replayed as profile buckets -- no new measurement is claimed here -- and
     * the model must price K3 strictly below K5 from those two buckets. The per-token reading of
     * the same figures (22.3 and 23.3 emitted tokens) is policy arithmetic, not a table property,
     * so it is not asserted here. */
    const double k3_ms = 36.3;
    const double k5_ms = 43.0;

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

/* The start budget is measured on arrival, not resolved: a card that held a foreign context when
 * the profile was written (or holds one now) resolves the same plan against a smaller budget. That
 * alone is not an identity change -- refusing it made the cost path unusable on any shared card --
 * so it binds and reports both values and the difference for the warning. */
TEST(a_smaller_start_budget_alone_is_not_a_placement_change) {
    const std::string same = " host=0 file=2147483648 dyn=1 res=9f2c41ab7e0d3355";
    DraftCostFingerprint a = runtime_fingerprint(), b = a;
    a.placement = "expert_tiered vram=17179869184" + same;
    b.placement = "expert_tiered vram=16865296384" + same;   /* 300 MiB less */
    std::string first;
    DraftCostPlacementDrift drift;
    CHECK(draft_cost_fingerprint_same(a, b, &first, &drift));
    CHECK(first.empty());
    CHECK(drift.budget_changed);
    CHECK(has(drift.budget, "rank 0"));
    CHECK(has(drift.budget, "profile=17179869184"));
    CHECK(has(drift.budget, "runtime=16865296384"));
    CHECK(has(drift.budget, "300.00 MiB smaller"));

    /* A larger budget is the same case: uninteresting while the plan is the same. */
    std::swap(a.placement, b.placement);
    std::string grew;
    DraftCostPlacementDrift up;
    CHECK(draft_cost_fingerprint_same(a, b, &grew, &up));
    CHECK(has(up.budget, "300.00 MiB larger"));

    /* Multiple ranks name the rank whose budget moved. */
    a.placement = "all_vram vram=17179869184" + same + ";" + "layer_offload vram=8589934592" + same;
    b.placement = "all_vram vram=17179869184" + same + ";" + "layer_offload vram=16528539648" + same;
    CHECK(draft_cost_fingerprint_same(a, b, &grew, &up));
    CHECK(has(up.budget, "rank 1"));
    CHECK(!has(up.budget, "rank 0"));

    /* An older, unfingerprinted placement string still compares as a whole. */
    a.placement = b.placement = "all_vram";
    CHECK(draft_cost_fingerprint_same(a, b, &grew, &up));
    CHECK(up.budget.empty());
}

/* A different plan behind a different budget is a different placement: the reason names the field
 * and says the budget was not the cause, and no tolerated drift is reported for a refused file. */
TEST(a_budget_that_resolves_another_plan_is_refused_by_field) {
    const std::string same = " host=0 file=0 dyn=1 res=9f2c41ab7e0d3355";
    DraftCostFingerprint a = runtime_fingerprint(), b = a;
    a.placement = "expert_tiered vram=17179869184" + same;
    b.placement = "all_vram vram=16865296384" + same;
    std::string first;
    DraftCostPlacementDrift drift;
    CHECK(!draft_cost_fingerprint_same(a, b, &first, &drift));
    CHECK(has(first, "placement[0].strategy"));
    CHECK(has(first, "profile=\"expert_tiered\""));
    CHECK(has(first, "runtime=\"all_vram\""));
    CHECK(has(first, "resolved-plan change"));
    CHECK(!drift.budget_changed);

    /* Same strategy and budget, different residency: a placement change on its own. */
    a.placement = "expert_tiered vram=17179869184" + same;
    b.placement = "expert_tiered vram=17179869184 host=0 file=0 dyn=1 res=0000000000000001";
    CHECK(!draft_cost_fingerprint_same(a, b, &first, &drift));
    CHECK(has(first, "placement[0].residency"));
}

/* What configure() does with that verdict: bind and report, or stay inactive and name the field. */
TEST(the_table_binds_a_drifted_start_budget_and_names_it) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.fingerprint.placement = "all_vram vram=17179869184 host=0 file=0 dyn=1 res=00000000000000ff";
    DraftCostRuntime rt;
    rt.fingerprint = p.fingerprint;
    rt.fingerprint.placement = "all_vram vram=16865296384 host=0 file=0 dyn=1 res=00000000000000ff";

    DraftCostTable t;
    CHECK_EQ(t.configure(p, rt), RAD_OK);
    CHECK(t.active());
    CHECK(t.reason().empty());
    CHECK(has(t.budget_drift(), "300.00 MiB smaller"));

    rt.fingerprint.placement = "expert_tiered vram=16865296384 host=0 file=0 dyn=1 res=00000000000000ff";
    CHECK_EQ(t.configure(p, rt), RAD_OK);
    CHECK(!t.active());
    CHECK(has(t.reason(), "placement[0].strategy"));
    CHECK(has(t.reason(), "resolved-plan change"));
    CHECK(t.budget_drift().empty());   /* clear() does not carry the earlier run's warning */
}

/* The fingerprint's gpus field is derived from the ranks' device names, not typed by the user:
 * identical cards keep the documented "Nx NAME" form, a mixed set stays comma-separated in rank
 * order so the string still names the deployment. */
TEST(the_gpus_field_names_the_ranks_in_order) {
    CHECK_EQ(draft_cost_gpus({ "Radeon AI PRO R9700", "Radeon AI PRO R9700" }),
             std::string("2x Radeon AI PRO R9700"));
    CHECK_EQ(draft_cost_gpus({ "A", "B" }), std::string("A, B"));
    CHECK_EQ(draft_cost_gpus({}), std::string("none"));
    CHECK_EQ(draft_cost_gpus({ "one" }), std::string("1x one"));
}

/* ------------------------------------------------------------------ the writer */

/* A parsed profile read back out is a byte-for-byte identity of what the reader understood:
 * fingerprint, execution, every priced field of every bucket and its origin mix. The writer's
 * output is parsed with the reader, so a roundtrip that drifts fails here. */
TEST(the_writer_round_trips_what_the_reader_understood) {
    const DraftCostProfile p =
        parse_or_fail(profile_text({ bucket_json(2, 8, 3, { 0, 8, 8, 8 }, 41.5, 0.7, 5,
                                                  { "ahead" }, "play", 32),
                                     bucket_json(0, 4, 2, { 0, 4, 4 }, 12.0) }));
    std::string text, err;
    int32_t skipped = -1;
    CHECK_EQ(draft_cost_profile_write(p, &text, &skipped, &err), RAD_OK);
    CHECK_EQ(skipped, 0);
    if (!err.empty()) fprintf(stderr, "  write error: %s\n", err.c_str());

    DraftCostProfile q;
    CHECK_EQ(draft_cost_profile_parse(text, &q, &err), RAD_OK);
    CHECK(draft_cost_fingerprint_same(p.fingerprint, q.fingerprint, nullptr));
    CHECK(p.execution == q.execution);
    CHECK_EQ((int)q.buckets.size(), (int)p.buckets.size());
    for (size_t i = 0; i < p.buckets.size() && i < q.buckets.size(); ++i) {
        const DraftCostBucket& a = p.buckets[i];
        const DraftCostBucket& b = q.buckets[i];
        CHECK_EQ(a.shape.ctx_class, b.shape.ctx_class);
        CHECK_EQ(a.shape.n_rows, b.shape.n_rows);
        CHECK_EQ(a.shape.verify_k, b.shape.verify_k);
        CHECK_EQ(a.shape.verify_tokens, b.shape.verify_tokens);
        CHECK_EQ(a.shape.history_tokens, b.shape.history_tokens);
        CHECK_EQ(a.shape.pipeline_depth, b.shape.pipeline_depth);
        CHECK_EQ(a.shape.draft_k, b.shape.draft_k);
        CHECK_EQ(a.flags, b.flags);
        CHECK_EQ(a.rounds, b.rounds);
        CHECK_EQ(a.draft_rows_total, b.draft_rows_total);
        CHECK_EQ(a.n, b.n);
        CHECK_EQ(a.mean_ms, b.mean_ms);
        CHECK_EQ(a.unc_ms, b.unc_ms);
        for (int32_t j = 0; j < a.shape.n_rows; ++j) {
            CHECK_EQ(a.origin_set[j], b.origin_set[j]);
            CHECK_EQ(a.shape.rows[j], b.shape.rows[j]);
            CHECK_EQ(a.shape.draft_rows[j], b.shape.draft_rows[j]);
        }
    }
}

/* `unknown` is not a file value. A bucket whose cycle nobody observed the replay state of is
 * refused by name, not written as `none` by the shape's default: the file would claim a plain run
 * that was never evidenced. */
TEST(the_writer_refuses_a_bucket_whose_replay_state_was_never_observed) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostBucket b;
    b.shape = query(0, 4, 2, { 0, 4, 4 });
    b.shape.replay = DraftReplay::Unknown;
    b.rounds = 2;
    b.draft_round_rows[1] = 4;
    b.draft_round_rows[2] = 4;
    b.draft_rows_total = 8;
    b.mean_ms = 10.0;
    b.n = 3;
    for (int32_t j = 0; j < b.shape.n_rows; ++j) b.origin_set[j] = ob(DraftOrigin::Head);
    p.buckets.push_back(b);

    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_E_INVAL);
    CHECK(err.find("replay") != std::string::npos);
    CHECK(text.empty());
}

/* No invented provenance: a bucket whose samples never captured an origin mix cannot be written.
 * The alternative -- defaulting rows to `none` -- would let a head row price a lookup row. */
TEST(the_writer_refuses_a_bucket_with_no_captured_origins) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostBucket b;
    b.shape = query(0, 4, 2, { 0, 4, 4 });
    b.rounds = 2;
    b.draft_round_rows[1] = 4;
    b.draft_round_rows[2] = 4;
    b.draft_rows_total = 8;
    b.mean_ms = 10.0;
    b.n = 3;
    /* origin_set stays at zero: no sample captured one. */
    p.buckets.push_back(b);

    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_E_INVAL);
    CHECK(err.find("origin") != std::string::npos);
}

/* The file has no name for a runtime-only diagnostic flag. Such a bucket is a diagnostic, not a
 * measurement: it is left out and counted, never written under a guessed name. */
TEST(the_writer_counts_buckets_whose_flag_it_cannot_name) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostBucket good;
    good.shape = query(0, 4, 2, { 0, 4, 4 });
    good.rounds = 2;
    good.draft_round_rows[1] = 4;
    good.draft_round_rows[2] = 4;
    good.draft_rows_total = 8;
    good.mean_ms = 10.0;
    good.n = 3;
    for (int32_t j = 0; j < good.shape.n_rows; ++j) good.origin_set[j] = ob(DraftOrigin::Head);
    p.buckets.push_back(good);

    DraftCostBucket truncated_bucket = good;
    truncated_bucket.flags = kDraftSampleShortened;   /* no file name */
    p.buckets.push_back(truncated_bucket);

    std::string text, err;
    int32_t skipped = 0;
    CHECK_EQ(draft_cost_profile_write(p, &text, &skipped, &err), RAD_OK);
    CHECK_EQ(skipped, 1);
    DraftCostProfile q;
    CHECK_EQ(draft_cost_profile_parse(text, &q, &err), RAD_OK);
    CHECK_EQ((int)q.buckets.size(), 1);
}

/* The output path is created, never replaced: an existing profile survives, the temporary is
 * gone, and the failure is visible. */
TEST(write_file_never_overwrites_an_existing_profile) {
    char dir[] = "/tmp/rad-draft-profile-XXXXXX";
    CHECK(::mkdtemp(dir) != nullptr);
    const std::string path = std::string(dir) + "/p.json";
    {
        std::ofstream f(path, std::ios::binary);
        f << "{\"format\":\"rad-draft-profile/3\"}";
    }
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    /* A profile with no bucket is refused by the writer (empty run, below), so the "fresh path"
     * half of this test needs one. */
    DraftCostBucket b;
    b.shape = query(0, 4, 1, { 0, 4 });
    b.rounds = 1;
    b.draft_round_rows[1] = 4;
    b.draft_rows_total = 4;
    b.mean_ms = 10.0;
    b.n = 3;
    for (int32_t j = 0; j < b.shape.n_rows; ++j) b.origin_set[j] = ob(DraftOrigin::Head);
    p.buckets.push_back(b);
    std::string err;
    const int rc = draft_cost_profile_write_file(path, p, nullptr, &err);
    CHECK_EQ(rc, RAD_E_IO);
    CHECK(err.find("already exists") != std::string::npos);

    std::ifstream f(path, std::ios::binary);
    std::string kept((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    CHECK_EQ(kept, std::string("{\"format\":\"rad-draft-profile/3\"}"));

    /* A fresh path publishes the whole file. */
    const std::string fresh = std::string(dir) + "/fresh.json";
    CHECK_EQ(draft_cost_profile_write_file(fresh, p, nullptr, &err), RAD_OK);
    DraftCostProfile q;
    CHECK_EQ(draft_cost_profile_read(fresh, &q, &err), RAD_OK);
}

/* A run that measured nothing is not a calibration: publishing an empty profile would let the
 * next start load it as a baseline whose every estimate is the fallback. The writer refuses, and a
 * profile whose every bucket is diagnostic (left out of the file) is empty for this purpose.
 * The reader still accepts an empty bucket list -- a hand-written file is not a measurement. */
TEST(the_writer_refuses_a_profile_with_no_measured_bucket) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;

    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_E_INVAL);
    CHECK(err.find("no bucket") != std::string::npos);

    char dir[] = "/tmp/rad-draft-profile-XXXXXX";
    CHECK(::mkdtemp(dir) != nullptr);
    const std::string path = std::string(dir) + "/empty.json";
    CHECK_EQ(draft_cost_profile_write_file(path, p, nullptr, &err), RAD_E_INVAL);
    std::ifstream f(path, std::ios::binary);
    CHECK(!f.good());   /* no file at all, not an empty one */

    /* Only diagnostic buckets a file cannot name: skipped on the way out, so it has none. */
    DraftCostBucket diag;
    diag.shape = query(0, 4, 1, { 0, 4 });
    diag.rounds = 1;
    diag.draft_round_rows[1] = 4;
    diag.draft_rows_total = 4;
    diag.mean_ms = 4.0;
    diag.n = 1;
    for (int32_t j = 0; j < diag.shape.n_rows; ++j) diag.origin_set[j] = ob(DraftOrigin::Head);
    diag.flags = kDraftSampleShortened;   /* kept for diagnosis, no name in the file */
    p.buckets.push_back(diag);
    int32_t skipped = 0;
    CHECK_EQ(draft_cost_profile_write(p, &text, &skipped, &err), RAD_E_INVAL);
    CHECK(err.find("no bucket") != std::string::npos);
    CHECK_EQ(skipped, 0);   /* the skip is only counted for a file that is written */
}

/* The origin mix follows the drafter that really filled the rows: a serial deployment's fallback
 * fills the rows the head left empty, so a serial bucket may carry lookup rows. Requiring the
 * lookup to be the loaded drafter refused every such export. The reverse -- a lookup-only bucket
 * naming a head row -- is still a broken record. */
TEST(a_serial_bucket_may_carry_lookup_rows) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostBucket b;
    b.shape = query(0, 4, 1, { 0, 4 });
    b.rounds = 1;
    b.draft_round_rows[1] = 4;
    b.draft_rows_total = 4;
    b.mean_ms = 10.0;
    b.n = 2;
    b.origin_set[0] = ob(DraftOrigin::Head);
    b.origin_set[1] = ob(DraftOrigin::Lookup);
    b.origin_set[2] = ob(DraftOrigin::Lookup);
    b.origin_set[3] = ob(DraftOrigin::None);
    p.buckets.push_back(b);

    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_OK);
    DraftCostProfile q;
    CHECK_EQ(draft_cost_profile_parse(text, &q, &err), RAD_OK);
    CHECK_EQ((int)q.buckets.size(), 1);
    CHECK(q.buckets[0].origin_set[1] == ob(DraftOrigin::Lookup));

    /* A head row under an n-gram-only bucket is still refused: the head was not the drafter. */
    p.buckets[0].shape.drafter = DraftDrafter::Ngram;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_E_INVAL);
    CHECK(err.find("origins do not fit") != std::string::npos);
}

/* Two cycles in one bucket need not agree on where a row's window came from: the head drops it in
 * one and the lookup fallback fills it in the next, with the same executed distribution. A single
 * value per row would either pick one sample's mix or throw the whole export away over a
 * difference that is not part of the key. The set keeps both, and the file says so. */
TEST(a_bucket_whose_samples_disagree_on_a_rows_origin_keeps_both) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostBucket b;
    b.shape = query(0, 4, 1, { 0, 4 });
    b.rounds = 1;
    b.draft_round_rows[1] = 4;
    b.draft_rows_total = 4;
    b.mean_ms = 10.0;
    b.n = 2;
    b.origin_set[0] = ob(DraftOrigin::Head) | ob(DraftOrigin::Lookup);
    b.origin_set[1] = ob(DraftOrigin::Lookup);
    b.origin_set[2] = ob(DraftOrigin::Head);
    b.origin_set[3] = ob(DraftOrigin::None);
    p.buckets.push_back(b);

    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_OK);
    CHECK(text.find("\"head+lookup\"") != std::string::npos);

    DraftCostProfile q;
    CHECK_EQ(draft_cost_profile_parse(text, &q, &err), RAD_OK);
    REQUIRE(q.buckets.size() == 1);
    CHECK_EQ((int)q.buckets[0].origin_set[0], (int)(ob(DraftOrigin::Head) | ob(DraftOrigin::Lookup)));
    /* Read -> write round-trips the set, order and all. */
    std::string again;
    CHECK_EQ(draft_cost_profile_write(q, &again, nullptr, &err), RAD_OK);
    CHECK_EQ(again, text);

    /* The set still answers to the drafter: a head bit under an n-gram bucket is refused. */
    p.buckets[0].shape.drafter = DraftDrafter::Ngram;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_E_INVAL);
    CHECK(err.find("origins do not fit") != std::string::npos);

    /* And an unnameable member is malformed, not a mix to guess at. */
    json bad = json::parse(again);
    bad["buckets"][0]["shape"]["origins"][0] = "head+speculative";
    CHECK_EQ(draft_cost_profile_parse(bad.dump(), &q, &err), RAD_E_INVAL);
    CHECK(has(err, "unknown origin"));
}

/* The same at the table's recording seam: two samples of one bucket naming different origins widen
 * the set instead of dropping the bucket and costing the caller the whole profile. */
TEST(the_table_widens_the_origin_set_instead_of_dropping_the_bucket) {
    DraftCostTable t;
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostRuntime rt;
    rt.fingerprint = p.fingerprint;
    rt.execution = p.execution;
    REQUIRE(t.configure(p, rt) == RAD_OK);

    const DraftVerifyShape shape = query(0, 4, 1, { 0, 4 });
    const PlanFixture pf = plan_for(shape);
    DraftCostSample s = sample_of(shape, 10.0);
    s.origins_valid = true;
    for (int32_t i = 0; i < shape.n_rows; ++i) s.origins[i] = DraftOrigin::Head;

    CHECK_EQ(t.observe(completed_cycle(pf), s), RAD_OK);
    s.origins[1] = DraftOrigin::Lookup;   /* the fallback filled row 1 this cycle */
    CHECK_EQ(t.observe(completed_cycle(pf), s), RAD_OK);

    const std::vector<DraftCostBucket>& bs = t.buckets();
    REQUIRE(bs.size() == 1);
    CHECK_EQ(bs[0].n, 2);
    CHECK_EQ((int)bs[0].origin_set[1], (int)(ob(DraftOrigin::Head) | ob(DraftOrigin::Lookup)));
    CHECK_EQ((int)bs[0].origin_set[0], (int)ob(DraftOrigin::Head));

    /* And that bucket writes: the export is not lost over the difference. */
    DraftCostProfile out;
    out.fingerprint = p.fingerprint;
    out.execution = p.execution;
    out.buckets = bs;
    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(out, &text, nullptr, &err), RAD_OK);
    CHECK(text.find("\"head+lookup\"") != std::string::npos);
}

/* The backend's round model, not this cycle's distribution: a uniform cycle on the serial backend
 * still writes `shrinking`. The distribution itself is in the bucket key, so a uniform bucket
 * does not price a query planning a different one. */
TEST(a_uniform_bucket_of_a_shrinking_backend_is_still_one_profile) {
    DraftCostProfile p;
    p.fingerprint = runtime_fingerprint();
    p.execution = DraftCostExecution::Shrinking;
    DraftCostBucket b;
    b.shape = query(0, 4, 2, { 0, 4, 4 });
    for (int32_t j = 0; j < b.shape.n_rows; ++j) b.origin_set[j] = ob(DraftOrigin::Head);
    b.rounds = 2;
    b.draft_round_rows[1] = 4;
    b.draft_round_rows[2] = 4;
    b.draft_rows_total = 8;
    b.mean_ms = 10.0;
    b.n = 3;
    p.buckets.push_back(b);

    std::string text, err;
    CHECK_EQ(draft_cost_profile_write(p, &text, nullptr, &err), RAD_OK);
    CHECK(text.find("\"execution\": \"shrinking\"") != std::string::npos);

    DraftCostRuntime r = runtime();
    r.execution = DraftCostExecution::Rectangular;
    DraftCostTable t;
    CHECK_EQ(t.configure(parse_or_fail(text), r), RAD_OK);
    CHECK(!t.active());   /* the backend is part of the identity, strictly */

    r.execution = DraftCostExecution::Shrinking;
    CHECK_EQ(t.configure(parse_or_fail(text), r), RAD_OK);
    CHECK(t.active());
    DraftCostEstimate e;
    CHECK_EQ(t.estimate(query(0, 4, 2, { 0, 4, 4 }), &e), RAD_OK);
    CHECK(e.source == DraftCostSource::Measured);
    /* A query planning a different distribution is a different key, not a different profile: no
     * bucket matches and the fallback borrows no anchor across another executed extent, so the
     * caller's named baseline is the answer -- never the bucket's price. */
    DraftCostEstimate other;
    CHECK_EQ(t.estimate(make_shape(0, 4, { 0, 4, 4 }, 0, 2, { 0, 4, 2 }, 0), &other), RAD_OK);
    CHECK(!other.valid);
}

/* The backend's round model: a serial head that runs rounds per row is shrinking even when a
 * cycle's K are uniform; a block drafter, no head, or drafting off is rectangular. */
TEST(the_execution_is_the_backend_round_model_not_a_cycle) {
    CHECK(draft_cost_execution(true, 3) == DraftCostExecution::Shrinking);
    CHECK(draft_cost_execution(true, 1) == DraftCostExecution::Shrinking);
    CHECK(draft_cost_execution(true, 0) == DraftCostExecution::Rectangular);   /* drafting off */
    CHECK(draft_cost_execution(false, 3) == DraftCostExecution::Rectangular);  /* block / no head */
}

/* A profile stamped for the other backend is refused by name, not restamped; the same file is
 * accepted where its execution matches. */
TEST(a_rectangular_profile_is_refused_on_a_shrinking_backend_by_name) {
    const DraftCostProfile p =
        parse_or_fail(profile_text({ bucket_json(0, 4, 2, { 0, 4, 4 }, 12.0) }));
    DraftCostRuntime r = runtime();
    r.execution = DraftCostExecution::Shrinking;
    DraftCostTable t;
    CHECK_EQ(t.configure(p, r), RAD_OK);
    CHECK(!t.active());
    CHECK(t.reason().find("execution") != std::string::npos);

    r.execution = DraftCostExecution::Rectangular;
    CHECK_EQ(t.configure(p, r), RAD_OK);
    CHECK(t.active());
}

RAD_TEST_MAIN()
