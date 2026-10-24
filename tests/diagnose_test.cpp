/* diagnose_test.cpp -- the --diagnose mode's arithmetic and report (core/diag/diagnose.h).
 *
 * No device, no model, no scheduler. The step ring is held to its lap accounting, the quantiles
 * to nearest rank, and a phase report built from hand-made steps, counters and request timings to
 * numbers worked out here by hand. What the report cannot know has to read as null, never 0.
 */
#include "rad_test.h"

#include "diag/diagnose.h"

#include "rad_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace rad;
using namespace rad::diag;

namespace {

StepRecord step(int64_t n, int phase, int n_seq, int prefill, int decode, double ms, int64_t end_ns) {
    StepRecord r;
    r.step = n;
    r.phase = phase;
    r.n_seq = n_seq;
    r.prefill_tok = prefill;
    r.decode_tok = decode;
    r.ns = (int64_t)std::llround(ms * 1e6);
    r.end_ns = end_ns;
    return r;
}

bool near(double a, double b, double tol = 1e-3) { return std::fabs(a - b) <= tol; }

}  /* namespace */

/* ------------------------------------------------------------------ ring */

TEST(ring_reads_what_came_after_a_mark) {
    StepRing ring(8);
    for (int i = 0; i < 5; ++i) ring.push(step(i, RAD_PHASE_DECODE, 1, 0, 1, 1.0, i));
    CHECK_EQ(ring.count(), (uint64_t)5);
    std::vector<StepRecord> out;
    CHECK_EQ(ring.copy_since(2, &out), (uint64_t)0);
    REQUIRE(out.size() == 3);
    CHECK_EQ(out[0].step, (int64_t)2);
    CHECK_EQ(out[2].step, (int64_t)4);
    out.clear();
    CHECK_EQ(ring.copy_since(5, &out), (uint64_t)0);   /* nothing new */
    CHECK(out.empty());
}

TEST(ring_counts_what_it_overwrote) {
    StepRing ring(4);
    for (int i = 0; i < 10; ++i) ring.push(step(i, RAD_PHASE_DECODE, 1, 0, 1, 1.0, i));
    std::vector<StepRecord> out;
    /* 10 pushed into 4 slots: 6..9 remain; a reader from 3 lost 3, 4 and 5. */
    CHECK_EQ(ring.copy_since(3, &out), (uint64_t)3);
    REQUIRE(out.size() == 4);
    CHECK_EQ(out[0].step, (int64_t)6);
    CHECK_EQ(out[3].step, (int64_t)9);
}

TEST(ring_of_no_capacity_records_nothing) {
    StepRing ring;
    ring.push(step(0, RAD_PHASE_DECODE, 1, 0, 1, 1.0, 0));
    CHECK_EQ(ring.count(), (uint64_t)0);
    std::vector<StepRecord> out;
    CHECK_EQ(ring.copy_since(0, &out), (uint64_t)0);
    CHECK(out.empty());
}

/* ------------------------------------------------------------------ quantiles */

TEST(quantiles_are_nearest_rank) {
    std::vector<double> v;
    for (int i = 100; i >= 1; --i) v.push_back(i);
    const Quantiles q = quantiles(v);
    CHECK_EQ(q.n, (int64_t)100);
    CHECK_EQ(q.min, 1.0);
    CHECK_EQ(q.p50, 50.0);
    CHECK_EQ(q.p90, 90.0);
    CHECK_EQ(q.p99, 99.0);
    CHECK_EQ(q.max, 100.0);
    CHECK(near(q.mean, 50.5));

    const Quantiles one = quantiles({ 7.0 });
    CHECK_EQ(one.p50, 7.0);
    CHECK_EQ(one.p99, 7.0);

    /* Three samples: ceil(0.5 * 3) = 2nd, ceil(0.9 * 3) = 3rd. A measured value, not a blend. */
    const Quantiles three = quantiles({ 3.0, 1.0, 2.0 });
    CHECK_EQ(three.p50, 2.0);
    CHECK_EQ(three.p90, 3.0);
}

TEST(no_samples_is_null_not_zero) {
    const json j = to_json(quantiles({}));
    CHECK_EQ(j["n"].get<int64_t>(), (int64_t)0);
    CHECK(j["p50"].is_null());
    CHECK(j["max"].is_null());
    CHECK(j["mean"].is_null());
}

/* ------------------------------------------------------------------ slow steps */

TEST(slow_steps_names_the_slowest_beside_the_median) {
    std::vector<StepRecord> s;
    for (int i = 0; i < 9; ++i) s.push_back(step(i, RAD_PHASE_DECODE, 2, 0, 2, 10.0 + i, i));
    s.push_back(step(9, RAD_PHASE_PREFILL, 1, 512, 0, 500.0, 9));   /* other shape: ignored */
    const json j = slow_steps(s, RAD_PHASE_DECODE);
    CHECK_EQ(j["steps"].get<int64_t>(), (int64_t)9);
    CHECK(near(j["typical"]["ms"].get<double>(), 14.0));            /* 10..18, the 5th */
    REQUIRE(j["slowest"].size() == 3);                               /* max(3, 1%) */
    CHECK_EQ(j["slowest"][0]["step"].get<int64_t>(), (int64_t)8);
    CHECK(near(j["slowest"][0]["ms"].get<double>(), 18.0));
    CHECK_EQ(j["slowest"][2]["step"].get<int64_t>(), (int64_t)6);

    const json none = slow_steps(s, RAD_PHASE_MIXED);
    CHECK_EQ(none["steps"].get<int64_t>(), (int64_t)0);
    CHECK(none["typical"].is_null());
    CHECK(none["slowest"].empty());
}

/* ------------------------------------------------------------------ phase report */

TEST(decode_phase_report_by_hand) {
    PhaseInput in;
    in.name = "decode_2";
    in.kind = PhaseKind::Decode;
    in.streams = 2;
    in.wall_ns = 2'500'000'000;

    /* Counters: 100 decode steps of 20 ms, 4 prefill steps; 300 drafts, 150 taken; position 1
     * reached 100 times and taken 80, position 2 reached 80 and taken 40. The `before` values are
     * there to be subtracted. */
    in.before.steps = 50;
    in.before.steps_decode = 40;
    in.before.step_ns_decode = 1'000'000'000;
    in.before.draft_tokens = 1000;
    in.before.draft_accepted = 600;
    in.after = in.before;
    in.after.steps = 154;
    in.after.steps_decode = 140;
    in.after.steps_prefill = 4;
    in.after.step_ns_decode = in.before.step_ns_decode + 100 * 20'000'000LL;
    in.after.draft_tokens = 1300;
    in.after.draft_accepted = 750;
    in.after.draft_depth_seen = 2;
    in.after.draft_pos_reached[0] = 100;
    in.after.draft_pos_accepted[0] = 80;
    in.after.draft_pos_reached[1] = 80;
    in.after.draft_pos_accepted[1] = 40;

    /* Steps: one prefill, then decode steps of 10, 10, 10, 30 ms committing 2, 2, 4, 2 tokens,
     * committed 10, 10, 20, 40 ms apart. */
    const int64_t ms = 1'000'000;
    in.steps.push_back(step(150, RAD_PHASE_PREFILL, 2, 40, 0, 50.0, 0));
    in.steps.push_back(step(151, RAD_PHASE_DECODE, 2, 0, 2, 10.0, 100 * ms));
    in.steps.push_back(step(152, RAD_PHASE_DECODE, 2, 0, 2, 10.0, 110 * ms));
    in.steps.push_back(step(153, RAD_PHASE_DECODE, 2, 0, 4, 10.0, 120 * ms));
    in.steps.push_back(step(154, RAD_PHASE_DECODE, 2, 0, 2, 30.0, 140 * ms));
    in.steps.push_back(step(155, RAD_PHASE_DECODE, 2, 0, 2, 10.0, 180 * ms));

    /* Requests: admitted at 5 ms after start, first token at 55 ms, last at 1055 ms. */
    for (int i = 0; i < 2; ++i) {
        RequestTiming r;
        r.start_ns = 0;
        r.admitted_ns = 5 * ms;
        r.first_ns = (55 + 10 * i) * ms;
        r.last_ns = (1055 + 10 * i) * ms;
        r.tokens = 101;
        r.prompt = 20;
        r.ok = true;
        in.requests.push_back(r);
    }

    const json j = phase_report(in);
    CHECK_EQ(j["kind"].get<std::string>(), std::string("decode"));
    CHECK(near(j["wall_s"].get<double>(), 2.5));

    const json& r = j["requests"];
    CHECK_EQ(r["ok"].get<int>(), 2);
    CHECK_EQ(r["tokens"].get<int64_t>(), (int64_t)202);
    CHECK(near(r["ttft_ms"]["p50"].get<double>(), 50.0));
    CHECK(near(r["ttft_ms"]["max"].get<double>(), 60.0));
    CHECK(near(r["queue_ms"]["p50"].get<double>(), 5.0));
    /* 100 tokens after the first over 1000 ms each. */
    CHECK(near(r["stream_tok_s"]["p50"].get<double>(), 100.0));
    /* 200 tokens over 55 ms .. 1065 ms. */
    CHECK(near(r["aggregate_tok_s"].get<double>(), 200.0 / 1.010));
    CHECK(!r.contains("prefill_tok_s"));

    const json& c = j["counters"];
    CHECK_EQ(c["steps"].get<int64_t>(), (int64_t)104);
    CHECK_EQ(c["steps_decode"].get<int64_t>(), (int64_t)100);
    CHECK(near(c["ms_per_decode_step"].get<double>(), 20.0));
    CHECK(near(c["draft_acceptance"].get<double>(), 0.5));
    REQUIRE(c["draft_by_position"].size() == 2);
    CHECK(near(c["draft_by_position"][0]["accepted"].get<double>(), 0.8));
    CHECK(near(c["draft_by_position"][1]["accepted"].get<double>(), 0.5));

    const json& s = j["steps"];
    CHECK_EQ(s["recorded"].get<int64_t>(), (int64_t)6);
    CHECK_EQ(s["lost"].get<int64_t>(), (int64_t)0);
    CHECK_EQ(s["decode_step_ms"]["n"].get<int64_t>(), (int64_t)5);
    CHECK(near(s["decode_step_ms"]["p50"].get<double>(), 10.0));
    CHECK(near(s["decode_step_ms"]["max"].get<double>(), 30.0));
    CHECK(near(s["decode_tok_per_step"].get<double>(), 12.0 / 5.0));
    /* Emission gaps 10, 10, 20, 40 ms; the first decode step has nothing before it. */
    CHECK_EQ(s["emission_interval_ms"]["n"].get<int64_t>(), (int64_t)4);
    CHECK(near(s["emission_interval_ms"]["p50"].get<double>(), 10.0));
    CHECK(near(s["emission_interval_ms"]["max"].get<double>(), 40.0));
    CHECK(near(s["jitter_emission_p99_over_p50"].get<double>(), 4.0));
    /* Step times 10,10,10,30,10: mean 14, sample sd sqrt(320/4) = 8.944, CV 0.639. */
    CHECK(near(s["jitter_step_cv"].get<double>(), std::sqrt(80.0) / 14.0));
    /* Rates per emission: 2/0.01, 4/0.02, 2/0.04 -> 200, 200, 200, 50; the slowest is 50. */
    CHECK(near(s["tok_s_1pct_low"].get<double>(), 50.0));
    CHECK_EQ(s["slow_decode_steps"]["slowest"][0]["step"].get<int64_t>(), (int64_t)154);
    CHECK(!s.contains("prefill_chunk_ms"));
}

TEST(prefill_phase_reports_prefill_rate_and_chunks) {
    PhaseInput in;
    in.name = "prefill";
    in.kind = PhaseKind::Prefill;
    const int64_t ms = 1'000'000;
    in.steps.push_back(step(1, RAD_PHASE_PREFILL, 1, 4096, 0, 100.0, 100 * ms));
    in.steps.push_back(step(2, RAD_PHASE_PREFILL, 1, 4096, 0, 120.0, 220 * ms));
    in.steps.push_back(step(3, RAD_PHASE_MIXED, 1, 1808, 1, 60.0, 280 * ms));
    RequestTiming r;
    r.start_ns = 0;
    r.admitted_ns = 0;
    r.first_ns = 400 * ms;
    r.last_ns = 400 * ms;
    r.tokens = 1;
    r.prompt = 10000;
    r.ok = true;
    in.requests.push_back(r);

    const json j = phase_report(in);
    const json& q = j["requests"];
    CHECK(near(q["ttft_ms"]["p50"].get<double>(), 400.0));
    CHECK(near(q["prefill_tok_s"]["p50"].get<double>(), 25000.0));
    /* One token: no stream rate, and no aggregate decode rate for a prefill phase. */
    CHECK_EQ(q["stream_tok_s"]["n"].get<int64_t>(), (int64_t)0);
    CHECK(q["aggregate_tok_s"].is_null());
    const json& s = j["steps"];
    CHECK_EQ(s["prefill_chunk_ms"]["n"].get<int64_t>(), (int64_t)3);
    CHECK(near(s["prefill_chunk_ms"]["max"].get<double>(), 120.0));
    /* Nothing decoded on its own: every decode statistic is absent, not zero. */
    CHECK_EQ(s["decode_step_ms"]["n"].get<int64_t>(), (int64_t)0);
    CHECK(s["decode_tok_per_step"].is_null());
    CHECK(s["jitter_step_cv"].is_null());
    CHECK(s["tok_s_1pct_low"].is_null());
    CHECK(j["counters"]["ms_per_decode_step"].is_null());
    CHECK(j["counters"]["draft_acceptance"].is_null());
}

TEST(cached_prompt_tokens_are_not_prefill_throughput) {
    PhaseInput in;
    in.kind = PhaseKind::Prefill;
    RequestTiming r;
    r.admitted_ns = 0;
    r.first_ns = 1'000'000'000;
    r.tokens = 1;
    r.prompt = 1000;
    r.cached = 1000;   /* all of it from the cache: nothing was computed */
    in.requests.push_back(r);
    const json j = phase_report(in);
    CHECK_EQ(j["requests"]["prefill_tok_s"]["n"].get<int64_t>(), (int64_t)0);
    CHECK_EQ(j["requests"]["ttft_ms"]["n"].get<int64_t>(), (int64_t)1);
}

TEST(lost_records_and_cache_hits_are_reported) {
    PhaseInput in;
    in.steps_lost = 17;
    RequestTiming r;
    r.prompt = 100;
    r.cached = 32;
    in.requests.push_back(r);
    const json j = phase_report(in);
    CHECK_EQ(j["steps"]["lost"].get<int64_t>(), (int64_t)17);
    CHECK_EQ(j["requests"]["cached_prompt_tokens"].get<int64_t>(), (int64_t)32);
}

TEST(stopped_on_eos_is_counted_apart_from_length) {
    PhaseInput in;
    for (int i = 0; i < 3; ++i) {
        RequestTiming r;
        r.ok = true;
        r.stopped = i < 2;
        in.requests.push_back(r);
    }
    CHECK_EQ(phase_report(in)["requests"]["stopped_on_eos"].get<int>(), 2);
}

/* ------------------------------------------------------------------ runs and interference */

TEST(runs_summarise_to_median_and_range) {
    std::vector<json> runs;
    for (double v : { 30.0, 10.0, 20.0 }) {
        json r;
        r["requests"]["aggregate_tok_s"] = v;
        r["counters"]["draft_acceptance"] = nullptr;   /* no run measured it */
        runs.push_back(r);
    }
    runs[1]["requests"]["ttft_ms"]["p50"] = 5.0;      /* one run only */
    const json s = summarize_runs(runs);
    CHECK_EQ(s["runs"].get<int64_t>(), (int64_t)3);
    CHECK(near(s["requests.aggregate_tok_s"]["median"].get<double>(), 20.0));
    CHECK(near(s["requests.aggregate_tok_s"]["min"].get<double>(), 10.0));
    CHECK(near(s["requests.aggregate_tok_s"]["max"].get<double>(), 30.0));
    CHECK(s["counters.draft_acceptance"].is_null());
    CHECK(near(s["requests.ttft_ms.p50"]["median"].get<double>(), 5.0));

    const json p = summarize_paths(runs, { "/requests/aggregate_tok_s" });
    CHECK_EQ(p.size(), (size_t)2);   /* runs, and the one path */
}

TEST(interference_finds_the_gap_the_arrival_made) {
    const int64_t ms = 1'000'000;
    std::vector<StepRecord> s;
    /* Emissions every 10 ms up to 100 ms, then a 70 ms gap while a prompt prefills (one prefill
     * step without decode tokens in it), then 10 ms again. */
    int64_t n = 0;
    for (int64_t t = 10; t <= 100; t += 10) s.push_back(step(n++, RAD_PHASE_DECODE, 4, 0, 4, 9.0, t * ms));
    s.push_back(step(n++, RAD_PHASE_PREFILL, 1, 8192, 0, 60.0, 160 * ms));
    for (int64_t t = 170; t <= 200; t += 10) s.push_back(step(n++, RAD_PHASE_DECODE, 5, 0, 5, 9.0, t * ms));
    const json j = interference_report(s, 105 * ms, 165 * ms);
    CHECK(near(j["window_ms"].get<double>(), 60.0));
    CHECK(near(j["typical_gap_ms"].get<double>(), 10.0));
    CHECK(near(j["longest_gap_ms"].get<double>(), 70.0));
    CHECK(near(j["stall_ratio"].get<double>(), 7.0));
    CHECK_EQ(j["emissions_in_window"].get<int64_t>(), (int64_t)0);

    /* No emissions before the window: nothing to compare with. */
    const json k = interference_report(s, 0, 5 * ms);
    CHECK(k["typical_gap_ms"].is_null());
    CHECK(k["stall_ratio"].is_null());
}

/* ------------------------------------------------------------------ workload */

TEST(synthetic_document_is_sized_deterministic_and_salted) {
    const std::string a = synthetic_document(5000, 1);
    CHECK_EQ(a.size(), (size_t)5000);
    CHECK(a == synthetic_document(5000, 1));
    const std::string b = synthetic_document(5000, 2);
    CHECK(a != b);
    /* Different from the first characters on, or a prefix cache would serve one from the other. */
    CHECK(a.compare(0, 64, b, 0, 64) != 0);
    CHECK(synthetic_document(0, 1).empty());
}

TEST(decode_prompts_differ_per_stream) {
    for (int i = 0; i < 8; ++i)
        for (int k = i + 1; k < 8; ++k) CHECK(decode_prompt(i) != decode_prompt(k));
    CHECK(!decode_prompt(0).empty());
}

TEST(builtin_corpus_covers_the_categories) {
    const std::vector<CorpusEntry> c = builtin_corpus();
    for (const char* cat : { "code", "reasoning", "prose", "json", "file_edit", "summarization" }) {
        int n = 0;
        for (const auto& e : c)
            if (e.category == cat) {
                ++n;
                CHECK(e.messages.is_array() && !e.messages.empty());
                CHECK(!e.messages[0]["content"].get<std::string>().empty());
            }
        CHECK(n >= 2);
    }
    /* Same text every time: a corpus that changed between runs would be a different workload. */
    CHECK(builtin_corpus()[0].messages == c[0].messages);
}

TEST(salt_makes_the_first_message_unique) {
    const json m = json::array({ json{ { "role", "system" }, { "content", "be brief" } },
                                 json{ { "role", "user" }, { "content", "hi" } } });
    const json a = salt_messages(m, "t1");
    const json b = salt_messages(m, "t2");
    CHECK_EQ(a[0]["content"].get<std::string>(), std::string("[t1]\nbe brief"));
    CHECK(a[0]["content"] != b[0]["content"]);
    CHECK(a[1] == m[1]);   /* only the head changes */
    CHECK(m[0]["content"] == "be brief");   /* the input is not touched */

    const json parts = json::array({ json{ { "role", "user" },
                                           { "content", json::array({ json{ { "type", "text" }, { "text", "x" } } }) } } });
    const json p = salt_messages(parts, "t3");
    REQUIRE(p[0]["content"].size() == 2);
    CHECK_EQ(p[0]["content"][0]["text"].get<std::string>(), std::string("[t3]\n"));
}

namespace {
struct TmpFile {
    std::string path;
    explicit TmpFile(const std::string& text) {
        char t[] = "/tmp/diag_test.XXXXXX";
        const int fd = mkstemp(t);
        path = t;
        FILE* f = fdopen(fd, "wb");
        fwrite(text.data(), 1, text.size(), f);
        fclose(f);
    }
    ~TmpFile() { std::remove(path.c_str()); }
};
}  /* namespace */

TEST(corpus_file_reads_messages_and_prompts) {
    TmpFile f("# a comment\n"
              "{\"category\": \"code\", \"messages\": [{\"role\": \"user\", \"content\": \"write\"}]}\n"
              "\n"
              "{\"category\": \"prose\", \"prompt\": \"tell\", \"max_tokens\": 99}\n");
    std::vector<CorpusEntry> c;
    std::string why;
    REQUIRE(load_corpus(f.path, &c, &why) == RAD_OK);
    REQUIRE(c.size() == 2);
    CHECK_EQ(c[0].category, std::string("code"));
    CHECK_EQ(c[0].max_tokens, 0);
    CHECK_EQ(c[1].messages[0]["content"].get<std::string>(), std::string("tell"));
    CHECK_EQ(c[1].max_tokens, 99);
}

TEST(corpus_file_refuses_a_bad_line_and_names_it) {
    std::vector<CorpusEntry> c;
    std::string why;
    {
        TmpFile f("{\"category\": \"code\", \"prompt\": \"a\"}\n{\"prompt\": \"no category\"}\n");
        CHECK_EQ(load_corpus(f.path, &c, &why), RAD_E_INVAL);
        CHECK(why.find(":2:") != std::string::npos);
    }
    {
        TmpFile f("{\"category\": \"code\"}\n");
        CHECK_EQ(load_corpus(f.path, &c, &why), RAD_E_INVAL);
    }
    {
        TmpFile f("{\"category\": \"code\", \"messages\": [{\"content\": \"no role\"}]}\n");
        CHECK_EQ(load_corpus(f.path, &c, &why), RAD_E_INVAL);
    }
    {
        TmpFile f("{\"category\": \"code\", \"prompt\": \"a\", \"max_tokens\": 0}\n");
        CHECK_EQ(load_corpus(f.path, &c, &why), RAD_E_INVAL);
    }
    {
        TmpFile f("not json\n");
        CHECK_EQ(load_corpus(f.path, &c, &why), RAD_E_INVAL);
    }
    {
        TmpFile f("# only a comment\n");
        CHECK_EQ(load_corpus(f.path, &c, &why), RAD_E_INVAL);
    }
    CHECK_EQ(load_corpus("/nonexistent/diag_corpus.jsonl", &c, &why), RAD_E_IO);
    CHECK(c.empty());   /* a refused file leaves the output alone */
}

/* ------------------------------------------------------------------ parity */

namespace {
/* A report with one category, one concurrency point and one prefill point, one run each, whose
 * requests produced `toks`. */
json report_with(const std::vector<std::vector<int32_t>>& toks, const std::string& fp = "abc") {
    PhaseInput in;
    for (const auto& t : toks) {
        RequestTiming r;
        r.output = t;
        in.requests.push_back(r);
    }
    const json run = phase_report(in);
    json rep;
    rep["workload"] = { { "fingerprint", fp } };
    rep["categories"] = json::array({ { { "category", "code" }, { "runs", json::array({ run }) } } });
    rep["concurrency"] = json::array({ { { "streams", 2 }, { "runs", json::array({ run }) } } });
    rep["prefill"] = json::array({ { { "requested_tokens", 1024 }, { "runs", json::array({ run }) } } });
    return rep;
}
}  /* namespace */

TEST(phase_report_keeps_every_requests_tokens) {
    const json r = report_with({ { 1, 2, 3 }, { 4 } });
    const json& o = r["categories"][0]["runs"][0]["requests"]["outputs"];
    REQUIRE(o.size() == 2);
    CHECK(o[0] == json::array({ 1, 2, 3 }));
    CHECK(o[1] == json::array({ 4 }));
}

TEST(identical_outputs_are_parity) {
    json out;
    std::string why;
    REQUIRE(compare_outputs(report_with({ { 1, 2, 3 }, { 4, 5 } }), report_with({ { 1, 2, 3 }, { 4, 5 } }), &out, &why) == RAD_OK);
    CHECK_EQ(out["compared"].get<int64_t>(), (int64_t)6);   /* 2 requests in each of 3 points */
    CHECK_EQ(out["identical"].get<int64_t>(), (int64_t)6);
    CHECK(out["differing"].empty());
    CHECK_EQ(out["verdict"].get<std::string>(), std::string("identical"));
}

TEST(a_difference_names_the_first_token_that_differs) {
    json out;
    std::string why;
    REQUIRE(compare_outputs(report_with({ { 1, 2, 3 }, { 4, 5, 6, 7 } }), report_with({ { 1, 2, 3 }, { 4, 5, 9 } }), &out, &why) == RAD_OK);
    CHECK_EQ(out["identical"].get<int64_t>(), (int64_t)3);
    REQUIRE(out["differing"].size() == 3);
    const json& d = out["differing"][0];
    CHECK_EQ(d["point"].get<std::string>(), std::string("categories code"));
    CHECK_EQ(d["request"].get<int64_t>(), (int64_t)2);
    CHECK_EQ(d["first_different_token"].get<int64_t>(), (int64_t)2);
    CHECK_EQ(d["tokens"].get<int64_t>(), (int64_t)4);
    CHECK_EQ(d["tokens_baseline"].get<int64_t>(), (int64_t)3);
    CHECK_EQ(out["verdict"].get<std::string>(), std::string("DIFFERS"));
    /* A request that stopped early is a difference at the shorter one's end. */
    REQUIRE(compare_outputs(report_with({ { 1, 2 } }), report_with({ { 1, 2, 3 } }), &out, &why) == RAD_OK);
    CHECK_EQ(out["differing"][0]["first_different_token"].get<int64_t>(), (int64_t)2);
}

TEST(points_only_one_side_ran_are_missing_not_counted) {
    json a = report_with({ { 1 } });
    json b = report_with({ { 1 } });
    b["prefill"][0]["requested_tokens"] = 65536;   /* the baseline ran another length */
    json out;
    std::string why;
    REQUIRE(compare_outputs(a, b, &out, &why) == RAD_OK);
    CHECK_EQ(out["compared"].get<int64_t>(), (int64_t)2);
    CHECK_EQ(out["identical"].get<int64_t>(), (int64_t)2);
    CHECK_EQ(out["missing"].size(), (size_t)2);
    CHECK_EQ(out["verdict"].get<std::string>(), std::string("identical where both ran; points missing"));
}

TEST(different_workloads_are_not_compared) {
    json out;
    std::string why;
    CHECK_EQ(compare_outputs(report_with({ { 1 } }, "abc"), report_with({ { 1 } }, "def"), &out, &why), RAD_E_INVAL);
    CHECK(why.find("different workloads") != std::string::npos);
    CHECK_EQ(compare_outputs(report_with({ { 1 } }, ""), report_with({ { 1 } }, ""), &out, &why), RAD_E_INVAL);
}

TEST(fingerprint_is_stable_and_sensitive) {
    const json a = { { "tokens", 256 }, { "runs", 3 } };
    const json b = { { "tokens", 257 }, { "runs", 3 } };
    CHECK_EQ(workload_fingerprint(a), workload_fingerprint(json::parse(a.dump())));
    CHECK(workload_fingerprint(a) != workload_fingerprint(b));
    CHECK_EQ(workload_fingerprint(a).size(), (size_t)16);
}

namespace {
/* A report whose one category ran two runs, each proposing `n` drafts and taking `k`, with
 * position 1 reached `n` times and taken `k1` times. Outputs identical, so only acceptance moves. */
json acceptance_report(int64_t n, int64_t k, int64_t k1) {
    std::vector<json> runs;
    for (int r = 0; r < 2; ++r) {
        PhaseInput in;
        in.after.draft_tokens = n;
        in.after.draft_accepted = k;
        in.after.draft_depth_seen = 1;
        in.after.draft_pos_reached[0] = n;
        in.after.draft_pos_accepted[0] = k1;
        RequestTiming t;
        t.output = { 7, 8 };
        in.requests.push_back(t);
        runs.push_back(phase_report(in));
    }
    json rep;
    rep["workload"] = { { "fingerprint", "f" } };
    rep["categories"] = json::array({ { { "category", "code" }, { "runs", runs },
                                        { "summary", summarize_runs(runs) },
                                        { "draft_by_position", pool_draft_positions(runs) } } });
    return rep;
}
}  /* namespace */

TEST(draft_positions_pool_counts_not_rates) {
    std::vector<json> runs(2);
    runs[0]["counters"]["draft_by_position"] = json::array({ { { "position", 1 }, { "reached", 10 }, { "accepted_n", 9 } },
                                                              { { "position", 2 }, { "reached", 9 }, { "accepted_n", 3 } } });
    runs[1]["counters"]["draft_by_position"] = json::array({ { { "position", 1 }, { "reached", 90 }, { "accepted_n", 45 } } });
    const json p = pool_draft_positions(runs);
    REQUIRE(p.size() == 2);
    CHECK_EQ(p[0]["reached"].get<int64_t>(), (int64_t)100);
    CHECK(near(p[0]["accepted"].get<double>(), 0.54));   /* 54 of 100, not the mean of 0.9 and 0.5 */
    CHECK(near(p[1]["accepted"].get<double>(), 1.0 / 3.0));
    CHECK(pool_draft_positions({}).empty());
}

TEST(an_acceptance_drop_fails_the_comparison) {
    json out;
    std::string why;
    /* 2 x 1000 drafts, 70% taken in the baseline. */
    REQUIRE(compare_outputs(acceptance_report(1000, 700, 800), acceptance_report(1000, 700, 800), &out, &why) == RAD_OK);
    CHECK_EQ(out["verdict"].get<std::string>(), std::string("identical"));
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("no drop"));
    CHECK(near(out["acceptance"][0]["z"].get<double>(), 0.0));

    /* 70% -> 60% over 2000 drafts each: z is about -4.7. */
    REQUIRE(compare_outputs(acceptance_report(1000, 600, 800), acceptance_report(1000, 700, 800), &out, &why) == RAD_OK);
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("DROPPED"));
    REQUIRE(out["acceptance_dropped"].size() == 1);
    CHECK_EQ(out["acceptance_dropped"][0]["what"].get<std::string>(), std::string("overall"));
    CHECK(out["acceptance_dropped"][0]["z"].get<double>() < -4.0);

    /* A collapse at one position alone is caught there. */
    REQUIRE(compare_outputs(acceptance_report(1000, 700, 500), acceptance_report(1000, 700, 800), &out, &why) == RAD_OK);
    REQUIRE(out["acceptance_dropped"].size() == 1);
    CHECK_EQ(out["acceptance_dropped"][0]["what"].get<std::string>(), std::string("position 1"));

    /* 70% -> 69%: within noise at this many drafts (z about -0.5), not a drop. */
    REQUIRE(compare_outputs(acceptance_report(1000, 690, 800), acceptance_report(1000, 700, 800), &out, &why) == RAD_OK);
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("no drop"));

    /* A rise is never a drop. */
    REQUIRE(compare_outputs(acceptance_report(1000, 800, 900), acceptance_report(1000, 700, 800), &out, &why) == RAD_OK);
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("no drop"));

    const std::string md = render_markdown([&] {
        json r = acceptance_report(1000, 600, 800);
        compare_outputs(r, acceptance_report(1000, 700, 800), &out, &why);
        r["parity"] = out;
        return r;
    }());
    CHECK(md.find("**DROPPED**") != std::string::npos);
}

namespace {
/* A report whose one concurrency point ran `rates.size()` runs of one request each, request i
 * proposing `n` drafts and taking `rates[i]` of them, position 1 counted the same way. */
json request_report(const std::vector<double>& rates, int64_t n) {
    std::vector<json> runs;
    for (double rate : rates) {
        PhaseInput in;
        const int64_t k = (int64_t)std::llround(rate * (double)n);
        in.after.draft_tokens = n;
        in.after.draft_accepted = k;
        in.after.draft_depth_seen = 1;
        in.after.draft_pos_reached[0] = n;
        in.after.draft_pos_accepted[0] = k;
        RequestTiming t;
        t.output = { 1 };
        t.drafted = n;
        t.accepted = k;
        in.requests.push_back(t);
        runs.push_back(phase_report(in));
    }
    json rep;
    rep["workload"] = { { "fingerprint", "f" } };
    rep["concurrency"] = json::array({ { { "streams", 8 }, { "runs", runs },
                                         { "draft_by_position", pool_draft_positions(runs) } } });
    return rep;
}
}  /* namespace */

TEST(phase_report_keeps_every_requests_drafts) {
    PhaseInput in;
    RequestTiming t;
    t.drafted = 30;
    t.accepted = 21;
    in.requests.push_back(t);
    const json d = phase_report(in)["requests"]["drafts"];
    REQUIRE(d.size() == 1);
    CHECK(d[0] == json::array({ 30, 21 }));
}

TEST(texts_that_diverged_are_not_a_drop) {
    /* Two runs of one build whose texts went different ways: every request's rate moves, by as
     * much up as down, over 2000 drafts a request. Pooled as coin flips, 0.79 against 0.80 over
     * 48000 drafts is z of about -3.8 -- a drop, which is the false alarm two runs of one build
     * raised on epyc; tested on the requests, t is about -0.5. */
    std::vector<double> base, now;
    for (int i = 0; i < 24; ++i) {
        base.push_back(0.80);
        now.push_back(0.80 + ((i % 2) ? 0.08 : -0.10));
    }
    json out;
    std::string why;
    REQUIRE(compare_outputs(request_report(now, 2000), request_report(base, 2000), &out, &why) == RAD_OK);
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("no drop"));
    const json& overall = out["acceptance"][0];
    CHECK_EQ(overall["test"].get<std::string>(), std::string("paired t over requests"));
    CHECK_EQ(overall["requests"].get<int64_t>(), (int64_t)24);
    /* The positions take the requests' dispersion: far above 1 here, so their z shrinks with it. */
    const json& pos = out["acceptance"][1];
    CHECK(pos["dispersion"].get<double>() > 10.0);
    CHECK(pos["z"].get<double>() > -3.0);
}

TEST(a_drop_in_every_request_is_a_drop) {
    /* Four requests, each 0.10 lower, give or take 0.02: t of about -12 on 3 degrees of freedom,
     * p between 0.0005 and 0.001. */
    json out;
    std::string why;
    REQUIRE(compare_outputs(request_report({ 0.68, 0.72, 0.70, 0.70 }, 1000), request_report({ 0.80, 0.80, 0.80, 0.80 }, 1000),
                            &out, &why) == RAD_OK);
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("DROPPED"));
    const json& overall = out["acceptance"][0];
    CHECK(near(overall["t"].get<double>(), -12.247, 0.01));
    CHECK(overall["p"].get<double>() < 0.001);
    CHECK(overall["p"].get<double>() > 0.0005);
}

TEST(identical_requests_test_to_zero) {
    json out;
    std::string why;
    REQUIRE(compare_outputs(request_report({ 0.7, 0.8, 0.9 }, 500), request_report({ 0.7, 0.8, 0.9 }, 500), &out, &why) == RAD_OK);
    CHECK_EQ(out["acceptance_verdict"].get<std::string>(), std::string("no drop"));
    CHECK(near(out["acceptance"][0]["t"].get<double>(), 0.0));
    CHECK(near(out["acceptance"][0]["p"].get<double>(), 0.5));
    CHECK(near(out["acceptance"][1]["dispersion"].get<double>(), 1.0));
}

TEST(parity_renders_with_its_differences) {
    json rep = report_with({ { 1, 2, 3 } });
    json cmp;
    std::string why;
    REQUIRE(compare_outputs(rep, report_with({ { 1, 9, 3 } }), &cmp, &why) == RAD_OK);
    cmp["baseline"] = "base.json";
    rep["parity"] = cmp;
    const std::string md = render_markdown(rep);
    CHECK(md.find("## Parity against base.json") != std::string::npos);
    CHECK(md.find("**DIFFERS**: 0 of 3 requests") != std::string::npos);
    CHECK(md.find("| categories code | 1 | 1 | 0 | 1 | 3 / 3 |") != std::string::npos);
}

/* ------------------------------------------------------------------ report */

TEST(markdown_renders_the_sweeps_from_json_alone) {
    const int64_t ms = 1'000'000;
    PhaseInput in;
    in.kind = PhaseKind::Decode;
    for (int i = 0; i < 4; ++i) in.steps.push_back(step(i, RAD_PHASE_DECODE, 1, 0, 1, 12.5, (i + 1) * 13 * ms));
    const json run = phase_report(in);
    PhaseInput in2 = in;
    for (int i = 0; i < 4; ++i) in2.steps[(size_t)i].end_ns = (i + 1) * 20 * ms;   /* a second run, slower */
    const json run2 = phase_report(in2);

    json rep;
    rep["engine"] = { { "container", "m.rad" }, { "arch", "test" }, { "tp", 2 }, { "backend", "host" },
                      { "plan", "dynamic placement" } };
    rep["workload"] = { { "description", "fixed" } };
    rep["verdict"] = "valid";
    rep["warnings"] = json::array({ "a warning" });
    rep["categories"] = json::array({ { { "category", "code" }, { "requests", 1 }, { "stopped_on_eos", 1 },
                                        { "summary", summarize_runs({ run }) } } });
    rep["concurrency"] = json::array({ { { "streams", 8 }, { "summary", summarize_runs({ run, run2 }) } } });
    rep["prefill"] = json::array({ { { "prompt_tokens", 4096 }, { "summary", summarize_runs({ run }) } } });
    rep["interference"] = { { "streams", 4 }, { "prompt_tokens", 4096 }, { "idle_ttft_ms", 300.0 },
                            { "summary", summarize_paths({ json{ { "arrival_ttft_ms", 450.0 } } },
                                                         { "/arrival_ttft_ms", "/interference/stall_ratio" }) } };
    rep["not_measured"] = json::array({ "per-kernel time" });

    const std::string md = render_markdown(rep);
    CHECK(md.find("# Diagnosis") == 0);
    CHECK(md.find("`m.rad`") != std::string::npos);
    CHECK(md.find("TP 2") != std::string::npos);
    CHECK(md.find("**Verdict: valid**") != std::string::npos);
    CHECK(md.find("- a warning") != std::string::npos);
    CHECK(md.find("| code | 1 |") != std::string::npos);
    CHECK(md.find("| 1 of 1 |") != std::string::npos);
    CHECK(md.find("| 8 | 2 |") != std::string::npos);
    CHECK(md.find("| 13 (13–20) |") != std::string::npos);   /* emission p50: 13 ms and 20 ms runs */
    CHECK(md.find("| 4096 | 1 |") != std::string::npos);
    CHECK(md.find("| TTFT of the arriving prompt | 450 ms |") != std::string::npos);
    CHECK(md.find("| Stall ratio (longest / typical) | – |") != std::string::npos);
    CHECK(md.find("## Not measured") != std::string::npos);
    CHECK(md.find("## Parity") == std::string::npos);   /* no baseline given */

    /* And the same text from a report that went through a file. */
    CHECK(render_markdown(json::parse(rep.dump())) == md);
}

/* The report's environment section: the knobs by name, from an allowlist, and never a value --
 * the report is pasted into issues and committed beside a baseline. */
TEST(environment_is_a_name_allowlist) {
    char tp[]      = "RADIANCE_TP=2";
    char strict[]  = "RAD_TEST_STRICT=1";
    char omp[]     = "OMP_NUM_THREADS=4";
    char other[]   = "OMP_WAIT_POLICY=passive";
    char key[]     = "MY_API_KEY=sk-very-secret-value";
    char home[]    = "HOME=/root";
    char radfake[] = "RADIATOR=not a prefix of RAD_ with a name";
    char* env[] = { tp, strict, omp, other, key, home, radfake, nullptr };

    const std::vector<std::string> v = diag::env_allowlist(env);
    CHECK_EQ(v.size(), (size_t)4);
    CHECK(std::find(v.begin(), v.end(), "RADIANCE_TP") != v.end());
    CHECK(std::find(v.begin(), v.end(), "RAD_TEST_STRICT") != v.end());
    CHECK(std::find(v.begin(), v.end(), "OMP_NUM_THREADS") != v.end());
    CHECK(std::find(v.begin(), v.end(), "OMP_WAIT_POLICY") != v.end());
    for (const std::string& n : v) {
        CHECK(n.find('=') == std::string::npos);            /* names, never values */
        CHECK(n.find("sk-") == std::string::npos);
        CHECK(n.find("API_KEY") == std::string::npos);
        CHECK(n.rfind("RADIANCE_", 0) == 0 || n.rfind("RAD_", 0) == 0 ||
              n.rfind("OMP_", 0) == 0);                      /* nothing outside the allowlist */
    }
    CHECK(std::is_sorted(v.begin(), v.end()));
    /* An empty environment is an empty section, not a fabricated one. */
    char* none[] = { nullptr };
    CHECK(diag::env_allowlist(none).empty());
}

RAD_TEST_MAIN()
