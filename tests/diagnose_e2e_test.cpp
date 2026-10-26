/* diagnose_e2e_test.cpp -- `radiance --diagnose DIR` end to end on the CPU: the real binary, a tiny
 * synthetic checkpoint (tests/tiny_model.h), the host backend, and the two files it writes.
 *
 * The arithmetic is checked against hand-computed numbers in tests/diagnose_test.cpp. What is here
 * is the run: that the mode starts, gets its workload through its own scheduler, and puts a report
 * on disk that says what it measured and what it could not. The cases are the claims the mode makes
 * to whoever reads the report --
 *
 *   Nothing came from the prefix cache. Every point has to show zero cached prompt tokens, or the
 *     run is not a measurement of this workload and has to say so instead of reporting times.
 *   The same build says the same words. Two runs with the same `--diagnose-salt` write the same
 *     tokens for every request, which is what makes `--diagnose-baseline` a comparison rather than
 *     a suggestion.
 *   A baseline is taken seriously: one's own report passes, a report written with another salt is
 *     refused with the salt named, and `--diagnose-baseline` without `--diagnose` is refused before
 *     anything is loaded. The threshold `--diagnose-acceptance-drop` names is the threshold the
 *     comparison used, read back from the note that names it.
 *   Absent is not zero. A point this deployment cannot hold is reported with the reason, and a run
 *     with no drafter reports its acceptance as absent rather than as 0 -- and with
 *     `--num-speculative-tokens`, the positions are there and the counts add up.
 */
#include "rad_test.h"
#include "tiny_model.h"

#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <map>
#include <string>
#include <vector>

using nlohmann::json;

#ifndef RAD_E2E_SERVER
#define RAD_E2E_SERVER ""
#endif
#ifndef RAD_E2E_HOME
#define RAD_E2E_HOME ""
#endif

namespace {

/* The checkpoints, written once per process: the plain one, and the weaker-residual one whose text
 * repeats n-grams, so prompt lookup has something to draft on (tiny_model.h, Gains). */
struct World {
    tiny::TempDir dir{ "diag" };
    std::string qwen35, qwen35_spec, out;
    bool ok = false;

    World() {
        if (!dir.ok()) return;
        qwen35 = dir / "qwen35";
        qwen35_spec = dir / "qwen35_spec";
        out = dir / "out";
        tiny::Qwen35 spec;
        spec.branch = 0.33f;
        if (mkdir(qwen35.c_str(), 0755) != 0 || mkdir(qwen35_spec.c_str(), 0755) != 0) return;
        ok = tiny::write_qwen35(qwen35, {}) && tiny::write_qwen35(qwen35_spec, spec);
    }
};
World& world() {
    static World w;
    return w;
}

struct Ran {
    int status = -1;
    std::string log;
};

std::string tail(const std::string& log, int n = 10) {
    size_t at = log.size();
    for (int i = 0; i <= n && at != std::string::npos && at > 0; ++i) at = log.rfind('\n', at - 1);
    return at == std::string::npos || at >= log.size() ? log : log.substr(at + 1);
}

/* One diagnose run of the real binary. libref is the only kernel library, so every op resolves to
 * the reference implementation and needs the debug flag; RADIANCE_HOST_VRAM_MIB sizes the device
 * the way the other CPU end-to-end tests do, which is also what makes a point the deployment
 * cannot hold happen. */
Ran diagnose(const std::string& model, const std::string& out,
             const std::vector<std::string>& more = {}, int timeout_s = 280) {
    std::vector<std::string> a = { RAD_E2E_SERVER,      "--model", model,
                                   "--kernels", "libref", "--debug-accept-reference-kernels",
                                   "--diagnose", out };
    a.insert(a.end(), more.begin(), more.end());
    const std::vector<std::string> e = { "ROCR_VISIBLE_DEVICES=99", "RADIANCE_HOST_VRAM_MIB=512",
                                         "OMP_WAIT_POLICY=passive", "OMP_NUM_THREADS=2",
                                         std::string("RADIANCE_HOME=") + RAD_E2E_HOME };
    Ran r;
    r.status = tiny::run(a, e, &r.log, timeout_s);
    return r;
}

std::string slurp(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

json report_of(const std::string& dir) {
    const std::string text = slurp(dir + "/diagnose.json");
    if (text.empty()) return json();
    return json::parse(text, nullptr, false);
}

/* Every request's tokens, keyed by point and ordinal: what `--diagnose-baseline` compares. */
std::map<std::string, std::vector<int64_t>> tokens_of(const json& j) {
    std::map<std::string, std::vector<int64_t>> out;
    for (const json& p : j.value("points", json::array()))
        for (const json& q : p.value("requests", json::array()))
            out[p.value("name", std::string()) + " #" + std::to_string(q.value("n", (int64_t)0))] =
                q.value("tokens", std::vector<int64_t>());
    return out;
}

std::string first_difference(const std::map<std::string, std::vector<int64_t>>& a,
                             const std::map<std::string, std::vector<int64_t>>& b) {
    if (a.size() != b.size())
        return std::to_string(a.size()) + " request(s) against " + std::to_string(b.size());
    for (const auto& e : a) {
        auto it = b.find(e.first);
        if (it == b.end()) return "the second run has no " + e.first;
        if (it->second != e.second) return e.first;
    }
    return "";
}

bool has_fail(const json& j) {
    for (const json& f : j.value("findings", json::array()))
        if (f.value("level", std::string()) == "fail") return true;
    return false;
}

}  /* namespace */

TEST(diagnose_writes_both_files_and_measures_every_point_it_can_hold) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::string out = world().out + "/once";
    const Ran r = diagnose(world().qwen35, out, { "--diagnose-salt", "11" });
    if (r.status != 0) fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
    REQUIRE_EQ(r.status, 0);
    const json j = report_of(out);
    REQUIRE(!j.is_discarded());
    CHECK(!slurp(out + "/diagnose.md").empty());

    /* The run's identity, so a baseline can be told apart from a different run's report. */
    CHECK_EQ(j.value("schema", 0), 3);
    CHECK(j["workload"].dump() != "0");
    CHECK_EQ(j["salt"].get<int64_t>(), 11);
    CHECK_EQ(j["engine"].value("backend", std::string()), std::string("host"));

    const auto points = j.value("points", json::array());
    CHECK(points.size() >= 12u);
    int64_t requests = 0, tokens = 0, measured = 0;
    for (const json& p : points) {
        /* Nothing from the cache. This is the claim the mode makes about its own numbers: a prompt
         * the cache served was never prefilled, so the step times beside it are not this
         * workload's. */
        const int64_t cached = p.contains("tokens") ? p["tokens"].value("cached", (int64_t)0) : 0;
        if (cached > 0)
            CHECK(p.value("measured", true) == false &&
                  has(p.value("not_measured", std::string()), "prefix cache"));
        /* A point that did not measure says why, in words, rather than reporting zeros. */
        if (!p.value("measured", true)) {
            CHECK(!p.value("not_measured", std::string()).empty());
            continue;
        }
        ++measured;
        for (const json& q : p.value("requests", json::array())) {
            ++requests;
            tokens += (int64_t)q.value("tokens", json::array()).size();
            /* Every request ran to its length: the point states max_tokens, and `ignore_eos` means
             * the model gets no vote. */
            CHECK(!q.value("tokens", json::array()).empty());
            CHECK_EQ(q.value("finish", std::string()), std::string("length"));
        }
        /* A step cannot put more tokens on the wire than there are sequences live in it: this run
         * has no drafter, so one token per sequence at most. A per-step record that counted its
         * predecessors' emissions is what this catches. */
        if (p["tokens_per_step"]["emitted"].contains("max"))
            CHECK(p["tokens_per_step"]["emitted"]["max"].get<double>() <=
                  (double)p.value("max_live", (int64_t)0));
        /* The point's own token count is the sum of its requests', which are the tokens a client
         * would have been sent. */
        int64_t sum = 0;
        for (const json& q : p.value("requests", json::array())) sum += (int64_t)q.value("tokens", json::array()).size();
        CHECK_EQ(sum, p["tokens"].value("out", (int64_t)-1));
        /* The counters and the records are two readings of the same steps, charged at the same
         * quiet boundary: the phase families sum to the point's step count, and so do the shapes
         * the distributions are made of. A count read at the sink's fill, one step ahead of the
         * drain below it, flaps run to run instead. */
        if (p.value("step_records_dropped", (int64_t)0) == 0) {
            CHECK_EQ(p["steps_by_phase"].value("prefill", (int64_t)0) +
                         p["steps_by_phase"].value("mixed", (int64_t)0) +
                         p["steps_by_phase"].value("decode", (int64_t)0),
                     p.value("steps", (int64_t)-1));
            CHECK_EQ(p["steps_by_shape"].value("prefill_chunk", (int64_t)0) +
                         p["steps_by_shape"].value("mixed", (int64_t)0) +
                         p["steps_by_shape"].value("decode", (int64_t)0) +
                         p["steps_by_shape"].value("draft_pass", (int64_t)0),
                     p.value("steps", (int64_t)-1));
        }

        /* The distributions it exists to produce. */
        CHECK(p["steps_by_shape"].value("decode", (int64_t)0) >= 0);
        /* The distribution and the count are made of the same steps: a point that measured decode
         * steps has a decode distribution, and one that did not says why. */
        if (p["steps_by_shape"].value("decode", (int64_t)0) > 0) {
            CHECK(p["step_us"]["decode"].contains("p50"));
            CHECK(p["step_us"]["decode"].contains("p99"));
            /* p50 <= p90 <= p99 <= max, which review() also checks; a series that fails it here is
             * a series the report should have refused to write. */
            const json& d = p["step_us"]["decode"];
            CHECK(d["p50"].get<double>() <= d["p90"].get<double>() + 1e-9);
            CHECK(d["p90"].get<double>() <= d["p99"].get<double>() + 1e-9);
            CHECK(d["p99"].get<double>() <= d["max"].get<double>() + 1e-9);
        }
        /* The interval between emissions needs two of them: a point whose tokens came from one step
         * has no interval, and says so rather than reporting one. */
        if (p["tokens_per_step"]["emitted"].value("count", (int64_t)0) >= 2) {
            CHECK(p["token_interval_us"].contains("p50"));
            CHECK(p["token_gap_us"].contains("p50"));
            /* The wait is the part of the interval the engine did not spend on a step, so it cannot
             * exceed the interval it is inside. */
            CHECK(p["token_gap_us"]["p50"].get<double>() <=
                  p["token_interval_us"]["p50"].get<double>() + 1e-6);
        } else if (p["tokens_per_step"]["emitted"].value("count", (int64_t)0) == 1) {
            CHECK(p["token_interval_us"].contains("absent"));
            CHECK(p["token_gap_us"].contains("absent"));
        }
    }
    CHECK(measured >= 8);
    CHECK(requests >= 20);
    CHECK(tokens > 0);
    CHECK(!has_fail(j));
    /* The markdown holds the same numbers, in tables, and says what the run is. */
    const std::string md = slurp(out + "/diagnose.md");
    CHECK(has(md, "# radiance --diagnose"));
    CHECK(has(md, "## Draft acceptance"));
    CHECK(has(md, "concurrency"));
    CHECK(has(md, "11"));                        /* the salt, stated in the report */
}

TEST(diagnose_twice_with_the_same_salt_writes_the_same_tokens) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::string a = world().out + "/same-a", b = world().out + "/same-b";
    REQUIRE_EQ(diagnose(world().qwen35, a, { "--diagnose-salt", "11" }).status, 0);
    REQUIRE_EQ(diagnose(world().qwen35, b, { "--diagnose-salt", "11" }).status, 0);
    const json ja = report_of(a), jb = report_of(b);
    REQUIRE(!ja.is_discarded() && !jb.is_discarded());
    /* The workload is fixed and the sampler is greedy, so the only thing that may differ between
     * two runs of one build is the machine's timing. Token parity is what makes every throughput
     * number in the report comparable across builds: a build that says different words is a
     * different build whatever its step times did. */
    const std::string diff = first_difference(tokens_of(ja), tokens_of(jb));
    CHECK(diff.empty());
    /* The points and their prompts are the same too -- a different request count would make the
     * parity check above compare different things. */
    CHECK_EQ(tokens_of(ja).size(), tokens_of(jb).size());
}

TEST(diagnose_baseline_accepts_its_own_report_and_names_what_it_checked) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::string a = world().out + "/base-a", b = world().out + "/base-b";
    REQUIRE_EQ(diagnose(world().qwen35, a, { "--diagnose-salt", "11" }).status, 0);
    const Ran r = diagnose(world().qwen35, b,
                           { "--diagnose-salt", "11", "--diagnose-baseline", a + "/diagnose.json" });
    if (r.status != 0) fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
    REQUIRE_EQ(r.status, 0);
    CHECK(has(r.log, "token-parity"));
    const json j = report_of(b);
    REQUIRE(!j.is_discarded());
    bool parity = false;
    for (const json& f : j.value("findings", json::array()))
        parity = parity || (f.value("id", std::string()) == "token-parity" &&
                            f.value("level", std::string()) == "note");
    CHECK(parity);
    CHECK(!has_fail(j));
}

TEST(diagnose_baseline_refuses_another_salt_by_naming_it) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::string a = world().out + "/salt-a", b = world().out + "/salt-b";
    REQUIRE_EQ(diagnose(world().qwen35, a, { "--diagnose-salt", "11" }).status, 0);
    const Ran r = diagnose(world().qwen35, b,
                           { "--diagnose-salt", "12", "--diagnose-baseline", a + "/diagnose.json" });
    /* Different salt means different prompts, so its tokens are not what this run would have
     * written: the comparison is refused with both numbers in the message rather than reported as
     * one token-parity failure per request. */
    CHECK(r.status != 0);
    CHECK(has(r.log, "salted its prompts with 11"));
    CHECK(has(r.log, "12"));
}

TEST(diagnose_the_acceptance_drop_flag_reaches_the_comparison) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::string a = world().out + "/drop-a", b = world().out + "/drop-b";
    /* Two runs of the spec model against each other: identical acceptance, so nothing fell, and
     * the note the comparison writes names the threshold it compared against. That note is the
     * only place the flag's value is observable end to end -- compare_acceptance's arithmetic is
     * held to hand-computed numbers in diagnose_test -- so a run that ignored the flag and kept
     * the default would be caught here saying 10% where 50% was asked for. */
    REQUIRE_EQ(diagnose(world().qwen35_spec, a,
                        { "--diagnose-salt", "3", "--num-speculative-tokens", "3" }).status,
               0);
    const Ran r = diagnose(world().qwen35_spec, b,
                           { "--diagnose-salt", "3", "--num-speculative-tokens", "3",
                             "--diagnose-baseline", a + "/diagnose.json",
                             "--diagnose-acceptance-drop", "0.50" });
    if (r.status != 0) fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
    REQUIRE_EQ(r.status, 0);
    CHECK(has(r.log, "no point's pooled acceptance fell by 50% or more"));
    /* The startup report says the flag was heard; the note above says it was obeyed. */
    CHECK(has(r.log, "0.50"));
    CHECK(!has(r.log, "fell by 10% or more"));
}

TEST(diagnose_a_baseline_without_a_run_is_refused_before_the_model_is_loaded) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::vector<std::string> a = { RAD_E2E_SERVER, "--model", world().qwen35,
                                         "--diagnose-baseline", world().out + "/none.json" };
    const std::vector<std::string> e = { "ROCR_VISIBLE_DEVICES=99",
                                         std::string("RADIANCE_HOME=") + RAD_E2E_HOME };
    std::string log;
    const int s = tiny::run(a, e, &log, 60);
    CHECK(s != 0);
    CHECK(has(log, "--diagnose is what makes the run"));
}

TEST(diagnose_with_speculation_reports_acceptance_by_position) {
    if (!world().ok || !RAD_E2E_SERVER[0]) SKIP("no checkpoint or no radiance binary");
    const std::string out = world().out + "/spec";
    const Ran r = diagnose(world().qwen35_spec, out,
                           { "--diagnose-salt", "3", "--num-speculative-tokens", "3" });
    if (r.status != 0) fprintf(stderr, "    child exited %d:\n%s\n", r.status, tail(r.log).c_str());
    REQUIRE_EQ(r.status, 0);
    const json j = report_of(out);
    REQUIRE(!j.is_discarded());
    /* The engine's own facts: what it speculated with, recorded beside the numbers it changes. */
    CHECK(!j["engine"].value("drafter", std::string()).empty());
    CHECK_EQ(j["engine"].value("speculative window", std::string()), std::string("3"));

    int64_t drafted = 0;
    bool positions = false;
    for (const json& p : j.value("points", json::array())) {
        if (!p.contains("acceptance")) continue;
        const json& a = p["acceptance"];
        const int64_t d = a.value("drafted", (int64_t)0);
        drafted += d;
        if (!d) {
            /* Nothing drafted here: absent with the reason, never 0% acceptance read as a broken
             * drafter. */
            CHECK(a["pooled"].contains("absent"));
            continue;
        }
        const double pooled = a["pooled"].value("value", -1.0);
        CHECK(a["pooled"].contains("value"));     /* drafted > 0, so the ratio is measurable */
        CHECK_NEAR(pooled, (double)a.value("accepted", (int64_t)0) / (double)d, 1e-9);
        for (const json& q : a.value("positions", json::array())) {
            positions = true;
            const int64_t reached = q.value("reached", (int64_t)0), hits = q.value("hits", (int64_t)0);
            CHECK(hits <= reached);
            if (reached > 0)
                CHECK_NEAR(q["conditional"]["value"].get<double>(),
                           (double)hits / (double)reached, 1e-9);
            else
                CHECK(q["conditional"].contains("absent"));
        }
        /* Per request as well: a point's acceptance is the sum of what its requests saw. */
        int64_t req_drafted = 0;
        for (const json& q : p.value("requests", json::array())) req_drafted += q.value("drafted", (int64_t)0);
        CHECK(req_drafted <= d);
    }
    /* On a checkpoint whose text repeats, prompt lookup has something to draft, so this run is a
     * measurement of acceptance and not an absence of one. */
    CHECK(drafted > 0);
    CHECK(positions);
    /* And by category, pooled over the points that share one. */
    const auto cats = j.value("by_category", json::array());
    CHECK(cats.size() >= 2u);
    int64_t cat_drafted = 0;
    for (const json& c : cats) cat_drafted += c["acceptance"].value("drafted", (int64_t)0);
    CHECK_EQ(cat_drafted, drafted);
}

RAD_TEST_MAIN()
