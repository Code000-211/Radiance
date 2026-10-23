/* diagnose.cpp -- the workload, the arithmetic and the two renderings behind --diagnose.
 *
 * No engine, no device, no scheduler: this file turns records and counters into numbers, and
 * numbers into text. Engine::run_diagnose (core/engine_diagnose.cpp) is the other half, and the
 * split is what makes the arithmetic testable to hand-computed values (tests/diagnose_test.cpp)
 * without a model.
 *
 * The workload is a table, not a formula: what a run measures has to be the same thing every time,
 * or a comparison across two builds says something about the workload instead. workload_id() is
 * its hash, and a baseline whose hash differs is refused.
 */
#include "diagnose.h"

#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace rad {
namespace diagnose {

using nlohmann::json;

/* ------------------------------------------------------------------ the workload
 * Prompts: fixed text, and the length it is made to. The long ones are grown with one token id
 * counted out, so a prefill point is a length rather than a coincidence of where the text ended;
 * what is measured at 2048 tokens is measured at 2048 tokens. */
const std::vector<Prompt>& prompts() {
    static const std::vector<Prompt> kPrompts = {
        { "prose", "prose",
          "The harbour was quiet before dawn, only the rigging ticking and the water moving under "
          "the hull. He counted the boats along the far wall while the sky turned, and by the time "
          "the first gull crossed the mouth of the bay the light was grey enough to read by. The "
          "log had yesterday's figures in it and nothing else worth writing down.", 0 },
        { "code", "code",
          "static int reserve_rows(Pool* p, int64_t rows) {\n"
          "    if (rows <= 0 || rows > p->cap) return RAD_E_INVAL;\n"
          "    int64_t need = rows * p->stride;\n"
          "    if (need > p->free) {\n"
          "        const int s = pool_grow(p, need - p->free);\n"
          "        if (s < 0) return s;\n"
          "    }\n"
          "    p->used += need;\n"
          "    return (int)(need / p->stride);\n"
          "}\n", 0 },
        { "structured", "structured",
          "{\"model\":\"radiance\",\"points\":[{\"name\":\"decode-floor\",\"concurrency\":1,"
          "\"max_tokens\":96},{\"name\":\"concurrency:4\",\"concurrency\":4,\"max_tokens\":96},"
          "{\"name\":\"prefill:512\",\"concurrency\":1,\"max_tokens\":1}],\"cache\":false,"
          "\"salt\":17,\"schema\":3}\n", 0 },
        { "recall", "recall",
          "The instruction, in full: repeat the sentence \"the north beam holds at forty degrees\" "
          "once for every paragraph that follows, changing only the number. Paragraph one: ", 0 },
        { "long-context", "long-context",
          "A long document, for a prefill that has to be read before anything is answered. ", 1024 },
        { "prefill-128", "prefill", "", 128 },
        { "prefill-512", "prefill", "", 512 },
        { "prefill-2048", "prefill", "", 2048 },
    };
    return kPrompts;
}

const std::vector<Point>& points() {
    using P = Point;
    static const std::vector<Point> kPoints = {
        /* One sequence alone, the shape every other number is compared with. */
        { "decode-floor", "decode", 1, 1, 0, 1, 96, 0, 0, 0 },
        /* Categories: one sequence each, so what moves is the text, not the batch. The drafter has
         * something to work with on code and JSON and little on a sentence it has never seen. */
        { "category:prose", "prose", 1, 1, 0, 1, 96, 0, 0, 0 },
        { "category:code", "code", 1, 1, 1, 1, 96, 0, 0, 0 },
        { "category:structured", "structured", 1, 1, 2, 1, 96, 0, 0, 0 },
        { "category:recall", "recall", 1, 1, 3, 1, 96, 0, 0, 0 },
        { "category:long-context", "long-context", 1, 1, 4, 1, 32, 0, 0, 0 },
        /* Concurrency: the same text at 1, 4 and 8 sequences. A step's cost moving with its rows
         * says the reads are shared; a step that costs eight times as much says they are not. */
        { "concurrency:1", "concurrency", 1, 1, 0, 1, 64, 0, 0, 0 },
        { "concurrency:4", "concurrency", 4, 4, 0, 1, 64, 0, 0, 0 },
        { "concurrency:8", "concurrency", 8, 4, 2, 2, 64, 0, 0, 0 },
        /* Prefill by length, generating nothing. */
        { "prefill:128", "prefill", 1, 1, 5, 1, 1, 0, 0, 0 },
        { "prefill:512", "prefill", 1, 1, 6, 1, 1, 0, 0, 0 },
        { "prefill:2048", "prefill", 1, 1, 7, 1, 1, 0, 0, 0 },
        /* A prompt arriving while four sequences decode: the cost of the chunk is paid by the
         * streams that were already running, so it is their burst interval that is measured on
         * each side of the arrival. One more prompt than the point keeps in flight, so the last
         * one is the arrival and the four before it are the streams that were already going. */
        { "interference", "interference", 4, 5, 0, 1, 64, 0, 24, 6 },
        { "interference:long", "interference", 1, 2, 0, 1, 64, 0, 16, 7 },
    };
    return kPoints;
}

/* fnv1a over everything that decides what a run does: the names, and the numbers behind them. */
static uint64_t fnv(const std::string& s, uint64_t h) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

uint64_t workload_id() {
    uint64_t h = 14695981039346656037ull;
    h = fnv("radiance-diagnose-1", h);
    for (const Prompt& p : prompts()) {
        h = fnv(p.name, h); h = fnv(p.category, h); h = fnv(p.text, h);
        h = fnv(std::to_string(p.tokens), h);
    }
    for (const Point& p : points()) {
        h = fnv(p.name, h); h = fnv(p.category, h);
        h = fnv(std::to_string(p.concurrency), h);
        h = fnv(std::to_string(p.n_prompts), h);
        h = fnv(std::to_string(p.first_prompt), h);
        h = fnv(std::to_string(p.repeat), h);
        h = fnv(std::to_string(p.max_tokens), h);
        h = fnv(std::to_string(p.warmup), h);
        h = fnv(std::to_string(p.interfere_after_tokens), h);
        h = fnv(std::to_string(p.interfere_prompt), h);
    }
    return h;
}

/* ------------------------------------------------------------------ measured and absent */
std::string measure_json(const Measure& m) {
    json j = json::object();
    if (m.present) j["value"] = m.value;
    else j["absent"] = m.absent;
    return j.dump();
}

Measure Measure::ratio(int64_t num, int64_t den, const std::string& why_if_none) {
    if (den <= 0) return Measure::missing(why_if_none);
    return Measure::of((double)num / (double)den);
}

/* ------------------------------------------------------------------ distributions
 * The interpolated order statistic: at p*(n-1) of the sorted samples, between the neighbours when
 * that is not a whole number. Every step counts once, so a phase's slow tail reaches p99 instead
 * of being averaged into the mean, which is the whole reason a run reports a distribution.
 *
 * A mean over a decaying window (SchedMetrics::decode_tps) is the right thing for a dashboard
 * watching one deployment and the wrong thing for comparing two builds: it hides how often the
 * step was twice its own median, which is what a scheduling change usually does. */
Measure percentile(std::vector<double> s, double p) {
    if (s.empty()) return Measure::missing("nothing to take a percentile of");
    std::sort(s.begin(), s.end());
    const double pos = (p / 100.0) * (double)(s.size() - 1);
    size_t lo = (size_t)std::floor(pos);
    if (lo >= s.size() - 1) return Measure::of(s.back());
    const double frac = pos - (double)lo;
    const double hi = s[lo + 1];
    return Measure::of(s[lo] + frac * (hi - s[lo]));
}

Measure mean_of(const std::vector<double>& s) {
    if (s.empty()) return Measure::missing("no samples");
    double sum = 0;
    for (double v : s) sum += v;
    return Measure::of(sum / (double)s.size());
}

Dist summarize(std::vector<double>& s, const std::string& nothing) {
    Dist d;
    d.count = (int64_t)s.size();
    if (s.empty()) { d.absent = nothing; return d; }
    std::sort(s.begin(), s.end());
    d.mean = mean_of(s);
    const std::vector<double> sorted = s;   /* percentile() sorts its own copy; the sorted one is
                                               cheap here and keeps the order statistic exact */
    d.p50 = percentile(sorted, 50.0);
    d.p90 = percentile(sorted, 90.0);
    d.p99 = percentile(sorted, 99.0);
    d.max = Measure::of(s.back());
    return d;
}

/* ------------------------------------------------------------------ acceptance */
Measure Acceptance::at(size_t j) const {
    if (j >= reached.size() || reached[j] == 0)
        return Measure::missing(fmt("no draft reached position %zu", j));
    return Measure::ratio((int64_t)hits[j], (int64_t)reached[j], "nothing reached that position");
}

int Acceptance::deepest() const {
    for (size_t j = reached.size(); j-- > 0;)
        if (reached[j] > 0) return (int)j;
    return -1;
}

/* ------------------------------------------------------------------ the JSON */
static json dist_json(const Dist& d) {
    json j = json::object();
    if (d.count == 0) { j["absent"] = d.absent; return j; }
    j["count"] = d.count;
    j["mean"] = d.mean.present ? json(d.mean.value) : json(d.mean.absent);
    j["p50"]  = d.p50.present  ? json(d.p50.value)  : json(d.p50.absent);
    j["p90"]  = d.p90.present  ? json(d.p90.value)  : json(d.p90.absent);
    j["p99"]  = d.p99.present  ? json(d.p99.value)  : json(d.p99.absent);
    j["max"]  = d.max.present  ? json(d.max.value)  : json(d.max.absent);
    return j;
}

static json measure_field(const Measure& m) {
    json j = json::object();
    if (m.present) j["value"] = m.value;
    else j["absent"] = m.absent;
    return j;
}

static json acceptance_json(const Acceptance& a) {    json j = json::object();
    j["drafted"] = a.drafted;
    j["accepted"] = a.accepted;
    j["pooled"] = measure_field(a.pooled());
    json pos = json::array();
    const int deep = a.deepest();
    for (int jn = 0; jn <= deep; ++jn) {
        json p = json::object();
        p["position"] = jn;
        p["reached"] = a.reached[(size_t)jn];
        p["hits"] = a.hits[(size_t)jn];
        p["conditional"] = measure_field(a.at((size_t)jn));
        pos.push_back(std::move(p));
    }
    j["positions"] = std::move(pos);
    return j;
}

/* Acceptance pooled over the points of one category, in the order the categories first appear.
 * The point-level numbers say where acceptance moved; these say whether a whole family of text
 * moved with it, which is the difference between a drafter and a workload. */
static std::vector<std::pair<std::string, Acceptance>> by_category(const Report& r) {
    std::vector<std::pair<std::string, Acceptance>> out;
    std::map<std::string, size_t> where;
    for (const PointReport& p : r.points) {
        auto it = where.find(p.category);
        if (it == where.end()) {
            it = where.emplace(p.category, out.size()).first;
            out.push_back({ p.category, Acceptance{} });
        }
        Acceptance& a = out[it->second].second;
        a.drafted  += p.acceptance.drafted;
        a.accepted += p.acceptance.accepted;
        for (size_t j = 0; j < p.acceptance.reached.size(); ++j) {
            if (a.reached.size() <= j) { a.reached.push_back(0); a.hits.push_back(0); }
            a.reached[j] += p.acceptance.reached[j];
            a.hits[j]    += p.acceptance.hits[j];
        }
    }
    return out;
}

json report_json_object(const Report& r) {
    json j = json::object();
    j["schema"] = kSchemaVersion;
    j["workload"] = r.workload;
    j["salt"] = r.salt;
    json eng = json::object();
    for (const auto& kv : r.engine) eng[kv.first] = kv.second;
    j["engine"] = std::move(eng);

    json absent = json::array();
    for (const auto& kv : r.absent) absent.push_back({ { "quantity", kv.first },
                                                       { "reason", kv.second } });
    j["absent"] = std::move(absent);

    json pts = json::array();
    for (const PointReport& p : r.points) {
        json o = json::object();
        o["name"] = p.name;
        o["category"] = p.category;
        o["concurrency"] = p.concurrency;
        o["max_live"] = p.max_live;
        o["measured"] = p.measured;
        if (!p.not_measured.empty()) o["not_measured"] = p.not_measured;
        o["absent"] = p.absent;
        o["wall_s"] = p.wall_s;
        o["requests"] = p.requests;
        o["steps"] = p.steps;
        o["step_records_dropped"] = p.step_records_dropped;
        o["tokens"] = { { "prompt", p.prompt_tokens }, { "cached", p.cached_tokens },
                        { "out", p.out_tokens }, { "prefill", p.prefill_tokens },
                        { "decode", p.decode_tokens } };
        /* The engine's phasing, then the same steps as the records saw them: a prefill that
         * completes emits a token, so it is a decode step to the counters and a prefill step to
         * the rows it carried. The distributions are made of the second. */
        o["steps_by_phase"] = { { "prefill", p.steps_prefill }, { "mixed", p.steps_mixed },
                                { "decode", p.steps_decode }, { "issued_ahead", p.steps_ahead },
                                { "preemptions", p.preemptions }, { "chunks", p.chunks } };
        o["steps_by_shape"] = { { "prefill_chunk", p.shape_prefill }, { "mixed", p.shape_mixed },
                                { "decode", p.shape_decode }, { "draft_pass", p.shape_draft } };
        o["step_us"] = { { "decode", dist_json(p.step_us_decode) },
                         { "mixed", dist_json(p.step_us_mixed) },
                         { "prefill", dist_json(p.step_us_prefill) },
                         { "all", dist_json(p.step_us_all) } };
        o["token_interval_us"] = dist_json(p.token_interval_us);
        o["token_gap_us"] = dist_json(p.token_gap_us);
        o["tokens_per_step"] = { { "emitted", dist_json(p.tokens_per_step) },
                                 { "prefill_steps", dist_json(p.tok_per_step_prefill) },
                                 { "mixed_steps", dist_json(p.tok_per_step_mixed) },
                                 { "decode_steps", dist_json(p.tok_per_step_decode) },
                                 { "rows", dist_json(p.rows_per_step) } };
        o["interval_before_arrival_us"] = dist_json(p.interval_before_arrival_us);
        o["interval_after_arrival_us"]  = dist_json(p.interval_after_arrival_us);
        o["acceptance"] = acceptance_json(p.acceptance);
        json reqs = json::array();
        int64_t n = 0;
        for (const RequestRecord& q : p.reqs) {
            json e = json::object();
            e["n"] = ++n;
            e["id"] = q.id;
            e["prompt"] = q.prompt;
            e["category"] = q.category;
            e["prompt_tokens"] = q.prompt_tokens;
            e["cached_tokens"] = q.cached_tokens;
            e["drafted"] = q.drafted;
            e["accepted"] = q.accepted;
            e["first_token_s"] = q.first_token_s;
            e["last_token_s"] = q.last_token_s;
            e["finish"] = q.finish;
            e["served_from_cache"] = q.served_from_cache;
            e["tokens"] = q.tokens;
            reqs.push_back(std::move(e));
        }
        o["requests"] = std::move(reqs);
        pts.push_back(std::move(o));
    }
    j["points"] = std::move(pts);

    json cats = json::array();
    for (const auto& c : by_category(r)) {
        cats.push_back({ { "category", c.first }, { "acceptance", acceptance_json(c.second) } });
    }
    j["by_category"] = std::move(cats);

    json found = json::array();
    for (const Finding& f : r.findings) {
        found.push_back({ { "level", f.level == Finding::Fail ? "fail"
                          : f.level == Finding::Warn ? "warn" : "note" },
                          { "id", f.id }, { "what", f.what } });
    }
    j["findings"] = std::move(found);
    return j;
}

std::string report_json(const Report& r) { return report_json_object(r).dump(2) + "\n"; }

/* ------------------------------------------------------------------ the markdown */
namespace {
/* One number, or the reason there is none, in the column's width. A dash with no reason next to
 * it is exactly what this report is written to avoid. */
std::string num(const Measure& m, int digits = 1) {
    if (!m.present) return "not measured (" + m.absent + ")";
    return fmt("%.*f", digits, m.value);
}
std::string dist_text(const Dist& d, int digits = 1) {
    if (d.count == 0) return "not measured (" + d.absent + ")";
    return fmt("%zu: p50 %s p90 %s p99 %s max %s", (size_t)d.count,
               num(d.p50, digits).c_str(), num(d.p90, digits).c_str(),
               num(d.p99, digits).c_str(), num(d.max, digits).c_str());
}
std::string rate(const Measure& m) {
    if (!m.present) return "not measured (" + m.absent + ")";
    return fmt("%.3f", m.value);
}
}  /* namespace */

std::string report_markdown(const Report& r) {
    std::string s;
    s += "# radiance --diagnose\n\n";
    s += "The engine ran a fixed serving workload through its own scheduler. Step times are the "
         "scheduler's own brackets (core/sched/steplog.h), not a client's arrival intervals.\n\n";
    s += "## Run\n\n";
    s += "| | |\n|---|---|\n";
    for (const auto& kv : r.engine) s += "| " + kv.first + " | " + kv.second + " |\n";
    s += fmt("| workload | %llu (a baseline with another number is a different workload) |\n",
             (unsigned long long)r.workload);
    s += fmt("| prefix-cache salt | %lld |\n", (long long)r.salt);
    s += "\n";

    if (!r.findings.empty()) {
        s += "## Findings\n\n";
        for (const Finding& f : r.findings)
            s += fmt("- **%s** `%s`: %s\n",
                     f.level == Finding::Fail ? "FAIL" : f.level == Finding::Warn ? "WARN" : "note",
                     f.id.c_str(), f.what.c_str());
        s += "\n";
    }

    if (!r.absent.empty()) {
        s += "## Not measured by this run\n\n";
        for (const auto& kv : r.absent) s += "- " + kv.first + ": " + kv.second + "\n";
        s += "\n";
    }

    s += "## Step time (microseconds), by step shape\n\n";
    s += "Each distribution is of the steps of that shape, and a step of no shape in the point is "
         "reported as absent with the reason rather than as no steps. The draft head's passes are "
         "kept out of these and counted apart; their cost is the head's, not the trunk's.\n\n";
    s += "| point | concurrency | steps | decode step | mixed step | prefill-chunk step | all steps |\n";
    s += "|---|---|---|---|---|---|---|\n";
    for (const PointReport& p : r.points) {
        s += fmt("| %s | %d | %lld | %s | %s | %s | %s |\n", p.name.c_str(), p.concurrency,
                 (long long)p.steps, dist_text(p.step_us_decode).c_str(),
                 dist_text(p.step_us_mixed).c_str(), dist_text(p.step_us_prefill).c_str(),
                 dist_text(p.step_us_all).c_str());
        if (!p.not_measured.empty())
            s += fmt("| _%s_ | | | **not a measurement**: %s | | | |\n", "", p.not_measured.c_str());
        for (const std::string& a : p.absent) s += fmt("| _%s_ | | | %s | | | |\n", "", a.c_str());
    }
    s += "\n";

    s += "## Tokens on the wire (microseconds)\n\n";
    s += "`interval` is from one step that emitted tokens to the end of the next one that did, and "
         "`waiting` is the part of that the engine spent not running a step -- the end of one such "
         "step to the start of the next. A long interval with a short wait is a long step; a long "
         "wait is a starved scheduler. A client sees these plus the socket, the detokeniser and its "
         "own queueing. A point whose tokens came from one step has neither, and says so.\n\n";
    s += "| point | live | interval | waiting | tokens/step | tokens/sequence | cached\n";
    s += "|---|---|---|---|---|---|---|\n";
    for (const PointReport& p : r.points) {
        const double per_seq = p.requests ? (double)p.out_tokens / (double)p.requests : 0.0;
        s += fmt("| %s | %lld | %s | %s | %s | %.1f | %lld |\n", p.name.c_str(),
                 (long long)p.max_live,
                 dist_text(p.token_interval_us).c_str(), dist_text(p.token_gap_us).c_str(),
                 dist_text(p.tokens_per_step).c_str(), per_seq, (long long)p.cached_tokens);
        if (p.interval_before_arrival_us.count || p.interval_after_arrival_us.count)
            s += fmt("| _%s: the streams already running_ | before the arrival: %s | after: %s | | | |\n",
                     p.name.c_str(), dist_text(p.interval_before_arrival_us).c_str(),
                     dist_text(p.interval_after_arrival_us).c_str());
    }
    s += "\n";

    s += "## Tokens per step, by shape\n\n";
    s += "The same quantity read three ways, because a change in one is not a change in another: "
         "the tokens a step of that shape emitted, and the rows its forward pass computed.\n\n";
    s += "| point | emitted | prefill steps | mixed steps | decode steps | rows |\n";
    s += "|---|---|---|---|---|---|\n";
    for (const PointReport& p : r.points)
        s += fmt("| %s | %s | %s | %s | %s | %s |\n", p.name.c_str(),
                 dist_text(p.tokens_per_step).c_str(), dist_text(p.tok_per_step_prefill).c_str(),
                 dist_text(p.tok_per_step_mixed).c_str(), dist_text(p.tok_per_step_decode).c_str(),
                 dist_text(p.rows_per_step).c_str());
    s += "\n";
    s += "| point | wall s | prompt tok | cached | out tok | prefill tok | decode tok | preempts |\n";
    s += "|---|---|---|---|---|---|---|---|\n";
    for (const PointReport& p : r.points)
        s += fmt("| %s | %.2f | %lld | %lld | %lld | %lld | %lld | %lld |\n", p.name.c_str(),
                 p.wall_s, (long long)p.prompt_tokens, (long long)p.cached_tokens,
                 (long long)p.out_tokens, (long long)p.prefill_tokens, (long long)p.decode_tokens,
                 (long long)p.preemptions);
    s += "\nNothing is meant to come from the cache: every prompt is salted per point and the cache "
         "is dropped before it. A non-zero `cached` row is reported as not a measurement.\n\n";

    s += "## Draft acceptance\n\n";
    s += "`pooled` is accepted over drafted; `by position` is conditional -- of the verifies that "
         "reached position j having accepted every earlier one, how many took it.\n\n";
    s += "| point | drafted | accepted | pooled | deepest | by position |\n";
    s += "|---|---|---|---|---|---|\n";    for (const PointReport& p : r.points) {
        std::string by;
        const int deep = p.acceptance.deepest();
        for (int j = 0; j <= deep; ++j)
            by += fmt("%s%d %s", j ? ", " : "", j, rate(p.acceptance.at((size_t)j)).c_str());
        if (deep < 0) by = "not measured (no draft was verified in this point)";
        s += fmt("| %s | %lld | %lld | %s | %s | %s |\n", p.name.c_str(),
                 (long long)p.acceptance.drafted, (long long)p.acceptance.accepted,
                 rate(p.acceptance.pooled()).c_str(),
                 deep < 0 ? "n/a" : fmt("%d", deep).c_str(), by.c_str());
    }
    s += "\n";

    /* The categories, pooled: a drafter that holds on code and loses on prose is a different
     * change from one that loses everywhere, and only the family view says which. */
    s += "### By category\n\n";
    s += "| category | drafted | accepted | pooled | by position |\n|---|---|---|---|---|\n";
    for (const auto& c : by_category(r)) {
        std::string by;
        const int deep = c.second.deepest();
        for (int j = 0; j <= deep; ++j)
            by += fmt("%s%d %s", j ? ", " : "", j, rate(c.second.at((size_t)j)).c_str());
        if (deep < 0) by = "not measured (no draft was verified in this category)";
        s += fmt("| %s | %lld | %lld | %s | %s |\n", c.first.c_str(),
                 (long long)c.second.drafted, (long long)c.second.accepted,
                 rate(c.second.pooled()).c_str(), by.c_str());
    }
    s += "\n";

    s += "## Token parity\n\n";
    int64_t reqs = 0, toks = 0;
    for (const PointReport& p : r.points)
        for (const RequestRecord& q : p.reqs) { ++reqs; toks += (int64_t)q.tokens.size(); }
    s += fmt("%lld requests, %lld generated tokens, every one of them in the JSON. "
             "`--diagnose-baseline FILE` compares them with an earlier report and fails on any "
             "difference.\n", (long long)reqs, (long long)toks);
    return s;
}

/* ------------------------------------------------------------------ against an earlier report */
bool parse_ok(const std::string& text, std::string* err) {
    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded()) { *err = "not JSON"; return false; }
    if (!j.is_object() || !j.contains("points") || !j["points"].is_array()) {
        *err = "no \"points\" array: not a --diagnose report";
        return false;
    }
    const int schema = j.value("schema", 0);
    if (schema != kSchemaVersion) {
        *err = fmt("schema %d, this build writes schema %d", schema, kSchemaVersion);
        return false;
    }
    return true;
}

bool same_workload(const std::string& base, const std::string& now, std::string* why) {
    const json b = json::parse(base, nullptr, false), n = json::parse(now, nullptr, false);
    if (b.is_discarded() || n.is_discarded()) { *why = "a report does not parse"; return false; }
    const uint64_t bw = b.value("workload", (uint64_t)0), nw = n.value("workload", (uint64_t)0);
    if (bw != nw) {
        *why = fmt("the baseline ran workload %llu and this ran %llu: different points, so its "
                   "numbers are not this run's comparison",
                   (unsigned long long)bw, (unsigned long long)nw);
        return false;
    }
    /* The salt is part of the workload in everything but name: it is in the prompts, so two runs
     * of one build with different salts generated different things. Saying so beats the twenty-odd
     * token-parity failures that comparing them would produce. */
    const int64_t bs = b.value("salt", (int64_t)0), ns = n.value("salt", (int64_t)0);
    if (bs != ns) {
        *why = fmt("the baseline salted its prompts with %lld and this run used %lld: different "
                   "prompts, so its tokens are not what this run would have written -- give both "
                   "the same --diagnose-salt", (long long)bs, (long long)ns);
        return false;
    }
    return true;
}

namespace {
/* Every request of a report, keyed by point and its ordinal within it: stable across builds, and
 * not the engine's request id, which moves when a point is refused and the run goes on. */
struct Key { std::string point; int64_t n; bool operator<(const Key& o) const {
    return point != o.point ? point < o.point : n < o.n; } };

std::map<Key, const json*> requests_of(const json& report, std::map<std::string, std::string>* cats) {
    std::map<Key, const json*> out;
    for (const json& p : report["points"]) {
        if (!p.contains("requests") || !p["requests"].is_array()) continue;
        const std::string point = p.value("name", std::string());
        for (const json& q : p["requests"]) {
            if (!q.contains("tokens") || !q["tokens"].is_array()) continue;
            out[{ point, q.value("n", (int64_t)0) }] = &q;
            if (cats) cats->insert({ point, p.value("category", std::string()) });
        }
    }
    return out;
}
}  /* namespace */

void compare_tokens(const std::string& base, const std::string& now, std::vector<Finding>* out,
                    int show) {
    const json b = json::parse(base, nullptr, false), n = json::parse(now, nullptr, false);
    if (b.is_discarded() || n.is_discarded()) {
        out->push_back({ Finding::Fail, "token-parity", "a report does not parse" });
        return;
    }
    const std::map<Key, const json*> old = requests_of(b, nullptr), cur = requests_of(n, nullptr);
    int64_t missing = 0, extra = 0, differ = 0, shown = 0;
    std::string first;
    for (const auto& e : cur) {
        auto it = old.find(e.first);
        if (it == old.end()) { ++missing; continue; }
        const json& bq = *it->second;
        const json& cq = *e.second;
        if (bq["tokens"] == cq["tokens"]) continue;
        ++differ;
        if (shown < show) {
            ++shown;
            /* Where they part, so the reader starts looking there rather than diffing two arrays
             * by hand. */
            const auto& bt = bq["tokens"];
            const auto& ct = cq["tokens"];
            size_t at = 0;
            while (at < bt.size() && at < ct.size() && bt[at] == ct[at]) ++at;
            std::string w = fmt("%s request %lld ('%s'): the first %zu token(s) match, then the "
                                "baseline has %s and this has %s (%zu tokens against %zu)",
                                e.first.point.c_str(), (long long)e.first.n,
                                cq.value("prompt", std::string()).c_str(), at,
                                at < bt.size() ? bt[at].dump().c_str() : "nothing",
                                at < ct.size() ? ct[at].dump().c_str() : "nothing",
                                bt.size(), ct.size());
            if (first.empty()) first = std::move(w);
        }
    }
    for (const auto& e : old) if (!cur.count(e.first)) ++extra;

    if (differ || missing || extra) {
        Finding f;
        f.level = Finding::Fail;
        f.id = "token-parity";
        f.what = fmt("%lld request(s) generated different tokens, %lld are in this run and not the "
                     "baseline, %lld in the baseline and not here",
                     (long long)differ, (long long)missing, (long long)extra);
        if (!first.empty()) f.what += "; " + first;
        out->push_back(std::move(f));
        return;
    }
    out->push_back({ Finding::Note, "token-parity",
                     fmt("all %zu request(s) generated exactly the baseline's tokens", cur.size()) });
}

void compare_acceptance(const std::string& base, const std::string& now, double drop,
                        std::vector<Finding>* out) {
    const json b = json::parse(base, nullptr, false), n = json::parse(now, nullptr, false);
    if (b.is_discarded() || n.is_discarded()) {
        out->push_back({ Finding::Warn, "acceptance-drop", "a report does not parse" });
        return;
    }
    std::map<std::string, const json*> bpts, npts;
    for (const json& p : b["points"]) bpts[p.value("name", std::string())] = &p;
    for (const json& p : n["points"]) npts[p.value("name", std::string())] = &p;
    /* And by category, which is where a drafter's change usually shows first: prose and code are
     * different amounts of text to have seen, and a category that moves alone says the workload
     * moved rather than the build. The arrays are held here rather than ranged over inline, because
     * the maps below keep pointers into them. */
    const json bcats_arr = b.value("by_category", json::array());
    const json ncats_arr = n.value("by_category", json::array());
    std::map<std::string, const json*> bcats, ncats;
    for (const json& c : bcats_arr) bcats[c.value("category", std::string())] = &c;
    for (const json& c : ncats_arr) ncats[c.value("category", std::string())] = &c;

    int64_t drops = 0;
    /* One empty object to point at where a report has no acceptance at all, so the loop below has
     * one path. */
    static const json empty = json::object();
    for (const auto& e : ncats) {
        auto it = bcats.find(e.first);
        if (it == bcats.end()) continue;
        const json& bx = it->second->contains("acceptance") ? it->second->at("acceptance") : empty;
        const json& nx = e.second->contains("acceptance") ? e.second->at("acceptance") : empty;
        const int64_t bd = bx.value("drafted", (int64_t)0), nd = nx.value("drafted", (int64_t)0);
        if (!bd || !nd) continue;
        const double br = (double)bx.value("accepted", (int64_t)0) / (double)bd;
        const double nr = (double)nx.value("accepted", (int64_t)0) / (double)nd;
        if (br > 0 && nr < br * (1.0 - drop)) {
            ++drops;
            out->push_back({ Finding::Warn, "acceptance-drop",
                             fmt("category %s: pooled acceptance %.3f over %lld drafted, the "
                                 "baseline's %.3f over %lld (%.0f%% of it)",
                                 e.first.c_str(), nr, (long long)nd, br, (long long)bd,
                                 100.0 * nr / br) });
        }
    }

    for (const auto& e : npts) {
        auto it = bpts.find(e.first);
        if (it == bpts.end()) continue;
        /* Held by reference: `value()` would copy, and the position arrays below are read through
         * the copy. */
        const json& ba = it->second->contains("acceptance") ? it->second->at("acceptance") : empty;
        const json& na = e.second->contains("acceptance") ? e.second->at("acceptance") : empty;
        const int64_t bd = ba.value("drafted", (int64_t)0), nd = na.value("drafted", (int64_t)0);
        /* Nothing drafted in either: no drafter here, then and now, which is not a drop. */
        if (!bd || !nd) continue;
        const double br = (double)ba.value("accepted", (int64_t)0) / (double)bd;
        const double nr = (double)na.value("accepted", (int64_t)0) / (double)nd;
        if (br > 0 && nr < br * (1.0 - drop)) {
            ++drops;
            out->push_back({ Finding::Warn, "acceptance-drop",
                             fmt("%s: pooled acceptance %.3f, the baseline's %.3f (%.0f%% of it)",
                                 e.first.c_str(), nr, br, 100.0 * nr / br) });
        }
        /* By position: a drafter that loses a deep position while holding the shallow ones is a
         * different change from one that loses them all, and only the position numbers say which. */
        if (!ba.contains("positions") || !na.contains("positions")) continue;
        std::map<int64_t, const json*> bpos;
        for (const json& p : ba["positions"]) bpos[p.value("position", (int64_t)-1)] = &p;
        for (const json& p : na["positions"]) {
            const int64_t j = p.value("position", (int64_t)-1);
            auto pit = bpos.find(j);
            if (pit == bpos.end()) continue;
            const int64_t brch = pit->second->value("reached", (int64_t)0);
            const int64_t nrch = p.value("reached", (int64_t)0);
            if (!brch || !nrch) continue;
            const double bh = (double)pit->second->value("hits", (int64_t)0) / (double)brch;
            const double nh = (double)p.value("hits", (int64_t)0) / (double)nrch;
            if (bh > 0 && nh < bh * (1.0 - drop))
                out->push_back({ Finding::Warn, "acceptance-drop",
                                 fmt("%s position %lld: conditional acceptance %.3f over %lld "
                                     "verifies, the baseline's %.3f over %lld",
                                     e.first.c_str(), (long long)j, nh, (long long)nrch, bh,
                                     (long long)brch) });
        }
    }
    if (!drops)
        out->push_back({ Finding::Note, "acceptance-drop",
                         fmt("no point's pooled acceptance fell by %.0f%% or more against the "
                             "baseline", drop * 100.0) });
}

void review(const std::string& now, std::vector<Finding>* out) {
    const json j = json::parse(now, nullptr, false);
    if (j.is_discarded()) { out->push_back({ Finding::Fail, "report", "the report does not parse" });
                            return; }
    int64_t not_measured = 0, dropped = 0, cached = 0, requests = 0, drafted = 0, tokens = 0;
    for (const json& p : j["points"]) {
        const std::string name = p.value("name", std::string());
        if (!p.value("measured", true)) {
            ++not_measured;
            out->push_back({ Finding::Warn, "not-a-measurement",
                             fmt("%s: %s", name.c_str(),
                                 p.value("not_measured", std::string("the run said so")).c_str()) });
        }
        const int64_t d = p.value("step_records_dropped", (int64_t)0);
        if (d) {
            dropped += d;
        }
        const int64_t c = p.contains("tokens") ? p["tokens"].value("cached", (int64_t)0) : 0;
        if (c) cached += c;
        if (p.contains("requests")) {
            requests += (int64_t)p["requests"].size();
            for (const json& q : p["requests"]) tokens += (int64_t)q.value("tokens", json::array()).size();
        }
        if (p.contains("acceptance")) drafted += p["acceptance"].value("drafted", (int64_t)0);
    }
    /* The report's own arithmetic, checked on the way out. A distribution whose p50 sits above its
     * p90, or whose mean is outside its own range, is a bug in this file or in the reader's parse,
     * and a reader comparing two builds would be comparing the bug. */
    int64_t bad = 0;
    for (const json& p : j["points"]) {
        for (const char* key : { "decode", "mixed", "prefill", "all" }) {
            if (!p.contains("step_us") || !p["step_us"].contains(key)) continue;
            const json& d = p["step_us"][key];
            if (!d.contains("p50") || !d["p50"].is_number()) continue;
            const double p50 = d["p50"], p90 = d.value("p90", p50), mx = d.value("max", p50);
            const double mean = d.value("mean", p50);
            if (p50 > p90 + 1e-9 || mean < p50 - (mx - p50) * 2 - 1e-9 || mx + 1e-9 < p90) ++bad;
        }
    }
    if (bad)
        out->push_back({ Finding::Fail, "report-arithmetic",
                         fmt("%lld step-time series is not ordered the way a distribution is: p50 "
                             "above p90, a max below p90, or a mean outside the range", (long long)bad) });
    if (!tokens)
        out->push_back({ Finding::Fail, "workload",
                         "the workload generated no token at all, so there is nothing to compare "
                         "and nothing to time" });
    if (!drafted)
        out->push_back({ Finding::Note, "acceptance",
                         "nothing was speculated in this run, so every acceptance figure is absent "
                         "rather than zero: this model declares no draft head, or speculation is "
                         "off (--n-spec 0)" });
    if (dropped)
        out->push_back({ Finding::Warn, "step-log",
                         fmt("%lld step record(s) were overwritten before they were read: the "
                             "distributions cover what survived", (long long)dropped) });
    if (cached)
        out->push_back({ Finding::Warn, "cache",
                         fmt("%lld prompt token(s) came from the prefix cache in a run that is "
                             "supposed to compute every one", (long long)cached) });
    if (!requests)
        out->push_back({ Finding::Fail, "workload", "no request of the workload completed" });
    (void)not_measured;
}

}  /* namespace diagnose */
}  /* namespace rad */
