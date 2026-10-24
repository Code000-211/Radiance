/* diagnose.cpp -- see diag/diagnose.h. */
#include "diagnose.h"

#include "diag/diag.h"

#include "rad_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

namespace rad {
namespace diag {

/* ================================================================== the ring */

void StepRing::reset(size_t capacity) {
    buf_.assign(capacity, StepRecord{});
    n_ = 0;
}

void StepRing::push(const StepRecord& r) noexcept {
    if (buf_.empty()) return;
    buf_[(size_t)(n_ % buf_.size())] = r;
    ++n_;
}

uint64_t StepRing::copy_since(uint64_t from, std::vector<StepRecord>* out) const {
    if (buf_.empty() || from >= n_) return 0;
    const uint64_t oldest = n_ > buf_.size() ? n_ - buf_.size() : 0;
    const uint64_t lost = from < oldest ? oldest - from : 0;
    for (uint64_t s = std::max(from, oldest); s < n_; ++s) out->push_back(buf_[(size_t)(s % buf_.size())]);
    return lost;
}

/* ================================================================== statistics */

/* Nearest rank: the smallest sample with at least p of the samples at or below it. No
 * interpolation, so every reported value is one that was measured. */
static size_t nearest_rank(size_t n, double p) {
    const size_t k = (size_t)std::ceil(p * (double)n);
    return k == 0 ? 0 : std::min(k - 1, n - 1);
}

Quantiles quantiles(std::vector<double> v) {
    Quantiles q;
    q.n = (int64_t)v.size();
    if (v.empty()) return q;
    std::sort(v.begin(), v.end());
    auto at = [&](double p) { return v[nearest_rank(v.size(), p)]; };
    q.min = v.front();
    q.max = v.back();
    q.p50 = at(0.50);
    q.p90 = at(0.90);
    q.p99 = at(0.99);
    q.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    return q;
}

json to_json(const Quantiles& q) {
    json j;
    j["n"] = q.n;
    if (q.n == 0) {
        for (const char* k : { "min", "p50", "p90", "p99", "max", "mean" }) j[k] = nullptr;
        return j;
    }
    auto r = [](double x) { return std::round(x * 1000.0) / 1000.0; };
    j["min"] = r(q.min);
    j["p50"] = r(q.p50);
    j["p90"] = r(q.p90);
    j["p99"] = r(q.p99);
    j["max"] = r(q.max);
    j["mean"] = r(q.mean);
    return j;
}

namespace {

double round3(double x) { return std::round(x * 1000.0) / 1000.0; }

/* A ratio that has no denominator is not a number to report. */
json ratio(double num, double den) {
    if (den <= 0) return nullptr;
    return round3(num / den);
}

/* The spread of a set of durations, as a coefficient of variation: the standard deviation over
 * the mean. Unitless, so a step that got faster and steadier and one that got faster and as
 * uneven can be told apart. */
json coeff_var(const std::vector<double>& v) {
    if (v.size() < 2) return nullptr;
    const double m = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();
    if (m <= 0) return nullptr;
    double ss = 0;
    for (double x : v) ss += (x - m) * (x - m);
    return round3(std::sqrt(ss / (double)(v.size() - 1)) / m);
}

std::string num(const json& v, const char* unit = "") {
    if (v.is_null()) return "–";
    char b[64];
    if (v.is_number_integer()) std::snprintf(b, sizeof b, "%lld%s", (long long)v.get<int64_t>(), unit);
    else std::snprintf(b, sizeof b, "%.3g%s", v.get<double>(), unit);
    return b;
}

/* Two-proportion z for k1/n1 against k0/n0, negative when the first is lower; NaN without data. */
double drop_z(int64_t n1, int64_t k1, int64_t n0, int64_t k0) {
    if (n1 <= 0 || n0 <= 0) return std::nan("");
    const double p1 = (double)k1 / (double)n1, p0 = (double)k0 / (double)n0;
    const double p = (double)(k1 + k0) / (double)(n1 + n0);
    const double se = std::sqrt(p * (1.0 - p) * (1.0 / (double)n1 + 1.0 / (double)n0));
    if (se <= 0) return p1 == p0 ? 0.0 : (p1 < p0 ? -INFINITY : INFINITY);
    return (p1 - p0) / se;
}

/* The regularised incomplete beta function I_x(a, b), by its continued fraction (Lentz). */
double inc_beta(double x, double a, double b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    if (x > (a + 1) / (a + b + 2)) return 1.0 - inc_beta(1.0 - x, b, a);
    const double lbeta = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
    const double front = std::exp(std::log(x) * a + std::log(1.0 - x) * b + lbeta) / a;
    const double tiny = 1e-300;
    double f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
    if (std::fabs(d) < tiny) d = tiny;
    d = 1 / d;
    f = d;
    for (int m = 1; m <= 300; ++m) {
        for (int odd = 0; odd < 2; ++odd) {
            const double num = odd ? -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                                   : m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
            d = 1 + num * d;
            if (std::fabs(d) < tiny) d = tiny;
            c = 1 + num / c;
            if (std::fabs(c) < tiny) c = tiny;
            d = 1 / d;
            f *= c * d;
        }
        if (std::fabs(c * d - 1) < 1e-12) break;
    }
    return front * f;
}

/* P(T <= t) for Student's t with `df` degrees of freedom. */
double t_cdf(double t, double df) {
    if (std::isinf(t)) return t < 0 ? 0.0 : 1.0;
    const double tail = 0.5 * inc_beta(df / (df + t * t), df / 2, 0.5);
    return t < 0 ? tail : 1.0 - tail;
}

/* The one-sided probability of z = -3: the threshold every acceptance test below uses, so a
 * request-level t test and a pooled z test call the same thing a drop. */
constexpr double kDropP = 0.0013498980316301;

const char* phase_name(int p) {
    return p == RAD_PHASE_DECODE ? "decode" : p == RAD_PHASE_PREFILL ? "prefill" : "mixed";
}

}  /* namespace */

json slow_steps(const std::vector<StepRecord>& steps, int phase) {
    std::vector<const StepRecord*> s;
    for (const StepRecord& r : steps)
        if (r.phase == phase) s.push_back(&r);
    json j;
    j["phase"] = phase_name(phase);
    j["steps"] = (int64_t)s.size();
    if (s.empty()) {
        j["typical"] = nullptr;
        j["slowest"] = json::array();
        return j;
    }
    std::sort(s.begin(), s.end(), [](const StepRecord* a, const StepRecord* b) { return a->ns < b->ns; });
    auto row = [](const StepRecord& r) {
        json x;
        x["step"] = r.step;
        x["ms"] = round3((double)r.ns / 1e6);
        x["n_seq"] = r.n_seq;
        x["prefill_tok"] = r.prefill_tok;
        x["decode_tok"] = r.decode_tok;
        return x;
    };
    j["typical"] = row(*s[(s.size() - 1) / 2]);
    const size_t k = std::min(s.size(), std::max<size_t>(3, s.size() / 100));
    json slow = json::array();
    for (size_t i = 0; i < k; ++i) slow.push_back(row(*s[s.size() - 1 - i]));
    j["slowest"] = slow;
    return j;
}

json pool_draft_positions(const std::vector<json>& runs) {
    std::vector<int64_t> reached, taken;
    for (const json& r : runs) {
        if (!r.contains("counters") || !r["counters"].contains("draft_by_position")) continue;
        for (const json& x : r["counters"]["draft_by_position"]) {
            const size_t p = (size_t)x.value("position", 0);
            if (p < 1) continue;
            if (reached.size() < p) {
                reached.resize(p, 0);
                taken.resize(p, 0);
            }
            reached[p - 1] += x.value("reached", (int64_t)0);
            taken[p - 1] += x.value("accepted_n", (int64_t)0);
        }
    }
    json out = json::array();
    for (size_t p = 0; p < reached.size(); ++p)
        out.push_back({ { "position", (int64_t)p + 1 }, { "reached", reached[p] }, { "accepted_n", taken[p] },
                        { "accepted", ratio((double)taken[p], (double)reached[p]) } });
    return out;
}

json phase_report(const PhaseInput& in) {
    json j;
    j["name"] = in.name;
    j["kind"] = in.kind == PhaseKind::Decode ? "decode" : "prefill";
    j["streams"] = in.streams;
    j["wall_s"] = round3((double)in.wall_ns / 1e9);

    /* ---- requests: what a client of this phase saw ---- */
    int ok = 0, stopped = 0;
    int64_t tokens = 0, drafted = 0, accepted = 0, cached = 0;
    std::vector<double> ttft_ms, queue_ms, stream_tps, prefill_tps;
    int64_t first_min = INT64_MAX, last_max = INT64_MIN, first_max = INT64_MIN;
    for (const RequestTiming& r : in.requests) {
        ok += r.ok;
        stopped += r.stopped;
        cached += r.cached;
        tokens += r.tokens;
        drafted += r.drafted;
        accepted += r.accepted;
        if (r.admitted_ns >= 0 && r.first_ns >= 0) {
            ttft_ms.push_back((double)(r.first_ns - r.admitted_ns) / 1e6);
            if (r.start_ns >= 0) queue_ms.push_back((double)(r.admitted_ns - r.start_ns) / 1e6);
            const int32_t computed = r.prompt - r.cached;
            if (computed > 0 && r.first_ns > r.admitted_ns)
                prefill_tps.push_back((double)computed * 1e9 / (double)(r.first_ns - r.admitted_ns));
        }
        if (r.first_ns >= 0 && r.last_ns > r.first_ns && r.tokens > 1)
            stream_tps.push_back((double)(r.tokens - 1) * 1e9 / (double)(r.last_ns - r.first_ns));
        if (r.first_ns >= 0) {
            first_min = std::min(first_min, r.first_ns);
            first_max = std::max(first_max, r.first_ns);
        }
        if (r.last_ns >= 0) last_max = std::max(last_max, r.last_ns);
    }
    json req;
    req["count"] = (int64_t)in.requests.size();
    req["ok"] = ok;
    req["stopped_on_eos"] = stopped;
    req["tokens"] = tokens;
    /* Prompt tokens the prefix cache served. The workload is salted so this is 0; anything else
     * means the phase measured a cache hit, and the driver says so. */
    req["cached_prompt_tokens"] = cached;
    {
        json outs = json::array(), drafts = json::array(), prompts = json::array();
        for (const RequestTiming& r : in.requests) {
            outs.push_back(r.output);
            drafts.push_back(json::array({ r.drafted, r.accepted }));
            prompts.push_back(r.prompt);
        }
        req["outputs"] = outs;
        /* Prompt tokens a request, so a difference can be read against what its prefill took:
         * a path that only long prompts take shows only in long prompts. */
        req["prompt_tokens"] = prompts;
        /* [proposed, accepted] a request: the unit the acceptance comparison pairs on. */
        req["drafts"] = drafts;
    }
    /* TTFT FROM ADMISSION, and the queue wait beside it: the first is the engine's latency, the
     * second is how long the scheduler kept the request waiting, and a client sees their sum. */
    req["ttft_ms"] = to_json(quantiles(ttft_ms));
    req["queue_ms"] = to_json(quantiles(queue_ms));
    req["stream_tok_s"] = to_json(quantiles(stream_tps));
    if (in.kind == PhaseKind::Prefill) req["prefill_tok_s"] = to_json(quantiles(prefill_tps));
    /* THE AGGREGATE DECODE RATE: every token after each stream's own first, over the span from the
     * earliest first token to the last token. The first tokens are left out because they are the
     * prefill's; a stream whose first token came late lowers the rate, as it would for a client. */
    if (in.kind == PhaseKind::Decode && first_max > INT64_MIN && last_max > first_max) {
        int64_t after_first = 0;
        for (const RequestTiming& r : in.requests) after_first += std::max<int64_t>(r.tokens - 1, 0);
        req["aggregate_tok_s"] = round3((double)after_first * 1e9 / (double)(last_max - first_min));
    } else {
        req["aggregate_tok_s"] = nullptr;
    }
    j["requests"] = req;

    /* ---- the engine's own counters over the phase ---- */
    const SchedMetrics& a = in.before;
    const SchedMetrics& b = in.after;
    const int64_t d_steps = b.steps - a.steps;
    const int64_t d_dec = b.steps_decode - a.steps_decode;
    json c;
    c["steps"] = d_steps;
    c["steps_decode"] = d_dec;
    c["steps_prefill"] = b.steps_prefill - a.steps_prefill;
    c["steps_mixed"] = b.steps_mixed - a.steps_mixed;
    c["steps_ahead"] = b.steps_ahead - a.steps_ahead;
    c["decode_rows_unspeculated"] = b.decode_rows_unspeculated - a.decode_rows_unspeculated;
    c["preemptions"] = b.preemptions - a.preemptions;
    c["prefill_tokens"] = b.prefill_tokens - a.prefill_tokens;
    c["decode_tokens"] = b.decode_tokens - a.decode_tokens;
    c["ms_per_decode_step"] = ratio((double)(b.step_ns_decode - a.step_ns_decode) / 1e6, (double)d_dec);
    c["draft_tokens"] = b.draft_tokens - a.draft_tokens;
    c["draft_accepted"] = b.draft_accepted - a.draft_accepted;
    c["draft_acceptance"] = ratio((double)(b.draft_accepted - a.draft_accepted),
                                  (double)(b.draft_tokens - a.draft_tokens));
    /* The conditional rate at each draft position: of the verifies that reached j, the share that
     * took j as well. The pooled rate above hides a single collapsed position. */
    json pos = json::array();
    const int depth = std::min(std::max(a.draft_depth_seen, b.draft_depth_seen), (int)SchedMetrics::RAD_METRICS_MAX_SPEC);
    for (int p = 0; p < depth; ++p) {
        const int64_t reached = b.draft_pos_reached[p] - a.draft_pos_reached[p];
        json x;
        x["position"] = p + 1;
        x["reached"] = reached;
        x["accepted_n"] = b.draft_pos_accepted[p] - a.draft_pos_accepted[p];
        x["accepted"] = ratio((double)(b.draft_pos_accepted[p] - a.draft_pos_accepted[p]), (double)reached);
        pos.push_back(x);
    }
    c["draft_by_position"] = pos;
    j["counters"] = c;

    /* ---- the steps themselves ---- */
    json st;
    st["recorded"] = (int64_t)in.steps.size();
    st["lost"] = (int64_t)in.steps_lost;
    std::vector<double> dec_ms, gaps_ms, rate;
    int64_t dec_tok = 0;
    int64_t prev_end = -1;
    for (const StepRecord& r : in.steps) {
        if (r.phase == RAD_PHASE_DECODE) {
            dec_ms.push_back((double)r.ns / 1e6);
            dec_tok += r.decode_tok;
        }
        /* EMISSIONS: a step that committed decode tokens is one burst a stream receives. The
         * interval between two is what a client measures as a token gap, plus delivery; it
         * includes the host's planning and commit between steps, which step_ns does not. */
        if (r.decode_tok > 0) {
            if (prev_end >= 0 && r.end_ns > prev_end) {
                const double g = (double)(r.end_ns - prev_end) / 1e6;
                gaps_ms.push_back(g);
                rate.push_back((double)r.decode_tok / (g / 1e3));
            }
            prev_end = r.end_ns;
        }
    }
    st["decode_step_ms"] = to_json(quantiles(dec_ms));
    st["decode_tok_per_step"] = ratio((double)dec_tok, (double)dec_ms.size());
    st["emission_interval_ms"] = to_json(quantiles(gaps_ms));
    /* JITTER, two ways: how uneven the steps are (coefficient of variation of decode step time),
     * and how far the slow tail of emission gaps sits from the typical one (p99 over p50). */
    st["jitter_step_cv"] = coeff_var(dec_ms);
    {
        const Quantiles q = quantiles(gaps_ms);
        st["jitter_emission_p99_over_p50"] = q.n && q.p50 > 0 ? json(round3(q.p99 / q.p50)) : json(nullptr);
    }
    /* THE 1% LOW TOKEN RATE: committed tokens over the emission interval, the slowest 1% of
     * emissions. With speculation a burst carries several tokens, so this is a rate per burst,
     * not one token's latency. Nearest rank, as quantiles(): floor(p*n) would take the second
     * slowest sample at exactly 100 intervals and flatter the slow edge. */
    {
        json low;
        if (!rate.empty()) {
            std::sort(rate.begin(), rate.end());
            low = round3(rate[nearest_rank(rate.size(), 0.01)]);
        }
        st["tok_s_1pct_low"] = low;
    }
    st["slow_decode_steps"] = slow_steps(in.steps, RAD_PHASE_DECODE);
    if (in.kind == PhaseKind::Prefill) {
        std::vector<double> pf;
        for (const StepRecord& r : in.steps)
            if (r.prefill_tok > 0) pf.push_back((double)r.ns / 1e6);
        st["prefill_chunk_ms"] = to_json(quantiles(pf));
    }
    j["steps"] = st;
    return j;
}

json summarize_runs(const std::vector<json>& runs) {
    static const std::vector<std::string> kPaths = {
        "/requests/aggregate_tok_s",       "/requests/stream_tok_s/p50", "/requests/ttft_ms/p50",
        "/requests/prefill_tok_s/p50",     "/counters/ms_per_decode_step",
        "/steps/decode_tok_per_step",      "/counters/draft_acceptance",
        "/steps/emission_interval_ms/p50", "/steps/emission_interval_ms/max",
        "/steps/jitter_emission_p99_over_p50", "/steps/jitter_step_cv", "/steps/tok_s_1pct_low",
        "/steps/prefill_chunk_ms/p50",
    };
    return summarize_paths(runs, kPaths);
}

json summarize_paths(const std::vector<json>& runs, const std::vector<std::string>& paths) {
    json j;
    j["runs"] = (int64_t)runs.size();
    for (const std::string& path : paths) {
        const json::json_pointer ptr(path);
        std::vector<double> v;
        for (const json& r : runs)
            if (r.contains(ptr) && r.at(ptr).is_number()) v.push_back(r.at(ptr).get<double>());
        std::string key = path.substr(1);
        std::replace(key.begin(), key.end(), '/', '.');
        if (v.empty()) {
            j[key] = nullptr;
            continue;
        }
        const Quantiles q = quantiles(v);
        j[key] = { { "median", round3(q.p50) }, { "min", round3(q.min) }, { "max", round3(q.max) } };
    }
    return j;
}

json interference_report(const std::vector<StepRecord>& steps, int64_t from_ns, int64_t to_ns) {
    json j;
    j["window_ms"] = to_ns > from_ns ? json(round3((double)(to_ns - from_ns) / 1e6)) : json(nullptr);
    std::vector<double> before;
    double worst = -1;
    int64_t in_window = 0, prev = -1;
    for (const StepRecord& r : steps) {
        if (r.decode_tok <= 0) continue;
        if (prev >= 0 && r.end_ns > prev) {
            const double g = (double)(r.end_ns - prev) / 1e6;
            if (r.end_ns <= from_ns) before.push_back(g);
            else if (prev < to_ns) worst = std::max(worst, g);   /* the gap overlaps the window */
        }
        if (r.end_ns > from_ns && r.end_ns <= to_ns) ++in_window;
        prev = r.end_ns;
    }
    const Quantiles b = quantiles(before);
    j["emissions_in_window"] = in_window;
    j["typical_gap_ms"] = b.n ? json(round3(b.p50)) : json(nullptr);
    j["longest_gap_ms"] = worst >= 0 ? json(round3(worst)) : json(nullptr);
    j["stall_ratio"] = b.n && b.p50 > 0 && worst >= 0 ? json(round3(worst / b.p50)) : json(nullptr);
    return j;
}

/* ================================================================== parity */

std::string workload_fingerprint(const json& p) {
    /* FNV-1a over the canonical dump: stable across builds, and the parameters are few. */
    const std::string d = p.dump();
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : d) { h ^= c; h *= 1099511628211ull; }
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
}

int compare_outputs(const json& now, const json& base, json* out, std::string* why) {
    auto fp = [](const json& r) {
        return r.contains("workload") ? r["workload"].value("fingerprint", std::string()) : std::string();
    };
    if (fp(now).empty() || fp(now) != fp(base)) {
        *why = "the two reports ran different workloads (workload.fingerprint " + fp(now) + " against " +
               fp(base) + "), so their prompts differ and their outputs cannot be compared";
        return RAD_E_INVAL;
    }
    struct Section { const char* name; const char* key; };
    static const Section kSections[] = { { "categories", "category" }, { "concurrency", "streams" },
                                         { "prefill", "requested_tokens" } };
    int64_t compared = 0, identical = 0;
    json differing = json::array(), missing = json::array(), acceptance = json::array(), dropped = json::array();
    for (const Section& sec : kSections) {
        const json empty = json::array();
        const json& a = now.contains(sec.name) ? now[sec.name] : empty;
        const json& b = base.contains(sec.name) ? base[sec.name] : empty;
        auto find = [&](const json& list, const json& key) -> const json* {
            for (const json& p : list)
                if (p.contains(sec.key) && p[sec.key] == key) return &p;
            return nullptr;
        };
        auto label = [&](const json& p) {
            const json& key = p[sec.key];
            return std::string(sec.name) + " " + (key.is_string() ? key.get<std::string>() : key.dump());
        };
        for (const json& pa : a) {
            const json& key = pa[sec.key];
            const std::string point = label(pa);
            const json* pb = find(b, key);
            if (!pb) {
                missing.push_back(point + ": not in the baseline");
                continue;
            }
            /* DRAFT ACCEPTANCE BESIDE THE TOKENS. Bit-identical builds accept identically; a build
             * that is not shows here first, because the draft head reads the trunk's hidden state
             * and an inaccuracy there costs acceptance before it changes an answer.
             *
             * THE UNIT IS THE REQUEST, NOT THE DRAFT. Drafts within one text are not independent --
             * a stretch of boilerplate is accepted wholesale -- so a test that counts every draft as
             * a coin flip takes two texts that differ for any reason for a drop: two runs of the
             * same build that diverged a few tokens in did exactly that. The overall rate is tested
             * on the requests, paired (the same prompt and seed on both sides), by a one-sided t
             * test; a drop is p below that of z = -3. The positions have only pooled counts, so
             * their two-proportion z is divided by the square root of the dispersion the requests
             * show beyond a binomial one, and a drop is that z below -3. */
            auto pooled = [](const json& p) {
                int64_t n = 0, k = 0;
                for (const json& r : p.value("runs", json::array()))
                    if (r.contains("counters")) {
                        n += r["counters"].value("draft_tokens", (int64_t)0);
                        k += r["counters"].value("draft_accepted", (int64_t)0);
                    }
                return std::make_pair(n, k);
            };
            std::vector<double> diffs;
            {
                const json& ra = pa.value("runs", json::array());
                const json& rb = (*pb).value("runs", json::array());
                for (size_t r = 0; r < std::min(ra.size(), rb.size()); ++r) {
                    const json da = ra[r].contains("requests") ? ra[r]["requests"].value("drafts", json::array()) : json::array();
                    const json db = rb[r].contains("requests") ? rb[r]["requests"].value("drafts", json::array()) : json::array();
                    for (size_t k = 0; k < std::min(da.size(), db.size()); ++k) {
                        const double n1 = da[k][0].get<double>(), k1 = da[k][1].get<double>();
                        const double n0 = db[k][0].get<double>(), k0 = db[k][1].get<double>();
                        if (n1 > 0 && n0 > 0) diffs.push_back(k1 / n1 - k0 / n0);
                    }
                }
            }
            const auto [n1, k1] = pooled(pa);
            const auto [n0, k0] = pooled(*pb);
            double phi = 1.0;
            if (n1 > 0 || n0 > 0) {
                json x = { { "point", point }, { "what", "overall" },
                           { "now", ratio((double)k1, (double)n1) }, { "baseline", ratio((double)k0, (double)n0) },
                           { "n_now", n1 }, { "n_baseline", n0 }, { "requests", (int64_t)diffs.size() } };
                bool drop = false;
                if (diffs.size() >= 2) {
                    const double m = (double)diffs.size();
                    const double mean = std::accumulate(diffs.begin(), diffs.end(), 0.0) / m;
                    double ss = 0;
                    for (double d : diffs) ss += (d - mean) * (d - mean);
                    const double se = std::sqrt(ss / (m - 1) / m);
                    const double t = se > 0 ? mean / se : (mean == 0 ? 0.0 : (mean < 0 ? -INFINITY : INFINITY));
                    const double p = t_cdf(t, m - 1);
                    x["test"] = "paired t over requests";
                    x["t"] = std::isfinite(t) ? json(round3(t)) : json(nullptr);
                    x["p"] = p;
                    drop = p < kDropP;
                    /* The requests' spread against a binomial one, for the positions below. */
                    if (n1 > 0 && n0 > 0) {
                        const double pp = (double)(k1 + k0) / (double)(n1 + n0);
                        const double se_bin = std::sqrt(pp * (1 - pp) * (1.0 / (double)n1 + 1.0 / (double)n0));
                        if (se_bin > 0) phi = std::max(1.0, (se / se_bin) * (se / se_bin));
                    }
                } else {
                    const double z = drop_z(n1, k1, n0, k0);
                    x["test"] = "pooled z (fewer than two requests to pair)";
                    x["z"] = std::isfinite(z) ? json(round3(z)) : json(nullptr);
                    drop = std::isfinite(z) && z < -3.0;
                }
                acceptance.push_back(x);
                if (drop) dropped.push_back(x);
            }
            {
                const json pn = pa.value("draft_by_position", json::array());
                const json pbp = (*pb).value("draft_by_position", json::array());
                for (size_t q = 0; q < std::min(pn.size(), pbp.size()); ++q) {
                    const int64_t a1 = pn[q].value("reached", (int64_t)0), b1 = pn[q].value("accepted_n", (int64_t)0);
                    const int64_t a0 = pbp[q].value("reached", (int64_t)0), b0 = pbp[q].value("accepted_n", (int64_t)0);
                    const double z = drop_z(a1, b1, a0, b0) / std::sqrt(phi);
                    json x = { { "point", point }, { "what", "position " + std::to_string(q + 1) },
                               { "now", ratio((double)b1, (double)a1) }, { "baseline", ratio((double)b0, (double)a0) },
                               { "n_now", a1 }, { "n_baseline", a0 }, { "test", "pooled z / sqrt(dispersion)" },
                               { "dispersion", round3(phi) },
                               { "z", std::isfinite(z) ? json(round3(z)) : json(nullptr) } };
                    acceptance.push_back(x);
                    if (std::isfinite(z) && z < -3.0) dropped.push_back(x);
                }
            }
            const json& ra = pa.value("runs", json::array());
            const json& rb = (*pb).value("runs", json::array());
            for (size_t r = 0; r < std::max(ra.size(), rb.size()); ++r) {
                if (r >= ra.size() || r >= rb.size()) {
                    missing.push_back(point + " run " + std::to_string(r + 1) + ": only one side has it");
                    continue;
                }
                const json& oa = ra[r]["requests"].value("outputs", json::array());
                const json& ob = rb[r]["requests"].value("outputs", json::array());
                for (size_t k = 0; k < std::max(oa.size(), ob.size()); ++k) {
                    if (k >= oa.size() || k >= ob.size()) {
                        missing.push_back(point + " run " + std::to_string(r + 1) + " request " +
                                          std::to_string(k + 1) + ": only one side has it");
                        continue;
                    }
                    ++compared;
                    const json& ta = oa[k];
                    const json& tb = ob[k];
                    if (ta == tb) {
                        ++identical;
                        continue;
                    }
                    size_t i = 0;
                    while (i < ta.size() && i < tb.size() && ta[i] == tb[i]) ++i;
                    const json pt = ra[r]["requests"].value("prompt_tokens", json::array());
                    differing.push_back({ { "point", point }, { "run", (int64_t)r + 1 }, { "request", (int64_t)k + 1 },
                                          { "prompt_tokens", k < pt.size() ? pt[k] : json(nullptr) },
                                          { "first_different_token", (int64_t)i },
                                          { "tokens", (int64_t)ta.size() }, { "tokens_baseline", (int64_t)tb.size() } });
                }
            }
        }
        for (const json& pb : b)
            if (!find(a, pb[sec.key])) missing.push_back(label(pb) + ": only in the baseline");
    }
    json j;
    j["compared"] = compared;
    j["identical"] = identical;
    j["differing"] = differing;
    j["missing"] = missing;
    j["acceptance"] = acceptance;
    j["acceptance_dropped"] = dropped;
    j["verdict"] = compared == 0           ? "nothing compared"
                 : !differing.empty()      ? "DIFFERS"
                 : !missing.empty()        ? "identical where both ran; points missing"
                                           : "identical";
    j["acceptance_verdict"] = dropped.empty() ? "no drop" : "DROPPED";
    *out = j;
    return RAD_OK;
}

/* ================================================================== workload */

std::string synthetic_document(int64_t chars, uint32_t salt) {
    static const char* const kPara[] = {
        "The survey covers the history of the measurement of time, from water clocks and candle "
        "marks to escapements, pendulums and the quartz oscillator. Each section names the "
        "instrument, the error it removed and the error it introduced in its place.",
        "Records from the observatory list the transit of each reference star, the clock reading "
        "at the transit, the correction applied and the observer. A missing reading is entered as "
        "missing; an interpolated one is marked, because a later reader cannot tell otherwise.",
        "The comparison table gives, for every pair of clocks, the drift over a week, the largest "
        "single-day deviation and the temperature at which it was observed, so that a drift caused "
        "by the room can be separated from one caused by the mechanism.",
        "In the appendix the same procedure is repeated with a different reference, and the two "
        "series are compared line by line. Where they disagree by more than the stated tolerance "
        "the entry is kept, flagged and discussed rather than averaged away.",
    };
    std::string s;
    s.reserve((size_t)chars + 512);
    uint32_t x = salt * 2654435761u + 12345u;
    for (int sec = 1; (int64_t)s.size() < chars; ++sec) {
        x = x * 1664525u + 1013904223u;
        char head[96];
        std::snprintf(head, sizeof head, "\n\nSection %d (series %u, run %u).\n", sec, salt, x % 100000u);
        s += head;
        s += kPara[(size_t)(sec + salt) % (sizeof kPara / sizeof kPara[0])];
    }
    s.resize((size_t)std::max<int64_t>(chars, 0));
    return s;
}

std::string decode_prompt(int stream) {
    static const char* const kTopics[] = {
        "the history of printing", "how bridges are inspected", "the water cycle of a river basin",
        "the design of a railway timetable", "how a compiler allocates registers",
        "the economics of a small harbour", "how glaciers record climate", "the logistics of a city library",
    };
    const size_t n = sizeof kTopics / sizeof kTopics[0];
    return std::string("Write a long, detailed essay about ") + kTopics[(size_t)stream % n] +
           ". Use several sections with headings, and keep writing until you are asked to stop.";
}

std::vector<CorpusEntry> builtin_corpus() {
    struct P { const char* cat; const char* text; };
    static const P kPrompts[] = {
        { "code", "Write a Python module that parses a CSV file of bank transactions, groups them by "
                  "month and category, and prints a summary table. Include type hints, docstrings and "
                  "unit tests using the standard library only." },
        { "code", "Implement a thread-safe LRU cache in C++17 with a fixed capacity, O(1) get and put, "
                  "and an eviction callback. Explain the locking strategy, then give the full code." },
        { "reasoning", "A train leaves city A at 08:00 at 80 km/h towards city B, 300 km away. Another "
                       "leaves B at 09:00 at 100 km/h towards A. Where and when do they meet? Then "
                       "generalise: derive the meeting point for arbitrary departure times and speeds, "
                       "and check the edge cases step by step." },
        { "reasoning", "Five people sit in a row. Anna is not at either end, Ben sits directly left of "
                       "Carla, Dora is not next to Ben, and Emil sits at the right end. List every "
                       "arrangement that satisfies all conditions and prove that there are no others." },
        { "prose", "Write a long short story about a lighthouse keeper on a remote island who receives a "
                   "letter that was posted forty years ago. Use vivid description and dialogue." },
        { "prose", "Write an essay on how public libraries have changed over the last century, with "
                   "sections on architecture, technology, funding and their role in communities." },
        { "json", "Return a JSON array of 25 fictional employees. Each object has: id (integer), name, "
                  "email, department (one of engineering, sales, support, finance), start_date "
                  "(YYYY-MM-DD), salary (integer), skills (array of strings) and manager_id (integer "
                  "or null). Output only the JSON." },
        { "json", "Produce a JSON Schema (draft 2020-12) for an online shop order: customer, shipping "
                  "address, line items with product id, quantity and price, payment details, status "
                  "history. Then give three example orders that validate against it, as JSON." },
        { "file_edit", "Here is a file:\n\n```python\nimport os\n\ndef load(path):\n    f = open(path)\n"
                       "    data = f.read()\n    return data\n\ndef save(path, data):\n    f = open(path, 'w')\n"
                       "    f.write(data)\n\ndef main():\n    d = load(os.environ['IN'])\n    save(os.environ['OUT'], "
                       "d.upper())\n\nmain()\n```\n\nRewrite it so files are closed with context managers, "
                       "errors are reported with clear messages, paths come from argparse instead of the "
                       "environment, and main only runs under `if __name__ == '__main__'`. Give the whole "
                       "new file, then a unified diff against the original." },
        { "file_edit", "Rename the function `calc` to `compute_total` everywhere in the following "
                       "JavaScript, add JSDoc comments to every function, and convert it to ES modules. "
                       "Output the complete updated file.\n\n```js\nconst tax = 0.2;\nfunction calc(items) "
                       "{\n  let t = 0;\n  for (const i of items) t += i.price * i.qty;\n  return t * (1 + tax);\n}"
                       "\nfunction fmt(v) { return v.toFixed(2) + ' EUR'; }\nmodule.exports = { calc, fmt };\n```" },
        { "summarization", nullptr },
        { "summarization", nullptr },
    };
    std::vector<CorpusEntry> out;
    uint32_t doc = 0;
    for (const P& p : kPrompts) {
        CorpusEntry e;
        e.category = p.cat;
        std::string text;
        if (p.text) {
            text = p.text;
        } else {
            /* A document to summarise: long enough that the answer is a summary and not a copy. */
            text = "Summarise the following report in about 300 words, then list its five most "
                   "important findings as bullet points.\n\n" + synthetic_document(6000, 1000u + doc++);
        }
        e.messages = json::array({ json{ { "role", "user" }, { "content", text } } });
        out.push_back(std::move(e));
    }
    return out;
}

int load_corpus(const std::string& path, std::vector<CorpusEntry>* out, std::string* why) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        *why = "cannot open " + path;
        return RAD_E_IO;
    }
    std::string text;
    char b[65536];
    for (size_t n; (n = std::fread(b, 1, sizeof b, f)) > 0;) text.append(b, n);
    std::fclose(f);
    std::vector<CorpusEntry> got;
    size_t line_no = 0;
    for (size_t pos = 0; pos <= text.size();) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        ++line_no;
        const size_t a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos || line[a] == '#') continue;
        const json j = json::parse(line, nullptr, /*allow_exceptions=*/false);
        auto bad = [&](const char* what) {
            *why = path + ":" + std::to_string(line_no) + ": " + what;
            return RAD_E_INVAL;
        };
        if (j.is_discarded() || !j.is_object()) return bad("not a JSON object");
        if (!j.contains("category") || !j["category"].is_string() || j["category"].get<std::string>().empty())
            return bad("needs a non-empty string \"category\"");
        CorpusEntry e;
        e.category = j["category"].get<std::string>();
        if (j.contains("messages")) {
            if (!j["messages"].is_array() || j["messages"].empty()) return bad("\"messages\" must be a non-empty array");
            for (const json& m : j["messages"])
                if (!m.is_object() || !m.contains("role") || !m.contains("content"))
                    return bad("every message needs \"role\" and \"content\"");
            e.messages = j["messages"];
        } else if (j.contains("prompt") && j["prompt"].is_string()) {
            e.messages = json::array({ json{ { "role", "user" }, { "content", j["prompt"] } } });
        } else {
            return bad("needs \"messages\" or a string \"prompt\"");
        }
        if (j.contains("max_tokens")) {
            if (!j["max_tokens"].is_number_integer() || j["max_tokens"].get<int64_t>() < 1 ||
                j["max_tokens"].get<int64_t>() > (1 << 20))
                return bad("\"max_tokens\" must be an integer from 1 to 1048576");
            e.max_tokens = j["max_tokens"].get<int>();
        }
        got.push_back(std::move(e));
    }
    if (got.empty()) {
        *why = path + ": no prompts in it";
        return RAD_E_INVAL;
    }
    *out = std::move(got);
    return RAD_OK;
}

json salt_messages(const json& messages, const std::string& tag) {
    json m = messages;
    if (!m.is_array() || m.empty()) return m;
    json& c = m[0]["content"];
    const std::string head = "[" + tag + "]\n";
    if (c.is_string()) {
        c = head + c.get<std::string>();
    } else if (c.is_array()) {
        c.insert(c.begin(), json{ { "type", "text" }, { "text", head } });
    } else {
        c = head;
    }
    return m;
}

/* ================================================================== report */

namespace {


/* "12.3 (11.9–12.8)": the median of a summarised number with its range, or the median alone. */
std::string cell(const json& summary, const char* key) {
    if (!summary.contains(key) || summary[key].is_null()) return "–";
    const json& v = summary[key];
    std::string out = num(v["median"]);
    if (v["min"] != v["max"]) out += " (" + num(v["min"]) + "–" + num(v["max"]) + ")";
    return out;
}

}  /* namespace */

std::vector<std::string> env_allowlist(char* const* env) {
    /* A fixed list of the engine's configuration knobs, not a prefix: a prefix takes every suffix
     * with it, so `RADIANCE_API_KEY_<anything>` would be reported by name. Nothing outside the
     * list is, and no value is ever read. */
    static const char* const kNames[] = {
        "HIP_VISIBLE_DEVICES",    "OMP_NUM_THREADS",       "OMP_WAIT_POLICY",
        "RADIANCE_CALIB_DIR",     "RADIANCE_CALIB_TOKENS", "RADIANCE_DRAFT_BLOCK_HOST",
        "RADIANCE_HEAT_RESIDENCY", "RADIANCE_HOME",        "RADIANCE_HOST_DEVICES",
        "RADIANCE_HOST_VRAM_MIB", "RADIANCE_KERNELS",      "RADIANCE_LOG",
        "RADIANCE_LOG_STEPS",     "RADIANCE_MTP_COUPLED",  "RADIANCE_PROFILE_EVERY",
        "RADIANCE_TP",            "RADIANCE_TUNE",         "ROCR_VISIBLE_DEVICES",
    };
    std::vector<std::string> out;
    for (char* const* e = env; e && *e; ++e) {
        const char* eq = std::strchr(*e, '=');
        const size_t n = eq ? (size_t)(eq - *e) : std::strlen(*e);
        for (const char* name : kNames)
            if (std::strlen(name) == n && std::strncmp(*e, name, n) == 0) {
                out.emplace_back(*e, n);
                break;
            }
    }
    std::sort(out.begin(), out.end());
    return out;
}

json engine_report(const json& server_info) {
    /* Only the model's identity and the cards: /server_info's `container` is the model's path and
     * its control vector carries a file, and neither belongs in a report pasted into an issue. The
     * card's own handle is left out too: it names a resource in this process, not a card. */
    json out = json::object();
    if (!server_info.is_object()) return out;
    for (const char* k : { "arch", "quant", "n_layers", "n_embd", "n_vocab", "n_ctx_train" })
        if (server_info.contains(k)) out[k] = server_info[k];
    if (server_info.contains("devices") && server_info["devices"].is_array()) {
        json kept = json::array();
        for (const json& d : server_info["devices"]) {
            json k = json::object();
            for (const char* f : { "rank", "name", "arch", "vram_bytes" })
                if (d.is_object() && d.contains(f)) k[f] = d[f];
            kept.push_back(std::move(k));
        }
        out["devices"] = std::move(kept);
    }
    return out;
}

std::string render_markdown(const json& rep) {
    std::string m = "# Diagnosis\n\n";
    if (rep.contains("engine")) {
        const json& e = rep["engine"];
        m += "Model `" + e.value("arch", std::string("?")) + "` (" +
             e.value("quant", std::string("quantisation not stated")) + "), TP " +
             num(e.value("tp", json())) + ", backend `" + e.value("backend", std::string()) +
             "`, " + e.value("plan", std::string("plan not stated")) + ".\n\n";
    }
    if (rep.contains("workload")) m += "Workload: " + rep["workload"].value("description", std::string()) + "\n\n";
    if (rep.contains("verdict")) m += "**Verdict: " + rep["verdict"].get<std::string>() + "**\n\n";
    if (rep.contains("warnings") && !rep["warnings"].empty()) {
        m += "**Warnings**\n\n";
        for (const auto& w : rep["warnings"]) m += "- " + w.get<std::string>() + "\n";
        m += "\n";
    }
    m += "Each cell is the median over the runs of a point, with the smallest and largest run in "
         "brackets where they differ. A dash is a number the run could not measure, not a zero.\n\n";

    if (rep.contains("categories") && !rep["categories"].empty()) {
        m += "## Decode by category (1 stream)\n\n"
             "| Category | Runs | Stream tok/s | ms/step | Tok/step | Draft acceptance | TTFT ms | 1% low tok/s | Ended on EOS |\n"
             "|---|---|---|---|---|---|---|---|---|\n";
        for (const auto& c : rep["categories"]) {
            const json s = c.value("summary", json::object());
            m += "| " + c.value("category", std::string("?")) + " | " + num(s.value("runs", json())) + " | " +
                 cell(s, "requests.stream_tok_s.p50") + " | " + cell(s, "counters.ms_per_decode_step") + " | " +
                 cell(s, "steps.decode_tok_per_step") + " | " + cell(s, "counters.draft_acceptance") + " | " +
                 cell(s, "requests.ttft_ms.p50") + " | " + cell(s, "steps.tok_s_1pct_low") + " | " +
                 num(c.value("stopped_on_eos", json())) + " of " + num(c.value("requests", json())) + " |\n";
        }
        m += "\nDraft acceptance by position (conditional, pooled over the runs):\n\n";
        for (const auto& c : rep["categories"]) {
            const json pos = c.value("draft_by_position", json::array());
            if (pos.empty()) continue;
            m += "- " + c.value("category", std::string("?")) + ":";
            for (const auto& p : pos)
                m += " " + num(p["position"]) + ": " + num(p["accepted"]) + " (" + num(p["reached"]) + ");";
            m += "\n";
        }
        m += "\n";
    }
    if (rep.contains("concurrency") && !rep["concurrency"].empty()) {
        m += "## Concurrency sweep (streams submitted together)\n\n"
             "| Streams | Runs | Aggregate tok/s | Per stream tok/s | ms/step | Tok/step | Draft acceptance | "
             "Emission p50 ms | Emission p99/p50 | 1% low tok/s |\n|---|---|---|---|---|---|---|---|---|---|\n";
        for (const auto& c : rep["concurrency"]) {
            const json s = c.value("summary", json::object());
            m += "| " + num(c.value("streams", json())) + " | " + num(s.value("runs", json())) + " | " + cell(s, "requests.aggregate_tok_s") +
                 " | " + cell(s, "requests.stream_tok_s.p50") + " | " + cell(s, "counters.ms_per_decode_step") +
                 " | " + cell(s, "steps.decode_tok_per_step") + " | " + cell(s, "counters.draft_acceptance") +
                 " | " + cell(s, "steps.emission_interval_ms.p50") + " | " +
                 cell(s, "steps.jitter_emission_p99_over_p50") + " | " + cell(s, "steps.tok_s_1pct_low") + " |\n";
        }
        m += "\n";
    }
    if (rep.contains("prefill") && !rep["prefill"].empty()) {
        m += "## Prefill sweep (1 stream, then decode at that depth)\n\n"
             "| Prompt tokens | Runs | TTFT ms | Prefill tok/s | Chunk ms p50 | Decode ms/step at depth | "
             "Decode tok/s at depth |\n|---|---|---|---|---|---|---|\n";
        for (const auto& c : rep["prefill"]) {
            const json s = c.value("summary", json::object());
            m += "| " + num(c.value("prompt_tokens", json())) + " | " + num(s.value("runs", json())) + " | " + cell(s, "requests.ttft_ms.p50") +
                 " | " + cell(s, "requests.prefill_tok_s.p50") + " | " + cell(s, "steps.prefill_chunk_ms.p50") +
                 " | " + cell(s, "counters.ms_per_decode_step") + " | " + cell(s, "requests.stream_tok_s.p50") + " |\n";
        }
        m += "\n";
    }
    if (rep.contains("interference") && rep["interference"].is_object()) {
        const json& i = rep["interference"];
        const json s = i.value("summary", json::object());
        m += "## A prompt arriving into running streams\n\n" + num(i.value("streams", json())) + " streams decoding, then one prompt of " +
             num(i.value("prompt_tokens", json())) + " tokens.\n\n| Measure | Value |\n|---|---|\n";
        m += "| TTFT of the arriving prompt | " + cell(s, "arrival_ttft_ms") + " ms |\n";
        m += "| Its TTFT on an idle engine (prefill sweep) | " + num(i.value("idle_ttft_ms", json())) + " ms |\n";
        m += "| Typical emission gap before it | " + cell(s, "interference.typical_gap_ms") + " ms |\n";
        m += "| Longest emission gap while it prefilled | " + cell(s, "interference.longest_gap_ms") + " ms |\n";
        m += "| Stall ratio (longest / typical) | " + cell(s, "interference.stall_ratio") + " |\n\n";
    }
    if (rep.contains("parity") && rep["parity"].is_object()) {
        const json& p = rep["parity"];
        m += "## Parity against " + p.value("baseline", std::string("the baseline")) + "\n\n**" +
             p.value("verdict", std::string()) + "**: " + num(p.value("identical", json())) + " of " +
             num(p.value("compared", json())) + " requests produced identical tokens.\n\n";
        const json& d = p.value("differing", json::array());
        if (!d.empty()) {
            m += "| Point | Run | Request | Prompt tokens | First different token | Tokens (now / baseline) |\n|---|---|---|---|---|---|\n";
            for (const auto& x : d)
                m += "| " + x["point"].get<std::string>() + " | " + num(x["run"]) + " | " + num(x["request"]) + " | " +
                     num(x.value("prompt_tokens", json())) + " | " + num(x["first_different_token"]) + " | " +
                     num(x["tokens"]) + " / " + num(x["tokens_baseline"]) + " |\n";
            m += "\n";
        }
        const json acc = p.value("acceptance", json::array());
        if (!acc.empty()) {
            m += "Draft acceptance against the baseline (**" + p.value("acceptance_verdict", std::string()) +
                 "**): overall by a paired t test over the requests, positions by the pooled z corrected for "
                 "the requests' dispersion; a drop is as unlikely as z below -3:\n\n"
                 "| Point | | Now | Baseline | Drafts now / baseline | Test |\n|---|---|---|---|---|---|\n";
            for (const auto& x : acc) {
                std::string test;
                if (x.contains("t")) {
                    char b[64];
                    std::snprintf(b, sizeof b, "t %s, p %.2g, %lld requests", num(x["t"]).c_str(),
                                  x.value("p", 1.0), (long long)x.value("requests", (int64_t)0));
                    test = b;
                } else {
                    test = "z " + num(x.value("z", json()));
                    if (x.contains("dispersion")) test += ", dispersion " + num(x["dispersion"]);
                }
                m += "| " + x.value("point", std::string()) + " | " + x.value("what", std::string()) + " | " +
                     num(x.value("now", json())) + " | " + num(x.value("baseline", json())) + " | " +
                     num(x.value("n_now", json())) + " / " + num(x.value("n_baseline", json())) + " | " + test + " |\n";
            }
            m += "\n";
        }
        for (const auto& x : p.value("missing", json::array())) m += "- missing: " + x.get<std::string>() + "\n";
        if (!p.value("missing", json::array()).empty()) m += "\n";
    }
    if (rep.contains("not_measured")) {
        m += "## Not measured\n\n";
        for (const auto& w : rep["not_measured"]) m += "- " + w.get<std::string>() + "\n";
    }
    return m;
}

}  /* namespace diag */
}  /* namespace rad */
