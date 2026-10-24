/* diagnose.h -- what the engine did to itself: the arithmetic and the report behind
 * `radiance --diagnose DIR`.
 *
 * The startup report (core/diag/diag.h) answers what the engine was configured to do. This answers
 * what it did: the engine starts as it would to serve, runs a fixed serving workload through its
 * own scheduler instead of listening, and writes DIR/diagnose.json and DIR/diagnose.md. The two
 * are separate on purpose -- one reads /proc and the option table, the other reads steps -- and
 * the report here names the startup report's findings it depends on (the backend, the drafter)
 * rather than repeating them.
 *
 * Why the engine measures itself and a client does not measure it: a step, not a token, is the
 * unit. A speculating engine emits a draft's worth of tokens and then waits, so a client's arrival
 * intervals carry that burst pattern rather than the step's, and every number it takes is mixed
 * with the socket, the JSON encoder and the detokeniser. The scheduler knows what each step
 * carried and how long it took, exactly (core/sched/steplog.h), so the step time, the tokens per
 * step and the interval between bursts are read from there.
 *
 * What the report answers:
 *   Step time      the distribution over a phase's steps, not a mean over a decaying window:
 *                  p50/p90/p99 by step kind, the interval between the steps that emitted tokens and
 *                  how much of it the engine spent waiting, and the tokens each step emitted.
 *   Acceptance     draft acceptance pooled and by draft position, per category, and against an
 *                  earlier report, so a change shows up as a drop at a position rather than as a
 *                  slower build.
 *   Parity         every token the workload generated, in the report. `--diagnose-baseline`
 *                  compares them with an earlier report and fails on any difference: a build that
 *                  changed what the model says is a different build, whatever its throughput did.
 *   Points         categories, concurrency, prefill by length, and interference: one prompt
 *                  arriving while streams are decoding, whose cost is measured as the change in
 *                  the burst interval of the decoders that were already running.
 *
 * Nothing is served from the prefix cache: each prompt carries a salt that changes with every
 * point, and the cache is dropped before every point. A point the cache did serve is reported as
 * not a measurement, because its prefill did not happen.
 *
 * A quantity the run could not measure is reported as absent, with the reason it is absent, never
 * as 0. A zero step time, a 0% acceptance and an empty distribution all look like measurements of
 * something fast or empty, and a report read beside a diff cannot tell them apart.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace diagnose {

/* ------------------------------------------------------------------ the workload
 * Fixed, and hashed (workload_id): a baseline of a different workload is not a baseline, and the
 * comparison says so rather than comparing two different runs' numbers.
 *
 * A prompt is fixed text, optionally grown to a token count so that prefill can be measured at a
 * length rather than at whatever the text happens to tokenise to. The growth is a count of one
 * token id, so a length is exact and the tokens are real ones. */
struct Prompt {
    const char* name;       /* "prose", "code", ...; also the category label's subject */
    const char* category;   /* what the point is about: prose, code, long-context, prefill, ... */
    const char* text;
    int tokens;             /* 0: however long the text tokenises to; >0: grown to exactly this */
};

/* One thing the workload measures: a number of sequences at once, on prompts [first, first + n),
 * each generating `max_tokens`. */
struct Point {
    const char* name;
    const char* category;
    int concurrency;        /* sequences the point keeps in flight */
    int n_prompts;          /* prompts it cycles through, from the table below */
    int first_prompt;
    int repeat;             /* each prompt run this many times, each with its own salt */
    int max_tokens;         /* per request */
    int warmup;             /* 1: bring the batch up, then measure a second pass */
    /* Interference: >0 holds the point's last request back and admits it once the streams already
     * running have each emitted this many tokens. It is built from `interfere_prompt`, and its
     * cost is measured as the change in those streams' burst interval -- not as a prefill
     * measurement wearing a decode point's clothes. */
    int interfere_after_tokens;
    int interfere_prompt;
};

/* The table. Every run of every build runs this, in this order. */
const std::vector<Prompt>& prompts();
const std::vector<Point>& points();
/* fnv1a over both tables: the workload's identity, recorded in the report and checked against a
 * baseline's. */
uint64_t workload_id();

/* ------------------------------------------------------------------ measured and absent */
/* A number, or the reason there is none. Rendered as {"value": x} or {"absent": "why"}: a reader
 * that wants a number and gets "absent" has to say why it has none either. */
struct Measure {
    bool present = false;
    double value = 0.0;
    std::string absent;
    static Measure of(double v) { Measure m; m.present = true; m.value = v; return m; }
    static Measure missing(std::string why) {
        Measure m; m.absent = std::move(why); return m;
    }
    /* A ratio over two counts: missing when the denominator is 0, which is "nothing of that kind
     * happened", not "it never worked". */
    static Measure ratio(int64_t num, int64_t den, const std::string& why_if_none);
};

/* The JSON text of one Measure, for a caller building text of its own. */
std::string measure_json(const Measure& m);

/* ------------------------------------------------------------------ distributions */
/* The shape of a set of samples. `percentile` is the standard interpolated order statistic: at
 * position p*(n-1) of the sorted samples, between the two neighbours when p*(n-1) is not an
 * integer. Every sample counts, unweighted, so a phase's slow tail is in p99 and not averaged
 * away. */
struct Dist {
    int64_t count = 0;
    Measure mean, p50, p90, p99, max;
    /* Why there is no distribution at all: "no decode step in this point", not an empty array a
     * reader has to guess about. */
    std::string absent;
};

/* Summarises `s`. `nothing` is the reason reported when `s` is empty. `s` is sorted in place. */
Dist summarize(std::vector<double>& s, const std::string& nothing);

/* The percentile of `s` alone, for a caller that wants one number; interpolated as above. Empty
 * `s` is absent. */
Measure percentile(std::vector<double> s, double p);
Measure mean_of(const std::vector<double>& s);

/* ------------------------------------------------------------------ acceptance */
/* Draft acceptance: `reached[j]` verifies evaluated draft position j (all earlier ones were
 * accepted), `hits[j]` of them took it. The conditional acceptance at j is hits/reached, and is
 * absent where nothing reached that deep -- a drafter that never got that far is not a drafter
 * that failed there. */
struct Acceptance {
    int64_t drafted = 0, accepted = 0;
    std::vector<int64_t> reached, hits;
    Measure pooled() const { return Measure::ratio(accepted, drafted, "no draft token was verified"); }
    Measure at(size_t j) const;
    /* The deepest position anything reached, -1 for none. */
    int deepest() const;
    bool empty() const { return drafted == 0 && reached.empty(); }
};

/* ------------------------------------------------------------------ the report */
/* What the report says about itself, and about the baseline it was compared with. The driver
 * collects these before rendering, so the markdown and the JSON carry the same list a reader of
 * the log sees. */
struct Finding {
    enum Level { Note = 0, Warn = 1, Fail = 2 };
    Level level = Note;
    std::string id;      /* stable: "token-parity", "acceptance-drop" */
    std::string what;    /* what differs, with the numbers */
};

/* One request of the workload: its tokens are the point of the whole exercise. */
struct RequestRecord {
    int64_t id = 0;
    std::string point, category, prompt;
    int64_t prompt_tokens = 0, cached_tokens = 0;
    std::vector<int32_t> tokens;
    int64_t drafted = 0, accepted = 0;
    /* Seconds from the request's own submission (the sink's clock), -1 when unmeasured. */
    double first_token_s = -1, last_token_s = -1;
    std::string finish;         /* the sink's reason, as a word */
    bool served_from_cache = false;
};

/* One point, measured. The step-derived numbers come from the records the scheduler wrote while
 * only this point was running; the counters are that point's share of the cumulative ones. */
struct PointReport {
    std::string name, category;
    int concurrency = 0;

    /* A point that did not measure what it is for: the cache served it, the deployment cannot hold
     * it, its requests failed. Reported, never quietly skipped. */
    bool measured = true;
    std::string not_measured;
    std::vector<std::string> absent;      /* "decode step time: no decode step ran" */

    double wall_s = 0;
    int64_t requests = 0, steps = 0;
    int64_t prompt_tokens = 0, cached_tokens = 0, out_tokens = 0;
    /* The point's `concurrency` is its limit on admissions; `max_live` is the most streams it had
     * generating at once. An interference point exceeds its own limit by the arriving prompt, on
     * purpose, so a step's token count is bounded by the second and not the first. */
    int64_t max_live = 0;
    int64_t prefill_tokens = 0, decode_tokens = 0;
    /* The scheduler's own counters: how it phased its steps, how many it issued ahead, how often
     * it had to recompute one. */
    int64_t steps_prefill = 0, steps_mixed = 0, steps_decode = 0;
    int64_t steps_ahead = 0, preemptions = 0, chunks = 0;
    /* And the same steps counted from the records, by what each one carried rather than by the
     * phase the engine was in. The two differ by one step at the end of a prefill, which completes
     * a chunk and emits a token at once: the counters phase it as decode, the rows say prefill.
     * The distributions below are made of these, so a reader comparing a count with a distribution
     * is comparing the same set of steps. */
    int64_t shape_prefill = 0, shape_mixed = 0, shape_decode = 0, shape_draft = 0;
    /* Interference points only: the interval between emitted tokens of the streams that were
     * already running, before and after the prompt arrived. */
    Dist interval_before_arrival_us, interval_after_arrival_us;
    /* Tokens per step whose shape the point measured: decode steps, and their mixed variants. */
    Dist step_us_decode, step_us_mixed, step_us_prefill, step_us_all;
    /* Tokens per step, by shape, so a change in one is not read as a change in another: the tokens
     * each shape put on the wire (`tokens_per_step` counts every step that emitted any), and the
     * rows each step's forward pass computed, which is the prefill points' quantity. */
    Dist tokens_per_step, tok_per_step_prefill, tok_per_step_mixed, tok_per_step_decode;
    Dist rows_per_step;
    /* Two readings of the same series of steps, both the engine's own clock, neither what a client
     * sees (the socket, the detokeniser and the scheduler's own queueing are not in them):
     * `token_interval_us` is from one step that emitted tokens to the end of the next one that did,
     * which is the interval a client's tokens arrive inside; `token_gap_us` is the part of that the
     * engine *waited* -- the end of one such step to the start of the next -- so a large interval
     * with a small gap is a long step, and a large gap is a starved scheduler. A point that emitted
     * tokens from one step only has neither. */
    Dist token_interval_us, token_gap_us;

    Acceptance acceptance;
    std::vector<RequestRecord> reqs;

    /* A step log that wrapped: the distributions cover what survived, and the report says how
     * much did not. */
    int64_t step_records_dropped = 0;
};

/* The whole run. */
struct Report {
    /* What ran, so two reports can be told apart: the engine's version, the model, the backend,
     * the drafter, the flags that change what a step costs. Rendered as the report's "engine". */
    std::vector<std::pair<std::string, std::string>> engine;
    uint64_t workload = 0;
    int64_t salt = 0;
    std::vector<PointReport> points;
    /* Quantities the run as a whole could not measure. */
    std::vector<std::pair<std::string, std::string>> absent;
    /* review() and the baseline comparison, filled by the driver before rendering. */
    std::vector<Finding> findings;
};

/* DIR/diagnose.json. The report's own schema version, so a baseline from an older layout is
 * refused rather than misread. */
constexpr int kSchemaVersion = 3;
std::string report_json(const Report& r);
/* DIR/diagnose.md: one table per family of numbers, in the order a reader asks them. */
std::string report_markdown(const Report& r);

/* ------------------------------------------------------------------ against an earlier report */
/* Token parity: every request the baseline holds must appear here with exactly the same tokens,
 * and every request here must be in the baseline. Any difference is a Fail, listing the first few
 * with their point and prompt, because a build that says different words is a different build no
 * matter what its step times did. */
void compare_tokens(const std::string& base, const std::string& now,
                    std::vector<Finding>* out, int show = 5);

/* The drop check: pooled acceptance and acceptance at each position, per point, against the
 * baseline. `drop` is the relative fall that counts (0.1 is a tenth of what it was): acceptance is
 * text-dependent, so a small move is the workload, and the report keeps the numbers either way. */
void compare_acceptance(const std::string& base, const std::string& now, double drop,
                        std::vector<Finding>* out);

/* What the report says about itself, without a baseline: a point that was not a measurement, a
 * step log that dropped records, a workload that came back empty, cache hits in a run that is
 * supposed to have none. */
void review(const std::string& now, std::vector<Finding>* out);

/* False with the reason when `text` is not a report this file can read: a caller says "the
 * baseline does not parse" rather than throwing across the driver. */
bool parse_ok(const std::string& text, std::string* err);

/* A workload's identity, and a baseline's, for the "different workload" refusal. */
bool same_workload(const std::string& base, const std::string& now, std::string* why);

}  /* namespace diagnose */
}  /* namespace rad */
