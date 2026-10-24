/* diagnose.h -- the --diagnose mode's arithmetic and its report.
 *
 * `radiance --diagnose DIR` starts the engine as it would to serve, then runs a fixed workload on
 * itself instead of listening (core/engine_diagnose.cpp) and writes DIR/diagnose.json and
 * DIR/diagnose.md. This file is the part with no device and no scheduler in it: the step record
 * the scheduler keeps, the statistics a phase reduces to, and the report. The driver feeds it;
 * tests/diagnose_test.cpp holds it to hand-computed numbers.
 *
 * WHY THE ENGINE MEASURES ITSELF. The numbers an operator wants -- milliseconds a decode step,
 * tokens a step, how often a draft is taken, how long a prefill takes, how uneven the steps are --
 * are all known inside the engine exactly, and outside it only through a client's clock and a
 * stream that delivers a speculative step's tokens in one burst. A client measuring inter-token
 * latency on a speculating engine is measuring the burst pattern; the engine knows the steps.
 *
 * WHAT A PHASE REPORTS, AND FROM WHERE.
 *   - SchedMetrics deltas taken while the engine is idle at the phase's edges: exact counts of
 *     steps by shape, tokens, drafts proposed and accepted, per draft position.
 *   - One StepRecord per step, written by the scheduler (Scheduler::step_finished): what the step
 *     carried and how long it took. From these: the distribution of decode step times, the
 *     interval between consecutive emissions (what a stream sees between token bursts), and the
 *     slowest steps against typical ones.
 *   - Each request's sink: admitted, first token, last token, tokens -- TTFT and stream rate.
 * A quantity the run cannot measure is reported as absent with the reason, never as 0.
 */
#pragma once

#include "sched/metrics.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rad {
namespace diag {

using json = nlohmann::ordered_json;

/* ONE ENGINE STEP, as the scheduler commits it. Plain data, fixed size: the scheduler writes one
 * per step under its own lock into a ring sized at init, so recording costs a store and no
 * allocation on the step path. */
struct StepRecord {
    int64_t step = 0;          /* SchedMetrics::steps after this step: a run-wide ordinal */
    int32_t phase = 0;         /* RAD_PHASE_PREFILL / DECODE / MIXED of the step's plan */
    int32_t n_seq = 0;         /* sequences the step carried */
    int32_t prefill_tok = 0;   /* prompt tokens it computed */
    int32_t decode_tok = 0;    /* tokens it committed to decoding sequences, accepted drafts included */
    int64_t ns = 0;            /* inside the step, the clock SchedMetrics::step_ns charges */
    int64_t end_ns = 0;        /* steady clock when it was committed: emissions are spaced by these */
};

/* A FIXED-CAPACITY RING OF STEP RECORDS with a run-wide sequence number. The writer and every
 * reader hold the scheduler's lock, so it needs no atomics of its own. A reader asks for what
 * came after a sequence number it saved; if the ring has lapped since, the records it lost are
 * counted, not invented. */
class StepRing {
public:
    explicit StepRing(size_t capacity = 0) { reset(capacity); }
    void reset(size_t capacity);

    void push(const StepRecord& r) noexcept;
    uint64_t count() const { return n_; }           /* records ever pushed */
    size_t   capacity() const { return buf_.size(); }

    /* Appends every record with sequence number >= `from` still in the ring to `out`; returns how
     * many after `from` were overwritten before they could be read. */
    uint64_t copy_since(uint64_t from, std::vector<StepRecord>* out) const;

private:
    std::vector<StepRecord> buf_;
    uint64_t n_ = 0;
};

/* ------------------------------------------------------------------ statistics */

/* Quantiles by nearest rank over a copy of the samples. `n == 0` means there were none, and every
 * other field is then meaningless; the JSON form says null rather than 0. */
struct Quantiles {
    int64_t n = 0;
    double min = 0, p50 = 0, p90 = 0, p99 = 0, max = 0, mean = 0;
};
Quantiles quantiles(std::vector<double> v);
json to_json(const Quantiles& q);

/* What a request's sink recorded, on one steady-clock axis (Sink::t_start_ns + its offsets). -1
 * where the sink never saw the event. */
struct RequestTiming {
    int64_t start_ns = -1;        /* the sink's construction: before submission */
    int64_t admitted_ns = -1;     /* the scheduler first gave it a step */
    int64_t first_ns = -1;        /* first token */
    int64_t last_ns = -1;         /* last token */
    int64_t tokens = 0;           /* produced */
    int32_t prompt = 0;           /* prompt tokens */
    int32_t cached = 0;           /* of those, served from the prefix cache */
    int64_t drafted = 0;
    int64_t accepted = 0;
    bool    ok = false;           /* finished on length or stop, not on an error or a cancel */
    bool    stopped = false;      /* finished on the model's own end of sequence, not on length */
    std::vector<int32_t> output;  /* the tokens it produced, for the parity comparison */
};

enum class PhaseKind { Decode, Prefill };

struct PhaseInput {
    std::string name;
    PhaseKind   kind = PhaseKind::Decode;
    int         streams = 1;
    int64_t     wall_ns = 0;
    SchedMetrics before, after;
    std::vector<StepRecord>    steps;          /* the steps the phase ran, in order */
    uint64_t                   steps_lost = 0;  /* records the ring overwrote before they were read */
    std::vector<RequestTiming> requests;
};

/* The phase's section of the report. Every derived number names what it was derived from. */
json phase_report(const PhaseInput& in);

/* The slowest steps of one shape against the typical ones: the slowest `max(3, 1%)` decode steps
 * with what they carried, and the median step beside them, so a reader can tell a step that was
 * slow because it carried more from one that was slow carrying the same. */
json slow_steps(const std::vector<StepRecord>& steps, int phase);

/* DRAFT ACCEPTANCE BY POSITION, POOLED OVER A POINT'S RUNS: at each position the verifies that
 * reached it and the share of those that took it, summed over the runs rather than a median of
 * per-run rates, because a run reaches the deep positions only a few times. This is the curve an
 * adaptive draft depth is tuned against. */
json pool_draft_positions(const std::vector<json>& runs);

/* THE RUNS OF ONE MEASUREMENT POINT, reduced to what a reader compares: for each headline number
 * of phase_report, the median over the runs and the smallest and largest run. A run that has no
 * value for a number does not count toward it; a number no run had is null. */
json summarize_runs(const std::vector<json>& runs);
/* The same over the numbers at the given JSON pointers, keyed by the pointer with '/' as '.'. */
json summarize_paths(const std::vector<json>& runs, const std::vector<std::string>& paths);

/* A DECODE STALL: what a prompt arriving into running streams costs them. `from`..`to` is the
 * window from the new request's admission to its first token; the report gives the longest gap
 * between two emissions that overlaps it, against the median gap of the emissions before it. */
json interference_report(const std::vector<StepRecord>& steps, int64_t from_ns, int64_t to_ns);

/* ------------------------------------------------------------------ parity */

/* WHAT TWO RUNS OF THE SAME WORKLOAD PRODUCED, COMPARED TOKEN FOR TOKEN. With the same prompts and
 * the same seed per request, a build whose logits are bit-identical samples identical tokens, at
 * any temperature -- so every request of the categories, the concurrency sweep and the prefill
 * sweep is a parity check of the paths it ran: long prefills, attention at depth, batched decode.
 * The interference point is left out: when its running streams are cancelled depends on timing.
 *
 * `now` and `base` are whole reports. Points are matched by their key (category, streams,
 * requested tokens), runs and requests by position. The result counts identical and differing
 * requests and names, for each difference, the first token where they part; a point or run that
 * only one side has is listed as missing rather than counted as either. Two reports of different
 * workloads (workload.fingerprint) are refused with RAD_E_INVAL: their prompts differ. */
/* Beside the tokens it compares each matched point's draft acceptance: identical for
 * bit-identical builds, and the first number to move when a build computes less accurately. The
 * overall rate by a one-sided paired t test over the requests (same prompt and seed on both
 * sides), the positions by the pooled two-proportion z divided by the square root of the
 * dispersion the requests show; either as unlikely as z below -3 is a drop, listed in
 * `acceptance_dropped`, and makes `acceptance_verdict` "DROPPED". */
int compare_outputs(const json& now, const json& base, json* out, std::string* why);

/* The part of a report's workload that decides which prompts and seeds it ran: equal fingerprints
 * mean comparable outputs. */
std::string workload_fingerprint(const json& workload_params);

/* ------------------------------------------------------------------ workload */

/* ONE PROMPT OF THE CATEGORY WORKLOAD: OpenAI-style chat messages and the category they are
 * reported under. `max_tokens` 0 means the run's --diagnose-tokens. */
struct CorpusEntry {
    std::string category;
    json        messages;
    int         max_tokens = 0;
};

/* A small built-in corpus, a few prompts in each of the categories a chat deployment typically
 * serves (code, reasoning, prose, json, file_edit, summarization), so a run without a corpus file
 * still reports by category. Speculative decoding accepts very differently across these, which is
 * why one decode number is not enough. */
std::vector<CorpusEntry> builtin_corpus();

/* Reads a corpus file: one JSON object a line, `{"category": "...", "messages": [...]}` or
 * `{"category": "...", "prompt": "..."}`, optionally with "max_tokens". Blank lines and lines
 * starting with '#' are skipped. Any other malformed line refuses the file, naming the line. */
int load_corpus(const std::string& path, std::vector<CorpusEntry>* out, std::string* why);

/* THE SAME MESSAGES MADE UNIQUE FROM THEIR FIRST CHARACTER. `tag` is put at the head of the first
 * message's content, so no two requests of a run share a prefix longer than the chat template's
 * own opening -- less than one cache block -- and the prefix cache cannot serve one from another.
 * The tag is deterministic (a request counter), so two builds see the same prompts; a hit from a
 * cache tier that outlived the process is caught by the driver's check of RequestTiming::cached. */
json salt_messages(const json& messages, const std::string& tag);


/* A deterministic document of about `chars` characters: numbered sections of fixed prose, every
 * section different in its numbers, so a prefix cache cannot serve one run's text to the next
 * when `salt` differs. Language-neutral enough to tokenise under any vocabulary. */
std::string synthetic_document(int64_t chars, uint32_t salt);

/* The decode prompts, one per stream: fixed instructions that ask for long prose, distinct per
 * stream so eight streams are eight conversations and not one prompt eight times. */
std::string decode_prompt(int stream);

/* ------------------------------------------------------------------ report */

/* The environment the workload ran under: the NAMES of the variables in `env` (environ-shaped,
 * null-terminated) that are on a fixed list of the engine's configuration knobs, sorted. Names
 * only, never values: a value may be a path, a host name or a token, and the report is pasted into
 * issues and committed beside a baseline (diag/diag.h's env_names states the same rule for the
 * startup report).
 *
 * A fixed list, not a prefix: a prefix admits every suffix, so `RADIANCE_API_KEY_<anything>` is a
 * name the report would print. A name the engine does not read as configuration is not reported. */
std::vector<std::string> env_allowlist(char* const* env);

/* The engine section of a report, built from the engine's /server_info object: the model's
 * architecture, quantisation and shapes, and the cards it ran on. The object itself is not copied,
 * because its `container` is the model's path and its control vector carries a file, and the report
 * is pasted into issues: a path names a person's home directory and nothing a reader needs. */
json engine_report(const json& server_info);

/* The Markdown view of a report: one table each for the categories, the concurrency sweep, the
 * prefill sweep and the interference point, from their `summary` sections plus the driver's
 * header. It reads only the JSON, so it can be regenerated from a saved diagnose.json; the
 * per-run detail stays in the JSON. It carries no path either: the header names the model by its
 * architecture and quantisation (engine_report), never by where it is stored. */
std::string render_markdown(const json& report);

}  /* namespace diag */
}  /* namespace rad */
