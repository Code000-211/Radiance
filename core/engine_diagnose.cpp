/* engine_diagnose.cpp -- `radiance --diagnose DIR`: the engine runs a fixed serving workload
 * through itself instead of listening, and writes DIR/diagnose.json and DIR/diagnose.md.
 *
 * The shape is core/engine_kld.cpp's -- that mode also replaces the HTTP server with a driver that
 * builds Requests, holds Sinks of its own, and waits. What differs is what is measured, and so how
 * the requests are shaped: KL mode wants one prefill per document and every logits row, this wants
 * steady decode at a stated concurrency and the generated tokens back.
 *
 * Three things make the numbers worth comparing across builds, and each is a rule here rather than
 * a convention:
 *
 *   Nothing from the cache. Every prompt is salted with a per-point, per-request token sequence
 *      built from `--diagnose-salt`, and the prefix cache is dropped before every point. A point
 *      the cache served anyway is reported as not a measurement: its prefill did not happen, so
 *      its step times describe a different run of the workload.
 *   Greedy, fixed length. Temperature 0, `ignore_eos`, `max_tokens` stated: the same build with
 *      the same salt generates the same tokens, which is what makes `--diagnose-baseline` a check
 *      rather than a hint.
 *   The engine's own clock. Step times and burst intervals come from the records the scheduler
 *      wrote while only this point was running (core/sched/steplog.h), drained between points. A
 *      client measuring the same workload would be measuring the socket as well.
 *
 * A point the deployment cannot hold -- a 2048-token prompt and a cache that small -- is reported
 * as absent with the reason, and the run goes on: a report of a machine that cannot run one point
 * is worth more than a run that stopped at it.
 */
#include "engine_priv.h"
#include "diagnose.h"
#include "sched/steplog.h"
#include "server/sink.h"
#include "text/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <thread>
#include <vector>

namespace rad {

namespace fs = std::filesystem;

namespace {

/* Steps kept for one point. A point of this workload is a few hundred steps at most; the ring is
 * sized generously so a run on a slow card, where a point takes more of them, does not start
 * dropping the records its own report is built from. 64 Ki records is 4 MiB, allocated once, for
 * the length of the mode. */
constexpr size_t kStepLogRecords = 1u << 16;

/* The clock behind every duration in the report: the same steady clock the step brackets use, so a
 * record's gap and a sink's first-token time are on one axis. */
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

const char* finish_word(server::Finish f) {
    switch (f) {
        case server::Finish::Stop:       return "stop";
        case server::Finish::StopString: return "stop_string";
        case server::Finish::Length:     return "length";
        case server::Finish::ToolCalls:  return "tool_calls";
        case server::Finish::Cancelled:  return "cancelled";
        case server::Finish::Error:      return "error";
        default:                         return "none";
    }
}

/* Nanoseconds to the microseconds the report is written in: step times are tens to hundreds of
 * microseconds, and a reader comparing two of them should not be doing arithmetic. */
std::vector<double> micros(const std::vector<int64_t>& ns) {
    std::vector<double> out;
    out.reserve(ns.size());
    for (int64_t v : ns) out.push_back((double)v / 1000.0);
    return out;
}

/* The workload's prompt as token ids: its fixed text, grown to its stated length with a filler
 * token, and salted so nothing in it has ever been cached. The salt names the point and the
 * request, so no two prompts in a run share a prefix and no run shares one with another -- and,
 * being derived from `--diagnose-salt` rather than from the clock, two runs of one build get the
 * same prompts, which is what makes their tokens comparable. */
int build_prompt(const Tokenizer& tok, int64_t n_vocab, const diagnose::Prompt& p, int64_t salt,
                 int point_index, int req_index, std::vector<int32_t>* out, std::string* err) {
    std::vector<int32_t> ids;
    if (p.text && *p.text) {
        /* Tokenised the way the server tokenises a rendered template: control tokens parsed,
         * nothing prepended, so the table's text becomes the ids a request would carry. */
        if (tok.encode(p.text, ids, false, true) < 0) {
            *err = fmt("the workload's prompt '%s' would not tokenise", p.name);
            return RAD_E_FORMAT;
        }
    }
    /* Grown, or cut, to the stated length, so a prefill point measures a length rather than
     * whatever the text happened to tokenise to. The filler is a plain id low in the vocabulary:
     * not BOS, not EOS, not a control token a template owns, and one every container has a row
     * for. */
    if (p.tokens > 0) {
        const int32_t filler = (int32_t)std::min<int64_t>(1000, std::max<int64_t>(0, n_vocab - 1));
        if ((int64_t)ids.size() > p.tokens) ids.resize((size_t)p.tokens);
        while ((int64_t)ids.size() < p.tokens) ids.push_back(filler);
    }
    std::vector<int32_t> salt_ids;
    const std::string tag = fmt(" radiance-diagnose-%lld-p%d-r%d ", (long long)salt, point_index,
                                req_index);
    if (tok.encode(tag, salt_ids, false, true) < 0 || salt_ids.empty()) {
        *err = "the cache salt would not tokenise";
        return RAD_E_FORMAT;
    }
    out->clear();
    out->reserve(salt_ids.size() + ids.size());
    out->insert(out->end(), salt_ids.begin(), salt_ids.end());
    out->insert(out->end(), ids.begin(), ids.end());
    return RAD_OK;
}

/* The point's distributions, from the records written while it ran.
 *
 * A burst is a step that put tokens on the wire. Two readings of consecutive bursts are reported:
 * the interval from one emission to the next, which is what a client's token stream is made of, and
 * the part of that the engine waited between the two steps, which separates a long step from a
 * starved scheduler. A speculating engine emits a draft's worth of tokens and then waits, so the
 * interval carries the burst pattern and the step time is what the engine paid: two quantities, and
 * only the second explains the first.
 *
 * `split_ns`, when set, is the instant a prompt arrived while the point's streams were decoding:
 * the interval either side of it is the point's answer, and it is the only way to price that arrival
 * -- the arriving request's own time says nothing about what it cost the sequences already running.
 * All series are in microseconds by the time they reach the report. */
void point_distributions(const std::vector<StepRecord>& recs, int64_t split_ns,
                         diagnose::PointReport* out) {
    std::vector<int64_t> all, dec, mix, pre, tok;
    std::vector<double> gap, gap_before, gap_after, interval;
    std::vector<int64_t> rows, emit_pre, emit_mix, emit_dec;
    int64_t prev_token_end = -1;
    for (const StepRecord& r : recs) {
        const int64_t us = r.end_ns - r.begin_ns;
        all.push_back(us);
        rows.push_back(r.n_tok);
        /* A draft pass is a step, and it belongs to the drafter rather than to the trunk: mixing
         * them into one decode distribution would charge a head round to every decode step. */
        if (r.draft_pass == 0 && r.n_prefill > 0 && r.n_decode > 0) {
            mix.push_back(us);
            emit_mix.push_back(r.tok_out);
        } else if (r.draft_pass == 0 && r.n_prefill > 0) {
            pre.push_back(us);
            emit_pre.push_back(r.tok_out);      /* a chunk that completes emits one; most emit none */
        } else if (r.draft_pass == 0 && r.n_decode > 0) {
            dec.push_back(us);
            emit_dec.push_back(r.tok_out);
        }
        if (r.tok_out > 0) {
            tok.push_back(r.tok_out);
            if (prev_token_end >= 0) {
                /* Two readings of one pair of steps: what the engine waited between them, and the
                 * whole interval from one emission to the next. */
                gap.push_back((double)(r.begin_ns - prev_token_end) / 1000.0);
                const double iv = (double)(r.end_ns - prev_token_end) / 1000.0;
                interval.push_back(iv);
                if (split_ns > 0) (r.begin_ns < split_ns ? gap_before : gap_after).push_back(iv);
            }
            prev_token_end = r.end_ns;
        }
    }
    const std::string none = "no step of that shape ran in this point";
    /* Counts, not durations: the token and row series are reported as they are. */
    auto counts = [](const std::vector<int64_t>& v) {
        std::vector<double> o;
        o.reserve(v.size());
        for (int64_t x : v) o.push_back((double)x);
        return o;
    };
    std::vector<double> d_all = micros(all), d_dec = micros(dec), d_mix = micros(mix),
                        d_pre = micros(pre), d_gap = gap, d_gb = gap_before, d_ga = gap_after,
                        d_iv = interval, d_rows = counts(rows),
                        d_ep = counts(emit_pre), d_em = counts(emit_mix),
                        d_ed = counts(emit_dec), d_tok = counts(tok);
    out->step_us_all     = diagnose::summarize(d_all, none);
    out->step_us_decode  = diagnose::summarize(d_dec, "no decode step ran in this point");
    out->step_us_mixed   = diagnose::summarize(d_mix,
                          "no step carried a prefill chunk beside a decoder");
    out->step_us_prefill = diagnose::summarize(d_pre, none);
    out->tokens_per_step = diagnose::summarize(d_tok, "no step emitted a token");
    /* Per shape, so a prefill that got longer is not read as a decode that got slower. */
    out->tok_per_step_prefill = diagnose::summarize(d_ep, "no prefill-chunk step ran in this point");
    out->tok_per_step_mixed   = diagnose::summarize(d_em, "no step carried a prefill chunk beside a decoder");
    out->tok_per_step_decode  = diagnose::summarize(d_ed, "no decode step ran in this point");
    out->rows_per_step        = diagnose::summarize(d_rows, "no step ran in this point");
    const char* one_burst = "the point emitted tokens from one step only, so there is no interval";
    out->token_gap_us      = diagnose::summarize(d_gap, one_burst);
    out->token_interval_us = diagnose::summarize(d_iv, one_burst);
    /* The same steps the distributions above are made of, counted the same way. */
    out->shape_prefill = (int64_t)pre.size();
    out->shape_mixed   = (int64_t)mix.size();
    out->shape_decode  = (int64_t)dec.size();
    out->shape_draft   = (int64_t)all.size() - out->shape_prefill - out->shape_mixed -
                         out->shape_decode;
    if (split_ns > 0) {
        out->interval_before_arrival_us = diagnose::summarize(d_gb, "no token came before the arrival");
        out->interval_after_arrival_us  = diagnose::summarize(d_ga, "no token came after the arrival");
    }
}

/* One point while it runs: its requests, their sinks, and when the interference prompt went in. */
struct RunningPoint {
    std::vector<std::unique_ptr<server::Sink>> sinks;
    std::vector<std::unique_ptr<Request>> reqs;
    std::vector<std::vector<int32_t>> prompts;
    std::vector<std::string> prompt_names;
    int64_t interfere_at_ns = -1;
};

}  /* namespace */

/* Runs one point: builds its prompts, submits them keeping `concurrency` in flight, waits, and
 * fills the report. Returns RAD_OK when the point ran, whether or not it measured what it is for:
 * a point the deployment cannot hold is an answer, not a failure of the run. A negative return is
 * the run failing -- a prompt that would not tokenise, a scheduler that refused the workload. */
int Engine::diagnose_point(const diagnose::Point& spec, int index, int64_t salt,
                           const Tokenizer& tok, diagnose::PointReport* out, std::string* err) {
    out->name = spec.name;
    out->category = spec.category;
    out->concurrency = spec.concurrency;
    const auto& table = diagnose::prompts();
    const int64_t n_vocab = meta_.n_vocab;
    const int64_t want_max = spec.max_tokens > 0 ? spec.max_tokens : 1;

    /* The prompts. `repeat` runs each one more than once, each with its own salt, so a repeated
     * point measures a second copy of the same shape rather than the cache's memory of the first.
     * An interference point's last prompt is the arriving one, built from `interfere_prompt`: the
     * ones before it are the streams that were already decoding when it came. */
    RunningPoint dp;
    const int total = std::max(1, spec.repeat) * spec.n_prompts;
    const bool arriving = spec.interfere_after_tokens > 0;
    for (int r = 0; r < std::max(1, spec.repeat); ++r)
        for (int i = 0; i < spec.n_prompts; ++i) {
            const int which = r * spec.n_prompts + i;
            const bool extra = arriving && which + 1 == total;
            const int pi = extra ? spec.interfere_prompt
                                 : spec.first_prompt + (i % std::max(1, spec.n_prompts));
            if (pi < 0 || pi >= (int)table.size()) {
                *err = fmt("point '%s' names prompt %d of the workload's %zu", spec.name, pi,
                           table.size());
                return RAD_E_INVAL;
            }
            std::vector<int32_t> ids;
            if (build_prompt(tok, n_vocab, table[(size_t)pi], salt, index, which, &ids, err) < 0)
                return RAD_E_INVAL;
            dp.prompts.push_back(std::move(ids));
            dp.prompt_names.push_back(extra ? std::string(table[(size_t)pi].name) + " (arrives)"
                                            : std::string(table[(size_t)pi].name));
        }
    const int n = (int)dp.prompts.size();

    /* A prompt the cache could never hold alone would wait at the head of the queue forever:
     * admission takes only a chunk the pool can hold, and nothing ever frees enough. Refused here,
     * naming the length that is too long, so the report says which point this deployment is not a
     * measurement of instead of hanging on it. */
    int64_t longest = 0;
    for (const auto& ids : dp.prompts) longest = std::max(longest, (int64_t)ids.size());
    if (!ranks_[0]->kv.can_ever_fit(longest + want_max)) {
        out->measured = false;
        out->not_measured = fmt("a %lld-token prompt with %lld tokens of output is more than this "
                                "KV pool can ever hold for one sequence, so the point did not run",
                                (long long)longest, (long long)want_max);
        out->absent.push_back("step time: the point did not run");
        RAD_WARN("diagnose: %s: %s", spec.name, out->not_measured.c_str());
        return RAD_OK;
    }

    /* Nothing from the cache. A deployment running without one answers RAD_E_UNSUPPORTED, which is
     * the same answer for this run's purposes: there is nothing there to serve. */
    if (sched_.reset_prefix_cache() == RAD_E_UNSUPPORTED)
        out->absent.push_back("cache exclusivity: prefix caching is off, so nothing was cached");

    /* The scheduler's counters are cumulative for the process and nothing resets them, so a point's
     * share is the difference between the snapshot after it and the one before. */
    const SchedMetrics m0 = sched_.metrics();
    std::vector<StepRecord> recs;
    sched_.drain_step_log(&recs);          /* whatever an earlier point left behind */

    /* Greedy and `ignore_eos`, so a rerun of the same build with the same salt says the same words;
     * `max_tokens` fixed, so a point's decode length is the point's and not the model's opinion of
     * where the sentence ended. */
    dp.sinks.resize((size_t)n);
    dp.reqs.resize((size_t)n);
    /* The streams that were already running when the arriving prompt came: everything but it. */
    const int pool = arriving ? n - 1 : n;
    int next = 0;
    const int limit = std::max(1, spec.concurrency);
    /* The arrival's moment: once every stream has emitted this many tokens. */
    const int64_t interfere_after = spec.interfere_after_tokens;
    /* Sequences submitted and not yet finished -- the number the point's concurrency is about, so a
     * finished slot is free from the moment its sink says done. */
    auto live = [&] {
        int k = 0;
        for (int i = 0; i < next; ++i) if (!dp.sinks[(size_t)i]->done()) ++k;
        return k;
    };
    /* The point's stated concurrency is the limit on *admissions*; an interference point admits one
     * more stream on purpose, so what was actually in flight is recorded apart from the limit. */
    int peak_live = 0;

    auto submit = [&](int i) -> int {
        dp.sinks[(size_t)i] = std::make_unique<server::Sink>((size_t)want_max + 64);
        auto r = std::make_unique<Request>();
        r->id         = ++diag_id_;
        r->prompt     = dp.prompts[(size_t)i];
        r->max_tokens = (int32_t)want_max;
        r->ignore_eos = true;
        r->sp.temp    = 0.0f;
        r->sink       = dp.sinks[(size_t)i].get();
        dp.sinks[(size_t)i]->id = r->id;
        int rc = RAD_OK;
        std::unique_ptr<Request> in = std::move(r);
        sched_.add_all(&in, 1, &rc);
        if (rc < 0) {
            *err = fmt("the scheduler refused a prompt of point '%s' (%zu tokens): %s", spec.name,
                       dp.prompts[(size_t)i].size(), rad_strerror(rc));
            return rc;
        }
        dp.reqs[(size_t)i] = std::move(in);
        /* The arrival is the prompt held back to the end of the pool, and only an interference
         * point has one: for any other point the last prompt is an ordinary stream, and marking it
         * would split the point's intervals at a moment that means nothing. */
        if (arriving && i + 1 == n) dp.interfere_at_ns = now_ns();
        peak_live = std::max(peak_live, live());
        return RAD_OK;
    };

    /* Keep `concurrency` of the streams in flight. Filling the queue all at once would make the
     * point's batch the admission order rather than the scheduler's, which is the very thing a
     * concurrency point is about. */
    while (next < pool && live() < limit) {
        if (int s = submit(next++); s < 0) return s;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto last_log = t0;
    bool loop_dead = false;
    for (;;) {
        /* The arriving prompt, once the streams already running have each produced their share. */
        if (arriving && next >= pool && dp.interfere_at_ns < 0) {
            int64_t least = INT64_MAX;
            for (int i = 0; i < pool; ++i)
                if (!dp.sinks[(size_t)i]->done())
                    least = std::min(least, dp.sinks[(size_t)i]->n_produced());
            if (least >= interfere_after || live() == 0) {
                /* The streams have each produced their share; or one ended early, and a point that
                 * waited forever for its arrival would report nothing at all. */
                if (int s = submit(next++); s < 0) return s;
            }
        }
        bool all_done = next >= n;
        for (int i = 0; all_done && i < next; ++i)
            if (!dp.sinks[(size_t)i]->done()) all_done = false;
        if (all_done) break;
        const int ls = loop_status_.load(std::memory_order_acquire);
        if (ls < 0) { loop_dead = true; break; }
        /* Top up before waiting, so a finished sequence's place is taken by the next admission in
         * the step it freed it. */
        while (next < pool && live() < limit) {
            if (int s = submit(next++); s < 0) return s;
        }
        bool waited = false;
        for (int i = 0; i < next; ++i)
            if (!dp.sinks[(size_t)i]->done()) { dp.sinks[(size_t)i]->wait(20); waited = true; break; }
        if (!waited) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const auto now = std::chrono::steady_clock::now();
        if (now - last_log >= std::chrono::seconds(15)) {
            last_log = now;
            int done = 0;
            for (int i = 0; i < next; ++i) done += dp.sinks[(size_t)i]->done() ? 1 : 0;
            RAD_INFO("diagnose: %s: %d / %d request(s), %.0f s", spec.name, done, n,
                     std::chrono::duration<double>(now - t0).count());
        }
    }
    out->wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out->max_live = peak_live;

    /* Everything the sinks and the scheduler still hold about these requests, read before they are
     * reaped and their Request objects go away. */
    const SchedMetrics m1 = sched_.metrics();
    /* The point is over when its last sink is full, but the step that filled it is recorded a
     * moment later, by the loop thread's step_finished(). Wait for the scheduler to go quiet: a
     * point's last step belongs to its own distribution, and not to the next point's. */
    bool quiet = false;
    for (int spin = 0; spin < 500 && !quiet; ++spin) {
        quiet = sched_.step_quiet();
        if (!quiet) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!quiet)
        out->absent.push_back(
            "step time: the scheduler had not finished its last step when this point's records "
            "were read, so the distributions cover the steps before it");
    uint64_t dropped = sched_.drain_step_log(&recs);
    out->steps = m1.steps - m0.steps;
    out->requests = n;
    out->prompt_tokens = m1.prompt_tokens - m0.prompt_tokens;
    out->cached_tokens = (m1.attn_cached_tokens - m0.attn_cached_tokens) +
                         (m1.linear_cached_tokens - m0.linear_cached_tokens);
    out->prefill_tokens = m1.prefill_tokens - m0.prefill_tokens;
    out->decode_tokens  = m1.decode_tokens  - m0.decode_tokens;
    out->steps_prefill  = m1.steps_prefill - m0.steps_prefill;
    out->steps_mixed    = m1.steps_mixed   - m0.steps_mixed;
    out->steps_decode   = m1.steps_decode  - m0.steps_decode;
    out->steps_ahead    = m1.steps_ahead   - m0.steps_ahead;
    out->preemptions    = m1.preemptions   - m0.preemptions;
    out->chunks         = m1.prefill_chunks_capped - m0.prefill_chunks_capped;
    out->acceptance.drafted  = m1.draft_tokens   - m0.draft_tokens;
    out->acceptance.accepted = m1.draft_accepted - m0.draft_accepted;
    for (int j = 0; j < m1.draft_depth_seen; ++j) {
        out->acceptance.reached.push_back(m1.draft_pos_reached[j] - m0.draft_pos_reached[j]);
        out->acceptance.hits.push_back(m1.draft_pos_accepted[j] - m0.draft_pos_accepted[j]);
    }
    out->step_records_dropped = (int64_t)dropped;

    for (int i = 0; i < n; ++i) {
        server::Sink* s = dp.sinks[(size_t)i].get();
        diagnose::RequestRecord q;
        q.id = (int64_t)s->id;
        q.point = spec.name;
        q.category = spec.category;
        q.prompt = dp.prompt_names[(size_t)i];
        q.prompt_tokens = s->n_prompt();
        q.cached_tokens = s->n_cached();
        q.drafted = s->n_drafted();
        q.accepted = s->n_draft_accepted();
        q.first_token_s = s->t_first_token_ns() >= 0 ? (double)s->t_first_token_ns() / 1e9 : -1.0;
        q.last_token_s  = s->t_last_token_ns()  >= 0 ? (double)s->t_last_token_ns()  / 1e9 : -1.0;
        q.finish = finish_word(s->finish_reason());
        q.served_from_cache = q.cached_tokens > 0;
        /* The tokens, off the ring: the consumer's side is the only place they exist once the
         * request is reaped. The sink was sized for them, so nothing overflowed; if it did, the
         * point says so rather than reporting a truncated generation as a shorter one. */
        std::vector<server::SinkToken> buf((size_t)want_max + 64);
        size_t got;
        while ((got = s->read(buf.data(), buf.size())) > 0)
            for (size_t k = 0; k < got; ++k) q.tokens.push_back(buf[k].token);
        if (s->overflowed()) {
            out->measured = false;
            out->not_measured = "a request's sink overflowed, so not all of its tokens are here";
        }
        if (s->finish_reason() == server::Finish::Error ||
            s->finish_reason() == server::Finish::Cancelled) {
            out->measured = false;
            out->not_measured = fmt("a request ended %s (%s), so the point did not run as written",
                                    q.finish.c_str(), rad_strerror(s->status()));
        }
        out->out_tokens += (int64_t)q.tokens.size();
        out->reqs.push_back(std::move(q));
        sched_.reap(s->id);
    }

    /* The cache check, after the point rather than before: a salt that failed to make the prompts
     * distinct, or a cache that was not emptied, shows up as a hit -- and a hit means the prefill
     * this point is measuring did not happen. */
    if (out->cached_tokens > 0) {
        out->measured = false;
        out->not_measured = fmt("the prefix cache served %lld of this point's %lld prompt tokens, "
                                "so its prefill did not happen and its times are not this "
                                "workload's", (long long)out->cached_tokens,
                                (long long)out->prompt_tokens);
    }
    if (loop_dead) {
        out->measured = false;
        out->not_measured = fmt("the step loop failed (%s) during the point",
                                rad_strerror(loop_status_.load()));
    }
    point_distributions(recs, dp.interfere_at_ns, out);
    if (recs.empty() && out->steps > 0)
        out->absent.push_back("step time: the scheduler wrote no record for this point");
    if (!out->measured)
        RAD_WARN("diagnose: %s: not a measurement: %s", spec.name, out->not_measured.c_str());
    else
        RAD_INFO("diagnose: %s: %d request(s), %lld step(s), %.2f s, %lld tokens out, "
                 "decode p50 %s us", spec.name, n, (long long)out->steps, out->wall_s,
                 (long long)out->out_tokens,
                 out->step_us_decode.p50.present ? fmt("%.1f", out->step_us_decode.p50.value).c_str()
                                                 : "not measured");
    return RAD_OK;
}

/* ------------------------------------------------------------------ the mode */
int Engine::run_diagnose() {
    const std::string dir = cfg_.diagnose_dir;
    if (!fs::exists(dir)) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            RAD_ERR("diagnose: cannot create %s: %s", dir.c_str(), ec.message().c_str());
            return RAD_E_IO;
        }
    } else if (!fs::is_directory(dir)) {
        RAD_ERR("diagnose: %s is not a directory", dir.c_str());
        return RAD_E_IO;
    }
    /* The generator and the embedding program run no scheduler (engine_gen.cpp), and the workload
     * is a serving workload: say so rather than measuring nothing and calling it a report. */
    if (gen_ || embedding_) {
        RAD_ERR("diagnose: the workload is a serving workload, and this program is a %s with no "
                "scheduler to run it through", embedding_ ? "embedding program" : "generator");
        return RAD_E_UNSUPPORTED;
    }

    Tokenizer tok;
    if (tok.init(vocab_) < 0) {
        RAD_ERR("diagnose: the workload's prompts are tokenised with the container's tokenizer");
        return RAD_E_STATE;
    }

    diagnose::Report rep;
    rep.workload = diagnose::workload_id();
    rep.salt = cfg_.diagnose_salt;
    rep.engine = {
        { "version", RAD_VERSION },
        { "model", cfg_.model },
        { "backend", device_is_host() ? "host" : "device" },
        { "tensor parallel", fmt("%d", cfg_.tp) },
        { "drafter", drafter_.depth > 0
                         ? fmt("%s depth %d", drafter_name_.c_str(), drafter_.depth)
                         : std::string("none: this model declares no draft head, so nothing is "
                                       "speculated") },
        { "speculative window", fmt("%d", cfg_.n_spec) },
        { "prefix cache", cfg_.prefix_cache ? "on, and dropped before every point" : "off" },
        { "max batched tokens", fmt("%lld", (long long)cfg_.max_tok) },
        { "max sequences", fmt("%d", cfg_.max_seqs) },
        { "chunk geometry", fmt("quantum %lld, checkpoint interval %lld",
                                (long long)geo_.quantum, (long long)geo_.checkpoint_interval) },
        { "deterministic", cfg_.deterministic ? "yes" : "no" },
        { "startup report", "core/diag/diag.h, in the log above these lines" },
    };

    /* The step records: one per step, for this mode only, kept until the point reading them is
     * over, and off again before the mode returns, so no other reader of the scheduler pays for
     * them. */
    sched_.enable_step_log(kStepLogRecords);

    for (size_t i = 0; i < diagnose::points().size(); ++i) {
        diagnose::PointReport pr;
        std::string err;
        const int s = diagnose_point(diagnose::points()[i], (int)i, cfg_.diagnose_salt, tok, &pr,
                                     &err);
        if (s < 0) {
            sched_.disable_step_log();
            RAD_ERR("diagnose: %s", err.c_str());
            return s;
        }
        rep.points.push_back(std::move(pr));
        if (loop_status_.load(std::memory_order_acquire) < 0) break;   /* the rest would hang */
    }
    sched_.disable_step_log();

    if (const int ls = loop_status_.load(std::memory_order_acquire); ls < 0) {
        RAD_ERR("diagnose: the step loop failed: %s", rad_strerror(ls));
        return ls;
    }

    /* What the report says about itself, and then about the baseline it was asked to compare with. */
    std::string text = diagnose::report_json(rep);
    diagnose::review(text, &rep.findings);

    if (!cfg_.diagnose_baseline.empty()) {
        std::ifstream in(cfg_.diagnose_baseline, std::ios::binary);
        const std::string body((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        if (body.empty()) {
            RAD_ERR("diagnose: cannot read --diagnose-baseline %s", cfg_.diagnose_baseline.c_str());
            return RAD_E_IO;
        }
        std::string why;
        if (!diagnose::parse_ok(body, &why)) {
            RAD_ERR("diagnose: --diagnose-baseline %s: %s", cfg_.diagnose_baseline.c_str(),
                    why.c_str());
            return RAD_E_FORMAT;
        }
        if (!diagnose::same_workload(body, text, &why)) {
            RAD_ERR("diagnose: %s", why.c_str());
            return RAD_E_FORMAT;
        }
        diagnose::compare_tokens(body, text, &rep.findings);
        /* The threshold the flag named, not the default it ships with: the startup report echoes
         * --diagnose-acceptance-drop, and a comparison that ignored it would report a fall the
         * operator had asked not to hear about (or stay quiet about one they had). */
        diagnose::compare_acceptance(body, text, cfg_.diagnose_acceptance_drop, &rep.findings);
    }

    for (const diagnose::Finding& f : rep.findings)
        RAD_LOG_AT(f.level == diagnose::Finding::Fail ? Log::Error : Log::Warn,
                   "diagnose: %s: %s", f.id.c_str(), f.what.c_str());

    /* The two files. The JSON is the record -- every token, every distribution, the engine's own
     * facts -- and the markdown is the reading of it: the same numbers, one table per family. */
    text = diagnose::report_json(rep);
    const std::string md = diagnose::report_markdown(rep);
    std::ofstream dj(dir + "/diagnose.json", std::ios::binary);
    std::ofstream dm(dir + "/diagnose.md", std::ios::binary);
    if (!dj || !dm) {
        RAD_ERR("diagnose: cannot write %s/diagnose.json and %s/diagnose.md", dir.c_str(),
                dir.c_str());
        return RAD_E_IO;
    }
    dj << text;
    dm << md;
    if (!dj || !dm) {
        RAD_ERR("diagnose: writing %s failed", dir.c_str());
        return RAD_E_IO;
    }
    size_t reqs = 0;
    for (const auto& p : rep.points) reqs += p.reqs.size();
    RAD_INFO("diagnose: wrote %s/diagnose.json and %s/diagnose.md: %zu point(s), %zu request(s)",
             dir.c_str(), dir.c_str(), rep.points.size(), reqs);

    /* A token-parity failure is the run's failure. The point of `--diagnose-baseline` is that a
     * build which says different words does not get to report its throughput instead. */
    for (const diagnose::Finding& f : rep.findings)
        if (f.level == diagnose::Finding::Fail) {
            RAD_ERR("diagnose: %s: %s", f.id.c_str(), f.what.c_str());
            return RAD_E_STATE;
        }
    return RAD_OK;
}

}  /* namespace rad */
