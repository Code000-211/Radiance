#include "server.h"

#include <cstdio>
#include "json_partial.h"
#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <thread>

namespace rad {
namespace server {

/* ------------------------------------------------------------------ per-choice state */

struct Server::Generation {
    std::shared_ptr<Sink>         sink;
    std::unique_ptr<IDetokenizer> detok;
    std::unique_ptr<ChatDeltaStream> delta;   /* chat streaming only */
    uint64_t    id = 0;
    int         index = 0;
    int         prompt_index = 0;
    std::string text;
    std::vector<LogprobEntry> logprobs;
    /* The reply against its reasoning block, every token the sink delivers: what
     * completion_tokens_details.reasoning_tokens reports. */
    ReasoningCount think;
    Finish      finish = Finish::None;
    bool        closed = false;      /* the server has decided this choice is over */
    size_t      scanned = 0;         /* how far the stop-string scan has reached */
    size_t      stop_cut = std::string::npos;
    size_t      emitted = 0;         /* streaming: text already handed to the delta stream */
    bool        retired = false;     /* reap() has been said for this id, at most once */
};

struct Server::StreamState {
    OaiRequest              r;
    std::vector<Generation> gens;
    std::function<bool()>   peer_closed;
    bool opened = false;
    bool done_sent = false;
    bool released = false;
};

/* ------------------------------------------------------------------ construction */

Server::Server(const Deps& d, const ServerOptions& o)
    : deps_(d), opt_(o),
      metrics_(o.model_id, o.max_ctx > 0 ? o.max_ctx : 32768) {
    lim_.model_id = opt_.model_id;
    lim_.max_ctx = opt_.max_ctx;
    lim_.default_max_tokens = opt_.default_max_tokens;
    lim_.max_tokens_cap = opt_.max_tokens_cap;
    lim_.max_n = opt_.max_n;
    lim_.max_stops = opt_.max_stops;
    lim_.max_stop_bytes = opt_.max_stop_bytes;
    lim_.allow_image = opt_.allow_image;
    lim_.allow_video = opt_.allow_video;
    lim_.allow_audio = opt_.allow_audio;
    lim_.supports_logprobs = opt_.supports_logprobs;
    lim_.default_reasoning_effort = opt_.default_reasoning_effort;
    lim_.default_thinking_budget = opt_.default_thinking_budget;
    lim_.default_sampling = opt_.default_sampling;
    lim_.default_dry_breakers_set = opt_.default_dry_breakers_set;
    lim_.default_template_kwargs = opt_.default_template_kwargs;
    lim_.reasoning_format = opt_.reasoning_format;

    odeps_.tok = deps_.tok;
    odeps_.chat = deps_.chat;
    odeps_.grammar = deps_.grammar;
    odeps_.mm = deps_.mm;

    queue_depth_ = opt_.queue_depth > 0 ? opt_.queue_depth : opt_.max_seqs * 8;
    started_at_ = unix_now();
    register_routes();
}

Server::~Server() { stop(); }

/* Sinks the engine may still be writing into must not be freed. `Request::sink` is a raw pointer
 * the scheduler holds for the life of the request, so the only safe moment to release one is
 * after the scheduler has reported it terminal. At shutdown we wait a bounded time for that and,
 * if it never comes, we leak the remainder on purpose and say so.
 *
 * A deliberate leak of a few KiB at process exit is the cheap failure. Freeing under a running
 * step is the expensive one: it corrupts whatever the allocator hands out next, and it surfaces
 * as garbage in an unrelated request several steps later, which is precisely the class of bug
 * spec §17 exists to keep out of this engine. */
void Server::drain_graveyard(int timeout_ms) {
    const int slice = 5;
    for (int waited = 0; waited <= timeout_ms; waited += slice) {
        sweep_abandoned();
        {
            std::lock_guard<std::mutex> lk(grave_m_);
            if (grave_.empty()) return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(slice));
    }
    std::lock_guard<std::mutex> lk(grave_m_);
    if (grave_.empty()) return;
    RAD_ERR("server: %zu cancelled requests are still held by the scheduler after %d ms. "
            "Stop the scheduler before destroying the server. Leaking their sinks rather than "
            "freeing memory a running step may still write to.",
            grave_.size(), timeout_ms);
    /* Hand the remainder to storage that outlives the process's teardown of this object. */
    static std::vector<std::vector<std::shared_ptr<Sink>>>* leaked =
        new std::vector<std::vector<std::shared_ptr<Sink>>>();
    leaked->push_back(std::move(grave_));
    grave_.clear();
}

/* WHAT IS SERVED. One table, and /server_info answers with it rather than with a copy -- a list
 * kept beside this one is a list that goes stale on the next route added.
 *
 * Method spellings follow vLLM's, including the ones that look wrong: /tokenize and /detokenize
 * are POST because their input is a body, /reset_prefix_cache is POST because it changes state,
 * and /ping answers both GET and POST because different load balancers send different ones and
 * neither is worth a 405 on a health probe. */
void Server::register_routes() {
    router_.post("/v1/chat/completions", [this](const HttpRequest& q) { return handle_chat(q); });
    router_.post("/v1/completions",      [this](const HttpRequest& q) { return handle_completion(q); });
    router_.post("/v1/embeddings",       [this](const HttpRequest& q) { return handle_embeddings(q); });
    router_.get ("/v1/models",           [this](const HttpRequest& q) { return handle_models(q); });
    router_.get ("/v1/models/:id",       [this](const HttpRequest& q) { return handle_models(q); });

    router_.get ("/metrics",             [this](const HttpRequest& q) { return handle_metrics(q); });
    router_.get ("/health",              [this](const HttpRequest& q) { return handle_health(q); });
    router_.get ("/ping",                [this](const HttpRequest& q) { return handle_ping(q); });
    router_.post("/ping",                [this](const HttpRequest& q) { return handle_ping(q); });
    router_.get ("/load",                [this](const HttpRequest& q) { return handle_load(q); });
    router_.get ("/version",             [this](const HttpRequest& q) { return handle_version(q); });
    router_.get ("/server_info",         [this](const HttpRequest& q) { return handle_server_info(q); });

    router_.post("/tokenize",            [this](const HttpRequest& q) { return handle_tokenize(q); });
    router_.post("/detokenize",          [this](const HttpRequest& q) { return handle_detokenize(q); });
    router_.get ("/get_tokenizer_info",  [this](const HttpRequest& q) { return handle_tokenizer_info(q); });

    router_.post("/reset_prefix_cache",  [this](const HttpRequest& q) { return handle_reset_prefix_cache(q); });

    /* The dashboard. `/` rather than only `/dashboard` because a person who has the host and port
     * of a running engine types exactly that, and answering it with a 404 is answering the
     * question "is this thing alive" with "no". */
    router_.get ("/",                    [this](const HttpRequest& q) { return handle_dashboard(q); });
    router_.get ("/dashboard",           [this](const HttpRequest& q) { return handle_dashboard(q); });
    router_.get ("/stats",               [this](const HttpRequest& q) { return handle_stats(q); });
    router_.get ("/graph",               [this](const HttpRequest& q) { return handle_graph(q); });
    router_.get ("/sessions",            [this](const HttpRequest& q) { return handle_sessions(q); });
}

/* WORKERS FOR EVERYTHING ADMISSION CAN LET IN, AND A RESERVE BESIDE THEM.
 *
 * A generation holds its connection's worker for as long as it runs -- a stream writes from it, a
 * blocking request waits on its sink in it -- so a pool no larger than the admission ceiling runs
 * out of workers before admission refuses anyone. The 429 can then never happen, and every
 * connection after that waits in the pool's queue behind generations that may run for minutes,
 * /health included. So the pool may grow to the ceiling plus a reserve no admitted generation can
 * occupy: that is what keeps a probe, the dashboard, a request about to be told 429 and a
 * client's idle keep-alive connections answerable while every admitted request is running. The
 * same number bounds the connections that may wait once even the reserve is busy. */
static const int64_t kSpareWorkers = 64;

int Server::run() {
    TransportOptions t;
    t.host = opt_.host;
    t.port = opt_.port;
    t.n_threads = opt_.n_threads;
    t.max_threads = queue_depth_ + kSpareWorkers;
    t.max_queued = kSpareWorkers;
    t.api_key = opt_.api_key;
    t.cors = opt_.cors;
    t.read_timeout_s = opt_.read_timeout_s;
    t.write_timeout_s = opt_.write_timeout_s;
    t.keep_alive_timeout_s = opt_.keep_alive_timeout_s;
    t.max_body_bytes = opt_.max_body_bytes;
    return transport_.serve(router_, t);
}

void Server::check_template_defaults(std::string* kwargs_why, std::string* effort_why) const {
    kwargs_why->clear();
    effort_why->clear();
    if (!deps_.chat) return;
    const std::string body = R"({"messages":[{"role":"user","content":"Hello"}],"max_tokens":1})";
    auto render = [&](const OaiLimits& l, std::string* why) {
        OaiRequest r;
        ApiError e;
        if (parse_chat_request(body, odeps_, l, &r, &e) >= 0) return true;
        if (why) *why = e.message;
        return false;
    };
    OaiLimits l = lim_;
    l.default_template_kwargs.clear();
    l.default_reasoning_effort.clear();
    if (!render(l, nullptr)) return;
    l.default_template_kwargs = lim_.default_template_kwargs;
    if (!l.default_template_kwargs.empty() && !render(l, kwargs_why)) return;
    l.default_reasoning_effort = lim_.default_reasoning_effort;
    if (!l.default_reasoning_effort.empty()) render(l, effort_why);
}

std::string Server::check_thinking_budget() const {
    if (!deps_.chat || lim_.default_thinking_budget < 0) return std::string();
    const std::string body = R"({"messages":[{"role":"user","content":"Hello"}],"max_tokens":1,)"
                             R"("thinking_token_budget":)" +
                             std::to_string(lim_.default_thinking_budget) + "}";
    OaiRequest r;
    ApiError e;
    return parse_chat_request(body, odeps_, lim_, &r, &e) < 0 ? e.message : std::string();
}

void Server::stop() {
    stopping_.store(true);
    transport_.stop();
    drain_graveyard(2000);
}

/* ------------------------------------------------------------------ admission */

bool Server::admit(int64_t n) {
    int64_t cur = in_flight_.fetch_add(n, std::memory_order_acq_rel) + n;
    if (cur > queue_depth_) {
        in_flight_.fetch_sub(n, std::memory_order_acq_rel);
        return false;
    }
    metrics_.set_queue_depth(cur);
    return true;
}

void Server::release(int64_t n) {
    int64_t cur = in_flight_.fetch_sub(n, std::memory_order_acq_rel) - n;
    metrics_.set_queue_depth(cur);
}

void Server::sweep_abandoned() {
    std::lock_guard<std::mutex> lk(grave_m_);
    size_t before = grave_.size();
    grave_.erase(std::remove_if(grave_.begin(), grave_.end(),
                                [this](const std::shared_ptr<Sink>& s) {
                                    if (!s->done()) return false;
                                    deps_.sched->reap(s->id);
                                    return true;
                                }),
                 grave_.end());
    if (grave_.size() > 256 && grave_.size() >= before)
        RAD_WARN("server: %zu cancelled requests the scheduler has not reported terminal; "
                 "IScheduler::cancel must finish them, not just unschedule them",
                 grave_.size());
}

/* ------------------------------------------------------------------ errors */

/* " from <addr:port>" for the request log, or nothing in-process, where no transport names a
 * peer. */
static std::string from_peer(const std::string& peer) {
    return peer.empty() ? std::string() : " from " + peer;
}

HttpResponse Server::error_response(const ApiError& e, const std::string& peer) {
    metrics_.observe_http(e.status);
    /* A refusal is the line the operator is looking for when a client says "rejected": the
     * status, the type, the message, and who asked. 4xx is the client's doing and said at info;
     * 5xx is ours and said as a warning. */
    if (e.status >= 500)
        RAD_WARN("req refused%s -> %d %s: %s", from_peer(peer).c_str(), e.status, e.type.c_str(),
                 e.message.c_str());
    else
        RAD_INFO("req refused%s -> %d %s: %s", from_peer(peer).c_str(), e.status, e.type.c_str(),
                 e.message.c_str());
    HttpResponse r = HttpResponse::json_body(e.status, e.dump());
    if (e.status == 429) r.set_header("Retry-After", std::to_string(opt_.retry_after_s));
    return r;
}

/* ------------------------------------------------------------------ the request log */

void Server::log_admitted(const OaiRequest& r) const {
    int64_t n_prompt = 0;
    for (const auto& p : r.prompts) n_prompt += (int64_t)p.tokens.size();
    const char* kind = r.endpoint == Endpoint::Chat ? "chat" : "completion";
    if (r.endpoint == Endpoint::Chat)
        RAD_INFO("req %s %s%s%s model=%s msgs=%d tools=%d prompt=%lld tok n=%d max_tokens=%lld",
                 r.id.c_str(), kind, r.stream ? " stream" : "", from_peer(r.peer).c_str(),
                 r.model.c_str(), r.n_messages, r.n_tools, (long long)n_prompt, r.n,
                 (long long)r.max_tokens);
    else
        RAD_INFO("req %s %s%s%s model=%s prompts=%zu prompt=%lld tok n=%d max_tokens=%lld",
                 r.id.c_str(), kind, r.stream ? " stream" : "", from_peer(r.peer).c_str(),
                 r.model.c_str(), r.prompts.size(), (long long)n_prompt, r.n,
                 (long long)r.max_tokens);
    if (!r.last_user.empty())
        RAD_DEBUG("req %s last user: %s", r.id.c_str(), r.last_user.c_str());
}

/* ------------------------------------------------------------------ submission */

int Server::submit_all(const OaiRequest& r, std::vector<Generation>* gens, ApiError* err) {
    const int n_choices = r.n_choices();
    gens->reserve((size_t)n_choices);

    /* THE SAMPLER GETS A SAY BEFORE ANYTHING IS QUEUED. A stage that did not resolve is not a
     * declare failure -- it becomes one when a request asks for it. Refusing here makes that a
     * 400 naming the parameter, instead of an admitted request that the step loop cancels three
     * steps in with a dead stream at the client.
     *
     * Once, not per choice: `n` varies only the seed, and no check here reads it. */
    {
        std::string why;
        if (deps_.sched->admit(r.sp, &why) < 0) {
            *err = err_unsupported("sampling", why.empty()
                                       ? "these sampling parameters cannot be served" : why);
            return RAD_E_INVAL;
        }
    }

    int idx = 0;
    std::vector<Request> rqs;
    rqs.reserve(r.prompts.size() * (size_t)r.n);
    const size_t first = gens->size();
    for (size_t p = 0; p < r.prompts.size(); ++p) {
        for (int k = 0; k < r.n; ++k, ++idx) {
            Generation g;
            g.index = idx;
            g.prompt_index = (int)p;
            g.sink = std::make_shared<Sink>();
            g.sink->set_has_logprobs(opt_.supports_logprobs);
            g.think.start(r.sp.reasoning);
            g.detok = deps_.tok->detokenizer(r.stop, r.preserved_tokens);
            if (!g.detok) {
                *err = err_not_implemented("the tokenizer did not supply an incremental decoder");
                return RAD_E_UNSUPPORTED;
            }

            Request rq;
            rq.prompt = r.prompts[p].tokens;
            rq.sp = r.sp;
            /* n identical seeds would give n identical completions, which is not what `n` means.
             * Offsetting is enough: the sampler's stream is seeded, not chained. */
            rq.sp.seed = r.sp.seed + (uint64_t)k;
            rq.max_tokens = r.max_tokens;
            rq.stop = r.stop;
            rq.ignore_eos = r.ignore_eos;
            rq.priority = r.priority;
            rq.stream = r.stream;
            rq.media = r.prompts[p].media;
            rq.sink = g.sink.get();
            rqs.push_back(std::move(rq));
            gens->push_back(std::move(g));
        }
    }
    std::vector<uint64_t> ids;
    deps_.sched->submit_all(rqs, &ids);
    for (size_t i = 0; i < ids.size(); ++i) {
        Generation& g = (*gens)[first + i];
        g.id = ids[i];
        g.sink->id = g.id;
    }
    return RAD_OK;
}

void Server::cancel_all(std::vector<Generation>& gens, bool count_disconnect) {
    for (auto& g : gens) {
        if (g.sink->done()) continue;
        g.sink->request_cancel();
        deps_.sched->cancel(g.id);
    }
    if (count_disconnect) metrics_.observe_disconnect();
}

/* Every exit from a request goes through here exactly once. A generation the engine has finished
 * is reaped -- that is the server telling the scheduler it may drop the Request, which it cannot
 * know on its own because we are the ones holding the sink. One the engine has not finished moves
 * to the graveyard, and is reaped by the sweep when it does. */
void Server::retire(std::vector<Generation>& gens) {
    std::vector<std::shared_ptr<Sink>> pending;
    for (auto& g : gens) {
        if (g.retired) continue;
        g.retired = true;
        if (g.sink->done()) deps_.sched->reap(g.id);
        else pending.push_back(g.sink);
    }
    if (pending.empty()) return;
    std::lock_guard<std::mutex> lk(grave_m_);
    for (auto& s : pending) grave_.push_back(std::move(s));
}

/* ------------------------------------------------------------------ draining */

bool Server::all_done(const std::vector<Generation>& gens) const {
    for (const auto& g : gens) if (!g.closed) return false;
    return true;
}

bool Server::pump(std::vector<Generation>& gens, const OaiRequest& r) {
    bool progress = false;
    SinkToken buf[128];

    size_t maxstop = 0;
    for (const auto& s : r.stop) maxstop = std::max(maxstop, s.size());

    for (auto& g : gens) {
        if (g.closed) continue;

        size_t n;
        while ((n = g.sink->read(buf, 128)) > 0) {
            progress = true;
            for (size_t i = 0; i < n; ++i) {
                g.think.push(r.sp.reasoning, buf[i].token);
                std::string piece = g.detok->push(buf[i].token);
                if (!piece.empty()) g.text += piece;
                if (r.want_logprobs)
                    g.logprobs.push_back(LogprobEntry{buf[i].token, piece, buf[i].logprob});
            }
        }

        /* The decoder may have swallowed the stop string itself, in which case our own scan below
         * can never find it -- it is searching text the stop string was already removed from. */
        if (maxstop && !g.closed && g.stop_cut == std::string::npos && g.detok->stopped()) {
            g.stop_cut = g.text.size();
            g.finish = Finish::StopString;
            g.closed = true;
            progress = true;
            g.sink->request_cancel();
            deps_.sched->cancel(g.id);
            continue;
        }

        /* Stop strings. The scheduler is given Request::stop and may stop first, but a stop
         * string straddles token boundaries and only the detokenised text can be searched, so the
         * authoritative cut is made here. Scanning restarts (maxstop-1) bytes behind the previous
         * end so a match spanning two drains is still found. */
        if (maxstop && g.stop_cut == std::string::npos && !g.text.empty()) {
            size_t start = g.scanned >= maxstop ? g.scanned - maxstop + 1 : 0;
            for (const auto& s : r.stop) {
                if (s.empty()) continue;
                size_t at = g.text.find(s, start);
                if (at != std::string::npos && (g.stop_cut == std::string::npos || at < g.stop_cut))
                    g.stop_cut = at;
            }
            g.scanned = g.text.size();
            if (g.stop_cut != std::string::npos) {
                g.text.resize(g.stop_cut);
                g.finish = Finish::StopString;
                g.closed = true;
                progress = true;
                /* The client will never see another token of this choice, so the blocks it holds
                 * are dead weight the moment we decide that (spec §14). */
                g.sink->request_cancel();
                deps_.sched->cancel(g.id);
                continue;
            }
        }

        if (g.sink->done()) {
            std::string tail = g.detok->flush();
            if (!tail.empty()) { g.text += tail; progress = true; }
            g.finish = g.sink->finish_reason();
            if (g.sink->overflowed()) metrics_.observe_overflow();
            g.closed = true;
            progress = true;
        }
    }
    return progress;
}

/* ------------------------------------------------------------------ usage */

/* THE REQUEST'S TOKENS, ONCE FOR BOTH PATHS. `n` choices share a prompt, so the prompt and its
 * cached part are counted once per prompt. Reasoning is reported only where the reply's block can
 * be counted by token (ReasoningSpan); a model without one reports no breakdown rather than a
 * zero that would claim there was no reasoning. */
UsageOut Server::usage_of(const OaiRequest& r, const std::vector<Generation>& gens) {
    UsageOut u;
    for (const auto& p : r.prompts) u.prompt += (int64_t)p.tokens.size();
    if (r.sp.reasoning.known()) u.reasoning = 0;
    for (const auto& g : gens) {
        u.completion += g.sink->n_produced();
        if (g.prompt_index * r.n == g.index) u.cached += g.sink->n_cached();
        if (u.reasoning >= 0) u.reasoning += g.think.tokens;
    }
    return u;
}

/* WHERE THE REQUEST'S TIME WENT, from its sinks: one for a choice's final chunk, all of them for
 * the request. Every time is moved onto one clock first -- each sink counts from its own
 * construction. See TimingsOut for what each field means. */
TimingsOut Server::timings_of(const Sink* const* sinks, size_t n) {
    TimingsOut t;
    int64_t admit = INT64_MAX, first_lo = INT64_MAX, first_hi = INT64_MIN, last = INT64_MIN;
    bool admitted = true;
    for (size_t i = 0; i < n; ++i) {
        const Sink& s = *sinks[i];
        const int64_t produced = s.n_produced();
        t.predicted_n      += produced;
        t.draft_n          += s.n_drafted();
        t.draft_n_accepted += s.n_draft_accepted();
        const int64_t tf = s.t_first_token_ns();
        if (tf < 0) continue;
        t.valid = true;
        t.cache_n  += s.n_cached();
        t.prompt_n += s.n_prompt() - s.n_cached();
        if (produced > 1) t.decode_n += produced - 1;
        const int64_t t0 = s.t_start_ns(), ta = s.t_admitted_ns();
        if (ta < 0) admitted = false;
        else admit = std::min(admit, t0 + ta);
        first_lo = std::min(first_lo, t0 + tf);
        first_hi = std::max(first_hi, t0 + tf);
        last     = std::max(last, t0 + s.t_last_token_ns());
    }
    if (!t.valid) return t;
    if (admitted) t.prompt_ms = (double)(first_hi - admit) * 1e-6;
    t.predicted_ms = (double)(last - first_lo) * 1e-6;
    return t;
}

TimingsOut Server::timings_of(const std::vector<Generation>& gens) {
    std::vector<const Sink*> v;
    v.reserve(gens.size());
    for (const auto& g : gens) v.push_back(g.sink.get());
    return timings_of(v.data(), v.size());
}

/* ------------------------------------------------------------------ metrics bookkeeping */

/* WHAT FAILED, for the client: the sentence the failing component gave, and the status. A failed
 * request always carries a status -- an engine that stopped carries the one that stopped it -- so
 * this never reads "ok". */
static std::string engine_failure(const Sink& s) {
    const int st = s.status() < 0 ? s.status() : RAD_E_STATE;
    const char* why = s.why();
    return why ? fmt("the engine failed this request: %s (%s)", why, rad_strerror(st))
               : fmt("the engine failed this request: %s", rad_strerror(st));
}

static SuccessReason to_reason(Finish f) {
    switch (f) {
        case Finish::Length:    return SuccessReason::Length;
        case Finish::ToolCalls: return SuccessReason::ToolCalls;
        case Finish::Cancelled: return SuccessReason::Abort;
        case Finish::Error:     return SuccessReason::Error;
        default:                return SuccessReason::Stop;
    }
}

void Server::record(const OaiRequest& r, const std::vector<Generation>& gens, Finish overall) {
    int64_t n_prompt = 0;
    for (const auto& p : r.prompts) n_prompt += (int64_t)p.tokens.size();

    int64_t n_gen = 0;
    double ttft = -1.0, tpot = -1.0, decode = -1.0, e2e = 0.0;
    for (const auto& g : gens) {
        int64_t produced = g.sink->n_produced();
        n_gen += produced;
        e2e = std::max(e2e, (double)g.sink->age_ns() * 1e-9);
        int64_t tf = g.sink->t_first_token_ns(), tl = g.sink->t_last_token_ns();
        if (tf >= 0) {
            double t = (double)tf * 1e-9;
            ttft = ttft < 0 ? t : std::min(ttft, t);
            if (produced > 1 && tl > tf) {
                double d = (double)(tl - tf) * 1e-9;
                decode = decode < 0 ? d : std::max(decode, d);
                double per = d / (double)(produced - 1);
                tpot = tpot < 0 ? per : std::max(tpot, per);
            }
        }
    }
    metrics_.observe_request(n_prompt, n_gen, ttft, tpot, e2e, decode,
                             r.n, r.max_tokens, to_reason(overall));

    /* THE OTHER HALF OF THE REQUEST LOG. The same figures /metrics aggregates, for this one
     * request: how it ended, what the prefix cache spared, the reasoning share when the parser
     * knows it, time to first token, the decode rate after it, and the whole. */
    const TimingsOut t = timings_of(gens);
    const UsageOut u = usage_of(r, gens);
    const double rate = (decode > 0.0 && n_gen > 1) ? (double)(n_gen - 1) / decode : -1.0;
    char reason[32] = "";
    if (u.reasoning >= 0) std::snprintf(reason, sizeof reason, " reason=%lld", (long long)u.reasoning);
    char ttft_s[32] = "ttft=-";
    if (ttft >= 0) std::snprintf(ttft_s, sizeof ttft_s, "ttft=%.2fs", ttft);
    char rate_s[32] = "";
    if (rate > 0) std::snprintf(rate_s, sizeof rate_s, " %.1f tok/s", rate);
    RAD_INFO("req %s done%s finish=%s prompt=%lld (cache %lld) gen=%lld%s %s%s %.2fs",
             r.id.c_str(), from_peer(r.peer).c_str(), finish_reason_oai(overall),
             (long long)n_prompt, (long long)(t.valid ? t.cache_n : 0), (long long)n_gen, reason,
             ttft_s, rate_s, e2e);
}

/* ------------------------------------------------------------------ non-streaming */

HttpResponse Server::blocking_completion(const HttpRequest& req, OaiRequest& r) {
    std::vector<Generation> gens;
    ApiError err;
    if (submit_all(r, &gens, &err) < 0) {
        retire(gens);
        release(r.n_choices());
        return error_response(err, req.peer);
    }

    for (;;) {
        pump(gens, r);
        if (all_done(gens)) break;

        /* A client that hangs up before the answer is ready is a cancellation, not something to
         * finish computing for nobody. */
        if (req.peer_gone()) {
            cancel_all(gens, true);
            retire(gens);
            release(r.n_choices());
            metrics_.observe_http(499);
            HttpResponse res = HttpResponse::text(499, "client closed request\n");
            return res;
        }
        if (stopping_.load()) {
            cancel_all(gens, false);
            retire(gens);
            release(r.n_choices());
            return error_response(err_overloaded("the server is shutting down"), req.peer);
        }
        /* Wait on whichever choice is still running. Bounded, so the peer check above runs. */
        for (auto& g : gens) if (!g.closed) { g.sink->wait(); break; }
    }

    Finish overall = Finish::Stop;
    const UsageOut usage = usage_of(r, gens);
    std::vector<ChoiceOut> choices;
    choices.reserve(gens.size());

    for (auto& g : gens) {

        ChoiceOut c;
        c.index = g.index;
        c.finish = g.finish;
        c.logprobs = std::move(g.logprobs);

        if (r.endpoint == Endpoint::Chat) {
            /* Parse whenever there is a parser, not only when tools were sent: the same parser
             * separates the reasoning block from the answer, and a request with no tools still
             * wants `reasoning_content` split out rather than glued to the front of the reply.
             * It is the same reader the streaming path feeds a piece at a time, fed the whole
             * reply at once, so the two agree by construction. With no tools in the request the
             * parser reads no calls and the tool bookkeeping below finds none. */
            if (r.reply_parser) {
                std::unique_ptr<IReplyReader> rd = r.reply_parser->open();
                rd->feed(g.text, nullptr);
                rd->finish(nullptr);
                c.msg = rd->message();
                /* Only claim a tool call when the arguments actually parse (json_partial.h). */
                bool complete = r.parse_tools && !c.msg.tool_calls.empty();
                for (const auto& tc : c.msg.tool_calls)
                    if (tc.name.empty() || !json_complete(tc.arguments)) complete = false;
                if (complete && (c.finish == Finish::Stop || c.finish == Finish::StopString))
                    c.finish = Finish::ToolCalls;
                else if (!c.msg.tool_calls.empty() && !complete) {
                    /* A half-finished call is content, not a call. The client gets what the model
                     * actually produced instead of a tool invocation with truncated arguments. */
                    c.msg.tool_calls.clear();
                    c.msg.content = g.text;
                }
            } else {
                c.msg.content = g.text;
            }
            for (auto& tc : c.msg.tool_calls) if (tc.id.empty()) tc.id = random_id("call_");
        } else {
            c.text = r.echo ? r.prompts[g.prompt_index].text + g.text : g.text;
        }
        /* The request's outcome for record() and /metrics: a failure or a cancellation over
         * anything, else a tool call over a plain stop -- the same promotion the streaming path
         * makes, so a blocking tool call is not counted as "stop". */
        if (c.finish == Finish::Error || c.finish == Finish::Cancelled) overall = c.finish;
        else if (c.finish == Finish::ToolCalls && overall == Finish::Stop) overall = c.finish;
        choices.push_back(std::move(c));
    }

    /* An engine-side failure is not a 200 with an empty answer. */
    for (auto& g : gens) {
        if (g.finish == Finish::Error) {
            const std::string why = engine_failure(*g.sink);
            retire(gens);
            release(r.n_choices());
            record(r, gens, Finish::Error);
            ApiError e;
            e.status = 500;
            e.type = "internal_server_error";
            e.message = why;
            return error_response(e, req.peer);
        }
    }

    record(r, gens, overall);
    retire(gens);
    release(r.n_choices());

    const TimingsOut timings = timings_of(gens);
    std::string body = r.endpoint == Endpoint::Chat
                           ? chat_completion_body(r, choices, usage, timings)
                           : text_completion_body(r, choices, usage, timings);
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, std::move(body));
}

/* ------------------------------------------------------------------ streaming */

static std::string text_chunk(const OaiRequest& r, int index, const std::string& delta,
                              const char* finish_reason, const TimingsOut* t = nullptr) {
    nlohmann::ordered_json ch;
    ch["text"] = delta;
    ch["index"] = index;
    ch["logprobs"] = nullptr;
    if (finish_reason) ch["finish_reason"] = finish_reason; else ch["finish_reason"] = nullptr;
    nlohmann::ordered_json o;
    o["id"] = r.id;
    o["object"] = "text_completion";
    o["created"] = r.created;
    o["model"] = r.model;
    o["choices"] = nlohmann::ordered_json::array({ch});
    if (t && t->valid) o["timings"] = timings_json(*t);
    return sse_event(o.dump());
}

HttpResponse Server::streaming_completion(const HttpRequest& req, OaiRequest& r) {
    auto st = std::make_shared<StreamState>();
    st->r = r;
    st->peer_closed = req.closed;

    ApiError err;
    if (submit_all(st->r, &st->gens, &err) < 0) {
        retire(st->gens);
        release(r.n_choices());
        return error_response(err, req.peer);
    }
    if (st->r.endpoint == Endpoint::Chat)
        for (auto& g : st->gens)
            g.delta.reset(new ChatDeltaStream(st->r, g.index, st->r.reply_parser.get()));

    HttpResponse res;
    res.status = 200;
    res.content_type = "text/event-stream; charset=utf-8";
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    /* nginx buffers proxied responses by default, which turns a token stream into one large
     * delivery at the end. This header is the documented opt-out and costs nothing elsewhere. */
    res.set_header("X-Accel-Buffering", "no");

    Server* self = this;

    res.next = [self, st](std::string& out) -> bool {
        out.clear();
        if (st->done_sent) return false;

        if (!st->opened) {
            st->opened = true;
            if (st->r.endpoint == Endpoint::Chat) {
                for (auto& g : st->gens) out += g.delta->first_chunk();
            } else if (st->r.echo) {
                /* `echo`, ON THE STREAMING PATH. The buffered path applies it by prepending the
                 * prompt to the choice's text; this is the equivalent, and without it the same
                 * request would answer differently depending on `stream`.
                 *
                 * The prompt goes out as the first content chunk, before any token, which is what
                 * "echo" means and what a client reassembling the deltas needs: concatenating the
                 * stream gives the same string the buffered response returns.
                 * Empty prompts emit nothing rather than an empty chunk. */
                for (auto& g : st->gens) {
                    const std::string& p = st->r.prompts[(size_t)g.prompt_index].text;
                    if (!p.empty()) out += text_chunk(st->r, g.index, p, nullptr);
                }
            }
            return true;
        }

        if (st->peer_closed && st->peer_closed()) {
            return false;
        }

        const bool progress = self->pump(st->gens, st->r);
        size_t maxstop = 0;
        for (const auto& s : st->r.stop) maxstop = std::max(maxstop, s.size());

        for (auto& g : st->gens) {
            /* Hold back the longest stop string minus one byte while the choice is still running:
             * otherwise the first half of a stop sequence is already on the wire by the time we
             * recognise it, and it cannot be taken back.
             *
             * ONLY WHILE IT IS RUNNING. Every hold-back rule here waits for bytes that may still
             * arrive, and a closed choice has none coming, so its whole text is the answer. That
             * includes a tail the escape rule would keep -- a trailing `\` or a partial `\uXXXX`
             * is ordinary text in a completion, and holding it back at the end drops it. */
            size_t safe = g.text.size();
            if (!g.closed) {
                /* FF01: Detokenizer already holds back partial stop prefixes (tokenizer.cpp:1495-1506).
                 * Only apply fallback holdback if g.detok is absent. */
                if (!g.detok && maxstop > 1) safe = safe > maxstop - 1 ? safe - (maxstop - 1) : 0;
                safe = json_stable_prefix(std::string_view(g.text.data(), safe));
                if (safe <= g.emitted) continue;
            }

            if (st->r.endpoint == Endpoint::Chat) {
                /* Only what is new: the reader keeps its own place in the reply, which is what
                 * makes a long reply cost its length rather than its length squared. */
                out += g.delta->push(std::string_view(g.text).substr(g.emitted, safe - g.emitted),
                                     g.closed);
            } else if (safe > g.emitted) {
                out += text_chunk(st->r, g.index, g.text.substr(g.emitted, safe - g.emitted),
                                  nullptr);
            }
            g.emitted = safe;
        }

        if (self->all_done(st->gens)) {
            const UsageOut u = usage_of(st->r, st->gens);
            Finish overall = Finish::Stop;
            for (auto& g : st->gens) {
                /* EACH CHOICE'S OWN TIMINGS ON ITS FINAL CHUNK, so a client that asked for no
                 * usage chunk still has them, as llama.cpp sends them. */
                const Sink* one = g.sink.get();
                const TimingsOut gt = timings_of(&one, 1);
                if (st->r.endpoint == Endpoint::Chat) {
                    out += g.delta->final_chunk(g.finish, gt);
                    Finish adj = g.delta->adjusted_finish(g.finish);
                    if (adj == Finish::Error || adj == Finish::Cancelled) overall = adj;
                    else if (adj == Finish::ToolCalls) overall = adj;
                } else {
                    out += text_chunk(st->r, g.index, "", finish_reason_oai(g.finish), &gt);
                    if (g.finish == Finish::Error) overall = Finish::Error;
                }
                /* A failed generation cannot become a non-200 -- the headers left long ago -- so
                 * it travels as an error event before [DONE]. Silence would look like a normal
                 * end of stream and the client would use a truncated answer. */
                if (g.finish == Finish::Error) {
                    ApiError e;
                    e.status = 500;
                    e.type = "internal_server_error";
                    e.message = engine_failure(*g.sink);
                    out += sse_error(e);
                }
            }
            /* THE USAGE CHUNK. `stream_options.include_usage` is parsed and validated on both
             * endpoints, so both have to honour it -- accepting a field and then ignoring it is
             * the outcome oai.h's header forbids.
             *
             * After the per-generation final chunks and before [DONE], which is where OpenAI puts
             * it, and once for the request rather than once per choice: `n` completions share one
             * prompt and one total, so emitting it per generation would count the prompt n times
             * and interleave usage between the choices' final chunks. */
            if (st->r.include_usage) {
                const TimingsOut t = timings_of(st->gens);
                out += st->r.endpoint == Endpoint::Chat ? chat_usage_chunk(st->r, u, t)
                                                        : text_usage_chunk(st->r, u, t);
            }

            out += SSE_DONE;
            st->done_sent = true;
            self->record(st->r, st->gens, overall);
            self->retire(st->gens);
            self->metrics_.observe_http(200);
            if (!st->released) { st->released = true; self->release(st->r.n_choices()); }
            return false;
        }

        if (!progress && out.empty()) {
            /* Nothing to send: sleep on the first live sink rather than spinning. The bounded
             * wait is what makes the peer-closed check above run a few times a second. */
            for (auto& g : st->gens) if (!g.closed) { g.sink->wait(); break; }
        }
        return true;
    };

    res.on_close = [self, st](bool ok) {
        if (!st->done_sent) {
            /* Either the peer went away or the write failed. Both are cancellations, and the
             * blocks have to go back now rather than when the generation would have ended. */
            self->cancel_all(st->gens, true);
            self->record(st->r, st->gens, Finish::Cancelled);
            self->retire(st->gens);
            RAD_DEBUG("server: stream %s abandoned by the client (write %s)",
                      st->r.id.c_str(), ok ? "ok" : "failed");
        }
        if (!st->released) { st->released = true; self->release(st->r.n_choices()); }
    };

    return res;
}

/* ------------------------------------------------------------------ endpoints */

HttpResponse Server::handle_chat(const HttpRequest& req) {
    if (!deps_.sched || !deps_.tok)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);

    OaiRequest r;
    ApiError err;
    if (parse_chat_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    r.peer = req.peer;

    sweep_abandoned();
    if (!admit(r.n_choices())) {
        metrics_.observe_rejected();
        return error_response(err_overloaded(
            "the admission queue is full; retry after " + std::to_string(opt_.retry_after_s) +
            "s. The server refuses rather than queueing without bound"), req.peer);
    }
    log_admitted(r);
    return r.stream ? streaming_completion(req, r) : blocking_completion(req, r);
}

HttpResponse Server::handle_completion(const HttpRequest& req) {
    if (!deps_.sched || !deps_.tok)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);

    OaiRequest r;
    ApiError err;
    if (parse_completion_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    r.peer = req.peer;

    sweep_abandoned();
    if (!admit(r.n_choices())) {
        metrics_.observe_rejected();
        return error_response(err_overloaded(
            "the admission queue is full; retry after " + std::to_string(opt_.retry_after_s) +
            "s. The server refuses rather than queueing without bound"), req.peer);
    }
    log_admitted(r);
    return r.stream ? streaming_completion(req, r) : blocking_completion(req, r);
}

HttpResponse Server::handle_embeddings(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);
    if (!deps_.embed)
        return error_response(err_not_implemented(
            "/v1/embeddings requires an embedding head; this model was not loaded with one"),
                req.peer);

    OaiRequest r;
    ApiError err;
    if (parse_embeddings_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);

    std::vector<std::vector<float>> vecs;
    UsageOut u;
    vecs.reserve(r.prompts.size());
    for (const auto& p : r.prompts) {
        std::vector<float> v;
        int s = deps_.embed->embed(p.tokens, v);
        if (s < 0) {
            ApiError e;
            e.status = 500;
            e.type = "internal_server_error";
            e.message = std::string("the embedding pass failed: ") + rad_strerror(s);
            return error_response(e, req.peer);
        }
        u.prompt += (int64_t)p.tokens.size();
        vecs.push_back(std::move(v));
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, embeddings_body(r, vecs, u));
}

HttpResponse Server::handle_models(const HttpRequest& req) {
    const std::string id = req.param("id");
    if (!id.empty() && id != opt_.model_id) {
        ApiError e;
        e.status = 404;
        e.type = "not_found_error";
        e.param = "model";
        e.message = "no such model: " + id;
        return error_response(e, req.peer);
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, models_body(opt_.model_id, started_at_));
}

HttpResponse Server::handle_metrics(const HttpRequest&) {
    if (deps_.sched) metrics_.sample(deps_.sched->metrics());
    metrics_.set_queue_depth(in_flight_.load(std::memory_order_relaxed));
    HttpResponse r;
    r.status = 200;
    /* The exposition format version is part of the content type; Prometheus uses it to decide how
     * to parse, and omitting it makes some scrapers fall back to OpenMetrics and fail. */
    r.content_type = "text/plain; version=0.0.4; charset=utf-8";
    r.body = metrics_.render();
    return r;
}

}  /* namespace server */
}  /* namespace rad */
