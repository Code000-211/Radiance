#include "server.h"
#include "server_priv.h"

#include <cstdio>
#include "json_partial.h"
#include "systemone.h"
#include "rad_internal.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

namespace rad {
namespace server {

/* ------------------------------------------------------------------ construction */

Server::Server(const Deps& d, const ServerOptions& o)
    : deps_(d), opt_(o),
      metrics_(o.model_id, o.max_ctx > 0 ? o.max_ctx : 32768) {
    lim_.model_id = opt_.model_id;
    lim_.version = opt_.version;
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
    lim_.reasoning_effort_map = opt_.reasoning_effort_map;
    lim_.reasoning_format = opt_.reasoning_format;

    odeps_.tok = deps_.tok;
    odeps_.chat = deps_.chat;
    odeps_.grammar = deps_.grammar;
    odeps_.mm = deps_.mm;
    odeps_.embed = deps_.embed;

    queue_depth_ = opt_.queue_depth > 0 ? opt_.queue_depth : opt_.max_seqs * 8;
    started_at_ = unix_now();
    register_routes();
}

Server::~Server() { stop(); }

/* A sink may be freed only after the scheduler reports its request terminal, since
 * `Request::sink` is a raw pointer the scheduler holds for the request's life. At shutdown this
 * waits a bounded time, then deliberately leaks the remainder (a few KiB at exit) rather than
 * free memory a running step may write into, which would corrupt an unrelated allocation
 * (spec §17). */
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

/* Every route, in one table that /server_info also reports.
 *
 * Methods follow vLLM's: /tokenize and /detokenize are POST (their input is a body),
 * /reset_prefix_cache is POST (it changes state), and /ping takes GET and POST because load
 * balancers differ. */
void Server::register_routes() {
    router_.post("/v1/chat/completions", [this](const HttpRequest& q) { return handle_chat(q); });
    router_.post("/v1/completions",      [this](const HttpRequest& q) { return handle_completion(q); });
    router_.post("/v1/embeddings",       [this](const HttpRequest& q) { return handle_embeddings(q); });
    router_.post("/v1/images/generations", [this](const HttpRequest& q) { return handle_images(q, false); });
    router_.post("/v1/images/edits",     [this](const HttpRequest& q) { return handle_images(q, true); });
    router_.post("/v1/systemone",        [this](const HttpRequest& q) { return handle_systemone(q); });
    /* Speech to text, for a model whose plugin says how it transcribes (RadAudioFormat). */
    router_.post("/v1/audio/speech", [this](const HttpRequest& q) { return handle_speech(q); });
    router_.post("/v1/videos",       [this](const HttpRequest& q) { return handle_video_create(q); });
    router_.get ("/v1/videos",       [this](const HttpRequest& q) { return handle_video_list(q); });
    router_.get ("/v1/videos/:id",   [this](const HttpRequest& q) { return handle_video_get(q); });
    router_.add ("DELETE", "/v1/videos/:id",
                 [this](const HttpRequest& q) { return handle_video_delete(q); });
    router_.get ("/v1/videos/:id/content",
                 [this](const HttpRequest& q) { return handle_video_content(q); });
    router_.post("/v1/audio/transcriptions",
                 [this](const HttpRequest& q) { return handle_transcription(q); });
    /* The same, live over a WebSocket (OpenAI's realtime transcription session). A plain GET
     * there is told what the path wants. */
    router_.socket("/v1/realtime", [this](const HttpRequest& q, WsConn& c) { handle_realtime(q, c); });
    router_.get ("/v1/realtime", [this](const HttpRequest& q) {
        ApiError e;
        e.status = 426;
        e.type = "invalid_request_error";
        e.message = "/v1/realtime is a WebSocket: connect with an Upgrade: websocket request";
        HttpResponse r = error_response(e, q.peer);
        r.set_header("Upgrade", "websocket");
        return r;
    });
    /* Speech made while its text is still being written (vLLM-Omni's realtime speech session). */
    router_.socket("/v1/audio/speech/stream",
                   [this](const HttpRequest& q, WsConn& c) { handle_speech_stream(q, c); });
    router_.get ("/v1/audio/speech/stream", [this](const HttpRequest& q) {
        ApiError e;
        e.status = 426;
        e.type = "invalid_request_error";
        e.message = "/v1/audio/speech/stream is a WebSocket: connect with an Upgrade: websocket "
                    "request";
        HttpResponse r = error_response(e, q.peer);
        r.set_header("Upgrade", "websocket");
        return r;
    });
    router_.get ("/v1/models",           [this](const HttpRequest& q) { return handle_models(q); });
    router_.get ("/v1/loras",            [this](const HttpRequest& q) { return handle_loras(q); });
    router_.get ("/v1/models/*id",       [this](const HttpRequest& q) { return handle_models(q); });
    router_.post("/v1/load_lora_adapter",   [this](const HttpRequest& q) { return handle_lora_admin(q, true); });
    router_.post("/v1/unload_lora_adapter", [this](const HttpRequest& q) { return handle_lora_admin(q, false); });
    /* The same routes without /v1, as llama-server answers them, for clients whose base URL is
     * the bare host. */
    router_.post("/chat/completions",    [this](const HttpRequest& q) { return handle_chat(q); });
    /* llama-server's native completion under both its names; /completions without /v1 is the
     * native endpoint there, not OpenAI's. */
    router_.post("/completion",          [this](const HttpRequest& q) { return handle_native_completion(q); });
    router_.post("/completions",         [this](const HttpRequest& q) { return handle_native_completion(q); });
    router_.post("/infill",              [this](const HttpRequest& q) { return handle_infill(q); });
    router_.post("/embeddings",          [this](const HttpRequest& q) { return handle_embeddings(q); });
    router_.post("/images/generations",  [this](const HttpRequest& q) { return handle_images(q, false); });
    router_.post("/images/edits",        [this](const HttpRequest& q) { return handle_images(q, true); });
    router_.post("/audio/transcriptions", [this](const HttpRequest& q) { return handle_transcription(q); });
    router_.post("/audio/speech",        [this](const HttpRequest& q) { return handle_speech(q); });
    router_.get ("/models",              [this](const HttpRequest& q) { return handle_models(q); });
    /* llama-server's: the prompt a chat request renders to, and how many tokens it is. */
    router_.post("/apply-template",      [this](const HttpRequest& q) { return handle_apply_template(q); });
    auto chat_tokens = [this](const HttpRequest& q) {
        return count_tokens(q, nullptr, ApiShape::OpenAI, "response.input_tokens");
    };
    router_.post("/v1/chat/completions/input_tokens", chat_tokens);
    router_.post("/chat/completions/input_tokens", chat_tokens);
    /* Anthropic's Messages API, translated onto the chat path (anthropic_to_chat). */
    router_.post("/v1/messages",         [this](const HttpRequest& q) { return handle_messages(q); });
    router_.post("/v1/messages/count_tokens", [this](const HttpRequest& q) {
        return count_tokens(q,
                            [](const std::string& b, const OaiLimits& l, std::string* c,
                               std::string* lb, ApiError* e) {
                                return anthropic_to_chat(b, l, c, lb, e);
                            },
                            ApiShape::Anthropic, nullptr);
    });
    /* OpenAI's Responses API, translated the same way (responses_to_chat). */
    router_.post("/v1/responses",        [this](const HttpRequest& q) { return handle_responses(q); });
    router_.post("/responses",           [this](const HttpRequest& q) { return handle_responses(q); });
    auto responses_tokens = [this](const HttpRequest& q) {
        return count_tokens(q,
                            [](const std::string& b, const OaiLimits& l, std::string* c,
                               std::string* ec, ApiError* e) {
                                return responses_to_chat(b, l, c, ec, e);
                            },
                            ApiShape::OpenAI, "response.input_tokens");
    };
    router_.post("/v1/responses/input_tokens", responses_tokens);
    router_.post("/responses/input_tokens", responses_tokens);

    router_.get ("/metrics",             [this](const HttpRequest& q) { return handle_metrics(q); });
    router_.get ("/health",              [this](const HttpRequest& q) { return handle_health(q); });
    router_.get ("/v1/health",           [this](const HttpRequest& q) { return handle_health(q); });
    router_.get ("/ping",                [this](const HttpRequest& q) { return handle_ping(q); });
    router_.post("/ping",                [this](const HttpRequest& q) { return handle_ping(q); });
    router_.get ("/load",                [this](const HttpRequest& q) { return handle_load(q); });
    router_.get ("/version",             [this](const HttpRequest& q) { return handle_version(q); });
    router_.get ("/server_info",         [this](const HttpRequest& q) { return handle_server_info(q); });
    router_.get ("/props",               [this](const HttpRequest& q) { return handle_props(q); });
    router_.get ("/diagnostics",         [this](const HttpRequest& q) { return handle_diagnostics(q); });

    router_.post("/tokenize",            [this](const HttpRequest& q) { return handle_tokenize(q); });
    router_.post("/detokenize",          [this](const HttpRequest& q) { return handle_detokenize(q); });
    router_.get ("/get_tokenizer_info",  [this](const HttpRequest& q) { return handle_tokenizer_info(q); });

    router_.post("/reset_prefix_cache",  [this](const HttpRequest& q) { return handle_reset_prefix_cache(q); });
    /* A calibration's phase boundary, for the recording harness. Under the same API-key gate as
     * every other state-changing route (http.cpp); a deployment without --draft-profile-out
     * answers 501. */
    router_.post("/admin/draft-calib-boundary",
                 [this](const HttpRequest& q) { return handle_draft_calib_boundary(q); });

    /* The dashboard, also at `/`, since that is what a person with the host and port types. */
    router_.get ("/",                    [this](const HttpRequest& q) { return handle_dashboard(q); });
    router_.get ("/dashboard",           [this](const HttpRequest& q) { return handle_dashboard(q); });
    router_.get ("/stats",               [this](const HttpRequest& q) { return handle_stats(q); });
    router_.get ("/graph",               [this](const HttpRequest& q) { return handle_graph(q); });
    router_.get ("/sessions",            [this](const HttpRequest& q) { return handle_sessions(q); });
    /* The control vector's switch, for the dashboard's Model view and for automation: a
     * deployment with no vector found answers 404 here. */
    router_.get ("/admin/control-vector",
                 [this](const HttpRequest& q) { return handle_control_vector(q); });
    router_.post("/admin/control-vector",
                 [this](const HttpRequest& q) { return handle_control_vector(q); });
}

/* Workers beyond the admission ceiling. A generation holds its connection's worker for its whole
 * run, so a pool capped at the ceiling would run out before admission refuses anyone: no 429, and
 * every later connection (/health included) queued behind minutes-long generations. The reserve
 * keeps probes, the dashboard, 429 answers and idle keep-alive connections serviceable. The same
 * number bounds the connections that may wait once the reserve is busy too. */
static const int64_t kSpareWorkers = 64;

void Server::note_ready(uint64_t begin_ns) {
    if (begin_ns == 0) return;
    const uint64_t now = rad_mono_ns();
    /* A stamp that is not in the past leaves the figure unset rather than writing a zero. */
    if (now <= begin_ns) return;
    startup_s_ = (int64_t)((now - begin_ns) / 1000000000ull);
}

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
    l.probe = true;
    if (!render(l, nullptr)) return;
    l.default_template_kwargs = lim_.default_template_kwargs;
    if (!l.default_template_kwargs.empty() && !render(l, kwargs_why)) return;
    l.default_reasoning_effort = lim_.default_reasoning_effort;
    if (!l.default_reasoning_effort.empty()) render(l, effort_why);
}

std::vector<std::string> Server::probe_reasoning_efforts() {
    lim_.reasoning_efforts.clear();
    if (!deps_.chat) return {};
    OaiLimits l = lim_;
    l.default_template_kwargs.clear();
    l.default_reasoning_effort.clear();
    l.reasoning_effort_map.clear();
    l.probe = true;
    template_default_effort_.clear();
    auto render = [&](const std::string& effort, std::string* text) {
        std::string body = R"({"messages":[{"role":"user","content":"Hello"}],"max_tokens":1)";
        if (!effort.empty()) body += R"(,"reasoning_effort":)" + json(effort).dump();
        body += "}";
        OaiRequest r;
        ApiError e;
        if (parse_chat_request(body, odeps_, l, &r, &e) < 0) return false;
        if (text) *text = r.prompts.empty() ? std::string() : r.prompts[0].text;
        return true;
    };
    /* A render that fails without an effort says nothing about efforts; one that takes a
     * nonsense word takes every word. */
    std::string plain;
    if (!render("", &plain) || render("radiance-probe-not-a-level", nullptr)) return {};
    std::vector<std::string> set, same;
    for (const char* w : { "low", "medium", "high", "xhigh", "max" }) {
        std::string text;
        if (!render(w, &text)) continue;
        set.push_back(w);
        if (text == plain) same.push_back(w);
    }
    /* The level a request naming none gets: the one level whose render is the same. A template may
     * spend it in a system prompt the client never sees (#113: Qwen3.8's is xhigh). Several, or
     * none, say nothing. */
    if (same.size() == 1) template_default_effort_ = same[0];
    lim_.reasoning_efforts = set;
    return set;
}

std::string Server::check_reasoning_effort_map() const {
    if (!deps_.chat) return std::string();
    OaiLimits l = lim_;
    l.default_template_kwargs.clear();
    l.default_reasoning_effort.clear();
    l.reasoning_effort_map.clear();
    l.probe = true;
    for (const auto& [from, to] : lim_.reasoning_effort_map) {
        const std::string body = R"({"messages":[{"role":"user","content":"Hello"}],"max_tokens":1,)"
                                 R"("reasoning_effort":)" + json(to).dump() + "}";
        OaiRequest r;
        ApiError e;
        if (parse_chat_request(body, odeps_, l, &r, &e) < 0)
            return from + "=" + to + ": the chat template refuses '" + to + "': " + e.message;
    }
    return std::string();
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
    /* Before the transport drains, not after: a handler that is waiting for the calibration
     * boundary would otherwise keep the drain sitting on its bounded wait, and the stop would look
     * like a hang for as long as a step loop that is already gone never answers. */
    if (deps_.sched) deps_.sched->draft_calib_close();
    transport_.stop();
    stop_videos();
    drain_graveyard(2000);
}

/* ------------------------------------------------------------------ admission */

std::optional<HttpResponse> Server::admit(int64_t n, const std::string& peer, ApiShape shape,
                                          const char* field) {
    sweep_abandoned();
    if (n > queue_depth_)
        return error_response(err_bad(field, "this request needs " + std::to_string(n) +
                                                 " admission places, and this server admits at "
                                                 "most " + std::to_string(queue_depth_) +
                                                 " requests at once (--max-queued-requests); "
                                                 "split it"),
                              peer, shape);
    int64_t cur = in_flight_.fetch_add(n, std::memory_order_acq_rel) + n;
    if (cur > queue_depth_) {
        in_flight_.fetch_sub(n, std::memory_order_acq_rel);
        metrics_.observe_rejected();
        return error_response(err_overloaded("the admission queue is full; retry after " +
                                             std::to_string(opt_.retry_after_s) + "s"),
                              peer, shape);
    }
    metrics_.set_queue_depth(cur);
    return std::nullopt;
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

/* " from <addr:port>" for the request log, or "" in-process. */
static std::string from_peer(const std::string& peer) {
    return peer.empty() ? std::string() : " from " + peer;
}

HttpResponse Server::error_response(const ApiError& e, const std::string& peer, ApiShape shape) {
    metrics_.observe_http(e.status);
    /* Log every refusal with status, type, message and peer: 4xx at info (the client's doing),
     * 5xx as a warning (ours). */
    if (e.status >= 500)
        RAD_WARN("req refused%s -> %d %s: %s", from_peer(peer).c_str(), e.status, e.type.c_str(),
                 e.message.c_str());
    else
        RAD_INFO("req refused%s -> %d %s: %s", from_peer(peer).c_str(), e.status, e.type.c_str(),
                 e.message.c_str());
    HttpResponse r = HttpResponse::json_body(
        e.status, shape == ApiShape::Anthropic ? anthropic_error_body(e) : e.dump());
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

    /* The sampler vets the parameters before anything is queued (IScheduler::admit), so an
     * unresolved stage is a 400 naming the parameter rather than a mid-stream cancellation.
     * Once, not per choice: `n` varies only the seed, which no check reads. */
    {
        std::string why;
        if (deps_.sched->admit(r.sp, &why) < 0) {
            *err = err_unsupported("sampling", why.empty()
                                       ? "these sampling parameters cannot be served" : why);
            return RAD_E_INVAL;
        }
    }

    /* Runtime LoRA: `model` picks the adapter. The served model's own name, or no name, is the
     * base; with adapters enabled anything else must be a loaded adapter, since a typo served by
     * the base would answer as the wrong model without a word. */
    int32_t  lora_slot = -1;
    uint64_t lora_uid = 0;
    std::shared_ptr<void> lora_hold;
    if (opt_.lora_acquire && !r.model.empty() && r.model != opt_.model_id) {
        if (opt_.lora_acquire(r.model, &lora_slot, &lora_uid, &lora_hold) != RAD_OK) {
            std::string names = opt_.model_id;
            if (opt_.lora_names)
                for (const std::string& n : opt_.lora_names()) names += ", " + n;
            ApiError e;
            e.status = 404;
            e.type = "not_found_error";
            e.param = "model";
            e.message = "no such model or adapter: " + r.model + " (served: " + names + ")";
            *err = e;
            return RAD_E_NOTFOUND;
        }
    }

    /* The control vector: the request's own choice, else the deployment's runtime default. A
     * deployment with no vector refuses the on state by name -- serving the stock model while the
     * client believes it asked for the vector would be a silent lie about what produced the text.
     * (Naming: `control_vector` is the field, `experimental_speed_projection` its alias.) */
    bool projection_on = false;
    {
        const bool named = r.control_vector >= 0;
        projection_on = named
            ? (r.control_vector == 1)
            : (deps_.control_vector.get_default && deps_.control_vector.get_default());
        if (projection_on &&
            !(deps_.control_vector.available && deps_.control_vector.available())) {
            *err = err_bad("control_vector",
                           "this deployment found no control vector for its model; the stock model "
                           "is what it serves");
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
            if (r.want_logprobs) g.sink->enable_logprobs(r.n_logprobs);
            g.think.start(r.sp.reasoning);
            g.detok = deps_.tok->detokenizer(r.stop, r.preserved_tokens);
            if (!g.detok) {
                *err = err_not_implemented("the tokenizer did not supply an incremental decoder");
                return RAD_E_UNSUPPORTED;
            }
            if (!r.stops_after.empty()) g.detok->defer_stops_until(r.stops_after);

            Request rq;
            rq.prompt = r.prompts[p].tokens;
            rq.sp = r.sp;
            rq.sp.use_control_vector = projection_on;
            /* Distinct seeds per choice, or `n` identical completions. An offset suffices: the
             * sampler's stream is seeded, not chained. */
            rq.sp.seed = r.sp.seed + (uint64_t)k;
            rq.max_tokens = r.max_tokens;
            /* With stop strings deferred past the reasoning block, the scheduler (which cannot
             * tell reasoning from answer) gets none; the decoder and pump() cut instead. */
            rq.stop = r.stops_after.empty() ? r.stop : std::vector<std::string>{};
            rq.ignore_eos = r.ignore_eos;
            rq.priority = r.priority;
            rq.stream = r.stream;
            rq.media = r.prompts[p].media;
            rq.session_label = r.session_label;
            rq.lora_slot = lora_slot;
            rq.lora_uid = lora_uid;
            g.lora_hold = lora_hold;
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
    /* Refused because the prompt is longer than the KV cache can ever hold (RAD_E_FULL at
     * submission): the client's error, answered now as a context overflow, with the choices
     * the scheduler did take cancelled. */
    for (size_t i = 0; i < ids.size(); ++i) {
        const Generation& g = (*gens)[first + i];
        if (ids[i] != 0 || g.sink->status() != RAD_E_FULL) continue;
        const std::string why = g.sink->why() ? g.sink->why() : "the prompt is too long";
        cancel_all(*gens, false);
        *err = err_bad("prompt", why);
        err->message = why;
        err->code = "context_length_exceeded";
        err->n_prompt_tokens = (int64_t)r.prompts[(size_t)g.prompt_index].tokens.size();
        return RAD_E_FULL;
    }
    return RAD_OK;
}

std::optional<HttpResponse> Server::abandoned(const HttpRequest& req,
                                              std::vector<Generation>& gens, int64_t n,
                                              ApiShape shape, const OaiRequest* r) {
    const bool gone = req.peer_gone();
    if (!gone && !stopping_.load()) return std::nullopt;
    cancel_all(gens, gone);
    if (r) record(*r, gens, Finish::Cancelled);
    retire(gens);
    release(n);
    if (!gone)
        return error_response(err_overloaded("the server is shutting down"), req.peer, shape);
    metrics_.observe_http(499);
    return HttpResponse::text(499, "client closed request\n");
}

void Server::cancel_all(std::vector<Generation>& gens, bool count_disconnect) {
    for (auto& g : gens) {
        if (g.sink->done()) continue;
        g.sink->request_cancel();
        deps_.sched->cancel(g.id);
    }
    if (count_disconnect) metrics_.observe_disconnect();
}

/* Every exit from a request goes through here exactly once. A finished generation is reaped
 * (the scheduler may drop the Request; only the server knows the sink is released). An
 * unfinished one moves to the graveyard and is reaped by the sweep once it finishes. Id 0 is a
 * request the scheduler refused or never received: it holds nothing, so there is nothing to
 * reap and nothing to wait for. */
void Server::retire(std::vector<Generation>& gens) {
    std::vector<std::shared_ptr<Sink>> pending;
    for (auto& g : gens) {
        if (g.retired) continue;
        g.retired = true;
        if (!g.id) continue;
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

/* A token's logprobs entry text: the bytes it adds to the reply, and the readable prefix of them. */
static void lp_text(const ITokenizer& tok, int32_t id, std::string* text, std::string* bytes) {
    *bytes = tok.reply_bytes(id);
    *text = bytes->substr(0, utf8_valid_prefix(*bytes));
}

/* How many entries from `from` went out with the reply: with a stop cut, every entry whose text
 * began before the cut (a straddling token contributed text the client got); otherwise all. */
static size_t lp_before_cut(const std::vector<LogprobEntry>& lp, size_t from, size_t cut) {
    if (cut == std::string::npos) return lp.size() - from;
    size_t start = from ? lp[from - 1].end : 0, n = 0;
    for (size_t k = from; k < lp.size() && start < cut; ++k, ++n) start = lp[k].end;
    return n;
}

static std::vector<LogprobEntry> kept_logprobs(const std::vector<LogprobEntry>& lp, size_t cut) {
    return std::vector<LogprobEntry>(lp.begin(), lp.begin() + (long)lp_before_cut(lp, 0, cut));
}

bool Server::pump(std::vector<Generation>& gens, const OaiRequest& r) {
    bool progress = false;
    SinkToken buf[128];
    /* Each read's alternatives; only a request that asked for them sizes this. */
    thread_local std::vector<SinkAlt> alts;
    if (r.want_logprobs) alts.resize(128 * (size_t)std::max(1, r.n_logprobs));

    size_t maxstop = 0;
    for (const auto& s : r.stop) maxstop = std::max(maxstop, s.size());

    for (auto& g : gens) {
        if (g.closed) continue;

        /* Read before the drain: the producer pushes its last tokens and then finishes, so once
         * the finish is seen every token is in the sink and the drain below takes them all. Read
         * after it, a finish landing between the last read and the check would close the choice
         * with those tokens unread, and the reply would lose its end. */
        const bool finished = g.sink->done();

        size_t n;
        while ((n = g.sink->read(buf, 128, r.want_logprobs ? alts.data() : nullptr)) > 0) {
            progress = true;
            for (size_t i = 0; i < n; ++i) {
                g.think.push(r.sp.reasoning, buf[i].token);
                if (r.shape == ApiShape::LlamaNative || r.return_tokens)
                    g.tokens.push_back(buf[i].token);
                std::string piece = g.detok->push(buf[i].token);
                if (!piece.empty()) g.text += piece;
                if (!r.want_logprobs) continue;
                LogprobEntry e;
                e.token = buf[i].token;
                e.logprob = buf[i].logprob;
                e.end = g.text.size();
                lp_text(*deps_.tok, e.token, &e.text, &e.bytes);
                const SinkAlt* a = alts.data() + i * (size_t)g.sink->alt_width();
                for (int32_t j = 0; j < buf[i].n_alt; ++j) {
                    LogprobAlt t;
                    t.token = a[j].id;
                    t.logprob = a[j].logprob;
                    lp_text(*deps_.tok, t.token, &t.text, &t.bytes);
                    e.top.push_back(std::move(t));
                }
                g.logprobs.push_back(std::move(e));
            }
        }

        /* The decoder may have swallowed the stop string itself; the scan below cannot find it
         * in text it was already removed from. */
        if (maxstop && !g.closed && g.stop_cut == std::string::npos && g.detok->stopped()) {
            g.stop_cut = g.text.size();
            g.stop_word = g.detok->stop_hit();
            g.finish = Finish::StopString;
            g.closed = true;
            progress = true;
            g.sink->request_cancel();
            deps_.sched->cancel(g.id);
            continue;
        }

        /* Stop strings. The scheduler gets Request::stop and may stop first, but a stop string
         * can straddle tokens, so the authoritative cut is made here on the detokenised text.
         * Scanning restarts (maxstop - 1) bytes behind the previous end so a match spanning two
         * drains is found.
         *
         * Stop strings deferred past the reasoning block match only after its closer, which is
         * searched incrementally from (closer length - 1) bytes behind the previous end. */
        size_t floor = 0;
        if (!r.stops_after.empty()) {
            if (g.stops_floor == std::string::npos) {
                const size_t n = r.stops_after.size();
                const size_t from = g.closer_scanned >= n ? g.closer_scanned - n + 1 : 0;
                const size_t m = g.text.find(r.stops_after, from);
                g.closer_scanned = g.text.size();
                if (m != std::string::npos) g.stops_floor = m + n;
            }
            floor = g.stops_floor;
        }
        if (maxstop && floor != std::string::npos && g.stop_cut == std::string::npos &&
            !g.text.empty()) {
            size_t start = g.scanned >= maxstop ? g.scanned - maxstop + 1 : 0;
            if (start < floor) start = floor;
            for (const auto& s : r.stop) {
                if (s.empty()) continue;
                size_t at = g.text.find(s, start);
                if (at != std::string::npos && (g.stop_cut == std::string::npos || at < g.stop_cut)) {
                    g.stop_cut = at;
                    g.stop_word = s;
                }
            }
            g.scanned = g.text.size();
            if (g.stop_cut != std::string::npos) {
                g.text.resize(g.stop_cut);
                g.finish = Finish::StopString;
                g.closed = true;
                progress = true;
                /* No further token reaches the client, so free its blocks now (spec §14). */
                g.sink->request_cancel();
                deps_.sched->cancel(g.id);
                continue;
            }
        }

        if (finished) {
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

/* The request's token usage, shared by both paths. `n` choices share a prompt, so prompt and
 * cached tokens count once per prompt. Reasoning is reported only when the block can be counted
 * by token (ReasoningSpan); otherwise no breakdown rather than a misleading zero. */
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

/* Where the request's time went, from its sinks (one for a choice's final chunk, all for the
 * request). Each sink counts from its own construction, so times are moved onto the steady clock
 * first; `received_ns` (body arrival, -1 when unknown) is already on it. Fields: TimingsOut. */
TimingsOut Server::timings_of(const Sink* const* sinks, size_t n, int64_t received_ns) {
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
    if (received_ns >= 0) {
        t.ttft_ms  = (double)(first_lo - received_ns) * 1e-6;
        t.total_ms = (double)(last - received_ns) * 1e-6;
    }
    return t;
}

TimingsOut Server::timings_of(const std::vector<Generation>& gens, int64_t received_ns) {
    std::vector<const Sink*> v;
    v.reserve(gens.size());
    for (const auto& g : gens) v.push_back(g.sink.get());
    return timings_of(v.data(), v.size(), received_ns);
}

/* ------------------------------------------------------------------ metrics bookkeeping */

/* A request's outcome over its choices, for record() and /metrics: an error over a cancellation
 * over a cut at max_tokens over a tool call over a plain stop. */
static Finish outcome(Finish overall, Finish f) {
    auto rank = [](Finish x) {
        switch (x) {
            case Finish::Error:     return 4;
            case Finish::Cancelled: return 3;
            case Finish::Length:    return 2;
            case Finish::ToolCalls: return 1;
            default:                return 0;
        }
    };
    return rank(f) > rank(overall) ? f : overall;
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

void Server::record(const OaiRequest& r, const std::vector<Generation>& gens, Finish overall,
                    bool quiet) {
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

    /* The request log's closing line: this request's share of what /metrics aggregates (finish,
     * cached prompt, reasoning tokens when known, TTFT, decode rate, total time). */
    const TimingsOut t = timings_of(gens, r.t_received_ns);
    const UsageOut u = usage_of(r, gens);
    const double rate = (decode > 0.0 && n_gen > 1) ? (double)(n_gen - 1) / decode : -1.0;
    char reason[32] = "";
    if (u.reasoning >= 0) std::snprintf(reason, sizeof reason, " reason=%lld", (long long)u.reasoning);
    char ttft_s[32] = "ttft=-";
    if (ttft >= 0) std::snprintf(ttft_s, sizeof ttft_s, "ttft=%.2fs", ttft);
    char rate_s[32] = "";
    if (rate > 0) std::snprintf(rate_s, sizeof rate_s, " %.1f tok/s", rate);
    RAD_LOG_AT(quiet ? Log::Debug : Log::Info,
               "req %s done%s finish=%s prompt=%lld (cache %lld) gen=%lld%s %s%s %.2fs",
               r.id.c_str(), from_peer(r.peer).c_str(), finish_reason_oai(overall),
               (long long)n_prompt, (long long)(t.valid ? t.cache_n : 0), (long long)n_gen,
               reason, ttft_s, rate_s, e2e);
}

/* ------------------------------------------------------------------ non-streaming */

HttpResponse Server::blocking_completion(const HttpRequest& req, OaiRequest& r) {
    r.t_received_ns = req.t_received_ns;
    std::vector<Generation> gens;
    ApiError err;
    if (submit_all(r, &gens, &err) < 0) {
        retire(gens);
        release(r.n_choices());
        return error_response(err, req.peer, r.shape);
    }

    for (;;) {
        pump(gens, r);
        if (all_done(gens)) break;

        /* A client that hangs up before the answer is ready cancels the request. */
        if (auto no = abandoned(req, gens, r.n_choices(), r.shape, &r)) return *no;
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
        c.stop_word = g.stop_word;
        if (r.want_logprobs) c.logprobs = kept_logprobs(g.logprobs, g.stop_cut);

        if (r.endpoint == Endpoint::Chat) {
            /* Parse whenever there is a parser, not only with tools: it also splits
             * `reasoning_content` from the answer. It is the reader the streaming path feeds
             * piecewise, fed the whole reply, so both paths agree. Without tools it reads no
             * calls. */
            if (r.reply_parser) {
                std::unique_ptr<IReplyReader> rd = r.reply_parser->open();
                rd->feed(g.text, nullptr);
                rd->finish(nullptr);
                c.msg = rd->message();
                /* Claim a tool call only when every call closed and its arguments parse
                 * (json_partial.h), the streaming path's rule (calls_complete). */
                const bool complete = r.parse_tools && calls_complete(c.msg);
                if (complete && (c.finish == Finish::Stop || c.finish == Finish::StopString))
                    c.finish = Finish::ToolCalls;
                if (c.msg.open_call) {
                    /* A half-finished call (necessarily the last) becomes content, so no tool is
                     * invoked with truncated arguments. Reasoning and earlier closed calls are
                     * kept, as the stream announced them; the finish reason is the stream's. */
                    c.msg.content = rd->content_with_open_call(g.text);
                    c.msg.tool_calls.pop_back();
                    c.msg.open_call = false;
                }
            } else {
                c.msg.content = g.text;
            }
            for (auto& tc : c.msg.tool_calls) if (tc.id.empty()) tc.id = random_id("call_");
        } else {
            c.text = r.echo ? r.prompts[g.prompt_index].text + g.text : g.text;
        }
        overall = outcome(overall, c.finish);
        choices.push_back(std::move(c));
    }

    /* An engine-side failure is not a 200 with an empty answer. */
    for (auto& g : gens) {
        if (g.finish == Finish::Error) {
            const std::string why = engine_failure(*g.sink);
            retire(gens);
            release(r.n_choices());
            record(r, gens, Finish::Error);
            return error_response(err_internal(why), req.peer, r.shape);
        }
    }

    record(r, gens, overall);
    retire(gens);
    release(r.n_choices());

    const TimingsOut timings = timings_of(gens, r.t_received_ns);
    if (r.shape == ApiShape::LlamaNative) {
        /* llama-server answers one result, and an array of them for several prompts. */
        json res = json::array();
        for (const auto& g : gens) res.push_back(native_result(r, g, g.text));
        metrics_.observe_http(200);
        return HttpResponse::json_body(200, json_dump(res.size() == 1 ? res[0] : res));
    }
    std::string body = r.shape == ApiShape::Anthropic
                           ? anthropic_message_body(r, choices.front(), usage)
                       : r.shape == ApiShape::Responses
                           ? responses_body(r, choices.front(), usage)
                       : r.endpoint == Endpoint::Chat
                           ? chat_completion_body(r, choices, usage, timings)
                           : text_completion_body(r, choices, usage, timings);
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, std::move(body));
}

/* ------------------------------------------------------------------ llama-server's native shape */

/* What a request was served with, under llama-server's names: its `generation_settings`. */
static nlohmann::ordered_json native_settings(const OaiRequest& r) {
    const SamplingParams& sp = r.sp;
    nlohmann::ordered_json p;
    p["seed"]               = sp.seed;
    p["temperature"]        = sp.temp;
    p["top_k"]              = sp.top_k;
    p["top_p"]              = sp.top_p;
    p["min_p"]              = sp.min_p;
    p["typical_p"]          = sp.typical_p;
    p["xtc_probability"]    = sp.xtc_probability;
    p["xtc_threshold"]      = sp.xtc_threshold;
    p["repeat_last_n"]      = sp.penalty_last_n;
    p["repeat_penalty"]     = sp.rep_penalty;
    p["presence_penalty"]   = sp.pres_penalty;
    p["frequency_penalty"]  = sp.freq_penalty;
    p["dry_multiplier"]     = sp.dry_multiplier;
    p["dry_base"]           = sp.dry_base;
    p["dry_allowed_length"] = sp.dry_allowed_length;
    p["dry_penalty_last_n"] = sp.dry_penalty_last_n;
    p["dry_sequence_breakers"] = sp.dry_seq_breakers;
    p["mirostat"]           = 0;
    p["stop"]               = std::vector<std::string>(r.stop.begin(),
                                                       r.stop.begin() + (long)std::min(r.n_client_stops, r.stop.size()));
    p["max_tokens"]         = r.max_tokens;
    p["n_predict"]          = r.max_tokens;
    p["ignore_eos"]         = r.ignore_eos;
    p["stream"]             = r.stream;
    p["grammar"]            = sp.grammar;
    p["n_probs"]            = r.want_logprobs ? r.n_logprobs : 0;
    p["post_sampling_probs"] = false;
    p["samplers"]           = sp.temp_last
        ? std::vector<std::string>{ "penalties", "dry", "top_k", "top_p", "min_p", "typ_p", "xtc", "temperature" }
        : std::vector<std::string>{ "penalties", "dry", "temperature", "top_k", "top_p", "min_p", "typ_p", "xtc" };
    return p;
}

static const char* native_stop_type(Finish f) {
    switch (f) {
        case Finish::Stop:       return "eos";
        case Finish::Length:     return "limit";
        case Finish::StopString: return "word";
        default:                 return "none";
    }
}

/* llama-server's final result for one choice: text, finish, prompt and timings. A stream's last
 * chunk is this with `content` empty, the text having gone out in earlier chunks. */
json Server::native_result(const OaiRequest& r, const Generation& g, const std::string& content) {
    const Sink* one = g.sink.get();
    const TimingsOut t = timings_of(&one, 1, r.t_received_ns);
    const int64_t n_prompt = (int64_t)r.prompts[(size_t)g.prompt_index].tokens.size();
    nlohmann::ordered_json o;
    o["index"]   = g.index;
    o["content"] = content;
    o["tokens"]  = r.return_tokens ? g.tokens : std::vector<int32_t>{};
    o["id_slot"] = 0;
    o["stop"]    = true;
    o["model"]   = r.model;
    o["tokens_predicted"] = g.sink->n_produced();
    o["tokens_evaluated"] = n_prompt;
    o["generation_settings"] = native_settings(r);
    o["prompt"]  = r.prompts[(size_t)g.prompt_index].text;
    o["has_new_line"] = g.text.find('\n') != std::string::npos;
    o["truncated"] = false;
    o["stop_type"] = native_stop_type(g.finish);
    o["stopping_word"] = g.stop_word;
    o["tokens_cached"] = n_prompt + g.sink->n_produced();
    if (t.valid) o["timings"] = timings_json(t);
    /* The whole reply's probabilities, as llama-server sends them; a stream sent them per
     * chunk instead. */
    if (r.want_logprobs && !r.stream) {
        const std::vector<LogprobEntry> kept = kept_logprobs(g.logprobs, g.stop_cut);
        o["completion_probabilities"] =
            nlohmann::ordered_json::parse("[" + logprobs_content(kept.data(), kept.size()) + "]");
    }
    return o;
}

/* ------------------------------------------------------------------ streaming */

static std::string text_chunk(const OaiRequest& r, int index, const std::string& delta,
                              const char* finish_reason, const TimingsOut* t = nullptr,
                              const std::string& logprobs = std::string()) {
    nlohmann::ordered_json ch;
    ch["text"] = delta;
    ch["index"] = index;
    if (!logprobs.empty()) ch["logprobs"] = nlohmann::ordered_json::parse(logprobs);
    else ch["logprobs"] = nullptr;
    if (finish_reason) ch["finish_reason"] = finish_reason; else ch["finish_reason"] = nullptr;
    nlohmann::ordered_json o;
    o["id"] = r.id;
    o["object"] = "text_completion";
    o["created"] = r.created;
    o["model"] = r.model;
    if (!r.system_fingerprint.empty()) o["system_fingerprint"] = r.system_fingerprint;
    o["choices"] = nlohmann::ordered_json::array({ch});
    if (t && t->valid) o["timings"] = timings_json(*t);
    return sse_event(o.dump());
}

/* llama-server's `prompt_progress` chunk: the endpoint's opening chunk (an assistant role with no
 * content, or empty text) carrying prompt tokens processed, the total, how many the prefix cache
 * supplied, and the time since the prompt phase began. */
static std::string progress_chunk(const OaiRequest& r, int index, const Sink& s) {
    const int64_t ta = s.t_admitted_ns();
    nlohmann::ordered_json pp;
    pp["total"]     = s.progress_total();
    pp["cache"]     = s.progress_cached();
    pp["processed"] = s.progress_processed();
    pp["time_ms"]   = ta >= 0 ? (s.age_ns() - ta) / 1000000 : 0;
    if (r.shape == ApiShape::LlamaNative) {
        /* llama-server's native chunk carries the progress at its top level. */
        nlohmann::ordered_json o;
        o["index"] = index;
        o["content"] = "";
        o["tokens"] = nlohmann::ordered_json::array();
        o["stop"] = false;
        o["prompt_progress"] = std::move(pp);
        return sse_event(o.dump());
    }
    nlohmann::ordered_json ch;
    if (r.endpoint == Endpoint::Chat) {
        ch["index"] = index;
        ch["delta"] = { { "role", "assistant" }, { "content", nullptr } };
    } else {
        ch["text"] = "";
        ch["index"] = index;
        ch["logprobs"] = nullptr;
    }
    ch["finish_reason"] = nullptr;
    nlohmann::ordered_json o;
    o["id"] = r.id;
    o["object"] = r.endpoint == Endpoint::Chat ? "chat.completion.chunk" : "text_completion";
    o["created"] = r.created;
    o["model"] = r.model;
    if (!r.system_fingerprint.empty()) o["system_fingerprint"] = r.system_fingerprint;
    o["choices"] = nlohmann::ordered_json::array({ch});
    o["prompt_progress"] = std::move(pp);
    return sse_event(o.dump());
}

HttpResponse Server::streaming_completion(const HttpRequest& req, OaiRequest& r) {
    r.t_received_ns = req.t_received_ns;
    auto st = std::make_shared<StreamState>();
    st->r = r;
    st->peer_closed = req.closed;

    ApiError err;
    if (submit_all(st->r, &st->gens, &err) < 0) {
        retire(st->gens);
        release(r.n_choices());
        return error_response(err, req.peer, r.shape);
    }
    if (st->r.endpoint == Endpoint::Chat)
        for (auto& g : st->gens)
            g.delta.reset(
                st->r.shape == ApiShape::Anthropic
                    ? (ReplyStream*)new AnthropicStream(st->r, g.index, st->r.reply_parser.get())
                : st->r.shape == ApiShape::Responses
                    ? (ReplyStream*)new ResponsesStream(st->r, g.index, st->r.reply_parser.get())
                    : (ReplyStream*)new ChatDeltaStream(st->r, g.index, st->r.reply_parser.get()));

    HttpResponse res;
    res.status = 200;
    res.content_type = "text/event-stream; charset=utf-8";
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    /* nginx buffers proxied responses by default; this header is its documented opt-out. */
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
                /* `echo` on the streaming path: the prompt goes out as the first content chunk,
                 * so the concatenated stream equals the buffered response. Empty prompts emit
                 * nothing. */
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
            /* Prompt progress first, before the first token: one chunk per reported advance, as
             * llama-server sends one per prefill batch. */
            if (st->r.return_progress) {
                const int32_t done = g.sink->progress_processed();
                if (done > g.progress_sent) {
                    out += progress_chunk(st->r, g.index, *g.sink);
                    g.progress_sent = done;
                }
            }
            if (g.delta && g.sink->n_produced() > 0) out += g.delta->started();
            /* While the choice runs, hold back the longest stop string minus one byte, or the
             * start of a stop sequence would already be on the wire when it is recognised.
             *
             * Only while running: a closed choice has no more bytes coming, so its whole text is
             * the answer, including a trailing `\` or partial `\uXXXX` the escape rule would
             * otherwise hold back. */
            size_t safe = g.text.size();
            if (!g.closed) {
                if (maxstop > 1) safe = safe > maxstop - 1 ? safe - (maxstop - 1) : 0;
                safe = json_stable_prefix(std::string_view(g.text.data(), safe));
                if (safe <= g.emitted) continue;
            }

            /* Logprobs for the tokens whose text this chunk carries: entries ending inside it
             * while the choice runs; once closed, the rest that went out with the reply
             * (lp_before_cut). */
            size_t lp_n = 0, lp_off = 0;
            if (st->r.want_logprobs) {
                lp_off = g.lp_sent ? g.logprobs[g.lp_sent - 1].end : 0;
                if (g.closed)
                    lp_n = lp_before_cut(g.logprobs, g.lp_sent, g.stop_cut);
                else
                    while (g.lp_sent + lp_n < g.logprobs.size() &&
                           g.logprobs[g.lp_sent + lp_n].end <= safe) ++lp_n;
            }
            const LogprobEntry* lp_at = lp_n ? &g.logprobs[g.lp_sent] : nullptr;
            g.lp_sent += lp_n;

            /* llama.cpp's `timings_per_token`: this choice's timings so far on every chunk. */
            TimingsOut now;
            if (st->r.timings_per_token) {
                const Sink* one = g.sink.get();
                now = timings_of(&one, 1, st->r.t_received_ns);
                if (g.delta) g.delta->set_timings(now);
            }
            if (st->r.endpoint == Endpoint::Chat) {
                /* Only the new text: the reader keeps its place, so a reply costs linear time. */
                if (lp_n) g.delta->add_logprobs(logprobs_content(lp_at, lp_n));
                out += g.delta->push(std::string_view(g.text).substr(g.emitted, safe - g.emitted),
                                     g.closed);
            } else if (st->r.shape == ApiShape::LlamaNative) {
                /* llama-server's native chunk: the new text, the ids that made it, the counts. */
                if (safe > g.emitted || g.tokens.size() > g.tokens_sent || lp_n) {
                    nlohmann::ordered_json o;
                    o["index"] = g.index;
                    o["content"] = g.text.substr(g.emitted, safe - g.emitted);
                    o["tokens"] = std::vector<int32_t>(g.tokens.begin() + (long)g.tokens_sent,
                                                       g.tokens.end());
                    o["stop"] = false;
                    o["id_slot"] = -1;
                    o["tokens_predicted"] = g.sink->n_produced();
                    o["tokens_evaluated"] =
                        (int64_t)st->r.prompts[(size_t)g.prompt_index].tokens.size();
                    if (st->r.timings_per_token && now.valid) o["timings"] = timings_json(now);
                    if (lp_n)
                        o["completion_probabilities"] = nlohmann::ordered_json::parse(
                            "[" + logprobs_content(lp_at, lp_n) + "]");
                    out += sse_event(o.dump());
                    g.tokens_sent = g.tokens.size();
                }
            } else if (safe > g.emitted || lp_n) {
                out += text_chunk(st->r, g.index, g.text.substr(g.emitted, safe - g.emitted),
                                  nullptr, st->r.timings_per_token ? &now : nullptr,
                                  lp_n ? logprobs_legacy(lp_at, lp_n, lp_off) : std::string());
            }
            g.emitted = safe;
            /* Of the sent entries only the last is read again (for where the next one starts);
             * keeping all would hold ~1.7 KB per token at 20 alternatives for a long stream. */
            if (g.lp_sent > 1) {
                g.logprobs.erase(g.logprobs.begin(), g.logprobs.begin() + (long)(g.lp_sent - 1));
                g.lp_sent = 1;
            }
        }

        if (self->all_done(st->gens)) {
            const UsageOut u = usage_of(st->r, st->gens);
            Finish overall = Finish::Stop;
            for (auto& g : st->gens) {
                /* Each choice's own timings on its final chunk, as llama.cpp sends them, so they
                 * arrive even without a usage chunk. */
                const Sink* one = g.sink.get();
                const TimingsOut gt = timings_of(&one, 1, st->r.t_received_ns);
                /* The 200 headers are already sent, so a failed generation becomes an error event
                 * in the stream; a silent end would pass a truncated answer as complete. */
                ApiError e;
                if (g.finish == Finish::Error) {
                    e = err_internal(engine_failure(*g.sink));
                    overall = Finish::Error;
                }
                if (st->r.endpoint == Endpoint::Chat) {
                    if (auto* a = dynamic_cast<AnthropicStream*>(g.delta.get()))
                        a->set_stop_word(g.stop_word);
                    if (g.finish == Finish::Error) {
                        out += g.delta->error_chunk(e, gt, u);
                    } else {
                        out += g.delta->final_chunk(g.finish, gt, u);
                        overall = outcome(overall, g.delta->adjusted_finish(g.finish));
                    }
                } else if (st->r.shape == ApiShape::LlamaNative) {
                    out += sse_event(json_dump(self->native_result(st->r, g, "")));
                    if (g.finish == Finish::Error) out += sse_error(e);
                    overall = outcome(overall, g.finish);
                } else {
                    out += text_chunk(st->r, g.index, "", finish_reason_oai(g.finish), &gt);
                    if (g.finish == Finish::Error) out += sse_error(e);
                    overall = outcome(overall, g.finish);
                }
            }
            /* The usage chunk (`stream_options.include_usage`, honoured on both endpoints).
             * After the final chunks and before [DONE], where OpenAI puts it, and once per
             * request: the `n` choices share one prompt and one total. */
            if (st->r.include_usage) {
                const TimingsOut t = timings_of(st->gens, st->r.t_received_ns);
                out += st->r.endpoint == Endpoint::Chat ? chat_usage_chunk(st->r, u, t)
                                                        : text_usage_chunk(st->r, u, t);
            }

            /* Anthropic's stream ends with message_stop and the Responses API's with
             * response.completed, which final_chunk wrote; chat completions' with [DONE]. */
            if (st->r.shape == ApiShape::OpenAI) out += SSE_DONE;
            st->done_sent = true;
            self->record(st->r, st->gens, overall);
            self->retire(st->gens);
            self->metrics_.observe_http(200);
            if (!st->released) { st->released = true; self->release(st->r.n_choices()); }
            return false;
        }

        if (!progress && out.empty()) {
            /* Nothing to send: sleep on the first live sink. The wait is bounded so the
             * peer-closed check above runs a few times a second. */
            for (auto& g : st->gens) if (!g.closed) { g.sink->wait(); break; }
        }
        return true;
    };

    res.on_close = [self, st](bool ok) {
        if (!st->done_sent) {
            /* The peer went away or the write failed: cancel now so the blocks are freed. */
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

/* The 501 for a text-generation endpoint this deployment cannot serve, or nothing. */
std::optional<HttpResponse> Server::refuse_text(const std::string& peer, ApiShape shape) {
    const char* why =
        !deps_.sched || !deps_.tok ? "the server has no engine attached"
        : opt_.embedding_model     ? "this is an embedding model: it serves /v1/embeddings and "
                                     "generates no text"
        : opt_.image_model         ? "this is an image model: it serves /v1/images/generations "
                                     "and /v1/images/edits and generates no text"
        : opt_.audio_model         ? "this is an audio model: it serves /v1/audio/speech and "
                                     "generates no text"
        : opt_.video_model         ? "this is a video model: it serves /v1/videos and generates "
                                     "no text"
                                   : nullptr;
    if (!why) return std::nullopt;
    return error_response(err_not_implemented(why), peer, shape);
}

HttpResponse Server::handle_chat(const HttpRequest& req) {
    if (auto no = refuse_text(req.peer)) return *no;
    OaiRequest r;
    ApiError err;
    if (parse_chat_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    return serve(req, r);
}

/* llama-server's /apply-template and its token count. Both parse exactly as /v1/chat/completions
 * does (same defaults, same refusals) and stop before queueing, so they report the prompt a chat
 * request would run. */
HttpResponse Server::handle_apply_template(const HttpRequest& req) {
    if (!deps_.tok || !deps_.chat)
        return error_response(err_not_implemented("rendering a chat requires a chat template; "
                                                  "this model was loaded without one"), req.peer);
    OaiRequest r;
    ApiError err;
    if (parse_chat_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    json o;
    o["prompt"] = r.prompts.empty() ? std::string() : r.prompts[0].text;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* The token count of the prompt a request would run, parsed as a chat request and not queued.
 * `to_chat` translates another API's body first (null for a chat body); `object` is the
 * response's `object` field, or null for none. */
HttpResponse Server::count_tokens(const HttpRequest& req, ToChat to_chat, ApiShape shape,
                                  const char* object) {
    if (!deps_.tok || !deps_.chat)
        return error_response(err_not_implemented("rendering a chat requires a chat template; "
                                                  "this model was loaded without one"),
                              req.peer, shape);
    std::string chat;
    ApiError err;
    OaiRequest r;
    if ((to_chat && to_chat(req.body, lim_, &chat, nullptr, &err) < 0) ||
        parse_chat_request(to_chat ? chat : req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer, shape);
    json o;
    if (object) o["object"] = object;
    o["input_tokens"] = r.prompts.empty() ? 0 : (int64_t)r.prompts[0].tokens.size();
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* Anthropic's Messages API: translated into the equivalent chat request and parsed by the chat
 * parser, so every default, refusal and template rule is the chat endpoint's; answers and errors
 * go out in Anthropic's shapes. */
HttpResponse Server::handle_messages(const HttpRequest& req) {
    if (auto no = refuse_text(req.peer, ApiShape::Anthropic)) return *no;
    std::string chat, label;
    bool cm = false;
    ApiError err;
    OaiRequest r;
    if (anthropic_to_chat(req.body, lim_, &chat, &label, &err, &cm) < 0 ||
        parse_chat_request(chat, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer, ApiShape::Anthropic);
    r.shape = ApiShape::Anthropic;
    r.context_management = cm;
    r.id = random_id("msg_");
    if (r.session_label.empty()) r.session_label = label;
    return serve(req, r);
}

HttpResponse Server::serve(const HttpRequest& req, OaiRequest& r) {
    r.peer = req.peer;
    if (auto no = admit(r.n_choices(), req.peer, r.shape, r.prompts.size() > 1 ? "prompt" : "n"))
        return *no;
    log_admitted(r);
    return r.stream ? streaming_completion(req, r) : blocking_completion(req, r);
}

/* llama-server's native /completion: the completions parser with llama-server's prompt forms,
 * answered in its shape (native_result). */
HttpResponse Server::handle_native_completion(const HttpRequest& req) {
    if (auto no = refuse_text(req.peer)) return *no;
    OaiRequest r;
    ApiError err;
    if (parse_completion_request(req.body, odeps_, lim_, &r, &err, /*native=*/true) < 0)
        return error_response(err, req.peer);
    r.shape = ApiShape::LlamaNative;
    return serve(req, r);
}

/* The first of `names` (the spellings llama.cpp's vocabulary loader recognises for a FIM token)
 * that is one token rendering as itself, or -1. */
static int32_t special_token(const ITokenizer& tok, std::initializer_list<const char*> names) {
    for (const char* n : names) {
        const std::vector<int32_t> t = tok.encode(n, false, /*parse_special=*/true);
        if (t.size() == 1 && tok.token_bytes(t[0]) == n) return t[0];
    }
    return -1;
}

/* llama-server's /infill: code completion between a prefix and a suffix (llama.vim,
 * llama.vscode), in llama-server's repository-level FIM layout (arXiv 2409.12186), each extra file
 * under its name, then prefix, suffix and middle:
 *
 *   [REPO]myproject\n [SEP]name\n text ... [SEP]filename\n [PRE]prefix+prompt [SUF]suffix [MID]
 *
 * A model without FIM tokens is refused. The completion is a native one, with every native
 * field. */
HttpResponse Server::handle_infill(const HttpRequest& req) {
    if (auto no = refuse_text(req.peer)) return *no;
    const ITokenizer& tok = *deps_.tok;
    const int32_t pre = special_token(tok, { "<|fim_prefix|>", "<fim_prefix>", "<fim-prefix>",
                                             "<｜fim▁begin｜>", "<PRE>", "▁<PRE>", "<|code_prefix|>" });
    const int32_t suf = special_token(tok, { "<|fim_suffix|>", "<fim_suffix>", "<fim-suffix>",
                                             "<｜fim▁hole｜>", "<SUF>", "▁<SUF>", "<|code_suffix|>" });
    const int32_t mid = special_token(tok, { "<|fim_middle|>", "<fim_middle>", "<fim-middle>",
                                             "<｜fim▁end｜>", "<MID>", "▁<MID>", "<|code_middle|>" });
    if (pre < 0 || suf < 0 || mid < 0)
        return error_response(err_not_implemented("this model has no fill-in-the-middle tokens, "
                                                  "so /infill cannot be served"), req.peer);
    const int32_t rep = special_token(tok, { "<|repo_name|>", "<|fim_repo|>", "<fim-repo>",
                                             "<reponame>", "<REPO>" });
    const int32_t sep = special_token(tok, { "<|file_sep|>", "<|fim_sep|>", "<fim-sep>",
                                             "<file_sep>", "<filename>" });

    json b;
    {
        ApiError e;
        if (parse_request_object(req.body, lim_, &b, &e) < 0) return error_response(e, req.peer);
    }
    auto text_of = [&](const char* k, std::vector<int32_t>* out, ApiError* e) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) return true;
        if (it->is_string()) {
            *out = tok.encode(it->get<std::string>(), false, false);
            return true;
        }
        if (it->is_array()) {
            for (const json& v : *it) {
                int32_t id = 0;
                if (v.is_string()) {
                    const std::vector<int32_t> t = tok.encode(v.get<std::string>(), false, false);
                    out->insert(out->end(), t.begin(), t.end());
                } else if (v.is_number_integer() && token_id(v, &id)) {
                    out->push_back(id);
                } else {
                    *e = err_bad(k, "expected a string, or strings and token ids");
                    return false;
                }
            }
            return true;
        }
        *e = err_bad(k, "expected a string");
        return false;
    };
    ApiError err;
    std::vector<int32_t> prefix, suffix, prompt;
    if (!text_of("input_prefix", &prefix, &err) || !text_of("input_suffix", &suffix, &err) ||
        !text_of("prompt", &prompt, &err))
        return error_response(err, req.peer);

    std::vector<int32_t> extra;
    if (rep >= 0) {
        extra.push_back(rep);
        const std::vector<int32_t> t = tok.encode("myproject\n", false, false);
        extra.insert(extra.end(), t.begin(), t.end());
    }
    auto ex = b.find("input_extra");
    if (ex != b.end() && ex->is_array()) {
        for (const json& c : *ex) {
            if (!c.is_object()) continue;
            auto fn = c.find("filename"), tx = c.find("text");
            if ((fn != c.end() && !fn->is_null() && !fn->is_string()) ||
                (tx != c.end() && !tx->is_null() && !tx->is_string()))
                return error_response(err_bad("input_extra[]", "filename and text must be strings"),
                                      req.peer);
            const std::string name = fn != c.end() && fn->is_string() ? fn->get<std::string>() : "tmp";
            std::vector<int32_t> t;
            if (sep >= 0) {
                extra.push_back(sep);
                t = tok.encode(name + "\n", false, false);
            } else {
                t = tok.encode("\n\n--- snippet ---\n\n", false, false);
            }
            extra.insert(extra.end(), t.begin(), t.end());
            t = tok.encode(tx != c.end() && tx->is_string() ? tx->get<std::string>() : std::string(),
                           false, false);
            extra.insert(extra.end(), t.begin(), t.end());
        }
    }
    if (sep >= 0) {
        extra.push_back(sep);
        const std::vector<int32_t> t = tok.encode("filename\n", false, false);
        extra.insert(extra.end(), t.begin(), t.end());
    }

    /* When everything does not fit beside the reply's budget, the extra files are trimmed first
     * (newest kept): they are context, while prefix and suffix are the edit. */
    int64_t reserve = 256;
    for (const char* k : { "max_tokens", "n_predict" }) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        if (!it->is_number_integer())
            return error_response(err_bad(k, "expected an integer"), req.peer);
        reserve = it->get<int64_t>();
    }
    if (reserve <= 0) reserve = 256;
    const ITokenizer::VocabInfo vi = tok.vocab_info();
    const int64_t fixed = (int64_t)(prefix.size() + prompt.size() + suffix.size()) + 3 +
                          (vi.add_bos && vi.bos >= 0 ? 1 : 0);
    if (opt_.max_ctx > 0) {
        const int64_t room = std::max<int64_t>(opt_.max_ctx - fixed - reserve, 0);
        if ((int64_t)extra.size() > room) extra.erase(extra.begin(), extra.end() - room);
    }
    std::vector<int32_t> ids;
    if (vi.add_bos && vi.bos >= 0) ids.push_back(vi.bos);
    ids.insert(ids.end(), extra.begin(), extra.end());
    ids.push_back(pre);
    ids.insert(ids.end(), prefix.begin(), prefix.end());
    ids.insert(ids.end(), prompt.begin(), prompt.end());
    ids.push_back(suf);
    ids.insert(ids.end(), suffix.begin(), suffix.end());
    ids.push_back(mid);

    for (const char* k : { "input_prefix", "input_suffix", "input_extra", "prompt" }) b.erase(k);
    b["prompt"] = ids;
    OaiRequest r;
    if (parse_completion_request(b.dump(), odeps_, lim_, &r, &err, /*native=*/true) < 0)
        return error_response(err, req.peer);
    r.shape = ApiShape::LlamaNative;
    return serve(req, r);
}

/* OpenAI's Responses API, like the Messages API: translated into the equivalent chat request,
 * served by the chat path, answered as a response object or its event stream. */
HttpResponse Server::handle_responses(const HttpRequest& req) {
    if (auto no = refuse_text(req.peer)) return *no;
    std::string chat, echo;
    ApiError err;
    OaiRequest r;
    std::map<std::string, std::pair<std::string, std::string>> ns;
    if (responses_to_chat(req.body, lim_, &chat, &echo, &err, &ns) < 0 ||
        parse_chat_request(chat, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    r.shape = ApiShape::Responses;
    r.tool_namespace = std::move(ns);
    r.id = random_id("resp_");
    r.echo_json = std::move(echo);
    return serve(req, r);
}

HttpResponse Server::handle_completion(const HttpRequest& req) {
    if (auto no = refuse_text(req.peer)) return *no;
    OaiRequest r;
    ApiError err;
    if (parse_completion_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    return serve(req, r);
}

HttpResponse Server::handle_embeddings(const HttpRequest& req) {
    if (!deps_.sched || !deps_.tok)
        return error_response(err_not_implemented("the server has no engine attached"),
                              req.peer);
    if (!deps_.embed)
        return error_response(err_not_implemented(
            opt_.image_model ? "/v1/embeddings requires an embedding model; this one makes images"
            : opt_.audio_model ? "/v1/embeddings requires an embedding model; this one makes audio"
            : opt_.video_model ? "/v1/embeddings requires an embedding model; this one makes video"
                               : "/v1/embeddings requires an embedding model; this one generates "
                                 "text"), req.peer);

    OaiRequest r;
    ApiError err;
    if (parse_embeddings_request(req.body, odeps_, lim_, &r, &err) < 0)
        return error_response(err, req.peer);
    r.peer = req.peer;

    const int64_t dim = deps_.embed->dim();
    if (r.dimensions > dim)
        return error_response(err_bad("dimensions", "this model's vectors are " +
                                                        std::to_string(dim) + " wide"), req.peer);
    /* An input is never split (Request::embed), so one longer than a step is refused here, with
     * the limit in the message. */
    for (size_t i = 0; i < r.prompts.size(); ++i) {
        const int64_t n = (int64_t)r.prompts[i].tokens.size();
        if (n > deps_.embed->max_input()) {
            ApiError e = err_bad("input", "input " + std::to_string(i) + " is " +
                                              std::to_string(n) + " tokens; this deployment "
                                              "embeds at most " +
                                              std::to_string(deps_.embed->max_input()) +
                                              " (--max-num-batched-tokens)");
            e.code = "context_length_exceeded";
            return error_response(e, req.peer);
        }
    }

    const int n_in = (int)r.prompts.size();
    if (auto no = admit(n_in, req.peer, ApiShape::OpenAI, "input")) return *no;
    log_admitted(r);

    /* Submit every input at once so the scheduler can batch them in one step. */
    std::vector<Generation> gens;
    std::vector<Request> rqs;
    gens.reserve((size_t)n_in);
    rqs.reserve((size_t)n_in);
    for (int i = 0; i < n_in; ++i) {
        Generation g;
        g.index = i;
        g.prompt_index = i;
        g.sink = std::make_shared<Sink>(16);
        Request rq;
        rq.prompt = r.prompts[(size_t)i].tokens;
        rq.media = r.prompts[(size_t)i].media;
        rq.embed = true;
        rq.max_tokens = 1;
        rq.sp.temp = 0.0f;
        rq.sink = g.sink.get();
        rqs.push_back(std::move(rq));
        gens.push_back(std::move(g));
    }
    std::vector<uint64_t> ids;
    deps_.sched->submit_all(rqs, &ids);
    for (size_t i = 0; i < ids.size(); ++i) {
        gens[i].id = ids[i];
        gens[i].sink->id = ids[i];
        if (!ids[i]) {
            cancel_all(gens, false);
            retire(gens);
            release(n_in);
            return error_response(err_internal("the scheduler refused input " + std::to_string(i)),
                                  req.peer);
        }
    }

    for (;;) {
        bool all = true;
        for (auto& g : gens) {
            if (!g.closed && g.sink->done()) { g.closed = true; g.finish = g.sink->finish_reason(); }
            all = all && g.closed;
        }
        if (all) break;
        if (auto no = abandoned(req, gens, n_in)) return *no;
        for (auto& g : gens) if (!g.closed) { g.sink->wait(); break; }
    }

    /* Vectors truncated to `dimensions`, then L2-normalised (as OpenAI's endpoint and
     * sentence-transformers' Normalize module return them); truncation breaks unit length, so
     * normalisation comes after. */
    std::vector<std::vector<float>> vecs;
    vecs.reserve((size_t)n_in);
    UsageOut u;
    const int64_t keep = r.dimensions > 0 ? r.dimensions : dim;
    for (auto& g : gens) {
        const std::vector<float>& v = g.sink->embedding();
        if (g.finish == Finish::Error || g.finish == Finish::Cancelled || (int64_t)v.size() != dim) {
            const std::string why = g.finish == Finish::Error ? engine_failure(*g.sink)
                                    : std::string("the engine returned no vector for input ") +
                                          std::to_string(g.index);
            retire(gens);
            release(n_in);
            return error_response(err_internal(why), req.peer);
        }
        std::vector<float> out(v.begin(), v.begin() + keep);
        double ss = 0.0;
        for (float x : out) ss += (double)x * x;
        const float inv = ss > 0.0 ? (float)(1.0 / std::sqrt(ss)) : 0.0f;
        for (float& x : out) x *= inv;
        u.prompt += (int64_t)r.prompts[(size_t)g.prompt_index].tokens.size();
        vecs.push_back(std::move(out));
    }
    retire(gens);
    release(n_in);
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, embeddings_body(r, vecs, u));
}

/* /v1/systemone (systemone.h): one engine request per question, each reading its answer
 * candidates at the prompt's last position (Request::token_scores) and sampling one discarded
 * token.
 *
 * Questions share the state they are asked over, and it is read once: the first question runs
 * alone and leaves the shared head in the prefix cache (attention blocks plus a convolution
 * checkpoint, Request::checkpoint_at); the rest are then submitted together and compute only their
 * own tails. d1's runner shares the head the same way, as one batched tree. */
HttpResponse Server::handle_systemone(const HttpRequest& req) {
    if (!deps_.sched || !deps_.tok)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);
    if (opt_.decision_type != "d1")
        return error_response(err_not_implemented(
            "/v1/systemone serves a decision model; this container does not declare one "
            "(convert d1 with --set decision.type=d1)"), req.peer);

    DecisionRequest r;
    ApiError err;
    if (parse_decision_request(req.body, odeps_, lim_, opt_.model_id, &r, &err) < 0)
        return error_response(err, req.peer);

    const int n = (int)r.questions.size();
    if (auto no = admit(n, req.peer, ApiShape::OpenAI, "questions")) return *no;
    RAD_INFO("systemone admitted%s questions=%d prompt=%lld tok", from_peer(req.peer).c_str(), n,
             (long long)r.input_tokens);

    std::vector<Generation> gens;
    std::vector<Request> rqs;
    std::vector<std::shared_ptr<TokenScores>> scores;
    for (int i = 0; i < n; ++i) {
        Generation g;
        g.index = i;
        g.sink = std::make_shared<Sink>(16);
        auto ts = std::make_shared<TokenScores>();
        ts->tokens = r.questions[(size_t)i].candidates;
        Request rq;
        rq.prompt = r.questions[(size_t)i].tokens;
        rq.media = r.media;
        rq.token_scores = ts;
        rq.max_tokens = 1;
        rq.sp.temp = 0.0f;
        rq.sink = g.sink.get();
        if (i == 0 && n > 1) rq.checkpoint_at = r.trunk;
        rqs.push_back(std::move(rq));
        gens.push_back(std::move(g));
        scores.push_back(std::move(ts));
    }
    /* Questions handed to the scheduler so far. An early exit retires those and drops the rest,
     * whose sinks would never finish. */
    size_t submitted = 0;
    auto fail = [&](const std::string& why) {
        gens.resize(submitted);
        retire(gens);
        release(n);
        return error_response(err_internal(why), req.peer);
    };

    /* Submits questions [first, last) together and waits for them. Sets `early` to the call's
     * response when the scheduler refuses one, the client leaves or the server stops. */
    std::unique_ptr<HttpResponse> early;
    auto run = [&](int first, int last) {
        std::vector<Request> part(std::make_move_iterator(rqs.begin() + first),
                                  std::make_move_iterator(rqs.begin() + last));
        std::vector<uint64_t> ids;
        deps_.sched->submit_all(part, &ids);
        submitted = (size_t)last;
        for (size_t k = 0; k < ids.size(); ++k) {
            Generation& g = gens[(size_t)first + k];
            g.id = ids[k];
            g.sink->id = ids[k];
            if (!ids[k]) {
                early = std::make_unique<HttpResponse>(
                    fail("the scheduler refused question " + r.questions[(size_t)first + k].name));
                return;
            }
        }
        for (;;) {
            bool all = true;
            for (int i = first; i < last; ++i) {
                Generation& g = gens[(size_t)i];
                if (!g.closed && g.sink->done()) { g.closed = true; g.finish = g.sink->finish_reason(); }
                all = all && g.closed;
            }
            if (all) return;
            /* Both conditions only ever turn true, so abandoned() answers once either holds. */
            if (req.peer_gone() || stopping_.load()) {
                gens.resize(submitted);
                early = std::make_unique<HttpResponse>(*abandoned(req, gens, n));
                return;
            }
            for (int i = first; i < last; ++i)
                if (!gens[(size_t)i].closed) { gens[(size_t)i].sink->wait(); break; }
        }
    };
    run(0, 1);
    if (early) return std::move(*early);
    /* A first question that failed fails the call; the rest would only repeat it. */
    if (gens[0].finish == Finish::Error || gens[0].finish == Finish::Cancelled)
        return fail(gens[0].finish == Finish::Error ? engine_failure(*gens[0].sink)
                                                    : "the engine cancelled question " +
                                                          r.questions[0].name);
    if (n > 1) {
        run(1, n);
        if (early) return std::move(*early);
    }

    json answers = json::object();
    for (size_t i = 0; i < gens.size(); ++i) {
        const Generation& g = gens[i];
        if (g.finish == Finish::Error || g.finish == Finish::Cancelled)
            return fail(g.finish == Finish::Error ? engine_failure(*g.sink)
                                                  : "the engine cancelled question " +
                                                        r.questions[i].name);
        json a;
        if (decision_answer(r.questions[i], scores[i]->logits, &a, &err) < 0) {
            retire(gens);
            release(n);
            return error_response(err, req.peer);
        }
        answers[r.questions[i].name] = std::move(a);
    }
    retire(gens);
    release(n);
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(json{
        { "answers", std::move(answers) },
        { "usage", { { "input_tokens", r.input_tokens }, { "output_tokens", 0 } } } }));
}

HttpResponse Server::handle_models(const HttpRequest& req) {
    const std::string id = req.param("id");
    /* Runtime LoRA: each loaded adapter is a model of its own, whose parent is the served one. */
    std::vector<std::string> adapters;
    if (opt_.lora_names) adapters = opt_.lora_names();
    const bool is_adapter = !id.empty() && std::find(adapters.begin(), adapters.end(), id) != adapters.end();
    if (!id.empty() && id != opt_.model_id && !is_adapter) {
        ApiError e;
        e.status = 404;
        e.type = "not_found_error";
        e.param = "model";
        e.message = "no such model: " + id;
        return error_response(e, req.peer);
    }
    ModelCard c;
    c.id = opt_.model_id;
    c.created = started_at_;
    c.max_model_len = opt_.max_ctx;
    c.root = opt_.model_path;
    c.capabilities = capabilities().dump();
    metrics_.observe_http(200);
    std::vector<ModelCard> cards{ c };
    for (const std::string& a : adapters) {
        ModelCard ac = c;
        ac.id = a;
        ac.parent = opt_.model_id;
        cards.push_back(ac);
        if (is_adapter && a == id) return HttpResponse::json_body(200, model_body(ac));
    }
    return HttpResponse::json_body(200, id.empty() ? models_body(cards) : model_body(c));
}

/* Runtime LoRA admin, vLLM's routes and bodies: {"lora_name", "lora_path"} to load, {"lora_name"}
 * to unload. The path is read on this machine, so without --api-key only a loopback client may
 * ask, the rule /diagnostics follows. */
/* A generator's adapter files: what `loras` and <lora:...> tags may name. */
HttpResponse Server::handle_loras(const HttpRequest& req) {
    using json = nlohmann::json;
    if (!opt_.gen_lora_names) {
        ApiError e;
        e.status = 404;
        e.type = "not_found_error";
        e.message = "this server takes no adapters in image or video requests (--lora, --lora-dir)";
        return error_response(e, req.peer);
    }
    json data = json::array();
    for (const std::string& n : opt_.gen_lora_names()) data.push_back({ { "id", n }, { "object", "lora" } });
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(json{ { "object", "list" }, { "data", std::move(data) } }));
}

HttpResponse Server::handle_lora_admin(const HttpRequest& req, bool load) {
    using json = nlohmann::json;
    auto fail = [&](int status, const char* type, const std::string& msg) {
        ApiError e;
        e.status = status;
        e.type = type;
        e.message = msg;
        return error_response(e, req.peer);
    };
    if (!opt_.lora_load || !opt_.lora_unload)
        return fail(404, "not_found_error", "this server was started without runtime LoRA (--lora or "
                                            "--max-loras)");
    if (opt_.api_key.empty() && !req.peer_is_loopback())
        return fail(403, "permission_error", "loading an adapter reads a path on this machine, and a "
                                             "server started without --api-key takes it only from a "
                                             "loopback client");
    const json j = json::parse(req.body, nullptr, false);
    if (j.is_discarded() || !j.is_object() || !j.contains("lora_name") || !j["lora_name"].is_string())
        return fail(400, "invalid_request_error", "the body must be a JSON object with lora_name");
    const std::string name = j["lora_name"].get<std::string>();
    if (name.empty() || name == opt_.model_id)
        return fail(400, "invalid_request_error", "lora_name must be non-empty and not the served model's name");
    std::string err;
    int s;
    int in_flight = 0;
    if (load) {
        if (!j.contains("lora_path") || !j["lora_path"].is_string())
            return fail(400, "invalid_request_error", "load_lora_adapter needs lora_path");
        s = opt_.lora_load(name, j["lora_path"].get<std::string>(), &err);
    } else {
        s = opt_.lora_unload(name, &err, &in_flight);
    }
    /* 409 for a load of a name that is still draining: the old adapter's requests are running. */
    if (s == RAD_E_NOTFOUND) return fail(404, "not_found_error", err);
    if (s == RAD_E_STATE) return fail(409, "conflict_error", err);
    if (s == RAD_E_FULL) return fail(503, "service_unavailable", err);
    if (s < 0) return fail(400, "invalid_request_error", err.empty() ? std::string(rad_strerror(s)) : err);
    /* An unload of an adapter in use drains it (lora/slots.h): accepted, not done. New requests
     * naming it are refused as an unknown model and it is off /v1/models; the requests counted
     * here finish, and the last of them frees the slot. Asking again answers with the count as it
     * is then, and 404 once it is gone. */
    if (!load && in_flight > 0) {
        metrics_.observe_http(202);
        return HttpResponse::json_body(202, json{ { "action", "draining" }, { "lora_name", name },
                                                  { "in_flight", in_flight } }.dump());
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json{ { "status", "ok" }, { "lora_name", name },
                                              { "action", load ? "loaded" : "unloaded" } }.dump());
}

HttpResponse Server::handle_metrics(const HttpRequest&) {
    if (deps_.sched) metrics_.sample(deps_.sched->metrics());
    metrics_.set_queue_depth(in_flight_.load(std::memory_order_relaxed));
    HttpResponse r;
    r.status = 200;
    /* The exposition format version belongs in the content type: Prometheus parses by it, and
     * without it some scrapers fall back to OpenMetrics and fail. */
    r.content_type = "text/plain; version=0.0.4; charset=utf-8";
    r.body = metrics_.render();
    return r;
}

}  /* namespace server */
}  /* namespace rad */
