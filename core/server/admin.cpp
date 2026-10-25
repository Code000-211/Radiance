/* core/server/admin.cpp -- vLLM's and llama-server's non-generation endpoints.
 *
 * /version /ping /load /tokenize /detokenize /get_tokenizer_info /server_info /reset_prefix_cache
 * /props /diagnostics.
 *
 * None of these touch the step path (no admission, no Sink, no cancellation); each is answered on
 * the HTTP thread before it returns, so they are kept out of server.cpp.
 *
 * Deliberately absent, answering 404: vLLM's /pooling, /classify, /score and /rerank (no
 * classifier or cross-encoder head), /sleep, /wake_up and /is_sleeping (no sleep mode; weights
 * are placed once, spec §5), /invocations (a SageMaker envelope), stored Responses-API responses
 * (would need a conversation store), and llama-server's /slots and /lora-adapters (a runtime LoRA
 * adapter is chosen per request by `model`; vLLM's load and unload routes are in server.cpp).
 */
#include "server.h"

#include "rad_internal.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

namespace rad {
namespace server {

namespace {

/* The body /health and /ping share. vLLM answers both, and neither redirects to the other, so a
 * probe never measures a redirect.
 *
 * `queue_cap` is the admission ceiling (`opt_.queue_depth` or max_seqs * 8), fixed while the
 * server runs. It is not called `queue_depth` because `radiance:http_queue_depth` is the occupancy;
 * the occupancy appears here as `in_flight` (and as `server_load` on /load). */
json health_body(const std::string& model, bool stopping, int64_t in_flight, int64_t depth) {
    json o;
    o["status"]      = stopping ? "stopping" : "ok";
    o["model"]       = model;
    o["in_flight"]   = in_flight;
    o["queue_cap"]   = depth;
    return o;
}

/* Parses a body that must be a JSON object (an empty body is {}, so every field takes its
 * default). Bounded like a generation body (parse_request_object), since /detokenize takes a
 * token array of any length. */
bool parse_object(const std::string& body, const OaiLimits& lim, json* out, ApiError* err) {
    if (body.empty()) {
        *out = json::object();
        return true;
    }
    return parse_request_object(body, lim, out, err) == RAD_OK;
}

bool get_bool_or(const json& b, const char* key, bool def) {
    auto it = b.find(key);
    return it != b.end() && it->is_boolean() ? it->get<bool>() : def;
}

}  /* namespace */

/* ================================================================== /version */
/* vLLM's {"version": "..."}, which compatibility checks read, plus the engine name and kernel ABI
 * version so a client need not fetch the whole /server_info for them. */
HttpResponse Server::handle_version(const HttpRequest&) {
    json o;
    o["version"]     = opt_.version;
    o["engine"]      = "radiance";
    o["abi_version"] = (int)RAD_ABI_VERSION;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /load */
/* vLLM's `server_load`: requests admitted and not yet finished, i.e. `in_flight`, which admission
 * control refuses on. `queue_cap` is included so a load balancer can tell how much room is left. */
HttpResponse Server::handle_load(const HttpRequest&) {
    json o;
    o["server_load"] = in_flight_.load(std::memory_order_relaxed);
    o["queue_cap"]   = queue_depth_;
    if (deps_.sched) {
        const SchedMetrics m = deps_.sched->metrics();
        o["running"] = m.running;
        o["waiting"] = m.waiting;
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /tokenize */
/* vLLM's two request shapes: `prompt` (completion form) and `messages` (chat form, rendered
 * through the template, then tokenised), plus llama-server's `content`.
 *
 * `add_special_tokens` defaults to true for `prompt` and false for `messages`, as in vLLM: a raw
 * prompt needs BOS prepended, while a rendered template already contains the model's opening.
 * `parse_special` is not the caller's choice: each form tokenises exactly as /v1/completions or
 * /v1/chat/completions would, so the count matches what that endpoint sees. */
HttpResponse Server::handle_tokenize(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);

    json b;
    ApiError err;
    if (!parse_object(req.body, lim_, &b, &err)) return error_response(err, req.peer);

    const bool want_strs = get_bool_or(b, "return_token_strs", false);
    std::vector<int32_t> toks;

    auto msgs = b.find("messages");
    auto prom = b.find("prompt");
    auto cont = b.find("content");
    if ((msgs != b.end()) + (prom != b.end()) + (cont != b.end()) > 1)
        return error_response(err_bad("prompt", "send one of `prompt`, `messages` or `content`"),
                              req.peer);

    /* llama-server's shape: `content` is a string, or strings and token ids mixed (as its
     * /completion prompt), with its own flags and defaults (no BOS, control tokens recognised).
     * Answered as `tokens`: ids, or with `with_pieces` {id, piece} objects. */
    if (cont != b.end()) {
        const bool add_special   = get_bool_or(b, "add_special", false);
        const bool parse_special = get_bool_or(b, "parse_special", true);
        const bool with_pieces   = get_bool_or(b, "with_pieces", false);
        auto take = [&](const json& v, bool first) -> bool {
            if (v.is_string()) {
                const std::vector<int32_t> t =
                    deps_.tok->encode(v.get<std::string>(), first && add_special, parse_special);
                toks.insert(toks.end(), t.begin(), t.end());
                return true;
            }
            int32_t id = 0;
            if (!v.is_number_integer() || !token_id(v, &id)) return false;
            toks.push_back(id);
            return true;
        };
        bool ok = true;
        if (cont->is_array()) {
            bool first = true;
            for (const json& v : *cont) { ok = ok && take(v, first); first = false; }
        } else {
            ok = cont->is_string() && take(*cont, true);
        }
        if (!ok)
            return error_response(err_bad("content", "must be a string, or an array of strings "
                                                     "and token ids"), req.peer);
        json arr = json::array();
        for (int32_t t : toks) {
            if (!with_pieces) { arr.push_back(t); continue; }
            const std::string piece = deps_.tok->token_bytes(t);
            json e;
            e["id"] = t;
            /* A piece that is not whole UTF-8 goes as its bytes, 0-255, as llama-server sends it. */
            if (utf8_valid_prefix(piece) == piece.size()) e["piece"] = piece;
            else e["piece"] = std::vector<int>((const unsigned char*)piece.data(),
                                               (const unsigned char*)piece.data() + piece.size());
            arr.push_back(std::move(e));
        }
        json o;
        o["tokens"]        = std::move(arr);
        o["count"]         = (int64_t)toks.size();
        o["max_model_len"] = opt_.max_ctx;
        metrics_.observe_http(200);
        return HttpResponse::json_body(200, json_dump(o));
    }

    if (msgs != b.end()) {
        if (!deps_.chat)
            return error_response(err_not_implemented(
                "tokenising `messages` requires a chat template; this model was not loaded with "
                "one. Send `prompt` with the text you want tokenised"), req.peer);
        if (!msgs->is_array() || msgs->empty())
            return error_response(err_bad("messages", "must be a non-empty array"), req.peer);

        ChatRenderOptions opt;
        opt.add_generation_prompt = get_bool_or(b, "add_generation_prompt", true);
        apply_template_defaults(b, lim_, &opt);
        auto kw = b.find("chat_template_kwargs");
        if (kw != b.end() && kw->is_object())
            for (auto it = kw->begin(); it != kw->end(); ++it)
                opt.template_kwargs[it.key()] = json_dump(it.value());
        map_reasoning_effort(lim_, &opt);

        json tools = json::array();
        auto t = b.find("tools");
        if (t != b.end() && t->is_array()) tools = *t;

        ChatRender render;
        std::string why;
        if (const int rs = deps_.chat->apply(*msgs, tools, opt, &render, &why); rs < 0)
            return error_response(template_refusal(opt, lim_, rs, why), req.peer);
        toks = deps_.tok->encode(render.prompt,
                                 get_bool_or(b, "add_special_tokens", false),
                                 /*parse_special=*/true);
    } else {
        if (prom == b.end() || !prom->is_string())
            return error_response(err_bad("prompt", "must be a string, or send `messages`"),
                                  req.peer);
        /* Encoded as /v1/completions encodes it, so the count is the count that endpoint sees. */
        toks = deps_.tok->encode(prom->get<std::string>(),
                                 get_bool_or(b, "add_special_tokens", true),
                                 /*parse_special=*/true);
    }

    json o;
    o["count"]         = (int64_t)toks.size();
    o["max_model_len"] = opt_.max_ctx;
    o["tokens"]        = toks;
    if (want_strs) {
        json strs = json::array();
        for (int32_t t : toks) strs.push_back(deps_.tok->piece(t));
        o["token_strs"] = std::move(strs);
    } else {
        o["token_strs"] = nullptr;
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /detokenize */
/* vLLM's response key is `prompt`, not `text` (the text a token array would arrive as); clients
 * read that name. */
HttpResponse Server::handle_detokenize(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);

    json b;
    ApiError err;
    if (!parse_object(req.body, lim_, &b, &err)) return error_response(err, req.peer);

    auto t = b.find("tokens");
    if (t == b.end() || !t->is_array())
        return error_response(err_bad("tokens", "must be an array of integers"), req.peer);

    std::vector<int32_t> toks;
    toks.reserve(t->size());
    for (const auto& v : *t) {
        if (!v.is_number_integer())
            return error_response(err_bad("tokens", "must be an array of integers"), req.peer);
        int32_t id = 0;
        if (!token_id(v, &id))
            return error_response(err_bad("tokens", "a token id must be in [0, 2147483647], "
                                                    "got " + v.dump()), req.peer);
        toks.push_back(id);
    }

    /* `prompt` is vLLM's key and `content` llama-server's, for the same text. */
    json o;
    o["prompt"] = deps_.tok->decode(toks);
    o["content"] = o["prompt"];
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /props */
/* llama-server's /props, for clients built against it: front-ends size their context from
 * default_generation_settings.n_ctx, show an image button for modalities.vision, and read the
 * default sampler settings. /server_info has all of it, but llama-server clients do not ask there.
 *
 * Only what this server can fill honestly, under llama-server's names: no GGUF metadata, no
 * per-slot state. n_ctx is the context one request may use (llama-server's per-slot n_ctx);
 * omitted when the context length is unknown. */
HttpResponse Server::handle_props(const HttpRequest&) {
    const SamplingParams& sp = opt_.default_sampling;
    /* Every field llama-server's `params` carries, so no client finds one missing. Samplers this
     * server does not run are reported at their off values, which is what it serves. */
    const int64_t n_predict = opt_.default_max_tokens > 0 ? opt_.default_max_tokens : -1;
    json params;
    params["n_predict"]           = n_predict;
    params["max_tokens"]          = n_predict;
    params["seed"]                = -1;
    params["temperature"]         = sp.temp;
    params["dynatemp_range"]      = 0.0;
    params["dynatemp_exponent"]   = 1.0;
    params["top_k"]               = sp.top_k;
    params["top_p"]               = sp.top_p;
    params["min_p"]               = sp.min_p;
    params["top_n_sigma"]         = -1.0;
    params["typical_p"]           = sp.typical_p;
    params["typ_p"]               = sp.typical_p;
    params["xtc_probability"]     = sp.xtc_probability;
    params["xtc_threshold"]       = sp.xtc_threshold;
    params["repeat_last_n"]       = sp.penalty_last_n;
    params["repeat_penalty"]      = sp.rep_penalty;
    params["presence_penalty"]    = sp.pres_penalty;
    params["frequency_penalty"]   = sp.freq_penalty;
    params["dry_multiplier"]      = sp.dry_multiplier;
    params["dry_base"]            = sp.dry_base;
    params["dry_allowed_length"]  = sp.dry_allowed_length;
    params["dry_penalty_last_n"]  = sp.dry_penalty_last_n;
    params["dry_sequence_breakers"] = dry_breakers(sp.dry_seq_breakers,
                                                   opt_.default_dry_breakers_set);
    params["mirostat"]            = 0;
    params["mirostat_tau"]        = 5.0;
    params["mirostat_eta"]        = 0.1;
    params["stop"]                = json::array();
    params["n_keep"]              = 0;
    params["n_discard"]           = 0;
    params["ignore_eos"]          = false;
    params["stream"]              = true;
    params["logit_bias"]          = json::array();
    params["n_probs"]             = 0;
    params["min_keep"]            = 0;
    params["grammar"]             = "";
    params["grammar_lazy"]        = false;
    params["grammar_triggers"]    = json::array();
    params["preserved_tokens"]    = json::array();
    params["chat_format"]         = "";
    params["reasoning_format"]    = "auto";
    params["reasoning_in_content"] = false;
    params["generation_prompt"]   = "";
    params["samplers"] = { "penalties", "dry", "temperature", "top_k", "top_p", "min_p", "typ_p",
                           "xtc" };
    params["backend_sampling"]    = false;
    params["timings_per_token"]   = false;
    params["post_sampling_probs"] = false;
    /* The adapters a request runs with when it names none: never any, since an adapter is chosen
     * by `model` (they are listed on /v1/models). */
    params["lora"]                = json::array();

    json gen;
    if (opt_.max_ctx > 0) gen["n_ctx"] = opt_.max_ctx;
    gen["params"] = std::move(params);

    json modal;
    modal["vision"] = deps_.mm && deps_.mm->accepts(MediaKind::Image);
    modal["video"]  = deps_.mm && deps_.mm->accepts(MediaKind::Video);
    modal["audio"]  = deps_.mm && deps_.mm->accepts(MediaKind::Audio);

    json o;
    o["default_generation_settings"] = std::move(gen);
    o["total_slots"] = opt_.max_seqs;
    o["model_alias"] = opt_.model_id;
    o["model_path"]  = opt_.model_path;
    o["modalities"]  = std::move(modal);
    o["endpoint_slots"]   = false;
    o["endpoint_props"]   = false;
    o["endpoint_metrics"] = true;
    o["chat_template"]      = deps_.chat ? deps_.chat->source() : std::string();
    o["chat_template_caps"] = json::parse(deps_.chat ? deps_.chat->caps() : std::string("{}"),
                                          nullptr, false);
    std::string bos, eos;
    if (deps_.tok) {
        const ITokenizer::VocabInfo v = deps_.tok->vocab_info();
        if (v.bos >= 0) bos = deps_.tok->piece(v.bos);
        if (v.eos >= 0) eos = deps_.tok->piece(v.eos);
    }
    o["bos_token"]   = bos;
    o["eos_token"]   = eos;
    o["build_info"]  = "radiance " + opt_.version;
    o["role"]        = "model";
    o["is_sleeping"] = false;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /get_tokenizer_info */
/* vLLM returns the HuggingFace tokenizer config verbatim, which a .rad container does not carry
 * (it holds a normalised vocabulary). This answers what that config is read for: vocabulary size,
 * the special ids, and whether a chat template is available. */
HttpResponse Server::handle_tokenizer_info(const HttpRequest& req) {
    if (!deps_.tok)
        return error_response(err_not_implemented("the server has no tokenizer attached"),
                              req.peer);

    const ITokenizer::VocabInfo v = deps_.tok->vocab_info();
    json o;
    o["tokenizer_class"]      = v.kind;
    o["vocab_size"]           = v.n_tokens;
    o["model_max_length"]     = opt_.max_ctx;
    o["bos_token_id"]         = v.bos >= 0 ? json(v.bos) : json(nullptr);
    o["eos_token_id"]         = v.eos >= 0 ? json(v.eos) : json(nullptr);
    o["eot_token_id"]         = v.eot >= 0 ? json(v.eot) : json(nullptr);
    o["unk_token_id"]         = v.unk >= 0 ? json(v.unk) : json(nullptr);
    o["pad_token_id"]         = v.pad >= 0 ? json(v.pad) : json(nullptr);
    o["add_bos_token"]        = v.add_bos;
    o["add_eos_token"]        = v.add_eos;
    o["chat_template"]        = deps_.chat != nullptr;
    o["supports_tools"]       = deps_.chat && deps_.chat->supports_tools();
    o["supports_reasoning"]   = deps_.chat && deps_.chat->supports_reasoning();
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== capabilities */
json Server::capabilities() const {
    json caps;
    caps["chat"]      = deps_.chat != nullptr;
    caps["tools"]     = deps_.chat && deps_.chat->supports_tools();
    caps["reasoning"] = deps_.chat && deps_.chat->supports_reasoning();
    /* An image model reads pictures only for edits and writes no text a grammar could constrain,
     * whatever its conditioning network's processor would take. */
    const bool img = opt_.image_model || opt_.video_model;
    caps["grammar"]   = !img && deps_.grammar != nullptr;
    caps["embedding"] = deps_.embed != nullptr;
    caps["image"]     = deps_.mm && deps_.mm->accepts(MediaKind::Image);
    caps["video"]     = !img && deps_.mm && deps_.mm->accepts(MediaKind::Video);
    caps["audio"]     = !img && deps_.mm && deps_.mm->accepts(MediaKind::Audio);
    caps["logprobs"]  = opt_.supports_logprobs;
    caps["image_generation"] = deps_.images != nullptr;
    caps["image_edit"]       = deps_.images != nullptr && deps_.images->info().max_refs > 0;
    caps["audio_generation"] = deps_.audio != nullptr;
    caps["video_generation"] = deps_.videos != nullptr;
    /* The efforts a chat request may send, probed from the template at startup; null when it
     * takes any word. "none" and "minimal" are this server's and turn thinking off. */
    if (lim_.reasoning_efforts.empty()) {
        caps["reasoning_efforts"] = nullptr;
    } else {
        std::vector<std::string> e = { "none", "minimal" };
        e.insert(e.end(), lim_.reasoning_efforts.begin(), lim_.reasoning_efforts.end());
        caps["reasoning_efforts"] = e;
    }
    return caps;
}

/* ================================================================== /server_info */
/* vLLM returns the repr() of its config; this returns the information as JSON: the server's own
 * state, plus the engine's, pre-rendered because the server knows nothing about devices,
 * containers or placement (ServerOptions::engine_info). `endpoints` is the router's own table, so
 * it cannot go stale. */
HttpResponse Server::handle_server_info(const HttpRequest&) {
    json srv;
    srv["model"]              = opt_.model_id;
    srv["version"]            = opt_.version;
    srv["max_model_len"]      = opt_.max_ctx;
    srv["max_num_seqs"]       = opt_.max_seqs;
    srv["queue_cap"]          = queue_depth_;
    /* null is `--default-max-tokens auto`: what the context leaves after the prompt. */
    srv["default_max_tokens"] = opt_.default_max_tokens > 0 ? json(opt_.default_max_tokens)
                                                            : json(nullptr);
    srv["max_tokens_cap"]     = opt_.max_tokens_cap > 0 ? json(opt_.max_tokens_cap) : json(nullptr);
    srv["max_n"]              = opt_.max_n;
    srv["max_stop_strings"]   = opt_.max_stops;
    srv["max_stop_bytes"]     = opt_.max_stop_bytes;
    srv["uptime_s"]           = unix_now() - started_at_;
    srv["in_flight"]          = in_flight_.load(std::memory_order_relaxed);

    json caps = capabilities();

    /* What a request that omits a field gets, under the request's own field names. */
    const SamplingParams& sp = opt_.default_sampling;
    json defs;
    defs["temperature"]        = sp.temp;
    defs["top_k"]              = sp.top_k;
    defs["top_p"]              = sp.top_p;
    defs["min_p"]              = sp.min_p;
    defs["typical_p"]          = sp.typical_p;
    defs["presence_penalty"]   = sp.pres_penalty;
    defs["frequency_penalty"]  = sp.freq_penalty;
    defs["repetition_penalty"] = sp.rep_penalty;
    defs["repeat_last_n"]      = sp.penalty_last_n;
    defs["dry_multiplier"]     = sp.dry_multiplier;
    defs["dry_base"]           = sp.dry_base;
    defs["dry_allowed_length"] = sp.dry_allowed_length;
    defs["dry_penalty_last_n"] = sp.dry_penalty_last_n;
    defs["dry_sequence_breakers"] = dry_breakers(sp.dry_seq_breakers,
                                                 opt_.default_dry_breakers_set);
    defs["xtc_probability"]    = sp.xtc_probability;
    defs["xtc_threshold"]      = sp.xtc_threshold;
    defs["max_tokens"]         = srv["default_max_tokens"];
    defs["reasoning_effort"]   = opt_.default_reasoning_effort.empty()
                                     ? json(nullptr) : json(opt_.default_reasoning_effort);
    json emap = json::object();
    for (const auto& [from, to] : lim_.reasoning_effort_map) emap[from] = to;
    defs["reasoning_effort_map"] = std::move(emap);
    defs["thinking_token_budget"] = opt_.default_thinking_budget < 0
                                        ? json(nullptr) : json(opt_.default_thinking_budget);
    json kw = json::object();
    for (const auto& [k, v] : opt_.default_template_kwargs) {
        json parsed = json::parse(v, nullptr, false);
        kw[k] = parsed.is_discarded() ? json(v) : std::move(parsed);
    }
    defs["chat_template_kwargs"] = std::move(kw);
    defs["reasoning_format"]     = opt_.reasoning_format;

    json o;
    o["server"]           = std::move(srv);
    o["request_defaults"] = std::move(defs);
    o["capabilities"]     = std::move(caps);
    o["endpoints"]    = router_.routes();

    /* Spliced in whole; the engine owns its contents. A blob that is not a JSON object is dropped
     * with a warning rather than corrupting the response. */
    if (!opt_.engine_info.empty()) {
        json e = json::parse(opt_.engine_info, nullptr, false);
        if (e.is_discarded() || !e.is_object())
            RAD_WARN("server: engine_info is not a JSON object; /server_info omits it");
        else
            o["engine"] = std::move(e);
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /diagnostics */
/* The startup report, regenerated per request (ServerOptions::diagnostics_json), behind the API
 * key like /server_info. Without a key it is served to loopback clients only: it describes the
 * host (kernel, boot parameters, PCI layout, memory limits), which a keyless server bound to
 * 0.0.0.0 would otherwise hand to anyone who can reach the port. */
HttpResponse Server::handle_diagnostics(const HttpRequest& req) {
    if (!opt_.diagnostics_json)
        return error_response(err_not_implemented(
            "the server was not given a diagnostics provider; this build has no engine attached"),
                req.peer);
    if (opt_.api_key.empty() && !req.peer_is_loopback()) {
        ApiError e;
        e.status = 403;
        e.type = "permission_error";
        e.message = "without --api-key, /diagnostics answers only loopback clients; start the "
                    "server with --api-key to read it from elsewhere";
        return error_response(e, req.peer);
    }
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, opt_.diagnostics_json());
}

/* ================================================================== /reset_prefix_cache */
/* 501 when there is no prefix cache to reset (vLLM answers 200 regardless, so a benchmark cannot
 * tell whether it started cold). Reports how many entries were dropped. */
HttpResponse Server::handle_reset_prefix_cache(const HttpRequest& req) {
    if (!deps_.sched)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);

    const int64_t n = deps_.sched->reset_prefix_cache();
    if (n < 0)
        return error_response(err_not_implemented(
            "this engine has no prefix cache (started with --no-prefix-cache, or the model "
            "declares no cacheable KV group)"), req.peer);

    json o;
    o["entries_dropped"] = n;
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(o));
}

namespace {

/* How long the HTTP thread waits for the step loop's answer. A calibration's step loop answers an
 * idle engine in microseconds; this only bounds the case where it has stopped (shutdown, or a set
 * that never finished), which is a busy answer, not a boundary. */
constexpr int kDraftCalibBoundaryWaitMs = 5000;

}  /* namespace */

/* ================================================================== /admin/draft-calib-boundary */
/* The calibration's phase boundary: the harness's one request between the warmup set it throws
 * away and the set it scores. Not a serving feature -- a deployment started without
 * --draft-profile-out answers 501, exactly as /reset_prefix_cache does without a cache, because a
 * 200 would tell the harness that the warmup no longer prices the scored set.
 *
 * 409 means the engine was not quiet: nothing was dropped and nothing is waited for, so the caller
 * decides whether to ask again. It is never a pause point and never aborts work. The wait for the
 * step loop's answer is bounded (kDraftCalibBoundaryWaitMs) and ends in that same busy answer, so
 * an engine that has stopped stepping cannot hang the HTTP thread. */
HttpResponse Server::handle_draft_calib_boundary(const HttpRequest& req) {
    if (!deps_.sched)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);

    const DraftCalibAck a = deps_.sched->draft_calib_boundary(kDraftCalibBoundaryWaitMs);
    if (a.rc == RAD_E_UNSUPPORTED)
        return error_response(err_not_implemented(
            "this deployment is not a draft calibration (started without --draft-profile-out), so "
            "it has no recording phase to bound"), req.peer);

    json o;
    o["epoch"] = a.epoch;
    if (a.rc == RAD_E_STATE) {
        metrics_.observe_http(409);
        o["accepted"] = false;
        o["reason"] =
            "the engine is not quiet (a request is queued or running, a step is open or issued "
            "ahead, or a cycle still owes its record); nothing was dropped -- ask again once the "
            "previous set has finished";
        return HttpResponse::json_body(409, json_dump(o));
    }
    metrics_.observe_http(200);
    o["accepted"] = true;
    o["buckets_dropped"] = a.buckets;
    o["samples_dropped"] = a.samples;
    o["cycles_dropped"] = a.cycles_dropped;
    return HttpResponse::json_body(200, json_dump(o));
}

/* ================================================================== /health, /ping */
/* Not counted in the HTTP status metrics (nor is /metrics): a load balancer probing every second
 * would dominate the 2xx/5xx ratio. */
HttpResponse Server::handle_health(const HttpRequest& req) { return handle_ping(req); }

HttpResponse Server::handle_ping(const HttpRequest&) {
    const bool stopping = stopping_.load();
    return HttpResponse::json_body(
        stopping ? 503 : 200,
        json_dump(health_body(opt_.model_id, stopping,
                              in_flight_.load(std::memory_order_relaxed), queue_depth_)));
}

}  /* namespace server */
}  /* namespace rad */
