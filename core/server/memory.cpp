/* memory.cpp -- /v1/memory: fork a conversation the cache holds, and archive one so a fork still
 * finds it after its client has compacted it away (core/sched/memory.h).
 *
 *   POST   /v1/memory/fork            questions over a session's or an archive's exact tokens
 *   POST   /v1/memory/archive         pin a session's conversation under an id
 *   GET    /v1/memory/archive         every archive
 *   GET    /v1/memory/archive/:id     one
 *   DELETE /v1/memory/archive/:id     unpin it
 *
 * A fork replays the conversation's tokens as the model saw them and appends each question as
 * the chat template renders a new user turn after an assistant turn. The prompt is then the
 * cached chain plus a few tokens: the prefix cache serves the chain, and on a hybrid model its
 * deepest snapshot (spec §7.3). Rendering the client's messages again would not do: a template
 * may drop earlier turns' reasoning, and a client may serialise tool calls differently.
 *
 * The template's text for that turn is found by rendering a stand-in conversation (a user turn,
 * an assistant turn, then the question) and taking what follows the assistant turn's content:
 * the assistant turn's closer, the question's turn and the generation prompt. So the fork needs
 * no knowledge of any template, only one that renders assistant content as given. Every other
 * field of a fork's body is a chat request's (sampling, max_tokens, reasoning, the control
 * vector, an adapter by `model`), parsed by the chat endpoint's own parser.
 *
 * These routes read and keep other clients' conversations, so without --api-key they answer
 * loopback clients only, as /diagnostics does.
 */
#include "server.h"
#include "server_priv.h"
#include "rad_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace rad {
namespace server {

namespace {

/* The stand-in assistant turn's content. Distinctive, plain ASCII and free of whitespace at its
 * ends, so no template trims or escapes it. */
const char kForkMark[] = "<<radiance-memory-fork-7f3c1e>>";
/* How many questions one fork may ask: each is a sequence over the whole conversation. */
const size_t kMaxQuestions = 16;

/* An archive id the client chose: what a path segment and a log line can hold as is. */
bool archive_id_ok(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_' || c == '.'))
            return false;
    return true;
}

ApiError not_found(const std::string& why) {
    ApiError e;
    e.status = 404;
    e.type = "not_found_error";
    e.message = why;
    return e;
}

ApiError conflict(int status, const std::string& why) {
    ApiError e;
    e.status = status;
    e.type = "invalid_request_error";
    e.message = why;
    return e;
}

json archive_json(const IScheduler::ArchiveStat& s) {
    json o;
    o["id"] = s.id;
    o["object"] = "memory.archive";
    o["session_id"] = s.label;
    o["tokens"] = s.tokens;
    o["pinned_tokens"] = s.pinned_tokens;
    o["snapshot_tokens"] = s.snapshot_pos;
    o["age_s"] = s.age_s;
    return o;
}

/* What the scheduler's answer about the memory means to a client. */
ApiError memory_error(int rc, const std::string& what) {
    if (rc == RAD_E_UNSUPPORTED)
        return err_not_implemented("this server keeps no prefix cache (--no-prefix-cache), so it "
                                   "has no conversation to fork or archive");
    if (rc == RAD_E_NOTFOUND) return not_found(what);
    return err_internal(fmt("%s: %s", what.c_str(), rad_strerror(rc)));
}

}  /* namespace */

std::optional<HttpResponse> Server::memory_refusal(const HttpRequest& req) {
    if (!deps_.sched || !deps_.tok)
        return error_response(err_not_implemented("the server has no engine attached"), req.peer);
    if (opt_.api_key.empty() && !req.peer_is_loopback()) {
        ApiError e;
        e.status = 403;
        e.type = "permission_error";
        e.message = "without --api-key, /v1/memory answers only loopback clients: it reads and "
                    "keeps other clients' conversations. Start the server with --api-key to use it "
                    "from elsewhere";
        return error_response(e, req.peer);
    }
    return std::nullopt;
}

/* ------------------------------------------------------------------ the fork */

HttpResponse Server::handle_memory_fork(const HttpRequest& req) {
    if (auto no = memory_refusal(req)) return *no;
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object())
        return error_response(err_bad("body", "expected a JSON object"), req.peer);

    /* ---- what to fork */
    const auto src = body.find("source");
    if (src == body.end() || !src->is_object())
        return error_response(err_bad("source", "expected {\"session\": ID} or {\"archive\": ID}"),
                              req.peer);
    IScheduler::Chain chain;
    std::string from_kind, from_id;
    if (src->contains("session") && (*src)["session"].is_string()) {
        from_kind = "session";
        from_id = session_label_of((*src)["session"].get<std::string>());
        const int rc = deps_.sched->session_chain(from_id, &chain);
        if (rc < 0)
            return error_response(memory_error(rc, "no request has finished under session_id '" +
                                  from_id + "'; send session_id with every request of the "
                                  "conversation"), req.peer);
    } else if (src->contains("archive") && (*src)["archive"].is_string()) {
        from_kind = "archive";
        from_id = (*src)["archive"].get<std::string>();
        const int rc = deps_.sched->archive_chain(from_id, &chain);
        if (rc < 0)
            return error_response(memory_error(rc, "no archive '" + from_id + "'"), req.peer);
    } else {
        return error_response(err_bad("source", "expected {\"session\": ID} or {\"archive\": ID}"),
                              req.peer);
    }
    if (chain.has_media)
        return error_response(conflict(422, "the conversation carried media, which a fork cannot "
                                            "replay from its tokens"), req.peer);
    if (chain.tokens.empty())
        return error_response(conflict(422, "the conversation holds no tokens"), req.peer);

    /* ---- the questions */
    std::vector<std::string> questions;
    const auto qs = body.find("questions");
    if (qs == body.end() || !qs->is_array() || qs->empty() || qs->size() > kMaxQuestions)
        return error_response(err_bad("questions", fmt("expected 1 to %zu strings", kMaxQuestions)),
                              req.peer);
    for (const auto& q : *qs) {
        if (!q.is_string() || q.get<std::string>().empty())
            return error_response(err_bad("questions", "every question must be a non-empty string"),
                                  req.peer);
        questions.push_back(q.get<std::string>());
    }
    bool at_end = false;
    if (const auto c = body.find("checkpoint_at_end"); c != body.end()) {
        if (!c->is_boolean())
            return error_response(err_bad("checkpoint_at_end", "expected a boolean"), req.peer);
        at_end = c->get<bool>();
    }

    /* ---- the rest is a chat request's, over a stand-in conversation per question. Fields that
     * would change what the turn renders, or how it is answered, are this route's own. */
    json chat = body;
    for (const char* k : { "source", "questions", "checkpoint_at_end", "messages", "stream",
                           "stream_options", "n", "session_id", "tools", "tool_choice",
                           "functions", "function_call", "parallel_tool_calls", "echo",
                           "logprobs", "top_logprobs", "continue_final_message",
                           "add_generation_prompt" })
        chat.erase(k);

    /* The stand-in is not the prompt that runs: its length bounds nothing. The real prompt is
     * checked against the context below. */
    OaiLimits lim = lim_;
    lim.max_ctx = 0;
    OaiRequest r;
    const size_t nq = questions.size();
    for (size_t i = 0; i < nq; ++i) {
        chat["messages"] = json::array({
            json{ { "role", "user" }, { "content", "." } },
            json{ { "role", "assistant" }, { "content", kForkMark } },
            json{ { "role", "user" }, { "content", questions[i] } } });
        OaiRequest one;
        ApiError err;
        if (parse_chat_request(json_dump(chat), odeps_, lim, &one, &err) < 0)
            return error_response(err, req.peer);
        const std::string& text = one.prompts[0].text;
        const size_t at = text.rfind(kForkMark);
        if (at == std::string::npos)
            return error_response(err_not_implemented(
                "this model's chat template does not render an assistant turn's content as given, "
                "so a fork cannot find where a new turn starts"), req.peer);
        const std::string suffix = text.substr(at + sizeof kForkMark - 1);
        std::vector<int32_t> tail = deps_.tok->encode(suffix, /*add_special=*/false,
                                                      /*parse_special=*/true);
        /* A conversation that ended on the turn's closer already holds the first suffix token. */
        if (!tail.empty() && chain.tokens.back() == tail.front()) tail.erase(tail.begin());

        PromptInput p;
        p.tokens.reserve(chain.tokens.size() + tail.size());
        p.tokens = chain.tokens;
        p.tokens.insert(p.tokens.end(), tail.begin(), tail.end());
        p.text = suffix;
        /* The first question writes the snapshot at the conversation's end, where an archive of
         * it resumes (Request::checkpoint_at). */
        if (i == 0 && at_end) p.checkpoint_at = (int64_t)chain.tokens.size();
        if (i == 0) {
            r = std::move(one);
            r.prompts[0] = std::move(p);
        } else {
            r.prompts.push_back(std::move(p));
        }
    }
    r.n = 1;
    r.stream = false;
    r.echo = false;
    /* A fork is not a turn of the conversation: it must not replace the session's tokens. */
    r.session_label.clear();
    /* The cache keys the chain under the control vector it ran with (media_keys); a fork that
     * names no choice runs the same way, so it hits. */
    if (r.control_vector < 0) r.control_vector = chain.use_control_vector ? 1 : 0;

    /* max_tokens was resolved without a context; the real prompt sets the room. */
    size_t longest = 0;
    for (const PromptInput& p : r.prompts) longest = std::max(longest, p.tokens.size());
    if (lim_.max_ctx > 0) {
        const int64_t room = lim_.max_ctx - (int64_t)longest;
        if (room <= 0) {
            ApiError e = err_bad("source", fmt("the conversation and the question are %zu tokens, "
                                               "and the context holds %lld", longest,
                                               (long long)lim_.max_ctx));
            e.code = "context_length_exceeded";
            return error_response(e, req.peer);
        }
        if (r.max_tokens <= 0 || r.max_tokens > room) r.max_tokens = (int32_t)room;
    }

    const int64_t n = (int64_t)nq;
    if (auto no = admit(n, req.peer, ApiShape::OpenAI, "questions")) return *no;
    RAD_INFO("memory fork admitted%s: %s '%s', %zu tokens, %zu question(s)",
             req.peer.empty() ? "" : (" from " + req.peer).c_str(), from_kind.c_str(),
             from_id.c_str(), chain.tokens.size(), nq);

    /* Questions run as one submission, except that a snapshot owed at the end is written by the
     * first one alone: the others then resume from it instead of each computing the stretch
     * since the last snapshot again (the /v1/systemone pattern). */
    std::vector<Generation> gens;
    auto run = [&](OaiRequest& part) -> std::optional<HttpResponse> {
        const size_t first = gens.size();
        ApiError e;
        if (submit_all(part, &gens, &e) < 0) {
            retire(gens);
            release(n);
            return error_response(e, req.peer);
        }
        /* submit_all numbers each call's choices from 0; usage_of reads them as one request's. */
        for (size_t i = first; i < gens.size(); ++i) {
            gens[i].index = (int)i;
            gens[i].prompt_index = (int)i;
        }
        for (;;) {
            pump(gens, part);
            if (all_done(gens)) return std::nullopt;
            if (auto no = abandoned(req, gens, n, ApiShape::OpenAI, &part)) return no;
            for (size_t i = first; i < gens.size(); ++i)
                if (!gens[i].closed) { gens[i].sink->wait(); break; }
        }
    };
    if (at_end && nq > 1) {
        OaiRequest head = r, rest = r;
        head.prompts.resize(1);
        rest.prompts.erase(rest.prompts.begin());
        if (auto no = run(head)) return *no;
        if (gens[0].finish != Finish::Error && gens[0].finish != Finish::Cancelled)
            if (auto no = run(rest)) return *no;
    } else {
        if (auto no = run(r)) return *no;
    }

    json answers = json::array();
    Finish overall = Finish::Stop;
    for (size_t i = 0; i < gens.size(); ++i) {
        Generation& g = gens[i];
        if (g.finish == Finish::Error || g.finish == Finish::Cancelled) {
            const std::string why = g.finish == Finish::Error
                ? engine_failure(*g.sink) : "the engine cancelled question " + std::to_string(i);
            record(r, gens, Finish::Error);
            retire(gens);
            release(n);
            return error_response(err_internal(why), req.peer);
        }
        std::string content = g.text, reasoning;
        if (r.reply_parser) {
            std::unique_ptr<IReplyReader> rd = r.reply_parser->open();
            rd->feed(g.text, nullptr);
            rd->finish(nullptr);
            const ParsedMessage m = rd->message();
            content = m.content;
            reasoning = m.reasoning;
        }
        if (g.finish == Finish::Length) overall = Finish::Length;
        json a;
        a["index"] = (int)i;
        a["question"] = questions[i];
        a["content"] = content;
        if (!reasoning.empty()) a["reasoning_content"] = reasoning;
        a["finish_reason"] = finish_reason_oai(g.finish);
        answers.push_back(std::move(a));
    }
    if (gens.size() < nq) {
        /* The first question failed and the rest never ran. */
        record(r, gens, Finish::Error);
        retire(gens);
        release(n);
        return error_response(err_internal("the first question failed"), req.peer);
    }

    const UsageOut u = usage_of(r, gens);
    record(r, gens, overall);
    retire(gens);
    release(n);

    json out;
    out["object"] = "memory.fork";
    out["source"] = json{ { from_kind, from_id } };
    out["source_tokens"] = (int64_t)chain.tokens.size();
    if (chain.lora_uid != 0 && r.model.empty())
        out["warning"] = "the conversation ran under an adapter and this fork names none in "
                         "`model`, so it computed the conversation again without it";
    out["answers"] = std::move(answers);
    out["usage"] = json{ { "prompt_tokens", u.prompt },
                         { "completion_tokens", u.completion },
                         { "total_tokens", u.prompt + u.completion },
                         { "prompt_tokens_details", json{ { "cached_tokens", u.cached } } } };
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(out));
}

/* ------------------------------------------------------------------ archives */

HttpResponse Server::handle_memory_archive_create(const HttpRequest& req) {
    if (auto no = memory_refusal(req)) return *no;
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded() || !body.is_object())
        return error_response(err_bad("body", "expected a JSON object"), req.peer);
    const auto sess = body.find("session_id");
    if (sess == body.end() || !sess->is_string() || sess->get<std::string>().empty())
        return error_response(err_bad("session_id", "expected the conversation's session_id"),
                              req.peer);
    const std::string label = session_label_of(sess->get<std::string>());
    std::string id;
    if (const auto a = body.find("archive_id"); a != body.end()) {
        if (!a->is_string() || !archive_id_ok(a->get<std::string>()))
            return error_response(err_bad("archive_id", "expected 1 to 128 of A-Z a-z 0-9 - _ ."),
                                  req.peer);
        id = a->get<std::string>();
    } else {
        id = random_id("arc_");
    }

    IScheduler::Chain chain;
    if (const int rc = deps_.sched->session_chain(label, &chain); rc < 0)
        return error_response(memory_error(rc, "no request has finished under session_id '" +
                              label + "'"), req.peer);
    if (chain.has_media)
        return error_response(conflict(422, "the conversation carried media, which a fork cannot "
                                            "replay from its tokens"), req.peer);

    /* On a model whose hits resume from a snapshot, one is owed at the conversation's end unless
     * a fork with checkpoint_at_end already wrote it: a request over the chain writes it there
     * and samples one token nobody reads. Not under an adapter, which this route cannot name;
     * that archive resumes from the deepest snapshot it has. */
    const int64_t owed = chain.lora_uid == 0 ? deps_.sched->snapshot_owed(chain) : 0;
    if (owed > 0) {
        if (auto no = admit(1, req.peer, ApiShape::OpenAI, "session_id")) return *no;
        std::vector<Generation> gens(1);
        Generation& g = gens[0];
        g.sink = std::make_shared<Sink>(16);
        std::vector<Request> one(1);
        Request& rq = one[0];
        rq.prompt = chain.tokens;
        rq.max_tokens = 1;
        rq.sp.temp = 0.0f;
        rq.sp.use_control_vector = chain.use_control_vector;
        rq.checkpoint_at = owed;
        rq.sink = g.sink.get();
        std::vector<uint64_t> ids;
        deps_.sched->submit_all(one, &ids);
        g.id = ids.empty() ? 0 : ids[0];
        g.sink->id = g.id;
        if (!g.id) {
            retire(gens);
            release(1);
            return error_response(err_internal("the scheduler refused the snapshot request"),
                                  req.peer);
        }
        while (!g.sink->done()) {
            if (auto no = abandoned(req, gens, 1)) return *no;
            g.sink->wait();
        }
        const Finish f = g.sink->finish_reason();
        const std::string why = f == Finish::Error ? engine_failure(*g.sink) : std::string();
        retire(gens);
        release(1);
        if (f == Finish::Error || f == Finish::Cancelled)
            return error_response(err_internal(why.empty() ? "the snapshot request was cancelled"
                                                           : why), req.peer);
    }

    IScheduler::ArchiveStat st;
    const int rc = deps_.sched->archive(id, label, chain, &st);
    if (rc == RAD_E_STATE)
        return error_response(conflict(409, "an archive '" + id + "' already exists"), req.peer);
    if (rc == RAD_E_FULL)
        return error_response(conflict(507, "the server keeps --memory-archives archives at once "
                                            "and has that many; delete one first"), req.peer);
    if (rc < 0) return error_response(memory_error(rc, "archiving failed"), req.peer);
    metrics_.observe_http(201);
    return HttpResponse::json_body(201, json_dump(archive_json(st)));
}

HttpResponse Server::handle_memory_archive_list(const HttpRequest& req) {
    if (auto no = memory_refusal(req)) return *no;
    std::vector<IScheduler::ArchiveStat> v;
    if (const int rc = deps_.sched->archives(&v); rc < 0)
        return error_response(memory_error(rc, "listing archives failed"), req.peer);
    json data = json::array();
    for (const auto& s : v) data.push_back(archive_json(s));
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(json{ { "object", "list" },
                                                        { "data", std::move(data) } }));
}

HttpResponse Server::handle_memory_archive_get(const HttpRequest& req) {
    if (auto no = memory_refusal(req)) return *no;
    const std::string id = req.param("id");
    std::vector<IScheduler::ArchiveStat> v;
    if (const int rc = deps_.sched->archives(&v); rc < 0)
        return error_response(memory_error(rc, "listing archives failed"), req.peer);
    for (const auto& s : v)
        if (s.id == id) {
            metrics_.observe_http(200);
            return HttpResponse::json_body(200, json_dump(archive_json(s)));
        }
    return error_response(not_found("no archive '" + id + "'"), req.peer);
}

HttpResponse Server::handle_memory_archive_delete(const HttpRequest& req) {
    if (auto no = memory_refusal(req)) return *no;
    const std::string id = req.param("id");
    if (const int rc = deps_.sched->unarchive(id); rc < 0)
        return error_response(memory_error(rc, "no archive '" + id + "'"), req.peer);
    metrics_.observe_http(200);
    return HttpResponse::json_body(200, json_dump(json{ { "id", id },
                                                        { "object", "memory.archive" },
                                                        { "deleted", true } }));
}

}  /* namespace server */
}  /* namespace rad */
