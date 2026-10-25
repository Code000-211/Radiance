/* core/server/oai.h -- the OpenAI schema, in and out.
 *
 * Errors are values. Nothing here throws; every entry point returns RAD_OK or a negative status
 * and fills an ApiError carrying its HTTP status, `param` and a message naming the problem. An
 * exception crossing the transport would close the connection with no body.
 *
 * A parameter that would change the answer and is not served is a 400 naming it, never a silent
 * ignore, so a client is not told its `logit_bias` was applied when it was not. Such a field is
 * accepted at its "off" value, which llama.cpp-built clients send on every request. An unknown
 * field is ignored, as llama.cpp and vLLM do (oai.cpp, key vetting).
 */
#pragma once
#include "iface.h"
#include "rad_core.h"
#include "sink.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace rad {
namespace server {

/* ================================================================== errors */

struct ApiError {
    int         status = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param;      /* "" when the error is not about one field */
    std::string code;       /* "" when there is nothing more specific to say */
    /* The legal values of `param` when the error is a value outside a known set; sent as
     * `supported` so a client can retry with one. */
    std::vector<std::string> supported;
    /* For a prompt the context cannot hold: its length and the context's, under llama-server's
     * names, for clients that compact their history on overflow. -1 leaves them out. */
    int64_t     n_prompt_tokens = -1;
    int64_t     n_ctx = -1;

    std::string dump() const;    /* the OpenAI error envelope, as a JSON string */
};

ApiError err_bad(const std::string& param, const std::string& why);
ApiError err_unsupported(const std::string& param, const std::string& why);
ApiError err_not_implemented(const std::string& what);
ApiError err_overloaded(const std::string& why);
ApiError err_internal(const std::string& why);

/* DRY's sequence breakers when a request names none: the deployment's when it set any (`set`, even
 * to an empty list), otherwise llama.cpp's four (newline, colon, quote, asterisk). */
std::vector<std::string> dry_breakers(const std::vector<std::string>& deployment, bool set);

/* Out of line on purpose: inlined, nlohmann's dump() trips a GCC 16 false-positive warning
 * (a speculatively devirtualised shared_ptr release reported as an out-of-bounds destroy) that
 * comes and goes with how deeply the caller inlines. */
std::string json_dump(const json& j);

/* ================================================================== a parsed request */

enum class Endpoint { Chat, Completion, Embedding };

/* The API a chat request came in through, which decides the shape of its answer. Every shape is
 * parsed into the same chat request (Anthropic's is translated first, anthropic_to_chat) and
 * served by the same path; only the wire bytes differ. */
enum class ApiShape { OpenAI, Anthropic, Responses, LlamaNative };

struct PromptInput {
    std::vector<int32_t> tokens;
    std::string          text;    /* the rendered prompt; needed for `echo` and for the logs */
    /* The media parts, processed and placed; null for a text prompt (Request::media). */
    std::shared_ptr<mm::PromptMedia> media;
    /* Where a snapshot is wanted on the way (Request::checkpoint_at); 0 for none. Set only by
     * the memory fork, at the end of the conversation it continues. */
    int64_t              checkpoint_at = 0;
};

struct OaiRequest {
    Endpoint    endpoint = Endpoint::Chat;
    std::string model;
    std::string id;               /* "chatcmpl-..." / "cmpl-..." */
    int64_t     created = 0;      /* unix seconds */

    ApiShape shape = ApiShape::OpenAI;
    bool stream = false;
    bool include_usage = false;   /* stream_options.include_usage */
    /* llama.cpp's streaming extras: `timings` on every chunk, not only the last; and
     * `prompt_progress` chunks while the prompt is read. */
    bool timings_per_token = false;
    bool return_progress = false;
    /* The client's conversation name (OpenRouter's `session_id`), "" if none; only a label on
     * /sessions. */
    std::string session_label;
    /* The build, as OpenAI's and llama.cpp's `system_fingerprint`. */
    std::string system_fingerprint;

    int             n = 1;
    SamplingParams  sp;
    /* The control vector, tri-state: -1 the request named no choice (the deployment's
     * runtime default applies), 0 it asked for the stock model, 1 for the loaded control vector.
     * Parsed from `control_vector` (or Strata's `experimental_speed_projection`); a deployment
     * with no vector refuses 1 rather than silently serving the stock model. */
    int             control_vector = -1;
    int32_t         max_tokens = 0;
    std::vector<std::string> stop;
    /* How many of `stop` the client sent; the rest are the chat format's end-of-turn markers.
     * A cut on a client stop is Anthropic's `stop_sequence`, on the rest its `end_turn`. */
    size_t          n_client_stops = 0;
    /* The assistant text a continued conversation ends in (assistant prefill), "" otherwise. The
     * OpenAI shape answers the whole message, as llama-server does; Anthropic's answers only what
     * follows it, as Anthropic's API does. */
    std::string     continued;
    /* Anthropic's `context_management` was sent: the answer reports `applied_edits: []`, as the
     * conversation is served as sent. */
    bool            context_management = false;
    /* The Responses API request fields its response object echoes (instructions, tools,
     * tool_choice, text, reasoning, ...), as a JSON object; "" for the other shapes. */
    std::string     echo_json;
    /* The Responses API's namespaced function tools, flattened for the chat path: the flattened
     * name ("<namespace>__<name>") to its namespace and name, which a function_call item reports
     * apart. Empty for the other shapes. */
    std::map<std::string, std::pair<std::string, std::string>> tool_namespace;
    /* llama-server's native /completion: the generated token ids beside the text. */
    bool            return_tokens = false;
    /* The reasoning block's closer when the reply starts inside the block, else "": the stop
     * strings then apply only after it (IDetokenizer::defer_stops_until). */
    std::string stops_after;
    /* Generate past end-of-text, to max_tokens. See Request::ignore_eos. */
    bool        ignore_eos = false;
    int             priority = 0;

    std::vector<PromptInput> prompts;   /* /v1/completions accepts a batch; chat has exactly one */
    bool echo = false;

    int  n_logprobs = 0;          /* 0 = off */
    bool want_logprobs = false;

    bool tools_present = false;
    /* OpenAI's deprecated functions API: the request sent `functions`, so the reply carries its
     * call as `function_call` (one at most) and ends in finish_reason "function_call". */
    bool legacy_functions = false;
    /* For the request log, counted during the parse: the messages and tools a chat request
     * carried, and a bounded one-line opening of the last user message for the debug level.
     * Zero and empty for completion requests. */
    int         n_messages = 0;
    int         n_tools = 0;
    std::string last_user;
    std::string peer;             /* the handler sets it from the transport; "" in-process */
    int64_t     t_received_ns = -1;   /* HttpRequest::t_received_ns, for timings' ttft and total */
    bool parse_tools = false;
    /* The reply parser this request's render produced. Per request, not per server: it binds the
     * model's format to the tools this call sent, so concurrent requests may need different
     * parsers. */
    std::shared_ptr<IReplyParser> reply_parser;

    /* Control tokens this reply must be decoded as text, so `tool_parser` can see its structure.
     * See ITokenizer::detokenizer. */
    std::vector<std::string> preserved_tokens;

    /* /v1/embeddings */
    bool base64_embeddings = false;
    /* OpenAI's `dimensions`: the vector's first `dimensions` components, renormalised (a
     * Matryoshka-trained model's shorter embedding). 0 is the model's own width. */
    int64_t dimensions = 0;
    /* /v1/embeddings: a soft-token budget for the request's images, 0 the server's. */
    int32_t image_tokens = 0;

    int  n_choices() const { return (int)prompts.size() * n; }
};

/* What the server needs to know about itself to validate a request. */
struct OaiLimits {
    std::string model_id = "radiance";
    std::string version;                  /* the build, for `system_fingerprint` */
    int64_t     max_ctx = 0;              /* 0 = unknown, no length check */
    /* What a request that names no max_tokens may generate. 0, the default as in vLLM, is
     * "whatever the context leaves", resolved against the prompt and max_ctx in
     * clamp_max_tokens. A small fixed default ends a reasoning model's answer inside its
     * reasoning with finish_reason "length", which chat clients do not show. */
    int64_t     default_max_tokens = 0;
    /* The most any request may generate; a larger max_tokens is clamped, not refused. 0 = none. */
    int64_t     max_tokens_cap = 0;
    int         max_n = 8;
    /* How many stop strings a request may send, and how long each may be. */
    int64_t     max_stops = 64;
    int64_t     max_stop_bytes = 4096;
    bool        allow_image = false;
    bool        allow_video = false;
    bool        allow_audio = false;
    bool        supports_logprobs = false;

    /* The deployment's default reasoning effort; empty uses the template's own default. Needed
     * because a template's default is not always servable: Qwen3.8's is `xhigh`, which on a
     * short task can spend the whole budget inside <think> and return empty content with
     * finish_reason "length". Most OpenAI clients never send the field. Applied only when the
     * request says nothing, so a caller always wins. */
    std::string default_reasoning_effort;

    /* The words the chat template accepts as `reasoning_effort`, found at startup by rendering
     * each known word (Server::probe_reasoning_efforts). Empty when the template takes any word
     * or nothing was probed. A refused effort is answered with this set instead of the
     * template's traceback (template_refusal). */
    std::vector<std::string> reasoning_efforts;

    /* --reasoning-effort-map: a request's effort word replaced by the deployment's before the
     * template sees it, e.g. OpenAI's "high" to a template's "xhigh". */
    std::map<std::string, std::string> reasoning_effort_map;

    /* A startup probe, not a request: the template's refusal is the expected answer and is not
     * logged (ChatRenderOptions::quiet). */
    bool        probe = false;

    /* The deployment's default `thinking_token_budget`, -1 for none. Agent frameworks size
     * max_tokens per turn and send no budget, and a reasoning block that spends all of max_tokens
     * leaves no answer. Applied only to a request that names no budget, and only where the
     * reply's reasoning block can be counted. */
    int64_t     default_thinking_budget = -1;

    /* The sampler a request that sends no sampler fields is served with, field by field.
     *
     * The struct's own defaults truncate only to the sampler's 1024 candidates (temperature 1,
     * top_k 0, top_p 1). A model tuned against a tighter cut occasionally draws from far down its
     * distribution without one, and an early end-of-turn ends an agent's turn with no tool call.
     * The deployment resolves these from what the checkpoint states.
     *
     * Only the sampler knobs are read; grammar fields are per request and filled in after this is
     * copied. A field the request names always wins. */
    SamplingParams default_sampling;
    /* SamplingParams cannot tell a stated empty breaker list from an unstated one: unstated is
     * llama-server's four, stated-empty is none. */
    bool           default_dry_breakers_set = false;

    /* The deployment's chat template variables, key -> JSON text, applied under the request's own
     * chat_template_kwargs so the request wins per key. The thinking keys follow parse_thinking's
     * rule: a request that says anything about thinking replaces them whole. */
    std::map<std::string, std::string> default_template_kwargs;
    /* "auto" returns reasoning as reasoning_content; "none" leaves it inline in content. */
    std::string reasoning_format = "auto";
};

/* What a request says about thinking, in any of its three spellings: `effort` when it names a
 * non-empty reasoning effort (top-level or in chat_template_kwargs), `thinking` when it says
 * anything at all (an effort, or enable_thinking in either place). */
struct RequestThinking {
    bool thinking = false;
    bool effort = false;
};
RequestThinking request_thinking(const json& b);

/* The deployment's template variables and reply format, onto `opt`, before the request's own
 * chat_template_kwargs are written over them. A request that says anything about thinking keeps
 * the deployment's enable_thinking out; one that names an effort keeps its reasoning_effort out.
 * The chat endpoint and /tokenize both render through this, so both count the same prompt. */
void apply_template_defaults(const json& b, const OaiLimits& lim, ChatRenderOptions* opt);

/* --reasoning-effort-map, applied to the effort chat_template_kwargs put in `opt`: a mapped word
 * replaces it, and one mapped to "none" or "minimal" turns thinking off instead. The chat endpoint
 * maps the top-level `reasoning_effort` itself; this handles the kwargs spelling for it and for
 * /tokenize. */
void map_reasoning_effort(const OaiLimits& lim, ChatRenderOptions* opt);

/* The 400 for a template that refused a render. An effort outside OaiLimits::reasoning_efforts is
 * reported against `reasoning_effort` with the set in `supported`, as OpenAI reports an
 * unsupported value; anything else against `messages` with the template's own sentence. */
ApiError template_refusal(const ChatRenderOptions& opt, const OaiLimits& lim, int rs,
                          const std::string& why);

/* True when `v` is legal for the sampler field a request spells `field`; otherwise `why` states
 * the legal set as a 400 would. The same table checks requests and the command-line flag that
 * sets the field's default, so an accepted default is always servable. A field with no bounds
 * takes any value. */
bool sampler_value_ok(const char* field, double v, std::string* why);

/* A conversation's label (OpenRouter's session_id, Anthropic's metadata.user_id) as the engine keeps
 * it: at most 128 bytes, cut at a UTF-8 character boundary. A lookup by label cuts the same way. */
std::string session_label_of(const std::string& s);

struct OaiDeps {
    ITokenizer*       tok = nullptr;
    IChatTemplate*    chat = nullptr;
    IGrammarCompiler* grammar = nullptr;
    IMultimodal*      mm = nullptr;
    IEmbedder*        embed = nullptr;
};

/* The request-key policy (oai.cpp, key vetting) as lists, so a test can cover every key. */
struct KeyPolicy {
    std::vector<std::string> common, chat, completion, embedding;   /* accepted per endpoint */
    std::vector<std::string> inert, inert_prefixes;
    std::vector<std::string> guarded;   /* refused unless at their neutral value */
};
KeyPolicy request_key_policy();

/* Parse. `body` is the raw request body; a malformed body is a 400 that says where. */
int parse_chat_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       OaiRequest* out, ApiError* err);
/* `native` is llama-server's /completion: `return_tokens`, and a prompt array of strings and
 * token ids mixed as one prompt, as llama-server reads it. */
int parse_completion_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err, bool native = false);
/* A media part's bytes out of a data: URL; any other URL is refused, never fetched. */
int decode_media_url(const std::string& url, const std::string& field,
                     std::vector<uint8_t>* out, ApiError* err);
/* Tokenise a prompt holding one RAD_MEDIA_MARKER per media part, replacing each marker with its
 * part's processed token run; `media` is consumed in order. */
int tokenize_with_media(const std::string& rendered, const OaiDeps& d,
                        std::vector<std::pair<MediaKind, std::vector<uint8_t>>>& media,
                        const MediaOpts& opts, PromptInput* out, ApiError* err);
/* Embeddings: fills `out->prompts` with one entry per input. */
int parse_embeddings_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err);

/* ================================================================== responses */

/* One emitted token's logprobs as the wire carries them. `bytes` is the token's bytes and `text`
 * their longest well-formed UTF-8 prefix (a byte-level token can hold half a character, which a
 * JSON string cannot), as llama-server spells it. `end` is where the token's text ends in the
 * choice's text, which decides whether it went out before a stop string cut the reply. */
struct LogprobAlt {
    int32_t     token = 0;
    std::string text, bytes;
    float       logprob = 0.0f;
};
struct LogprobEntry {
    int32_t     token = 0;
    std::string text, bytes;
    float       logprob = 0.0f;
    std::vector<LogprobAlt> top;
    size_t      end = 0;
};

/* The entries as comma-joined JSON array elements, in OpenAI's chat shape with llama-server's
 * `id` beside each token; also llama-server's native `completion_probabilities`. */
std::string logprobs_content(const LogprobEntry* e, size_t n);
/* OpenAI's legacy /v1/completions `logprobs` object, `offset` the text offset of the first entry,
 * with llama-server's `content` array beside the legacy arrays. */
std::string logprobs_legacy(const LogprobEntry* e, size_t n, size_t offset);

struct ChoiceOut {
    int           index = 0;
    std::string   text;          /* raw generated text */
    std::string   stop_word;     /* the stop string that cut it, "" when none or unknown */
    ParsedMessage msg;           /* chat: content / reasoning / tool_calls after parsing */
    Finish        finish = Finish::None;
    std::vector<LogprobEntry> logprobs;
};

struct UsageOut {
    int64_t prompt = 0;
    int64_t completion = 0;
    int64_t cached = 0;
    /* Of `completion`, the tokens inside the reasoning block, markers included (OpenAI's
     * completion_tokens_details.reasoning_tokens). -1 when the block cannot be counted; the
     * breakdown is then left out. */
    int64_t reasoning = -1;
};

/* One request's timing, in llama.cpp's `timings` shape (read by proxies and dashboards built for
 * llama-server, e.g. llama-swap). /metrics holds the same quantities summed over all requests.
 *
 *   prompt_n      prompt tokens computed, excluding prefix-cache hits; cache_n is those
 *   prompt_ms     from the request's first scheduled step to its first token: the prefill,
 *                 without the admission wait
 *   predicted_n   tokens generated: usage.completion_tokens
 *   predicted_ms  from the first token to the last
 *   draft_n, draft_n_accepted
 *                 speculative tokens proposed for this request and accepted; present when any
 *                 were proposed, as in llama.cpp
 *   ttft_ms       from having the whole request body to the first token: parse, chat template,
 *                 tokenisation, admission wait and prefill. Body transfer time is excluded, so
 *                 a remote client sees the server's own share
 *   total_ms      from the same moment to the last token
 *
 * Rates are prompt_n / prompt_ms and (predicted_n - 1) / predicted_ms, since the first token
 * belongs to the prompt phase; the latter matches /metrics' time per output token. A rate with a
 * zero count or time is null. Over several choices the counts are summed and each time spans the
 * earliest choice's start to the latest choice's end. */
struct TimingsOut {
    bool    valid = false;       /* some choice produced a token; nothing is reported otherwise */
    int64_t prompt_n = 0, cache_n = 0;
    double  prompt_ms = -1.0;    /* < 0: the engine did not report the admission */
    int64_t predicted_n = 0;
    int64_t decode_n = 0;        /* of predicted_n, the tokens after each choice's first */
    double  predicted_ms = 0.0;
    int64_t draft_n = 0, draft_n_accepted = 0;
    double  ttft_ms = -1.0, total_ms = -1.0;   /* < 0: the body's arrival is not known */
};

/* `timings` is added at the top level when it is valid. */
std::string chat_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t = TimingsOut());
std::string text_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t = TimingsOut());
std::string embeddings_body(const OaiRequest& r, const std::vector<std::vector<float>>& vecs,
                            const UsageOut& u);
/* One model entry: what /v1/models lists and /v1/models/:id returns. */
struct ModelCard {
    std::string id;
    int64_t     created = 0;
    /* The served context (--max-model-len, else the container's trained length), vLLM's model
     * card field. Clients size their own context from it (history compression, UI meters);
     * without it they guess from the model name. 0 leaves the field out. */
    int64_t     max_model_len = 0;
    /* What the server loaded (ServerOptions::model_path), vLLM's `root`. Empty leaves it out. */
    std::string root;
    /* What the model can be asked for (tools, reasoning, images), as /server_info reports it.
     * OpenAI's model object has no such field. A JSON object as text, spliced in verbatim; empty
     * leaves it out. */
    std::string capabilities;
    /* A runtime LoRA adapter's card names the model it runs on, vLLM's `parent`. Empty for the
     * served model itself, which then carries no such field. */
    std::string parent;
};
/* The list, for GET /v1/models. */
std::string models_body(const ModelCard& m);
/* ...with the runtime LoRA adapters after the served model. */
std::string models_body(const std::vector<ModelCard>& cards);
/* The entry itself, for GET /v1/models/:id: OpenAI's retrieve answers the model object, not a
 * list of one, and its SDK reads `id` off the top level. */
std::string model_body(const ModelCard& m);

/* The usage chunk of each endpoint's stream: an SSE event with an empty choices array and only
 * `usage`. Sent when stream_options.include_usage is set, once per stream after every choice's
 * final chunk, since `n` choices share one prompt and one total. */
std::string text_usage_chunk(const OaiRequest& r, const UsageOut& u,
                             const TimingsOut& t = TimingsOut());
std::string chat_usage_chunk(const OaiRequest& r, const UsageOut& u,
                             const TimingsOut& t = TimingsOut());
/* The `timings` object, for a stream's final chunks. */
json timings_json(const TimingsOut& t);

/* Whether every call in a parsed reply is runnable: none left open, each named, each one's
 * arguments a single JSON value (json_partial.h). This alone decides finish_reason
 * `tool_calls`, streamed or not. */
bool calls_complete(const ParsedMessage& m);

/* ================================================================== streaming
 *
 * One per choice. Feeds the generated text to the choice's reply reader as it grows and writes
 * each increment the reader completes as an SSE chunk. The reader never revises what it has
 * delivered, so nothing here diffs or holds back: the concatenated deltas equal the buffered
 * answer.
 *
 * Reading the reply is shared by every API; the subclass owns the wire shape of an increment:
 * OpenAI's chunk (ChatDeltaStream), Anthropic's event (AnthropicStream) or the Responses API's
 * (ResponsesStream).
 */
class ReplyStream {
public:
    ReplyStream(const OaiRequest& r, int index, const IReplyParser* parser);
    virtual ~ReplyStream() = default;

    /* What the stream sends when it opens, before the request has run: the Responses API's
     * response.created and response.in_progress, which report the response object's state as
     * OpenAI's do. The chat and Anthropic streams send nothing here. */
    virtual std::string first_chunk() = 0;
    /* What the stream sends once the reply's first token exists, ahead of any of its text, which
     * a reader or a stop string may still hold back: OpenAI's opening chunk, Anthropic's
     * message_start. A client that times its first event then times the first token. */
    virtual std::string started() { return std::string(); }

    /* Feed the text generated since the last push; returns the bytes for what it completed.
     * `is_final` says the generation is over, which resolves whatever the reader was holding. */
    std::string push(std::string_view appended, bool is_final);

    /* What closes the choice: the finish reason, and this choice's `timings` when valid
     * (llama.cpp sends them on the final chunk). `u` is the request's usage, for the shapes that
     * close a message with it. */
    virtual std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) = 0;
    /* Closes a choice the engine failed, in place of final_chunk, with the shape's own failure so
     * a client never takes a truncated reply as complete. OpenAI's chunks have no failure event,
     * so the default is the final chunk with finish_reason "error" followed by the error object. */
    virtual std::string error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u);

    /* The whole generation parsed (the non-streaming shape). Complete once push has been called
     * with is_final. */
    const ParsedMessage& parsed() const { return last_; }
    Finish adjusted_finish(Finish f) const;

    /* llama.cpp's `timings_per_token`, for the shapes that carry timings. */
    virtual void set_timings(const TimingsOut&) {}
    /* Logprobs entries (logprobs_content's elements) for the next chunk written, for the shapes
     * that carry them; the others drop them. */
    virtual void add_logprobs(const std::string&) {}

protected:
    /* The bytes for the increments one push completed. Each emitter folds every increment into
     * `last_` through absorb() first. */
    virtual std::string emit(const std::vector<ReplyDelta>& ev) = 0;
    bool absorb(const ReplyDelta& d);
    static void quote(std::string& out, std::string_view s);

    std::unique_ptr<IReplyReader> reader_;
    std::string id_, model_, fingerprint_;
    int64_t     created_;
    int         index_;
    bool        finished_ = false;

    ParsedMessage            last_;
    std::vector<ReplyDelta>  ev_;
};

/* Anthropic's Messages API, translated into the equivalent /v1/chat/completions body (`*chat`),
 * so every refusal and default is the chat endpoint's. `*label` is the conversation label if the
 * request carries one (metadata.user_id); `*context_management` is set when the request sent
 * one (OaiRequest::context_management). Claude Code's per-request billing stamp in the system
 * prompt is held constant, as llama-server does, so consecutive turns can reuse the prefix
 * cache. */
int anthropic_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* label, ApiError* err, bool* context_management = nullptr);
/* The non-streaming answer, and an error, in Anthropic's shapes. */
std::string anthropic_message_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u);
std::string anthropic_error_body(const ApiError& e);
/* OpenAI's Responses API, translated into the equivalent chat request like Anthropic's. `*echo`
 * holds the request fields the response object repeats. previous_response_id needs server-side
 * storage and is refused; non-function tool types have no chat equivalent and are dropped, as
 * llama-server does. A namespace's function tools are flattened to "<namespace>__<name>", and
 * `*ns` (when given) maps each flattened name back for the reply's function_call items. */
int responses_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* echo, ApiError* err,
                      std::map<std::string, std::pair<std::string, std::string>>* ns = nullptr);
std::string responses_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u);

/* Anthropic's stop_reason for a finish and the stop string that caused it. */
const char* anthropic_stop_reason(const OaiRequest& r, Finish f, const std::string& stop_word);

/* OpenAI's chat.completion.chunk. The opening chunk (`role: assistant`, empty content) goes out
 * with the choice's first token (started), not when the stream opens, as vLLM and llama-server
 * send it: a client that times its first data chunk (llama-benchy's ttfr) then times the first
 * token, not the start of the prefill. It precedes every other chunk of the choice, also when
 * the reply has no text. */
class ChatDeltaStream final : public ReplyStream {
public:
    ChatDeltaStream(const OaiRequest& r, int index, const IReplyParser* parser)
        : ReplyStream(r, index, parser), legacy_(r.legacy_functions) {}
    std::string first_chunk() override;
    std::string started() override;
    std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) override;

    /* llama.cpp's `timings_per_token`: the timings the next chunks carry beside `choices`, as the
     * final chunk does. Invalid timings carry nothing. */
    void set_timings(const TimingsOut& t) override;
    /* OpenAI's `logprobs.content` on the next chunk out. Entries wait for a chunk because a
     * push may complete no delta (a tool call's opening, a held-back byte). */
    void add_logprobs(const std::string& elems) override;

private:
    std::string emit(const std::vector<ReplyDelta>& ev) override;
    std::string chunk(json delta, const char* finish_reason, const TimingsOut* t = nullptr);
    std::string lp_;       /* pending logprobs entries, comma-joined */

    /* Delta chunks are written, not built: the frame around the delta object is serialised once
     * and the delta written with json_quote_append. The bytes match what chunk() writes. */
    void delta_open(std::string& out);
    void delta_close(std::string& out);
    /* The opening chunk, once, ahead of the first one written. */
    void open_role(std::string& out);
    bool        role_sent_ = false;
    std::string head_;     /* `data: {...,"choices":[{"index":N,"delta":`, built on first use */
    std::string timings_;  /* `"timings":{...}` for every delta chunk, or "" */
    bool        legacy_;   /* OaiRequest::legacy_functions */
};

/* Anthropic's events: message_start once the first token exists (started); each content block
 * (thinking, text, each tool_use) opened, filled by deltas and stopped; then message_delta with
 * the stop reason and usage, and message_stop. message_start precedes every other event, an
 * error included. The prefill of a continued conversation is not repeated (`continued`). */
class AnthropicStream final : public ReplyStream {
public:
    AnthropicStream(const OaiRequest& r, int index, const IReplyParser* parser);
    std::string first_chunk() override;
    std::string started() override;
    std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) override;
    /* Anthropic's `error` event, which ends the stream with no message_stop. */
    std::string error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u) override;
    /* The stop string that cut the reply, for stop_reason and stop_sequence. */
    void set_stop_word(const std::string& w) { stop_word_ = w; }

private:
    enum class Block { None, Thinking, Text, Tool };
    std::string emit(const std::vector<ReplyDelta>& ev) override;
    void open(std::string& out, Block b, const ToolCall* tc);
    void close(std::string& out);
    void text_delta(std::string& out, const std::string& text);
    void release_held(std::string& out);
    static void event(std::string& out, const char* type, const json& data);

    const OaiRequest* r_;
    int64_t     n_input_ = 0;
    std::string skip_;           /* the prefill still to be passed over */
    std::string held_;           /* content matching the start of `skip_` so far, not yet sent */
    std::string stop_word_;
    Block       block_ = Block::None;
    int         n_blocks_ = 0;   /* blocks opened so far: the next one's index */
    bool        started_ = false;   /* message_start sent */
};

/* The Responses API's events: response.created and .in_progress; per output item (reasoning, the
 * message, each function call) item added, content part added, deltas, done; then
 * response.completed (or .incomplete at the token limit) with the whole response. Every event
 * carries output_index, content_index and sequence_number, by which the OpenAI SDK's stream
 * accumulator places deltas. */
class ResponsesStream final : public ReplyStream {
public:
    ResponsesStream(const OaiRequest& r, int index, const IReplyParser* parser);
    std::string first_chunk() override;
    std::string final_chunk(Finish f, const TimingsOut& t, const UsageOut& u) override;
    /* response.failed, with the items finished so far and the error. */
    std::string error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u) override;

private:
    enum class Item { None, Reasoning, Message, Call };
    std::string emit(const std::vector<ReplyDelta>& ev) override;
    void open(std::string& out, Item k, size_t call);
    void close(std::string& out);
    void event(std::string& out, const char* type, json data);
    json response(const char* status) const;

    const OaiRequest* r_;
    Item        item_ = Item::None;
    std::string item_id_, text_;  /* the open item's id and its text so far */
    size_t      call_ = 0;            /* the open call's index into last_.tool_calls */
    std::vector<json> items_;  /* every closed item, in output order */
    int64_t     seq_ = 0;
    UsageOut    usage_;
    const char* incomplete_ = nullptr;
};

/* The SSE terminator. Written once at the end of every stream, failed ones included, since a
 * blocked client reader has no other way to learn the stream is over. */
extern const char* const SSE_DONE;

std::string sse_event(const std::string& json_text);
std::string sse_error(const ApiError& e);

/* ================================================================== small shared helpers */

std::string base64_encode(const void* data, size_t len);
bool        base64_decode(const std::string& in, std::vector<uint8_t>* out);
/* A token id as a request spells it: an integer in [0, INT32_MAX]. False otherwise, and the
 * caller refuses the request; narrowing would make 2^32 + 5 an alias of token 5. */
bool        token_id(const json& v, int32_t* out);

/* Parse a request body into a tree of at most json_value_budget(max_ctx) values, nested at most
 * kJsonMaxDepth deep. A syntax error returns discarded, as json::parse does. Exceeding the value
 * budget sets `*over`, exceeding the depth sets `*deep`, and either returns a partial tree the
 * caller refuses. See the definition for why the tree, not just the body, is bounded. */
json        parse_json_bounded(const std::string& body, int64_t max_ctx, bool* over, bool* deep);
/* The body of a generation endpoint: parse_json_bounded, and a 400 naming the broken bound or
 * saying it is not one JSON object. */
int         parse_request_object(const std::string& body, const OaiLimits& lim, json* b,
                                 ApiError* err);
int64_t     json_value_budget(int64_t max_ctx);
constexpr int kJsonMaxDepth = 1024;

std::string random_id(const char* prefix);
int64_t     unix_now();

}  /* namespace server */
}  /* namespace rad */
