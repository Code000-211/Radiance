#include "oai.h"
#include "http.h"
#include "json_partial.h"
#include "rad_internal.h"
#include "mm/processor.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <random>
#include <set>
#include <type_traits>

namespace rad {
namespace server {

const char* const SSE_DONE = "data: [DONE]\n\n";

/* ================================================================== small helpers */

int64_t unix_now() { return (int64_t)std::time(nullptr); }

std::string json_dump(const json& j) { return j.dump(); }

std::string random_id(const char* prefix) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* hex = "0123456789abcdef";
    uint64_t a = rng(), b = rng();
    std::string s = prefix;
    for (int i = 0; i < 16; ++i) s.push_back(hex[(a >> (i * 4)) & 0xF]);
    for (int i = 0; i < 8; ++i)  s.push_back(hex[(b >> (i * 4)) & 0xF]);
    return s;
}

static const char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16 | (uint32_t)p[i + 1] << 8 | p[i + 2];
        out.push_back(kB64[(v >> 18) & 63]); out.push_back(kB64[(v >> 12) & 63]);
        out.push_back(kB64[(v >> 6) & 63]);  out.push_back(kB64[v & 63]);
    }
    if (i < len) {
        uint32_t v = (uint32_t)p[i] << 16;
        bool two = (i + 1 < len);
        if (two) v |= (uint32_t)p[i + 1] << 8;
        out.push_back(kB64[(v >> 18) & 63]);
        out.push_back(kB64[(v >> 12) & 63]);
        out.push_back(two ? kB64[(v >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

bool base64_decode(const std::string& in, std::vector<uint8_t>* out) {
    /* One table lookup per character: 0..63 a digit, -2 padding or whitespace (skipped anywhere),
     * -1 invalid. Media parts are megabytes, so a per-character branch ladder is too slow. */
    static const auto kVal = [] {
        std::array<int8_t, 256> t{};
        t.fill(-1);
        for (int i = 0; i < 26; ++i) { t[(size_t)('A' + i)] = (int8_t)i; t[(size_t)('a' + i)] = (int8_t)(26 + i); }
        for (int i = 0; i < 10; ++i) t[(size_t)('0' + i)] = (int8_t)(52 + i);
        t[(size_t)'+'] = 62;
        t[(size_t)'/'] = 63;
        for (char c : { '=', '\n', '\r', ' ', '\t' }) t[(size_t)(uint8_t)c] = -2;
        return t;
    }();
    out->resize(in.size() * 3 / 4 + 3);
    uint8_t* o = out->data();
    const uint8_t* p = (const uint8_t*)in.data();
    const uint8_t* const end = p + in.size();
    uint32_t acc = 0;
    int bits = 0;
    while (p < end) {
        /* Fast path: four valid digits at once. */
        if (bits == 0 && end - p >= 4) {
            const int a = kVal[p[0]], b = kVal[p[1]], c = kVal[p[2]], d = kVal[p[3]];
            if ((a | b | c | d) >= 0) {
                const uint32_t v = ((uint32_t)a << 18) | ((uint32_t)b << 12) | ((uint32_t)c << 6) | (uint32_t)d;
                o[0] = (uint8_t)(v >> 16); o[1] = (uint8_t)(v >> 8); o[2] = (uint8_t)v;
                o += 3;
                p += 4;
                continue;
            }
        }
        const int v = kVal[*p++];
        if (v == -2) continue;
        if (v < 0) { out->clear(); return false; }   /* nothing of a refused payload is left */
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            *o++ = (uint8_t)((acc >> bits) & 0xFF);
        }
    }
    out->resize((size_t)(o - out->data()));
    return true;
}

/* The parsed tree is bounded, not only the body. Each JSON value costs a 16-byte node plus
 * container slack however short its text, so `[1,1,1,...]` grows about eightfold and a body at
 * the transport limit could become gigabytes before any field is checked. A legitimate request
 * needs few values per context token (a token-id prompt is one per token; everything else is
 * small), so past a budget scaled from the context the parse keeps nothing and is refused. */
int64_t json_value_budget(int64_t max_ctx) {
    const int64_t kFloor = 1 << 20;        /* what a request needs with no context to scale by */
    const int64_t kPerToken = 16;          /* a batch of up to this many full-context prompts */
    return kFloor + kPerToken * std::max<int64_t>(max_ctx, 0);
}

/* Depth is bounded too: the parser does not recurse, but `dump` and destroying a field-by-field
 * copy recurse once per level, and a request's schema is serialised on the way to its grammar.
 * A few hundred KB of brackets, well inside the value budget, would overflow the worker's stack.
 * No legitimate request nests near it; the grammar compiler refuses schemas at half the bound. */
json parse_json_bounded(const std::string& body, int64_t max_ctx, bool* over, bool* deep) {
    const int64_t budget = json_value_budget(max_ctx);
    int64_t n = 0;
    *over = false;
    *deep = false;
    /* Each completed value counts once: a scalar at `value`, a container at its end. Once over
     * or too deep, every event returns false so the parser keeps nothing more. */
    json::parser_callback_t keep = [&](int depth, json::parse_event_t ev, json&) {
        if (!*deep && depth > kJsonMaxDepth) *deep = true;
        if (!*over && (ev == json::parse_event_t::value ||
                       ev == json::parse_event_t::object_end ||
                       ev == json::parse_event_t::array_end))
            *over = ++n > budget;
        return !*over && !*deep;
    };
    return json::parse(body, keep, /*allow_exceptions=*/false);
}

bool token_id(const json& v, int32_t* out) {
    if (v.is_number_unsigned()) {
        if (v.get<uint64_t>() > (uint64_t)INT32_MAX) return false;
    } else if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > INT32_MAX) {
        return false;
    }
    *out = (int32_t)v.get<int64_t>();
    return true;
}

/* ================================================================== errors */

std::string ApiError::dump() const {
    json e;
    e["message"] = message;
    e["type"] = type;
    if (param.empty()) e["param"] = nullptr; else e["param"] = param;
    if (code.empty()) e["code"] = nullptr; else e["code"] = code;
    if (!supported.empty()) e["supported"] = supported;
    if (n_prompt_tokens >= 0) e["n_prompt_tokens"] = n_prompt_tokens;
    if (n_ctx >= 0) e["n_ctx"] = n_ctx;
    json o;
    o["error"] = e;
    return o.dump();
}

ApiError err_bad(const std::string& param, const std::string& why) {
    ApiError e;
    e.status = 400;
    e.param = param;
    e.message = param.empty() ? why : (param + ": " + why);
    return e;
}

ApiError err_unsupported(const std::string& param, const std::string& why) {
    ApiError e = err_bad(param, why);
    e.code = "unsupported_parameter";
    return e;
}

ApiError err_not_implemented(const std::string& what) {
    ApiError e;
    e.status = 501;
    e.type = "not_implemented_error";
    e.message = what;
    return e;
}

ApiError err_overloaded(const std::string& why) {
    ApiError e;
    e.status = 429;
    e.type = "rate_limit_error";
    e.code = "server_overloaded";
    e.message = why;
    return e;
}

std::vector<std::string> dry_breakers(const std::vector<std::string>& deployment, bool set) {
    if (set || !deployment.empty()) return deployment;
    return { "\n", ":", "\"", "*" };
}

ApiError err_internal(const std::string& why) {
    ApiError e;
    e.status = 500;
    e.type = "internal_server_error";
    e.message = why;
    return e;
}

/* ================================================================== key vetting
 *
 * Four kinds of field; the lists below are the whole policy.
 *
 * `kAccepted*`: read and honoured, in OpenAI's, vLLM's and llama.cpp's spellings alike
 * (`repetition_penalty` and `repeat_penalty`, `max_tokens` and `n_predict`).
 *
 * `kInert`: cannot change the answer. Routing and bookkeeping labels (`user`, `metadata`), and
 * llama.cpp's speed and slot knobs (`cache_prompt`, `id_slot`, `speculative.*`, ...); this server
 * caches and speculates on its own terms. Accepted and ignored.
 *
 * `kGuarded`: known, not implemented, and would change the answer (mirostat, top-n-sigma,
 * dynamic temperature, logit bias). llama.cpp-built clients send them on every request at their
 * "off" value, so each is accepted at that value and refused by name at any other. A 200 thus
 * means every answer-changing field was honoured.
 *
 * Anything else is ignored, as llama.cpp and vLLM do: an unknown field is far more often a
 * client's own extension than a misunderstood request. Ignored names are logged at debug.
 */
namespace {

const char* kAcceptedCommon[] = {
    "model", "stream", "stream_options", "n", "seed", "stop", "max_tokens", "n_predict",
    "temperature", "top_p", "top_k", "min_p", "typical_p",
    "presence_penalty", "frequency_penalty", "repetition_penalty", "repeat_penalty",
    "repeat_last_n", "xtc_probability", "xtc_threshold",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "dry_sequence_breakers", "samplers",
    "grammar", "json_schema", "priority", "logprobs", "ignore_eos",
    "timings_per_token", "return_progress", "session_id", "n_probs",
    "control_vector", "experimental_speed_projection",
};

const char* kAcceptedChat[] = {
    "messages", "tools", "tool_choice", "response_format", "top_logprobs",
    "parallel_tool_calls", "add_generation_prompt", "max_completion_tokens",
    "chat_template_kwargs", "reasoning_effort", "enable_thinking", "preserve_thinking",
    "thinking_token_budget", "thinking_budget", "thinking_budget_tokens",
    "reasoning_budget_tokens", "reasoning_format", "continue_final_message",
    "functions", "function_call",
};

/* `preserved_tokens` is llama-server's: control tokens a raw completion's text keeps. On chat the
 * reply format decides them (guarded below); a raw completion has no format, and a client that
 * renders its own template and parses its own reply (e.g. a tool-call sidecar) needs it to see
 * a format built from control tokens. */
const char* kAcceptedCompletion[] = { "prompt", "echo", "preserved_tokens" };

const char* kAcceptedEmbedding[] = { "model", "input", "messages", "encoding_format",
                                      "dimensions", "image_tokens", "prompt_name" };

/* Listed explicitly so ignoring them is a recorded decision, and the debug line of ignored names
 * shows only undecided fields. */
const char* kInert[] = {
    /* OpenAI's and OpenRouter's bookkeeping */
    "user", "metadata", "store", "service_tier", "safety_identifier", "prompt_cache_key",
    "prediction",
    /* llama.cpp's own speed, slot and transport knobs */
    "cache_prompt", "id_slot", "n_cache_reuse", "backend_sampling", "reasoning_control",
    "t_max_prompt_ms", "n_keep", "n_discard", "sse_ping_interval", "verbose", "response_fields",
    /* the parameters of a guarded sampler, which mean nothing while it is off */
    "mirostat_tau", "mirostat_eta", "dynatemp_exponent",
};
/* llama.cpp's speculative-decoding knobs travel as dotted keys ("speculative.n_max"). */
const char* kInertPrefixes[] = { "speculative." };

/* A field this server does not implement and the value at which it does nothing. */
struct Guarded {
    const char* key;
    bool (*neutral)(const json& v);
    const char* off;     /* how a client turns it off, for the refusal */
    const char* why;     /* why it is not served */
};

bool is_zero(const json& v)      { return v.is_null() || (v.is_number() && v.get<double>() == 0.0); }
bool non_positive(const json& v) { return v.is_null() || (v.is_number() && v.get<double>() <= 0.0); }
bool falsy(const json& v)        { return v.is_null() || (v.is_boolean() && !v.get<bool>()); }
/* json::empty() is false for every string, the empty one included, so a string is tested as one. */
bool empty_value(const json& v) {
    if (v.is_string()) return v.get_ref<const std::string&>().empty();
    return v.is_null() || ((v.is_object() || v.is_array()) && v.empty());
}
bool at_most_one(const json& v)  { return v.is_null() || (v.is_number() && v.get<double>() <= 1.0); }
bool text_only(const json& v) {
    if (v.is_null()) return true;
    if (!v.is_array()) return false;
    for (const json& m : v) if (!m.is_string() || m.get<std::string>() != "text") return false;
    return true;
}

const Guarded kGuarded[] = {
    { "logit_bias", empty_value, "an empty object",
      "the device sampler has no per-token bias stage" },
    { "mirostat", is_zero, "0", "mirostat is not one of the device samplers" },
    { "top_n_sigma", non_positive, "-1", "top-n-sigma is not one of the device samplers" },
    { "dynatemp_range", non_positive, "0",
      "dynamic temperature is not one of the device samplers" },
    { "min_keep", is_zero, "0", "the cut stages keep no minimum candidate count" },
    { "post_sampling_probs", falsy, "false",
      "logprobs are the raw distribution's; the distribution after the sampler's stages is not "
      "returned" },
    { "best_of", at_most_one, "1", "sampling several candidates and returning the best is not "
                                   "implemented" },
    { "suffix", empty_value, "an empty string", "infilling through /v1/completions is not "
                                                "implemented" },
    { "lora", empty_value, "an empty list",
      "a per-request adapter list is not served; a runtime LoRA adapter is chosen by `model` "
      "(--lora)" },
    { "modalities", text_only, "[\"text\"]", "output modalities other than text are not "
                                             "implemented" },
    { "audio", empty_value, "null", "audio output is not implemented" },
    { "n_indent", is_zero, "0", "indentation-limited infill is not implemented" },
    { "t_max_predict_ms", non_positive, "0", "a generation time limit is not implemented" },
    { "return_tokens", falsy, "false", "token ids are not returned beside the text" },
    { "grammar_lazy", falsy, "false",
      "a caller's grammar applies from the first token; the lazy form is the chat format's own" },
    { "grammar_triggers", empty_value, "an empty list",
      "a caller's grammar applies from the first token; the lazy form is the chat format's own" },
    { "preserved_tokens", empty_value, "an empty list",
      "the chat format names the control tokens a reply keeps" },
};

template <size_t N>
bool in_list(const char* const (&list)[N], const std::string& k) {
    for (const char* s : list) if (k == s) return true;
    return false;
}

int vet_keys(const json& body, Endpoint ep, ApiError* err, bool native = false) {
    std::string ignored;
    for (auto it = body.begin(); it != body.end(); ++it) {
        const std::string& k = it.key();
        if (in_list(kInert, k)) continue;
        bool prefixed = false;
        for (const char* p : kInertPrefixes) prefixed = prefixed || k.rfind(p, 0) == 0;
        if (prefixed) continue;
        if (ep == Endpoint::Embedding) {
            if (in_list(kAcceptedEmbedding, k)) continue;
        } else {
            if (in_list(kAcceptedCommon, k)) continue;
            if (ep == Endpoint::Chat && in_list(kAcceptedChat, k)) continue;
            if (ep == Endpoint::Completion && in_list(kAcceptedCompletion, k)) continue;
        }
        /* llama-server's native completion returns the token ids, so the field is served there. */
        if (native && k == "return_tokens") continue;
        bool guarded = false;
        for (const Guarded& g : kGuarded) {
            if (k != g.key) continue;
            guarded = true;
            if (!g.neutral(it.value())) {
                *err = err_unsupported(k, std::string(g.why) + "; it is accepted only at " +
                                              g.off + ", where it does nothing");
                return RAD_E_UNSUPPORTED;
            }
        }
        if (!guarded) ignored += (ignored.empty() ? "" : ", ") + k;
    }
    if (!ignored.empty()) RAD_DEBUG("request fields ignored: %s", ignored.c_str());
    return RAD_OK;
}

/* Typed getters that refuse rather than coerce: a client sending "0.7" as a string has a bug,
 * and accepting it would hide the bug. */
bool get_num(const json& b, const char* k, double* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_number()) { *err = err_bad(k, "expected a number"); return false; }
    *out = it->get<double>();
    return true;
}
/* Every integer field read here is stored in 32 bits, so a value that does not fit is refused
 * rather than narrowed: 2^32 would become 0 and 2^31 negative, both "no limit" for `max_tokens`.
 * nlohmann keeps a literal above INT64_MAX as unsigned, hence the unsigned branch. */
bool get_int(const json& b, const char* k, int64_t* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_number_integer()) { *err = err_bad(k, "expected an integer"); return false; }
    const bool fits = it->is_number_unsigned()
                          ? it->get<uint64_t>() <= (uint64_t)INT32_MAX
                          : it->get<int64_t>() >= INT32_MIN && it->get<int64_t>() <= INT32_MAX;
    if (!fits) {
        *err = err_bad(k, "must fit in a 32-bit signed integer, got " + it->dump());
        return false;
    }
    *out = it->get<int64_t>();
    return true;
}
bool get_bool(const json& b, const char* k, bool* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_boolean()) { *err = err_bad(k, "expected a boolean"); return false; }
    *out = it->get<bool>();
    return true;
}
bool get_str(const json& b, const char* k, std::string* out, ApiError* err) {
    auto it = b.find(k);
    if (it == b.end() || it->is_null()) return true;
    if (!it->is_string()) { *err = err_bad(k, "expected a string"); return false; }
    *out = it->get<std::string>();
    return true;
}
/* A string member of a nested object (a message, a content block, a tool): absent or null leaves
 * `*out` unchanged, any other type is refused under `param`. Used instead of json::value(),
 * which throws on a type mismatch and would surface as a 500. */
bool member_str(const json& o, const char* k, const char* param, std::string* out, ApiError* err) {
    auto it = o.find(k);
    if (it == o.end() || it->is_null()) return true;
    if (!it->is_string()) { *err = err_bad(param, "expected a string"); return false; }
    *out = it->get<std::string>();
    return true;
}
/* The same read where later validation checks the type: a non-string member reads as `def`. */
std::string str_or(const json& o, const char* k, const std::string& def = std::string()) {
    auto it = o.find(k);
    return it != o.end() && it->is_string() ? it->get<std::string>() : def;
}

/* One field under several spellings (OpenAI's, vLLM's, llama.cpp's), read as a number (double)
 * or a 32-bit integer (int64_t). Any may be sent; two that disagree are refused, since no reading
 * order could be what the caller meant. */
template <class T>
bool get_any(const json& b, std::initializer_list<const char*> keys, T* out, bool* present,
             ApiError* err) {
    const char* named = nullptr;
    T v = *out;
    for (const char* k : keys) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        T x = 0;
        bool ok;
        if constexpr (std::is_same_v<T, double>) ok = get_num(b, k, &x, err);
        else ok = get_int(b, k, &x, err);
        if (!ok) return false;
        if (named && x != v) {
            *err = err_bad(k, std::string("disagrees with ") + named + ", which is the same field");
            return false;
        }
        v = x;
        named = k;
    }
    *out = v;
    if (present) *present = named != nullptr;
    return true;
}

}  /* namespace */

/* A conversation's label for /sessions: at most 128 bytes, cut at a UTF-8 character boundary
 * because the label is serialised into JSON. */
std::string session_label_of(const std::string& s) {
    constexpr size_t kMax = 128;
    if (s.size() <= kMax) return s;
    size_t cut = kMax;
    while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) --cut;
    return s.substr(0, cut);
}

KeyPolicy request_key_policy() {
    KeyPolicy p;
    p.common.assign(std::begin(kAcceptedCommon), std::end(kAcceptedCommon));
    p.chat.assign(std::begin(kAcceptedChat), std::end(kAcceptedChat));
    p.completion.assign(std::begin(kAcceptedCompletion), std::end(kAcceptedCompletion));
    p.embedding.assign(std::begin(kAcceptedEmbedding), std::end(kAcceptedEmbedding));
    p.inert.assign(std::begin(kInert), std::end(kInert));
    p.inert_prefixes.assign(std::begin(kInertPrefixes), std::end(kInertPrefixes));
    for (const Guarded& g : kGuarded) p.guarded.push_back(g.key);
    return p;
}

/* The body of every generation endpoint: one JSON object, parsed into a bounded tree. */
int parse_request_object(const std::string& body, const OaiLimits& lim, json* b, ApiError* err) {
    bool over = false, deep = false;
    *b = parse_json_bounded(body, lim.max_ctx, &over, &deep);
    if (deep) {
        *err = err_bad("", "the request body nests more than " + std::to_string(kJsonMaxDepth) +
                               " levels deep");
        return RAD_E_INVAL;
    }
    if (over) {
        char msg[160];
        snprintf(msg, sizeof msg,
                 "the request body holds more than %lld JSON values, which no request this "
                 "server can serve needs", (long long)json_value_budget(lim.max_ctx));
        *err = err_bad("", msg);
        return RAD_E_INVAL;
    }
    if (b->is_discarded() || !b->is_object()) {
        *err = err_bad("", "request body is not a JSON object");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* ================================================================== sampling params */

namespace {

/* One row per bounded sampler field. `lo_open` excludes the lower bound; `hi` of infinity is no
 * upper bound. `why` is the refusal text, or null to print the closed interval and the value. */
struct SamplerRange {
    const char* field;
    double      lo, hi;
    bool        lo_open;
    const char* why;
};

constexpr double kInf = std::numeric_limits<double>::infinity();

const SamplerRange kSamplerRanges[] = {
    { "temperature",        0.0,  2.0,  false, nullptr },
    { "top_p",              0.0,  1.0,  true,  "must be in (0, 1]" },
    { "top_k",              0.0,  kInf, false,
      "must be >= 0 (0 samples from the 1024 most likely tokens, the sampler's candidate width)" },
    { "min_p",              0.0,  1.0,  false, nullptr },
    { "typical_p",          0.0,  1.0,  true,  "must be in (0, 1]" },
    { "presence_penalty",   -2.0, 2.0,  false, nullptr },
    { "frequency_penalty",  -2.0, 2.0,  false, nullptr },
    { "repetition_penalty", 0.0,  kInf, true,  "must be > 0" },
    { "repeat_last_n",      -1.0, kInf, false, "must be >= -1" },
    { "xtc_probability",    0.0,  1.0,  false, nullptr },
    { "xtc_threshold",      0.0,  1.0,  false, nullptr },
    { "dry_base",           1.0,  kInf, false, "must be >= 1" },
    { "dry_allowed_length", 1.0,  kInf, false, "must be >= 1" },
    { "dry_penalty_last_n", -1.0, kInf, false, "must be >= -1" },
};

}  /* namespace */

/* Tested as "inside the range" so a NaN (which a flag can parse, though JSON cannot carry one)
 * fails every row. */
bool sampler_value_ok(const char* field, double v, std::string* why) {
    for (const SamplerRange& r : kSamplerRanges) {
        if (std::strcmp(r.field, field)) continue;
        const bool in = (r.lo_open ? v > r.lo : v >= r.lo) && v <= r.hi;
        if (in) return true;
        if (why) {
            if (r.why) *why = r.why;
            else {
                char b[160];
                snprintf(b, sizeof b, "must be in [%g, %g], got %g", r.lo, r.hi, v);
                *why = b;
            }
        }
        return false;
    }
    return true;
}

/* ================================================================== the sampler order
 *
 * llama.cpp's `samplers`: which stages run and in what order, as names ("top_k", "temperature",
 * with llama.cpp's aliases) or a string of letters ("kypmxt"). A stage the list leaves out does
 * not run, whatever its own field says (llama.cpp's meaning).
 *
 * The chain here is fixed (core/sample/sampler.h): penalties, DRY, top-k, top-p, min-p, typical,
 * XTC, with temperature either before the cuts (vLLM's order, the default) or after them
 * (llama.cpp's). A list is honoured when its enabled stages appear in that relative order with
 * temperature at either end of the cuts, and refused by name otherwise. Only enabled stages are
 * compared, since a stage at its identity value commutes with everything; this admits
 * llama.cpp's default list (typ_p before top_p) whenever typical is off. top-n-sigma is guarded
 * at its off value (kGuarded), so it may appear anywhere. */
static int apply_sampler_order(const json& b, SamplingParams* sp, ApiError* err) {
    auto it = b.find("samplers");
    if (it == b.end() || it->is_null()) return RAD_OK;

    enum { PEN, DRY, TOPK, TOPP, MINP, TYP, XTC, TEMP, SIGMA, N_STAGE };
    static const char* const kName[N_STAGE] = { "penalties", "dry", "top_k", "top_p", "min_p",
                                                "typ_p", "xtc", "temperature", "top_n_sigma" };
    auto by_name = [](std::string n) -> int {
        std::replace(n.begin(), n.end(), '-', '_');
        if (n == "penalties") return PEN;
        if (n == "dry") return DRY;
        if (n == "top_k") return TOPK;
        if (n == "top_p" || n == "nucleus") return TOPP;
        if (n == "min_p") return MINP;
        if (n == "typ_p" || n == "typ" || n == "typical_p" || n == "typical") return TYP;
        if (n == "xtc") return XTC;
        if (n == "temperature" || n == "temp") return TEMP;
        if (n == "top_n_sigma") return SIGMA;
        return -1;
    };
    auto by_char = [](char c) -> int {
        switch (c) {
            case 'e': return PEN;  case 'd': return DRY;  case 'k': return TOPK;
            case 'p': return TOPP; case 'm': return MINP; case 'y': return TYP;
            case 'x': return XTC;  case 't': return TEMP; case 's': return SIGMA;
            default:  return -1;
        }
    };
    auto unknown = [&](const std::string& what) {
        *err = err_unsupported("samplers", what + " names no sampler this server runs (infill and "
                                                  "adaptive_p are not served)");
        return RAD_E_UNSUPPORTED;
    };
    std::vector<int> list;
    if (it->is_string()) {
        for (char c : it->get<std::string>()) {
            const int st = by_char(c);
            if (st < 0) return unknown(std::string("'") + c + "'");
            list.push_back(st);
        }
    } else if (it->is_array()) {
        for (const json& n : *it) {
            if (!n.is_string()) { *err = err_bad("samplers", "expected sampler names"); return RAD_E_INVAL; }
            const int st = by_name(n.get<std::string>());
            if (st < 0) return unknown("\"" + n.get<std::string>() + "\"");
            list.push_back(st);
        }
    } else {
        *err = err_bad("samplers", "expected an array of sampler names or a string of letters");
        return RAD_E_INVAL;
    }

    auto listed = [&](int st) { return std::find(list.begin(), list.end(), st) != list.end(); };
    /* Stages the list leaves out are off. Temperature 0 stays greedy: it is the argmax either
     * way. */
    if (!listed(PEN))  { sp->rep_penalty = 1.0f; sp->freq_penalty = 0.0f; sp->pres_penalty = 0.0f; }
    if (!listed(DRY))  sp->dry_multiplier = 0.0f;
    if (!listed(TOPK)) sp->top_k = 0;
    if (!listed(TOPP)) sp->top_p = 1.0f;
    if (!listed(MINP)) sp->min_p = 0.0f;
    if (!listed(TYP))  sp->typical_p = 1.0f;
    if (!listed(XTC))  sp->xtc_probability = 0.0f;
    if (!listed(TEMP) && sp->temp > 0.0f) sp->temp = 1.0f;

    bool on[N_STAGE] = {};
    on[PEN]  = sp->rep_penalty != 1.0f || sp->freq_penalty != 0.0f || sp->pres_penalty != 0.0f;
    on[DRY]  = sp->dry_multiplier > 0.0f;
    on[TOPK] = sp->top_k > 0;
    on[TOPP] = sp->top_p < 1.0f;
    on[MINP] = sp->min_p > 0.0f;
    on[TYP]  = sp->typical_p < 1.0f;
    on[XTC]  = sp->xtc_probability > 0.0f;
    on[TEMP] = sp->temp > 0.0f && sp->temp != 1.0f;
    if (sp->temp <= 0.0f) return RAD_OK;     /* greedy: no order to honour */

    /* The enabled stages, first occurrence each, in the list's order. */
    std::vector<int> seq;
    for (int st : list)
        if (on[st] && std::find(seq.begin(), seq.end(), st) == seq.end()) seq.push_back(st);
    auto pos = [&](int st) {
        return (int)(std::find(seq.begin(), seq.end(), st) - seq.begin());
    };
    auto refuse = [&](int a, int bst) {
        *err = err_unsupported("samplers", std::string("\"") + kName[a] + "\" after \"" +
                               kName[bst] + "\" is an order this server's sampler chain cannot run "
                               "(penalties, dry, top_k, top_p, min_p, typ_p, xtc, with temperature "
                               "first or last)");
        return RAD_E_UNSUPPORTED;
    };
    /* Everything but temperature in the chain's own relative order. */
    static const int kChain[] = { PEN, DRY, TOPK, TOPP, MINP, TYP, XTC };
    int last = -1, last_st = -1;
    for (int st : kChain) {
        if (!on[st]) continue;
        const int p = pos(st);
        if (p < last) return refuse(last_st, st);
        last = p;
        last_st = st;
    }
    if (!on[TEMP]) return RAD_OK;
    /* Temperature: after the penalties, and before all enabled cuts or after all of them. top-k
     * commutes with it (a monotone rescale keeps the same k largest), so it is not a cut here. */
    const int pt = pos(TEMP);
    for (int st : { PEN, DRY })
        if (on[st] && pos(st) > pt) return refuse(st, TEMP);
    bool before = false, after = false;
    int first_cut = -1;
    for (int st : { TOPP, MINP, TYP, XTC }) {
        if (!on[st]) continue;
        if (first_cut < 0) first_cut = st;
        (pos(st) > pt ? before : after) = true;
    }
    if (before && after) return refuse(TEMP, first_cut);
    sp->temp_last = after;
    return RAD_OK;
}

static int parse_sampling(const json& b, const OaiLimits& lim, OaiRequest* r, ApiError* err) {
    double d;
    int64_t i64;
    bool bl;
    auto ok = [err](const char* k, double v) {
        std::string why;
        if (sampler_value_ok(k, v, &why)) return true;
        *err = err_bad(k, why);
        return false;
    };

    /* Start from the deployment's defaults; each field below overrides only when the request
     * names it. Done here, not per endpoint, so /v1/completions and /v1/chat/completions sample
     * identically. */
    r->sp = lim.default_sampling;

    d = r->sp.temp;
    if (!get_num(b, "temperature", &d, err)) return RAD_E_INVAL;
    if (!ok("temperature", d)) return RAD_E_INVAL;
    r->sp.temp = (float)d;

    d = r->sp.top_p;
    if (!get_num(b, "top_p", &d, err)) return RAD_E_INVAL;
    if (!ok("top_p", d)) return RAD_E_INVAL;
    r->sp.top_p = (float)d;

    i64 = r->sp.top_k;
    if (!get_int(b, "top_k", &i64, err)) return RAD_E_INVAL;
    if (!ok("top_k", (double)i64)) return RAD_E_INVAL;
    r->sp.top_k = (int)i64;

    d = r->sp.min_p;
    if (!get_num(b, "min_p", &d, err)) return RAD_E_INVAL;
    if (!ok("min_p", d)) return RAD_E_INVAL;
    r->sp.min_p = (float)d;

    d = r->sp.typical_p;
    if (!get_num(b, "typical_p", &d, err)) return RAD_E_INVAL;
    if (!ok("typical_p", d)) return RAD_E_INVAL;
    r->sp.typical_p = (float)d;

    d = r->sp.pres_penalty;
    if (!get_num(b, "presence_penalty", &d, err)) return RAD_E_INVAL;
    if (!ok("presence_penalty", d)) return RAD_E_INVAL;
    r->sp.pres_penalty = (float)d;

    d = r->sp.freq_penalty;
    if (!get_num(b, "frequency_penalty", &d, err)) return RAD_E_INVAL;
    if (!ok("frequency_penalty", d)) return RAD_E_INVAL;
    r->sp.freq_penalty = (float)d;

    d = r->sp.rep_penalty;
    if (!get_any(b, { "repetition_penalty", "repeat_penalty" }, &d, nullptr, err))
        return RAD_E_INVAL;
    if (!ok("repetition_penalty", d)) return RAD_E_INVAL;
    r->sp.rep_penalty = (float)d;

    i64 = r->sp.penalty_last_n;
    if (!get_int(b, "repeat_last_n", &i64, err)) return RAD_E_INVAL;
    if (!ok("repeat_last_n", (double)i64)) return RAD_E_INVAL;
    r->sp.penalty_last_n = (int)i64;

    d = r->sp.xtc_probability;
    if (!get_num(b, "xtc_probability", &d, err)) return RAD_E_INVAL;
    if (!ok("xtc_probability", d)) return RAD_E_INVAL;
    r->sp.xtc_probability = (float)d;

    d = r->sp.xtc_threshold;
    if (!get_num(b, "xtc_threshold", &d, err)) return RAD_E_INVAL;
    if (!ok("xtc_threshold", d)) return RAD_E_INVAL;
    r->sp.xtc_threshold = (float)d;

    d = r->sp.dry_multiplier;
    if (!get_num(b, "dry_multiplier", &d, err)) return RAD_E_INVAL;
    r->sp.dry_multiplier = (float)d;
    d = r->sp.dry_base;
    if (!get_num(b, "dry_base", &d, err)) return RAD_E_INVAL;
    if (!ok("dry_base", d)) return RAD_E_INVAL;
    r->sp.dry_base = (float)d;
    i64 = r->sp.dry_allowed_length;
    if (!get_int(b, "dry_allowed_length", &i64, err)) return RAD_E_INVAL;
    if (!ok("dry_allowed_length", (double)i64)) return RAD_E_INVAL;
    r->sp.dry_allowed_length = (int)i64;
    i64 = r->sp.dry_penalty_last_n;
    if (!get_int(b, "dry_penalty_last_n", &i64, err)) return RAD_E_INVAL;
    if (!ok("dry_penalty_last_n", (double)i64)) return RAD_E_INVAL;
    r->sp.dry_penalty_last_n = (int)i64;

    /* Absent means llama.cpp's four breakers (newline, colon, quote, asterisk), not none; a sent
     * list, even empty, replaces them. The sampler resolves breakers only when DRY runs. */
    auto dsb = b.find("dry_sequence_breakers");
    if (dsb == b.end() || dsb->is_null()) {
        r->sp.dry_seq_breakers = dry_breakers(r->sp.dry_seq_breakers, lim.default_dry_breakers_set);
    } else {
        r->sp.dry_seq_breakers.clear();
        if (!dsb->is_array()) {
            *err = err_bad("dry_sequence_breakers", "expected an array of strings");
            return RAD_E_INVAL;
        }
        for (const auto& s : *dsb) {
            if (!s.is_string()) {
                *err = err_bad("dry_sequence_breakers", "expected an array of strings");
                return RAD_E_INVAL;
            }
            r->sp.dry_seq_breakers.push_back(s.get<std::string>());
        }
    }

    /* An absent seed is random, per OpenAI's contract; seed 0 would make every unseeded request
     * draw the same sample. -1 is random too: llama-server's spelling of "no seed", which its
     * clients send back from /props. */
    auto sit = b.find("seed");
    if (sit != b.end() && !sit->is_null() && !sit->is_number_integer()) {
        *err = err_bad("seed", "expected an integer");
        return RAD_E_INVAL;
    }
    if (sit != b.end() && !sit->is_null() &&
        !(!sit->is_number_unsigned() && sit->get<int64_t>() == -1)) {
        r->sp.seed = (uint64_t)sit->get<int64_t>();
    } else {
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        r->sp.seed = rng();
    }

    /* n is served by fan-out: n independent requests over one prompt, nearly free after the first
     * through the prefix cache (spec 7.3). Their seeds are offset from the request's so the n
     * completions differ. */
    i64 = 1;
    if (!get_int(b, "n", &i64, err)) return RAD_E_INVAL;
    if (i64 < 1 || i64 > lim.max_n) {
        char msg[128];
        snprintf(msg, sizeof msg, "must be in [1, %d]", lim.max_n);
        *err = err_bad("n", msg);
        return RAD_E_INVAL;
    }
    r->n = (int)i64;

    i64 = 0;
    if (!get_int(b, "priority", &i64, err)) return RAD_E_INVAL;
    r->priority = (int)i64;

    /* max_tokens, also spelled max_completion_tokens (OpenAI) and n_predict (llama.cpp). A
     * negative value is llama.cpp's "no limit": stored as 0, which clamp_max_tokens resolves to
     * what the context leaves; unlike an absent field it ignores --default-max-tokens. */
    int64_t mt = 0;
    bool mt_named = false;
    if (!get_any(b, { "max_tokens", "max_completion_tokens", "n_predict" }, &mt, &mt_named,
                     err))
        return RAD_E_INVAL;
    if (mt_named && mt == 0) { *err = err_bad("max_tokens", "must be >= 1, or -1 for no limit"); return RAD_E_INVAL; }
    r->max_tokens = (int32_t)(mt > 0 ? mt : mt < 0 ? 0 : lim.default_max_tokens);

    /* ignore_eos: generate to max_tokens whatever the model emits. See Request::ignore_eos. */
    if (!get_bool(b, "ignore_eos", &r->ignore_eos, err)) return RAD_E_INVAL;

    /* The control vector (core/cvec.h): an opt-in per request, and the only place an API client
     * turns it on when the deployment's default is off. `control_vector` is the field's name;
     * `experimental_speed_projection` is Strata's spelling of the same switch, kept as an alias so
     * a client written against that reports what it means -- and read only when the request did not
     * send the field itself, so the field wins when a body carries both. A JSON null is "not sent",
     * as everywhere else in this parser. */
    for (const char* key : { "control_vector", "experimental_speed_projection" }) {
        const auto it = b.find(key);
        if (it == b.end() || it->is_null()) continue;
        bool on = false;
        if (!get_bool(b, key, &on, err)) return RAD_E_INVAL;
        r->control_vector = on ? 1 : 0;
        break;
    }

    /* stop: a string or an array of strings. Bounded in count and length because every stop
     * string is searched in each piece of generated text (by the decoder and the server's scan),
     * so per-token cost is count x longest. OpenAI allows four; the bounds are far above that
     * and configurable (--max-stop-strings, --max-stop-bytes). */
    const size_t max_stops = (size_t)lim.max_stops, max_stop_bytes = (size_t)lim.max_stop_bytes;
    auto st = b.find("stop");
    if (st != b.end() && !st->is_null()) {
        if (st->is_string()) r->stop.push_back(st->get<std::string>());
        else if (st->is_array()) {
            if (st->size() > max_stops) {
                *err = err_bad("stop", "at most " + std::to_string(max_stops) +
                                           " stop strings are accepted, got " +
                                           std::to_string(st->size()));
                return RAD_E_INVAL;
            }
            for (const auto& s : *st) {
                if (!s.is_string()) {
                    *err = err_bad("stop", "expected a string or an array of strings");
                    return RAD_E_INVAL;
                }
                r->stop.push_back(s.get<std::string>());
            }
        } else {
            *err = err_bad("stop", "expected a string or an array of strings");
            return RAD_E_INVAL;
        }
        for (const auto& s : r->stop) {
            if (s.size() > max_stop_bytes) {
                *err = err_bad("stop", "a stop string may be at most " +
                                           std::to_string(max_stop_bytes) + " bytes, got one of " +
                                           std::to_string(s.size()));
                return RAD_E_INVAL;
            }
        }
    }

    bl = false;
    if (!get_bool(b, "stream", &bl, err)) return RAD_E_INVAL;
    r->stream = bl;

    /* llama.cpp's streaming extras: timings on every chunk, and prompt progress chunks. */
    if (!get_bool(b, "timings_per_token", &r->timings_per_token, err)) return RAD_E_INVAL;
    if (!get_bool(b, "return_progress", &r->return_progress, err)) return RAD_E_INVAL;

    /* OpenRouter's `session_id`: the client's conversation name, used only as a label on
     * /sessions and the dashboard. A non-string value is ignored. */
    auto sid = b.find("session_id");
    if (sid != b.end() && sid->is_string()) r->session_label = session_label_of(sid->get<std::string>());

    RAD_TRY(apply_sampler_order(b, &r->sp, err));

    auto so = b.find("stream_options");
    if (so != b.end() && !so->is_null()) {
        if (!so->is_object()) { *err = err_bad("stream_options", "expected an object"); return RAD_E_INVAL; }
        bool iu = false;
        if (!get_bool(*so, "include_usage", &iu, err)) return RAD_E_INVAL;
        r->include_usage = iu;
    }

    return RAD_OK;
}

/* ================================================================== thinking
 *
 * Three spellings ask whether the model reasons before answering:
 * `chat_template_kwargs: {"enable_thinking": false}` (vLLM's clients), top-level
 * `enable_thinking` (a common shorthand), and OpenAI's `reasoning_effort`, whose "none" and
 * "minimal" turn thinking off.
 *
 * The kwarg alone is not enough: the vendored llama.cpp chat layer copies chat_template_kwargs
 * into the template context and then overwrites `enable_thinking` with its own boolean field
 * (chat-auto-parser-helpers.cpp). The kwarg is therefore lifted into that field.
 *
 * Deployment defaults go in first and the request overwrites them key by key, but a default in
 * one spelling must not survive a request using another. A request that says anything about
 * thinking drops the deployment's enable_thinking; one that names an effort also drops the
 * deployment's effort. Otherwise `--chat-template-kwargs '{"enable_thinking":false}'` would
 * disable thinking for a caller sending `reasoning_effort: "high"`.
 */
RequestThinking request_thinking(const json& b) {
    auto kw = b.find("chat_template_kwargs");
    const bool obj = kw != b.end() && kw->is_object();
    auto top = b.find("enable_thinking");
    /* An empty reasoning_effort string reads as unset, so it names no effort either. */
    auto re = b.find("reasoning_effort");
    RequestThinking t;
    t.effort = (obj && kw->contains("reasoning_effort")) ||
               (re != b.end() && re->is_string() && !re->get<std::string>().empty());
    t.thinking = t.effort || (top != b.end() && !top->is_null()) ||
                 (obj && kw->contains("enable_thinking"));
    return t;
}

void apply_template_defaults(const json& b, const OaiLimits& lim, ChatRenderOptions* opt) {
    const RequestThinking says = request_thinking(b);
    for (const auto& [k, v] : lim.default_template_kwargs) {
        if (k == "enable_thinking") {
            if (says.thinking) continue;
            opt->enable_thinking = v == "true";
        }
        if (k == "reasoning_effort" && says.effort) continue;
        opt->template_kwargs[k] = v;
    }
    opt->reasoning_format = lim.reasoning_format;
}

void map_reasoning_effort(const OaiLimits& lim, ChatRenderOptions* opt) {
    if (lim.reasoning_effort_map.empty()) return;
    auto it = opt->template_kwargs.find("reasoning_effort");
    if (it == opt->template_kwargs.end()) return;
    const json v = json::parse(it->second, nullptr, false);
    if (!v.is_string()) return;
    auto m = lim.reasoning_effort_map.find(v.get<std::string>());
    if (m == lim.reasoning_effort_map.end()) return;
    if (m->second == "none" || m->second == "minimal") {
        opt->enable_thinking = false;
        opt->template_kwargs.erase(it);
    } else {
        it->second = json(m->second).dump();
    }
}

ApiError template_refusal(const ChatRenderOptions& opt, const OaiLimits& lim, int rs,
                          const std::string& why) {
    /* With a known set and an effort outside it, the effort is the reason: no other probed
     * variable was in the render. The set is more actionable than the template's traceback. */
    auto it = opt.template_kwargs.find("reasoning_effort");
    if (it != opt.template_kwargs.end() && !lim.reasoning_efforts.empty()) {
        const json v = json::parse(it->second, nullptr, false);
        const std::vector<std::string>& set = lim.reasoning_efforts;
        if (v.is_string() && std::find(set.begin(), set.end(), v.get<std::string>()) == set.end()) {
            std::string list;
            for (size_t i = 0; i < set.size(); ++i)
                list += (i ? (i + 1 == set.size() ? " and " : ", ") : "") + ("'" + set[i] + "'");
            ApiError e = err_bad("reasoning_effort",
                                 "'" + v.get<std::string>() + "' is not a level this model's chat "
                                 "template takes. It takes " + list + "; 'none' and 'minimal' "
                                 "turn thinking off");
            e.code = "unsupported_value";
            e.supported = { "none", "minimal" };
            e.supported.insert(e.supported.end(), set.begin(), set.end());
            return e;
        }
    }
    return rs == RAD_E_UNSUPPORTED
               ? err_unsupported("messages", why.empty() ? "the chat template refused this request"
                                                         : why)
               : err_bad("messages", why.empty() ? "the chat template could not render this "
                                                   "request" : why);
}

static int parse_thinking(const json& b, const OaiLimits& lim, ChatRenderOptions* opt,
                          ApiError* err) {
    auto kw = b.find("chat_template_kwargs");
    const bool has_kw = kw != b.end() && !kw->is_null();
    if (has_kw && !kw->is_object()) {
        *err = err_bad("chat_template_kwargs", "expected an object");
        return RAD_E_INVAL;
    }
    const RequestThinking says = request_thinking(b);
    apply_template_defaults(b, lim, opt);

    if (has_kw) {
        for (auto it = kw->begin(); it != kw->end(); ++it) {
            /* dump(), not get<string>(): the template context takes JSON, so strings stay
             * quoted. See ChatRenderOptions. */
            opt->template_kwargs[it.key()] = it.value().dump();
            if (it.key() == "enable_thinking") {
                if (!it.value().is_boolean()) {
                    *err = err_bad("chat_template_kwargs.enable_thinking", "expected a boolean");
                    return RAD_E_INVAL;
                }
                opt->enable_thinking = it.value().get<bool>();
            }
        }
    }
    map_reasoning_effort(lim, opt);

    bool think = opt->enable_thinking;
    if (!get_bool(b, "enable_thinking", &think, err)) return RAD_E_INVAL;
    opt->enable_thinking = think;

    /* Top-level `preserve_thinking` is shorthand for the template variable, as llama.cpp maps it
     * and as some clients send it. Qwen3.6+ templates read it to keep earlier assistant turns'
     * reasoning in the prompt; other templates ignore it. The top-level spelling wins. */
    auto pt = b.find("preserve_thinking");
    if (pt != b.end() && !pt->is_null()) {
        if (!pt->is_boolean()) {
            *err = err_bad("preserve_thinking", "expected a boolean");
            return RAD_E_INVAL;
        }
        opt->template_kwargs["preserve_thinking"] = pt->dump();
    }

    /* Reasoning effort is the template's vocabulary, so there is no allow-list: OpenAI's ladder is
     * none/minimal/low/medium/high, while e.g. Qwen3.8's template takes xhigh/medium/low and
     * raises on anything else.
     *
     * The server owns one mapping: "none" and "minimal" mean enable_thinking = false. Every other
     * word is passed as template context; a template that refuses it says so through
     * ChatPrompt::error (core/text/chat.h). An explicit chat_template_kwargs entry wins, since
     * emplace does not overwrite. */
    std::string effort;
    if (!get_str(b, "reasoning_effort", &effort, err)) return RAD_E_INVAL;
    /* --reasoning-effort-map, before "none" and "minimal" are read, so a map can name either. */
    if (auto m = lim.reasoning_effort_map.find(effort);
        !effort.empty() && m != lim.reasoning_effort_map.end())
        effort = m->second;
    /* The deployment's default, where the request is silent. "none" and "minimal" act as an
     * enable_thinking default, dropped by any word about thinking; a level is an effort default,
     * dropped by an effort. */
    if (effort.empty()) {
        const std::string& def = lim.default_reasoning_effort;
        const bool off = def == "none" || def == "minimal";
        if (off ? !says.thinking : !says.effort) effort = def;
    }
    if (!effort.empty()) {
        if (effort == "none" || effort == "minimal") opt->enable_thinking = false;
        else opt->template_kwargs.emplace("reasoning_effort", json(effort).dump());
    }
    return RAD_OK;
}

/* ================================================================== grammar / response_format */

/* A non-null `schema_json` returns the schema instead of compiling it, for the chat endpoint:
 * the chat layer builds the grammar from the schema and its tool-call syntax together, and keys
 * `has_response_format` off `inputs.json_schema` (chat-auto-parser-generator.cpp). A precompiled
 * GBNF would be overwritten by its empty result. /v1/completions has no template, passes null,
 * and the schema is compiled here. */
static int parse_output_format(const json& b, const OaiDeps& d, OaiRequest* r, ApiError* err,
                               std::string* schema_json = nullptr) {
    std::string gram;
    if (!get_str(b, "grammar", &gram, err)) return RAD_E_INVAL;

    /* A schema goes back as JSON for the template path, or is compiled into the grammar. */
    auto use_schema = [&](const json& schema, const char* param) -> int {
        if (schema_json) { *schema_json = schema.dump(); return RAD_OK; }
        if (!d.grammar) {
            *err = err_not_implemented(std::string(param) + " requires the JSON-schema to GBNF "
                                       "compiler, which this build was wired without");
            return RAD_E_UNSUPPORTED;
        }
        std::string out, why;
        const int st = d.grammar->from_json_schema(schema, out, why);
        if (st < 0) {
            *err = err_bad(param, "the schema could not be compiled to a grammar: " + why);
            return st;
        }
        r->sp.grammar = std::move(out);
        return RAD_OK;
    };

    auto rf = b.find("response_format");
    const bool has_rf = rf != b.end() && !rf->is_null();
    if (!gram.empty() && has_rf) {
        *err = err_bad("response_format", "cannot be combined with an explicit `grammar`");
        return RAD_E_INVAL;
    }
    /* llama.cpp's `json_schema`: the schema itself at the top level, the shorthand for a
     * response_format of type json_schema. */
    auto js = b.find("json_schema");
    if (js != b.end() && !js->is_null()) {
        if (!gram.empty() || has_rf) {
            *err = err_bad("json_schema", "cannot be combined with `grammar` or `response_format`");
            return RAD_E_INVAL;
        }
        if (!js->is_object()) { *err = err_bad("json_schema", "expected a JSON Schema object"); return RAD_E_INVAL; }
        return use_schema(*js, "json_schema");
    }
    if (!gram.empty()) { r->sp.grammar = gram; return RAD_OK; }
    if (!has_rf) return RAD_OK;

    if (!rf->is_object()) { *err = err_bad("response_format", "expected an object"); return RAD_E_INVAL; }
    std::string type;
    if (!get_str(*rf, "type", &type, err)) return RAD_E_INVAL;
    if (type.empty() || type == "text") return RAD_OK;

    json schema;
    if (type == "json_object") {
        schema = json::object();
        schema["type"] = "object";
    } else if (type == "json_schema") {
        auto w = rf->find("json_schema");
        if (w == rf->end() || !w->is_object()) {
            *err = err_bad("response_format.json_schema", "expected an object");
            return RAD_E_INVAL;
        }
        auto s = w->find("schema");
        if (s == w->end() || !s->is_object()) {
            *err = err_bad("response_format.json_schema.schema", "expected a JSON Schema object");
            return RAD_E_INVAL;
        }
        schema = *s;
    } else {
        *err = err_bad("response_format.type",
                       "must be one of \"text\", \"json_object\", \"json_schema\", got \"" +
                       type + "\"");
        return RAD_E_INVAL;
    }
    return use_schema(schema, "response_format");
}

/* Appends a token-id array, refused under `param` unless every element is a valid id. */
static int token_ids(const json& arr, const char* param, std::vector<int32_t>* out,
                     ApiError* err) {
    for (const auto& v : arr) {
        int32_t id = 0;
        if (!v.is_number_integer()) {
            *err = err_bad(param, "token arrays must contain integers only");
            return RAD_E_INVAL;
        }
        if (!token_id(v, &id)) {
            *err = err_bad(param, "a token id must be in [0, 2147483647], got " + v.dump());
            return RAD_E_INVAL;
        }
        out->push_back(id);
    }
    return RAD_OK;
}

/* Compiles the request's grammar at admission, while failure is still a 400; on the step path
 * it would fail a request that is already streaming. `param` names the field it came from. */
static int compile_grammar(const OaiDeps& d, OaiRequest* r, const char* param, ApiError* err) {
    if (r->sp.grammar.empty() || !d.grammar) return RAD_OK;
    std::string why;
    if (d.grammar->compile(r->sp.grammar, &r->sp.grammar_program, why) >= 0) return RAD_OK;
    *err = err_bad(param, "the grammar this request implies cannot be run: " + why);
    return RAD_E_INVAL;
}

/* ================================================================== multimodal content parts */

/* Only `data:` URLs are accepted: fetching arbitrary http(s) URLs would make the server an SSRF
 * proxy into everything the deployment can reach. */
int decode_media_url(const std::string& url, const std::string& field,
                     std::vector<uint8_t>* out, ApiError* err) {
    if (url.rfind("data:", 0) != 0) {
        *err = err_bad(field, "only data: URLs are accepted; the server does not fetch remote URLs "
                              "on a request's behalf");
        return RAD_E_INVAL;
    }
    size_t comma = url.find(',');
    if (comma == std::string::npos) {
        *err = err_bad(field, "malformed data: URL");
        return RAD_E_INVAL;
    }
    const std::string meta = url.substr(5, comma - 5);
    if (meta.find("base64") == std::string::npos) {
        *err = err_bad(field, "data: URL must be base64 encoded");
        return RAD_E_INVAL;
    }
    if (!base64_decode(url.substr(comma + 1), out)) {
        *err = err_bad(field, "invalid base64");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* The deepest bracket nesting in a JSON text, counted without parsing (brackets outside string
 * literals, valid JSON or not). */
static int json_text_depth(const std::string& s) {
    int d = 0, most = 0;
    bool in_str = false, esc = false;
    for (char c : s) {
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '[' || c == '{') most = std::max(most, ++d);
        else if ((c == ']' || c == '}') && d > 0) --d;
    }
    return most;
}

/* True when `s` is valid JSON nesting deeper than the bound. Invalid JSON is rendered as a plain
 * string, so its brackets do not count. The cheap count runs first; json::accept (no tree, no
 * recursion) only when it is over. */
static bool json_text_too_deep(const std::string& s) {
    return json_text_depth(s) > kJsonMaxDepth && json::accept(s);
}

/* Strings the chat template parses as JSON are depth-bounded like the body (parse_json_bounded).
 * Tool-call arguments and tool-message content arrive as strings; the template layer parses and
 * re-serialises them recursively, so a string of brackets could overflow the worker's stack. */
static int check_template_json(const json& msg, const std::string& role, ApiError* err) {
    if (role == "assistant") {
        auto tcs = msg.find("tool_calls");
        if (tcs != msg.end() && tcs->is_array()) {
            for (const auto& tc : *tcs) {
                if (!tc.is_object()) continue;
                auto fn = tc.find("function");
                if (fn == tc.end() || !fn->is_object()) continue;
                auto args = fn->find("arguments");
                if (args != fn->end() && args->is_string() &&
                    json_text_too_deep(args->get_ref<const std::string&>())) {
                    *err = err_bad("messages[].tool_calls[].function.arguments",
                                   "nests more than " + std::to_string(kJsonMaxDepth) +
                                       " levels deep");
                    return RAD_E_INVAL;
                }
            }
        }
    }
    if (role == "tool") {
        auto c = msg.find("content");
        if (c != msg.end() && c->is_string() &&
            json_text_too_deep(c->get_ref<const std::string&>())) {
            *err = err_bad("messages[].content", "a tool result nests more than " +
                                                     std::to_string(kJsonMaxDepth) +
                                                     " levels deep");
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* Read and decode one media content part. Returns 1 with `*kind` and `*bytes` set for an image,
 * video or audio part, 0 for any other type (the caller decides), negative with `*err` for a
 * media part that cannot be served. `field` prefixes the error's field name
 * ("messages[].content[]" for chat, "input[]" for embeddings).
 *
 * Image and video: OpenAI's {"type":"image_url","image_url":{"url":...}} (or a bare string), the
 * Responses API's "input_image", and vLLM's {"type":"video_url","video_url":{"url":...}} and
 * "input_video". Audio: OpenAI's {"type":"input_audio","input_audio":{"data": base64, "format":
 * ...}} with any format the decoder admits, or vLLM's {"type":"audio_url","audio_url":{"url":
 * data URL}}. */
static int read_media_part(const json& part, const std::string& type, const std::string& field,
                           const OaiDeps& d, const OaiLimits& lim, MediaKind* kind,
                           std::vector<uint8_t>* bytes, ApiError* err) {
    /* A {"url": ...} object or a bare string under `key`, decoded from its data: URL. */
    auto from_url = [&](const char* key) -> int {
        const std::string fld = field + "." + key;
        auto it = part.find(key);
        std::string url;
        if (it != part.end() && it->is_object() && it->contains("url") && it->at("url").is_string())
            url = it->at("url").get<std::string>();
        else if (it != part.end() && it->is_string())
            url = it->get<std::string>();
        else {
            *err = err_bad(fld, "expected {\"url\": ...}");
            return RAD_E_INVAL;
        }
        RAD_TRY(decode_media_url(url, fld + ".url", bytes, err));
        return 1;
    };
    const bool is_image = type == "image_url" || type == "input_image";
    const bool is_video = type == "video_url" || type == "input_video";
    if (is_image || is_video) {
        *kind = is_image ? MediaKind::Image : MediaKind::Video;
        const char* what = is_image ? "image" : "video";
        const bool allowed = is_image ? lim.allow_image : lim.allow_video;
        if (!allowed || !d.mm || !d.mm->accepts(*kind)) {
            *err = err_not_implemented(
                std::string(what) + " input requires a vision encoder; this "
                "model was loaded without one");
            return RAD_E_UNSUPPORTED;
        }
        return from_url(is_image ? "image_url" : "video_url");
    }
    if (type == "input_audio" || type == "audio_url") {
        *kind = MediaKind::Audio;
        if (!lim.allow_audio || !d.mm || !d.mm->accepts(MediaKind::Audio)) {
            *err = err_not_implemented(
                "audio input requires an audio encoder; this model was loaded "
                "without one");
            return RAD_E_UNSUPPORTED;
        }
        if (type == "audio_url") return from_url("audio_url");
        auto ia = part.find("input_audio");
        if (ia == part.end() || !ia->is_object() || !ia->contains("data") ||
            !ia->at("data").is_string()) {
            *err = err_bad(field + ".input_audio", "expected {\"data\": ..., \"format\": ...}");
            return RAD_E_INVAL;
        }
        auto fit = ia->find("format");
        const std::string fmt =
            fit != ia->end() && fit->is_string() ? fit->get<std::string>() : std::string();
        /* The decoder detects the container from its bytes; the format is checked anyway so a
         * typo is reported as such rather than as an undecodable clip. */
        static const char* const kAudioFormats[] = { "wav", "mp3", "flac", "ogg", "opus",
                                                     "m4a", "aac", "webm", "aiff" };
        if (!in_list(kAudioFormats, fmt)) {
            std::string names;
            for (const char* f : kAudioFormats)
                names += (names.empty() ? "" : ", ") + std::string(f);
            *err = err_bad(field + ".input_audio.format", "must be one of " + names);
            return RAD_E_INVAL;
        }
        if (!base64_decode(ia->at("data").get<std::string>(), bytes)) {
            *err = err_bad(field + ".input_audio.data", "invalid base64");
            return RAD_E_INVAL;
        }
        return 1;
    }
    return 0;
}

/* Whether any string in `v` (a key or a value, the media parts' own markers aside) contains
 * RAD_MEDIA_MARKER. Iterative: the tree is up to kJsonMaxDepth deep. */
static bool spells_media_marker(const json& v) {
    std::vector<const json*> todo = { &v };
    while (!todo.empty()) {
        const json* j = todo.back();
        todo.pop_back();
        if (j->is_string()) {
            if (j->get_ref<const std::string&>().find(RAD_MEDIA_MARKER) != std::string::npos)
                return true;
        } else if (j->is_object()) {
            auto t = j->find("type");
            if (t != j->end() && *t == "media_marker") continue;
            for (auto it = j->begin(); it != j->end(); ++it) {
                if (it.key().find(RAD_MEDIA_MARKER) != std::string::npos) return true;
                todo.push_back(&it.value());
            }
        } else if (j->is_array()) {
            for (const json& e : *j) todo.push_back(&e);
        }
    }
    return false;
}

/* Rewrite content parts in place: text stays, media becomes the marker and its bytes go to
 * `media` as (kind, bytes) in the order the markers appear in the rendered prompt. */
static int lower_content_parts(json& messages, const OaiDeps& d, const OaiLimits& lim,
                               std::vector<std::pair<MediaKind, std::vector<uint8_t>>>* media,
                               ApiError* err) {
    if (!messages.is_array()) { *err = err_bad("messages", "expected an array"); return RAD_E_INVAL; }
    if (messages.empty()) { *err = err_bad("messages", "must not be empty"); return RAD_E_INVAL; }

    for (auto& msg : messages) {
        if (!msg.is_object()) { *err = err_bad("messages[]", "expected an object"); return RAD_E_INVAL; }
        std::string role;
        auto rit = msg.find("role");
        if (rit == msg.end() || !rit->is_string()) {
            *err = err_bad("messages[].role", "expected a string");
            return RAD_E_INVAL;
        }
        role = rit->get<std::string>();
        if (role != "system" && role != "developer" && role != "user" &&
            role != "assistant" && role != "tool") {
            *err = err_bad("messages[].role", "unknown role \"" + role + "\"");
            return RAD_E_INVAL;
        }
        RAD_TRY(check_template_json(msg, role, err));

        auto cit = msg.find("content");
        if (cit == msg.end() || cit->is_null()) {
            if (role == "assistant" && msg.contains("tool_calls")) continue;
            *err = err_bad("messages[].content",
                           "required for every message except an assistant message that carries "
                           "tool_calls");
            return RAD_E_INVAL;
        }
        if (cit->is_string()) continue;
        if (!cit->is_array()) {
            *err = err_bad("messages[].content", "expected a string or an array of content parts");
            return RAD_E_INVAL;
        }

        for (auto& part : *cit) {
            std::string type;
            auto tit = part.find("type");
            if (tit == part.end() || !tit->is_string()) {
                *err = err_bad("messages[].content[].type", "expected a string");
                return RAD_E_INVAL;
            }
            type = tit->get<std::string>();

            if (type == "text") {
                if (!part.contains("text") || !part.at("text").is_string()) {
                    *err = err_bad("messages[].content[].text", "expected a string");
                    return RAD_E_INVAL;
                }
                continue;
            }
            /* Media parts (read_media_part) become a media_marker part, not text: the chat layer
             * joins text parts with a newline but a marker with nothing, matching the model's own
             * template (<|vision_end|> runs straight into the following text). As text, each
             * picture would add a '\n' token the model never saw. */
            {
                MediaKind kind;
                std::vector<uint8_t> bytes;
                const int mr = read_media_part(part, type, "messages[].content[]", d, lim, &kind,
                                               &bytes, err);
                if (mr < 0) return mr;
                if (mr > 0) {
                    media->emplace_back(kind, std::move(bytes));
                    part.erase(type == "input_audio" ? "input_audio"
                               : type == "audio_url" ? "audio_url"
                               : (type == "image_url" || type == "input_image") ? "image_url"
                                                                                : "video_url");
                    part["type"] = "media_marker";
                    part["text"] = RAD_MEDIA_MARKER;
                    continue;
                }
            }
            *err = err_bad("messages[].content[].type",
                           "unsupported content part type \"" + type + "\"");
            return RAD_E_INVAL;
        }
    }
    /* Beside media, text that spells the marker would be taken for a media part's place in the
     * rendered prompt (tokenize_with_media); refused, as the embeddings endpoint refuses it. With
     * no media the marker is plain text. */
    if (!media->empty() && spells_media_marker(messages)) {
        *err = err_bad("messages", std::string("a message contains \"") + RAD_MEDIA_MARKER +
                                       "\", which this server reserves for the place of a media "
                                       "part, in a request that carries media");
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* Each marker becomes its media part's token run: the processor's placeholders (the model's pad
 * ids, which a hashed n-gram embedding reads like any other) with the model's surrounding markers
 * and timestamps. The encoder's rows replace the placeholder embeddings later (spec §11); here
 * only the ids and offsets, and so the rotary layout, are fixed.
 *
 * Media is processed here, on the request's worker, because decoding and resizing video is the
 * most expensive host work a request brings and must not stall the scheduler thread. */
int tokenize_with_media(const std::string& rendered, const OaiDeps& d,
                        std::vector<std::pair<MediaKind, std::vector<uint8_t>>>& media,
                        const MediaOpts& opts, PromptInput* out, ApiError* err) {
    out->text = rendered;
    auto pm = std::make_shared<mm::PromptMedia>();
    const std::string marker = RAD_MEDIA_MARKER;
    size_t pos = 0, mi = 0;
    for (;;) {
        size_t at = rendered.find(marker, pos);
        std::string seg = rendered.substr(pos, at == std::string::npos ? std::string::npos : at - pos);
        if (!seg.empty()) {
            /* add_special is false for every segment: the template already renders the special
             * tokens, and an extra BOS would shift every position after it. */
            auto ids = d.tok->encode(seg, false, /*parse_special=*/true);
            out->tokens.insert(out->tokens.end(), ids.begin(), ids.end());
        }
        if (at == std::string::npos) break;
        if (mi >= media.size()) {
            *err = err_bad("messages",
                           "the chat template rendered more media markers than the request had "
                           "media parts");
            return RAD_E_INVAL;
        }
        std::shared_ptr<mm::Item> item;
        std::string why;
        const int s = d.mm->prepare(media[mi].first, media[mi].second, opts, &item, &why);
        if (s < 0 || !item) {
            *err = err_bad("messages[].content[]", "the media could not be used: " +
                                                       (why.empty() ? std::string(rad_strerror(s))
                                                                    : why));
            return s < 0 ? s : RAD_E_INVAL;
        }
        media[mi].second.clear();
        media[mi].second.shrink_to_fit();
        mm::Placed pl;
        pl.item = item;
        pl.tok = (int32_t)out->tokens.size();
        out->tokens.insert(out->tokens.end(), item->expand.begin(), item->expand.end());
        pm->items.push_back(std::move(pl));
        ++mi;
        pos = at + marker.size();
    }
    if (mi != media.size()) {
        *err = err_bad("messages",
                       "the chat template dropped a media part: the request carried more media "
                       "than the rendered prompt has markers for");
        return RAD_E_INVAL;
    }
    pm->layout();
    out->media = std::move(pm);
    return RAD_OK;
}

/* ================================================================== the context bound
 *
 * The context clamps max_tokens instead of refusing the request. max_tokens is an upper bound and
 * the context is another that holds regardless, so refusing yields no tokens where clamping
 * yields the same ones. Clients commonly send a fixed large max_tokens and cannot compute the
 * remaining window without the tokeniser.
 *
 * A prompt that does not fit on its own is still an error, naming how much to cut.
 *
 * This is the one exception to "a 200 means every field was honoured": the vetted keys are
 * settings, this is a bound the context would hit anyway. The reply still says so through
 * finish_reason "length" and `usage`.
 */
/* The deployment's cap (OaiLimits::max_tokens_cap) clamps the same way. A max_tokens of 0
 * (the `--default-max-tokens auto` default) means what the context leaves, resolved here
 * against this prompt so the scheduler always sees a finite bound. */
static int clamp_max_tokens(size_t n_prompt, const OaiLimits& lim, const char* field,
                            OaiRequest* out, ApiError* err) {
    if (lim.max_tokens_cap > 0 &&
        (out->max_tokens <= 0 || (int64_t)out->max_tokens > lim.max_tokens_cap))
        out->max_tokens = (int32_t)lim.max_tokens_cap;

    if (lim.max_ctx <= 0) return RAD_OK;          /* no bound declared, nothing to clamp against */

    const int64_t room = lim.max_ctx - (int64_t)n_prompt;
    if (room <= 0) {
        /* Agents detect overflow by the message text to compact their history, so the sentence
         * carries both OpenAI's "maximum context length is N tokens" and llama-server's
         * "exceeds the available context size", beside OpenAI's code and llama-server's numbers. */
        char msg[384];
        snprintf(msg, sizeof msg,
                 "This model's maximum context length is %lld tokens: the request (%zu tokens) "
                 "exceeds the available context size (%lld tokens) and leaves nothing to generate "
                 "into; shorten the prompt by at least %lld tokens",
                 (long long)lim.max_ctx, n_prompt, (long long)lim.max_ctx, (long long)(1 - room));
        *err = err_bad(field, msg);
        err->message = msg;
        err->code = "context_length_exceeded";
        err->n_prompt_tokens = (int64_t)n_prompt;
        err->n_ctx = lim.max_ctx;
        return RAD_E_INVAL;
    }

    if (out->max_tokens <= 0) {
        out->max_tokens = (int32_t)std::min<int64_t>(room, INT32_MAX);
    } else if ((int64_t)out->max_tokens > room) {
        /* Debug, not Warn: clients sending a constant max_tokens trigger this on every normal
         * request. */
        RAD_DEBUG("max_tokens %d does not fit beside %zu prompt tokens in a %lld token context; "
                  "serving %lld", (int)out->max_tokens, n_prompt, (long long)lim.max_ctx,
                  (long long)room);
        out->max_tokens = (int32_t)room;
    }
    return RAD_OK;
}

/* ================================================================== chat */

/* The reasoning block by token id, and its budget.
 *
 * The render names the markers as text; the engine counts and closes the block by token, so the
 * markers are encoded here once per request. A marker that is not one token leaves the span
 * unknown: reasoning is still split out, but not counted and not closable by forcing a token.
 * The opener matters only when the reply must open its own block.
 *
 * Budget spellings: vLLM's `thinking_token_budget`, Qwen's `thinking_budget`, and llama.cpp's.
 * A budget on an unknown span is a 400 naming why, not unbounded reasoning. The deployment's
 * default applies only where the span is known. */
static int parse_thinking_budget(const json& b, const OaiDeps& d, const OaiLimits& lim,
                                 const ChatRender& render, SamplingParams* sp, ApiError* err) {
    ReasoningSpan span;
    size_t n_close = 0, n_open = 0;
    if (!render.think_close.empty() && d.tok) {
        const std::vector<int32_t> c = d.tok->encode(render.think_close, false, true);
        std::vector<int32_t> o;
        if (!render.think_open.empty()) o = d.tok->encode(render.think_open, false, true);
        n_close = c.size();
        n_open  = o.size();
        /* An opener that is one control token followed by text (Gemma 4's `<|channel>thought`)
         * is found by that first token; the text after it counts as reasoning. An opener with a
         * second control token is not, since its first token may open something else. */
        bool open_first = o.size() > 1 && d.tok->reply_bytes(o[0]).empty() &&
                          !d.tok->token_bytes(o[0]).empty();
        for (size_t i = 1; open_first && i < o.size(); ++i)
            if (d.tok->reply_bytes(o[i]).empty()) open_first = false;
        if (c.size() == 1 && (render.think_starts_open || o.size() == 1 || open_first)) {
            span.close       = c[0];
            span.open        = (o.size() == 1 || open_first) ? o[0] : -1;
            span.starts_open = render.think_starts_open;
        }
    }
    sp->reasoning = span;

    int64_t budget = -1;
    const char* named = nullptr;
    for (const char* k : { "thinking_token_budget", "thinking_budget", "thinking_budget_tokens",
                           "reasoning_budget_tokens" }) {
        auto it = b.find(k);
        if (it == b.end() || it->is_null()) continue;
        int64_t v = 0;
        if (!get_int(b, k, &v, err)) return RAD_E_INVAL;
        /* A negative budget means "none" (llama.cpp) and still overrides the deployment's
         * default. */
        if (v < 0) v = -1;
        if (named && v != budget) {
            *err = err_bad(k, std::string("disagrees with ") + named + ", which is the same field");
            return RAD_E_INVAL;
        }
        budget = v;
        named = k;
    }
    if (!named) {
        if (span.known()) sp->thinking_budget = (int32_t)lim.default_thinking_budget;
        return RAD_OK;
    }
    if (budget < 0) return RAD_OK;
    if (!span.known()) {
        std::string why;
        if (render.think_close.empty())
            why = "this model's chat template has no reasoning block to hold a budget on";
        else if (n_close != 1)
            why = "this model's reasoning block ends in `" + render.think_close + "`, which is " +
                  std::to_string(n_close) + " tokens, and a budget is held by forcing that end "
                  "as one token";
        else
            why = "this reply has to open its own reasoning block, and the opener `" +
                  render.think_open + "` is " + std::to_string(n_open) + " tokens and does not "
                  "start with a lone control token, so the block cannot be found by token";
        *err = err_unsupported(named, why);
        return RAD_E_UNSUPPORTED;
    }
    sp->thinking_budget = (int32_t)budget;
    return RAD_OK;
}

/* Logprobs, in each endpoint's spelling, into one count.
 *
 *   chat         `logprobs` true, with `top_logprobs` alternatives (0 when absent, OpenAI's default)
 *   completions  `logprobs` the number of alternatives, OpenAI's legacy integer
 *   any          llama-server's `n_probs`, which wins over both
 *
 * A count of 0 still returns each token's own logprob on chat (OpenAI's meaning) and nothing on
 * completions (llama-server treats `logprobs` 0 and `n_probs` 0 as off). Values are the raw
 * distribution's, as OpenAI and llama-server report them. */
static int parse_logprobs(const json& b, const OaiLimits& lim, Endpoint ep, OaiRequest* out,
                          ApiError* err) {
    bool want = false;
    int64_t top = 0;
    const char* field = "logprobs";
    if (ep == Endpoint::Chat) {
        if (!get_bool(b, "logprobs", &want, err)) return RAD_E_INVAL;
        if (!get_int(b, "top_logprobs", &top, err)) return RAD_E_INVAL;
        auto t = b.find("top_logprobs");
        const bool named = t != b.end() && !t->is_null();
        if (!want && named) {
            *err = err_bad("top_logprobs", "requires logprobs to be true");
            return RAD_E_INVAL;
        }
        if (named) field = "top_logprobs";
    } else {
        auto lit = b.find("logprobs");
        if (lit != b.end() && !lit->is_null()) {
            if (!lit->is_number_integer()) {
                *err = err_bad("logprobs", "expected an integer (the number of alternatives)");
                return RAD_E_INVAL;
            }
            top = lit->get<int64_t>();
            want = top > 0;
        }
    }
    auto np = b.find("n_probs");
    if (np != b.end() && !np->is_null()) {
        if (!np->is_number_integer()) {
            *err = err_bad("n_probs", "expected an integer (the number of alternatives)");
            return RAD_E_INVAL;
        }
        top = np->get<int64_t>();
        want = top > 0;
        field = "n_probs";
    }
    if (top < 0 || top > RAD_MAX_LOGPROBS) {
        *err = err_bad(field, "must be in [0, " + std::to_string(RAD_MAX_LOGPROBS) + "]");
        return RAD_E_INVAL;
    }
    if (want && !lim.supports_logprobs) {
        *err = err_unsupported(field, "the kernel libraries loaded have no sample_logprobs");
        return RAD_E_UNSUPPORTED;
    }
    out->want_logprobs = want;
    out->n_logprobs = (int)top;
    out->sp.n_probs = want ? (int)top : -1;
    return RAD_OK;
}

int parse_chat_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                       OaiRequest* out, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Chat, err));

    out->endpoint = Endpoint::Chat;
    out->id = random_id("chatcmpl-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!lim.version.empty()) out->system_fingerprint = "radiance-" + lim.version;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    RAD_TRY(parse_sampling(b, lim, out, err));
    std::string schema_json;
    RAD_TRY(parse_output_format(b, d, out, err, &schema_json));

    RAD_TRY(parse_logprobs(b, lim, Endpoint::Chat, out, err));

    if (!d.chat) {
        *err = err_not_implemented(
            "/v1/chat/completions requires a chat template; this model was loaded "
            "without one. Use /v1/completions");
        return RAD_E_UNSUPPORTED;
    }

    auto mit = b.find("messages");
    if (mit == b.end()) { *err = err_bad("messages", "is required"); return RAD_E_INVAL; }
    json messages = *mit;
    /* Deprecated functions-API turns in current form: an assistant `function_call` becomes one
     * tool call and a `function` message its result, paired by synthesised call ids. */
    if (messages.is_array()) {
        std::string call_id;
        int n_calls = 0;
        for (json& m : messages) {
            const std::string role = str_or(m, "role");
            auto fc = m.find("function_call");
            if (role == "assistant" && fc != m.end() && fc->is_object() && !m.contains("tool_calls")) {
                call_id = "call_" + std::to_string(n_calls++);
                m["tool_calls"] = json::array(
                    { json{ { "id", call_id }, { "type", "function" }, { "function", *fc } } });
                m.erase("function_call");
            } else if (role == "function") {
                m["role"] = "tool";
                m["tool_call_id"] = call_id;
            }
        }
    }
    if (messages.is_array()) {
        out->n_messages = (int)messages.size();
        for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
            if (!it->is_object() || !it->contains("role") || it->at("role") != "user") continue;
            const auto cit = it->find("content");
            if (cit == it->end()) break;
            /* Keep at most 160 bytes, copying no more than that from a possibly huge message,
             * and cut on a UTF-8 lead byte. */
            constexpr size_t kKeep = 160;
            std::string t;
            auto take = [&t](const std::string& s) {
                if (t.size() <= kKeep) t.append(s, 0, kKeep + 1 - t.size());
            };
            if (cit->is_string()) take(cit->get_ref<const std::string&>());
            else if (cit->is_array())
                for (const auto& part : *cit)
                    if (str_or(part, "type") == "text") take(str_or(part, "text"));
            for (char& c : t) if (c == '\n' || c == '\r' || c == '\t') c = ' ';
            if (t.size() > kKeep) {
                size_t cut = kKeep - 3;
                while (cut > 0 && ((unsigned char)t[cut] & 0xC0) == 0x80) --cut;
                t.resize(cut);
                t += "...";
            }
            out->last_user = std::move(t);
            break;
        }
    }

    /* OpenAI's deprecated functions API: `functions` is `tools` with each entry unwrapped, and
     * `function_call` is `tool_choice` ("none", "auto" or {"name": ...}). Rewritten into the
     * current fields and refused alongside them. The old API returns at most one call, so the
     * grammar admits one. */
    auto fns = b.find("functions");
    if (fns != b.end() && !fns->is_null()) {
        if (b.contains("tools") && !b.at("tools").is_null()) {
            *err = err_bad("functions", "send `tools` or the deprecated `functions`, not both");
            return RAD_E_INVAL;
        }
        if (!fns->is_array()) { *err = err_bad("functions", "expected an array"); return RAD_E_INVAL; }
        json t = json::array();
        for (const json& f : *fns) t.push_back(json{ { "type", "function" }, { "function", f } });
        b["tools"] = std::move(t);
        b["parallel_tool_calls"] = false;
        out->legacy_functions = true;
    }
    auto fcall = b.find("function_call");
    if (fcall != b.end() && !fcall->is_null()) {
        if (b.contains("tool_choice") && !b.at("tool_choice").is_null()) {
            *err = err_bad("function_call",
                           "send `tool_choice` or the deprecated `function_call`, not both");
            return RAD_E_INVAL;
        }
        if (fcall->is_string()) {
            b["tool_choice"] = *fcall;
        } else if (fcall->is_object() && fcall->contains("name") && fcall->at("name").is_string()) {
            b["tool_choice"] = json{ { "type", "function" },
                                     { "function", { { "name", fcall->at("name") } } } };
        } else {
            *err = err_bad("function_call", "expected \"none\", \"auto\" or {\"name\": ...}");
            return RAD_E_INVAL;
        }
    }

    /* tools and tool_choice. */
    json tools = json::array();
    auto tit = b.find("tools");
    if (tit != b.end() && !tit->is_null()) {
        if (!tit->is_array()) { *err = err_bad("tools", "expected an array"); return RAD_E_INVAL; }
        for (const auto& t : *tit) {
            if (!t.is_object() || !t.contains("type") || t.at("type") != "function" ||
                !t.contains("function") || !t.at("function").is_object() ||
                !t.at("function").contains("name") ||
                !t.at("function").at("name").is_string()) {
                *err = err_bad("tools[]",
                               "expected {\"type\":\"function\",\"function\":{\"name\":...}}");
                return RAD_E_INVAL;
            }
        }
        tools = *tit;
        out->n_tools = (int)tools.size();
    }

    /* A forced choice ("required", or an object naming one function) makes the call grammar the
     * whole reply instead of a lazy one: reasoning, then the calls (one when
     * `parallel_tool_calls` is false), then the end. A named function narrows the calls to it. */
    std::string tool_choice = "auto", named;
    auto tcit = b.find("tool_choice");
    if (tcit != b.end() && !tcit->is_null()) {
        if (tcit->is_string()) {
            tool_choice = tcit->get<std::string>();
        } else if (tcit->is_object()) {
            auto f = tcit->find("function");
            if (f == tcit->end() || !f->is_object() || !f->contains("name") ||
                !f->at("name").is_string()) {
                *err = err_bad("tool_choice",
                               "an object tool_choice must be {\"type\":\"function\","
                               "\"function\":{\"name\":...}}");
                return RAD_E_INVAL;
            }
            named = f->at("name").get<std::string>();
            tool_choice = "required";
        } else {
            *err = err_bad("tool_choice", "expected a string or an object");
            return RAD_E_INVAL;
        }
    }
    if (tool_choice == "none") {
        tools = json::array();
    } else if (tool_choice == "required") {
        if (tools.empty()) {
            *err = err_bad("tool_choice", "a tool call is required, but `tools` is empty");
            return RAD_E_INVAL;
        }
        if (!named.empty() &&
            std::none_of(tools.begin(), tools.end(), [&](const json& t) {
                return t.at("function").at("name") == named;
            })) {
            *err = err_bad("tool_choice", "names \"" + named + "\", which is not in `tools`");
            return RAD_E_INVAL;
        }
        if (!schema_json.empty() || !out->sp.grammar.empty()) {
            *err = err_bad("tool_choice", "a forced tool call is the whole reply, so it cannot be "
                                          "combined with `response_format` or `grammar`");
            return RAD_E_INVAL;
        }
    } else if (tool_choice != "auto") {
        *err = err_bad("tool_choice", "must be \"none\", \"auto\", \"required\", or an object");
        return RAD_E_INVAL;
    }

    /* Enforced by the grammar: the auto-parser emits a repeatable call section only when this is
     * true (chat-auto-parser-generator.cpp). */
    bool parallel = true;
    if (!get_bool(b, "parallel_tool_calls", &parallel, err)) return RAD_E_INVAL;

    if (!tools.empty() && !d.chat->supports_tools()) {
        *err = err_unsupported("tools",
                               "this model's chat template does not declare tool support");
        return RAD_E_UNSUPPORTED;
    }
    if (!tools.empty()) {
        const std::string why = d.chat->tool_refusal();
        if (!why.empty()) {
            *err = err_unsupported("tools", "this model's tool calls cannot be read back: " + why);
            return RAD_E_UNSUPPORTED;
        }
    }
    out->tools_present = !tools.empty();

    ChatRenderOptions opt;
    opt.quiet = lim.probe;
    opt.tool_choice = tool_choice;
    opt.tool_name = named;
    opt.parallel_tool_calls = parallel;
    const bool agp_named = b.contains("add_generation_prompt") && !b.at("add_generation_prompt").is_null();
    if (!get_bool(b, "add_generation_prompt", &opt.add_generation_prompt, err)) return RAD_E_INVAL;

    /* llama.cpp's `reasoning_format`: "none" leaves reasoning in `content` between the template's
     * markers; "auto" and "deepseek" move it to `reasoning_content` (the default behaviour).
     * "deepseek-legacy" (both at once) is not served. */
    std::string rfmt;
    if (!get_str(b, "reasoning_format", &rfmt, err)) return RAD_E_INVAL;
    if (rfmt == "deepseek-legacy") {
        *err = err_unsupported("reasoning_format", "\"deepseek-legacy\" keeps the reasoning in "
                               "`content` and `reasoning_content` both, which is not served; send "
                               "\"none\" or \"deepseek\"");
        return RAD_E_UNSUPPORTED;
    } else if (!rfmt.empty() && rfmt != "none" && rfmt != "auto" && rfmt != "deepseek") {
        *err = err_bad("reasoning_format", "must be \"none\", \"auto\" or \"deepseek\"");
        return RAD_E_INVAL;
    }

    /* Assistant prefill: a conversation ending in an assistant message is continued, not
     * answered (llama.cpp's default and Anthropic's meaning). vLLM's `continue_final_message`
     * requests it explicitly: true, or llama.cpp's "content" / "reasoning_content". Naming
     * `add_generation_prompt` opens a new turn instead, as in vLLM; naming it true together with
     * `continue_final_message` is refused, as in llama.cpp. */
    {
        std::string cont;
        auto cf = b.find("continue_final_message");
        if (cf != b.end() && !cf->is_null()) {
            if (cf->is_boolean()) cont = cf->get<bool>() ? "auto" : "";
            else if (cf->is_string() && (*cf == "content" || *cf == "reasoning_content"))
                cont = cf->get<std::string>();
            else {
                *err = err_bad("continue_final_message",
                               "expected a boolean, \"content\" or \"reasoning_content\"");
                return RAD_E_INVAL;
            }
            if (!cont.empty() && agp_named && opt.add_generation_prompt) {
                *err = err_bad("continue_final_message",
                               "cannot be combined with add_generation_prompt true");
                return RAD_E_INVAL;
            }
        }
        const json* last = messages.is_array() && !messages.empty() ? &messages.back() : nullptr;
        const bool ends_assistant = last && str_or(*last, "role") == "assistant";
        if (cont.empty() && cf == b.end() && ends_assistant && !agp_named) cont = "auto";
        if (!cont.empty()) {
            if (!ends_assistant) {
                *err = err_bad("continue_final_message",
                               "the last message is not an assistant message to continue");
                return RAD_E_INVAL;
            }
            if (last->contains("tool_calls") && !last->at("tool_calls").is_null() &&
                !last->at("tool_calls").empty()) {
                *err = err_bad("messages", "an assistant message that made tool calls cannot be "
                                           "continued");
                return RAD_E_INVAL;
            }
            if (messages.size() >= 2 && str_or(messages[messages.size() - 2], "role") == "assistant") {
                *err = err_bad("messages", "two assistant messages at the end cannot be "
                                           "continued");
                return RAD_E_INVAL;
            }
            opt.continue_final = cont;
            opt.add_generation_prompt = false;
            const json& c = last->contains("content") ? last->at("content") : json();
            if (c.is_string()) out->continued = c.get<std::string>();
            else if (c.is_array())
                for (const json& p : c)
                    if (str_or(p, "type") == "text") out->continued += str_or(p, "text");
        }
    }

    /* The caller's constraint goes into the render, which merges it with the format's tool-call
     * syntax into one grammar: a raw GBNF as `grammar`, a `response_format` as its schema. */
    opt.grammar     = out->sp.grammar;
    opt.json_schema = schema_json;

    RAD_TRY(parse_thinking(b, lim, &opt, err));
    /* The request's reasoning_format overrides the deployment's (--reasoning-format), applied by
     * parse_thinking's template defaults. */
    if (!rfmt.empty()) opt.reasoning_format = rfmt;

    std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
    RAD_TRY(lower_content_parts(messages, d, lim, &media, err));

    ChatRender render;
    std::string why;
    const int rs = d.chat->apply(messages, tools, opt, &render, &why);
    if (rs < 0) {
        *err = template_refusal(opt, lim, rs, why);
        return rs;
    }
    if (render.prompt.empty()) {
        *err = err_bad("messages", "the chat template rendered an empty prompt");
        return RAD_E_INVAL;
    }

    /* A requested constraint that came back without a grammar is refused, not served
     * unconstrained. */
    if (render.grammar.empty() && !(opt.grammar.empty() && opt.json_schema.empty())) {
        *err = err_unsupported(opt.json_schema.empty() ? "grammar" : "response_format",
                               "this model's chat template produced no grammar for it, so the "
                               "constraint could not be enforced");
        return RAD_E_UNSUPPORTED;
    }

    /* What the render decided. A tool-calling template may return a grammar when none went in;
     * it is lazy, so the reply is free prose until a call starts. Dropping `grammar_lazy` would
     * force every reply to be a tool call. */
    out->sp.grammar          = render.grammar;
    out->sp.grammar_lazy     = render.grammar_lazy;
    out->sp.grammar_triggers = render.grammar_triggers;
    out->sp.grammar_prefix   = render.generation_prompt;
    out->reply_parser        = render.parser;
    out->preserved_tokens    = render.preserved_tokens;
    out->parse_tools         = out->tools_present && render.parser != nullptr;
    RAD_TRY(parse_thinking_budget(b, d, lim, render, &out->sp, err));

    /* The format's end-of-turn markers join the caller's stop strings; both end generation with
     * `finish_reason: "stop"` (sink.cpp). n_client_stops records where the caller's end. */
    out->n_client_stops = out->stop.size();
    for (const auto& s : render.additional_stops)
        if (std::find(out->stop.begin(), out->stop.end(), s) == out->stop.end())
            out->stop.push_back(s);

    /* A reply starting inside the reasoning block applies its stop strings only after the
     * block's closer (Detokenizer::defer_stops_until). */
    if (render.think_starts_open && !render.think_close.empty())
        out->stops_after = render.think_close;

    /* The grammar may be the caller's or the template's own. */
    RAD_TRY(compile_grammar(d, out, out->tools_present ? "tools" : "grammar", err));

    const std::string& rendered = render.prompt;

    PromptInput p;
    if (media.empty()) {
        /* add_special is false: the template renders the model's special tokens, BOS included;
         * another would shift every position. */
        p.tokens = d.tok->encode(rendered, false, /*parse_special=*/true);
        p.text = rendered;
    } else {
        RAD_TRY(tokenize_with_media(rendered, d, media, MediaOpts{}, &p, err));
    }
    if (p.tokens.empty()) {
        *err = err_bad("messages", "the rendered prompt tokenised to nothing");
        return RAD_E_INVAL;
    }
    RAD_TRY(clamp_max_tokens(p.tokens.size(), lim, "messages", out, err));
    out->prompts.push_back(std::move(p));
    return RAD_OK;
}

/* ================================================================== completions */

int parse_completion_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err, bool native) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Completion, err, native));
    if (native && !get_bool(b, "return_tokens", &out->return_tokens, err)) return RAD_E_INVAL;

    out->endpoint = Endpoint::Completion;
    out->id = random_id("cmpl-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!lim.version.empty()) out->system_fingerprint = "radiance-" + lim.version;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    RAD_TRY(parse_sampling(b, lim, out, err));
    RAD_TRY(parse_output_format(b, d, out, err));

    RAD_TRY(compile_grammar(d, out, b.contains("grammar") ? "grammar" : "response_format", err));

    if (!get_bool(b, "echo", &out->echo, err)) return RAD_E_INVAL;

    /* llama-server's `preserved_tokens`: control tokens in these strings are rendered in the
     * text rather than dropped. Bounded like stop strings, since the decoder checks each per
     * token. */
    {
        auto pt = b.find("preserved_tokens");
        if (pt != b.end() && !pt->is_null()) {
            if (!pt->is_array() || pt->size() > (size_t)lim.max_stops) {
                *err = err_bad("preserved_tokens", "expected an array of at most " +
                                                       std::to_string(lim.max_stops) + " strings");
                return RAD_E_INVAL;
            }
            for (const auto& s : *pt) {
                if (!s.is_string() || s.get<std::string>().size() > (size_t)lim.max_stop_bytes) {
                    *err = err_bad("preserved_tokens", "expected strings of at most " +
                                                           std::to_string(lim.max_stop_bytes) +
                                                           " bytes");
                    return RAD_E_INVAL;
                }
                out->preserved_tokens.push_back(s.get<std::string>());
            }
        }
    }

    RAD_TRY(parse_logprobs(b, lim, Endpoint::Completion, out, err));
    /* `echo` with logprobs would need the prompt's logprobs, which never leave the card. */
    if (out->echo && out->want_logprobs) {
        *err = err_unsupported("echo", "logprobs for the prompt are not returned; send echo "
                                       "without logprobs, or logprobs without echo");
        return RAD_E_UNSUPPORTED;
    }

    auto pit = b.find("prompt");
    if (pit == b.end() || pit->is_null()) {
        *err = err_bad("prompt", "is required");
        return RAD_E_INVAL;
    }

    /* A raw prompt has no template or roles to protect, so control tokens in it are parsed, as
     * in vLLM and llama.cpp: a client rendering its own template means "<|im_start|>" as the
     * token. Token ids can name any control token anyway. */
    auto push_text = [&](const std::string& s) {
        PromptInput p;
        p.text = s;
        p.tokens = d.tok->encode(s, true, /*parse_special=*/true);
        out->prompts.push_back(std::move(p));
    };
    auto push_ids = [&](const json& arr) -> int {
        PromptInput p;
        RAD_TRY(token_ids(arr, "prompt", &p.tokens, err));
        p.text = d.tok->decode(p.tokens);
        out->prompts.push_back(std::move(p));
        return RAD_OK;
    };

    /* llama-server's mixed prompt: strings and token ids in one array form one prompt, strings
     * tokenised in place, BOS only on the first element. */
    bool mixed = false;
    if (native && pit->is_array()) {
        bool s = false, n = false;
        for (const auto& v : *pit) { s = s || v.is_string(); n = n || v.is_number_integer(); }
        mixed = s && n;
    }
    if (mixed) {
        PromptInput p;
        bool first = true;
        for (const auto& v : *pit) {
            if (v.is_string()) {
                const std::vector<int32_t> t = d.tok->encode(v.get<std::string>(), first, true);
                p.tokens.insert(p.tokens.end(), t.begin(), t.end());
            } else {
                int32_t id = 0;
                if (!v.is_number_integer() || !token_id(v, &id)) {
                    *err = err_bad("prompt", "a mixed prompt holds strings and token ids only");
                    return RAD_E_INVAL;
                }
                p.tokens.push_back(id);
            }
            first = false;
        }
        p.text = d.tok->decode(p.tokens);
        out->prompts.push_back(std::move(p));
    } else if (pit->is_string()) {
        push_text(pit->get<std::string>());
    } else if (pit->is_array()) {
        if (pit->empty()) { *err = err_bad("prompt", "must not be empty"); return RAD_E_INVAL; }
        if (pit->front().is_string()) {
            for (const auto& s : *pit) {
                if (!s.is_string()) {
                    *err = err_bad("prompt", "a prompt array must be all strings or all tokens");
                    return RAD_E_INVAL;
                }
                push_text(s.get<std::string>());
            }
        } else if (pit->front().is_number_integer()) {
            RAD_TRY(push_ids(*pit));
        } else if (pit->front().is_array()) {
            for (const auto& a : *pit) {
                if (!a.is_array()) {
                    *err = err_bad("prompt", "a batch of token arrays must be all arrays");
                    return RAD_E_INVAL;
                }
                RAD_TRY(push_ids(a));
            }
        } else {
            *err = err_bad("prompt", "expected a string, an array of strings, an array of token "
                                     "ids, or an array of token id arrays");
            return RAD_E_INVAL;
        }
    } else {
        *err = err_bad("prompt", "expected a string or an array");
        return RAD_E_INVAL;
    }

    /* The batch shares one `max_tokens`, so the longest prompt sets the clamp. */
    size_t longest = 0;
    for (const auto& p : out->prompts) {
        if (p.tokens.empty()) { *err = err_bad("prompt", "tokenised to nothing"); return RAD_E_INVAL; }
        if (p.tokens.size() > longest) longest = p.tokens.size();
    }
    RAD_TRY(clamp_max_tokens(longest, lim, "prompt", out, err));
    return RAD_OK;
}

/* ================================================================== embeddings */

/* One embedding input from text and its media, with the tokenizer's special tokens around the
 * whole. tokenize_with_media adds none, and with no template to place them the tokenizer's own
 * BOS/EOS rule applies; a BOS shifts the media runs by one. */
static int embedding_input(const std::string& text,
                           std::vector<std::pair<MediaKind, std::vector<uint8_t>>>& media,
                           const OaiDeps& d, int32_t image_tokens, OaiRequest* out,
                           ApiError* err) {
    PromptInput p;
    if (media.empty()) {
        p.text = text;
        p.tokens = d.tok->encode(text, true, /*parse_special=*/true);
    } else {
        MediaOpts mo;
        mo.image_tokens = image_tokens;
        RAD_TRY(tokenize_with_media(text, d, media, mo, &p, err));
        const ITokenizer::VocabInfo vi = d.tok->vocab_info();
        if (vi.add_bos && vi.bos >= 0) {
            p.tokens.insert(p.tokens.begin(), vi.bos);
            for (auto& pl : p.media->items) pl.tok += 1;
            p.media->layout();
        }
        if (vi.add_eos && vi.eos >= 0) p.tokens.push_back(vi.eos);
    }
    out->prompts.push_back(std::move(p));
    return RAD_OK;
}

int parse_embeddings_request(const std::string& body, const OaiDeps& d, const OaiLimits& lim,
                             OaiRequest* out, ApiError* err) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_keys(b, Endpoint::Embedding, err));

    out->endpoint = Endpoint::Embedding;
    out->id = random_id("embd-");
    out->created = unix_now();
    out->model = lim.model_id;
    if (!get_str(b, "model", &out->model, err)) return RAD_E_INVAL;
    if (out->model.empty()) out->model = lim.model_id;

    std::string enc = "float";
    if (!get_str(b, "encoding_format", &enc, err)) return RAD_E_INVAL;
    if (enc != "float" && enc != "base64") {
        *err = err_bad("encoding_format", "must be \"float\" or \"base64\"");
        return RAD_E_INVAL;
    }
    out->base64_embeddings = (enc == "base64");

    if (b.contains("dimensions") && !b.at("dimensions").is_null()) {
        const json& dv = b.at("dimensions");
        if (!dv.is_number_integer() || dv.get<int64_t>() <= 0) {
            *err = err_bad("dimensions", "must be a positive integer");
            return RAD_E_INVAL;
        }
        /* Bounded by the model's width in the handler, which knows it. */
        out->dimensions = dv.get<int64_t>();
    }
    /* A soft-token budget for this request's images (Gemma 4's processor); other processors
     * refuse it. */
    if (b.contains("image_tokens") && !b.at("image_tokens").is_null()) {
        const json& iv = b.at("image_tokens");
        const int64_t n = iv.is_number_integer() ? iv.get<int64_t>() : 0;
        if (n != 70 && n != 140 && n != 280 && n != 560 && n != 1120) {
            *err = err_bad("image_tokens", "must be one of 70, 140, 280, 560, 1120");
            return RAD_E_INVAL;
        }
        out->image_tokens = (int32_t)n;
    }

    /* sentence-transformers' `prompt_name`: the model's own text in front of each text or
     * content-part input. A model's default prompt applies when none is named, as in its
     * reference library. */
    std::string prefix;
    const bool prompt_named = b.contains("prompt_name") && !b.at("prompt_name").is_null();
    {
        std::string pname;
        if (prompt_named) {
            if (!b.at("prompt_name").is_string()) {
                *err = err_bad("prompt_name", "expected a string");
                return RAD_E_INVAL;
            }
            pname = b.at("prompt_name").get<std::string>();
        } else if (d.embed) {
            pname = d.embed->default_prompt();
        }
        if (!pname.empty() && (!d.embed || !d.embed->prompt(pname, &prefix))) {
            *err = err_bad("prompt_name", "this model states no prompt named \"" + pname + "\"");
            return RAD_E_INVAL;
        }
    }

    /* vLLM's chat embedding (`messages`): one input rendered by the model's chat template
     * (EmbeddingGemma 2's concatenates the system text, then each message's text and media
     * placeholders), with the tokenizer's special tokens around the whole. */
    if (b.contains("messages") && !b.at("messages").is_null()) {
        if (b.contains("input") && !b.at("input").is_null()) {
            *err = err_bad("messages", "a request takes `input` or `messages`, not both");
            return RAD_E_INVAL;
        }
        if (prompt_named) {
            *err = err_bad("prompt_name", "a `messages` input is rendered by the chat template "
                                          "and takes no prompt");
            return RAD_E_INVAL;
        }
        IChatTemplate* tpl = d.embed ? d.embed->message_template() : nullptr;
        if (!tpl) {
            *err = err_bad("messages", "this model has no chat template to render messages with");
            return RAD_E_INVAL;
        }
        json messages = b.at("messages");
        std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
        RAD_TRY(lower_content_parts(messages, d, lim, &media, err));
        ChatRenderOptions opt;
        opt.add_generation_prompt = false;
        ChatRender render;
        std::string why;
        if (tpl->apply(messages, json::array(), opt, &render, &why) < 0 || render.prompt.empty()) {
            *err = err_bad("messages", "the chat template did not render the messages" +
                                           (why.empty() ? std::string() : ": " + why));
            return RAD_E_INVAL;
        }
        return embedding_input(render.prompt, media, d, out->image_tokens, out, err);
    }

    auto it = b.find("input");
    if (it == b.end() || it->is_null()) { *err = err_bad("input", "is required"); return RAD_E_INVAL; }

    auto push_ids = [&](const json& arr) -> int {
        /* A named prompt cannot precede token ids and is refused; a default prompt applies to
         * text inputs only. */
        if (prompt_named && !prefix.empty()) {
            *err = err_bad("prompt_name", "a prompt is text and cannot go in front of token ids");
            return RAD_E_INVAL;
        }
        PromptInput p;
        RAD_TRY(token_ids(arr, "input", &p.tokens, err));
        out->prompts.push_back(std::move(p));
        return RAD_OK;
    };
    /* Control tokens are parsed as for a /v1/completions prompt: no template wraps the input. */
    auto push_text = [&](const std::string& s) {
        std::vector<std::pair<MediaKind, std::vector<uint8_t>>> none;
        return embedding_input(prefix + s, none, d, out->image_tokens, out, err);
    };

    /* One input of content parts (text, image, video, audio, in order) yields one vector. Parts
     * are joined with nothing between them, each media part becoming its token run, and the
     * tokenizer's special tokens go around the whole, matching what HuggingFace's processor feeds
     * the model (EmbeddingGemma 2's template concatenates text and placeholders). */
    auto push_parts = [&](const json& parts, ApiError* e) -> int {
        std::vector<std::pair<MediaKind, std::vector<uint8_t>>> media;
        std::string rendered = prefix;
        for (const auto& part : parts) {
            if (!part.is_object()) {
                *e = err_bad("input[]", "a content part must be an object");
                return RAD_E_INVAL;
            }
            auto tit = part.find("type");
            if (tit == part.end() || !tit->is_string()) {
                *e = err_bad("input[].type", "expected a string");
                return RAD_E_INVAL;
            }
            const std::string type = tit->get<std::string>();
            if (type == "text") {
                if (!part.contains("text") || !part.at("text").is_string()) {
                    *e = err_bad("input[].text", "expected a string");
                    return RAD_E_INVAL;
                }
                const std::string t = part.at("text").get<std::string>();
                /* Text containing the marker would be taken for a media part. */
                if (t.find(RAD_MEDIA_MARKER) != std::string::npos) {
                    *e = err_bad("input[].text", std::string("contains \"") + RAD_MEDIA_MARKER +
                                                     "\", which this server reserves for media");
                    return RAD_E_INVAL;
                }
                rendered += t;
                continue;
            }
            MediaKind kind;
            std::vector<uint8_t> bytes;
            const int mr = read_media_part(part, type, "input[]", d, lim, &kind, &bytes, e);
            if (mr < 0) return mr;
            if (mr == 0) {
                *e = err_bad("input[].type", "unsupported content part type \"" + type + "\"");
                return RAD_E_INVAL;
            }
            media.emplace_back(kind, std::move(bytes));
            rendered += RAD_MEDIA_MARKER;
        }
        return embedding_input(rendered, media, d, out->image_tokens, out, e);
    };
    auto is_parts = [](const json& a) { return a.is_array() && !a.empty() && a.front().is_object(); };

    if (it->is_string()) {
        RAD_TRY(push_text(it->get<std::string>()));
    } else if (it->is_array()) {
        if (it->empty()) { *err = err_bad("input", "must not be empty"); return RAD_E_INVAL; }
        if (it->front().is_number_integer()) {
            RAD_TRY(push_ids(*it));
        } else if (is_parts(*it)) {
            RAD_TRY(push_parts(*it, err));
        } else {
            /* A batch: each element is one input (a string, a token array or content parts). */
            for (const auto& el : *it) {
                if (el.is_string()) {
                    RAD_TRY(push_text(el.get<std::string>()));
                } else if (is_parts(el)) {
                    RAD_TRY(push_parts(el, err));
                } else if (el.is_array()) {
                    RAD_TRY(push_ids(el));
                } else {
                    *err = err_bad("input", "expected strings, token arrays or arrays of content "
                                            "parts");
                    return RAD_E_INVAL;
                }
            }
        }
    } else {
        *err = err_bad("input", "expected a string or an array");
        return RAD_E_INVAL;
    }

    for (size_t i = 0; i < out->prompts.size(); ++i) {
        const int64_t n = (int64_t)out->prompts[i].tokens.size();
        if (n == 0) { *err = err_bad("input", "tokenised to nothing"); return RAD_E_INVAL; }
        if (lim.max_ctx > 0 && n > lim.max_ctx) {
            *err = err_bad("input", "input " + std::to_string(i) + " is " + std::to_string(n) +
                                        " tokens, more than the model context of " +
                                        std::to_string(lim.max_ctx));
            err->code = "context_length_exceeded";
            err->n_prompt_tokens = n;
            err->n_ctx = lim.max_ctx;
            return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* ================================================================== responses */

static json tool_calls_json(const std::vector<ToolCall>& tcs) {
    json a = json::array();
    for (const auto& t : tcs) {
        json f;
        f["name"] = t.name;
        f["arguments"] = t.arguments;
        json c;
        c["id"] = t.id;
        c["type"] = "function";
        c["function"] = f;
        a.push_back(std::move(c));
    }
    return a;
}

static json usage_json(const UsageOut& u) {
    json d;
    d["cached_tokens"] = u.cached;
    json o;
    o["prompt_tokens"] = u.prompt;
    o["completion_tokens"] = u.completion;
    o["total_tokens"] = u.prompt + u.completion;
    o["prompt_tokens_details"] = d;
    /* The share of completion_tokens spent reasoning, so a reply cut at max_tokens inside its
     * reasoning can be told apart from one that answered nothing. */
    if (u.reasoning >= 0) {
        json c;
        c["reasoning_tokens"] = u.reasoning;
        o["completion_tokens_details"] = c;
    }
    return o;
}

json timings_json(const TimingsOut& t) {
    auto per = [](int64_t n, double ms) -> json {
        return n > 0 && ms > 0.0 ? json(ms / (double)n) : json(nullptr);
    };
    auto rate = [](int64_t n, double ms) -> json {
        return n > 0 && ms > 0.0 ? json((double)n * 1000.0 / ms) : json(nullptr);
    };
    const json pms = t.prompt_ms >= 0.0 ? json(t.prompt_ms) : json(nullptr);
    json o;
    o["cache_n"]                = t.cache_n;
    o["prompt_n"]               = t.prompt_n;
    o["prompt_ms"]              = pms;
    o["prompt_per_token_ms"]    = per(t.prompt_n, t.prompt_ms);
    o["prompt_per_second"]      = rate(t.prompt_n, t.prompt_ms);
    o["predicted_n"]            = t.predicted_n;
    o["predicted_ms"]           = t.predicted_ms;
    o["predicted_per_token_ms"] = per(t.decode_n, t.predicted_ms);
    o["predicted_per_second"]   = rate(t.decode_n, t.predicted_ms);
    o["ttft_ms"]                = t.ttft_ms >= 0.0 ? json(t.ttft_ms) : json(nullptr);
    o["total_ms"]               = t.total_ms >= 0.0 ? json(t.total_ms) : json(nullptr);
    if (t.draft_n > 0) {
        o["draft_n"]          = t.draft_n;
        o["draft_n_accepted"] = t.draft_n_accepted;
    }
    return o;
}

/* OpenAI's completion object of type `object` around `choices`, as JSON text. */
static std::string completion_object(const OaiRequest& r, const char* object, json choices,
                                     const UsageOut& u, const TimingsOut& t) {
    json o;
    o["id"] = r.id;
    o["object"] = object;
    o["created"] = r.created;
    o["model"] = r.model;
    if (!r.system_fingerprint.empty()) o["system_fingerprint"] = r.system_fingerprint;
    o["choices"] = std::move(choices);
    o["usage"] = usage_json(u);
    if (t.valid) o["timings"] = timings_json(t);
    return o.dump();
}

/* The usage chunk for `stream_options.include_usage`, on both endpoints. OpenAI's contract: an
 * empty choices array, after the last content chunk and before [DONE]. Only `object` differs. */
static std::string usage_chunk(const OaiRequest& r, const UsageOut& u, const TimingsOut& t,
                               const char* object) {
    return sse_event(completion_object(r, object, json::array(), u, t));
}

std::string text_usage_chunk(const OaiRequest& r, const UsageOut& u, const TimingsOut& t) {
    return usage_chunk(r, u, t, "text_completion");
}

std::string chat_usage_chunk(const OaiRequest& r, const UsageOut& u, const TimingsOut& t) {
    return usage_chunk(r, u, t, "chat.completion.chunk");
}

static json lp_bytes(const std::string& b) {
    json a = json::array();
    for (unsigned char c : b) a.push_back((int)c);
    return a;
}

static json lp_entry(const LogprobEntry& e) {
    json top = json::array();
    for (const LogprobAlt& a : e.top)
        top.push_back(json{ { "id", a.token }, { "token", a.text }, { "bytes", lp_bytes(a.bytes) },
                            { "logprob", a.logprob } });
    return json{ { "id", e.token }, { "token", e.text }, { "bytes", lp_bytes(e.bytes) },
                 { "logprob", e.logprob }, { "top_logprobs", std::move(top) } };
}

static json lp_content(const LogprobEntry* e, size_t n) {
    json a = json::array();
    for (size_t i = 0; i < n; ++i) a.push_back(lp_entry(e[i]));
    return a;
}

std::string logprobs_content(const LogprobEntry* e, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        if (i) s += ',';
        s += lp_entry(e[i]).dump();
    }
    return s;
}

/* OpenAI's legacy shape: parallel arrays, and the alternatives as one {text: logprob} object per
 * token. Alternatives with the same text (distinct ids that decode alike) collapse to the
 * likelier. */
static json lp_legacy(const LogprobEntry* e, size_t n, size_t offset) {
    json toks = json::array(), lps = json::array(), tops = json::array(), offs = json::array();
    for (size_t i = 0; i < n; ++i) {
        toks.push_back(e[i].text);
        lps.push_back(e[i].logprob);
        json t = json::object();
        for (const LogprobAlt& a : e[i].top)
            if (!t.contains(a.text)) t[a.text] = a.logprob;
        tops.push_back(std::move(t));
        /* The token's offset is where the previous one ended: its own `text` can be shorter than
         * what it added (half a character, held back). */
        offs.push_back(offset);
        offset = e[i].end;
    }
    return json{ { "tokens", std::move(toks) }, { "token_logprobs", std::move(lps) },
                 { "top_logprobs", std::move(tops) }, { "text_offset", std::move(offs) },
                 { "content", lp_content(e, n) } };
}

std::string logprobs_legacy(const LogprobEntry* e, size_t n, size_t offset) {
    return lp_legacy(e, n, offset).dump();
}

static json chat_logprobs_json(const std::vector<LogprobEntry>& lp) {
    json o;
    o["content"] = lp_content(lp.data(), lp.size());
    return o;
}

std::string chat_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t) {
    json arr = json::array();
    for (const auto& c : choices) {
        json m;
        m["role"] = "assistant";
        if (c.msg.content.empty() && !c.msg.tool_calls.empty()) m["content"] = nullptr;
        else m["content"] = c.msg.content;
        if (!c.msg.reasoning.empty()) m["reasoning_content"] = c.msg.reasoning;
        if (!c.msg.tool_calls.empty() && r.legacy_functions)
            m["function_call"] = json{ { "name", c.msg.tool_calls[0].name },
                                       { "arguments", c.msg.tool_calls[0].arguments } };
        else if (!c.msg.tool_calls.empty())
            m["tool_calls"] = tool_calls_json(c.msg.tool_calls);

        json ch;
        ch["index"] = c.index;
        ch["message"] = std::move(m);
        if (r.want_logprobs) ch["logprobs"] = chat_logprobs_json(c.logprobs);
        else ch["logprobs"] = nullptr;
        ch["finish_reason"] = r.legacy_functions && c.finish == Finish::ToolCalls
                                  ? "function_call" : finish_reason_oai(c.finish);
        arr.push_back(std::move(ch));
    }
    return completion_object(r, "chat.completion", std::move(arr), u, t);
}

std::string text_completion_body(const OaiRequest& r, const std::vector<ChoiceOut>& choices,
                                 const UsageOut& u, const TimingsOut& t) {
    json arr = json::array();
    for (const auto& c : choices) {
        json ch;
        ch["text"] = c.text;
        ch["index"] = c.index;
        if (r.want_logprobs) {
            ch["logprobs"] = lp_legacy(c.logprobs.data(), c.logprobs.size(), 0);
        } else {
            ch["logprobs"] = nullptr;
        }
        ch["finish_reason"] = finish_reason_oai(c.finish);
        arr.push_back(std::move(ch));
    }
    return completion_object(r, "text_completion", std::move(arr), u, t);
}

std::string embeddings_body(const OaiRequest& r, const std::vector<std::vector<float>>& vecs,
                            const UsageOut& u) {
    json data = json::array();
    for (size_t i = 0; i < vecs.size(); ++i) {
        json e;
        e["object"] = "embedding";
        if (r.base64_embeddings)
            e["embedding"] = base64_encode(vecs[i].data(), vecs[i].size() * sizeof(float));
        else
            e["embedding"] = vecs[i];
        e["index"] = (int64_t)i;
        data.push_back(std::move(e));
    }
    json usage;
    usage["prompt_tokens"] = u.prompt;
    usage["total_tokens"] = u.prompt;
    json o;
    o["object"] = "list";
    o["data"] = std::move(data);
    o["model"] = r.model;
    o["usage"] = std::move(usage);
    return o.dump();
}

static json model_card(const ModelCard& c) {
    json m;
    m["id"] = c.id;
    m["object"] = "model";
    m["created"] = c.created;
    m["owned_by"] = "radiance";
    if (c.max_model_len > 0) m["max_model_len"] = c.max_model_len;
    if (!c.root.empty()) m["root"] = c.root;
    if (!c.parent.empty()) m["parent"] = c.parent;
    if (!c.capabilities.empty()) {
        json caps = json::parse(c.capabilities, nullptr, false);
        if (!caps.is_discarded() && caps.is_object()) m["capabilities"] = std::move(caps);
    }
    return m;
}

std::string models_body(const ModelCard& c) {
    json o;
    o["object"] = "list";
    o["data"] = json::array({model_card(c)});
    return o.dump();
}

std::string models_body(const std::vector<ModelCard>& cards) {
    json o;
    o["object"] = "list";
    o["data"] = json::array();
    for (const ModelCard& c : cards) o["data"].push_back(model_card(c));
    return o.dump();
}

std::string model_body(const ModelCard& c) {
    return model_card(c).dump();
}

/* ================================================================== SSE */

std::string sse_event(const std::string& json_text) {
    return "data: " + json_text + "\n\n";
}

std::string sse_error(const ApiError& e) {
    /* After the headers are sent the status cannot change, so the error travels in the
     * stream. */
    return sse_event(e.dump());
}

/* ================================================================== the streaming path */

ReplyStream::ReplyStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : reader_(parser ? parser->open() : nullptr), id_(r.id), model_(r.model),
      fingerprint_(r.system_fingerprint), created_(r.created), index_(index) {}

/* Fold one increment into `last_`. Call ids are minted here, since the model writes none.
 * Returns false for an increment that adds nothing and is not sent. */
bool ReplyStream::absorb(const ReplyDelta& d) {
    switch (d.kind) {
    case ReplyDelta::Reasoning:
        if (d.text.empty()) return false;
        last_.reasoning += d.text;
        return true;
    case ReplyDelta::Content:
        if (d.text.empty()) return false;
        last_.content += d.text;
        return true;
    case ReplyDelta::CallBegin: {
        ToolCall tc;
        tc.id   = random_id("call_");
        tc.name = d.text;
        last_.tool_calls.push_back(std::move(tc));
        return true;
    }
    case ReplyDelta::CallArgs:
        if (d.text.empty() || d.call >= last_.tool_calls.size()) return false;
        last_.tool_calls[d.call].arguments += d.text;
        return true;
    case ReplyDelta::CallEnd:
        return true;
    }
    return false;
}

std::string ReplyStream::push(std::string_view appended, bool is_final) {
    if (finished_) return std::string();
    std::string out;
    if (!reader_) {
        /* No parser: the text is the content as it arrives. */
        if (!appended.empty()) {
            ev_.clear();
            ReplyDelta d;
            d.kind = ReplyDelta::Content;
            d.text = std::string(appended);
            ev_.push_back(std::move(d));
            out = emit(ev_);
        }
        finished_ = is_final;
        return out;
    }
    ev_.clear();
    if (!appended.empty()) reader_->feed(appended, &ev_);
    if (is_final) reader_->finish(&ev_);
    out = emit(ev_);
    if (is_final) {
        finished_ = true;
        /* The reader's message is authoritative; keep the call ids this stream minted. */
        ParsedMessage m = reader_->message();
        for (size_t i = 0; i < m.tool_calls.size() && i < last_.tool_calls.size(); ++i)
            m.tool_calls[i].id = last_.tool_calls[i].id;
        last_ = std::move(m);
    }
    return out;
}

std::string ReplyStream::error_chunk(const ApiError& e, const TimingsOut& t, const UsageOut& u) {
    return final_chunk(Finish::Error, t, u) + sse_error(e);
}

bool calls_complete(const ParsedMessage& m) {
    if (m.tool_calls.empty() || m.open_call) return false;
    for (const auto& tc : m.tool_calls)
        if (tc.name.empty() || !json_complete(tc.arguments)) return false;
    return true;
}

Finish ReplyStream::adjusted_finish(Finish f) const {
    if (f != Finish::Stop && f != Finish::StopString) return f;
    /* Claim tool_calls only if every call closed and its arguments parse (json_partial.h), so a
     * truncated call is not run. Its deltas have already been sent. */
    return calls_complete(last_) ? Finish::ToolCalls : f;
}

/* ------------------------------------------------------------------ OpenAI's chunks */

std::string ChatDeltaStream::chunk(json delta, const char* finish_reason, const TimingsOut* t) {
    json ch;
    ch["index"] = index_;
    ch["delta"] = std::move(delta);
    if (!lp_.empty()) {
        ch["logprobs"] = json{ { "content", json::parse("[" + lp_ + "]") } };
        lp_.clear();
    }
    if (finish_reason) ch["finish_reason"] = finish_reason;
    else ch["finish_reason"] = nullptr;
    json o;
    o["id"] = id_;
    o["object"] = "chat.completion.chunk";
    o["created"] = created_;
    o["model"] = model_;
    if (!fingerprint_.empty()) o["system_fingerprint"] = fingerprint_;
    o["choices"] = json::array({ch});
    if (t && t->valid) o["timings"] = timings_json(*t);
    return sse_event(o.dump());
}

void ChatDeltaStream::delta_open(std::string& out) {
    /* Built from nlohmann's serialisation of each field, on first use rather than in the
     * constructor, so a field nlohmann refuses throws where chunk() would. */
    if (head_.empty())
        head_ = "data: {\"id\":" + json(id_).dump() +
                ",\"object\":\"chat.completion.chunk\",\"created\":" + json(created_).dump() +
                ",\"model\":" + json(model_).dump() +
                (fingerprint_.empty() ? std::string()
                                      : ",\"system_fingerprint\":" + json(fingerprint_).dump()) +
                ",\"choices\":[{\"index\":" +
                json(index_).dump() + ",\"delta\":";
    out += head_;
}

void ChatDeltaStream::add_logprobs(const std::string& elems) {
    if (elems.empty()) return;
    if (!lp_.empty()) lp_ += ',';
    lp_ += elems;
}

void ChatDeltaStream::delta_close(std::string& out) {
    if (!lp_.empty()) {
        out += ",\"logprobs\":{\"content\":[";
        out += lp_;
        out += "]}";
        lp_.clear();
    }
    if (timings_.empty()) { out += ",\"finish_reason\":null}]}\n\n"; return; }
    out += ",\"finish_reason\":null}],";
    out += timings_;
    out += "}\n\n";
}

void ChatDeltaStream::set_timings(const TimingsOut& t) {
    timings_.clear();
    if (t.valid) timings_ = "\"timings\":" + timings_json(t).dump();
}

/* Malformed UTF-8 falls back to nlohmann, which throws the same exception chunk() would. */
void ReplyStream::quote(std::string& out, std::string_view s) {
    if (!json_quote_append(out, s)) out += json(std::string(s)).dump();
}

std::string ChatDeltaStream::first_chunk() { return std::string(); }

std::string ChatDeltaStream::started() {
    std::string out;
    open_role(out);
    return out;
}

/* The bytes chunk() writes for this delta, without taking the pending logprobs or timings,
 * which belong to the output chunk that follows. */
void ChatDeltaStream::open_role(std::string& out) {
    if (role_sent_) return;
    role_sent_ = true;
    delta_open(out);
    out += "{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n";
}

/* One SSE chunk per increment, in reader order. A call is announced with its id, name and empty
 * arguments before its arguments flow, since clients cannot resolve an unnamed entry. */
std::string ChatDeltaStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out;
    for (const ReplyDelta& d : ev) {
        if (!absorb(d) || d.kind == ReplyDelta::CallEnd) continue;
        /* The functions API has one call; any further call is not sent. */
        if (legacy_ && (d.kind == ReplyDelta::CallBegin || d.kind == ReplyDelta::CallArgs) &&
            (d.kind == ReplyDelta::CallBegin ? last_.tool_calls.size() - 1 : d.call) > 0)
            continue;
        open_role(out);
        delta_open(out);
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            out += "{\"reasoning_content\":";
            quote(out, d.text);
            out += '}';
            break;
        case ReplyDelta::Content:
            out += "{\"content\":";
            quote(out, d.text);
            out += '}';
            break;
        case ReplyDelta::CallBegin: {
            const ToolCall& tc = last_.tool_calls.back();
            if (legacy_) {
                out += "{\"function_call\":{\"name\":";
                quote(out, tc.name);
                out += ",\"arguments\":\"\"}}";
                break;
            }
            out += "{\"tool_calls\":[{\"index\":";
            out += std::to_string((int64_t)d.call);
            out += ",\"id\":";
            quote(out, tc.id);
            out += ",\"type\":\"function\",\"function\":{\"name\":";
            quote(out, tc.name);
            out += ",\"arguments\":\"\"}}]}";
            break;
        }
        case ReplyDelta::CallArgs:
            if (legacy_) {
                out += "{\"function_call\":{\"arguments\":";
                quote(out, d.text);
                out += "}}";
                break;
            }
            out += "{\"tool_calls\":[{\"index\":";
            out += std::to_string((int64_t)d.call);
            out += ",\"function\":{\"arguments\":";
            quote(out, d.text);
            out += "}}]}";
            break;
        case ReplyDelta::CallEnd:
            break;
        }
        delta_close(out);
    }
    return out;
}

std::string ChatDeltaStream::final_chunk(Finish f, const TimingsOut& t, const UsageOut&) {
    /* Resolve whatever the reader holds; a no-op if push already saw is_final. */
    std::string out = push(std::string_view(), true);
    open_role(out);
    const Finish a = adjusted_finish(f);
    out += chunk(json::object(), legacy_ && a == Finish::ToolCalls ? "function_call"
                                                                   : finish_reason_oai(a), &t);
    return out;
}

/* ================================================================== Anthropic's Messages API */

namespace {

/* Claude Code's system prompt opens with
 * `x-anthropic-billing-header: cc_version=...; cc_entrypoint=cli; cch=a5145;` where cch changes
 * per request, so the prompt's first tokens differ every turn and defeat the prefix cache. It is
 * a billing stamp meaningless to the model, held constant as in llama-server. */
void hold_billing_stamp(std::string& s) {
    static const char kHead[] = "x-anthropic-billing-header:";
    if (s.rfind(kHead, 0) != 0) return;
    const size_t at = s.find("cch=", sizeof kHead - 1);
    if (at == std::string::npos) return;
    const size_t end = s.find(';', at + 4);
    if (end == std::string::npos) return;
    for (size_t i = at + 4; i < end; ++i) s[i] = 'f';
}

/* An image block's source as a chat content part. A URL source stays a URL, which the chat path
 * refuses like every remote fetch (decode_media_url). `at` names the block in a refusal. */
int anthropic_image(const json& block, const std::string& at, json* out, ApiError* err) {
    auto sit = block.find("source");
    if (sit == block.end() || !sit->is_object()) {
        *err = err_bad(at + ".source", "expected an object");
        return RAD_E_INVAL;
    }
    const json& src = *sit;
    std::string type, url;
    if (!member_str(src, "type", (at + ".source.type").c_str(), &type, err)) return RAD_E_INVAL;
    if (type == "base64") {
        std::string media = "image/jpeg", data;
        if (!member_str(src, "media_type", (at + ".source.media_type").c_str(), &media, err) ||
            !member_str(src, "data", (at + ".source.data").c_str(), &data, err))
            return RAD_E_INVAL;
        url = "data:" + media + ";base64," + data;
    } else if (!member_str(src, "url", (at + ".source.url").c_str(), &url, err)) {
        return RAD_E_INVAL;
    }
    *out = json{ { "type", "image_url" }, { "image_url", { { "url", url } } } };
    return RAD_OK;
}

/* The top-level fields of a Messages request. Anthropic's API refuses a field it does not
 * define, and its clients send beta fields that change what the request means (a `thread`
 * continuation sends only the tail of the conversation), so a field outside these lists is
 * refused by name rather than ignored. The refusal says "Extra inputs are not permitted", as
 * Anthropic's does; Claude Code matches those words to retry without a beta field it can
 * drop. */
const char* kAnthropicFields[] = {
    "model", "messages", "system", "max_tokens", "temperature", "top_p", "top_k", "stream",
    "stop_sequences", "tools", "tool_choice", "thinking", "metadata", "output_config",
    "output_format", "context_management", "thread",
};

/* Accepted and not acted on: none of them can change the answer here. Capacity and placement
 * (`service_tier`, `inference_geo`, `speed`), prompt-cache control (`cache_control`,
 * `diagnostics`; the prefix cache is the server's own), and the inputs of Anthropic's
 * classifiers and refusal fallbacks (`safeguards`, `fallbacks`, `context_hint`), which do not
 * run here. */
const char* kAnthropicInert[] = {
    "service_tier", "inference_geo", "speed", "cache_control", "diagnostics",
    "safeguards", "fallbacks", "context_hint",
};

struct AnthropicRefusal {
    const char* key;
    const char* why;
};
/* Known, not served, and able to change the answer. */
const AnthropicRefusal kAnthropicRefused[] = {
    { "mcp_servers", "remote MCP servers are not connected from here; send their tools in "
                     "`tools` and their results back as tool_result blocks" },
    { "container", "code execution containers are not served" },
};

int vet_anthropic_fields(const json& b, ApiError* err) {
    for (auto it = b.begin(); it != b.end(); ++it) {
        const std::string& k = it.key();
        if (in_list(kAnthropicFields, k) || in_list(kAnthropicInert, k)) continue;
        for (const AnthropicRefusal& r : kAnthropicRefused) {
            if (k != r.key) continue;
            *err = err_unsupported(k, r.why);
            return RAD_E_UNSUPPORTED;
        }
        *err = err_unsupported(k, "Extra inputs are not permitted: this server does not serve "
                                  "the field, and one it does not know could change the answer");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* Anthropic's effort levels. */
bool anthropic_effort(const json& v) {
    if (!v.is_string()) return false;
    const std::string& s = v.get_ref<const std::string&>();
    return s == "low" || s == "medium" || s == "high" || s == "xhigh" || s == "max";
}

/* `output_config` (or the deprecated top-level `output_format`, the same schema): `effort`
 * becomes the chat request's `reasoning_effort`, which the chat template takes as its own level
 * names (a level it does not take is refused there, and --reasoning-effort-map renames them);
 * `format` becomes `response_format`. `task_budget` is a pacing hint the model is shown on
 * Anthropic's side; nothing here shows it, so it is accepted and not acted on. */
int anthropic_output_config(const json& b, json* o, ApiError* err) {
    auto put_format = [o, err](const json& f, const std::string& at) {
        std::string type;
        if (!f.is_object() || !member_str(f, "type", (at + ".type").c_str(), &type, err)) {
            if (!f.is_object()) *err = err_bad(at, "expected an object");
            return false;
        }
        auto s = f.find("schema");
        if (type != "json_schema" || s == f.end() || !s->is_object()) {
            *err = err_bad(at, "expected {\"type\":\"json_schema\",\"schema\":{...}}");
            return false;
        }
        (*o)["response_format"] = json{ { "type", "json_schema" },
                                        { "json_schema", { { "name", "output" },
                                                           { "schema", *s } } } };
        return true;
    };
    auto of = b.find("output_format");
    const bool legacy = of != b.end() && !of->is_null();
    auto oc = b.find("output_config");
    if (oc == b.end() || oc->is_null()) {
        if (legacy && !put_format(*of, "output_format")) return RAD_E_INVAL;
        return RAD_OK;
    }
    if (!oc->is_object()) { *err = err_bad("output_config", "expected an object"); return RAD_E_INVAL; }
    for (auto it = oc->begin(); it != oc->end(); ++it) {
        const std::string& k = it.key();
        const json& v = it.value();
        if (k == "effort") {
            if (!anthropic_effort(v)) {
                *err = err_bad("output_config.effort",
                               "must be \"low\", \"medium\", \"high\", \"xhigh\" or \"max\"");
                return RAD_E_INVAL;
            }
            (*o)["reasoning_effort"] = v;
        } else if (k == "format") {
            if (legacy) {
                *err = err_bad("output_format", "cannot be combined with output_config.format");
                return RAD_E_INVAL;
            }
            if (!put_format(v, "output_config.format")) return RAD_E_INVAL;
        } else if (k == "task_budget") {
            if (!v.is_object()) {
                *err = err_bad("output_config.task_budget", "expected an object");
                return RAD_E_INVAL;
            }
        } else {
            *err = err_unsupported("output_config." + k, "Extra inputs are not permitted");
            return RAD_E_UNSUPPORTED;
        }
    }
    if (legacy && !o->contains("response_format") && !put_format(*of, "output_format"))
        return RAD_E_INVAL;
    return RAD_OK;
}

/* `context_management` asks for edits of the conversation's history (clearing old tool uses or
 * thinking) before the model reads it. The conversation is served as sent, so the request is
 * taken and its answer reports that no edit was applied (`applied_edits: []`), which is
 * Anthropic's own report for a request no edit triggered on. */
int anthropic_context_management(const json& b, bool* sent, ApiError* err) {
    auto cm = b.find("context_management");
    if (cm == b.end() || cm->is_null()) return RAD_OK;
    if (!cm->is_object()) {
        *err = err_bad("context_management", "expected an object");
        return RAD_E_INVAL;
    }
    auto ed = cm->find("edits");
    if (ed != cm->end() && !ed->is_null()) {
        if (!ed->is_array()) {
            *err = err_bad("context_management.edits", "expected an array");
            return RAD_E_INVAL;
        }
        for (const json& e : *ed)
            if (!e.is_object() || !e.contains("type") || !e.at("type").is_string()) {
                *err = err_bad("context_management.edits[]", "expected an object with a `type`");
                return RAD_E_INVAL;
            }
    }
    if (sent) *sent = true;
    return RAD_OK;
}

/* `thread` asks Anthropic to keep the conversation server-side. `create` sends the whole
 * conversation and is served as any request; `continue` sends only what follows a stored
 * message, and nothing is stored here. */
int anthropic_thread(const json& b, ApiError* err) {
    auto th = b.find("thread");
    if (th == b.end() || th->is_null()) return RAD_OK;
    std::string type;
    if (!th->is_object() || !member_str(*th, "type", "thread.type", &type, err)) {
        if (!th->is_object()) *err = err_bad("thread", "expected an object");
        return RAD_E_INVAL;
    }
    if (type == "create") return RAD_OK;
    if (type == "continue") {
        *err = err_unsupported("thread", "conversations are not stored here, so a thread cannot "
                               "be continued; send the whole conversation in `messages`");
        return RAD_E_UNSUPPORTED;
    }
    *err = err_bad("thread.type", "must be \"create\" or \"continue\", got \"" + type + "\"");
    return RAD_E_INVAL;
}

/* A system message's text: a string or text blocks, joined with newlines. An effort-only
 * message (no text beside an `output_config`) yields "" and sets `*effort`. */
int system_message(const json& m, std::string* out, json* effort, ApiError* err) {
    for (auto it = m.begin(); it != m.end(); ++it)
        if (it.key() != "role" && it.key() != "content" && it.key() != "clear_at" &&
            it.key() != "output_config") {
            *err = err_unsupported("messages[]." + it.key(), "Extra inputs are not permitted "
                                   "on a system message");
            return RAD_E_UNSUPPORTED;
        }
    auto oc = m.find("output_config");
    if (oc != m.end() && !oc->is_null()) {
        if (!oc->is_object() || oc->size() != 1 || !oc->contains("effort") ||
            !anthropic_effort(oc->at("effort"))) {
            *err = err_bad("messages[].output_config",
                           "expected {\"effort\": \"low\"|\"medium\"|\"high\"|\"xhigh\"|\"max\"}");
            return RAD_E_INVAL;
        }
        *effort = oc->at("effort");
    }
    auto c = m.find("content");
    std::string t;
    if (c == m.end() || c->is_null()) {
    } else if (c->is_string()) {
        t = c->get<std::string>();
    } else if (c->is_array()) {
        for (const json& blk : *c) {
            const std::string type = str_or(blk, "type");
            if (type != "text") {
                *err = err_unsupported("messages[].content[].type", "a system message carries "
                                       "text blocks only, not \"" + type + "\"");
                return RAD_E_UNSUPPORTED;
            }
            std::string bt;
            if (!member_str(blk, "text", "messages[].content[].text", &bt, err)) return RAD_E_INVAL;
            if (!t.empty()) t += "\n";
            t += bt;
        }
    } else {
        *err = err_bad("messages[].content", "expected a string or text blocks");
        return RAD_E_INVAL;
    }
    if (t.empty() && effort->is_null()) {
        *err = err_bad("messages[].content", "must not be empty");
        return RAD_E_INVAL;
    }
    *out = std::move(t);
    return RAD_OK;
}

bool user_after(const json& msgs, size_t at) {
    for (size_t j = at + 1; j < msgs.size(); ++j)
        if (msgs[j].is_object() && str_or(msgs[j], "role") == "user") return true;
    return false;
}

/* A system message as the conversation reads it: its text, "" when there is none to render,
 * and the effort it sets. `clear_at: "next_user_message"` renders the text until a later user
 * message (a tool_result-only one included) exists, then never again. An effort applies from
 * the next user message on, so one after the last user message does not apply to this reply. */
int system_turn(const json& msgs, size_t at, std::string* text, json* effort, ApiError* err) {
    json e;
    RAD_TRY(system_message(msgs[at], text, &e, err));
    std::string when = "never";
    if (!member_str(msgs[at], "clear_at", "messages[].clear_at", &when, err)) return RAD_E_INVAL;
    if (when != "never" && when != "next_user_message") {
        *err = err_bad("messages[].clear_at",
                       "must be \"never\" or \"next_user_message\", got \"" + when + "\"");
        return RAD_E_INVAL;
    }
    const bool later = user_after(msgs, at);
    if (when == "next_user_message" && later) text->clear();
    if (!e.is_null() && later) *effort = std::move(e);
    return RAD_OK;
}

/* Marks, while the conversation is lowered, a user message that holds a system message's text. */
const char kLowered[] = "rad_lowered_system";

/* Each run of system texts lowered from later system messages joins a neighbouring user message
 * as text blocks, in order: the user message right after the run, else the one right before
 * it, else one user message of their own. A user message of its own beside another would be
 * two user turns in a row, which templates that enforce alternation refuse. A newline sits
 * between a folded text and the user's own, since some templates join text blocks with
 * nothing. */
json fold_lowered_system(json msgs) {
    json out = json::array();
    auto blocks = [](json& m) {
        if (m["content"].is_string())
            m["content"] = json::array({ json{ { "type", "text" }, { "text", m["content"] } } });
    };
    auto text = [](std::string t) { return json{ { "type", "text" }, { "text", std::move(t) } }; };
    for (size_t k = 0; k < msgs.size();) {
        if (!msgs[k].contains(kLowered)) {
            out.push_back(std::move(msgs[k++]));
            continue;
        }
        std::vector<std::string> run;
        for (; k < msgs.size() && msgs[k].contains(kLowered); ++k)
            run.push_back(msgs[k]["content"].get<std::string>());
        if (k < msgs.size() && str_or(msgs[k], "role") == "user") {
            json& next = msgs[k];
            blocks(next);
            json parts = json::array();
            for (std::string& t : run) parts.push_back(text(std::move(t) + "\n"));
            for (json& p : next["content"]) parts.push_back(std::move(p));
            next["content"] = std::move(parts);
        } else if (!out.empty() && str_or(out.back(), "role") == "user") {
            blocks(out.back());
            for (std::string& t : run) out.back()["content"].push_back(text("\n" + t));
        } else {
            std::string joined;
            for (const std::string& t : run) joined += (joined.empty() ? "" : "\n") + t;
            out.push_back(json{ { "role", "user" }, { "content", joined } });
        }
    }
    return out;
}

}  /* namespace */

int anthropic_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* label, ApiError* err, bool* context_management) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    RAD_TRY(vet_anthropic_fields(b, err));
    RAD_TRY(anthropic_thread(b, err));
    RAD_TRY(anthropic_context_management(b, context_management, err));
    json o = json::object();
    json msgs = json::array();

    /* The system prompt: the `system` field (a string or text blocks) and the system messages
     * at the head of the conversation, as one system message. The templates read a system
     * message only where it leads, so the text of a later one joins a user message at its
     * position (fold_lowered_system). */
    std::string sys_text;
    bool has_system = false;
    auto sys = b.find("system");
    if (sys != b.end() && !sys->is_null()) {
        has_system = true;
        if (sys->is_string()) {
            sys_text = sys->get<std::string>();
            hold_billing_stamp(sys_text);
        } else if (sys->is_array()) {
            for (const json& blk : *sys) {
                if (str_or(blk, "type") != "text") {
                    *err = err_bad("system", "expected a string or text blocks");
                    return RAD_E_INVAL;
                }
                std::string t;
                if (!member_str(blk, "text", "system[].text", &t, err)) return RAD_E_INVAL;
                hold_billing_stamp(t);
                if (!sys_text.empty()) sys_text += "\n";
                sys_text += t;
            }
        } else {
            *err = err_bad("system", "expected a string or text blocks");
            return RAD_E_INVAL;
        }
    }

    auto mit = b.find("messages");
    if (mit == b.end() || !mit->is_array()) {
        *err = err_bad("messages", "is required, as an array");
        return RAD_E_INVAL;
    }
    /* The effort a system message sets, which overrides the top-level one. */
    json effort;
    size_t i = 0;
    for (; i < mit->size(); ++i) {
        const json& m = (*mit)[i];
        if (!m.is_object() || str_or(m, "role") != "system") break;
        std::string t;
        RAD_TRY(system_turn(*mit, i, &t, &effort, err));
        if (t.empty()) continue;
        hold_billing_stamp(t);
        if (!sys_text.empty()) sys_text += "\n\n";
        sys_text += t;
        has_system = true;
    }
    if (has_system) msgs.push_back(json{ { "role", "system" }, { "content", sys_text } });

    for (; i < mit->size(); ++i) {
        const json& m = (*mit)[i];
        if (!m.is_object()) { *err = err_bad("messages[]", "expected an object"); return RAD_E_INVAL; }
        std::string role;
        if (!member_str(m, "role", "messages[].role", &role, err)) return RAD_E_INVAL;
        if (role == "system") {
            std::string t;
            RAD_TRY(system_turn(*mit, i, &t, &effort, err));
            if (!t.empty())
                msgs.push_back(json{ { "role", "user" }, { "content", t }, { kLowered, true } });
            continue;
        }
        if (role != "user" && role != "assistant") {
            *err = err_bad("messages[].role",
                           "must be \"user\", \"assistant\" or \"system\", got \"" + role + "\"");
            return RAD_E_INVAL;
        }
        const json content = m.contains("content") ? m.at("content") : json();
        if (content.is_string()) {
            msgs.push_back(json{ { "role", role }, { "content", content } });
            continue;
        }
        if (!content.is_array()) {
            *err = err_bad("messages[].content", "expected a string or content blocks");
            return RAD_E_INVAL;
        }
        /* Tool results go first. Anthropic puts them in the user turn after the calls, beside the
         * user's next words; the chat format wants each as a tool message right after the calling
         * assistant turn, with the user's words after them. */
        json parts = json::array(), calls = json::array(), results = json::array();
        std::string reasoning;
        for (const json& blk : content) {
            if (!blk.is_object()) {
                *err = err_bad("messages[].content[]", "expected an object");
                return RAD_E_INVAL;
            }
            std::string type;
            if (!member_str(blk, "type", "messages[].content[].type", &type, err)) return RAD_E_INVAL;
            if (type == "text") {
                std::string t;
                if (!member_str(blk, "text", "messages[].content[].text", &t, err)) return RAD_E_INVAL;
                parts.push_back(json{ { "type", "text" }, { "text", t } });
            } else if (type == "image") {
                json part;
                RAD_TRY(anthropic_image(blk, "messages[].content[]", &part, err));
                parts.push_back(std::move(part));
            } else if (type == "thinking") {
                std::string t;
                if (!member_str(blk, "thinking", "messages[].content[].thinking", &t, err))
                    return RAD_E_INVAL;
                reasoning += t;
            } else if (type == "redacted_thinking") {
                /* Encrypted for Anthropic's own models; nothing here can read it. */
            } else if (type == "tool_use") {
                std::string id, name;
                if (!member_str(blk, "id", "messages[].content[].id", &id, err) ||
                    !member_str(blk, "name", "messages[].content[].name", &name, err))
                    return RAD_E_INVAL;
                const json input = blk.value("input", json::object());
                if (!input.is_object()) {
                    *err = err_bad("messages[].content[].input", "expected an object");
                    return RAD_E_INVAL;
                }
                calls.push_back(json{ { "id", id },
                                      { "type", "function" },
                                      { "function", { { "name", name },
                                                      { "arguments", input.dump() } } } });
            } else if (type == "tool_result") {
                std::string use_id;
                if (!member_str(blk, "tool_use_id", "messages[].content[].tool_use_id", &use_id, err))
                    return RAD_E_INVAL;
                json rc = blk.contains("content") ? blk.at("content") : json("");
                json tc;
                if (rc.is_string()) {
                    tc = rc;
                } else if (rc.is_array()) {
                    json rp = json::array();
                    bool media = false;
                    std::string flat;
                    for (const json& c : rc) {
                        if (!c.is_object()) {
                            *err = err_bad("messages[].content[].content[]", "expected an object");
                            return RAD_E_INVAL;
                        }
                        std::string ct;
                        if (!member_str(c, "type", "messages[].content[].content[].type", &ct, err))
                            return RAD_E_INVAL;
                        if (ct == "text") {
                            std::string t;
                            if (!member_str(c, "text", "messages[].content[].content[].text", &t, err))
                                return RAD_E_INVAL;
                            flat += t;
                            rp.push_back(json{ { "type", "text" }, { "text", t } });
                        } else if (ct == "image") {
                            json part;
                            RAD_TRY(anthropic_image(c, "messages[].content[].content[]", &part, err));
                            media = true;
                            rp.push_back(std::move(part));
                        } else {
                            *err = err_unsupported("messages[].content[].content", "a tool result "
                                                   "block of type \"" + ct + "\" is not served");
                            return RAD_E_UNSUPPORTED;
                        }
                    }
                    tc = media ? rp : json(flat);
                } else {
                    tc = "";
                }
                results.push_back(json{ { "role", "tool" },
                                        { "tool_call_id", use_id },
                                        { "content", tc } });
            } else {
                /* Unservable blocks that would change the answer (documents, server tool
                 * results) are refused by name, not dropped. */
                *err = err_unsupported("messages[].content[].type",
                                       "a content block of type \"" + type + "\" is not served");
                return RAD_E_UNSUPPORTED;
            }
        }
        for (json& t : results) msgs.push_back(std::move(t));
        if (parts.empty() && calls.empty() && reasoning.empty()) continue;
        json nm = { { "role", role } };
        if (role == "assistant") {
            /* An assistant turn's text is one string, as every template renders it. */
            std::string text;
            for (const json& p : parts)
                if (p.at("type") == "text") text += p.at("text").get<std::string>();
            nm["content"] = text;
        } else {
            nm["content"] = parts.empty() ? json("") : parts;
        }
        if (!calls.empty()) nm["tool_calls"] = calls;
        if (!reasoning.empty()) nm["reasoning_content"] = reasoning;
        msgs.push_back(std::move(nm));
    }
    o["messages"] = fold_lowered_system(std::move(msgs));

    /* Client tools become functions. Server tools (Anthropic's web search, code execution,
     * named by `type`) run on Anthropic's side and are refused. */
    auto tit = b.find("tools");
    if (tit != b.end() && !tit->is_null() && !tit->is_array()) {
        *err = err_bad("tools", "expected an array");
        return RAD_E_INVAL;
    }
    if (tit != b.end() && tit->is_array() && !tit->empty()) {
        json tools = json::array();
        for (const json& t : *tit) {
            std::string type = "custom", name;
            if (t.is_object() && (!member_str(t, "type", "tools[].type", &type, err) ||
                                  !member_str(t, "name", "tools[].name", &name, err)))
                return RAD_E_INVAL;
            if (!t.is_object() || (type != "custom" && !t.contains("input_schema"))) {
                *err = err_unsupported("tools", "the server tool \"" + type + "\" runs on "
                                       "Anthropic's side and is not served; send client tools");
                return RAD_E_UNSUPPORTED;
            }
            json fn = { { "name", name },
                        { "parameters", t.contains("input_schema") ? t.at("input_schema") : json::object() } };
            if (t.contains("description")) fn["description"] = t.at("description");
            tools.push_back(json{ { "type", "function" }, { "function", fn } });
        }
        o["tools"] = std::move(tools);
    }
    auto tc = b.find("tool_choice");
    if (tc != b.end() && tc->is_string()) {
        /* The string form: "auto" and "none" as the chat path takes them, "any" as its
         * "required". Naming a tool takes the object form. */
        const std::string v = tc->get<std::string>();
        if (v == "auto" || v == "none") o["tool_choice"] = v;
        else if (v == "any") o["tool_choice"] = "required";
        else {
            *err = err_bad("tool_choice", "must be \"auto\", \"any\", \"none\" or an object, got "
                                           "\"" + v + "\"");
            return RAD_E_INVAL;
        }
    } else if (tc != b.end() && !tc->is_null() && !tc->is_object()) {
        *err = err_bad("tool_choice", "expected a string or an object");
        return RAD_E_INVAL;
    } else if (tc != b.end() && tc->is_object()) {
        std::string type = "auto", name;
        if (!member_str(*tc, "type", "tool_choice.type", &type, err) ||
            !member_str(*tc, "name", "tool_choice.name", &name, err))
            return RAD_E_INVAL;
        if (type == "auto" || type == "none") o["tool_choice"] = type;
        else if (type == "any") o["tool_choice"] = "required";
        else if (type == "tool")
            o["tool_choice"] = json{ { "type", "function" }, { "function", { { "name", name } } } };
        else {
            *err = err_bad("tool_choice.type", "must be auto, any, tool or none");
            return RAD_E_INVAL;
        }
        bool one = false;
        if (!get_bool(*tc, "disable_parallel_tool_use", &one, err)) return RAD_E_INVAL;
        if (one) o["parallel_tool_calls"] = false;
    }

    /* `model` included, since the reply names it back as Anthropic's does. */
    for (const char* k : { "model", "max_tokens", "temperature", "top_p", "top_k", "stream" })
        if (b.contains(k)) o[k] = b.at(k);
    if (b.contains("stop_sequences")) o["stop"] = b.at("stop_sequences");

    /* Extended thinking: on with a budget, off, or on with the template's own judgement;
     * `between_tools` is how the newest models turn it off. `display` and `block_binding` are
     * about how Anthropic returns and checks its signed thinking blocks; a local model's
     * reasoning comes back whole and unsigned, so they change nothing here. */
    auto th = b.find("thinking");
    if (th != b.end() && !th->is_null()) {
        if (!th->is_object()) { *err = err_bad("thinking", "expected an object"); return RAD_E_INVAL; }
        for (auto it = th->begin(); it != th->end(); ++it)
            if (it.key() != "type" && it.key() != "budget_tokens" && it.key() != "display" &&
                it.key() != "block_binding") {
                *err = err_unsupported("thinking." + it.key(), "Extra inputs are not permitted");
                return RAD_E_UNSUPPORTED;
            }
        std::string type;
        if (!member_str(*th, "type", "thinking.type", &type, err)) return RAD_E_INVAL;
        if (type == "disabled" || type == "between_tools") {
            o["enable_thinking"] = false;
        } else if (type == "enabled" || type == "adaptive") {
            o["enable_thinking"] = true;
            if (th->contains("budget_tokens")) o["thinking_token_budget"] = th->at("budget_tokens");
        } else {
            *err = err_bad("thinking.type", "must be \"enabled\", \"adaptive\", \"disabled\" or "
                                            "\"between_tools\", got \"" + type + "\"");
            return RAD_E_INVAL;
        }
    }

    RAD_TRY(anthropic_output_config(b, &o, err));
    if (!effort.is_null()) o["reasoning_effort"] = effort;

    /* metadata.user_id labels the conversation (Claude Code puts its session there). */
    auto md = b.find("metadata");
    if (md != b.end() && !md->is_null()) {
        if (!md->is_object()) { *err = err_bad("metadata", "expected an object"); return RAD_E_INVAL; }
        std::string uid;
        if (!member_str(*md, "user_id", "metadata.user_id", &uid, err)) return RAD_E_INVAL;
        if (label && !uid.empty()) *label = session_label_of(uid);
    }

    *chat = o.dump();
    return RAD_OK;
}

const char* anthropic_stop_reason(const OaiRequest& r, Finish f, const std::string& stop_word) {
    switch (f) {
        case Finish::ToolCalls: return "tool_use";
        case Finish::Length:    return "max_tokens";
        case Finish::StopString: {
            /* A client stop sequence is stop_sequence; the format's end-of-turn marker is
             * end_turn. */
            const auto end = r.stop.begin() + (long)std::min(r.n_client_stops, r.stop.size());
            if (!stop_word.empty() && std::find(r.stop.begin(), end, stop_word) != end)
                return "stop_sequence";
            return "end_turn";
        }
        default: return "end_turn";
    }
}

static json anthropic_usage(const UsageOut& u) {
    return json{ { "input_tokens", u.prompt - u.cached },
                 { "output_tokens", u.completion },
                 { "cache_read_input_tokens", u.cached },
                 { "cache_creation_input_tokens", 0 } };
}

std::string anthropic_message_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u) {
    json blocks = json::array();
    if (!c.msg.reasoning.empty())
        blocks.push_back(json{ { "type", "thinking" }, { "thinking", c.msg.reasoning },
                               { "signature", "" } });
    std::string text = c.msg.content;
    if (!r.continued.empty() && text.rfind(r.continued, 0) == 0) text.erase(0, r.continued.size());
    if (!text.empty()) blocks.push_back(json{ { "type", "text" }, { "text", text } });
    for (const ToolCall& tc : c.msg.tool_calls) {
        json in = json::parse(tc.arguments, nullptr, false);
        if (in.is_discarded() || !in.is_object()) in = json::object();
        blocks.push_back(json{ { "type", "tool_use" }, { "id", tc.id }, { "name", tc.name },
                               { "input", in } });
    }
    const char* sr = anthropic_stop_reason(r, c.finish, c.stop_word);
    json o;
    o["id"] = r.id;
    o["type"] = "message";
    o["role"] = "assistant";
    o["model"] = r.model;
    o["content"] = std::move(blocks);
    o["stop_reason"] = sr;
    o["stop_sequence"] = std::strcmp(sr, "stop_sequence") == 0 ? json(c.stop_word) : json(nullptr);
    o["usage"] = anthropic_usage(u);
    if (r.context_management)
        o["context_management"] = json{ { "applied_edits", json::array() } };
    return o.dump();
}

std::string anthropic_error_body(const ApiError& e) {
    json o;
    o["type"] = "error";
    o["error"] = json{ { "type", anthropic_error_type(e.status) }, { "message", e.message } };
    return o.dump();
}

AnthropicStream::AnthropicStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : ReplyStream(r, index, parser), r_(&r), skip_(r.continued) {
    for (const PromptInput& p : r.prompts) n_input_ += (int64_t)p.tokens.size();
}

void AnthropicStream::event(std::string& out, const char* type, const json& data) {
    out += "event: ";
    out += type;
    out += "\ndata: ";
    out += data.dump();
    out += "\n\n";
}

std::string AnthropicStream::first_chunk() { return std::string(); }

std::string AnthropicStream::started() {
    if (started_) return std::string();
    started_ = true;
    json m;
    m["id"] = id_;
    m["type"] = "message";
    m["role"] = "assistant";
    m["model"] = model_;
    m["content"] = json::array();
    m["stop_reason"] = nullptr;
    m["stop_sequence"] = nullptr;
    m["usage"] = json{ { "input_tokens", n_input_ }, { "output_tokens", 0 } };
    std::string out;
    event(out, "message_start", json{ { "type", "message_start" }, { "message", m } });
    return out;
}

void AnthropicStream::open(std::string& out, Block b, const ToolCall* tc) {
    close(out);
    json cb;
    if (b == Block::Thinking)
        cb = { { "type", "thinking" }, { "thinking", "" }, { "signature", "" } };
    else if (b == Block::Text) cb = { { "type", "text" }, { "text", "" } };
    else cb = { { "type", "tool_use" }, { "id", tc->id }, { "name", tc->name },
                { "input", json::object() } };
    event(out, "content_block_start",
          json{ { "type", "content_block_start" }, { "index", n_blocks_ }, { "content_block", cb } });
    ++n_blocks_;
    block_ = b;
}

void AnthropicStream::close(std::string& out) {
    if (block_ == Block::None) return;
    /* A thinking block ends with its signature, which clients send back next turn. A local model
     * has none; the empty one matches llama-server. */
    if (block_ == Block::Thinking)
        event(out, "content_block_delta",
              json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                    { "delta", { { "type", "signature_delta" }, { "signature", "" } } } });
    event(out, "content_block_stop", json{ { "type", "content_block_stop" }, { "index", n_blocks_ - 1 } });
    block_ = Block::None;
}

std::string AnthropicStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out = ev.empty() ? std::string() : started();
    for (const ReplyDelta& d : ev) {
        if (!absorb(d)) continue;
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            if (block_ != Block::Thinking) open(out, Block::Thinking, nullptr);
            event(out, "content_block_delta",
                  json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                        { "delta", { { "type", "thinking_delta" }, { "thinking", d.text } } } });
            break;
        case ReplyDelta::Content: {
            /* The prefill is the caller's own text; Anthropic's API answers what follows it, and
             * the buffered body drops it when the reply repeats the whole of it. A start that
             * matches the prefill so far is held until the reply completes the prefill (dropped)
             * or departs from it (sent whole). */
            std::string text = d.text;
            if (!skip_.empty()) {
                held_ += text;
                const size_t n = std::min(held_.size(), skip_.size());
                const bool same = held_.compare(0, n, skip_, 0, n) == 0;
                if (same && held_.size() < skip_.size()) break;
                text = same ? held_.substr(skip_.size()) : held_;
                held_.clear();
                skip_.clear();
            }
            text_delta(out, text);
            break;
        }
        case ReplyDelta::CallBegin:
            release_held(out);
            open(out, Block::Tool, &last_.tool_calls.back());
            break;
        case ReplyDelta::CallArgs:
            event(out, "content_block_delta",
                  json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                        { "delta", { { "type", "input_json_delta" }, { "partial_json", d.text } } } });
            break;
        case ReplyDelta::CallEnd:
            close(out);
            break;
        }
    }
    return out;
}

void AnthropicStream::text_delta(std::string& out, const std::string& text) {
    if (text.empty()) return;
    if (block_ != Block::Text) open(out, Block::Text, nullptr);
    event(out, "content_block_delta",
          json{ { "type", "content_block_delta" }, { "index", n_blocks_ - 1 },
                { "delta", { { "type", "text_delta" }, { "text", text } } } });
}

/* Content held against the prefill is the reply's own text once nothing more can complete the
 * prefill: the reply ended, or moved on to a call. */
void AnthropicStream::release_held(std::string& out) {
    if (held_.empty()) return;
    std::string text;
    text.swap(held_);
    skip_.clear();
    text_delta(out, text);
}

std::string AnthropicStream::final_chunk(Finish f, const TimingsOut&, const UsageOut& u) {
    std::string out = started();
    out += push(std::string_view(), true);
    release_held(out);
    close(out);
    const char* sr = anthropic_stop_reason(*r_, adjusted_finish(f), stop_word_);
    json delta = { { "stop_reason", sr },
                   { "stop_sequence", std::strcmp(sr, "stop_sequence") == 0 ? json(stop_word_)
                                                                             : json(nullptr) } };
    json md = { { "type", "message_delta" }, { "delta", delta }, { "usage", anthropic_usage(u) } };
    if (r_->context_management)
        md["context_management"] = json{ { "applied_edits", json::array() } };
    event(out, "message_delta", md);
    event(out, "message_stop", json{ { "type", "message_stop" } });
    return out;
}

std::string AnthropicStream::error_chunk(const ApiError& e, const TimingsOut&, const UsageOut&) {
    return started() + "event: error\ndata: " + anthropic_error_body(e) + "\n\n";
}

/* ================================================================== OpenAI's Responses API */

namespace {

/* An input content part as the chat path's: text, or an image as an image_url. */
int responses_part(const json& p, json* out, ApiError* err) {
    if (!p.is_object()) { *err = err_bad("input[].content[]", "expected an object"); return RAD_E_INVAL; }
    std::string type;
    if (!member_str(p, "type", "input[].content[].type", &type, err)) return RAD_E_INVAL;
    if (type == "input_text" || type == "output_text" || type == "text") {
        std::string text;
        if (!member_str(p, "text", "input[].content[].text", &text, err)) return RAD_E_INVAL;
        *out = json{ { "type", "text" }, { "text", text } };
        return RAD_OK;
    }
    if (type == "input_image") {
        const json u = p.contains("image_url") ? p.at("image_url") : json();
        std::string url;
        if (u.is_string()) url = u.get<std::string>();
        else if (u.is_object() && !member_str(u, "url", "input[].content[].image_url.url", &url, err))
            return RAD_E_INVAL;
        *out = json{ { "type", "image_url" }, { "image_url", { { "url", url } } } };
        return RAD_OK;
    }
    *err = err_unsupported("input[].content[].type",
                           "a content part of type \"" + type + "\" is not served");
    return RAD_E_UNSUPPORTED;
}

/* The id a response item carries for a call, from the call id this server minted ("call_..."). */
std::string fc_id(const std::string& call_id) {
    return "fc_" + (call_id.rfind("call_", 0) == 0 ? call_id.substr(5) : call_id);
}

json response_usage(const UsageOut& u) {
    return json{ { "input_tokens", u.prompt },
                 { "input_tokens_details", { { "cached_tokens", u.cached } } },
                 { "output_tokens", u.completion },
                 { "output_tokens_details", { { "reasoning_tokens", std::max<int64_t>(u.reasoning, 0) } } },
                 { "total_tokens", u.prompt + u.completion } };
}

json reasoning_item(const std::string& id, const std::string& text, const char* status) {
    return json{ { "id", id }, { "type", "reasoning" }, { "summary", json::array() },
                 { "content", json::array({ json{ { "type", "reasoning_text" }, { "text", text } } }) },
                 { "status", status } };
}
json text_part(const std::string& text) {
    return json{ { "type", "output_text" }, { "text", text }, { "annotations", json::array() },
                 { "logprobs", json::array() } };
}
json message_item(const std::string& id, const std::string& text, const char* status) {
    return json{ { "id", id }, { "type", "message" }, { "role", "assistant" }, { "status", status },
                 { "content", text.empty() && std::strcmp(status, "in_progress") == 0
                                  ? json::array() : json::array({ text_part(text) }) } };
}
/* A call to a flattened namespace tool reports its namespace and its own name apart, as the
 * Responses API's function_call item does. */
json call_item(const OaiRequest& r, const ToolCall& tc, const char* status) {
    json it = { { "id", fc_id(tc.id) }, { "type", "function_call" }, { "call_id", tc.id },
                { "name", tc.name }, { "arguments", tc.arguments }, { "status", status } };
    auto ns = r.tool_namespace.find(tc.name);
    if (ns != r.tool_namespace.end()) {
        it["namespace"] = ns->second.first;
        it["name"] = ns->second.second;
    }
    return it;
}

/* The response object, around its output items, with the request fields it echoes. */
json response_object(const OaiRequest& r, const char* status, const json& output,
                     const UsageOut* u, const char* incomplete) {
    json o = r.echo_json.empty() ? json::object() : json::parse(r.echo_json, nullptr, false);
    if (!o.is_object()) o = json::object();
    o["id"] = r.id;
    o["object"] = "response";
    o["created_at"] = r.created;
    o["status"] = status;
    o["model"] = r.model;
    o["output"] = output;
    o["error"] = nullptr;
    o["incomplete_details"] = incomplete ? json{ { "reason", incomplete } } : json(nullptr);
    if (u) o["usage"] = response_usage(*u);
    else o["usage"] = nullptr;
    return o;
}

}  /* namespace */

int responses_to_chat(const std::string& body, const OaiLimits& lim, std::string* chat,
                      std::string* echo, ApiError* err,
                      std::map<std::string, std::pair<std::string, std::string>>* ns) {
    json b;
    RAD_TRY(parse_request_object(body, lim, &b, err));
    auto prev = b.find("previous_response_id");
    if (prev != b.end() && prev->is_string() && !prev->get<std::string>().empty()) {
        *err = err_unsupported("previous_response_id", "responses are not stored on this server; "
                               "send the conversation as `input`");
        return RAD_E_UNSUPPORTED;
    }
    /* Refused rather than computed and dropped: several choices (a response is one reply), and
     * logprobs, which the Responses API reports per output_text part, a token split this server
     * does not make. */
    {
        int64_t n = 1;
        if (!get_int(b, "n", &n, err)) return RAD_E_INVAL;
        if (n != 1) {
            *err = err_unsupported("n", "a response is one reply; send one request a reply");
            return RAD_E_UNSUPPORTED;
        }
        bool lp = false;
        int64_t top = 0, np = -1;
        if (!get_bool(b, "logprobs", &lp, err) || !get_int(b, "top_logprobs", &top, err) ||
            !get_int(b, "n_probs", &np, err))
            return RAD_E_INVAL;
        const char* asked = lp ? "logprobs" : top > 0 ? "top_logprobs" : np > 0 ? "n_probs" : nullptr;
        auto inc = b.find("include");
        if (inc != b.end() && inc->is_array())
            for (const json& i : *inc)
                if (i == "message.output_text.logprobs") asked = "include";
        if (asked) {
            *err = err_unsupported(asked, "logprobs are not served on /v1/responses; "
                                          "/v1/chat/completions serves them");
            return RAD_E_UNSUPPORTED;
        }
    }
    /* Fields the Responses API does not define pass through to the chat request, as in
     * llama-server (e.g. chat_template_kwargs, top_k). Its own fields are translated below. The
     * chat stream's usage and progress chunks have no Responses frame and are dropped;
     * response.completed carries the usage. */
    json o = b;
    for (const char* k : { "input", "instructions", "tools", "tool_choice", "max_output_tokens",
                           "reasoning", "text", "previous_response_id", "store", "metadata",
                           "include", "truncation", "background", "conversation", "prompt",
                           "stream_options", "return_progress", "logprobs", "top_logprobs",
                           "n_probs" })
        o.erase(k);
    json msgs = json::array();
    if (b.contains("instructions") && b.at("instructions").is_string())
        msgs.push_back(json{ { "role", "system" }, { "content", b.at("instructions") } });

    auto in = b.find("input");
    if (in == b.end()) { *err = err_bad("input", "is required"); return RAD_E_INVAL; }
    auto last_assistant = [&]() -> json* {
        return !msgs.empty() && msgs.back().at("role") == "assistant" ? &msgs.back() : nullptr;
    };
    if (in->is_string()) {
        msgs.push_back(json{ { "role", "user" }, { "content", *in } });
    } else if (in->is_array()) {
        for (const json& item : *in) {
            if (!item.is_object()) { *err = err_bad("input[]", "expected an object"); return RAD_E_INVAL; }
            std::string type = item.contains("role") ? "message" : "";
            if (!member_str(item, "type", "input[].type", &type, err)) return RAD_E_INVAL;
            if (type == "message") {
                std::string role;
                if (!member_str(item, "role", "input[].role", &role, err)) return RAD_E_INVAL;
                const json content = item.contains("content") ? item.at("content") : json("");
                if (role == "assistant") {
                    /* An earlier model turn: its text joins any preceding reasoning item or call
                     * of the same turn. */
                    std::string text;
                    if (content.is_string()) {
                        text = content.get<std::string>();
                    } else if (content.is_array()) {
                        for (const json& p : content) {
                            std::string t;
                            if (!member_str(p, "text", "input[].content[].text", &t, err))
                                return RAD_E_INVAL;
                            text += t;
                        }
                    }
                    if (json* prev = last_assistant()) {
                        (*prev)["content"] = prev->at("content").get<std::string>() + text;
                    } else {
                        msgs.push_back(json{ { "role", "assistant" }, { "content", text } });
                    }
                    continue;
                }
                if (role != "user" && role != "system" && role != "developer") {
                    *err = err_bad("input[].role", "must be user, system, developer or assistant");
                    return RAD_E_INVAL;
                }
                json parts = json::array();
                if (content.is_string()) {
                    parts = content;
                } else if (content.is_array()) {
                    for (const json& p : content) {
                        json part;
                        RAD_TRY(responses_part(p, &part, err));
                        parts.push_back(std::move(part));
                    }
                }
                msgs.push_back(json{ { "role", role }, { "content", parts } });
            } else if (type == "function_call") {
                std::string id, name, args = "{}", space;
                if (!member_str(item, "call_id", "input[].call_id", &id, err) ||
                    !member_str(item, "name", "input[].name", &name, err) ||
                    !member_str(item, "arguments", "input[].arguments", &args, err) ||
                    !member_str(item, "namespace", "input[].namespace", &space, err))
                    return RAD_E_INVAL;
                /* A namespaced call, under the name its tool was flattened to below. */
                if (!space.empty()) name = space + "__" + name;
                json call = { { "id", id }, { "type", "function" },
                              { "function", { { "name", name }, { "arguments", args } } } };
                json* prev = last_assistant();
                if (!prev) {
                    msgs.push_back(json{ { "role", "assistant" }, { "content", "" } });
                    prev = &msgs.back();
                }
                (*prev)["tool_calls"].push_back(std::move(call));
            } else if (type == "function_call_output") {
                const json out = item.contains("output") ? item.at("output") : json("");
                json content;
                if (out.is_string()) {
                    content = out;
                } else if (out.is_array()) {
                    content = json::array();
                    for (const json& p : out) {
                        json part;
                        RAD_TRY(responses_part(p, &part, err));
                        content.push_back(std::move(part));
                    }
                } else {
                    content = "";
                }
                std::string id;
                if (!member_str(item, "call_id", "input[].call_id", &id, err)) return RAD_E_INVAL;
                msgs.push_back(json{ { "role", "tool" }, { "tool_call_id", id }, { "content", content } });
            } else if (type == "reasoning") {
                /* An earlier turn's reasoning, for templates that keep it across turns: the
                 * content's text, else the summary's. */
                std::string text;
                for (const char* k : { "content", "summary" }) {
                    if (!text.empty() || !item.contains(k) || !item.at(k).is_array()) continue;
                    for (const json& p : item.at(k)) {
                        std::string t;
                        if (!member_str(p, "text", "input[].content[].text", &t, err)) return RAD_E_INVAL;
                        text += t;
                    }
                }
                msgs.push_back(json{ { "role", "assistant" }, { "content", "" },
                                     { "reasoning_content", text } });
            } else {
                *err = err_unsupported("input[].type",
                                       "an input item of type \"" + type + "\" is not served");
                return RAD_E_UNSUPPORTED;
            }
        }
    } else {
        *err = err_bad("input", "expected a string or an array of items");
        return RAD_E_INVAL;
    }
    o["messages"] = std::move(msgs);

    /* Function tools, and the function tools a namespace groups (Codex sends every MCP server's
     * tools that way), flattened to "<namespace>__<name>": the chat path has one flat list, and
     * the reply's function_call items report the two apart again (OaiRequest::tool_namespace).
     * Built-in tools (web search, file search, computer use) and freeform custom tools have no
     * chat-completions counterpart and are left out. */
    json echo_tools = json::array();
    auto tit = b.find("tools");
    if (tit != b.end() && tit->is_array()) {
        json tools = json::array();
        std::set<std::string> names;
        auto add = [&](const json& t, const std::string& space) -> int {
            std::string name;
            if (!member_str(t, "name", space.empty() ? "tools[].name" : "tools[].tools[].name",
                            &name, err))
                return RAD_E_INVAL;
            const std::string flat = space.empty() ? name : space + "__" + name;
            if (!names.insert(flat).second) {
                *err = err_bad("tools", "two tools are named \"" + flat + "\" (a namespace's tools "
                               "are offered to the model as <namespace>__<name>)");
                return RAD_E_INVAL;
            }
            json fn = { { "name", flat },
                        { "parameters", t.contains("parameters") ? t.at("parameters") : json::object() } };
            if (t.contains("description")) fn["description"] = t.at("description");
            tools.push_back(json{ { "type", "function" }, { "function", fn } });
            if (!space.empty() && ns) (*ns)[flat] = { space, name };
            return RAD_OK;
        };
        for (const json& t : *tit) {
            const std::string type = str_or(t, "type");
            if (type == "namespace") {
                std::string space;
                if (!member_str(t, "name", "tools[].name", &space, err)) return RAD_E_INVAL;
                if (space.empty()) {
                    *err = err_bad("tools[].name", "a namespace needs a name");
                    return RAD_E_INVAL;
                }
                auto inner = t.find("tools");
                if (inner == t.end() || !inner->is_array()) {
                    *err = err_bad("tools[].tools", "a namespace holds an array of tools");
                    return RAD_E_INVAL;
                }
                for (const json& f : *inner) {
                    if (str_or(f, "type") != "function") {
                        RAD_DEBUG("responses: tool of type \"%s\" in namespace \"%s\" left out: "
                                  "no chat equivalent", str_or(f, "type", "?").c_str(),
                                  space.c_str());
                        continue;
                    }
                    RAD_TRY(add(f, space));
                }
                echo_tools.push_back(t);
                continue;
            }
            if (type != "function") {
                RAD_DEBUG("responses: tool of type \"%s\" left out: no chat equivalent",
                          str_or(t, "type", "?").c_str());
                continue;
            }
            RAD_TRY(add(t, ""));
            echo_tools.push_back(t);
        }
        if (!tools.empty()) o["tools"] = std::move(tools);
    }
    auto tc = b.find("tool_choice");
    if (tc != b.end() && !tc->is_null()) {
        std::string type = "auto", name, space;
        if (tc->is_object() && (!member_str(*tc, "type", "tool_choice.type", &type, err) ||
                                !member_str(*tc, "name", "tool_choice.name", &name, err) ||
                                !member_str(*tc, "namespace", "tool_choice.namespace", &space, err)))
            return RAD_E_INVAL;
        if (!space.empty()) name = space + "__" + name;
        if (!tc->is_object()) o["tool_choice"] = *tc;
        else if (type == "function")
            o["tool_choice"] = json{ { "type", "function" }, { "function", { { "name", name } } } };
        else o["tool_choice"] = type;
    }
    if (b.contains("max_output_tokens")) o["max_tokens"] = b.at("max_output_tokens");
    auto re = b.find("reasoning");
    if (re != b.end() && re->is_object() && re->contains("effort") && re->at("effort").is_string())
        o["reasoning_effort"] = re->at("effort");
    /* Structured output: text.format is response_format under another name. */
    auto tx = b.find("text");
    if (tx != b.end() && tx->is_object() && tx->contains("format") && tx->at("format").is_object()) {
        const json& f = tx->at("format");
        std::string type = "text", name = "schema";
        if (!member_str(f, "type", "text.format.type", &type, err) ||
            !member_str(f, "name", "text.format.name", &name, err))
            return RAD_E_INVAL;
        if (type == "json_schema")
            o["response_format"] = json{ { "type", "json_schema" },
                                          { "json_schema", { { "name", name },
                                                             { "schema", f.contains("schema") ? f.at("schema") : json::object() } } } };
        else if (type == "json_object")
            o["response_format"] = json{ { "type", "json_object" } };
    }
    *chat = o.dump();

    json e;
    e["instructions"] = b.contains("instructions") ? b.at("instructions") : json(nullptr);
    e["tools"] = echo_tools;
    e["tool_choice"] = b.contains("tool_choice") ? b.at("tool_choice") : json("auto");
    e["parallel_tool_calls"] = !b.contains("parallel_tool_calls") || b.at("parallel_tool_calls") != false;
    e["temperature"] = b.contains("temperature") ? b.at("temperature") : json(nullptr);
    e["top_p"] = b.contains("top_p") ? b.at("top_p") : json(nullptr);
    e["max_output_tokens"] = b.contains("max_output_tokens") ? b.at("max_output_tokens") : json(nullptr);
    e["text"] = b.contains("text") ? b.at("text") : json{ { "format", { { "type", "text" } } } };
    e["reasoning"] = b.contains("reasoning") ? b.at("reasoning")
                                             : json{ { "effort", nullptr }, { "summary", nullptr } };
    e["metadata"] = b.contains("metadata") ? b.at("metadata") : json::object();
    e["previous_response_id"] = nullptr;
    e["store"] = false;
    e["truncation"] = "disabled";
    if (echo) *echo = e.dump();
    return RAD_OK;
}

std::string responses_body(const OaiRequest& r, const ChoiceOut& c, const UsageOut& u) {
    json out = json::array();
    if (!c.msg.reasoning.empty())
        out.push_back(reasoning_item(random_id("rs_"), c.msg.reasoning, "completed"));
    if (!c.msg.content.empty())
        out.push_back(message_item(random_id("msg_"), c.msg.content, "completed"));
    for (const ToolCall& tc : c.msg.tool_calls) out.push_back(call_item(r, tc, "completed"));
    const bool cut = c.finish == Finish::Length;
    return response_object(r, cut ? "incomplete" : "completed", out, &u,
                           cut ? "max_output_tokens" : nullptr).dump();
}

ResponsesStream::ResponsesStream(const OaiRequest& r, int index, const IReplyParser* parser)
    : ReplyStream(r, index, parser), r_(&r) {}

void ResponsesStream::event(std::string& out, const char* type, json data) {
    data["type"] = type;
    data["sequence_number"] = seq_++;
    out += "event: ";
    out += type;
    out += "\ndata: ";
    out += data.dump();
    out += "\n\n";
}

json ResponsesStream::response(const char* status) const {
    const bool done = std::strcmp(status, "in_progress") != 0;
    return response_object(*r_, status, json(items_), done ? &usage_ : nullptr, incomplete_);
}

std::string ResponsesStream::first_chunk() {
    std::string out;
    event(out, "response.created", json{ { "response", response("in_progress") } });
    event(out, "response.in_progress", json{ { "response", response("in_progress") } });
    return out;
}

void ResponsesStream::open(std::string& out, Item k, size_t call) {
    close(out);
    item_ = k;
    text_.clear();
    const int64_t at = (int64_t)items_.size();
    if (k == Item::Reasoning) {
        item_id_ = random_id("rs_");
        json it = reasoning_item(item_id_, "", "in_progress");
        it["content"] = json::array();
        event(out, "response.output_item.added", json{ { "output_index", at }, { "item", it } });
        event(out, "response.content_part.added",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", { { "type", "reasoning_text" }, { "text", "" } } } });
    } else if (k == Item::Message) {
        item_id_ = random_id("msg_");
        event(out, "response.output_item.added",
              json{ { "output_index", at }, { "item", message_item(item_id_, "", "in_progress") } });
        event(out, "response.content_part.added",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", text_part("") } });
    } else {
        call_ = call;
        const ToolCall& tc = last_.tool_calls[call];
        item_id_ = fc_id(tc.id);
        event(out, "response.output_item.added",
              json{ { "output_index", at }, { "item", call_item(*r_, tc, "in_progress") } });
    }
}

void ResponsesStream::close(std::string& out) {
    if (item_ == Item::None) return;
    const int64_t at = (int64_t)items_.size();
    json item;
    if (item_ == Item::Reasoning) {
        event(out, "response.reasoning_text.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "text", text_ } });
        event(out, "response.content_part.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", { { "type", "reasoning_text" }, { "text", text_ } } } });
        item = reasoning_item(item_id_, text_, "completed");
    } else if (item_ == Item::Message) {
        event(out, "response.output_text.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "text", text_ }, { "logprobs", json::array() } });
        event(out, "response.content_part.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "content_index", 0 },
                    { "part", text_part(text_) } });
        item = message_item(item_id_, text_, "completed");
    } else {
        /* A call the reply ended inside was announced but never closed. Its item is
         * "incomplete", the Responses API's status for an item cut short, so a client acting on
         * the terminal response does not run it with truncated arguments. */
        const ToolCall& tc = last_.tool_calls[call_];
        const bool cut = finished_ && last_.open_call && call_ + 1 == last_.tool_calls.size();
        event(out, "response.function_call_arguments.done",
              json{ { "item_id", item_id_ }, { "output_index", at }, { "arguments", tc.arguments } });
        item = call_item(*r_, tc, cut ? "incomplete" : "completed");
    }
    event(out, "response.output_item.done", json{ { "output_index", at }, { "item", item } });
    items_.push_back(std::move(item));
    item_ = Item::None;
}

std::string ResponsesStream::emit(const std::vector<ReplyDelta>& ev) {
    std::string out;
    for (const ReplyDelta& d : ev) {
        if (!absorb(d)) continue;
        /* The open item's index; read after open(), which may close one first. */
        auto at = [this] { return (int64_t)items_.size(); };
        switch (d.kind) {
        case ReplyDelta::Reasoning:
            if (item_ != Item::Reasoning) open(out, Item::Reasoning, 0);
            text_ += d.text;
            event(out, "response.reasoning_text.delta",
                  json{ { "item_id", item_id_ }, { "output_index", at() },
                        { "content_index", 0 }, { "delta", d.text } });
            break;
        case ReplyDelta::Content:
            if (item_ != Item::Message) open(out, Item::Message, 0);
            text_ += d.text;
            event(out, "response.output_text.delta",
                  json{ { "item_id", item_id_ }, { "output_index", at() },
                        { "content_index", 0 }, { "delta", d.text }, { "logprobs", json::array() } });
            break;
        case ReplyDelta::CallBegin:
            open(out, Item::Call, last_.tool_calls.size() - 1);
            break;
        case ReplyDelta::CallArgs:
            event(out, "response.function_call_arguments.delta",
                  json{ { "item_id", item_id_ }, { "output_index", at() }, { "delta", d.text } });
            break;
        case ReplyDelta::CallEnd:
            close(out);
            break;
        }
    }
    return out;
}

std::string ResponsesStream::final_chunk(Finish f, const TimingsOut&, const UsageOut& u) {
    std::string out = push(std::string_view(), true);
    close(out);
    usage_ = u;
    const bool cut = adjusted_finish(f) == Finish::Length;
    incomplete_ = cut ? "max_output_tokens" : nullptr;
    event(out, cut ? "response.incomplete" : "response.completed",
          json{ { "response", response(cut ? "incomplete" : "completed") } });
    return out;
}

std::string ResponsesStream::error_chunk(const ApiError& e, const TimingsOut&, const UsageOut& u) {
    std::string out;
    usage_ = u;
    json resp = response("failed");
    resp["error"] = json{ { "code", "server_error" }, { "message", e.message } };
    event(out, "response.failed", json{ { "response", std::move(resp) } });
    return out;
}

}  /* namespace server */
}  /* namespace rad */
