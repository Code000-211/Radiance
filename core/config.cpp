/* config.cpp -- the command line. Every pool is budgeted explicitly, with no profiling pass,
 * utilisation fraction or workload inference. The KV-versus-weights trade is workload-dependent
 * and the operator's to make; re-optimising it quietly would make benchmarks irreproducible
 * (spec §6).
 */
#include "rad_core.h"
#include "format/rope_override.h"
#include "sample/host_ref.h"
#include "server/oai.h"
#include "diag/diag.h"
#include "util/kernel_list.h"
#include "place/planner.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace rad {

struct Flag {
    const char* name;
    const char* arg;     /* null for a boolean */
    const char* help;
};

/* Built from kDraftWindowDefault so the default is stated once. Defined before the table that
 * points at it, which fixes their initialisation order. */
static const std::string g_draft_window_help =
    std::string("fixed | adaptive: how many drafts each sequence carries. fixed drafts\n"
                "                                    --num-speculative-tokens every step; adaptive makes that\n"
                "                                    the ceiling and follows each sequence's acceptance and the\n"
                "                                    measured step cost (a serial draft head only). Default ") +
    draft_window_name(kDraftWindowDefault);

static const Flag g_flags[] = {
    { "--model",              "PATH",   "the .rad container, a checkpoint served as it stands, or a\n"
                                        "Hugging Face repo id (owner/name), fetched before startup" },
    { "--download-dir",       "DIR",    "where a repo id is fetched to; default\n"
                                        "$XDG_CACHE_HOME/radiance/hf, else ~/.cache/radiance/hf" },
    { "--radiance-home",      "DIRS",   "where kernels/, architectures/ and quantizers/ live; several\n"
                                        "homes separated by ':' are searched in order, the first winning" },
    { "--kernels",            "LIST",   "the kernel libraries to use, separated by ',' or ':', first wins; unnamed ones are not loaded. Default: every library found" },

    { "--tp",                 "N",      "tensor-parallel ranks" },
    { "--tp-wire",            "MODE",   "cross-rank all-reduce payload: exact (default) or wht6, a\n"
                                        "lossy wire: rotated 6-bit at 2 ranks, tiered int8 at 3\n"
                                        "and 4, none at 8" },
    { "--tp-wire-min-kb",     "N",      "smallest all-reduce, in KiB, the lossy wire is used for" },
    { "--p2p",                "MODE",   "how the collectives reach the other cards: auto (default)\n"
                                        "asks the driver, on requires peer access and stops without\n"
                                        "it, off exchanges through pinned host memory" },
    { "--max-num-batched-tokens", "N",  "the largest token count one step may carry" },
    { "--busy-prefill-chunk", "MODE",   "a prefill chunk's share of a step that has other work: auto\n"
                                        "(default) leaves decode rows and new requests their room,\n"
                                        "off takes the whole step, N holds the chunk to N tokens" },
    { "--prefill-interleave", "N",      "decode-only steps after each prefill chunk while other\n"
                                        "sequences decode, at most 64; 0 (default) lets every step\n"
                                        "carry a chunk" },
    { "--max-num-seqs",       "N",      "the largest number of sequences one step may carry" },
    { "--max-model-len",      "N",      "context bound; 0 takes the model's training context" },
    { "--rope-scaling",       "SPEC",   "yarn:FACTOR[:ORIG]: static YaRN over the container's own rotary\n"
                                        "keys, no reconvert (docs/YARN.md); none: off" },

    { "--gpu-headroom-mib",   "N",      "VRAM left unclaimed on every card (default 96)" },
    { "--kv-cache-dtype",     "T",      "bf16 | fp8 | fp8_v4 | k6_v4 -- fp8 halves what a token of\n"
                                        "context costs; fp8_v4 keeps V in 4 bits (0.78 of fp8), k6_v4\n"
                                        "also K in 6 (0.69); both Qwen3.8-27B and -Flash-Next" },
    { "--host-pool-mib",      "N",      "pinned host tier" },
    { "--host-headroom-mib",  "N",      "host RAM to leave free for the system when radiance\n"
    "                                    pins its host memory at startup (default 4096)" },
    { "--embedding-placement", "WHERE", "ram (default) | vram: where the token embedding lives. ram\n"
    "                                    pins it in host memory beside --host-pool-mib, read over the\n"
    "                                    link in place; by default a model whose output head reads the\n"
    "                                    table, or a host without the memory, keeps it on the card" },
    { "--vision-tower-placement", "WHERE", "vram (default) | ram: where the encoder towers live (the\n"
    "                                    vision tower; the audio encoder too). ram pins them in host\n"
    "                                    memory beside --host-pool-mib, read over the link once an\n"
    "                                    image, and gives their VRAM to the KV cache" },
    { "--ngram-placement",    "WHERE",  "auto (default) | vram | ram | disk: where an n-gram\n"
    "                                    embedding table lives (Qwen3.8-Flash-Next: 47.7 GiB).\n"
    "                                    vram puts all of it on every card; ram copies it into\n"
    "                                    host memory at load, pinned for the cards to read\n"
    "                                    (locked, where only a host kernel reads it); disk reads\n"
    "                                    its rows from the container on the host. auto takes the\n"
    "                                    first of the three that holds it" },
    { "--weights-disk-tier",  nullptr,  "read weights that fit neither VRAM nor the host pool from the\n"
    "                                    container on disk (O_DIRECT); a routed expert is read a layer\n"
    "                                    at a time, so a step is paced by the drive" },
    { "--prefix-cache-mib",   "N",      "copy finished conversations' KV to host memory, up to N MiB,\n"
    "                                    so their VRAM can go to experts and a return restores\n"
    "                                    instead of re-prefilling. A full tier drops its least\n"
    "                                    recently used entries. 0 is off" },
    { "--prefix-cache-ttl",   "S",      "seconds a cached conversation no request uses is kept, in VRAM\n"
    "                                    or host memory (default 3600). 0 keeps it until the memory\n"
    "                                    is wanted" },
    { "--memory-archives",    "N",      "conversations the memory endpoints may pin at once (default 8):\n"
    "                                    each keeps its cached blocks and one snapshot slot until\n"
    "                                    deleted. 0 refuses every archive" },
    { "--memory-dir",         "DIR",    "where archives are written so a restart keeps them, and read\n"
    "                                    back from at startup. Needs --prefix-cache-mib: they are\n"
    "                                    written from host memory. Default: memory only" },

    { "--lora",               "ADAPTER", "NAME=PATH: serve a LoRA adapter beside the base model\n"
    "                                    (repeatable). A text model takes a PEFT directory and\n"
    "                                    serves it as model NAME; an image or video generator takes\n"
    "                                    a .safetensors file (kohya, ComfyUI, PEFT, LyCORIS) that\n"
    "                                    a request names in `loras`. The base stays as it is"},
    { "--lora-dir",           "DIR",    "image and video generators: a request may name any\n"
    "                                    NAME.safetensors under DIR in `loras`, opened when first\n"
    "                                    asked for"},
    { "--max-loras",          "N",      "adapter slots resident at once (default: the number of\n"
    "                                    --lora, 2 with --lora-dir alone). A text model's slot holds\n"
    "                                    one adapter, a generator's one request's combination"},
    { "--max-lora-rank",      "R",      "the rank each slot is sized for (default 16 for a text\n"
    "                                    model, 64 for a generator); a text model's larger adapter is\n"
    "                                    refused, a generator's larger combination truncated"},
    { "--lora-target-modules", "LIST",  "comma-separated HF module leaf names adapters may\n"
    "                                    target (q_proj,k_proj,...). Default: what the --lora\n"
    "                                    adapters target. Needed with --max-loras and no --lora"},

    { "--placement",          "MODE",   "all_vram | layer_offload | expert_tiered | auto" },
    { "--control-vector",     "MODE",   "off | on | disabled: load the control-vector file found\n"
    "                                    for this model and start requests off (off) or on (on).\n"
    "                                    disabled -- the default, also with no flag at all -- loads\n"
    "                                    nothing: no projection op is declared, and a request that\n"
    "                                    asks for the vector is refused" },
    { "--deterministic",      nullptr,  "pin placement to the plan the loader made, so the resident\n"
    "                                    set cannot change under the heat engine. The sampler is\n"
    "                                    already order-independent and needs no pinning; note that\n"
    "                                    this does NOT make the engine bit-reproducible across\n"
    "                                    batch positions -- see spec §19" },
    { "--checkpoint-interval","N",      "linear-attention state checkpoint interval, in tokens" },
    { "--prefill-span",       "N",      "checkpoint intervals one step may carry of a prefill alone\n"
    "                                    in its step, from the prompt's second interval on (default\n"
    "                                    1); needs --max-num-batched-tokens of N intervals. Every\n"
    "                                    other step is planned at the budget less the extra\n"
    "                                    intervals, and the checkpoints inside a span are not\n"
    "                                    written" },
    /* A total, not per sequence: the snapshot store is one pool of N slots
     * (KVManager::set_checkpoint_store). */
    { "--checkpoint-slots",   "N",      "linear-state or sliding-window snapshots the server holds in\n"
    "                                    total, in pinned host memory. Default `auto`: what\n"
    "                                    --checkpoint-policy keeps for a --max-model-len session, plus\n"
    "                                    one, for each of --max-num-seqs, within an eighth of host\n"
    "                                    memory. N fixes the total; 0 keeps none" },
    { "--no-prefix-cache",    nullptr,  "disable prefix caching" },
    { "--checkpoint-policy",  "NAME",   "which snapshots a session retains: geometric-backoff "
                                        "(default), tip-only, keep-all" },

    { "--num-speculative-tokens", "N",  "speculative window; 0 disables. Default `auto`: the drafter's\n"
    "                                    own operating point, which the plugin states. A container\n"
    "                                    with no draft head is unaffected -- auto resolves to 0." },
    { "--draft-window",       "MODE",   g_draft_window_help.c_str() },
    { "--draft-model",        "PATH",   "NOT IMPLEMENTED -- merge the drafter at convert time" },

    { "--host",               "ADDR",   "listen address" },
    { "--port",               "N",      "listen port" },
    { "--api-key",            "KEY",    "require `Authorization: Bearer KEY` or `X-Api-Key: KEY` on every request but /health,\n"
    "                                    /v1/health, /ping and the dashboard page" },
    { "--served-model-name",  "NAME",   "the model id /v1/models lists and responses carry; default the container's name" },
    { "--mm-max-patches",     "N",      "patches one encoder pass carries, and so the largest image (a patch is 16x16\n"
                                        "pixels). Default `auto`: 16384 when the container carries a vision tower. 0\n"
                                        "serves text only and keeps the tower off the card" },
    { "--generation-config",  "PATH",   "a generation_config.json whose sampler settings become the default\n"
    "                                    for requests that send none. Without it a <model>.generation.json\n"
    "                                    beside the container is used, then the container's own\n"
    "                                    `generation.*` metadata, then the built-in defaults" },
    { "--override-chat-template", "PATH", "a Jinja chat template file, used in place of the one the container carries" },
    { "--temp",               "F",      "default temperature for requests that do not set one" },
    { "--top-k",              "N",      "default top_k for requests that do not set one; 0 is the 1024 most\n"
    "                                    likely tokens, the sampler's candidate width" },
    { "--top-p",              "F",      "default top_p for requests that do not set one" },
    { "--min-p",              "F",      "default min_p (0 = off) for requests that do not set one.\n"
    "                                    These four override whatever the container or its generation\n"
    "                                    config asked for; unset, they are resolved from it" },
    { "--reasoning-effort",   "S",      "default `reasoning_effort` for chat requests that do not set one.\n"
    "                                    The value goes to the model's chat template verbatim, so the legal\n"
    "                                    set is the template's: Qwen3.8 takes xhigh|medium|low and defaults\n"
    "                                    to xhigh, which on a short task can spend the whole token budget\n"
    "                                    inside <think> and answer nothing. \"none\" turns thinking off" },
    { "--reasoning-effort-map", "MAP",  "FROM=TO[,FROM=TO...]: a `reasoning_effort` a request sends, replaced before\n"
    "                                    the chat template sees it -- high=xhigh lets a client that always sends\n"
    "                                    OpenAI's top level reach Qwen3.8's. TO is checked against the template at\n"
    "                                    startup; none and minimal turn thinking off. Repeatable" },
    { "--thinking-token-budget", "N",   "default `thinking_token_budget` for chat requests that do not set one: at\n"
    "                                    N reasoning tokens the reasoning block is closed and the answer follows,\n"
    "                                    so a client's max_tokens is not spent on reasoning alone. Unset is no\n"
    "                                    limit. Needs a reasoning block that ends in one token, as </think> is" },
    { "--typical-p",          "F",      "default typical_p, (0, 1]. This flag and the eleven below set the default\n"
    "                                    for requests that do not name the field; a request that names it wins" },
    { "--presence-penalty",   "F",      "default presence_penalty, [-2, 2]" },
    { "--frequency-penalty",  "F",      "default frequency_penalty, [-2, 2]" },
    { "--repetition-penalty", "F",      "default repetition_penalty, > 0" },
    { "--repeat-last-n",      "N",      "default repeat_last_n, the tokens those penalties look back over; -1 is all" },
    { "--dry-multiplier",     "F",      "default dry_multiplier; 0 is DRY off" },
    { "--dry-base",           "F",      "default dry_base, >= 1" },
    { "--dry-allowed-length", "N",      "default dry_allowed_length, >= 1" },
    { "--dry-penalty-last-n", "N",      "default dry_penalty_last_n; -1 is the whole context" },
    { "--dry-sequence-breakers", "JSON", "default dry_sequence_breakers, a JSON array of at most 16 strings" },
    { "--xtc-probability",    "F",      "default xtc_probability, [0, 1]" },
    { "--xtc-threshold",      "F",      "default xtc_threshold, [0, 1]" },
    { "--default-max-tokens", "N",      "max_tokens for requests that do not set one: N, or `auto` (default),\n"
    "                                    whatever the context leaves after the prompt" },
    { "--max-tokens-cap",     "N",      "the most any request may generate. A larger max_tokens is cut to it rather\n"
    "                                    than refused, and the reply ends with finish_reason \"length\". 0 is none\n"
    "                                    (default)" },
    { "--max-n",              "N",      "the largest `n` a request may ask for (default: the larger of 8 and\n"
    "                                    --max-num-seqs)" },
    { "--max-queued-requests", "N",     "requests admitted and unfinished at once, past which the answer is 429\n"
    "                                    (default 8 x --max-num-seqs)" },
    { "--max-stop-strings",   "N",      "stop strings one request may send (default 64)" },
    { "--max-stop-bytes",     "N",      "bytes in one stop string (default 4096)" },
    { "--chat-template-kwargs", "JSON", "variables handed to the chat template on every chat request: a JSON object,\n"
    "                                    or @FILE to read one. Repeatable, a later key replacing an earlier one.\n"
    "                                    A key the request's own chat_template_kwargs names is the request's, and a\n"
    "                                    request that says anything about thinking replaces enable_thinking and\n"
    "                                    reasoning_effort here" },
    { "--reasoning-format",   "MODE",   "auto (default): a reasoning block comes back as reasoning_content.\n"
    "                                    none: it stays inline in content" },

    { "--http-threads",       "N",      "HTTP workers kept while idle (default: one per core, at least 4)" },
    { "--read-timeout",       "S",      "seconds a client may go silent while sending a request (default 30)" },
    { "--write-timeout",      "S",      "seconds one write to a client may block (default 600)" },
    { "--keep-alive-timeout", "S",      "seconds an idle connection is held open between requests (default 5)" },
    { "--max-body-mib",       "N",      "the largest request body; images and video arrive base64 inside it\n"
    "                                    (default 512)" },
    { "--retry-after",        "S",      "the Retry-After a 429 carries, in seconds (default 1)" },
    { "--no-cors",            nullptr,  "send no CORS headers, so a page on another origin cannot read the API" },

    { "--image-min-pixels",   "N",      "an image is resized to at least N pixels (default: the container's)" },
    { "--image-max-pixels",   "N",      "and to at most N; one encoder pass (--mm-max-patches) still bounds it" },
    { "--video-min-pixels",   "N",      "a video's sampled frames together resize to at least N pixels" },
    { "--video-max-pixels",   "N",      "and to at most N" },
    { "--video-fps",          "F",      "frames sampled per second of video" },
    { "--video-min-frames",   "N",      "the fewest frames sampled from a video" },
    { "--video-max-frames",   "N",      "the most frames sampled from a video" },
    { "--video-max-frame-tokens", "N",  "the most prompt tokens one video frame may become; 0 is no cap (a token\n"
    "                                    budget processor -- Gemma 4's -- takes 70, 140, 280, 560 or 1120)" },
    { "--image-tokens",       "N",      "a token-budget processor's image budget (Gemma 4: 70, 140, 280, 560, 1120;\n"
    "                                    default the container's); a request's `image_tokens` overrides it" },
    { "--max-source-pixels",  "N",      "the largest image or video frame the decoder accepts, before any resize\n"
    "                                    (default 67108864). The media flags above default to what the container\n"
    "                                    states, and do nothing for a model that takes no media" },

    { "--kld-record",         "DIR",    "instead of serving, run --kld-corpus through prefill and write this\n"
    "                                    model's log-probabilities at every scored position to DIR: the\n"
    "                                    reference a quantised model is measured against" },
    { "--kld-ref",            "DIR",    "instead of serving, score this model against the reference in DIR:\n"
    "                                    mean, median, 99/99.9/99.99th percentile KL divergence and top-1\n"
    "                                    agreement, over the positions and tokens DIR was recorded at" },
    { "--kld-corpus",         "FILE",   "JSONL, one {\"prompt\", \"score_from\"?, \"source\"?} a line; --kld-record only" },
    { "--kld-out",            "FILE",   "the --kld-ref report as JSON, and every position's numbers in FILE.rows" },
    { "--kld-lora",           "NAME",   "run --kld-record / --kld-ref through the --lora adapter NAME" },
    { "--kld-lora-sources",   "PREFIX", "--kld-lora only for documents whose source starts with PREFIX; the\n"
    "                                    rest run as the base, in the same steps (a mixed batch)" },

    { "--debug-graph",        nullptr,  "print the entire declared graph and exit-quality detail" },
    { "--debug-selection",    nullptr,  "print the selection table" },
    { "--debug-placement",    nullptr,  "print the placement plan" },
    { "--debug-accept-reference-kernels", nullptr,
      "serve even when ops resolve only to a reference library. NOT a production configuration: the reference\n"
      "                              implementation is an oracle and a last resort, and falling back\n"
      "                              to it is orders of magnitude of speed with nothing to say so." },
    { "--profile-ops",        nullptr,  "per-op device timing (see the caveat it prints)" },
    { "--live",               nullptr,  "the live view: cards, throughput, draft acceptance, the expert plane.\n"
    "                                    Redirected it prints one greppable key=value line instead" },
    { "-v",                   nullptr,  "verbose" },
    { "-vv",                  nullptr,  "very verbose" },
    { "--help",               nullptr,  "this" },
};

void config_usage(const char* argv0) {
    fprintf(stderr, "usage: %s --model FILE [options]\n\n", argv0);
    /* Align help continuation lines under the first, whatever indentation the string carries. */
    for (const Flag& f : g_flags) {
        char left[64];
        snprintf(left, sizeof left, "%s%s%s", f.name, f.arg ? " " : "", f.arg ? f.arg : "");
        std::string help;
        for (const char* p = f.help; *p; ++p) {
            help += *p;
            if (*p != '\n') continue;
            help.append(35, ' ');
            while (p[1] == ' ') ++p;
        }
        fprintf(stderr, "  %-32s %s\n", left, help.c_str());
    }
    fprintf(stderr,
        "\nEvery pool is budgeted explicitly: the split between KV blocks and resident weights\n"
        "is the operator's to set.\n");
}

/* The four names exactly, lower case; anything else is refused rather than mapped to a tier. */
static bool parse_table_placement(const char* v, TablePlacement* out) {
    if      (!std::strcmp(v, "auto")) *out = TablePlacement::Auto;
    else if (!std::strcmp(v, "vram")) *out = TablePlacement::Vram;
    else if (!std::strcmp(v, "ram"))  *out = TablePlacement::Ram;
    else if (!std::strcmp(v, "disk")) *out = TablePlacement::Disk;
    else return false;
    return true;
}

/* The whole argument must be a number: atof reads "high" as 0, which for --temp silently means
 * greedy decoding. */
static bool parse_num(const char* flag, const char* v, double* out) {
    char* end = nullptr;
    errno = 0;
    const double d = std::strtod(v, &end);
    if (!*v || !end || *end || errno == ERANGE || !std::isfinite(d)) {
        RAD_ERR("%s takes a number, not '%s'", flag, v);
        return false;
    }
    *out = d;
    return true;
}

static bool parse_int(const char* flag, const char* v, int64_t lo, int64_t hi, int64_t* out) {
    char* end = nullptr;
    errno = 0;
    const long long n = std::strtoll(v, &end, 10);
    if (!*v || !end || *end || errno == ERANGE) {
        RAD_ERR("%s takes a whole number, not '%s'", flag, v);
        return false;
    }
    if (n < lo || n > hi) {
        if (lo > INT32_MIN && (hi == INT32_MAX || hi == INT64_MAX))
            RAD_ERR("%s must be >= %lld, not %lld", flag, (long long)lo, n);
        else
            RAD_ERR("%s must be in [%lld, %lld], not %lld", flag, (long long)lo, (long long)hi, n);
        return false;
    }
    *out = n;
    return true;
}

/* Sampler defaults are checked against the request range (server::sampler_value_ok) at startup;
 * an out-of-range default would 400 every request that omits the field. */
static bool sampler_num(const char* flag, const char* field, const char* v, double* out) {
    if (!parse_num(flag, v, out)) return false;
    std::string why;
    if (!server::sampler_value_ok(field, *out, &why)) {
        RAD_ERR("%s %s: %s %s", flag, v, field, why.c_str());
        return false;
    }
    return true;
}
static bool sampler_int(const char* flag, const char* field, const char* v, int* out) {
    int64_t n = 0;
    double d = 0;
    if (!parse_int(flag, v, INT32_MIN, INT32_MAX, &n) || !sampler_num(flag, field, v, &d))
        return false;
    *out = (int)n;
    return true;
}

/* Names the renderer sets itself. Kwargs are written over the template context
 * (common_chat_template_direct_apply_impl), so a kwarg with one of these names would replace every
 * request's messages, tools or special tokens. */
static const char* const kTemplateOwnNames[] = {
    "messages", "tools", "bos_token", "eos_token", "add_generation_prompt",
};

/* --chat-template-kwargs: a JSON object, or @FILE holding one, merged key by key into `out`. */
static bool parse_template_kwargs(const char* v, std::map<std::string, std::string>* out) {
    std::string text = v;
    std::string src = "--chat-template-kwargs";
    if (!text.empty() && text[0] == '@') {
        std::ifstream f(text.substr(1), std::ios::binary);
        if (!f) {
            RAD_ERR("--chat-template-kwargs %s: cannot read %s", v, text.c_str() + 1);
            return false;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        src += " " + text.substr(1);
        text = ss.str();
    }
    const nlohmann::json j = nlohmann::json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        RAD_ERR("%s takes a JSON object, such as '{\"enable_thinking\": false}'", src.c_str());
        return false;
    }
    for (auto it = j.begin(); it != j.end(); ++it) {
        for (const char* own : kTemplateOwnNames)
            if (it.key() == own) {
                RAD_ERR("%s: '%s' is set by the server for every request and cannot be a template "
                        "variable", src.c_str(), own);
                return false;
            }
        if (it.key() == "enable_thinking" && !it.value().is_boolean()) {
            RAD_ERR("%s: enable_thinking takes true or false", src.c_str());
            return false;
        }
        /* The template context takes JSON text, so strings stay quoted. */
        (*out)[it.key()] = it.value().dump();
    }
    return true;
}

/* Returns 0 on success, 1 for --help (caller exits 0), negative on a bad argument. An unknown flag
 * is an error, not a warning, since a mistyped budget flag changes the plan. */
int config_parse(int argc, char** argv, Config* c) {
    /* --help wins from anywhere on the line, even after an otherwise fatal argument. */
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) return 1;

    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { RAD_ERR("%s needs an argument", argv[i]); return nullptr; }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const char* v = nullptr;
        auto S = [&](const char* n) { return a == n; };
        /* The flag's argument as a whole number in [lo, hi], stored into *out. */
        auto num = [&](int64_t lo, int64_t hi, auto* out) {
            int64_t n = 0;
            if (!(v = need(i)) || !parse_int(a.c_str(), v, lo, hi, &n)) return false;
            *out = static_cast<std::remove_reference_t<decltype(*out)>>(n);
            return true;
        };

        if (S("-v"))                           log_set_level(Log::Debug);
        else if (S("-vv"))                     log_set_level(Log::Trace);
        else if (S("--model"))                 { if (!(v = need(i))) return RAD_E_INVAL; c->model = v; }
        else if (S("--download-dir"))          { if (!(v = need(i))) return RAD_E_INVAL; c->download_dir = v; }
        else if (S("--radiance-home"))         { if (!(v = need(i))) return RAD_E_INVAL; c->radiance_home = v; }
        else if (S("--kernels"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kernel_hierarchy = rad_split_kernel_list(v); }
        else if (S("--tp"))                    { if (!num(1, INT32_MAX, &c->tp)) return RAD_E_INVAL; }
        else if (S("--tp-wire")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "exact")) c->tp_wire_exact = 1;
            else if (!std::strcmp(v, "wht6"))  c->tp_wire_exact = 0;
            else { RAD_ERR("--tp-wire takes exact or wht6, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--p2p")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "auto")) c->p2p = RAD_P2P_AUTO;
            else if (!std::strcmp(v, "on"))   c->p2p = RAD_P2P_ON;
            else if (!std::strcmp(v, "off"))  c->p2p = RAD_P2P_OFF;
            else { RAD_ERR("--p2p takes auto, on or off, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--tp-wire-min-kb"))        { if (!num(0, INT64_MAX, &c->tp_wire_min_kb)) return RAD_E_INVAL; }
        else if (S("--max-num-batched-tokens")){ if (!num(1, INT64_MAX, &c->max_tok)) return RAD_E_INVAL; }
        else if (S("--busy-prefill-chunk")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "auto")) c->busy_prefill_chunk = -1;
            else if (!std::strcmp(v, "off"))  c->busy_prefill_chunk = 0;
            else if (!parse_int(a.c_str(), v, 1, INT64_MAX, &c->busy_prefill_chunk)) return RAD_E_INVAL;
        }
        else if (S("--prefill-interleave"))    { if (!num(0, 64, &c->prefill_interleave)) return RAD_E_INVAL; }
        else if (S("--max-num-seqs"))          { if (!num(1, INT64_MAX, &c->max_seqs)) return RAD_E_INVAL; }
        else if (S("--max-model-len"))         { if (!num(0, INT64_MAX, &c->max_ctx)) return RAD_E_INVAL; }
        else if (S("--rope-scaling")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* Checked here so a typo is refused rather than served with the trained rotary. */
            RopeSpec rs;
            if (!rope_spec_parse(v, &rs) || rs.kind == RopeSpec::Keep) {
                RAD_ERR("--rope-scaling takes yarn:FACTOR (above 1) with an optional :ORIG trained "
                        "length, or none -- got '%s'", v);
                return RAD_E_INVAL;
            }
            c->rope_scaling = v;
        }
        /* The two pools are always sized by the budget resolver (core/mem/vram_budget.h); point
         * old launch lines at the flags that move the split. */
        else if (S("--vram-weights-mib") || S("--vram-kv-mib")) {
            RAD_ERR("%s was removed: the card is split between the weights and the cache from what "
                    "the model needs. --gpu-headroom-mib takes card space away, --max-model-len and "
                    "--max-num-seqs bound the cache, and --host-pool-mib decides which experts stay "
                    "on the card", a.c_str());
            return RAD_E_INVAL;
        }
        /* --gpu-headroom is accepted as an alias of --gpu-headroom-mib. */
        else if (S("--gpu-headroom-mib") || S("--gpu-headroom"))
                                               { if (!num(0, INT64_MAX, &c->gpu_headroom_mib)) return RAD_E_INVAL; }
        else if (S("--host-headroom-mib"))      { if (!num(0, INT64_MAX, &c->host_headroom_mib)) return RAD_E_INVAL; }
        else if (S("--kv-cache-dtype")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "bf16") && std::strcmp(v, "fp8") && std::strcmp(v, "fp8_v4") &&
                std::strcmp(v, "k6_v4")) {
                RAD_ERR("--kv-cache-dtype takes bf16, fp8, fp8_v4 or k6_v4, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->kv_cache_dtype = v;
        }
        else if (S("--host-pool-mib"))         { if (!num(0, INT64_MAX, &c->host_pool_mib)) return RAD_E_INVAL; }
        else if (S("--embedding-placement") || S("--ngram-placement")) {
            const bool emb = S("--embedding-placement");
            if (!(v = need(i))) return RAD_E_INVAL;
            TablePlacement p;
            /* The embedding has two tiers: its lookup writes device activations, so there is no
             * disk tier, and with ram the default there is nothing for auto to choose. */
            if (emb && (!parse_table_placement(v, &p) ||
                        (p != TablePlacement::Ram && p != TablePlacement::Vram))) {
                if (parse_table_placement(v, &p) && p == TablePlacement::Auto)
                    RAD_ERR("--embedding-placement auto was removed: ram, the default, holds the "
                            "table in pinned host memory, and vram keeps it on the card");
                else
                    RAD_ERR("--embedding-placement takes ram or vram, not '%s'", v);
                return RAD_E_INVAL;
            }
            if (!emb && !parse_table_placement(v, &p)) {
                RAD_ERR("%s takes auto, vram, ram or disk, not '%s'", a.c_str(), v);
                return RAD_E_INVAL;
            }
            (emb ? c->embedding_placement : c->ngram_placement) = p;
            if (emb) c->embedding_placement_given = true;
        }
        else if (S("--vision-tower-placement")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            TablePlacement p;
            /* Two tiers, as for the embedding: an encoder writes device activations. */
            if (!parse_table_placement(v, &p) || (p != TablePlacement::Ram && p != TablePlacement::Vram)) {
                RAD_ERR("--vision-tower-placement takes vram or ram, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->tower_placement = p;
        }
        else if (S("--weights-disk-tier"))     { c->weights_disk_tier = true; }
        else if (S("--prefix-cache-mib"))      { if (!num(0, INT32_MAX, &c->prefix_cache_mib)) return RAD_E_INVAL; }
        else if (S("--prefix-cache-ttl"))      { if (!num(0, INT32_MAX, &c->prefix_cache_ttl)) return RAD_E_INVAL; }
        else if (S("--memory-archives"))       { if (!num(0, 4096, &c->memory_archives)) return RAD_E_INVAL; }
        else if (S("--memory-dir"))            { if (!(v = need(i))) return RAD_E_INVAL; c->memory_dir = v; }
        else if (S("--lora")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const char* eq = std::strchr(v, '=');
            if (!eq || eq == v || !eq[1]) {
                RAD_ERR("--lora wants NAME=PATH, got '%s'", v);
                return RAD_E_INVAL;
            }
            c->lora.adapters.emplace_back(std::string(v, eq), std::string(eq + 1));
        }
        else if (S("--lora-dir"))              { if (!(v = need(i))) return RAD_E_INVAL; c->lora.dir = v; }
        else if (S("--max-loras"))             { if (!num(0, 64, &c->lora.max_loras)) return RAD_E_INVAL; }
        else if (S("--max-lora-rank"))         { if (!num(1, 512, &c->lora.max_rank)) return RAD_E_INVAL; }
        else if (S("--lora-target-modules")) {
            /* Module leaf names, as PEFT's target_modules lists them: a dotted path or a pattern
             * would match no leaf and target nothing without a word. */
            if (!(v = need(i))) return RAD_E_INVAL;
            std::set<std::string> leaves;
            std::string cur;
            for (const char* q = v;; ++q) {
                if (*q == ',' || !*q) {
                    if (!cur.empty()) leaves.insert(cur);
                    cur.clear();
                    if (!*q) break;
                } else if (std::isalnum((unsigned char)*q) || *q == '_') {
                    cur += *q;
                } else if (*q != ' ') {
                    RAD_ERR("--lora-target-modules takes module leaf names (q_proj,k_proj,...); "
                            "'%s' holds '%c'", v, *q);
                    return RAD_E_INVAL;
                }
            }
            if (leaves.empty()) {
                RAD_ERR("--lora-target-modules names no module");
                return RAD_E_INVAL;
            }
            c->lora.targets.assign(leaves.begin(), leaves.end());
        }
        /* The prefix cache has no disk tier; the removed flags name their replacements. */
        else if (S("--prefix-cache-host-mib")) {
            RAD_ERR("--prefix-cache-host-mib was renamed: use --prefix-cache-mib");
            return RAD_E_INVAL;
        }
        else if (S("--prefix-cache-dir") || S("--prefix-cache-disk-mib")) {
            RAD_ERR("%s was removed: the prefix cache is kept in VRAM and host memory only "
                    "(--prefix-cache-mib), and --prefix-cache-ttl says how long", a.c_str());
            return RAD_E_INVAL;
        }
        else if (S("--placement")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The planner's names, checked here so a typo stops the parse rather than startup
             * after the container is opened. */
            Strategy st;
            if (strategy_parse(v, &st) < 0) {
                RAD_ERR("--placement takes all_vram, layer_offload, expert_tiered or auto, not "
                        "'%s'", v);
                return RAD_E_INVAL;
            }
            c->placement = v;
        }
        else if (S("--control-vector")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "off") && std::strcmp(v, "on") && std::strcmp(v, "disabled")) {
                RAD_ERR("--control-vector takes off, on or disabled -- got '%s'", v);
                return RAD_E_INVAL;
            }
            c->control_vector = v;
        }
        else if (S("--deterministic"))         c->deterministic = true;
        else if (S("--checkpoint-interval"))   { if (!num(0, INT64_MAX, &c->checkpoint_interval)) return RAD_E_INVAL; }
        else if (S("--prefill-span"))          { if (!num(1, INT32_MAX, &c->prefill_span)) return RAD_E_INVAL; }
        else if (S("--checkpoint-slots")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) {
                c->checkpoint_slots = -1;
            } else if (!*v || std::strspn(v, "0123456789") != std::strlen(v)) {
                RAD_ERR("--checkpoint-slots takes auto or a whole number, not '%s'", v);
                return RAD_E_INVAL;
            } else if (!parse_int(a.c_str(), v, 0, INT64_MAX, &c->checkpoint_slots)) {
                return RAD_E_INVAL;
            }
        }
        else if (S("--no-prefix-cache"))       c->prefix_cache = false;
        else if (S("--checkpoint-policy")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* Validated, so an unknown name does not silently fall back to the default. */
            if (std::strcmp(v, "geometric-backoff") && std::strcmp(v, "tip-only") &&
                std::strcmp(v, "keep-all")) {
                RAD_ERR("--checkpoint-policy '%s' is not one of geometric-backoff, tip-only, "
                        "keep-all", v);
                return RAD_E_INVAL;
            }
            c->checkpoint_policy = v;
        }
        else if (S("--num-speculative-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* "auto" is -1, resolved once the container is open from the architecture plugin
             * (rad_arch_probe): a serial head's measured operating point, or a block drafter's
             * trained block length. Any negative depth means auto downstream, so only the word
             * is accepted. */
            int64_t n = -1;
            if (std::strcmp(v, "auto") && !parse_int(a.c_str(), v, 0, INT32_MAX, &n))
                return RAD_E_INVAL;
            c->n_spec = (int)n;
        }
        else if (S("--draft-window")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "fixed"))    c->draft_window = DraftWindow::Fixed;
            else if (!std::strcmp(v, "adaptive")) c->draft_window = DraftWindow::Adaptive;
            else {
                RAD_ERR("--draft-window takes fixed or adaptive, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--draft-model")) {
            /* Refused, not ignored: separate draft models are not supported (spec §10), and the
             * drafter comes from the container (rad-convert --draft-model merges one in; an MTP
             * head inside the checkpoint needs no flag). Kept so vLLM users get a message saying
             * so. */
            if (!(v = need(i))) return RAD_E_INVAL;
            RAD_ERR("--draft-model is not implemented. This engine loads its drafter from the "
                    "model container, not from a second one: merge it at conversion time with "
                    "`rad-convert --draft-model DIR` and pass the container to --model. "
                    "An MTP head that ships inside the target checkpoint needs neither flag.");
            return RAD_E_INVAL;
        }
        else if (S("--host"))                  { if (!(v = need(i))) return RAD_E_INVAL; c->host = v; }
        else if (S("--port"))                  { if (!num(0, 65535, &c->port)) return RAD_E_INVAL; }
        else if (S("--api-key"))               { if (!(v = need(i))) return RAD_E_INVAL; c->api_key = v; }
        else if (S("--served-model-name")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!*v) { RAD_ERR("--served-model-name takes a non-empty name"); return RAD_E_INVAL; }
            c->served_model_name = v;
        }
        else if (S("--mm-max-patches")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) c->mm_max_patches = -1;
            else if (!parse_int(a.c_str(), v, 0, INT64_MAX, &c->mm_max_patches)) return RAD_E_INVAL;
        }
        else if (S("--reasoning-effort"))      { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->reasoning_effort = v; }
        else if (S("--reasoning-effort-map"))  {
            if (!(v = need(i))) return RAD_E_INVAL;
            const std::string all = v;
            size_t at = 0;
            while (at <= all.size()) {
                const size_t end = std::min(all.find(',', at), all.size());
                const std::string pair = all.substr(at, end - at);
                const size_t eq = pair.find('=');
                if (eq == std::string::npos || eq == 0 || eq + 1 == pair.size() ||
                    pair.find('=', eq + 1) != std::string::npos) {
                    RAD_ERR("--reasoning-effort-map takes FROM=TO pairs separated by commas, "
                            "not '%s'", pair.c_str());
                    return RAD_E_INVAL;
                }
                c->reasoning_effort_map[pair.substr(0, eq)] = pair.substr(eq + 1);
                at = end + 1;
            }
        }
        else if (S("--thinking-token-budget")) { if (!num(0, INT32_MAX, &c->thinking_token_budget)) return RAD_E_INVAL; }
        else if (S("--generation-config"))     { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->generation_config = v; }
        else if (S("--override-chat-template")) { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->override_chat_template = v; }
        /* The range check refuses negatives, which these fields use for "not set". */
        else if (S("--temp") || S("--top-p") || S("--min-p")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const char* field = S("--temp") ? "temperature" : S("--top-p") ? "top_p" : "min_p";
            double d = 0;
            if (!sampler_num(a.c_str(), field, v, &d)) return RAD_E_INVAL;
            (S("--temp") ? c->sample_temp : S("--top-p") ? c->sample_top_p : c->sample_min_p) = (float)d;
        }
        else if (S("--top-k"))                 { if (!(v = need(i))) return RAD_E_INVAL;
                                                 if (!sampler_int("--top-k", "top_k", v, &c->sample_top_k))
                                                     return RAD_E_INVAL; }
        else if (S("--typical-p") || S("--presence-penalty") || S("--frequency-penalty") ||
                 S("--repetition-penalty") || S("--dry-multiplier") || S("--dry-base") ||
                 S("--xtc-probability") || S("--xtc-threshold")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            struct F { const char* flag; const char* field; std::optional<float> Config::* m; };
            static const F kF[] = {
                { "--typical-p",          "typical_p",          &Config::sample_typical_p },
                { "--presence-penalty",   "presence_penalty",   &Config::sample_pres_penalty },
                { "--frequency-penalty",  "frequency_penalty",  &Config::sample_freq_penalty },
                { "--repetition-penalty", "repetition_penalty", &Config::sample_rep_penalty },
                { "--dry-multiplier",     "dry_multiplier",     &Config::sample_dry_multiplier },
                { "--dry-base",           "dry_base",           &Config::sample_dry_base },
                { "--xtc-probability",    "xtc_probability",    &Config::sample_xtc_probability },
                { "--xtc-threshold",      "xtc_threshold",      &Config::sample_xtc_threshold },
            };
            for (const F& f : kF) {
                if (!S(f.flag)) continue;
                double d = 0;
                if (!sampler_num(f.flag, f.field, v, &d)) return RAD_E_INVAL;
                c->*f.m = (float)d;
            }
        }
        else if (S("--repeat-last-n") || S("--dry-allowed-length") || S("--dry-penalty-last-n")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const char* field = S("--repeat-last-n") ? "repeat_last_n"
                              : S("--dry-allowed-length") ? "dry_allowed_length"
                                                          : "dry_penalty_last_n";
            int n = 0;
            if (!sampler_int(a.c_str(), field, v, &n)) return RAD_E_INVAL;
            /* A DRY request with an allowed length below 1 is refused at admission, so such a
             * default would refuse every request that turns DRY on without stating one. */
            if (S("--dry-allowed-length") && n < 1) {
                RAD_ERR("--dry-allowed-length must be >= 1, not %d", n);
                return RAD_E_INVAL;
            }
            (S("--repeat-last-n") ? c->sample_penalty_last_n
             : S("--dry-allowed-length") ? c->sample_dry_allowed_length
                                         : c->sample_dry_penalty_last_n) = n;
        }
        else if (S("--dry-sequence-breakers")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            const nlohmann::json j = nlohmann::json::parse(v, nullptr, false);
            bool strings = j.is_array();
            if (strings) for (const auto& e : j) strings = strings && e.is_string();
            if (!strings) {
                RAD_ERR("--dry-sequence-breakers takes a JSON array of strings, such as "
                        "'[\"\\n\", \":\"]', not '%s'", v);
                return RAD_E_INVAL;
            }
            /* The sampler refuses more (sample/host_ref.h), so the default must not exceed it. */
            if (j.size() > kDryMaxBreakers) {
                RAD_ERR("--dry-sequence-breakers holds %zu strings; the sampler takes at most %zu",
                        j.size(), kDryMaxBreakers);
                return RAD_E_INVAL;
            }
            c->sample_dry_seq_breakers = j.get<std::vector<std::string>>();
        }
        else if (S("--default-max-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) c->default_max_tokens = 0;
            else if (!parse_int(a.c_str(), v, 1, INT32_MAX, &c->default_max_tokens)) {
                RAD_ERR("--default-max-tokens takes a token count >= 1, or auto");
                return RAD_E_INVAL;
            }
        }
        else if (S("--max-tokens-cap"))        { if (!num(0, INT32_MAX, &c->max_tokens_cap)) return RAD_E_INVAL; }
        else if (S("--max-n"))                 { if (!num(1, INT32_MAX, &c->max_n)) return RAD_E_INVAL; }
        else if (S("--max-queued-requests"))   { if (!num(1, INT32_MAX, &c->max_queued_requests)) return RAD_E_INVAL; }
        else if (S("--max-stop-strings"))      { if (!num(1, INT32_MAX, &c->max_stop_strings)) return RAD_E_INVAL; }
        else if (S("--max-stop-bytes"))        { if (!num(1, INT32_MAX, &c->max_stop_bytes)) return RAD_E_INVAL; }
        else if (S("--chat-template-kwargs"))  { if (!(v = need(i)) ||
                                                     !parse_template_kwargs(v, &c->chat_template_kwargs))
                                                     return RAD_E_INVAL; }
        else if (S("--reasoning-format")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The reply reader only distinguishes lifting reasoning out of content or not. */
            if (std::strcmp(v, "auto") && std::strcmp(v, "none")) {
                RAD_ERR("--reasoning-format takes auto or none, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->reasoning_format = v;
        }
        else if (S("--http-threads") || S("--read-timeout") || S("--write-timeout") ||
                 S("--keep-alive-timeout") || S("--retry-after")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The transport keeps at least four workers (server/http.cpp), so fewer is refused. */
            const int64_t lo = S("--http-threads") ? 4 : S("--retry-after") ? 0 : 1;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, lo, INT32_MAX, &n)) return RAD_E_INVAL;
            (S("--http-threads")    ? c->http_threads
             : S("--read-timeout")  ? c->http_read_timeout_s
             : S("--write-timeout") ? c->http_write_timeout_s
             : S("--retry-after")   ? c->retry_after_s
                                    : c->http_keep_alive_timeout_s) = (int)n;
        }
        else if (S("--max-body-mib"))          { if (!num(1, (int64_t)1 << 20, &c->http_max_body_mib)) return RAD_E_INVAL; }
        else if (S("--no-cors"))               c->cors = false;
        else if (S("--image-min-pixels") || S("--image-max-pixels") || S("--video-min-pixels") ||
                 S("--video-max-pixels") || S("--max-source-pixels")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, 1, INT64_MAX, &n)) return RAD_E_INVAL;
            (S("--image-min-pixels")   ? c->image_min_pixels
             : S("--image-max-pixels") ? c->image_max_pixels
             : S("--video-min-pixels") ? c->video_min_pixels
             : S("--video-max-pixels") ? c->video_max_pixels
                                       : c->max_source_pixels) = n;
        }
        else if (S("--video-fps")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            double d = 0;
            if (!parse_num(a.c_str(), v, &d)) return RAD_E_INVAL;
            if (!(d > 0.0)) { RAD_ERR("--video-fps must be > 0, not '%s'", v); return RAD_E_INVAL; }
            c->video_fps = d;
        }
        else if (S("--image-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, 1, INT32_MAX, &n)) return RAD_E_INVAL;
            c->image_tokens = (int32_t)n;
        }
        else if (S("--video-min-frames") || S("--video-max-frames") || S("--video-max-frame-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, S("--video-max-frame-tokens") ? 0 : 1, INT32_MAX, &n))
                return RAD_E_INVAL;
            (S("--video-min-frames")   ? c->video_min_frames
             : S("--video-max-frames") ? c->video_max_frames
                                       : c->video_max_frame_tokens) = (int32_t)n;
        }
        else if (S("--kld-record"))            { if (!(v = need(i))) return RAD_E_INVAL; c->kld_record = v; }
        else if (S("--kld-ref"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kld_ref = v; }
        else if (S("--kld-corpus"))            { if (!(v = need(i))) return RAD_E_INVAL; c->kld_corpus = v; }
        else if (S("--kld-out"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kld_out = v; }
        else if (S("--kld-lora"))              { if (!(v = need(i))) return RAD_E_INVAL; c->kld_lora = v; }
        else if (S("--kld-lora-sources"))      { if (!(v = need(i))) return RAD_E_INVAL; c->kld_lora_sources = v; }
        else if (S("--debug-graph"))           { c->debug_graph = true; c->debug_selection = true; }
        else if (S("--debug-selection"))       c->debug_selection = true;
        else if (S("--debug-placement"))       c->debug_placement = true;
        else if (S("--debug-accept-reference-kernels")) c->accept_reference_kernels = true;
        else if (S("--profile-ops"))           c->profile_ops = true;
        else if (S("--live"))                  c->live_view = true;
        else {
            RAD_ERR("unknown option '%s'", a.c_str());
            return RAD_E_INVAL;
        }
        /* Record the option's source for the startup report (core/diag/diag.h), under the
         * table's spelling so aliases report as their option. */
        c->given[S("--gpu-headroom") ? "--gpu-headroom-mib" : a] = "command line";
    }

    /* Both variables are read as the tools read them (rad_home() and rad_hierarchy_from_env() in
     * tools/iface.cpp), so a plugin validated with rad-kbench or rad-tune serves the same way. Set
     * but empty is unset, as for RADIANCE_KERNELS: an empty home would load no plugin at all. */
    if (c->radiance_home.empty()) {
        const char* h = getenv("RADIANCE_HOME");
        const bool set = h && *h;
        c->radiance_home = set ? h : RAD_DEFAULT_HOME;
        if (set) c->given["--radiance-home"] = "environment (RADIANCE_HOME)";
    }
    if (c->kernel_hierarchy.empty()) {
        const char* e = getenv("RADIANCE_KERNELS");
        if (e && *e) {
            c->kernel_hierarchy = rad_split_kernel_list(e);
            c->given["--kernels"] = "environment (RADIANCE_KERNELS)";
        }
    }
    /* No built-in default, deliberately: a named library would be wrong on other cards and would
     * rank the installed one behind the reference implementation. An empty hierarchy loads every
     * library found, ordered with the reference last (core/plugin/loader.cpp). A list loads only
     * the named libraries, in its order. */
    if (c->model.empty()) {
        RAD_ERR("--model is required");
        return RAD_E_INVAL;
    }

    /* KL mode: exactly one direction, each with the inputs it reads. */
    if (!c->kld_record.empty() && !c->kld_ref.empty()) {
        RAD_ERR("--kld-record writes a reference and --kld-ref scores against one; give one");
        return RAD_E_INVAL;
    }
    if (!c->kld_record.empty() && c->kld_corpus.empty()) {
        RAD_ERR("--kld-record needs --kld-corpus: the reference is recorded over a corpus");
        return RAD_E_INVAL;
    }
    if (!c->kld_ref.empty() && !c->kld_corpus.empty()) {
        RAD_ERR("--kld-ref reads its tokens from the reference, which recorded them; drop "
                "--kld-corpus");
        return RAD_E_INVAL;
    }
    if (!c->kld_out.empty() && c->kld_ref.empty()) {
        RAD_ERR("--kld-out is the --kld-ref report");
        return RAD_E_INVAL;
    }

    /* Runtime LoRA: the slot flags need slots, the named adapters must fit them, and KL mode's
     * adapter is one of them. */
    {
        const LoraConfig& L = c->lora;
        std::set<std::string> names;
        for (const auto& [name, path] : L.adapters)
            if (!names.insert(name).second) {
                RAD_ERR("--lora names the adapter '%s' twice", name.c_str());
                return RAD_E_INVAL;
            }
        const bool rank_given = c->given.count("--max-lora-rank") != 0;
        if (!L.enabled() && (rank_given || !L.targets.empty())) {
            RAD_ERR("%s sizes runtime LoRA slots, and there are none: give --lora or --max-loras",
                    rank_given ? "--max-lora-rank" : "--lora-target-modules");
            return RAD_E_INVAL;
        }
        if (!L.dir.empty()) {
            struct stat sb{};
            if (stat(L.dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
                RAD_ERR("--lora-dir '%s' is not a directory", L.dir.c_str());
                return RAD_E_INVAL;
            }
        }
        if (!c->kld_lora.empty() && !c->kld()) {
            RAD_ERR("--kld-lora scores through an adapter in KL mode; give --kld-record or "
                    "--kld-ref");
            return RAD_E_INVAL;
        }
        if (!c->kld_lora.empty() && !names.count(c->kld_lora)) {
            RAD_ERR("--kld-lora '%s' is not one of the adapters --lora names", c->kld_lora.c_str());
            return RAD_E_INVAL;
        }
        if (!c->kld_lora_sources.empty() && c->kld_lora.empty()) {
            RAD_ERR("--kld-lora-sources picks the documents that run through --kld-lora's adapter; "
                    "give --kld-lora");
            return RAD_E_INVAL;
        }
    }

    /* --reasoning-effort is a template variable ("none" also sets enable_thinking false), so
     * either key in --chat-template-kwargs as well would give two defaults for one setting. */
    if (!c->reasoning_effort.empty() &&
        (c->chat_template_kwargs.count("reasoning_effort") ||
         c->chat_template_kwargs.count("enable_thinking"))) {
        RAD_ERR("--reasoning-effort and a reasoning_effort or enable_thinking key in "
                "--chat-template-kwargs both set the default for thinking; give one");
        return RAD_E_INVAL;
    }
    if (c->default_max_tokens > 0 && c->max_tokens_cap > 0 &&
        c->default_max_tokens > c->max_tokens_cap) {
        RAD_ERR("--default-max-tokens %lld is above --max-tokens-cap %lld, so no request could be "
                "served at the default", (long long)c->default_max_tokens,
                (long long)c->max_tokens_cap);
        return RAD_E_INVAL;
    }
    /* A request's choices must all fit the queue, else it is always 429. The defaults always fit,
     * so this is checked only when one was given. */
    if (c->max_n > 0 || c->max_queued_requests > 0) {
        const int64_t n = c->max_n > 0 ? c->max_n : std::max<int64_t>(8, c->max_seqs);
        const int64_t q = c->max_queued_requests > 0 ? c->max_queued_requests : c->max_seqs * 8;
        if (n > q) {
            RAD_ERR("--max-n %lld is above the %lld requests the queue admits "
                    "(--max-queued-requests), so a request asking for it could never be admitted",
                    (long long)n, (long long)q);
            return RAD_E_INVAL;
        }
    }
    if (c->image_min_pixels && c->image_max_pixels && *c->image_min_pixels > *c->image_max_pixels) {
        RAD_ERR("--image-min-pixels is above --image-max-pixels"); return RAD_E_INVAL; }
    if (c->video_min_pixels && c->video_max_pixels && *c->video_min_pixels > *c->video_max_pixels) {
        RAD_ERR("--video-min-pixels is above --video-max-pixels"); return RAD_E_INVAL; }
    if (c->video_min_frames && c->video_max_frames && *c->video_min_frames > *c->video_max_frames) {
        RAD_ERR("--video-min-frames is above --video-max-frames"); return RAD_E_INVAL; }

    return RAD_OK;
}

/* ------------------------------------------------------------------ the startup report's rows */
namespace {

std::string onoff(bool b) { return b ? "on" : "off"; }
std::string num(int64_t v) { return std::to_string(v); }
std::string or_none(const std::string& s, const char* none) { return s.empty() ? std::string(none) : s; }
std::string fnum(double v) {
    char b[48];
    snprintf(b, sizeof b, "%g", v);
    return b;
}
template <class T>
std::string opt(const std::optional<T>& v, const char* unset) {
    if (!v) return unset;
    if constexpr (std::is_floating_point_v<T>) return fnum((double)*v);
    else return std::to_string(*v);
}
std::string joined(const std::map<std::string, std::string>& m, const char* sep) {
    std::string s;
    for (const auto& [k, v] : m) s += (s.empty() ? "" : ",") + k + sep + v;
    return s.empty() ? "none" : s;
}

using Show = std::string (*)(const Config&);
struct ShowRow { const char* flag; Show show; };

/* How each option's value is shown, one entry per option of the table above. config_rows walks
 * the table, so a missing entry shows "(not shown)" and fails diag_test. Sentinels for "resolved
 * at startup" (-1, 0, empty) are shown by meaning; the engine's decisions replace them. */
const ShowRow kShow[] = {
    { "--model",              [](const Config& c) { return c.model; } },
    { "--download-dir",       [](const Config& c) { return or_none(c.download_dir, "the XDG cache (~/.cache/radiance/hf)"); } },
    { "--radiance-home",      [](const Config& c) { return c.radiance_home; } },
    { "--kernels",            [](const Config& c) {
          std::string s;
          for (const std::string& k : c.kernel_hierarchy) s += (s.empty() ? "" : ",") + k;
          return or_none(s, "every library found"); } },
    { "--tp",                 [](const Config& c) { return num(c.tp); } },
    { "--tp-wire",            [](const Config& c) { return std::string(c.tp_wire_exact ? "exact" : "wht6"); } },
    { "--tp-wire-min-kb",     [](const Config& c) { return num(c.tp_wire_min_kb); } },
    { "--p2p",                [](const Config& c) {
          return std::string(c.p2p == RAD_P2P_ON ? "on" : c.p2p == RAD_P2P_OFF ? "off" : "auto"); } },
    { "--max-num-batched-tokens", [](const Config& c) { return num(c.max_tok); } },
    { "--busy-prefill-chunk", [](const Config& c) {
          return c.busy_prefill_chunk < 0 ? std::string("auto")
               : c.busy_prefill_chunk == 0 ? std::string("off") : num(c.busy_prefill_chunk); } },
    { "--prefill-interleave", [](const Config& c) {
          return c.prefill_interleave > 0 ? num(c.prefill_interleave) : std::string("off"); } },
    { "--max-num-seqs",       [](const Config& c) { return num(c.max_seqs); } },
    { "--max-model-len",      [](const Config& c) {
          return c.max_ctx ? num(c.max_ctx) : std::string("0 (the model's training context)"); } },
    { "--rope-scaling",       [](const Config& c) { return or_none(c.rope_scaling, "the checkpoint's own"); } },
    { "--gpu-headroom-mib",   [](const Config& c) { return num(c.gpu_headroom_mib); } },
    { "--kv-cache-dtype",     [](const Config& c) { return c.kv_cache_dtype; } },
    { "--host-pool-mib",      [](const Config& c) { return num(c.host_pool_mib); } },
    { "--host-headroom-mib",  [](const Config& c) { return num(c.host_headroom_mib); } },
    { "--embedding-placement", [](const Config& c) { return std::string(table_placement_name(c.embedding_placement)); } },
    { "--ngram-placement",    [](const Config& c) { return std::string(table_placement_name(c.ngram_placement)); } },
    { "--vision-tower-placement", [](const Config& c) { return std::string(table_placement_name(c.tower_placement)); } },
    { "--weights-disk-tier",  [](const Config& c) { return onoff(c.weights_disk_tier); } },
    { "--prefix-cache-mib",   [](const Config& c) {
          return c.prefix_cache_mib ? num(c.prefix_cache_mib) : std::string("0 (off)"); } },
    { "--prefix-cache-ttl",   [](const Config& c) {
          return c.prefix_cache_ttl ? num(c.prefix_cache_ttl) + " s" : std::string("0 (never)"); } },
    { "--memory-archives",    [](const Config& c) { return num(c.memory_archives); } },
    { "--memory-dir",         [](const Config& c) { return or_none(c.memory_dir, "memory only"); } },
    { "--lora",               [](const Config& c) {
          std::string o;
          for (const auto& [n, p] : c.lora.adapters) o += (o.empty() ? "" : ", ") + n + "=" + p;
          return o.empty() ? std::string("none") : o; } },
    { "--lora-dir",           [](const Config& c) {
          return c.lora.dir.empty() ? std::string("none") : c.lora.dir; } },
    { "--max-loras",          [](const Config& c) {
          return c.lora.enabled() ? num(c.lora.slots()) : std::string("0 (off)"); } },
    { "--max-lora-rank",      [](const Config& c) {
          return c.lora.max_rank > 0 ? num(c.lora.max_rank) : std::string("auto (16 text, 64 generator)"); } },
    { "--lora-target-modules", [](const Config& c) {
          std::string o;
          for (const std::string& t : c.lora.targets) o += (o.empty() ? "" : ",") + t;
          return o.empty() ? std::string(c.lora.enabled() ? "the adapters' own" : "-") : o; } },
    { "--placement",          [](const Config& c) { return c.placement; } },
    { "--control-vector",     [](const Config& c) { return c.control_vector; } },
    { "--deterministic",      [](const Config& c) { return onoff(c.deterministic); } },
    { "--checkpoint-interval", [](const Config& c) { return num(c.checkpoint_interval); } },
    { "--prefill-span",       [](const Config& c) { return num(c.prefill_span); } },
    { "--checkpoint-slots",   [](const Config& c) {
          return c.checkpoint_slots < 0 ? std::string("auto") : num(c.checkpoint_slots); } },
    { "--no-prefix-cache",    [](const Config& c) {
          return std::string(c.prefix_cache ? "not given: prefix caching on" : "prefix caching off"); } },
    { "--checkpoint-policy",  [](const Config& c) { return c.checkpoint_policy; } },
    { "--num-speculative-tokens", [](const Config& c) { return c.n_spec < 0 ? std::string("auto") : num(c.n_spec); } },
    { "--draft-window",       [](const Config& c) { return std::string(draft_window_name(c.draft_window)); } },
    { "--draft-model",        [](const Config&) { return std::string("not implemented (the container's own drafter)"); } },
    { "--host",               [](const Config& c) { return c.host; } },
    { "--port",               [](const Config& c) { return num(c.port); } },
    /* Never print the secret, only whether it is set. */
    { "--api-key",            [](const Config& c) { return std::string(c.api_key.empty() ? "not set" : "set"); } },
    { "--served-model-name",  [](const Config& c) { return or_none(c.served_model_name, "the container's name"); } },
    { "--mm-max-patches",     [](const Config& c) { return c.mm_max_patches < 0 ? std::string("auto") : num(c.mm_max_patches); } },
    { "--generation-config",  [](const Config& c) { return or_none(c.generation_config, "none"); } },
    { "--override-chat-template", [](const Config& c) { return or_none(c.override_chat_template, "none (the container's)"); } },
    { "--temp",               [](const Config& c) { return c.sample_temp < 0 ? std::string("unset") : fnum(c.sample_temp); } },
    { "--top-k",              [](const Config& c) { return c.sample_top_k < 0 ? std::string("unset") : num(c.sample_top_k); } },
    { "--top-p",              [](const Config& c) { return c.sample_top_p < 0 ? std::string("unset") : fnum(c.sample_top_p); } },
    { "--min-p",              [](const Config& c) { return c.sample_min_p < 0 ? std::string("unset") : fnum(c.sample_min_p); } },
    { "--reasoning-effort",   [](const Config& c) { return or_none(c.reasoning_effort, "the template's default"); } },
    { "--reasoning-effort-map", [](const Config& c) { return joined(c.reasoning_effort_map, "="); } },
    { "--thinking-token-budget", [](const Config& c) {
          return c.thinking_token_budget < 0 ? std::string("none") : num(c.thinking_token_budget); } },
    { "--typical-p",          [](const Config& c) { return opt(c.sample_typical_p, "the request default"); } },
    { "--presence-penalty",   [](const Config& c) { return opt(c.sample_pres_penalty, "the request default"); } },
    { "--frequency-penalty",  [](const Config& c) { return opt(c.sample_freq_penalty, "the request default"); } },
    { "--repetition-penalty", [](const Config& c) { return opt(c.sample_rep_penalty, "the request default"); } },
    { "--repeat-last-n",      [](const Config& c) { return opt(c.sample_penalty_last_n, "the request default"); } },
    { "--dry-multiplier",     [](const Config& c) { return opt(c.sample_dry_multiplier, "the request default"); } },
    { "--dry-base",           [](const Config& c) { return opt(c.sample_dry_base, "the request default"); } },
    { "--dry-allowed-length", [](const Config& c) { return opt(c.sample_dry_allowed_length, "the request default"); } },
    { "--dry-penalty-last-n", [](const Config& c) { return opt(c.sample_dry_penalty_last_n, "the request default"); } },
    { "--dry-sequence-breakers", [](const Config& c) {
          /* Invalid UTF-8 is replaced, not thrown on: these strings come from outside. */
          return c.sample_dry_seq_breakers
                     ? nlohmann::json(*c.sample_dry_seq_breakers)
                           .dump(-1, ' ', false, nlohmann::json::error_handler_t::replace)
                     : std::string("the request default"); } },
    { "--xtc-probability",    [](const Config& c) { return opt(c.sample_xtc_probability, "the request default"); } },
    { "--xtc-threshold",      [](const Config& c) { return opt(c.sample_xtc_threshold, "the request default"); } },
    { "--default-max-tokens", [](const Config& c) {
          return c.default_max_tokens > 0 ? num(c.default_max_tokens)
                                          : std::string("auto (what the context leaves after the prompt)"); } },
    { "--max-tokens-cap",     [](const Config& c) { return c.max_tokens_cap > 0 ? num(c.max_tokens_cap) : std::string("none"); } },
    { "--max-n",              [](const Config& c) {
          return c.max_n > 0 ? num(c.max_n)
                             : std::string("auto (the larger of 8 and --max-num-seqs)"); } },
    { "--max-queued-requests", [](const Config& c) {
          return c.max_queued_requests > 0 ? num(c.max_queued_requests)
                                           : std::string("auto (8 x --max-num-seqs)"); } },
    { "--max-stop-strings",   [](const Config& c) { return num(c.max_stop_strings); } },
    { "--max-stop-bytes",     [](const Config& c) { return num(c.max_stop_bytes); } },
    { "--chat-template-kwargs", [](const Config& c) { return joined(c.chat_template_kwargs, "="); } },
    { "--reasoning-format",   [](const Config& c) { return c.reasoning_format; } },
    { "--http-threads",       [](const Config& c) {
          return c.http_threads > 0 ? num(c.http_threads)
                                    : std::string("auto (one a hardware thread, at least 4)"); } },
    { "--read-timeout",       [](const Config& c) { return num(c.http_read_timeout_s) + " s"; } },
    { "--write-timeout",      [](const Config& c) { return num(c.http_write_timeout_s) + " s"; } },
    { "--keep-alive-timeout", [](const Config& c) { return num(c.http_keep_alive_timeout_s) + " s"; } },
    { "--max-body-mib",       [](const Config& c) { return num(c.http_max_body_mib); } },
    { "--retry-after",        [](const Config& c) { return num(c.retry_after_s) + " s"; } },
    { "--no-cors",            [](const Config& c) {
          return std::string(c.cors ? "not given: CORS headers sent" : "no CORS headers"); } },
    { "--image-min-pixels",   [](const Config& c) { return opt(c.image_min_pixels, "the container's"); } },
    { "--image-max-pixels",   [](const Config& c) { return opt(c.image_max_pixels, "the container's"); } },
    { "--video-min-pixels",   [](const Config& c) { return opt(c.video_min_pixels, "the container's"); } },
    { "--video-max-pixels",   [](const Config& c) { return opt(c.video_max_pixels, "the container's"); } },
    { "--video-fps",          [](const Config& c) { return opt(c.video_fps, "the container's"); } },
    { "--video-min-frames",   [](const Config& c) { return opt(c.video_min_frames, "the container's"); } },
    { "--video-max-frames",   [](const Config& c) { return opt(c.video_max_frames, "the container's"); } },
    { "--video-max-frame-tokens", [](const Config& c) { return opt(c.video_max_frame_tokens, "the container's"); } },
    { "--image-tokens",       [](const Config& c) { return opt(c.image_tokens, "the container's"); } },
    { "--max-source-pixels",  [](const Config& c) { return opt(c.max_source_pixels, "67108864"); } },
    { "--kld-record",         [](const Config& c) { return or_none(c.kld_record, "none"); } },
    { "--kld-ref",            [](const Config& c) { return or_none(c.kld_ref, "none"); } },
    { "--kld-corpus",         [](const Config& c) { return or_none(c.kld_corpus, "none"); } },
    { "--kld-out",            [](const Config& c) { return or_none(c.kld_out, "none"); } },
    { "--kld-lora",           [](const Config& c) { return or_none(c.kld_lora, "none (the base)"); } },
    { "--kld-lora-sources",   [](const Config& c) { return or_none(c.kld_lora_sources, "every document"); } },
    { "--debug-graph",        [](const Config& c) { return onoff(c.debug_graph); } },
    { "--debug-selection",    [](const Config& c) { return onoff(c.debug_selection); } },
    { "--debug-placement",    [](const Config& c) { return onoff(c.debug_placement); } },
    { "--debug-accept-reference-kernels", [](const Config& c) { return onoff(c.accept_reference_kernels); } },
    { "--profile-ops",        [](const Config& c) { return onoff(c.profile_ops); } },
    { "--live",               [](const Config& c) { return onoff(c.live_view); } },
    { "-v",                   [](const Config& c) { return onoff(c.given.count("-v") > 0); } },
    { "-vv",                  [](const Config& c) { return onoff(c.given.count("-vv") > 0); } },
};

}  /* namespace */

namespace diag {

std::vector<ConfigRow> config_rows(const Config& c) {
    std::vector<ConfigRow> out;
    for (const Flag& f : g_flags) {
        /* --help is not a setting. */
        if (!std::strcmp(f.name, "--help")) continue;
        ConfigRow r;
        r.flag = f.name;
        r.value = "(not shown)";
        for (const ShowRow& s : kShow)
            if (!std::strcmp(s.flag, f.name)) { r.value = s.show(c); break; }
        auto g = c.given.find(f.name);
        r.source = g != c.given.end() ? g->second : "default";
        out.push_back(std::move(r));
    }
    return out;
}

std::vector<std::string> config_option_names() {
    std::vector<std::string> out;
    for (const Flag& f : g_flags)
        if (std::strcmp(f.name, "--help")) out.push_back(f.name);
    return out;
}

}  /* namespace diag */

}  /* namespace rad */
