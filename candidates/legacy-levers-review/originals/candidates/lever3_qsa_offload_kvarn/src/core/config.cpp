/* config.cpp -- the command line. Every pool is budgeted explicitly: no profiling pass, no
 * utilisation fraction, no inference about what the workload will be. The trade between KV blocks
 * and resident weights is real and workload-dependent -- a long-context low-concurrency server and
 * a short-context high-concurrency one want opposite splits -- but it is the operator's to make.
 * An engine that quietly re-optimises the split is an engine whose benchmarks do not reproduce
 * (spec §6).
 */
#include "rad_core.h"
#include "runtime/rad_ridgefill_wrapper.h"
#include "sample/host_ref.h"
#include "server/oai.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace rad {

struct Flag {
    const char* name;
    const char* arg;     /* null for a boolean */
    const char* help;
};

static const Flag g_flags[] = {
    { "--model",              "PATH",   "the .rad container, or a checkpoint served as it stands" },
    { "--radiance-home",      "DIRS",   "where kernels/, architectures/ and quantizers/ live; several\n"
                                        "homes separated by ':' are searched in order, the first winning" },
    { "--kernels",            "LIST",   "kernel plugin hierarchy, comma-separated, first wins" },

    { "--tp",                 "N",      "tensor-parallel ranks" },
    { "--tp-wire",            "MODE",   "cross-rank all-reduce payload: exact (default) or wht6, a\n"
                                        "lossy wire: rotated 6-bit at 2 ranks, tiered int8 at 4,\n"
                                        "none at 8" },
    { "--tp-wire-min-kb",     "N",      "smallest all-reduce, in KiB, the lossy wire is used for" },
    { "--max-num-batched-tokens", "N",  "the largest token count one step may carry" },
    { "--max-num-seqs",       "N",      "the largest number of sequences one step may carry" },
    { "--max-model-len",      "N",      "context bound; 0 takes the model's training context" },

    { "--gpu-headroom-mib",   "N",      "VRAM left unclaimed on every card (default 96)" },
    { "--prefill-stage-mib",  "N",      "prefill staging buffer size in MiB per buffer (default 280)" },
    /* THE FLAG THAT DECIDES WHETHER MULTI-TURN WORKS, which is more than the split it names.
     * The KV side is not just concurrency: prefix caching holds a finished sequence's blocks
     * so the next turn can reuse them, so when the pool cannot hold the LIVE SESSIONS the
     * cache drops prefixes the next turn wanted and that turn re-prefills its whole context.
     * At long contexts that is two orders of magnitude of time-to-first-token, bought back for
     * a fraction of a percent of the decode step. See core/mem/prefix.h. */
    { "--expert-vs-cache-ratio", "R",   "weights vs KV split of what is claimed (default "
                                          "0.75). The KV side also holds cached prefixes, so "
                                          "too little of it makes every agent turn re-prefill" },
    { "--vram-weights-mib",   "N",      "the resident weight slab; 0 derives it from the card" },
    { "--vram-kv-mib",        "N",      "the paged KV pool; 0 derives it from the card" },
    { "--kv-cache-dtype",     "T",      "bf16 | fp8 | k4v4 -- fp8 halves what a token of context costs; k4v4 (KVarN 1:1) quarters it" },
    { "--kvarn",              nullptr,  "enable KVarN 1:1 INT4 KV-cache compression with FWHT-256 rotation" },
    { "--qsa-offload",        nullptr,  "enable QSA Cross-Layer Host-Offloading (PCIe prefetch to VRAM)" },
    { "--host-pool-mib",      "N",      "pinned host tier" },
    { "--embedding-placement", "WHERE", "auto (default) | vram | ram: where the token embedding lives. ram\n"
    "                                    keeps it in the pinned host pool, read over the link in place;\n"
    "                                    auto keeps it on the card unless the KV cache could then not\n"
    "                                    hold one --max-model-len sequence. disk is refused: the lookup\n"
    "                                    writes the card's activations, which no host read can reach" },
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
    { "--prefix-cache-host-mib", "N",   "copy finished conversations' KV to host memory, up to N MiB,\n"
    "                                    so their VRAM can go to experts and a return restores\n"
    "                                    instead of re-prefilling. 0 is off. Part of it holds\n"
    "                                    linear-state snapshots on a hybrid model; startup says how many" },
    { "--prefix-cache-dir",   "DIR",    "where the disk tier keeps them. Not --weights-disk-tier, which is\n"
    "                                    about WEIGHTS" },
    { "--prefix-cache-disk-mib", "N",   "and copy them on to disk under --prefix-cache-dir, up to N MiB,\n"
    "                                    so a full host tier gives slots up for nothing. Holds\n"
    "                                    conversation content unencrypted. 0 is off" },
    { "--ple-backend",        "MODE",   "PLE table backend: device/gpu (default, zero-copy from host RAM/SHM)\n"
    "                                    or host/cpu/nvme (original CPU AVX sparse reads from NVMe)" },

    { "--placement",          "MODE",   "all_vram | layer_offload | expert_tiered | auto" },
    { "--deterministic",      nullptr,  "pin placement to the plan the loader made, so the resident\n"
    "                                    set cannot change under the heat engine. The sampler is\n"
    "                                    already order-independent and needs no pinning; note that\n"
    "                                    this does NOT make the engine bit-reproducible across\n"
    "                                    batch positions -- see spec §19" },
    { "--checkpoint-interval","N",      "linear-attention state checkpoint interval, in tokens" },
    /* TOTAL, NOT PER SEQUENCE. `ck_free_` is one global free list of N slots handed out by
     * pop_back (core/mem/kv.cpp), so an operator reading "per sequence" and setting 2 gets
     * 2 for the whole server rather than 2 x --max-num-seqs. At the default 8 slots and
     * 8 sequences that is one snapshot a sequence, which is what an append-only chat wants --
     * the next turn's hit lands at the end of the last one. */
    { "--checkpoint-slots",   "N",      "linear-state snapshots retained IN TOTAL (not "
                                          "per sequence); compare against --max-num-seqs" },
    { "--no-prefix-cache",    nullptr,  "disable prefix caching" },
    { "--checkpoint-policy",  "NAME",   "which snapshots a session retains: geometric-backoff "
                                        "(default), tip-only, keep-all" },

    { "--num-speculative-tokens", "N",  "speculative window; 0 disables. Default `auto`: the drafter's\n"
    "                                    own operating point, which the plugin states. A container\n"
    "                                    with no draft head is unaffected -- auto resolves to 0." },
    { "--adaptive-spec",      nullptr,  "enable dynamic speculative depth adaptation with baseline floor 3" },
    { "--draft-model",        "PATH",   "NOT IMPLEMENTED -- merge the drafter at convert time" },

    { "--host",               "ADDR",   "listen address" },
    { "--port",               "N",      "listen port" },
    { "--api-key",            "KEY",    "require `Authorization: Bearer KEY` on every request but /health, /ping and the dashboard page" },
    { "--served-model-name",  "NAME",   "the model id /v1/models lists and responses carry; default the container's name" },
    { "--mm-max-patches",     "N",      "patches one encoder pass carries, and so the largest image (a patch is 16x16\n"
                                        "pixels). Default `auto`: 16384 when the container carries a vision tower. 0\n"
                                        "serves text only and keeps the tower off the card" },
    { "--generation-config",  "PATH",   "a generation_config.json whose sampler settings become the default\n"
    "                                    for requests that send none. Without it the container's own\n"
    "                                    `generation.*` metadata is used, then a <model>.generation.json\n"
    "                                    beside the container, then the built-in defaults" },
    { "--override-chat-template", "PATH", "a Jinja chat template file, used in place of the one the container carries" },
    { "--temp",               "F",      "default temperature for requests that do not set one" },
    { "--top-k",              "N",      "default top_k (0 = off) for requests that do not set one" },
    { "--top-p",              "F",      "default top_p for requests that do not set one" },
    { "--min-p",              "F",      "default min_p (0 = off) for requests that do not set one.\n"
    "                                    These four override whatever the container or its generation\n"
    "                                    config asked for; unset, they are resolved from it" },
    { "--reasoning-effort",   "S",      "default `reasoning_effort` for chat requests that do not set one.\n"
    "                                    The value goes to the model's chat template verbatim, so the legal\n"
    "                                    set is the template's: Qwen3.8 takes xhigh|medium|low and defaults\n"
    "                                    to xhigh, which on a short task can spend the whole token budget\n"
    "                                    inside <think> and answer nothing. \"none\" turns thinking off" },
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
    { "--dry-allowed-length", "N",      "default dry_allowed_length" },
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
    { "--video-max-frame-tokens", "N",  "the most prompt tokens one video frame may become; 0 is no cap" },
    { "--max-source-pixels",  "N",      "the largest image or video frame the decoder accepts, before any resize\n"
    "                                    (default 67108864). The media flags above default to what the container\n"
    "                                    states, and do nothing for a model that takes no media" },

    { "--ridgefill",          "MODE",   "RidgeFill cold prefill acceleration: off|speed|quality|plumb (default off)" },
    { "--ridgefill-projector", "DIR",   "path to RidgeFill projector folder (default <model dir>/projector)" },

    { "--kld-record",         "DIR",    "instead of serving, run --kld-corpus through prefill and write this\n"
    "                                    model's log-probabilities at every scored position to DIR: the\n"
    "                                    reference a quantised model is measured against" },
    { "--kld-ref",            "DIR",    "instead of serving, score this model against the reference in DIR:\n"
    "                                    mean, median, 99/99.9/99.99th percentile KL divergence and top-1\n"
    "                                    agreement, over the positions and tokens DIR was recorded at" },
    { "--kld-corpus",         "FILE",   "JSONL, one {\"prompt\", \"score_from\"?, \"source\"?} a line; --kld-record only" },
    { "--kld-out",            "FILE",   "the --kld-ref report as JSON, and every position's numbers in FILE.rows" },

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
    for (const Flag& f : g_flags) {
        char left[64];
        snprintf(left, sizeof left, "%s%s%s", f.name, f.arg ? " " : "", f.arg ? f.arg : "");
        fprintf(stderr, "  %-32s %s\n", left, f.help);
    }
    fprintf(stderr,
        "\nEvery pool is budgeted explicitly. The trade between KV blocks and resident weights\n"
        "is the operator's to make: an engine that re-optimises it quietly is an engine whose\n"
        "benchmarks do not reproduce.\n");
}

/* The four names exactly, lower case: what a table's placement costs is the operator's to state,
 * so a spelling this build does not know stops the server rather than picking a tier. */
static bool parse_table_placement(const char* v, TablePlacement* out) {
    if      (!std::strcmp(v, "auto")) *out = TablePlacement::Auto;
    else if (!std::strcmp(v, "vram")) *out = TablePlacement::Vram;
    else if (!std::strcmp(v, "ram"))  *out = TablePlacement::Ram;
    else if (!std::strcmp(v, "disk")) *out = TablePlacement::Disk;
    else return false;
    return true;
}

static std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

/* A NUMBER IS THE WHOLE ARGUMENT OR IT IS REFUSED. atof reads "high" as 0, and for --temp that is
 * greedy decoding: a typo that changes every reply and says nothing. */
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

/* A SAMPLER DEFAULT IS CHECKED AGAINST THE RANGE A REQUEST'S OWN VALUE IS (server::sampler_value_ok),
 * here rather than at the first request. A default outside it would be a 400 on every request
 * that omits the field -- a server that starts, reports healthy, and answers nothing. */
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

/* THE NAMES THE RENDERER SETS ITSELF. The template context is built from these and the variables
 * are written over them (common_chat_template_direct_apply_impl), so a variable by one of these
 * names would replace the conversation, the tools or the special tokens of every request. */
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
        /* dump(): the template context takes JSON, so a string arrives still quoted. */
        (*out)[it.key()] = it.value().dump();
    }
    return true;
}

/* Returns 0 on success, 1 for --help (caller exits 0), negative on a bad argument. An unknown
 * flag is an error naming it, not a warning: a typo in a budget flag is the difference between
 * the plan you asked for and one the planner invented. */
int config_parse(int argc, char** argv, Config* c) {
    bool ple_backend_explicit = false;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { RAD_ERR("%s needs an argument", argv[i]); return nullptr; }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        const char* v = nullptr;
        auto S = [&](const char* n) { return a == n; };

        if (S("--help") || S("-h"))            return 1;
        else if (S("-v"))                      log_set_level(Log::Debug);
        else if (S("-vv"))                     log_set_level(Log::Trace);
        else if (S("--model"))                 { if (!(v = need(i))) return RAD_E_INVAL; c->model = v; }
        else if (S("--radiance-home"))         { if (!(v = need(i))) return RAD_E_INVAL; c->radiance_home = v; }
        else if (S("--kernels"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kernel_hierarchy = split_commas(v); }
        else if (S("--tp"))                    { if (!(v = need(i))) return RAD_E_INVAL; c->tp = atoi(v); }
        else if (S("--tp-wire")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if      (!std::strcmp(v, "exact")) c->tp_wire_exact = 1;
            else if (!std::strcmp(v, "wht6"))  c->tp_wire_exact = 0;
            else { RAD_ERR("--tp-wire takes exact or wht6, not '%s'", v); return RAD_E_INVAL; }
        }
        else if (S("--tp-wire-min-kb"))        { if (!(v = need(i))) return RAD_E_INVAL; c->tp_wire_min_kb = atoll(v); }
        else if (S("--max-num-batched-tokens")){ if (!(v = need(i))) return RAD_E_INVAL; c->max_tok = atoll(v); }
        else if (S("--max-num-seqs"))          { if (!(v = need(i))) return RAD_E_INVAL; c->max_seqs = atoll(v); }
        else if (S("--max-model-len"))         { if (!(v = need(i))) return RAD_E_INVAL; c->max_ctx = atoll(v); }
        else if (S("--vram-weights-mib"))      { if (!(v = need(i))) return RAD_E_INVAL; c->vram_weights_mib = atoll(v); }
        else if (S("--vram-kv-mib"))           { if (!(v = need(i))) return RAD_E_INVAL; c->vram_kv_mib = atoll(v); }
        /* --gpu-headroom takes the short name too: it is the flag an operator reaches for and
         * "-mib" is the unit, which every other size flag in this list also spells out. Both
         * parse to the same field. */
        else if (S("--gpu-headroom-mib") || S("--gpu-headroom"))
                                               { if (!(v = need(i))) return RAD_E_INVAL; c->gpu_headroom_mib = atoll(v); }
        else if (S("--prefill-stage-mib") || S("--prefill-stage"))
                                               { if (!(v = need(i))) return RAD_E_INVAL; c->prefill_stage_mib = atoll(v); }
        else if (S("--kv-cache-dtype")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::strcmp(v, "bf16") && std::strcmp(v, "fp8") && std::strcmp(v, "k4v4")) {
                RAD_ERR("--kv-cache-dtype takes bf16, fp8, or k4v4, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->kv_cache_dtype = v;
            if (!std::strcmp(v, "k4v4")) c->kvarn = true;
        }
        else if (S("--kvarn")) {
            c->kvarn = true;
            c->kv_cache_dtype = "k4v4";
        }
        else if (S("--qsa-offload")) {
            c->qsa_offload = true;
        }
        else if (S("--expert-vs-cache-ratio")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            c->expert_cache_ratio = atof(v);
            if (!(c->expert_cache_ratio > 0.0 && c->expert_cache_ratio < 1.0)) {
                RAD_ERR("--expert-vs-cache-ratio is the WEIGHTS share of what is claimed and must "
                        "be strictly between 0 and 1, not '%s'. 0.75 gives three quarters to "
                        "weights and a quarter to the KV cache.", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--host-pool-mib"))         { if (!(v = need(i))) return RAD_E_INVAL; c->host_pool_mib = atoll(v); }
        else if (S("--embedding-placement") || S("--ngram-placement")) {
            const bool emb = S("--embedding-placement");
            if (!(v = need(i))) return RAD_E_INVAL;
            TablePlacement p;
            if (!parse_table_placement(v, &p)) {
                RAD_ERR("%s takes auto, vram, ram or disk, not '%s'", a.c_str(), v);
                return RAD_E_INVAL;
            }
            (emb ? c->embedding_placement : c->ngram_placement) = p;
        }
        else if (S("--weights-disk-tier"))     { c->weights_disk_tier = true; }
        else if (S("--prefix-cache-host-mib"))     { if (!(v = need(i))) return RAD_E_INVAL; c->prefix_cache_host_mib = atoll(v); }
        else if (S("--prefix-cache-dir"))          { if (!(v = need(i))) return RAD_E_INVAL; c->prefix_cache_dir = v; }
        else if (S("--prefix-cache-disk-mib"))     { if (!(v = need(i))) return RAD_E_INVAL; c->prefix_cache_disk_mib = atoll(v); }
        else if (S("--ple-backend") || S("--ple-storage") || S("--ple-mode")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            ple_backend_explicit = true;
            if (!std::strcmp(v, "device") || !std::strcmp(v, "gpu") ||
                !std::strcmp(v, "shm") || !std::strcmp(v, "zerocopy") || !std::strcmp(v, "zero-copy")) {
                c->ple_backend = "device";
            } else if (!std::strcmp(v, "host") || !std::strcmp(v, "cpu") ||
                       !std::strcmp(v, "nvme") || !std::strcmp(v, "mmap")) {
                c->ple_backend = "host";
            } else {
                RAD_ERR("--ple-backend takes device|gpu or host|cpu|nvme, not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--placement"))             { if (!(v = need(i))) return RAD_E_INVAL; c->placement = v; }
        else if (S("--deterministic"))         c->deterministic = true;
        else if (S("--checkpoint-interval"))   { if (!(v = need(i))) return RAD_E_INVAL; c->checkpoint_interval = atoll(v); }
        else if (S("--checkpoint-slots"))      { if (!(v = need(i))) return RAD_E_INVAL; c->checkpoint_slots = atoll(v); }
        else if (S("--no-prefix-cache"))       c->prefix_cache = false;
        else if (S("--prefix-aware-prefetch"))    c->prefix_aware_prefetch = true;
        else if (S("--no-prefix-aware-prefetch")) c->prefix_aware_prefetch = false;
        else if (S("--ced-prefetch"))             c->ced_prefetch = true;
        else if (S("--no-ced-prefetch"))          c->ced_prefetch = false;
        else if (S("--ced-lookahead"))            { if (!(v = need(i))) return RAD_E_INVAL; c->ced_lookahead = atoi(v); }
        else if (S("--ced-threshold"))            { if (!(v = need(i))) return RAD_E_INVAL; c->ced_threshold = (float)atof(v); }
        else if (S("--ced-topk"))                 { if (!(v = need(i))) return RAD_E_INVAL; c->ced_top_k = atoi(v); }
        else if (S("--ced-reuse-ridgefill"))      c->ced_reuse_ridgefill = true;
        else if (S("--checkpoint-policy")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* Named here rather than accepted blind: a policy name this build does not know would
             * otherwise fall through to the default and the measurement would compare the default
             * against itself. */
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
            /* "auto" is -1 and the engine resolves it once the container is open: the depth is a
             * property of the DRAFTER and the ARCHITECTURE PLUGIN is the one that knows it
             * (rad_arch_probe): a serial head names a measured throughput trade-off, a block
             * drafter names the block length its checkpoint was trained at. */
            if (!std::strcmp(v, "auto")) c->n_spec = -1;
            else c->n_spec = atoi(v);
            /* Every negative depth reads as auto downstream, so only the word may say it. */
            if (std::strcmp(v, "auto") && c->n_spec < 0) {
                RAD_ERR("--num-speculative-tokens takes auto or a depth >= 0 (0 turns speculation "
                        "off), not '%s'", v);
                return RAD_E_INVAL;
            }
        }
        else if (S("--adaptive-spec")) {
            c->adaptive_spec = true;
        }
        else if (S("--draft-model")) {
            /* REFUSED, NOT QUIETLY IGNORED. Accepting this flag and then serving the container's
             * own drafter -- or none -- would hand the caller a server that is not the one they
             * asked for, with nothing in the output to say so.
             *
             * It refuses rather than disappearing because a caller coming from vLLM will reach
             * for it, and a message that says where the drafter actually comes from is the whole
             * value the flag can still deliver. This engine loads its drafter from the model
             * container: rad-convert's OWN --draft-model merges one in, and an MTP head
             * that ships inside the target checkpoint needs no flag at all. spec.md §10's
             * "separate draft models load as a second plugin" is a design, not a feature. */
            if (!(v = need(i))) return RAD_E_INVAL;
            RAD_ERR("--draft-model is not implemented. This engine loads its drafter from the "
                    "model container, not from a second one: merge it at conversion time with "
                    "`rad-convert --draft-model DIR` and pass the container to --model. "
                    "An MTP head that ships inside the target checkpoint needs neither flag.");
            return RAD_E_INVAL;
        }
        else if (S("--host"))                  { if (!(v = need(i))) return RAD_E_INVAL; c->host = v; }
        else if (S("--port"))                  { if (!(v = need(i))) return RAD_E_INVAL; c->port = atoi(v); }
        else if (S("--api-key"))               { if (!(v = need(i))) return RAD_E_INVAL; c->api_key = v; }
        else if (S("--served-model-name")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!*v) { RAD_ERR("--served-model-name takes a non-empty name"); return RAD_E_INVAL; }
            c->served_model_name = v;
        }
        else if (S("--mm-max-patches")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (!std::strcmp(v, "auto")) c->mm_max_patches = -1;
            else {
                char* end = nullptr;
                const long long n = std::strtoll(v, &end, 10);
                if (!end || *end || n < 0) {
                    RAD_ERR("--mm-max-patches takes auto or a patch count >= 0, not '%s'", v);
                    return RAD_E_INVAL;
                }
                c->mm_max_patches = n;
            }
        }
        else if (S("--reasoning-effort"))      { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->reasoning_effort = v; }
        else if (S("--thinking-token-budget")) { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 0, INT32_MAX,
                                                                &c->thinking_token_budget))
                                                     return RAD_E_INVAL; }
        else if (S("--generation-config"))     { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->generation_config = v; }
        else if (S("--override-chat-template")) { if (!(v = need(i))) return RAD_E_INVAL;
                                                 c->override_chat_template = v; }
        /* A NEGATIVE VALUE IS REFUSED BY THE RANGE: these four fields use a negative number to
         * mean "not set", so `--temp -0.5` taken as given would be read as no flag at all and
         * the container's default served in its place. */
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
            /* The sampler refuses a DRY request with more (sample/host_ref.h), and every request
             * leaving the field to this default would be that request. */
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
        else if (S("--max-tokens-cap"))        { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 0, INT32_MAX, &c->max_tokens_cap))
                                                     return RAD_E_INVAL; }
        else if (S("--max-n"))                 { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 1, INT32_MAX, &c->max_n))
                                                     return RAD_E_INVAL; }
        else if (S("--max-queued-requests"))   { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 1, INT32_MAX, &c->max_queued_requests))
                                                     return RAD_E_INVAL; }
        else if (S("--max-stop-strings"))      { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 1, INT32_MAX, &c->max_stop_strings))
                                                     return RAD_E_INVAL; }
        else if (S("--max-stop-bytes"))        { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 1, INT32_MAX, &c->max_stop_bytes))
                                                     return RAD_E_INVAL; }
        else if (S("--chat-template-kwargs"))  { if (!(v = need(i)) ||
                                                     !parse_template_kwargs(v, &c->chat_template_kwargs))
                                                     return RAD_E_INVAL; }
        else if (S("--reasoning-format")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* Two values, because the reply reader makes one distinction: whether a reasoning
             * block is lifted out of the content or left in it. */
            if (std::strcmp(v, "auto") && std::strcmp(v, "none")) {
                RAD_ERR("--reasoning-format takes auto or none, not '%s'", v);
                return RAD_E_INVAL;
            }
            c->reasoning_format = v;
        }
        else if (S("--http-threads") || S("--read-timeout") || S("--write-timeout") ||
                 S("--keep-alive-timeout") || S("--retry-after")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            /* The transport keeps at least four workers whatever it is told (server/http.cpp);
             * fewer would be a number this flag accepted and the server did not serve. */
            const int64_t lo = S("--http-threads") ? 4 : S("--retry-after") ? 0 : 1;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, lo, INT32_MAX, &n)) return RAD_E_INVAL;
            (S("--http-threads")    ? c->http_threads
             : S("--read-timeout")  ? c->http_read_timeout_s
             : S("--write-timeout") ? c->http_write_timeout_s
             : S("--retry-after")   ? c->retry_after_s
                                    : c->http_keep_alive_timeout_s) = (int)n;
        }
        else if (S("--max-body-mib"))          { if (!(v = need(i)) ||
                                                     !parse_int(a.c_str(), v, 1, (int64_t)1 << 20, &c->http_max_body_mib))
                                                     return RAD_E_INVAL; }
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
        else if (S("--video-min-frames") || S("--video-max-frames") || S("--video-max-frame-tokens")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            int64_t n = 0;
            if (!parse_int(a.c_str(), v, S("--video-max-frame-tokens") ? 0 : 1, INT32_MAX, &n))
                return RAD_E_INVAL;
            (S("--video-min-frames")   ? c->video_min_frames
             : S("--video-max-frames") ? c->video_max_frames
                                       : c->video_max_frame_tokens) = (int32_t)n;
        }
        else if (S("--ridgefill")) {
            if (!(v = need(i))) return RAD_E_INVAL;
            if (std::string(v) != "off" && std::string(v) != "speed" && std::string(v) != "quality" && std::string(v) != "plumb") {
                RAD_ERR("--ridgefill must be off, speed, quality, or plumb (got '%s')", v); return RAD_E_INVAL; }
            c->ridgefill = v;
        }
        else if (S("--ridgefill-projector"))   { if (!(v = need(i))) return RAD_E_INVAL; c->ridgefill_projector = v; }
        else if (S("--kld-record"))            { if (!(v = need(i))) return RAD_E_INVAL; c->kld_record = v; }
        else if (S("--kld-ref"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kld_ref = v; }
        else if (S("--kld-corpus"))            { if (!(v = need(i))) return RAD_E_INVAL; c->kld_corpus = v; }
        else if (S("--kld-out"))               { if (!(v = need(i))) return RAD_E_INVAL; c->kld_out = v; }
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
    }

    if (c->radiance_home.empty()) {
        const char* h = getenv("RADIANCE_HOME");
        c->radiance_home = h ? h : RAD_DEFAULT_HOME;
    }
    if (c->kernel_hierarchy.empty()) {
        /* RADIANCE_KERNELS, ahead of the built-in default.
         *
         * The tools read the same variable through rad_hierarchy_from_env() in tools/iface.cpp,
         * and the engine has to honour it too: otherwise an operator who validated a third-party
         * plugin with rad-kbench or rad-tune has no way to serve with it short of editing every
         * launcher's command line. Same spelling, same separator. */
        const char* e = getenv("RADIANCE_KERNELS");
        if (e && *e) c->kernel_hierarchy = split_commas(e);
    }
    if (!ple_backend_explicit) {
        const char* ple_env = getenv("RADIANCE_PLE_BACKEND");
        if (!ple_env) ple_env = getenv("RADIANCE_PLE_STORAGE");
        if (!ple_env) ple_env = getenv("RADIANCE_PLE_MODE");
        if (ple_env && *ple_env) {
            if (!std::strcmp(ple_env, "host") || !std::strcmp(ple_env, "cpu") ||
                !std::strcmp(ple_env, "nvme") || !std::strcmp(ple_env, "mmap")) {
                c->ple_backend = "host";
            } else if (!std::strcmp(ple_env, "device") || !std::strcmp(ple_env, "gpu") ||
                       !std::strcmp(ple_env, "shm") || !std::strcmp(ple_env, "zerocopy") || !std::strcmp(ple_env, "zero-copy")) {
                c->ple_backend = "device";
            }
        }
    }
    if (c->ple_backend.empty()) c->ple_backend = "device";
    setenv("RADIANCE_PLE_BACKEND", c->ple_backend.c_str(), 1);

    if (!c->kvarn) {
        if (const char* kvarn_env = std::getenv("RADIANCE_KVARN")) {
            if (*kvarn_env && std::strcmp(kvarn_env, "0")) {
                c->kvarn = true;
                c->kv_cache_dtype = "k4v4";
            }
        }
    }
    if (!c->qsa_offload) {
        if (const char* off_env = std::getenv("RADIANCE_QSA_OFFLOAD")) {
            if (*off_env && std::strcmp(off_env, "0")) {
                c->qsa_offload = true;
            }
        }
    }
    /* AND THERE IS NO BUILT-IN DEFAULT, deliberately. Naming a library here -- the RDNA4 one, say
     * -- would make the core carry the name of one card's kernels, and on any other machine that
     * name matches nothing: the engine would warn about a library that is not installed and then
     * rank the one that IS installed behind the reference implementation, which matches every op
     * and would answer for all of them.
     *
     * An empty hierarchy is not "no order". The loader orders what it finds: everything named by
     * --kernels or RADIANCE_KERNELS first, then every other plugin, then the reference library
     * last by construction because it matches anything (core/plugin/loader.cpp). So dropping a
     * kernel library into $RADIANCE_HOME/kernels is enough to be used, on any card, with no
     * configuration -- and naming one ahead of another is what settles a tie between two that
     * both serve an op. */
    if (c->model.empty()) {
        RAD_ERR("--model is required");
        return RAD_E_INVAL;
    }
    if (c->tp < 1) { RAD_ERR("--tp must be >= 1"); return RAD_E_INVAL; }

    /* THE KL MODE: one direction, and each direction with what it reads. A run given both would
     * have to pick which one the operator meant, and picking wrong records a quantised model as
     * the yardstick. */
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

    /* TWO FLAGS THAT WOULD SAY THE SAME THING. --reasoning-effort is a template variable with one
     * extra meaning ("none" is enable_thinking false), so naming either key in
     * --chat-template-kwargs as well leaves two defaults for one setting and no order to read. */
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
    /* A request is admitted only if all its choices fit the queue, so an `n` past it would be a
     * 429 however long the caller waited. Checked only when one of the two was stated: the
     * defaults (8, or --max-num-seqs; and 8 x --max-num-seqs) always fit. */
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

    /* Normalise and export RidgeFill flags to environment for plugin compatibility */
    {
        auto rf_cfg = rad::ridgefill::InteropConfig::from_env_and_flags(c->ridgefill, c->ridgefill_projector);
        rf_cfg.apply_to_env();
    }

    return RAD_OK;
}

}  /* namespace rad */
