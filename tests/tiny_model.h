/* tiny_model.h -- small Hugging Face-style checkpoints written on the fly, for tests that run the
 * engine or rad-convert end to end without model files.
 *
 * A checkpoint is a directory holding config.json, model.safetensors (bf16 unless a tensor states
 * otherwise) and a byte-level tokenizer.json. The weights are seeded uniform noise, each projection
 * scaled to keep its input's variance.
 *
 * Gains. A random transformer whose layers dominate the residual stream averages its context into
 * one fixed point, and greedy decoding then repeats a single token, which hides a row that reads
 * another row's state. `branch` scales the projections that write the residual stream, so the
 * current token's embedding steers the next token and greedy text runs through varied cycles.
 * `head` sets the logits' spread (standard deviation about 2): peaked enough that a tie is rare,
 * flat enough that temperature sampling departs from greedy.
 *
 * The tokenizer is GPT-2 byte-level BPE with no merges: token id b is byte b for b < 256, followed
 * by the special tokens. A text prompt of n bytes is n tokens, so a test can reason about positions
 * from the text alone. The model's vocabulary is wider than the tokenizer's, as real checkpoints
 * pad theirs.
 *
 * Each writer takes a shape struct whose defaults are the smallest shape the plugin serves on the
 * host backend; the tensor names are the ones the Hugging Face release uses.
 */
#pragma once
#include "rad_plugin.h"
#include "rad_test.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tiny {

/* ------------------------------------------------------------------ files */

/* A new directory under the test's temporary directory (radtest::tmpdir()), which the harness
 * removes when the binary passes. A kept one is made beside that directory instead, so it outlives
 * the run, for a person or a script to read. */
struct TempDir {
    std::string path;
    explicit TempDir(const char* tag = "tiny", bool keep = false) {
        const std::string base = keep
            ? std::filesystem::path(radtest::tmpdir()).parent_path().string()
            : radtest::tmpdir();
        std::string tmpl = base + "/radiance_" + tag + ".XXXXXX";
        if (mkdtemp(tmpl.data())) path = tmpl;
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
    bool ok() const { return !path.empty(); }
    std::string operator/(const std::string& name) const { return path + "/" + name; }
};

inline bool write_text(const std::string& path, const std::string& body) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(body.data(), 1, body.size(), f) == body.size();
    return std::fclose(f) == 0 && ok;
}

inline std::string read_text(const std::string& path) {
    std::string out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

/* ------------------------------------------------------------------ tensors */

struct Tensor {
    std::string          name;
    std::string          dtype;   /* safetensors spelling: BF16, F32 */
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

/* splitmix64: a seed and a stream, nothing else, so the same seed writes the same checkpoint on
 * every platform. */
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    /* Uniform in [-1, 1). */
    float uniform() { return (float)((double)(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0); }
};

inline int64_t numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
}

/* Elements `center + amp * u`, u uniform in [-1, 1). */
inline Tensor noise(const std::string& name, std::vector<int64_t> shape, Rng& r, float amp,
                    float center = 0.0f, const char* dtype = "BF16") {
    Tensor t{ name, dtype, shape, {} };
    const bool f32 = std::string(dtype) == "F32";
    const int64_t n = numel(shape);
    t.bytes.resize((size_t)n * (f32 ? 4 : 2));
    for (int64_t i = 0; i < n; ++i)
        rad_store_f32(t.bytes.data(), f32 ? RAD_F32 : RAD_BF16, i, center + amp * r.uniform());
    return t;
}

/* A projection [n_out, n_in] whose output has about the input's variance: uniform with
 * amplitude sqrt(3 / n_in) has variance 1 / n_in. */
inline Tensor linear(const std::string& name, int64_t n_out, int64_t n_in, Rng& r,
                     float gain = 1.0f) {
    return noise(name, { n_out, n_in }, r, gain * std::sqrt(3.0f / (float)n_in));
}

/* One safetensors file holding `ts` in order, with the header padded to 8 bytes. */
inline bool write_safetensors(const std::string& path, const std::vector<Tensor>& ts) {
    nlohmann::ordered_json h = nlohmann::ordered_json::object();
    uint64_t off = 0;
    for (const Tensor& t : ts) {
        h[t.name] = { { "dtype", t.dtype }, { "shape", t.shape },
                      { "data_offsets", { off, off + t.bytes.size() } } };
        off += t.bytes.size();
    }
    std::string hdr = h.dump();
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint64_t n = hdr.size();
    bool ok = std::fwrite(&n, 8, 1, f) == 1 &&
              std::fwrite(hdr.data(), 1, hdr.size(), f) == hdr.size();
    for (const Tensor& t : ts)
        ok = ok && std::fwrite(t.bytes.data(), 1, t.bytes.size(), f) == t.bytes.size();
    return std::fclose(f) == 0 && ok;
}

/* The tensors of a safetensors file write_safetensors wrote, in file order; empty if unreadable. */
inline std::vector<Tensor> read_safetensors(const std::string& path) {
    std::vector<Tensor> ts;
    const std::string raw = read_text(path);
    if (raw.size() < 8) return ts;
    uint64_t n = 0;
    std::memcpy(&n, raw.data(), 8);
    if (8 + n > raw.size()) return ts;
    const auto h = nlohmann::ordered_json::parse(raw.substr(8, n), nullptr, false);
    if (!h.is_object()) return ts;
    for (auto it = h.begin(); it != h.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto& v = it.value();
        Tensor t{ it.key(), v["dtype"].get<std::string>(), v["shape"].get<std::vector<int64_t>>(),
                  {} };
        const uint64_t a = v["data_offsets"][0].get<uint64_t>();
        const uint64_t b = v["data_offsets"][1].get<uint64_t>();
        if (8 + n + b > raw.size() || b < a) return {};
        t.bytes.assign(raw.begin() + (long)(8 + n + a), raw.begin() + (long)(8 + n + b));
        ts.push_back(std::move(t));
    }
    return ts;
}

/* ------------------------------------------------------------------ the tokenizer */

/* The GPT-2 byte alphabet: byte b's symbol, as UTF-8. Printable Latin-1 bytes map to themselves;
 * the other 68 map to U+0100 upwards in byte order. */
inline std::string byte_symbol(int b) {
    auto printable = [](int c) {
        return (c >= 0x21 && c <= 0x7E) || (c >= 0xA1 && c <= 0xAC) || (c >= 0xAE && c <= 0xFF);
    };
    int cp = b;
    if (!printable(b)) {
        int k = 0;
        for (int c = 0; c < b; ++c) k += !printable(c);
        cp = 0x100 + k;
    }
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}

/* Ids of the special tokens every tiny tokenizer carries, after the 256 bytes. */
enum : int32_t { kEndOfText = 256, kImStart = 257, kImEnd = 258, kTokenizerSize = 259 };

/* tokenizer.json: byte-level BPE, ids 0..255 the bytes, no merges, then the specials, then
 * `extra` special tokens from kTokenizerSize on. */
inline std::string tokenizer_json(const std::vector<std::string>& extra = {}) {
    nlohmann::ordered_json vocab = nlohmann::ordered_json::object();
    for (int b = 0; b < 256; ++b) vocab[byte_symbol(b)] = b;
    std::vector<std::string> specials = { "<|endoftext|>", "<|im_start|>", "<|im_end|>" };
    specials.insert(specials.end(), extra.begin(), extra.end());
    nlohmann::ordered_json added = nlohmann::ordered_json::array();
    for (int i = 0; i < (int)specials.size(); ++i) {
        vocab[specials[(size_t)i]] = 256 + i;
        added.push_back({ { "id", 256 + i }, { "content", specials[(size_t)i] },
                          { "single_word", false }, { "lstrip", false }, { "rstrip", false },
                          { "normalized", false }, { "special", true } });
    }
    nlohmann::ordered_json j;
    j["version"] = "1.0";
    j["added_tokens"] = added;
    j["normalizer"] = nullptr;
    j["pre_tokenizer"] = { { "type", "ByteLevel" }, { "add_prefix_space", false },
                           { "trim_offsets", false }, { "use_regex", true } };
    j["post_processor"] = nullptr;
    j["decoder"] = { { "type", "ByteLevel" }, { "add_prefix_space", false },
                     { "trim_offsets", false }, { "use_regex", true } };
    j["model"] = { { "type", "BPE" }, { "dropout", nullptr }, { "unk_token", nullptr },
                   { "byte_fallback", false }, { "vocab", vocab },
                   { "merges", nlohmann::ordered_json::array() } };
    return j.dump();
}

/* The ids a byte-level tokenizer with no merges gives `text`. */
inline std::vector<int32_t> byte_ids(const std::string& text) {
    std::vector<int32_t> ids;
    for (unsigned char c : text) ids.push_back((int32_t)c);
    return ids;
}

/* config.json, model.safetensors and tokenizer.json in `dir`. */
inline bool write_checkpoint(const std::string& dir, const nlohmann::ordered_json& config,
                             const std::vector<Tensor>& ts,
                             const std::vector<std::string>& extra_specials = {}) {
    return write_text(dir + "/config.json", config.dump(2)) &&
           write_safetensors(dir + "/model.safetensors", ts) &&
           write_text(dir + "/tokenizer.json", tokenizer_json(extra_specials));
}

/* ------------------------------------------------------------------ qwen35: attention + GDN */

/* Qwen3.5 dense (`qwen3_5`, served by qwen35_bf16): gated full attention every `interval`-th layer
 * and gated delta net between, with the multimodal release's `model.language_model.` names. */
struct Qwen35 {
    int64_t layers = 4, hidden = 64, heads = 4, kv_heads = 4, head_dim = 32, ffn = 128;
    int64_t vocab = 320, ctx = 4096, interval = 2;
    double  rotary = 0.5;                       /* partial_rotary_factor */
    int64_t lin_k_heads = 2, lin_v_heads = 4, lin_k_dim = 16, lin_v_dim = 16, conv = 4;
    float    branch = 0.5f, head = 2.0f;        /* see Gains */
    uint64_t seed = 1;
};

inline bool write_qwen35(const std::string& dir, const Qwen35& s) {
    Rng r(s.seed);
    nlohmann::ordered_json types = nlohmann::ordered_json::array();
    for (int64_t l = 0; l < s.layers; ++l)
        types.push_back((l + 1) % s.interval == 0 ? "full_attention" : "linear_attention");
    nlohmann::ordered_json c;
    c["architectures"] = { "Qwen3_5ForConditionalGeneration" };
    c["model_type"] = "qwen3_5";
    c["vocab_size"] = s.vocab;
    c["hidden_size"] = s.hidden;
    c["intermediate_size"] = s.ffn;
    c["num_hidden_layers"] = s.layers;
    c["num_attention_heads"] = s.heads;
    c["num_key_value_heads"] = s.kv_heads;
    c["head_dim"] = s.head_dim;
    c["max_position_embeddings"] = s.ctx;
    c["rms_norm_eps"] = 1e-6;
    c["rope_parameters"] = { { "rope_type", "default" }, { "rope_theta", 10000.0 },
                             { "partial_rotary_factor", s.rotary } };
    c["full_attention_interval"] = s.interval;
    c["layer_types"] = types;
    c["linear_num_key_heads"] = s.lin_k_heads;
    c["linear_num_value_heads"] = s.lin_v_heads;
    c["linear_key_head_dim"] = s.lin_k_dim;
    c["linear_value_head_dim"] = s.lin_v_dim;
    c["linear_conv_kernel_dim"] = s.conv;
    c["tie_word_embeddings"] = false;
    c["eos_token_id"] = (int)kEndOfText;

    const int64_t H = s.hidden, q = s.heads * s.head_dim, kv = s.kv_heads * s.head_dim;
    const int64_t kdim = s.lin_k_heads * s.lin_k_dim, vdim = s.lin_v_heads * s.lin_v_dim;
    const int64_t conv_dim = 2 * kdim + vdim;
    const std::string m = "model.language_model.";
    std::vector<Tensor> ts;
    ts.push_back(noise(m + "embed_tokens.weight", { s.vocab, H }, r, 1.0f));
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        /* GemmaRMSNorm gains: the weight is an offset from 1. */
        ts.push_back(noise(p + "input_layernorm.weight", { H }, r, 0.1f));
        if ((l + 1) % s.interval == 0) {
            /* Query and output gate interleaved per head: [heads, 2 * head_dim] rows. */
            ts.push_back(linear(p + "self_attn.q_proj.weight", 2 * q, H, r));
            ts.push_back(linear(p + "self_attn.k_proj.weight", kv, H, r));
            ts.push_back(linear(p + "self_attn.v_proj.weight", kv, H, r));
            ts.push_back(linear(p + "self_attn.o_proj.weight", H, q, r, s.branch));
            ts.push_back(noise(p + "self_attn.q_norm.weight", { s.head_dim }, r, 0.1f));
            ts.push_back(noise(p + "self_attn.k_norm.weight", { s.head_dim }, r, 0.1f));
        } else {
            ts.push_back(linear(p + "linear_attn.in_proj_qkv.weight", conv_dim, H, r));
            ts.push_back(linear(p + "linear_attn.in_proj_z.weight", vdim, H, r));
            ts.push_back(linear(p + "linear_attn.in_proj_a.weight", s.lin_v_heads, H, r));
            ts.push_back(linear(p + "linear_attn.in_proj_b.weight", s.lin_v_heads, H, r));
            ts.push_back(noise(p + "linear_attn.conv1d.weight", { conv_dim, 1, s.conv }, r, 0.5f));
            /* A_log = log(A) with A in [1, 8]; dt_bias around 1. */
            ts.push_back(noise(p + "linear_attn.A_log", { s.lin_v_heads }, r, 1.0f, 1.0f, "F32"));
            ts.push_back(noise(p + "linear_attn.dt_bias", { s.lin_v_heads }, r, 0.5f, 1.0f, "F32"));
            /* A plain gated RMSNorm: the gain itself. */
            ts.push_back(noise(p + "linear_attn.norm.weight", { s.lin_v_dim }, r, 0.1f, 1.0f));
            ts.push_back(linear(p + "linear_attn.out_proj.weight", H, vdim, r, s.branch));
        }
        ts.push_back(noise(p + "post_attention_layernorm.weight", { H }, r, 0.1f));
        ts.push_back(linear(p + "mlp.gate_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.up_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.down_proj.weight", H, s.ffn, r, s.branch));
    }
    ts.push_back(noise(m + "norm.weight", { H }, r, 0.1f));
    ts.push_back(linear("lm_head.weight", s.vocab, H, r, s.head));
    return write_checkpoint(dir, c, ts);
}

/* ------------------------------------------------------------------ lfm2: attention + conv */

/* LFM2 (`lfm2`, served by lfm2_bf16): short gated convolutions and full attention as layer_types
 * says, a tied head, and the stated FFN width (block_auto_adjust_ff_dim off). */
struct Lfm2 {
    int64_t layers = 4, hidden = 64, heads = 4, kv_heads = 2, ffn = 128, vocab = 320, ctx = 4096;
    int64_t conv = 3;                           /* conv_L_cache */
    float    branch = 0.2f, head = 0.5f;        /* see Gains; head is the tied table's amplitude */
    uint64_t seed = 2;
};

inline bool write_lfm2(const std::string& dir, const Lfm2& s) {
    Rng r(s.seed);
    const int64_t H = s.hidden, hd = s.hidden / s.heads, kv = s.kv_heads * hd;
    nlohmann::ordered_json types = nlohmann::ordered_json::array();
    for (int64_t l = 0; l < s.layers; ++l) types.push_back(l % 2 ? "full_attention" : "conv");
    nlohmann::ordered_json c;
    c["architectures"] = { "Lfm2ForCausalLM" };
    c["model_type"] = "lfm2";
    c["vocab_size"] = s.vocab;
    c["hidden_size"] = H;
    c["intermediate_size"] = s.ffn;
    c["block_auto_adjust_ff_dim"] = false;
    c["num_hidden_layers"] = s.layers;
    c["num_attention_heads"] = s.heads;
    c["num_key_value_heads"] = s.kv_heads;
    c["max_position_embeddings"] = s.ctx;
    c["norm_eps"] = 1e-5;
    c["rope_parameters"] = { { "rope_type", "default" }, { "rope_theta", 1000000.0 } };
    c["conv_bias"] = false;
    c["conv_L_cache"] = s.conv;
    c["layer_types"] = types;
    c["tie_word_embeddings"] = true;
    c["eos_token_id"] = (int)kEndOfText;

    const std::string m = "model.";
    std::vector<Tensor> ts;
    /* Tied: the head reads this table, so its scale sets the logits' (standard deviation about
     * sqrt(H) * 0.58). */
    ts.push_back(noise(m + "embed_tokens.weight", { s.vocab, H }, r, s.head));
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        ts.push_back(noise(p + "operator_norm.weight", { H }, r, 0.1f, 1.0f));
        if (l % 2) {
            ts.push_back(linear(p + "self_attn.q_proj.weight", H, H, r));
            ts.push_back(linear(p + "self_attn.k_proj.weight", kv, H, r));
            ts.push_back(linear(p + "self_attn.v_proj.weight", kv, H, r));
            ts.push_back(linear(p + "self_attn.out_proj.weight", H, H, r, s.branch));
            ts.push_back(noise(p + "self_attn.q_layernorm.weight", { hd }, r, 0.1f, 1.0f));
            ts.push_back(noise(p + "self_attn.k_layernorm.weight", { hd }, r, 0.1f, 1.0f));
        } else {
            ts.push_back(linear(p + "conv.in_proj.weight", 3 * H, H, r));
            ts.push_back(noise(p + "conv.conv.weight", { H, 1, s.conv }, r, 0.5f));
            ts.push_back(linear(p + "conv.out_proj.weight", H, H, r, s.branch));
        }
        ts.push_back(noise(p + "ffn_norm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "feed_forward.w1.weight", s.ffn, H, r));
        ts.push_back(linear(p + "feed_forward.w3.weight", s.ffn, H, r));
        ts.push_back(linear(p + "feed_forward.w2.weight", H, s.ffn, r, s.branch));
    }
    /* Signed gains around 0: with a tied head an embedding-led stream would otherwise score its own
     * token highest, and greedy text would repeat one token. */
    ts.push_back(noise(m + "embedding_norm.weight", { H }, r, 1.0f));
    return write_checkpoint(dir, c, ts);
}

/* ------------------------------------------------------------------ gemma4: sliding + global */

/* Gemma 4 dense (`gemma4`, served by the gemma4 plugin as written), shaped like the 31B: sliding
 * layers with a v projection and global layers that read V from K at a wider head, a tied head,
 * soft-capped logits and a per-layer scalar. The window is short, so a corpus document crosses it.
 * Every GEMM's K is a whole number of 128 columns, the group of the int4 recipes gemma4_bf16
 * serves. */
struct Gemma4 {
    int64_t layers = 4, hidden = 128, heads = 4, kv_heads = 2, head_dim = 32, ffn = 256;
    int64_t global_head_dim = 64, global_kv_heads = 1, window = 32, interval = 2;
    int64_t vocab = 320, ctx = 4096;
    float    branch = 0.5f, head = 0.35f;       /* see Gains; head is the tied table's amplitude */
    uint64_t seed = 3;
    /* A Gemma4 assistant head in a `draft/` companion folder, off by default: every checkpoint
     * written before this field exists keeps the exact bytes it had. The plugin picks the folder
     * up as a companion (its own config and weights, under `draft.`) and declares the head only
     * when max_spec > 0, so a fixture that sets this and a run without speculation still serves
     * the trunk alone. */
    bool     assistant = false;
    int64_t  assistant_width = 128, assistant_layers = 1, assistant_heads = 4, assistant_ffn = 256;
};

/* The assistant's own checkpoint, written into `<dir>/draft/`. Its geometry must satisfy the checks
 * in arch/gemma4_bf16's Gemma4Head::declare: the trunk's width and vocabulary, the sliding and
 * global head dims and kv counts, the window, and the trunk's rotary thetas -- including the
 * global partial-rotary factor that turns 64 dims into the trunk's 16 rotating ones. */
inline bool write_gemma4_assistant(const std::string& dir, const Gemma4& s) {
    Rng r(s.seed ^ 0xA5A5A5A5ull);
    const int64_t H  = s.hidden;                 /* backbone_hidden_size */
    const int64_t n  = s.assistant_width;
    const int64_t qs = s.assistant_heads * s.head_dim;
    const int64_t qg = s.assistant_heads * s.global_head_dim;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    nlohmann::ordered_json t;
    t["architectures"] = { "Gemma4AssistantForCausalLM" };
    t["model_type"] = "gemma4_assistant";
    t["backbone_hidden_size"] = H;
    t["hidden_size"] = n;
    t["num_hidden_layers"] = s.assistant_layers;
    t["num_attention_heads"] = s.assistant_heads;
    t["num_key_value_heads"] = s.kv_heads;
    t["num_global_key_value_heads"] = s.global_kv_heads;
    t["head_dim"] = s.head_dim;
    t["global_head_dim"] = s.global_head_dim;
    t["intermediate_size"] = s.assistant_ffn;
    t["sliding_window"] = s.window;
    t["vocab_size"] = s.vocab;
    t["rms_norm_eps"] = 1e-6;
    /* The trunk's layer_types order, cut to the head's layer count: layer 0 is sliding, which is
     * the class whose trunk layer it reads. */
    nlohmann::ordered_json types = nlohmann::ordered_json::array();
    for (int64_t l = 0; l < s.assistant_layers; ++l)
        types.push_back((l + 1) % s.interval == 0 ? "full_attention" : "sliding_attention");
    t["layer_types"] = types;
    t["rope_parameters"] = {
        { "sliding_attention", { { "rope_type", "default" }, { "rope_theta", 10000.0 } } },
        { "full_attention", { { "rope_type", "proportional" }, { "rope_theta", 1000000.0 },
                              { "partial_rotary_factor", 0.25 } } } };
    /* The head has no soft-cap of its own on the 31B; leave it out, so no cap op is declared. */
    t["tie_word_embeddings"] = false;
    t["max_position_embeddings"] = s.ctx;

    std::vector<Tensor> ts;
    ts.push_back(linear("model.embed_tokens.weight", s.vocab, n, r));
    ts.push_back(noise("model.norm.weight", { n }, r, 0.1f, 1.0f));
    ts.push_back(linear("pre_projection.weight", n, 2 * H, r));
    ts.push_back(linear("post_projection.weight", H, n, r));
    for (int64_t l = 0; l < s.assistant_layers; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const bool global = (l + 1) % s.interval == 0;
        const int64_t hd = global ? s.global_head_dim : s.head_dim;
        const int64_t q  = global ? qg : qs;
        ts.push_back(noise(p + "input_layernorm.weight", { n }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "self_attn.q_norm.weight", { hd }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "post_attention_layernorm.weight", { n }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "pre_feedforward_layernorm.weight", { n }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "post_feedforward_layernorm.weight", { n }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "layer_scalar", { 1 }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "self_attn.q_proj.weight", q, n, r));
        ts.push_back(linear(p + "self_attn.o_proj.weight", n, q, r, s.branch));
        ts.push_back(linear(p + "mlp.gate_proj.weight", s.assistant_ffn, n, r));
        ts.push_back(linear(p + "mlp.up_proj.weight", s.assistant_ffn, n, r));
        ts.push_back(linear(p + "mlp.down_proj.weight", n, s.assistant_ffn, r, s.branch));
    }
    return write_text(dir + "/config.json", t.dump(2)) &&
           write_safetensors(dir + "/model.safetensors", ts);
}

inline bool write_gemma4(const std::string& dir, const Gemma4& s) {
    Rng r(s.seed);
    const int64_t H = s.hidden;
    nlohmann::ordered_json types = nlohmann::ordered_json::array();
    for (int64_t l = 0; l < s.layers; ++l)
        types.push_back((l + 1) % s.interval == 0 ? "full_attention" : "sliding_attention");
    nlohmann::ordered_json t;
    t["model_type"] = "gemma4_text";
    t["vocab_size"] = s.vocab;
    t["hidden_size"] = H;
    t["intermediate_size"] = s.ffn;
    t["num_hidden_layers"] = s.layers;
    t["num_attention_heads"] = s.heads;
    t["num_key_value_heads"] = s.kv_heads;
    t["head_dim"] = s.head_dim;
    t["global_head_dim"] = s.global_head_dim;
    t["num_global_key_value_heads"] = s.global_kv_heads;
    t["attention_k_eq_v"] = true;
    t["sliding_window"] = s.window;
    t["layer_types"] = types;
    t["max_position_embeddings"] = s.ctx;
    t["rms_norm_eps"] = 1e-6;
    t["final_logit_softcapping"] = 30.0;
    t["num_kv_shared_layers"] = 0;
    t["hidden_size_per_layer_input"] = 0;
    t["hidden_activation"] = "gelu_pytorch_tanh";
    t["tie_word_embeddings"] = true;
    t["rope_parameters"] = {
        { "sliding_attention", { { "rope_type", "default" }, { "rope_theta", 10000.0 } } },
        { "full_attention", { { "rope_type", "proportional" }, { "rope_theta", 1000000.0 },
                              { "partial_rotary_factor", 0.25 } } } };
    nlohmann::ordered_json c;
    c["architectures"] = { "Gemma4ForConditionalGeneration" };
    c["model_type"] = "gemma4";
    c["text_config"] = t;
    c["tie_word_embeddings"] = true;
    c["eos_token_id"] = (int)kEndOfText;
    /* With the assistant head the checkpoint must also select the plugin that has one. The pair
     * (arch, quant) is matched exactly and `gemma4` names no quant, so the head's plugin is
     * `gemma4_bf16`, whose quant is `compressed-tensors`. A `quant_method` with no config group is
     * not imported (checkpoint.cpp:990), so the bf16 tensors here stay trivial and no weight is
     * read as quantised. */
    if (s.assistant)
        c["quantization_config"] = { { "quant_method", "compressed-tensors" } };

    const std::string m = "model.language_model.";
    std::vector<Tensor> ts;
    /* Tied, and scaled by sqrt(hidden) at the lookup. Each block's output passes a post-norm before
     * the residual add, so `branch` is that norm's gain. */
    ts.push_back(noise(m + "embed_tokens.weight", { s.vocab, H }, r, s.head));
    const float post = s.branch;
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        const bool global = (l + 1) % s.interval == 0;
        const int64_t hd = global ? s.global_head_dim : s.head_dim;
        const int64_t kv = (global ? s.global_kv_heads : s.kv_heads) * hd;
        ts.push_back(noise(p + "input_layernorm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "self_attn.q_proj.weight", s.heads * hd, H, r));
        ts.push_back(linear(p + "self_attn.k_proj.weight", kv, H, r));
        if (!global) ts.push_back(linear(p + "self_attn.v_proj.weight", kv, H, r));
        ts.push_back(linear(p + "self_attn.o_proj.weight", H, s.heads * hd, r));
        ts.push_back(noise(p + "self_attn.q_norm.weight", { hd }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "self_attn.k_norm.weight", { hd }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "post_attention_layernorm.weight", { H }, r, 0.1f * post, post));
        ts.push_back(noise(p + "pre_feedforward_layernorm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "mlp.gate_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.up_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.down_proj.weight", H, s.ffn, r));
        ts.push_back(noise(p + "post_feedforward_layernorm.weight", { H }, r, 0.1f * post, post));
        ts.push_back(noise(p + "layer_scalar", { 1 }, r, 0.1f, 1.0f));
    }
    /* Signed gains around 0, as for LFM2's tied head. */
    ts.push_back(noise(m + "norm.weight", { H }, r, 1.0f));
    return write_checkpoint(dir, c, ts) &&
           (!s.assistant || write_gemma4_assistant(dir + "/draft", s));
}

/* ------------------------------------------------------------------ llama: dense attention */

/* LLaMA (`llama`, served by llama_dense_fp8 once rad-convert has made its linears block fp8):
 * every width a multiple of the 128 x 128 scale block. */
struct Llama {
    int64_t layers = 2, hidden = 128, heads = 4, kv_heads = 2, head_dim = 64, ffn = 256;
    int64_t vocab = 320, ctx = 4096;
    float    branch = 0.5f, head = 2.0f;        /* see Gains */
    uint64_t seed = 4;
};

inline bool write_llama(const std::string& dir, const Llama& s) {
    Rng r(s.seed);
    const int64_t H = s.hidden, q = s.heads * s.head_dim, kv = s.kv_heads * s.head_dim;
    nlohmann::ordered_json c;
    c["architectures"] = { "LlamaForCausalLM" };
    c["model_type"] = "llama";
    c["vocab_size"] = s.vocab;
    c["hidden_size"] = H;
    c["intermediate_size"] = s.ffn;
    c["num_hidden_layers"] = s.layers;
    c["num_attention_heads"] = s.heads;
    c["num_key_value_heads"] = s.kv_heads;
    c["head_dim"] = s.head_dim;
    c["max_position_embeddings"] = s.ctx;
    c["rms_norm_eps"] = 1e-5;
    c["rope_parameters"] = { { "rope_type", "default" }, { "rope_theta", 10000.0 } };
    c["hidden_act"] = "silu";
    c["tie_word_embeddings"] = false;
    c["bos_token_id"] = (int)kImStart;
    c["eos_token_id"] = (int)kEndOfText;

    const std::string m = "model.";
    std::vector<Tensor> ts;
    ts.push_back(noise(m + "embed_tokens.weight", { s.vocab, H }, r, 1.0f));
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        ts.push_back(noise(p + "input_layernorm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "self_attn.q_proj.weight", q, H, r));
        ts.push_back(linear(p + "self_attn.k_proj.weight", kv, H, r));
        ts.push_back(linear(p + "self_attn.v_proj.weight", kv, H, r));
        ts.push_back(linear(p + "self_attn.o_proj.weight", H, q, r, s.branch));
        ts.push_back(noise(p + "post_attention_layernorm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "mlp.gate_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.up_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.down_proj.weight", H, s.ffn, r, s.branch));
    }
    ts.push_back(noise(m + "norm.weight", { H }, r, 0.1f, 1.0f));
    ts.push_back(linear("lm_head.weight", s.vocab, H, r, s.head));
    return write_checkpoint(dir, c, ts);
}

/* The tiny LLaMA as an int4 W4A16 release in compressed-tensors' pack-quantized form, as
 * llm-compressor writes it: every attention and MLP projection as X.weight_packed (the signed code
 * q plus 8, eight a word along K from the low bits), X.weight_scale (bf16, one per 128 of K) and,
 * with `asym`, X.weight_zero_point (the signed zero point plus 8, eight a word along N), the value
 * being (q - zero_point) * scale. Each scale is a power of two, so every value is exact in bf16 and
 * the release is the same numbers as a bf16 checkpoint of them. The head, the norms and the
 * embedding stay bf16, as such a release leaves them. */
inline bool write_llama_ct_int4(const std::string& dir, const Llama& s, bool asym) {
    if (!write_llama(dir, s)) return false;
    std::vector<Tensor> in = read_safetensors(dir + "/model.safetensors"), out;
    const int64_t G = 128;
    for (Tensor& t : in) {
        const std::string sfx = "_proj.weight";
        if (t.name.size() < sfx.size() ||
            t.name.compare(t.name.size() - sfx.size(), sfx.size(), sfx) != 0) {
            out.push_back(std::move(t));
            continue;
        }
        const std::string base = t.name.substr(0, t.name.size() - 7);
        const int64_t N = t.shape[0], K = t.shape[1], ng = K / G;
        Tensor p{ base + ".weight_packed", "I32", { N, K / 8 }, {} };
        Tensor sc{ base + ".weight_scale", "BF16", { N, ng }, {} };
        Tensor zp{ base + ".weight_zero_point", "I32", { (N + 7) / 8, ng }, {} };
        p.bytes.assign((size_t)(N * K / 2), 0);
        sc.bytes.resize((size_t)(N * ng * 2));
        zp.bytes.assign((size_t)((N + 7) / 8 * ng * 4), 0);
        auto put = [](std::vector<uint8_t>& b, int64_t word, int nib, int v) {
            uint32_t w;
            std::memcpy(&w, b.data() + word * 4, 4);
            w |= (uint32_t)(v & 15) << (4 * nib);
            std::memcpy(b.data() + word * 4, &w, 4);
        };
        for (int64_t n = 0; n < N; ++n)
            for (int64_t g = 0; g < ng; ++g) {
                float lo = 0, hi = 0;
                for (int64_t k = g * G; k < (g + 1) * G; ++k) {
                    const float v = rad_load_f32(t.bytes.data(), RAD_BF16, n * K + k);
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
                const float span = asym ? (hi - lo) / 15.0f : std::max(-lo, hi) / 7.0f;
                const float scale = std::exp2(std::ceil(std::log2(std::max(span, 1e-20f))));
                const int z = asym ? std::clamp((int)std::lround(-lo / scale) - 8, -8, 7) : 0;
                rad_store_f32(sc.bytes.data(), RAD_BF16, n * ng + g, scale);
                if (asym) put(zp.bytes, (n / 8) * ng + g, (int)(n % 8), z + 8);
                for (int64_t k = g * G; k < (g + 1) * G; ++k) {
                    const float v = rad_load_f32(t.bytes.data(), RAD_BF16, n * K + k);
                    const int q = std::clamp((int)std::lround(v / scale) + z, -8, 7);
                    put(p.bytes, n * (K / 8) + k / 8, (int)(k % 8), q + 8);
                }
            }
        out.push_back(std::move(p));
        out.push_back(std::move(sc));
        if (asym) out.push_back(std::move(zp));
    }
    nlohmann::ordered_json c = nlohmann::ordered_json::parse(read_text(dir + "/config.json"));
    c["quantization_config"] = {
        { "quant_method", "compressed-tensors" }, { "format", "pack-quantized" },
        { "config_groups",
          { { "group_0",
              { { "format", "pack-quantized" }, { "targets", { "Linear" } },
                { "input_activations", nullptr },
                { "weights",
                  { { "num_bits", 4 }, { "type", "int" }, { "symmetric", !asym },
                    { "strategy", "group" }, { "group_size", G }, { "dynamic", false },
                    { "actorder", asym ? nullptr : nlohmann::ordered_json("static") } } } } } } },
        { "ignore", { "lm_head" } } };
    return write_text(dir + "/config.json", c.dump(2)) &&
           write_safetensors(dir + "/model.safetensors", out);
}

/* The recipe that serves an int4 release on the 4-bit GEMMs: every imported grid unsigned with a
 * u4 zero plane and its scales in f16. */
inline const char* kInt4ImportRecipe = "blk.*.weight  import  codes=u4 zero=u4 scale=f16\n";

/* ------------------------------------------------------------------ qwen3asr: speech to text */

/* Qwen3-ASR (`qwen3_asr`, served by qwen3asr_bf16 from the checkpoint as it ships): a Whisper-style
 * audio tower, its projector and a dense Qwen3 decoder, with the release's names. The tower keeps
 * the release's framing (100 frames a second, 13 rows of it, attention in 104-row blocks) at
 * small widths; its processor_config.json is the feature extractor's. The clip's tokens are
 * specials after the tokenizer's own: kAsrAudioStart, kAsrAudioEnd, kAsrAudioPad, kAsrText. */
enum : int32_t { kAsrAudioStart = kTokenizerSize, kAsrAudioEnd, kAsrAudioPad, kAsrText };

struct Qwen3Asr {
    int64_t layers = 2, hidden = 64, heads = 4, kv_heads = 2, head_dim = 32, ffn = 128;
    int64_t vocab = 320, ctx = 4096;
    int64_t a_layers = 2, a_dim = 64, a_heads = 2, a_ffn = 128, a_channels = 8, mels = 32;
    float    branch = 0.5f, head = 2.0f;        /* see Gains */
    uint64_t seed = 7;
};

inline bool write_qwen3asr(const std::string& dir, const Qwen3Asr& s) {
    Rng r(s.seed);
    const int64_t H = s.hidden, q = s.heads * s.head_dim, kv = s.kv_heads * s.head_dim;
    const int64_t A = s.a_dim, C = s.a_channels;
    const int64_t f3 = ((((s.mels + 1) / 2) + 1) / 2 + 1) / 2;
    nlohmann::ordered_json c;
    c["architectures"] = { "Qwen3ASRForConditionalGeneration" };
    c["model_type"] = "qwen3_asr";
    c["audio_token_id"] = (int)kAsrAudioPad;
    c["audio_config"] = { { "model_type", "qwen3_asr_encoder" }, { "d_model", A },
                          { "encoder_attention_heads", s.a_heads },
                          { "num_key_value_heads", s.a_heads }, { "encoder_ffn_dim", s.a_ffn },
                          { "downsample_hidden_size", C }, { "num_mel_bins", s.mels },
                          { "encoder_layers", s.a_layers }, { "n_window", 50 },
                          { "n_window_infer", 800 }, { "output_dim", H },
                          { "activation_function", "gelu" }, { "scale_embedding", false } };
    c["text_config"] = { { "model_type", "qwen3" }, { "vocab_size", s.vocab },
                         { "hidden_size", H }, { "intermediate_size", s.ffn },
                         { "num_hidden_layers", s.layers }, { "num_attention_heads", s.heads },
                         { "num_key_value_heads", s.kv_heads }, { "head_dim", s.head_dim },
                         { "max_position_embeddings", s.ctx }, { "rms_norm_eps", 1e-6 },
                         { "rope_theta", 1000000.0 }, { "hidden_act", "silu" },
                         { "tie_word_embeddings", true } };
    c["eos_token_id"] = (int)kImEnd;

    std::vector<Tensor> ts;
    const std::string t = "model.audio_tower.";
    ts.push_back(noise(t + "conv2d1.weight", { C, 1, 3, 3 }, r, 0.5f));
    ts.push_back(noise(t + "conv2d1.bias", { C }, r, 0.1f));
    ts.push_back(noise(t + "conv2d2.weight", { C, C, 3, 3 }, r, std::sqrt(3.0f / (float)(9 * C))));
    ts.push_back(noise(t + "conv2d2.bias", { C }, r, 0.1f));
    ts.push_back(noise(t + "conv2d3.weight", { C, C, 3, 3 }, r, std::sqrt(3.0f / (float)(9 * C))));
    ts.push_back(noise(t + "conv2d3.bias", { C }, r, 0.1f));
    ts.push_back(linear(t + "conv_out.weight", A, C * f3, r));
    for (int64_t l = 0; l < s.a_layers; ++l) {
        const std::string p = t + "layers." + std::to_string(l) + ".";
        ts.push_back(noise(p + "self_attn_layer_norm.weight", { A }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "self_attn_layer_norm.bias", { A }, r, 0.1f));
        for (const char* w : { "q_proj", "k_proj", "v_proj" }) {
            ts.push_back(linear(p + "self_attn." + w + ".weight", A, A, r));
            ts.push_back(noise(p + "self_attn." + w + ".bias", { A }, r, 0.1f));
        }
        ts.push_back(linear(p + "self_attn.out_proj.weight", A, A, r, s.branch));
        ts.push_back(noise(p + "self_attn.out_proj.bias", { A }, r, 0.1f));
        ts.push_back(noise(p + "final_layer_norm.weight", { A }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "final_layer_norm.bias", { A }, r, 0.1f));
        ts.push_back(linear(p + "fc1.weight", s.a_ffn, A, r));
        ts.push_back(noise(p + "fc1.bias", { s.a_ffn }, r, 0.1f));
        ts.push_back(linear(p + "fc2.weight", A, s.a_ffn, r, s.branch));
        ts.push_back(noise(p + "fc2.bias", { A }, r, 0.1f));
    }
    ts.push_back(noise(t + "ln_post.weight", { A }, r, 0.1f, 1.0f));
    ts.push_back(noise(t + "ln_post.bias", { A }, r, 0.1f));
    const std::string pj = "model.multi_modal_projector.";
    ts.push_back(linear(pj + "linear_1.weight", A, A, r));
    ts.push_back(noise(pj + "linear_1.bias", { A }, r, 0.1f));
    ts.push_back(linear(pj + "linear_2.weight", H, A, r));
    ts.push_back(noise(pj + "linear_2.bias", { H }, r, 0.1f));

    const std::string m = "model.language_model.";
    ts.push_back(noise(m + "embed_tokens.weight", { s.vocab, H }, r, 1.0f));
    for (int64_t l = 0; l < s.layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        ts.push_back(noise(p + "input_layernorm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "self_attn.q_proj.weight", q, H, r));
        ts.push_back(linear(p + "self_attn.k_proj.weight", kv, H, r));
        ts.push_back(linear(p + "self_attn.v_proj.weight", kv, H, r));
        ts.push_back(linear(p + "self_attn.o_proj.weight", H, q, r, s.branch));
        ts.push_back(noise(p + "self_attn.q_norm.weight", { s.head_dim }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "self_attn.k_norm.weight", { s.head_dim }, r, 0.1f, 1.0f));
        ts.push_back(noise(p + "post_attention_layernorm.weight", { H }, r, 0.1f, 1.0f));
        ts.push_back(linear(p + "mlp.gate_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.up_proj.weight", s.ffn, H, r));
        ts.push_back(linear(p + "mlp.down_proj.weight", H, s.ffn, r, s.branch));
    }
    ts.push_back(noise(m + "norm.weight", { H }, r, 0.1f, 1.0f));

    nlohmann::ordered_json fe;
    fe["feature_extractor"] = { { "feature_extractor_type", "Qwen3ASRFeatureExtractor" },
                                { "feature_size", s.mels }, { "hop_length", 160 },
                                { "n_fft", 400 }, { "sampling_rate", 16000 }, { "dither", 0.0 },
                                { "n_window", 50 } };
    return write_checkpoint(dir, c, ts,
                            { "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<asr_text>" }) &&
           write_text(dir + "/processor_config.json", fe.dump(2));
}

/* ------------------------------------------------------------------ PEFT LoRA adapters */

/* A PEFT LoRA adapter for the checkpoint a writer above put in `base`: every 2-D
 * `<module>.weight` whose leaf is in `targets` gets bf16 noise lora_A [r, in] and lora_B [out, r],
 * with adapter_config.json's `r` and `lora_alpha`. The delta (alpha / r) * B.A is about `gain`
 * times the weight's own spread. With `merged` set, the checkpoint with every delta folded in
 * (f32 sum, rounded to bf16) is written there too: what the adapter served at runtime must match
 * within rounding. */
inline bool write_lora(const std::string& dir, const std::string& base,
                       const std::vector<std::string>& targets, int64_t r, double alpha,
                       uint64_t seed, float gain, const std::string& merged = "") {
    Rng rng(seed);
    std::vector<Tensor> ws = read_safetensors(base + "/model.safetensors");
    if (ws.empty()) return false;
    const float scale = (float)(alpha / (double)r);
    std::vector<Tensor> ad;
    nlohmann::ordered_json leaves = nlohmann::ordered_json::array();
    for (Tensor& w : ws) {
        const std::string sfx = ".weight";
        if (w.shape.size() != 2 || w.dtype != "BF16" || w.name.size() <= sfx.size() ||
            w.name.compare(w.name.size() - sfx.size(), sfx.size(), sfx) != 0)
            continue;
        const std::string module = w.name.substr(0, w.name.size() - sfx.size());
        const std::string leaf = module.substr(module.rfind('.') + 1);
        bool hit = false;
        for (const std::string& t : targets) hit = hit || t == leaf;
        if (!hit) continue;
        const int64_t n_out = w.shape[0], n_in = w.shape[1];
        Tensor A = noise("base_model.model." + module + ".lora_A.weight", { r, n_in }, rng,
                         std::sqrt(3.0f / (float)n_in));
        Tensor B = noise("base_model.model." + module + ".lora_B.weight", { n_out, r }, rng,
                         gain * std::sqrt(3.0f / (float)r) / scale);
        if (!merged.empty()) {
            for (int64_t o = 0; o < n_out; ++o)
                for (int64_t i = 0; i < n_in; ++i) {
                    float d = 0.0f;
                    for (int64_t j = 0; j < r; ++j)
                        d += rad_load_f32(B.bytes.data(), RAD_BF16, o * r + j) *
                             rad_load_f32(A.bytes.data(), RAD_BF16, j * n_in + i);
                    const float v = rad_load_f32(w.bytes.data(), RAD_BF16, o * n_in + i);
                    rad_store_f32(w.bytes.data(), RAD_BF16, o * n_in + i, v + scale * d);
                }
        }
        ad.push_back(std::move(A));
        ad.push_back(std::move(B));
    }
    for (const std::string& t : targets) leaves.push_back(t);
    nlohmann::ordered_json c;
    c["peft_type"] = "LORA";
    c["base_model_name_or_path"] = base;
    c["r"] = r;
    c["lora_alpha"] = alpha;
    c["target_modules"] = leaves;
    c["bias"] = "none";
    c["use_dora"] = false;
    c["use_rslora"] = false;
    c["modules_to_save"] = nullptr;
    if (ad.empty() || ::mkdir(dir.c_str(), 0755) != 0 ||
        !write_text(dir + "/adapter_config.json", c.dump(2)) ||
        !write_safetensors(dir + "/adapter_model.safetensors", ad))
        return false;
    if (merged.empty()) return true;
    return ::mkdir(merged.c_str(), 0755) == 0 &&
           write_text(merged + "/config.json", read_text(base + "/config.json")) &&
           write_text(merged + "/tokenizer.json", read_text(base + "/tokenizer.json")) &&
           write_safetensors(merged + "/model.safetensors", ws);
}

/* The rad-convert recipe that makes llama_dense_fp8's linears block fp8, as
 * data/recipes/minicpm5-2b-dspark.recipe does for the trunk. */
inline const char* kLlamaFp8Recipe =
    "blk.*.attn_qkv.weight     rtn codes=fp8_e4m3 block=128x128 scale=bf16\n"
    "blk.*.attn_output.weight  rtn codes=fp8_e4m3 block=128x128 scale=bf16\n"
    "blk.*.ffn_gate_up.weight  rtn codes=fp8_e4m3 block=128x128 scale=bf16\n"
    "blk.*.ffn_down.weight     rtn codes=fp8_e4m3 block=128x128 scale=bf16\n";

/* The same trunk as OCP-MXFP4: e2m1 codes with an E8M0 exponent per 32 of K, read by gemm_nt_q at
 * mxfp4a8 against the E4M3 activation pair scaled per 128. */
inline const char* kLlamaMxfp4Recipe =
    "blk.*.attn_qkv.weight     rtn codes=fp4_e2m1 group=32 scale=e8m0\n"
    "blk.*.attn_output.weight  rtn codes=fp4_e2m1 group=32 scale=e8m0\n"
    "blk.*.ffn_gate_up.weight  rtn codes=fp4_e2m1 group=32 scale=e8m0\n"
    "blk.*.ffn_down.weight     rtn codes=fp4_e2m1 group=32 scale=e8m0\n";

/* ------------------------------------------------------------------ Qwen-Image 2.1 */

/* Qwen-Image 2.1 (`QwenImage21Pipeline`, served by qwenimage21) as a diffusers pipeline directory:
 * model_index.json, a Qwen3-VL text encoder with no vision tower, the single-stream DiT, the
 * residual 2-D VAE, the scheduler's config and the tokenizer. The latent keeps the model's 64
 * channels (the plugin requires them); every other width is cut. The DiT's 64-wide heads take the
 * bf16 attention path of a host library that has one. */
struct QwenImage {
    int64_t te_layers = 2, te_hidden = 64, te_heads = 4, te_kv_heads = 2, te_head_dim = 16;
    int64_t te_ffn = 128, vocab = 320;
    int64_t dit_layers = 2, dit_heads = 2, dit_head_dim = 64, mlp_ratio = 2;
    int64_t vae_dim = 8, vae_res = 1;
    float    branch = 0.5f;                     /* see Gains */
    uint64_t seed = 5;
};

inline bool write_qwen_image(const std::string& dir, const QwenImage& s) {
    Rng r(s.seed);
    using oj = nlohmann::ordered_json;
    for (const char* sub : { "text_encoder", "transformer", "vae", "scheduler", "tokenizer" })
        if (mkdir((dir + "/" + sub).c_str(), 0755) != 0 && errno != EEXIST) return false;
    oj index;
    index["_class_name"] = "QwenImage21Pipeline";
    index["_diffusers_version"] = "0.36.0";
    if (!write_text(dir + "/model_index.json", index.dump(2))) return false;

    /* ---- the text encoder: Qwen3-VL's language model, interleaved M-RoPE */
    const int64_t H = s.te_hidden, q = s.te_heads * s.te_head_dim, kv = s.te_kv_heads * s.te_head_dim;
    oj tc;
    tc["model_type"] = "qwen3_vl_text";
    tc["vocab_size"] = s.vocab;
    tc["hidden_size"] = H;
    tc["intermediate_size"] = s.te_ffn;
    tc["num_hidden_layers"] = s.te_layers;
    tc["num_attention_heads"] = s.te_heads;
    tc["num_key_value_heads"] = s.te_kv_heads;
    tc["head_dim"] = s.te_head_dim;
    tc["max_position_embeddings"] = 32768;
    tc["rms_norm_eps"] = 1e-6;
    tc["rope_theta"] = 5000000.0;
    const int64_t half = s.te_head_dim / 2, sec = half / 3;
    tc["rope_scaling"] = { { "mrope_interleaved", true },
                           { "mrope_section", { half - 2 * sec, sec, sec } },
                           { "rope_type", "default" } };
    oj te;
    te["architectures"] = { "Qwen3VLForConditionalGeneration" };
    te["model_type"] = "qwen3_vl";
    te["text_config"] = tc;
    te["eos_token_id"] = (int)kImEnd;
    std::vector<Tensor> tt;
    const std::string m = "model.language_model.";
    tt.push_back(noise(m + "embed_tokens.weight", { s.vocab, H }, r, 1.0f));
    for (int64_t l = 0; l < s.te_layers; ++l) {
        const std::string p = m + "layers." + std::to_string(l) + ".";
        tt.push_back(noise(p + "input_layernorm.weight", { H }, r, 0.1f, 1.0f));
        tt.push_back(linear(p + "self_attn.q_proj.weight", q, H, r));
        tt.push_back(linear(p + "self_attn.k_proj.weight", kv, H, r));
        tt.push_back(linear(p + "self_attn.v_proj.weight", kv, H, r));
        tt.push_back(linear(p + "self_attn.o_proj.weight", H, q, r, s.branch));
        tt.push_back(noise(p + "self_attn.q_norm.weight", { s.te_head_dim }, r, 0.1f, 1.0f));
        tt.push_back(noise(p + "self_attn.k_norm.weight", { s.te_head_dim }, r, 0.1f, 1.0f));
        tt.push_back(noise(p + "post_attention_layernorm.weight", { H }, r, 0.1f, 1.0f));
        tt.push_back(linear(p + "mlp.gate_proj.weight", s.te_ffn, H, r));
        tt.push_back(linear(p + "mlp.up_proj.weight", s.te_ffn, H, r));
        tt.push_back(linear(p + "mlp.down_proj.weight", H, s.te_ffn, r, s.branch));
    }
    tt.push_back(noise(m + "norm.weight", { H }, r, 0.1f, 1.0f));
    if (!write_text(dir + "/text_encoder/config.json", te.dump(2)) ||
        !write_safetensors(dir + "/text_encoder/model.safetensors", tt))
        return false;

    /* ---- the DiT */
    const int64_t hd = s.dit_head_dim, D = hd * s.dit_heads, ff = D * s.mlp_ratio, C = 64;
    const int64_t ax0 = hd / 4, ax1 = (hd - ax0) / 2;
    oj tr;
    tr["_class_name"] = "QwenImageTransformer2DModel";
    tr["patch_size"] = 1;
    tr["causal_condition"] = 1;
    tr["in_channels"] = C;
    tr["out_channels"] = C;
    tr["num_layers"] = s.dit_layers;
    tr["attention_head_dim"] = hd;
    tr["num_attention_heads"] = s.dit_heads;
    tr["mlp_ratio"] = s.mlp_ratio;
    tr["context_in_dim"] = H;
    tr["axes_dims_rope"] = { ax0, ax1, hd - ax0 - ax1 };
    tr["eps"] = 1e-6;
    std::vector<Tensor> dt;
    dt.push_back(linear("time_text_embed.timestep_embedder.linear_1.weight", D, 256, r));
    dt.push_back(linear("time_text_embed.timestep_embedder.linear_2.weight", D, D, r));
    dt.push_back(linear("modulation.1.weight", 4 * D, D, r, 0.5f));
    dt.push_back(linear("norm_out.linear.weight", D, D, r, 0.5f));
    dt.push_back(noise("txt_in.text_norm.weight", { H }, r, 0.1f));
    dt.push_back(linear("txt_in.in_layer.weight", D, H, r));
    dt.push_back(linear("txt_in.out_layer.weight", D, D, r));
    dt.push_back(linear("img_in.weight", D, C, r));
    dt.push_back(linear("proj_out.weight", C, D, r));
    for (int64_t l = 0; l < s.dit_layers; ++l) {
        const std::string p = "transformer_blocks." + std::to_string(l) + ".";
        dt.push_back(noise(p + "attn.norm_q.weight", { hd }, r, 0.1f, 1.0f));
        dt.push_back(noise(p + "attn.norm_k.weight", { hd }, r, 0.1f, 1.0f));
        dt.push_back(linear(p + "attn.to_q.weight", D, D, r));
        dt.push_back(linear(p + "attn.to_k.weight", D, D, r));
        dt.push_back(linear(p + "attn.to_v.weight", D, D, r));
        dt.push_back(linear(p + "attn.to_out.0.weight", D, D, r, s.branch));
        dt.push_back(linear(p + "img_mlp.gate_layer.weight", ff, D, r));
        dt.push_back(linear(p + "img_mlp.proj.weight", ff, D, r));
        dt.push_back(linear(p + "img_mlp.out.weight", D, ff, r, s.branch));
    }
    if (!write_text(dir + "/transformer/config.json", tr.dump(2)) ||
        !write_safetensors(dir + "/transformer/diffusion_pytorch_model.safetensors", dt))
        return false;

    /* ---- the VAE: every level s.vae_dim wide, Wan 2.2's temporal flags */
    const int64_t V = s.vae_dim;
    oj va;
    va["_class_name"] = "AutoencoderKLQwenImage21";
    va["is_residual"] = true;
    va["z_dim"] = C;
    va["base_dim"] = V;
    va["decoder_base_dim"] = V;
    va["dim_mult"] = { 1, 1, 1, 1, 1 };
    va["num_res_blocks"] = s.vae_res;
    va["temperal_downsample"] = { false, true, true, false };
    va["in_channels"] = 4;
    va["out_channels"] = 4;
    oj mean = oj::array(), sd = oj::array();
    for (int64_t i = 0; i < C; ++i) {
        mean.push_back(0.02 * (double)(i % 7) - 0.06);
        sd.push_back(1.5 + 0.05 * (double)(i % 5));
    }
    va["latents_mean"] = mean;
    va["latents_std"] = sd;
    std::vector<Tensor> vt;
    auto conv = [&](const std::string& n, int64_t co, int64_t ci, int64_t k, float gain = 1.0f) {
        vt.push_back(noise(n + ".weight", { co, ci, k, k }, r,
                           gain * std::sqrt(3.0f / (float)(ci * k * k))));
        vt.push_back(noise(n + ".bias", { co }, r, 0.05f));
    };
    auto gamma = [&](const std::string& n, int64_t c, int rank) {
        if (rank == 4) vt.push_back(noise(n + ".gamma", { c, 1, 1, 1 }, r, 0.1f, 1.0f));
        else           vt.push_back(noise(n + ".gamma", { c, 1, 1 }, r, 0.1f, 1.0f));
    };
    auto res = [&](const std::string& n, int64_t ci, int64_t co) {
        if (ci != co) conv(n + ".conv_shortcut", co, ci, 1);
        gamma(n + ".norm1", ci, 4);
        conv(n + ".conv1", co, ci, 3);
        gamma(n + ".norm2", co, 4);
        conv(n + ".conv2", co, co, 3, s.branch);
    };
    auto attn = [&](const std::string& n, int64_t c) {
        gamma(n + ".norm", c, 3);
        conv(n + ".to_qkv", 3 * c, c, 1);
        conv(n + ".proj", c, c, 1, s.branch);
    };
    conv("encoder.conv_in", V, 4, 3);
    for (int i = 0; i < 5; ++i) {
        const std::string b = "encoder.down_blocks." + std::to_string(i);
        for (int64_t k = 0; k < s.vae_res; ++k) res(b + ".resnets." + std::to_string(k), V, V);
        if (i != 4) conv(b + ".downsampler.resample.1", V, V, 3);
    }
    res("encoder.mid_block.resnets.0", V, V);
    attn("encoder.mid_block.attentions.0", V);
    res("encoder.mid_block.resnets.1", V, V);
    gamma("encoder.norm_out", V, 4);
    conv("encoder.conv_out", 2 * C, V, 3);
    conv("quant_conv", 2 * C, 2 * C, 1);
    conv("post_quant_conv", C, C, 1);
    conv("decoder.conv_in", V, C, 3);
    res("decoder.mid_block.resnets.0", V, V);
    attn("decoder.mid_block.attentions.0", V);
    res("decoder.mid_block.resnets.1", V, V);
    for (int i = 0; i < 5; ++i) {
        const std::string b = "decoder.up_blocks." + std::to_string(i);
        for (int64_t k = 0; k < s.vae_res + 1; ++k) res(b + ".resnets." + std::to_string(k), V, V);
        if (i != 4) conv(b + ".upsampler.resample.1", V, V, 3);
    }
    gamma("decoder.norm_out", V, 4);
    conv("decoder.conv_out", 4, V, 3);
    if (!write_text(dir + "/vae/config.json", va.dump(2)) ||
        !write_safetensors(dir + "/vae/diffusion_pytorch_model.safetensors", vt))
        return false;

    oj sch;
    sch["_class_name"] = "FlowMatchEulerDiscreteScheduler";
    sch["shift"] = 1.0;
    sch["use_dynamic_shifting"] = false;
    return write_text(dir + "/scheduler/scheduler_config.json", sch.dump(2)) &&
           write_text(dir + "/tokenizer/tokenizer.json", tokenizer_json());
}

/* The rad-convert recipe that makes the DiT's linears block fp8, as a shipped fp8 container holds
 * them; the plugin then takes the fp8 forms of its norms and attention scores. The text encoder's
 * 64-wide linears are no whole 128 x 128 block and stay bf16. */
inline const char* kQwenImageFp8Recipe =
    "dit.blk.*.attn_qkv.weight     rtn codes=fp8_e4m3 block=128x128 scale=bf16\n"
    "dit.blk.*.attn_output.weight  rtn codes=fp8_e4m3 block=128x128 scale=bf16\n"
    "dit.blk.*.ffn_gate_up.weight  rtn codes=fp8_e4m3 block=128x128 scale=bf16\n"
    "dit.blk.*.ffn_down.weight     rtn codes=fp8_e4m3 block=128x128 scale=bf16\n";

/* The lm head quantised as the shipped containers hold it, one rule each, after the trunk's:
 * block fp8 (the 27B family, data/recipes/q38-27b-fp8-df2.recipe) and int8 with a bf16 scale per
 * row and 128 of K (Flash-Next's, Gemma 4 31B's). Declared over all its planes, the head is joined
 * on the host (abi/rad_planes.h). */
inline const char* kLlamaFp8HeadRule =
    "output.weight             rtn codes=fp8_e4m3 block=128x128 scale=bf16\n";
inline const char* kLlamaI8HeadRule =
    "output.weight             rtn codes=i8 group=128 scale=bf16 clamp=sym\n";

/* ------------------------------------------------------------------ running the tools */

/* The current environment with `set` applied in order: "NAME=value" sets NAME, a bare "NAME"
 * removes it, and a later entry for a name overrides an earlier one. */
inline std::vector<std::string> environment(const std::vector<std::string>& set) {
    std::vector<std::string> env;
    for (char** e = environ; *e; ++e) env.push_back(*e);
    for (const std::string& s : set) {
        const size_t eq = s.find('=');
        const std::string name = (eq == std::string::npos ? s : s.substr(0, eq)) + "=";
        std::vector<std::string> kept;
        for (const std::string& kv : env)
            if (kv.compare(0, name.size(), name) != 0) kept.push_back(kv);
        if (eq != std::string::npos) kept.push_back(s);
        env.swap(kept);
    }
    return env;
}

/* A child with stdout and stderr on one pipe. */
struct Child {
    pid_t pid = -1;
    int   fd = -1;

    bool start(const std::vector<std::string>& argv, const std::vector<std::string>& env_set) {
        int p[2];
        if (pipe2(p, O_CLOEXEC) != 0) return false;
        std::vector<std::string> env = environment(env_set);
        std::vector<char*> av, ev;
        for (const std::string& a : argv) av.push_back(const_cast<char*>(a.c_str()));
        for (const std::string& e : env) ev.push_back(const_cast<char*>(e.c_str()));
        av.push_back(nullptr);
        ev.push_back(nullptr);
        pid = fork();
        if (pid < 0) { close(p[0]); close(p[1]); return false; }
        if (pid == 0) {
            dup2(p[1], 1);
            dup2(p[1], 2);
            execve(av[0], av.data(), ev.data());
            _exit(127);
        }
        close(p[1]);
        fd = p[0];
        return true;
    }

    /* Appends what the child writes to `out` until it closes the pipe, or for at most
     * `timeout_s` seconds when that is positive; false when the time ran out. */
    bool read_all(std::string* out, int timeout_s = 0) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        char buf[4096];
        for (;;) {
            if (timeout_s > 0) {
                const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - std::chrono::steady_clock::now()).count();
                pollfd p{ fd, POLLIN, 0 };
                if (left <= 0 || poll(&p, 1, (int)left) == 0) return false;
            }
            const ssize_t n = ::read(fd, buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return true;
            out->append(buf, (size_t)n);
        }
    }

    /* The exit status (128 + signal for a signalled child), after reading what is left. A child
     * still writing after `timeout_s` seconds (when positive) is killed, and the status is 124,
     * as timeout(1) reports it. */
    int wait(std::string* out, int timeout_s = 0) {
        bool late = false;
        if (fd >= 0) {
            late = !read_all(out, timeout_s);
            if (late) kill(pid, SIGKILL);
            close(fd);
            fd = -1;
        }
        int st = 0;
        while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
        pid = -1;
        if (late) return 124;
        return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    }

    ~Child() {
        if (pid > 0) { kill(pid, SIGKILL); std::string rest; wait(&rest); }
    }
};

/* Runs `argv` to completion, or for at most `timeout_s` seconds when that is positive (status
 * 124); returns its exit status and leaves its output in `out`. */
inline int run(const std::vector<std::string>& argv, const std::vector<std::string>& env_set,
               std::string* out, int timeout_s = 0) {
    Child c;
    if (!c.start(argv, env_set)) return -1;
    return c.wait(out, timeout_s);
}

}  /* namespace tiny */
