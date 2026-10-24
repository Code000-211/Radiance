/* checkpoint.cpp -- see checkpoint.h. */
#include "checkpoint.h"
#include "rad_plugin.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <unistd.h>

namespace rad {

/* ================================================================== metadata */

void MetaOwn::set(const std::string& k, const std::string& v) {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) { vals[i] = v; return; }
    kv(k, v);
}

bool MetaOwn::has(const std::string& k) const {
    return std::find(keys.begin(), keys.end(), k) != keys.end();
}

const char* MetaOwn::get(const std::string& k) const {
    for (size_t i = 0; i < keys.size(); ++i)
        if (keys[i] == k) return vals[i].c_str();
    return nullptr;
}

void MetaOwn::finish() {
    kp.clear();
    vp.clear();
    for (auto& s : keys) kp.push_back(s.c_str());
    for (auto& s : vals) vp.push_back(s.c_str());
    m.arch_id = arch.c_str();
    m.name    = name.c_str();
    m.quant   = quant.c_str();
    m.n_kv    = (int)keys.size();
    m.kv_key  = kp.data();
    m.kv_val  = vp.data();
}

/* ---------------------------------------------------------------- config.json
 *
 * A safetensors checkpoint carries its architecture and its dimensions in config.json, and both
 * have to reach the plugin: RadModelMeta's typed fields are what declare sizes everything from, so
 * an unparsed config means a graph built on zeros. Parsing it is the core's job and not the
 * architecture plugin's, because the plugin is handed RadModelMeta and never sees the file.
 *
 * Two shapes exist and both are handled. A text-only model puts its dimensions at the top level; a
 * multimodal one nests them under `text_config` and puts the tower under `vision_config`. The
 * nested case is not a special case here: EVERY key is flattened into the free-form list with a
 * dotted path, and the typed fields are then read from `text_config.X` falling back to `X`. So a
 * plugin that wants `vision_config.depth` finds it under exactly that name whatever the nesting.
 */
static void flatten_json(const nlohmann::json& j, const std::string& prefix, MetaOwn* out) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
        const nlohmann::json& v = it.value();
        if (v.is_object())            flatten_json(v, key, out);
        else if (v.is_boolean())      out->kv(key, v.get<bool>() ? "1" : "0");
        else if (v.is_number_integer()) out->kv(key, fmt("%lld", (long long)v.get<int64_t>()));
        else if (v.is_number())       out->kv(key, fmt("%.17g", v.get<double>()));
        else if (v.is_string())       out->kv(key, v.get<std::string>());
        else if (v.is_array()) {
            /* An array of scalars becomes a space-separated string: mrope_section "11 11 10",
             * layer_types "linear_attention full_attention ...". A plugin splits it. Arrays of
             * objects are not flattened -- nothing in a config uses one for a dimension. */
            std::string s;
            bool ok = true;
            for (const auto& e : v) {
                if (!s.empty()) s += ' ';
                if (e.is_number_integer()) s += fmt("%lld", (long long)e.get<int64_t>());
                else if (e.is_number())    s += fmt("%.17g", e.get<double>());
                else if (e.is_string())    s += e.get<std::string>();
                else { ok = false; break; }
            }
            if (ok) out->kv(key, s);
        }
    }
}

/* A SECOND config.json, flattened under a prefix and nothing else.
 *
 * None of meta_from_config_json's typed extraction applies: a drafter's `num_hidden_layers` is 5
 * and the target's is 64, and writing the drafter's into RadModelMeta would size the trunk from
 * the draft head. So the whole of it goes into the free-form table with `draft.` in front, which
 * is where an architecture plugin reads a dimension the struct does not name anyway. The
 * `text_config` aliasing is deliberately not repeated -- a drafter is a text model by
 * construction, and aliasing here would collide with the target's bare keys. */
int meta_merge_config_json(const std::string& dir, const std::string& prefix, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A draft checkpoint carries its own geometry there -- the "
                "block size it was trained for, which trunk layers it reads, the selector's rank "
                "-- and none of it is derivable from the target.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) { RAD_ERR("%s: %s", path.c_str(), e.what()); return RAD_E_FORMAT; }
    flatten_json(j, prefix, out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- generation_config.json
 *
 * HOW THE CHECKPOINT SAYS IT WANTS TO BE SAMPLED, carried into the container so that serving it
 * does not depend on a file nobody copied. It is a separate file from config.json because it is a
 * separate thing: config.json is the architecture and is required, this is the sampler and is
 * advisory, and a checkpoint that omits it is read without complaint.
 *
 * It matters more than "advisory" suggests. A model tuned against a truncated tail and served
 * without one will occasionally draw a token from far down its distribution, and end-of-turn is
 * one such token -- so the visible symptom is not bad prose, it is a reply that stops early.
 *
 * Flattened under `generation.` rather than merged into the bare keys: `temperature` and `top_k`
 * are common enough words that a bare spelling would collide with an architecture that happens to
 * use one, and the prefix is what tells a reader which file a value came from. */
int meta_merge_generation_config(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/generation_config.json";
    std::ifstream f(path);
    if (!f) return RAD_OK;               /* absent is not an error; nothing is written */
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_WARN("%s: %s; no sampling defaults are carried", path.c_str(), e.what());
        return RAD_OK;
    }
    if (!j.is_object()) {
        RAD_WARN("%s: not a JSON object; no sampling defaults are carried", path.c_str());
        return RAD_OK;
    }
    flatten_json(j, "generation", out);
    return RAD_OK;
}

/* ---------------------------------------------------------------- the preprocessor configs
 *
 * HOW AN IMAGE AND A VIDEO BECOME PATCHES, carried for the reason the sampling defaults are:
 * serving the model must not depend on a file nobody copied. The engine's processor (core/mm)
 * reads the pixel bands, the normalisation and the frame rate from here; the geometry the encoder
 * itself needs is already in config.json's vision_config. Flattened under their own prefixes --
 * `size.shortest_edge` means two different things in the two files. Absent is not an error: a
 * text model has neither. */
int meta_merge_preprocessor_configs(const std::string& dir, MetaOwn* out) {
    static const struct { const char* file; const char* prefix; } kFiles[] = {
        { "preprocessor_config.json", "preprocessor" },
        { "video_preprocessor_config.json", "video_preprocessor" },
    };
    for (const auto& f : kFiles) {
        const std::string path = dir + "/" + f.file;
        std::ifstream in(path);
        if (!in) continue;
        nlohmann::json j;
        try { in >> j; }
        catch (const std::exception& e) {
            RAD_WARN("%s: %s; it is not carried", path.c_str(), e.what());
            continue;
        }
        if (j.is_object()) flatten_json(j, f.prefix, out);
    }
    return RAD_OK;
}

int meta_from_config_json(const std::string& dir, MetaOwn* out) {
    const std::string path = dir + "/config.json";
    std::ifstream f(path);
    if (!f) {
        RAD_ERR("%s: no config.json. A safetensors checkpoint carries its architecture and its "
                "dimensions there; without it nothing can be sized.", dir.c_str());
        return RAD_E_NOTFOUND;
    }
    nlohmann::json j;
    try { f >> j; }
    catch (const std::exception& e) {
        RAD_ERR("%s: %s", path.c_str(), e.what());
        return RAD_E_FORMAT;
    }

    flatten_json(j, "", out);

    /* Alias text_config.X to a bare X wherever no bare X already exists.
     *
     * A multimodal checkpoint nests the language model's dimensions under `text_config` and a
     * text-only one does not; that nesting is the ONLY difference between the two files. The
     * struct fields below already prefer the prefixed spelling, but the free-form key/value table
     * is handed to the plugin verbatim, so without this a plugin reading `partial_rotary_factor`
     * finds nothing on exactly the checkpoints that have a vision tower -- and the failure is at
     * declare, after the architecture has already been matched. One rule, applied once here, is
     * better than every plugin having to know what a vision tower does to its config.
     *
     * The bare key wins where both exist: a top-level value is the file's own answer, and the
     * prefixed one is the aliased fallback. */
    {
        const size_t n = out->keys.size();
        for (size_t i = 0; i < n; ++i) {
            const std::string& k = out->keys[i];
            if (k.rfind("text_config.", 0) != 0) continue;
            const std::string bare = k.substr(12);
            bool have = false;
            for (size_t j2 = 0; j2 < n; ++j2) if (out->keys[j2] == bare) { have = true; break; }
            if (!have) out->kv(bare, out->vals[i]);
        }
    }

    auto gs = [&](const char* k) -> const char* { return out->get(k); };
    /* text_config first, then the top level: the nesting is the only difference between a
     * multimodal config and a text one, and every field below appears in exactly one of the two. */
    auto gi = [&](const char* k, int64_t dflt) -> int64_t {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs(k);
        return v ? strtoll(v, nullptr, 10) : dflt;
    };
    auto gf = [&](const char* k, double dflt) -> double {
        const char* v = gs((std::string("text_config.") + k).c_str());
        if (!v) v = gs(k);
        return v ? strtod(v, nullptr) : dflt;
    };

    /* HuggingFace's model_type is the architecture id, minus the underscores its naming convention
     * uses and GGUF's does not: `qwen3_5` and GGUF's `qwen35` are the same architecture, and a
     * plugin should not have to claim both spellings. */
    if (const char* mt = gs("model_type")) {
        out->arch.clear();
        for (const char* p = mt; *p; ++p) if (*p != '_') out->arch += *p;
    }
    if (const char* nm = gs("_name_or_path")) out->name = nm;

    RadModelMeta& m = out->m;
    m.n_layers    = gi("num_hidden_layers", 0);
    m.n_embd      = gi("hidden_size", 0);
    m.n_head      = gi("num_attention_heads", 0);
    m.n_head_kv   = gi("num_key_value_heads", m.n_head);
    /* Read head_dim; do NOT derive it. On qwen35 it is 256 while hidden_size/num_attention_heads
     * is 213, and a derived value makes every downstream shape silently wrong. */
    m.head_dim    = gi("head_dim", m.n_head ? m.n_embd / m.n_head : 0);
    /* ...AND A ROUTED CHECKPOINT HAS NO `intermediate_size` AT ALL. A routed config states
     * `moe_intermediate_size` (one expert's width) and `shared_expert_intermediate_size` and
     * nothing else, so reading only the dense spelling leaves n_ff at zero -- and geom_from refuses
     * a model with a zero feed-forward width by name, which reads as a broken container rather
     * than as an unread key. The per-expert width is what a GEMM sees, so it is what n_ff means;
     * the shared expert's is its own key and the plugin reads it from the free-form table. */
    m.n_ff        = gi("intermediate_size", gi("moe_intermediate_size", 0));
    m.n_vocab     = gi("vocab_size", 0);
    m.n_ctx_train = gi("max_position_embeddings", 0);
    m.n_expert        = gi("num_experts", gi("num_local_experts", 0));
    m.n_expert_used   = gi("num_experts_per_tok", 0);
    m.n_expert_shared = gi("shared_expert_intermediate_size", 0) ? 1 : 0;
    m.rms_eps     = (float)gf("rms_norm_eps", 1e-6);
    m.rope_theta  = (float)gf("rope_parameters.rope_theta", gf("rope_theta", 10000.0));
    m.rope_scale  = (float)gf("rope_scaling.factor", 1.0);

    /* The quantisation descriptor, which with the architecture selects the plugin (spec §2.4).
     * "fp8" plus its format, so qwen35_fp8 and qwen35_bf16 are told apart by the checkpoint
     * rather than by a flag. */
    if (const char* qm = gs("quantization_config.quant_method")) {
        out->quant = qm;
        if (const char* fm = gs("quantization_config.fmt")) { out->quant += "_"; out->quant += fm; }
    }

    out->finish();
    return RAD_OK;
}

int meta_from_gguf(GgufFile& g, MetaOwn* out) {
    if (!g.get_str("general.architecture", &out->arch)) {
        RAD_ERR("the GGUF has no general.architecture, so no architecture plugin can be chosen");
        return RAD_E_FORMAT;
    }
    g.get_str("general.name", &out->name);

    /* Dimensions come from metadata, so one plugin covers every size in a family. The keys are
     * GGUF's `<arch>.<field>` convention; anything the struct does not name is passed through in
     * the free-form KV list, which the core never interprets. */
    const std::string a = out->arch;
    auto gi = [&](const char* suffix, int64_t dflt) -> int64_t {
        int64_t v = 0;
        return g.get_i64(a + "." + suffix, &v) ? v : dflt;
    };
    auto gf = [&](const char* suffix, double dflt) -> double {
        double v = 0;
        return g.get_f64(a + "." + suffix, &v) ? v : dflt;
    };

    RadModelMeta& m = out->m;
    m.n_layers    = gi("block_count", 0);
    m.n_embd      = gi("embedding_length", 0);
    m.n_head      = gi("attention.head_count", 0);
    m.n_head_kv   = gi("attention.head_count_kv", m.n_head);
    m.head_dim    = gi("attention.key_length", m.n_head ? m.n_embd / m.n_head : 0);
    m.n_ff        = gi("feed_forward_length", 0);
    m.n_ctx_train = gi("context_length", 0);
    m.n_expert       = gi("expert_count", 0);
    m.n_expert_used  = gi("expert_used_count", 0);
    m.n_expert_shared= gi("expert_shared_count", 0);
    m.rms_eps     = (float)gf("attention.layer_norm_rms_epsilon", 1e-6);
    m.rope_theta  = (float)gf("rope.freq_base", 10000.0);
    m.rope_scale  = (float)gf("rope.scaling.factor", 1.0);

    std::vector<std::string> toks;
    if (g.get_strs("tokenizer.ggml.tokens", &toks)) m.n_vocab = (int64_t)toks.size();
    if (m.n_vocab == 0) m.n_vocab = gi("vocab_size", 0);

    /* Everything else, verbatim and stringified. A plugin reads what it needs; the core never
     * interprets these. Arrays are summarised rather than expanded -- a 151k-token array in a
     * key/value list is not metadata, it is the vocab, and the vocab has its own section. */
    for (const GgufKV& kv : g.kv()) {
        if (kv.key.rfind("tokenizer.", 0) == 0) continue;
        std::string v;
        if (kv.is_array) {
            v = fmt("[%zu values]", kv.type == GGUF_STR ? kv.s.size()
                                  : !kv.i.empty()       ? kv.i.size() : kv.f.size());
            if (kv.type != GGUF_STR && kv.i.size() <= 8 && !kv.i.empty()) {
                v.clear();
                for (size_t j = 0; j < kv.i.size(); ++j)
                    v += fmt("%s%lld", j ? "," : "", (long long)kv.i[j]);
            }
        } else if (!kv.s.empty()) v = kv.s.front();
        else if (!kv.f.empty() && kv.type == GGUF_F32) v = fmt("%g", kv.f.front());
        else if (!kv.f.empty() && kv.type == GGUF_F64) v = fmt("%g", kv.f.front());
        else if (!kv.i.empty()) v = fmt("%lld", (long long)kv.i.front());
        if (v.size() > 240) v.resize(240);
        out->kv(kv.key, v);
    }

    /* THE DESCRIPTOR A SAFETENSORS RELEASE WOULD CARRY. A float file -- all F32 (0), mostly F16
     * (1) or mostly BF16 (32), the vectors F32 either way -- is an unquantised checkpoint and
     * selects the plugin a bf16 release does. Any other file type names its quantisation, which no
     * plugin serves as it stands. */
    int64_t ftype = 0;
    const bool typed = g.get_i64("general.file_type", &ftype);
    out->quant = !typed || ftype == 0 || ftype == 1 || ftype == 32
                     ? std::string() : fmt("gguf_ftype_%lld", (long long)ftype);
    out->finish();
    return RAD_OK;
}

/* ================================================================== the checkpoint */

int Checkpoint::open_gguf(const std::string& path) {
    RAD_TRY(gguf_.open(path.c_str()));
    kind_ = Gguf;
    for (const GgufTensor& t : gguf_.tensors()) {
        SrcTensor s;
        s.name  = t.name;
        s.ggml  = t.ggml_type;
        s.rad_dtype = gguf_to_rad_dtype(t.ggml_type);
        /* GGUF is fastest-first; everything from here on is row-major. */
        for (int d = (int)t.n_dims - 1; d >= 0; --d) s.shape.push_back(t.ne[d]);
        s.data  = gguf_.tensor_data(t);
        s.bytes = (int64_t)t.bytes;
        if (!s.data) {
            RAD_ERR("%s: '%s' has no readable extent", path.c_str(), t.name.c_str());
            return RAD_E_FORMAT;
        }
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return meta_from_gguf(gguf_, &meta_);
}

int Checkpoint::add_shard(const std::string& p, const std::string& prefix) {
    shards_.push_back(std::make_unique<SafeTensorsFile>());
    RAD_TRY(shards_.back()->open(p));
    for (const auto& t : shards_.back()->tensors()) {
        if (t.dtype == RAD_DT_INVALID) {
            RAD_ERR("%s: '%s' has dtype %s, which has no RAD equivalent", p.c_str(),
                    t.name.c_str(), t.dtype_str.c_str());
            return RAD_E_DTYPE;
        }
        SrcTensor s;
        s.name      = prefix + t.name;
        s.rad_dtype = t.dtype;
        s.shape     = t.shape;         /* safetensors is already row-major */
        s.data      = t.data;
        s.bytes     = (int64_t)t.bytes;
        index_.emplace(s.name, tensors_.size());
        tensors_.push_back(std::move(s));
    }
    return RAD_OK;
}

namespace {

/* Every shard an index names, once each and in name order, or the one model.safetensors. */
int shard_list(const std::string& dir, const std::string& index, bool index_given,
               std::vector<std::string>* out) {
    std::unordered_map<std::string, std::string> map;
    const int s = st_load_index(index, &map);
    if (s == RAD_OK) {
        for (auto& [t, f] : map)
            if (std::find(out->begin(), out->end(), f) == out->end()) out->push_back(f);
        std::sort(out->begin(), out->end());
        for (auto& f : *out) f = dir + "/" + f;
        return RAD_OK;
    }
    if (s != RAD_E_NOTFOUND || index_given) return s;
    out->push_back(dir + "/model.safetensors");
    return RAD_OK;
}

}  /* namespace */

int Checkpoint::open_safetensors(const std::string& path) {
    kind_ = SafeT;
    /* One shard, an index, or the directory holding either: one namespace whichever it is. */
    std::vector<std::string> shards;
    const bool is_index = path.size() > 10 && path.compare(path.size() - 10, 10, "index.json") == 0;
    const bool is_file = path.size() > 12 &&
                         path.compare(path.size() - 12, 12, ".safetensors") == 0;
    if (is_file) {
        shards.push_back(path);
    } else {
        RAD_TRY(shard_list(dir_, is_index ? path : dir_ + "/model.safetensors.index.json",
                           is_index, &shards));
    }
    for (const auto& sh : shards) RAD_TRY(add_shard(sh));
    std::string reduced_head = dir_ + "/mtp-lm-head-reduced.safetensors";
    if (const char* env_p = std::getenv("RADIANCE_MTP_HEAD_REDUCED")) {
        if (*env_p) reduced_head = env_p;
    }
    if (std::find(shards.begin(), shards.end(), reduced_head) == shards.end() &&
        ::access(reduced_head.c_str(), F_OK) == 0) {
        RAD_TRY(add_shard(reduced_head));
    }

    RAD_TRY(meta_from_config_json(dir_, &meta_));
    RAD_TRY(meta_merge_generation_config(dir_, &meta_));
    RAD_TRY(meta_merge_preprocessor_configs(dir_, &meta_));
    /* FP8's block, where the checkpoint states one. */
    if (const char* bs = meta_.get("quantization_config.weight_block_size")) {
        long long r = 0, c = 0;
        if (std::sscanf(bs, "%lld %lld", &r, &c) == 2 && r > 0 && c > 0) {
            fp8_br_ = r;
            fp8_bc_ = c;
        }
    }
    meta_.finish();
    return RAD_OK;
}

int Checkpoint::open(const std::string& path) {
    path_ = path;
    const bool gguf = path.size() > 5 && path.compare(path.size() - 5, 5, ".gguf") == 0;
    dir_ = path;
    const size_t slash = path.find_last_of('/');
    auto ends = [&](const char* s) {
        const size_t n = std::strlen(s);
        return path.size() > n && path.compare(path.size() - n, n, s) == 0;
    };
    const bool is_file = gguf || ends(".safetensors") || ends("index.json");
    if (is_file) dir_ = slash == std::string::npos ? "." : path.substr(0, slash);
    return gguf ? open_gguf(path) : open_safetensors(path);
}

int Checkpoint::add_draft(const std::string& dir, const std::string& prefix) {
    if (kind_ != SafeT) {
        RAD_ERR("a drafter is merged into a safetensors target's namespace; %s is not one",
                path_.c_str());
        return RAD_E_UNSUPPORTED;
    }
    RAD_TRY(meta_merge_config_json(dir, prefix.empty() || prefix.back() != '.'
                                            ? prefix
                                            : prefix.substr(0, prefix.size() - 1), &meta_));
    std::vector<std::string> shards;
    RAD_TRY(shard_list(dir, dir + "/model.safetensors.index.json", false, &shards));
    for (const auto& sh : shards) RAD_TRY(add_shard(sh, prefix));
    meta_.finish();
    return RAD_OK;
}

const SrcTensor* Checkpoint::find(const std::string& n) const {
    auto it = index_.find(n);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

/* ONE EXPERT'S SLICE OF A STACKED TENSOR, AS A VIEW.
 *
 * Slicing dim 0 of a row-major tensor is an offset and a shorter shape -- nothing is copied, so
 * the 512 declared weights of a MoE layer read 512 disjoint windows of the same mapped bytes.
 *
 * IT IS ALSO EXACT FOR A BLOCK-QUANTISED SOURCE, and that is not luck: ggml packs blocks along the
 * ROW, and a dim-0 slice is a whole number of rows, so a sub-tensor is a whole number of blocks.
 * `bytes % shape[0] == 0` therefore holds for every dtype and the division is the real byte count
 * rather than an estimate of one. A slice along any other dimension would not have that property
 * and is not offered. */
const SrcTensor* Checkpoint::slice(const SrcTensor& st, int index, const char* declared) {
    if (st.shape.size() < 2) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which has rank %zu. Only a STACKED tensor "
                "-- rank 2 or more, with the stack on dim 0 -- has sub-tensors to take.",
                declared, index, st.name.c_str(), st.shape.size());
        return nullptr;
    }
    const int64_t n = st.shape[0];
    if (index < 0 || index >= n) {
        RAD_ERR("'%s' asks for sub-tensor %d of '%s', which holds %lld of them",
                declared, index, st.name.c_str(), (long long)n);
        return nullptr;
    }
    if (!st.data || st.bytes <= 0 || st.bytes % n != 0) {
        RAD_ERR("'%s': '%s' is %lld byte(s) over %lld sub-tensor(s), which does not divide -- the "
                "slice would straddle two of them",
                declared, st.name.c_str(), (long long)st.bytes, (long long)n);
        return nullptr;
    }
    SrcTensor v;
    /* The name carries the index because every diagnostic below this point prints it, and
     * "experts.gate_up_proj" said 512 times for 512 different weights cannot be read. */
    v.name      = st.name + "[" + std::to_string(index) + "]";
    v.ggml      = st.ggml;
    v.rad_dtype = st.rad_dtype;
    v.shape.assign(st.shape.begin() + 1, st.shape.end());
    v.bytes     = st.bytes / n;
    v.data      = (const uint8_t*)st.data + (size_t)index * (size_t)v.bytes;
    sliced_.push_back(std::move(v));
    return &sliced_.back();
}

/* ================================================================== logical weights */

namespace {

/* FP8's scale plane is a companion tensor: `X.weight` is scaled by `X.weight_scale_inv`.
 * Q4 affine's companion tensors: `X.weight_q4` has `X.weight_scale` and `X.weight_zero`. */
std::string scale_name(const std::string& codes) {
    if (codes.size() >= 10 && codes.compare(codes.size() - 10, 10, ".weight_q4") == 0)
        return codes.substr(0, codes.size() - 10) + ".weight_scale";
    if (codes.size() >= 9 && codes.compare(codes.size() - 9, 9, "weight_q4") == 0)
        return codes.substr(0, codes.size() - 9) + "weight_scale";
    return codes + "_scale_inv";
}

std::string zero_name(const std::string& codes) {
    if (codes.size() >= 10 && codes.compare(codes.size() - 10, 10, ".weight_q4") == 0)
        return codes.substr(0, codes.size() - 10) + ".weight_zero";
    if (codes.size() >= 9 && codes.compare(codes.size() - 9, 9, "weight_q4") == 0)
        return codes.substr(0, codes.size() - 9) + "weight_zero";
    return "";
}

}  /* namespace */

std::string ckpt_tensor_of(const SrcTensor& t) {
    const size_t br = t.name.size() && t.name.back() == ']' ? t.name.rfind('[') : std::string::npos;
    return br == std::string::npos ? t.name : t.name.substr(0, br);
}

bool ckpt_reads_any(const CkptWeight& w, const std::unordered_set<std::string>& names,
                    std::unordered_set<std::string>* hit) {
    bool any = false;
    for (const auto* list : { &w.pieces, &w.scales, &w.zeroes })
        for (const SrcTensor* t : *list) {
            if (!t) continue;
            const std::string n = ckpt_tensor_of(*t);
            if (names.count(n)) {
                any = true;
                if (hit) hit->insert(n);
            }
        }
    return any;
}

int ckpt_resolve(Checkpoint& ck, const Program& prog, const std::string& name, bool optional,
                 CkptWeight* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    *out = CkptWeight{};
    out->name = name;

    /* EVERY map for the name, in declaration order. A CONCAT map may be declared in several
     * calls -- the ABI's `src[8]` against a 128-shard tensor -- and they join end to end; every
     * other repeat was refused at declare, so more than one implies CONCAT. */
    std::vector<const RadNameMap*> maps;
    for (const auto& nm : prog.name_map)
        if (nm.declared && name == nm.declared) maps.push_back(&nm);
    if (maps.empty()) return refuse(RAD_E_NOTFOUND, "the name map has no entry for it");
    struct Src1 { const char* name; int index; };
    std::vector<Src1> srcs;
    for (const RadNameMap* pm : maps)
        for (int k = 0; k < pm->n_src; ++k) srcs.push_back({ pm->src[k], pm->src_index[k] });
    out->concat_dim = maps.front()->concat_dim;

    /* THE MODE DECIDES WHAT SEVERAL SOURCES MEAN.
     *
     *   RAD_MAP_CONCAT   all of them, joined -- gate and up into gate_up.
     *   RAD_MAP_COPY     ALTERNATIVES, first present wins: a tied lm_head is output.weight where
     *                    the checkpoint ships one and the embedding where it does not.
     *
     * Treating a COPY's alternatives as a concatenation is silently wrong in both directions: a
     * checkpoint holding both writes the head at twice its rows, and one holding only the
     * embedding fails at the missing first source instead of falling back to it. */
    const bool alternatives = maps.front()->mode == RAD_MAP_COPY && srcs.size() > 1;
    for (const Src1& s1 : srcs) {
        const SrcTensor* st = ck.find(s1.name);
        const SrcTensor* sc = st ? ck.find(scale_name(s1.name)) : nullptr;
        const std::string zn = zero_name(s1.name);
        const SrcTensor* sz = (st && !zn.empty()) ? ck.find(zn) : nullptr;
        /* A SUB-TENSOR INDEX IS TAKEN, never ignored: ignoring it gives every expert a copy of
         * the whole stack, in a model that loads and answers. Its scale plane is stacked the same
         * way and sliced at the same index. */
        if (st && s1.index >= 0) {
            st = ck.slice(*st, s1.index, name.c_str());
            if (!st) return refuse(RAD_E_SHAPE, "a sub-tensor it names is not there");
            if (sc && !(sc = ck.slice(*sc, s1.index, name.c_str())))
                return refuse(RAD_E_SHAPE, "its scale plane cannot be sliced at the same index");
            if (sz && !(sz = ck.slice(*sz, s1.index, name.c_str())))
                return refuse(RAD_E_SHAPE, "its zero plane cannot be sliced at the same index");
        }
        if (!st) {
            if (alternatives) continue;
            if (optional) { out->pieces.clear(); break; }
            return refuse(RAD_E_NOTFOUND, std::string("it needs checkpoint tensor '") + s1.name +
                                          "', which is not in " + ck.path());
        }
        out->pieces.push_back(st);
        out->scales.push_back(sc);
        out->zeroes.push_back(sz);
        if (alternatives) break;
    }
    if (out->pieces.empty()) {
        if (optional) return refuse(RAD_E_NOTFOUND, "none of its tensors is in the checkpoint");
        std::string tried;
        for (size_t k = 0; k < srcs.size(); ++k) tried += (k ? " or " : "") + std::string(srcs[k].name);
        return refuse(RAD_E_NOTFOUND, "it needs one of " + tried + ", and " + ck.path() +
                                      " holds none of them");
    }

    /* THE LOGICAL SHAPE: the pieces joined along concat_dim, every other extent agreeing. */
    const SrcTensor& p0 = *out->pieces[0];
    const int d = out->concat_dim;
    if (p0.shape.empty() || p0.shape.size() > RAD_MAX_RANK || d < 0 || d >= (int)p0.shape.size())
        return refuse(RAD_E_SHAPE, "'" + p0.name + "' cannot be joined along dim " +
                                   std::to_string(d));
    out->rank = (uint32_t)p0.shape.size();
    for (uint32_t i = 0; i < out->rank; ++i) out->shape[i] = p0.shape[i];
    for (size_t k = 1; k < out->pieces.size(); ++k) {
        const SrcTensor& pk = *out->pieces[k];
        bool fits = pk.shape.size() == p0.shape.size();
        for (uint32_t i = 0; fits && i < out->rank; ++i)
            if ((int)i != d && pk.shape[i] != p0.shape[i]) fits = false;
        if (!fits)
            return refuse(RAD_E_SHAPE, "'" + pk.name + "' and '" + p0.name +
                                       "' do not join along dim " + std::to_string(d));
        out->shape[d] += pk.shape[(size_t)d];
    }
    rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);

    const bool is_q4_affine = !out->zeroes.empty() && out->zeroes[0] != nullptr &&
                              !out->scales.empty() && out->scales[0] != nullptr &&
                              (p0.rad_dtype == RAD_U8 || p0.rad_dtype == RAD_U4);
    if (is_q4_affine) {
        if (!out->scales[0]->shape.empty() &&
            p0.shape.back() * 2 == out->scales[0]->shape.back() * 128) {
            out->shape[out->rank - 1] = p0.shape.back() * 2;
        }
        rad_enc_view(out->rank, out->shape, &out->rows, &out->cols);
        out->trivial = true;
        rad_enc_clear(&out->enc);
        rad_enc_copy_str(out->enc.scheme, "affine");
        rad_enc_add_plane(&out->enc, "codes", RAD_U4, 1, 1);
        rad_enc_add_plane(&out->enc, "scale", out->scales[0]->rad_dtype, 1, 128);
        rad_enc_add_plane(&out->enc, "zero", out->zeroes[0]->rad_dtype, 1, 128);
        return RAD_OK;
    }

    /* THE ENCODING THE PIECES ARE, when it is trivial (spec §4.3). */
    const uint32_t dt = p0.rad_dtype;
    bool same = true, scaled = true, unscaled = true;
    for (size_t k = 0; k < out->pieces.size(); ++k) {
        if (out->pieces[k]->rad_dtype != dt) same = false;
        if (out->scales[k]) unscaled = false; else scaled = false;
    }
    if (dt == RAD_DT_INVALID) {
        out->why = std::string("'") + p0.name + "' is a GGUF " +
                   (gguf_dtype(p0.ggml) ? gguf_dtype(p0.ggml)->name : "block") + " tensor";
    } else if (!same) {
        out->why = "its pieces are stored in different dtypes";
    } else if ((dt == RAD_F8E4M3 || dt == RAD_F8E5M2) && scaled) {
        /* BLOCK-SCALED FP8, byte for byte: the codes and the scale plane are the planes of
         * `fp8_e4m3*bf16[128x128]` (whatever the scale's own dtype). Joined only along rows, and
         * only where each piece but the last fills its last row block -- a scale covering the
         * tail of one piece and the head of the next belongs to neither. */
        const int64_t br = ck.fp8_block_rows(), bc = ck.fp8_block_cols();
        const uint32_t sdt = out->scales[0]->rad_dtype;
        out->trivial = true;
        if (d != 0 && out->pieces.size() > 1) {
            out->trivial = false;
            out->why = "its FP8 pieces are joined along a column, across their scale blocks";
        }
        for (size_t k = 0; k < out->pieces.size() && out->trivial; ++k) {
            const SrcTensor& pk = *out->pieces[k];
            const SrcTensor& sk = *out->scales[k];
            int64_t r = 0, c = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &r, &c);
            const int64_t want_r = (r + br - 1) / br, want_c = (c + bc - 1) / bc;
            int64_t sr = 0, scl = 0;
            rad_enc_view((uint32_t)sk.shape.size(), sk.shape.data(), &sr, &scl);
            if (sk.rad_dtype != sdt || sr != want_r || scl != want_c) {
                out->trivial = false;
                out->why = "'" + sk.name + "' is not the [" + std::to_string(want_r) + ", " +
                           std::to_string(want_c) + "] " + std::to_string(br) + "x" +
                           std::to_string(bc) + " scale plane of '" + pk.name + "'";
            } else if (k + 1 < out->pieces.size() && r % br) {
                out->trivial = false;
                out->why = "'" + pk.name + "' ends inside a scale block, so the next piece's "
                           "scales cannot follow it";
            }
        }
        if (out->trivial) {
            rad_enc_clear(&out->enc);
            rad_enc_copy_str(out->enc.scheme, "affine");
            rad_enc_add_plane(&out->enc, "codes", dt, 1, 1);
            rad_enc_add_plane(&out->enc, "scale", (sdt == RAD_F32) ? (uint32_t)RAD_BF16 : sdt, br, bc);
        }
    } else if (!unscaled) {
        out->why = "some of its pieces carry a scale plane and some do not";
    } else if (dt == RAD_F8E4M3 || dt == RAD_F8E5M2) {
        out->why = "it is FP8 with no scale plane beside it";
    } else {
        out->trivial = rad_dtype_bits(dt) >= 8;
        if (out->trivial) out->enc = rad_enc_plain(dt);
        else out->why = std::string("it is ") + rad_dtype_name(dt);
    }
    return RAD_OK;
}

namespace {

/* ---------------------------------------------------------------- GGUF block types
 *
 * The ggml block types a weight can be read from as f32. Their GEOMETRY is core/text's -- that is
 * what sizes a tensor -- and decoding one is content, so the short table is here, where a weight
 * is read. The K-quants are refused by name rather than half-implemented: a super-block scale read
 * wrong is a model that talks and is wrong. */
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7,
       GGML_Q8_0 = 8, GGML_BF16 = 30 };

int ggml_rows_f32(uint32_t type, const uint8_t* p, int64_t n, float* dst) {
    auto half = [&](const uint8_t* q) { uint16_t h; std::memcpy(&h, q, 2); return rad_f16_to_f32(h); };
    switch (type) {
        case GGML_Q8_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 34) {
                const float d = half(p);
                for (int j = 0; j < 32; ++j) dst[b * 32 + j] = d * (float)(int8_t)p[2 + j];
            }
            return RAD_OK;
        case GGML_Q4_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 18) {
                const float d = half(p);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)((p[2 + j] & 0x0F) - 8);
                    dst[b * 32 + j + 16] = d * (float)((p[2 + j] >> 4)   - 8);
                }
            }
            return RAD_OK;
        case GGML_Q4_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 20) {
                const float d = half(p), m = half(p + 2);
                for (int j = 0; j < 16; ++j) {
                    dst[b * 32 + j]      = d * (float)(p[4 + j] & 0x0F) + m;
                    dst[b * 32 + j + 16] = d * (float)(p[4 + j] >> 4)   + m;
                }
            }
            return RAD_OK;
        case GGML_Q5_0:
            for (int64_t b = 0; b < n / 32; ++b, p += 22) {
                const float d = half(p);
                uint32_t qh; std::memcpy(&qh, p + 2, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[6 + j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int hi = (int)((p[6 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    dst[b * 32 + j]      = d * (float)lo;
                    dst[b * 32 + j + 16] = d * (float)hi;
                }
            }
            return RAD_OK;
        case GGML_Q5_1:
            for (int64_t b = 0; b < n / 32; ++b, p += 24) {
                const float d = half(p), m = half(p + 2);
                uint32_t qh; std::memcpy(&qh, p + 4, 4);
                for (int j = 0; j < 16; ++j) {
                    const int lo = (int)((p[8 + j] & 0x0F) | (((qh >> j) & 1u) << 4));
                    const int hi = (int)((p[8 + j] >> 4)   | (((qh >> (j + 16)) & 1u) << 4));
                    dst[b * 32 + j]      = d * (float)lo + m;
                    dst[b * 32 + j + 16] = d * (float)hi + m;
                }
            }
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

/* Where flat element `i` of the joined tensor lives: which piece, and the element in it. A join
 * along dim d repeats, for each index of the dims before d, every piece's [d..] extent in turn. */
struct Joined {
    std::vector<int64_t> inner;   /* each piece's elements per outer index */
    int64_t total_inner = 0;
    int     n = 0;
    void init(const std::vector<const SrcTensor*>& pieces, int d) {
        n = (int)pieces.size();
        inner.assign((size_t)n, 1);
        total_inner = 0;
        for (int k = 0; k < n; ++k) {
            for (size_t i = (size_t)d; i < pieces[(size_t)k]->shape.size(); ++i)
                inner[(size_t)k] *= pieces[(size_t)k]->shape[i];
            total_inner += inner[(size_t)k];
        }
    }
    /* Visit [a, b) as runs that lie in one piece: f(piece, element in it, count, at). */
    template <class F> int walk(int64_t a, int64_t b, F f) const {
        while (a < b) {
            const int64_t o = a / total_inner;
            int64_t w = a % total_inner, k = 0;
            while (w >= inner[(size_t)k]) w -= inner[(size_t)k++];
            const int64_t run = std::min<int64_t>(b - a, inner[(size_t)k] - w);
            const int rc = f((int)k, o * inner[(size_t)k] + w, run);
            if (rc < 0) return rc;
            a += run;
        }
        return RAD_OK;
    }
};

/* `n` elements of a stored-dtype piece from element `e`, as f32. */
int piece_f32(const SrcTensor& t, int64_t e, int64_t n, float* out) {
    if (t.rad_dtype == RAD_DT_INVALID) {
        /* a GGUF block type: whole rows only, and a row is whole blocks */
        const int64_t cols = t.shape.back();
        if (cols <= 0 || e % cols || n % cols) return RAD_E_SHAPE;
        const int64_t rows = t.numel() / cols, rb = t.bytes / (rows ? rows : 1);
        return ggml_rows_f32(t.ggml, (const uint8_t*)t.data + (e / cols) * rb, n, out);
    }
    const uint8_t* p = (const uint8_t*)t.data;
    switch (t.rad_dtype) {
        case RAD_F32:
            std::memcpy(out, (const float*)t.data + e, (size_t)n * 4);
            return RAD_OK;
        case RAD_BF16: {
            /* The top half of an f32, so a shift -- and a loop the compiler vectorises, which the
             * per-element dtype switch below is not. Most of a checkpoint is this. */
            const uint16_t* q = (const uint16_t*)t.data + e;
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t u = (uint32_t)q[i] << 16;
                std::memcpy(out + i, &u, 4);
            }
            return RAD_OK;
        }
        case RAD_F16: case RAD_F8E4M3: case RAD_F8E5M2:
        case RAD_I8: case RAD_U8: case RAD_I16: case RAD_I32: case RAD_U32: case RAD_U16:
            for (int64_t i = 0; i < n; ++i) out[i] = rad_load_f32(p, t.rad_dtype, e + i);
            return RAD_OK;
        default:
            return RAD_E_UNSUPPORTED;
    }
}

}  /* namespace */

int ckpt_rows_f32(const CkptWeight& w, int64_t r0, int64_t nr, float* out, std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (r0 < 0 || nr < 0 || r0 + nr > w.rows) return refuse(RAD_E_SHAPE, "rows out of range");
    Joined j;
    j.init(w.pieces, w.concat_dim);
    /* the runs come in order, so the cursor is the elements already written */
    int64_t at = 0;
    const int rc = j.walk(r0 * w.cols, (r0 + nr) * w.cols, [&](int k, int64_t e, int64_t n) {
        const int s = piece_f32(*w.pieces[(size_t)k], e, n, out + at);
        at += n;
        return s;
    });
    if (rc < 0)
        return refuse(rc, "'" + w.pieces[0]->name + "' is stored in a form that cannot be read "
                          "as f32 (" + (w.pieces[0]->rad_dtype == RAD_DT_INVALID
                                            ? std::string(gguf_dtype(w.pieces[0]->ggml)
                                                              ? gguf_dtype(w.pieces[0]->ggml)->name
                                                              : "?")
                                            : std::string(rad_dtype_name(w.pieces[0]->rad_dtype))) +
                          ")");
    /* FP8 IS ITS CODE TIMES ITS BLOCK'S SCALE: the pieces are joined along rows, so logical row
     * r is in the piece whose rows cover it, and its scales are that piece's. */
    if (!w.scales.empty() && w.scales[0]) {
        const int64_t br = w.enc.n_planes > 1 ? w.enc.plane[1].block[0] : 128;
        const int64_t bc = w.enc.n_planes > 1 ? w.enc.plane[1].block[1] : 128;
        int64_t base = 0;
        for (size_t k = 0; k < w.pieces.size(); ++k) {
            const SrcTensor& pk = *w.pieces[k];
            const SrcTensor& sk = *w.scales[k];
            int64_t pr = 0, pc = 0;
            rad_enc_view((uint32_t)pk.shape.size(), pk.shape.data(), &pr, &pc);
            const int64_t sc = (pc + bc - 1) / bc;
            for (int64_t r = std::max(r0, base); r < std::min(r0 + nr, base + pr); ++r)
                for (int64_t c = 0; c < w.cols; ++c)
                    out[(r - r0) * w.cols + c] *=
                        rad_load_f32(sk.data, sk.rad_dtype, ((r - base) / br) * sc + c / bc);
            base += pr;
        }
    }
    return RAD_OK;
}

int ckpt_plane_rows(const CkptWeight& w, int k, int64_t r0, int64_t nr, void* dst,
                    std::string* why) {
    auto refuse = [&](int rc, std::string m) {
        if (why) *why = std::move(m);
        return rc;
    };
    if (!w.trivial || k < 0 || k >= w.enc.n_planes)
        return refuse(RAD_E_STATE, "not a plane of a trivial encoding");
    const std::vector<const SrcTensor*>& src = k == 0 ? w.pieces : (k == 1 ? w.scales : w.zeroes);
    int64_t pr = 0, pc = 0;
    rad_enc_plane_dims(&w.enc.plane[k], w.rows, w.cols, &pr, &pc);
    if (r0 < 0 || nr < 0 || r0 + nr > pr) return refuse(RAD_E_SHAPE, "plane rows out of range");
    const uint32_t dt = w.enc.plane[k].dtype;
    const int64_t eb = rad_dtype_bits(dt) / 8;
    uint8_t* o = (uint8_t*)dst;
    if (k == 0) {
        /* the codes: the pieces joined, bytes as they are */
        const int bits = rad_dtype_bits(dt);
        if (bits < 8) {
            const int64_t row_bytes = rad_dtype_bytes(dt, pc);
            for (int64_t r = 0; r < nr; ++r) {
                const int64_t src_row = r0 + r;
                std::memcpy(o + r * row_bytes,
                            (const uint8_t*)src[0]->data + src_row * row_bytes,
                            (size_t)row_bytes);
            }
            return RAD_OK;
        }
        Joined j;
        j.init(src, w.concat_dim);
        return j.walk(r0 * pc, (r0 + nr) * pc, [&](int p, int64_t e, int64_t n) {
            std::memcpy(o, (const uint8_t*)src[(size_t)p]->data + e * eb, (size_t)(n * eb));
            o += n * eb;
            return RAD_OK;
        });
    }
    /* an FP8 scale or affine scale/zero plane: the pieces' scale rows one after another */
    int64_t base = 0;
    for (const SrcTensor* s : src) {
        int64_t sr = 0, scl = 0;
        rad_enc_view((uint32_t)s->shape.size(), s->shape.data(), &sr, &scl);
        const int64_t lo = std::max(r0, base), hi = std::min(r0 + nr, base + sr);
        if (hi > lo) {
            if (s->rad_dtype == RAD_F32 && dt == RAD_BF16) {
                const float* in = (const float*)s->data + (lo - base) * scl;
                uint16_t* out_bf16 = (uint16_t*)o;
                const int64_t count = (hi - lo) * scl;
                for (int64_t i = 0; i < count; ++i) {
                    out_bf16[i] = rad_f32_to_bf16(in[i]);
                }
                o += count * sizeof(uint16_t);
            } else {
                std::memcpy(o, (const uint8_t*)s->data + (lo - base) * scl * eb,
                            (size_t)((hi - lo) * scl * eb));
                o += (hi - lo) * scl * eb;
            }
        }
        base += sr;
    }
    return RAD_OK;
}

}  /* namespace rad */
