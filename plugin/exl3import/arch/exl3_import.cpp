/* exl3_import.cpp -- a CONVERT-ONLY architecture plugin: it lets rad-convert write an exllamav3 (EXL3) checkpoint
 * into a .rad container weight for weight, before an architecture plugin that serves the model exists.
 *
 * rad-convert runs an architecture plugin's declare to learn which weights a container holds (spec §2.4), and
 * refuses a checkpoint no plugin claims. This plugin declares every tensor of the EXL3 checkpoint under its own
 * checkpoint name and nothing else: no buffers, no ops, no logits. So the container it gives is a faithful copy
 * of the checkpoint in radiance's format, and a plugin written later for the model serves it by declaring the
 * same names (README.md, "The contract").
 *
 *   the EXL3 linear <X>  (<X>.trellis, .suh, .svh, a codebook marker)   one weight <X>.weight, exl3 encoding,
 *                                                                        written by the `exl3` quantiser
 *   every other tensor <T> (norms, embeddings, biases, routers, ...)    one weight <T>, as the checkpoint
 *                                                                        holds it (or as a recipe rule says)
 *
 * The checkpoint is found by `exl3_import.src` (rad-convert --set), else $EXL3_IMPORT_SRC: the same directory
 * as the recipe's `src=`. rad-convert sees <X>.weight only through the hole file tools/exl3_inventory.py writes
 * (--add): the importer reads the planes from src= and ignores the rows it is handed.
 *
 * SELECTION. The plugin claims ($EXL3_IMPORT_ARCH, $EXL3_IMPORT_QUANT), by default ("exl3import", "exl3"), so
 * installed beside the serving plugins it claims nothing until the convert script names the checkpoint's own
 * id (config.json model_type without underscores, e.g. "deepseekv4"). The container then carries that id, and
 * the plugin that serves it claims it.
 *
 * GROUPS (spec §4.2). A tensor under `<P>layers.<N>.` is in layer N when <P> is the trunk's prefix (the one
 * whose layer count is num_hidden_layers, else the first), and in layer num_hidden_layers + ... for every
 * other prefix (an MTP head, a vision tower), numbered after the trunk in name order. One under
 * `.experts.<E>.` is routed expert E of its layer, and its slot is the rank of its name with the expert number
 * taken out among that layer's expert names (gate_proj, up_proj, down_proj -> down 0, gate 1, up 2).
 *
 * MIXED EXPERTS. The container addresses a layer's expert group by a fixed stride, so every expert in it has one
 * layout: the same encoding and shape at every slot (core/format/radfile.cpp, index_experts). A checkpoint whose
 * experts differ inside a layer (SAGE-style mixed bit widths, an expert missing a weight) is written the way
 * radiance's own containers write protected bf16 experts (arch/common/rad_block_moe_fp8.h): the layer's most
 * common expert form is its expert group, numbered densely in expert order, and every other expert's weights
 * are tensors of the layer (expert -1). Each keeps its checkpoint name, which carries its real expert number;
 * a group member's number in the directory is its place among the group's members. Ties go to the form of the
 * lowest-numbered expert.
 *
 * Its step is never run: a container is served by the model's own plugin. Declared with no logits, a serve
 * of this plugin is refused at startup.
 */
#include "rad_abi.h"
#include "rad_builder.h"
#include "rad_encoding.h"

#include "exl3_encoding.h"
#include "exl3_src.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace exl3_import {

const char kPlugin[] = "exl3_import";

/* The tensors of one EXL3 linear <X> besides <X>.trellis: they are its planes, not weights of their own. */
const char* const kParts[] = {"trellis", "suh", "svh", "su", "sv", "mul1", "mcg"};

const char* env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return v && *v ? v : dflt;
}

/* A refusal, on stderr: the builder's notes are printed only for a declare that succeeds. */
int refuse(int status, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
int refuse(int status, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "radiance: %s: ", kPlugin);
    std::vfprintf(stderr, fmt, ap);
    std::fputc('\n', stderr);
    va_end(ap);
    return status;
}

bool ends_with(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

/* `<P>layers.<N>.`: the prefix and N, or false. The last occurrence wins (a tower inside a wrapper). */
bool layer_of(const std::string& name, std::string* prefix, int* n) {
    static const char kKey[] = "layers.";
    size_t at = std::string::npos;
    for (size_t p = name.find(kKey); p != std::string::npos; p = name.find(kKey, p + 1))
        if (p == 0 || name[p - 1] == '.') at = p;
    if (at == std::string::npos) return false;
    size_t q = at + sizeof kKey - 1, d = q;
    while (d < name.size() && name[d] >= '0' && name[d] <= '9') ++d;
    if (d == q || d >= name.size() || name[d] != '.') return false;
    *prefix = name.substr(0, at);
    *n = std::atoi(name.c_str() + q);
    return true;
}

/* `.experts.<E>.`: E and the name with E taken out (the slot's key), or false. */
bool expert_of(const std::string& name, int* e, std::string* key) {
    static const char kKey[] = ".experts.";
    const size_t at = name.find(kKey);
    if (at == std::string::npos) return false;
    const size_t q = at + sizeof kKey - 1;
    size_t d = q;
    while (d < name.size() && name[d] >= '0' && name[d] <= '9') ++d;
    if (d == q || d >= name.size() || name[d] != '.') return false;
    *e = std::atoi(name.c_str() + q);
    *key = name.substr(0, q) + "*" + name.substr(d);
    return true;
}

struct Logical {
    std::string name;    /* the declared weight and the checkpoint tensor it is made from */
    bool exl3 = false;   /* an EXL3 linear (<name> is <X>.weight) */
    int layer = -1;
    int real = -1;       /* the expert number its name carries, -1 for none */
    int expert = -1;     /* its place in the layer's expert group, -1 outside one */
    int slot = 0;
    std::string key;     /* an expert weight's name with the number taken out */
    RadEncoding enc{};
    int64_t shape[RAD_MAX_RANK] = {0};
    uint32_t rank = 0;
    std::string form;    /* encoding and shape, as a comparable string */
};

int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    const char* src = rad_meta_gets(meta, "exl3_import.src", nullptr);
    if (!src || !*src) src = env_or("EXL3_IMPORT_SRC", nullptr);
    if (!src)
        return refuse(RAD_E_INVAL, "no EXL3 checkpoint named: pass rad-convert --set exl3_import.src=DIR (the "
                                   "recipe's src=) or set EXL3_IMPORT_SRC");
    if (ctx && ctx->world_size > 1)
        return refuse(RAD_E_UNSUPPORTED, "declares whole weights for rad-convert; it does not serve, at --tp %d or "
                                         "any other", (int)ctx->world_size);
    std::string why;
    exl3::Source* s = exl3::Source::open(src, &why);
    if (!s) return refuse(RAD_E_FORMAT, "%s", why.c_str());

    /* The linears, then every other tensor. */
    const std::vector<std::string> names = s->names();
    std::set<std::string> linears;
    for (const std::string& n : names)
        if (ends_with(n, ".trellis")) linears.insert(n.substr(0, n.size() - 8));
    std::vector<Logical> logical;
    for (const std::string& n : linears) {
        Logical L;
        L.name = n + ".weight";
        L.exl3 = true;
        logical.push_back(L);
    }
    for (const std::string& n : names) {
        const size_t dot = n.rfind('.');
        if (dot != std::string::npos && linears.count(n.substr(0, dot))) {
            bool part = false;
            for (const char* p : kParts) part = part || n.compare(dot + 1, std::string::npos, p) == 0;
            if (part) continue;
        }
        if (ends_with(n, ".weight") && linears.count(n.substr(0, n.size() - 7)))
            return refuse(RAD_E_DUPLICATE, "'%s' is both a tensor of the checkpoint and the EXL3 linear its .trellis "
                                           "makes", n.c_str());
        Logical L;
        L.name = n;
        logical.push_back(L);
    }

    /* Layers: the trunk's prefix keeps its numbers, every other prefix follows it. */
    std::map<std::string, int> count;   /* prefix -> 1 + its largest layer number */
    for (const Logical& L : logical) {
        std::string p;
        int n = 0;
        if (layer_of(L.name, &p, &n) && n + 1 > count[p]) count[p] = n + 1;
    }
    std::string trunk;
    for (const auto& kv : count)
        if (kv.second == meta->n_layers) { trunk = kv.first; break; }
    if (trunk.empty() && !count.empty()) trunk = count.begin()->first;
    std::map<std::string, int> base;
    int next = (int)(meta->n_layers > 0 ? meta->n_layers : (count.count(trunk) ? count[trunk] : 0));
    if (!trunk.empty()) base[trunk] = 0;
    if (!trunk.empty() && count[trunk] > next) next = count[trunk];
    for (const auto& kv : count)
        if (kv.first != trunk) { base[kv.first] = next; next += kv.second; }

    /* Every weight's name map entry and encoding (the recipe's, for an EXL3 linear). */
    for (Logical& L : logical) {
        /* The name map is how the checkpoint is searched for a weight, even under its own name. */
        RadNameMap map;
        std::memset(&map, 0, sizeof map);
        map.declared = L.name.c_str();
        map.mode = RAD_MAP_COPY;
        map.n_src = 1;
        map.src[0] = L.name.c_str();
        map.src_index[0] = -1;
        if (rad_decl_name_map(b, &map) != RAD_OK)
            return refuse(RAD_E_INVAL, "'%s': the name map refused it", L.name.c_str());
        const int st = rad_weight_encoding(b, L.name.c_str(), &L.enc, L.shape, &L.rank);
        if (st != RAD_OK)
            return refuse(st, "'%s' has no encoding (%d)%s", L.name.c_str(), st,
                          L.exl3 ? ": is the hole file (tools/exl3_inventory.py --holes) passed with --add?" : "");
        char e[256];
        rad_enc_format(&L.enc, e, sizeof e);
        if (L.exl3 && exl3_encoding_read(&L.enc, nullptr, nullptr, nullptr) != RAD_OK)
            return refuse(RAD_E_DTYPE, "'%s' is an EXL3 linear and its encoding is %s: the recipe must import it "
                                       "with the exl3 quantiser (`*.weight exl3 src=$EXL3_SRC bits=0 codebook=auto`)",
                          L.name.c_str(), e);
        L.form = e;
        for (uint32_t r = 0; r < L.rank && r < RAD_MAX_RANK; ++r) L.form += " " + std::to_string(L.shape[r]);
    }

    /* Layers and experts. */
    for (Logical& L : logical) {
        std::string p;
        int n = 0;
        if (layer_of(L.name, &p, &n)) L.layer = base[p] + n;
        int e = 0;
        if (L.layer >= 0 && expert_of(L.name, &e, &L.key)) L.real = e;
    }

    /* Each layer's expert group: the most common expert form, an expert's form being its weights' forms by
     * slot key (a missing weight is a form of its own). */
    std::map<int, std::set<std::string>> keys;                       /* layer -> its expert weight keys */
    std::map<int, std::map<int, std::map<std::string, std::string>>> unit;   /* layer -> expert -> key -> form */
    for (const Logical& L : logical)
        if (L.real >= 0) {
            keys[L.layer].insert(L.key);
            unit[L.layer][L.real][L.key] = L.form;
        }
    std::map<int, std::map<int, int>> place;   /* layer -> real expert -> its place in the group */
    int64_t n_outside = 0, n_outside_layers = 0;
    for (auto& [layer, experts] : unit) {
        std::map<std::string, std::vector<int>> by_form;   /* an expert's whole form -> its experts, in order */
        std::vector<std::string> order;                    /* forms by their lowest expert */
        for (const auto& [real, forms] : experts) {
            std::string whole;
            for (const std::string& k : keys[layer]) {
                const auto f = forms.find(k);
                whole += k + "=" + (f == forms.end() ? std::string("(absent)") : f->second) + ";";
            }
            if (!by_form.count(whole)) order.push_back(whole);
            by_form[whole].push_back(real);
        }
        const std::string* best = &order.front();
        for (const std::string& f : order)
            if (by_form[f].size() > by_form[*best].size()) best = &f;
        const std::vector<int>& members = by_form[*best];
        for (size_t i = 0; i < members.size(); ++i) place[layer][members[i]] = (int)i;
        if (keys[layer].size() > 8)
            return refuse(RAD_E_UNSUPPORTED, "layer %d's experts have %zu weights each; a movement group holds at "
                                             "most 8 slots", layer, keys[layer].size());
        const int64_t out = (int64_t)(experts.size() - members.size());
        if (out > 0) {
            n_outside += out;
            ++n_outside_layers;
            rad_note(b, "%s: layer %d: %zu of %zu experts are its expert group; %lld differ and are tensors of the "
                        "layer", kPlugin, layer, members.size(), experts.size(), (long long)out);
        }
    }
    for (Logical& L : logical) {
        if (L.real < 0) continue;
        const auto& k = keys[L.layer];
        const auto it = place[L.layer].find(L.real);
        if (it == place[L.layer].end()) continue;   /* outside the group: a tensor of the layer */
        L.expert = it->second;
        L.slot = (int)std::distance(k.begin(), k.find(L.key));
    }

    /* Declare each one plane by plane: a declaration that takes several planes as one operand needs a kernel to
     * join them, and this plugin declares no op (core/build/rad_builder.cpp, finish_layouts). */
    int64_t n_exl3 = 0, n_plain = 0;
    for (const Logical& L : logical) {
        int64_t rows = 0, cols = 0;
        rad_enc_view(L.rank, L.shape, &rows, &cols);
        for (int k = 0; k < L.enc.n_planes; ++k) {
            const RadEncPlane& p = L.enc.plane[k];
            RadWeightDecl d;
            std::memset(&d, 0, sizeof d);
            d.dtype = p.dtype;
            if (L.enc.n_planes == 1 && p.kind == RAD_PLANE_TILED && p.block[0] == 1 && p.block[1] == 1) {
                d.rank = L.rank;   /* a plain tensor keeps its shape */
                for (uint32_t r = 0; r < L.rank && r < RAD_MAX_RANK; ++r) d.shape[r] = L.shape[r];
            } else {
                d.rank = 2;
                rad_enc_plane_dims(&p, rows, cols, &d.shape[0], &d.shape[1]);
            }
            d.access = L.real >= 0 ? RAD_ACCESS_CONDITIONAL : RAD_ACCESS_PER_TOKEN;
            d.shard = RAD_SHARD_NONE;
            d.group.layer = L.layer;
            d.group.expert = L.expert;
            d.group.slot = L.expert >= 0 ? L.slot : 0;
            d.source = L.name.c_str();
            d.planes = p.role;
            const std::string handle = L.enc.n_planes == 1 ? L.name : L.name + "@" + p.role;
            if (rad_decl_weight(b, handle.c_str(), &d) == RAD_NULL_HANDLE) return RAD_E_INVAL;
        }
        ++(L.exl3 ? n_exl3 : n_plain);
    }
    if (n_outside)
        rad_note(b, "%s: %lld experts in %lld layers differ from their layer's group and are tensors of the layer",
                 kPlugin, (long long)n_outside, (long long)n_outside_layers);
    rad_note(b, "%s: %lld EXL3 linears and %lld other tensors of %s, declared under their checkpoint names; a "
                "convert-only plugin (no ops)", kPlugin, (long long)n_exl3, (long long)n_plain, src);
    return RAD_OK;
}

}  // namespace exl3_import

extern "C" const char* rad_arch_id(void) { return exl3_import::env_or("EXL3_IMPORT_ARCH", "exl3import"); }
extern "C" const char* rad_arch_quant(void) { return exl3_import::env_or("EXL3_IMPORT_QUANT", "exl3"); }
/* An imported GPTQ/Quark/... checkpoint is never this plugin's: it reads EXL3 checkpoints only. */
extern "C" int rad_arch_serves_imports(void) { return 0; }
extern "C" int rad_arch_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    return exl3_import::declare(b, m, c);
}
extern "C" void rad_arch_step(struct RadCtx*, const struct RadBatch*) {}
extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {RAD_PLUGIN_ARCH, "exl3_import", "0.1.0",
                                       "convert-only: an EXL3 checkpoint declared tensor for tensor, for rad-convert",
                                       ""};
    return &info;
}
