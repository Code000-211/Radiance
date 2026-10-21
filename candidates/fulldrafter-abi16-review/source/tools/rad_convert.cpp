/* rad-convert -- safetensors or GGUF in, .rad out.
 *
 * A .rad holds each logical weight as the canonical planes of its ENCODING (spec §4.1, §4.2):
 * which kernel library rearranges them at load, and how, is not this tool's concern, and nothing
 * here knows any kernel's layout. What it does is quantise. A RECIPE maps logical weight names to
 * quantiser plugins and their options (spec §4.4); the first rule that matches a weight decides
 * it, and a weight no rule matches keeps the checkpoint's own encoding -- which has to be one the
 * engine serves as it is (bf16, f16, f32, block-scaled FP8), or the recipe is incomplete and the
 * weight is named.
 *
 * IT STILL RUNS DECLARE, for two reasons. The architecture plugin's name map is the only thing
 * that knows which checkpoint tensors make a declared weight -- fusions, alternatives, expert
 * slices. And declare is where every resolved kernel's layout hook is asked about the encoding
 * the recipe plans (rad_weight_encoding answers with it): a recipe nothing can serve is refused
 * before a byte is written, by the same check the engine makes at load.
 *
 * Every weight it quantises is measured against its source through the core's decoder, and the
 * worst are reported: a format change that cannot be ranked is one nobody can argue about.
 */
#include "iface.h"
#include "format/checkpoint.h"
#include "format/encoding.h"
#include "format/imatrix.h"
#include "format/recipe.h"
#include "plugin/registry.h"
#include "text/tokenizer.h"
#include "rad_plugin.h"
#include "rad_quant.h"

#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_set>

using namespace rad;

namespace {

bool file_exists(const std::string& p) { std::ifstream f(p); return (bool)f; }

/* `blk.12.attn_q.weight` -> 12; -1 for a model-level name. The convention is GGUF's, and an
 * imatrix's names follow it. */
int32_t layer_from_name(const std::string& n) {
    if (n.rfind("blk.", 0) != 0) return -1;
    size_t i = 4, v = 0;
    if (i >= n.size() || n[i] < '0' || n[i] > '9') return -1;
    while (i < n.size() && n[i] >= '0' && n[i] <= '9') v = v * 10 + (size_t)(n[i++] - '0');
    return (int32_t)v;
}

/* ================================================================== the recipe, typed */
/* A rule with its quantiser found and its options typed by that quantiser's own table, once: a
 * misspelt key is refused before declare, not discovered at the forty-thousandth weight. The
 * RadParam strings point into `keys` and `vals`, which never change after build(). */
struct Compiled {
    const RecipeRule*       rule = nullptr;
    const RadQuantizerInfo* q = nullptr;
    std::vector<std::string> keys, vals;
    std::vector<RadParam>   params;

    int build(const RecipeRule& r, Registry& reg, std::string* why) {
        rule = &r;
        const Registry::QuantRow* row = reg.quantizer(r.quantizer);
        if (!row || !row->info) {
            std::string have;
            for (const auto& qr : reg.quantizers())
                if (qr.info) have += std::string(have.empty() ? "" : ", ") + qr.info->name;
            *why = r.origin + ": no quantiser '" + r.quantizer + "' is loaded (" +
                   (have.empty() ? std::string("none are -- is $RADIANCE_HOME/quantizers there?")
                                 : "loaded: " + have) + ")";
            return RAD_E_NOTFOUND;
        }
        q = row->info;
        keys.reserve(r.options.size());
        vals.reserve(r.options.size());
        for (const auto& [k, v] : r.options) {
            const RadQuantOption* opt = nullptr;
            for (int i = 0; i < q->n_options; ++i)
                if (k == q->options[i].key) opt = &q->options[i];
            if (!opt) {
                std::string takes;
                for (int i = 0; i < q->n_options; ++i)
                    takes += std::string(i ? ", " : "") + q->options[i].key;
                *why = r.origin + ": " + q->name + " takes no option '" + k + "' (it takes " +
                       takes + ")";
                return RAD_E_INVAL;
            }
            char* end = nullptr;
            if (opt->type == RAD_P_INT) {
                const long long x = std::strtoll(v.c_str(), &end, 0);
                if (!end || *end) { *why = r.origin + ": " + k + "=" + v + " is not an integer";
                                    return RAD_E_INVAL; }
                keys.push_back(k);
                vals.push_back(v);
                params.push_back(RAD_INT(nullptr, x));
            } else if (opt->type == RAD_P_F64) {
                const double x = std::strtod(v.c_str(), &end);
                if (!end || *end) { *why = r.origin + ": " + k + "=" + v + " is not a number";
                                    return RAD_E_INVAL; }
                keys.push_back(k);
                vals.push_back(v);
                params.push_back(RAD_F64(nullptr, x));
            } else {
                keys.push_back(k);
                vals.push_back(v);
                params.push_back(RAD_STR(nullptr, nullptr));
            }
        }
        for (size_t i = 0; i < params.size(); ++i) {
            params[i].key = keys[i].c_str();
            if (params[i].kind == RAD_P_STR) params[i].sval = vals[i].c_str();
        }
        return RAD_OK;
    }
};

/* ================================================================== the plan */
/* One logical weight this run writes: where it comes from, what decides it, what it becomes. */
struct Logical {
    CkptWeight      src;
    const Compiled* rule = nullptr;
    RadEncoding     enc{};
    int             status = RAD_OK;
    std::string     why;
    /* from the declarations that take it */
    bool    declared = false;
    int32_t layer = -1, expert = -1, slot = 0;
    std::string first_decl;
    /* the report */
    double  err = -1, err_imp = -1;
};

RadQuantWeight quant_weight(const Logical& L, const float* importance) {
    RadQuantWeight qw{};
    qw.name = L.src.name.c_str();
    qw.rank = L.src.rank;
    for (uint32_t i = 0; i < RAD_MAX_RANK; ++i) qw.shape[i] = L.src.shape[i];
    qw.rows = L.src.rows;
    qw.cols = L.src.cols;
    qw.layer = L.layer;
    qw.expert = L.expert;
    qw.importance = importance;
    return qw;
}

/* Movement-unit order: model-level weights first, then each layer's own, then its experts in id
 * order with their slots adjacent -- which is what makes base + id * stride true. */
bool plan_less(const Logical* a, const Logical* b) {
    auto key = [](const Logical* p) {
        return std::make_tuple(p->layer < 0 ? -1 : p->layer, p->expert, p->slot, p->src.name);
    };
    return key(a) < key(b);
}

/* THE IMATRIX'S COLUMN IMPORTANCES FOR A WEIGHT, when it has them: its sum of squared inputs a
 * column, per call. The imatrix names a tensor the GGUF way without its suffix, and keeps a MoE
 * layer's experts as one entry -- so `blk.5.ffn_down_exps.9.weight` is expert 9 of
 * `blk.5.ffn_down_exps`. Empty when nothing matches or the widths disagree. */
std::vector<float> importance_of(const ImatrixFile& im, const std::string& name, int64_t cols) {
    std::vector<float> out;
    std::string base = name;
    if (base.size() > 7 && base.compare(base.size() - 7, 7, ".weight") == 0)
        base.resize(base.size() - 7);
    int64_t ex = 0;
    const ImatrixEntry* e = im.find(base);
    if (!e) {
        const size_t dot = base.find_last_of('.');
        if (dot == std::string::npos) return out;
        char* end = nullptr;
        ex = std::strtoll(base.c_str() + dot + 1, &end, 10);
        if (!end || *end || dot + 1 == base.size()) return out;
        e = im.find(base.substr(0, dot));
        if (!e || ex >= e->n_expert) return out;
    }
    if (!e->sum2 || e->cols != cols) return out;
    double calls = e->counts ? (double)e->counts[ex] : (double)e->ncall;
    if (calls <= 0) calls = 1;
    out.resize((size_t)cols);
    for (int64_t j = 0; j < cols; ++j) out[(size_t)j] = (float)(e->sum2[ex * cols + j] / calls);
    return out;
}

void usage() {
    std::printf(
        "rad-convert -- safetensors or GGUF in, .rad out\n"
        "\n"
        "usage: rad-convert [options] <input> -o <model.rad>\n"
        "\n"
        "  <input>              a .gguf, a .safetensors, an index.json, a checkpoint directory,\n"
        "                       or a .rad container (extended, not re-encoded -- see --in-place)\n"
        "  -o, --out FILE       the container to write\n"
        "      --recipe FILE    the quantisation recipe: one rule a line,\n"
        "                         PATTERN  QUANTISER  key=value ...\n"
        "                       the first rule whose glob matches a logical weight name decides\n"
        "                       it; a weight no rule matches keeps the checkpoint's encoding.\n"
        "                       $NAME in a value is the environment's\n"
        "      --quant RULE     'PATTERN=QUANTISER:key=value,...', ahead of the recipe file's\n"
        "                       rules. Repeatable\n"
        "      --list-quantizers  every quantiser loaded and the options each takes, then exit\n"
        "      --arch ID        override the architecture id from the checkpoint\n"
        "      --home DIR       $RADIANCE_HOME (kernels/, architectures/, quantizers/)\n"
        "      --kernels A:B    kernel plugin hierarchy, first wins. Default $RADIANCE_KERNELS.\n"
        "                       Declare checks the recipe against the kernels it resolves\n"
        "      --tokenizer F    tokenizer.json; its declared chain is baked instead of being\n"
        "                       reconstructed from a pre-tokenizer name\n"
        "      --imatrix F      llama.cpp imatrix: the expert-popularity profile, the column\n"
        "                       importances handed to each quantiser, and the weighting of the\n"
        "                       error report\n"
        "      --set K=V        a metadata key, seen by declare AND written into the container,\n"
        "                       so a convert-time choice that must also hold at serve time is\n"
        "                       stated once. Repeatable\n"
        "      --max-tok N      the max_tok declare is run at (default 8192)\n"
        "      --max-ctx N      the context bound declare is run at, as --max-model-len is on\n"
        "                       the engine. Default: the checkpoint's training context\n"
        "      --draft-model DIR  a drafter that ships as its own repository (DFlash2), merged\n"
        "                       under `draft.`\n"
        "      --draft-head FILE  a safetensors file whose tensors are merged UNPREFIXED: the\n"
        "                       target matrices a drafter borrows but does not ship (a bf16\n"
        "                       lm_head, an embedding). Refused on a name already taken\n"
        "      --kv-cache-dtype T  bf16 | fp8, the cache width declare is run at. Default: the\n"
        "                       first of bf16, fp8 at which every op resolves -- the container\n"
        "                       is the same at either, so this only picks the kernels it checks\n"
        "      --max-spec N     the max_spec declare is run at (default 4, or the block a\n"
        "                       drafter states); 0 leaves a draft head out\n"
        "      --reuse FILE     a container converted earlier from the SAME checkpoint. A weight\n"
        "                       it holds with the same encoding, quantiser, options and shape is\n"
        "                       copied from it instead of quantised again\n"
        "      --changed FILE   with --reuse: checkpoint tensors (one name a line) that differ from\n"
        "                       the checkpoint the --reuse container was converted from. Every\n"
        "                       weight that reads one is converted; the rest are copied\n"
        "      --in-place       with --reuse: EXTEND that container instead of writing a new one.\n"
        "                       Its weights stay where they are; new ones are written past its\n"
        "                       end and the header last. Refused when a weight it holds would\n"
        "                       change or a new weight is an expert. <file>.pre-append keeps the\n"
        "                       old header and length\n"
        "      --plan-only      declare, plan every weight's encoding, print the plan, and stop\n"
        "                       before writing anything\n"
        "      --report N       the worst N quantisation errors to print (default 20)\n"
        "  -v, --verbose\n"
        "  -h, --help\n");
}

}  /* namespace */

int main(int argc, char** argv) {
    std::string in, out, arch_override, home = rad_home(), tokjson, imat_path, draft_dir;
    std::vector<std::string> draft_heads;
    std::vector<std::string> hierarchy = rad_hierarchy_from_env();
    std::vector<std::string> recipe_files, quant_flags;
    int64_t  max_tok = 8192;
    int64_t  max_ctx = 0;            /* 0: the checkpoint's training context */
    static const int kMaxSpecDefault = 4;
    int      max_spec = kMaxSpecDefault;
    int      report_n = 20;
    bool     verbose = false, plan_only = false, list_quant = false;
    std::string reuse_path, changed_path;
    bool        in_place = false;
    int         kv_dtype = RAD_DT_INVALID;
    std::vector<std::pair<std::string, std::string>> set_kv;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "rad-convert: %s needs a value\n", what);
                                 std::exit(2); }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-o" || a == "--out") out = next("--out");
        else if (a == "--recipe")    recipe_files.push_back(next("--recipe"));
        else if (a == "--quant")     quant_flags.push_back(next("--quant"));
        else if (a == "--list-quantizers") list_quant = true;
        else if (a == "--arch")      arch_override = next("--arch");
        else if (a == "--home")      home = next("--home");
        else if (a == "--kernels")   hierarchy = rad_split_list(next("--kernels"));
        else if (a == "--tokenizer") tokjson = next("--tokenizer");
        else if (a == "--imatrix")   imat_path = next("--imatrix");
        else if (a == "--set") {
            const std::string kv = next("--set");
            const size_t eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) {
                std::fprintf(stderr, "rad-convert: --set wants KEY=VALUE, got '%s'\n", kv.c_str());
                return 2;
            }
            set_kv.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        }
        else if (a == "--max-tok")   max_tok = std::atoll(next("--max-tok"));
        else if (a == "--max-ctx")   max_ctx = std::atoll(next("--max-ctx"));
        else if (a == "--max-spec")  max_spec = std::atoi(next("--max-spec"));
        else if (a == "--draft-model") draft_dir = next("--draft-model");
        else if (a == "--draft-head") draft_heads.push_back(next("--draft-head"));
        else if (a == "--plan-only") plan_only = true;
        else if (a == "--report")    report_n = std::atoi(next("--report"));
        else if (a == "--reuse")     reuse_path = next("--reuse");
        else if (a == "--changed")   changed_path = next("--changed");
        else if (a == "--in-place")  in_place = true;
        else if (a == "--kv-cache-dtype") {
            const std::string v = next("--kv-cache-dtype");
            if (v == "fp8")       kv_dtype = RAD_F8E4M3;
            else if (v == "bf16") kv_dtype = RAD_BF16;
            else { std::fprintf(stderr, "rad-convert: --kv-cache-dtype takes bf16 "
                                                         "or fp8, not '%s'\n", v.c_str());
                                    return 2; }
        }
        else if (a == "-v" || a == "--verbose") { verbose = true; log_set_level(Log::Debug); }
        else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "rad-convert: unrecognised argument '%s'. Try --help.\n",
                         a.c_str());
            return 2;
        } else if (in.empty()) in = a;
        else { std::fprintf(stderr, "rad-convert: more than one input given\n"); return 2; }
    }

    /* ---------------------------------------------------------------- --list-quantizers */
    if (list_quant) {
        Registry& reg = rad_tools_registry();
        std::vector<std::string> qdirs;
        for (const std::string& d : rad_home_dirs(home)) {
            struct stat st {};
            if (stat((d + "/quantizers").c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                qdirs.push_back(d + "/quantizers");
        }
        if (qdirs.empty()) qdirs.push_back(rad_home_dirs(home).front() + "/quantizers");
        if (reg.load_dirs(qdirs, {}) < 0) return 1;
        for (const auto& qr : reg.quantizers()) {
            if (!qr.info) continue;
            std::printf("%s  (%s)\n    %s\n", qr.info->name, qr.plugin.c_str(), qr.info->doc);
            for (int k = 0; k < qr.info->n_options; ++k) {
                const RadQuantOption& o = qr.info->options[k];
                std::printf("    %-12s %-4s %s%s%s\n", o.key,
                            o.type == RAD_P_INT ? "int" : o.type == RAD_P_F64 ? "num" : "str",
                            o.doc ? o.doc : "", o.deflt ? "  [default " : "",
                            o.deflt ? (std::string(o.deflt) + "]").c_str() : "");
            }
        }
        return 0;
    }

    if (!changed_path.empty() && (reuse_path.empty() || in_place)) {
        std::fprintf(stderr, "rad-convert: --changed says which weights not to copy from the "
                             "container --reuse names, and needs --reuse without --in-place (a "
                             "weight cannot be replaced in place)\n");
        return 2;
    }
    if (in_place) {
        if (reuse_path.empty()) reuse_path = in;
        if (out.empty()) out = reuse_path;
    }
    if (in.empty() || out.empty()) { usage(); return 2; }
    /* A CONTAINER AS THE SOURCE IS EXTENDED, NOT RE-WRITTEN. Its entries are the encoding a
     * multi-plane quantiser produced, and re-encoding them would be the requantisation this route
     * exists to avoid; the metadata and vocabulary it carries are the model's. So a `.rad` input
     * requires --in-place, and --reuse defaults to it. */
    const bool container_in = in.size() > 4 && in.compare(in.size() - 4, 4, ".rad") == 0;
    if (container_in && !in_place) {
        std::fprintf(stderr, "rad-convert: %s is a container. Extending it in place keeps every "
                             "entry's encoding and bytes; writing a second file from it would "
                             "re-encode every weight. Pass --in-place (and -o another path only "
                             "for the copy you own).\n", in.c_str());
        return 2;
    }
    /* THE FILE EXTENDED IS THE FILE READ. An in-place extension records every carried entry at
     * the offset it has in the source, so source and target have to be the same bytes -- and a
     * 113 GiB container is not something this tool copies to make that true. Copy it yourself
     * (`cp --reflink=auto TARGET.rad WORK.rad`, or plain `cp` where the filesystem has no
     * reflink), then extend WORK.rad: the command is the same, pointed at the copy. */
    if (container_in && in_place) {
        struct stat si{}, so{};
        const bool same = ::stat(in.c_str(), &si) == 0 && ::stat(out.c_str(), &so) == 0 &&
                          si.st_dev == so.st_dev && si.st_ino == so.st_ino;
        if (!same) {
            std::fprintf(stderr, "rad-convert: --in-place extends the file it reads, and -o names "
                                 "'%s'. Copy %s yourself and point rad-convert at the copy; this "
                                 "tool will not duplicate a container.\n", out.c_str(),
                         in.c_str());
            return 2;
        }
    }
    /* THE OUTPUT IS OPENED WITH O_TRUNC. Pointed at the --reuse container it would empty the file
     * every weight is about to be copied from; --in-place is the way to write into that file. */
    if (!reuse_path.empty()) {
        struct stat so{}, sr{};
        const bool same = ::stat(out.c_str(), &so) == 0 && ::stat(reuse_path.c_str(), &sr) == 0 &&
                          so.st_dev == sr.st_dev && so.st_ino == sr.st_ino;
        if (same != in_place) {
            std::fprintf(stderr, in_place
                ? "rad-convert: --in-place writes the --reuse container itself; -o names another\n"
                : "rad-convert: -o is the --reuse container, and writing it would truncate the "
                  "file its weights are read from. Pass --in-place to extend it instead\n");
            return 2;
        }
    }

    /* ---------------------------------------------------------------- the recipe */
    Recipe recipe;
    {
        std::string why;
        for (const std::string& q : quant_flags)
            if (recipe.add_flag(q, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        for (const std::string& rf : recipe_files) {
            std::ifstream f(rf);
            if (!f) { RAD_ERR("--recipe %s: cannot read it", rf.c_str()); return 2; }
            std::stringstream ss;
            ss << f.rdbuf();
            if (recipe.add_text(ss.str(), rf, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        }
    }

    /* ---------------------------------------------------------------- source */
    Checkpoint ck;
    if (ck.open(in) < 0) return 1;
    MetaOwn& meta = ck.meta();
    /* The drafter merges BEFORE declare: the architecture plugin decides whether to declare a
     * draft block by looking for `draft.architectures` in the table. */
    if (!draft_dir.empty()) {
        if (ck.add_draft(draft_dir, "draft.") < 0) return 1;
        RAD_INFO("drafter  %s (%s)", draft_dir.c_str(),
                 rad_meta_gets(&meta.m, "draft.architectures", "?"));
    }
    /* THE TARGET MATRICES A DRAFTER BORROWS AND DOES NOT SHIP. A bf16 DFlash2 declares its draft
     * head over the target's `lm_head.weight`; a container whose recipe quantised that head holds
     * no bf16 rows to read, and dequantising its i8 codes would be the lossy wire bf16 mode
     * exists to avoid. This file supplies the original bf16 tensor under its own name, and the
     * name map finds it exactly as it would find it in a safetensors target. UNPREFIXED on
     * purpose: the plugin names `lm_head.weight`, not `draft.lm_head.weight`. */
    for (const std::string& head : draft_heads) {
        if (ck.add_tensors(head, "") < 0) return 1;
        RAD_INFO("head     %s", head.c_str());
    }
    if (!arch_override.empty()) meta.arch = arch_override;
    /* --set, added after the checkpoint's own config and the drafter's, so an operator's key is
     * the last word. Declare reads RadModelMeta and so does the engine's, from the container's
     * header, so a choice that changes the graph is stated once and held at both. */
    for (const auto& kv : set_kv) meta.set(kv.first, kv.second);

    /* THE CONTAINER --reuse NAMES, read with pread and never paged through its mapping. */
    RadFile reuse;
    if (!reuse_path.empty() && reuse.open(reuse_path.c_str()) != RAD_OK) return 1;
    /* THE TENSORS --changed NAMES: a checkpoint that differs from the --reuse container's own in
     * these and nowhere else -- a weight edit, a fine-tune of some matrices. --reuse matches a
     * weight on how it was converted and not on its source bytes, so without this list it would
     * copy the old weight for every one of them. */
    std::unordered_set<std::string> changed, changed_read;
    if (!changed_path.empty()) {
        std::ifstream cf(changed_path);
        if (!cf) { RAD_ERR("--changed %s: cannot read it", changed_path.c_str()); return 1; }
        std::string line;
        while (std::getline(cf, line)) {
            const size_t a = line.find_first_not_of(" \t\r");
            const size_t z = line.find_last_not_of(" \t\r");
            if (a == std::string::npos || line[a] == '#') continue;
            changed.insert(line.substr(a, z - a + 1));
        }
        if (changed.empty()) {
            RAD_ERR("--changed %s names no tensor", changed_path.c_str());
            return 1;
        }
    }
    /* EXTENDED, NOT RE-DECIDED. The container's radiance.* keys are the convert-time choices its
     * weights were made under, and a run that keeps those weights declares under the same ones.
     * `radiance.encoder` is not a choice; this run derives it. */
    int inherited = 0;
    if (in_place) {
        for (int64_t i = 0; i < reuse.meta_count(); ++i) {
            const RadFileKV& kv = reuse.meta_at(i);
            const std::string k = reuse.str(kv.key);
            if (k.rfind("radiance.", 0) != 0 || k == "radiance.encoder" || meta.has(k)) continue;
            std::string v;
            if (kv.type == RAD_P_STR)       v = reuse.str(kv.v.s);
            else if (kv.type == RAD_P_INT)  v = fmt("%lld", (long long)kv.v.i);
            else                            v = fmt("%.17g", kv.v.f);
            meta.kv(k, v);
            ++inherited;
        }
    }
    /* Published once more, unconditionally: every merge above may have moved the strings the
     * previous finish() pointed at. */
    meta.finish();
    if (inherited)
        RAD_INFO("--in-place: %d radiance.* key(s) carried over from %s", inherited,
                 reuse_path.c_str());
    if (reuse.is_open() && std::string(reuse.str(reuse.header().arch_id)) != meta.arch) {
        RAD_ERR("--reuse %s is a '%s' container and this run writes '%s'.", reuse_path.c_str(),
                reuse.str(reuse.header().arch_id), meta.arch.c_str());
        return 1;
    }

    RAD_INFO("source   %s (%zu tensors)", ck.path().c_str(), ck.tensors().size());
    RAD_INFO("arch     %s%s%s", meta.m.arch_id, meta.quant.empty() ? "" : ", quant ",
             meta.quant.c_str());

    /* ---------------------------------------------------------------- plugins */
    LoadedPlugins plugins;
    if (rad_tools_load_plugins(home, hierarchy, meta.arch, meta.quant, &plugins) < 0) {
        RAD_ERR("rad-convert runs the architecture plugin's declare: its name map says which "
                "checkpoint tensors make each weight, and the kernels it resolves are what the "
                "recipe is checked against");
        return 1;
    }
    Registry& reg = rad_tools_registry();
    std::deque<Compiled> compiled;
    std::map<const RecipeRule*, const Compiled*> by_rule;
    for (const RecipeRule& r : recipe.rules()) {
        compiled.emplace_back();
        std::string why;
        if (compiled.back().build(r, reg, &why) < 0) { RAD_ERR("%s", why.c_str()); return 2; }
        by_rule[&r] = &compiled.back();
    }
    if (!recipe.empty()) RAD_INFO("recipe   %zu rule(s)", recipe.rules().size());

    ImatrixFile im;
    if (!imat_path.empty() && im.open(imat_path) < 0) return 1;

    /* ---------------------------------------------------------------- declare */
    RadBuildCtx ctx{};
    ctx.rank = 0;
    ctx.world_size = 1;
    ctx.max_tok = max_tok;
    /* THE CONTEXT BOUND IS A DECLARE INPUT: an architecture whose graph changes shape with it --
     * Qwen3.8-Flash-Next's QSA is exactly dense attention up to its indexer budget and a different
     * attention above -- can refuse to declare at the checkpoint's own number. */
    ctx.max_seqs = 256;
    ctx.max_ctx = max_ctx > 0 ? max_ctx : meta.m.n_ctx_train;
    ctx.scope = "";
    /* NON-ZERO ON PURPOSE: a container carries every weight any deployment of the model could ask
     * for, so a draft head is declared even though converting is not drafting. A drafter trained
     * for one block says so (rad_arch_probe), as it does for `--num-speculative-tokens auto`. */
    if (max_spec == kMaxSpecDefault) {
        RadArchProbe probe{};
        (void)rad_tools_probe(plugins, meta.m, &probe);
        if (probe.draft_depth > 0 && probe.draft_depth_fixed) {
            max_spec = probe.draft_depth;
            RAD_INFO("declaring at max_spec %d, the only depth this drafter serves", max_spec);
        } else if (const char* bs = meta.get("draft.dflash_config.block_size")) {
            /* A block-diffusion drafter denoises a fixed block: `block_size` positions, the
             * anchor plus `block_size - 1` drafts, and rad_block_dflash2.h refuses a block that
             * is not a power of two. Declaring at the default 4 would ask for 5. */
            const int b = std::atoi(bs);
            if (b >= 2 && (b & (b - 1)) == 0) {
                max_spec = b - 1;
                RAD_INFO("declaring at max_spec %d, the block this drafter was trained for",
                         max_spec);
            }
        }
    }
    ctx.max_spec = max_spec;
    /* AND THE ENCODER, for the draft head's reason. */
    if (rad_meta_geti(&meta.m, "vision_config.patch_size", 0) > 0) ctx.max_enc_patches = 16384;

    /* THE ENCODING EVERY WEIGHT WILL HAVE, answered to declare as rad_weight_encoding: the
     * recipe's when a rule matches, the checkpoint's own when it is trivial. Remembered once
     * answered; a name the name map does not know yet is not, since declare may map it later. */
    std::map<std::string, Logical> logical;
    auto plan_weight = [&](const std::string& name, const Program& prog, bool optional,
                           Logical** out) -> int {
        auto it = logical.find(name);
        if (it != logical.end()) { *out = &it->second; return it->second.status; }
        CkptWeight cw;
        std::string why;
        const int rc = ckpt_resolve(ck, prog, name, optional, &cw, &why);
        if (rc == RAD_E_NOTFOUND) return rc;
        Logical L;
        L.src = std::move(cw);
        if (rc < 0) {
            L.status = rc;
            L.why = why;
        } else if (const RecipeRule* r = recipe.match(name)) {
            /* A CONTAINER ENTRY IS NOT RE-ENCODED. A rule that matches one asks for exactly the
             * requantisation an in-place extension exists to avoid; refusing here is earlier and
             * clearer than the writer's "a weight cannot be replaced in place" later. Scope the
             * recipe to the names it was written for. */
            if (L.src.entry) {
                L.status = RAD_E_UNSUPPORTED;
                L.why = r->origin + " (" + r->quantizer + ") matches it, and it is already an "
                        "entry of " + reuse_path + " in its served encoding. An in-place "
                        "extension carries entries as they are; scope the recipe to the weights "
                        "this run adds";
                auto ins = logical.emplace(name, std::move(L)).first;
                *out = &ins->second;
                return ins->second.status;
            }
            L.rule = by_rule[r];
            const RadQuantWeight qw = quant_weight(L, nullptr);
            const int st = L.rule->q->encoding(L.rule->params.data(), (int)L.rule->params.size(),
                                               &qw, &L.enc);
            if (st == RAD_E_UNSUPPORTED && L.src.trivial) {
                /* declined -- a norm under a pattern meant for matrices -- and kept as it is */
                L.rule = nullptr;
                L.enc = L.src.enc;
            } else if (st != RAD_OK) {
                L.status = st;
                L.why = r->origin + " (" + r->quantizer + ") refuses it: " + rad_strerror(st);
            } else if (!rad_enc_valid(&L.enc)) {
                L.status = RAD_E_FORMAT;
                L.why = r->quantizer + " answered with a malformed encoding";
            }
        } else if (!L.src.trivial) {
            L.status = RAD_E_UNSUPPORTED;
            L.why = "no recipe rule matches it, and its checkpoint form is not one the engine "
                    "serves as it is: " + L.src.why;
        } else {
            L.enc = L.src.enc;
        }
        auto ins = logical.emplace(name, std::move(L)).first;
        *out = &ins->second;
        return ins->second.status;
    };
    auto provide = [&](const std::string& name, const Program& prog, WeightSource* ws) -> int {
        Logical* L = nullptr;
        const int st = plan_weight(name, prog, false, &L);
        if (st != RAD_OK) return st;
        ws->enc = L->enc;
        ws->rank = L->src.rank;
        for (int i = 0; i < RAD_MAX_RANK; ++i) ws->shape[i] = L->src.shape[i];
        return RAD_OK;
    };
    auto planning_errors = [&]() {
        int n = 0;
        for (const auto& [name, L] : logical)
            if (L.status != RAD_OK) {
                RAD_ERR("  '%s': %s", name.c_str(), L.why.c_str());
                ++n;
            }
        return n;
    };

    /* AN OP THAT RESOLVED IN NO DOMAIN has no kernel to read its weights, so nothing has checked
     * the encoding planned for them. Device holes alone are placement options, not refusals. */
    auto dead_ops = [](const Program& p) {
        size_t n = 0;
        for (size_t i = 1; i < p.ops.size(); ++i) {
            bool any = false;
            for (const Band& bd : p.ops[i].bands)
                for (int d = 0; d < RAD_N_DOMAINS && !any; ++d) if (bd.dom[d]) any = true;
            if (!any) ++n;
        }
        return n;
    };
    /* THE CACHE WIDTH IS A SERVE CHOICE the container does not record: the weights and their
     * encodings are the same at either, and only the kernels that check them differ -- an
     * attention that writes an fp8 cache may have no bf16 form. Unnamed, the first width at which
     * every op resolves is the one checked against; if none does, the one with the fewest holes
     * is reported. */
    const std::vector<int> widths = kv_dtype != RAD_DT_INVALID
                                        ? std::vector<int>{ kv_dtype }
                                        : std::vector<int>{ RAD_BF16, RAD_F8E4M3 };
    auto width_name = [](int dt) { return dt == RAD_F8E4M3 ? "fp8" : "bf16"; };
    Program prog;
    size_t dead = 0;
    for (size_t k = 0; k < widths.size(); ++k) {
        Program trial;
        ctx.kv_dtype = widths[k];
        if (rad_tools_declare(plugins, meta.m, ctx, &trial, {}, nullptr, provide,
                              /*instances=*/false) < 0) {
            if (planning_errors())
                RAD_ERR("the weights above could not be planned, and declare saw each of them as "
                        "the plain dtype it was declared at -- fix those first");
            return 1;
        }
        const size_t d = dead_ops(trial);
        if (k > 0 && d == 0)
            RAD_INFO("checked at --kv-cache-dtype %s: at %s, %zu op(s) have no kernel",
                     width_name(widths[k]), width_name(widths[0]), dead);
        if (k == 0 || d < dead) { prog = std::move(trial); dead = d; }
        if (dead == 0) break;
    }
    if (dead) {
        RAD_ERR("%zu op(s) resolved to no kernel in any domain at --kv-cache-dtype %s, so nothing "
                "can read the weights they take:", dead, width_name(prog.ctx.kv_dtype));
        for (const auto& m : prog.misses) RAD_ERR("    %s", m.line().c_str());
        return 1;
    }

    /* ---------------------------------------------------------------- the plan */
    /* Every logical weight a declaration takes, once, with the group the declarations give it. */
    int bad = 0;
    for (size_t h = 1; h < prog.weights.size(); ++h) {
        const WeightInfo& w = prog.weights[h];
        Logical* L = nullptr;
        const int st = plan_weight(w.source, prog, w.decl.optional != 0, &L);
        if (st == RAD_E_NOTFOUND && w.decl.optional) continue;
        if (st == RAD_E_NOTFOUND) {
            CkptWeight cw;
            std::string why;
            (void)ckpt_resolve(ck, prog, w.source, false, &cw, &why);
            RAD_ERR("'%s' is declared over '%s': %s", w.name.c_str(), w.source.c_str(),
                    why.c_str());
            ++bad;
            continue;
        }
        if (st != RAD_OK) continue;             /* named below, with the others */
        const RadWeightGroup& g = w.decl.group;
        if (!L->declared) {
            L->declared = true;
            L->layer = g.layer;
            L->expert = g.expert;
            L->slot = g.slot;
            L->first_decl = w.name;
        } else if (L->layer != g.layer || L->expert != g.expert || L->slot != g.slot) {
            RAD_ERR("'%s' and '%s' both take '%s' and put it in different movement groups "
                    "((%d, %d, %d) and (%d, %d, %d)); a weight is stored once",
                    L->first_decl.c_str(), w.name.c_str(), w.source.c_str(), L->layer, L->expert,
                    L->slot, g.layer, g.expert, g.slot);
            ++bad;
        }
        /* THE ENCODING DECLARE CHOSE KERNELS FOR IS THE ONE THAT WILL BE WRITTEN. */
        if (!w.enc_known || !rad_enc_equal(&w.enc, &L->enc)) {
            RAD_ERR("'%s' was declared over an encoding of '%s' other than the one planned for "
                    "it", w.name.c_str(), w.source.c_str());
            ++bad;
        }
    }
    /* A HEAD FILE CARRIES MORE THAN THE ONE TENSOR A DRAFTER BORROWS -- a teacher matrices file
     * holds the embedding beside the head, and the target container already serves that
     * embedding under its own entry name. SAY SO rather than refuse: the merge did what it was
     * asked, and which source won is exactly the thing an operator cannot see from outside. A
     * COLLISION on an entry's own name was already refused at merge time, so nothing here can
     * have replaced a stored weight. */
    if (!ck.merged().empty()) {
        std::unordered_set<std::string> merged_names(ck.merged().begin(), ck.merged().end());
        std::unordered_set<std::string> hit;
        for (const auto& it : logical)
            if (it.second.status == RAD_OK)
                ckpt_reads_any(it.second.src, merged_names, &hit);
        for (const std::string& n : ck.merged()) {
            if (hit.count(n)) {
                RAD_INFO("head     '%s' is read", n.c_str());
                continue;
            }
            RAD_WARN("head     '%s' was merged and no declaration reads it; the container's own "
                     "entry of that weight stands", n.c_str());
        }
    }
    bad += planning_errors();
    if (bad) return 1;

    std::vector<Logical*> plan;
    for (auto& [name, L] : logical)
        if (L.declared && L.status == RAD_OK) plan.push_back(&L);
    std::sort(plan.begin(), plan.end(), plan_less);

    {
        int64_t n_q = 0, n_kept = 0;
        for (const Logical* L : plan) (L->rule ? n_q : n_kept)++;
        RAD_INFO("plan     %zu weight(s): %lld quantised by the recipe, %lld kept as the "
                 "checkpoint holds them", plan.size(), (long long)n_q, (long long)n_kept);
        for (const auto& c : compiled) {
            int64_t n = 0;
            for (const Logical* L : plan) if (L->rule == &c) ++n;
            if (n == 0)
                RAD_WARN("%s: rule '%s %s' matches no weight", c.rule->origin.c_str(),
                         c.rule->pattern.c_str(), c.rule->quantizer.c_str());
        }
    }

    if (plan_only) {
        int64_t bytes = 0;
        for (const Logical* L : plan) {
            int64_t b = 0;
            for (int k = 0; k < L->enc.n_planes; ++k)
                b += rad_enc_plane_bytes(&L->enc.plane[k], L->src.rows, L->src.cols);
            bytes += b;
            if (verbose)
                RAD_DEBUG("  %-52s %-34s %10s  %s%s%s", L->src.name.c_str(),
                          enc_name(L->enc).c_str(), humanb(b).c_str(),
                          L->rule ? L->rule->rule->quantizer.c_str() : "as is",
                          L->rule ? " " : "", L->rule ? L->rule->rule->options_text().c_str() : "");
        }
        RAD_INFO("--plan-only: about %s would be written; nothing was written to %s.",
                 humanb(bytes).c_str(), out.c_str());
        return 0;
    }

    /* ---------------------------------------------------------------- write */
    RadWriter w;
    RAD_TRY(in_place ? w.begin_append(out.c_str(), reuse) : w.begin(out.c_str()));

    w.set_arch(meta.arch);
    /* An extended container keeps its name: the checkpoint this run reads its new weights from
     * need not be the one it was made from. */
    w.set_model_name(in_place ? std::string(reuse.str(reuse.header().model_name))
                              : (meta.name.empty() ? meta.arch : meta.name));
    w.set_quant(meta.quant);
    w.set_recipe(in_place && recipe.empty() ? std::string(reuse.str(reuse.header().recipe))
                                            : recipe.text());
    /* The checkpoint as its last two path components -- for a Hugging Face download, the
     * repository id. The rest of the path is where one machine keeps its models, and a container
     * is meant to leave that machine. */
    std::string ck_id = ck.path();
    while (ck_id.size() > 1 && ck_id.back() == '/') ck_id.pop_back();
    if (const size_t a = ck_id.find_last_of('/'); a != std::string::npos && a > 0)
        if (const size_t b = ck_id.find_last_of('/', a - 1); b != std::string::npos)
            ck_id.erase(0, b + 1);
    w.set_created_by(in_place
        ? fmt("%s; extended in place by rad-convert " __DATE__ " from %s",
              reuse.str(reuse.header().created_by), ck_id.c_str())
        : fmt("rad-convert " __DATE__ " from %s", ck_id.c_str()));

    /* THE MODEL'S METADATA, through the one implementation of that decision: typed fields, then
     * every free-form row at the type the extended container held it (string for a fresh
     * convert or a key this run added). Checkpoint::write_metadata owns it so a test can hold the
     * production code to its contract rather than a copy of it. */
    ck.write_metadata(w, in_place && reuse.is_open() ? &reuse : nullptr);
    /* THE CONTAINER CARRIES AN ENCODER'S WEIGHTS: what `--mm-max-patches auto` keys on. Written
     * only when nothing above already stated it -- a container that carried the key emitted it
     * through write_metadata, and a second row of one key is a table where a lookup can read the
     * stale one. */
    if (prog.encoder.modalities != 0 && !meta.has("radiance.encoder"))
        w.meta_s("radiance.encoder", prog.encoder_name.c_str());

    /* ---------------------------------------------------------------- the weights
     *
     * EVERY WEIGHT IS ADDED FIRST, IN PLAN ORDER, AND WRITTEN AFTER IN ANY ORDER. Adding one is
     * what places it -- its planes' offsets in the blob follow from the ones before it -- and that
     * is the only part of the writer with an order; a plane write is a pwrite at its own offset.
     * So the writing is parallel: a Flash-Next checkpoint is 49152 expert slices of a few
     * megabytes each, and one at a time -- decode, a quantiser's OpenMP region too short to
     * amortise its own fork, the error -- it keeps about four of thirty-two cores busy for an
     * hour.
     *
     * A SMALL WEIGHT IS ONE THREAD'S. The pool takes them in plan order, which is roughly file
     * order, and nested parallelism is off, so a quantiser's own OpenMP region runs on the thread
     * that called it. A LARGE ONE IS EVERY THREAD'S: those run after, one at a time, with the
     * decode, the quantiser and the error each parallel over rows -- a 51 GB n-gram table, an
     * lm_head, the attention linears. */
#ifdef _OPENMP
    omp_set_max_active_levels(1);
#endif
    constexpr int64_t kChunk = (int64_t)64 << 20;
    constexpr int64_t kSmall = (int64_t)32 << 20;   /* f32 bytes one thread quantises alone */

    struct Job {
        Logical*            L = nullptr;
        int64_t             idx = -1;
        const RadFileEntry* re = nullptr;   /* copied from the --reuse container */
        int64_t             bytes = 0;      /* its planes */
    };
    std::vector<Job> jobs;
    jobs.reserve(plan.size());
    int64_t reused_inplace = 0, n_inplace = 0, n_changed = 0;

    /* --- EVERY ENTRY THE CONTAINER HOLDS IS CARRIED, IN ITS OWN ORDER, BEFORE ANYTHING NEW.
     *
     * This is the whole integrity contract of an in-place extension: an entry is recorded at the
     * offset it already has, with its own encoding, quantiser, options, shape and movement unit,
     * so no byte of it is read, decoded or rewritten. It also carries entries this run's
     * declarations no longer name -- an MTP head a DFlash2 drafter displaces is still in the
     * file, still where it was, and still readable by a build that asks for it; dropping it would
     * turn "add a drafter" into "replace the drafter" without saying so.
     *
     * The original directory already satisfies the writer's movement-unit invariant, which is
     * why the entries go back in exactly that order; new weights are added after, and the writer
     * refuses a new expert outright. */
    if (in_place && ck.is_container()) {
        const RadFile& src = ck.container();
        for (int64_t i = 0; i < src.entry_count(); ++i) {
            const RadFileEntry& e = src.entry(i);
            const std::string nm = src.str(e.name);
            RadWriter::Weight ew;
            ew.name      = nm;
            ew.quantizer = src.str(e.quantizer);
            ew.options   = src.str(e.options);
            ew.enc       = src.encoding(e);
            ew.rank      = e.rank;
            for (uint32_t k = 0; k < RAD_MAX_RANK; ++k) ew.shape[k] = e.shape[k];
            ew.layer     = e.layer;
            ew.expert    = e.expert;
            ew.slot      = e.slot;
            /* A DECLARATION, IF THERE IS ONE, MUST AGREE WITH THE ENTRY -- otherwise the kernels
             * were chosen for a different shape and the carried bytes would be served wrongly. */
            auto it = logical.find(nm);
            if (it != logical.end() && it->second.status == RAD_OK) {
                const Logical& L = it->second;
                bool same = L.layer == e.layer && L.expert == e.expert && L.slot == e.slot &&
                            L.src.rank == e.rank;
                for (uint32_t k = 0; same && k < RAD_MAX_RANK; ++k)
                    if (L.src.shape[k] != e.shape[k]) same = false;
                if (!same) {
                    RAD_ERR("'%s' is in %s at rank %u, unit (%d, %d, %d), and this run declares "
                            "it at rank %u, unit (%d, %d, %d). An entry is carried as it is; a "
                            "declaration that disagrees is a plugin and a container built for "
                            "different models.", nm.c_str(), reuse_path.c_str(),
                            (unsigned)e.rank, e.layer, e.expert, e.slot, L.src.rank, L.layer,
                            L.expert, L.slot);
                    w.abort();
                    return 1;
                }
            }
            if (w.add_existing(ew, src, e) < 0) { w.abort(); return 1; }
            reused_inplace += (int64_t)e.bytes;
            ++n_inplace;
        }
    }

    for (Logical* Lp : plan) {
        Logical& L = *Lp;
        /* AN ENTRY ALREADY CARRIED ABOVE IS NOT ADDED AGAIN. */
        if (L.src.entry) continue;
        RadWriter::Weight ww;
        ww.name = L.src.name;
        ww.quantizer = L.rule ? L.rule->rule->quantizer : "";
        ww.options = L.rule ? L.rule->rule->options_text() : "";
        ww.enc = L.enc;
        ww.rank = L.src.rank;
        for (uint32_t i = 0; i < RAD_MAX_RANK; ++i) ww.shape[i] = L.src.shape[i];
        ww.layer = L.layer;
        ww.expert = L.expert;
        ww.slot = L.slot;
        int64_t plane_total = 0;
        for (int k = 0; k < L.enc.n_planes; ++k)
            plane_total += rad_enc_plane_bytes(&L.enc.plane[k], L.src.rows, L.src.cols);

        /* --- REUSED: the same encoding, quantiser, options and shape in the --reuse container
         * are the same planes, given the same checkpoint -- which --reuse leaves to the operator,
         * since nothing in either container records a Hessian's contents. */
        const RadFileEntry* re = reuse.is_open() ? reuse.find(ww.name) : nullptr;
        const bool reads_changed =
            !changed.empty() && ckpt_reads_any(L.src, changed, &changed_read);
        if (re && reads_changed) {
            re = nullptr;
            ++n_changed;
        }
        if (re) {
            bool same = rad_enc_equal(&reuse.encoding(*re), &ww.enc) &&
                        ww.quantizer == reuse.str(re->quantizer) &&
                        ww.options == reuse.str(re->options) && re->rank == ww.rank &&
                        re->layer == ww.layer && re->expert == ww.expert && re->slot == ww.slot;
            for (uint32_t i = 0; same && i < RAD_MAX_RANK; ++i)
                if (re->shape[i] != ww.shape[i]) same = false;
            if (in_place) {
                if (!same) {
                    RAD_ERR("--in-place: '%s' is in %s as %s by '%s %s', and this run would write "
                            "it as %s by '%s %s'. A weight cannot be replaced in place; convert "
                            "to a new file.", ww.name.c_str(), reuse_path.c_str(),
                            enc_name(reuse.encoding(*re)).c_str(), reuse.str(re->quantizer),
                            reuse.str(re->options), enc_name(ww.enc).c_str(),
                            ww.quantizer.c_str(), ww.options.c_str());
                    w.abort();
                    return 1;
                }
                if (w.add_existing(ww, reuse, *re) < 0) { w.abort(); return 1; }
                reused_inplace += (int64_t)re->bytes;
                ++n_inplace;
                continue;
            }
            if (!same) re = nullptr;
        }
        const int64_t idx = w.add_weight(ww);
        if (idx < 0) { w.abort(); return 1; }
        jobs.push_back(Job{ Lp, idx, re, plane_total });
    }
    /* A NAME NO WEIGHT READS changes nothing, so it is a typo or a tensor this program does not
     * convert -- and either way the operator believes a weight is being converted that is not. */
    if (!changed.empty()) {
        std::vector<std::string> unread;
        for (const std::string& n : changed)
            if (!changed_read.count(n)) unread.push_back(n);
        if (!unread.empty()) {
            std::sort(unread.begin(), unread.end());
            RAD_ERR("--changed names %zu tensor(s) no weight of this conversion reads, e.g. '%s'",
                    unread.size(), unread[0].c_str());
            w.abort();
            return 1;
        }
        RAD_INFO("--changed: %zu tensor(s) read by %lld weight(s), which are converted rather "
                 "than copied", changed.size(), (long long)n_changed);
    }

    /* ---------------------------------------------------------------- vocab */
    /* core/text interprets the declared chain and serialises it; the writer places it and owns the
     * one string blob its token text goes into. */
    struct WriterSink : StringSink {
        RadWriter* w;
        rad_stroff intern(std::string_view s) override { return w->intern_string(s); }
    } sink;
    sink.w = &w;

    VocabBuild vb;
    if (!tokjson.empty()) {
        if (parse_tokenizer_json(tokjson.c_str(), vb) < 0) { w.abort(); return 1; }
    } else if (ck.is_container()) {
        /* THE VOCABULARY THE CONTAINER ALREADY CARRIES. It is the model's -- the target's own
         * tokenizer, baked at its conversion -- and an extension must reproduce it exactly: a
         * second tokeniser would re-tokenise every prompt differently while the weights stayed
         * the same. Nothing about it depends on the weights being added. */
        const VocabSection vs = ck.container().vocab();
        if (!vs) {
            RAD_ERR("%s carries no vocabulary section, so there is none to carry into the "
                    "extension", reuse_path.c_str());
            w.abort();
            return 1;
        }
        if (vocab_deserialize(ck.container().base(), (uint64_t)ck.container().size(), vs.h,
                              vb) < 0) {
            RAD_ERR("%s: its vocabulary section does not read back", reuse_path.c_str());
            w.abort();
            return 1;
        }
        RAD_INFO("vocab    %u token(s), carried from %s", (unsigned)vb.tokens.size(),
                 reuse_path.c_str());
    } else if (ck.is_gguf()) {
        /* GGUF has no declared chain, only the pre-tokeniser NAME llama.cpp derived by hashing the
         * original tokenizer.json. core/text reconstructs from that name where it can and REFUSES
         * where it cannot -- point --tokenizer at the real file when this fails. */
        if (vocab_from_gguf(ck.gguf(), vb) < 0) {
            RAD_ERR("pass --tokenizer <tokenizer.json> to bake the declared chain instead of the "
                    "one a pre-tokeniser name implies.");
            w.abort();
            return 1;
        }
    } else if (file_exists(ck.dir() + "/tokenizer.json")) {
        const std::string tj = ck.dir() + "/tokenizer.json";
        RAD_INFO("vocab    %s", tj.c_str());
        if (parse_tokenizer_json(tj.c_str(), vb) < 0) { w.abort(); return 1; }
    } else {
        RAD_ERR("a safetensors checkpoint carries no vocab and there is no tokenizer.json beside "
                "it. Pass --tokenizer tokenizer.json.");
        w.abort();
        return 1;
    }
    /* Serialised at section offset 0: the writer does not know where the section lands until it
     * has sized the string blob this call is still growing, so finish() rebases. */
    std::vector<uint8_t> vocab_blob;
    if (vocab_serialize(vb, sink, 0, vocab_blob) < 0) { w.abort(); return 1; }
    if (w.set_vocab_section(vocab_blob.data(), (int64_t)vocab_blob.size()) < 0) {
        w.abort();
        return 1;
    }

    /* ---------------------------------------------------------------- expert profile */
    if (im.entries().size()) {
        /* Shares are per layer, so the planner compares experts within a layer rather than
         * across layers with different token counts. */
        std::map<int, std::vector<std::pair<int, double>>> per_layer;
        for (const auto& e : im.entries()) {
            if (e.n_expert <= 1 || !e.counts) continue;
            const int32_t layer = layer_from_name(e.name);
            if (layer < 0) continue;
            for (int64_t x = 0; x < e.n_expert; ++x)
                per_layer[layer].emplace_back((int)x, im.count(e.name, x));
        }
        int64_t rows = 0;
        for (auto& [layer, v] : per_layer) {
            double tot = 0;
            for (auto& [e, cnt] : v) tot += cnt;
            if (tot <= 0) continue;
            for (auto& [e, cnt] : v) { w.add_profile(layer, e, (float)(cnt / tot)); ++rows; }
        }
        RAD_INFO("expert profile: %lld row(s) from %s", (long long)rows, imat_path.c_str());
        if (rows == 0)
            RAD_WARN("the imatrix carried no per-expert counts, so the planner will start cold. "
                     "A legacy-format imatrix cannot carry them at all.");
    }
    /* An extended container keeps the profile it had, unless this run brought its own. */
    if (in_place && imat_path.empty())
        for (int64_t i = 0; i < reuse.profile_count(); ++i) {
            const RadFileProfile& pr = reuse.profile()[i];
            w.add_profile(pr.layer, pr.expert, pr.share);
        }

    /* EVERY TABLE IS COMPLETE NOW -- the weights placed, the vocab and the profile in -- so the
     * blob can be written where it will stay rather than spilled and copied in at the end, which
     * on a 113 GiB container is minutes of the disk doing nothing but move it (RadWriter::seal).
     * An extended container already writes in place. */
    if (!in_place && w.seal() < 0) { w.abort(); return 1; }

    std::atomic<int64_t> written{0}, reused{reused_inplace}, n_reused{n_inplace};

    /* Read `n` rows from `r0` of a checkpoint weight as f32, the rows split across threads. */
    auto rows_f32 = [](const CkptWeight& src, int64_t r0, int64_t n, float* out,
                       std::string* why) -> int {
        const int64_t piece = std::max<int64_t>(1, (n + 255) / 256);
        const int64_t np = (n + piece - 1) / piece;
        int rc = RAD_OK;
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t p = 0; p < np; ++p) {
            const int64_t a = p * piece, m = std::min(piece, n - a);
            std::string w1;
            if (ckpt_rows_f32(src, r0 + a, m, out + a * src.cols, &w1) != RAD_OK) {
#pragma omp critical(rad_convert_err)
                if (rc == RAD_OK) { rc = RAD_E_FORMAT; *why = w1; }
            }
        }
        return rc;
    };

    /* One weight, start to finish. Its own buffers, so any thread may run it. */
    auto run_job = [&](const Job& j) -> int {
        Logical& L = *j.L;
        const std::string& name = L.src.name;
        const int64_t idx = j.idx;
        std::vector<uint8_t> stage;

        if (j.re) {
            for (int k = 0; k < L.enc.n_planes; ++k) {
                const RadFilePlane& fp = reuse.plane(*j.re, k);
                for (int64_t off = 0; off < (int64_t)fp.bytes; off += kChunk) {
                    const int64_t n = std::min<int64_t>(kChunk, (int64_t)fp.bytes - off);
                    stage.resize((size_t)n);
                    if (reuse.read_at(stage.data(), n, (int64_t)fp.offset + off) != RAD_OK ||
                        w.write_plane(idx, k, off, stage.data(), n) != RAD_OK)
                        return RAD_E_IO;
                }
                reuse.drop_range((int64_t)fp.offset, (int64_t)fp.bytes);
            }
            written += j.bytes;
            reused += j.bytes;
            ++n_reused;
            if (verbose)
                RAD_DEBUG("  %-52s %-30s %10s reused", name.c_str(), enc_name(L.enc).c_str(),
                          humanb(j.bytes).c_str());
            return RAD_OK;
        }

        /* --- KEPT AS THE CHECKPOINT HOLDS IT: each plane's bytes from the mapped source, joined
         * where the name map joins pieces. A join along rows is the pieces one after another, so
         * each goes straight from the mapping; any other join is assembled a chunk at a time. */
        if (!L.rule) {
            for (int k = 0; k < L.enc.n_planes; ++k) {
                const std::vector<const SrcTensor*>& src = k == 0 ? L.src.pieces : L.src.scales;
                const int64_t pb = w.plane_bytes(idx, k);
                if (L.src.concat_dim == 0 || src.size() == 1) {
                    int64_t off = 0;
                    for (const SrcTensor* s : src) {
                        if (w.write_plane(idx, k, off, s->data, s->bytes) != RAD_OK)
                            return RAD_E_IO;
                        off += s->bytes;
                    }
                    if (off != pb) {
                        RAD_ERR("'%s': its sources hold %lld bytes of plane '%s' and the "
                                "encoding sizes it at %lld", name.c_str(), (long long)off,
                                L.enc.plane[k].role, (long long)pb);
                        return RAD_E_SHAPE;
                    }
                } else {
                    int64_t pr = 0, pc = 0;
                    rad_enc_plane_dims(&L.enc.plane[k], L.src.rows, L.src.cols, &pr, &pc);
                    const int64_t rb = rad_enc_row_bytes(L.enc.plane[k].dtype, pc);
                    const int64_t step = std::max<int64_t>(1, kChunk / std::max<int64_t>(rb, 1));
                    for (int64_t r0 = 0; r0 < pr; r0 += step) {
                        const int64_t nr = std::min(step, pr - r0);
                        stage.resize((size_t)(nr * rb));
                        std::string why;
                        if (ckpt_plane_rows(L.src, k, r0, nr, stage.data(), &why) != RAD_OK) {
                            RAD_ERR("'%s': %s", name.c_str(), why.c_str());
                            return RAD_E_FORMAT;
                        }
                        if (w.write_plane(idx, k, r0 * rb, stage.data(), nr * rb) != RAD_OK)
                            return RAD_E_IO;
                    }
                }
            }
            written += j.bytes;
            if (verbose)
                RAD_DEBUG("  %-52s %-30s %10s as is", name.c_str(), enc_name(L.enc).c_str(),
                          humanb(j.bytes).c_str());
            return RAD_OK;
        }

        /* --- QUANTISED, a block of rows at a time. The quantiser says how many rows one call
         * may take (0: all of them); the converter takes a multiple of that near 64 MiB of f32,
         * so a 51 GB table streams and a GPTQ weight is handed whole. A plane banded by rows is
         * written as each block lands; a plane that covers every row -- a codebook, a per-tensor
         * scale -- is held until the last call has written it. */
        const Compiled& c = *L.rule;
        const std::vector<float> imp = imat_path.empty()
                                           ? std::vector<float>()
                                           : importance_of(im, name, L.src.cols);
        const RadQuantWeight qw = quant_weight(L, imp.empty() ? nullptr : imp.data());
        {
            /* the group is known now and the encoding was chosen without it: it may not move */
            RadEncoding again{};
            if (c.q->encoding(c.params.data(), (int)c.params.size(), &qw, &again) != RAD_OK ||
                !rad_enc_equal(&again, &L.enc)) {
                RAD_ERR("'%s': %s answers with another encoding once it is told the weight's "
                        "movement group, and declare could not tell it that. An encoding may "
                        "depend on the weight's name and shape, not on where it is placed.",
                        name.c_str(), c.q->name);
                return RAD_E_STATE;
            }
        }
        const int64_t R = L.src.rows, C = L.src.cols;
        int64_t rb = c.q->row_block ? c.q->row_block(c.params.data(), (int)c.params.size(), &qw)
                                    : 0;
        if (rb <= 0 || rb > R) rb = R;
        int64_t blk = rb;
        if (rb < R) {
            const int64_t want = std::max<int64_t>(1, kChunk / std::max<int64_t>(C * 4, 1));
            blk = std::max<int64_t>(rb, want - want % rb);
            if (blk > R) blk = R;
        }
        const int np = L.enc.n_planes;
        std::vector<int64_t> prow_bytes((size_t)np), pband((size_t)np);
        std::vector<std::vector<uint8_t>> whole((size_t)np), band((size_t)np);
        bool misaligned = false;
        for (int k = 0; k < np; ++k) {
            const RadEncPlane& p = L.enc.plane[k];
            int64_t pr = 0, pc = 0;
            rad_enc_plane_dims(&p, R, C, &pr, &pc);
            prow_bytes[(size_t)k] = rad_enc_row_bytes(p.dtype, pc);
            const bool banded = p.kind == RAD_PLANE_TILED && p.block[0] > 0;
            pband[(size_t)k] = banded ? p.block[0] : 0;
            if (banded && blk < R && blk % p.block[0]) misaligned = true;
            if (!banded) whole[(size_t)k].assign((size_t)(pr * prow_bytes[(size_t)k]), 0);
        }
        if (misaligned) {
            RAD_ERR("'%s': %s takes %lld rows a call, which splits a block of its own encoding's "
                    "scales", name.c_str(), c.q->name, (long long)rb);
            return RAD_E_SHAPE;
        }
        /* Measured through the decoder every reader uses: the core's for plain and affine, the
         * quantiser's own hook for a scheme of its own (core/format/encoding.h). A quantiser's
         * decoder is not promised to be reentrant, so its rows are decoded in one call. */
        bool core_decodes = true;
        const bool own_scheme = !rad_enc_is(&L.enc, "plain") && !rad_enc_is(&L.enc, "affine");
        std::vector<float> f, ref, dec;
        double num = 0, den = 0, num_i = 0, den_i = 0;
        for (int64_t r0 = 0; r0 < R; r0 += blk) {
            const int64_t n = std::min(blk, R - r0);
            f.resize((size_t)(n * C));
            std::string why;
            if (rows_f32(L.src, r0, n, f.data(), &why) != RAD_OK) {
                RAD_ERR("'%s': %s", name.c_str(), why.c_str());
                return RAD_E_FORMAT;
            }
            ref = f;                               /* a quantiser may rotate its input in place */
            std::vector<void*> ptr((size_t)np);
            for (int k = 0; k < np; ++k) {
                if (!pband[(size_t)k]) { ptr[(size_t)k] = whole[(size_t)k].data(); continue; }
                const int64_t rows_k = (n + pband[(size_t)k] - 1) / pband[(size_t)k];
                band[(size_t)k].assign((size_t)(rows_k * prow_bytes[(size_t)k]), 0);
                ptr[(size_t)k] = band[(size_t)k].data();
            }
            const int st = c.q->quantize(c.params.data(), (int)c.params.size(), &qw, f.data(),
                                         r0, n, ptr.data());
            if (st != RAD_OK) {
                RAD_ERR("'%s': %s could not quantise rows [%lld, %lld): %s", name.c_str(),
                        c.q->name, (long long)r0, (long long)(r0 + n), rad_strerror(st));
                return st;
            }
            for (int k = 0; k < np; ++k) {
                if (!pband[(size_t)k]) continue;
                if (w.write_plane(idx, k, (r0 / pband[(size_t)k]) * prow_bytes[(size_t)k],
                                  band[(size_t)k].data(), (int64_t)band[(size_t)k].size()) !=
                    RAD_OK)
                    return RAD_E_IO;
            }
            /* THE ERROR, through the core's decoder over exactly these rows. */
            if (core_decodes) {
                EncodedView v;
                const void* const* pd = (const void* const*)ptr.data();
                enc_view(L.enc, n, C, pd, &v);
                dec.resize((size_t)(n * C));
                const int64_t piece = own_scheme ? n : std::max<int64_t>(1, (n + 255) / 256);
                const int64_t npc = (n + piece - 1) / piece;
                int drc = RAD_OK;
#pragma omp parallel for schedule(dynamic, 1) if (!own_scheme)
                for (int64_t p = 0; p < npc; ++p) {
                    const int64_t a = p * piece, m = std::min(piece, n - a);
                    std::string w1;
                    if (enc_decode_rows(v, a, m, dec.data() + a * C, &w1) != RAD_OK) {
#pragma omp critical(rad_convert_err)
                        if (drc == RAD_OK) { drc = RAD_E_FORMAT; why = w1; }
                    }
                }
                if (drc != RAD_OK) {
                    RAD_WARN("'%s': nothing decodes what %s wrote (%s); its error is not "
                             "reported", name.c_str(), c.q->name, why.c_str());
                    core_decodes = false;
                } else {
                    const float* pi = imp.empty() ? nullptr : imp.data();
                    const int64_t cnt = n * C;
#pragma omp parallel for reduction(+ : num, den, num_i, den_i) schedule(static)
                    for (int64_t i = 0; i < cnt; ++i) {
                        const double d = (double)ref[(size_t)i] - dec[(size_t)i];
                        const double x = ref[(size_t)i];
                        num += d * d;
                        den += x * x;
                        if (pi) {
                            const double wj = pi[i % C];
                            num_i += wj * d * d;
                            den_i += wj * x * x;
                        }
                    }
                }
            }
        }
        for (int k = 0; k < np; ++k)
            if (!pband[(size_t)k] &&
                w.write_plane(idx, k, 0, whole[(size_t)k].data(),
                              (int64_t)whole[(size_t)k].size()) != RAD_OK)
                return RAD_E_IO;
        if (core_decodes && den > 0) L.err = std::sqrt(num / den);
        if (core_decodes && den_i > 0) L.err_imp = std::sqrt(num_i / den_i);
        written += j.bytes;
        if (verbose)
            RAD_DEBUG("  %-52s %-30s %10s %s rel %.3e", name.c_str(), enc_name(L.enc).c_str(),
                      humanb(j.bytes).c_str(), c.q->name, L.err);
        return RAD_OK;
    };

    std::vector<size_t> small, large;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const Logical& L = *jobs[i].L;
        const bool big = !jobs[i].re && L.rule && L.src.rows * L.src.cols * 4 > kSmall;
        (big ? large : small).push_back(i);
    }
    std::atomic<int> failed{RAD_OK};
    const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < small.size(); ++i) {
        if (failed.load(std::memory_order_relaxed) != RAD_OK) continue;
        const int st = run_job(jobs[small[i]]);
        if (st < 0) {
            int ok = RAD_OK;
            failed.compare_exchange_strong(ok, st);
        }
    }
    for (size_t i = 0; i < large.size() && failed.load() == RAD_OK; ++i) {
        const int st = run_job(jobs[large[i]]);
        if (st < 0) failed = st;
    }
    if (failed.load() != RAD_OK) { w.abort(); return 1; }
    RAD_INFO("weights  %zu written in %.0f s: %zu a thread each, %zu across every thread",
             jobs.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                              .count(), small.size(), large.size());

    const auto tf = std::chrono::steady_clock::now();
    if (w.finish() < 0) return 1;
    RAD_INFO("finish   %.0f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - tf)
                                    .count());

    /* ---------------------------------------------------------------- the report */
    RAD_INFO("wrote %s: %lld weight(s), %s of weights, %u vocab token(s)", out.c_str(),
             (long long)w.weight_count(), humanb(written.load()).c_str(),
             (unsigned)vb.tokens.size());
    if (in_place)
        RAD_INFO("--in-place: %lld weight(s), %s, kept where they were in %s; %lld added past "
                 "its end", (long long)n_reused.load(), humanb(reused.load()).c_str(),
                 reuse_path.c_str(), (long long)(w.weight_count() - n_reused.load()));
    else if (reuse.is_open())
        RAD_INFO("--reuse: %lld weight(s), %s, copied from %s; the other %lld were written",
                 (long long)n_reused.load(), humanb(reused.load()).c_str(), reuse_path.c_str(),
                 (long long)(w.weight_count() - n_reused.load()));

    std::vector<const Logical*> measured;
    for (const Logical* L : plan) if (L->err >= 0) measured.push_back(L);
    if (!measured.empty()) {
        const bool by_imp = !imat_path.empty();
        auto key = [&](const Logical* L) { return by_imp && L->err_imp >= 0 ? L->err_imp : L->err; };
        std::sort(measured.begin(), measured.end(),
                  [&](const Logical* a, const Logical* b) { return key(a) > key(b); });
        std::map<std::string, std::pair<int64_t, double>> per_rule;
        for (const Logical* L : measured) {
            auto& pr = per_rule[L->rule->rule->origin];
            ++pr.first;
            pr.second += L->err;
        }
        RAD_INFO("quantisation error, relative Frobenius norm%s:",
                 by_imp ? " (imatrix-weighted where the imatrix covers the weight)" : "");
        for (const auto& [origin, pr] : per_rule)
            RAD_INFO("  %-24s %6lld weight(s), mean %.3e", origin.c_str(), (long long)pr.first,
                     pr.second / (double)pr.first);
        const size_t n = verbose ? measured.size()
                                 : std::min<size_t>(measured.size(), (size_t)std::max(report_n, 0));
        if (n) RAD_INFO("  worst %zu:", n);
        for (size_t i = 0; i < n; ++i) {
            const Logical* L = measured[i];
            if (L->err_imp >= 0)
                RAD_INFO("    %.3e  (%.3e weighted)  %-52s %s", L->err, L->err_imp,
                         L->src.name.c_str(), enc_name(L->enc).c_str());
            else
                RAD_INFO("    %.3e  %-52s %s", L->err, L->src.name.c_str(),
                         enc_name(L->enc).c_str());
        }
    }
    return 0;
}
