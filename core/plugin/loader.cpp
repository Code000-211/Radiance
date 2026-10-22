/* loader.cpp -- dlopen, the ABI gate, the configured hierarchy, and the registration of one
 * plugin's schemas and kernel rows. spec.md §2, §18.
 */
#include "registry.h"
#include "../device/device.h"
#include "../format/encoding.h"

#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <climits>
#include <cerrno>
#include <cstdarg>
#include <cstring>

namespace rad {

namespace {

/* dlsym returns void*; converting it to a function pointer is conditionally supported in
 * standard C++ and works on every target here. The cast is isolated in one place. */
template <class Fn>
Fn dl_fn(void* h, const char* name) {
    dlerror();
    return reinterpret_cast<Fn>(dlsym(h, name));
}

std::string stem_of(const std::string& path) {
    size_t slash = path.find_last_of('/');
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    if (base.size() > 3 && base.compare(base.size() - 3, 3, ".so") == 0)
        base.resize(base.size() - 3);
    if (base.compare(0, 3, "lib") == 0 && base.size() > 3) base.erase(0, 3);
    return base;
}

const char* or_empty(const char* s) { return s ? s : ""; }

/* Whether a plugin's declared build target list (separated by spaces, commas or semicolons)
 * names this device's architecture. */
bool target_covers(const std::string& declared, const char* arch) {
    for (size_t i = 0; i < declared.size(); ) {
        size_t e = declared.find_first_of(" ,;", i);
        if (e == std::string::npos) e = declared.size();
        if (e > i && declared.compare(i, e - i, arch) == 0) return true;
        i = e + 1;
    }
    return false;
}

/* Two schemas for one op are the same when every positional slot means the same thing: parameter
 * keys, types and requiredness (what the selector matches on) and operand count, roles and
 * optionality (arguments are positional). Operand names are for diagnostics only (ABI), so "b"
 * and "w_gate" in slot 1 agree. */
bool schema_same(const RadOpSchema& a, const RadOpSchema& b, std::string* diff) {
    if (a.n_params != b.n_params) {
        *diff = fmt("%d parameters vs %d", a.n_params, b.n_params);
        return false;
    }
    for (int i = 0; i < a.n_params; ++i) {
        const RadParamSpec& x = a.params[i];
        const RadParamSpec& y = b.params[i];
        if (std::strcmp(or_empty(x.key), or_empty(y.key)) != 0) {
            *diff = fmt("parameter %d is '%s' vs '%s'", i, or_empty(x.key), or_empty(y.key));
            return false;
        }
        if (x.type != y.type) {
            *diff = fmt("parameter '%s' is %s vs %s", or_empty(x.key), param_kind_name(x.type),
                        param_kind_name(y.type));
            return false;
        }
        if (x.required != y.required) {
            *diff = fmt("parameter '%s' is %s vs %s", or_empty(x.key),
                        x.required ? "required" : "optional", y.required ? "required" : "optional");
            return false;
        }
    }
    if (a.n_operands != b.n_operands) {
        *diff = fmt("%d operands vs %d", a.n_operands, b.n_operands);
        return false;
    }
    for (int i = 0; i < a.n_operands; ++i) {
        if (a.operands[i].role != b.operands[i].role ||
            a.operands[i].optional != b.operands[i].optional) {
            *diff = fmt("operand %d ('%s' / '%s') has a different role", i,
                        or_empty(a.operands[i].name), or_empty(b.operands[i].name));
            return false;
        }
    }
    return true;
}

}  /* namespace */

/* ------------------------------------------------------------------ diagnostics */
void Registry::err_(const char* f, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, f);
    vsnprintf(msg, sizeof msg, f, ap);
    va_end(ap);
    load_errors_.emplace_back(msg);
    RAD_ERR("%s", msg);
}

/* ------------------------------------------------------------------ staging */
/* dlopen and read the plugin's identity without registering anything. Loading has two phases
 * because the hierarchy is ordered by plugin name, which lives inside the .so: the whole directory
 * is opened before anything registers, and registration order fixes an op's schema. */
int Registry::stage_(const std::string& path, Staged* out) {
    out->path = path;
    out->file = stem_of(path);

    /* RTLD_LOCAL is required: every kernel plugin exports the same symbol names (rad_kernel_count
     * and friends), so with RTLD_GLOBAL the first plugin loaded would answer for all later ones.
     * Symbols are looked up per handle. */
    out->code_from = rad::dev_code_mark();
    out->handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    out->code_to = rad::dev_code_mark();
    if (!out->handle) {
        err_("plugin '%s' failed to load: %s", path.c_str(), or_empty(dlerror()));
        return RAD_E_IO;
    }

    auto abi = dl_fn<uint32_t (*)(void)>(out->handle, "rad_plugin_abi_version");
    if (!abi) {
        err_("'%s' is not a radiance plugin: it exports no rad_plugin_abi_version", path.c_str());
        unstage_(*out);
        return RAD_E_ABI;
    }
    const uint32_t v = abi();
    if (v != RAD_ABI_VERSION) {
        /* Named by file, not RadPluginInfo::name: that struct's layout may differ across ABI
         * versions. Both versions are printed so the operator can tell which side is stale. */
        err_("plugin '%s' was built against ABI version %u; this engine is version %u -- refused",
             path.c_str(), v, (unsigned)RAD_ABI_VERSION);
        unstage_(*out);
        return RAD_E_ABI;
    }

    auto info_fn = dl_fn<const RadPluginInfo* (*)(void)>(out->handle, "rad_plugin_info");
    if (!info_fn || !(out->info = info_fn())) {
        err_("plugin '%s' exports no usable rad_plugin_info", path.c_str());
        unstage_(*out);
        return RAD_E_INVAL;
    }
    out->name = out->info->name && out->info->name[0] ? out->info->name : out->file;
    if (auto ref = dl_fn<RadPluginReferenceFn>(out->handle, "rad_plugin_reference"))
        out->reference = out->info->kind == RAD_PLUGIN_KERNEL && ref() != 0;
    return RAD_OK;
}

void Registry::unstage_(Staged& s) {
    if (s.handle) dlclose(s.handle);
    s.handle = nullptr;
    s.info = nullptr;
}

/* ------------------------------------------------------------------ registration */
int Registry::commit_(Staged& s, int order) {
    for (const Plugin& p : plugins_) {
        if (p.name == s.name) {
            err_("two plugins claim the name '%s': %s and %s -- refusing the second",
                 s.name.c_str(), p.path.c_str(), s.path.c_str());
            unstage_(s);
            return RAD_E_DUPLICATE;
        }
    }

    Plugin p;
    p.name         = s.name;
    p.file         = s.file;
    p.path         = s.path;
    p.version      = or_empty(s.info->version);
    p.description  = or_empty(s.info->description);
    p.build_target = or_empty(s.info->build_target);
    p.kind         = s.info->kind;
    p.reference    = s.reference;
    p.order        = order;
    p.handle       = s.handle;
    p.close        = dl_fn<RadPluginCloseFn>(s.handle, "rad_plugin_close");

    std::vector<const RadOpSchema*> staged_schemas;
    std::vector<std::pair<const RadKernelInfo*, int>> staged_rows;   /* the row and its index */
    std::vector<const RadQuantizerInfo*> staged_quants;
    int n_declared_schemas = 0;

    if (p.kind == RAD_PLUGIN_ARCH) {
        auto id_fn = dl_fn<const char* (*)(void)>(s.handle, "rad_arch_id");
        p.arch_declare = dl_fn<int (*)(RadBuilder*, const RadModelMeta*, const RadBuildCtx*)>(
                             s.handle, "rad_arch_declare");
        p.arch_step = dl_fn<void (*)(RadCtx*, const RadBatch*)>(s.handle, "rad_arch_step");
        if (!id_fn || !p.arch_declare || !p.arch_step) {
            err_("architecture plugin '%s' is missing rad_arch_id, rad_arch_declare or "
                 "rad_arch_step", p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }
        p.arch_id = or_empty(id_fn());
        if (p.arch_id.empty()) {
            err_("architecture plugin '%s' returns an empty architecture id", p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }
        /* The second half of the key. Optional: a plugin without it serves an unquantised
         * container, which "" means. */
        if (auto q_fn = dl_fn<const char* (*)(void)>(s.handle, "rad_arch_quant"))
            p.arch_quant = or_empty(q_fn());
        /* Optional: absent means the plugin may serve an imported quantisation (startup.h). */
        if (auto i_fn = dl_fn<int (*)(void)>(s.handle, "rad_arch_serves_imports"))
            p.arch_serves_imports = i_fn() != 0;

        /* Optional: answers what the core needs before it builds the RadBuildCtx for declare.
         * Absent means every answer is zero (rad_builder.h). */
        p.arch_probe = dl_fn<int (*)(const RadModelMeta*, RadArchProbe*)>(s.handle,
                                                                         "rad_arch_probe");
        /* Optional: absent means the reply format is read off the chat template. */
        p.arch_chat_format = dl_fn<const RadChatFormat* (*)(const RadModelMeta*)>(
            s.handle, "rad_arch_chat_format");
        /* Optional: absent means the model serves no speech route. */
        p.arch_audio_format = dl_fn<const RadAudioFormat* (*)(const RadModelMeta*)>(
            s.handle, "rad_arch_audio_format");
        /* Optional: absent means an audio generator's prompt takes a request's texts as given. */
        p.arch_text_prepare = dl_fn<int64_t (*)(int, const char*, char*, int64_t)>(
            s.handle, "rad_arch_text_prepare");
        /* A video generator's rotary positions; the driver refuses to start without it. */
        p.arch_video_positions = dl_fn<int64_t (*)(int, const RadVideoGeom*, float*, int64_t)>(
            s.handle, "rad_arch_video_positions");

        /* The (architecture id, quantisation) pair selects the plugin, so no two plugins may claim
         * the same pair (§2.4). Sharing an id at different quantisations is normal (qwen35_bf16
         * and qwen35_fp8). */
        for (const Plugin& q : plugins_) {
            if (q.arch_id == p.arch_id && q.arch_quant == p.arch_quant) {
                err_("architecture id '%s' at quantisation '%s' is claimed by both '%s' and '%s' "
                     "-- refusing the second", p.arch_id.c_str(),
                     p.arch_quant.empty() ? "(unquantised)" : p.arch_quant.c_str(),
                     q.name.c_str(), p.name.c_str());
                unstage_(s);
                return RAD_E_DUPLICATE;
            }
        }
    } else if (p.kind == RAD_PLUGIN_KERNEL) {
        auto sc_n  = dl_fn<int (*)(void)>(s.handle, "rad_kernel_schema_count");
        auto sc_at = dl_fn<const RadOpSchema* (*)(int)>(s.handle, "rad_kernel_schema_at");
        auto k_n   = dl_fn<int (*)(void)>(s.handle, "rad_kernel_count");
        auto k_at  = dl_fn<const RadKernelInfo* (*)(int)>(s.handle, "rad_kernel_at");
        if (!sc_n || !sc_at || !k_n || !k_at) {
            err_("kernel plugin '%s' is missing rad_kernel_schema_count/at or "
                 "rad_kernel_count/at", p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }

        /* All schemas are checked before any is committed, so a plugin refused on one op leaves
         * nothing registered. */
        n_declared_schemas = sc_n();
        for (int i = 0; i < n_declared_schemas; ++i) {
            const RadOpSchema* sch = sc_at(i);
            if (!sch || !sch->op || !sch->op[0]) {
                err_("kernel plugin '%s' declares a schema with no op name at index %d",
                     p.name.c_str(), i);
                unstage_(s);
                return RAD_E_INVAL;
            }
            std::string diff;
            auto it = schemas_.find(sch->op);
            if (it != schemas_.end()) {
                if (!schema_same(*it->second.schema, *sch, &diff)) {
                    /* Disagreeing positional schemas would silently change a call's meaning
                     * (§2.3). The whole plugin is refused, not just the op: it was built against
                     * a different vocabulary. */
                    err_("op '%s' has two different schemas: '%s' fixed it first, '%s' disagrees "
                         "(%s) -- refusing '%s'",
                         sch->op, it->second.plugin.c_str(), p.name.c_str(), diff.c_str(),
                         p.name.c_str());
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
                continue;   /* same-shape re-declaration: expected, every plugin declares its ops */
            }
            for (const RadOpSchema* q : staged_schemas) {
                if (std::strcmp(q->op, sch->op) == 0 && !schema_same(*q, *sch, &diff)) {
                    err_("kernel plugin '%s' declares op '%s' twice with different schemas (%s)",
                         p.name.c_str(), sch->op, diff.c_str());
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
            }
            staged_schemas.push_back(sch);
        }

        for (int i = 0, n = k_n(); i < n; ++i) {
            const RadKernelInfo* k = k_at(i);
            if (!k || !k->name || !k->op || !k->op[0]) {
                err_("kernel plugin '%s' declares a kernel with no name or op at index %d",
                     p.name.c_str(), i);
                unstage_(s);
                return RAD_E_INVAL;
            }
            /* A row with no entry point would resolve and then fail to run. */
            if (!k->launch) {
                err_("kernel '%s' in plugin '%s' has no launch function", k->name, p.name.c_str());
                unstage_(s);
                return RAD_E_INVAL;
            }
            if (k->domain != RAD_DOMAIN_DEVICE && k->domain != RAD_DOMAIN_HOST) {
                err_("kernel '%s' in plugin '%s' declares domain %d, which is neither device nor "
                     "host", k->name, p.name.c_str(), k->domain);
                unstage_(s);
                return RAD_E_INVAL;
            }
            /* Every tunable axis must carry a default that is one of its own values. Untuned
             * shapes (no cache, an unmeasured shape, another card's cache) run the defaults, with
             * no other fallback (spec §15). Checked at load rather than first use so a bad axis
             * fails at startup, not when some shape first misses the cache. */
            for (int v = 0; v < k->n_tunables; ++v) {
                const RadTunable& t = k->tunables[v];
                if (!t.key || !t.key[0] || !t.values || t.n_values <= 0) {
                    err_("kernel '%s' in plugin '%s' declares tunable %d with no key or no legal "
                         "values", k->name, p.name.c_str(), v);
                    unstage_(s);
                    return RAD_E_INVAL;
                }
                bool ok = false;
                for (int j = 0; j < t.n_values; ++j) if (t.values[j] == t.deflt) ok = true;
                if (!ok) {
                    err_("kernel '%s' in plugin '%s': tunable '%s' defaults to %lld, which is not "
                         "one of its declared values",
                         k->name, p.name.c_str(), t.key, (long long)t.deflt);
                    unstage_(s);
                    return RAD_E_INVAL;
                }
                for (int w = 0; w < v; ++w)
                    if (k->tunables[w].key && std::strcmp(k->tunables[w].key, t.key) == 0) {
                        err_("kernel '%s' in plugin '%s' declares tunable '%s' twice", k->name,
                             p.name.c_str(), t.key);
                        unstage_(s);
                        return RAD_E_INVAL;
                    }
            }
            staged_rows.emplace_back(k, i);
        }

        /* A kernel library with no code for this card has its device rows dropped: constraints
         * do not say which card a row was compiled for, so the rows would resolve and then fail
         * to launch, and an installation with libraries for several cards would serve only the
         * first. Coverage is read from the fat binaries its dlopen registered, which the launch
         * consults (core/device/aql_code.cpp), not from build_target. Host rows stay. On the host
         * backend no GPU code runs at all (set_no_device_code). */
        int n_device = 0;
        for (const auto& r : staged_rows) n_device += r.first->domain == RAD_DOMAIN_DEVICE;
        std::string why;
        if (no_device_code_)
            why = "no card is visible, so this process runs on the host backend and no GPU code "
                  "object runs here";
        if (n_device > 0 && s.code_to > s.code_from &&
            (no_device_code_ || rad::dev_code_runs(s.code_from, s.code_to, code_device_, &why) == 0)) {
            std::vector<std::pair<const RadKernelInfo*, int>> keep;
            for (const auto& r : staged_rows)
                if (r.first->domain != RAD_DOMAIN_DEVICE) keep.push_back(r);
            RAD_WARN("kernel plugin '%s' has no device code for device %d: %s. Its %d device "
                     "kernel(s) are left out of selection here%s.", p.name.c_str(), code_device_,
                     why.c_str(), n_device,
                     keep.empty() ? "" : fmt(" and its %zu host kernel(s) stay", keep.size()).c_str());
            staged_rows.swap(keep);
            p.device_code_absent = true;
        }
    } else if (p.kind == RAD_PLUGIN_QUANT) {
        auto q_n  = dl_fn<int (*)(void)>(s.handle, "rad_quant_count");
        auto q_at = dl_fn<const RadQuantizerInfo* (*)(int)>(s.handle, "rad_quant_at");
        if (!q_n || !q_at) {
            err_("quantiser plugin '%s' is missing rad_quant_count or rad_quant_at",
                 p.name.c_str());
            unstage_(s);
            return RAD_E_INVAL;
        }
        /* All checked before any is registered, as with kernel schemas. */
        for (int i = 0, n = q_n(); i < n; ++i) {
            const RadQuantizerInfo* q = q_at(i);
            if (!q || !q->name || !q->name[0] || !q->encoding || !q->quantize) {
                err_("quantiser plugin '%s' declares a quantiser with no name, no encoding hook or "
                     "no quantize hook at index %d", p.name.c_str(), i);
                unstage_(s);
                return RAD_E_INVAL;
            }
            /* Quantiser names are unique, so a recipe means the same thing whatever the load
             * order. */
            auto clash = [&](const char* other) { return std::strcmp(other, q->name) == 0; };
            for (const QuantRow& r : quants_)
                if (clash(r.info->name)) {
                    err_("quantiser '%s' is offered by both '%s' and '%s' -- refusing '%s'",
                         q->name, r.plugin.c_str(), p.name.c_str(), p.name.c_str());
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
            for (const RadQuantizerInfo* o : staged_quants)
                if (clash(o->name)) {
                    err_("quantiser plugin '%s' declares '%s' twice", p.name.c_str(), q->name);
                    unstage_(s);
                    return RAD_E_DUPLICATE;
                }
            staged_quants.push_back(q);
        }
    } else {
        err_("plugin '%s' declares kind %u, which is not a kernel, architecture or quantiser "
             "plugin", p.name.c_str(), p.kind);
        unstage_(s);
        return RAD_E_INVAL;
    }

    /* Called once after load, before any selection (rad_abi.h). A plugin that cannot open is
     * refused. */
    if (auto open = dl_fn<RadPluginOpenFn>(s.handle, "rad_plugin_open")) {
        int st = open();
        /* RAD_E_UNSUPPORTED: the plugin does not serve this machine (e.g. a library for another
         * card). A skip, not a failure. */
        if (st == RAD_E_UNSUPPORTED) {
            RAD_INFO("plugin '%s' declines this machine and is not loaded", p.name.c_str());
            unstage_(s);
            return 1;
        }
        if (st < 0) {
            err_("plugin '%s' refused to open: %s", p.name.c_str(), rad_strerror(st));
            unstage_(s);
            return st;
        }
    }

    /* What the plugin declares, including ops an earlier plugin already fixed; schema_owner()
     * answers which plugin fixed an op. */
    p.n_schemas = n_declared_schemas;
    p.n_kernels = (int)staged_rows.size();
    for (const RadOpSchema* sch : staged_schemas) schemas_[sch->op] = SchemaEntry{sch, p.name};
    for (const RadQuantizerInfo* q : staged_quants) {
        quants_.push_back(QuantRow{ q, p.name });
        /* Register its decoder so every weight reader can decode its schemes. */
        if (q->decode) enc_add_decoder(q->decode);
    }
    const auto concurrent = dl_fn<RadKernelConcurrentFn>(s.handle, "rad_kernel_concurrent");
    const auto collective = dl_fn<RadKernelCollectiveFn>(s.handle, "rad_kernel_collective");
    /* Per row when exported, else the library's own answer: a plugin that ships a device row and
     * its host oracle in one library must not have the oracle's status leak onto the device row. */
    const auto kern_ref = dl_fn<RadKernelReferenceFn>(s.handle, "rad_kernel_reference");
    for (const auto& sr : staged_rows) {
        KernelRow row;
        row.info         = sr.first;
        row.plugin       = p.name;
        row.plugin_order = order;
        row.index        = sr.second;
        row.concurrent   = concurrent && concurrent(sr.second) != 0;
        const int coll   = collective ? collective(sr.second) : 0;
        row.collective   = coll != 0;
        row.releases_link = coll == 2;
        row.reference    = kern_ref ? kern_ref(sr.second) != 0 : p.reference;
        rows_.push_back(row);
    }

    /* A library built for another card fails at every launch with an unhelpful driver error;
     * warn once here instead. Only a warning: build_target is author-written text, not derived
     * from the binary, so it cannot justify a refusal. "host" (CPU kernels) and "" (no claim) say
     * nothing about a device. Applies to libraries whose dlopen registered no fat binaries. */
    if (p.kind == RAD_PLUGIN_KERNEL && s.code_to == s.code_from && !p.build_target.empty() &&
        p.build_target != "host") {
        RadDeviceProps dp{};
        if (rad_dev_count() > code_device_ && rad_dev_props(code_device_, &dp) == RAD_OK &&
            dp.arch[0] && !target_covers(p.build_target, dp.arch))
            RAD_WARN("kernel plugin '%s' declares it was built for %s and this machine reports %s. "
                     "Its kernels will fail to launch; build it for %s, or leave it out with "
                     "--kernels.",
                     p.name.c_str(), p.build_target.c_str(), dp.arch, dp.arch);
    }

    RAD_DEBUG("plugin %d: %s %s (%s) -- %d kernels, %d schemas, %zu fat binaries, %s", order,
              p.name.c_str(), p.version.c_str(), p.path.c_str(), p.n_kernels, p.n_schemas,
              s.code_to - s.code_from, p.build_target.empty() ? "portable" : p.build_target.c_str());
    plugins_.push_back(std::move(p));
    index_rows_();
    s.handle = nullptr;   /* the Plugin owns it now */
    return RAD_OK;
}

/* The per-op index select() walks, in (hierarchy order, declaration order). Rebuilt whole on
 * every registration; there are only hundreds of rows, at startup. */
void Registry::index_rows_() {
    by_op_.clear();
    for (const KernelRow& r : rows_) by_op_[r.info->op].push_back(&r);
    for (auto& kv : by_op_) {
        std::stable_sort(kv.second.begin(), kv.second.end(),
                         [](const KernelRow* a, const KernelRow* b) {
                             if (a->plugin_order != b->plugin_order)
                                 return a->plugin_order < b->plugin_order;
                             return a->index < b->index;
                         });
    }
}

/* ------------------------------------------------------------------ the public loading API */
int Registry::load_plugin(const std::string& path, int order) {
    Staged s;
    int st = stage_(path, &s);
    if (st < 0) return st;
    return commit_(s, order);
}

int Registry::load_dir(const std::string& dir, const std::vector<std::string>& hierarchy) {
    return load_dirs({ dir }, hierarchy);
}

int Registry::load_dirs(const std::vector<std::string>& dirs,
                        const std::vector<std::string>& hierarchy) {
    int first_error = RAD_OK;

    /* Directories in the given order, files in filename order, so an unconfigured set loads
     * reproducibly. A file name an earlier directory already supplied is shadowed: the operator's
     * own plugins come ahead of an installation's, and loading both would be a duplicate name. */
    std::vector<std::string> files, stems;
    std::string where;
    for (const std::string& dir : dirs) {
        DIR* d = opendir(dir.c_str());
        if (!d) {
            err_("plugin directory '%s': %s", dir.c_str(), std::strerror(errno));
            return RAD_E_IO;
        }
        where += (where.empty() ? "" : ", ") + dir;
        std::vector<std::string> here;
        while (const dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() < 4 || n.compare(n.size() - 3, 3, ".so") != 0) continue;
            const std::string path = dir + "/" + n;
            struct stat st {};
            if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
            here.push_back(path);
        }
        closedir(d);
        std::sort(here.begin(), here.end());
        for (const std::string& f : here) {
            const std::string stem = stem_of(f);
            auto at = std::find(stems.begin(), stems.end(), stem);
            if (at != stems.end()) {
                RAD_INFO("plugin %s is shadowed by %s, which comes first on the search path",
                         f.c_str(), files[(size_t)(at - stems.begin())].c_str());
                continue;
            }
            files.push_back(f);
            stems.push_back(stem);
        }
    }

    std::vector<Staged> staged;
    for (const std::string& f : files) {
        Staged s;
        int st = stage_(f, &s);
        if (st < 0) {
            if (first_error == RAD_OK) first_error = st;
            continue;
        }
        staged.push_back(s);
    }

    /* An entry matches a library's declared name or its file: `libr4d`, `r4d`, `libr4d.so` and a
     * path to it are one library. */
    auto names = [](const std::string& h, const Staged& s) {
        return h == s.name || h == s.file || stem_of(h) == s.file;
    };
    std::string found;   /* every kernel library on the search path, for a list that names none */
    for (const Staged& s : staged)
        if (s.info && s.info->kind == RAD_PLUGIN_KERNEL)
            found += (found.empty() ? "" : ", ") + s.name;

    /* A non-empty hierarchy is exclusive. Ordering alone cannot keep a library out: selection
     * runs per domain, so an op the named library has no device row for would fall to an unnamed
     * one. Unnamed kernel libraries are closed before registering anything. Architecture and
     * quantiser plugins are never ranked. */
    if (!hierarchy.empty()) {
        std::vector<Staged> kept;
        std::string left_out;
        for (Staged& s : staged) {
            const bool kernel = s.info && s.info->kind == RAD_PLUGIN_KERNEL;
            const bool named = std::find_if(hierarchy.begin(), hierarchy.end(),
                                            [&](const std::string& h) {
                                                return names(h, s);
                                            }) != hierarchy.end();
            if (!kernel || named) {
                kept.push_back(s);
                continue;
            }
            left_out += (left_out.empty() ? "" : ", ") + s.name;
            unstage_(s);
        }
        staged.swap(kept);
        if (!left_out.empty())
            RAD_INFO("kernel libraries on the search path that --kernels (or RADIANCE_KERNELS) "
                     "does not name, and so not loaded: %s", left_out.c_str());
    }

    /* A named library takes its position in the list; unnamed ones follow. A reference library
     * (RadPluginReferenceFn) is pinned below every other unnamed plugin, because it matches every
     * op it declares and would make anything behind it unreachable. Naming it places it where
     * named, e.g. ahead of a library to compare against. */
    const size_t kUnnamed = (size_t)INT_MAX - 1;
    std::vector<size_t> rank(staged.size());
    for (size_t i = 0; i < staged.size(); ++i)
        rank[i] = staged[i].reference ? (size_t)INT_MAX : kUnnamed;
    for (size_t h = 0; h < hierarchy.size(); ++h) {
        bool hit = false;
        for (size_t i = 0; i < staged.size(); ++i) {
            if (names(hierarchy[h], staged[i])) {
                rank[i] = h;
                hit = true;
            }
        }
        /* The list is what loads: an entry that matches nothing loads nothing in its place. */
        if (!hit)
            RAD_WARN("--kernels (or RADIANCE_KERNELS) names '%s', which is no kernel library in "
                     "%s, so nothing loads in its place; the kernel libraries there are: %s",
                     hierarchy[h].c_str(), where.c_str(), found.empty() ? "none" : found.c_str());
    }
    std::vector<size_t> idx(staged.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](size_t a, size_t b) { return rank[a] < rank[b]; });

    /* Continue from what is already loaded, so loading kernels/ then architectures/ does not put
     * two plugins at position 0. */
    int order = 0;
    for (const Plugin& p : plugins_) order = std::max(order, p.order + 1);

    for (size_t i : idx) {
        int st = commit_(staged[i], order);
        if (st < 0) {
            if (first_error == RAD_OK) first_error = st;
            continue;   /* refused, and the next plugin still gets its chance */
        }
        if (st > 0) continue;   /* declined this machine */
        ++order;
    }

    int st = check_complete();
    if (st < 0 && first_error == RAD_OK) first_error = st;
    return first_error;
}

/* ------------------------------------------------------------------ the tuning cache (§15) */
int Registry::load_tune(const std::string& radiance_home, const std::string& machine) {
    const std::string key = machine.empty() ? rad_machine_key() : machine;
    const int s = tune_.load(radiance_home, key);
    if (s < 0) return s;
    /* `load` refuses a file whose machine key (line 2, tunecache.h) is another machine's, so
     * these rows are this machine's. */
    return (int)tune_.rows().size();
}

/* A row whose op has no schema in the loaded set cannot have its parameters checked, so a typo
 * would become a silent miss. Checked once the set is complete, because an override plugin may
 * rely on the schema of a plugin below it, which loads later. */
int Registry::check_complete() {
    int st = RAD_OK;
    for (const KernelRow& r : rows_) {
        if (schemas_.find(r.info->op) == schemas_.end()) {
            err_("kernel '%s' in plugin '%s' implements op '%s', which no loaded plugin declares a "
                 "schema for", r.info->name, r.plugin.c_str(), r.info->op);
            st = RAD_E_NOSCHEMA;
        }
    }
    return st;
}

/* ------------------------------------------------------------------ lookup and teardown */
const Plugin* Registry::plugin(std::string_view name) const {
    for (const Plugin& p : plugins_)
        if (p.name == name || p.file == name) return &p;
    return nullptr;
}

const Registry::QuantRow* Registry::quantizer(std::string_view name) const {
    for (const QuantRow& q : quants_)
        if (name == q.info->name) return &q;
    return nullptr;
}

const std::vector<const KernelRow*>* Registry::rows_for(std::string_view op) const {
    auto it = by_op_.find(std::string(op));
    return it == by_op_.end() ? nullptr : &it->second;
}

void Registry::close() {
    fini_instances();
    for (const QuantRow& q : quants_)
        if (q.info && q.info->decode) enc_remove_decoder(q.info->decode);
    /* Reverse load order: a later plugin may depend on state an earlier one established. */
    for (auto it = plugins_.rbegin(); it != plugins_.rend(); ++it) {
        if (it->close) it->close();
        if (it->handle) dlclose(it->handle);
    }
    plugins_.clear();
    rows_.clear();
    quants_.clear();
    by_op_.clear();
    schemas_.clear();
    load_errors_.clear();
    clear_selections();
}

Registry::~Registry() { close(); }

}  /* namespace rad */
