/* startup.cpp -- the tools' seam onto core/plugin and core/build, and the shared CLI plumbing.
 *
 * rad-convert, rad-check and rad-tune are the engine's startup with a different last step, so they
 * do not get their own loader: they call the same Registry and the same Builder the engine will,
 * through the two functions below. A tool that resolved kernels differently from the engine would
 * be a tool whose answers are about a different program.
 */
#include "build/startup.h"
#include "build/rad_build.h"
#include "format/tunecache.h"
#include "plugin/registry.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>

namespace rad {

/* One Registry for the life of the process. A Resolved, a Band and a KernelRow all point into a
 * plugin's static tables, and the Program outlives the call that built it, so the registry cannot
 * be a local. registry.h says the same thing about the engine. */
Registry& rad_tools_registry() {
    static Registry r;
    return r;
}

/* ------------------------------------------------------------------ the reference-kernel gate */

/* An op counts as "on the reference" when NO domain and NO band offers a kernel from a library that
 * is not a reference implementation (RadPluginReferenceFn).
 *
 * The narrower reading matters. A reference library is usually a HOST plugin, so every op's host
 * band resolves to it -- that is what makes the host domain work at all, and flagging it would
 * flag the entire vocabulary of every model. What is a hazard is an op a reference is the ONLY
 * implementation of: wherever the planner puts it, it runs naive generic code.
 *
 * The op is still counted once per declared instance, because "rmsnorm resolves to ref" reads very
 * differently at one layer and at sixty-four. */
int rad_reference_ops(const Program& p, std::vector<std::string>* out) {
    int n = 0;
    for (size_t i = 1; i < p.ops.size(); ++i) {
        const OpInfo& oi = p.ops[i];
        bool real = false;
        for (const Band& b : oi.bands)
            for (int d = 0; d < RAD_N_DOMAINS && !real; ++d)
                if (b.dom[d] && b.dom[d].row && !b.dom[d].row->reference)
                    real = true;
        if (real) continue;
        ++n;
        /* By op NAME and deduplicated: a 64-layer model declares sixty-four identical rmsnorms and
         * the useful sentence is "rmsnorm has no kernel", not sixty-four copies of it. */
        if (out && std::find(out->begin(), out->end(), oi.op) == out->end())
            out->push_back(oi.op);
    }
    return n;
}

int rad_gate_reference_kernels(const Program& p, bool allow, const char* what) {
    std::vector<std::string> ops;
    const int n = rad_reference_ops(p, &ops);
    if (n == 0) return RAD_OK;

    std::string list;
    for (size_t i = 0; i < ops.size(); ++i) { if (i) list += ", "; list += ops[i]; }
    /* Which reference libraries those are, by the name each one gave itself. */
    std::vector<std::string> refs;
    for (size_t i = 1; i < p.ops.size(); ++i)
        for (const Band& b : p.ops[i].bands)
            for (int d = 0; d < RAD_N_DOMAINS; ++d)
                if (b.dom[d] && b.dom[d].row && b.dom[d].row->reference &&
                    std::find(refs.begin(), refs.end(), b.dom[d].row->plugin) == refs.end())
                    refs.push_back(b.dom[d].row->plugin);
    std::string libs;
    for (size_t i = 0; i < refs.size(); ++i) { if (i) libs += ", "; libs += refs[i]; }

    if (allow) {
        RAD_WARN("%d declared op(s) resolve to a reference implementation (%s) and "
                 "--debug-accept-reference-kernels was passed, so %s continues. This is not a "
                 "production configuration: %zu distinct op(s) run the naive generic "
                 "implementation -- %s", n, libs.c_str(), what, ops.size(), list.c_str());
        return RAD_OK;
    }

    RAD_ERR("%d declared op(s) resolve only to a reference implementation (%s).\n"
            "    %zu distinct op(s): %s\n"
            "  A reference library exists to be an ORACLE and a last resort, not a deployment. "
            "Falling back to it silently is a model running orders of magnitude below the machine "
            "with nothing in the output to say so, which is the failure spec §17 exists to prevent "
            "-- so %s refuses rather than serving it.\n"
            "  Either supply a kernel plugin that covers these ops, or pass "
            "--debug-accept-reference-kernels to accept the cost deliberately.",
            n, libs.c_str(), ops.size(), list.c_str(), what);
    return RAD_E_NOKERNEL;
}

std::vector<std::string> rad_home_dirs(const std::string& home) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i <= home.size()) {
        size_t e = home.find(':', i);
        if (e == std::string::npos) e = home.size();
        if (e > i) out.push_back(home.substr(i, e - i));
        i = e + 1;
    }
    if (out.empty()) out.push_back(".");
    return out;
}

std::string rad_home_file(const std::string& home, const std::string& rel) {
    const std::vector<std::string> dirs = rad_home_dirs(home);
    for (const std::string& d : dirs) {
        struct stat st {};
        if (stat((d + "/" + rel).c_str(), &st) == 0) return d + "/" + rel;
    }
    return dirs.front() + "/" + rel;
}

/* The homes on the path that have the directory `sub`. */
static std::vector<std::string> home_subdirs(const std::string& home, const char* sub) {
    std::vector<std::string> out;
    for (const std::string& d : rad_home_dirs(home)) {
        struct stat st {};
        const std::string p = d + "/" + sub;
        if (stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) out.push_back(p);
    }
    return out;
}

int rad_tools_load_plugins(const std::string& home, const std::vector<std::string>& hierarchy,
                           const std::string& arch_id, const std::string& quant,
                           LoadedPlugins* out) {
    if (!out) return RAD_E_INVAL;
    *out = LoadedPlugins{};

    Registry& r = rad_tools_registry();

    /* Kernels first and in hierarchy order, then architectures. The order matters for kernels and
     * not for architectures: no two architecture plugins may claim the same id, so there is
     * nothing for a hierarchy to break a tie between.
     *
     * A home on the path need not have every directory -- one holding a single kernel library is
     * the ordinary shape of a home of one's own -- but kernels/ and architectures/ must each be
     * somewhere on it, or the scan names the first home's as missing. */
    const std::string first = rad_home_dirs(home).front();
    std::vector<std::string> kdirs = home_subdirs(home, "kernels");
    if (kdirs.empty()) kdirs.push_back(first + "/kernels");
    int s = r.load_dirs(kdirs, hierarchy);
    if (s < 0) return s;

    std::vector<std::string> adirs = home_subdirs(home, "architectures");
    if (adirs.empty()) adirs.push_back(first + "/architectures");
    s = r.load_dirs(adirs, {});
    if (s < 0) return s;

    /* The quantisers, which rad-convert drives and the oracle decodes through for a scheme the
     * core does not define. A home without the directory has none, which is not an error for an
     * engine serving a container whose encodings the core decodes itself. */
    const std::vector<std::string> qdirs = home_subdirs(home, "quantizers");
    if (!qdirs.empty()) {
        s = r.load_dirs(qdirs, {});
        if (s < 0) return s;
    }

    RAD_TRY(r.check_complete());

    /* The tuning cache, once the kernel set is known. A missing file is not an error and leaves
     * every shape on its plugin's first variant; a present one moves the shapes rad-tune measured.
     * Reported when it has rows, because "which variant am I running" is otherwise a question with
     * no visible answer -- --debug-graph prints the per-shape source beside each kernel. */
    const std::string tune = rad_home_file(home, TuneCache::rel_path(rad_machine_key()));
    const std::string tune_home = tune.substr(0, tune.size() - TuneCache::rel_path(rad_machine_key()).size() - 1);
    const int n_tuned = r.load_tune(tune_home);
    if (n_tuned > 0) RAD_INFO("tune cache: %d row(s) from %s", n_tuned, tune.c_str());

    /* The selection key is the PAIR (spec §2.4). An exact match on both halves is preferred;
     * if no quant-specific plugin exists, an unquantised/universal plugin (empty arch_quant)
     * serves as fallback. */
    const Plugin* fallback = nullptr;
    for (const Plugin& p : r.plugins()) {
        if (p.kind == RAD_PLUGIN_KERNEL) {
            out->kernels.push_back({ p.name, p.version });
        } else if (p.kind == RAD_PLUGIN_ARCH && !arch_id.empty() && p.arch_id == arch_id) {
            if (p.arch_quant == quant) {
                out->arch_plugin  = p.name;
                out->arch_version = p.version;
                out->arch_id      = p.arch_id;
                out->arch_quant   = p.arch_quant;
            } else if (p.arch_quant.empty()) {
                fallback = &p;
            }
        }
    }
    if (out->arch_plugin.empty() && fallback) {
        out->arch_plugin  = fallback->name;
        out->arch_version = fallback->version;
        out->arch_id      = fallback->arch_id;
        out->arch_quant   = fallback->arch_quant;
    }

    if (out->kernels.empty()) {
        RAD_ERR("no kernel plugin loaded from %s/kernels. Every weight's layout comes from the "
                "kernel that will read it, so there is nothing to lay a container out for.",
                home.c_str());
        return RAD_E_NOKERNEL;
    }
    if (!arch_id.empty() && out->arch_plugin.empty()) {
        /* No match, no model (spec §1). Name what IS installed AND at which quantisation, because
         * the two causes look identical otherwise: an architecture nobody has written a plugin for,
         * and one that is written but only for another weight format. The second is the common one
         * for a family with more than one member. */
        std::string have, same;
        for (const Plugin& p : r.plugins()) {
            if (p.kind != RAD_PLUGIN_ARCH) continue;
            const std::string q = p.arch_quant.empty() ? "(unquantised)" : p.arch_quant;
            if (!have.empty()) have += ", ";
            have += p.arch_id + " @ " + q;
            if (p.arch_id == arch_id) { if (!same.empty()) same += ", "; same += q; }
        }
        if (!same.empty())
            RAD_ERR("no architecture plugin claims '%s' at quantisation '%s'. '%s' is served at: "
                    "%s. Either the container's quantisation descriptor is not one this project "
                    "has a plugin for, or the container was written before the descriptor was "
                    "recorded -- rad-info prints it.",
                    arch_id.c_str(), quant.empty() ? "(unquantised)" : quant.c_str(),
                    arch_id.c_str(), same.c_str());
        else
            RAD_ERR("no architecture plugin claims '%s'. %s/architectures offers: %s",
                    arch_id.c_str(), home.c_str(), have.empty() ? "(none)" : have.c_str());
        return RAD_E_NOTFOUND;
    }
    return RAD_OK;
}

/* WHAT THE ARCHITECTURE PLUGIN SAYS BEFORE IT DECLARES. Optional export: a plugin without one
 * leaves every field zero, which the caller reads as "no opinion" and answers from its own
 * defaults.
 *
 * Not an error path. A probe that fails is a plugin declining to answer, not a model that cannot
 * be served -- the caller falls back to its own default and the plugin gets to refuse properly in
 * declare, where it has the whole configuration in front of it and can say why. */
int rad_tools_probe(const LoadedPlugins& lp, const RadModelMeta& meta, RadArchProbe* out) {
    if (!out) return RAD_E_INVAL;
    *out = RadArchProbe{};
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    if (!ap || !ap->arch_probe) return RAD_E_UNSUPPORTED;
    const int s = ap->arch_probe(&meta, out);
    if (s < 0) {
        RAD_WARN("%s: rad_arch_probe returned %s; the core will use its own defaults",
                 lp.arch_plugin.c_str(), rad_strerror(s));
        *out = RadArchProbe{};
        return s;
    }
    return RAD_OK;
}

const RadChatFormat* rad_tools_chat_format(const LoadedPlugins& lp, const RadModelMeta& meta) {
    const Plugin* ap = rad_tools_registry().plugin(lp.arch_plugin);
    if (!ap || !ap->arch_chat_format) return nullptr;
    return ap->arch_chat_format(&meta);
}

WeightSourceFn rad_container_sources(const RadFile& f) {
    return [&f](const std::string& name, const Program&, WeightSource* out) {
        const RadFileEntry* e = f.find(name);
        if (!e) return RAD_E_NOTFOUND;
        out->enc = f.encoding(*e);
        out->rank = e->rank;
        for (int i = 0; i < RAD_MAX_RANK; ++i) out->shape[i] = e->shape[i];
        return RAD_OK;
    };
}

WeightSourceFn CheckpointSources::fn() {
    return [this](const std::string& name, const Program& prog, WeightSource* out) {
        return answer(name, prog, out);
    };
}

int CheckpointSources::answer(const std::string& name, const Program& prog, WeightSource* out) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = got_.find(name);
    if (it == got_.end()) {
        CkptWeight w;
        std::string why;
        const int st = ckpt_resolve(ck_, prog, name, false, &w, &why);
        /* Not found is not remembered as an answer -- declare may map the name later -- but
         * its reason is, for the loader to give if it never is. */
        if (st == RAD_E_NOTFOUND) {
            missing_[name] = why;
            return st;
        }
        missing_.erase(name);
        if (st < 0) {
            refused_[name] = why;
            return st;
        }
        it = got_.emplace(name, std::move(w)).first;
    }
    const CkptWeight& w = it->second;
    if (!w.trivial) {
        refused_[name] = w.why;
        return RAD_E_UNSUPPORTED;
    }
    out->enc = w.enc;
    out->rank = w.rank;
    for (int i = 0; i < RAD_MAX_RANK; ++i) out->shape[i] = w.shape[i];
    return RAD_OK;
}

const CkptWeight* CheckpointSources::find(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = got_.find(name);
    return it == got_.end() || !it->second.trivial ? nullptr : &it->second;
}

std::string CheckpointSources::missing(const std::string& name) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = missing_.find(name);
    return it == missing_.end() ? std::string() : it->second;
}

std::vector<std::string> CheckpointSources::refused() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    for (const auto& [n, why] : refused_) out.push_back(n + ": " + why);
    return out;
}

int rad_tools_declare(const LoadedPlugins& lp, const RadModelMeta& meta, const RadBuildCtx& ctx,
                      Program* out, const std::function<int(RadBuilder*, int64_t)>& also,
                      const Program* ref, const WeightSourceFn& sources, bool instances) {
    if (!out) return RAD_E_INVAL;

    Registry& r = rad_tools_registry();
    const Plugin* ap = r.plugin(lp.arch_plugin);
    if (!ap || !ap->arch_declare) {
        RAD_ERR("architecture plugin '%s' is not loaded or exports no rad_arch_declare",
                lp.arch_plugin.c_str());
        return RAD_E_NOTFOUND;
    }

    /* The Builder takes the Registry directly: there is no selector adapter between the two. */
    Builder b(r, meta, ctx);
    if (ref) b.set_reference(ref);
    if (sources) b.set_sources(sources);
    b.set_instances(instances);

    int s = ap->arch_declare(&b, &meta, &ctx);
    if (s < 0) {
        RAD_ERR("%s: rad_arch_declare returned %s", lp.arch_plugin.c_str(), rad_strerror(s));
        for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
        return s;
    }

    /* The sampler declares here, into the same builder and before finish(), so its ops go through
     * the same resolution as the plugin's. A sampler op that does not resolve is deliberately NOT
     * a declare failure -- a deployment that never sends `typical_p` does not need a typical
     * kernel -- so this returning negative means something structural, not a miss. */
    if (also) {
        /* THE WIDTH THE PLUGIN ACTUALLY DECLARED, handed over rather than re-derived. The sampler
         * needs to know how much vocabulary THIS RANK produces, and the one place that is a fact
         * is the logits buffer the plugin just named. Computing it a second time from n_vocab and
         * world_size puts the sharding rule in two files, and the two disagree the moment a plugin
         * stops splitting the vocabulary -- which arch/common/rad_arch.h currently does, on
         * purpose. A sampler reading 248320 logits out of a 124160-wide row is not a crash; it is
         * fluent text with the wrong tokens. */
        int64_t lw = 0;
        if (b.program().logits_buf != RAD_NULL_HANDLE &&
            (size_t)b.program().logits_buf < b.program().buffers.size()) {
            const RadBufDecl& d = b.program().buffers[b.program().logits_buf].decl;
            if (d.rank > 0) lw = d.shape[d.rank - 1];
        }
        s = also(&b, lw);
        if (s < 0) {
            for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
            return s;
        }
    }

    s = b.finish();
    /* Declare ALWAYS completes: a structural error is reported whole, not first-failure-at-the-top
     * (spec §3.1). */
    for (const auto& e : b.errors()) RAD_ERR("    %s", e.c_str());
    if (s < 0) return s;
    if (b.status() < 0) return b.status();

    *out = std::move(b.program());
    return RAD_OK;
}

}  /* namespace rad */
