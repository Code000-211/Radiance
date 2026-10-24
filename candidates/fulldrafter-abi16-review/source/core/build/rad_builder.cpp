/* rad_builder.cpp -- the Builder, and the C surface of rad_builder.h that drives it.
 *
 * Handles are 1-based indices into the Program's tables, so index 0 is a sentinel and a failed
 * decl_* is falsy. That is not a convention for its own sake: it is what lets an architecture
 * plugin write
 *
 *     rad_op h = decl_op(b, "rmsnorm_had_quant_i8", ...);
 *     if (!h) { ... emit the unfused sequence ... }
 *
 * and have fusion be a selection rather than a compiler pass (spec §2.3).
 */
#include "rad_build.h"
#include "../../arch/common/rad_tree_candidate.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <set>

namespace rad {

/* ================================================================== interning */
const char* intern(std::string_view s) {
    /* A node-based set: insertion never moves an existing element, so the pointer we hand out
     * stays valid for the life of the process. Declare runs once per rank on its own thread
     * (spec §1), hence the lock -- taken a few thousand times at startup and never again. */
    static std::unordered_set<std::string> pool;
    static std::mutex m;
    std::lock_guard<std::mutex> lk(m);
    return pool.emplace(s).first->c_str();
}

/* ================================================================== small helpers */
/* Names for the enums the graph dump prints. `bld_` prefixed because tier_name and site_name in
 * rad_core.h belong to the planner and these are the same kind of thing for a different axis --
 * two components should not race for one symbol. */
const char* bld_domain_name(int d) { return d == RAD_DOMAIN_HOST ? "host" : "device"; }

/* The full-detail rendering of one miss, used by --debug-graph and by the device-hole report.
 * Defined here rather than in the header so that the domain has exactly one spelling. */
std::string Program::Miss::line() const {
    return fmt("%-22s %s  [%s] -> %s", op.c_str(), geom.c_str(), bld_domain_name(domain),
               why.c_str());
}

const char* bld_access_name(int a) {
    switch (a) {
        case RAD_ACCESS_PER_TOKEN:   return "per-token";
        case RAD_ACCESS_PER_REQUEST: return "per-request";
        case RAD_ACCESS_CONDITIONAL: return "conditional";
        case RAD_ACCESS_RARE:        return "rare";
        case RAD_ACCESS_VOCAB:       return "vocab";
        default:                     return "?";
    }
}

const char* bld_buf_kind_name(int k) {
    switch (k) {
        case RAD_BUF_TRANSIENT: return "transient";
        case RAD_BUF_PERSIST:   return "persist";
        case RAD_BUF_DERIVED:   return "derived";
        default:                return "?";
    }
}

const char* bld_kv_kind_name(int k) {
    switch (k) {
        case RAD_KV_FULL:   return "full";
        case RAD_KV_WINDOW: return "window";
        case RAD_KV_LINEAR: return "linear";
        case RAD_KV_CONV:   return "conv";
        default:            return "?";
    }
}

static int64_t decl_numel(uint32_t rank, const int64_t* shape) {
    if (rank == 0 || rank > RAD_MAX_RANK) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        if (shape[i] <= 0) return 0;
        n *= shape[i];
    }
    return n;
}

/* ================================================================== construction */
Builder::Builder(Registry& reg, const RadModelMeta& meta, const RadBuildCtx& ctx)
    : reg_(reg) {
    prog_.meta = meta;
    prog_.ctx  = ctx;
    scope_ = ctx.scope ? ctx.scope : "";

    /* Sentinels. Handle 0 is never valid, so a zeroed struct is an unset handle. */
    prog_.weights.emplace_back();
    prog_.buffers.emplace_back();
    prog_.ops.emplace_back();
    prog_.kv_groups.emplace_back();
    op_bufs_.emplace_back();
    kv_decl_at_.push_back(-1);
    kv_consumer_.push_back(-1);
}

uint32_t Builder::err(int code, const char* f, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(msg, sizeof msg, f, ap);
    va_end(ap);
    errors_.emplace_back(msg);
    /* First class wins: a later error is usually a consequence of the first, and the list carries
     * all of them anyway. */
    if (status_ == RAD_OK) status_ = code;
    RAD_ERR("declare: %s", msg);
    return RAD_NULL_HANDLE;
}

/* The scope prefixes declared names so a drafter, a vision tower and the language model can share
 * one builder without colliding (spec §10, §11). Op names are NOT scoped: they are the shared
 * vocabulary the selector matches on. */
std::string Builder::scoped(const char* name) const {
    if (scope_.empty()) return name ? name : "";
    return scope_ + "." + (name ? name : "");
}

/* ================================================================== weights */
rad_weight Builder::decl_weight(const char* name, const RadWeightDecl* d) {
    if (finished_) return err(RAD_E_STATE, "decl_weight('%s') after declare finished", name ? name : "");
    if (!name || !*name || !d) return err(RAD_E_INVAL, "decl_weight: null name or declaration");

    std::string n = scoped(name);
    if (weight_by_name_.count(n))
        return err(RAD_E_DUPLICATE, "weight '%s' declared twice; the container has one entry "
                                    "for that name and two declarations cannot both own it",
                   n.c_str());

    int64_t numel = decl_numel(d->rank, d->shape);
    if (numel == 0)
        return err(RAD_E_SHAPE, "weight '%s': rank %u with a zero or negative extent",
                   n.c_str(), d->rank);

    WeightInfo w;
    w.name = n;
    w.decl = *d;
    /* Under tensor parallel the plugin declares dimensions ALREADY DIVIDED (spec §9), so this is
     * what THIS RANK holds. The core does not reason about sharding propagation and does not
     * divide anything: a sharding inference pass is a compiler pass, and its failure mode is a
     * model that produces fluent wrong text on a rank count nobody tested. */
    w.logical_bytes = rad_dtype_bytes(d->dtype, numel);
    if (w.logical_bytes == 0 && d->dtype < RAD_DT_PLUGIN_BASE)
        return err(RAD_E_DTYPE, "weight '%s': dtype %u is not one the core can size", n.c_str(),
                   d->dtype);

    /* WHAT IT IS A VIEW OF (spec §4.3): the source's encoding as the model holds it, and the planes
     * this declaration takes. With nothing behind the builder the weight is its own plain plane
     * in the declared dtype, which is what a test or a synthetic tool declares against. */
    w.source = (d->source && *d->source) ? scoped(d->source) : n;
    WeightSource ws;
    w.enc_known = lookup_source(w.source, &ws) == RAD_OK;
    w.enc = w.enc_known ? ws.enc : rad_enc_plain(d->dtype);
    int64_t vr = 0, vc = 0;            /* the declared extents as a [rows, cols] view */
    rad_enc_view(d->rank, d->shape, &vr, &vc);
    if (!w.enc_known) {
        w.n_sel = 1;
        w.sel[0] = 0;
    } else if (!d->planes || !*d->planes) {
        w.n_sel = w.enc.n_planes;
        for (int k = 0; k < w.n_sel; ++k) w.sel[k] = k;
    } else {
        std::string list(d->planes);
        size_t at = 0;
        while (at <= list.size()) {
            size_t comma = list.find(',', at);
            if (comma == std::string::npos) comma = list.size();
            std::string role = list.substr(at, comma - at);
            while (!role.empty() && role.front() == ' ') role.erase(role.begin());
            while (!role.empty() && role.back() == ' ') role.pop_back();
            const int k = rad_enc_find(&w.enc, role.c_str());
            char e[256];
            rad_enc_format(&w.enc, e, sizeof e);
            if (k < 0)
                return err(RAD_E_SHAPE, "weight '%s' takes plane '%s' of '%s', which is %s and has "
                                        "no such plane", n.c_str(), role.c_str(), w.source.c_str(), e);
            if (w.n_sel == RAD_ENC_MAX_PLANES)
                return err(RAD_E_INVAL, "weight '%s': more than %d planes", n.c_str(),
                           RAD_ENC_MAX_PLANES);
            w.sel[w.n_sel++] = k;
            at = comma + 1;
        }
    }

    /* THE DECLARATION AGAINST THE PLANES. One plane is declared at its own dtype and extents --
     * this rank's share of them under a shard -- and several at the codes' dtype and the logical
     * extents. A dtype that is not the plane's is an exact widening the loader performs, or it is
     * a quantisation, which belongs to rad-convert and is refused here by name. */
    const RadEncPlane& p0 = w.enc.plane[w.sel[0]];
    if (p0.dtype != d->dtype) {
        const bool widen = w.n_sel == 1 && d->dtype == RAD_F32 &&
                           (p0.dtype == RAD_BF16 || p0.dtype == RAD_F16);
        char e[256];
        rad_enc_format(&w.enc, e, sizeof e);
        if (!widen)
            return err(RAD_E_DTYPE, "weight '%s' is declared %s, and '%s' holds %s as %s: only an "
                                    "exact widening is done at load -- anything else is a "
                                    "quantisation, which a recipe does before the container is "
                                    "written (spec §4.4)", n.c_str(), rad_dtype_name(d->dtype),
                       w.source.c_str(), p0.role, e);
        w.widen = true;
    }
    int64_t sr = 0, sc = 0;            /* the source's logical [rows, cols] */
    if (w.enc_known) rad_enc_view(ws.rank, ws.shape, &sr, &sc);
    for (int k = 0; k < w.n_sel; ++k) {
        const RadEncPlane& pk = w.enc.plane[w.sel[k]];
        int64_t fr = vr, fc = vc;      /* the plane's whole extents */
        if (w.n_sel == 1) {
            w.sel_rows[k] = vr;
            w.sel_cols[k] = vc;
            if (w.enc_known) rad_enc_plane_dims(&pk, sr, sc, &fr, &fc);
        } else {
            rad_enc_plane_dims(&pk, vr, vc, &w.sel_rows[k], &w.sel_cols[k]);
            fr = sr; fc = sc;
        }
        /* Unsharded, the declaration IS the plane (or, for several, the logical weight), and a
         * mismatch is the plugin and the model disagreeing about a shape -- named now rather than
         * found at load as a byte count. A shard's share is the loader's to check. */
        const bool whole = d->shard == RAD_SHARD_NONE || prog_.ctx.world_size <= 1;
        /* A RESHAPE IS NOT A DISAGREEMENT: one whole plane of single elements, a byte or more
         * each, declared under another shape of the same count -- a conv kernel [O, C, T, H, W]
         * read as the [O, C*T*H*W] matrix it is. Row-major, its bytes do not move. */
        const bool reshape = w.n_sel == 1 && whole && pk.kind == RAD_PLANE_TILED &&
                             pk.block[0] == 1 && pk.block[1] == 1 &&
                             rad_dtype_bits(pk.dtype) >= 8 && fr * fc == vr * vc;
        if (w.enc_known && whole && !reshape && (k == 0 || w.n_sel == 1) &&
            (w.n_sel == 1 ? (fr != vr || fc != vc) : (sr != vr || sc != vc)))
            return err(RAD_E_SHAPE, "weight '%s' is declared [%lld, %lld], and %s of '%s' is "
                                    "[%lld, %lld]", n.c_str(), (long long)vr, (long long)vc,
                       w.n_sel == 1 ? pk.role : "the logical extent", w.source.c_str(),
                       (long long)(w.n_sel == 1 ? fr : sr), (long long)(w.n_sel == 1 ? fc : sc));
    }

    /* The stored form defaults to the selected plane as it is. finish_layouts() overwrites it when
     * the resolved kernel asks for something else through its layout hook (spec §4.3); a kernel
     * with no hook, or one that answers RAD_E_UNSUPPORTED, takes exactly this. */
    w.identity     = true;
    w.stored_dtype = w.widen ? d->dtype : p0.dtype;
    w.stored_bytes = w.widen ? rad_dtype_bytes(d->dtype, vr * vc)
                             : vr * rad_enc_row_bytes(p0.dtype, vc);
    w.stored_rank  = d->rank;
    for (uint32_t i = 0; i < RAD_MAX_RANK; ++i)
        w.stored_shape[i] = i < d->rank ? d->shape[i] : 0;
    w.align        = RAD_ALIGN_UNIT;

    prog_.weights.push_back(w);
    const rad_weight h = (rad_weight)(prog_.weights.size() - 1);
    weight_by_name_.emplace(prog_.weights.back().name, h);
    return h;
}

int Builder::lookup_source(const std::string& name, WeightSource* out) {
    if (!sources_) return RAD_E_NOTFOUND;
    auto it = source_cache_.find(name);
    if (it != source_cache_.end()) {
        *out = it->second.second;
        return it->second.first;
    }
    WeightSource ws;
    const int st = sources_(name, prog_, &ws);
    /* Only an answer is remembered: a weight not found before its name map entry was declared
     * may be found after it. */
    if (st == RAD_OK) source_cache_.emplace(name, std::make_pair(st, ws));
    *out = ws;
    return st;
}

int Builder::weight_encoding(const char* source, RadEncoding* out, int64_t* shape,
                             uint32_t* rank) {
    if (!source || !*source || !out) return RAD_E_INVAL;
    WeightSource ws;
    const int st = lookup_source(scoped(source), &ws);
    if (st != RAD_OK) return st;
    *out = ws.enc;
    if (rank) *rank = ws.rank;
    if (shape)
        for (int i = 0; i < RAD_MAX_RANK; ++i) shape[i] = i < (int)ws.rank ? ws.shape[i] : 0;
    return RAD_OK;
}

/* ================================================================== buffers */
int64_t buffer_bytes(const RadBufDecl& d) {
    int64_t numel = decl_numel(d.rank, d.shape);
    if (numel == 0) return 0;
    return rad_dtype_bytes(d.dtype, numel);
}

rad_buf Builder::decl_buffer(const char* name, const RadBufDecl* d) {
    if (finished_) return err(RAD_E_STATE, "decl_buffer('%s') after declare finished", name ? name : "");
    if (!name || !*name || !d) return err(RAD_E_INVAL, "decl_buffer: null name or declaration");

    std::string n = scoped(name);
    for (size_t i = 1; i < prog_.buffers.size(); ++i)
        if (prog_.buffers[i].name == n)
            return err(RAD_E_DUPLICATE, "buffer '%s' declared twice", n.c_str());

    if (d->kind < RAD_BUF_TRANSIENT || d->kind > RAD_BUF_DERIVED)
        return err(RAD_E_INVAL, "buffer '%s': unknown kind %d", n.c_str(), d->kind);
    if (d->domain != RAD_DOMAIN_DEVICE && d->domain != RAD_DOMAIN_HOST)
        return err(RAD_E_INVAL, "buffer '%s': unknown domain %d", n.c_str(), d->domain);

    BufferInfo b;
    b.name = n;
    b.decl = *d;
    b.bytes = buffer_bytes(*d);
    if (b.bytes == 0)
        return err(RAD_E_SHAPE, "buffer '%s': the core cannot size dtype %u at rank %u -- an "
                                "activation must be a dtype the buffer planner understands",
                   n.c_str(), d->dtype, d->rank);

    prog_.buffers.push_back(std::move(b));
    return (rad_buf)(prog_.buffers.size() - 1);
}

/* ================================================================== KV groups */
rad_kvgroup Builder::decl_kv_group(const char* name, const RadKVGroupDecl* d) {
    if (finished_) return err(RAD_E_STATE, "decl_kv_group('%s') after declare finished", name ? name : "");
    if (!name || !*name || !d) return err(RAD_E_INVAL, "decl_kv_group: null name or declaration");
    if (d->kind < RAD_KV_FULL || d->kind > RAD_KV_CONV)
        return err(RAD_E_INVAL, "kv group '%s': unknown kind %d", name, d->kind);

    std::string n = scoped(name);
    for (size_t i = 1; i < prog_.kv_groups.size(); ++i)
        if (prog_.kv_groups[i].name == n)
            return err(RAD_E_DUPLICATE, "kv group '%s' declared twice", n.c_str());

    KVGroupInfo g;
    g.name = n;
    g.decl = *d;
    prog_.kv_groups.push_back(std::move(g));
    kv_decl_at_.push_back((int32_t)prog_.ops.size());   /* where in the op list we are now */
    kv_consumer_.push_back(-1);
    return (rad_kvgroup)(prog_.kv_groups.size() - 1);
}

int Builder::bind_layer_kv(int layer, rad_kvgroup g) {
    if (g == 0 || g >= prog_.kv_groups.size()) {
        err(RAD_E_INVAL, "bind_layer_kv(layer %d): kv group handle %u is not one this builder "
                         "handed out", layer, g);
        return RAD_E_INVAL;
    }
    if (layer < 0) { err(RAD_E_INVAL, "bind_layer_kv: layer %d", layer); return RAD_E_INVAL; }
    auto& v = prog_.kv_groups[g].layers;
    if (std::find(v.begin(), v.end(), layer) == v.end()) v.push_back(layer);
    return RAD_OK;
}

/* Which op reads this group.
 *
 * Nothing in the ABI links an op to a KV group: rad_decl_op names parameters and weight operands,
 * and a group is not either of those. Declaration order is therefore the binding, and it is the
 * order the documentation already prescribes -- "declare the group, declare the op, then ask
 * rad_kv_block_size()" (docs/OPS.md, spec §7.2). The consuming op is the first op declared after
 * the group that carries a block_size, and an op serves at most one group, so a hybrid model's
 * full and sliding-window groups each claim their own attention op.
 *
 * "Carries a block_size" means any of three things, and all three are checked because an
 * architecture plugin may legitimately do none of the others: the op DECLARED one, its SCHEMA
 * names one (attn_paged does, so the op is recognisable even when the plugin left the value off
 * precisely because it is about to ask for it), or its resolved kernel CONSTRAINS one. */
int32_t Builder::find_kv_consumer(rad_kvgroup g) {
    if (kv_consumer_[g] >= 0) return kv_consumer_[g];

    for (int32_t i = kv_decl_at_[g]; i < (int32_t)prog_.ops.size(); ++i) {
        bool claimed = false;
        for (size_t k = 1; k < kv_consumer_.size(); ++k) if (kv_consumer_[k] == i) claimed = true;
        if (claimed) continue;

        const OpInfo& o = prog_.ops[i];
        bool carries = o.base.has("block_size");
        if (!carries && o.schema && o.schema->params)
            for (int c = 0; c < o.schema->n_params && !carries; ++c)
                if (o.schema->params[c].key && !std::strcmp(o.schema->params[c].key, "block_size"))
                    carries = true;
        if (!carries)
            for (const Band& b : o.bands)
                for (int d = 0; d < RAD_N_DOMAINS && !carries; ++d)
                    if (b.dom[d] && b.dom[d].row && b.dom[d].row->info)
                        for (int c = 0; c < b.dom[d].row->info->n_constraints; ++c)
                            if (!std::strcmp(b.dom[d].row->info->constraints[c].key, "block_size"))
                                carries = true;
        if (carries) { kv_consumer_[g] = i; return i; }
    }
    return -1;
}

int64_t Builder::kv_block_size(rad_kvgroup g) {
    if (g == 0 || g >= prog_.kv_groups.size()) {
        err(RAD_E_INVAL, "rad_kv_block_size: handle %u is not a kv group this builder handed out", g);
        return 0;
    }
    KVGroupInfo& gi = prog_.kv_groups[g];

    /* A per-sequence state has no block size and asking for one is a category error, not a
     * failure. Say so once and return 0; the caller wants bytes_per_state. */
    if (gi.decl.kind == RAD_KV_LINEAR || gi.decl.kind == RAD_KV_CONV) {
        RAD_WARN("kv group '%s' is per-sequence state; it has no block size", gi.name.c_str());
        return 0;
    }
    if (gi.block_size) return gi.block_size;

    int32_t consumer = find_kv_consumer(g);
    if (consumer < 0) {
        /* A silent zero here becomes a division by zero in the block manager three components
         * later, so it is an error with the reason attached. */
        err(RAD_E_STATE,
            "rad_kv_block_size('%s'): no op that reads this group has been declared yet. The "
            "block size comes off the RESOLVED attention kernel's block_size constraint, so the "
            "attention op has to exist before there is an answer (spec §7.2)", gi.name.c_str());
        return 0;
    }

    const OpInfo& o = prog_.ops[consumer];

    /* The kernel's own constraint is the answer, read with core/plugin/'s own accessor: a second
     * constraint walk here would be a second reading of the same rows. A core that picked its own
     * block size would be a core that has to be edited when a kernel changes (spec §7.2). */
    for (const Band& b : o.bands)
        for (int d = 0; d < RAD_N_DOMAINS; ++d) {
            if (!b.dom[d] || !b.dom[d].row) continue;
            long long bs = 0;
            if (kernel_constraint(*b.dom[d].row, "block_size", RAD_C_EQ, &bs) && bs > 0) {
                gi.block_size = bs;
                return gi.block_size;
            }
        }

    /* No kernel constrained it: this kernel serves any block size (libref does exactly
     * that -- correct, slow, no geometry constraints, spec §17). Then the number is the plugin's
     * declared parameter, which is still not the core choosing one. */
    long long v = 0;
    if (o.base.get_i("block_size", &v) && v > 0) { gi.block_size = v; return gi.block_size; }

    /* Nothing to read it off, and the reason is almost always the same one, so it is named here in
     * full rather than left as "0".
     *
     * A constraint whose key the geometry does not carry does NOT hold (rad_util.cpp, and it is
     * libr4d's rule for good reasons). So a kernel constrained to `block_size == 16` cannot be
     * selected by a query that omits block_size -- which means an attention op declared WITHOUT a
     * block_size can never resolve to the kernel whose constraint the answer was supposed to come
     * from. The plugin has to declare the block size it wants; the resolved kernel's constraint
     * then either confirms it or refuses it by name, and THAT is what keeps the number the
     * kernel's rather than the core's (spec §7.2). */
    bool resolved_anywhere = false;
    for (const Band& b : o.bands)
        for (int d = 0; d < RAD_N_DOMAINS; ++d) if (b.dom[d]) resolved_anywhere = true;

    err(RAD_E_STATE,
        "rad_kv_block_size('%s'): op '%s' consumes this group but %s. Declare block_size on the "
        "op with the value this model wants: a kernel constrained to a block size cannot be "
        "selected by a query that omits it, so leaving it off means the resolution it was to be "
        "read from never happens (spec §7.2)",
        gi.name.c_str(), o.op.c_str(),
        resolved_anywhere ? "neither its resolved kernel nor its declared parameters name a "
                            "block_size"
                          : "it resolved to no kernel at all, so there is no constraint to read");
    return 0;
}

/* ================================================================== ops */
void Builder::record_tune_request(const Resolved& r) {
    if (!r.row || !r.row->info || r.row->info->n_tunables <= 0) return;
    /* ONE REQUEST PER KERNEL INSTANTIATION, and the instantiation is (kernel, plugin, the input
     * params it was resolved at). Every distinct shape a kernel is asked for is a separate
     * benchmark, and the same shape asked for by sixty-four layers is one (spec §15).
     *
     * `geom_key` and not geom.str(): the tuner and the selector must key the same way or the cache
     * never hits, and geom_key is tunecache.h's canonical spelling -- sorted, and taken BEFORE the
     * tuned axes were folded into the geometry. geom.str() preserves declaration order and now
     * also carries the axes, so keying on it would make the key depend on its own answer. */
    const std::string key = r.row->info->name + std::string("\x1f") + r.row->plugin +
                            std::string("\x1f") + r.geom_key;
    for (auto& t : prog_.tune_requests)
        if (t.kernel + std::string("\x1f") + t.plugin + std::string("\x1f") + t.geom_key == key)
            return;

    Program::TuneRequest req;
    req.kernel     = r.row->info->name;
    req.plugin     = r.row->plugin;
    req.geom       = r.geom;
    req.geom_key   = r.geom_key;
    req.info       = r.row->info;
    prog_.tune_requests.push_back(std::move(req));
}

/* The geometry a band is DISPLAYED as: the fixed parameters plus the ranged one at the band's
 * upper bound. Diagnostics only -- a resolved band already carries the geometry the registry froze
 * it at, and that is the one the kernel is handed. */
static Geometry display_geometry(const OpInfo& oi, const Band& b) {
    Geometry g = oi.base;
    if (!oi.ranged_key.empty() && b.hi != INT64_MAX) g.set_i(oi.ranged_key, b.hi);
    return g;
}

void Builder::resolve_bands(OpInfo& oi) {
    /* The bucket table is core/plugin/'s and this component does not second-guess it: the
     * boundaries are the union of the constraint values candidate kernels place on the ranged
     * parameter, clamped to the declared range, and both domains are resolved AT EACH BAND'S UPPER
     * BOUND -- the worst case in the band and therefore the only value safe to select and size on.
     * An empty ranged_key comes back as exactly one band whose hi is INT64_MAX, so the
     * no-ranged-parameter case needs no separate path and band_for() always finds it. */
    /* A KERNEL THAT CANNOT READ A WEIGHT'S ENCODING IS NOT A CANDIDATE FOR THE OP. Its layout
     * hook says so with RAD_E_DTYPE, and a device row with no hook reads one plane as it is and
     * nothing else -- so a bf16 head goes to the bf16 GEMM rather than being refused by the fp8
     * one, and the same declaration serves a checkpoint and its quantised container. The host
     * domain's hookless rows are libref, which reads canonically and is gated elsewhere. */
    const Registry::RowFilter accept = [&](const KernelRow& r, const Geometry& g) -> std::string {
        const RadKernelInfo* ki = r.info;
        for (size_t wi = 0; wi < oi.weights.size(); ++wi) {
            const rad_weight h = oi.weights[wi];
            if (h == 0 || h >= prog_.weights.size()) continue;
            const WeightInfo& w = prog_.weights[h];
            if (sources_ && !w.enc_known && w.decl.optional) continue;
            const int operand = wi < oi.weight_opd.size() && oi.weight_opd[wi] >= 0
                                    ? (int)oi.weight_opd[wi] : (int)wi;
            char enc[256];
            if (!ki->layout) {
                if (ki->domain == RAD_DOMAIN_DEVICE && w.n_sel > 1) {
                    rad_enc_format(&w.enc, enc, sizeof enc);
                    return fmt("reads '%s' as stored, and it is %d planes of %s", w.name.c_str(),
                               w.n_sel, enc);
                }
                continue;
            }
            RadTensor t[RAD_ENC_MAX_PLANES];
            weight_planes(w, t);
            RadLayout L{};
            if (ki->layout(g.params(), g.n_params(), operand, &w.enc, w.sel, t, w.n_sel, &L) ==
                RAD_E_DTYPE) {
                rad_enc_format(&w.enc, enc, sizeof enc);
                return fmt("does not read '%s' encoded %s", w.name.c_str(), enc);
            }
        }
        return std::string();
    };
    oi.bands = reg_.build_bands(oi.op, oi.ranged_key, oi.range_lo, oi.range_hi, oi.base,
                                oi.weights.empty() ? nullptr : &accept);

    if (oi.bands.empty()) {
        /* No rows for this op at all. The range still owes an answer, so the miss report names a
         * band rather than saying nothing about the op. */
        oi.bands.emplace_back();
        oi.bands.back().hi = oi.ranged_key.empty() ? INT64_MAX : oi.range_hi;
        for (int d = 0; d < RAD_N_DOMAINS; ++d)
            oi.bands.back().miss[d] = "no kernel implements this op in this domain";
    }

    for (Band& b : oi.bands) {
        for (int d = 0; d < RAD_N_DOMAINS; ++d) {
            Resolved& r = b.dom[d];

            if (!r) {
                /* One entry per op per band per domain that resolved to nothing, each carrying the
                 * constraint that refused it. build_bands records and does not abort -- a device
                 * miss is fatal for that band and a host miss removes a placement option, and
                 * which of those matters is not selection's call. Declare always completes and
                 * reports the list whole (spec §3.1). */
                prog_.misses.push_back({ oi.op, display_geometry(oi, b).str(), b.span,
                                         b.miss[d].empty() ? "no kernel matched" : b.miss[d],
                                         d });
                continue;
            }

            /* Scratch at the same value the band was selected at. r.geom IS that value, frozen by
             * the registry, so asking it here cannot drift from what the choice was made against.
             * RadArgs carries no tensors: a scratch hook is a function of the geometry, which is
             * the whole reason it can be called at declare at all. */
            if (r.row->info && r.row->info->scratch) {
                RadArgs a{};
                a.p          = r.geom.params();
                a.n_p        = r.geom.n_params();
                a.rank       = prog_.ctx.rank;
                a.world_size = prog_.ctx.world_size;
                int64_t s = r.row->info->scratch(&a);
                if (s < 0)
                    err(RAD_E_SCRATCH, "%s: kernel %s refused to size its scratch for %s: %s",
                        oi.op.c_str(), r.row->info->name, r.geom.str().c_str(),
                        rad_strerror((int)s));
                else if (s > r.scratch_bytes)
                    r.scratch_bytes = s;
            }
            if (r.scratch_bytes > prog_.scratch_bytes) prog_.scratch_bytes = r.scratch_bytes;

            record_tune_request(r);
        }
    }
}

/* init() once per resolved instance -- one op, one band, one domain -- run once the WHOLE graph
 * has resolved (spec §2.1). Not during resolution: one kernel's failing init would then abort
 * declare before the miss report was assembled, and the miss report is the point of the phase.
 *
 * A negative init is not a runtime condition. The kernel matched the constraints and then refused
 * the instance, so the band becomes a miss and the report names it here, rather than the failure
 * surfacing at the first launch three frames away. */
void Builder::finish_instances() {
    for (size_t i = 1; i < prog_.ops.size(); ++i) {
        OpInfo& oi = prog_.ops[i];
        for (Band& b : oi.bands) {
            for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                Resolved& r = b.dom[d];
                if (!r) continue;
                const int s = reg_.init_instance(r, prog_.ctx.rank, prog_.ctx.world_size);
                if (s >= 0) continue;

                err(s, "%s: kernel %s matched %s and then its init() refused the instance: %s",
                    oi.op.c_str(), r.row->info ? r.row->info->name : "?",
                    r.geom.str().c_str(), rad_strerror(s));
                b.miss[d] = fmt("init() refused the instance: %s", rad_strerror(s));
                prog_.misses.push_back({ oi.op, display_geometry(oi, b).str(), b.span,
                                         b.miss[d], d });
                r = Resolved{};
            }
        }
    }
}
rad_op Builder::decl_op(const char* opname, const RadParam* p, int n_p,
                        const rad_weight* w, int n_w) {
    if (finished_) return err(RAD_E_STATE, "decl_op('%s') after declare finished", opname ? opname : "");
    if (!opname || !*opname) return err(RAD_E_INVAL, "decl_op: null op name");
    if (n_p < 0 || n_w < 0) return err(RAD_E_INVAL, "decl_op('%s'): negative count", opname);

    /* ---- the schema, checked by the component that owns it.
     *
     * Registry::validate covers the whole parameter side: no such op (with the near miss named, so
     * a typo and an unimplemented op are not the same diagnostic), an unknown key, a key given
     * twice, a type mismatch, an empty range, and the one-ranged-parameter rule. A second copy of
     * that check here would be a second answer to one question, and the two would drift. The
     * sentence it fills in is written for an architecture-plugin author, so it goes into the
     * report unchanged. */
    std::string verr;
    const int v = reg_.validate(opname, p, n_p, &verr);
    if (v < 0) return err(v, "%s", verr.c_str());

    const RadOpSchema* sc = reg_.schema(opname);
    if (!sc) return err(RAD_E_NOSCHEMA, "op '%s' has no declared schema", opname);

    /* ---- parameters. validate() already guaranteed the shape of this list, so this only has to
     * split the fixed parameters from the single ranged one. */
    Geometry base;
    std::string ranged_key;
    int64_t lo = 0, hi = 0;

    for (int i = 0; i < n_p; ++i) {
        const RadParam& q = p[i];
        switch (q.kind) {
            case RAD_P_INT:   base.set_i(q.key, q.ival); break;
            case RAD_P_STR:   base.set_s(q.key, q.sval); break;
            /* A float parameter carries a value the kernel reads and NOTHING THE SELECTOR MATCHES
             * ON: a tolerance is not a predicate, and a kernel that cares about eps to that
             * precision has a bug (docs/OPS.md). So it goes into the geometry, where the kernel
             * finds it, and never becomes a band boundary. */
            case RAD_P_F64:   base.set_f(q.key, q.dval); break;
            case RAD_P_RANGE: ranged_key = q.key; lo = q.ival; hi = q.ihi; break;
            default:
                return err(RAD_E_INVAL, "op '%s': parameter '%s' has kind %d, which is not INT, "
                                        "F64, STR or RANGE", opname, q.key, q.kind);
        }
    }

    /* ---- operand count, which validate() does not see: it is handed the parameters and not the
     * operands. Arguments are positional, so a count that disagrees with the schema means the same
     * call means different things to different kernels -- not a diagnosable failure at issue, but
     * silent numerical garbage (spec §2.3). Only the WEIGHT operands are named at declare, so that
     * is what is counted here; the full count is checked against this same schema at issue. */
    size_t before = errors_.size();
    int w_req = 0, w_max = 0, n_tab = 0;
    for (int i = 0; sc->operands && i < sc->n_operands; ++i) {
        if (sc->operands[i].role == RAD_OPD_WTAB) { ++n_tab; continue; }
        if (sc->operands[i].role != RAD_OPD_WEIGHT) continue;
        ++w_max;
        if (!sc->operands[i].optional) ++w_req;
    }
    /* ---- and where the operand is a TABLE, the run length.
     *
     * A RAD_OPD_WTAB position consumes a RUN of declared weights, so the count alone does not
     * say how the list divides. The rule is EQUAL DIVISION: whatever is left after the fixed
     * weight positions have taken one each is split evenly across the table positions, in
     * declaration order. That is exactly right for the case tables exist for -- a grouped GEMM's
     * codes and scale planes are one of each per expert, always the same count.
     *
     * WHERE IT DOES NOT COME OUT WHOLE, THE FIRST POSITIONS TAKE ONE MORE. A grouped GEMM whose
     * n experts are C classes by number mod C, declared class by class as a codes table and a
     * scales table each, gives its first `n % C` classes one expert more than the rest -- exactly
     * the positions this rule lengthens. A list grouped any other way is refused at issue: the
     * plugin names each table's run there, and a run other than the one this division bound is
     * refused. A bound run must also be contiguous, checked when the tables are bound before the
     * first step, and its entries of one shape, checked at its first issue. A list shorter than
     * the tables is refused here. */
    int per_tab = 0, tab_extra = 0;
    if (n_tab > 0) {
        const int rest = n_w - w_max;
        if (rest < n_tab) {
            err(RAD_E_SCHEMA,
                "op '%s': %d weight operands declared; the schema has %d fixed weight operands and "
                "%d weight TABLE operands, so %d leaves a table empty. A table takes one weight "
                "per expert", opname, n_w, w_max, n_tab, rest);
        } else {
            per_tab = rest / n_tab;
            tab_extra = rest % n_tab;
        }
    } else if (n_w < w_req || n_w > w_max) {
        err(RAD_E_SCHEMA, "op '%s': %d weight operands declared, but the schema has %d (%d "
                          "required) among its %d operands", opname, n_w, w_max, w_req,
            sc->n_operands);
    }

    for (int i = 0; i < n_w; ++i)
        if (w[i] == 0 || w[i] >= prog_.weights.size())
            err(RAD_E_INVAL, "op '%s': weight operand %d is handle %u, which this builder did not "
                             "hand out", opname, i, w[i]);

    if (errors_.size() != before) return RAD_NULL_HANDLE;

    /* ---- resolve. */
    OpInfo oi;
    oi.op         = opname;
    oi.base       = std::move(base);
    oi.schema     = sc;
    oi.index      = (int32_t)prog_.ops.size();
    if (!ranged_key.empty()) {
        oi.ranged_key = ranged_key;
        oi.range_lo   = lo;
        oi.range_hi   = hi;
    }
    oi.weights.assign(w, w + n_w);
    /* The many-to-one map from declared weight to schema operand position, built here so the
     * layout pass, the converter and the issue path all read one answer. A fixed weight position
     * takes one; a table position takes `per_tab` in a row, and the first `tab_extra` of them one
     * more. */
    oi.weight_opd.assign((size_t)n_w, -1);
    {
        int wi = 0, ti = 0;
        for (int k = 0; sc->operands && k < sc->n_operands && wi < n_w; ++k) {
            const int role = sc->operands[k].role;
            if (role == RAD_OPD_WTAB) {
                const int run = per_tab + (ti++ < tab_extra ? 1 : 0);
                for (int j = 0; j < run && wi < n_w; ++j) oi.weight_opd[(size_t)wi++] = k;
            } else if (role == RAD_OPD_WEIGHT) {
                oi.weight_opd[(size_t)wi++] = k;
            }
        }
    }

    if (ref_) {
        const size_t at = prog_.ops.size();
        if (at >= ref_->ops.size() || ref_->ops[at].op != oi.op)
            return err(RAD_E_STATE, "sizing declare: op %zu is '%s', and the real declare had "
                                    "'%s' there -- the plugin declares a different graph at a "
                                    "smaller max_tok", at, opname,
                       at < ref_->ops.size() ? ref_->ops[at].op.c_str() : "nothing");
        /* AND WITH THE SAME FIXED PARAMETERS. The kernel was chosen for the real declare's, and it
         * runs with them: an op whose fixed geometry follows max_tok -- a row count declared as a
         * constant rather than a range -- would be handed the real size while its buffers are the
         * smaller level's, and write past them into whatever the arena lent out. */
        const Geometry& rb = ref_->ops[at].base;
        auto role_of = [&](const char* key) {
            for (int k = 0; sc->params && k < sc->n_params; ++k)
                if (sc->params[k].key && std::strcmp(sc->params[k].key, key) == 0)
                    return sc->params[k].role;
            return (int)RAD_PROLE_NONE;
        };
        const RadParam* ra = rb.params();
        const RadParam* rq = oi.base.params();
        const int na = rb.n_params(), nq = oi.base.n_params();
        std::string moved = na == nq ? "" : "the parameter list itself";
        for (int x = 0; x < na && moved.empty(); ++x) {
            const RadParam* y = nullptr;
            for (int k = 0; k < nq; ++k)
                if (std::strcmp(rq[k].key, ra[x].key) == 0) { y = &rq[k]; break; }
            if (!y || y->kind != ra[x].kind) { moved = ra[x].key; break; }
            bool same = false;
            switch (ra[x].kind) {
                case RAD_P_INT:
                    /* A CAPACITY MAY SHRINK: its kernel writes by its operands' extents. */
                    same = y->ival == ra[x].ival ||
                           (role_of(ra[x].key) == RAD_PROLE_CAPACITY && y->ival <= ra[x].ival);
                    break;
                case RAD_P_STR: same = std::strcmp(y->sval ? y->sval : "",
                                                   ra[x].sval ? ra[x].sval : "") == 0; break;
                default:        same = y->dval == ra[x].dval; break;
            }
            if (!same) moved = ra[x].key;
        }
        if (!moved.empty())
            return err(RAD_E_STATE, "sizing declare: op %zu ('%s') changes '%s' with the step size "
                                    "(%s at the smaller max_tok, %s at the real one), and its "
                                    "kernel was chosen and runs with the real one", at, opname,
                       moved.c_str(), oi.base.str().c_str(), rb.str().c_str());
        oi.bands = ref_->ops[at].bands;
    } else {
        resolve_bands(oi);
    }

    rad_op h = (rad_op)prog_.ops.size();
    bool resolved = true;
    for (const Band& b : oi.bands)
        if (!b.dom[RAD_DOMAIN_DEVICE] && !b.dom[RAD_DOMAIN_HOST]) resolved = false;

    prog_.ops.push_back(std::move(oi));
    op_bufs_.emplace_back();

    /* An op recorded but not handed back: the graph dump still shows it and its holes, and the
     * plugin gets a falsy handle so it can offer the unfused sequence instead. A miss is not an
     * error yet -- it becomes one only if the plugin issues it (spec §2.3, §3.2). */
    return resolved ? h : RAD_NULL_HANDLE;
}

int Builder::op_resolved(rad_op h) const {
    if (h == 0 || h >= prog_.ops.size()) return 0;
    const OpInfo& o = prog_.ops[h];
    if (o.bands.empty()) return 0;
    for (const Band& b : o.bands)
        if (!b.dom[RAD_DOMAIN_DEVICE] && !b.dom[RAD_DOMAIN_HOST]) return 0;
    return 1;
}

int Builder::op_reads(rad_op h, const rad_buf* b, int n) {
    if (h == 0 || h >= prog_.ops.size()) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (b[i] == 0 || b[i] >= prog_.buffers.size()) return RAD_E_INVAL;
        op_bufs_[h].reads.push_back(b[i]);
    }
    return RAD_OK;
}

int Builder::op_writes(rad_op h, const rad_buf* b, int n) {
    if (h == 0 || h >= prog_.ops.size()) return RAD_E_INVAL;
    for (int i = 0; i < n; ++i) {
        if (b[i] == 0 || b[i] >= prog_.buffers.size()) return RAD_E_INVAL;
        op_bufs_[h].writes.push_back(b[i]);
    }
    return RAD_OK;
}

/* SEE rad_buf_concurrent. A buffer a second lane touches is not bounded by the op range its
 * declarations imply, so liveness hands it the whole program. Recorded rather than applied here
 * because liveness runs at finish(), after every declaration is in. */
int Builder::buf_concurrent(rad_buf b) {
    if (finished_) { err(RAD_E_STATE, "buf_concurrent after declare finished"); return RAD_E_STATE; }
    if (b == 0 || b >= prog_.buffers.size()) return RAD_E_INVAL;
    concurrent_bufs_.insert(b);
    return RAD_OK;
}

/* SEE rad_weight_shard_span. Checked against the declaration here, where the plugin that got it
 * wrong can be named; the loader checks it against the container. */
int Builder::weight_shard_span(rad_weight h, int64_t lo, int64_t hi) {
    if (finished_) { err(RAD_E_STATE, "weight_shard_span after declare finished"); return RAD_E_STATE; }
    if (h == 0 || h >= prog_.weights.size()) return RAD_E_INVAL;
    WeightInfo& w = prog_.weights[h];
    const RadWeightDecl& d = w.decl;
    if (d.shard != RAD_SHARD_ROW && d.shard != RAD_SHARD_COL) {
        err(RAD_E_INVAL, "weight '%s': a shard span on a weight that is not row- or column-sharded",
            w.name.c_str());
        return RAD_E_INVAL;
    }
    if (lo < 0 || hi <= lo) {
        err(RAD_E_INVAL, "weight '%s': shard span [%lld, %lld) is empty", w.name.c_str(),
            (long long)lo, (long long)hi);
        return RAD_E_INVAL;
    }
    /* The declared extent is the rank's own, so it has to be the span's length: each stacked
     * part's for ROW, the columns for COL. */
    bool ok;
    if (d.shard == RAD_SHARD_ROW) {
        if (d.n_row_parts > 1) {
            ok = true;
            for (int g = 0; g < d.n_row_parts; ++g) ok = ok && d.row_parts[g] == hi - lo;
        } else {
            ok = d.rank >= 1 && d.shape[0] == hi - lo;
        }
    } else {
        ok = d.rank == 2 && d.shape[1] == hi - lo;
    }
    if (!ok) {
        err(RAD_E_SHAPE, "weight '%s': shard span [%lld, %lld) is %lld wide and the declared "
                         "extent of the split dimension is not", w.name.c_str(), (long long)lo,
            (long long)hi, (long long)(hi - lo));
        return RAD_E_SHAPE;
    }
    w.shard_lo = lo;
    w.shard_hi = hi;
    return RAD_OK;
}

/* ================================================================== name map */
int Builder::decl_name_map(const RadNameMap* m) {
    if (finished_) { err(RAD_E_STATE, "decl_name_map after declare finished"); return RAD_E_STATE; }
    if (!m || !m->declared) { err(RAD_E_INVAL, "decl_name_map: null map"); return RAD_E_INVAL; }
    if (m->n_src < 1 || m->n_src > 8) {
        err(RAD_E_INVAL, "name map for '%s': %d source tensors, 1..8 allowed", m->declared, m->n_src);
        return RAD_E_INVAL;
    }
    if (m->mode != RAD_MAP_COPY && m->mode != RAD_MAP_CONCAT) {
        err(RAD_E_INVAL, "name map for '%s': unknown mode %d", m->declared, m->mode);
        return RAD_E_INVAL;
    }

    RadNameMap c{};
    c.declared   = intern(scoped(m->declared));
    c.mode       = m->mode;
    c.n_src      = m->n_src;
    c.concat_dim = m->concat_dim;
    for (int i = 0; i < m->n_src; ++i) {
        if (!m->src[i]) {
            err(RAD_E_INVAL, "name map for '%s': source %d is null", m->declared, i);
            return RAD_E_INVAL;
        }
        c.src[i] = intern(m->src[i]);
    }
    /* AND `src_index`, BECAUSE THIS COPY IS FIELD BY FIELD. Every field added to RadNameMap has to
     * be added here too, and the cost of missing one is not that the value is lost: `RadNameMap
     * c{}` value-initialises it to ZERO, and zero means "sub-tensor 0 of this source" while the
     * ABI's own default and arch/common/rad_arch.h's map_base() both say -1, "the whole tensor".
     * A dropped field is therefore every map in every model quietly asking for slice 0 of its
     * source (rad_convert.cpp refuses an index it cannot serve rather than ignoring it). */
    for (int i = 0; i < m->n_src; ++i) c.src_index[i] = m->src_index[i];
    for (int i = m->n_src; i < 8; ++i) c.src_index[i] = -1;

    /* A CONCAT MAP MAY BE DECLARED IN PIECES, AND EVERY OTHER REPEAT IS A BUG.
     *
     * `src[8]` is the ABI's cap and a sharded checkpoint blows straight through it:
     * Qwen3.8-Flash-Next stores its 320-million-row n-gram embedding as 128 tensors named
     * `...ngram_embedding.shard_K.weight`. Growing the array is the wrong answer -- there is one
     * map per declared weight and the full model declares about fifty thousand of them, so 8 -> 128
     * would cost ~80 MB of Program to serve one tensor.
     *
     * So successive CONCAT maps for one declared name EXTEND it, in call order, and rad-convert
     * joins their sources end to end. arch/common/rad_arch.h's map_concat_shards() is the one-line
     * form; nothing else in the tree needs more than eight.
     *
     * Every OTHER repeat is refused rather than merged, and the refusal is the point. A lookup
     * built with `std::map::emplace` KEEPS THE FIRST and drops the rest without a word, so two
     * maps for one weight -- a copy-paste, a loop with a constant index -- convert a model whose
     * weights came from the wrong tensors and say nothing. A merge rule that is a merge rule has
     * to say which repeats it is not. */
    auto seen = name_map_first_.find(c.declared);
    if (seen != name_map_first_.end()) {
        const RadNameMap& first = prog_.name_map[seen->second];
        if (c.mode != RAD_MAP_CONCAT || first.mode != RAD_MAP_CONCAT) {
            err(RAD_E_INVAL, "name map for '%s': declared twice, and only a CONCAT map may be "
                             "extended -- a repeated COPY map means two sources for one weight, and "
                             "which one wins is not something a declaration can express",
                c.declared);
            return RAD_E_INVAL;
        }
        if (c.concat_dim != first.concat_dim) {
            err(RAD_E_INVAL, "name map for '%s': extended at concat_dim %d having been declared at "
                             "%d; the pieces of one concatenation join along one axis",
                c.declared, c.concat_dim, first.concat_dim);
            return RAD_E_INVAL;
        }
    } else {
        name_map_first_.emplace(c.declared, prog_.name_map.size());
    }

    prog_.name_map.push_back(c);
    return RAD_OK;
}

/* Which buffer the core samples from. Refusing a second call rather than letting the last one win:
 * two plugins declaring into one builder (a target and its drafter, spec §10) both have logits,
 * and silently keeping the second would sample the draft model's distribution into the target's
 * tokens -- fluent wrong text, and nothing would say so. The drafter's own logits are reached
 * through its scope, not through this. */
int Builder::declare_logits(rad_buf b) {
    if (finished_) { err(RAD_E_STATE, "declare_logits after declare finished"); return RAD_E_STATE; }
    if (b == RAD_NULL_HANDLE || (size_t)b >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_logits: %u is not a declared buffer", (unsigned)b);
        return RAD_E_INVAL;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE && prog_.logits_buf != b) {
        err(RAD_E_STATE, "declare_logits: '%s' was already named as the logits buffer; '%s' is a "
                         "second one. A scoped sub-model's logits are its own, not the step's.",
            prog_.buffers[prog_.logits_buf].name.c_str(), prog_.buffers[b].name.c_str());
        return RAD_E_STATE;
    }
    if (b == prog_.drafter_tree_buf) {
        err(RAD_E_INVAL, "declare_logits: buffer %u cannot be the drafter tree buffer", (unsigned)b);
        return RAD_E_INVAL;
    }
    prog_.logits_buf = b;
    return RAD_OK;
}


/* WHAT THE ENCODER IS. Checked here, at declare, for the drafter's reason: an encoder is exercised
 * only when a request carries media, so a declaration the engine cannot drive would otherwise
 * surface on the first image as a pass that reads or writes past a buffer.
 *
 * `max_patches` must be what the deployment asked for, because the plugin sized every activation
 * of its tower with RadBuildCtx::max_enc_patches and the core stages up to this many patches; a
 * disagreement is a tower declared for one pass size and driven at another. `out` has to hold a
 * whole pass's rows, since the core reads them back from it. */
int Builder::declare_encoder(const RadEncoderDecl* d) {
    if (finished_) { err(RAD_E_STATE, "declare_encoder after declare finished"); return RAD_E_STATE; }
    if (!d) { err(RAD_E_INVAL, "declare_encoder: null declaration"); return RAD_E_INVAL; }
    if (d->modalities == 0) return RAD_OK;   /* declaring nothing is declaring nothing */
    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (prog_.encoder.modalities != 0) {
        err(RAD_E_STATE, "declare_encoder: '%s' already declared this program's encoder; '%s' is "
                         "a second one", prog_.encoder_name.c_str(), nm);
        return RAD_E_STATE;
    }
    if ((d->modalities & ~(uint32_t)(RAD_MM_IMAGE | RAD_MM_VIDEO)) != 0) {
        err(RAD_E_INVAL, "declare_encoder '%s': modalities 0x%x names a kind this engine has no "
                         "processor for", nm, (unsigned)d->modalities);
        return RAD_E_INVAL;
    }
    if (d->patch_dim <= 0 || d->merge <= 0 || d->n_embd <= 0) {
        err(RAD_E_INVAL, "declare_encoder '%s': patch_dim %lld, merge %lld, n_embd %lld -- all "
                         "three must be positive", nm, (long long)d->patch_dim,
            (long long)d->merge, (long long)d->n_embd);
        return RAD_E_INVAL;
    }
    if (d->max_patches <= 0 || d->max_patches != prog_.ctx.max_enc_patches ||
        d->max_patches % d->merge != 0) {
        err(RAD_E_INVAL, "declare_encoder '%s': max_patches %lld, but this program was declared "
                         "with max_enc_patches %lld and a merge of %lld. The plugin sized its "
                         "activations with max_enc_patches and the core stages up to that many "
                         "patches a pass; they cannot differ, and a pass's patches must divide into "
                         "whole output rows.", nm, (long long)d->max_patches,
            (long long)prog_.ctx.max_enc_patches, (long long)d->merge);
        return RAD_E_INVAL;
    }
    if (d->out == RAD_NULL_HANDLE || (size_t)d->out >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_encoder '%s': out %u is not a declared buffer", nm,
            (unsigned)d->out);
        return RAD_E_INVAL;
    }
    const RadBufDecl& ob = prog_.buffers[d->out].decl;
    int64_t numel = 1;
    for (uint32_t i = 0; i < ob.rank; ++i) numel *= ob.shape[i];
    if (ob.dtype != RAD_BF16 || ob.domain != RAD_DOMAIN_DEVICE ||
        numel < d->max_patches / d->merge * d->n_embd) {
        err(RAD_E_INVAL, "declare_encoder '%s': out '%s' must be a device bf16 buffer of at least "
                         "%lld x %lld elements -- one pass's rows", nm,
            prog_.buffers[d->out].name.c_str(), (long long)(d->max_patches / d->merge),
            (long long)d->n_embd);
        return RAD_E_INVAL;
    }
    prog_.encoder = *d;
    prog_.encoder_name = nm;
    prog_.encoder.name = nullptr;   /* the plugin's storage; encoder_name is the copy */
    return RAD_OK;
}

/* WHAT THE DRAFT HEAD IS. Every field is checked here rather than at the first draft pass: a
 * drafter is exercised only once a request is speculating, so a decl that is wrong in a way the
 * engine cannot act on would otherwise surface as acceptance that is quietly zero.
 *
 * `depth` must equal what the deployment was configured for. The plugin was TOLD the number in
 * RadBuildCtx::max_spec and sized its own buffers with it, so a disagreement is not a preference
 * to reconcile -- it is the plugin having declared a graph for one depth and a drafter for
 * another, and the engine would drive the second against the first. */
int Builder::declare_drafter(const RadDrafterDecl* d) {
    if (finished_) {
        err(RAD_E_STATE, "declare_drafter after declare finished");
        return RAD_E_STATE;
    }
    if (!d) { err(RAD_E_INVAL, "declare_drafter: null declaration"); return RAD_E_INVAL; }
    if (d->kind == RAD_DRAFT_NONE) return RAD_OK;   /* declaring nothing is declaring nothing */

    const char* nm = d->name && *d->name ? d->name : "(unnamed)";
    if (d->kind != RAD_DRAFT_SERIAL && d->kind != RAD_DRAFT_BLOCK) {
        err(RAD_E_INVAL, "declare_drafter '%s': kind %d is neither RAD_DRAFT_SERIAL nor "
                         "RAD_DRAFT_BLOCK", nm, d->kind);
        return RAD_E_INVAL;
    }
    if (prog_.drafter.kind != RAD_DRAFT_NONE) {
        err(RAD_E_STATE, "declare_drafter: '%s' already declared this program's drafter; '%s' is a "
                         "second one. One step has one draft head -- `draft_pass` carries a sign "
                         "and a round, and it has no room for a third meaning.",
            prog_.drafter_name.c_str(), nm);
        return RAD_E_STATE;
    }
    if (d->depth <= 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': depth %d. A drafter that proposes nothing is not "
                         "declared at all -- leave the call out.", nm, d->depth);
        return RAD_E_INVAL;
    }
    if (prog_.ctx.max_spec > 0 && d->depth != prog_.ctx.max_spec) {
        err(RAD_E_INVAL, "declare_drafter '%s': depth %d, but this program was declared with "
                         "max_spec %d. The plugin sized its own buffers with max_spec and the "
                         "engine drives this depth; they cannot differ.",
            nm, d->depth, prog_.ctx.max_spec);
        return RAD_E_INVAL;
    }
    if (d->proposal != RAD_NULL_HANDLE && (size_t)d->proposal >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_drafter '%s': proposal buffer %u is not a declared buffer",
            nm, (unsigned)d->proposal);
        return RAD_E_INVAL;
    }
    if (d->proposal_pitch < 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': proposal_pitch %lld", nm,
            (long long)d->proposal_pitch);
        return RAD_E_INVAL;
    }
    /* A BLOCK PASS EMBEDS THE MASK ID AT EVERY POSITION IT HAS NOT FILLED. Without one there is
     * nothing to run, and the failure without this check is a block of whatever token id 0 is. */
    if (d->kind == RAD_DRAFT_BLOCK && d->mask_token < 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': RAD_DRAFT_BLOCK with no mask_token. Every "
                         "position the block has not filled embeds it; there is nothing to run.",
            nm);
        return RAD_E_INVAL;
    }
    /* A BLOCK drafter's proposal cannot come from the sampler: the whole block is filled by one
     * pass and the sampler writes one token a row. SERIAL may use either. */
    if (d->kind == RAD_DRAFT_BLOCK && d->proposal == RAD_NULL_HANDLE) {
        err(RAD_E_INVAL, "declare_drafter '%s': RAD_DRAFT_BLOCK with no proposal buffer. One pass "
                         "fills %d positions and the sampler writes one token a row, so there is "
                         "nowhere for the rest of the block to arrive.", nm, d->depth);
        return RAD_E_INVAL;
    }
    /* THE BLOCK'S ROW COUNT, defaulted and then checked against the depth it has to carry. The
     * head runs over the LAST `depth` rows of the block (rad_builder.h), so a block shorter than
     * the depth names rows that do not exist, and a block more than one row longer than the depth
     * leaves leading rows that nothing reads -- neither is a convention, both are a mistake in
     * the plugin, and both produce a drafter whose proposals are simply shifted. */
    int64_t block = d->block;
    if (d->kind == RAD_DRAFT_BLOCK) {
        if (block == 0) block = (int64_t)d->depth + 1;
        const bool ok_single = (block == (int64_t)d->depth || block == (int64_t)d->depth + 1);
        const bool ok_multi  = (block >= 2 && ((int64_t)d->depth % (block - 1) == 0));
        if (block < 2 || (!ok_single && !ok_multi)) {
            err(RAD_E_INVAL, "declare_drafter '%s': block %lld rows at depth %d. Multi-stage or block drafter "
                             "requires stages * (block - 1) == depth or block in {depth, depth + 1}.",
                nm, (long long)block, d->depth);
            return RAD_E_INVAL;
        }
    } else if (block != 0) {
        err(RAD_E_INVAL, "declare_drafter '%s': block %lld on a SERIAL drafter. A serial drafter "
                         "runs one row a round; there is no block to size.",
            nm, (long long)block);
        return RAD_E_INVAL;
    }

    /* THE NAME LIVES IN drafter_name AND ONLY THERE. `drafter.name` is the plugin's pointer, into
     * storage this program does not own, and pointing it at drafter_name instead would not
     * survive the program being moved or copied out of the builder: a short name is stored inside
     * the std::string object itself, so the pointer would name the builder's copy after it is
     * gone. Left null, a reader that forgets this gets nothing rather than freed memory; the
     * engine points its own copy of the decl at its own copy of the string. */
    prog_.drafter      = *d;
    prog_.drafter_name = nm;
    prog_.drafter.name = nullptr;
    prog_.drafter.block = block;
    if (prog_.drafter.proposal_pitch == 0) prog_.drafter.proposal_pitch = 1;
    return RAD_OK;
}

/* Both serial accumulators and block selectors publish the same device-written packed rows.
 * Validate their layout without changing target branching capability. */
int Builder::declare_drafter_tree(rad_buf tree, int64_t node_capacity) {
    if (finished_) {
        err(RAD_E_STATE, "declare_drafter_tree after declare finished");
        return RAD_E_STATE;
    }
    if (prog_.drafter_tree_buf != RAD_NULL_HANDLE || prog_.drafter_tree_capacity != 0) {
        err(RAD_E_STATE, "declare_drafter_tree: tree buffer already declared (%u); duplicate declaration",
            (unsigned)prog_.drafter_tree_buf);
        return RAD_E_STATE;
    }
    if (prog_.drafter.kind != RAD_DRAFT_BLOCK && prog_.drafter.kind != RAD_DRAFT_SERIAL) {
        err(RAD_E_INVAL, "declare_drafter_tree: requires an existing RAD_DRAFT_BLOCK or "
                         "RAD_DRAFT_SERIAL drafter");
        return RAD_E_INVAL;
    }
    if (node_capacity < 1 || node_capacity > (int64_t)radiance::RAD_TREE_MAX_NODES) {
        err(RAD_E_INVAL, "declare_drafter_tree: node_capacity %lld outside [1, %zu]",
            (long long)node_capacity, radiance::RAD_TREE_MAX_NODES);
        return RAD_E_INVAL;
    }
    if (tree == RAD_NULL_HANDLE || (size_t)tree >= prog_.buffers.size()) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer %u is not a declared buffer", (unsigned)tree);
        return RAD_E_INVAL;
    }
    if (tree == prog_.drafter.proposal) {
        err(RAD_E_INVAL, "declare_drafter_tree: tree buffer %u cannot be the linear proposal buffer",
            (unsigned)tree);
        return RAD_E_INVAL;
    }
    if (prog_.logits_buf != RAD_NULL_HANDLE && tree == prog_.logits_buf) {
        err(RAD_E_INVAL, "declare_drafter_tree: tree buffer %u cannot be the logits buffer",
            (unsigned)tree);
        return RAD_E_INVAL;
    }

    const BufferInfo& buf = prog_.buffers[tree];
    if (buf.decl.dtype != RAD_I32) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer '%s' dtype %u != RAD_I32 (%u)",
            buf.name.c_str(), buf.decl.dtype, (unsigned)RAD_I32);
        return RAD_E_INVAL;
    }
    if (buf.decl.rank != 2) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer '%s' rank %u != 2",
            buf.name.c_str(), buf.decl.rank);
        return RAD_E_INVAL;
    }
    const int64_t expected_width = 1 + 2 * node_capacity;
    if (buf.decl.shape[1] != expected_width) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer '%s' row width %lld != %lld (1 + 2 * capacity %lld)",
            buf.name.c_str(), (long long)buf.decl.shape[1], (long long)expected_width, (long long)node_capacity);
        return RAD_E_INVAL;
    }
    if (buf.decl.shape[0] <= 0) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer '%s' rows %lld <= 0",
            buf.name.c_str(), (long long)buf.decl.shape[0]);
        return RAD_E_INVAL;
    }
    if (prog_.ctx.max_seqs > 0 && buf.decl.shape[0] < prog_.ctx.max_seqs) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer '%s' rows %lld < max_seqs %lld",
            buf.name.c_str(), (long long)buf.decl.shape[0], (long long)prog_.ctx.max_seqs);
        return RAD_E_INVAL;
    }
    if (buf.decl.shape[0] > INT64_MAX / (expected_width * (int64_t)sizeof(int32_t))) {
        err(RAD_E_INVAL, "declare_drafter_tree: buffer '%s' byte size overflows int64",
            buf.name.c_str());
        return RAD_E_INVAL;
    }

    prog_.drafter_tree_buf = tree;
    prog_.drafter_tree_capacity = node_capacity;
    return RAD_OK;
}

void Builder::note_v(const char* f, va_list ap) {
    char msg[1024];
    vsnprintf(msg, sizeof msg, f ? f : "", ap);
    prog_.notes.emplace_back(msg);
}

/* ================================================================== finalise */
void Builder::finish_weight_uses() {
    /* Layer offload is a scheduling problem and not a prediction one precisely because this walk
     * exists: declare enumerated every op with its weight operands in order, so the core knows
     * the entire future access sequence statically and prefetch is exact (spec §5).
     *
     * An op that did not resolve still counts here. That widens a weight's interval by at most the
     * op the plugin fell back from, and widening it means prefetching earlier and freeing later --
     * conservative in the safe direction. */
    for (size_t i = 1; i < prog_.ops.size(); ++i) {
        const OpInfo& o = prog_.ops[i];
        for (rad_weight h : o.weights) {
            if (h == 0 || h >= prog_.weights.size()) continue;
            WeightInfo& w = prog_.weights[h];
            if (w.first_use_op < 0) w.first_use_op = o.index;
            w.last_use_op = o.index;
        }
    }
}

/* The weight layout pass (spec §4.3).
 *
 * Nothing in the core knows that libr4d's gemm_w4a8 wants its codes in WMMA fragment order: libr4d
 * says so through its layout hook, given the weight's encoding and this rank's share of its
 * planes, and the manifest records what it said. The loader produces it with the same kernel's
 * relayout. The hook is also where a kernel REFUSES an encoding it does not read, and that refusal
 * is a declare-time error naming the weight, the encoding and the kernel.
 *
 * A weight touched by two kernels that want DIFFERENT stored forms is a real, diagnosable failure
 * and it is named here: a weight is stored once. Discovering that at load, as a shape mismatch,
 * would be much worse. */
void Builder::finish_layouts() {
    /* Who asked for each weight's layout, so a conflict names both kernels rather than one. */
    std::vector<std::string> claimant(prog_.weights.size());

    for (size_t i = 1; i < prog_.ops.size(); ++i) {
        const OpInfo& o = prog_.ops[i];
        if (o.weights.empty()) continue;

        /* The hook indexes the schema's POSITIONAL operand list, not the weight list, and
         * OpInfo::weight_opd is that map -- recorded at declare because RAD_OPD_WTAB makes it
         * many-to-one, so it is not a function of the weight's index. */
        for (size_t wi = 0; wi < o.weights.size(); ++wi) {
            rad_weight h = o.weights[wi];
            if (h == 0 || h >= prog_.weights.size()) continue;
            /* An optional weight the source does not hold is absent, and no kernel reads it. */
            if (sources_ && !prog_.weights[h].enc_known && prog_.weights[h].decl.optional)
                continue;
            int operand = wi < o.weight_opd.size() && o.weight_opd[wi] >= 0
                              ? (int)o.weight_opd[wi] : (int)wi;

            for (size_t bi = 0; bi < o.bands.size(); ++bi) {
                const Band& b = o.bands[bi];
                for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                    const Resolved& r = b.dom[d];
                    if (!r || !r.row->info) continue;
                    const RadKernelInfo* ki = r.row->info;
                    WeightInfo& w = prog_.weights[h];

                    /* THIS RANK'S PLANES, as geometry: what the kernel will be handed, minus the
                     * bytes. */
                    RadTensor t[RAD_ENC_MAX_PLANES];
                    weight_planes(w, t);

                    /* A ROW WITH NO LAYOUT HOOK IS NOT SILENT: it is asserting that it reads the
                     * selected plane AS IT IS. Reading it as "no opinion" instead admits a whole
                     * class of bug.
                     *
                     * The bands of one op can resolve to different kernels -- M <= 32 to one and
                     * M <= 64 to another -- and they all read THE SAME STORED BYTES, because a
                     * weight is stored once. If one of them relayouts and another does not, the
                     * second reads the first one's arrangement under a different interpretation,
                     * and nothing downstream can catch that: the byte count matches.
                     *
                     * The shape it takes, on an fp8 lm_head: logits_gemm_fp8 serves M <= 32 and
                     * asks for codes and scales in one row; the band above it falls to
                     * logits_gemm_bf16, which has no hook and reads the E4M3 codes as bf16. Such a
                     * model loads, serves every request under 32 rows correctly, and returns EMPTY
                     * completions the moment enough sequences batch together.
                     *
                     * THE HOST DOMAIN IS EXEMPT from claiming the plane as it is, and that is not
                     * an oversight. libref reads every weight canonically and declares no hook, so
                     * every op with a relaying device kernel would collide with its own reference
                     * fallback and no model would declare at all. The reference reading a
                     * relayed weight is a real hazard, but it is the one
                     * rad_gate_reference_kernels already refuses to serve on. */
                    RadLayout L{};
                    const int st = ki->layout ? ki->layout(r.geom.params(), r.geom.n_params(),
                                                           operand, &w.enc, w.sel, t, w.n_sel, &L)
                                              : RAD_E_UNSUPPORTED;
                    char enc[256];
                    rad_enc_format(&w.enc, enc, sizeof enc);
                    if (st == RAD_E_DTYPE) {
                        err(st, "%s: kernel %s does not read '%s' (operand %d) encoded %s -- "
                                "convert it to an encoding the kernel reads, or select another "
                                "kernel", o.op.c_str(), ki->name, w.name.c_str(), operand, enc);
                        continue;
                    }
                    if (st < 0 && st != RAD_E_UNSUPPORTED) {
                        err(st, "%s: kernel %s could not describe what it stores for operand %d "
                                "('%s', %s): %s", o.op.c_str(), ki->name, operand, w.name.c_str(),
                            enc, rad_strerror(st));
                        continue;
                    }
                    if (st == RAD_E_UNSUPPORTED && d != RAD_DOMAIN_DEVICE) continue;

                    const std::string tag = st == RAD_OK ? (L.tag && *L.tag ? L.tag : ki->name)
                                                         : std::string();
                    if (!claimant[h].empty() && w.layout_tag != tag) {
                        err(RAD_E_SHAPE,
                            "weight '%s' is wanted in two stored forms: '%s' by %s and '%s' by %s. "
                            "A weight is stored once. Bands of the same op resolving to different "
                            "kernels is the usual cause: every band reads the same stored bytes, "
                            "so a kernel that relayouts has to cover the whole range or none of it",
                            w.name.c_str(),
                            w.layout_tag.empty() ? "(as it is)" : w.layout_tag.c_str(),
                            claimant[h].c_str(), tag.empty() ? "(as it is)" : tag.c_str(),
                            ki->name);
                        continue;
                    }
                    if (!claimant[h].empty()) continue;   /* the same form, asked again */
                    claimant[h] = ki->name;

                    if (tag.empty()) {
                        if (w.n_sel != 1)
                            err(RAD_E_SHAPE, "%s: kernel %s reads '%s' as it is, but the "
                                             "declaration takes %d planes of '%s' as one operand, "
                                             "and only a kernel's layout can join planes",
                                o.op.c_str(), ki->name, w.name.c_str(), w.n_sel,
                                w.source.c_str());
                        continue;   /* the stored defaults decl_weight set stand */
                    }
                    if (w.widen) {
                        err(RAD_E_DTYPE, "'%s' is declared %s over a %s plane and kernel %s "
                                         "relayouts it; a widening and a relayout are not both "
                                         "done at load", w.name.c_str(),
                            rad_dtype_name(w.decl.dtype),
                            rad_dtype_name(w.enc.plane[w.sel[0]].dtype), ki->name);
                        continue;
                    }
                    w.layout_tag   = tag;
                    w.identity     = false;
                    w.lay_op       = (int32_t)i;
                    w.lay_band     = (int32_t)bi;
                    w.lay_dom      = d;
                    w.lay_operand  = operand;
                    w.stored_dtype = L.dtype ? L.dtype : w.decl.dtype;
                    w.stored_rank  = L.rank;
                    for (uint32_t k = 0; k < RAD_MAX_RANK; ++k)
                        w.stored_shape[k] = k < L.rank ? L.shape[k] : 0;
                    if (L.align > 0) w.align = L.align;
                    /* The hook's byte count wins: it includes whatever padding the layout implies,
                     * and the core cannot derive that from a shape. */
                    if (L.bytes > 0) {
                        w.stored_bytes = L.bytes;
                    } else {
                        int64_t nel = 1;
                        for (uint32_t k = 0; k < L.rank && k < RAD_MAX_RANK; ++k) nel *= L.shape[k];
                        const int64_t bs = rad_dtype_bytes(w.stored_dtype, nel);
                        if (bs <= 0) {
                            err(RAD_E_SHAPE, "%s: kernel %s stores '%s' as dtype %u and gave no "
                                             "byte count, which the core cannot size",
                                o.op.c_str(), ki->name, w.name.c_str(), w.stored_dtype);
                            continue;
                        }
                        w.stored_bytes = bs;
                    }
                }
            }
        }
    }

    /* A weight taking several planes as one operand needs a kernel to join them; one no op reads
     * has none. */
    for (size_t h = 1; h < prog_.weights.size(); ++h) {
        const WeightInfo& w = prog_.weights[h];
        if (w.n_sel > 1 && w.identity && claimant[h].empty())
            err(RAD_E_SHAPE, "'%s' takes %d planes of '%s' as one weight and no op reads it, so "
                             "nothing says how they are stored", w.name.c_str(), w.n_sel,
                w.source.c_str());
    }
}

void Builder::finish_kv_groups() {
    for (size_t g = 1; g < prog_.kv_groups.size(); ++g) {
        KVGroupInfo& gi = prog_.kv_groups[g];
        const RadKVGroupDecl& d = gi.decl;

        if ((d.kind == RAD_KV_FULL || d.kind == RAD_KV_WINDOW) && gi.block_size == 0) {
            /* The plugin may never have asked. The manifest still needs the answer, and asking
             * here reports the same diagnostic it would have got. */
            kv_block_size((rad_kvgroup)g);
        }

        int64_t elem_n_head = d.n_head_kv > 0 ? d.n_head_kv : 1;
        /* EVERY LAYER BOUND TO THE GROUP, because that is what a page is. KVManager documents both
         * of these as covering the whole bundle -- "one page covers every layer bound to the group.
         * That is where the zero-padding property lives" -- and its own fallback formula multiplies
         * by the layer count. A figure computed here for ONE layer under-sizes the pool by the
         * layer count, because that fallback is unreachable whenever the builder supplies a value:
         * the manager then divides a one-layer page across every bound layer, giving a per-layer
         * stride that is a fraction of a layer and a layer base pointer that need not even be
         * 2-byte aligned. It surfaces as an alignment refusal from the attention kernel at the
         * SECOND bound layer, layer 0 being at offset 0. */
        const int64_t n_bound = gi.layers.empty() ? 1 : (int64_t)gi.layers.size();
        switch (d.kind) {
            case RAD_KV_FULL:
            case RAD_KV_WINDOW:
                /* One block holds K and V for block_size tokens: the pair is the unit the block
                 * manager allocates, so it is the unit sized here. */
                gi.bytes_per_block =
                    rad_dtype_bytes(d.dtype, 2 * gi.block_size * elem_n_head * d.head_dim) * n_bound;
                gi.bytes_per_state = 0;
                break;
            case RAD_KV_LINEAR:
                /* Per sequence and fixed size: 32 heads of 128x128 fp32 is 2 MiB a layer, which
                 * is why it cannot be snapshotted per block (spec §7.3). */
                gi.bytes_per_block = 0;
                gi.bytes_per_state = rad_dtype_bytes(
                    d.dtype, elem_n_head * (d.state_dim[0] ? d.state_dim[0] : 1)
                                         * (d.state_dim[1] ? d.state_dim[1] : 1)) * n_bound;
                break;
            case RAD_KV_CONV: {
                /* A rolling window of conv_width-1 + n_spec entries, so a speculative rejection
                 * is a change of read offset rather than a recompute (spec §10). The window is
                 * sized for the largest speculative window this deployment was configured for.
                 *
                 * EXACTLY THE STATE TENSOR'S EXTENT, conv_window_entries(), and not one more. The
                 * kernels lay the slots out from the tensor, so an allocation one entry wider a
                 * slot puts the manager's slot `s` -- the bytes it clears when it hands the slot
                 * out, and saves and restores at a checkpoint -- over the start of the kernels'
                 * slot `s + 1`. Handing a slot to a new sequence then zeroes part of a running
                 * sequence's window. conv_state_index's cursor cycles by the same number, so it
                 * stays inside the slot. */
                int64_t entries = conv_window_entries(d, prog_.ctx.max_spec);
                if (entries <= 0) entries = 1;
                gi.bytes_per_block = 0;
                gi.bytes_per_state =
                    rad_dtype_bytes(d.dtype, elem_n_head * d.head_dim * entries) * n_bound;
                break;
            }
            default: break;
        }
        std::sort(gi.layers.begin(), gi.layers.end());

        /* A group nothing is bound to is a group the block manager cannot size: how many state
         * instances a sequence owns is the layer count, and assuming one would make the page size
         * a guess. KVManager::configure refuses it, so it is refused here where the name of the
         * missing call is still obvious. */
        if (gi.layers.empty())
            err(RAD_E_STATE,
                "kv group '%s' has no layers bound: call rad_bind_layer_kv(b, layer, g) for every "
                "layer that owns state in this group, or the block manager cannot tell how many "
                "state instances a sequence holds", gi.name.c_str());
    }
}

void Builder::finish_buffer_liveness() {
    const int32_t last_op = (int32_t)prog_.ops.size() - 1;
    /* A program that runs a second lane gets a second scratch region with it: see
     * Program::scratch_regions. The concurrent buffers are how an architecture says it has one. */
    prog_.scratch_regions = concurrent_bufs_.empty() ? 1 : 2;

    for (size_t i = 1; i < prog_.buffers.size(); ++i) {
        BufferInfo& b = prog_.buffers[i];

        /* PERSIST survives across steps and DERIVED is written once per step by the core before
         * any op runs. Both are live at every op by construction, and saying so here rather than
         * leaving -1 means one rule reads the field and the plan and the dump agree. */
        if (b.decl.kind != RAD_BUF_TRANSIENT) {
            b.first_def = 0;
            b.last_use  = last_op > 0 ? last_op : 0;
            continue;
        }

        /* The second lane: the declared range is not a bound on when the bytes are in use. */
        if (concurrent_bufs_.count((rad_buf)i)) {
            b.first_def = 0;
            b.last_use  = last_op > 0 ? last_op : 0;
            continue;
        }

        int32_t first = -1, last = -1;
        for (size_t o = 1; o < op_bufs_.size(); ++o) {
            auto touches = [&](const std::vector<rad_buf>& v) {
                return std::find(v.begin(), v.end(), (rad_buf)i) != v.end();
            };
            if (!touches(op_bufs_[o].reads) && !touches(op_bufs_[o].writes)) continue;
            if (first < 0) first = (int32_t)o;
            last = (int32_t)o;
        }

        if (first < 0) {
            /* Nothing declared a use. The sound assumption is the whole program: an op may issue
             * any handle obtained during declare, in any order, so declaration position proves
             * nothing about when a buffer is read. It costs sharing, and --debug-graph says how
             * much (see rad_build.h on rad_op_reads/rad_op_writes). */
            b.first_def = 0;
            b.last_use  = last_op > 0 ? last_op : 0;
        } else {
            b.first_def = first;
            b.last_use  = last;
        }

        /* THE LOGITS BUFFER IS READ AFTER THE DECLARED OP LIST AND NOT BY IT. The plugin's lm_head
         * WRITES it and declares that write, so liveness derived from declarations alone ends its
         * life at that op -- and everything that actually consumes it comes later: the sampler's
         * stages (core/sample/sampler.cpp declares its ops with no buffer lists, because its own
         * planes are PERSIST for the same reason), the logprob readback, and the engine.
         *
         * Extending it to the last op is the truthful statement and costs nothing measurable: the
         * lm_head is at the end of the graph, so the set of buffers still live after it is nearly
         * empty and nothing that shared with logits before is prevented from doing so. The
         * undeclared-use guard in core/runtime/issue.cpp is what refuses the untruthful version. */
        if ((rad_buf)i == prog_.logits_buf && last_op > b.last_use) b.last_use = last_op;
    }
}

int Builder::finish() {
    if (finished_) return status_;
    finished_ = true;

    /* A sizing declare wants its buffers and their liveness and nothing else: its kernels are the
     * real declare's, already initialised, and its weights and KV groups are never used. */
    if (ref_) {
        finish_buffer_liveness();
        return status_;
    }

    /* A WEIGHT DECLARED BEFORE ITS NAME MAP was read as plain in its declared dtype, and its
     * kernels were chosen for that: the checkpoint or the recipe behind the builder is searched
     * through the name map, so asking first finds nothing. Asked again now, a source that answers
     * is that ordering, named rather than served as the wrong encoding. */
    if (sources_)
        for (size_t h = 1; h < prog_.weights.size(); ++h) {
            const WeightInfo& w = prog_.weights[h];
            WeightSource ws;
            if (!w.enc_known && lookup_source(w.source, &ws) == RAD_OK)
                err(RAD_E_STATE, "weight '%s' was declared before the name map entry for '%s', so "
                                 "it was read as plain %s and its kernels chosen for that -- "
                                 "declare the map first (rad_weight_encoding)", w.name.c_str(),
                    w.source.c_str(), rad_dtype_name(w.decl.dtype));
        }

    if (instances_) finish_instances();
    finish_weight_uses();
    finish_layouts();
    finish_kv_groups();
    finish_buffer_liveness();

    int s = plan_buffers(prog_, &bufrep_);
    if (s < 0 && status_ == RAD_OK) status_ = s;

    if (!prog_.misses.empty()) {
        /* Reported whole, at the end, never the first failure at the top (spec §1 step 4). Which
         * of these is fatal is the engine's call: a device miss covered by a host kernel costs
         * speed, a host miss costs a placement option, and one covered by neither costs the
         * model.
         *
         * SPLIT BY DOMAIN, because the two counts mean different things and one total means
         * neither. Every caller of this that is not the engine -- rad-check, rad-tune, rad-convert
         * -- reads this line and nothing else, so it has to say which kind it found. */
        /* A device miss whose op resolved on the HOST for the same band is not a hole -- the
         * planner runs it there (engine_bringup.cpp's miss_report says the same thing at more
         * length). Counting it as a device miss makes a deliberately host-only op open every run
         * with a warning that names a hole there is not. */
        std::set<std::pair<std::string, std::string>> no_host;
        for (const Program::Miss& m : prog_.misses)
            if (m.domain == RAD_DOMAIN_HOST) no_host.emplace(m.op, m.span);
        size_t dev = 0;
        for (const Program::Miss& m : prog_.misses)
            if (m.domain != RAD_DOMAIN_HOST && no_host.count(std::make_pair(m.op, m.span))) ++dev;
        /* A host-only miss set is not a warning at all -- see the report in engine_bringup.cpp --
         * so it does not get warned about here either. The device count is the one that means
         * something to a tool reading this line and nothing else. */
        RAD_LOG_AT(dev ? Log::Warn : Log::Debug,
                   "declare: %zu device band(s) and %zu host band(s) resolved to nothing; "
                   "run with --debug-graph for the full list", dev, prog_.misses.size() - dev);
    }
    return status_;
}

}  /* namespace rad */

/* ================================================================== the C surface */
/* Every entry point in rad_builder.h. Each is a static_cast and a forward: the handle IS the
 * builder, so there is no table lookup between a plugin's call and the work. */
extern "C" {

static rad::Builder* B(RadBuilder* b) { return static_cast<rad::Builder*>(b); }

rad_weight rad_decl_weight(RadBuilder* b, const char* name, const RadWeightDecl* d) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_weight(name, d);
}

int rad_weight_encoding(RadBuilder* b, const char* source, RadEncoding* out, int64_t* shape,
                        uint32_t* rank) {
    if (!b) return RAD_E_INVAL;
    return B(b)->weight_encoding(source, out, shape, rank);
}

rad_buf rad_decl_buffer(RadBuilder* b, const char* name, const RadBufDecl* d) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_buffer(name, d);
}

rad_kvgroup rad_decl_kv_group(RadBuilder* b, const char* name, const RadKVGroupDecl* d) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_kv_group(name, d);
}

int64_t rad_kv_block_size(RadBuilder* b, rad_kvgroup g) {
    if (!b) return 0;
    return B(b)->kv_block_size(g);
}

int rad_bind_layer_kv(RadBuilder* b, int layer, rad_kvgroup g) {
    if (!b) return RAD_E_INVAL;
    return B(b)->bind_layer_kv(layer, g);
}

rad_op rad_decl_op(RadBuilder* b, const char* op, const RadParam* params, int n_params,
                   const rad_weight* weights, int n_weights) {
    if (!b) return RAD_NULL_HANDLE;
    return B(b)->decl_op(op, params, n_params, weights, n_weights);
}

int rad_op_resolved(RadBuilder* b, rad_op h) {
    if (!b) return 0;
    return B(b)->op_resolved(h);
}

/* Proposed additions to rad_builder.h -- defined here so the mechanism exists and core-side
 * callers (rad-convert, the tests) can use it. A C plugin cannot until the declarations
 * move into the frozen header. */
int rad_op_reads(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!b || (!bufs && n)) return RAD_E_INVAL;
    return B(b)->op_reads(h, bufs, n);
}

int rad_op_writes(RadBuilder* b, rad_op h, const rad_buf* bufs, int n) {
    if (!b || (!bufs && n)) return RAD_E_INVAL;
    return B(b)->op_writes(h, bufs, n);
}

int rad_buf_concurrent(RadBuilder* b, rad_buf buf) {
    if (!b) return RAD_E_INVAL;
    return B(b)->buf_concurrent(buf);
}

int rad_weight_shard_span(RadBuilder* b, rad_weight w, int64_t lo, int64_t hi) {
    if (!b) return RAD_E_INVAL;
    return B(b)->weight_shard_span(w, lo, hi);
}

int rad_decl_name_map(RadBuilder* b, const RadNameMap* m) {
    if (!b) return RAD_E_INVAL;
    return B(b)->decl_name_map(m);
}

int rad_disable_branched_tree(RadBuilder* b) {
    if (!b) return RAD_E_INVAL;
    B(b)->disable_branched_tree();
    return RAD_OK;
}

int rad_declare_logits(RadBuilder* b, rad_buf logits) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_logits(logits);
}

int rad_declare_drafter(RadBuilder* b, const RadDrafterDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_drafter(d);
}

int rad_declare_drafter_tree(RadBuilder* b, rad_buf tree, int64_t node_capacity) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_drafter_tree(tree, node_capacity);
}

int rad_declare_encoder(RadBuilder* b, const RadEncoderDecl* d) {
    if (!b) return RAD_E_INVAL;
    return B(b)->declare_encoder(d);
}

void rad_note(RadBuilder* b, const char* f, ...) {
    if (!b) return;
    va_list ap;
    va_start(ap, f);
    B(b)->note_v(f, ap);
    va_end(ap);
}

/* ---- model metadata accessors. The free-form key/value the container header carried, which the
 * struct above does not name. The core never interprets these; a plugin reads what it needs. */
static const char* meta_lookup(const RadModelMeta* m, const char* key) {
    if (!m || !key || !m->kv_key || !m->kv_val) return nullptr;
    for (int i = 0; i < m->n_kv; ++i)
        if (m->kv_key[i] && !std::strcmp(m->kv_key[i], key)) return m->kv_val[i];
    return nullptr;
}

long long rad_meta_geti(const RadModelMeta* m, const char* key, long long dflt) {
    const char* v = meta_lookup(m, key);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    long long r = std::strtoll(v, &end, 0);
    return (end && end != v) ? r : dflt;
}

double rad_meta_getf(const RadModelMeta* m, const char* key, double dflt) {
    const char* v = meta_lookup(m, key);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    double r = std::strtod(v, &end);
    return (end && end != v) ? r : dflt;
}

const char* rad_meta_gets(const RadModelMeta* m, const char* key, const char* dflt) {
    const char* v = meta_lookup(m, key);
    return v ? v : dflt;
}

}  /* extern "C" */
