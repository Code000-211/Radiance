/* planner.cpp -- the placement planner. See planner.h for the design; this file is the mechanics.
 *
 * The planner runs once, between declare and load, and it is pure: it reads the Program, the
 * Config and the container's profile, and it writes a Plan. It queries no device, reads no clock
 * and looks at no environment. That is not tidiness -- it is what makes Config::deterministic
 * implementable (spec §17) and what makes the resident set identical on every tensor-parallel
 * rank (planner.h, MoveUnit).
 *
 * DO NOT ADD A DEVICE QUERY HERE, and the reason is concrete rather than principled. Every pool is
 * budgeted explicitly from Config (spec §6) -- no profiling pass, no utilisation fraction. A
 * planner that instead derived a budget from RadDeviceProps would have to cope with rad_dev_count()
 * reporting devices that are not the target: an integrated GPU enumerates alongside the discrete
 * ones and reports tens of GiB of "VRAM" that is really system RAM, so summing capacity across
 * devices overstates the budget by that much and produces a plan that cannot load. Any future code
 * that does look at props must be per-device and must filter on RadDeviceProps::arch, never on
 * index and never on capacity.
 */
#include "planner.h"

#include <algorithm>
#include <cstring>
#include <map>

namespace rad {

/* ------------------------------------------------------------------ names */
/* These live here because placement is the component that owns what a tier and a site MEAN.
 * rad_core.h declares them; if a second definition appears, one of us is wrong about ownership. */
const char* tier_name(Tier t) {
    switch (t) {
        case Tier::VRAM:         return "vram";
        case Tier::HostPinned:   return "pinned";
        case Tier::HostPageable: return "pageable";
        case Tier::SSD:          return "ssd";
        case Tier::Mapped:       return "mapped";
    }
    return "?";
}

const char* table_placement_name(TablePlacement p) {
    switch (p) {
        case TablePlacement::Auto: return "auto";
        case TablePlacement::Vram: return "vram";
        case TablePlacement::Ram:  return "ram";
        case TablePlacement::Disk: return "disk";
    }
    return "?";
}

const char* site_name(Site s) {
    switch (s) {
        case Site::Device:         return "device";
        case Site::DeviceStaged:   return "staged";
        case Site::DeviceZeroCopy: return "zero-copy";
        case Site::Host:           return "host";
    }
    return "?";
}

const char* strategy_name(Strategy s) {
    switch (s) {
        case Strategy::AllVram:      return "all_vram";
        case Strategy::LayerOffload: return "layer_offload";
        case Strategy::ExpertTiered: return "expert_tiered";
        case Strategy::Auto:         return "auto";
    }
    return "?";
}

int strategy_parse(const char* s, Strategy* out) {
    if (!s || !out) return RAD_E_INVAL;
    if (!std::strcmp(s, "all_vram"))      { *out = Strategy::AllVram;      return RAD_OK; }
    if (!std::strcmp(s, "layer_offload")) { *out = Strategy::LayerOffload; return RAD_OK; }
    if (!std::strcmp(s, "expert_tiered")) { *out = Strategy::ExpertTiered; return RAD_OK; }
    if (!std::strcmp(s, "auto"))          { *out = Strategy::Auto;         return RAD_OK; }
    return RAD_E_INVAL;
}

Budgets Budgets::from(const Config& c) {
    Budgets b;
    b.vram_weights = c.vram_weights_mib * 1024 * 1024;
    b.host_pool    = c.host_pool_mib * 1024 * 1024;
    b.ssd          = c.weights_disk_tier;
    return b;
}

/* ------------------------------------------------------------------ sizes and keys */
int64_t weight_bytes(const WeightInfo& w) {
    /* The layout pass's answer when it has run: the stored layout is what actually occupies a
     * tier, and for a kernel that re-layouts its weights it is not the logical size. Before the
     * layout pass -- which is the state a planner unit test builds -- fall back to the declared
     * dtype and shape, which is what logical_bytes is. */
    if (w.stored_bytes > 0) return w.stored_bytes;
    if (w.logical_bytes > 0) return w.logical_bytes;
    int64_t n = w.decl.rank ? 1 : 0;
    for (uint32_t i = 0; i < w.decl.rank; ++i) n *= w.decl.shape[i];
    return rad_dtype_bytes(w.decl.dtype, n);
}


static inline bool is_ple_device(const Config* cfg) {
    const char* env = std::getenv("RADIANCE_PLE_BACKEND");
    if (!env) env = std::getenv("RADIANCE_PLE_STORAGE");
    if (!env) env = std::getenv("RADIANCE_PLE_MODE");
    if (env && *env) {
        if (!std::strcmp(env, "host") || !std::strcmp(env, "cpu") ||
            !std::strcmp(env, "nvme") || !std::strcmp(env, "mmap")) {
            return false;
        }
        if (!std::strcmp(env, "device") || !std::strcmp(env, "gpu") ||
            !std::strcmp(env, "shm") || !std::strcmp(env, "zerocopy") || !std::strcmp(env, "zero-copy")) {
            return true;
        }
    }
    if (cfg && cfg->ple_backend == "host") return false;
    return true;
}

/* See planner.h. Planner::resolve_sites, Planner::build_units and Planner::pin_mapped, run over
 * the same program without allocating a Plan -- the budget resolver calls this before a Planner
 * exists. Kept adjacent to the three so a change to any of them has this in the same diff. */
void weight_footprint(const Program& prog, WeightFootprint* out, TablePlacement embedding,
                      TablePlacement ngram) {
    if (!out) return;
    *out = WeightFootprint{};

    /* resolve_sites: a weight every one of whose ops lacks a device band cannot be read on the
     * card, so it is never copied there whatever the budgets say. */
    std::vector<uint8_t> device_ok(prog.weights.size(), 1u), host_ok(prog.weights.size(), 1u);
    for (const OpInfo& o : prog.ops) {
        bool dev = !o.bands.empty(), host = !o.bands.empty();
        for (const Band& b : o.bands) {
            if (!b.dom[RAD_DOMAIN_DEVICE]) dev = false;
            if (!b.dom[RAD_DOMAIN_HOST]) host = false;
        }
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= device_ok.size()) continue;
            if (!dev) device_ok[w] = 0u;
            if (!host) host_ok[w] = 0u;
        }
    }

    std::vector<uint8_t> zero_copy_mapped(prog.weights.size(), 0u);
    if (is_ple_device(nullptr)) {
        for (const OpInfo& o : prog.ops) {
            if (o.op == "embed_lookup_q") {
                for (rad_weight w : o.weights) {
                    if (w != RAD_NULL_HANDLE && w < prog.weights.size()) {
                        if (prog.weights[w].decl.access == RAD_ACCESS_VOCAB ||
                            prog.weights[w].stored_bytes > (1LL << 30)) {
                            zero_copy_mapped[w] = 1u;
                        }
                    }
                }
            }
        }
    }

    /* build_units: a model-level weight is its own unit, everything else groups by the declared
     * (layer, expert). Ordered, for the same reason build_units is. */
    struct U { int32_t layer = -1, expert = -1; int64_t bytes = 0; bool mapped = false;
               bool emb = false, ngram = false, host = true; };
    std::map<std::pair<int32_t, int32_t>, size_t> by_group;
    std::vector<U> units;

    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w) {
        const WeightInfo& wi = prog.weights[w];
        const int32_t layer = wi.decl.group.layer, expert = wi.decl.group.expert;
        size_t u;
        if (layer < 0) {
            u = units.size();
            units.push_back(U{});
        } else {
            auto key = std::make_pair(layer, expert);
            auto it = by_group.find(key);
            if (it == by_group.end()) {
                u = units.size();
                by_group.emplace(key, u);
                units.push_back(U{});
                units.back().layer = layer;
                units.back().expert = expert;
            } else {
                u = it->second;
            }
        }
        U& mu = units[u];
        const int64_t a = wi.align > 0 ? wi.align : RAD_ALIGN_UNIT;
        mu.bytes = align_up(mu.bytes, a) + weight_bytes(wi);
        if (w < device_ok.size() && !device_ok[w]) mu.mapped = true;
        if (w < host_ok.size() && !host_ok[w]) mu.host = false;
        /* The placement flags' tables: model-level, so the unit is this one weight. */
        if (layer < 0) {
            mu.emb   = is_embedding_table(prog, w);
            mu.ngram = is_ngram_table(prog, w);
        }
    }

    /* And one align_up a unit, because that is how Mover::init spends the slab: `us.static_off =
     * align_up(static_need, RAD_ALIGN_UNIT)` before every unit it places. A budget that summed raw
     * bytes would be short by one alignment per unit, which on a 512-expert MoE is real. */
    std::map<int32_t, int64_t> layer_experts;
    bool ngram_host = true;
    for (const U& mu : units) if (mu.ngram && !mu.host) ngram_host = false;
    for (const U& mu : units) {
        const int64_t b = align_up(mu.bytes, RAD_ALIGN_UNIT);
        /* pin_embedding moves a table out of the statics only where nothing has mapped it. */
        if (mu.emb && !mu.mapped) out->embedding += b;
        if (mu.ngram) out->ngram += b;
        if (mu.ngram && !mu.mapped) out->ngram_device = true;
        /* pin_mapped: an n-gram table a device kernel reads stays off the card unless it was
         * placed in VRAM. */
        if (mu.mapped || (mu.ngram && ngram != TablePlacement::Vram)) out->mapped += b;
        else if (mu.emb && embedding == TablePlacement::Ram) out->hosted += b;
        else if (mu.expert >= 0) { out->experts += b; layer_experts[mu.layer] += b; }
        else                     out->statics += b;
    }
    for (const auto& le : layer_experts)
        out->expert_layer_max = std::max(out->expert_layer_max, le.second);
    out->ngram_host = out->ngram > 0 && ngram_host;
    out->n_units = (int64_t)units.size();
}
std::string weight_class_key(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (size_t i = 0; i < name.size();) {
        if (name[i] >= '0' && name[i] <= '9') {
            out += '*';
            while (i < name.size() && name[i] >= '0' && name[i] <= '9') ++i;
        } else {
            out += name[i++];
        }
    }
    return out;
}

/* ------------------------------------------------------------------ site availability */
/* Host-site execution needs a RAD_DOMAIN_HOST kernel from a kernel plugin, and where none
 * resolved that site is UNAVAILABLE for that op (spec §5.1). The planner does not take that as a
 * flag from somewhere; it reads the Band's per-domain resolution, because the resolution IS the
 * fact and anything else can drift from it.
 *
 * EVERY band, not the band that happens to run. A bucket table exists because the same op serves
 * M=1 at decode and M=4096 at prefill (spec §2.2), and a site that is available for one and not
 * the other is a site that changes under load. Placement is decided once; it has to hold for
 * every step the plan is legal for.
 *
 * BOTH DOMAINS ARE TRACKED, and they answer different questions. Host says whether the host SITE
 * is available -- an option the budgets may or may not take. Device says whether the weight may be
 * placed on a card AT ALL, and a weight whose op has no device kernel anywhere may not: it is
 * pinned to the container's mapping by pin_mapped below, before any budget is consulted.
 *
 * Tracking host alone would rest on the argument that a missing device kernel is fatal at declare
 * and never reaches the planner. That is true only of an op with NO kernel in either domain: an op
 * such as `embed_lookup_q` resolves on the host and nowhere else when no device kernel for it is
 * loaded, and declare reports that device band as unserved without refusing the model. */
void Planner::resolve_sites(const Program& prog) {
    host_ok_.assign(prog.weights.size(), 1u);
    device_ok_.assign(prog.weights.size(), 1u);

    for (const OpInfo& o : prog.ops) {
        bool host = !o.bands.empty();
        bool dev  = !o.bands.empty();
        for (const Band& b : o.bands) {
            if (!b.dom[RAD_DOMAIN_HOST]) host = false;
            if (!b.dom[RAD_DOMAIN_DEVICE]) dev = false;
        }
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= host_ok_.size()) continue;
            if (!host) host_ok_[w] = 0u;
            /* AND THE OTHER DIRECTION, which is not the same statement. The absence of a host
             * kernel removes a placement OPTION; the absence of a DEVICE one removes every option
             * but host, and that is a decision rather than a preference: bytes an op can only be
             * computed over on the CPU must not be copied to a card that cannot read them.
             *
             * Declare's asymmetry still holds -- a band with neither kernel is fatal before the
             * planner runs -- so reaching here with `!dev` means `host` is true. */
            if (!dev) device_ok_[w] = 0u;
        }
    }
}

/* ------------------------------------------------------------------ the mapped tier */
/* A weight no device kernel can read is pinned to the container's mapping BEFORE any budget is
 * consulted, because it is not a budget question -- and so is an n-gram table --ngram-placement put
 * in RAM or on disk, whatever kernels could read it on a card. The other three off-device tiers are
 * what the planner falls back to when VRAM runs out; this one is where the weight has to be
 * whatever the budgets say, and charging a pool for bytes that are never copied would make the
 * operator size a pool against a number that does not exist.
 *
 * THE WHOLE UNIT GOES, not the one weight. A unit is the movement granularity and a half-mapped one
 * has no meaning -- and it costs nothing here, because `device_ok_` is cleared for EVERY weight
 * operand of a host-only op, so the unit's other members are the same op's scale and constants.
 * What this does exclude is a plugin that groups a host-only weight with a device one: that
 * collapses the whole group onto the host, so the report names it. */
void Planner::pin_mapped(const Program& prog, Plan* p) {
    const TablePlacement ng = in_.config->ngram_resolved;
    for (MoveUnit& mu : p->units) {
        bool host_only = false;
        for (rad_weight w : mu.weights)
            if (w < device_ok_.size() && !device_ok_[w]) host_only = true;
        /* --ngram-placement keeps an n-gram table OFF THE CARD unless it says vram, whether or not
         * a device kernel could read it there: ram and disk are where the operator put it, and the
         * strategies below would otherwise spend the card on 48 GB of rows read sixteen a token. */
        const bool ngram = mu.layer < 0 && !mu.weights.empty() &&
                           is_ngram_table(prog, mu.weights.front());
        if (!host_only && !(ngram && ng != TablePlacement::Vram)) continue;

        mu.tier = Tier::Mapped;
        mu.movable = false;
        if (host_only) {
            mu.site = Site::Host;
            mu.reason = "no device kernel for its op; read from the container's mapping";
        } else if (ng == TablePlacement::Ram) {
            mu.site = Site::DeviceZeroCopy;
            mu.reason = "--ngram-placement ram: a pinned copy in RAM the card reads in place, made "
                        "at load";
        } else {
            mu.site = Site::Host;
            mu.reason = "--ngram-placement disk: the host gather reads it from the container";
            for (rad_weight w : mu.weights)
                if (w < host_ok_.size() && !host_ok_[w])
                    note_unfit(p, mu, "--ngram-placement disk needs a host kernel for its gather, "
                                      "and none was loaded");
        }
        /* --ngram-placement ram: the load phase copies an n-gram table out of the mapping and
         * publishes the copy -- locked memory the host kernel reads, or pinned memory the card
         * reads where a device kernel gathers. Every pool still sees the mapped tier; the report
         * shows what the copy costs. */
        if (ngram && ng == TablePlacement::Ram) {
            if (host_only) mu.reason = "--ngram-placement ram: a locked copy in RAM, made at load";
            else           p->ngram_copy_card = true;
            p->ngram_copy_bytes += mu.bytes;
        }
        /* AND ONLY WHAT LIES IN THE CONTAINER AS THE KERNEL READS IT can be read from the mapping,
         * or copied out of it: a weight its kernel rearranges at load, or a shard of one, has no
         * place in the file that already looks the way the kernel reads it (spec §5.1). */
        if (!in_.container)
            note_unfit(p, mu, "it is read from a container's mapping or copied out of one -- and "
                              "a checkpoint served directly has none; convert it to a container");
        else
            for (rad_weight w : mu.weights)
                if (w < prog.weights.size() &&
                    !weight_file_direct(prog.weights[w], prog.ctx.world_size))
                    note_unfit(p, mu, "it is read from the container's mapping or copied out of "
                                      "it -- and its stored form is not the container's (a "
                                      "relayout, a widening or a shard)");
        if (host_only)
            for (rad_weight w : mu.weights)
                if (w < device_ok_.size() && device_ok_[w])
                    RAD_WARN("placement: '%s' is grouped with a weight only a host kernel can "
                             "read, so the whole group stays in the container's mapping",
                             prog.weights[w].name.c_str());
    }
}

/* ------------------------------------------------------------------ units */
int access_strength(int access) {
    switch (access) {
        case RAD_ACCESS_PER_TOKEN:   return 4;
        case RAD_ACCESS_VOCAB:       return 4;
        case RAD_ACCESS_PER_REQUEST: return 2;
        case RAD_ACCESS_CONDITIONAL: return 1;
        case RAD_ACCESS_RARE:        return 0;
        default:                     return 4;   /* an access class we do not know is not one to
                                                  * gamble a demotion on */
    }
}

void Planner::build_units(const Program& prog, Plan* p) {
    p->unit_of_weight.assign(prog.weights.size(), -1);
    p->units.clear();

    /* Keyed on the declared RadWeightGroup, which is the plugin's statement about what moves
     * together. Ordered, not hashed: the unit order has to be a function of the model and not of
     * a hash seed, or two ranks can rank the same profile differently and disagree about which
     * experts are resident. */
    std::map<std::pair<int32_t, int32_t>, int32_t> by_group;

    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w) {
        const WeightInfo& wi = prog.weights[w];
        const int32_t layer = wi.decl.group.layer;
        const int32_t expert = wi.decl.group.expert;

        /* A model-level weight is its own unit. Grouping every model-level weight together would
         * make the embedding table and the output norm one movement unit of wildly different
         * access classes, and the mover would have to carry the larger of them for the smaller. */
        int32_t u;
        if (layer < 0) {
            u = (int32_t)p->units.size();
            p->units.push_back(MoveUnit{});
            p->units.back().layer = -1;
            p->units.back().expert = -1;
        } else {
            auto key = std::make_pair(layer, expert);
            auto it = by_group.find(key);
            if (it == by_group.end()) {
                u = (int32_t)p->units.size();
                by_group.emplace(key, u);
                p->units.push_back(MoveUnit{});
                p->units.back().layer = layer;
                p->units.back().expert = expert;
            } else {
                u = it->second;
            }
        }

        MoveUnit& mu = p->units[(size_t)u];
        const int64_t a = wi.align > 0 ? wi.align : RAD_ALIGN_UNIT;
        const int64_t at = align_up(mu.bytes, a);
        mu.weights.push_back(w);
        mu.offset.push_back(at);
        mu.bytes = at + weight_bytes(wi);
        if (!in_.container || !weight_file_direct(wi, prog.ctx.world_size)) mu.file_direct = false;
        if (wi.first_use_op >= 0 && (mu.first_use_op < 0 || wi.first_use_op < mu.first_use_op))
            mu.first_use_op = wi.first_use_op;
        if (wi.last_use_op > mu.last_use_op) mu.last_use_op = wi.last_use_op;
        if (mu.weights.size() == 1 ||
            access_strength(wi.decl.access) > access_strength(mu.access))
            mu.access = wi.decl.access;
        p->unit_of_weight[w] = u;
    }

    /* Warm start from the profile. It ranks which experts BEGIN resident and nothing else: it is
     * worth about a point of hit rate against an arbitrary order and honestly not more, because
     * imatrix popularity is not decode popularity and the heat engine re-ranks from real routing
     * within a few dispatches regardless. A warm start, not a policy (spec §5.3). */
    if (in_.profile && in_.n_profile > 0) {
        std::map<std::pair<int32_t, int32_t>, float> share;
        for (int64_t i = 0; i < in_.n_profile; ++i)
            share[std::make_pair(in_.profile[i].layer, in_.profile[i].expert)] =
                in_.profile[i].share;
        for (MoveUnit& mu : p->units) {
            auto it = share.find(std::make_pair(mu.layer, mu.expert));
            if (it != share.end()) mu.profile_share = it->second;
        }
    }
}

/* ------------------------------------------------------------------ slab classes */
/* One class per (unit kind, byte size). A slab slot is interchangeable within its class and
 * nowhere else, which is what lets a promotion be an integer swap instead of an allocation, and
 * what lets a slot be LAYER-AGNOSTIC: heat promotes the hottest pooled expert against the coldest
 * resident one wherever they sit, so capacity moves to the layers that need it. A per-layer
 * ranking cannot repair the residency skew a global profile leaves behind, and the slab being
 * layer-agnostic is the whole reason a global ranking is legal.
 */
void Planner::build_classes(Plan* p) {
    std::map<std::pair<int, int64_t>, int32_t> seen;

    for (MoveUnit& mu : p->units) {
        /* A MAPPED UNIT HAS NO CLASS AND NO SLOT. Classes exist so a slab slot is interchangeable
         * within one; a unit that is never copied anywhere has nothing to be interchangeable with,
         * and giving it a stride would charge its whole size to a pool it does not occupy. Its
         * slot_bytes stays 0, which is what every budget below spends against. */
        if (mu.tier == Tier::Mapped) continue;
        const int kind = mu.expert >= 0 ? 2 : (mu.layer >= 0 ? 1 : 0);
        const int64_t stride = align_up(mu.bytes, RAD_ALIGN_UNIT);
        auto key = std::make_pair(kind, stride);
        auto it = seen.find(key);
        if (it == seen.end()) {
            const int32_t c = (int32_t)p->classes.size();
            seen.emplace(key, c);
            SlabClass sc;
            sc.name = kind == 2 ? "expert" : (kind == 1 ? "layer" : "model");
            sc.name += "/" + humanb(stride);
            sc.stride = stride;
            p->classes.push_back(sc);
            mu.slab_class = c;
        } else {
            mu.slab_class = it->second;
        }
        mu.slot_bytes = p->classes[(size_t)mu.slab_class].stride;
        p->classes[(size_t)mu.slab_class].n_units++;
    }
}

/* ------------------------------------------------------------------ the ranking */
/* Rank-invariant by construction. The key is (profile share desc, layer, expert, unit index) and
 * every component of it is a property of the model or of the container -- nothing rank-local,
 * nothing timed, nothing from the device. Under tensor parallel every rank runs this and gets the
 * same order, which is what makes a slab slot mean the same thing on every card. */
void Planner::rank_units(Plan* p, std::vector<int32_t>* order) const {
    order->clear();
    order->reserve(p->units.size());
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) order->push_back(i);
    std::stable_sort(order->begin(), order->end(), [&](int32_t a, int32_t b) {
        const MoveUnit& ua = p->units[(size_t)a];
        const MoveUnit& ub = p->units[(size_t)b];
        if (ua.profile_share != ub.profile_share) return ua.profile_share > ub.profile_share;
        /* EXPERT-MAJOR, NOT LAYER-MAJOR, AND THE TIE-BREAK IS THE WHOLE POLICY WHEN THERE IS NO
         * PROFILE -- which is the common case, because a container without one gives every unit
         * share 0.0 and every comparison above falls through to here.
         *
         * Layer-major fills every expert of layer 0, then layer 1, and stops wherever the budget
         * runs out, so the layers past the cut begin the run with NOTHING resident. The heat
         * engine cannot repair that quickly: a promotion needs the candidate to beat the victim by
         * `hysteresis * cold + min_gain`, and an expert in a starved layer is routed to rarely
         * enough that the skew is worked off a few hundred units at a time, paying both link
         * directions for every one.
         *
         * WHAT THE SKEW COSTS IS ON-DEMAND STREAMING, not mover churn. A starved layer misses on
         * most of what it routes to and streams those shards over the link inside the GEMM, so its
         * expert projections run several times longer than a well-stocked layer's, every step.
         *
         * Expert-major puts expert 0 of every layer before expert 1 of any, so the budget cut
         * lands at the same expert index in every layer and residency starts balanced. That is
         * not a better RANKING -- with no profile there is no information to rank on -- it is the
         * absence of an arbitrary one, and it leaves the heat engine's moves to do the job it
         * exists for instead of repairing the fill. heat.h makes the same argument about its own
         * tie-break: ties go to the layer that can most afford it, never to the low index.
         *
         * NO CAPACITY IS ADDED BY THIS, and the per-layer residency histogram in HeatEngine::report
         * is where that shows: the mean resident count a layer is unchanged and only its spread
         * collapses. Well-stocked layers give up some slots and get slightly slower; starved ones
         * gain more than that back. The largest single beneficiary is whichever layer runs more
         * than once a step -- a draft head runs once per draft round -- because every slot there is
         * spent several times a step.
         *
         * Uniform is still the right fill when there is no profile to rank on, and the heat engine
         * already sees the extra activations of a multiply-run layer in its own counter, so the
         * planner does not weight for it. */
        if (ua.expert != ub.expert) return ua.expert < ub.expert;
        return ua.layer < ub.layer;
    });
}

/* ------------------------------------------------------------------ assignment */
namespace {

/* The three ways a unit can live off the device, in the order the planner tries them. Split out
 * so the expert path and the layer path cannot drift: they are the same mechanism at two
 * granularities and they have to spell the fallback the same way. */
struct HostFall {
    int64_t host_used = 0;
    int64_t host_budget = 0;
    bool    ssd = false;

    /* Returns true and sets tier/site, or false when nothing below VRAM will take it.
     *
     * THE SITE DECIDES THE TIER, which is the independent-axes claim made concrete. Only a unit
     * the DEVICE has to read needs pinned memory; a host-site unit is read by a host kernel and
     * pinning it buys nothing. That matters because pinned memory is genuinely scarce -- past a
     * driver-dependent total the registration call starts returning success and leaving the
     * process broken -- so pinning tens of gigabytes of weights only the CPU touches is not a
     * waste, it is a failure mode. Both kinds are charged against the same --host-pool-mib,
     * because an unbudgeted tier is how a process gets OOM-killed. */
    bool place(MoveUnit& mu, bool host_kernel, const char** reason) {
        if (host_used + mu.slot_bytes <= host_budget) {
            host_used += mu.slot_bytes;
            if (host_kernel && mu.access == RAD_ACCESS_PER_TOKEN) {
                /* llama.cpp -ngl: the bytes stay in host memory and the arithmetic happens there
                 * too, so a weight read every token is never moved at all. It costs the
                 * activation crossing at the run boundary, which is why runs are the objective. */
                mu.tier = Tier::HostPageable;
                mu.site = Site::Host;
                *reason = "host kernel resolved; host run";
                return true;
            }
            /* Everything below is read by the DEVICE out of host memory, so it must be pinned. */
            mu.tier = Tier::HostPinned;
            if (mu.access == RAD_ACCESS_CONDITIONAL) {
                /* THE HYBRID TIER. A routed expert that is not resident is NOT copied into
                 * VRAM and then multiplied: the GEMM's tile names the pinned pool and streams the
                 * shard over the link as it computes, so the transfer is inside the kernel rather
                 * than in front of it, and a miss costs one shard rather than a shard plus a
                 * slab slot plus an eviction. No slot, so it costs no residency either. */
                mu.site = Site::DeviceZeroCopy;
                *reason = "pooled expert; streamed over the link";
            } else {
                /* Read every token with no host kernel: staging into a slab slot is right,
                 * because zero-copy would re-stream the same bytes every single step while a
                 * staged copy pays once and the prefetch schedule hides it. */
                mu.site = Site::DeviceStaged;
                *reason = "no host kernel; staged into a slab slot";
            }
            return true;
        }
        /* THE FILE TIER, for a unit that lies in the container as its kernel reads it.
         *
         * A Tier::SSD unit has NO ADDRESS until something stages it, and the two kinds of unit
         * are staged two ways. A SCHEDULED one is `Plan::prefetch` -- an exact schedule built from
         * first_use_op, because declare enumerated the whole access sequence. A ROUTED one cannot
         * be scheduled: the router picks it on the card, and a grouped MoE GEMM is handed every
         * expert's pointer at issue. So the file stager reads ALL of a routed layer's file-tier
         * experts into a VRAM buffer before the layer's first expert op, on every pass, routed to
         * or not (place/stager.h). Every pointer then exists at issue; what it costs is the whole
         * layer's file-tier bytes a pass, which a prefill chunk -- routing to nearly every expert
         * of a layer -- reads anyway, and which makes a decode step as slow as the drive. That is
         * the trade --weights-disk-tier asks for by name, and the plan sizes the two buffers it
         * needs (Plan::file_stage_vram).
         *
         * Never MOVABLE: the heat engine promotes out of the pinned pool, and a unit with no pool
         * slot has nothing to promote from. assign_expert_tiered says so where it sets the bit. */
        if (ssd && mu.file_direct) {
            mu.tier = Tier::SSD;
            mu.site = Site::DeviceStaged;
            *reason = mu.access == RAD_ACCESS_CONDITIONAL
                          ? "host pool exhausted; read from the container a routed layer at a time"
                          : "host pool exhausted; file tier";
            return true;
        }
        return false;
    }
};

}  /* namespace */

int Planner::assign_all_vram(Plan* p) {
    int64_t used = 0;
    for (MoveUnit& mu : p->units) {
        /* A PREPLACED UNIT IS ALREADY PLACED and no strategy may move it: a mapped one has no
         * device kernel that could read it anywhere else, and a token embedding
         * --embedding-placement put in the host pool is read there in place. Every assignment pass
         * below begins with this line. */
        if (mu.preplaced()) continue;
        mu.tier = Tier::VRAM;
        mu.site = Site::Device;
        mu.movable = false;      /* the degenerate case: the mover never runs */
        used += mu.slot_bytes;
    }
    if (p->vram_budget > 0 && used > p->vram_budget) {
        p->notes.push_back(fmt("all_vram needs %s of weights against a --vram-weights-mib "
                                  "budget of %s: %s short",
                                  humanb(used).c_str(), humanb(p->vram_budget).c_str(),
                                  humanb(used - p->vram_budget).c_str()));
        return RAD_E_NOMEM;
    }
    return RAD_OK;
}

/* LAYER OFFLOAD IS A SCHEDULING PROBLEM, and this is the whole of the decision half: pick which
 * layers leave VRAM. The prefetch that follows is exact, because declare enumerated every op with
 * its weight operands in order and WeightInfo::first_use_op is that order.
 *
 * CONTIGUOUS HOST RUNS ARE AN EXPLICIT OBJECTIVE (spec §5.3). A host-site op leaves its activation
 * in host memory, so every boundary between a device run and a host run costs a link crossing. A
 * few hundred KB is nothing; alternating per layer is 2L of them. llama.cpp gets this free because
 * -ngl assigns a contiguous suffix; here the suffix is chosen on purpose, and for a second reason
 * as well: the FIRST layers are needed earliest in the step, so a staged prefix has to have landed
 * before the step can start at all, while a staged suffix has the whole step's duration to arrive.
 * Spilling the tail is the only choice that gives the mover any lead at all.
 */
int Planner::assign_layer_offload(Plan* p) {
    for (MoveUnit& mu : p->units) {
        if (mu.preplaced()) continue;
        mu.tier = Tier::VRAM; mu.site = Site::Device;
    }

    int64_t total = 0;
    for (const MoveUnit& mu : p->units) if (!mu.preplaced()) total += mu.slot_bytes;
    if (p->vram_budget <= 0 || total <= p->vram_budget) return RAD_OK;

    /* Layers in EXECUTION order. The layer index is normally that order already, but the order
     * declare produced is the authority -- a plugin is free to declare a tower whose layer
     * numbering is not its issue order, and a run computed against the wrong order is not a run. */
    std::vector<int32_t> layers;
    std::map<int32_t, int64_t> bytes_of;
    std::map<int32_t, int32_t> first_of;
    for (const MoveUnit& mu : p->units) {
        if (mu.layer < 0) continue;
        if (!bytes_of.count(mu.layer)) { layers.push_back(mu.layer); first_of[mu.layer] = mu.first_use_op; }
        bytes_of[mu.layer] += mu.slot_bytes;
        if (mu.first_use_op >= 0 && (first_of[mu.layer] < 0 || mu.first_use_op < first_of[mu.layer]))
            first_of[mu.layer] = mu.first_use_op;
    }
    std::stable_sort(layers.begin(), layers.end(), [&](int32_t a, int32_t b) {
        if (first_of[a] != first_of[b]) return first_of[a] < first_of[b];
        return a < b;
    });

    /* RESERVE THE STAGING RING BEFORE CHOOSING THE CUT. A spilled layer with no host kernel is
     * staged, and a staged unit needs a slab slot to be staged into -- so a plan that spills
     * exactly to the budget and only then discovers it has nowhere to stage is a plan that does
     * not load. Deciding this before the cut means asking whether ANY layer lacks a host kernel,
     * which over-reserves when the spill happens to land only on host-capable ones. That is the
     * right direction to be wrong in. */
    /* AND THE RING IS PER CLASS, NOT ONE RING FOR THE WIDEST CLASS.
     *
     * The mover allocates a staging ring in EVERY slab class that has one (Mover::init sums
     * n_slab + shadow + n_stage per class, and finish() charges (n_shadow + n_stage) * stride per
     * class to slab_overhead). A hybrid model has two layer classes -- attention layers and linear
     * layers are different strides -- so reserving one ring at the widest stride is short by a
     * whole ring, and the plan then reports a fit that the mover refuses for want of VRAM. A plan
     * that does not load is worse than one that spills a layer too many, because the second serves.
     *
     * Summed over the classes that actually carry a layer unit, which is the same set the mover
     * builds a slab for. The tail pad is per class too, for the same reason. */
    bool any_stage = ssd_;
    std::map<int32_t, int64_t> stride_of;
    for (const MoveUnit& mu : p->units) {
        if (mu.layer < 0 || mu.slab_class < 0) continue;
        stride_of[mu.slab_class] = p->classes[(size_t)mu.slab_class].stride;
        for (rad_weight w : mu.weights) if (!host_ok_[w]) { any_stage = true; break; }
    }
    int64_t reserve = 0;
    if (any_stage)
        for (const auto& kv : stride_of)
            reserve += (int64_t)RAD_PLACE_STAGE_RING * kv.second + RAD_ALIGN_UNIT;

    int64_t need = total + reserve - p->vram_budget;
    /* Walk the tail. `cut` is the first layer, in execution order, that leaves VRAM -- so the
     * spilled set is a suffix of the execution order and therefore ONE run by construction. */
    size_t cut = layers.size();
    int64_t freed = 0;
    while (cut > 0 && freed < need) { --cut; freed += bytes_of[layers[cut]]; }
    if (freed < need) {
        p->notes.push_back(fmt("layer_offload cannot fit: every layer spilled frees %s against "
                                  "%s needed; the model-level weights alone exceed the budget",
                                  humanb(freed).c_str(), humanb(need).c_str()));
    }

    HostFall fall{0, p->host_budget, ssd_};
    int rc = RAD_OK;
    for (size_t i = cut; i < layers.size(); ++i) {
        for (MoveUnit& mu : p->units) {
            if (mu.layer != layers[i] || mu.preplaced()) continue;
            bool host_kernel = true;
            for (rad_weight w : mu.weights) if (!host_ok_[w]) { host_kernel = false; break; }
            if (!fall.place(mu, host_kernel, &mu.reason)) {
                mu.reason = "fits in no tier";
                note_unfit(p, mu, ssd_ ? "neither VRAM nor the host pool has room"
                                       : "the host pool is full and no --weights-disk-tier was given");
                rc = RAD_E_NOMEM;
            }
        }
    }
    return rc;
}

/* EXPERT OFFLOAD IS A PREDICTION PROBLEM, and this is the initial guess. Everything that is not
 * conditional is pinned to VRAM first: a PER_TOKEN weight is read on every forward pass, so
 * spilling one to buy residency for an expert that MIGHT be routed to is a certain cost against a
 * probabilistic saving. The conditional units then take what is left, ranked by the container's
 * profile, and the heat engine re-ranks from real routing from the first dispatch onward.
 */
int Planner::assign_expert_tiered(Plan* p) {
    int64_t fixed = 0;
    for (MoveUnit& mu : p->units) {
        if (mu.preplaced()) continue;
        mu.tier = Tier::VRAM;
        mu.site = Site::Device;
        if (mu.access != RAD_ACCESS_CONDITIONAL) fixed += mu.slot_bytes;
    }

    if (p->vram_budget > 0 && fixed > p->vram_budget) {
        /* The dense part alone does not fit. That is a layer-offload problem wearing an expert
         * hat, and answering it here with a second copy of the run-picking logic is how two
         * granularities of one mechanism drift apart. Hand it over. */
        RAD_WARN("placement: the non-routed weights alone need %s against a %s weight budget; "
                 "falling through to layer offload for them",
                 humanb(fixed).c_str(), humanb(p->vram_budget).c_str());
        RAD_TRY(assign_layer_offload(p));
        fixed = 0;
        for (const MoveUnit& mu : p->units)
            if (mu.access != RAD_ACCESS_CONDITIONAL && mu.tier == Tier::VRAM) fixed += mu.slot_bytes;
    }
    /* WHAT LAYER OFFLOAD ALREADY PUT IN THE HOST POOL: the dense units of the layers it spilled.
     * The routed experts below are placed again whatever it did with them, but the dense units
     * stay where it put them, so the expert fill spends the pool that is left and not the whole
     * of --host-pool-mib a second time. */
    int64_t host_spent = 0;
    for (const MoveUnit& mu : p->units)
        if (!mu.preplaced() && mu.access != RAD_ACCESS_CONDITIONAL &&
            (mu.tier == Tier::HostPinned || mu.tier == Tier::HostPageable))
            host_spent += mu.slot_bytes;

    std::vector<int32_t> order;
    rank_units(p, &order);

    /* RESERVE THE SPARE SLOTS OUT OF THE BUDGET. The slab has to hold the residents AND the
     * shadow slots a promotion writes into, and they all come out of --vram-weights-mib -- so a
     * fill that spends the whole budget on residents leaves the mover with nowhere to move into
     * and, worse, makes the mover's allocation overrun the pool the operator sized.
     *
     * Two passes, and the second is enough: the reserve depends only on the first pass's slot
     * counts, and the first pass over-counts (it had more room), so the reserve it produces is an
     * upper bound. A third pass can only give slots back, never take more. */
    HostFall fall{host_spent, p->host_budget, ssd_};
    int rc = RAD_OK;

    /* ONE FILL: the conditional units in rank order, at most `cap` of them resident and the rest to
     * the host pool or the file tier; what it costs the VRAM side comes back -- the residents, the
     * shadow slots and the staging ring, and the file stager's two buffers. */
    struct Cost { int64_t used = 0, spare = 0, file = 0, residents = 0; };
    auto fill = [&](int64_t room, int64_t cap) -> Cost {
        Cost k;
        for (int32_t ui : order) {
            MoveUnit& mu = p->units[(size_t)ui];
            if (mu.access != RAD_ACCESS_CONDITIONAL) continue;
            mu.tier = Tier::VRAM;
            mu.site = Site::Device;
            mu.reason = "fits in vram";
        }
        for (SlabClass& sc : p->classes) sc.n_shadow = 0;
        p->unfit.clear();
        fall = HostFall{host_spent, p->host_budget, ssd_};
        rc = RAD_OK;

        for (int32_t ui : order) {
            MoveUnit& mu = p->units[(size_t)ui];
            if (mu.access != RAD_ACCESS_CONDITIONAL) continue;
            if (k.residents < cap && k.used + mu.slot_bytes <= room) {
                k.used += mu.slot_bytes;
                ++k.residents;
                mu.tier = Tier::VRAM;
                mu.site = Site::Device;
                mu.reason = "resident: ranked by the container's profile";
                /* Movable is what the heat engine is allowed to touch. Under
                 * Config::deterministic this is left false everywhere and the resident set is
                 * exactly what this loop decided, for the whole run. */
                mu.movable = p->dynamic;
                continue;
            }
            bool host_kernel = true;
            for (rad_weight w : mu.weights) if (!host_ok_[w]) { host_kernel = false; break; }
            if (!fall.place(mu, host_kernel, &mu.reason)) {
                mu.tier = Tier::SSD;
                mu.site = Site::DeviceStaged;
                mu.reason = "fits in no tier";
                /* With the file tier on, only a unit the container does not hold as its kernel
                 * reads it gets here -- a relayout, a widening or a shard (HostFall::place). */
                note_unfit(p, mu, ssd_ ? "neither VRAM nor the host pool has room, and its "
                                         "stored form is not the container's (a relayout, a "
                                         "widening or a shard), so the file tier cannot read it "
                                         "in place"
                                       : "the host pool is full and no --weights-disk-tier was "
                                         "given");
                rc = RAD_E_NOMEM;
                continue;
            }
            /* A pooled expert is still MOVABLE: it is exactly what the heat engine promotes.
             * Movability is about whether placement may change, not about where it is. A
             * file-tier one is not: the mover promotes out of a pool slot and it has none -- the
             * file stager reads it a layer at a time instead. */
            mu.movable = p->dynamic && mu.tier != Tier::SSD;
        }

        /* How many slots the fill actually used, per class, and therefore how many spares to
         * keep. A static plan reserves none: nothing moves, so nothing needs somewhere to move
         * into. */
        std::vector<int64_t> resident(p->classes.size(), 0), staged(p->classes.size(), 0);
        for (const MoveUnit& mu : p->units) {
            if (mu.slab_class < 0) continue;
            if (mu.movable && mu.tier == Tier::VRAM) ++resident[(size_t)mu.slab_class];
            /* THE STAGING RING IS A SLAB ALLOCATION TOO, so this reserve has to charge for it.
             * finish() sizes it from the units that ended up staged, which is known only after
             * the fill -- the same chicken-and-egg the shadow slots have, and solved the same way.
             *
             * A ring exists only when something falls to the host or file tier, so a model that
             * fits entirely in VRAM never exercises this. On one that spills, leaving the ring
             * uncharged overruns the weight budget by a few slots, and the mover then refuses the
             * statically placed weights on a plan the report called a fit. A ROUTED file-tier
             * unit is not counted: the file stager reads it into buffers of its own. */
            if (mu.site == Site::DeviceStaged && mu.access != RAD_ACCESS_CONDITIONAL)
                ++staged[(size_t)mu.slab_class];
        }
        for (size_t c = 0; c < p->classes.size(); ++c) {
            const int64_t ring = std::min<int64_t>(RAD_PLACE_STAGE_RING, staged[c]);
            const int64_t shadow = (p->dynamic && resident[c] > 0)
                ? std::min<int64_t>(RAD_PLACE_SHADOW_SLOTS, std::max<int64_t>(1, resident[c] / 4))
                : 0;
            p->classes[c].n_shadow = shadow;
            if (shadow + ring <= 0) continue;
            /* Plus the slab's tail padding, which is a real allocation and therefore a real
             * charge: a kernel that assembles a packed code window from a wide read legitimately
             * reads a few bytes past the last slot's data. Inside the slab that lands in a
             * neighbour; at the end it runs off the allocation and faults the card. */
            k.spare += (shadow + ring) * p->classes[c].stride + RAD_ALIGN_UNIT;
        }

        /* THE FILE STAGER'S BUFFERS: two, each the largest routed layer's file-tier units at the
         * stager's packing, and the same tail pad. */
        std::map<int32_t, int64_t> layer_bytes;
        for (const MoveUnit& mu : p->units)
            if (mu.tier == Tier::SSD && mu.access == RAD_ACCESS_CONDITIONAL && mu.layer >= 0)
                layer_bytes[mu.layer] += align_up(mu.bytes, RAD_ALIGN_UNIT);
        p->file_layer_max = 0;
        for (const auto& kv : layer_bytes) p->file_layer_max = std::max(p->file_layer_max, kv.second);
        k.file = p->file_layer_max > 0 ? 2 * p->file_layer_max + RAD_ALIGN_UNIT : 0;
        p->file_stage_vram = k.file;
        return k;
    };

    const int64_t uncapped = INT64_MAX;
    const int64_t room0 = p->vram_budget > 0 ? p->vram_budget - fixed : INT64_MAX;
    Cost k = fill(room0, uncapped);
    if (p->vram_budget > 0) k = fill(room0 - k.spare, uncapped);

    /* THE FILE BUFFERS COME OUT OF THE SAME BUDGET, AND THEY PULL THE OTHER WAY. The two-pass fill
     * above made as many residents as fit beside the shadow slots; a routed layer it then put on
     * the file tier needs two buffers it never paid for. Each resident given up pays for some of
     * them -- and may put one more unit into a layer's file-tier share, growing them again -- so
     * the resident count steps down, in rank order, from what the fill made until the residents,
     * the shadow slots and the buffers fit together. Rank order spreads residency over the layers
     * (rank_units), which is what keeps the largest layer's share -- the buffer -- small. A plan
     * with nothing on the file tier never enters the loop. */
    if (p->vram_budget > 0 && k.file > 0) {
        int64_t cap = k.residents;
        while (fixed + k.used + k.spare + k.file > p->vram_budget && cap > 0)
            k = fill(uncapped, --cap);
        if (fixed + k.used + k.spare + k.file > p->vram_budget) {
            p->notes.push_back(fmt("the file tier's two layer buffers need %s, beside %s of "
                                   "fixed weights, against a --vram-weights-mib budget of %s",
                                   humanb(k.file).c_str(), humanb(fixed).c_str(),
                                   humanb(p->vram_budget).c_str()));
            return RAD_E_NOMEM;
        }
    }
    return rc;
}

void Planner::note_unfit(Plan* p, const MoveUnit& mu, const char* why) const {
    const std::string what = mu.weights.empty()
        ? std::string("<no weights>")
        : weight_class_key(in_.program->weights[mu.weights.front()].name);
    for (Plan::Unfit& u : p->unfit)
        if (u.what == what && u.why == why) { ++u.count; u.bytes += mu.bytes; return; }
    p->unfit.push_back(Plan::Unfit{what, why, 1, mu.bytes});
}

/* ------------------------------------------------------------------ op-level agreement */
/* WeightInfo carries the site, but an OP runs in exactly one domain. Two weight operands of one
 * op that disagree about whether the arithmetic is on the host is not a placement the run phase
 * can execute -- a host kernel cannot read a VRAM pointer and a device kernel cannot read a
 * pageable one. So the planner reconciles, and it reconciles DOWNWARD: any disagreement collapses
 * to the device side, staged, because staging is the site that works from every tier.
 *
 * It is a repair and it is reported. A plan that needs it is usually a plugin that put a
 * per-layer weight in a model-level group, and the note is how that gets noticed. */
void Planner::enforce_op_site_agreement(const Program& prog, Plan* p) {
    for (const OpInfo& o : prog.ops) {
        bool any_host = false, any_dev = false;
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= p->unit_of_weight.size()) continue;
            const int32_t u = p->unit_of_weight[w];
            if (u < 0) continue;
            if (p->units[(size_t)u].site == Site::Host) any_host = true;
            else                                        any_dev = true;
        }
        if (!(any_host && any_dev)) continue;
        for (rad_weight w : o.weights) {
            if (w == RAD_NULL_HANDLE || w >= p->unit_of_weight.size()) continue;
            const int32_t u = p->unit_of_weight[w];
            if (u < 0 || p->units[(size_t)u].site != Site::Host) continue;
            MoveUnit& mu = p->units[(size_t)u];
            /* EXCEPT A MAPPED ONE, which cannot be demoted: staging works from every tier but
             * this one, because no device kernel for the op exists to read the staged copy. The
             * op would be unrunnable either way, so the plan says so instead of quietly producing
             * a placement that cannot serve. A weight reaches here only if the plugin grouped it
             * with a host-only weight or an op reads both kinds -- both are plugin bugs. */
            if (mu.tier == Tier::Mapped) {
                p->notes.push_back(fmt("op '%s' reads a weight only a host kernel can read "
                                       "alongside one placed on the device; no site serves both",
                                       o.op.c_str()));
                RAD_ERR("placement: op '%s' mixes a mapped weight with a device one. The mapped "
                        "weight has no device kernel to be staged for.", o.op.c_str());
                continue;
            }
            mu.site = Site::DeviceStaged;
            mu.reason = "op mixes host and device operands; demoted to staged";
            /* A repair, not a misfit, so it does not go in the plan's did-not-fit list -- a plan
             * that was repaired still fits, and marking it otherwise would make a caller treat a
             * working plan as a failure. It IS visible: the class table shows a `staged` site
             * where a `host` one was intended, and this line says which op caused it. */
            RAD_WARN("placement: op '%s' mixes host-site and device-site weight operands; layer "
                     "%d demoted to staged", o.op.c_str(), (int)mu.layer);
        }
    }
}

/* ------------------------------------------------------------------ prefetch */
/* The exact prefetch schedule. `at_op` is a DEADLINE, not a launch time: declare enumerated the
 * whole access sequence, so the planner knows precisely when each staged unit's bytes are needed
 * and states that. WHEN to start the transfer is the mover's problem, because it is a question
 * about slot supply -- a slot returns to the free list only when its quarantine event signals --
 * and a lead baked in here would be a constant guessing at a queue depth the planner cannot see.
 *
 * No policy and no predictor: a unit appears here exactly once, at the op that first reads it. */
void Planner::schedule_prefetch(Plan* p) {
    p->prefetch.clear();
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) {
        const MoveUnit& mu = p->units[(size_t)i];
        if (mu.site != Site::DeviceStaged) continue;
        if (mu.access == RAD_ACCESS_CONDITIONAL) continue;   /* predicted, not scheduled */
        p->prefetch.push_back(Plan::Prefetch{mu.first_use_op < 0 ? 0 : mu.first_use_op, i});
    }
    std::stable_sort(p->prefetch.begin(), p->prefetch.end(),
                     [](const Plan::Prefetch& a, const Plan::Prefetch& b) {
                         if (a.at_op != b.at_op) return a.at_op < b.at_op;
                         return a.unit < b.unit;
                     });
}

/* ------------------------------------------------------------------ finish */
void Planner::finish(const Program& prog, Plan* p) {
    /* Slab slots. Only a MOVABLE unit needs one: a statically resident weight is allocated once
     * at a fixed offset and never relocates, so giving it a slot would buy nothing and cost the
     * mover a slot it could have used. WeightInfo::slab_slot == -1 means exactly this. */
    std::vector<int64_t> next(p->classes.size(), 0);
    std::vector<int64_t> staged(p->classes.size(), 0);
    for (MoveUnit& mu : p->units) {
        if (mu.movable && mu.tier == Tier::VRAM && mu.slab_class >= 0) {
            mu.slab_slot = (int32_t)next[(size_t)mu.slab_class]++;
        } else {
            mu.slab_slot = -1;
        }
        /* The ring serves the exact prefetch; a routed file-tier unit is read into the file
         * stager's own buffers instead (Plan::file_stage_vram). */
        if (mu.site == Site::DeviceStaged && mu.slab_class >= 0 &&
            mu.access != RAD_ACCESS_CONDITIONAL)
            ++staged[(size_t)mu.slab_class];
    }
    for (size_t c = 0; c < p->classes.size(); ++c) {
        p->classes[c].n_slots = next[c];
        p->classes[c].movable = next[c] > 0;
        /* The staging ring, capped at the number of units that could use it: a class with one
         * staged unit does not need two slots to double-buffer a single copy. */
        p->classes[c].n_stage = std::min<int64_t>(RAD_PLACE_STAGE_RING, staged[c]);
        /* n_shadow was set by the assignment pass, which is the only place that can reserve it
         * out of the budget. Recomputing it here from the resident count would produce a second,
         * differing number, and the mover would allocate against one while the operator was
         * charged for the other. */
        if (!p->classes[c].movable) p->classes[c].n_shadow = 0;
        const int64_t slots = p->classes[c].n_slots + p->classes[c].n_shadow +
                              p->classes[c].n_stage;
        p->slab_overhead += (p->classes[c].n_shadow + p->classes[c].n_stage) * p->classes[c].stride;
        if (slots > 0) p->slab_overhead += RAD_ALIGN_UNIT;   /* the slab's tail pad */
    }

    /* Host runs, from the final assignment rather than from the intent. One entry is one run. */
    std::vector<std::pair<int32_t, int64_t>> host_layers;
    for (const MoveUnit& mu : p->units)
        if (mu.site == Site::Host && mu.layer >= 0) host_layers.push_back({mu.layer, mu.bytes});
    std::sort(host_layers.begin(), host_layers.end());
    for (size_t i = 0; i < host_layers.size();) {
        Plan::HostRun r;
        r.first_layer = host_layers[i].first;
        r.last_layer = host_layers[i].first;
        r.bytes = 0;
        while (i < host_layers.size() &&
               (host_layers[i].first == r.last_layer || host_layers[i].first == r.last_layer + 1)) {
            r.last_layer = host_layers[i].first;
            r.bytes += host_layers[i].second;
            ++i;
        }
        p->host_runs.push_back(r);
    }
    /* Two crossings a run: the activation goes to host at the run's first layer and comes back at
     * its last. This is the number the contiguity objective exists to hold down, so it is
     * computed and printed rather than assumed. */
    p->link_crossings = 2 * (int32_t)p->host_runs.size();

    /* Push the unit's decision down onto every weight, which is what the rest of the engine
     * reads. WeightInfo is frozen and the planner is the component that fills these three. */
    p->weight.assign(prog.weights.size(), Placement{});
    p->totals = Plan::Totals{};
    for (int32_t i = 0; i < (int32_t)p->units.size(); ++i) {
        const MoveUnit& mu = p->units[(size_t)i];
        for (rad_weight w : mu.weights) {
            Placement& pl = p->weight[w];
            pl.tier = mu.tier;
            pl.site = mu.site;
            pl.slab_slot = mu.slab_slot;
            pl.unit = i;
            pl.reason = mu.reason;
            pl.bytes = weight_bytes(prog.weights[w]);
            p->totals.bytes[(int)mu.tier] += pl.bytes;
            if (mu.host_forced) p->embedding_host_used += pl.bytes;
        }
    }
}

/* ------------------------------------------------------------------ the lookup tables */
bool read_only_by(const Program& prog, rad_weight w, const char* op) {
    bool read = false;
    for (const OpInfo& o : prog.ops)
        for (rad_weight ow : o.weights)
            if (ow == w) {
                if (o.op != op) return false;
                read = true;
            }
    return read;
}

/* --embedding-placement ram: THE TOKEN EMBEDDING goes to the pinned host pool before any strategy
 * runs, and stays there. The site is DeviceZeroCopy: the host pool is mapped (Pools::configure),
 * so the lookup kernel is handed the pool's device-side address and reads its M rows of n_embd
 * over the link in place -- a few KiB a decode step -- while the table, the largest model-level
 * weight a dense model has, gives its VRAM to the KV cache. Every strategy and every VRAM total
 * skips the unit through MoveUnit::preplaced(), as they skip the mapped tier, and weight_footprint
 * has already counted it out of the statics with the same test, so the budget the card was split
 * by is the one the planner spends. A layer unit is never one: is_embedding_table asks for a
 * model-level weight, and a layer belongs to the offload strategies. */
int Planner::pin_embedding(const Program& prog, const Config& cfg, Plan* p) const {
    if (cfg.embedding_resolved != TablePlacement::Ram) return RAD_OK;
    int rc = RAD_OK, found = 0;
    for (MoveUnit& mu : p->units) {
        /* build_units gives a model-level weight a unit of its own: it is the unit's one weight. */
        if (mu.layer >= 0 || mu.weights.empty() || mu.tier == Tier::Mapped) continue;
        const rad_weight w = mu.weights.front();
        if (!is_embedding_table(prog, w)) continue;
        ++found;
        const std::string& name = prog.weights[w].name;
        if (mu.slot_bytes > p->host_budget) {
            RAD_ERR("placement: --embedding-placement ram: '%s' needs %s of the pinned host pool "
                    "and %s of --host-pool-mib is left%s", name.c_str(),
                    humanb(mu.slot_bytes).c_str(), humanb(p->host_budget).c_str(),
                    p->file_stage_reserve ? " after the file tier's staging" : "");
            p->notes.push_back(fmt("--embedding-placement ram: '%s' (%s) does not fit the host "
                                   "pool", name.c_str(), humanb(mu.slot_bytes).c_str()));
            rc = RAD_E_NOMEM;
            continue;
        }
        p->host_budget -= mu.slot_bytes;
        p->embedding_host_bytes += mu.slot_bytes;
        mu.tier = Tier::HostPinned;
        mu.site = Site::DeviceZeroCopy;
        mu.movable = false;
        mu.host_forced = true;
        mu.reason = "--embedding-placement ram; read over the link in place";
        RAD_INFO("placement: '%s' (%s) in the pinned host pool, read in place "
                 "(--embedding-placement %s)", name.c_str(), humanb(mu.bytes).c_str(),
                 cfg.embedding_placement == TablePlacement::Auto ? "auto" : "ram");
    }
    /* Engine::plan resolves Ram only where weight_footprint found a table with this same test,
     * so an empty pass is a plan built from a different program than the budget was: a wiring
     * fault, said by name rather than leaving the table on the card the budget split without it. */
    if (!found) {
        RAD_ERR("placement: --embedding-placement ram, and this rank's program has no model-level "
                "weight that only embed_lookup reads");
        p->notes.push_back("--embedding-placement ram: no token embedding table to place");
        rc = RAD_E_STATE;
    }
    return rc;
}

/* ------------------------------------------------------------------ plan */
int Planner::plan(Plan* out) {
    if (!out || !in_.program || !in_.config) return RAD_E_INVAL;
    const Program& prog = *in_.program;
    const Config& cfg = *in_.config;

    *out = Plan{};
    out->rank = in_.rank;
    out->world_size = in_.world_size;
    out->dynamic = !cfg.deterministic;

    const Budgets b = Budgets::from(cfg);
    out->vram_budget = b.vram_weights;
    out->host_budget = b.host_pool;
    ssd_ = b.ssd;

    if (strategy_parse(cfg.placement.c_str(), &out->strategy) < 0) {
        RAD_ERR("placement: unknown strategy '%s'; expected all_vram, layer_offload, "
                "expert_tiered or auto", cfg.placement.c_str());
        return RAD_E_INVAL;
    }

    resolve_sites(prog);
    build_units(prog, out);
    /* BEFORE build_classes and before assign: the mapped tier is not a fallback the budgets
     * choose, it is where a weight no device kernel can read has to be. */
    pin_mapped(prog, out);
    build_classes(out);

    /* CHARGE THE FILE TIER'S STAGING POOL BEFORE PLACING ANYTHING IN THE HOST POOL.
     *
     * A unit read from the container lands in a pinned buffer, and those come out of
     * --host-pool-mib. Handing HostFall the whole budget lets it fill the pool with pinned units,
     * and the mover then asks the exhausted pool for its staging slots and refuses a plan the
     * report has just called a fit. Reserving here is the same rule the VRAM side follows for the
     * slab ring: the component that budgets an allocation has to budget ALL of it.
     *
     * Sized off the widest class that carries a LAYER unit, because that is the only kind either
     * offload strategy spills -- model-level weights stay put in both. It is an upper bound on
     * what the mover will compute from the units that actually land on the file tier, and over-
     * reserving costs one spilled layer where under-reserving costs the whole plan.
     *
     * A model with ROUTED units also gets the read windows a routed layer streams through
     * (RAD_PLACE_FILE_WINDOWS), whether or not this plan ends up putting one there: which units
     * land on the file tier is decided by the fill below, and that fill spends the host budget
     * this line leaves it. */
    if (ssd_) {
        int64_t widest = 0;
        bool routed = false;
        for (const MoveUnit& mu : out->units) {
            if (mu.access == RAD_ACCESS_CONDITIONAL && mu.layer >= 0) routed = true;
            if (mu.layer < 0 || mu.slab_class < 0) continue;
            widest = std::max(widest, out->classes[(size_t)mu.slab_class].stride);
        }
        const int64_t windows = routed ? (int64_t)RAD_PLACE_FILE_WINDOWS *
                                         RAD_PLACE_FILE_WINDOW_BYTES : 0;
        const int64_t need = (int64_t)RAD_PLACE_FILE_STAGE_SLOTS * widest + windows;
        if (need > 0) {
            if (need >= out->host_budget) {
                RAD_ERR("placement: the file tier needs %s of pinned staging (%d slots of %s%s) "
                        "and --host-pool-mib is %s. Raise it, or drop --weights-disk-tier.",
                        humanb(need).c_str(), RAD_PLACE_FILE_STAGE_SLOTS, humanb(widest).c_str(),
                        windows ? fmt(", and %d read windows of %s", RAD_PLACE_FILE_WINDOWS,
                                      humanb(RAD_PLACE_FILE_WINDOW_BYTES).c_str()).c_str() : "",
                        humanb(out->host_budget).c_str());
                return RAD_E_NOMEM;
            }
            out->host_budget -= need;
            out->file_stage_reserve = need;
            out->file_windows = windows;
        }
    }

    /* NOT a RAD_TRY: a refusal still leaves `out` filled in far enough to print. */
    const int hrc = pin_embedding(prog, cfg, out);

    int rc = assign(out);
    enforce_op_site_agreement(prog, out);
    schedule_prefetch(out);
    finish(prog, out);
    return hrc < 0 ? hrc : rc;
}

int Planner::assign(Plan* p) {
    int64_t total = 0;
    bool has_conditional = false;
    for (const MoveUnit& mu : p->units) {
        if (!mu.preplaced()) total += mu.slot_bytes;
        if (mu.access == RAD_ACCESS_CONDITIONAL) has_conditional = true;
    }

    Strategy s = p->strategy;
    if (s == Strategy::Auto) {
        /* `auto` is a choice between three named strategies and nothing more -- it never invents a
         * fourth. It fits if it can, prefers spilling routed experts over dense layers when there
         * are experts to spill (a conditional weight is read only when selected; a dense one is
         * read every token), and offloads layers otherwise. */
        if (p->vram_budget <= 0 || total <= p->vram_budget) s = Strategy::AllVram;
        else if (has_conditional)                           s = Strategy::ExpertTiered;
        else                                                s = Strategy::LayerOffload;
        p->strategy = s;
        RAD_DEBUG("placement: auto chose %s (%s of weights against a %s budget)",
                  strategy_name(s), humanb(total).c_str(), humanb(p->vram_budget).c_str());
    }

    if (s != Strategy::AllVram && p->vram_budget <= 0) {
        /* Every pool is budgeted explicitly (spec §6). An offload strategy with no number to
         * spill against would have to infer one from free VRAM, and an engine that quietly
         * re-optimises its own split is an engine whose benchmarks do not reproduce. */
        RAD_ERR("placement: %s needs --vram-weights-mib; there is nothing to spill against",
                strategy_name(s));
        return RAD_E_INVAL;
    }

    switch (s) {
        case Strategy::AllVram:      return assign_all_vram(p);
        case Strategy::LayerOffload: return assign_layer_offload(p);
        case Strategy::ExpertTiered: return assign_expert_tiered(p);
        default:                     return RAD_E_INVAL;
    }
}

/* ------------------------------------------------------------------ the report (spec §16) */
std::string Plan::report(const Program& prog) const {
    /* One line per weight CLASS. A model with eleven thousand routed experts has eleven thousand
     * weights and three lines worth reading; printing one line each is how a report becomes
     * something nobody opens. */
    struct Row {
        int64_t n = 0, bytes = 0;
        int64_t per_tier[kTierCount] = {0};
        int64_t per_site[4] = {0, 0, 0, 0};
    };
    std::map<std::string, Row> rows;
    std::vector<std::string> order;

    for (rad_weight w = 1; (size_t)w < prog.weights.size() && (size_t)w < weight.size(); ++w) {
        const std::string key = weight_class_key(prog.weights[w].name);
        if (!rows.count(key)) order.push_back(key);
        Row& r = rows[key];
        ++r.n;
        r.bytes += weight[w].bytes;
        r.per_tier[(int)weight[w].tier] += 1;
        r.per_site[(int)weight[w].site] += 1;
    }

    size_t kw = 5;
    for (const std::string& k : order) kw = std::max(kw, k.size());

    std::string s;
    s += fmt("placement plan  rank %d/%d  strategy %s  %s\n\n",
             rank, world_size, strategy_name(strategy),
             dynamic ? "dynamic" : "static (--deterministic)");

    s += fmt("  %-*s %8s %12s  %-34s %s\n", (int)kw, "class", "count", "bytes", "tier", "site");
    for (const std::string& k : order) {
        const Row& r = rows[k];
        std::string tiers, sites;
        for (int t = 0; t < kTierCount; ++t)
            if (r.per_tier[t])
                tiers += fmt("%s%s %lld", tiers.empty() ? "" : " / ", tier_name((Tier)t),
                             (long long)r.per_tier[t]);
        for (int t = 0; t < 4; ++t)
            if (r.per_site[t])
                sites += fmt("%s%s %lld", sites.empty() ? "" : " / ", site_name((Site)t),
                             (long long)r.per_site[t]);
        s += fmt("  %-*s %8lld %12s  %-34s %s\n", (int)kw, k.c_str(), (long long)r.n,
                 humanb(r.bytes).c_str(), tiers.c_str(), sites.c_str());
    }

    s += "\n  pool                budget          used\n";
    s += fmt("  vram weights  %12s  %12s\n",
             vram_budget > 0 ? humanb(vram_budget).c_str() : "uncapped",
             humanb(totals.bytes[(int)Tier::VRAM]).c_str());
    s += fmt("  host pinned   %12s  %12s\n",
             host_budget > 0 ? humanb(host_budget).c_str() : "none",
             humanb(totals.bytes[(int)Tier::HostPinned] - embedding_host_used).c_str());
    if (totals.bytes[(int)Tier::HostPageable])
        s += fmt("  host pageable %12s  %12s\n", "-",
                 humanb(totals.bytes[(int)Tier::HostPageable]).c_str());
    s += fmt("  ssd tier      %12s  %12s\n", "unbounded",
             humanb(totals.bytes[(int)Tier::SSD]).c_str());
    /* NO BUDGET COLUMN, and the dash is the statement: a mapped weight occupies the container's
     * own mapping and no pool at all, so there is nothing for an operator to size. What it does
     * cost is page cache, which is the OS's to manage. */
    if (totals.bytes[(int)Tier::Mapped])
        s += fmt("  mapped        %12s  %12s   (the container's mapping; no pool)\n", "-",
                 humanb(totals.bytes[(int)Tier::Mapped]).c_str());
    /* Taken out of --host-pool-mib before any weight was placed, so the `host pinned` budget above
     * is smaller than the flag by exactly this. Printed rather than folded in: an operator who
     * raises --host-pool-mib and sees less of it reach the weights than expected is owed the
     * reason on the same page. */
    if (file_stage_reserve)
        s += fmt("  file staging  %12s  %12s   (%d slots%s, pinned)\n", "-",
                 humanb(file_stage_reserve).c_str(), RAD_PLACE_FILE_STAGE_SLOTS,
                 file_windows ? fmt(" and %d read windows", RAD_PLACE_FILE_WINDOWS).c_str() : "");
    /* The same, for --embedding-placement ram: out of --host-pool-mib before any strategy, read
     * in place. And the n-gram copy, which no pool holds: locked RAM the load phase fills. */
    if (embedding_host_bytes)
        s += fmt("  embedding     %12s  %12s   (--embedding-placement ram, pinned, read in place)\n",
                 "-", humanb(embedding_host_bytes).c_str());
    if (ngram_copy_bytes)
        s += fmt("  n-gram copy   %12s  %12s   (--ngram-placement ram, %s, filled at load)\n",
                 "-", humanb(ngram_copy_bytes).c_str(),
                 ngram_copy_card ? "pinned RAM the card reads" : "locked RAM");
    /* Out of the VRAM weight budget, beside the residents and not among them: the two buffers a
     * routed layer's file-tier experts are read into, each sized to the layer that puts the most
     * there. */
    if (file_stage_vram)
        s += fmt("  file buffers  %12s  %12s   (two of %s: the most one routed layer reads from "
                 "the container)\n", "-", humanb(file_stage_vram).c_str(),
                 humanb(file_layer_max).c_str());

    if (slab_overhead)
        s += fmt("  slab overhead %12s  %12s   (shadow slots + the staging ring)\n", "-",
                 humanb(slab_overhead).c_str());

    bool any_slab = false;
    for (const SlabClass& c : classes) if (c.movable || c.n_stage) any_slab = true;
    if (any_slab) {
        s += "\n  slab               stride       slots     shadow      stage\n";
        for (const SlabClass& c : classes) {
            if (!c.movable && !c.n_stage) continue;
            s += fmt("  %-16s %10s  %10lld %10lld %10lld\n", c.name.c_str(),
                     humanb(c.stride).c_str(), (long long)c.n_slots, (long long)c.n_shadow,
                     (long long)c.n_stage);
        }
    }

    if (!host_runs.empty()) {
        s += fmt("\n  host runs: %d run%s, %d link crossing%s a step\n", (int)host_runs.size(),
                 host_runs.size() == 1 ? "" : "s", link_crossings,
                 link_crossings == 1 ? "" : "s");
        for (const HostRun& r : host_runs)
            s += fmt("    layers %d-%d  %s\n", (int)r.first_layer, (int)r.last_layer,
                     humanb(r.bytes).c_str());
    }

    if (!prefetch.empty())
        s += fmt("\n  %d staged unit%s with an exact prefetch deadline; the first is op %d\n",
                 (int)prefetch.size(), prefetch.size() == 1 ? "" : "s", (int)prefetch[0].at_op);

    if (!notes.empty() || !unfit.empty()) {
        s += "\n  DID NOT FIT\n";
        for (const std::string& n : notes) s += "    " + n + "\n";
        /* Aggregated, so nine hundred stranded experts are one line. The count and the bytes are
         * what an operator needs to size the next run; the identity of expert 107 of layer 47 is
         * not, and printing it nine hundred times buries the two numbers that are. */
        for (const Unfit& u : unfit)
            s += fmt("    %lld x %s (%s): %s\n", (long long)u.count, u.what.c_str(),
                     humanb(u.bytes).c_str(), u.why.c_str());
    }
    return s;
}

}  /* namespace rad */
