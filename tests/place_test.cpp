/* place_test.cpp -- the placement planner, the mover's slot bookkeeping, and the heat engine.
 *
 * The key test is `heat_does_not_thrash`. Dynamic placement pays off only if it declines to move
 * when moving would not help, since a swap costs two shards over the link where a miss costs one.
 * The test runs the same alternating access pattern with the guards at their shipped values and
 * with them removed, and compares the two.
 */
#include "rad_test.h"

#include "device/device.h"
#include "device/directio.h"
#include "mem/pools.h"
#include "mem/vram_budget.h"
#include "place/heat.h"
#include "place/mover.h"
#include "place/stager.h"
#include "place/planner.h"

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace rad;

/* ================================================================== a synthetic Program */
namespace {

constexpr int64_t MiB = 1024 * 1024;
constexpr int64_t KiB = 1024;

/* Two kernel rows, one per domain. The planner reads Band::dom[] to learn whether a host site
 * exists, so fixtures resolve real rows. */
const RadKernelInfo g_dev_info = { "fake_dev", "gemm", "gemm", "", "", "",
                                   RAD_DOMAIN_DEVICE, 0, nullptr, 0, nullptr, 0,
                                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_host_info = { "fake_host", "gemm", "gemm", "", "", "",
                                    RAD_DOMAIN_HOST, 0, nullptr, 0, nullptr, 0,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
KernelRow g_dev_row;
KernelRow g_host_row;

struct Fixture {
    Program prog;

    Fixture() {
        g_dev_row.info = &g_dev_info;
        g_host_row.info = &g_host_info;
        prog.weights.resize(1);        /* index 0 is the sentinel */
        prog.ctx.rank = 0;
        prog.ctx.world_size = 1;
    }

    rad_weight weight(const std::string& name, int64_t bytes, int access, int32_t layer,
                      int32_t expert, int shard = RAD_SHARD_NONE) {
        WeightInfo w;
        w.name = name;
        w.decl.dtype = RAD_U8;         /* one byte per element, so bytes are exact */
        w.decl.rank = 1;
        w.decl.shape[0] = bytes;
        w.decl.access = access;
        w.decl.shard = shard;
        w.decl.group.layer = layer;
        w.decl.group.expert = expert;
        w.logical_bytes = bytes;
        w.align = RAD_ALIGN_UNIT;
        prog.weights.push_back(w);
        return (rad_weight)(prog.weights.size() - 1);
    }

    void op(const std::string& name, const std::vector<rad_weight>& ws, bool host_kernel) {
        OpInfo o;
        o.op = name;
        o.index = (int32_t)prog.ops.size();
        o.weights = ws;
        Band b;
        b.hi = 1 << 20;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_dev_row;
        if (host_kernel) b.dom[RAD_DOMAIN_HOST].row = &g_host_row;
        else             b.miss[RAD_DOMAIN_HOST] = "no host kernel for this geometry";
        o.bands.push_back(b);
        for (rad_weight w : ws) {
            WeightInfo& wi = prog.weights[w];
            if (wi.first_use_op < 0) wi.first_use_op = o.index;
            wi.last_use_op = o.index;
        }
        prog.ops.push_back(o);
    }

    /* An op only a host kernel serves, its device band a miss. Unlike op(), it leaves the
     * weights' first and last use as they were. */
    void host_op(const std::string& name, const std::vector<rad_weight>& ws) {
        OpInfo o;
        o.op = name;
        o.index = (int32_t)prog.ops.size();
        o.weights = ws;
        Band b;
        b.hi = 1 << 20;
        b.dom[RAD_DOMAIN_HOST].row = &g_host_row;
        b.miss[RAD_DOMAIN_DEVICE] = "no device kernel";
        o.bands.push_back(b);
        prog.ops.push_back(o);
    }
};

Config base_config() {
    Config c;
    c.placement = "auto";
    c.vram_weights_mib = 0;
    c.host_pool_mib = 0;
    return c;
}

/* A dense model: `layers` layers of one weight each, plus one model-level weight. */
void build_dense(Fixture& f, int layers, int64_t per_layer, bool host_kernel) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, host_kernel);
    for (int l = 0; l < layers; ++l) {
        const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_down.weight", per_layer,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        f.op("gemm", {w}, host_kernel);
    }
}

/* One MoE layer: `n` experts of `each` bytes, plus one dense model-level weight. */
void build_moe(Fixture& f, int n, int64_t each, std::vector<rad_weight>* experts) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int e = 0; e < n; ++e) {
        const rad_weight w = f.weight("blk.0.ffn_gate_exps." + std::to_string(e) + ".weight", each,
                                      RAD_ACCESS_CONDITIONAL, 0, e);
        f.op("moe_gemm", {w}, false);
        if (experts) experts->push_back(w);
    }
}

/* `layers` routed layers of `n` experts, each a unit of two weights (gate_up of `gu` bytes, down
 * of `dn`) read by two grouped ops per layer, as a routed block declares them: a unit whose
 * weights are not contiguous in memory as they are in a container. */
void build_moe_layers(Fixture& f, int layers, int n, int64_t gu, int64_t dn) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < layers; ++l) {
        std::vector<rad_weight> g, d;
        for (int e = 0; e < n; ++e) {
            const std::string b = "blk." + std::to_string(l) + ".";
            g.push_back(f.weight(b + "ffn_gate_up_exps." + std::to_string(e) + ".weight", gu,
                                 RAD_ACCESS_CONDITIONAL, l, e));
            d.push_back(f.weight(b + "ffn_down_exps." + std::to_string(e) + ".weight", dn,
                                 RAD_ACCESS_CONDITIONAL, l, e));
        }
        f.op("moe_gemm", g, false);
        f.op("moe_gemm", d, false);
    }
}

/* Every placement decision as one string, so any two differing plans differ here. */
std::string digest(const Program& prog, const Plan& p) {
    std::string s = fmt("strategy=%s dynamic=%d\n", strategy_name(p.strategy), (int)p.dynamic);
    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w)
        s += fmt("%s tier=%s site=%s slot=%d\n", prog.weights[w].name.c_str(),
                 tier_name(p.weight[w].tier), site_name(p.weight[w].site),
                 (int)p.weight[w].slab_slot);
    return s;
}

int count_tier(const Program& prog, const Plan& p, Tier t) {
    int n = 0;
    for (rad_weight w = 1; (size_t)w < prog.weights.size(); ++w)
        if (p.weight[w].tier == t) ++n;
    return n;
}

}  /* namespace */

/* ================================================================== the planner */

TEST(planner_fits_an_all_vram_model) {
    Fixture f;
    build_dense(f, 4, 1 * MiB, false);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 5 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], (int64_t)0);
    CHECK_EQ(count_tier(f.prog, p, Tier::VRAM), 5);
    CHECK(p.fits());
    CHECK(p.host_runs.empty());
    CHECK_EQ(p.link_crossings, 0);
    /* Everything on tier 0 and nothing for the mover to do. */
    for (const MoveUnit& mu : p.units) CHECK(!mu.movable);
}

TEST(planner_reports_what_did_not_fit) {
    Fixture f;
    build_dense(f, 4, 1 * MiB, false);

    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 2;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_NOMEM);
    CHECK(!p.fits());
    /* The report must say what did not fit; the status code alone does not. */
    CHECK(p.report(f.prog).find("DID NOT FIT") != std::string::npos);
}

TEST(planner_host_runs_are_contiguous) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/true);

    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;      /* 9 MiB of weights: four layers must leave */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* One contiguous run: alternating per layer would cost eight link crossings per step for the
     * same bytes, a run costs two. */
    CHECK_EQ((int)p.host_runs.size(), 1);
    CHECK_EQ(p.link_crossings, 2);
    if (!p.host_runs.empty()) {
        CHECK_EQ(p.host_runs[0].first_layer, 4);
        CHECK_EQ(p.host_runs[0].last_layer, 7);
    }
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 5 * MiB);
    /* Pageable, not pinned: the host site decides the tier. A host kernel reads a host-site
     * layer, so pinning buys nothing, and pinned memory is scarce (past a driver-dependent total,
     * registration succeeds but leaves the process broken). */
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPageable], 4 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], (int64_t)0);

    /* The run is a suffix, so the device keeps the early layers, which a staged prefix would have
     * to land before the step could start. */
    for (const MoveUnit& mu : p.units) {
        if (mu.layer < 0) continue;
        if (mu.layer < 4) { CHECK(mu.site == Site::Device); CHECK(mu.tier == Tier::VRAM); }
        else              { CHECK(mu.site == Site::Host);   CHECK(mu.tier == Tier::HostPageable); }
    }
}

TEST(planner_stages_when_no_host_kernel_resolved) {
    /* As above but with no RAD_DOMAIN_HOST kernel resolved, so the host site is unavailable and
     * the spilled bytes get a different site: tier and site are independent. */
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/false);

    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    CHECK(p.host_runs.empty());
    CHECK_EQ(p.link_crossings, 0);
    /* Two more layers spill than in the host-kernel case, for the staging ring: a staged unit
     * needs a slab slot, so the planner reserves the ring from the VRAM budget before choosing
     * the cut. Otherwise the plan would fill the budget and have nowhere to stage. */
    CHECK_EQ(p.slab_overhead, (int64_t)RAD_PLACE_STAGE_RING * MiB + RAD_ALIGN_UNIT);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], 7 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 2 * MiB);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);
    for (const MoveUnit& mu : p.units)
        if (mu.layer >= 1) CHECK(mu.site == Site::DeviceStaged);

    /* Every staged unit has an exact deadline from the declared op order, and the prefetch
     * schedule is sorted by it. */
    CHECK_EQ((int)p.prefetch.size(), 7);
    for (size_t i = 1; i < p.prefetch.size(); ++i)
        CHECK(p.prefetch[i - 1].at_op <= p.prefetch[i].at_op);
    CHECK(p.prefetch[0].at_op > 0);
}

TEST(planner_shards_under_world_size_2) {
    /* The plugin declares dimensions already divided (spec §9), so a rank's unit is its own
     * shard. Invariant: the resident set is identical on every rank, since a slab slot means the
     * same everywhere only if every rank chose the same experts for it. */
    Fixture whole, half0, half1;
    std::vector<rad_weight> e_whole, e_half;
    build_moe(whole, 8, 4 * MiB, &e_whole);
    build_moe(half0, 8, 2 * MiB, &e_half);
    build_moe(half1, 8, 2 * MiB, nullptr);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 1024;
    c.host_pool_mib = 1024;

    Plan pw, p0, p1;
    { PlannerInput in{&whole.prog, &c, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&pw)); }
    { PlannerInput in{&half0.prog, &c, 0, 2, nullptr, 0}; CHECK_OK(Planner(in).plan(&p0)); }
    { PlannerInput in{&half1.prog, &c, 1, 2, nullptr, 0}; CHECK_OK(Planner(in).plan(&p1)); }

    /* Each rank plans against its own card with half the expert bytes; the embedding is
     * replicated, so this is not simply half of everything. */
    CHECK_EQ(pw.totals.bytes[(int)Tier::VRAM], 33 * MiB);
    CHECK_EQ(p0.totals.bytes[(int)Tier::VRAM], 17 * MiB);

    /* Same decisions on both ranks, weight for weight; anything rank-local in the ranking (a
     * device query, a clock, a hash seed) fails this. */
    CHECK_EQ(digest(half0.prog, p0), digest(half1.prog, p1));
    CHECK_EQ(p0.rank, 0);
    CHECK_EQ(p1.rank, 1);
}

TEST(planner_spills_experts_before_dense_weights) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;      /* 1 MiB embedding; the rest is the expert slab */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* Three resident, not four: the fourth slot is the shadow. Spare slots come from the weights
     * pool (--gpu-headroom-mib is memory the engine never touches), so dynamic placement costs a
     * resident unit and the report says how many. */
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 3 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], 6 * MiB);
    CHECK_EQ(p.slab_overhead, 1 * MiB + RAD_ALIGN_UNIT);
    /* The total fits the plan's budget, so the mover cannot overrun the pool sized for it. */
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);

    /* The embedding is read every token and never leaves. */
    CHECK(p.weight[1].tier == Tier::VRAM);
    /* A pooled expert is zero-copy, not staged: the GEMM streams it over the link, so a miss
     * costs one shard and no slab slot. */
    int zero_copy = 0;
    for (const MoveUnit& mu : p.units)
        if (mu.tier == Tier::HostPinned && mu.site == Site::DeviceZeroCopy) ++zero_copy;
    CHECK_EQ(zero_copy, 6);
}

/* A re-split rank's Config reaches both readers of the pool sizes. Engine::plan gives a rank whose
 * cache costs less per token its own Config (vram_budget_rank_split); Pools::configure sizes the
 * weights pool and Budgets::from the slab from the same copy, or the pool is wasted or overrun.
 * plan_check_weights_pool checks this at startup. */
TEST(a_resplit_ranks_config_reaches_the_pools_and_the_planner) {
    Fixture f;
    build_moe(f, 8, 1 * MiB, nullptr);

    Config common = base_config();
    common.placement = "expert_tiered";
    common.vram_weights_mib = 5;      /* 1 MiB embedding, three experts and a shadow */
    common.vram_kv_mib = 8;
    common.host_pool_mib = 64;
    const RankSplit rs = vram_budget_rank_split(common.vram_weights_mib, common.vram_kv_mib, 2,
                                                64 * MiB);
    CHECK(rs.resplit);
    Config rc = common;
    rc.vram_weights_mib = rs.weights_mib;
    rc.vram_kv_mib = rs.kv_mib;

    Pools pools;
    CHECK_OK(pools.configure(rc, 0, &rad_malloc_allocator(), &rad_malloc_allocator()));
    CHECK_EQ(pools.weights().bytes(), rs.weights_mib * MiB);
    CHECK_EQ(pools.kv().bytes(), rs.kv_mib * MiB);
    CHECK_EQ(Budgets::from(rc).vram_weights, pools.weights().bytes());

    Plan grown, flat;
    { PlannerInput in{&f.prog, &rc, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&grown)); }
    { PlannerInput in{&f.prog, &common, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&flat)); }
    CHECK_EQ(grown.vram_budget, pools.weights().bytes());
    CHECK_EQ(plan_check_weights_pool(grown, pools.weights().bytes()), RAD_OK);
    /* The grown pool is used: more experts are on the card. */
    CHECK(grown.totals.bytes[(int)Tier::VRAM] > flat.totals.bytes[(int)Tier::VRAM]);
    CHECK(grown.totals.bytes[(int)Tier::VRAM] + grown.slab_overhead <= pools.weights().bytes());
    /* A plan from the common split against the rank's grown pool is refused. */
    CHECK_EQ(plan_check_weights_pool(flat, pools.weights().bytes()), RAD_E_STATE);
}

/* expert_vram_floor places every expert. A lending cache leaves the expert slab exactly the floor,
 * so it must cover the residents, the slab's spare slots, and the slack each first-fit pass can
 * strand, or some experts get no tier and the server refuses to start. Two strides give two slab
 * classes with their own shadow slots, as in a mixed-precision container. */
TEST(the_expert_floor_places_every_expert) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 8; ++l) {
        const int64_t gu = l < 4 ? 600 * KiB : 1 * MiB, dn = l < 4 ? 300 * KiB : 500 * KiB;
        std::vector<rad_weight> g, d;
        for (int e = 0; e < 16; ++e) {
            const std::string b = "blk." + std::to_string(l) + ".";
            g.push_back(f.weight(b + "ffn_gate_up_exps." + std::to_string(e) + ".weight", gu,
                                 RAD_ACCESS_CONDITIONAL, l, e));
            d.push_back(f.weight(b + "ffn_down_exps." + std::to_string(e) + ".weight", dn,
                                 RAD_ACCESS_CONDITIONAL, l, e));
        }
        f.op("moe_gemm", g, false);
        f.op("moe_gemm", d, false);
    }
    WeightFootprint wf;
    weight_footprint(f.prog, &wf, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK(wf.expert_spare > 0);

    const int64_t host = 100 * MiB;
    auto plan_with = [&](int64_t floor, Plan* p) {
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = (wf.statics + floor + MiB - 1) / MiB;
        c.host_pool_mib = host / MiB;
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        return Planner(in).plan(p);
    };

    const int64_t floor = expert_vram_floor(wf, host);
    CHECK(floor > 0 && floor < wf.experts);
    Plan p;
    CHECK_OK(plan_with(floor, &p));
    CHECK(p.unfit.empty());
    CHECK_EQ(count_tier(f.prog, p, Tier::SSD), 0);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);

    /* A floor counting only the experts the pool cannot hold plus one unit of slack strands
     * some, because of the spare slots. */
    Plan q;
    CHECK(plan_with(wf.experts - host + wf.expert_unit_max, &q) == RAD_E_NOMEM);
    CHECK(!q.unfit.empty());
}

TEST(planner_warm_starts_from_the_container_profile) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    /* Experts 5, 6 and 7 are popular in the calibration profile. The profile is only a warm
     * start: it decides which experts begin resident. */
    std::vector<RadFileProfile> prof;
    for (int e = 0; e < 8; ++e) prof.push_back(RadFileProfile{0, e, e >= 5 ? 0.3f : 0.01f, 0.0f});

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 6;      /* 1 MiB embedding + three experts + one shadow slot */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        if (mu.expert >= 5) CHECK(mu.tier == Tier::VRAM);
        else                CHECK(mu.tier == Tier::HostPinned);
    }
}

TEST(planner_refuses_an_offload_strategy_with_no_budget) {
    /* Every pool is budgeted explicitly (spec §6). An offload strategy without a budget would
     * have to infer one, making benchmarks irreproducible. */
    Fixture f;
    build_dense(f, 4, 1 * MiB, true);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 0;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_EQ(Planner(in).plan(&p), RAD_E_INVAL);
}

TEST(deterministic_mode_pins_placement_to_static) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    c.deterministic = true;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan a, b;
    CHECK_OK(Planner(in).plan(&a));
    CHECK_OK(Planner(in).plan(&b));

    /* Identical plans across runs, and nothing movable, so the heat engine stays idle and
     * residency (hence summation order) is fixed for the run (spec §17). */
    CHECK_EQ(digest(f.prog, a), digest(f.prog, b));
    CHECK(!a.dynamic);
    for (const MoveUnit& mu : a.units) CHECK(!mu.movable);

    HeatEngine h;
    CHECK_OK(h.init(a, HeatParams{}));
    CHECK(!h.enabled());
    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK(sw.empty());

    /* Control: with determinism off the same fixture is movable. */
    c.deterministic = false;
    PlannerInput in2{&f.prog, &c, 0, 1, nullptr, 0};
    Plan d;
    CHECK_OK(Planner(in2).plan(&d));
    CHECK(d.dynamic);
    int movable = 0;
    for (const MoveUnit& mu : d.units) if (mu.movable) ++movable;
    CHECK_EQ(movable, 8);
}

TEST(planner_reads_routed_experts_it_cannot_hold_from_the_container) {
    /* Two routed layers of eight 1.5 MiB experts, a VRAM budget for a few and a host pool for
     * two. With the file tier on, the rest go there and the plan still fits: they are served by
     * reading the whole layer into one of two buffers charged to the same VRAM budget. */
    Fixture f;
    build_moe_layers(f, 2, 8, 1 * MiB, MiB / 2);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 24;
    /* The file staging reserve (four slots of the widest unit plus the read windows) and room
     * for two pooled experts. */
    const int64_t reserve = 4 * (MiB + MiB / 2) + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES;
    c.host_pool_mib = (reserve + 3 * MiB) / MiB;
    c.weights_disk_tier = true;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK(p.fits());
    CHECK_EQ(p.file_windows, (int64_t)RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES);

    std::vector<int64_t> per_layer(2, 0);
    int ssd = 0, pooled = 0, resident = 0;
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        if (mu.tier == Tier::SSD) {
            ++ssd;
            /* Staged, never movable: the heat engine promotes from a pool slot, which a file-tier
             * unit lacks. */
            CHECK(mu.site == Site::DeviceStaged);
            CHECK(!mu.movable);
            per_layer[(size_t)mu.layer] += align_up(mu.bytes, RAD_ALIGN_UNIT);
        } else if (mu.tier == Tier::HostPinned) {
            ++pooled;
        } else if (mu.tier == Tier::VRAM) {
            ++resident;
        }
    }
    CHECK(ssd > 0);
    CHECK_EQ(pooled, 2);
    CHECK(resident > 0);
    CHECK_EQ(resident + pooled + ssd, 16);
    /* Two buffers, each sized for the largest layer's file-tier units at the stager's packing. */
    CHECK_EQ(p.file_layer_max, std::max(per_layer[0], per_layer[1]));
    CHECK(p.file_stage_vram >= 2 * p.file_layer_max + (int64_t)RAD_ALIGN_UNIT);
    /* Residents, shadow slots and the two buffers fit the VRAM budget, so the mover cannot
     * overrun the pool. */
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead + p.file_stage_vram <= p.vram_budget);
    /* The prefetch ring is unused for these; the file stager has its own buffers. */
    for (const SlabClass& sc : p.classes) CHECK_EQ(sc.n_stage, (int64_t)0);

    /* Without the tier the same budgets strand them, and the refusal names the flag. */
    c.weights_disk_tier = false;
    c.host_pool_mib = 3;
    PlannerInput in2{&f.prog, &c, 0, 1, nullptr, 0};
    Plan q;
    CHECK_EQ(Planner(in2).plan(&q), RAD_E_NOMEM);
    CHECK(!q.unfit.empty());
    bool named = false;
    for (const Plan::Unfit& u : q.unfit)
        if (u.why.find("--weights-disk-tier") != std::string::npos) named = true;
    CHECK(named);
}

TEST(weight_class_key_collapses_indices) {
    CHECK_EQ(weight_class_key("blk.12.ffn_gate_exps.3.weight"), std::string("blk.*.ffn_gate_exps.*.weight"));
    CHECK_EQ(weight_class_key("token_embd.weight"), std::string("token_embd.weight"));
}

/* ================================================================== the heat engine */
namespace {

/* An eight-expert MoE with four resident, ready for the heat engine. */
struct HeatFixture {
    Fixture f;
    Plan    plan;
    Config  cfg;

    explicit HeatFixture(int n_expert = 8, int64_t vram_mib = 7) {
        std::vector<rad_weight> ex;
        build_moe(f, n_expert, 1 * MiB, &ex);
        cfg = base_config();
        cfg.placement = "expert_tiered";
        cfg.vram_weights_mib = vram_mib;
        cfg.host_pool_mib = 64;
        PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
        Planner(in).plan(&plan);
    }

    int32_t unit_of_expert(int32_t e) const {
        for (int32_t i = 0; i < (int32_t)plan.units.size(); ++i)
            if (plan.units[(size_t)i].expert == e) return i;
        return -1;
    }
};

}  /* namespace */

TEST(heat_promotes_a_hot_pooled_expert) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    CHECK_OK(h.init(hf.plan, hp));
    CHECK(h.enabled());

    /* Expert 7 is pooled (no profile, so the first four win on unit order). Ten tokens route to
     * it, and the credit is that count, not a flat one: an expert chosen by several tokens is
     * likelier to be chosen again. */
    const int32_t hot = hf.unit_of_expert(7);
    CHECK(hot >= 0);
    CHECK(h.tier(hot) != Tier::VRAM);

    const int32_t ids[1] = {7};
    const int32_t counts[1] = {10};
    h.observe_ids(0, ids, counts, 1);

    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK(!sw.empty());
    if (!sw.empty()) {
        CHECK_EQ(sw[0].promote, hot);
        CHECK(h.tier(sw[0].demote) == Tier::VRAM);
        /* Promotion and demotion come as a pair; promoting alone would overfill the slab. */
        CHECK(sw[0].promote != sw[0].demote);
    }
}

TEST(heat_does_not_thrash_on_an_alternating_pattern) {
    /* The four resident and four pooled experts are selected in strict alternation, so both
     * halves are of equal value and their heats converge within one decay step. A swap here moves
     * two shards over the link for no gain, on the link the zero-copy tier uses.
     *
     * The guards (hysteresis, min_gain) are the only defence; there is no admission filter. The
     * same pattern with guards removed is the control, since an engine that never proposes would
     * also score zero. */
    auto run = [](HeatParams hp) {
        HeatFixture hf;
        HeatEngine h;
        h.init(hf.plan, hp);
        CHECK(h.tier(hf.unit_of_expert(0)) == Tier::VRAM);
        CHECK(h.tier(hf.unit_of_expert(7)) != Tier::VRAM);

        const int32_t resident[4] = {0, 1, 2, 3};
        const int32_t pooled[4] = {4, 5, 6, 7};
        const int32_t ones[4] = {1, 1, 1, 1};

        uint64_t proposed = 0;
        for (int step = 0; step < 200; ++step) {
            h.observe_ids(0, (step & 1) ? pooled : resident, ones, 4);
            std::vector<Swap> sw;
            h.step(&sw);
            /* Nothing is issued, so release the units; otherwise busy flags, not policy, would
             * silence the engine. */
            for (const Swap& s : sw) h.abandon(s);
            proposed += sw.size();
        }
        return proposed;
    };

    HeatParams shipped;                    /* hysteresis 1.25, min_gain 1.5 */
    const uint64_t guarded = run(shipped);

    HeatParams none = shipped;
    none.hysteresis = 1.0f;
    none.min_gain = 0.0f;
    const uint64_t unguarded = run(none);

    CHECK_EQ(guarded, (uint64_t)0);
    /* Control: without guards the pattern churns whenever the pooled half was credited last. */
    CHECK(unguarded > 100);
}

/* A prefill chunk is credited as one capped histogram. Ten times the cap's selections credits a
 * tenth of each count, keeping the experts' order while one chunk cannot outweigh the decode steps
 * around it; a histogram under the cap is credited in full. */
TEST(heat_caps_what_one_histogram_credits) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    hp.credit_cap = 512.0f;
    CHECK_OK(h.init(hf.plan, hp));
    const int32_t a = hf.unit_of_expert(4), b = hf.unit_of_expert(5), c = hf.unit_of_expert(6);

    const int32_t big_ids[2] = {4, 5};
    const int32_t big[2] = {4000, 1120};
    h.observe_ids(0, big_ids, big, 2);
    CHECK(std::fabs(h.heat(a) - 400.0f) < 1e-3f);
    CHECK(std::fabs(h.heat(b) - 112.0f) < 1e-3f);

    const int32_t small_ids[1] = {6};
    const int32_t small[1] = {12};
    h.observe_ids(0, small_ids, small, 1);
    CHECK_EQ(h.heat(c), 12.0f);
    CHECK(std::fabs(h.heat(a) - 400.0f * hp.decay) < 1e-3f);
}

TEST(heat_respects_the_move_budget) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    hp.moves_per_step = 3;
    hp.min_gain = 0.5f;
    CHECK_OK(h.init(hf.plan, hp));

    /* Every pooled expert is hot, so the guards refuse nothing and only the move budget can bound
     * the proposals. */
    const int32_t ids[4] = {4, 5, 6, 7};
    const int32_t counts[4] = {20, 20, 20, 20};
    h.observe_ids(0, ids, counts, 4);

    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK_EQ((int)sw.size(), 3);

    /* No unit appears twice: a proposal marks both halves busy, or the mover would repeat a
     * transfer. */
    for (size_t i = 0; i < sw.size(); ++i)
        for (size_t j = i + 1; j < sw.size(); ++j) {
            CHECK(sw[i].promote != sw[j].promote);
            CHECK(sw[i].demote != sw[j].demote);
        }
}

namespace {

/* A xorshift32 draw below n. */
struct Rnd {
    uint32_t x;
    uint32_t operator()(uint32_t n) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x % n; }
};

/* The heat tests' plan: an embedding and four layers of 16 experts (expert e of `bytes(e)`),
 * expert_tiered with `vram_mib` of VRAM for weights. */
template <class Bytes>
Plan expert_plan(Fixture& f, Bytes bytes, int64_t vram_mib, int64_t host_mib) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                              std::to_string(e) + ".weight",
                                          bytes(e), RAD_ACCESS_CONDITIONAL, l, e);
            f.op("moe_gemm", {w}, false);
        }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = vram_mib;
    cfg.host_pool_mib = host_mib;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    Plan plan;
    CHECK_OK(Planner(in).plan(&plan));
    return plan;
}

/* What step() must propose: up to `moves` picks on `ref`, a copy of the engine, each pair marked
 * busy as step() marks it. */
std::vector<Swap> repeated_picks(HeatEngine& ref, int moves) {
    std::vector<Swap> want;
    for (int mv = 0; mv < moves; ++mv) {
        Swap sw;
        if (!ref.pick(&sw)) break;
        ref.set_busy(sw.promote, true);
        ref.set_busy(sw.demote, true);
        want.push_back(sw);
    }
    return want;
}

bool same_swaps(const std::vector<Swap>& a, const std::vector<Swap>& b) {
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
        same = a[i].promote == b[i].promote && a[i].demote == b[i].demote;
    return same;
}

/* The mover's side of proposed swaps: three in four land, the rest are abandoned. */
void land_or_abandon(HeatEngine& h, const std::vector<Swap>& got, Rnd& rnd) {
    for (const Swap& sw : got) {
        if (rnd(4) == 0) { h.abandon(sw); continue; }
        h.set_tier(sw.promote, Tier::VRAM);
        h.set_tier(sw.demote, Tier::HostPinned);
        h.set_busy(sw.promote, false);
        h.set_busy(sw.demote, false);
    }
}

}  /* namespace */

/* step() ranks every movable unit once and proposes from both ends; pick() states the policy one
 * pair per call. They must agree exactly: same pairs in the same order, same units left busy, same
 * stopping point. Checked over random heats with frequent ties (small integer counts), random busy
 * units, and four layers with differing resident counts, exercising every tie-break. */
TEST(heat_step_proposes_exactly_what_repeated_picks_would) {
    Fixture f;
    const Plan plan = expert_plan(f, [](int) { return 1 * MiB; }, 29, 128);

    Rnd rnd{0x9E3779B9u};
    int agreed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        HeatParams hp;
        hp.moves_per_step = 1 + (int)rnd(8);
        hp.min_gain = (float)rnd(3);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(3);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        /* Uneven residency across layers exercises the resident-count tie-break. */
        for (int k = 0; k < 6; ++k) {
            const int32_t u = (int32_t)rnd((uint32_t)n_unit);
            h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
        }
        for (int l = 0; l < 4; ++l) {
            int32_t ids[16], counts[16];
            for (int e = 0; e < 16; ++e) { ids[e] = e; counts[e] = (int32_t)rnd(4); }
            h.observe_ids(l, ids, counts, 16);
        }
        for (int k = 0; k < 5; ++k) h.set_busy((int32_t)rnd((uint32_t)n_unit), true);

        HeatEngine ref = h;
        const std::vector<Swap> want = repeated_picks(ref, hp.moves_per_step);
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = same_swaps(got, want);
        for (int32_t u = 0; same && u < n_unit; ++u) same = h.busy(u) == ref.busy(u);
        CHECK(same);
        if (same) ++agreed;
    }
    CHECK_EQ(agreed, 400);
}

/* Two slab classes, as on a rank holding an uneven slice of every expert (three blocks of one
 * parity, two of the other), so even and odd experts' units differ in size. A swap stays inside a
 * class (the promotion takes and the demotion frees a slot of that class), each class has its own
 * move budget, and step() still matches repeated picks. */
TEST(heat_swaps_stay_inside_a_slab_class) {
    Fixture f;
    const Plan plan = expert_plan(f, [](int e) { return (e % 2 ? 2 : 3) * MiB; }, 81, 256);
    int expert_classes = 0;
    for (const SlabClass& sc : plan.classes) expert_classes += sc.name.rfind("expert", 0) == 0;
    CHECK_EQ(expert_classes, 2);

    Rnd rnd{0x2545F491u};
    int agreed = 0, crossed = 0, proposed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        HeatParams hp;
        hp.moves_per_step = 1 + (int)rnd(8);
        hp.min_gain = (float)rnd(3);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(3);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        for (int k = 0; k < 6; ++k) {
            const int32_t u = (int32_t)rnd((uint32_t)n_unit);
            h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
        }
        /* Some layers route only one parity, so one class can run out of candidates while the
         * other still has pairs. */
        for (int l = 0; l < 4; ++l) {
            int32_t ids[16], counts[16];
            const int only = (int)rnd(3);
            for (int e = 0; e < 16; ++e) {
                ids[e] = e;
                counts[e] = (only < 2 && e % 2 != only) ? 0 : (int32_t)rnd(5);
            }
            h.observe_ids(l, ids, counts, 16);
        }
        for (int k = 0; k < 5; ++k) h.set_busy((int32_t)rnd((uint32_t)n_unit), true);

        /* Reference: repeated picks, at most `moves_per_step` per class. A pick from a class at
         * its budget is set aside, busy in the copy only. */
        HeatEngine ref = h;
        std::vector<Swap> want;
        std::vector<uint8_t> aside((size_t)n_unit, 0);
        std::vector<int> per(plan.classes.size(), 0);
        for (int guard = 0; guard < n_unit; ++guard) {
            Swap sw;
            if (!ref.pick(&sw)) break;
            ref.set_busy(sw.promote, true);
            ref.set_busy(sw.demote, true);
            int& n = per[(size_t)plan.units[(size_t)sw.promote].slab_class];
            if (n >= hp.moves_per_step) {
                aside[(size_t)sw.promote] = aside[(size_t)sw.demote] = 1;
                continue;
            }
            ++n;
            want.push_back(sw);
        }
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = same_swaps(got, want);
        for (int32_t u = 0; same && u < n_unit; ++u)
            same = aside[(size_t)u] || h.busy(u) == ref.busy(u);
        CHECK(same);
        if (same) ++agreed;
        for (const Swap& sw : got) {
            ++proposed;
            crossed += plan.units[(size_t)sw.promote].slab_class !=
                       plan.units[(size_t)sw.demote].slab_class;
        }
    }
    CHECK_EQ(agreed, 400);
    CHECK_EQ(crossed, 0);
    CHECK(proposed > 0);
}

/* The same agreement across dispatches. step() maintains each layer's two ends incrementally
 * through decays, credits, tier changes and units entering or leaving flight, so a long run mixing
 * all four is compared dispatch by dispatch against pick() on a copy. */
TEST(heat_step_agrees_with_picks_over_a_run_of_dispatches) {
    Fixture f;
    const Plan plan = expert_plan(f, [](int) { return 1 * MiB; }, 29, 128);

    Rnd rnd{0x2545F491u};
    for (int run = 0; run < 20; ++run) {
        HeatParams hp;
        hp.moves_per_step = 1 + (int)rnd(3);
        hp.min_gain = 1.0f + (float)rnd(4);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(4);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        int agreed = 0, proposed = 0;
        for (int d = 0; d < 300; ++d) {
            /* Per-dispatch histograms: mostly quiet, a few hot experts, occasional bursts. */
            for (int l = 0; l < 4; ++l) {
                int32_t counts[16];
                for (int e = 0; e < 16; ++e)
                    counts[e] = rnd(5) == 0 ? (int32_t)rnd(d % 37 == 0 ? 40 : 6) : 0;
                h.observe(l, counts, 16, 1.0f);
            }
            HeatEngine ref = h;
            const std::vector<Swap> want = repeated_picks(ref, hp.moves_per_step);
            std::vector<Swap> got;
            CHECK_OK(h.step(&got));
            const bool same = same_swaps(got, want);
            CHECK(same);
            if (same) ++agreed;
            proposed += (int)got.size();
            /* Most swaps land, some are abandoned, and occasionally a unit changes tier
             * independently. */
            land_or_abandon(h, got, rnd);
            if (rnd(10) == 0) {
                const int32_t u = (int32_t)rnd((uint32_t)n_unit);
                h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
            }
        }
        CHECK_EQ(agreed, 300);
        CHECK(proposed > 0);
    }
}

/* Histograms are matched to units by expert id, so the layouts that could break that are checked:
 * an expert count that is not a whole number of blocks, an expert id no unit holds, an immovable
 * expert, two units naming one expert, a movable unit with a layer but no expert id, one with no
 * layer, and histograms shorter or longer than the layer. Every heat and priced read is compared
 * per dispatch with a per-unit model, and every proposal with pick() on a copy. Layers are skipped
 * at random so step(), not the next observe(), rebuilds an end a tier change invalidated. */
TEST(heat_credits_and_prices_every_layout_the_plan_can_have) {
    Plan plan;
    plan.dynamic = true;
    const int n_ex[5] = {5, 17, 33, 16, 1};
    auto add = [&](int32_t layer, int32_t expert, bool movable, int64_t bytes) {
        MoveUnit mu;
        mu.layer = layer; mu.expert = expert; mu.movable = movable; mu.bytes = bytes;
        mu.tier = plan.units.size() % 3 == 0 ? Tier::VRAM : Tier::HostPinned;
        plan.units.push_back(mu);
        return (int32_t)plan.units.size() - 1;
    };
    /* by_expert[l][e]: the unit a histogram entry credits (the last unit naming it), or -1. */
    std::vector<std::vector<int32_t>> by_expert(5);
    for (int l = 0; l < 5; ++l) {
        by_expert[(size_t)l].assign((size_t)n_ex[l], -1);
        for (int e = 0; e < n_ex[l]; ++e) {
            if (l == 2 && (e == 7 || e == 20)) continue;                 /* a hole */
            by_expert[(size_t)l][(size_t)e] = add(l, e, e % 7 != 6, (int64_t)(1 + l * 64 + e) << 10);
        }
    }
    by_expert[1][3] = add(1, 3, true, 5 << 10);                          /* a second unit for (1, 3) */
    add(3, -1, true, 7 << 10);
    add(3, -1, true, 9 << 10);
    add(-1, -1, true, 11 << 10);
    add(-1, -1, true, 13 << 10);
    add(-1, -1, false, 15 << 10);
    const int32_t n_unit = (int32_t)plan.units.size();

    HeatParams hp;
    hp.moves_per_step = 3;
    HeatEngine h;
    CHECK_OK(h.init(plan, hp));
    CHECK(h.enabled());

    std::vector<float> want((size_t)n_unit, 0.0f);
    uint64_t routed = 0, resident = 0, stream = 0;
    Rnd rnd{0x9E3779B9u};
    int agreed = 0, proposed = 0, compared = 0;
    for (int d = 0; d < 400; ++d) {
        for (int l = 0; l < 5; ++l) {
            if (rnd(6) == 0) continue;
            /* Shorter than, equal to, or longer than the layer; entries past the layer's last
             * expert must be ignored, not read as the next layer's. */
            const int n = std::max(0, n_ex[l] + (int)rnd(9) - 4);
            std::vector<int32_t> counts((size_t)n + 1);   /* never empty: a null histogram is ignored */
            for (int e = 0; e < n; ++e) counts[(size_t)e] = rnd(3) == 0 ? (int)rnd(7) - 1 : 0;
            const float w = rnd(4) == 0 ? 3.0f : 1.0f;
            const uint64_t passes = w >= 1.0f ? (uint64_t)(w + 0.5f) : 1u;
            h.observe(l, counts.data(), n, w);
            for (int32_t u = 0; u < n_unit; ++u)
                if (plan.units[(size_t)u].layer == l) want[(size_t)u] = want[(size_t)u] * hp.decay;
            for (int e = 0; e < std::min(n, n_ex[l]); ++e) {
                const int32_t u = by_expert[(size_t)l][(size_t)e];
                if (u < 0 || counts[(size_t)e] <= 0) continue;
                want[(size_t)u] = want[(size_t)u] + (float)counts[(size_t)e] * w;
                routed += passes;
                if (h.tier(u) == Tier::VRAM) resident += passes;
                else stream += (uint64_t)plan.units[(size_t)u].bytes * passes;
            }
        }
        for (int32_t u = 0; u < n_unit; ++u) {
            if (!plan.units[(size_t)u].movable) continue;
            CHECK_EQ(h.heat(u), want[(size_t)u]);
            ++compared;
        }
        CHECK_EQ(h.stats().routed, routed);
        CHECK_EQ(h.stats().routed_resident, resident);
        CHECK_EQ(h.stats().stream_bytes, stream);

        HeatEngine ref = h;
        const std::vector<Swap> expect = repeated_picks(ref, hp.moves_per_step);
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        const bool same = same_swaps(got, expect);
        CHECK(same);
        if (same) ++agreed;
        proposed += (int)got.size();
        land_or_abandon(h, got, rnd);
        if (rnd(5) == 0) {
            const int32_t u = (int32_t)rnd((uint32_t)n_unit);
            h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
        }
    }
    CHECK_EQ(agreed, 400);
    CHECK(proposed > 0);
    CHECK(compared > 0);
}

TEST(heat_is_off_when_nothing_is_movable) {
    Fixture f;
    build_dense(f, 4, 1 * MiB, false);
    Config c = base_config();
    c.placement = "all_vram";
    c.vram_weights_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    HeatEngine h;
    CHECK_OK(h.init(p, HeatParams{}));
    CHECK(!h.enabled());   /* all in VRAM: the mover never runs either */
}

/* ================================================================== the mover */

/* A dispatch's compute work, standing in for the layer's kernels. Every mover test that drives
 * dispatches submits this to the compute stream. With an empty compute stream:
 *
 *   The quarantine barrier is vacuous. It is an event on the compute stream meaning "every kernel
 *   that could still name the freed slot has retired"; on an empty stream it completes before the
 *   mover's transfer starts.
 *
 *   The lead gate is a no-op. Its host wait paces the dispatch thread to the card, and so supplies
 *   the mover with slots; on an empty stream it paces nothing.
 *
 *   The loop outruns the transfers. A transfer takes microseconds and an empty iteration
 *   nanoseconds, so a dispatch-count bound assumes a relation between dispatch rate and link
 *   latency that does not hold (on the host backend the worker may not even be scheduled).
 */
struct ComputeWork {
    static constexpr int64_t kBytes = 2 * 1024 * 1024;
    void* a = nullptr;
    void* b = nullptr;

    ComputeWork() {
        a = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
        b = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
    }
    ~ComputeWork() {
        if (a) rad_dev_free(a, RAD_MEM_DEVICE);
        if (b) rad_dev_free(b, RAD_MEM_DEVICE);
    }
    ComputeWork(const ComputeWork&) = delete;
    ComputeWork& operator=(const ComputeWork&) = delete;

    /* Issued right after begin_dispatch, where a layer's launches go: the mover records its
     * ordering gate after them, which is why engine_tick sits at the end of a layer dispatch. */
    void run(RadStream s) const {
        if (a && b) rad_memcpy_async(a, b, kBytes, s);
    }
};

namespace {

/* Loops waiting for the mover are bounded by wall-clock time, not dispatch count. The lead gate
 * paces this thread to the compute stream only; on a loaded machine the host backend's mover
 * worker can go unscheduled for hundreds of dispatches, so a count bound fails spuriously while a
 * deadline fails only if the mover has stopped. */
struct Deadline {
    std::chrono::steady_clock::time_point end;
    explicit Deadline(int seconds = 60)
        : end(std::chrono::steady_clock::now() + std::chrono::seconds(seconds)) {}
    bool passed() const { return std::chrono::steady_clock::now() >= end; }
};

bool any_busy(const HeatEngine& h, const Plan& p) {
    for (int32_t u = 0; u < (int32_t)p.units.size(); ++u)
        if (h.busy(u)) return true;
    return false;
}

}  /* namespace */

TEST(slot_pool_hands_out_and_takes_back) {
    SlotPool s;
    s.init(4, 64);
    CHECK(s.empty());
    CHECK_EQ(s.take(), (int64_t)-1);   /* starvation is the normal steady state, not an error */
    s.add_free(2);
    s.add_free(3);
    CHECK_EQ(s.n_free(), (int64_t)2);
    /* Lowest first, so occupied slots pack toward the bottom and the top can be handed back. */
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    CHECK(s.empty());
}

/* take_where returns the lowest free slot the predicate accepts and leaves the others free, still
 * handed out lowest first (Mover::pool_slot_for uses it to keep a layer's pool slots together). */
TEST(a_slot_pool_takes_the_lowest_slot_a_predicate_wants) {
    SlotPool s;
    s.init(8, 64);
    for (int64_t k : {1, 3, 4, 6, 7}) s.add_free(k);
    CHECK_EQ(s.take_where([](int64_t k) { return k % 3 == 0; }), (int64_t)3);
    CHECK_EQ(s.take_where([](int64_t k) { return k == 5; }), (int64_t)-1);   /* not free */
    CHECK_EQ(s.n_free(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)1);
    CHECK_EQ(s.take_where([](int64_t k) { return k > 4; }), (int64_t)6);
    CHECK_EQ(s.take(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)7);
    CHECK(s.empty());
}

TEST(a_slot_pool_grows_only_into_the_capacity_it_was_given) {
    SlotPool s;
    s.init(2, 64, 5);
    CHECK_EQ(s.size(), (int64_t)2);
    CHECK_EQ(s.capacity(), (int64_t)5);
    CHECK_EQ(s.take(), (int64_t)-1);   /* both live slots are held by the plan */

    s.set_target(4);
    CHECK_EQ(s.size(), (int64_t)4);
    CHECK_EQ(s.n_free(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);

    s.set_target(99);                  /* clamped to the capacity */
    CHECK_EQ(s.size(), (int64_t)5);
    CHECK_EQ(s.take(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)-1);
}

TEST(a_slot_pool_shrinks_past_what_nothing_holds) {
    SlotPool s;
    s.init(2, 64, 6);
    s.set_target(6);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    /* 4 and 5 are free, so the mark falls to 4 at once and stops at held slot 3. */
    s.set_target(0);
    CHECK_EQ(s.size(), (int64_t)4);
    CHECK_EQ(s.n_free(), (int64_t)0);
    CHECK_EQ(s.take(), (int64_t)-1);   /* nothing above the target is handed out */

    s.add_free(3);
    CHECK_EQ(s.size(), (int64_t)3);
    s.add_free(2);
    CHECK_EQ(s.size(), (int64_t)2);    /* the original live set is the floor */
}

TEST(a_slot_a_shrink_stranded_is_never_handed_out_twice) {
    SlotPool s;
    s.init(0, 64, 4);
    s.set_target(4);
    CHECK_EQ(s.n_free(), (int64_t)4);
    s.set_target(2);                   /* 2 and 3 were free, so the mark falls at once */
    CHECK_EQ(s.size(), (int64_t)2);
    CHECK_EQ(s.n_free(), (int64_t)2);
    s.set_target(4);                   /* regrows without duplicating heap entries */
    CHECK_EQ(s.n_free(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)0);
    CHECK_EQ(s.take(), (int64_t)1);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    CHECK_EQ(s.take(), (int64_t)-1);
}

TEST(a_slot_pool_regrows_every_slot_a_shrink_dropped) {
    /* A dropped slot that take() skipped on the way to a lower one. */
    {
        SlotPool s;
        s.init(0, 64, 4);
        s.set_target(4);
        s.set_target(2);
        CHECK_EQ(s.take(), (int64_t)0);
        CHECK_EQ(s.take(), (int64_t)1);
        CHECK_EQ(s.take(), (int64_t)-1);   /* 2 and 3 are above the target */
        s.set_target(4);
        CHECK_EQ(s.n_free(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)3);
        CHECK_EQ(s.take(), (int64_t)-1);
    }
    /* A slot held above the target at shrink time, freed afterwards. */
    {
        SlotPool s;
        s.init(2, 64, 6);
        s.set_target(6);
        CHECK_EQ(s.take(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)3);
        CHECK_EQ(s.take(), (int64_t)4);
        s.set_target(2);
        CHECK_EQ(s.size(), (int64_t)5);    /* the mark stops at the slot 4 holds */
        s.add_free(4);
        CHECK_EQ(s.size(), (int64_t)4);
        s.set_target(6);
        CHECK_EQ(s.size(), (int64_t)6);
        CHECK_EQ(s.n_free(), (int64_t)2);
        CHECK_EQ(s.take(), (int64_t)4);
        CHECK_EQ(s.take(), (int64_t)5);
        CHECK_EQ(s.take(), (int64_t)-1);
        /* The full capacity is back: every slot returned comes out once. */
        for (int64_t k = 2; k < 6; ++k) s.add_free(k);
        CHECK_EQ(s.n_free(), (int64_t)4);
        for (int64_t k = 2; k < 6; ++k) CHECK_EQ(s.take(), k);
        CHECK_EQ(s.take(), (int64_t)-1);
    }
}

TEST(file_tier_alignment_is_the_container_alignment) {
    /* O_DIRECT needs aligned offsets, lengths and buffers, and every RadFileEntry offset is
     * RAD_ALIGN_UNIT-aligned, so the fast path needs no head adjustment. */
    CHECK_EQ(FileTier::direct_align(1), (int64_t)RAD_ALIGN_UNIT);
    CHECK_EQ(FileTier::direct_align(RAD_ALIGN_UNIT), (int64_t)RAD_ALIGN_UNIT);
    CHECK_EQ(FileTier::direct_align(RAD_ALIGN_UNIT + 1), (int64_t)(2 * RAD_ALIGN_UNIT));
    FileTier t;
    CHECK(!t.ok());
    CHECK_EQ(t.read(0, 16, nullptr, 0), RAD_E_STATE);
}

TEST(mover_publishes_addresses_for_every_tier) {
    HeatFixture hf;
    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));

    ResidencyTable* rt = m.residency();
    for (rad_weight w = 1; (size_t)w < hf.f.prog.weights.size(); ++w) {
        const WeightSlot* s = rt->lookup(w);
        CHECK(s != nullptr);
        /* Every unit here is in VRAM or the pinned pool, so every weight has an address (a
         * file-tier unit has none until staged). */
        if (s) CHECK(s->ptr != nullptr);
    }
    CHECK(rt->lookup(9999u) == nullptr);
}

namespace {
/* A position-hashed container byte, so a weight read from the wrong offset or copied to the wrong
 * place in its unit mismatches on almost every byte. */
uint8_t container_byte(int64_t i) {
    uint64_t z = (uint64_t)i + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (uint8_t)((z ^ (z >> 31)) >> 56);
}

/* What a published weight holds, read back off the device after the mover's copies. */
bool published_matches(Mover& m, const Program& prog, rad_weight w,
                       const std::vector<uint8_t>& img, RadStream s) {
    const WeightSlot* slot = m.residency()->lookup(w);
    if (!slot || !slot->ptr) return false;
    const WeightInfo& wi = prog.weights[w];
    std::vector<uint8_t> got((size_t)wi.stored_bytes);
    rad_stream_sync(m.transfer_stream());
    if (rad_memcpy_async(got.data(), slot->ptr, wi.stored_bytes, s) < 0) return false;
    rad_stream_sync(s);
    return std::memcmp(got.data(), img.data() + wi.file_offset, (size_t)wi.stored_bytes) == 0;
}
}  /* namespace */

TEST(the_file_stager_reads_each_routed_layer_from_the_container) {
    /* Three routed layers of four 20 MiB experts; 0 and 2 are popular and stay resident, so 1
     * and 3 of each layer are read from the container. This places resident bytes between
     * file-tier ones (skipped, not read through), makes a layer's file-tier part (40 MiB) wider
     * than a read window (32 MiB) so weights split across windows, and leaves each weight a
     * non-whole number of blocks so the memory layout differs from the container's. A hole between
     * layers is skipped too. */
    Fixture f;
    const int64_t gu = 14 * MiB + 1000, dn = 6 * MiB + 24;
    const int L = 3;
    build_moe_layers(f, L, 4, gu, dn);

    int64_t off = 4 * (int64_t)RAD_ALIGN_UNIT;
    int32_t prev = -1;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        WeightInfo& wi = f.prog.weights[w];
        if (prev >= 0 && wi.decl.group.layer != prev) off += 2 * MiB;   /* the hole */
        prev = wi.decl.group.layer;
        wi.stored_bytes = wi.logical_bytes;
        wi.file_offset = (uint64_t)off;
        off = align_up(off + wi.stored_bytes, RAD_ALIGN_UNIT);
    }
    std::vector<uint8_t> img((size_t)off);
    for (int64_t i = 0; i < off; ++i) img[(size_t)i] = container_byte(i);
    const std::string path = "place_test_file_tier.rad";
    {
        FILE* fp = std::fopen(path.c_str(), "wb");
        CHECK(fp != nullptr);
        if (!fp) return;
        CHECK_EQ(std::fwrite(img.data(), 1, img.size(), fp), img.size());
        std::fclose(fp);
    }
    {
        DirectFile probe;
        if (probe.open(path.c_str(), RAD_DIO_READ) < 0) {
            std::printf("  SKIP the working directory has no O_DIRECT\n");
            std::remove(path.c_str());
            return;
        }
    }

    std::vector<RadFileProfile> prof;
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < 4; ++e)
            prof.push_back(RadFileProfile{l, e, e == 0 ? 0.5f : e == 2 ? 0.4f : 0.01f, 0.0f});
    Config c = base_config();
    c.placement = "expert_tiered";
    c.deterministic = true;          /* no shadow slots, so residents are exactly the fill's */
    c.vram_weights_mib = 210;
    c.host_pool_mib = (4 * 21 * MiB + RAD_PLACE_FILE_WINDOWS * RAD_PLACE_FILE_WINDOW_BYTES) / MiB + 1;
    c.weights_disk_tier = true;
    PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    std::vector<std::vector<rad_weight>> file_w((size_t)L), res_w((size_t)L);
    for (const MoveUnit& mu : p.units) {
        if (mu.expert < 0) continue;
        CHECK(mu.tier == ((mu.expert & 1) ? Tier::SSD : Tier::VRAM));
        for (rad_weight w : mu.weights)
            (mu.tier == Tier::SSD ? file_w : res_w)[(size_t)mu.layer].push_back(w);
    }

    Mover m;
    MoverConfig mc;
    mc.container_path = path;
    CHECK_OK(m.init(f.prog, p, mc));
    RadStream s = nullptr;
    CHECK_OK(rad_stream_create(&s, 0));
    CHECK_OK(m.bind_compute_stream(s));
    FileStager fs;
    CHECK_OK(fs.init(f.prog, p, &m));
    CHECK(fs.enabled());
    CHECK_EQ(fs.n_layers(), (size_t)L);

    /* Whether every file-tier weight of layer `l` is published with its container bytes (want),
     * or none is. */
    auto all = [&](int l, bool want) {
        int ok = 0;
        for (rad_weight w : file_w[(size_t)l]) {
            const WeightSlot* slot = m.residency()->lookup(w);
            const bool have = slot && slot->ptr;
            if (have != want) continue;
            if (!want || published_matches(m, f.prog, w, img, s)) ++ok;
        }
        return ok == (int)file_w[(size_t)l].size();
    };
    for (int l = 0; l < L; ++l) CHECK(all(l, false));

    /* Two passes; in the second the stager knows where a pass ends. Op 0 is the embedding and
     * layer l's expert ops are 2l+1 and 2l+2. */
    for (int pass = 0; pass < 2; ++pass) {
        RadBatch b{};
        CHECK(fs.begin_pass(&b));
        bool resync = false;
        CHECK_OK(fs.before_op(0, s, &resync));          /* the pass's first op reads layer 0 */
        CHECK(resync);
        CHECK(all(0, true) && all(1, false) && all(2, false));
        CHECK_OK(fs.before_op(1, s, &resync));          /* layer 0 starts; layer 1 read ahead */
        CHECK(all(0, true) && all(1, true) && all(2, false));
        CHECK_OK(fs.after_op(1, s));
        CHECK_OK(fs.before_op(1, s, &resync));          /* a second slice of layer 0: nothing */
        CHECK(!resync);
        CHECK_OK(fs.before_op(2, s, &resync));
        CHECK_OK(fs.before_op(3, s, &resync));          /* layer 1 starts: 0 released, 2 read */
        CHECK(all(0, false) && all(1, true) && all(2, true));
        CHECK_OK(fs.before_op(4, s, &resync));
        CHECK_OK(fs.before_op(5, s, &resync));          /* layer 2 starts: 1 released */
        CHECK(all(0, false) && all(1, false) && all(2, true));
        CHECK_OK(fs.before_op(6, s, &resync));
        CHECK_OK(fs.end_pass(s));
        for (int l = 0; l < L; ++l) CHECK(all(l, false));
    }
    /* A resident expert is never touched. */
    for (int l = 0; l < L; ++l)
        for (rad_weight w : res_w[(size_t)l]) {
            const WeightSlot* slot = m.residency()->lookup(w);
            CHECK(slot && slot->ptr);
        }
    /* Three layers per pass, each at least two reads (the resident expert between the file-tier
     * ones is skipped, not read), and every unit read once. */
    CHECK_EQ(m.stats().file_layers, (uint64_t)(2 * L));
    CHECK_EQ(m.stats().file_units, (uint64_t)(2 * L * 2));
    CHECK(m.stats().file_preads >= (uint64_t)(2 * L * 2));
    rad_stream_sync(m.transfer_stream());
    rad_stream_destroy(s);
    std::remove(path.c_str());
}

TEST(mover_promotes_and_demotes_through_the_shadow_slots) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    CHECK_OK(h.init(hf.plan, hp));

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));

    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    const int32_t hot = hf.unit_of_expert(4);
    const rad_weight hw = hf.plan.units[(size_t)hot].weights.front();
    void* before = m.residency()->lookup(hw)->ptr;
    const uint32_t gen_before = m.residency()->lookup(hw)->generation;

    /* Every pooled expert is wanted. Each iteration is a whole dispatch (retire, observe, decide,
     * issue), as a real caller runs it; slot supply only turns over if all four repeat.
     *
     * The pool free list starts empty, so the first round can only promote, into the shadow slot.
     * The pool slot that frees is available only after its copy lands and its quarantine clears,
     * many iterations later on a card. A loop that stops issuing early would never see a demotion
     * on a card, yet pass on the host backend where copies land within the dispatch; hence the
     * loop keeps issuing until both happen. */
    ComputeWork work;
    const int32_t ids[4] = {4, 5, 6, 7};
    const int32_t counts[4] = {10, 10, 10, 10};
    int d = 0;
    for (Deadline dl; !dl.passed(); ++d) {
        CHECK_OK(m.begin_dispatch(d, &h));
        work.run(compute);
        h.observe_ids(0, ids, counts, 4);
        std::vector<Swap> sw;
        CHECK_OK(h.step(&sw));
        for (const Swap& s : sw) m.issue(s, &h);
        if (m.stats().promotions >= 1 && m.stats().demotions >= 1) break;
    }
    /* Bounded, not exact: retire() publishes only transfers that have landed (it polls
     * rad_event_query), so the dispatch count depends on the link. Only convergence and both
     * halves of the swap are asserted. */

    CHECK(m.stats().promotions >= 1);
    CHECK(m.stats().demotions >= 1);
    CHECK(m.stats().h2d_bytes > 0);
    CHECK(m.stats().d2h_bytes > 0);
    /* The first round found no pool slot for its demotion, and the starvation counter records it. */
    CHECK(m.stats().starved_pool >= 1);

    const WeightSlot* after = m.residency()->lookup(hw);
    CHECK(after->ptr != before);
    CHECK(after->generation > gen_before);
    CHECK(h.tier(hot) == Tier::VRAM);

    /* Once the mover is quiescent nothing is left busy: every proposal was published or handed
     * back. A unit may still be busy when the loop above breaks (`busy` clears when the slot
     * leaves quarantine, several dispatches after the copy lands), so stop proposing and keep
     * retiring until it settles. */
    for (Deadline dl; any_busy(h, hf.plan) && !dl.passed(); ++d) {
        CHECK_OK(m.begin_dispatch(d, &h));
        work.run(compute);
    }
    for (int32_t u = 0; u < (int32_t)hf.plan.units.size(); ++u) CHECK(!h.busy(u));

    rad_stream_destroy(compute);
}

namespace {
int32_t resident_count(const HeatEngine& h, const Plan& p) {
    int32_t n = 0;
    for (int32_t u = 0; u < (int32_t)p.units.size(); ++u)
        if (h.tier(u) == Tier::VRAM) ++n;
    return n;
}
}  /* namespace */

/* ---- the loan. The KV cache lends the slab its unused VRAM, in place. A plain device allocation
 * stands in for the cache's carve: the mover is given the stretches and a loan in tokens, and
 * nothing about what the owner keeps there. */
namespace {
struct Loan {
    static constexpr int64_t kStrip = 4 * MiB;
    static constexpr int64_t kUnitBytes = 64 * 1024;   /* bytes of a stretch per owner unit */
    static constexpr int64_t kUnitTokens = 16;
    void* mem = nullptr;
    std::vector<LoanStrip> strips;
    Loan() {
        mem = rad_dev_alloc(2 * kStrip, RAD_MEM_DEVICE);
        for (int k = 0; k < 2; ++k) {
            LoanStrip st;
            st.top = (char*)mem + (k + 1) * kStrip;
            st.bytes = kStrip;
            st.unit_bytes = kUnitBytes;
            st.unit_tokens = kUnitTokens;
            strips.push_back(st);
        }
    }
    ~Loan() { if (mem) rad_dev_free(mem, RAD_MEM_DEVICE); }
    bool inside(const void* p) const {
        return (const char*)p >= (const char*)mem && (const char*)p < (const char*)mem + 2 * kStrip;
    }
    /* Everything the stretches can hold. */
    static constexpr int64_t kAll = kStrip / kUnitBytes * kUnitTokens;
};

struct LoanRig {
    HeatFixture hf;
    HeatEngine  h;
    Mover       m;
    Loan        loan;
    RadStream   compute = nullptr;
    ComputeWork work;
    int         d = 0;
    int32_t     planned = 0;
    /* None by default: on a slab this small the production count would leave most of the loan
     * empty. Tests of turnover set their own. */
    int         turnover = 0;

    explicit LoanRig(int n_expert = 8) : hf(n_expert) {}

    int build() {
        HeatParams hp;
        RAD_TRY(h.init(hf.plan, hp));
        MoverConfig mc;
        mc.turnover_slots = turnover;
        RAD_TRY(m.init(hf.f.prog, hf.plan, mc));
        RAD_TRY(rad_stream_create(&compute, 0));
        RAD_TRY(m.bind_compute_stream(compute));
        RAD_TRY(m.lend_from(Mover::kLendKV, loan.strips));
        planned = resident_count(h, hf.plan);
        return RAD_OK;
    }
    ~LoanRig() { if (compute) { rad_stream_sync(compute); rad_stream_destroy(compute); } }

    /* One dispatch as the engine runs it: retire, the step's work, then the loan's fill. */
    int dispatch() {
        RAD_TRY(m.begin_dispatch(d++, &h));
        work.run(compute);
        return m.balance(&h);
    }
    int32_t pooled() const {
        int32_t n = 0;
        for (int32_t u = 0; u < (int32_t)hf.plan.units.size(); ++u)
            if (hf.plan.units[(size_t)u].tier == Tier::HostPinned &&
                hf.plan.units[(size_t)u].movable) ++n;
        return n;
    }
    const void* addr(int32_t u) {
        return m.residency()->lookup(hf.plan.units[(size_t)u].weights.front())->ptr;
    }
};
}  /* namespace */

TEST(the_lent_slots_hold_nothing_until_a_loan_covers_them) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK(r.pooled() > 0);
    /* Capacity exists but nothing is offered: the stretches are the owner's until lent. */
    CHECK(r.m.flex_capacity() > 0);
    CHECK_EQ(r.m.flex_offered_bytes(), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    /* A dispatch with nothing lent fills nothing. */
    CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().promotions, (uint64_t)0);
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK(r.m.flex_offered_bytes() > 0);
    CHECK_EQ(r.m.lend_from(Mover::kLendKV, r.loan.strips), RAD_E_STATE);   /* carved once */

    /* A mover with nothing to borrow has nothing to lend into; not an error. */
    HeatFixture hf;
    Mover none;
    MoverConfig nc;
    CHECK_OK(none.init(hf.f.prog, hf.plan, nc));
    CHECK_OK(none.lend_from(Mover::kLendKV, {}));
    CHECK_EQ(none.flex_capacity(), (int64_t)0);
    CHECK_OK(none.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_EQ(none.flex_offered_bytes(), (int64_t)0);
}

/* An empty lent slot wastes VRAM every step, so a loan is filled in the dispatch it arrives in:
 * every slot beyond the turnover slots at once, hottest units first. */
TEST(every_lent_slot_is_filled_in_one_dispatch) {
    LoanRig r;
    CHECK_OK(r.build());
    const int64_t slots = r.m.flex_capacity() / r.hf.plan.classes[0].stride;
    const int64_t expect = std::min<int64_t>(slots - r.m.turnover_slots(), r.pooled());
    CHECK(expect > 1);

    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(r.dispatch());
    /* All issued in one call. */
    int64_t issued = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) if (r.h.busy(u)) ++issued;
    CHECK_EQ(issued, expect);

    for (Deadline dl; resident_count(r.h, r.hf.plan) < r.planned + expect && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned + (int32_t)expect);
    CHECK_EQ(r.m.flex_used_bytes(), expect * r.hf.plan.classes[0].stride);
    CHECK(r.m.loan_held(Mover::kLendKV) > 0);
    CHECK(r.m.loan_held(Mover::kLendKV) <= Loan::kAll);

    /* Every unit the fill promoted lives inside the stretches, at least one slot apart, never
     * overlapping and never in the alignment unit left empty at a stretch's top. */
    std::vector<const char*> at;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        if (r.hf.plan.units[(size_t)u].tier != Tier::HostPinned) continue;
        const char* p = (const char*)r.addr(u);
        if (!r.loan.inside(p)) continue;
        at.push_back(p);
        for (const LoanStrip& st : r.loan.strips)
            if (p >= st.top - st.bytes && p < st.top)
                CHECK(p + r.hf.plan.classes[0].stride <= st.top - (int64_t)RAD_ALIGN_UNIT);
    }
    CHECK_EQ((int64_t)at.size(), expect);
    std::sort(at.begin(), at.end());
    for (size_t k = 1; k < at.size(); ++k)
        CHECK(at[k] - at[k - 1] >= r.hf.plan.classes[0].stride);
}

/* The turnover slots stay free, and a shrinking loan does not take them for good. The fill stops
 * that many slots short so the engine's swaps have somewhere to land. Free slots are at the loan's
 * top, which a reclaiming cache takes first; the mover then rebuilds them by dropping the coldest
 * lent units, which still have host copies, so nothing crosses the link. */
TEST(the_turnover_slots_stay_free_and_are_rebuilt_by_drops) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_EQ(r.m.turnover_slots(), (int64_t)2);
    const int64_t slots = r.m.flex_capacity() / r.hf.plan.classes[0].stride;
    CHECK(r.pooled() > slots);           /* more candidates than room: only the reserve stops the fill */

    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    const int32_t full = resident_count(r.h, r.hf.plan);
    CHECK_EQ(full, r.planned + (int32_t)(slots - 2));
    CHECK_EQ(r.m.flex_used_bytes(), (slots - 2) * r.hf.plan.classes[0].stride);

    /* The fill packs from the bottom, so a loan cut to what is held takes exactly the free top
     * slots, and the next dispatch drops two lent units to free two more. */
    const int64_t held = r.m.loan_held(Mover::kLendKV);
    CHECK(held < Loan::kAll);
    const uint64_t d2h = r.m.stats().d2h_bytes;
    CHECK_OK(r.m.set_loan(Mover::kLendKV, held));
    for (Deadline dl; (r.m.stats().dropped < 2 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    for (int k = 0; k < 4; ++k) CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().dropped, (uint64_t)2);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);                          /* nothing crossed the link */
    CHECK_EQ(resident_count(r.h, r.hf.plan), full - 2);
    CHECK_EQ(r.m.flex_used_bytes(), (slots - 4) * r.hf.plan.classes[0].stride);

    /* A staging step fills them (it swaps nothing, and a unit left out is one its stager must
     * copy). The next ordinary step drops the two coldest again. */
    CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
    r.work.run(r.compute);
    CHECK_OK(r.m.balance(&r.h, /*turnover=*/false));
    for (Deadline dl; any_busy(r.h, r.hf.plan) && !dl.passed();) {
        CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
        r.work.run(r.compute);
        CHECK_OK(r.m.balance(&r.h, false));
    }
    CHECK_EQ(resident_count(r.h, r.hf.plan), full);
    for (Deadline dl; (r.m.stats().dropped < 4 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(resident_count(r.h, r.hf.plan), full - 2);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);

    /* While held, an idle tick fills them as a staging step does; once released, the next drops
     * them. */
    r.m.hold_turnover(true);
    for (Deadline dl; (resident_count(r.h, r.hf.plan) < full || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK_EQ(resident_count(r.h, r.hf.plan), full);
    r.m.hold_turnover(false);
    for (Deadline dl; (r.m.stats().dropped < 6 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK_EQ(resident_count(r.h, r.hf.plan), full - 2);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);
}

namespace {
/* Stamp every weight of every unit with its own bytes (check=false), or verify them where the
 * residency table now says they are (check=true), via device copies so it also runs on a card. */
uint8_t stamp_byte(rad_weight w, size_t i) {
    uint32_t v = (uint32_t)w * 0x9E3779B1u ^ (uint32_t)i * 0x85EBCA6Bu;
    v ^= v >> 15; v *= 0x2C1B3C6Du; v ^= v >> 12; v *= 0x297A2D39u; v ^= v >> 15;
    return (uint8_t)v;
}
int stamp_units(LoanRig& r, bool check, int* bad) {
    RadStream s = nullptr;
    RAD_TRY(rad_stream_create(&s, 0));
    std::vector<uint8_t> buf;
    for (const MoveUnit& mu : r.hf.plan.units)
        for (rad_weight w : mu.weights) {
            const size_t n = (size_t)r.hf.f.prog.weights[w].logical_bytes;
            void* at = r.m.residency()->lookup(w)->ptr;
            buf.resize(n);
            if (check) {
                RAD_TRY(rad_memcpy_async(buf.data(), at, (int64_t)n, s));
                RAD_TRY(rad_stream_sync(s));
                for (size_t i = 0; i < n; ++i)
                    if (buf[i] != stamp_byte(w, i)) { ++*bad; break; }
            } else {
                for (size_t i = 0; i < n; ++i) buf[i] = stamp_byte(w, i);
                RAD_TRY(rad_memcpy_async(at, buf.data(), (int64_t)n, s));
                RAD_TRY(rad_stream_sync(s));
            }
        }
    rad_stream_destroy(s);
    return RAD_OK;
}
/* A full loan, every unit stamped, then one swap whose budgeted victim finds the pool full: the
 * promotion lands in a working slot and the victim takes the host copy of a lent unit. */
int donate_once(LoanRig& r, bool held = false) {
    RAD_TRY(stamp_units(r, false, nullptr));
    RAD_TRY(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        RAD_TRY(r.dispatch());
    int32_t hot = -1, cold = -1;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        if (!r.hf.plan.units[(size_t)u].movable) continue;
        if (hot < 0 && r.h.tier(u) != Tier::VRAM) hot = u;
        if (cold < 0 && r.h.tier(u) == Tier::VRAM && !r.loan.inside(r.addr(u))) cold = u;
    }
    if (hot < 0 || cold < 0) return RAD_E_STATE;
    Swap sw;
    sw.promote = hot;
    sw.demote = cold;
    r.h.set_busy(hot, true);
    r.h.set_busy(cold, true);
    r.m.hold_turnover(held);
    RAD_TRY(r.m.issue(sw, &r.h));
    for (Deadline dl; any_busy(r.h, r.hf.plan) && !dl.passed();) RAD_TRY(r.dispatch());
    if (held) return r.h.tier(hot) == Tier::VRAM && r.h.tier(cold) == Tier::VRAM ? RAD_OK : RAD_E_STATE;
    return r.h.tier(hot) == Tier::VRAM && r.h.tier(cold) != Tier::VRAM ? RAD_OK : RAD_E_STATE;
}
}  /* namespace */

/* A budgeted victim with no free pool slot takes a lent unit's host copy (Mover::donor_slot); the
 * pool is sized to its units, so otherwise the budgeted range turns over only as fast as working
 * slots free up. A recall then moves the donor (it has no host copy to drop to), and every unit's
 * bytes stay where its table entry says. */
TEST(a_budgeted_victim_takes_a_lent_units_host_copy_and_a_recall_moves_the_donor) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(donate_once(r));
    CHECK_EQ(r.m.stats().donated, (uint64_t)1);
    CHECK_EQ(r.m.stats().starved_pool, (uint64_t)0);
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.stats().relocated, (uint64_t)1);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
    /* The donor stays in VRAM: a working slot was free for it. */
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned + 1);
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* Holds the compute stream behind a long copy on another stream, so anything recorded on it
 * afterwards (a quarantine barrier) stays pending until the copy finishes. */
struct ComputeHold {
    static constexpr int64_t kBytes = 64ll << 20;
    void* a = nullptr;
    void* b = nullptr;
    RadStream s = nullptr;
    RadEvent e = nullptr;

    int hold(RadStream compute, int rounds) {
        a = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
        b = rad_dev_alloc(kBytes, RAD_MEM_DEVICE);
        if (!a || !b) return RAD_E_NOMEM;
        RAD_TRY(rad_stream_create(&s, 0));
        RAD_TRY(rad_event_create(&e));
        for (int i = 0; i < rounds; ++i) RAD_TRY(rad_memcpy_async(a, b, kBytes, s));
        RAD_TRY(rad_event_record(e, s));
        return rad_event_wait(compute, e);
    }
    ~ComputeHold() {
        if (s) { rad_stream_sync(s); rad_stream_destroy(s); }
        if (e) rad_event_destroy(e);
        if (a) rad_dev_free(a, RAD_MEM_DEVICE);
        if (b) rad_dev_free(b, RAD_MEM_DEVICE);
    }
};

/* A recall right after a donation, with the card behind: the swap's copies have landed but the
 * slots they freed (the victim's budgeted slot, the promoted unit's pool slot) are still in
 * quarantine behind the compute stream. The donor is owed those slots (donor_slot), so the recall
 * drains the stream before looking; otherwise it finds none, stops, and strands the slots it
 * already reclaimed from other units. */
TEST(a_recall_moves_a_donor_whose_slots_are_still_in_quarantine) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(stamp_units(r, false, nullptr));
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    int32_t hot = -1, cold = -1;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        if (!r.hf.plan.units[(size_t)u].movable) continue;
        if (hot < 0 && r.h.tier(u) != Tier::VRAM) hot = u;
        if (cold < 0 && r.h.tier(u) == Tier::VRAM && !r.loan.inside(r.addr(u))) cold = u;
    }
    CHECK(hot >= 0 && cold >= 0);
    if (hot < 0 || cold < 0) return;

    CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
    r.work.run(r.compute);
    Swap sw;
    sw.promote = hot;
    sw.demote = cold;
    r.h.set_busy(hot, true);
    r.h.set_busy(cold, true);
    CHECK_OK(r.m.issue(sw, &r.h));
    CHECK_EQ(r.m.stats().donated, (uint64_t)1);
    ComputeHold hold;
    CHECK_OK(hold.hold(r.compute, 48));
    CHECK_EQ(rad_event_query(hold.e), 0);           /* the card is still behind at recall time */

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.stats().relocated, (uint64_t)1);
    CHECK_EQ(r.m.stats().stranded, (uint64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* No donation while the turnover slots are held (through a prompt and the steps after it): the
 * demotion is starved and the budgeted victim stays. */
TEST(a_held_turnover_takes_no_host_copy) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(donate_once(r, /*held=*/true));
    CHECK_EQ(r.m.stats().donated, (uint64_t)0);
    CHECK(r.m.stats().starved_pool >= 1);
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* A demotion of a unit with no slab slot is refused before it looks for a pool slot: with the
 * pool full that search takes a lent unit's host copy, which the refusal could not give back. */
TEST(a_demotion_with_no_slab_slot_takes_no_host_copy) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    int32_t pooled = -1;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size() && pooled < 0; ++u)
        if (r.hf.plan.units[(size_t)u].movable && r.h.tier(u) != Tier::VRAM) pooled = u;
    CHECK(pooled >= 0);
    if (pooled < 0) return;

    CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
    r.work.run(r.compute);
    Swap sw;
    sw.demote = pooled;
    r.h.set_busy(pooled, true);
    CHECK_EQ(r.m.issue(sw, &r.h), RAD_E_FULL);
    CHECK_EQ(r.m.stats().donated, (uint64_t)0);
    CHECK(!r.h.busy(pooled));
}

/* A loan shrinking under a donor moves it out before the slot is given back (balance's sweep):
 * an on-card copy into the budgeted slot its victim left, and the shrink completes once it lands.
 * The working slot it took is rebuilt by demoting the coldest budgeted unit, and every unit's
 * bytes stay where its table entry says. */
TEST(a_shrinking_loan_moves_a_donor_out_before_its_slot_is_given_back) {
    LoanRig r(24);
    r.turnover = 2;
    CHECK_OK(r.build());
    CHECK_OK(donate_once(r));
    CHECK_EQ(r.m.stats().donated, (uint64_t)1);
    const uint64_t d2h = r.m.stats().d2h_bytes;

    CHECK_OK(r.m.set_loan(Mover::kLendKV, 0));
    for (Deadline dl; (r.m.loan_held(Mover::kLendKV) > 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(r.m.stats().relocated, (uint64_t)1);
    CHECK_EQ(r.m.stats().d2h_bytes - d2h, (uint64_t)r.hf.plan.classes[0].stride);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
    int bad = 0;
    CHECK_OK(stamp_units(r, true, &bad));
    CHECK_EQ(bad, 0);
}

/* A unit in a lent slot keeps its host copy, so giving the slot back changes its address without
 * a copy across the link. */
TEST(a_shrinking_loan_drops_units_without_copying_them) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    const int32_t full = resident_count(r.h, r.hf.plan);
    CHECK(full > r.planned);
    const uint64_t d2h = r.m.stats().d2h_bytes;

    CHECK_OK(r.m.set_loan(Mover::kLendKV, 0));
    for (Deadline dl; (r.m.loan_held(Mover::kLendKV) > 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    CHECK_EQ(r.m.stats().d2h_bytes, d2h);                          /* nothing crossed the link */
    CHECK_EQ(r.m.stats().dropped, (uint64_t)(full - r.planned));
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));
}

/* reclaim() returns the memory before it returns: every unit past the loan is back on its host
 * copy, and nothing that could still read the slot is in flight. */
TEST(a_recall_takes_the_slots_back_before_it_returns) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(r.dispatch());                 /* a fill in flight, the hard case */
    CHECK(any_busy(r.h, r.hf.plan));

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    CHECK(!any_busy(r.h, r.hf.plan));
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));

    /* The slots are usable again once the loan returns. */
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK(resident_count(r.h, r.hf.plan) > r.planned);
}

/* An idle server still fills what it is lent, via its own tick, so a loan made while idle is not
 * left empty until the next request pays for filling it. */
TEST(an_idle_mover_fills_what_it_is_lent) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK(resident_count(r.h, r.hf.plan) > r.planned);
    CHECK(!any_busy(r.h, r.hf.plan));
}

/* Two lenders move independently. The arena reclaims its memory for every prefill while the KV
 * cache keeps lending, so recalling one must leave the other's slots exactly as they were. */
TEST(a_second_lender_is_recalled_without_touching_the_first) {
    LoanRig r;
    CHECK_OK(r.build());
    Loan arena;
    /* The arena lends in bytes: one unit is one byte. */
    for (LoanStrip& st : arena.strips) { st.unit_bytes = 1; st.unit_tokens = 1; }
    CHECK_OK(r.m.lend_from(Mover::kLendArena, arena.strips));
    CHECK_EQ(r.m.lend_from(Mover::kLendArena, arena.strips), RAD_E_STATE);
    CHECK_EQ(r.m.loan_ceiling(Mover::kLendArena) > 0, true);

    /* Only part of the cache is lent, so the pooled units overflow its slots and the fill (lowest
     * slot first, the cache's segment before the arena's) reaches both. */
    const int64_t kv_part = Loan::kAll / 3;
    CHECK_OK(r.m.set_loan(Mover::kLendKV, kv_part));
    CHECK_OK(r.m.set_loan(Mover::kLendArena, r.m.loan_ceiling(Mover::kLendArena)));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    int32_t in_kv = 0, in_arena = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        in_kv += r.loan.inside(r.addr(u));
        in_arena += arena.inside(r.addr(u));
    }
    CHECK(in_kv > 0);
    CHECK(in_arena > 0);
    const int64_t kv_held = r.m.loan_held(Mover::kLendKV);

    CHECK_OK(r.m.reclaim(Mover::kLendArena, 0, &r.h));
    CHECK_EQ(r.m.loan_held(Mover::kLendArena), (int64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), kv_held);
    int32_t kv_after = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) {
        CHECK(!arena.inside(r.addr(u)));
        kv_after += r.loan.inside(r.addr(u));
    }
    CHECK_EQ(kv_after, in_kv);
    CHECK_EQ(r.m.loan(Mover::kLendKV), kv_part);

    /* Re-lent: the arena's slots fill again while the cache's stay put. */
    CHECK_OK(r.m.set_loan(Mover::kLendArena, r.m.loan_ceiling(Mover::kLendArena)));
    for (Deadline dl; !dl.passed();) {
        CHECK_OK(r.dispatch());
        int32_t n = 0;
        for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) n += arena.inside(r.addr(u));
        if (n > 0 && !any_busy(r.h, r.hf.plan)) break;
    }
    int32_t again = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) again += arena.inside(r.addr(u));
    CHECK(again > 0);
}

TEST(slot_pool_segments_have_their_own_targets) {
    SlotPool s;
    s.init(2, 1);
    s.add_free(0);
    s.add_free(1);
    const int a = s.add_segment(3);
    const int b = s.add_segment(3);
    CHECK_EQ(s.seg_lo(a), 2);
    CHECK_EQ(s.seg_lo(b), 5);
    CHECK_EQ(s.n_free(), 2);                       /* nothing lent yet */
    s.set_target(b, s.seg_lo(b) + 3);
    CHECK_EQ(s.n_free(), 5);
    CHECK(!s.covered(2));
    CHECK(s.covered(5));
    /* Lowest covered slot first, skipping the segment that covers nothing. */
    CHECK_EQ(s.take(), 0);
    CHECK_EQ(s.take(), 1);
    CHECK_EQ(s.take(), 5);
    s.set_target(a, s.seg_lo(a) + 2);
    CHECK_EQ(s.take(), 2);
    /* Shrinking one segment leaves the other alone; the mark stops at what is held. */
    s.set_target(b, s.seg_lo(b));
    CHECK_EQ(s.top_held(b), 5);
    CHECK(s.covered(3));
    CHECK(!s.covered(6));
    s.add_free(5);
    CHECK_EQ(s.top_held(b), -1);
    CHECK_EQ(s.take(), 3);
    CHECK_EQ(s.take(), -1);
}

TEST(mover_prefetches_staged_units_against_the_declared_order) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/false);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK_EQ((int)p.prefetch.size(), 7);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, p, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    /* Each staged unit's pool address before staging. Staging republishes it into a slab slot,
     * so a changed pointer shows a unit landed without exposing the mover's slot table. */
    std::vector<void*> pooled;
    for (const Plan::Prefetch& pf : p.prefetch)
        pooled.push_back(
            m.residency()->lookup(p.units[(size_t)pf.unit].weights.front())->ptr);

    CHECK_OK(m.begin_dispatch(0, nullptr));
    /* At op 0 nothing is due, but the mover starts transfers until it runs out of slots. The ring
     * is two deep, so it gets exactly two ahead and stops; it does not skip the unit it cannot
     * serve, since the schedule is in deadline order. This is pure slot bookkeeping, identical on
     * every backend. */
    CHECK_OK(m.prefetch_to_op(0));
    CHECK_EQ(m.stats().prefetched, (uint64_t)RAD_PLACE_STAGE_RING);
    CHECK(m.stats().starved_slab >= 1);
    for (int i = 0; i < RAD_PLACE_STAGE_RING; ++i) {
        const rad_weight w = p.units[(size_t)p.prefetch[(size_t)i].unit].weights.front();
        CHECK(m.residency()->lookup(w)->ptr != nullptr);
        /* The residency table carries the mover's event, so rad_issue waits on the compute stream,
         * not the host. */
        CHECK(m.residency()->lookup(w)->ready != nullptr);
    }

    /* Walk the ops as a step does and check exact prefetch: each op's bytes land through a
     * two-slot ring. How many dispatches a copy takes depends on the link, so no dispatch index is
     * asserted. */
    ComputeWork work;
    int dispatch = 1;
    for (size_t i = 0; i < p.prefetch.size(); ++i) {
        const rad_weight w = p.units[(size_t)p.prefetch[i].unit].weights.front();
        bool landed = false;
        for (Deadline dl; !landed && !dl.passed();) {
            CHECK_OK(m.begin_dispatch(dispatch++, nullptr));
            work.run(compute);
            CHECK_OK(m.prefetch_to_op(p.prefetch[i].at_op));
            const WeightSlot* s = m.residency()->lookup(w);
            /* In a slab slot (pointer moved off the pool) and settled (copy landed). */
            landed = s->ptr != pooled[i] && s->ready == nullptr;
        }
        CHECK(landed);
    }

    /* Every scheduled unit was staged exactly once: the cursor never rewinds within a step or
     * skips a unit, so ring turnover never re-fetches. */
    CHECK_EQ(m.stats().prefetched, (uint64_t)p.prefetch.size());
    /* Through a two-slot ring it must have starved and recycled; otherwise a mover allocating a
     * slot per unit would also pass. */
    CHECK(m.stats().starved_slab >= 1);
    CHECK((int64_t)p.prefetch.size() > RAD_PLACE_STAGE_RING);

    rad_stream_destroy(compute);
}

/* A routed layer's pooled units staged for one pass (Mover::stage_units). Each copied unit's
 * weights are published at the copy, settled (the caller orders readers with one wait), holding the
 * pool slot's bytes. A unit that does not fit keeps its pool address and is counted; adjacent pool
 * slots share a copy; resident units are never copied; unstaging republishes the pool addresses. */
TEST(a_staged_layer_is_published_at_its_copies_and_unstaged_at_its_pool_slots) {
    HeatFixture hf;
    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    std::vector<int32_t> units, pooled;
    for (int32_t u = 0; u < (int32_t)hf.plan.units.size(); ++u) {
        if (hf.plan.units[(size_t)u].expert < 0) continue;
        units.push_back(u);
        if (hf.plan.units[(size_t)u].tier == Tier::HostPinned) pooled.push_back(u);
    }
    REQUIRE(pooled.size() >= 2);
    REQUIRE(pooled.size() < units.size());           /* some are resident */

    std::vector<void*> before(hf.f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < hf.f.prog.weights.size(); ++w)
        before[w] = m.residency()->lookup(w)->ptr;

    /* Room for every pooled unit but one. */
    const int64_t each = align_up(hf.plan.units[(size_t)pooled[0]].bytes, RAD_ALIGN_UNIT);
    const int64_t cap = each * (int64_t)(pooled.size() - 1);
    std::vector<char> buf((size_t)cap);
    RadEvent done = nullptr;
    CHECK_OK(rad_event_create_local(&done));

    std::vector<int32_t> staged;
    CHECK_OK(m.stage_units(units, buf.data(), cap, nullptr, done, &staged));
    CHECK_EQ(staged.size(), pooled.size() - 1);
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)1);
    CHECK_EQ(m.stats().staged_units, (uint64_t)staged.size());
    CHECK_EQ(m.stats().stage_layers, (uint64_t)1);

    /* Adjacent pool slots go as one copy, so copies equal the runs of staged pool addresses,
     * fewer than the units since a fresh pool hands out slots in order. */
    uint64_t runs = 0;
    const char* run_end = nullptr;
    for (int32_t u : staged) {
        const MoveUnit& mu = hf.plan.units[(size_t)u];
        const char* slot = (const char*)before[mu.weights[0]] - mu.offset[0];
        if (slot != run_end) ++runs;
        run_end = slot + each;
    }
    CHECK_EQ(m.stats().stage_copies, runs);
    CHECK(runs < (uint64_t)staged.size());

    /* The copies run on the mover's stream (a worker thread on the host backend). */
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    std::vector<uint8_t> is_staged(hf.plan.units.size(), 0);
    for (size_t k = 0; k < staged.size(); ++k) {
        const int32_t u = staged[k];
        is_staged[(size_t)u] = 1;
        CHECK(hf.plan.units[(size_t)u].tier == Tier::HostPinned);
        const MoveUnit& mu = hf.plan.units[(size_t)u];
        for (size_t i = 0; i < mu.weights.size(); ++i) {
            const WeightSlot* sl = m.residency()->lookup(mu.weights[i]);
            CHECK(sl->ptr == buf.data() + (int64_t)k * each + mu.offset[i]);
            CHECK(sl->ready == nullptr);
            CHECK(std::memcmp(sl->ptr, before[mu.weights[i]],
                              (size_t)hf.f.prog.weights[mu.weights[i]].logical_bytes) == 0);
        }
    }
    for (int32_t u : units) {
        if (is_staged[(size_t)u]) continue;
        for (rad_weight w : hf.plan.units[(size_t)u].weights)
            CHECK(m.residency()->lookup(w)->ptr == before[w]);
    }

    m.unstage_units(staged);
    for (int32_t u : units)
        for (rad_weight w : hf.plan.units[(size_t)u].weights)
            CHECK(m.residency()->lookup(w)->ptr == before[w]);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_event_destroy(done);
    rad_stream_destroy(compute);
}

/* The prefill stager's order over a pass of two routed layers: at the pass's first op the first
 * layer goes out, and the second too if the ring has room; each layer returns to its pool slots
 * when the next layer starts; nothing is staged in a pass that was not armed.
 *
 * A layer is released when the next starts, not at its own last op: a step longer than the MoE
 * block's row pass issues a layer's ops once per pass, so the last op runs again and would read
 * copies the mover is already overwriting. */
TEST(the_stager_stages_ahead_and_releases_each_layer_when_the_next_starts) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    std::vector<rad_weight> l0, l1;
    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < 8; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                          std::to_string(e) + ".weight", 1 * MiB,
                                          RAD_ACCESS_CONDITIONAL, l, e);
            (l ? l1 : l0).push_back(w);
        }
    f.op("attn0", {}, false);
    f.op("moe0", l0, false);
    f.op("attn1", {}, false);
    f.op("moe1", l1, false);
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 12;
    cfg.host_pool_mib = 64;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    auto pooled_of = [&](const std::vector<rad_weight>& ws) {
        std::vector<rad_weight> p;
        for (rad_weight w : ws)
            if (plan.weight[w].tier == Tier::HostPinned) p.push_back(w);
        return p;
    };
    const std::vector<rad_weight> p0 = pooled_of(l0), p1 = pooled_of(l1);
    REQUIRE(!p0.empty() && !p1.empty());
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    const int64_t half = each * 8;
    /* The ring's region and, above it, its other place (the span's shared level). */
    std::vector<char> region((size_t)(4 * half));
    char* const low = region.data() + 2 * half;
    auto in_ring = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < region.data() || p >= region.data() + 2 * half) return false;
        }
        return true;
    };
    auto in_low = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < low || p >= low + 2 * half) return false;
        }
        return true;
    };
    auto at_home = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) if (m.residency()->lookup(w)->ptr != home[w]) return false;
        return true;
    };

    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), low, 2 * half, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)2);
    CHECK_EQ(st.ring_bytes(), 2 * half);

    RadBatch b{};
    b.n_tok = 8;
    st.set_ring(st.ring_full());
    CHECK(!st.begin_pass(&b));                        /* too few rows */
    b.n_tok = 128;
    st.set_ring(nullptr);
    CHECK(!st.begin_pass(&b));                        /* the level lends the ring's places */
    st.set_ring(st.ring_full());
    CHECK(st.begin_pass(&b));

    bool resync = false;
    CHECK_OK(st.before_op(0, compute, &resync));      /* embed: both layers go out */
    CHECK(resync);
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.after_op(0, compute));
    CHECK_OK(st.before_op(1, compute, &resync));      /* attn0: nothing */
    CHECK(!resync);
    CHECK_OK(st.before_op(2, compute, &resync));      /* moe0: nothing left to stage */
    CHECK(!resync);
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.after_op(2, compute));                /* moe0 is layer 0's last op... */
    CHECK(in_ring(p0));
    CHECK_OK(st.before_op(2, compute, &resync));      /* ...and runs again for the next rows */
    CHECK(in_ring(p0));
    CHECK_OK(st.after_op(2, compute));
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.before_op(3, compute, &resync));      /* attn1 */
    CHECK(in_ring(p0));
    CHECK_OK(st.before_op(4, compute, &resync));      /* moe1: layer 0 is past every pass */
    CHECK(at_home(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.after_op(4, compute));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p1));
    CHECK_EQ(m.stats().stage_layers, (uint64_t)2);
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);

    /* A pass that ends after its first op leaves nothing published at a copy. */
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK(in_ring(p0));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));

    /* At the span's shared level the ring is in its other place: the pass moves it there, and the
     * next pass at the full plan moves it back. */
    st.set_ring(st.ring_low());
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK(resync);
    CHECK(in_low(p0));
    CHECK(in_low(p1));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));
    st.set_ring(st.ring_full());
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK(in_ring(p0));
    CHECK(in_ring(p1));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));
    /* The copies are the mover stream's, and the region is this test's. */
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* The ring over many layers and passes: room for 3.5 layers, six layers per pass, three passes,
 * so placements lap the region and wrap. At every op, layers published at copies do not overlap;
 * at a layer's first op all its pooled units are at copies holding the pool bytes and earlier
 * layers are home; copies run more than one layer ahead when room allows; nothing falls back to a
 * read in place. */
TEST(the_stager_ring_laps_without_overwriting_a_published_layer) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    constexpr int kL = 6, kE = 8;
    std::vector<std::vector<rad_weight>> lw(kL);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kE; ++e)
            lw[(size_t)l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                             std::to_string(e) + ".weight", 1 * MiB,
                                             RAD_ACCESS_CONDITIONAL, l, e));
    for (int l = 0; l < kL; ++l) {
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), lw[(size_t)l], false);
    }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 16;
    cfg.host_pool_mib = 128;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    std::vector<std::vector<rad_weight>> pw(kL);
    int64_t most = 0;
    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    for (int l = 0; l < kL; ++l) {
        for (rad_weight w : lw[(size_t)l])
            if (plan.weight[w].tier == Tier::HostPinned) pw[(size_t)l].push_back(w);
        most = std::max(most, each * (int64_t)pw[(size_t)l].size());
    }
    REQUIRE(most > 0);
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    /* 3.5 times the heaviest layer: three go out at the pass's first op and the fourth wraps past
     * the region's end. */
    const int64_t ring = most * 7 / 2 / (2 * RAD_ALIGN_UNIT) * (2 * RAD_ALIGN_UNIT);
    std::vector<char> region((size_t)ring);
    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), nullptr, ring, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)kL);
    REQUIRE(st.ring_bytes() == ring);

    auto in_ring = [&](rad_weight w) {
        const char* p = (const char*)m.residency()->lookup(w)->ptr;
        return p >= region.data() && p < region.data() + ring;
    };
    /* The layers published at copies; false if any two of their units share a byte. */
    auto published = [&](std::vector<int>* ls) {
        std::vector<std::pair<const char*, const char*>> spans;
        ls->clear();
        for (int l = 0; l < kL; ++l) {
            bool any = false;
            for (rad_weight w : pw[(size_t)l])
                if (in_ring(w)) {
                    const char* p = (const char*)m.residency()->lookup(w)->ptr;
                    spans.emplace_back(p, p + f.prog.weights[w].logical_bytes);
                    any = true;
                }
            if (any) ls->push_back(l);
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); ++i)
            if (spans[i].first < spans[i - 1].second) return false;
        return true;
    };

    RadBatch b{};
    b.n_tok = 128;
    st.set_ring(st.ring_full());
    bool ran_ahead = false;
    std::vector<int> ls;
    for (int pass = 0; pass < 3; ++pass) {
        CHECK(st.begin_pass(&b));
        const int n_ops = 1 + 2 * kL;
        for (int h = 0; h < n_ops; ++h) {
            bool resync = false;
            CHECK_OK(st.before_op(h, compute, &resync));
            CHECK(published(&ls));
            if (h >= 2 && h % 2 == 0) {
                const int l = (h - 2) / 2;            /* moe_l: the layer's first and last op */
                for (int k = 0; k < l; ++k)
                    for (rad_weight w : pw[(size_t)k]) CHECK(!in_ring(w));
                /* The stager's wait orders the op; the test drains the mover's stream to read
                 * the copies. */
                CHECK_OK(rad_stream_sync(m.transfer_stream()));
                for (rad_weight w : pw[(size_t)l]) {
                    CHECK(in_ring(w));
                    CHECK(std::memcmp(m.residency()->lookup(w)->ptr, home[w],
                                      (size_t)f.prog.weights[w].logical_bytes) == 0);
                }
                if (ls.size() > 2) ran_ahead = true;
            }
            CHECK_OK(st.after_op(h, compute));
        }
        CHECK_OK(st.end_pass(compute));
        for (int l = 0; l < kL; ++l)
            for (rad_weight w : pw[(size_t)l]) CHECK(m.residency()->lookup(w)->ptr == home[w]);
    }
    CHECK(ran_ahead);
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* The next step's first layers and the draft head. Six trunk layers plus a declared draft head's
 * seventh, which is never staged. When the next step continues with a prompt, a pass's end stages
 * the next trunk pass's first PrefillStager::lead() layers; a draft pass in between neither arms
 * nor touches them; the next armed trunk pass reads them in place, holding the pool's bytes; and a
 * trunk pass that does not arm returns them home before resolving anything. */
TEST(the_stager_stages_the_next_steps_first_layers_and_never_the_draft_head) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    constexpr int kL = 7, kE = 8;                   /* layer 6 is the draft head's */
    std::vector<std::vector<rad_weight>> lw(kL);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kE; ++e)
            lw[(size_t)l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                             std::to_string(e) + ".weight", 1 * MiB,
                                             RAD_ACCESS_CONDITIONAL, l, e));
    for (int l = 0; l < kL; ++l) {
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), lw[(size_t)l], false);
    }
    f.prog.drafter.kind = RAD_DRAFT_SERIAL;
    f.prog.meta.n_layers = kL - 1;
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 16;
    cfg.host_pool_mib = 128;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    std::vector<std::vector<rad_weight>> pw(kL);
    int64_t most = 0;
    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    for (int l = 0; l < kL; ++l) {
        for (rad_weight w : lw[(size_t)l])
            if (plan.weight[w].tier == Tier::HostPinned) pw[(size_t)l].push_back(w);
        most = std::max(most, each * (int64_t)pw[(size_t)l].size());
    }
    REQUIRE(most > 0);
    for (int l = 0; l < kL; ++l) REQUIRE(!pw[(size_t)l].empty());
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    /* Room for every trunk layer at once, so what the ring holds is set by pacing, not size. */
    const int64_t ring = most * 8 / (2 * RAD_ALIGN_UNIT) * (2 * RAD_ALIGN_UNIT);
    std::vector<char> region((size_t)ring);
    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), nullptr, ring, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)(kL - 1));
    const int lead = (int)PrefillStager::lead();
    REQUIRE(lead >= 1 && lead < kL - 1);

    auto in_ring = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < region.data() || p >= region.data() + ring) return false;
        }
        return true;
    };
    auto at_home = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) if (m.residency()->lookup(w)->ptr != home[w]) return false;
        return true;
    };
    auto copies_hold_the_pool = [&](int l) {
        CHECK_OK(rad_stream_sync(m.transfer_stream()));
        for (rad_weight w : pw[(size_t)l])
            if (std::memcmp(m.residency()->lookup(w)->ptr, home[w],
                            (size_t)f.prog.weights[w].logical_bytes) != 0) return false;
        return true;
    };
    /* A whole trunk pass, checking at each layer's MoE that it reads copies and that the head's
     * layer stays home. */
    const int n_ops = 1 + 2 * (kL - 1);
    auto run_pass = [&]() {
        for (int h = 0; h < n_ops; ++h) {
            bool resync = false;
            CHECK_OK(st.before_op(h, compute, &resync));
            if (h >= 2 && h % 2 == 0) {
                const int l = (h - 2) / 2;
                CHECK(in_ring(l));
                CHECK(copies_hold_the_pool(l));
            }
            CHECK(at_home(kL - 1));
            CHECK_OK(st.after_op(h, compute));
        }
    };

    RadBatch b{};
    b.n_tok = 128;
    st.set_ring(st.ring_full());

    /* No next step: the end leaves everything home. staged() is read after the pass ends, so it
     * outlives end_pass. */
    CHECK(!st.staged());
    st.set_next_prefill(false);
    CHECK(st.begin_pass(&b));
    run_pass();
    CHECK_OK(st.end_pass(compute));
    CHECK(st.staged());
    for (int l = 0; l < kL; ++l) CHECK(at_home(l));

    /* The next step continues: the first `lead` layers go out at the end. */
    st.set_next_prefill(true);
    CHECK(st.begin_pass(&b));
    run_pass();
    CHECK_OK(st.end_pass(compute));
    for (int l = 0; l < kL; ++l) CHECK(l < lead ? in_ring(l) : at_home(l));
    for (int l = 0; l < lead; ++l) CHECK(copies_hold_the_pool(l));

    /* A draft pass neither arms nor returns them, and leaves staged() as the trunk pass set it. */
    RadBatch d = b;
    d.draft_pass = -1;
    CHECK(!st.begin_pass(&d));
    CHECK(st.staged());
    for (int l = 0; l < lead; ++l) CHECK(in_ring(l));

    /* The next armed trunk pass reads them in place and does not restage them at its first op. */
    const uint64_t layers_before = m.stats().stage_layers;
    st.set_next_prefill(true);
    CHECK(st.begin_pass(&b));
    {
        bool resync = false;
        CHECK_OK(st.before_op(0, compute, &resync));
        CHECK_EQ(m.stats().stage_layers, layers_before);
        CHECK_OK(st.after_op(0, compute));
    }
    for (int h = 1; h < n_ops; ++h) {
        bool resync = false;
        CHECK_OK(st.before_op(h, compute, &resync));
        if (h >= 2 && h % 2 == 0) {
            const int l = (h - 2) / 2;
            CHECK(in_ring(l));
            CHECK(copies_hold_the_pool(l));
        }
        CHECK_OK(st.after_op(h, compute));
    }
    CHECK_OK(st.end_pass(compute));
    for (int l = 0; l < kL; ++l) CHECK(l < lead ? in_ring(l) : at_home(l));

    /* A trunk pass too small to arm returns everything home before resolving a table. */
    b.n_tok = 8;
    CHECK(!st.begin_pass(&b));
    CHECK(!st.staged());
    for (int l = 0; l < kL; ++l) CHECK(at_home(l));
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* The wrap at a chunk boundary, and what it waits on.
 *
 * Four trunk layers and one op past them: the first op a pass issues after its last routed layer
 * is where the next chunk's first layers go out, while the pass is still open. The card is held at
 * a gate behind the second layer's last op, so the `free` events of that layer and of the two
 * after it stay pending, and the ring is three layers wide, which is room enough for the three
 * early places and means the first of them laps a place this pass has already given back. So:
 *
 *   while the card is held, none of the three copies has landed -- the wrap ordered them behind
 *   the reads of the bytes it is taking, not behind anything it invented, and a copy waiting on a
 *   read the card has not retired blocks the transfer stream it shares with the other two; and
 *   opening the gate lands all three with end_pass still to be called, which is the whole point:
 *   the next chunk's weights crossed the link before this chunk's pass was retired.
 *
 * The next chunk adopts all three without staging a byte again. */
TEST(the_stager_wraps_the_ring_at_the_chunk_boundary_behind_the_reads_it_laps) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    constexpr int kL = 4, kE = 8;
    std::vector<std::vector<rad_weight>> lw(kL);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kE; ++e)
            lw[(size_t)l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                             std::to_string(e) + ".weight", 1 * MiB,
                                             RAD_ACCESS_CONDITIONAL, l, e));
    for (int l = 0; l < kL; ++l) {
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), lw[(size_t)l], false);
    }
    const int h_tail = 1 + 2 * kL;               /* the pass's op past the last routed layer */
    f.op("read_state", {}, false);
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 16;
    cfg.host_pool_mib = 128;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    std::vector<std::vector<rad_weight>> pw(kL);
    int64_t most = 0;
    for (int l = 0; l < kL; ++l) {
        for (rad_weight w : lw[(size_t)l])
            if (plan.weight[w].tier == Tier::HostPinned) pw[(size_t)l].push_back(w);
        REQUIRE(!pw[(size_t)l].empty());
        most = std::max(most, each * (int64_t)pw[(size_t)l].size());
    }
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    /* Three of the heaviest layer: enough that the wrap stages all lead() early layers, and little
     * enough that the first of them wraps past the region's end and onto bytes this pass used. */
    const int64_t ring = most * 3;
    std::vector<char> region((size_t)ring);
    /* Stamp the pool, so a copy's bytes say which layer they came from, and clear the region, so
     * "has this copy run" is something the test can read back rather than infer. */
    for (int l = 0; l < kL; ++l)
        for (size_t i = 0; i < pw[(size_t)l].size(); ++i)
            std::memset(home[pw[(size_t)l][i]], (int)(0x40 + 7 * l + i),
                        (size_t)f.prog.weights[pw[(size_t)l][i]].logical_bytes);
    std::memset(region.data(), 0, (size_t)ring);
    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), nullptr, ring, 64));
    REQUIRE(st.enabled());
    REQUIRE(st.n_layers() == (size_t)kL);
    REQUIRE(st.ring_bytes() == ring);
    const int lead = (int)PrefillStager::lead();
    REQUIRE(lead == kL - 1);                     /* so every early layer has a gate to wait on */

    auto in_ring = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < region.data() || p >= region.data() + ring) return false;
        }
        return true;
    };
    auto at_home = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) if (m.residency()->lookup(w)->ptr != home[w]) return false;
        return true;
    };
    auto copies_hold_the_pool = [&](int l) {
        for (rad_weight w : pw[(size_t)l])
            if (std::memcmp(m.residency()->lookup(w)->ptr, home[w],
                            (size_t)f.prog.weights[w].logical_bytes) != 0) return false;
        return true;
    };

    RadBatch b{};
    b.n_tok = 128;
    void* gate = nullptr;                        /* holds the card behind layer 1's last op */
    st.set_ring(st.ring_full());
    st.set_next_prefill(true);
    CHECK(st.begin_pass(&b));

    bool resync = false;
    for (int h = 0; h <= 2 * kL; ++h) {
        resync = false;
        CHECK_OK(st.before_op(h, compute, &resync));
        if (h >= 2 && h % 2 == 0) {
            const int l = (h - 2) / 2;
            CHECK(in_ring(l));                   /* staged ahead, and read where it was copied */
            CHECK_OK(rad_stream_sync(m.transfer_stream()));
            CHECK(copies_hold_the_pool(l));
        }
        if (h == 4) {
            /* Behind this point the card does not reach, until the test opens the gate: every free
             * event recorded from here on, and every copy that waits on one, stays pending. */
            CHECK_OK(rad_stream_gate(compute, &gate));
            REQUIRE(gate);
        }
        CHECK_OK(st.after_op(h, compute));
    }

    /* Poison the region: the only writes it can take from here are the wrap's, since every copy
     * this pass issued has landed and the card is held behind the reads. Whatever is at the early
     * layers' addresses when the wrap returns was written by the wrap or not at all. */
    std::memset(region.data(), 0xAB, (size_t)ring);

    /* The op past the last routed layer: the wrap goes out here, and it gives the last layer's
     * place back rather than carry it to the end of the pass, so the tables change. */
    const uint64_t staged_before = m.stats().stage_layers;
    resync = false;
    CHECK_OK(st.before_op(h_tail, compute, &resync));
    CHECK(resync);
    int n_early = 0;
    while (n_early < kL && in_ring(n_early)) ++n_early;
    CHECK_EQ(n_early, lead);
    CHECK_EQ(m.stats().stage_layers, staged_before + (uint64_t)n_early);
    CHECK(at_home(kL - 1));                      /* the last layer's place went back early */
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);

    /* Nothing the wrap staged has landed: its first copy waits on a read the held card has not
     * retired, and the other two queue behind it on the mover's stream, in order. An event recorded
     * on that stream now stays pending for as long as the card is held -- and the only way it stops
     * pending is a copy that waited on nothing and landed at once, which is what this waits for.
     * The region says the same thing in the copies' own bytes. */
    RadEvent probe = nullptr;
    CHECK_OK(rad_event_create(&probe));
    CHECK_OK(rad_event_record(probe, m.transfer_stream()));
    const auto stop = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    while (rad_event_query(probe) == 0 && std::chrono::steady_clock::now() < stop)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK_EQ(rad_event_query(probe), 0);
    rad_event_destroy(probe);
    for (int l = 0; l < n_early; ++l) CHECK(!copies_hold_the_pool(l));

    /* The card reaches the reads, and the three layers land -- with end_pass still to be called.
     * If the wrap had waited on the pass's end instead, none of this would hold yet. */
    CHECK_OK(rad_gate_open(gate));
    gate = nullptr;
    CHECK_OK(rad_stream_sync(compute));
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    for (int l = 0; l < n_early; ++l) {
        CHECK(in_ring(l));
        CHECK(copies_hold_the_pool(l));
    }

    /* The end retires what the pass staged and leaves the next chunk's layers published. */
    CHECK_OK(st.end_pass(compute));
    for (int l = 0; l < n_early; ++l) CHECK(in_ring(l));
    for (int l = n_early; l < kL; ++l) CHECK(at_home(l));

    /* The next chunk adopts them: its first op stages none of them again. */
    const uint64_t adopted = m.stats().stage_layers;
    st.set_next_prefill(true);
    CHECK(st.begin_pass(&b));
    resync = false;
    CHECK_OK(st.before_op(0, compute, &resync));
    CHECK_EQ(m.stats().stage_layers, adopted);
    for (int l = 0; l < n_early; ++l) CHECK(in_ring(l));

    /* And the same boundary with the last layer issued on the other lane. Only the join before
     * end_pass says what is behind a read on that stream, so the wrap leaves that place published
     * and stages what fits around it -- and end_pass, which runs behind the join, takes it home. */
    RadStream lane = nullptr;
    CHECK_OK(rad_stream_create(&lane, 0));
    for (int h = 1; h <= 2 * kL; ++h) {
        const RadStream s = h >= 2 * kL - 1 ? lane : compute;    /* the last layer on lane 1 */
        CHECK_OK(st.before_op(h, s, &resync));
        CHECK_OK(st.after_op(h, s));
    }
    CHECK(in_ring(kL - 1));
    resync = false;
    CHECK_OK(st.before_op(h_tail, compute, &resync));
    CHECK(in_ring(kL - 1));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(kL - 1));
    CHECK_OK(rad_stream_sync(lane));
    rad_stream_destroy(lane);

    /* A pass too small to arm takes the early layers home with it: nothing the wrap published for
     * a pass that never comes stays at a copy. */
    b.n_tok = 8;
    CHECK(!st.begin_pass(&b));
    for (int l = 0; l < kL; ++l) CHECK(at_home(l));
    CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    if (gate) rad_gate_open(gate);
    rad_stream_destroy(compute);
}

/* Three trunk layers, which is exactly lead(): a pass that routes that many stages all of them for
 * the next pass, so no later layer of that pass is left to give their places back. Adoption has to
 * clear `early`, or the wrap and end_pass skip them for good, the ring fills and staging stops with
 * no error at all -- a pass would stage its own layers once and 0 from then on. */
TEST(the_stager_gives_back_the_places_it_adopts_from_the_previous_pass) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    constexpr int kL = 3, kE = 8;
    std::vector<std::vector<rad_weight>> lw(kL);
    for (int l = 0; l < kL; ++l)
        for (int e = 0; e < kE; ++e)
            lw[(size_t)l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                             std::to_string(e) + ".weight", 1 * MiB,
                                             RAD_ACCESS_CONDITIONAL, l, e));
    for (int l = 0; l < kL; ++l) {
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), lw[(size_t)l], false);
    }
    const int h_tail = 1 + 2 * kL;               /* the pass's op past the last routed layer */
    f.op("read_state", {}, false);
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 16;
    cfg.host_pool_mib = 128;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    REQUIRE(plan.dynamic);

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    const int64_t each = align_up(1 * MiB, RAD_ALIGN_UNIT);
    std::vector<std::vector<rad_weight>> pw(kL);
    int64_t most = 0;
    for (int l = 0; l < kL; ++l) {
        for (rad_weight w : lw[(size_t)l])
            if (plan.weight[w].tier == Tier::HostPinned) pw[(size_t)l].push_back(w);
        REQUIRE(!pw[(size_t)l].empty());
        most = std::max(most, each * (int64_t)pw[(size_t)l].size());
    }
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        home[w] = m.residency()->lookup(w)->ptr;

    /* Exactly one pass of room: the next pass can only stage at all if the last one's places came
     * back before it. */
    const int64_t ring = most * kL;
    std::vector<char> region((size_t)ring);
    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), nullptr, ring, 64));
    REQUIRE(st.enabled());
    REQUIRE(st.n_layers() == (size_t)kL);
    const int lead = (int)PrefillStager::lead();
    REQUIRE(lead == kL);                          /* so a pass stages every layer early */

    auto in_ring = [&](int l) {
        for (rad_weight w : pw[(size_t)l]) {
            const char* p = (const char*)m.residency()->lookup(w)->ptr;
            if (p < region.data() || p >= region.data() + ring) return false;
        }
        return true;
    };

    RadBatch b{};
    b.n_tok = 128;
    uint64_t prev = m.stats().stage_layers;
    for (int pass = 0; pass < 4; ++pass) {
        st.set_ring(st.ring_full());
        st.set_next_prefill(true);
        CHECK(st.begin_pass(&b));
        bool resync = false;
        for (int h = 0; h <= 2 * kL; ++h) {
            CHECK_OK(st.before_op(h, compute, &resync));
            CHECK_OK(st.after_op(h, compute));
        }
        CHECK_OK(st.before_op(h_tail, compute, &resync));
        CHECK_OK(st.end_pass(compute));
        const uint64_t now = m.stats().stage_layers;
        /* The first pass stages its own layers and the next pass's (2 * lead); every pass after it
         * stages the pass after it, which is the proof its adopted places were given back. */
        CHECK_EQ(now - prev, (uint64_t)(pass == 0 ? 2 * lead : lead));
        prev = now;
        CHECK_EQ(m.stats().stage_overflow, (uint64_t)0);
        for (int l = 0; l < kL; ++l) CHECK(in_ring(l));
        CHECK_OK(rad_stream_sync(compute));
    }

    /* A pass too small to arm takes the last pass's places home with it. */
    b.n_tok = 8;
    CHECK(!st.begin_pass(&b));
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* --no-prefill-overlap is the A/B's control arm: the engine asks this with the parsed flag, so a
 * refactor that drops the flag from the decision leaves the control arm staging anyway. */
TEST(prefill_overlap_stages_only_a_continuing_prompt_with_the_flag_on) {
    CHECK(prefill_overlap_stages(true, RAD_PHASE_PREFILL, true));
    CHECK(prefill_overlap_stages(true, RAD_PHASE_MIXED, true));
    CHECK(!prefill_overlap_stages(false, RAD_PHASE_PREFILL, true));
    CHECK(!prefill_overlap_stages(false, RAD_PHASE_MIXED, true));
    CHECK(!prefill_overlap_stages(true, RAD_PHASE_DECODE, true));
    CHECK(!prefill_overlap_stages(true, RAD_PHASE_PREFILL, false));
}

TEST(mover_reserves_from_the_weight_pool_rather_than_beside_it) {
    /* The production path. Pools has already allocated the weights pool as one block; allocating
     * beside it would spend the budget twice and surface later as an unrelated out-of-memory.
     * Reserving from Pools::weights() also means a promotion can never reach a KV block. */
    HeatFixture hf;
    Pools pools;
    Config c = hf.cfg;
    c.vram_kv_mib = 16;
    c.gpu_headroom_mib = 1;
    CHECK_OK(pools.configure(c, /*vram_capacity=*/0, &rad_malloc_allocator(),
                             &rad_malloc_allocator()));

    const int64_t weights_before = pools.weights().used();
    const int64_t host_before = pools.host_pinned().used();

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc, &pools));

    /* Every byte the mover took came from the pools' budgets, none from KV. */
    CHECK(pools.weights().used() > weights_before);
    CHECK(pools.host_pinned().used() > host_before);
    CHECK_EQ(pools.kv().used(), (int64_t)0);
    CHECK(pools.weights().ok());
    CHECK(pools.host_pinned().ok());
    for (rad_weight w = 1; (size_t)w < hf.f.prog.weights.size(); ++w)
        CHECK(m.residency()->lookup(w)->ptr != nullptr);
}

TEST(mover_declines_to_move_under_a_static_plan) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    c.deterministic = true;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(f.prog, p, mc));
    Swap s{0, 1};
    CHECK_OK(m.issue(s, nullptr));
    CHECK_EQ(m.stats().promotions, (uint64_t)0);
    CHECK_EQ(m.stats().h2d_bytes, (uint64_t)0);
}

TEST(the_plan_report_names_every_class_once) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    const std::string r = p.report(f.prog);
    /* Eight experts, one line: the class key collapses indexed weights in the report. */
    size_t n = 0, at = 0;
    const std::string key = "blk.*.ffn_gate_exps.*.weight";
    while ((at = r.find(key, at)) != std::string::npos) { ++n; at += key.size(); }
    CHECK_EQ((int)n, 1);
    CHECK(r.find("token_embd.weight") != std::string::npos);
    CHECK(r.find("zero-copy") != std::string::npos);
    CHECK(r.find("expert_tiered") != std::string::npos);
}

/* ================================================================== the planner's own helpers
 *
 * Small functions the planner is built from, where two components must agree or a budget
 * overruns a pool. The mover, the report and the budget resolver call them instead of rederiving
 * the answer, so they are tested directly.
 */

TEST(the_strategy_names_round_trip_and_an_unknown_one_is_refused) {
    for (Strategy s : { Strategy::AllVram, Strategy::LayerOffload, Strategy::ExpertTiered,
                        Strategy::Auto }) {
        Strategy back = Strategy::Auto;
        CHECK_OK(strategy_parse(strategy_name(s), &back));
        CHECK((int)back == (int)s);
    }
    /* These spellings are the --placement flag's legal set, so they are pinned literally. */
    CHECK_EQ(std::string(strategy_name(Strategy::AllVram)), std::string("all_vram"));
    CHECK_EQ(std::string(strategy_name(Strategy::LayerOffload)), std::string("layer_offload"));
    CHECK_EQ(std::string(strategy_name(Strategy::ExpertTiered)), std::string("expert_tiered"));
    CHECK_EQ(std::string(strategy_name(Strategy::Auto)), std::string("auto"));

    /* *out is untouched on a refusal, so a caller ignoring the status keeps its prior value. */
    Strategy keep = Strategy::ExpertTiered;
    CHECK_EQ(strategy_parse("layerOffload", &keep), RAD_E_INVAL);
    CHECK((int)keep == (int)Strategy::ExpertTiered);
    CHECK_EQ(strategy_parse("", &keep), RAD_E_INVAL);
    CHECK_EQ(strategy_parse(nullptr, &keep), RAD_E_INVAL);
    CHECK((int)keep == (int)Strategy::ExpertTiered);
    CHECK_EQ(strategy_parse("all_vram", nullptr), RAD_E_INVAL);
}

/* The strongest access class in a group decides the group's, so this ordering keeps the heat
 * engine from demoting what the next token needs. VOCAB ranks equal to per-token: lm_head and the
 * embedding are read every token, and are a separate class only for their size and the gather. */
TEST(access_strength_orders_the_classes_and_is_pessimistic_about_an_unknown_one) {
    CHECK_EQ(access_strength(RAD_ACCESS_PER_TOKEN), access_strength(RAD_ACCESS_VOCAB));
    CHECK(access_strength(RAD_ACCESS_PER_TOKEN) > access_strength(RAD_ACCESS_PER_REQUEST));
    CHECK(access_strength(RAD_ACCESS_PER_REQUEST) > access_strength(RAD_ACCESS_CONDITIONAL));
    CHECK(access_strength(RAD_ACCESS_CONDITIONAL) > access_strength(RAD_ACCESS_RARE));
    CHECK_EQ(access_strength(RAD_ACCESS_RARE), 0);

    /* An unknown access class (from a newer plugin) is treated as per-token, not as rare as a
       zero default would. */
    CHECK_EQ(access_strength(9999), access_strength(RAD_ACCESS_PER_TOKEN));
    CHECK_EQ(access_strength(-1), access_strength(RAD_ACCESS_PER_TOKEN));
}

/* The stored layout is what occupies a tier. A kernel that re-layouts its weights may store a
 * different byte count than the declared shape, so budgeting the logical size would short the
 * pool. */
TEST(weight_bytes_prefers_the_stored_layout_then_the_logical_then_the_shape) {
    WeightInfo w;
    w.decl.dtype = RAD_BF16;
    w.decl.rank = 2;
    w.decl.shape[0] = 4;
    w.decl.shape[1] = 8;
    /* Nothing resolved yet: the declared dtype and shape, as in planner unit-test fixtures. */
    CHECK_EQ(weight_bytes(w), (int64_t)(4 * 8 * 2));

    w.logical_bytes = 100;
    CHECK_EQ(weight_bytes(w), 100ll);
    w.stored_bytes = 160;                 /* the layout pass re-laid this one out */
    CHECK_EQ(weight_bytes(w), 160ll);

    /* A rank-0 weight has no elements, not one. */
    WeightInfo z;
    z.decl.dtype = RAD_BF16;
    z.decl.rank = 0;
    CHECK_EQ(weight_bytes(z), 0ll);
}

/* The footprint must agree with the planner: the VRAM budget resolver calls it before a Planner
 * exists and gives the planner a budget computed from it. It must reproduce the unit grouping, the
 * per-unit alignment and the mapped-tier pin. */
TEST(the_weight_footprint_splits_statics_from_the_elastic_expert_plane) {
    Fixture f;
    std::vector<rad_weight> experts;
    build_moe(f, /*n=*/4, /*each=*/2 * MiB, &experts);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);

    /* One model-level weight and four experts: five units. */
    CHECK_EQ(fp.n_units, 5ll);
    CHECK_EQ(fp.statics, 1 * MiB);
    CHECK_EQ(fp.experts, 4 * 2 * MiB);
    CHECK_EQ(fp.mapped, 0ll);

    /* Every weight is in exactly one bucket; one counted twice or not at all would silently skew
       the budget by its size. */
    int64_t declared = 0;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        declared += weight_bytes(f.prog.weights[w]);
    CHECK_EQ(fp.statics + fp.experts + fp.mapped + fp.hosted, declared);
}

TEST(weights_in_one_layer_are_one_unit_and_are_aligned_once_each) {
    /* Two weights in each of two layers, at a size not a multiple of the unit alignment, so the
       padding shows in the total. */
    Fixture f;
    const int64_t odd = 1000;
    for (int l = 0; l < 2; ++l) {
        const rad_weight a = f.weight("blk." + std::to_string(l) + ".attn.weight", odd,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        const rad_weight b = f.weight("blk." + std::to_string(l) + ".ffn.weight", odd,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        f.op("gemm", { a, b }, false);
    }

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.n_units, 2ll);              /* the declared group is the unit, not the weight */

    /* One align_up inside the unit and one for the unit, as Mover::init spends the slab. Raw byte
       sums would fall short by an alignment per unit and overrun the weights pool on large MoEs. */
    const int64_t per_unit = align_up(align_up(odd, RAD_ALIGN_UNIT) + odd, RAD_ALIGN_UNIT);
    CHECK_EQ(fp.statics, 2 * per_unit);
    CHECK(fp.statics > 4 * odd);
}

/* A weight no device kernel can read never enters a pool, and its whole unit goes with it, since
 * a unit is what moves. Counting it against VRAM would reserve card memory for something read only
 * in the container's mapping. */
TEST(a_weight_with_no_device_band_takes_its_whole_unit_to_the_mapped_tier) {
    Fixture f;
    /* Layer 0 has two weights; the op reading one of them has only a host kernel. */
    const rad_weight a = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight b = f.weight("blk.0.b.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    f.op("gemm", { a }, /*host_kernel=*/false);

    f.host_op("host_op", { b });

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.n_units, 1ll);
    CHECK_EQ(fp.statics, 0ll);
    CHECK_EQ(fp.mapped, 6 * MiB);          /* both weights: the unit is what moves */
}

/* On the host backend the weights pool is host memory, so a host-only weight stays in the
 * container's mapping only if it is stored as its kernel reads it. One its host kernel widens at
 * load takes a copy in the pool instead, and the footprint counts it the same way. */
TEST(on_the_host_backend_a_converted_host_only_weight_takes_a_pool_copy) {
    Fixture f;
    const rad_weight direct = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight widened = f.weight("output_norm.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, -1, -1);
    f.prog.weights[widened].widen = true;
    for (rad_weight w : { direct, widened }) f.host_op("host_op", { w });

    WeightFootprint card, host;
    weight_footprint(f.prog, &card, TablePlacement::Vram, TablePlacement::Disk, false, true);
    weight_footprint(f.prog, &host, TablePlacement::Vram, TablePlacement::Disk, true, true);
    CHECK_EQ(card.mapped, 6 * MiB);          /* on a card both stay in the mapping */
    CHECK_EQ(card.statics, 0ll);
    CHECK_EQ(host.mapped, 4 * MiB);
    CHECK_EQ(host.statics, 2 * MiB);         /* the widened one is the pool's */

    Config cfg = base_config();
    cfg.vram_weights_mib = 8;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    in.host_backend = true;
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.weight[direct].tier == Tier::Mapped);
    CHECK(plan.weight[widened].tier == Tier::VRAM);   /* the weights pool, host memory here */

    /* A checkpoint served directly has no container mapping, so on the host backend both take a
     * pool copy and the footprint counts both, or the planner would overfill the pool. */
    WeightFootprint ckpt;
    weight_footprint(f.prog, &ckpt, TablePlacement::Vram, TablePlacement::Disk, true, false);
    CHECK_EQ(ckpt.mapped, 0ll);
    CHECK_EQ(ckpt.statics, 6 * MiB);
    Plan cplan;
    PlannerInput cin{&f.prog, &cfg, 0, 1, nullptr, 0};
    cin.host_backend = true;
    cin.container = false;
    CHECK_OK(Planner(cin).plan(&cplan));
    CHECK(cplan.weight[direct].tier == Tier::VRAM);
    CHECK(cplan.weight[widened].tier == Tier::VRAM);
}

/* On the host backend an op may read a weight from the container's mapping beside one its kernel
 * converted into the pool (Flash-Next's hc_read: a gain read as stored, mixing matrices joined at
 * load). Both are host memory and the op runs its host kernel, so the plan reports no site
 * disagreement, which on a card would mean the op cannot run. */
TEST(on_the_host_backend_an_op_reads_a_mapped_and_a_pooled_weight_together) {
    Fixture f;
    const rad_weight gain = f.weight("blk.0.hc_norm.weight", 1 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight mix = f.weight("blk.0.hc_down.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, -1, -1);
    f.prog.weights[mix].widen = true;
    f.host_op("hc_read", { gain, mix });

    Config cfg = base_config();
    cfg.vram_weights_mib = 8;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    in.host_backend = true;
    int rc = 0;
    const std::string err = stderr_of([&] { rc = Planner(in).plan(&plan); });
    CHECK_OK(rc);
    CHECK(plan.weight[gain].tier == Tier::Mapped);
    CHECK(plan.weight[mix].tier == Tier::VRAM);   /* the weights pool, host memory here */
    for (const std::string& n : plan.notes) CHECK(!has(n, "no site serves both"));
    CHECK(!has(err, "mixes a mapped weight"));
}

/* A unit that cannot be read from the mapping is one misfit, however many weights it groups. */
TEST(a_mapped_unit_that_cannot_be_read_in_place_is_counted_once) {
    Fixture f;
    const rad_weight a = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight b = f.weight("blk.0.b.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    f.prog.weights[a].widen = true;
    f.prog.weights[b].widen = true;
    f.host_op("host_op", { a, b });

    Config cfg = base_config();
    cfg.vram_weights_mib = 8;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    (void)Planner(in).plan(&plan);
    CHECK(plan.weight[a].tier == Tier::Mapped);
    REQUIRE_EQ(plan.unfit.size(), (size_t)1);
    CHECK_EQ(plan.unfit[0].count, 1ll);
    CHECK_EQ(plan.unfit[0].bytes, plan.units[(size_t)plan.weight[a].unit].bytes);
}

/* --embedding-placement fixture: the embedding is read by `embed_lookup`, as every arch declares
 * it, and the head by a GEMM; the flag moves only the former. */
void build_dense_lookup(Fixture& f, int layers, int64_t per_layer, int64_t emb = 1 * MiB) {
    const rad_weight m = f.weight("token_embd.weight", emb, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup", {m}, false);
    for (int l = 0; l < layers; ++l) {
        const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_down.weight", per_layer,
                                      RAD_ACCESS_PER_TOKEN, l, -1);
        f.op("gemm", {w}, false);
    }
    const rad_weight o = f.weight("output.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("logits_gemm", {o}, false);
}

/* An n-gram table: model-level, read only by `embed_lookup_q`, which only a host kernel serves. */
rad_weight add_ngram_table(Fixture& f, int64_t bytes) {
    const rad_weight t = f.weight("ngram.weight", bytes, RAD_ACCESS_VOCAB, -1, -1);
    f.host_op("embed_lookup_q", { t });
    return t;
}

/* Tables are identified by their readers: a model-level weight read only by embed_lookup is the
 * embedding, one read only by embed_lookup_q an n-gram table. A head read by a GEMM is neither, and
 * neither is a table also read by a GEMM (a tied head would stream over the link every step).
 * Placed in RAM, the embedding moves from statics to hosted, since the resolver sizes a derived
 * weight budget from the statics. */
TEST(the_placement_flags_find_their_tables_by_what_reads_them) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_ngram_table(f, 8 * MiB);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 1 * MiB);
    CHECK_EQ(fp.ngram, 8 * MiB);
    CHECK(!fp.ngram_device);
    CHECK_EQ(fp.hosted, 0ll);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 2 * MiB);
    CHECK_EQ(fp.mapped, 8 * MiB);

    weight_footprint(f.prog, &fp, TablePlacement::Ram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 1 * MiB);
    CHECK_EQ(fp.hosted, 1 * MiB);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 1 * MiB);

    CHECK(is_embedding_table(f.prog, 1));
    CHECK(!is_embedding_table(f.prog, 2));                 /* a layer's GEMM operand */
    CHECK(!is_embedding_table(f.prog, 4));                 /* the head */
    CHECK(is_ngram_table(f.prog, 5));
    CHECK(!is_ngram_table(f.prog, 1));

    /* A head tied to the table: one weight read by both the lookup and the logits GEMM. */
    Fixture t;
    const rad_weight m = t.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    t.op("embed_lookup", {m}, false);
    t.op("logits_gemm", {m}, false);
    CHECK(!is_embedding_table(t.prog, m));
    weight_footprint(t.prog, &fp, TablePlacement::Ram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 0ll);
    CHECK_EQ(fp.hosted, 0ll);
}

TEST(embedding_ram_goes_to_the_pinned_pool_read_in_place_and_costs_no_vram) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 9;           /* exactly the layers and the head, not the embedding */
    cfg.host_pool_mib = 2;
    cfg.embedding_placement = TablePlacement::Ram;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));

    const rad_weight emb = 1;           /* declared first */
    CHECK(plan.weight[emb].tier == Tier::HostPinned);
    CHECK(plan.weight[emb].site == Site::DeviceZeroCopy);
    CHECK(!plan.units[(size_t)plan.weight[emb].unit].movable);
    CHECK_EQ(plan.embedding_host_bytes, 1 * MiB);
    for (rad_weight w = 2; (size_t)w < f.prog.weights.size(); ++w)
        CHECK(plan.weight[w].tier == Tier::VRAM);

    /* Vram keeps it with the other weights. */
    cfg.embedding_resolved = TablePlacement::Vram;
    cfg.vram_weights_mib = 10;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK(plan2.weight[emb].tier == Tier::VRAM);
    CHECK_EQ(plan2.embedding_host_bytes, 0ll);
}

/* The embedding gets its own room: the resolver sizes Config::embedding_host_mib on top of
 * --host-pool-mib, so a dense model with no pool still places the table, and the experts keep all
 * of --host-pool-mib after the table is pinned. */
TEST(embedding_ram_brings_its_own_room_beside_the_host_pool) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 9;
    cfg.host_pool_mib = 0;              /* no weights pool */
    cfg.embedding_host_mib = 1;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.weight[1].tier == Tier::HostPinned);
    CHECK_EQ(plan.host_budget, 0ll);

    cfg.host_pool_mib = 3;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK(plan2.weight[1].tier == Tier::HostPinned);
    CHECK_EQ(plan2.host_budget, 3 * MiB);  /* --host-pool-mib, whole */
}

/* An encoder tower as architectures declare one: a stem and a merger at model level, one layer
 * unit per block numbered past the text layers, every weight read once per request. */
void add_tower(Fixture& f, int first_layer, int blocks, int64_t per_block) {
    const rad_weight s = f.weight("v.patch_embd.weight", 1 * MiB, RAD_ACCESS_PER_REQUEST, -1, -1);
    f.op("vit_gemm", {s}, false);
    for (int b = 0; b < blocks; ++b) {
        const rad_weight w = f.weight("v.blk." + std::to_string(b) + ".attn_qkv.weight", per_block,
                                      RAD_ACCESS_PER_REQUEST, first_layer + b, -1);
        f.op("vit_gemm", {w}, false);
    }
    const rad_weight m = f.weight("v.merger.weight", 1 * MiB, RAD_ACCESS_PER_REQUEST, -1, -1);
    f.op("vit_gemm", {m}, false);
}

/* The tower is identified by its access class at every level: the footprint counts every unit
 * whose weights are all per-request, model-level and per-block alike, within the statics (the
 * resolver subtracts it from them). */
TEST(the_footprint_counts_an_encoder_tower_at_every_level) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.tower, 1 * MiB + 3 * 2 * MiB + 1 * MiB);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 2 * MiB + fp.tower);

    Fixture t;
    build_dense_lookup(t, /*layers=*/2, /*per_layer=*/4 * MiB);
    weight_footprint(t.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.tower, 0ll);
}

/* A tower's own lookup table: model-level, read once a request and only by embed_lookup, like
 * Gemma 4's patch-position tables. Both the embedding's test and the tower's take it. */
rad_weight add_tower_lookup(Fixture& f, int64_t bytes) {
    const rad_weight t = f.weight("v.pos_x.weight", bytes, RAD_ACCESS_PER_REQUEST, -1, -1);
    f.op("embed_lookup", {t}, false);
    return t;
}

/* Both tables in RAM: pin_embedding pins the tower's lookup table and pin_towers skips it, so the
 * split Engine::plan makes (the statics less the embedding and less tower_pin_bytes) is what the
 * planner keeps on the card, and every text weight stays there. Taking the table out twice
 * leaves the split short by it, and auto offloads the last layers in issue order. With the
 * embedding on the card the table stays with the tower. */
TEST(a_tower_lookup_table_the_embedding_pins_leaves_the_card_once) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    const rad_weight pos = add_tower_lookup(f, 1 * MiB);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.embedding, 1 * MiB + 1 * MiB);                      /* the token table and pos */
    CHECK_EQ(fp.tower, 1 * MiB + 3 * 2 * MiB + 1 * MiB + 1 * MiB);  /* stem, blocks, merger, pos */
    CHECK_EQ(fp.tower_lookup, 1 * MiB);
    CHECK_EQ(tower_pin_bytes(fp, TablePlacement::Vram), fp.tower);

    const int64_t emb = fp.embedding, towers = tower_pin_bytes(fp, TablePlacement::Ram);
    Config cfg = base_config();
    cfg.placement = "auto";
    cfg.vram_weights_mib = (fp.statics - emb - towers) / MiB;       /* the layers and the head */
    cfg.host_pool_mib = 0;
    cfg.embedding_placement = TablePlacement::Ram;
    cfg.embedding_resolved = TablePlacement::Ram;
    cfg.embedding_host_mib = emb / MiB;
    cfg.tower_resolved = TablePlacement::Ram;
    cfg.tower_host_mib = towers / MiB;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.strategy == Strategy::AllVram);
    CHECK(plan.weight[pos].tier == Tier::HostPinned);
    CHECK_EQ(plan.embedding_host_bytes, emb);
    CHECK_EQ(plan.tower_host_bytes, towers);
    CHECK_EQ(plan.host_budget, 0ll);
    int64_t vram = 0;
    for (const MoveUnit& mu : plan.units)
        if (mu.tier == Tier::VRAM) vram += mu.slot_bytes;
    CHECK_EQ(vram, cfg.vram_weights_mib * MiB);

    /* The table taken out of the statics twice: a table short, and the tail leaves the card. */
    cfg.vram_weights_mib = (fp.statics - emb - fp.tower) / MiB;
    Plan plan2;
    CHECK(Planner(in).plan(&plan2) < 0);
    CHECK(plan2.strategy == Strategy::LayerOffload);

    /* The embedding on the card: pin_towers pins the table with the rest of the tower. */
    const int64_t whole = tower_pin_bytes(fp, TablePlacement::Vram);
    Config vc = base_config();
    vc.placement = "auto";
    vc.vram_weights_mib = (fp.statics - whole) / MiB;
    vc.host_pool_mib = 0;
    vc.embedding_placement = TablePlacement::Vram;
    vc.embedding_resolved = TablePlacement::Vram;
    vc.tower_resolved = TablePlacement::Ram;
    vc.tower_host_mib = whole / MiB;
    Plan plan3;
    PlannerInput vin{&f.prog, &vc, 0, 1, nullptr, 0};
    CHECK_OK(Planner(vin).plan(&plan3));
    CHECK(plan3.strategy == Strategy::AllVram);
    CHECK(plan3.weight[pos].tier == Tier::HostPinned);
    CHECK_EQ(plan3.embedding_host_bytes, 0ll);
    CHECK_EQ(plan3.tower_host_bytes, whole);
    CHECK_EQ(plan3.host_budget, 0ll);
}

/* --vision-tower-placement ram: every tower unit goes to the pinned pool, read in place, before
 * any strategy runs, so a weights budget sized for the text model alone still plans and keeps the
 * text model on the card. Vram keeps the tower with the other weights. */
TEST(tower_ram_pins_every_tower_unit_in_the_pool_and_costs_no_vram) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    const size_t first_tower = f.prog.weights.size();
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 10;          /* exactly the text model: layers, embedding and head */
    cfg.host_pool_mib = 0;
    cfg.tower_resolved = TablePlacement::Ram;
    cfg.tower_host_mib = 8;             /* the resolver's room beside the pool */
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        if ((size_t)w < first_tower) {
            CHECK(plan.weight[w].tier == Tier::VRAM);
            continue;
        }
        CHECK(plan.weight[w].tier == Tier::HostPinned);
        CHECK(plan.weight[w].site == Site::DeviceZeroCopy);
        const MoveUnit& mu = plan.units[(size_t)plan.weight[w].unit];
        CHECK(!mu.movable);
        CHECK(mu.preplaced());
    }
    CHECK_EQ(plan.tower_host_bytes, 8 * MiB);
    CHECK_EQ(plan.forced_host_used, 8 * MiB);
    CHECK_EQ(plan.host_budget, 0ll);
    CHECK(plan.report(f.prog).find("--vision-tower-placement ram") != std::string::npos);

    cfg.tower_resolved = TablePlacement::Vram;
    cfg.tower_host_mib = 0;
    cfg.vram_weights_mib = 18;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        CHECK(plan2.weight[w].tier == Tier::VRAM);
    CHECK_EQ(plan2.tower_host_bytes, 0ll);
}

/* A tower too large for its room is refused, and so is ram for a program with no tower (the
 * resolver only picks Ram when the footprint found one, so that is a wiring fault). */
TEST(tower_ram_refuses_a_pool_too_small_and_a_program_with_no_tower) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_tower(f, /*first_layer=*/2, /*blocks=*/3, /*per_block=*/2 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 10;
    cfg.host_pool_mib = 0;
    cfg.tower_resolved = TablePlacement::Ram;
    cfg.tower_host_mib = 7;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(in).plan(&plan), RAD_E_NOMEM);

    Fixture t;
    build_dense_lookup(t, /*layers=*/2, /*per_layer=*/4 * MiB);
    cfg.tower_host_mib = 8;
    Plan plan2;
    PlannerInput tin{&t.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(tin).plan(&plan2), RAD_E_STATE);
}

/* A table that is not a whole number of pages still leaves the report's pinned total consistent:
 * the pool slot is charged rounded up while the tier total counts weight bytes, so the report
 * subtracts the same weight bytes it added. */
TEST(the_report_counts_a_hosted_embedding_once_in_one_unit) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB, /*emb=*/3 * MiB + 100);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.host_pool_mib = 8;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK_EQ(plan.forced_host_used, 3 * MiB + 100);
    CHECK(plan.embedding_host_bytes >= plan.forced_host_used);
    const std::string r = plan.report(f.prog);
    const size_t at = r.find("host pinned");
    CHECK(at != std::string::npos);
    if (at != std::string::npos) {
        const std::string line = r.substr(at, r.find('\n', at) - at);
        CHECK(line.find('-') == std::string::npos);           /* no negative total */
    }
    CHECK(r.find("--embedding-placement ram") != std::string::npos);
}

/* Ram resolved for a program with no table read only by embed_lookup means the budget was split
 * for a different program; the plan refuses rather than keep the table on a split without it. */
TEST(embedding_ram_with_no_table_to_place_is_refused) {
    Fixture t;
    const rad_weight m = t.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    t.op("embed_lookup", {m}, false);
    t.op("logits_gemm", {m}, false);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.host_pool_mib = 4;
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&t.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(in).plan(&plan), RAD_E_STATE);
    CHECK(!plan.notes.empty());
    CHECK(plan.weight[m].tier == Tier::VRAM);
}

TEST(embedding_ram_that_does_not_fit_the_host_pool_is_refused_with_the_plan_filled_in) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.host_pool_mib = 0;              /* nowhere to put it */
    cfg.embedding_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_EQ(Planner(in).plan(&plan), RAD_E_NOMEM);
    CHECK_EQ(plan.weight.size(), f.prog.weights.size());   /* finish() ran, so the report prints */
    CHECK(!plan.notes.empty());
}

/* --ngram-placement ram keeps the table on the mapped tier (counted by no pool) and names the
 * load-time copy: the reason says where rows come from, the report what the copy costs in RAM. */
TEST(ngram_ram_stays_mapped_and_reports_the_copy) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    const rad_weight t = add_ngram_table(f, 8 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.ngram_resolved = TablePlacement::Ram;
    Plan plan;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    CHECK_OK(Planner(in).plan(&plan));
    CHECK(plan.weight[t].tier == Tier::Mapped);
    CHECK(plan.weight[t].site == Site::Host);
    CHECK_EQ(plan.ngram_copy_bytes, 8 * MiB);
    CHECK(std::string(plan.weight[t].reason).find("--ngram-placement ram") != std::string::npos);
    CHECK(plan.report(f.prog).find("n-gram copy") != std::string::npos);

    cfg.ngram_resolved = TablePlacement::Disk;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK_EQ(plan2.ngram_copy_bytes, 0ll);
    CHECK(plan2.report(f.prog).find("n-gram copy") == std::string::npos);
}

/* An n-gram table read by both a device and a host kernel: --ngram-placement decides where it is
 * and so who gathers. vram counts and places it with the statics; ram keeps it mapped at the
 * zero-copy site (the pinned copy a card reads) and reports the copy; disk keeps it mapped at the
 * host site, where the host gather reads the container. */
TEST(an_ngram_table_a_device_kernel_reads_goes_where_the_placement_says) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    const rad_weight t = f.weight("ngram.weight", 8 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup_q", {t}, /*host_kernel=*/true);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Vram, false, true);
    CHECK(fp.ngram_device);
    CHECK(fp.ngram_host);
    CHECK_EQ(fp.ngram, 8 * MiB);
    CHECK_EQ(fp.mapped, 0ll);
    CHECK_EQ(fp.statics, 4 * MiB + 2 * MiB + 8 * MiB);
    for (TablePlacement off : { TablePlacement::Ram, TablePlacement::Disk }) {
        weight_footprint(f.prog, &fp, TablePlacement::Vram, off, false, true);
        CHECK_EQ(fp.ngram, 8 * MiB);
        CHECK_EQ(fp.mapped, 8 * MiB);
        CHECK_EQ(fp.statics, 4 * MiB + 2 * MiB);
    }

    Config cfg = base_config();
    cfg.placement = "all_vram";
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};

    cfg.ngram_resolved = TablePlacement::Vram;
    Plan on_card;
    CHECK_OK(Planner(in).plan(&on_card));
    CHECK(on_card.weight[t].tier == Tier::VRAM);
    CHECK(on_card.weight[t].site == Site::Device);
    CHECK_EQ(on_card.ngram_copy_bytes, 0ll);

    cfg.ngram_resolved = TablePlacement::Ram;
    Plan in_ram;
    CHECK_OK(Planner(in).plan(&in_ram));
    CHECK(in_ram.weight[t].tier == Tier::Mapped);
    CHECK(in_ram.weight[t].site == Site::DeviceZeroCopy);
    CHECK_EQ(in_ram.ngram_copy_bytes, 8 * MiB);
    CHECK(in_ram.ngram_copy_card);
    CHECK(std::string(in_ram.weight[t].reason).find("the card reads") != std::string::npos);
    CHECK(in_ram.report(f.prog).find("pinned RAM the card reads") != std::string::npos);

    cfg.ngram_resolved = TablePlacement::Disk;
    Plan on_disk;
    CHECK_OK(Planner(in).plan(&on_disk));
    CHECK(on_disk.weight[t].tier == Tier::Mapped);
    CHECK(on_disk.weight[t].site == Site::Host);
    CHECK_EQ(on_disk.ngram_copy_bytes, 0ll);
    CHECK(std::string(on_disk.weight[t].reason).find("--ngram-placement disk") !=
          std::string::npos);
}

/* Disk needs the host gather kernel: an n-gram table only a device kernel reads has no host site,
 * and the footprint reports that. */
TEST(an_ngram_table_with_no_host_kernel_cannot_stay_on_disk) {
    Fixture f;
    const rad_weight t = f.weight("ngram.weight", 8 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup_q", {t}, /*host_kernel=*/false);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK(fp.ngram_device);
    CHECK(!fp.ngram_host);
}

TEST(an_empty_program_has_an_empty_footprint) {
    Fixture f;
    WeightFootprint fp;
    fp.statics = 1;                         /* the out parameter is reset, not accumulated */
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk, false, true);
    CHECK_EQ(fp.statics, 0ll);
    CHECK_EQ(fp.experts, 0ll);
    CHECK_EQ(fp.mapped, 0ll);
    CHECK_EQ(fp.n_units, 0ll);
    /* a null out parameter does not crash */
    weight_footprint(f.prog, nullptr, TablePlacement::Vram, TablePlacement::Disk, false, true);
}

/* ================================================================== planner decisions */
namespace {

/* `layers` routed layers of `n` experts of `each` bytes, one weight and one op per layer, plus a
 * model-level embedding. */
void build_moe_grid(Fixture& f, int layers, int n, int64_t each) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < layers; ++l) {
        std::vector<rad_weight> g;
        for (int e = 0; e < n; ++e)
            g.push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." + std::to_string(e) +
                                 ".weight", each, RAD_ACCESS_CONDITIONAL, l, e));
        f.op("moe_gemm", g, false);
    }
}

/* Resident (VRAM) routed units per layer. */
std::vector<int> residents_per_layer(const Plan& p, int layers) {
    std::vector<int> n((size_t)layers, 0);
    for (const MoveUnit& mu : p.units)
        if (mu.expert >= 0 && mu.layer >= 0 && mu.layer < layers && mu.tier == Tier::VRAM)
            ++n[(size_t)mu.layer];
    return n;
}

int spread(const std::vector<int>& v, size_t from = 0, size_t to = SIZE_MAX) {
    int lo = INT32_MAX, hi = INT32_MIN;
    for (size_t i = from; i < v.size() && i < to; ++i) { lo = std::min(lo, v[i]); hi = std::max(hi, v[i]); }
    return hi - lo;
}

}  /* namespace */

/* Without a calibration profile every share is zero and the tie-break is the whole policy. It is
 * expert-major: expert 0 of every layer before expert 1 of any, so the budget cut lands at the
 * same expert everywhere and every layer starts with its share of the slab. Layer-major filling
 * leaves the last layers (and a draft head) with nothing resident, streaming every expert over the
 * link each step. With a profile, the profile decides. */
TEST(planner_spreads_residents_over_the_layers_and_follows_a_profile) {
    Fixture f;
    build_moe_grid(f, 4, 16, 1 * MiB);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 31;
    c.host_pool_mib = 128;
    Plan p;
    { PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0}; CHECK_OK(Planner(in).plan(&p)); }
    const std::vector<int> flat = residents_per_layer(p, 4);
    const int total = flat[0] + flat[1] + flat[2] + flat[3];
    CHECK(total >= 8 && total < 64);
    CHECK(spread(flat) <= 1);

    /* Layer 3 is hot in the calibration set: it fills first, and the rest still spreads. */
    std::vector<RadFileProfile> prof;
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) prof.push_back(RadFileProfile{ l, e, l == 3 ? 0.05f : 0.001f, 0.0f });
    Plan q;
    {
        PlannerInput in{&f.prog, &c, 0, 1, prof.data(), (int64_t)prof.size()};
        CHECK_OK(Planner(in).plan(&q));
    }
    const std::vector<int> hot = residents_per_layer(q, 4);
    CHECK_EQ(hot[0] + hot[1] + hot[2] + hot[3], total);
    CHECK(total > 16);
    CHECK_EQ(hot[3], 16);
    CHECK(spread(hot, 0, 3) <= 1);
}

/* `auto`, the default, is one of the three named strategies: everything on the card when it fits
 * (or no budget caps it), routed experts spilled when there are any, whole layers otherwise; the
 * plan says which and why. */
TEST(auto_placement_picks_the_strategy_the_model_and_budget_call_for) {
    struct Case { const char* what; bool moe; int64_t mib; Strategy want; };
    for (const Case& k : { Case{ "fits", true, 64, Strategy::AllVram },
                           Case{ "no budget", true, 0, Strategy::AllVram },
                           Case{ "routed experts spill", true, 4, Strategy::ExpertTiered },
                           Case{ "dense layers spill", false, 3, Strategy::LayerOffload } }) {
        Fixture f;
        if (k.moe) build_moe(f, 8, 1 * MiB, nullptr);
        else       build_dense(f, 4, 1 * MiB, /*host_kernel=*/true);
        Config c = base_config();
        c.vram_weights_mib = k.mib;
        c.host_pool_mib = 64;
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        Plan p;
        CHECK_OK(Planner(in).plan(&p));
        if (p.strategy != k.want) fprintf(stderr, "    %s: auto chose %s\n", k.what, strategy_name(p.strategy));
        CHECK(p.strategy == k.want);
        CHECK(!p.strategy_why.empty());
    }
}

/* The spilled layers are a suffix of execution order, which a plugin's layer numbering need not
 * follow: declared last-numbered first, the layers executed last are layers 0-3. */
TEST(layer_offload_spills_the_layers_that_run_last_not_the_highest_numbered) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, true);
    for (int l = 7; l >= 0; --l)
        f.op("gemm", { f.weight("blk." + std::to_string(l) + ".ffn_down.weight", 1 * MiB,
                                RAD_ACCESS_PER_TOKEN, l, -1) }, true);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    for (const MoveUnit& mu : p.units) {
        if (mu.layer < 0) continue;
        CHECK(mu.site == (mu.layer < 4 ? Site::Host : Site::Device));
    }
    REQUIRE_EQ(p.host_runs.size(), 1u);
    CHECK_EQ(p.host_runs[0].first_layer, 0);
    CHECK_EQ(p.host_runs[0].last_layer, 3);
}

/* When the dense weights alone overflow the budget, expert_tiered spills layers first and then
 * places the routed experts. The host pool is spent once across both: the experts get what the
 * spilled dense units left, the rest are listed as unfit, and nothing past the pool is placed,
 * which the mover would refuse at reserve. */
TEST(an_expert_fill_after_a_layer_spill_spends_the_host_pool_once) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, true);
    for (int l = 0; l < 4; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        f.op("gemm", { f.weight(b + "ffn_down.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, l, -1) }, true);
        f.op("moe_gemm", { f.weight(b + "ffn_gate_exps.0.weight", 1 * MiB, RAD_ACCESS_CONDITIONAL, l, 0),
                           f.weight(b + "ffn_gate_exps.1.weight", 1 * MiB, RAD_ACCESS_CONDITIONAL, l, 1) },
             true);
    }
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 5;          /* below the 9 MiB of dense weights */
    /* What the layer spill places: three layers' dense units and experts. The expert fill
     * that follows keeps none resident, so it has the six dense MiB left over for eight experts. */
    c.host_pool_mib = 12;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    (void)stderr_of([&] { (void)Planner(in).plan(&p); });
    int64_t host = 0, dense_host = 0, unfit = 0, lost = 0;
    for (const MoveUnit& mu : p.units) {
        const bool off = mu.tier == Tier::HostPinned || mu.tier == Tier::HostPageable;
        if (off) {
            host += mu.slot_bytes;
            if (mu.expert < 0) dense_host += mu.slot_bytes;
        }
        if (mu.reason && std::string(mu.reason) == "fits in no tier") ++unfit;
        else if (!off && mu.tier != Tier::VRAM) ++lost;
    }
    CHECK(dense_host > 0);
    CHECK(host <= p.host_budget);
    /* Every unit is placed or listed as unfit, never left in between. */
    int64_t listed = 0;
    for (const Plan::Unfit& u : p.unfit) listed += u.count;
    CHECK_EQ(listed, unfit);
    CHECK(unfit > 0);
    CHECK_EQ(lost, 0ll);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);
}

/* An op runs in one domain, so its weights must agree on the site. An op reading a host-run
 * layer's weight beside a device one demotes the host unit to staged, which every tier supports,
 * and says so; the plan still fits. */
TEST(an_op_reading_host_and_device_weights_demotes_the_host_one_to_staged) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/true);
    /* Layer 7 runs on the host under this budget, layer 0 on the device. */
    f.op("mix", { (rad_weight)9, (rad_weight)2 }, true);
    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;
    c.host_pool_mib = 64;
    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    int rc = 0;
    const std::string log = stderr_of([&] { rc = Planner(in).plan(&p); });
    CHECK_OK(rc);
    CHECK(has(log, "mixes host-site and device-site"));
    for (const MoveUnit& mu : p.units) {
        if (mu.layer == 7) {
            CHECK(mu.site == Site::DeviceStaged);
            CHECK(mu.reason && has(mu.reason, "demoted"));
        } else if (mu.layer >= 4) {
            CHECK(mu.site == Site::Host);
        }
    }
    CHECK(p.fits());
}

/* ================================================================== heat tie-breaks
 *
 * Most residents sit at exactly zero heat, so the tie-break is effectively the policy: on equal
 * heat, promote into the layer holding the fewest residents and demote from the one holding the
 * most, then the lower unit index. step() is checked against pick() elsewhere; these pin pick()
 * itself against hand-set residency.
 */
namespace {
struct TwoLayers {
    Fixture f;
    Plan plan;
    HeatEngine h;
    int build(int res0, int res1) {
        build_moe_grid(f, 2, 8, 1 * MiB);
        Config c = base_config();
        c.placement = "expert_tiered";
        c.vram_weights_mib = 12;
        c.host_pool_mib = 64;
        PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
        RAD_TRY(Planner(in).plan(&plan));
        RAD_TRY(h.init(plan, HeatParams{}));
        /* Residency as asked: the first `res` experts of each layer. */
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
            const MoveUnit& mu = plan.units[(size_t)u];
            if (mu.expert < 0) continue;
            h.set_tier(u, mu.expert < (mu.layer == 0 ? res0 : res1) ? Tier::VRAM : Tier::HostPinned);
        }
        return RAD_OK;
    }
    int32_t unit(int layer, int expert) const {
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u)
            if (plan.units[(size_t)u].layer == layer && plan.units[(size_t)u].expert == expert) return u;
        return -1;
    }
    void heat(int layer, int expert, float credit) {
        const int32_t counts[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int32_t c[8];
        std::memcpy(c, counts, sizeof c);
        c[expert] = 1;
        h.observe(layer, c, 8, credit);
    }
};
}  /* namespace */

TEST(on_equal_heat_the_thinner_layer_is_promoted_into_and_the_fuller_one_gives_up) {
    TwoLayers t;
    REQUIRE_EQ(t.build(5, 2), RAD_OK);
    CHECK_EQ(t.h.layer_residents(t.unit(0, 0)), 5);
    CHECK_EQ(t.h.layer_residents(t.unit(1, 0)), 2);
    t.heat(0, 6, 4.0f);
    t.heat(1, 6, 4.0f);
    CHECK_EQ(t.h.heat(t.unit(0, 6)), t.h.heat(t.unit(1, 6)));
    Swap s;
    REQUIRE(t.h.pick(&s));
    CHECK_EQ(s.promote, t.unit(1, 6));
    /* Every resident is at zero: the victim comes from the layer holding five, lowest index first. */
    CHECK_EQ(s.demote, t.unit(0, 0));
    std::vector<Swap> sw;
    CHECK_OK(t.h.step(&sw));
    REQUIRE(!sw.empty());
    CHECK_EQ(sw[0].promote, s.promote);
    CHECK_EQ(sw[0].demote, s.demote);
}

TEST(on_equal_heat_and_residents_the_lower_unit_wins) {
    TwoLayers t;
    REQUIRE_EQ(t.build(3, 3), RAD_OK);
    t.heat(1, 5, 4.0f);
    t.heat(0, 5, 4.0f);
    Swap s;
    REQUIRE(t.h.pick(&s));
    const int32_t a = t.unit(0, 5), b = t.unit(1, 5);
    CHECK_EQ(s.promote, std::min(a, b));
    CHECK_EQ(s.demote, std::min(t.unit(0, 0), t.unit(1, 0)));
}

/* The guard is `hot >= cold * hysteresis + min_gain`, inclusive: against a zero-heat victim a
 * candidate at exactly min_gain is promoted, and one a float step below it is not. */
TEST(the_promotion_guard_is_inclusive_at_its_boundary) {
    const HeatParams hp;
    for (bool at : { true, false }) {
        TwoLayers t;
        REQUIRE_EQ(t.build(4, 4), RAD_OK);
        const float credit = at ? hp.min_gain : std::nextafter(hp.min_gain, 0.0f);
        t.heat(0, 6, credit);
        CHECK_EQ(t.h.heat(t.unit(0, 6)), credit);
        Swap s;
        CHECK_EQ(t.h.pick(&s), at);
        std::vector<Swap> sw;
        CHECK_OK(t.h.step(&sw));
        CHECK_EQ(sw.empty(), !at);
    }
}

/* ================================================================== the loan fill, by layer */

/* On a cold start every heat is equal, so the fill's tie-break decides where a loan's slots go:
 * the layer with the fewest residents first, counting the promotions the batch has issued, since
 * none has retired yet. A fill in unit order gives the first layers everything and the last layers
 * nothing. Every slot is issued in the dispatch the loan arrives in. */
TEST(a_cold_fill_spreads_the_loan_over_the_layers) {
    Fixture f;
    build_moe_grid(f, 4, 16, 256 * KiB);
    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 3;
    c.host_pool_mib = 64;
    Plan plan;
    { PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0}; REQUIRE_EQ(Planner(in).plan(&plan), RAD_OK); }
    HeatEngine h;
    REQUIRE_EQ(h.init(plan, HeatParams{}), RAD_OK);
    Mover m;
    MoverConfig mc;
    mc.turnover_slots = 0;
    REQUIRE_EQ(m.init(f.prog, plan, mc), RAD_OK);
    RadStream compute = nullptr;
    REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
    CHECK_OK(m.bind_compute_stream(compute));
    Loan loan;
    CHECK_OK(m.lend_from(Mover::kLendKV, loan.strips));
    const std::vector<int> before = residents_per_layer(plan, 4);
    CHECK(spread(before) <= 1);
    int32_t pooled = 0, cls = -1;
    for (const MoveUnit& mu : plan.units) {
        if (mu.expert < 0) continue;
        pooled += mu.tier == Tier::HostPinned;
        cls = mu.slab_class;
    }
    REQUIRE(cls >= 0);
    const int64_t slots = m.flex_capacity() / plan.classes[(size_t)cls].stride;
    const int64_t expect = std::min<int64_t>(slots, pooled);
    CHECK(expect >= 8);

    ComputeWork work;
    int d = 0;
    CHECK_OK(m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(m.begin_dispatch(d++, &h));
    work.run(compute);
    CHECK_OK(m.balance(&h));
    int64_t issued = 0;
    for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) issued += h.busy(u);
    CHECK_EQ(issued, expect);
    auto now = [&]() {
        std::vector<int> n(4, 0);
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u) {
            const MoveUnit& mu = plan.units[(size_t)u];
            if (mu.expert >= 0 && h.tier(u) == Tier::VRAM) ++n[(size_t)mu.layer];
        }
        return n;
    };
    const int planned = before[0] + before[1] + before[2] + before[3];
    for (Deadline dl; any_busy(h, plan) && !dl.passed();) {
        CHECK_OK(m.begin_dispatch(d++, &h));
        work.run(compute);
        CHECK_OK(m.balance(&h));
    }
    const std::vector<int> after = now();
    CHECK_EQ(after[0] + after[1] + after[2] + after[3], planned + (int)expect);
    CHECK(spread(after) <= 1);
    CHECK_OK(rad_stream_sync(compute));
    rad_stream_destroy(compute);
}

/* ================================================================== teardown */

namespace {

/* The VRAM this process holds, as KFD counts it per process (/sys/class/kfd/kfd/proc/<pid>/
 * vram_<gpu>, summed over the cards). The card-wide figure also moves with other processes and
 * with this process's page tables, which grow as address ranges are first mapped and are not
 * returned on free: a page, or a page plus a 2 MiB directory, of difference there is no leak.
 * Without KFD (the host backend) the backend's own figure. */
int process_vram(int64_t* out) {
    const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string((long long)getpid());
    DIR* d = opendir(dir.c_str());
    if (!d) return dev_vram_used(0, out);
    int64_t sum = 0;
    bool any = false;
    while (const dirent* e = readdir(d)) {
        if (std::strncmp(e->d_name, "vram_", 5) != 0) continue;
        FILE* f = std::fopen((dir + "/" + e->d_name).c_str(), "r");
        long long v = 0;
        if (f && std::fscanf(f, "%lld", &v) == 1) { sum += v; any = true; }
        if (f) std::fclose(f);
    }
    closedir(d);
    if (!any) return dev_vram_used(0, out);
    *out = sum;
    return RAD_OK;
}

/* process_vram once it holds still. A stream is a hardware queue, and KFD gives each queue a
 * context save area (the topology's cwsr_size + ctl_stack_size: 29.3 MiB on a 64-CU RDNA4 card)
 * that can be released a few milliseconds after the queue is destroyed, so a reading straight
 * after a stream's destruction may still include it (#124). Reads every 10 ms until 100 ms pass
 * without a change, for at most 2 s. */
int settled_vram(int64_t* out) {
    int64_t v = 0;
    RAD_TRY(process_vram(&v));
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (int still = 0; still < 10 && std::chrono::steady_clock::now() < end;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        int64_t w = 0;
        RAD_TRY(process_vram(&w));
        still = w == v ? still + 1 : 0;
        v = w;
    }
    *out = v;
    return RAD_OK;
}

}  /* namespace */

/* Tools and tests build and tear down movers repeatedly, and on a card every event or stream is
 * a hardware queue and VRAM. A mover that staged, promoted and filled a loan gives back every
 * event and every byte of device memory it took. */
/* The runtime and the driver keep some memory for the life of the process once a path is first
 * used: page tables for address ranges first mapped (a page, or a page and a 2 MiB directory). So
 * the work runs once before the count is taken, and the count after a second run may differ by a
 * page and a directory, never by a buffer. Both counts are taken once the figure holds still: a
 * destroyed stream's queue save area can outlive it by a few milliseconds (settled_vram, #124). */
TEST(a_mover_gives_back_every_event_and_device_byte) {
    auto run = [&]() {
        const int64_t events0 = rad_dev_live_events();
        {
            Fixture f;
            build_dense(f, 8, 1 * MiB, /*host_kernel=*/false);
            Config c = base_config();
            c.placement = "layer_offload";
            c.vram_weights_mib = 5;
            c.host_pool_mib = 64;
            PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
            Plan p;
            REQUIRE_EQ(Planner(in).plan(&p), RAD_OK);
            Mover m;
            MoverConfig mc;
            REQUIRE_EQ(m.init(f.prog, p, mc), RAD_OK);
            RadStream compute = nullptr;
            REQUIRE_EQ(rad_stream_create(&compute, 0), RAD_OK);
            CHECK_OK(m.bind_compute_stream(compute));
            CHECK(rad_dev_live_events() > events0);
            ComputeWork work;
            for (int d = 0; d < 20; ++d) {
                CHECK_OK(m.begin_dispatch(d, nullptr));
                work.run(compute);
                CHECK_OK(m.prefetch_to_op(p.prefetch[(size_t)d % p.prefetch.size()].at_op));
            }
            CHECK(m.stats().prefetched > 0);
            CHECK_OK(rad_stream_sync(compute));
            rad_stream_destroy(compute);
        }
        {
            LoanRig r;
            REQUIRE_EQ(r.build(), RAD_OK);
            CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
            for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
                CHECK_OK(r.dispatch());
            CHECK(r.m.stats().promotions > 0);
        }
        CHECK_EQ(rad_dev_live_events(), events0);
    };
    run();
    int64_t vram0 = 0, vram1 = 0;
    CHECK_OK(settled_vram(&vram0));
    run();
    CHECK_OK(settled_vram(&vram1));
    CHECK(vram1 - vram0 <= 2 * MiB + 4096);
}

RAD_TEST_MAIN()
