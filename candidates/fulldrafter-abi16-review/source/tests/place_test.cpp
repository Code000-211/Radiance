/* place_test.cpp -- the placement planner, the mover's slot bookkeeping, and the heat engine.
 *
 * The interesting test in here is `heat_does_not_thrash`. Promotion is easy to get right and easy
 * to test; what decides whether dynamic placement is worth running at all is whether it declines
 * to move when moving would not help, because a swap costs two shards over the same link a miss
 * costs one. The test runs the same alternating access pattern twice, once with the guards at
 * their shipped values and once with them removed, and the difference between the two numbers is
 * the guards doing their job.
 *
 * ---- THE FAKE DEVICE LAYER -------------------------------------------------------------------
 *
 * core/place/mover.cpp calls the device layer, which another component owns. Under the normal
 * CMake build this test links the real backend (host or HIP) and the block below is not compiled.
 * Building it standalone defines RAD_PLACE_TEST_FAKE_DEVICE and supplies a malloc/memcpy
 * stand-in:
 *
 *   g++ -std=c++20 -DRAD_PLACE_TEST_FAKE_DEVICE -Iinclude -Icore -Itests \
 *       tests/place_test.cpp core/place/planner.cpp core/place/heat.cpp core/place/mover.cpp \
 *       core/util/rad_util.cpp -o place_test
 *
 * The fake completes every event immediately, which is the right shape for testing the STATE
 * MACHINE -- Free/Copying/Published/Quarantine -- and the wrong shape for testing the pipelining.
 * Pipelining is a property of a real stream and is measured on a card, not asserted here.
 */
#include "rad_test.h"

#include "device/directio.h"
#include "place/heat.h"
#include "place/mover.h"
#include "place/stager.h"
#include "place/planner.h"
#include "place/prefetch.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace rad;

/* ================================================================== the fake device layer */
#ifdef RAD_PLACE_TEST_FAKE_DEVICE
namespace {
struct FakeStream { int id; };
struct FakeEvent { int recorded; };
}  /* namespace */

extern "C" {
int  rad_dev_count(void) { return 1; }
int  rad_dev_set(int) { return RAD_OK; }
int  rad_dev_props(int, RadDeviceProps* o) {
    if (!o) return RAD_E_INVAL;
    std::memset(o, 0, sizeof *o);
    std::snprintf(o->name, sizeof o->name, "fake");
    std::snprintf(o->arch, sizeof o->arch, "host");
    o->is_host_backend = 1;
    return RAD_OK;
}
int  rad_dev_enable_peer(int, int) { return RAD_E_UNSUPPORTED; }

void* rad_dev_alloc(int64_t bytes, int) { return bytes > 0 ? std::malloc((size_t)bytes) : nullptr; }
void  rad_dev_free(void* p, int) { std::free(p); }
void* rad_dev_host_ptr(void* p) { return p; }
void* rad_dev_device_ptr(void* p) { return p; }

int  rad_stream_create(RadStream* out, int) { *out = new FakeStream{0}; return RAD_OK; }
void rad_stream_destroy(RadStream s) { delete (FakeStream*)s; }
int  rad_stream_sync(RadStream) { return RAD_OK; }

int  rad_event_create(RadEvent* out) { *out = new FakeEvent{0}; return RAD_OK; }
int  rad_event_create_local(RadEvent* out) { *out = new FakeEvent{0}; return RAD_OK; }
int  rad_event_create_as(RadEvent* out, unsigned) { *out = new FakeEvent{0}; return RAD_OK; }
void rad_event_destroy(RadEvent e) { delete (FakeEvent*)e; }
int  rad_event_record(RadEvent e, RadStream) { ((FakeEvent*)e)->recorded = 1; return RAD_OK; }
int  rad_event_wait(RadStream, RadEvent) { return RAD_OK; }
int  rad_event_query(RadEvent) { return 1; }   /* everything has always landed */
int  rad_event_sync(RadEvent) { return RAD_OK; }
void rad_dev_host_wrote(void) {}
int  rad_event_elapsed_ms(RadEvent, RadEvent, float* o) { if (o) *o = 0.0f; return RAD_OK; }

int  rad_memcpy_async(void* d, const void* s, int64_t n, RadStream) {
    if (n > 0) std::memcpy(d, s, (size_t)n);
    return RAD_OK;
}
int  rad_memcpy_2d_async(void*, int64_t, const void*, int64_t, int64_t, int64_t, RadStream) {
    return RAD_OK;
}
int  rad_memset_async(void* d, int v, int64_t n, RadStream) {
    if (n > 0) std::memset(d, v, (size_t)n);
    return RAD_OK;
}
const char* rad_dev_last_error(void) { return ""; }
}  /* extern "C" */
#endif  /* RAD_PLACE_TEST_FAKE_DEVICE */

/* ================================================================== a synthetic Program */
namespace {

constexpr int64_t MiB = 1024 * 1024;

/* Two kernel rows, one per domain. The planner reads Band::dom[] to learn whether a host site
 * exists at all, so a fixture that wants to exercise that has to resolve real rows. */
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
        w.decl.dtype = RAD_U8;         /* one byte an element: bytes are exactly what is asked */
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

/* `layers` routed layers of `n` experts, each expert a unit of TWO weights -- gate_up of `gu`
 * bytes and down of `dn` -- read by two grouped ops a layer, which is the shape a routed block
 * declares: a unit whose weights are not contiguous in memory the way they are in a container. */
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

/* Every placement decision, as one string. Two plans that differ anywhere differ here. */
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
    /* The degenerate case: everything on tier 0 and nothing for the mover to do. */
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
    /* The report is the deliverable, not the status code: an operator needs the number. */
    CHECK(p.report(f.prog).find("DID NOT FIT") != std::string::npos);
}

TEST(planner_host_runs_are_contiguous) {
    Fixture f;
    build_dense(f, 8, 1 * MiB, /*host_kernel=*/true);

    Config c = base_config();
    c.placement = "layer_offload";
    c.vram_weights_mib = 5;      /* 9 MiB of weights: four layers have to leave */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* ONE run, not four. Alternating per layer would cost eight crossings a step for the same
     * bytes; a run costs two. That is the whole objective. */
    CHECK_EQ((int)p.host_runs.size(), 1);
    CHECK_EQ(p.link_crossings, 2);
    if (!p.host_runs.empty()) {
        CHECK_EQ(p.host_runs[0].first_layer, 4);
        CHECK_EQ(p.host_runs[0].last_layer, 7);
    }
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 5 * MiB);
    /* PAGEABLE, not pinned, and that is the site deciding the tier. A host-site layer is read by
     * a host kernel; pinning it buys nothing and pinned memory is the scarce kind -- past a
     * driver-dependent total the registration starts succeeding and leaving the process broken. */
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPageable], 4 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], (int64_t)0);

    /* And it is a SUFFIX, so the layers still on the device are exactly the early ones -- the
     * ones a staged prefix would have had to land before the step could start at all. */
    for (const MoveUnit& mu : p.units) {
        if (mu.layer < 0) continue;
        if (mu.layer < 4) { CHECK(mu.site == Site::Device); CHECK(mu.tier == Tier::VRAM); }
        else              { CHECK(mu.site == Site::Host);   CHECK(mu.tier == Tier::HostPageable); }
    }
}

TEST(planner_stages_when_no_host_kernel_resolved) {
    /* Same model, same budget, one difference: declare resolved no RAD_DOMAIN_HOST kernel. The
     * host SITE is then unavailable and the same bytes go to the same tier with a different site
     * -- which is exactly the claim that tier and site are independent axes. */
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
    /* Two layers more than the host-kernel case spill, and the difference is exactly the staging
     * ring: a staged unit needs a slab slot to be staged INTO, and the planner reserves that out
     * of the VRAM budget before it chooses the cut. A plan that spills to exactly the budget and
     * only then finds it has nowhere to stage is a plan that does not load. */
    CHECK_EQ(p.slab_overhead, (int64_t)RAD_PLACE_STAGE_RING * MiB + RAD_ALIGN_UNIT);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], 7 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 2 * MiB);
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);
    for (const MoveUnit& mu : p.units)
        if (mu.layer >= 1) CHECK(mu.site == Site::DeviceStaged);

    /* Layer offload is a SCHEDULING problem: every staged unit has an exact deadline, taken off
     * the declared order, and the schedule is sorted by it. */
    CHECK_EQ((int)p.prefetch.size(), 7);
    for (size_t i = 1; i < p.prefetch.size(); ++i)
        CHECK(p.prefetch[i - 1].at_op <= p.prefetch[i].at_op);
    CHECK(p.prefetch[0].at_op > 0);
}

TEST(planner_shards_under_world_size_2) {
    /* The plugin declares dimensions ALREADY DIVIDED (spec §9), so a rank's unit is its own
     * shard. What the planner owes tensor parallel is not the arithmetic -- it is the invariant
     * that the resident SET is identical on every rank, because a slab slot only means the same
     * thing everywhere if every rank chose the same experts to put in it. */
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

    /* Each rank plans against its OWN card, holding half the expert bytes. The embedding is
     * replicated and is the same on both, which is why this is not simply "half of everything". */
    CHECK_EQ(pw.totals.bytes[(int)Tier::VRAM], 33 * MiB);
    CHECK_EQ(p0.totals.bytes[(int)Tier::VRAM], 17 * MiB);

    /* THE INVARIANT. Same decisions on both ranks, weight for weight. Anything rank-local
     * creeping into the ranking -- a device query, a clock, a hash seed -- fires here. */
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
    c.vram_weights_mib = 5;      /* 1 MiB embedding, and the rest is the expert slab */
    c.host_pool_mib = 64;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));

    /* THREE resident, not four, and the fourth slot is the shadow. Spare slots come out of
     * --vram-weights-mib like everything else in the slab -- there is nowhere else for them to
     * come from, because --gpu-headroom-mib is memory the engine promises not to touch -- so
     * dynamic placement costs a resident unit and the report says how many. */
    CHECK_EQ(p.totals.bytes[(int)Tier::VRAM], 3 * MiB);
    CHECK_EQ(p.totals.bytes[(int)Tier::HostPinned], 6 * MiB);
    CHECK_EQ(p.slab_overhead, 1 * MiB + RAD_ALIGN_UNIT);
    /* And the whole of it fits the budget the operator stated, which is the property that stops
     * the mover overrunning the pool that was sized for it. */
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead <= p.vram_budget);

    /* The embedding is read every token: it never leaves, whatever the experts want. */
    CHECK(p.weight[1].tier == Tier::VRAM);
    /* A pooled expert is zero-copy, not staged: the GEMM streams it over the link as it computes,
     * so a miss costs one shard and no slab slot at all. */
    int zero_copy = 0;
    for (const MoveUnit& mu : p.units)
        if (mu.tier == Tier::HostPinned && mu.site == Site::DeviceZeroCopy) ++zero_copy;
    CHECK_EQ(zero_copy, 6);
}

TEST(planner_warm_starts_from_the_container_profile) {
    Fixture f;
    std::vector<rad_weight> ex;
    build_moe(f, 8, 1 * MiB, &ex);

    /* Experts 5, 6 and 7 are the popular ones in the calibration set. The profile is a warm
     * start and not a policy -- it decides which experts BEGIN resident and nothing after. */
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
    /* Every pool is budgeted explicitly (spec §6). An offload strategy with nothing to spill
     * against would have to infer a budget, and an engine that quietly re-optimises its own split
     * is an engine whose benchmarks do not reproduce. */
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

    /* Identical plan across runs, and it is a real mode rather than a flag checked nowhere:
     * nothing is movable, so the heat engine has nothing to act on and residency -- and therefore
     * the summation order -- is fixed for the whole run (spec §17). */
    CHECK_EQ(digest(f.prog, a), digest(f.prog, b));
    CHECK(!a.dynamic);
    for (const MoveUnit& mu : a.units) CHECK(!mu.movable);

    HeatEngine h;
    CHECK_OK(h.init(a, HeatParams{}));
    CHECK(!h.enabled());
    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK(sw.empty());

    /* ...and the same plan with determinism off IS movable, so the test above is testing the
     * mode and not an accident of the fixture. */
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
    /* Two routed layers of eight 1.5 MiB experts against a VRAM budget that holds a few of them
     * and a host pool that holds two. With the file tier on, the rest go there -- and the plan
     * still fits, because what serves them is a read of the whole layer into one of two buffers
     * the plan charges against the same VRAM budget. */
    Fixture f;
    build_moe_layers(f, 2, 8, 1 * MiB, MiB / 2);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 24;
    /* The file staging reserve -- four slots of the widest layer unit and the read windows -- and
     * room for two pooled experts behind it. */
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
            /* Staged, never movable: the heat engine promotes out of a pool slot and a file-tier
             * unit has none. */
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
    /* The buffers are the largest layer's file-tier units, twice, at the stager's packing. */
    CHECK_EQ(p.file_layer_max, std::max(per_layer[0], per_layer[1]));
    CHECK(p.file_stage_vram >= 2 * p.file_layer_max + (int64_t)RAD_ALIGN_UNIT);
    /* And everything the VRAM side holds fits the budget the operator stated -- the residents,
     * the shadow slots and the two buffers -- so the mover cannot overrun the pool. */
    CHECK(p.totals.bytes[(int)Tier::VRAM] + p.slab_overhead + p.file_stage_vram <= p.vram_budget);
    /* The exact prefetch's ring is not for these: the file stager reads into its own buffers. */
    for (const SlabClass& sc : p.classes) CHECK_EQ(sc.n_stage, (int64_t)0);

    /* Without the tier the same budgets strand them, by name. */
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

    /* Expert 7 was pooled by the planner (no profile, so the first four win on unit order). Ten
     * of the step's tokens routed to it -- the credit is that count, not a flat one, because an
     * expert several tokens chose is far likelier to be chosen again than one a single token
     * chose, and weighting by it moves no extra bytes. */
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
        /* The pair is returned together because the DECISION is a comparison between them:
         * promoting without the matching demotion would grow the resident set past the slab. */
        CHECK(sw[0].promote != sw[0].demote);
    }
}

TEST(heat_does_not_thrash_on_an_alternating_pattern) {
    /* THE IMPORTANT ONE. The four resident experts and the four pooled ones are selected in
     * strict alternation, one token each, so the two halves are of exactly equal value and their
     * heats converge to within one decay step of each other. A swap here moves two shards over
     * the link to arrive somewhere no better -- both directions of the bus spent to return to
     * where it started, on the link the zero-copy tier is trying to use.
     *
     * The guards are the only defence, because there is no admission filter. Running the same
     * pattern with them removed is the control: the difference between the two counts is the
     * hysteresis doing its job, and without the control the first number proves nothing -- an
     * engine that never proposes anything would also score zero. */
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
            /* Nothing is issued, so nothing publishes; hand the units straight back or the engine
             * goes quiet on busy flags rather than on policy. */
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
    /* And the control: with the guards off the same pattern churns on every dispatch that ends
     * with the pooled half credited, which is what the guards exist to stop. */
    CHECK(unguarded > 100);
}

TEST(heat_respects_the_move_budget) {
    HeatFixture hf;
    HeatEngine h;
    HeatParams hp;
    hp.moves_per_dispatch = 3;
    hp.min_gain = 0.5f;
    CHECK_OK(h.init(hf.plan, hp));

    /* Every pooled expert is hot, so the guards have no reason to refuse and the ONLY thing that
     * can bound the proposals is the budget. That is the point of the test: a budget that is only
     * respected when the policy would have stopped anyway is not a budget. */
    const int32_t ids[4] = {4, 5, 6, 7};
    const int32_t counts[4] = {20, 20, 20, 20};
    h.observe_ids(0, ids, counts, 4);

    std::vector<Swap> sw;
    CHECK_OK(h.step(&sw));
    CHECK_EQ((int)sw.size(), 3);

    /* And no unit appears twice: a proposal marks both halves busy on the way out, or the mover
     * issues the same transfer three times. */
    for (size_t i = 0; i < sw.size(); ++i)
        for (size_t j = i + 1; j < sw.size(); ++j) {
            CHECK(sw[i].promote != sw[j].promote);
            CHECK(sw[i].demote != sw[j].demote);
        }
}

/* step() ranks every movable unit once and proposes from both ends of that ranking; the policy is
 * stated by pick(), one pair a call. They must agree exactly -- the same pairs in the same order,
 * the same units left busy, the same reason for stopping -- or the fast path is a different policy.
 * Checked over random heats with deliberate ties (small integer counts, so equal heats are common),
 * units busy at random, and four layers whose resident counts differ, so every tie-break is
 * exercised. */
TEST(heat_step_proposes_exactly_what_repeated_picks_would) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                              std::to_string(e) + ".weight",
                                          1 * MiB, RAD_ACCESS_CONDITIONAL, l, e);
            f.op("moe_gemm", {w}, false);
        }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 29;
    cfg.host_pool_mib = 128;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    Plan plan;
    CHECK_OK(Planner(in).plan(&plan));

    uint32_t x = 0x9E3779B9u;
    auto rnd = [&](uint32_t n) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x % n; };
    int agreed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        HeatParams hp;
        hp.moves_per_dispatch = 1 + (int)rnd(8);
        hp.min_gain = (float)rnd(3);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(3);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        /* Uneven residency across layers, so the resident-count tie-break has work to do. */
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
        std::vector<Swap> want;
        for (int mv = 0; mv < hp.moves_per_dispatch; ++mv) {
            Swap sw;
            if (!ref.pick(&sw)) break;
            ref.set_busy(sw.promote, true);
            ref.set_busy(sw.demote, true);
            want.push_back(sw);
        }
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].promote == want[i].promote && got[i].demote == want[i].demote;
        for (int32_t u = 0; same && u < n_unit; ++u) same = h.busy(u) == ref.busy(u);
        CHECK(same);
        if (same) ++agreed;
    }
    CHECK_EQ(agreed, 400);
}

/* TWO SLAB CLASSES, which is what a rank holding an uneven slice of every expert has: three blocks
 * of one parity's experts and two of the other's, so the even experts' units are one size and the
 * odd experts' another. A swap has to stay inside a class -- the promotion takes a slot of its own
 * class and the demotion frees one of its own -- each class has its own move budget, and step()
 * still has to propose exactly what repeated picks would. */
TEST(heat_swaps_stay_inside_a_slab_class) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                              std::to_string(e) + ".weight",
                                          (e % 2 ? 2 : 3) * MiB, RAD_ACCESS_CONDITIONAL, l, e);
            f.op("moe_gemm", {w}, false);
        }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 81;
    cfg.host_pool_mib = 256;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    Plan plan;
    CHECK_OK(Planner(in).plan(&plan));
    int expert_classes = 0;
    for (const SlabClass& sc : plan.classes) expert_classes += sc.name.rfind("expert", 0) == 0;
    CHECK_EQ(expert_classes, 2);

    uint32_t x = 0x2545F491u;
    auto rnd = [&](uint32_t n) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x % n; };
    int agreed = 0, crossed = 0, proposed = 0;
    for (int trial = 0; trial < 400; ++trial) {
        HeatParams hp;
        hp.moves_per_dispatch = 1 + (int)rnd(8);
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
        /* Only one parity routed in some layers, so a class can run out of candidates at one end
         * while the other still has pairs to propose. */
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

        /* The reference: repeated picks, each class taking at most `moves_per_dispatch` -- the
         * budget is per class. A pick from a class already at its budget is set aside, busy in
         * the copy only. */
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
            if (n >= hp.moves_per_dispatch) {
                aside[(size_t)sw.promote] = aside[(size_t)sw.demote] = 1;
                continue;
            }
            ++n;
            want.push_back(sw);
        }
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = got.size() == want.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].promote == want[i].promote && got[i].demote == want[i].demote;
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

/* The same agreement ACROSS DISPATCHES. step() decides a dispatch that proposes nothing from each
 * layer's two ends, which are kept current through decays, credits, tier changes and units going
 * in and out of flight rather than recomputed -- so the check that matters is a long run of all
 * four, compared dispatch by dispatch against pick() on a copy, including why it stopped. */
TEST(heat_step_agrees_with_picks_over_a_run_of_dispatches) {
    Fixture f;
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m}, false);
    for (int l = 0; l < 4; ++l)
        for (int e = 0; e < 16; ++e) {
            const rad_weight w = f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                              std::to_string(e) + ".weight",
                                          1 * MiB, RAD_ACCESS_CONDITIONAL, l, e);
            f.op("moe_gemm", {w}, false);
        }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 29;
    cfg.host_pool_mib = 128;
    PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
    Plan plan;
    CHECK_OK(Planner(in).plan(&plan));

    uint32_t x = 0x2545F491u;
    auto rnd = [&](uint32_t n) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x % n; };
    for (int run = 0; run < 20; ++run) {
        HeatParams hp;
        hp.moves_per_dispatch = 1 + (int)rnd(3);
        hp.min_gain = 1.0f + (float)rnd(4);
        hp.hysteresis = 1.0f + 0.25f * (float)rnd(4);
        HeatEngine h;
        CHECK_OK(h.init(plan, hp));
        if (!h.enabled()) return;
        const int32_t n_unit = (int32_t)plan.units.size();
        int agreed = 0, proposed = 0;
        for (int d = 0; d < 300; ++d) {
            /* A dispatch's histograms: mostly quiet, a few hot experts, the occasional burst. */
            for (int l = 0; l < 4; ++l) {
                int32_t counts[16];
                for (int e = 0; e < 16; ++e)
                    counts[e] = rnd(5) == 0 ? (int32_t)rnd(d % 37 == 0 ? 40 : 6) : 0;
                h.observe(l, counts, 16, 1.0f);
            }
            HeatEngine ref = h;
            std::vector<Swap> want;
            for (int mv = 0; mv < hp.moves_per_dispatch; ++mv) {
                Swap sw;
                if (!ref.pick(&sw)) break;
                ref.set_busy(sw.promote, true);
                ref.set_busy(sw.demote, true);
                want.push_back(sw);
            }
            std::vector<Swap> got;
            CHECK_OK(h.step(&got));
            bool same = got.size() == want.size();
            for (size_t i = 0; same && i < got.size(); ++i)
                same = got[i].promote == want[i].promote && got[i].demote == want[i].demote;
            CHECK(same);
            if (same) ++agreed;
            proposed += (int)got.size();
            /* What the mover does with them: most land, some are abandoned, and now and then a
             * unit moves for a reason of its own. */
            for (const Swap& sw : got) {
                if (rnd(4) == 0) { h.abandon(sw); continue; }
                h.set_tier(sw.promote, Tier::VRAM);
                h.set_tier(sw.demote, Tier::HostPinned);
                h.set_busy(sw.promote, false);
                h.set_busy(sw.demote, false);
            }
            if (rnd(10) == 0) {
                const int32_t u = (int32_t)rnd((uint32_t)n_unit);
                h.set_tier(u, h.tier(u) == Tier::VRAM ? Tier::HostPinned : Tier::VRAM);
            }
        }
        CHECK_EQ(agreed, 300);
        CHECK(proposed > 0);
    }
}

/* THE HISTOGRAM IS LINED UP WITH THE HEAT BY EXPERT ID, so the shapes that break that are the ones
 * to check: a layer whose expert count is not a whole number of blocks, an expert id no unit
 * holds, an expert that may not move, two units naming one expert, a movable unit of a layer with
 * no expert id, one with no layer, and a histogram shorter or longer than the layer. Every heat and
 * every priced read is compared, dispatch by dispatch, with a per-unit model of the rule, and every
 * proposal with pick() on a copy -- with layers skipped now and then, so an end invalidated by a
 * tier change is rebuilt by step() rather than by the next observe(). */
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
    /* by_expert[l][e]: the unit a histogram entry credits -- the LAST unit naming it, or -1. */
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
    hp.moves_per_dispatch = 3;
    HeatEngine h;
    CHECK_OK(h.init(plan, hp));
    CHECK(h.enabled());

    std::vector<float> want((size_t)n_unit, 0.0f);
    uint64_t routed = 0, resident = 0, stream = 0;
    uint32_t x = 0x9E3779B9u;
    auto rnd = [&](uint32_t n) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x % n; };
    int agreed = 0, proposed = 0, compared = 0;
    for (int d = 0; d < 400; ++d) {
        for (int l = 0; l < 5; ++l) {
            if (rnd(6) == 0) continue;
            /* Shorter than the layer, the layer exactly, or longer; the tail past the layer's last
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
        std::vector<Swap> expect;
        for (int mv = 0; mv < hp.moves_per_dispatch; ++mv) {
            Swap sw;
            if (!ref.pick(&sw)) break;
            ref.set_busy(sw.promote, true);
            ref.set_busy(sw.demote, true);
            expect.push_back(sw);
        }
        std::vector<Swap> got;
        CHECK_OK(h.step(&got));
        bool same = got.size() == expect.size();
        for (size_t i = 0; same && i < got.size(); ++i)
            same = got[i].promote == expect[i].promote && got[i].demote == expect[i].demote;
        CHECK(same);
        if (same) ++agreed;
        proposed += (int)got.size();
        for (const Swap& sw : got) {
            if (rnd(4) == 0) { h.abandon(sw); continue; }
            h.set_tier(sw.promote, Tier::VRAM);
            h.set_tier(sw.demote, Tier::HostPinned);
            h.set_busy(sw.promote, false);
            h.set_busy(sw.demote, false);
        }
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
    CHECK(!h.enabled());   /* all-in-VRAM: the mover never runs either */
}

/* ================================================================== the mover */

/* A DISPATCH'S COMPUTE WORK, standing in for the layer's kernels. Every mover test that drives
 * dispatches submits this to the compute stream; without it such a test asserts something that is
 * not true of a card.
 *
 * Three things are wrong with a dispatch loop that puts nothing on the compute stream:
 *
 *   THE QUARANTINE BARRIER BECOMES VACUOUS. It is an event recorded on the compute stream and its
 *   whole claim is "every kernel that could still name the slot being freed has retired". With an
 *   empty stream that event completes before the mover's transfer has even started, and the
 *   barrier is not exercised at all.
 *
 *   THE LEAD GATE BECOMES A NO-OP. Its host wait is what paces the dispatch thread to the card
 *   and therefore what supplies the mover with slots; against an empty stream it returns
 *   immediately and paces nothing.
 *
 *   AND THE LOOP OUTRUNS THE TRANSFERS IT IS WAITING FOR. A transfer lands in wall-clock
 *   microseconds; an iteration of an empty loop is nanoseconds. Bounding such a loop by a
 *   DISPATCH COUNT asserts a relationship between dispatch rate and link latency that does not
 *   exist: on a card a small H2D copy is still in flight many iterations later, and on the host
 *   backend the stream's worker thread may not be scheduled at all until something yields.
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

    /* Issued right after begin_dispatch, because that is where a layer's launches go: the mover
     * records its ordering gate AFTER them, which is the whole reason engine_tick sits at the end
     * of a layer dispatch and not at the start. */
    void run(RadStream s) const {
        if (a && b) rad_memcpy_async(a, b, kBytes, s);
    }
};

namespace {

/* THE BOUND ON A LOOP THAT WAITS FOR THE MOVER IS WALL-CLOCK TIME, NOT A DISPATCH COUNT, for the
 * reason above. The lead gate paces this thread to the COMPUTE stream, and nothing paces it to the
 * mover's: on a loaded machine the host backend's mover worker can go unscheduled for hundreds of
 * dispatches while the compute worker keeps turning over, and a loop bounded by a count then fails
 * with the mover working exactly as designed. A deadline fails only when it has stopped. */
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
    /* Lowest first, so the occupied slots pack toward the bottom and the top of the range can be
     * handed back. */
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
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

    s.set_target(99);                  /* past the capacity, and clamped to it */
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
    /* 4 and 5 are free, so the mark falls to 4 at once and stops at the slot unit 3 holds. */
    s.set_target(0);
    CHECK_EQ(s.size(), (int64_t)4);
    CHECK_EQ(s.n_free(), (int64_t)0);
    CHECK_EQ(s.take(), (int64_t)-1);   /* nothing above the target is handed out again */

    s.add_free(3);
    CHECK_EQ(s.size(), (int64_t)3);
    s.add_free(2);
    CHECK_EQ(s.size(), (int64_t)2);    /* the original live set is the floor the target asked for */
}

TEST(a_slot_a_shrink_stranded_is_never_handed_out_twice) {
    SlotPool s;
    s.init(0, 64, 4);
    s.set_target(4);
    CHECK_EQ(s.n_free(), (int64_t)4);
    s.set_target(2);                   /* 2 and 3 were free, so the mark falls at once */
    CHECK_EQ(s.size(), (int64_t)2);
    CHECK_EQ(s.n_free(), (int64_t)2);
    s.set_target(4);                   /* and comes back without duplicating the heap entries */
    CHECK_EQ(s.n_free(), (int64_t)4);
    CHECK_EQ(s.take(), (int64_t)0);
    CHECK_EQ(s.take(), (int64_t)1);
    CHECK_EQ(s.take(), (int64_t)2);
    CHECK_EQ(s.take(), (int64_t)3);
    CHECK_EQ(s.take(), (int64_t)-1);
}

TEST(a_slot_pool_regrows_every_slot_a_shrink_dropped) {
    /* A dropped slot that take() walked past on the way to a lower one. */
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
    /* A slot something held above the target when the shrink came, freed while it waited. */
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
        /* And the capacity is whole again: everything handed back comes back out, once each. */
        for (int64_t k = 2; k < 6; ++k) s.add_free(k);
        CHECK_EQ(s.n_free(), (int64_t)4);
        for (int64_t k = 2; k < 6; ++k) CHECK_EQ(s.take(), k);
        CHECK_EQ(s.take(), (int64_t)-1);
    }
}

TEST(file_tier_alignment_is_the_container_alignment) {
    /* O_DIRECT wants offsets, lengths and buffers on the same alignment, and every RadFileEntry
     * offset is RAD_ALIGN_UNIT-aligned by construction -- which is why the fast path needs no
     * head adjustment at all. */
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
        /* Every unit in this plan is in VRAM or the pinned pool, so every weight has an address.
         * A file-tier unit would legitimately have none until it is staged. */
        if (s) CHECK(s->ptr != nullptr);
    }
    CHECK(rt->lookup(9999u) == nullptr);
}

namespace {
/* A container byte: every position its own value, so a weight read from the wrong offset, or
 * copied to the wrong place inside its unit, compares unequal on almost every byte. */
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
    /* Three routed layers of four 20 MiB experts, with experts 0 and 2 the popular ones, so the
     * plan keeps those resident and reads 1 and 3 of every layer from the container. That puts a
     * resident expert's bytes between two file-tier ones, which the reader must skip rather than
     * read through; it makes a layer's file-tier part (40 MiB) wider than a read window (32), so
     * weights are split across windows; and neither weight of an expert is a whole number of
     * blocks, so a unit is laid out in memory unlike the container. A hole between the layers is
     * skipped too. */
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
    c.deterministic = true;          /* no shadow slots: the residents are exactly the fill's */
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
    int m_rc = m.init(f.prog, p, mc);
    if (m_rc == RAD_E_NOMEM) {
        std::printf("  SKIP not enough free device VRAM for 210 MiB container test\n");
        std::remove(path.c_str());
        return;
    }
    REQUIRE_OK(m_rc);
    RadStream s = nullptr;
    CHECK_OK(rad_stream_create(&s, 0));
    CHECK_OK(m.bind_compute_stream(s));
    FileStager fs;
    CHECK_OK(fs.init(f.prog, p, &m));
    CHECK(fs.enabled());
    CHECK_EQ(fs.n_layers(), (size_t)L);

    /* Every file-tier weight of layer `l` published (and holding its container bytes), or none. */
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

    /* Two passes, the second after the stager has learned where a pass ends. Op 0 is the
     * embedding and layer l's two expert ops are 2l+1 and 2l+2. */
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
    /* Three layers a pass, each two runs at least -- the resident expert between its two
     * file-tier ones is skipped, not read -- and every unit read once. */
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

    /* Every pooled expert is wanted. The loop below is a WHOLE DISPATCH every iteration -- retire,
     * observe, decide, issue -- because that is the only thing a real caller ever does, and the
     * supply model only turns over if each of those four happens repeatedly.
     *
     * A DEMOTION CANNOT BE ISSUED BY A DISPATCH THAT DOES NOT ISSUE. The pool free list starts
     * empty: every pooled unit occupies its slot, so the first round can only promote, out of the
     * shadow slot. The pool slot that promotion frees becomes available only after its copy lands
     * AND its quarantine clears, which on a card is many of these iterations later. A loop that
     * issues for a few dispatches and then only retires therefore finds nothing asking for a pool
     * slot by the time one exists, and `demotions` stays zero forever -- while passing on the host
     * backend, where a copy is a memcpy on a worker thread and lands inside the dispatch that
     * issued it. That difference is the whole reason both backends are tested. */
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
    /* Bounded, not exact: the publish happens out of a queue of transfers that HAVE LANDED --
     * retire() polls rad_event_query and skips anything still in flight -- so how many dispatches
     * it takes is a property of the link, not of the policy. What is asserted is that it
     * converges, and that both halves of the swap actually happen. */

    CHECK(m.stats().promotions >= 1);
    CHECK(m.stats().demotions >= 1);
    CHECK(m.stats().h2d_bytes > 0);
    CHECK(m.stats().d2h_bytes > 0);
    /* The first round found no pool slot for its demotion, and saying so is the whole point of
     * the counter: a starved mover and a satisfied one look identical without it. */
    CHECK(m.stats().starved_pool >= 1);

    const WeightSlot* after = m.residency()->lookup(hw);
    CHECK(after->ptr != before);
    CHECK(after->generation > gen_before);
    CHECK(h.tier(hot) == Tier::VRAM);

    /* Nothing is left busy ONCE THE MOVER IS QUIESCENT: every proposal was either issued and
     * published or handed straight back, or the engine would go quiet on busy flags rather than
     * on policy. The qualifier is load-bearing. A unit is legitimately busy at the instant the
     * loop above broke -- `busy` clears when the move's slot comes out of quarantine, which is
     * several dispatches after the copy lands -- so the invariant can only be read after letting
     * what is in flight drain. Stop proposing, keep retiring, and it must settle. */
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

/* ---- THE LOAN. The KV cache lends the slab VRAM it is not using, in place. These stand a plain
 * device allocation in for the cache's carve: the mover is told the stretches and a loan in tokens,
 * and nothing about what the owner keeps there. */
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

    int build() {
        HeatParams hp;
        RAD_TRY(h.init(hf.plan, hp));
        MoverConfig mc;
        RAD_TRY(m.init(hf.f.prog, hf.plan, mc));
        RAD_TRY(rad_stream_create(&compute, 0));
        RAD_TRY(m.bind_compute_stream(compute));
        RAD_TRY(m.lend_from(Mover::kLendKV, loan.strips));
        planned = resident_count(h, hf.plan);
        return RAD_OK;
    }
    ~LoanRig() { if (compute) { rad_stream_sync(compute); rad_stream_destroy(compute); } }

    /* One whole dispatch, as the engine runs one: retire, the step's work, then the loan's fill. */
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
    /* Slots, and not memory the slab holds: the stretches are the owner's until it lends them. */
    CHECK(r.m.flex_capacity() > 0);
    CHECK_EQ(r.m.flex_offered_bytes(), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    /* ...and a dispatch with nothing lent fills nothing. */
    CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().promotions, (uint64_t)0);
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK(r.m.flex_offered_bytes() > 0);
    CHECK_EQ(r.m.lend_from(Mover::kLendKV, r.loan.strips), RAD_E_STATE);   /* carved once */

    /* A mover given nothing to borrow has nothing to lend into, and that is not an error. */
    HeatFixture hf;
    Mover none;
    MoverConfig nc;
    CHECK_OK(none.init(hf.f.prog, hf.plan, nc));
    CHECK_OK(none.lend_from(Mover::kLendKV, {}));
    CHECK_EQ(none.flex_capacity(), (int64_t)0);
    CHECK_OK(none.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_EQ(none.flex_offered_bytes(), (int64_t)0);
}

/* A LENT SLOT LEFT EMPTY IS VRAM BOUGHT AND WASTED ON EVERY STEP, so a loan is filled in the
 * dispatch it arrives in -- every slot at once, hottest units first -- and not a few units a step. */
TEST(every_lent_slot_is_filled_in_one_dispatch) {
    LoanRig r;
    CHECK_OK(r.build());
    const int64_t slots = r.m.flex_capacity() / r.hf.plan.classes[0].stride;
    const int64_t expect = std::min<int64_t>(slots, r.pooled());
    CHECK(expect > 1);

    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(r.dispatch());
    /* Issued in one call, all of them. */
    int64_t issued = 0;
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u) if (r.h.busy(u)) ++issued;
    CHECK_EQ(issued, expect);

    for (Deadline dl; resident_count(r.h, r.hf.plan) < r.planned + expect && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned + (int32_t)expect);
    CHECK_EQ(r.m.flex_used_bytes(), expect * r.hf.plan.classes[0].stride);
    CHECK(r.m.loan_held(Mover::kLendKV) > 0);
    CHECK(r.m.loan_held(Mover::kLendKV) <= Loan::kAll);

    /* INSIDE THE STRETCHES, AND NOWHERE ELSE. Every unit the fill promoted now reads out of the
     * owner's memory, one slot apart, never overlapping and never in the alignment unit left
     * empty at the top of a stretch. */
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

/* A UNIT IN A LENT SLOT KEEPS ITS HOST COPY, so giving the slot back is a change of address and
 * not a copy across the link. */
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

/* A REQUEST THAT CANNOT WAIT GETS THE MEMORY BEFORE THE CALL RETURNS: every unit past the loan is
 * back on its host copy, and nothing that could still read the slot is in flight. */
TEST(a_recall_takes_the_slots_back_before_it_returns) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    CHECK_OK(r.dispatch());                 /* a fill in flight -- the hard case */
    CHECK(any_busy(r.h, r.hf.plan));

    CHECK_OK(r.m.reclaim(Mover::kLendKV, 0, &r.h));
    CHECK_EQ(r.m.loan_held(Mover::kLendKV), (int64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), (int64_t)0);
    CHECK_EQ(resident_count(r.h, r.hf.plan), r.planned);
    CHECK(!any_busy(r.h, r.hf.plan));
    for (int32_t u = 0; u < (int32_t)r.hf.plan.units.size(); ++u)
        CHECK(!r.loan.inside(r.addr(u)));

    /* And the slots are the slab's again the moment the loan comes back. */
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK(resident_count(r.h, r.hf.plan) > r.planned);
}

/* AN IDLE SERVER STILL FILLS WHAT IT IS LENT. There is no step to ride on, so the fill is its own
 * tick; without it the loan made while the server was quiet would sit empty until the next request
 * paid for filling it. */
TEST(an_idle_mover_fills_what_it_is_lent) {
    LoanRig r;
    CHECK_OK(r.build());
    CHECK_OK(r.m.set_loan(Mover::kLendKV, Loan::kAll));
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.hf.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    CHECK(resident_count(r.h, r.hf.plan) > r.planned);
    CHECK(!any_busy(r.h, r.hf.plan));
}

/* TWO LENDERS MOVE INDEPENDENTLY. The arena takes its memory back for every prefill while the KV
 * cache goes on lending, so a recall of one must leave the other's slots exactly as they were --
 * one mark over both would either hold the arena's memory past its recall or give up the cache's
 * with it. */
TEST(a_second_lender_is_recalled_without_touching_the_first) {
    LoanRig r;
    CHECK_OK(r.build());
    Loan arena;
    /* The arena lends in bytes: a unit of its measure is one byte. */
    for (LoanStrip& st : arena.strips) { st.unit_bytes = 1; st.unit_tokens = 1; }
    CHECK_OK(r.m.lend_from(Mover::kLendArena, arena.strips));
    CHECK_EQ(r.m.lend_from(Mover::kLendArena, arena.strips), RAD_E_STATE);
    CHECK_EQ(r.m.loan_ceiling(Mover::kLendArena) > 0, true);

    /* A PART of the cache lent, so the units the pool holds cannot all fit in its slots and the
     * fill -- lowest slot first, the cache's segment before the arena's -- reaches both. */
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

    /* And back: the arena's slots fill again while the cache's stay put. */
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
    /* The lowest covered slot first, skipping the segment that covers nothing. */
    CHECK_EQ(s.take(), 0);
    CHECK_EQ(s.take(), 1);
    CHECK_EQ(s.take(), 5);
    s.set_target(a, s.seg_lo(a) + 2);
    CHECK_EQ(s.take(), 2);
    /* Shrinking one segment leaves the other alone, and the mark stops at what is held. */
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

    /* Where each staged unit's bytes are BEFORE anything is staged: the pinned pool. Staging
     * republishes it into a slab slot, so a changed pointer is how the test sees a unit land
     * without the mover having to expose its slot table. */
    std::vector<void*> pooled;
    for (const Plan::Prefetch& pf : p.prefetch)
        pooled.push_back(
            m.residency()->lookup(p.units[(size_t)pf.unit].weights.front())->ptr);

    CHECK_OK(m.begin_dispatch(0, nullptr));
    /* At op 0 nothing is due yet, but the mover starts anyway and keeps starting until it runs
     * out of slots: the deadline is the plan's and the lead is the supply's. The ring is two
     * deep, so it gets exactly two ahead and stops -- it does NOT skip past the unit it could not
     * serve, because the schedule is in deadline order and serving a later unit first is how a
     * prefetch turns into a cache. This much is pure slot bookkeeping and is the same on any
     * backend. */
    CHECK_OK(m.prefetch_to_op(0));
    CHECK_EQ(m.stats().prefetched, (uint64_t)RAD_PLACE_STAGE_RING);
    CHECK(m.stats().starved_slab >= 1);
    for (int i = 0; i < RAD_PLACE_STAGE_RING; ++i) {
        const rad_weight w = p.units[(size_t)p.prefetch[(size_t)i].unit].weights.front();
        CHECK(m.residency()->lookup(w)->ptr != nullptr);
        /* The residency table carries the mover's event, so rad_issue waits on the compute stream
         * rather than on the host. */
        CHECK(m.residency()->lookup(w)->ready != nullptr);
    }

    /* Now walk the ops the way a step does, and assert THE CONTRACT OF EXACT PREFETCH: when an op
     * runs, the bytes it names have landed. Not "have landed by dispatch two" -- how many
     * dispatches a 1 MiB copy takes is the link's business, and on a card it is many -- but that
     * it lands at all, with a ring only two slots deep, which is the mover's.
     * Asserting an exact count at an exact dispatch index asserts instead that a copy completes
     * instantly, which is only true of the host backend's memcpy. */
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
            /* In a slab slot (the pointer moved off the pool) and settled (the copy landed). */
            landed = s->ptr != pooled[i] && s->ready == nullptr;
        }
        CHECK(landed);
    }

    /* Every unit in the schedule was staged EXACTLY ONCE. The cursor never rewinds inside a step
     * and never skips the unit it could not serve, so the ring turning over cannot re-fetch
     * anything -- which is the difference between a prefetch and a cache. */
    CHECK_EQ(m.stats().prefetched, (uint64_t)p.prefetch.size());
    /* And it got there through a two-slot ring, so it had to starve and recycle to do it. Without
     * this the test would also pass against a mover that quietly allocated a slot per unit. */
    CHECK(m.stats().starved_slab >= 1);
    CHECK((int64_t)p.prefetch.size() > RAD_PLACE_STAGE_RING);

    rad_stream_destroy(compute);
}

/* A ROUTED LAYER'S POOLED UNITS, STAGED FOR ONE PASS (Mover::stage_units). Each copied unit's
 * weights are published at the copy, settled -- the caller orders their readers with one wait --
 * and hold the pool slot's bytes; a unit the buffer has no room for keeps its pool address and is
 * counted; a resident unit is never copied; and unstaging publishes the pool addresses again. */
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
    REQUIRE(pooled.size() < units.size());           /* some are resident, too */

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

    /* The copies are on the mover's stream: on the host backend a worker thread runs them. */
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

/* Compare the whole buffer, including untouched alignment holes, against bytes made independently
 * of the pool. The input orders deliberately leave pool slots out and revisit them backwards. */
TEST(staging_preserves_exact_spans_across_holes_skips_and_capacity_limits) {
    constexpr int64_t a = RAD_ALIGN_UNIT;
    const int64_t sizes[] = {a, a, a, a, a - 19, a, 2 * a, a, a};
    Fixture f;
    Plan p;
    p.dynamic = true;
    p.classes.resize(2);
    p.classes[0].stride = a;
    p.classes[0].n_shadow = 1;
    p.classes[1].stride = 2 * a;
    for (int32_t u = 0; u < 9; ++u) {
        MoveUnit mu;
        mu.bytes = sizes[u];
        mu.slab_class = u == 6 ? 1 : 0;
        mu.tier = u == 7 ? Tier::VRAM : u == 8 ? Tier::HostPageable : Tier::HostPinned;
        mu.site = mu.tier == Tier::VRAM ? Site::Device : Site::DeviceZeroCopy;
        mu.weights = {
            f.weight("stage." + std::to_string(u) + ".first", 32,
                     RAD_ACCESS_CONDITIONAL, 0, u),
            f.weight("stage." + std::to_string(u) + ".second", sizes[u] - a / 2,
                     RAD_ACCESS_CONDITIONAL, 0, u)};
        for (rad_weight w : mu.weights) f.prog.weights[w].align = 32;
        mu.offset = {0, a / 2};
        p.units.push_back(mu);
    }
    Mover m;
    MoverConfig mc;
    REQUIRE_OK(m.init(f.prog, p, mc));
    RadStream compute = nullptr;
    REQUIRE_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));
    RadEvent after = nullptr, done = nullptr;
    REQUIRE_OK(rad_event_create_local(&after));
    REQUIRE_OK(rad_event_create_local(&done));
    const int64_t capacity = 8 * a;
    char* dst = (char*)rad_dev_alloc(capacity, RAD_MEM_DEVICE);
    REQUIRE(dst != nullptr);

    std::vector<void*> home(f.prog.weights.size(), nullptr);
    std::vector<std::vector<uint8_t>> data(9);
    for (int32_t u = 0; u < 9; ++u) {
        const MoveUnit& mu = p.units[(size_t)u];
        data[(size_t)u].resize((size_t)mu.bytes);
        for (int64_t j = 0; j < mu.bytes; ++j)
            data[(size_t)u][(size_t)j] = container_byte((u + 1) * 3 * a + j);
        for (rad_weight w : mu.weights) home[w] = m.residency()->lookup(w)->ptr;
        CHECK_OK(rad_memcpy_async(home[mu.weights[0]], data[(size_t)u].data(), mu.bytes, compute));
    }
    CHECK_OK(rad_stream_sync(compute));

    auto stage = [&](const std::vector<int32_t>& input, int64_t cap,
                     const std::vector<int32_t>& expected, int result = RAD_OK) {
        constexpr uint8_t untouched = 0xd7;
        std::vector<uint8_t> want((size_t)capacity, untouched), got((size_t)capacity);
        int64_t off = 0;
        uint64_t bytes = 0;
        for (int32_t u : expected) {
            const auto& image = data[(size_t)u];
            std::memcpy(want.data() + off, image.data(), image.size());
            off += a * (u == 6 ? 2 : 1);
            bytes += image.size();
        }
        const MoverStats before = m.stats();
        CHECK_OK(rad_memset_async(dst, untouched, capacity, compute));
        CHECK_OK(rad_event_record(after, compute));
        std::vector<int32_t> staged{9999};
        const int rc = m.stage_units(input, dst, cap, after, done, &staged);
        CHECK_EQ(rc, result);
        CHECK(staged == expected);
        CHECK_EQ(m.stats().h2d_bytes - before.h2d_bytes, bytes);
        CHECK_EQ(m.stats().staged_bytes - before.staged_bytes, bytes);
        CHECK_EQ(m.stats().stage_layers - before.stage_layers,
                 (uint64_t)(result == RAD_OK && !expected.empty()));
        CHECK_EQ(m.stats().staged_units - before.staged_units,
                 result == RAD_OK ? (uint64_t)expected.size() : (uint64_t)0);
        if (result == RAD_OK && !expected.empty())
            CHECK_OK(rad_event_wait(compute, done));
        else
            CHECK_OK(rad_stream_sync(m.transfer_stream()));
        CHECK_OK(rad_memcpy_async(got.data(), dst, capacity, compute));
        CHECK_OK(rad_stream_sync(compute));
        CHECK(got == want);
        for (rad_weight w = 1; (size_t)w < home.size(); ++w) {
            void* expected_ptr = home[w];
            off = 0;
            if (result == RAD_OK) {
                for (int32_t u : expected) {
                    const MoveUnit& mu = p.units[(size_t)u];
                    for (size_t k = 0; k < mu.weights.size(); ++k)
                        if (mu.weights[k] == w) expected_ptr = dst + off + mu.offset[k];
                    off += a * (u == 6 ? 2 : 1);
                }
            }
            CHECK(m.residency()->lookup(w)->ptr == expected_ptr);
            CHECK(m.residency()->lookup(w)->ready == nullptr);
        }
        if (result == RAD_OK) m.unstage_units(staged);
        for (rad_weight w = 1; (size_t)w < home.size(); ++w)
            CHECK(m.residency()->lookup(w)->ptr == home[w]);
    };

    stage({0, 1, 2, 3}, 4 * a, {0, 1, 2, 3});       /* contiguous, exact fit */
    stage({0, 2, 1, 3}, 4 * a, {0, 2, 1, 3});       /* source holes and original order */
    stage({3, 4, 5}, 3 * a, {3, 4, 5});             /* alignment holes stay untouched */
    stage({0, 7, 8, 1}, 2 * a, {0, 1});             /* resident and pageable units skipped */
    stage({}, capacity, {});
    const uint64_t overflow = m.stats().stage_overflow;
    stage({0}, a - 1, {});
    stage({0, 6, 1}, 2 * a, {0, 1});                /* overflow does not stop later fits */
    CHECK_EQ(m.stats().stage_overflow - overflow, (uint64_t)2);
    stage({0, 1, -1}, capacity, {0, 1}, RAD_E_INVAL); /* flush prefix, publish none */
    stage({0, 1, 99}, capacity, {0, 1}, RAD_E_INVAL);

    /* A promotion remains Copying until retire(), even when the backend finishes its bytes.
     * Its host slot still exists, so only the in-flight check can exclude this middle unit. */
    CHECK_OK(m.prefetch_unit_async(2, -1, nullptr));
    stage({1, 2, 3}, 3 * a, {1, 3});
    stage({2, 7, 8}, capacity, {});
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_dev_free(dst, RAD_MEM_DEVICE);
    rad_event_destroy(done);
    rad_event_destroy(after);
    rad_stream_destroy(compute);
}

/* THE PREFILL STAGER'S ORDER over a pass of two routed layers: the first layer is staged at the
 * pass's first op, the second at the first layer's first op, each layer is published back at its
 * pool slots once the next layer starts, and nothing is staged in a pass it did not arm for.
 *
 * NOT AT ITS OWN LAST OP. A step longer than the MoE block's pass of rows issues a layer's ops once
 * per pass, so its last op runs again; released at the first run, the second would read copies
 * the mover is already overwriting with the layer two on, and every token past the first pass of
 * a step would be garbage. */
TEST(the_stager_stages_each_layer_one_layer_ahead_and_releases_it_when_the_next_starts) {
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
    std::vector<char> region((size_t)(2 * half));
    auto in_buf = [&](rad_weight w, int b) {
        const char* p = (const char*)m.residency()->lookup(w)->ptr;
        return p >= region.data() + b * half && p < region.data() + (b + 1) * half;
    };
    auto at_home = [&](const std::vector<rad_weight>& ws) {
        for (rad_weight w : ws) if (m.residency()->lookup(w)->ptr != home[w]) return false;
        return true;
    };

    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), 2 * half, 64));
    REQUIRE(st.enabled());
    CHECK_EQ(st.n_layers(), (size_t)2);

    RadBatch b{};
    b.n_tok = 8;
    st.set_full_arena(true);
    CHECK(!st.begin_pass(&b));                        /* too few rows */
    b.n_tok = 128;
    st.set_full_arena(false);
    CHECK(!st.begin_pass(&b));                        /* the arena's top is lent */
    st.set_full_arena(true);
    CHECK(st.begin_pass(&b));

    bool resync = false;
    CHECK_OK(st.before_op(0, compute, &resync));      /* embed: the first layer goes out */
    CHECK(resync);
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    CHECK(at_home(p1));
    CHECK_OK(st.after_op(0, compute));
    CHECK_OK(st.before_op(1, compute, &resync));      /* attn0: nothing */
    CHECK(!resync);
    CHECK_OK(st.before_op(2, compute, &resync));      /* moe0: the second layer goes out */
    CHECK(resync);
    for (rad_weight w : p1) CHECK(in_buf(w, 1));
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    CHECK_OK(st.after_op(2, compute));                /* moe0 is layer 0's last op... */
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    CHECK_OK(st.before_op(2, compute, &resync));      /* ...and runs again for the next rows */
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    CHECK_OK(st.after_op(2, compute));
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    for (rad_weight w : p1) CHECK(in_buf(w, 1));
    CHECK_OK(st.before_op(3, compute, &resync));      /* attn1 */
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    CHECK_OK(st.before_op(4, compute, &resync));      /* moe1: layer 0 is past every pass */
    CHECK(at_home(p0));
    for (rad_weight w : p1) CHECK(in_buf(w, 1));
    CHECK_OK(st.after_op(4, compute));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p1));
    CHECK_EQ(m.stats().stage_layers, (uint64_t)2);

    /* A pass that ends after its first layer went out leaves nothing published at a copy. */
    CHECK(st.begin_pass(&b));
    CHECK_OK(st.before_op(0, compute, &resync));
    for (rad_weight w : p0) CHECK(in_buf(w, 0));
    CHECK_OK(st.end_pass(compute));
    CHECK(at_home(p0));
    CHECK(at_home(p1));
    /* The copies are the mover stream's, and the region is this test's. */
    CHECK_OK(rad_stream_sync(m.transfer_stream()));
    rad_stream_destroy(compute);
}

/* A STAGED LAYER MAY NOT OUTLIVE ITS PASS. At the arena's full plan the stager's buffers are its
 * own; in a step smaller than the full plan the scheduler lends that whole top of the arena to the
 * expert slab (set_full_arena(false)), and the mover fills the lent slots with units. A copy left
 * published past end_pass would then be overwritten by a DIFFERENT expert's bytes while the
 * residency table still names it, so every weight must read back the bytes the load phase gave it.
 *
 * Three routed layers, because a two-layer plan never stages more than one layer ahead and the
 * wrap branch this guards needed a third. */
TEST(a_staged_layer_does_not_outlive_its_pass_and_survives_the_arena_loan_being_refilled) {
    Fixture f;
    const rad_weight emb = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {emb}, false);
    std::vector<rad_weight> layers[3];
    for (int l = 0; l < 3; ++l) {
        for (int e = 0; e < 8; ++e)
            layers[l].push_back(f.weight("blk." + std::to_string(l) + ".ffn_gate_exps." +
                                          std::to_string(e) + ".weight", 1 * MiB,
                                          RAD_ACCESS_CONDITIONAL, l, e));
        f.op("attn" + std::to_string(l), {}, false);
        f.op("moe" + std::to_string(l), layers[l], false);
    }
    Config cfg = base_config();
    cfg.placement = "expert_tiered";
    cfg.vram_weights_mib = 4;
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

    /* Every weight row holds one byte value of its own, so a row overwritten by another unit's
     * bytes is a wrong byte, not a plausible one. */
    std::vector<void*> home(f.prog.weights.size(), nullptr);
    std::vector<uint8_t> fill((size_t)MiB);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        home[w] = m.residency()->lookup(w)->ptr;
        REQUIRE(home[w] != nullptr);
        std::memset(fill.data(), (int)(uint8_t)w, fill.size());
        REQUIRE_OK(rad_memcpy_async(home[w], fill.data(), (int64_t)fill.size(), compute));
        REQUIRE_OK(rad_stream_sync(compute));
    }

    /* kMaxBufs layers' worth of region, so that whatever RADIANCE_PREFILL_BUFFERS makes the stager
     * split it into, each buffer still holds a whole layer's units. */
    const int64_t half = align_up(1 * MiB, RAD_ALIGN_UNIT) * 8;
    std::vector<char> region((size_t)(PrefillStager::kMaxBufs * half));
    REQUIRE_OK(m.lend_from(Mover::kLendArena,
                           {{region.data() + region.size(), (int64_t)region.size(), 1, 1}}));
    REQUIRE_OK(m.set_loan(Mover::kLendArena, 0));

    PrefillStager st;
    CHECK_OK(st.init(f.prog, plan, &m, region.data(), (int64_t)region.size(), 64));
    REQUIRE(st.enabled());
    REQUIRE_EQ(st.n_layers(), (size_t)3);

    /* The non-final chunk: the arena's full plan, so the buffers are the stager's. */
    RadBatch chunk{};
    chunk.n_tok = 128;
    chunk.n_out = 0;
    chunk.phase = RAD_PHASE_PREFILL;
    st.set_full_arena(true);
    REQUIRE(st.begin_pass(&chunk));
    bool resync = false;
    for (rad_op op = 0; op < 7; ++op) {
        CHECK_OK(st.before_op(op, compute, &resync));
        CHECK_OK(st.after_op(op, compute));
    }
    CHECK_OK(st.end_pass(compute));
    CHECK_OK(rad_stream_sync(m.transfer_stream()));

    /* A step smaller than the full plan lends the top back to the slab, and the mover fills those
     * slots -- writing over whatever a stager left published there. */
    RadBatch small{};
    small.n_tok = 1;
    small.n_out = 1;
    st.set_full_arena(false);
    CHECK(!st.begin_pass(&small));
    CHECK_OK(st.end_pass(compute));
    CHECK_OK(m.set_loan(Mover::kLendArena, (int64_t)region.size()));
    for (int i = 0; i < 4; ++i) CHECK_OK(m.idle_tick(nullptr));
    CHECK_OK(rad_stream_sync(m.transfer_stream()));

    /* Every weight resolves to bytes the load phase wrote -- first and last byte of each row. */
    std::vector<uint8_t> probe(f.prog.weights.size() * 2, 0);
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        const void* p = m.residency()->lookup(w)->ptr;
        REQUIRE(p != nullptr);
        const int64_t n = f.prog.weights[w].logical_bytes;
        CHECK_OK(rad_memcpy_async(&probe[(size_t)w * 2], p, 1, compute));
        CHECK_OK(rad_memcpy_async(&probe[(size_t)w * 2 + 1], (const char*)p + n - 1, 1, compute));
    }
    CHECK_OK(rad_stream_sync(compute));
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w) {
        CHECK_EQ((int)probe[(size_t)w * 2], (int)(uint8_t)w);
        CHECK_EQ((int)probe[(size_t)w * 2 + 1], (int)(uint8_t)w);
    }
    rad_stream_destroy(compute);
}

TEST(mover_reserves_from_the_weight_pool_rather_than_beside_it) {
    /* THE PRODUCTION PATH. Pools has already allocated --vram-weights-mib as one block; a mover
     * that called the allocator beside it would spend that budget twice and the overrun would
     * surface as an out-of-memory from some unrelated launch much later. Reserving from
     * Pools::weights() is also what makes it structurally impossible for a promotion to reach a
     * KV block: there is no path from the weights pool to the KV pool. */
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

    /* Every byte the mover took came out of the stated budgets, and none of it out of KV. */
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
    /* Eight experts, one line. The class key is what makes a report of eleven thousand weights
     * three lines long instead of eleven thousand. */
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
 * Four small functions the planner is built out of, and every one of them is a place where two
 * components have to agree on an answer or a budget overruns a pool. They are exposed for exactly
 * that reason -- the mover, the report and the budget resolver all call them rather than deriving
 * the answer again -- so they are worth pinning on their own rather than only through a plan.
 */

TEST(the_strategy_names_round_trip_and_an_unknown_one_is_refused) {
    for (Strategy s : { Strategy::AllVram, Strategy::LayerOffload, Strategy::ExpertTiered,
                        Strategy::Auto }) {
        Strategy back = Strategy::Auto;
        CHECK_OK(strategy_parse(strategy_name(s), &back));
        CHECK((int)back == (int)s);
    }
    /* The four spellings are the --placement flag's legal set, so they are pinned literally:
       a rename here renames an operator's command line. */
    CHECK_EQ(std::string(strategy_name(Strategy::AllVram)), std::string("all_vram"));
    CHECK_EQ(std::string(strategy_name(Strategy::LayerOffload)), std::string("layer_offload"));
    CHECK_EQ(std::string(strategy_name(Strategy::ExpertTiered)), std::string("expert_tiered"));
    CHECK_EQ(std::string(strategy_name(Strategy::Auto)), std::string("auto"));

    /* AND *out IS UNTOUCHED ON A REFUSAL, so a caller that ignores the status does not get a
       strategy it never asked for -- it gets the one it started with. */
    Strategy keep = Strategy::ExpertTiered;
    CHECK_EQ(strategy_parse("layerOffload", &keep), RAD_E_INVAL);
    CHECK((int)keep == (int)Strategy::ExpertTiered);
    CHECK_EQ(strategy_parse("", &keep), RAD_E_INVAL);
    CHECK_EQ(strategy_parse(nullptr, &keep), RAD_E_INVAL);
    CHECK((int)keep == (int)Strategy::ExpertTiered);
    CHECK_EQ(strategy_parse("all_vram", nullptr), RAD_E_INVAL);
}

/* THE STRONGEST ACCESS CLASS IN A GROUP DECIDES THE GROUP'S, so the ordering is what keeps the
 * heat engine from demoting something the next token needs. VOCAB ranks WITH per-token rather
 * than under it: lm_head and the embedding table are read every token, and what makes them their
 * own class is their size and the gather, not any doubt about whether they are read. */
TEST(access_strength_orders_the_classes_and_is_pessimistic_about_an_unknown_one) {
    CHECK_EQ(access_strength(RAD_ACCESS_PER_TOKEN), access_strength(RAD_ACCESS_VOCAB));
    CHECK(access_strength(RAD_ACCESS_PER_TOKEN) > access_strength(RAD_ACCESS_PER_REQUEST));
    CHECK(access_strength(RAD_ACCESS_PER_REQUEST) > access_strength(RAD_ACCESS_CONDITIONAL));
    CHECK(access_strength(RAD_ACCESS_CONDITIONAL) > access_strength(RAD_ACCESS_RARE));
    CHECK_EQ(access_strength(RAD_ACCESS_RARE), 0);

    /* AN ACCESS CLASS THIS BUILD DOES NOT KNOW IS NOT ONE TO GAMBLE A DEMOTION ON. A plugin
       declaring a class added after this core was built must not have its weights treated as
       rare, which is what a zero default would do. */
    CHECK_EQ(access_strength(9999), access_strength(RAD_ACCESS_PER_TOKEN));
    CHECK_EQ(access_strength(-1), access_strength(RAD_ACCESS_PER_TOKEN));
}

/* THE STORED LAYOUT IS WHAT OCCUPIES A TIER. A kernel that re-layouts its weights stores a
 * different number of bytes from the declared shape, and budgeting the logical size would leave
 * the pool short by the difference on every such weight. */
TEST(weight_bytes_prefers_the_stored_layout_then_the_logical_then_the_shape) {
    WeightInfo w;
    w.decl.dtype = RAD_BF16;
    w.decl.rank = 2;
    w.decl.shape[0] = 4;
    w.decl.shape[1] = 8;
    /* Nothing resolved yet: the declared dtype and shape, which is the state a planner unit test
       builds its fixtures in. */
    CHECK_EQ(weight_bytes(w), (int64_t)(4 * 8 * 2));

    w.logical_bytes = 100;
    CHECK_EQ(weight_bytes(w), 100ll);
    w.stored_bytes = 160;                 /* the layout pass has run and it re-laid this one out */
    CHECK_EQ(weight_bytes(w), 160ll);

    /* A rank-0 weight has no elements rather than one. */
    WeightInfo z;
    z.decl.dtype = RAD_BF16;
    z.decl.rank = 0;
    CHECK_EQ(weight_bytes(z), 0ll);
}

/* THE FOOTPRINT MUST AGREE WITH THE PLANNER, because the VRAM budget resolver calls it before a
 * Planner exists and then hands the planner a budget computed from it. The three rules it has to
 * reproduce are the unit grouping, the per-unit alignment, and the mapped-tier pin. */
TEST(the_weight_footprint_splits_statics_from_the_elastic_expert_plane) {
    Fixture f;
    std::vector<rad_weight> experts;
    build_moe(f, /*n=*/4, /*each=*/2 * MiB, &experts);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk);

    /* One model-level weight and four experts: five units. */
    CHECK_EQ(fp.n_units, 5ll);
    CHECK_EQ(fp.statics, 1 * MiB);
    CHECK_EQ(fp.experts, 4 * 2 * MiB);
    CHECK_EQ(fp.mapped, 0ll);

    /* AND EVERY WEIGHT IS IN EXACTLY ONE OF THE THREE. A weight counted twice, or not at all, is
       a budget wrong by its size and nothing downstream can tell. */
    int64_t declared = 0;
    for (rad_weight w = 1; (size_t)w < f.prog.weights.size(); ++w)
        declared += weight_bytes(f.prog.weights[w]);
    CHECK_EQ(fp.statics + fp.experts + fp.mapped + fp.hosted, declared);
}

TEST(weights_in_one_layer_are_one_unit_and_are_aligned_once_each) {
    /* Two weights in layer 0 and two in layer 1, at a size that is not a multiple of the unit
       alignment, so the padding is visible in the total. */
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
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk);
    CHECK_EQ(fp.n_units, 2ll);              /* the declared group is the unit, not the weight */

    /* ONE align_up A UNIT AND ONE INSIDE IT, which is how Mover::init spends the slab. A budget
       that summed raw bytes would be short by an alignment per unit -- on a 512-expert MoE that
       is real memory, and the failure is an overrun of the pool the operator sized. */
    const int64_t per_unit = align_up(align_up(odd, RAD_ALIGN_UNIT) + odd, RAD_ALIGN_UNIT);
    CHECK_EQ(fp.statics, 2 * per_unit);
    CHECK(fp.statics > 4 * odd);
}

/* A WEIGHT NO DEVICE KERNEL CAN READ NEVER ENTERS A POOL, whatever the budgets say -- and the
 * WHOLE UNIT goes with it, because a unit is what moves. Counting those bytes against VRAM would
 * reserve card memory for something that is only ever read in the container's mapping. */
TEST(a_weight_with_no_device_band_takes_its_whole_unit_to_the_mapped_tier) {
    Fixture f;
    /* Layer 0 has two weights and only a HOST kernel for the op that reads one of them. */
    const rad_weight a = f.weight("blk.0.a.weight", 4 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    const rad_weight b = f.weight("blk.0.b.weight", 2 * MiB, RAD_ACCESS_PER_TOKEN, 0, -1);
    f.op("gemm", { a }, /*host_kernel=*/false);

    OpInfo host_only;
    host_only.op = "host_op";
    host_only.index = (int32_t)f.prog.ops.size();
    host_only.weights = { b };
    Band hb;
    hb.hi = 1 << 20;
    hb.dom[RAD_DOMAIN_HOST].row = &g_host_row;      /* and NO device band at all */
    hb.miss[RAD_DOMAIN_DEVICE] = "no device kernel";
    host_only.bands.push_back(hb);
    f.prog.ops.push_back(host_only);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk);
    CHECK_EQ(fp.n_units, 1ll);
    CHECK_EQ(fp.statics, 0ll);
    CHECK_EQ(fp.mapped, 6 * MiB);          /* both weights: the unit is what moves */
}

/* --embedding-placement. The fixture's embedding is read by an `embed_lookup`, as every arch
 * declares it, and its head by a GEMM: the first is the table the flag moves, the second never. */
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
    OpInfo o;
    o.op = "embed_lookup_q";
    o.index = (int32_t)f.prog.ops.size();
    o.weights = { t };
    Band hb;
    hb.hi = 1 << 20;
    hb.dom[RAD_DOMAIN_HOST].row = &g_host_row;
    hb.miss[RAD_DOMAIN_DEVICE] = "no device kernel";
    o.bands.push_back(hb);
    f.prog.ops.push_back(o);
    return t;
}

/* THE TABLES ARE FOUND BY WHAT READS THEM. A model-level weight only embed_lookup reads is the
 * embedding; one only embed_lookup_q reads is an n-gram table; a head a GEMM reads is neither, and
 * neither is a table a GEMM reads beside the lookup -- a tied head would stream whole over the
 * link every step. Placed in RAM the embedding leaves the statics for the hosted bucket, because
 * the resolver sizes a derived weight budget from the statics. */
TEST(the_placement_flags_find_their_tables_by_what_reads_them) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    add_ngram_table(f, 8 * MiB);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk);
    CHECK_EQ(fp.embedding, 1 * MiB);
    CHECK_EQ(fp.ngram, 8 * MiB);
    CHECK(!fp.ngram_device);
    CHECK_EQ(fp.hosted, 0ll);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 2 * MiB);
    CHECK_EQ(fp.mapped, 8 * MiB);

    weight_footprint(f.prog, &fp, TablePlacement::Ram, TablePlacement::Disk);
    CHECK_EQ(fp.embedding, 1 * MiB);
    CHECK_EQ(fp.hosted, 1 * MiB);
    CHECK_EQ(fp.statics, 2 * 4 * MiB + 1 * MiB);

    CHECK(is_embedding_table(f.prog, 1));
    CHECK(!is_embedding_table(f.prog, 2));                 /* a layer's GEMM operand */
    CHECK(!is_embedding_table(f.prog, 4));                 /* the head */
    CHECK(is_ngram_table(f.prog, 5));
    CHECK(!is_ngram_table(f.prog, 1));

    /* A head tied to the table: the same weight read by the lookup and the logits GEMM. */
    Fixture t;
    const rad_weight m = t.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    t.op("embed_lookup", {m}, false);
    t.op("logits_gemm", {m}, false);
    CHECK(!is_embedding_table(t.prog, m));
    weight_footprint(t.prog, &fp, TablePlacement::Ram, TablePlacement::Disk);
    CHECK_EQ(fp.embedding, 0ll);
    CHECK_EQ(fp.hosted, 0ll);
}

TEST(embedding_ram_goes_to_the_pinned_pool_read_in_place_and_costs_no_vram) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/2, /*per_layer=*/4 * MiB);
    Config cfg = base_config();
    cfg.placement = "all_vram";
    cfg.vram_weights_mib = 9;           /* layers and the head exactly: not the embedding */
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

    /* Vram, the default, leaves it with the other weights. */
    cfg.embedding_resolved = TablePlacement::Vram;
    cfg.vram_weights_mib = 10;
    Plan plan2;
    CHECK_OK(Planner(in).plan(&plan2));
    CHECK(plan2.weight[emb].tier == Tier::VRAM);
    CHECK_EQ(plan2.embedding_host_bytes, 0ll);
}

/* A TABLE THAT IS NOT A WHOLE NUMBER OF PAGES still leaves the report's pinned total at what the
 * pooled units use: the pool slot is charged rounded up and the tier total counts weight bytes, so
 * the report subtracts the same weight bytes it added. */
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
    CHECK_EQ(plan.embedding_host_used, 3 * MiB + 100);
    CHECK(plan.embedding_host_bytes >= plan.embedding_host_used);
    const std::string r = plan.report(f.prog);
    const size_t at = r.find("host pinned");
    CHECK(at != std::string::npos);
    if (at != std::string::npos) {
        const std::string line = r.substr(at, r.find('\n', at) - at);
        CHECK(line.find('-') == std::string::npos);           /* no negative total */
    }
    CHECK(r.find("--embedding-placement ram") != std::string::npos);
}

/* A plan resolved to Ram for a program with no table only embed_lookup reads is a budget split
 * for a different program; it refuses rather than leave the table on a card split without it. */
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
    CHECK_EQ(plan.weight.size(), f.prog.weights.size());   /* finish() ran: the report prints */
    CHECK(!plan.notes.empty());
}

/* --ngram-placement ram leaves the table on the mapped tier, which no pool counts, and names the
 * copy the load phase makes: its reason says where the rows will come from, and the report what
 * the copy costs in RAM. */
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

/* An n-gram table a device kernel reads as well as the host one: --ngram-placement says where it
 * is, and so who gathers. vram counts it with the statics and places it with them; ram keeps it on
 * the mapped tier at the zero-copy site, the pinned copy a card reads, and reports that copy; disk
 * keeps it on the mapped tier at the host site, where the host gather reads the container. */
TEST(an_ngram_table_a_device_kernel_reads_goes_where_the_placement_says) {
    Fixture f;
    build_dense_lookup(f, /*layers=*/1, /*per_layer=*/4 * MiB);
    const rad_weight t = f.weight("ngram.weight", 8 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup_q", {t}, /*host_kernel=*/true);

    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Vram);
    CHECK(fp.ngram_device);
    CHECK(fp.ngram_host);
    CHECK_EQ(fp.ngram, 8 * MiB);
    CHECK_EQ(fp.mapped, 0ll);
    CHECK_EQ(fp.statics, 4 * MiB + 2 * MiB + 8 * MiB);
    for (TablePlacement off : { TablePlacement::Ram, TablePlacement::Disk }) {
        weight_footprint(f.prog, &fp, TablePlacement::Vram, off);
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

/* The host gather's own kernel is what makes disk possible at all: an n-gram table only a device
 * kernel reads has no host site, and the footprint says so. */
TEST(an_ngram_table_with_no_host_kernel_cannot_stay_on_disk) {
    Fixture f;
    const rad_weight t = f.weight("ngram.weight", 8 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed_lookup_q", {t}, /*host_kernel=*/false);
    WeightFootprint fp;
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk);
    CHECK(fp.ngram_device);
    CHECK(!fp.ngram_host);
}

TEST(an_empty_program_has_an_empty_footprint) {
    Fixture f;
    WeightFootprint fp;
    fp.statics = 1;                         /* and the out parameter is reset, not accumulated */
    weight_footprint(f.prog, &fp, TablePlacement::Vram, TablePlacement::Disk);
    CHECK_EQ(fp.statics, 0ll);
    CHECK_EQ(fp.experts, 0ll);
    CHECK_EQ(fp.mapped, 0ll);
    CHECK_EQ(fp.n_units, 0ll);
    /* and a null one is not a crash */
    weight_footprint(f.prog, nullptr, TablePlacement::Vram, TablePlacement::Disk);
}

TEST(ced_prefetch_lookahead_and_lease_safety) {
    Fixture f;
    const int L = 6;
    const int E = 8;
    build_moe_layers(f, L, E, 256 * 1024, 256 * 1024);

    Config c = base_config();
    c.placement = "expert_tiered";
    c.vram_weights_mib = 12;
    c.host_pool_mib = 32;

    PlannerInput in{&f.prog, &c, 0, 1, nullptr, 0};
    Plan p;
    CHECK_OK(Planner(in).plan(&p));
    CHECK(p.fits());

    HeatEngine heat;
    HeatParams hp;
    hp.min_residency_dispatches = 30;
    CHECK_OK(heat.init(p, hp));
    CHECK(heat.enabled());
    CHECK_EQ(heat.min_residency_dispatches(), 30);

    CedConfig ccfg;
    ccfg.enabled = true;
    ccfg.lookahead = 4;
    ccfg.threshold = 0.05f;
    ccfg.top_k = 2;
    ccfg.rank = 16;
    ccfg.hidden_dim = 64;
    ccfg.n_experts = E;
    ccfg.n_clusters = 4;
    ccfg.max_prefetch_bytes = 4 * 1024 * 1024;

    CedPrefetcher prefetcher;
    CHECK_OK(prefetcher.init(f.prog, p, nullptr, &heat, ccfg));
    CHECK(prefetcher.enabled());

    /* 1. Lookahead routing check: ΔL = 4
     * Dispatch at layer 0 targets layer 4 (0 + 4 = 4).
     * Verify that prefetcher correctly maps (layer 4, expert) to unit index in plan. */
    const int32_t u_l4_e0 = prefetcher.find_unit_id(4, 0);
    const int32_t u_l4_e1 = prefetcher.find_unit_id(4, 1);
    CHECK(u_l4_e0 >= 0);
    CHECK(u_l4_e1 >= 0);
    CHECK_EQ(p.units[(size_t)u_l4_e0].layer, 4);
    CHECK_EQ(p.units[(size_t)u_l4_e0].expert, 0);

    /* Demote layer 4 units to HostPinned so they require prefetching into VRAM */
    for (int e = 0; e < E; ++e) {
        int32_t u = prefetcher.find_unit_id(4, e);
        if (u >= 0) heat.set_tier(u, Tier::HostPinned);
    }
    CHECK(heat.tier(u_l4_e0) != Tier::VRAM);
    CHECK(heat.tier(u_l4_e1) != Tier::VRAM);

    /* 2. Dispatch at layer 0 with a dummy hidden state */
    std::vector<float> h0(64, 0.5f);
    CHECK_OK(prefetcher.on_layer_dispatch(0, h0.data(), nullptr));

    /* Candidates were evaluated and prefetch was issued for target units */
    const CedStats& s1 = prefetcher.stats();
    CHECK(s1.predictions > 0);
    CHECK(s1.prefetch_issued > 0);

    /* 3. Hold-down lease protection:
     * Newly prefetched units must be promoted to VRAM and receive a 30-dispatch hold-down lease. */
    int prefetched_count = 0;
    for (int e = 0; e < E; ++e) {
        int32_t u = prefetcher.find_unit_id(4, e);
        if (u >= 0 && heat.tier(u) == Tier::VRAM) {
            CHECK(heat.is_leased(u));
            CHECK_EQ(heat.lease_remaining(u), 30ull);
            ++prefetched_count;
        }
    }
    CHECK(prefetched_count > 0);

    /* 4. Active leases cannot be evicted:
     * When all resident units in the slab class are leased, find_eviction_candidate must return -1. */
    const size_t klass = (size_t)p.units[(size_t)u_l4_e0].slab_class;
    for (size_t u = 0; u < p.units.size(); ++u) {
        if ((size_t)p.units[u].slab_class == klass && heat.tier((int32_t)u) == Tier::VRAM) {
            heat.register_prefetch((int32_t)u);
            CHECK(heat.is_leased((int32_t)u));
        }
    }
    CHECK_EQ(heat.find_eviction_candidate(klass), -1);

    /* 5. Lease safety under prefetcher:
     * When all resident units are actively leased, prefetcher skips eviction rather than evicting. */
    CHECK_OK(prefetcher.on_layer_dispatch(1, h0.data(), nullptr));

    /* 6. Expiration of leases after 30 dispatches:
     * Advance HeatEngine by 30 dispatches: */
    heat.advance_dispatch(30);
    CHECK_EQ(heat.dispatches(), 30ull);

    /* Leases must now be expired */
    CHECK(!heat.is_leased(u_l4_e0));
    CHECK_EQ(heat.lease_remaining(u_l4_e0), 0ull);
    /* Now eviction candidate search succeeds */
    int32_t evict = heat.find_eviction_candidate(klass);
    CHECK(evict >= 0);

    /* 7. Probability threshold tau filtering and top-k capping */
    CedConfig high_tau_cfg = ccfg;
    high_tau_cfg.threshold = 0.999f;
    prefetcher.set_config(high_tau_cfg);
    prefetcher.reset_stats();
    CHECK_OK(prefetcher.on_layer_dispatch(0, h0.data(), nullptr));
    CHECK_EQ(prefetcher.stats().prefetch_issued, 0ull);

    /* 8. Bandwidth throttling check:
     * Set max_prefetch_bytes to 0; any candidate will exceed quota and trigger pcie_throttled */
    CedConfig throttle_cfg = ccfg;
    throttle_cfg.threshold = 0.001f;
    throttle_cfg.max_prefetch_bytes = 0;
    prefetcher.set_config(throttle_cfg);
    prefetcher.reset_stats();
    CHECK_OK(prefetcher.on_layer_dispatch(0, h0.data(), nullptr));
    CHECK(prefetcher.stats().pcie_throttled > 0);

    /* 9. Layer boundary check: dispatching at layer >= (total_layers - lookahead) */
    prefetcher.reset_stats();
    CHECK_OK(prefetcher.on_layer_dispatch(2, h0.data(), nullptr));
    CHECK_EQ(prefetcher.stats().predictions, 0ull);
}

TEST(ced_prefetch_with_active_mover_and_clustering) {
    HeatFixture hf(8, 7);
    Mover m;
    MoverConfig mc;
    CHECK_OK(m.init(hf.f.prog, hf.plan, mc));
    RadStream compute = nullptr;
    CHECK_OK(rad_stream_create(&compute, 0));
    CHECK_OK(m.bind_compute_stream(compute));

    HeatEngine heat;
    HeatParams hp;
    hp.min_residency_dispatches = 30;
    CHECK_OK(heat.init(hf.plan, hp));

    CedConfig ccfg;
    ccfg.enabled = true;
    ccfg.lookahead = 0;
    ccfg.hidden_dim = 64;
    ccfg.rank = 16;
    ccfg.n_experts = 8;
    ccfg.n_clusters = 4;
    ccfg.cluster_top_k = 2;
    ccfg.filter_by_cluster = true;
    ccfg.threshold = 0.05f;
    ccfg.top_k = 2;

    CedPrefetcher prefetcher;
    CHECK_OK(prefetcher.init(hf.f.prog, hf.plan, &m, &heat, ccfg));
    CHECK(prefetcher.enabled());

    /* 1. Locate a host-pinned expert unit to prefetch */
    int32_t target_unit = -1;
    for (size_t u = 0; u < hf.plan.units.size(); ++u) {
        if (!m.is_unit_resident((int32_t)u) && hf.plan.units[u].expert >= 0) {
            target_unit = (int32_t)u;
            break;
        }
    }
    CHECK(target_unit >= 0);
    const int32_t target_cls = hf.plan.units[(size_t)target_unit].slab_class;
    const int target_layer = hf.plan.units[(size_t)target_unit].layer;

    /* 2. Verify Mover queries with real pools for target slab class */
    CHECK(m.has_free_slab_slot((size_t)target_cls));
    CHECK(m.free_slab_slots((size_t)target_cls) > 0);


    /* 3. Evaluate CED projection and dispatch prefetch */
    std::vector<float> h0(64, 0.5f);
    CHECK_OK(prefetcher.on_layer_dispatch(target_layer, h0.data(), nullptr));
    const CedStats& s = prefetcher.stats();
    CHECK(s.predictions > 0);

    /* 4. Issue direct async prefetch of target unit into free slab slot */
    const int rc = m.prefetch_unit_async(target_unit, -1, &heat);
    CHECK_OK(rc);
    CHECK(m.stats().prefetched > 0);
    CHECK(heat.is_leased(target_unit));
    CHECK_EQ(heat.lease_remaining(target_unit), 30ull);

    /* 5. Verify lease immunity: cannot be chosen as eviction victim while leased */
    const int32_t slab_cls = hf.plan.units[(size_t)target_unit].slab_class;
    for (size_t u = 0; u < hf.plan.units.size(); ++u) {
        if (hf.plan.units[u].slab_class == slab_cls && heat.tier((int32_t)u) == Tier::VRAM) {
            heat.register_prefetch((int32_t)u);
        }
    }
    CHECK_EQ(heat.find_eviction_candidate(slab_cls), -1);

    /* 6. Verify expiration and eviction candidate recovery */
    heat.advance_dispatch(30);
    CHECK(!heat.is_leased(target_unit));
    CHECK_EQ(heat.lease_remaining(target_unit), 0ull);
    const int32_t victim = heat.find_eviction_candidate(slab_cls);
    CHECK(victim >= 0);

    /* 7. Verify Clustered Expert Distribution (CED) cluster mass computation */
    std::vector<float> mock_probs = {0.4f, 0.05f, 0.01f, 0.02f, 0.3f, 0.1f, 0.02f, 0.1f};
    std::vector<float> cluster_mass;
    prefetcher.projector().compute_cluster_mass(mock_probs, cluster_mass);
    CHECK_EQ(cluster_mass.size(), 4ull);
    // Cluster 0 (experts 0,1) mass = 0.45, Cluster 2 (experts 4,5) mass = 0.40
    CHECK_NEAR(cluster_mass[0], 0.45f, 1e-4);
    CHECK_NEAR(cluster_mass[2], 0.40f, 1e-4);

    /* 8. Verify report string */
    std::string rep = prefetcher.report();
    CHECK(!rep.empty());
    CHECK(rep.find("CED MoE Prefetcher") != std::string::npos);

    rad_stream_sync(m.transfer_stream());
    rad_stream_destroy(compute);
}

RAD_TEST_MAIN()
