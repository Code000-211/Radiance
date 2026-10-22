/* engine_start_fill_test.cpp -- serve()'s start-fill completion through Engine::prepare_serving,
 * with a real planner plan and a real Mover on the rank.
 *
 * The planner's choice is not under test: the plan is built by Planner over a fixture program and
 * installed on the rank through the peer (test hooks only, no production override, no runtime
 * option). Everything the engine does with it afterwards is production code:
 *
 *  - complete_start_fill() runs its held-turnover wave even when the mover already answers
 *    fill_settled() true at entry -- the loop would otherwise run zero times -- and only accepts
 *    the placement once the wave leaves `fill_settled()` true (Review 2, finding 1). The fixture
 *    drives the real mover into the state place_test pins: the idle balance filled everything the
 *    loan covers except the turnover reserve. The engine has to finish it, so the reserve is
 *    initial and the resolve has nothing to move.
 *  - The same call on an already settled mover is the zero-iteration path exactly: the wave runs,
 *    nothing is left to fill, and the proof is unchanged -- no promotion, no move.
 *  - A failing fill must reach serve. The completion returns the status and prepare_serving()
 *    returns it before begin_draft_calibration(), so the run never gets a fingerprint, a profile
 *    or a price for a placement it did not reach (Review 2, finding 2).
 *
 * The deployment around the rank is a fixture (no model, no plugin, no card): the program, the KV
 * carve the loan comes from and the compute stream. The Mover, the loan bookkeeping, the heat
 * engine and every engine-side decision are real.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include "device/device.h"
#include "place/heat.h"
#include "place/mover.h"
#include "place/planner.h"

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t MiB = 1024 * 1024;

const RadKernelInfo g_dev_info = { "fake_dev", "gemm", "gemm", "", "", "",
                                   RAD_DOMAIN_DEVICE, 0, nullptr, 0, nullptr, 0,
                                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_host_info = { "fake_host", "gemm", "gemm", "", "", "",
                                    RAD_DOMAIN_HOST, 0, nullptr, 0, nullptr, 0,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
KernelRow g_dev_row;
KernelRow g_host_row;

/* The fault injected into the fill: 0 is none, a negative status is returned by
 * Engine::complete_start_fill before its first wave. The weak definition in core/engine.cpp
 * leaves every other engine test alone. */
std::atomic<int> g_fill_error{0};

struct Fixture {
    Program prog;

    Fixture() {
        g_dev_row.info  = &g_dev_info;
        g_host_row.info = &g_host_info;
        prog.weights.resize(1);        /* index 0 is the sentinel */
        prog.ctx.rank = 0;
        prog.ctx.world_size = 1;
    }

    rad_weight weight(const std::string& name, int64_t bytes, int access, int32_t layer,
                      int32_t expert) {
        WeightInfo w;
        w.name = name;
        w.decl.dtype = RAD_U8;         /* one byte per element, so bytes are exact */
        w.decl.rank = 1;
        w.decl.shape[0] = bytes;
        w.decl.access = access;
        w.decl.group.layer = layer;
        w.decl.group.expert = expert;
        w.logical_bytes = bytes;
        w.align = RAD_ALIGN_UNIT;
        prog.weights.push_back(w);
        return (rad_weight)(prog.weights.size() - 1);
    }

    void op(const std::string& name, const std::vector<rad_weight>& ws) {
        OpInfo o;
        o.op = name;
        o.index = (int32_t)prog.ops.size();
        o.weights = ws;
        Band b;
        b.hi = 1 << 20;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_dev_row;
        b.miss[RAD_DOMAIN_HOST] = "no host kernel for this geometry";
        o.bands.push_back(b);
        for (rad_weight w : ws) {
            WeightInfo& wi = prog.weights[w];
            if (wi.first_use_op < 0) wi.first_use_op = o.index;
            wi.last_use_op = o.index;
        }
        prog.ops.push_back(o);
    }
};

/* One MoE layer: `n` experts of `each` bytes, plus one dense model-level weight. */
void build_moe(Fixture& f, int n, int64_t each) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", {m});
    for (int e = 0; e < n; ++e) {
        const rad_weight w = f.weight("blk.0.ffn_gate_exps." + std::to_string(e) + ".weight", each,
                                      RAD_ACCESS_CONDITIONAL, 0, e);
        f.op("moe_gemm", {w});
    }
}

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

/* The KV carve the mover is lent, as place_test.cpp builds it: `strips` stretches of device
 * memory, each paying `unit_tokens` tokens per `unit_bytes` of slab. */
struct Loan {
    static constexpr int64_t kStrip = 4 * MiB;
    static constexpr int64_t kUnitBytes = 64 * 1024;
    static constexpr int64_t kUnitTokens = 16;
    void* mem = nullptr;
    std::vector<LoanStrip> strips;
    explicit Loan(int n_strips) {
        mem = rad_dev_alloc(n_strips * kStrip, RAD_MEM_DEVICE);
        for (int k = 0; k < n_strips; ++k) {
            LoanStrip st;
            st.top = (char*)mem + (int64_t)(k + 1) * kStrip;
            st.bytes = kStrip;
            st.unit_bytes = kUnitBytes;
            st.unit_tokens = kUnitTokens;
            strips.push_back(st);
        }
    }
    ~Loan() { if (mem) rad_dev_free(mem, RAD_MEM_DEVICE); }
    int64_t all() const { return (int64_t)strips.size() * kStrip / kUnitBytes * kUnitTokens; }
};

/* The planner's own deployment (place_fill_episode_test.cpp's shape: a routed layer on the host
 * pool, the rest resident) plus the loan and the compute stream the rank's mover runs on. */
struct Rig {
    Fixture   f;
    Plan      plan;
    Config    cfg;
    Loan      loan;
    RadStream compute = nullptr;

    Rig(int n_expert, int strips, int turnover) : loan(strips), turnover_(turnover) {
        build_moe(f, n_expert, 1 * MiB);
        cfg.placement = "expert_tiered";
        cfg.vram_weights_mib = 8;
        cfg.host_pool_mib = 64;
        cfg.draft_policy = DraftPolicyKind::Cost;   /* serve()'s completion path */
        PlannerInput in{ &f.prog, &cfg, 0, 1, nullptr, 0 };
        Planner(in).plan(&plan);
    }
    ~Rig() {
        if (compute) { rad_stream_sync(compute); rad_stream_destroy(compute); }
    }

    int build(Engine& e) {
        RAD_TRY(rad_stream_create(&compute, 0));
        return EngineTestPeer::inject_plan_rank(e, f.prog, plan, turnover_, loan.strips, compute);
    }
    /* Only the reserve stops the start fill from filling every lent slot. */
    int32_t pooled() const {
        int32_t n = 0;
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u)
            if (plan.units[(size_t)u].tier == Tier::HostPinned && plan.units[(size_t)u].movable) ++n;
        return n;
    }
private:
    int turnover_ = 0;
};

}  /* namespace */

/* The test build's seam (core/engine.cpp, weak): a fill that fails. The real host mover has no
 * deterministic failure to drive, so the fault is injected at the one point serve's completion
 * would take it from -- the status the engine must propagate. */
extern "C" int rad_test_start_fill_error() {
    return g_fill_error.load(std::memory_order_relaxed);
}

/* Review 2, finding 1: fill_settled() alone is not the completion proof. The idle balance leaves
 * the turnover reserve free by contract; the engine's completion has to hold turnover and run the
 * balance anyway. Without the wave the reserve stays empty into the first regular step and its
 * promotions read as moves. */
TEST(complete_start_fill_fills_the_reserve_the_idle_balance_left_free) {
    Rig r(/*n_expert*/ 24, /*strips*/ 8, /*turnover*/ 2);
    Engine e(r.cfg);
    REQUIRE_EQ(r.build(e), RAD_OK);
    Mover&      m = EngineTestPeer::mover_of(e, 0);
    HeatEngine& h = EngineTestPeer::heat_of(e, 0);
    const int64_t stride = r.plan.classes[0].stride;
    const int64_t slots  = m.flex_capacity() / stride;
    REQUIRE_EQ(m.turnover_slots(), (int64_t)2);
    REQUIRE(slots >= 8);
    REQUIRE(r.pooled() >= slots);

    CHECK_OK(m.set_loan(Mover::kLendKV, r.loan.all()));
    /* The idle tick fills with turnover held free; repeat (syncing the mover's stream, as the
     * engine's completion does) until nothing is in flight. */
    for (Deadline dl; (m.flex_used_bytes() == 0 || any_busy(h, r.plan)) && !dl.passed();) {
        CHECK_OK(rad_stream_sync(m.transfer_stream()));
        CHECK_OK(m.idle_tick(&h));
    }
    for (int k = 0; k < 8 && any_busy(h, r.plan); ++k) {
        CHECK_OK(rad_stream_sync(m.transfer_stream()));
        CHECK_OK(m.idle_tick(&h));
    }
    CHECK_EQ(m.flex_used_bytes(), (slots - 2) * stride);
    CHECK_EQ((int64_t)h.stats().promotions, slots - 2);
    /* The state the review's repro handed to the completion: the reserve is free. */
    CHECK(!m.fill_settled());

    /* The production call: one held-turnover wave fills the reserve, the next sync retires it. */
    CHECK_OK(EngineTestPeer::complete_start_fill(e));
    CHECK(m.fill_settled());
    CHECK(EngineTestPeer::start_placement_settled(e));
    CHECK_EQ(m.flex_used_bytes(), slots * stride);
    /* The whole start fill is initial -- reserve included -- so the resolve has nothing to move. */
    CHECK_EQ(h.stats().initial_fill_promotions, h.stats().promotions);
    CHECK_EQ(h.stats().moves((int64_t)h.stats().initial_fill_promotions), (int64_t)0);
}

/* The zero-iteration case: the mover is already settled at entry, so the old loop condition
 * (fill_settled()) would never enter its body. The fixed completion runs its wave regardless --
 * the completion proves the reserve by holding turnover and balancing, it does not infer it from
 * the predicate. On a settled mover that wave finds nothing to fill and changes nothing. */
TEST(complete_start_fill_holds_turnover_from_an_already_settled_mover) {
    Rig r(/*n_expert*/ 24, /*strips*/ 8, /*turnover*/ 2);
    Engine e(r.cfg);
    REQUIRE_EQ(r.build(e), RAD_OK);
    Mover&      m = EngineTestPeer::mover_of(e, 0);
    HeatEngine& h = EngineTestPeer::heat_of(e, 0);
    CHECK_OK(m.set_loan(Mover::kLendKV, r.loan.all()));
    CHECK_OK(EngineTestPeer::complete_start_fill(e));
    REQUIRE(m.fill_settled());

    /* Exactly the entry the review named: settled before the call. */
    const uint64_t promotions = h.stats().promotions;
    const uint64_t initial    = h.stats().initial_fill_promotions;
    const int64_t  used       = m.flex_used_bytes();
    CHECK_OK(EngineTestPeer::complete_start_fill(e));
    CHECK(m.fill_settled());
    CHECK(EngineTestPeer::start_placement_settled(e));
    CHECK_EQ(h.stats().promotions, promotions);
    CHECK_EQ(h.stats().initial_fill_promotions, initial);
    CHECK_EQ(m.flex_used_bytes(), used);
    CHECK_EQ(h.stats().moves((int64_t)initial), (int64_t)0);
}

/* Review 2, finding 2: the completion's error is serve's. prepare_serving() runs the loan, the
 * fill and then the calibration check; the injected fill error comes back before the check, so no
 * calibration is armed and the placement stays unresolved (prices held). */
TEST(a_start_fill_error_stops_serving_before_the_profile) {
    Rig r(/*n_expert*/ 24, /*strips*/ 8, /*turnover*/ 2);
    Engine e(r.cfg);
    REQUIRE_EQ(r.build(e), RAD_OK);
    Mover& m = EngineTestPeer::mover_of(e, 0);
    CHECK_OK(m.set_loan(Mover::kLendKV, r.loan.all()));

    g_fill_error.store(RAD_E_IO, std::memory_order_relaxed);
    CHECK_EQ(EngineTestPeer::prepare_serving(e), RAD_E_IO);
    g_fill_error.store(0, std::memory_order_relaxed);

    /* Nothing was claimed: the fill did not run, so it is not settled, the placement cannot
     * resolve, and serve never reached the calibration that arms a profile. */
    CHECK(!m.fill_settled());
    CHECK(!EngineTestPeer::start_placement_settled(e));
    CHECK(!EngineTestPeer::calibrating(e));

    /* The same call without the fault completes the start fill and reaches the check. */
    CHECK_OK(EngineTestPeer::prepare_serving(e));
    CHECK(m.fill_settled());
    CHECK(EngineTestPeer::start_placement_settled(e));
}

RAD_TEST_MAIN()
