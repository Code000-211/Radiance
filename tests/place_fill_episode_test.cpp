/* place_fill_episode_test.cpp -- the start fill as an episode: its waves, its turnover reserve,
 * and the brackets that follow it with no idle tick between them.
 *
 * Three production dimensions the placement review named as untested (MAIN-6, the findings open on
 * 8699a27). Each case asserts the invariant the fix has to establish; on 8699a27 the two that
 * carry a finding fail and name the line.
 *
 *  - "more fill units than move records". The base pool is `max(2*(shadow+turnover), 4)` records,
 *    but Mover::lend_from adds one per lent slot and Mover::fill only ever takes slots past
 *    n_fixed, so a start fill of the whole loan is not record-limited: it goes out in one wave.
 *    The first case pins that (green). Measured with the per-slot records removed locally, it goes
 *    red with `no_record` set and the fill stopping after four slots -- which is the review's
 *    premise, true of the pool as `init` sizes it, not of the pool a loan is filled from.
 *  - "turnover > 0". `Mover::fill_settled()` must not answer "materialised" while
 *    `turnover_slots` lent slots are empty: the reserve is part of the start placement.
 *    Engine::start_placement_settled() reads exactly that answer, resolves there and freezes the
 *    drift exemption; without the check the first regular step fills the reserve and its
 *    promotions read as moves (measured on 8699a27: 2). The case pins the fixed answer.
 *  - "continuous requests". A serving engine has no idle tick between brackets and grows the loan
 *    after each of them (Engine::loan_tick(idle=false)). Mover::fill decides `initial` from
 *    `dispatches`, so a wave issued after the first observation is not initial and counts as
 *    drift. The third case separates the two: growth while the start episode is still open (zero
 *    dispatches) is the start fill; growth after the first observation is a real move, as
 *    `Engine::rank_fill_exempt` freezes the exemption at the resolve. There is no blanket
 *    initial-fill exception for later growth (measured on 8699a27: 16 of 24).
 *
 * The fixture is the planner's own deployment (place_test.cpp's shape: a routed layer on the host
 * pool, the rest resident), a plain device allocation standing in for the KV carve, and the mover's
 * real fill/balance/idle_tick. Nothing here re-implements the fill: the cases assert on HeatStats,
 * MoverStats and Mover::fill_settled().
 *
 * The glob in tests/CMakeLists.txt picks this file up with no list entry (it includes no engine.h).
 * The engine-level half -- step_loop reading start_placement_settled() with a real mover and a
 * queue rather than the weak seam -- is not here: a served dense host fixture plans every weight
 * as static (the resolved weights budget covers statics), so it has no movable unit and never
 * lends, and a MoE fixture the planner would tier is not available. That proof stays open.
 */
#include "rad_test.h"

#include "device/device.h"
#include "place/heat.h"
#include "place/mover.h"
#include "place/planner.h"

#include <chrono>
#include <cstdlib>
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

Config base_config() {
    Config c;
    c.placement = "auto";
    c.vram_weights_mib = 0;
    c.host_pool_mib = 0;
    return c;
}

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
    /* Everything the stretches can hold. */
    int64_t all() const { return (int64_t)strips.size() * kStrip / kUnitBytes * kUnitTokens; }
};

struct Rig {
    Fixture     f;
    Plan        plan;
    Config      cfg;
    HeatEngine  h;
    Mover       m;
    Loan        loan;
    RadStream   compute = nullptr;
    void*       buf_a = nullptr;
    void*       buf_b = nullptr;
    int         d = 0;
    int         turnover = 0;

    Rig(int n_expert, int strips) : loan(strips) {
        build_moe(f, n_expert, 1 * MiB);
        cfg = base_config();
        cfg.placement = "expert_tiered";
        cfg.vram_weights_mib = 8;
        cfg.host_pool_mib = 64;
        PlannerInput in{&f.prog, &cfg, 0, 1, nullptr, 0};
        Planner(in).plan(&plan);
    }
    ~Rig() {
        if (compute) { rad_stream_sync(compute); rad_stream_destroy(compute); }
        if (buf_a) rad_dev_free(buf_a, RAD_MEM_DEVICE);
        if (buf_b) rad_dev_free(buf_b, RAD_MEM_DEVICE);
    }

    int build() {
        HeatParams hp;
        RAD_TRY(h.init(plan, hp));
        MoverConfig mc;
        mc.turnover_slots = turnover;
        RAD_TRY(m.init(f.prog, plan, mc));
        RAD_TRY(rad_stream_create(&compute, 0));
        RAD_TRY(m.bind_compute_stream(compute));
        buf_a = rad_dev_alloc(2 * MiB, RAD_MEM_DEVICE);
        buf_b = rad_dev_alloc(2 * MiB, RAD_MEM_DEVICE);
        if (!buf_a || !buf_b) return RAD_E_NOMEM;
        RAD_TRY(m.lend_from(Mover::kLendKV, loan.strips));
        return RAD_OK;
    }

    /* One dispatch as the engine runs it: retire, the step's work, then the loan's fill. */
    int dispatch() {
        RAD_TRY(m.begin_dispatch(d++, &h));
        if (buf_a && buf_b) rad_memcpy_async(buf_a, buf_b, 2 * MiB, compute);
        return m.balance(&h);
    }
    /* One fill and the copies it issued, run to completion. The offered loan, not
     * flex_capacity(), bounds how many slots that is. */
    int fill_now() {
        RAD_TRY(dispatch());
        for (Deadline dl; any_busy(h, plan) && !dl.passed();) RAD_TRY(dispatch());
        return RAD_OK;
    }
    int32_t pooled() const {
        int32_t n = 0;
        for (int32_t u = 0; u < (int32_t)plan.units.size(); ++u)
            if (plan.units[(size_t)u].tier == Tier::HostPinned && plan.units[(size_t)u].movable) ++n;
        return n;
    }
};

}  /* namespace */

/* The move-record pool starts at `max(2*(shadow+turn), 4)` entries, but `Mover::lend_from`
 * adds one per lent slot ("A record for every lendable slot", mover.cpp) -- a fill of the whole
 * loan must not be limited by the base count. This case pins that: a start fill several times the
 * base pool goes out in the dispatch the loan arrives in, with `no_record` untouched. It is the
 * regression for the review's "more fill units than move records" premise, which held only for
 * the pool as `init` sizes it. */
TEST(a_fill_bigger_than_the_base_record_pool_goes_out_in_one_wave) {
    Rig r(/*n_expert*/ 64, /*strips*/ 8);
    REQUIRE_EQ(r.build(), RAD_OK);
    const int64_t slots = r.m.flex_capacity() / r.plan.classes[0].stride;
    REQUIRE(slots > 4);                       /* the base pool's floor: max(2*(shadow+turn), 4) */
    REQUIRE(r.pooled() > slots);

    CHECK_OK(r.m.set_loan(Mover::kLendKV, r.loan.all()));
    /* One dispatch, as the engine's first balance: every free lent slot is issued at once. */
    CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().no_record, (uint64_t)0);
    CHECK_EQ(r.m.flex_used_bytes(), r.m.flex_capacity());   /* every lent slot has a copy in flight */
    CHECK(!r.m.fill_settled());                            /* and none has retired yet */

    /* Let the copies land: one start placement, all of it initial. */
    for (Deadline dl; (r.h.stats().promotions < (uint64_t)slots && any_busy(r.h, r.plan)) &&
                      !dl.passed();)
        CHECK_OK(r.dispatch());
    CHECK_EQ(r.m.stats().no_record, (uint64_t)0);
    CHECK_EQ(r.h.stats().promotions, (uint64_t)slots);
    CHECK_EQ(r.h.stats().initial_fill_promotions, r.h.stats().promotions);
    CHECK(r.m.fill_settled());
}

/* The idle tick fills with `turnover=true` and leaves the reserve free -- that is the documented
 * behaviour. `fill_settled()` must not answer "the start placement is materialised" while those
 * slots are empty: Engine::start_placement_settled() reads exactly this answer, so an idle engine
 * resolves its placement there and freezes the drift exemption before the reserve is filled. */
TEST(the_reserved_turnover_slots_keep_the_start_placement_unsettled) {
    Rig r(/*n_expert*/ 24, /*strips*/ 8);
    r.turnover = 2;
    REQUIRE_EQ(r.build(), RAD_OK);
    REQUIRE_EQ(r.m.turnover_slots(), (int64_t)2);
    const int64_t slots = r.m.flex_capacity() / r.plan.classes[0].stride;
    REQUIRE(r.pooled() >= slots);       /* only the reserve stops the fill from filling everything */

    CHECK_OK(r.m.set_loan(Mover::kLendKV, r.loan.all()));
    /* The idle tick, repeatedly: fill with the reserve held free, until nothing is in flight. */
    for (Deadline dl; (r.m.flex_used_bytes() == 0 || any_busy(r.h, r.plan)) && !dl.passed();)
        CHECK_OK(r.m.idle_tick(&r.h));
    for (int k = 0; k < 8 && any_busy(r.h, r.plan); ++k) CHECK_OK(r.m.idle_tick(&r.h));

    CHECK_EQ(r.m.flex_used_bytes(), (slots - 2) * r.plan.classes[0].stride);
    CHECK_EQ(r.h.stats().promotions, (uint64_t)(slots - 2));
    /* Two lent slots are covered by the loan and empty: the placement is not materialised. */
    CHECK_EQ(r.h.stats().initial_fill_promotions, (uint64_t)(slots - 2));

    /* THE INVARIANT. `fill_settled()` answers false until the working reserve is all that is
     * free, so Engine::start_placement_settled() cannot resolve here and freeze the exemption at
     * slots-2 -- which made the first regular step's two promotions read as `moves(frozen)` = 2
     * on 8699a27. The engine holds the prices until those slots are filled and freezes a count
     * that covers them. */
    CHECK(!r.m.fill_settled());

    /* The first regular step's fill (`balance(heat, turnover=false)`) fills the reserve. These are
     * the same start fill continuing. */
    for (Deadline dl; (r.m.flex_used_bytes() < slots * r.plan.classes[0].stride ||
                       any_busy(r.h, r.plan)) && !dl.passed();) {
        CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
        rad_memcpy_async(r.buf_a, r.buf_b, 2 * MiB, r.compute);
        CHECK_OK(r.m.balance(&r.h, /*turnover=*/false));
    }
    CHECK_EQ(r.m.flex_used_bytes(), slots * r.plan.classes[0].stride);
    /* Materialised now, and every promotion of the start fill is inside the exemption a resolve
     * freezes here -- the plan's own fill, no heat decision yet. */
    CHECK(r.m.fill_settled());
    CHECK_EQ(r.h.stats().moves((int64_t)r.h.stats().initial_fill_promotions), (int64_t)0);

    /* Continuous requests: brackets back to back with no idle tick between them, each holding the
     * turnover slots and balancing with `turnover=false`. With the start fill done they add no
     * move, so the drift above is the start fill and not the steady state. */
    const int64_t after_start_fill = r.h.stats().moves((int64_t)r.h.stats().initial_fill_promotions);
    for (int k = 0; k < 4; ++k) {
        CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
        rad_memcpy_async(r.buf_a, r.buf_b, 2 * MiB, r.compute);
        CHECK_OK(r.m.balance(&r.h, /*turnover=*/false));
    }
    for (Deadline dl; any_busy(r.h, r.plan) && !dl.passed();) {
        CHECK_OK(r.m.begin_dispatch(r.d++, &r.h));
        CHECK_OK(r.m.balance(&r.h, /*turnover=*/false));
    }
    CHECK_EQ(r.h.stats().moves((int64_t)r.h.stats().initial_fill_promotions), after_start_fill);
}


/* The multi-wave start fill, driven by the loan actually growing -- what a serving engine does:
 * loan_tick grows the loan in prefill-chunk steps after every bracket (core/engine.cpp, `idle=
 * false`, no idle_tick), so a start loan larger than the first offer arrives in several waves.
 * The waves that arrive before the first observation are the start fill: every heat is zero, the
 * units are the plan's own lent slots, and the amount is a function of the loan and the plan.
 * Growth after the first observation is a heat decision and must stay distinguishable from that
 * start episode: `Mover::fill` marks it `initial=false`, and `Engine::rank_fill_exempt` exempts
 * only the initial count a resolve freezes, so it counts as a move. There is no blanket
 * initial-fill exception for later growth.
 *
 * No record premise here: the pool is sized per lent slot (see the first case), so the waves come
 * from the offer growing, which is what production does. */
TEST(loan_growth_is_the_start_fill_only_before_the_first_observation) {
    Rig r(/*n_expert*/ 64, /*strips*/ 8);
    REQUIRE_EQ(r.build(), RAD_OK);
    const int64_t slots = r.m.flex_capacity() / r.plan.classes[0].stride;
    REQUIRE(slots >= 8);

    /* A loan is measured per stretch, so a small offer still covers the top slot of every stretch:
     * one slot costs 272 tokens (a stride plus the alignment unit, in 16-token units), and here
     * the stretches hold three slots each. */
    CHECK_OK(r.m.set_loan(Mover::kLendKV, 512));
    CHECK_OK(r.fill_now());
    const int64_t offered = (int64_t)r.h.stats().promotions;
    REQUIRE(offered > 0);
    REQUIRE(offered < slots);
    /* Before any observation this is the plan's own fill: initial by definition. */
    CHECK_EQ(r.h.stats().initial_fill_promotions, (uint64_t)offered);
    CHECK(r.m.fill_settled());

    /* The loan grows while the start episode is still open: the next wave covers the second slot
     * of every stretch, still at zero dispatches. It is the same start fill, in a second wave. */
    CHECK_OK(r.m.set_loan(Mover::kLendKV, 768));
    CHECK_OK(r.fill_now());
    const int64_t covered = (int64_t)r.h.stats().promotions;
    REQUIRE(covered > offered);
    REQUIRE(covered < slots);
    CHECK_EQ(r.h.stats().initial_fill_promotions, (uint64_t)covered);
    CHECK(r.m.fill_settled());
    CHECK_EQ(r.h.stats().moves((int64_t)r.h.stats().initial_fill_promotions), (int64_t)0);

    /* The first bracket: the step routes. From here a fill is a heat decision. */
    std::vector<int32_t> counts((size_t)64, 0);
    counts[0] = 4;
    r.h.observe(0, counts.data(), (int32_t)counts.size());
    REQUIRE_EQ(r.h.stats().dispatches, (uint64_t)1);

    /* The loan grows again after the bracket, as loan_tick does. This wave is not the start fill:
     * it is later growth the exemption must not cover. */
    CHECK_OK(r.m.set_loan(Mover::kLendKV, r.loan.all()));
    CHECK_OK(r.fill_now());
    CHECK_EQ(r.h.stats().promotions, (uint64_t)slots);
    CHECK_EQ(r.h.stats().initial_fill_promotions, (uint64_t)covered);
    CHECK(r.m.fill_settled());
    /* The start episode stays exactly the two zero-dispatch waves; the post-bracket wave reads as
     * `slots - covered` moves, which is how a real later drift remains visible. */
    CHECK_EQ(r.h.stats().moves((int64_t)r.h.stats().initial_fill_promotions),
             (int64_t)(slots - covered));
}

RAD_TEST_MAIN()
