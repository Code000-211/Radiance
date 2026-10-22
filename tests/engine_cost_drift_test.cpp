/* engine_cost_drift_test.cpp -- a residency move between the idle tick and the next cost
 * planning, through Engine::step_loop's own path.
 *
 * The scheduler's idle ticks (tier_tick, expire_tick, loan_tick) run after step_finished, so a
 * promotion or demotion they publish leaves no per-step move stamp; the bracket of the closing
 * cycle cannot see it. The loop therefore hands the cumulative count since the profile was bound
 * in at the top of every iteration -- before promote_next()/step() open the next cycle -- and a
 * drifted run must stop there instead of pricing a cycle on a placement it has left, and instead
 * of falling back to the ceiling (docs/ADAPTIVE-MTP.md).
 *
 * What runs here is the production loop, the production scheduler and the engine's existing error
 * path (fail_loop: stop flag, barrier abort, Scheduler::shutdown, loop status). What the fixture
 * cannot produce is a real tier pump -- its rank has no mover that could publish a move -- so the
 * mover's evidence is injected through the test build's weak seam (rad_test_residency_moves,
 * core/engine.cpp). The move itself is the only simulated fact.
 *
 * The ordering is the finding: the seam counts the loop's own reads -- the baseline is the first,
 * the top of every iteration follows -- so the test waits for the third. By then the first
 * iteration has run end to end (its empty cycle opened, its idle branch and its ticks done), and
 * the move published next stands between an idle tick and the next planning. The store stays
 * asynchronous: if it lands after a later top read, that iteration opens a cycle and the one
 * after it is refused. The test thread reads no scheduler state before the loop has joined.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

/* Only FakeKV and FakePC: this binary runs on the real device layer, as the engine does. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "sched_program.h"

namespace {

/* The loop thread's own seam calls: 1 is step_loop's baseline read, 2 the top of the first
 * iteration, 3 the top of the second -- the first moment the first iteration has run end to end.
 * The test publishes the move once that call has happened. */
std::atomic<int>  g_seam_calls{0};
std::atomic<bool> g_first_idle_done{false};
std::atomic<bool> g_moved{false};

}  /* namespace */

/* The test build's seam (core/engine.cpp, weak): the mover's cumulative evidence since the
 * profile was bound. One move, published by the test thread after the loop's own third read. */
extern "C" int64_t rad_test_residency_moves() {
    const bool moved = g_moved.load(std::memory_order_acquire);
    if (g_seam_calls.fetch_add(1, std::memory_order_acq_rel) + 1 >= 3)
        g_first_idle_done.store(true, std::memory_order_release);
    return moved ? 1 : 0;
}

namespace {

using rad::Engine;
using rad::EngineTestPeer;

/* The declared program and its fakes, without a Scheduler of their own: the engine owns that one. */
struct HostProgram {
    rad::Program       prog;
    rad::Config        cfg;
    sched_fake::FakeKV kv;
    sched_fake::FakePC pc;
    std::vector<void*> owned;

    ~HostProgram() { for (void* p : owned) std::free(p); }
};

TEST(a_move_between_the_idle_tick_and_the_next_plan_ends_the_cost_run) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    /* The scheduler's batch builder allocates through the device layer. With HIP compiled in,
     * select_backend() may bind a card before this test can say otherwise; the guarantee this
     * case needs is the compile-time one (configure -DRAD_WITH_HIP=OFF). */
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    HostProgram h;
    sched_fake::build_program(h.prog, h.cfg, h.kv, h.pc, h.owned,
                              /*blocks*/ 4096, /*states*/ 64, /*max_tok*/ 64, /*max_ctx*/ 4096);
    h.cfg.n_spec         = 6;
    h.cfg.draft_policy   = rad::DraftPolicyKind::Cost;
    h.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    h.prog.drafter.depth = 6;
    h.prog.drafter.name  = "mtp-test";

    Engine e(h.cfg);
    /* --draft-profile's reader, before init: the baseline the deployment is admitted on. One
     * usable bucket, so the run prices measured cycles rather than the named fallback. */
    rad::DraftCostProfile prof;
    prof.fingerprint.model     = "test";
    prof.fingerprint.quant     = "q";
    prof.fingerprint.tp_wire   = "w";
    prof.fingerprint.gpus      = "g";
    prof.fingerprint.kernels   = "k";
    prof.fingerprint.kv_format = "bf16";
    prof.fingerprint.placement = "all_vram";
    prof.fingerprint.tp        = 1;
    prof.buckets.emplace_back();
    prof.buckets.back().shape.n_rows        = 1;
    prof.buckets.back().shape.verify_k      = 1;
    prof.buckets.back().shape.verify_tokens = 2;
    prof.buckets.back().shape.rows[1]       = 1;
    prof.buckets.back().shape.draft_k       = 1;
    prof.buckets.back().shape.draft_rows[1] = 1;
    prof.buckets.back().shape.drafter       = rad::DraftDrafter::Serial;
    prof.buckets.back().rounds              = 1;
    prof.buckets.back().draft_round_rows[1] = 1;
    prof.buckets.back().draft_rows_total    = 1;
    prof.buckets.back().mean_ms             = 12.0;
    prof.buckets.back().unc_ms              = 1.0;
    prof.buckets.back().n                   = 3;
    rad::DraftCostRuntime rt;
    rt.fingerprint = prof.fingerprint;
    CHECK_OK(EngineTestPeer::sched(e).draft_costs().configure(prof, rt));
    REQUIRE(EngineTestPeer::sched(e).draft_costs().active());
    CHECK(!EngineTestPeer::sched(e).draft_costs().fallback_admitted());

    RadStream stream = nullptr;
    REQUIRE_EQ(rad_stream_create(&stream, 0), RAD_OK);
    /* The scheduler is this fixture's, not one the engine started, so ~Engine does not fini it.
     * The guard runs that on every exit, including the `return` a REQUIRE takes, and destroys the
     * stream after it. Declared after `e`, so it runs before ~Engine; `inited` keeps fini off a
     * scheduler whose init failed and whose kv_ is therefore unset. */
    struct SchedGuard {
        Engine&   e;
        RadStream stream;
        bool      inited = false;
        ~SchedGuard() {
            if (inited) EngineTestPeer::fini_sched(e);
            if (stream) rad_stream_destroy(stream);
        }
    } sched_guard{e, stream};
    REQUIRE_EQ(EngineTestPeer::init_sched(e, h.prog, h.cfg, &h.kv, &h.pc, stream), RAD_OK);
    sched_guard.inited = true;
    EngineTestPeer::drafter(e, h.prog.drafter);
    EngineTestPeer::barrier(e, 1);
    /* step_loop binds rank 0's device; the loop needs the rank, not a program on it. */
    EngineTestPeer::add_rank(e);

    g_seam_calls.store(0, std::memory_order_relaxed);
    g_first_idle_done.store(false, std::memory_order_relaxed);
    g_moved.store(false, std::memory_order_relaxed);

    int rc = RAD_OK;
    std::atomic<bool> loop_done{false};
    std::thread loop([&] {
        rc = EngineTestPeer::step_loop(e);
        loop_done.store(true, std::memory_order_release);
    });
    struct LoopGuard {
        Engine&      e;
        std::thread& t;
        ~LoopGuard() {
            EngineTestPeer::stop(e);
            if (t.joinable()) t.join();
        }
    } loop_guard{e, loop};

    /* Nothing is queued, so every iteration plans a cycle with no rows and then runs the idle
     * branch. The seam sets this on its third call: the first iteration has run to its end, so the
     * move published here stands between an idle tick and the planning after it. No scheduler
     * state is touched from this thread while the loop runs. */
    bool idled = false;
    for (int i = 0; i < 5000 && !idled; ++i) {
        idled = g_first_idle_done.load(std::memory_order_acquire);
        if (!idled) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(idled);

    g_moved.store(true, std::memory_order_release);

    /* The loop ends itself on the latch; stop() only bounds the wait (a no-op once it has). */
    for (int i = 0; i < 30000 && !loop_done.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    EngineTestPeer::stop(e);
    loop.join();

    CHECK_EQ(rc, RAD_E_STATE);
    CHECK(EngineTestPeer::sched(e).draft_cost_aborted());
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
    CHECK_EQ(EngineTestPeer::sched(e).draft_costs().reason(),
             std::string("residency drifted from the fingerprint's start placement"));
    CHECK(!EngineTestPeer::sched(e).draft_costs().fallback_admitted());
    /* Read after the join: the loop owned this state until it returned. The plan in force is an
     * idle cycle's -- nothing is queued, so it names no row -- and the refusal opened none after
     * it. The store was still asynchronous, so this proves "a cycle after the move was refused",
     * not strictly "the immediately next one". */
    const rad::DraftPlan* p = EngineTestPeer::sched(e).draft().plan();
    REQUIRE(p != nullptr);
    CHECK_EQ(p->n_seq, 0);
}

}  /* namespace */

RAD_TEST_MAIN()
