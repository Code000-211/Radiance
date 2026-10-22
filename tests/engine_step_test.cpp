/* engine_step_test.cpp -- Engine::StepOut::complete, the only completion fact a step carries.
 *
 * collect_step sets it after await_step_end() has synced the step's end event and every rank's,
 * and leaves it alone on every other path; the reset for the next step belongs to the issue site
 * (core/engine.cpp:496), which runs before the next step is issued. These cases drive the branch a
 * default-constructed Engine can reach: no model, no ranks, an empty scheduler plan, so the step
 * has no sampled rows.
 *
 * Left open here, and not to be read as covered by the cases below -- each needs a started engine,
 * and Engine exposes no per-step entry point (start/serve/dump only):
 *   - a pending end event. The host backend completes an unrecorded event immediately
 *     (core/device/host.cpp), so there is no deterministic way to hold a wait in flight.
 *   - a second participating rank, its success and its failure. ranks_ is filled by start().
 *   - the reset of `complete` at the issue site, and the finish/watermark path
 *     (Engine::finish_serial_drafter): that one also needs serial_done_ > 0, slots_host_ and
 *     draft_out_, which only a run fills.
 */
#include "rad_test.h"

#include "engine.h"

namespace rad {

/* The friend core/engine.h declares under RAD_TEST_HOOKS. Compiled into this unit and into the
 * test copy of the engine units (tests/CMakeLists.txt), never into rad_core. */
struct EngineTestPeer {
    using StepOut = Engine::StepOut;
    static int  collect(Engine& e, const RadBatch* b, StepOut& so) { return e.collect_step(b, so); }
    static void history_issued(Engine& e, bool on) { e.serial_hist_issued_ = on; }
    static void step_end(Engine& e, RadEvent ev) { e.step_end_ = ev; }
};

}  /* namespace rad */

namespace {

using rad::Engine;
using rad::EngineTestPeer;

/* A step with nothing owed returns before any wait and reports no completion. */
TEST(engine_step_nothing_owed_returns_before_any_wait) {
    Engine e(rad::Config{});
    EngineTestPeer::StepOut so;
    so.n_rows = 0;
    CHECK_EQ(EngineTestPeer::collect(e, nullptr, so), RAD_OK);
    CHECK(!so.complete);
}

/* A history pass that was issued is waited on. With one rank, step_end_ is the end event of the
 * only participating rank. The engine owns the event from here: step_state_fini destroys it. */
TEST(engine_step_a_waited_history_pass_is_complete) {
    Engine e(rad::Config{});
    EngineTestPeer::StepOut so;
    so.n_rows = 0;
    RadEvent ev = nullptr;
    REQUIRE_EQ(rad_event_create(&ev), RAD_OK);
    EngineTestPeer::history_issued(e, true);
    EngineTestPeer::step_end(e, ev);
    CHECK_EQ(EngineTestPeer::collect(e, nullptr, so), RAD_OK);
    CHECK(so.complete);
}

/* A failed wait returns the error and does not set the fact. The failure here is a missing
 * handle, not a participating rank's error -- see the open list at the top. */
TEST(engine_step_a_failed_wait_stays_incomplete) {
    Engine e(rad::Config{});
    EngineTestPeer::StepOut so;
    so.n_rows = 0;
    EngineTestPeer::history_issued(e, true);   /* no end event was ever set */
    CHECK(EngineTestPeer::collect(e, nullptr, so) < 0);
    CHECK(!so.complete);
}

/* collect_step does not clear the previous step's `complete`. A second call that owes nothing
 * returns before any wait and leaves the old value readable, which is why the issue site resets
 * it and why a reader must know which step the fact belongs to. */
TEST(engine_step_does_not_clear_a_previous_completion) {
    Engine e(rad::Config{});
    EngineTestPeer::StepOut so;
    so.n_rows = 0;
    RadEvent ev = nullptr;
    REQUIRE_EQ(rad_event_create(&ev), RAD_OK);
    EngineTestPeer::history_issued(e, true);
    EngineTestPeer::step_end(e, ev);
    REQUIRE_EQ(EngineTestPeer::collect(e, nullptr, so), RAD_OK);
    REQUIRE(so.complete);
    EngineTestPeer::history_issued(e, false);
    CHECK_EQ(EngineTestPeer::collect(e, nullptr, so), RAD_OK);
    CHECK(so.complete);   /* stale by construction; core/engine.cpp:496 is what prevents it */
}

}  /* namespace */

RAD_TEST_MAIN()
