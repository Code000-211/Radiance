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
 *   - the reset of `complete` at the issue site, and the publishing half of the finish/watermark
 *     path (Engine::finish_serial_drafter with serial_done_ > 0): that one also needs slots_host_
 *     and draft_out_, which only a run fills. The row-less half of the same function -- a
 *     history-only step whose SerialHold ends the snapshot with history_ran -- is covered in
 *     engine_history_test.cpp, on a scheduler a default-constructed engine can hold.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

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

/* A history pass that was issued is waited on and the wait succeeds. This engine has no ranks, so
 * no rank took part: the wait covers step_end_ alone, the host event set below, and says nothing
 * about any rank's completion. What the case shows is the branch -- an issued pass that is waited
 * on sets the fact -- not a multi-rank step. The engine owns the event from here: step_state_fini
 * destroys it. */
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

/* collect_step does not clear the previous step's `complete`: a second call that owes nothing
 * returns before any wait and leaves the old value readable. That is all this case shows -- that
 * the branch does not reset the flag. It does not prove the reset at a real step change, which
 * happens at the issue site (core/engine.cpp:496) and is only reachable from a started engine
 * (see the open list at the top). */
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
