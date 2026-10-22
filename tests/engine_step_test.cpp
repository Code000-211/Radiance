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
 *   - a second participating rank, its success and its failure. ranks_ is filled by start(), and
 *     the cases below create a rank only for its evidence slot: no pass of theirs runs.
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

/* The scheduler reads one DraftStepEvidence per step from the ranks' evidence slots. The rank
 * thread notes every pass of the step's bracket into the same slot, so this is the aggregation the
 * reader depends on: a draft pass's tape must count as the step's, repeated notes must not lose the
 * strongest class, the next step must not inherit anything, and a slot shared by two steps must not
 * answer for the older one. No card is involved: the notes are the exact calls the rank thread
 * makes (Rank::note_evidence), injected through the test peer because no rank thread runs here, and
 * the read is Engine::step_evidence. */
TEST(engine_step_evidence_merges_a_steps_passes_and_isolates_the_next) {
    Engine e(rad::Config{});
    using Ctx = rad::Ctx;

    Ctx::StepEvidence trunk{};
    trunk.step = 5;
    trunk.tape = Ctx::kStepTapePlay;
    trunk.staged = true;

    EngineTestPeer::note_evidence(e, 0, trunk, Ctx::kStepTapeNone, /*moved=*/false);
    /* A draft pass ran inside the same bracket and recorded: the slot keeps Record, not Play. */
    EngineTestPeer::note_evidence(e, 0, trunk, Ctx::kStepTapeRecord, /*moved=*/true);

    rad::DraftStepEvidence ev;
    REQUIRE(EngineTestPeer::step_evidence(e, 5, &ev));
    CHECK(ev.replay_known);
    CHECK(ev.replay == rad::DraftReplay::Record);
    CHECK(ev.staged);
    CHECK(ev.moved);

    /* Repeating the trunk pass's note does not downgrade what the draft pass filed. */
    EngineTestPeer::note_evidence(e, 0, trunk, Ctx::kStepTapeNone, /*moved=*/false);
    REQUIRE(EngineTestPeer::step_evidence(e, 5, &ev));
    CHECK(ev.replay == rad::DraftReplay::Record);
    CHECK(ev.staged);
    CHECK(ev.moved);

    /* The next step has its own slot: a trunk pass with no tape and nothing staged inherits none
     * of the previous step's facts, and step 5 keeps its own. */
    Ctx::StepEvidence next{};
    next.step = 6;
    next.tape = Ctx::kStepTapeNone;
    EngineTestPeer::note_evidence(e, 0, next, Ctx::kStepTapeNone, /*moved=*/false);
    rad::DraftStepEvidence ev6;
    REQUIRE(EngineTestPeer::step_evidence(e, 6, &ev6));
    CHECK(ev6.replay_known);
    CHECK(ev6.replay == rad::DraftReplay::None);
    CHECK(!ev6.staged);
    CHECK(!ev6.moved);
    REQUIRE(EngineTestPeer::step_evidence(e, 5, &ev));
    CHECK(ev.replay == rad::DraftReplay::Record);
    CHECK(ev.staged);
    CHECK(ev.moved);

    /* Across ranks too: one rank's None never overrides another's Record, and a rank that never
     * ran the step is a miss, not a measured None. */
    Ctx::StepEvidence s7{};
    s7.step = 7;
    s7.tape = Ctx::kStepTapePlay;
    EngineTestPeer::note_evidence(e, 0, s7, Ctx::kStepTapeNone, /*moved=*/false);
    EngineTestPeer::note_evidence(e, 1, s7, Ctx::kStepTapeRecord, /*moved=*/true);
    rad::DraftStepEvidence ev7;
    REQUIRE(EngineTestPeer::step_evidence(e, 7, &ev7));
    CHECK(ev7.replay == rad::DraftReplay::Record);
    CHECK(ev7.moved);
    REQUIRE(EngineTestPeer::step_evidence(e, 6, &ev6));
    CHECK(ev6.replay == rad::DraftReplay::None);
    CHECK(!ev6.staged);

    /* A step no rank noted is a pure miss: step 8's slot holds nothing, so the read returns false
     * and writes only replay_known = false; replay/staged/moved stay as the caller handed them in.
     * Here that is a fresh default, so replay reads Unknown -- never a measured None. Reader only:
     * the caller's reset at core/engine.cpp:908 does not run from this test, and a pre-filled `out`
     * would come back unchanged; that normalization stays open. */
    rad::DraftStepEvidence miss;
    CHECK(!EngineTestPeer::step_evidence(e, 8, &miss));
    CHECK(!miss.replay_known);
    CHECK(miss.replay == rad::DraftReplay::Unknown);

    /* Slot aliasing: 5 and 9 share slot 1 (step & 3). The note for 9 replaces the slot, and the
     * read for 5 afterwards is a miss -- not 9's facts answered under the old step number. */
    Ctx::StepEvidence s9{};
    s9.step = 9;
    s9.tape = Ctx::kStepTapeNone;
    EngineTestPeer::note_evidence(e, 0, s9, Ctx::kStepTapeNone, /*moved=*/false);
    rad::DraftStepEvidence ev9;
    REQUIRE(EngineTestPeer::step_evidence(e, 9, &ev9));
    CHECK(ev9.replay_known);
    CHECK(ev9.replay == rad::DraftReplay::None);
    CHECK(!ev9.staged);
    CHECK(!ev9.moved);
    rad::DraftStepEvidence stale5;
    CHECK(!EngineTestPeer::step_evidence(e, 5, &stale5));   /* the slot now names step 9 */
    CHECK(!stale5.replay_known);
}

RAD_TEST_MAIN()
