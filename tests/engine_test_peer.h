/* engine_test_peer.h -- the test-side peer of rad::Engine, for the tests that link the engine
 * units built a second time with RAD_TEST_HOOKS (tests/CMakeLists.txt).
 *
 * One definition, in a header, because the peer is a friend: core/engine.h declares
 * `friend struct EngineTestPeer` inside the class, so a second, different definition in another
 * translation unit would be the same one-class-two-definitions error the include list in
 * tests/CMakeLists.txt exists to prevent. Every test unit that needs the peer includes this file.
 */
#pragma once

#include "engine.h"

#include <cstdint>
#include <memory>
#include <string>

namespace rad {

struct EngineTestPeer {
    using StepOut = Engine::StepOut;

    /* The completion branch of collect_step, on an engine that owes nothing. */
    static int  collect(Engine& e, const RadBatch* b, StepOut& so) { return e.collect_step(b, so); }
    static void history_issued(Engine& e, bool on) { e.serial_hist_issued_ = on; }
    static void step_end(Engine& e, RadEvent ev) { e.step_end_ = ev; }

    /* What SchedulerBridge::submit_all does, without a server in between: the test owns the
     * request and the engine has no other way in before serve(). */
    static int  submit(Engine& e, std::unique_ptr<Request> r) {
        int rc = RAD_OK;
        e.sched_.add_all(&r, 1, &rc);
        return rc;
    }

    /* The serial drafter's per-step facts, valid until the next step's issue resets them. */
    static bool history_issued(const Engine& e) { return e.serial_hist_issued_; }
    static int  rows_of(const Engine& e) { return e.serial_rows_; }
    static int  rounds(const Engine& e) { return e.serial_rounds_; }
    static int  done_rows(const Engine& e) { return e.serial_done_; }

    /* /server_info's engine half: what start() resolved, including the drafter it declared. */
    static std::string info(const Engine& e) { return e.engine_info(); }

    /* The history watermark: the position up to which the head's K and V are recorded as written.
     * Public on the scheduler, and the only fact that says whether a pass was recorded rather than
     * merely run. */
    static int64_t watermark(Engine& e, uint64_t request_id) {
        return e.sched_.draft_watermark(request_id);
    }
};

}  /* namespace rad */
