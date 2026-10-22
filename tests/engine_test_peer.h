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
#include <vector>

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

    /* The finish half of the step path, for a fixture that supplies the facts a started engine's
     * issue path would: this is Engine::finish_serial_drafter, and its SerialHold is the only
     * caller of Scheduler::serial_end on the step path. */
    static int  finish(Engine& e, const StepOut& so) { return e.finish_serial_drafter(so); }

    /* The issue half, as the step path calls it. A fixture with no ranks runs the function itself:
     * advance_all() and collect_issue() see an empty rank list, and the barrier has one party, so
     * what it builds and records is the real thing on the host side even though no card runs it. */
    static int  issue(Engine& e, int step, StepOut& so) { return e.issue_serial_drafter(step, so); }
    static void barrier(Engine& e, int parties) { e.barrier_ = std::make_unique<RankBarrier>(parties); }
    static void drafter(Engine& e, const RadDrafterDecl& d) { e.drafter_ = d; }

    /* The engine's own scheduler, for a fixture that declares its program itself (sched_program.h)
     * instead of loading one through a plugin. init() is the call start() makes after bringup
     * (core/engine.cpp:1641); fini() is what ~Engine does not do for a scheduler the test started. */
    static Scheduler& sched(Engine& e) { return e.sched_; }
    static int init_sched(Engine& e, Program& prog, const Config& cfg, IKVManager* kv,
                          IPrefixCache* pc) {
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        return e.sched_.init(prog, cfg, kv, pc, rio);
    }
    static void fini_sched(Engine& e) { e.sched_.fini(); }

    /* One step's commit, as the engine does after collect_step: every entry committed the same
     * token with no accepted draft, which is what a prefill chunk's step reports. */
    static int  commit(Engine& e, int32_t token = 0, uint8_t eos = 0) {
        const StepPlan& p = e.sched_.plan();
        std::vector<int32_t> acc(p.e.size(), 0), tok(p.e.size(), token);
        std::vector<uint8_t> end(p.e.size(), eos);
        Scheduler::StepResult r{};
        r.n_seq      = (int)p.e.size();
        r.n_accepted = acc.data();
        r.token      = tok.data();
        r.eos        = end.data();
        return e.sched_.commit(r);
    }

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
