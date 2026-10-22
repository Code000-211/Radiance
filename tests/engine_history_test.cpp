/* engine_history_test.cpp -- the row-less history pass, through the Engine's own collect and
 * finish, on a serial head that keeps its own KV.
 *
 * Why a second fixture. engine_loop_test's Gemma 4 assistant head declares itself cache-less
 * (rad_drafter_shared_kv, arch/gemma4_bf16/gemma4_bf16.cpp:1937), so Scheduler::serial_begin
 * skips every row that does not draft (core/sched/scheduler.cpp:2266-2267) and a mid-prompt chunk
 * builds no round-0 pass at all. The row-less branch Engine::collect_step repairs
 * (core/engine.cpp:1044) is then unreachable from that run, and the rule it feeds --
 * `hist_ran = serial_hist_issued_ && so.complete` (core/engine_draft.cpp:322) -- is never
 * exercised with a real pass behind it. This file declares the one head that makes a mid-prompt
 * chunk owe its history: a RAD_DRAFT_SERIAL head with a KV cache of its own, i.e. no
 * rad_drafter_shared_kv.
 *
 * Host only, no model file, no plugin. The program is the minimal deployment the scheduler tests
 * run against (sched_program.h); the KV manager and prefix cache are that header's fakes; the
 * engine is a default-constructed one whose scheduler this case initialises directly, with the
 * same call Engine::start makes after bringup (core/engine.cpp:1642). What runs is real:
 * Scheduler::step's plan, serial_begin's round-0 build, Engine::collect_step's row-less branch,
 * Scheduler::commit, and Engine::finish_serial_drafter's SerialHold -- the only caller of
 * Scheduler::serial_end on the step path, and the only place a watermark moves.
 *
 * What it does not reach, and why. Engine::issue_serial_drafter builds a RadBatch per round and
 * enqueues it on every rank through the barrier, which needs a started engine. So the issuance
 * record is set here exactly as that function sets it once round 0's batch came back non-null
 * (core/engine_draft.cpp:192), and the batch it would have read is asserted from the scheduler
 * just above. The scheduler half of that claim -- that a mid-prompt chunk with an own-KV head
 * builds a history pass and that only a reported pass moves the watermark -- is
 * a_history_only_rows_watermark_needs_its_pass in sched_draft_test.cpp; this file is the engine
 * half. The Gemma 4/shared-KV case stays its own, in engine_loop_test.cpp, unchanged.
 *
 * Also open, and not to be read as covered: an issued pass whose wait did not complete. The
 * engine's loop never finishes a step whose wait failed (collect_step's error goes to fail_loop
 * first), and the host backend completes an unrecorded event immediately, so neither this
 * fixture nor the loop's can produce that state without inventing a sequence the engine cannot
 * enter. engine_step_test.cpp carries the same note.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

/* Only FakeKV and FakePC: this binary runs on the real device layer, as the engine does. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "sched_program.h"

namespace {

using rad::Engine;
using rad::EngineTestPeer;
using rad::Scheduler;
using rad::StepEntry;
using rad::StepPlan;

/* The declared program and its fakes, without a Scheduler of their own: the engine owns that one.
 * Declared before the engine, because the scheduler keeps a pointer into `prog`. */
struct HostProgram {
    rad::Program       prog;
    rad::Config        cfg;
    sched_fake::FakeKV kv;
    sched_fake::FakePC pc;
    std::vector<void*> owned;

    ~HostProgram() { for (void* p : owned) std::free(p); }
};

/* A mid-prompt chunk with an own-KV serial head: no logit row, and a round-0 pass built anyway.
 * The two facts the finish needs are the two this case supplies on both sides of the seam -- the
 * issuance record from the batch just built, the completion fact from the engine's own wait --
 * and the watermark it moves is the chunk's end, exactly what the pass covered. Two chunks in a
 * row then show the progression: the second starts at the first one's watermark, and no pass
 * redoes what the first one recorded.
 *
 * 192 tokens against 64 a step: chunks [0,64) and [64,128) are the two the case runs, both short
 * of the prompt end. The tail is not needed here; it has a logit row and engine_loop_test covers
 * what a step with rows does. */
TEST(a_rowless_chunk_with_a_built_pass_moves_the_watermark_on_completion) {
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
    h.cfg.n_spec = 3;
    h.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    h.prog.drafter.depth = 3;
    h.prog.drafter.name  = "mtp-test";
    /* The point of the fixture: this head keeps its own KV, so serial_begin does not skip a row
     * that cannot draft. */
    REQUIRE(!h.prog.drafter_shared_kv);

    Engine e(h.cfg);
    REQUIRE_EQ(EngineTestPeer::init_sched(e, h.prog, h.cfg, &h.kv, &h.pc), RAD_OK);
    EngineTestPeer::drafter(e, h.prog.drafter);

    auto req = std::make_unique<rad::Request>();
    req->id         = 1;
    req->prompt.resize(192);
    for (size_t i = 0; i < req->prompt.size(); ++i)
        req->prompt[i] = (int32_t)(1 + (i * 37) % 318);
    req->max_tokens = 8;
    req->ignore_eos = true;
    req->sp.temp    = 0.0f;
    req->sink       = nullptr;
    REQUIRE_EQ(EngineTestPeer::submit(e, std::move(req)), RAD_OK);

    /* One chunk, in the engine's own order: plan, issue the passes, mark the end, collect, commit,
     * finish. `from` is where the chunk starts, `to` where it ends. */
    auto chunk = [&](int64_t from, int64_t to) {
        const RadBatch* b = EngineTestPeer::sched(e).step();
        REQUIRE(b != nullptr);
        const StepPlan& plan = EngineTestPeer::sched(e).plan();
        REQUIRE_EQ(plan.e.size(), 1u);
        const StepEntry& en = plan.e[0];
        CHECK(en.is_prefill);
        CHECK(!en.produces_token);          /* no logit row: n_rows will be 0 */
        CHECK_EQ(en.ctx_len, from);
        CHECK_EQ(en.ctx_len + en.n_tokens, to);

        /* What issue_serial_drafter's serial_begin() call produces: nobody drafts, one round is
         * still asked for because round 0 is the history pass, and that pass is built. */
        int rounds = h.prog.drafter.depth;
        Scheduler::SerialRow rows[3];
        CHECK_EQ(EngineTestPeer::sched(e).serial_begin(&rounds, rows, 3), 0);
        CHECK_EQ(rounds, 1);
        const RadBatch* hb = EngineTestPeer::sched(e).serial_batch(0, b->step);
        REQUIRE(hb != nullptr);
        CHECK(!EngineTestPeer::sched(e).serial_batch_failed());
        CHECK_EQ(hb->n_draft_out, 0);
        CHECK_EQ(hb->n_tok, to - from);
        CHECK_EQ(hb->ctx_lens[0], from);     /* it starts at the watermark, not at the chunk head */

        /* The issue-time facts, set as the step path sets them from that build. */
        EngineTestPeer::issued(e, rounds, /*rows*/ 0, /*history*/ true);

        RadEvent ev = nullptr;
        REQUIRE_EQ(rad_event_create(&ev), RAD_OK);
        EngineTestPeer::step_end(e, ev);

        EngineTestPeer::StepOut so;
        so.n_rows = 0;                       /* the chunk produced no logit row */
        CHECK_EQ(EngineTestPeer::collect(e, b, so), RAD_OK);
        CHECK(so.complete);                 /* the wait over the issued pass ran */

        REQUIRE_EQ(EngineTestPeer::commit(e, 11), RAD_OK);
        CHECK_EQ(EngineTestPeer::sched(e).find(1)->n_computed, to);

        CHECK_EQ(EngineTestPeer::finish(e, so), RAD_OK);
        /* The finish's SerialHold reported history_ran, and serial_end moved the watermark over
         * exactly the history the pass wrote. */
        CHECK_EQ(EngineTestPeer::watermark(e, 1), to);
    };

    chunk(0, 64);
    chunk(64, 128);

    EngineTestPeer::fini_sched(e);
}

}  /* namespace */

RAD_TEST_MAIN()
