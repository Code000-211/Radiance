/* engine_history_test.cpp -- the row-less history pass, through the Engine's own issue, collect
 * and finish, on a serial head that keeps its own KV.
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
 * same call Engine::start makes after bringup (core/engine.cpp:1641). What runs is real:
 * Scheduler::step's plan, Engine::issue_serial_drafter (serial_begin's round-0 build, the batch,
 * and the record that a pass was issued), Engine::collect_step's row-less branch,
 * Scheduler::commit, and Engine::finish_serial_drafter's SerialHold -- the only caller of
 * Scheduler::serial_end on the step path, and the only place a watermark moves.
 *
 * The issue function needs no ranks for this: advance_all() and collect_issue() loop over an
 * empty rank list, and the barrier is given one party, so both of its releases return at once.
 * What that leaves unexercised is the card side -- no batch is enqueued, no kernel runs -- which
 * is the truth this fixture claims and no more. The batch it builds is the scheduler's, and the
 * record it sets is its own; the test only reads both.
 *
 * The scheduler half of the same claim -- that a mid-prompt chunk with an own-KV head builds a
 * history pass and that only a reported pass moves the watermark -- is
 * a_history_only_rows_watermark_needs_its_pass in sched_draft_test.cpp; this file is the engine
 * half. The Gemma 4/shared-KV case stays its own, in engine_loop_test.cpp, unchanged.
 *
 * Also open, and not to be read as covered: an issued pass whose wait did not complete, and a
 * step with a second participating rank. The engine's loop never finishes a step whose wait
 * failed (collect_step's error goes to fail_loop first), and the host backend completes an
 * unrecorded event immediately, so neither this fixture nor the loop's can produce that state
 * without inventing a sequence the engine cannot enter. engine_step_test.cpp carries the same
 * note.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include <chrono>
#include <memory>
#include <thread>

/* Only FakeKV and FakePC: this binary runs on the real device layer, as the engine does. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "sched_program.h"

namespace {

using rad::Engine;
using rad::EngineTestPeer;
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
 * The two facts the finish reads are the real function's -- the issuance record from the batch it
 * built, the completion fact from the engine's own wait -- and the watermark it moves is the
 * chunk's end, exactly what the pass covered. Two chunks in a row then show the progression: the
 * second starts where the first ended, and no pass redoes what the first one recorded.
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
    /* The batch builder stages each step on the rank's stream, as the engine's rank threads do;
     * this fixture has no rank thread, so it creates the one stream those would. */
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

    /* One end event for both chunks, created the way step_state_init creates it and freed by
     * step_state_fini: one handle for the engine's life, never overwritten. */
    RadEvent ev = nullptr;
    REQUIRE_EQ(rad_event_create_as(&ev, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT), RAD_OK);
    EngineTestPeer::step_end(e, ev);

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
     * finish. `from` is where the chunk starts, `to` where it ends. Returns through REQUIRE on the
     * first fact it cannot continue without, which is why the caller gates the second chunk on
     * `chunk_done`: a REQUIRE leaves the lambda, not the case. */
    bool chunk_done = false;
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

        EngineTestPeer::StepOut so;
        so.n_rows = 0;                      /* the chunk produced no logit row */
        REQUIRE_EQ(EngineTestPeer::issue(e, b->step, so), RAD_OK);

        /* What the real issue made of serial_begin()'s answer: nobody drafts, one round is still
         * asked for because round 0 is the history pass -- and that pass was built. A round count
         * alone would not say so; the record does, and it is set only for a non-null round-0
         * batch. */
        CHECK(EngineTestPeer::history_issued(e));
        CHECK_EQ(EngineTestPeer::rounds(e), 1);
        CHECK_EQ(EngineTestPeer::rows_of(e), 0);
        CHECK_EQ(EngineTestPeer::done_rows(e), 0);
        const RadBatch* hb = EngineTestPeer::sched(e).batch_for(0);
        REQUIRE(hb != nullptr);
        CHECK(!EngineTestPeer::sched(e).serial_batch_failed());
        CHECK_EQ(hb->n_draft_out, 0);
        CHECK_EQ(hb->n_tok, to - from);

        /* What mark_step_end does on rank 0's stream, and the only rank here has no thread to do
         * it: without the record, await_step_end() in collect_step would return over an event
         * nothing enqueued behind, and the pass would not be ordered before the reads below. */
        REQUIRE_EQ(rad_event_record(ev, stream), RAD_OK);

        CHECK_EQ(EngineTestPeer::collect(e, b, so), RAD_OK);
        CHECK(so.complete);                 /* the wait over the issued pass ran */
        CHECK_EQ(hb->ctx_lens[0], from);     /* read after the wait: the batch's arrays live on the
                                             * rank's slab, which the step's upload fills. Covers this
                                             * chunk's context; not proof it started at the watermark:
                                             * with ctx_len == from that holds either way, so the
                                             * watermark check below carries the progression. */

        REQUIRE_EQ(EngineTestPeer::commit(e, 11), RAD_OK);
        CHECK_EQ(EngineTestPeer::sched(e).find(1)->n_computed, to);

        CHECK_EQ(EngineTestPeer::finish(e, so), RAD_OK);
        /* The finish's SerialHold reported history_ran, and serial_end moved the watermark over
         * exactly the history the pass wrote. */
        CHECK_EQ(EngineTestPeer::watermark(e, 1), to);
        chunk_done = true;                  /* last line: only a chunk that ran to its end counts */
    };

    chunk(0, 64);
    if (chunk_done) chunk(64, 128);         /* a failed first chunk leaves nothing to compare */
}

/* The engine's idle branch is the other releaser of a deferred KV free. A request cancelled while
 * a drafter snapshot still names it keeps its blocks (Scheduler::cancel: pending_free); commit()
 * normally gives them back, and with no next step to commit, Engine::step_loop's idle branch
 * drains every rank's stream and calls release_deferred(). This case builds exactly that state on
 * the scheduler the engine itself runs -- a committed step, a block snapshot opened and closed
 * around the cancel -- then runs the real step_loop until its sweep has fired. No direct
 * release_deferred() call: the release under test is the loop's.
 *
 * The pool is sized so the admission after the sweep has teeth: 20 blocks of 16 tokens. Request 1
 * holds 10 after its 160-token prompt; the 192-token prompt submitted next needs 12, so a sweep
 * that did not run leaves free_blocks at 10 and step() without a plan for it. What the rank below
 * contributes is only the branch's entry condition (step_loop binds ranks_[0]); it has no program
 * and issues no pass, so the sync it guards has nothing to drain here -- the release and its
 * effect on the pool are the facts this case asserts. */
TEST(idle_step_loop_sweeps_a_cancelled_requests_deferred_free) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    HostProgram h;
    sched_fake::build_program(h.prog, h.cfg, h.kv, h.pc, h.owned,
                              /*blocks*/ 20, /*states*/ 64, /*max_tok*/ 512, /*max_ctx*/ 4096);

    Engine e(h.cfg);
    RadStream stream = nullptr;
    REQUIRE_EQ(rad_stream_create(&stream, 0), RAD_OK);
    struct SchedGuard {
        Engine&   e;
        RadStream stream;
        bool      inited = false;
        ~SchedGuard() {
            /* The last step's uploads point into the builder's staging ring, which fini() frees;
             * a rank thread has synced its stream by the time the scheduler is finalized, and this
             * fixture must too or stream_destroy would run the queued task over freed memory. */
            if (inited && stream) (void)rad_stream_sync(stream);
            if (inited) EngineTestPeer::fini_sched(e);
            if (stream) rad_stream_destroy(stream);
        }
    } sched_guard{e, stream};
    REQUIRE_EQ(EngineTestPeer::init_sched(e, h.prog, h.cfg, &h.kv, &h.pc, stream), RAD_OK);
    sched_guard.inited = true;
    EngineTestPeer::drafter(e, h.prog.drafter);
    EngineTestPeer::barrier(e, 1);
    EngineTestPeer::add_rank(e);

    auto req1 = std::make_unique<rad::Request>();
    req1->id         = 1;
    req1->prompt.resize(160);
    for (size_t i = 0; i < req1->prompt.size(); ++i)
        req1->prompt[i] = (int32_t)(1 + (i * 37) % 318);
    req1->max_tokens = 8;
    req1->ignore_eos = true;
    req1->sink       = nullptr;
    const int64_t before = h.kv.free_blocks(0);
    REQUIRE_EQ(EngineTestPeer::submit(e, std::move(req1)), RAD_OK);

    /* Its one step, committed the way the loop commits: the plan the block snapshot below reads.
     * The step takes the prompt's ten blocks, so the pool proves the deferral below. */
    REQUIRE(EngineTestPeer::sched(e).step() != nullptr);
    REQUIRE_EQ(EngineTestPeer::commit(e, 11), RAD_OK);
    REQUIRE_EQ(h.kv.free_blocks(0), before - 10);

    /* The drafter's snapshot names request 1; the cancel therefore defers the free, and the
     * snapshot's end does not sweep it. */
    rad::Scheduler::BlockRow rows[4];
    (void)EngineTestPeer::sched(e).block_begin(8, 7, 0, 99, rows, 4);
    CHECK_EQ(EngineTestPeer::sched(e).cancel(1), RAD_OK);
    EngineTestPeer::sched(e).block_end();
    REQUIRE(EngineTestPeer::sched(e).has_deferred());
    CHECK(h.kv.has_sequence(1));

    /* The real loop. With nothing queued it reaches its idle branch, which syncs the ranks and
     * sweeps the deferred free; stop() ends it once that has happened. */
    std::thread loop([&] { EngineTestPeer::step_loop(e); });
    bool swept = false;
    for (int i = 0; i < 1000 && !swept; ++i) {
        swept = !EngineTestPeer::sched(e).has_deferred();
        if (!swept) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    EngineTestPeer::stop(e);
    loop.join();

    CHECK(swept);
    CHECK(!h.kv.has_sequence(1));
    CHECK_EQ(h.kv.free_blocks(0), before);

    /* Freed blocks, so the next admission is planned: 12 blocks against the 20 the sweep gave
     * back. Without the sweep the pool has 10 and this step builds no plan. */
    auto req2 = std::make_unique<rad::Request>();
    req2->id         = 2;
    req2->prompt.resize(192);
    for (size_t i = 0; i < req2->prompt.size(); ++i)
        req2->prompt[i] = (int32_t)(1 + (i * 37) % 318);
    req2->max_tokens = 8;
    req2->ignore_eos = true;
    req2->sink       = nullptr;
    REQUIRE_EQ(EngineTestPeer::submit(e, std::move(req2)), RAD_OK);
    const RadBatch* b2 = EngineTestPeer::sched(e).step();
    REQUIRE(b2 != nullptr);
    const StepPlan& p2 = EngineTestPeer::sched(e).plan();
    REQUIRE_EQ(p2.e.size(), 1u);
    CHECK_EQ(p2.e[0].seq, (uint64_t)2);
    CHECK(p2.e[0].is_prefill);
    CHECK_EQ(EngineTestPeer::commit(e, 11), RAD_OK);
}

}  /* namespace */

RAD_TEST_MAIN()
