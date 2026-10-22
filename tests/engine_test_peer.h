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
/* Engine::Rank is complete here only for its evidence slots: a host fixture has no rank thread,
 * and the aggregation those slots perform is what the peer has to reach. */
#include "engine_priv.h"

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

    /* The stager the engine's own Ctx would use for a pass (Engine::test_set_stager). A fixture
     * whose deployment stages nothing attaches one to check the step's placement evidence. */
    static void stager(Engine& e, OpStager* s) { e.test_set_stager(s); }

    /* The engine's own scheduler, for a fixture that declares its program itself (sched_program.h)
     * instead of loading one through a plugin. init() is the call start() makes after bringup
     * (core/engine.cpp:1641); fini() is what ~Engine does not do for a scheduler the test started. */
    static Scheduler& sched(Engine& e) { return e.sched_; }
    static int init_sched(Engine& e, Program& prog, const Config& cfg, IKVManager* kv,
                          IPrefixCache* pc, RadStream stream = nullptr) {
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, stream, &prog });
        return e.sched_.init(prog, cfg, kv, pc, rio);
    }
    static void fini_sched(Engine& e) { e.sched_.fini(); }

    /* --draft-profile's production reader, the call serve() makes immediately before
     * Scheduler::init. `runtime` is what the caller derived; passing it in is what lets a fixture
     * without a started engine reach the mismatch and admission branches, which a default
     * fingerprint would otherwise hide behind a refused derivation. */
    static int load_profile(Engine& e, const DraftCostRuntime& runtime) {
        return e.load_draft_cost_profile(runtime);
    }

    /* --draft-profile-out's two production calls (core/engine.cpp): the startup check serve() makes
     * before Scheduler::init, and the export it makes at the end. A host fixture reaches every
     * refusal that runs before the identity is derived; the derivation itself needs ranks. */
    static int begin_calibration(Engine& e) { return e.begin_draft_calibration(); }
    static int export_calibration(Engine& e) { return e.export_draft_cost_profile(); }
    /* The armed state begin_draft_calibration would have produced, for the export branches a
     * fixture without ranks cannot reach through the derivation. */
    static void set_calibrating(Engine& e, const DraftCostRuntime& r) {
        e.draft_calib_runtime_ = r;
        e.draft_calibrating_    = true;
    }

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

    /* --- RADIANCE_MTP_COUPLED (tests/engine_couple_short_row_test.cpp) -----------------------
     * A rank with the engine's own Ctx over a fixture program, and the calls that reach the
     * coupled step path. Without bringup there is no rank, and both halves of the coupling need
     * one: the readback reads rank 0's ctx (buf_ptr, stream), and the advance writes the rank's
     * StepSlots. The Ctx creates its own stream, as a rank's does, so it never outlives one the
     * fixture destroys first. `prog` is copied: Ctx::bind_buffers rewrites the copy's buffer
     * pointers, and the scheduler must see the same program the Ctx bound. */
    static int bind_rank(Engine& e, int32_t index, const Program& prog) {
        if ((int32_t)e.ranks_.size() <= index) e.ranks_.resize((size_t)index + 1);
        if (!e.ranks_[(size_t)index]) e.ranks_[(size_t)index] = std::make_unique<Engine::Rank>();
        Engine::Rank& rk = *e.ranks_[(size_t)index];
        rk.index  = index;
        rk.device = 0;
        rk.program = prog;
        CtxDesc d;
        d.program        = &rk.program;
        d.rank           = index;
        d.world_size     = 1;
        d.device         = 0;
        d.record_passes  = false;
        return rk.ctx.init(d);
    }
    static RadStream rank_stream(Engine& e, int32_t index) {
        return e.ranks_[(size_t)index]->ctx.stream();
    }
    /* Where the head's proposals are read from (advance.h, ADV_TAKE). */
    static void tok_buf(Engine& e, int32_t index, rad_buf b) { e.ranks_[(size_t)index]->tok_buf = b; }
    /* The vocabulary the coupled bind validates sampled states against; bringup sets it from the
     * loaded model. */
    static void meta_vocab(Engine& e, int64_t n) { e.meta_.n_vocab = n; }
    /* Scheduler::init over the rank's own program and stream, as serve() does: the builder
     * uploads on the stream the Ctx runs on, so the two cannot diverge. */
    static int init_sched_rank0(Engine& e, const Config& cfg, IKVManager* kv, IPrefixCache* pc) {
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, e.ranks_[0]->ctx.stream(), &e.ranks_[0]->program });
        return e.sched_.init(e.ranks_[0]->program, cfg, kv, pc, rio);
    }
    static int  step_state_init(Engine& e) { return e.step_state_init(); }
    /* The rank's sampler, declared the way Engine::declare_sampler does (minus the logits
     * sizing, which needs a plugin). The coupled finish calls set_mtp_couple on every rank, and
     * an undeclared sampler refuses that call with RAD_E_STATE -- so a coupled fixture must
     * declare one even though nothing samples through it. */
    static int declare_sampler(Engine& e, int32_t index, RadBuilder* b, const SamplerConfig& cfg) {
        return e.ranks_[(size_t)index]->sampler.declare(b, cfg);
    }
    static bool coupled(const Engine& e) { return e.mtp_couple_; }
    static int64_t couple_rows(const Engine& e) { return e.couple_rows_; }
    static RadMtpCoupleState* couple_readback(Engine& e) { return e.couple_readback_; }
    /* The card's state plane as the Ctx bound it (arena base + offset), which is what the 2D
     * readback copies from. */
    static void* couple_plane(Engine& e) {
        return e.ranks_[0]->ctx.buf_ptr(e.couple_state_buf_);
    }
    static RadMtpCoupleDraw couple_draw(const Engine& e, int i) {
        return e.couple_draw_stage_[(size_t)i];
    }
    /* The frozen per-row plan the finish reads: the row's card slot and the rounds it ran. */
    static Scheduler::SerialRow serial_row(const Engine& e, int i) {
        return e.serial_row_[(size_t)i];
    }
    static int k_of(const Engine& e, int i) { return e.serial_k_[(size_t)i]; }
    /* The card's slots as the step read them back (advance_impl.h, ADV_ACCEPT/ADV_TAKE write
     * gen/tok on the host backend, so these are production state). */
    static StepSlot* slots(Engine& e) { return e.slots_host_; }

    /* The release kind a fixture's tiering hook answers for (Engine::issue_what_). */
    static constexpr int kIssueSampler = Engine::kIssueSampler;

    /* One note into a rank's evidence slot: the exact call the rank thread makes after a pass
     * (Rank::note_evidence), injected from the fixture because no rank thread, device or Ctx pass
     * runs here. Ranks up to `rank` are created first: Engine::step_evidence dereferences every
     * rank, as production fills ranks_ in order, so a fixture must not leave a null slot. */
    static void note_evidence(Engine& e, int32_t rank, const Ctx::StepEvidence& ev,
                              uint8_t pass_tape, bool moved) {
        if ((int32_t)e.ranks_.size() <= rank) e.ranks_.resize((size_t)rank + 1);
        for (int32_t i = 0; i <= rank; ++i)
            if (!e.ranks_[(size_t)i]) e.ranks_[(size_t)i] = std::make_unique<Engine::Rank>();
        e.ranks_[(size_t)rank]->note_evidence(ev, pass_tape, moved);
    }
    static bool step_evidence(Engine& e, int32_t step, DraftStepEvidence* out) {
        return e.step_evidence(step, out);
    }

    /* One rank, for the branch of step_loop that needs ranks_[0]: it binds the rank's device and
     * gives the idle branch a rank to walk. Its Ctx has no program, so ctx.stream() is null and
     * the branch's rad_stream_sync(nullptr) fails -- the engine ignores that error. The rank
     * exists so the branch can run, not to prove a rank-stream synchronization; that proof stays
     * open (a fixture would need a real stream on the rank). */
    static void add_rank(Engine& e) { e.ranks_.push_back(std::make_unique<Engine::Rank>()); }
    /* The stop serve() raises after its transport ends; a fixture that runs step_loop itself sets
     * it once the branch under test has run. */
    static void stop(Engine& e) { e.stopping_.store(true, std::memory_order_release); }
    /* serve()'s stepper, run by a fixture that has no transport: the same loop, including the idle
     * branch that calls rad_stream_sync per rank and then release_deferred(). */
    static int  step_loop(Engine& e) { return e.step_loop(); }
};

}  /* namespace rad */
