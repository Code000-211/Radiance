/* engine_couple_short_row_test.cpp -- a cycle whose drafting rows have different K, driven
 * through the Engine's own coupled path.
 *
 * The review's gap: no fixture runs a cycle in which the rows' K differ AND the coupling binds.
 * The scheduler fixture tests/sched_draft_test.cpp composes a split cycle, but it re-derives the
 * engine's K in test code (it cannot see serial_k_) and binds no states. Every coupling test
 * (tests/mtp_couple_plan_test.cpp, tests/sched_test.cpp) binds one row with a uniform round
 * count. Since the rounds shrink to the active prefix (core/sched/scheduler.cpp, serial_batch),
 * a shortened row has no card state past its own K; publishing or binding the cycle's deepest
 * round reads rounds the card never wrote. This case composes the three:
 *
 *   Scheduler::serial_begin/serial_batch  ->  Engine::issue_serial_drafter  ->  StepSlot::gen
 *   (per-row K from the shrinking batches)    (the 2D readback arrangement)     (the card's record)
 *        ->  Engine::finish_serial_drafter  ->  mtp_couple_bind_rounds(serial_k_[i])
 *
 * The two requests start from the same adaptive prior and are committed with different
 * acceptance, so their per-sequence windows separate: the deeper row runs every round of the
 * cycle, the shorter one leaves after its own. Nothing here re-derives K -- the assertions read
 * serial_k_, StepSlot::gen and the staged draws, all production state. What the fixture
 * supplies is the card's other half (the per-round states and the proposals), because no kernel
 * runs on the host backend.
 *
 * Host only, no model, no plugin. The program is the scheduler tests' minimal deployment
 * (sched_program.h) plus the two coupling planes a coupled head declares. The rank is the
 * engine's own Ctx over that program, so the readback's buf_ptr/stream and the advance's
 * StepSlots are the production objects. `RADIANCE_MTP_COUPLED=1` is what step_state_init reads to
 * arm the coupling; the sampler is declared so set_mtp_couple accepts the binding call -- the
 * requests never enter its map, so the call answers NOTFOUND, which the finish tolerates.
 *
 * THE MUTATION THIS CASE IS FOR: binding the cycle's depth (serial_done_) instead of
 * serial_k_[i] makes the shorter row read states past its K. The fixture leaves those holding
 * another draw, as a plane the card wrote only part of does, and the bind refuses the step with
 * RAD_E_STATE. Run it by hand (engine_draft.cpp:485); the case itself only asserts the correct
 * behaviour.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include "build/rad_build.h"
#include "plugin/registry.h"
#include "sample/sampler.h"

#include <cstdlib>
#include <cstring>
#include <memory>

/* Only FakeKV and FakePC: this binary runs on the real device layer, as the engine does. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "sched_program.h"

namespace {

using rad::Engine;
using rad::EngineTestPeer;

constexpr int kDepth = 6;

struct HostProgram {
    rad::Program       prog;
    rad::Config        cfg;
    sched_fake::FakeKV kv;
    sched_fake::FakePC pc;
    std::vector<void*> owned;

    ~HostProgram() { for (void* p : owned) std::free(p); }
};

void add_plane(rad::Program& prog, std::vector<void*>& owned, const char* name, int64_t rows,
               int64_t words, rad_buf* out) {
    rad::BufferInfo b;
    b.name          = name;
    b.decl.dtype    = RAD_I32;
    b.decl.rank     = 3;
    b.decl.shape[0] = kDepth;
    b.decl.shape[1] = rows;
    b.decl.shape[2] = words;
    b.decl.kind     = RAD_BUF_PERSIST;
    b.bytes         = kDepth * rows * words * 4;
    void* p = std::calloc(1, (size_t)b.bytes);
    owned.push_back(p);
    b.ptr = p;
    *out  = (rad_buf)prog.buffers.size();
    prog.buffers.push_back(b);
}

void add_i32(rad::Program& prog, std::vector<void*>& owned, const char* name, int64_t d0,
             int64_t d1, rad_buf* out) {
    rad::BufferInfo b;
    b.name          = name;
    b.decl.dtype    = RAD_I32;
    b.decl.rank     = 2;
    b.decl.shape[0] = d0;
    b.decl.shape[1] = d1;
    b.decl.kind     = RAD_BUF_PERSIST;
    b.bytes         = d0 * d1 * 4;
    void* p = std::calloc(1, (size_t)b.bytes);
    owned.push_back(p);
    b.ptr = p;
    *out  = (rad_buf)prog.buffers.size();
    prog.buffers.push_back(b);
}

/* The fixture's one op is never issued here (no rank pass runs), but Ctx::bind_ops places an op
 * with no device launch on the host domain and then refuses it for having no host kernel. A
 * launch that never runs keeps it where sched_program.h declares it. */
int never_launched(const RadArgs*, RadStream) { return RAD_OK; }

/* Arena offsets are the buffer plan's, which this fixture does not run; Ctx::bind_arena refuses a
 * buffer without one. Lay them out in declaration order. */
void assign_arena(rad::Program& prog) {
    int64_t off = 0;
    for (size_t i = 1; i < prog.buffers.size(); ++i) {
        prog.buffers[i].arena_offset = off;
        off += (prog.buffers[i].bytes + 63) & ~(int64_t)63;
    }
    prog.arena_bytes = off;
}

/* The requests' own acceptance, committed after each cycle: the deeper row accepts every draft
 * it was given, the shorter one none. That difference is what separates the two adaptive
 * windows. `published` is what the engine actually published for the row (serial_k_), so the
 * commit never claims more than the row verified. */
void commit_cycle(Engine& e, uint64_t deep_seq) {
    const rad::StepPlan& p = EngineTestPeer::sched(e).plan();
    std::vector<int32_t> acc(p.e.size(), 0), tok(p.e.size(), 11);
    std::vector<uint8_t> eos(p.e.size(), 0);
    for (size_t i = 0; i < p.e.size(); ++i) {
        if (p.e[i].seq != deep_seq) continue;
        int k = 0;
        for (int r = 0; r < EngineTestPeer::rows_of(e); ++r)
            if (EngineTestPeer::serial_row(e, r).seq == deep_seq) k = EngineTestPeer::k_of(e, r);
        acc[i] = k < p.e[i].n_spec ? k : p.e[i].n_spec;
    }
    rad::Scheduler::StepResult r{};
    r.n_seq      = (int)p.e.size();
    r.n_accepted = acc.data();
    r.token      = tok.data();
    r.eos        = eos.data();
    REQUIRE_EQ(EngineTestPeer::sched(e).commit(r), RAD_OK);
}

}  /* namespace */

TEST(a_shortened_coupled_row_binds_only_the_rounds_the_card_wrote) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    /* The batch builder allocates through the device layer. With HIP compiled in,
     * select_backend() may bind a card before this test can say otherwise; the guarantee this
     * case needs is the compile-time one (configure -DRAD_WITH_HIP=OFF). */
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    HostProgram h;
    sched_fake::build_program(h.prog, h.cfg, h.kv, h.pc, h.owned,
                              /*blocks*/ 4096, /*states*/ 64, /*max_tok*/ 512, /*max_ctx*/ 4096);
    sched_fake::kGdnKi.launch = &never_launched;
    h.cfg.n_spec       = kDepth;
    h.cfg.draft_window = rad::DraftWindow::Adaptive;
    h.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    h.prog.drafter.depth = kDepth;
    h.prog.drafter.name  = "mtp-test";

    /* The two planes the coupling reads, and the proposal plane the head's greedy passes write
     * (advance.h, ADV_TAKE reads rk.tok_buf when the head declares no proposal buffer). */
    rad_buf draws = 0, states = 0, tokens = 0;
    add_plane(h.prog, h.owned, "spechub.mtp_couple_draws", h.cfg.max_seqs,
              (int64_t)RAD_MTP_COUPLE_DRAW_WORDS, &draws);
    add_plane(h.prog, h.owned, "spechub.mtp_couple_state", h.cfg.max_seqs,
              (int64_t)RAD_MTP_COUPLE_STATE_WORDS, &states);
    add_i32(h.prog, h.owned, "sampler.tokens", h.cfg.max_seqs, kDepth + 1, &tokens);
    assign_arena(h.prog);

    /* step_state_init reads the flag and refuses a run whose declaration disagrees with it. */
    REQUIRE_EQ(setenv("RADIANCE_MTP_COUPLED", "1", 1), 0);
    struct EnvGuard {
        ~EnvGuard() { unsetenv("RADIANCE_MTP_COUPLED"); }
    } env_guard;

    Engine e(h.cfg);
    /* The rank first: it owns the Ctx (and the stream) the scheduler uploads on, and the program
     * the Ctx binds is the one the scheduler must read. */
    REQUIRE_EQ(EngineTestPeer::bind_rank(e, 0, h.prog), RAD_OK);
    struct SchedGuard {
        Engine& e;
        bool    inited = false;
        ~SchedGuard() {
            if (inited) EngineTestPeer::fini_sched(e);
        }
    } sched_guard{e};
    REQUIRE_EQ(EngineTestPeer::init_sched_rank0(e, h.cfg, &h.kv, &h.pc), RAD_OK);
    sched_guard.inited = true;
    EngineTestPeer::drafter(e, h.prog.drafter);
    EngineTestPeer::barrier(e, 1);
    EngineTestPeer::tok_buf(e, 0, tokens);
    EngineTestPeer::meta_vocab(e, h.prog.meta.n_vocab);
    REQUIRE_EQ(EngineTestPeer::step_state_init(e), RAD_OK);
    REQUIRE(EngineTestPeer::coupled(e));

    /* The rank's sampler: declared (couple_ is read from the flag here) so the finish's
     * set_mtp_couple accepts the call and answers NOTFOUND for a request it never admitted. The
     * flag's own kernel library is the only one this case loads: the chain refuses to declare
     * coupled without `mtp_couple_verify`, and every other op it names may stay unresolved
     * because no sampling step is built or run here. */
    {
        rad::Registry     reg;
        REQUIRE_EQ(reg.load_plugin(home() + "/kernels/spechub.so", 0), RAD_OK);
        RadModelMeta meta{};
        meta.n_vocab = h.prog.meta.n_vocab;
        RadBuildCtx bc{};
        bc.rank = 0;
        bc.world_size = 1;
        bc.max_seqs = h.cfg.max_seqs;
        bc.max_tok = h.cfg.max_tok;
        bc.max_ctx = h.cfg.max_ctx;
        bc.max_spec = kDepth;
        rad::Builder sb(reg, meta, bc);
        rad::SamplerConfig scfg{};
        scfg.n_vocab        = h.prog.meta.n_vocab;
        scfg.n_vocab_local  = h.prog.meta.n_vocab;
        scfg.max_seqs       = h.cfg.max_seqs;
        scfg.max_spec       = kDepth;
        scfg.max_hist       = 64;
        scfg.max_candidates = 8;
        REQUIRE_EQ(EngineTestPeer::declare_sampler(e, 0, &sb, scfg), RAD_OK);
    }

    for (int id = 1; id <= 2; ++id) {
        auto req = std::make_unique<rad::Request>();
        req->id         = (uint64_t)id;
        req->prompt.resize(16);
        for (size_t i = 0; i < req->prompt.size(); ++i)
            req->prompt[i] = (int32_t)(1 + ((size_t)id * 7 + i * 37) % 318);
        req->max_tokens = 4096;
        req->ignore_eos = true;
        req->sp.temp    = 0.0f;
        req->sink       = nullptr;
        REQUIRE_EQ(EngineTestPeer::submit(e, std::move(req)), RAD_OK);
    }

    const int64_t stride = EngineTestPeer::couple_rows(e);
    REQUIRE_EQ(stride, h.cfg.max_seqs);
    RadMtpCoupleState* plane = static_cast<RadMtpCoupleState*>(
        EngineTestPeer::couple_plane(e));
    RadMtpCoupleState* rb = EngineTestPeer::couple_readback(e);
    REQUIRE(plane != nullptr);
    REQUIRE(rb != nullptr);
    const RadStream stream = EngineTestPeer::rank_stream(e, 0);

    bool found = false;
    int  k_deep = 0, k_short = 0;
    for (int cycle = 0; cycle < 200 && !found; ++cycle) {
        const RadBatch* b = EngineTestPeer::sched(e).step();
        REQUIRE(b != nullptr);
        const rad::StepPlan& plan = EngineTestPeer::sched(e).plan();

        /* THE ARRANGEMENT. The card's plane carries one round per kDepth, one row per card slot;
         * every cell is stamped with its own (round, row) here. The readback must reproduce
         * exactly [0, serial_done_) rounds x serial_rows_ rows of it and leave the rest of the
         * pinned buffer at the sentinel -- the copy's pitch, its height and its width, checked
         * against the counts the issue itself derived. */
        for (int rd = 0; rd < kDepth; ++rd)
            for (int64_t i = 0; i < stride; ++i) {
                RadMtpCoupleState s{};
                s.round = rd + 1;
                s.x     = (int32_t)i;
                plane[(size_t)rd * (size_t)stride + (size_t)i] = s;
            }
        for (int64_t c = 0; c < (int64_t)kDepth * stride; ++c) {
            rb[c] = RadMtpCoupleState{};
            rb[c].round = -1;
            rb[c].x     = -1;
        }

        EngineTestPeer::StepOut so;
        so.n_rows = 0;                  /* no logit row: the sampler half stays out of this case */
        so.complete = true;             /* what the step's wait reports after the issue */
        REQUIRE_EQ(EngineTestPeer::issue(e, b->step, so), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);

        const int rows = EngineTestPeer::rows_of(e);
        const int done = EngineTestPeer::done_rows(e);
        for (int rd = 0; rd < kDepth; ++rd)
            for (int64_t i = 0; i < stride; ++i) {
                const RadMtpCoupleState& s = rb[(size_t)rd * (size_t)stride + (size_t)i];
                if (rd < done && i < rows) {
                    CHECK_EQ(s.round, rd + 1);
                    CHECK_EQ(s.x, (int32_t)i);
                } else {
                    CHECK_EQ(s.round, -1);   /* untouched: the copy's height is serial_done_ */
                    CHECK_EQ(s.x, -1);
                }
            }

        /* The trunk's agreement row, from the card's own slots: what a step with sampled rows
         * would report, so the finish's card-vs-host check runs on real values rather than being
         * skipped by first_row < 0. */
        so.first_row.assign(plan.e.size(), 0);
        so.accepted.assign(plan.e.size(), 0);
        so.token.assign(plan.e.size(), 0);
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const rad::StepSlot& S = EngineTestPeer::slots(e)[plan.e[i].card];
            so.accepted[i] = S.acc;
            so.token[i]    = S.tok[0];
        }

        /* The card's half of the coupling, for the rounds that bind. Every round the row ran
         * carries this step's draw; a round it did not run keeps another draw, which is what a
         * plane the card wrote only part of holds. Both the round count and the per-round token
         * are production values (serial_k_'s batches wrote gen/tok through the advance). */
        for (int i = 0; i < rows; ++i) {
            const rad::Scheduler::SerialRow R = EngineTestPeer::serial_row(e, i);
            const rad::StepSlot& S = EngineTestPeer::slots(e)[R.card];
            const RadMtpCoupleDraw& d = EngineTestPeer::couple_draw(e, i);
            for (int rd = 1; rd <= done; ++rd) {
                RadMtpCoupleState s{};
                s.draw   = d;
                s.round  = rd;
                s.x      = S.tok[rd];
                s.status = 0;                 /* greedy draw: mode 0 has no sampler status */
                if (rd > (int)S.gen) s.draw.seed ^= 0x5a5a5a5a5a5a5a5aull;   /* never written */
                rb[(size_t)(rd - 1) * (size_t)stride + (size_t)i] = s;
            }
        }

        int deep = 0, shallow = 0;
        for (int i = 0; i < rows; ++i)
            if (EngineTestPeer::serial_row(e, i).seq == 1) k_deep = EngineTestPeer::k_of(e, i);
            else                                              k_short = EngineTestPeer::k_of(e, i);
        deep    = k_deep;
        shallow = k_short;
        const bool split = rows == 2 && deep != shallow && shallow > 0;

        /* The commit the step reports, then the finish -- the order serve() runs them in
         * (core/engine.cpp: the scheduler commits, then the drafter's finish publishes and
         * binds). A finish before the commit is a different step's. */
        commit_cycle(e, /*deep_seq*/ 1);
        CHECK_EQ(EngineTestPeer::finish(e, so), RAD_OK);
        if (split) {
            found = true;
            /* The card's own record agrees with the frozen plan for every row: StepSlot::gen is
             * the round the advance last took a proposal in, serial_k_ the rounds the batches
             * gave the row. Neither is recomputed here. */
            for (int i = 0; i < rows; ++i) {
                const rad::Scheduler::SerialRow R = EngineTestPeer::serial_row(e, i);
                CHECK_EQ((int)EngineTestPeer::slots(e)[R.card].gen, EngineTestPeer::k_of(e, i));
            }
        }

        rad::DraftStepEvidence live;
        live.replay_known = true;
        live.replay       = rad::DraftReplay::None;
        EngineTestPeer::sched(e).step_finished(live);
        EngineTestPeer::sched(e).promote_next();
    }

    REQUIRE(found);
    CHECK_EQ(k_deep, kDepth);      /* the deepest row runs the cycle's every round */
    CHECK(k_short > 0 && k_short < kDepth);
}

RAD_TEST_MAIN()
