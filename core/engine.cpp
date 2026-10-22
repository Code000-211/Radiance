/* engine.cpp -- the step loop and the server: Engine construction, the per-rank threads, the
 * step loop and the server it runs under. Bring-up, the drafters and the buffer probe live in
 * their own files (engine_priv.h).
 */
#include "engine_priv.h"
#include "gen/driver.h"
#include "gen/audio_driver.h"
#include "gen/video_driver.h"
#include "kld.h"
#include "mm/processor.h"
#include "mm/media.h"

#include "build/startup.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "format/radfile.h"
#include "gen_config.h"
#include "sample/accept.h"
#include "sched/scheduler.h"
#include "server/adapters.h"
#include "server/server.h"
#include "server/sink.h"
#include "server/liveview.h"
#include "text/tokenizer.h"
#include "util/stallwatch.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <functional>
#include <utility>
#include <vector>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

#ifdef RAD_TEST_HOOKS
/* Test-build-only seam: tests/CMakeLists.txt compiles the engine units a second time, for one test
 * binary alone, with RAD_TEST_HOOKS. The loop calls this once per step, at the point where the
 * step is issued and its completion flag reset but the wait has not run -- the one place where a
 * driver outside the loop can see what the previous step left, what the reset left, and the history
 * watermark before the finish moves it, and where it can withhold the step's completion proof.
 * tests/engine_loop_test.cpp defines it strongly. rad_core is never compiled with RAD_TEST_HOOKS,
 * so the production library carries neither this definition nor the call in any configuration. */
extern "C" __attribute__((weak)) void rad_test_step_point(int step, int rows, bool prior_complete,
                                                          bool complete) {
    (void)step; (void)rows; (void)prior_complete; (void)complete;
}

/* The same test copy's second seam: called once in serve(), after the scheduler is initialised and
 * before any rank or the stepper runs, so a driver can put one request in while there is no
 * concurrent access. tests/engine_loop_test.cpp defines it strongly; the weak definition keeps
 * every other test build that compiles these units linkable. */
extern "C" __attribute__((weak)) void rad_test_serving() {}

/* The same test copy's third seam: the evidence handed to Scheduler::step_finished, after the
 * step's join and before the scheduler stamps it. A driver checks that what a step's cycle is keyed
 * by is that step's own -- the ahead pass has run by now -- rather than reading the same fields off
 * the engine, which would restate the call. tests/engine_loop_test.cpp defines it strongly. */
extern "C" __attribute__((weak)) void rad_test_step_evidence(int step, int replay_known, int replay,
                                                            int staged, int moved) {
    (void)step; (void)replay_known; (void)replay; (void)staged; (void)moved;
}

/* A placement move a fixture triggers in the release it names: the rank loop adds the value to the
 * heat counters tiering_dispatch reads, so a host run without a tiered plan can still be the
 * mover. `what` is the release (kIssueAll/kIssueModel/kIssueSampler). tests/engine_loop_move_test.cpp
 * defines it strongly. */
extern "C" __attribute__((weak)) int64_t rad_test_tiering_moves(int rank, int32_t step, int what) {
    (void)rank; (void)step; (void)what;
    return 0;
}

/* The third seam: the server's running state, reported from run_server()'s watcher once
 * Server::port() turns nonzero. That is the first moment a stop is one the transport accepts --
 * before it, httplib's stop() closes nothing and the signal is dropped (core/server/http.cpp). A
 * driver binds its one stop to this rather than to the signal handler's disposition, which is
 * installed before run() and so says nothing about whether a stop would survive. */
extern "C" __attribute__((weak)) void rad_test_serving_ready(int port) { (void)port; }

/* A placement move a fixture publishes while the engine idles, where no rank thread runs and no
 * host plan can raise the heat counters: the engine's cumulative count adds it, so the move
 * between two brackets can be replayed. tests/engine_cost_drift_test.cpp defines it strongly. */
extern "C" __attribute__((weak)) int64_t rad_test_residency_moves() { return 0; }
#endif

namespace rad {

/* The operator's RADIANCE_* names, captured at construction for the diagnostics report. */
Engine::Engine(Config cfg)
    : cfg_(std::move(cfg)), env_set_(diag::env_names(environ, "RADIANCE_")) {}

int32_t* Engine::Readback::reserve(int64_t count) {
    if (count <= n) return p;
    if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    p = static_cast<int32_t*>(rad_dev_alloc(count * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED));
    n = p ? count : 0;
    return p;
}

Engine::Readback::~Readback() {
    if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
}

Engine::~Engine() {
    /* The control vector's per-rank direction copies. Uploaded at declare, outside any arena, so
     * nothing else frees them; each rank's copy is freed with that rank's device bound, since the
     * allocator is per device. */
    for (const auto& r : ranks_) {
        if (!r || (size_t) r->index >= cvec_views_.size()) continue;
        const float* d = cvec_views_[(size_t) r->index].dir;
        if (!d) continue;
        if (rad_dev_set(r->device) < 0) continue;
        rad_dev_free((void*) d, RAD_MEM_DEVICE);
        cvec_views_[(size_t) r->index].dir = nullptr;
    }
    cvec_views_.clear();
    /* --profile-ops prints its final table here, the last point where the program still holds
     * the op table the rows name. */
    if (cfg_.profile_ops) for (auto& r : ranks_) r->ctx.dump_profile();
    for (float* p : kld_host_) if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    if (pooled_host_) rad_dev_free(pooled_host_, RAD_MEM_HOST_PINNED);
    for (auto& r : ranks_) if (r->score_host) rad_dev_free(r->score_host, RAD_MEM_HOST_PINNED);
    /* The tiering summary, here for the same reason, and only for a dynamic plan. Its per-layer
     * residency histogram shows whether the global ranking starved a layer, which a hit rate
     * cannot. */
    for (auto& r : ranks_) {
        if (!r->heat.enabled()) continue;
        LogRank lr(r->index);
        RAD_INFO("%s", r->heat.report().c_str());
        RAD_INFO("  routing evidence: %llu layer-histograms credited, %llu layer-sweeps found "
                 "nothing new, %llu dropped by a ring shallower than the run-ahead",
                 (unsigned long long)r->heat_credited, (unsigned long long)r->heat_idle,
                 (unsigned long long)r->ctx.route_dropped());
        RAD_INFO("%s", r->mover.report().c_str());
    }
    /* Drain the transfer streams before the contexts. Ctx::shutdown drains the compute streams
     * and frees the arena, but the mover's stream also copies into the arena (the prefill
     * stager's ring, lent slab slots) and would otherwise drain only later, with the mover. A fill
     * landing in a freed arena is a GPU memory fault. */
    for (auto& r : ranks_)
        if (RadStream s = r->mover.transfer_stream()) (void)rad_stream_sync(s);
    for (auto& r : ranks_) r->ctx.shutdown();
    step_state_fini();
    ranks_.clear();
    for (TableCopy& c : table_copies_) table_copy_free(&c);
    table_copies_.clear();
    for (void* p : card_copies_) rad_dev_free(p, RAD_MEM_HOST_MAPPED);
    card_copies_.clear();
    rad_tools_registry().close();
}

/* ================================================================== 7. serve */
/* The step loop, where every component meets. Threading per spec §1: HTTP threads feed a request
 * queue, one scheduler thread produces one step, and one thread per tensor-parallel rank owns one
 * device.
 *
 * The batch is shared by pointer and the handoff is a barrier (rank_barrier.h), entered twice per
 * step: to release the ranks against the batch, and to join them before the sampler's answer is
 * read. The scheduler thread takes part, so the width is tp + 1.
 */

int Engine::declare_sampler(int rank, RadBuilder* b, int64_t logits_width, bool sizing) {
    Rank& r = *ranks_[(size_t)rank];
    /* A sizing declare uses a throwaway sampler: the chain must be in the Program so the buffer
     * list matches, but the rank's sampler keeps the real declare's handles. */
    Sampler scratch;
    Sampler& smp = sizing ? scratch : r.sampler;
    rad_buf scratch_tok = 0;
    rad_buf& tok = sizing ? scratch_tok : r.tok_buf;

    SamplerConfig sc{};
    sc.n_vocab       = meta_.n_vocab;
    /* This rank's vocabulary width, from the declared logits buffer (see rad_tools_declare).
     * meta_.n_vocab is the whole model's on every rank. */
    sc.n_vocab_local = logits_width > 0 ? logits_width : meta_.n_vocab;
    /* This rank's first vocabulary row, which the width cannot tell. The rule restates
     * arch/common/rad_arch.h's shard_begin rather than passing through the ABI, and is checked
     * against the declared width, so a plugin that shards differently is refused at startup
     * instead of sampling with the wrong ids. */
    sc.vocab_off = 0;
    if (sc.n_vocab_local < sc.n_vocab) {
        /* rad_arch.h vocab_per_rank: the even share when tp divides the vocabulary, otherwise
         * ceil(n / tp) rounded up to a 128-row block so a block-scaled head can be cut there; the
         * last rank is short. */
        const int64_t ceil_per = (sc.n_vocab + cfg_.tp - 1) / cfg_.tp;
        const int64_t per = (sc.n_vocab % cfg_.tp == 0) ? sc.n_vocab / cfg_.tp
                                                        : (ceil_per + 127) / 128 * 128;
        sc.vocab_off = per * rank;
        const int64_t want = std::min<int64_t>(per, sc.n_vocab - sc.vocab_off);
        if (sc.vocab_off >= sc.n_vocab || want != sc.n_vocab_local) {
            RAD_ERR("rank %d declared %lld logits columns of a %lld-token vocabulary at --tp %d; "
                    "the contiguous equal split this sampler assumes wants %lld. The plugin and "
                    "core/engine.cpp disagree about which rows this rank owns.",
                    rank, (long long)sc.n_vocab_local, (long long)sc.n_vocab, cfg_.tp,
                    (long long)want);
            return RAD_E_INVAL;
        }
    }
    sc.max_seqs      = cfg_.max_seqs;
    sc.max_spec      = cfg_.n_spec;
    sc.rank          = rank;
    sc.world_size    = cfg_.tp;
    RAD_TRY(smp.declare(b, sc));

    /* Where the chain writes the sampled ids. The engine declares it because rad-kbench drives the
     * same chain against its own buffer. PERSIST, since the step loop copies it back after the
     * step, when a transient's storage may belong to another buffer. */
    RadBufDecl d{};
    d.dtype    = RAD_I32;
    d.rank     = 1;
    d.shape[0] = cfg_.max_seqs * (cfg_.n_spec > 0 ? cfg_.n_spec + 1 : 1);
    d.kind     = RAD_BUF_PERSIST;
    tok = rad_decl_buffer(b, "sampler.tokens", &d);
    return tok ? RAD_OK : RAD_E_STATE;
}

/* ------------------------------------------------------------------ the tiering seam
 *
 * The heat engine dispatches per step, not per layer: the mover relocates only between steps
 * (Ctx::run_step syncs the residency table once at its top), so a mid-step table edit would be
 * seen by some issues and not others.
 *
 * It runs once per step, after the step's forward pass is queued:
 *
 *   - After, so it runs under the pass instead of in front of it (ahead of a draft pass the
 *     device is idle). Its changes cannot reach the pass just issued: table edits apply at the
 *     next run_step, and a freed slot is quarantined behind an event recorded after this pass.
 *   - On the trunk pass only, since a swap rarely clears the hysteresis between passes of one
 *     step and each call ranks the whole movable set. Draft-pass histograms accumulate in the
 *     step's bank and are credited next step, weighted by pass count.
 *   - `begin_dispatch` only retires; the issue call below is the other half. Retiring without
 *     issuing stalls the mover (mover.h).
 *   - It credits the previous step's histograms, whose copies land asynchronously while that step
 *     runs (spec §5.5); `histogram_ready` says which have landed.
 *
 * On long prefill chunks the host runs many steps ahead of the card, so most histograms have not
 * landed and are skipped (about 2% credited with 2048-token chunks, nearly all at decode shapes).
 * The counters report this, and the link panel's expert-streaming share reads near zero for the
 * same reason.
 *
 * `heat_seen` prevents crediting a bank twice: a bank keeps its counts until the next report, so a
 * step where a layer did not route (a drafter pass) would re-credit the stale plane.
 */
void Engine::tiering_dispatch(Rank& r, int step) {
    if (!r.heat.enabled()) return;

    r.mover.begin_dispatch(step, &r.heat);

    const int n_layer = r.ctx.n_route_layers();
    if ((int)r.heat_seen.size() != n_layer) r.heat_seen.assign((size_t)n_layer, -1);
    /* Find all histograms first, then credit them in order. Each is a card-written row in host
     * memory (every line a cache miss), and one credit is too little work to hide the next fetch.
     * Prefetching two rows ahead keeps 32 lines in flight, about what one core sustains. */
    int8_t landed[kRouteBanks];
    std::memset(landed, -1, sizeof landed);
    std::vector<Rank::HeatRow>& rows = r.heat_rows;
    rows.clear();
    for (int l = 0; l < n_layer; ++l) {
        /* Drain every landed histogram for the layer. The host runs this once per issued step,
         * possibly far ahead of the card, so several may have landed since the last visit, and the
         * newest is often still in flight. */
        int seen = r.heat_seen[(size_t)l];
        bool any = false;
        for (;;) {
            const int bank = r.ctx.next_histogram(l, seen, step, landed);
            if (bank < 0) break;
            const int32_t* h = r.ctx.last_histogram(l, bank);
            if (!h) break;
            rows.push_back({l, bank, h});
            seen = r.ctx.last_histogram_step(l, bank);
            any = true;
        }
        if (!any) ++r.heat_idle;
    }
    const int32_t n_ex = (int32_t)r.ctx.route_n_expert();
    constexpr size_t kAhead = 2;
    for (size_t i = 0; i < rows.size() && i < kAhead; ++i) r.heat.prefetch(rows[i].h, n_ex);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i + kAhead < rows.size()) r.heat.prefetch(rows[i + kAhead].h, n_ex);
        const Rank::HeatRow& hr = rows[i];
        /* A bank can hold several passes (a draft head reports once per pass into one bank), so
         * weight it by the report count, or the layer's experts are underpriced and drained. */
        const int reps = r.ctx.last_histogram_reports(hr.layer, hr.bank);
        r.heat.observe(hr.layer, hr.h, n_ex, reps > 1 ? (float)reps : 1.0f);
        r.heat_seen[(size_t)hr.layer] = r.ctx.last_histogram_step(hr.layer, hr.bank);
        r.ctx.mark_credited(hr.layer, hr.bank);
        ++r.heat_credited;
    }

    /* No swaps behind a pass the stager armed: that step is bound by its own copies on the same
     * transfer stream, and the turnover slots are filled for it (below). Histograms are still
     * credited; the next unstaged step swaps on them. */
    std::vector<Swap> swaps;
    if (!r.stager.staged() && r.heat.step(&swaps) < 0) return;
    for (const Swap& s : swaps) {
        /* Contract: every proposed unit is busy until `issue` takes it or `abandon` returns it;
         * a dropped proposal stays busy forever. */
        if (r.mover.issue(s, &r.heat) < 0) r.heat.abandon(s);
    }

    /* Fill after the swaps: swaps are the policy's decisions, fills only use spare memory, and a
     * fill must not take a move record the next swap needs.
     *
     * Behind a staged pass the turnover slots are filled too, since that step swaps nothing; left
     * free, the stager would copy those residents instead (at TP2 a 160K-token prompt staged 9.6%
     * more and ran 3-5% slower).
     *
     * They stay filled until kTurnoverAfter consecutive unstaged steps (idle ticks included).
     * Otherwise a short answer or a pause drops them and the next prompt's first chunk copies them
     * back, scattering the lent residents through the pool runs the stager coalesces on (at TP2,
     * back-to-back 26K-token prompts: 33 copies per staged layer instead of 16). Decode reaches the
     * count in a few steps. */
    constexpr int kTurnoverAfter = 4;
    r.unstaged_steps = r.stager.staged() ? 0 : r.unstaged_steps + 1;
    const bool hold = r.unstaged_steps < kTurnoverAfter;
    r.mover.hold_turnover(hold);
    (void)r.mover.balance(&r.heat, !hold);
}
/* ------------------------------------------------------------------ the rank thread */
#ifdef RAD_TEST_HOOKS
void Engine::test_set_stager(OpStager* s) {
    /* Every rank, as bringup does (engine_bringup.cpp): a pass and its evidence are per rank, and
     * a stager attached to one only would leave the others answering for a pass it never armed. */
    for (auto& r : ranks_) r->ctx.set_stager(s);
}
#endif

void Engine::rank_loop(int index) {
    Rank& r = *ranks_[(size_t)index];
    if (r.ctx.bind_thread() < 0) {
        RAD_ERR("rank %d: could not bind its device", index);
        step_status_.store(RAD_E_DEVICE, std::memory_order_release);
    }

    for (;;) {
        barrier_->arrive_and_wait();                    /* released against this step's batch */
        if (stopping_.load(std::memory_order_acquire)) break;

        const RadBatch* b = step_batch_.load(std::memory_order_acquire);
        if (b) {
            /* This rank's view of the step, pointing at this card's staging. Its upload was issued
             * on this rank's stream, so no cross-rank fence is needed. */
            const RadBatch* mine = gen_ ? gen_->batch_for(index) : sched_.batch_for(index);
            if (!mine) {
                RAD_ERR("rank %d: the builder has no batch for this rank", index);
                step_status_.store(RAD_E_STATE, std::memory_order_release);
                barrier_->arrive_and_wait();
                continue;
            }
            b = mine;
            const int what = issue_what_.load(std::memory_order_acquire);
            const int64_t moves_before = r.ev_moves_before;
            int s = what == kIssueSampler ? RAD_OK : r.ctx.run_step(b);
            if (s < 0) {
                RAD_ERR("rank %d step %d: %s", index, b->step, r.ctx.step_error());
                step_status_.store(s, std::memory_order_release);
            } else if (b->n_out > 0 && what != kIssueModel) {
                /* Wait for the scheduler's sampler staging, which runs while the device executes
                 * the forward pass enqueued above, so the wait is hidden behind device work. */
                int bst = RAD_OK;
                {
                    std::unique_lock<std::mutex> lk(build_mu_);
                    build_cv_.wait(lk, [this] { return build_ready_; });
                    bst = build_status_;
                }
                /* The sampler chain goes on the same stream behind the model with no host sync
                 * (spec §13). */
                if (bst >= 0) {
                    /* The scored rows first: the sampler may rewrite a row in place. */
                    if (b->draft_pass == 0 && !score_rows_.empty()) s = issue_scored_rows(r);
                    if (s >= 0) s = r.sampler.run(&r.ctx, r.program.logits_buf, r.tok_buf);
                    if (s < 0) {
                        RAD_ERR("rank %d step %d: sampler: %s", index, b->step, rad_strerror(s));
                        step_status_.store(s, std::memory_order_release);
                    }
                }
            }
            /* Behind the pass, once a step: see tiering_dispatch. */
            if (s >= 0 && b->draft_pass == 0 && what != kIssueModel) tiering_dispatch(r, b->step);

            /* Freeze what this step's passes did, keyed by the step that ran them. Read here, on
             * the thread and right after the passes: the next step's ahead pass is released before
             * this one is collected, so a delta read at collect time would report it. The stager
             * is per pass, which is why the trunk pass's Ctx record is what carries it; the tape
             * is merged over every pass of the step (Ctx::last_pass_tape), and `moved` spans from
             * the counters prepare_release snapshotted to here, the step's own window including
             * its tiering dispatch.
             *
             * A sampler-only release runs the tiering dispatch of a step issued ahead and notes
             * too, while the record still names that step: without it, a step with no draft pass
             * (a shared-KV head whose plan drew K=0) would lose its move -- the next release's
             * base is already the ahead step's. The merge is idempotent, so the repeat of the
             * trunk pass's own facts costs nothing. */
            int64_t injected = 0;
#ifdef RAD_TEST_HOOKS
            if (s >= 0) injected = rad_test_tiering_moves(index, b->step, what);
#endif
            const Ctx::StepEvidence rec = r.ctx.last_step_evidence();
            if (s >= 0 && (what != kIssueSampler || rec.step == b->step)) {
                const HeatStats hs = r.heat.stats();
                int64_t moves = (int64_t)hs.promotions + (int64_t)hs.demotions + injected;
                r.note_evidence(rec, r.ctx.last_pass_tape(), moves > moves_before);
            }

        }
        barrier_->arrive_and_wait();                    /* joined; the scheduler may read */
    }
}

/* ------------------------------------------------------------------ the scheduler thread */
bool Engine::step_evidence(int32_t step, DraftStepEvidence* out) const {
    if (!out) return false;
    bool known = false;
    for (const auto& r : ranks_) {
        const Rank::StepSnap& s = r->ev[(uint32_t)step & 3u];
        if (s.step != step) continue;   /* this rank never ran the step: not evidence of None */
        known = true;
        out->staged = out->staged || s.staged;
        out->moved  = out->moved  || s.moved;
        if (s.tape == Ctx::kStepTapeRecord) out->replay = DraftReplay::Record;
        else if (s.tape == Ctx::kStepTapeAudit && out->replay != DraftReplay::Record)
            out->replay = DraftReplay::Audit;
        else if (s.tape == Ctx::kStepTapePlay &&
                 (out->replay == DraftReplay::Unknown || out->replay == DraftReplay::None))
            out->replay = DraftReplay::Play;
        else if (s.tape == Ctx::kStepTapeNone && out->replay == DraftReplay::Unknown)
            out->replay = DraftReplay::None;
    }
    out->replay_known = known;
    return known;
}

int64_t Engine::residency_moves() const {
    int64_t moves = 0;
    for (const auto& r : ranks_) {
        const HeatStats hs = r->heat.stats();
        moves += (int64_t)hs.promotions + (int64_t)hs.demotions;
    }
#ifdef RAD_TEST_HOOKS
    moves += rad_test_residency_moves();
#endif
    return moves;
}

int Engine::prepare_release(const RadBatch* b) {
    /* The placement counters when this step's window opens, before the arena level's recall can
     * move a unit: the rank thread reports `moved` for the step from here (Rank::ev_moves_before),
     * so a later step's reclaim cannot stand in for this one. */
    for (auto& rk : ranks_) {
        const HeatStats hs = rk->heat.stats();
        rk->ev_moves_before = (int64_t)hs.promotions + (int64_t)hs.demotions;
    }
    /* The arena level, recalling lent arena memory if this step needs it. Every rank thread is
     * parked, so the level can change and a recall can drain both streams. */
    if (const int as = arena_step(b); as < 0) return as;
    /* Grow the followers' pools to match rank 0's before the step's block table is used:
     * admission grew rank 0's cache and the new ids are already in the table. Done here, not on
     * the 100 ms arbiter tick, because the step is about to run; safe because every rank thread is
     * parked. */
    for (size_t i = 1; i < ranks_.size(); ++i) {
        if (rad_dev_set(ranks_[i]->device) < 0) continue;
        if (ranks_[i]->kv.follow(ranks_[0]->kv) < 0)
            step_status_.store(RAD_E_NOMEM, std::memory_order_release);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    return RAD_OK;
}

int Engine::issue_ahead(const RadBatch** out) {
    *out = nullptr;
    if (!pipelining_) return RAD_OK;
    for (int i = 0; i < serial_rows_ && serial_done_ > 0; ++i)
        sched_.expect_draft(serial_row_[(size_t)i].seq, serial_done_);
    for (int i = 0; i < block_rows_; ++i)
        sched_.expect_draft(block_row_[(size_t)i].seq, drafter_.depth);
    const RadBatch* nb = sched_.step_next();
    if (!nb) return RAD_OK;
    /* Its rows' tokens, positions and slots, advanced on the card behind this step's acceptance
     * and drafts. */
    RAD_TRY(advance_all(ADV_TRUNK, 0, false));
    RAD_TRY(prepare_release(nb));
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(nb, std::memory_order_release);
    barrier_->arrive_and_wait();
    barrier_->arrive_and_wait();
    const int st = step_status_.load(std::memory_order_acquire);
    if (st < 0) {
        RAD_ERR("step %d, issued ahead, failed: %s -- the engine cannot continue", nb->step,
                rad_strerror(st));
        return st;
    }
    *out = nb;
    return RAD_OK;
}

void Engine::phase_mark(int p) {
    const auto now = std::chrono::steady_clock::now();
    phase_us_[p] += std::chrono::duration<double, std::micro>(now - phase_t_).count();
    phase_t_ = now;
}

/* Every error exit from the step loop goes through here, which does three things:
 *
 *   - aborts the barrier, so the parked rank threads see `stopping_`;
 *   - fails every queued or running request and answers its connection; the scheduler then
 *     refuses new submissions;
 *   - sets `loop_status_`, which run_server watches to stop HTTP, so the process exits with the
 *     error instead of queueing requests nothing will drain (spec §17). */
int Engine::fail_loop(int st) {
    stopping_.store(true, std::memory_order_release);
    if (barrier_) barrier_->abort();
    sched_.shutdown("the engine stopped on an error and is not serving", st < 0 ? st : RAD_E_STATE);
    loop_status_.store(st < 0 ? st : RAD_E_STATE, std::memory_order_release);
    return st;
}

int Engine::step_loop() {
    /* The batch is built on rank 0's stream into rank 0's memory, so this thread binds rank 0's
     * device; the backend's current device is thread-local and starts at device 0. */
    if (rad_dev_set(ranks_[0]->device) < 0)
        RAD_ERR("step loop: could not bind rank 0's device");
    /* The cost profile's fingerprint names the residency at start. Every cycle's move evidence is
     * reported against this count, so a move between two brackets still invalidates what follows
     * it (Scheduler::step_finished). */
    draft_moves_baseline_ = residency_moves();
    StepOut so;
    int idle = 0;
    std::vector<uint64_t> cancel;
    /* Not in KL mode: its scored rows are copied out of the logits buffer after collection, and a
     * pass issued ahead would overwrite them. A block drafter pipelines only on the card
     * (engine_draft.cpp); the host's driver stages from the committed sequence. */
    block_card_ = drafter_.kind == RAD_DRAFT_BLOCK && drafter_.depth > 0 && !cfg_.kld() &&
                  !embedding_ && [] { const char* v = std::getenv("RADIANCE_DRAFT_BLOCK_HOST");
                                      return !(v && *v && *v != '0'); }();
    pipelining_ = !cfg_.kld() && !embedding_ && (drafter_.kind != RAD_DRAFT_BLOCK || block_card_) &&
                  (serial_drafting() || block_card_ || cfg_.n_spec <= 0);
    const bool t_on = (int)log_level() >= (int)Log::Debug;
    /* The next step's forward pass when already issued behind the previous step, before its
     * readback (Scheduler::step_next). */
    const RadBatch* ahead = nullptr;
    while (!stopping_.load(std::memory_order_acquire)) {
        lora_epoch_.fetch_add(1, std::memory_order_acq_rel);
        phase_t_ = std::chrono::steady_clock::now();
        /* The profile is bound to the residency the run started on. Read the cumulative evidence
         * here -- the rank threads are parked at the barrier, so no rank writes -- and before the
         * cycle below opens: the mover's idle tick publishes its move after step_finished, and a
         * cycle priced before this read would be priced on the placement the run has left. */
        sched_.draft_residency(residency_moves() - draft_moves_baseline_);
        /* Encode media for waiting requests while nothing is in flight. The scheduler issues no
         * step ahead while a request waits, and a request with unencoded media waits. */
        if (!ahead && sched_.encodes_pending()) {
            if (const int es = run_encodes(); es < 0) return fail_loop(es);
        }
        const bool issued = ahead != nullptr;
        const RadBatch* b = ahead;
        ahead = nullptr;
        if (issued) sched_.promote_next();
        else        b = sched_.step();
        /* A cycle whose policy ended on a residency drift is refused above; the run stops here,
         * on the engine's existing error path, instead of drafting the rest of it at the ceiling. */
        if (sched_.draft_cost_aborted()) {
            RAD_ERR("draft cost: the residency no longer matches the profile; stopping instead of "
                    "drafting at the ceiling");
            return fail_loop(RAD_E_STATE);
        }
        if (!b) {
            /* Idle: back off in small steps instead of spinning; the first wait is short so a new
             * request into an empty engine is not delayed. */
            step_batch_.store(nullptr, std::memory_order_release);
            drain_completed();
            sched_.publish(/*idle=*/true);
            /* Work is waiting only on the encoder: loop immediately. */
            if (sched_.encodes_pending()) { idle = 0; continue; }
            /* Release deferred by the last step. A request cancelled while a drafter pass was being
             * built keeps its blocks until the passes naming them run (Scheduler::cancel), and
             * commit() normally releases them; an idle engine makes no commit, and those blocks
             * may be what blocks the next admission. Drain the streams first so no in-flight pass
             * still reads them. */
            if (sched_.has_deferred()) {
                for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
                sched_.release_deferred();
            }
            tier_tick();
            expire_tick(rad_mono_ns());
            loan_tick(/*idle=*/true);
            if (idle < 200) ++idle;
            std::this_thread::sleep_for(std::chrono::microseconds(idle * 5));
            continue;
        }
        idle = 0;
        if (t_on) phase_mark(PH_PLAN);

        /* Zero freshly assigned recurrent slots before the step that reads them. The manager
         * records the grab but cannot write (StateClear in core/mem/kv.h); the list is empty except
         * on a sequence's first step. Each rank clears by its own plan: the slot number is the
         * scheduler's, but the state size depends on the heads this rank holds, which can differ
         * per rank (rad_arch.h remainder_place). */
        for (const StateClear& c : ranks_[0]->kv.state_clears()) {
            for (auto& rk : ranks_) {
                const KVGroupPlan* gp = rk->kv.plan(c.group);
                if (!gp || !gp->stateful()) continue;
                const int64_t bytes = gp->layer_bytes_per_state();
                for (int64_t l = 0; l < gp->n_layers; ++l)
                    if (void* p = rk->kv.state_ptr(c.group, (int32_t)l, c.slot))
                        rad_memset_async(p, 0, bytes, rk->ctx.stream());
            }
        }
        ranks_[0]->kv.clear_state_clears();

        /* Entries that want logits rows, in BatchBuilder's out_ids order: those with
         * produces_token, 1 + n_spec rows each. Row and sequence indices differ (a mid-prompt
         * chunk produces no row). */
        const StepPlan& plan = sched_.plan();
        /* One entry per request; Sampler::build_step expands each into 1 + n_draft rows itself
         * (giving each position its RNG coordinate), so one push per row would stage (1+n)^2 rows.
         * `first_row` indexes sampled tokens, so it counts rows. */
        so.reqs.clear();
        so.first_row.assign(plan.e.size(), -1);
        int32_t logit_row = 0;
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (!e.produces_token) continue;
            so.first_row[i] = logit_row;
            logit_row += 1 + e.n_spec;
            so.reqs.push_back(const_cast<Request*>(e.req));
        }
        so.n_rows = logit_row;
#ifdef RAD_TEST_HOOKS
        /* What the previous step left, read before the reset below. The step point in front of
         * collect_step reports it beside the value the reset leaves. */
        const bool rad_test_prior_complete = so.complete;
#endif
        /* StepOut outlives the step (it is the loop's), so last step's completion must not be
         * readable as this one's before any wait has run. */
        so.complete = false;
        if (const int ss = stage_scored_rows(plan, so); ss < 0) return fail_loop(ss);

        /* The sampler's host work, run while the ranks enqueue (see build_mu_ in engine.h). It
         * touches only request state and host grammar automata; Sampler::run issues the uploads
         * after the gate opens.
         *
         * Cancellations are collected, not applied: the rank threads read scheduler state through
         * batch_for() meanwhile. They are applied after the join, before collect_step. */
        auto stage_sampler = [&]() -> int {
        if (!so.reqs.empty() && !embedding_) {
            /* Per-request sampler state is built on first sight: the grammar (compiled at
             * admission) is positioned and DRY sequence breakers are resolved against the
             * vocabulary. Without this, `grammar` and `dry_sequence_breakers` would silently do
             * nothing. Every rank does it, since each masks its own vocabulary shard.
             *
             * A failure (e.g. a left-recursive grammar) cancels the request, not the engine. The
             * server compiles at admission, so this should be unreachable. */
            for (Request* rq : so.reqs) {
                if (!rq) continue;
                bool failed = false;
                for (auto& rk : ranks_) {
                    if (rk->sampler.has_request(rq->id)) continue;
                    std::string err;
                    if (rk->sampler.begin_request(rq->id, rq->sp, &err) < 0) {
                        RAD_ERR("sampler: request %llu could not be started: %s; cancelling it",
                                (unsigned long long)rq->id, err.c_str());
                        failed = true;
                        break;
                    }
                }
                if (failed) {
                    for (auto& rk : ranks_) rk->sampler.end_request(rq->id);
                    cancel.push_back(rq->id);
                    continue;
                }
                rq->speculation_fixed = ranks_[0]->sampler.speculation_fixed(rq->id);
            }
            if (t_on) phase_mark(PH_BEGIN);
            for (auto& rk : ranks_) {
                const int s = rk->sampler.build_step(so.reqs, b->step);
                if (s < 0) { RAD_ERR("sampler build_step: %s", rad_strerror(s)); return s; }
                /* A request the sampler could not serve is cancelled; the step goes on. */
                for (uint64_t id : rk->sampler.take_failed()) cancel.push_back(id);
            }
            if (t_on) phase_mark(PH_BUILD);
        }
        return RAD_OK;
        };

        cancel.clear();
        int bst = RAD_OK;

        /* RADIANCE_LOG_STEPS: one line per step with its phase and each sequence's slot, query
         * length and context. Off by default (one line per decode step). */
        static const bool log_steps = std::getenv("RADIANCE_LOG_STEPS") != nullptr;
        if (log_steps) {
            std::string s = fmt("step %d %s n_seq=%lld n_tok=%lld", b->step,
                                b->phase == RAD_PHASE_PREFILL ? "prefill" :
                                b->phase == RAD_PHASE_DECODE  ? "decode"  : "MIXED",
                                (long long)b->n_seq, (long long)b->n_tok);
            for (int i = 0; i < b->n_seq && i < 8; ++i)
                s += fmt("  [%d] slot=%d q=%d ctx=%d", i, plan.e[(size_t)i].slot,
                         plan.e[(size_t)i].n_tokens, plan.e[(size_t)i].ctx_len);
            RAD_INFO("%s", s.c_str());
        }

        /* Restore the recurrent half of a prefix hit before the step that reads it (the adopted
         * blocks carry the attention layers), copying from the snapshot's host slot. Enqueued on
         * each rank's stream before the release, so no synchronisation is needed.
         *
         * Under the scheduler's lock, because an HTTP thread's cancel mutates the sequence table.
         * Taken only when the step has a restore. */
        bool any_restore = false;
        for (const StepEntry& e : plan.e) if (e.ckpt_restore >= 0) { any_restore = true; break; }
        if (any_restore) {
            const int rs = sched_.locked([&]() -> int {
                for (const StepEntry& e : plan.e) {
                    if (e.ckpt_restore < 0) continue;
                    if (!state_slots_of(e.seq, &ckpt_slots_)) continue;
                    /* The window behind the hit, in the blocks admit_one seeded. */
                    if (ranks_[0]->kv.has_window_snapshot() &&
                        !ranks_[0]->kv.window_rows(e.seq, e.ctx_len, &ckpt_win_)) {
                        RAD_ERR("sequence %llu: no window rows at %d to restore into",
                                (unsigned long long)e.seq, (int)e.ctx_len);
                        return RAD_E_STATE;
                    }
                    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
                        auto& rk = ranks_[ri];
                        if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                        const int s = rk->kv.checkpoint_restore(ckpt_slots_.data(),
                                                                (int)ckpt_slots_.size(),
                                                                e.ckpt_restore, rk->ctx.stream(),
                                                                &ckpt_win_);
                        if (s < 0) {
                            RAD_ERR("could not restore sequence %llu from checkpoint slot %d on "
                                    "rank %d: %s", (unsigned long long)e.seq,
                                    (int)e.ckpt_restore, (int)ri, rad_strerror(s));
                            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                            return s;
                        }
                    }
                    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                }
                return RAD_OK;
            });
            if (rs < 0) return fail_loop(rs);
        }

        /* Host time before the release, with the device idle; time between the two arrivals is
         * the ranks enqueueing. The split separates host-bound from device-bound steps. */
        if (t_on) phase_mark(PH_PRE);

        if (!issued) {
            /* Reset the gate while every rank is parked, so none sees a stale ready flag. */
            {
                std::lock_guard<std::mutex> lk(build_mu_);
                build_ready_  = false;
                build_status_ = RAD_OK;
            }
            if (const int ps = prepare_release(b); ps < 0) return fail_loop(ps);
            /* An embedding program has no sampler to issue behind its pass. */
            issue_what_.store(embedding_ ? kIssueModel : kIssueAll, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();      /* release the ranks */
            {
                /* Stage the sampler while the ranks enqueue and the device runs the pass. */
                bst = stage_sampler();
                {
                    std::lock_guard<std::mutex> lk(build_mu_);
                    build_ready_  = true;
                    build_status_ = bst;
                }
                build_cv_.notify_all();
            }
            barrier_->arrive_and_wait();      /* wait for them */
        } else {
            /* The forward pass was issued ahead and the previous step has now committed, so the
             * sampler is staged from the correct state and its chain queued behind the running
             * pass. */
            bst = stage_sampler();
            {
                std::lock_guard<std::mutex> lk(build_mu_);
                build_ready_  = true;
                build_status_ = bst;
            }
            issue_what_.store(kIssueSampler, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();
            barrier_->arrive_and_wait();
        }
        if (t_on) phase_mark(PH_RANKS);

        /* Deferred out of the staging, where the rank threads were reading the scheduler. */
        for (uint64_t id : cancel) sched_.cancel(id);
        cancel.clear();

        /* A staging failure is known only after the ranks were released; the gate passed it to
         * them, so they skipped the sampler and reached the join. */
        if (bst < 0) return fail_loop(bst);

        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            /* A device fault is not recoverable in-process (spec §17). */
            RAD_ERR("step %d failed: %s -- the engine cannot continue", b->step, rad_strerror(st));
            return fail_loop(st);
        }

        /* Snapshot after the step that produced the state: the chunker ended this chunk at the
         * checkpoint position, so copy 0 holds exactly that state. The copy to the host slot is
         * issued before collect_step's readback and before any step issued ahead, which orders the
         * next write of the state after it. Under the scheduler's lock, as the restore. */
        bool any_save = false;
        for (const StepEntry& e : plan.e)
            if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0) { any_save = true; break; }
        if (any_save) {
            const int ss = sched_.locked([&]() -> int {
                for (const StepEntry& e : plan.e) {
                    if (e.ckpt_slot < 0 || e.ckpt_tok < 0) continue;
                    if (!state_slots_of(e.seq, &ckpt_slots_)) continue;
                    /* The window behind the checkpoint position. The chunk ending there still
                     * holds it, so a miss is a broken invariant; skipping would commit a slot
                     * holding another sequence's pages. */
                    if (ranks_[0]->kv.has_window_snapshot() &&
                        !ranks_[0]->kv.window_rows(e.seq, e.ctx_len + e.ckpt_tok + 1,
                                                   &ckpt_win_)) {
                        RAD_ERR("sequence %llu: the window behind checkpoint %d is not held",
                                (unsigned long long)e.seq, (int)(e.ctx_len + e.ckpt_tok + 1));
                        return RAD_E_STATE;
                    }
                    for (auto& rk : ranks_) {
                        if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                        const int s = rk->kv.checkpoint_save(ckpt_slots_.data(),
                                                             (int)ckpt_slots_.size(),
                                                             e.ckpt_slot, rk->ctx.stream(),
                                                             &ckpt_win_);
                        if (s < 0) {
                            RAD_ERR("rank %d could not snapshot sequence %llu into checkpoint slot "
                                    "%d: %s", rk->index, (unsigned long long)e.seq,
                                    (int)e.ckpt_slot, rad_strerror(s));
                            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                            return s;
                        }
                    }
                    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
                }
                return RAD_OK;
            });
            if (ss < 0) return fail_loop(ss);
        }

        if (t_on) phase_mark(PH_SAVE);
        /* Queue the drafter's passes behind the step before waiting: the card computes acceptance
         * itself and every pass reads it there (sched/advance.h), so one wait covers the whole
         * step. */
        st = block_card_ ? issue_block_drafter(b->step, so) : issue_serial_drafter(b->step, so);
        if (st < 0) return fail_loop(st);
        st = mark_step_end(b, so);
        if (st < 0) return fail_loop(st);
        /* Then the next step's forward pass, before this step is read back. Its rows continue from
         * the card's acceptance and verify the new drafts; its sampler waits for this step's
         * commit. The card runs straight on while the host waits, commits and stages. */
        st = issue_ahead(&ahead);
        if (st < 0) return fail_loop(st);
#ifdef RAD_TEST_HOOKS
        rad_test_step_point(b->step, (int)so.n_rows, rad_test_prior_complete, so.complete);
#endif
        st = embedding_ ? collect_pooled(b, so) : collect_step(b, so);
        if (t_on) phase_mark(PH_COLLECT);
        if (st < 0) return fail_loop(st);
        /* The step's sampler chain, verify op included, has returned: one line at the first
         * coupled verify that actually completed. */
        couple_verify_evidence(b->step);
        if (cfg_.kld()) {
            st = score_rows(b, so);
            if (st < 0) return fail_loop(st);
        }

        /* Buffer probe after the trunk step; drafters dump after their own passes. */
        dump_buffers(ranks_[0]->ctx, b->step);

        /* --profile-ops also prints during the run. Counters are cumulative, so each dump covers
         * the run so far. Every 32 steps by default (RADIANCE_PROFILE_EVERY): per-op timing
         * synchronises around every launch (over a thousand per step on a large model), so a
         * profiled step is several times slower. */
        static const int prof_every = [] {
            const char* v = std::getenv("RADIANCE_PROFILE_EVERY");
            const int n = (v && *v) ? (int)std::strtol(v, nullptr, 10) : 32;
            return n > 0 ? n : 32;
        }();
        if (cfg_.profile_ops && b->step > 0 && (b->step % prof_every) == 0)
            for (auto& rk : ranks_) rk->ctx.dump_profile();

        /* One line per step at -vv, the only log view of the loop itself. */
        RAD_TRACE("step %d: %lld tok / %lld seq, %lld logit row(s) -> tokens %s", b->step,
                  (long long)b->n_tok, (long long)b->n_seq, (long long)b->n_out,
                  so.token.empty() ? "(none)" : fmt("%d%s", so.token[0],
                                                    so.token.size() > 1 ? " ..." : "").c_str());
        /* And the input ids (first 32): wrong ids look like a wrong forward pass from the logits
         * alone. */
        if ((int)log_level() >= (int)Log::Trace && b->n_tok > 0 && b->token_ids) {
            std::string ids;
            const int64_t n = std::min<int64_t>(b->n_tok, 32);
            std::vector<int32_t> host((size_t)n);
            if (rad_memcpy_async(host.data(), b->token_ids, n * 4, ranks_[0]->ctx.stream()) >= 0 &&
                rad_stream_sync(ranks_[0]->ctx.stream()) >= 0) {
                for (int64_t i = 0; i < n; ++i) ids += fmt("%s%d", i ? " " : "", host[(size_t)i]);
                RAD_TRACE("step %d in: [%s%s]", b->step, ids.c_str(),
                          b->n_tok > n ? " ..." : "");
            }
        }

        Scheduler::StepResult res{};
        res.n_seq      = (int)plan.e.size();
        res.n_accepted = so.accepted.data();
        res.n_raw      = so.raw_match.data();
        res.token      = so.token.data();
        res.eos        = so.eos.data();
        if (!so.lp.empty()) {
            res.lp        = so.lp.data();
            res.first_row = so.first_row.data();
        }
        st = sched_.commit(res);
        if (t_on) phase_mark(PH_COMMIT);
        if (st < 0) { RAD_ERR("commit: %s", rad_strerror(st)); return fail_loop(st); }

        /* The drafter for the next step. The host's block driver runs here, since it stages from
         * the committed sequence. A serial one, or a block one on the card, already ran; what
         * remains is to check its acceptance against the host's and publish its proposals. Each
         * driver returns immediately for a kind that is not its own. */
        st = drafter_.kind != RAD_DRAFT_BLOCK ? finish_serial_drafter(so)
           : block_card_                      ? finish_block_drafter(so)
                                              : run_block_drafter(b->step);
        if (st < 0) return fail_loop(st);
        /* The step ends here, after its draft rounds, so step timing includes them
         * (Scheduler::step_finished). What the tape and the placement engines did inside that
         * bracket goes with it: the scheduler cannot see either, and a state nobody established
         * must reach A as Unknown, not as a measured None. The evidence is the one the rank
         * threads froze for this exact step (note_evidence), not a counter delta read now -- the
         * next step's ahead pass has been released since. A step no rank reported is Unknown. */
        DraftStepEvidence ev;
        /* No rank ran a pass for this step (a sampler-only release with nothing issued ahead):
         * Unknown, so A excludes the sample instead of pricing it as undisturbed. */
        if (!step_evidence(b->step, &ev)) ev = DraftStepEvidence{};
        /* The per-step fields above describe this bracket; the residency is cumulative, so a move
         * the scheduler's own ticks made between two brackets still reaches the bracket measured
         * after it. */
        ev.moves_since_bind = residency_moves() - draft_moves_baseline_;
#ifdef RAD_TEST_HOOKS
        rad_test_step_evidence(b->step, (int)ev.replay_known, (int)ev.replay, (int)ev.staged,
                               (int)ev.moved);
#endif
        sched_.step_finished(ev);
        tier_tick();
        loan_tick(/*idle=*/false);
        const uint64_t now = rad_mono_ns();
        arena_tick(now);
        expire_tick(now);
        drain_completed();
        sched_.publish(/*idle=*/false);

        /* One line every 128 steps; the phases sum to the step. `ranks` is the enqueue; device time
         * shows up in `collect`, the step's only host synchronisation. */
        if (t_on) {
            phase_mark(PH_DRAFT);
            if (++phase_n_ == 128) {
                double tot = 0;
                for (int i = 0; i < PH_N; ++i) tot += phase_us_[i];
                RAD_DEBUG("step %s: %.2f ms = plan %.2f begin %.2f build %.2f pre %.2f "
                          "ranks %.2f save %.2f collect %.2f commit %.2f draft %.2f (mean/128)",
                          b->phase == RAD_PHASE_DECODE ? "decode" : "other", tot / 128000.0,
                          phase_us_[PH_PLAN] / 128000.0,  phase_us_[PH_BEGIN] / 128000.0,
                          phase_us_[PH_BUILD] / 128000.0, phase_us_[PH_PRE] / 128000.0,
                          phase_us_[PH_RANKS] / 128000.0, phase_us_[PH_SAVE] / 128000.0,
                          phase_us_[PH_COLLECT] / 128000.0, phase_us_[PH_COMMIT] / 128000.0,
                          phase_us_[PH_DRAFT] / 128000.0);
                for (int i = 0; i < PH_N; ++i) phase_us_[i] = 0;
                phase_n_ = 0;
            }
        }
    }
    return RAD_OK;
}

/* Issue the copy of the sampled ids into pinned memory behind the sampler, without waiting.
 * Issued by the serial drafter before its passes (a head sampling through the sampler's buffer
 * overwrites the ids), otherwise just before the wait. */
int Engine::collect_issue(StepOut& so) {
    so.rb = nullptr;
    so.lg.clear();
    if (so.n_rows <= 0) return RAD_OK;
    Rank& r0 = *ranks_[0];
    void* src = r0.ctx.buf_ptr(r0.tok_buf);
    if (!src) {
        RAD_ERR("the sampler's token buffer has no address -- it was declared but never planned");
        return RAD_E_STATE;
    }
    so.rb = readback_.reserve(so.n_rows);
    if (!so.rb) return RAD_E_NOMEM;
    RAD_TRY(rad_memcpy_async(so.rb, src, (int64_t)so.n_rows * 4, r0.ctx.stream()));

    /* The first logits row at -vv, the only direct view of the forward pass's output (rad-kbench
     * checks single ops, not whole layers). */
    const int64_t nv = meta_.n_vocab;
    if ((int)log_level() >= (int)Log::Trace && nv > 0) {
        void* lp = r0.ctx.buf_ptr(r0.program.logits_buf);
        if (lp) {
            so.lg.resize((size_t)nv);
            rad_memcpy_async(so.lg.data(), lp, nv * 4, r0.ctx.stream());
        }
    }
    return RAD_OK;
}

int Engine::mark_step_end(const RadBatch* b, StepOut& so) {
    if (embedding_) RAD_TRY(collect_pooled_issue(b, so));
    else if (so.n_rows > 0 && !so.rb) RAD_TRY(collect_issue(so));
    RAD_TRY(rad_event_record(step_end_, ranks_[0]->ctx.stream()));
    /* Each other rank's own end, recorded here rather than in the rank thread: the join barrier
     * above already proves every rank has enqueued this step, so a record enqueued now on its
     * stream covers all of its work and none of the step issued ahead below. */
    for (size_t i = 1; i < ranks_.size(); ++i)
        RAD_TRY(rad_event_record(rank_end_[i], ranks_[i]->ctx.stream()));
    return RAD_OK;
}

/* The step's completion, on every rank. step_end_ is card-local and the rank barrier joins
 * enqueues, so it alone would let the host commit -- and advance a history water level -- while
 * another rank's stream is still running. The extra waits cost nothing in the common case: the
 * ranks were released together, so theirs are done when rank 0's fires. */
int Engine::await_step_end() {
    RAD_TRY(rad_event_sync(step_end_));
    for (size_t i = 1; i < ranks_.size(); ++i) RAD_TRY(rad_event_sync(rank_end_[i]));
    return RAD_OK;
}

/* Issue the copy of an embedding step's pooled rows (the first n_seq) into pinned memory on rank
 * 0's stream; every rank holds the same vectors. collect_pooled waits. */
int Engine::collect_pooled_issue(const RadBatch* b, StepOut& /*so*/) {
    if (b->n_seq <= 0) return RAD_OK;
    Rank& r0 = *ranks_[0];
    const void* src = r0.ctx.buf_ptr(r0.program.pooled_buf);
    if (!src) {
        RAD_ERR("the pooled output was declared but never planned");
        return RAD_E_STATE;
    }
    return rad_memcpy_async(pooled_host_, src,
                            (int64_t)b->n_seq * pooled_dim_ * (int64_t)sizeof(float),
                            r0.ctx.stream());
}

/* Hand each sequence's vector to its sink, and set eos for every sequence so the commit finishes
 * it: an embedding request is one step long. The token is never emitted (Request::embed). */
int Engine::collect_pooled(const RadBatch* b, StepOut& so) {
    const StepPlan& plan = sched_.plan();
    so.token.assign(plan.e.size(), 0);
    so.accepted.assign(plan.e.size(), 0);
    so.raw_match.assign(plan.e.size(), 0);
    so.eos.assign(plan.e.size(), 1);
    if (plan.e.empty()) return RAD_OK;
    RAD_TRY(await_step_end());
    so.complete = true;
    if ((int64_t)plan.e.size() != (int64_t)b->n_seq) {
        RAD_ERR("an embedding step planned %zu sequences and its batch carries %lld",
                plan.e.size(), (long long)b->n_seq);
        return RAD_E_STATE;
    }
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (!e.req) continue;
        if (!e.produces_token) {
            RAD_ERR("an embedding step carried part of request %llu's input; an encoder must see "
                    "the whole of it in one step", (unsigned long long)e.req->id);
            return RAD_E_STATE;
        }
        const float* row = pooled_host_ + (int64_t)i * pooled_dim_;
        server::sink_embedding(e.req->sink, row, pooled_dim_);
    }
    return RAD_OK;
}

/* Scored rows (Request::token_scores): the logits row at a scored prompt's last position, copied
 * to the host before the sampler can rewrite it in place. stage_scored_rows picks the rows while
 * every rank is parked; issue_scored_rows copies one row per rank stream ahead of its sampler;
 * collect_scored_rows reads each request's ids after the step's wait, before the commit. A whole
 * row is copied (512 KB at a 128K vocabulary), which is cheaper than one copy per id. */
int Engine::stage_scored_rows(const StepPlan& plan, const StepOut& so) {
    score_rows_.clear();
    if (embedding_) return RAD_OK;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const Request* r = plan.e[i].req;
        if (so.first_row[i] < 0 || !r || !r->token_scores || !r->output.empty()) continue;
        score_rows_.push_back({ so.first_row[i], r->token_scores.get() });
    }
    if (score_rows_.empty()) return RAD_OK;
    const int64_t n = (int64_t)score_rows_.size();
    for (auto& rk : ranks_) {
        if (n <= rk->score_rows_cap) continue;
        if (ranks_.size() > 1) RAD_TRY(rad_dev_set(rk->device));
        if (rk->score_host) rad_dev_free(rk->score_host, RAD_MEM_HOST_PINNED);
        const int64_t width = rk->sampler.config().n_vocab_local;
        rk->score_host = static_cast<float*>(
            rad_dev_alloc(n * width * (int64_t)sizeof(float), RAD_MEM_HOST_PINNED));
        rk->score_rows_cap = rk->score_host ? n : 0;
        if (!rk->score_host) {
            RAD_ERR("scored rows: no pinned memory for %lld logits rows", (long long)n);
            return RAD_E_NOMEM;
        }
    }
    if (ranks_.size() > 1) RAD_TRY(rad_dev_set(ranks_[0]->device));
    return RAD_OK;
}

int Engine::issue_scored_rows(Rank& r) {
    const int64_t width = r.sampler.config().n_vocab_local;
    const char* logits = static_cast<const char*>(r.ctx.buf_ptr(r.program.logits_buf));
    if (!logits || !r.score_host) return RAD_E_STATE;
    for (size_t k = 0; k < score_rows_.size(); ++k)
        RAD_TRY(rad_memcpy_async(r.score_host + (int64_t)k * width,
                                 logits + (int64_t)score_rows_[k].first * width * 4,
                                 width * 4, r.ctx.stream()));
    return RAD_OK;
}

int Engine::collect_scored_rows() {
    if (score_rows_.empty()) return RAD_OK;
    /* step_end_ fences rank 0's stream; every other rank's copies are fenced here. */
    for (size_t i = 1; i < ranks_.size(); ++i) {
        RAD_TRY(rad_dev_set(ranks_[i]->device));
        RAD_TRY(rad_stream_sync(ranks_[i]->ctx.stream()));
    }
    if (ranks_.size() > 1) RAD_TRY(rad_dev_set(ranks_[0]->device));
    for (size_t k = 0; k < score_rows_.size(); ++k) {
        TokenScores& ts = *score_rows_[k].second;
        ts.logits.assign(ts.tokens.size(), 0.0f);
        for (size_t j = 0; j < ts.tokens.size(); ++j) {
            const int64_t t = ts.tokens[j];
            for (auto& rk : ranks_) {
                const SamplerConfig& sc = rk->sampler.config();
                if (t < sc.vocab_off || t >= sc.vocab_off + sc.n_vocab_local) continue;
                ts.logits[j] = rk->score_host[(int64_t)k * sc.n_vocab_local + (t - sc.vocab_off)];
                break;
            }
        }
    }
    score_rows_.clear();
    return RAD_OK;
}

/* One sampled row's logprobs into the step's table, sized on first use. A non-finite row
 * (RAD_E_INVAL) is left at token -1, and the scheduler fails only that request. A row outside the
 * step, or never staged for logprobs, is an engine error. */
int Engine::row_logprobs(StepOut& so, int32_t row, int32_t token) {
    if (so.lp.empty()) so.lp.resize((size_t)so.n_rows);
    if (row < 0 || row >= so.n_rows) return RAD_E_STATE;
    TokenLogprobs& lp = so.lp[(size_t)row];
    const int s = ranks_[0]->sampler.logprobs_for(row, token, &lp);
    if (s == RAD_E_INVAL) { lp = TokenLogprobs{}; return RAD_OK; }
    return s;
}

/* Wait for the sampled ids and build the scheduler's per-sequence arrays. The step path's only
 * host synchronisation: the host needs the tokens to commit and stream them (spec §5.4). */
int Engine::collect_step(const RadBatch* b, StepOut& so) {
    const StepPlan& plan = sched_.plan();
    so.token.assign(plan.e.size(), 0);
    so.accepted.assign(plan.e.size(), 0);
    so.raw_match.assign(plan.e.size(), 0);
    so.eos.assign(plan.e.size(), 0);
    so.lp.clear();
    /* A step with no sampled rows has nothing to read back, so the host does not wait for a readback
     * here and can run ahead. That is not completion evidence, and a history-only or mid-prompt
     * pass still has to be proven before anything advances on it -- so wait when the step actually
     * issued one, and only then. A pure prefill step issues no draft pass and keeps running ahead.
     * Either way the fact is `so.complete`, never the row count or the batch pointer. */
    if (so.n_rows <= 0) {
        if (!serial_hist_issued_) return RAD_OK;
        RAD_TRY(await_step_end());
        so.complete = true;
        return RAD_OK;
    }

    so.rows.resize((size_t)so.n_rows);
    /* Wait on the step-end events, not the streams: a step issued ahead is queued behind them. */
    RAD_TRY(await_step_end());
    so.complete = true;
    std::memcpy(so.rows.data(), so.rb, (size_t)so.n_rows * sizeof(int32_t));
    so.rb = nullptr;
    RAD_TRY(collect_scored_rows());
    const std::vector<float>& lg = so.lg;
    const int64_t nv = meta_.n_vocab;
    if (!lg.empty()) {
        /* Top three of the first sampled row, to tell a fixed spike in the distribution from a
         * sampler that ignores it. */
        int64_t i0 = 0, i1 = 0, i2 = 0;
        for (int64_t i = 1; i < nv; ++i) {
            if (lg[(size_t)i] > lg[(size_t)i0])      { i2 = i1; i1 = i0; i0 = i; }
            else if (lg[(size_t)i] > lg[(size_t)i1]) { i2 = i1; i1 = i; }
            else if (lg[(size_t)i] > lg[(size_t)i2]) { i2 = i; }
        }
        double sum = 0, amax = 0;
        for (int64_t i = 0; i < nv; ++i) { sum += lg[(size_t)i]; amax = std::max(amax, (double)std::fabs(lg[(size_t)i])); }
        RAD_TRACE("step %d logits: top %lld=%.4f  %lld=%.4f  %lld=%.4f   mean %.4f  amax %.4f",
                  b->step, (long long)i0, (double)lg[(size_t)i0], (long long)i1,
                  (double)lg[(size_t)i1], (long long)i2, (double)lg[(size_t)i2],
                  sum / (double)nv, amax);
    }

    for (size_t i = 0; i < plan.e.size(); ++i) {
        if (so.first_row[i] < 0) continue;
        const StepEntry& e = plan.e[i];

        if (e.n_spec > 0) {
            /* Accept draft[k] when it equals the token the target itself drew at position k. This
             * is exact at any temperature without rejection sampling or probability planes,
             * because the sampler gives every verified position its own RNG coordinate
             * (`p.pos = n_out + k`, sampler.cpp): y_k is the target's own draw at that output
             * position, and each later token is conditioned on a prefix accepted only because it
             * matched. The output is a draw from the target.
             *
             * It is not byte-identical to non-speculative decoding above temperature 0: a verify
             * step computes 1 + n_spec rows and split-K depends on M, so logits differ slightly
             * and a pick near a CDF boundary can flip. Greedy output matches exactly; both are
             * deterministic.
             *
             * The cost is a lower acceptance rate than rejection sampling (P(y == x) instead of
             * min(1, p/q)). `so.rows` holds the sampled tokens (the argmax at temperature 0). */
            if (!e.req) continue;
            int32_t out[RAD_SCHED_MAX_SPEC + 1] = { 0 };
            int32_t acc = 0;
            RAD_TRY(accept_greedy(e.draft, e.n_spec, &so.rows[(size_t)so.first_row[i]], out, &acc));
            /* The uncensored tally, taken before the cut below: a match is a comparison the target
             * made, whether or not the step went on to commit it. */
            so.raw_match[i] = acc;

            /* Truncated at the first end-of-generation token, so `accepted` and `token` (which
             * drive the KV rollback, the conv cursor and the emitted text) say how far the step
             * really got. */
            acc = accept_cut_at_eog(out, acc,
                                    [&](int32_t t) { return vocab_ && vocab_->is_eog(t); });

            so.accepted[i] = acc;
            so.token[i]    = out[acc];
            so.eos[i]      = vocab_ && vocab_->is_eog(so.token[i]) ? 1 : 0;
            /* Each emitted token's logprobs come from its own row, the distribution it was drawn
             * from. */
            if (e.req && e.req->sp.n_probs >= 0)
                for (int32_t j = 0; j <= acc; ++j)
                    RAD_TRY(row_logprobs(so, so.first_row[i] + j, out[j]));
            /* Advance the grammar over every emitted token, accepted drafts included, up to the
             * truncated `acc`, so it sees exactly what the client sees. */
            if (e.req)
                for (auto& rk : ranks_)
                    for (int32_t j = 0; j <= acc; ++j) {
                        const int as = rk->sampler.accept_token(e.req->id, out[j]);
                        /* A parse that outgrew the grammar's bounds fails the request at its next
                         * mask; not an engine defect. */
                        if (as == RAD_E_FULL || as == RAD_E_NOMEM)
                            RAD_WARN("engine: req %llu: its grammar's parse outgrew the bounds a "
                                     "grammar may use at token %d; the request fails at its next "
                                     "step", (unsigned long long)e.req->id, out[j]);
                        else if (as < 0)
                            RAD_ERR("engine: req %llu emitted token %d (position %d of %d "
                                    "accepted, n_spec %d) that its grammar forbids -- the mask and "
                                    "the pick disagree. The same condition on the speculative WALK "
                                    "is expected and is a debug line in gbnf.cpp; only this one, "
                                    "on an ACCEPTED token, is a defect.",
                                    (unsigned long long)e.req->id, out[j], j, acc, e.n_spec);
                    }
            continue;
        }
        so.token[i] = so.rows[(size_t)so.first_row[i]];
        so.eos[i]   = vocab_ && vocab_->is_eog(so.token[i]) ? 1 : 0;
        if (e.req && e.req->sp.n_probs >= 0)
            RAD_TRY(row_logprobs(so, so.first_row[i], so.token[i]));
        /* Advance the grammar on every rank: each builds the mask for its own vocabulary shard from
         * its own automaton. */
        if (e.req)
            for (auto& rk : ranks_) rk->sampler.accept_token(e.req->id, so.token[i]);
    }

    /* Update may_speculate here, because a lazy grammar becomes constraining inside accept_token
     * above; a stale value would give a constrained request a draft the sampler refuses. */
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (!e.req) continue;
        Request* r = const_cast<Request*>(e.req);
        r->may_speculate = ranks_[0]->sampler.allows_speculation(r->id);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the KV loan */
/* The only place the boundary between the KV cache and the expert slab moves.
 *
 * The budget divides the card once at startup, so an unused KV block is VRAM no expert can use.
 * The cache therefore lends what it does not hold at addresses it already owns: no allocation,
 * mapping or driver call. There is exactly one lender, so the same memory is never in two pools.
 *
 * The order is the safety argument:
 *
 *   1. the cache reports what it can spare (above its highest held block, less its buffer);
 *   2. the cache's mark is set to the larger of that and what any rank's slab actually holds, so
 *      a shrink completes only as units leave the slots;
 *   3. only then are the slabs told the offer, which they fill into or drain out of.
 *
 * A request that cannot wait for a shrink takes the blocks at once through a callback into every
 * slab (KVManager::set_reclaim, installed in start()). */
void Engine::loan_tick(bool idle) {
    if (!lending_) return;
    const uint64_t now = rad_mono_ns();
    if (now - loan_last_ns_ < 100000000ull) return;    /* 100 ms */
    loan_last_ns_ = now;

    /* Under the scheduler's lock: the loan moves rank 0's live mark and free count, which an HTTP
     * thread's cancel changes and /metrics reads under that lock. */
    const auto lk = sched_.hold();
    Rank& lead = *ranks_[0];

    /* tier_tick has already released every entry whose host copy landed, so `spare` counts
     * finished conversations. The loan grows only in steps of at least a prefill chunk, so small
     * fluctuations do not move the boundary; it shrinks at once, the cache regaining blocks as the
     * slabs vacate them. */
    const int64_t cur   = lead.kv.loan_tokens();
    const int64_t spare = lead.kv.spare_tokens();
    const int64_t chunk = std::max<int64_t>(cfg_.max_tok, 1);
    int64_t offer = cur;
    if (spare < cur || spare >= cur + chunk) offer = spare;
    int64_t held = 0;
    for (auto& r : ranks_) held = std::max(held, r->mover.loan_held(Mover::kLendKV));

    /* The cache moves first when the loan grows, the slabs first when it shrinks, so no block is
     * ever both a cache block and a slab slot. */
    if (lead.kv.set_loan(std::max(offer, held)) < 0) {
        RAD_ERR("kv loan: the cache refused a loan of %lld tokens it said it could spare; lending "
                "stops", (long long)std::max(offer, held));
        lending_ = false;
        return;
    }
    for (size_t i = 1; i < ranks_.size(); ++i) (void)ranks_[i]->kv.follow(lead.kv);
    for (auto& r : ranks_) {
        if (rad_dev_set(r->device) < 0) continue;
        (void)r->mover.set_loan(Mover::kLendKV, offer);
        if (idle) {
            const int st = r->mover.idle_tick(&r->heat);
            if (st < 0) RAD_WARN("rank %d: filling the lent slots while idle: %s", r->index,
                                 rad_strerror(st));
        }
    }
    (void)rad_dev_set(ranks_[0]->device);
}

/* ------------------------------------------------------------------ the arena's level */
/* Pick the step's arena level and how much of the arena the slab may keep. A step's rows are its
 * tokens or, for the drafter's passes, one per drafted position per sequence; the smallest level
 * serving that many is used, else the full plan, which lends nothing.
 *
 * If the level needs memory the slab holds, it is reclaimed synchronously before the step: units
 * in those slots revert to their host copies (an address change, not a copy) and both streams
 * that can read a slot are drained. A prefill pays this on its first chunk only, since the loan
 * does not regrow until steps have been small for a while (arena_tick). */
int Engine::arena_step(const RadBatch* b) {
    if (arena_rows_.size() <= 1 || !b) return RAD_OK;
    /* An encoder pass needs the full plan: its activations are sized by the pass, and a plugin
     * declares them small under shape_probe. */
    const int64_t need = b->enc ? INT64_MAX : std::max<int64_t>(
        b->n_tok, (int64_t)b->n_seq * (1 + std::max(cfg_.n_spec, 0)));
    const int level = arena_level(need);
    const uint64_t now = rad_mono_ns();
    arena_last_ = arena_rows_[(size_t)level];
    if (arena_last_ >= arena_floor_) { arena_floor_ = arena_last_; arena_floor_ns_ = now; }
    /* Whether the next step continues a prompt, so the stager stages its first layers at this
     * step's end. Never for a step issued ahead (a decode step whose plan the scheduler does not
     * hold). */
    const bool more = b->phase != RAD_PHASE_DECODE && sched_.prefill_continues();
    for (auto& r : ranks_) {
        const int64_t allow = r->arena_lent[(size_t)level];
        if (r->mover.loan(Mover::kLendArena) > allow ||
            r->mover.loan_held(Mover::kLendArena) > allow) {
            RAD_TRY(rad_dev_set(r->device));
            const int st = r->mover.reclaim(Mover::kLendArena, allow, &r->heat);
            if (st < 0) {
                RAD_ERR("rank %d: taking the arena back from the expert slab for a %lld-token step: "
                        "%s", r->index, (long long)b->n_tok, rad_strerror(st));
                return st;
            }
        }
        RAD_TRY(r->ctx.set_level(level));
        /* The full plan lends nothing, so its top (the staging region) is the step's ring; the
         * span's shared level keeps a place of its own for it (stager.h); every other level
         * lends both. */
        char* const ring = level == 0 ? r->stager.ring_full()
                         : arena_last_ == arena_span_rows_ && r->arena_low_off >= 0
                               ? r->stager.ring_low() : nullptr;
        r->stager.set_ring(ring);
        /* The next step's first layers go ahead only from the full plan, into the region at the
         * top that no level's buffers use. At the shared level the ring is memory the full plan
         * computes in, and a copy still going there when such a step starts would race it; a
         * pass's own copies are behind its own ops, which wait for them. */
        r->stager.set_next_prefill(more && level == 0);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    return RAD_OK;
}

int Engine::arena_level(int64_t rows) const { return arena_level_for(arena_rows_, rows); }

/* The arena loan grows only after large steps stop. A prefill is chunks with decode steps between,
 * and lending after every chunk would refill and recall the slab each time. So the floor (the
 * largest step seen) holds for two seconds after it was last reached, then falls to the latest
 * step's size. */
void Engine::arena_tick(uint64_t now_ns) {
    if (arena_rows_.size() <= 1) return;
    if (arena_last_ < arena_floor_ && now_ns - arena_floor_ns_ > 2000000000ull) {
        arena_floor_ = arena_last_;
        arena_floor_ns_ = now_ns;
    }
    const int level = arena_level(arena_floor_);
    for (auto& r : ranks_) {
        const int64_t want = r->arena_lent[(size_t)level];
        if (want <= r->mover.loan(Mover::kLendArena)) continue;
        if (rad_dev_set(r->device) < 0) continue;
        (void)r->mover.set_loan(Mover::kLendArena, want);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
}

/* ------------------------------------------------------------------ the idle-tier tick */
/* Runs every iteration, busy or idle. A host copy is reported once its event signals, so the
 * first tick after it lands collects it and hands out the next; a slower clock would only delay
 * releasing a finished conversation's VRAM. When the host tier has refused a copy for lack of
 * arena room, room is made at once.
 *
 * The scheduler's lock covers only the bookkeeping (planning and reporting touch rank 0's
 * manager, the prefix cache and the index an HTTP cancel changes). Issuing and polling touch none
 * of that, and a transfer's records stay pending while it is out. */
void Engine::tier_tick() {
    Rank& lead = *ranks_[0];
    if (!pump_ || !lead.tierx.enabled()) return;
    TierPump& P = *pump_;
    const uint64_t now = rad_mono_ns();

    /* ---- a copy that has landed on every rank is reported ---------------------------------- */
    if (!P.copy.empty()) {
        bool all = true, good = true;
        for (size_t i = 0; i < ranks_.size(); ++i) {
            if (P.copy_st[i] == 0) P.copy_st[i] = ranks_[i]->tierx.copy_state();
            if (P.copy_st[i] == 0) all = false;
            else if (P.copy_st[i] < 0) good = false;
        }
        if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
        if (all) {
            /* A copy is good only if every rank's shard landed; otherwise the host copy is
             * partly stale, so the whole batch fails and every slot goes back. */
            const std::vector<char> ok(P.copy.size(), good ? 1 : 0);
            {
                const auto lk = sched_.hold();
                lead.tierx.report_pass(P.copy, ok, now);
            }
            P.copy.clear();
        }
    }

    /* ---- the next copy ---------------------------------------------------------------------- */
    if (P.copy.empty() && lead.tiers.writeback_waiting()) {
        {
            const auto lk = sched_.hold();
            lead.tiers.plan_writeback(P.max_copy, &P.copy);
        }
        if (!P.copy.empty()) {
            P.copy_st.assign(ranks_.size(), 0);
            for (size_t i = 0; i < ranks_.size(); ++i) {
                const int st = ranks_[i]->tierx.issue_copy(P.copy, ranks_[i]->ctx.stream());
                if (st < 0) P.copy_st[i] = st;
            }
            if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
        }
    }

    /* ---- what is not live leaves VRAM -------------------------------------------------------- */
    /* Release an entry once its host copy has landed, if the expert slab can use the blocks; a
     * later turn restores it with a copy. Otherwise (no lending, or the loan already covers every
     * slab slot) it stays in VRAM, and a request needing room takes it through trim. Snapshots
     * are already in host memory. */
    if (lending_) {
        int64_t ceiling = 0;
        for (auto& r : ranks_) ceiling = std::max(ceiling, r->mover.loan_ceiling(Mover::kLendKV));
        const auto lk = sched_.hold();
        if (lead.kv.loan_tokens() < ceiling) (void)lead.prefix.drop_clean();
    }

    /* ---- capacity ---------------------------------------------------------------------------- */
    /* When the host arena refused copies, drop host-only entries worth less than them to make
     * room (IdleTiers::plan). Drops move no bytes, so the plan is reported at once. */
    if (lead.tiers.starved()) {
        std::vector<TierJob> jobs;
        lead.tiers.plan(P.slow_budget, &jobs);
        if (!jobs.empty()) {
            const std::vector<char> ok(jobs.size(), 1);
            const auto lk = sched_.hold();
            lead.tierx.report_pass(jobs, ok, now);
        }
    }
}

void Engine::expire_tick(uint64_t now_ns) {
    if (cfg_.prefix_cache_ttl <= 0 || now_ns - expire_last_ns_ < 1000000000ull) return;
    expire_last_ns_ = now_ns;
    PrefixCache& pc = ranks_[0]->prefix;
    if (!pc.enabled()) return;
    const auto lk = sched_.hold();
    (void)pc.expire(now_ns, (uint64_t)cfg_.prefix_cache_ttl * 1000000000ull);
}

bool Engine::flush_copies() {
    if (!pump_ || !ranks_[0]->tierx.enabled()) return false;
    TierPump& P = *pump_;
    Rank& lead = *ranks_[0];
    bool landed = false;
    const uint64_t t0 = rad_mono_ns();
    /* Counted only when something was in flight or waiting. */
    struct Note {
        IdleTiers& t; uint64_t t0; bool worked = false;
        ~Note() { if (worked) t.note_flush((int64_t)(rad_mono_ns() - t0)); }
    } note{lead.tiers, t0};
    /* Bounded: each round lands one batch of P.max_copy entries; a larger shortfall cannot be
     * relieved in one step. */
    for (int round = 0; round < 64; ++round) {
        if (!P.copy.empty()) {
            note.worked = true;
            bool good = true;
            for (size_t i = 0; i < ranks_.size(); ++i) {
                if (P.copy_st[i] == 0) P.copy_st[i] = ranks_[i]->tierx.wait_copy();
                if (P.copy_st[i] < 0) good = false;
            }
            if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
            const std::vector<char> ok(P.copy.size(), good ? 1 : 0);
            lead.tierx.report_pass(P.copy, ok, rad_mono_ns());
            P.copy.clear();
            landed = landed || good;
        }
        if (!lead.tiers.writeback_waiting()) break;
        lead.tiers.plan_writeback(P.max_copy, &P.copy);
        if (P.copy.empty()) break;
        P.copy_st.assign(ranks_.size(), 0);
        for (size_t i = 0; i < ranks_.size(); ++i) {
            const int st = ranks_[i]->tierx.issue_copy(P.copy, ranks_[i]->ctx.stream());
            if (st < 0) P.copy_st[i] = st;
        }
        if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
    }
    return landed;
}

/* Drop the sampler's per-request state for requests that reached a terminal state. The server
 * drains their text through the sink and calls reap(), which releases the Request; only the HTTP
 * thread knows when it has stopped writing. */
void Engine::drain_completed() {
    /* Taken, not read: HTTP cancels append to the list, and nothing else consumes it, so each id
     * is seen exactly once. */
    sched_.take_completed(&done_);
    for (uint64_t id : done_) for (auto& rk : ranks_) rk->sampler.end_request(id);
}

namespace {
/* FNV-1a over the plan's tier/site per weight: two runs that resolved the same strategy and the
 * same budgets but keep a different set of experts resident are different deployments, and the
 * strategy name alone would not tell them apart. Index 0 is the sentinel. */
uint64_t plan_residency_digest(const Plan& p) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 1; i < p.weight.size(); ++i) {
        const uint8_t b[2] = { (uint8_t)p.weight[i].tier, (uint8_t)p.weight[i].site };
        for (uint8_t x : b) {
            h ^= x;
            h *= 1099511628211ull;
        }
    }
    return h;
}
}  /* namespace */

int Engine::draft_cost_runtime(DraftCostRuntime* out) const {
    if (!out) return RAD_E_INVAL;
    if (ranks_.empty()) {
        RAD_ERR("draft cost runtime: no rank has been planned, so this deployment has no "
                "identity a cost profile could be bound to");
        return RAD_E_STATE;
    }
    DraftCostFingerprint& fp = out->fingerprint;
    /* Every field is what this deployment is, not what a flag asked for: the profile is refused
     * on any difference (DraftCostTable::configure names the first). The gpus string is the
     * ranks' devices in order -- product name, arch and the equipment a same-named SKU can
     * differ in; kernels is the loaded libraries, the arch plugin and the device arch; the model
     * name falls back to the container's arch id where a container carries no model_name (the
     * checkpoint path already does the same). */
    fp.model   = meta_.name ? meta_.name : "";
    if (fp.model.empty() && meta_.arch_id) fp.model = meta_.arch_id;
    fp.quant   = meta_.quant && *meta_.quant ? meta_.quant : "bf16";
    fp.tp      = (int32_t)ranks_.size();
    /* The same string engine_info() reports, "n/a" at TP1 included, so a record is reproducible
     * from the info output. */
    fp.tp_wire = cfg_.tp > 1 ? (cfg_.tp_wire_exact ? "exact" : "wht6") : "n/a";
    std::vector<std::string> gpus;
    std::string arch;
    for (const auto& rk : ranks_) {
        RadDeviceProps p{};
        if (const int prc = rad_dev_props(rk->device, &p); prc != RAD_OK) {
            RAD_ERR("draft cost runtime: device %d: %s", rk->device, rad_dev_last_error());
            return prc;
        }
        /* The product name alone collides across SKUs and steppings; the arch id and the
         * equipment do not (rad_device.h names the name field "for logs and the tuning cache"). */
        std::string g = p.name;
        if (p.arch[0]) { g += " ("; g += p.arch; g += ")"; }
        g += fmt(" dev=%d cu=%d vram=%lld", p.device_id, p.n_cu, (long long)p.vram_bytes);
        gpus.push_back(std::move(g));
        if (arch.empty()) arch = p.arch[0] ? p.arch : "";
    }
    fp.gpus      = draft_cost_gpus(gpus);
    /* The arch plugin is part of the kernel identity: two plugins over the same kernel libraries
     * are still two different selectors (build/startup.h keeps the two lists apart). */
    fp.kernels   = plugins_.libraries();
    if (!plugins_.arch_plugin.empty()) {
        if (!fp.kernels.empty()) fp.kernels += ' ';
        fp.kernels += plugins_.arch_plugin + "@" +
                      (plugins_.arch_version.empty() ? "0" : plugins_.arch_version);
    }
    if (!arch.empty()) fp.kernels += " " + arch;
    fp.kv_format = cfg_.kv_cache_dtype;

    /* `placement` is the resolved plan, not the flag: `auto` resolves against a budget that
     * depends on the free VRAM at this start (place/planner.cpp, mem/vram_budget.cpp), so two
     * runs asking for `auto` can land on different strategies and different expert residency.
     * Each rank contributes its strategy, its resolved budgets and a digest of its residency. A
     * rank whose plan is still `auto` -- or has no weights -- makes the identity unknowable; the
     * profile is refused then rather than matched against a guess. */
    std::string placement;
    for (size_t i = 0; i < ranks_.size(); ++i) {
        const Plan& pl = ranks_[i]->plan;
        if (pl.strategy == Strategy::Auto || pl.weight.size() <= 1) {
            RAD_ERR("draft cost runtime: rank %zu has no resolved placement plan, so a cost "
                    "profile cannot be bound to it", i);
            return RAD_E_STATE;
        }
        if (!placement.empty()) placement += ';';
        placement += fmt("%s vram=%lld host=%lld file=%lld dyn=%d res=%016llx",
                         strategy_name(pl.strategy), (long long)pl.vram_budget,
                         (long long)pl.host_budget, (long long)pl.file_stage_reserve,
                         pl.dynamic ? 1 : 0, (unsigned long long)plan_residency_digest(pl));
    }
    fp.placement = std::move(placement);

    /* The backend's round model, not this cycle's distribution: the serial head runs one round per
     * row, so it is shrinking even when a cycle's K are uniform. A block drafter, no drafter, or
     * drafting switched off (n_spec == 0, already resolved from auto in declare()) has no rounds.
     * The distribution keeps its identity in the bucket key. */
    out->execution = draft_cost_execution(serial_drafting(), cfg_.n_spec);
    return RAD_OK;
}

int Engine::load_draft_cost_profile(const DraftCostRuntime& runtime) {
    if (cfg_.draft_profile.empty()) return RAD_OK;
    DraftCostProfile profile;
    std::string err;
    const int rc = draft_cost_profile_read(cfg_.draft_profile, &profile, &err);
    if (rc != RAD_OK) {
        RAD_ERR("--draft-profile '%s': %s", cfg_.draft_profile.c_str(),
                err.empty() ? rad_strerror(rc) : err.c_str());
        return rc;
    }

    RAD_TRY(sched_.draft_costs().configure(profile, runtime));

    if (!sched_.draft_costs().active())
        RAD_WARN("--draft-profile '%s' is not used for this deployment: %s",
                 cfg_.draft_profile.c_str(), sched_.draft_costs().reason().c_str());
    else
        RAD_INFO("--draft-profile '%s': %zu buckets match this deployment",
                 cfg_.draft_profile.c_str(), sched_.draft_costs().buckets().size());
    return RAD_OK;
}

/* --draft-profile-out: the calibration mode. Refused, not ignored, when it cannot produce an
 * honest profile: the two flags together would read and write the same identity, an existing path
 * would be replaced, and only the serial head can be recorded. All of them are checked here, at
 * startup, so a run that cannot export says so before serving instead of after.
 *
 * Both policies calibrate. Under legacy the run never verifies a window it did not just draft, so
 * the shape a cost deployment prices first -- a cycle whose paired step verified nothing, before
 * any plan published -- cannot arise. A cost run executes exactly that start cycle, and the table
 * is put in record-only mode for it: the run measures the fallback baseline it really runs and
 * prices nothing from the buckets it is still filling. */
int Engine::begin_draft_calibration() {
    if (cfg_.draft_profile_out.empty()) return RAD_OK;
    if (!cfg_.draft_profile.empty()) {
        RAD_ERR("--draft-profile-out with --draft-profile: a calibration run measures its own "
                "profile and does not read one; pass only one of the two");
        return RAD_E_INVAL;
    }
    const bool cost = cfg_.draft_policy == DraftPolicyKind::Cost;
    if (cfg_.kld()) {
        RAD_ERR("--draft-profile-out: the KL mode scores the trunk and drafts nothing, so it "
                "measures no draft cycle to export");
        return RAD_E_INVAL;
    }
    /* The file's drafter dimension can only name the serial head: the block drafter proposes a
     * whole block in one pass and the lookup drafter's rounds are the trunk's own, so neither has
     * a per-row K this format could record. A run without the head would export a profile that
     * misnames what produced it. */
    if (!serial_drafting() || cfg_.n_spec <= 0) {
        RAD_ERR("--draft-profile-out: needs a serial draft head and a positive ceiling (drafter "
                "kind %d, depth %d, ceiling %d), because the profile names its rounds per row",
                (int)drafter_.kind, (int)drafter_.depth, cfg_.n_spec);
        return RAD_E_INVAL;
    }
    /* The writer links into place as the real gate (EEXIST); this only turns the common case into
     * a startup error. Anything else -- a missing directory, no permission -- is not "free". */
    struct stat st;
    if (::stat(cfg_.draft_profile_out.c_str(), &st) == 0) {
        RAD_ERR("--draft-profile-out '%s' already exists; a calibration run does not overwrite a "
                "profile -- move it or choose another path", cfg_.draft_profile_out.c_str());
        return RAD_E_INVAL;
    }
    if (errno != ENOENT) {
        RAD_ERR("--draft-profile-out '%s': %s", cfg_.draft_profile_out.c_str(),
                std::strerror(errno));
        return RAD_E_IO;
    }
    /* The identity is derived here, before the scheduler starts: an incomplete one must fail
     * before the run, not at export, and the profile is then written with the fingerprint this run
     * was bound to. The table is the recording's target with no buckets of its own: under legacy
     * nothing can be priced from it, and a cost calibration admits the fallback baseline it
     * measures and never prices at all (record_only). */
    RAD_TRY(draft_cost_runtime(&draft_calib_runtime_));
    DraftCostProfile empty;
    empty.fingerprint = draft_calib_runtime_.fingerprint;
    empty.execution   = draft_calib_runtime_.execution;
    RAD_TRY(sched_.draft_costs().configure(empty, draft_calib_runtime_));
    if (!sched_.draft_costs().active()) {
        RAD_ERR("--draft-profile-out: the recording table refused this runtime: %s",
                sched_.draft_costs().reason().c_str());
        return RAD_E_STATE;
    }
    if (cost) {
        /* The run executes the named baseline until a cycle is measured, and it is that baseline
         * -- including the start cycle nothing precedes -- that the file records. */
        sched_.draft_costs().admit_fallback();
        sched_.draft_costs().record_only();
    }
    draft_calibrating_ = true;
    return RAD_OK;
}

int Engine::export_draft_cost_profile() {
    if (!draft_calibrating_) return RAD_OK;
    /* Everything the step loop measured has to reach the table before the table is read. */
    sched_.flush_draft_observations();

    DraftCostProfile p;
    p.fingerprint = draft_calib_runtime_.fingerprint;
    p.execution   = draft_calib_runtime_.execution;
    /* The fingerprint names the plan resolved at start. A bucket carrying kDraftSamplePlacement
     * was measured while the mover was moving a unit (Scheduler::step_finished takes that from the
     * rank's own move evidence), so its time is the cost of a residency this header does not
     * name. The file has no field that could name the drifted residency, so those observations are
     * left out -- only the affected ones: a clean cycle before the first move still measured the
     * start plan and stays. */
    int32_t placement = 0;
    int32_t usable = 0;
    for (const DraftCostBucket& b : sched_.draft_costs().buckets()) {
        if (b.flags & kDraftSamplePlacement) { ++placement; continue; }
        if (draft_bucket_usable(b)) ++usable;
        p.buckets.push_back(b);
    }
    /* A run that measured no cycle (no request served, no serial round executed) -- or one whose
     * every cycle was placement-affected -- must not leave a file a reader would take as a
     * baseline: the writer refuses the empty one too, but the reason belongs in the log rather than
     * in a generic parse error. */
    if (p.buckets.empty()) {
        if (placement > 0)
            RAD_ERR("--draft-profile-out '%s': all %d measured bucket(s) were placement-affected, "
                    "so none describes the start residency the fingerprint names; a calibration "
                    "needs cycles the mover did not disturb", cfg_.draft_profile_out.c_str(),
                    (int)placement);
        else
            RAD_ERR("--draft-profile-out '%s': no bucket was measured; the run recorded no draft "
                    "cycle, which is not a calibration", cfg_.draft_profile_out.c_str());
        return RAD_E_INVAL;
    }
    /* Every remaining bucket is a diagnostic the table never prices from -- a shortened run, an
     * unpriced truncation, a warmup/record/audit/admission bracket. Writing those would report a
     * calibration whose file can price nothing, so the export fails here instead. */
    if (usable == 0) {
        RAD_ERR("--draft-profile-out '%s': none of the %d kept bucket(s) is priceable (%d "
                "placement-affected, the rest diagnostic only); a file of diagnostics is not a "
                "calibration", cfg_.draft_profile_out.c_str(), (int)p.buckets.size(),
                (int)placement);
        return RAD_E_INVAL;
    }

    int32_t skipped = 0;
    std::string err;
    const int rc = draft_cost_profile_write_file(cfg_.draft_profile_out, p, &skipped, &err);
    if (rc != RAD_OK) {
        RAD_ERR("--draft-profile-out '%s': %s", cfg_.draft_profile_out.c_str(),
                err.empty() ? rad_strerror(rc) : err.c_str());
        return rc;
    }
    if (skipped > 0)
        RAD_WARN("--draft-profile-out: %d diagnostic bucket(s) left out (the file has no name for "
                 "their flag)", skipped);
    if (placement > 0)
        RAD_WARN("--draft-profile-out: %d placement-affected bucket(s) left out (a mover change "
                 "inside the bracket; the fingerprint names the plan, not the drifted residency)",
                 (int)placement);
    RAD_INFO("--draft-profile-out '%s': %zu bucket(s), fingerprint %s, execution %s",
             cfg_.draft_profile_out.c_str(), p.buckets.size(),
             draft_calib_runtime_.fingerprint.model.c_str(),
             draft_calib_runtime_.execution == DraftCostExecution::Shrinking ? "shrinking"
                                                                            : "rectangular");
    return RAD_OK;
}

int Engine::serve(std::unique_ptr<StallWatch> watch) {
    startup_watch_ = std::move(watch);
    if (ranks_.empty() || !ranks_[0]->ctx.program()) {
        RAD_ERR("serve: start() has not run, or stopped at --debug-graph");
        return RAD_E_STATE;
    }
    embedding_ = ranks_[0]->program.pooled_buf != RAD_NULL_HANDLE;
    const bool audio_gen = ranks_[0]->program.is_audio_generator();
    const bool video_gen = ranks_[0]->program.is_video_generator();
    const bool generator = ranks_[0]->program.is_pass_driven();
    /* The profile flags configure speculative decoding's cost table. A generator or an embedding
     * model never initialises the scheduler that owns it, so both flags would be read by nobody.
     * Refused here, where every path still returns before serving: silently ignoring a flag that
     * asks for a calibration file or a cost baseline is worse than not starting. */
    if ((generator || embedding_) &&
        (!cfg_.draft_profile.empty() || !cfg_.draft_profile_out.empty())) {
        RAD_ERR("--draft-profile/--draft-profile-out configure speculative decoding, but the '%s' "
                "plugin makes %s: the scheduler is never initialised there, so the flag would be "
                "ignored", plugins_.arch_plugin.c_str(),
                generator ? "a generator" : "an embedding model");
        return RAD_E_UNSUPPORTED;
    }
    if (ranks_[0]->program.logits_buf == RAD_NULL_HANDLE && !embedding_ && !generator) {
        RAD_ERR("the '%s' plugin declared no logits buffer (rad_declare_logits), no pooled output "
                "(rad_declare_pooled) and no generator (rad_declare_generator, "
                "rad_declare_audio_generator, rad_declare_video_generator), so a step produces "
                "nothing anyone can read. An encoder-only plugin cannot serve.",
                plugins_.arch_plugin.c_str());
        return RAD_E_UNSUPPORTED;
    }
    if (generator) {
        const char* const what = audio_gen ? "audio" : video_gen ? "video" : "images";
        if (cfg_.kld() || cfg_.n_spec > 0) {
            RAD_ERR("the '%s' plugin makes %s: --kld and speculative decoding have no tokens to "
                    "work on", plugins_.arch_plugin.c_str(), what);
            return RAD_E_UNSUPPORTED;
        }
        for (size_t i = 1; i < ranks_.size(); ++i)
            if (ranks_[i]->program.is_audio_generator() != audio_gen ||
                ranks_[i]->program.is_video_generator() != video_gen ||
                !ranks_[i]->program.is_pass_driven()) {
                RAD_ERR("rank %zu declared a different generator from rank 0's", i);
                return RAD_E_STATE;
            }
        cfg_.prefix_cache = false;
        prefix_off_why_ = audio_gen
            ? "an audio generator: a prompt is read by its own request's frames and by nothing "
              "after them"
            : video_gen
            ? "a video generator: a prompt is read by its own request's steps and keeps no KV"
            : "an image generator: a conditioning is read by its own request's images and by "
              "nothing after them";
        decide("--no-prefix-cache", "prefix caching off", prefix_off_why_);
    }
    if (embedding_) {
        if (cfg_.kld() || cfg_.n_spec > 0) {
            RAD_ERR("the '%s' plugin is an embedding model: it generates nothing, so --kld and "
                    "speculative decoding have nothing to work on", plugins_.arch_plugin.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const Program& p0 = ranks_[0]->program;
        pooled_dim_ = p0.buffers[p0.pooled_buf].decl.shape[1];
        for (size_t i = 1; i < ranks_.size(); ++i) {
            const Program& pi = ranks_[i]->program;
            if (pi.pooled_buf == RAD_NULL_HANDLE ||
                pi.buffers[pi.pooled_buf].decl.shape[1] != pooled_dim_) {
                RAD_ERR("rank %zu declared a different pooled output from rank 0's", i);
                return RAD_E_STATE;
            }
        }
        pooled_host_ = static_cast<float*>(
            rad_dev_alloc(cfg_.max_seqs * pooled_dim_ * (int64_t)sizeof(float),
                          RAD_MEM_HOST_PINNED));
        if (!pooled_host_) {
            RAD_ERR("no pinned host memory for a step's pooled rows (%lld x %lld f32)",
                    (long long)cfg_.max_seqs, (long long)pooled_dim_);
            return RAD_E_NOMEM;
        }
        /* No prefix cache: an encoder keeps no KV, and with bidirectional attention a shared
         * prefix does not compute to the same rows. */
        cfg_.prefix_cache = false;
        prefix_off_why_ = "an embedding model: an encoder keeps no KV to reuse, and attending both "
                          "ways means a shared prefix does not compute to the same rows";
        decide("--no-prefix-cache", "prefix caching off", prefix_off_why_);
        RAD_INFO("embedding model: %lld-wide pooled vectors, one step an input, up to %lld "
                 "tokens an input", (long long)pooled_dim_, (long long)cfg_.max_tok);
    }

    /* The scheduler drives one block manager, rank 0's. Block ids are identical on every rank:
     * all carve the same block count (Engine::configure_kv caps each at the smallest rank's), even
     * where a group's share differs in bytes or is zero (rad_kv_group_zero). Only the pool base
     * differs per rank (KVPoolBinding). Check the counts, since a mismatch would make one block
     * table mean different things on different cards. */
    for (size_t i = 1; i < ranks_.size(); ++i) {
        const auto& a = ranks_[0]->kv.plans();
        const auto& c = ranks_[i]->kv.plans();
        if (a.size() != c.size()) {
            RAD_ERR("rank %zu carved %zu KV groups and rank 0 carved %zu", i, c.size(), a.size());
            return RAD_E_STATE;
        }
        for (size_t g = 0; g < a.size(); ++g) {
            if (a[g].n_blocks == c[g].n_blocks && a[g].n_states == c[g].n_states) continue;
            RAD_ERR("KV group '%s': rank 0 carved %lld blocks / %lld states, rank %zu carved "
                    "%lld / %lld. One block table serves every rank, so they must agree.",
                    a[g].name.c_str(), (long long)a[g].n_blocks, (long long)a[g].n_states,
                    i, (long long)c[g].n_blocks, (long long)c[g].n_states);
            return RAD_E_STATE;
        }
    }

    /* Back to rank 0's device before building the scheduler: the loops above left this thread on
     * the last rank's device, and the scheduler's staging buffers and event must live on rank 0,
     * whose stream it is given (spec §1). */
    RAD_TRY(rad_dev_set(ranks_[0]->device));

    kv_iface_ = std::make_unique<KVBinding<KVManager>>(&ranks_[0]->kv);
    pc_iface_ = std::make_unique<PCBinding<PrefixCache>>(&ranks_[0]->prefix);
    /* The control vector's identity partitions the block cache (PCBinding::salt_id): a request
     * that runs with the projection on never adopts a block or a linear-state snapshot produced
     * without it, or the other way round. Zero with no vector loaded, and then every chain hashes
     * exactly as it did before the feature existed. */
    static_cast<PCBinding<PrefixCache>*>(pc_iface_.get())->salt_id = cvec_salt_;
    /* The restore path, installed only when the tiers are on. The scheduler calls it before the
     * admission lookup without knowing about tiers, so it is a callable rather than an include in
     * core/sched.
     *
     * Every rank restores its own shard. Rank 0's TierExec holds the policy: it allocates the
     * blocks and reports the job, and the others follow into the same block ids. A failure only
     * shortens the prefix, so no error is returned. */
    if (ranks_[0]->tierx.enabled()) {
        auto* b = static_cast<PCBinding<PrefixCache>*>(pc_iface_.get());
        /* Called inside Scheduler::step() under its lock, which makes touching the cache and rank
         * 0's manager safe; nothing here may call back into the scheduler. */
        b->restore_fn = [this](const Request& req) {
            const uint64_t t0 = rad_mono_ns();
            PrefixCache& pc = ranks_[0]->prefix;
            std::vector<MMContentKey> mk;
            media_keys(req, &mk, cvec_salt_for(req.sp.use_control_vector));
            const std::vector<BlockHash> want =
                pc.restorable_from(req.prompt, mk);
            if (want.empty()) return;
            /* The whole chain in one restore. With 4-token blocks a long conversation is tens of
             * thousands of entries; batched, it is one launch per (group, layer) per staging batch
             * and one copy per run of arena slots instead of a round trip per entry. */
            std::vector<TierJob> jobs;
            jobs.reserve(want.size());
            for (const BlockHash& h : want) {
                TierJob j;
                if (!ranks_[0]->tierx.plan_promote(h, &j)) break;
                jobs.push_back(std::move(j));
            }
            if (jobs.empty()) return;
            std::vector<char> ok(jobs.size(), 1);
            for (auto& r : ranks_) {
                std::vector<char> mine;
                (void)r->tierx.issue_restore(jobs, r->ctx.stream(), &mine);
                for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
            }
            /* Back to rank 0's device: the admission continues by growing rank 0's pool. */
            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
            ranks_[0]->tierx.report_pass(jobs, ok, rad_mono_ns());
            /* Rebind every entry that came back, not only the leading run: the tiers already
             * marked each resident at its new blocks. */
            const std::vector<int32_t>& cg = pc.cached_groups();
            for (size_t i = 0; i < jobs.size(); ++i) {
                TierJob& j = jobs[i];
                if (!ok[i]) continue;
                const int rb = pc.rebind(j.key, j.blocks);
                if (rb != RAD_OK) {
                    /* No entry to land in: release the reserved blocks and forget the tier record,
                     * which would otherwise release them later from under their next owner. */
                    ranks_[0]->tierx.quiesce();
                    for (size_t k = 0; k < cg.size() && k < j.blocks.size(); ++k)
                        if (j.blocks[k] >= 0) ranks_[0]->kv.release_block(cg[k], j.blocks[k]);
                    ranks_[0]->tiers.forget(j.key);
                }
            }
            ranks_[0]->tiers.note_restore((int64_t)(rad_mono_ns() - t0));
        };
        /* An eviction that would lose an entry waits for its copy instead. Called by the prefix
         * cache binding under the scheduler's lock when dropping already-copied entries did not
         * free enough: all waiting copies are landed now, so those entries can leave VRAM. */
        b->flush_fn = [this]() -> bool { return flush_copies(); };
    }
    /* Per-rank card, stream and program. The batch is built once and shared by pointer on the
     * host (spec §1), but its device side is staged per rank: reading another card's staging is a
     * PCIe read and unordered against the copies that fill it (see Set in batch.h). */
    std::vector<BatchBuilder::RankIO> rio;
    rio.reserve(ranks_.size());
    for (auto& rk : ranks_)
        rio.push_back({ rk->device, rk->ctx.stream(), &rk->program });
    /* Validated before any path starts serving: a calibration run that cannot export must say so
     * now, not after a complete run. */
    RAD_TRY(begin_draft_calibration());
    if (video_gen) {
        /* As for the image generator below; the plugin lays out each request's rows. */
        std::vector<gen::VideoDriver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::VideoDriver>();
        RAD_TRY(d->init(ranks_[0]->program, rad_tools_video_positions(plugins_), gio));
        vid_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else if (audio_gen) {
        /* As for the image generator below. */
        std::vector<gen::AudioDriver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::AudioDriver>();
        RAD_TRY(d->init(ranks_[0]->program, kv_iface_.get(), gio));
        aud_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else if (generator) {
        /* The generator's driver replaces the scheduler, which is never initialised, so its
         * token-step staging is never allocated. */
        std::vector<gen::Driver::RankIO> gio;
        for (auto& rk : ranks_) gio.push_back({ rk->device, rk->ctx.stream() });
        auto d = std::make_unique<gen::Driver>();
        RAD_TRY(d->init(ranks_[0]->program, kv_iface_.get(), gio));
        img_gen_ = d.get();
        gen_ = std::move(d);
        RAD_TRY(rad_dev_set(ranks_[0]->device));
    } else {
        /* The runtime is derived before the read: when the deployment's identity cannot be
         * established, a profile is refused at this one race-free point instead of being loaded
         * and left inactive with a warning. No profile, no derivation. */
        if (!cfg_.draft_profile.empty()) {
            DraftCostRuntime runtime;
            RAD_TRY(draft_cost_runtime(&runtime));
            RAD_TRY(load_draft_cost_profile(runtime));
        }
        /* A runtime that cannot record a pass runs every one of them live and unkeyed, which is a
         * fact about the deployment the cost view can carry. A recording runtime may play, record
         * or audit, and which one a pass does exists only once it runs -- after its plan is
         * issued -- so its cycles stay unpriced rather than declare a state they cannot know. */
        RAD_TRY(sched_.init(ranks_[0]->program, cfg_, kv_iface_.get(), pc_iface_.get(), rio, &geo_,
                            ranks_[0]->ctx.records_passes() ? DraftReplay::Unknown
                                                            : DraftReplay::None));
#ifdef RAD_TEST_HOOKS
        /* The one point where a submission is race-free: the scheduler holds its KV and limits,
         * and neither the rank threads nor the stepper exist yet. */
        rad_test_serving();
#endif
    }
    /* The smallest step the prefill stager overlaps. A chunk held back for other work
     * (--busy-prefill-chunk) stays at or above it; below it every routed layer reads its pooled
     * experts synchronously, which costs far more than sharing the step saves. The maximum over
     * ranks, since the step waits for the slowest. */
    {
        int64_t stage_rows = 0;
        for (auto& rk : ranks_)
            if (rk->stager.enabled()) stage_rows = std::max(stage_rows, rk->stager.min_rows());
        sched_.set_stage_rows(stage_rows);
        /* The shortest prompt a span pays for where routed experts are read from host memory: a
         * spanned step runs at the full plan, which lends the expert slab nothing, while every
         * other step runs at the span's shared level and lends it the span's extra rows. On
         * Qwen3.8-Flash-Next at TP2 (two R9700, 12 GiB host pool) the span loses to one-interval
         * steps up to 12K tokens and wins from 18K, hence eight intervals. With every expert on
         * the card there is no such cost and every lone prefill spans. */
        sched_.set_span_min(stage_rows > 0 && arena_span_rows_ > 0
                                ? 8 * std::max<int64_t>(geo_.checkpoint_interval, 0) : 0);
    }

    for (auto& r : ranks_) r->sampler.attach_vocab(vocab_view_.get());

    barrier_ = std::make_unique<RankBarrier>(cfg_.tp + 1);
    stopping_.store(false, std::memory_order_release);
    step_status_.store(RAD_OK, std::memory_order_release);
    loop_status_.store(RAD_OK, std::memory_order_release);
    done_.reserve((size_t)cfg_.max_seqs);

    rank_threads_.clear();
    for (int i = 0; i < cfg_.tp; ++i) rank_threads_.emplace_back([this, i] { rank_loop(i); });
    std::thread stepper([this] { if (gen_) gen_loop(); else step_loop(); });

    if (cfg_.kld()) {
        print_startup_report(RAD_OK);
        startup_watch_.reset();
    }
    int s = cfg_.kld() ? run_kld() : run_server();
    startup_watch_.reset();

    /* Stop the step loop before the server's storage is destroyed: Request::sink points into it,
     * and an unfinished request still writes there (server.h). */
    stopping_.store(true, std::memory_order_release);
    barrier_->abort();
    if (stepper.joinable()) stepper.join();
    for (auto& t : rank_threads_) if (t.joinable()) t.join();
    rank_threads_.clear();
    /* A calibration export is part of the run's result, not a courtesy: its failure is the
     * engine's status. A run the step loop aborted measured nothing a profile could stand on, so
     * it writes no file. The step-loop error still wins when both are non-zero. */
    const int ls = loop_status_.load(std::memory_order_acquire);
    const int xrc = ls < 0 ? RAD_OK : export_draft_cost_profile();
    if (ls < 0) return ls;
    if (xrc != RAD_OK) return xrc;
    return s;
}

/* The state slots a sequence owns, one per KV group, -1 where the group is not stateful. Read from
 * rank 0 and applied to every rank: only rank 0's KVManager tracks sequences (kv_iface_), and
 * every rank's pool has identical geometry, so a slot index means the same offset everywhere. */
bool Engine::state_slots_of(uint64_t seq, std::vector<int32_t>* out) const {
    const KVManager& kv = ranks_[0]->kv;
    out->assign((size_t)kv.n_groups(), -1);
    bool any = false;
    for (int32_t g = 0; g < kv.n_groups(); ++g) {
        const KVGroupPlan* p = kv.plan(g);
        if (!p || !p->stateful()) continue;
        const int32_t s = kv.state_slot(seq, g);
        if (s < 0) return false;      /* the sequence is gone, or holds no state to copy */
        (*out)[(size_t)g] = s;
        any = true;
    }
    /* A window snapshot is a snapshot with no state slots (KVManager::has_window_snapshot). */
    return any || (kv.has_window_snapshot() && kv.has_sequence(seq));
}

/* ------------------------------------------------------------------ the HTTP frontend */
/* The control vector, for /stats and /server_info. `available` is
 * what a client should test before sending the field: without a loaded vector a request that asks
 * for it is refused, so a UI can hide the control rather than offer a switch that fails. */
std::string Engine::cvec_json() const {
    nlohmann::ordered_json o;
    o["available"] = cvec_.loaded();
    o["default"]   = cvec_default_.load();
    if (cvec_.loaded()) {
        /* The family the file was validated against, not a label the UI supplies: this is the same
         * `meta_.arch_id` the loader compared against the file's own model_hint. */
        o["arch"]    = meta_.arch_id ? meta_.arch_id : "";
        o["first"]   = cvec_.first;
        o["last"]    = cvec_.last;
        o["n_embd"]  = cvec_.n_embd;
        o["n_layers"] = cvec_.n_layers;
        o["scale"]   = cvec_.user_scale;
        o["file"]    = cvec_.path;
        o["mode"]    = "project";
    }
    return o.dump();
}

/* The engine's part of /server_info: model, cards, KV, drafting and wire, for a running process.
 * The server holds only interfaces, so the engine renders this object and the server splices it in
 * verbatim. */
std::string Engine::engine_info() const {
    nlohmann::ordered_json e;
    e["arch"]        = meta_.arch_id ? meta_.arch_id : "";
    e["quant"]       = meta_.quant ? meta_.quant : "";
    e["n_layers"]    = meta_.n_layers;
    e["n_embd"]      = meta_.n_embd;
    e["n_vocab"]     = meta_.n_vocab;
    e["n_ctx_train"] = meta_.n_ctx_train;
    e["container"]   = cfg_.model;

    e["tensor_parallel"]        = cfg_.tp;
    e["tp_wire"]                = cfg_.tp > 1 ? (cfg_.tp_wire_exact ? "exact" : "wht6") : "n/a";
    e["tp_wire_min_kb"]         = cfg_.tp_wire_exact ? 0 : cfg_.tp_wire_min_kb;
    e["max_num_batched_tokens"] = cfg_.max_tok;
    e["placement"]              = cfg_.placement;
    /* Each lookup table's requested and resolved placement. */
    e["embedding_placement"]    = { { "asked", table_placement_name(cfg_.embedding_placement) },
                                    { "resolved", table_placement_name(cfg_.embedding_resolved) } };
    e["ngram_placement"]        = { { "asked", table_placement_name(cfg_.ngram_placement) },
                                    { "resolved", table_placement_name(cfg_.ngram_resolved) } };
    e["vision_tower_placement"] = { { "asked", table_placement_name(cfg_.tower_placement) },
                                    { "resolved", table_placement_name(cfg_.tower_resolved) } };
    e["prefix_cache"]           = cfg_.prefix_cache;
    e["control_vector"]         = nlohmann::ordered_json::parse(cvec_json());
    e["checkpoint_slots"]       = cfg_.checkpoint_slots;
    e["deterministic"]          = cfg_.deterministic;

    nlohmann::ordered_json devs = nlohmann::ordered_json::array();
    for (const auto& r : ranks_) {
        RadDeviceProps p{};
        if (rad_dev_props(r->device, &p) < 0) continue;
        nlohmann::ordered_json d;
        d["rank"]       = r->index;
        d["device"]     = r->device;
        d["name"]       = p.name;
        d["arch"]       = p.arch;
        d["vram_bytes"] = p.vram_bytes;
        d["vram_free"]  = p.vram_free;
        devs.push_back(std::move(d));
    }
    e["devices"] = std::move(devs);

    /* Rank 0's; every rank declares the same graph (see the miss report in
     * engine_bringup.cpp). */
    const Program& p0 = ranks_[0]->program;
    e["ops"]     = (int64_t)p0.ops.size() - 1;
    e["weights"] = (int64_t)p0.weights.size() - 1;

    nlohmann::ordered_json kvs = nlohmann::ordered_json::array();
    const KVManager& kv = ranks_[0]->kv;
    for (int32_t g = 0; g < kv.n_groups(); ++g) {
        const KVGroupPlan* pl = kv.plan(g);
        /* Skip group 0, the sentinel that makes handle 0 mean "none". */
        if (!pl || pl->name.empty() || pl->n_layers <= 0) continue;
        nlohmann::ordered_json k;
        k["name"]     = pl->name;
        k["kind"]     = bld_kv_kind_name(pl->kind);
        k["n_layers"] = pl->n_layers;
        k["bytes"]    = pl->pool_bytes;
        if (pl->paged()) {
            k["block_size"] = pl->block_size;
            k["blocks"]     = pl->n_blocks;
            k["tokens"]     = pl->n_blocks * pl->block_size;
        } else {
            k["states"] = pl->n_seq_states();
        }
        kvs.push_back(std::move(k));
    }
    e["kv_groups"] = std::move(kvs);

    nlohmann::ordered_json sp;
    if (drafter_.kind == RAD_DRAFT_NONE) {
        sp["enabled"] = false;
    } else {
        sp["enabled"] = true;
        sp["drafter"] = drafter_name_;
        sp["kind"]    = drafter_.kind == RAD_DRAFT_BLOCK ? "block" : "serial";
        sp["depth"]   = drafter_.depth;
    }
    e["speculation"] = std::move(sp);
    return e.dump();
}

/* ------------------------------------------------------------------ a clean stop
 *
 * SIGINT and SIGTERM are handled so `~Engine` runs: --profile-ops prints its table, the placement
 * summary prints, and in-flight requests drain instead of being cut.
 *
 * The handler only stores to a lock-free atomic; a watcher thread does the work, because
 * `Server::stop` takes locks and joins threads, which is not async-signal-safe. A second signal
 * gets the default disposition, so the operator can still force an immediate kill.
 */
namespace {
std::atomic<bool> g_signalled{false};
void on_signal(int) { g_signalled.store(true, std::memory_order_relaxed); }

/* ================================================================== sampling defaults */
/* The sampler for a request that sends no sampler settings. Parsing and range checks live in
 * core/gen_config.h; this is the precedence order and the container lookup:
 *
 *   --temp / --top-k / --top-p / --min-p    an operator overriding, field by field, and last
 *   --generation-config PATH                a generation_config.json named outright
 *   <model>.generation.json                 one placed beside the container
 *   the model's `generation.*` metadata     generation_config.json, as the container or the
 *                                          checkpoint carries it
 *   SamplingParams' own defaults            the untruncated floor
 *
 * The flags override field by field. Of the file sources, the first that states anything is used
 * whole, never blended with a lower one.
 *
 * `stated` and `source`, when given, receive what the source in force read and its name ("" for
 * none), which gen_sampling_sources turns into a reason per field. */
SamplingParams resolve_sampling_defaults(const RadModelMeta& meta, bool container,
                                         const Config& cfg, GenSampling* stated = nullptr,
                                         std::string* source = nullptr) {
    GenSampling g;
    std::string src;

    if (!cfg.generation_config.empty() &&
        gen_sampling_from_file(cfg.generation_config, /*required=*/true, &g))
        src = cfg.generation_config;

    if (!g.any()) {
        const std::string side = gen_sidecar_path(cfg.model);
        if (gen_sampling_from_file(side, /*required=*/false, &g)) src = side;
    }
    if (!g.any()) {
        auto get = [&meta](const char* field) {
            return rad_meta_gets(&meta, (std::string("generation.") + field).c_str(), nullptr);
        };
        if (gen_sampling_from_kv(get, &g)) src = container ? "the container" : "the checkpoint";
    }

    SamplingParams sp = g.onto(SamplingParams{});

    std::string over;
    if (cfg.sample_temp  >= 0.0f) { sp.temp  = cfg.sample_temp;  over += " --temp"; }
    if (cfg.sample_top_p >= 0.0f) { sp.top_p = cfg.sample_top_p; over += " --top-p"; }
    if (cfg.sample_min_p >= 0.0f) { sp.min_p = cfg.sample_min_p; over += " --min-p"; }
    if (cfg.sample_top_k >= 0)    { sp.top_k = cfg.sample_top_k; over += " --top-k"; }

    /* The remaining fields have no file source (generation_config.json supplies only the four
     * above): SamplingParams' defaults unless a flag sets them; request_overrides() logs those. */
    if (cfg.sample_typical_p)          sp.typical_p          = *cfg.sample_typical_p;
    if (cfg.sample_rep_penalty)        sp.rep_penalty        = *cfg.sample_rep_penalty;
    if (cfg.sample_pres_penalty)       sp.pres_penalty       = *cfg.sample_pres_penalty;
    if (cfg.sample_freq_penalty)       sp.freq_penalty       = *cfg.sample_freq_penalty;
    if (cfg.sample_penalty_last_n)     sp.penalty_last_n     = *cfg.sample_penalty_last_n;
    if (cfg.sample_dry_multiplier)     sp.dry_multiplier     = *cfg.sample_dry_multiplier;
    if (cfg.sample_dry_base)           sp.dry_base           = *cfg.sample_dry_base;
    if (cfg.sample_dry_allowed_length) sp.dry_allowed_length = *cfg.sample_dry_allowed_length;
    if (cfg.sample_dry_penalty_last_n) sp.dry_penalty_last_n = *cfg.sample_dry_penalty_last_n;
    if (cfg.sample_dry_seq_breakers)   sp.dry_seq_breakers   = *cfg.sample_dry_seq_breakers;
    if (cfg.sample_xtc_probability)    sp.xtc_probability    = *cfg.sample_xtc_probability;
    if (cfg.sample_xtc_threshold)      sp.xtc_threshold      = *cfg.sample_xtc_threshold;

    const std::string from = src.empty() ? (over.empty() ? "nothing stated it" : "the command line")
                                         : src;
    if (stated) *stated = g;
    if (source) *source = src;
    RAD_INFO("sampling defaults for requests that send none: %s (from %s%s%s)",
             GenSampling::describe(sp).c_str(), from.c_str(),
             over.empty() || src.empty() ? "" : ", overridden by", over.c_str());
    if (src.empty() && over.empty())
        RAD_WARN("this checkpoint states no sampling of its own and none was given, so a request "
                 "that sends no sampler fields is served with the whole vocabulary live at "
                 "temperature 1 -- which ends a reply early often enough to notice. Pass "
                 "--generation-config, or put the checkpoint's generation_config.json beside the "
                 "container as %s.", gen_sidecar_path(cfg.model).c_str());
    return sp;
}

/* "presence_penalty 1.5, max_tokens auto, ...": every request default or bound the command line
 * changed, in the request's field names; "" if none. Excludes the four fields
 * resolve_sampling_defaults reports. */
std::string request_overrides(const Config& c) {
    std::string out;
    auto add = [&out](const std::string& s) { out += (out.empty() ? "" : ", ") + s; };
    char b[64];
    auto f = [&](const char* k, const std::optional<float>& v) {
        if (v) { snprintf(b, sizeof b, "%s %g", k, (double)*v); add(b); }
    };
    auto i = [&](const char* k, const std::optional<int>& v) {
        if (v) add(std::string(k) + " " + std::to_string(*v));
    };
    f("typical_p", c.sample_typical_p);
    f("presence_penalty", c.sample_pres_penalty);
    f("frequency_penalty", c.sample_freq_penalty);
    f("repetition_penalty", c.sample_rep_penalty);
    i("repeat_last_n", c.sample_penalty_last_n);
    f("dry_multiplier", c.sample_dry_multiplier);
    f("dry_base", c.sample_dry_base);
    i("dry_allowed_length", c.sample_dry_allowed_length);
    i("dry_penalty_last_n", c.sample_dry_penalty_last_n);
    if (c.sample_dry_seq_breakers)
        add("dry_sequence_breakers " + std::to_string(c.sample_dry_seq_breakers->size()) +
            " strings");
    f("xtc_probability", c.sample_xtc_probability);
    f("xtc_threshold", c.sample_xtc_threshold);
    if (c.default_max_tokens == 0) add("max_tokens auto");
    else if (c.default_max_tokens > 0) add("max_tokens " + std::to_string(c.default_max_tokens));
    if (c.max_tokens_cap > 0) add("max_tokens capped at " + std::to_string(c.max_tokens_cap));
    if (c.max_n > 0) add("n up to " + std::to_string(c.max_n));
    if (c.max_stop_strings != 64 || c.max_stop_bytes != 4096)
        add("stop up to " + std::to_string(c.max_stop_strings) + " strings of " +
            std::to_string(c.max_stop_bytes) + " bytes");
    for (const auto& [k, v] : c.chat_template_kwargs) add("template " + k + "=" + v);
    if (c.reasoning_format != "auto") add("reasoning_format " + c.reasoning_format);
    if (c.thinking_token_budget >= 0)
        add("thinking_token_budget " + std::to_string(c.thinking_token_budget));
    return out;
}
}  /* namespace */
/* The HTTP server over the engine. `/v1/completions`, `/v1/chat/completions`, `/v1/models`,
 * `/metrics` and `/health` are live. `/v1/embeddings` answers 501 unless the program is an
 * embedding program, and so do image, video and audio content parts unless the program declared an
 * encoder that accepts them (MEDIA, below). The chat endpoint answers 501 only when the model has
 * no chat template: a guessed template is a guessed prompt format. With one, the template's
 * grammar and lazy triggers reach the sampler and its derived reply parser comes back through
 * IChatTemplate::apply. */
int Engine::run_server() {
    /* A stop that arrived between the previous serve and this one is this run's to act on: the
     * watcher below consumes g_signalled at the moment it stops the transport, and nothing here
     * clears it. Clearing at entry would swallow a signal raised after the previous run's stop
     * and leave this listener with no consumer. */
    /* The rank-0 sampler lets admission refuse, by name, a request whose sampling stage has no
     * kernel (IScheduler::admit). The tiers are passed even when they move nothing, because they
     * hold the session table the sessions view reads. */
    server::SchedulerBridge sched_bridge(&sched_, ranks_.empty() ? nullptr : &ranks_[0]->sampler,
                                        ranks_.empty() ? nullptr : &ranks_[0]->tiers);
    server::TokenizerBridge tok_bridge(vocab_);
    server::ChatTemplateBridge chat_bridge;
    server::GrammarBridge grammar_bridge(vocab_view_.get());

    /* One snapshot function feeds both the live view (spec §16) and the web dashboard, so they
     * cannot disagree. It runs on the drawing thread and must not slow the engine: it reads the
     * scheduler's published copy (Scheduler::publish), the driver's VRAM counter, and rank 0's
     * plan, immutable after start().
     *
     * The cards' fixed facts are read once here, before serving: rad_dev_props enters the device
     * runtime (properties, device switch, free-memory query, peer walk) and also opens the counter
     * dev_vram_used reads. */
    std::vector<server::CardStat> card_facts;
    for (const auto& r : ranks_) {
        RadDeviceProps pr{};
        if (rad_dev_props(r->device, &pr) < 0) continue;
        server::CardStat c;
        c.index      = r->device;
        c.name       = pr.name;
        c.vram_total = pr.vram_bytes;
        card_facts.push_back(std::move(c));
    }
    server::LiveView::Source snap = [this, &sched_bridge, card_facts](server::LiveSnapshot& s) {
        server::snapshot_from(sched_bridge.metrics(), &s);
        s.model = meta_.name && *meta_.name ? meta_.name : "radiance";

        s.cards.clear();
        for (const server::CardStat& f : card_facts) {
            /* vram_used is sampled per call. A card whose counter cannot be read is left out.
             * Utilisation, power and temperature stay 0 ("not sampled"): the device layer does
             * not expose them. */
            int64_t used = 0;
            if (dev_vram_used(f.index, &used) != RAD_OK) continue;
            server::CardStat c = f;
            c.vram_used = used;
            s.cards.push_back(std::move(c));
        }

        /* The expert plane from rank 0's placement plan; empty for a dense model, and the
         * renderers then omit the panel. */
        s.experts = server::ExpertPlane{};
        if (!ranks_.empty()) {
            const Plan& p = ranks_[0]->plan;
            /* Read once for both the expert and link halves of the panel, so they agree. */
            const MoverStats ms = ranks_[0]->mover.stats();
            int32_t n_layer = 0, n_expert = 0;
            for (const MoveUnit& mu : p.units)
                if (mu.expert >= 0) {
                    n_layer  = std::max(n_layer,  mu.layer + 1);
                    n_expert = std::max(n_expert, mu.expert + 1);
                }
            if (n_layer > 0 && n_expert > 0) {
                s.experts.n_layers  = n_layer;
                s.experts.n_experts = n_expert;
                s.experts.tier.assign((size_t)n_layer * (size_t)n_expert, (uint8_t)Tier::VRAM);
                /* The live tier when the heat engine runs, else the plan's (where a unit started).
                 * HeatEngine::tier is a plain vector written by the dispatch thread at step
                 * boundaries, so a racing read sees either side of a single store. */
                const HeatEngine& heat = ranks_[0]->heat;
                const bool live_tier = heat.enabled();
                s.experts.heat_enabled = live_tier;
                s.experts.layer_units.assign((size_t)n_layer, 0);
                s.experts.layer_resident.assign((size_t)n_layer, 0);
                for (size_t u = 0; u < p.units.size(); ++u) {
                    const MoveUnit& mu = p.units[u];
                    if (mu.expert < 0 || mu.layer < 0) continue;
                    const Tier t = live_tier ? heat.tier((int32_t)u) : mu.tier;
                    s.experts.tier[(size_t)mu.layer * (size_t)n_expert + (size_t)mu.expert]
                        = (uint8_t)t;
                    /* Count residency over units, not plane cells: the plane is a rectangle a
                     * model need not fill (a draft head's expert count can differ), and empty
                     * cells default to VRAM. */
                    const int ti = (int)t >= 0 && (int)t < kTierCount ? (int)t : (int)Tier::SSD;
                    s.experts.units[ti]++;
                    s.experts.bytes[ti] += mu.bytes;
                    s.experts.n_units++;
                    s.experts.total_bytes += mu.bytes;
                    if (mu.movable) s.experts.movable++;
                    s.experts.layer_units[(size_t)mu.layer]++;
                    if (t == Tier::VRAM) s.experts.layer_resident[(size_t)mu.layer]++;
                }
                s.experts.promotions   = (int64_t)ms.promotions;
                s.experts.demotions    = (int64_t)ms.demotions;
                s.experts.no_record    = (int64_t)ms.no_record;
                s.experts.starved_slab = (int64_t)ms.starved_slab;
                s.experts.starved_pool = (int64_t)ms.starved_pool;
                s.experts.quarantined  = (int64_t)ms.quarantined;
                s.experts.stage_layers   = (int64_t)ms.stage_layers;
                s.experts.staged_units   = (int64_t)ms.staged_units;
                s.experts.staged_bytes   = (int64_t)ms.staged_bytes;
                s.experts.stage_overflow = (int64_t)ms.stage_overflow;
                s.experts.stage_copies   = (int64_t)ms.stage_copies;
                s.experts.flex_bytes    = ranks_[0]->mover.flex_used_bytes();
                s.experts.flex_capacity = ranks_[0]->mover.flex_offered_bytes();
                /* The heat engine's decision counts (the mover counts bytes): dispatches that
                 * proposed nothing and why, and whether a promotion was a first (investment) or a
                 * readmit of a just-demoted unit (pure loss). */
                const HeatStats hs = heat.stats();
                s.experts.dispatches        = (int64_t)hs.dispatches;
                s.experts.refused           = (int64_t)hs.refused;
                s.experts.no_candidate      = (int64_t)hs.no_candidate;
                s.experts.distinct_promoted = (int64_t)hs.distinct_promoted;
                s.experts.readmits          = (int64_t)hs.readmits;
                /* The routed hit rate: residency describes the plan, this describes the traffic. */
                s.experts.routed            = (int64_t)hs.routed;
                s.experts.routed_resident   = (int64_t)hs.routed_resident;

            }

            /* The link panel. All-reduce counts come from rank 0 only: the ranks are symmetric,
             * so summing would double them. Moves and streamed experts are summed over ranks:
             * each rank moves its own shard over its own link. `world` lets a reader recover the
             * per-card figure to compare against one link's ceiling. */
            s.passes.present = true;
            for (const auto& rk : ranks_) {
                const Ctx::PassCounts pc = rk->ctx.pass_counts();
                s.passes.played         += pc.played;
                s.passes.recorded       += pc.recorded;
                s.passes.audited        += pc.audited;
                s.passes.issued_unkeyed += pc.issued_unkeyed;
                s.passes.issued_refused += pc.issued_refused;
                s.passes.issued_first   += pc.issued_first;
                s.passes.dispatched     += pc.dispatched;
                s.passes.unordered      += pc.unordered;
            }

            s.link.present        = true;
            s.link.world          = (int)ranks_.size();
            s.link.ar_calls       = ranks_[0]->ctx.ar_calls();
            s.link.ar_bytes       = ranks_[0]->ctx.ar_bytes();
            s.link.h2d_bytes = s.link.d2h_bytes = s.link.ssd_reads = 0;
            s.link.ssd_bytes = s.link.prefetched = s.link.prefetch_bytes = 0;
            s.link.stream_bytes = s.link.staged_bytes = 0;
            for (const auto& rk : ranks_) {
                const MoverStats rms = rk->mover.stats();
                s.link.h2d_bytes      += (int64_t)rms.h2d_bytes;
                s.link.d2h_bytes      += (int64_t)rms.d2h_bytes;
                s.link.ssd_reads      += (int64_t)rms.ssd_reads;
                s.link.ssd_bytes      += (int64_t)rms.ssd_bytes;
                s.link.prefetched     += (int64_t)rms.prefetched;
                s.link.prefetch_bytes += (int64_t)rms.prefetch_bytes;
                s.link.staged_bytes   += (int64_t)rms.staged_bytes;
                if (rk->heat.enabled()) s.link.stream_bytes += (int64_t)rk->heat.stats().stream_bytes;
            }
        }
    };

    server::Deps d{};
    d.sched = &sched_bridge;
    d.tok   = &tok_bridge;
    d.grammar = &grammar_bridge;
    d.live  = snap;
    /* The control vector, when one was found. The admin endpoint flips the
     * default through this; nothing rebuilds a graph, so a toggle takes effect on the next request
     * and leaves the running ones alone. */
    if (cvec_.loaded()) {
        d.control_vector.available  = [this] { return cvec_loaded(); };
        d.control_vector.get_default = [this] { return cvec_default_on(); };
        d.control_vector.set_default = [this](bool on) { set_cvec_default(on); };
        d.control_vector.json        = [this] { return cvec_json(); };
    }

    /* An embedding model serves /v1/embeddings and no chat. Its named prompts are the checkpoint's
     * sentence-transformers configuration (`sentence_transformers.prompts.*`); a `messages` input
     * renders through its chat template. */
    struct EmbedInfo final : server::IEmbedder {
        int64_t d = 0, n = 0;
        const RadModelMeta* meta = nullptr;
        server::IChatTemplate* tpl = nullptr;
        int64_t dim() const override { return d; }
        int64_t max_input() const override { return n; }
        bool prompt(const std::string& name, std::string* text) const override {
            const std::string key = "sentence_transformers.prompts." + name;
            const char* v = rad_meta_gets(meta, key.c_str(), nullptr);
            if (!v) return false;
            *text = v;
            return true;
        }
        std::string default_prompt() const override {
            const char* v = rad_meta_gets(meta, "sentence_transformers.default_prompt_name", "");
            return v ? std::string(v) : std::string();
        }
        server::IChatTemplate* message_template() const override { return tpl; }
    } embed_info;
    embed_info.d = pooled_dim_;
    embed_info.n = cfg_.max_tok;
    embed_info.meta = &meta_;
    if (embedding_) d.embed = &embed_info;

    /* The template loader warns on its own about a missing or broken template; this only decides
     * whether the chat endpoint exists. The reply format is the architecture plugin's when it
     * declares one, else derived from the template. */
    const RadChatFormat* declared_format = rad_tools_chat_format(plugins_, meta_);
    /* An image model's template is its conditioning network's; nothing is served as chat. */
    const bool have_tpl = !gen_ && vocab_ && !vocab_->chat_template().empty() &&
                          chat_bridge.init(*vocab_, declared_format, plugins_.arch_plugin) >= 0 &&
                          chat_bridge.ready();
    if (embedding_) {
        /* No chat endpoint: the model's template, when it has one, renders a `messages` input. */
        if (have_tpl) embed_info.tpl = &chat_bridge;
    } else if (have_tpl) {
        d.chat = &chat_bridge;
    } else if (!gen_) {
        RAD_WARN("no chat template in this container; /v1/chat/completions will answer 501 and "
                 "/v1/completions is the endpoint to use");
    }

    server::ServerOptions o{};
    o.embedding_model = embedding_;
    o.image_model = img_gen_ != nullptr;
    o.audio_model = aud_gen_ != nullptr;
    o.video_model = vid_gen_ != nullptr;
    o.decision_type = rad_meta_gets(&meta_, "decision.type", "");
    o.model_id = !cfg_.served_model_name.empty() ? cfg_.served_model_name
               : meta_.name && *meta_.name       ? meta_.name : "radiance";
    o.host     = cfg_.host;
    o.port     = cfg_.port;
    o.api_key  = cfg_.api_key;
    o.max_seqs = cfg_.max_seqs;
    /* `n` fans one request out over the batch, so the default scales with --max-num-seqs; the
     * floor of 8 keeps a small batch from refusing what it would queue. */
    o.max_n    = cfg_.max_n > 0 ? (int)cfg_.max_n : (int)std::max<int64_t>(o.max_n, cfg_.max_seqs);
    if (cfg_.max_n <= 0)
        decide("--max-n", std::to_string(o.max_n), "the larger of 8 and --max-num-seqs");
    o.max_ctx  = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
    o.model_path = cfg_.model;
    o.default_reasoning_effort = cfg_.reasoning_effort;
    o.reasoning_effort_map = cfg_.reasoning_effort_map;
    o.default_thinking_budget = cfg_.thinking_token_budget;
    if (!gen_) {
        /* The four sampler defaults a file can state: the served value and, per field, its
         * source. A generator samples by its own rules and serves no text requests. */
        GenSampling stated;
        std::string src;
        o.default_sampling =
            resolve_sampling_defaults(meta_, file_ != nullptr, cfg_, &stated, &src);
        for (const GenFieldSource& f : gen_sampling_sources(stated, src, o.default_sampling, cfg_))
            decide(f.flag, f.value, f.why);
    }
    if (cfg_.served_model_name.empty())
        decide("--served-model-name", o.model_id,
               meta_.name && *meta_.name ? "the name the container states"
                                         : "the container states no name");
    o.default_dry_breakers_set = cfg_.sample_dry_seq_breakers.has_value();
    /* An embedding model samples nothing, so it has no logprobs to give. */
    o.supports_logprobs = !embedding_ && ranks_[0]->sampler.has_logprobs();
    /* -1 keeps ServerOptions' default, `auto` (0), resolved per request against the context, so
     * it needs a known context. */
    if (cfg_.default_max_tokens >= 0) o.default_max_tokens = cfg_.default_max_tokens;
    if (o.default_max_tokens == 0 && o.max_ctx <= 0) {
        RAD_ERR("a request that names no max_tokens gets what the context leaves after its prompt "
                "(--default-max-tokens auto), and this model states no context; pass "
                "--max-model-len, or --default-max-tokens N");
        return RAD_E_INVAL;
    }
    o.max_tokens_cap  = cfg_.max_tokens_cap;
    o.queue_depth     = cfg_.max_queued_requests;      /* 0: the server's 8 x max_seqs */
    o.max_stops       = cfg_.max_stop_strings;
    o.max_stop_bytes  = cfg_.max_stop_bytes;
    o.default_template_kwargs = cfg_.chat_template_kwargs;
    o.reasoning_format = cfg_.reasoning_format;
    o.n_threads       = server::idle_workers(cfg_.http_threads);
    if (cfg_.http_threads <= 0)
        decide("--http-threads", std::to_string(o.n_threads),
               "one a hardware thread, at least 4");
    o.read_timeout_s  = cfg_.http_read_timeout_s;
    o.write_timeout_s = cfg_.http_write_timeout_s;
    o.keep_alive_timeout_s = cfg_.http_keep_alive_timeout_s;
    o.max_body_bytes  = (size_t)cfg_.http_max_body_mib << 20;
    o.retry_after_s   = cfg_.retry_after_s;
    o.cors            = cfg_.cors;
    {
        const std::string moved = request_overrides(cfg_);
        if (!moved.empty()) RAD_INFO("request defaults and bounds from the command line: %s",
                                     moved.c_str());
    }
    o.engine_info = engine_info();
    /* Per request, since OpInfo::issued changes once the run phase runs
     * (ServerOptions::graph_json). */
    o.graph_json  = [this] { return dump_graph_json(ranks_[0]->program); };
    /* Per request, from a reading at most a second old (Engine::diagnostics_json). */
    o.diagnostics_json = [this] { return diagnostics_json(); };
    if (cfg_.lora.enabled() && ranks_[0]->program.is_pass_driven()) {
        /* A generator's request names its adapters in `loras`; `model` stays the model. */
        o.gen_lora_acquire = [this](const std::vector<std::pair<std::string, double>>& u,
                                    int32_t* slot, std::shared_ptr<void>* hold, std::string* note,
                                    std::string* e) {
            std::vector<LoraUse> uses;
            for (const auto& [n, w] : u) uses.push_back({ n, w });
            return gen_lora_acquire(uses, slot, hold, note, e);
        };
        o.gen_lora_names = [this] { return gen_lora_names(); };
        o.lora_load = [this](const std::string& n, const std::string& d, std::string* e) {
            return lora_load(n, d, e);
        };
        o.lora_unload = [this](const std::string& n, std::string* e, int* in_flight) {
            return lora_unload(n, e, in_flight);
        };
    } else if (cfg_.lora.enabled()) {
        /* `model` picks the adapter, and the served model's own name picks the base. */
        for (const auto& [name, dir] : cfg_.lora.adapters)
            if (name == o.model_id) {
                RAD_ERR("lora: the adapter '%s' has the served model's own name, so a request "
                        "naming it would run the base; rename it or give --served-model-name",
                        name.c_str());
                return RAD_E_INVAL;
            }
        o.lora_acquire = [this](const std::string& m, int32_t* slot, uint64_t* uid,
                                std::shared_ptr<void>* hold) {
            return lora_acquire(m, slot, uid, hold);
        };
        o.lora_names = [this] { return lora_names(); };
        o.lora_load = [this](const std::string& n, const std::string& d, std::string* e) {
            return lora_load(n, d, e);
        };
        o.lora_unload = [this](const std::string& n, std::string* e, int* in_flight) {
            return lora_unload(n, e, in_flight);
        };
    }

    /* Media, when the program declared an encoder. The processor's geometry comes from the
     * container and the encoder states its own; patch width and merge must agree, checked here
     * rather than found as garbage rows. */
    std::unique_ptr<mm::Processor> mm_proc;
    std::unique_ptr<server::MultimodalBridge> mm_bridge;
    /* How the model hears, for the speech routes: the plugin's RadAudioFormat copied out. */
    server::SpeechFormat speech_fmt;
    const RadEncoderDecl& encd = ranks_[0]->program.encoder;
    if (encd.modalities != 0) {
        mm::VisionConfig vc;
        std::string why;
        if (vc.from_meta(meta_, &why) < 0) {
            RAD_ERR("the '%s' encoder has no processor: %s", ranks_[0]->program.encoder_name.c_str(),
                    why.c_str());
            return RAD_E_INVAL;
        }
        vc.max_patches = encd.max_patches;
        /* Flags override the container field by field, and the result is checked as a whole (a
         * minimum given alone can exceed the container's maximum). Flags a processor would ignore
         * are refused by name: LFM2-VL takes pictures only, and a token-budget processor (Gemma
         * 4's) sizes by soft tokens (--image-tokens, --video-max-frame-tokens), not pixel
         * bands. */
        if (vc.family == mm::Family::Lfm2 &&
            (cfg_.video_min_pixels || cfg_.video_max_pixels || cfg_.video_min_frames ||
             cfg_.video_max_frames || cfg_.video_fps || cfg_.video_max_frame_tokens)) {
            RAD_ERR("media: this model's processor takes pictures, not video, so the --video-* "
                    "flags do nothing on it");
            return RAD_E_INVAL;
        }
        if (vc.family == mm::Family::Gemma4) {
            if (cfg_.image_min_pixels || cfg_.image_max_pixels || cfg_.video_min_pixels ||
                cfg_.video_max_pixels || cfg_.video_min_frames) {
                RAD_ERR("media: this model's processor sizes images and video frames by a token "
                        "budget, so --image-min/max-pixels, --video-min/max-pixels and "
                        "--video-min-frames do nothing; use --image-tokens and "
                        "--video-max-frame-tokens");
                return RAD_E_INVAL;
            }
            if (cfg_.image_tokens)           vc.image_soft_tokens = *cfg_.image_tokens;
            if (cfg_.video_max_frame_tokens) vc.video_soft_tokens = *cfg_.video_max_frame_tokens;
        } else {
            if (cfg_.image_tokens) {
                RAD_ERR("media: --image-tokens is a token budget, and this model's processor "
                        "sizes images by pixels (--image-min-pixels, --image-max-pixels)");
                return RAD_E_INVAL;
            }
            if (cfg_.image_min_pixels)       vc.image_min_pixels  = *cfg_.image_min_pixels;
            if (cfg_.image_max_pixels)       vc.image_max_pixels  = *cfg_.image_max_pixels;
            if (cfg_.video_min_pixels)       vc.video_min_pixels  = *cfg_.video_min_pixels;
            if (cfg_.video_max_pixels)       vc.video_max_pixels  = *cfg_.video_max_pixels;
            if (cfg_.video_min_frames)       vc.min_frames        = *cfg_.video_min_frames;
            if (cfg_.video_max_frame_tokens) vc.max_frame_rows    = *cfg_.video_max_frame_tokens;
        }
        if (cfg_.video_fps)              vc.fps               = *cfg_.video_fps;
        if (cfg_.video_max_frames)       vc.max_frames        = *cfg_.video_max_frames;
        if (cfg_.max_source_pixels)      vc.max_source_pixels = *cfg_.max_source_pixels;
        if (cfg_.media_flags_set() && vc.check(&why) < 0) {
            RAD_ERR("media: the media flags and what the container states disagree: %s",
                    why.c_str());
            return RAD_E_INVAL;
        }
        /* The processor caps the image band at one encoder pass; warn if more was asked for. */
        const int64_t pass_px = encd.max_patches * vc.patch * vc.patch;
        if (cfg_.image_max_pixels && *cfg_.image_max_pixels > pass_px)
            RAD_WARN("media: --image-max-pixels %lld is above what one encoder pass takes (%lld "
                     "pixels at --mm-max-patches %lld); images are resized to at most that",
                     (long long)*cfg_.image_max_pixels, (long long)pass_px,
                     (long long)encd.max_patches);
        if (vc.patch_dim() != encd.patch_dim || (int64_t)vc.merge * vc.merge != encd.merge) {
            RAD_ERR("the container's processor cuts %lld-element patches merged %lld to a row; the "
                    "'%s' encoder takes %lld-element patches merged %lld", (long long)vc.patch_dim(),
                    (long long)vc.merge * vc.merge, ranks_[0]->program.encoder_name.c_str(),
                    (long long)encd.patch_dim, (long long)encd.merge);
            return RAD_E_INVAL;
        }
        if (vc.family == mm::Family::Gemma4 &&
            (int64_t)std::max(vc.image_soft_tokens, vc.video_soft_tokens) * vc.merge * vc.merge >
                encd.max_patches) {
            RAD_ERR("media: an image of %d soft tokens or a video frame of %d is more patches than "
                    "one encoder pass holds (--mm-max-patches %lld)", vc.image_soft_tokens,
                    vc.video_soft_tokens, (long long)encd.max_patches);
            return RAD_E_INVAL;
        }
        mm_proc = std::make_unique<mm::Processor>();
        const int pi = mm_proc->init(
            vc,
            [&tok_bridge](const std::string& t) {
                return tok_bridge.encode(t, /*add_special=*/false, /*parse_special=*/false);
            },
            [this](const std::string& t) { return vocab_->find(t); });
        if (pi < 0) {
            RAD_ERR("media: the container's processor configuration does not fit this deployment "
                    "or this vocabulary (a marker it names is missing, or a segment is larger "
                    "than one encoder pass): %s", rad_strerror(pi));
            return pi;
        }
        o.allow_image = (encd.modalities & RAD_MM_IMAGE) != 0;
        o.allow_video = (encd.modalities & RAD_MM_VIDEO) != 0;
        if (!mm::decoder_available())
            RAD_WARN("media: the '%s' encoder is loaded but this build has no media decoder "
                     "(RAD_WITH_FFMPEG), so every media part will be refused",
                     ranks_[0]->program.encoder_name.c_str());
        else if (mm_proc->config().family == mm::Family::Lfm2)
            RAD_INFO("media: images through '%s'; a picture resizes to %lld..%lld pixels, or one "
                     "past %g times that is cut into %d..%d tiles of %d pixels a side plus a "
                     "thumbnail; %lld patches a pass", ranks_[0]->program.encoder_name.c_str(),
                     (long long)mm_proc->config().image_min_pixels,
                     (long long)mm_proc->config().image_max_pixels,
                     mm_proc->config().max_pixels_tolerance, mm_proc->config().min_tiles,
                     mm_proc->config().max_tiles, mm_proc->config().tile,
                     (long long)encd.max_patches);
        else if (mm_proc->config().family == mm::Family::Gemma4)
            RAD_INFO("media: %s%s through '%s'; an image becomes at most %d soft tokens (a "
                     "request's image_tokens may ask for 70..1120), a video %d a frame at %.3g "
                     "frames a second, at most %d frames; %lld patches a pass",
                     o.allow_image ? "images" : "", o.allow_video ? " and video" : "",
                     ranks_[0]->program.encoder_name.c_str(), mm_proc->config().image_soft_tokens,
                     mm_proc->config().video_soft_tokens, mm_proc->config().fps,
                     mm_proc->config().max_frames, (long long)encd.max_patches);
        else
            RAD_INFO("media: %s%s through '%s'; an image resizes to at most %lld pixels (%lld "
                     "patches a pass), a video to %lld pixels over all its frames",
                     o.allow_image ? "images" : "", o.allow_video ? " and video" : "",
                     ranks_[0]->program.encoder_name.c_str(),
                     (long long)mm_proc->config().image_max_pixels, (long long)encd.max_patches,
                     (long long)mm_proc->config().video_max_pixels);
    }
    /* Audio, when the program declared an audio encoder: framing comes from the container; the
     * encoder states the frame width and frames per output row. */
    const RadEncoderDecl& aend = ranks_[0]->program.audio_encoder;
    if (aend.modalities != 0) {
        mm::AudioConfig ac;
        std::string why;
        if (ac.from_meta(meta_, &why) < 0) {
            RAD_ERR("the '%s' encoder has no feature extractor: %s",
                    ranks_[0]->program.audio_encoder_name.c_str(), why.c_str());
            return RAD_E_INVAL;
        }
        if (ac.n_mels != aend.patch_dim) {
            RAD_ERR("the container's feature extractor makes %d-bin frames; the '%s' encoder takes "
                    "%lld", ac.n_mels, ranks_[0]->program.audio_encoder_name.c_str(),
                    (long long)aend.patch_dim);
            return RAD_E_INVAL;
        }
        /* The tokens around a clip: the config's ids where it names them (Gemma 4's boa/eoa),
         * else the spellings the architecture plugin states (RadAudioFormat). Each must be one
         * token of this vocabulary, or the prompt would carry text where the model was trained on
         * a marker. */
        const RadAudioFormat* afmt = rad_tools_audio_format(plugins_, meta_);
        auto one_token = [&](const char* spell, int32_t* id) -> int {
            if (*id >= 0 || !spell || !*spell) return RAD_OK;
            const std::vector<int32_t> t = tok_bridge.encode(spell, /*add_special=*/false,
                                                             /*parse_special=*/true);
            if (t.size() != 1) {
                RAD_ERR("the architecture plugin spells a clip's marker '%s', which this "
                        "vocabulary reads as %zu tokens rather than one", spell, t.size());
                return RAD_E_INVAL;
            }
            *id = t[0];
            return RAD_OK;
        };
        if (afmt) {
            RAD_TRY(one_token(afmt->audio_open, &ac.tok_start));
            RAD_TRY(one_token(afmt->audio_close, &ac.tok_end));
            auto str = [](const char* p) { return std::string(p ? p : ""); };
            speech_fmt.transcribe_prompt = str(afmt->transcribe_prompt);
            speech_fmt.language_prompt   = str(afmt->language_prompt);
            speech_fmt.transcript_open   = str(afmt->transcript_open);
            speech_fmt.language_lead     = str(afmt->language_lead);
            for (uint32_t i = 0; afmt->languages && i < afmt->n_languages; ++i)
                speech_fmt.languages.emplace_back(str(afmt->languages[2 * i]),
                                                  str(afmt->languages[2 * i + 1]));
        }
        if (ac.tok_start < 0 || ac.tok_end < 0) {
            RAD_ERR("the '%s' encoder's clips have no opening or closing token: the container "
                    "names none and the architecture plugin states none (rad_arch_audio_format)",
                    ranks_[0]->program.audio_encoder_name.c_str());
            return RAD_E_INVAL;
        }
        ac.merge = (int32_t)aend.merge;
        ac.max_frames = aend.max_patches;
        if (!mm_proc) mm_proc = std::make_unique<mm::Processor>();
        if (const int rc = mm_proc->init_audio(ac, &why); rc < 0) {
            RAD_ERR("the '%s' encoder's audio front end cannot be built: %s",
                    ranks_[0]->program.audio_encoder_name.c_str(), why.c_str());
            return rc;
        }
        o.allow_audio = true;
        if (mm::decoder_available())
            RAD_INFO("media: audio through '%s'; a clip is cut at %.3g s, %d mel bins a %d ms frame, "
                     "%d frames a token; %lld frames a pass",
                     ranks_[0]->program.audio_encoder_name.c_str(),
                     (double)mm_proc->audio_config().max_samples / (double)ac.rate, ac.n_mels,
                     ac.hop * 1000 / ac.rate, ac.merge, (long long)aend.max_patches);
    }
    if (mm_proc) {
        mm_bridge = std::make_unique<server::MultimodalBridge>(
            mm_proc.get(), encd.modalities | aend.modalities);
        d.mm = mm_bridge.get();
        if (speech_fmt.transcribes() && o.allow_audio) {
            d.speech = &speech_fmt;
            RAD_INFO("speech: /v1/audio/transcriptions through '%s', %zu languages",
                     ranks_[0]->program.audio_encoder_name.c_str(), speech_fmt.languages.size());
        }
    }

    /* An image generator's endpoints: the driver's queue, the declared prompts and sizes, and the
     * processor that prepares a reference picture for the conditioning network. */
    struct ImageBridge final : server::IImageGen {
        gen::Driver*         drv = nullptr;
        const mm::Processor* proc = nullptr;
        server::ImageGenInfo inf;
        Tokenizer            aux;
        bool                 has_aux = false;
        const server::ImageGenInfo& info() const override { return inf; }
        int encode_aux(const std::string& text, std::vector<int32_t>* out) const override {
            if (!has_aux) return RAD_E_UNSUPPORTED;
            out->clear();
            return aux.encode(text, *out, /*add_special=*/true, /*parse_special=*/true);
        }
        int submit(std::unique_ptr<gen::Request> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int picture(const mm::Frame& f, std::shared_ptr<mm::Item>* out, std::string* why) override {
            if (!proc) {
                if (why) *why = "this model reads no pictures";
                return RAD_E_UNSUPPORTED;
            }
            return proc->image_from(f, out, why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } image_bridge;
    if (img_gen_) {
        const Program& pg = ranks_[0]->program;
        const RadGeneratorDecl& gd = pg.generator;
        server::ImageGenInfo& in = image_bridge.inf;
        in.name = pg.gen_str.name;
        in.tmpl_text = pg.gen_str.tmpl_text;
        in.tmpl_edit = pg.gen_str.tmpl_edit;
        in.tmpl_image = pg.gen_str.tmpl_image;
        in.tmpl_image_sep = pg.gen_str.tmpl_image_sep;
        in.drop_prefix = pg.gen_str.drop_prefix;
        in.tmpl_transparent = pg.gen_str.tmpl_transparent;
        in.latent_scale = gd.latent_scale;
        in.size_multiple = gd.size_multiple;
        in.pixel_channels = gd.pixel_channels;
        in.max_tokens = gd.max_tokens;
        /* Edits need the conditioning network to read pictures, unless only the generator's
         * own encoder reads them. */
        in.refs_encoder_only = gd.layout == RAD_GEN_LAYOUT_JOINT_STEP;
        in.max_refs = mm_proc || in.refs_encoder_only ? gd.max_refs : 0;
        in.ref_shrink_only = gd.ref_shrink_only != 0;
        in.ref_area = gd.ref_area;
        in.default_width = gd.default_width;
        in.default_height = gd.default_height;
        in.default_steps = gd.default_steps;
        in.max_lanes = gd.max_lanes;
        in.max_cond_tokens = img_gen_->max_cond_tokens();
        in.fixed_schedule = !pg.gen_str.sigmas.empty();
        in.fixed_steps = (int64_t)pg.gen_str.sigmas.size();
        in.default_cfg = gd.default_cfg;
        in.cfg_original = gd.cfg_original != 0;
        in.has_default_negative = pg.gen_str.has_default_negative;
        in.default_negative = pg.gen_str.default_negative;
        in.min_tokens = gd.min_tokens;
        in.pad_token = gd.pad_token;
        in.aux_tokens = gd.aux_tokens != 0;
        in.aux_max_tokens = gd.aux_max_tokens;
        if (in.aux_tokens) {
            if (!aux_vocab_) {
                RAD_ERR("images: '%s' reads its prompt through a second tokenizer and %s carries "
                        "none; convert it with rad-convert --tokenizer2", in.name.c_str(),
                        cfg_.model.c_str());
                return RAD_E_FORMAT;
            }
            RAD_TRY(image_bridge.aux.init(aux_vocab_));
            image_bridge.has_aux = true;
        }
        in.samplers = gd.train_steps > 0 ? gd.samplers : 0;
        in.default_sampler = gd.default_sampler;
        in.default_scheduler = gd.default_scheduler;
        in.chunk_tokens = gd.chunk_tokens;
        in.chunk_bos = gd.chunk_bos;
        in.chunk_eos = gd.chunk_eos;
        in.chunk_pad = gd.pad_token;
        image_bridge.drv = img_gen_;
        image_bridge.proc = mm_proc.get();
        d.images = &image_bridge;
        /* The served model id is the generator's own name (qwen-image-2.1), not the container's
         * architecture name. */
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "an image generator: its own name");
        }
        RAD_INFO("images: '%s', %lldx%lld by default, up to %lld latent tokens, %lld picture(s) an "
                 "edit, %s", in.name.c_str(), (long long)in.default_width,
                 (long long)in.default_height, (long long)in.max_tokens, (long long)in.max_refs,
                 in.fixed_schedule ? (std::to_string(in.fixed_steps) + " fixed steps").c_str()
                                   : (std::to_string(in.default_steps) + " steps by default").c_str());
        if (in.samplers)
            RAD_INFO("images: sampler %s, scheduler %s by default", gen::sampler_name(in.default_sampler),
                     gen::scheduler_name(in.default_scheduler));
    }

    /* An audio generator's endpoint: the driver's queue, the declared prompt and limits, and the
     * plugin's rewrite of a request's texts. */
    struct AudioBridge final : server::IAudioGen {
        gen::AudioDriver*      drv = nullptr;
        const LoadedPlugins*   plugins = nullptr;
        server::AudioGenInfo   inf;
        const server::AudioGenInfo& info() const override { return inf; }
        int prepare(int field, const std::string& in, std::string* out,
                    std::string* why) const override {
            const int s = rad_tools_text_prepare(*plugins, field, in, out);
            if (s < 0 && why)
                *why = s == RAD_E_INVAL ? "the model cannot take this text"
                                        : std::string(rad_strerror(s));
            return s;
        }
        int submit(std::unique_ptr<gen::AudioRequest> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } audio_bridge;
    if (aud_gen_) {
        const Program& pg = ranks_[0]->program;
        const RadAudioGenDecl& ad = pg.audio_gen;
        server::AudioGenInfo& in = audio_bridge.inf;
        in.name = pg.agen_str.name;
        in.tmpl = pg.agen_str.tmpl;
        in.tmpl_plain = pg.agen_str.tmpl_plain;
        in.neg_tmpl = pg.agen_str.neg_tmpl;
        in.max_prompt = ad.max_prompt;
        in.max_frames = ad.max_frames;
        in.frame_rate_num = ad.frame_rate_num;
        in.frame_rate_den = ad.frame_rate_den;
        in.sample_rate = ad.sample_rate;
        in.channels = ad.channels;
        in.default_steps = ad.voc_chunk > 0 ? 0 : ad.default_steps;
        in.default_cfg = ad.default_cfg;
        in.max_lanes = ad.max_lanes;
        audio_bridge.drv = aud_gen_;
        audio_bridge.plugins = &plugins_;
        d.audio = &audio_bridge;
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "an audio generator: its own name");
        }
        RAD_INFO("audio: '%s', %lld Hz x %lld, up to %lld frames (%lld s), %lld at once, %s",
                 in.name.c_str(), (long long)in.sample_rate, (long long)in.channels,
                 (long long)in.max_frames,
                 (long long)(in.max_frames * in.frame_rate_den / in.frame_rate_num),
                 (long long)in.max_lanes,
                 ad.voc_chunk > 0
                     ? fmt("played %lld frames at a time as they are made",
                           (long long)ad.voc_chunk).c_str()
                     : fmt("%lld steps a window by default", (long long)in.default_steps).c_str());
    }

    /* A video generator's endpoints: the driver's queue and the declared prompt, sizes and
     * lengths. */
    struct VideoBridge final : server::IVideoGen {
        gen::VideoDriver*    drv = nullptr;
        server::VideoGenInfo inf;
        const server::VideoGenInfo& info() const override { return inf; }
        int submit(std::unique_ptr<gen::VideoRequest> r, std::string* why) override {
            return drv->submit(std::move(r), why);
        }
        int64_t queued() const override { return drv->queued(); }
        int64_t running() const override { return drv->running(); }
    } video_bridge;
    if (vid_gen_) {
        const Program& pg = ranks_[0]->program;
        const RadVideoGenDecl& vd = pg.video_gen;
        server::VideoGenInfo& in = video_bridge.inf;
        in.name = pg.vgen_str.name;
        in.tmpl = pg.vgen_str.tmpl;
        in.max_text = vd.max_text;
        in.fps_num = vd.fps_num;
        in.fps_den = vd.fps_den;
        in.frames_per_chunk = vd.frames_per_chunk;
        in.frames_tail = vd.frames_tail;
        in.min_chunks = vd.min_chunks;
        in.max_chunks = vd.max_chunks;
        in.size_multiple = vd.size_multiple;
        in.min_side = vd.min_side;
        in.max_area = vd.max_area;
        in.max_aspect = vd.max_aspect;
        in.default_width = vd.default_width;
        in.default_height = vd.default_height;
        in.default_chunks = vd.default_chunks;
        in.sample_rate = vd.sample_rate;
        in.channels = vd.audio_channels;
        in.default_steps = vd.default_steps;
        in.max_steps = vd.max_steps;
        in.fixed_steps = vd.n_rungs > 0;
        in.upscale_max = vd.upscale_max;
        in.upscale_refine = vd.upscale_refine;
        in.step_evals = vd.sched_kind == 1;
        in.guidance = vd.guidance != 0;
        for (int i = 0; i < 8; ++i) in.guidance_default[i] = vd.guidance_default[i];
        in.default_negative = pg.vgen_str.default_negative;
        in.image_cond = vd.image_cond != 0;
        in.image_crf = vd.image_crf;
        in.length_head = vd.duration_out != RAD_NULL_HANDLE;
        in.decoders = pg.vgen_str.decoders;
        in.decoder_max_area = pg.vgen_str.decoder_max_area;
        if (in.image_cond && in.image_crf > 0 && !mm::h264_roundtrip_available())
            RAD_WARN("video: '%s' prepares a first frame with an H.264 round trip at crf %d, which "
                     "needs libx264; this build's libavcodec has none, so input_reference is "
                     "refused", in.name.c_str(), (int)in.image_crf);
        in.max_lanes = vd.max_lanes;
        video_bridge.drv = vid_gen_;
        d.videos = &video_bridge;
        if (cfg_.served_model_name.empty() && !in.name.empty()) {
            o.model_id = in.name;
            decide("--served-model-name", o.model_id, "a video generator: its own name");
        }
        const double fps = (double)in.fps_num / (double)in.fps_den;
        RAD_INFO("video: '%s', %lldx%lld by default (up to %lld pixels), %.2f to %.2f s at %g fps "
                 "with %lld Hz x %lld sound, %lld steps by default",
                 in.name.c_str(), (long long)in.default_width, (long long)in.default_height,
                 (long long)in.max_area,
                 (double)(in.frames_per_chunk * in.min_chunks + in.frames_tail) / fps,
                 (double)(in.frames_per_chunk * in.max_chunks + in.frames_tail) / fps, fps,
                 (long long)in.sample_rate, (long long)in.channels, (long long)in.default_steps);
        if (in.upscale_max > 0.0)
            RAD_INFO("video: latent upscaling up to %gx, the last %lld evaluations at the full size "
                     "by default", in.upscale_max, (long long)in.upscale_refine);
    }

    if (encd.modalities == 0 && aend.modalities == 0 && cfg_.media_flags_set())
        RAD_WARN("media: the media flags do nothing here -- this model %s",
                 cfg_.mm_max_patches == 0 ? "is served text only (--mm-max-patches 0)"
                                          : "takes no images or video");

    server::Server srv(d, o);
    if (cfg_.max_queued_requests <= 0)
        decide("--max-queued-requests", std::to_string(srv.queue_cap()), "8 x --max-num-seqs");

    /* A default the template refuses would 400 every chat request that relies on it. Bad
     * --chat-template-kwargs stop startup; --reasoning-effort only warns, since clients that send
     * their own effort are unaffected. */
    if (d.chat && (!cfg_.chat_template_kwargs.empty() || !cfg_.reasoning_effort.empty())) {
        std::string kw_why, eff_why;
        srv.check_template_defaults(&kw_why, &eff_why);
        if (!kw_why.empty()) {
            RAD_ERR("--chat-template-kwargs: the chat template refuses a request carrying them: %s",
                    kw_why.c_str());
            return RAD_E_INVAL;
        }
        if (!eff_why.empty())
            RAD_WARN("--reasoning-effort %s: the chat template refuses it, so every chat request "
                     "that sends no reasoning effort of its own will be answered 400: %s",
                     cfg_.reasoning_effort.c_str(), eff_why.c_str());
    }
    /* Log the efforts the template accepts (a client sending another gets this set in its 400,
     * and /server_info shows it). A map target the template refuses stops startup. */
    if (d.chat) {
        const std::vector<std::string> efforts = srv.probe_reasoning_efforts();
        std::string mapped;
        for (const auto& [from, to] : cfg_.reasoning_effort_map)
            mapped += fmt("%s%s -> %s", mapped.empty() ? "" : ", ", from.c_str(), to.c_str());
        if (!efforts.empty()) {
            std::string list;
            for (const std::string& e : efforts) list += (list.empty() ? "" : ", ") + e;
            RAD_INFO("chat: the template takes reasoning_effort %s, and none or minimal turn "
                     "thinking off%s%s", list.c_str(), mapped.empty() ? "" : "; mapped: ",
                     mapped.c_str());
            /* What a request naming no effort gets, which a client cannot see: Qwen3.8's template
             * renders xhigh into a system prompt (#113). */
            const std::string& dflt = srv.template_default_effort();
            if (cfg_.reasoning_effort.empty() && !dflt.empty())
                RAD_INFO("chat: a request naming no reasoning_effort renders as '%s', the "
                         "template's own default (--reasoning-effort sets another)", dflt.c_str());
        } else if (!mapped.empty()) {
            RAD_INFO("chat: reasoning_effort mapped: %s", mapped.c_str());
        }
        if (const std::string why = srv.check_reasoning_effort_map(); !why.empty()) {
            RAD_ERR("--reasoning-effort-map %s", why.c_str());
            return RAD_E_INVAL;
        }
    }
    /* A thinking budget the model cannot hold stops startup; it would silently apply to no
     * request. */
    if (const std::string why = srv.check_thinking_budget(); !why.empty()) {
        RAD_ERR("--thinking-token-budget %lld: %s", (long long)cfg_.thinking_token_budget,
                why.c_str());
        return RAD_E_INVAL;
    }
    print_startup_report(RAD_OK);
    const char* const keyed = o.api_key.empty() ? "" : "; a bearer key is required";
    if (embedding_)
        RAD_INFO("serving on http://%s:%d  (/v1/embeddings; completions and chat are 501%s)",
                 o.host.c_str(), o.port, keyed);
    else if (aud_gen_)
        RAD_INFO("serving on http://%s:%d  (/v1/audio/speech; completions and chat are 501%s)",
                 o.host.c_str(), o.port, keyed);
    else if (vid_gen_)
        RAD_INFO("serving on http://%s:%d  (/v1/videos; completions and chat are 501%s)",
                 o.host.c_str(), o.port, keyed);
    else if (gen_)
        RAD_INFO("serving on http://%s:%d  (/v1/images/generations%s; completions and chat are "
                 "501%s)", o.host.c_str(), o.port,
                 d.images && d.images->info().max_refs > 0 ? " and /v1/images/edits" : "", keyed);
    else
        RAD_INFO("serving on http://%s:%d  (/v1/completions%s%s)", o.host.c_str(), o.port,
                 d.chat ? "; /v1/chat/completions with tools and reasoning" : "; chat is 501",
                 keyed);

    /* The live view (spec §16, off by default): the console's throughput heartbeat
     * (core/server/liveview.h). Redirected, it prints one key=value line per interval with no
     * escape sequences; the renderer is chosen once by isatty(). LiveView holds no data and only
     * calls `snap`. Scoped here so its thread is joined before Deps goes out of scope. */
    std::unique_ptr<server::LiveView> live;
    if (cfg_.live_view) {
        live = std::make_unique<server::LiveView>(snap);
        live->start();
    }

    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;      /* a second one takes the default disposition and kills */
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    std::atomic<bool> watching{true};
#ifdef RAD_TEST_HOOKS
    /* Once, and from the state itself: Server::port() is 0 until the transport is inside its
     * accept loop, which is where httplib's stop() starts closing the listening socket instead of
     * dropping the stop. The loop test's stop is bound to this report. */
    bool serving_reported = false;
#endif
    std::thread watcher([&] {
        while (watching.load(std::memory_order_relaxed)) {
#ifdef RAD_TEST_HOOKS
            if (!serving_reported) {
                if (const int port = srv.port(); port != 0) {
                    serving_reported = true;
                    rad_test_serving_ready(port);
                }
            }
#endif
            /* This exchange is the flag's one reset point: the signal is cleared where it is
             * acted on, so one raised between two serve runs reaches the next run's watcher. */
            if (g_signalled.exchange(false, std::memory_order_acq_rel)) {
                RAD_INFO("stopping: draining in-flight requests");
                srv.stop();
                return;
            }
            /* The step loop died and its requests were already failed (Engine::fail_loop); stop
             * the transport so no new request is queued. */
            if (loop_status_.load(std::memory_order_acquire) < 0) {
                RAD_ERR("stopping: the step loop failed with %s",
                        rad_strerror(loop_status_.load(std::memory_order_acquire)));
                srv.stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    /* Startup ends here; a quiet server from now on is an idle one. */
    startup_watch_.reset();
    srv.note_ready(startup_begin_ns_);
    const int rc = srv.run();
    /* The watcher tests `watching` first, so clearing it ends the thread without a second stop;
     * run() has already torn the transport down. Joining costs up to one sleep interval. */
    watching.store(false, std::memory_order_relaxed);
    watcher.join();
    if (live) live->stop();
    return rc;
}

}  /* namespace rad */
