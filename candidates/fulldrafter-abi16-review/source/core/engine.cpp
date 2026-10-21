/* engine.cpp -- the step loop and the server.
 *
 * The Engine's construction, the per-rank threads, the one loop that runs a step, and the server
 * it runs under; bring-up, the drafters and the buffer probe have files of their own. See
 * engine_priv.h for what the four share.
 */
#include "engine_priv.h"
#include "kld.h"
#include "mm/processor.h"

#include "build/startup.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "format/radfile.h"
#include "gen_config.h"
#include "sample/accept.h"
#include "sched/scheduler.h"

#include "sched/tree_state.h"

static constexpr int kIssueTreeCommit = 3;
#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

#if __has_include("arch/common/rad_kv_compact.h")
#include "arch/common/rad_kv_compact.h"
#elif __has_include("../arch/common/rad_kv_compact.h")
#include "../arch/common/rad_kv_compact.h"
#endif
#include "server/adapters.h"
#include "server/server.h"
#include "server/liveview.h"
#include "text/tokenizer.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <sys/mman.h>
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

namespace rad {



Engine::Engine(Config cfg) : cfg_(std::move(cfg)) {}

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
    /* WHERE --profile-ops LEAVES ITS FINAL TABLE. Here, because this is the last moment the
     * program still holds the op table the rows name -- and it has to be printed somewhere: a
     * flag that pays for the synchronisations, warns that the run is not comparable, and then
     * says nothing is worse than no flag at all. */
    if (cfg_.profile_ops) for (auto& r : ranks_) r->ctx.dump_profile();
    for (float* p : kld_host_) if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    for (uint16_t* p : kld_tap_host_) if (p) rad_dev_free(p, RAD_MEM_HOST_PINNED);
    if (kld_anchor_host_) rad_dev_free(kld_anchor_host_, RAD_MEM_HOST_PINNED);
    /* The disk thread first: it reads the host arena and writes the store, both of which go with
     * the ranks and kv_disk_ below. */
    if (pump_) pump_->io.stop();
    /* WHAT THE TIERING ACTUALLY DID, and it is printed here for the same reason: this is the last
     * moment the plan the rows name is still alive. Only when the plan was dynamic -- a static
     * one has nothing to say and a banner saying nothing is a banner people learn to skip. The
     * per-layer residency histogram at the end of it is the number that says whether the global
     * ranking starved a layer, which is the failure this engine has and a hit rate cannot show. */
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        if (r->heat.enabled()) {
            RAD_INFO("%s", r->heat.report().c_str());
            RAD_INFO("  routing evidence: %llu layer-histograms credited, %llu layer-sweeps found "
                     "nothing new, %llu dropped by a ring shallower than the run-ahead",
                     (unsigned long long)r->heat_credited, (unsigned long long)r->heat_idle,
                     (unsigned long long)r->ctx.route_dropped());
            RAD_INFO("%s", r->mover.report().c_str());
        }
        if (r->prefetcher.enabled()) {
            RAD_INFO("%s", r->prefetcher.report().c_str());
        }
    }
    for (auto& r : ranks_) r->ctx.shutdown();
    step_state_fini();
    ranks_.clear();
    for (TableCopy& c : table_copies_) table_copy_free(&c);
    table_copies_.clear();
    for (void* p : card_copies_) rad_dev_free(p, RAD_MEM_HOST_MAPPED);
    card_copies_.clear();
    for (const ShmCopy& s : shm_copies_) {
        rad_dev_host_unregister(s.ptr);
        munmap(s.ptr, s.bytes);
    }
    shm_copies_.clear();
    rad_tools_registry().close();
}


/* ================================================================== 7. serve */
/* The step loop, which is the one place every component meets. Threading is spec §1's exactly:
 * HTTP threads feed a request queue, ONE scheduler thread produces ONE step, one thread per
 * tensor-parallel rank each owning one device.
 *
 * The ranks share an address space, so the batch is shared BY POINTER and the whole of the
 * handoff is a barrier -- no serialisation, no pipe, no pickling, which is most of what a Python
 * engine's multiprocessing budget actually buys (rank_barrier.h). The barrier is entered TWICE
 * per step: once to release the ranks against the batch, once to join them before the sampler's
 * answer is read. The scheduler thread is a participant on both sides, which is why the barrier's
 * width is tp + 1 and not tp.
 */

int Engine::declare_sampler(int rank, RadBuilder* b, int64_t logits_width, bool sizing) {
    Rank& r = *ranks_[(size_t)rank];
    /* A SIZING DECLARE DECLARES INTO A SAMPLER OF ITS OWN. The chain has to be in the Program for
     * the buffer list to match the real one, but the rank's sampler keeps the handles the real
     * declare gave it -- the ones every step issues. */
    Sampler scratch;
    Sampler& smp = sizing ? scratch : r.sampler;
    rad_buf scratch_tok = 0;
    rad_buf& tok = sizing ? scratch_tok : r.tok_buf;

    SamplerConfig sc{};
    sc.n_vocab       = meta_.n_vocab;
    /* WHAT THIS RANK ACTUALLY PRODUCES, taken from the logits buffer the plugin declared rather
     * than re-derived from n_vocab and world_size. `program.meta` is the CONTAINER's metadata and
     * carries the whole model's n_vocab on every rank, so reading it here would be wrong the
     * moment a vocabulary is split; and computing the split a second time here would be wrong
     * again the moment a plugin decides not to split it, which is where rad_arch.h stands
     * and says why. Either mistake is a sampler reading past the end of every logits row. */
    sc.n_vocab_local = logits_width > 0 ? logits_width : meta_.n_vocab;
    /* THIS RANK'S FIRST VOCABULARY ROW, and it is the one thing the declared width cannot say:
     * a width tells you how many rows a rank owns, not which. The rule is arch/common/rad_arch.h's
     * shard_begin -- ceil(total/world) per rank, rounded up to a 128-row block, the last one short -- and it is restated here
     * rather than carried through the ABI, so the restatement is CHECKED against what the plugin
     * actually declared. A plugin that shards some other way gets a named refusal at startup
     * instead of a sampler that names rank 0's tokens with rank 1's ids. */
    sc.vocab_off = 0;
    if (sc.n_vocab_local < sc.n_vocab) {
        /* rad_arch.h vocab_per_rank, restated: the even share when the world divides the vocabulary;
         * otherwise ceil(n / tp) rounded up to a whole 128-row block so a block-scaled head can be
         * cut there. */
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

    /* Where the chain writes the sampled ids. The sampler takes it as an argument rather than
     * declaring it, because rad-kbench drives the same chain against a buffer it owns -- so the
     * engine is the one that has to provide one. PERSIST, not transient: the step loop copies it
     * back after the step, by which time a transient's storage may belong to another buffer. */
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
 * WHERE THE HEAT ENGINE RUNS, AND WHY IT IS A STEP AND NOT A LAYER. mover.h writes its run-phase
 * contract as "the order a layer dispatch calls it", and ctx.cpp states the invariant that
 * decides the unit: THE MOVER RELOCATES BETWEEN STEPS, NEVER DURING ONE -- `weights_dirty_` is
 * honoured once at the top of run_step, so a table edit made mid-step would be seen by some
 * issues and not others. A step is therefore the dispatch, and this pair brackets it.
 *
 * IT RUNS ONCE A STEP, AFTER THE STEP'S FORWARD PASS IS ON THE QUEUE:
 *
 *   - AFTER, because nothing in it is work the pass waits for. Ahead of a draft pass the device
 *     is idle -- the previous pass was drained to read its proposal -- so whatever the rank thread
 *     does before its first dispatch is time the device sits out; behind the issue it runs under
 *     the pass instead. Nothing it changes can reach the pass just issued: table edits are
 *     honoured at the top of the NEXT run_step, and a slot it frees is quarantined behind an event
 *     on the compute stream recorded after this pass.
 *   - ONCE A STEP, on the trunk pass and not the draft passes, because a swap almost never
 *     clears the hysteresis between two passes of one step and every call ranks the whole
 *     movable set. The draft head's histograms are not lost by it: they accumulate in the step's
 *     bank and are credited at the next step, weighted by the passes they stand for.
 *   - `begin_dispatch` only RETIRES. It never starts anything, so a loop that retires without
 *     issuing is not a slower engine but a stuck one (mover.h says so at length). The issue call
 *     below is the other half and they are written together for that reason.
 *   - The histograms this credits are the PREVIOUS step's -- a copy is asynchronous and lands
 *     while the step that produced it is still running (spec §5.5) -- and `histogram_ready` says
 *     which have landed.
 *
 * ON A LONG PREFILL CHUNK ALMOST NONE OF THIS LANDS, and the counters below say so rather than
 * leaving it to be inferred from a placement that will not settle. This runs BEFORE run_step and
 * every issue path is enqueue-and-return, so on a chunk worth most of a second of device time the
 * host is many steps ahead of the card: the freshest bank holds a copy the device has not reached,
 * `histogram_ready` is 0, and the layer is skipped. Through a prefill of 2048-token chunks about 2%
 * of the layer-histograms are credited and the rest skipped as not yet landed, against effectively
 * all of them at a decode shape, where the step is short enough that the device keeps up. The same
 * shortfall is what the link panel's expert-streaming share is computed from, so it reads near zero
 * through a prefill that is in fact streaming.
 *
 * `heat_seen` is what stops a layer being credited twice. A bank holds its counts until the next
 * report overwrites them, so a step in which some layer did not route -- a drafter pass, or a
 * bisect running fewer layers -- would otherwise credit the stale plane again and again.
 */
void Engine::tiering_dispatch(Rank& r, const RadBatch* b) {
    if (!r.heat.enabled() || !b) return;
    const int step = b->step;

    r.mover.begin_dispatch(step, &r.heat);

    const int n_layer = r.ctx.n_route_layers();
    if ((int)r.heat_seen.size() != n_layer) r.heat_seen.assign((size_t)n_layer, -1);
    /* THE SWEEP'S HISTOGRAMS ARE FOUND FIRST AND CREDITED SECOND, in the same order. A histogram
     * is a row the card copied into host memory, so every line of it is a miss, and one layer's
     * credit is too little work to hide the next row's fetch behind. With the rows known up front
     * each is prefetched while an earlier one is credited: two rows ahead is 32 lines in flight,
     * about what one core keeps outstanding to memory, so further ahead would only queue. */
    int8_t landed[kRouteBanks];
    std::memset(landed, -1, sizeof landed);
    std::vector<Rank::HeatRow>& rows = r.heat_rows;
    rows.clear();
    for (int l = 0; l < n_layer; ++l) {
        /* DRAIN THE LAYER, do not take one and move on. The host reaches this once per step it
         * ISSUES, and on a long chunk it issues far ahead of the card -- so between two visits the
         * device can have produced many of this layer's histograms. Taking only the newest would
         * throw the rest away and, worse, usually find the newest still in flight and take
         * nothing. */
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
    std::vector<int32_t> coalesced;
    for (size_t i = 0; i < rows.size(); ) {
        const int l = rows[i].layer;
        size_t j = i + 1;
        while (j < rows.size() && rows[j].layer == l) ++j;
        if (j == i + 1) {
            if (i + kAhead < rows.size()) r.heat.prefetch(rows[i + kAhead].h, n_ex);
            const Rank::HeatRow& hr = rows[i];
            const int reps = r.ctx.last_histogram_reports(hr.layer, hr.bank);
            const float rep_wt = reps > 1 ? std::min((float)reps, 2.0f) : 1.0f;
            r.heat.observe(hr.layer, hr.h, n_ex, rep_wt);
            if (r.prefetcher.enabled() && b->n_seq <= 1) {
                r.prefetcher.on_step_routing(hr.layer, hr.h, n_ex);
            }
            r.heat_seen[(size_t)hr.layer] = r.ctx.last_histogram_step(hr.layer, hr.bank);
            r.ctx.mark_credited(hr.layer, hr.bank);
            ++r.heat_credited;
        } else {
            if (coalesced.size() < (size_t)n_ex) coalesced.resize((size_t)n_ex);
            std::fill(coalesced.begin(), coalesced.begin() + n_ex, 0);
            for (size_t k = i; k < j; ++k) {
                if (k + kAhead < rows.size()) r.heat.prefetch(rows[k + kAhead].h, n_ex);
                const Rank::HeatRow& hr = rows[k];
                const int reps = std::max(1, r.ctx.last_histogram_reports(hr.layer, hr.bank));
                const int rep_factor = reps > 1 ? std::min(reps, 2) : 1;
                for (int32_t e = 0; e < n_ex; ++e) {
                    coalesced[(size_t)e] += hr.h[e] * rep_factor;
                }
                r.heat_seen[(size_t)hr.layer] = r.ctx.last_histogram_step(hr.layer, hr.bank);
                r.ctx.mark_credited(hr.layer, hr.bank);
                ++r.heat_credited;
            }
            r.heat.observe(l, coalesced.data(), n_ex, 1.0f);
            if (r.prefetcher.enabled() && b->n_seq <= 1) {
                r.prefetcher.on_step_routing(l, coalesced.data(), n_ex);
            }
        }
        i = j;
    }

    const bool is_final_pf = ((b->phase == RAD_PHASE_PREFILL || b->phase == RAD_PHASE_MIXED) &&
                              b->n_out > 0 && b->n_tok >= 512);
    if (is_final_pf) {
        /* Prefill-to-decode transition: burst-promote hot non-resident experts into VRAM slab.
         * Under high concurrency (n_seq >= 4), throttle burst to avoid PCIe all-reduce contention. */
        const int burst = (b->n_seq <= 2) ? 16 : (b->n_seq <= 4 ? 8 : 4);
        r.heat.set_burst_moves(burst);
    }

    std::vector<Swap> swaps;
    if (r.heat.step(&swaps) < 0) return;
    for (const Swap& s : swaps) {
        if (r.mover.issue(s, &r.heat) < 0) r.heat.abandon(s);
    }

    /* AFTER the swaps, because a swap is what the policy asked for and a fill is what the memory
     * allows: spending a move record on a fill that the next swap then needs would let a
     * boundary the arbiter happens to be moving reorder the engine's own decisions. */
    (void)r.mover.balance(&r.heat);
}

void Engine::on_prefix_hit(const Request& r, const std::vector<int32_t>& units) {
    if (!cfg_.prefix_aware_prefetch || units.empty()) return;
    for (auto& rk : ranks_) {
        if (!rk->heat.enabled()) continue;
        for (int32_t u : units) {
            rk->heat.boost_unit(u, 5.0f);
        }
        /* Boost units' heat scores, but keep dispatch burst capped to avoid PCIe link congestion during concurrency */
        rk->heat.set_burst_moves(4);
    }
}
/* ------------------------------------------------------------------ the rank thread */
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
            /* This rank's view of the step: the same numbers, pointed at this card's staging.
             * The upload that filled it was issued on THIS stream, so the ordering is an
             * ordinary same-stream dependency and no cross-rank fence is needed. */
            const RadBatch* mine = sched_.batch_for(index);
            if (!mine) {
                RAD_ERR("rank %d: the builder has no batch for this rank", index);
                step_status_.store(RAD_E_STATE, std::memory_order_release);
                barrier_->arrive_and_wait();
                continue;
            }
            b = mine;
            const int what = issue_what_.load(std::memory_order_acquire);
            RadBatch commit_batch{};
            if (what == kIssueTreeCommit) {
                commit_batch = *b;
                commit_batch.draft_pass = -2;
                commit_batch.draft_out_ids = sched_.tree_paths();
                commit_batch.n_draft_out = b->n_seq * sched_.tree_path_pitch();
                commit_batch.num_accepted = sched_.tree_paths() + commit_batch.n_draft_out + 2;
                b = &commit_batch;
            }
            int s = what == kIssueSampler ? RAD_OK : r.ctx.run_step(b);
            if (s < 0) {
                RAD_ERR("rank %d step %d: %s", index, b->step, r.ctx.step_error());
                step_status_.store(s, std::memory_order_release);
            } else if (b->n_out > 0 && what != kIssueModel && what != kIssueTreeCommit) {
                /* WAIT FOR THE STAGING, WHICH THE SCHEDULER IS DOING RIGHT NOW UNDER THE MODEL
                 * THIS THREAD JUST ENQUEUED. Everything above is enqueue-and-return, so arriving
                 * here means the device has a forward pass in front of it and this wait costs
                 * nothing it was not going to spend in collect_step anyway. */
                int bst = RAD_OK;
                {
                    std::unique_lock<std::mutex> lk(build_mu_);
                    build_cv_.wait(lk, [this] { return build_ready_; });
                    bst = build_status_;
                }
                /* The sampler chain is issued on the SAME stream, right behind the model, with no
                 * host synchronisation between them: it is more ops on the compute stream and not
                 * a second phase (spec §13). */
                if (bst >= 0) {
                    s = r.sampler.run(&r.ctx, r.program.logits_buf, r.tok_buf);
                    if (s < 0) {
                        RAD_ERR("rank %d step %d: sampler: %s", index, b->step, rad_strerror(s));
                        step_status_.store(s, std::memory_order_release);
                    }
                }
            }
            /* Behind the pass, once a step: see tiering_dispatch. */
            if (s >= 0 && b->draft_pass == 0 && what != kIssueModel) tiering_dispatch(r, b);

        }
        barrier_->arrive_and_wait();                    /* joined; the scheduler may read */
    }
}

/* ------------------------------------------------------------------ the scheduler thread */
int Engine::prepare_release(const RadBatch* b) {
    /* THE ARENA LEVEL, AND THE ARENA'S MEMORY BACK FROM THE SLAB IF THIS STEP NEEDS IT. Every
     * rank thread is parked, so a level can change and a recall can drain both streams. */
    if (const int as = arena_step(b); as < 0) return as;
    /* THE FOLLOWERS' POOLS, BEFORE THE TABLE THAT INDEXES THEM. Admission grew rank 0's cache for
     * this step and the ids are already in the block table; a follower still at the old mark
     * would read pages it never mapped. Here rather than on the arbiter's tick because the tick
     * is 100 ms away and this step is about to run, and safe here because every rank thread is
     * parked at the release. */
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
    if (!pipelining_ || sched_.tree_paths()) return RAD_OK;
    for (int i = 0; i < serial_rows_ && serial_done_ > 0; ++i)
        sched_.expect_draft(serial_row_[(size_t)i].seq, serial_done_);
    const RadBatch* nb = sched_.step_next();
    if (!nb) return RAD_OK;
    /* Its rows' tokens, positions and slots, from where this step's acceptance and drafts leave
     * them -- in stream order behind all of it. */
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

/* EVERY WAY OUT OF THE STEP LOOP ON AN ERROR GOES THROUGH HERE, because leaving it is three
 * obligations and forgetting one is a server that looks alive and serves nothing:
 *
 *   - the rank threads are parked on the barrier, and only an abort lets them see `stopping_`;
 *   - every request queued or running is waiting on a step that will never come, so it is failed
 *     now and its connection answered, and the scheduler refuses anything submitted after;
 *   - run_server watches `loop_status_` and takes the HTTP side down, so the process exits with
 *     the error rather than accepting requests into a queue nothing drains (spec §17). */
int Engine::fail_loop(int st) {
    stopping_.store(true, std::memory_order_release);
    if (barrier_) barrier_->abort();
    sched_.shutdown("the engine stopped on an error and is not serving", st < 0 ? st : RAD_E_STATE);
    loop_status_.store(st < 0 ? st : RAD_E_STATE, std::memory_order_release);
    return st;
}

int Engine::step_loop() {
    /* The batch is built on rank 0's stream into rank 0's memory and shared by pointer, so
     * this thread lives on rank 0's device. The backend's current device is thread-local:
     * a fresh thread starts on device 0, which is right only by accident. */
    if (rad_dev_set(ranks_[0]->device) < 0)
        RAD_ERR("step loop: could not bind rank 0's device");
    StepOut so;
    int idle = 0;
    std::vector<uint64_t> cancel;
    struct TreeOutput { rad_buf rows = 0, reset = 0, span = 0; };
    std::vector<TreeOutput> tree_outputs;
    /* THE PATH/SPAN UPLOAD IS THE FULL-CONTEXT DRAFTER'S, AND ONLY ITS. `tree_path_rows`,
     * `tree_path_reset` and `dloop_context_span` are buffers that drafter declares so that this
     * loop can feed its append and its cache length; a GDN/QSA/PLE tree commit reads the same
     * staging region (`tree_paths()`) directly and needs none of them.
     *
     * The distinction matters for the built-in drafters that now also publish a tree
     * (RADIANCE_DARTREE over DFlash2 or MTP): their target declares the tree VERIFIER, so
     * `tree_paths()` exists and the commit release below must run -- while none of the three
     * upload buffers exists, because there is no persistent drafter cache to append to. Keying
     * this on the ops that actually exist is what keeps one release serving both. */
    const bool dloop_append = !ranks_.empty() && dloop_full_enabled(ranks_[0]->program);
    if (sched_.tree_paths() && dloop_append) {
        tree_outputs.resize(ranks_.size());
        for (size_t r = 0; r < ranks_.size(); ++r) {
            const auto& buffers = ranks_[r]->program.buffers;
            for (size_t h = 1; h < buffers.size(); ++h) {
                if (buffers[h].name == "tree_path_rows") tree_outputs[r].rows = static_cast<rad_buf>(h);
                if (buffers[h].name == "tree_path_reset") tree_outputs[r].reset = static_cast<rad_buf>(h);
                if (buffers[h].name == "dloop_context_span") tree_outputs[r].span = static_cast<rad_buf>(h);
            }
            const auto& out = tree_outputs[r];
            if (r == 0) span_buf_ = out.span;
            if (!out.rows || !out.reset || buffers[out.rows].decl.dtype != RAD_I32 ||
                buffers[out.rows].decl.rank != 2 || buffers[out.rows].decl.shape[1] != 16 ||
                buffers[out.rows].decl.shape[0] < cfg_.max_seqs ||
                buffers[out.reset].decl.dtype != RAD_I32 || buffers[out.reset].decl.rank != 1 ||
                buffers[out.reset].decl.shape[0] < cfg_.max_seqs) {
                RAD_ERR("DLoop tree-state backend requires persistent tree_path_rows/reset buffers");
                return fail_loop(RAD_E_SHAPE);
            }
            /* THE CONTEXT SPAN IS OPTIONAL AND ITS SHAPE IS NOT. A backend that declares one gets
             * its [start,count] pairs uploaded with the path; one that does not is the ring
             * backend, which has no persistent cache to feed. A buffer that IS declared but
             * misshaped is refused here rather than mis-staged every step. */
            if (out.span && (buffers[out.span].decl.dtype != RAD_I32 ||
                             buffers[out.span].decl.rank != 2 ||
                             buffers[out.span].decl.shape[1] != 2 ||
                             buffers[out.span].decl.shape[0] < cfg_.max_seqs)) {
                RAD_ERR("dloop_context_span must be a persistent i32 [max_seqs, 2] buffer");
                return fail_loop(RAD_E_SHAPE);
            }
        }
    }
    /* Not in the KL mode: its scored rows are copied out of the logits buffer after the step is
     * collected, and a pass issued ahead would have overwritten them by then. */
    /* NOTHING IS ISSUED AHEAD OF A TREE. A tree's node count is only known once the rounds have run
     * and the topology is built, while a step issued ahead has to be PLANNED -- and its `n_spec`, its
     * logits rows and its block tables are all sized from the count the drafter was predicted to
     * propose. A predicted count that comes out wrong is a verify step with rows no proposal
     * describes, or a proposal with nodes no row verifies; both are silent at the kernel level. So
     * a program that declared a tree proposal gives up the overlap, exactly as a block drafter
     * already does, and keeps the arithmetic.
     *
     * Everything else keeps the old rule. */
    const bool drafter_declared_tree = !ranks_.empty() && drafter_tree_declared(ranks_[0]->program);
    pipelining_ = !cfg_.kld() && drafter_.kind != RAD_DRAFT_BLOCK &&
                  (serial_drafting() || cfg_.n_spec <= 0) && !drafter_declared_tree;
    /* RADIANCE_PHASE_TIMING=1 turns the per-phase step timer on WITHOUT the rest of the debug level:
     * -v / RADIANCE_LOG=debug also switches on grammar and sampler instrumentation, which halves
     * decode throughput, so the phase split measured under it is not the split of a serving run.
     * Off by default; the line is printed at info level when this is set. */
    static const bool phase_env = std::getenv("RADIANCE_PHASE_TIMING") != nullptr;
    const bool t_on = phase_env || (int)log_level() >= (int)Log::Debug;
    /* Read once. See the block in the idle branch below for what it is for. */
    /* "kv" | "arena" | "all" | "kv:<name>,<name>" -- the last narrows the KV scrub to named groups,
     * which is the level below "it is the pool" and the one that names the defect. */
    const char* const scrub_s_ = std::getenv("RADIANCE_DEBUG_IDLE_SCRUB");
    std::string scrub_groups_;
    int scrub_ = 0;
    if (scrub_s_ && *scrub_s_) {
        if (!std::strncmp(scrub_s_, "kv:", 3)) { scrub_ = 1; scrub_groups_ = std::string(",") + (scrub_s_ + 3) + ","; }
        else if (!std::strcmp(scrub_s_, "kv"))    scrub_ = 1;
        else if (!std::strcmp(scrub_s_, "arena")) scrub_ = 2;
        else                                      scrub_ = 3;
    }
    if (scrub_)
        RAD_WARN("RADIANCE_DEBUG_IDLE_SCRUB=%d: the %s%s%s will be ZEROED every time the engine "
                 "goes idle. This is a determinism diagnostic and it is not free.", scrub_,
                 (scrub_ & 1) ? "KV pool" : "", scrub_ == 3 ? " and the " : "",
                 (scrub_ & 2) ? "activation arena" : "");

    /* THE NEXT STEP'S FORWARD PASS, when it is already on the card: issued behind the step before
     * it, before that step was read back (Scheduler::step_next). */
    const RadBatch* ahead = nullptr;
    while (!stopping_.load(std::memory_order_acquire)) {
        phase_t_ = std::chrono::steady_clock::now();
        /* MEDIA A WAITING REQUEST NEEDS, encoded while nothing is in flight: the last step has been
         * collected and none was issued ahead of it -- the scheduler issues none while a request
         * is waiting, and one whose media is not encoded waits. */
        if (!ahead && sched_.encodes_pending()) {
            if (const int es = run_encodes(); es < 0) return fail_loop(es);
        }
        const bool issued = ahead != nullptr;
        const RadBatch* b = ahead;
        ahead = nullptr;
        if (issued) sched_.promote_next();
        else        b = sched_.step();
        if (!b) {
            if (sched_.step_status() < 0) return fail_loop(sched_.step_status());
            /* Nothing to run is not an error. Back off in small steps rather than spinning a core
             * at idle; the first wait is short, because a request arriving into an empty engine
             * should not pay for the queue having been empty. */
            step_batch_.store(nullptr, std::memory_order_release);
            drain_completed();
            sched_.publish(/*idle=*/true);
            /* Nothing could run because what can run is waiting on the encoder: go round now. */
            if (sched_.encodes_pending()) { idle = 0; continue; }
            /* A RELEASE THE LAST STEP DEFERRED AND NO COMMIT WILL NOW MAKE. A request cancelled
             * while a drafter pass was being built keeps its blocks until the passes that name
             * them have run (Scheduler::cancel), and commit() is where that normally lands -- an
             * idle engine runs none, and blocks held by a cancelled request can be exactly what
             * keep the next one from being admitted. The streams are drained first so no pass
             * still in flight can be reading those blocks when they are handed on. */
            if (sched_.has_deferred()) {
                for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
                sched_.release_deferred();
            }
            tier_tick();
            loan_tick(/*idle=*/true);
            /* RADIANCE_DEBUG_IDLE_SCRUB -- a DIAGNOSTIC, not a fix, and the one question it
             * answers is whether request N+1 differs from request N because of bytes request N
             * left behind. The engine is idle here: no sequence is live, so nothing in either the
             * KV pool or the activation arena is owned by anybody and zeroing all of it is safe.
             *
             * `kv` zeroes every KV group's pool -- paged blocks included, which are NOT cleared on
             * grab (see StateClear in mem/kv.h). `arena` zeroes the buffer plan, which is where
             * every RAD_BUF_PERSIST buffer lives -- `qsa_tail` and `mtp_qsa_tail` among them, and
             * those are indexed by SEQUENCE SLOT and are cleared for a new sequence by nothing at
             * all. `all` does both.
             *
             * If the difference survives the scrub, bytes left behind are not its cause. */
            if (scrub_ && idle == 0) {
                for (auto& rk : ranks_) {
                    if (scrub_ & 1)
                        for (const KVGroupPlan& p : rk->kv.plans()) {
                            if (!p.base || p.pool_bytes <= 0) continue;
                            /* `name`, or `name@L`, or `name@LO-HI` -- a LAYER RANGE, because the
                             * groups are bundles and the layers in one bundle are not one thing.
                             * kv_attn carries the twelve trunk QSA layers AND the MTP draft head's
                             * at index 12, and "which of those" is the question the group level
                             * cannot answer. The pool is layer-major, so a layer is a contiguous
                             * `layer_stride()` at `base + L * layer_stride()`. */
                            int64_t lo = 0, hi = p.n_layers - 1;
                            if (!scrub_groups_.empty()) {
                                const size_t at = scrub_groups_.find("," + p.name + "@");
                                const size_t plain = scrub_groups_.find("," + p.name + ",");
                                if (at == std::string::npos && plain == std::string::npos) continue;
                                if (at != std::string::npos) {
                                    const char* q = scrub_groups_.c_str() + at + p.name.size() + 2;
                                    lo = hi = std::strtoll(q, nullptr, 10);
                                    if (const char* d = std::strchr(q, '-'))
                                        if (d < std::strchr(q, ',')) hi = std::strtoll(d + 1, nullptr, 10);
                                }
                            }
                            if (lo < 0) lo = 0;
                            if (hi > p.n_layers - 1) hi = p.n_layers - 1;
                            const int64_t stride = p.layer_stride();
                            for (int64_t L = lo; L <= hi && stride > 0; ++L)
                                rad_memset_async((char*)p.base + L * stride, 0, stride,
                                                 rk->ctx.stream());
                        }
                    /* The arena's lent top holds expert weights, so it comes back first. */
                    if ((scrub_ & 2) && !rk->arena_lent.empty()) {
                        (void)rad_dev_set(rk->device);
                        (void)rk->mover.reclaim(Mover::kLendArena, 0, &rk->heat);
                        (void)rk->ctx.set_level(0);
                        arena_floor_ = cfg_.max_tok;
                        arena_floor_ns_ = rad_mono_ns();
                    }
                    if (scrub_ & 2)
                        if (void* ab = rk->ctx.arena_base())
                            if (const int64_t n = rk->ctx.arena_bytes(); n > 0)
                                rad_memset_async(ab, 0, n, rk->ctx.stream());
                    rad_stream_sync(rk->ctx.stream());
                }
            }
            if (idle < 200) ++idle;
            std::this_thread::sleep_for(std::chrono::microseconds(idle * 5));
            continue;
        }
        idle = 0;
        if (t_on) phase_mark(PH_PLAN);

        /* A FRESHLY ASSIGNED RECURRENT SLOT IS ZEROED BEFORE THE STEP THAT READS IT. The manager
         * recorded the grab and cannot do the write itself (StateClear in core/mem/kv.h); this is
         * the first moment a stream exists and the last before the ranks are released, and it is
         * off the hot path -- the list is empty on every step but a sequence's first. Every rank
         * clears its own pool: the slot number is the scheduler's, the memory is the rank's. */
        for (const StateClear& c : ranks_[0]->kv.state_clears()) {
            const KVGroupPlan* gp = ranks_[0]->kv.plan(c.group);
            if (!gp || !gp->stateful()) continue;
            const int64_t bytes = gp->layer_bytes_per_state();
            for (auto& rk : ranks_)
                for (int64_t l = 0; l < gp->n_layers; ++l)
                    if (void* p = rk->kv.state_ptr(c.group, (int32_t)l, c.slot))
                        rad_memset_async(p, 0, bytes, rk->ctx.stream());
        }
        ranks_[0]->kv.clear_state_clears();

        /* Which entries want a logits row, in the order BatchBuilder wrote out_ids: entries with
         * produces_token, 1 + n_spec rows each. Everything downstream indexes off this, because
         * row index and sequence index are NOT the same -- a prefill chunk in the middle of a long
         * prompt produces no row at all. */
        const StepPlan& plan = sched_.plan();
        /* ONE ENTRY PER REQUEST, and the row counter is separate. Sampler::build_step expands
         * each request into 1 + n_draft rows of its own -- that is where the per-position RNG
         * coordinate comes from -- so pushing the request once per row here would stage (1+n)^2
         * rows and overrun the arena whenever a drafter sets Request::n_draft. `first_row` is what
         * indexes the sampled tokens, so it counts rows and not requests. */
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

        /* THE SAMPLER'S HOST WORK, staged by this lambda either in front of the ranks or -- by
         * default, see build_mu_ in engine.h -- underneath them. It touches request state and the
         * host grammar automata, which no rank thread reaches: every upload in the sampler chain
         * is issued by Sampler::run, after the gate this fills.
         *
         * CANCELLATIONS ARE COLLECTED, NOT APPLIED. A cancel touches scheduler state that the
         * rank threads are reading through batch_for() while this runs; they are applied after the
         * join, ahead of collect_step. */
        auto stage_sampler = [&]() -> int {
        if (!so.reqs.empty()) {
            /* PER-REQUEST SAMPLER STATE IS BUILT HERE, ON FIRST SIGHT. A grammar arrives compiled
             * from admission and is only positioned here; a DRY sequence-breaker set is resolved
             * against the vocabulary once. Both belong off the step path, and this is the first
             * point on the scheduler thread where a request is known to be about to sample.
             * Without it `seqs_` stays permanently empty, and `grammar` and
             * `dry_sequence_breakers` are accepted by the API, carried through the request, and
             * then quietly do nothing -- the one outcome core/server/oai.h's header forbids.
             *
             * EVERY RANK, because every rank masks its own vocabulary shard from its own automaton.
             *
             * A FAILURE HERE FAILS THE REQUEST, NOT THE ENGINE. A grammar this machine cannot run
             * (left recursion, say) is a bad request, and taking the server down with it would let
             * one caller end every other caller's generation. The server compiles at admission so
             * this should already be unreachable; if it is reached, the request is cancelled with
             * the reason in the log and the rest of the step goes on. */
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
                const int s = rk->sampler.build_step(so.reqs);
                if (s < 0) { RAD_ERR("sampler build_step: %s", rad_strerror(s)); return s; }
                /* A request the sampler could not serve is cancelled, not returned: the step goes
                 * on for everyone else in the batch. */
                for (uint64_t id : rk->sampler.take_failed()) cancel.push_back(id);
            }
            if (t_on) phase_mark(PH_BUILD);
        }
        return RAD_OK;
        };

        cancel.clear();
        int bst = RAD_OK;

        /* ONE LINE PER STEP, on request. The shape of a step -- its phase, how many sequences,
         * how many tokens each, and what context each is continuing from -- is the first thing
         * anyone debugging a wrong answer needs, and no other log line carries it: -v prints the
         * tokenizer and the loader and nothing about the steps that follow. Off by
         * default because it is a line per step and decode steps are thousands of them. */
        if (std::getenv("RADIANCE_LOG_STEPS")) {
            std::string s = fmt("step %d %s n_seq=%lld n_tok=%lld", b->step,
                                b->phase == RAD_PHASE_PREFILL ? "prefill" :
                                b->phase == RAD_PHASE_DECODE  ? "decode"  : "MIXED",
                                (long long)b->n_seq, (long long)b->n_tok);
            for (int i = 0; i < b->n_seq && i < 8; ++i)
                s += fmt("  [%d] slot=%d q=%d ctx=%d", i, plan.e[(size_t)i].slot,
                         plan.e[(size_t)i].n_tokens, plan.e[(size_t)i].ctx_len);
            RAD_INFO("%s", s.c_str());
        }

        /* THE RECURRENT HALF OF A PREFIX HIT, restored BEFORE the step that reads it. The adopted
         * blocks carry the attention layers; this carries the recurrent ones. Enqueued on each
         * rank's own stream ahead of the release, so it is ordered before the kernels without a
         * synchronisation -- the same way collect_step's readback is ordered after them.
         *
         * UNDER THE SCHEDULER'S LOCK, because it reads the sequence table and backs checkpoint
         * slots in rank 0's manager, which an HTTP thread's cancel mutates under that lock. Taken
         * only on a step that has a restore in it. */
        bool any_restore = false;
        for (const StepEntry& e : plan.e) if (e.ckpt_restore >= 0) { any_restore = true; break; }
        if (any_restore) {
            const int rs = sched_.locked([&]() -> int {
                for (const StepEntry& e : plan.e) {
                    if (e.ckpt_restore < 0) continue;
                    if (!state_slots_of(e.seq, &ckpt_slots_)) continue;
                    for (auto& rk : ranks_) {
                    /* THIS RANK'S DEVICE, BECAUSE THE COPY MAY HAVE TO MAP MEMORY FIRST. A
                     * checkpoint slot is backed by the rank that writes it
                     * (KVManager::back_checkpoint), and a commit must land on the card whose pool
                     * it backs -- a commit on card 0 mapped into rank 1's range succeeds, grants
                     * access to the wrong card, and rank 1's next kernel faults on a page that is
                     * present somewhere else. */
                        if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                        const int s = rk->kv.checkpoint_restore(ckpt_slots_.data(),
                                                                (int)ckpt_slots_.size(),
                                                                e.ckpt_restore, rk->ctx.stream());
                        if (s < 0) {
                            RAD_ERR("could not restore sequence %llu from checkpoint slot %d: %s",
                                    (unsigned long long)e.seq, (int)e.ckpt_restore,
                                    rad_strerror(s));
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

        /* WHICH SIDE OF THE BARRIER THE TIME IS ON. Everything above this point is scheduler-thread
         * work the device is idle for -- build_step, the grammar, the plan; everything between the
         * two arrivals is the ranks ENQUEUEING, not running. A step that is slow for a host reason
         * and one that is slow for a device reason look identical in total step time; this split
         * is the only thing that tells them apart. */
        if (t_on) phase_mark(PH_PRE);

        if (!issued) {
            /* The gate is reset while every rank is still parked at the release, so there is no
             * window in which one can read a stale ready flag from the step before. */
            {
                std::lock_guard<std::mutex> lk(build_mu_);
                build_ready_  = false;
                build_status_ = RAD_OK;
            }
            if (const int ps = prepare_release(b); ps < 0) return fail_loop(ps);
            issue_what_.store(kIssueAll, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();      /* release the ranks */
            {
                /* HERE IS THE POINT OF ALL OF IT: the ranks are enqueueing the forward pass and
                 * the device is running it while this thread stages the sampler and walks the
                 * grammar. */
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
            /* THE FORWARD PASS IS ALREADY ON THE CARD, issued behind the last step, and that step
             * has now committed -- so the sampler is staged from exactly the state it would have
             * been, and the ranks put the chain behind the pass while the card is still in it. */
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

        /* A staging failure is discovered with the ranks already released, so they have to be
         * let go before this thread leaves -- the gate above carries the failure, so they have
         * skipped the sampler and arrived at the join. */
        if (bst < 0) return fail_loop(bst);

        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            /* A device fault is not recoverable in process, and we do not serve from a wedged
             * queue (spec §17): stop, release anyone parked on the barrier, and return. */
            RAD_ERR("step %d failed: %s -- the engine cannot continue", b->step, rad_strerror(st));
            return fail_loop(st);
        }

        /* AND THE SNAPSHOT, taken AFTER the step that produced the state. The chunker split this
         * chunk so its last token IS the checkpoint position, so what copy 0 holds now is exactly
         * the state at that position. It is issued before collect_step's readback synchronises, so
         * the copy overlaps whatever is left of the step rather than adding to it.
         *
         * Under the scheduler's lock for the reason the restore above is, and only on a step that
         * writes a checkpoint. */
        bool any_save = false;
        for (const StepEntry& e : plan.e)
            if (e.ckpt_slot >= 0 && e.ckpt_tok >= 0) { any_save = true; break; }
        if (any_save) {
            const int ss = sched_.locked([&]() -> int {
                for (const StepEntry& e : plan.e) {
                    if (e.ckpt_slot < 0 || e.ckpt_tok < 0) continue;
                    if (!state_slots_of(e.seq, &ckpt_slots_)) continue;
                    for (auto& rk : ranks_) {
                    /* THIS RANK'S DEVICE, for the reason the restore gives: the copy may have to
                     * back the slot first, and the pages belong on this rank's card. */
                        if (ranks_.size() > 1) (void)rad_dev_set(rk->device);
                        const int s = rk->kv.checkpoint_save(ckpt_slots_.data(),
                                                             (int)ckpt_slots_.size(),
                                                             e.ckpt_slot, rk->ctx.stream());
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
        st = mark_trunk_end(b, so);
        if (st < 0) return fail_loop(st);
        /* Linear serial passes can overlap host commit. Serial-tree passes cannot: target winner
         * compaction also moves slots in a shared head KV group. Run their history after those
         * moves and commit, so it rewrites the committed head prefix before any round reads it. */
        bool tree_step = sched_.tree_paths() != nullptr;
        for (const auto& e : plan.e)
            if (e.req && e.req->tree_parent) tree_step = true;
        /* WHAT DRIVES THE DRAFTER HERE, AND WHAT DOES NOT.
         *
         * A DLoop deployment's drafter has no passes of its own in the step loop: it is issued on
         * the TREE-COMMIT release below (`draft_pass == -2`), because its persistent cache may only
         * receive rows the host has already accepted. So the two drivers are skipped for it, and
         * that is what this test is about -- NOT the presence of a tree.
         *
         * A built-in drafter (DFlash2, MTP) that proposes a tree is still an ordinary block or
         * serial drafter: the tree is its PROPOSAL SHAPE, and the target's own tree-commit release
         * is a separate pass that commits the winner path into the stateful layers. Skipping the
         * drivers because a tree exists would leave such a deployment proposing nothing at all --
         * the failure looks like speculation switched off, not like a wiring mistake. */
        const bool drafter_on_commit_release = !ranks_.empty() && dloop_full_enabled(ranks_[0]->program);
        const bool serial_post_commit = serial_tree_ && serial_drafting() &&
                                        !drafter_on_commit_release;
        if (!drafter_on_commit_release) {
            st = serial_post_commit ? advance_all(ADV_ACCEPT, 0, false)
                                    : issue_serial_drafter(b->step, b, so);
            if (st < 0) return fail_loop(st);
            st = issue_block_drafter(b->step, b, so);
            if (st < 0) return fail_loop(st);
        }
        st = mark_step_end();
        if (st < 0) return fail_loop(st);
        /* AND THE NEXT STEP'S FORWARD PASS, BEHIND ALL OF IT, before this step is read back. Its
         * rows continue where the card's acceptance leaves them and verify what the drafter just
         * proposed; its sampler waits for this step's commit, which it is staged from. So the card
         * goes from this step's last pass straight into the next step's first, and the host's
         * wait, commit and staging all happen while it runs. */
        ahead = nullptr;
        if (!tree_step && !serial_post_commit) {
            st = issue_ahead(&ahead);
            if (st < 0) return fail_loop(st);
        }
        st = collect_step(b, so);
        if (t_on) phase_mark(PH_COLLECT);
        if (st < 0) return fail_loop(st);
        if (int32_t* paths = sched_.tree_paths()) {
            const int64_t S = b->n_seq;
            const int64_t P = sched_.tree_path_pitch();
            const int32_t* lengths = paths + S * P + 2;
            const int32_t* absolute = lengths + S * 2;
            const int32_t* resets = absolute + S * P;
            /* THE SPAN CROSSES WITH THE PATH AND THE RESET FLAGS, on the same stream and before the
             * release: the append reads all three and a partial upload would append a chunk whose
             * bounds are the previous step's. It is read from the mapped region's tail, which
             * collect_step wrote with plain host stores. */
            const int32_t* span = sched_.dloop_span();
            /* ONLY THE FULL-CONTEXT DRAFTER OWNS THESE BUFFERS; see the note at tree_outputs. A
             * program with a tree VERIFIER but no persistent drafter cache has the staging region
             * and none of the three, and it still runs the commit release below. */
            for (size_t r = 0; r < tree_outputs.size(); ++r) {
                auto& rk = ranks_[r];
                st = rad_dev_set(rk->device);
                if (st < 0) return fail_loop(st);
                /* Preserve the small-tree contiguous upload; wide staging needs a pitched copy
                 * into DLoop's unchanged16-wide append buffer. */
                st = P == 16
                    ? rad_memcpy_async(rk->ctx.buf_ptr(tree_outputs[r].rows),
                                       absolute, S * 16 * sizeof(int32_t), rk->ctx.stream())
                    : rad_memcpy_2d_async(rk->ctx.buf_ptr(tree_outputs[r].rows), 16 * sizeof(int32_t),
                                          absolute, P * sizeof(int32_t),
                                          16 * sizeof(int32_t), S, rk->ctx.stream());
                if (st < 0) return fail_loop(st);
                st = rad_memcpy_async(rk->ctx.buf_ptr(tree_outputs[r].reset),
                                      resets, S * sizeof(int32_t), rk->ctx.stream());
                if (st < 0) return fail_loop(st);
                if (tree_outputs[r].span && span) {
                    st = rad_memcpy_async(rk->ctx.buf_ptr(tree_outputs[r].span),
                                          span, S * 2 * sizeof(int32_t), rk->ctx.stream());
                    if (st < 0) return fail_loop(st);
                }
            }
            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
            issue_what_.store(kIssueTreeCommit, std::memory_order_release);
            step_batch_.store(b, std::memory_order_release);
            barrier_->arrive_and_wait();
            barrier_->arrive_and_wait();
            st = step_status_.load(std::memory_order_acquire);
            if (st < 0) return fail_loop(st);
        }
        if (cfg_.kld()) {
            st = score_rows(b, so);
            if (st < 0) return fail_loop(st);
        }

        /* The buffer probe, after the trunk step: RADIANCE_DUMP_BUF names the buffers to print
         * and trace level gates it, because dumping reads device memory back to the host and no
         * serving run should pay for that. The drafters dump their own buffers after their own
         * passes, for the reason engine_priv.h gives. */
        if ((int)log_level() >= (int)Log::Trace)
            dump_buffers(ranks_[0]->ctx, b->step, std::getenv("RADIANCE_DUMP_BUF"));

        /* --profile-ops PRINTS WHILE THE SERVER IS STILL RUNNING rather than once at the end.
         * The counters are cumulative, so each dump is the run so far and the us/call column is
         * an average over it; a reader wants the table during the run it is diagnosing.
         *
         * Every 32 steps by default, and RADIANCE_PROFILE_EVERY moves it. Per-op timing restores
         * a synchronisation around every launch -- over a thousand of them a step on a large
         * model -- so a profiled step is several times slower and 32 of them is minutes. Often
         * enough to read, rare enough not to bury the log. */
        static const int prof_every = [] {
            const char* v = std::getenv("RADIANCE_PROFILE_EVERY");
            const int n = (v && *v) ? (int)std::strtol(v, nullptr, 10) : 32;
            return n > 0 ? n : 32;
        }();
        if (cfg_.profile_ops && b->step > 0 && (b->step % prof_every) == 0)
            for (auto& rk : ranks_) rk->ctx.dump_profile();

        /* One line a step at -vv. It is the only view of the loop there is: everything else the
         * engine prints happens at startup, so a request that goes in and never comes out looks
         * exactly like a request that never arrived. */
        RAD_TRACE("step %d: %lld tok / %lld seq, %lld logit row(s) -> tokens %s", b->step,
                  (long long)b->n_tok, (long long)b->n_seq, (long long)b->n_out,
                  so.token.empty() ? "(none)" : fmt("%d%s", so.token[0],
                                                    so.token.size() > 1 ? " ..." : "").c_str());
        /* AND WHAT WENT IN. Every op in the forward pass can agree with the reference and the
         * answer still be wrong if the ids are not the ids the model was trained on; the two look
         * identical from the logits, which is why this is printed beside them rather than left to
         * be inferred. Bounded at 32 because a 2048-token prefill's ids are not a log line. */
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
        res.token      = so.token.data();
        res.eos        = so.eos.data();
        st = sched_.commit(res);
        if (t_on) phase_mark(PH_COMMIT);
        if (st < 0) { RAD_ERR("commit: %s", rad_strerror(st)); return fail_loop(st); }

        if (serial_post_commit) {
            st = issue_serial_drafter(b->step, b, so);
            if (st < 0) return fail_loop(st);
            st = mark_step_end();
            if (st < 0) return fail_loop(st);
        }

        /* Commit and publication overlap a drafter already issued behind the trunk; its readback
         * waits below. Stateful block drafters and prefill still run from the committed sequence.
         * Keep the publication decision: finish_block_drafter clears block_preissued_. */
        const bool publish_early = drafter_.kind == RAD_DRAFT_SERIAL || block_preissued_;
        if (publish_early) {
            drain_completed();
            sched_.publish(/*idle=*/false);
            /* Maintenance uses committed state, not the pending proposals. Tier copies fence on
             * the compute stream; loan ticks only adjust offers while idle is false. */
            tier_tick();
            loan_tick(/*idle=*/false);
            arena_tick(rad_mono_ns());
            st = rad_event_sync(step_end_);
            if (st < 0) return fail_loop(st);
        }
        st = drafter_.kind == RAD_DRAFT_BLOCK
                 ? (block_preissued_ ? finish_block_drafter(so) : run_block_drafter(b->step))
                 : finish_serial_drafter(so);
        if (st < 0) return fail_loop(st);
        /* AND NOW THE STEP IS OVER. The draft rounds above are forward passes and they belong to
         * the step that paid for them; charging the step at commit would leave them in the gap
         * between one iteration and the next, which nothing measures. Scheduler::step_finished
         * says what that costs. */
        sched_.step_finished();
        if (!publish_early) {
            tier_tick();
            loan_tick(/*idle=*/false);
            arena_tick(rad_mono_ns());
            drain_completed();
            sched_.publish(/*idle=*/false);
        }

        /* ONE LINE EVERY 128 STEPS, and it adds up to the step. `ranks` is the enqueue, not the
         * execution: the device work is waited for in `collect`, which is the only host
         * synchronisation a step has, so `collect` is where device time shows up and any host
         * cost the ranks could have been running under is time `collect` did not get to hide. */
        if (t_on) {
            phase_mark(PH_DRAFT);
            if (++phase_n_ == 128) {
                double tot = 0;
                for (int i = 0; i < PH_N; ++i) tot += phase_us_[i];
                RAD_LOG_AT(phase_env ? Log::Info : Log::Debug,
                          "step %s: %.2f ms = plan %.2f begin %.2f build %.2f pre %.2f "
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

/* THE SAMPLED IDS ON THEIR WAY TO THE HOST: a copy into pinned memory behind the sampler, and
 * nothing waited for. Issued by the serial drafter before its passes -- a head that samples
 * through the sampler's own buffer overwrites the ids -- and otherwise right before the wait. */
int Engine::collect_issue(const RadBatch* b, StepOut& so) {
    so.rb = nullptr;
    so.lg.clear();
    if (so.n_rows <= 0) return RAD_OK;
    Rank& r0 = *ranks_[0];
    void* src = r0.ctx.buf_ptr(r0.tok_buf);
    if (!src) {
        RAD_ERR("the sampler's token buffer has no address -- it was declared but never planned");
        return RAD_E_STATE;
    }
    so.rb = sample_readback_.reserve(so.n_rows);
    if (!so.rb) return RAD_E_NOMEM;
    RAD_TRY(rad_memcpy_async(so.rb, src, (int64_t)so.n_rows * 4, r0.ctx.stream()));

    /* THE FIRST FEW LOGITS, at -vv. There is no other window into the forward pass: rad-kbench can
     * falsify one op against the oracle but nothing compares a whole layer, so when every op agrees
     * and the model still answers the same token to every prompt, the question "are the logits even
     * a function of the input" has no cheaper answer than looking. One copy on a path that already
     * synchronises, and only when the operator asked for trace. */
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

int Engine::mark_trunk_end(const RadBatch* b, StepOut& so) {
    if (so.n_rows > 0 && !so.rb) RAD_TRY(collect_issue(b, so));
    return rad_event_record(trunk_end_, ranks_[0]->ctx.stream());
}

int Engine::mark_step_end() {
    return rad_event_record(step_end_, ranks_[0]->ctx.stream());
}

/* Wait for the sampled ids and turn them into the three per-sequence arrays the scheduler wants.
 * Only the trunk readback is needed here; the draft passes and their readbacks are waited after
 * commit, so the host's acceptance and commit overlap their GPU work. */
int Engine::collect_step(const RadBatch* b, StepOut& so) {
    const StepPlan& plan = sched_.plan();
    so.token.assign(plan.e.size(), 0);
    so.accepted.assign(plan.e.size(), 0);
    so.eos.assign(plan.e.size(), 0);
    int32_t* paths = sched_.tree_paths();
    const int64_t S = b->n_seq;
    const int64_t P = sched_.tree_path_pitch();
    int32_t* lengths = paths ? paths + S * P + 2 : nullptr;
    int32_t* starts = lengths ? lengths + S : nullptr;
    int32_t* absolute = starts ? starts + S : nullptr;
    int32_t* resets = absolute ? absolute + S * P : nullptr;
    /* THE CONTEXT SPAN IS STAGED INTO THE MAPPED REGION'S OWN TAIL, which was sized for it at init
     * and is reused every step. It is not a StepOut vector: this buffer is allocated once, its
     * writes are ordinary host stores, and the upload that carries them rides the same stream, in
     * the same order, as the path and reset flags it is read with -- so there is no per-step
     * allocation and no vector whose lifetime has to outlast the barrier.
     *
     * `starts` is the same array the QSA commit reads, and it is ALREADY the running trunk row
     * counter: it is what `absolute` is built from and what the append's span is measured in. */
    int32_t* span = sched_.dloop_span();
    const bool want_span = span != nullptr && span_buf_ != 0;
    if (want_span) std::fill(span, span + S * 2, 0);
    /* TRUNK-ROW COORDINATES, WHICH IS WHAT THE FULL-CONTEXT APPEND READS.
     *
     * `absolute` used to carry the OUT-ROW of each accepted node: `so.first_row[i] + best_nodes[j]`,
     * a row of the logits buffer. That was correct while the taps were the KLD recording's, whose
     * rows are gathered through `out_ids`. The full-context drafter's taps are ONE ROW A TOKEN of
     * the step -- including the tokens of a chunk that samples nothing -- so the row a node names
     * must be its TRUNK row, and `starts[i]` is exactly the accumulated trunk row start. The
     * mapping is `starts[i] + node` for both halves, which is the same expression the scheduler's
     * own QSA commit uses (`rad_qsa.h`, `tok = starts[s] + row[j]`).
     *
     * `paths` and `lengths` keep their old meaning: the QSA commit and the GDN tree kernels read
     * node indices and accepted path lengths, and neither is touched by this cutover. */
    auto stage_tree = [&] {
        if (!paths) return;
        std::fill(paths, paths + S * P, -1);
        std::fill(absolute, absolute + S * P, -1);
        paths[S * P] = 0;
        paths[S * P + 1] = 1;
        int32_t start = 0;
        for (int64_t i = 0; i < S; ++i) {
            const auto& e = plan.e[static_cast<size_t>(i)];
            lengths[i] = e.is_prefill ? 0 : 1;
            starts[i] = start;
            /* RESET ONLY AT A TRUE SEQUENCE START, WHICH IS NOT THE SAME AS "A PREFILL ENTRY".
             *
             * The old rule set 1 on every prefill entry, and against a 32-row RING that was
             * harmless -- the ring was rebuilt from the whole window on every base pass, so
             * clearing it lost nothing a later step did not recompute. Against a PERSISTENT cache it
             * is fatal in exactly the case this contract exists to serve: the second chunk of a
             * prompt would clear the first chunk's K/V, the third would clear the second's, and a
             * long prefill would end with the drafter believing the sequence began at its last
             * chunk. The symptom is fluent output with collapsed acceptance, which is why this is
             * a correctness rule and not a tuning one.
             *
             * `ctx_len` is "computed tokens BEFORE this step", so 0 is the only value that means
             * this entry's first token is the sequence's first. That covers a fresh sequence and a
             * slot reused by a new one -- which is why the flag is per-OCCUPATION and not
             * per-slot-index.
             *
             * THERE IS NO PREFIX-HIT CASE TO GET RIGHT, because a full-context drafter disables the
             * prefix cache at Scheduler::init: a hit skips recomputing a prompt this drafter's
             * persistent cache is built by appending, so no hit is ever taken and every prompt runs
             * from position 0. The consequence is not something this line has to accommodate --
             * the reuse is gone (core/sched/scheduler.cpp). */
            resets[i] = (e.ctx_len == 0) ? 1 : 0;
            /* THE SPAN IS THE CHUNK, AND EVERY CHUNK IS TAKEN -- INCLUDING THE ONES THAT SAMPLE
             * NOTHING. A prefill entry's whole token range is a contiguous run of tap rows, so
             * count >= 0 names it and no path is needed. A decode entry's verified set is a node
             * path and not a run, so count == -1 selects `path` instead. The reset flag keeps its
             * meaning: 1 at a true sequence start, 0 for every later chunk of the same sequence.
             *
             * An entry with no tokens at all is an empty chunk, which the op allows; it advances no
             * length and reads no row. */
            if (span) {
                span[i * 2] = e.is_prefill ? start : 0;
                span[i * 2 + 1] = e.is_prefill ? e.n_tokens : (e.n_tokens > 0 ? -1 : 0);
            }
            if (e.is_prefill) {
                /* A prefill chunk's rows are the chunk; the path is unused but staged -1 so a
                 * stale entry from the step before cannot be read by a mistaken consumer. */
                if (e.produces_token) absolute[i * P] = start;
            } else {
                paths[i * P] = 0;
                absolute[i * P] = start;
            }
            sched_.accepted_leaf(e.seq, 0);
            start += e.n_tokens;
        }
    };
    if (so.n_rows <= 0) {
        if (paths && trunk_end_) RAD_TRY(rad_event_sync(trunk_end_));
        stage_tree();
        return RAD_OK;
    }

    so.rows.resize((size_t)so.n_rows);
    if (!so.rb) return RAD_E_STATE;
    /* Never wait for step_end_ here: that would put FullDrafter ahead of acceptance and commit.
     * mark_trunk_end recorded this event after the sampled-token copy and before draft work. */
    if (trunk_end_) RAD_TRY(rad_event_sync(trunk_end_));
    std::memcpy(so.rows.data(), so.rb, (size_t)so.n_rows * sizeof(int32_t));
    if (spechub_)
        for (const int32_t token : so.rows)
            if (token < 0 || token >= meta_.n_vocab) {
                RAD_ERR("SpecHub verifier refused invalid target support or proposal metadata");
                return RAD_E_STATE;
            }
    so.rb = nullptr;
    stage_tree(); /* the trunk fence also retires the preceding winner's mapped reads */
    const std::vector<float>& lg = so.lg;
    const int64_t nv = meta_.n_vocab;
    if (!lg.empty()) {
        /* The top three of the first sampled row. A sampler that answers the same token to every
         * prompt is either reading a distribution with a fixed spike in it or not reading the
         * distribution at all, and the two look identical from outside. */
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
            /* THE COMPARISON IS AGAINST THE TARGET'S OWN SAMPLED TOKEN, AND THAT IS EXACT AT ANY
             * TEMPERATURE. It looks as though everything but temperature 0 has to be refused: a
             * positive temperature needs rejection sampling, and rejection sampling needs the
             * target's probability planes, which this path does not bring back. Both halves are
             * true and the conclusion does not follow, because there is a third rule that needs
             * no planes at all:
             *
             *   accept draft[k] when it equals the token the target ITSELF drew at position k.
             *
             * That is exact because the sampler gives every verified position its own RNG
             * coordinate -- `p.pos = n_out + k`, "its randomness is the same whether it arrived
             * through speculation or one token at a time" (sampler.cpp). So y_k is the draw the
             * target's own distribution specifies at that output position. The first emitted token
             * is y_0 whether or not the draft agreed; every later one is conditioned on a prefix
             * that was accepted only because it matched, so the conditioning is the one the target
             * asked for. The emitted sequence is a draw from the target, not an approximation of
             * one.
             *
             * IT IS NOT, HOWEVER, BYTE-IDENTICAL TO A NON-SPECULATIVE RUN ABOVE TEMPERATURE 0, and
             * that is worth stating because the paragraph above sounds like it should be. The rule
             * is exact given the same logits; the logits themselves are shape-dependent, because a
             * verify step computes 1 + n_spec rows where a decode step computes one and split-K
             * picks its count from M. At temperature 0 that survives the argmax and the text
             * matches exactly. Above it a sampled pick near a CDF boundary can land the other way,
             * so spec and non-spec agree on most prompts and not all -- deterministic and
             * reproducible either way, and a property of the GEMMs rather than of acceptance.
             *
             * It accepts less often than optimal rejection sampling -- P(y == x) rather than
             * min(1, p/q) -- and that is the whole price. Refusing instead would cost the request
             * every drafted token AND take the step down with it, failing every other request
             * batched alongside, for the common case of a client that left temperature at 1.
             *
             * `so.rows` is the sampler's OUTPUT, which is the sampled token -- at temperature 0
             * that is the argmax, which is why the greedy-acceptance name still fits. */
            if (!e.req) continue;
            int32_t out[RAD_SCHED_MAX_SPEC + 1] = { 0 };
            int32_t acc = 0;
            int32_t bonus = 0;
            int32_t best_nodes[RAD_SCHED_MAX_SPEC + 1] = { 0 };

            radiance::ActiveTreeRecord trec;
            const bool has_draft_tree = radiance::TreeRegistry::instance().get(e.seq, &trec);

            if (has_draft_tree && !trec.tree.empty()) {
                RAD_TRY(accept_tree_greedy(trec.tree,
                                           &so.rows[(size_t)so.first_row[i]],
                                           e.n_tokens,
                                           out,
                                           &acc,
                                           &bonus,
                                           best_nodes));

                /* Check for EOG truncation before KV compaction and rollback */
                for (int32_t j = 0; j <= acc; ++j) {
                    if (vocab_ && vocab_->is_eog(out[j])) { acc = j; break; }
                }

                if (e.draft) {
                    for (int32_t j = 0; j < acc; ++j) {
                        const_cast<int32_t*>(e.draft)[j] = out[j];
                    }
                }

                /* Winning branch KV cache compaction (Milestone 3):
                 * Relocate non-identity candidate slots into contiguous sequence positions. */
                if (acc > 0) {
                    bool needs_compaction = false;
                    for (int32_t k = 1; k <= acc; ++k) {
                        if (best_nodes[k] != k) {
                            needs_compaction = true;
                            break;
                        }
                    }

                    if (needs_compaction) {
                        for (auto& rk : ranks_) {
                            for (size_t g = 0; g < rk->kv.plans().size(); ++g) {
                                const auto& plan = rk->kv.plans()[g];
                                /* Block keys are pooled per logical block, not per candidate
                                 * token; QSA winner replay rebuilds them after acceptance. */
                                if (!plan.paged() || (paths && plan.name == "kv_qsa_bkey")) continue;

                                const std::vector<int32_t>& bt = rk->kv.block_table(e.seq, static_cast<int32_t>(g));
                                const int64_t first_pos = rk->kv.first_block_pos(e.seq, static_cast<int32_t>(g));
                                const int32_t bs = static_cast<int32_t>(plan.block_size);

                                /* Every non-identity accepted node, up to a full window. The
                                 * device takes RAD_KV_COMPACT_MAX_MOVES a launch; ascending chunks
                                 * on one stream keep the hazard-free order, because move k's
                                 * target precedes every later move's source. */
                                radiance::RadKVSlotMove moves[RAD_SCHED_MAX_SPEC];
                                const int32_t n_moves = radiance::rad_kv_compact_build_moves(
                                    best_nodes,
                                    acc,
                                    static_cast<int64_t>(e.ctx_len),
                                    bt,
                                    first_pos,
                                    bs,
                                    moves,
                                    RAD_SCHED_MAX_SPEC
                                );

                                if (n_moves > 0 && bs > 0) {
                                    const int64_t block_bytes = plan.layer_bytes_per_block();
                                    const int32_t n_heads = static_cast<int32_t>(plan.n_head_kv > 0 ? plan.n_head_kv : 1);
                                    const int32_t tok_bytes = static_cast<int32_t>(block_bytes / (bs * n_heads));
                                    const int64_t kv_head_stride = static_cast<int64_t>(bs) * tok_bytes;

                                    radiance::RadKVCompactParams compact{};
                                    compact.kv_pool_base = plan.base;
                                    compact.layer_stride = plan.layer_stride();
                                    compact.kv_block_stride = block_bytes;
                                    compact.kv_head_stride = kv_head_stride;
                                    compact.block_size = bs;
                                    compact.tok_bytes = tok_bytes;
                                    compact.n_layers = static_cast<int32_t>(plan.n_layers);
                                    compact.n_heads = n_heads;
                                    for (int32_t m0 = 0; m0 < n_moves; m0 += RAD_KV_COMPACT_MAX_MOVES) {
                                        compact.n_moves = std::min<int32_t>(n_moves - m0, RAD_KV_COMPACT_MAX_MOVES);
                                        std::copy_n(moves + m0, compact.n_moves, compact.moves);
                                        RAD_TRY(kv_compact_winning_branch(compact, rk->ctx.stream()));
                                    }
                                }
                            }
                        }
                    }
                }

                // Scheduler::commit owns rollback, including the next_open_ guard.
                radiance::TreeRegistry::instance().clear(e.seq);
            } else {
                if (has_draft_tree) {
                    radiance::TreeRegistry::instance().clear(e.seq);
                }
                RAD_TRY(accept_greedy(e.draft, e.n_spec, &so.rows[(size_t)so.first_row[i]], out, &acc));
                for (int32_t j = 0; j <= acc; ++j) best_nodes[j] = j;
            }

            /* THE TURN CAN END IN THE MIDDLE OF AN ACCEPTED RUN, and stopping only on the last
             * token of it is how the engine generates past the end of a turn. accept_greedy stops
             * at the first DISAGREEMENT, not at the first end-of-generation token -- and a drafter
             * that correctly predicts the model is about to stop AGREES about the stop token, so
             * the run continues past it. Everything after it is the model writing the next turn:
             * `<|im_end|><|im_start|>user ...` decoded, with the control tokens rendering as
             * nothing, which reads as a model that rambles rather than as a bug in the accept
             * path. Truncating here rather than in the scheduler keeps one definition of "how far
             * did this step get" -- `accepted` and `token` are what the KV rollback, the conv
             * cursor and the emitted text are all derived from. */
            for (int32_t j = 0; j <= acc; ++j) {
                if (vocab_ && vocab_->is_eog(out[j])) { acc = j; break; }
            }
            /* Publish the current path even without a stateful tree-commit staging region.
             * Empty spells the identity for a chain; no history consumer may reuse an old tree. */
            const bool real_tree = e.req && e.req->tree_parent;
            sched_.accepted_leaf(e.seq, best_nodes[acc],
                                 real_tree ? best_nodes : nullptr,
                                 real_tree ? acc + 1 : 0);
            if (paths) {
                lengths[i] = acc + 1;
                /* TRUNK ROWS, NOT OUT ROWS. `starts[i]` is this entry's first row of the step and
                 * `best_nodes[j]` is the j-th accepted node of the winning path, so the sum is the
                 * absolute tap-plane row that node's hidden state occupies -- which is the
                 * coordinate the full-context append's `path` operand is read in. The QSA commit
                 * computes the same expression from the same two arrays (rad_qsa.h,
                 * `tok = starts[s] + row[j]`), so the two consumers of `absolute` agree by
                 * construction rather than by coincidence. */
                for (int32_t j = 0; j <= acc; ++j) {
                    paths[i * P + j] = best_nodes[j];
                    absolute[i * P + j] = starts[i] + best_nodes[j];
                }
                /* THE DECODE SPAN NAMES THE PATH. count == -1 is the contract's spelling for
                 * "the verified set is these nodes, not a run", and `lengths[i]` is how many of
                 * them are real -- one the append op also reads, so the two cannot disagree. */
                if (span) {
                    span[i * 2] = 0;
                    span[i * 2 + 1] = -1;
                }
            }

            so.accepted[i] = acc;
            so.token[i]    = out[acc];
            so.eos[i]      = vocab_ && vocab_->is_eog(so.token[i]) ? 1 : 0;
            if (spechub_)
                RAD_TRACE("SpecHub verify req=%llu pos=%llu root_y=%d accepted=%d leaf=%d bonus=%d",
                          (unsigned long long)e.seq, (unsigned long long)e.req->output.size(),
                          out[0], acc, best_nodes[acc], out[acc]);
            /* EVERY token that leaves, in order: the grammar's state has to advance over the
             * accepted drafts too, or the next step's mask is built n_accepted tokens behind.
             * Bounded by the truncated `acc`, so it sees exactly the tokens the client does. */
            if (e.req && e.req->state == ReqState::Running)
                for (auto& rk : ranks_)
                    for (int32_t j = 0; j <= acc; ++j) {
                        const int as = rk->sampler.accept_token(e.req->id, out[j]);
                        /* A parse that outgrew the grammar's bounds is the request's, not a
                         * defect: the backend is dead and the request fails at its next mask. */
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
        sched_.accepted_leaf(e.seq, 0, nullptr, 0);
        so.token[i] = so.rows[(size_t)so.first_row[i]];
        so.eos[i]   = vocab_ && vocab_->is_eog(so.token[i]) ? 1 : 0;
        /* The grammar must see the token it just emitted, or the next step's mask is built from a
         * state one token behind the sequence. EVERY rank, not just rank 0: each rank builds the
         * mask for its own vocabulary shard from its own automaton, so a rank whose grammar did
         * not advance masks the next step against the previous token's state. */
        if (e.req && e.req->state == ReqState::Running)
            for (auto& rk : ranks_) rk->sampler.accept_token(e.req->id, so.token[i]);
    }

    /* WHETHER EACH REQUEST MAY BE SPECULATED ON NEXT STEP, decided HERE and not in the block that
     * starts requests, because the answer changes exactly where this loop just changed it: a lazy
     * grammar becomes constraining inside accept_token, above. Deciding it any later means the
     * scheduler sizes the next step from a stale answer and hands a constrained request a draft
     * the sampler will then refuse. */
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (!e.req) continue;
        Request* r = const_cast<Request*>(e.req);
        if (!ranks_.empty()) {
            r->may_speculate = ranks_[0]->sampler.allows_speculation(r->id);
        }
    }
    return RAD_OK;
}

/* Requests that reached a terminal state. The server drains the text through the sink and calls
 * reap(), which is what finally lets the Request go -- only the HTTP thread knows when it has
 * stopped writing into it. All this does is drop the sampler's per-request host state. */
/* ------------------------------------------------------------------ the idle-tier tick */
/* EVERY ITERATION, ON BOTH PATHS. A copy to host memory is asynchronous and reported when its
 * event has signalled, so the tick that collects it is simply the first one after it landed; the
 * next copy is handed out on the same tick. Tying either to a slower clock would only delay the
 * moment a finished conversation's VRAM can be given up. The disk half runs once a second -- its
 * thresholds are measured in minutes -- and at once when the host tier has refused a copy for
 * want of room, since that copy is waiting on it.
 *
 * THE SCHEDULER'S LOCK IS HELD FOR THE BOOKKEEPING ONLY. Planning a batch and reporting one touch
 * rank 0's manager, the prefix cache and -- through the tiers' drop sinks -- the index an HTTP
 * thread's cancel also changes; issuing the transfers and polling them touch none of that, and a
 * transfer's records are pending the whole time it is out, which is what keeps everything else off
 * them without a lock. */
/* THE ONE PLACE THE BOUNDARY BETWEEN THE KV CACHE AND THE EXPERT SLAB MOVES.
 *
 * The budget divides the card once, at startup: a KV block nothing holds is VRAM no expert can
 * use, and an expert that does not fit is read across the link on the critical path of every step
 * that routes to it. The cache therefore LENDS what it is not holding, at the addresses it already
 * owns -- nothing is allocated, mapped or released, so the loan moves with no driver call and the
 * card's free VRAM is the same before and after. There is exactly one lender: two owners with two
 * opinions about how much is spare is the same memory in two pools.
 *
 * THE ORDER IS THE WHOLE SAFETY ARGUMENT:
 *
 *   1. the cache says what it can spare -- above the highest block it holds, less its buffer;
 *   2. the cache's mark is set from the larger of that and what every rank's slab ACTUALLY
 *      HOLDS -- a shrink is met as units leave the slots, and reading the offer instead would put
 *      the same blocks in both pools;
 *   3. only then are the slabs told the offer, which they fill into or drain out of.
 *
 * A request that cannot wait for a shrink does not wait for one: the cache calls back into every
 * slab and has the blocks at once (KVManager::set_reclaim, installed in start()). */
void Engine::loan_tick(bool idle) {
    if (!lending_) return;
    const uint64_t now = rad_mono_ns();
    if (now - loan_last_ns_ < 100000000ull) return;    /* 100 ms */
    loan_last_ns_ = now;

    /* UNDER THE SCHEDULER'S LOCK: the loan moves rank 0's live mark and free count, which an HTTP
     * thread's cancel changes under that lock when it frees a sequence, and /metrics reads them
     * under it. */
    const auto lk = sched_.hold();
    Rank& lead = *ranks_[0];

    /* A FINISHED CONVERSATION'S VRAM GOES FIRST, when the slabs have a use for it. An entry whose
     * host copy has landed loses nothing by leaving -- a return restores it from host memory --
     * so while any slab has slots the loan does not yet cover, every such entry is released and
     * its blocks count toward what the cache can spare below. Past that point the VRAM would sit
     * empty in the slab, and the entries stay where a return finds them for free. */
    int64_t ceiling = 0;
    for (auto& r : ranks_) ceiling = std::max(ceiling, r->mover.loan_ceiling(Mover::kLendKV));
    if (lead.kv.loan_tokens() < ceiling) (void)lead.prefix.drop_clean();

    /* WHAT THE CACHE CAN DO WITHOUT, AND WHAT THE SLABS STILL OCCUPY. A loan grows as soon as the
     * cache can spare it -- the slots are filled on the next dispatch or the next idle tick -- but
     * only by at least a prefill chunk, so a cache breathing by a few blocks does not have its
     * boundary chase it. It shrinks at once, and the cache gets the blocks back as the slabs
     * leave them: the cache's mark is never above the deepest slot any rank's slab still holds. */
    const int64_t cur   = lead.kv.loan_tokens();
    const int64_t spare = lead.kv.spare_tokens();
    const int64_t chunk = std::max<int64_t>(cfg_.max_tok, 1);
    int64_t offer = cur;
    if (spare < cur || spare >= cur + chunk) offer = spare;
    int64_t held = 0;
    for (auto& r : ranks_) held = std::max(held, r->mover.loan_held(Mover::kLendKV));

    /* THE CACHE FIRST WHEN THE LOAN GROWS, THE SLABS FIRST WHEN IT SHRINKS. Either way no block is
     * both handed out by the cache and a slot in a slab: a growing loan is lent only once the
     * cache has stopped handing those blocks out, and a shrinking one is returned to the cache
     * only as the slabs leave it. */
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
/* WHICH LEVEL THE STEP RUNS AT, AND WHAT THE SLAB MAY KEEP OF THE ARENA WHILE IT DOES.
 *
 * The rows a step puts through the arena are its tokens and, after it, the drafter's passes --
 * at most one row per drafted position of every sequence. The smallest level serving that many is
 * the one the step runs at; a step larger than every level runs at the full plan, which lends
 * nothing.
 *
 * A LEVEL THAT NEEDS MORE THAN THE SLAB LEFT IT TAKES IT BACK BEFORE THE STEP, synchronously: the
 * units in the slots it needs go back to their host copies, which is a change of address and not
 * a copy, and both streams that can read a slot are drained before this returns. The first chunk
 * of a prefill pays that once; the chunks after it find the memory already theirs, because the
 * loan does not grow again until the steps have been small for a while (arena_tick). */
int Engine::arena_step(const RadBatch* b) {
    if (arena_rows_.size() <= 1 || !b) return RAD_OK;
    /* AN ENCODER PASS TAKES THE FULL PLAN: its activations are sized by the pass, not by a step's
     * rows, and only the full plan holds them (a plugin declares them small under shape_probe). */
    const int64_t need = b->enc ? INT64_MAX : std::max<int64_t>(
        b->n_tok, (int64_t)b->n_seq * (1 + std::max(cfg_.n_spec, 0)));
    int level = 0;
    for (size_t k = 1; k < arena_rows_.size(); ++k)
        if (arena_rows_[k] >= need && (level == 0 || arena_rows_[k] < arena_rows_[(size_t)level]))
            level = (int)k;
    const uint64_t now = rad_mono_ns();
    arena_last_ = arena_rows_[(size_t)level];
    if (arena_last_ >= arena_floor_) { arena_floor_ = arena_last_; arena_floor_ns_ = now; }
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
        /* The full plan lends nothing, so its top -- the staging region -- is the step's. */
        r->stager.set_full_arena(level == 0);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    return RAD_OK;
}

/* THE LOAN GROWS ONLY ONCE THE LARGE STEPS HAVE STOPPED. A prefill is a run of chunks with decode
 * steps between them, and lending the arena out after every chunk would refill the slab across
 * the link and recall it at the next one. So the floor -- the largest step seen -- holds for two
 * seconds after it was last reached, and only then falls to the size of the latest step. */
void Engine::arena_tick(uint64_t now_ns) {
    if (arena_rows_.size() <= 1) return;
    if (arena_last_ < arena_floor_ && now_ns - arena_floor_ns_ > 2000000000ull) {
        arena_floor_ = arena_last_;
        arena_floor_ns_ = now_ns;
    }
    int level = 0;
    for (size_t k = 1; k < arena_rows_.size(); ++k)
        if (arena_rows_[k] >= arena_floor_ &&
            (level == 0 || arena_rows_[k] < arena_rows_[(size_t)level]))
            level = (int)k;
    for (auto& r : ranks_) {
        const int64_t want = r->arena_lent[(size_t)level];
        if (want <= r->mover.loan(Mover::kLendArena)) continue;
        if (rad_dev_set(r->device) < 0) continue;
        (void)r->mover.set_loan(Mover::kLendArena, want);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
}

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
            /* A COPY IS GOOD ONLY IF EVERY RANK'S SHARD LANDED. One that did not leaves an entry
             * whose host copy is part this card's bytes and part stale, which is not restorable
             * anywhere, so the whole batch is a failure and every slot goes back. */
            const std::vector<char> ok(P.copy.size(), good ? 1 : 0);
            {
                const auto lk = sched_.hold();
                lead.tierx.report_pass(P.copy, ok, now);
            }
            P.copy.clear();
        }
    }

    /* ---- reads off disk that have landed --------------------------------------------------- */
    /* THEIR REQUESTS GO NEXT STEP: the records are host records now, held for them, and the
     * admission that asks again restores them out of host memory. */
    {
        TierIO::Fetch f;
        while (P.io.take_fetch(&f)) {
            for (size_t i = 0; i < f.fences.size() && i < ranks_.size(); ++i) {
                if (!f.fences[i]) continue;
                (void)rad_dev_set(ranks_[i]->device);
                rad_event_destroy(f.fences[i]);
            }
            if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
            const auto lk = sched_.hold();
            lead.tierx.report_pass(f.jobs, f.ok, now);
            P.waiting.erase(std::remove(P.waiting.begin(), P.waiting.end(), f.tag), P.waiting.end());
        }
    }

    /* ---- the disk batch the thread finished ------------------------------------------------ */
    {
        std::vector<TierJob> jobs;
        std::vector<char> ok;
        if (P.io.take(&jobs, &ok)) {
            const auto lk = sched_.hold();
            lead.tierx.report_pass(jobs, ok, now);
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

    /* ---- the slow half ---------------------------------------------------------------------- */
    /* AS SOON AS A DISK COPY IS WAITING, not on the clock: until it lands, the entry's host slot
     * cannot be given up for nothing, and a full arena is then a restore that cannot read through
     * it. The thread takes one batch at a time, so this is also as fast as the disk takes them. */
    if (!P.io.busy() && (lead.tiers.disk_waiting() || lead.tiers.starved() ||
                         now - tier_last_ns_ >= 1000000000ull)) {
        tier_last_ns_ = now;
        std::vector<TierJob> jobs;
        lead.tiers.plan(P.slow_budget, &jobs);
        if (!jobs.empty() && !P.io.submit(std::move(jobs))) {
            /* The thread is not running -- no disk tier -- and a plan without one holds only drops,
             * which move nothing. They are reported here. */
            std::vector<char> ok(jobs.size(), 1);
            const auto lk = sched_.hold();
            lead.tierx.report_pass(jobs, ok, now);
        }
    }
}

/* HAND A READ OFF DISK TO THE IO THREAD, fenced on every rank's transfer stream: the slots it
 * writes may have been given up by entries a restore is still copying out of. False when it could
 * not be handed over, and then the reads are reported unstarted -- the records stay on disk and the
 * request is admitted with what the host tier holds. */
bool Engine::submit_fetch(uint64_t tag, std::vector<TierJob>&& jobs) {
    TierPump& P = *pump_;
    TierIO::Fetch f;
    f.tag = tag;
    f.jobs = std::move(jobs);
    int st = RAD_OK;
    for (auto& r : ranks_) {
        RadEvent e = nullptr;
        if (st == RAD_OK) st = r->tierx.fence(&e);
        f.fences.push_back(e);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    if (st == RAD_OK && P.io.submit_fetch(std::move(f))) {
        P.waiting.push_back(tag);
        return true;
    }
    RAD_ERR("kv tiers: a read off disk could not be handed to the IO thread (%s); the request is "
            "admitted with what host memory holds", st < 0 ? rad_strerror(st) : "not running");
    for (size_t i = 0; i < f.fences.size(); ++i) {
        if (!f.fences[i]) continue;
        (void)rad_dev_set(ranks_[i]->device);
        rad_event_destroy(f.fences[i]);
    }
    if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
    const std::vector<char> ok(f.jobs.size(), 0);
    ranks_[0]->tierx.report_pass(f.jobs, ok, rad_mono_ns());
    return false;
}

void Engine::drain_completed() {
    /* TAKEN, NOT READ. The list is appended to by HTTP threads' cancels, so walking it by
     * reference would race them; and nothing else consumes it -- the server learns of a terminal
     * request through its sink and reaps it on its own -- so each id is seen here exactly once and
     * the cost stays proportional to what finished rather than to everything that ever has. */
    sched_.take_completed(&done_);
    for (uint64_t id : done_) for (auto& rk : ranks_) rk->sampler.end_request(id);
}

int Engine::serve() {
    if (ranks_.empty() || !ranks_[0]->ctx.program()) {
        RAD_ERR("serve: start() has not run, or stopped at --debug-graph");
        return RAD_E_STATE;
    }
    if (ranks_[0]->program.logits_buf == RAD_NULL_HANDLE) {
        RAD_ERR("the '%s' plugin declared no logits buffer (rad_declare_logits), so there is "
                "nothing to sample. An encoder-only plugin cannot serve completions.",
                plugins_.arch_plugin.c_str());
        return RAD_E_UNSUPPORTED;
    }

    /* The scheduler drives ONE block manager. Block ids are pure bookkeeping and identical on
     * every rank -- the heads divide equally, so every rank's pool has the same geometry and its
     * free list evolves identically. What differs per rank is the pool BASE, and that lives in
     * KVPoolBinding. So rank 0's manager owns the bookkeeping and the others exist to own their
     * own card's allocation. If the geometries ever disagree the block tables mean different
     * things on different cards, so it is checked rather than assumed. */
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

    /* BACK TO RANK 0 BEFORE THE SCHEDULER IS BUILT. The per-rank loops above left this
     * thread on the LAST rank's device, and the scheduler is given ranks_[0]->ctx.stream()
     * -- so its staging buffers and its staging EVENT would be created on one card and
     * recorded on another card's stream, which fails every step with "could not record
     * the staging event". One batch serves every rank (spec §1), and rank 0 owns it. */
    RAD_TRY(rad_dev_set(ranks_[0]->device));

    kv_iface_ = std::make_unique<KVBinding<KVManager>>(&ranks_[0]->kv);
    pc_iface_ = std::make_unique<PCBinding<PrefixCache>>(&ranks_[0]->prefix);
    {
        auto* pcb = static_cast<PCBinding<PrefixCache>*>(pc_iface_.get());
        pcb->hot_units_fn = [this]() {
            return ranks_.empty() ? std::vector<int32_t>{} : ranks_[0]->heat.active_units(0.5f);
        };
        pcb->on_prefix_hit = [this](const Request& r, const std::vector<int32_t>& units) {
            on_prefix_hit(r, units);
        };
    }
    /* THE RESTORE PATH, installed only when the tiers are on. The scheduler calls this before the
     * admission lookup and must not know what a tier is; the engine does, so the knowledge lives
     * here as a callable rather than as an include in core/sched.
     *
     * EVERY RANK RESTORES ITS OWN SHARD, and rank 0 goes first because its TierExec is the one
     * holding the policy -- it allocates the blocks and reports the job, and the others follow
     * into the same block ids. A rank that fails leaves the prefix shorter and nothing worse,
     * which is why none of this returns an error upward: a miss is always safe. */
    if (ranks_[0]->tierx.enabled()) {
        auto* b = static_cast<PCBinding<PrefixCache>*>(pc_iface_.get());
        /* CALLED FROM INSIDE Scheduler::step(), UNDER ITS LOCK -- which is what makes it safe to
         * touch the cache and rank 0's manager here, and why nothing below may call back into the
         * scheduler. */
        /* A restored session's state needs a device checkpoint slot, and which snapshot gives
         * one up is the cache's decision: it knows which ones a lookup this step handed out. */
        ranks_[0]->tierx.set_checkpoint_room(
            [this] { return ranks_[0]->prefix.free_checkpoint_for_restore(); });
        b->restore_fn = [this](const Request& req) -> IPrefixCache::Restore {
            using R = IPrefixCache::Restore;
            PrefixCache& pc = ranks_[0]->prefix;
            TierPump& P = *pump_;
            if (std::find(P.waiting.begin(), P.waiting.end(), req.id) != P.waiting.end())
                return R::Wait;
            std::vector<BlockHash> chain;
            std::vector<MMContentKey> mk;
            media_keys(req, &mk);
            const std::vector<BlockHash> want = pc.restorable_from(req.prompt, mk, &chain);
            /* THE DISK PART IS READ FIRST, ON THE IO THREAD, while this request waits and every
             * other one runs; the restore below then copies host memory and nothing else. Asked
             * only when something could be on disk or in flight: a chain wholly in VRAM is the
             * common case, and it is thousands of keys. */
            if (!want.empty() || !P.waiting.empty()) {
                std::vector<TierJob> reads;
                if (ranks_[0]->tiers.plan_fetch(chain, &reads, rad_mono_ns())) {
                    if (reads.empty() || submit_fetch(req.id, std::move(reads))) return R::Wait;
                }
            }
            if (want.empty()) return R::Done;
            /* THE WHOLE CHAIN IN ONE RESTORE. A 4-token block makes a long conversation tens of
             * thousands of entries, and one transfer an entry is tens of thousands of host round
             * trips before the turn can start; as one restore it is a launch per (group, layer)
             * per staging batch and a copy per run of arena slots. */
            std::vector<TierJob> jobs;
            jobs.reserve(want.size());
            for (const BlockHash& h : want) {
                TierJob j;
                if (!ranks_[0]->tierx.plan_promote(h, &j)) break;
                jobs.push_back(std::move(j));
            }
            if (jobs.empty()) return R::Done;
            std::vector<char> ok(jobs.size(), 1);
            for (auto& r : ranks_) {
                std::vector<char> mine;
                (void)r->tierx.issue_restore(jobs, r->ctx.stream(), &mine);
                for (size_t i = 0; i < ok.size(); ++i) ok[i] = ok[i] && i < mine.size() && mine[i];
            }
            /* Back to rank 0's card before anything else runs on this thread: the admission this
             * restore is part of grows rank 0's pool next. */
            if (ranks_.size() > 1) (void)rad_dev_set(ranks_[0]->device);
            ranks_[0]->tierx.report_pass(jobs, ok, rad_mono_ns());
            /* EVERY ENTRY THAT CAME BACK IS REBOUND, not only the leading run: the tiers have
             * already marked each one resident at its new blocks, and an entry the cache did not
             * take would name blocks nothing references. */
            const std::vector<int32_t>& cg = pc.cached_groups();
            for (size_t i = 0; i < jobs.size(); ++i) {
                TierJob& j = jobs[i];
                if (!ok[i]) continue;
                const int rb = pc.rebind(j.key, j.blocks);
                if (rb != RAD_OK) {
                    /* A PROMOTION WITH NO ENTRY TO LAND IN. The blocks were reserved for the entry
                     * and nothing else holds them; the record the tiers just marked resident names
                     * them too, and would be released later from under whoever the pool gives them
                     * to. Both are let go, so the blocks return to the pool and the key is simply
                     * unknown. */
                    for (size_t k = 0; k < cg.size() && k < j.blocks.size(); ++k)
                        if (j.blocks[k] >= 0) ranks_[0]->kv.release_block(cg[k], j.blocks[k]);
                    ranks_[0]->tiers.forget(j.key);
                    if (j.ck_dev >= 0) ranks_[0]->kv.free_checkpoint(j.ck_dev);
                    continue;
                }
                /* AND THE RECURRENT HALF, into the slot the restore allocated for it. Without this
                 * the session gets its attention hit back and replays every linear layer from
                 * token zero, which on a long transcript is most of a prefill. NOBODY ELSE OWNS
                 * THAT SLOT UNTIL THE CACHE TAKES IT: if the index entry the snapshot belongs to
                 * was retired, the slot has no path back to the free list. */
                if (j.ck_dev >= 0 && pc.reattach_snapshot(j.key, j.ck_dev) != RAD_OK)
                    ranks_[0]->kv.free_checkpoint(j.ck_dev);
            }
            return R::Done;
        };
    }
    /* EVERY RANK'S CARD, STREAM AND ARENA. The batch is one build shared by pointer (spec §1) and
     * that is right for the host side; the DEVICE side is staged per rank, because a rank reading
     * another card's staging is both a PCIe read on the step path and, worse, unordered against
     * the copies that fill it -- see the Set comment in batch.h. */
    std::vector<BatchBuilder::RankIO> rio;
    rio.reserve(ranks_.size());
    for (auto& rk : ranks_)
        rio.push_back({ rk->device, rk->ctx.stream(), &rk->program });
    RAD_TRY(sched_.init(ranks_[0]->program, cfg_, kv_iface_.get(), pc_iface_.get(), rio, &geo_));
    /* The step staging the budget kept room for exists now, so a checkpoint slot backed from here
     * on leaves the headroom and only what is still to be built on first use. */
    set_checkpoint_floors(std::max<int64_t>(cfg_.gpu_headroom_mib, 0) * (1ll << 20) +
                          late_lazy_bytes_);

    for (auto& r : ranks_) r->sampler.attach_vocab(vocab_view_.get());

    barrier_ = std::make_unique<RankBarrier>(cfg_.tp + 1);
    stopping_.store(false, std::memory_order_release);
    step_status_.store(RAD_OK, std::memory_order_release);
    loop_status_.store(RAD_OK, std::memory_order_release);
    done_.reserve((size_t)cfg_.max_seqs);

    rank_threads_.clear();
    for (int i = 0; i < cfg_.tp; ++i) rank_threads_.emplace_back([this, i] { rank_loop(i); });
    std::thread stepper([this] { step_loop(); });

    int s = cfg_.diagnose_mode() ? run_diagnose() : cfg_.kld() ? run_kld() : run_server();

    /* Stop the step loop before anything the server owns is destroyed: Request::sink is a raw
     * pointer into the server's storage, and a request the scheduler has not finished is a request
     * still writing into it (server.h). */
    stopping_.store(true, std::memory_order_release);
    barrier_->abort();
    if (stepper.joinable()) stepper.join();
    for (auto& t : rank_threads_) if (t.joinable()) t.join();
    rank_threads_.clear();
    /* A step loop that died is the answer, whatever the transport said on the way down: the
     * process should exit with the engine's error, not report a clean shutdown of a server that
     * stopped serving. */
    const int ls = loop_status_.load(std::memory_order_acquire);
    return ls < 0 ? ls : s;
}

/* THE STATE SLOTS A SEQUENCE OWNS, one per KV group, -1 where the group is not stateful.
 *
 * Read from RANK 0 and applied to every rank, which is correct for the same reason the batch hands
 * one `state_index` row to all of them: only rank 0's KVManager tracks sequences (kv_iface_ is
 * bound to it), while every rank's pool has identical geometry, so a slot index means the same
 * offset everywhere and only the base pointer differs. Looking the sequence up in any other rank's
 * manager returns "not found". */
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
    return any;
}

/* ------------------------------------------------------------------ the HTTP frontend */
/* WHAT IS WIRED AND WHAT IS NOT, stated rather than discovered at runtime. `/v1/completions`,
 * `/v1/chat/completions`, `/v1/models`, `/metrics` and `/health` are live. `/v1/embeddings` is
 * 501 (no embedding head in the container), and so are image, video and audio content parts
 * unless the program declared an encoder that accepts them (MEDIA, below).
 *
 * THE CHAT ENDPOINT IS 501 ONLY WHEN THE MODEL HAS NO TEMPLATE, and that distinction is the whole
 * reason `chat_bridge` is conditional below. A container that carries a chat template gets the
 * full path: the template's grammar and lazy triggers reach the sampler, and the reply parser the
 * auto-parser derives from that same template comes back out through IChatTemplate::apply bound
 * to the request that asked for it. A container without one gets an endpoint that says so,
 * because a chat template guessed is a prompt format guessed, and the failure mode of guessing is
 * fluent text in the wrong shape. */
/* WHAT /server_info SAYS ABOUT THE ENGINE.
 *
 * The server cannot derive any of this: it holds interfaces and knows nothing about devices,
 * containers or placement. Rather than growing ServerOptions a field per fact, the engine renders
 * the object it owns and the server splices it in verbatim.
 *
 * IT IS THE OPERATOR'S FIRST QUESTION, asked of a process that is already running: which model,
 * on how many cards, with how much KV, drafting how deeply, over which wire. All of it is printed
 * at startup too -- and a startup log is on a machine, in a file, from a process that may have
 * been restarted since. */
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
    /* What each lookup table was asked for and where it went: auto's answer is the one a
     * dashboard reading this needs, and the request is the one the command line shows. */
    e["embedding_placement"]    = { { "asked", table_placement_name(cfg_.embedding_placement) },
                                    { "resolved", table_placement_name(cfg_.embedding_resolved) } };
    e["ngram_placement"]        = { { "asked", table_placement_name(cfg_.ngram_placement) },
                                    { "resolved", table_placement_name(cfg_.ngram_resolved) } };
    e["prefix_cache"]           = cfg_.prefix_cache;
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

    /* Rank 0's, and every rank declares the same graph -- see the miss report in
     * engine_bringup.cpp for why that is relied on rather than assumed. */
    const Program& p0 = ranks_[0]->program;
    e["ops"]     = (int64_t)p0.ops.size() - 1;
    e["weights"] = (int64_t)p0.weights.size() - 1;

    nlohmann::ordered_json kvs = nlohmann::ordered_json::array();
    const KVManager& kv = ranks_[0]->kv;
    for (int32_t g = 0; g < kv.n_groups(); ++g) {
        const KVGroupPlan* pl = kv.plan(g);
        /* Group 0 is the "no KV group" sentinel every program carries so that a handle of zero is
         * absent rather than valid. It has no layers and no pool, and reporting it would be
         * reporting the absence of a thing. */
        if (!pl || pl->name.empty() || pl->n_layers <= 0) continue;
        nlohmann::ordered_json k;
        k["name"]     = pl->name;
        k["kind"]     = pl->kind == RAD_KV_FULL   ? "full"
                      : pl->kind == RAD_KV_WINDOW ? "window"
                      : pl->kind == RAD_KV_LINEAR ? "linear"
                      : pl->kind == RAD_KV_CONV   ? "conv" : "none";
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
 * SIGINT AND SIGTERM ARE HANDLED, so a serving process has a way to end other than being killed
 * where it stands. Unhandled, `~Engine` never runs: `--profile-ops` collects a table nobody
 * prints, the placement banner is unreachable by construction, and an in-flight request is cut
 * rather than drained.
 *
 * A WATCHER THREAD AND NOT THE HANDLER ITSELF. `Server::stop` takes locks, joins threads and
 * drains a graveyard; none of that is async-signal-safe, and calling it from a handler is the
 * deadlock that only shows up under load. The handler does the one thing it may -- store to a
 * lock-free atomic -- and this thread does the work. A second signal is left to the default
 * disposition, so an operator who has decided not to wait still gets an immediate kill.
 */
namespace {
std::atomic<bool> g_signalled{false};
void on_signal(int) { g_signalled.store(true, std::memory_order_relaxed); }
}  /* namespace */

/* ================================================================== sampling defaults */
/* WHAT A REQUEST THAT SENDS NO SAMPLER SETTINGS IS SERVED WITH.
 *
 * The reading and the range checks belong to core/gen_config.h, which carries the argument for
 * why any of this exists. What is here is the ORDER, and the container lookup, which is the one
 * part that reads the model's metadata:
 *
 *   --temp / --top-k / --top-p / --min-p    an operator overriding, field by field, and last
 *   --generation-config PATH                a generation_config.json named outright
 *   <model>.generation.json                 one placed beside the container
 *   the model's `generation.*` metadata     generation_config.json, as the container or the
 *                                          checkpoint carries it
 *   SamplingParams' own defaults            the untruncated floor
 *
 * Each source is consulted only if the one above it said nothing at all, so a deployment that
 * names a file gets that file's sampler and not a blend of it with the container's. */
SamplingParams resolve_sampling_defaults(const RadModelMeta& meta, bool container,
                                         const Config& cfg) {
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

    /* THE REST OF THE SAMPLER HAS NO FILE SOURCE: a generation_config.json is read for the four
     * fields above and no others, so these are SamplingParams' own values unless a flag sets
     * them, and request_overrides() logs the ones that did. */
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

namespace {
/* "presence_penalty 1.5, max_tokens auto, ..." -- every request default and bound the command line
 * moved, under the request's spelling of the field, or "" when it moved none. The four fields
 * resolve_sampling_defaults describes are not repeated. */
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
int Engine::run_server() {
    /* The rank-0 sampler comes along so admission can refuse a request whose sampling stage this
     * build has no kernel for, by name and at the door (IScheduler::admit). The tiers come along
     * because they are where a stored conversation is RECORDED, and that is passed whether or not
     * any tier can move anything: with the caps at zero the object still holds the session table,
     * and withholding it on that ground would leave the sessions view reporting an empty server. */
    server::SchedulerBridge sched_bridge(&sched_, ranks_.empty() ? nullptr : &ranks_[0]->sampler,
                                        ranks_.empty() ? nullptr : &ranks_[0]->tiers);
    server::TokenizerBridge tok_bridge(vocab_);
    server::ChatTemplateBridge chat_bridge;
    server::GrammarBridge grammar_bridge(vocab_view_.get());

    /* THE SNAPSHOT OF WHAT ONLY THE ENGINE CAN SEE, and there is ONE of it. The live view
     * (spec §16) and the web dashboard are two renderers of the same numbers, so they are given
     * the same function rather than each reaching into the engine on its own -- two readers is
     * two chances to disagree about what the engine was doing.
     *
     * Called from whichever thread is drawing, and it must not be able to slow the engine: it
     * reads the scheduler's published copy (Scheduler::publish), the driver's VRAM counter
     * through a descriptor opened here, and rank 0's plan, which is immutable after start().
     *
     * THE CARDS' FIXED FACTS ARE READ ONCE, HERE, on the engine thread before serving. The props
     * call enters the device runtime -- device properties, a device switch, a free-memory query,
     * a peer walk -- and is meant for startup; it also opens the counter dev_vram_used reads. */
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
            /* vram_used is sampled per call rather than cached: it is the number that moves, and
             * it is the one an operator watching for a leak is watching. A card whose counter
             * cannot be read is left out rather than drawn empty. Utilisation, power and
             * temperature stay 0 -- "not sampled" -- because the device layer does not expose
             * them and a plausible invented number is worse than a blank. */
            int64_t used = 0;
            if (dev_vram_used(f.index, &used) != RAD_OK) continue;
            server::CardStat c = f;
            c.vram_used = used;
            s.cards.push_back(std::move(c));
        }

        /* THE EXPERT PLANE, from rank 0's placement plan. A dense model has no expert units, so
         * this stays empty and both renderers omit the panel -- which is why it is filled
         * unconditionally rather than behind a "is this MoE" test the engine would have to
         * invent. When there are experts, this is the picture the heat engine is judged by. */
        s.experts = server::ExpertPlane{};
        if (!ranks_.empty()) {
            const Plan& p = ranks_[0]->plan;
            /* Read ONCE for both halves of this panel: the same struct answers "what did the
             * mover do to the experts" and "what did it put on the link", and two reads is two
             * chances for the two halves of one picture to disagree. */
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
                /* THE LIVE TIER WHERE THERE IS ONE, and the plan's otherwise. The plan is where a
                 * unit STARTED; once the heat engine runs, the picture this panel exists to show
                 * is the one that moves, and reading the plan would draw a frozen plane under an
                 * engine doing nothing but change it. HeatEngine::tier is a plain vector
                 * read and the writes are made by the dispatch thread at a step boundary, so a
                 * draw racing one gets the value from either side of a single store. */
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
                    /* RESIDENCY IS COUNTED OVER THE UNITS, NOT OVER THE PLANE. The plane is a
                     * rectangle and a model need not fill it -- a draft head's expert count can
                     * differ from the trunk's -- so tallying cells would count
                     * (layer, expert) pairs that were never placed as VRAM-resident, which is the
                     * one direction the number must not be wrong in. */
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
                s.experts.flex_bytes    = ranks_[0]->mover.flex_used_bytes();
                s.experts.flex_capacity = ranks_[0]->mover.flex_offered_bytes();
                /* THE HEAT ENGINE'S OWN TALLY, which answers a different question from the
                 * mover's. The mover counts bytes that went; these count decisions -- how many
                 * dispatches proposed nothing and why, and whether a promotion was of a unit
                 * never promoted before (investment) or one just demoted (pure loss). */
                const HeatStats hs = heat.stats();
                s.experts.dispatches        = (int64_t)hs.dispatches;
                s.experts.refused           = (int64_t)hs.refused;
                s.experts.no_candidate      = (int64_t)hs.no_candidate;
                s.experts.distinct_promoted = (int64_t)hs.distinct_promoted;
                s.experts.readmits          = (int64_t)hs.readmits;
                /* AND WHETHER THE PLACEMENT IS ACTUALLY PAYING. Residency is a property of the
                 * plan; the hit rate is a property of the TRAFFIC, and a plan holding 80% of the
                 * units can still miss on most tokens if the hot ones are the ones outside. */
                s.experts.routed            = (int64_t)hs.routed;
                s.experts.routed_resident   = (int64_t)hs.routed_resident;

            }

            /* THE LINK, and the two things that share it.
             *
             * THE COLLECTIVE IS ONE RANK'S AND THE MOVES ARE EVERY RANK'S, which is why the two
             * halves of this panel are summed differently. The ranks are symmetric in the
             * all-reduce -- each pushes the same message -- so rank 0's count IS the machine's and
             * summing it would double it. A promotion, a demotion and a streamed expert are the
             * opposite: each rank holds its own shard of the plane and pulls its own bytes over
             * its own link, so the machine's traffic is the sum and rank 0 alone is a half of it
             * at --tp 2. Reading one rank for all of them would put the move and stream rows at
             * 1/world of the truth while the all-reduce beside them stayed right, which is the
             * worse of the two errors: the SHARES would be wrong, and the shares are the whole
             * point of the panel.
             *
             * `world` is published so the reader can still recover a per-card figure, which is the
             * one to hold against a single link's ceiling. */
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

    /* load_from_vocab warns on its own when the template is missing or will not compile; all we
     * decide here is whether the endpoint exists. The reply format is the architecture plugin's
     * when it declares one, and the template's otherwise. */
    const RadChatFormat* declared_format = rad_tools_chat_format(plugins_, meta_);
    if (vocab_ && !vocab_->chat_template().empty() &&
        chat_bridge.init(*vocab_, declared_format, plugins_.arch_plugin) >= 0 &&
        chat_bridge.ready()) {
        d.chat = &chat_bridge;
    } else {
        RAD_WARN("no chat template in this container; /v1/chat/completions will answer 501 and "
                 "/v1/completions is the endpoint to use");
    }

    server::ServerOptions o{};
    o.model_id = !cfg_.served_model_name.empty() ? cfg_.served_model_name
               : meta_.name && *meta_.name       ? meta_.name : "radiance";
    o.host     = cfg_.host;
    o.port     = cfg_.port;
    o.api_key  = cfg_.api_key;
    o.max_seqs = cfg_.max_seqs;
    /* `n` fans one request out over the batch, so a server that runs more sequences a step takes
     * more choices a request; the floor keeps a small batch from refusing what it would queue. */
    o.max_n    = cfg_.max_n > 0 ? (int)cfg_.max_n : (int)std::max<int64_t>(o.max_n, cfg_.max_seqs);
    o.max_ctx  = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
    o.default_reasoning_effort = cfg_.reasoning_effort;
    o.default_thinking_budget = cfg_.thinking_token_budget;
    o.default_sampling = resolve_sampling_defaults(meta_, file_ != nullptr, cfg_);
    o.default_dry_breakers_set = cfg_.sample_dry_seq_breakers.has_value();
    /* -1 leaves ServerOptions' own default, which is `auto` (0). That is resolved per request
     * against the context, so it needs one to resolve against. */
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
    o.n_threads       = cfg_.http_threads;
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
    /* Per request, not once: the dump carries OpInfo::issued, which only becomes true once the
     * run phase has run. See ServerOptions::graph_json. */
    o.graph_json  = [this] { return dump_graph_json(ranks_[0]->program); };

    /* MEDIA, WHEN THE PROGRAM DECLARED AN ENCODER. The processor reads its geometry from the
     * container and the encoder states the one it was built for; the two meet at the patch width
     * and the merge, and a disagreement is refused here rather than discovered as garbage rows. */
    std::unique_ptr<mm::Processor> mm_proc;
    std::unique_ptr<server::MultimodalBridge> mm_bridge;
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
        /* THE COMMAND LINE OVER THE CONTAINER, field by field, and the result checked as a whole:
         * a minimum stated alone can land above the maximum the container states. */
        if (cfg_.image_min_pixels)       vc.image_min_pixels  = *cfg_.image_min_pixels;
        if (cfg_.image_max_pixels)       vc.image_max_pixels  = *cfg_.image_max_pixels;
        if (cfg_.video_min_pixels)       vc.video_min_pixels  = *cfg_.video_min_pixels;
        if (cfg_.video_max_pixels)       vc.video_max_pixels  = *cfg_.video_max_pixels;
        if (cfg_.video_fps)              vc.fps               = *cfg_.video_fps;
        if (cfg_.video_min_frames)       vc.min_frames        = *cfg_.video_min_frames;
        if (cfg_.video_max_frames)       vc.max_frames        = *cfg_.video_max_frames;
        if (cfg_.video_max_frame_tokens) vc.max_frame_rows    = *cfg_.video_max_frame_tokens;
        if (cfg_.max_source_pixels)      vc.max_source_pixels = *cfg_.max_source_pixels;
        if (cfg_.media_flags_set() && vc.check(&why) < 0) {
            RAD_ERR("media: the media flags and what the container states disagree: %s",
                    why.c_str());
            return RAD_E_INVAL;
        }
        /* The processor lowers the image band to one encoder pass; an operator who asked for
         * more is told, rather than finding the cap in the startup line below. */
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
        mm_proc = std::make_unique<mm::Processor>();
        RAD_TRY(mm_proc->init(vc, [&tok_bridge](const std::string& t) {
            return tok_bridge.encode(t, /*add_special=*/false, /*parse_special=*/false);
        }));
        mm_bridge = std::make_unique<server::MultimodalBridge>(mm_proc.get(), encd.modalities);
        d.mm = mm_bridge.get();
        o.allow_image = (encd.modalities & RAD_MM_IMAGE) != 0;
        o.allow_video = (encd.modalities & RAD_MM_VIDEO) != 0;
        if (!mm::decoder_available())
            RAD_WARN("media: the '%s' encoder is loaded but this build has no media decoder "
                     "(RAD_WITH_FFMPEG), so every media part will be refused",
                     ranks_[0]->program.encoder_name.c_str());
        else
            RAD_INFO("media: %s%s through '%s'; an image resizes to at most %lld pixels (%lld "
                     "patches a pass), a video to %lld pixels over all its frames",
                     o.allow_image ? "images" : "", o.allow_video ? " and video" : "",
                     ranks_[0]->program.encoder_name.c_str(),
                     (long long)mm_proc->config().image_max_pixels, (long long)encd.max_patches,
                     (long long)mm_proc->config().video_max_pixels);
    }

    if (encd.modalities == 0 && cfg_.media_flags_set())
        RAD_WARN("media: the media flags do nothing here -- this model %s",
                 cfg_.mm_max_patches == 0 ? "is served text only (--mm-max-patches 0)"
                                          : "takes no images or video");

    server::Server srv(d, o);

    /* A TEMPLATE DEFAULT THE TEMPLATE REFUSES is a 400 on every chat request that leaves it to
     * the default, found by the first caller rather than the operator. --chat-template-kwargs
     * stops startup; --reasoning-effort only warns, because a deployment whose clients all send
     * their own effort serves correctly with one the template would refuse. */
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
    /* A BUDGET THE MODEL CANNOT HOLD STOPS STARTUP, as --chat-template-kwargs does: the default
     * would apply to no request at all, and the operator who set it would not find out. */
    if (const std::string why = srv.check_thinking_budget(); !why.empty()) {
        RAD_ERR("--thinking-token-budget %lld: %s", (long long)cfg_.thinking_token_budget,
                why.c_str());
        return RAD_E_INVAL;
    }
    RAD_INFO("serving on http://%s:%d  (/v1/completions%s)%s", o.host.c_str(), o.port,
             d.chat ? "; /v1/chat/completions with tools and reasoning" : "; chat is 501",
             o.api_key.empty() ? "" : "  -- a bearer key is required");

    /* THE LIVE VIEW (spec §16), constructed here and nowhere else.
     *
     * `core/server/liveview.{h,cpp}` is the console's view of a running engine, and the
     * throughput heartbeat the server otherwise has no form of: nothing here logs a request,
     * and everything else the engine prints happens at startup, so a loop that stalls looks
     * exactly like a loop with nothing to do.
     *
     * Off by default, because §16 opens with "everything below is off by default". Redirected, it
     * prints one greppable key=value line per interval and no escape sequences at all -- which is
     * the property that makes it usable under nohup, and the reason the renderer is chosen once by
     * isatty() rather than per draw.
     *
     * The snapshot function is the only part the engine owns: LiveView holds no data and reads no
     * device. Held by value in this scope so its thread is joined before Deps goes out of it. */
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
    std::thread watcher([&] {
        while (watching.load(std::memory_order_relaxed)) {
            if (g_signalled.load(std::memory_order_relaxed)) {
                RAD_INFO("stopping: draining in-flight requests");
                srv.stop();
                return;
            }
            /* THE STEP LOOP HAS DIED, so nothing will ever run another step. Every request it
             * held has already been failed and answered (Engine::fail_loop); what is left is to
             * stop taking new ones, which a transport left up would accept into a queue nobody
             * drains. */
            if (loop_status_.load(std::memory_order_acquire) < 0) {
                RAD_ERR("stopping: the step loop failed with %s",
                        rad_strerror(loop_status_.load(std::memory_order_acquire)));
                srv.stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    const int rc = srv.run();
    /* The loop tests `watching` FIRST, so clearing it ends the watcher WITHOUT a second stop: a
     * normal return from run() has already torn the transport down. It costs one sleep interval
     * to join, which is the price of not polling a condition variable nobody else waits on. */
    watching.store(false, std::memory_order_relaxed);
    watcher.join();
    if (live) live->stop();
    return rc;
}

}  /* namespace rad */
