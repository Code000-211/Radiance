/* engine_draft.cpp -- the two drafter drivers (spec §10).
 *
 * A draft head runs as EXTRA PASSES behind the sampled trunk step, and there are two shapes:
 * SERIAL runs `depth` dependent passes; BLOCK runs a context pass and one pass that fills the
 * whole block. Stateless block queries can be issued before commit, so their GPU work overlaps
 * the host's acceptance and commit. Stateful block passes are issued after commit.
 *
 * Split out of the step loop's file because both drivers publish their proposals after commit.
 * See engine_priv.h.
 */
#include "engine_priv.h"

#include "sched/scheduler.h"
#include "sched/tree_state.h"
#include "device/device.h"

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#else
#include "../arch/common/rad_tree_candidate.h"
#endif
#include "../arch/common/rad_mtp_tree_config.h"
#include "../arch/common/rad_spechub_config.h"
#include "../mine/spechub.h"   /* rad_mtp_couple_state_valid: the kernels' own validator */
#define RAD_DLOOP_LOADER_ONLY
#include "../arch/common/rad_block_dloop.h"
#undef RAD_DLOOP_LOADER_ONLY

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the card's step state */
/* Every rank's StepSlots, zeroed, and its sorted end-of-generation ids; and the pinned host copy
 * of rank 0's slots that the step reads back. Once, at start: nothing on the step path allocates. */
int Engine::step_state_init() {
    // This driver only publishes proposals declared by the architecture plugin.
    const char* sidecar = std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
    if (sidecar && *sidecar) {
        if (drafter_.kind != RAD_DRAFT_BLOCK || drafter_.proposal == 0) {
            RAD_ERR("RADIANCE_FULLDRAFTER_SIDECAR is set, but no block drafter with active proposals "
                    "was declared (kind=%d, proposal=%d); refusing to silently substitute another drafter",
                    (int)drafter_.kind, (int)drafter_.proposal);
            return RAD_E_INVAL;
        }
        RAD_INFO("RADIANCE_FULLDRAFTER_SIDECAR connected: drafter '%s' (kind=%d, depth=%d, block=%lld, proposal=%d)",
                 drafter_.name, (int)drafter_.kind, drafter_.depth, (long long)drafter_.block, (int)drafter_.proposal);
    }
    std::vector<int32_t> eog;
    if (vocab_) eog = vocab_->eog();
    std::sort(eog.begin(), eog.end());
    eog.erase(std::unique(eog.begin(), eog.end()), eog.end());
    const int64_t n_slot = cfg_.max_seqs > 0 ? cfg_.max_seqs : 1;
    spechub_ = spechub_enabled();
    if (!spechub_config_valid()) return RAD_E_INVAL;
    if (spechub_) {
        const char* no_lookup = std::getenv("RADIANCE_NO_PROMPT_LOOKUP");
        if (drafter_.kind != RAD_DRAFT_SERIAL || !drafter_.name ||
            std::strcmp(drafter_.name, "mtp") || cfg_.n_spec != 2 || cfg_.adaptive_spec ||
            ranks_.size() != 2 || !no_lookup || std::strcmp(no_lookup, "1") ||
            std::getenv("RADIANCE_NO_DRAFT") || std::getenv("RADIANCE_DRAFT_RAW")) {
            RAD_ERR("SpecHub requires TP2 serial MTP, n_spec=2, adaptive off, "
                    "RADIANCE_NO_PROMPT_LOOKUP=1 and active unmodified proposals");
            return RAD_E_UNSUPPORTED;
        }
        for (size_t i = 1; i < ranks_[0]->program.buffers.size(); ++i) {
            const std::string& name = ranks_[0]->program.buffers[i].name;
            const auto ends = [&](const char* suffix) {
                const size_t n = std::strlen(suffix);
                return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
            };
            if (ends("mtp_spechub_draws")) spechub_draw_buf_ = (rad_buf)i;
            if (ends("mtp_spechub_state")) spechub_state_buf_ = (rad_buf)i;
        }
        if (!spechub_draw_buf_ || !spechub_state_buf_ ||
            !drafter_tree_declared(ranks_[0]->program) ||
            ranks_[0]->program.drafter_tree_capacity != 3) {
            RAD_ERR("SpecHub requested but the architecture declared no live k2 proposal");
            return RAD_E_UNSUPPORTED;
        }
        RAD_INFO("SpecHub LIVE verifier=%s: GPU Q proposal + filtered-target sampling; one position, "
                 "two siblings; separate proposal/tail/transport/residual RNG streams",
                 spechub_transport() ? "transport" : "match-control");
    }
    mtp_couple_ = mtp_coupled_enabled();
    if (!mtp_coupled_config_valid()) return RAD_E_INVAL;
    if (mtp_couple_) {
        /* Drafts reach the sampler only through MTP's set_draft: a prompt-lookup proposal could
         * coincide with a sampled x and inherit a Q it was never drawn from. */
        const char* no_lookup = std::getenv("RADIANCE_NO_PROMPT_LOOKUP");
        if (drafter_.kind != RAD_DRAFT_SERIAL || !drafter_.name || std::strcmp(drafter_.name, "mtp") ||
            ranks_.empty() || drafter_tree_declared(ranks_[0]->program) || !drafter_.proposal ||
            !no_lookup || std::strcmp(no_lookup, "1")) {
            RAD_ERR("RADIANCE_MTP_COUPLED requires active linear serial MTP drafting and "
                    "RADIANCE_NO_PROMPT_LOOKUP=1 (RADIANCE_NO_DRAFT must be unset)");
            return RAD_E_UNSUPPORTED;
        }
        const auto& bufs = ranks_[0]->program.buffers;
        for (size_t i = 1; i < bufs.size(); ++i) {
            const std::string& name = bufs[i].name;
            const auto ends = [&](const char* suffix) {
                const size_t k = std::strlen(suffix);
                return name.size() >= k && name.compare(name.size() - k, k, suffix) == 0;
            };
            if (ends("mtp_couple_draws")) couple_draw_buf_ = (rad_buf)i;
            if (ends("mtp_couple_state")) couple_state_buf_ = (rad_buf)i;
        }
        const RadBufDecl* dd = couple_draw_buf_ ? &bufs[couple_draw_buf_].decl : nullptr;
        const RadBufDecl* sd = couple_state_buf_ ? &bufs[couple_state_buf_].decl : nullptr;
        if (!dd || !sd || dd->rank != 2 || dd->shape[0] < n_slot ||
            dd->shape[1] != (int64_t)RAD_MTP_COUPLE_DRAW_WORDS || sd->rank != 3 ||
            sd->shape[0] < drafter_.depth || sd->shape[1] < n_slot ||
            sd->shape[2] != (int64_t)RAD_MTP_COUPLE_STATE_WORDS) {
            RAD_ERR("RADIANCE_MTP_COUPLED requested but the architecture declared no coupled MTP "
                    "draw/state planes for %d rounds", drafter_.depth);
            return RAD_E_UNSUPPORTED;
        }
        couple_rows_ = sd->shape[1];
        RAD_INFO("MTP coupled LIVE: %d linear round(s), x~Q from the final reranked top8, maximal "
                 "coupling in the target sampler (FP64 law; not seed-identical to the ordinary pick)",
                 drafter_.depth);
    }
    const int64_t slot_bytes = n_slot * (int64_t)sizeof(StepSlot);
    if (spechub_) {
        spechub_by_card_.resize(static_cast<size_t>(n_slot));
        spechub_draw_stage_ = static_cast<RadSpechubDraw*>(
            rad_dev_alloc(n_slot * sizeof(RadSpechubDraw), RAD_MEM_HOST_PINNED));
        spechub_readback_ = static_cast<RadSpechubState*>(
            rad_dev_alloc(n_slot * sizeof(RadSpechubState), RAD_MEM_HOST_PINNED));
        if (!spechub_draw_stage_ || !spechub_readback_) return RAD_E_NOMEM;
    }
    if (mtp_couple_) {
        couple_by_card_.assign(static_cast<size_t>(n_slot), RadMtpCoupleDraw{});
        couple_draw_stage_ = static_cast<RadMtpCoupleDraw*>(
            rad_dev_alloc(n_slot * sizeof(RadMtpCoupleDraw), RAD_MEM_HOST_PINNED));
        couple_readback_ = static_cast<RadMtpCoupleState*>(rad_dev_alloc(
            (int64_t)drafter_.depth * couple_rows_ * (int64_t)sizeof(RadMtpCoupleState),
            RAD_MEM_HOST_PINNED));
        if (!couple_draw_stage_ || !couple_readback_) return RAD_E_NOMEM;
    }
    /* Tree publication uses fixed-size staging allocated before the first runtime step. */
    if (!ranks_.empty() && drafter_tree_declared(ranks_[0]->program)) {
        block_tree_out_.reserve(static_cast<size_t>(n_slot) *
                                static_cast<size_t>(1 + 2 * ranks_[0]->program.drafter_tree_capacity));
        block_rows_.resize(static_cast<size_t>(n_slot));
        draft_out_.reserve(static_cast<size_t>(n_slot)*16);
        tree_chain_notice_.assign(static_cast<size_t>(n_slot),0);
        tree_exact_to_=tree_qsa_exact_to(ranks_[0]->program);
    }
    if (!ranks_.empty() && drafter_.kind == RAD_DRAFT_SERIAL &&
        drafter_tree_declared(ranks_[0]->program)) {
        const Program& p = ranks_[0]->program;
        if (!p.drafter_tree_buf || p.drafter_tree_capacity < 2) {
            RAD_ERR("drafter '%s': a serial tree requires a packed device buffer and capacity >= 2",
                    drafter_.name);
            return RAD_E_INVAL;
        }
        if (!std::strcmp(drafter_.name, "mtp") && mtp_tree_config().error) {
            RAD_ERR("%s", mtp_tree_config().error);
            return RAD_E_INVAL;
        }
        if (mtp_tree_config().pruned()) {
            if (std::strcmp(drafter_.name, "mtp") || drafter_.depth != cfg_.n_spec ||
                p.drafter_tree_capacity != 8 || cfg_.n_spec != 7 || cfg_.adaptive_spec ||
                cfg_.max_tok < 8) {
                RAD_ERR("MTP pruned requires declared capacity7, five actual rounds, adaptive drafting off, "
                        "and room for eight frontier rows");
                return RAD_E_UNSUPPORTED;
            }
            serial_pruned_ = true;
            pruned_tile_ = (int)std::min<int64_t>(cfg_.max_seqs, cfg_.max_tok / 8);
        }
        serial_tree_ = true;
        serial_row_.resize(static_cast<size_t>(n_slot));
        if (!readback_.reserve(n_slot * (1 + 2 * p.drafter_tree_capacity)))
            return RAD_E_NOMEM;
    }
    if (mtp_tree_config().pruned() && !serial_pruned_) {
        RAD_ERR("MTP pruned was requested but no compatible serial MTP tree was declared");
        return RAD_E_UNSUPPORTED;
    }
    for (auto& rk : ranks_) {
        RAD_TRY(rad_dev_set(rk->device));
        rk->step_slots = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_DEVICE));
        rk->n_eog = (int32_t)eog.size();
        rk->eog = static_cast<int32_t*>(
            rad_dev_alloc(std::max<int64_t>(1, (int64_t)eog.size()) * 4, RAD_MEM_DEVICE));
        if (!rk->step_slots || !rk->eog) {
            RAD_ERR("step state: no device memory for %lld slot(s) on card %d",
                    (long long)n_slot, rk->device);
            return RAD_E_NOMEM;
        }
        RAD_TRY(rad_memset_async(rk->step_slots, 0, slot_bytes, rk->ctx.stream()));
        if (!eog.empty())
            RAD_TRY(rad_memcpy_async(rk->eog, eog.data(), (int64_t)eog.size() * 4,
                                     rk->ctx.stream()));
        RAD_TRY(rad_stream_sync(rk->ctx.stream()));
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0]->device));
    slots_host_ = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_HOST_PINNED));
    if (!slots_host_) return RAD_E_NOMEM;
    RAD_TRY(rad_event_create_as(&step_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    RAD_TRY(rad_event_create_as(&trunk_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));

    radiance::TreeRegistry::instance().clear_all();
    return RAD_OK;
}

void Engine::step_state_fini() {
    for (auto& rk : ranks_) {
        (void)rad_dev_set(rk->device);
        if (rk->step_slots) rad_dev_free(rk->step_slots, RAD_MEM_DEVICE);
        if (rk->eog) rad_dev_free(rk->eog, RAD_MEM_DEVICE);
        rk->step_slots = nullptr;
        rk->eog = nullptr;
    }
    if (slots_host_) rad_dev_free(slots_host_, RAD_MEM_HOST_PINNED);
    if (spechub_draw_stage_) rad_dev_free(spechub_draw_stage_, RAD_MEM_HOST_PINNED);
    if (spechub_readback_) rad_dev_free(spechub_readback_, RAD_MEM_HOST_PINNED);
    if (couple_draw_stage_) rad_dev_free(couple_draw_stage_, RAD_MEM_HOST_PINNED);
    if (couple_readback_) rad_dev_free(couple_readback_, RAD_MEM_HOST_PINNED);
    spechub_draw_stage_ = nullptr;
    spechub_readback_ = nullptr;
    couple_draw_stage_ = nullptr;
    couple_readback_ = nullptr;
    slots_host_ = nullptr;
    if (trunk_end_) rad_event_destroy(trunk_end_);
    trunk_end_ = nullptr;
    if (step_end_) rad_event_destroy(step_end_);
    step_end_ = nullptr;
    radiance::TreeRegistry::instance().clear_all();
}

/* ------------------------------------------------------------------ the serial driver
 *
 * `RAD_DRAFT_SERIAL`: one forward pass of the head per drafted token, after the trunk step. Round 1
 * consumes the token the sampler chose and the hidden state of the position that token came from,
 * and which position that is depends on how many drafts the verify accepted. Its pass is the
 * history pass, which the head runs anyway and whose rows already include that position
 * (RadBatch::draft_out_ids).
 *
 * Linear mode issues acceptance, history and rounds before host commit. Serial-tree mode runs
 * acceptance on the current trunk batch too, but delays the same history/round driver until target
 * winner compaction and host commit finish. Head KV can share the target's compacted group: history
 * must overwrite the moved scratch at every newly committed position before a head query reads it.
 * Its tokens and winner tap rows are then host-known; subsequent rounds still use the same slots.
 *
 * Every pass is an ordinary step: a RadBatch run through the same barrier and the same sampler
 * chain. What makes it a draft pass is `draft_pass`, which the architecture plugin reads to run
 * the head instead of the trunk.
 *
 * The tokens go out through set_draft(), the same door the prompt-lookup drafter uses, so the
 * next step cannot tell them apart -- and a sequence the head declined leaves n_draft at 0,
 * which is exactly the case where the n-gram drafter gets its turn.
 *
 * NOTHING HERE NAMES A MODEL. An MTP head has this shape, and an EAGLE or Medusa head declares
 * RAD_DRAFT_SERIAL and runs on it unchanged.
 */
bool Engine::serial_drafting() const {
    /* THE BISECT KNOB COVERS BOTH DRIVERS. RADIANCE_NO_DRAFT means "propose nothing" -- it is
     * how a speculative deployment is compared against itself with the drafts taken out, which
     * separates "is the deferred state right" from "is the rollback right". */
    static const bool no_draft = std::getenv("RADIANCE_NO_DRAFT") != nullptr;
    return drafter_.kind == RAD_DRAFT_SERIAL && drafter_.depth > 0 && !no_draft;
}

/* One advance on every rank, against the build the scheduler just produced -- over its first
 * `rows` rows when that is given, and all of them otherwise. */
int Engine::advance_all(int mode, int round, bool from_proposal, int rows) {
    for (size_t i = 0; i < ranks_.size(); ++i) {
        Rank& rk = *ranks_[i];
        AdvanceArgs a = sched_.advance_args((int)i);
        if (rows >= 0 && rows < a.n_seq) a.n_seq = rows;
        a.mode  = mode;
        a.round = round;
        a.st    = rk.step_slots;
        a.eog   = rk.eog;
        a.n_eog = rk.n_eog;
        if (from_proposal) {
            a.src = static_cast<const int32_t*>(drafter_.proposal ? rk.ctx.buf_ptr(drafter_.proposal)
                                                                  : rk.ctx.buf_ptr(rk.tok_buf));
            a.src_pitch = (drafter_.proposal && drafter_.proposal_pitch > 0)
                              ? drafter_.proposal_pitch : 1;
        } else {
            a.src = static_cast<const int32_t*>(rk.ctx.buf_ptr(rk.tok_buf));
        }
        if (!a.src || !a.st) {
            RAD_ERR("drafter '%s': rank %zu has no %s to advance from", drafter_.name, i,
                    a.st ? "proposal buffer" : "step state");
            return RAD_E_STATE;
        }
        RAD_TRY(step_advance(a, rk.ctx.stream()));
    }
    return RAD_OK;
}

/* The serial passes: pre-commit for a linear proposal, post-commit for a tree proposal.
 * Tree acceptance was already issued against the trunk batch, before collect_step changed its
 * drafts and compacted its target KV. Re-running it here would use the wrong batch and inputs. */
int Engine::issue_serial_drafter(int step, const RadBatch* b, StepOut& so) {
    serial_rounds_ = 0;
    serial_rows_   = 0;
    serial_done_   = 0;
    /* A step that may be followed by one issued ahead needs the card's acceptance whether or not
     * anything drafts: that step's rows continue from it. */
    if (!serial_drafting()) {
        if (pipelining_) RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
        return RAD_OK;
    }

    if (!serial_tree_) {
        RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
        if (!so.rb) RAD_TRY(collect_issue(b, so));
    }

    /* The declaration must match max_spec; pruned's actual transformer depth is separate. */
    int rounds = spechub_ ? 1 : serial_pruned_ ? MtpTreeConfig::depth : drafter_.depth;
    serial_row_.resize((size_t)cfg_.max_seqs);
    /* `n` counts the rows that can DRAFT. It can be zero while the history pass still has work:
     * a sequence three chunks into a long prompt has hidden states the head's attention will need
     * and no token to propose from yet. */
    const int n = sched_.serial_begin(&rounds, serial_row_.data(), (int)serial_row_.size(),
                                      serial_tree_);
    if (n < 0) return n;
    if (rounds <= 0) return RAD_OK;
    /* Tree row capacity and serial conditioning depth are independent. The MTP
     * experiment shares its startup bound with the architecture's history planes;
     * all other serial tree drafters retain the existing three-round limit. */
    const int tree_rounds = !std::strcmp(drafter_.name, "mtp") ? mtp_tree_config().rounds : 3;
    if (serial_tree_ && rounds > tree_rounds) rounds = tree_rounds;
    serial_rounds_ = rounds;
    serial_rows_   = n;
    if (spechub_ && n > 0) {
        std::fill(spechub_by_card_.begin(), spechub_by_card_.end(), RadSpechubDraw{});
        for (const auto& e : sched_.plan().e)
            if (e.req && e.card >= 0 && e.card < cfg_.max_seqs)
                spechub_by_card_[static_cast<size_t>(e.card)] =
                    {e.req->sp.seed, e.req->output.size(), e.seq};
        for (int i = 0; i < n; ++i) {
            const auto& row = serial_row_[static_cast<size_t>(i)];
            const auto& draw = spechub_by_card_[static_cast<size_t>(row.card)];
            if (draw.seq != row.seq) return RAD_E_STATE;
            spechub_draw_stage_[i] = draw;
        }
        for (auto& rk : ranks_) {
            void* dst = rk->ctx.buf_ptr(spechub_draw_buf_);
            if (!dst) return RAD_E_STATE;
            RAD_TRY(rad_memcpy_async(dst, spechub_draw_stage_, n * sizeof(RadSpechubDraw),
                                     rk->ctx.stream()));
        }
    }
    if (mtp_couple_ && n > 0) RAD_TRY(stage_couple_draws(n));

    /* RADIANCE_DEBUG_STAGE: WHERE A DRAFT PASS'S HOST TIME GOES. It splits the pass four ways --
     * the batch and its advance, the greedy sampler chain, and the BARRIER PAIR that hands the
     * pass to the rank threads and waits for them to finish issuing -- and prints a running mean
     * of each:
     *
     *   D draft: <n> passes -- batch, sampler, release, issue, in us a pass
     *
     * All four are host time the device spends running the pass before, so they cost the step
     * nothing until their sum exceeds a pass's device time.
     *
     * Off by default; four clock reads a pass when it is on. */
    static const bool stg = std::getenv("RADIANCE_DEBUG_STAGE") != nullptr;
    static double a_batch = 0, a_samp = 0, a_rel = 0, a_iss = 0;
    static long   n_pass = 0;

    if (serial_pruned_) {
        if (rounds != 5) return RAD_E_STATE;
        const RadBatch* hist = sched_.serial_batch(0, step);
        if (hist) {
            RAD_TRY(advance_all(ADV_HIST, 0, false));
            RAD_TRY(run_block_pass(hist, step, "pruned-history"));
            if (hist->n_draft_out > 0) serial_done_ = 1;
        }
        for (int first = 0; first < n; first += pruned_tile_) {
            const int count = std::min(pruned_tile_, n - first);
            for (int rd = 2; rd <= 5; ++rd) {
                const RadBatch* db = sched_.serial_frontier_batch(rd, step, first, count);
                if (!db) return RAD_E_STATE;
                RAD_TRY(run_block_pass(db, step, "pruned-frontier"));
                serial_done_ = rd;
            }
        }
    }
    /* Pass 0, then rounds 2..rounds: round 1 has no pass of its own. */
    if (!serial_pruned_) for (int rd = 0; rd <= rounds; rd += rd == 0 ? 2 : 1) {
        if (rd > 0 && n <= 0) break;
        const auto tA = stg ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const RadBatch* db = sched_.serial_batch(rd, step);
        if (!db) { if (rd == 0) continue; break; }
        /* PASS 0 IS THE HISTORY PASS: the head attends to its own K and V, nothing else in the
         * engine writes them, and a prefill leaves them untouched. It is also DRAFT ROUND 1 for
         * every row that drafts -- that round's inputs are the last row of the row's history run
         * (RadBatch::draft_out_ids) -- so it ends in the lm_head over those rows. Round r >= 2
         * embeds the token round r-1 proposed, and the advance takes it from that round's
         * proposal buffer on the card. */
        const int drafted = rd == 0 ? (int)db->n_draft_out : n;
        RAD_TRY(advance_all(rd == 0 ? ADV_HIST : ADV_ROUND, rd, rd > 0));
        const auto tB = stg ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        /* GREEDY, AND NOT THE REQUEST'S SAMPLER. A draft is a proposal the target then verifies
         * greedily (core/sample/accept.h), so sampling it at the request's temperature would
         * spend that request's RNG stream on tokens that may never be emitted and lower
         * acceptance for nothing. Scheduler-thread work, before the release, like build_step.
         *
         * A head with its own proposal buffer samples nothing at all, so its chain is zero rows
         * wide -- the same chain the history pass builds, for the same reason. The one exception
         * is RADIANCE_MTP_COUPLED: its proposal op samples x~Q on the head's own disjoint
         * stream, and the target sampler's maximal coupling makes that pay under the same
         * equality acceptance. */
        for (auto& rk : ranks_) {
            int s = rk->sampler.build_greedy(drafter_.proposal ? 0 : drafted);
            if (s < 0) {
                RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
                return s;
            }
        }

        const auto tC = stg ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        issue_what_.store(kIssueAll, std::memory_order_release);
        step_batch_.store(db, std::memory_order_release);
        /* THE TWO CROSSINGS ARE NOT THE SAME MEASUREMENT and timing them together hides which.
         * The first RELEASES -- the rank threads are already parked on it, so the drafter is last
         * in and returns as soon as it has notified, which prices the wake. The second WAITS for
         * them to finish issuing the pass, so it prices the ISSUE. */
        barrier_->arrive_and_wait();
        const auto tC2 = stg ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        barrier_->arrive_and_wait();
        const auto tD = stg ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

        int st = step_status_.load(std::memory_order_acquire);
        if (st < 0) {
            RAD_ERR("drafter '%s': pass %d of step %d failed: %s -- the engine cannot continue",
                    drafter_.name, rd, step, rad_strerror(st));
            return st;
        }
        if (stg) {
            using us = std::chrono::duration<double, std::micro>;
            a_batch += us(tB - tA).count();
            a_samp  += us(tC - tB).count();
            a_rel   += us(tC2 - tC).count();
            a_iss   += us(tD - tC2).count();
            if (++n_pass % 400 == 0)
                std::fprintf(stderr, "D draft: %ld passes -- batch %.1f, sampler %.1f, "
                                     "release %.1f, issue %.1f us a pass\n",
                             n_pass, a_batch / (double)n_pass, a_samp / (double)n_pass,
                             a_rel / (double)n_pass, a_iss / (double)n_pass);
        }
        if (rd > 0) serial_done_ = rd;
        else if (drafted > 0) serial_done_ = 1;
    }
    if (serial_done_ <= 0 || n <= 0) return RAD_OK;

    /* The last round's proposal, which no later round takes, into the slots; and rank 0's slots
     * back to the host behind it. Over the drafting rows alone: at depth 1 the last pass is the
     * history pass, whose rows after them only owed history (Scheduler::serial_batch). */
    if (!serial_pruned_) RAD_TRY(advance_all(ADV_TAKE, serial_done_, true, n));
    Rank& r0 = *ranks_[0];
    if (serial_tree_) {
        const int64_t count = (int64_t)n * (1 + 2 * r0.program.drafter_tree_capacity);
        void* const src = r0.ctx.buf_ptr(r0.program.drafter_tree_buf);
        if (!src || !readback_.p || count > readback_.n) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(readback_.p, src, count * sizeof(int32_t), r0.ctx.stream()));
        if (spechub_) {
            const void* state = r0.ctx.buf_ptr(spechub_state_buf_);
            if (!state) return RAD_E_STATE;
            RAD_TRY(rad_memcpy_async(spechub_readback_, state,
                                     n * sizeof(RadSpechubState), r0.ctx.stream()));
        }
    }
    if (mtp_couple_) {
        /* Rounds 1..done of the drafting rows, in the same stream order as the slots below. */
        const void* state = r0.ctx.buf_ptr(couple_state_buf_);
        if (!state || serial_done_ > drafter_.depth) return RAD_E_STATE;
        const int64_t pitch = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
        RAD_TRY(rad_memcpy_2d_async(couple_readback_, pitch, state, pitch,
                                    (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), serial_done_,
                                    r0.ctx.stream()));
    }
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    return RAD_OK;
}

/* AFTER COMMIT: the host's acceptance against the card's, then the proposals out through
 * set_draft and the history watermark moved. The card's acceptance positioned every draft pass,
 * so a disagreement is a defect in one of the two rules and the step is refused rather than
 * served with drafts from the wrong positions. */

int Engine::finish_serial_drafter(const StepOut& so) {
    if (!serial_drafting() || serial_rounds_ <= 0) return RAD_OK;
    const StepPlan& plan = sched_.plan();
    if (serial_done_ > 0 && serial_rows_ > 0) {
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0) continue;
            const StepSlot& S = slots_host_[e.card];   /* in range: the builder refuses others */
            if (S.acc != so.accepted[i] || S.tok[0] != so.token[i]) {
                RAD_ERR("drafter '%s': the card accepted %d draft(s) ending in token %d for "
                        "sequence %llu and the host %d ending in %d; the draft passes ran from "
                        "the card's", drafter_.name, S.acc, S.tok[0],
                        (unsigned long long)e.seq, so.accepted[i], so.token[i]);
                return RAD_E_STATE;
            }
        }
    }
    sched_.serial_end(serial_done_);
    if (serial_done_ <= 0 || serial_rows_ <= 0) return RAD_OK;

    Rank& r0 = *ranks_[0];
    const int rounds = serial_rounds_, done = serial_done_;
    if (!serial_pruned_) {
        draft_out_.assign((size_t)serial_rows_ * (size_t)rounds, 0);
        for (int i = 0; i < serial_rows_; ++i) {
            const StepSlot& S = slots_host_[serial_row_[(size_t)i].card];
            for (int rd = 1; rd <= done; ++rd)
                draft_out_[(size_t)i * (size_t)rounds + (size_t)(rd - 1)] = S.tok[rd];
        }
    }
    for (int i = 0; i < serial_rows_; ++i) {
        /* A REQUEST THE SAMPLER WILL NOT SPECULATE ON GETS NO DRAFT. That is a constraining
         * grammar whose backend cannot snapshot its state: the mask for draft position i depends
         * on which of 0..i-1 were accepted, and the sampler refuses the whole step rather than
         * masking against a state the sequence is not in -- which would be a hard failure on the
         * next step rather than a skipped draft here. */
        const uint64_t seq = serial_row_[(size_t)i].seq;
        if (!r0.sampler.allows_speculation(seq)) continue;

        if (serial_tree_) {
            int32_t anchor = -1;
            int64_t pos = -1;
            if (!sched_.block_committed_anchor(seq, serial_row_[(size_t)i].slot, &anchor, &pos) ||
                anchor < 0 || pos < 0 || pos > INT32_MAX - (int64_t)radiance::RAD_TREE_MAX_DEPTH)
                continue;
            const int cap = static_cast<int>(r0.program.drafter_tree_capacity);
            const int32_t* const row = readback_.p + (int64_t)i * (1 + 2 * cap);
            radiance::CandidateTree ctree;
            if (!radiance::dloop_tree_from_packed(ctree, static_cast<uint32_t>(anchor), row,
                                                 cap, meta_.n_vocab)) {
                RAD_ERR("drafter '%s': invalid root-inclusive tree proposal (count %d, anchor %d)",
                        drafter_.name, (int)row[0], (int)anchor);
                return RAD_E_STATE;
            }
            if (spechub_) {
                const auto& state = spechub_readback_[i];
                if (state.status != 1 || state.draw.seq != seq ||
                    state.root != anchor || ctree.count < 2 || ctree.count > 3 ||
                    static_cast<int32_t>(ctree.nodes[1].token_id) != state.pair[0] ||
                    (ctree.count == 3 &&
                     static_cast<int32_t>(ctree.nodes[2].token_id) != state.pair[1])) {
                    RAD_ERR("SpecHub GPU proposal metadata does not match its packed tree");
                    return RAD_E_STATE;
                }
            }
            if (spechub_)
                RAD_TRACE("SpecHub proposal req=%llu pos=%llu root=%d pair=(%d,%d) q_support=%d",
                          (unsigned long long)seq,
                          (unsigned long long)spechub_readback_[i].draw.pos, anchor,
                          spechub_readback_[i].pair[0], spechub_readback_[i].pair[1],
                          spechub_readback_[i].count);
            int max_depth = 0;
            for (size_t j = 0; j < ctree.count; ++j)
                max_depth = std::max(max_depth, ctree.nodes[j].depth);
            if ((serial_pruned_ || spechub_) && (!r0.sampler.allows_branching(seq) ||
                (tree_exact_to_ > 0 && pos + 1 + max_depth > tree_exact_to_))) {
                RAD_ERR("MTP experiment tree is unsupported for request %llu; refusing chain fallback",
                        (unsigned long long)seq);
                return RAD_E_UNSUPPORTED;
            }
            if (!serial_pruned_ && !spechub_ && (!r0.sampler.allows_branching(seq) ||
                (tree_exact_to_ > 0 && pos + 1 + max_depth > tree_exact_to_)))
                ctree = radiance::dloop_greedy_chain(ctree);
            const int rc = sched_.set_draft_tree(seq, ctree, static_cast<uint32_t>(pos));
            if (rc == RAD_OK && spechub_)
                for (auto& rk : ranks_)
                    RAD_TRY(rk->sampler.set_spechub(seq, spechub_readback_[i]));
            if (rc == RAD_OK || rc == RAD_E_NOTFOUND) continue;
            if (serial_pruned_ || spechub_) return rc;
            /* Preserve the serial driver's legal-spine fallback for an inadmissible topology. */
            RAD_DEBUG("drafter '%s': sequence %llu's tree is not admissible (%s); proposing "
                      "its %d-node spine as a chain", drafter_.name, (unsigned long long)seq,
                      rad_strerror(rc), done);
        }
        const int rc = sched_.set_draft(seq, &draft_out_[(size_t)i * (size_t)rounds], done);
        if (!mtp_couple_) continue;
        /* Validated whether or not the request survived; attached only to what was published.
         * A truncated publication keeps a prefix, whose rows carry their own rounds' states. */
        RAD_TRY(bind_couple(i, done, slots_host_[serial_row_[(size_t)i].card]));
        for (auto& rk : ranks_) {
            const int st = rk->sampler.set_mtp_couple(seq, couple_bound_, rc == RAD_OK ? done : 0);
            if (st < 0 && st != RAD_E_NOTFOUND) return st;
        }
    }
    return RAD_OK;
}

/* RADIANCE_MTP_COUPLED: the draw for every drafting row, before any serial pass reads it. The
 * coordinate is the PRE-commit output size, which strictly grows between two steps that draft
 * one request (each commits >= 1 token), so (base, round) never repeats a draft stream. */
int Engine::stage_couple_draws(int n) {
    std::fill(couple_by_card_.begin(), couple_by_card_.end(), RadMtpCoupleDraw{});
    for (const auto& e : sched_.plan().e)
        if (e.req && e.card >= 0 && (size_t)e.card < couple_by_card_.size())
            couple_by_card_[(size_t)e.card] = {e.req->sp.seed, (uint64_t)e.req->output.size(), e.seq,
                                               e.card, e.req->sp.greedy() ? 0u : 1u};
    for (int i = 0; i < n; ++i) {
        const auto& row = serial_row_[(size_t)i];
        if (row.card < 0 || (size_t)row.card >= couple_by_card_.size() ||
            couple_by_card_[(size_t)row.card].seq != row.seq ||
            couple_by_card_[(size_t)row.card].card != row.card) {
            RAD_ERR("MTP coupled: drafting row %d (sequence %llu, card %d) has no plan entry on "
                    "its card", i, (unsigned long long)row.seq, (int)row.card);
            return RAD_E_STATE;
        }
        couple_draw_stage_[i] = couple_by_card_[(size_t)row.card];
    }
    for (auto& rk : ranks_) {
        void* dst = rk->ctx.buf_ptr(couple_draw_buf_);
        if (!dst) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(dst, couple_draw_stage_, n * (int64_t)sizeof(RadMtpCoupleDraw),
                                 rk->ctx.stream()));
    }
    return RAD_OK;
}

/* After commit: drafting row `row`'s rounds as rank 0's ops left them, checked against the draw
 * staged for that row and the card's own slot (written through RadBatch::seq_ids, independently
 * of the row index), then bound to the verify positions the next step samples: the committed
 * output size is base + n_acc for every request still running. A stochastic row that did not come
 * back sampled, or whose x is not the card's draft, is a broken path and refuses the step: a
 * status-0 bypass or a misattached Q must never pass for coupling. */
int Engine::bind_couple(int row, int done, const StepSlot& S) {
    const RadMtpCoupleDraw& d = couple_draw_stage_[row];
    for (int rd = 1; rd <= done; ++rd) {
        RadMtpCoupleState st = couple_readback_[(int64_t)(rd - 1) * couple_rows_ + row];
        const bool sampled = d.mode == 1;
        if (std::memcmp(&st.draw, &d, sizeof d) || st.round != rd || st.x != S.tok[rd] ||
            st.status != (sampled ? 1 : 0) ||
            (sampled && !rad_mtp_couple_state_valid(st, meta_.n_vocab))) {
            RAD_ERR("MTP coupled: sequence %llu round %d returned status %d round %d x %d for "
                    "card token %d (draw %s); refusing the step", (unsigned long long)d.seq, rd,
                    st.status, st.round, st.x, S.tok[rd],
                    std::memcmp(&st.draw, &d, sizeof d) ? "stale" : "current");
            return RAD_E_STATE;
        }
        st.pos = d.base + (uint64_t)S.n_acc + (uint64_t)(rd - 1);
        couple_bound_[rd - 1] = st;
        if (sampled && (int)log_level() >= (int)Log::Trace) {
            double qx = 0;
            for (int k = 0; k < st.count; ++k) if (st.ids[k] == st.x) qx = st.q[k];
            RAD_TRACE("MTP coupled proposal req=%llu round=%d base=%llu pos=%llu x=%d q(x)=%.9g "
                      "support=%d", (unsigned long long)d.seq, rd, (unsigned long long)d.base,
                      (unsigned long long)st.pos, st.x, qx, st.count);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the block driver
 *
 * `RAD_DRAFT_BLOCK`: the same slot in the step as the serial driver and a completely different
 * shape. SERIAL is one forward a drafted token, each embedding what the last one proposed. A
 * BLOCK drafter is block-diffusion: ONE backbone pass over a block of `block` positions -- an
 * anchor and `block - 1` masked slots -- proposes `depth` tokens together, so depth is free and
 * there are no rounds to chain. `block` is the drafter's own (it is `depth + 1` when the anchor
 * row predicts nothing and `depth` when it does; rad_builder.h has both conventions).
 *
 * What there is instead is TWO passes with different shapes, and the split is what a
 * block-diffusion architecture asks for:
 *
 *   THE CONTEXT PASS mirrors the trunk step that just ran. The drafter reads the trunk's tapped
 *   hidden states, projects them, and stores K and V for those positions in a cache of its own --
 *   so it runs over exactly the rows the trunk produced, in exactly that order.
 *
 *   THE QUERY PASS runs the drafter's own layers over the block and fills every masked position.
 *   The tokens land in the buffer the plugin named as `proposal` -- not in the sampler's, because
 *   what chooses here is the drafter's own selector and it is not a sampler. That is why
 *   RAD_DRAFT_BLOCK requires a proposal buffer and RAD_DRAFT_SERIAL does not.
 *
 * They go out through set_draft() like every other proposal, so nothing downstream can tell a
 * block draft from a prompt-lookup one.
 */
/* Block out_ids name hidden-state rows, not rows for the trunk sampler. Model-only issue skips
 * its staging gate and leaves the trunk sampler parameters untouched. */
int Engine::run_block_pass(const RadBatch* db, int step, const char* what) {
    issue_what_.store(kIssueModel, std::memory_order_release);
    step_batch_.store(db, std::memory_order_release);
    barrier_->arrive_and_wait();
    barrier_->arrive_and_wait();
    const int s = step_status_.load(std::memory_order_acquire);
    if (s < 0)
        RAD_ERR("drafter '%s': %s pass of step %d failed: %s -- the engine cannot continue",
                drafter_.name, what, step, rad_strerror(s));
    if (s >= 0 && (int)log_level() >= (int)Log::Trace)
        dump_buffers(ranks_[0]->ctx, step, std::getenv("RADIANCE_DUMP_BUF"), what);
    return s;
}

int Engine::issue_block_drafter(int step, const RadBatch* b, StepOut& so) {
    block_preissued_ = false;
    block_rows_count_ = 0;
    static const bool no_draft = std::getenv("RADIANCE_NO_DRAFT") != nullptr;
    static const char* const raw_name = std::getenv("RADIANCE_DRAFT_RAW");
    const bool stateless_block = drafter_.name && std::strcmp(drafter_.name, "fulldrafter") == 0;
    if (!ranks_.empty() && ranks_[0]->program.drafter_tree_buf) return RAD_OK;
    if (drafter_.kind != RAD_DRAFT_BLOCK || drafter_.depth <= 0 || no_draft ||
        !stateless_block || raw_name || b->phase != RAD_PHASE_DECODE) return RAD_OK;
    for (const StepEntry& e : sched_.plan().e)
        if (!e.produces_token || e.is_prefill) return RAD_OK;

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    if (!so.rb) RAD_TRY(collect_issue(b, so));
    const int block = drafter_.block > 0 ? (int)drafter_.block : drafter_.depth + 1;
    block_rows_.resize((size_t)cfg_.max_seqs);
    block_rows_count_ = sched_.block_begin(block, drafter_.depth, (int)drafter_.window,
                                          drafter_.mask_token, block_rows_.data(),
                                          (int)block_rows_.size(), /*pre_commit=*/true);
    draft_out_.clear();
    if (block_rows_count_ > 0) {
        const RadBatch* qb = sched_.block_query(step);
        if (!qb) return RAD_E_STATE;
        RAD_TRY(advance_all(ADV_BLOCK, 0, false));
        RAD_TRY(run_block_pass(qb, step, "query"));
        Rank& r0 = *ranks_[0];
        void* src = r0.ctx.buf_ptr(drafter_.proposal);
        if (!src) return RAD_E_STATE;
        const int64_t n_raw = (int64_t)block_rows_count_ * drafter_.depth;
        int32_t* const raw = readback_.reserve(n_raw);
        if (!raw) return RAD_E_NOMEM;
        RAD_TRY(rad_memcpy_async(raw, src, n_raw * 4, r0.ctx.stream()));
    }
    block_preissued_ = true;
    return RAD_OK;
}

/* step_end_ has completed the proposal readback; commit has resolved the actual anchor. */
int Engine::finish_block_drafter(const StepOut& so) {
    (void)so;
    sched_.block_end();
    block_preissued_ = false;
    if (block_rows_count_ <= 0) return RAD_OK;
    /* THE LINEAR PLANE ONLY, AND THE DECLARATION IS WHAT SAYS WHICH. A program that declared a
     * tree proposal read its rows out of the packed plane and never wrote this one; publishing
     * column 0 of the tree's bytes as a token list would propose parent indices as tokens. */
    if (ranks_.empty() || !drafter_tree_declared(ranks_[0]->program)) {
        const size_t n_raw = (size_t)block_rows_count_ * (size_t)drafter_.depth;
        draft_out_.assign(readback_.p, readback_.p + n_raw);
    }
    return publish_block_drafts(/*committed=*/true);
}

int Engine::run_block_drafter(int step) {
    if (drafter_.kind != RAD_DRAFT_BLOCK || drafter_.depth <= 0) return RAD_OK;
    static const bool no_draft = std::getenv("RADIANCE_NO_DRAFT") != nullptr;
    if (no_draft) return RAD_OK;

    const int steps = drafter_.depth;
    /* THE ROW COUNT IS THE DRAFTER'S, not `depth + 1`. Both are legal and the plugin knows which
     * one its checkpoint was trained in; rad_builder.h has the two conventions side by side, and
     * declare_drafter has already defaulted and checked this. */
    const int block = drafter_.block > 0 ? (int)drafter_.block : drafter_.depth + 1;
    block_rows_.resize((size_t)cfg_.max_seqs);
    const int n = sched_.block_begin(block, steps, (int)drafter_.window, drafter_.mask_token,
                                    block_rows_.data(), (int)block_rows_.size());
    block_rows_count_ = n;

    Rank& r0 = *ranks_[0];
    void* prop = r0.ctx.buf_ptr(drafter_.proposal);
    if (!prop) {
        RAD_ERR("drafter '%s': `%s` was declared as the proposal buffer but never planned",
                drafter_.name, r0.program.buffers[drafter_.proposal].name.c_str());
        return RAD_E_STATE;
    }

    /* THE BISECT. RADIANCE_DRAFT_RAW=<buffer> drafts column 0 of that buffer instead of the
     * drafter's proposal, which for DFlash2 (`df_cand`) is the draft head's own argmax at every
     * mask row with the selector's walk thrown away. The two halves of such a drafter fail the
     * same way from the outside -- acceptance near zero, output still exactly correct -- and they
     * are otherwise hard to tell apart: if the backbone is sound the argmax already accepts, and
     * if it is not, no path through the candidates can save it. Debug-only, and it makes the
     * drafter WORSE when everything works, which is the point.
     *
     * A NAME, because the buffer is the plugin's own and the core has no business knowing which
     * drafters have an intermediate worth reading. The match is on the suffix, since a drafter
     * declares into a scope; being a debug knob that names one buffer, that is exactly as precise
     * as it needs to be. */
    static const char* const raw_name = std::getenv("RADIANCE_DRAFT_RAW");
    if (raw_name && *raw_name && !raw_buf_) {
        const std::string want(raw_name);
        for (size_t i = 1; i < r0.program.buffers.size(); ++i) {
            const std::string& nm = r0.program.buffers[i].name;
            if (nm.size() >= want.size() &&
                nm.compare(nm.size() - want.size(), want.size(), want) == 0) {
                raw_buf_ = (rad_buf)i;
                break;
            }
        }
        if (!raw_buf_) {
            RAD_ERR("RADIANCE_DRAFT_RAW=%s names no buffer in this program", raw_name);
            return RAD_E_NOTFOUND;
        }
        RAD_INFO("drafter '%s': proposing column 0 of `%s` instead of `%s` (RADIANCE_DRAFT_RAW)",
                 drafter_.name, r0.program.buffers[raw_buf_].name.c_str(),
                 r0.program.buffers[drafter_.proposal].name.c_str());
    }

    /* THE CONTEXT PASS RUNS EVEN WHEN NOTHING CAN DRAFT for KV-backed block drafters (DFlash2).
     * FullDrafter is stateless (no drafter KV cache; step() at draft_pass < 0 issues zero ops),
     * so skipping the empty context pass avoids a redundant batch build and worker barrier on
     * every prefill chunk and decode step. */
    static const bool no_ctx = std::getenv("RADIANCE_DRAFT_NOCTX") != nullptr;
    const bool stateless_block = drafter_.name && std::strcmp(drafter_.name, "fulldrafter") == 0;
    if (!no_ctx && !stateless_block)
        if (const RadBatch* cb = sched_.block_context(step))
            RAD_TRY(run_block_pass(cb, step, "context"));

    /* EMPTY UNTIL THIS CALL'S QUERY PASS FILLS IT. It is a member, so a query batch that could
     * not be built would otherwise leave the previous step's proposals standing -- published
     * below as this step's, and sized for however many rows drafted then, which may be fewer
     * than drafted now. */
    draft_out_.clear();
    block_tree_out_.clear();
    if (n > 0) {
        if (const RadBatch* qb = sched_.block_query(step)) {
            RAD_TRY(run_block_pass(qb, step, "query"));
            if (r0.program.drafter_tree_buf) {
                if (raw_buf_) {
                    RAD_ERR("RADIANCE_DRAFT_RAW cannot replace a declared tree proposal");
                    return RAD_E_INVAL;
                }
                const int64_t pitch=1+2*r0.program.drafter_tree_capacity;
                const int64_t elements=static_cast<int64_t>(n)*pitch;
                int32_t* raw=readback_.reserve(elements);
                void* src=r0.ctx.buf_ptr(r0.program.drafter_tree_buf);
                if (!raw || !src) return RAD_E_STATE;
                RAD_TRY(rad_memcpy_async(raw,src,elements*4,r0.ctx.stream()));
                RAD_TRY(rad_stream_sync(r0.ctx.stream()));
                block_tree_out_.assign(raw,raw+elements);
            } else {
                draft_out_.assign((size_t)n * (size_t)steps, 0);
                const rad_buf from = raw_buf_ ? raw_buf_ : drafter_.proposal;
                void* src = (from == drafter_.proposal) ? prop : r0.ctx.buf_ptr(from);
                if (!src) {
                    RAD_ERR("`%s` was declared but never planned",
                            r0.program.buffers[from].name.c_str());
                    return RAD_E_STATE;
                }
                /* A raw buffer is [row][width] and the proposal is [seq][steps]: one may be strided
                 * candidate count and the other is dense, so the bisect copies the whole candidate
                 * plane and takes column zero rather than pretending the two have the same shape. */
                const int64_t k = raw_buf_ ? r0.program.buffers[from].decl.shape[1] : 1;
                const int64_t n_raw = (int64_t)n * steps * k;
                int32_t* const raw = readback_.reserve(n_raw);
                if (!raw) return RAD_E_NOMEM;
                RAD_TRY(rad_memcpy_async(raw, src, n_raw * 4, r0.ctx.stream()));
                RAD_TRY(rad_stream_sync(r0.ctx.stream()));
                for (size_t i = 0; i < draft_out_.size(); ++i) draft_out_[i] = raw[i * (size_t)k];
            }
        }
    }
    sched_.block_end();
    if (n <= 0 || (draft_out_.empty() && block_tree_out_.empty())) return RAD_OK;

    return publish_block_drafts(/*committed=*/false);
}

int Engine::publish_block_drafts(bool committed) {
    Rank& r0 = *ranks_[0];
    const int steps = drafter_.depth;
    for (int i = 0; i < block_rows_count_; ++i) {
        /* A request the sampler will not speculate on gets no draft; see finish_serial_drafter. */
        if (!r0.sampler.allows_speculation(block_rows_[(size_t)i].seq)) continue;

        const auto& row = block_rows_[(size_t)i];
        int32_t anchor = row.anchor_token;
        int64_t pos = row.root_pos;
        if (committed && !sched_.block_committed_anchor(row.seq, row.slot, &anchor, &pos))
            continue;
        if (anchor < 0 || pos < 0 || pos > INT32_MAX - (int64_t)radiance::RAD_TREE_MAX_DEPTH) continue;
        const uint32_t anchor_tok = static_cast<uint32_t>(anchor);
        const uint32_t root_pos = static_cast<uint32_t>(pos);

        radiance::CandidateTree ctree;
        if (r0.program.drafter_tree_buf) {
            const int cap=static_cast<int>(r0.program.drafter_tree_capacity);
            const size_t pitch=static_cast<size_t>(1+2*cap);
            if (block_tree_out_.size()<(static_cast<size_t>(i)+1)*pitch) {
                RAD_ERR("drafter '%s': the query pass declared %zu tree row(s) for %d sequence(s)",
                        drafter_.name, block_tree_out_.size()/std::max<size_t>(1,pitch),
                        block_rows_count_);
                return RAD_E_STATE;
            }
            const int32_t* const prow = block_tree_out_.data()+static_cast<size_t>(i)*pitch;
            /* COUNT 0 IS THE SELECTOR'S DECLARED DECLINE, and only 0 is. `dflash_tree_select`'s
             * contract (mine/dflash_tree_select.h) is that a row whose data cannot produce a tree --
             * an anchor outside the vocabulary, a non-finite unary term on a live candidate -- writes
             * count 0 rather than failing a launch it has already issued, and the host oracle
             * produces the same 0 for the same bytes. That is a skipped draft: the sequence gets no
             * proposal this step and decodes one token.
             *
             * ANYTHING ELSE IS MALFORMED AND STAYS A FAILURE. A negative count, or one past the
             * declared capacity, is a producer bug; silently continuing would drop a defect on the
             * floor and the only symptom would be acceptance that is quietly worse. */
            if (prow[0] == 0) continue;
            if (prow[0] < 0 || prow[0] > cap) {
                RAD_ERR("drafter '%s': packed tree row %d declares %d node(s); the declared "
                        "capacity is %d", drafter_.name, i, (int)prow[0], cap);
                return RAD_E_STATE;
            }
            if (!radiance::dloop_tree_from_packed(ctree,anchor_tok,prow,cap,meta_.n_vocab)) {
                RAD_ERR("drafter '%s': invalid root-inclusive tree proposal (count %d, anchor %d)",
                        drafter_.name, (int)prow[0], (int)anchor);
                return RAD_E_STATE;
            }
            int max_depth=0;
            for (size_t j=0;j<ctree.count;++j) max_depth=std::max(max_depth,ctree.nodes[j].depth);
            const bool past_qsa=tree_exact_to_>0 && pos+1+max_depth>tree_exact_to_;
            if (past_qsa || !r0.sampler.allows_branching(row.seq)) {
                ctree=radiance::dloop_greedy_chain(ctree);
                if (past_qsa && row.card>=0 && static_cast<size_t>(row.card)<tree_chain_notice_.size() &&
                    tree_chain_notice_[static_cast<size_t>(row.card)]!=row.seq) {
                    tree_chain_notice_[static_cast<size_t>(row.card)]=row.seq;
                    RAD_DEBUG("DLoop sequence %llu: greedy-chain fallback past QSA exactness bound %lld",
                              (unsigned long long)row.seq,(long long)tree_exact_to_);
                }
            }
        } else {
            uint32_t draft_tokens[16] = {0};
            const int max_k = std::min(steps, 15);
            int copy_k = 0;
            for (int s = 0; s < max_k; ++s) {
                const int32_t tok = draft_out_[(size_t)i * (size_t)steps + (size_t)s];
                if (tok < 0) break;
                draft_tokens[copy_k++] = static_cast<uint32_t>(tok);
            }
            if (copy_k <= 0) continue;
            ctree.build_linear(anchor_tok, draft_tokens, copy_k);
        }

        const int rc = sched_.set_draft_tree(row.seq, ctree, root_pos);
        if (rc < 0 && rc != RAD_E_NOTFOUND) {
            if (r0.program.drafter_tree_buf) {
                RAD_ERR("drafter '%s': scheduler rejected declared tree: %s",drafter_.name,rad_strerror(rc));
                return rc;
            }
            // Preserve the legacy linear driver's unsupported-topology behavior.
            continue;
        }
    }
    return RAD_OK;
}


}  /* namespace rad */
