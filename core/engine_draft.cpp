/* engine_draft.cpp -- the two drafter drivers (spec §10).
 *
 * A draft head runs as extra passes after the trunk step. SERIAL runs `depth` dependent passes;
 * BLOCK runs a context pass, then one pass that fills the whole block. The architecture plugin
 * declares which (rad_declare_drafter); nothing here depends on the model.
 */
#include "engine_priv.h"

#include "../arch/common/rad_spechub_config.h"
#include "sched/mtp_couple_bind.h"
#include "sched/mtp_couple_plan.h"
#include "sched/scheduler.h"
#include "device/device.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the card's step state */
/* Every rank's zeroed StepSlots and sorted end-of-generation ids, and the pinned host copy of rank
 * 0's slots the step reads back. Allocated once at start, so the step path never allocates. */
int Engine::step_state_init() {
    std::vector<int32_t> eog;
    if (vocab_) eog = vocab_->eog();
    std::sort(eog.begin(), eog.end());
    eog.erase(std::unique(eog.begin(), eog.end()), eog.end());
    const int64_t n_slot = cfg_.max_seqs > 0 ? cfg_.max_seqs : 1;
    const int64_t slot_bytes = n_slot * (int64_t)sizeof(StepSlot);
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

    /* RADIANCE_MTP_COUPLED: the head's two planes, found by name in the program it declared, and
     * the pinned staging the step path needs beside them. A program without them has no coupled
     * path, so the flag's absence and its contradiction are told apart here: the flag is on and the
     * head declared nothing means the architecture refused it (it returns RAD_E_UNSUPPORTED at its
     * own declare), and a silent no-op would serve uncoupled drafts while the flag said otherwise. */
    mtp_couple_ = mtp_coupled_enabled();
    if (mtp_couple_) {
        /* THE DRAFTER SHAPE IS CHECKED HERE, where the declaration is in hand, not against
         * environment variables nobody else reads. Coupling is defined on a LINEAR chain: a block
         * drafter proposes a whole block at once from one pass and writes it to its proposal
         * buffer, so there is no per-round draw to couple and no per-round state to read back.
         * Left unchecked, a block drafter would silently serve uncoupled drafts while the flag
         * said otherwise. */
        if (drafter_.kind != RAD_DRAFT_SERIAL || drafter_.depth <= 0) {
            RAD_ERR("RADIANCE_MTP_COUPLED requires the linear serial draft chain (RAD_DRAFT_SERIAL "
                    "with depth > 0); this model declares drafter '%s' kind %d depth %d, which has "
                    "no per-round draw to couple",
                    drafter_.name ? drafter_.name : "(none)", drafter_.kind, drafter_.depth);
            return RAD_E_UNSUPPORTED;
        }
        const Program& prog = ranks_[0]->program;
        for (size_t i = 0; i < prog.buffers.size(); ++i) {
            const std::string& name = prog.buffers[i].name;
            auto ends = [&](const char* suffix) {
                const size_t n = std::strlen(suffix);
                return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
            };
            if (ends("mtp_couple_draws")) couple_draws_buf_ = (rad_buf)i;
            if (ends("mtp_couple_state")) couple_state_buf_ = (rad_buf)i;
        }
        if (!couple_draws_buf_ || !couple_state_buf_) {
            RAD_ERR("RADIANCE_MTP_COUPLED is set but the architecture declared no coupled MTP "
                    "planes; the flag and the model disagree");
            return RAD_E_STATE;
        }
        const RadBufDecl& sd = prog.buffers[couple_state_buf_].decl;
        if (sd.rank != 3 || sd.shape[2] != (int64_t)RAD_MTP_COUPLE_STATE_WORDS) {
            RAD_ERR("RADIANCE_MTP_COUPLED: 'mtp_couple_state' is not the [rounds, rows, %d]-word "
                    "plane the coupling reads", (int)RAD_MTP_COUPLE_STATE_WORDS);
            return RAD_E_STATE;
        }
        couple_rows_ = sd.shape[1];
        /* ONE ENTRY PER CARD SLOT, indexed by StepEntry::card exactly as slots_host_ is: the plan
         * names a request's sequence on the slot the advance writes, and the draft row carries the
         * same `card`. A vector left empty makes every lookup miss, so the first step that drafts
         * would refuse with "has no plan entry on its card" instead of staging its draw. */
        couple_by_card_.assign((size_t)n_slot, RadMtpCoupleDraw{});
        couple_draw_stage_ = static_cast<RadMtpCoupleDraw*>(
            rad_dev_alloc(n_slot * (int64_t)sizeof(RadMtpCoupleDraw), RAD_MEM_HOST_PINNED));
        couple_readback_ = static_cast<RadMtpCoupleState*>(rad_dev_alloc(
            (int64_t)drafter_.depth * couple_rows_ * (int64_t)sizeof(RadMtpCoupleState),
            RAD_MEM_HOST_PINNED));
        if (!couple_draw_stage_ || !couple_readback_) return RAD_E_NOMEM;
        RAD_INFO("MTP coupled LIVE: %d linear round(s), x~Q from the final reranked top8, maximal "
                 "coupling at the verify; fixed-seed tokens are NOT preserved", drafter_.depth);
    }

    RAD_TRY(rad_event_create_as(&step_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
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
    slots_host_ = nullptr;
    if (couple_draw_stage_) rad_dev_free(couple_draw_stage_, RAD_MEM_HOST_PINNED);
    if (couple_readback_) rad_dev_free(couple_readback_, RAD_MEM_HOST_PINNED);
    couple_draw_stage_ = nullptr;
    couple_readback_ = nullptr;
    if (step_end_) rad_event_destroy(step_end_);
    step_end_ = nullptr;
}

/* ------------------------------------------------------------------ the serial driver
 *
 * RAD_DRAFT_SERIAL: one head pass per drafted token after the trunk step (MTP, EAGLE, Medusa).
 * Round 1 consumes the sampled token and the hidden state of the position it came from, which
 * depends on how many drafts were accepted; it runs in the history pass, whose rows include that
 * position (RadBatch::draft_out_ids).
 *
 * The card, not the host, resolves those positions (sched/advance.h): behind the trunk's sampler
 * every rank runs the acceptance into its StepSlots, and an advance before each draft pass fills
 * in its positions and tokens. So the host issues the whole step (trunk, history, every round)
 * without waiting and reads back once after the last round, instead of idling the card for a
 * readback, commit and batch build per step.
 *
 * Each pass is an ordinary RadBatch through the same barrier and sampler chain; `draft_pass`
 * tells the plugin to run the head. Tokens go out through set_draft(), like prompt lookup's; a
 * sequence the head declined keeps n_draft 0, leaving it to the n-gram drafter.
 */
bool Engine::serial_drafting() const {
    return drafter_.kind == RAD_DRAFT_SERIAL && drafter_.depth > 0;
}

/* One advance on every rank against the scheduler's latest build, over its first `rows` rows
 * when given, else all. */
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

/* Issue the step's draft passes behind it, without waiting. After the trunk's ranks join: the
 * acceptance on every rank, the sampled-id readback (before a round can overwrite them), the
 * history pass and every round, then the final proposal into the slots and their readback.
 * Nothing synchronises here; collect_step has the step's one wait. */
int Engine::issue_serial_drafter(int step, StepOut& so) {
    serial_rounds_ = 0;
    serial_rows_   = 0;
    serial_done_   = 0;
    /* A step issued ahead continues from the card's acceptance, so run it even without drafts. */
    if (!serial_drafting()) {
        if (pipelining_) RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
        return RAD_OK;
    }

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    RAD_TRY(collect_issue(so));

    int rounds = drafter_.depth;
    serial_row_.resize((size_t)cfg_.max_seqs);
    /* `n` counts rows that can draft. It can be zero while the history pass still has work (a
     * mid-prompt sequence has hidden states the head needs but no token to propose from). */
    const int n = sched_.serial_begin(&rounds, serial_row_.data(), (int)serial_row_.size());
    if (rounds <= 0) return RAD_OK;
    serial_rounds_ = rounds;
    serial_rows_   = n;
    /* Before ANY serial pass reads a draw: the round-1 pass is the history pass below. */
    if (mtp_couple_ && n > 0) RAD_TRY(stage_couple_draws(n));

    /* RADIANCE_DEBUG_STAGE: running means of a draft pass's host time: batch + advance, greedy
     * sampler chain, barrier release and issue. They cost the step nothing until their sum exceeds
     * the previous pass's device time. */
    static const bool stg = std::getenv("RADIANCE_DEBUG_STAGE") != nullptr;
    static double a_batch = 0, a_samp = 0, a_rel = 0, a_iss = 0;
    static long   n_pass = 0;

    /* Pass 0, then rounds 2..rounds: round 1 has no pass of its own. */
    for (int rd = 0; rd <= rounds; rd += rd == 0 ? 2 : 1) {
        if (rd > 0 && n <= 0) break;
        const auto tA = std::chrono::steady_clock::now();
        const RadBatch* db = sched_.serial_batch(rd, step);
        if (!db) { if (rd == 0) continue; break; }
        /* Pass 0 is the history pass, writing the head's own K and V (nothing else does). It is
         * also draft round 1 for drafting rows, whose inputs are the last row of their history run
         * (RadBatch::draft_out_ids), so it ends in the lm_head over those rows. Round r >= 2 embeds
         * round r-1's proposal, which the advance takes from the proposal buffer on the card. */
        const int drafted = rd == 0 ? (int)db->n_draft_out : n;
        RAD_TRY(advance_all(rd == 0 ? ADV_HIST : ADV_ROUND, rd, rd > 0));
        const auto tB = std::chrono::steady_clock::now();

        /* Greedy, not the request's sampler: the target verifies against its own draw
         * (core/sample/accept.h), so sampling a draft would spend the request's RNG stream and
         * lower acceptance. A head with its own proposal buffer samples nothing (a zero-row
         * chain). */
        for (auto& rk : ranks_) {
            int s = rk->sampler.build_greedy(drafter_.proposal ? 0 : drafted);
            if (s < 0) {
                RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
                return s;
            }
        }

        const auto tC = std::chrono::steady_clock::now();
        issue_what_.store(kIssueAll, std::memory_order_release);
        step_batch_.store(db, std::memory_order_release);
        /* Timed separately: the first crossing releases the parked ranks (the wake), the second
         * waits for them to finish issuing (the issue). */
        barrier_->arrive_and_wait();
        const auto tC2 = std::chrono::steady_clock::now();
        barrier_->arrive_and_wait();
        const auto tD = std::chrono::steady_clock::now();

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
                RAD_INFO("draft: %ld passes; batch %.1f, sampler %.1f, release %.1f, issue %.1f us "
                         "a pass", n_pass, a_batch / (double)n_pass, a_samp / (double)n_pass,
                         a_rel / (double)n_pass, a_iss / (double)n_pass);
        }
        if (rd > 0) serial_done_ = rd;
        else if (drafted > 0) serial_done_ = 1;
    }
    if (serial_done_ <= 0 || n <= 0) return RAD_OK;

    /* The last round's proposal into the slots, then rank 0's slots back to the host. Only the
     * drafting rows: at depth 1 the last pass is the history pass, whose later rows only carried
     * history (Scheduler::serial_batch). */
    RAD_TRY(advance_all(ADV_TAKE, serial_done_, true, n));
    Rank& r0 = *ranks_[0];
    if (mtp_couple_) {
        /* Rounds 1..done of the drafting rows, in the SAME stream order as the slots below: the
         * states come back behind the proposal that they were drawn beside. */
        const void* state = r0.ctx.buf_ptr(couple_state_buf_);
        if (!state || serial_done_ > (int)drafter_.depth) return RAD_E_STATE;
        const int64_t pitch = couple_rows_ * (int64_t)sizeof(RadMtpCoupleState);
        RAD_TRY(rad_memcpy_2d_async(couple_readback_, pitch, state, pitch,
                                    (int64_t)n * (int64_t)sizeof(RadMtpCoupleState), serial_done_,
                                    r0.ctx.stream()));
    }
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    return RAD_OK;
}

/* After commit: check the host's acceptance against the card's, publish the proposals through
 * set_draft, and move the history watermark. The card's acceptance positioned every draft pass,
 * so a disagreement is a defect and the step is refused. */
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
    draft_out_.assign((size_t)serial_rows_ * (size_t)rounds, 0);
    for (int i = 0; i < serial_rows_; ++i) {
        const StepSlot& S = slots_host_[serial_row_[(size_t)i].card];
        for (int rd = 1; rd <= done; ++rd)
            draft_out_[(size_t)i * (size_t)rounds + (size_t)(rd - 1)] = S.tok[rd];
    }
    for (int i = 0; i < serial_rows_; ++i) {
        /* No draft for a request the sampler will not speculate on (a constraining grammar that
         * cannot snapshot its state); the sampler would refuse the next step otherwise. */
        if (!r0.sampler.allows_speculation(serial_row_[(size_t)i].seq)) continue;
        const uint64_t seq = serial_row_[(size_t)i].seq;
        const int rc = sched_.set_draft(seq, &draft_out_[(size_t)i * (size_t)rounds], done);
        if (!mtp_couple_) continue;
        /* BOUND ONLY FOR A REQUEST STILL RUNNING. A truncated publication keeps a prefix, whose
         * rows carry their own rounds' states; a request this commit finished (a max_tokens or
         * context cut), or one the scheduler refused, binds nothing -- its rows are carried
         * terminal work and will never sample again. set_mtp_couple(…, 0) clears what an earlier
         * step left, and for a request already gone from the sampler it is RAD_E_NOTFOUND.
         *
         * The position coordinate is the request's committed output length AFTER this commit -- the
         * value the next build_step reads as its n_out. The rule is mtp_couple_bind_rounds
         * (sched/mtp_couple_bind.h), where it is driven against a real commit without an Engine. */
        const MtpCoupleBindPoint bp = mtp_couple_bind_point(plan, seq);
        int n_bound = 0;
        std::string why;
        const int bs = mtp_couple_bind_rounds(
            bp, couple_draw_stage_[(size_t)i], &couple_readback_[(size_t)i], (int)couple_rows_,
            done, slots_host_[serial_row_[(size_t)i].card], meta_.n_vocab, couple_bound_,
            &n_bound, &why);
        if (bs < 0) { RAD_ERR("%s", why.c_str()); return bs; }
        for (auto& rk : ranks_) {
            const int st = rk->sampler.set_mtp_couple(seq, couple_bound_,
                                                      rc == RAD_OK ? n_bound : 0);
            if (st < 0 && st != RAD_E_NOTFOUND) return st;
        }
    }
    return RAD_OK;
}

/* One line, once, at the first coupled verify that completed on the device. What makes it
 * evidence is where it reads: the sampler's staged rows AFTER collect_step's wait, so the op it
 * names was built with bound rows and its batch returned -- an executed verify, not an enqueue, a
 * LIVE flag or a terminal row bound for a request that will never sample again (those are skipped
 * at bind, sched/mtp_couple_bind.h). Request numbers and the kernel name only: no tokens, no keys,
 * no logits. */
void Engine::couple_verify_evidence(int step) {
    if (!mtp_couple_ || couple_evidence_) return;
    Sampler& smp = ranks_[0]->sampler;
    if (!smp.staged_any_couple()) return;
    int64_t n = 0;
    smp.staged_params(&n);
    std::vector<uint64_t> seqs;
    int64_t rows = 0;
    for (int64_t r = 0; r < n; ++r) {
        const RadMtpCoupleState& c = smp.staged_couple(r);
        if (c.round <= 0) continue;
        ++rows;
        if (std::find(seqs.begin(), seqs.end(), c.draw.seq) == seqs.end())
            seqs.push_back(c.draw.seq);
    }
    if (rows <= 0) return;
    std::string ids;
    for (size_t i = 0; i < seqs.size(); ++i)
        ids += fmt("%s%llu", i ? "," : "", (unsigned long long)seqs[i]);
    couple_evidence_ = true;
    RAD_INFO("MTP coupled verify completed: step %d, %lld bound row(s) over request(s) [%s], "
             "kernel %s", step, (long long)rows, ids.c_str(),
             ranks_[0]->ctx.ip_kernel_name(smp.staged_couple_op()));
}

/* RADIANCE_MTP_COUPLED: the draw for every drafting row, before any serial pass reads it.
 *
 * The coordinate is the PRE-commit output size, which strictly grows between two steps that draft
 * one request (each commits at least one token), so (base, round) never repeats a draft stream. The
 * card is the slot the advance writes, and the plugin refuses a row whose staged draw names another
 * -- which is what makes a stale draw a refusal here rather than a wrong token later. The rule
 * itself is sched/mtp_couple_plan.h, where it is checked without a card. */
int Engine::stage_couple_draws(int n) {
    std::string why;
    const int stg = mtp_couple_plan_stage(sched_.plan(),
                                          (int)(cfg_.max_seqs > 0 ? cfg_.max_seqs : 1),
                                          serial_row_.data(), n, &couple_by_card_,
                                          couple_draw_stage_, &why);
    if (stg < 0) {
        RAD_ERR("MTP coupled: %s", why.c_str());
        return stg;
    }
    for (auto& rk : ranks_) {
        void* dst = rk->ctx.buf_ptr(couple_draws_buf_);
        if (!dst) return RAD_E_STATE;
        RAD_TRY(rad_memcpy_async(dst, couple_draw_stage_,
                                 n * (int64_t)sizeof(RadMtpCoupleDraw), rk->ctx.stream()));
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the block driver
 *
 * RAD_DRAFT_BLOCK (block diffusion, e.g. DFlash2): one backbone pass over `block` positions (an
 * anchor and `block - 1` masked slots) proposes `depth` tokens at once, with no rounds. `block` is
 * the drafter's own: depth + 1 when the anchor row predicts nothing, depth when it does
 * (rad_builder.h). Two passes:
 *
 *   Context pass: mirrors the trunk step just run. The drafter projects the trunk's tapped hidden
 *   states and stores K and V for those positions in its own cache, over exactly the trunk's rows
 *   in order.
 *
 *   Query pass: the drafter's layers over the block fill every masked position. Its own selector
 *   writes the tokens to the plugin's `proposal` buffer, which is why RAD_DRAFT_BLOCK requires
 *   one.
 *
 * Proposals go out through set_draft() like every other.
 *
 * Two drivers run these passes. run_block_drafter, after the step commits, stages both from the
 * committed sequence and reads the proposals back: two host waits a step with the card idle
 * between them, and no step issued ahead. The card's driver (issue_block_drafter, the default)
 * queues both behind the step before its readback, from the card's acceptance, as the serial
 * driver does; RADIANCE_DRAFT_BLOCK_HOST=1 selects the host's.
 */

/* One drafter pass through the barrier. Neither pass samples (no logits in the context pass; the
 * query pass has its own selector), so the sampler chain is zero rows. */
int Engine::block_pass(const RadBatch* db, const char* what, int step) {
    for (auto& rk : ranks_) {
        int s = rk->sampler.build_greedy(0);
        if (s < 0) {
            RAD_ERR("drafter '%s': build_greedy: %s", drafter_.name, rad_strerror(s));
            return s;
        }
    }
    issue_what_.store(kIssueAll, std::memory_order_release);
    step_batch_.store(db, std::memory_order_release);
    barrier_->arrive_and_wait();
    barrier_->arrive_and_wait();
    const int s = step_status_.load(std::memory_order_acquire);
    if (s < 0)
        RAD_ERR("drafter '%s': %s pass of step %d failed: %s -- the engine cannot continue",
                drafter_.name, what, step, rad_strerror(s));
    /* Probe after each pass; by the step loop's dump these buffers hold stale values. */
    if (s >= 0) dump_buffers(ranks_[0]->ctx, step, what);
    return s;
}

int Engine::run_block_drafter(int step) {
    if (drafter_.kind != RAD_DRAFT_BLOCK || drafter_.depth <= 0) return RAD_OK;

    const int steps = drafter_.depth;
    /* The row count is the drafter's (defaulted and checked by declare_drafter). */
    const int block = drafter_.block > 0 ? (int)drafter_.block : drafter_.depth + 1;
    std::vector<Scheduler::BlockRow> rows((size_t)cfg_.max_seqs);
    const int n = sched_.block_begin(block, steps, (int)drafter_.window, drafter_.mask_token,
                                  rows.data(), (int)rows.size());

    Rank& r0 = *ranks_[0];
    void* prop = r0.ctx.buf_ptr(drafter_.proposal);
    if (!prop) {
        RAD_ERR("drafter '%s': `%s` was declared as the proposal buffer but never planned",
                drafter_.name, r0.program.buffers[drafter_.proposal].name.c_str());
        return RAD_E_STATE;
    }

    /* The context pass runs even when nothing drafts: a mid-prompt sequence has hidden states the
     * drafter will need, and skipping would leave a permanent hole in its cache. */
    if (const RadBatch* cb = sched_.block_context(step)) RAD_TRY(block_pass(cb, "context", step));

    /* Cleared so a query batch that could not be built does not republish the previous step's
     * proposals. */
    draft_out_.clear();
    if (n > 0) {
        if (const RadBatch* qb = sched_.block_query(step)) {
            RAD_TRY(block_pass(qb, "query", step));
            const int64_t n_raw = (int64_t)n * steps;
            int32_t* const raw = readback_.reserve(n_raw);
            if (!raw) return RAD_E_NOMEM;
            RAD_TRY(rad_memcpy_async(raw, prop, n_raw * 4, r0.ctx.stream()));
            RAD_TRY(rad_stream_sync(r0.ctx.stream()));
            draft_out_.assign(raw, raw + n_raw);
        }
    }
    sched_.block_end();
    if (n <= 0 || draft_out_.empty()) return RAD_OK;

    for (int i = 0; i < n; ++i) {
        /* A request the sampler will not speculate on gets no draft; see finish_serial_drafter. */
        if (!r0.sampler.allows_speculation(rows[(size_t)i].seq)) continue;
        sched_.set_draft(rows[(size_t)i].seq, &draft_out_[(size_t)i * (size_t)steps], steps);
    }
    return RAD_OK;
}

/* The card's driver: both passes behind the step, without waiting. After the trunk's ranks join,
 * the acceptance runs on every rank (ADV_ACCEPT), leaving each sequence's anchor and committed
 * end in its StepSlot. The context pass mirrors the trunk step, whose positions the host knows.
 * The query pass's rows are device rows that the card positions at the committed end (ADV_BLOCK),
 * and its proposals go into the slots behind the anchor (ADV_BTAKE): exactly the token row the
 * next verify reads (ADV_TRUNK), so the step issued ahead verifies them with no host round trip.
 * collect_step's wait covers both passes and the slots' readback. */
int Engine::issue_block_drafter(int step, StepOut& so) {
    block_rows_ = 0;
    if (!block_card_) return RAD_OK;

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    /* The sampled ids before either pass, as the serial driver reads them. */
    RAD_TRY(collect_issue(so));

    const int steps = drafter_.depth;
    const int block = drafter_.block > 0 ? (int)drafter_.block : drafter_.depth + 1;
    block_row_.resize((size_t)cfg_.max_seqs);
    const int n = sched_.block_begin(block, steps, (int)drafter_.window, drafter_.mask_token,
                                     block_row_.data(), (int)block_row_.size(), /*on_card=*/true);

    /* Even when nothing drafts; see run_block_drafter. */
    if (const RadBatch* cb = sched_.block_context(step)) RAD_TRY(block_pass(cb, "context", step));
    if (n <= 0) return RAD_OK;
    const RadBatch* qb = sched_.block_query(step);
    if (!qb) return RAD_OK;
    RAD_TRY(advance_all(ADV_BLOCK, 0, false));
    RAD_TRY(block_pass(qb, "query", step));
    RAD_TRY(advance_all(ADV_BTAKE, steps, true, n));
    Rank& r0 = *ranks_[0];
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    block_rows_ = n;
    return RAD_OK;
}

/* After commit: check the host's acceptance against the card's, which positioned the query pass
 * (a disagreement is a defect and the step is refused, as in finish_serial_drafter), move the
 * drafter's filled range, and publish each row's proposal from the slots. */
int Engine::finish_block_drafter(const StepOut& so) {
    if (!block_card_) return RAD_OK;
    if (block_rows_ > 0) {
        const StepPlan& plan = sched_.plan();
        for (size_t i = 0; i < plan.e.size(); ++i) {
            const StepEntry& e = plan.e[i];
            if (so.first_row[i] < 0) continue;
            const StepSlot& S = slots_host_[e.card];   /* in range: the builder refuses others */
            if (S.acc != so.accepted[i] || S.tok[0] != so.token[i]) {
                RAD_ERR("drafter '%s': the card accepted %d draft(s) ending in token %d for "
                        "sequence %llu and the host %d ending in %d; the query pass ran from the "
                        "card's", drafter_.name, S.acc, S.tok[0], (unsigned long long)e.seq,
                        so.accepted[i], so.token[i]);
                sched_.block_end();
                return RAD_E_STATE;
            }
        }
    }
    sched_.block_end();
    if (block_rows_ <= 0) return RAD_OK;

    Rank& r0 = *ranks_[0];
    const int steps = drafter_.depth;
    draft_out_.assign((size_t)block_rows_ * (size_t)steps, 0);
    for (int i = 0; i < block_rows_; ++i) {
        const StepSlot& S = slots_host_[block_row_[(size_t)i].card];
        for (int j = 0; j < steps; ++j)
            draft_out_[(size_t)i * (size_t)steps + (size_t)j] = S.tok[1 + j];
    }
    for (int i = 0; i < block_rows_; ++i) {
        /* As in finish_serial_drafter: the step issued ahead only carries a request whose answer
         * is fixed (Scheduler::step_next_locked), so this agrees with the count it verifies. */
        if (!r0.sampler.allows_speculation(block_row_[(size_t)i].seq)) continue;
        sched_.set_draft(block_row_[(size_t)i].seq, &draft_out_[(size_t)i * (size_t)steps], steps);
    }
    return RAD_OK;
}

}  /* namespace rad */
