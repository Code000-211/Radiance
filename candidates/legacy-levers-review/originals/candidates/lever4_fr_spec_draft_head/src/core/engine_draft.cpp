/* engine_draft.cpp -- the two drafter drivers (spec §10).
 *
 * A draft head runs as EXTRA PASSES after the trunk step has been sampled and committed, and
 * there are two shapes of it: SERIAL runs `depth` dependent passes, BLOCK runs a context pass and
 * then one pass that fills the whole block. Which one this program has is what the architecture
 * plugin DECLARED (rad_declare_drafter); nothing in this file knows what model it is.
 *
 * Split out of the step loop's file because both drivers are the same idea from the loop's point
 * of view -- one call after commit -- and neither is on the trunk path. See engine_priv.h.
 */
#include "engine_priv.h"

#include "sched/scheduler.h"
#include "device/device.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the card's step state */
/* Every rank's StepSlots, zeroed, and its sorted end-of-generation ids; and the pinned host copy
 * of rank 0's slots that the step reads back. Once, at start: nothing on the step path allocates. */
int Engine::step_state_init() {
    std::vector<int32_t> eog;
    if (vocab_) eog = vocab_->eog();
    std::sort(eog.begin(), eog.end());
    eog.erase(std::unique(eog.begin(), eog.end()), eog.end());
    const int64_t n_slot = cfg_.max_seqs > 0 ? cfg_.max_seqs : 1;
    const int64_t slot_bytes = n_slot * (int64_t)sizeof(StepSlot);

    /* Load reduced vocabulary token ids into pinned host memory if available */
    if (ckpt_) {
        if (const SrcTensor* vt = ckpt_->find("mtp.lm_head_reduced.vocab_ids")) {
            const int64_t n_v = vt->numel();
            n_vocab_ids_ = (int32_t)n_v;
            const int64_t v_bytes = n_v * (int64_t)sizeof(int32_t);
            vocab_ids_host_ = static_cast<int32_t*>(rad_dev_alloc(v_bytes, RAD_MEM_HOST_PINNED));
            if (vocab_ids_host_) {
                if (vt->rad_dtype == RAD_I64) {
                    const int64_t* src64 = static_cast<const int64_t*>(vt->data);
                    for (int64_t j = 0; j < n_v; ++j) vocab_ids_host_[j] = (int32_t)src64[j];
                } else if (vt->rad_dtype == RAD_I32) {
                    std::memcpy(vocab_ids_host_, vt->data, (size_t)v_bytes);
                }
            }
        }
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

        if (vocab_ids_host_ && n_vocab_ids_ > 0) {
            rk->n_vocab_ids = n_vocab_ids_;
            rk->vocab_ids_dev = static_cast<int32_t*>(
                rad_dev_alloc((int64_t)n_vocab_ids_ * (int64_t)sizeof(int32_t), RAD_MEM_DEVICE));
            if (!rk->vocab_ids_dev) {
                RAD_ERR("step state: no device memory for reduced vocab ids on card %d", rk->device);
                return RAD_E_NOMEM;
            }
            RAD_TRY(rad_memcpy_async(rk->vocab_ids_dev, vocab_ids_host_,
                                     (int64_t)n_vocab_ids_ * (int64_t)sizeof(int32_t),
                                     rk->ctx.stream()));
        }

        RAD_TRY(rad_stream_sync(rk->ctx.stream()));
    }
    if (!ranks_.empty()) RAD_TRY(rad_dev_set(ranks_[0]->device));
    slots_host_ = static_cast<StepSlot*>(rad_dev_alloc(slot_bytes, RAD_MEM_HOST_PINNED));
    if (!slots_host_) return RAD_E_NOMEM;
    RAD_TRY(rad_event_create_as(&step_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    RAD_TRY(rad_event_create_as(&trunk_end_, RAD_EVENT_LOCAL | RAD_EVENT_HOST_WAIT));
    return RAD_OK;
}

void Engine::step_state_fini() {
    for (auto& rk : ranks_) {
        (void)rad_dev_set(rk->device);
        if (rk->step_slots) rad_dev_free(rk->step_slots, RAD_MEM_DEVICE);
        if (rk->eog) rad_dev_free(rk->eog, RAD_MEM_DEVICE);
        if (rk->vocab_ids_dev) rad_dev_free(rk->vocab_ids_dev, RAD_MEM_DEVICE);
        rk->step_slots = nullptr;
        rk->eog = nullptr;
        rk->vocab_ids_dev = nullptr;
        rk->n_vocab_ids = 0;
    }
    if (slots_host_) rad_dev_free(slots_host_, RAD_MEM_HOST_PINNED);
    slots_host_ = nullptr;
    if (vocab_ids_host_) rad_dev_free(vocab_ids_host_, RAD_MEM_HOST_PINNED);
    vocab_ids_host_ = nullptr;
    n_vocab_ids_ = 0;
    if (trunk_end_) rad_event_destroy(trunk_end_);
    trunk_end_ = nullptr;
    if (step_end_) rad_event_destroy(step_end_);
    step_end_ = nullptr;
}

/* ------------------------------------------------------------------ the serial driver
 *
 * `RAD_DRAFT_SERIAL`: one forward pass of the head per drafted token, after the trunk step. Round 1
 * consumes the token the sampler chose and the hidden state of the position that token came from,
 * and which position that is depends on how many drafts the verify accepted. Its pass is the
 * history pass, which the head runs anyway and whose rows already include that position
 * (RadBatch::draft_out_ids).
 *
 * THE CARD DECIDES THAT, NOT THE HOST (sched/advance.h). Behind the trunk's sampler every rank
 * runs the acceptance itself and keeps what it implies in its StepSlots; every draft pass is then
 * staged with its positions and tokens left to the card, and an advance in front of the pass fills
 * them in. So the host issues the whole step -- trunk, history, every round -- without waiting for
 * any of it, and reads the step back once, after the last round. Reading the sampled ids back and
 * staging round 1 from them instead would leave the card idle for a wake, a readback, a commit and
 * a batch build on every step.
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
            if (rk.vocab_ids_dev) {
                a.vocab_ids = rk.vocab_ids_dev;
                a.n_vocab_ids = rk.n_vocab_ids;
            }
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

/* THE STEP'S DRAFT PASSES, ISSUED BEHIND IT WITHOUT WAITING FOR IT. Called once the trunk step's
 * ranks have joined: acceptance on every rank, the sampled ids on their way to the host before a
 * draft round can overwrite them, then the history pass and every round, and last the final
 * proposal into the slots and the slots on their way back. Nothing here synchronises; the step's
 * one wait is collect_step's. */
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

    RAD_TRY(advance_all(ADV_ACCEPT, 0, false));
    if (!so.rb) RAD_TRY(collect_issue(b, so));

    int rounds = drafter_.depth;
    serial_row_.resize((size_t)cfg_.max_seqs);
    /* `n` counts the rows that can DRAFT. It can be zero while the history pass still has work:
     * a sequence three chunks into a long prompt has hidden states the head's attention will need
     * and no token to propose from yet. */
    const int n = sched_.serial_begin(&rounds, serial_row_.data(), (int)serial_row_.size());
    if (rounds <= 0) return RAD_OK;
    serial_rounds_ = rounds;
    serial_rows_   = n;

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

    /* Pass 0, then rounds 2..rounds: round 1 has no pass of its own. */
    for (int rd = 0; rd <= rounds; rd += rd == 0 ? 2 : 1) {
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
         * wide -- the same chain the history pass builds, for the same reason. */
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
    RAD_TRY(advance_all(ADV_TAKE, serial_done_, true, n));
    Rank& r0 = *ranks_[0];
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
    draft_out_.assign((size_t)serial_rows_ * (size_t)rounds, 0);
    for (int i = 0; i < serial_rows_; ++i) {
        const StepSlot& S = slots_host_[serial_row_[(size_t)i].card];
        for (int rd = 1; rd <= done; ++rd)
            draft_out_[(size_t)i * (size_t)rounds + (size_t)(rd - 1)] = S.tok[rd];
    }
    for (int i = 0; i < serial_rows_; ++i) {
        /* A REQUEST THE SAMPLER WILL NOT SPECULATE ON GETS NO DRAFT. That is a constraining
         * grammar whose backend cannot snapshot its state: the mask for draft position i depends
         * on which of 0..i-1 were accepted, and the sampler refuses the whole step rather than
         * masking against a state the sequence is not in -- which would be a hard failure on the
         * next step rather than a skipped draft here. */
        if (!r0.sampler.allows_speculation(serial_row_[(size_t)i].seq)) continue;
        sched_.set_draft(serial_row_[(size_t)i].seq,
                         &draft_out_[(size_t)i * (size_t)rounds], done);
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
int Engine::run_block_drafter(int step) {
    if (drafter_.kind != RAD_DRAFT_BLOCK || drafter_.depth <= 0) return RAD_OK;
    static const bool no_draft = std::getenv("RADIANCE_NO_DRAFT") != nullptr;
    if (no_draft) return RAD_OK;

    const int steps = drafter_.depth;
    /* THE ROW COUNT IS THE DRAFTER'S, not `depth + 1`. Both are legal and the plugin knows which
     * one its checkpoint was trained in; rad_builder.h has the two conventions side by side, and
     * declare_drafter has already defaulted and checked this. */
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

    /* Neither pass samples: the context pass produces no logits at all, and the query pass ends in
     * the drafter's own selector. A zero-row sampler chain is what the serial driver's history
     * pass builds for the same reason. */
    auto run_pass = [&](const RadBatch* db, const char* what) -> int {
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
        /* THE PROBE HAS TO SEE INSIDE THESE TOO. A drafter that runs as extra passes leaves every
         * one of its buffers holding the PREVIOUS step's values at the moment the step loop dumps,
         * so reading them there answers a question one iteration out of date -- which is how a
         * context pass that produces NaN reads as a query pass that consumes it. */
        if (s >= 0 && (int)log_level() >= (int)Log::Trace)
            dump_buffers(r0.ctx, step, std::getenv("RADIANCE_DUMP_BUF"), what);
        return s;
    };

    /* THE CONTEXT PASS RUNS EVEN WHEN NOTHING CAN DRAFT. A sequence three chunks into a long
     * prompt has hidden states the drafter's attention will need and no token to anchor on yet --
     * exactly the case where skipping it would leave a hole nothing ever fills. */
    /* RADIANCE_DRAFT_NOCTX=1 runs no context pass at all, so the query block attends to whatever
     * the drafter's KV cache already held. It answers one question and it is the question that
     * splits the drafter in half: if acceptance does not move, the conditioning is not reaching
     * the query pass and the backbone is drafting from the anchor token alone. */
    static const bool no_ctx = std::getenv("RADIANCE_DRAFT_NOCTX") != nullptr;
    if (!no_ctx)
        if (const RadBatch* cb = sched_.block_context(step)) RAD_TRY(run_pass(cb, "context"));

    /* EMPTY UNTIL THIS CALL'S QUERY PASS FILLS IT. It is a member, so a query batch that could
     * not be built would otherwise leave the previous step's proposals standing -- published
     * below as this step's, and sized for however many rows drafted then, which may be fewer
     * than drafted now. */
    draft_out_.clear();
    if (n > 0) {
        if (const RadBatch* qb = sched_.block_query(step)) {
            RAD_TRY(run_pass(qb, "query"));
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
    sched_.block_end();
    if (n <= 0 || draft_out_.empty()) return RAD_OK;

    for (int i = 0; i < n; ++i) {
        /* A request the sampler will not speculate on gets no draft; see finish_serial_drafter. */
        if (!r0.sampler.allows_speculation(rows[(size_t)i].seq)) continue;
        sched_.set_draft(rows[(size_t)i].seq, &draft_out_[(size_t)i * (size_t)steps], steps);
    }
    return RAD_OK;
}


}  /* namespace rad */
