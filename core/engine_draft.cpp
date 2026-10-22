/* engine_draft.cpp -- the two drafter drivers (spec §10).
 *
 * A draft head runs as extra passes after the trunk step. SERIAL runs `depth` dependent passes;
 * BLOCK runs a context pass, then one pass that fills the whole block. The architecture plugin
 * declares which (rad_declare_drafter); nothing here depends on the model.
 */
#include "engine_priv.h"

#include "sched/scheduler.h"
#include "device/device.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
 * history pass and every round, each round's proposal published into the slots behind its pass,
 * then the slots' readback. Nothing synchronises here; collect_step has the step's one wait.
 *
 * A null batch means "nothing to run" only when the scheduler says so (serial_batch_failed):
 * round 1, which round 0 carries, or a round whose plan has no row left. It never means "no
 * drafting row": a cycle with no drafting row still owes round 0 the history pass of every
 * history-only row whose hist_n is positive, and a failed build for those rows returns the same
 * null. The drafting-row count cannot tell the two apart, so it is not consulted; a null the
 * cycle owes is refused, because running on would publish a round the card never ran and move
 * the watermark over history never written. The word is per call -- the scheduler resets it
 * before the attempt, sets it on the failure in that same call, and this reads it before the
 * snapshot can end -- so an earlier success cannot cover a later failure. */
int Engine::issue_serial_drafter(int step, StepOut& so) {
    serial_rounds_ = 0;
    serial_rows_   = 0;
    serial_done_   = 0;
    serial_hist_ran_ = false;
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
    /* The depth each row's plan gives it, as the passes below are issued: a round's batch is the
     * active prefix, so row i's own depth is the last round it was still in the batch. */
    serial_k_.assign((size_t)n, 0);

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
        if (!db) {
            /* See above: the scheduler, not the drafting-row count, says whether the null was
             * idle. It answers for the call just made, before the next one overwrites it. */
            if (sched_.serial_batch_failed()) {
                RAD_ERR("drafter '%s': draft pass %d of step %d got no batch and the scheduler "
                        "reports it as failed; the open cycle owes it", drafter_.name, rd, step);
                return RAD_E_STATE;
            }
            continue;
        }
        /* Pass 0 is the history pass, writing the head's own K and V (nothing else does). It is
         * also draft round 1 for drafting rows, whose inputs are the last row of their history run
         * (RadBatch::draft_out_ids), so it ends in the lm_head over those rows. Round r >= 2 embeds
         * round r-1's proposal, which the advance takes from the proposal buffer on the card.
         *
         * The rows this pass proposes for: a round's whole batch, the history pass's drafting rows
         * (first in the batch, RadBatch::draft_out_ids). The rounds shrink to the rows still
         * drafting (E2), so a round's batch is the active prefix and not every drafting row. */
        const int drafted = rd == 0 ? (int)db->n_draft_out : (int)db->n_seq;
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
        /* Publish this pass's proposals into the rows' slots before the prefix can shrink: a row
         * that leaves after this round gets no later pass to read its proposal, and the deepest
         * round's row block no longer names it (StepSlot::gen). The history pass is round 1's
         * pass, so it takes round 1. */
        if (drafted > 0) RAD_TRY(advance_all(ADV_TAKE, rd == 0 ? 1 : rd, true, drafted));
        for (int i = 0; i < drafted && i < serial_rows_; ++i)
            serial_k_[(size_t)i] = rd == 0 ? 1 : rd;
        if (rd > 0) serial_done_ = rd;
        else if (drafted > 0) serial_done_ = 1;
        /* Round 0 is the history pass. Its watermark may move only once it has run on every
         * participating rank, and the host can only confirm rank 0: collect_step's one wait
         * fences step_end_, recorded on rank 0's stream after these passes, while another rank's
         * stream is never waited on here. A second rank therefore leaves the word false, which
         * costs a repeated history pass and no watermark (scheduler.cpp). */
        if (rd == 0) serial_hist_ran_ = ranks_.size() == 1;
    }
    if (serial_done_ <= 0 || n <= 0) return RAD_OK;

    /* Rank 0's slots back to the host; the proposals are already in them (one per row's own
     * round), so nothing else names a round. */
    Rank& r0 = *ranks_[0];
    RAD_TRY(rad_memcpy_async(slots_host_, r0.step_slots,
                             (int64_t)cfg_.max_seqs * (int64_t)sizeof(StepSlot), r0.ctx.stream()));
    return RAD_OK;
}

/* One row's proposals for the step issued ahead. A refusal is fatal: that step was planned with
 * the count expect_draft announced, and the scheduler cannot withdraw a draft for a step it
 * already planned (set_draft only takes the count that step expects), so carrying on would
 * verify the previous cycle's proposals. NOTFOUND is not a refusal, because every caller still
 * holds its snapshot across the call: block_holds() is true for each row, so reap() and cancel()
 * defer the release instead of dropping it (scheduler.cpp), and the request is still in by_id_.
 * The one way left is a concurrent Scheduler::shutdown, which clears the snapshot first; by then
 * no step is left that could verify the old proposals. The host block driver has no step issued
 * ahead at all (pipelining is off there), so it never has one to lose either. */
int Engine::publish_draft(uint64_t seq, const int32_t* tok, int n) {
    const int st = sched_.set_draft(seq, tok, n);
    if (st == RAD_E_NOTFOUND) return RAD_OK;
    if (st < 0)
        RAD_ERR("drafter '%s': sequence %llu: publishing %d draft(s): %s", drafter_.name,
                (unsigned long long)seq, n, rad_strerror(st));
    return st;
}

namespace {
/* Ends the serial snapshot on every exit of the finish below, in the destructor, so that it runs
 * after the last set_draft: the hold is what keeps reap() from dropping a row between the cycle
 * and its publication. A refused cycle ends it here instead of leaving that to
 * Scheduler::shutdown, and does so with 0 rounds, which suppresses the draft rows' verified
 * positions -- the cycle's work is thrown away -- while a history-only row still moves if round 0
 * really ran (`history_ran`, which the driver sets only on proof). */
struct SerialHold {
    Scheduler& sched;
    const bool hist_ran;
    bool held = true;
    SerialHold(Scheduler& s, bool h) : sched(s), hist_ran(h) {}
    ~SerialHold() { if (held) sched.serial_end(0, hist_ran); }
    void release(int rounds) { held = false; sched.serial_end(rounds, hist_ran); }
};

/* The block host driver's snapshot, armed once the context pass has run: block_end() moves the
 * filled range over that pass, so ending the snapshot before it -- or after it failed -- would
 * claim a range nobody wrote. Those exits stay with Scheduler::shutdown (fail_loop), with step()'s
 * clear as the next cycle's backstop. */
struct BlockHold {
    Scheduler& sched;
    explicit BlockHold(Scheduler& s) : sched(s) {}
    ~BlockHold() { sched.block_end(); }
};
}  /* namespace */

/* After commit: check the host's acceptance against the card's, publish the proposals through
 * set_draft, and move the history watermark. The card's acceptance positioned every draft pass,
 * so a disagreement is a defect and the step is refused. Each row publishes StepSlot::gen rounds,
 * which the card wrote per row: the cycle's deepest round is not every row's count once the rounds
 * shrink to the active prefix. */
int Engine::finish_serial_drafter(const StepOut& so) {
    if (!serial_drafting() || serial_rounds_ <= 0) return RAD_OK;
    SerialHold hold(sched_, serial_hist_ran_);
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
    if (serial_done_ <= 0 || serial_rows_ <= 0) return RAD_OK;   /* round 0 only: hold ends it, with history_ran */

    Rank& r0 = *ranks_[0];
    const int rounds = serial_rounds_;
    draft_out_.assign((size_t)serial_rows_ * (size_t)rounds, 0);
    /* Each row publishes the rounds it actually finished (StepSlot::gen, written by ADV_TAKE),
     * not the cycle's deepest: with shrinking rounds a row that left the prefix after its own
     * K has no proposal past it. The count is checked against the frozen plan, not the cycle
     * depth: a row with k = 2 that reports 4 would publish whatever the buffer holds at its row
     * index from a round it never ran, and `gen < k` is a publication the card dropped. So the
     * row must report exactly the rounds its own plan gave it, and the plan must still name that
     * sequence in that slot -- a slot recycled between issue and finish would otherwise publish
     * another request's proposals. */
    for (int i = 0; i < serial_rows_; ++i) {
        const Scheduler::SerialRow& R = serial_row_[(size_t)i];
        /* Searched per row: serial_begin emits the drafting rows in plan order today, but the
         * cycle sorts them by descending K, and a cursor would then walk past a row's entry. */
        size_t pe = 0;
        while (pe < plan.e.size() && plan.e[pe].seq != R.seq) ++pe;
        const int k = serial_k_[(size_t)i];
        if (pe >= plan.e.size() || plan.e[pe].card != R.card || plan.e[pe].slot != R.slot) {
            RAD_ERR("drafter '%s': sequence %llu drafted in card slot %lld, which the step's "
                    "plan no longer names for it", drafter_.name, (unsigned long long)R.seq,
                    (long long)R.card);
            return RAD_E_STATE;
        }
        const StepSlot& S = slots_host_[R.card];
        if (S.gen != k) {
            RAD_ERR("drafter '%s': sequence %llu finished %d draft round(s); its plan ran %d of "
                    "the cycle's %d", drafter_.name, (unsigned long long)R.seq, (int)S.gen, k,
                    rounds);
            return RAD_E_STATE;
        }
        for (int rd = 1; rd <= k; ++rd)
            draft_out_[(size_t)i * (size_t)rounds + (size_t)(rd - 1)] = S.tok[rd];
    }
    int st = RAD_OK;
    for (int i = 0; i < serial_rows_ && st >= 0; ++i) {
        /* No draft for a request the sampler will not speculate on (a constraining grammar that
         * cannot snapshot its state); the sampler would refuse the next step otherwise. */
        if (!r0.sampler.allows_speculation(serial_row_[(size_t)i].seq)) continue;
        st = publish_draft(serial_row_[(size_t)i].seq,
                           &draft_out_[(size_t)i * (size_t)rounds], serial_k_[(size_t)i]);
    }
    /* The snapshot ends with hold's scope, after the publish: serial_end releases the hold that
     * defers the release of a request cancelled or reaped while the rounds ran. Dropped earlier,
     * the server's reap could take a request out of the scheduler between the end and its
     * set_draft above and that publication would be lost. A failed publication releases with 0
     * rounds: the cycle is refused, so its verified positions are redone, not recorded. */
    hold.release(st < 0 ? 0 : serial_done_);
    return st;
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
    /* Before block_begin: the proposal buffer is the rank's, not the block plan's, so a missing
     * one returns before the snapshot exists. */
    Rank& r0 = *ranks_[0];
    void* prop = r0.ctx.buf_ptr(drafter_.proposal);
    if (!prop) {
        RAD_ERR("drafter '%s': `%s` was declared as the proposal buffer but never planned",
                drafter_.name, r0.program.buffers[drafter_.proposal].name.c_str());
        return RAD_E_STATE;
    }
    std::vector<Scheduler::BlockRow> rows((size_t)cfg_.max_seqs);
    const int n = sched_.block_begin(block, steps, (int)drafter_.window, drafter_.mask_token,
                                  rows.data(), (int)rows.size());

    /* The context pass runs even when nothing drafts: a mid-prompt sequence has hidden states the
     * drafter will need, and skipping would leave a permanent hole in its cache. A step with rows
     * owes that mirror pass, so nullptr is a failed build and not "no pass needed"; a failure here
     * still has to clear the snapshot, but block_end would then claim the range this pass did not
     * write, so the snapshot is left to Scheduler::shutdown (fail_loop), with step()'s clear as the
     * next cycle's backstop. */
    const RadBatch* cb = sched_.block_context(step);
    if (!cb && !sched_.plan().e.empty()) {
        RAD_ERR("drafter '%s': no context batch for step %d, which has %zu row(s); the scheduler's "
                "build failed", drafter_.name, step, sched_.plan().e.size());
        return RAD_E_STATE;
    }
    if (cb) RAD_TRY(block_pass(cb, "context", step));

    /* The context pass ran: every exit below -- a failed query pass, a readback that could not be
     * reserved or copied, the publish -- ends the snapshot in hold's scope. */
    BlockHold hold(sched_);

    /* Cleared so a query batch that could not be built does not republish the previous step's
     * proposals. A row that may draft was named by block_begin, so its query batch is owed:
     * nullptr is a failed build, and carrying on would publish nothing for rows the step issued
     * ahead waits on. */
    draft_out_.clear();
    if (n > 0) {
        const RadBatch* qb = sched_.block_query(step);
        if (!qb) {
            RAD_ERR("drafter '%s': no query batch for step %d with %d row(s) drafting; the "
                    "scheduler's build failed", drafter_.name, step, n);
            return RAD_E_STATE;
        }
        RAD_TRY(block_pass(qb, "query", step));
        const int64_t n_raw = (int64_t)n * steps;
        int32_t* const raw = readback_.reserve(n_raw);
        if (!raw) return RAD_E_NOMEM;
        RAD_TRY(rad_memcpy_async(raw, prop, n_raw * 4, r0.ctx.stream()));
        RAD_TRY(rad_stream_sync(r0.ctx.stream()));
        draft_out_.assign(raw, raw + n_raw);
    }
    if (draft_out_.empty()) return RAD_OK;

    int st = RAD_OK;
    for (int i = 0; i < n && st >= 0; ++i) {
        /* A request the sampler will not speculate on gets no draft; see finish_serial_drafter. */
        if (!r0.sampler.allows_speculation(rows[(size_t)i].seq)) continue;
        st = publish_draft(rows[(size_t)i].seq, &draft_out_[(size_t)i * (size_t)steps], steps);
    }
    /* As serial_end in finish_serial_drafter: the snapshot ends after the publish, so the hold
     * keeps reap() from dropping a row between the end and its set_draft. */
    return st;
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

    /* Even when nothing drafts; see run_block_drafter. A step with rows owes this mirror pass, so
     * nullptr is a failed build there, not an idle pass. */
    const RadBatch* cb = sched_.block_context(step);
    if (!cb && !sched_.plan().e.empty()) {
        RAD_ERR("drafter '%s': no context batch for step %d, which has %zu row(s); the scheduler's "
                "build failed", drafter_.name, step, sched_.plan().e.size());
        return RAD_E_STATE;
    }
    if (cb) RAD_TRY(block_pass(cb, "context", step));
    if (n <= 0) return RAD_OK;
    const RadBatch* qb = sched_.block_query(step);
    if (!qb) {
        RAD_ERR("drafter '%s': no query batch for step %d with %d row(s) drafting; the scheduler's "
                "build failed", drafter_.name, step, n);
        return RAD_E_STATE;
    }
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
    if (block_rows_ <= 0) {
        sched_.block_end();
        return RAD_OK;
    }

    Rank& r0 = *ranks_[0];
    const int steps = drafter_.depth;
    draft_out_.assign((size_t)block_rows_ * (size_t)steps, 0);
    for (int i = 0; i < block_rows_; ++i) {
        const StepSlot& S = slots_host_[block_row_[(size_t)i].card];
        for (int j = 0; j < steps; ++j)
            draft_out_[(size_t)i * (size_t)steps + (size_t)j] = S.tok[1 + j];
    }
    int st = RAD_OK;
    for (int i = 0; i < block_rows_ && st >= 0; ++i) {
        /* As in finish_serial_drafter: the step issued ahead only carries a request whose answer
         * is fixed (Scheduler::step_next_locked), so this agrees with the count it verifies. */
        if (!r0.sampler.allows_speculation(block_row_[(size_t)i].seq)) continue;
        st = publish_draft(block_row_[(size_t)i].seq, &draft_out_[(size_t)i * (size_t)steps],
                           steps);
    }
    sched_.block_end();
    return st;
}

}  /* namespace rad */
