/* scheduler.h -- continuous batching, chunked prefill, a priority queue and preemption by
 * recompute. One scheduler decision per step, producing one RadBatch (spec §7.1).
 *
 * Host time spent here is GPU idle time. The step path allocates nothing: its storage is sized at
 * max_tok / max_seqs, and allocation happens only on arrival and completion (HTTP threads).
 *
 * Unlike vLLM's scheduler it also owns:
 *   - Chunk boundaries. A prefill chunk splits at a linear-state checkpoint interval, so a
 *     checkpoint is always written instead of only when a step happens to end on a boundary.
 *     The cost is an occasional short chunk; the hit rate then depends on the workload, not on
 *     scheduling luck (§7.3).
 *   - Draft depth, controlled against measured acceptance (draft.h, §10).
 */
#pragma once
#include "batch.h"
#include "draft.h"
#include "draft_confidence.h"
#include "draft_cost.h"
#include "draft_predictor.h"
#include "metrics.h"

#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace rad {

/* The scheduler's own per-request state. Request (rad_core.h) is shared across components and
 * frozen; state only the scheduler needs lives here. */
struct SchedReq {
    std::unique_ptr<Request> req;
    uint64_t arrival = 0;        /* monotonic; the tiebreak below priority. A preempted request
                                  * keeps it, so it re-enters ahead of later arrivals */
    /* The slot's generation: bumped every time this SchedReq is handed to another request. A
     * cost plan names (req_id, slot, gen) so a recycled slot is never read as the request the
     * plan was written for (core/sched/draft_types.h). */
    uint32_t gen = 0;
    int32_t  slot = -1;
    /* The card's index for this sequence: the StepSlot the advance reads and writes (advance.h),
     * and the sequence identity a batch carries. Not `slot`, which is taken on arrival and can
     * exceed the card's max_seqs StepSlots. Taken on admission from a pool of max_seqs and
     * returned with the sequence's KV (free_kv, preempt_back). */
    int32_t  card = -1;
    bool     running = false;

    /* Wall clock, for the live request table; `arrival` is a counter and cannot measure time. */
    int64_t  arrived_us = 0;
    int64_t  first_tok_us = 0;   /* 0 until the first output token: this request's TTFT */

    /* The drafter's proposal for the next step, and the conv-window read cursor a rejection
     * moves rather than rolls back (§10). */
    int32_t  draft[RAD_SCHED_MAX_SPEC] = {};
    int32_t  n_draft = 0;
    int32_t  conv_cursor = 0;
    /* RADIANCE_DRAFT_CONFIDENCE (draft_confidence.h): the head's q(x) and max q for the drafts of
     * the publication standing in `draft`, from set_draft_confidence(); every set_draft() clears
     * them, and the commit of the step that verifies the publication records and clears them. */
    float    conf_q[RAD_MTP_COUPLE_MAX_ROUNDS] = {};
    float    conf_qmax[RAD_MTP_COUPLE_MAX_ROUNDS] = {};
    int32_t  conf_n = 0;

    /* A serial drafter's valid history: positions [0, draft_filled) of the head's own layer hold
     * K and V for the sequence's real tokens. Only the head writes that layer, so before drafting
     * it runs a history pass over what the last step added. */
    int64_t  draft_filled = 0;

    /* The range [draft_lo, draft_hi) of a block drafter's own KV cache that holds real K and V.
     * The low end is non-zero when a prefix hit left the drafter's group uncached
     * (hit_holes_draft_); drafting then waits until the window has slid past it. */
    int64_t  draft_lo = 0;
    int64_t  draft_hi = 0;

    /* Clears all drafter state for the sequence in this slot, on admission and on preemption.
     * Slots are recycled, so anything not reset here leaks into an unrelated sequence: a stale
     * `draft_filled`, for one, empties the next prefill's history pass (hs = max(draft_filled,
     * ctx_len)), and the MTP head drafts against stale pool contents, making greedy output vary
     * per request with speculation on. */
    void drafter_reset() {
        n_draft      = 0;
        draft_next   = 0;
        conv_cursor  = 0;
        conf_n       = 0;
        draft_filled = 0;
        draft_lo     = 0;
        draft_hi     = 0;
    }

    /* The adaptive draft window's estimate (--draft-window adaptive, draft.h). Not reset by
     * drafter_reset(): it describes the text, which preemption does not change. */
    DraftAdapt adapt;
    /* The confidence window (--draft-window confidence, draft.h): the window in force for this
     * cycle, taken when the cycle opens (draft_cycle) from `conf_window_next`, which the last
     * publication's q set (set_draft_confidence) and every publication clears first (set_draft).
     * 0 is the fallback depth, which a cycle with more than one decoding row also takes. Two
     * fields so the four sites of one cycle read one window although the engine publishes the
     * next window's q inside the cycle. */
    int32_t    conf_window = 0;
    int32_t    conf_window_next = 0;
    /* The proposal the step being planned carries came from prompt lookup (StepEntry::ngram). */
    bool       draft_ngram = false;

    /* The cost policy's per-request ledger, charged once per cycle that planned this request and
     * reported once at its end (Scheduler::draft_cost_report): how many such cycles there were, how
     * many of them the policy could price nothing for, and how many tiers it priced in each. A
     * cycle whose shape no bucket matches prices nothing and issues the named baseline, so a run
     * whose profile has no bucket for its row count shows up here as unpriced cycles rather than as
     * a depth decision -- the distinction the plan ledger alone cannot make on such a run. Reset on
     * admission; nothing here is read per step or per token. */
    struct DraftCostLedger {
        int32_t cycles   = 0;
        int32_t unpriced = 0;
        /* Priced tiers per cycle, indexed by how many there were (0 counts the unpriced cycles). */
        int32_t tiers[RAD_DRAFT_MAX_TIERS + 1] = {};
    } cost;

    int32_t  sched_tokens = 0;   /* tokens given to this request in the current step */
    int32_t  entry = -1;         /* its index in the current StepPlan, or -1 */
    /* Its index in the next step, already issued while this one runs (step_next), or -1; and the
     * drafts the drafter issued for it, which that step verifies. */
    int32_t  entry_next = -1;
    int32_t  draft_next = 0;

    /* Cancelled while a step in flight or a drafter snapshot still names it; its blocks are
     * released once nothing does. Freed earlier, a kernel still writing a block would put one
     * sequence's KV in another's attention, and a drafter pass built after the free would carry
     * an all -1 block-table row with a non-zero length, reading block -1 below the pool. */
    bool     pending_free = false;
    /* Reaped in the same window. A disconnect finishes the sink and the server reaps at once;
     * destroying the Request then would leave commit() reading a null `req`, or committing one
     * client's tokens into a re-admitted slot. */
    bool     reap_pending = false;
    int64_t  pending_ckpt = -1;  /* absolute position this step's chunk checkpoints at */

    /* The checkpoint the linear state is seeded from on admission, handed to the sequence's first
     * entry as StepEntry::ckpt_restore. Never held across a step: nothing pins a checkpoint slot,
     * so a failed admission gives the hit back (admit_one) and the next attempt looks it up
     * again. */
    int32_t  ckpt_src = -1;

    /* The prefix hit this admission attempt found, in tokens, and whether its accounting is still
     * owed. Counted once, when admission succeeds; counting each retry of a request that does not
     * fit would inflate the prompt and hit totals. */
    int32_t  hit_attn = 0;
    bool     hit_owed = false;
    /* Why that hit was cut to nothing, for the counters (SchedMetrics::hit_cut_*). */
    enum class HitCut : uint8_t { None, NoSnapshot, Window, Adopt };
    HitCut   hit_cut = HitCut::None;

    /* The whole sequence is prompt ++ output; n_computed indexes into it. */
    int64_t  seq_len() const {
        return (int64_t)req->prompt.size() + (int64_t)req->output.size();
    }
};

/* What the executing path did inside one step's cycle bracket. The engine sees the tape and the
 * placement engines; the scheduler does not. `replay_known` stays false unless a caller that
 * actually knows reports, so a state nobody established is Unknown rather than a measured None
 * (draft_cost.h: the replay state is part of a sample's key). The engine freezes each field on the
 * rank thread as the step's own pass returns (Engine::step_evidence), because the next step's ahead
 * pass is released before this step is collected: a counter delta read here would describe that
 * pass. */
struct DraftStepEvidence {
    bool        replay_known = false;
    DraftReplay replay       = DraftReplay::Unknown;
    bool        staged       = false;  /* the trunk pass armed the pooled experts through the ring */
    bool        moved        = false;  /* the placement engine promoted or demoted */
    /* Per-rank tape classes for the same step, for the trace only (RADIANCE_DRAFT_TRACE): which
     * rank ran what is invisible in the merged state above, and a mixed bracket -- one rank
     * playing, one unevidenced -- is exactly the case the merge rule has to be read against. Never
     * read by the policy or the sample gate. */
    enum { kEvidenceRanks = 8, kEvidenceAbsent = 255 };
    int32_t     n_ranks = 0;                        /* ranks in the deployment */
    uint8_t     tape[kEvidenceRanks] = {};         /* Ctx::StepTape, kEvidenceAbsent if it did not run */
    /* Placement moves on every rank since the cost profile was bound, cumulative. The per-step
     * `moved` above only covers the bracket it names: the scheduler's tier and arena ticks run
     * after step_finished, so a move between two brackets would otherwise never be stamped on the
     * bracket measured after it. > 0 means this run's residency is no longer the one the
     * fingerprint names. */
    int64_t     moves_since_bind = 0;
    /* The start placement is not materialised yet: the plan's lent slots are still being filled
     * (Engine::start_placement_settled). A bracket that ran while that fill was in flight measured
     * a residency no profile names, even when nothing retired inside it -- the mover's own
     * evidence only covers the bracket, and the fill's copies retire over many. */
    bool        placement_unresolved = false;
};

class Scheduler {
public:
    /* What the sampler hands back after a step: per sequence, how many drafts survived, the token
     * sampled at the accepted position, and whether the sequence ended. No logits. */
    struct StepResult {
        int             n_seq = 0;
        const int32_t*  n_accepted = nullptr;  /* [n_seq] drafted tokens accepted, 0..n_spec */
        const int32_t*  n_raw = nullptr;       /* [n_seq] matches before the EOG cut, >= n_accepted */
        const int32_t*  token = nullptr;       /* [n_seq] token sampled at the accepted position */
        const uint8_t*  eos = nullptr;         /* [n_seq] 1 when the sequence ended */
        /* Logprobs, null on a step where no request asked: entry i's j-th emitted token (its
         * accepted drafts first, then the sampled token) is lp[first_row[i] + j]. */
        const TokenLogprobs* lp = nullptr;
        const int32_t*       first_row = nullptr;  /* [n_seq] */
    };

    /* What commit left a sequence at: the length it was committed at and whether it was still
     * Running when its entry was retired. Copied under the lock in commit because reap() may
     * destroy the Request before the serial bind reads it; the plan's StepEntry::req must not be
     * dereferenced after commit. */
    struct CommittedSeq {
        uint64_t seq = 0;
        uint64_t n_out = 0;
        bool     live = false;
    };

    /* `geo` is the already-resolved chunk geometry. The engine resolves it once because
     * PrefixCache::configure runs first and needs its checkpoint interval; pass nullptr to
     * resolve one here. `pass_replay` is the state every pass of this deployment runs under, when
     * the caller can evidence it (`None` for a runtime that cannot record); `Unknown` otherwise. */
    int  init(Program& prog, const Config& cfg, IKVManager* kv, IPrefixCache* pc,
              const std::vector<BatchBuilder::RankIO>& ranks,
              const ChunkGeometry* geo = nullptr,
              DraftReplay pass_replay = DraftReplay::Unknown);

    /* The batch the last build produced for one rank; see BatchBuilder::batch_for. */
    const RadBatch* batch_for(int rank) const { return builder_.batch_for(rank); }
    /* The advance's view of that build on one rank; see BatchBuilder::advance_args. Also carries
     * the serial head's shape, which only the draft passes' modes read. */
    AdvanceArgs advance_args(int rank) const {
        AdvanceArgs a = builder_.advance_args(rank);
        a.shared_kv = serial_shared_ ? 1 : 0;
        a.rot_shift = serial_rot_shift_;
        return a;
    }
    void fini();

    /* ---------------------------------------------------------- off the step path */
    int      add(std::unique_ptr<Request> r);
    /* `n` requests under one hold of the queue lock, each accepted or refused as add() would, with
     * its result in rc[i]. For the sequences of one API request, which must reach the same step. */
    void     add_all(std::unique_ptr<Request>* rs, size_t n, int* rc);
    /* Frees blocks immediately rather than at completion (spec §14); otherwise the pool shrinks
     * under exactly the load that cancels. */
    int      cancel(uint64_t id);
    Request* find(uint64_t id);
    /* The serial head's valid-history end for a sequence (`SchedReq::draft_filled`), or -1 when the
     * id is not live. The watermark is otherwise only visible through the next history pass, which
     * for a history-only row starts where the watermark already is. */
    int64_t  draft_watermark(uint64_t id) const;
    /* Whether the request is still admitted and Running, read under the lock. For an evidence
     * line written after a step's wait, where the staging that named it is older than the wait. */
    bool     running(uint64_t id) const;
    /* What the step just committed left for `seq`, copied by commit under the lock. False when
     * that step carried no such sequence. */
    bool     committed_seq(uint64_t seq, uint64_t* n_out, bool* live) const;

    /* Requests that reached a terminal state since the last call, oldest first, swapped into
     * `out` under the lock (old contents discarded). Only the engine consumes it, to drop
     * per-request sampler state; the server learns through the sink and calls reap(). A swap
     * because cancel() appends from HTTP threads; swapping the same vectors keeps both
     * capacities, so steady state allocates nothing. */
    void     take_completed(std::vector<uint64_t>* out);
    int      reap(uint64_t id);

    /* ---------------------------------------------------------- the lock, for the engine
     *
     * Every mutation of the KV manager and the prefix cache happens under this lock; neither has
     * its own, and both are reached from the step path, from HTTP threads (cancel(), reap(),
     * reset_prefix_cache()) and from the engine's passes between steps (idle tiers, elastic
     * rebalance, checkpoint copies). Taken once per pass that has work, never per sequence.
     *
     * Nothing run under it may call back into the scheduler: the mutex is not recursive and every
     * public method takes it. */
    template <class Fn> auto locked(Fn&& fn) {
        std::lock_guard<std::mutex> lk(mu_);
        return fn();
    }
    /* The same lock, held for the rest of the caller's scope. */
    [[nodiscard]] std::unique_lock<std::mutex> hold() { return std::unique_lock<std::mutex>(mu_); }

    /* ---------------------------------------------------------- work the device still owes
     *
     * A request cancelled while a drafter pass was being built keeps its blocks until the passes
     * that name them are done (SchedReq::pending_free). commit() releases them; an idle engine
     * runs no commit, so it drains its streams and calls release_deferred() instead.
     * has_deferred() is the cheap test for whether that is needed. */
    bool     has_deferred() const;
    void     release_deferred();

    /* The engine has stopped and will run no more steps. Every queued or running request is
     * failed with `reason` so its connection gets an answer, and later submissions are refused the
     * same way. `status` is the cause, and what each of those requests reports. */
    void     shutdown(const char* reason, int status = RAD_E_STATE);

    /* The waiting queue's request ids in service order, for the live view and for tests (a
     * preempted request re-enters at the front). Returns how many were written. */
    int      queue_snapshot(uint64_t* out, int max) const;

    /* ---------------------------------------------------------- who is in flight
     *
     * Per-request state for the dashboard and for debugging a stall. Copied out under the step
     * lock, since a pointer into scheduler storage could outlive a reaped request. */
    struct ReqStat {
        uint64_t id = 0;
        int      state = 0;            /* ReqState, as an int -- the caller has the enum */
        bool     running = false;
        int64_t  prompt_tokens = 0;
        int64_t  computed_tokens = 0;  /* of the prompt, how much is in the KV: prefill progress */
        int64_t  cached_tokens = 0;    /* of those, how many came from the prefix cache */
        int64_t  output_tokens = 0;
        int64_t  max_tokens = 0;
        int64_t  ctx_tokens = 0;       /* prompt ++ output */
        /* This request's share of the paged pools: the blocks it holds and the tokens they cover.
         * It differs from `ctx_tokens` (what it uses) by a partial block and by whatever a
         * rejected draft left. */
        int64_t  kv_blocks = 0;
        int64_t  kv_tokens = 0;
        int32_t  n_draft = 0;          /* the proposal standing for the next step */
        int32_t  slot = -1;
        double   age_s = 0;
        double   ttft_s = -1;          /* negative until the first token */
        double   decode_tps = 0;       /* this request's own, since its first token */
    };
    /* Oldest first, since the oldest is the one most likely stuck. Returns how many were
     * written. */
    int requests(ReqStat* out, int max) const;

    /* The drafter's proposal for a request's next step. n is clamped to the sequence's window
     * (draft_cap): the controller's depth, or what the adaptive policy gives the sequence. */
    int      set_draft(uint64_t id, const int32_t* tokens, int n);

    /* RADIANCE_DRAFT_CONFIDENCE (draft_confidence.h), a diagnostic that changes no plan: whether
     * this run records; the head's q(x) and max q for each draft of the publication set_draft()
     * just made (cut to what it kept); and, at stop, the records written to the path as TSV.
     * init() reads the path from the environment; enable_draft_confidence() is the same for a
     * test. */
    bool     draft_confidence_on() const { return conf_on_; }
    /* Whether the engine should hand the head's q to set_draft_confidence(): the diagnostic
     * above, or --draft-window confidence, whose next window is made from it. */
    bool     wants_draft_confidence() const { return conf_on_ || draft_.confidence(); }
    /* At stop, under --draft-window confidence: one line with the windows verified per depth. */
    void     report_draft_window() const;
    void     enable_draft_confidence(const std::string& path);
    int      set_draft_confidence(uint64_t id, const float* q, const float* q_max, int n);
    int      write_draft_confidence();

    /* ---------------------------------------------------------- media the encoder owes
     * A request carrying media is admitted only once every item is encoded; until then it keeps
     * its queue position and later requests are admitted around it. The engine takes the items
     * (take_encodes) and encodes them between steps, which keeps the encoder off the step path. */
    bool     encodes_pending() const;
    void     take_encodes(std::vector<std::shared_ptr<mm::Item>>* out);
    /* An encoder pass over whole segments; see BatchBuilder::build_encoder. */
    const RadBatch* build_encoder(const std::vector<BatchBuilder::EncoderPart>& parts, int step,
                                  const RadEncoderDecl& enc) {
        std::lock_guard<std::mutex> lk(mu_);
        return builder_.build_encoder(parts, step, enc);
    }

    /* Drop every prefix-cache entry; returns how many, or RAD_E_UNSUPPORTED without a prefix
     * cache. Takes the step lock because the step path inserts into the map (live sequences hold
     * their own block references). See IScheduler::reset_prefix_cache. */
    int64_t  reset_prefix_cache();

    /* ---------------------------------------------------------- the step path */
    /* One decision, one batch. Null when there is nothing to run, which is not an error. */
    const RadBatch* step();

    /* ---------------------------------------------------- the next step, before this one commits
     *
     * A decode step's successor needs nothing the host learns from it: every row continues where
     * the card's acceptance leaves it (sched/advance.h), with drafts proposed on the card, and the
     * host can bound how far that is. So the next forward pass is issued while this one runs.
     *
     *   expect_draft   after the drafter is issued: how many drafts the next step verifies for a
     *               sequence, capped at its window (under the adaptive policy possibly fewer
     *               than the rounds that ran).
     *   step_next      plans and builds that step: every row of this one, as a device row, at
     *               the furthest context it can reach. Null (step() runs after the commit
     *               instead) when only committed state can decide: a request awaiting admission,
     *               a prefill chunk to continue, a grammar, the context's end, or a pool that
     *               cannot grow.
     *   promote_next   once this step has committed: the next step becomes the open one.
     *
     * While the next step is open, commit() leaves continuing requests' blocks in place, and a
     * finished request keeps them until that step has run. */
    void            expect_draft(uint64_t id, int n);
    const RadBatch* step_next();
    const RadBatch* step_next_locked();
    void            promote_next();
    /* Fold the step's outcome back in: append tokens, roll back rejected speculation, retire
     * finished requests, feed the draft controller. */
    int             commit(const StepResult& r);
    /* Ends the step for timing; call once per loop iteration that ran a step, after the drafter,
     * so the draft rounds are charged to the step (see the definition). Without it the counters
     * stay correct but there is no timing. `ev` is what only the caller can see: the replay state
     * the tape ran in and whether placement changed inside the bracket. */
    void            step_finished(const DraftStepEvidence& ev = DraftStepEvidence{});
    /* ---------------------------------------------------- the serial drafter (spec §10)
     *
     * A serial draft head (RAD_DRAFT_SERIAL) runs as extra steps after the trunk step that fed it,
     * because round 1 needs the sampled token and its position, which depends on how many drafts
     * the verify accepted. The engine drives it through:
     *
     *   serial_begin   before commit(), right after the step is issued, while plan() describes it.
     *               Says which sequences may draft, reserves lookahead KV up to the furthest
     *               position acceptance could reach, and clamps `rounds` to the current depth and
     *               then to the deepest row that still drafts. What it writes is the whole set of
     *               passes to run: a round past it has no row, and none may be asked for.
     *   serial_batch   one batch per pass: 0 is the history, which also carries round 1 for the
     *               drafting rows (RadBatch::draft_out_ids), then rounds from 2. A round runs only
     *               the rows whose frozen K reaches it, deepest first, so every round's rows are a
     *               prefix of the snapshot and a row's depth is the last round it was in. Rows
     *               continuing a sampled verify are device rows (advance.h): the card fills their
     *               positions and tokens, so no pass waits on the host. A null return alone does not
     *               say why; read serial_batch_failed() with it.
     *   serial_end     after commit(), with the rounds that ran and whether the history pass did:
     *               moves the history watermark.
     *   set_draft   publishes each row's tokens like any other draft. A row the head declined
     *               keeps n_draft 0 and the n-gram drafter fills in.
     */
    struct SerialRow {
        uint64_t seq = 0;
        int32_t  slot = -1;
        int32_t  card = -1;
    };
    int serial_begin(int* rounds, SerialRow* out, int max);
    const RadBatch* serial_batch(int round, int step);
    /* Round 0 is the history pass (null when no row needs one); round 1 is always null, because
     * round 0 carries it. Call serial_end() once the step has committed.
     *
     * `history_ran` is the driver's word that the round-0 pass it built actually executed: a built
     * batch is not a run pass (the cycle may be rejected, the build may come back null, the host
     * pass may fail), and only the driver sees which. It defaults to false so a caller that cannot
     * report the outcome repeats the history pass instead of claiming a watermark it did not earn. */
    void serial_end(int rounds_done, bool history_ran = false);

    /* Did the last serial_batch() fail to build a pass it owed? A null return alone conflates that
     * with "nothing owed" (empty snapshot, round 1, a round no row reaches), and with a round-0
     * history pass that no drafting row survives into. Not derivable from `n == 0`: the history a
     * K0-only snapshot still owes is exactly what a row count cannot see. Set only where a pass
     * was owed and no batch came back, so the driver can fail the cycle instead of reading the
     * error as idleness. Cleared by serial_begin() and at every serial_batch() entry, so it speaks
     * for the last call only. Orthogonal to `history_ran`: this is "could it be built", that is
     * "did it run". */
    bool serial_batch_failed() const;

    /* ----------------------------------------------------- the block drafter (spec §10)
     *
     * A block-diffusion drafter (RAD_DRAFT_BLOCK) proposes a whole block of `block` positions in
     * one pass, so there are no rounds; instead it runs two passes of different shapes.
     *
     *   block_begin       right after commit(), while plan() still describes the step just run.
     *                  Reserves the draft KV block the query pass writes, and returns which
     *                  sequences may draft.
     *   block_context     the context pass, draft_pass -1. Its entries mirror the trunk step's
     *                  row for row (BlockSeq).
     *   block_query       the query pass, draft_pass +1. One entry per drafting sequence: the
     *                  anchor at its next position, then `block - 1` masks; `out_ids` names the
     *                  last `depth` rows.
     *   block_end         moves the filled range over what actually ran.
     *
     * The drafter writes its proposal to its declared proposal buffer, not through the sampler,
     * and the engine publishes it through set_draft().
     *
     * On the card (`on_card`), block_begin runs before commit() instead, right after the step is
     * issued, so both passes queue behind it with no readback in between. The context pass is
     * unchanged: it mirrors the trunk step, whose positions the host knows. The query pass's
     * anchor depends on acceptance, so its rows are device rows (advance.h, ADV_BLOCK) staged at
     * the furthest position acceptance could leave, and the reservation covers that far.
     * block_end still runs after commit().
     */
    struct BlockRow {
        uint64_t seq = 0;
        int32_t  slot = -1;
        int32_t  card = -1;
    };
    /* `block` is the drafter's trained block in rows, `depth` how many tokens it proposes (the
     * last `depth` rows), `window` its attention window (0 for none) and `mask` the token id its
     * masked query slots embed. Returns how many sequences may draft and fills `out` with them,
     * in the order block_query() emits them. */
    int block_begin(int block, int depth, int window, int32_t mask, BlockRow* out, int max,
                    bool on_card = false);
    const RadBatch* block_context(int step);
    const RadBatch* block_query(int step);
    void block_end();

    const StepPlan&      plan() const { return plan_; }
    const ChunkGeometry& geometry() const { return geo_; }
    DraftController&     draft() { return draft_; }
    /* The observation path's counters, and one request's learned state: what a test (or a
     * deployment's diagnostics) reads to see the hooks are called, not merely configured. The
     * state is meaningful only while the slot holds that (req_id, slot_gen). */
    int64_t draft_cycles_done() const { return draft_cycles_done_; }
    int64_t draft_cycles_dropped() const { return draft_cycles_dropped_; }
    /* A's cost table -- what the cycle samples measured -- and how many samples its observe()
     * refused. Read-only use is the step path's; a deployment (or a test) may configure the
     * profile through the mutable form. */
    const DraftCostTable& draft_costs() const { return draft_costs_; }
    DraftCostTable&       draft_costs() { return draft_costs_; }
    /* Delivers every cycle record that is already fully measured into the cost table -- the same
     * close the shutdown path performs -- so a profile exported from a running engine is not
     * missing the cycles still held in the observation ring. A pending row is closed as dropped,
     * so this is exactly what shutdown would have recorded, only earlier. */
    void flush_draft_observations();
    int64_t               draft_cost_refused() const { return draft_cost_refused_; }
    /* The engine's cumulative residency evidence since the profile was bound, handed in before it
     * opens the next cycle. Same sticky rule step_finished applies inside a bracket: a move the
     * scheduler's own ticks publish between two brackets leaves no per-step stamp, and a cycle
     * priced after it would be priced on a placement the run has left. */
    void                  draft_residency(int64_t moves_since_bind);
    /* Set when a cost cycle was refused because no plan can price the run: the residency drifted
     * from the profile, or the ladder's coverage gap makes the offered top unpriced where the named
     * baseline would run deeper. Either way the run has no profile it can price with, and the
     * engine ends it on its error path rather than drafting at the ceiling. Not a fallback
     * admission; the table stays inactive. */
    bool                  draft_cost_aborted() const { return draft_cost_aborted_; }
    /* Which latch stopped the run: the ladder's coverage gap (the baseline drafts deeper than the
     * ladder's top, so no cycle of this deployment can price it -- a configuration) rather than a
     * residency drift from the profile (an environment the run cannot repair). The engine names the
     * one that fired; both end the run the same way. */
    bool                  draft_cost_gap() const { return draft_cost_gap_; }
    const DraftPredictState* draft_predict_state(int32_t slot) const {
        if (slot < 0 || (size_t)slot >= draft_predict_state_.size()) return nullptr;
        return &draft_predict_state_[(size_t)slot];
    }
    SchedMetrics         metrics() const;

    /* The observers' copy. Dashboards, scrapers and admin calls read what the engine thread last
     * published, never the scheduler's state behind the step lock, so an observer cannot slow
     * the step. publish() runs on the engine thread once a step (at most every 20 ms while idle),
     * builds the copy under mu_ and swaps it in only if no reader holds the copy's lock; a
     * skipped swap happens at the next call. */
    void                 publish(bool idle);
    SchedMetrics         observed_metrics() const;
    int                  observed_requests(ReqStat* out, int max) const;

    /* plan_chunk's tail-checkpoint rule (see the definition). */
    int64_t tail_checkpoint(int64_t target) const;
    /* Visible for tests: the chunk the scheduler would give a request at `n_computed` under a
     * token budget, and the checkpoint position it will land on (-1 for none). `keep_at` is the
     * request's own checkpoint (Request::checkpoint_at), 0 for none. */
    int64_t plan_chunk(int64_t n_computed, int64_t n_target, int64_t budget,
                       int64_t* checkpoint_pos, int64_t keep_at = 0) const;

    /* The smallest step the prefill stager overlaps, in tokens (0 = no stager), set once by the
     * engine after init. --busy-prefill-chunk keeps a held-back chunk at or above it (busy_cap).
     * Also logs the resolved policy at startup. */
    void    set_stage_rows(int64_t rows);
    /* The shortest prefill that spans, in tokens (0: every one alone in its step), set once by
     * the engine after init. Where routed experts are read from host memory, the arena a spanned
     * step takes is room the experts lose for it, which on a short prompt costs more than the span
     * saves (Engine::serve). */
    void    set_span_min(int64_t tokens);
    /* Whether the step just planned leaves a prompt with at least the stager's smallest pass still
     * to prefill, so the next step most likely carries a chunk the stager arms for. */
    bool    prefill_continues() const;
    /* Visible for tests: the most a prefill chunk at `n_computed` of a `n_target`-token sequence
     * may take of `budget` when `reserve` tokens of other work are still to be planned in the step
     * and `shared` says rows are already planned ahead of it. Returns `budget` when
     * --busy-prefill-chunk is off, or when auto has nothing to reserve. */
    int64_t busy_cap(int64_t n_computed, int64_t n_target, int64_t budget, int64_t reserve,
                     bool shared = false, bool rows_only = false) const;

private:
    Program*      prog_ = nullptr;
    Config        cfg_{};
    ChunkGeometry geo_{};
    KVGeom        kvgeom_{};
    /* Does this model carry recurrent state a prefix hit would have to restore? See admit_one. */
    bool          has_linear_state_ = false;
    /* set_stage_rows: the prefill stager's smallest pass, 0 when it is not armed. */
    int64_t       stage_rows_ = 0;
    /* set_span_min: the shortest prefill a span is taken for, 0 for any. */
    int64_t       span_min_ = 0;
    /* Set by admit_one when it held the admitted chunk back: what it would have taken. */
    int64_t       admit_full_ = 0;
    /* Whether this step's prefill may span (ChunkGeometry::span): set by step() for the step it
     * plans, true only for a prefill alone in its step, and read by plan_chunk. True outside a
     * step, so plan_chunk called directly answers for a span. */
    bool          span_now_ = true;
    IKVManager*   kv_ = nullptr;
    IPrefixCache* pc_ = nullptr;

    /* Does a prefix-cache hit leave a hole in a block drafter's own KV? Only if some paged group
     * is not prefix-cached. prefix.cpp skips windowed groups, so a windowed drafter has K and V
     * only from the hit position and must wait for its window to slide past it. A full-attention
     * drafter's group is cached like the trunk's and its floor is 0; treating it as windowed
     * would stop drafting for good, since it has no window to slide. Read off
     * IPrefixCache::cached_groups() so prefix.cpp's rule lives in one place. */
    bool          hit_holes_draft_ = true;
    /* Does a hit leave a hole in the trunk: a paged group with a trunk layer (bound below
     * meta.n_layers; drafter layers bind from `base_layer` up) that the cache does not index?
     * Unlike a drafter hole, recomputing the tail cannot repair it; see admit_one. */
    bool          hit_holes_trunk_ = false;
    /* Draft refill: the drafter's window where a hit can leave a hole in a windowed block
     * drafter's cache, else 0. The drafter would be declined until its window slid past the hole,
     * which on a chat's follow-up turn is most of the reply, so the hit ends one window before the
     * prompt's end (HitCap) and the recompute fills the drafter's window too. Output is unchanged.
     * A hit whose nearest checkpoint below that point gives back more than two windows is kept
     * whole. */
    int64_t       draft_refill_ = 0;
    /* A serial head with no cache of its own (rad_drafter_shared_kv). It owes no history, so
     * serial_begin gives it one row per drafting sequence and none otherwise, and its rounds stay
     * at the committed position (advance.h). False and 0 for every other drafter. */
    bool          serial_shared_ = false;
    int32_t       serial_rot_shift_ = 0;
    BatchBuilder  builder_;
    DraftController draft_;

    /* Where this sequence's block-drafter cache begins; block_begin's guard reads it and
     * block_end records it. */
    int64_t draft_floor(const SchedReq& s, int64_t ctx_len) const {
        if (s.draft_hi != 0 || s.draft_lo != 0) return s.draft_lo;
        return hit_holes_draft_ ? ctx_len : 0;
    }

    /* The window a sequence may draft into: the configured depth, or under the adaptive window
     * min(D, its stop) (DraftController::window_for, which starts the estimate on first use), or
     * under the cost policy what this cycle's frozen plan assigns it, or under the confidence
     * window the one this cycle took from the q of the window in flight. Every site that caps a
     * proposal applies it: step(), set_draft(), expect_draft() and the rounds in serial_begin().
     * With the policy off it is next_depth(). */
    int draft_cap(SchedReq& s) const {
        if (draft_.cost_policy()) {
            const int k = planned_k(s);
            if (k >= 0) return k;
            /* A plan is in force but does not name this row: it was admitted after the cycle
             * opened and was not priced, so it drafts nothing this cycle. Only a cycle whose
             * plan failed outright runs legacy. */
            return draft_.plan() ? 0 : draft_.next_depth();
        }
        if (draft_.confidence())
            return s.conf_window > 0 ? s.conf_window : draft_.confidence_fallback();
        return draft_.adaptive() ? draft_.window_for(s.adapt) : draft_.next_depth();
    }
    /* A policy that gives each sequence its own window, which the rounds, expect_draft() and
     * set_draft() cut to. */
    bool per_seq_window() const {
        return draft_.adaptive() || draft_.cost_policy() || draft_.confidence();
    }

    /* Whether the plan bars the n-gram fallback from this row (draft_plan_bars_fill). A row the
     * plan does not name is barred too while a plan is in force: k[] is the budget for every
     * drafter, and an unpriced row has none. */
    bool bars_fill(const SchedReq& s) const {
        const DraftPlan* p = draft_.plan();
        if (!p) return false;
        if (!s.req) return true;
        for (int32_t i = 0; i < p->n_seq; ++i)
            if (p->req_id[i] == s.req->id && p->slot[i] == s.slot && p->slot_gen[i] == s.gen)
                return draft_plan_bars_fill(*p, i);
        return true;
    }

    /* This cycle's planned K for the request, or -1 when the cost policy is off or its plan
     * does not name it. A planned 0 is a decision: the n-gram fallback must not fill it. */
    int planned_k(const SchedReq& s) const {
        if (!draft_.cost_policy() || !s.req) return -1;
        return draft_.planned_k(s.req->id, s.slot, s.gen);
    }

    /* The step's head depth D is chosen once per cycle by draft_cycle(), the first thing step()
     * and promote_next() do. A proposal passes four sites that must agree on it: the verifying
     * plan, the drafting rounds, expect_draft() and set_draft(). A commit or a disconnect changes
     * the running set mid-cycle, so a D read live could move between them; held, a proposal
     * drafted under one D and planned under the next is only ever cut shorter.
     *
     * `draft_est_` is the list plan_step reads, reserved at init so the step path does not
     * allocate. */
    std::vector<DraftAdapt*> draft_est_;
    /* `direct_step` is the caller: true from step(), false from promote_next(). The cycle's key and
     * its verify/history fields name the window the paired step executes, and only step() builds
     * that step after the plan (Scheduler::draft_cap re-caps it). */
    void    draft_cycle(bool direct_step);

    /* The calibration recording (--draft-profile-out): the same decode rows and the same
     * observation ring as the cost policy, opened under the legacy/adaptive decision instead.
     * Recording never touches the controller's plan, so draft_cap()/bars_fill() and the Cost
     * lock are exactly what they are without it: the ring's record is the only output, and the
     * engine exports it after the run. Armed at init from Config::draft_profile_out. */
    void    draft_record_cycle();
    /* One view per decode row at this cycle's open; the cost policy's cycle and the recording
     * read the same rows. Caller holds mu_. */
    int32_t draft_fill_view();
    /* A cycle with no plan publishes legacy drafts; the step paired with the next plan verifies
     * them. Freeze the count this row really published (DraftPlanner::remember_published), so the
     * next plan declares and prices the step that runs, not the plan before the failure. Caller
     * holds mu_. */
    void    draft_freeze_published(SchedReq& s, int n);
    /* The ahead path's side of the same freeze: the step built under a cycle with no plan verifies
     * exactly its own per-row n_spec, whether or not the head publishes for that row. That second
     * case needs a row the sampler will not speculate on -- a state today's backends cannot
     * produce, so the rule is kept as a guard rather than as something a run reaches. Caller holds
     * mu_. */
    void    draft_freeze_ahead(const StepPlan& p);
    bool    draft_record_ = false;
    /* The serial head's executed depth: min(--num-speculative-tokens, the drafter's own). The
     * recording synthesises each cycle's plan at this bound, because that is the deepest round
     * the head can publish. */
    int32_t draft_head_depth_ = 0;
    /* The depth the previous recording cycle drafted, -1 before the first: the paired step verifies
     * that window, which is what the record's plan carries as history rows. */
    int32_t draft_record_prev_k_ = -1;
    /* A cycle wider than the format's widest window was dropped; said once, counted after. */
    bool    draft_record_over_rows_warned_ = false;

    /* The state the last completed bracket actually ran under, as the ranks froze it
     * (DraftStepEvidence), starting from what the runtime proves about its own tape: `None` when
     * no rank can record (every pass runs unkeyed), `Unknown` when a pass may play, record or
     * audit -- the cold-start answer, and all a run with no completed bracket can know. The next
     * cycle's plan names it as its *candidate* -- a prognosis, not a certificate: the next pass
     * has its own key, so a new or audited key records where the last bracket played. Nothing
     * reads it back as a measurement: the sample is filed, gated and priced under the state the
     * bracket was observed to run (step_finished, draft_cost_observe), and a Record/Audit bucket
     * is never a price (draft_bucket_usable). */
    DraftReplay draft_cycle_replay_ = DraftReplay::Unknown;

    /* The cost policy's cycle: fill a DraftBatchView from the decode rows, let the controller
     * predict and choose, and keep the plan the rest of the cycle reads. Reserved at init, so
     * the step path does not allocate. */
    void    draft_cost_cycle(bool direct_step);
    /* The request's own line for the cost policy: `cycles` planning cycles, `unpriced` of them
     * without a single priced candidate, and the priced-tier histogram. One line per request, at
     * its end (release_locked) -- never per step, and it costs no synchronisation: the scheduler's
     * lock is already held, and the counters are charged where the plan is issued. */
    void    draft_cost_report(const SchedReq& s);
    std::vector<DraftSeqView>    draft_view_seq_;
    std::vector<DraftPrediction> draft_pred_seq_;
    std::vector<uint64_t>        draft_pred_id_;
    DraftBatchView               draft_view_;
    DraftPredictions             draft_preds_;
    uint64_t                     draft_cycle_id_ = 0;
    uint32_t                     slot_gen_ = 0;
    bool                         draft_cost_failed_ = false;
    /* RADIANCE_DRAFT_TRACE: a bounded per-cycle trace of what was planned, what the ranks ran and
     * which shape the sample was filed under. Off by default, silent again after this many lines,
     * and spent only by request-bearing cycles: an idle or prefill cycle plans nothing, so it
     * cannot consume the window before the run reaches the cycles worth logging. A diagnostic for
     * one run, not a log stream. */
    int32_t                      draft_trace_left_ = 64;
    void draft_trace_plan(const DraftPlan& p);
    void draft_trace_bracket(const DraftStepEvidence& ev);
    void draft_trace_sample(const DraftCostSample& s, int rc);

    /* The cost policy's two hooks, owned here: A's table and B's per-request predictor. The
     * predictor keeps one state per request slot; `slot` is the request's for its whole life
     * (preemption keeps it, only a new request bumps the generation), so an estimate survives a
     * preempt/resume without the scheduler copying anything.
     *
     * A's table starts empty: no profile is loaded yet, so every estimate is invalid and the
     * policy issues its named fallback. The sample that feeds it is the cycle ring's own record:
     * step_finished() stamps the step bracket -- total_ms and the executed replay state the engine
     * reports -- onto the open record, and draft_ring_close() hands that record to
     * DraftCostTable::observe. The planner cannot time its own cycle, so the plan it produced
     * says timing_valid = false; the ring's copy is the one the bracket makes valid. */
    DraftCostTable                 draft_costs_;
    DraftPredictModel              draft_predict_;
    std::vector<DraftPredictState> draft_predict_state_;

    /* The cycle ring: a plan stays alive from the cycle that issued it until the step that
     * verifies its drafts has committed, so the observation names the same plan its rows came
     * from. Fixed capacity, storage reserved at init: the step path allocates nothing. A record
     * whose slot is reused is dropped and counted -- a cycle that never completed is not
     * evidence (draft_observation_deliverable), so it is discarded rather than delivered. */
    struct DraftCycleRecord {
        DraftPlan plan;                          /* the copy the observer is handed */
        std::vector<uint64_t>     id;
        std::vector<int32_t>      slot;
        std::vector<uint32_t>     gen;
        std::vector<int32_t>      k;
        std::vector<DraftDrafter> fill;
        std::vector<int32_t>      ctx_class;
        std::vector<int32_t>      history_tokens;
        std::vector<DraftSeqObservation> seq;    /* one per plan row, accumulated */
        std::vector<uint8_t>      sampled;       /* row i: its cycle sampled, not the argmax */
        std::vector<uint8_t>      state;         /* row i: 0 pending, 1 reported, 2 dropped */
        int32_t rows[RAD_SCHED_MAX_SPEC + 1] = {};
        /* The step bracket, filled by step_finished(). `timed` is the honest gate: a step that
         * was never bracketed has no total_ms and cannot be a measurement. */
        double   total_ms = 0.0;
        int32_t  prefill_tokens = 0;             /* prefill tokens that shared the step */
        int64_t  admitted0 = 0;                  /* m_.admitted at open, for the admission flag */
        DraftReplay replay = DraftReplay::Unknown;
        uint8_t  placed = 0;                     /* placement changed inside the bracket */
        bool     timed = false;
        int32_t  n_seq = 0;
        uint64_t cycle_id = 0;
        bool     active = false;
    };
    /* Capacity in cycles, not in steps: a record is delivered at the next cycle's open, so one is
     * in flight; the rest is a backstop for a cycle that never got another open. */
    enum { kDraftRing = 4 };
    DraftCycleRecord draft_ring_[kDraftRing];
    /* The record this cycle's verify and publications belong to, 0 when the open failed. */
    uint64_t         draft_ring_cur_ = 0;
    int64_t          draft_cycles_done_ = 0;      /* observations the predictor accepted */
    int64_t          draft_cycles_dropped_ = 0;   /* incomplete, evicted or refused */
    bool             draft_predict_refused_ = false;  /* the refusal warning was already logged */
    int64_t          draft_cost_refused_ = 0;     /* samples observe() did not take */
    bool             draft_cost_warned_ = false;
    /* The run's residency diverged from the one the profile's fingerprint names (sticky): the
     * scheduler's own ticks move units between brackets, where no per-step evidence reaches. Every
     * record opened after the divergence is marked placed, so no cycle measured on the drifted
     * placement can be priced or exported. */
    bool             draft_drift_ = false;
    /* A cost cycle was refused after the drift: the run cannot price and must not fall back to the
     * ceiling, so the next planning site reports it and the engine stops. One-way. */
    bool             draft_cost_aborted_ = false;
    /* The abort above came from the ladder's coverage gap, not from a residency drift; set by the
     * first latch only, so the engine's diagnostic names the cause that stopped the run. */
    bool             draft_cost_gap_ = false;

    /* The one sticky drift decision, shared by step_finished (the bracket that just closed) and
     * draft_residency (before the next cycle opens). Requires mu_. */
    void draft_drift_note(int64_t moves_since_bind);

    /* The row's label provenance (whose window the verify measured) is stamped at report time;
     * see draft_ring_report(). */
    void draft_ring_open(const DraftPlan& plan);
    DraftCycleRecord* draft_ring_cycle(uint64_t cycle_id);
    void draft_ring_publish(const SchedReq& s, int32_t generated, int32_t published,
                            DraftOrigin origin);
    void draft_ring_report(const SchedReq& s, const StepEntry& e, int32_t accepted,
                           uint64_t raw_match);
    /* The confidence record of the head window this commit verified (draft_confidence.h). */
    void draft_conf_report(SchedReq& s, const StepEntry& e, int32_t accepted, int32_t raw,
                           int32_t rows);
    bool                         conf_on_ = false;
    std::string                  conf_path_;
    std::vector<DraftConfRecord> conf_rec_;
    int64_t                      conf_dropped_ = 0;   /* records past the bound, not kept */
    int64_t                      conf_short_ = 0;     /* windows verified past their q */
    void draft_ring_drop_row(uint64_t req_id, int32_t slot, uint32_t gen);
    void draft_ring_close(uint64_t cycle_id);
    /* Close every record still open, delivering the ones already fully measured and counting the
     * rest as dropped. Idempotent; the scheduler calls it at shutdown and at fini, so the last
     * completed cycle reaches the predictor before its request identities are released. */
    void draft_ring_flush();
    /* Hand one resolved record to A: the executed distribution, the verify window that ran, the
     * step bracket and the evidence step_finished() stamped on it. */
    void draft_cost_observe(DraftCycleRecord& r);

    /* The built-in prompt-lookup drafter (scheduler.cpp). Fills s.draft and returns how many
     * tokens it proposed; 0 when the sequence has no repeated pattern to extend. The pattern
     * lengths it tries, longest first. */
    int draft_ngram(SchedReq& s, int depth) const;
    enum { kNgramMin = 2, kNgramMax = 4 };

    mutable std::mutex mu_;

    SchedMetrics metrics_locked() const;
    int          requests_locked(ReqStat* out, int max) const;
    /* The published copy and its lock (publish()). The scratch pair is the engine thread's, filled
     * under mu_ and then swapped with the published pair, so a publication allocates nothing once
     * the vectors have grown to the request count. */
    mutable std::mutex   pub_mu_;
    SchedMetrics         pub_m_;
    std::vector<ReqStat> pub_reqs_;
    SchedMetrics         scratch_m_;
    std::vector<ReqStat> scratch_reqs_;
    int64_t              pub_at_us_ = 0;

    std::vector<std::unique_ptr<SchedReq>> reqs_;   /* by slot */
    std::vector<int32_t> free_slots_;
    std::vector<int32_t> free_cards_;   /* card indices not held by any sequence; see SchedReq::card */
    std::unordered_map<uint64_t, int32_t> by_id_;

    /* Both kept sorted by (priority desc, arrival asc). running_ is sorted so the preemption
     * victim is the back; waiting_ so admission takes the front. */
    std::vector<int32_t> running_;
    std::vector<int32_t> waiting_;
    std::vector<uint64_t> completed_;

    /* Requests whose release waits on something that still refers to them (a deferred reap or a
     * deferred free). A count, so the sweeps cost nothing on the steps where there are none. */
    int      n_deferred_ = 0;
    /* Items a waiting request needs encoded, in the order its requests reached the queue's head;
     * each at most once (Item::queued). mu_ held. */
    std::vector<std::shared_ptr<mm::Item>> enc_want_;
    /* Are all of this request's items encoded? The ones that are not are queued for the encoder. */
    bool media_ready(SchedReq& s);

    /* Set by shutdown(); add() refuses from then on. */
    bool     closed_ = false;
    const char* closed_reason_ = "";
    int      closed_status_ = RAD_E_STATE;

    /* The longest sequence (prompt plus output) this deployment can address. The batch builder's
     * block tables and the rotary tables are sized for it, so no position at or past it may be
     * scheduled, whatever max_tokens a request carries. */
    int64_t  ctx_limit_ = 0;
    /* The longest sequence the KV pool can ever hold alone, when that is less than ctx_limit_,
     * else -1; and the refusal a longer prompt gets, naming it (a sink keeps the pointer, so it
     * lives here). See add_locked. */
    int64_t     pool_tokens_ = -1;
    std::string pool_refusal_;

    /* Scratch for a prefix-cache hit's block ids, one row per cached group. Reserved at init so
     * an admission does not reach the allocator. */
    std::vector<std::vector<int32_t>> hit_blocks_;
    /* Scratch for publishing a finished sequence into the prefix cache. Reused so close_request
     * does not allocate per completion. */
    std::vector<std::vector<int32_t>> pub_blocks_;

    StepPlan plan_;
    StepPlan plan_next_;
    bool     next_open_ = false;
    /* Is a step the card may still be running holding this request? */
    bool in_step(const SchedReq& s) const {
        return (step_open_ && s.entry >= 0) || (next_open_ && s.entry_next >= 0);
    }

    /* The serial drafter's rows, valid between serial_begin() and serial_end(). */
    struct SerialSeq {
        uint64_t       seq = 0;
        int32_t        slot = -1;
        int32_t        card = -1;
        int32_t        hist_row = -1;   /* the trunk row the history pass starts at */
        int32_t        hist_n = 0;      /* how many positions it has to fill, 0 for none */
        int64_t        hist_pos = 0;    /* the head index that first row is */
        /* The verify step this row continues from, as the card's advance reads it (advance.h):
         * whether it sampled, the token row of its first verified position, its first sampled
         * row, and how many history positions come before the verified ones. */
        bool           dev = false;
        int32_t        dev_tok = 0;
        int32_t        dev_out = -1;
        int32_t        lead = 0;
        /* Its index among the drafting rows, or -1 for a row that only owes history (e.g. a
         * sequence mid-prompt, with hidden states but no token to draft from). */
        int32_t        draft_idx = -1;
        /* The depth this cycle gave the row, fixed at serial_begin: draft round r > 1 runs only
         * the rows with k >= r, and the driver freezes each row's depth from the rounds it saw. */
        int32_t        k = 0;
        /* No Request* here, as for BlockSeq: serial_batch reads serial_ after the lock was
         * released and retaken. */
        /* The furthest position round 1 can write at: the verify step's last position, reached
         * when every draft is accepted. The card knows the real one; this bounds it. */
        int64_t        base_pos = 0;
    };
    std::vector<SerialSeq> serial_;
    StepPlan            serial_plan_;
    /* The last serial_batch() owed a pass and built none; see serial_batch_failed(). */
    bool                serial_failed_ = false;

    /* The block drafter's rows, valid between block_begin() and block_end(): one per entry of the
     * trunk step just run, in order, with `ctx_len`/`n_tokens` mirroring it. The drafter reads
     * the trunk's tapped hidden states positionally from its own buffer, so row i must be the
     * trunk's row i. `can_draft` says whether the row also gets a query pass. */
    struct BlockSeq {
        uint64_t       seq = 0;
        int32_t        slot = -1;
        int32_t        card = -1;
        int32_t        ctx_len = 0;
        int32_t        n_tokens = 0;
        int64_t        base_pos = 0;      /* the anchor's position: the sequence's next one */
        bool           can_draft = false;
        /* On the card: the anchor's position is the card's, and `base_pos` the furthest it can
         * be (Scheduler::block_begin). */
        bool           dev = false;
        /* A placeholder: the trunk step finished this sequence, so it has no KV and drafts
         * nothing, but keeps its row so later rows stay aligned with the trunk's. The pass stores
         * nothing for it (a negative slot is padding). */
        bool           mirror = false;
        /* No Request* here: block_ is read by later passes with the lock released in between,
         * and a disconnect could free the Request. Rows hold `slot` plus `seq`, resolved through
         * block_req(). */
    };
    std::vector<BlockSeq> block_;
    StepPlan           block_plan_;
    int32_t            block_len_ = 0;
    int32_t            block_depth_ = 0;   /* tokens proposed; the head takes the last this many */
    /* `block` - 1 copies of the drafter's mask token, which a query entry's `draft` points at:
     * token_at() walks prompt ++ output ++ draft, so the mask positions come from the accessor
     * every other pass uses. */
    std::vector<int32_t> block_mask_;
    int      step_ = 0;
    uint64_t arrival_ = 0;

    SchedMetrics m_{};
    std::chrono::steady_clock::time_point step_start_{};
    /* When the last step ended. An idle engine runs no steps, so nothing adds time to the
     * throughput window; metrics() decays what it reports by this. */
    std::chrono::steady_clock::time_point last_step_at_ = std::chrono::steady_clock::now();
    int64_t step_prefill_tok_ = 0, step_decode_tok_ = 0;
    /* --prefill-interleave: decode-only steps still owed before the next chunk, and how many
     * prefilling requests the last step() held back. step_next_locked stops issuing ahead once
     * the hold is spent, or the chunk would wait as long as the decoders keep stepping ahead. */
    int64_t interleave_left_ = 0;
    int64_t held_prefill_ = 0;
    /* The throughput window (step_finished): each kind's tokens and the wall time the steps
     * covered, both decayed with the same time constant, so their ratio is tokens over time. */
    double  tps_win_prefill_ = 0.0, tps_win_decode_ = 0.0, tps_win_time_ = 0.0;
    bool    step_open_ = false;
    /* The step just committed: its sequences' committed length and liveness, in plan order. */
    std::vector<CommittedSeq> committed_;
    /* The single-sequence decode floor (rad::DecodeFloor), fed by step_finished. */
    DecodeFloor decode_floor_;

    /* arrival; mu_ held */
    int  add_locked(std::unique_ptr<Request> r);

    /* ordering */
    bool ahead_of(int32_t a, int32_t b) const;
    void queue_insert(std::vector<int32_t>& q, int32_t slot);
    static void queue_erase(std::vector<int32_t>& q, int32_t slot);

    /* lifecycle */
    /* Ends a request: counts it by `st` and queues its id for take_completed. `publish` is
     * false only at shutdown, where a possibly half-computed sequence is not worth indexing. A
     * failed request's sink is failed with `status` and `why`, which reach the client's error;
     * RAD_OK leaves an already-failed sink (an overflow) as it is. */
    void close_request(SchedReq& s, ReqState st, const char* reason, bool defer_free = false,
                       bool publish = true, int status = RAD_OK, const char* why = nullptr);
    void reserve_ckpt(StepEntry& e, const Request& r, int64_t pos, int32_t tok);
    /* What reap() does once nothing points at the slot: give the blocks back, forget the id,
     * destroy the Request and return the slot. Called by reap() directly, or by the sweep in
     * commit() for a request whose reap arrived while something still pointed at it. */
    void release_locked(SchedReq& s);
    /* The sequence's KV back to the manager and its card index back to the pool, together: a step
     * in flight still addresses both, so both wait for the same moment. */
    void free_kv(SchedReq& s);
    void drop_card(SchedReq& s);
    /* Is this request still named by a drafter snapshot? block_ and serial_ outlive the lock. */
    bool block_holds(uint64_t id) const;
    /* Release every deferred reap and free whose request nothing refers to any more. At most one
     * step late: what blocked it is either the step just committed or a drafter snapshot that
     * block_end() / serial_end() has since cleared. */
    void sweep_deferred();
    /* Give back a prefix hit an admission attempt took and then could not use: the adopted
     * blocks, the checkpoint it would have restored, and the position it would have resumed at.
     * The next attempt looks the prompt up again. */
    void undo_hit(SchedReq& s);
    /* The live Request behind a snapshot row, or null if it went away. */
    const Request* block_req(const BlockSeq& d) const;
    const Request* serial_req(const SerialSeq& m) const;
    /* Both snapshots resolve the same way; the two wrappers exist because the row types differ. */
    const Request* live_req(int32_t slot, uint64_t id) const;
    void preempt_back();
    /* Give blocks back from the prefix cache until the pool has headroom again. */
    void trim_prefix_cache();
    /* RAD_OK, or a negative status when the request cannot be admitted now, nor anything behind
     * it. `reserve`: tokens of the step to leave for the requests queued behind this one
     * (--busy-prefill-chunk; busy_cap); 0 plans the chunk on the whole budget. */
    int  admit_one(SchedReq& s, int64_t budget, int64_t reserve = 0);
    /* The queue positions one step's admission took, kept to reuse the allocation. */
    std::vector<size_t> took_;
    /* A step's prefill entries while step() moves its decode entries first; reserved at init,
     * since std::stable_partition would take a temporary buffer from the allocator every step. */
    std::vector<StepEntry> part_;

    /* The conv rolling window, width-1 + num_spec + 1 entries (spec §10). */
    int32_t conv_window() const;
};

}  /* namespace rad */
