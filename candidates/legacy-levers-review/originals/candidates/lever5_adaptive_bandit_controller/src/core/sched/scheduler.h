/* scheduler.h -- continuous batching, chunked prefill, a priority queue, preemption by recompute.
 * ONE scheduler decision per step, producing ONE RadBatch (spec §7.1).
 *
 * Host time spent here is time the GPU spends idle, so this component is written to be cheap and
 * to build things once. Everything on the step path comes out of a pool sized at max_tok /
 * max_seqs; the only allocation the scheduler performs is on arrival and completion, which are
 * the HTTP thread's problem and not the step's.
 *
 * Two things it owns that vLLM's does not:
 *
 *   - CHUNK BOUNDARIES, and it uses that. A prefill chunk splits at a linear-state checkpoint
 *     interval, so a checkpoint is always WRITTEN rather than landing wherever a step happened to
 *     end. vLLM writes one only when a step's end coincides with a boundary, which is why a
 *     checkpoint landing in request-unique tokens silently drops linear caching to zero. The cost
 *     of doing it this way is an occasional short chunk; the benefit is that hit rate is a
 *     property of the workload rather than of scheduling luck (§7.3).
 *
 *   - DRAFT DEPTH, controlled against measured acceptance (draft.h, §10).
 */
#pragma once
#include "batch.h"
#include "draft.h"
#include "metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace rad {

/* Wall-clock microseconds on a monotonic base. */
static inline int64_t mono_us() {
    return (int64_t)std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

/* The scheduler's own per-request state. Request (rad_core.h) is shared across components and
 * frozen; anything only the scheduler needs lives here, so adding a scheduling concept never
 * touches a struct three other components read. */
struct SchedReq {
    std::unique_ptr<Request> req;
    uint64_t arrival = 0;        /* monotonic; the tiebreak below priority, and what makes a
                                  * preempted request re-enter AHEAD of everything admitted after
                                  * it rather than at the back of the queue */
    int32_t  slot = -1;
    /* THE CARD'S INDEX FOR THIS SEQUENCE: which StepSlot the advance reads and writes (advance.h),
     * and so the sequence identity a batch carries. It is NOT `slot`. A slot is taken on arrival,
     * so a queue behind the admission limit -- or a burst arriving while the last one's requests
     * are still held for their final commit -- numbers slots past max_seqs, and the card has
     * max_seqs StepSlots. The card index is taken on admission from a pool of exactly that many
     * and given back with the sequence's KV (free_kv, preempt_back), so it lives as long as any
     * step in flight can still name it and never longer. */
    int32_t  card = -1;
    bool     running = false;

    /* WALL CLOCK, for the live request table. `arrival` above is a monotonic COUNTER -- it
     * orders the queue and cannot say how long anything has been in it. */
    int64_t  arrived_us = 0;
    int64_t  first_tok_us = 0;   /* 0 until the first output token: this request's TTFT */

    /* The drafter's proposal for the next step, and the conv-window read cursor a rejection
     * moves rather than rolls back (§10). */
    int32_t  draft[RAD_SCHED_MAX_SPEC] = {};
    int32_t  n_draft = 0;
    int32_t  conv_cursor = 0;
    int32_t  accepted_streak = 0;
    int32_t  rejected_streak = 0;
    int32_t  last_accepted = 3;
    float    ema_accept = 2.8f;

    /* --- Multi-Armed Bandit Speculative Controller (Pillars 1-3) --- */
    static constexpr int BANDIT_MAX_ARMS = 7; /* arms K=0..6 */
    static constexpr int ACCEPT_HISTORY_SIZE = 8;
    int32_t  bandit_pulls[BANDIT_MAX_ARMS] = { 1, 2, 2, 4, 5, 4, 1 };
    float    bandit_mean_tps[BANDIT_MAX_ARMS] = { 0.10f, 0.15f, 0.20f, 0.26f, 0.30f, 0.28f, 0.22f };
    int32_t  bandit_tot_pulls = 19;
    int32_t  bandit_last_arm = 4;
    int64_t  last_commit_us = 0;
    int32_t  accept_length_history[ACCEPT_HISTORY_SIZE] = { 3, 3, 3, 3, 3, 3, 3, 3 };
    int32_t  history_cursor = 0;

    void init_bandit(int default_depth) {
        bandit_tot_pulls = 0;
        last_commit_us = 0;
        bandit_last_arm = default_depth > 0 ? std::min(default_depth, 5) : 4;
        bandit_pulls[0] = 1; bandit_mean_tps[0] = 0.10f;
        bandit_pulls[1] = 2; bandit_mean_tps[1] = 0.15f;
        bandit_pulls[2] = 2; bandit_mean_tps[2] = 0.20f;
        bandit_pulls[3] = 4; bandit_mean_tps[3] = 0.26f;
        bandit_pulls[4] = 5; bandit_mean_tps[4] = 0.30f;
        bandit_pulls[5] = 4; bandit_mean_tps[5] = 0.28f;
        bandit_pulls[6] = 1; bandit_mean_tps[6] = 0.22f;
        bandit_tot_pulls = 19;
        for (int i = 0; i < ACCEPT_HISTORY_SIZE; ++i) accept_length_history[i] = 3;
        history_cursor = 0;
    }

    void observe_draft(int drafted, int accepted, int64_t step_time_us = 0) {
        if (drafted <= 0) return;
        last_accepted = accepted;
        accept_length_history[history_cursor++ % ACCEPT_HISTORY_SIZE] = accepted;

        if (accepted >= 3 || (accepted == drafted && accepted >= 2)) {
            ++accepted_streak;
            rejected_streak = 0;
        } else if (accepted == 2) {
            accepted_streak = std::min(accepted_streak, 1);
            rejected_streak = 0;
        } else if (accepted == 1) {
            accepted_streak = 0;
            rejected_streak = 0;
        } else {
            accepted_streak = 0;
            ++rejected_streak;
        }
        ema_accept = 0.70f * ema_accept + 0.30f * (float)accepted;

        /* Update Bandit statistics for arm K = drafted */
        int k = drafted;
        if (k < 1) k = 1;
        if (k >= BANDIT_MAX_ARMS) k = BANDIT_MAX_ARMS - 1;

        int64_t now_us = mono_us();
        int64_t seq_elapsed_us = (last_commit_us > 0) ? (now_us - last_commit_us) : 0;
        last_commit_us = now_us;

        /* Step latency model: measure true sequence interval if valid,
         * or fallback to step_time_us / hardware GEMM profile */
        float elapsed_ms;
        if (seq_elapsed_us >= 4000 && seq_elapsed_us <= 40000) {
            elapsed_ms = (float)seq_elapsed_us / 1000.0f;
        } else if (step_time_us >= 4000 && step_time_us <= 40000) {
            elapsed_ms = (float)step_time_us / 1000.0f;
        } else {
            elapsed_ms = 7.5f + 0.90f * (float)k;
        }

        const float obs_tps = (float)(1 + accepted) / elapsed_ms;
        const int pulls = ++bandit_pulls[k];
        ++bandit_tot_pulls;
        const float alpha = std::max(0.12f, 1.0f / (float)std::min(pulls, 15));
        bandit_mean_tps[k] = (1.0f - alpha) * bandit_mean_tps[k] + alpha * obs_tps;
        bandit_last_arm = k;
    }

    int suggest_depth(int default_depth, bool adaptive, bool multi_seq = false, int max_cap = 5) const {
        if (!adaptive) return default_depth;
        if (max_cap <= 0) return 0;
        const int cap = std::min(max_cap, BANDIT_MAX_ARMS - 1);
        if (cap <= 1) return cap;

        /* In multi_seq concurrency (e.g. C8), each sequence operates independently (Pillar 3).
         * Under high concurrency, cap at 4 to prevent tail latency degradation. */
        const int effective_cap = multi_seq ? std::min(cap, 4) : cap;

        /* Lower Confidence Bound (LCB) ranking (Pillar 2):
         * Evaluates true measured throughput across arms K=1..effective_cap. */
        const float total_ln = std::log((float)(bandit_tot_pulls + 2));
        const float c_lcb = multi_seq ? 0.015f : 0.010f;

        int best_arm = std::min(default_depth > 0 ? default_depth : 3, effective_cap);
        float best_lcb = -1e9f;

        for (int k = 1; k <= effective_cap; ++k) {
            const int n_k = std::max(1, bandit_pulls[k]);
            const float uncertainty = c_lcb * std::sqrt(total_ln / (float)n_k);
            const float lcb = bandit_mean_tps[k] - uncertainty;
            if (lcb > best_lcb) {
                best_lcb = lcb;
                best_arm = k;
            }
        }

        /* Exploration: periodically probe adjacent arm */
        if ((bandit_tot_pulls & 0x7) == 0 && bandit_tot_pulls > 0) {
            if (best_arm < effective_cap && bandit_pulls[best_arm + 1] <= bandit_pulls[best_arm]) {
                return best_arm + 1;
            } else if (best_arm > 2 && bandit_pulls[best_arm - 1] < bandit_pulls[best_arm]) {
                return best_arm - 1;
            }
        }

        /* Dynamic Floor Removal (Pillar 1):
         * When text is consistently unpredictable (repeated rejections with low EMA acceptance),
         * dynamically lower depth down to K=1 (or 2) to eliminate verification bottlenecks. */
        if (rejected_streak >= 3 || ema_accept < 0.8f) {
            return 1;
        } else if (rejected_streak >= 2 || ema_accept < 1.2f) {
            return std::clamp(best_arm, 1, 2);
        } else if (rejected_streak >= 1 && ema_accept < 1.6f) {
            return std::clamp(best_arm, 1, 3);
        }

        /* Dynamic promotion on fluent streaks:
         * In single-stream, quickly reach depth 5 for maximum throughput (> 200 t/s).
         * In concurrency (C8), carefully promote to depth 4 to balance throughput and latency. */
        if (!multi_seq) {
            if ((accepted_streak >= 2 && last_accepted >= 3) ||
                (accepted_streak >= 1 && ema_accept >= 2.0f && last_accepted >= 2) ||
                (ema_accept >= 2.30f) || (best_arm >= 5)) {
                return std::min(effective_cap, 5);
            }
            if ((accepted_streak >= 1 && last_accepted >= 2) ||
                (ema_accept >= 1.70f) || (best_arm >= 4)) {
                return std::min(effective_cap, 4);
            }
            return std::clamp(best_arm, 3, effective_cap);
        } else {
            if ((accepted_streak >= 2 && ema_accept >= 2.40f && last_accepted >= 3) ||
                (accepted_streak >= 3 && ema_accept >= 2.20f && last_accepted >= 2)) {
                return std::min(effective_cap, 4);
            }
            if ((accepted_streak >= 1 && ema_accept >= 1.85f && last_accepted >= 2) ||
                (best_arm >= 4)) {
                return std::min(effective_cap, 4);
            }
            return std::clamp(best_arm, 2, effective_cap);
        }
    }

    /* HOW MUCH OF A SERIAL DRAFTER`S OWN ATTENTION HISTORY IS VALID: positions [0, draft_filled)
     * of the head`s layer hold K and V computed from the tokens the sequence actually has. The
     * head attends to its own layer and nothing else in the engine writes it -- a prefill fills
     * the trunk`s layers and leaves the head`s untouched -- so the drafter runs a history pass
     * over whatever the last step added before it drafts. Zero on admission and on preemption,
     * which is what makes a recomputed sequence refill it rather than trust a stale watermark. */
    int64_t  draft_filled = 0;

    /* THE RANGE OF A BLOCK DRAFTER'S OWN KV CACHE THAT HOLDS REAL K AND V, as [draft_lo,
     * draft_hi). Two numbers rather than a serial head's one because the low end CAN move: such a
     * drafter has a KV group of its own, and whether a hit leaves that group short depends on
     * whether prefix.cpp caches it -- see hit_holes_draft_. A WINDOWED drafter's group is skipped
     * (a hit's blocks do not describe a cache whose floor moves), so a sequence that hits at
     * position P has trunk K and V from 0 and draft K and V only from P, and drafting waits until
     * the drafter's window has slid past P rather than attending to blocks nobody wrote.
     *
     * A FULL-ATTENTION DRAFTER'S GROUP IS CACHED LIKE THE TRUNK'S and the floor stays 0. Treating
     * one as windowed disables drafting outright: a windowless drafter has no window to slide, so
     * "wait until the window has passed P" is a condition that can never come true, and a workload
     * that hits the prefix cache on every turn would then never draft again.
     * Both zero on admission and on preemption, which is what makes a recomputed sequence refill
     * rather than trust a stale watermark. */
    int64_t  draft_lo = 0;
    int64_t  draft_hi = 0;

    /* EVERY PIECE OF DRAFTER STATE THAT DESCRIBES THE SEQUENCE IN THIS SLOT, cleared together.
     *
     * A slot is RECYCLED: add() takes one off `free_slots_` and the SchedReq in it is the last
     * request's object, so anything not reset here is inherited by a sequence that has nothing to
     * do with it. Both watermarks above say "zero on admission and on preemption", and a list of
     * fields spelled out at each call site is what lets one of them go unreset; hence one function,
     * called from both.
     *
     * WHAT A SURVIVING WATERMARK COSTS. `draft_filled` carried into the next request at the
     * previous one's total length makes its prefill compute `hs = max(draft_filled, ctx_len)` =
     * the stale watermark, `he` clamps up to meet it, and the history pass comes out EMPTY -- so
     * the MTP head's layer of the attention cache is never written for the whole prompt and the
     * head drafts against whatever the pool last held. The drafts are then wrong, so acceptance
     * moves, so the committed lengths move, so the arithmetic of every later step moves: the same
     * greedy prompt yields a different completion per request, reproducible request-for-request
     * across processes, and only ever with speculation on. */
    void drafter_reset() {
        n_draft         = 0;
        draft_next      = 0;
        conv_cursor     = 0;
        draft_filled    = 0;
        draft_lo        = 0;
        draft_hi        = 0;
        accepted_streak = 0;
        rejected_streak = 0;
        last_accepted   = 3;
        ema_accept      = 2.8f;
        init_bandit(3);
    }

    int32_t  sched_tokens = 0;   /* tokens given to this request in the current step */
    int32_t  entry = -1;         /* its index in the current StepPlan, or -1 */
    /* Its index in the NEXT step, already issued while this one runs (step_next), or -1; and the
     * drafts the drafter issued for it, which that step verifies. */
    int32_t  entry_next = -1;
    int32_t  draft_next = 0;

    /* Cancelled while its tokens were in the step the device is still running, or while a
     * drafter's snapshot still names it. Its blocks are released at commit rather than now,
     * because handing a block to another sequence while a kernel is still writing it is a data
     * race whose symptom is one sequence's KV appearing in another's attention -- and it would
     * present as a model quality problem. A drafter pass built after the free would be worse
     * still: its block-table row is all -1 with a non-zero length, and attention reads block -1,
     * which is below the layer's region of the pool. */
    bool     pending_free = false;
    /* REAPED WHILE ITS TOKENS WERE IN THE STEP THE DEVICE IS STILL RUNNING, which is the same
     * window `pending_free` covers and a strictly worse thing to get wrong. The server reaps a
     * generation as soon as its sink is finished, and cancelling a request finishes the sink
     * immediately -- so a client that disconnects mid-decode has its Request destroyed and its
     * slot handed back while commit() still holds a StepEntry pointing at that slot. commit() then
     * dereferences a null `req`, or worse, finds the slot re-admitted and commits one client's
     * tokens into another's output. The release is deferred to commit instead. */
    bool     reap_pending = false;
    int64_t  pending_ckpt = -1;  /* absolute position this step's chunk checkpoints at */

    /* The checkpoint the linear state should be SEEDED from on admission. Neither Request nor
     * KVManager::ensure carries it, so the scheduler holds the answer here and hands it to the
     * sequence's first entry as StepEntry::ckpt_restore.
     *
     * NEVER HELD ACROSS A STEP. A checkpoint slot is only the cache's until the cache retires it,
     * and between two steps the retention policy, another admission's reservation or an idle
     * tier can retire it, reallocate it and save a different sequence into it -- nothing pins it.
     * So an admission that fails after the lookup gives the hit back (admit_one) and the next
     * attempt looks it up again, rather than restoring whatever the slot holds by then. */
    int32_t  ckpt_src = -1;

    /* The prefix hit this admission attempt found, in tokens, and whether its accounting is still
     * owed. Counted once, when the admission SUCCEEDS: a request at the head of the queue that
     * does not fit is retried every step, and counting each attempt would inflate the prompt and
     * hit totals by however long it waited. */
    int32_t  hit_attn = 0;
    bool     hit_owed = false;

    /* The whole sequence is prompt ++ output; n_computed indexes into it. */
    int64_t  seq_len() const {
        return (int64_t)req->prompt.size() + (int64_t)req->output.size();
    }
};

class Scheduler {
public:
    /* What the sampler hands back after a step. The scheduler needs three numbers per sequence and
     * no logits: how many drafted tokens survived, the token sampled at the accepted position, and
     * whether the sequence ended. Everything else is the sampler's business. */
    struct StepResult {
        int             n_seq = 0;
        const int32_t*  n_accepted = nullptr;  /* [n_seq] drafted tokens accepted, 0..n_spec */
        const int32_t*  token = nullptr;       /* [n_seq] token sampled at the accepted position */
        const uint8_t*  eos = nullptr;         /* [n_seq] 1 when the sequence ended */
    };

    /* `geo` is the already-resolved chunk geometry. The engine resolves it once and passes it,
     * because PrefixCache::configure needs the checkpoint interval out of it and runs first;
     * pass nullptr to resolve one here instead. */
    int  init(Program& prog, const Config& cfg, IKVManager* kv, IPrefixCache* pc,
              const std::vector<BatchBuilder::RankIO>& ranks,
              const ChunkGeometry* geo = nullptr);

    /* The batch the last build produced for one rank; see BatchBuilder::batch_for. */
    const RadBatch* batch_for(int rank) const { return builder_.batch_for(rank); }
    /* The advance's view of that build on one rank; see BatchBuilder::advance_args. */
    AdvanceArgs advance_args(int rank) const { return builder_.advance_args(rank); }
    void fini();

    /* ---------------------------------------------------------- off the step path */
    int      add(std::unique_ptr<Request> r);
    /* `n` requests under one hold of the queue lock, each accepted or refused as add() would, with
     * its result in rc[i]. For the sequences of one API request, which have to reach the same
     * step. */
    void     add_all(std::unique_ptr<Request>* rs, size_t n, int* rc);
    /* Cancellation propagates to the scheduler and frees blocks IMMEDIATELY rather than at
     * completion (spec §14). A cancelled request holding its blocks until the step that would
     * have finished it is a pool that shrinks under exactly the load that cancels. */
    int      cancel(uint64_t id);
    Request* find(uint64_t id);

    /* Requests that reached a terminal state since the last call, oldest first, SWAPPED OUT under
     * the lock into `out` (whose old contents are discarded). The engine drains this to drop its
     * per-request sampler state; it is the only consumer. The server does not read it -- it
     * learns of a terminal request through the sink and calls reap(), which is what finally
     * releases the Request object.
     *
     * A swap rather than a reference: an HTTP thread's cancel() appends to this list under the
     * lock, and a caller walking a reference to it without the lock walks a vector another thread
     * may be reallocating. Handing the same vector back and forth also keeps both capacities, so
     * a steady state allocates nothing. */
    void     take_completed(std::vector<uint64_t>* out);
    int      reap(uint64_t id);

    /* ---------------------------------------------------------- the lock, for the engine
     *
     * EVERY MUTATION OF THE KV MANAGER AND THE PREFIX CACHE HAPPENS UNDER THIS LOCK. Neither has
     * a lock of its own, and both are reached from two kinds of thread: the step path, and HTTP
     * threads running cancel(), reap() and reset_prefix_cache(), which publish into the cache,
     * evict from it and free sequences. The engine's own passes between steps -- the idle tiers,
     * the elastic rebalance, the checkpoint copies -- mutate the same structures, so they run
     * inside this too. Taken once per pass that has work, never per sequence.
     *
     * NOTHING RUN UNDER IT MAY CALL BACK INTO THE SCHEDULER: the mutex is not recursive, and
     * every public method here takes it. */
    template <class Fn> auto locked(Fn&& fn) {
        std::lock_guard<std::mutex> lk(mu_);
        return fn();
    }
    /* The same lock, held for the rest of the caller's scope. */
    [[nodiscard]] std::unique_lock<std::mutex> hold() { return std::unique_lock<std::mutex>(mu_); }

    /* ---------------------------------------------------------- work the device still owes
     *
     * A request cancelled while a drafter pass was being built keeps its blocks until the passes
     * that name them are done (see SchedReq::pending_free). commit() releases them once nothing
     * refers to them; an engine that has gone idle runs no commit, so it drains its streams and
     * calls release_deferred() instead. has_deferred() is the cheap test for whether that is
     * needed at all. */
    bool     has_deferred() const;
    void     release_deferred();

    /* THE ENGINE HAS STOPPED AND WILL RUN NO MORE STEPS. Every request still queued or running is
     * failed with `reason` so its connection is answered rather than left waiting on a step that
     * will never come, and any request submitted afterwards is refused the same way. `status` is
     * what stopped it, and what each of those requests reports. */
    void     shutdown(const char* reason, int status = RAD_E_STATE);

    /* The waiting queue in SERVICE order, by request id -- what the live view shows and what a
     * test asserts a preempted request re-entered at the front of. Returns how many were written. */
    int      queue_snapshot(uint64_t* out, int max) const;

    /* ---------------------------------------------------------- who is in flight
     *
     * WHAT EVERY REQUEST IS DOING RIGHT NOW, for the dashboard and for anyone debugging a stall.
     * The aggregate metrics answer "is the engine busy"; they cannot answer "which of these eight
     * is stuck", and a request that has been in `Waiting` for forty seconds because its prompt
     * does not fit is invisible in every number above.
     *
     * A snapshot under the step lock, copied out: holding a pointer into the scheduler's storage
     * across a step is holding a pointer to a request that may be reaped. */
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
        /* THIS REQUEST'S SHARE OF THE PAGED POOLS, summed over every paged group: the blocks it
         * holds and the tokens they cover. `ctx_tokens` is what it is USING; this is what it has
         * TAKEN, and they differ by a partial block and by whatever a rejected draft left. */
        int64_t  kv_blocks = 0;
        int64_t  kv_tokens = 0;
        int32_t  n_draft = 0;          /* the proposal standing for the next step */
        int32_t  slot = -1;
        double   age_s = 0;
        double   ttft_s = -1;          /* negative until the first token */
        double   decode_tps = 0;       /* this request's own, since its first token */
    };
    /* Newest first is wrong here: a dashboard wants the oldest at the top, because the oldest is
     * the one that is stuck. Returns how many were written. */
    int requests(ReqStat* out, int max) const;

    /* The drafter's proposal for a request's next step. n is clamped to the controller's depth. */
    int      set_draft(uint64_t id, const int32_t* tokens, int n);

    /* ---------------------------------------------------------- media the encoder owes
     * A REQUEST CARRYING MEDIA IS ADMITTED ONLY ONCE EVERY ITEM IS ENCODED. Until then it keeps
     * its place in the queue, the requests behind it are admitted around it, and its items wait
     * here for the engine, which runs the encoder between steps and marks them done (take_encodes).
     * Encoding before admission rather than chunk by chunk keeps the encoder off the step path:
     * a step never waits for a tower, and the batch builder only ever copies rows that exist. */
    bool     encodes_pending() const;
    void     take_encodes(std::vector<std::shared_ptr<mm::Item>>* out);
    /* An encoder pass over whole segments; see BatchBuilder::build_encoder. */
    const RadBatch* build_encoder(const std::vector<BatchBuilder::EncoderPart>& parts, int step) {
        std::lock_guard<std::mutex> lk(mu_);
        return builder_.build_encoder(parts, step);
    }
    int64_t  encoder_patches() const { return builder_.encoder_patches(); }

    /* Drop every prefix-cache entry, returning how many were dropped, or RAD_E_UNSUPPORTED when
     * this engine has no prefix cache. Under the step lock: releasing the cache's references is
     * safe against a running step -- a block a live sequence holds stays allocated because the
     * sequence holds its own reference -- but the map itself is not, and the step path inserts
     * into it. See IScheduler::reset_prefix_cache for why the endpoint exists. */
    int64_t  reset_prefix_cache();

    /* ---------------------------------------------------------- the step path */
    /* One decision, one batch. Null when there is nothing to run, which is not an error. */
    const RadBatch* step();

    /* ---------------------------------------------------- THE NEXT STEP, BEFORE THIS ONE COMMITS
     *
     * A decode step's successor does not need anything the host learns from the step: every row
     * continues where the card's acceptance leaves it (sched/advance.h), with the drafts the
     * drafter proposed on the card, and the host can bound how far that is. So the engine issues
     * the next step's forward pass while this one is still running, and the card goes straight
     * from one to the other.
     *
     *   expect_draft   after the drafter is issued: how many drafts it proposes for a sequence,
     *               which is how many the next step verifies.
     *   step_next      plans and builds that step: every row of this one, as a device row, at
     *               the furthest context it can reach. Null -- and the engine plans with step()
     *               after this one commits -- whenever the step needs a decision only a
     *               committed state can make: a request waiting for admission, a prefill chunk
     *               to continue, a grammar, the context's end, or a pool that cannot grow.
     *   promote_next   once this step has committed: the next step becomes the open one.
     *
     * While the next step is open, commit() leaves each continuing request's blocks where they
     * are -- the next step is writing past the committed end -- and a request that finishes keeps
     * them until the next step has run. */
    void            expect_draft(uint64_t id, int n);
    const RadBatch* step_next();
    const RadBatch* step_next_locked();
    void            promote_next();
    bool            next_open() const { return next_open_; }
    /* Fold the step's outcome back in: append tokens, roll back rejected speculation, retire
     * finished requests, feed the draft controller. */
    int             commit(const StepResult& r);
    /* THE ENGINE'S LOOP IS ROUND, AND THE STEP ENDS HERE -- after the drafter, not at commit.
     * A serial draft head runs as extra steps AFTER the trunk step that fed it (see below), so
     * closing the step at commit would leave those rounds in no step at all: their time is charged
     * to nothing, `decode_tps` reads high and the reported step time reads short. The definition in
     * scheduler.cpp is the argument. Call it once per loop iteration that ran a step; a caller that
     * does not is left with correct counters and no timing. */
    void            step_finished();

    /* ---------------------------------------------------- the SERIAL drafter (spec §10)
     *
     * A serial draft head (RAD_DRAFT_SERIAL) runs as EXTRA STEPS of its own, after the
     * trunk step that fed it. It cannot run inside that step: round 1 consumes the token the
     * sampler chose and the hidden state of the position that token came from, and which position
     * that is depends on how many drafts the verify accepted -- neither is known until the trunk
     * step has been sampled and committed.
     *
     * So the engine drives it, and these three calls are the scheduler's half:
     *
     *   serial_begin   BEFORE commit(), right after the step is issued, while plan() describes it.
     *               Says which sequences may draft, reserves the lookahead KV they may write
     *               into -- for the furthest position acceptance could reach, since how far it
     *               reached is the card's to decide -- and clamps `rounds` to the controller's
     *               current depth.
     *   serial_batch   one batch per pass: 0 the history, which also carries round 1 for the
     *               rows that draft (RadBatch::draft_out_ids), then the rounds from 2. Every row
     *               that continues a sampled verify is a DEVICE row (advance.h): the card fills
     *               its positions from the acceptance it computed, and its tokens from the sampled
     *               ids and each round's proposal, so no pass waits on the host.
     *   serial_end     AFTER commit(), with the rounds that ran: moves the history watermark to
     *               the last position the committed acceptance says the history pass wrote.
     *   set_draft   the engine publishes each row's tokens through the ordinary door, so a head's
     *               draft and a prompt-lookup draft are the same thing to the next step. A row
     *               the head declined leaves n_draft at 0 and the n-gram drafter fills in.
     */
    struct SerialRow {
        uint64_t seq = 0;
        int32_t  slot = -1;
        int32_t  card = -1;
    };
    int serial_begin(int* rounds, SerialRow* out, int max);
    const RadBatch* serial_batch(int round, int step);
    /* Round 0 is the HISTORY pass -- null when no row needs one -- and round 1 is always null,
     * because round 0 carries it. Call serial_end() once the step has committed so the watermark
     * moves. */
    void serial_end(int rounds_done);

    /* ----------------------------------------------------- the BLOCK drafter (spec §10)
     *
     * A block-diffusion drafter (RAD_DRAFT_BLOCK), and the shape is the whole point: ONE pass
     * over a block of `block` positions proposes all of them, so depth costs nothing and there is
     * no chain of rounds. What replaces it is TWO passes with different shapes, and the scheduler's
     * half is telling each of them which rows it runs over.
     *
     *   block_begin       right after commit(), while plan() still describes the step just run.
     *                  Reserves the block of draft KV the query pass will write, and answers
     *                  which sequences may draft.
     *   block_context     the CONTEXT pass, draft_pass -1. Its entries MIRROR that trunk step's --
     *                  same sequence, same ctx_len, same n_tokens -- because the drafter reads
     *                  the trunk's tapped hidden states positionally out of a buffer of its own,
     *                  and a mirror is what makes row i of the batch row i of it. No gather,
     *                  no offset, and no special case for a rejected suffix: those positions are
     *                  inside the block the query pass is about to overwrite.
     *   block_query       the QUERY pass, draft_pass +1. One entry a drafting sequence, `block`
     *                  tokens: the anchor at the sequence's own next position and a mask at each
     *                  of the `block - 1` after it. `out_ids` names THE LAST `depth` ROWS, which
     *                  is the one rule both trained conventions reduce to: at block = depth + 1
     *                  the anchor row predicts nothing and each mask row predicts the token at
     *                  ITS OWN position, and at block = depth every row predicts the token AFTER
     *                  it and the anchor is not wasted. rad_builder.h argues the pair.
     *   block_end         moves the filled range over what actually ran.
     *
     * The proposal does not come back through the sampler: the drafter writes it to the buffer it
     * declared as its proposal, and the engine publishes it through set_draft() like any other.
     */
    struct BlockRow {
        uint64_t seq = 0;
        int32_t  slot = -1;
        int32_t  card = -1;
    };
    /* `block` is the drafter's trained block in ROWS and `depth` how many tokens it proposes out
     * of them -- the head takes the last `depth` -- and `window` is its attention window (0 for
     * none); `mask` is the token id its masked query slots embed. Returns how many sequences may
     * draft, and fills `out` with those -- in the order block_query() will emit them. */
    int block_begin(int block, int depth, int window, int32_t mask, BlockRow* out, int max);
    const RadBatch* block_context(int step);
    const RadBatch* block_query(int step);
    void block_end();

    const StepPlan&      plan() const { return plan_; }
    const StepPlan&      plan_next() const { return plan_next_; }
    const ChunkGeometry& geometry() const { return geo_; }
    DraftController&     draft() { return draft_; }
    SchedMetrics         metrics() const;

    /* THE OBSERVERS' COPY. A dashboard, a scraper or an admin call reads what the engine thread
     * last published and never the scheduler's own state, because that state is behind the lock
     * every step takes: a reader holding it for as long as its own work takes delays the step by
     * exactly that long, and an observer must not be able to slow what it observes.
     *
     * publish() is the engine thread's, once a step and at most every 20 ms while idle. It builds
     * the copy under the scheduler's lock -- the engine thread's own lock, so nothing waits -- and
     * swaps it in only if no reader is copying out at that moment; a swap it skips is made at the
     * next call. The readers take only the copy's lock, which the engine thread never waits on. */
    void                 publish(bool idle);
    SchedMetrics         observed_metrics() const;
    int                  observed_requests(ReqStat* out, int max) const;
    size_t               num_waiting() const { std::lock_guard<std::mutex> lk(mu_); return waiting_.size(); }
    size_t               num_running() const { std::lock_guard<std::mutex> lk(mu_); return running_.size(); }

    /* Visible for tests: the chunk the scheduler would give a request at `n_computed` under a
     * token budget, and the checkpoint position it will land on (-1 for none). */
    int64_t plan_chunk(int64_t n_computed, int64_t n_target, int64_t budget,
                       int64_t* checkpoint_pos) const;

private:
    Program*      prog_ = nullptr;
    Config        cfg_{};
    ChunkGeometry geo_{};
    KVGeom        kvgeom_{};
    /* Does this model carry recurrent state a prefix hit would have to restore? See admit_one. */
    bool          has_linear_state_ = false;
    IKVManager*   kv_ = nullptr;
    IPrefixCache* pc_ = nullptr;

    /* DOES A PREFIX-CACHE HIT LEAVE A HOLE IN A BLOCK DRAFTER'S OWN KV? Only if some PAGED group
     * is not prefix-cached. prefix.cpp skips a WINDOWED group -- "a hit's blocks do not describe
     * a cache whose floor moves" -- so a windowed drafter really does start at the hit position
     * and really must wait for its window to slide past it. A FULL-attention drafter's group is
     * cached exactly like the trunk's: the hit hands over blocks the PREVIOUS sequence's own
     * context pass wrote, so the content is there and the floor is 0.
     *
     * Read off IPrefixCache::cached_groups() rather than re-deriving prefix.cpp's rule, which
     * also skips on a block-size mismatch. A bound like this lives in exactly one place; a second
     * copy is free to drift out of step with the first. */
    bool          hit_holes_draft_ = true;
    BatchBuilder  builder_;
    DraftController draft_;

    /* Where this sequence's block-drafter cache begins, as the guard in block_begin reads it
     * and block_end records it. One rule, one place. */
    int64_t draft_floor(const SchedReq& s, int64_t ctx_len) const {
        if (s.draft_hi != 0 || s.draft_lo != 0) return s.draft_lo;
        return hit_holes_draft_ ? ctx_len : 0;
    }

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
    std::atomic<bool>    pub_dirty_{true};
    mutable std::atomic<bool> observer_demand_{false};

    std::vector<std::unique_ptr<SchedReq>> reqs_;   /* by slot */
    std::vector<int32_t> free_slots_;
    std::vector<int32_t> free_cards_;   /* card indices not held by any sequence; see SchedReq::card */
    std::unordered_map<uint64_t, int32_t> by_id_;

    /* Both kept sorted by (priority desc, arrival asc). running_ is sorted so the preemption
     * victim is the back; waiting_ so admission takes the front. */
    std::vector<int32_t> running_;
    std::vector<int32_t> waiting_;
    std::vector<uint64_t> completed_;

    /* Requests whose release is waiting on something that still refers to them -- a deferred
     * reap or a deferred free. A count, so the sweeps that look for them cost nothing on the
     * steps where there are none, which is nearly all of them. */
    int      n_deferred_ = 0;
    /* Items a waiting request needs encoded, in the order its requests reached the queue's head;
     * each at most once (Item::queued). mu_ held. */
    std::vector<std::shared_ptr<mm::Item>> enc_want_;
    /* Are all of this request's items encoded; the ones that are not are queued for the encoder. */
    bool media_ready(SchedReq& s);

    /* Set by shutdown(); add() refuses from then on. */
    bool     closed_ = false;
    const char* closed_reason_ = "";
    int      closed_status_ = RAD_E_STATE;

    /* THE LONGEST SEQUENCE THIS DEPLOYMENT CAN ADDRESS, prompt and output together. The batch
     * builder sizes its block tables for it and the rotary tables are this long, so no position
     * at or past it may be scheduled -- whatever max_tokens a request carries, including none. */
    int64_t  ctx_limit_ = 0;

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

    /* The serial drafter's rows, valid between serial_begin() and serial_end(). `base_pos`
     * is the position round 1 writes at: the emitted token's own position, which is exactly the
     * committed length because that token has not been fed to the model yet. */
    struct SerialSeq {
        uint64_t       seq = 0;
        int32_t        slot = -1;
        int32_t        card = -1;
        int32_t        hist_row = -1;   /* the trunk row the history pass starts at */
        int32_t        hist_n = 0;      /* how many positions it has to fill, 0 for none */
        int64_t        hist_pos = 0;    /* the head index that first row is */
        /* THE VERIFY STEP THIS ROW CONTINUES FROM, as the card's advance reads it (advance.h):
         * whether it sampled, the token row of its first verified position, its first sampled
         * row, and how many history positions come before the verified ones. */
        bool           dev = false;
        int32_t        dev_tok = 0;
        int32_t        dev_out = -1;
        int32_t        lead = 0;
        /* ITS INDEX AMONG THE DRAFTING ROWS, or -1 for a row that only owes history. A sequence
         * in the middle of a long prompt has hidden states worth turning into the head`s K and V
         * and no token to draft from yet, and so does every sequence when a prefill is queued
         * behind this step -- so the two sets are not the same set. */
        int32_t        draft_idx = -1;
        /* NO Request* HERE, for the reason BlockSeq gives: serial_ is taken under the lock in
         * serial_begin and read again by serial_batch, once a round, after the lock has been
         * released and retaken -- and a client can disconnect in any of those windows. */
        /* THE FURTHEST POSITION ROUND 1 CAN WRITE AT: the verify step's last position, reached
         * when every draft is accepted. The card knows the real one; this bounds it. */
        int64_t        base_pos = 0;
    };
    std::vector<SerialSeq> serial_;
    StepPlan            serial_plan_;

    /* The block drafter's rows, valid between block_begin() and block_end(). One entry for EVERY
     * entry of the trunk step just run, in its order; `can_draft` says whether it also gets a
     * query pass. `ctx_len`/`n_tokens` mirror that step's entry, which is the whole contract with
     * the drafter's own source buffer. */
    struct BlockSeq {
        uint64_t       seq = 0;
        int32_t        slot = -1;
        int32_t        card = -1;
        int32_t        ctx_len = 0;
        int32_t        n_tokens = 0;
        int64_t        base_pos = 0;      /* the anchor's position: the sequence's next one */
        bool           can_draft = false;
        /* A ROW KEPT ONLY TO HOLD ITS PLACE. The trunk step finished this sequence -- it ended,
         * failed or was cancelled at commit -- so its KV is gone and it drafts nothing. It is
         * still a row of the context pass, because the drafter reads the trunk's hidden states
         * positionally: dropping it would slide every later sequence onto its neighbour's rows.
         * With no blocks behind it the pass stores nothing for it (a negative slot is padding). */
        bool           mirror = false;
        /* NO Request* HERE, DELIBERATELY. block_ is taken under the lock in block_begin and read
         * again in block_context, block_query and block_end, and the lock is RELEASED between
         * them -- so a client that disconnects in that window can have its Request destroyed
         * under the second pass, leaving BatchBuilder::build reading tokens off freed memory.
         * The snapshot is a WEAK reference: `slot` plus `seq`, resolved through block_req(). */
    };
    std::vector<BlockSeq> block_;
    StepPlan           block_plan_;
    int32_t            block_len_ = 0;
    int32_t            block_depth_ = 0;   /* tokens proposed; the head takes the LAST this many */
    /* `block` - 1 copies of the drafter's mask token, which is what a query entry's `draft` array
     * points at: token_at() walks prompt ++ output ++ draft, so the mask positions fall straight
     * out of the accessor every other pass already uses. */
    std::vector<int32_t> block_mask_;
    int      step_ = 0;
    uint64_t arrival_ = 0;

    SchedMetrics m_{};
    std::chrono::steady_clock::time_point step_start_{};
    /* When the last step ended. An idle engine runs no steps, so the throughput window has
     * nothing adding time to it, and metrics() decays what it reports by this. */
    std::chrono::steady_clock::time_point last_step_at_ = std::chrono::steady_clock::now();
    int64_t step_prefill_tok_ = 0, step_decode_tok_ = 0;
    /* The throughput window (step_finished): each kind's tokens and the wall time the steps
     * covered, both decayed with the same time constant, so their ratio is tokens over time. */
    double  tps_win_prefill_ = 0.0, tps_win_decode_ = 0.0, tps_win_time_ = 0.0;
    bool    step_open_ = false;

    /* arrival; mu_ held */
    int  add_locked(std::unique_ptr<Request> r);

    /* ordering */
    bool ahead_of(int32_t a, int32_t b) const;
    void queue_insert(std::vector<int32_t>& q, int32_t slot);
    static void queue_erase(std::vector<int32_t>& q, int32_t slot);

    /* lifecycle */
    /* `publish` is false only when the engine is shutting down: a sequence the device may have
     * left half-computed is not worth indexing, and nothing will ever hit it.
     *
     * A FAILED REQUEST SAYS WHAT FAILED: its sink is failed with `status` and `why`, which the
     * server puts in the error the client gets. RAD_OK leaves a sink that has failed already (an
     * overflow) as it is. */
    void close_request(SchedReq& s, ReqState st, const char* reason, bool defer_free = false,
                       bool publish = true, int status = RAD_OK, const char* why = nullptr);
    /* What reap() does once nothing points at the slot any more: give the blocks back, forget the
     * id, destroy the Request and return the slot. Called by reap() directly, or by the sweep in
     * commit() for a request whose reap arrived while something still pointed at it. */
    void release_locked(SchedReq& s);
    /* The sequence's KV back to the manager and its card index back to the pool, together:
     * both are what a step in flight still addresses, so both wait for the same moment. */
    void free_kv(SchedReq& s);
    void drop_card(SchedReq& s);
    /* Is this request still referenced by a drafter's snapshot? block_ and serial_ outlive the
     * lock. */
    bool block_holds(uint64_t id) const;
    /* Release every deferred reap and every deferred free whose request nothing refers to any
     * more. At most one step late: the reference that blocked it is either the step just
     * committed or a drafter snapshot that block_end() / serial_end() has since cleared. */
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
    /* RAD_OK, a negative status when the request cannot be admitted now -- nothing behind it is
     * either -- or kAdmitWait when its context is still being read off disk and the next request
     * may go ahead of it. */
    int  admit_one(SchedReq& s, int64_t budget);
    static constexpr int kAdmitWait = 1;
    /* The queue positions one step's admission took, kept to reuse the allocation. */
    std::vector<size_t> took_;

    /* The conv rolling window, width-1 + num_spec + 1 entries (spec §10). */
    int32_t conv_window() const;
};

}  /* namespace rad */
