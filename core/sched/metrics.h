/* metrics.h -- the scheduler's snapshot for the server's /metrics endpoint (spec §14).
 *
 * Plain numbers only; each exporter (Prometheus, the live view) does its own formatting. Fixed
 * arrays, so a scrape allocates nothing and never contends with the step path for the allocator.
 */
#pragma once
#include "rad_core.h"

namespace rad {

enum { RAD_SCHED_MAX_KV_GROUPS = 16 };

struct SchedMetrics {
    int64_t step = 0;

    /* Queue state, instantaneous. */
    int64_t queue_depth = 0;         /* requests waiting, including preempted ones */
    int64_t running = 0;
    int64_t preempted_waiting = 0;   /* of queue_depth, how many are awaiting recompute */

    /* Cumulative counters. Monotonic, so a scraper can rate() them. */
    int64_t admitted = 0;
    int64_t finished = 0;
    int64_t failed = 0;
    int64_t cancelled = 0;
    int64_t preemptions = 0;
    int64_t steps = 0;

    /* Steps by shape. `steps_mixed` counts steps carrying both a prefill chunk and decode rows.
     * `decode_rows_unspeculated` counts every decode row that took no draft, for any reason
     * (depth 0, a request that may not speculate, a drafter with nothing to propose). */
    int64_t steps_prefill = 0;
    int64_t steps_mixed = 0;
    int64_t steps_decode = 0;
    /* --prefill-interleave: decode-only steps the hold inserted, issued ahead or not; each kept a
     * ready chunk out. */
    int64_t steps_held = 0;
    int64_t decode_rows_unspeculated = 0;
    /* Steps whose forward pass was enqueued behind the previous step before it committed
     * (Scheduler::step_next). Every other step leaves the card idle during the host's commit and
     * staging. */
    int64_t steps_ahead = 0;
    /* Prefill chunks capped to leave room for decode rows or a waiting admission in the same
     * step (--busy-prefill-chunk). Zero under `off`. */
    int64_t prefill_chunks_capped = 0;

    /* Cumulative nanoseconds spent inside steps, total and for decode steps only. Wall time over
     * a step count is not a step time: it charges idle gaps between requests to steps and mixes
     * prefill with decode. `step_ns_decode / steps_decode` is the decode step time. Measured from
     * `step_start_`, the clock `decode_tps` uses, so no extra synchronisation (spec §16). */
    int64_t step_ns = 0;
    int64_t step_ns_decode = 0;

    /* The single-sequence decode floor and the most recent window, nanoseconds: both window means
     * of single-sequence decode steps (DecodeFloor), 0 until a gated window completes. A decode
     * step at more sequences is a different measurement and does not count. */
    int64_t step_ns_decode_best = 0;
    int64_t step_ns_decode_recent = 0;

    /* Per KV group. `name` points into the Program, which outlives any scrape. */
    int n_kv_groups = 0;
    struct Group {
        const char* name = "";
        int64_t blocks_total = 0;
        int64_t blocks_free = 0;
        float   utilisation = 0.0f;   /* 1 - free/total */
        /* Tokens one block covers, 0 for a stateful group. */
        int64_t block_tokens = 0;
        /* A windowed group recycles blocks below the window, so its size bounds how far back a
         * sequence sees, not how much context fits. */
        bool    window = false;
        /* The group in bytes, the only unit that sums across groups: a stateful group has no
         * token axis, and paged groups share one, so summed tokens overcount. */
        int64_t bytes_total = 0;
        int64_t bytes_used = 0;
        /* Bytes the group was carved to address. bytes_total is what the card holds now; the
         * difference is the VRAM the elastic pool released behind unheld blocks. */
        int64_t bytes_carved = 0;
    } kv[RAD_SCHED_MAX_KV_GROUPS];

    /* Prefix caching (§7.3). Attention and linear are reported separately because they hit at
     * different granularities. */
    int64_t prompt_tokens = 0;
    int64_t attn_cached_tokens = 0;
    int64_t linear_cached_tokens = 0;
    float   attn_hit_rate = 0.0f;
    /* Cumulative, from the prefix cache. Rising evictions under a steady workload mean the pool
     * cannot hold the working set. */
    int64_t prefix_evictions = 0;
    int64_t prefix_ckpt_evictions = 0;
    /* prefix_ckpt_evictions by reason (retention policy, least recently used of a full store),
     * and reservations that went without a snapshot. */
    int64_t prefix_ckpt_retired = 0, prefix_ckpt_lru = 0, prefix_ckpt_refused = 0;
    int64_t prefix_expired = 0, prefix_ckpt_expired = 0;   /* --prefix-cache-ttl */
    /* Admissions whose attention hit was cut to nothing: no snapshot at or below it (recurrent
     * state or a window snapshot), a windowed trunk group no window snapshot covers, or blocks
     * that could not be adopted. `hit_cut_tokens` is the checkpoint shortfall: prompt tokens the
     * attention cache held and the admission computed again, because the hit resumed at a
     * snapshot below it or was cut to nothing. */
    int64_t hit_cut_no_snapshot = 0, hit_cut_window = 0, hit_cut_adopt = 0;
    int64_t hit_cut_tokens = 0;
    /* Tokens of context held only by the cache, i.e. what eviction would free. Excludes blocks
     * shared with a running sequence. Same token axis as the paged groups, so comparable with
     * kv_tokens_used. */
    int64_t prefix_held_tokens = 0;
    float   linear_hit_rate = 0.0f;
    int64_t checkpoints_written = 0;

    /* Throughput: tokens over wall time in a 1-second decaying window (Scheduler::step_finished);
     * the counters beside them are exact. */
    int64_t prefill_tokens = 0;
    int64_t decode_tokens = 0;
    float   prefill_tps = 0.0f;
    float   decode_tps = 0.0f;

    /* Speculation (§10). */
    int64_t draft_tokens = 0;
    int64_t draft_accepted = 0;
    /* Prefix hits ended early so a windowed block drafter has context (Scheduler::draft_refill_),
     * and the reusable tokens they gave back to be recomputed. */
    int64_t draft_refills = 0;
    int64_t draft_refill_tokens = 0;
    float   draft_acceptance = 0.0f;
    int     draft_depth = 0;          /* what the controller would choose right now */
    /* Per draft position: `reached` counts verifies where every earlier position was accepted,
     * `accepted` how often this one was then taken. Their ratio is the conditional acceptance,
     * which says what a deeper draft would buy. `draft_depth_seen` bounds the valid entries. */
    enum { RAD_METRICS_MAX_SPEC = 64 };
    int     draft_depth_seen = 0;
    int64_t draft_pos_reached [RAD_METRICS_MAX_SPEC] = {};
    int64_t draft_pos_accepted[RAD_METRICS_MAX_SPEC] = {};
    /* Adaptive window (--draft-window adaptive, draft.h); all zero while off. The gauge is the
     * mean window over running sequences; the cumulative histogram counts verifies by window,
     * index = drafts verified. */
    bool    draft_adaptive = false;
    float   draft_depth_adaptive = 0.0f;
    int64_t draft_verify_depth[RAD_METRICS_MAX_SPEC + 1] = {};
    /* The cost policy's delivery ledger, cumulative. `RAD_OK` from the predictor is not evidence
     * that a cycle taught anything: every row can be skipped by a recycled (req_id, slot_gen) and
     * the call still succeeds -- and a cycle that never completed is never delivered at all. So
     * the three row outcomes are counted apart, plus the cycles that were dropped instead of
     * delivered (incomplete, evicted, or refused by the predictor). All zero while Cost is off. */
    int64_t draft_obs_attempted = 0;   /* rows of deliverable cycles offered to the predictor */
    int64_t draft_obs_applied = 0;     /* rows whose labels moved a request's estimate */
    int64_t draft_obs_skipped = 0;     /* rows that taught nothing */
    int64_t draft_obs_discarded = 0;   /* cycles not delivered: incomplete or refused */
    /* Valid observations handed to an inactive cost table: the deployment is observing, not
     * learning. Zero while a profile is installed, so "nothing was absorbed" and "nothing was
     * offered" cannot be read as the same fact. */
    int64_t draft_obs_inactive = 0;
};


/* Estimator for the single-sequence decode floor (SchedMetrics::step_ns_decode_best): a min over
 * window means, not over raw step brackets. A decode step issued ahead behind a step with long
 * host work (a prefill chunk, a KV checkpoint save) has its forward pass finished before its own
 * bracket opens, so the bracket holds only readback and commit, about a tenth of a real step; a
 * raw min would latch that forever. Such brackets are rejected before they enter a window, and the
 * window mean dilutes what remains. */
struct DecodeFloor {
    /* Steps averaged into one floor candidate: about what the dashboard's 500 ms poll spans at a
     * single-sequence step, so the engine's floor and the page's are over comparable windows. */
    static constexpr int kWindow = 32;
    static constexpr int64_t kHorizon = 8 * kWindow;

    void update(int64_t ns) {
        /* A degenerate bracket is ~0.09 of a real step and a real one ~0.96 of the mean, so half
         * the reference separates them. The reference is a running mean over every step, the
         * rejected ones included, with a horizon of kHorizon steps: degenerates are rare enough to
         * barely move it, and when the real step gets faster (a slow warm-up, a shorter context)
         * it follows instead of rejecting every later step. No rejection until a window's worth
         * of samples has been seen. */
        ++seen_;
        mean_ += (ns - mean_) / (double)(seen_ < kHorizon ? seen_ : kHorizon);
        if (seen_ > kWindow && ns * 2 < mean_) return;
        sum_ += ns;
        if (++count_ < kWindow) return;
        const int64_t mean = sum_ / count_;
        /* The first window ran before the gate could judge and may carry degenerates, so it is
         * recorded by neither side. */
        if (warmed_) {
            if (best_ == 0 || mean < best_) best_ = mean;
            recent_ = mean;
        }
        warmed_ = true;
        sum_ = 0;
        count_ = 0;
    }
    /* The fastest window mean seen. */
    int64_t best() const { return best_; }
    /* The most recent completed window mean, from the same stream and window size as best();
     * 0 until the first gated window completes. */
    int64_t recent() const { return recent_; }

private:
    int64_t sum_      = 0;
    int     count_    = 0;      /* samples in the window being accumulated */
    int64_t best_     = 0;
    int64_t recent_   = 0;
    bool    warmed_   = false;  /* one window has completed */
    int64_t seen_     = 0;      /* samples seen, rejected ones included */
    double  mean_     = 0.0;    /* the rejection reference */
};

}  /* namespace rad */
