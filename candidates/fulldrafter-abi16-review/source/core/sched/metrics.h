/* metrics.h -- the scheduler's snapshot, for the server's /metrics endpoint (spec §14).
 *
 * A plain struct of numbers with no strings built and nothing formatted. Formatting belongs to
 * whoever is exporting -- Prometheus wants vLLM-compatible names, the live view wants a terminal
 * line, and a scheduler that rendered either would have to be edited to add the other. Fixed
 * arrays, so a scrape allocates nothing and cannot contend with the step path for the allocator.
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

    /* STEPS BY SHAPE, AND WHAT SPECULATION DID NOT REACH. `steps_mixed` over `steps` says how
     * often the load produces a step carrying both a prefill chunk and decode rows; an agentic
     * load produces them often enough to be worth measuring.
     *
     * `decode_rows_unspeculated` counts every decode row that took no draft, whatever the reason
     * -- depth 0, a request that may not speculate, a drafter with nothing to propose. Counting
     * ROWS rather than any one reason keeps it the answer to "how much of my decode is running
     * unspeculated" as the set of reasons changes. Without it the loss is invisible: throughput
     * falls and every explanation fits. */
    int64_t steps_prefill = 0;
    int64_t steps_mixed = 0;
    int64_t steps_decode = 0;
    int64_t decode_rows_unspeculated = 0;
    /* STEPS ISSUED AHEAD: a step whose forward pass went onto the card behind the step before it,
     * before that step committed (Scheduler::step_next). Every other step waits for the host's
     * commit and staging with the card idle, so steps_ahead against steps is how much of the run
     * kept the card fed through the host's part of it. */
    int64_t steps_ahead = 0;

    /* HOW LONG THE ENGINE SPENT INSIDE A STEP, cumulative nanoseconds, and the decode half of it
     * separately. Neither is a derived rate -- they are counters, and that is the point.
     *
     * A STEP TIME IS NOT WALL TIME OVER A STEP COUNT, and a gauge that computes it that way is
     * wrong in two directions at once. An engine that idles between requests -- which is every
     * agent, every chat client, every tool loop -- charges the idle gap to the steps around it,
     * inflating the step time by however long nothing was running. And a prefill chunk costs an
     * order of magnitude more than a decode step, so a load that mixes them averages two
     * populations and calls the result the step time. Both faults are invisible without an
     * instrument: the number is plausible, it moves with load, and nothing contradicts it.
     *
     * With these, `step_ns_decode / steps_decode` is a decode step time that needs no window, no
     * saturation assumption and no second endpoint. Charged from the scheduler's own
     * `step_start_`, which is the same clock `decode_tps` already uses -- so this adds no
     * synchronisation and no timing mode (spec §16). */
    int64_t step_ns = 0;
    int64_t step_ns_decode = 0;

    /* Per KV group. `name` points into the Program, which outlives any scrape. */
    int n_kv_groups = 0;
    struct Group {
        const char* name = "";
        int64_t blocks_total = 0;
        int64_t blocks_free = 0;
        float   utilisation = 0.0f;   /* 1 - free/total */
        /* Tokens one block covers, 0 for a stateful group. A pool measured in blocks cannot be
         * compared with a request measured in tokens, and every reader wants that comparison. */
        int64_t block_tokens = 0;
        /* AND THE SAME GROUP IN BYTES, which is the measure that survives being added to the next
         * group's. Neither blocks nor tokens does: a stateful group has no token axis, and two
         * paged groups share one, so a token of context takes a slot in both and summing them
         * reports a pool the card does not have. */
        int64_t bytes_total = 0;
        int64_t bytes_used = 0;
        /* WHAT IT WAS CARVED TO ADDRESS, as against bytes_total, which is what the card is
         * actually holding for it. The two differ by what the elastic pool has released: the
         * cache releases the VRAM behind blocks nothing holds, and the difference between these
         * two numbers is exactly what it gave back. Equal otherwise. */
        int64_t bytes_carved = 0;
    } kv[RAD_SCHED_MAX_KV_GROUPS];

    /* Prefix caching (§7.3). Attention and linear are reported separately because they hit at
     * different granularities and a single number would hide exactly the effect the decoupling
     * was for. */
    int64_t prompt_tokens = 0;
    int64_t attn_cached_tokens = 0;
    int64_t linear_cached_tokens = 0;
    float   attn_hit_rate = 0.0f;
    /* Cumulative, from the prefix cache itself. Rising evictions under a steady workload
     * mean the pool cannot hold the working set and turn-2 latency is about to collapse. */
    int64_t prefix_evictions = 0;
    int64_t prefix_ckpt_evictions = 0;
    /* WHAT THE CACHE IS HOLDING ON ITS OWN NOW, in tokens of context: the part of the pool that
     * an eviction would actually free, because no live sequence is reading it. The pool figure
     * above counts a held block whoever holds it, and on a server serving conversations most of
     * it is this rather than the running requests -- so without it a working cache and a leak are
     * the same picture. Blocks shared with a running sequence are NOT in it; counting them would
     * make one long request read as a cache that would not release its context. Same token axis
     * as the paged groups, so it is comparable with kv_tokens_used. */
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
    float   draft_acceptance = 0.0f;
    int     draft_depth = 0;          /* what the controller would choose right now */
    /* Per draft POSITION, so a cliff can be told from a decay: `reached` is how often every
     * earlier position was accepted and this one was therefore evaluated, `accepted` how often it
     * was then taken. Their ratio is the conditional hazard, which is the number that says what a
     * deeper draft would actually buy. `draft_depth_seen` bounds how much of the arrays is real. */
    enum { RAD_METRICS_MAX_SPEC = 64 };
    int     draft_depth_seen = 0;
    int64_t draft_pos_reached [RAD_METRICS_MAX_SPEC] = {};
    int64_t draft_pos_accepted[RAD_METRICS_MAX_SPEC] = {};
    /* SPECULATION AS EXECUTED, not as configured: every decode row counted under the window it
     * actually verified. `draft_depth` above is what the policy would choose now, which on a
     * request-censored step, a clamped window or a step issued ahead is not what ran. This array
     * is the only place the executed depth is counted, and it is a count of rows so that a
     * two-sequence step is not read as one.
     *
     * It is also the coverage the acceptance-censored `draft_pos_*` arrays cannot give: a position
     * counts there only when every earlier one was accepted, so a shallow workload and a rejected
     * deep one leave the same trace. A row here names the depth the step verified whatever the
     * sampler then accepted. */
    int64_t draft_exec_rows[RAD_METRICS_MAX_SPEC + 1] = {};
    /* THE ADAPTIVE BANDIT (draft.h), zero when it is not armed. `attempts` counts intervals that
     * ran one arm end to end; `credited` those of them clean enough to score; `tokens` and `ns`
     * are the reward's numerator and denominator, so `tokens / ns` is the rate the policy is
     * optimising and nothing else has to be derived for a reader to check it. */
    int     draft_arm = 0;            /* the arm held, 0 when unarmed */
    int     draft_arms = 0;           /* how many arms exist */
    int64_t bandit_attempts = 0;
    int64_t bandit_credited = 0;
    int64_t bandit_tokens = 0;
    int64_t bandit_ns = 0;
};

}  /* namespace rad */
