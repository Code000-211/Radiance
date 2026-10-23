/* steplog.h -- what every step carried and how long it took, kept for the engine to measure itself.
 *
 * /metrics answers "what is it doing now" with means over a decaying window (sched/metrics.h). A
 * run that measures a build needs the distribution instead, and it needs it exactly: the interval
 * between two bursts of tokens, the share of steps that carried a prefill chunk beside a decoder,
 * how many draft tokens a step verified. A client clock cannot see any of that. It sees a
 * speculating engine emit a draft's worth of tokens and then wait, so its own arrival intervals
 * carry the burst pattern, not the step's, and everything is mixed with the network and the
 * detokeniser.
 *
 * So the scheduler writes one fixed record per step, in the bracket it already takes for
 * `step_ns` (Scheduler::step_finished). Enabled once, before the first step: a ring of records
 * allocated then and never grown, so the step path adds one store and no allocation, no
 * synchronisation beyond the lock the bracket is already taken under, and no clock other than the
 * one the bracket reads. Off, nothing is written and nothing is kept: a serving process pays
 * nothing (Engine::run_diagnose).
 *
 * A ring, because a run of any length outgrows a list: `cap` steps, then the oldest go. The reader
 * drains between phases (Scheduler::drain_step_log) and counts what was overwritten; a phase whose
 * records were overwritten reports its step-time distribution over what survived, and says so
 * rather than pretending to the whole (core/diagnose.h).
 */
#pragma once
#include <chrono>
#include <cstdint>
#include <vector>

namespace rad {

/* One step, as the scheduler saw it. Plain counters, so a reader needs no scheduler state to
 * interpret it, and `end_ns - begin_ns` is the same bracket `SchedMetrics::step_ns` accumulates. */
struct StepRecord {
    int64_t step      = 0;    /* RadBatch::step */
    int64_t begin_ns  = 0;    /* steady clock: the step's bracket opened */
    int64_t end_ns    = 0;    /* and closed, after its draft rounds */
    /* The gap this step waited after the previous one ended: the device idle between bursts,
     * 0 when a pass was already queued (a step issued ahead), and no record before the first. */
    int64_t since_prev_ns = -1;

    int32_t n_seq    = 0;     /* sequences the step carried */
    int32_t n_tok    = 0;     /* rows of its forward pass */
    int32_t n_out    = 0;     /* logits rows it sampled */
    int32_t n_decode = 0;     /* of its entries, the decode rows */
    int32_t n_prefill= 0;     /* and the prefill chunks */
    int32_t tok_out  = 0;     /* tokens its sequences emitted: 1 per decode row, plus accepted drafts */
    int32_t n_accepted = 0;   /* draft tokens accepted, a prefix of n_spec */

    /* The step's shape. RAD_PHASE_PREFILL / _DECODE / _MIXED as plan_.phase states it. `draft_pass`
     * is RadBatch::draft_pass: 0 for a trunk step, r > 0 a draft head's round, r < 0 a history
     * pass, so a reader can keep the drafter's steps out of the trunk's distribution. */
    int32_t phase = 0;
    int32_t draft_pass = 0;
    bool    issued_ahead = false;  /* enqueued behind the previous step (Scheduler::step_next) */
    bool    prefill_held = false;  /* a ready chunk this step left out (--prefill-interleave) */
};

/* The ring. The scheduler owns one; the reader drains it under the scheduler's lock. */
class StepLog {
public:
    /* Keeps records for `cap` steps, dropping the oldest past that. Allocates once, here. */
    void enable(size_t cap);
    /* Stops recording and frees the ring. Safe on the step path's far side (the mode is over). */
    void disable();
    bool enabled() const { return !ring_.empty(); }

    /* The step path: one slot, no allocation, no lock (the caller holds the scheduler's). */
    void push(const StepRecord& r);

    /* Copies out what is held and clears it, with the number dropped since the last drain. The
     * records keep their order, oldest first. */
    uint64_t drain(std::vector<StepRecord>* out);

    /* Records held now, and the capacity: a reader that wants to know whether it is losing
     * records between drains. */
    size_t size() const { return count_; }
    size_t capacity() const { return ring_.size(); }

private:
    std::vector<StepRecord> ring_;
    size_t   head_ = 0;    /* where the next record goes */
    size_t   count_ = 0;   /* slots held, <= ring_.size() */
    uint64_t dropped_ = 0; /* overwritten without being read, since the last drain */
};

/* Steady-clock nanoseconds for a record. */
inline int64_t step_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  /* namespace rad */
