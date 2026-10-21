/* core/server/sink.h -- the per-request channel from the scheduler thread to the HTTP thread.
 *
 * `Request::sink` is the void* hook for this (rad_core.h). Exactly one scheduler thread writes it
 * and exactly one HTTP thread reads it, so it is a single-producer single-consumer ring and needs
 * no lock on the data path: a relaxed load of the head, a store into a slot, a release store of
 * the head. The producer never allocates and never blocks, which is the requirement -- a step
 * that can be delayed by a slow HTTP client is a step that has coupled every request in the batch
 * to the worst one.
 *
 * The wake-up is the one place a mutex appears, and only on the consumer's side of it. The
 * producer calls notify_one() without holding the lock, which can lose a wake-up in the window
 * between the consumer testing the predicate and sleeping; the consumer therefore waits with a
 * bounded timeout (RAD_SINK_POLL_MS) rather than indefinitely. The cost is named: a lost wake-up
 * adds at most that many milliseconds to one token's latency. The benefit is that the scheduler
 * thread never takes a lock an HTTP thread might be holding, and the same timeout doubles as the
 * interval at which the streaming path polls whether the client is still there.
 *
 * Tokens, not text: detokenisation is stateful and per-request (spec §12) and belongs on the HTTP
 * thread. Pushing an int32 keeps the producer's write to eight bytes and no allocation.
 */
#pragma once
#include "rad_core.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace rad {
namespace server {

/* How long the consumer sleeps before re-testing. Also the connection-liveness poll interval. */
static constexpr int RAD_SINK_POLL_MS = 25;

enum class Finish : int {
    None = 0,      /* still running */
    Stop,          /* EOS */
    StopString,    /* a stop string matched */
    Length,        /* max_tokens */
    ToolCalls,     /* the parser found a complete tool call and the model stopped */
    Cancelled,     /* client went away, or /cancel */
    Error,         /* the engine failed the request; `status` says why */
};

/* The OpenAI `finish_reason` string. Cancelled has no OpenAI spelling; vLLM reports "abort" and
 * dashboards key off that, so we match vLLM rather than inventing one. */
const char* finish_reason_oai(Finish f);

struct SinkToken {
    int32_t token = 0;
    float   logprob = 0.0f;   /* 0 when the sampler does not supply one; see Sink::has_logprobs */
};

class Sink {
public:
    /* `capacity` is rounded up to a power of two. 8192 tokens is ~64 KiB and is far more than any
     * HTTP consumer can fall behind by without being dead: at 100 tok/s that is 80 seconds of
     * slack. Overflow is not silently dropped -- it fails the request, because a stream missing
     * tokens in the middle is worse than a stream that stops and says so. */
    explicit Sink(size_t capacity = 8192);

    /* Waits for the producer to be finished touching this object before letting it go.
     *
     * Observing `done()` is not enough. The producer publishes the terminal state and only then
     * wakes the consumer, so between those two instructions the consumer can already be inside
     * this destructor -- and destroying a condition_variable while another thread is inside
     * notify_one() on it is a use-after-free in pthread_cond_signal. The producer therefore sets
     * a quiesced flag as its genuinely last act, and this waits for that flag rather than for the
     * state. The window is a handful of instructions wide, so only a data-race detector reports
     * it. */
    ~Sink();

    /* ------------------------------------------------------------------ producer: scheduler */

    /* Returns false only on overflow, which also marks the request failed. */
    bool push(int32_t token, float logprob = 0.0f);
    bool push_span(const int32_t* tokens, size_t n, const float* logprobs = nullptr);

    /* Terminal. Idempotent -- the first call wins, so a cancel racing a natural stop does not
     * produce two finish reasons. */
    void finish(Finish f);
    /* `why` says what failed, in a sentence a client can read; a string that outlives the sink
     * (a literal), or null. */
    void fail(int rad_status, const char* why = nullptr);

    /* The scheduler polls this to learn that the client hung up, and frees the blocks (spec §14).
     * It is also told by IScheduler::cancel(); this is the cheap path that needs no lookup. */
    bool cancel_requested() const { return cancel_.load(std::memory_order_acquire); }

    /* Admission accounting, set once by the scheduler when the prompt is placed. */
    void set_prompt_stats(int32_t n_prompt, int32_t n_cached);

    /* THE FIRST TIME THE SCHEDULER GIVES THIS REQUEST A STEP: where its prompt phase starts, so
     * the time it waited in the queue is not counted as prefill. A preempted request admitted
     * again keeps the first mark. */
    void mark_admitted();
    /* A verify step's speculation for this request: `n` tokens proposed, `accepted` of them
     * taken. Summed over the request, the per-request half of /metrics' draft counters. */
    void add_drafts(int32_t n, int32_t accepted);

    /* ------------------------------------------------------------------ consumer: HTTP */

    /* Copies up to `max` tokens out. Returns how many. Never blocks. */
    size_t read(SinkToken* out, size_t max);

    /* Sleeps until there is something to read, the request is terminal, or `timeout_ms` elapses.
     * Returns true if there is now something to look at. */
    bool wait(int timeout_ms = RAD_SINK_POLL_MS);

    bool   done() const { return finish_.load(std::memory_order_acquire) != (int)Finish::None; }
    Finish finish_reason() const { return (Finish)finish_.load(std::memory_order_acquire); }
    int    status() const { return status_.load(std::memory_order_acquire); }
    const char* why() const { return why_.load(std::memory_order_acquire); }
    bool   overflowed() const { return overflow_.load(std::memory_order_acquire); }

    /* The HTTP thread calls this when the peer disappears. It does NOT free anything itself --
     * the server calls IScheduler::cancel(id) as well; this only stops the producer sooner. */
    void request_cancel();

    /* Timings, in steady-clock nanoseconds since the sink was constructed. -1 until observed. */
    int64_t t_first_token_ns() const { return t_first_.load(std::memory_order_acquire); }
    int64_t t_last_token_ns() const  { return t_last_.load(std::memory_order_acquire); }
    int64_t t_admitted_ns() const    { return t_admit_.load(std::memory_order_acquire); }
    int64_t age_ns() const;
    /* The steady clock at construction, which the times above count from: what puts two sinks'
     * times on one axis. */
    int64_t t_start_ns() const { return t_start_ns_; }

    int64_t n_drafted() const        { return n_draft_.load(std::memory_order_acquire); }
    int64_t n_draft_accepted() const { return n_draft_acc_.load(std::memory_order_acquire); }
    /* One counter a possible verified window, 0..kMaxDepthRows; see n_rows_at_depth. */
    enum { kMaxDepthRows = 64 };
    /* HOW MANY OF THIS REQUEST'S DECODE ROWS VERIFIED EACH WINDOW, by n_spec, so a comparator can
     * pair a request with the depth its acceptance was actually measured at. Acceptance alone
     * cannot do that: a row that verified three tokens and one that verified zero both have an
     * acceptance number, and the reached[] array is conditioned on the earlier positions having
     * been accepted, so neither says what ran. Fixed size (one slot a possible window), written by
     * the single producer. */
    int64_t n_rows_at_depth(int32_t depth) const {
        return (depth >= 0 && depth <= kMaxDepthRows)
                   ? n_depth_[depth].load(std::memory_order_relaxed) : 0;
    }

    int64_t n_produced() const { return (int64_t)head_.load(std::memory_order_acquire); }
    int32_t n_prompt() const { return n_prompt_.load(std::memory_order_acquire); }
    int32_t n_cached() const { return n_cached_.load(std::memory_order_acquire); }

    /* The engine sets this at wiring time if its sampler returns per-token probabilities. Without
     * it the logprobs field of a response is null rather than zeros (spec: nothing unimplemented
     * returns success). */
    void set_has_logprobs(bool v) { has_logprobs_ = v; }
    bool has_logprobs() const { return has_logprobs_; }

    /* The engine's request id, so a dropped connection can reach IScheduler::cancel. */
    uint64_t id = 0;

private:
    void notify();

    std::vector<SinkToken> ring_;
    size_t                 mask_ = 0;

    std::atomic<uint64_t> head_{0};      /* produced count; producer writes, both read */
    std::atomic<uint64_t> tail_{0};      /* consumed count; consumer writes, both read */

    std::atomic<int>      finish_{(int)Finish::None};
    std::atomic<int>      status_{RAD_OK};
    std::atomic<const char*> why_{nullptr};
    std::atomic<bool>     overflow_{false};
    std::atomic<bool>     cancel_{false};
    std::atomic<bool>     quiesced_{false};   /* the producer will not touch this object again */

    std::atomic<int64_t>  t_first_{-1};
    std::atomic<int64_t>  t_last_{-1};
    std::atomic<int64_t>  t_admit_{-1};
    std::atomic<int64_t>  n_draft_{0};
    std::atomic<int64_t>  n_draft_acc_{0};
    /* One counter a possible verified window, 0..kMaxDepthRows; see n_rows_at_depth. */
    std::atomic<int64_t>  n_depth_[(int)kMaxDepthRows + 1]{};
    int64_t               t_start_ns_ = 0;

    std::atomic<int32_t>  n_prompt_{0};
    std::atomic<int32_t>  n_cached_{0};
    bool                  has_logprobs_ = false;

    std::mutex              m_;
    std::condition_variable cv_;
};

/* The scheduler's call sites, so core/sched needs one include and no casts of its own. A null
 * sink is legal and means nobody is listening -- an offline or benchmark request. */
inline bool sink_push(void* s, int32_t token, float logprob = 0.0f) {
    return s ? static_cast<Sink*>(s)->push(token, logprob) : true;
}
inline bool sink_push_span(void* s, const int32_t* tokens, size_t n, const float* logprobs = nullptr) {
    return s ? static_cast<Sink*>(s)->push_span(tokens, n, logprobs) : true;
}
inline void sink_finish(void* s, Finish f) {
    if (s) static_cast<Sink*>(s)->finish(f);
}
inline void sink_fail(void* s, int rad_status, const char* why = nullptr) {
    if (s) static_cast<Sink*>(s)->fail(rad_status, why);
}
inline bool sink_cancelled(const void* s) {
    return s && static_cast<const Sink*>(s)->cancel_requested();
}
inline void sink_prompt_stats(void* s, int32_t n_prompt, int32_t n_cached) {
    if (s) static_cast<Sink*>(s)->set_prompt_stats(n_prompt, n_cached);
}
inline void sink_admitted(void* s) {
    if (s) static_cast<Sink*>(s)->mark_admitted();
}
inline void sink_drafts(void* s, int32_t n, int32_t accepted) {
    if (s) static_cast<Sink*>(s)->add_drafts(n, accepted);
}

}  /* namespace server */
}  /* namespace rad */
