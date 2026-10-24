/* core/server/metrics.h -- Prometheus /metrics, with vLLM's metric names.
 *
 * Names, bucket boundaries and the label set (`model_name`, `engine`) are copied exactly from vLLM,
 * so existing vLLM dashboards and alerts work unchanged against radiance. Where vLLM renamed a
 * metric both spellings are emitted (`vllm:gpu_cache_usage_perc`, which dashboards query, and
 * `vllm:kv_cache_usage_perc`, which current vLLM emits).
 *
 * Anything vLLM lacks goes under `radiance:`; a new `vllm:` name would pass for a metric a
 * dashboard knows.
 *
 * Not emitted: `vllm:iteration_tokens_total` and `vllm:request_queue_time_seconds`. They belong to
 * the scheduler's step loop, which the HTTP boundary cannot observe, and an approximation would
 * mislead. They stay absent until the scheduler reports them.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rad {
namespace server {

struct SchedMetrics;

/* ------------------------------------------------------------------ primitives */

class Counter {
public:
    void inc(double v = 1.0) {
        double cur = v_.load(std::memory_order_relaxed);
        while (!v_.compare_exchange_weak(cur, cur + v, std::memory_order_relaxed)) {}
    }
    /* Raises the count to `total`, a cumulative value kept elsewhere; never lowers it. */
    void advance_to(double total) { inc(std::max(0.0, total - get())); }
    double get() const { return v_.load(std::memory_order_relaxed); }
private:
    std::atomic<double> v_{0.0};
};

class Gauge {
public:
    void set(double v) { v_.store(v, std::memory_order_relaxed); }
    double get() const { return v_.load(std::memory_order_relaxed); }
private:
    std::atomic<double> v_{0.0};
};

/* Cumulative buckets, the Prometheus convention: bucket i counts observations <= bounds[i] and
 * the implicit +Inf bucket is `count`. Atomic, since every HTTP thread observes while /metrics
 * is scraped. */
class Histogram {
public:
    explicit Histogram(std::vector<double> bounds);
    void observe(double v);

    const std::vector<double>& bounds() const { return bounds_; }
    /* Read the buckets in ascending order, then the count, for a consistent picture (observe). */
    uint64_t bucket(size_t i) const { return cnt_[i].load(std::memory_order_acquire); }
    uint64_t count() const { return n_.load(std::memory_order_acquire); }
    double   sum() const { return sum_.load(std::memory_order_relaxed); }

private:
    std::vector<double> bounds_;
    std::unique_ptr<std::atomic<uint64_t>[]> cnt_;
    std::atomic<uint64_t> n_{0};
    std::atomic<double>   sum_{0.0};
};

/* vLLM's bucket layouts, verbatim. Changing one breaks every histogram quantile a dashboard
 * computes across versions, so they are constants, not options. */
std::vector<double> buckets_ttft();          /* vllm:time_to_first_token_seconds */
std::vector<double> buckets_tpot();          /* vllm:time_per_output_token_seconds */
std::vector<double> buckets_latency();       /* every vllm:request_*_seconds */
std::vector<double> buckets_1_2_5(int64_t max_value);   /* token-count histograms */

/* ------------------------------------------------------------------ the registry */

/* The finish reasons `vllm:request_success_total` is broken down by. A fixed set, because label
 * cardinality must never depend on user input. */
enum class SuccessReason { Stop = 0, Length, ToolCalls, Abort, Error, N_ };

class Metrics {
public:
    Metrics(std::string model_name, int64_t max_model_len);

    /* --- observed at the HTTP boundary, per finished request --- */
    void observe_request(int64_t n_prompt, int64_t n_generated,
                         double ttft_s, double tpot_s, double e2e_s, double decode_s,
                         int n_choices, int64_t max_tokens, SuccessReason r);
    void observe_rejected();        /* 429: the admission queue was full */
    void observe_disconnect();      /* the client went away mid-stream */
    void observe_overflow();        /* the sink ring filled: the HTTP side could not keep up */
    void observe_http(int status);

    void set_queue_depth(int64_t d) { queue_depth_.set((double)d); }

    /* --- pulled from the scheduler at scrape time --- */
    void sample(const SchedMetrics& m);

    /* Prometheus text exposition format, version 0.0.4. */
    std::string render() const;


private:
    std::string model_;
    std::string labels_;        /* {model_name="...",engine="0"} */

    Gauge   g_running_, g_waiting_, g_kv_util_, g_prefix_hit_;
    Gauge   g_kv_max_tokens_, g_kv_max_bytes_;
    Gauge   g_prefill_tps_, g_decode_tps_, g_draft_accept_, g_preempted_;
    /* The decode floor, in seconds: the fastest single-sequence window this process has run,
     * and the most recent one. Both stay 0 until a window of 32 single-stream steps completes,
     * and both reset on restart -- the render omits them rather than expose an unmeasured 0. */
    Gauge   g_step_best_, g_step_recent_;
    Gauge   queue_depth_;
    /* Per-position conditional draft acceptance (see render() and core/sched/draft.h). */
    enum { MAX_DRAFT_POS = 64 };
    Gauge   g_draft_pos_[MAX_DRAFT_POS];
    Gauge   g_draft_pos_reached_[MAX_DRAFT_POS];
    Gauge   g_draft_pos_accepted_[MAX_DRAFT_POS];
    std::atomic<int> draft_pos_seen_{0};
    /* Adaptive draft window (--draft-window adaptive): the running sequences' mean window, and
     * verifies by the window they carried (index = drafts). Rendered only while the policy is on. */
    Gauge   g_draft_depth_adapt_;
    Gauge   g_draft_vdepth_[MAX_DRAFT_POS + 1];
    std::atomic<bool> draft_adapt_on_{false};
    std::atomic<int>  draft_vdepth_top_{0};   /* the deepest window with a verify counted */

    /* sample() turns the scheduler's cumulative totals into counter increments (read, add the
     * difference). Concurrent scrapes would both add the same difference, so they are serialised. */
    std::mutex sample_mu_;

    Counter c_prompt_tokens_, c_generation_tokens_, c_preemptions_;
    /* Engine steps. With speculation a step emits 1 + the accepted drafts, so tokens/s does not
     * give step time; two scrapes of this counter and a clock do. */
    Counter c_steps_;
    /* Steps by shape, and decode rows that ran unspeculated. Counters, so two scrapes give the
     * rate over a window rather than an average dominated by startup. */
    Counter c_steps_prefill_, c_steps_mixed_, c_steps_decode_, c_rows_unspec_, c_steps_ahead_;
    Counter c_prefill_capped_, c_steps_held_;
    /* What the mixed steps are made of (docs/PREFILL-OVERLAP.md): who rode with whom, how much of
     * the prefill ran beside decode rows, and how many chunks rode in a step the prefill stager
     * left unarmed. */
    Counter c_steps_multi_pf_, c_rows_dec_mixed_, c_rows_pf_mixed_, c_tok_pf_mixed_,
            c_rows_pf_unstaged_;
    /* Seconds inside a step, all shapes and decode-only. Wall clock over a step count is not a
     * step time (see SchedMetrics::step_ns). */
    Counter c_step_secs_, c_step_secs_decode_;
    /* Exact totals behind the rate gauges; see render(). */
    Counter c_prefill_tokens_, c_decode_tokens_, c_draft_tokens_, c_draft_accepted_;
    Counter c_draft_refills_, c_draft_refill_tokens_;
    Counter c_attn_cached_, c_linear_cached_, c_checkpoints_;
    Counter c_prefix_evict_, c_prefix_ckpt_evict_;
    Counter c_prefix_expired_, c_prefix_ckpt_expired_;
    Counter c_ckpt_retired_, c_ckpt_lru_, c_ckpt_refused_;
    Counter c_cut_no_snapshot_, c_cut_window_, c_cut_adopt_, c_cut_tokens_;
    Gauge   g_linear_hit_;
    Counter c_success_[(size_t)SuccessReason::N_];
    Counter c_rejected_, c_disconnect_, c_overflow_;
    Counter c_http_2xx_, c_http_4xx_, c_http_5xx_;

    Histogram h_ttft_, h_tpot_, h_e2e_, h_decode_;
    Histogram h_prompt_tokens_, h_generation_tokens_, h_params_n_, h_params_max_tokens_;
};

}  /* namespace server */
}  /* namespace rad */
