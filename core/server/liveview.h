/* core/server/liveview.h -- the live view (spec §16).
 *
 * Per-card utilisation and VRAM, throughput, draft acceptance, and the expert plane coloured by
 * tier. A terminal view only: no web page, no port, no JSON.
 *
 * Piped or redirected, it prints ordinary lines instead (spec §16). There are two renderers,
 * chosen once by isatty(), and the non-interactive one emits no escape sequences, so a log file
 * stays greppable.
 *
 * The view owns no data: an engine-supplied snapshot function fills a LiveSnapshot and this only
 * draws it, which keeps the server free of any device dependency.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace rad {
namespace server {

struct SchedMetrics;

struct CardStat {
    int         index = 0;
    std::string name;
    double      util = 0.0;        /* 0..1 */
    int64_t     vram_used = 0;
    int64_t     vram_total = 0;
    double      power_w = 0.0;     /* 0 = not sampled */
    double      temp_c = 0.0;      /* 0 = not sampled */
};

/* rad::kTierCount, restated so this header need not include the program description (a tier
 * arrives here as a byte). liveview.cpp static_asserts that the two agree, since the per-tier
 * arrays below are indexed by tier. */
enum { kLiveTiers = 5 };

/* The expert plane and the placement engine's activity on it.
 *
 * The plane is one cell per (layer, expert), valued by rad::Tier. The counters say what the
 * picture cannot: residency per tier (how much of the model is in VRAM), per-layer counts
 * (whether residency is spread, since an in-order fill can starve the last layers and the draft
 * head while the global count looks healthy), and moves (whether link traffic is the engine
 * finding a working set or re-admitting what it just evicted).
 *
 * Counted once, here, so the terminal view and the web page draw the same tallies.
 */
struct ExpertPlane {
    int                  n_layers = 0;
    int                  n_experts = 0;
    std::vector<uint8_t> tier;     /* row-major, size n_layers * n_experts */

    /* Residency by rad::Tier, sized by the tier count so an unknown tier is never folded into the
     * SSD bucket. `bytes` is this rank's shard, which is what must add up against the VRAM bar. */
    int64_t units[kLiveTiers] = {0};
    int64_t bytes[kLiveTiers] = {0};
    int64_t n_units = 0;           /* expert units in the plan; not n_layers * n_experts, which
                                    * counts cells a ragged model never had */
    int64_t total_bytes = 0;
    int64_t movable = 0;           /* how many of them the heat engine is allowed to relocate */

    /* Per layer, so a starved layer shows as a number, not only as a dark band. */
    std::vector<int32_t> layer_units;
    std::vector<int32_t> layer_resident;

    /* Mover activity, cumulative since start; readers take rates by subtracting two polls. */
    int64_t promotions = 0, demotions = 0;
    /* Separate progress from thrash at the same move count; see HeatStats. */
    int64_t distinct_promoted = 0, readmits = 0;
    /* Why a dispatch moved nothing. `refused` is the guard declining (the healthy steady state);
     * the starved counts mean the mover had nowhere to put the bytes. */
    int64_t dispatches = 0, refused = 0, no_candidate = 0;
    /* Whether the router found the expert in VRAM. `routed` counts expert selections seen by the
     * heat engine, `routed_resident` the subset already resident; the difference was streamed
     * across the link (priced by LinkStat::stream_bytes). This measures how much of the work, not
     * of the model, is resident. */
    int64_t routed = 0, routed_resident = 0;
    int64_t no_record = 0, starved_slab = 0, starved_pool = 0, quarantined = 0;
    /* The prefill stager's copies (Mover::stage_units), rank 0: layers staged, units and bytes
     * copied onto the card, pooled units a buffer had no room for, and the copies that carried
     * the units. */
    int64_t stage_layers = 0, staged_units = 0, staged_bytes = 0, stage_overflow = 0;
    int64_t stage_copies = 0;
    /* Slab VRAM borrowed from the KV cache: `flex_bytes` is what the slab holds in lent VRAM,
     * `flex_capacity` what the cache lends right now. A gap between them is lent VRAM holding
     * nothing. */
    int64_t flex_bytes = 0, flex_capacity = 0;
    bool    heat_enabled = false;  /* false = the plan is static and none of the above will move */
};

/* Interconnect traffic by consumer.
 *
 * Three things share the link: the tensor-parallel all-reduce (set by model width and batch), the
 * expert mover's promotions and demotions (set by placement policy), and the MoE GEMM streaming a
 * routed expert that is not in VRAM (what the moves try to avoid). All are cumulative byte counts
 * on one clock, so rates from two polls compare directly: is the link busy because the model is
 * wide or because placement is thrashing?
 *
 * `ar_bytes` is the message payload summed over issues (see Ctx::ar_bytes). `h2d`/`d2h` are the
 * mover's measured copy sizes.
 */
struct LinkStat {
    bool    present = false;       /* false when nothing measured any of it */
    int     world = 1;
    int64_t ar_calls = 0, ar_bytes = 0;
    int64_t h2d_bytes = 0, d2h_bytes = 0;
    int64_t ssd_reads = 0, ssd_bytes = 0;
    /* The MoE GEMM reading a routed expert that was not resident: the traffic promotions aim to
     * remove. */
    int64_t stream_bytes = 0;
    int64_t prefetched = 0, prefetch_bytes = 0;
    /* The prefill stager's copies of the next layer's pooled experts (core/place/stager.h).
     * Included in `h2d_bytes`, like `prefetch_bytes`; subtract both for promotions alone. */
    int64_t staged_bytes = 0;
};

/* How the ranks ran their passes, summed over ranks (see Ctx::PassCounts). A played pass costs
 * the host a packet copy; every other kind walks the architecture's issue path, so the non-played
 * share is where a step's host time goes. */
struct PassStat {
    bool    present = false;
    int64_t played = 0, recorded = 0, audited = 0;
    int64_t issued_unkeyed = 0, issued_refused = 0, issued_first = 0;
    int64_t dispatched = 0, unordered = 0;
};

struct LiveSnapshot {
    int64_t     step = 0;
    std::string model;
    /* Copied, not referenced, so the drawing thread can hold a snapshot without racing the
     * scheduler. */
    int64_t waiting = 0, running = 0, preempted = 0;
    double  kv_util = 0.0, prefix_hit_rate = 0.0;
    int64_t prefix_evictions = 0;
    double  prefill_tps = 0.0, decode_tps = 0.0, draft_acceptance = 0.0;
    int64_t total_requests = 0, preemptions = 0;

    std::vector<CardStat> cards;
    ExpertPlane           experts;
    LinkStat              link;
    PassStat              passes;

    /* The activation arena's levels (Engine::arena_rows_), largest step first, and the one the
     * last step ran at: `arena_rows` is the step size that level serves and `arena_level` its
     * index. Empty when the plan declared no level, which is a plugin that answers no sizing
     * declare or a budget below the smallest level. */
    std::vector<int64_t> arena_levels;
    int64_t              arena_rows  = 0;
    int                  arena_level = 0;
};

void snapshot_from(const SchedMetrics& m, LiveSnapshot* out);

class LiveView {
public:
    using Source = std::function<void(LiveSnapshot&)>;

    /* `out` defaults to stdout. `interval_ms` is the redraw period; 250 ms shows a stall and
     * costs nothing measurable. */
    LiveView(Source src, int interval_ms = 250, FILE* out = nullptr);
    ~LiveView();

    void start();
    void stop();


    /* Public for tests: a snapshot in, a string out, no terminal involved. */
    std::string render_screen(const LiveSnapshot& s) const;
    std::string render_line(const LiveSnapshot& s) const;


private:
    void loop();

    Source            src_;
    int               interval_ms_;
    FILE*             out_;
    bool              tty_ = false;
    int               cols_ = 100;
    std::atomic<bool> running_{false};
    std::thread       thread_;
};

}  /* namespace server */
}  /* namespace rad */
