/* vram_budget.cpp -- see vram_budget.h for why a derived budget is not the §6 violation it looks
 * like. This file is the arithmetic and the receipt. */
#include "mem/vram_budget.h"

#include <algorithm>
#include <cmath>

namespace rad {

namespace {
constexpr int64_t kMiB = 1024 * 1024;

/* DOWN, ALWAYS. The pools take MiB and multiply back up, so a budget rounded up is a budget that
 * asks the card for more than the arithmetic said was there -- which fails at the allocation
 * rather than here, with the driver's message instead of ours. */
int64_t to_mib(int64_t bytes) { return bytes > 0 ? bytes / kMiB : 0; }

/* UP, FOR THE ONE BUDGET THAT IS A FLOOR. The KV budget is a CAPACITY -- truncated, the cache
 * addresses a few fewer tokens and nothing else changes. The weights budget is `statics +
 * experts`, a figure the pool must COVER, and a byte short is a tensor with nowhere to land. An
 * expert share pads the number enough to hide that; a dense model is budgeted its static
 * footprint to the byte, and there the rounding direction decides whether it loads. The MiB this
 * claims comes out of the headroom, which is what the headroom is for. */
int64_t to_mib_up(int64_t bytes) { return bytes > 0 ? (bytes + kMiB - 1) / kMiB : 0; }
}  /* namespace */

/* WHAT THE CACHE BACKS, counted on its PAGED part. The ceiling is the recurrent and conv state of
 * max_seqs sequences plus their context, and the state is spent before a single token is cached --
 * on a hybrid model at a few dozen sequences it is gigabytes a card. A ratio over the whole
 * ceiling counts that state as context and reports full-length sessions the cache cannot hold. */
static double paged_share(const VramBudget& b) {
    const double ctx = (double)(b.kv_ceiling - b.kv_fixed);
    return ctx > 0 ? std::max(0.0, (double)(b.kv - b.kv_fixed)) / ctx : 0.0;
}

/* The resolver, with its three log lines behind `log`: vram_budget_trial asks the same question
 * of a configuration it may not use, and a refusal printed for a split nobody takes reads as the
 * startup failing. */
static int resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out, bool log) {
    VramBudget b;
    if (per_rank.empty()) { if (out) *out = b; return RAD_OK; }

    /* THE SMALLEST CARD AND THE LARGEST PROGRAM, because one split is applied to every rank. Ranks
     * of one deployment normally hold the same shard shapes and the same program, so these folds
     * are no-ops -- they exist so that a mixed pair degrades to the card that binds instead of
     * overcommitting the smaller one on the first allocation. */
    b.capacity   = per_rank[0].capacity;
    int64_t freeb = per_rank[0].free;
    int64_t layer_max = 0;
    for (const VramFacts& f : per_rank) {
        b.capacity   = std::min(b.capacity, f.capacity);
        freeb        = std::min(freeb, f.free);
        b.arena      = std::max(b.arena, f.arena);
        b.staging    = std::max(b.staging, f.staging);
        if ((int64_t)f.n_queues * f.queue_bytes > b.queues) {
            b.queues      = (int64_t)f.n_queues * f.queue_bytes;
            b.n_queues    = f.n_queues;
            b.queue_bytes = f.queue_bytes;
        }
        b.kv_ceiling = std::max(b.kv_ceiling, f.kv_ceiling);
        b.kv_fixed   = std::max(b.kv_fixed, f.kv_fixed);
        b.statics    = std::max(b.statics, f.statics);
        b.expert_ceiling = std::max(b.expert_ceiling, f.experts);
        layer_max        = std::max(layer_max, f.expert_layer_max);
    }
    b.headroom = std::max<int64_t>(cfg.gpu_headroom_mib, 0) * kMiB;
    int64_t host_reserve = 0;
    for (const VramFacts& f : per_rank) host_reserve = std::max(host_reserve, f.host_reserve);
    const bool shared = host_reserve > b.headroom;
    if (shared) b.headroom = host_reserve;

    /* A host backend has no VRAM to speak of and reports none; there is nothing here to resolve
     * and inventing a capacity would be exactly the inference §6 forbids. */
    if (b.capacity <= 0) {
        b.weights = cfg.vram_weights_mib * kMiB;
        b.kv      = cfg.vram_kv_mib * kMiB;
        if (out) *out = b;
        return RAD_OK;
    }

    /* MEASURED, NOT ASSUMED, AND IT IS THE ONLY TERM THAT IS. Everything the driver and the loaded
     * code objects already hold shows up here, including a desktop compositor on the same card --
     * which is a real configuration on a workstation and the reason this is a line in the report
     * rather than a constant folded into the headroom. Read after declare(), so every kernel
     * plugin's init has run and its code object is resident. */
    b.already_held = std::max<int64_t>(b.capacity - freeb, 0);

    const bool want_w  = cfg.vram_weights_mib <= 0;
    const bool want_kv = cfg.vram_kv_mib <= 0;
    b.derived_weights = want_w;
    b.derived_kv      = want_kv;

    /* What is left after the terms nobody gets to spend. */
    int64_t fixed = b.already_held + b.headroom + b.arena + b.staging + b.queues;
    if (cfg.checkpoint_slots > 0) fixed += cfg.checkpoint_slots * 58 * kMiB;
    if (!want_w)  fixed += cfg.vram_weights_mib * kMiB;   /* a stated pool is fixed too */
    if (!want_kv) fixed += cfg.vram_kv_mib * kMiB;
    b.claimable = b.capacity - fixed;

    if (b.claimable < 0) {
        b.report = fmt(
            "VRAM does not cover the fixed costs before a single pool is sized:\n"
            "  card total         %10s\n"
            "  already held       %10s   (driver context, code objects, anything else on the card)\n"
            "  --gpu-headroom-mib %10s\n"
            "  activation arena   %10s   (buffer plan + kernel scratch)\n"
            "  step staging       %10s   (block tables, media and KV tier staging)\n"
            "  hardware queues    %10s   (%d at %s each, measured)\n"
            "%s%s"
            "  ------------------------------\n"
            "  SHORT BY           %10s\n"
            "Lower --gpu-headroom-mib, lower --max-num-batched-tokens (the arena scales with it), "
            "or free the card.\n",
            humanb(b.capacity).c_str(), humanb(b.already_held).c_str(),
            humanb(b.headroom).c_str(), humanb(b.arena).c_str(), humanb(b.staging).c_str(),
            humanb(b.queues).c_str(), b.n_queues, humanb(b.queue_bytes).c_str(),
            want_w  ? "" : fmt("  --vram-weights-mib %10s   (stated)\n",
                               humanb(cfg.vram_weights_mib * kMiB).c_str()).c_str(),
            want_kv ? "" : fmt("  --vram-kv-mib      %10s   (stated)\n",
                               humanb(cfg.vram_kv_mib * kMiB).c_str()).c_str(),
            humanb(-b.claimable).c_str());
        if (log) RAD_ERR("%s", b.report.c_str());
        if (out) *out = b;
        return RAD_E_NOMEM;
    }

    b.weights = want_w  ? 0 : cfg.vram_weights_mib * kMiB;
    b.kv      = want_kv ? 0 : cfg.vram_kv_mib * kMiB;

    /* ---- the static weights come off the top ------------------------------------------------
     *
     * They are a floor and not a share. Only a DERIVED weight budget pays them here: an operator
     * who stated --vram-weights-mib has already decided what the weights get, and second-guessing
     * that is the re-optimisation §6 forbids. */
    if (want_w) {
        b.statics = std::min(b.statics, b.claimable);
        if (b.statics >= b.claimable && b.claimable > 0) {
            /* Not fatal: the planner will demote what does not fit and the model still serves,
             * badly. Name it, because "why is this model 8x slower than it should be" has no other
             * answer at startup. */
            if (log) RAD_WARN("the static (non-expert) weights alone want %s of the %s this card has to "
                     "give. Every expert will stream from host memory and the KV cache has "
                     "nothing left -- this model wants more VRAM, fewer ranks of it, or a smaller "
                     "--max-num-batched-tokens.",
                     humanb(b.statics).c_str(), humanb(b.claimable).c_str());
        }
    }
    b.elastic = want_w ? b.claimable - b.statics : b.claimable;
    if (b.elastic < 0) b.elastic = 0;

    /* ---- and only what is left is what the ratio divides ------------------------------------- */
    if (want_w && want_kv) {
        const double r = (cfg.expert_cache_ratio > 0.0 && cfg.expert_cache_ratio < 1.0)
                       ? cfg.expert_cache_ratio : 0.75;
        int64_t kv  = (int64_t)((double)b.elastic * (1.0 - r));
        int64_t exp = b.elastic - kv;

        /* BOTH SIDES ARE CAPPED AT WHAT THEY COULD USE, AND THE SLACK CROSSES OVER. A paged cache
         * cannot address past max_seqs x max_ctx tokens and an expert plane that is entirely
         * resident has no use for another byte -- so a share above either ceiling is memory
         * nothing will ever touch. Cap, then hand the remainder to whichever side is still short,
         * then stop: what neither can take is genuinely unusable and is reported as such rather
         * than allocated into a pool that reports itself three quarters empty. */
        if (b.kv_ceiling > 0 && kv > b.kv_ceiling) kv = b.kv_ceiling;
        /* ZERO EXPERTS IS A CEILING OF ZERO, NOT AN UNMEASURED ONE. weight_footprint() runs on
         * every rank before this call, so expert_ceiling is always known -- and guarding this cap
         * with `> 0` would read "dense" as "unlimited", handing a model with no experts at all a
         * large expert budget against nothing to put in it while its cache goes short.
         * kv_ceiling keeps its guard: that one IS genuinely unset when plan_groups fails. */
        if (exp > b.expert_ceiling) exp = b.expert_ceiling;
        int64_t spare = b.elastic - kv - exp;
        if (spare > 0 && b.expert_ceiling > exp) {
            const int64_t take = std::min(spare, b.expert_ceiling - exp);
            exp += take; spare -= take;
        }
        if (spare > 0 && b.kv_ceiling > kv) {
            const int64_t take = std::min(spare, b.kv_ceiling - kv);
            kv += take; spare -= take;
        }
        b.experts  = exp;
        b.kv       = kv;
        b.weights  = b.statics + exp;
        b.unusable = spare > 0 ? spare : 0;
    } else if (want_kv) {
        b.kv = b.kv_ceiling > 0 ? std::min(b.claimable, b.kv_ceiling) : b.claimable;
        b.unusable = b.claimable - b.kv;
    } else if (want_w) {
        /* The KV pool was stated, so `claimable` is already net of it; the weights take the rest,
         * still capped at what the model has to put there. */
        int64_t exp = b.elastic;
        if (exp > b.expert_ceiling) {
            b.unusable = exp - b.expert_ceiling;
            exp = b.expert_ceiling;
        }
        b.experts = exp;
        b.weights = b.statics + exp;
    }

    /* ---- THE PREFILL STAGING REGION, out of the expert share ---------------------------------
     *
     * A prefill chunk routes to nearly every expert of every layer, so the part of each layer the
     * budget leaves off the card is read across the link by the layer's GEMM -- on the critical
     * path, at the link's rate, while the rest of the layer's work leaves the link idle. Two
     * buffers of one layer's share of that let the next layer's copy run behind the current one
     * (core/place/stager.h). They come out of the expert share because they exist only for the
     * experts that do not fit, and they sit at the top of the activation arena, which every step
     * smaller than the full plan lends back to the expert slab: a decode step keeps the residency
     * they cost.
     *
     * The share is the budget's -- what the slab alone leaves off -- which is the most a step can
     * see: the loans the KV cache and the smaller arena levels make only ever add residency. A
     * layer whose pooled part is larger than a buffer stages what fits and streams the rest. */
    if (want_w && layer_max > 0 && b.expert_ceiling > b.experts && cfg.placement != "all_vram" &&
        cfg.placement != "layer_offload") {
        const double f = (double)(b.expert_ceiling - b.experts) / (double)b.expert_ceiling;
        int64_t one = ((int64_t)std::ceil(f * (double)layer_max) + RAD_ALIGN_UNIT - 1) /
                            RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
        int n_bufs = 2;
        if (cfg.prefill_stage_mib > 0) {
            const int64_t target_one = cfg.prefill_stage_mib * kMiB;
            if (target_one > one && 3 * target_one < b.experts) {
                one = target_one;
            }
        }
        if (const char* env = std::getenv("RADIANCE_PREFILL_BUFFERS")) {
            const int v = std::atoi(env);
            if (v >= 2 && v <= 4) n_bufs = v;
        } else if (cfg.prefill_stage_mib >= 220) {
            n_bufs = 3;
        }
        if (n_bufs * one < b.experts) {
            b.stage    = n_bufs * one;
            b.experts -= b.stage;
            b.weights  = b.statics + b.experts;
        }
    }

    /* HOW MUCH OF THE ADVERTISED CONTEXT THE POOL BACKS, AND IT IS A PROVISIONING CHOICE RATHER
     * THAN A SHORTFALL.
     *
     * A KV share under max_seqs x max_ctx is not a warning. The argument that it would be -- that
     * the server accepts at the door what it cannot serve -- is wrong for a PAGED cache, and the
     * difference is the whole of vLLM's block manager: a sequence holds only the blocks its tokens
     * actually occupy (KVManager::ensure grows it by ceil_div(n_tokens, block_size) as it runs,
     * drops them at completion, and returns RAD_E_FULL for the scheduler to preempt rather than
     * failing the request). Every sequence reaching max_ctx at the same moment is the worst case,
     * not the workload -- real loads run well under half backing -- and sizing for the worst case
     * spends VRAM that a model whose experts do not fit would rather have for them.
     *
     * So it is reported and not warned about. What IS still checked, and belongs in the KV manager
     * because it is a real limit rather than a ratio, is that ONE max-length sequence fits: a
     * context no single request can complete is a configuration error however much slack the pool
     * has (core/engine_bringup.cpp, configure_kv). */
    if (log && b.derived_kv && b.kv_ceiling > 0 && cfg.max_ctx > 0)
        RAD_INFO("kv: the pool backs %.0f%% of the worst case (%lld sequences all at %lld tokens), "
                 "%.1f full-length sessions. That is concurrency at length before the scheduler "
                 "preempts, not a bound on any one request. WITH PREFIX CACHING a finished "
                 "sequence's blocks stay held so the next turn can reuse them, so the demand is "
                 "live SESSIONS and not running requests -- when they do not all fit, the cache "
                 "evicts prefixes the next turn wanted and that turn re-prefills its whole "
                 "context. --expert-vs-cache-ratio moves it.",
                 100.0 * paged_share(b),
                 (long long)cfg.max_seqs, (long long)cfg.max_ctx,
                 (double)cfg.max_seqs * paged_share(b));
    /* Rounding the floor up claims up to a MiB the split did not hand out. Take it back from the
     * cache where the cache is ours to size -- a paged pool a MiB smaller addresses a handful
     * fewer tokens and nothing else -- so the headroom stays whole. Where the KV budget was
     * STATED it is not ours to shave and the MiB comes out of the headroom instead. */
    cfg.vram_budget_derived = want_w || want_kv;
    cfg.vram_weights_mib = to_mib_up(b.weights);
    const int64_t w_over = cfg.vram_weights_mib * kMiB - b.weights;
    cfg.vram_kv_mib      = want_kv ? to_mib(std::max<int64_t>(b.kv - w_over, 0)) : to_mib(b.kv);

    /* THE RECEIPT. Every term, in the order it was spent, so that a split anybody disagrees with
     * can be argued about against numbers instead of re-derived from sysfs by hand. The unclaimed
     * line is not padding: to_mib truncates, and what it drops is real VRAM that ends up beside
     * the headroom. */
    const int64_t granted   = cfg.vram_weights_mib * kMiB + cfg.vram_kv_mib * kMiB;
    const int64_t unclaimed = b.capacity - b.already_held - b.arena - b.staging - b.queues - b.stage -
                              granted;
    const double  ela       = b.elastic > 0 ? (double)b.elastic : 1.0;
    b.report = fmt(
        "VRAM budget, per card:\n"
        "  card total         %10s\n"
        "  already held       %10s   measured: driver context, code objects, anything else resident\n"
        "  activation arena   %10s   computed from the buffer plan\n"
        "  step staging       %10s   the step batch's block tables and media staging, and the\n"
        "                                  KV tiers' copy staging\n"
        "  hardware queues    %10s   %d a card at %s each, measured: the engine's streams\n"
        "  %-18s %10s   %s\n"
        "  ------------------------------\n"
        "  claimable          %10s\n"
        "  static weights     %10s   %s\n"
        "  ------------------------------\n"
        "  elastic            %10s   what --expert-vs-cache-ratio %.2f divides\n"
        "    experts          %10s   %.0f%% of it%s\n"
        "    kv cache         %10s   %.0f%% of it%s\n"
        "%s"
        "%s"
        "  ==============================\n"
        "  vram_weights       %10s   %s\n"
        "  vram_kv            %10s   %s\n"
        "  unclaimed          %10s   the headroom, the unusable above, and what MiB rounding\n"
        "                                  could not place\n",
        humanb(b.capacity).c_str(), humanb(b.already_held).c_str(), humanb(b.arena).c_str(),
        humanb(b.staging).c_str(),
        humanb(b.queues).c_str(), b.n_queues, humanb(b.queue_bytes).c_str(),
        shared ? "left to the host" : "--gpu-headroom-mib", humanb(b.headroom).c_str(),
        shared ? "integrated graphics: this memory is the host's; --vram-kv-mib states a pool "
                 "outright"
               : "left unclaimed on purpose",
        humanb(b.claimable).c_str(),
        humanb(b.statics).c_str(),
        b.derived_weights ? "off the top: every rank's own shard, and not a share of anything"
                          : "(not charged here -- --vram-weights-mib was stated)",
        humanb(b.elastic).c_str(),
        (cfg.expert_cache_ratio > 0.0 && cfg.expert_cache_ratio < 1.0) ? cfg.expert_cache_ratio
                                                                       : 0.75,
        humanb(b.experts).c_str(), 100.0 * (double)b.experts / ela,
        b.expert_ceiling == 0
            ? " -- a dense model, so there is no expert plane for this share to fill and the\n"
              "                                  cache took everything it could address"
        : (b.experts >= b.expert_ceiling)
            ? " -- the whole expert plane is resident, so more would do nothing" : "",
        humanb(b.kv).c_str(), 100.0 * (double)b.kv / ela,
        (b.kv_ceiling > 0 && b.kv >= b.kv_ceiling)
            ? fmt(" -- backs %lld sequences all at %lld tokens at once, which is the worst case and\n"
                  "                                  more would be unusable",
                  (long long)cfg.max_seqs, (long long)cfg.max_ctx).c_str()
            : (b.kv_ceiling > 0
                   /* SAID AS A SESSION COUNT, because that is the number to compare against
                    * --max-num-seqs, and "28%" makes the reader do the arithmetic.
                    *
                    * AND IT NAMES SESSIONS, NOT RUNNING REQUESTS. "Paged blocks are held only
                    * while a sequence occupies them" stops being true once prefix caching takes a
                    * KV reference on every block it indexes: a finished sequence's blocks stay
                    * held so the next turn can reuse them. So the demand of a multi-turn workload
                    * is the number of LIVE SESSIONS, and counting running requests instead
                    * under-counts by exactly the thing that makes agent turns cheap. When the pool
                    * cannot hold them all, the cache evicts prefixes the next turn wants and that
                    * turn re-prefills its whole context (radiance:prefix_cache_evictions_total). */
                   ? fmt(" -- %.0f%% backing of the %lld x %lld worst case: %.1f full-length\n"
                         "                                  sessions. With prefix caching a "
                         "finished sequence's blocks stay held\n"
                         "                                  for reuse, so the demand is LIVE "
                         "SESSIONS and not running requests",
                         100.0 * paged_share(b),
                         (long long)cfg.max_seqs, (long long)cfg.max_ctx,
                         (double)cfg.max_seqs * paged_share(b)).c_str()
                   : ""),
        b.unusable > 0
            ? fmt("    unusable         %10s   neither side could take it: both are at their "
                  "ceiling\n", humanb(b.unusable).c_str()).c_str()
            : "",
        b.stage > 0
            ? fmt("    prefill staging  %10s   out of the experts: buffers of one layer's "
                  "non-resident experts,\n"
                  "                                  at the arena's top and lent back to the "
                  "slab while steps are small\n", humanb(b.stage).c_str()).c_str()
            : "",
        humanb(cfg.vram_weights_mib * kMiB).c_str(),
        b.derived_weights ? "static + experts" : "stated by --vram-weights-mib",
        humanb(cfg.vram_kv_mib * kMiB).c_str(),
        b.derived_kv ? "the cache's share" : "stated by --vram-kv-mib",
        humanb(unclaimed).c_str());

    if (out) *out = b;
    return RAD_OK;
}

int vram_budget_resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out) {
    return resolve(cfg, per_rank, out, true);
}

int vram_budget_trial(const Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out) {
    Config c = cfg;
    return resolve(c, per_rank, out, false);
}

}  /* namespace rad */
