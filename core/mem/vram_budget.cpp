/* vram_budget.cpp -- the budget arithmetic and its printed report. See vram_budget.h for why a
 * derived budget does not violate §6. */
#include "mem/vram_budget.h"

#include <algorithm>
#include <cmath>

namespace rad {

namespace {
constexpr int64_t kMiB = 1024 * 1024;

/* Rounds down: the pools multiply MiB back up, and rounding up would ask the card for more than
 * the arithmetic found, failing at allocation with the driver's message instead of ours. */
int64_t to_mib(int64_t bytes) { return bytes > 0 ? bytes / kMiB : 0; }

/* Rounds up, for the weights budget only. The KV budget is a capacity and may be truncated; the
 * weights budget (`statics + experts`) must cover every tensor, and a dense model is budgeted its
 * static footprint to the byte. The extra MiB comes out of the headroom. */
int64_t to_mib_up(int64_t bytes) { return bytes > 0 ? (bytes + kMiB - 1) / kMiB : 0; }

/* How much the driver's own holding moves after `already_held` is read. On ROCm 7.2 the reading
 * has come in 4 MiB below what the driver later held, leaving the first kernel launches short of
 * headroom. Four times that, charged before anything is split. */
constexpr int64_t kDriverDrift = 16 * kMiB;
}  /* namespace */

/* The fraction of the paged (context) part of the ceiling the cache backs. The ceiling also
 * includes fixed recurrent and conv state (gigabytes per card on a hybrid model), which a ratio
 * over the whole ceiling would wrongly count as context. */
static double paged_share(const VramBudget& b) {
    const double ctx = (double)(b.kv_ceiling - b.kv_fixed);
    return ctx > 0 ? std::max(0.0, (double)(b.kv - b.kv_fixed)) / ctx : 0.0;
}

/* The named part of `already held`: kernel instances that hold device bytes of their own report
 * them through RadMemFn, and init() ran before the reading, so those bytes are inside already held.
 * Printed as a sub-line, never subtracted a second time; empty when no instance declares any. */
static std::string plugin_held_line(const VramBudget& b) {
    if (b.plugin_held <= 0) return std::string();
    return fmt("    of which plugins %10s   kernel instances' own reservations, the PLE VRAM cache\n"
               "                                  among them, counted inside already held\n",
               humanb(b.plugin_held).c_str());
}

/* The resolver. `log` is false for vram_budget_trial, whose split may never be used, so its
 * refusals must not print as startup failures. */
static int resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out, bool log) {
    VramBudget b;
    if (per_rank.empty()) { if (out) *out = b; return RAD_OK; }

    /* The smallest card and the largest program, since one split serves every rank. Normally a
     * no-op; a mixed pair degrades to the card that binds. An uneven world's cheaper ranks are
     * re-split afterwards (vram_budget_rank_split, from Engine::plan). */
    b.capacity   = per_rank[0].capacity;
    int64_t freeb = per_rank[0].free;
    int64_t layer_max = 0, kv_one = 0;
    for (const VramFacts& f : per_rank) {
        b.capacity   = std::min(b.capacity, f.capacity);
        freeb        = std::min(freeb, f.free);
        b.arena      = std::max(b.arena, f.arena);
        b.staging    = std::max(b.staging, f.staging);
        b.prefetch   = std::max(b.prefetch, f.prefetch);
        if ((int64_t)f.n_queues * f.queue_bytes > b.queues) {
            b.queues      = (int64_t)f.n_queues * f.queue_bytes;
            b.n_queues    = f.n_queues;
            b.queue_bytes = f.queue_bytes;
        }
        b.kv_ceiling = std::max(b.kv_ceiling, f.kv_ceiling);
        b.kv_fixed   = std::max(b.kv_fixed, f.kv_fixed);
        b.statics    = std::max(b.statics, f.statics);
        b.plugin_held = std::max(b.plugin_held, f.plugin_held);
        b.expert_ceiling = std::max(b.expert_ceiling, f.experts);
        layer_max        = std::max(layer_max, f.expert_layer_max);
        /* The rank that needs most on the card sets the floor. */
        b.expert_floor = std::max(b.expert_floor, f.expert_floor);
        kv_one = std::max(kv_one, f.kv_one);
    }
    b.lends = true;
    for (const VramFacts& f : per_rank) b.lends = b.lends && f.lends;
    b.headroom = std::max<int64_t>(cfg.gpu_headroom_mib, 0) * kMiB;
    int64_t host_reserve = 0;
    for (const VramFacts& f : per_rank) host_reserve = std::max(host_reserve, f.host_reserve);
    const bool shared = host_reserve > b.headroom;
    if (shared) b.headroom = host_reserve;

    /* A device that reports no capacity has nothing to split, and inventing one is the inference
     * §6 forbids. Each pool gets what it needs: weights what the planner copies out of the
     * container (weight_footprint), the cache its ceiling. (The host backend reports half the
     * machine's memory and is split like a card.) */
    if (b.capacity <= 0) {
        b.weights = std::max<int64_t>(b.statics + b.expert_ceiling, kMiB);
        b.kv      = std::max<int64_t>(b.kv_ceiling, kMiB);
        cfg.vram_weights_mib = to_mib_up(b.weights);
        cfg.vram_kv_mib      = to_mib_up(b.kv);
        if (log)
            b.report = fmt("the device reports no capacity: weights pool %s (what the planner "
                           "copies out of the container), cache %s (every sequence at the full "
                           "context)", humanb(cfg.vram_weights_mib * kMiB).c_str(),
                           humanb(cfg.vram_kv_mib * kMiB).c_str());
        if (out) *out = b;
        return RAD_OK;
    }

    /* The only measured term: the driver, loaded code objects and anything else on the card (e.g.
     * a desktop compositor), so it is reported on its own line rather than folded into the
     * headroom. Read after declare(), once every kernel plugin's code object is resident.
     *
     * `held_out` is taken back out: it is the engine's own allocation that the program does not
     * see (core/cvec.h's direction table), which the headroom already covers. */
    int64_t held_out = 0;
    for (const VramFacts& f : per_rank) held_out = std::max(held_out, f.held_out);
    b.already_held = std::max<int64_t>(b.capacity - freeb - held_out, 0);

    /* What is left after the fixed terms. */
    b.drift = kDriverDrift;
    b.claimable = b.capacity - (b.already_held + b.drift + b.headroom + b.arena + b.staging +
                                b.queues + b.prefetch);

    if (b.claimable < 0) {
        b.report = fmt(
            "VRAM does not cover the fixed costs before a single pool is sized:\n"
            "  card total         %10s\n"
            "  already held       %10s   (driver context, code objects, anything else on the card)\n"
            "%s"
            "  driver drift       %10s   (what that moves by after it is read)\n"
            "  --gpu-headroom-mib %10s\n"
            "  activation arena   %10s   (buffer plan + kernel scratch)\n"
            "  step staging       %10s   (block tables, media and KV tier staging)\n"
            "  hardware queues    %10s   (%d at %s each, measured)\n"
            "  expert prefetch    %10s   (its ring of expert units and buffers)\n"
            "  ------------------------------\n"
            "  short by           %10s\n"
            "Lower --gpu-headroom-mib, lower --max-num-batched-tokens (the arena scales with it), "
            "or free the card.\n",
            humanb(b.capacity).c_str(), humanb(b.already_held).c_str(),
            plugin_held_line(b).c_str(), humanb(b.drift).c_str(),
            humanb(b.headroom).c_str(), humanb(b.arena).c_str(), humanb(b.staging).c_str(),
            humanb(b.queues).c_str(), b.n_queues, humanb(b.queue_bytes).c_str(),
            humanb(b.prefetch).c_str(), humanb(-b.claimable).c_str());
        if (log) RAD_ERR("%s", b.report.c_str());
        if (out) *out = b;
        return RAD_E_NOMEM;
    }

    /* ---- the static weights come off the top, as a floor ------------------------------------ */
    b.statics = std::min(b.statics, b.claimable);
    if (b.statics >= b.claimable && b.claimable > 0) {
        /* Not fatal: the planner demotes what does not fit and the model serves slowly. Warn so
         * the slowdown is explained at startup. */
        if (log) RAD_WARN("the static (non-expert) weights alone want %s of the %s this card can "
                          "give, so every expert streams from host memory and the KV cache gets "
                          "nothing; use more cards or a smaller --max-num-batched-tokens",
                          humanb(b.statics).c_str(), humanb(b.claimable).c_str());
    }
    b.elastic = b.claimable - b.statics;
    if (b.elastic < 0) b.elastic = 0;

    /* ---- the rest is divided ---------------------------------------------------------------- */
    {
        /* An unset cache ceiling (plan_groups failed) bounds nothing. An expert ceiling of zero is
         * real: weight_footprint always measures it. */
        const int64_t kv_cap = b.kv_ceiling > 0 ? b.kv_ceiling : b.elastic;
        int64_t kv = 0, exp = 0;
        if (kv_cap + b.expert_ceiling <= b.elastic) {
            /* Both fit: nothing trades. */
            kv  = kv_cap;
            exp = b.expert_ceiling;
        } else if (b.lends) {
            /* The cache lends: size it for what live requests could need; it gives back what
             * they do not. The experts keep what the host pool cannot hold. */
            exp = std::min(b.expert_floor, b.elastic);
            kv  = std::min(kv_cap, std::max(b.elastic - exp, kv_one));
            exp = std::min(std::max(b.expert_ceiling, b.expert_floor), b.elastic - kv);
            /* One full-length sequence wins; experts it pushes off the card have no tier and the
             * planner refuses them. Warn here too, with the flags that fix it. */
            if (log && b.expert_floor > 0 && exp < b.expert_floor)
                RAD_WARN("one --max-model-len sequence's cache (%s) leaves the experts %s, short of "
                         "the %s the pinned host pool cannot hold: raise --host-pool-mib or lower "
                         "--max-model-len",
                         humanb(kv_one).c_str(), humanb(exp).c_str(),
                         humanb(b.expert_floor).c_str());
        } else {
            /* No lender: an unused block would sit empty, so experts come first and the cache
             * keeps one --max-model-len sequence. */
            kv  = std::min(kv_cap, std::max(b.elastic - b.expert_ceiling, kv_one));
            exp = std::min(b.expert_ceiling, b.elastic - kv);
        }
        if (kv > b.elastic) kv = b.elastic;
        if (exp < 0) exp = 0;
        b.experts  = exp;
        b.kv       = kv;
        b.weights  = b.statics + exp;
        b.unusable = std::max<int64_t>(b.elastic - kv - exp, 0);
    }
    /* The weights pool must exist even when empty (every weight read from the container's
     * mapping on the host backend), since the pools treat a zero budget as an unresolved one. One
     * MiB, out of the headroom. */
    b.weights = std::max<int64_t>(b.weights, kMiB);

    /* ---- the prefill staging region ----------------------------------------------------------
     *
     * Two layers' share of the experts left off the card, for the prefill stager's ring
     * (core/place/stager.h). It exists only for experts that do not fit, so it comes out of the
     * expert share (or, when the cache lends, the cache's). It sits at the top of the activation
     * arena, which steps smaller than the full plan lend back to the expert slab, so decode keeps
     * the residency.
     *
     * The share is what the slab alone leaves off, the most a step can see: KV and arena loans only
     * add residency. A layer larger than its share stages what fits and streams the rest. */
    bool stage_from_experts = false;
    if (layer_max > 0 && b.expert_ceiling > b.experts && cfg.placement != "all_vram" &&
        cfg.placement != "layer_offload") {
        const double f = (double)(b.expert_ceiling - b.experts) / (double)b.expert_ceiling;
        const int64_t one = ((int64_t)std::ceil(f * (double)layer_max) + RAD_ALIGN_UNIT - 1) /
                            RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
        if (b.lends) {
            /* Expert residency above its host-pool floor pays first, so the cache keeps its
             * configured coverage; otherwise the cache pays. Either way the region is lent to the
             * slab in small steps. */
            if (b.experts - 2 * one >= b.expert_floor) {
                b.stage = 2 * one;
                b.experts -= b.stage;
                stage_from_experts = true;
                b.weights = b.statics + b.experts;
            } else if (b.kv - 2 * one >= kv_one) {
                b.stage = 2 * one;
                b.kv   -= b.stage;
            }
        } else if (2 * one < b.experts) {
            b.stage    = 2 * one;
            b.experts -= b.stage;
            stage_from_experts = true;
            b.weights  = b.statics + b.experts;
        }
    }

    /* How much of max_seqs x max_ctx the pool backs: a provisioning choice, reported, not warned
     * about. A paged cache holds only the blocks a sequence occupies (KVManager::ensure grows it,
     * and RAD_E_FULL makes the scheduler preempt), so all sequences at max_ctx at once is the
     * worst case, not the workload, and sizing for it takes VRAM from experts that do not fit.
     * The real limit, that one max-length sequence fits, is checked in configure_kv
     * (core/engine_bringup.cpp). */
    if (log && b.kv_ceiling > 0 && cfg.max_ctx > 0)
        RAD_INFO("kv: the pool backs %.0f%% of the worst case (%lld sequences all at %lld tokens), "
                 "%.1f full-length sessions; a larger --host-pool-mib gives the cache more",
                 100.0 * paged_share(b),
                 (long long)cfg.max_seqs, (long long)cfg.max_ctx,
                 (double)cfg.max_seqs * paged_share(b));
    /* Rounding the weights up claims up to a MiB the split did not hand out; take it back from
     * the cache so the headroom stays whole. */
    cfg.vram_weights_mib = to_mib_up(b.weights);
    const int64_t w_over = cfg.vram_weights_mib * kMiB - b.weights;
    cfg.vram_kv_mib      = to_mib(std::max<int64_t>(b.kv - w_over, 0));
    /* A cache given its whole ceiling is rounded up to cover it, out of the headroom as the
     * weights' MiB is: truncated, a ceiling under a MiB would be an empty pool and one just over a
     * whole MiB a pool short of its own context. */
    if (b.kv_ceiling > 0 && b.kv >= b.kv_ceiling && cfg.vram_kv_mib * kMiB < b.kv_ceiling)
        cfg.vram_kv_mib = to_mib_up(b.kv_ceiling);

    /* The report: every term in the order spent. `unclaimed` includes what to_mib truncated,
     * which stays beside the headroom. */
    const int64_t granted   = cfg.vram_weights_mib * kMiB + cfg.vram_kv_mib * kMiB;
    const int64_t unclaimed = b.capacity - b.already_held - b.drift - b.arena - b.staging - b.queues -
                              b.prefetch - b.stage - granted;
    const double  ela       = b.elastic > 0 ? (double)b.elastic : 1.0;
    b.report = fmt(
        "VRAM budget, per card:\n"
        "  card total         %10s\n"
        "  already held       %10s   measured: driver context, code objects, anything else resident\n"
        "%s"
        "  driver drift       %10s   what that moves by after it is read: kept free\n"
        "  activation arena   %10s   computed from the buffer plan\n"
        "  step staging       %10s   the step batch's block tables and media staging, and the\n"
        "                                  KV tiers' copy staging\n"
        "  hardware queues    %10s   %d a card at %s each, measured: the engine's streams\n"
        "  expert prefetch    %10s   the next-layer prefetch's ring of expert units and buffers\n"
        "  %-18s %10s   %s\n"
        "  ------------------------------\n"
        "  claimable          %10s\n"
        "  static weights     %10s   %s\n"
        "  ------------------------------\n"
        "  elastic            %10s   %s\n"
        "    experts          %10s   %.0f%% of it%s\n"
        "    kv cache         %10s   %.0f%% of it%s\n"
        "%s"
        "%s"
        "  ==============================\n"
        "  vram_weights       %10s   %s\n"
        "  vram_kv            %10s   %s\n"
        "  unclaimed          %10s   the headroom, the unusable above, and what MiB rounding\n"
        "                                  could not place\n",
        humanb(b.capacity).c_str(), humanb(b.already_held).c_str(), plugin_held_line(b).c_str(),
        humanb(b.drift).c_str(), humanb(b.arena).c_str(), humanb(b.staging).c_str(),
        humanb(b.queues).c_str(), b.n_queues, humanb(b.queue_bytes).c_str(), humanb(b.prefetch).c_str(),
        shared ? "left to the host" : "--gpu-headroom-mib", humanb(b.headroom).c_str(),
        shared ? "integrated graphics: this memory is the host's"
               : "left unclaimed on purpose",
        humanb(b.claimable).c_str(),
        humanb(b.statics).c_str(),
        "off the top: every rank's own shard, and not a share of anything",
        humanb(b.elastic).c_str(),
        b.lends ? "the experts keep what the host pool cannot hold; the cache takes the rest\n"
                  "                                  and lends what no live request holds to the "
                  "expert slab"
                : "no lender: the experts first, the cache at least one full-length sequence",
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
                   /* Given as a session count, comparable with --max-num-seqs. Sessions, not
                    * running requests: prefix caching holds a finished sequence's blocks for the
                    * next turn, so multi-turn demand is live sessions. When they do not fit, the
                    * cache evicts prefixes and the next turn re-prefills its whole context
                    * (radiance:prefix_cache_evictions_total). */
                   ? fmt(" -- %.0f%% backing of the %lld x %lld worst case: %.1f full-length\n"
                         "                                  sessions. With prefix caching a "
                         "finished sequence's blocks stay held\n"
                         "                                  for reuse, so the demand is live "
                         "sessions, not running requests",
                         100.0 * paged_share(b),
                         (long long)cfg.max_seqs, (long long)cfg.max_ctx,
                         (double)cfg.max_seqs * paged_share(b)).c_str()
                   : ""),
        b.unusable > 0
            ? fmt("    unusable         %10s   neither side could take it: both are at their "
                  "ceiling\n", humanb(b.unusable).c_str()).c_str()
            : "",
        b.stage > 0
            ? fmt("    prefill staging  %10s   out of the %s: two buffers of one layer's "
                  "non-resident experts,\n"
                  "                                  at the arena's top and lent back to the "
                  "slab while steps are small\n", humanb(b.stage).c_str(),
                  stage_from_experts ? "experts" : "cache").c_str()
            : "",
        humanb(cfg.vram_weights_mib * kMiB).c_str(), "static + experts",
        humanb(cfg.vram_kv_mib * kMiB).c_str(), "the cache's share",
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

RankSplit vram_budget_rank_split(int64_t common_weights_mib, int64_t common_kv_mib,
                                 int64_t need_kv_mib, int64_t weights_cap_bytes) {
    RankSplit s;
    s.weights_mib = common_weights_mib;
    s.kv_mib      = common_kv_mib;
    if (need_kv_mib < 0 || need_kv_mib >= common_kv_mib) return s;
    const int64_t saved = common_kv_mib - need_kv_mib;
    /* At least the common weights pool and at most every weight resident: the weights pool is one
     * allocation, so MiB this rank could never place stay unclaimed rather than allocated empty. */
    const int64_t cap   = std::max(common_weights_mib, to_mib_up(weights_cap_bytes));
    const int64_t grown = std::min(common_weights_mib + saved, cap);
    s.kv_mib       = need_kv_mib;
    s.weights_mib  = grown;
    s.unusable_mib = saved - (grown - common_weights_mib);
    s.resplit      = true;
    return s;
}

}  /* namespace rad */
