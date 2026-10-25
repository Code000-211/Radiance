/* vram_budget.h -- how many bytes each VRAM pool gets, decided once and printed in full.
 *
 * Spec §6 forbids deriving budgets by inference, such as from the free VRAM at startup, which
 * moves whenever another process touches the card. This is arithmetic on known, printed, stable
 * terms instead:
 *
 *   capacity             totalGlobalMem, fixed for the part.
 *   already held         capacity - free VRAM, measured once after every kernel plugin has
 *                        initialised and its code objects are resident: the driver context and
 *                        loaded code. The only observed term, printed on its own line so a
 *                        desktop on the card shows up there instead of shrinking a pool.
 *   --gpu-headroom-mib   what the operator leaves alone (default 96 MiB).
 *   arena                activation arena plus scratch, from arena_plan_bytes() on the buffer
 *                        plan, the same call that later allocates it.
 *
 * What is left is claimable, spent in two steps:
 *
 *   1. Static weights come off the top. Every non-expert weight (attention projections, norms,
 *      embedding, head) must be resident; the one exception is a token embedding that
 *      --embedding-placement puts in the pinned host pool, which weight_footprint then excludes.
 *      Under tensor parallelism each rank's share is already divided (declare runs per rank,
 *      spec §9). They are a floor, not a share, so the cache can never push attention weights to
 *      host memory. If they do not fit, startup refuses with the shortfall named.
 *
 *   2. The rest is divided between the expert plane and the KV cache. When the cache lends idle
 *      blocks to the expert slab, the experts get what the pinned host pool cannot hold and the
 *      cache gets the rest up to its ceiling, lending back every block no live request holds.
 *      Without a lender an unused block would sit empty, so experts come first and the cache keeps
 *      one --max-model-len sequence. The cache never gets less than that sequence.
 *
 * Both sides are capped at what they can use (the cache at max_seqs x max_ctx tokens,
 * KVManager::ceiling; the experts at the whole plane) and one side's excess goes to the other.
 *
 * No flag states a pool size: unbudgeted VRAM is unusable, and on a model whose experts do not
 * fit every unbudgeted GiB costs decode time. Operators move the inputs instead: --gpu-headroom-mib,
 * --max-model-len and --max-num-seqs (cache bound), --host-pool-mib (which experts must stay on
 * the card).
 *
 * The host backend uses half the machine's memory as its capacity (host_vram_bytes) and splits it
 * like a card; its weights pool holds what a host kernel cannot read where the container stores it
 * (mapped_unit in core/place/planner.h). A device reporting no capacity gets what each pool needs:
 * the weights the planner copies, and the cache at its ceiling.
 */
#pragma once
#include "../rad_core.h"

#include <string>
#include <vector>

namespace rad {

/* One rank's card and program. Every field is read from the device or computed from the declared
 * program; none is policy. */
struct VramFacts {
    int64_t capacity   = 0;   /* RadDeviceProps::vram_bytes -- totalGlobalMem */
    int64_t free       = 0;   /* RadDeviceProps::vram_free, measured now */
    int64_t arena      = 0;   /* arena_plan_bytes() for this rank's program, total incl. scratch */
    int64_t staging    = 0;   /* BatchBuilder::device_bytes(): the step slab and the media staging,
                               * and TierExec::device_bytes() when a KV tier is configured */
    /* Hardware queues this rank creates after `free` was read (compute stream, expert mover, and
     * the second compute lane if declared), at `queue_bytes` each. Measured by creating one: the
     * driver backs every queue with its own VRAM beside the engine's kernel-argument ring (33 MiB
     * per queue on an R9700, 8x the ring). */
    int     n_queues    = 0;
    int64_t queue_bytes = 0;
    /* The next-layer expert prefetch's ring and buffers (xprefetch.h), when enabled; its stream is
     * one of `n_queues`. */
    int64_t prefetch    = 0;
    int64_t kv_ceiling = 0;   /* KVManager::ceiling(max_ctx) for this rank's declared groups */
    int64_t kv_fixed   = 0;   /* KVManager::fixed_bytes(): the part of the ceiling that is state */
    /* weight_footprint() for this rank; under tensor parallelism already this card's share. */
    int64_t statics    = 0;   /* non-expert weights -- the floor, not a share of anything */
    int64_t experts    = 0;   /* the expert plane: what trades against the KV cache */
    int64_t expert_layer_max = 0;   /* the most expert bytes one routed layer holds */
    /* Experts the pinned host pool cannot hold, plus the slab's spare slots: the expert slab's
     * floor when the cache lends (expert_vram_floor, core/place/planner.h). */
    int64_t expert_floor = 0;
    int64_t kv_one     = 0;   /* all configured fixed state plus pages for one full-length context */
    /* The cache lends idle blocks to the expert slab: a placement that moves units (expert_tiered
     * or auto), not --deterministic, on a device with virtual memory management. A lending cache
     * can be sized for what live requests could need; a non-lending one would hold it empty. */
    bool    lends      = false;
    /* A loaded control vector's direction table: device bytes the engine already holds, which are
     * not part of the program or the pools and must not shrink what the plan may claim. It comes
     * out of the headroom, as every deployment keeps that unclaimed. Without this, a deployment
     * that merely *offers* a vector would plan its expert cache against a smaller card than one
     * that does not, which is exactly the difference "off == stock" cannot allow. */
    int64_t held_out   = 0;
    /* Kernel instances' own device reservations, as each reports them through RadMemFn
     * (abi/rad_abi.h): the PLE VRAM cache's payload, tags and counters. They are allocated in the
     * plugin's init(), before this reading, so they are already inside `already_held`; the figure
     * only names that part of it, and the budget must not charge it a second time. */
    int64_t plugin_held = 0;
    /* Integrated graphics: host memory the card must leave to the host, since device memory is
     * host memory there. 0 on a discrete card. */
    int64_t host_reserve = 0;
};

/* The resolved split in bytes, with every term it came from. Always filled, including for a
 * device that reports no capacity, because the placement and pool reports want the totals. */
struct VramBudget {
    int64_t capacity = 0, already_held = 0, headroom = 0, arena = 0, staging = 0;
    int64_t plugin_held = 0;          /* the named part of already_held (VramFacts::plugin_held) */
    int64_t drift = 0;                /* kDriverDrift: what already_held moves by after it is read */
    int64_t queues = 0;               /* n_queues * queue_bytes, from the rank that needs most */
    int64_t prefetch = 0;             /* the expert prefetch's ring and buffers */
    int     n_queues = 0;
    int64_t queue_bytes = 0;
    int64_t claimable = 0;            /* after the fixed terms above */
    int64_t statics = 0;              /* off the top, before the split sees anything */
    int64_t elastic = 0;              /* claimable - statics: what experts and cache share */
    int64_t expert_floor = 0;         /* the experts the host pool cannot hold */
    bool    lends = false;            /* every rank's cache lends its idle blocks */
    int64_t kv_ceiling = 0, expert_ceiling = 0;
    int64_t kv_fixed = 0;             /* the state part of kv_ceiling, which no context uses */
    int64_t experts = 0, kv = 0;      /* the two shares, each capped at its ceiling */
    int64_t unusable = 0;             /* elastic neither side could take -- both were full */
    /* Prefill staging region: two buffers, each one layer's share of the experts left off the
     * card, funded by cache slack or expert residency above the host-pool floor, appended to the
     * top of the activation arena (core/place/stager.h). Zero when every expert is resident or the
     * placement cannot move any. */
    int64_t stage = 0;
    int64_t weights = 0;              /* statics + experts: what vram_weights is set to */
    std::string report;       /* the block the engine prints at startup */
};

/* Writes cfg.vram_weights_mib / cfg.vram_kv_mib, the one number every downstream consumer reads
 * (an uneven world's cheaper rank re-splits it with vram_budget_rank_split). Takes the smallest
 * capacity and free VRAM across ranks and the largest arena, so the split fits every card.
 * Refuses, by name and size, when the fixed terms alone exceed the card. */
int vram_budget_resolve(Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out);

/* The same split on a copy of `cfg`, without writing or logging anything: for choosing between
 * two programs before resolving one (Engine::plan's table placement). */
int vram_budget_trial(const Config& cfg, const std::vector<VramFacts>& per_rank, VramBudget* out);

/* Re-split for an uneven world's cheaper rank. The resolver derives the most expensive rank's
 * split, and one block table serves every rank, so a rank whose cache costs less per token (e.g.
 * one with no attention cache) would over-reserve KV. Given the common split, the MiB this rank
 * needs to page the same tokens, and the bytes of all its weights resident (statics + all experts
 * + spare slots), moves the difference to the weights pool up to that cap; the remainder stays
 * unclaimed. No change when the need is not below the common pool. */
struct RankSplit {
    int64_t weights_mib  = 0;
    int64_t kv_mib       = 0;
    int64_t unusable_mib = 0;   /* saved from the KV pool but past the weights cap: never allocated */
    bool    resplit      = false;
};
RankSplit vram_budget_rank_split(int64_t common_weights_mib, int64_t common_kv_mib,
                                 int64_t need_kv_mib, int64_t weights_cap_bytes);

}  /* namespace rad */
