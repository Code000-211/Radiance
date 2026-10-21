/* engine_priv.h -- what the Engine's translation units share and nothing outside them sees.
 *
 * `Engine` is one class across four files because its phases have almost nothing to say to each
 * other: bring-up runs once and touches every subsystem; the step loop runs a thousand times a
 * second and touches four; the drafters are two shapes of extra pass; the probe is debug-only and
 * reads everything. Splitting them is not decomposition of the class -- the public surface in
 * engine.h is unchanged -- it is putting each phase where it can be read on its own.
 *
 * `Engine::Rank` lives here rather than in engine.h because it names every subsystem's concrete
 * type, and engine.h is included by main.cpp and the tools: a forward declaration there is what
 * keeps a caller that only wants to serve a model from compiling the pool allocator.
 */
#pragma once
#include "engine.h"

#include "mem/pools.h"
#include "mem/kv.h"
#include "mem/prefix.h"
#include "mem/kvtier.h"
#include "mem/sstier.h"
#include "runtime/tierexec.h"
#include "build/rad_build.h"
#include "place/heat.h"
#include "place/planner.h"
#include "place/mover.h"
#include "place/stager.h"
#include "place/prefetch.h"
#include "runtime/ctx.h"
#include "sample/sampler.h"
#include "sched/advance.h"

namespace rad {

/* ================================================================== per-rank state */
/* One thread per tensor-parallel rank, each owning one device. No IPC, no shared-memory
 * handshake, no serialisation between ranks -- which is most of what a Python engine spends its
 * multiprocessing budget on. Because the ranks share an address space, the step batch is shared
 * BY POINTER rather than broadcast (spec §1). */
struct Engine::Rank {
    int       index = 0;
    int       device = 0;
    Program   program;          /* declare runs once PER RANK: dimensions are already divided */
    Plan      plan;
    Pools     pools;
    Mover     mover;
    /* THE POLICY HALF OF TIERED PLACEMENT. Bound to the plan at bringup and off unless the plan
     * is dynamic; `heat_seen[l]` is the last step whose histogram layer l has already been
     * credited, because a histogram sits in its bank until the NEXT report overwrites it and
     * crediting it twice would double every count. */
    HeatEngine       heat;
    std::vector<int> heat_seen;
    /* One sweep's histograms, found before any is credited; kept so a sweep allocates nothing. */
    struct HeatRow { int layer; int bank; const int32_t* h; };
    std::vector<HeatRow> heat_rows;
    /* WHY A LAYER'S HISTOGRAM WENT UNCREDITED, which the credited total cannot separate on its
     * own. A histogram is skipped either because its copy has not landed yet or because the bank
     * still holds a step already credited, and those are different faults: the first says the host
     * ran ahead of the device and the sample was lost, the second says the layer did not route.
     * Without the split, a placement engine starved of evidence reads exactly like a model whose
     * layers are idle. */
    uint64_t heat_credited = 0, heat_idle = 0;
    Residency residency;
    KVManager kv;
    PrefixCache prefix;
    /* THE IDLE SESSION TIERS (spec 7.3). Rank 0 owns the policy and the session index,
     * because they describe CONVERSATIONS and a conversation is not sharded -- but the
     * transfer is per rank, since each rank holds its own shard of every KV block and
     * has its own pool to gather out of. The disk store is rank 0's alone for the same
     * reason the policy is: one store, keyed by a hash of tokens every rank agrees on.
     * Both are inert unless the operator gave a tier a size. */
    IdleTiers tiers;
    TierExec  tierx;
    /* The activation arena's smaller levels, smallest step first (Engine::probe_arena_levels). */
    std::vector<ArenaLevel> arena_levels;
    /* Per Ctx level (0 is the full plan): the bytes at the top of this rank's arena that level
     * leaves to the expert slab. */
    std::vector<int64_t>    arena_lent;
    /* THE PREFILL STAGER (core/place/stager.h): the next routed layer's pooled experts, copied into
     * the top of this rank's arena while a full-plan step runs the layer before. */
    PrefillStager stager;
    /* THE FILE STAGER (core/place/stager.h): a routed layer's file-tier experts, read from the
     * container into one of the mover's two file buffers before the layer's first expert op, on
     * every pass. With both stagers in use the chain hands each the passes it armed for. */
    FileStager  fstager;
    StagerChain stagers;
    CedPrefetcher prefetcher;
    Ctx       ctx;
    Sampler   sampler;          /* one per rank: each samples its own vocab shard (spec §9) */
    rad_buf   tok_buf = 0;      /* where the chain writes the sampled ids */
    /* THE CARD'S STEP STATE (sched/advance.h): one StepSlot a scheduler slot, and the
     * end-of-generation ids acceptance stops at, sorted. Every rank keeps its own and computes the
     * same values from the same sampled ids. */
    StepSlot* step_slots = nullptr;
    int32_t*  eog = nullptr;
    int32_t   n_eog = 0;
    int32_t*  vocab_ids_dev = nullptr;
    int32_t   n_vocab_ids = 0;
};

/* THE TIERS' WORK IN FLIGHT (Engine::tier_tick). One copy batch out on every rank's transfer
 * stream at a time, with each rank's answer kept until all of them have one -- a rank that has
 * landed must not be asked again and forget it failed -- and one disk batch out on the IO thread. */
struct TierPump {
    std::vector<TierJob> copy;
    std::vector<int>     copy_st;    /* per rank: 0 in flight, 1 landed, negative failed */
    TierIO               io;
    int64_t              max_copy = 0;
    int                  slow_budget = 0;
    /* The requests a read off disk is out for (IdleTiers::plan_fetch). Asked again while their
     * read is out, they wait without walking their prompt; the read's collection lets them go. */
    std::vector<uint64_t> waiting;
};

/* THE BUFFER PROBE (engine_probe.cpp). Declared here because three files call it: the step
 * loop dumps after a trunk step, and each drafter dumps after its own passes -- a drafter that
 * runs as extra passes leaves every one of its buffers holding the PREVIOUS step's values at the
 * moment the step loop dumps. */
void dump_buffers(Ctx& ctx, int step, const char* names, const char* phase = "");

}  /* namespace rad */
