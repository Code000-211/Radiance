/* ctx.h -- the run phase. Ordinary C++ with one rule: it may only issue handles obtained during
 * declare (spec §3.2). Nothing else is constrained, and its whole job is to be cheap.
 *
 * Everything expensive already happened. Declare resolved every op against the kernel hierarchy
 * per band and per domain, the buffer plan sized one arena for the worst step, and the planner
 * assigned every weight a tier and a site. What is left for a step is: index a bucket table with
 * an integer, copy a tensor template per operand, patch in the pointers, and launch. The per-issue
 * work is filling a struct we already own, not constructing one.
 *
 * COST. An issue is single-digit nanoseconds. Nearly all of it is the RadTensor template copies at
 * 112 bytes each; the band scan is one or two compares off a cache line and the RadArgs patch is a
 * single store. tests/runtime_test.cpp times it and prints it, so the number is checkable rather
 * than claimed, and it is a tripwire -- a hundredfold miss means somebody put an allocation or a
 * map lookup back on this path.
 *
 * At a 64-layer model issuing ~20 ops a layer that is ~1280 issues, so order ten microseconds of
 * CORE time per decode step. Spec §3.2 accepts roughly 1-2 ms of launch overhead per step at 64
 * layers; the rest of that budget is the driver's own per-launch dispatch cost -- order 1 us per
 * hipLaunchKernel, which is what 1280 launches actually costs -- and it is not ours to spend. It is
 * also what HIP-graph capture would attack if v1 wanted it, which §3.2 says is a change to that
 * section and not to the ABI. The core is therefore not where the step budget goes, against a
 * Python engine's 10-20 ms of the same thing.
 *
 * A residency table with a pending event adds one rad_event_wait -- once per weight per
 * GENERATION, not per issue, which is why residency.h makes the mover bump it.
 */
#pragma once
#include "../rad_core.h"
#include "arena.h"
#include "rank_barrier.h"
#include "residency.h"
#include "stager.h"
#include "device/device.h"

#include <atomic>
#include <span>
#include <string>
#include <vector>
#include <unordered_map>

/* RadCtx is opaque in the ABI: rad_runtime.h forward-declares it and nothing else. Completing it
 * here as an empty base gives a well-defined static_cast from the C handle to the C++ object,
 * instead of a reinterpret_cast that merely happens to work. The run phase is the only component
 * that ever holds one, so this is the right place for the definition. */
struct RadCtx { };

namespace rad {

/* The widest op in the conventional vocabulary is gdn_conv_prep at sixteen operands, with
 * gdn_recurrent_update behind it at fourteen -- both grew when the two kernel plugins' schemas
 * were reconciled (docs/OPS.md, "How a schema is written"). Thirty-two leaves room for an
 * architecture plugin's private ops without putting a variable-length array on the hot path. */
constexpr int MAX_OPERANDS = 32;

typedef void (*ArchStepFn)(RadCtx*, const RadBatch*);

/* ------------------------------------------------------------------ the issue plan */
/* One resolved (op, band, domain) with everything that does not change per issue already filled
 * in: the geometry view the kernel reads its parameters out of, the scratch pointer, the instance
 * init() returned, the rank and the world size. Per issue we set `t` and `n_t` and call `launch`.
 *
 * `launch` is hoisted out of row->info so the fast path chases one pointer rather than two, and
 * `row` is diagnostics only -- it is read when a launch fails and never otherwise. */
struct IssuePlan {
    RadArgs          args{};
    RadLaunchFn      launch = nullptr;
    const KernelRow* row    = nullptr;
};

/* Per declared op. Trivially copyable on purpose: the bucket table and the issue plans live in two
 * flat Ctx-owned arrays and this holds offsets into them, so a step touches one small dense record
 * per op instead of walking two vectors' worth of indirection. */
struct OpPlan {
    const OpInfo* info     = nullptr;
    int32_t       band_off = 0;      /* into Ctx::band_table_ */
    int32_t       n_bands  = 0;
    int32_t       plan_off = 0;      /* into Ctx::issue_table_, band * RAD_N_DOMAINS + domain */
    /* The schema's operand count, copied here so the arity check is a compare against a hot field
     * rather than two pointer chases into a cold OpInfo. -1 when the op has no schema. */
    int32_t       n_opd_schema = -1;
    bool          issued   = false;  /* did the run phase ever issue it (spec §16's graph dump) */
    /* IS THIS OP A COLLECTIVE -- i.e. does issuing it put bytes on the interconnect? Set once at
     * bind, so the accounting on the issue path is a predicted branch and not a string compare.
     * The link is shared with the expert mover, so separating "how much of it is the all-reduce"
     * needs the collective's own traffic counted. */
    bool          collective = false;
};

/* ------------------------------------------------------------------ routing handoff */
/* A RING OF STAGING BANKS, indexed by step, so that the heat engine reads a histogram the device
 * actually wrote rather than one still queued behind it.
 *
 * TWO BANKS ARE NOT ENOUGH AND THE REASON IS THE HOST'S RUN-AHEAD. The dispatch that consumes
 * these runs before the step is issued and every issue path returns immediately, so on a long
 * prefill chunk -- most of a second of device time -- the host reaches the end of the prompt while
 * the card is still on its first chunks. With two banks, the copy for step S is overwritten by the
 * one for S+2 long before the device has reached either, the freshest bank always holds a copy in
 * flight, and the engine skips the layer: over 41 chunks of 2048 tokens, two banks credit 79
 * layer-histograms and skip 3800 as not yet landed.
 *
 * THE BYTES ARE NOT THE PROBLEM -- the device does execute every copy, in order, spread over its
 * own progress. A shallow ring overwrites them before anyone reads them. A ring as deep as the
 * run-ahead keeps each one until it is credited, and the dispatch drains every landed bank rather
 * than only the newest, so a burst of host steps followed by a wait still credits every histogram
 * the card produced in between.
 *
 * DEPTH IS A MEMORY-FOR-EVIDENCE TRADE AND IT IS CHEAP: one int32 an expert a bank a layer, which
 * on a 49-layer 254-expert plane is about 50 KiB a bank. What it cannot do is exceed the run-ahead
 * for free, so the engine counts what it still loses and prints it -- see the routing-evidence
 * line at shutdown. */
static const int kRouteBanks = 32;

struct RouteSlot {
    RadRouting routing{};
    int32_t*   staging[kRouteBanks] = {};   /* this layer's row of the bank's host copy */
    int        step[kRouteBanks]    = {};
    /* HOW MANY TIMES THIS LAYER REPORTED INTO THAT BANK, which is not always once. A bank is
     * indexed by the step, so every report a layer makes WITHIN one step lands in the same one
     * and overwrites the last -- and a speculative draft head reports once per draft pass, n_spec
     * times a step. Uncredited, the heat engine sees a fraction of that layer's routing and prices
     * its experts accordingly, which starves a draft head's resident set relative to every other
     * layer's. The histogram itself cannot be summed here -- it arrives by an async copy off the
     * compute stream -- but the COUNT is free, and one pass credited n times is the right
     * expectation when the passes route from the same distribution.
     *
     * THIS IS AN ACCOUNTING FIX, NOT A PERFORMANCE LEVER, and the distinction is the useful part.
     * Crediting the passes raises that layer's resident expert count and tightens the spread
     * across layers, and the step time does not move.
     *
     * WHY IT IS NULL IS THE THING TO REMEMBER: RESIDENCY AGAINST TIME IS CONVEX, STEEPLY. A cache
     * whose hot set is already in has nothing left to gain from the marginal entry, and a layer
     * holding more than half its experts already has its hot set in. Rescuing a layer at 35-45%
     * residency is worth about a millisecond a step; topping one up from 55% to 63% is worth zero
     * -- do not price the second from the first.
     */
    int        reports[kRouteBanks] = {};
    /* Whether the drain has taken this bank yet, which is what makes "the ring lost one" an exact
     * count rather than an upper bound: a bank recycled while this is false held evidence nobody
     * read. */
    bool       credited[kRouteBanks] = {};
    int        reported   = -1;      /* the last step this layer reported at all */
};

/* ------------------------------------------------------------------ construction */
/* Where a KV group's pool lives, per group, bound at load. RAD_OPK_KV resolves through this:
 * RadKVGroupBatch carries the slot mapping, the block table and the sequence lengths, but the pool
 * BASE belongs to the block manager and not to a per-step batch -- it does not change from step to
 * step and rebuilding it every step would be work the arena exists to avoid.
 *
 * One cache per BOUND LAYER, so the operand's `offset` is the layer and `layer_stride` is what
 * separates two layers of the same group. A group with no bound layers never gets here: the
 * builder refuses it at declare, because its page size would otherwise be a guess. */
/* ONE LAYER'S CACHE, WITH ITS SHAPE. The shape is here and not in the kernel because the core is
 * where every number in it already lives: the group's RadKVGroupDecl gives the head count, the head
 * dimension, the per-head state geometry and the conv width, and the block manager decided the
 * block size, the block count and the state count. A kernel handed a flat span would have to
 * reconstruct all of that from its own parameters -- and cannot: the conv window's depth is
 * `conv_width - 1 + n_spec` and the slot count is `max_num_seqs`, neither of which appears in any
 * op's parameter list. Shims check a cache operand's rank -- 4 for a paged cache, 3 for a conv
 * window -- and refuse anything else, so a RAD_OPK_KV operand may not arrive as a flat byte span.
 *
 * The layouts are the ones libref/ref_ops.h states once for the whole project:
 *
 *     FULL / WINDOW  [n_blocks, n_head_kv, block_size, 2 * head_dim]   K then V inside a slot
 *     LINEAR         [n_states, n_head_kv, state_dim[0], state_dim[1]]
 *     CONV           [n_states, n_head_kv * head_dim, conv_slots]      channel-major, then time
 */
struct KVPoolBinding {
    void*    base = nullptr;
    int64_t  layer_stride = 0;      /* bytes between one layer's cache and the next */
    int64_t  bytes = 0;             /* the whole group's pool, for the bounds check */
    uint32_t dtype = RAD_DT_INVALID;
    /* THE OPERAND NAMES AN ABSOLUTE MODEL LAYER AND THE GROUP HOLDS ONE CACHE PER BOUND LAYER, so
     * the rebase is a POSITION LOOKUP and not a subtraction. Subtracting the first bound layer is
     * right only when a group's layers are contiguous, and they need not be -- 15 attention layers
     * scattered through 64, say, where the fifth of them at absolute index 19 would come out as
     * slot 16 of a 15-slot pool. Indexed by absolute layer; -1 means the group does not
     * hold that layer, which is a caller bug rather than a runtime condition. */
    std::vector<int32_t> layer_slot;
    uint32_t rank = 0;              /* 0 when the group has no plan: the operand stays a flat span */
    int64_t  shape[RAD_MAX_RANK] = {0};
    std::vector<RadTensor> layer_tensor;
};

struct CtxDesc {
    Program*        program     = nullptr;
    int             rank        = 0;
    int             world_size  = 1;
    int             device      = 0;        /* the one device this rank owns */
    RadStream       stream      = nullptr;  /* null: the Ctx creates its own compute stream */
    ResidencyTable* residency   = nullptr;  /* null in the all-in-VRAM case */
    ArchStepFn      arch_step   = nullptr;  /* the architecture plugin's rad_arch_step */
    bool            profile_ops = false;    /* spec §16, and it changes what you measure */
    /* Whether a repeated pass may be recorded and played back. False for a caller whose pass
     * body issues work the batch does not describe -- a test driving another component through
     * run_step -- so every pass is issued, since a recording keyed on the batch would stand for
     * passes that differ. */
    bool            record_passes = true;

    /* By rad_kvgroup. Empty until the KV manager has allocated, which is why it is bound here and
     * not derived: a Ctx built before the pools exist would resolve every cache operand to null. */
    std::vector<KVPoolBinding> kv;
};

/* ------------------------------------------------------------------ Ctx */
class Ctx : public RadCtx {
public:
    Ctx() = default;
    ~Ctx();
    Ctx(const Ctx&) = delete;
    Ctx& operator=(const Ctx&) = delete;

    int  init(const CtxDesc& d);
    void shutdown();

    /* ---------------------------------------------------------- the hot path */
    int issue(rad_op h, const RadOperand* opd, int n_opd, int64_t n);
    /* A weight stager for the passes it arms for (runtime/stager.h); null for none. */
    void set_stager(OpStager* s) { stager_ = s; }

    /* ---------------------------------------------------------- the step driver */
    /* Sets the batch and calls the architecture plugin's rad_arch_step through the function
     * pointer the engine handed us. Under tensor parallel N rank threads each own one Ctx and one
     * device and are released against the SAME RadBatch by pointer (spec §1); the batch is const
     * and nobody writes it during a step. Returns RAD_OK, or the negative status of the first
     * kernel that refused -- rad_arch_step is void, so the abort has to be latched. */
    int run_step(const RadBatch* b);

    /* The rank thread must call this once before its first run_step: the device backend's "current
     * device" is thread-local, and init may well have run on the engine's thread. */
    int bind_thread();

    /* ---------------------------------------------------------- run-phase queries */
    const RadBatch* batch()      const { return batch_; }
    int             rank()       const { return rank_; }
    int             world_size() const { return world_size_; }
    RadStream       stream()     const { return stream_; }
    Program*        program()    const { return program_; }

    /* The lane subsequent issues launch on, and the stream that is. Lane 0 is `stream_` and is
     * what every architecture that does not ask for a second lane ever sees. See lane1_ for the
     * three things a second lane may not do. */
    RadStream       cur_stream() const { return lane_ ? lane1_ : stream_; }
    int             lane()       const { return lane_; }
    int             set_lane(int lane);
    /* Builds the second stream and its events on first use. Never called by an architecture that
     * does not ask for a lane, which is what keeps this feature free when it is unused. */
    int             lane_init();
    /* Order `to` behind everything already issued on `from`: an event recorded on `from` that
     * `to` waits for. The HOST does not block, and neither stream is drained -- this is the only
     * thing that makes two lanes a pipeline rather than a race. */
    int             lane_join(int from, int to);

    void* weight_ptr(rad_weight w);
    void* buf_ptr(rad_buf b);
    void* kv_ptr(rad_kvgroup g, int layer) const {
        if ((size_t)g >= kv_.size() || !kv_[g].base) return nullptr;
        const KVPoolBinding& b = kv_[g];
        const int64_t slot = (layer >= 0 && (size_t)layer < b.layer_slot.size())
                                 ? b.layer_slot[(size_t)layer]
                                 : -1;
        if (slot < 0) return nullptr;
        return (char*)b.base + slot * b.layer_stride;
    }

    /* ---------------------------------------------------------- routing (spec §5.5) */
    int route_report(int layer, const RadRouting* r);
    /* Layer `layer`'s row of the card-side histogram, when it holds `n_expert` counts; null
     * otherwise. A report naming this row is taken in place (see rad_route_counts). */
    int32_t* route_counts(int layer, int64_t n_expert) const {
        if (!route_dev_ || layer < 0 || (size_t)layer >= route_.size() || n_expert <= 0 ||
            n_expert > route_n_expert_)
            return nullptr;
        return route_dev_ + (int64_t)layer * route_n_expert_;
    }
    /* How many layers reported routing, which is how many the heat engine has to sweep. Zero on a
     * dense model, where nothing routes and bind_routing allocated nothing. */
    int               n_route_layers() const { return (int)route_.size(); }
    int64_t           route_n_expert() const { return route_n_expert_; }
    /* DRAIN, DO NOT SAMPLE. `next_histogram` returns the oldest staging bank holding a LANDED
     * histogram for a step past `after_step` and before `before_step`, or -1; a caller loops on it
     * with the step of the last one it credited and so sees every histogram the device produced,
     * in order, rather than whichever one happened to be newest when it looked. The three readers
     * below take the bank it returned. Null / -1 / 0 for a layer that has never reported.
     *
     * `before_step` IS THE STEP THE CALLER IS ABOUT TO ISSUE, and a bank of that step or later is
     * not finished. A step runs as more than one pass when it drafts, every pass reports into the
     * step's one bank, and the drain runs again before each of them: a bank credited between two
     * passes stands for the passes so far, and the rest land after `after_step` has moved to that
     * step, where nothing will ever read them.
     *
     * `landed`, when given, is kRouteBanks entries of -1 that this fills in as it asks: whether a
     * bank's copy has landed is the bank's answer, not the layer's, so a sweep over every layer
     * asks each bank's event once rather than once a layer. */
    int               next_histogram(int layer, int after_step, int before_step,
                                     int8_t* landed = nullptr) const;
    const int32_t*    last_histogram(int layer, int bank) const;
    int               last_histogram_step(int layer, int bank) const;
    /* How many times that layer reported into that bank. 1 for every layer of a non-speculative
     * step; n_spec for a draft head. See RouteSlot::reports. */
    int               last_histogram_reports(int layer, int bank) const;
    /* HISTOGRAMS THE RING LOST: a bank recycled for a later step while it still held an
     * uncredited copy, which is what says the ring is shallower than the host's run-ahead. Nothing
     * reads a dropped histogram, so this is the whole of the error in the routing-evidence
     * accounting. A further pass of the SAME step reporting into its bank is not a loss: the bank
     * counts it in `reports` and is credited for all of them at once. */
    uint64_t          route_dropped() const { return route_dropped_; }
    /* Told by the drain, so the ring knows which banks are still owed a reader. */
    void              mark_credited(int layer, int bank);
    const RadRouting* last_routing(int layer) const;
    void              set_route_d2h(bool on) { route_d2h_ = on; }
    bool              route_d2h() const { return route_d2h_; }
    int               route_flush();

    /* ---------------------------------------------------------- the placement seam */
    /* Change an op's execution site. Refuses with RAD_E_NOKERNEL, naming the op and the band, if
     * the requested domain has a hole in its bucket table -- §5.1 says the planner was told at
     * declare which sites are unavailable, so asking for one anyway is a planner bug and not a
     * runtime condition to paper over. */
    int  set_op_domain(rad_op h, int domain);
    int  op_domain(rad_op h) const;
    /* The mover calls this after relocating weights. WeightInfo::generation is honoured BETWEEN
     * steps rather than per issue: a per-issue atomic load of a cold WeightInfo line costs more
     * than the relocation it guards against, and within a step the ResidencyTable is the
     * authority on where a weight is. */
    void note_weights_moved() { weights_dirty_.store(true, std::memory_order_release); }

    /* ---------------------------------------------------------- diagnostics */
    /* OpPlan::issued -> Program::ops[i].issued, which is what /graph publishes. Called at the end
     * of every step, and it must be: without it the graph dump reports every op in the model as
     * never issued, and a reader using that to find a dead declaration gets all of them. */
    void        flush_issued();
    void        dump_profile() const;      /* spec §16, opt-in, with its caveat printed */
    const char* step_error() const { return step_fail_.c_str(); }
    int64_t     stream_waits() const { return n_stream_waits_; }
    /* Host synchronisations the runtime performed on the STEP PATH. Zero, always, unless per-op
     * timing is on -- which is the caveat in spec §16 expressed as a number a test can read,
     * rather than a claim in a comment. Teardown's drain is not counted; it is not a step. */
    int64_t     host_syncs()   const { return n_host_syncs_; }
    /* WHAT THIS RANK PUT ON THE INTERCONNECT, and why it is counted here rather than in the
     * kernel. The link between the two cards carries two completely different things -- the
     * tensor-parallel all-reduce after every attention and MLP block, and the expert mover's
     * promotions and demotions -- and they compete for the same PCIe bytes. The mover counts its
     * own (MoverStats) and this counts the collective's, so "is the link busy because the model is
     * wide or because placement is thrashing" has an answer.
     *
     * `ar_bytes` is the MESSAGE PAYLOAD: numel * the operand's element size, summed over issues.
     * At two ranks that IS the bytes this rank pushes across the link -- libr4d's one-shot exact
     * kernel pushes the whole input into the peer's scratch once, and the two-shot's
     * reduce-scatter plus all-gather is (N-1)/N of it twice, which is the same at N=2. At more
     * ranks it is a floor. A lossy wire (--tp-wire) compresses below it and the number stays the
     * payload, because what a reader is comparing against the mover's bytes is the traffic the
     * model asked for, not the encoding it got.
     *
     * Relaxed atomics: written by this rank's step thread, read by whatever thread is drawing. */
    int64_t     ar_calls() const { return ar_calls_.load(std::memory_order_relaxed); }
    int64_t     ar_bytes() const { return ar_bytes_.load(std::memory_order_relaxed); }

    /* HOW EACH PASS WAS RUN, since the host cost of a pass is almost all in which of these it
     * took: a play writes the recorded packets, every other outcome walks the architecture's
     * issue path. `issued_*` split the passes that had no tape to play by why -- the pass had no
     * key, its key was refused, or its key was seen for the first time. `audited` counts the
     * plays that re-issued to check the tape. `dispatched` is the kernels the plays wrote and
     * `unordered` how many of them went without the barrier bit (see haz_mark). Same threading as
     * ar_calls. */
    struct PassCounts {
        int64_t played = 0, recorded = 0, audited = 0;
        int64_t issued_unkeyed = 0, issued_refused = 0, issued_first = 0;
        int64_t dispatched = 0, unordered = 0;
    };
    PassCounts  pass_counts() const {
        PassCounts c;
        c.played         = pc_played_.load(std::memory_order_relaxed);
        c.recorded       = pc_recorded_.load(std::memory_order_relaxed);
        c.audited        = pc_audited_.load(std::memory_order_relaxed);
        c.issued_unkeyed = pc_unkeyed_.load(std::memory_order_relaxed);
        c.issued_refused = pc_refused_.load(std::memory_order_relaxed);
        c.issued_first   = pc_first_.load(std::memory_order_relaxed);
        c.dispatched     = pc_dispatched_.load(std::memory_order_relaxed);
        c.unordered      = pc_unordered_.load(std::memory_order_relaxed);
        return c;
    }
    void*       arena_base()   const { return arena_.base(); }
    int64_t     arena_bytes()  const { return arena_.bytes(); }
    void*       scratch()      const { return arena_.scratch(); }

    /* ---- the arena's levels ------------------------------------------------------------------
     *
     * Level 0 is the plan the arena was built for: every buffer sized for the largest step. A
     * later level is the same buffers packed for a step of at most `rows` tokens (ArenaLevel,
     * core/build/rad_bufplan.cpp), so the arena past its `end` is free while the steps are that
     * small -- and the engine lends it to the expert slab. The fixed region is at the same address
     * in every level, which is what lets a persistent buffer outlive a change of level.
     *
     * A LEVEL IS CHANGED BETWEEN STEPS, NEVER DURING ONE, and only while this rank's thread is
     * parked: every issue reads the active table, and a step that saw two would hand one op a
     * buffer another op of the same step had placed somewhere else. The operand guard in issue
     * checks against the active level's extents, so a step larger than its level is refused by
     * name rather than written past the level's end into lent memory. */
    int         add_level(int64_t rows, const std::vector<int64_t>& offset,
                          const std::vector<RadBufDecl>& decl);
    int         set_level(int level);
    int         level() const { return level_; }
    int         n_levels() const { return (int)level_tmpl_.size(); }

private:
    int  prepare();
    int  bind_arena();
    int  bind_buffers();
    int  bind_weights(bool refresh_only);
    int  bind_ops();
    int  bind_routing();
    int  bind_profiling();
    int  check_domain(rad_op h, int domain) const;
    /* Whether the kernel serving this op on `domain` reads the layout the weights are STORED in.
     * A weight is stored once, in the claiming DEVICE row's arrangement, so a domain move can
     * hand a kernel a plane it cannot read -- same byte count, different order, no shape check
     * anywhere can see it. Fills `why` with the sentence and leaves the severity to the caller. */
    int  check_domain_layout(rad_op h, int domain, std::string* why) const;
    void profile_collect();
    /* Which staging bank the heat engine may read. Mid-step that is the one this step is NOT
     * filling; between steps it is simply the freshest, so "one step stale" means exactly one
     * step and never two. */
    uint64_t route_dropped_ = 0;

    /* The handle-to-pointer resolution, shared by the issue path and rad_weight_ptr. Waits on the
     * mover's event where one is pending, on the STREAM, never on the host. */
    void* resolve_weight(rad_weight w);

    /* ---- weight TABLES (RAD_OPK_WTAB), which a routed mixture of experts issues and nothing
     * else does.
     *
     * A run of consecutively declared weights becomes a DEVICE ARRAY OF POINTERS, because after
     * placement a layer's experts are not in one place and `base + e*stride` is true only of the
     * all-in-VRAM case. One table per (op, operand position), bound at prepare and kept.
     *
     * The first issue resolves every entry and uploads the whole array from `host`, which is
     * pinned because the copy is asynchronous and never written again. After that the device
     * array changes only by patch (runtime/wtab.h): the residency table names the weights the
     * mover changed (ResidencyTable::changes), sync_residency queues each one against the tables
     * holding it, and the table's next issue resolves those entries and writes the ones whose
     * pointer differs from `cur`. A step moves a handful of experts, so that is a handful of
     * resolves, where re-walking every table whenever anything moved would be the whole expert
     * set of every routed layer on most steps. */
    struct WeightTable {
        uint64_t   key   = 0;         /* (op << 32) | operand position */
        rad_weight first = 0;
        int64_t    n     = 0;
        void**     host  = nullptr;
        void*      dev   = nullptr;
        /* THE SHAPE AGREEMENT IS CHECKED ONCE, AT THE FIRST ISSUE, beside the first upload. What
         * the check compares -- dtype, rank, shape and stride of each entry against entry 0 -- are
         * declaration facts, fixed before a step has ever run and incapable of changing
         * afterwards. */
        bool       checked = false;
        std::vector<void*>   cur;     /* what the device array holds, entry for entry */
        std::vector<int32_t> pend;    /* entries a residency change named since the last issue */
        bool                 full = false;   /* every entry: the templates were rebound */
    };
    /* Which tables hold a weight, as a list per weight: `wmemb_head_[w]` indexes `wmemb_`, and
     * -1 means none. A table's entries are declared weights, and nothing stops two ops sharing a
     * run, so a weight can be in more than one. */
    struct WMemb { int32_t slot, entry, next; };
    std::vector<int32_t>  wmemb_head_;
    std::vector<WMemb>    wmemb_;
    /* Whether a weight was ever resolved straight into an operand (RAD_W, rad_weight_ptr) rather
     * than through a table. Such a pointer is part of the pass's arguments, so a move of that
     * weight bumps `dense_epoch_`; a move of a table-only weight changes one device entry. */
    std::vector<uint8_t>  w_direct_;
    uint64_t              dense_epoch_ = 0;
    /* The residency table's changes, queued against the tables they touch. Once a pass, before
     * any of it is issued. */
    void sync_residency();
    /* Brings the table's device array up to date on the current stream: every entry on the first
     * issue, afterwards the queued ones. */
    int  table_refresh(rad_op h, int operand, WeightTable& t);
    /* ------------------------------------------------------------------ recorded passes
     *
     * A DECODE PASS IS THE SAME PASS STEP AFTER STEP: the same kernels, with the same arguments,
     * on the same queues. Everything the host does to produce it -- the architecture's block code,
     * the operand resolution in `issue`, the kernel library's argument packing, the dispatch -- is
     * therefore the same work repeated, and it is almost the whole of a rank thread's time. So a
     * pass is recorded once (the backend keeps the finished packets and argument blocks, see
     * DeviceBackend::tape_begin) and after that written straight into the queues.
     *
     * WHAT MAKES TWO PASSES THE SAME is the KEY: every field of the batch that a pass's shape or
     * arguments can depend on -- counts, the host-side bounds, every device pointer, each KV
     * group's geometry -- and the runtime state the issue path reads: the arena level, the weight
     * templates, the directly resolved weights (dense_epoch_), the op domains, and the lane and
     * host-arena bookkeeping a pass starts from. The batch builder rounds the two bounds that grow
     * with the context (batch.cpp, bound_bucket), which is what keeps a key standing for hundreds
     * of steps. The step number is not in it: nothing a pass issues depends on it except the
     * routing bank, which is the host's.
     *
     * WHAT A TAPE CANNOT HOLD is the host work inside a pass, and it is cut around each piece and
     * replays it: a HOST-site op (joined and run again, on the operands it recorded), a routed
     * layer's report (the host bookkeeping; the copy is in the tape), and a weight table's refresh
     * (the patch of whatever the mover changed since, see table_refresh). A table the mover did
     * not touch costs its step a flag test.
     *
     * TRUSTED ONLY AFTER IT IS CHECKED. A key is recorded the second time it is seen, recorded
     * again and compared the third, and only then played; a played tape is compared against a
     * fresh issue again every kAuditPlays plays. A difference means the pass depends on something
     * the key does not carry, and it fails the step naming the first dispatch that differs -- a
     * silent difference would be wrong output. */
    enum TapeKind : uint8_t { kTapePlay = 0, kTapeTable = 1, kTapeHost = 2, kTapeRoute = 3 };
    struct TapeStep {
        uint8_t kind = kTapePlay;
        uint8_t lane = 0;        /* the lane the host work ran on */
        int16_t opd = 0;         /* kTapeTable: the operand position, for its error message */
        int32_t a = 0, b = 0;    /* play [a, b) | table slot, op | host op index | route index */
    };
    struct TapeHost {
        rad_op           h = 0;
        RadLaunchFn      launch = nullptr;
        RadArgs          args{};           /* as it ran; `t` points at `t` below */
        RadTensor        t[MAX_OPERANDS] = {};
        int64_t          n = 0;
        bool             drain = false;    /* an operand the arena does not track: drain instead */
    };
    struct PassTape {
        uint64_t key = 0;
        RadTape  dev = nullptr;
        std::vector<TapeStep>  steps;
        std::vector<TapeHost>  host;
        std::vector<std::pair<int, RadRouting>> route;
        /* What the recording added to the Ctx's counters, and the bookkeeping it ended on. */
        int64_t  issues = 0, ar_calls = 0, ar_bytes = 0, stream_waits = 0, host_syncs = 0;
        bool     touch_pending[2] = {};
        bool     scratch_open[2] = {};
        int64_t  plays = 0;
        uint64_t used = 0;
        int32_t  dispatches = 0, unordered = 0;   /* its kernels, and those haz_mark left unordered */
    };
    /* WHAT EACH DEVICE OP OF A PASS BEING RECORDED READS AND WRITES, as byte ranges, so the tape
     * can be played with the dispatches that depend on nothing still running left unordered
     * behind it (haz_mark). The packet processor otherwise waits for each kernel to finish before
     * it launches the next, and on independent kernels that wait is pure idle.
     *
     * Ranges come from the operands the op was issued with and the roles its schema gives them --
     * a weight is never written in a pass, a table's entries are written only between played
     * stretches -- and the op's scratch is a range like any other. What cannot be bounded makes
     * the op OPAQUE, ordered on both sides: a kernel whose plugin does not promise it touches
     * nothing else (rad_kernel_concurrent), a collective (its flags live outside every operand),
     * an op with no schema, an operand with no extent. */
    struct HazOp {
        int32_t  rec0 = 0, rec1 = 0;      /* its dispatches in the tape */
        uint32_t op = 0;
        bool     opaque = false;
        uint32_t rd = 0, n_rd = 0, wr = 0, n_wr = 0;   /* into haz_rng_ */
    };
    std::vector<HazOp> haz_ops_;
    std::vector<std::pair<uintptr_t, uintptr_t>> haz_rng_;
    void haz_note(const OpPlan& p, const IssuePlan& ip, const RadTensor* t, int n_opd, int rec0,
                  int rec1);
    int  haz_mark(PassTape& t);

    static constexpr size_t  kTapes      = 96;     /* recorded passes kept, least recently used out */
    static constexpr int64_t kAuditPlays = 1024;

    std::vector<PassTape> tapes_;
    uint64_t              tape_clock_ = 0;
    /* Keys seen once, and keys recorded once and not yet verified. */
    uint64_t              tape_seen_[128] = {};
    size_t                tape_seen_at_ = 0;
    std::vector<uint64_t> tape_refused_;       /* keys whose pass could not be recorded */
    uint64_t              tape_told_ = 0;      /* draft passes whose first recording was logged */
    bool                  tape_on_ = false;    /* the backend can, and nothing needs every issue */
    bool                  record_passes_ = true;   /* CtxDesc::record_passes */
    PassTape*             rec_ = nullptr;      /* the tape being recorded, while one is */
    int                   rec_mark_ = 0;       /* the backend record its current play step starts at */
    std::atomic<uint64_t> domain_epoch_{0};    /* bumped by set_op_domain */

    uint64_t pass_key(const RadBatch* b) const;
    int      run_pass(const RadBatch* b);
    int      record_pass(const RadBatch* b, uint64_t key, PassTape* out, bool compare_only = false);
    int      play_pass(PassTape& t);
    void     tape_cut();                        /* while recording: close the current play step */
    void     tape_drop(PassTape& t);
    static bool tapes_agree(const PassTape& a, const PassTape& b, std::string* why);
    bool     tape_seen(uint64_t key);           /* true the second time a key is asked about */
    /* A routed layer's report. With `copy` false it is the host bookkeeping alone, which is what a
     * replay runs: the copy itself is in the tape. */
    int      route_report_impl(int layer, const RadRouting* r, bool copy);

    std::vector<WeightTable>              wtabs_;
    std::unordered_map<uint64_t, size_t>  wtab_of_;

    /* Resolves the run, refreshes the table if it moved, and fills `out` with the tensor the
     * kernel sees: entry 0's dtype and trailing extents, `shape[0]` = n, `stride[0]` = 0.
     * Returns RAD_OK, or aborts the step and returns its status. */
    int ensure_weight_table(uint64_t key, rad_weight first, int64_t n, size_t* slot_out);
    int bind_weight_tables();
    int resolve_weight_table(rad_op h, int operand, rad_weight first, int64_t n, RadTensor* out);
    int weight_table_tensor(rad_op h, int operand, const WeightTable& t, const RadTensor& e0,
                            RadTensor* out);
    void free_weight_tables();

    /* Cold, noinline paths. Kept out of issue() so the hot path stays one straight line. */
    int  abort_step(rad_op h, int band, int dom, const IssuePlan& ip, int64_t n, int status);
    int  abort_step_msg(rad_op h, int status, std::string what);

public:
    /* What rad_step_fail() reaches. Same effect as an operand rejection -- the step stops and the
     * message is what the engine prints -- with the reason coming from the caller. */
    int  step_fail(const char* what);
private:
    [[noreturn]] void fatal_issue(rad_op h, int64_t n, const char* why) const;

    /* ---- the op oracle. runtime/oracle.cpp; inert unless RADIANCE_OP_ORACLE names an op.
     * It runs the HOST kernel for the same (op, band) on a host copy of the operands the device
     * kernel was just given, and compares. Everything about it is off the hot path: `oracle_` is
     * null in every run that did not ask for it, and the one branch that reads it is the same
     * predictable not-taken compare as the profiling branch beside it. */
    struct OracleState;
    OracleState* oracle_ = nullptr;
    void  oracle_open();
    void  oracle_close();
    bool  oracle_arm(rad_op h) const;
    /* Extents, aliasing and the per-operand byte count for one issue. Separate from the copy
     * because a weight TABLE operand is an array of pointers and its snapshot is the STACKED
     * plane behind them -- a size the operand itself does not state. */
    void  oracle_size(rad_op h, int n_opd);
    int64_t oracle_opd_bytes(int i) const;
    void  oracle_snap(int n_opd, bool post);
    void  oracle_run(rad_op h, const IssuePlan& hip, int64_t seq, int n_opd);
    /* Put a re-laid weight back into the form a dense reference reads, through the DEVICE row's
     * own inverse, into the logical form the HOST row describes. False when the comparison cannot
     * be made honestly, which is a named skip. */
    bool  oracle_unrelayout(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int64_t seq,
                          int n_opd);
    /* Repeat mode (RADIANCE_OP_REPEAT): no host kernel. The DEVICE plan is re-launched on the
     * restored bytes and the two results are byte-compared, which names a kernel that is not a
     * function of its inputs. A host-vs-device oracle structurally cannot see that: both of its
     * sides run once, so a kernel that answers differently on the same bytes agrees with itself
     * exactly as often as it disagrees. */
    void  oracle_repeat(rad_op h, const IssuePlan& dip, int64_t seq, int n_opd);
    const char* ip_kernel_name(rad_op h) const;
    const IssuePlan* oracle_before(rad_op h, int band, int n_opd);
    void  oracle_after(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int n_opd);

    Program*        program_   = nullptr;
    std::vector<KVPoolBinding> kv_;   /* by rad_kvgroup; RAD_OPK_KV resolves through it */
    ResidencyTable* residency_ = nullptr;
    OpStager*       stager_ = nullptr;
    bool            staging_ = false;     /* the stager armed for the pass being issued */
    int issue_body(rad_op h, const RadOperand* opd, int n_opd, int64_t n);
    ArchStepFn      arch_step_ = nullptr;

    int       rank_       = 0;
    int       world_size_ = 1;
    int       device_     = 0;
    RadStream stream_     = nullptr;
    bool      own_stream_ = false;

    /* ---- THE SECOND LANE, and it exists for exactly one thing: overlapping the tensor-parallel
     * all-reduce with compute at prefill, which is worth on the order of 15% of prefill
     * throughput. Nothing else may use it without reading the three constraints below.
     *
     * ONE: EVERY COLLECTIVE MUST RIDE THE SAME LANE. The all-reduce carries a per-block `seq`
     * counter and double-buffers its scratch on `seq & 1`, so two launches that race differently
     * on the two ranks would pick DIFFERENT slots for the same message and deadlock or corrupt.
     * Both ranks run the same architecture code, so putting every all-reduce on lane 1 keeps them
     * ordered against each other and therefore identically numbered on both ranks. Splitting the
     * collectives ACROSS lanes is the bug this comment exists to prevent.
     *
     * TWO: A DEVICE-WIDE ARRIVAL COUNTER MUST BE KEYED ON THE STREAM. `split_ctr` in
     * r4d_gemm_fp8a8.hip is the split-K finisher's counter, self-resetting, and it is sound only
     * because two launches of this kernel never overlap -- they are ordered by the same stream.
     * One counter block for the whole device would hold only with one compute stream a rank; a
     * second lane running a narrow GEMM beside one on lane 0 breaks it. There is therefore one
     * counter block a STREAM, so ordering does the work again -- and any OTHER plugin kernel with
     * a device-wide counter has to do the same or stay off lane 1, because nothing here can check
     * it.
     *
     * THREE: THE BUFFER PLANNER PACKS FROM ONE LINEAR ORDER. Two lanes can therefore write the
     * same arena bytes through two buffers whose declared lifetimes do not overlap. The
     * architecture names the buffers its second lane touches to `rad_buf_concurrent`, which hands
     * each of them the whole program -- a few MiB for one block's transients.
     *
     * FOUR: A SLICE MUST LAND IN THE SAME BUCKET-TABLE BAND AS THE WHOLE. This is the one that
     * would silently change the model's output rather than break it. An op's kernel is chosen by
     * which band `n` falls in, and `--debug-graph` prints the boundaries a model actually has.
     * They are sparse and uneven -- a representative set is
     *
     *     8   16   56   64   256   1024   4096   8192   16384   49152   65536
     *
     * so a 4096-token chunk cut in HALF gives 2048 and 2048, both inside (1024, 4096] and both
     * resolving to the same kernel -- while cutting it in FOUR gives 1024, which is the TOP of the
     * band below and picks a different kernel with a different summation order. The ragged last
     * chunk of a prompt is worse: 16449 tokens is four full chunks and then 65, and halving 65
     * crosses the boundaries at 64 and 56. So the rule a pipelined path must obey is that every
     * slice sits strictly above the largest boundary below T, which at a 4096-token chunk means
     * TWO slices and no more.
     *
     * The T-dependent selections inside the architecture hold at two slices: `ar_wire_is_exact`
     * only flips below T = 13 (`numel * 2 < wire_min_bytes`), and the gated-GEMM fold and the
     * fused attention prologue both band at T <= 64.
     *
     * WHAT MAKES ALL OF THIS WORTH IT: a many-block SPINNING kernel on one stream does not starve
     * a bandwidth-bound kernel on another. The all-reduce launches min(M, 256) blocks that spin on
     * a peer flag while the fabric drains, and if those held their CUs there would be nothing to
     * overlap with. Run a dependent-FMA spinner and a large copy on two streams and the pair costs
     * the MAX of the two rather than the sum: the copy finishes inside the spinner's window at
     * every block count, including counts higher than the part can hold resident at once. */
    RadStream lane1_      = nullptr;
    bool      own_lane1_  = false;
    int       lane_       = 0;
    RadEvent  lane_ev_[2] = { nullptr, nullptr };
    /* THE LAST DEVICE OP ON EACH LANE THAT TOUCHED HOST-DOMAIN MEMORY. A host op reads and writes
     * the host arena at issue, on this thread, while the device may still be working through what
     * this thread issued before it. What it has to be ordered behind is every device op that
     * touched the host arena before it -- the producer of its input, or a reader of bytes its
     * output reuses -- and on each lane those complete no later than the last of them. So a device
     * op with a host-domain buffer operand re-records its lane's event, and a host op waits on the
     * recorded ones instead of on everything the stream holds (see the host join in Ctx::issue). */
    RadEvent  host_touch_ev_[2]      = { nullptr, nullptr };
    bool      host_touch_pending_[2] = { false, false };
    int       host_join(const RadOperand* opd, int n_opd);
    bool      host_join_drains(const RadOperand* opd, int n_opd) const;
    int       host_join_wait(bool drain);
    /* A lane issued a kernel that uses the shared scratch region and the other lane has not been
     * joined behind it since. The issue path reads these to keep two scratch users off the region
     * at once; lane_join clears its `from` side. See the scratch block in Ctx::issue. */
    bool      scratch_open_[2]    = { false, false };
    bool      scratch_join_warned_ = false;
    /* WHERE IN ITS REGION THE NEXT SCRATCH USER STARTS. Consecutive users are handed consecutive
     * slices rather than all the same bytes, wrapping to the start when one does not fit, so two
     * of them that are otherwise independent share nothing and may run at once (haz_mark). Back
     * to zero at the start of every pass, so a pass issued twice hands its ops the same slices --
     * the pointers are in its recording's arguments. */
    int64_t   scratch_cur_[2]     = { 0, 0 };
    void*     scratch_take(int region, int64_t bytes);
    /* The kernel that last opened each lane's scratch, and its row count, for the warning. */
    const char* scratch_kernel_[2] = { nullptr, nullptr };
    int64_t     scratch_n_[2]      = { 0, 0 };

    Arena arena_;        /* device: the buffer plan plus the shared scratch region */
    Arena host_arena_;   /* only if the program declares RAD_DOMAIN_HOST buffers */

    const RadBatch* batch_ = nullptr;

    /* Per-issue tensor storage, bound into every IssuePlan::args.t at prepare so an issue stores
     * n_t and nothing else. One array, because issues do not nest. */
    RadTensor tbuf_[MAX_OPERANDS];

    /* Prebuilt tensor templates, indexed by handle. A buffer's template already carries its arena
     * pointer, dtype, shape and packed strides; a weight's already carries WeightInfo::ptr. An
     * issue copies one and patches at most the data pointer and dim 0. */
    std::span<const RadTensor> buf_tmpl_;
    std::vector<int32_t>   buf_bits_;
    std::vector<uint8_t>   buf_host_;   /* per handle: a RAD_DOMAIN_HOST buffer */
    std::vector<uint8_t>   buf_dense_;  /* per handle: 1 if contiguous packed with offset=0 and default cols */
    /* Every level's templates; `buf_tmpl_` points into the active one, so the issue path reads
     * one span whatever level it is at. */
    std::vector<std::vector<RadTensor>> level_tmpl_;
    int                    level_ = 0;
    std::vector<RadTensor> w_tmpl_;
    std::vector<int32_t>   w_bits_;
    std::vector<uint32_t>  w_gen_;        /* the generation each template was built from */
    std::vector<uint32_t>  waited_gen_;   /* the generation we last made the stream wait on */

    std::vector<OpPlan>                 plans_;
    /* Ops issued for the first time during the current step, flushed to Program at step end. */
    std::vector<uint32_t>               first_issued_;
    std::vector<int64_t>                band_table_;
    std::vector<IssuePlan>              issue_table_;
    std::vector<std::atomic<uint8_t>>   op_domain_;   /* sized once, never resized */

    std::vector<RouteSlot> route_;
    int64_t                route_n_expert_ = 0;
    /* A BANK IS ONE STEP'S HISTOGRAMS FOR EVERY LAYER, [kRouteBanks][n_layer][n_expert] in pinned
     * host memory. A report copies one layer's counts into that layer's row on the card,
     * [n_layer][n_expert], which is a copy within the card and orders nothing else; the end of the
     * pass copies the rows it touched into the step's bank in one transfer and records the bank's
     * event. See route_report. */
    int32_t*               route_pool_     = nullptr;   /* host */
    int32_t*               route_dev_      = nullptr;   /* card */
    RadEvent               route_done_[kRouteBanks] = {};
    /* The layers this pass reported into `route_bank_`, as [lo, hi]; lo > hi when none. */
    int                    route_bank_ = -1, route_lo_ = 1, route_hi_ = 0;
    bool                   route_d2h_ = true;

    /* Profiling. Off by default; when on it restores synchronisations that change the mover's slot
     * supply, so a profiled run is not comparable against an unprofiled one (spec §16). */
    bool                 profiling_ = false;
    struct ProfSlot { RadEvent a = nullptr, b = nullptr; int32_t op = 0; };
    std::vector<ProfSlot> prof_ring_;
    size_t                prof_n_ = 0;
    bool                  prof_overflowed_ = false;
    std::vector<double>   prof_ms_;
    std::vector<int64_t>  prof_calls_;
    /* Which rows were timed on the HOST clock rather than by a device event pair. The two are not
     * the same measurement -- a host row includes page faults and link reads, a device row is
     * kernel time -- so the table says which each one is rather than letting a reader assume. */
    std::vector<uint8_t>  prof_host_;
    /* The stream drain a host-site op forces, timed apart from the op itself: it is the cost of
     * the pipeline the op STOPS, not of the arithmetic it does, and the two differ by two orders
     * of magnitude on the PLE gather. */
    std::vector<double>   prof_join_ms_;

    std::atomic<bool> weights_dirty_{false};
    /* Bumped whenever bind_weights rewrites the templates: any weight pointer resolved from one
     * before may have moved. */
    uint64_t          w_tmpl_epoch_ = 1;

    bool        in_step_     = false;
    int         step_status_ = RAD_OK;
    std::string step_fail_;

    int64_t n_issues_ = 0;
    int64_t n_stream_waits_ = 0;
    int64_t n_host_syncs_ = 0;
    std::atomic<int64_t> ar_calls_{0}, ar_bytes_{0};
    std::atomic<int64_t> pc_played_{0}, pc_recorded_{0}, pc_audited_{0};
    std::atomic<int64_t> pc_unkeyed_{0}, pc_refused_{0}, pc_first_{0};
    std::atomic<int64_t> pc_dispatched_{0}, pc_unordered_{0};
};

}  /* namespace rad */
