/* ctx.h -- the run phase. Ordinary C++ with one rule: it may only issue handles obtained during
 * declare (spec §3.2).
 *
 * Declare already resolved every op against the kernel hierarchy per band and domain, the buffer
 * plan sized one arena for the worst step, and the planner assigned every weight a tier and site.
 * A step indexes a bucket table, copies a tensor template per operand, patches in pointers and
 * launches.
 *
 * Cost: an issue is single-digit nanoseconds, mostly the 112-byte RadTensor template copies.
 * tests/runtime_test.cpp times it; a hundredfold miss means an allocation or map lookup got onto
 * this path. A 64-layer model issues ~1280 ops a decode step, ~10 us of core time; launching them
 * through hipLaunchKernel would cost 1-2 ms of driver time, which the device backend avoids by
 * writing AQL packets itself (spec §3.2).
 *
 * A pending residency event adds one rad_event_wait per weight per generation, not per issue
 * (residency.h).
 */
#pragma once
#include "../rad_core.h"
#include "arena.h"
#include "rank_barrier.h"
#include "residency.h"
#include "stager.h"
#include "device/device.h"

#include <atomic>
#include <memory>
#include <string>
#include <deque>
#include <vector>
#include <unordered_map>

/* RadCtx is opaque in the ABI (rad_runtime.h only forward-declares it). Completing it as an empty
 * base makes the C handle to C++ object conversion a well-defined static_cast. */
struct RadCtx { };

namespace rad {

class Registry;
struct XpState;
struct XpLayer;

/* The widest conventional op is gdn_conv_prep at sixteen operands (docs/OPS.md). Thirty-two leaves
 * room for an architecture plugin's private ops without a variable-length array on the hot path. */
constexpr int MAX_OPERANDS = 32;

typedef void (*ArchStepFn)(RadCtx*, const RadBatch*);

/* "[shape]/[stride]" of a tensor's live dimensions, for diagnostics. */
std::string tensor_shape_str(const RadTensor& t);

/* ------------------------------------------------------------------ the issue plan */
/* One resolved (op, band, domain) with everything that does not change per issue filled in: the
 * geometry view, the scratch pointer, the instance init() returned, rank and world size. Per issue
 * we set `t` and `n_t` and call `launch`.
 *
 * `launch` is hoisted out of row->info to save a pointer chase; `row` is read only when a launch
 * fails. */
struct IssuePlan {
    RadArgs          args{};
    RadLaunchFn      launch = nullptr;
    const KernelRow* row    = nullptr;
    /* For a row that ends the link hold itself: the geometry's parameters plus `link_gate`. Null
     * for every other row, so a step that holds nothing (every decode step) never reads the word. */
    const RadParam*  gate_p = nullptr;
    int              gate_n_p = 0;
    /* Resolved::ranged_div: an issued value this does not divide is refused. 0 for most rows. */
    int64_t          ranged_div = 0;
};

/* Per declared op. Trivially copyable: it holds offsets into two flat Ctx-owned arrays (bucket
 * table, issue plans), so a step touches one small dense record per op. */
struct OpPlan {
    const OpInfo* info     = nullptr;
    int32_t       band_off = 0;      /* into Ctx::band_table_ */
    int32_t       n_bands  = 0;
    int32_t       plan_off = 0;      /* into Ctx::issue_table_, band * RAD_N_DOMAINS + domain */
    /* The schema's operand count, cached so the arity check avoids a cold OpInfo. -1 when the op
     * has no schema. */
    int32_t       n_opd_schema = -1;
    bool          issued   = false;  /* did the run phase ever issue it (spec §16's graph dump) */
    /* Whether issuing it puts bytes on the interconnect. Set at bind from the resolved rows
     * (rad_kernel_collective) and the conventional all_reduce. The weight stager's copies share
     * the link and give way to it while it runs (Ctx::issue_body). */
    bool          collective = false;
    /* The conventional all_reduce, whose `n` is the message's element count, so its link bytes
     * can be counted (ar_bytes) apart from the expert mover's. */
    bool          billed = false;
    /* Runtime LoRA: this gemm's row in Program::lora_sites, -1 for every op that is not one. */
    int32_t       lora_site = -1;
};

/* ------------------------------------------------------------------ routing handoff */
/* A ring of routing-histogram staging banks, indexed by step, so the heat engine reads histograms
 * the device has actually written.
 *
 * The ring must be as deep as the host's run-ahead. Issue returns immediately, so on a long
 * prefill the host can be many steps ahead of the card; a shallow ring overwrites a bank before
 * its copy lands, and the layer's evidence is lost. The drain credits every landed bank, not only
 * the newest.
 *
 * Cost is one int32 per expert per layer per bank (~50 KiB a bank for 49 layers x 254 experts).
 * Losses still happen past the ring's depth and are counted (route_dropped) and printed at
 * shutdown. */
static const int kRouteBanks = 32;

struct RouteSlot {
    RadRouting routing{};
    int32_t*   staging[kRouteBanks] = {};   /* this layer's row of the bank's host copy */
    int        step[kRouteBanks]    = {};
    /* How many times this layer reported into that bank. A draft head reports once per draft
     * pass (n_spec times a step) into the same step's bank, each overwriting the last. The
     * histograms arrive by async copy and cannot be summed here, so the bank is credited `reports`
     * times, which is right when the passes route from the same distribution. Without it the heat
     * engine undercounts the draft head's routing and starves its resident set. */
    int        reports[kRouteBanks] = {};
    /* Whether the drain has taken this bank. A bank recycled while this is false is an exact
     * count of one lost histogram. */
    bool       credited[kRouteBanks] = {};
    int        reported   = -1;      /* the last step this layer reported at all */
};

/* ------------------------------------------------------------------ construction */
/* Where a KV group's pool lives, bound at load; RAD_OPK_KV resolves through it. The per-step
 * RadKVGroupBatch carries slot mapping, block table and lengths; the pool base belongs to the
 * block manager and does not change between steps.
 *
 * One cache per bound layer: the operand's `offset` is the layer, `layer_stride` separates layers.
 * The builder refuses a group with no bound layers at declare.
 *
 * The shape lives here because the core holds every number in it (RadKVGroupDecl plus the block
 * manager's sizes); a kernel cannot rebuild it from op parameters, since the conv window depth
 * (`conv_width - 1 + n_spec`) and slot count (`max_num_seqs`) are in none. Shims refuse a cache
 * operand that is not rank 4 (paged) or rank 3 (conv), so it never arrives as a flat span.
 * Layouts, as libref/ref_ops.h defines them:
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
    /* Absolute model layer -> cache slot in this group. A lookup, not a subtraction, because a
     * group's layers need not be contiguous. -1: the group does not hold that layer (a caller
     * bug). */
    std::vector<int32_t> layer_slot;
    uint32_t rank = 0;              /* 0 when the group has no plan: the operand stays a flat span */
    int64_t  shape[RAD_MAX_RANK] = {0};
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
     * issues work the batch does not describe (e.g. a test driving another component through
     * run_step), since a recording keyed on the batch would replay the wrong pass. */
    bool            record_passes = true;

    /* By rad_kvgroup. Bound rather than derived because the pools may not exist yet when the Ctx
     * is built. */
    std::vector<KVPoolBinding> kv;

    /* The kernel registry, read only by the op oracle (RADIANCE_OP_ORACLE) to resolve the
     * reference library's host rows. Null: the oracle checks a band only where the reference
     * library won its host row. */
    Registry*       registry    = nullptr;
};

/* ------------------------------------------------------------------ Ctx */
class Ctx : public RadCtx {
public:
    Ctx();   /* out of line, beside ~Ctx: xp_ points to a type this header only names */
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
    /* Sets the batch and calls the architecture plugin's rad_arch_step. Under tensor parallel each
     * rank thread owns one Ctx and device, and all share the same const RadBatch by pointer
     * (spec §1). Returns RAD_OK or the first refusing kernel's status, latched because
     * rad_arch_step is void. */
    int run_step(const RadBatch* b);

    /* Call once on the rank thread before its first run_step: the backend's current device is
     * thread-local, and init may have run on another thread. */
    int bind_thread();
    /* The runtime LoRA slot every row of the generator passes that follow runs with, -1 for none
     * (Ctx::issue_lora). Set by the pass loop before it releases the rank for a pass. */
    void set_gen_lora_slot(int32_t s) { gen_lora_slot_ = s; }

    /* ---------------------------------------------------------- run-phase queries */
    const RadBatch* batch()      const { return batch_; }
    int             rank()       const { return rank_; }
    /* Whether repeated passes are recorded and played back (prepare() says when they are not). */
    bool            records_passes() const { return tape_on_; }
    int             world_size() const { return world_size_; }
    RadStream       stream()     const { return stream_; }
    Program*        program()    const { return program_; }

    /* The lane subsequent issues launch on, and its stream. Lane 0 is `stream_`. See lane1_ for
     * the constraints on a second lane. */
    RadStream       cur_stream() const { return lane_ ? lane1_ : stream_; }
    int             lane()       const { return lane_; }
    int             set_lane(int lane);
    /* Builds the second stream and its events on first use; free when no lane is asked for. */
    int             lane_init();
    /* Order `to` behind everything already issued on `from` (event record + stream wait). The
     * host does not block and neither stream is drained. */
    int             lane_join(int from, int to);

    void* weight_ptr(rad_weight w);
    void* buf_ptr(rad_buf b);

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
    /* How many layers report routing (the heat engine's sweep). Zero on a dense model. */
    int               n_route_layers() const { return (int)route_.size(); }
    int64_t           route_n_expert() const { return route_n_expert_; }

    /* ---------------------------------------------------- the next-layer expert prefetch
     * (xprefetch.h). xp_open at bringup, after placement has made its pinned pools, with their
     * device ranges and unit strides. Where there is nothing to do it stays off and the three
     * calls are no-ops. */
    void xp_open(const uint64_t* lo, const uint64_t* hi, const uint64_t* stride, int n);
    void xp_close();
    int  xp_predict(const RadPrefetchNext* next, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd);
    int  xp_mark();
    int  xp_step(int layer);
    /* Drain, do not sample. `next_histogram` returns the oldest bank holding a landed histogram
     * for a step in (`after_step`, `before_step`), or -1. A caller loops on it with the step it
     * last credited, so it sees every histogram in order. The readers below take that bank; they
     * return null / -1 / 0 for a layer that never reported.
     *
     * `before_step` is the step about to be issued; its bank is unfinished, because a drafting
     * step runs several passes into one bank and the drain runs before each.
     *
     * `landed`, when given, is kRouteBanks entries initialised to -1 and filled in as banks are
     * queried, so a sweep over all layers queries each bank's event once. */
    int               next_histogram(int layer, int after_step, int before_step,
                                     int8_t* landed = nullptr) const;
    const int32_t*    last_histogram(int layer, int bank) const;
    int               last_histogram_step(int layer, int bank) const;
    /* Reports into that bank: 1 normally, n_spec for a draft head. See RouteSlot::reports. */
    int               last_histogram_reports(int layer, int bank) const;
    /* Histograms the ring lost: banks recycled for a later step while still uncredited, i.e. the
     * ring is shallower than the host's run-ahead. Another pass of the same step reporting into
     * its bank is not a loss (it counts in `reports`). */
    uint64_t          route_dropped() const { return route_dropped_; }
    /* Called by the drain, so the ring knows which banks are still unread. */
    void              mark_credited(int layer, int bank);
    const RadRouting* last_routing(int layer) const;

    /* ---------------------------------------------------------- the placement seam */
    /* Change an op's execution site. Refuses with RAD_E_NOKERNEL, naming op and band, if the
     * domain has a hole in its bucket table: the planner knew the unavailable sites at declare
     * (§5.1), so this is a planner bug. */
    int  set_op_domain(rad_op h, int domain);
    int  op_domain(rad_op h) const;
    /* ---------------------------------------------------------- diagnostics */
    /* OpPlan::issued -> Program::ops[i].issued, which /graph publishes. Must run at the end of
     * every step, or the graph dump reports every op as never issued. */
    void        flush_issued();
    void        dump_profile() const;      /* spec §16, opt-in, with its caveat printed */
    const char* step_error() const { return step_fail_.c_str(); }
    int64_t     stream_waits() const { return n_stream_waits_; }
    /* Host synchronisations on the step path. Always zero unless per-op timing is on (spec §16);
     * tests read it. Teardown's drain is not counted. */
    int64_t     host_syncs()   const { return n_host_syncs_; }
    /* This rank's all-reduce traffic. The inter-card link carries both the tensor-parallel
     * all-reduce and the expert mover's transfers; the mover counts its own (MoverStats), so the
     * two can be told apart.
     *
     * `ar_bytes` is the message payload, numel * element size, summed over issues. At two ranks
     * that equals the bytes pushed across the link (one-shot and two-shot alike); at more ranks it
     * is a floor. A lossy wire (--tp-wire) still counts the uncompressed payload.
     *
     * Relaxed atomics: written by this rank's step thread, read by any thread. */
    int64_t     ar_calls() const { return ar_calls_.load(std::memory_order_relaxed); }
    int64_t     ar_bytes() const { return ar_bytes_.load(std::memory_order_relaxed); }

    /* How each pass was run. A play writes recorded packets; every other outcome walks the issue
     * path. `issued_*` split unplayed passes by cause: no key, key refused, key seen first time.
     * `audited` counts plays re-issued to check the tape. `dispatched` is the kernels plays wrote,
     * `unordered` those without the barrier bit (haz_mark). Same threading as ar_calls. */
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
    void*       scratch()      const { return arena_.scratch(); }

    /* ---- the arena's levels ------------------------------------------------------------------
     *
     * Level 0 is the plan the arena was built for, every buffer sized for the largest step. A
     * later level packs the same buffers for steps of at most `rows` tokens (ArenaLevel,
     * core/build/rad_bufplan.cpp); the arena past its `end` is then lent to the expert slab. The
     * fixed region has the same address in every level, so persistent buffers survive a change.
     *
     * Invariant: the level changes only between steps, while this rank's thread is parked. The
     * operand guard in issue checks the active level's extents, so a step larger than its level
     * is refused instead of writing into lent memory. */
    int         add_level(int64_t rows, const std::vector<int64_t>& offset,
                          const std::vector<RadBufDecl>& decl);
    int         set_level(int level);
    int         level() const { return level_; }

    /* The key a pass over `b` is recorded and played under (see "recorded passes" below); 0 for a
     * pass that is never recorded. Public so tests can check that it reads every dependency. */
    uint64_t    pass_key(const RadBatch* b) const;

private:
    int  prepare();
    int  bind_arena();
    int  bind_buffers();
    int  bind_weights();
    int  bind_ops();
    int  issue_lora(rad_op h, const RadOperand* opd, int n_opd, int64_t n);
    int32_t gen_lora_slot_ = -1;
    int  bind_routing();
    int  bind_profiling();
    int  check_domain(rad_op h, int domain) const;
    /* Whether the kernel serving this op on `domain` reads the layout the weights are stored in.
     * A weight is stored once, in the claiming device row's arrangement, so a domain move can hand
     * a kernel a plane of the right size in the wrong order. Fills `why`; the caller picks the
     * severity. */
    int  check_domain_layout(rad_op h, int domain, std::string* why) const;
    void profile_collect();
    uint64_t route_dropped_ = 0;

    /* Handle-to-pointer resolution for the issue path and rad_weight_ptr. Waits on a pending mover
     * event on the stream, never on the host. */
    void* resolve_weight(rad_weight w);

    /* ---- weight tables (RAD_OPK_WTAB), issued by routed mixture-of-experts layers.
     *
     * A run of consecutively declared weights becomes a device array of pointers, because after
     * placement a layer's experts are not contiguous. One table per (op, operand position), bound
     * at prepare.
     *
     * The first issue resolves every entry and uploads the array from `host` (pinned, since the
     * copy is async). After that the device array changes only by patch (runtime/wtab.h):
     * sync_residency queues ResidencyTable::changes against the tables holding each weight, and
     * the next issue resolves those entries and writes the ones that differ from `cur`. */
    struct WeightTable {
        uint64_t   key   = 0;         /* (op << 32) | operand position */
        rad_weight first = 0;
        int64_t    n     = 0;
        void**     host  = nullptr;
        void*      dev   = nullptr;
        /* Entries' dtype, rank, shape and stride are checked against entry 0 once, at the first
         * issue; they are declaration facts and cannot change. */
        bool       checked = false;
        std::vector<void*>   cur;     /* what the device array holds, entry for entry */
        std::vector<int32_t> pend;    /* entries a residency change named since the last issue */
        bool                 full = false;   /* every entry: the templates were rebound */
    };
    /* Which tables hold a weight, as a linked list per weight: `wmemb_head_[w]` indexes `wmemb_`,
     * -1 means none. Two ops may share a run, so a weight can be in several tables. */
    struct WMemb { int32_t slot, entry, next; };
    std::vector<int32_t>  wmemb_head_;
    std::vector<WMemb>    wmemb_;
    /* Whether a weight was ever resolved directly into an operand (RAD_W, rad_weight_ptr) rather
     * than through a table. Such a pointer is in the pass's arguments, so moving it bumps
     * `dense_epoch_`; moving a table-only weight changes one device entry. */
    std::vector<uint8_t>  w_direct_;
    uint64_t              dense_epoch_ = 0;
    /* Queues the residency table's changes against the tables they touch. Once per pass, before
     * anything is issued. */
    void sync_residency();
    /* Brings the table's device array up to date on the current stream: every entry on the first
     * issue, afterwards the queued ones. */
    int  table_refresh(rad_op h, int operand, WeightTable& t);
    /* ------------------------------------------------------------------ recorded passes
     *
     * A decode pass repeats step after step: same kernels, arguments and queues. The host work
     * that produces it (block code, operand resolution, argument packing, dispatch) is most of a
     * rank thread's time, so a pass is recorded once (DeviceBackend::tape_begin keeps the finished
     * packets and argument blocks) and then written straight into the queues.
     *
     * Key: every batch field a pass's shape or arguments can depend on (counts, host-side bounds,
     * device pointers, KV group geometry) plus the runtime state the issue path reads (arena
     * level, weight templates, dense_epoch_, op domains, lane and host-arena bookkeeping). The
     * batch builder rounds the context-growing bounds (batch.cpp, bound_bucket) so a key holds for
     * many steps. The step number is excluded; only the host-side routing bank depends on it.
     *
     * Host work inside a pass is cut out of the tape and replayed: host-site ops (rerun on their
     * recorded operands), routed-layer reports (host bookkeeping; the copy is in the tape) and
     * weight-table refreshes (table_refresh). An untouched table costs a flag test.
     *
     * Verification: a key is recorded on its second sighting (first, for decode), recorded again
     * and compared on the next, and only then played. Every kAuditPlays plays the tape is compared against a fresh
     * issue. A difference means the key misses a dependency; the step fails, naming the first
     * differing dispatch. */
    enum TapeKind : uint8_t { kTapePlay = 0, kTapeTable = 1, kTapeHost = 2, kTapeRoute = 3,
                              kTapeXp = 4, kTapeXpMark = 5, kTapeXpStep = 6 };
    /* A prefetch call (Ctx::xp_predict), run again where the pass made it. */
    struct TapeXp {
        RadPrefetchNext next{};
        rad_buf         h = 0;
        int64_t         r0 = 0, rows = 0, n_embd = 0;
    };
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
        std::vector<TapeXp> xp;
        /* What the recording added to the Ctx's counters, and the bookkeeping it ended on. */
        int64_t  issues = 0, ar_calls = 0, ar_bytes = 0, stream_waits = 0, host_syncs = 0;
        bool     touch_pending[2] = {};
        bool     scratch_open[2] = {};
        int64_t  plays = 0;
        uint64_t used = 0;
        int32_t  dispatches = 0, unordered = 0;   /* its kernels, and those haz_mark left unordered */
    };
    /* The byte ranges each device op of a recorded pass reads and writes, so haz_mark can leave
     * dispatches that depend on nothing still running without the barrier bit. Otherwise the
     * packet processor waits for each kernel to finish before launching the next.
     *
     * Ranges come from the issued operands and their schema roles (weights are never written in a
     * pass; table entries only between played stretches); scratch is a range like any other. An
     * op is opaque, ordered on both sides, if it cannot be bounded: a kernel not marked
     * rad_kernel_concurrent, a collective (its flags are outside every operand), an op with no
     * schema, or an operand with no extent. */
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

    static constexpr size_t  kTapes      = 24;     /* recorded passes kept, least recently used out */
    static constexpr int64_t kAuditPlays = 1024;

    std::vector<PassTape> tapes_;
    uint64_t              tape_clock_ = 0;
    /* Keys seen once, and keys recorded once and not yet verified. */
    uint64_t              tape_seen_[32] = {};
    size_t                tape_seen_at_ = 0;
    std::vector<uint64_t> tape_refused_;       /* keys whose pass could not be recorded */
    uint64_t              tape_told_ = 0;      /* draft passes whose first recording was logged */
    bool                  tape_on_ = false;    /* the backend can, and nothing needs every issue */
    bool                  record_passes_ = true;   /* CtxDesc::record_passes */
    PassTape*             rec_ = nullptr;      /* the tape being recorded, while one is */
    int                   rec_mark_ = 0;       /* the backend record its current play step starts at */
    std::atomic<uint64_t> domain_epoch_{0};    /* bumped by set_op_domain */

    int      run_pass(const RadBatch* b);
    int      record_pass(const RadBatch* b, uint64_t key, PassTape* out);
    int      play_pass(PassTape& t);
    void     tape_cut();                        /* while recording: close the current play step */
    /* While recording: `st` becomes a host step that re-runs at every play. Closes the current
     * play step, appends `st` on this lane, and runs `run` with the device tape paused. */
    template <class F> int tape_host(TapeStep st, F&& run) {
        tape_cut();
        st.lane = (uint8_t)lane_;
        rec_->steps.push_back(st);
        tape_pause(true);
        const int rc = run();
        tape_pause(false);
        return rc;
    }
    void     tape_drop(PassTape& t);
    static bool tapes_agree(const PassTape& a, const PassTape& b, std::string* why);
    bool     tape_seen(uint64_t key);           /* true the second time a key is asked about */
    /* A routed layer's report. `copy` false runs only the host bookkeeping, as a replay does; the
     * copy is in the tape. */
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

    /* Cold, noinline paths, kept out of issue(). */
    int  abort_step(rad_op h, int band, int dom, const IssuePlan& ip, int64_t n, int status);
    int  abort_step_msg(rad_op h, int status, std::string what);

public:
    /* Backs rad_step_fail(): stops the step like an operand rejection, with the caller's
     * message. */
    int  step_fail(const char* what);
    /* The device kernel `h` resolved to on this rank ("?" if it resolved none). For log
     * attribution only. */
    const char* ip_kernel_name(rad_op h) const;
private:
    [[noreturn]] void fatal_issue(rad_op h, int64_t n, const char* why) const;

    /* ---- the op oracle (runtime/oracle.cpp), inert unless RADIANCE_OP_ORACLE names an op. Runs
     * the host kernel for the same (op, band) on a host copy of the device kernel's operands and
     * compares. `oracle_` is null otherwise, so the hot path pays one not-taken branch. */
    struct OracleState;
    OracleState* oracle_ = nullptr;
    void  oracle_open();
    void  oracle_close();
    bool  oracle_arm(rad_op h) const;
    /* Extents, aliasing and per-operand byte counts for one issue. Separate from the copy because
     * a weight-table operand's snapshot is the stacked plane behind its pointers, a size the
     * operand does not state. */
    void  oracle_size(rad_op h, int n_opd);
    int64_t oracle_opd_bytes(int i) const;
    void  oracle_snap(int n_opd, bool post);
    void  oracle_run(rad_op h, const IssuePlan& hip, int64_t seq, int n_opd);
    /* Converts a re-laid weight back, through the device row's inverse, into the logical form the
     * host row reads. False (a named skip) when no valid comparison is possible. */
    bool  oracle_unrelayout(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int64_t seq,
                          int n_opd);
    /* Repeat mode (RADIANCE_OP_REPEAT): re-launches the device plan on the restored inputs and
     * byte-compares the two results, finding kernels that are not a function of their inputs,
     * which a host-vs-device comparison cannot detect. */
    void  oracle_repeat(rad_op h, const IssuePlan& dip, int64_t seq, int n_opd);
    /* The reference library's host plan for (op, band): the band's own host plan when a reference
     * library won it, else one resolved from `registry_` on first use. Null when none exists. */
    const IssuePlan* oracle_reference(rad_op h, int band);
    const IssuePlan* oracle_before(rad_op h, int band, int n_opd);
    void  oracle_after(rad_op h, const IssuePlan& hip, const IssuePlan& dip, int n_opd);

    Program*        program_   = nullptr;
    Registry*       registry_  = nullptr;   /* CtxDesc::registry */
    std::vector<KVPoolBinding> kv_;   /* by rad_kvgroup; RAD_OPK_KV resolves through it */
    ResidencyTable* residency_ = nullptr;
    OpStager*       stager_ = nullptr;
    bool            staging_ = false;     /* the stager armed for the pass being issued */
    int issue_body(rad_op h, const RadOperand* opd, int n_opd, int64_t n);
    ArchStepFn      arch_step_ = nullptr;

    /* The expert prefetch's state (xp_state.h), null while it is off. */
    std::unique_ptr<XpState> xp_;
    void xp_free(XpState& x);
    void xp_serve(XpState& x);
    void xp_judge(XpState& x);
    bool xp_bind(XpLayer& L, int layer);
    int  xp_predict_run(const RadPrefetchNext& next, rad_buf h, int64_t r0, int64_t rows, int64_t n_embd);
    int  xp_mark_run();
    int  xp_step_run(int layer);

    int       rank_       = 0;
    int       world_size_ = 1;
    int       device_     = 0;
    RadStream stream_     = nullptr;
    bool      own_stream_ = false;

    /* ---- The second lane: overlaps the tensor-parallel all-reduce with compute at prefill
     * (order of 15% prefill throughput). Any use must respect these constraints:
     *
     * 1. Every collective rides the same lane. The all-reduce double-buffers its scratch on a
     *    per-block `seq & 1`, so collectives that race differently on two ranks pick different
     *    slots for the same message and deadlock or corrupt.
     *
     * 2. A device-wide arrival counter must be per stream. `split_ctr` in r4d_gemm_fp8a8.hip (the
     *    self-resetting split-K finisher counter) is sound only because its launches are ordered
     *    by one stream, so there is one counter block per stream. Any other plugin kernel with a
     *    device-wide counter must do the same or stay off lane 1; nothing here can check it.
     *
     * 3. The buffer planner packs from one linear order, so two lanes can alias arena bytes
     *    through buffers with disjoint declared lifetimes. The architecture passes every buffer
     *    its second lane touches to `rad_buf_concurrent`, which makes it live for the whole
     *    program.
     *
     * 4. A slice must land in the same bucket-table band as the whole, or it picks a kernel with
     *    a different summation order and silently changes output. `--debug-graph` prints a model's
     *    band boundaries; a representative set is
     *
     *        8   16   56   64   256   1024   4096   8192   16384   49152   65536
     *
     *    Halving a 4096-token chunk gives 2048 + 2048, inside (1024, 4096]; quartering gives
     *    1024, the top of the band below. A ragged last chunk of 65 halves across 64 and 56. Every
     *    slice must sit strictly above the largest boundary below T, which at 4096 means at most
     *    two slices. The architecture's own T-dependent selections hold at two slices:
     *    `ar_wire_is_exact` flips only below T = 13, and the gated-GEMM fold and fused attention
     *    prologue band at T <= 64.
     *
     * The overlap works because a many-block spinning kernel on one stream does not starve a
     * bandwidth-bound kernel on another: the all-reduce's min(M, 256) blocks spin on a peer flag,
     * and a spinner plus a large copy on two streams cost the max of the two, not the sum. */
    RadStream lane1_      = nullptr;
    bool      own_lane1_  = false;
    int       lane_       = 0;
    RadEvent  lane_ev_[2] = { nullptr, nullptr };
    /* The last device op on each lane that touched host-domain memory. A host op runs at issue on
     * this thread and must be ordered behind every earlier device op that touched the host arena;
     * on each lane those complete no later than the last one. So a device op with a host-domain
     * operand re-records its lane's event, and a host op waits on those events instead of draining
     * the streams (see the host join in Ctx::issue). */
    RadEvent  host_touch_ev_[2]      = { nullptr, nullptr };
    bool      host_touch_pending_[2] = { false, false };
    int       host_join(const RadOperand* opd, int n_opd);
    bool      host_join_drains(const RadOperand* opd, int n_opd) const;
    int       host_join_wait(bool drain);
    /* A lane issued a scratch user and the other lane has not been joined behind it since. Keeps
     * two lanes off the shared region at once; lane_join clears its `from` side. See the scratch
     * block in Ctx::issue. */
    bool      scratch_open_[2]    = { false, false };
    bool      scratch_join_warned_ = false;
    /* Where in its region the next scratch user starts. Consecutive users get consecutive slices,
     * wrapping when one does not fit, so independent users share no bytes and may overlap
     * (haz_mark). Reset every pass, so a re-issued pass gets the same slices as its recording. */
    int64_t   scratch_cur_[2]     = { 0, 0 };
    void*     scratch_take(int region, int64_t bytes);
    /* The kernel that last opened each lane's scratch, and its row count, for the warning. */
    const char* scratch_kernel_[2] = { nullptr, nullptr };
    int64_t     scratch_n_[2]      = { 0, 0 };

    Arena arena_;        /* device: the buffer plan plus the shared scratch region */
    Arena host_arena_;   /* only if the program declares RAD_DOMAIN_HOST buffers */

    const RadBatch* batch_ = nullptr;

    /* Per-issue tensor storage, bound into every IssuePlan::args.t at prepare. One array, because
     * issues do not nest. */
    RadTensor tbuf_[MAX_OPERANDS];

    /* Prebuilt tensor templates by handle: a buffer's carries its arena pointer, dtype, shape and
     * packed strides; a weight's carries WeightInfo::ptr. An issue copies one and patches at most
     * the data pointer and dim 0. */
    std::vector<RadTensor> buf_tmpl_;
    std::vector<int32_t>   buf_bits_;
    std::vector<uint8_t>   buf_host_;   /* per handle: a RAD_DOMAIN_HOST buffer */
    /* Every level's templates; `buf_tmpl_` is a copy of the active one. */
    std::vector<std::vector<RadTensor>> level_tmpl_;
    int                    level_ = 0;
    std::vector<RadTensor> w_tmpl_;
    std::vector<int32_t>   w_bits_;
    /* The generation each lane's stream was last made to wait on, a weight. Per lane: the wait
     * orders only the stream it is put on, so a weight resolved on lane 1 (the Gemma 4 26B-A4B's
     * dense arm, which a decode step issues on lane 1 and an overlapped prefill on lane 0) must
     * still be waited on when lane 0 resolves it. Indexed by lane_, like the other per-lane state
     * here: set_lane admits lanes 0 and 1 only. */
    std::vector<uint32_t>  waited_gen_[2];

    std::vector<OpPlan>                 plans_;
    /* Ops not yet seen issued. flush_issued walks only these and drops each once it fires. */
    std::vector<uint32_t>               unflushed_;
    std::vector<int64_t>                band_table_;
    std::vector<IssuePlan>              issue_table_;
    /* Parameter lists (geometry plus `link_gate`) for the launches that take the gate. A deque,
     * so IssuePlan::gate_p stays valid as entries are added. */
    std::deque<std::vector<RadParam>>   gate_params_;
    std::vector<std::atomic<uint8_t>>   op_domain_;   /* sized once, never resized */

    std::vector<RouteSlot> route_;
    int64_t                route_n_expert_ = 0;
    /* A bank is one step's histograms for every layer: [kRouteBanks][n_layer][n_expert] in pinned
     * host memory. A report copies a layer's counts into its row of the card-side
     * [n_layer][n_expert] (an on-card copy that orders nothing else); the end of the pass copies
     * the touched rows into the step's bank in one transfer and records the bank's event. See
     * route_report. */
    int32_t*               route_pool_     = nullptr;   /* host */
    int32_t*               route_dev_      = nullptr;   /* card */
    RadEvent               route_done_[kRouteBanks] = {};
    /* The layers this pass reported into `route_bank_`, as [lo, hi]; lo > hi when none. */
    int                    route_bank_ = -1, route_lo_ = 1, route_hi_ = 0;
    int                    route_flush();

    /* Profiling. Off by default; when on it adds synchronisations that change the mover's slot
     * supply, so profiled and unprofiled runs are not comparable (spec §16). */
    bool                 profiling_ = false;
    struct ProfSlot { RadEvent a = nullptr, b = nullptr; int32_t op = 0; uint8_t lane = 0; };
    std::vector<ProfSlot> prof_ring_;
    size_t                prof_n_ = 0;
    bool                  prof_overflowed_ = false;
    std::vector<double>   prof_ms_;
    std::vector<int64_t>  prof_calls_;
    /* Rows timed on the host clock rather than by a device event pair. A host row includes page
     * faults and link reads, a device row is kernel time, so the table labels each. */
    std::vector<uint8_t>  prof_host_;
    /* The stream drain a host-site op forces, timed apart from the op: it is the cost of stopping
     * the pipeline, which can exceed the op itself by 100x (PLE gather). */
    std::vector<double>   prof_join_ms_;
    /* Lane idle time before each op: previous op's end on that lane to this one's start. Stream
     * waits (staged copies, the other lane, a host op) and late host issue land here, accounting
     * for wall time the ops' own times do not cover. */
    std::vector<double>   prof_wait_ms_;

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
