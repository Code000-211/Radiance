/* mover.h -- residency and the mover: where every weight unit lives and how it moves.
 *
 * Three storage tiers, one address space of slots:
 *
 *   VRAM slab: fixed-stride slots per weight class. Slots within a class are interchangeable, so
 *   a promotion is an integer swap rather than an allocation, and the slab never fragments.
 *
 *   Pinned host pool, same stride. The card reads a non-resident expert from it zero-copy
 *   inside the GEMM. It is owned here, not read through the container's mmap, because under
 *   memory pressure the kernel's paging makes decode reads several times slower, and
 *   MADV_WILLNEED makes it worse.
 *
 *   File tier: O_DIRECT reads into pinned buffers, with predictable latency and no page-cache
 *   copy. No buffered fallback: a tier whose latency depends on the page cache cannot be budgeted.
 *
 * ---- Transfers never synchronise the host ----------------------------------------------------
 *
 * The mover runs on its own stream and the compute stream waits on its events (rad_event_wait,
 * never rad_event_sync), so placement changes without serialising the step. The lead gate is a
 * host wait on how far the host runs ahead, not on a transfer (MoverConfig::max_lead_dispatches).
 *
 * ---- Shadow slots and quarantine -------------------------------------------------------------
 *
 * A promotion copies into a shadow slot (allocated, holding nothing) and only then publishes the
 * new pointer, so a slot is never overwritten while a kernel may read it. Getting this order
 * wrong does not crash; it produces fluent output that differs from run to run.
 *
 * A freed slot goes into quarantine, because kernels issued in earlier layers still name it. The
 * barrier is an event recorded on the compute stream after the publish; when it signals, every
 * launch that could name the slot has retired.
 */
#pragma once
#include "../device/device.h"
#include "../device/directio.h"
#include "../mem/pools.h"
#include "heat.h"
#include "planner.h"
#include "residency_iface.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ slot bookkeeping */
/* Free-list bookkeeping for the VRAM slab and the pinned host pool. Pure host state, testable
 * without a card.
 *
 * Slots are numbered in segments laid end to end. Segment 0 is the budgeted range; each later
 * one is memory another pool lends (KV cache blocks, the arena's top). Each segment has its own
 * target, below which slots may be handed out, because lenders reclaim independently.
 *
 * The lowest free slot goes first, so occupied slots pack toward a segment's bottom and its top
 * can be returned; a LIFO would keep handing out high slots. `live` is how far a segment's slots
 * exist and `target` how far they should: a shrink lowers `live` only past unheld slots, so it
 * never evicts.
 *
 * The free list is rebuilt from the held flags whenever a target moves; one carried across a
 * shrink and regrow would lose slots freed above the target. take() and add_free() stay heap
 * operations.
 */
class SlotPool {
public:
    /* Segment 0 starts with `n_live` slots, all held (the plan's resident set) until add_free()
     * releases them. `n_cap` is how far segment 0 may grow into address space the caller has
     * reserved and backs on demand; the default is no growth. */
    void init(int64_t n_live, int64_t stride, int64_t n_cap = -1) {
        stride_ = stride;
        n_ = n_cap > n_live ? n_cap : n_live;
        segs_.assign(1, Seg{ 0, n_, n_live > 0 ? n_live : 0, n_live > 0 ? n_live : 0 });
        free_.clear();
        held_.assign((size_t)(n_ > 0 ? n_ : 0), 1);
        n_usable_ = 0;
        n_held_ = n_live > 0 ? n_live : 0;
    }
    /* A segment of `n_slots` after every existing one, covering nothing and holding nothing until
     * a target reaches into it. Returns its index; slot numbers in it start at seg_lo(index). */
    int add_segment(int64_t n_slots) {
        if (n_slots < 0) n_slots = 0;
        segs_.push_back(Seg{ n_, n_ + n_slots, n_, n_ });
        held_.resize((size_t)(n_ + n_slots), 1);
        n_ += n_slots;
        return (int)segs_.size() - 1;
    }
    int64_t seg_lo(int k) const { return segs_[(size_t)k].lo; }

    void add_free(int64_t slot) {
        if (slot < 0 || slot >= n_) return;
        if (!held_[(size_t)slot]) return;
        held_[(size_t)slot] = 0;
        --n_held_;
        Seg& g = segs_[(size_t)seg_of(slot)];
        if (slot < g.target) push(slot);
        else settle(g);
    }
    bool empty() const { return n_usable_ == 0; }
    /* How many slots exist, over every segment. */
    int64_t size() const {
        int64_t n = 0;
        for (const Seg& g : segs_) n += g.live - g.lo;
        return n;
    }
    int64_t capacity() const { return n_; }
    int64_t stride() const { return stride_; }
    int64_t n_free() const { return n_usable_; }
    /* Slots in use: a resident unit, a staged one, or a copy in flight. Fill and drain decide
     * against this, because the working slots must stay free whatever the target is. */
    int64_t held() const { return n_held_; }
    /* Where segment `k`'s covered slots end, as a slot number. */
    int64_t target(int k = 0) const { return segs_[(size_t)k].target; }
    /* Whether `slot` is below its segment's target, i.e. may be handed out. */
    bool covered(int64_t slot) const {
        if (slot < 0 || slot >= n_) return false;
        return slot < segs_[(size_t)seg_of(slot)].target;
    }
    /* The lowest free covered slot, or -1. Starvation is the normal steady state: callers retry
     * on a later dispatch rather than treat -1 as an error. */
    int64_t take() { return take_at_least(0); }

    /* Sets where segment `k`'s covered slots end, as a slot number inside it. Growing is
     * immediate (the caller has already backed the memory). Shrinking lowers `live` only past
     * slots nothing holds; the rest follows as those are released. */
    void set_target(int64_t target) { set_target(0, target); }
    void set_target(int k, int64_t target) {
        Seg& g = segs_[(size_t)k];
        g.target = target < g.lo ? g.lo : (target > g.hi ? g.hi : target);
        /* Nothing holds a slot at or past `live` (take() hands out only covered slots and `live`
         * never falls past a held one), so a set flag there only means the slot never existed. */
        for (int64_t s = g.live; s < g.target; ++s) held_[(size_t)s] = 0;
        if (g.target > g.live) g.live = g.target;
        else settle(g);
        rebuild();
    }

    /* take(), restricted to slots >= `lo`. Free slots below `lo` stay listed; there are at most
     * the working slots of them, so this is a few heap operations. */
    int64_t take_at_least(int64_t lo) {
        std::vector<int64_t> keep;
        int64_t got = -1;
        while (!free_.empty()) {
            std::pop_heap(free_.begin(), free_.end(), std::greater<int64_t>());
            const int64_t s = free_.back();
            free_.pop_back();
            if (!covered(s) || held_[(size_t)s]) continue;   /* never a slot to hand out */
            if (s < lo) { keep.push_back(s); continue; }
            held_[(size_t)s] = 1;
            ++n_held_;
            --n_usable_;
            got = s;
            break;
        }
        for (int64_t s : keep) {
            free_.push_back(s);
            std::push_heap(free_.begin(), free_.end(), std::greater<int64_t>());
        }
        return got;
    }
    /* take(), restricted to the lowest free slot `want` accepts, or -1. A linear scan; the free
     * list holds only slots passing through a swap, a few dozen. */
    template <class Want> int64_t take_where(Want want) {
        size_t best = free_.size();
        for (size_t i = 0; i < free_.size(); ++i) {
            const int64_t s = free_[i];
            if (!covered(s) || held_[(size_t)s] || !want(s)) continue;
            if (best == free_.size() || s < free_[best]) best = i;
        }
        if (best == free_.size()) return -1;
        const int64_t s = free_[best];
        free_[best] = free_.back();
        free_.pop_back();
        std::make_heap(free_.begin(), free_.end(), std::greater<int64_t>());
        held_[(size_t)s] = 1;
        ++n_held_;
        --n_usable_;
        return s;
    }
    /* The highest held slot of segment `k` (a unit, a copy in flight, a quarantined slot), or
     * -1. Scans down from `live`. */
    int64_t top_held(int k = 0) const {
        const Seg& g = segs_[(size_t)k];
        for (int64_t s = g.live; s-- > g.lo;)
            if (held_[(size_t)s]) return s;
        return -1;
    }
    /* How many existing slots at or past `from` something holds, over every segment. */
    int64_t held_from(int64_t from) const {
        int64_t k = 0;
        for (const Seg& g : segs_)
            for (int64_t s = std::max(g.lo, from); s < g.live; ++s) k += held_[(size_t)s];
        return k;
    }

private:
    struct Seg { int64_t lo, hi, target, live; };
    int seg_of(int64_t slot) const {
        for (size_t k = segs_.size(); k-- > 1;)
            if (slot >= segs_[k].lo) return (int)k;
        return 0;
    }
    void push(int64_t s) {
        free_.push_back(s);
        std::push_heap(free_.begin(), free_.end(), std::greater<int64_t>());
        ++n_usable_;
    }
    void settle(Seg& g) { while (g.live > g.target && !held_[(size_t)(g.live - 1)]) --g.live; }
    /* Every covered slot that nothing holds, once each. Ascending order is already a min-heap. */
    void rebuild() {
        free_.clear();
        for (const Seg& g : segs_)
            for (int64_t s = g.lo; s < g.target; ++s)
                if (!held_[(size_t)s]) free_.push_back(s);
        n_usable_ = (int64_t)free_.size();
    }

    int64_t stride_ = 0, n_ = 0, n_usable_ = 0, n_held_ = 0;
    std::vector<Seg> segs_;
    std::vector<int64_t> free_;
    std::vector<uint8_t> held_;
};

/* ------------------------------------------------------------------ the file tier */
/* A thin wrapper over the device layer's DirectFile, which owns the O_DIRECT alignment rules
 * (the prefix cache reads the same way). This adds the tier's accounting: a file-tier read is a
 * blocking pread inside a dispatch, and the counters make its cost visible. */
class FileTier {
public:
    /* Opens the container `path` with O_DIRECT; refuses rather than fall back to buffered reads,
     * whose latency cannot be budgeted. */
    int  open(const std::string& path);
    bool ok() const { return f_.is_open(); }
    void close() { f_.close(); }

    /* One expert-sized read into a pinned buffer. Blocking and counted: a dispatch that meets a
     * file-tier unit stalls until the read returns. This is the cost the plan accepted by putting
     * the cold tail on disk; frequent reads call for a different plan, not a change here. */
    int  read(uint64_t offset, int64_t bytes, void* pinned_dst, int64_t dst_cap);

    /* RadFileEntry offsets are RAD_ALIGN_UNIT-aligned (rad_format.h), which is DirectFile's
     * alignment, so the mover's reads need no bounce buffer. */
    static int64_t direct_align(int64_t v) { return align_up(v, DirectFile::alignment()); }

private:
    DirectFile f_;
};

/* ------------------------------------------------------------------ residency */
class Residency final : public ResidencyTable {
public:
    void init(const Program& prog);

    const WeightSlot* lookup(rad_weight w) const override {
        return (size_t)w < slot_.size() ? &slot_[w] : nullptr;
    }
    uint64_t epoch() const override { return epoch_; }
    const std::vector<rad_weight>& changes() const override { return changed_; }
    void clear_changes() override {
        for (rad_weight w : changed_) noted_[w] = 0;
        changed_.clear();
    }

    /* The only two writers; both bump the generation. publish() settles a slot; begin_transfer()
     * marks a copy into `ptr` in flight that the compute stream must wait on (`ready`). The run
     * phase memoises its wait per generation, so a new event at an old generation would be
     * skipped (residency_iface.h). */
    void publish(rad_weight w, void* ptr) { begin_transfer(w, ptr, nullptr); }
    void begin_transfer(rad_weight w, void* ptr, RadEvent ready);
    void settle(rad_weight w) {
        if ((size_t)w >= slot_.size()) return;
        slot_[w].ready = nullptr;
        ++epoch_;
    }

private:
    void note(rad_weight w) {
        if (noted_[w]) return;
        noted_[w] = 1;
        changed_.push_back(w);
    }

    std::vector<WeightSlot> slot_;
    uint64_t                epoch_ = 1;
    std::vector<rad_weight> changed_;
    std::vector<uint8_t>    noted_;
};

/* ------------------------------------------------------------------ configuration */
struct MoverConfig {
    /* Shadow and staging slot counts come from the plan (SlabClass::n_shadow, n_stage), which
     * budgets them; a second knob here could only cause an overrun.
     *
     * How far the host may run ahead of the cards, in layer dispatches. A freed slot recycles
     * only when its quarantine event on the compute stream signals, so the move rate is shadow
     * slots / compute queue depth; 0 (unbounded) queues a whole step and starves the mover. 2 is
     * the smallest depth that keeps slots recycling with a dispatch of issue margin. A per-step
     * gate does nothing, since one step is already the whole model's queued work. */
    int max_lead_dispatches = 2;

    /* Lent slots per slab class kept free for swaps to land in (Mover::balance). A knob rather
     * than a plan field because the planner charges nothing for them: they are KV cache memory,
     * free only while it lends. A class whose shadow slots the quarter-of-the-slab cap reduced
     * gets proportionally fewer, so a small loan is not left emptier than it is filled. */
    int turnover_slots = RAD_PLACE_TURNOVER_SLOTS;

    /* The container, for the file tier. Empty means no file tier; a unit planned there is then
     * an error. */
    std::string container_path;
    /* Pinned staging buffers for file-tier reads, one class stride each; the count the planner
     * charged to --host-pool-mib. */
    int stage_slots = RAD_PLACE_FILE_STAGE_SLOTS;
};

/* ------------------------------------------------------------------ the loan */
/* One stretch of another pool's VRAM that may be lent to the slab. The owner keeps it backed,
 * lends from the top down and states the loan in its own measure (tokens, for the KV cache). A
 * loan of T tokens is the top floor(T / unit_tokens) * unit_bytes bytes of every stretch. */
struct LoanStrip {
    char*   top = nullptr;       /* one past the stretch's highest byte */
    int64_t bytes = 0;           /* how far below `top` it runs */
    int64_t unit_bytes = 0;
    int64_t unit_tokens = 0;
};

struct MoverStats {
    uint64_t dispatches = 0;
    uint64_t promotions = 0, demotions = 0;
    uint64_t h2d_bytes = 0, d2h_bytes = 0;
    uint64_t prefetched = 0, prefetch_bytes = 0;
    /* Why a move did not go out: every move record busy, or a free list empty. These separate a
     * starved mover from a satisfied one. */
    uint64_t no_record = 0, starved_slab = 0, starved_pool = 0;
    uint64_t quarantined = 0;
    /* Demotions out of a lent slot that copied nothing (the unit kept its host copy). Also
     * counted in `demotions`. */
    uint64_t dropped = 0;
    /* Host copies given to budgeted victims (Mover::donor_slot); donors moved rather than
     * dropped by a recall or shrink; recalls that found no free slot for a donor. */
    uint64_t donated = 0, relocated = 0, stranded = 0;
    uint64_t ssd_reads = 0, ssd_bytes = 0;
    /* The prefill stager (stage_units). `stage_overflow`: units no buffer had room for, read in
     * place. `stage_copies`: copies issued; neighbouring pool slots go as one. */
    uint64_t stage_layers = 0, staged_units = 0, staged_bytes = 0, stage_overflow = 0;
    uint64_t stage_copies = 0;
    /* The file stager (stage_file_units), including the host time spent in preads and waiting
     * for a window's copies to drain. */
    uint64_t file_layers = 0, file_units = 0, file_bytes = 0, file_preads = 0;
    uint64_t file_read_ns = 0, file_wait_ns = 0;
};

/* ------------------------------------------------------------------ the mover */
class Mover {
public:
    ~Mover();
    Mover() = default;
    Mover(const Mover&) = delete;
    Mover& operator=(const Mover&) = delete;

    /* Allocates every pool the plan asks for and assigns every unit a slot; the load phase fills
     * the published addresses. On RAD_E_NOMEM the log names the pool.
     *
     * Always pass `budget`: Pools already reserved the weights pool, so allocating beside it
     * spends the budget twice, and reserving from Pools::weights() keeps promotions out of KV
     * blocks (mem/pools.h). Null is the standalone-test path and logs a warning. */
    int init(const Program& prog, const Plan& plan, const MoverConfig& cfg, Pools* budget = nullptr);

    /* The compute stream this rank issues on. Transfers are ordered against it and the
     * quarantine barrier is recorded on it; without one the mover refuses to move. */
    int bind_compute_stream(RadStream s);

    /* The mover's stream, to be shared by any subsystem moving bytes off the critical path. Each
     * HIP stream costs a hardware queue, and on RDNA4 a fourth one is expensive: on two R9700s
     * serving Flash-Next, an idle fourth stream takes the decode step from 20.5 to 33.9 ms
     * (GPU_MAX_HW_QUEUES 1-3 are fine, 4 and 8 are not). The engine already uses three (compute,
     * the second compute lane, this one).
     *
     * Null before init() and on a rank with nothing to move; the caller then uses its own stream.
     * The run phase issues here too, so a sharer's large copy delays transfers the compute stream
     * waits on; gate the two rather than add a queue. */
    RadStream transfer_stream() const { return mover_; }

    ResidencyTable* residency() { return &res_; }
    /* For Tier::Mapped weights, which live in the container mapping the load phase holds: the
     * one address the mover does not own, recorded so the residency table stays the only record
     * of weight addresses. A mapped unit has no slab class, so nothing republishes over it. */
    void publish_mapped(rad_weight w, void* p) { res_.publish(w, p); }

    /* ---- the run phase, in the order a layer dispatch calls it -----------------------------
     *
     * begin_dispatch  retire what landed, gate the host's lead, record the ordering event
     * prefetch_to_op  release what this op has finished with, then start every transfer the
     *                 exact schedule says is due, while slots are free
     * issue           one heat proposal
     *
     * There is deliberately no end_dispatch: the quarantine barrier must be recorded downstream
     * of the previous dispatch's launches, which begin_dispatch's retire() is.
     *
     * A caller must issue as well as retire every dispatch. The pool free list starts empty, so a
     * loop that only retires promotes into the shadow slots once and never demotes; on the host
     * backend copies land within their dispatch, so this shows only on a real card. The compute
     * stream must also carry the dispatch's real work, or the quarantine barrier and the lead
     * gate are meaningless.
     *
     * In production, begin_dispatch and issue run once a step from Engine::tiering_dispatch,
     * since the mover relocates only between steps (Ctx::run_step syncs the residency table at
     * its top). prefetch_to_op is called only by tests/place_test.cpp; the
     * exact-schedule path is kept built and tested deliberately.
     */
    int begin_dispatch(int dispatch, HeatEngine* heat);
    int prefetch_to_op(int32_t op_index);
    int issue(const Swap& s, HeatEngine* heat);

    /* ---- the loan: VRAM another pool is not using ------------------------------------------
     *
     * An expert that does not fit the slab is read across the link on every step that routes to
     * it, so unused VRAM elsewhere is lent to the slab: the part of the KV cache's carve nothing
     * holds, and the part of the activation arena that steps smaller than the largest do not
     * reach. Both are addresses the lender already owns, so no allocation or mapping is involved.
     * Only the caller that arbitrates the whole card may lend.
     *
     * Each lender lends in its own measure (KV cache: tokens; arena: bytes) into its own slab
     * segment, and they move independently, so every call names its lender.
     *
     * A unit in a lent slot keeps its host copy (weights never change), so giving the slot back
     * is a change of address: a recall waits only for kernels to stop naming the slot. */

    enum Lender { kLendKV = 0, kLendArena = 1, kLenders = 2 };

    /* Registers the stretches a lender may ever lend; once per lender, after its memory is
     * carved. Each stretch is cut into slots of each class's stride from the top down, so a small
     * loan covers a prefix of a large loan's slots and the deepest slot is given back first. */
    int     lend_from(int lender, const std::vector<LoanStrip>& strips);

    /* How much a lender lends, in its own measure. A target, not a transaction: newly covered
     * slots can be filled at once, while uncovered ones empty as their units drop and their
     * quarantine clears. loan_held() is how far into the lender's memory the slab still reaches;
     * the lender may not reuse those bytes until it has fallen to the loan. */
    int     set_loan(int lender, int64_t amount);
    int64_t loan_held(int lender) const;
    int64_t loan(int lender) const { return loan_[lender]; }

    /* set_loan(), completed before returning: units past `amount` are dropped to their host
     * copies and both streams drained. A host wait, for when the lender needs more than it kept
     * in hand (a request larger than the cache's buffer, a step larger than the arena allows). */
    int     reclaim(int lender, int64_t amount, HeatEngine* heat);

    /* Bytes the lent slots hold, bytes the loans offer, and bytes every carved slot could hold.
     * The first two are stored because the HTTP thread reads them for the dashboard and only the
     * rank thread may walk the slot pools. */
    int64_t flex_used_bytes() const { return flex_used_.load(std::memory_order_relaxed); }
    int64_t flex_offered_bytes() const { return flex_offered_.load(std::memory_order_relaxed); }
    int64_t flex_capacity() const;
    /* The lent slots the fill leaves free for the engine to swap into
     * (MoverConfig::turnover_slots), summed over the slab classes. */
    int64_t turnover_slots() const {
        int64_t n = 0;
        for (const ClassPools& cp : pools_) n += cp.n_turn;
        return n;
    }
    /* The loan that covers every slot carved from this lender; more would sit empty. Zero when
     * nothing is carved. */
    int64_t loan_ceiling(int lender) const;

    /* Fills free lent slots and empties slots the loan no longer covers: the half-swaps the heat
     * engine never proposes, since its decisions need a victim. Every free slot past the turnover
     * slots is filled in one call, hottest units first, since an empty slot wastes VRAM every
     * step. The turnover slots stay free for the engine's swaps; filled, they would cap the move
     * rate at the shadow slots. When fewer than the working plus turnover slots are free, the
     * coldest lent units are dropped (an address change, never a copy).
     *
     * `turnover` false fills the turnover slots too, for steps bound by their own staging copies
     * (Engine::tiering_dispatch); the next call with true drops them back out. */
    int balance(HeatEngine* heat, bool turnover = true);

    /* retire() then balance() for an idle rank, so a loan made while idle is filled before the
     * next request. Call only while the rank thread is parked. */
    int idle_tick(HeatEngine* heat);
    /* Whether idle ticks keep the turnover slots filled; the engine holds them through a prompt
     * and the few steps after it (Engine::tiering_dispatch). */
    void hold_turnover(bool held) { hold_turnover_ = held; }
    /* Each class's pinned pool as the card addresses it: [lo, hi) and the unit stride. At most `max`. */
    int pool_ranges(uint64_t* lo, uint64_t* hi, uint64_t* stride, int max) const {
        int n = 0;
        for (const ClassPools& cp : pools_)
            if (cp.pool_dev && cp.stride > 0 && n < max) {
                lo[n] = (uint64_t)(uintptr_t)cp.pool_dev;
                hi[n] = lo[n] + (uint64_t)cp.pool.size() * (uint64_t)cp.stride;
                stride[n] = (uint64_t)cp.stride;
                ++n;
            }
        return n;
    }

    /* Where a unit's bytes are right now, as a device-visible address, or null for a file-tier
     * unit that has not been staged. */
    void* unit_base(int32_t unit) const;

    /* ---- staging within a pass: the prefill stager (core/place/stager.h) ----------------------
     *
     * The only calls that change a unit's address within a step, and only the published one: the
     * unit stays in its pinned pool slot, and one routed layer's ops read a copy in a
     * caller-owned buffer.
     *
     * stage_units   copies each pooled unit of `units` with no slab slot and no copy in flight
     *               into `dst`, packed at the allocation unit; a unit with no room is read in
     *               place. Runs on the mover's stream behind `after` (if set) and records `done`.
     *               Copies are published settled: the caller waits on `done` once instead of
     *               once per weight (a few hundred per layer).
     * unstage_units publishes those units at their pool slots again. The caller must already have
     *               ordered the next writer of `dst` behind every reader of the copies.
     * stage_need    the room stage_units would use with no cap, for sizing shares of a region;
     *               `largest`, if set, gets the largest unit's room.
     *
     * The span form packs into up to two stretches (a ring region that wraps is two).
     */
    struct StageSpan { char* dst; int64_t cap; };
    int  stage_units(const std::vector<int32_t>& units, char* dst, int64_t cap, RadEvent after,
                     RadEvent done, std::vector<int32_t>* staged);
    int  stage_units(const std::vector<int32_t>& units, const StageSpan* spans, int n_spans,
                     RadEvent after, RadEvent done, std::vector<int32_t>* staged);
    void unstage_units(const std::vector<int32_t>& staged);
    int64_t stage_need(const std::vector<int32_t>& units, int64_t* largest = nullptr) const;

    /* ---- routed layers read from the container: the file stager (core/place/stager.h) --------
     *
     * A file-tier unit has no address, but a grouped GEMM takes every expert's pointer at issue.
     * So before a routed layer's first expert op its file-tier units are read into a VRAM buffer
     * and published there; afterwards unstage_units publishes them as null again. The plan
     * charges the two buffers to the weights pool (Plan::file_stage_vram).
     *
     * stage_file_units  reads every file-tier unit of `units` into `dst`, packed at RAD_ALIGN_UNIT
     *                   in the order given, through the pinned read windows: container-order
     *                   runs, each window draining to `dst` on the mover's stream while the next
     *                   fills, behind `after` (if set), with `done` recorded last. Published
     *                   settled. RAD_E_SCRATCH if the units exceed `cap`. The reads block this
     *                   thread.
     * file_buffer       buffer 0 or 1 of the plan's two, or null when the plan has none. */
    int   stage_file_units(const std::vector<int32_t>& units, char* dst, int64_t cap,
                           RadEvent after, RadEvent done, std::vector<int32_t>* staged);
    char* file_buffer(int i) const {
        return (file_vram_ && (i == 0 || i == 1)) ? file_vram_ + i * file_buf_bytes_ : nullptr;
    }
    int64_t file_buffer_bytes() const { return file_vram_ ? file_buf_bytes_ : 0; }

    const MoverStats& stats() const { return stats_; }
    std::string report() const;

private:
    /* One in-flight relocation: Free -> Copying -> Published -> Quarantine -> Free. Transfers
     * are asynchronous; every transition is made by the dispatch thread at a layer boundary. No
     * mutex on the read side (it is on every layer's critical path) and no background thread,
     * which would make placement depend on host timing. */
    struct Move {
        enum class State { Free, Copying, Published, Quarantine };
        /* Promote/Demote: the heat engine's swaps. Stage: the exact prefetch copying an offloaded
         * unit into a slot; Release: that slot returned after its last use this step. Drop: a
         * unit in a lent slot back to its kept host copy. Release and Drop copy nothing but still
         * need the quarantine, which guards issued kernels, not bytes in flight. Migrate: a donor
         * (Mover::donor_slot) from an uncovered lent slot (`slab_slot`) to a free budgeted one
         * (`to_slot`). */
        enum class Kind { Promote, Demote, Stage, Release, Drop, Migrate };
        State    state = State::Free;
        Kind     kind = Kind::Promote;
        int32_t  unit = -1;
        int64_t  slab_slot = -1;
        int64_t  pool_slot = -1;
        int64_t  to_slot = -1;
        RadEvent ev = nullptr;
        /* A promotion the fill issued into a free slot, not a swap's decision. */
        bool     fill = false;

        void clear() {
            state = State::Free;
            unit = -1;
            slab_slot = pool_slot = to_slot = -1;
            fill = false;
        }
        /* Whether the move hands `slab_slot` back once its quarantine clears. */
        bool frees_slab() const { return kind != Kind::Promote && kind != Kind::Stage; }
    };

    struct UnitState {
        Tier    tier = Tier::VRAM;
        Site    site = Site::Device;
        int64_t slab_slot = -1;      /* index into class slab, or -1 */
        int64_t pool_slot = -1;      /* index into class host pool, or -1 */
        int64_t static_off = -1;     /* byte offset into the static VRAM region, or -1 */
        int32_t klass = -1;
        int64_t bytes = 0;
    };

    /* One weight class's memory. Budgeted slots sit at a fixed stride from one base in the
     * weights pool. Flex slots after them are lent addresses scattered over another pool's memory
     * (KV cache stripes, arena top), so only slot_ptr() can resolve a slot index. Each lender's
     * flex slots form one segment ordered by covering loan, so any loan covers a prefix. */
    struct ClassPools {
        SlotPool slab, pool;
        void*    slab_base = nullptr;
        void*    pool_base = nullptr;
        void*    pool_dev = nullptr;   /* the pinned pool as the device addresses it */
        int64_t  stride = 0;
        int64_t  n_fixed = 0;          /* slots in the budgeted range; the flex ones start here */
        int64_t  n_work = 0;           /* shadow + staging: free slots the mover needs to move at all */
        int64_t  n_turn = 0;           /* lent slots kept free past those: the fill stops short
                                        * of them and balance() drops lent units to rebuild them
                                        * (MoverConfig::turnover_slots) */
        int64_t  n_donated = 0;        /* units in lent slots that gave their host copy away
                                        * (Mover::donor_slot); at most n_turn of them */
        std::vector<char*>   flex_ptr;
        /* One slab segment per lender that carved slots for this class. `need[i]` is the loan,
         * in the lender's measure, that covers the segment's i-th slot; ascending. */
        struct Lent { int lender = 0; int seg = -1; int64_t first = 0; std::vector<int64_t> need; };
        std::vector<Lent>    lent;
        /* Units can sit past the loan only after a shrink, since nothing is placed in an
         * uncovered slot. Set on a shrink; cleared by a balance() sweep that removed them all. */
        bool     strand_check = true;

        char* slot_ptr(int64_t slot) const {
            return slot < n_fixed ? (char*)slab_base + slot * stride
                                  : flex_ptr[(size_t)(slot - n_fixed)];
        }
        bool is_flex(int64_t slot) const { return slot >= n_fixed; }
    };

    const Program* prog_ = nullptr;
    const Plan*    plan_ = nullptr;
    MoverConfig    cfg_;
    Residency      res_;
    MoverStats     stats_;
    Pools*         budget_ = nullptr;
    /* True only when this object called the allocator itself (the test path). A Pool
     * reservation belongs to the Pool; freeing it here would be a double free. */
    bool           owns_memory_ = false;

    RadStream mover_ = nullptr;
    RadStream compute_ = nullptr;
    RadEvent  gate_ = nullptr;          /* recorded on compute, waited on by the mover stream */
    std::vector<RadCopySeg> stage_segs_;   /* stage_units' list, kept so a layer allocates nothing */
    struct StagePick { const char* src; int32_t u; int64_t bytes, need; char* dst; };
    std::vector<StagePick>  stage_picks_;  /* the units stage_units may take, in pool order */
    int       gate_dispatch_ = -1;

    /* A dispatch's swap copies (demotions, promotions, migrations) go to the card's DMA engine as
     * one list (flush_moves), sent at the end of balance() and before anything reads their
     * events. Each move's event is recorded after the list. */
    std::vector<RadCopySeg> move_segs_;
    std::vector<int32_t>    move_recs_;
    void queue_move_copy(void* dst, const void* src, int64_t bytes, int rec);
    int  flush_moves();
    std::vector<ClassPools> pools_;
    std::vector<UnitState>  unit_;
    std::vector<Move>       moves_;
    /* Per unit, how many records are Copying it, so unit_in_flight() needs no walk of the
     * records. The prefill stager asks about every pooled unit it stages, and the record count
     * grows with the move budget. */
    std::vector<uint16_t>   copying_;
    /* balance()'s drop candidates, (heat, unit), kept between calls so a step allocates nothing. */
    std::vector<std::pair<float, int32_t>> drop_cand_;
    bool hold_turnover_ = false;
    /* donor_slot()'s candidates per class, by layer, each layer sorted coldest first (hottest at
     * the back). Ranked once per dispatch; `at` is that dispatch. */
    struct DonorRank { int at = -1; std::vector<std::vector<std::pair<float, int32_t>>> by_layer; };
    std::vector<DonorRank> donors_;
    /* Per class, the layer of the unit each pool slot held at layout. The pool is filled in unit
     * order, so a layer's pooled units start out adjacent (pool_slot_for). */
    std::vector<std::vector<int32_t>> pool_layer_;
    int32_t n_layers_ = 0;

    void*   static_vram_ = nullptr;
    int64_t static_bytes_ = 0;
    void*   pageable_ = nullptr;
    int64_t pageable_bytes_ = 0;

    FileTier file_;
    void*    stage_ = nullptr;          /* pinned staging ring for file-tier reads */
    int64_t  stage_stride_ = 0;
    int      stage_next_ = 0;

    /* The file stager's two VRAM buffers, one allocation, and the pinned windows it reads
     * through, each with the event behind the copies out of it. */
    char*    file_vram_ = nullptr;
    int64_t  file_buf_bytes_ = 0;
    char*    windows_ = nullptr;
    int      n_windows_ = 0;
    int      win_next_ = 0;
    std::vector<RadEvent> win_ev_;
    std::vector<uint8_t>  win_rec_;

    /* The lead gate's ring: one event a dispatch, and a host wait on the one from
     * max_lead_dispatches ago. */
    std::vector<RadEvent> lead_;
    int                   lead_next_ = 0;
    int32_t               prefetch_cursor_ = 0;
    int32_t               last_op_ = -1;
    int                   dispatch_ = 0;

    /* Each lender's loan in its own measure, and the totals the dashboard reads. */
    int64_t              loan_[kLenders] = {};
    std::atomic<int64_t> flex_used_{0};
    std::atomic<int64_t> flex_offered_{0};
    /* Where the last free record was found. Records free up roughly in the order they were
     * taken, so starting there keeps the search short when lent slots put many in flight. */
    mutable size_t       rec_cursor_ = 0;

    int  retire(HeatEngine* heat);
    int  record_gate();
    void republish(int32_t unit);
    void* reserve(const char* what, int64_t bytes, int kind);
    int  free_move_record() const;
    int  start_stage(int32_t unit, int64_t slab_slot);
    void release_slot(int32_t unit);
    bool unit_in_flight(int32_t unit) const;
    /* Whether stage_units would copy `unit`: pooled, with no slab slot and no copy in flight. */
    bool stageable(int32_t unit) const;
    /* A unit in a lent slot back to the host copy it kept. See Move::Kind::Drop. */
    int  drop(int32_t unit, HeatEngine* heat);
    /* The pool slot a demotion copies `unit` into: one its layer's units started out in, else a
     * same-layer donor's, else any free one, else any donor's; -1 when there is none. */
    int64_t pool_slot_for(int32_t unit, HeatEngine* heat);
    /* A pool slot for a budgeted victim when none is free: the host copy of a unit in a KV-lent
     * slot, which stays in VRAM without it. From a unit of `layer`, or of any layer when
     * negative; -1 when no unit may give one. Rationale at the definition. */
    int64_t donor_slot(size_t klass, int32_t layer, HeatEngine* heat);
    /* A donor out of a lent slot into a free budgeted one, if one is free: false when none is. */
    bool migrate(int32_t unit, HeatEngine* heat);
    /* One promotion out of the unit's pool slot into `slot`, which the caller has taken. `fill`
     * marks one the fill issued rather than a swap, for HeatStats::fill_promotions. */
    int  promote_to(int32_t unit, int64_t slot, bool fill = false);
    /* Promote the hottest pooled units of one class into up to `room` free slots. */
    int  fill(size_t klass, int64_t room, HeatEngine* heat);
    void count_flex();
};

}  /* namespace rad */
