/* mover.h -- residency and the mover: the substrate every tier stands on.
 *
 * Three storage tiers, one address space of slots:
 *
 *   FIXED-STRIDE VRAM SLAB SLOTS PER WEIGHT CLASS. A slot is interchangeable within its class, so
 *   a promotion is an integer swap and not an allocation, and a slab that never fragments is a
 *   slab whose hit rate does not depend on the order promotions happened to arrive in.
 *
 *   A PINNED HOST POOL, at the same stride. Pinned rather than pageable because the whole point of
 *   the host tier is that a card can DMA out of it: a device-visible pinned pool is what makes a
 *   non-resident expert a zero-copy read inside the GEMM instead of a bounce-buffer memcpy in
 *   front of it. It is also allocated and owned here rather than read through the container's
 *   mmap, because with an mmap the KERNEL decides what stays resident and under real memory
 *   pressure it decides badly for this workload -- the same reads run several times slower inside
 *   a decode run than in isolation, and MADV_WILLNEED makes it worse, which is the tell that the
 *   faults are not too small and that a hint cannot fix a policy disagreement.
 *
 *   A FILE TIER READ WITH O_DIRECT INTO PINNED BUFFERS. Predictable latency, no page-cache
 *   double-copy, and the read lands where the mover can DMA from it without a bounce. A buffered
 *   fallback is refused rather than offered: a tier whose latency is the page cache's mood is not
 *   a tier you can budget against.
 *
 * ---- THE ONE PROPERTY EVERYTHING ELSE RESTS ON -----------------------------------------------
 *
 * THE MOVER RUNS ON A DEDICATED STREAM AND THE COMPUTE STREAM WAITS ON ITS EVENTS. NO HOST
 * SYNCHRONISATION IS INVOLVED IN A TRANSFER. That is what lets placement be dynamic without
 * serialising the step, and it is the difference between MoE offload being unusable and being
 * fast. rad_issue waits with rad_event_wait, never rad_event_sync.
 *
 * The lead gate below IS a host wait, and it is not a contradiction: it throttles how far the
 * host runs ahead of the cards. It is a SUPPLY mechanism, not a transfer synchronisation, and the
 * reason it exists is spelled out at MoverConfig::max_lead_dispatches.
 *
 * ---- SHADOW SLOTS AND QUARANTINE -------------------------------------------------------------
 *
 * A promotion writes into a SHADOW slot -- a slab slot that is allocated but holds nothing -- and
 * only then publishes the new pointer. So a live slot is never DMA'd over while a kernel might
 * still be reading it, and a promotion never invalidates a pointer the run phase is already
 * holding. Getting that ordering wrong does not crash: it produces fluent output with a different
 * hash every run, which is the single most expensive bug class in this design.
 *
 * The freed slot then goes into QUARANTINE rather than straight back to the free list, because
 * kernels issued in earlier layers still name its old address. The barrier is an event recorded
 * on the COMPUTE stream after the publish; when it signals, every launch that could still name
 * the slot has retired.
 */
#pragma once
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
/* Pure host state, no device call in it, so the whole supply half is testable without a card.
 * Used for both the VRAM slab and the pinned host pool -- they are the same structure with
 * different backing memory, and writing it twice is how two free lists come to disagree about
 * what a slot index means.
 *
 * ---- SEGMENTS: ONE RANGE FOR THE SLAB'S OWN SLOTS, ONE FOR EACH LENDER ------------------------
 *
 * The slots are numbered in SEGMENTS laid end to end. Segment 0 is the budgeted range the pool
 * was built with; every later one is a stretch of another pool's memory lent to this one -- the
 * KV cache's blocks, the activation arena's top -- added once its addresses are known. Each
 * segment has its own TARGET: the slots below it are the ones that may be handed out, and a
 * lender moves only its own. Two lenders cannot share one target, because what one of them can
 * spare says nothing about the other: the arena takes its memory back for every prefill while the
 * cache is still lending, and a single mark over both would either hold the arena's slots past its
 * recall or give up the cache's with it.
 *
 * ---- WHY THE LOWEST FREE SLOT, AND WHY A LIVE MARK -------------------------------------------
 *
 * A lent segment is VRAM its owner can want back, and it has to be handed back when it does. That
 * is only possible if the occupied slots pack toward the BOTTOM of the segment, so its top is free
 * to retire. Handing out the lowest free slot is what packs them; a LIFO stack hands back the slot
 * released most recently, which after a busy period is exactly the high slot the pool wants to
 * give up, and the mark would never fall.
 *
 * A segment's `live` is how far its slots exist right now and `target` how far they should. They
 * differ while a shrink waits on the units still sitting above the target: the mark stops at the
 * highest slot something holds, so shrinking can never evict -- it only declines to hand a slot
 * out again.
 *
 * THE FREE LIST IS REBUILT WHENEVER A TARGET MOVES, from the held flags, and holds exactly the
 * covered slots nothing holds. Carried across a shrink and a regrow instead, it lists only some of
 * the slots the shrink dropped -- a slot freed while it sat above the target is never pushed, and
 * one the heap still held is discarded by the first take() that walks past it -- and a regrow
 * that trusts it never hands the others out again. The rebuild is a scan of the capacity at the
 * arbiter's cadence; take() and add_free, which the mover calls several times a dispatch, stay
 * heap operations.
 */
class SlotPool {
public:
    /* `n_live` slots exist in segment 0 and every one of them is HELD until add_free says
     * otherwise, which is what the plan's resident set is. `n_cap` is how far segment 0 may ever
     * grow -- address space the caller has reserved and will back on demand -- and defaults to no
     * growth at all. */
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
    int     n_segments() const { return (int)segs_.size(); }
    int64_t seg_lo(int k) const { return segs_[(size_t)k].lo; }
    int64_t seg_hi(int k) const { return segs_[(size_t)k].hi; }

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
    /* Slots something is in: a resident unit, a staged one, or a copy still in flight. The
     * number the fill and drain decisions are made against, because the working slots have to
     * stay free whatever the target is. */
    int64_t held() const { return n_held_; }
    /* Where segment `k`'s covered slots end, as a slot number. */
    int64_t target(int k = 0) const { return segs_[(size_t)k].target; }
    /* May this slot be handed out -- is it below its segment's target? */
    bool covered(int64_t slot) const {
        if (slot < 0 || slot >= n_) return false;
        return slot < segs_[(size_t)seg_of(slot)].target;
    }
    /* -1 when there is none. A caller that treats that as "try again next dispatch" is right;
     * one that treats it as an error is wrong -- slot starvation is the normal steady state. */
    int64_t take() {
        while (!free_.empty()) {
            std::pop_heap(free_.begin(), free_.end(), std::greater<int64_t>());
            const int64_t s = free_.back();
            free_.pop_back();
            if (!covered(s) || held_[(size_t)s]) continue;   /* never a slot to hand out */
            held_[(size_t)s] = 1;
            ++n_held_;
            --n_usable_;
            return s;
        }
        return -1;
    }

    /* Where segment `k`'s covered slots should end, as a slot number inside it. Growing is
     * immediate -- the caller has already backed the memory. Shrinking is a statement of intent:
     * the mark falls as far as the slots nothing holds allow, and the rest follows as those
     * retire. */
    void set_target(int64_t target) { set_target(0, target); }
    void set_target(int k, int64_t target) {
        Seg& g = segs_[(size_t)k];
        g.target = target < g.lo ? g.lo : (target > g.hi ? g.hi : target);
        /* Nothing ever holds a slot at or past the mark -- take() hands out only covered slots,
         * and the mark never falls past a held one -- so a set flag there only says the slot has
         * never existed. */
        for (int64_t s = g.live; s < g.target; ++s) held_[(size_t)s] = 0;
        if (g.target > g.live) g.live = g.target;
        else settle(g);
        rebuild();
    }

    /* take(), restricted to slots at or above `lo`. The slots below it that are free stay free
     * and stay listed; there are at most the working slots of them, so this is a handful of heap
     * operations. */
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
    /* The highest slot of segment `k` something holds -- a unit, a copy in flight, a slot in
     * quarantine -- or -1. A scan down from the segment's mark. */
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
/* A thin skin over the device layer's DirectFile, which is where the O_DIRECT mechanics live and
 * where they belong: the prefix cache reads the same way for the same reasons, and two
 * implementations of "align the offset, the length and the buffer" is two places for the
 * alignment rule to drift. What this adds is the tier's own accounting -- a file-tier read is a
 * blocking pread in the middle of a dispatch and the counter is how a run that leans on it stops
 * being a mystery in the throughput. */
class FileTier {
public:
    /* O_DIRECT, and a refusal rather than a buffered fallback: a tier whose latency is the page
     * cache's mood is not a tier you can budget against. `path` is the container. */
    int  open(const std::string& path);
    bool ok() const { return f_.is_open(); }
    void close() { f_.close(); }

    /* One expert-sized read into a pinned buffer. Blocking, counted, and a cliff: a dispatch that
     * meets a file-tier unit stops until the read returns. That is the price of a bet the plan
     * made -- give the cold tail to disk and buy a host pool big enough to hold a working set
     * that can actually move -- paid in full each time the bet is wrong. What changes in response
     * is the plan, not this function. */
    int  read(uint64_t offset, int64_t bytes, void* pinned_dst, int64_t dst_cap);

    /* Every RadFileEntry offset is RAD_ALIGN_UNIT-aligned by construction (rad_format.h) and that
     * is DirectFile's alignment, so the mover's reads take the path with no bounce in it. */
    static int64_t direct_align(int64_t v) { return align_up(v, DirectFile::alignment()); }

    uint64_t reads() const { return reads_; }
    uint64_t bytes_read() const { return bytes_; }

private:
    DirectFile f_;
    uint64_t   reads_ = 0, bytes_ = 0;
};

/* ------------------------------------------------------------------ residency */
class Residency final : public ResidencyTable {
public:
    void init(const Program& prog, const Plan& plan);

    const WeightSlot* lookup(rad_weight w) const override {
        return (size_t)w < slot_.size() ? &slot_[w] : nullptr;
    }
    uint64_t epoch() const override { return epoch_; }
    const std::vector<rad_weight>& changes() const override { return changed_; }
    void clear_changes() override {
        for (rad_weight w : changed_) noted_[w] = 0;
        changed_.clear();
    }

    /* The only two writers, and BOTH BUMP THE GENERATION. `publish` settles a slot -- the bytes
     * are there, nothing to wait on. `begin_transfer` opens one: a copy into `ptr` is in flight
     * and `ready` is the event the compute stream must be ordered behind.
     *
     * begin_transfer bumping is the part that is easy to get wrong and expensive to debug. The
     * run phase memoises the wait per generation, so installing a new event at the old generation
     * makes it skip the wait entirely (residency_iface.h). */
    void publish(rad_weight w, void* ptr);
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
    /* Shadow slots and the staging ring are NOT here. They are budgeted by the planner out of
     * --vram-weights-mib and reported in the plan (SlabClass::n_shadow, n_stage), and a second
     * knob for the same quantity is a second number for the mover and the operator to disagree
     * about -- with an overrun as the direction that disagreement goes wrong in.
     *
     * ---- HOW FAR THE HOST MAY RUN AHEAD OF THE CARDS, IN LAYER DISPATCHES --------------------
     *
     * 0 is unbounded, and unbounded STARVES THE MOVER. A freed slot returns to the free list only
     * after its quarantine event signals, and that event is on the COMPUTE stream -- so the
     * recycle latency is however deep that stream happens to be queued, and the move rate is
     * `shadow slots / recycle latency`. Let the host run and the stream carries a whole step's
     * work, and almost every round finds no free slot to move into. The guard is not what binds;
     * the engine simply never gets the supply to spend.
     *
     * Holding the host near the cards is therefore not a latency measure but a SUPPLY one. 2 is
     * the smallest depth that recovers the effect while leaving the issue path a dispatch of
     * margin. A coarser gate -- per step rather than per layer dispatch -- does nothing, because
     * one step is already the whole model's worth of queued work; the granularity has to be the
     * layer. */
    int max_lead_dispatches = 2;

    /* The container, for the file tier. Empty means there is no file tier and a unit the planner
     * put there is an error rather than a slow path. */
    std::string container_path;
    /* Pinned staging buffers for file-tier reads. Each is one class stride, and the count is the
     * planner's -- RAD_PLACE_FILE_STAGE_SLOTS, which is what the operator was charged for out of
     * --host-pool-mib. A separate default here would be a second number for one allocation. */
    int stage_slots = RAD_PLACE_FILE_STAGE_SLOTS;
};

/* ------------------------------------------------------------------ the loan */
/* ONE STRETCH OF ANOTHER POOL'S VRAM THAT MAY BE LENT TO THE SLAB, described from its top down.
 *
 * The owner keeps it backed and lends it from the top downwards, and it states how much it has
 * lent in its own measure -- tokens, for the KV cache -- so the mover can tell which slots a loan
 * covers without knowing what the owner stores there. `unit_bytes` of the stretch are one unit of
 * that measure and a unit is `unit_tokens` tokens: a loan of T tokens is the top
 * floor(T / unit_tokens) * unit_bytes bytes of every stretch. */
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
    /* Why a move did not go out. `no_record` is every in-flight move record busy -- the shadow
     * slots, still waiting on their copies. The two `starved` counts are the free lists empty.
     * Without these a starved mover is indistinguishable from a satisfied one. */
    uint64_t no_record = 0, starved_slab = 0, starved_pool = 0;
    uint64_t quarantined = 0;
    /* Demotions that copied nothing: the unit sat in a lent slot and kept its host copy, so going
     * back to the host tier was a change of address. Counted inside `demotions` as well. */
    uint64_t dropped = 0;
    uint64_t ssd_reads = 0, ssd_bytes = 0;
    /* The prefill stager's copies (stage_units): layers staged, units and bytes copied, and
     * pooled units a buffer had no room left for -- those were read across the link in place. */
    uint64_t stage_layers = 0, staged_units = 0, staged_bytes = 0, stage_overflow = 0;
    /* The file stager's reads (stage_file_units): routed layers read from the container, the
     * units and bytes they carried, the preads that carried them, and the host time spent inside
     * those reads and waiting for a window's copies to drain -- the two halves of what a step
     * that reads the drive costs the thread issuing it. */
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

    /* Allocates every pool the plan asked for and lays every unit into a slot. Does NOT read a
     * byte of the model: the load phase fills the addresses this hands it. Returns RAD_E_NOMEM
     * naming the pool that failed, because "out of memory" without which pool is not a
     * diagnosis.
     *
     * `budget` IS THE PRODUCTION PATH AND SHOULD ALWAYS BE PASSED. Pools already allocated
     * --vram-weights-mib as one block; a mover that called rad_dev_alloc beside it would spend
     * that budget TWICE and the overrun would surface as an out-of-memory from an unrelated
     * launch much later. Reserving from Pools::weights() is also what makes it structurally
     * impossible for a promotion to reach a KV block -- there is no path from the weights pool to
     * the KV pool (mem/pools.h), which is the whole reason the weight slab is a fixed budget.
     *
     * Null is the standalone-test path and says so at Warn. It is not a supported deployment. */
    int init(const Program& prog, const Plan& plan, const MoverConfig& cfg, Pools* budget = nullptr);

    /* The compute stream this rank issues on. The mover's transfers are ordered against it and
     * the quarantine barrier is recorded on it, so without one the mover refuses to move rather
     * than storing a table edit blind. */
    int bind_compute_stream(RadStream s);

    /* THE MAINTENANCE STREAM, AND WHY ANYTHING ELSE MAY HAVE IT.
     *
     * A HIP stream costs a hardware queue, and on RDNA4 a fourth concurrent queue is not a small
     * cost: on a two-card R9700 Flash-Next server, a fourth stream that never carries a single
     * byte takes the decode step from 20.5 ms to 33.9 (GPU_MAX_HW_QUEUES 1, 2 or 3 all run at
     * full speed; 4 and 8 do not). The engine already issues on three -- compute, the second
     * compute lane, and this one -- so a subsystem that wants a stream to move bytes off the
     * critical path must TAKE THIS ONE rather than make a fourth.
     *
     * Null before init(), and null on a rank with nothing to move. A caller that gets null must
     * fall back to its own stream: three streams is under the cliff either way.
     *
     * THE MOVER'S RUN PHASE ISSUES ON IT, so this stream is not idle: a sharer queueing a hundred
     * megabytes on it will sit in front of a transfer the compute stream is waiting on. Gate the
     * two -- do not answer it by making a fourth queue. */
    RadStream transfer_stream() const { return mover_; }

    ResidencyTable* residency() { return &res_; }
    /* THE ONE ADDRESS THE MOVER DOES NOT OWN. A Tier::Mapped weight lives in the container's
     * mapping and the LOAD PHASE is what holds the RadFile, so it publishes the pointer and the
     * mover only records it -- the residency table is the mover's, and a second table would be a
     * second answer to "where is this weight". Nothing here ever moves it afterwards: a mapped
     * unit has no slab class, so unit_base() has no address of its own to republish over this. */
    void publish_mapped(rad_weight w, void* p) { res_.publish(w, p); }

    /* ---- the run phase, in the order a layer dispatch calls it -----------------------------
     *
     * begin_dispatch  retire whatever landed, gate the host's lead, record the ordering event
     * prefetch_to_op  release what this op has finished with, then start everything the exact
     *                 schedule says is due -- and keep starting, as long as slots are free
     * issue           one heat proposal
     *
     * There is no end_dispatch, and the absence is deliberate. The quarantine barrier belongs
     * DOWNSTREAM of the publish in stream order, and begin_dispatch's retire() is exactly that
     * point: by the time it runs, every launch the previous dispatch made has been issued. A
     * symmetric end-of-dispatch hook would record the barrier before those launches and the
     * quarantine would be satisfied for a reason that has nothing to do with the kernels it is
     * guarding.
     *
     * ---- A CALLER MUST DO ALL OF IT, EVERY DISPATCH -------------------------------------------
     *
     * begin_dispatch only RETIRES; it never starts anything. A loop that retires without issuing
     * is not a slower engine, it is a stuck one: the pool free list starts empty, so the first
     * round can only promote out of the shadow slots, and the pool slot that promotion frees
     * arrives several dispatches later -- by which time, if nothing is asking, the resident set
     * has grown by the shadow count and never shrinks again. The symptom is zero demotions
     * forever, and it appears only on a real card: on the host backend every copy lands inside
     * the dispatch that issued it, so the same mistake passes unnoticed.
     *
     * The compute stream must also carry the dispatch's actual work. The quarantine barrier is an
     * event on that stream and its claim is "every kernel that could still name this slot has
     * retired"; against an empty stream the claim is vacuous and the lead gate paces nothing.
     */
    /* `begin_dispatch` and `issue` are the engine's, once a step: Engine::tiering_dispatch is the
     * seam, and it is a step and not a layer because the mover relocates BETWEEN steps and never
     * during one (runtime/ctx.cpp honours `weights_dirty_` once, at the top of run_step).
     *
     * `prefetch_to_op` has no caller but tests/place_test.cpp, and that is DELIBERATE -- the
     * exact-schedule path is kept built and tested rather than deleted and rewritten later. Do
     * not read "no caller" there as the dead-surface shape it resembles elsewhere in this tree. */
    int begin_dispatch(int dispatch, HeatEngine* heat);
    int prefetch_to_op(int32_t op_index);
    int issue(const Swap& s, HeatEngine* heat);

    /* ---- the loan: VRAM another pool is not using ------------------------------------------
     *
     * VRAM some other pool carved and is not using is worth more here than it is there: an
     * expert unit that does not fit the slab is read across the link on the critical path of
     * every step that routes to it. The KV cache keeps its whole carve backed and LENDS the slab
     * the part nothing holds, and the activation arena lends the part a step smaller than the
     * largest one does not reach -- each at addresses it already owns, so the slab grows into
     * that memory with no allocation, no mapping and no driver call, and gives it back the same
     * way. The ONLY caller that may lend is the one that arbitrates the whole card: two callers
     * with two opinions about how much is spare is the same memory in two pools.
     *
     * EACH LENDER LENDS IN ITS OWN MEASURE, into its own segment of the slab (SlotPool): the KV
     * cache in tokens, the arena in bytes. They move independently -- a prefill takes the arena's
     * memory back while the cache goes on lending -- so every call names the lender it is about.
     *
     * A UNIT IN A LENT SLOT KEEPS ITS HOST COPY. Weights never change, so the pinned pool slot it
     * was promoted from still holds exactly its bytes, and giving the slot back is a change of
     * address rather than a copy across the link. That is what makes a loan cheap to recall:
     * the cache waits for kernels to stop naming the slot, never for bytes to move. */

    enum Lender { kLendKV = 0, kLendArena = 1, kLenders = 2 };

    /* The stretches a lender may ever lend, once per lender, after its memory is carved. Carves
     * every stretch into slots of each class's stride from the top down, so the slots a small loan
     * covers are the ones a large loan covers first, and the slot a loan gives back first is the
     * deepest. */
    int     lend_from(int lender, const std::vector<LoanStrip>& strips);

    /* HOW MUCH A LENDER LENDS, in its own measure. A TARGET, not a transaction: slots the loan
     * newly covers can be filled at once, while slots it no longer covers are emptied as their
     * units are dropped and their quarantine clears -- `loan_held` is how far down the lender's
     * memory the slab still reaches, and the lender may not use those bytes again until it has
     * fallen. */
    int     set_loan(int lender, int64_t amount);
    int64_t loan_held(int lender) const;
    int64_t loan(int lender) const { return loan_[lender]; }

    /* THE SAME, NOW, for a lender that cannot wait: every unit past `amount` is dropped to its
     * host copy and both streams that can touch a slot are drained before this returns, so the
     * memory is the lender's the moment it does. A host wait, taken only when the lender needs
     * more than it kept in hand -- a request larger than the cache's buffer, a step larger than
     * the arena's lent level allows. */
    int     reclaim(int lender, int64_t amount, HeatEngine* heat);

    /* What the lent slots hold and what the loan offers, in bytes, and every slot the stretches
     * could ever hold. Read by the HTTP thread for the dashboard, hence stored rather than
     * computed from the slot pools, which only this thread may walk. */
    int64_t flex_used_bytes() const { return flex_used_.load(std::memory_order_relaxed); }
    int64_t flex_offered_bytes() const { return flex_offered_.load(std::memory_order_relaxed); }
    int64_t flex_capacity() const;
    /* THE LOAN PAST WHICH NOTHING MORE WOULD BE USED, in the lender's measure: the one that
     * covers every slot carved from it. The slots were carved no deeper than the units the host
     * pool holds, so a loan this size already has room for every unit that could move -- more VRAM
     * from the lender would sit empty. Zero with nothing carved. */
    int64_t loan_ceiling(int lender) const;

    /* FILL EVERY SLOT THERE IS ROOM IN, AND EMPTY EVERY SLOT THE LOAN NO LONGER COVERS -- the
     * half-swaps the heat engine never proposes, because its decision is a COMPARISON and a
     * comparison needs a victim. Not bounded by a count: a slot left empty while a unit streams
     * across the link is VRAM bought and wasted on every step, so every free slot is filled in
     * one call, hottest units first, and the copies run at the rate the link carries them. */
    int balance(HeatEngine* heat);

    /* balance() for a rank with no step to ride on. Retires what has landed and fills what is
     * free, so a loan made while the server is idle is filled before the next request rather
     * than by it. Only while the rank thread is parked. */
    int idle_tick(HeatEngine* heat);

    /* Where a unit's bytes are right now, as a device-visible address, or null for a file-tier
     * unit that has not been staged. */
    void* unit_base(int32_t unit) const;

    /* Unit residency and slab slot state queries for CED prefetching */
    bool is_unit_resident(int32_t unit) const;
    int32_t unit_klass(int32_t unit) const;
    int64_t unit_bytes(int32_t unit) const;
    bool has_free_slab_slot(size_t klass) const;
    int64_t free_slab_slots(size_t klass) const;

    /* Asynchronous lookahead prefetch of a unit from host-pinned pool to VRAM slab slot.
     * If victim >= 0, evicts victim (which must have an expired lease) to reuse its slab slot.
     * Queues DMA on transfer_stream(). */
    int prefetch_unit_async(int32_t unit, int32_t victim = -1, HeatEngine* heat = nullptr, int lease_dispatches = -1);

    /* ---- STAGING WITHIN A PASS: the prefill stager (core/place/stager.h) ----------------------
     *
     * Everything else the mover does happens between steps, and a unit's address is fixed for the
     * length of a step. These two are the exception, and they change only the PUBLISHED address,
     * never where a unit lives: the unit stays in its pinned pool slot, and for one routed layer's
     * ops the run phase reads a copy of it in a buffer the caller owns.
     *
     * stage_units   copy every unit of `units` that lives in the pinned pool, has no slab slot and
     *               has no copy of its own in flight into `dst`, one after another at the allocation
     *               unit, until `cap` has no room for the next -- on the mover's stream, behind
     *               `after` when it is not null, with `done` recorded behind the last copy. Each
     *               copied unit's weights are then published at the copy, SETTLED: the caller
     *               orders the ops that read them behind `done` with one stream wait, where an
     *               event on every weight would cost the run phase one wait per weight -- a few
     *               hundred a layer. `staged` gets the units copied, in order.
     * unstage_units publish those units at their pool slots again. The caller must already have
     *               ordered the next writer of `dst` behind every op that read the copies; this
     *               only moves the address the NEXT issue resolves.
     */
    int  stage_units(const std::vector<int32_t>& units, char* dst, int64_t cap, RadEvent after,
                     RadEvent done, std::vector<int32_t>* staged);
    void unstage_units(const std::vector<int32_t>& staged);

    /* ---- ROUTED LAYERS READ FROM THE CONTAINER: the file stager (core/place/stager.h) --------
     *
     * A routed unit the plan put on the file tier has no address at all (unit_base is null), and a
     * grouped GEMM is handed every expert's pointer at issue -- so before a routed layer's first
     * expert op, every one of its file-tier units is read into a VRAM buffer and published there,
     * and after the layer it is published as null again (unstage_units does that half). The
     * buffers are the two the plan charged out of --vram-weights-mib (Plan::file_stage_vram).
     *
     * stage_file_units  read every unit of `units` that is on the file tier into `dst`, packed at
     *                   RAD_ALIGN_UNIT in the order given, through the pinned read windows: the
     *                   layer's weights are read in container order as long runs -- a run skips a
     *                   gap of resident units rather than reading through it -- each window drains
     *                   to `dst` on the mover's stream while the next fills, behind `after` when it
     *                   is not null, with `done` recorded behind the last copy. Published SETTLED,
     *                   for the reason stage_units gives. Refuses with RAD_E_SCRATCH when the units
     *                   do not fit `cap`, which a buffer sized by the plan never meets. The reads
     *                   are BLOCKING on this thread; that is the price of the tier and the stats
     *                   above carry it.
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
    /* One in-flight relocation. The state machine is Free -> Copying -> Published -> Quarantine
     * -> Free, and every transition is made by the DISPATCH THREAD at a layer boundary. The
     * transfers are asynchronous; the table edits are not. No mutex on the read side, because
     * that side is the critical path of every layer, and no background thread mutating placement
     * mid-layer, because placement would then depend on host timing and stop being reproducible. */
    struct Move {
        enum class State { Free, Copying, Published, Quarantine };
        /* Five kinds, one state machine. Promote and Demote are the heat engine's swaps. Stage is
         * the exact prefetch bringing an offloaded unit into a slot. Release is the same slot
         * going back once the declared order says nothing will read it again this step -- it
         * copies nothing and still needs the quarantine, because the barrier is about kernels
         * already issued and not about bytes in flight. Drop is a unit in a lent slot going back
         * to the host copy it kept: no copy either, and the same quarantine for the same reason. */
        enum class Kind { Promote, Demote, Stage, Release, Drop };
        State    state = State::Free;
        Kind     kind = Kind::Promote;
        int32_t  unit = -1;
        int64_t  slab_slot = -1;
        int64_t  pool_slot = -1;
        RadEvent ev = nullptr;
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

    /* One weight class's memory. The slab is two kinds of slot rather than one: the BUDGETED slots
     * come out of --vram-weights-mib like every other weight allocation and sit at a fixed stride
     * from one base, and the FLEX slots past them are addresses inside another pool's memory --
     * the KV cache's carve, the activation arena's top -- lent while that pool is not using them.
     * They cannot be one range because the flex slots are scattered over every layer's stripe of
     * the cache, wherever a stripe has room for one.
     *
     * A slot index therefore carries no information about which kind it is; slot_ptr is the only
     * place that distinction exists. Each lender's flex slots are one segment of the slab, ordered
     * by the loan that covers each, so a loan of any size is a prefix of its segment. */
    struct ClassPools {
        SlotPool slab, pool;
        void*    slab_base = nullptr;
        void*    pool_base = nullptr;
        void*    pool_dev = nullptr;   /* the pinned pool as the device addresses it */
        int64_t  stride = 0;
        int64_t  n_fixed = 0;          /* slots in the budgeted range; the flex ones start here */
        int64_t  n_work = 0;           /* shadow + staging: free slots the mover needs to move at all */
        std::vector<char*>   flex_ptr;
        /* ONE SEGMENT OF THE SLAB PER LENDER THAT CARVED SLOTS FOR THIS CLASS. `need[i]` is the
         * loan, in the lender's measure, that covers the segment's i-th slot; ascending, so a
         * loan of any size covers a prefix. */
        struct Lent { int lender = 0; int seg = -1; int64_t first = 0; std::vector<int64_t> need; };
        std::vector<Lent>    lent;
        /* A slot leaves coverage only when a loan shrinks, and nothing is ever placed in an
         * uncovered one, so units stranded past the loan exist only after a shrink. Set there, and
         * cleared by a balance() sweep that dropped every one of them. */
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
    /* True only when this object called the allocator itself, which is the test path. A Pool
     * reservation is owned by the Pool and freeing it here would be a double free. */
    bool           owns_memory_ = false;

    RadStream mover_ = nullptr;
    RadStream compute_ = nullptr;
    RadEvent  gate_ = nullptr;          /* recorded on compute, waited on by the mover stream */
    int       gate_dispatch_ = -1;

    std::vector<ClassPools> pools_;
    std::vector<UnitState>  unit_;
    std::vector<Move>       moves_;

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

    /* THE LOAN, in the owner's tokens, and what the dashboard reads of it. */
    int64_t              loan_[kLenders] = {};
    std::atomic<int64_t> flex_used_{0};
    std::atomic<int64_t> flex_offered_{0};
    /* Where the last free record was found. Records free up roughly in the order they were taken,
     * so starting the next search there makes it a step rather than a walk of every record, which
     * matters once the lent slots have put a record's worth of them in flight. */
    mutable size_t       rec_cursor_ = 0;

    int  retire(HeatEngine* heat);
    int  record_gate();
    void republish(int32_t unit);
    void* reserve(const char* what, int64_t bytes, int kind);
    int  free_move_record() const;
    int  start_stage(int32_t unit, int64_t slab_slot);
    void release_slot(int32_t unit);
    bool unit_in_flight(int32_t unit) const;
    /* A unit in a lent slot back to the host copy it kept. See Move::Kind::Drop. */
    int  drop(int32_t unit, HeatEngine* heat);
    /* One promotion out of the unit's pool slot into `slot`, which the caller has taken. */
    int  promote_to(int32_t unit, int64_t slot);
    /* Promote the hottest pooled units of one class into up to `room` free slots. */
    int  fill(size_t klass, int64_t room, HeatEngine* heat);
    void count_flex();
};

}  /* namespace rad */
