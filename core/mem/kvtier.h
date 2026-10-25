/* kvtier.h -- where a finished conversation's KV is kept once VRAM has a better use for it
 * (spec §7.3): VRAM, then host memory.
 *
 * The prefix cache keeps a finished turn's blocks referenced for the next turn. They are valuable
 * while the session is live and dead weight once it stops; on a card whose experts do not fit,
 * VRAM held by an idle conversation costs every decode step link traffic.
 *
 * Copy on idle, free on demand. An entry is copied to host memory as soon as the request that
 * published it finishes, and its VRAM copy is kept. The entry is then clean: giving up the VRAM
 * copy costs only the cache's reference, and that happens as soon as something wants the memory
 * (the expert slab's loan, or an allocation the free list cannot serve). A conversation that
 * returns earlier resumes from VRAM; one that returns later is restored from host and keeps the
 * host copy, so its next turn copies only the tokens that turn added.
 *
 *     published --(copy, at once)--> clean on device --(memory wanted)--> host
 *                                        ^                                  |
 *                                        +------------(a hit restores)------+
 *
 * Host memory is the last tier. When a copy needs a slot and the arena is full, the least
 * valuable host-only entries are dropped, one per copy refused, and only ones less valuable than
 * that copy: older, or as old and deeper in their chain. A copy nothing can make room for is not
 * taken and its entry stays in VRAM only. A conversation larger than the tier so keeps its start
 * on the host, the part a lookup reaches first, instead of dropping its first block for its last
 * one. Nothing goes to disk: a disk tier would be written at the prefill rate to save re-prefills
 * only for what outgrows VRAM and host memory together. Every entry also expires after
 * --prefix-cache-ttl seconds unused, from whichever tier holds it (PrefixCache::expire).
 *
 * There is no idle threshold for leaving VRAM: a clean entry costs nothing to keep until the
 * memory is wanted and nothing to release once it is. Restoring a 200K-token prefix over PCIe
 * takes a fraction of a second; re-prefilling it takes minutes.
 *
 * This object moves no bytes. It decides what moves and hands out TierJob lists; the engine runs
 * them on its transfer stream and reports back, as with KVManager's BlockCopy and
 * take_pending_copies. core/mem has no stream, and the step path must not synchronise (spec §5.4).
 *
 * Addressing: the pool is layer-major (core/mem/kv.h). Layer i of a paged group owns
 * `[base + i*layer_stride, + layer_stride)`, and a block within it starts at
 * `block_index * layer_bytes_per_block()`. A logical block is n_layers fragments of a few hundred
 * bytes, not one extent; `base + block * bytes_per_block` would address n_layers blocks of layer
 * zero. Such corruption yields fluent, plausible, wrong text. The unit of transfer is therefore
 * one layer of many blocks: within a layer blocks are dense rows, so packing is gather_rows and
 * restoring is scatter_rows, one launch per (group, layer) per batch.
 *
 * One record, one tier, one list. If an entry sat on two tiers' LRU lists, each tier's eviction
 * would invalidate the other's references. So a key has exactly one Rec, in exactly one tier, on
 * at most that tier's list. A clean device record is on no list (the lists hold entries no longer
 * in VRAM). A Rec with a transfer outstanding is `pending` and on no list: it cannot be a victim,
 * be planned again or be erased until the engine reports the job.
 */
#pragma once
#include "mem/hostarena.h"
#include "mem/kv.h"
#include "mem/prefix.h"
#include "rad_core.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rad {

enum class KVTier : uint8_t { Device = 0, Host = 1 };
const char* kv_tier_name(KVTier t);

/* One entry moving one step. Block ids travel in the job because the record is `pending` and
 * must not be read until the job reports. */
struct TierJob {
    enum class Kind : uint8_t {
        WriteBack,      /* gather_rows out of the pool into a host slot; the entry stays in VRAM */
        ToDevice,       /* allocate blocks, scatter_rows in; the host copy is kept */
        DropHost,       /* the arena is full: this record's bytes go */
    };
    Kind      kind = Kind::WriteBack;
    BlockHash key;
    /* Per cached group, in PrefixCache::cached_groups() order: the blocks to read for a
     * write-back, or the blocks the engine allocated for a promotion. */
    std::vector<int32_t> blocks;
    int32_t   host_slot = -1;
    int64_t   bytes = 0;

    /* Whether a promotion's scatter may have run. A restore reserves every block first and cuts
     * the chain where the reservation ran out; entries past the cut were never started and stay
     * on the host tier. Only a started promotion drops its entry on failure. */
    bool      started = false;
};

/* ------------------------------------------------------------------ the policy */
class IdleTiers {
public:
    struct Config {
        int64_t host_cap_bytes  = 0;   /* 0 is off */
        int64_t entry_bytes     = 0;   /* one entry: sum over cached groups of bytes_per_block */
        /* ---- the linear half ----
         * `ckpt_bytes` is one snapshot including every rank's shard, like `entry_bytes`. The
         * snapshots live in the engine's checkpoint store, `ckpt_slots` deep; reported here, not
         * allocated (--prefix-cache-mib covers blocks only). */
        int64_t ckpt_bytes      = 0;
        int64_t ckpt_slots      = 0;   /* the checkpoint store's capacity, for the report and view */
        /* Tokens one entry covers (the cache's block size), for the report and the warning when
         * the tier cannot hold a conversation. */
        int64_t block_tokens    = 0;
    };

    int  configure(const Config& c);
    void close();
    bool host_on() const { return cfg_.host_cap_bytes > 0 && arena_.enabled(); }
    std::string report() const;

    /* PrefixCache published this entry with these blocks. A record without a host copy is queued
     * for one: once published, the request has stopped writing it. `depth` is the entry's block
     * index in its chain; among entries used at the same moment the deeper one is worth less. */
    void note_used(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns,
                   int64_t depth = 0);
    /* A lookup hit `n` blocks of a chain. Only the idle clock moves; no records are created. One
     * lock for the whole chain, since a long prefix is thousands of blocks. */
    void note_hits(const BlockHash* keys, int64_t n, uint64_t now_ns);
    /* The entry left the cache for good; its tier storage is returned. A record with a job out is
     * marked `doomed` instead and erased when the job reports. */
    void forget(const BlockHash& key);
    /* An off-device entry was recomputed by a request that could not reach the host copy, and the
     * cache took its blocks as the device copy. The tier copy is dropped (two batches' results
     * agree only to batch-shape rounding) and the record is requeued for a fresh copy. */
    void rebound(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns,
                 int64_t depth = 0);

    /* Device for keys this object has never seen: the cache holds them in the pool. */
    KVTier tier_of(const BlockHash& key) const;
    bool   resident_off_device(const BlockHash& key) const;

    /* ---------------- copy on idle ----------------
     *
     * Up to `max_entries` records waiting for a host copy, oldest first, as WriteBack jobs with
     * host slots allocated. Blocks only: snapshots are already in host memory.
     *
     * Each returned record is `pending` until report(). A job the full arena cannot take stays
     * queued when a host-only entry is worth less than it, and the next plan() drops one such
     * entry for it; otherwise it leaves the queue and its entry stays in VRAM only (Stats::
     * held_back, and a warning naming --prefix-cache-mib). */
    void plan_writeback(int64_t max_entries, std::vector<TierJob>* out);
    bool writeback_waiting() const;
    /* The arena refused a copy since the last plan(), so a plan is due now. */
    bool starved() const;

    /* Entries that became clean since the last call. The cache decides whether each VRAM copy
     * actually goes (a block a running sequence holds would free nothing), reports drops through
     * drop_commit, and re-asks about the rest itself. */
    void take_clean(std::vector<BlockHash>* out);
    /* Host copy landed and no job out: the blocks can leave VRAM at no cost. */
    bool droppable(const BlockHash& key) const;
    /* The cache released the blocks; the record becomes a host record on the host list. Its
     * snapshot stays in its store slot. */
    void drop_commit(const BlockHash& key);

    /* ---------------- capacity ----------------
     *
     * Make room for the copies refused since the last plan, one DropHost job each, taking the
     * least valuable host-only entries and only ones worth less than the copy they make room
     * for; `budget` bounds one plan. */
    void plan(int budget, std::vector<TierJob>* out);

    /* ---------------- expiry ----------------
     *
     * Host records unused since `cutoff_ns`, oldest first, up to `max` (--prefix-cache-ttl). The
     * cache drops each one; records with a job out are not listed and are reconsidered later. */
    void expired(uint64_t cutoff_ns, size_t max, std::vector<BlockHash>* out) const;

    /* A promotion for a lookup that hit off-device. False when the entry is not in host memory or
     * has a job out. */
    bool plan_promote(const BlockHash& key, TierJob* out);

    /* The engine finished or failed a job. A failure restores the record's previous state with
     * its storage intact, except a started ToDevice, which may have half-scattered and so drops
     * the entry. A successful ToDevice restarts the record's clock at `now_ns` when nonzero. */
    void report(const TierJob& job, bool ok, uint64_t now_ns = 0);

    /* The key was forgotten while its job was out, or is unknown. The engine checks this before
     * touching a finished job's blocks, which may now belong to another sequence. */
    bool doomed(const BlockHash& key) const;

    /* ---------------- archives ----------------
     *
     * A pinned record stays off the host list, so it is never a capacity drop's victim and never
     * expires (PrefixCache::pin). The cache counts pins and calls this when a key's pin starts or
     * ends. A key may be pinned before its record exists: the tiers learn of an entry only when
     * it is published, and a lost entry recomputed later is still the archive's. */
    void set_pinned(const BlockHash& key, bool on);

    /* Called at every site that erases an off-device record (capacity drop, failed promotion,
     * expiry). Without it the cache would keep the entry, resident_off_device would answer false,
     * and the stale block ids would be served as a hit. The cache drops the entry without
     * releasing blocks. */
    void set_drop_sink(std::function<void(const BlockHash&)> f) { on_drop_ = std::move(f); }

    /* ---------------- the linear half of an idle session ----------------
     *
     * Called by PrefixCache::commit_checkpoint and retire_ckpt so the session view can show which
     * conversations hold a snapshot and where they resume. Snapshots stay in their store slots in
     * host memory; the tiers only record them. A checkpoint is committed mid-request, before the
     * chain is published, so this creates the record if needed. */
    void note_snapshot(const BlockHash& key, int32_t slot, int64_t pos);
    /* The cache retired the snapshot at `key`. The record's blocks are untouched: blocks and
     * snapshots are retired by different policies. */
    void release_snapshot(const BlockHash& key);
    int64_t  ckpt_bytes() const { return cfg_.ckpt_bytes; }
    int64_t  ckpt_slots() const { return cfg_.ckpt_slots; }

    /* ---------------- the session view ----------------
     *
     * The cache is keyed by block; a session is one published chain, the blocks one request
     * inserted, i.e. a leaf of the prefix trie. Two conversations sharing a system prompt are two
     * sessions that both own the shared blocks, so per-session sizes overlap and do not sum to
     * the total. Anything reporting these numbers must say so. */
    struct SessionInfo {
        uint64_t    id = 0;
        std::string model;
        std::string label;     /* the client's own name for the conversation, or "" */
        int64_t     n_tokens = 0;
        int64_t     bytes = 0;
        uint64_t    first_seen_ns = 0;
        uint64_t    last_used_ns = 0;
        int64_t     on_device = 0, on_host = 0;                /* blocks, by tier */
        /* Linear state on a hybrid model, counted apart from blocks: a snapshot is thousands of
         * blocks' worth of bytes and there are only --checkpoint-slots of them. */
        int64_t     snapshots = 0;                             /* this session holds */
        int64_t     snap_bytes = 0;
        /* Token position of the deepest snapshot on the chain, 0 if none. On a hybrid model hits
         * clamp to it, so this is what the next turn skips; n_tokens minus it gets prefilled. */
        int64_t     resume_tokens = 0;
        bool        moving = false;                            /* a job is outstanding */
    };
    /* Host memory held for the cache, each block counted once (including clean entries' copies),
     * and the configured cap. */
    struct Occupancy {
        int64_t host_bytes = 0;
        int64_t host_cap_bytes = 0;
    };
    /* Record one request's published chain (from PrefixCache::insert). Kept even with every tier
     * off, since a VRAM-only server still stores conversations. A session is a leaf, so a chain a
     * later turn extends is absorbed into it: one row per conversation, not per turn. */
    void note_session(uint64_t id, const std::vector<BlockHash>& chain, int64_t n_tokens,
                      uint64_t now_ns, const std::string& label = std::string());
    void set_model(const std::string& m) { model_ = m; }
    /* Newest activity first, tier histograms recomputed from the records. The walk takes the lock
     * in slices and yields between them (YieldingMutex), so rows may describe moments a few
     * microseconds apart. */
    std::vector<SessionInfo> sessions(Occupancy* occ = nullptr) const;

    struct Stats {
        /* `to_host` counts copies to host memory (link traffic); `dropped` counts VRAM copies
         * then given up. An entry can be copied and never dropped. */
        int64_t to_host = 0, dropped = 0, promoted = 0;
        /* `host_drops` is entries the full arena gave up; `held_back` copies it did not take,
         * because every host entry was worth as much (their entries stay in VRAM only). */
        int64_t host_drops = 0, failures = 0, held_back = 0;
        int64_t host_bytes = 0;
        int64_t host_hits = 0;
        /* Step-thread host time under the scheduler's lock: restores issued at admission, and
         * copies landed synchronously under pressure. Not link time. */
        int64_t restores = 0, restore_ns = 0, flushes = 0, flush_ns = 0;
    };
    /* The step thread's own time on the tiers (see Stats). */
    void note_restore(int64_t ns) { std::lock_guard lk(mu_); ++stats_.restores; stats_.restore_ns += ns; }
    void note_flush(int64_t ns)   { std::lock_guard lk(mu_); ++stats_.flushes;  stats_.flush_ns += ns; }
    /* A copy taken under the lock; HTTP threads read it while the engine moves records. */
    Stats        stats() const;
    HostArena&   arena() { return arena_; }
    /* Bytes per slot; under tensor parallelism one slot holds every rank's shard. */
    int64_t      entry_bytes() const { return cfg_.entry_bytes; }

private:
    /* The host list's order, least valuable first: least recently used, and among entries used
     * at the same moment (a chain is stamped with one clock) the deepest. */
    struct LruKey {
        uint64_t used_ns = 0;
        int64_t  depth = 0;
    };
    struct LruOrder {
        bool operator()(const LruKey& a, const LruKey& b) const {
            return a.used_ns != b.used_ns ? a.used_ns < b.used_ns : a.depth > b.depth;
        }
    };
    using HostLru = std::multimap<LruKey, BlockHash, LruOrder>;

    struct Rec {
        KVTier   tier = KVTier::Device;
        uint64_t used_ns = 0;
        int64_t  depth = 0;         /* block index in its chain (note_used) */
        /* Host copy: on a device record it makes the record clean; on a host record it is the
         * only copy. */
        int32_t  host_slot = -1;
        bool     pending = false;
        bool     doomed  = false;   /* forget() arrived while pending */
        bool     queued  = false;   /* in wb_q_: waiting for a copy */
        bool     clean_q = false;   /* in clean_q_: a drop the cache has not been asked about */
        bool     listed  = false;   /* on host_lru_; `it` is valid */
        std::vector<int32_t> blocks;
        /* ---- the linear half ----
         * Snapshot store slot and token position, for the session view. `ck_slot` is -1 for most
         * records: one block in (interval / block_size) is a checkpoint position. */
        int32_t  ck_slot = -1;
        int64_t  ck_pos = 0;
        HostLru::iterator it;
    };

    /* Blocks in VRAM with no host copy yet. */
    bool wants_copy(const Rec& r) const;
    void queue_copy(const BlockHash& key, Rec& r);
    void queue_clean(const BlockHash& key, Rec& r);
    /* Every block-arena free goes through give_slot, because a freed slot unblocks a stuck
     * write-back. free_host frees the record's own slot. */
    void give_slot(int32_t slot);
    void free_host(Rec& r);
    void unlist(Rec& r);
    void relist(const BlockHash& key, Rec& r);
    void touch(const BlockHash& key, Rec& r, uint64_t now_ns);

    struct Session {
        int64_t  n_tokens = 0;
        uint64_t first_seen_ns = 0;
        uint64_t last_used_ns = 0;
        std::vector<BlockHash> chain;
        std::string label;
    };

    /* Erase one row and its root index entry together. */
    void drop_session(uint64_t id);
    /* Every row whose chain starts at `key`, which has gone for good. */
    void drop_rooted(const BlockHash& key);
    /* The body of close(), for a caller that already holds the lock. */
    void reset();

    /* The engine loop makes every mutation; an HTTP thread walks the same containers in
     * sessions(), so every mutator takes this lock. No method calls another that takes it, so a
     * non-recursive mutex suffices; the drop sink runs under it and calls into the prefix cache,
     * which never calls back.
     *
     * The view's walk is far longer than any engine-side hold, so it locks one slice of records
     * at a time and, between slices, waits until no other taker is waiting. An engine thread then
     * waits at most one slice. A plain mutex cannot promise that: the walker could re-lock ahead
     * of a woken waiter every time. */
    struct YieldingMutex {
        std::mutex       m;
        std::atomic<int> waiting{0};
        void lock() {
            waiting.fetch_add(1, std::memory_order_relaxed);
            m.lock();
            waiting.fetch_sub(1, std::memory_order_relaxed);
        }
        void unlock() { m.unlock(); }
        /* The view's: taken only while no other taker is waiting. */
        void lock_after_others() {
            for (;;) {
                while (waiting.load(std::memory_order_relaxed) > 0) std::this_thread::yield();
                m.lock();
                if (waiting.load(std::memory_order_relaxed) == 0) return;
                m.unlock();
            }
        }
    };
    mutable YieldingMutex mu_;

    Config   cfg_;
    HostArena arena_;
    std::unordered_map<BlockHash, Rec, BlockHashHash> recs_;
    /* Keys an archive pins (set_pinned); relist() skips them. */
    std::unordered_set<BlockHash, BlockHashHash> pinned_;
    /* Host-only records by value, begin() least valuable (LruOrder). Keyed by the clock, not
     * arrival order: a record arrives when its VRAM copy goes, which says nothing about its last
     * use. */
    HostLru host_lru_;
    /* Work queues in arrival order. Each key appears at most once (the Rec's flag says so); a
     * key whose record changed is skipped when reached rather than searched for. */
    std::deque<BlockHash> wb_q_;       /* records wanting a host copy */
    std::vector<BlockHash> clean_q_;   /* records whose VRAM copy may now go */
    /* For each copy the arena refused since the last plan() that a host-only entry is worth less
     * than, its own value, most valuable first; plan() drops one entry worth less for each. */
    std::vector<LruKey> room_;
    uint64_t held_back_warned_ns_ = 0;
    /* A write-back found the arena full with nothing to drop; no copy is planned until a slot is
     * freed, so ticks do not spin on the queue. */
    bool     wb_stuck_ = false;
    std::unordered_map<uint64_t, Session> sessions_;
    /* Sessions by first block. A chain is reachable only from position zero, so once its first
     * block goes (forget(), or a drop here via drop_rooted) its rows go too; this keeps the table
     * bounded. Sessions sharing a system prompt share a root, hence a list. */
    std::unordered_map<BlockHash, std::vector<uint64_t>, BlockHashHash> roots_;
    std::string model_;
    std::function<void(const BlockHash&)> on_drop_;
    Stats    stats_;
    std::string report_;
};

}  /* namespace rad */
