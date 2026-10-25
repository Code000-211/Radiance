/* prefix.h -- prefix caching (spec §7.3). Full and linear attention need separate mechanisms.
 *
 * Full attention: content-hashed blocks with a chained hash, so a block's identity includes its
 * whole prefix. LRU eviction and copy-on-write on fork. Multimodal spans are keyed by their
 * processed patches, not the file bytes (MMContentKey).
 *
 * Linear attention: state checkpoints. A GDN layer's recurrent state at position N depends on all
 * of 0..N and is one fixed-size object per sequence (on Qwen3.5-27B, 48 heads of 128x128 fp32 is
 * 3 MiB a layer, 144 MiB across its 48 linear layers), far too large to snapshot per 16-token
 * block. This follows vLLM's `--mamba-cache-mode align` with two changes:
 *
 *   (a) The checkpoint interval is decoupled from the attention block size. vLLM inflates the
 *       attention block to the mamba page (528 tokens for Qwen3.5, 2240 for the 35B-A3B), so
 *       shorter prompts get no attention hits either. Here the attention block stays 16 (§7.2
 *       takes it from the kernel; libr4d's paged attention is compiled for 16) and
 *       Config::checkpoint_interval is a separate, larger number. A hit between checkpoints
 *       reuses the attention blocks and replays only the linear layers from the last checkpoint:
 *       in CacheHit, `n_attn_tokens` is free, `[n_linear_tokens, n_attn_tokens)` costs a
 *       linear-layers-only pass, and the rest is ordinary prefill.
 *
 *   (b) Checkpoints are taken deliberately. vLLM writes one only when a step happens to end on a
 *       boundary. The scheduler here owns chunk boundaries (spec §7.1) and splits a chunk at the
 *       interval (next_checkpoint_after()), so a checkpoint is always written, at the cost of an
 *       occasional short chunk.
 *
 * Sizing: the cache holds a KV reference on every block it indexes, so pool demand scales with
 * live sessions, not running requests. Hit rate falls off a cliff with pool utilisation: intact
 * below about 75%, half gone by about 81%, and zero once the sessions no longer fit, because each
 * turn evicts the prefixes the next one needs. Output is unaffected; TTFT grows 10-100x. A context
 * of T tokens occupies about 1.8 x T pool-tokens across the paged groups. The budget gives the
 * cache what the experts can spare (core/mem/vram_budget.h); --host-pool-mib moves that.
 *
 * Monitoring: `kv_util` (/stats, or vllm:kv_cache_usage_perc) predicts trouble and
 * `radiance:prefix_cache_evictions_total` counts it. The hit rate lags: it falls only after a
 * reusable prefix is already gone.
 */
#pragma once
#include "mem/kv.h"
#include "rad_core.h"

#include <algorithm>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class IdleTiers;

/* ------------------------------------------------------------------ block identity */
/* 128 bits, chained: identical spans under different prefixes are different blocks. The width
 * makes a collision improbable; the token compare on lookup makes a wrong hit impossible. */
struct BlockHash {
    uint64_t hi = 0, lo = 0;
    bool operator==(const BlockHash& o) const { return hi == o.hi && lo == o.lo; }
    bool operator!=(const BlockHash& o) const { return !(*this == o); }
    bool valid() const { return hi != 0 || lo != 0; }
    std::string str() const;
};

struct BlockHashHash {
    size_t operator()(const BlockHash& h) const { return (size_t)(h.lo ^ (h.hi * 0x9e3779b97f4a7c15ull)); }
};

/* A multimodal span, keyed by content. Placeholder token ids are the same for every image, so
 * `hash` is the item's content key (core/mm: its processed patches, grid and tiling) mixed with
 * the run's row offset, by media_keys in core/sched/mem_iface.h (spec §7.3, §11). The same picture
 * at the same resolution and place hits even if the file bytes differ, and different pictures
 * never alias. Hashing the file instead would hit with stale KV whenever preprocessing changes. */
struct MMContentKey {
    int32_t  tok_begin = 0;    /* [begin, end) in prompt-token coordinates */
    int32_t  tok_end = 0;
    uint64_t hash = 0;
};

/* The chained hash of one block. `parent` is the previous block's hash (a zeroed BlockHash for
 * block 0), `block_begin` is the block's first token position. */
BlockHash chain_block_hash(const BlockHash& parent, const int32_t* toks, int32_t n,
                           const MMContentKey* mm, int32_t n_mm, int64_t block_begin);

/* ------------------------------------------------------------------ the split lookup */
struct CacheHit {
    int32_t n_attn_tokens   = 0;   /* attention KV reused directly, always a whole block multiple */
    int32_t n_linear_tokens = 0;   /* linear state reused; <= n_attn_tokens, a checkpoint multiple */
    int32_t checkpoint_slot = -1;  /* the snapshot holding that state, -1 if none */
    int32_t given_back      = 0;   /* reusable tokens a HitCap took off the whole hit */
};

/* Ends a hit early on purpose: at or below `tokens`, on a block and (with recurrent state) on the
 * deepest checkpoint at or below it, unless that gives back more than `max_give_back` reusable
 * tokens, in which case the whole hit stands. For a drafter whose KV the cache does not hold
 * (Scheduler::init, --draft-refill): recomputing the prompt's tail gives it context, and the
 * give-back limit stops a far-off checkpoint from turning that into a long recompute. */
struct HitCap {
    int64_t tokens = 0;
    int64_t max_give_back = 0;
};

/* ------------------------------------------------------------------ retention policy */
/* Which checkpoints a session keeps. Spec §19.1 treats geometric backoff as unmeasured, so the
 * policy is a replaceable class with two alternatives alongside it. */
class CheckpointPolicy {
public:
    virtual ~CheckpointPolicy() = default;
    virtual const char* name() const = 0;
    /* `positions` ascending, one per checkpoint the session holds; fill `keep` with the subset to
     * retain. The tip, positions.back(), must always be kept: an appending transcript resumes
     * from it. */
    virtual void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const = 0;
    /* The most snapshots the policy keeps for one session that grows to `ctx` tokens with a
     * checkpoint on every `interval`, retention applied after each: the session's standing share
     * of the store. The base applies select() to its own output one checkpoint at a time, which
     * is cheap only for a policy that keeps few. */
    virtual int64_t held(int64_t ctx, int64_t interval) const;
};

/* Keep the tip and, at every scale S = interval x 2^k, the shallowest checkpoint in the tip's
 * bucket [j*S, (j+1)*S) and in the bucket before it. Depths behind the tip grow geometrically, a
 * session keeps about log2(tip / interval) + 2 snapshots, and a turn that diverges d tokens behind
 * the tip resumes less than max(2d, 2 x interval) below the divergence (spec §7.3). Buckets are
 * fixed token ranges, so the rule applied again after more checkpoints keeps the same ladder, and
 * a snapshot it drops is never one it would select later. */
class GeometricBackoff final : public CheckpointPolicy {
public:
    /* `interval` is the resolved checkpoint interval, the smallest scale. */
    explicit GeometricBackoff(int64_t interval) : interval_(interval > 0 ? interval : 1) {}
    const char* name() const override { return "geometric-backoff"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;

private:
    int64_t interval_;
};

/* The two ends of the trade, for comparison. */
class KeepTipOnly final : public CheckpointPolicy {
public:
    const char* name() const override { return "tip-only"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
    int64_t held(int64_t, int64_t) const override { return 1; }
};
class KeepAll final : public CheckpointPolicy {
public:
    const char* name() const override { return "keep-all"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
    int64_t held(int64_t ctx, int64_t interval) const override;
};

/* The policy --checkpoint-policy names, at the resolved checkpoint interval. config_parse admits
 * only the three names, so any other is geometric backoff. */
std::unique_ptr<CheckpointPolicy> make_checkpoint_policy(const std::string& name, int64_t interval);

/* ------------------------------------------------------------------ the store's size */
/* --checkpoint-slots auto. Every sequence --max-num-seqs admits gets its session's standing share
 * at --max-model-len (CheckpointPolicy::held) plus one slot for the checkpoint its running turn
 * writes while the previous turn's snapshots are still held. The store is pinned for the life of
 * the process, so the count is bounded by two byte budgets the engine measures: `share`, the part
 * of host memory the store may take, and `room`, what the host can pin after everything still to
 * be allocated. Under the share it never drops below kCheckpointSlotsFloor (or `want`, if fewer);
 * the room is a hard bound. */
constexpr int64_t kCheckpointSlotsFloor = 8;

struct CheckpointSlots {
    int64_t per_seq = 0;      /* snapshots a sequence is given */
    int64_t want = 0;         /* --max-num-seqs x per_seq */
    int64_t slots = 0;        /* the store's size */
    enum class Bound { Want, Share, Room } bound = Bound::Want;
};

CheckpointSlots checkpoint_slots_auto(const CheckpointPolicy& policy, int64_t max_seqs,
                                      int64_t ctx, int64_t interval, int64_t slot_bytes,
                                      int64_t share_bytes, int64_t room_bytes);

/* ------------------------------------------------------------------ the cache */
class PrefixCache {
public:
    /* `kv` must already be configured. The cache holds a KV reference on every block it indexes,
     * so an indexed block is never on the free list and cannot be handed to another sequence.
     *
     * `checkpoint_interval` is the scheduler's resolved interval, not cfg.checkpoint_interval:
     * chunk_geometry_resolve rounds the flag up to the chunk quantum. Using the raw value would
     * make reserve_checkpoint refuse every checkpoint and lookups search the wrong stride. */
    int configure(const Config& cfg, KVManager* kv, int64_t checkpoint_interval);
    bool enabled() const { return enabled_; }
    int64_t block_size() const { return block_size_; }
    int64_t checkpoint_interval() const { return interval_; }
    const std::vector<int32_t>& cached_groups() const { return groups_; }
    std::string report() const;

    /* Overrides the policy configure() installed from Config::checkpoint_policy (custom policies,
     * tests). Call after configure(), which would undo it. */
    void set_policy(std::unique_ptr<CheckpointPolicy> p);
    const CheckpointPolicy& policy() const { return *policy_; }

    /* ---------------- full attention ---------------- */
    /* Longest cached prefix, in whole blocks, plus the checkpoint that covers the linear state
     * for a prefix of it. `blocks_out`, when given, receives the block ids per cached group, in
     * cached_groups() order, ready for KVManager::adopt().
     *
     * Never returns the whole prompt: one token must remain to compute a logit from, so a full
     * hit is backed off by one block. `cap`, when given, ends the hit early (HitCap). `session`,
     * when not 0, is the request being admitted: it takes over the snapshots on the chain it hit
     * (take_over). */
    CacheHit lookup(const std::vector<int32_t>& tokens,
                    const std::vector<MMContentKey>& mm,
                    std::vector<std::vector<int32_t>>* blocks_out = nullptr,
                    const HitCap* cap = nullptr, uint64_t session = 0) const;

    /* Publish the whole blocks a sequence just filled. `blocks[i]` is the block table of
     * cached_groups()[i]. Takes a KV reference on every newly indexed block. */
    int insert(uint64_t session,
               const std::vector<int32_t>& tokens,
               const std::vector<MMContentKey>& mm,
               const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens,
               const std::string& label = std::string());

    /* Release up to `n_entries` least recently used entries' KV references; returns how many went.
     * Entries a live sequence also holds are skipped, since releasing them frees nothing. With the
     * idle tiers on, an entry whose host copy has landed moves to the host tier instead of being
     * forgotten; only one with no host copy is lost. */
    int64_t evict_lru(int64_t n_entries);
    /* Free up to `n_blocks` pool blocks for the scheduler's reserve without losing anything a
     * later turn could reuse. One oldest-first walk that skips entries a live sequence also holds
     * and, with the tiers on, entries whose host copy has not landed yet. Pressure that cannot
     * wait goes to evict_lru. Returns how many blocks went back. */
    int64_t trim(int64_t n_blocks);
    /* Give up every VRAM copy that costs nothing to drop: host copy landed and no running sequence
     * reading it. Used when the memory has a better use (the expert slab's loan). Entries still
     * shared are retried after a sequence frees blocks. Returns how many entries left VRAM. */
    int64_t drop_clean();
    int64_t size() const { return (int64_t)map_.size(); }
    /* Entries holding blocks in the pool, as opposed to only a copy off the card. */
    int64_t resident() const { return (int64_t)lru_.size(); }
    /* Pool held by the cache alone, in tokens: blocks nothing else references, i.e. what an
     * eviction would actually free. The entry count overstates this, because entries share blocks
     * with every live sequence that hit them. */
    int64_t unshared_tokens() const;
    /* Forget every entry, in VRAM or off it, and give back whatever the tiers held for them. */
    void    clear();
    /* Diagnostic: the LRU order, oldest first. */
    std::vector<BlockHash> lru_order() const;
    /* Diagnostic: clean entries drop_clean() is waiting to recheck, each listed once. */
    size_t drop_waiting() const { return drop_wait_.size(); }

    /* ---------------- idle session tiers (core/mem/kvtier.h) ---------------- */
    /* Installed after configure() when the tiers are on. Null means a plain VRAM-only cache;
     * every use is guarded. */
    void set_tiers(IdleTiers* t) { tiers_ = t; }

    /* The off-device entries this prompt's chain would hit if restored, in order, from where
     * lookup stopped. The caller promotes them and looks up again; empty means the prefix ended
     * on a miss or token mismatch. Separate from lookup because lookup is const and has no
     * device; a promotion allocates blocks and runs a scatter between steps. */
    std::vector<BlockHash> restorable_from(const std::vector<int32_t>& tokens,
                                           const std::vector<MMContentKey>& mm) const;

    /* The tiers lost the bytes. Erases the entry without releasing blocks: an off-device entry
     * owns none, and its recorded ids now belong to other sequences. Installed as
     * IdleTiers::set_drop_sink. */
    void drop_offdevice(const BlockHash& key);

    /* A restore finished and the entry's blocks now live at these ids. The restore's raw
     * reservations (KVManager::reserve_block) become cache-held references. */
    int rebind(const BlockHash& key, const std::vector<int32_t>& blocks);

    /* ---------------- archives ----------------
     *
     * A pinned chain survives every way the cache lets go of an entry: LRU eviction, trim and
     * --prefix-cache-ttl expiry, and its snapshot survives retention and the store's LRU too. The
     * memory endpoints pin the chain a conversation had when its client compacted it, so a later
     * question can still fork it (core/sched/memory.h).
     *
     * A pinned entry may still leave VRAM once its host copy has landed, since that loses nothing,
     * and the tiers never drop a pinned host copy (IdleTiers::set_pinned). Only the deepest
     * snapshot is pinned, not the chain's whole ladder: one is all a fork resumes from, and the
     * store is shared with every live conversation.
     *
     * Pins count, so two archives of one chain share its entries and each releases only its own.
     * A pin is keyed by hash, not by entry: an entry the tiers lost and a later request recomputed
     * is pinned again. clear() drops every pin with everything else. */
    struct Pinned {
        std::vector<BlockHash> entries;   /* the chain's blocks from position 0, as pinned */
        BlockHash snapshot;               /* the deepest valid snapshot among them, if any */
        int64_t   snapshot_pos = 0;       /* 0 when there is none */
    };
    /* Pins the whole blocks of the first `n_tokens` of `tokens` that the cache holds, up to the
     * first miss, and the deepest valid snapshot within them. */
    void pin(const std::vector<int32_t>& tokens, const std::vector<MMContentKey>& mm,
             int64_t n_tokens, Pinned* out);
    /* Releases exactly what pin() reported. */
    void unpin(const Pinned& p);
    /* The deepest valid snapshot on this chain within the blocks the cache holds, 0 if none. */
    int64_t deepest_snapshot(const std::vector<int32_t>& tokens,
                             const std::vector<MMContentKey>& mm, int64_t n_tokens) const;
    bool    pinned(const BlockHash& h) const { return pins_.count(h) > 0; }
    bool    snapshot_pinned(const BlockHash& h) const { return ck_pins_.count(h) > 0; }
    int64_t pinned_blocks() const { return (int64_t)pins_.size(); }


    /* ---------------- the linear half of an idle session ----------------
     *
     * On a hybrid model a session also needs its recurrent state back, held in a checkpoint slot:
     * the engine's snapshot store in pinned host memory, KVManager::checkpoint_bytes() per rank,
     * --checkpoint-slots in total for the server. Without a reachable snapshot a session reuses
     * nothing: the scheduler clamps reuse to the last reachable checkpoint because RadBatch has
     * one query length for all KV groups (Scheduler::admit_one). A snapshot is restored straight
     * from its slot wherever the blocks are. There is no tier below the store; a retired snapshot
     * is gone and the session replays from an older one. */

    /* Keep a waiting request's chain on the card. An admission that restored a chain and was then
     * refused gives its hit back with clean entries; without a hold, drop_clean would drop them
     * and every retry would restore the chain again. drop_clean skips held entries until
     * release(); trim and evict_lru still take them. Idempotent per request. */
    void hold(uint64_t req, const std::vector<int32_t>& tokens, const std::vector<MMContentKey>& mm);
    void release(uint64_t req);
    /* Start of a step's admissions. Snapshots the previous step's lookups handed out have been
     * read by its forward by now, and the ones it reserved have been saved into (the engine
     * issues the save before the next step is planned); until this call they keep their slots
     * against retention, reserve_checkpoint's eviction and expiry. */
    void begin_step() { handed_.clear(); reserved_.clear(); }

    /* ---------------- expiry ----------------
     *
     * Remove everything no request has used for `ttl_ns`: VRAM entries, host-only entries and
     * snapshots, each returned to its pool. An entry is used by a hit or a publish; a snapshot by
     * a use of its prefix entry or a lookup handing it out. Nothing is demoted first.
     *
     * A hit or publish stamps every block of the chain with one clock, so a prefix is never older
     * than what follows it: conversations expire from the tip inwards and a shared system prompt
     * outlives its readers. Returns how many entries went. */
    int64_t expire(uint64_t now_ns, uint64_t ttl_ns);
    /* ---------------- linear checkpoints ---------------- */
    /* The next position strictly after `pos` at which the scheduler must split a chunk so that a
     * checkpoint lands on a boundary (spec §7.1, §7.3). */
    int64_t next_checkpoint_after(int64_t pos) const;
    bool    is_checkpoint(int64_t pos) const { return interval_ > 0 && pos > 0 && pos % interval_ == 0; }

    /* Reserve a snapshot slot for `session` at `pos`. When the store is full, applies the
     * retention policy, then evicts the least recently used snapshot (a use is its reservation or
     * a lookup resuming from it), a tip like any other. A snapshot handed out this step is never
     * released, by retention or eviction: its admission restores from the slot ahead of the step.
     * Nor is one this call reserved and commit_checkpoint has not yet committed: the engine saves
     * into the slot after the step, and a later reservation in the same step must not run
     * retention over a save target (a second request reserving a snapshot a first session holds
     * below its tip). RAD_E_FULL when nothing can be given up. */
    int  reserve_checkpoint(uint64_t session, int64_t pos, const BlockHash& prefix,
                            int32_t* slot_out);
    /* The state for `prefix` is now in its slot; lookup() offers it only after this (Ckpt::valid). */
    void commit_checkpoint(const BlockHash& prefix);
    /* Every checkpoint of a session, ascending. */
    std::vector<int64_t> checkpoints(uint64_t session) const;
    /* Apply the retention policy to one session now, keeping also what a lookup resumed from
     * since the session's tip was written. Returns how many snapshots were released. A session
     * left with none is erased from the index. */
    int64_t retain(uint64_t session);

    /* Sessions in the checkpoint index. Invariant: since a session is a request id, this tracks
     * sessions holding a live snapshot, not every request served. */
    int64_t sessions_tracked() const { return (int64_t)ck_by_session_.size(); }

    struct Stats {
        int64_t lookups = 0, block_hits = 0, block_misses = 0;
        int64_t ckpt_hits = 0, ckpt_misses = 0;
        int64_t evictions = 0, ckpt_evictions = 0;
        /* ckpt_evictions by reason: released by the retention policy, or taken as the least
         * recently used of a store retention left full. `ckpt_refused` counts reservations that
         * found every slot handed out this step and went without a snapshot. */
        int64_t ckpt_retired = 0, ckpt_lru = 0, ckpt_refused = 0;
        /* Entries whose VRAM copy was given up with the host copy kept; not evictions, since
         * nothing was lost. */
        int64_t drops = 0;
        /* Entries and snapshots removed by --prefix-cache-ttl, from any tier. */
        int64_t expired = 0, ckpt_expired = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    struct Entry {
        std::vector<int32_t> blocks;    /* one per cached group, in groups_ order */
        std::vector<int32_t> tokens;    /* the block's token ids, compared on hit */
        /* On the LRU exactly while the cache holds the blocks; `lru` is valid only then. An
         * off-device entry owns no block, so eviction must not reach it. */
        bool in_lru = false;
        bool drop_wait = false;         /* in drop_wait_: clean, but a sequence still reads it */
        /* Last hit or publish, monotonic clock. Every path that moves an entry to the LRU's back
         * stamps it, so the list is in this order and expiry stops at the first entry it keeps. */
        uint64_t used_ns = 0;
        std::list<BlockHash>::iterator lru;
    };
    struct Ckpt {
        uint64_t  session = 0;
        int64_t   pos = 0;
        BlockHash prefix;
        int32_t   slot = -1;
        /* `clock_` values: the last use (reservation or hand-out, the eviction order), the
         * reservation, and the last hand-out (0 if none). */
        uint64_t  used = 0, made = 0, resumed = 0;
        uint64_t  used_ns = 0;          /* reserved, or handed out by a lookup */
        /* The state has been written into the slot. Reservation happens when the chunk is
         * planned and the copy after the step that computes it; in between the slot still holds
         * the previous tenant's state and must not be handed out. */
        bool      valid = false;
    };

    std::vector<BlockHash> chain_all(const std::vector<int32_t>& tokens,
                                     const std::vector<MMContentKey>& mm,
                                     int64_t n_blocks) const;
    /* The deepest valid snapshot at or below `limit` tokens on the chain `hs`, or end(). */
    std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator
    deepest_on(const std::vector<BlockHash>& hs, int64_t limit) const;
    /* How many of the chain's first blocks the cache holds, in VRAM or off it. */
    int64_t held_prefix(const std::vector<BlockHash>& hs, const std::vector<int32_t>& tokens) const;
    void touch(const BlockHash& h, uint64_t now_ns) const;
    /* A live sequence also holds one of this entry's blocks, so releasing it frees nothing. */
    bool shared_with_live(const Entry& e) const;
    /* Whether `e` holds the block_size_ token ids at `src`, so a hash collision is a miss. */
    bool same_tokens(const Entry& e, const int32_t* src) const;
    /* Gives the entry's block in every group back to the KV manager. */
    void release_blocks(const Entry& e);
    /* Retire one snapshot: free its slot and tell the tiers. The session index is left to the
     * caller. */
    void retire_ckpt(std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator it);
    /* Move every snapshot on the chain `hs` at or below `upto` tokens under `session`. A
     * conversation's snapshots belong to its latest turn: sessions are request ids, so otherwise
     * each earlier turn's deepest snapshot would stay a protected tip, and a running turn would be
     * a second session beside its conversation, two ladders where retention should keep one.
     * The admission's lookup and the publish both call it. */
    void take_over(uint64_t session, const std::vector<BlockHash>& hs, int64_t upto) const;
    /* Take one key out of a session's index entry, and the entry when it empties. */
    void unindex_ckpt(uint64_t session, const BlockHash& key);
    /* For the debug lines: whether `c` is its session's deepest snapshot, and seconds since a
     * monotonic stamp. */
    bool is_tip(const Ckpt& c) const;
    static double idle_s(uint64_t used_ns);

    KVManager*  kv_ = nullptr;
    IdleTiers*  tiers_ = nullptr;
    bool        enabled_ = false;
    int64_t     block_size_ = 0;
    int64_t     interval_ = 0;
    std::vector<int32_t> groups_;         /* paged groups this cache indexes */
    bool        has_linear_ = false;

    mutable std::unordered_map<BlockHash, Entry, BlockHashHash> map_;
    mutable std::list<BlockHash> lru_;    /* front = oldest; resident entries only */
    /* Clean entries a sequence was still reading at drop_clean's last pass, and the pool's
     * sequence-free count then. They are rechecked only once that count moves. */
    std::vector<BlockHash> drop_wait_;
    std::vector<BlockHash> drop_scratch_;
    uint64_t    frees_seen_ = 0;
    /* hold(): keys per waiting request, and holders per key. A release also triggers a recheck
     * of drop_wait_. */
    std::unordered_map<uint64_t, std::vector<BlockHash>> holds_;
    std::unordered_map<BlockHash, int32_t, BlockHashHash> held_;
    bool        released_ = false;
    mutable std::unordered_map<BlockHash, Ckpt, BlockHashHash> ck_by_prefix_;
    mutable std::unordered_map<uint64_t, std::vector<BlockHash>> ck_by_session_;
    /* Snapshots lookup handed out since begin_step; kept in their slots until the next one. */
    mutable std::vector<BlockHash> handed_;
    bool handed(const BlockHash& h) const {
        return std::find(handed_.begin(), handed_.end(), h) != handed_.end();
    }
    /* Snapshots reserve_checkpoint reserved and commit_checkpoint has not committed: the step's
     * save targets, kept in their slots until the commit, or until the next step begins when the
     * save was skipped (its request gone). */
    mutable std::vector<BlockHash> reserved_;
    bool reserved(const BlockHash& h) const {
        return std::find(reserved_.begin(), reserved_.end(), h) != reserved_.end();
    }
    mutable uint64_t clock_ = 0;
    /* Archive pins (pin()), counted: entries by chain hash, and snapshots by the same key their
     * Ckpt is indexed under. */
    std::unordered_map<BlockHash, int32_t, BlockHashHash> pins_;
    std::unordered_map<BlockHash, int32_t, BlockHashHash> ck_pins_;
    /* Moves a key's pin count by `delta` and tells the tiers when it starts or stops. */
    void pin_entry(const BlockHash& h, int delta);
    std::unique_ptr<CheckpointPolicy> policy_;
    mutable Stats stats_;
    std::string report_;
};

}  /* namespace rad */
