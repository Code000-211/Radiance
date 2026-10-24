/* prefix.h -- prefix caching (spec §7.3). TWO mechanisms, because full attention and linear
 * attention cannot share one.
 *
 * FULL ATTENTION: content-hashed blocks with a CHAINED hash, so a block's identity includes its
 * whole prefix. LRU eviction, copy-on-write on fork, and multimodal inputs hashed BY CONTENT --
 * an image is a cache hit, because the hash is taken over the EMBEDDING span rather than the
 * image bytes.
 *
 * LINEAR ATTENTION: a state checkpoint. A GDN layer's recurrent state at position N depends on
 * all of 0..N and is one fixed-size object per sequence -- 32 heads of 128x128 fp32 is 2 MiB a
 * layer, roughly 96 MiB across the 48 linear layers of a Qwen3-Next. Snapshotting that per 16-token
 * block is not affordable at any block count. We take vLLM's `--mamba-cache-mode align` mechanism
 * and change two things:
 *
 *   (a) THE CHECKPOINT INTERVAL IS DECOUPLED FROM THE ATTENTION BLOCK SIZE. vLLM inflates the
 *       attention block to match the mamba page -- 528 tokens for Qwen3.5, 2240 for the 35B-A3B --
 *       and since prefix caching is block-granular, every prompt shorter than that gets a 0% hit
 *       rate on ATTENTION as well. We could not inflate the attention block even if we wanted to:
 *       §7.2 takes it from the resolved kernel and libr4d's paged attention is compiled for 16.
 *       So attention caches at 16-token granularity and Config::checkpoint_interval is a separate,
 *       much larger number. A hit landing BETWEEN checkpoints reuses the attention blocks and
 *       replays only the linear layers forward from the last checkpoint. That is what CacheHit's
 *       two token counts are: `n_attn_tokens` is free, `[n_linear_tokens, n_attn_tokens)` costs a
 *       linear-layers-only forward pass, and the rest is an ordinary prefill.
 *
 *   (b) CHECKPOINTS ARE TAKEN DELIBERATELY, NOT OPPORTUNISTICALLY. vLLM writes one only when a
 *       scheduler step happens to end on a boundary, which is why a checkpoint landing in
 *       request-unique tokens silently drops caching to zero. Our scheduler owns chunk boundaries
 *       (spec §7.1) and splits a chunk AT the interval, so one is always written. The cost is an
 *       occasional short chunk; the benefit is that hit rate is a property of the workload rather
 *       than of scheduling luck. `next_checkpoint_after()` is what the scheduler splits on.
 *
 * ============================== WHAT IT COSTS TO HOLD, AND WHEN IT STOPS WORKING ==============
 *
 * The cache takes a KV REFERENCE on every block it indexes, so a finished sequence's blocks stay
 * held for the next turn. That is the whole point -- and it means the pool's demand is the number
 * of LIVE SESSIONS, not of running requests. Size it the other way and the failure is not graceful.
 *
 * THE THRESHOLD IS POOL UTILISATION, and it is a CLIFF rather than a slope. Below roughly 75%
 * utilisation the cache is whole; by around 81% half of it is gone; once the sessions no longer
 * fit at all the cache is not degraded but DEFEATED -- each turn evicts the prefixes the next turn
 * wants, so every turn re-prefills its whole context and the hit rate is zero. The generated text
 * is unaffected either way; the only difference is whether the prefixes still exist when the next
 * turn asks for them, and that difference is one to two orders of magnitude of time to first
 * token. Size the cache so live sessions fit with room to spare -- a context of T tokens occupies
 * roughly 1.8 x T pool-tokens across the paged groups -- and keep --expert-vs-cache-ratio on the
 * safe side of that.
 *
 * WHAT TO WATCH: `kv_util` (/stats, or vllm:kv_cache_usage_perc) is the predictor, and
 * `radiance:prefix_cache_evictions_total` is the event -- the hit rate is neither, because it only
 * falls once a prefix that would have been reused is already gone.
 */
#pragma once
#include "mem/kv.h"
#include "rad_core.h"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

class IdleTiers;

/* ------------------------------------------------------------------ block identity */
/* 128 bits, chained. A block's identity includes its whole prefix, so two identical 16-token spans
 * sitting at the same offset under different prefixes are DIFFERENT blocks and cannot be served
 * for one another. That is the property; the width is only there to make an accidental collision
 * improbable, and the stored token ids below make it impossible. */
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

/* A multimodal span, keyed BY CONTENT. The placeholder token ids inside an image span are
 * identical for every image, so hashing token ids alone would serve one request another's picture.
 * The hash is taken over the ENCODER OUTPUT for the span -- the embedding bytes -- not over the
 * image file, so a re-encode of the same picture at the same resolution hits even when the JPEG
 * differs byte-for-byte, and two different pictures never alias (spec §7.3, §11).
 *
 * The cost, stated: a non-deterministic encoder produces different embedding bytes for the same
 * image and therefore MISSES. A miss is the safe direction; the alternative -- hashing the image
 * bytes -- gets a hit and the wrong embeddings when the preprocessing changes. */
struct MMContentKey {
    int32_t  tok_begin = 0;    /* [begin, end) in prompt-token coordinates */
    int32_t  tok_end = 0;
    uint64_t hash = 0;
};
uint64_t mm_content_hash(const void* embedding, int64_t bytes);

/* The chained hash of one block. `parent` is the previous block's hash (a zeroed BlockHash for
 * block 0), `block_begin` is the block's first token position. */
BlockHash chain_block_hash(const BlockHash& parent, const int32_t* toks, int32_t n,
                           const MMContentKey* mm, int32_t n_mm, int64_t block_begin);

/* ------------------------------------------------------------------ the split lookup */
struct CacheHit {
    int32_t n_attn_tokens   = 0;   /* attention KV reused directly, always a whole block multiple */
    int32_t n_linear_tokens = 0;   /* linear state reused; <= n_attn_tokens, a checkpoint multiple */
    int32_t checkpoint_slot = -1;  /* the snapshot holding that state, -1 if none */
    std::vector<int32_t> prefetch_units; /* prefix-aware prefetch: experts needed by this prefix */
};

/* ------------------------------------------------------------------ retention policy */
/* Spec §19.1 records geometric backoff as a GUESS until there is a workload to measure. It is
 * therefore a policy OBJECT and not an if-statement, so replacing it is writing a class rather
 * than rewriting the cache. Two alternatives ship alongside it for exactly that reason. */
class CheckpointPolicy {
public:
    virtual ~CheckpointPolicy() = default;
    virtual const char* name() const = 0;
    /* `positions` ascending, one entry per checkpoint the session holds. Fill `keep` with the
     * subset to retain. THE TIP -- positions.back() -- MUST ALWAYS BE KEPT, because an appending
     * agent transcript is what this cache is for and the tip is the only thing it needs. */
    virtual void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const = 0;
};

/* Keep the tip, then 1x, 2x, 4x, 8x checkpoints back. Bounds the snapshot count at log(n) and the
 * recompute at one interval-doubling, which is the trade §19.1 describes. Distances are counted in
 * CHECKPOINTS rather than in tokens: the scheduler splits chunks at the interval so the spacing is
 * uniform, and counting in checkpoints stays correct if it ever is not. */
class GeometricBackoff final : public CheckpointPolicy {
public:
    const char* name() const override { return "geometric-backoff"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
};

/* The two obvious ends of the trade, kept so a measurement can be run against them. */
class KeepTipOnly final : public CheckpointPolicy {
public:
    const char* name() const override { return "tip-only"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
};
class KeepAll final : public CheckpointPolicy {
public:
    const char* name() const override { return "keep-all"; }
    void select(const std::vector<int64_t>& positions, std::vector<int64_t>* keep) const override;
};

/* ------------------------------------------------------------------ the cache */
class PrefixCache {
public:
    /* `kv` must already be configured: the cache holds a KV-manager reference on every block it
     * indexes, which is the invariant that makes a hit safe -- a block in the map is never in the
     * free list, so it can never be handed to another sequence underneath a cache entry.
     *
     * `checkpoint_interval` IS THE SCHEDULER'S RESOLVED INTERVAL, NOT cfg.checkpoint_interval.
     * The two are different numbers: chunk_geometry_resolve rounds the operator's request UP to
     * the chunk quantum so that "split at the interval" is always also a legal chunk split. If
     * this cache read the raw config value while the scheduler split on the rounded one, every
     * checkpoint would be refused by reserve_checkpoint -- with an error saying the scheduler
     * splits at the interval precisely so this cannot happen -- and restore() would walk back in
     * the wrong stride looking for snapshots nothing had written. It is a parameter rather than a
     * config read so that there is one place the number comes from. */
    int configure(const Config& cfg, KVManager* kv, int64_t checkpoint_interval);
    bool enabled() const { return enabled_; }
    int64_t block_size() const { return block_size_; }
    int64_t checkpoint_interval() const { return interval_; }
    const std::vector<int32_t>& cached_groups() const { return groups_; }
    std::string report() const;

    /* AFTER configure(), which installs the one Config::checkpoint_policy names. This is the
     * in-process override -- a policy that is not one of the three, or one a test wants to
     * substitute -- and calling it first would be undone. */
    void set_policy(std::unique_ptr<CheckpointPolicy> p);
    const CheckpointPolicy& policy() const { return *policy_; }

    /* ---------------- full attention ---------------- */
    /* Longest cached prefix, in whole blocks, plus the checkpoint that covers the linear state
     * for a prefix of it. `blocks_out`, when given, receives the block ids per cached group, in
     * cached_groups() order, ready for KVManager::adopt().
     *
     * NEVER returns the whole prompt: at least one token must remain for the step to compute a
     * logit from, so a full hit is backed off by one block. That is not a subtlety the caller
     * should have to know. */
    CacheHit lookup(const std::vector<int32_t>& tokens,
                    const std::vector<MMContentKey>& mm,
                    std::vector<std::vector<int32_t>>* blocks_out = nullptr) const;

    /* Publish the whole blocks a sequence just filled. `blocks[i]` is the block table of
     * cached_groups()[i]. Takes a KV reference on every newly indexed block. */
    int insert(uint64_t session,
               const std::vector<int32_t>& tokens,
               const std::vector<MMContentKey>& mm,
               const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens,
               const std::vector<int32_t>& hot_units = {});

    /* Drop the least recently used entries, releasing their KV references. Returns how many were
     * dropped. Blocks a live sequence still holds stay allocated -- releasing the cache's
     * reference is not freeing the block.
     *
     * A CLEAN ENTRY IS RELEASED AND KEPT. With the idle tiers on, an entry whose host copy has
     * landed loses nothing by leaving VRAM, so pressure on the pool moves it to the host tier
     * instead of forgetting it: the next turn restores it rather than re-prefilling it. Only an
     * entry with no host copy yet is lost. Either way the blocks go back to the pool. */
    int64_t evict_lru(int64_t n_entries);
    /* GIVE UP EVERY VRAM COPY THAT CAN GO FOR NOTHING: the entries whose host copy has landed and
     * that no running sequence is still reading. Called when the memory has a better use -- the
     * expert slab's loan -- and returns how many entries left VRAM. An entry a sequence still
     * holds would free nothing, so it is asked about again once a sequence has let go of blocks. */
    int64_t drop_clean();
    int64_t size() const { return (int64_t)map_.size(); }
    /* How many entries hold blocks in the pool, as opposed to only a copy off the card. */
    int64_t resident() const { return (int64_t)lru_.size(); }
    /* HOW MUCH OF THE POOL THIS CACHE IS HOLDING ON ITS OWN, in tokens of context. The entry
     * count cannot answer that: an entry's blocks are SHARED with every live sequence that hit on
     * them, and on a server carrying long conversations that is most of the index -- so an index
     * size read as "memory the cache is keeping back" says a request reading its own context is
     * the cache refusing to release it. What an eviction would actually free is the blocks
     * nothing else references, which is what this counts. */
    int64_t unshared_tokens() const;
    /* Forget every entry, in VRAM or off it, and give back whatever the tiers held for them. */
    void    clear();
    /* Diagnostic: the LRU order, oldest first. */
    std::vector<BlockHash> lru_order() const;

    /* ---------------- idle session tiers (core/mem/kvtier.h) ---------------- */
    /* Installed after configure() when the tiers are on. A null pointer is the whole of the "off"
     * path: every call site below is guarded, and with no tiers the cache is a plain VRAM-only
     * prefix cache. */
    void set_tiers(IdleTiers* t) { tiers_ = t; }

    /* THE CHAIN THIS PROMPT WOULD HIT IF THE TIERS GAVE IT BACK, in order, starting at the block
     * `lookup` stopped on. The caller promotes them and asks again; an empty answer means the
     * prefix stopped for an ordinary reason -- a miss, or a token compare -- and no amount of
     * restoring would lengthen it.
     *
     * It is a separate call rather than something lookup does because lookup is const and has no
     * device: a promotion allocates blocks and runs a scatter, which is the engine's to do
     * between steps and not something a cache query may trigger under the caller. */
    /* `chain`, when given, receives every key the walk matched, wherever it is -- what the tiers
     * need to see whether any of it is still being read off disk (IdleTiers::plan_fetch). */
    std::vector<BlockHash> restorable_from(const std::vector<int32_t>& tokens,
                                           const std::vector<MMContentKey>& mm,
                                           std::vector<BlockHash>* chain = nullptr) const;

    /* THE TIERS LOST THE BYTES. Erase the entry WITHOUT releasing blocks: an off-device entry
     * owns none -- demotion released them -- and its recorded ids belong to whoever the pool gave
     * them to since. Releasing those would free another sequence's block; leaving the entry in
     * the map would serve them. Installed as IdleTiers::set_drop_sink. */
    void drop_offdevice(const BlockHash& key);

    /* A restore finished: the entry's blocks live at these ids now. The cache takes a reference on
     * each, because the restore holds them only as raw reservations (KVManager::reserve_block) and
     * this is where they become cache-held blocks with the usual invariant. */
    int rebind(const BlockHash& key, const std::vector<int32_t>& blocks);


    /* ---------------- the linear half of an idle session ----------------
     *
     * ON A HYBRID MODEL THE BLOCKS ARE ONLY HALF OF WHAT A SESSION NEEDS BACK. The other half is
     * the recurrent state, and it lives in a CHECKPOINT SLOT: a device-resident object of
     * KVManager::checkpoint_bytes(), of which there are --checkpoint-slots IN TOTAL for the whole
     * server. Eight by default, against the six or seven a single 200K session retains under
     * geometric backoff -- so a second session evicts the first's snapshots, tip included.
     *
     * Tiering the blocks and leaving the snapshot behind therefore gets the trade backwards: it
     * frees the plentiful resource and holds on to the scarce one. And what the session loses is
     * not part of its hit, it is ALL of it -- the scheduler clamps reuse to the last checkpoint a
     * hit can reach, because RadBatch carries one query length for every KV group and the linear
     * layers cannot be replayed over a longer range than the attention ones (core/sched/
     * scheduler.cpp, admit_one). A restored session with no reachable snapshot reuses zero tokens
     * and re-prefills its whole transcript. The blocks without the state are not a partial win;
     * they are unusable.
     *
     * So a snapshot moves with the blocks whose hash it is keyed by, and these calls are the
     * cache's whole part in that. The bytes are the engine's to move. */

    /* Its bytes are on another tier now: give the device slot back and stop offering it. The index
     * entry STAYS, because that is what a promotion puts the snapshot back into, and an entry
     * whose state is not in the pool is exactly the `valid` flag's existing meaning -- lookup
     * already walks past one and falls back to an older position. */
    int  detach_snapshot(const BlockHash& key);
    /* And back again, into a slot the caller has already filled. */
    int  reattach_snapshot(const BlockHash& key, int32_t slot);
    /* A DEVICE CHECKPOINT SLOT FOR A RESTORE. The pool is --checkpoint-slots deep for the whole
     * server, and it is full whenever as many sessions have run since; a session restored then
     * would get its blocks back and replay every recurrent layer from an older point, because its
     * state has nowhere to land. One slot is freed by detaching the least recently used snapshot
     * the tiers hold a copy of -- which the next restore of THAT session brings back -- and never
     * one whose only copy is the slot, and never one a lookup has handed out since the step began:
     * that request reads the slot in this step's forward, and a restore writes it before the
     * forward runs. True when a slot was freed. */
    bool free_checkpoint_for_restore();
    /* A step's admissions begin. The snapshots the last step's lookups handed out have been read
     * by its forward, which committed before this step was planned. */
    void begin_step() { handed_.clear(); }
    /* The tiers lost it -- a capacity drop, a failed promotion, or an entry they were told to
     * forget. Erase the index entry; there is no device slot to give back. */
    void drop_snapshot(const BlockHash& key);
    /* How many snapshots the index holds whose bytes the tiers have. Reported rather than derived
     * because it is the number that says whether hybrid tiering is doing anything at all: a
     * session can move its whole chain off the device and still be re-prefilling its linear
     * layers from zero on every hit, and nothing else distinguishes the two. */
    int64_t snapshots_off_device() const;
    /* ---------------- linear checkpoints ---------------- */
    /* The next position strictly after `pos` at which the scheduler must split a chunk so that a
     * checkpoint lands on a boundary (spec §7.1, §7.3). */
    int64_t next_checkpoint_after(int64_t pos) const;
    bool    is_checkpoint(int64_t pos) const { return interval_ > 0 && pos > 0 && pos % interval_ == 0; }

    /* Reserve a snapshot slot for `session` at `pos`. Applies the retention policy when the slot
     * pool is exhausted, never evicting a session tip. RAD_E_FULL when even that is not enough. */
    int  reserve_checkpoint(uint64_t session, int64_t pos, const BlockHash& prefix,
                            int32_t* slot_out);
    /* The state named by `prefix` is now IN its slot. Until this is called the reservation is only
     * a promise and lookup() will not offer it -- see Ckpt::valid. */
    void commit_checkpoint(const BlockHash& prefix);
    /* Every checkpoint of a session, ascending. */
    std::vector<int64_t> checkpoints(uint64_t session) const;
    /* Apply the retention policy to one session now. Returns how many snapshots were released.
     * A session left holding nothing is forgotten -- see the note at the definition for why that
     * matters more than the bytes. */
    int64_t retain(uint64_t session);

    /* How many sessions the checkpoint index still knows about. Exposed because it is an
     * INVARIANT rather than a statistic: a session is a request id, so this must track the
     * sessions holding a live snapshot and not the number of requests the process has served. */
    int64_t sessions_tracked() const { return (int64_t)ck_by_session_.size(); }

    struct Stats {
        int64_t lookups = 0, block_hits = 0, block_misses = 0;
        int64_t ckpt_hits = 0, ckpt_misses = 0;
        int64_t evictions = 0, ckpt_evictions = 0;
        /* Entries whose VRAM copy was given up with their host copy kept. Not evictions: nothing
         * was lost, and a counter that mixed the two would read a working tier as a thrashing
         * cache. */
        int64_t drops = 0;
    };
    const Stats& stats() const { return stats_; }

private:
    struct Entry {
        std::vector<int32_t> blocks;    /* one per cached group, in groups_ order */
        std::vector<int32_t> tokens;    /* the block's token ids, compared on hit */
        /* ON THE LRU EXACTLY WHILE THE CACHE HOLDS THE BLOCKS. An entry whose bytes are only off
         * the card owns no block, so it has nothing an eviction could free, and walking past it
         * under pressure would erase the tier's copy to make room it never occupied. `lru` is
         * valid only while this is set. */
        bool in_lru = false;
        bool drop_wait = false;         /* in drop_wait_: clean, but a sequence still reads it */
        std::list<BlockHash>::iterator lru;
        std::vector<int32_t> hot_units; /* prefix-aware prefetch: experts needed by this prefix */
    };
    struct Ckpt {
        uint64_t  session = 0;
        int64_t   pos = 0;
        BlockHash prefix;
        int32_t   slot = -1;
        uint64_t  used = 0;
        /* HAS THE STATE ACTUALLY BEEN WRITTEN INTO THE SLOT YET. reserve_checkpoint runs when the
         * scheduler PLANS the chunk; the copy happens after the step that computes the state. A
         * lookup between those two points would hand out a slot holding the previous tenant's
         * history, which is the exact failure this whole mechanism exists to prevent. */
        bool      valid = false;
        /* ARE THE BYTES STILL IN THE POOL. The idle tiers move a snapshot along with the blocks
         * it is keyed by, and `slot` is -1 for the whole time they hold it. Kept as its own flag
         * rather than inferred from the slot so that the two reasons a snapshot cannot be served
         * -- not written yet, and not here -- stay distinguishable in a debugger and in a report. */
        bool      off_device = false;
    };

    std::vector<BlockHash> chain_all(const std::vector<int32_t>& tokens,
                                     const std::vector<MMContentKey>& mm,
                                     int64_t n_blocks) const;
    void touch(const BlockHash& h) const;
    /* Retire one snapshot whatever tier holds its bytes: a device slot goes back to the pool, and
     * tiered bytes go back to the tiers. One place, because three call sites that each remembered
     * to check `off_device` would be three places to forget. The session index is the caller's,
     * since every caller is already rebuilding or erasing it. */
    void retire_ckpt(std::unordered_map<BlockHash, Ckpt, BlockHashHash>::iterator it);

    KVManager*  kv_ = nullptr;
    IdleTiers*  tiers_ = nullptr;
    bool        enabled_ = false;
    int64_t     block_size_ = 0;
    int64_t     interval_ = 0;
    std::vector<int32_t> groups_;         /* paged groups this cache indexes */
    bool        has_linear_ = false;

    mutable std::unordered_map<BlockHash, Entry, BlockHashHash> map_;
    mutable std::list<BlockHash> lru_;    /* front = oldest; resident entries only */
    /* CLEAN ENTRIES A SEQUENCE WAS STILL READING when drop_clean last looked, and the pool's count
     * of freed sequences at that moment. Only a sequence letting go of blocks can make one of them
     * droppable, so they are looked at again only when that count has moved. */
    std::vector<BlockHash> drop_wait_;
    std::vector<BlockHash> drop_scratch_;
    uint64_t    frees_seen_ = 0;
    mutable std::unordered_map<BlockHash, Ckpt, BlockHashHash> ck_by_prefix_;
    std::unordered_map<uint64_t, std::vector<BlockHash>> ck_by_session_;
    /* Snapshots lookup handed out since begin_step: not a restore's to take (free_checkpoint_for_restore). */
    mutable std::vector<BlockHash> handed_;
    mutable uint64_t clock_ = 0;
    std::unique_ptr<CheckpointPolicy> policy_;
    mutable Stats stats_;
    std::string report_;
};

}  /* namespace rad */
