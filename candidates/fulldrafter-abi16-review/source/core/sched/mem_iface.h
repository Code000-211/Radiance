/* mem_iface.h -- the seam between core/sched and core/mem.
 *
 * core/mem owns the concrete `KVManager` and `PrefixCache`. The scheduler binds to the pure-virtual
 * mirrors below rather than to those classes directly, for two reasons and neither is abstraction
 * for its own sake:
 *
 *   - it lets tests/sched_test.cpp drive the scheduler against a block pool it can make
 *     arbitrarily small, which is the only way to test admission refusal and preemption at all;
 *   - it costs one indirect call per SEQUENCE per step (at most max_seqs, a few hundred), not one
 *     per token, so it is off the part of the step path that is actually hot.
 *
 * `KVBinding` / `PCBinding` at the bottom adapt the concrete classes and require no change on
 * their side. They are templates so that they are only type-checked where the engine instantiates
 * them -- so a method core/mem does not provide is a compile error at exactly one line rather
 * than a comment nobody reads.
 *
 * The interface below mirrors core/mem/kv.h and core/mem/prefix.h.
 *
 * `CacheHit` comes from core/mem/prefix.h and is NOT redefined here. Two identical definitions of
 * a struct that crosses a component boundary is an ODR violation that costs nothing today and
 * becomes a silent layout mismatch the first time one side adds a field.
 */
#pragma once
#include <functional>
#include "mem/prefix.h"
#include "rad_core.h"
#include "mm/processor.h"

#include <vector>

namespace rad {

/* ------------------------------------------------------------------ the KV manager */
/* Keyed by SEQUENCE ID, which is the request id. The manager owns the block tables; Request::blocks
 * is not the source of truth and the scheduler does not read it. Two allocators disagreeing about
 * what a sequence owns is not a leak, it is the bug where a freed sequence's state slot is handed
 * to the next request while the first still reads it. */
struct IKVManager {
    virtual ~IKVManager() = default;

    virtual int  add_sequence(uint64_t seq) = 0;
    virtual bool has_sequence(uint64_t seq) const = 0;
    virtual void free_sequence(uint64_t seq) = 0;

    /* Grow the sequence's mapping to cover `n_tokens` in every group. RAD_E_FULL when a group's
     * free list is empty, which is a scheduling signal and not an error: the caller preempts a
     * victim and retries. */
    virtual int  ensure(uint64_t seq, int64_t n_tokens) = 0;

    /* Adopt blocks a prefix-cache hit found. `n_tokens` must be a whole multiple of the group's
     * block size -- a partial block is never shared, because the next token written into it would
     * corrupt the cache entry. */
    virtual int  adopt(uint64_t seq, int32_t group, const std::vector<int32_t>& blocks,
                       int64_t n_tokens) = 0;

    /* Free the blocks and mark the sequence for re-prefill. */
    virtual int  preempt(uint64_t seq) = 0;

    /* Could a sequence of this length EVER fit, in an empty pool? If not, preempting the whole
     * server would not help and the request is failed immediately rather than the engine
     * deadlocking on it (spec §17). */
    virtual bool can_ever_fit(int64_t n_tokens) const = 0;

    virtual const std::vector<int32_t>& block_table(uint64_t seq, int32_t group) const = 0;
    virtual int32_t state_slot(uint64_t seq, int32_t group) const = 0;

    /* THE WHOLE state_index ROW FOR A SEQUENCE, and how wide it is. One entry, normally: the
     * sequence's state. A group whose kernel asked for scratch beside the state names that too,
     * and the manager owns the mapping from column to physical slot because it owns the pool --
     * the scheduler stages the row and does not read it. Returns the width written, 0 for a
     * paged group. `max` bounds the write. */
    virtual int state_index_row(uint64_t seq, int32_t group, int32_t* out, int max) const = 0;
    virtual int64_t state_index_width(int32_t group) const = 0;

    virtual int64_t free_blocks(int32_t group) const = 0;
    virtual int64_t total_blocks(int32_t group) const = 0;
    /* What anything is HOLDING, as against what may be handed out right now. See
     * KVManager::held_blocks: on an elastic pool the two differ by whatever the cache has
     * released, and an occupancy figure built from the second reads an idle server as full. */
    virtual int64_t held_blocks(int32_t group) const = 0;

    /* THE SAME POOL IN BYTES. Blocks and tokens are both un-summable across groups -- a stateful
     * group has neither, and two paged groups share one token axis -- so bytes are the only
     * measure a reader can add up to "how much of the card is the KV cache holding". Asked of the
     * manager rather than derived here, because the manager owns the plan that knows whether a
     * group is counted in blocks or in states. */
    virtual int64_t pool_bytes(int32_t group) const = 0;
    virtual int64_t pool_bytes_used(int32_t group) const = 0;
    /* What the carve gave the group, which an elastic pool may have released part of. */
    virtual int64_t pool_bytes_carved(int32_t group) const = 0;

    /* HOW MANY TOKENS ONE BLOCK OF THIS GROUP COVERS, or 0 for a stateful group. The pool's
     * capacity in blocks is already here; the live request table wants it in TOKENS, because that
     * is the unit a request is measured in and the only one the two can be compared in. */
    virtual int64_t block_tokens(int32_t group) const = 0;

    /* SPECULATIVE REJECTION ROLLBACK (spec §10). A verify writes KV for 1 + n_spec
     * positions and the sampler accepts a prefix of them; the sequence's length must come back to
     * the accepted position and the blocks past it must go to the free list. `ensure` is monotone
     * and cannot express it.
     *
     * It is a slot-table edit and nothing else -- crucially it does NOT touch the linear or conv
     * groups, whose rollback is a change of read offset the kernels perform themselves. Without
     * this the pool leaks one block per rejected block-crossing, which is a slow leak and
     * therefore the worst kind. */
    virtual int  rollback(uint64_t seq, int64_t n_tokens) = 0;

    /* THE ABSOLUTE TOKEN POSITION that block_table(seq, group)[0] covers. Zero for a
     * FULL group and non-zero for a WINDOW group, whose blocks below the window are recycled: the
     * step batch's slot mapping is `(pos - first_block_pos) / block_size`, and with the offset
     * missing every sliding-window sequence writes its KV to the wrong slot the moment it passes
     * its window. KVManager::Seq already tracks this privately as `first_block_pos`. */
    virtual int64_t first_block_pos(uint64_t seq, int32_t group) const = 0;

    /* THE ENTRIES OF block_table(seq, group) THAT CHANGED since the previous call: everything
     * below the returned index is unchanged, and INT64_MAX says nothing changed. Reading clears
     * it, so the batch builder is its one caller: it keeps the device's copy of a table and
     * stages only what moved. 0 is always a correct answer -- it restages the whole table. */
    virtual int64_t take_table_changes(uint64_t seq, int32_t group) const = 0;
};

/* ------------------------------------------------------------------ the prefix cache */
struct IPrefixCache {
    virtual ~IPrefixCache() = default;

    /* Longest cached prefix, plus the checkpoint covering the linear state for a prefix of it.
     * `blocks_out` receives the block ids per cached group, in cached_groups() order, ready for
     * IKVManager::adopt. */
    /* GIVE BACK WHATEVER THE TIERS HOLD FOR THIS PROMPT, before the lookup that would otherwise
     * stop at it. `Wait` when part of it is still being read off disk: the request is not
     * admitted this step -- the ones behind it may be -- and is asked again on the next. Admitting
     * it anyway would stop its hit at the first entry not yet read, and the hit length is what
     * its prefill is chunked against, so the same prompt would run different shapes depending on
     * how far a read had got.
     *
     * The scheduler calls it and must not know what a tier is: a restore allocates blocks and
     * runs a device scatter, so it is the engine's to perform and is installed as a callable
     * rather than reached through this header. A failure is not an error here -- it leaves a
     * shorter prefix, and a shorter prefix is always safe. */
    enum class Restore { Done, Wait };
    virtual Restore restore(const Request&) { return Restore::Done; }
    /* A step's admissions begin: what the last step's lookups handed out has been read. */
    virtual void begin_step() {}

    virtual CacheHit lookup(const Request& r,
                            std::vector<std::vector<int32_t>>* blocks_out) const = 0;
    virtual const std::vector<int32_t>& cached_groups() const = 0;

    /* PUBLISH what this sequence computed, so the next request sharing the prefix can adopt it.
     * Without this the cache is only ever read: every lookup misses, `cached_tokens` is always
     * zero, and a chat client re-prefills its whole history on every turn, which for a long
     * conversation is seconds a message. `blocks[i]` is the block table of cached_groups()[i], and
     * `n_tokens` how many of them are backed. Takes a KV reference on every newly indexed block,
     * so it must be called BEFORE the sequence's own blocks are released. */
    virtual std::vector<int32_t> current_hot_units() const { return {}; }
    virtual int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
                       int64_t n_tokens) {
        return insert(r, blocks, n_tokens, {});
    }
    virtual int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
                       int64_t n_tokens, const std::vector<int32_t>& hot_units) {
        (void)hot_units;
        return insert(r, blocks, n_tokens);
    }

    /* Release the least recently used entries, and with them their KV references. The cache holds
     * blocks no live sequence needs, so this is what the scheduler tries BEFORE preempting a
     * running request: a cached prefix is a saving, a running request is work already done. */
    virtual int64_t evict(int64_t n_entries) = 0;

    /* FORGET EVERYTHING, wherever it is. Not a loop over evict(): with idle tiers an eviction moves
     * an entry whose host copy has landed off the card and keeps it, so a reset built out of
     * evictions would leave every such conversation behind. Returns how many entries went. */
    virtual int64_t reset() {
        int64_t dropped = 0;
        for (int guard = 0; guard < 65536; ++guard) {
            const int64_t n = evict(1024);
            if (n <= 0) break;
            dropped += n;
        }
        return dropped;
    }

    /* HOW MANY ENTRIES THE CACHE HAS THROWN AWAY, which is the leading indicator of the failure
     * mode that matters operationally: once the pool is utilised past the reserve the scheduler
     * keeps free (a quarter of it), the cache begins evicting prefixes that are about to be
     * reused, and an agent's next turn re-prefills its whole context -- turning a cached turn of
     * a few seconds into a full prefill of a very long prompt. `prefix_hit_rate` only falls AFTER
     * that has happened; this count rises first. */
    virtual int64_t evictions() const = 0;
    virtual int64_t ckpt_evictions() const = 0;

    /* WHAT IT IS HOLDING ON ITS OWN RIGHT NOW, in tokens of context, as against the counters
     * above which are cumulative. A finished turn's blocks stay held so the next turn can reuse
     * them, so on any server serving conversations MOST of a full KV pool is this rather than the
     * live sequences -- and with one number for the pool there is no way to see that, which makes
     * a working cache indistinguishable from a leak.
     *
     * ON ITS OWN is the whole of it. A cached block a running sequence is also reading is not
     * memory the cache is keeping back from anybody, and counting those would make a single 200K
     * request look like a cache refusing to release 200K. */
    virtual int64_t held_tokens() const = 0;

    /* The next position strictly after `pos` at which the scheduler must split a chunk so that a
     * checkpoint lands on a boundary. */
    virtual int64_t next_checkpoint_after(int64_t pos) const = 0;

    /* Reserve the snapshot slot the scheduler is about to WRITE. Retention is budgeted (§7.3:
     * ~96 MiB a snapshot, so 64 of them is 6 GiB), which makes RAD_E_FULL an ordinary condition
     * and not a failure -- the scheduler simply does not write a checkpoint at that boundary and
     * the next hit replays from the previous one. */
    virtual int reserve_checkpoint(const Request& r, int64_t pos, int32_t* slot_out) = 0;

    /* The state that reservation named is now IN its slot. A reservation is made when the chunk is
     * PLANNED and the state exists only after the step that computes it, so a lookup in between
     * would adopt a slot holding the previous tenant's history. Until this is called, the
     * reservation is a promise and lookup will not offer it. */
    virtual void commit_checkpoint(const Request& r, int64_t pos) = 0;
};

/* ------------------------------------------------------------------ block geometry */
/* Everything the step batch needs to turn a token position into a KV slot, read off the declared
 * program rather than off the manager: block size is KVGroupInfo::block_size, which the build
 * component already took from the resolved attention kernel's constraints (§7.2). Duplicating it
 * through a second interface would be a second place for it to be wrong. */
struct KVGeom {
    struct G {
        int     kind = RAD_KV_FULL;
        int64_t block_size = 1;
        int64_t window = 0;
        bool    paged = true;
        /* Columns in this group's state_index row. The MANAGER decides it -- it depends on what
         * the group's kernel wants beside the state -- so KVGeom::of leaves it at 1 and the
         * scheduler fills it in from the configured pool. */
        int64_t sidx_width = 1;
    };
    std::vector<G> g;

    static KVGeom of(const std::vector<KVGroupInfo>& groups) {
        KVGeom k;
        k.g.reserve(groups.size());
        for (const KVGroupInfo& gi : groups) {
            G e;
            e.kind = gi.decl.kind;
            e.paged = (e.kind == RAD_KV_FULL || e.kind == RAD_KV_WINDOW);
            e.block_size = gi.block_size > 0 ? gi.block_size : 1;
            e.window = gi.decl.window;
            k.g.push_back(e);
        }
        return k;
    }

    /* Flat KV slot for absolute token position `pos`, given the sequence's block table for this
     * group and the position its first block covers. */
    int32_t slot(int gi, const std::vector<int32_t>& bt, int64_t first_pos, int64_t pos) const {
        const G& e = g[(size_t)gi];
        int64_t bi = (pos - first_pos) / e.block_size;
        if (bi < 0 || bi >= (int64_t)bt.size()) return -1;
        return (int32_t)(bt[(size_t)bi] * e.block_size + (pos % e.block_size));
    }
};

/* ------------------------------------------------------------------ binding the concrete classes */
/* Templates, so nothing here is type-checked until the engine instantiates it against the real
 * classes. That instantiation is where a mismatch with core/mem is reported. */
template <class KV>
struct KVBinding final : IKVManager {
    KV* m;
    explicit KVBinding(KV* k) : m(k) {}

    int  add_sequence(uint64_t s) override          { return m->add_sequence(s); }
    bool has_sequence(uint64_t s) const override    { return m->has_sequence(s); }
    void free_sequence(uint64_t s) override         { m->free_sequence(s); }
    int  ensure(uint64_t s, int64_t n) override     { return m->ensure(s, n); }
    int  adopt(uint64_t s, int32_t g, const std::vector<int32_t>& b, int64_t n) override {
        return m->adopt(s, g, b, n);
    }
    int  preempt(uint64_t s) override               { return m->preempt(s); }
    bool can_ever_fit(int64_t n) const override     { return m->can_ever_fit(n); }
    const std::vector<int32_t>& block_table(uint64_t s, int32_t g) const override {
        return m->block_table(s, g);
    }
    int32_t state_slot(uint64_t s, int32_t g) const override { return m->state_slot(s, g); }
    int state_index_row(uint64_t s, int32_t g, int32_t* o, int n) const override {
        return m->state_index_row(s, g, o, n);
    }
    int64_t state_index_width(int32_t g) const override { return m->state_index_width(g); }
    int64_t free_blocks(int32_t g) const override   { return m->free_blocks(g); }
    int64_t total_blocks(int32_t g) const override  { return m->total_blocks(g); }
    int64_t held_blocks(int32_t g) const override   { return m->held_blocks(g); }
    int64_t pool_bytes(int32_t g) const override      { return m->pool_bytes(g); }
    int64_t pool_bytes_used(int32_t g) const override { return m->pool_bytes_used(g); }
    int64_t pool_bytes_carved(int32_t g) const override { return m->pool_bytes_carved(g); }
    int64_t block_tokens(int32_t g) const override {
        const KVGroupPlan* p = m->plan(g);
        return p && p->paged() ? p->block_size : 0;
    }
    int     rollback(uint64_t s, int64_t n) override { return m->rollback(s, n); }
    int64_t first_block_pos(uint64_t s, int32_t g) const override {
        return m->first_block_pos(s, g);
    }
    int64_t take_table_changes(uint64_t s, int32_t g) const override {
        return m->take_table_changes(s, g);
    }
};

/* The prefix cache's own lookup takes tokens and content-hashed multimodal keys rather than a
 * Request, because the placeholder ids inside a media run are identical for every picture: hashing
 * the ids alone would serve one request another's image. Each run is keyed by its item's content
 * key (core/mm, taken over the processed patches) and its segment, so the same picture at the same
 * place hits -- and skips the encoder along with the prefill -- and two pictures never alias. */
inline void media_keys(const Request& r, std::vector<MMContentKey>* out) {
    out->clear();
    if (!r.media) return;
    const mm::PromptMedia& pm = *r.media;
    for (const mm::RopeLayout::Run& run : pm.rope.runs) {
        const mm::Item& it = *pm.items[(size_t)run.item].item;
        MMContentKey k;
        k.tok_begin = run.tok;
        k.tok_end = run.tok + run.n;
        k.hash = it.key_hi ^ (it.key_lo * 0x9e3779b97f4a7c15ull) ^
                 ((uint64_t)(uint32_t)run.row0 * 0xc2b2ae3d27d4eb4full);
        out->push_back(k);
    }
}

template <class PC>
struct PCBinding final : IPrefixCache {
    PC* m;
    /* Scratch for a request's media keys, reused so a text request's lookup allocates nothing. */
    mutable std::vector<MMContentKey> mm_;
    mutable std::vector<int32_t> blk_;
    explicit PCBinding(PC* p) : m(p) {}

    /* Installed by the engine for prefix-aware expert prefetching; empty otherwise. */
    std::function<std::vector<int32_t>()> hot_units_fn;
    std::function<void(const Request&, const std::vector<int32_t>&)> on_prefix_hit;

    std::vector<int32_t> current_hot_units() const override {
        return hot_units_fn ? hot_units_fn() : std::vector<int32_t>{};
    }

    CacheHit lookup(const Request& r,
                    std::vector<std::vector<int32_t>>* out) const override {
        media_keys(r, &mm_);
        CacheHit h = m->lookup(r.prompt, mm_, out);
        if (h.n_attn_tokens > 0 && !h.prefetch_units.empty() && on_prefix_hit) {
            on_prefix_hit(r, h.prefetch_units);
        }
        return h;
    }

    /* Installed by the engine when the idle tiers are on; empty otherwise. */
    std::function<Restore(const Request&)> restore_fn;
    Restore restore(const Request& r) override {
        return restore_fn ? restore_fn(r) : Restore::Done;
    }
    void begin_step() override { m->begin_step(); }
    const std::vector<int32_t>& cached_groups() const override { return m->cached_groups(); }

    /* prompt ++ output, for the same reason reserve_checkpoint chains both: the next turn of a
     * conversation sends the reply back as part of its prompt, so a cache that indexed only the
     * prompt would stop hitting at the first assistant token and re-prefill everything after it. */
    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens) override {
        return insert(r, blocks, n_tokens, {});
    }
    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens, const std::vector<int32_t>& hot_units) override {
        std::vector<int32_t> toks;
        toks.reserve(r.prompt.size() + r.output.size());
        toks.insert(toks.end(), r.prompt.begin(), r.prompt.end());
        toks.insert(toks.end(), r.output.begin(), r.output.end());
        if (n_tokens > (int64_t)toks.size()) n_tokens = (int64_t)toks.size();
        /* r.id is the session, and it is the SAME id reserve_checkpoint keys its snapshots by --
         * so the tiers' session view and the checkpoint index are talking about one thing. */
        media_keys(r, &mm_);
        return m->insert(r.id, toks, mm_, blocks, n_tokens, hot_units);
    }
    int64_t evict(int64_t n) override { return m->evict_lru(n); }
    int64_t reset() override {
        const int64_t n = m->size();
        m->clear();
        return n;
    }
    int64_t evictions() const override { return m->stats().evictions; }
    int64_t ckpt_evictions() const override { return m->stats().ckpt_evictions; }
    int64_t held_tokens() const override { return m->unshared_tokens(); }

    int64_t next_checkpoint_after(int64_t p) const override { return m->next_checkpoint_after(p); }
    int reserve_checkpoint(const Request& r, int64_t pos, int32_t* slot_out) override {
        /* The cache keys a snapshot by the CHAINED HASH of the prefix it covers, and the chain is
         * built here rather than inside the cache because the cache does not hold the tokens --
         * that is the whole reason this interface passes a Request. It must be the SAME chain
         * lookup() walks, or a checkpoint reserved at `pos` and a later hit at `pos` name two
         * different entries and the linear cache silently never hits.
         *
         * The prefix is prompt ++ output: a checkpoint written during decode covers tokens the
         * request has generated, and keying it on the prompt alone would alias every continuation
         * of the same prompt onto one snapshot -- which is one session's recurrent state served to
         * another. Whole blocks only; a partial block is never keyed, because the next token
         * written into it would change what the key stands for. */
        return m->reserve_checkpoint(r.id, pos, prefix_hash(r, pos), slot_out);
    }

    void commit_checkpoint(const Request& r, int64_t pos) override {
        m->commit_checkpoint(prefix_hash(r, pos));
    }

private:
    /* The chain reserve_checkpoint keys on, factored out so commit cannot drift from it: two
     * copies of this walk that agreed when written would be two copies that could stop agreeing,
     * and the symptom would be a checkpoint that is written and never found. */
    BlockHash prefix_hash(const Request& r, int64_t pos) const {
        media_keys(r, &mm_);
        const int64_t bs = m->block_size();
        BlockHash h{};
        if (bs <= 0) return h;
        const int64_t np = (int64_t)r.prompt.size(), no = (int64_t)r.output.size();
        const int64_t have = np + no;
        const int64_t n_blocks = (pos < have ? pos : have) / bs;
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t base = b * bs;
            const int32_t* ptr = nullptr;
            if (base + bs <= np) {
                ptr = r.prompt.data() + base;
            } else if (base >= np) {
                ptr = r.output.data() + (base - np);
            } else {
                if ((int64_t)blk_.size() < bs) blk_.resize((size_t)bs);
                for (int64_t j = 0; j < bs; ++j) {
                    const int64_t k = base + j;
                    blk_[(size_t)j] = k < np ? r.prompt[(size_t)k] : r.output[(size_t)(k - np)];
                }
                ptr = blk_.data();
            }
            h = chain_block_hash(h, ptr, (int32_t)bs, mm_.data(), (int32_t)mm_.size(), base);
        }
        return h;
    }
};

}  /* namespace rad */
