/* kv.h -- the KV block manager (spec §7.2).
 *
 * Paged, per KV group, each group with its OWN block geometry, its own pool and its own slot
 * mapping. Block size comes from KVGroupInfo::block_size, which the build component took off the
 * resolved attention kernel's constraints -- never a core constant. A core that picked its own
 * block size would be a core that has to be edited when a kernel changes.
 *
 * Hybrid models are the normal case, not an extension. Four state kinds:
 *
 *   RAD_KV_FULL    paged, per-token, growing
 *   RAD_KV_WINDOW  paged, per-token, bounded by the window -- blocks below the window are recycled
 *   RAD_KV_LINEAR  per-sequence, FIXED size: the GDN recurrent state
 *   RAD_KV_CONV    per-sequence, FIXED size: a rolling window of conv_width-1 + n_spec entries
 *
 * The last two are not paged the same way, and they are allocated, mapped and freed BY THIS
 * MANAGER anyway, because otherwise two allocators disagree about what a sequence owns. That
 * disagreement is not a memory leak, it is a correctness bug: the scheduler frees a sequence and
 * one of the two allocators keeps handing its state slot to the next request.
 */
#pragma once
#include "mem/pools.h"
#include "rad_core.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* ------------------------------------------------------------------ group sizing */
/* vLLM slices a hybrid model's layers into equal-sized groups and PADS each bundle up to a
 * multiple of that size -- and a padding layer is allocated real memory. On a 48 linear / 16 full
 * / 5 drafter model its rule ("group size = the smallest bundle") picks 5, which divides neither
 * 48 nor 16, so it allocates 15 groups x 5 = 75 slots for 69 real layers, and rounds the
 * per-request requirement up once per group -- fifteen round-ups on every request, which is where
 * most of the loss actually is. Replacing that rule with a least-waste search recovers a
 * substantial fraction of the KV tokens at a byte-identical budget.
 *
 * Radiance does not need the search, because it does not have the constraint the search works
 * around. A KV group here is one state kind with one geometry and ITS OWN POOL (spec §7.2), so
 * there is no uniform page size to pad up to and no reason to subdivide a bundle: every state
 * kind is exactly one group covering exactly its own layers. Padding is structurally zero and the
 * group count is structurally minimal.
 *
 * This struct exists so `--debug-placement` can print that claim next to what vLLM would have
 * done, and so the test can assert it rather than trusting the comment. */
struct BundleComparison {
    std::vector<int64_t> bundles;        /* layers per state kind, as declared */
    int64_t stock_group_size = 0;        /* vLLM's rule: the smallest bundle (with the 1.5x escape) */
    int64_t stock_groups = 0;
    int64_t stock_slots = 0;             /* allocated layer-slots, padding included */
    int64_t patched_group_size = 0;      /* the least-waste group size for the same rule */
    int64_t patched_groups = 0;
    int64_t patched_slots = 0;
    int64_t our_groups = 0;              /* == bundles.size() */
    int64_t our_slots = 0;               /* == sum(bundles); padding is zero by construction */
    std::string report() const;
};
BundleComparison compare_bundling(const std::vector<int64_t>& bundles);

/* ------------------------------------------------------------------ per-group plan */
struct KVGroupPlan {
    int32_t     index = -1;            /* position in Program::kv_groups */
    std::string name;
    int         kind = RAD_KV_FULL;
    int64_t     n_layers = 0;
    int64_t     n_head_kv = 1;

    int64_t     block_size = 0;        /* tokens per block, from the resolved kernel */
    int64_t     bytes_per_block = 0;   /* covers EVERY layer bound to the group -- see the note
                                        * on bundling above: one page, one bundle, no padding */
    int64_t     bytes_per_state = 0;   /* LINEAR/CONV: per sequence, every layer */
    int64_t     conv_slots = 0;        /* CONV: conv_width - 1 + n_spec */
    /* WINDOW: the per-sequence block cap, and the part of it that is not attention history.
     * `window_reserve_tokens` is the scheduler's speculative forward reach (see
     * speculative_window_reserve in rad_core.h) that the cap has to hold ON TOP of the window,
     * because a verify reserves its draft rows before anyone knows which of them commit. The
     * block count is `window_blocks_for(window, block_size, window_reserve_tokens)`, and the
     * step batch stages its block-table rectangle from the same expression -- a table staged one
     * row short silently truncates the NEWEST blocks, which every kernel reads as the current
     * context. */
    int64_t     window_reserve_tokens = 0;
    int64_t     window_blocks = 0;

    int64_t     n_blocks = 0;          /* paged groups: blocks carved from the pool */
    int64_t     n_states = 0;          /* LINEAR/CONV: n_seq_states * state_copies */
    /* HOW MANY PHYSICAL SLOTS ONE LOGICAL STATE COSTS. One, normally: a sequence owns a state.
     * Under speculation a linear kernel needs somewhere to put what it cannot commit yet -- it
     * has run 1 + n_spec tokens through the recurrence before anyone knows which survive -- and
     * the scheme libr4d uses keeps the committed state in one slot and the per-token replay
     * factors in a second, then folds the accepted prefix in at the start of the next step. Two
     * slots a sequence, not 1 + n_spec: the factors are a few hundred floats a head against a
     * 128x128 state, so they fit inside a slot with room to spare and the pool grows by 2x
     * rather than by 9x. Copy k of logical slot s is the physical slot s + k * n_seq_states.
     * One again under a bound tree (KVGroupInfo::tree_bound): its kernels read and commit copy 0
     * and keep their factors in arena buffers. */
    int64_t     state_copies = 1;
    int64_t n_seq_states() const { return state_copies > 0 ? n_states / state_copies : n_states; }
    int64_t     pool_offset = 0;
    int64_t     pool_bytes = 0;
    void*       base = nullptr;

    bool paged() const { return kind == RAD_KV_FULL || kind == RAD_KV_WINDOW; }
    bool stateful() const { return kind == RAD_KV_LINEAR || kind == RAD_KV_CONV; }

    /* THE POOL IS LAYER-MAJOR, and that is not a free choice -- the ABI already made it. A
     * RAD_OPK_KV operand carries the LAYER as its offset and the core resolves it as
     * `base + layer * layer_stride` (RadOperand in rad_runtime.h, KVPoolBinding in
     * core/runtime/ctx.h), because an attention kernel takes ONE base pointer for the layer it is
     * running and indexes inside it with the block table and the slot mapping. Interleaving the
     * layers inside a block would mean every kernel needed the layer count and the bundle stride
     * as well, to reach the same bytes.
     *
     * So layer i owns [base + i*layer_stride, + layer_stride): n_blocks blocks of
     * `bytes_per_block / n_layers` for a paged group, n_states states of
     * `bytes_per_state / n_layers` for a stateful one. The totals are identical either way, which
     * is why sizing does not care and addressing does. */
    int64_t layer_bytes_per_block() const {
        return n_layers > 0 ? bytes_per_block / n_layers : bytes_per_block;
    }
    int64_t layer_bytes_per_state() const {
        return n_layers > 0 ? bytes_per_state / n_layers : bytes_per_state;
    }
    int64_t layer_stride() const {
        return paged() ? n_blocks * layer_bytes_per_block() : n_states * layer_bytes_per_state();
    }
};

/* A copy the caller must execute on its copy stream before the destination block is read. The
 * manager does not issue device copies itself: it has no stream, and a memcpy on the step path is
 * a synchronisation the engine spends its whole design avoiding (spec §5.4). */
struct BlockCopy {
    int32_t group = -1;
    int32_t src_block = -1;
    int32_t dst_block = -1;
};

/* A STATE SLOT THAT STILL HOLDS THE PREVIOUS TENANT'S HISTORY. The manager names the work and the
 * engine issues it, for the same reason BlockCopy exists: no stream here, and a synchronisation on
 * the step path is what this design spends itself avoiding (spec §5.4).
 *
 * WHY A RECURRENT SLOT HAS TO BE CLEARED AND A PAGED ONE DOES NOT. A paged sequence reads only the
 * blocks its own block table names, so a recycled block is unreachable until it is written. A
 * recurrent sequence has no table: `gdn_chunk_scan` reads `h0` out of the slot and there is no
 * operand in its schema that says "this sequence has none". An uncleared slot IS the last
 * sequence's entire compressed history, so the second request a server answers silently continues
 * the first: the same prompt gives a different distribution on a fresh process than it does after
 * another request has run.
 *
 * The conv group carries `has_init` and does not need this. It is cleared anyway: two rules about
 * one pool is how one of them rots. */
struct StateClear {
    int32_t group = -1;
    int32_t slot  = -1;
};

/* ------------------------------------------------------------------ manager */
class KVManager {
public:
    /* Sizes every group out of one pool and refuses, by name and by how much, if the fixed costs
     * alone do not fit. The split between the paged groups is not a policy: they are sized to
     * exhaust at the same total token count, and any other split strands memory in whichever
     * group outlasts the others.
     *
     * Both budgets come from the operator and neither is derived here: the byte budget is whatever
     * `kv_pool` still has (which is --vram-kv-mib), and the linear-snapshot count is
     * Config::checkpoint_slots (--checkpoint-slots). A checkpoint is ~96 MiB on a Qwen3-Next and
     * 64 of them is 6 GiB, so how many to retain is a budget decision and therefore not this
     * component's to make (spec §6, §7.3). */
    int configure(const std::vector<KVGroupInfo>& groups, const Config& cfg, Pool& kv_pool);

    /* configure(), split at the seam the budget resolver needs. plan_groups() sizes every group
     * from the declaration and the run configuration -- no pool, no bytes -- and ceiling() then
     * says what a FULL cache for `ctx` tokens on max_seqs sequences would cost. Pools are
     * allocated before carve() runs, so a derived --vram-kv-mib has to be decided from these two
     * and nothing else (core/mem/vram_budget.cpp). carve() spends the pool it is given. */
    int     plan_groups(const std::vector<KVGroupInfo>& groups, const Config& cfg);
    int64_t ceiling(int64_t ctx) const;
    /* The part of every ceiling that is not context: recurrent and conv state for max_seqs
     * sequences, the checkpoint slots and the padding. Set by plan_groups(). */
    int64_t fixed_bytes() const { return fixed_; }
    int     carve(Pool& kv_pool);

    const std::vector<KVGroupPlan>& plans() const { return plans_; }
    const KVGroupPlan* plan(int32_t group) const;
    int32_t n_groups() const { return (int32_t)plans_.size(); }
    const BundleComparison& bundling() const { return bundling_; }
    std::string report() const;

    /* ---------------- sequences ---------------- */
    int  add_sequence(uint64_t seq);
    bool has_sequence(uint64_t seq) const;
    void free_sequence(uint64_t seq);
    /* HOW MANY TIMES A SEQUENCE HAS LET GO OF ITS BLOCKS -- freed or preempted -- over the life of
     * the manager. A block's reference count falls for no other reason the prefix cache cannot
     * see, so this is what tells it a block it found shared may no longer be. */
    uint64_t seq_frees() const { return seq_frees_; }

    /* Grow the sequence's mapping to cover `n_tokens` in every group. Unshares a shared partial
     * block first (copy-on-write) and appends the copy request to the pending list.
     * RAD_E_FULL when a group's free list is empty -- the caller preempts and retries. */
    int  ensure(uint64_t seq, int64_t n_tokens);

    /* Adopt blocks a prefix-cache hit found. `n_tokens` must be a whole multiple of the group's
     * block size: a partial block is never shared, because the next token written into it would
     * corrupt the cache entry. Bumps each block's refcount. */
    int  adopt(uint64_t seq, int32_t group, const std::vector<int32_t>& blocks, int64_t n_tokens);

    /* Fork: the child shares every WHOLE block of the parent and gets its own copy of the
     * partial one on first write (copy-on-write, spec §7.3). Stateful groups cannot be shared --
     * a recurrent state is one object, not a chain -- so the child gets its own slot and the
     * caller must snapshot-restore into it. */
    int  fork(uint64_t parent, uint64_t child);

    std::vector<BlockCopy> take_pending_copies();

    /* The prefix cache's handle on a block. It holds a reference on every block it indexes, and
     * that reference is the invariant that makes a hit safe: a block named by the cache is never
     * in the free list, so it can never be handed to another sequence underneath a cache entry.
     * Nothing else in the engine may call these -- the block table is the manager's property. */
    /* TAKE A FREE BLOCK, with no sequence attached to it yet. The idle-session tiers need this
     * and nothing else does: a restore has bytes to put somewhere before any sequence has asked
     * for them, and the prefix cache takes the reference afterwards. Every other caller reaches
     * blocks through a sequence, which is why grab_block itself stays private -- this is the one
     * documented way to hold a block that no Seq owns, and the holder is responsible for
     * release_block if the restore then fails. */
    int  reserve_block(int32_t group, int32_t* out);
    /* THE SAME FOR A WHOLE CHAIN AT ONCE: up to `n` blocks appended to `out`, and how many. The
     * loan is recalled for all of them before the first is taken, because a restore is sized
     * before it starts and one recall a block would drain the slab a buffer at a time -- the
     * reserve above stops at whatever the cache kept in hand. */
    int64_t reserve_blocks(int32_t group, int64_t n, std::vector<int32_t>* out);
    /* The same, for a block another rank already chose. See the definition. */
    int  reserve_specific_block(int32_t group, int32_t block);
    /* The cache's reference on a block a sequence already holds. The block is marked as the
     * cache's for cache_only_blocks(). */
    int  retain_block(int32_t group, int32_t block);
    /* A reserved block's reference handed to the cache as it stands: marked, not counted again. */
    void cache_reserved_block(int32_t group, int32_t block);
    /* Lets go of the cache's reference on a marked block and of a reservation's on any other. */
    void release_block(int32_t group, int32_t block);
    int32_t block_refcount(int32_t group, int32_t block) const;
    /* BLOCKS ONLY THE CACHE HOLDS: marked, and at a reference count of one. This is what an
     * eviction would give back, and what a dashboard reports as cache rather than as context.
     * Kept current at every reference-count change so that reading it costs nothing: the figure
     * is read while the scheduler's lock is held, and a walk of the cache to compute it would
     * stall the step for as long as it ran. */
    int64_t cache_only_blocks(int32_t group) const;

    const std::vector<int32_t>& block_table(uint64_t seq, int32_t group) const;
    int32_t state_slot(uint64_t seq, int32_t group) const;
    int     state_index_row(uint64_t seq, int32_t group, int32_t* out, int max) const;
    int64_t state_index_width(int32_t group) const;

    /* Slots handed to a sequence since the last drain, oldest first. The engine zeroes each on the
     * compute stream before the step that first reads it and then calls clear_state_clears(). */
    const std::vector<StateClear>& state_clears() const { return clears_; }
    void clear_state_clears() { clears_.clear(); }
    int64_t seq_tokens(uint64_t seq) const;

    /* ---------------- accounting ---------------- */
    int64_t free_blocks(int32_t g) const;

    /* HOW MANY BLOCKS OF A GROUP ANYTHING IS HOLDING. free_blocks() is what may be handed out
     * RIGHT NOW, which while the cache is lending is bounded by the live mark and not by the
     * carve, so `carve - free` counts every lent block as occupied -- an idle server reading 97%
     * full. The occupancy question wants this instead: a lent block is neither held nor
     * unavailable, because the loan is recalled on demand up to the carve. */
    int64_t held_blocks(int32_t g) const;

    int64_t total_blocks(int32_t group) const;
    int64_t free_states(int32_t group) const;

    /* THE POOL IN BYTES, WHICH IS THE ONLY UNIT THE GROUPS CAN BE ADDED IN. Blocks cannot be:
     * a paged group and a stateful one count different objects, and two paged groups share ONE
     * token axis -- a token of context takes a slot in each of them -- so summing either measure
     * reports a pool several times the size of the one the card holds. Bytes are disjoint VRAM
     * and add up.
     *
     * The used figure is what is HELD rather than what is reachable: a block a sequence owns and
     * a block only the prefix cache still references both count, because both are VRAM the next
     * request cannot be given. A block stops counting the moment it reaches the free list,
     * whether it got there because its sequence ended, because the cache evicted it, or because
     * the idle tiers copied it to host memory and released it. */
    int64_t pool_bytes(int32_t group) const;
    int64_t pool_bytes_used(int32_t group) const;
    /* THE CARVE, which the cache may not be keeping. pool_bytes() is what the cache keeps for
     * itself right now; this is what the operator budget gave it to address, and the difference
     * between them is what it has lent to the expert slab. */
    int64_t pool_bytes_carved(int32_t group) const;
    int64_t blocks_for(int32_t group, int64_t n_tokens) const;

    /* Could a sequence of this length EVER fit, in an empty pool? If not, preempting the whole
     * server would not help and the request is failed immediately rather than the engine
     * deadlocking on it (spec §17). */
    bool can_ever_fit(int64_t n_tokens) const;

    /* ---------------- the loan ----------------
     *
     * THE PAGED GROUPS LEND THE EXPERT SLAB WHAT NOTHING HOLDS, AT THE ADDRESSES THEY ALREADY
     * OWN. The carve decides every address once and backs all of it before anything runs, and
     * nothing here ever unmaps it. What moves is only how many of a group's blocks the cache may
     * hand out: the blocks above that mark, in every layer's stripe, are where the slab keeps
     * expert units -- units the MoE GEMM would otherwise read across the link on every step that
     * routes to them. No memory is allocated or released in either direction, so no driver call
     * is made and what the card reports free never changes.
     *
     * THE LOAN IS STATED IN TOKENS, counted down from the top of the carve. The FULL groups are
     * sized to run out at the same total token count (carve()), so one number is the same depth
     * in each of them: a loan of T tokens is the top floor(T / block_size) blocks of every FULL
     * group. A WINDOW group is sized per sequence rather than on that axis and is never lent.
     *
     * A BLOCK IS NOT LENT WHILE ANYTHING HOLDS IT, and a held block is a block a sequence owns OR
     * one the prefix cache still indexes. So a loan cannot evict anything: it is a claim on what
     * is free. Everything that makes a block free -- a sequence ending, an eviction, an idle-tier
     * demotion to host memory or disk -- makes it lendable, and the min-heap free list is what
     * keeps the free blocks at the top, where a loan can reach them.
     *
     * THE BUFFER IS DERIVED AND NOT STATED. It is the free blocks the cache keeps in hand so that
     * a request finds room without a recall, and the right size for it is a property of the
     * workload rather than a number an operator can know: one prefill chunk -- the most KV a
     * single step can consume -- or the largest recent shortfall, whichever is larger. The second
     * term decays, so an idle server settles at the floor and a ramping one has the buffer
     * leading the ramp. */
    int64_t buffer_bytes() const;

    /* Is this group lent from? The FULL paged groups, and only those. */
    bool    lendable(int32_t group) const;

    /* HOW MUCH THE CACHE CAN LEND RIGHT NOW, in tokens: the smallest distance, over the lendable
     * groups, from the top of the carve down to the highest block anything holds plus the buffer.
     * Rank 0 only -- a follower's held history is empty. Also where the buffer's trend term
     * decays, so it is called once per arbiter tick. */
    int64_t spare_tokens();

    /* LEND `tokens` FROM THE TOP, or take back what a smaller number no longer covers. Lending
     * past a held block is refused (RAD_E_STATE): the arbiter asked for more than spare_tokens()
     * said there was. Taking back is always allowed here, and it is the caller's to make true --
     * the slab must have left those blocks before this runs. */
    int     set_loan(int64_t tokens);
    int64_t loan_tokens() const { return loan_; }

    /* WHO GIVES IT BACK WHEN A REQUEST CANNOT WAIT. A request that needs more blocks than the
     * cache kept in hand calls this with the loan it can still afford, and the owner of the
     * slabs drains them to it before returning -- so admission is decided against the whole
     * carve, exactly as it would be with nothing lent. Without one, what is lent is not
     * recalled and the request is refused (RAD_E_FULL). */
    void    set_reclaim(std::function<int(int64_t tokens)> f) { reclaim_ = std::move(f); }

    /* WHETHER A DECOMMIT ACTUALLY GIVES THE CARD BACK, for the checkpoint region, which is the one
     * part of the pool still backed on demand. Measured once at bringup rather than assumed,
     * because it is a property of the driver: where it is false a freed checkpoint slot keeps its
     * pages, since unmapping them would cost driver calls and return nothing. */
    void set_release_frees_card(bool yes) { release_frees_card_ = yes; }

    /* MATCH A LEADER'S LOAN, FOR A RANK THAT DOES NOT KNOW WHAT IT IS HOLDING.
     *
     * Block ids are SHARED ACROSS RANKS -- RadBatch carries one block table and every rank indexes
     * its own shard of the pool with it -- and only rank 0's KVManager tracks sequences. So a
     * follower takes its mark from the leader instead of computing one: its own slab occupies the
     * same blocks of its own pool, and an id the leader hands out below the mark has to be one the
     * follower's slab has left too. */
    int     follow(const KVManager& leader);

    /* HOW MANY OF A GROUP'S BLOCKS THE CACHE MAY HAND OUT: the carve less what is lent. */
    int64_t live_blocks(int32_t group) const;
    int64_t committed_bytes() const;

    /* HOW MUCH OF THE CHECKPOINT REGION MUST STAY AVAILABLE, in bytes, counting only what is not
     * committed already. The number is the HIGH WATER MARK PLUS ONE rather than the whole cap:
     * the cap is what the operator allows and the high water is what this workload takes, and
     * holding one slot past it means the allocation that would advance it always has room. */
    int64_t ck_reserve_bytes() const;

    /* BACK THIS RANK'S COPY OF A SLOT, AND SAY SO IF THE CARD WILL NOT.
     *
     * A slot's LIFETIME is decided once: rank 0's PrefixCache owns the free list and every other
     * rank takes the id it is handed. Its BACKING is per rank, because every rank writes its own
     * shard of the snapshot into the same offset of its OWN pool. alloc_checkpoint backs the slot
     * on every rank before it hands the id out (set_checkpoint_peers), so on the paths that use a
     * slot this is a check that finds it backed. Called before EVERY access rather than once:
     * idempotence is what makes that safe, and a commit of a range already committed is a no-op.
     *
     * RAD_E_FULL when the commit would take the card under its floor (set_checkpoint_floor). */
    int back_checkpoint(int32_t slot);

    /* THE CARD'S FLOOR FOR A CHECKPOINT SLOT'S PAGES: the card is read before a slot is backed,
     * and a slot that would leave it under `keep_free` bytes is refused as a full pool.
     *
     * The budget counts the whole checkpoint region as the cache's, but the region is backed as
     * slots are taken, and until then its bytes read as free. The allocations nobody sizes in
     * advance -- a kernel's code object and a spilling kernel's scratch, both made on the
     * kernel's first launch -- are made out of that same free memory. A slot that took the last
     * of it would leave the next first launch nothing, which is a fault inside the HIP runtime
     * and not an error. A refused slot is a snapshot fewer on the card, which the retention
     * policy answers and the host tier covers.
     *
     * `free_bytes` reads the card's free memory into its argument: the engine reads the device, a
     * test reads a simulated card, and no floor is set on a backend with no card. */
    using FreeBytes = std::function<int(int64_t*)>;
    void set_checkpoint_floor(FreeBytes free_bytes, int64_t keep_free) {
        ck_free_bytes_ = std::move(free_bytes);
        ck_keep_free_  = keep_free;
    }

    /* THE OTHER RANKS' MANAGERS, given to the one whose free list hands the ids out. Every rank
     * writes its shard of a snapshot into the same slot of its own pool, so a slot is backed on
     * every card or handed out on none: a card that cannot back it is a full pool at allocation,
     * which the prefix cache answers by retiring a snapshot, rather than a failed copy after the
     * step that produced the state, which stops the engine. */
    void set_checkpoint_peers(std::vector<KVManager*> peers) { ck_peers_ = std::move(peers); }

    /* BACK EVERY FREE SLOT NOW, ON EVERY RANK, and withdraw the ones the cards have no room for.
     * For a driver that never takes a mapping's pages back (set_release_frees_card(false)): a
     * slot backed later costs exactly what it costs now, and backing it now puts the region's
     * cost on the card before serving starts, where the startup reading shows it. A withdrawn
     * slot is never handed out. `withdrawn` counts them; RAD_E_FULL is not returned. */
    int back_free_checkpoints(int64_t* withdrawn);

    int64_t checkpoint_stride() const { return ck_stride_; }

    /* ---------------- preemption by recompute (spec §7.1, §17) ---------------- */
    /* Free the sequence's blocks and take it back to zero tokens. Its state slots are freed too: a
     * recurrent state that is not snapshotted cannot be resumed, so a preempted sequence replays
     * from its last checkpoint or from zero.
     *
     * IT DOES NOT MARK ANYTHING, and carries no "needs re-prefill" flag. The scheduler's own
     * `r.n_computed = 0` (preempt_back) is what drives the replay, and a second piece of state
     * saying the same thing is a second thing to keep true. seq_tokens() == 0 is the reading, and
     * it cannot drift. */
    int  preempt(uint64_t seq);

    /* ---------------- speculative rejection (spec §10) ---------------- */
    /* Bring the sequence back to `n_tokens` and return the paged blocks past it. A verify writes
     * KV for 1 + n_spec positions and the sampler accepts a prefix of them, so the rest has to go
     * back; `ensure` is monotone and cannot express it.
     *
     * PAGED GROUPS ONLY, and that is the whole point. The linear state and the conv window have
     * already absorbed the rejected tokens and are NOT recomputed: libr4d keeps the conv cache as
     * a rolling window of conv_width-1 + n_spec entries and reads at the slot the last accepted
     * token left, so a rejection there is a change of read offset that the kernels perform
     * themselves (core/sample/accept.h). Freeing a state slot here would destroy the state that
     * offset points into.
     *
     * Rolling back is not preemption: the sequence keeps its state slots and its window base.
     * Without this the pool leaks one block per rejected block-crossing, which is a slow leak and
     * therefore the worst kind. */
    int  rollback(uint64_t seq, int64_t n_tokens);

    /* The absolute token position that block_table(seq, group)[0] covers. Zero for a FULL group;
     * non-zero for a WINDOW group, whose blocks below the window have been recycled. The step
     * batch's slot mapping is (pos - first_block_pos) / block_size, and without the offset every
     * sliding-window sequence writes its KV to the wrong slot the moment it passes its window. */
    int64_t first_block_pos(uint64_t seq, int32_t group) const;

    /* THE ENTRIES OF A SEQUENCE'S TABLE THAT CHANGED since the last call: every entry below the
     * returned index is what it was then, and kNoDirty says none changed. The call clears the
     * mark, so it has exactly one reader -- the batch builder, which stages only the changed tail
     * of a table the device already holds. A sequence it has not seen answers 0. */
    int64_t take_table_changes(uint64_t seq, int32_t group) const;
    static constexpr int64_t kNoDirty = INT64_MAX;

    /* Free victims, in the order given, until `seq` can be extended to `n_tokens`.
     *   RAD_OK        room was made; *preempted lists who paid for it
     *   RAD_E_FULL    even an empty pool cannot serve it -- fail the request, do not retry
     * The caller supplies the victim order because priority is the scheduler's business, not the
     * allocator's. */
    int  make_room(uint64_t seq, int64_t n_tokens,
                   const std::vector<uint64_t>& victims,
                   std::vector<uint64_t>* preempted);

    /* ---------------- linear-state checkpoints (spec §7.3) ---------------- */
    int64_t checkpoint_bytes() const { return checkpoint_bytes_; }
    int64_t checkpoint_slots() const { return (int64_t)ck_slot_base_.size(); }
    int64_t free_checkpoint_slots() const { return (int64_t)ck_free_.size(); }
    int  alloc_checkpoint(int32_t* slot_out);
    void free_checkpoint(int32_t slot);
    void* checkpoint_ptr(int32_t slot) const;

    /* SNAPSHOT AND RESUME A SEQUENCE'S WHOLE LINEAR STATE. This is what makes a prefix hit legal
     * on a hybrid model: the attention half of a hit is carried by the adopted blocks, and the
     * recurrent half is carried by these two calls and by nothing else. Without them a hit
     * resumes every recurrent layer from whatever the slot last held -- which reads as fluent
     * text, so it is not a failure anyone notices without a determinism check.
     *
     * The pool is LAYER-MAJOR and a sequence's state is therefore strided, not contiguous, so the
     * blob is a gather: for every stateful group, in `plans()` order, for every layer, its
     * `layer_bytes_per_state()` bytes. Both directions walk that identically, which is the whole
     * requirement -- a checkpoint is only ever read back by this same build.
     *
     * COPY 0 ONLY. A speculating sequence owns `state_copies` physical slots and the committed
     * state is copy 0 (KVGroupPlan); the replay factors in the others describe tokens that have
     * not been accepted and mean nothing to a later reader.
     *
     * Enqueued on `s`, not synchronous: the save has to follow the step that produced the state
     * and the restore has to precede the step that consumes it, and both are ordered by the
     * stream rather than by waiting. RAD_E_INVAL on a bad slot or an unknown sequence. */
    int checkpoint_save(const int32_t* state_slots, int n_slots, int32_t ck_slot, RadStream s);
    int checkpoint_restore(const int32_t* state_slots, int n_slots, int32_t ck_slot, RadStream s);

    /* Byte address of a state slot, for the slot-mapping builder. Null when the pool has no
     * host-visible base (a device pool), which is the normal case -- the offset is what the step
     * batch actually carries. */
    void*   state_ptr(int32_t group, int32_t layer, int32_t slot) const;

private:
    /* carve() is the wrapper and carve_into() is the work, so that a carve which returns part way
     * through cannot leave plans_ answering guards for block state that was never sized. forget()
     * empties both together and is what plan_groups() starts from. */
    int  carve_into(Pool& kv_pool);
    void forget();

    /* One traversal serving both checkpoint directions; see checkpoint_save. */
    int checkpoint_copy(const int32_t* state_slots, int n_slots, int32_t ck_slot, bool save,
                        RadStream s);

    struct Seq {
        uint64_t id = 0;
        int64_t  n_tokens = 0;
        std::vector<std::vector<int32_t>> blocks;   /* per group */
        std::vector<int64_t> first_block_pos;       /* WINDOW: token pos of blocks[0] */
        std::vector<int32_t> state;                 /* per group, -1 when not stateful */
        /* Per group, the lowest table entry written since the batch builder last staged this
         * sequence: everything below it is unchanged. An append or a truncation lowers it to
         * where it touched; anything that shifts or replaces entries lowers it to zero; kNoDirty
         * is nothing. Mutable because reading it is what clears it -- take_table_changes. */
        mutable std::vector<int64_t> dirty_from;
    };

    int  grab_block(int32_t g, int32_t* out);
    void drop_block(int32_t g, int32_t b);
    /* Every increment of a held block's reference count, so the cache-only count follows it. */
    void add_ref(size_t g, int32_t b);
    int  grab_state(int32_t g, int32_t* out);
    void drop_state(int32_t g, int32_t s);
    Seq* find(uint64_t seq);
    const Seq* find(uint64_t seq) const;

    std::vector<KVGroupPlan> plans_;
    std::vector<std::vector<int32_t>> free_blocks_;   /* per group */
    std::vector<std::vector<int32_t>> refcount_;      /* per group, per block */
    /* Per group, per block: the prefix cache holds one of the references. A free block is never
     * marked. `cache_only_` is, per group, how many marked blocks are at a count of one. */
    std::vector<std::vector<uint8_t>> cached_;
    std::vector<int64_t>              cache_only_;
    std::vector<std::vector<int32_t>> free_states_;   /* per group */
    std::unordered_map<uint64_t, Seq> seqs_;
    std::vector<BlockCopy> pending_;

    std::vector<int64_t> ck_slot_base_;               /* pool offsets of the checkpoint slots */
    std::vector<int32_t> ck_free_;
    int64_t   checkpoint_bytes_ = 0;
    std::vector<StateClear> clears_;                  /* grabbed slots the engine must zero */
    void*     pool_base_ = nullptr;
    BundleComparison bundling_;
    std::string report_;
    int64_t   max_seqs_ = 0;
    int64_t   max_ctx_ = 0;
    int64_t   max_tok_ = 0;      /* --max-num-batched-tokens; the buffer floor is one of these */
    /* plan_groups() computes these; carve() spends against them and ceiling() reports on them. */
    int64_t   fixed_ = 0;      /* the non-negotiable floor: states, conv slots, checkpoints, padding */
    int64_t   ck_total_ = 0;   /* the checkpoint part of fixed_, named for the refusal message */
    int64_t   ckpts_ = 0;      /* --checkpoint-slots, as plan_groups() resolved it */
    /* THE SLOT STRIDE AND WHERE THE SLOTS BEGIN. The stride is checkpoint_bytes_ rounded up to a
     * whole number of commit granules, so a slot can be backed and released without touching a
     * neighbour; the offset is where the eager startup commit stops, because everything from
     * there up is backed as it is handed out. Zero on a model with no linear state. */
    bool      release_frees_card_ = true;
    bool      ck_elastic_ = false;
    int64_t   ck_high_water_ = 0;
    int64_t   ck_backed_ = 0;
    int64_t   ck_stride_ = 0;
    int64_t   ck_region_off_ = 0;
    FreeBytes ck_free_bytes_;        /* set_checkpoint_floor */
    int64_t   ck_keep_free_ = 0;
    std::vector<KVManager*> ck_peers_;
    /* RAD_OK when this rank has the slot backed or room above the floor to back it. */
    int  ck_room(int32_t slot) const;
    bool ck_backed(int32_t slot) const;
    /* The commit itself, with no floor: callers have asked ck_room on every rank first. */
    int  ck_commit(int32_t slot);
    /* ck_room on every rank, then ck_commit on every rank. */
    int  ck_back_everywhere(int32_t slot);
    /* What a checkpoint slot is aligned and sized to. RAD_ALIGN_UNIT on a device with no virtual
     * memory management, where nothing is released and the stride is only padding. */
    int64_t ck_align() const;

    /* ---- the loan. See the public block for what this is and why a block that anything holds
     * is never lent. ----
     *
     * `live_` is how many of a group's blocks the cache may hand out -- the carve less the loan --
     * and it gates the free list rather than the carve: every id the carve created is in
     * `free_blocks_`, and an id at or above live_ is simply not handed out. THAT IS WHY THE FREE
     * LIST IS A MIN-HEAP. Handing out the lowest free id keeps the held blocks packed at the
     * bottom of the pool, so the top is what drains and the top is the only part a loan can
     * reach; a stack would hand back the id freed most recently, which after a busy period is
     * exactly the high block the loan wants. The heap also makes the gate free: the smallest free
     * id is below live_ whenever any free id is, so no scan is needed to find one.
     *
     * `held_hist_` counts held blocks per bucket, so "what is the highest bucket anything still
     * holds" is a walk of a few thousand counters instead of a scan of every refcount. It is what
     * decides how far a loan may reach, and it is maintained on the allocation path because a
     * scan there is not affordable and a stale answer would lend a block out from under a live
     * sequence. */
    /* THE BUFFER, PER GROUP, IN BLOCKS. `chunk_blocks_` is the floor -- one prefill chunk, fixed
     * at plan time; `short_peak_` is the largest shortfall the admission path has met recently,
     * raised by grow_for and decayed by every spare_tokens() so it forgets a burst in a few
     * seconds.
     *
     * THE SHORTFALL IS THE SIGNAL AND THE RECALL IS NOT. grow_for recalls the shortfall PLUS the
     * buffer; feeding that back would be a buffer that sizes itself from its own last value and
     * climbs until it is the carve. What is recorded is what was asked for and not there. */
    std::vector<int64_t>  chunk_blocks_;
    std::vector<int64_t>  short_peak_;
    Pool*    pool_ = nullptr;
    int64_t  loan_ = 0;                     /* tokens lent from the top of every lendable group */
    uint64_t seq_frees_ = 0;
    std::function<int(int64_t)> reclaim_;
    std::vector<int64_t>  live_;            /* per group: blocks the cache may hand out */
    std::vector<int64_t>  free_usable_;     /* per group: free ids below live_ */
    std::vector<int64_t>  bucket_blocks_;   /* per group: blocks one held_hist_ bucket covers */
    std::vector<std::vector<int32_t>> held_hist_;   /* per group, per bucket: held blocks */

    int64_t available_blocks(int32_t g) const;
    /* The buffer this group wants in hand, in blocks. See buffer_bytes(). */
    int64_t  buf_blocks(size_t i) const;
    /* One past the highest block anything holds, to a bucket. */
    int64_t  held_top(size_t i) const;
    /* The blocks a loan of `tokens` takes off the top of group i. */
    int64_t  lent_blocks(size_t i, int64_t tokens) const;
    int  grow_for(int32_t g, int64_t blocks);
    void note_held(int32_t g, int32_t b, int delta);
    bool      configured_ = false;
    static const std::vector<int32_t> kNoBlocks;
};

}  /* namespace rad */
