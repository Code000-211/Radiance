/* sched_fakes.h -- the fakes the scheduler tests run against: the device layer as host memory,
 * and core/mem's KV manager and prefix cache as small faithful models of their interfaces
 * (core/sched/mem_iface.h; see that header for what must be reconciled with core/mem).
 *
 * The device fakes are weak, so a linked core/device/ backend would replace them at link time; a
 * test binary checks that they are the ones in use (g_fake_device). A binary that runs on the
 * real device layer and wants only the fakes of core/mem (mem_test, which compares FakeKV with
 * KVManager) defines SCHED_FAKES_KV_ONLY first. Include this once per test binary.
 */
#pragma once
#include "rad_test.h"

#include "sched/scheduler.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifndef SCHED_FAKES_KV_ONLY
/* Set by the fake rad_dev_alloc: a binary that reads it false runs on the real backend. */
static bool g_fake_device = false;

/* ================================================================== the device fakes */
/* Weak, so a linked-in real host backend replaces them. Device memory is host memory here, so the
 * assertions on cu_seqlens and the slot mapping read exactly the bytes a kernel would. */
extern "C" {

__attribute__((weak)) void* rad_dev_alloc(int64_t bytes, int) {
    g_fake_device = true;
    return std::calloc(1, (size_t)(bytes > 0 ? bytes : 1));
}
__attribute__((weak)) void rad_dev_free(void* p, int) { std::free(p); }

/* The builder binds a card before allocating a rank's staging slab. Stubbed because, if missing,
 * the linker would pull in dispatch.cpp.o, whose strong rad_dev_alloc/rad_memcpy_async would
 * replace every weak fake above and run this test on the real HIP backend with a null stream. */
__attribute__((weak)) int rad_dev_set(int) { return RAD_OK; }

__attribute__((weak)) int rad_memcpy_async(void* d, const void* s, int64_t n, RadStream) {
    if (n > 0) std::memcpy(d, s, (size_t)n);
    return RAD_OK;
}
/* The builder zeroes the encoder rows of every rank but the first. */
__attribute__((weak)) int rad_memset_async(void* d, int v, int64_t n, RadStream) {
    if (n > 0) std::memset(d, v, (size_t)n);
    return RAD_OK;
}
__attribute__((weak)) int rad_memcpy_ranges_async(void* d, const void* s, int64_t prefix,
                                                  const RadCopyRange* r, int n, RadStream) {
    if (prefix > 0) std::memcpy(d, s, (size_t)prefix);
    for (int i = 0; i < n; ++i)
        std::memcpy((char*)d + r[i].dst, (const char*)s + r[i].src, (size_t)r[i].bytes);
    return RAD_OK;
}
__attribute__((weak)) int rad_memcpy_list_async(const RadCopySeg* g, int n, unsigned, RadStream) {
    for (int i = 0; i < n; ++i)
        if (g[i].bytes > 0) std::memcpy(g[i].dst, g[i].src, (size_t)g[i].bytes);
    return RAD_OK;
}
__attribute__((weak)) int rad_link_hold(RadStream, int) { return RAD_OK; }
__attribute__((weak)) void* rad_link_gate(RadStream) { return nullptr; }

static int g_fake_event;
__attribute__((weak)) int  rad_event_create(RadEvent* e) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) int  rad_event_create_local(RadEvent* e) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) int  rad_event_create_as(RadEvent* e, unsigned) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) void rad_event_destroy(RadEvent) {}
__attribute__((weak)) int  rad_event_record(RadEvent, RadStream) { return RAD_OK; }
__attribute__((weak)) int  rad_event_sync(RadEvent) { return RAD_OK; }

}  /* extern "C" */
#endif  /* SCHED_FAKES_KV_ONLY */

/* ================================================================== the one-shot build failure */
/* core/sched/batch.cpp calls this weak hook once per BatchBuilder::build, after the empty-plan
 * check. The call and the always-false default exist only in the test binaries' own copy of
 * batch.cpp (rad_test_hooks in tests/CMakeLists.txt, compiled with RAD_TEST_HOOKS); rad_core is
 * never compiled with it, so no production library carries the seam. This strong definition
 * replaces the default, so a test can make exactly one build of a plan that has work to do return
 * null: the builder's own refusal path, which is what C's serial_batch_failed() and E's driver
 * must handle, reached without a runtime switch. One shot on purpose -- a permanently failing
 * builder would only prove that the engine stops, not that a later pass recovers. */
static bool g_fail_next_batch_build = false;

extern "C" bool rad_test_fail_batch_build() {
    if (!g_fail_next_batch_build) return false;
    g_fail_next_batch_build = false;
    return true;
}

/* Make the next build that has work to do fail. */
inline void sched_fail_next_build() { g_fail_next_batch_build = true; }

/* ================================================================== fake core/mem */
namespace sched_fake {
using namespace rad;

/* A small but faithful block manager: a free list per group, refcounted blocks (an adopted
 * prefix-cache block stays out of the pool while the cache names it), and per-sequence state slots
 * for the linear and conv groups. Shaped like core/mem/kv.h (sequence-keyed, owns the tables). */
struct FakeKV : IKVManager {
    std::vector<KVGroupInfo> gs;
    KVGeom geom;
    int64_t max_ctx = 0;

    std::vector<std::vector<int32_t>> freelist;    /* per group */
    std::vector<std::vector<int32_t>> refcnt;      /* per group, per block */
    std::vector<int64_t> total;
    std::vector<std::vector<int32_t>> state_free;  /* per group */

    struct Seq {
        std::vector<std::vector<int32_t>> blocks;  /* per group */
        std::vector<int32_t> state;                /* per group, -1 when not stateful */
        int64_t n = 0;
        /* Per group, the lowest table entry changed since take_table_changes last read it, kept
         * as the real manager keeps it, so these tests run the builder's partial restage. */
        mutable std::vector<int64_t> dirty;
    };
    std::unordered_map<uint64_t, Seq> seqs;
    static inline const std::vector<int32_t> kNone{};

    void setup(const std::vector<KVGroupInfo>& groups, int64_t ctx,
               int64_t blocks_per_paged_group, int64_t n_states) {
        gs = groups;
        geom = KVGeom::of(gs);
        max_ctx = ctx;
        freelist.assign(gs.size(), {});
        refcnt.assign(gs.size(), {});
        total.assign(gs.size(), 0);
        state_free.assign(gs.size(), {});
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged) {
                total[i] = blocks_per_paged_group;
                refcnt[i].assign((size_t)blocks_per_paged_group, 0);
                for (int64_t b = blocks_per_paged_group - 1; b >= 0; --b)
                    freelist[i].push_back((int32_t)b);
            } else {
                for (int64_t k = n_states - 1; k >= 0; --k) state_free[i].push_back((int32_t)k);
            }
        }
    }

    void retain(int32_t g, int32_t b) { ++refcnt[(size_t)g][(size_t)b]; }
    void release(int32_t g, int32_t b) {
        if (--refcnt[(size_t)g][(size_t)b] == 0) freelist[(size_t)g].push_back(b);
    }
    int32_t grab(int32_t g) {
        int32_t b = freelist[(size_t)g].back();
        freelist[(size_t)g].pop_back();
        refcnt[(size_t)g][(size_t)b] = 1;
        return b;
    }

    int64_t rows_for(int gi, int64_t n) const {
        const KVGeom::G& e = geom.g[(size_t)gi];
        return e.paged ? (n + e.block_size - 1) / e.block_size : 1;
    }

    int add_sequence(uint64_t id) override {
        if (seqs.count(id)) return RAD_OK;
        Seq q;
        q.blocks.assign(gs.size(), {});
        /* Sized for the whole context here, on arrival, so a growing table does not reach the
         * allocator inside a step the scheduler tests count allocations in. */
        for (size_t i = 0; i < gs.size(); ++i)
            if (geom.g[i].paged) q.blocks[i].reserve((size_t)rows_for((int)i, max_ctx));
        q.state.assign(gs.size(), -1);
        q.dirty.assign(gs.size(), 0);
        seqs.emplace(id, std::move(q));
        return RAD_OK;
    }
    bool has_sequence(uint64_t id) const override { return seqs.count(id) != 0; }

    void free_sequence(uint64_t id) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged)
                for (int32_t b : it->second.blocks[i]) release((int32_t)i, b);
            it->second.blocks[i].clear();
            if (it->second.state[i] >= 0) {
                state_free[i].push_back(it->second.state[i]);
                it->second.state[i] = -1;
            }
        }
        seqs.erase(it);
    }

    int ensure(uint64_t id, int64_t n) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        Seq& q = it->second;
        /* All or nothing, as KVManager: a partial grow would leave the sequence holding blocks it
         * cannot use, and the caller unable to tell whether preempting would help. A state slot
         * is checked here too, before any block is taken. */
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!geom.g[i].paged) {
                if (q.state[i] < 0 && state_free[i].empty()) return RAD_E_FULL;
                continue;
            }
            int64_t need = rows_for((int)i, n) - (int64_t)q.blocks[i].size();
            if (need > 0 && need > (int64_t)freelist[i].size()) return RAD_E_FULL;
        }
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged) {
                int64_t need = rows_for((int)i, n) - (int64_t)q.blocks[i].size();
                if (need > 0) q.dirty[i] = std::min<int64_t>(q.dirty[i], (int64_t)q.blocks[i].size());
                for (int64_t k = 0; k < need; ++k) q.blocks[i].push_back(grab((int32_t)i));
            } else if (q.state[i] < 0) {
                if (state_free[i].empty()) return RAD_E_FULL;
                q.state[i] = state_free[i].back();
                state_free[i].pop_back();
                q.blocks[i].assign(1, q.state[i]);
            }
        }
        if (n > q.n) q.n = n;
        return RAD_OK;
    }

    int adopt(uint64_t id, int32_t g, const std::vector<int32_t>& blocks, int64_t n) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        if (n % geom.g[(size_t)g].block_size != 0) return RAD_E_INVAL;
        Seq& q = it->second;
        q.blocks[(size_t)g] = blocks;
        q.dirty[(size_t)g] = 0;
        for (int32_t b : blocks) retain(g, b);
        if (n > q.n) q.n = n;
        return RAD_OK;
    }

    int preempt(uint64_t id) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (geom.g[i].paged)
                for (int32_t b : it->second.blocks[i]) release((int32_t)i, b);
            it->second.blocks[i].clear();
            it->second.dirty[i] = 0;
            if (it->second.state[i] >= 0) {
                state_free[i].push_back(it->second.state[i]);
                it->second.state[i] = -1;
            }
        }
        it->second.n = 0;
        return RAD_OK;
    }

    int rollback(uint64_t id, int64_t n) override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return RAD_E_NOTFOUND;
        /* Nothing to give back: no entry changes, so none is marked (as KVManager). */
        if (n >= it->second.n) return RAD_OK;
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!geom.g[i].paged) continue;   /* linear state is NOT rolled back (spec §10) */
            int64_t keep = rows_for((int)i, n);
            std::vector<int32_t>& bl = it->second.blocks[i];
            while ((int64_t)bl.size() > keep) { release((int32_t)i, bl.back()); bl.pop_back(); }
            it->second.dirty[i] = std::min<int64_t>(it->second.dirty[i], (int64_t)bl.size());
        }
        it->second.n = n;
        return RAD_OK;
    }

    bool can_ever_fit(int64_t n) const override {
        for (size_t i = 0; i < gs.size(); ++i)
            if (geom.g[i].paged && rows_for((int)i, n) > total[i]) return false;
        return true;
    }

    const std::vector<int32_t>& block_table(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        return it == seqs.end() ? kNone : it->second.blocks[(size_t)g];
    }
    int32_t state_slot(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        return it == seqs.end() ? -1 : it->second.state[(size_t)g];
    }
    /* Columns per sequence in a state group's index row. One is the state itself; two adds a
     * second slot beside it, 1000 + the state's, as a kernel keeping scratch beside its state
     * would ask for. A paged group always has one. */
    int64_t sidx_width = 1;
    int64_t state_index_width(int32_t g) const override {
        return geom.g[(size_t)g].paged ? 1 : sidx_width;
    }
    int state_index_row(uint64_t id, int32_t g, int32_t* out, int max) const override {
        if (!out || max <= 0) return 0;
        const int32_t s = state_slot(id, g);
        const int n = (int)std::min<int64_t>(state_index_width(g), max);
        for (int j = 0; j < n; ++j) out[j] = (s < 0 || j == 0) ? s : 1000 * j + s;
        return n;
    }
    int64_t first_block_pos(uint64_t, int32_t) const override { return 0; }
    int64_t take_table_changes(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return 0;
        const int64_t d = it->second.dirty[(size_t)g];
        it->second.dirty[(size_t)g] = INT64_MAX;
        return d;
    }
    /* Snapshot slots the store has given back; every other slot is held. */
    std::vector<int32_t> ck_gone;
    bool checkpoint_held(int32_t slot) const override {
        return std::find(ck_gone.begin(), ck_gone.end(), slot) == ck_gone.end();
    }
    /* Blocks an elastic pool has handed back to the card: carved and free, but not available
     * until the pool grows again. free_blocks() is bounded by that and held_blocks() is not. */
    int64_t unbacked = 0;
    int64_t free_blocks(int32_t g) const override {
        const int64_t n = (int64_t)freelist[(size_t)g].size() - unbacked;
        return n > 0 ? n : 0;
    }
    int64_t total_blocks(int32_t g) const override { return total[(size_t)g]; }
    int64_t held_blocks(int32_t g) const override {
        return total[(size_t)g] - (int64_t)freelist[(size_t)g].size();
    }
    /* A byte size per object, so the sum is checkable. The real manager reads these off the
     * carved plan; here the test sets them, and a group left at zero contributes nothing. */
    std::vector<int64_t> block_bytes, state_bytes;
    int64_t pool_bytes(int32_t g) const override {
        const KVGeom::G& gg = geom.g[(size_t)g];
        return gg.paged ? total[(size_t)g] * bb(g)
                        : (int64_t)(state_free[(size_t)g].size() + n_states_out(g)) * sb(g);
    }
    int64_t pool_bytes_used(int32_t g) const override {
        const KVGeom::G& gg = geom.g[(size_t)g];
        return gg.paged ? (total[(size_t)g] - free_blocks(g)) * bb(g) : n_states_out(g) * sb(g);
    }
    /* This pool is not elastic, so what it was carved to address is what it is holding. */
    int64_t pool_bytes_carved(int32_t g) const override { return pool_bytes(g); }
    int64_t bb(int32_t g) const {
        return g < (int32_t)block_bytes.size() ? block_bytes[(size_t)g] : 0;
    }
    int64_t sb(int32_t g) const {
        return g < (int32_t)state_bytes.size() ? state_bytes[(size_t)g] : 0;
    }
    /* States handed out, which the fake tracks only through the sequences holding them. */
    int64_t n_states_out(int32_t g) const {
        int64_t n = 0;
        for (const auto& kv : seqs) if (kv.second.state[(size_t)g] >= 0) ++n;
        return n;
    }
    int64_t block_tokens(int32_t g) const override {
        const KVGeom::G& gg = geom.g[(size_t)g];
        return gg.paged ? gg.block_size : 0;
    }
};

struct FakePC : IPrefixCache {
    FakeKV* kv = nullptr;
    int32_t attn = 0, lin = 0, src = -1;
    int32_t next_slot = 0;
    int32_t reserved = 0;
    int64_t evicted = 0, ckpt_evicted = 0;
    int64_t held = 0;            /* what the cache would report holding now */
    int64_t interval = 256;
    std::vector<int32_t> groups;                     /* the paged groups this cache indexes */
    std::vector<std::vector<int32_t>> seeded;        /* blocks the cache holds a reference on */

    /* Pretend an earlier request filled the cache: take `n_tokens` worth of blocks out of the pool
     * and hold a reference on them, which is what makes a hit safe (a block the cache names is
     * never in the free list). */
    void seed(int32_t n_tokens) {
        attn = n_tokens;
        seeded.assign(groups.size(), {});
        for (size_t i = 0; i < groups.size(); ++i) {
            int32_t g = groups[i];
            int64_t nb = n_tokens / kv->geom.g[(size_t)g].block_size;
            for (int64_t b = 0; b < nb; ++b) seeded[i].push_back(kv->grab(g));
        }
    }

    std::vector<uint64_t> restores;
    void restore(const Request& r) override { restores.push_back(r.id); }

    mutable int64_t lookups = 0;
    /* A cap (HitCap) as prefix.cpp applies one: attention ends at a block at or below it, the
     * linear state at the deepest checkpoint under that (checkpoints at multiples of `interval`,
     * none below `ckpt_lo`), and the whole hit stands if that gives back too much. */
    int64_t ckpt_lo = 0;
    CacheHit lookup(const Request&, std::vector<std::vector<int32_t>>* out,
                    const HitCap* cap) const override {
        ++lookups;
        if (out) {
            out->resize(groups.size());
            for (size_t i = 0; i < groups.size() && i < seeded.size(); ++i)
                (*out)[i] = seeded[i];
        }
        CacheHit h{ attn, lin, src };
        const int64_t whole = src >= 0 ? lin : 0;
        if (cap && attn > cap->tokens / 16 * 16 && whole > cap->tokens) {
            const int64_t a = cap->tokens / 16 * 16;
            int64_t l = std::min<int64_t>(lin, a) / interval * interval;
            if (l < ckpt_lo) l = 0;
            if (whole - l <= cap->max_give_back) {
                h.n_attn_tokens = (int32_t)a;
                h.n_linear_tokens = (int32_t)l;
                h.checkpoint_slot = l > 0 ? src : -1;
                h.given_back = (int32_t)(whole - l);
                if (out)
                    for (auto& b : *out)
                        if ((int64_t)b.size() > a / 16) b.resize((size_t)(a / 16));
            }
        }
        return h;
    }
    const std::vector<int32_t>& cached_groups() const override { return groups; }

    /* What the scheduler published and asked to be evicted, recorded rather than acted on: these
     * tests check that the scheduler publishes before it frees and evicts before it preempts, not
     * the cache's own indexing, which prefix.cpp's tests cover. */
    struct Published { uint64_t seq; int64_t n_tokens; size_t n_groups; };
    std::vector<Published> published;
    int64_t evict_calls = 0, evict_grant = 0;

    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens) override {
        published.push_back({ r.id, n_tokens, blocks.size() });
        return RAD_OK;
    }
    /* Grants only what a test asked for, so "the cache had nothing to give" is reachable and the
     * preemption path below it still runs. */
    int64_t evict(int64_t n) override {
        ++evict_calls;
        if (evict_grant <= 0) return 0;
        int64_t took = n < evict_grant ? n : evict_grant;
        if (evict_frees) {
            /* An entry is one block in every cached group: give that many back to the pool. */
            int64_t have = seeded.empty() ? 0 : (int64_t)seeded[0].size();
            took = std::min(took, have);
            for (size_t i = 0; i < groups.size() && i < seeded.size(); ++i)
                for (int64_t k = 0; k < took; ++k) {
                    kv->release(groups[i], seeded[i].back());
                    seeded[i].pop_back();
                }
        }
        evict_grant -= took;
        evicted += took;
        return took;
    }
    bool evict_frees = false;    /* evict() returns the seeded blocks to the pool */
    std::vector<uint64_t> holds, releases;
    void hold(const Request& r) override { holds.push_back(r.id); }
    void release(const Request& r) override { releases.push_back(r.id); }
    /* What the reserve asked for, call by call. */
    std::vector<int64_t> trims;
    int64_t trim(int64_t n) override {
        trims.push_back(n);
        return 0;
    }
    /* Cumulative, like the real cache's: the scheduler publishes it as a counter and the server
     * takes deltas, so a per-call value would read as a cache that never evicts. */
    int64_t evictions() const override { return evicted; }
    int64_t ckpt_evictions() const override { return ckpt_evicted; }
    int64_t held_tokens() const override { return held; }
    int64_t next_checkpoint_after(int64_t pos) const override {
        return ((pos / interval) + 1) * interval;
    }
    std::vector<int64_t> committed;
    void commit_checkpoint(const Request&, int64_t pos) override { committed.push_back(pos); }
    int reserve_checkpoint(const Request&, int64_t, int32_t* slot_out) override {
        ++reserved;
        *slot_out = next_slot++;
        return RAD_OK;
    }
};

}  /* namespace sched_fake */
