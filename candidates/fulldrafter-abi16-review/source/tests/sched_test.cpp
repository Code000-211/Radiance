/* sched_test.cpp -- the scheduler, the chunker, the step batch, the derivations and the draft
 * controller, against fakes for core/mem and for the device layer.
 *
 * The device fakes are WEAK, so wherever core/device/ is linked in its host backend wins at link
 * time and this file needs no edit; without it they are what lets a scheduler test run at all. The
 * KV manager and prefix cache are ordinary implementations of the interfaces in
 * core/sched/mem_iface.h -- see that header for what has to be reconciled with core/mem.
 */
#include "rad_test.h"

#include "sched/scheduler.h"
#include "sched/derive.h"
#include "sched/advance_impl.h"
#include "sched/tree_state.h"
#include "sample/accept.h"
#include "server/sink.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>

using namespace rad;

/* ================================================================== allocation counter */
/* The step path must not allocate. Counting through a replaced global operator new is the only
 * way to assert that and mean it: a review can miss a vector that grows on the ten-thousandth
 * request, and that is exactly the one that shows up as a latency spike nobody can reproduce. */
static long g_allocs = 0;
static bool g_counting = false;

/* malloc/free is a matched pair for a replaced global new/delete, but GCC's
 * -Wmismatched-new-delete does not model the replacement and reports every inlined delete in this
 * translation unit as a free() on a pointer that came from new. */
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void* operator new(size_t n) {
    if (g_counting) ++g_allocs;
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void* operator new[](size_t n) { return operator new(n); }
void  operator delete(void* p) noexcept { std::free(p); }
void  operator delete[](void* p) noexcept { std::free(p); }
void  operator delete(void* p, size_t) noexcept { std::free(p); }
void  operator delete[](void* p, size_t) noexcept { std::free(p); }

/* ================================================================== the device fakes */
/* Weak, so a linked-in real host backend replaces them. Device memory is host memory here,
 * which is what makes the batch's device arrays readable below -- the assertions on cu_seqlens and
 * the slot mapping are reading exactly the bytes a kernel would. */
extern "C" {

__attribute__((weak)) void* rad_dev_alloc(int64_t bytes, int) {
    return std::calloc(1, (size_t)(bytes > 0 ? bytes : 1));
}
__attribute__((weak)) void rad_dev_free(void* p, int) { std::free(p); }

/* The builder binds a card before it allocates that rank's staging slab, because rad_dev_alloc has
 * no device argument. Stubbed here for the same reason the rest is -- and stubbed at all because a
 * MISSING one is worse than a wrong one: the linker would pull dispatch.cpp.o in to resolve it,
 * and that object's STRONG rad_dev_alloc/rad_memcpy_async would replace every weak fake above,
 * putting this test on the real HIP backend with the null stream it hands the scheduler. */
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

static int g_fake_event;
__attribute__((weak)) int  rad_event_create(RadEvent* e) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) int  rad_event_create_local(RadEvent* e) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) int  rad_event_create_as(RadEvent* e, unsigned) { *e = &g_fake_event; return RAD_OK; }
__attribute__((weak)) void rad_event_destroy(RadEvent) {}
__attribute__((weak)) int  rad_event_record(RadEvent, RadStream) { return RAD_OK; }
__attribute__((weak)) int  rad_event_sync(RadEvent) { return RAD_OK; }

}  /* extern "C" */

/* ================================================================== fake core/mem */
namespace {

/* A small but faithful block manager: a free list per group, refcounted blocks so an adopted
 * prefix-cache block is not returned to the pool while the cache still names it, and per-sequence
 * state slots for the linear and conv groups. Mirrors core/mem/kv.h's shape (sequence-keyed, the
 * manager owns the block tables) so the reconciliation is a rename and not a redesign. */
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
         * as the real manager keeps it so the builder's partial restage is what these tests run. */
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
        /* All or nothing: a partial grow would leave the sequence holding blocks it cannot use
         * and the caller unable to tell whether preempting would help. */
        for (size_t i = 0; i < gs.size(); ++i) {
            if (!geom.g[i].paged) continue;
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
    /* One column, the state itself: this fake has no kernel asking for scratch beside it. */
    int64_t state_index_width(int32_t) const override { return 1; }
    int state_index_row(uint64_t id, int32_t g, int32_t* out, int max) const override {
        if (!out || max <= 0) return 0;
        out[0] = state_slot(id, g);
        return 1;
    }
    int64_t first_block_pos(uint64_t, int32_t) const override { return 0; }
    int64_t take_table_changes(uint64_t id, int32_t g) const override {
        auto it = seqs.find(id);
        if (it == seqs.end()) return 0;
        const int64_t d = it->second.dirty[(size_t)g];
        it->second.dirty[(size_t)g] = INT64_MAX;
        return d;
    }
    /* Blocks an elastic pool has handed back to the card: still carved and free, but not
     * available to hand out until the pool grows again. free_blocks() is bounded by that and
     * held_blocks() is not, which is the difference an elastic pool presents. */
    int64_t unbacked = 0;
    int64_t free_blocks(int32_t g) const override {
        const int64_t n = (int64_t)freelist[(size_t)g].size() - unbacked;
        return n > 0 ? n : 0;
    }
    int64_t total_blocks(int32_t g) const override { return total[(size_t)g]; }
    int64_t held_blocks(int32_t g) const override {
        return total[(size_t)g] - (int64_t)freelist[(size_t)g].size();
    }
    /* A BYTE SIZE PER OBJECT SO THE SUM IS CHECKABLE. The real manager reads these off the carved
     * plan; here they are whatever the test set, and a group left at zero contributes nothing --
     * which is what a group the fixture does not care about should do. */
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
     * and hold a reference on them, which is what makes a hit safe -- a block the cache names is
     * never in the free list. */
    void seed(int32_t n_tokens) {
        attn = n_tokens;
        seeded.assign(groups.size(), {});
        for (size_t i = 0; i < groups.size(); ++i) {
            int32_t g = groups[i];
            int64_t nb = n_tokens / kv->geom.g[(size_t)g].block_size;
            for (int64_t b = 0; b < nb; ++b) seeded[i].push_back(kv->grab(g));
        }
    }

    /* The requests whose context is still being read off disk. */
    std::vector<uint64_t> reading;
    std::vector<uint64_t> restores;
    Restore restore(const Request& r) override {
        restores.push_back(r.id);
        return std::find(reading.begin(), reading.end(), r.id) != reading.end() ? Restore::Wait
                                                                                  : Restore::Done;
    }

    mutable int64_t lookups = 0;
    CacheHit lookup(const Request&, std::vector<std::vector<int32_t>>* out) const override {
        ++lookups;
        if (out) {
            out->resize(groups.size());
            for (size_t i = 0; i < groups.size() && i < seeded.size(); ++i)
                (*out)[i] = seeded[i];
        }
        return CacheHit{ attn, lin, src };
    }
    const std::vector<int32_t>& cached_groups() const override { return groups; }

    /* What the scheduler published, and what it asked to be evicted. Recorded rather than acted
     * on: these tests are about whether the scheduler publishes before it frees and evicts before
     * it preempts, not about the cache's own indexing, which prefix.cpp's tests cover. */
    struct Published { uint64_t seq; int64_t n_tokens; size_t n_groups; };
    std::vector<Published> published;
    int64_t evict_calls = 0, evict_grant = 0;

    int insert(const Request& r, const std::vector<std::vector<int32_t>>& blocks,
               int64_t n_tokens) override {
        published.push_back({ r.id, n_tokens, blocks.size() });
        return RAD_OK;
    }
    /* Grants only what a test asked it to, so "the cache had nothing to give" is reachable and
     * the preemption path below it still gets exercised. */
    int64_t evict(int64_t n) override {
        ++evict_calls;
        if (evict_grant <= 0) return 0;
        int64_t took = n < evict_grant ? n : evict_grant;
        evict_grant -= took;
        evicted += took;
        return took;
    }
    /* Cumulative, like the real cache's: the scheduler publishes it as a counter and the
     * server takes deltas, so a fake that reported per-call would read as a cache that
     * never evicts. */
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

/* ------------------------------------------------------------------ a declared program */
/* Two paged groups would be redundant here; what matters is that a paged group and a linear one
 * coexist, because the whole chunk geometry is the reconciliation between them.
 *
 * THE OP CARRIES A SCHEMA because that is where the chunk parameter is identified
 * (RAD_PROLE_SEQ_CHUNK), rather than by matching a list of spellings. The geometry and the
 * kernel's constraint must name the SAME key -- a constraint whose key the op's geometry does not
 * carry does not hold at all (rad_abi.h) -- so declaring "chunk" here and constraining on
 * "chunk_size" would be a pair the real system cannot produce. */
static const RadParamSpec g_gdn_ps[] = {
    { "M",     RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE },
    { "chunk", RAD_P_INT, RAD_DERIVED,  RAD_PROLE_SEQ_CHUNK },
};
static const RadOpSchema g_gdn_schema = { "gdn_chunk_scan", g_gdn_ps, 2, nullptr, 0, "test" };
static RadConstraint g_gdn_c[] = { RAD_CDIV("chunk", 64) };
static RadKernelInfo g_gdn_ki;
static KernelRow     g_gdn_row;

struct Fixture {
    Program prog;
    Config  cfg;
    FakeKV  kv;
    FakePC  pc;
    Scheduler sch;
    std::vector<void*> owned;
    RadStream stream = nullptr;

    ~Fixture() {
        sch.fini();
        if (stream) rad_stream_destroy(stream);
        for (void* p : owned) std::free(p);
    }

    void build(int64_t blocks = 4096, int64_t states = 64, int64_t max_tok = 512) {
        g_gdn_ki = RadKernelInfo{};
        g_gdn_ki.name = "gdn_fake";
        g_gdn_ki.op = "gdn_chunked";
        g_gdn_ki.constraints = g_gdn_c;
        g_gdn_ki.n_constraints = 1;
        g_gdn_row = KernelRow{ &g_gdn_ki, "fake", 0, 0 };

        prog.meta.n_ctx_train = 4096;

        /* index 0 is the sentinel every handle table in Program carries */
        prog.ops.emplace_back();
        prog.buffers.emplace_back();

        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        prog.kv_groups.push_back(attn);

        KVGroupInfo lin;
        lin.name = "gdn";
        lin.decl.kind = RAD_KV_LINEAR;
        lin.decl.state_dim[0] = 128; lin.decl.state_dim[1] = 128;
        prog.kv_groups.push_back(lin);

        KVGroupInfo conv;
        conv.name = "conv";
        conv.decl.kind = RAD_KV_CONV;
        conv.decl.conv_width = 4;
        prog.kv_groups.push_back(conv);

        OpInfo op;
        op.op = "gdn_chunk_scan";
        op.schema = &g_gdn_schema;
        Band b;
        b.hi = 4096;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_gdn_row;
        /* The architecture plugin's own declared geometry, in arch/common/rad_block_gdn.h's
         * spelling, which is the key its schema marks RAD_PROLE_SEQ_CHUNK. Both this and the
         * kernel's constraint contribute. */
        b.dom[RAD_DOMAIN_DEVICE].geom.set_i("chunk", 64);
        op.bands.push_back(b);
        prog.ops.push_back(op);

        add_derived("seq_start", "seq_start_index", 257);
        add_derived("chunk_bounds", "state_chunk_bounds", 4097);
        add_derived("conv_idx", "conv_state_index", 256);

        cfg.max_tok = max_tok;
        cfg.max_seqs = 32;
        cfg.max_ctx = 4096;
        cfg.checkpoint_interval = 256;
        cfg.prefix_cache = true;
        /* `Config::n_spec` defaults to -1, which is `auto`: a REQUEST that engine_bringup resolves
         * against rad_arch_probe once the plugin is selected. This fixture builds a Scheduler
         * directly and so never runs that resolution, and DraftController::configure rightly
         * refuses a negative depth. The cases that are about speculation set their own depth. */
        cfg.n_spec = 0;

        kv.setup(prog.kv_groups, cfg.max_ctx, blocks, states);
        pc.kv = &kv;
        pc.groups.clear();
        for (size_t i = 0; i < prog.kv_groups.size(); ++i)
            if (kv.geom.g[i].paged) pc.groups.push_back((int32_t)i);
    }

    void add_derived(const char* name, const char* derive, int64_t n, rad_kvgroup kv = 0) {
        BufferInfo b;
        b.name = name;
        b.decl.dtype = RAD_I32;
        b.decl.rank = 1;
        b.decl.shape[0] = n;
        b.decl.kind = RAD_BUF_DERIVED;
        b.decl.derive = derive;
        b.decl.kv = kv;
        b.bytes = n * 4;
        void* p = std::calloc(1, (size_t)b.bytes);
        owned.push_back(p);
        b.ptr = p;
        prog.buffers.push_back(b);
    }

    int start() {
        if (!stream) rad_stream_create(&stream, 0);
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, stream, &prog });
        return sch.init(prog, cfg, &kv, &pc, rio);
    }

    const int32_t* derived(const char* name) const {
        for (const BufferInfo& b : prog.buffers)
            if (b.name == name) return (const int32_t*)b.ptr;
        return nullptr;
    }

    uint64_t next_id = 1;
    uint64_t submit(int n_prompt, int max_tokens = 64, int priority = 0, bool ignore_eos = false) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->priority = priority;
        r->ignore_eos = ignore_eos;
        r->max_tokens = max_tokens;
        r->prompt.resize((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) r->prompt[(size_t)i] = 1000 + i;
        uint64_t id = r->id;
        if (sch.add(std::move(r)) < 0) return 0;
        return id;
    }

    /* The same, with the server's sink attached, for what a request tells its client. */
    uint64_t submit_to(server::Sink* sink, int n_prompt, int max_tokens) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->max_tokens = max_tokens;
        r->sink = sink;
        r->prompt.resize((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) r->prompt[(size_t)i] = 1000 + i;
        uint64_t id = r->id;
        if (sch.add(std::move(r)) < 0) return 0;
        return id;
    }

    /* Commit a step in which every sequence produced one token and none ended. */
    void commit_plain(int32_t token = 7) {
        const StepPlan& p = sch.plan();
        std::vector<int32_t> acc((size_t)p.e.size(), 0), tok((size_t)p.e.size(), token);
        std::vector<uint8_t> eos((size_t)p.e.size(), 0);
        Scheduler::StepResult r;
        r.n_seq = (int)p.e.size();
        r.n_accepted = acc.data();
        r.token = tok.data();
        r.eos = eos.data();
        CHECK_OK(sch.commit(r));
    }
};

}  /* namespace */

/* ================================================================== geometry + chunking */
TEST(chunk_geometry_is_computed_not_configured) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    const ChunkGeometry& g = f.sch.geometry();
    CHECK_EQ(g.attn_block, 16);           /* off the paged group */
    CHECK_EQ(g.state_chunk, 64);          /* off the resolved GDN kernel's DIV constraint */
    CHECK_EQ(g.quantum, 64);              /* lcm */
    CHECK_EQ(g.checkpoint_interval, 256); /* already a multiple of the quantum */
    CHECK_EQ(g.min_max_tok, 64);
}

TEST(chunk_geometry_rounds_the_checkpoint_interval_up_to_a_legal_chunk) {
    Fixture f;
    f.build();
    f.cfg.checkpoint_interval = 200;      /* not a multiple of 64 */
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().checkpoint_interval, 256);
}

TEST(chunk_geometry_refuses_a_budget_it_cannot_serve) {
    Fixture f;
    f.build();
    f.cfg.max_tok = 32;                   /* below one quantum */
    log_set_level(Log::Error);
    CHECK_EQ(f.start(), RAD_E_INVAL);
}

TEST(prefill_chunks_split_at_the_checkpoint_interval) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    int64_t ck = -1;
    /* The budget would allow 512 tokens; the split takes 256 so the chunk ENDS on the interval. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck), 256);
    CHECK_EQ(ck, 256);

    CHECK_EQ(f.sch.plan_chunk(256, 1000, 512, &ck), 256);
    CHECK_EQ(ck, 512);

    /* The final chunk is allowed to be short -- that is the accepted cost of owning the
     * boundaries -- and writes no checkpoint because it does not reach one. */
    CHECK_EQ(f.sch.plan_chunk(768, 1000, 512, &ck), 232);
    CHECK_EQ(ck, -1);

    /* A budget short of the interval still ends on a quantum, never in the middle of a tile. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 100, &ck), 64);
    CHECK_EQ(ck, -1);

    /* A budget that cannot hold one quantum schedules nothing rather than a misaligned chunk. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 50, &ck), 0);
}

TEST(a_chunk_ending_on_a_checkpoint_carries_it_into_the_batch) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.submit(1000, 8);

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, 256);
    CHECK_EQ(b->n_checkpoints, 1);
    /* The snapshot is the state after the LAST token of the chunk, and the split put that token
     * at the interval boundary. */
    CHECK_EQ(b->checkpoint_tok[0], 255);
    CHECK_EQ(b->checkpoint_slot[0], 0);
    CHECK_EQ(b->phase, RAD_PHASE_PREFILL);
}

/* ================================================================== admission */
TEST(admission_refuses_when_the_kv_pool_is_short) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);   /* 20 x 16 = 320 tokens of paged KV */
    CHECK_OK(f.start());

    f.submit(160, 8);      /* 10 blocks */
    f.submit(160, 8);      /* 10 blocks -- exactly fills the pool */
    f.submit(160, 8);      /* must not be admitted */

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_seq, 2);
    CHECK_EQ(f.kv.free_blocks(0), 0);

    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.running, 2);
    CHECK_EQ(m.queue_depth, 1);
    CHECK_EQ(m.admitted, 2);
}

/* An observer reads the copy the engine thread published, never the live state, so what it sees
 * moves only at a publication -- and a publication carries exactly what the live readers say. */
TEST(observers_read_what_the_engine_thread_published) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    f.submit(160, 8);
    f.submit(160, 8);
    f.submit(160, 8);
    CHECK(f.sch.step() != nullptr);

    std::vector<Scheduler::ReqStat> seen(8), live(8);
    CHECK_EQ(f.sch.observed_metrics().admitted, 0);            /* nothing published yet */
    CHECK_EQ(f.sch.observed_requests(seen.data(), 8), 0);

    f.sch.publish(/*idle=*/false);
    const SchedMetrics o = f.sch.observed_metrics();
    const SchedMetrics m = f.sch.metrics();
    CHECK_EQ(o.running, m.running);
    CHECK_EQ(o.queue_depth, m.queue_depth);
    CHECK_EQ(o.admitted, 2);
    const int n = f.sch.observed_requests(seen.data(), 8);
    CHECK_EQ(n, f.sch.requests(live.data(), 8));
    CHECK_EQ(n, 3);
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(seen[(size_t)i].id, live[(size_t)i].id);
        CHECK_EQ(seen[(size_t)i].state, live[(size_t)i].state);
        CHECK_EQ(seen[(size_t)i].prompt_tokens, live[(size_t)i].prompt_tokens);
    }

    /* An idle publication inside 20 ms of the last one is skipped; a step's never is. */
    f.submit(160, 8);
    f.sch.publish(/*idle=*/true);
    CHECK_EQ(f.sch.observed_metrics().queue_depth, 1);
    f.sch.publish(/*idle=*/false);
    CHECK_EQ(f.sch.observed_metrics().queue_depth, 2);
    CHECK_EQ(f.sch.observed_requests(seen.data(), 8), 4);
}

/* The sequences of one API request go in under one hold of the lock, each accepted or refused on
 * its own terms: a duplicate id is refused without taking its neighbours with it. */
TEST(add_all_queues_a_set_under_one_lock_and_refuses_members_singly) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    std::unique_ptr<Request> rs[3];
    for (int i = 0; i < 3; ++i) {
        rs[i] = std::make_unique<Request>();
        rs[i]->id = f.next_id++;
        rs[i]->max_tokens = 8;
        rs[i]->prompt.assign(32, 1000 + i);
    }
    rs[2]->id = rs[0]->id;
    int rc[3] = { -1, -1, -1 };
    f.sch.add_all(rs, 3, rc);
    CHECK_EQ(rc[0], RAD_OK);
    CHECK_EQ(rc[1], RAD_OK);
    CHECK_EQ(rc[2], RAD_E_DUPLICATE);

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_seq, 2);
    CHECK_EQ(f.sch.metrics().admitted, 2);
}

TEST(a_prefix_hit_starts_the_request_at_the_last_usable_checkpoint) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(304);       /* the attention blocks match 304 tokens deep (19 blocks) */
    f.pc.lin = 256;       /* but the last linear checkpoint is at 256 */
    f.pc.src = 3;
    int64_t free_before = f.kv.free_blocks(0);

    uint64_t a = f.submit(1000, 8);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    /* Recompute resumes from the checkpoint rather than replaying only the linear layers over
     * (256, 300], because RadBatch carries ONE query length for every KV group. The chunk still
     * ends on the next interval, so the next checkpoint is still written. */
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    CHECK_EQ(b->ctx_lens[0], 256);
    CHECK_EQ(b->n_tok, 256);
    CHECK_EQ(b->n_checkpoints, 1);
    CHECK_EQ(b->checkpoint_tok[0], 255);

    /* THE RECURRENT HALF OF THE HIT IS CARRIED, and it is carried exactly once. The adopted blocks
     * describe the attention layers; the linear ones resume from this slot and from nothing
     * else, so an entry that skipped 256 tokens without naming a snapshot would be a sequence
     * continuing from whatever the slot last held. */
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 3);

    /* The hit is CLAIMED, not merely counted: the sequence adopted the cache's 16 blocks, which
     * cost the pool nothing because the cache already held them, and then allocated only the 16
     * new blocks its 256-token chunk needs. A hit that is not adopted is a sequence attending to
     * blocks it does not own. */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 32);
    CHECK_EQ(f.kv.block_table(a, 0)[0], f.pc.seeded[0][0]);
    CHECK_EQ(f.kv.free_blocks(0), free_before - 16);

    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.prompt_tokens, 1000);
    CHECK_EQ(m.attn_cached_tokens, 304);     /* what the block cache matched */
    CHECK_EQ(m.linear_cached_tokens, 256);   /* what was actually skipped */
    CHECK_EQ(m.checkpoints_written, 1);

    /* ...and the snapshot this step takes is not offered to anyone until the step has run. The
     * reservation happened when the chunk was planned; the state exists only afterwards. */
    CHECK(f.pc.committed.empty());
    f.commit_plain();
    CHECK_EQ((int)f.pc.committed.size(), 1);
    CHECK_EQ(f.pc.committed[0], 512);        /* ctx_len 256 + ckpt_tok 255 + 1 */
}

/* A hit that skipped tokens with no snapshot to resume from is the corruption this whole mechanism
 * exists to prevent, so it is refused outright rather than served with a zeroed state. */
TEST(a_hit_with_no_checkpoint_slot_reuses_nothing) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(304);
    f.pc.lin = 256;
    f.pc.src = -1;        /* the cache found a prefix but no snapshot covering it */

    uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().linear_cached_tokens, 0);
}

TEST(a_hit_that_no_checkpoint_covers_replays_from_zero) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(192);
    f.pc.lin = 0;         /* nothing checkpointed yet */

    uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.metrics().linear_cached_tokens, 0);
}

/* ================================================================== preemption */
/* A REQUEST WHOSE CONTEXT IS STILL BEING READ OFF DISK WAITS, AND ONLY IT DOES. It is not admitted
 * short -- the hit length is what its prefill is shaped by -- and it does not hold up the queue:
 * the request behind it is asked and runs. Once the read is in, it is admitted where it stood. */
TEST(a_request_waiting_on_a_disk_read_lets_the_queue_past_it) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    const uint64_t a = f.submit(160, 8);
    const uint64_t b = f.submit(160, 8);
    const uint64_t c = f.submit(160, 8);
    f.pc.reading = {a, c};

    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ(s1->n_seq, 1);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.pc.lookups, 1);                    /* nothing looked up for the ones that wait */
    uint64_t q[8];
    int n = f.sch.queue_snapshot(q, 8);
    REQUIRE_EQ(n, 2);
    CHECK_EQ(q[0], a);                            /* still in the order they came */
    CHECK_EQ(q[1], c);
    f.commit_plain();

    /* Asked again every step, and admitted the step its read is in. */
    f.pc.reading = {c};
    f.pc.restores.clear();
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->n_seq, 2);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Waiting);
    CHECK(std::find(f.pc.restores.begin(), f.pc.restores.end(), c) != f.pc.restores.end());
    f.commit_plain();

    f.pc.reading.clear();
    const RadBatch* s3 = f.sch.step();
    CHECK(s3 != nullptr);
    CHECK_EQ(s3->n_seq, 3);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.queue_snapshot(q, 8), 0);
    CHECK_EQ(f.sch.metrics().admitted, 3);
}

TEST(preemption_takes_the_lowest_priority_victim_and_requeues_it_to_the_front) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64, /*priority=*/0);   /* arrives first, low priority */
    uint64_t b = f.submit(160, 64, /*priority=*/5);   /* arrives second, high priority */
    uint64_t c = f.submit(160, 64, /*priority=*/0);   /* stays waiting */
    CHECK(a && b && c);

    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ(s1->n_seq, 2);
    f.commit_plain();

    /* Both are now decoding at position 160, which is a block boundary: the next token needs a
     * new block and the pool has none. The victim must be A -- lowest priority, and the one that
     * would be re-admitted last anyway -- not the head-of-line B. */
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->n_seq, 1);

    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);      /* preemption is by RECOMPUTE */
    CHECK_EQ(f.sch.find(a)->state_slot, -1);

    /* Re-queued to the FRONT: it keeps its original arrival, so it sits ahead of C, which arrived
     * later at the same priority. It has already been served once and its blocks were taken from
     * it; sending it to the back would charge it twice. */
    uint64_t q[8];
    int n = f.sch.queue_snapshot(q, 8);
    CHECK_EQ(n, 2);
    CHECK_EQ(q[0], a);
    CHECK_EQ(q[1], c);

    CHECK_EQ(f.sch.metrics().preemptions, 1);
    CHECK_EQ(f.sch.metrics().preempted_waiting, 1);
}

/* ================================================================== the step batch */
TEST(a_mixed_step_has_the_right_boundaries_and_slot_mapping) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t a = f.submit(100, 64);
    CHECK(a);
    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ(s1->n_tok, 100);
    f.commit_plain(4242);

    /* A is decoding at position 100; B arrives and prefills 64 tokens in the same step. */
    uint64_t b = f.submit(64, 64);
    CHECK(b);
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->phase, RAD_PHASE_MIXED);
    CHECK_EQ(s2->n_seq, 2);
    CHECK_EQ(s2->n_tok, 65);

    CHECK_EQ(s2->cu_seqlens[0], 0);
    CHECK_EQ(s2->cu_seqlens[1], 1);
    CHECK_EQ(s2->cu_seqlens[2], 65);
    CHECK_EQ(s2->q_lens[0], 1);
    CHECK_EQ(s2->q_lens[1], 64);
    CHECK_EQ(s2->ctx_lens[0], 100);
    CHECK_EQ(s2->ctx_lens[1], 0);
    CHECK_EQ(s2->max_q_len, 64);
    /* A CEILING ON A COARSE GRID, not the exact longest context: 100 rounds up to the 1024-token
     * floor, so the step's arguments stay put while the context grows under it. */
    CHECK_EQ(s2->max_ctx_len, 1024);

    /* The decode row feeds the token that was sampled last step, not a prompt token. */
    CHECK_EQ(s2->token_ids[0], 4242);
    CHECK_EQ(s2->positions[0], 100);
    CHECK_EQ(s2->token_ids[1], 1000);
    CHECK_EQ(s2->positions[1], 0);

    /* Slot mapping for the paged group: block id times block size plus the offset in the block.
     * Position 100 is block 6, offset 4. */
    const RadKVGroupBatch& g0 = s2->kv[0];
    CHECK_EQ(g0.block_size, 16);
    const std::vector<int32_t>& bta = f.kv.block_table(a, 0);
    const std::vector<int32_t>& btb = f.kv.block_table(b, 0);
    CHECK_EQ(g0.slot_mapping[0], bta[6] * 16 + 4);
    /* B's first prefill token is the first slot of its first block. */
    CHECK_EQ(g0.slot_mapping[1], btb[0] * 16);
    /* Block tables are one row per sequence, pitched from the step's context bound so the two
     * move together: the 1024-token bound, plus the longest query (64) and the draft lookahead
     * (2), is 1090 tokens, 69 blocks of 16, rounded up to the bound's 1024-token grid of 64
     * blocks -- 128. The columns past a row's own blocks read -1. */
    CHECK_EQ(g0.block_table_pitch, 128);
    CHECK_EQ(g0.block_table[0], bta[0]);
    CHECK_EQ(g0.block_table[6], bta[6]);
    CHECK_EQ(g0.block_table[7], -1);
    CHECK_EQ(g0.block_table[128], btb[0]);
    CHECK_EQ(g0.seqused[0], 101);
    CHECK_EQ(g0.seqused[1], 64);

    /* The linear group is per sequence, not per token: every token of a sequence maps to its one
     * state instance, and the block table row is that slot. */
    const RadKVGroupBatch& g1 = s2->kv[1];
    CHECK_EQ(g1.slot_mapping[0], f.kv.state_slot(a, 1));
    CHECK_EQ(g1.slot_mapping[1], f.kv.state_slot(b, 1));
    CHECK_EQ(g1.state_index[0], f.kv.state_slot(a, 1));
    CHECK(f.kv.state_slot(a, 1) >= 0);
}

TEST(derived_metadata_is_computed_once_per_step_into_the_arena) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    f.submit(100, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    f.submit(64, 64);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);

    const int32_t* ss = f.derived("seq_start");
    CHECK(ss != nullptr);
    CHECK_EQ(ss[0], 0);
    CHECK_EQ(ss[1], 1);
    CHECK_EQ(ss[2], 65);

    /* Chunk bounds never cross a sequence, and split on ABSOLUTE multiples of the chunk length.
     * The decode row spans position 100..100 and the prefill row 0..63, so neither crosses 64 --
     * one chunk each, boundaries at 0, 1, 65. */
    const int32_t* cb = f.derived("chunk_bounds");
    CHECK(cb != nullptr);
    CHECK_EQ(cb[0], 0);
    CHECK_EQ(cb[1], 1);
    CHECK_EQ(cb[2], 65);

    /* The conv index is the state slot's base in a rolling window of width-1 + num_spec + 1 = 4,
     * plus the cursor the accepted count moves. */
    const int32_t* ci = f.derived("conv_idx");
    CHECK(ci != nullptr);
    const StepPlan& p = f.sch.plan();
    int32_t cslot = f.kv.state_slot(p.e[0].seq, 2);   /* group 2 is the conv group */
    CHECK(cslot >= 0);
    CHECK_EQ(ci[0], cslot * 4 + (p.e[0].conv_cursor % 4));
}

TEST(state_chunk_bounds_split_a_long_prefill_at_absolute_multiples) {
    DeriveInput in;
    StepPlan plan;
    int32_t cu[2] = { 0, 200 };
    std::vector<int32_t> pos(200);
    for (int i = 0; i < 200; ++i) pos[(size_t)i] = 100 + i;   /* starts mid-chunk on purpose */
    in.plan = &plan;
    in.cu_seqlens = cu;
    in.positions = pos.data();
    in.n_tok = 200;
    in.n_seq = 1;
    in.state_chunk = 64;

    int idx = derive_index("state_chunk_bounds");
    CHECK(idx >= 0);
    int32_t out[16];
    int64_t n = 0;
    CHECK_OK(derive_row(idx)->fn(in, out, 16, &n));
    /* Absolute 128, 192, 256 fall inside [100, 300): local 28, 92, 156, then the tail. */
    CHECK_EQ(n, 5);
    CHECK_EQ(out[0], 0);
    CHECK_EQ(out[1], 28);
    CHECK_EQ(out[2], 92);
    CHECK_EQ(out[3], 156);
    CHECK_EQ(out[4], 200);

    CHECK_EQ(derive_index("no_such_derivation"), -1);
}

/* ================================================================== cancellation */
TEST(cancellation_frees_blocks_immediately) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    f.commit_plain();

    /* Not at completion, and not at the next step: now. A cancelled request that keeps its pool
     * share until it would have finished is a pool that shrinks under exactly the load that
     * generates cancellations. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.sch.metrics().running, 0);
    CHECK_EQ(f.sch.metrics().cancelled, 1);
    CHECK_OK(f.sch.reap(a));

    /* A step that is still on the device is the one exception, and it is a one-step deferral and
     * not a one-request one: releasing a block a running kernel is writing hands it to a new
     * sequence mid-step, and that presents as a model quality problem rather than as a crash. */
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    CHECK_OK(f.sch.cancel(b));
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.sch.metrics().running, 0);          /* out of the running set at once */
    CHECK_EQ(f.kv.free_blocks(0), before - 10);    /* but the blocks are still the device's */
    f.commit_plain();
    CHECK_EQ(f.kv.free_blocks(0), before);
    /* And it is not resurrected: the commit must not append the token the step sampled for a
     * request that was cancelled while that step was running. */
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Cancelled);
}

/* A REAP THAT ARRIVES WHILE THE STEP IS STILL ON THE DEVICE, which is not a corner case: the
 * server reaps a generation as soon as its sink is finished, and cancelling a request finishes the
 * sink at once, so EVERY client that disconnects mid-decode arrives here. Releasing the Request
 * there destroys it under commit(), which then reads `r.state` off a null unique_ptr and takes the
 * process down. Returning the SLOT there is the worse half of the same failure and leaves no
 * trace: a request admitted before the commit lands is handed the entry, and one client's sampled
 * token is appended to another client's output. */
TEST(a_reap_during_the_open_step_is_deferred_to_commit) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);

    /* What Server::cancel_all and Server::retire do back to back on a disconnect. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));

    /* The Request outlives the reap by exactly one commit, and so does its slot: neither the id
     * nor the blocks are released while a StepEntry still points at them. */
    CHECK(f.sch.find(a) != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);

    /* A request admitted in that window must not be given the slot the open step is holding. */
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.find(b) != nullptr);

    f.commit_plain();
    CHECK(f.sch.find(a) == nullptr);              /* released now, and exactly once */
    CHECK_EQ(f.kv.free_blocks(0), before);

    /* And the request admitted in the window is untouched by the commit that reaped the other. */
    CHECK(f.sch.find(b) != nullptr);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    f.commit_plain();
    CHECK(f.sch.find(b) != nullptr);
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));
    CHECK_EQ(f.kv.free_blocks(0), before);
}

/* ================================ a full-context drafter refuses prefix reuse ================================
 *
 * THE GATE AT ADMISSION. A full-context drafter fills a PERSISTENT K/V cache by appending the
 * target's verified taps chunk by chunk, so a prefix hit -- which reuses the target's blocks and
 * skips recomputing the prompt -- would leave its `length[slot]` counting positions its cache never
 * filled, and every later attention would read them as context the target never produced. There is
 * no rule that repairs that afterwards, so the REUSE is what goes (Scheduler::init,
 * core/sched/scheduler.cpp) and no hit is ever taken.
 *
 * THIS IS OBSERVED WHERE IT MATTERS: the request is submitted, the step is planned, and the numbers
 * the engine actually produced are read -- `ctx_len` (where the prefill starts) and `n_tok` (how
 * much of the prompt it covers). A full-context drafter must plan the WHOLE prompt from 0; the same
 * fixture with the same cache and the same hit, declaring the ring op instead, must reuse it.
 *
 * The two runs differ in ONE declared op, so removing the gate makes the first run take the hit and
 * this test fails on the planned numbers rather than on a restated rule.
 *
 * The cache is the fixture's own FakePC, answering a concrete 64-token hit for a 160-token prompt. */
namespace {

/* The program a full-context drafter declares, distinguished by the one op only that block names.
 * `Fixture::build` already sets up the groups, the derived arrays and the KV geometry; this only
 * adds the marker op and rebuilds the geometry the manager mirrors. */
void mark_drafter(Fixture& f, const char* op_name) {
    OpInfo op;
    op.op = op_name;
    f.prog.ops.push_back(op);
    /* The declared ops are what the gate asks about, and they are read at init, so the marker has to
     * be in place before start(). */
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups.clear();
    for (size_t i = 0; i < f.prog.kv_groups.size(); ++i)
        if (f.kv.geom.g[i].paged) f.pc.groups.push_back((int32_t)i);
}

}  /* namespace */

TEST(a_full_context_drafter_takes_no_prefix_hit_while_another_drafter_still_does) {
    constexpr int kPrompt = 160;

    /* A RING-STYLE DRAFTER: the cache stays, and the hit is taken. This is the control -- it shows
     * the hit is real and would be observable if the gate were not there. */
    {
        Fixture f;
        f.build();
        mark_drafter(f, "dloop_ctx_append");
        f.cfg.checkpoint_interval = 64;
        CHECK_OK(f.start());
        f.pc.seed(64);          /* an earlier request cached 64 tokens of this prompt */
        f.pc.lin = 64;
        f.pc.src = 3;
        uint64_t a = f.submit(kPrompt, 8);
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        /* THE REUSE HAPPENED: the prefill starts at the cached 64 and only the rest is planned. */
        CHECK_EQ(f.sch.find(a)->n_computed, 64);
        CHECK_EQ(f.sch.plan().e[0].ctx_len, 64);
        CHECK(f.pc.lookups > 0);
    }

    /* THE SAME CACHE, THE SAME HIT, THE SAME PROMPT -- and a full-context drafter. */
    {
        Fixture f;
        f.build();
        mark_drafter(f, "dloop_full_append");
        f.cfg.checkpoint_interval = 64;
        CHECK_OK(f.start());
        f.pc.seed(64);
        f.pc.lin = 64;
        f.pc.src = 3;
        uint64_t a = f.submit(kPrompt, 8);
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        /* No prefix is skipped: the first checkpoint-sized prefill chunk starts at position zero. */
        CHECK_EQ(f.sch.find(a)->n_computed, 0);
        CHECK_EQ(f.sch.plan().e[0].ctx_len, 0);
        CHECK_EQ(b->n_tok, 64);
        /* AND THE CACHE WAS NEVER CONSULTED, which is stronger than "the hit was declined": the
         * lookup is the thing that must not run, because a lookup that returned a hit some other
         * code path then honoured would not be visible in the planned numbers alone. */
        CHECK_EQ(f.pc.lookups, 0);
    }
}

/* ============================================ a block drafter's own cache across a prefix hit */
TEST(a_prefix_hit_does_not_stop_a_full_attention_drafter) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 7;                 /* copied into the scheduler at init, so it goes first */
    CHECK_OK(f.start());

    /* A REAL HIT, both halves of it: the attention blocks AND the linear checkpoint have to
     * reach the same depth or the scheduler recomputes from 0 and ctx_len is 0 -- which is a
     * test that passes whatever this code does. Seeding only the paged half takes no hit and
     * makes this case vacuous. */
    f.pc.seed(256);
    f.pc.lin = 256;
    f.pc.src = 3;

    uint64_t a = f.submit(272, 64);
    CHECK(f.sch.step() != nullptr);                    /* recomputes 0..256 off the checkpoint */
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);                    /* 256..272, and this one yields a token */
    CHECK_EQ(f.sch.find(a)->n_computed, 272);
    /* Non-zero is the whole point: this is the floor the guard would take after a hit. */
    CHECK(f.sch.plan().e[0].ctx_len > 0);
    f.commit_plain(11);

    /* window 0 is a drafter that attends over its whole history, so "wait for the window to
     * slide past the hit" is a condition that can never come true. Every paged group here IS
     * prefix-cached, so the hit handed over blocks the previous sequence's own context pass
     * wrote: there is no hole and the floor is 0. Taking the floor to be the hit position
     * instead stops a windowless drafter from ever drafting again after a hit.
     *
     * THE OTHER SIDE -- a drafter whose group the cache SKIPS must still decline -- is not
     * testable here: giving the fixture a second paged group the FakePC does not index makes
     * its lookup hand back a block list that does not cover it, so the hit is refused and
     * ctx_len is 0 either way. That case needs a windowed drafter over a cache that leaves a
     * hole, which this fixture cannot construct. */
    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(/*block=*/8, /*depth=*/7, /*window=*/0, /*mask=*/99, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)a);
}

/* ================================================================== speculation */
TEST(rejection_rolls_back_paged_kv_and_moves_the_linear_read_offset) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));

    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_spec, 4);
    CHECK_EQ(b->n_tok, 5);
    CHECK_EQ(b->q_lens[0], 5);
    CHECK_EQ(b->token_ids[0], 11);
    CHECK_EQ(b->token_ids[1], 21);
    CHECK_EQ(b->token_ids[4], 24);

    int64_t blocks_before = f.kv.free_blocks(0);
    int32_t cursor_before = f.sch.plan().e[0].conv_cursor;

    /* Two of the four drafts survive. */
    int32_t acc[1] = { 2 };
    int32_t tok[1] = { 99 };
    uint8_t eos[1] = { 0 };
    Scheduler::StepResult r;
    r.n_seq = 1; r.n_accepted = acc; r.token = tok; r.eos = eos;
    CHECK_OK(f.sch.commit(r));

    const Request* ra = f.sch.find(a);
    /* Three tokens committed: the two accepted drafts and the bonus. */
    CHECK_EQ((int)ra->output.size(), 4);
    CHECK_EQ(ra->output[1], 21);
    CHECK_EQ(ra->output[2], 22);
    CHECK_EQ(ra->output[3], 99);
    CHECK_EQ(ra->n_computed, 163);
    CHECK_EQ(ra->n_accepted, 2);
    /* Paged KV gave blocks back for the rejected tail; the state slot did not move, because a
     * linear rollback is a read offset and not a free. */
    CHECK(f.kv.free_blocks(0) >= blocks_before);
    CHECK(f.kv.state_slot(a, 1) >= 0);

    /* The conv cursor advanced by 1 + accepted, modulo the rolling window. */
    f.submit(1, 1);   /* nothing; just to force a plan rebuild below */
    const RadBatch* b2 = f.sch.step();
    CHECK(b2 != nullptr);
    int32_t W = 4 - 1 + 4 + 1;
    CHECK_EQ(f.sch.plan().e[0].conv_cursor, (cursor_before + 3) % W);
}

/* A decode measurement needs the request to last as long as the harness asked for; otherwise the
 * model decides how many steps are measured, one prompt ending after eight tokens and the next
 * running for hundreds, so the window measures the prompt rather than the change under test. Both
 * halves are pinned here -- that EOS stops the ordinary request, and that max_tokens still stops
 * the one ignoring it, because a flag that suppressed BOTH would be an endless generation rather
 * than a benchmark. */
TEST(ignore_eos_runs_to_max_tokens_and_nothing_else_changes) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t plain = f.submit(8, /*max_tokens=*/4);
    uint64_t keep  = f.submit(8, /*max_tokens=*/4, /*priority=*/0, /*ignore_eos=*/true);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(7);                  /* both prefilled, one token each */

    /* Every sequence emits end-of-generation on this step. */
    const StepPlan& p = f.sch.plan();
    CHECK_EQ((int)p.e.size(), 2);
    std::vector<int32_t> acc((size_t)p.e.size(), 0), tok((size_t)p.e.size(), 9);
    std::vector<uint8_t> eos((size_t)p.e.size(), 1);
    CHECK(f.sch.step() != nullptr);
    Scheduler::StepResult r;
    r.n_seq = (int)p.e.size();
    r.n_accepted = acc.data(); r.token = tok.data(); r.eos = eos.data();
    CHECK_OK(f.sch.commit(r));

    CHECK_EQ((int)f.sch.find(plain)->state, (int)ReqState::Finished);
    CHECK_EQ(f.sch.find(plain)->finish_reason, std::string("stop"));
    CHECK_EQ((int)f.sch.find(keep)->state, (int)ReqState::Running);

    /* It ends on the budget instead, and says so. */
    while (f.sch.find(keep)->state == ReqState::Running && f.sch.step() != nullptr) {
        const StepPlan& q = f.sch.plan();
        std::vector<int32_t> a2((size_t)q.e.size(), 0), t2((size_t)q.e.size(), 9);
        std::vector<uint8_t> e2((size_t)q.e.size(), 1);
        Scheduler::StepResult r2;
        r2.n_seq = (int)q.e.size();
        r2.n_accepted = a2.data(); r2.token = t2.data(); r2.eos = e2.data();
        CHECK_OK(f.sch.commit(r2));
    }
    const Request* rk = f.sch.find(keep);
    CHECK_EQ((int)rk->state, (int)ReqState::Finished);
    CHECK_EQ(rk->finish_reason, std::string("length"));
    CHECK_EQ((int)rk->output.size(), 4);
}

/* A block that wants to treat the two kinds of row differently can only do it if each kind is a
 * contiguous run of sequences, because cu_seqlens is read as cu[n] and cu[n+1]. The scheduler
 * walks `running_` in PRIORITY order, so without the sort a high-priority request part way
 * through its prompt lands ahead of a low-priority one that is decoding -- which is exactly the
 * arrangement built here. */
TEST(a_mixed_step_puts_its_decode_rows_first) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t b = f.submit(1000, 64, /*priority=*/1);   /* several chunks, sorts first */
    uint64_t a = f.submit(160,  64, /*priority=*/0);   /* one chunk, then decodes */

    const RadBatch* s0 = f.sch.step();
    CHECK(s0 != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ(f.sch.plan().n_seq_decode, 0);            /* both prefill: nothing decodes yet */
    f.commit_plain(11);

    /* a finished its prompt and is decoding; b is on its second chunk and still outranks it. */
    const RadBatch* s1 = f.sch.step();
    CHECK(s1 != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ(f.sch.plan().e[0].seq, a);                /* the decode row, despite the priority */
    CHECK_EQ(f.sch.plan().e[0].is_prefill, false);
    CHECK_EQ(f.sch.plan().e[1].seq, b);
    CHECK_EQ(f.sch.plan().e[1].is_prefill, true);
    CHECK_EQ(f.sch.plan().n_seq_decode, 1);
    CHECK_EQ(s1->n_seq_decode, 1);
    CHECK_EQ(s1->phase, RAD_PHASE_MIXED);
    /* The per-half bounds: a decodes one row, b's chunk is the interval. */
    CHECK_EQ(s1->max_q_len_decode, 1);
    CHECK_EQ(s1->max_q_len_prefill, 256);
    CHECK_EQ(s1->max_q_len, 256);

    /* AND THE ENTRY INDEX FOLLOWED THE ROW. `entry` is how a request finds its own result, so a
     * sort that leaves it behind hands one request another's token. */
    f.commit_plain(12);
    CHECK_EQ((int)f.sch.find(a)->output.size(), 2);    /* 11 from the prefill, 12 from the decode */
    CHECK_EQ(f.sch.find(a)->output[1], 12);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);    /* still prefilling: no token yet */
}

/* A STEP THAT CARRIES A PREFILL CHUNK STILL TAKES A DRAFT. Issuing the chunked path for every
 * sequence in a mixed step would force the draft to be refused, because that path has no
 * num_accepted operand to roll a rejection back with; the blocks issue the recurrent path over the
 * decode half instead. These are the two shapes such a refusal would fire on. */
TEST(a_queued_request_does_not_suppress_the_draft) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    f.cfg.max_seqs = 3;                      /* a slot is free, so the queue WILL be admitted */
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64);
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    f.submit(160, 64);
    CHECK_EQ(f.sch.metrics().queue_depth, 1);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));

    const RadBatch* s = f.sch.step();
    CHECK(s != nullptr);
    CHECK_EQ(s->n_spec, 4);                  /* the draft survives the admission */
    CHECK_EQ((int)f.sch.plan().e.size(), 3); /* and the third was admitted into the same step */
    CHECK_EQ(f.sch.metrics().queue_depth, 0);
    CHECK_EQ(s->phase, RAD_PHASE_MIXED);
    /* The decode rows sorted first and carry the window; the prefill chunk carries none. */
    CHECK_EQ(s->n_seq_decode, 2);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
    CHECK_EQ(f.sch.plan().e[1].n_spec, 4);
    CHECK_EQ(f.sch.plan().e[2].n_spec, 0);
    /* Two decode rows of 1 + 4, so the split the blocks issue over is 10 tokens and then the
     * chunk. Getting this wrong is how a per-token op reaches into the other half. */
    CHECK_EQ(s->n_tok_decode, 10);
    CHECK_EQ(s->max_q_len_decode, 5);
    CHECK_EQ(s->max_q_len_prefill, 160);
}

TEST(a_running_prefill_does_not_suppress_the_draft) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160,  64);
    f.submit(1000, 64);                      /* several chunks: mixed for the next few steps */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));

    const RadBatch* s = f.sch.step();
    CHECK(s != nullptr);
    CHECK_EQ(s->phase, RAD_PHASE_MIXED);
    CHECK_EQ(s->n_spec, 4);
    CHECK_EQ(s->n_seq_decode, 1);
    CHECK_EQ(s->n_tok_decode, 5);
    CHECK_EQ(f.sch.plan().e[0].seq, a);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
}

/* THE ACCOUNTING IS EXACT AND IS CHARGED PER DRAFTED TOKEN, not per step: it is the number a
 * scraped acceptance rate divides by. The depth itself is a flag and this does not test it. */
TEST(the_draft_accounting_is_exact_and_clamps_what_was_not_proposed) {
    DraftController d;
    DraftController::Params p;
    p.max_depth = 8;
    CHECK_OK(d.configure(p));
    for (int i = 0; i < 100; ++i) d.observe(8, 0);
    CHECK_EQ(d.drafted(), 800);
    CHECK_EQ(d.accepted(), 0);
    CHECK_NEAR(d.acceptance(), 0.0, 1e-9);
    d.observe(2, 5);                 /* accepted past the draft is clamped, not counted */
    CHECK_EQ(d.accepted(), 2);
    CHECK_EQ(d.drafted(), 802);
    /* Position 0 was reached by every verify and taken by exactly the 1 accepted one (the 100
     * rejects took none); position 1 was reached only by that second verify, where it was taken.
     * Nothing past 1 was ever reached. */
    CHECK_EQ(d.reached(0), 101);
    CHECK_EQ(d.hits(0), 1);
    CHECK_EQ(d.reached(1), 1);
    CHECK_EQ(d.hits(1), 1);
    CHECK_EQ(d.reached(2), 0);
    CHECK_EQ(d.depth_seen(), 8);

    /* Depth 0 is speculation off, and it configures. */
    DraftController z;
    DraftController::Params zp;
    zp.max_depth = 0;
    CHECK_OK(z.configure(zp));
    CHECK_EQ(z.next_depth(), 0);
    CHECK(!z.bandit_active());
}

/* THE BANDIT IS OPT-IN AND DOES NOTHING WHEN IT IS NOT ASKED FOR. This is the whole default: the
 * reported depth of an unarmed controller is whatever the flags say, whatever evidence arrives. */
TEST(the_draft_depth_does_not_move_when_the_bandit_is_unarmed) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 3;
    CHECK_OK(c.configure(p));
    CHECK(!c.bandit_active());
    for (int t = 0; t < 200; ++t) {
        DraftController::BanditStep s;
        s.step_chosen = s.step_depth = s.out_depth = 3;
        s.rows = 1;
        s.emitted = 4;
        s.ns = 1000;
        c.observe_step(s);
        CHECK_EQ(c.next_depth(), 3);
    }
    CHECK_EQ(c.bandit_attempts(), 0);
    CHECK_EQ(c.bandit_credited(), 0);
}

/* ARMS ARE 1..min(window, the drafter's own depth, 8), and no arm is ever past the window. A
 * controller with one legal arm is legal, and so is a window of one. */
TEST(the_bandit_never_holds_an_arm_past_the_window_or_the_drafter) {
    {
        DraftController c;
        DraftController::Params p;
        p.max_depth = 1;
        p.bandit = true;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.bandit_arms(), 1);
        CHECK_EQ(c.next_depth(), 1);
    }
    {
        DraftController c;
        DraftController::Params p;
        p.max_depth = 3;
        p.drafter_depth = 2;
        p.bandit = true;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.bandit_arms(), 2);
        CHECK(c.next_depth() >= 1 && c.next_depth() <= 2);
    }
    {
        /* No declared drafter depth bounds nothing: the window alone decides. */
        DraftController c;
        DraftController::Params p;
        p.max_depth = 3;
        p.bandit = true;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.bandit_arms(), 3);
    }
    {
        /* Five is the deepest a linear arm can be: at six and above the unarmed path's two-stage
         * wide branch picks its depth from device state after the drafter has run, so the armed
         * path refuses that deployment outright (see the refusal cases below). */
        DraftController c;
        DraftController::Params p;
        p.max_depth = 5;
        p.drafter_depth = 8;      /* declared deeper than the window: the window still bounds */
        p.bandit = true;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.bandit_arms(), 5);
        CHECK(c.next_depth() <= 5);
    }
    /* Every refusal names a path whose depth this controller cannot own. */
    {
        DraftController c;
        DraftController::Params p;
        p.max_depth = 3;
        p.bandit = true;
        p.tree_state = true;
        CHECK_EQ(c.configure(p), RAD_E_UNSUPPORTED);
        p.tree_state = false;
        p.adaptive = true;
        CHECK_EQ(c.configure(p), RAD_E_UNSUPPORTED);
        p.adaptive = false;
        p.max_depth = 0;
        CHECK_EQ(c.configure(p), RAD_E_UNSUPPORTED);
        /* The two-stage wide branch: at six and above the unarmed path picks 8 from device state
         * after the drafter has run, so no step could be attributed to an arm. */
        p.max_depth = 6;
        CHECK_EQ(c.configure(p), RAD_E_UNSUPPORTED);
        p.max_depth = 3;
        p.drafter_depth = 0;
        CHECK_OK(c.configure(p));
    }
}

/* THE COLD SWEEP IS BOUNDED BY INTERVALS, NOT BY CREDITED SAMPLES. An interval that ran an arm but
 * cannot be scored -- a mixed step, a clamped window, an ineligible row -- still advances the
 * sweep, so a workload that never yields a clean measurement cannot leave the controller
 * exploring forever. */
TEST(the_cold_sweep_ends_on_intervals_and_never_needs_a_credited_sample) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 3;
    p.bandit = true;
    CHECK_OK(c.configure(p));
    CHECK_EQ(c.bandit_arm(), 1);

    /* AN ATTEMPT THAT CANNOT BE CREDITED: the interval ran the arm's verify and paid for its draft
     * passes, but a prefill chunk shared the step so its time is not the arm's. */
    auto mixed = [&] {
        DraftController::BanditStep s;
        s.step_epoch = c.bandit_epoch();
        s.step_chosen = c.bandit_arm();
        s.step_depth = c.bandit_arm();
        s.out_depth = c.bandit_arm();
        s.prefill_tok = 512;
        s.rows = 1;
        s.emitted = 1;
        s.ns = 1000;
        return s;
    };
    for (int i = 0; i < 3; ++i) c.observe_step(mixed());
    CHECK_EQ(c.bandit_attempts(), 3);
    CHECK_EQ(c.bandit_credited(), 0);
    CHECK_EQ(c.bandit_arm(), 2);          /* three intervals an arm, then the next */

    for (int i = 0; i < 3; ++i) c.observe_step(mixed());
    CHECK_EQ(c.bandit_arm(), 3);

    for (int i = 0; i < 3; ++i) c.observe_step(mixed());
    /* Swept, with nothing measured: the fallback is the SHALLOWEST arm, never an unmeasured deeper
     * one that would be chosen for having no evidence at all. */
    CHECK_EQ(c.bandit_arm(), 1);
    CHECK_EQ(c.bandit_credited(), 0);
    CHECK_EQ(c.bandit_attempts(), 9);
    CHECK_EQ(c.bandit_pulls(1), 0);
    CHECK_EQ(c.bandit_pulls(2), 0);
    CHECK_EQ(c.bandit_pulls(3), 0);
    for (int i = 0; i < 200; ++i) c.observe_step(mixed());
    CHECK_EQ(c.bandit_attempts(), 209);
    CHECK_EQ(c.bandit_arm(), 1);
}

/* AN INTERVAL THAT STRADDLES AN ARM CHANGE IS AN ATTEMPT AND NOT A MEASUREMENT. The engine
 * publishes the next step's drafts BEFORE this one commits, so the interval in which the policy
 * moves verifies the previous arm's publication and pays for the new arm's. It spent the step, so
 * it counts toward the sweep's bound; its cost is not one arm's, so it is never scored. */
TEST(a_transition_interval_counts_as_an_attempt_and_is_never_credited) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 2;
    p.bandit = true;
    CHECK_OK(c.configure(p));
    CHECK_EQ(c.bandit_arm(), 1);          /* the sweep starts at the cheapest arm */

    DraftController::BanditStep s;
    s.step_epoch = c.bandit_epoch();
    s.step_chosen = 2;                   /* the step was planned under arm 2 ... */
    s.step_depth = 1;                    /* ... and verified arm 1's publication */
    s.out_depth = 2;                     /* while its own draft passes ran arm 2's window */
    s.rows = 1;
    s.emitted = 2;
    s.ns = 1000;
    CHECK(c.is_attempt(s, c.bandit_epoch()));
    CHECK(!c.is_credit(s, c.bandit_epoch()));
    /* A PREVIOUS epoch is not even an attempt: the step straddled the change itself. */
    CHECK(!c.is_attempt(s, c.bandit_epoch() - 1));
}

/* AN INTERVAL THAT CARRIED NO PROPOSAL AT ALL SPENT NO ARM'S COST, so it is not an attempt: the
 * workload that never drafts must leave the sweep where it is -- at the shallowest arm -- rather
 * than walk every arm and report one it never ran. */
TEST(an_interval_that_proposed_nothing_is_not_an_attempt) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 3;
    p.bandit = true;
    CHECK_OK(c.configure(p));
    DraftController::BanditStep s;
    s.step_epoch = c.bandit_epoch();
    s.step_chosen = c.bandit_arm();
    s.out_depth = 0;       /* may_speculate false: no proposal published */
    s.rows = 1;
    s.emitted = 1;         /* it decoded, one token a step */
    s.ns = 1000;
    for (int i = 0; i < 100; ++i) c.observe_step(s);
    CHECK_EQ(c.bandit_attempts(), 0);
    CHECK_EQ(c.bandit_credited(), 0);
    CHECK_EQ(c.bandit_arm(), 1);
}

/* NO CREDIT FOR AN INTERVAL THAT STRADDLED AN ARM CHANGE, WAS CLIPPED, OR SHARED A PREFILL. Each of
 * these is an interval whose cost is not about one arm, and crediting it would teach the policy
 * the wrong depth. */
TEST(an_interval_is_credited_only_when_both_its_halves_ran_the_chosen_arm) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 2;
    p.bandit = true;
    CHECK_OK(c.configure(p));

    auto clean = [&] {
        DraftController::BanditStep s;
        s.step_epoch = c.bandit_epoch();
        s.step_chosen = c.bandit_arm();
        s.step_depth = c.bandit_arm();
        s.out_depth = c.bandit_arm();
        s.rows = 1;
        s.emitted = 2;
        s.ns = 1000;
        return s;
    };

    {   /* a previous epoch: the step was planned before the policy moved */
        DraftController::BanditStep s = clean();
        s.step_epoch = c.bandit_epoch() - 1;
        CHECK(!c.is_attempt(s, c.bandit_epoch()));
        CHECK(!c.is_credit(s, c.bandit_epoch()));
    }
    {   /* the drafts this interval paid for were another arm's */
        DraftController::BanditStep s = clean();
        s.out_depth = c.bandit_arm() + 1;
        CHECK(c.is_attempt(s, c.bandit_epoch()));
        CHECK(!c.is_credit(s, c.bandit_epoch()));
    }
    {   /* no drafts published at all: the arm's plan ran, its tail did not */
        DraftController::BanditStep s = clean();
        s.out_depth = -1;
        CHECK(c.is_attempt(s, c.bandit_epoch()));
        CHECK(!c.is_credit(s, c.bandit_epoch()));
    }
    {   /* every row clamped below the arm: the context bound or the request shortened it */
        DraftController::BanditStep s = clean();
        s.step_depth = c.bandit_arm() - 1;
        CHECK(c.is_attempt(s, c.bandit_epoch()));
        CHECK(!c.is_credit(s, c.bandit_epoch()));
    }
    {   /* a prefill chunk shared the step, so its time is not the arm's */
        DraftController::BanditStep s = clean();
        s.prefill_tok = 512;
        CHECK(c.is_attempt(s, c.bandit_epoch()));
        CHECK(!c.is_credit(s, c.bandit_epoch()));
    }
    {   /* and the clean one is the control: without it the negatives prove nothing */
        DraftController::BanditStep s = clean();
        CHECK(c.is_credit(s, c.bandit_epoch()));
        c.observe_step(s);
        CHECK_EQ(c.bandit_credited(), 1);
        CHECK_EQ(c.bandit_tokens(), 2);
        CHECK_EQ(c.bandit_ns(), 1000);
    }
}

/* THE ARM WITH THE BEST MEASURED REWARD HOLDS. The reward is emitted decode tokens per row over
 * the real interval, so an arm whose steps emit fewer tokens for the same time loses. */
TEST(the_bandit_holds_the_arm_with_the_best_measured_reward) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 3;
    p.bandit = true;
    CHECK_OK(c.configure(p));

    /* Sweep, then keep feeding each arm what it is worth: arm 2 emits twice what arm 1 does in the
     * same interval, arm 3 half of arm 1. */
    for (int t = 0; t < 60; ++t) {
        const int arm = c.bandit_arm();
        DraftController::BanditStep s;
        s.step_epoch = c.bandit_epoch();
        s.step_chosen = arm;
        s.step_depth = arm;
        s.out_depth = arm;
        s.rows = 1;
        s.emitted = arm == 2 ? 4 : arm == 1 ? 2 : 1;
        s.ns = 1000;
        c.observe_step(s);
    }
    CHECK_EQ(c.bandit_arm(), 2);
    CHECK(c.bandit_credited() > 0);
    CHECK(c.bandit_tokens() > 0);
    CHECK(c.bandit_ns() > 0);
    CHECK(c.bandit_reward(2) > c.bandit_reward(1));
    CHECK(c.bandit_reward(1) > c.bandit_reward(3));
    /* A depth outside the arms has no evidence and cannot be reported as if it had. */
    CHECK_EQ(c.bandit_pulls(4), 0);
    CHECK_NEAR(c.bandit_reward(4), 0.0, 1e-12);
}

/* WHAT A DEPLOYMENT LEARNED SURVIVES ADMISSION, IDLE AND PREFILL; ONLY configure() CLEARS IT.
 * Resetting on arrival would restart the sweep on every request and leave the controller
 * permanently exploring -- which is exactly the behaviour that made the old per-request policy
 * unusable. */
TEST(the_bandits_learning_survives_resets_and_only_a_new_deployment_clears_it) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 2;
    p.bandit = true;
    CHECK_OK(c.configure(p));

    for (int t = 0; t < 20; ++t) {
        DraftController::BanditStep s;
        s.step_epoch = c.bandit_epoch();
        s.step_chosen = s.step_depth = s.out_depth = c.bandit_arm();
        s.rows = 1;
        s.emitted = 2;
        s.ns = 1000;
        c.observe_step(s);
    }
    const int64_t attempts = c.bandit_attempts();
    const int64_t credited = c.bandit_credited();
    const int arm = c.bandit_arm();
    CHECK(attempts > 0);
    CHECK(credited > 0);

    c.reset_adaptive();                     /* new request / prefill completion / idle */
    c.observe(4, 0);                        /* the acceptance accounting, which is not a decision */
    CHECK_EQ(c.bandit_attempts(), attempts);
    CHECK_EQ(c.bandit_credited(), credited);
    CHECK_EQ(c.bandit_arm(), arm);
    CHECK(c.bandit_reward(arm) > 0.0);

    CHECK_OK(c.configure(p));               /* a new deployment: a new controller */
    CHECK_EQ(c.bandit_attempts(), 0);
    CHECK_EQ(c.bandit_credited(), 0);
    CHECK_EQ(c.bandit_arm(), 1);
}

/* ================================================================== the armed controller, in the scheduler */
/* THE ARM IS WHAT THE STEP EXECUTES, and the accounting follows the whole interval -- verify plus
 * the draft passes it paid for -- not `plan.e[].n_spec` alone. This drives the real step/commit/
 * set_draft/step_finished order the engine uses: the proposal for the next step is published AFTER
 * the commit, so an interval's own verify was sized by the previous interval's publication. */
TEST(the_armed_bandit_sizes_the_step_and_is_credited_by_its_whole_interval) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 3;
    f.prog.drafter.kind = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 3;
    /* The arming test must not inherit the bisect knob from whoever launched ctest; it is put back
     * afterwards so no later case sees a changed environment. */
    const char* saved_no_draft = std::getenv("RADIANCE_NO_DRAFT");
    const std::string saved = saved_no_draft ? saved_no_draft : "";
    unsetenv("RADIANCE_NO_DRAFT");
    setenv("RADIANCE_ADAPTIVE_BANDIT", "1", 1);
    const int rc = f.start();
    unsetenv("RADIANCE_ADAPTIVE_BANDIT");
    if (saved_no_draft) setenv("RADIANCE_NO_DRAFT", saved.c_str(), 1);
    REQUIRE_OK(rc);
    REQUIRE(f.sch.draft().bandit_active());
    CHECK_EQ(f.sch.draft().bandit_arms(), 3);

    const uint64_t a = f.submit(8, 400);
    REQUIRE(a != 0);
    int32_t draft[3] = { 21, 22, 23 };

    /* THE PREFILL INTERVAL: it publishes the first proposal (clamped to the arm, which is 1 while
     * the cold sweep runs) and is not an attempt, because it committed no decode token. */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    CHECK_EQ((int)f.sch.find(a)->n_draft, 1);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().bandit_attempts, 0);

    /* ONE CLEAN DECODE INTERVAL: the step verifies the 1 draft the arm published, and the interval
     * publishes another 1 for the next one. Both halves are arm 1, so it is credited. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 1);
    f.commit_plain(12);
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    f.sch.step_finished();
    {
        const SchedMetrics m = f.sch.metrics();
        CHECK_EQ(m.bandit_attempts, 1);
        CHECK_EQ(m.bandit_credited, 1);
        CHECK_EQ(m.draft_arm, 1);
        CHECK_EQ(m.draft_arms, 3);
        CHECK(m.bandit_tokens > 0);
        CHECK(m.bandit_ns > 0);
    }

    /* THE SWEEP: three intervals an arm. The policy moves to arm 2 at the third interval's
     * step_finished, which runs AFTER that interval's set_draft -- so the proposal already
     * published is still arm 1's, and the step that verifies it is a TRANSITION: it ran (counted
     * as an attempt) but its depth is not arm 2's (not credited). This one-interval lag is
     * inherent to deciding the arm from a whole interval's cost and is why the check below is two
     * steps deep, not one. */
    for (int i = 0; i < 2; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(13);
        CHECK_OK(f.sch.set_draft(a, draft, 3));
        f.sch.step_finished();
    }
    CHECK_EQ(f.sch.metrics().draft_arm, 2);
    {
        const int64_t credited = f.sch.metrics().bandit_credited;
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().e[0].n_spec, 1);     /* the previous arm's publication */
        f.commit_plain(14);
        CHECK_OK(f.sch.set_draft(a, draft, 3));    /* now arm 2 publishes two */
        CHECK_EQ((int)f.sch.find(a)->n_draft, 2);
        f.sch.step_finished();
        CHECK_EQ(f.sch.metrics().bandit_credited, credited);  /* a transition, not a measurement */
    }
    /* AND THE STEP AFTER IT VERIFIES THE ARM, not the proposal's length and not a global ceiling. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 2);

    /* AN INTERVAL WHOSE PUBLISHED PROPOSAL WAS CLAMPED CREDITS NOTHING. The step verifies the arm
     * (two), but max_tokens leaves room for one draft, so the draft pass this interval pays for is
     * a shorter window: its cost is two arms' worth of work and one arm's worth of proposal, which
     * is not a measurement of either. */
    f.commit_plain(15);
    {
        Request* r = f.sch.find(a);
        REQUIRE(r != nullptr);
        r->max_tokens = (int32_t)r->output.size() + 2;   /* rem - 1 == 1 */
        const int64_t credited = f.sch.metrics().bandit_credited;
        CHECK_OK(f.sch.set_draft(a, draft, 3));
        CHECK_EQ((int)f.sch.find(a)->n_draft, 1);
        f.sch.step_finished();
        const SchedMetrics m = f.sch.metrics();
        CHECK_EQ(m.draft_arm, 2);
        CHECK_EQ(m.bandit_credited, credited);
        CHECK(m.bandit_attempts >= 4);
    }
}

/* THE UNARMED PATH IS UNTOUCHED, INCLUDING THE TWO-STAGE WIDE BRANCH. A window of six or more
 * deliberately returns 8 on the legacy path -- the second stage is gated on device state after the
 * drafter has run -- and that is the behaviour the armed controller must neither inherit nor
 * disturb. */
TEST(the_unarmed_depth_policy_keeps_its_two_stage_wide_branch) {
    {
        DraftController c;
        DraftController::Params p;
        p.max_depth = 8;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.next_depth(1), 8);      /* the wide branch, whatever the concurrency */
        CHECK_EQ(c.next_depth(5), 8);
        CHECK(!c.bandit_active());
    }
    {
        /* Below six the window itself is the answer, for both adaptive settings. */
        DraftController c;
        DraftController::Params p;
        p.max_depth = 3;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.next_depth(1), 3);
        p.adaptive = true;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.next_depth(1), 3);
        CHECK_EQ(c.next_depth(4), 3);
    }
    {
        /* A tree verifier keeps its own node budget: the whole window for one sequence, four
         * rows with concurrency. */
        DraftController c;
        DraftController::Params p;
        p.max_depth = 7;
        p.tree_state = true;
        CHECK_OK(c.configure(p));
        CHECK_EQ(c.next_depth(1), 7);      /* tree_draft_depth(7, single sequence) */
        CHECK_EQ(c.next_depth(2), 4);
    }
}

/* THE SWITCH IS OPT-IN AT THE SCHEDULER, and a program whose depth is not this deployment's to
 * choose is refused rather than silently served at a constant depth with the flag apparently on. */
TEST(the_bandit_is_off_by_default_and_refused_where_a_linear_arm_does_not_describe_the_step) {
    {   /* default: untouched, and the controller says so */
        Fixture f;
        f.build();
        f.cfg.n_spec = 3;
        f.prog.drafter.kind = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 3;
        REQUIRE_OK(f.start());
        CHECK(!f.sch.draft().bandit_active());
    }
    {   /* a tree verifier names a depth from device state; the arm would not describe the step */
        Fixture f;
        f.build();
        f.cfg.n_spec = 3;
        f.prog.drafter.kind = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 3;
        OpInfo commit;
        commit.op = "gdn_tree_commit";
        f.prog.ops.push_back(commit);
        setenv("RADIANCE_ADAPTIVE_BANDIT", "1", 1);
        const int rc = f.start();
        unsetenv("RADIANCE_ADAPTIVE_BANDIT");
        CHECK_EQ(rc, RAD_E_UNSUPPORTED);
    }
    {   /* and the value is validated, so a typo is not read as "on" */
        Fixture f;
        f.build();
        f.cfg.n_spec = 3;
        f.prog.drafter.kind = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 3;
        setenv("RADIANCE_ADAPTIVE_BANDIT", "yes", 1);
        const int rc = f.start();
        unsetenv("RADIANCE_ADAPTIVE_BANDIT");
        CHECK_EQ(rc, RAD_E_INVAL);
    }
    {   /* a window of six or more: the wide branch owns the depth, so the flag is refused */
        Fixture f;
        f.build();
        f.cfg.n_spec = 7;
        f.prog.drafter.kind = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 7;
        setenv("RADIANCE_ADAPTIVE_BANDIT", "1", 1);
        const int rc = f.start();
        unsetenv("RADIANCE_ADAPTIVE_BANDIT");
        CHECK_EQ(rc, RAD_E_UNSUPPORTED);
    }
}

/* ================================================================== the step path */
TEST(the_step_path_does_not_allocate) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    for (int i = 0; i < 6; ++i) f.submit(64, 4096);

    /* Warm up: the first steps grow the request's block vector and its output, both of which are
     * sized once and never again. What must be zero is the STEADY-STATE step. */
    for (int i = 0; i < 12; ++i) {
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
    }

    g_allocs = 0;
    g_counting = true;
    for (int i = 0; i < 40; ++i) {
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& p = f.sch.plan();
        Scheduler::StepResult r;
        static int32_t acc[64], tok[64];
        static uint8_t eos[64];
        for (size_t k = 0; k < p.e.size(); ++k) { acc[k] = 0; tok[k] = 5; eos[k] = 0; }
        r.n_seq = (int)p.e.size();
        r.n_accepted = acc; r.token = tok; r.eos = eos;
        f.sch.commit(r);
    }
    g_counting = false;

    /* Not "few". None: an allocator call here is a declare-phase bug (IMPLEMENTATION.md), and
     * this is the assertion that finds it the day it is written. */
    CHECK_EQ(g_allocs, 0L);
}

/* THE CARD'S BLOCK TABLES ARE STAGED AS THE ENTRIES THAT CHANGED, so they are checked whole after
 * every build: every row of a paged group reads the manager's table and then -1 to the pitch,
 * and a state group's row reads the slot. The run moves the tables every way a real one does --
 * appends at block crossings, rejected drafts rolled back, requests ending and their rows taken
 * by the next, a late arrival's prefill -- and a row staged from a stale picture of the card
 * reads a block the sequence no longer holds, or a -1 where it holds one. */
TEST(the_cards_block_tables_match_the_manager_after_every_build) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    std::vector<uint64_t> ids;
    ids.push_back(f.submit(40, 9));
    ids.push_back(f.submit(70, 200));
    ids.push_back(f.submit(17, 90));
    uint32_t rng = 12345;
    int checked = 0;
    for (int step = 0; step < 80; ++step) {
        if (step == 6) ids.push_back(f.submit(33, 120));
        if (step == 20) ids.push_back(f.submit(90, 60));
        for (uint64_t id : ids) {
            int32_t draft[4] = { 31, 32, 33, 34 };
            (void)f.sch.set_draft(id, draft, 4);
        }
        const RadBatch* b = f.sch.step();
        if (!b) break;
        const StepPlan& p = f.sch.plan();
        for (int gi = 0; gi < b->n_kv_groups; ++gi) {
            const RadKVGroupBatch& g = b->kv[gi];
            for (int64_t i = 0; i < b->n_seq; ++i) {
                const uint64_t seq = p.e[(size_t)i].seq;
                const int32_t* row = g.block_table + i * g.block_table_pitch;
                if (!f.kv.geom.g[(size_t)gi].paged) {
                    CHECK_EQ(row[0], f.kv.state_slot(seq, gi));
                    continue;
                }
                const std::vector<int32_t>& bt = f.kv.block_table(seq, gi);
                for (int64_t j = 0; j < g.block_table_pitch; ++j)
                    CHECK_EQ(row[j], j < (int64_t)bt.size() ? bt[(size_t)j] : -1);
                ++checked;
            }
        }
        Scheduler::StepResult r;
        static int32_t acc[64], tok[64];
        static uint8_t eos[64];
        for (size_t k = 0; k < p.e.size(); ++k) {
            rng = rng * 1664525u + 1013904223u;
            acc[k] = p.e[k].n_spec > 0 ? (int32_t)((rng >> 16) % (uint32_t)(p.e[k].n_spec + 1)) : 0;
            tok[k] = 5;
            eos[k] = 0;
        }
        r.n_seq = (int)p.e.size();
        r.n_accepted = acc; r.token = tok; r.eos = eos;
        CHECK_OK(f.sch.commit(r));
    }
    /* The run reached steady decode with rows to compare, not an empty loop. */
    CHECK(checked > 100);
}

/* A SEQUENCE THAT LEAVES A ROW AND COMES BACK TO IT. Its change marks are taken wherever it is
 * written, so the copy it left behind stops being current the moment it is staged elsewhere: a
 * builder that still trusted that row would restage only what changed after the marks were taken,
 * and keep the entries a rollback replaced before them. The scheduler's own passes change the
 * rows a sequence sits in (a draft round without the sequences that did not draft), so this drives
 * the builder directly with the row orders written out. */
TEST(a_sequence_back_in_a_row_it_left_is_restaged_whole) {
    Fixture f;
    f.build();
    ChunkGeometry geo;
    CHECK_OK(chunk_geometry_resolve(f.prog, f.cfg, &geo));
    KVGeom kvg = KVGeom::of(f.prog.kv_groups);
    for (size_t i = 0; i < kvg.g.size(); ++i) {
        const int64_t w = f.kv.state_index_width((int32_t)i);
        if (w > 0) kvg.g[i].sidx_width = w;
    }
    BatchBuilder bb;
    if (!f.stream) rad_stream_create(&f.stream, 0);
    std::vector<BatchBuilder::RankIO> rio;
    rio.push_back({ 0, f.stream, &f.prog });
    CHECK_OK(bb.init(f.prog, f.cfg, geo, kvg, rio));

    for (uint64_t id = 1; id <= 3; ++id) {
        CHECK_OK(f.kv.add_sequence(id));
        CHECK_OK(f.kv.ensure(id, 80));
    }
    StepPlan p;
    auto build = [&](std::initializer_list<uint64_t> rows) {
        p.reset();
        int32_t card = 0;
        for (uint64_t id : rows) {
            StepEntry e;
            e.seq = id;
            e.card = card++;
            e.n_tokens = 1;
            e.ctx_len = (int32_t)f.kv.seqs[id].n - 1;
            p.e.push_back(e);
            p.n_tok += 1;
        }
        const RadBatch* b = bb.build(p, f.kv, 0);
        CHECK(b != nullptr);
        if (!b) return;
        const RadKVGroupBatch& g = b->kv[0];
        for (int64_t i = 0; i < b->n_seq; ++i) {
            const std::vector<int32_t>& bt = f.kv.block_table(p.e[(size_t)i].seq, 0);
            const int32_t* row = g.block_table + i * g.block_table_pitch;
            for (int64_t j = 0; j < g.block_table_pitch; ++j)
                CHECK_EQ(row[j], j < (int64_t)bt.size() ? bt[(size_t)j] : -1);
        }
    };

    build({ 1, 2, 3 });
    /* Sequence 3's last three blocks are handed to another sequence and replaced: its table keeps
     * its length and changes below it. */
    CHECK_OK(f.kv.rollback(3, 32));
    CHECK_OK(f.kv.add_sequence(4));
    CHECK_OK(f.kv.ensure(4, 48));
    CHECK_OK(f.kv.ensure(3, 80));
    build({ 1, 3 });      /* 3 staged in row 1, its marks taken there */
    build({ 1, 2, 3 });   /* and back in row 2, which still holds its old blocks */
    bb.fini();
}

/* TWO CONV GROUPS, AND EACH ONE'S DERIVATION READS ITS OWN WINDOW.
 *
 * Held as ONE slot array and ONE window number, both would be filled for whichever RAD_KV_CONV
 * group was declared LAST, and a second group's row would stay -1.
 *
 * WHAT THAT COSTS IS NARROWER THAN IT LOOKS: these two feed `DeriveInput` and nothing else, and
 * the only derivation that reads them -- `conv_state_index` -- is called by no architecture plugin
 * at present. A conv kernel takes its slot from RadKVGroupBatch::state_index, which is per group.
 * So the failure is wrong values in a buffer nobody reads, not a convolution reading no history.
 *
 * The two groups here have DIFFERENT WIDTHS on purpose: `conv_state_index` returns
 * `slot * window + cursor % window`, so a derivation served the wrong group's depth returns a
 * plausible number and the only way to tell is to make the two depths disagree.
 */
TEST(each_conv_group_derives_against_its_own_window) {
    auto two_groups = [](Fixture& f) {
        KVGroupInfo conv2;
        conv2.name = "conv_deep";
        conv2.decl.kind = RAD_KV_CONV;
        conv2.decl.conv_width = 10;
        f.prog.kv_groups.push_back(conv2);
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    };

    /* ---- an UNNAMED group is refused, not guessed at ----------------------------------------- */
    {
        Fixture f;
        f.build();                       /* declares `conv_idx` with no group -- fine at one */
        two_groups(f);
        /* The refusal logs at Error and that is the point of it; the line is expected here. */
        CHECK_EQ(f.start(), RAD_E_INVAL);
    }

    /* ---- named, and each reads its own ------------------------------------------------------- */
    Fixture f;
    f.build();
    two_groups(f);
    /* Group 2 is the width-4 conv build() declared; group 3 is the deep one just added. */
    for (BufferInfo& b : f.prog.buffers) if (b.name == "conv_idx") b.decl.kv = 2;
    f.add_derived("conv_idx_deep", "conv_state_index", 256, 3);
    CHECK_OK(f.start());

    CHECK(f.submit(4) != 0);
    CHECK(f.sch.step() != nullptr);

    const int32_t* a = f.derived("conv_idx");
    const int32_t* c = f.derived("conv_idx_deep");
    CHECK(a != nullptr);
    CHECK(c != nullptr);
    /* NEITHER IS -1. Under one shared slot array the group that is not the last-declared one gets
     * -1 for every sequence, which is what this pins -- so that the first architecture to call
     * this derivation finds it working. */
    if (a && c) {
        CHECK(a[0] >= 0);
        CHECK(c[0] >= 0);
    }
}


/* ================================================================== the derivation registry
 *
 * A DERIVATION NAMED IN A SCHEMA AND NOT IN THIS TABLE IS A DECLARE-TIME REFUSAL BY NAME, with
 * the known list printed beside it -- the same treatment a misspelled op gets (spec §2.3). The
 * alternative is a buffer the step loop leaves untouched, which a kernel then reads as whatever
 * the arena last held: on a decode step that is another sequence's index row, and the output is
 * fluent and wrong.
 *
 * So the registry itself has a property worth pinning: every row is reachable by its own name,
 * every row has a function, and a name that is not in it resolves to nothing rather than to row
 * zero.
 */
TEST(every_derivation_is_reachable_by_name_and_an_unknown_one_is_not) {
    const int n = derive_count();
    CHECK(n > 0);

    for (int i = 0; i < n; ++i) {
        const DeriveRow* r = derive_row(i);
        CHECK(r != nullptr);
        if (!r) continue;
        CHECK(r->name && *r->name);
        CHECK(r->fn != nullptr);          /* a row with no function is a name that resolves and
                                           * then writes nothing, which is the failure above */
        CHECK(r->doc && *r->doc);         /* it is printed in the refusal's known list */
        CHECK_EQ(derive_index(r->name), i);
    }

    /* NAMES ARE UNIQUE, or derive_index answers with whichever row came first and the second one
       can never be selected. */
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            CHECK(std::strcmp(derive_row(i)->name, derive_row(j)->name) != 0);

    /* -1 for a name the core does not know, rather than 0 -- which is a valid index. */
    CHECK_EQ(derive_index("not_a_derivation"), -1);
    CHECK_EQ(derive_index(""), -1);
    CHECK_EQ(derive_index(nullptr), -1);
    /* A near miss on a real name is still a miss: the point of the table is that a typo is
       reported rather than approximated. */
    CHECK_EQ(derive_index("seq_start_idx"), -1);

    /* And an index outside the table is null rather than a read past it. */
    CHECK(derive_row(-1) == nullptr);
    CHECK(derive_row(n) == nullptr);
    CHECK(derive_row(n + 1000) == nullptr);
}

/* ================================================================== request lifetime */

/* THE SAMPLER SIZES ITS ROWS FROM Request::n_draft, and a drafter can set it for a step that
 * preemption then takes away. The request comes back as a prefill with one logits row, and a
 * stale window of four would have the sampler stage five -- every request after it in the step
 * then samples from its neighbour's row. */
TEST(a_preempted_request_comes_back_with_no_draft_window) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64, /*priority=*/0);
    uint64_t b = f.submit(160, 64, /*priority=*/5);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_EQ(f.sch.find(a)->n_draft, 4);

    /* Both need a new block at 160 and the pool has none: A, the lower priority, is taken. */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    CHECK_EQ(f.sch.find(a)->n_draft, 0);
    CHECK(f.sch.find(a)->draft == nullptr);
    f.commit_plain();

    /* And when it is admitted again, the window it carries is the one its entry has. */
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 0);
    CHECK_EQ(f.sch.find(a)->n_draft, 0);
}

/* EACH TERMINAL ID IS HANDED OVER ONCE. take_completed swaps the list out under the lock. A
 * reference into scheduler storage that nothing clears would have every finished request walked
 * again on every step for the life of the process -- and walked without the lock while an HTTP
 * thread's cancel appends to it. */
TEST(terminal_requests_are_taken_once) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t a = f.submit(8, /*max_tokens=*/1);
    uint64_t b = f.submit(8, /*max_tokens=*/64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();                               /* A reaches its one token */
    CHECK_OK(f.sch.cancel(b));

    std::vector<uint64_t> done;
    f.sch.take_completed(&done);
    CHECK_EQ((int)done.size(), 2);
    CHECK(done[0] == a);
    CHECK(done[1] == b);
    f.sch.take_completed(&done);
    CHECK_EQ((int)done.size(), 0);
}

/* EVERY ROW NAMES ONE OF THE CARD'S STEP SLOTS. A slot is taken on arrival, so requests queued
 * behind the admission limit are numbered past max_seqs, and so are requests arriving while
 * finished ones are still held for their reap. The advance indexes max_seqs StepSlots by the
 * row's card index; a slot number there reads and writes past the end of them. */
TEST(every_row_names_one_of_the_cards_step_slots) {
    Fixture f;
    f.build();
    f.cfg.max_seqs = 2;
    CHECK_OK(f.start());

    for (int i = 0; i < 6; ++i) CHECK(f.submit(8, /*max_tokens=*/1) != 0);
    for (int round = 0; round < 3; ++round) {
        CHECK(f.sch.step() != nullptr);
        const StepPlan& p = f.sch.plan();
        CHECK_EQ((int)p.e.size(), 2);
        CHECK(p.e[0].card >= 0 && p.e[0].card < 2);
        CHECK(p.e[1].card >= 0 && p.e[1].card < 2);
        CHECK(p.e[0].card != p.e[1].card);
        /* Nothing was reaped, so the later rounds' slots did run past max_seqs. */
        if (round > 0) CHECK(p.e[0].slot >= 2 && p.e[1].slot >= 2);
        f.commit_plain();                           /* each reaches its one token */
    }
    CHECK_EQ(f.sch.metrics().queue_depth, 0);
}

/* THE CONTEXT PASS IS A MIRROR OF THE TRUNK STEP, ROW FOR ROW. The drafter reads the trunk's
 * hidden states positionally, so a sequence the commit finished still needs its rows -- left out,
 * every sequence after it drafts from its neighbour's hidden states. It stores nothing: its KV is
 * gone, so its slots are padding. */
TEST(a_block_drafter_context_pass_keeps_the_rows_of_a_finished_sequence) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, /*max_tokens=*/1);
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    f.commit_plain();
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(/*block=*/8, /*depth=*/7, /*window=*/0, /*mask=*/99, rows, 4), 1);
    CHECK(rows[0].seq == b);

    const RadBatch* cb = f.sch.block_context(1);
    REQUIRE(cb != nullptr);
    CHECK_EQ(cb->n_seq, 2);
    CHECK_EQ(cb->n_tok, 320);
    CHECK_EQ(cb->cu_seqlens[1], 160);
    CHECK_EQ(cb->kv[0].slot_mapping[0], -1);        /* A's rows store nothing */
    CHECK(cb->kv[0].slot_mapping[160] >= 0);        /* B's rows are B's */

    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    CHECK_EQ(qb->n_seq, 1);
    f.sch.block_end();
}

TEST(multistage_block_drafter_staging_geometry) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 12;
    f.cfg.adaptive_spec = true;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    Scheduler::BlockRow rows[4];
    // Multi-stage block drafter: block=7, depth=12 (2 * 6 = 12)
    CHECK_EQ(f.sch.block_begin(7, 12, 0, 99, rows, 4), 1);
    CHECK_EQ(rows[0].seq, a);

    const RadBatch* cb = f.sch.block_context(1);
    REQUIRE(cb != nullptr);
    CHECK_EQ(cb->n_seq, 1);

    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    CHECK_EQ(qb->n_seq, 1);
    CHECK_EQ(qb->n_tok, 7);
    CHECK_EQ(qb->draft_pass, 1);
    CHECK_EQ(qb->n_spec, 4);          /* scheduled proposal depth, not verified query rows */
    CHECK_EQ(f.sch.block_plan().e[0].n_spec, 0);
    CHECK_EQ(f.sch.block_plan().e[0].hidden_n, 6);
    CHECK_EQ(f.sch.block_plan().e[0].hidden_row, 1);
    CHECK_EQ(f.sch.block_plan().e[0].n_tokens, 7);
    CHECK_EQ(f.sch.block_plan().e[0].n_draft, 6);
    f.sch.block_end();

    // Invalid multi-stage geometry (non-divisible depth)
    CHECK_EQ(f.sch.block_begin(7, 13, 0, 99, rows, 4), 0);
    // Invalid block < 2
    CHECK_EQ(f.sch.block_begin(1, 12, 0, 99, rows, 4), 0);
    // Invalid geometry with depth < block: block=10, depth=3
    CHECK_EQ(f.sch.block_begin(10, 3, 0, 99, rows, 4), 0);
    // Single-stage block == depth + 1 (block=8, depth=7)
    CHECK_EQ(f.sch.block_begin(8, 7, 0, 99, rows, 4), 1);
    f.sch.block_end();
    // Single-stage block == depth (block=7, depth=7)
    CHECK_EQ(f.sch.block_begin(7, 7, 0, 99, rows, 4), 1);
    f.sch.block_end();
}

/* THE QUERY PASS NAMES, A DRAFTING SEQUENCE, THE TRUNK ROW ITS ANCHOR WAS SAMPLED AT: the verify
 * step's out_ids row of its last accepted position, first_row + n_accepted. A drafter that reads
 * the trunk's tapped hidden states gathers through it (RadBatch::draft_out_ids), so the pass's
 * operands are the same pointers every step and the recording replays. Reading row `s` instead
 * takes the hidden state of whatever position happens to sit there. */
TEST(a_block_query_pass_names_the_trunk_row_of_each_anchor) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 64);
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    const RadBatch* v = f.sch.step();
    REQUIRE(v != nullptr);
    CHECK_EQ(v->n_out, 10);                          /* 1 + 4 rows a sequence */
    REQUIRE((int)f.sch.plan().e.size() == 2);
    const bool a_first = f.sch.plan().e[0].seq == a;

    int32_t acc[2] = { 2, 1 };                       /* drafts accepted, in plan order */
    int32_t tok[2] = { 99, 98 };
    uint8_t eos[2] = { 0, 0 };
    Scheduler::StepResult r;
    r.n_seq = 2; r.n_accepted = acc; r.token = tok; r.eos = eos;
    CHECK_OK(f.sch.commit(r));

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(/*block=*/5, /*depth=*/4, /*window=*/0, /*mask=*/7, rows, 4), 2);
    CHECK_EQ(rows[0].tap_row, 2);
    CHECK_EQ(rows[1].tap_row, 5 + 1);
    CHECK(rows[0].seq == (a_first ? a : b));

    CHECK(f.sch.block_context(1) != nullptr);
    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    /* The batch's staging crosses to the slab ASYNC -- a worker thread even on the host backend --
     * so the device-side array holds the staged values only once the compute stream has drained. */
    CHECK_EQ(rad_stream_sync(f.stream), RAD_OK);

    CHECK_EQ(qb->n_seq, 2);
    CHECK_EQ(qb->n_draft_out, 2);
    CHECK_EQ(f.sch.block_plan().e[0].src_row, 2);
    CHECK_EQ(f.sch.block_plan().e[1].src_row, 6);
    REQUIRE(qb->draft_out_ids != nullptr);
    CHECK_EQ(qb->draft_out_ids[0], 2);
    CHECK_EQ(qb->draft_out_ids[1], 6);
    f.sch.block_end();
}

/* A CANCEL BETWEEN TWO DRAFTER PASSES. The passes are built one at a time with the lock released
 * between them, so the cancel lands after the context pass was built and before the query pass
 * is. Freeing then would build the query pass over a sequence with no blocks: a block-table row
 * of -1 under a non-zero length, which attention reads as block -1. */
TEST(a_cancel_between_drafter_passes_keeps_the_blocks_until_the_passes_are_built) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 7;
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(8, 7, 0, 99, rows, 4), 1);
    CHECK(f.sch.block_context(1) != nullptr);

    CHECK_OK(f.sch.cancel(a));
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK(f.kv.has_sequence(a));                     /* not freed under the passes */

    const RadBatch* qb = f.sch.block_query(1);
    REQUIRE(qb != nullptr);
    CHECK(qb->kv[0].block_table[0] >= 0);            /* a real block, not -1 */
    f.sch.block_end();

    /* The drafter is done with it, so it goes -- at the next commit, or here for an engine that
     * has gone idle and runs none. */
    CHECK(f.sch.has_deferred());
    f.sch.release_deferred();
    CHECK(!f.kv.has_sequence(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK(!f.sch.has_deferred());
    CHECK_OK(f.sch.reap(a));
    CHECK(f.sch.find(a) == nullptr);
}

/* A HIT THAT RESUMES FROM A SNAPSHOT IS NOT CARRIED ACROSS A FAILED ADMISSION. The slot is named
 * by id only; between two attempts it can be retired and handed to another sequence's checkpoint,
 * and an attempt that trusted the old id would restore that sequence's recurrent state. The next
 * attempt has to look the prompt up again -- and the accounting counts the admission once. */
TEST(a_failed_admission_gives_back_its_checkpoint_hit) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t r = f.submit(160, 64);  /* 10 blocks, and a miss */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    CHECK_EQ(f.pc.lookups, 1);

    f.pc.seed(256);                  /* 16 blocks the cache holds */
    f.pc.lin = 256;
    f.pc.src = 3;

    /* A adopts the 16 cached blocks and needs 16 more for its first chunk; 13 are left. */
    uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_EQ(f.pc.lookups, 2);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);          /* the hit went back */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 0);
    CHECK_EQ(f.sch.metrics().prompt_tokens, 160);   /* not counted until it is admitted */
    f.commit_plain();

    /* Meanwhile the cache's snapshot at 256 moved to another slot. */
    f.pc.src = 5;
    CHECK_OK(f.sch.cancel(r));
    CHECK_OK(f.sch.reap(r));

    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.pc.lookups, 3);
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 5);
    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.prompt_tokens, 160 + 1000);
    CHECK_EQ(m.attn_cached_tokens, 256);
    CHECK_EQ(m.linear_cached_tokens, 256);
}

/* A LOOKAHEAD THE POOL REFUSES IS A REASON NOT TO DRAFT, NOT A REASON TO SKIP THE ROW. The history
 * pass writes the draft head's K and V for positions the sequence already holds blocks for; a row
 * dropped here leaves them unwritten for good, because the next step's history starts at its own
 * context. */
TEST(a_refused_lookahead_still_runs_the_history_pass) {
    Fixture f;
    f.build(/*blocks=*/10, /*states=*/64);          /* exactly the prompt; nothing to look ahead into */
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 0);   /* nobody drafts */
    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK_EQ(hb->n_tok, 160);                             /* but the history is written */
    f.sch.serial_end(0);
}

/* DRAFT ROUND 1 RIDES THE HISTORY PASS. A verifying sequence that drafts owes the head's history at
 * all 1 + n_spec verified positions, and the card repeats the last committed one in the rows past
 * it, so the run's last row is round 1's whole input: the pass names that row, one a drafting
 * sequence, and round 1 has no pass of its own. */
TEST(the_history_pass_carries_draft_round_one) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    const uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->n_spec, 4);

    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)a);

    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK(hb->draft_pass < 0);
    CHECK_EQ(hb->n_tok, 5);                    /* every verified position, the last included */
    CHECK_EQ(hb->n_draft_out, 1);
    CHECK_EQ(hb->draft_out_ids[0], 4);         /* the run's last row */

    CHECK(f.sch.serial_batch(1, 1) == nullptr);
    const RadBatch* r2 = f.sch.serial_batch(2, 1);
    REQUIRE(r2 != nullptr);
    CHECK_EQ(r2->draft_pass, 2);
    CHECK_EQ(r2->n_tok, 1);
    CHECK_EQ(r2->n_draft_out, 0);
    f.sch.serial_end(2);
}

TEST(serial_tree_history_uses_the_current_committed_winners_in_a_mixed_batch) {
    Fixture f;
    f.build();
    /* A paged-only target needs no recurrent tree bindings for this lifecycle test. */
    f.prog.kv_groups.resize(1);
    f.prog.ops.resize(1);
    f.prog.buffers.resize(1);
    f.prog.drafter_tree_capacity = 8;
    f.cfg.n_spec = 7;
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    CHECK_OK(f.start());
    const uint64_t a = f.submit(8), b = f.submit(10);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_OK(rad_stream_sync(f.stream));
    f.commit_plain(11);

    radiance::CandidateTree tree;
    REQUIRE(tree.add_node(0, -1, 11, 0));
    REQUIRE(tree.add_node(1, 0, 21, 1));
    REQUIRE(tree.add_node(2, 1, 22, 2));
    REQUIRE(tree.add_node(3, 0, 31, 1));
    REQUIRE(tree.add_node(4, 1, 32, 2));
    CHECK_OK(f.sch.set_draft_tree(a, tree, 8));
    CHECK_OK(f.sch.set_draft_tree(b, tree, 10));
    const int32_t previous_path[] = {0, 2};
    f.sch.accepted_leaf(a, 2, previous_path, 2);
    f.sch.accepted_leaf(b, 2, previous_path, 2);
    const uint64_t prefill = f.submit(5), finished = f.submit(3, 1);
    const RadBatch* trunk = f.sch.step();
    REQUIRE(trunk != nullptr);
    CHECK_OK(rad_stream_sync(f.stream));
    const StepPlan& plan = f.sch.plan();
    REQUIRE(plan.e.size() == 4);
    std::vector<int32_t> sampled((size_t)trunk->n_out, -1);
    std::vector<int32_t> accepted(4), token(4);
    std::vector<uint8_t> eos(4, 0);
    int32_t starts[4] = {};
    int32_t first = 0;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        starts[i] = first;
        first += e.n_tokens;
        if (e.seq == a) {
            sampled[(size_t)e.dev_out] = 31;
            sampled[(size_t)e.dev_out + 3] = 41;
        } else if (e.seq == b) {
            sampled[(size_t)e.dev_out] = 21;
            sampled[(size_t)e.dev_out + 1] = 32;
            sampled[(size_t)e.dev_out + 4] = 42;
        } else {
            sampled[(size_t)e.dev_out] = e.seq == prefill ? 51 : 61;
        }
    }
    StepSlot slots[32]{};
    AdvanceArgs advance = f.sch.advance_args(0);
    advance.mode = ADV_ACCEPT;
    advance.src = sampled.data();
    advance.st = slots;
    for (int i = 0; i < advance.n_seq; ++i) adv_row(advance, i);
    int32_t tap_a = -1, tap_b = -1, tap_prefill = -1;
    for (size_t i = 0; i < plan.e.size(); ++i) {
        const StepEntry& e = plan.e[i];
        if (e.n_spec > 0) {
            int32_t emitted[16], path[16], bonus = -1;
            CHECK_OK(accept_tree_greedy(tree, sampled.data() + e.dev_out, e.n_tokens,
                                      emitted, &accepted[i], &bonus, path));
            CHECK_EQ(accepted[i], e.seq == a ? 1 : 2);
            for (int j = 0; j <= accepted[i]; ++j)
                CHECK_EQ(slots[e.card].path[j], path[j]);
            CHECK_EQ(path[accepted[i]], e.seq == a ? 3 : 4);
            token[i] = bonus;
            for (int j = 0; j < accepted[i]; ++j)
                const_cast<int32_t*>(e.draft)[j] = emitted[j];
            f.sch.accepted_leaf(e.seq, path[accepted[i]], path, accepted[i] + 1);
            radiance::TreeRegistry::instance().clear(e.seq);
            if (e.seq == a) tap_a = starts[i];
            else tap_b = starts[i];
        } else {
            token[i] = sampled[(size_t)e.dev_out];
            f.sch.accepted_leaf(e.seq, 0, nullptr, 0);
            if (e.seq == prefill) tap_prefill = starts[i];
        }
    }
    Scheduler::StepResult result;
    result.n_seq = 4;
    result.n_accepted = accepted.data();
    result.token = token.data();
    result.eos = eos.data();
    CHECK_OK(f.sch.commit(result));
    CHECK_EQ((int)f.sch.find(finished)->state, (int)ReqState::Finished);
    REQUIRE(tap_a >= 0 && tap_b >= 0 && tap_prefill >= 0);

    int rounds = 3;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4, true), 3);
    CHECK_EQ(rounds, 3);
    CHECK(rows[0].seq == a && rows[1].seq == b && rows[2].seq == prefill);
    const RadBatch* history = f.sch.serial_batch(0, 2);
    REQUIRE(history != nullptr);
    CHECK_OK(rad_stream_sync(f.stream));
    CHECK_EQ(history->n_seq, 3);
    CHECK_EQ(history->n_tok, 10);
    CHECK_EQ(history->n_draft_out, 3);
    const int32_t expected_tokens[] = {31, 41, 21, 32, 42, 1001, 1002, 1003, 1004, 51};
    const int32_t expected_positions[] = {8, 9, 10, 11, 12, 0, 1, 2, 3, 4};
    const int32_t expected_taps[] = {
        tap_a, tap_a + 3, tap_b, tap_b + 1, tap_b + 4,
        tap_prefill, tap_prefill + 1, tap_prefill + 2, tap_prefill + 3, tap_prefill + 4,
    };
    advance = f.sch.advance_args(0);
    advance.mode = ADV_HIST;
    advance.src = sampled.data();
    advance.st = slots;
    for (int i = 0; i < advance.n_seq; ++i) adv_row(advance, i);
    for (int i = 0; i < 10; ++i) {
        CHECK_EQ(history->token_ids[i], expected_tokens[i]);
        CHECK_EQ(history->positions[i], expected_positions[i]);
        CHECK_EQ(history->out_ids[i], expected_taps[i]);
    }
    CHECK_EQ(history->draft_out_ids[0], 1);
    CHECK_EQ(history->draft_out_ids[1], 4);
    CHECK_EQ(history->draft_out_ids[2], 9);
    for (int i = 0; i < 3; ++i) {
        int32_t anchor = -1;
        int64_t root = -1;
        CHECK(f.sch.block_committed_anchor(rows[i].seq, rows[i].slot, &anchor, &root));
        CHECK_EQ(anchor, i == 0 ? 41 : i == 1 ? 42 : 51);
        CHECK_EQ(root, i == 0 ? 10 : i == 1 ? 13 : 5);
    }
    const RadBatch* round2 = f.sch.serial_batch(2, 2);
    REQUIRE(round2 != nullptr);
    CHECK_OK(rad_stream_sync(f.stream));
    const int32_t proposals[] = {81, 82, 83};
    advance = f.sch.advance_args(0);
    advance.mode = ADV_ROUND;
    advance.round = 2;
    advance.src = proposals;
    advance.st = slots;
    for (int i = 0; i < advance.n_seq; ++i) adv_row(advance, i);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(round2->token_ids[i], proposals[i]);
        CHECK_EQ(round2->positions[i], i == 0 ? 10 : i == 1 ? 13 : 5);
    }
    f.sch.serial_end(2);

    /* A following chain owes only its new row, not the speculative rounds or the old tree path. */
    REQUIRE(f.sch.step() != nullptr);
    for (const StepEntry& e : f.sch.plan().e) f.sch.accepted_leaf(e.seq, 0, nullptr, 0);
    f.commit_plain(71);
    rounds = 3;
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4, true), 3);
    history = f.sch.serial_batch(0, 3);
    REQUIRE(history != nullptr);
    CHECK_OK(rad_stream_sync(f.stream));
    CHECK_EQ(history->n_tok, 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_EQ(history->token_ids[i], 71);
        CHECK_EQ(history->out_ids[i], i);
        CHECK_EQ(history->positions[i], i == 0 ? 10 : i == 1 ? 13 : 5);
    }
    f.sch.serial_end(1);
}

TEST(pruned_real_tree_state_preserves_five_rounds_and_seven_verify_proposals) {
    for (int live : {1, 2}) {
        Fixture f;
        f.build();
        /* Full-only KV avoids recurrent binding fakes, but the real tree-state policy is on. */
        f.prog.kv_groups.resize(1);
        f.prog.ops.resize(1);
        f.prog.buffers.resize(1);
        OpInfo commit;
        commit.op = "gdn_tree_commit";
        f.prog.ops.push_back(commit);
        OpInfo pack;
        pack.op = "mtp_pruned_pack";
        f.prog.ops.push_back(pack);
        f.prog.drafter.kind = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 7;
        f.prog.drafter_tree_capacity = 8;  /* Root + seven proposals. */
        f.prog.ctx.max_spec = 7;
        f.cfg.n_spec = 7;
        f.cfg.prefix_cache = false;
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        REQUIRE(tree_state_enabled(f.prog));
        CHECK_OK(f.start());
        uint64_t ids[2]{};
        for (int i = 0; i < live; ++i) ids[i] = f.submit(8);
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        f.commit_plain(11);

        int rounds = 5;
        Scheduler::SerialRow rows[2]{};
        REQUIRE(f.sch.serial_begin(&rounds, rows, 2, true) == live);
        CHECK_EQ(rounds, 5);  /* C2 must not inherit the legacy concurrent cap of four. */
        const RadBatch* history = f.sch.serial_batch(0, 1);
        REQUIRE(history != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        CHECK_EQ(history->n_seq, live);
        CHECK_EQ(history->n_tok, 8 * live);
        CHECK_EQ(history->n_draft_out, live);
        for (int i = 0; i < live; ++i) CHECK(rows[i].seq == ids[i]);
        for (int round = 2; round <= 5; ++round) {
            const RadBatch* frontier = f.sch.serial_frontier_batch(round, round, 0, live);
            REQUIRE(frontier != nullptr);
            CHECK_OK(rad_stream_sync(f.stream));
            CHECK_EQ(frontier->n_seq, live);
            CHECK_EQ(frontier->n_tok, 8 * live);
            for (int i = 0; i < frontier->n_tok; ++i)
                CHECK_EQ(frontier->positions[i], 8 + round - 2);
        }
        f.sch.serial_end(5);
        CHECK(f.sch.serial_frontier_batch(2, 1, 0, live) == nullptr);

        /* A valid root + B7 tree reaches the public publication and next-step planner paths. */
        radiance::CandidateTree tree;
        REQUIRE(tree.add_node(0, -1, 11, 0));
        REQUIRE(tree.add_node(1, 0, 21, 1));
        REQUIRE(tree.add_node(2, 0, 22, 1));
        REQUIRE(tree.add_node(3, 1, 31, 2));
        REQUIRE(tree.add_node(4, 2, 32, 2));
        REQUIRE(tree.add_node(5, 3, 41, 3));
        REQUIRE(tree.add_node(6, 4, 42, 3));
        REQUIRE(tree.add_node(7, 5, 51, 4));
        for (int i = 0; i < live; ++i)
            CHECK_OK(f.sch.set_draft_tree(ids[i], tree, 8));
        const RadBatch* verify = f.sch.step();
        REQUIRE(verify != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        CHECK_EQ(verify->n_seq, live);
        CHECK_EQ(verify->n_tok, 8 * live);
        REQUIRE(f.sch.plan().e.size() == (size_t)live);
        for (const StepEntry& e : f.sch.plan().e) {
            CHECK_EQ(e.n_spec, 7);  /* A separate regression: no B7 -> B4 planner clipping. */
            CHECK_EQ(e.n_tokens, 8);
            for (int32_t i = 0; i < e.n_spec; ++i)
                CHECK_EQ((int64_t)e.draft[i], (int64_t)tree[(size_t)i + 1].token_id);
            f.sch.accepted_leaf(e.seq, 0, nullptr, 0);
        }
        f.commit_plain(71);
        rounds = 5;
        REQUIRE(f.sch.serial_begin(&rounds, rows, 2, true) == live);
        CHECK_EQ(rounds, 5);
        history = f.sch.serial_batch(0, 2);
        REQUIRE(history != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        CHECK_EQ(history->n_seq, live);
        CHECK_EQ(history->n_tok, live);
        CHECK_EQ(history->n_draft_out, live);
        for (int i = 0; i < live; ++i) {
            CHECK_EQ(history->token_ids[i], 71);
            CHECK_EQ(history->positions[i], 8);
        }
        f.sch.serial_end(1);
    }
}

TEST(pruned_context_tail_keeps_history_without_blocking_an_eligible_peer) {
    /* topk=8 keeps the tail inside QSA exactness; topk=7 crosses both bounds.
     * Neither context-ineligible tail may prevent the other request from drafting. */
    for (int topk : {8, 7}) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/1024);
        f.prog.kv_groups.resize(1);
        f.prog.ops.resize(1);
        f.prog.buffers.resize(1);
        /* Keep full-only KV, while enabling the actual tree-state controller branch. */
        OpInfo commit;
        commit.op = "gdn_tree_commit";
        f.prog.ops.push_back(commit);
        OpInfo pack;
        pack.op = "mtp_pruned_pack";
        f.prog.ops.push_back(pack);
        OpInfo qsa;
        qsa.op = "qsa_select";
        qsa.base.set_i("topk", topk);
        qsa.base.set_i("ratio", 64);
        f.prog.ops.push_back(qsa);
        f.prog.drafter.kind = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 7;
        f.prog.drafter_tree_capacity = 8;  /* Root + seven proposals. */
        f.prog.ctx.max_spec = 7;
        f.cfg.max_ctx = 512;
        f.cfg.n_spec = 7;
        f.cfg.prefix_cache = false;
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
        REQUIRE(tree_state_enabled(f.prog));
        CHECK_OK(f.start());

        const uint64_t tail = f.submit(507), eligible = f.submit(8);
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        f.commit_plain(11);
        CHECK_EQ(f.sch.find(tail)->n_computed, 507);
        CHECK_EQ(f.sch.find(eligible)->n_computed, 8);
        CHECK_EQ((int)f.sch.find(tail)->state, (int)ReqState::Running);

        int rounds = 5;
        Scheduler::SerialRow rows[2]{};
        REQUIRE(f.sch.serial_begin(&rounds, rows, 2, true) == 1);
        CHECK_EQ(rounds, 5);
        CHECK(rows[0].seq == eligible);
        const RadBatch* history = f.sch.serial_batch(0, 1);
        REQUIRE(history != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        CHECK_EQ(history->n_seq, 2);
        CHECK_EQ(history->n_tok, 515);
        CHECK_EQ(history->n_draft_out, 1);
        CHECK_EQ(history->draft_out_ids[0], 7);
        /* Drafting rows come first; the entire tail prompt still owes history. */
        for (int i = 0; i < 8; ++i) {
            CHECK_EQ(history->positions[i], i);
            CHECK_EQ(history->token_ids[i], i == 7 ? 11 : 1001 + i);
        }
        for (int i = 0; i < 507; ++i) {
            CHECK_EQ(history->positions[8 + i], i);
            CHECK_EQ(history->token_ids[8 + i], i == 506 ? 11 : 1001 + i);
        }
        for (int round = 2; round <= 5; ++round) {
            const RadBatch* frontier = f.sch.serial_frontier_batch(round, round, 0, 1);
            REQUIRE(frontier != nullptr);
            CHECK_OK(rad_stream_sync(f.stream));
            CHECK_EQ(frontier->n_seq, 1);
            CHECK_EQ(frontier->n_tok, 8);
            for (int i = 0; i < 8; ++i) CHECK_EQ(frontier->positions[i], 8 + round - 2);
        }
        CHECK(f.sch.serial_frontier_batch(2, 1, 0, 2) == nullptr);
        f.sch.serial_end(5);
        CHECK(f.sch.serial_batch(0, 1) == nullptr);
        CHECK(f.sch.serial_frontier_batch(2, 1, 0, 1) == nullptr);

        /* Reopening this committed plan proves both history watermarks advanced:
         * only the verified rows remain, not either prompt's host-known lead. */
        rounds = 5;
        REQUIRE(f.sch.serial_begin(&rounds, rows, 2, true) == 1);
        CHECK_EQ(rounds, 5);
        history = f.sch.serial_batch(0, 1);
        REQUIRE(history != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        CHECK_EQ(history->n_seq, 2);
        CHECK_EQ(history->n_tok, 2);
        CHECK_EQ(history->positions[0], 7);
        CHECK_EQ(history->positions[1], 506);
        f.sch.serial_end(1);

        /* Both requests continue into the next committed step. */
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        f.commit_plain(71);
        rounds = 5;
        REQUIRE(f.sch.serial_begin(&rounds, rows, 2, true) == 1);
        CHECK_EQ(rounds, 5);
        CHECK(rows[0].seq == eligible);
        history = f.sch.serial_batch(0, 2);
        REQUIRE(history != nullptr);
        CHECK_OK(rad_stream_sync(f.stream));
        CHECK_EQ(history->n_seq, 2);
        CHECK_EQ(history->n_tok, 2);
        CHECK_EQ(history->n_draft_out, 1);
        CHECK_EQ(history->draft_out_ids[0], 0);
        CHECK_EQ(history->positions[0], 8);
        CHECK_EQ(history->positions[1], 507);
        CHECK_EQ(history->token_ids[0], 71);
        CHECK_EQ(history->token_ids[1], 71);
        CHECK_EQ((int)f.sch.find(tail)->state, (int)ReqState::Running);
        CHECK_EQ((int)f.sch.find(eligible)->state, (int)ReqState::Running);
        f.sch.serial_end(1);
    }
}

TEST(pruned_serial_begin_still_refuses_qsa_inexact_drafting) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/1024);
    f.prog.kv_groups.resize(1);
    f.prog.ops.resize(1);
    f.prog.buffers.resize(1);
    OpInfo pack;
    pack.op = "mtp_pruned_pack";
    f.prog.ops.push_back(pack);
    OpInfo qsa;
    qsa.op = "qsa_select";
    qsa.base.set_i("topk", 7);
    qsa.base.set_i("ratio", 64);       /* dense-exact through 511 */
    f.prog.ops.push_back(qsa);
    f.cfg.max_ctx = 1024;             /* enough normal context for the six-position lookahead */
    f.cfg.n_spec = 7;
    f.cfg.prefix_cache = false;
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    CHECK_OK(f.start());
    const uint64_t id = f.submit(507);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_OK(rad_stream_sync(f.stream));
    f.commit_plain(11);
    CHECK_EQ(f.sch.find(id)->n_computed, 507);

    int rounds = 5;
    Scheduler::SerialRow rows[1]{};
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 1, true), RAD_E_UNSUPPORTED);
    f.sch.serial_end(0);
}

/* A STEP IS ISSUED AHEAD OF A COMMIT ONLY FOR A REQUEST WHOSE SPECULATION THAT COMMIT CANNOT
 * CHANGE. The ahead step is sized before the commit in front of it, so a request the sampler has
 * not started yet -- whose grammar, if any, nobody has asked -- is not issued ahead; one whose
 * grammar rewinds is, grammar or not. */
TEST(a_step_is_issued_ahead_only_once_the_requests_speculation_is_fixed) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    const uint64_t id = f.submit(8, 64);
    CHECK(f.sch.step() != nullptr);                 /* the prompt */
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);                 /* the first decode */
    CHECK(f.sch.step_next() == nullptr);            /* the sampler has not started it */

    Request* r = f.sch.find(id);
    REQUIRE(r != nullptr);
    r->sp.grammar = "root ::= \"a\"";
    r->speculation_fixed = true;                    /* a grammar whose backend rewinds */
    CHECK(f.sch.step_next() != nullptr);
}

/* THE CONTEXT BOUNDS A SEQUENCE WHATEVER max_tokens SAYS, "no limit" included. Nothing past the
 * last position the deployment can address has a rotary row or a block-table column; a sequence
 * allowed to run into it corrupts memory instead of stopping. */
TEST(a_sequence_stops_at_the_context_whatever_max_tokens_says) {
    Fixture f;
    f.build();
    f.cfg.max_ctx = 170;
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, /*max_tokens=*/0);        /* unlimited */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();

    int32_t draft[4] = { 21, 22, 23, 24 };
    for (int guard = 0; guard < 32 && f.sch.find(a)->state == ReqState::Running; ++guard) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        const StepEntry& e = f.sch.plan().e[0];
        CHECK(e.ctx_len + e.n_tokens <= 170);           /* no position at or past the end */
        /* Every draft accepted: the most a step can emit, so the bound has to cut it short. */
        int32_t acc[1] = { e.n_spec };
        int32_t tok[1] = { 9 };
        uint8_t eos[1] = { 0 };
        Scheduler::StepResult r;
        r.n_seq = 1; r.n_accepted = acc; r.token = tok; r.eos = eos;
        CHECK_OK(f.sch.commit(r));
    }
    const Request* ra = f.sch.find(a);
    CHECK_EQ((int)ra->state, (int)ReqState::Finished);
    CHECK_EQ(ra->finish_reason, std::string("length"));
    CHECK_EQ((int)(ra->prompt.size() + ra->output.size()), 170);

    /* A prompt that fills the context has nothing to generate into and is never run. */
    uint64_t full = f.submit(170, 8);
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ((int)f.sch.find(full)->state, (int)ReqState::Failed);

    /* And asking for more than the context holds reserves only what it can hold. */
    uint64_t big = f.submit(8, 2147483647);
    CHECK(f.sch.find(big)->output.capacity() <= (size_t)(170 + RAD_SCHED_MAX_SPEC));
}

/* THE CACHE'S RESERVE IS MEASURED AGAINST WHAT IS HELD, NOT AGAINST WHAT IS BACKED. An elastic
 * pool that has handed its unused top back to the card reads almost no free blocks however little
 * anything holds, and trimming against that drains the whole cache after every request. */
TEST(the_cache_trim_does_not_mistake_released_vram_for_occupancy) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.evict_grant = 1 << 20;
    f.kv.unbacked = 30;                              /* the pool gave 30 blocks back */

    f.submit(160, /*max_tokens=*/1);                 /* holds 10 of 40 */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();                                /* finishes and publishes */
    CHECK_EQ((int)f.pc.published.size(), 1);
    CHECK_EQ(f.pc.evict_calls, 0);
}

/* AN ENGINE THAT HAS STOPPED ANSWERS EVERY REQUEST IT HELD AND REFUSES NEW ONES, rather than
 * leaving their connections waiting on a step that will never come. */
TEST(shutdown_fails_what_is_queued_and_refuses_what_follows) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    const int64_t before = f.kv.free_blocks(0);
    uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    uint64_t b = f.submit(160, 64);

    f.sch.shutdown("stopped");
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Failed);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Failed);
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK(f.submit(8, 8) == 0);
    CHECK(f.sch.step() == nullptr);
}

/* AND EACH OF THOSE REQUESTS SAYS WHAT STOPPED IT. The sink carries the status and the sentence
 * the server puts in the 500: without them a client read "the engine failed this request: ok"
 * after a step failed as unsupported. */
TEST(shutdown_fails_each_sink_with_the_status_that_stopped_the_engine) {
    server::Sink running, waiting, late;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());

    CHECK(f.submit_to(&running, 160, 64) != 0);
    CHECK(f.sch.step() != nullptr);
    CHECK(f.submit_to(&waiting, 160, 64) != 0);

    f.sch.shutdown("the engine stopped", RAD_E_UNSUPPORTED);
    CHECK(f.submit_to(&late, 8, 8) == 0);
    for (server::Sink* s : { &running, &waiting, &late }) {
        CHECK(s->done());
        CHECK_EQ((int)s->finish_reason(), (int)server::Finish::Error);
        CHECK_EQ(s->status(), (int)RAD_E_UNSUPPORTED);
        CHECK_EQ(std::string(s->why() ? s->why() : ""), std::string("the engine stopped"));
    }
}

/* A REQUEST'S OWN SPECULATION AND ITS ADMISSION REACH ITS SINK, which is what lets a response say
 * how many tokens were drafted for it and when its prefill started: the aggregate counters cannot
 * be split between requests that overlap. The count is the drafts each verify step proposed for
 * this request and the ones it accepted -- the same numbers /metrics sums. */
TEST(a_sink_carries_its_requests_admission_and_draft_counts) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());

    const uint64_t id = f.submit_to(&sink, 40, 60);
    CHECK(id != 0);
    CHECK(sink.t_admitted_ns() < 0);
    int64_t drafted = 0, accepted = 0;
    /* And the window each decode row verified, which is the pairing the sink exists to allow: a
     * comparator can tell a request measured at four drafts from one measured at one. */
    int64_t rows_at[8] = {};
    int64_t decode_rows = 0;
    uint32_t rng = 99;
    for (int step = 0; step < 40 && !sink.done(); ++step) {
        int32_t draft[4] = { 31, 32, 33, 34 };
        (void)f.sch.set_draft(id, draft, 4);
        const RadBatch* b = f.sch.step();
        if (!b) break;
        CHECK(sink.t_admitted_ns() >= 0);
        const StepPlan& p = f.sch.plan();
        Scheduler::StepResult r;
        static int32_t acc[64], tok[64];
        static uint8_t eos[64];
        for (size_t k = 0; k < p.e.size(); ++k) {
            rng = rng * 1664525u + 1013904223u;
            acc[k] = p.e[k].n_spec > 0 ? (int32_t)((rng >> 16) % (uint32_t)(p.e[k].n_spec + 1)) : 0;
            tok[k] = 5;
            eos[k] = 0;
            if (!p.e[k].is_prefill) {
                ++decode_rows;
                if (p.e[k].n_spec >= 0 && p.e[k].n_spec < 8) ++rows_at[p.e[k].n_spec];
            }
            if (!p.e[k].is_prefill && p.e[k].n_spec > 0) {
                drafted += p.e[k].n_spec;
                accepted += acc[k];
            }
        }
        r.n_seq = (int)p.e.size();
        r.n_accepted = acc; r.token = tok; r.eos = eos;
        CHECK_OK(f.sch.commit(r));
    }
    CHECK(drafted > 0);
    CHECK_EQ(sink.n_drafted(), drafted);
    CHECK_EQ(sink.n_draft_accepted(), accepted);
    /* EVERY decode row is recorded under the window it actually verified, depth 0 included, so the
     * histogram sums to the row count and not to a subset of it. */
    int64_t hist_rows = 0;
    for (int d = 0; d < 8; ++d) {
        CHECK_EQ(sink.n_rows_at_depth(d), rows_at[d]);
        hist_rows += sink.n_rows_at_depth(d);
    }
    CHECK_EQ(hist_rows, decode_rows);
    CHECK(rows_at[4] > 0);
}

/* THE STAGED BLOCK-TABLE RECTANGLE MUST BE AS WIDE AS THE MANAGER'S CAP, TO THE BLOCK.
 *
 * build() pitches a row from the step's context bound and clamps it to the group's max_blocks. A
 * rectangle one row narrower than the table the manager can hand out therefore drops the NEWEST
 * entries -- the ones every attention kernel reads as the current context -- and nothing refuses:
 * the sequence silently attends to a block older than it should. This asserts the staged width is
 * the manager's own expression, at the DFlash2 drafter's real geometry (2048-token window,
 * 16-token block) for both the native depth 7 (block 8) and depth 8. The two reserves differ --
 * 16 tokens against 18 -- and at 16-token blocks land in the same cap, so the assertion is against
 * the expression rather than against a constant.
 *
 * The fake manager here does not recycle, so the table is filled to exactly the cap: that is the
 * worst case for the rectangle and the case where a missing row is the newest one. */
TEST(the_staged_block_table_holds_the_manager_window_cap) {
    const int depths[2] = { 7, 8 };
    for (int di = 0; di < 2; ++di) {
        const int depth = depths[di];
        Fixture f;
        f.build();
        /* AFTER build(), which pins n_spec to 0 -- this fixture builds a Scheduler directly and
         * never runs the engine's auto-resolution. */
        f.cfg.n_spec = depth;

        KVGroupInfo swa;
        swa.name = "draft";
        swa.decl.kind = RAD_KV_WINDOW;
        swa.decl.dtype = RAD_BF16;
        swa.decl.window = 2048;
        swa.decl.n_head_kv = 2;
        swa.decl.head_dim = 16;
        swa.block_size = 16;
        swa.layers = { 6 };
        f.prog.kv_groups.push_back(swa);
        const int32_t gi = (int32_t)f.prog.kv_groups.size() - 1;
        /* Re-set the fake up over the grown declaration: the new group needs a free list and a
         * geometry row, and nothing is allocated yet. */
        f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);

        const int64_t expect =
            window_blocks_for(2048, 16, speculative_window_reserve(depth));
        CHECK(expect > (2048 + 15) / 16);   /* the reserve really widens the cap */

        KVGeom kvg = KVGeom::of(f.prog.kv_groups);
        if (!f.stream) rad_stream_create(&f.stream, 0);
        BatchBuilder bb;
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, f.stream, &f.prog });
        ChunkGeometry geo;
        CHECK_OK(chunk_geometry_resolve(f.prog, f.cfg, &geo));
        CHECK_OK(bb.init(f.prog, f.cfg, geo, kvg, rio));

        CHECK_OK(f.kv.add_sequence(1));
        /* Exactly the cap's worth of tokens, so the table ends on its newest block. */
        CHECK_OK(f.kv.ensure(1, expect * 16));
        StepPlan p;
        StepEntry e;
        e.seq = 1;
        e.card = 0;
        e.n_tokens = 1;
        e.ctx_len = (int32_t)(expect * 16 - 1);
        p.e.push_back(e);
        p.n_tok = 1;
        const RadBatch* b = bb.build(p, f.kv, 0);
        CHECK(b != nullptr);
        /* THE STAGING CROSSES ASYNC -- a worker thread even on the host backend -- so the device
         * plane holds the staged values only once the build's stream has drained. Reading it
         * earlier reads whatever the slab held, which is zeroes on a fresh allocation; this is
         * the idiom the other device-plane cases in this file use. */
        CHECK_EQ(rad_stream_sync(f.stream), RAD_OK);
        if (b) {
            const RadKVGroupBatch& g = b->kv[(size_t)gi];
            /* THE CONTRACT: the staged rectangle is the manager's cap, not the window's ceil and
             * not the context bound. That is the field size_set() fills. */
            CHECK_EQ(g.max_blocks, expect);
            /* The pitch is this STEP's staging width and may legitimately be smaller than the
             * cap (it follows the context bound), but never smaller than the table it carries. */
            CHECK(g.block_table_pitch >= (int64_t)f.kv.block_table(1, gi).size());
            const std::vector<int32_t>& bt = f.kv.block_table(1, gi);
            CHECK_EQ((int64_t)bt.size(), expect);
            /* The newest block -- the row a 129-wide rectangle would have dropped -- is staged
             * at its own index and is the last one in the table. */
            CHECK_EQ(g.block_table[expect - 1], bt.back());
            CHECK_EQ(g.block_table[expect - 1], bt[(size_t)(expect - 1)]);
        }
        bb.fini();
    }
}

RAD_TEST_MAIN()
