/* sched_test.cpp -- the scheduler, chunker, step batch, derivations and draft controller, against
 * fakes for core/mem and the device layer (sched_fakes.h).
 */
#include "rad_test.h"

#include "mm/processor.h"
#include "sched/mtp_couple_bind.h"
#include "sched/scheduler.h"
#include "sched/advance_impl.h"
#include "sched/derive.h"
#include "server/sink.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include <unistd.h>

using namespace rad;

/* ================================================================== allocation counter */
/* The step path must not allocate. Counting through a replaced global operator new is the only
 * reliable check: review can miss a vector that grows on the ten-thousandth request, which shows
 * up as an unreproducible latency spike. */
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
/* The nothrow forms too (std::get_temporary_buffer uses them): left to the runtime, their blocks
 * would reach the free() below, a mismatch under ASan, and their allocations would go uncounted. */
void* operator new(size_t n, const std::nothrow_t&) noexcept {
    if (g_counting) ++g_allocs;
    return std::malloc(n ? n : 1);
}
void* operator new[](size_t n, const std::nothrow_t& t) noexcept { return operator new(n, t); }
void  operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void  operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void  operator delete(void* p) noexcept { std::free(p); }
void  operator delete[](void* p) noexcept { std::free(p); }
void  operator delete(void* p, size_t) noexcept { std::free(p); }
void  operator delete[](void* p, size_t) noexcept { std::free(p); }

#include "sched_fakes.h"

using namespace sched_fake;

namespace {

/* ------------------------------------------------------------------ a declared program */
/* A paged group beside a linear one: the chunk geometry reconciles the two, so a second paged
 * group would add nothing.
 *
 * The op carries a schema because that is where the chunk parameter is identified
 * (RAD_PROLE_SEQ_CHUNK). The geometry and the kernel's constraint must name the same key, since a
 * constraint on a key the geometry lacks does not hold (rad_abi.h). */
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

    ~Fixture() { sch.fini(); for (void* p : owned) std::free(p); }

    void build(int64_t blocks = 4096, int64_t states = 64, int64_t max_tok = 512,
               int64_t max_ctx = 4096, bool windowed = false) {
        g_gdn_ki = RadKernelInfo{};
        g_gdn_ki.name = "gdn_fake";
        g_gdn_ki.op = "gdn_chunked";
        g_gdn_ki.constraints = g_gdn_c;
        g_gdn_ki.n_constraints = 1;
        g_gdn_row = KernelRow{ &g_gdn_ki, "fake", 0, 0 };

        prog.meta.n_ctx_train = 4096;
        prog.meta.n_vocab = 512;

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

        /* A windowed paged group, as sliding-window checkpoints pair with a full group. Not
         * prefix-cached: prefix.cpp indexes only groups whose blocks below the window stay put.
         * Tests wanting a prefix cache over a WINDOW+FULL pair trim pc.groups to the cached ones. */
        if (windowed) {
            KVGroupInfo win;
            win.name = "swa";
            win.decl.kind = RAD_KV_WINDOW;
            win.decl.window = 64;
            win.block_size = 16;
            win.layers = { 2 };   /* bound to a trunk layer, the group a trunk attention reads;
                                   * a drafter's own group is bound at the trunk's layer count and
                                   * above (rad_bind_layer_kv at the drafter declare) */
            prog.kv_groups.push_back(win);
            prog.meta.n_layers = 16;   /* trunk layers 0..15; the drafter tests' appended group
                                        * stays unbound, and unbound is not trunk-owned */
        }

        OpInfo op;
        op.op = "gdn_chunk_scan";
        op.schema = &g_gdn_schema;
        Band b;
        b.hi = 4096;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_gdn_row;
        /* The architecture plugin's declared geometry, in arch/common/rad_block_gdn.h's spelling,
         * the key its schema marks RAD_PROLE_SEQ_CHUNK. Both this and the kernel's constraint
         * contribute. */
        b.dom[RAD_DOMAIN_DEVICE].geom.set_i("chunk", 64);
        op.bands.push_back(b);
        prog.ops.push_back(op);

        add_derived("seq_start", "seq_start_index", 257);
        add_derived("chunk_bounds", "state_chunk_bounds", 4097);
        add_derived("conv_idx", "conv_state_index", 256);

        cfg.max_tok = max_tok;
        cfg.max_seqs = 32;
        cfg.max_ctx = max_ctx;
        cfg.checkpoint_interval = 256;
        cfg.prefix_cache = true;
        /* Config::n_spec defaults to -1 (`auto`), which engine_bringup resolves; this fixture
         * skips that, and DraftController::configure refuses a negative depth. Speculation tests
         * set their own depth. */
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
        std::vector<BatchBuilder::RankIO> rio;
        rio.push_back({ 0, nullptr, &prog });
        return sch.init(prog, cfg, &kv, &pc, rio);
    }

    const int32_t* derived(const char* name) const {
        for (const BufferInfo& b : prog.buffers)
            if (b.name == name) return (const int32_t*)b.ptr;
        return nullptr;
    }

    uint64_t next_id = 1;
    /* A request for prompt tokens 1000, 1001, ... */
    std::unique_ptr<Request> request(int n_prompt, int max_tokens) {
        auto r = std::make_unique<Request>();
        r->id = next_id++;
        r->max_tokens = max_tokens;
        r->prompt.resize((size_t)n_prompt);
        for (int i = 0; i < n_prompt; ++i) r->prompt[(size_t)i] = 1000 + i;
        return r;
    }
    /* Its id, or 0 when the scheduler refused it. */
    uint64_t add(std::unique_ptr<Request> r) {
        const uint64_t id = r->id;
        return sch.add(std::move(r)) < 0 ? 0 : id;
    }

    uint64_t submit(int n_prompt, int max_tokens = 64, int priority = 0, bool ignore_eos = false) {
        auto r = request(n_prompt, max_tokens);
        r->priority = priority;
        r->ignore_eos = ignore_eos;
        return add(std::move(r));
    }

    /* One whose prompt carries a picture the encoder has not run yet: it waits in the queue until
     * `*item` says it is encoded. */
    uint64_t submit_unencoded(int n_prompt, int max_tokens, std::shared_ptr<mm::Item>* item) {
        auto r = request(n_prompt, max_tokens);
        *item = std::make_shared<mm::Item>();
        r->media = std::make_shared<mm::PromptMedia>();
        r->media->items.push_back(mm::Placed{*item, 0});
        return add(std::move(r));
    }

    /* The same, with the server's sink attached, for what a request tells its client. */
    uint64_t submit_to(server::Sink* sink, int n_prompt, int max_tokens) {
        auto r = request(n_prompt, max_tokens);
        r->sink = sink;
        return add(std::move(r));
    }

    /* Commit the planned step: every entry sampled `token`, ended when `eos`, and accepted
     * `acc(entry)` drafts (none when `acc` is empty). */
    int commit(int32_t token, uint8_t eos = 0,
               const std::function<int32_t(const StepEntry&)>& acc = nullptr) {
        const StepPlan& p = sch.plan();
        std::vector<int32_t> a(p.e.size(), 0), t(p.e.size(), token);
        std::vector<uint8_t> e(p.e.size(), eos);
        if (acc)
            for (size_t i = 0; i < p.e.size(); ++i) a[i] = acc(p.e[i]);
        Scheduler::StepResult r;
        r.n_seq = (int)p.e.size();
        r.n_accepted = a.data();
        r.token = t.data();
        r.eos = e.data();
        return sch.commit(r);
    }

    /* Commit a step in which every sequence produced one token and none ended. */
    void commit_plain(int32_t token = 7) { CHECK_OK(commit(token)); }
};

/* A uniformly random number of `e`'s drafts accepted, 0..n_spec, from an LCG. */
int32_t random_accept(uint32_t& rng, const StepEntry& e) {
    rng = rng * 1664525u + 1013904223u;
    return e.n_spec > 0 ? (int32_t)((rng >> 16) % (uint32_t)(e.n_spec + 1)) : 0;
}

}  /* namespace */

/* ================================================================== the fakes are linked */
/* Every check below reads device memory as host memory through the weak fakes. One device call
 * core/ makes and the fakes lack pulls the real backend's dispatch.cpp.o into the link, which
 * replaces every fake; the binary then still runs, on another backend, and proves something
 * else. */
TEST(the_device_fakes_are_the_ones_linked) {
    g_fake_device = false;
    void* p = rad_dev_alloc(16, RAD_MEM_DEVICE);
    CHECK(p != nullptr);
    CHECK(g_fake_device);
    rad_dev_free(p, RAD_MEM_DEVICE);
}

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

/* A model with no linear state checkpoints only for a trunk window: the snapshot of the window
 * behind a prompt's tail is what a prefix hit resumes from. A drafter's own window is recomputed
 * by draft refill, so it takes no checkpoint and leaves the hit at any block. */
TEST(chunk_geometry_snapshots_a_trunk_window_and_not_a_drafters) {
    for (int drafter = 0; drafter < 2; ++drafter) {
        Program prog;
        prog.meta.n_layers = 16;
        prog.kv_groups.emplace_back();
        KVGroupInfo attn;
        attn.name = "attn";
        attn.decl.kind = RAD_KV_FULL;
        attn.block_size = 16;
        attn.layers = { 0 };
        prog.kv_groups.push_back(attn);
        KVGroupInfo win;
        win.name = "swa";
        win.decl.kind = RAD_KV_WINDOW;
        win.decl.window = 64;
        win.block_size = 16;
        win.layers = { drafter ? 16 : 1 };
        win.drafter = drafter != 0;
        prog.kv_groups.push_back(win);
        Config cfg;
        cfg.max_tok = 512;
        cfg.checkpoint_interval = 256;
        cfg.prefix_cache = true;
        ChunkGeometry g;
        CHECK_OK(chunk_geometry_resolve(prog, cfg, &g));
        CHECK_EQ(g.checkpoint_interval, drafter ? 0ll : 256ll);
        CHECK(g.window_ckpt == !drafter);
    }
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
    /* The budget would allow 512 tokens; the split takes 256 so the chunk ends on the interval. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck), 256);
    CHECK_EQ(ck, 256);

    CHECK_EQ(f.sch.plan_chunk(256, 1000, 512, &ck), 256);
    CHECK_EQ(ck, 512);

    /* The final chunk ends at the prompt's last quantum boundary and writes a checkpoint there, so
     * the next turn of a conversation resumes from it instead of from 768. The remainder, under a
     * quantum, is the next chunk and writes none: the accepted cost of owning the boundaries. */
    CHECK_EQ(f.sch.plan_chunk(768, 1000, 512, &ck), 192);
    CHECK_EQ(ck, 960);
    CHECK_EQ(f.sch.plan_chunk(960, 1000, 512, &ck), 40);
    CHECK_EQ(ck, -1);
    /* A prompt that ends less than a quantum past where the chunk starts has no boundary to
     * split at, and one inside the first interval is cheap enough to prefill again. */
    CHECK_EQ(f.sch.plan_chunk(768, 830, 512, &ck), 62);
    CHECK_EQ(ck, -1);
    CHECK_EQ(f.sch.plan_chunk(0, 200, 512, &ck), 200);
    CHECK_EQ(ck, -1);
    /* And a prompt ending on a quantum boundary takes its checkpoint at the end, unsplit. */
    CHECK_EQ(f.sch.plan_chunk(768, 960, 512, &ck), 192);
    CHECK_EQ(ck, 960);

    /* A budget short of the interval still ends on a quantum, never in the middle of a tile. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 100, &ck), 64);
    CHECK_EQ(ck, -1);

    /* A budget that cannot hold one quantum schedules nothing rather than a misaligned chunk. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 50, &ck), 0);
}

/* A request's own checkpoint (Request::checkpoint_at): the end of a head other requests share, on
 * the quantum below it. It splits a chunk as the interval does and gives way to nothing. */
TEST(prefill_chunks_split_where_the_request_asks_for_a_checkpoint) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    int64_t ck = -1;
    /* Inside the first interval, where nothing else would write one: split at 128, the quantum
     * under 150, then the rest with none. */
    CHECK_EQ(f.sch.plan_chunk(0, 200, 512, &ck, 150), 128);
    CHECK_EQ(ck, 128);
    CHECK_EQ(f.sch.plan_chunk(128, 200, 512, &ck, 150), 72);
    CHECK_EQ(ck, -1);
    /* Past the interval checkpoint that comes first, and before the next one. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck, 600), 256);
    CHECK_EQ(ck, 256);
    CHECK_EQ(f.sch.plan_chunk(512, 1000, 512, &ck, 600), 64);
    CHECK_EQ(ck, 576);
    /* On a boundary the chunk would end on anyway, it is that boundary's checkpoint. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 512, &ck, 256), 256);
    CHECK_EQ(ck, 256);
    /* A head shorter than a quantum has no boundary to keep. */
    CHECK_EQ(f.sch.plan_chunk(0, 200, 512, &ck, 40), 200);
    CHECK_EQ(ck, -1);
    /* And a budget short of it still ends on a quantum, the checkpoint left for a later chunk. */
    CHECK_EQ(f.sch.plan_chunk(0, 1000, 100, &ck, 150), 64);
    CHECK_EQ(ck, -1);
    CHECK_EQ(f.sch.plan_chunk(64, 1000, 512, &ck, 150), 64);
    CHECK_EQ(ck, 128);
}

TEST(prefill_chunks_span_intervals_on_the_same_pieces) {
    Fixture f;
    f.build();                            /* interval 256, budget 512 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().span, 2);

    int64_t ck = -1;
    /* The first interval is never spanned. */
    CHECK_EQ(f.sch.plan_chunk(0, 2000, 512, &ck), 256);
    CHECK_EQ(ck, 256);
    /* From a boundary past it, two intervals and a checkpoint at the end; the one between them
     * is not written. */
    CHECK_EQ(f.sch.plan_chunk(256, 2000, 512, &ck), 512);
    CHECK_EQ(ck, 768);
    CHECK_EQ(f.sch.plan_chunk(1280, 2000, 512, &ck), 512);
    CHECK_EQ(ck, 1792);
    /* The tail checkpoint where a span of 1 puts it. */
    CHECK_EQ(f.sch.plan_chunk(1792, 2000, 512, &ck), 192);
    CHECK_EQ(ck, 1984);
    /* A last chunk that reaches the prompt's end runs through to its tail checkpoint: the pieces
     * are [512, 768) and [768, 960), as a span of 1 cuts them. */
    CHECK_EQ(f.sch.plan_chunk(512, 1000, 512, &ck), 448);
    CHECK_EQ(ck, 960);
    /* And without one, it stops on the last boundary it crosses, where a span of 1 writes the
     * prompt's deepest checkpoint; the rest is the next chunk. */
    CHECK_EQ(f.sch.plan_chunk(512, 830, 512, &ck), 256);
    CHECK_EQ(ck, 768);
    CHECK_EQ(f.sch.plan_chunk(768, 830, 512, &ck), 62);
    CHECK_EQ(ck, -1);
    /* A budget short of the span steps back to the last boundary it crosses. */
    CHECK_EQ(f.sch.plan_chunk(256, 2000, 500, &ck), 256);
    CHECK_EQ(ck, 512);
    /* Off the interval (a resume from a tail checkpoint) the chunk ends at the next boundary. */
    CHECK_EQ(f.sch.plan_chunk(960, 3000, 512, &ck), 64);
    CHECK_EQ(ck, 1024);
}

/* A span moves no checkpoint and no piece, from a prompt's start or from any checkpoint a turn
 * resumes from: every chunk end and checkpoint it makes is one a span of 1 makes, and the deepest
 * checkpoint (where the next turn resumes) is the same. */
TEST(a_span_writes_a_subset_of_the_checkpoints_and_the_same_deepest_one) {
    struct Walk { std::vector<int64_t> ends, cks; };
    auto walk = [](Fixture& f, int64_t from, int64_t target) {
        Walk w;
        for (int64_t n = from; n < target; ) {
            int64_t ck = -1;
            const int64_t c = f.sch.plan_chunk(n, target, f.cfg.max_tok, &ck);
            if (c <= 0) break;
            n += c;
            w.ends.push_back(n);
            if (ck >= 0) w.cks.push_back(ck);
        }
        return w;
    };
    Fixture one;
    one.build();                          /* interval 256, budget 512, quantum 64 */
    CHECK_OK(one.start());
    Fixture two;
    two.build();
    two.cfg.prefill_span = 2;
    CHECK_OK(two.start());
    int bad_end = 0, bad_ck = 0, bad_deep = 0, spanned = 0;
    for (int64_t target = 1; target <= 2600; target += 7) {
        const Walk ref = walk(one, 0, target);
        /* From the start, and from every checkpoint a span of 1 leaves to resume from. */
        std::vector<int64_t> from = { 0 };
        for (int64_t c : ref.cks) if (c < target) from.push_back(c);
        for (int64_t s0 : from) {
            const Walk a = walk(one, s0, target);
            const Walk b = walk(two, s0, target);
            for (int64_t e : b.ends)
                bad_end += std::find(a.ends.begin(), a.ends.end(), e) == a.ends.end();
            for (int64_t c : b.cks)
                bad_ck += std::find(a.cks.begin(), a.cks.end(), c) == a.cks.end();
            const int64_t da = a.cks.empty() ? -1 : a.cks.back();
            const int64_t db = b.cks.empty() ? -1 : b.cks.back();
            bad_deep += da != db;
            spanned += b.ends.size() < a.ends.size();
        }
    }
    CHECK_EQ(bad_end, 0);
    CHECK_EQ(bad_ck, 0);
    CHECK_EQ(bad_deep, 0);
    CHECK(spanned > 0);

    /* One prompt by hand: the first interval alone, then two a chunk, and the last chunk to the
     * tail checkpoint a span of 1 writes (1664) with the rest, under a quantum, after it. */
    int64_t ck = -1;
    CHECK_EQ(two.sch.plan_chunk(0, 1700, 512, &ck), 256);    CHECK_EQ(ck, 256);
    CHECK_EQ(two.sch.plan_chunk(256, 1700, 512, &ck), 512);  CHECK_EQ(ck, 768);
    CHECK_EQ(two.sch.plan_chunk(768, 1700, 512, &ck), 512);  CHECK_EQ(ck, 1280);
    CHECK_EQ(two.sch.plan_chunk(1280, 1700, 512, &ck), 384); CHECK_EQ(ck, 1664);
    CHECK_EQ(two.sch.plan_chunk(1664, 1700, 512, &ck), 36);  CHECK_EQ(ck, -1);
}

/* A span is for a prefill alone in its step. Alone it takes two intervals; while another request
 * is in the engine (admitted, prefilling or decoding) no step is longer than the shared budget, one
 * interval here; alone again it spans again from the next boundary. */
TEST(a_span_is_taken_only_by_a_prefill_alone_in_its_step) {
    Fixture f;
    f.build();                            /* interval 256, budget 512 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().shared_tok, 256);
    const uint64_t a = f.submit(3000, 4);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 256);           /* the first interval */
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);           /* alone: a span */
    f.commit_plain();
    const uint64_t b = f.submit(100, 3);
    int with_b = 0, wide_with_b = 0, spans_after = 0;
    for (int k = 0; k < 40; ++k) {
        if (!f.sch.step()) break;
        const StepPlan& p = f.sch.plan();
        bool has_b = false;
        int64_t a_tok = 0;
        for (const StepEntry& e : p.e) {
            if (e.seq == b) has_b = true;
            if (e.seq == a) a_tok = e.n_tokens;
        }
        if (has_b) {
            ++with_b;
            if (p.n_tok > 256 || a_tok > 256) ++wide_with_b;
        } else if (with_b > 0 && a_tok > 256) {
            ++spans_after;                /* b has finished: a is alone again */
        }
        f.commit_plain();
    }
    CHECK(with_b >= 3);
    CHECK_EQ(wide_with_b, 0);
    CHECK(spans_after >= 1);
}

/* With a shortest spanning prefill set (set_span_min), a shorter prompt alone in the engine takes
 * one interval a step, and a longer one spans through to its last intervals. */
TEST(a_span_min_keeps_a_short_prompt_at_one_interval) {
    Fixture f;
    f.build();                            /* interval 256, budget 512 */
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    f.sch.set_span_min(2048);
    const uint64_t a = f.submit(1500, 2);
    int wide = 0, steps = 0;
    for (int k = 0; k < 40; ++k) {
        if (!f.sch.step()) break;
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == a && e.is_prefill) { ++steps; wide += e.n_tokens > 256; }
        f.commit_plain();
    }
    CHECK(steps >= 5);
    CHECK_EQ(wide, 0);
    const uint64_t b = f.submit(3000, 2);
    int spans = 0, last_pf = 0;
    for (int k = 0; k < 40; ++k) {
        if (!f.sch.step()) break;
        for (const StepEntry& e : f.sch.plan().e)
            if (e.seq == b && e.is_prefill) { spans += e.n_tokens > 256; last_pf = e.n_tokens; }
        f.commit_plain();
    }
    CHECK(spans >= 4);
    CHECK(last_pf > 0);
}

/* A shared step under a span is planned as a deployment without one plans it: with two requests
 * from the start, every step they share is the same, entry for entry, at budget 512 with a span of
 * two as at budget 256 with none, --busy-prefill-chunk's shares included. */
TEST(a_shared_step_under_a_span_is_planned_as_without_one) {
    std::vector<std::vector<std::pair<uint64_t, int32_t>>> plans[2];
    for (int m = 0; m < 2; ++m) {
        Fixture f;
        f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/m == 0 ? 512 : 256);
        if (m == 0) f.cfg.prefill_span = 2;
        CHECK_OK(f.start());
        const uint64_t a = f.submit(3000, 4);
        const uint64_t b = f.submit(300, 6);
        (void)a;
        for (int k = 0; k < 40; ++k) {
            if (!f.sch.step()) break;
            std::vector<std::pair<uint64_t, int32_t>> row;
            bool has_b = false;
            for (const StepEntry& e : f.sch.plan().e) {
                row.push_back({ e.seq, e.n_tokens });
                has_b = has_b || e.seq == b;
            }
            if (!has_b) break;            /* b has finished: from here a is alone */
            plans[m].push_back(row);
            f.commit_plain();
        }
    }
    CHECK(plans[0].size() >= 4);
    CHECK(plans[0] == plans[1]);
}

/* No interval grid, no span: a window snapshot splits no chunk, and a model without checkpoints
 * has no interval. The shared budget still follows the flag, as the engine states it to the plugin
 * (RadBuildCtx::max_mixed_tok), and steps that share it admit as before. */
TEST(a_span_needs_an_interval_grid) {
    Program prog;
    prog.meta.n_layers = 16;
    prog.kv_groups.emplace_back();
    KVGroupInfo attn;
    attn.name = "attn";
    attn.decl.kind = RAD_KV_FULL;
    attn.block_size = 16;
    attn.layers = { 0 };
    prog.kv_groups.push_back(attn);
    KVGroupInfo win;
    win.name = "swa";
    win.decl.kind = RAD_KV_WINDOW;
    win.decl.window = 64;
    win.block_size = 16;
    win.layers = { 1 };
    prog.kv_groups.push_back(win);
    Config cfg;
    cfg.max_tok = 512;
    cfg.checkpoint_interval = 256;
    cfg.prefix_cache = true;
    cfg.prefill_span = 2;
    ChunkGeometry g;
    CHECK_OK(chunk_geometry_resolve(prog, cfg, &g));
    CHECK(g.window_ckpt);
    CHECK_EQ(g.span, 1ll);
    CHECK_EQ(g.shared_tok, 256ll);

    Fixture f;
    f.build();
    f.cfg.checkpoint_interval = 0;
    f.cfg.prefill_span = 2;
    CHECK_OK(f.start());
    CHECK_EQ(f.sch.geometry().span, 1ll);
    f.submit(100, 3);
    f.submit(100, 3);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int64_t)f.sch.plan().e.size(), 2ll);
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
    /* The snapshot is the state after the chunk's last token, which the split put at the interval
     * boundary. */
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
 * changes only at a publication, and a publication carries exactly what the live readers say. */
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

/* The sequences of one API request are added under one lock hold, each accepted or refused on its
 * own: a duplicate id is refused without taking its neighbours with it. */
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

/* A scored prompt (Request::token_scores) names real vocabulary rows, and drafts nothing. */
TEST(a_scored_request_names_real_vocabulary_rows) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int32_t n_vocab = (int32_t)f.prog.meta.n_vocab;
    REQUIRE(n_vocab > 2);
    std::unique_ptr<Request> rs[4];
    const std::vector<std::vector<int32_t>> ids = { { 1, 2 }, {}, { n_vocab }, { -1 } };
    for (int i = 0; i < 4; ++i) {
        rs[i] = std::make_unique<Request>();
        rs[i]->id = f.next_id++;
        rs[i]->max_tokens = 1;
        rs[i]->prompt.assign(16, 7);
        rs[i]->token_scores = std::make_shared<TokenScores>();
        rs[i]->token_scores->tokens = ids[(size_t)i];
    }
    Request* kept = rs[0].get();
    int rc[4] = { -1, -1, -1, -1 };
    f.sch.add_all(rs, 4, rc);
    CHECK_EQ(rc[0], RAD_OK);
    CHECK_EQ(rc[1], RAD_E_INVAL);
    CHECK_EQ(rc[2], RAD_E_INVAL);
    CHECK_EQ(rc[3], RAD_E_INVAL);
    CHECK(!kept->may_speculate);
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
     * (256, 300], because RadBatch carries one query length for every KV group. The chunk still
     * ends on the next interval, so the next checkpoint is still written. */
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    CHECK_EQ(b->ctx_lens[0], 256);
    CHECK_EQ(b->n_tok, 256);
    CHECK_EQ(b->n_checkpoints, 1);
    CHECK_EQ(b->checkpoint_tok[0], 255);

    /* The recurrent half of the hit is carried exactly once. The adopted blocks describe the
     * attention layers; the linear ones resume from this slot alone, so an entry that skipped 256
     * tokens without naming a snapshot would continue from whatever the slot last held. */
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, 3);

    /* The hit is claimed, not merely counted: the sequence adopted the cache's 16 blocks, which
     * cost the pool nothing since the cache already held them, then allocated only the 16 new
     * blocks its 256-token chunk needs. An unadopted hit would attend to blocks it does not own. */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 32);
    CHECK_EQ(f.kv.block_table(a, 0)[0], f.pc.seeded[0][0]);
    CHECK_EQ(f.kv.free_blocks(0), free_before - 16);

    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.prompt_tokens, 1000);
    CHECK_EQ(m.attn_cached_tokens, 304);     /* what the block cache matched */
    CHECK_EQ(m.linear_cached_tokens, 256);   /* what was actually skipped */
    CHECK_EQ(m.checkpoints_written, 1);
    /* The checkpoint shortfall: the 48 cached tokens above the snapshot are computed again. */
    CHECK_EQ(m.hit_cut_tokens, 48);
    CHECK_EQ(m.hit_cut_no_snapshot, 0);

    /* The snapshot this step takes is not offered to anyone until the step has run: the
     * reservation happens when the chunk is planned, the state exists only afterwards. */
    CHECK(f.pc.committed.empty());
    f.commit_plain();
    CHECK_EQ((int)f.pc.committed.size(), 1);
    CHECK_EQ(f.pc.committed[0], 512);        /* ctx_len 256 + ckpt_tok 255 + 1 */
}

/* A hit that skipped tokens with no snapshot to resume from would corrupt the linear state, so it
 * is refused outright rather than served with a zeroed state. */
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
    /* Counted once, with its reason and the whole hit as the shortfall. */
    CHECK_EQ(f.sch.metrics().hit_cut_no_snapshot, 1);
    CHECK_EQ(f.sch.metrics().hit_cut_tokens, 304);
}

/* A restore from a slot the store gave back is never built. The cache keeps a handed-out snapshot
 * until the restoring step has run, so this is a guard: the step is abandoned, the request gives
 * its hit back and requeues, and the next step prefills it from a fresh lookup. */
TEST(a_restore_from_a_slot_the_store_gave_back_is_prefilled_again) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.pc.seed(304);
    f.pc.lin = 256;
    f.pc.src = 3;
    f.kv.ck_gone = {3};
    const int64_t free_before = f.kv.free_blocks(0);

    const uint64_t a = f.submit(1000, 8);
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.kv.free_blocks(0), free_before);      /* the adopted blocks went back */

    /* The next lookup finds no snapshot, and the request runs from zero. */
    f.pc.src = -1;
    f.kv.ck_gone.clear();
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ(b->ctx_lens[0], 0);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, -1);
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

/* A hit is refused on a model with a paged group the prefix cache does not index. Here the
 * windowed group is paged, bound to a trunk layer, and skipped by the cache (prefix.cpp recycles
 * its blocks below the window), which pc.groups = {0} reproduces. The hit is deep and
 * checkpoint-covered, so every other clamp passes it. Adopting it would cover group 0 only, while
 * the windowed trunk layers read positions below ctx_len this sequence never wrote: the previous
 * request's KV, read as this one's prefix. No tail recompute repairs a trunk hole. */
TEST(a_hit_on_a_model_with_a_windowed_group_reuses_nothing) {
    Fixture f;
    f.build(4096, 64, 512, 4096, /*windowed=*/true);
    f.pc.groups = {0};    /* a cache over the FULL group only (windowed groups are not
                           * prefix-cacheable); must precede start(), which reads it once */
    CHECK_OK(f.start());
    f.pc.seed(304);       /* a deep, block-aligned attention hit: 19 blocks */
    f.pc.lin = 256;       /* its linear state checkpoints at 256: the resume point exists */
    f.pc.src = 3;         /* and names a snapshot, so no checkpoint clamp applies */

    uint64_t a = f.submit(300, 8);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);

    /* The whole hit is refused, not trimmed: the request runs from zero. */
    CHECK_EQ(f.sch.find(a)->n_computed, 0);
    CHECK_EQ(f.sch.find(a)->n_cached, 0);
    CHECK_EQ(b->ctx_lens[0], 0);
    CHECK_EQ(f.sch.plan().e[0].ckpt_restore, -1);

    /* Nothing is adopted: the 16 blocks covering the 256-token chunk are all freshly grabbed. */
    CHECK_EQ((int)f.kv.block_table(a, 0).size(), 16);
    CHECK(std::find(f.pc.seeded[0].begin(), f.pc.seeded[0].end(),
                    f.kv.block_table(a, 0)[0]) == f.pc.seeded[0].end());

    /* The cache still reported the match, and the metrics say so: this is a real hit the
     * scheduler refused, not a lookup that found nothing. */
    SchedMetrics m = f.sch.metrics();
    CHECK_EQ(m.attn_cached_tokens, 304);
    CHECK_EQ(m.linear_cached_tokens, 0);
}

/* A request whose media is not encoded yet waits, and only it does: it is not admitted without
 * its rows, and the request behind it is asked and runs. Once its items are encoded it is
 * admitted where it stood. */
TEST(a_request_waiting_on_its_encoder_lets_the_queue_past_it) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    std::shared_ptr<mm::Item> ia, ic;
    const uint64_t a = f.submit_unencoded(160, 8, &ia);
    const uint64_t b = f.submit(160, 8);
    const uint64_t c = f.submit_unencoded(160, 8, &ic);

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

    /* Asked again every step, and admitted the step its items are in. */
    ia->encoded = true;
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->n_seq, 2);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Waiting);
    f.commit_plain();

    ic->encoded = true;
    const RadBatch* s3 = f.sch.step();
    CHECK(s3 != nullptr);
    CHECK_EQ(s3->n_seq, 3);
    CHECK_EQ((int)f.sch.find(c)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.queue_snapshot(q, 8), 0);
    CHECK_EQ(f.sch.metrics().admitted, 3);
}

/* ================================================================== preemption */
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

    /* Both now decode at position 160, a block boundary: the next token needs a new block and
     * the pool has none. The victim must be A (lowest priority, and re-admitted last anyway),
     * not the head-of-line B. */
    const RadBatch* s2 = f.sch.step();
    CHECK(s2 != nullptr);
    CHECK_EQ(s2->n_seq, 1);

    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.find(a)->n_computed, 0);      /* preemption is by recompute */
    CHECK_EQ(f.sch.find(a)->state_slot, -1);

    /* Requeued to the front: it keeps its original arrival, so it sits ahead of C, which arrived
     * later at the same priority. */
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
    /* A ceiling on a coarse grid, not the exact longest context: 100 rounds up to the 1024-token
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
    /* Block tables are one row per sequence, pitched from the step's context bound: 1024 tokens
     * plus the longest query (64) and the draft lookahead (2) is 1090 tokens, 69 blocks of 16,
     * rounded up to the bound's 64-block grid: 128. Columns past a row's own blocks read -1. */
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

    /* Chunk bounds never cross a sequence and split on absolute multiples of the chunk length.
     * The decode row spans 100..100 and the prefill row 0..63, so neither crosses 64: one chunk
     * each, boundaries at 0, 1, 65. */
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

/* RadBatch::lora_slot: every row of a sequence carries its request's runtime LoRA slot, -1 for a
 * base request, in the step's row order (decode rows first). A step with no adapter row carries
 * none, so the core issues no adapter op on it. */
TEST(the_step_batch_carries_each_rows_adapter_slot) {
    Fixture f;
    f.build();
    f.cfg.lora.max_loras = 2;
    CHECK_OK(f.start());

    f.submit(16, 1);
    const RadBatch* base = f.sch.step();
    CHECK(base != nullptr);
    CHECK_EQ(base->n_tok, (int64_t)16);
    CHECK(base->lora_slot == nullptr);
    f.commit_plain();

    auto r = f.request(100, 64);
    r->lora_slot = 1;
    r->lora_uid = 7;
    CHECK(f.add(std::move(r)) != 0);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK(b->lora_slot != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)100);
    for (int64_t i = 0; i < b->n_tok; ++i) CHECK_EQ(b->lora_slot[i], 1);
    f.commit_plain();

    f.submit(64, 64);
    b = f.sch.step();
    CHECK(b != nullptr);
    CHECK_EQ(b->n_tok, (int64_t)65);
    CHECK_EQ(b->lora_slot[0], 1);                          /* the adapter request's decode row */
    for (int64_t i = 1; i < 65; ++i) CHECK_EQ(b->lora_slot[i], -1);   /* the base prefill */
}

/* With no adapters served the field is null and the step slab is byte for byte what it was before
 * the field existed: the slot array is the only thing adapters add to it. */
TEST(no_adapters_no_slot_array) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.submit(32, 8);
    const RadBatch* b = f.sch.step();
    CHECK(b != nullptr);
    CHECK(b->lora_slot == nullptr);

    Config on = f.cfg;
    on.lora.max_loras = 2;
    const int64_t off_bytes = BatchBuilder::device_bytes(f.prog, f.cfg, f.kv.geom);
    const int64_t on_bytes  = BatchBuilder::device_bytes(f.prog, on, f.kv.geom);
    CHECK(off_bytes > 0);
    CHECK_EQ(on_bytes - off_bytes, (f.cfg.max_tok * 4 + 255) / 256 * 256);
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

    /* Freed now, not at completion or at the next step: a cancelled request that kept its pool
     * share until it would have finished would shrink the pool under exactly the load that
     * produces cancellations. */
    CHECK_OK(f.sch.cancel(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.sch.metrics().running, 0);
    CHECK_EQ(f.sch.metrics().cancelled, 1);
    CHECK_OK(f.sch.reap(a));

    /* The one exception is a step still on the device, and it defers by one step, not one
     * request: releasing a block a running kernel is writing hands it to a new sequence
     * mid-step, which shows up as a quality problem rather than a crash. */
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);
    CHECK_OK(f.sch.cancel(b));
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.sch.metrics().running, 0);          /* out of the running set at once */
    CHECK_EQ(f.kv.free_blocks(0), before - 10);    /* but the blocks are still the device's */
    f.commit_plain();
    CHECK_EQ(f.kv.free_blocks(0), before);
    /* The commit must not append the token the step sampled for a request cancelled while that
     * step was running. */
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Cancelled);
}

/* A reap while the step is still on the device. Common: the server reaps as soon as a sink
 * finishes, and cancelling finishes it, so every mid-decode disconnect lands here. Releasing the
 * Request now would destroy it under commit(); returning its slot now would let a newly admitted
 * request take the entry and receive another client's sampled token. */
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

    /* The Request and its slot outlive the reap by exactly one commit: neither the id nor the
     * blocks are released while a StepEntry still points at them. */
    CHECK(f.sch.find(a) != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ(f.kv.free_blocks(0), before - 10);

    /* A request admitted in that window must not be given the slot the open step is holding. */
    uint64_t b = f.submit(160, 64);
    CHECK(f.sch.find(b) != nullptr);

    f.commit_plain();
    CHECK(f.sch.find(a) == nullptr);              /* released now, and exactly once */
    CHECK_EQ(f.kv.free_blocks(0), before);

    /* The request admitted in the window is untouched by the commit that reaped the other. */
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

/* ============================================ a block drafter's own cache across a prefix hit */
TEST(a_prefix_hit_does_not_stop_a_full_attention_drafter) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 7;                 /* copied into the scheduler at init, so it goes first */
    CHECK_OK(f.start());

    /* A real hit, both halves: the attention blocks and the linear checkpoint must reach the same
     * depth, or the scheduler recomputes from 0, ctx_len is 0 and the test passes vacuously. */
    f.pc.seed(256);
    f.pc.lin = 256;
    f.pc.src = 3;

    uint64_t a = f.submit(272, 64);
    CHECK(f.sch.step() != nullptr);                    /* recomputes 0..256 off the checkpoint */
    CHECK_EQ(f.sch.find(a)->n_computed, 256);
    f.commit_plain();
    CHECK(f.sch.step() != nullptr);                    /* 256..272, and this one yields a token */
    CHECK_EQ(f.sch.find(a)->n_computed, 272);
    /* Non-zero is the point: this is the floor the guard would take after a hit. */
    CHECK(f.sch.plan().e[0].ctx_len > 0);
    f.commit_plain(11);

    /* Window 0 is a drafter attending over its whole history, so it can never wait for the window
     * to slide past the hit. Every paged group here is prefix-cached, so the adopted blocks were
     * written by the previous sequence's own context pass: no hole, floor 0. A windowed drafter
     * whose group the cache skips is the draft-refill cases below. */
    Scheduler::BlockRow rows[4];
    CHECK_EQ(f.sch.block_begin(/*block=*/8, /*depth=*/7, /*window=*/0, /*mask=*/99, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)a);
}

/* ====================================== a prefix hit and a windowed block drafter (draft refill)
 * The drafter's own windowed KV group is not prefix-cached, so a hit leaves it without K and V for
 * the cached tokens. The scheduler ends such a hit one drafter window before the prompt's end, at
 * a checkpoint, so the recompute covers the window and the drafter drafts from the first decode. */
namespace {
struct RefillRun {
    int64_t n_after_first = 0;     /* n_computed after the admitting step */
    int rows = -1;                 /* drafting rows block_begin offers at the first decode step */
    SchedMetrics m;
};
RefillRun refill_run(int n_prompt, int32_t seeded, int64_t ckpt_lo = 0) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
    f.cfg.n_spec = 7;
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    f.prog.drafter.window = 512;
    KVGroupInfo w;                 /* the drafter's own group: paged, windowed, not prefix-cached,
                                    * unbound. The trunk clamp reads bound trunk layers only
                                    * (scheduler.h, hit_holes_trunk_), so it leaves this hole to
                                    * the refill. */
    w.name = "draft";
    w.decl.kind = RAD_KV_WINDOW;
    w.decl.window = 512;
    w.block_size = 16;
    f.prog.kv_groups.push_back(w);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0 };
    f.pc.ckpt_lo = ckpt_lo;
    CHECK_OK(f.start());
    if (seeded > 0) { f.pc.seed(seeded); f.pc.lin = seeded; f.pc.src = 3; }
    RefillRun r;
    const uint64_t a = f.submit(n_prompt, 64);
    Scheduler::BlockRow rows[4];
    for (int st = 0; st < 40; ++st) {
        CHECK(f.sch.step() != nullptr);
        if (st == 0) r.n_after_first = f.sch.find(a)->n_computed;
        f.commit_plain(11);
        const int n = f.sch.block_begin(8, 7, 512, 99, rows, 4);
        f.sch.block_end();
        if (!f.sch.find(a)->output.empty()) { r.rows = n; break; }
    }
    r.m = f.sch.metrics();
    return r;
}
}  /* namespace */

TEST(a_prefix_hit_ends_a_drafter_window_before_the_prompt_and_the_drafter_drafts_at_once) {
    const RefillRun r = refill_run(/*prompt=*/1100, /*seeded=*/1024);
    /* The cap is 1100 - 512 = 588: attention 576, and the checkpoint under it 512. */
    CHECK_EQ(r.n_after_first, 512);
    CHECK_EQ(r.rows, 1);
    CHECK_EQ(r.m.draft_refills, (int64_t)1);
    CHECK_EQ(r.m.draft_refill_tokens, (int64_t)512);
    CHECK_EQ(r.m.linear_cached_tokens, (int64_t)512);    /* what was actually skipped */
}

TEST(a_prefix_hit_already_a_window_back_is_kept_whole) {
    /* 5000 - 512 is past the 1024-token hit, so the drafter's window is recomputed anyway. */
    const RefillRun r = refill_run(5000, 1024);
    CHECK_EQ(r.n_after_first, 1024);
    CHECK_EQ(r.m.draft_refills, (int64_t)0);
    CHECK_EQ(r.rows, 1);
}

TEST(a_prefix_hit_with_no_checkpoint_near_the_cap_is_kept_whole) {
    /* No checkpoint below 2048: the cap would give back all 2048 tokens, more than two windows,
     * so the whole hit stands and the drafter waits for its window as it would without a cap. */
    const RefillRun r = refill_run(2200, 2048, /*ckpt_lo=*/2048);
    CHECK_EQ(r.m.draft_refills, (int64_t)0);
    CHECK_EQ(r.m.linear_cached_tokens, (int64_t)2048);
    CHECK_EQ(r.rows, 0);
}

TEST(no_prefix_hit_gives_nothing_back) {
    const RefillRun r = refill_run(1100, 0);
    CHECK_EQ(r.m.draft_refills, (int64_t)0);
    CHECK_EQ(r.rows, 1);
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
    CHECK_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_OK(f.commit(99, 0, [](const StepEntry&) { return 2; }));

    const Request* ra = f.sch.find(a);
    /* Three tokens committed: the two accepted drafts and the bonus. */
    CHECK_EQ((int)ra->output.size(), 4);
    CHECK_EQ(ra->output[1], 21);
    CHECK_EQ(ra->output[2], 22);
    CHECK_EQ(ra->output[3], 99);
    CHECK_EQ(ra->n_computed, 163);
    CHECK_EQ(ra->n_accepted, 2);
    /* Paged KV gave back blocks for the rejected tail; the state slot did not move, since a linear
     * rollback is a read offset, not a free. */
    CHECK(f.kv.free_blocks(0) >= blocks_before);
    CHECK(f.kv.state_slot(a, 1) >= 0);

    /* The conv cursor advanced by 1 + accepted, modulo the rolling window. */
    f.submit(1, 1);   /* only to force a plan rebuild below */
    const RadBatch* b2 = f.sch.step();
    CHECK(b2 != nullptr);
    int32_t W = 4 - 1 + 4 + 1;
    CHECK_EQ(f.sch.plan().e[0].conv_cursor, (cursor_before + 3) % W);
}

/* THE COUPLED BIND AFTER A REAL COMMIT.
 *
 * Engine::finish_serial_drafter binds each round at the request's committed output length, read
 * from the plan after commit -- the value the next Sampler::build_step stages that row at. This
 * drives the same rule (sched/mtp_couple_bind.h) against the real Scheduler::commit: a live
 * commit, a max_tokens cut, and a card reused by a new request afterwards.
 *
 * The old engine formula (the draw's pre-commit base + the slot's one-based n_acc) agrees on every
 * commit of a request that is still running, and drifts by exactly the tokens a cut withheld. A
 * commit that emits fewer than the card accepted has closed the request; its plan entry is carried
 * for the bind, but the request will never sample again. Binding it would put a coupled verify on
 * a terminal row, which reads as a live round in the log without being one. */
TEST(the_coupled_bind_skips_a_request_the_commit_closed) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 3;
    CHECK_OK(f.start());
    const int64_t vocab = f.prog.meta.n_vocab;

    int32_t draft[3] = { 21, 22, 23 };
    auto readback = [&](uint64_t seq, uint64_t base, int rounds) {
        RadMtpCoupleDraw d{};
        d.seq  = seq;
        d.base = base;
        d.mode = 1;
        std::vector<RadMtpCoupleState> rb((size_t)rounds);
        for (int rd = 1; rd <= rounds; ++rd) {
            RadMtpCoupleState& s = rb[(size_t)(rd - 1)];
            s.draw   = d;
            s.round  = rd;
            s.status = 1;
            s.count  = 1;
            s.x      = draft[rd - 1];       /* what the card holds for that round */
            for (int i = 0; i < 8; ++i) s.ids[i] = -1;
            s.ids[0] = s.x;
            s.q[0]   = 1.0;
        }
        return rb;
    };
    auto slot = [&]() {
        StepSlot S{};
        S.acc = 3;
        S.n_acc = 4;
        for (int rd = 1; rd <= 3; ++rd) S.tok[rd] = draft[rd - 1];
        return S;
    };

    /* ---- a live commit: three drafts accepted, the request keeps running. */
    const uint64_t a = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    const uint64_t a_base = f.sch.find(a)->output.size();   /* the draw's pre-commit coordinate */
    CHECK_OK(f.commit(9, 0, [](const StepEntry&) { return 3; }));
    REQUIRE(f.sch.find(a) != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(a)->output.size(), (int)a_base + 4);

    MtpCoupleBindPoint pa;
    pa.found = f.sch.committed_seq(a, &pa.n_out, &pa.live);
    CHECK(pa.found);
    CHECK(pa.live);
    CHECK_EQ(pa.n_out, (uint64_t)f.sch.find(a)->output.size());

    const std::vector<RadMtpCoupleState> rba = readback(a, a_base, 3);
    const StepSlot Sa = slot();
    RadMtpCoupleState bound[RAD_MTP_COUPLE_MAX_ROUNDS]{};
    int n_bound = -1;
    std::string why;
    CHECK_EQ(mtp_couple_bind_rounds(pa, rba[0].draw, rba.data(), 1, 3, Sa, vocab, bound,
                                    &n_bound, &why), RAD_OK);
    CHECK_EQ(n_bound, 3);
    for (int k = 0; k < 3; ++k) {
        /* The next build stages row k at n_out + k. */
        CHECK_EQ(bound[k].pos, pa.n_out + (uint64_t)k);
        CHECK(rad_mtp_couple_binds(bound[k], a, rba[0].draw.seed, pa.n_out, k, draft[k], vocab));
    }
    /* The old formula on the same data: identical, because this commit emitted its whole
     * acceptance. */
    for (int k = 0; k < 3; ++k)
        CHECK_EQ(a_base + 4 + (uint64_t)k, pa.n_out + (uint64_t)k);

    /* ---- a cut commit: the card accepted three drafts, the budget lets two through. */
    const uint64_t b = f.submit(160, /*max_tokens=*/3);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);                                     /* b's first output token */
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    CHECK_OK(f.sch.set_draft(b, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    const uint64_t b_base = f.sch.find(b)->output.size();
    CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return e.seq == b ? 3 : 0; }));
    REQUIRE(f.sch.find(b) != nullptr);
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Finished);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 3);          /* 1 + 2, not 1 + 4 */

    /* The commit snapshot still names b -- the terminal step is carried into the bind -- but it is
     * not live: nothing binds, so no coupled verify is staged for a request that is gone. */
    MtpCoupleBindPoint pb;
    pb.found = f.sch.committed_seq(b, &pb.n_out, &pb.live);
    CHECK(pb.found);
    CHECK(!pb.live);
    CHECK_EQ(pb.n_out, (uint64_t)f.sch.find(b)->output.size());

    const std::vector<RadMtpCoupleState> rbb = readback(b, b_base, 3);
    const StepSlot Sb = slot();
    CHECK_EQ(mtp_couple_bind_rounds(pb, rbb[0].draw, rbb.data(), 1, 3, Sb, vocab, bound,
                                    &n_bound, &why), RAD_OK);
    CHECK_EQ(n_bound, 0);
    /* THE OLD FORMULA would have bound b's rounds at base + n_acc + k -- two past the length it
     * actually committed, on a row no build will stage. */
    for (int k = 0; k < 3; ++k) {
        const uint64_t old_pos = b_base + 4 + (uint64_t)k;
        CHECK(old_pos != pb.n_out + (uint64_t)k);
        RadMtpCoupleState stale = rbb[(size_t)k];
        stale.pos = old_pos;
        CHECK(!rad_mtp_couple_binds(stale, b, rbb[0].draw.seed, pb.n_out, k, draft[k], vocab));
    }

    /* ---- reuse: the next request on a card binds at its own length, and the closed request's
     * rounds are not accepted by it. */
    const uint64_t c = f.submit(160, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(c, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    const uint64_t c_base = f.sch.find(c)->output.size();
    CHECK_OK(f.commit(9, 0, [](const StepEntry&) { return 3; }));
    MtpCoupleBindPoint pc;
    pc.found = f.sch.committed_seq(c, &pc.n_out, &pc.live);
    CHECK(pc.found);
    CHECK(pc.live);
    CHECK_EQ(pc.n_out, (uint64_t)f.sch.find(c)->output.size());
    CHECK_EQ(pc.n_out, c_base + 4);
    CHECK(!rad_mtp_couple_binds(rbb[0], c, rbb[0].draw.seed, pc.n_out, 0, draft[0], vocab));
    RadMtpCoupleState own = readback(c, c_base, 1)[0];
    own.pos = rad_mtp_couple_verify_pos(pc.n_out, 1);
    CHECK(rad_mtp_couple_binds(own, c, own.draw.seed, pc.n_out, 0, draft[0], vocab));
}

/* THE BIND POINT OUTLIVES A REAP.
 *
 * commit() retires the step, and a request it finished that the next step does not carry is pinned
 * by nothing any more. The server calls reap() from an HTTP thread as soon as it sees the finish,
 * and reap() destroys the Request -- while Engine::finish_serial_drafter still has to bind that
 * row. The bind point is therefore copied by commit under the scheduler lock
 * (Scheduler::committed_seq), not read off the plan's StepEntry::req, which is dangling by then.
 * This reaps a non-carried terminal request between commit and bind: the plan-pointer version
 * could only answer by reading freed memory, and the snapshot answers from the copy.
 */
TEST(the_committed_bind_point_outlives_a_reap) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 3;
    CHECK_OK(f.start());
    const int64_t vocab = f.prog.meta.n_vocab;

    int32_t draft[3] = { 21, 22, 23 };
    const uint64_t a = f.submit(160, 64);                  /* live through the commit */
    const uint64_t b = f.submit(160, /*max_tokens=*/3);    /* closed on it by the cut */
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(a, draft, 3));
    CHECK_OK(f.sch.set_draft(b, draft, 3));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return e.seq == b ? 3 : 0; }));
    CHECK_EQ((int)f.sch.find(b)->state, (int)ReqState::Finished);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);

    /* The server's reap lands between commit and bind: b's Request is destroyed here. */
    CHECK_OK(f.sch.reap(b));
    CHECK(f.sch.find(b) == nullptr);

    MtpCoupleBindPoint pb;
    pb.found = f.sch.committed_seq(b, &pb.n_out, &pb.live);
    CHECK(pb.found);
    CHECK(!pb.live);                                       /* terminal: nothing binds */
    CHECK_EQ(pb.n_out, (uint64_t)3);

    MtpCoupleBindPoint pa;
    pa.found = f.sch.committed_seq(a, &pa.n_out, &pa.live);
    CHECK(pa.found);
    CHECK(pa.live);
    CHECK_EQ(pa.n_out, (uint64_t)f.sch.find(a)->output.size());

    RadMtpCoupleState rb[RAD_MTP_COUPLE_MAX_ROUNDS]{};
    RadMtpCoupleState out[RAD_MTP_COUPLE_MAX_ROUNDS]{};
    StepSlot S{};
    S.acc = 3;
    S.n_acc = 4;
    for (int rd = 1; rd <= 3; ++rd) S.tok[rd] = draft[rd - 1];
    int n_bound = -1;
    std::string why;
    CHECK_EQ(mtp_couple_bind_rounds(pb, RadMtpCoupleDraw{}, rb, 1, 3, S, vocab, out, &n_bound,
                                    &why), RAD_OK);
    CHECK_EQ(n_bound, 0);
}

/* A decode benchmark needs the request to last as long as asked, not as long as the model
 * chooses. EOS still stops an ordinary request, and max_tokens still stops one ignoring EOS. */
TEST(ignore_eos_runs_to_max_tokens_and_nothing_else_changes) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());

    uint64_t plain = f.submit(8, /*max_tokens=*/4);
    uint64_t keep  = f.submit(8, /*max_tokens=*/4, /*priority=*/0, /*ignore_eos=*/true);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(7);                  /* both prefilled, one token each */

    /* Every sequence emits end-of-generation on this step. */
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK(f.sch.step() != nullptr);
    CHECK_OK(f.commit(9, 1));

    CHECK_EQ((int)f.sch.find(plain)->state, (int)ReqState::Finished);
    CHECK_EQ(f.sch.find(plain)->finish_reason, std::string("stop"));
    CHECK_EQ((int)f.sch.find(keep)->state, (int)ReqState::Running);

    /* It ends on the budget instead, and says so. */
    while (f.sch.find(keep)->state == ReqState::Running && f.sch.step() != nullptr)
        CHECK_OK(f.commit(9, 1));
    const Request* rk = f.sch.find(keep);
    CHECK_EQ((int)rk->state, (int)ReqState::Finished);
    CHECK_EQ(rk->finish_reason, std::string("length"));
    CHECK_EQ((int)rk->output.size(), 4);
}

/* Decode rows come first in a mixed step: a block that treats the two kinds differently needs each
 * as a contiguous run, since cu_seqlens is read as cu[n] and cu[n+1]. `running_` is walked in
 * priority order, so without the sort a high-priority prefill lands ahead of a decoding row. */
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

    /* The entry index followed the row. `entry` is how a request finds its own result, so a sort
     * that leaves it behind hands one request another's token. */
    f.commit_plain(12);
    CHECK_EQ((int)f.sch.find(a)->output.size(), 2);    /* 11 from the prefill, 12 from the decode */
    CHECK_EQ(f.sch.find(a)->output[1], 12);
    CHECK_EQ((int)f.sch.find(b)->output.size(), 0);    /* still prefilling: no token yet */
}

/* A step carrying a prefill chunk still takes a draft: the blocks issue the recurrent path over
 * the decode half, because the chunked path has no num_accepted operand to roll a rejection back
 * with. These are the two shapes a refusal would fire on. */
TEST(a_queued_request_does_not_suppress_the_draft) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    f.cfg.max_seqs = 3;                      /* a slot is free, so the queue will be admitted */
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
    /* Two decode rows of 1 + 4, so the blocks split after 10 tokens and then the chunk. A wrong
     * split lets a per-token op reach into the other half. */
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

/* ================================================================== a prefill sharing a step */
/* --busy-prefill-chunk. The starving shape: a checkpoint interval as large as the step budget,
 * so a long prefill's chunk can take the whole step every step. Here 512 and 512, quantum 64; the
 * sequence is long enough for many full chunks. */
static void share_fixture(Fixture& f, int64_t mode) {
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/4096);
    f.cfg.checkpoint_interval = 512;
    f.cfg.busy_prefill_chunk = mode;
}

/* With `off`, once the long prefill owns the step a newly arriving request is not even admitted
 * until it ends. */
TEST(off_a_long_prefill_keeps_a_new_request_out_of_every_step) {
    Fixture f;
    share_fixture(f, /*off=*/0);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);
    f.commit_plain();
    uint64_t b = f.submit(100, 64);
    for (int k = 0; k < 4; ++k) {
        CHECK(f.sch.step() != nullptr);
        CHECK_EQ((int)f.sch.plan().e.size(), 1);
        CHECK_EQ(f.sch.plan().e[0].seq, a);
        CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);
        f.commit_plain();
    }
    CHECK_EQ(f.sch.metrics().queue_depth, 1);
    CHECK(f.sch.find(b) != nullptr);
    CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 0);
}

/* auto: the same arrival is admitted in the next step, and its prompt finishes in two. */
TEST(auto_a_request_arriving_during_a_long_prefill_is_admitted_at_once) {
    Fixture f;
    share_fixture(f, /*auto=*/-1);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);       /* alone: the whole step */
    f.commit_plain();
    uint64_t b = f.submit(100, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.plan().e.size(), 2);
    CHECK_EQ(f.sch.metrics().queue_depth, 0);
    int64_t a_tok = 0, b_tok = 0;
    for (const StepEntry& e : f.sch.plan().e) {
        if (e.seq == a) a_tok = e.n_tokens;
        if (e.seq == b) b_tok = e.n_tokens;
    }
    CHECK(a_tok >= 64 && a_tok % 64 == 0);
    CHECK(b_tok > 0);
    CHECK(a_tok + b_tok <= 512);
    CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 1);
    f.commit_plain(11);
    /* b's 100 tokens are done by the next step and it decodes beside the prefill from then on */
    for (int k = 0; k < 4; ++k) {
        CHECK(f.sch.step() != nullptr);
        bool saw_b = false;
        for (const StepEntry& e : f.sch.plan().e) if (e.seq == b) saw_b = true;
        CHECK(saw_b);
        f.commit_plain(12);
    }
    CHECK((int)f.sch.find(b)->output.size() >= 2);
}

/* A decode row behind a long prefill in running_ (the prefill outranks it) is planned every step
 * under auto; under off it is left out of every step that begins on an interval boundary. */
static int decode_rows_seen(int64_t mode) {
    Fixture f;
    share_fixture(f, mode);
    CHECK_OK(f.start());
    uint64_t b = f.submit(100, 512, /*priority=*/0);
    CHECK(f.sch.step() != nullptr);            /* b's whole prompt: it decodes from the next step */
    f.commit_plain(11);
    f.submit(3500, 64, /*priority=*/1);        /* outranks b: walked first */
    int seen = 0;
    for (int k = 0; k < 8; ++k) {
        CHECK(f.sch.step() != nullptr);
        for (const StepEntry& e : f.sch.plan().e) if (e.seq == b && !e.is_prefill) ++seen;
        f.commit_plain(12);
    }
    return seen;
}

TEST(auto_a_decode_row_behind_a_long_prefill_gets_every_step) {
    CHECK_EQ(decode_rows_seen(-1), 8);
    CHECK(decode_rows_seen(0) < 8);
}

/* With nothing else waiting the cap does not apply, and a lone prefill is planned chunk for chunk
 * the same under both modes, which keeps every single-request gate unchanged. */
TEST(auto_a_lone_prefill_is_planned_as_before) {
    std::vector<int32_t> plans[2];
    for (int m = 0; m < 2; ++m) {
        Fixture f;
        share_fixture(f, m == 0 ? 0 : -1);
        CHECK_OK(f.start());
        f.submit(3000, 4);
        for (int k = 0; k < 10; ++k) {
            if (!f.sch.step()) break;
            plans[m].push_back(f.sch.plan().e[0].n_tokens);
            f.commit_plain();
        }
        CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 0);
    }
    CHECK(plans[0] == plans[1]);
}

/* An explicit N holds the chunk to N while something waits. */
TEST(an_explicit_busy_prefill_chunk_holds_the_chunk_to_it) {
    Fixture f;
    share_fixture(f, 128);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);       /* alone */
    f.commit_plain();
    f.submit(100, 64);
    CHECK(f.sch.step() != nullptr);
    for (const StepEntry& e : f.sch.plan().e)
        if (e.seq == a) CHECK_EQ(e.n_tokens, 128);
}

/* ================================================================== decode-only steps between chunks */
/* --prefill-interleave. A sequence decodes while a 2000-token prompt prefills in 256-token chunks
 * (the checkpoint interval). Each step's shape is 'C' when it carries a chunk of the prompt and
 * 'D' when it carries decode rows only; `staged` records prefill_continues() after each step,
 * which arms the prefill stager for the next one. */
struct Interleaved {
    std::string shape;
    std::string staged;
    int64_t steps_held = 0;
};

static Interleaved interleave_run(int64_t n, bool decoder, int steps) {
    Fixture f;
    f.build();
    f.cfg.prefill_interleave = n;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(64);
    if (decoder) {
        f.submit(100, 512);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
    }
    const uint64_t p = f.submit(2000, 4);
    Interleaved out;
    for (int k = 0; k < steps; ++k) {
        if (!f.sch.step()) break;
        bool chunk = false;
        for (const StepEntry& e : f.sch.plan().e) if (e.seq == p && e.is_prefill) chunk = true;
        out.shape += chunk ? 'C' : 'D';
        out.staged += f.sch.prefill_continues() ? '+' : '-';
        f.commit_plain(12);
    }
    out.steps_held = f.sch.metrics().steps_held;
    return out;
}

/* Between a commit and the next step (where an encoder pass is prepared) the committed step's
 * plan names a request the scheduler may already have released: a client cancelled it and the
 * server reaped it. prefill_continues() reads only an open step's plan (#127). */
TEST(prefill_continues_reads_no_committed_plan) {
    Fixture f;
    f.build();
    CHECK_OK(f.start());
    f.sch.set_stage_rows(64);
    const uint64_t p = f.submit(2000, 4);
    CHECK(f.sch.step() != nullptr);
    CHECK(f.sch.prefill_continues());
    f.commit_plain(12);
    CHECK_OK(f.sch.cancel(p));
    CHECK_OK(f.sch.reap(p));
    CHECK(!f.sch.prefill_continues());
}

TEST(interleave_off_puts_a_chunk_in_every_step) {
    const Interleaved r = interleave_run(0, /*decoder=*/true, 8);
    CHECK_EQ(r.shape, std::string("CCCCCCCC"));
    CHECK_EQ(r.staged, std::string("+++++++-"));      /* the eighth chunk is the last */
    CHECK_EQ(r.steps_held, 0);
}

/* N decode-only steps after each chunk, which admit nothing, and the stager is not armed for a
 * chunk the next step will hold: its staged experts would be dropped. */
TEST(interleave_runs_n_decode_only_steps_after_each_chunk) {
    const Interleaved r = interleave_run(2, /*decoder=*/true, 10);
    CHECK_EQ(r.shape, std::string("CDDCDDCDDC"));
    CHECK_EQ(r.staged, std::string("----------"));
    CHECK_EQ(r.steps_held, 6);
}

/* Nothing decodes beside a lone prefill, so nothing is held. */
TEST(interleave_leaves_a_lone_prefill_alone) {
    const Interleaved r = interleave_run(2, /*decoder=*/false, 8);
    CHECK_EQ(r.shape, std::string("CCCCCCCC"));
    CHECK_EQ(r.steps_held, 0);
}

/* After a prompt's last chunk there is nothing to interleave: the next request is admitted in the
 * next step. */
TEST(interleave_does_not_hold_after_a_prompts_last_chunk) {
    Fixture f;
    f.build();
    f.cfg.prefill_interleave = 2;
    CHECK_OK(f.start());
    f.submit(100, 512);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    f.submit(200, 4);                      /* one chunk, beside the decoder */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().phase, RAD_PHASE_MIXED);
    f.commit_plain(12);
    const uint64_t q = f.submit(300, 4);
    CHECK(f.sch.step() != nullptr);
    bool admitted = false;
    for (const StepEntry& e : f.sch.plan().e) if (e.seq == q && e.is_prefill) admitted = true;
    CHECK(admitted);
    CHECK_EQ(f.sch.metrics().steps_held, 0);
}

/* A decode-only step of the hold may be issued ahead; it counts toward N and as a held step, and
 * once the hold is spent the next step is a full one that carries the chunk. */
TEST(interleave_counts_steps_issued_ahead_and_then_serves_the_chunk) {
    Fixture f;
    f.build();
    f.cfg.prefill_interleave = 2;
    CHECK_OK(f.start());
    const uint64_t d = f.submit(100, 512);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);
    f.sch.find(d)->speculation_fixed = true;
    const uint64_t p = f.submit(2000, 4);
    CHECK(f.sch.step() != nullptr);                     /* chunk beside the decoder */
    CHECK(f.sch.step_next() == nullptr);                /* a prompt continues: no issuing ahead */
    f.commit_plain(12);

    CHECK(f.sch.step() != nullptr);                     /* held: the decoder alone */
    CHECK_EQ((int)f.sch.plan().e.size(), 1);
    CHECK(f.sch.step_next() != nullptr);                /* the hold's second step, ahead */
    f.commit_plain(12);
    f.sch.promote_next();
    CHECK_EQ(f.sch.metrics().steps_held, 2);
    CHECK(f.sch.step_next() == nullptr);                /* the hold is spent */
    f.commit_plain(12);

    CHECK(f.sch.step() != nullptr);
    bool chunk = false;
    for (const StepEntry& e : f.sch.plan().e) if (e.seq == p && e.is_prefill) chunk = true;
    CHECK(chunk);
}

/* The arithmetic at a real deployment's shape: a 2048-token step and interval, the stager arming
 * at 1025 tokens (so a held-back chunk keeps 1024 and its step-mates take the rest). */
TEST(busy_cap_keeps_the_stager_floor_and_never_leaves_a_runt) {
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/2048, /*max_ctx=*/8192);
    f.cfg.checkpoint_interval = 2048;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1025);
    /* nothing reserved: the budget */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 0), 2048);
    /* a little reserved at an interval boundary: 2048 - 272 would leave a 272-token runt before
     * the boundary, so the interval splits in two staged halves instead */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 272), 1024);
    /* half way to the boundary the rest of the interval already fits under the cap */
    CHECK(f.sch.busy_cap(1024, 100000, 2048, 272) >= 1024);
    CHECK_EQ(f.sch.plan_chunk(1024, 100000, f.sch.busy_cap(1024, 100000, 2048, 272), nullptr), 1024);
    /* more reserved than the floor allows for: the floor */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 1500), 1024);
    /* a prompt's last stretch, shorter than the cap: all of it */
    CHECK_EQ(f.sch.plan_chunk(0, 1500, f.sch.busy_cap(0, 1500, 2048, 272), nullptr), 1500);
    /* a budget already under the floor: never more than the budget */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 512, 64), 512);
    /* decode rows alone behind it (2 rows x (1 + 3 drafts)): no split; the chunk ends a quantum
     * short of the boundary (1984) and the next step takes the remaining 64, the pair a prefill
     * already runs with decode rows ahead of it */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 8, false, /*rows_only=*/true), 2040);
    CHECK_EQ(f.sch.plan_chunk(0, 100000, f.sch.busy_cap(0, 100000, 2048, 8, false, true), nullptr), 1984);
    /* the same reserve with a share in it splits the interval */
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 8 + 256, false, false), 1024);
}

/* The share left for a request that then does not start (its media is not encoded yet) goes back
 * to the chunk that made room for it: the step is planned as if nothing had waited. */
TEST(auto_a_share_nobody_takes_goes_back_to_the_prefill) {
    Fixture f;
    share_fixture(f, -1);
    CHECK_OK(f.start());
    uint64_t a = f.submit(3500, 64);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain();
    std::shared_ptr<mm::Item> ib;
    REQUIRE(f.submit_unencoded(100, 64, &ib) != 0);
    CHECK(f.sch.step() != nullptr);
    REQUIRE_EQ((int)f.sch.plan().e.size(), 1);
    CHECK_EQ(f.sch.plan().e[0].seq, a);
    CHECK_EQ(f.sch.plan().e[0].n_tokens, 512);
    CHECK_EQ(f.sch.metrics().prefill_chunks_capped, 0);
    CHECK_EQ(f.sch.metrics().queue_depth, 1);
}

/* An explicit N also holds a chunk back for rows planned ahead of it (the operator trades prefill
 * speed for every other row's latency); auto leaves those alone. */
TEST(an_explicit_busy_prefill_chunk_applies_to_rows_ahead_auto_does_not) {
    for (int64_t mode : { (int64_t)-1, (int64_t)128 }) {
        Fixture f;
        share_fixture(f, mode);
        CHECK_OK(f.start());
        uint64_t b = f.submit(100, 512, /*priority=*/1);   /* outranks the prefill: planned first */
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        uint64_t a = f.submit(3500, 64, /*priority=*/0);
        CHECK(f.sch.step() != nullptr);                    /* b decodes, a is admitted behind it */
        f.commit_plain(12);
        int64_t most = 0; bool saw_b = true;
        for (int k = 0; k < 3; ++k) {
            CHECK(f.sch.step() != nullptr);
            bool here = false;
            for (const StepEntry& e : f.sch.plan().e) {
                if (e.seq == a) most = std::max<int64_t>(most, e.n_tokens);
                if (e.seq == b) here = true;
            }
            saw_b = saw_b && here;
            f.commit_plain(12);
        }
        CHECK(saw_b);
        if (mode > 0) CHECK_EQ(most, 128);
        else          CHECK(most > 128);
    }
}

TEST(busy_cap_is_the_budget_when_off) {
    Fixture f;
    f.build(/*blocks=*/8192, /*states=*/64, /*max_tok=*/2048, /*max_ctx=*/8192);
    f.cfg.checkpoint_interval = 2048;
    f.cfg.busy_prefill_chunk = 0;
    CHECK_OK(f.start());
    f.sch.set_stage_rows(1025);
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 272), 2048);
    CHECK_EQ(f.sch.busy_cap(0, 100000, 2048, 1500), 2048);
}

namespace {
/* A small deterministic coin: true with probability `pr`. */
struct Coin {
    uint64_t s = 12345;
    bool operator()(double pr) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return ((double)((s >> 33) & 0xFFFFFF) / (double)0x1000000) < pr;
    }
};
}  /* namespace */

TEST(the_draft_depth_is_the_configured_one_whatever_the_acceptance_trace) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = 8;
    CHECK_OK(c.configure(p));
    CHECK_EQ(c.next_depth(), 8);

    /* A workload that rejects almost everything. A dynamic policy would collapse the depth here;
     * this one does not move: the depth is the flag. */
    Coin coin;
    for (int t = 0; t < 4000; ++t) {
        int d = c.next_depth();
        CHECK_EQ(d, 8);
        int a = 0;
        while (a < d && coin(0.1)) ++a;
        c.observe(d, a);
    }
    CHECK_EQ(c.next_depth(), 8);
    CHECK(c.acceptance() > 0.0f && c.acceptance() < 0.3f);

    /* The accounting is exact and charged per drafted token, not per step: it is what a scraped
     * acceptance rate divides by. */
    DraftController d2;
    DraftController::Params q;
    q.max_depth = 8;
    CHECK_OK(d2.configure(q));
    for (int i = 0; i < 100; ++i) d2.observe(8, 0);
    CHECK_EQ(d2.drafted(), 800);
    CHECK_EQ(d2.accepted(), 0);
    CHECK_NEAR(d2.acceptance(), 0.0, 1e-9);
    d2.observe(2, 5);                 /* accepted past the draft is clamped, not counted */
    CHECK_EQ(d2.accepted(), 2);
    CHECK_EQ(d2.drafted(), 802);

    /* Depth 0 is speculation off, and it configures. */
    DraftController d3;
    DraftController::Params z;
    z.max_depth = 0;
    CHECK_OK(d3.configure(z));
    CHECK_EQ(d3.next_depth(), 0);
}

/* ------------------------------------------------------------------ the adaptive window
 * --draft-window adaptive (draft.h). The controller cases turn the policy on through Params, which
 * is what Scheduler::init does with a serial drafter; the scheduler cases start a Fixture whose
 * program declares one and whose Config asks for the window. */

namespace {
DraftController adaptive_controller(int max_depth) {
    DraftController c;
    DraftController::Params p;
    p.max_depth = max_depth;
    p.adapt = true;
    CHECK_OK(c.configure(p));
    return c;
}

/* An estimate as a test states it: acceptance `a0` at the first position and `rest` after it,
 * started, with its stop where the controller puts it. */
DraftAdapt estimate(const DraftController& c, double a0, double rest) {
    DraftAdapt s;
    s.a[0] = (float)a0;
    for (int j = 1; j < RAD_SCHED_MAX_SPEC; ++j) s.a[j] = (float)rest;
    /* The stop draft.h states: the deepest draft emitted at least kDraftMinChance of the time. */
    double p = 1.0;
    int d = 0;
    while (d < c.next_depth()) {
        p *= (double)s.a[d];
        if (p < DraftController::kDraftMinChance) break;
        ++d;
    }
    s.depth = d < DraftController::kMinDepth ? DraftController::kMinDepth : d;
    return s;
}

/* What a scheduler fixture with the adaptive window looks like: a serial head of `depth`
 * declared by the program, --num-speculative-tokens `depth`, --draft-window adaptive. */
void adaptive_fixture(Fixture& f, int depth) {
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = depth;
    f.cfg.draft_window = DraftWindow::Adaptive;
    f.prog.drafter.kind = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = depth;
}

/* What `fn` logs at Info and above: the log lines a case checks for. */
std::string info_of(const std::function<void()>& fn) {
    const Log level = log_level();
    log_set_level(Log::Info);
    std::string text = stderr_of(fn);
    log_set_level(level);
    return text;
}
}  /* namespace */

TEST(the_adaptive_window_follows_each_sequences_own_acceptance) {
    DraftController c = adaptive_controller(6);
    CHECK(c.adaptive());
    CHECK_EQ(c.next_depth(), 6);                 /* the ceiling is still the flag */
    CHECK_EQ(c.step_depth(), 0);                 /* no step has chosen D: the ceiling stands */

    /* Nothing measured yet: both start from the prior, which drafts the whole window. */
    DraftAdapt code, summary;
    CHECK_EQ(c.window_for(code), 6);
    CHECK_EQ(c.window_for(summary), 6);

    /* Two texts through one controller. The drafter almost always agrees on one; on the other it
     * agrees at the first position and rarely after it. Each stop follows its own text. */
    Coin coin;
    for (int t = 0; t < 600; ++t) {
        int d = c.window_for(code), a = 0;
        while (a < d && coin(0.95)) ++a;
        c.observe(d, a);
        c.adapt_observe(code, d, a);

        d = c.window_for(summary);
        a = 0;
        while (a < d && coin(a == 0 ? 0.6 : 0.2)) ++a;
        c.observe(d, a);
        c.adapt_observe(summary, d, a);
    }
    CHECK_EQ(c.window_for(code), 6);
    CHECK(c.window_for(summary) >= 1);
    CHECK(c.window_for(summary) <= 2);
    CHECK(code.a[0] > 0.85f);
    CHECK_NEAR(summary.a[0], 0.6, 0.2);
    CHECK(c.yield(code, 6) > c.yield(summary, 6));
    /* The global hazard is still counted, and the windows the verifies carried. */
    CHECK(c.reached(0) == 1200);
    CHECK(c.verified_at(6) >= 600);
    CHECK(c.verified_at(1) + c.verified_at(2) > 0);
    CHECK_EQ(c.verified_at(1) + c.verified_at(2) + c.verified_at(3) + c.verified_at(4) +
             c.verified_at(5) + c.verified_at(6), (int64_t)1200);

    /* A stop that went shallow comes back when the text changes: the positions it stopped
     * drafting drift toward the deepest it still drafts, so a text the drafter predicts again is
     * found without anyone drafting deep first. */
    for (int t = 0; t < 600; ++t) {
        const int d = c.window_for(summary);
        int a = 0;
        while (a < d && coin(0.95)) ++a;
        c.adapt_observe(summary, d, a);
    }
    CHECK_EQ(c.window_for(summary), 6);
}

/* A stop moves only past the hold: it is the sequence's share of the step's pass shape, and a
 * shape following every flicker of its estimate would re-record its passes for nothing
 * (runtime/ctx.h). It never goes below one draft, or the estimate could never recover. */
TEST(the_adaptive_window_moves_only_past_its_hold_and_never_below_one_draft) {
    DraftController c = adaptive_controller(6);
    DraftAdapt s;
    CHECK_EQ(c.window_for(s), 6);
    for (int t = 0; t < DraftController::kHold - 1; ++t) CHECK_EQ(c.adapt_observe(s, 6, 0), 6);
    CHECK(c.adapt_observe(s, 6, 0) < 6);                       /* the kHold-th moves */

    for (int t = 0; t < 400; ++t) c.adapt_observe(s, c.window_for(s), 0);
    CHECK_EQ(c.window_for(s), DraftController::kMinDepth);

    /* Off, the window is the flag and an observation is not taken. */
    DraftController off;
    DraftController::Params q;
    q.max_depth = 5;
    CHECK_OK(off.configure(q));
    CHECK(!off.adaptive());
    DraftAdapt w;
    CHECK_EQ(off.window_for(w), 5);
    CHECK_EQ(off.adapt_observe(w, 5, 0), 5);
    CHECK_EQ(w.depth, 0);
    off.observe(5, 2);
    CHECK_EQ(off.verified_at(5), (int64_t)0);
    off.step_time(3, 1, 12.0);                                  /* and no step is timed */
    CHECK_EQ(off.cost_samples(3, 1), (int64_t)0);

    /* Adaptive with no speculation is no policy at all. */
    DraftController zero = adaptive_controller(0);
    CHECK(!zero.adaptive());
}

/* The step cost is measured. A key (the step's deepest window, its class of running sequences)
 * with samples is its own mean; one without borrows the nearest measured key's, scaled by the
 * prior's shape; with nothing measured it is the prior. Same times, same choice. */
TEST(the_adaptive_window_prices_a_step_from_measured_step_times) {
    DraftController c = adaptive_controller(6);
    CHECK_NEAR(c.step_cost(1, 1), 1.06, 1e-12);                /* the prior */
    CHECK_NEAR(c.step_cost(6, 1), 1.36, 1e-12);

    for (int i = 0; i < 10; ++i) c.step_time(3, 1, 12.0);
    CHECK_EQ(c.cost_samples(3, 1), (int64_t)10);
    CHECK_NEAR(c.step_cost(3, 1), 12.0, 1e-9);
    CHECK_NEAR(c.step_cost(5, 1), 12.0 * 1.30 / 1.18, 1e-9);   /* borrowed, the prior's shape */
    CHECK_NEAR(c.step_cost(3, 9), 12.0, 1e-9);                 /* the nearest class */
    CHECK_EQ(c.cost_samples(3, 9), (int64_t)0);
    /* The classes: 1, 2-3, 4-7, 8-15, 16 and up. */
    c.step_time(3, 2, 20.0);
    CHECK_NEAR(c.step_cost(3, 3), 20.0, 1e-9);
    CHECK_NEAR(c.step_cost(3, 4), 20.0, 1e-9);                 /* 4-7 borrows from 2-3 */
    c.step_time(3, 16, 40.0);
    CHECK_NEAR(c.step_cost(3, 1000), 40.0, 1e-9);
    CHECK_NEAR(c.step_cost(3, 9), 40.0, 1e-9);                 /* 8-15: the nearest, 16 and up */

    /* A step that waited on something else counts for at most four times the mean. */
    const double before = c.step_cost(3, 1);
    c.step_time(3, 1, 1.0e6);
    CHECK(c.step_cost(3, 1) <= before * 4.0);
    CHECK(c.step_cost(3, 1) > before);

    /* Nonsense is not a sample. */
    c.step_time(0, 1, 5.0);
    c.step_time(7, 1, 5.0);
    c.step_time(2, 1, -1.0);
    c.step_time(2, 0, 5.0);
    CHECK_EQ(c.cost_samples(2, 1), (int64_t)0);

    /* One sequence that accepts nearly everything: on the prior the whole window pays... */
    DraftController p = adaptive_controller(6);
    DraftAdapt s = estimate(p, 0.9, 0.9);
    DraftAdapt* one[1] = { &s };
    CHECK_EQ(p.plan_step(one, 1, 1), 6);
    /* ...and where the steps past three were measured dear, three is what pays. */
    DraftController m = adaptive_controller(6);
    const double t[7] = { 0, 10.0, 11.0, 12.0, 30.0, 30.0, 30.0 };
    for (int d = 1; d <= 6; ++d) m.step_time(d, 1, t[d]);
    DraftAdapt u = estimate(m, 0.9, 0.9);
    DraftAdapt* mu[1] = { &u };
    CHECK_EQ(m.plan_step(mu, 1, 1), 3);
    CHECK_EQ(m.window_for(u), 3);
    /* Deterministic: the same times and the same estimate give the same depth. */
    DraftController m2 = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) m2.step_time(d, 1, t[d]);
    DraftAdapt u2 = estimate(m2, 0.9, 0.9);
    DraftAdapt* mu2[1] = { &u2 };
    CHECK_EQ(m2.plan_step(mu2, 1, 1), 3);
}

/* The head depth is the step's. A serial head runs as many rounds as the step's deepest window,
 * so a sequence that accepts deep is worth deep rounds only while its neighbours' loss to them
 * (every round is a pass over every drafting row) does not outweigh it. */
TEST(the_adaptive_window_chooses_the_head_depth_over_the_whole_batch) {
    DraftController c = adaptive_controller(6);
    DraftAdapt deep = estimate(c, 0.95, 0.95);
    DraftAdapt shallow = estimate(c, 0.5, 0.15);
    CHECK_EQ(deep.depth, 6);
    CHECK_EQ(shallow.depth, 1);                 /* its second draft is emitted 7.5 % of the time */

    /* Side by side on the prior: deep rounds pay for the deep sequence, and the shallow one stops
     * at its own first draft under them. */
    DraftAdapt* two[2] = { &deep, &shallow };
    CHECK_EQ(c.plan_step(two, 2, 2), 6);
    CHECK_EQ(c.window_for(deep), 6);
    CHECK_EQ(c.window_for(shallow), 1);

    /* The same deep sequence among seven shallow ones, at a class whose steps were measured to
     * grow by three units per round: eight rows pay for the rounds, and the one that gains does not
     * repay it. Every window comes down to one. */
    DraftController b = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) b.step_time(d, 8, 10.0 + 3.0 * d);
    DraftAdapt bd = estimate(b, 0.95, 0.95);
    DraftAdapt bs[7];
    DraftAdapt* eight[8] = { &bd };
    for (int i = 0; i < 7; ++i) { bs[i] = estimate(b, 0.5, 0.15); eight[i + 1] = &bs[i]; }
    CHECK_EQ(b.plan_step(eight, 8, 8), 1);
    CHECK_EQ(b.window_for(bd), 1);
    /* Alone at its own class, unmeasured, the same sequence takes the whole window. */
    DraftController a = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) a.step_time(d, 8, 10.0 + 3.0 * d);
    DraftAdapt ad = estimate(a, 0.95, 0.95);
    DraftAdapt* lone[1] = { &ad };
    CHECK(a.plan_step(lone, 1, 1) > 1);

    /* Rows that do not draft are part of the step: they emit a token whatever D is and pay for
     * every round. Here three plain rows beside one drafting row at the same measured class. */
    DraftController r = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) r.step_time(d, 4, 10.0 + 3.0 * d);
    DraftAdapt rd = estimate(r, 0.95, 0.95);
    DraftAdapt* rone[1] = { &rd };
    const int with_plain = r.plan_step(rone, 1, 4);
    DraftController r2 = adaptive_controller(6);
    for (int d = 1; d <= 6; ++d) r2.step_time(d, 4, 10.0 + 3.0 * d);
    DraftAdapt rd2 = estimate(r2, 0.95, 0.95);
    DraftAdapt* rone2[1] = { &rd2 };
    const int alone = r2.plan_step(rone2, 1, 1);    /* class 1 unmeasured: borrows class 4-7 */
    CHECK(with_plain <= alone);

    /* D moves past a hold and a hysteresis. The first step that drafts takes the best outright;
     * after that a better depth waits kHold steps. */
    DraftController h = adaptive_controller(6);
    DraftAdapt hs = estimate(h, 0.95, 0.95);
    DraftAdapt* hone[1] = { &hs };
    CHECK_EQ(h.plan_step(hone, 1, 1), 6);
    for (int d = 1; d <= 6; ++d) h.step_time(d, 1, d <= 2 ? 10.0 : 40.0);
    for (int t = 0; t < DraftController::kHold - 1; ++t) CHECK_EQ(h.plan_step(hone, 1, 1), 6);
    CHECK_EQ(h.plan_step(hone, 1, 1), 2);
    /* No drafting sequence: D stays where it is. */
    CHECK_EQ(h.plan_step(hone, 0, 3), 2);
}

/* The flag, for a drafting head (serial or block). --draft-window adaptive with prompt lookup
 * alone, or with no speculation, keeps the configured depth and says so at startup; the proposal
 * then reaches the step uncut. The default is the fixed window. */
TEST(the_adaptive_window_is_a_flag_for_a_drafting_head_and_said_when_there_is_none) {
    {
        Fixture f;
        adaptive_fixture(f, 4);
        int st = RAD_E_STATE;
        const std::string log = info_of([&] { st = f.start(); });
        CHECK_OK(st);
        CHECK(f.sch.draft().adaptive());
        CHECK(log.find("draft depth: adaptive") != std::string::npos);
    }
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.cfg.draft_window = DraftWindow::Fixed;
        CHECK_OK(f.start());
        CHECK(!f.sch.draft().adaptive());
        Config d;
        CHECK(d.draft_window == kDraftWindowDefault);      /* the default */
    }
    /* A block drafter takes the policy with its rounds fixed. It proposes its whole block in one
     * pass, and each sequence verifies the block's prefix up to its own window: the whole block
     * while its estimate is the prior, one draft once its drafts stop being taken. Startup says so. */
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.prog.drafter.kind = RAD_DRAFT_BLOCK;
        int st = RAD_E_STATE;
        const std::string log = info_of([&] { st = f.start(); });
        CHECK_OK(st);
        CHECK(f.sch.draft().adaptive());
        CHECK(log.find("draft depth: adaptive") != std::string::npos);
        CHECK(log.find("with a block drafter") != std::string::npos);
        CHECK(log.find("the window is fixed") == std::string::npos);

        const uint64_t a = f.submit(160, 2000);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[4] = { 21, 22, 23, 24 };
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_EQ(f.sch.find(a)->n_draft, 4);
        for (int t = 0; t < 64; ++t) {               /* no draft is ever taken */
            REQUIRE(f.sch.step() != nullptr);
            f.commit_plain(9);
            CHECK_OK(f.sch.set_draft(a, draft, 4));
        }
        CHECK_EQ(f.sch.find(a)->n_draft, 1);
        REQUIRE(f.sch.step() != nullptr);
        for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, 1);
        CHECK(f.sch.metrics().draft_adaptive);
        CHECK(f.sch.metrics().draft_verify_depth[4] > 0);
        CHECK(f.sch.metrics().draft_verify_depth[1] > 0);
    }
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.prog.drafter.kind = RAD_DRAFT_NONE;
        f.prog.drafter.depth = 0;
        int st = RAD_E_STATE;
        const std::string log = info_of([&] { st = f.start(); });
        CHECK_OK(st);
        CHECK(!f.sch.draft().adaptive());
        CHECK(log.find("--draft-window adaptive: the window is fixed at 4") != std::string::npos);
        CHECK(log.find("prompt lookup only") != std::string::npos);

        const uint64_t a = f.submit(160, 64);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain(11);
        int32_t draft[4] = { 21, 22, 23, 24 };
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_EQ(f.sch.find(a)->n_draft, 4);
        CHECK(!f.sch.metrics().draft_adaptive);
    }
    {
        Fixture f;
        adaptive_fixture(f, 4);
        f.cfg.n_spec = 0;
        const std::string log = info_of([&] { CHECK_OK(f.start()); });
        CHECK(!f.sch.draft().adaptive());
        CHECK(log.find("does not speculate") != std::string::npos);
    }
}

/* Each sequence has its own stop, and the rounds follow the deepest. The step issued ahead of a
 * commit verifies, per sequence, the first rounds up to its own window, and the drafter's longer
 * proposal is cut to that count rather than refused as a mismatch. */
TEST(the_adaptive_window_is_per_sequence_and_the_rounds_follow_the_deepest) {
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    const uint64_t a = f.submit(160, 2000);
    const uint64_t b = f.submit(160, 2000);
    CHECK(f.sch.step() != nullptr);
    f.commit_plain(11);

    /* `a`'s drafts are never taken and `b`'s always are. */
    int32_t draft[4] = { 21, 22, 23, 24 };
    auto commit_ab = [&]() {
        CHECK_OK(f.commit(9, 0, [&](const StepEntry& e) { return e.seq == b ? e.n_spec : 0; }));
    };
    for (int t = 0; t < 64; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_OK(f.sch.set_draft(b, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        commit_ab();
    }

    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    const RadBatch* s = f.sch.step();
    REQUIRE(s != nullptr);
    CHECK_EQ(f.sch.draft().step_depth(), 4);
    CHECK_EQ(s->n_spec, 4);
    CHECK_EQ(s->n_tok, 2 + 5);
    for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, e.seq == a ? 1 : 4);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 2.5, 1e-6);

    /* The head runs for the deepest. */
    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 2);
    CHECK_EQ(rounds, 4);
    REQUIRE(f.sch.serial_batch(0, 1) != nullptr);
    for (int rd = 2; rd <= rounds; ++rd) REQUIRE(f.sch.serial_batch(rd, 1) != nullptr);

    /* The step issued ahead: `a` verifies one of the four rounds, `b` all four. */
    f.sch.find(a)->speculation_fixed = true;
    f.sch.find(b)->speculation_fixed = true;
    f.sch.expect_draft(a, rounds);
    f.sch.expect_draft(b, rounds);
    const RadBatch* nb = f.sch.step_next();
    REQUIRE(nb != nullptr);
    CHECK_EQ(nb->n_spec, 4);
    CHECK_EQ(nb->n_tok, 2 + 5);

    /* Its commit, the watermark, and the proposals published for the step already on the card:
     * four rounds' worth to both, and each keeps what its step was planned to verify. */
    commit_ab();
    f.sch.serial_end(rounds);
    CHECK_OK(f.sch.set_draft(a, draft, rounds));
    CHECK_OK(f.sch.set_draft(b, draft, rounds));
    CHECK_EQ(f.sch.find(a)->n_draft, 1);
    CHECK_EQ(f.sch.find(b)->n_draft, 4);
    f.sch.promote_next();
    for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, e.seq == a ? 1 : 4);
    CHECK(f.sch.metrics().draft_verify_depth[1] > 0);
    CHECK(f.sch.metrics().draft_verify_depth[4] > 0);
}

/* A new sequence starts from the finished sequences, one vote each. Pooled verifies measure deep
 * positions almost only on texts good enough to reach them; here two texts the head predicts and
 * two it does not set the prior together. Until four have finished, the fixed prior applies. */
TEST(the_adaptive_window_starts_a_sequence_from_finished_sequences_not_pooled_verifies) {
    DraftController c = adaptive_controller(6);
    /* The pooled counts a deep-drafting good text leaves: position 5 accepted 95 % of the times
     * it was reached. They do not set the prior. */
    for (int t = 0; t < 1000; ++t) c.observe(6, 6);
    CHECK_NEAR(c.prior_at(5), DraftController::kPrior, 1e-9);
    DraftAdapt fresh;
    (void)c.window_for(fresh);
    CHECK_NEAR(fresh.a[5], DraftController::kPrior, 1e-6);

    auto finished = [&](double a0, double rest) {
        DraftAdapt s = estimate(c, a0, rest);
        s.n[0] = 100;
        return s;
    };
    /* Too few verifies to be more than its own prior: not counted. */
    DraftAdapt young = finished(0.1, 0.1);
    young.n[0] = 3;
    c.adapt_retire(young);
    c.adapt_retire(finished(0.95, 0.95));
    c.adapt_retire(finished(0.95, 0.95));
    c.adapt_retire(finished(0.3, 0.2));
    CHECK_NEAR(c.prior_at(5), DraftController::kPrior, 1e-9);   /* three: not yet */
    c.adapt_retire(finished(0.3, 0.2));
    CHECK_NEAR(c.prior_at(0), (0.95 + 0.95 + 0.3 + 0.3) / 4.0, 1e-6);
    CHECK_NEAR(c.prior_at(5), (0.95 + 0.95 + 0.2 + 0.2) / 4.0, 1e-6);
    DraftAdapt next;
    (void)c.window_for(next);
    CHECK_NEAR(next.a[5], (0.95 + 0.95 + 0.2 + 0.2) / 4.0, 1e-6);
    /* An unstarted estimate is not a sequence to count. */
    c.adapt_retire(DraftAdapt{});
    CHECK_NEAR(c.prior_at(5), (0.95 + 0.95 + 0.2 + 0.2) / 4.0, 1e-6);

    /* And through the scheduler: four requests whose head drafts are all refused finish, and the
     * fifth starts at one draft where the first four started at the whole window. */
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    std::vector<uint64_t> ids;
    for (int i = 0; i < 4; ++i) ids.push_back(f.submit(160, 48));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 0.0, 1e-6);     /* none drafted yet */
    int32_t draft[4] = { 21, 22, 23, 24 };
    for (uint64_t id : ids) CHECK_OK(f.sch.set_draft(id, draft, 4));
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 4.0, 1e-6);
    for (int t = 0; t < 200 && f.sch.metrics().finished < 4; ++t) {
        for (uint64_t id : ids)
            if (f.sch.find(id)) (void)f.sch.set_draft(id, draft, 4);
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(11);
    }
    CHECK_EQ(f.sch.metrics().finished, (int64_t)4);
    CHECK(f.sch.draft().prior_at(0) < 0.2);
    /* The step depth came down with them; a fresh sequence starting at the fixed prior would take
     * it back to four at its first cycle (the hold has long passed), and this one does not. */
    CHECK_EQ(f.sch.draft().step_depth(), 1);
    const uint64_t e = f.submit(160, 48);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_OK(f.sch.set_draft(e, draft, 4));
    REQUIRE(f.sch.step() != nullptr);                  /* its first cycle chooses D */
    f.commit_plain(11);
    CHECK_EQ(f.sch.draft().step_depth(), 1);
    CHECK_OK(f.sch.set_draft(e, draft, 4));
    CHECK_EQ(f.sch.find(e)->n_draft, 1);
}

/* Prompt lookup is not the head. An n-gram fallback draft counts in the reported acceptance but
 * stays out of the sequence's estimate, which prices the head's next window: refused lookups leave
 * the window alone, refused head drafts bring it down. */
TEST(the_adaptive_window_is_not_fed_by_prompt_lookup) {
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    /* A prompt that repeats, so the lookup always has a continuation to propose. */
    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 2000;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;
    const uint64_t a = r->id;
    CHECK_OK(f.sch.add(std::move(r)));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);                               /* the pattern goes on */

    int lookups = 0;
    for (int t = 0; t < 4 * DraftController::kHold; ++t) {
        REQUIRE(f.sch.step() != nullptr);              /* nothing published: the lookup fills */
        const StepEntry& e = f.sch.plan().e[0];
        if (e.n_spec > 0) {
            CHECK(e.ngram);
            CHECK(e.n_spec <= f.sch.draft().step_window());
            ++lookups;
        }
        f.commit_plain(100 + (t + 1) % 8);             /* every lookup refused */
    }
    CHECK(lookups >= 2 * DraftController::kHold);
    const SchedMetrics m = f.sch.metrics();
    CHECK(m.draft_tokens > 0);                         /* reported, as ever */
    CHECK(m.draft_pos_reached[0] >= lookups);
    CHECK_EQ(m.draft_accepted, (int64_t)0);
    CHECK_NEAR(m.draft_depth_adaptive, 4.0, 1e-6);     /* the head's window did not move */

    /* The head's own refusals are what move it. */
    int32_t draft[4] = { 21, 22, 23, 24 };
    for (int t = 0; t < 4 * DraftController::kHold; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        CHECK(!f.sch.plan().e[0].ngram);
        f.commit_plain(100 + (t + 1) % 8);
    }
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 1.0, 1e-6);
}

/* The scheduler's half of the cost: a decode step that verified drafts is timed at its deepest
 * window and row count, and the next cycle's head depth reads it. Measured dear past one round at
 * two rows, both windows (and the rounds) come down to one. */
TEST(the_adaptive_window_times_its_steps_and_the_next_cycle_reads_them) {
    Fixture f;
    adaptive_fixture(f, 4);
    CHECK_OK(f.start());
    REQUIRE(f.sch.draft().adaptive());

    const uint64_t a = f.submit(160, 2000);
    const uint64_t b = f.submit(160, 2000);
    /* Every draft taken, so both stops stay at the whole window and only the cost moves D. */
    auto commit_all = [&]() {
        CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.is_prefill ? 0 : e.n_spec; }));
    };
    REQUIRE(f.sch.step() != nullptr);                  /* both prompts */
    commit_all();
    f.sch.step_finished();                             /* a prefill step: not a sample */
    for (int d = 1; d <= 4; ++d) CHECK_EQ(f.sch.draft().cost_samples(d, 2), (int64_t)0);

    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().n_spec, 4);
    commit_all();
    f.sch.step_finished();
    CHECK_EQ(f.sch.draft().cost_samples(4, 2), (int64_t)1);

    /* The hold, then a cycle that reads steps past one round as four times dearer. */
    for (int d = 1; d <= 4; ++d)
        for (int i = 0; i < 64; ++i) f.sch.draft().step_time(d, 2, d == 1 ? 10.0 : 40.0);
    for (int t = 0; t < DraftController::kHold + 1; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_OK(f.sch.set_draft(b, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        commit_all();
    }
    CHECK_EQ(f.sch.draft().step_depth(), 1);
    CHECK_OK(f.sch.set_draft(a, draft, 4));
    CHECK_OK(f.sch.set_draft(b, draft, 4));
    REQUIRE(f.sch.step() != nullptr);
    for (const StepEntry& e : f.sch.plan().e) CHECK_EQ(e.n_spec, 1);
    int rounds = 4;
    Scheduler::SerialRow rows[4];
    CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 2);
    CHECK_EQ(rounds, 1);
    f.sch.serial_end(0);
    CHECK_NEAR(f.sch.metrics().draft_depth_adaptive, 1.0, 1e-6);
}

/* ================================================================== the step path */
/* The single-sequence decode floor the dashboard reads instead of a typed ruler. Timing is real
 * wall clock, so the assertions are the ones that hold whatever the machine measured: a step of a
 * shape that must not sample leaves the floor at zero (a min only ever goes down, so a wrongly
 * sampled step could never be told apart later -- it has to be caught while the floor is still
 * unset), and a single-sequence step sets it. */
TEST(the_decode_floor_estimator_does_not_latch_a_degenerate_bracket) {
    /* The estimator is tested on a SYNTHETIC stream because the defect is a property of the
     * estimator, not of the scheduler's ability to produce a degenerate bracket: the fixture runs
     * steps in microseconds and cannot manufacture a 1.34 ms bracket behind a 15 ms one. What the
     * live lane showed -- floor 1.34 ms against a 15.70 ms mean -- is reproduced here directly. */
    using Est = rad::DecodeFloor;

    /* First, the OLD construction, shown to fail. A min over raw brackets latches the single
     * degenerate sample and never recovers, no matter how many honest steps follow. This is the
     * defect, encoded as a test so the fix is measured against it rather than asserted. */
    {
        int64_t best = 0;
        const int64_t real = 15'000'000, degenerate = 1'340'000;
        auto old_min = [&](int64_t ns) { if (best == 0 || ns < best) best = ns; };
        old_min(degenerate);                       /* one degenerate, early */
        for (int i = 0; i < 500; ++i) old_min(real);
        CHECK_EQ(best, degenerate);                /* latched -- this is the bug */
    }

    /* The NEW construction: a min over WINDOW MEANS. The same single degenerate is diluted by
     * the honest steps sharing its window, so the floor tracks the real floor instead. */
    {
        Est est;
        const int64_t real = 15'000'000, degenerate = 1'340'000;
        est.update(degenerate);                    /* same degenerate, same position */
        for (int i = 0; i < 500; ++i) est.update(real);
        const int64_t floor = est.best();
        CHECK(floor > 0);
        CHECK(floor > degenerate);                 /* did NOT latch the degenerate */
        /* Within 25% of the real floor: the window mean is a faithful estimator. */
        CHECK(floor >= real * 3 / 4 && floor <= real);
    }

    /* Robust to a HIGH rate of degenerates: the live lane's checkpoint interval (256 tokens)
     * implies they recur. One in four here. The assertion is that the floor tracks the REAL step
     * time, not merely that it beats the degenerate -- with the rejection gate in place the latter
     * would be trivially true and prove nothing. */
    {
        Est est;
        const int64_t real = 15'000'000, degenerate = 1'340'000;
        for (int i = 0; i < 400; ++i) est.update((i % 4 == 0) ? degenerate : real);
        const int64_t floor = est.best();
        CHECK(floor >= real * 90 / 100);           /* tracks the real step, not the poison */
        CHECK(floor <= real);
    }

    /* A steady stream with no degenerates: the floor equals the true floor. */
    {
        Est est;
        const int64_t real = 14'500'000;
        for (int i = 0; i < 200; ++i) est.update(real);
        CHECK_EQ(est.best(), real);
    }
}

TEST(the_decode_floor_rejects_degenerate_brackets_so_the_floor_is_the_true_floor) {
    /* Window-averaging dilutes a LONE degenerate but not a RECURRENT one. At the live lane's
     * observed ~1.4% degenerate rate (a decode step issued ahead behind a long host commit --
     * prefill, or a KV checkpoint every 256 tokens), a min over hundreds of windows reliably
     * catches a window holding two or three degenerates, and the floor lands ~10% below the true
     * floor. That residual is phantom drift: a healthy engine reads "restart pays back" against a
     * floor it can no longer reach because the floor was never reachable -- it was an artifact.
     * So a candidate far below the running mean of accepted steps is rejected outright: the
     * degenerates sit at ~9% of the mean, real single-sequence steps at ~96%, so the two
     * populations are ~10x apart and any threshold between them separates them. 0.5 is the
     * midpoint of that gap, not a value tuned to a single distribution. */
    using Est = rad::DecodeFloor;
    const int64_t true_floor = 14'400'000, typical = 15'000'000, degen = 1'340'000;

    /* A realistic stream: ~1.4% degenerates, the rest jittered around the typical step, with the
     * true floor occasionally reached. The floor must land near the TRUE floor, not near the
     * degenerate-contaminated window mean. */
    Est est;
    uint32_t rng = 12345;
    for (int i = 0; i < 400 * Est::kWindow; ++i) {
        rng = rng * 1664525u + 1013904223u;
        const bool d = ((rng >> 16) % 1000u) < 14u;                 /* ~1.4% */
        int64_t ns;
        if (d) ns = degen;
        else {
            rng = rng * 1664525u + 1013904223u;
            ns = true_floor + (int64_t)((rng >> 8) % (typical - true_floor + 1));  /* floor..typical */
        }
        est.update(ns);
    }
    const int64_t floor = est.best();
    /* Within 5% of the true floor. The un-rejected estimator lands ~10% low here, so this fails
     * without the rejection gate. */
    CHECK(floor >= true_floor * 95 / 100);
    CHECK(floor <= true_floor * 115 / 100);

    /* The bootstrap hole: the gate needs a window of accepted samples before it can judge one, so
     * the first window is un-gated. A degenerate-heavy FIRST window must therefore be recorded by
     * neither side, or the floor latches an unreachable value forever. */
    {
        Est est;
        for (int i = 0; i < Est::kWindow; ++i) est.update(degen);   /* whole first window poisoned */
        CHECK_EQ(est.best(), (int64_t)0);
        CHECK_EQ(est.recent(), (int64_t)0);
        for (int i = 0; i < 40 * Est::kWindow; ++i) est.update(typical);
        CHECK(est.best() >= typical * 95 / 100);   /* floor came from a gated window, not bootstrap */
    }

    /* And the rejection must NOT eat real steps: a stream of only real steps (no degenerates)
     * still records a floor equal to the true floor. */
    Est clean;
    for (int i = 0; i < 100 * Est::kWindow; ++i) clean.update(true_floor);
    CHECK_EQ(clean.best(), true_floor);
}

TEST(the_decode_floor_reports_a_recent_window_for_like_for_like_drift) {
    /* The advisory compares "how fast is it now" against "how fast has it ever been". If those two
     * numbers come from different instruments -- the engine's floor versus the browser's per-poll
     * mean -- they disagree by pure order statistics: a min over hundreds of windows sits below a
     * min over the last sixty samples even on a perfectly flat engine. Measured live 2026-10-18:
     * a 7.7% phantom gap on a lane that had not changed. So the engine reports BOTH sides of the
     * comparison, from the same stream and the same window size. */
    using Est = rad::DecodeFloor;
    const int64_t base = 15'000'000;

    /* A steady engine with realistic run-to-run jitter: the recent window and the floor are both
     * window means of the same population, so they must agree closely. */
    {
        Est est;
        for (int i = 0; i < 400 * Est::kWindow; ++i) {
            const int64_t jitter = ((i % 7) - 3) * 200'000;      /* +/- 0.6 ms, deterministic */
            est.update(base + jitter);
        }
        const int64_t floor = est.best(), recent = est.recent();
        CHECK(floor > 0);
        CHECK(recent > 0);
        /* The phantom gap must be small: this is the number the deadband is sized against. */
        const int64_t gap_pct = 100 * (recent - floor) / floor;
        CHECK(gap_pct < 3);
    }

    /* A degraded engine: the recent window rises, the floor remembers the old best. The verdict
     * must separate -- that is the whole point of the advisory. */
    {
        Est est;
        for (int i = 0; i < 200 * Est::kWindow; ++i) est.update(base);
        const int64_t floor = est.best();
        for (int i = 0; i < 20 * Est::kWindow; ++i) est.update(base * 6 / 5);   /* 20% slower */
        const int64_t recent = est.recent();
        CHECK(recent > floor * 115 / 100);
        /* The floor does not chase the degradation: a restart is exactly what would restore it. */
        CHECK_EQ(est.best(), floor);
    }

    /* Before any window completes there is no recent reading -- the page must not divide by it. */
    {
        Est est;
        for (int i = 0; i < Est::kWindow - 1; ++i) est.update(base);
        CHECK_EQ(est.recent(), (int64_t)0);
        est.update(base);                        /* completes window 1: the un-gated warmup */
        CHECK_EQ(est.recent(), (int64_t)0);      /* warmup is recorded by neither side */
        for (int i = 0; i < Est::kWindow; ++i) est.update(base);
        CHECK(est.recent() > 0);
    }
}

TEST(the_decode_floor_records_only_single_sequence_decode_steps) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    /* A prefill step samples nothing. Budget 320 tokens: this test walks 1 prefill + one window,
     * at most six windows until the floor is set and one more, 257 steps. */
    uint64_t a = f.submit(160, 320, /*priority=*/0, /*ignore_eos=*/true);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);

    /* The floor is a min over WINDOW MEANS of single-sequence decode steps, so it stays 0 until a
     * full window (DecodeFloor::kWindow) has run -- unlike the old raw min, which latched the very
     * first bracket. One short of a window and the floor is still unset. */
    for (int i = 1; i < rad::DecodeFloor::kWindow; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().n_seq_decode, 1);
        CHECK_EQ(f.sch.plan().phase, RAD_PHASE_DECODE);
        f.commit_plain(12);
        f.sch.step_finished();
    }
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);

    /* The step that completes the FIRST window leaves the floor unset: window 1 is the un-gated
     * warmup (the rejection gate needs a window of accepted samples before it can judge one), so
     * it is recorded by neither side. */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(13);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);

    /* The first gated window sets the floor: positive, and no larger than the cumulative decode
     * time it was drawn from. The fixture's steps take microseconds and vary widely, so the gate
     * may reject some and the window take more than kWindow steps to fill. */
    for (int i = 0; i < 6 * rad::DecodeFloor::kWindow && !f.sch.metrics().step_ns_decode_best;
         ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(14);
        f.sch.step_finished();
    }
    const int64_t floor1 = f.sch.metrics().step_ns_decode_best;
    CHECK(floor1 > 0);
    CHECK(floor1 <= f.sch.metrics().step_ns_decode);

    /* A further full window can only lower the floor or leave it: it is a min over windows. */
    for (int i = 0; i < rad::DecodeFloor::kWindow; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(15);
        f.sch.step_finished();
    }
    const int64_t floor2 = f.sch.metrics().step_ns_decode_best;
    CHECK(floor2 > 0);
    CHECK(floor2 <= floor1);
    (void)a;
}

TEST(the_decode_floor_ignores_a_decode_step_issued_behind_a_prefill) {
    /* The decode step issued behind the prefill's host-side staging is a real single-sequence
     * decode step: it is eligible, and the window mean dilutes its short bracket. Decode steps
     * that begin with it produce a floor. The fixture's steps take microseconds and vary widely,
     * so the gate may reject some and a window take more than kWindow steps to fill. */
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    uint64_t a = f.submit(160, 256, /*priority=*/0, /*ignore_eos=*/true);
    REQUIRE(f.sch.step() != nullptr);          /* prefill */
    f.commit_plain(21);
    f.sch.step_finished();

    /* Window 1 (holding the prefill-following step) is the un-gated warmup, so the floor only
     * appears once a gated window completes. */
    for (int i = 0; i < 6 * rad::DecodeFloor::kWindow && !f.sch.metrics().step_ns_decode_best;
         ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().phase, RAD_PHASE_DECODE);
        CHECK_EQ(f.sch.plan().n_seq_decode, 1);
        CHECK(f.sch.plan().n_tok_decode >= 1); /* a token DID advance; token count is not the guard */
        f.commit_plain(22);
        f.sch.step_finished();
    }
    /* The window filled, counting the prefill-following step: the floor is set. */
    CHECK(f.sch.metrics().step_ns_decode_best > 0);
    (void)a;
}

TEST(the_decode_floor_ignores_batched_and_mixed_steps) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    /* Two requests prefilled, then decoding TOGETHER: every decode step here carries two rows, so
     * the floor must still be unset at the end -- a batched step is a different measurement and
     * keying the floor to it would report the batch arriving as drift. */
    uint64_t a = f.submit(160, 64);
    uint64_t b = f.submit(160, 64);
    REQUIRE(f.sch.step() != nullptr);                  /* both prompts */
    f.commit_plain(11);
    f.sch.step_finished();
    for (int i = 0; i < 4; ++i) {
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().n_seq_decode, 2);
        CHECK_EQ(f.sch.plan().phase, RAD_PHASE_DECODE);
        f.commit_plain(12);
        f.sch.step_finished();
    }
    CHECK(f.sch.metrics().steps_decode > 0);            /* decode steps really ran ... */
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);   /* ... and none of them sampled */
    (void)a; (void)b;
}

TEST(the_decode_floor_ignores_a_lone_decode_row_inside_a_mixed_step) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    /* The subtle case: a mixed step carrying exactly ONE decode row alongside a prefill chunk.
     * n_seq_decode is 1, so a check on the row count alone would sample it -- but that row shared
     * the step with a chunk an order of magnitude dearer, and its time is not the mechanical floor.
     * The phase gate is what excludes it. */
    uint64_t b = f.submit(1000, 64, /*priority=*/1);    /* several chunks */
    uint64_t a = f.submit(160,  64, /*priority=*/0);    /* one chunk, then decodes */
    REQUIRE(f.sch.step() != nullptr);                   /* both prompts */
    f.commit_plain(11);
    f.sch.step_finished();

    REQUIRE(f.sch.step() != nullptr);                   /* a decodes, b still chunking */
    CHECK_EQ(f.sch.plan().n_seq_decode, 1);
    CHECK_EQ(f.sch.plan().phase, RAD_PHASE_MIXED);
    f.commit_plain(12);
    f.sch.step_finished();
    CHECK_EQ(f.sch.metrics().step_ns_decode_best, (int64_t)0);
    (void)a; (void)b;
}

TEST(the_step_path_does_not_allocate) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());

    for (int i = 0; i < 6; ++i) f.submit(64, 4096);

    /* Warm up: the first steps grow the request's block vector and its output, each sized once.
     * What must be zero is the steady-state step. */
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

    /* Zero, not "few": an allocator call here is a declare-phase bug (IMPLEMENTATION.md). */
    CHECK_EQ(g_allocs, 0L);
}

/* The same for a mixed step: decode rows beside a prefill chunk, which the step reorders (decode
 * rows first) and checkpoints. A reordering through a temporary buffer reaches the allocator on
 * every such step. */
TEST(a_mixed_step_does_not_allocate) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    CHECK_OK(f.start());
    for (int i = 0; i < 6; ++i) f.submit(64, 4096);
    for (int i = 0; i < 4; ++i) {
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
    }
    /* A long prompt behind them, admitted before the count starts: the fakes' own bookkeeping
     * (the cache's checkpoint log) is sized here too. */
    f.pc.committed.reserve(64);
    f.submit(3000, 8, /*priority=*/5);             /* ahead of them, so the step reorders */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();

    g_allocs = 0;
    g_counting = true;
    int mixed = 0;
    for (int i = 0; i < 8; ++i) {
        if (!f.sch.step()) break;
        mixed += f.sch.plan().phase == RAD_PHASE_MIXED;
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
    CHECK_EQ(mixed, 8);
    CHECK_EQ(g_allocs, 0L);
}

/* The card's block tables are staged as changed entries, so they are checked whole after every
 * build: a paged row reads the manager's table then -1 to the pitch, a state row reads the slot.
 * The run exercises appends at block crossings, rollbacks of rejected drafts, rows reused after a
 * request ends, and a late prefill; a row staged from a stale picture would read a wrong block. */
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
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
    }
    /* The run reached steady decode with rows to compare, not an empty loop. */
    CHECK(checked > 100);
}

/* A sequence that leaves a row and returns to it is restaged whole. Its change marks are taken
 * wherever it is written, so the old row's copy goes stale; restaging only later changes would
 * keep entries a rollback replaced earlier. The scheduler's own passes reorder rows, so this
 * drives the builder directly with explicit row orders. */
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
    std::vector<BatchBuilder::RankIO> rio;
    rio.push_back({ 0, nullptr, &f.prog });
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

/* Two conv groups, each derivation reading its own window. With one shared slot array and window,
 * only the last-declared RAD_KV_CONV group would be filled and the other's row would stay -1.
 * (Today only `conv_state_index` reads these, and no architecture calls it; conv kernels use the
 * per-group RadKVGroupBatch::state_index.) The widths differ because `conv_state_index` returns
 * `slot * window + cursor % window`, and only differing depths expose the wrong group's.
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

    /* ---- an unnamed group is refused, not guessed at ----------------------------------------- */
    {
        Fixture f;
        f.build();                       /* declares `conv_idx` with no group -- fine at one */
        two_groups(f);
        /* The refusal logs at Error by design; the line is expected here. */
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
    /* Neither is -1. Under one shared slot array the group that is not last-declared gets -1 for
     * every sequence, so this keeps the derivation working for the first architecture to call it. */
    if (a && c) {
        CHECK(a[0] >= 0);
        CHECK(c[0] >= 0);
    }
}

/* ================================================================== the derivation registry
 *
 * A derivation named in a schema but missing here is refused by name at declare, with the known
 * list printed (spec §2.3); otherwise a kernel would read whatever the arena last held. So every
 * row is reachable by name and has a function, and an unknown name resolves to -1, not row 0.
 */
TEST(every_derivation_is_reachable_by_name_and_an_unknown_one_is_not) {
    const int n = derive_count();
    CHECK(n > 0);

    for (int i = 0; i < n; ++i) {
        const DeriveRow* r = derive_row(i);
        CHECK(r != nullptr);
        if (!r) continue;
        CHECK(r->name && *r->name);
        CHECK(r->fn != nullptr);          /* a row with no function would resolve and then write
                                           * nothing, the failure above */
        CHECK(r->doc && *r->doc);         /* it is printed in the refusal's known list */
        CHECK_EQ(derive_index(r->name), i);
    }

    /* Names are unique, or derive_index would answer with the first row and the second could
       never be selected. */
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            CHECK(std::strcmp(derive_row(i)->name, derive_row(j)->name) != 0);

    /* -1 for a name the core does not know, not 0, which is a valid index. */
    CHECK_EQ(derive_index("not_a_derivation"), -1);
    CHECK_EQ(derive_index(""), -1);
    CHECK_EQ(derive_index(nullptr), -1);
    /* A near miss on a real name is still a miss: a typo is reported, not approximated. */
    CHECK_EQ(derive_index("seq_start_idx"), -1);

    /* An index outside the table gives null, not a read past it. */
    CHECK(derive_row(-1) == nullptr);
    CHECK(derive_row(n) == nullptr);
    CHECK(derive_row(n + 1000) == nullptr);
}

/* ================================================================== request lifetime */

/* A preempted request loses its draft window. The sampler sizes rows from Request::n_draft, and
 * the request returns as a one-row prefill; a stale window of four would stage five rows and shift
 * every later request onto its neighbour's row. */
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

    /* When it is admitted again, the window it carries is its entry's. */
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 0);
    CHECK_EQ(f.sch.find(a)->n_draft, 0);
}

/* Each terminal id is handed over once: take_completed swaps the list out under the lock. A
 * reference into scheduler storage that nothing cleared would have every finished request walked
 * again on every step, without the lock, while an HTTP thread's cancel appends to it. */
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

/* Every row names one of the card's step slots. Slots are taken on arrival, so queued requests
 * and arrivals during a pending reap are numbered past max_seqs; the advance indexes max_seqs
 * StepSlots by the row's card index, never by that slot number. */
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

/* The context pass mirrors the trunk step row for row: the drafter reads hidden states
 * positionally, so a sequence the commit finished keeps its rows (as padding that stores nothing),
 * or every later sequence would draft from its neighbour's states. */
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

/* A cancel between two drafter passes (built one at a time, lock released between) must not free
 * the blocks: the query pass would get a block-table row of -1 under a non-zero length. */
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

    /* The drafter is done with it, so it is released at the next commit, or here for an idle
     * engine that runs none. */
    CHECK(f.sch.has_deferred());
    f.sch.release_deferred();
    CHECK(!f.kv.has_sequence(a));
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK(!f.sch.has_deferred());
    CHECK_OK(f.sch.reap(a));
    CHECK(f.sch.find(a) == nullptr);
}

/* A failed admission gives back its checkpoint hit. The slot is named by id only and may be
 * reassigned between attempts, so trusting the old id would restore another sequence's recurrent
 * state. The next attempt looks the prompt up again, and the admission is counted once. */
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
    uint64_t a = f.submit(600, 8);
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
    CHECK_EQ(m.prompt_tokens, 160 + 600);
    CHECK_EQ(m.attn_cached_tokens, 256);
    CHECK_EQ(m.linear_cached_tokens, 256);
}

/* A refused lookahead stops drafting, not the row: the history pass writes the draft head's K and
 * V for positions the sequence already holds, and a dropped row would leave them unwritten for
 * good, since the next step's history starts at its own context. */
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

/* Draft round 1 rides the history pass. A drafting verifier owes the head's history at all
 * 1 + n_spec verified positions, and the card repeats the last committed one past it, so the
 * run's last row is round 1's input: the pass names that row, and round 1 has no pass of its own. */
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

/* The serial driver's two shapes over one verify step. A head with its own cache owes the history
 * of every verified position and steps one position per round; a head reading the trunk's cache
 * (rad_drafter_shared_kv) owes nothing, runs one row per drafting sequence and does not step.
 * Qwen's MTP head uses the first shape, which must not change. */
namespace {
struct SerialShape {
    int     hist_tok = 0, hist_out = 0, hist_dout = 0;
    int32_t hist_ctx = 0, hist_row = 0;
    int32_t round_ctx[4] = {0, 0, 0, 0};
};

void serial_shape(bool shared, SerialShape& r) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.cfg.n_spec = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    f.prog.drafter_shared_kv = shared;
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
    const RadBatch* hb = f.sch.serial_batch(0, 1);
    REQUIRE(hb != nullptr);
    CHECK(hb->draft_pass < 0);
    r.hist_tok  = (int)hb->n_tok;
    r.hist_out  = (int)hb->n_out;
    r.hist_dout = (int)hb->n_draft_out;
    r.hist_ctx  = hb->ctx_lens[0];
    r.hist_row  = hb->out_ids[0];
    for (int rd = 2; rd <= 3; ++rd) {
        const RadBatch* rb = f.sch.serial_batch(rd, 1);
        REQUIRE(rb != nullptr);
        CHECK_EQ(rb->n_tok, 1);
        r.round_ctx[rd] = rb->ctx_lens[0];
    }
    f.sch.serial_end(3);
}
}  /* namespace */

TEST(a_head_with_a_cache_of_its_own_is_staged_as_it_always_was) {
    SerialShape o;
    serial_shape(false, o);
    CHECK_EQ(o.hist_tok, 5);                    /* every verified position */
    CHECK_EQ(o.hist_out, 5);                    /* one trunk row each */
    CHECK_EQ(o.hist_dout, 1);
    CHECK_EQ(o.hist_ctx, 160);                  /* the history starts at the first verified one */
    CHECK_EQ(o.hist_row, 0);                    /* the verify step's first row */
    CHECK_EQ(o.round_ctx[2], 165);              /* the furthest round 2 can sit: one past the last */
    CHECK_EQ(o.round_ctx[3], 166);              /* ...and the head steps one position a round */
}

TEST(a_head_with_no_cache_of_its_own_runs_one_row_and_stays_put) {
    SerialShape s;
    serial_shape(true, s);
    CHECK_EQ(s.hist_tok, 1);                    /* round 1 alone */
    CHECK_EQ(s.hist_out, 1);
    CHECK_EQ(s.hist_dout, 1);
    CHECK_EQ(s.hist_ctx, 164);                  /* the furthest the committed position can be */
    CHECK_EQ(s.hist_row, 0);                    /* the run's first trunk row; the card adds acc */
    CHECK_EQ(s.round_ctx[2], 164);              /* and every round stays there */
    CHECK_EQ(s.round_ctx[3], 164);
}

/* A head with no cache of its own is owed no history: a prefill chunk that drafts nothing runs no
 * pass for it, while a head with its own cache is given the whole chunk. */
TEST(a_head_with_no_cache_of_its_own_runs_no_pass_when_nothing_drafts) {
    for (int shared = 0; shared < 2; ++shared) {
        Fixture f;
        f.build(/*blocks=*/10, /*states=*/64);      /* exactly the prompt: nothing to look ahead into */
        f.cfg.n_spec = 4;
        f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
        f.prog.drafter.depth = 4;
        f.prog.drafter_shared_kv = shared != 0;
        CHECK_OK(f.start());
        f.submit(160, 64);
        CHECK(f.sch.step() != nullptr);
        f.commit_plain();
        int rounds = 4;
        Scheduler::SerialRow rows[4];
        CHECK_EQ(f.sch.serial_begin(&rounds, rows, 4), 0);
        const RadBatch* hb = f.sch.serial_batch(0, 1);
        if (shared) {
            CHECK(hb == nullptr);
        } else {
            REQUIRE(hb != nullptr);
            CHECK_EQ(hb->n_tok, 160);
        }
        f.sch.serial_end(0);
    }
}

/* A step is issued ahead of a commit only once the commit cannot change the request's
 * speculation: a request the sampler has not started (grammar unchecked) is not issued ahead; one
 * with a rewinding grammar, or none, is. */
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

/* The context bounds a sequence whatever max_tokens says, "no limit" included. Positions past the
 * last one the deployment can address have no rotary row or block-table column; a sequence
 * running into them would corrupt memory instead of stopping. */
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
        /* Every draft accepted, the most a step can emit, so the bound must cut it short. */
        CHECK_OK(f.commit(9, 0, [](const StepEntry& x) { return x.n_spec; }));
    }
    const Request* ra = f.sch.find(a);
    CHECK_EQ((int)ra->state, (int)ReqState::Finished);
    CHECK_EQ(ra->finish_reason, std::string("length"));
    CHECK_EQ((int)(ra->prompt.size() + ra->output.size()), 170);

    /* A prompt that fills the context has nothing to generate into and is never run. */
    uint64_t full = f.submit(170, 8);
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ((int)f.sch.find(full)->state, (int)ReqState::Failed);

    /* Asking for more than the context holds reserves only what it can hold. */
    uint64_t big = f.submit(8, 2147483647);
    CHECK(f.sch.find(big)->output.capacity() <= (size_t)(170 + RAD_SCHED_MAX_SPEC));
}

/* The cache's reserve is measured against what is held, not what is backed. An elastic pool that
 * has handed its unused top back to the card reads almost no free blocks however little is held,
 * and trimming against that would drain the whole cache after every request. */
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
    CHECK(f.pc.trims.empty());
}

/* When live sequences hold more than the reserve allows, the trim asks for the shortfall once.
 * Nothing the cache can give closes it entirely, and a loop evicting until it did would also take
 * the conversation this close has just published. */
TEST(the_cache_trim_asks_for_the_shortfall_and_not_for_the_whole_cache) {
    Fixture f;
    f.build(/*blocks=*/40, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.evict_grant = 1 << 20;

    f.submit(160, /*max_tokens=*/1);                 /* 10 blocks, finishes on the first commit */
    f.submit(176, /*max_tokens=*/64);                /* 11 blocks each, still running after it */
    f.submit(176, /*max_tokens=*/64);
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ(f.kv.held_blocks(0), 32ll);             /* 8 free against a reserve of 10 */
    f.commit_plain();
    CHECK_EQ((int)f.pc.published.size(), 1);
    REQUIRE(f.pc.trims.size() == 1u);
    CHECK_EQ(f.pc.trims[0], 2ll);
    CHECK_EQ(f.pc.evict_calls, 0);
}

/* The cache gives way to a new request as it does to a running one. With nothing running no other
 * path takes blocks back from the cache, so a pool full of finished conversations would keep the
 * request waiting forever. */
TEST(an_admission_short_of_blocks_takes_them_from_the_cache) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.seed(240);                                  /* 15 of 20 held by finished conversations */
    f.pc.attn = 0;                                   /* and none of them is this prompt */
    f.pc.evict_frees = true;
    f.pc.evict_grant = 1 << 20;
    const uint64_t a = f.submit(160, 8);             /* 10 blocks, 5 free */
    CHECK(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Running);
    CHECK(f.pc.evict_calls >= 1);
    CHECK(f.pc.holds.empty());
}

/* A refused admission holds what its restore brought back and lets go once the request is gone,
 * so the next attempt finds the chain on the card instead of copying it again. */
TEST(a_refused_admission_holds_its_chain_until_the_request_goes) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    f.pc.seed(240);
    f.pc.attn = 0;
    const uint64_t a = f.submit(160, 8);             /* the cache gives nothing back */
    CHECK(f.sch.step() == nullptr);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Waiting);
    REQUIRE_EQ(f.pc.holds.size(), 1u);
    CHECK_EQ(f.pc.holds[0], a);
    CHECK_OK(f.sch.cancel(a));
    CHECK(std::find(f.pc.releases.begin(), f.pc.releases.end(), a) != f.pc.releases.end());
}

/* An engine that has stopped answers every request it held and refuses new ones, rather than
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

/* Each of those requests says what stopped it: the sink carries the status and the sentence the
 * server puts in the 500, rather than a generic "ok". */
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

/* A request's own draft counts and admission time reach its sink, so a response can report them
 * (aggregate counters cannot be split between overlapping requests). The counts are the same
 * proposed/accepted numbers /metrics sums. */
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
    uint32_t rng = 99;
    for (int step = 0; step < 40 && !sink.done(); ++step) {
        int32_t draft[4] = { 31, 32, 33, 34 };
        (void)f.sch.set_draft(id, draft, 4);
        const RadBatch* b = f.sch.step();
        if (!b) break;
        CHECK(sink.t_admitted_ns() >= 0);
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) {
            const int32_t acc = random_accept(rng, e);
            if (!e.is_prefill && e.n_spec > 0) {
                drafted += e.n_spec;
                accepted += acc;
            }
            return acc;
        }));
    }
    CHECK(drafted > 0);
    CHECK_EQ(sink.n_drafted(), drafted);
    CHECK_EQ(sink.n_draft_accepted(), accepted);
}

/* ================================================================== the step issued ahead */
/* A scripted model both pipelines follow. Entry i of step t accepts ahead_acc(seq, t) of its
 * drafts, then draws ahead_tok(seq, t); the drafts verified at step t are ahead_draft(seq, t, j).
 * Every token is distinct from every other and from the prompts, so prompt lookup proposes
 * nothing and a misplaced token cannot match by accident. */
namespace {
uint32_t ahead_mix(uint64_t a, uint64_t b) {
    uint64_t x = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull);
    x ^= x >> 29; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 32;
    return (uint32_t)x;
}
int32_t ahead_tok(uint64_t seq, int t) { return 20000 + (int32_t)seq * 1000 + t; }
int32_t ahead_draft(uint64_t seq, int t, int j) {
    return 100000 + (int32_t)seq * 10000 + t * 10 + j;
}
int32_t ahead_acc(uint64_t seq, int t, int n_spec) {
    return n_spec > 0 ? (int32_t)(ahead_mix(seq, (uint64_t)t) % (uint32_t)(n_spec + 1)) : 0;
}

/* A build's per-row arrays as the card holds them (device memory is host memory here). Each
 * slot is checked against the build's own block table, since two runs that allocated blocks in
 * a different order hold different block ids for the same position. */
struct BatchRows {
    std::vector<uint64_t> seq;
    std::vector<int32_t>  tok, pos, cu, q, ctx, acc, used;
    int bad_slots = 0;
    bool operator==(const BatchRows& o) const {
        return seq == o.seq && tok == o.tok && pos == o.pos && cu == o.cu && q == o.q &&
               ctx == o.ctx && acc == o.acc && used == o.used;
    }
};

BatchRows read_rows(const RadBatch* b, const StepPlan& p, const FakeKV& kv) {
    BatchRows r;
    for (int64_t i = 0; i < b->n_seq; ++i) {
        r.seq.push_back(p.e[(size_t)i].seq);
        r.q.push_back(b->q_lens[i]);
        r.ctx.push_back(b->ctx_lens[i]);
        r.acc.push_back(b->num_accepted ? b->num_accepted[i] : 1);
    }
    for (int64_t i = 0; i <= b->n_seq; ++i) r.cu.push_back(b->cu_seqlens[i]);
    for (int64_t t = 0; t < b->n_tok; ++t) {
        r.tok.push_back(b->token_ids[t]);
        r.pos.push_back(b->positions[t]);
    }
    for (int g = 0; g < b->n_kv_groups; ++g) {
        const RadKVGroupBatch& G = b->kv[g];
        for (int64_t i = 0; i < b->n_seq; ++i) {
            r.used.push_back(G.seqused[i]);
            const int32_t* bt = G.block_table + i * G.block_table_pitch;
            for (int32_t t = b->cu_seqlens[i]; t < b->cu_seqlens[i + 1]; ++t) {
                const int64_t p_ = b->positions[t];
                int64_t want = -1;
                if (kv.geom.g[(size_t)g].paged) {
                    const int64_t bi = p_ / G.block_size;
                    if (bi < G.block_table_pitch && bt[bi] >= 0)
                        want = bt[bi] * G.block_size + p_ % G.block_size;
                } else {
                    want = kv.state_slot(r.seq[(size_t)i], g);
                }
                if (want < 0 || G.slot_mapping[t] != want) ++r.bad_slots;
            }
        }
    }
    return r;
}

struct AheadRun {
    std::vector<BatchRows> steps;
    std::vector<std::vector<int32_t>> out;
    std::vector<int64_t> held;      /* paged blocks each sequence holds at the end */
    int ahead = 0;                  /* steps that ran issued ahead */
    int card_host_disagree = 0;
};

void ahead_setup(Fixture& f, int n_spec, std::vector<uint64_t>& ids) {
    f.build();
    f.cfg.n_spec = n_spec;
    CHECK_OK(f.start());
    for (int n : { 40, 56, 72 }) {
        auto r = f.request(n, 0);
        r->speculation_fixed = true;     /* no grammar: the commit cannot change speculation */
        ids.push_back(f.add(std::move(r)));
    }
}

int ahead_commit(Fixture& f, int t) {
    const StepPlan& p = f.sch.plan();
    std::vector<int32_t> a(p.e.size()), tok(p.e.size());
    std::vector<uint8_t> eos(p.e.size(), 0);
    for (size_t i = 0; i < p.e.size(); ++i) {
        a[i] = ahead_acc(p.e[i].seq, t, p.e[i].n_spec);
        tok[i] = ahead_tok(p.e[i].seq, t);
    }
    Scheduler::StepResult r;
    r.n_seq = (int)p.e.size();
    r.n_accepted = a.data();
    r.token = tok.data();
    r.eos = eos.data();
    return f.sch.commit(r);
}

void ahead_publish(Fixture& f, const std::vector<uint64_t>& ids, int n_spec, int t) {
    for (uint64_t id : ids) {
        int32_t d[8];
        for (int j = 0; j < n_spec; ++j) d[j] = ahead_draft(id, t, j);
        if (n_spec > 0) CHECK_OK(f.sch.set_draft(id, d, n_spec));
    }
}

void ahead_finish(Fixture& f, const std::vector<uint64_t>& ids, AheadRun& run) {
    for (uint64_t id : ids) {
        run.out.push_back(f.sch.find(id)->output);
        run.held.push_back((int64_t)f.kv.block_table(id, 0).size());
    }
}

/* Every step built by step() after the commit. */
AheadRun ahead_reference(int n_spec, int steps) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, n_spec, ids);
    AheadRun run;
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    for (int t = 0; t < steps; ++t) {
        CHECK_OK(ahead_commit(f, t));
        ahead_publish(f, ids, n_spec, t + 1);
        b = f.sch.step();
        if (!b) break;
        run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    ahead_finish(f, ids, run);
    return run;
}

/* The engine's order with a step issued ahead (Engine::step_loop): the step's sampler, the card's
 * acceptance into the StepSlots, the draft head's proposals into them, expect_draft and
 * step_next, the trunk advance on the new build, then the commit, the drafter's set_draft and
 * promote_next. */
AheadRun ahead_pipelined(int n_spec, int steps) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, n_spec, ids);
    AheadRun run;
    std::vector<StepSlot> slots((size_t)f.cfg.max_seqs);
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    for (int t = 0; t < steps; ++t) {
        /* The sampler's rows for step t: the drafts it agrees with, then its own token. */
        const StepPlan& p = f.sch.plan();
        std::vector<int32_t> y((size_t)b->n_out + 1, -1);
        for (const StepEntry& e : p.e) {
            if (e.dev_out < 0) continue;
            const int32_t a = ahead_acc(e.seq, t, e.n_spec);
            for (int j = 0; j <= e.n_spec; ++j)
                y[(size_t)(e.dev_out + j)] = j < a ? ahead_draft(e.seq, t, j) : ahead_tok(e.seq, t);
        }
        AdvanceArgs acc = f.sch.advance_args(0);
        acc.mode = ADV_ACCEPT;
        acc.st = slots.data();
        acc.src = y.data();
        for (int i = 0; i < acc.n_seq; ++i) adv_row(acc, i);
        for (const StepEntry& e : p.e) {
            const StepSlot& S = slots[(size_t)e.card];
            if (S.acc != ahead_acc(e.seq, t, e.n_spec) || S.tok[0] != ahead_tok(e.seq, t))
                ++run.card_host_disagree;
            /* The draft head's proposals for step t + 1 (ADV_ROUND / ADV_TAKE on a card). */
            for (int j = 0; j < n_spec; ++j)
                slots[(size_t)e.card].tok[1 + j] = ahead_draft(e.seq, t + 1, j);
            if (n_spec > 0) f.sch.expect_draft(e.seq, n_spec);
        }
        const RadBatch* nb = f.sch.step_next();
        if (nb) {
            AdvanceArgs tr = f.sch.advance_args(0);
            tr.mode = ADV_TRUNK;
            tr.st = slots.data();
            for (int i = 0; i < tr.n_seq; ++i) adv_row(tr, i);
        }
        CHECK_OK(ahead_commit(f, t));
        ahead_publish(f, ids, n_spec, t + 1);
        if (nb) {
            f.sch.promote_next();
            b = nb;
            ++run.ahead;
        } else {
            b = f.sch.step();
            if (!b) break;
        }
        run.steps.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    ahead_finish(f, ids, run);
    return run;
}
}  /* namespace */

/* A step issued ahead is the step the scheduler would have built after the commit: the same
 * tokens, positions, slots (each through its own table), lengths, context and num_accepted on
 * every row, once the card's advance has filled in what the host only bounded. The host sizes
 * such a step at the furthest context it can reach and the card's acceptance places it; a
 * disagreement puts K/V at wrong positions or reads the wrong token, and the output is silently
 * wrong. Run with drafts (a serial head's pipeline) and without (no drafter). */
TEST(a_step_issued_ahead_equals_the_step_built_after_the_commit) {
    for (int n_spec : { 0, 4 }) {
        const int T = 24;
        const AheadRun a = ahead_reference(n_spec, T);
        const AheadRun b = ahead_pipelined(n_spec, T);
        CHECK_EQ(b.ahead, T);                       /* every step after the first ran ahead */
        CHECK_EQ(b.card_host_disagree, 0);
        REQUIRE_EQ(a.steps.size(), b.steps.size());
        REQUIRE_EQ(a.steps.size(), (size_t)T + 1);
        for (size_t t = 0; t < a.steps.size(); ++t) {
            const bool same = a.steps[t] == b.steps[t];
            CHECK(same);
            if (!same) std::fprintf(stderr, "    n_spec %d: step %zu differs\n", n_spec, t);
            CHECK_EQ(a.steps[t].bad_slots, 0);
            CHECK_EQ(b.steps[t].bad_slots, 0);
        }
        CHECK(a.out == b.out);
        /* The pipelined run holds blocks up to the furthest position a step issued ahead could
         * reach, and gives back rejected drafts' blocks only on a commit nothing runs ahead of:
         * never fewer than the reference, and at most one block more. */
        REQUIRE_EQ(a.held.size(), b.held.size());
        for (size_t i = 0; i < a.held.size(); ++i) {
            CHECK(b.held[i] >= a.held[i]);
            CHECK(b.held[i] <= a.held[i] + 1);
        }
    }
}

/* ================================================================== a block drafter on the card */
namespace {
/* Every pass of a block drafter's run: each step's trunk batch, then its context and query
 * passes, as the card holds them once its advances have run. */
struct BlockRun {
    std::vector<BatchRows> trunk, context, query;
    std::vector<std::vector<int32_t>> out;
    int ahead = 0;
    int card_host_disagree = 0;
};
constexpr int kBlk = 8, kBlkDepth = 7;
constexpr int32_t kBlkMask = 99;

/* The host's driver: both passes built after the commit, from the committed sequence. */
BlockRun block_reference(int steps) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, kBlkDepth, ids);
    BlockRun run;
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    Scheduler::BlockRow rows[4];
    for (int t = 0; t < steps; ++t) {
        CHECK_OK(ahead_commit(f, t));
        CHECK_EQ(f.sch.block_begin(kBlk, kBlkDepth, 0, kBlkMask, rows, 4), 3);
        const RadBatch* cb = f.sch.block_context(t + 1);
        if (!cb) { CHECK(cb != nullptr); return run; }
        run.context.push_back(read_rows(cb, f.sch.plan(), f.kv));
        const RadBatch* qb = f.sch.block_query(t + 1);
        if (!qb) { CHECK(qb != nullptr); return run; }
        /* The query plan is the scheduler's own; its rows are the drafting sequences in order. */
        StepPlan qp;
        for (int i = 0; i < 3; ++i) { StepEntry e; e.seq = rows[i].seq; qp.e.push_back(e); }
        run.query.push_back(read_rows(qb, qp, f.kv));
        f.sch.block_end();
        ahead_publish(f, ids, kBlkDepth, t + 1);
        b = f.sch.step();
        if (!b) break;
        run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    for (uint64_t id : ids) run.out.push_back(f.sch.find(id)->output);
    return run;
}

/* The card's driver (Engine::issue_block_drafter): the step's acceptance into the StepSlots, both
 * passes built before the commit with the query rows placed by ADV_BLOCK, the proposals into the
 * slots (ADV_BTAKE), expect_draft and step_next with ADV_TRUNK, then the commit, block_end, the
 * drafts published and promote_next. */
BlockRun block_pipelined(int steps) {
    Fixture f;
    std::vector<uint64_t> ids;
    ahead_setup(f, kBlkDepth, ids);
    BlockRun run;
    std::vector<StepSlot> slots((size_t)f.cfg.max_seqs);
    const RadBatch* b = f.sch.step();
    if (!b) return run;
    run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    Scheduler::BlockRow rows[4];
    for (int t = 0; t < steps; ++t) {
        const StepPlan& p = f.sch.plan();
        std::vector<int32_t> y((size_t)b->n_out + 1, -1);
        for (const StepEntry& e : p.e) {
            if (e.dev_out < 0) continue;
            const int32_t a = ahead_acc(e.seq, t, e.n_spec);
            for (int j = 0; j <= e.n_spec; ++j)
                y[(size_t)(e.dev_out + j)] = j < a ? ahead_draft(e.seq, t, j) : ahead_tok(e.seq, t);
        }
        AdvanceArgs acc = f.sch.advance_args(0);
        acc.mode = ADV_ACCEPT;
        acc.st = slots.data();
        acc.src = y.data();
        for (int i = 0; i < acc.n_seq; ++i) adv_row(acc, i);
        for (const StepEntry& e : p.e) {
            const StepSlot& S = slots[(size_t)e.card];
            if (S.acc != ahead_acc(e.seq, t, e.n_spec) || S.tok[0] != ahead_tok(e.seq, t))
                ++run.card_host_disagree;
        }

        CHECK_EQ(f.sch.block_begin(kBlk, kBlkDepth, 0, kBlkMask, rows, 4, /*on_card=*/true), 3);
        const RadBatch* cb = f.sch.block_context(t + 1);
        if (!cb) { CHECK(cb != nullptr); return run; }
        run.context.push_back(read_rows(cb, f.sch.plan(), f.kv));
        const RadBatch* qb = f.sch.block_query(t + 1);
        if (!qb) { CHECK(qb != nullptr); return run; }
        AdvanceArgs q = f.sch.advance_args(0);
        q.mode = ADV_BLOCK;
        q.st = slots.data();
        for (int i = 0; i < q.n_seq; ++i) adv_row(q, i);
        StepPlan qp;
        for (int i = 0; i < 3; ++i) { StepEntry e; e.seq = rows[i].seq; qp.e.push_back(e); }
        run.query.push_back(read_rows(qb, qp, f.kv));
        /* The block's proposals for step t + 1, taken behind the anchor. */
        std::vector<int32_t> prop((size_t)3 * kBlkDepth);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < kBlkDepth; ++j)
                prop[(size_t)(i * kBlkDepth + j)] = ahead_draft(rows[i].seq, t + 1, j);
        AdvanceArgs tk = f.sch.advance_args(0);
        tk.mode = ADV_BTAKE;
        tk.round = kBlkDepth;
        tk.src = prop.data();
        tk.src_pitch = kBlkDepth;
        tk.st = slots.data();
        for (int i = 0; i < tk.n_seq; ++i) adv_row(tk, i);
        for (int i = 0; i < 3; ++i) f.sch.expect_draft(rows[i].seq, kBlkDepth);

        const RadBatch* nb = f.sch.step_next();
        if (nb) {
            AdvanceArgs tr = f.sch.advance_args(0);
            tr.mode = ADV_TRUNK;
            tr.st = slots.data();
            for (int i = 0; i < tr.n_seq; ++i) adv_row(tr, i);
        }
        CHECK_OK(ahead_commit(f, t));
        f.sch.block_end();
        ahead_publish(f, ids, kBlkDepth, t + 1);
        if (nb) {
            f.sch.promote_next();
            b = nb;
            ++run.ahead;
        } else {
            b = f.sch.step();
            if (!b) break;
        }
        run.trunk.push_back(read_rows(b, f.sch.plan(), f.kv));
    }
    for (uint64_t id : ids) run.out.push_back(f.sch.find(id)->output);
    return run;
}
}  /* namespace */

/* A block drafter on the card runs the passes the host's driver runs after the commit: the same
 * trunk steps, and context and query passes with the same positions, lengths and slots once the
 * card's advances have filled in what the host only bounded, and the same query tokens. Its query
 * block is staged before the step's acceptance is known, at the furthest anchor; a wrong position
 * drafts from the wrong place (acceptance collapses, no error) and a wrong slot writes the
 * drafter's K/V into another block. Every step after the first runs issued ahead.
 *
 * The context pass's token ids are not compared: that pass reads hidden states and no ids
 * (Scheduler::block_context), and staged before the commit, its accepted positions are not in the
 * sequence yet. */
TEST(a_block_drafter_on_the_card_runs_the_passes_the_hosts_driver_runs) {
    const int T = 24;
    const BlockRun a = block_reference(T);
    const BlockRun b = block_pipelined(T);
    CHECK_EQ(b.ahead, T);
    CHECK_EQ(b.card_host_disagree, 0);
    REQUIRE_EQ(a.trunk.size(), (size_t)T + 1);
    REQUIRE_EQ(a.trunk.size(), b.trunk.size());
    REQUIRE_EQ(a.context.size(), b.context.size());
    REQUIRE_EQ(a.query.size(), b.query.size());
    for (size_t t = 0; t < a.trunk.size(); ++t) {
        CHECK(a.trunk[t] == b.trunk[t]);
        CHECK_EQ(b.trunk[t].bad_slots, 0);
    }
    for (size_t t = 0; t < a.context.size(); ++t) {
        BatchRows ca = a.context[t], cb = b.context[t];
        ca.tok.clear();
        cb.tok.clear();
        const bool same_c = ca == cb, same_q = a.query[t] == b.query[t];
        CHECK(same_c);
        CHECK(same_q);
        if (!same_c || !same_q)
            std::fprintf(stderr, "    step %zu: context %s, query %s\n", t,
                         same_c ? "same" : "differs", same_q ? "same" : "differs");
        CHECK_EQ(a.context[t].bad_slots, 0);
        CHECK_EQ(b.context[t].bad_slots, 0);
        CHECK_EQ(a.query[t].bad_slots, 0);
        CHECK_EQ(b.query[t].bad_slots, 0);
    }
    CHECK(a.out == b.out);
}

/* ================================================================== the token budget per token */
/* A verify step emits up to 1 + n_spec tokens at once. The budget (max_tokens, and the context)
 * is applied token by token before each reaches the sink: trimming `output` afterwards would leave
 * the overshoot already streamed, and the same prompt at the same max_tokens would come back
 * longer with speculation on. `output` alone cannot show this; the sink is the client's view. */
TEST(a_verify_step_at_the_max_tokens_edge_streams_exactly_max_tokens) {
    server::Sink sink;
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    const uint64_t id = f.submit_to(&sink, 40, /*max_tokens=*/3);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(5);
    int32_t draft[4] = { 21, 22, 23, 24 };
    CHECK_OK(f.sch.set_draft(id, draft, 4));
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e[0].n_spec, 4);
    CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));   /* all accepted */

    CHECK_EQ(sink.n_produced(), (int64_t)3);
    server::SinkToken got[8];
    REQUIRE_EQ(sink.read(got, 8), (size_t)3);
    CHECK_EQ(got[0].token, 5); CHECK_EQ(got[1].token, 21); CHECK_EQ(got[2].token, 22);
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Length);
    const Request* r = f.sch.find(id);
    CHECK_EQ((int)r->output.size(), 3);
    CHECK_EQ(r->finish_reason, std::string("length"));
}

/* The context bounds the stream the same way, with no max_tokens at all. */
TEST(a_verify_step_at_the_context_edge_streams_exactly_the_room_left) {
    server::Sink sink;
    Fixture f;
    f.build();
    f.cfg.max_ctx = 170;
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    const uint64_t id = f.submit_to(&sink, 160, /*max_tokens=*/0);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(5);
    int32_t draft[4] = { 21, 22, 23, 24 };
    for (int guard = 0; guard < 8 && !sink.done(); ++guard) {
        CHECK_OK(f.sch.set_draft(id, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(f.commit(9, 0, [](const StepEntry& e) { return e.n_spec; }));
    }
    CHECK(sink.done());
    CHECK_EQ(sink.n_produced(), (int64_t)10);           /* 170 - 160 */
    CHECK_EQ((int)f.sch.find(id)->output.size(), 10);
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Length);
}

/* An end-of-generation token ends an ordinary request after its accepted drafts and the token are
 * streamed; with ignore_eos the request runs on, and the same tokens reach the client. */
TEST(an_end_of_generation_token_after_accepted_drafts_ends_the_request_after_them) {
    for (int ignore = 0; ignore < 2; ++ignore) {
        server::Sink sink;
        Fixture f;
        f.build();
        f.cfg.n_spec = 4;
        CHECK_OK(f.start());
        auto r = f.request(40, 64);
        r->sink = &sink;
        r->ignore_eos = ignore != 0;
        const uint64_t id = f.add(std::move(r));
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain(5);
        int32_t draft[4] = { 21, 22, 23, 24 };
        CHECK_OK(f.sch.set_draft(id, draft, 4));
        REQUIRE(f.sch.step() != nullptr);
        CHECK_OK(f.commit(9, /*eos=*/1, [](const StepEntry&) { return 2; }));
        CHECK_EQ(sink.n_produced(), (int64_t)4);          /* 5, then 21 22 9 */
        if (ignore) {
            CHECK(!sink.done());
            CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Running);
        } else {
            CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Stop);
            CHECK_EQ(f.sch.find(id)->finish_reason, std::string("stop"));
        }
    }
}

/* ================================================================== a recycled slot */
/* Slots are recycled, and every piece of drafter state must start over with the request in it. A
 * serial head's history watermark left from the slot's last tenant empties the next prompt's
 * history pass, so the head drafts against another request's K/V and greedy output varies from
 * request to request with speculation on. */
TEST(a_request_in_a_recycled_slot_runs_its_serial_heads_whole_history) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    CHECK_OK(f.start());

    auto history = [&](int expect_rows) {
        int rounds = 4;
        Scheduler::SerialRow rows[4];
        (void)f.sch.serial_begin(&rounds, rows, 4);
        const RadBatch* hb = f.sch.serial_batch(0, 1);
        if (expect_rows >= 0) {
            CHECK(hb != nullptr);
            if (hb) CHECK_EQ(hb->n_tok, expect_rows);
        }
    };
    const uint64_t a = f.submit(160, 2);
    REQUIRE(f.sch.step() != nullptr);
    history(160);
    f.commit_plain(11);
    f.sch.serial_end(1);
    REQUIRE(f.sch.step() != nullptr);              /* the decode that finishes it */
    history(-1);
    f.commit_plain(12);
    f.sch.serial_end(1);
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_OK(f.sch.reap(a));

    f.submit(96, 8);                               /* takes the slot A left */
    REQUIRE(f.sch.step() != nullptr);
    history(96);
    f.commit_plain(13);
    f.sch.serial_end(1);
}

/* The same for a block drafter's filled range: a range left from the slot's last tenant (here a
 * prefix hit's floor at 512) would keep the next request's drafter waiting for a window that
 * never slides past it. */
TEST(a_request_in_a_recycled_slot_starts_its_block_drafters_cache_from_nothing) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/8192);
    f.cfg.n_spec = 7;
    f.prog.drafter.kind = RAD_DRAFT_BLOCK;
    f.prog.drafter.window = 512;
    KVGroupInfo w;                                 /* the drafter's own group, not prefix-cached */
    w.name = "draft";
    w.decl.kind = RAD_KV_WINDOW;
    w.decl.window = 512;
    w.block_size = 16;
    f.prog.kv_groups.push_back(w);
    f.kv.setup(f.prog.kv_groups, f.cfg.max_ctx, 4096, 64);
    f.pc.groups = { 0 };
    CHECK_OK(f.start());

    /* A: a hit resumed at 512 leaves its drafter's cache starting there. */
    f.pc.seed(1024); f.pc.lin = 1024; f.pc.src = 3;
    const uint64_t a = f.submit(1100, 1);
    Scheduler::BlockRow rows[4];
    for (int st = 0; st < 8 && f.sch.find(a)->state != ReqState::Finished; ++st) {
        REQUIRE(f.sch.step() != nullptr);
        if (st == 0) CHECK_EQ(f.sch.plan().e[0].ctx_len, 512);
        f.commit_plain(11);
        (void)f.sch.block_begin(8, 7, 512, 99, rows, 4);
        f.sch.block_end();
    }
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_OK(f.sch.reap(a));

    /* B, no hit, in A's slot: its drafter's cache starts at its own first row. */
    f.pc.attn = 0; f.pc.lin = 0; f.pc.src = -1;
    const uint64_t b = f.submit(96, 8);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(11);
    CHECK_EQ(f.sch.block_begin(8, 7, 512, 99, rows, 4), 1);
    CHECK_EQ((int)rows[0].seq, (int)b);
    f.sch.block_end();
}

/* ================================================================== num_accepted */
/* num_accepted is one-based on every row and staged on every step: 1 on every prefill row (first
 * chunk, warm chunk, or admission), the previous verify's accepted count + 1 on a decode row, and
 * the card's array equal to the plan whether or not the step drafted. A prefill row at 0 has a
 * warm chunk read its rolling windows at -1 (a foreign token, or the slot before); an array
 * staged only on drafting steps leaves the linear kernels a stale count. Both are silent. */
TEST(num_accepted_is_one_based_on_every_row_and_staged_on_every_step) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    const uint64_t big = f.submit(600, 4);          /* three chunks: 256, 256, 88 */
    const uint64_t spec = f.submit(40, 200);
    const uint64_t plain = f.submit(48, 200);
    auto r = f.sch.find(plain);
    r->may_speculate = false;
    std::unordered_map<uint64_t, int32_t> last_acc;
    uint32_t rng = 777;
    int prefill_rows = 0, verify_rows = 0, plain_rows = 0;
    for (int step = 0; step < 30; ++step) {
        int32_t draft[4] = { 31, 32, 33, 34 };
        if (f.sch.find(spec) && f.sch.find(spec)->n_computed > 0) (void)f.sch.set_draft(spec, draft, 4);
        const RadBatch* b = f.sch.step();
        if (!b) break;
        REQUIRE(b->num_accepted != nullptr);
        const StepPlan& p = f.sch.plan();
        for (size_t i = 0; i < p.e.size(); ++i) {
            const StepEntry& e = p.e[i];
            CHECK(e.n_accepted >= 1);
            CHECK_EQ(b->num_accepted[i], e.n_accepted);
            if (e.is_prefill) {
                CHECK_EQ(e.n_accepted, 1);
                ++prefill_rows;
            } else {
                CHECK_EQ(e.n_accepted, last_acc[e.seq] + 1);
                if (e.n_spec > 0) ++verify_rows; else ++plain_rows;
            }
        }
        CHECK_OK(f.commit(7, 0, [&](const StepEntry& e) {
            const int32_t a = random_accept(rng, e);
            if (!e.is_prefill) last_acc[e.seq] = a;
            return a;
        }));
    }
    (void)big;
    CHECK(prefill_rows >= 5);
    CHECK(verify_rows > 5);
    CHECK(plain_rows > 5);
}

/* ================================================================== an embedding input */
/* An embedding input is admitted whole in one step even on a model that checkpoints its linear
 * state: the checkpoint splits do not apply to an input that never resumes. Split there, the
 * whole-or-nothing rule refused it every step, and as the head of the queue it held back every
 * request behind it for good. */
TEST(an_embedding_input_longer_than_the_checkpoint_interval_is_admitted_whole) {
    Fixture f;
    f.build();                                     /* linear state, interval 256, max_tok 512 */
    CHECK_OK(f.start());
    auto r = f.request(300, 1);
    r->embed = true;
    const uint64_t e = f.add(std::move(r));
    const uint64_t n = f.submit(40, 4);
    REQUIRE(e != 0);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    CHECK_EQ((int)f.sch.find(e)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(n)->state, (int)ReqState::Running);
    bool whole = false;
    for (const StepEntry& x : f.sch.plan().e)
        if (x.seq == e) whole = x.n_tokens == 300 && x.produces_token && x.ckpt_slot < 0;
    CHECK(whole);
}

/* ================================================================== speculation gates */
/* A request that may not speculate (a constraining grammar that cannot snapshot) gets no draft
 * from either drafter: set_draft takes nothing and prompt lookup proposes nothing, though its
 * text repeats. A draft for it makes the sampler refuse the step. */
TEST(a_request_that_may_not_speculate_gets_no_draft_from_either_drafter) {
    Fixture f;
    f.build();
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    auto r = std::make_unique<Request>();
    r->id = f.next_id++;
    r->max_tokens = 200;
    r->may_speculate = false;
    r->prompt.resize(160);
    for (int i = 0; i < 160; ++i) r->prompt[(size_t)i] = 100 + i % 8;
    const uint64_t a = r->id;
    CHECK_OK(f.sch.add(std::move(r)));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);
    int32_t draft[4] = { 101, 102, 103, 104 };
    for (int t = 0; t < 20; ++t) {
        CHECK_OK(f.sch.set_draft(a, draft, 4));
        CHECK_EQ(f.sch.find(a)->n_draft, 0);
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().e[0].n_spec, 0);
        CHECK_EQ(f.sch.plan().e[0].n_draft, 0);
        f.commit_plain(100 + (t + 1) % 8);           /* the pattern goes on */
    }
    CHECK_EQ(f.sch.metrics().draft_tokens, (int64_t)0);
}

/* ================================================================== the step issued ahead, ends */
/* A request carried by a step issued ahead outlives its end by that step: a cancel and reap keep
 * its blocks and its Request until the step ahead has committed, and take_completed withholds its
 * id until then, because that step still writes its blocks and its sampler is staged after this
 * commit. Released early, a block the step ahead writes goes to a new sequence (cross-request KV
 * corruption) and the sampler state goes away under a staged step. */
TEST(a_cancel_while_a_step_runs_ahead_keeps_the_request_until_that_step_commits) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    auto r = f.request(40, 64);
    r->speculation_fixed = true;
    const uint64_t a = f.add(std::move(r));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    REQUIRE(f.sch.step_next() != nullptr);
    const int64_t held = f.kv.free_blocks(0);
    CHECK(held < before);

    CHECK_OK(f.sch.cancel(a));
    CHECK_OK(f.sch.reap(a));
    std::vector<uint64_t> done, all;
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    CHECK(done.empty());
    CHECK(f.sch.find(a) != nullptr);
    CHECK_EQ(f.kv.free_blocks(0), held);

    f.commit_plain();                              /* the step in flight */
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    CHECK(done.empty());
    CHECK_EQ(f.kv.free_blocks(0), held);           /* the step ahead still writes them */
    CHECK(f.sch.find(a) != nullptr);

    f.sch.promote_next();
    f.commit_plain();                              /* the step ahead */
    CHECK(f.sch.find(a) == nullptr);
    CHECK_EQ(f.kv.free_blocks(0), before);
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    f.sch.take_completed(&done);
    all.insert(all.end(), done.begin(), done.end());
    CHECK_EQ(all.size(), (size_t)1);
    CHECK(!all.empty() && all[0] == a);
}

/* A request that finishes at the commit of a step whose successor is already issued keeps its
 * blocks for that successor, which writes them, and returns them at its commit. */
TEST(a_request_finishing_under_a_step_issued_ahead_keeps_its_blocks_for_that_step) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    auto r = f.request(40, /*max_tokens=*/2);
    r->speculation_fixed = true;
    const uint64_t a = f.add(std::move(r));
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    REQUIRE(f.sch.step() != nullptr);
    REQUIRE(f.sch.step_next() != nullptr);
    const int64_t held = f.kv.free_blocks(0);
    f.commit_plain();                              /* the second token: finished */
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Finished);
    CHECK_EQ(f.kv.free_blocks(0), held);
    f.sch.promote_next();
    f.commit_plain();
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ((int)f.sch.find(a)->output.size(), 2);   /* the step ahead emitted nothing */
}

/* ================================================================== terminal paths */
/* A lone request that outgrows the whole pool fails with the reason the client sees and gives its
 * blocks back; nothing is preempted, since there is nobody to preempt and preempting itself would
 * loop. */
TEST(a_request_no_pool_can_hold_fails_kv_exhausted_and_frees_its_blocks) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/10, /*states=*/64);         /* 160 tokens */
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    const uint64_t a = f.submit_to(&sink, 150, 64);
    for (int t = 0; t < 20 && !sink.done(); ++t) {
        if (!f.sch.step()) break;
        f.commit_plain();
    }
    CHECK(sink.done());
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Failed);
    CHECK_EQ(f.sch.find(a)->finish_reason, std::string("kv_exhausted"));
    CHECK_EQ(sink.status(), (int)RAD_E_FULL);
    CHECK_EQ(f.kv.free_blocks(0), before);
    CHECK_EQ(f.sch.metrics().preemptions, 0);
    CHECK(f.sch.step() == nullptr);
}

/* The lowest-priority running request that cannot grow while the requests ahead of it hold the
 * pool is preempted by recompute and waits for them, rather than failed: it fits the pool on its
 * own, so failing it returns an error for a request the engine can serve. It re-enters at the
 * front of the queue and runs once they finish. Only a request running alone is failed. */
TEST(the_last_request_short_of_a_block_yields_to_the_requests_ahead_of_it) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);         /* 320 tokens */
    CHECK_OK(f.start());
    std::vector<uint64_t> ahead;
    for (int i = 0; i < 3; ++i) ahead.push_back(f.submit(70, 2));   /* 5 blocks, room to 80 */
    const uint64_t last = f.submit_to(&sink, 80, 4);                /* 5 blocks, full at 80 */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)4);
    CHECK_EQ(f.kv.free_blocks(0), (int64_t)0);
    f.commit_plain();

    /* The last needs a sixth block for position 80 and the pool has none. */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)3);
    CHECK_EQ((int)f.sch.find(last)->state, (int)ReqState::Preempted);
    CHECK(f.sch.find(last)->finish_reason.empty());
    CHECK(!sink.done());
    CHECK_EQ(f.sch.metrics().preemptions, 1);
    uint64_t q[4];
    CHECK_EQ(f.sch.queue_snapshot(q, 4), 1);
    CHECK_EQ(q[0], last);
    for (uint64_t id : ahead) CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Running);
    f.commit_plain();                              /* the three finish (max_tokens 2) */

    for (int t = 0; t < 8 && !sink.done(); ++t) {
        if (!f.sch.step()) break;
        f.commit_plain();
    }
    CHECK(sink.done());
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Length);
    CHECK_EQ(sink.n_produced(), (int64_t)4);
}

/* A prompt the KV pool cannot hold even alone is refused when it is submitted, with the sink
 * failed and the reason naming the pool's size, and the queue behind it moves on. Admitted, it
 * would wait at the head of the queue forever (admission takes only what the pool holds), and
 * its client would get no reply at all. A prompt of exactly the pool's size runs. */
TEST(a_prompt_the_kv_pool_can_never_hold_is_refused_when_submitted) {
    server::Sink refused, fits;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);         /* 320 tokens, under the 4096 context */
    CHECK_OK(f.start());
    auto r = f.request(321, 8);
    r->sink = &refused;
    CHECK_EQ(f.sch.add(std::move(r)), (int)RAD_E_FULL);
    CHECK(refused.done());
    CHECK_EQ((int)refused.finish_reason(), (int)server::Finish::Error);
    CHECK_EQ(refused.status(), (int)RAD_E_FULL);
    CHECK(has(refused.why() ? refused.why() : "", "maximum context length is 320 tokens"));

    const uint64_t ok = f.submit_to(&fits, 320, 1);
    REQUIRE(ok != 0);
    for (int t = 0; t < 4 && !fits.done(); ++t) {      /* two chunks: 256, then 64 */
        REQUIRE(f.sch.step() != nullptr);
        CHECK_EQ(f.sch.plan().e[0].seq, ok);
        f.commit_plain();
    }
    CHECK(fits.done());
    CHECK_EQ(fits.n_produced(), (int64_t)1);
    CHECK_EQ((int)fits.finish_reason(), (int)server::Finish::Length);
}

/* A client that hung up is closed Cancelled at the next commit, and its blocks go back then. */
TEST(a_client_hang_up_closes_the_request_cancelled_at_the_next_commit) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    const uint64_t a = f.submit_to(&sink, 40, 64);
    REQUIRE(f.sch.step() != nullptr);
    sink.request_cancel();
    f.commit_plain();
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Cancelled);
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Cancelled);
    CHECK_EQ(f.kv.free_blocks(0), before);
}

/* A sink the client stopped draining fails the request rather than drop tokens from the middle of
 * the stream, and the request's blocks go back. */
TEST(a_sink_overflow_fails_the_request_and_returns_its_blocks) {
    server::Sink sink(16);                         /* the smallest ring: 16 tokens */
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const int64_t before = f.kv.free_blocks(0);
    const uint64_t a = f.submit_to(&sink, 40, 64);
    for (int t = 0; t < 30 && !sink.done(); ++t) {
        REQUIRE(f.sch.step() != nullptr);
        f.commit_plain();
    }
    CHECK(sink.overflowed());
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Failed);
    CHECK_EQ(f.sch.find(a)->finish_reason, std::string("sink_overflow"));
    CHECK_EQ((int)sink.finish_reason(), (int)server::Finish::Error);
    CHECK_EQ(f.kv.free_blocks(0), before);
}

/* A token whose requested logprobs could not be computed fails the request, and neither it nor
 * anything after it reaches the client: a token without the logprobs it was asked for would be a
 * silently incomplete answer. Both the prefill's token and an accepted draft's. */
TEST(a_token_whose_logprobs_failed_fails_the_request_and_is_not_sent) {
    for (int verify = 0; verify < 2; ++verify) {
        server::Sink sink;
        Fixture f;
        f.build();
        f.cfg.n_spec = 4;
        CHECK_OK(f.start());
        auto r = f.request(40, 64);
        r->sink = &sink;
        r->sp.n_probs = 0;
        const uint64_t id = f.add(std::move(r));
        REQUIRE(f.sch.step() != nullptr);
        if (verify) {
            f.commit_plain(5);
            int32_t draft[4] = { 21, 22, 23, 24 };
            CHECK_OK(f.sch.set_draft(id, draft, 4));
            REQUIRE(f.sch.step() != nullptr);
        }
        /* Rows: the first emitted token's logprobs are good; the second's failed. */
        TokenLogprobs lp[6];
        for (auto& x : lp) { x.token = 1; x.logprob = -0.5f; }
        lp[1].token = -1;
        if (!verify) lp[0].token = -1;
        int32_t acc = 3, tok = 9, first = 0;
        uint8_t eos = 0;
        Scheduler::StepResult res;
        res.n_seq = 1;
        res.n_accepted = &acc;
        res.token = &tok;
        res.eos = &eos;
        res.lp = lp;
        res.first_row = &first;
        const int64_t had = sink.n_produced();
        CHECK_OK(f.sch.commit(res));
        CHECK_EQ(sink.n_produced(), had + (verify ? 1 : 0));   /* only the good one */
        CHECK_EQ((int)f.sch.find(id)->state, (int)ReqState::Failed);
        CHECK_EQ(sink.status(), (int)RAD_E_STATE);
    }
}

/* Priority before arrival: a later, higher-priority request is admitted first. */
TEST(a_later_higher_priority_request_is_admitted_first) {
    Fixture f;
    f.build();
    f.cfg.max_seqs = 1;
    CHECK_OK(f.start());
    const uint64_t lo = f.submit(40, 4, /*priority=*/0);
    const uint64_t hi = f.submit(40, 4, /*priority=*/3);
    uint64_t q[4];
    REQUIRE_EQ(f.sch.queue_snapshot(q, 4), 2);
    CHECK_EQ(q[0], hi);
    CHECK_EQ(q[1], lo);
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)1);
    CHECK_EQ(f.sch.plan().e[0].seq, hi);
    CHECK_EQ((int)f.sch.find(lo)->state, (int)ReqState::Waiting);
}

/* The head of the queue blocks admission behind it: a smaller request that would fit waits too,
 * so a large request cannot be starved by a stream of small ones. */
TEST(the_head_of_the_queue_blocks_admission_behind_it) {
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    const uint64_t x = f.submit(160, 64);          /* 10 of 20 blocks */
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain();
    const uint64_t head = f.submit(200, 4);        /* 13 blocks: does not fit beside x */
    const uint64_t small = f.submit(16, 4);        /* 1 block: would */
    REQUIRE(f.sch.step() != nullptr);
    CHECK_EQ((int)f.sch.find(x)->state, (int)ReqState::Running);
    CHECK_EQ((int)f.sch.find(head)->state, (int)ReqState::Waiting);
    CHECK_EQ((int)f.sch.find(small)->state, (int)ReqState::Waiting);
    CHECK_EQ(f.sch.plan().e.size(), (size_t)1);
}

/* ================================================================== preemption continues */
/* Preemption by recompute is the one path that re-prefills output tokens. The re-admitted request
 * stages prompt ++ output at positions 0.., its conv cursor starts over, and its client's stream
 * goes on with no token repeated or skipped. */
TEST(a_preempted_request_continues_its_stream_exactly) {
    server::Sink sink;
    Fixture f;
    f.build(/*blocks=*/20, /*states=*/64);
    CHECK_OK(f.start());
    auto r = f.request(160, 64);
    r->sink = &sink;
    const uint64_t a = f.add(std::move(r));
    const uint64_t b = f.submit(160, 64, /*priority=*/5);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(100);
    CHECK_EQ(sink.n_produced(), (int64_t)1);
    REQUIRE(f.sch.step() != nullptr);              /* both need a block at 160: a is preempted */
    CHECK_EQ((int)f.sch.find(a)->state, (int)ReqState::Preempted);
    f.commit_plain(101);
    CHECK_EQ(sink.n_produced(), (int64_t)1);
    CHECK_OK(f.sch.cancel(b));
    CHECK_OK(f.sch.reap(b));

    const RadBatch* rb = f.sch.step();
    REQUIRE(rb != nullptr);
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)1);
    const StepEntry& e = f.sch.plan().e[0];
    CHECK_EQ(e.seq, a);
    CHECK_EQ(e.ctx_len, 0);
    CHECK_EQ(e.n_tokens, 161);
    CHECK_EQ(e.conv_cursor, 0);
    CHECK(e.produces_token);
    REQUIRE_EQ(rb->n_tok, (int64_t)161);
    bool staged = true;
    for (int i = 0; i < 160; ++i)
        staged = staged && rb->token_ids[i] == 1000 + i && rb->positions[i] == i;
    CHECK(staged);
    CHECK_EQ(rb->token_ids[160], 100);
    CHECK_EQ(rb->positions[160], 160);
    f.commit_plain(102);
    REQUIRE(f.sch.step() != nullptr);
    f.commit_plain(103);
    CHECK_EQ(sink.n_produced(), (int64_t)3);
    server::SinkToken got[4];
    REQUIRE_EQ(sink.read(got, 4), (size_t)3);
    CHECK_EQ(got[0].token, 100); CHECK_EQ(got[1].token, 102); CHECK_EQ(got[2].token, 103);
    const std::vector<int32_t> want = { 100, 102, 103 };
    CHECK(f.sch.find(a)->output == want);
}

/* ================================================================== media rows across chunks */
/* A prompt whose tokens 50..149 are one picture's 100 encoder rows, prefilled in 64-token chunks:
 * each chunk stages exactly the rows of the picture it covers, at the batch rows of those tokens,
 * and a history pass (which reads one token on) the rows one further. A wrong offset gives the
 * model another part of the picture while the output stays fluent. Row r of the item holds the
 * value r * 4 + column, so a misplaced row reads as a wrong number. */
TEST(media_rows_are_staged_by_chunk_at_the_tokens_that_take_them) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/64);
    f.prog.encoder.modalities = RAD_MM_IMAGE;
    f.prog.encoder.n_embd = 4;
    f.prog.encoder.patch_dim = 4;
    f.prog.encoder.merge = 1;
    f.prog.encoder.max_patches = 128;
    f.cfg.n_spec = 4;
    f.prog.drafter.kind  = RAD_DRAFT_SERIAL;
    f.prog.drafter.depth = 4;
    CHECK_OK(f.start());

    auto item = std::make_shared<mm::Item>();
    item->n_seg = 1;
    item->seg_rows = 100;
    item->encoded = true;
    item->embd.resize(100 * 4);
    for (int r = 0; r < 100; ++r)
        for (int c = 0; c < 4; ++c) item->embd[(size_t)(r * 4 + c)] = (uint16_t)(r * 4 + c);
    auto req = f.request(200, 4);
    req->media = std::make_shared<mm::PromptMedia>();
    req->media->items.push_back(mm::Placed{ item, 50 });
    mm::RopeLayout::Run run;
    run.tok = 50; run.n = 100; run.w = 10; run.base = 50; run.shift = -40; run.item = 0;
    req->media->rope.runs.push_back(run);
    f.add(std::move(req));

    /* Batch row q of a pass: the item row it must carry, or -1 for a text row. */
    auto check_pass = [&](const RadBatch* b, int64_t first_tok, int64_t shift, int64_t n) {
        int64_t want = 0;
        for (int64_t q = 0; q < n; ++q) {
            const int64_t ti = first_tok + q + shift;
            if (ti < 50 || ti >= 150) continue;
            REQUIRE(want < b->n_mm_rows);
            CHECK_EQ(b->mm_rows[want], (int32_t)q);
            const uint16_t* e = static_cast<const uint16_t*>(b->mm_embd) + want * 4;
            for (int c = 0; c < 4; ++c) CHECK_EQ((int)e[c], (int)((ti - 50) * 4 + c));
            ++want;
        }
        CHECK_EQ(b->n_mm_rows, want);
    };
    for (int chunk = 0; chunk < 4; ++chunk) {
        const RadBatch* b = f.sch.step();
        REQUIRE(b != nullptr);
        REQUIRE_EQ(b->n_tok, (int64_t)(chunk < 3 ? 64 : 8));
        check_pass(b, 64 * chunk, 0, b->n_tok);
        if (chunk == 0) {
            /* M-RoPE: a picture token at its grid cell, a text token before it at its index. */
            CHECK_EQ(b->rope_mixed, 1);
            const int64_t n = b->n_tok;
            CHECK_EQ(b->rope_pos[61], 50); CHECK_EQ(b->rope_pos[n + 61], 51);
            CHECK_EQ(b->rope_pos[2 * n + 61], 51);
            CHECK_EQ(b->rope_pos[49], 49); CHECK_EQ(b->rope_pos[2 * n + 49], 49);
            int rounds = 4;
            Scheduler::SerialRow rows[4];
            (void)f.sch.serial_begin(&rounds, rows, 4);
            const RadBatch* hb = f.sch.serial_batch(0, 1);
            REQUIRE(hb != nullptr);
            REQUIRE_EQ(hb->n_tok, (int64_t)64);
            check_pass(hb, 0, /*shift=*/1, 64);
            f.sch.serial_end(0);
        }
        f.commit_plain();
    }
}

/* ================================================================== scored rows (KL mode) */
/* In KL mode a scored prompt names a logits row at every position it prefills, after the
 * sampler's rows and in entry order, which is the order Engine::score_rows maps back. A row
 * miscounted here scores the wrong position against the reference and biases every comparison
 * the instrument makes. */
TEST(scored_prompt_rows_follow_the_sampler_rows_in_entry_order) {
    Fixture f;
    f.build();
    f.cfg.kld_record = "/nonexistent";             /* only kld() is read here */
    CHECK_OK(f.start());
    auto x = f.request(600, 4);
    x->score = true;
    const uint64_t xid = f.add(std::move(x));
    const uint64_t y = f.submit(40, 8);
    const RadBatch* b = f.sch.step();
    REQUIRE(b != nullptr);
    /* X's first chunk (0..255, no token) then Y's whole prompt (256..295, one sampled row). */
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)2);
    CHECK_EQ(f.sch.plan().e[0].seq, xid);
    REQUIRE_EQ(b->n_out, (int64_t)(1 + 256));
    CHECK_EQ(b->out_ids[0], 295);
    bool ok = true;
    for (int j = 0; j < 256; ++j) ok = ok && b->out_ids[1 + j] == j;
    CHECK(ok);
    f.commit_plain();

    /* Y decodes first (decode rows lead), then X's second chunk. */
    b = f.sch.step();
    REQUIRE(b != nullptr);
    REQUIRE_EQ(f.sch.plan().e.size(), (size_t)2);
    CHECK_EQ(f.sch.plan().e[0].seq, y);
    REQUIRE_EQ(b->n_out, (int64_t)(1 + 256));
    CHECK_EQ(b->out_ids[0], 0);
    ok = true;
    for (int j = 0; j < 256; ++j) ok = ok && b->out_ids[1 + j] == 1 + j;
    CHECK(ok);
}

/* ================================================================== the linear-state chunk walk */
/* state_chunk_bounds and state_chunk_seq against a per-token definition: tokens belong to the same
 * chunk exactly when they are in the same sequence and the same absolute multiple of C. A chunk
 * crossing a sequence or a multiple carries a partial tile's state into the next chunk, which
 * corrupts the recurrent state silently. A cap one short is refused without a write past it. */
TEST(the_state_chunk_walk_matches_a_per_token_definition) {
    const DeriveRow* bounds = derive_row(derive_index("state_chunk_bounds"));
    const DeriveRow* seqs = derive_row(derive_index("state_chunk_seq"));
    REQUIRE(bounds && seqs);
    const uint32_t seed = 0xC4A11u;
    uint32_t rng = seed;
    auto next = [&](uint32_t n) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) % n; };
    int bad = 0;
    for (int c = 0; c < 2000 && !bad; ++c) {
        const int64_t C = (int64_t)1 << (2 * next(4));          /* 1, 4, 16, 64 */
        const int n_seq = 1 + (int)next(6);
        std::vector<int32_t> cu(1, 0), pos;
        for (int s = 0; s < n_seq; ++s) {
            const int32_t p0 = (int32_t)next(300), len = (int32_t)next(5) == 0 ? 0 : (int32_t)next(80);
            for (int32_t j = 0; j < len; ++j) pos.push_back(p0 + j);
            cu.push_back(cu.back() + len);
        }
        /* The definition: a new chunk wherever (sequence, position / C) changes. */
        std::vector<int32_t> want_b(1, 0), want_s;
        for (int s = 0; s < n_seq; ++s)
            for (int32_t t = cu[(size_t)s]; t < cu[(size_t)s + 1]; ++t) {
                const bool last = t + 1 == cu[(size_t)s + 1] ||
                                  pos[(size_t)t + 1] / C != pos[(size_t)t] / C;
                if (last) { want_b.push_back(t + 1); want_s.push_back(s); }
            }
        DeriveInput in;
        in.cu_seqlens = cu.data();
        in.positions = pos.data();
        in.n_tok = (int64_t)pos.size();
        in.n_seq = n_seq;
        in.state_chunk = C;
        for (int which = 0; which < 2 && !bad; ++which) {
            const DeriveRow* row = which ? seqs : bounds;
            const std::vector<int32_t>& want = which ? want_s : want_b;
            const int64_t need = (int64_t)want.size();
            std::vector<int32_t> out((size_t)need + 8, 0x7ead);
            int64_t n = -1;
            const int st = row->fn(in, out.data(), need, &n);
            bool ok = st == RAD_OK && n == need &&
                      std::equal(want.begin(), want.end(), out.begin());
            if (need > 0) {
                std::vector<int32_t> canary((size_t)need + 8, 0x7ead);
                int64_t m = -1;
                ok = ok && row->fn(in, canary.data(), need - 1, &m) == RAD_E_FULL;
                for (size_t k = (size_t)need - 1; k < canary.size(); ++k)
                    ok = ok && canary[k] == 0x7ead;
            }
            if (!ok) {
                bad = 1;
                std::fprintf(stderr, "    seed 0x%x case %d: %s, C %lld, %d sequences, %zu tokens\n",
                             seed, c, row->name, (long long)C, n_seq, pos.size());
            }
        }
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== the pass's host bounds */
/* Every row of every build, issued ahead or not, has a block-table column for every position it
 * can write: the card writes slot -1 past the pitch, which silently drops a verified token's K/V.
 * And the two host-side bounds a recorded pass is keyed by, max_ctx_len and the pitch, cross
 * their buckets on the same builds; apart, each crossing re-records every pass twice. */
TEST(every_row_has_a_table_column_and_the_pass_bounds_cross_buckets_together) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, /*max_tok=*/512, /*max_ctx=*/16384);
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    auto r = f.request(1000, 0);
    r->speculation_fixed = true;
    const uint64_t id = f.add(std::move(r));
    uint32_t rng = 4242;
    int32_t last_ctx = -1;
    int64_t last_pitch = -1;
    int ahead_builds = 0, crossings = 0, bad_cover = 0, apart = 0;
    auto check = [&](const RadBatch* b) {
        const RadKVGroupBatch& g = b->kv[0];
        for (int64_t i = 0; i < b->n_seq; ++i) {
            if (g.block_table_pitch * g.block_size < (int64_t)b->ctx_lens[i] + b->q_lens[i]) ++bad_cover;
            if (b->ctx_lens[i] > b->max_ctx_len) ++bad_cover;
        }
        if (b->n_seq == 1 && b->q_lens[0] == 5) {        /* steady decode: one shape */
            if (last_ctx >= 0 && (b->max_ctx_len != last_ctx) != (g.block_table_pitch != last_pitch))
                ++apart;
            if (last_ctx >= 0 && b->max_ctx_len != last_ctx) ++crossings;
            last_ctx = b->max_ctx_len;
            last_pitch = g.block_table_pitch;
        }
    };
    int32_t draft[4] = { 31, 32, 33, 34 };
    bool open = false;                               /* a step issued ahead is current */
    for (int t = 0; t < 3000; ++t) {
        const RadBatch* b = nullptr;
        if (!open) {
            (void)f.sch.set_draft(id, draft, 4);
            b = f.sch.step();
            REQUIRE(b != nullptr);
            check(b);
        }
        open = false;
        const bool ahead = t % 3 == 0 && f.sch.plan().e[0].produces_token;
        const RadBatch* nb = nullptr;
        if (ahead) {
            f.sch.expect_draft(id, 4);
            nb = f.sch.step_next();
            if (nb) { check(nb); ++ahead_builds; }
        }
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
        if (nb) {
            CHECK_OK(f.sch.set_draft(id, draft, 4));
            f.sch.promote_next();
            open = true;
        }
    }
    CHECK_EQ(bad_cover, 0);
    CHECK_EQ(apart, 0);
    CHECK(crossings >= 4);
    CHECK(ahead_builds > 500);
}

/* ================================================================== a wider state index */
/* A state group whose kernel reads more than one column per sequence (the anchor scheme's
 * committed and scratch slots) gets the manager's whole row at the group's pitch on every build,
 * through rows reused after a request ends and sequences moving between rows. A stale or
 * misaligned column reads another sequence's recurrent state. */
TEST(a_wider_state_index_row_is_the_managers_row_on_every_build) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64);
    f.kv.sidx_width = 2;
    f.cfg.n_spec = 4;
    CHECK_OK(f.start());
    std::vector<uint64_t> ids;
    ids.push_back(f.submit(40, 9));
    ids.push_back(f.submit(70, 200));
    ids.push_back(f.submit(17, 90));
    uint32_t rng = 2468;
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
            if (f.kv.geom.g[(size_t)gi].paged) continue;
            REQUIRE_EQ(g.state_index_pitch, (int64_t)2);
            for (int64_t i = 0; i < b->n_seq; ++i) {
                int32_t want[2] = { -2, -2 };
                CHECK_EQ(f.kv.state_index_row(p.e[(size_t)i].seq, gi, want, 2), 2);
                CHECK_EQ(g.state_index[i * 2], want[0]);
                CHECK_EQ(g.state_index[i * 2 + 1], want[1]);
                CHECK(want[0] >= 0);
                ++checked;
            }
        }
        CHECK_OK(f.commit(5, 0, [&](const StepEntry& e) { return random_accept(rng, e); }));
    }
    CHECK(checked > 100);
}

RAD_TEST_MAIN()

/* ================================================================== the plan without a span */
namespace {

/* A 64-bit FNV-1a fingerprint of a step: every plan entry the batch builder reads and every array
 * of the RadBatch it built, each over the extent the step fills. */
struct StepFp {
    uint64_t h = 1469598103934665603ull;
    void raw(const void* p, size_t n) {
        const auto* c = (const unsigned char*)p;
        for (size_t i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
    }
    void i(int64_t v) { raw(&v, sizeof v); }
    void arr(const int32_t* p, int64_t n) {
        i(p ? n : -1);
        if (p) for (int64_t k = 0; k < n; ++k) i(p[k]);
    }
    void step(const StepPlan& p, const RadBatch& b) {
        i((int64_t)p.e.size());
        for (const StepEntry& e : p.e) {
            i((int64_t)e.seq); i(e.n_tokens); i(e.ctx_len); i(e.n_spec); i(e.is_prefill);
            i(e.ckpt_tok); i(e.ckpt_slot); i(e.ckpt_restore);
        }
        i(b.phase); i(b.n_tok); i(b.n_seq); i(b.n_out); i(b.n_ahead); i(b.max_q_len);
        i(b.max_ctx_len); i(b.n_seq_decode); i(b.n_tok_decode); i(b.n_checkpoints);
        arr(b.token_ids, b.n_tok + b.n_ahead);
        arr(b.positions, b.n_tok);
        arr(b.cu_seqlens, b.n_seq + 1);
        arr(b.q_lens, b.n_seq);
        arr(b.ctx_lens, b.n_seq);
        arr(b.out_ids, b.n_out);
        arr(b.checkpoint_tok, b.n_checkpoints);
        arr(b.checkpoint_slot, b.n_checkpoints);
        i(b.n_kv_groups);
        for (int g = 0; g < b.n_kv_groups; ++g) {
            const RadKVGroupBatch& k = b.kv[g];
            arr(k.slot_mapping, k.slot_mapping ? b.n_tok : 0);
            arr(k.seqused, k.seqused ? b.n_seq : 0);
            i(k.block_table_pitch);
            arr(k.block_table, k.block_table ? b.n_seq * k.block_table_pitch : 0);
            i(k.state_index_pitch);
            arr(k.state_index, k.state_index ? b.n_seq * std::max<int64_t>(k.state_index_pitch, 1)
                                             : 0);
        }
    }
};

/* Long and short prompts arriving while others prefill and decode, every step fingerprinted. */
uint64_t scripted_steps(int64_t max_tok, int64_t interval, int64_t busy, int* n_steps) {
    Fixture f;
    f.build(/*blocks=*/4096, /*states=*/64, max_tok, /*max_ctx=*/8192);
    f.cfg.checkpoint_interval = interval;
    f.cfg.busy_prefill_chunk = busy;
    CHECK_OK(f.start());
    struct Arrival { int at, n_prompt, max_tokens; };
    const Arrival arrivals[] = { { 0, 3000, 6 }, { 0, 200, 9 }, { 3, 1500, 4 }, { 5, 40, 20 },
                                 { 9, 2600, 3 }, { 12, 700, 5 }, { 30, 5000, 2 } };
    StepFp fp;
    *n_steps = 0;
    for (int k = 0; k < 600; ++k) {
        for (const Arrival& a : arrivals)
            if (a.at == k) CHECK(f.submit(a.n_prompt, a.max_tokens) != 0);
        const RadBatch* b = f.sch.step();
        if (!b) {
            if (k > 30) break;
            continue;
        }
        fp.step(f.sch.plan(), *b);
        ++*n_steps;
        f.commit_plain();
    }
    return fp.h;
}

}  /* namespace */

/* Without a span the scheduler plans and stages the pinned steps: the fingerprints of a scripted
 * workload's every step (plan entries and RadBatch arrays) at budgets of one, two and four
 * intervals, with --busy-prefill-chunk auto, off and fixed, taken from the scheduler as it was
 * before --prefill-span. */
TEST(without_a_span_the_scheduler_plans_and_stages_the_pinned_steps) {
    struct Case { int64_t max_tok, interval, busy; uint64_t want; };
    const Case cases[] = {
        { 256, 256, -1, 0x17b8231a20db9d46ull }, { 512, 256, -1, 0xe155e72b6a4cc94full }, { 1024, 256, -1, 0x54ba52dac3f4f61cull },
        { 1024, 256, 0, 0x54ba52dac3f4f61cull }, { 1024, 256, 128, 0x3d9cbbf533e3c4b5ull }, { 2048, 512, -1, 0x400205b7b9afcf12ull },
    };
    for (const Case& c : cases) {
        int n = 0;
        const uint64_t h = scripted_steps(c.max_tok, c.interval, c.busy, &n);
        fprintf(stderr, "    pin: max_tok %lld interval %lld busy %lld: %d steps, 0x%016llxull\n",
                (long long)c.max_tok, (long long)c.interval, (long long)c.busy, n,
                (unsigned long long)h);
        CHECK(n > 20);
        CHECK_EQ(h, c.want);
    }
}
