/* runtime_test.cpp -- the run phase, against a hand-built Program and a kernel plugin that
 * records its calls.
 *
 * Building the Program by hand keeps each failure attributable: through a real builder, a bucket
 * table indexed off by one and a declare phase that built wrong boundaries look the same.
 *
 * The device layer is the real one, on the host backend: device memory is host memory and each
 * stream is a worker thread, so ordering bugs show as races. "A pending weight causes a stream
 * wait, not a host sync" is checked through Ctx::stream_waits() and the device layer's count of
 * host blocks (rad_dev_host_blocks).
 */
#include "rad_test.h"

#include "device/device.h"
#include "runtime/arena.h"
#include "plugin/registry.h"
#include "runtime/ctx.h"
#include "runtime/kvbind.h"
#include "mem/kv.h"
#include "runtime/rank_barrier.h"
#include "runtime/residency.h"
#include "runtime/wtab.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <thread>

using rad::Ctx;
using rad::CtxDesc;
using rad::Program;

/* ================================================================== the fake kernel plugin */
/* Records what it was called with. Its push_back allocates, which is fine: the rule is that the
 * runtime does not allocate on the step path, and this stands in for a kernel launch. */
struct Rec {
    const char* kernel = nullptr;
    long long   sk = 0;   /* a tuned axis, read the way a real kernel reads one */
    RadStream   stream = nullptr;
    int         n_t = 0;
    RadTensor   t[8]{};
    long long   M = -1;
    int         rank = 0, world = 0;
    void*       scratch = nullptr;
    int64_t     scratch_bytes = 0;
    void*       instance = nullptr;
};

static std::vector<Rec> g_rec;

static int record(const char* name, const RadArgs* a, RadStream s) {
    Rec r;
    r.kernel = name;
    rad_args_geti(a, "sk", &r.sk);
    r.stream = s;
    r.n_t = a->n_t;
    for (int i = 0; i < a->n_t && i < 8; ++i) r.t[i] = a->t[i];
    rad_args_geti(a, "M", &r.M);
    r.rank = a->rank;
    r.world = a->world_size;
    r.scratch = a->scratch;
    r.scratch_bytes = a->scratch_bytes;
    r.instance = a->instance;
    g_rec.push_back(r);
    return RAD_OK;
}

static int k_m16 (const RadArgs* a, RadStream s) { return record("m16",  a, s); }
static int k_m64 (const RadArgs* a, RadStream s) { return record("m64",  a, s); }
static int k_pre (const RadArgs* a, RadStream s) { return record("prefill", a, s); }
static int k_bad (const RadArgs* a, RadStream s) { record("bad", a, s); return RAD_E_SHAPE; }
static int k_null(const RadArgs*,   RadStream)            { return RAD_OK; }   /* for the cost probe */

/* ================================================================== the fake residency table */
/* A slot per weight handle below slot.size(); none beyond. */
struct TableResidency : rad::ResidencyTable {
    std::vector<rad::WeightSlot> slot;
    mutable int                  lookups = 0;
    uint64_t                     ep = 1;

    std::vector<rad_weight>      changed;

    const rad::WeightSlot* lookup(rad_weight w) const override {
        ++lookups;
        return (size_t)w < slot.size() ? &slot[(size_t)w] : nullptr;
    }
    uint64_t epoch() const override { return ep; }
    const std::vector<rad_weight>& changes() const override { return changed; }
    void clear_changes() override { changed.clear(); }
    /* What the mover does: a new pointer, a new generation, and the weight named as changed. */
    void move(rad_weight w, void* p) {
        slot[(size_t)w].ptr = p;
        ++slot[(size_t)w].generation;
        ++ep;
        changed.push_back(w);
    }
};

/* ================================================================== the fixture */
static std::function<void(Ctx&)> g_body;
static void arch_step_trampoline(RadCtx* c, const RadBatch*) {
    if (g_body) g_body(*static_cast<Ctx*>(c));
}

/* The gemm_nt schema, three positional operands. Static so the pointer outlives the Program. */
static const RadParamSpec   g_gemm_params[] = {
    { "M", RAD_P_INT, RAD_REQUIRED }, { "N", RAD_P_INT, RAD_REQUIRED },
    { "K", RAD_P_INT, RAD_REQUIRED }, { "dtype", RAD_P_STR, RAD_REQUIRED },
};
static const RadOperandSpec g_gemm_opds[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOpSchema g_gemm_schema = {
    "gemm_nt", g_gemm_params, 4, g_gemm_opds, 3, "A[M,K] x B[N,K]^T -> Y[M,N]"
};

/* The same GEMM over a table of experts' weights. */
static const RadOperandSpec g_moe_opds[] = {
    { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WTAB, 0 }, { "y", RAD_OPD_OUT, 0 },
};
static const RadOpSchema g_moe_schema = {
    "moe_gemm", g_gemm_params, 4, g_moe_opds, 3, "A[M,K] x B[e][N,K]^T -> Y[M,N]"
};

static uint16_t g_weight_bytes[64];   /* somewhere for WeightInfo::ptr to point */
static uint16_t g_moved_bytes[64];    /* where a "relocated" weight lives */

struct Fixture {
    Program                    P;
    std::deque<RadKernelInfo>  infos;
    std::deque<rad::KernelRow> rows;
    std::vector<rad::KVPoolBinding> kv;      /* CtxDesc::kv */
    rad::Registry*             registry = nullptr;
    Ctx                        ctx;
    RadBatch                   batch{};

    Fixture() {
        P.meta.n_layers = 2;
        P.meta.n_expert = 4;
        P.arena_bytes   = 8192;
        P.scratch_bytes = 256;
        P.weights.resize(1);   /* handles are 1-based; index 0 is the sentinel */
        P.buffers.resize(1);
        P.ops.resize(1);
    }

    rad_weight add_weight(const char* name, uint32_t dt, int64_t d0, int64_t d1, void* ptr) {
        rad::WeightInfo w;
        w.name = name;
        w.decl.dtype = dt;
        w.decl.rank = 2;
        w.decl.shape[0] = d0;
        w.decl.shape[1] = d1;
        w.ptr = ptr;
        P.weights.push_back(w);
        return (rad_weight)(P.weights.size() - 1);
    }

    rad_buf add_buffer(const char* name, uint32_t dt, int64_t d0, int64_t d1, int64_t off) {
        rad::BufferInfo b;
        b.name = name;
        b.decl.dtype = dt;
        b.decl.rank = 2;
        b.decl.shape[0] = d0;
        b.decl.shape[1] = d1;
        b.decl.domain = RAD_DOMAIN_DEVICE;
        b.bytes = rad_dtype_bytes(dt, d0 * d1);
        b.arena_offset = off;
        P.buffers.push_back(b);
        return (rad_buf)(P.buffers.size() - 1);
    }

    /* One op, one band per (hi, launch, kernel name). A null launch leaves that band unresolved,
     * like a hole in the bucket table. */
    rad_op add_op(const char* op, const char* ranged_key,
                  const std::vector<int64_t>& hi,
                  const std::vector<RadLaunchFn>& launch,
                  const std::vector<const char*>& kname,
                  const RadOpSchema* schema,
                  const std::vector<rad_weight>& weights) {
        rad::OpInfo o;
        o.op = op;
        o.index = (int32_t)P.ops.size();
        o.ranged_key = ranged_key;
        o.range_lo = 1;
        o.range_hi = hi.empty() ? 0 : hi.back();
        o.schema = schema;
        o.weights = weights;
        o.base.set_i("N", 8);
        o.base.set_i("K", 4);
        o.base.set_s("dtype", "bf16");

        for (size_t i = 0; i < hi.size(); ++i) {
            rad::Band b;
            b.hi = hi[i];
            b.miss[RAD_DOMAIN_HOST] = "the fake plugin has no host kernels";
            if (launch[i]) {
                infos.push_back(RadKernelInfo{});
                RadKernelInfo& ki = infos.back();
                ki.name   = kname[i];
                ki.op     = op;
                ki.family = "test";
                ki.domain = RAD_DOMAIN_DEVICE;
                ki.launch = launch[i];

                rows.push_back(rad::KernelRow{});
                rad::KernelRow& kr = rows.back();
                kr.info = &ki;
                kr.plugin = "fake";
                kr.index = (int)i;

                rad::Resolved& r = b.dom[RAD_DOMAIN_DEVICE];
                r.row = &kr;
                r.scratch_bytes = 128;
                r.tune_source = "axis defaults";
                r.geom.set_i(ranged_key, hi[i]);
                r.geom.set_i("N", 8);
                r.geom.set_i("K", 4);
                r.geom.set_s("dtype", "bf16");
            } else {
                b.miss[RAD_DOMAIN_DEVICE] = "no kernel serves this band";
            }
            o.bands.push_back(b);
        }
        P.ops.push_back(o);
        return (rad_op)(P.ops.size() - 1);
    }

    int start(rad::ResidencyTable* res = nullptr, bool profile = false, bool record = false) {
        CtxDesc d;
        d.program     = &P;
        d.rank        = 1;
        d.world_size  = 2;
        d.arch_step   = arch_step_trampoline;
        d.residency   = res;
        d.profile_ops = profile;
        d.kv          = kv;
        d.registry    = registry;
        /* The arch bodies here issue whatever a case asks, which the batch does not describe, so
         * no pass may be recorded and replayed under the batch's key (a card records passes; the
         * host backend cannot). A case about the recording decision itself asks for it. */
        d.record_passes = record;
        return ctx.init(d);
    }

    int step(int n_tok, int step_no, const std::function<void(Ctx&)>& body) {
        batch.phase = n_tok > 1 ? RAD_PHASE_PREFILL : RAD_PHASE_DECODE;
        batch.step  = step_no;
        batch.n_tok = n_tok;
        batch.n_seq = 1;
        g_body = body;
        int s = ctx.run_step(&batch);
        g_body = nullptr;
        return s;
    }
};

/* ================================================================== tests */

/* Bucket boundaries are the constraint values and a band is inclusive at its upper bound (spec
 * §2.2). Off by one here means a prefill kernel running at M=1, a loss invisible in the output. */
TEST(bucket_table_indexes_at_band_boundaries) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 16, 64, 8192 },
                         { k_m16, k_m64, k_pre }, { "m16", "m64", "prefill" },
                         &g_gemm_schema, { w });
    CHECK_OK(f.start());

    const int64_t probes[] = { 1, 15, 16, 17, 63, 64, 65, 8191, 8192 };
    const char*   want[]   = { "m16", "m16", "m16", "m64", "m64", "m64",
                               "prefill", "prefill", "prefill" };

    g_rec.clear();
    CHECK_OK(f.step(1, 0, [&](Ctx& c) {
        for (int64_t n : probes) {
            RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
            CHECK_OK(c.issue(op, opd, 3, n));
        }
    }));

    CHECK_EQ(g_rec.size(), sizeof(probes) / sizeof(probes[0]));
    for (size_t i = 0; i < g_rec.size() && i < sizeof(want) / sizeof(want[0]); ++i) {
        CHECK_EQ(std::string(g_rec[i].kernel), std::string(want[i]));
        /* The geometry the kernel sees is the band's, collapsed to its upper bound. */
        CHECK_EQ(g_rec[i].M, (long long)(i < 3 ? 16 : i < 6 ? 64 : 8192));
    }

    /* RAD_N_BATCH means the batch's token count, which almost every op wants. */
    g_rec.clear();
    CHECK_OK(f.step(40, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(std::string(g_rec[0].kernel), std::string("m64"));

    /* The rank's identity reaches the kernel, since collectives need it (spec §9). */
    CHECK_EQ(g_rec[0].rank, 1);
    CHECK_EQ(g_rec[0].world, 2);
    CHECK_EQ(g_rec[0].scratch_bytes, (int64_t)128);
    CHECK(g_rec[0].scratch == f.ctx.scratch());
    CHECK(g_rec[0].stream == f.ctx.stream());
}

/* rad_buf_ptr is arena_base + BufferInfo::arena_offset, and an operand offset is element
 * arithmetic on top of it; nothing is allocated and no table is walked. */
TEST(arena_offsets_and_operand_offsets) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    char* base = (char*)f.ctx.arena_base();
    CHECK(base != nullptr);
    CHECK((char*)f.ctx.buf_ptr(x) == base + 0);
    CHECK((char*)f.ctx.buf_ptr(y) == base + 1024);
    CHECK(f.ctx.weight_ptr(w) == (void*)g_weight_bytes);

    /* The scratch region is at the tail, past the plan, page-aligned. */
    CHECK((char*)f.ctx.scratch() >= base + 8192);
    CHECK(((uintptr_t)f.ctx.scratch() & (RAD_ALIGN_UNIT - 1)) == 0);

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        /* y offset by 8 bf16 elements = 16 bytes, one row of the [64, 8] plane. The row count
         * must come down to 63 with it: an offset does not shrink the extents, so
         * `RAD_B_AT(y, 8)` alone describes 64 rows starting one row in, reaching eight elements
         * past the buffer, which the runtime refuses by name (core/runtime/issue.cpp). */
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B_N(y, 8, 63) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(g_rec[0].n_t, 3);
    CHECK(g_rec[0].t[0].data == base + 0);
    CHECK(g_rec[0].t[1].data == (void*)g_weight_bytes);
    CHECK(g_rec[0].t[2].data == base + 1024 + 16);

    /* Shapes and strides come off the declaration; dtype off the weight's stored layout. */
    CHECK_EQ(g_rec[0].t[0].shape[0], (int64_t)64);
    CHECK_EQ(g_rec[0].t[0].shape[1], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[0], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[1], (int64_t)1);
    CHECK_EQ(g_rec[0].t[1].shape[0], (int64_t)8);
    CHECK_EQ(g_rec[0].t[1].shape[1], (int64_t)4);
    CHECK_EQ((int)g_rec[0].t[1].dtype, (int)RAD_BF16);
}

/* rows > 0 narrows dim 0 and leaves every stride alone. That is how a prefill chunk of `rows`
 * tokens issues against a buffer sized at max_tok; it is only sound on the outermost dimension,
 * which is why the ABI narrows dim 0 and nothing else. */
TEST(rows_narrows_dim0_and_keeps_strides) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    g_rec.clear();
    CHECK_OK(f.step(7, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B_N(x, 0, 7), RAD_W(w), RAD_B_N(y, 0, 7) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(g_rec[0].t[0].shape[0], (int64_t)7);
    CHECK_EQ(g_rec[0].t[0].shape[1], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[0], (int64_t)4);   /* unchanged: it is a product of dims 1.. */
    CHECK_EQ(g_rec[0].t[0].stride[1], (int64_t)1);
    CHECK_EQ(g_rec[0].t[2].shape[0], (int64_t)7);
    CHECK_EQ(g_rec[0].t[2].shape[1], (int64_t)8);
    CHECK_EQ(g_rec[0].t[2].stride[0], (int64_t)8);
    /* The buffer itself is still the full max_tok extent; only this view is narrow. */
    CHECK_EQ(f.P.buffers[x].decl.shape[0], (int64_t)64);
}

/* A kernel returning negative at issue is a selection bug, not a runtime condition: the step
 * aborts, its requests fail, the kernel and geometry are named, and nothing falls back silently
 * (spec §17). */
TEST(negative_launch_aborts_the_step_by_name) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op good = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    rad_op bad  = f.add_op("gemm_nt", "M", { 8192 }, { k_bad }, { "wrongshape" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    g_rec.clear();
    int second = RAD_OK;
    int s = f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        c.issue(bad, opd, 3, RAD_N_BATCH);
        /* rad_arch_step is void and cannot know the step is lost, so it keeps going. The rest of
         * the step must not launch: the output is already wrong, and the first error would be
         * buried under many more. */
        second = c.issue(good, opd, 3, RAD_N_BATCH);
    });

    CHECK_EQ(s, RAD_E_SHAPE);
    CHECK_EQ(second, RAD_E_SHAPE);
    CHECK_EQ(g_rec.size(), (size_t)1);              /* only the failing kernel ever ran */
    CHECK_EQ(std::string(g_rec[0].kernel), std::string("bad"));

    const char* msg = f.ctx.step_error();
    CHECK(std::strstr(msg, "wrongshape") != nullptr);          /* the kernel, by name */
    CHECK(std::strstr(msg, "fake") != nullptr);                /* and the plugin it came from */
    CHECK(std::strstr(msg, "gemm_nt") != nullptr);             /* the op */
    CHECK(std::strstr(msg, "N=8") != nullptr);                 /* the geometry */
    CHECK(std::strstr(msg, "K=4") != nullptr);
    CHECK(std::strstr(msg, "M=8192") != nullptr);

    /* The next step starts clean: the failure is per step, not per Ctx. */
    g_rec.clear();
    CHECK_OK(f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(good, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
}

/* A weight the mover has in flight is waited on with a stream wait, once per generation, and the
 * host never blocks. This lets placement be dynamic without serialising the step (spec §5.4). */
TEST(pending_weight_waits_on_the_stream_not_the_host) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.expert3.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });

    TableResidency res;
    res.slot.assign(f.P.weights.size(), rad::WeightSlot{});
    rad::WeightSlot& slot = res.slot[(size_t)w];
    RadEvent moving = nullptr;
    CHECK_OK(rad_event_create(&moving));
    slot.ptr        = g_moved_bytes;        /* the mover relocated it out from under WeightInfo */
    slot.ready      = moving;               /* and the transfer is still in flight */
    slot.generation = 1;

    CHECK_OK(f.start(&res));

    const int64_t blocks0 = rad::rad_dev_host_blocks();

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));   /* same generation, second time */
    }));

    /* One wait, not two: a stream wait orders everything issued after it, so repeating it per
     * issue would add thousands of runtime calls per step for nothing. */
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)1);
    /* No host synchronisation of any kind: none the runtime counts (Ctx::host_syncs), and none
     * that reached the device layer by a route the runtime does not count. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)0);
    CHECK_EQ(rad::rad_dev_host_blocks() - blocks0, (int64_t)0);
    /* The residency table, not WeightInfo, decided where the bytes are. */
    CHECK_EQ(g_rec.size(), (size_t)2);
    CHECK(g_rec[0].t[1].data == (void*)g_moved_bytes);
    CHECK(g_rec[0].t[1].data != (void*)g_weight_bytes);

    /* A new transfer into the slot bumps the generation, which re-arms the wait. */
    slot.generation = 2;
    CHECK_OK(f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)2);

    /* A settled weight (no event) costs no wait, the common case even under expert offload since
     * a hot expert stays put for many steps. */
    slot.ready = nullptr;
    slot.generation = 3;
    CHECK_OK(f.step(4, 2, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)2);

    f.ctx.shutdown();
    rad_event_destroy(moving);
}

/* With no residency table (all in VRAM: everything on tier 0, the mover never runs) the lookup
 * collapses to WeightInfo::ptr and there is nothing to wait on. The common case must not pay for
 * the dynamic one. */
TEST(all_in_vram_collapses_to_weightinfo_ptr) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start(nullptr));

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(f.ctx.stream_waits(), (int64_t)0);
    CHECK(g_rec[0].t[1].data == (void*)g_weight_bytes);
}

/* A weight table resolves only the entries that moved. A routed layer's table is its whole expert
 * set and the mover changes a few experts per step, so a full re-resolve would be tens of
 * thousands of lookups. The table resolves the entries the residency table names and patches the
 * device array. */
TEST(a_weight_table_resolves_only_the_entries_that_moved) {
    static uint16_t expert_bytes[4][32], moved_expert[32];
    Fixture f;
    std::vector<rad_weight> w;
    for (int e = 0; e < 4; ++e)
        w.push_back(f.add_weight(("blk.0.e" + std::to_string(e)).c_str(), RAD_BF16, 8, 4,
                                 expert_bytes[e]));
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op  op = f.add_op("moe_gemm", "M", { 8192 }, { k_pre }, { "prefill" }, &g_moe_schema, w);

    TableResidency res;
    res.slot.assign(f.P.weights.size(), rad::WeightSlot{});
    for (int e = 0; e < 4; ++e) res.slot[(size_t)w[(size_t)e]].ptr = expert_bytes[e];
    CHECK_OK(f.start(&res));

    auto issue_twice = [&](int step_no) {
        g_rec.clear();
        CHECK_OK(f.step(4, step_no, [&](Ctx& c) {
            RadOperand opd[3] = { RAD_B(x), RAD_WTAB(w[0], 4), RAD_B(y) };
            CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
            CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
        }));
    };
    /* What the kernel would read: the device table, copied back through the step's stream so the
     * check holds on the fakes and on a real backend. */
    auto entries = [&]() {
        std::vector<void*> v(4, nullptr);
        CHECK_OK(rad_memcpy_async(v.data(), g_rec[0].t[1].data, 4 * sizeof(void*), f.ctx.stream()));
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        return v;
    };

    issue_twice(0);
    CHECK_EQ(res.lookups, 4);                      /* the first issue walked, the second did not */
    std::vector<void*> v = entries();
    for (int e = 0; e < 4; ++e) CHECK(v[(size_t)e] == (void*)expert_bytes[e]);

    issue_twice(1);                                /* nothing moved: no walk */
    CHECK_EQ(res.lookups, 4);

    /* The mover relocates expert 2 and says so: one entry is resolved, not four. */
    res.move(w[2], moved_expert);
    issue_twice(2);
    CHECK_EQ(res.lookups, 5);
    v = entries();
    CHECK(v[2] == (void*)moved_expert);
    CHECK(v[1] == (void*)expert_bytes[1]);
    CHECK(v[3] == (void*)expert_bytes[3]);

    /* Moved and moved back before the next issue: resolved once more, and the device array ends
     * where the residency table says. */
    res.move(w[1], moved_expert);
    res.move(w[1], expert_bytes[1]);
    issue_twice(3);
    v = entries();
    CHECK(v[1] == (void*)expert_bytes[1]);
    CHECK(v[2] == (void*)moved_expert);
}

/* A table's run is fixed by the declaration. The weights declared at a table position (OpInfo::
 * weight_opd) are bound as one table before the first step. A declared run with a gap is refused
 * there (the kernel strides from one base, so a gap reads the wrong expert), and an issue naming
 * any other run is refused at issue, by name. */
TEST(a_weight_table_issued_against_its_declaration_is_refused) {
    static uint16_t expert_bytes[4][32];
    auto experts = [](Fixture& f) {
        std::vector<rad_weight> w;
        for (int e = 0; e < 4; ++e)
            w.push_back(f.add_weight(("blk.0.e" + std::to_string(e)).c_str(), RAD_BF16, 8, 4,
                                     expert_bytes[e]));
        return w;
    };

    {
        /* Experts 0, 1 and 3 at the table, 2 missing from the run: refused when the tables are
         * bound, so the Ctx never runs a step. */
        Fixture f;
        const std::vector<rad_weight> w = experts(f);
        f.add_buffer("x", RAD_BF16, 64, 4, 0);
        f.add_buffer("y", RAD_BF16, 64, 8, 1024);
        rad_op op = f.add_op("moe_gemm", "M", { 8192 }, { k_pre }, { "prefill" }, &g_moe_schema,
                             { w[0], w[1], w[3] });
        f.P.ops[op].weight_opd = { 1, 1, 1 };
        CHECK_EQ(f.start(), RAD_E_INVAL);
    }

    Fixture f;
    const std::vector<rad_weight> w = experts(f);
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("moe_gemm", "M", { 8192 }, { k_pre }, { "prefill" }, &g_moe_schema, w);
    f.P.ops[op].weight_opd = { 1, 1, 1, 1 };
    CHECK_OK(f.start());

    /* The declared run is served. */
    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_WTAB(w[0], 4), RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);

    /* One entry short, and the same length one expert along: both name experts the declaration
     * did not put at this table, and neither reaches the kernel. */
    int step_no = 1;
    for (const RadOperand tab : { RAD_WTAB(w[0], 3), RAD_WTAB(w[1], 3) }) {
        g_rec.clear();
        const int s = f.step(4, step_no++, [&](Ctx& c) {
            RadOperand opd[3] = { RAD_B(x), tab, RAD_B(y) };
            c.issue(op, opd, 3, RAD_N_BATCH);
        });
        CHECK_EQ(s, RAD_E_INVAL);
        CHECK_EQ(g_rec.size(), (size_t)0);
        CHECK(std::strstr(f.ctx.step_error(), "A table's run is fixed by the declaration") !=
              nullptr);
    }
}

/* Routing never blocks the compute stream: the histogram is copied back asynchronously and the
 * heat engine drains it on a later step (spec §5.5). */
TEST(route_report_is_asynchronous_and_drains_oldest_first) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    CHECK_OK(f.start());

    int32_t counts_a[4] = { 7, 0, 3, 1 };
    int32_t counts_b[4] = { 1, 1, 1, 9 };
    RadRouting r{};
    r.n_expert = 4;
    r.top_k = 2;

    const int64_t blocks0 = rad::rad_dev_host_blocks();

    /* Nothing reported, so the drain says there is nothing rather than handing back a bank of the
     * zeros it was allocated with. */
    CHECK_EQ(f.ctx.next_histogram(0, -1, 0), -1);

    r.expert_count = counts_a;
    CHECK_OK(f.step(4, 0, [&](Ctx& c) { CHECK_OK(c.route_report(0, &r)); }));
    r.expert_count = counts_b;
    CHECK_OK(f.step(4, 1, [&](Ctx& c) { CHECK_OK(c.route_report(0, &r)); }));

    /* Both steps survive a consumer that did not look between them, oldest first: the consumer
     * runs at the host's pace and histograms arrive at the card's. A bank is identified by its
     * step, not its index, since the ring's depth is hidden. Spin on the drain: a bank becomes
     * readable when its event completes, not when run_step returns. */
    int b0 = -1;
    while ((b0 = f.ctx.next_histogram(0, -1, 2)) < 0) std::this_thread::yield();
    CHECK_EQ(f.ctx.last_histogram_step(0, b0), 0);
    const int32_t* h0 = f.ctx.last_histogram(0, b0);
    CHECK(h0 != nullptr);
    if (h0) { CHECK_EQ(h0[0], 7); CHECK_EQ(h0[2], 3); }

    int b1 = -1;
    while ((b1 = f.ctx.next_histogram(0, 0, 2)) < 0) std::this_thread::yield();
    CHECK(b1 != b0);
    CHECK_EQ(f.ctx.last_histogram_step(0, b1), 1);
    const int32_t* h1 = f.ctx.last_histogram(0, b1);
    CHECK(h1 != nullptr);
    if (h1) CHECK_EQ(h1[3], 9);

    /* Past the newest there is nothing, which is what ends a caller's drain loop. */
    CHECK_EQ(f.ctx.next_histogram(0, 1, 2), -1);

    /* No host synchronisation anywhere in that. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)0);
    CHECK_EQ(rad::rad_dev_host_blocks() - blocks0, (int64_t)0);

    /* The routing buffers themselves are recorded for the mover, not read. */
    const RadRouting* rec = f.ctx.last_routing(0);
    CHECK(rec != nullptr);
    if (rec) CHECK_EQ(rec->n_expert, (int64_t)4);

    /* A layer the model does not have is refused rather than corrupting a neighbour's bank. */
    CHECK_EQ(f.ctx.route_report(99, &r), RAD_E_INVAL);
}

/* A step that drafts reports the same layer once per pass, all into the step's one bank. The drain
 * runs before every pass, so it must not take that bank until the step is over; taken early, it
 * would cover only the passes so far and miss the rest. */
TEST(a_bank_is_credited_once_for_every_pass_of_its_step) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    CHECK_OK(f.start());

    int32_t counts[4] = { 2, 0, 5, 1 };
    RadRouting r{};
    r.n_expert = 4;
    r.top_k = 2;
    r.expert_count = counts;

    const auto report = [&](Ctx& c) { CHECK_OK(c.route_report(1, &r)); };
    const auto landed = [&](int after, int before) {
        /* The copy is asynchronous; wait for it the way the drain would see it land. */
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        return f.ctx.next_histogram(1, after, before);
    };

    CHECK_OK(f.step(4, 7, report));
    CHECK_EQ(landed(-1, 7), -1);           /* step 7 is still being issued */
    CHECK_OK(f.step(1, 7, report));
    CHECK_EQ(landed(-1, 7), -1);
    CHECK_OK(f.step(1, 7, report));

    const int b = landed(-1, 8);           /* the next step's dispatch */
    CHECK(b >= 0);
    CHECK_EQ(f.ctx.last_histogram_step(1, b), 7);
    CHECK_EQ(f.ctx.last_histogram_reports(1, b), 3);
    f.ctx.mark_credited(1, b);
    CHECK_EQ(landed(7, 8), -1);

    /* Three passes of one step are not two losses. A bank recycled for a later step before anyone
     * read it is one. */
    CHECK_EQ(f.ctx.route_dropped(), (uint64_t)0);
    CHECK_OK(f.step(1, 9, report));
    CHECK_OK(f.step(1, 9 + rad::kRouteBanks, report));
    CHECK_EQ(f.ctx.route_dropped(), (uint64_t)1);
}

/* Every event the context creates at bind is one Ctx::shutdown destroys: the routing ring holds
 * one per bank, not two. */
TEST(shutdown_destroys_every_event_the_context_created) {
    const int64_t before = rad::rad_dev_live_events();
    {
        Fixture f;
        f.add_buffer("x", RAD_BF16, 64, 4, 0);
        CHECK_OK(f.start());
        CHECK(rad::rad_dev_live_events() >= before + rad::kRouteBanks);
    }
    CHECK_EQ(rad::rad_dev_live_events(), before);
}

/* A weight operand is held to the same fit as a buffer one: a slice may name fewer rows than the
 * declaration, or start further in, and never reach past its end. */
TEST(a_weight_slice_that_reaches_past_the_weight_is_refused) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    RadOperand half = RAD_W(w);
    half.offset = 4 * 4;               /* rows 4..7 */
    half.rows   = 4;
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), half, RAD_B(y) };
        CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));

    RadOperand past = half;
    past.offset = 5 * 4;               /* rows 5..8: one row past the declaration */
    const int s = f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), past, RAD_B(y) };
        c.issue(op, opd, 3, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_SHAPE);
    CHECK(std::strstr(f.ctx.step_error(), "reaches past weight 'blk.0.w'") != nullptr);
}

/* Two kernels using the shared scratch region, one on each lane, with no join between them: the
 * second is ordered behind the first. Each launch below queues its work on the stream it is
 * given, so an unordered pair really overlaps and an early reader would see the poison. */
static const int64_t kScrBytes = 4 << 20;
static uint8_t* g_scr_src = nullptr;
static uint8_t* g_scr_dst = nullptr;

static int k_scr_write(const RadArgs* a, RadStream s) {
    rad_memset_async(a->scratch, 0xEE, kScrBytes, s);
    for (int i = 0; i < 8; ++i) rad_memcpy_async(a->scratch, g_scr_src, kScrBytes, s);
    return record("scr_write", a, s);
}
static int k_scr_read(const RadArgs* a, RadStream s) {
    rad_memcpy_async(g_scr_dst, a->scratch, kScrBytes, s);
    return record("scr_read", a, s);
}

TEST(scratch_users_on_two_lanes_are_ordered_one_behind_the_other) {
    Fixture f;
    f.P.scratch_bytes = kScrBytes;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op wr = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_write }, { "scr_write" }, nullptr, {});
    rad_op rd = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_read }, { "scr_read" }, nullptr, {});
    f.P.ops[wr].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    f.P.ops[rd].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    CHECK_OK(f.start());

    g_scr_src = (uint8_t*)rad_dev_alloc(kScrBytes, RAD_MEM_DEVICE);
    g_scr_dst = (uint8_t*)rad_dev_alloc(kScrBytes, RAD_MEM_DEVICE);
    CHECK(g_scr_src && g_scr_dst);
    if (!g_scr_src || !g_scr_dst) return;

    for (int round = 0; round < 4; ++round) {
        const int first = round & 1;   /* which lane writes; the other reads */
        for (int64_t i = 0; i < kScrBytes; ++i) g_scr_src[i] = (uint8_t)(i * 7 + round);
        std::memset(g_scr_dst, 0, (size_t)kScrBytes);
        CHECK_OK(f.step(4, round, [&](Ctx& c) {
            RadOperand opd[1] = { RAD_B(x) };
            CHECK_OK(c.set_lane(first));
            CHECK_OK(c.issue(wr, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(first ^ 1));
            CHECK_OK(c.issue(rd, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(0));
        }));
        /* run_step ends by joining lane 1 into lane 0, so lane 0 drained is both drained. */
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        CHECK_EQ(std::memcmp(g_scr_dst, g_scr_src, (size_t)kScrBytes), 0);
    }

    rad_dev_free(g_scr_src, RAD_MEM_DEVICE);
    rad_dev_free(g_scr_dst, RAD_MEM_DEVICE);
    g_scr_src = g_scr_dst = nullptr;
}

/* A program that declares a second lane gets a scratch region per lane, so scratch users on two
 * lanes get disjoint bytes and neither waits for the other. The pointer each launch saw is the
 * check: a shared region would put both at the same address. */
static void* g_scr_seen[2] = { nullptr, nullptr };
static int k_scr_seen0(const RadArgs* a, RadStream s) { g_scr_seen[0] = a->scratch; return record("s0", a, s); }
static int k_scr_seen1(const RadArgs* a, RadStream s) { g_scr_seen[1] = a->scratch; return record("s1", a, s); }

TEST(a_declared_second_lane_has_its_own_scratch_region) {
    Fixture f;
    f.P.scratch_bytes   = kScrBytes;
    f.P.scratch_regions = 2;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op a0 = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_seen0 }, { "s0" }, nullptr, {});
    rad_op a1 = f.add_op("gemm_nt", "M", { 8192 }, { k_scr_seen1 }, { "s1" }, nullptr, {});
    f.P.ops[a0].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    f.P.ops[a1].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = kScrBytes;
    CHECK_OK(f.start());

    for (int first = 0; first < 2; ++first) {
        g_scr_seen[0] = g_scr_seen[1] = nullptr;
        CHECK_OK(f.step(4, first, [&](Ctx& c) {
            RadOperand opd[1] = { RAD_B(x) };
            CHECK_OK(c.set_lane(first));
            CHECK_OK(c.issue(a0, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(first ^ 1));
            CHECK_OK(c.issue(a1, opd, 1, RAD_N_BATCH));
            CHECK_OK(c.set_lane(0));
        }));
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        REQUIRE(g_scr_seen[0] && g_scr_seen[1]);
        const int64_t gap = (const char*)g_scr_seen[1] - (const char*)g_scr_seen[0];
        CHECK((gap < 0 ? -gap : gap) >= kScrBytes);
        /* The same plan entry, issued from the other lane, follows the lane and not the entry. */
        CHECK_EQ((const char*)g_scr_seen[first] == (const char*)f.ctx.scratch(), true);
    }
}

/* Operands are positional (spec §2.3), so a miscount is silent numerical garbage at the kernel,
 * not a diagnosable failure. The runtime checks the given arity against the schema. */
TEST(operand_arity_is_checked_against_the_schema) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    g_rec.clear();
    int s = f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[2] = { RAD_B(x), RAD_W(w) };
        c.issue(op, opd, 2, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_SCHEMA);
    CHECK_EQ(g_rec.size(), (size_t)0);
    CHECK(std::strstr(f.ctx.step_error(), "schema declares 3") != nullptr);
}

/* A raw operand is a pointer the core did not allocate (a RadBatch field). It has no BufferInfo,
 * hence no dtype and no element size, so an offset on one is refused by name. */
TEST(raw_operand_carries_the_pointer_and_refuses_an_offset) {
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("rope", "M", { 8192 }, { k_pre }, { "prefill" }, nullptr, {});
    CHECK_OK(f.start());

    int32_t positions[4] = { 0, 1, 2, 3 };

    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[2] = { RAD_B(x), RAD_P(positions) };
        opd[1].rows = 4;
        CHECK_OK(c.issue(op, opd, 2, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK(g_rec[0].t[1].data == (void*)positions);
    CHECK_EQ(g_rec[0].t[1].shape[0], (int64_t)4);

    g_rec.clear();
    int s = f.step(4, 1, [&](Ctx& c) {
        RadOperand opd[2] = { RAD_B(x), RAD_P(positions) };
        opd[1].offset = 2;
        c.issue(op, opd, 2, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_INVAL);
    CHECK(std::strstr(f.ctx.step_error(), "no dtype") != nullptr);
}

/* Runtime LoRA's issue chain (Ctx::issue_lora): a gemm that is a LoRA site is followed by
 * lora_shrink and lora_expand over its own activation, output and rows, but only on a step that
 * has an adapter row (RadBatch::lora_slot non-null). A step without one, or a draft pass, issues
 * the gemm alone, the pass an engine without adapters runs; an encoder pass reaching a site is
 * refused by name; a generator pass runs its one slot in every row. */
static int k_gemm  (const RadArgs* a, RadStream s) { return record("gemm",   a, s); }
static int k_shrink(const RadArgs* a, RadStream s) { return record("shrink", a, s); }
static int k_expand(const RadArgs* a, RadStream s) { return record("expand", a, s); }

TEST(adapter_ops_follow_a_lora_site_only_on_a_step_with_an_adapter_row) {
    Fixture f;
    const rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    const rad_buf x  = f.add_buffer("x", RAD_BF16, 16, 4, 0);
    const rad_buf y  = f.add_buffer("y", RAD_BF16, 16, 8, 256);
    const rad_buf pa = f.add_buffer("lora.a", RAD_BF16, 2, 16, 1024);    /* [S, R * K] */
    const rad_buf pb = f.add_buffer("lora.b", RAD_BF16, 2, 32, 1280);    /* [S, N * R] */
    const rad_buf ps = f.add_buffer("lora.scale", RAD_F32, 2, 8, 1536);
    const rad_buf t0 = f.add_buffer("lora.t0", RAD_F32, 16, 4, 2048);
    const rad_buf t1 = f.add_buffer("lora.t1", RAD_F32, 16, 4, 2304);
    const rad_buf rows = f.add_buffer("lora.rows", RAD_I32, 32, 1, 2560);   /* [S * 16] */
    const rad_op g  = f.add_op("gemm_nt", "M", { 64 }, { k_gemm }, { "gemm" }, &g_gemm_schema, { w });
    const rad_op sh = f.add_op("lora_shrink", "M", { 64 }, { k_shrink }, { "shrink" }, nullptr, {});
    const rad_op ex = f.add_op("lora_expand", "M", { 64 }, { k_expand }, { "expand" }, nullptr, {});
    f.P.ops[g].lora_site = 0;
    f.P.ops[sh].anchor = (int32_t)g;
    f.P.ops[ex].anchor = (int32_t)g;
    rad::LoraIssueSite site;
    site.gemm = g; site.shrink = sh; site.expand = ex;
    site.opd_a = 0; site.opd_y = 2;
    site.a_dtype = RAD_BF16; site.y_dtype = RAD_BF16;
    site.pool = 0; site.t = 0;
    site.R = 4; site.K = 4; site.N = 8;
    f.P.lora_sites.push_back(site);
    rad::LoraPool pool;
    pool.weight = "blk.0.w";
    pool.a = pa; pool.b = pb; pool.scale = ps;
    pool.R = 4; pool.K = 4; pool.N = 8; pool.S = 2;
    f.P.lora_pools.push_back(pool);
    rad::LoraScratch scr;
    scr.R = 4;
    scr.lane[0] = t0; scr.lane[1] = t1;
    f.P.lora_t.push_back(scr);
    f.P.lora_rows = rows;
    f.P.lora_rows_n = 16;
    CHECK_OK(f.start());

    auto body = [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        CHECK_OK(c.issue(g, opd, 3, RAD_N_BATCH));
    };
    auto names = [] {
        std::string s;
        for (const Rec& r : g_rec) s += std::string(s.empty() ? "" : " ") + r.kernel;
        return s;
    };

    /* No adapter row: the gemm alone. */
    f.batch.lora_slot = nullptr;
    g_rec.clear();
    CHECK_OK(f.step(4, 0, body));
    CHECK_EQ(names(), std::string("gemm"));

    /* An adapter row: the pair behind the gemm, over its activation and its output, each row's
     * slot read from the step batch. */
    int32_t slots[4] = { 1, -1, 0, -1 };
    f.batch.lora_slot = slots;
    g_rec.clear();
    CHECK_OK(f.step(4, 1, body));
    REQUIRE_EQ(names(), std::string("gemm shrink expand"));
    CHECK(g_rec[1].t[0].data == g_rec[0].t[0].data);     /* shrink reads the gemm's x */
    CHECK(g_rec[1].t[2].data == (void*)slots);
    CHECK_EQ(g_rec[1].t[2].shape[0], (int64_t)4);
    CHECK(g_rec[2].t[4].data == g_rec[0].t[2].data);     /* expand adds into the gemm's y */
    CHECK(g_rec[2].t[0].data == g_rec[1].t[4].data);     /* over shrink's t */
    CHECK_EQ(g_rec[1].t[4].shape[0], (int64_t)4);        /* t holds the step's rows */

    /* A draft pass runs the base weights: the gemm alone. */
    f.batch.draft_pass = 1;
    g_rec.clear();
    CHECK_OK(f.step(4, 2, body));
    CHECK_EQ(names(), std::string("gemm"));
    f.batch.draft_pass = 0;

    /* An encoder pass never reaches a site; one that does is refused, not run without its delta. */
    f.batch.enc = 1;
    g_rec.clear();
    const int s = f.step(4, 3, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        c.issue(g, opd, 3, RAD_N_BATCH);
    });
    CHECK_EQ(s, RAD_E_UNSUPPORTED);
    CHECK(std::strstr(f.ctx.step_error(), "encoder pass") != nullptr);
    CHECK_EQ(names(), std::string("gemm"));

    /* A generator pass runs one slot in every row, whatever the gemm's rows are: its slot operand
     * is that slot's run of the slot-row table; with no slot, the gemm alone. */
    f.batch.enc = 0;
    f.batch.gen = RAD_GEN_PASS_DENOISE;
    f.batch.lora_slot = nullptr;
    f.ctx.set_gen_lora_slot(-1);
    g_rec.clear();
    CHECK_OK(f.step(4, 4, body));
    CHECK_EQ(names(), std::string("gemm"));
    f.ctx.set_gen_lora_slot(1);
    g_rec.clear();
    CHECK_OK(f.step(4, 5, body));
    REQUIRE_EQ(names(), std::string("gemm shrink expand"));
    CHECK(g_rec[1].t[2].data == (void*)((char*)f.ctx.buf_ptr(rows) + 1 * 16 * 4));
    CHECK_EQ(g_rec[1].t[2].shape[0], (int64_t)4);
    CHECK(g_rec[2].t[1].data == g_rec[1].t[2].data);
    /* Rows past the slot table's run are issued in chunks of it, each over its own rows. */
    f.P.lora_rows_n = 3;
    g_rec.clear();
    CHECK_OK(f.step(4, 6, body));
    REQUIRE_EQ(names(), std::string("gemm shrink expand shrink expand"));
    CHECK_EQ(g_rec[1].t[0].shape[0], (int64_t)3);
    CHECK_EQ(g_rec[3].t[0].shape[0], (int64_t)1);
    CHECK(g_rec[3].t[0].data == (void*)((char*)g_rec[1].t[0].data + 3 * 4 * 2));   /* x row 3 */
    CHECK(g_rec[4].t[4].data == (void*)((char*)g_rec[2].t[4].data + 3 * 8 * 2));   /* y row 3 */
    CHECK(g_rec[3].t[2].data == (void*)((char*)f.ctx.buf_ptr(rows) + 1 * 3 * 4));
    f.P.lora_rows_n = 16;
    f.ctx.set_gen_lora_slot(-1);
    f.batch.gen = 0;
}

/* Site is a property of the weights (spec §5.1). A domain with no kernel in some band is refused,
 * naming the op and band: the planner was told at declare which sites are unavailable, so asking
 * anyway is a planner bug. */
TEST(execution_site_refuses_a_domain_with_no_kernel) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, nullptr, { w });
    CHECK_OK(f.start());

    CHECK_EQ(f.ctx.op_domain(op), RAD_DOMAIN_DEVICE);
    CHECK_EQ(f.ctx.set_op_domain(op, RAD_DOMAIN_HOST), RAD_E_NOKERNEL);
    CHECK_EQ(f.ctx.op_domain(op), RAD_DOMAIN_DEVICE);   /* and it did not move */
    CHECK_EQ(f.ctx.set_op_domain(op, RAD_DOMAIN_DEVICE), RAD_OK);
}

/* A hole in an op's bucket table is a declare-time failure that surfaces here only if the plugin
 * issues into it. Bands that do resolve still work, which makes the fusion-probe pattern (spec
 * §2.3) safe. */
TEST(a_band_with_a_hole_does_not_poison_its_neighbours) {
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = f.add_op("rmsnorm", "M", { 16, 8192 }, { nullptr, k_pre },
                         { nullptr, "prefill" }, nullptr, {});
    CHECK_OK(f.start());

    g_rec.clear();
    CHECK_OK(f.step(64, 0, [&](Ctx& c) {
        RadOperand opd[1] = { RAD_B(x) };
        CHECK_OK(c.issue(op, opd, 1, 64));   /* the resolved band */
    }));
    CHECK_EQ(g_rec.size(), (size_t)1);
}

/* Per-op timing is opt-in and adds synchronisations that change the mover's slot supply, so
 * profiled runs are not comparable with unprofiled ones (spec §16). An unprofiled step never
 * blocks the host; a profiled one blocks once, at the step boundary on the compute stream, however
 * many ops it timed (one host-wait event per op would exhaust interrupt signals on a large
 * prefill). */
TEST(profiling_is_opt_in_and_costs_a_synchronisation) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    CHECK_OK(f.start(nullptr, /*profile=*/true));

    CHECK_EQ(f.ctx.host_syncs(), (int64_t)0);   /* nothing yet: the cost lands at the step boundary */
    g_rec.clear();
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        for (int i = 0; i < 3; ++i) CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
    }));
    CHECK_EQ(g_rec.size(), (size_t)3);
    /* Three timed launches, one host block at the end of the step: the cost that changes the
     * mover's slot supply, and why cross-build comparisons run without profiling. Every other test
     * here has host_syncs() == 0. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)1);
    f.ctx.dump_profile();
}

/* A kernel that keeps counters prints them where the dump has both halves: its numbers and the time
 * this run measured for its op (RadStatsDumpFn, abi/rad_abi.h). The phase it is handed is the
 * step's, which is what separates a prefill gather's timing from a decode one; a phase the op never
 * ran in is -1, so a kernel can print `na` rather than a zero it would read as fast. */
static double g_stats_ms[3] = { -2.0, -2.0, -2.0 };
static int    g_stats_calls = 0;
static int    g_stats_n = -1;
static void*  g_stats_inst = nullptr;

static int g_stats_anchor = 0;   /* stands in for the instance the plugin's init hands out */
static void stats_dump(void* inst, const double* ms, int n) {
    ++g_stats_calls;
    g_stats_n = n;
    g_stats_inst = inst;
    for (int i = 0; i < 3 && i < n; ++i) g_stats_ms[i] = ms[i];
}

TEST(the_profile_dump_hands_a_kernel_its_own_per_phase_times) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 8192 }, { k_pre }, { "prefill" }, &g_gemm_schema, { w });
    f.infos.back().stats_dump = stats_dump;
    f.P.ops[op].bands[0].dom[RAD_DOMAIN_DEVICE].instance = &g_stats_anchor;
    CHECK_OK(f.start(nullptr, /*profile=*/true));

    g_stats_calls = 0;
    g_stats_ms[0] = g_stats_ms[1] = g_stats_ms[2] = -2.0;
    for (int n_tok : { 4, 1 }) {              /* a prefill step, then a decode one */
        CHECK_OK(f.step(n_tok, 0, [&](Ctx& c) {
            RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
            CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
        }));
    }
    f.ctx.dump_profile();
    CHECK_EQ(g_stats_calls, 1);               /* once an op, not once a phase */
    CHECK_EQ(g_stats_n, 3);
    CHECK(g_stats_inst != nullptr);           /* the instance the rows were launched with */
    CHECK(g_stats_ms[0] >= 0.0);              /* timed in a prefill step... */
    CHECK(g_stats_ms[1] >= 0.0);              /* ...and in a decode one */
    CHECK(g_stats_ms[2] == -1.0);             /* and never in a mixed one */
}

/* An op that resolved into two bands has two instances, and each is handed its own time. Handing
 * both the op's cross-band sum -- or printing one instance's line with the other's numbers -- is
 * the bug this reddens on: a prefill instance's time is not a decode instance's. */
static int    g_two_anchor[2] = { 0, 0 };
static int    g_two_calls[2]  = { 0, 0 };
static double g_two_ms[2][3];
static void*  g_two_inst[2]   = { nullptr, nullptr };
static void stats_dump_two(void* inst, const double* ms, int n) {
    const int k = inst == &g_two_anchor[0] ? 0 : 1;
    ++g_two_calls[k];
    g_two_inst[k] = inst;
    for (int i = 0; i < 3; ++i) g_two_ms[k][i] = i < n ? ms[i] : -3.0;
}

TEST(a_second_instance_is_handed_its_own_time) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    /* Two bands of one op, each with its own kernel row and its own instance: the shape a small-M
     * and a large-M band of the PLE gather take, both counted. */
    rad_op op = f.add_op("gemm_nt", "M", { 16, 8192 }, { k_pre, k_pre }, { "small", "large" },
                         &g_gemm_schema, { w });
    for (RadKernelInfo& ki : f.infos) ki.stats_dump = stats_dump_two;
    f.P.ops[op].bands[0].dom[RAD_DOMAIN_DEVICE].instance = &g_two_anchor[0];
    f.P.ops[op].bands[1].dom[RAD_DOMAIN_DEVICE].instance = &g_two_anchor[1];
    CHECK_OK(f.start(nullptr, /*profile=*/true));

    for (int n_tok : { 64, 1 }) {             /* the 8192 band in a prefill step, then the 16 one */
        CHECK_OK(f.step(n_tok, 0, [&](Ctx& c) {
            RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
            CHECK_OK(c.issue(op, opd, 3, RAD_N_BATCH));
        }));
    }
    g_two_calls[0] = g_two_calls[1] = 0;
    for (auto& m : g_two_ms) m[0] = m[1] = m[2] = -2.0;
    f.ctx.dump_profile();

    CHECK_EQ(g_two_calls[0], 1);              /* once an instance, not once an op */
    CHECK_EQ(g_two_calls[1], 1);
    CHECK(g_two_inst[0] == &g_two_anchor[0]);
    CHECK(g_two_inst[1] == &g_two_anchor[1]);
    CHECK(g_two_ms[1][0] >= 0.0);             /* the large band was timed in the prefill step */
    CHECK(g_two_ms[1][1] == -1.0);            /* and never in the decode one */
    CHECK(g_two_ms[0][1] >= 0.0);             /* the small band: the decode step only */
    CHECK(g_two_ms[0][0] == -1.0);
}

/* The barrier is a condvar over threads in one process, not IPC: ranks share an address space and
 * the batch is shared by pointer (spec §1). It only has to order the release. */
TEST(rank_barrier_releases_all_ranks) {
    const int n = 4;
    rad::RankBarrier b(n);
    std::atomic<int> before{0}, after{0};
    std::vector<std::thread> th;
    for (int i = 0; i < n; ++i)
        th.emplace_back([&] {
            before.fetch_add(1);
            b.arrive_and_wait();
            /* Everybody arrived before anybody left: the whole contract. */
            if (before.load() == n) after.fetch_add(1);
            b.arrive_and_wait();
        });
    for (auto& t : th) t.join();
    CHECK_EQ(after.load(), n);
}

/* A decode step's launches would cost 1-2 ms of driver time through hipLaunchKernel (spec §3.2),
 * so the core's own share must stay far below that. Measured here: everything the runtime does per
 * issue (band scan, tensor materialisation, RadArgs patch, indirect call) against a launch hook
 * that returns immediately.
 */
TEST(per_issue_cost) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_buf    x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_buf    y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = f.add_op("gemm_nt", "M", { 16, 64, 8192 },
                         { k_null, k_null, k_null }, { "a", "b", "c" }, &g_gemm_schema, { w });
    CHECK_OK(f.start());

    const int reps = 200000;
    double    ns   = 0.0;
    CHECK_OK(f.step(1, 0, [&](Ctx& c) {
        RadOperand opd[3] = { RAD_B(x), RAD_W(w), RAD_B(y) };
        /* Warm. */
        for (int i = 0; i < 10000; ++i) c.issue(op, opd, 3, RAD_N_BATCH);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) c.issue(op, opd, 3, RAD_N_BATCH);
        auto t1 = std::chrono::steady_clock::now();
        ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
    }));

    fprintf(stderr,
            "       rad_issue: %.1f ns per call (3 operands, one weight, no residency table)\n"
            "       -> ~%.0f us of core time for a 1280-issue decode step (64 layers x 20 ops),\n"
            "          against the 1-2 ms the same launches cost through hipLaunchKernel\n"
            "          (spec 3.2), which the device backend avoids with AQL packets.\n",
            ns, ns * 1280 / 1000.0);

    /* Loose on purpose: a regression tripwire, not a benchmark. A hundredfold miss means an
     * allocation or a map lookup is on the hot path. */
    CHECK(ns < 2000.0);
}

/* ================================================================== the arena's size
 *
 * Ctx::bind_arena allocates the arena at the end of startup; the VRAM budget resolver subtracts it
 * from the card before the pools are allocated. Both use arena_plan_bytes, so the property is that
 * the resolver's number is exactly what Arena::init allocates.
 */

TEST(the_predicted_arena_size_is_the_one_the_arena_allocates) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 16, 16, 0);        /* 512 B at offset 0 */
    f.add_buffer("y", RAD_BF16, 16, 16, 512);      /* 512 B after it */
    f.P.arena_bytes   = 1024;
    f.P.scratch_bytes = 256;

    int64_t dev = 0, total = 0;
    const int64_t got = rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(got, total);

    rad::Arena a;
    CHECK_OK(a.init(dev, f.P.scratch_bytes, RAD_MEM_DEVICE));
    CHECK(a.live());
    /* The prediction is exact, not an upper bound: a resolver told "at least this much" cannot
     * hand the remainder to a pool. */
    CHECK_EQ(a.bytes(), total);
    CHECK_EQ(a.scratch_bytes(), 256ll);
    /* Scratch starts on a unit boundary past the plan, so a kernel wanting an aligned workspace
     * gets one and the buffers below are not overlapped. */
    CHECK_EQ((char*)a.scratch() - a.base(), rad::align_up(dev, RAD_ALIGN_UNIT));
    a.release();
    CHECK(!a.live());
    CHECK(a.base() == nullptr);
}

/* The arena prediction counts every scratch region, so the budget resolver charges the card for
 * what Arena::init allocates. */
TEST(the_predicted_arena_counts_every_scratch_region) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 16, 16, 0);
    f.P.arena_bytes     = 512;
    f.P.scratch_bytes   = 300;
    f.P.scratch_regions = 2;
    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    rad::Arena a;
    CHECK_OK(a.init(dev, f.P.scratch_bytes, RAD_MEM_DEVICE, f.P.scratch_regions));
    CHECK_EQ(a.bytes(), total);
    CHECK_EQ(a.scratch_regions(), 2);
    CHECK_EQ((char*)a.scratch(1) - (char*)a.scratch(0), rad::align_up(300, RAD_ALIGN_UNIT));
    CHECK((char*)a.scratch(1) + 300 <= a.base() + a.bytes());
    a.release();
}

/* The high-water mark is over offset + size, not a sum of sizes. The buffer plan reuses offsets
 * for buffers whose lifetimes do not overlap, so summing would reserve several times the needed
 * arena, which comes out of the weight slab. */
TEST(overlapping_buffer_lifetimes_do_not_each_buy_their_own_bytes) {
    Fixture f;
    f.add_buffer("a", RAD_BF16, 64, 64, 0);        /* 8 KiB */
    f.add_buffer("b", RAD_BF16, 64, 64, 0);        /* the same 8 KiB, reused */
    f.add_buffer("c", RAD_BF16, 64, 64, 8192);     /* live alongside them */
    f.P.arena_bytes   = 16384;
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(dev, 16384ll);                        /* not 24576 */
    CHECK_EQ(total, rad::align_up(16384ll, RAD_ALIGN_UNIT));
}

/* A buffer the plan never placed is not charged for. bind_arena refuses such a program by name;
 * until then, a negative offset must not be folded into a size. */
TEST(an_unplaced_buffer_is_skipped_rather_than_counted) {
    Fixture f;
    f.add_buffer("placed", RAD_BF16, 16, 16, 0);
    rad::BufferInfo unplaced;
    unplaced.name = "unplaced";
    unplaced.decl.dtype = RAD_BF16;
    unplaced.decl.rank = 2;
    unplaced.decl.shape[0] = 1024;
    unplaced.decl.shape[1] = 1024;
    unplaced.bytes = rad_dtype_bytes(RAD_BF16, 1024 * 1024);
    unplaced.arena_offset = -1;
    f.P.buffers.push_back(unplaced);
    f.P.arena_bytes   = 512;
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(dev, 512ll);
    CHECK(total < 1024 * 1024);
}

/* A host-domain buffer lives in the other arena, so it must not raise the device one: the
 * resolver subtracts the device figure from the card, and charging host bytes against VRAM takes
 * memory from the weights for nothing. */
TEST(a_host_domain_buffer_does_not_raise_the_device_arena) {
    Fixture f;
    f.add_buffer("dev", RAD_BF16, 16, 16, 0);      /* 512 B */
    rad::BufferInfo h;
    h.name = "host";
    h.decl.dtype = RAD_BF16;
    h.decl.rank = 2;
    h.decl.shape[0] = 256;
    h.decl.shape[1] = 256;
    h.decl.domain = RAD_DOMAIN_HOST;
    h.bytes = rad_dtype_bytes(RAD_BF16, 256 * 256);
    h.arena_offset = 0;
    f.P.buffers.push_back(h);
    f.P.arena_bytes   = 99999;                     /* the plan's own total covers both domains */
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    /* With a host buffer present, the device figure is the device high-water mark, not the plan's
       combined total; that is why the two are tracked apart. */
    CHECK_EQ(dev, 512ll);
    CHECK_EQ(total, rad::align_up(512ll, RAD_ALIGN_UNIT));
}

/* A zero-byte arena would hand every buffer a null base and turn a plan bug into an anonymous
 * segfault. One page instead, so a stray offset lands somewhere reportable; the prediction must
 * include that floor. */
TEST(an_empty_program_still_buys_one_page) {
    Fixture f;
    f.P.arena_bytes   = 0;
    f.P.scratch_bytes = 0;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(dev, 0ll);
    CHECK_EQ(total, (int64_t)RAD_ALIGN_UNIT);

    rad::Arena a;
    CHECK_OK(a.init(0, 0, RAD_MEM_DEVICE));
    CHECK(a.base() != nullptr);
    CHECK_EQ(a.bytes(), (int64_t)RAD_ALIGN_UNIT);
    CHECK_EQ(a.scratch_bytes(), 0ll);
}

TEST(the_arena_refuses_a_negative_request_rather_than_allocating_one) {
    rad::Arena a;
    CHECK_EQ(a.init(-1, 0, RAD_MEM_DEVICE), RAD_E_INVAL);
    CHECK(!a.live());
    CHECK_EQ(a.init(0, -1, RAD_MEM_DEVICE), RAD_E_INVAL);
    CHECK(!a.live());
}

/* Scratch is padded on its own because it starts on a unit boundary: summing plan and scratch
 * before rounding would under-reserve by up to one unit, and the kernel reading past it is the
 * largest in the program. */
TEST(scratch_is_rounded_separately_from_the_plan) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 1, 1, 0);
    f.P.arena_bytes   = 1;                         /* both deliberately unaligned */
    f.P.scratch_bytes = 1;

    int64_t dev = 0, total = 0;
    rad::arena_plan_bytes(f.P, &dev, &total);
    CHECK_EQ(total, 2 * (int64_t)RAD_ALIGN_UNIT);
    CHECK(total > rad::align_up(2ll, RAD_ALIGN_UNIT));

    rad::Arena a;
    CHECK_OK(a.init(dev, f.P.scratch_bytes, RAD_MEM_DEVICE));
    CHECK_EQ(a.bytes(), total);
}

/* The out parameters are optional, and the return is the total either way. */
TEST(arena_plan_bytes_returns_the_total_with_or_without_its_out_parameters) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 8, 8, 0);
    f.P.arena_bytes   = 128;
    f.P.scratch_bytes = 64;
    int64_t dev = 0, total = 0;
    const int64_t a = rad::arena_plan_bytes(f.P, &dev, &total);
    const int64_t b = rad::arena_plan_bytes(f.P, nullptr, nullptr);
    CHECK_EQ(a, b);
    CHECK_EQ(a, total);
}

/* A weight-table patch lands at its place in stream order: work issued before it reads the old
 * entry, work after it the new one. On the host backend the patch is a host write, which without
 * draining the stream could reach the table before a queued reader issued ahead of it. A chain of
 * copies holds that reader back long enough for a queue-jumping write to show. */
TEST(a_weight_table_patch_lands_in_stream_order) {
    const int64_t big = 4 << 20;
    const int     chain = 16;
    RadStream s = nullptr;
    REQUIRE(rad_stream_create(&s, 0) >= 0);
    auto* blob_src = (uint8_t*)rad_dev_alloc(big, RAD_MEM_HOST_PINNED);
    auto* blob_dst = (uint8_t*)rad_dev_alloc(big, RAD_MEM_DEVICE);
    auto* tab   = (void**)rad_dev_alloc(64, RAD_MEM_DEVICE);
    auto* stage = (void**)rad_dev_alloc(64, RAD_MEM_HOST_PINNED);
    REQUIRE(blob_src && blob_dst && tab && stage);
    std::memset(blob_src, 0x5A, (size_t)big);
    for (int round = 0; round < 3; ++round) {
        void* const before = (void*)(uintptr_t)(0x1000 + round);
        void* const after  = (void*)(uintptr_t)(0x2000 + round);
        stage[0] = before;
        stage[1] = stage[2] = nullptr;
        REQUIRE(rad_memcpy_async(tab, stage, sizeof(void*), s) >= 0);
        REQUIRE(rad_stream_sync(s) >= 0);

        for (int i = 0; i < chain; ++i) REQUIRE(rad_memcpy_async(blob_dst, blob_src, big, s) >= 0);
        REQUIRE(rad_memcpy_async(stage + 1, tab, sizeof(void*), s) >= 0);
        rad::WtabPatch p;
        p.n = 1;
        p.dst[0] = tab;
        p.val[0] = after;
        CHECK_OK(rad::wtab_patch(p, s));
        REQUIRE(rad_memcpy_async(stage + 2, tab, sizeof(void*), s) >= 0);
        REQUIRE(rad_stream_sync(s) >= 0);
        CHECK(stage[1] == before);
        CHECK(stage[2] == after);
    }
    rad_dev_free(stage, RAD_MEM_HOST_PINNED);
    rad_dev_free(tab, RAD_MEM_DEVICE);
    rad_dev_free(blob_dst, RAD_MEM_DEVICE);
    rad_dev_free(blob_src, RAD_MEM_HOST_PINNED);
    rad_stream_destroy(s);
}

/* ================================================================== issue-path refusals
 *
 * Each refusal returns its status, names what it refused in step_error(), launches nothing, and
 * leaves the next step clean. A guard that loosens reopens silent arena aliasing or a read of the
 * wrong bytes; one that refuses wrongly stops a serving engine. */
namespace {
int issue_once(Fixture& f, int step_no, rad_op op, std::vector<RadOperand> opd) {
    g_rec.clear();
    return f.step(4, step_no, [&](Ctx& c) {
        c.issue(op, opd.data(), (int)opd.size(), RAD_N_BATCH);
    });
}

void refused(Fixture& f, int st, int want, const char* named) {
    CHECK_EQ(st, want);
    CHECK(g_rec.empty());
    const bool says = std::strstr(f.ctx.step_error(), named) != nullptr;
    CHECK(says);
    if (!says) std::fprintf(stderr, "    step_error: %s\n", f.ctx.step_error());
}

/* The step after a refusal runs: the failure is per step, not per Ctx. */
void next_step_clean(Fixture& f, int step_no, rad_op op, std::vector<RadOperand> opd) {
    CHECK_OK(issue_once(f, step_no, op, opd));
    CHECK_EQ(g_rec.size(), (size_t)1);
}

/* An op with no schema, so any operand count and kind is accepted by arity. */
rad_op loose_op(Fixture& f, const char* name = "rope") {
    return f.add_op(name, "M", { 8192 }, { k_pre }, { "prefill" }, nullptr, {});
}
}  /* namespace */

/* A buffer operand may name fewer rows than its declaration, or a column slice inside it, and
 * never reach past its end: the bytes there belong to whatever buffer the plan packed next. */
TEST(a_buffer_operand_reaching_past_its_declaration_is_refused_by_name) {
    Fixture f;
    rad_buf y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = loose_op(f);
    CHECK_OK(f.start());
    /* The last four columns of every row: inside, though it starts four elements in. */
    CHECK_OK(issue_once(f, 0, op, { RAD_B_COL(y, 4, 4) }));
    CHECK_EQ(g_rec.size(), (size_t)1);
    CHECK_EQ(g_rec[0].t[0].shape[1], (int64_t)4);
    CHECK_EQ(g_rec[0].t[0].stride[0], (int64_t)8);
    int step = 1;
    for (const RadOperand o : { RAD_B_COL(y, 5, 4),        /* one column past the last row's end */
                                RAD_B_N(y, 8, 64),         /* a row in, all 64 rows */
                                RAD_B_N(y, 0, 65) }) {     /* one row more than declared */
        refused(f, issue_once(f, step++, op, { o }), RAD_E_SHAPE, "reaches past buffer 'y'");
        next_step_clean(f, step++, op, { RAD_B(y) });
    }
}

/* A negative offset reaches before the buffer: refused like an overrun. */
TEST(a_negative_operand_offset_is_refused) {
    Fixture f;
    rad_buf y = f.add_buffer("y", RAD_BF16, 64, 8, 1024);
    rad_op op = loose_op(f);
    CHECK_OK(f.start());
    refused(f, issue_once(f, 0, op, { RAD_B_N(y, -8, 4) }), RAD_E_SHAPE, "reaches past buffer 'y'");
    next_step_clean(f, 1, op, { RAD_B(y) });
}

/* An op may touch a buffer only inside the lifetime the plan gave it: outside it, the arena may
 * have handed the bytes to another buffer. */
TEST(an_operand_outside_its_buffers_planned_lifetime_is_refused) {
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op first = loose_op(f, "rope");
    rad_op later = loose_op(f, "rope");
    f.P.buffers[x].first_def = (int32_t)first;
    f.P.buffers[x].last_use  = (int32_t)first;
    CHECK_OK(f.start());
    next_step_clean(f, 0, first, { RAD_B(x) });
    refused(f, issue_once(f, 1, later, { RAD_B(x) }), RAD_E_STATE, "names buffer 'x', which this op never declared");
    next_step_clean(f, 2, first, { RAD_B(x) });
}

/* An element offset into packed 4-bit data that is not a whole byte cannot be expressed as a
 * pointer; it is refused for a buffer and a weight alike, and a whole-byte one is taken. */
TEST(a_sub_byte_offset_into_packed_data_is_refused) {
    static uint8_t packed[64];
    Fixture f;
    rad_buf q = f.add_buffer("q4", RAD_I4, 64, 8, 0);
    rad_weight w = f.add_weight("w4", RAD_I4, 8, 8, packed);
    rad_op op = loose_op(f);
    CHECK_OK(f.start());
    CHECK_OK(issue_once(f, 0, op, { RAD_B_N(q, 2, 4) }));
    CHECK(g_rec.size() == 1 && (char*)g_rec[0].t[0].data == (char*)f.ctx.buf_ptr(q) + 1);
    refused(f, issue_once(f, 1, op, { RAD_B_N(q, 1, 4) }), RAD_E_ALIGN, "buffer 'q4'");
    RadOperand wo = RAD_W(w);
    wo.offset = 3;
    wo.rows = 1;
    refused(f, issue_once(f, 2, op, { wo }), RAD_E_ALIGN, "weight 'w4'");
    next_step_clean(f, 3, op, { RAD_B(q) });
}

/* A weight with no pointer and no residency table claiming it has nowhere to be read from. */
TEST(a_weight_with_no_pointer_is_refused_at_issue) {
    Fixture f;
    rad_weight w = f.add_weight("blk.0.w", RAD_BF16, 8, 4, nullptr);
    rad_weight ok = f.add_weight("blk.1.w", RAD_BF16, 8, 4, g_weight_bytes);
    rad_op op = loose_op(f);
    CHECK_OK(f.start());
    refused(f, issue_once(f, 0, op, { RAD_W(w) }), RAD_E_STATE, "weight 'blk.0.w' has no pointer");
    next_step_clean(f, 1, op, { RAD_W(ok) });
}

/* A weight table is a run of whole weights: an offset or column slice has no meaning for it, and
 * a run past the last declared weight names weights that do not exist. */
TEST(a_weight_table_with_an_offset_a_slice_or_too_many_entries_is_refused) {
    static uint16_t expert_bytes[4][32];
    Fixture f;
    std::vector<rad_weight> w;
    for (int e = 0; e < 4; ++e)
        w.push_back(f.add_weight(("blk.0.e" + std::to_string(e)).c_str(), RAD_BF16, 8, 4,
                                 expert_bytes[e]));
    rad_op op = loose_op(f, "moe_gemm");
    CHECK_OK(f.start());
    RadOperand off = RAD_WTAB(w[0], 4); off.offset = 2;
    RadOperand col = RAD_WTAB(w[0], 4); col.cols = 2;
    refused(f, issue_once(f, 0, op, { off }), RAD_E_INVAL, "is a weight table and carries an offset");
    refused(f, issue_once(f, 1, op, { col }), RAD_E_INVAL, "is a weight table and carries an offset");
    refused(f, issue_once(f, 2, op, { RAD_WTAB(w[0], 5) }), RAD_E_INVAL,
            "names a weight table of 5 entries");
    next_step_clean(f, 3, op, { RAD_WTAB(w[0], 4) });
}

/* Operands past the runtime's fixed per-issue array, and handles declare never returned, are
 * refused rather than read out of bounds. */
TEST(too_many_operands_or_an_unknown_handle_is_refused) {
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = loose_op(f);
    CHECK_OK(f.start());
    std::vector<RadOperand> many((size_t)rad::MAX_OPERANDS + 1, RAD_B(x));
    refused(f, issue_once(f, 0, op, many), RAD_E_INVAL, "the runtime carries at most 32");
    refused(f, issue_once(f, 1, op, { RAD_B(999) }), RAD_E_INVAL, "buffer handle 999");
    refused(f, issue_once(f, 2, op, { RAD_W(999) }), RAD_E_INVAL, "weight handle 999");
    next_step_clean(f, 3, op, { RAD_B(x) });
}

/* ================================================================== KV operands */
namespace {
alignas(64) uint16_t g_kv_pool[4 * 2048];

/* Group 1: full attention over layers 3, 7 and 11, rank 4. Group 2: a conv state, rank 3, over
 * layer 0. Group 0 is the sentinel every handle table carries, with no pool. */
void bind_kv(Fixture& f) {
    f.kv.assign(3, rad::KVPoolBinding{});
    rad::KVPoolBinding& a = f.kv[1];
    a.base = g_kv_pool;
    a.layer_stride = 4 * 2 * 16 * 8 * 2;              /* [4 blocks, 2 heads, 16, 2*4] bf16 */
    a.bytes = 3 * a.layer_stride;
    a.dtype = RAD_BF16;
    a.layer_slot = { -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 2 };
    a.rank = 4;
    a.shape[0] = 4; a.shape[1] = 2; a.shape[2] = 16; a.shape[3] = 8;
    rad::KVPoolBinding& c = f.kv[2];
    c.base = (char*)g_kv_pool + a.bytes;
    c.layer_stride = 6 * 32 * 5 * 2;
    c.bytes = c.layer_stride;
    c.dtype = RAD_BF16;
    c.layer_slot = { 0 };
    c.rank = 3;
    c.shape[0] = 6; c.shape[1] = 32; c.shape[2] = 5;
}
}  /* namespace */

/* A KV operand resolves its absolute layer through the group's binding: layer 7 of a group over
 * layers {3, 7, 11} is the group's second cache, at one layer stride, with the group's shape and
 * packed strides. A wrong slot makes a layer attend to another layer's cache, silently. A layer
 * the group does not hold and a group with no pool are refused. */
TEST(a_kv_operand_resolves_through_the_groups_layer_binding) {
    Fixture f;
    bind_kv(f);
    rad_op op = loose_op(f, "attn_paged");
    CHECK_OK(f.start());
    CHECK_OK(issue_once(f, 0, op, { RAD_KV(1, 7), RAD_KV(1, 11), RAD_KV(2, 0) }));
    REQUIRE_EQ(g_rec.size(), (size_t)1);
    const RadTensor& t = g_rec[0].t[0];
    CHECK(t.data == (char*)g_kv_pool + f.kv[1].layer_stride);
    CHECK(g_rec[0].t[1].data == (char*)g_kv_pool + 2 * f.kv[1].layer_stride);
    CHECK_EQ((int)t.rank, 4);
    CHECK_EQ((int)t.dtype, (int)RAD_BF16);
    const int64_t shape[4] = { 4, 2, 16, 8 }, stride[4] = { 256, 128, 8, 1 };
    for (int d = 0; d < 4; ++d) { CHECK_EQ(t.shape[d], shape[d]); CHECK_EQ(t.stride[d], stride[d]); }
    const RadTensor& c = g_rec[0].t[2];
    CHECK_EQ((int)c.rank, 3);
    CHECK_EQ(c.shape[2], (int64_t)5);
    CHECK_EQ(c.stride[0], (int64_t)160);

    refused(f, issue_once(f, 1, op, { RAD_KV(1, 5) }), RAD_E_INVAL, "asks KV group 1 for layer 5");
    refused(f, issue_once(f, 2, op, { RAD_KV(1, 12) }), RAD_E_INVAL, "asks KV group 1 for layer 12");
    refused(f, issue_once(f, 3, op, { RAD_KV(0, 3) }), RAD_E_STATE, "names KV group 0");
    refused(f, issue_once(f, 4, op, { RAD_KV(3, 3) }), RAD_E_STATE, "names KV group 3");
    next_step_clean(f, 5, op, { RAD_KV(1, 3) });
}

/* The binding the engine builds from the block manager's plans: each bound layer at its position
 * in the group, the pool and stride from the plan, and the shape libref's layouts define from the
 * declaration and the manager's sizes. */
TEST(kv_pools_bind_each_layer_to_its_place_in_the_group) {
    Program P;
    P.kv_groups.resize(4);
    rad::KVGroupInfo& full = P.kv_groups[1];
    full.decl.kind = RAD_KV_FULL; full.decl.dtype = RAD_BF16;
    full.decl.n_head_kv = 2; full.decl.head_dim = 64;
    full.layers = { 3, 7, 11 };
    rad::KVGroupInfo& lin = P.kv_groups[2];
    lin.decl.kind = RAD_KV_LINEAR; lin.decl.dtype = RAD_F32;
    lin.decl.n_head_kv = 4; lin.decl.state_dim[0] = 32; lin.decl.state_dim[1] = 16;
    lin.layers = { 0, 1, 2 };
    rad::KVGroupInfo& conv = P.kv_groups[3];
    conv.decl.kind = RAD_KV_CONV; conv.decl.dtype = RAD_BF16;
    conv.decl.n_head_kv = 4; conv.decl.head_dim = 8;
    conv.layers = { 2 };

    std::vector<rad::KVGroupPlan> plans(4);
    plans[1].kind = RAD_KV_FULL; plans[1].n_layers = 3; plans[1].block_size = 16;
    plans[1].n_blocks = 10; plans[1].bytes_per_block = 3 * 2 * 16 * 128 * 2;
    plans[1].base = g_kv_pool; plans[1].pool_bytes = 10 * plans[1].bytes_per_block;
    plans[2].kind = RAD_KV_LINEAR; plans[2].n_layers = 3; plans[2].n_states = 5;
    plans[2].bytes_per_state = 3 * 4 * 32 * 16 * 4;
    plans[3].kind = RAD_KV_CONV; plans[3].n_layers = 1; plans[3].n_states = 5; plans[3].conv_slots = 7;
    std::vector<rad::KVPoolBinding> b;
    rad::bind_kv_pools(P, [&](int32_t g) -> const rad::KVGroupPlan* {
        return g == 0 ? nullptr : &plans[(size_t)g];
    }, &b);
    REQUIRE_EQ(b.size(), (size_t)4);
    CHECK(b[0].base == nullptr);
    CHECK_EQ(b[1].layer_slot.size(), (size_t)12);
    CHECK_EQ(b[1].layer_slot[3], 0);
    CHECK_EQ(b[1].layer_slot[7], 1);
    CHECK_EQ(b[1].layer_slot[11], 2);
    CHECK_EQ(b[1].layer_slot[5], -1);
    CHECK_EQ(b[1].layer_stride, plans[1].layer_stride());
    CHECK_EQ(b[1].layer_stride, (int64_t)10 * 2 * 16 * 128 * 2);
    CHECK_EQ((int)b[1].rank, 4);
    CHECK_EQ(b[1].shape[0], (int64_t)10); CHECK_EQ(b[1].shape[1], (int64_t)2);
    CHECK_EQ(b[1].shape[2], (int64_t)16); CHECK_EQ(b[1].shape[3], (int64_t)128);
    CHECK_EQ((int)b[2].rank, 4);
    CHECK_EQ(b[2].shape[0], (int64_t)5); CHECK_EQ(b[2].shape[2], (int64_t)32);
    CHECK_EQ(b[2].shape[3], (int64_t)16);
    CHECK_EQ((int)b[2].dtype, (int)RAD_F32);
    CHECK_EQ((int)b[3].rank, 3);
    CHECK_EQ(b[3].shape[0], (int64_t)5); CHECK_EQ(b[3].shape[1], (int64_t)32);
    CHECK_EQ(b[3].shape[2], (int64_t)7);
    CHECK_EQ(b[3].layer_slot[2], 0);
}

/* ================================================================== arena levels */
/* A level packs the transient buffers for smaller steps at new offsets with smaller shapes; the
 * active level decides every buffer's address and the extent an operand is checked against, so a
 * step larger than its level is refused rather than written into memory lent to the expert slab.
 * A level that moves a fixed buffer or reaches outside the transient region is refused when it
 * is added. */
TEST(arena_levels_move_transient_buffers_and_bound_their_operands) {
    Fixture f;
    f.P.arena_bytes = 8192;
    f.P.arena_fixed_bytes = 1024;
    rad_buf keep = f.add_buffer("keep", RAD_BF16, 16, 16, 0);
    f.P.buffers[keep].decl.kind = RAD_BUF_PERSIST;
    rad_buf x = f.add_buffer("x", RAD_BF16, 64, 4, 1024);
    rad_buf y = f.add_buffer("y", RAD_BF16, 64, 8, 2048);
    rad_op op = loose_op(f);
    CHECK_OK(f.start());

    std::vector<int64_t> off = { 0, 0, 4096, 4096 + 128 };
    std::vector<RadBufDecl> decl(4);
    for (size_t i = 1; i < 4; ++i) decl[i] = f.P.buffers[i].decl;
    decl[2].shape[0] = 16;
    decl[3].shape[0] = 16;
    const int lv = f.ctx.add_level(16, off, decl);
    REQUIRE_EQ(lv, 1);

    char* base = (char*)f.ctx.arena_base();
    CHECK_OK(f.ctx.set_level(1));
    CHECK_EQ(f.ctx.level(), 1);
    CHECK((char*)f.ctx.buf_ptr(x) == base + 4096);
    CHECK((char*)f.ctx.buf_ptr(y) == base + 4096 + 128);
    CHECK((char*)f.ctx.buf_ptr(keep) == base);
    CHECK_OK(issue_once(f, 0, op, { RAD_B(x) }));
    CHECK(g_rec.size() == 1 && g_rec[0].t[0].shape[0] == 16 && g_rec[0].t[0].data == base + 4096);
    refused(f, issue_once(f, 1, op, { RAD_B_N(x, 0, 32) }), RAD_E_SHAPE, "reaches past buffer 'x'");

    CHECK_OK(f.ctx.set_level(0));
    CHECK((char*)f.ctx.buf_ptr(x) == base + 1024);
    CHECK_OK(issue_once(f, 2, op, { RAD_B_N(x, 0, 32) }));
    CHECK(g_rec.size() == 1 && g_rec[0].t[0].shape[0] == 32);

    CHECK_EQ(f.ctx.set_level(2), RAD_E_INVAL);
    CHECK_EQ(f.ctx.set_level(-1), RAD_E_INVAL);
    std::vector<int64_t> moved = off;
    moved[1] = 512;                                   /* the fixed buffer moved */
    CHECK_EQ(f.ctx.add_level(16, moved, decl), RAD_E_STATE);
    std::vector<int64_t> low = off;
    low[2] = 512;                                     /* into the fixed region */
    CHECK_EQ(f.ctx.add_level(16, low, decl), RAD_E_STATE);
    std::vector<int64_t> high = off;
    high[3] = (int64_t)1 << 30;                       /* past the arena */
    CHECK_EQ(f.ctx.add_level(16, high, decl), RAD_E_STATE);
    CHECK_EQ(f.ctx.add_level(16, std::vector<int64_t>(2, 0), decl), RAD_E_INVAL);
}

/* The engine runs a step at the smallest level that holds it, and at the full plan (level 0) when
 * none does; a level too small writes a step's activations into lent memory. */
TEST(a_step_runs_at_the_smallest_arena_level_that_holds_it) {
    const std::vector<int64_t> rows = { 8192, 16, 256, 64 };
    CHECK_EQ(rad::arena_level_for(rows, 1), 1);
    CHECK_EQ(rad::arena_level_for(rows, 16), 1);
    CHECK_EQ(rad::arena_level_for(rows, 17), 3);
    CHECK_EQ(rad::arena_level_for(rows, 64), 3);
    CHECK_EQ(rad::arena_level_for(rows, 65), 2);
    CHECK_EQ(rad::arena_level_for(rows, 256), 2);
    CHECK_EQ(rad::arena_level_for(rows, 257), 0);
    CHECK_EQ(rad::arena_level_for({ 8192 }, 1), 0);
    CHECK_EQ(rad::arena_level_for({}, 1), 0);
}

/* ================================================================== the rank barrier */
/* abort() releases ranks parked on the barrier and every later arrival returns at once, so a
 * fault on one rank cannot leave the others parked and the engine can exit. */
TEST(an_aborted_rank_barrier_releases_its_waiters_and_every_later_arrival) {
    rad::RankBarrier b(4);
    std::atomic<int> entered{0}, left{0};
    std::vector<std::thread> th;
    for (int i = 0; i < 3; ++i)
        th.emplace_back([&] { entered.fetch_add(1); b.arrive_and_wait(); left.fetch_add(1); });
    while (entered.load() < 3) std::this_thread::yield();
    b.abort();
    for (auto& t : th) t.join();
    CHECK_EQ(left.load(), 3);
    for (int i = 0; i < 5; ++i) b.arrive_and_wait();   /* alone, four short: returns anyway */
    CHECK(true);
}

/* No rank leaves round r before every rank has arrived at it, and none reaches round r + 2 while
 * another is still in r: a fast rank re-entering at once waits on the next generation. */
TEST(no_rank_passes_a_barrier_round_before_every_rank_reached_it) {
    const int n = 4, rounds = 2000;
    rad::RankBarrier b(n);
    std::atomic<int> arrived[4];
    for (auto& a : arrived) a.store(0);
    std::atomic<int> bad{0};
    std::vector<std::thread> th;
    for (int t = 0; t < n; ++t)
        th.emplace_back([&, t] {
            for (int r = 0; r < rounds; ++r) {
                arrived[t].store(r + 1, std::memory_order_release);
                b.arrive_and_wait();
                for (int j = 0; j < n; ++j) {
                    const int a = arrived[j].load(std::memory_order_acquire);
                    if (a < r + 1 || a > r + 2) bad.fetch_add(1);
                }
            }
        });
    for (auto& t : th) t.join();
    CHECK_EQ(bad.load(), 0);
}

/* ================================================================== host-site ops and new lanes
 *
 * On the host backend each stream is a worker thread and device memory is host memory, so an
 * unordered reader really runs ahead: a chain of large copies holds the writer back long enough
 * for an early read to see the poison. */
namespace {
const int64_t kOrdBytes = 4 << 20;
uint8_t* g_ord_src = nullptr;      /* the chain's input, then the value to write */
uint8_t* g_ord_big = nullptr;      /* where the chain writes */
/* Pinned host memory, which a card's copies read and write as the host backend's do. */
int32_t* g_ord_value = nullptr;    /* what the late write writes */
int32_t* g_ord_raw = nullptr;      /* [4] memory the arena does not track */
int32_t* g_ord_stage = nullptr;    /* [4] staging for device-arena reads and writes */
int32_t  g_ord_seen = -1;

/* A device op: a chain of copies, then the value into its output operand, all on its stream. */
int k_late_write(const RadArgs* a, RadStream s) {
    for (int i = 0; i < 16; ++i) rad_memcpy_async(g_ord_big, g_ord_src, kOrdBytes, s);
    rad_memcpy_async(a->t[0].data, g_ord_value, sizeof *g_ord_value, s);
    return record("late_write", a, s);
}
/* A host op: reads its operand at launch, on the issuing thread. */
int k_host_read(const RadArgs* a, RadStream s) {
    std::memcpy(&g_ord_seen, a->t[0].data, sizeof g_ord_seen);
    return record("host_read", a, s);
}
/* A device op on whichever lane it is issued: copies operand 0 to operand 1 on its stream. */
int k_copy(const RadArgs* a, RadStream s) {
    rad_memcpy_async(a->t[1].data, a->t[0].data, sizeof(int32_t), s);
    return record("copy", a, s);
}

/* A host row for every band of `op`, from a library that is not the reference. */
void attach_host_row(Fixture& f, rad_op op, RadLaunchFn fn, const char* name) {
    for (rad::Band& band : f.P.ops[op].bands) {
        f.infos.push_back(RadKernelInfo{});
        RadKernelInfo& ki = f.infos.back();
        ki.name = name; ki.op = "rope"; ki.family = "test"; ki.domain = RAD_DOMAIN_HOST;
        ki.launch = fn;
        f.rows.push_back(rad::KernelRow{});
        rad::KernelRow& kr = f.rows.back();
        kr.info = &ki;
        kr.plugin = "fake";
        band.dom[RAD_DOMAIN_HOST].row = &kr;
        band.dom[RAD_DOMAIN_HOST].geom.set_i("M", band.hi);
        band.miss[RAD_DOMAIN_HOST].clear();
    }
}

/* An op with a host row only, which the host backend runs on the host. */
rad_op host_op(Fixture& f, RadLaunchFn fn, const char* name) {
    rad_op op = f.add_op("rope", "M", { 8192 }, { nullptr }, { nullptr }, nullptr, {});
    attach_host_row(f, op, fn, name);
    return op;
}

void ord_setup() {
    if (!g_ord_src) {
        g_ord_src = (uint8_t*)rad_dev_alloc(kOrdBytes, RAD_MEM_HOST_PINNED);
        g_ord_big = (uint8_t*)rad_dev_alloc(kOrdBytes, RAD_MEM_DEVICE);
        g_ord_value = (int32_t*)rad_dev_alloc(sizeof(int32_t), RAD_MEM_HOST_PINNED);
        g_ord_raw = (int32_t*)rad_dev_alloc(4 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        g_ord_stage = (int32_t*)rad_dev_alloc(4 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        std::memset(g_ord_src, 0x5A, (size_t)kOrdBytes);
    }
}

/* One int32 of device memory written or read through the stream, so the access is valid on a
 * card's device arena as on the host backend's. */
void dev_put(Fixture& f, void* dst, int32_t v) {
    *g_ord_stage = v;
    CHECK_OK(rad_memcpy_async(dst, g_ord_stage, sizeof v, f.ctx.stream()));
    CHECK_OK(rad_stream_sync(f.ctx.stream()));
}
int32_t dev_get(Fixture& f, const void* src) {
    *g_ord_stage = 0;
    CHECK_OK(rad_memcpy_async(g_ord_stage, src, sizeof(int32_t), f.ctx.stream()));
    CHECK_OK(rad_stream_sync(f.ctx.stream()));
    return *g_ord_stage;
}
}  /* namespace */

/* A host-site op reads what device ops issued before it wrote into the host arena: it waits on
 * the last device op that touched the host arena, without draining the stream. Unordered, a host
 * gather reads row ids a device op has not written yet: valid ids, wrong rows. */
TEST(a_host_op_reads_what_an_earlier_device_op_wrote_to_the_host_arena) {
    ord_setup();
    Fixture f;
    rad_buf hb = f.add_buffer("ids", RAD_I32, 4, 1, 0);
    f.P.buffers[hb].decl.domain = RAD_DOMAIN_HOST;
    rad_op dev = f.add_op("rope", "M", { 8192 }, { k_late_write }, { "late_write" }, nullptr, {});
    rad_op hst = host_op(f, k_host_read, "host_read");
    CHECK_OK(f.start());
    /* The host backend runs an op with no device row on the host already; a card needs asking. */
    REQUIRE(f.ctx.set_op_domain(hst, RAD_DOMAIN_HOST) == RAD_OK);
    for (int round = 0; round < 3; ++round) {
        *g_ord_value = 1000 + round;
        *(int32_t*)f.ctx.buf_ptr(hb) = -7;            /* the poison */
        g_ord_seen = -1;
        CHECK_OK(f.step(4, round, [&](Ctx& c) {
            const RadOperand o[1] = { RAD_B(hb) };
            CHECK_OK(c.issue(dev, o, 1, RAD_N_BATCH));
            CHECK_OK(c.issue(hst, o, 1, RAD_N_BATCH));
        }));
        CHECK_EQ(g_ord_seen, 1000 + round);
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
    }
    /* A wait on the device op's event, once per host op: a host block the runtime counts. */
    CHECK_EQ(f.ctx.host_syncs(), (int64_t)3);
}

/* A host op naming memory the arena does not track (a raw pointer) cannot know which device op
 * wrote it, so it drains the stream first and still reads the written value. */
TEST(a_host_op_on_untracked_memory_drains_the_stream_first) {
    ord_setup();
    Fixture f;
    rad_op dev = f.add_op("rope", "M", { 8192 }, { k_late_write }, { "late_write" }, nullptr, {});
    rad_op hst = host_op(f, k_host_read, "host_read");
    CHECK_OK(f.start());
    REQUIRE(f.ctx.set_op_domain(hst, RAD_DOMAIN_HOST) == RAD_OK);
    for (int round = 0; round < 3; ++round) {
        *g_ord_value = 2000 + round;
        g_ord_raw[0] = -7;
        g_ord_seen = -1;
        CHECK_OK(f.step(4, round, [&](Ctx& c) {
            const RadOperand o[1] = { RAD_P_T(g_ord_raw, RAD_I32) };
            CHECK_OK(c.issue(dev, o, 1, RAD_N_BATCH));
            CHECK_OK(c.issue(hst, o, 1, RAD_N_BATCH));
        }));
        CHECK_EQ(g_ord_seen, 2000 + round);
    }
}

/* Where an operand lies, as the engine knows it (RadTensor::space, ABI 29). A plugin that accounts
 * for what it read has to tell a link read from a device one, and the engine knows what the driver
 * does not: the backend locks its host pool itself and hsa_amd_pointer_info reports those ranges as
 * unknown (core/device/aql.cpp). So the operand says host for a host-domain buffer, device for one
 * in the arena, and unknown for a raw pointer the core never allocated -- never host by default. */
TEST(an_operand_carries_the_space_the_engine_placed_it_in) {
    ord_setup();
    Fixture f;
    rad_buf hb = f.add_buffer("ids", RAD_I32, 8, 1, 0);
    f.P.buffers[hb].decl.domain = RAD_DOMAIN_HOST;
    rad_buf db = f.add_buffer("dv", RAD_I32, 8, 1, 256);
    rad_op op = f.add_op("rope", "M", { 8192 }, { k_m64 }, { "m64" }, nullptr, {});
    CHECK_OK(f.start());
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        const RadOperand o[3] = { RAD_B(hb), RAD_B(db), RAD_P_T(g_ord_raw, RAD_I32) };
        CHECK_OK(c.issue(op, o, 3, RAD_N_BATCH));
    }));
    REQUIRE(!g_rec.empty());
    CHECK_EQ(g_rec.back().t[0].space, RAD_SPACE_HOST);
    CHECK_EQ(g_rec.back().t[1].space, RAD_SPACE_DEVICE);
    CHECK_EQ(g_rec.back().t[2].space, RAD_SPACE_UNKNOWN);
}

/* A second lane built mid-step starts behind everything lane 0 issued, even though the join that
 * asked for it ran before the lane existed. Unordered, the first picture after startup came out
 * different from every later one. */
TEST(a_lane_built_mid_step_starts_behind_lane_zero) {
    ord_setup();
    Fixture f;
    rad_buf x = f.add_buffer("x", RAD_I32, 4, 1, 0);
    rad_buf y = f.add_buffer("y", RAD_I32, 4, 1, 256);
    rad_op wr = f.add_op("rope", "M", { 8192 }, { k_late_write }, { "late_write" }, nullptr, {});
    rad_op cp = f.add_op("rope", "M", { 8192 }, { k_copy }, { "copy" }, nullptr, {});
    /* No scratch: two scratch users on two lanes are ordered by the scratch rule instead. */
    f.P.ops[wr].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = 0;
    f.P.ops[cp].bands[0].dom[RAD_DOMAIN_DEVICE].scratch_bytes = 0;
    CHECK_OK(f.start());
    *g_ord_value = 4242;
    dev_put(f, f.ctx.buf_ptr(x), -7);
    dev_put(f, f.ctx.buf_ptr(y), -8);
    CHECK_OK(f.step(4, 0, [&](Ctx& c) {
        const RadOperand w[1] = { RAD_B(x) };
        CHECK_OK(c.issue(wr, w, 1, RAD_N_BATCH));
        CHECK_OK(c.lane_join(0, 1));                  /* no lane 1 yet: nothing to order */
        CHECK_OK(c.set_lane(1));                      /* built here, on first use */
        const RadOperand o[2] = { RAD_B(x), RAD_B(y) };
        CHECK_OK(c.issue(cp, o, 2, RAD_N_BATCH));
        CHECK_OK(c.set_lane(0));
    }));
    CHECK_EQ(dev_get(f, f.ctx.buf_ptr(y)), 4242);
}

/* A calibration tap's gram_accum sums across passes while its plugin counts the summed rows in
 * step; a played pass reissues the sum without calling step (#122). A program holding one records
 * no passes; one without records them wherever the backend can. */
TEST(a_program_with_a_calibration_tap_records_no_passes) {
    {
        Fixture f;
        f.add_buffer("x", RAD_BF16, 64, 4, 0);
        loose_op(f);
        CHECK_OK(f.start(nullptr, false, /*record=*/true));
        CHECK_EQ(f.ctx.records_passes(), rad::tape_supported());
    }
    Fixture f;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    loose_op(f);
    loose_op(f, "gram_accum");
    CHECK_OK(f.start(nullptr, false, /*record=*/true));
    CHECK(!f.ctx.records_passes());
}

/* ================================================================== the pass key */
/* A recorded pass is replayed while its key holds, so the key must change with every batch field
 * and runtime state a pass's shape or arguments read, and must not change with the step number
 * alone. A field missing from the key replays stale arguments until the next audit. A pass that
 * cannot be recorded (encoder, generator, encoder rows) keys to 0. */
TEST(the_pass_key_changes_with_every_field_a_pass_depends_on) {
    Fixture f;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    rad_op op = loose_op(f);
    attach_host_row(f, op, k_host_read, "host_read");
    f.P.scratch_bytes = 256;
    CHECK_OK(f.start());

    static int32_t arr[16][8];
    RadKVGroupBatch kv[2]{};
    RadBatch b{};
    auto fill = [&]() {
        std::memset(&b, 0, sizeof b);
        std::memset(kv, 0, sizeof kv);
        b.phase = RAD_PHASE_DECODE; b.step = 5; b.n_tok = 4; b.n_seq = 2; b.n_out = 2;
        b.max_q_len = 2; b.max_ctx_len = 1024; b.n_kv_groups = 2; b.kv = kv; b.n_spec = 1;
        b.n_seq_decode = 2; b.n_tok_decode = 4; b.max_q_len_decode = 2;
        int k = 0;
        b.token_ids = arr[k++]; b.positions = arr[k++]; b.cu_seqlens = arr[k++];
        b.seq_ids = arr[k++]; b.out_ids = arr[k++]; b.q_lens = arr[k++]; b.ctx_lens = arr[k++];
        b.num_accepted = arr[k++];
        for (int g = 0; g < 2; ++g) {
            kv[g].group = g; kv[g].slot_mapping = arr[k++]; kv[g].block_table = arr[k++];
            kv[g].block_table_pitch = 64; kv[g].seqused = arr[k++]; kv[g].state_index = arr[k++];
            kv[g].state_index_pitch = 1; kv[g].block_size = 16; kv[g].max_blocks = 64;
        }
    };
    fill();
    const uint64_t k0 = f.ctx.pass_key(&b);
    CHECK(k0 != 0);
    b.step = 6;
    CHECK_EQ(f.ctx.pass_key(&b), k0);                 /* the step number alone is not a dependency */

    int changed = 0, total = 0;
    auto expect_change = [&](const char* what, const std::function<void()>& mutate) {
        fill();
        mutate();
        ++total;
        const bool moved = f.ctx.pass_key(&b) != k0;
        changed += moved;
        if (!moved) std::fprintf(stderr, "    the pass key ignores %s\n", what);
    };
    static const int32_t other[4] = {};
    expect_change("phase", [&] { b.phase = RAD_PHASE_PREFILL; });
    expect_change("n_tok", [&] { b.n_tok = 5; });
    expect_change("n_seq", [&] { b.n_seq = 3; });
    expect_change("n_out", [&] { b.n_out = 3; });
    expect_change("max_q_len", [&] { b.max_q_len = 3; });
    expect_change("max_ctx_len", [&] { b.max_ctx_len = 2048; });
    expect_change("n_kv_groups", [&] { b.n_kv_groups = 1; });
    expect_change("n_spec", [&] { b.n_spec = 2; });
    expect_change("draft_pass", [&] { b.draft_pass = -1; });
    expect_change("n_checkpoints", [&] { b.n_checkpoints = 1; });
    expect_change("n_seq_decode", [&] { b.n_seq_decode = 1; });
    expect_change("n_tok_decode", [&] { b.n_tok_decode = 3; });
    expect_change("max_q_len_decode", [&] { b.max_q_len_decode = 3; });
    expect_change("max_q_len_prefill", [&] { b.max_q_len_prefill = 7; });
    expect_change("n_draft_out", [&] { b.n_draft_out = 1; });
    expect_change("n_ahead", [&] { b.n_ahead = 9; });
    expect_change("rope_mixed", [&] { b.rope_mixed = 1; });
    expect_change("rope_pos", [&] { b.rope_pos = other; });
    expect_change("token_ids", [&] { b.token_ids = other; });
    expect_change("positions", [&] { b.positions = other; });
    expect_change("cu_seqlens", [&] { b.cu_seqlens = other; });
    expect_change("seq_ids", [&] { b.seq_ids = other; });
    expect_change("out_ids", [&] { b.out_ids = other; });
    expect_change("q_lens", [&] { b.q_lens = other; });
    expect_change("ctx_lens", [&] { b.ctx_lens = other; });
    expect_change("num_accepted", [&] { b.num_accepted = other; });
    expect_change("spec_parent", [&] { b.spec_parent = other; });
    expect_change("checkpoint_tok", [&] { b.checkpoint_tok = other; });
    expect_change("checkpoint_slot", [&] { b.checkpoint_slot = other; });
    expect_change("draft_out_ids", [&] { b.draft_out_ids = other; });
    for (int g = 0; g < 2; ++g) {
        expect_change("kv.group", [&] { kv[g].group = 7; });
        expect_change("kv.slot_mapping", [&] { kv[g].slot_mapping = (int32_t*)other; });
        expect_change("kv.block_table", [&] { kv[g].block_table = (int32_t*)other; });
        expect_change("kv.block_table_pitch", [&] { kv[g].block_table_pitch = 128; });
        expect_change("kv.seqused", [&] { kv[g].seqused = (int32_t*)other; });
        expect_change("kv.state_index", [&] { kv[g].state_index = (int32_t*)other; });
        expect_change("kv.state_index_pitch", [&] { kv[g].state_index_pitch = 2; });
        expect_change("kv.block_size", [&] { kv[g].block_size = 32; });
        expect_change("kv.max_blocks", [&] { kv[g].max_blocks = 65; });
    }
    CHECK_EQ(changed, total);

    /* Runtime state the issue path reads: the op domains and a second lane. */
    fill();
    CHECK_OK(f.ctx.set_op_domain(op, RAD_DOMAIN_HOST));
    CHECK(f.ctx.pass_key(&b) != k0);
    const uint64_t k1 = f.ctx.pass_key(&b);
    CHECK_OK(f.ctx.lane_init());
    CHECK(f.ctx.pass_key(&b) != k1);

    /* Never recorded. */
    fill(); b.enc = 1;        CHECK_EQ(f.ctx.pass_key(&b), (uint64_t)0);
    fill(); b.gen = 1;        CHECK_EQ(f.ctx.pass_key(&b), (uint64_t)0);
    fill(); b.n_mm_rows = 3;  CHECK_EQ(f.ctx.pass_key(&b), (uint64_t)0);
}

/* The arena level is part of the key: a pass recorded at one level carries its buffers' addresses
 * and must not replay at another. */
TEST(the_pass_key_changes_with_the_arena_level) {
    Fixture f;
    f.P.arena_bytes = 8192;
    f.add_buffer("x", RAD_BF16, 64, 4, 0);
    CHECK_OK(f.start());
    std::vector<int64_t> off = { 0, 4096 };
    std::vector<RadBufDecl> decl(2);
    decl[1] = f.P.buffers[1].decl;
    decl[1].shape[0] = 16;
    REQUIRE_EQ(f.ctx.add_level(16, off, decl), 1);
    RadBatch b{};
    b.n_tok = 1; b.n_seq = 1;
    const uint64_t k0 = f.ctx.pass_key(&b);
    CHECK_OK(f.ctx.set_level(1));
    CHECK(f.ctx.pass_key(&b) != k0);
    CHECK_OK(f.ctx.set_level(0));
    CHECK_EQ(f.ctx.pass_key(&b), k0);
}

/* ================================================================== the op oracle */
namespace {
std::string radiance_home() {
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "";
}

/* y = a + b, and the same plus one: a right kernel and a wrong one, f32 [M, 8]. */
template <int Bias>
int k_add_f32(const RadArgs* a, RadStream s) {
    const float* x = (const float*)a->t[0].data;
    const float* y = (const float*)a->t[1].data;
    float* o = (float*)a->t[2].data;
    for (int64_t i = 0; i < a->t[2].shape[0] * 8; ++i) o[i] = x[i] + y[i] + (float)Bias;
    return record(Bias ? "add_wrong" : "add_right", a, s);
}
/* Right, but leaves its result on the lane's stream behind a chain of copies. */
int k_add_late(const RadArgs* a, RadStream s) {
    static float* tmp = (float*)rad_dev_alloc(64 * sizeof(float), RAD_MEM_HOST_PINNED);
    const float* x = (const float*)a->t[0].data;
    const float* y = (const float*)a->t[1].data;
    for (int64_t i = 0; i < a->t[2].shape[0] * 8; ++i) tmp[i] = x[i] + y[i];
    for (int i = 0; i < 16; ++i) rad_memcpy_async(g_ord_big, g_ord_src, kOrdBytes, s);
    rad_memcpy_async(a->t[2].data, tmp, a->t[2].shape[0] * 8 * (int64_t)sizeof(float), s);
    return record("add_late", a, s);
}
/* Not a function of its inputs: adds how many times it has run. */
int k_add_counter(const RadArgs* a, RadStream s) {
    static int calls = 0;
    ++calls;
    const float* x = (const float*)a->t[0].data;
    const float* y = (const float*)a->t[1].data;
    float* o = (float*)a->t[2].data;
    for (int64_t i = 0; i < a->t[2].shape[0] * 8; ++i) o[i] = x[i] + y[i] + (float)calls;
    return record("add_counter", a, s);
}

struct OracleRun {
    std::string log;
    int status = RAD_OK;
};

/* One issue of `add` over f32 [4, 8] with the oracle (or repeat mode) armed through `env`, the
 * device row `dev` (null for none) and a host row `host` from a library that is not the
 * reference. `lane` issues it on that lane. */
OracleRun oracle_run(rad::Registry* reg, const char* env, RadLaunchFn dev, RadLaunchFn host,
                     int lane = 0) {
    OracleRun out;
    Fixture f;
    f.registry = reg;
    rad_buf a = f.add_buffer("a", RAD_F32, 4, 8, 0);
    rad_buf b = f.add_buffer("b", RAD_F32, 4, 8, 256);
    rad_buf y = f.add_buffer("y", RAD_F32, 4, 8, 512);
    f.P.arena_bytes = 1024;
    rad_op op = f.add_op("add", "M", { 8192 }, { dev }, { "add_device" }, nullptr, {});
    rad::OpInfo& oi = f.P.ops[op];
    oi.base = rad::Geometry();
    oi.base.set_i("n", 8);
    oi.base.set_s("dtype", "f32");
    if (reg) oi.schema = reg->schema("add");
    if (dev) {
        rad::Resolved& r = oi.bands[0].dom[RAD_DOMAIN_DEVICE];
        r.geom = oi.base;
        r.geom.set_i("M", 8192);
    }
    if (host) {
        f.infos.push_back(RadKernelInfo{});
        RadKernelInfo& ki = f.infos.back();
        ki.name = "add_host_wrong"; ki.op = "add"; ki.family = "test";
        ki.domain = RAD_DOMAIN_HOST; ki.launch = host;
        f.rows.push_back(rad::KernelRow{});
        rad::KernelRow& kr = f.rows.back();
        kr.info = &ki;
        kr.plugin = "hostlib";                          /* not a reference library */
        rad::Resolved& r = oi.bands[0].dom[RAD_DOMAIN_HOST];
        r.row = &kr;
        r.geom = oi.base;
        r.geom.set_i("M", 8192);
        oi.bands[0].miss[RAD_DOMAIN_HOST].clear();
    }
    const char* var = std::strchr(env, '=');
    const std::string key(env, (size_t)(var - env));
    setenv(key.c_str(), var + 1, 1);
    out.log = stderr_of([&] {
        out.status = f.start();
        if (out.status < 0) return;
        float* pa = (float*)f.ctx.buf_ptr(a);
        float* pb = (float*)f.ctx.buf_ptr(b);
        for (int i = 0; i < 32; ++i) { pa[i] = (float)i; pb[i] = 0.5f * (float)i; }
        out.status = f.step(4, 0, [&](Ctx& c) {
            CHECK_OK(c.set_lane(lane));
            const RadOperand o[3] = { RAD_B(a), RAD_B(b), RAD_B(y) };
            CHECK_OK(c.issue(op, o, 3, RAD_N_BATCH));
            CHECK_OK(c.set_lane(0));
        });
        CHECK_OK(rad_stream_sync(f.ctx.stream()));
        f.ctx.shutdown();
    });
    unsetenv(key.c_str());
    return out;
}

/* The fake kernels below compute on the host, reading and writing the arena in place, which only
 * the host backend allows. The oracle's comparison does not depend on the backend. */
const char* const kOracleHostOnly =
    "the fake kernels compute on the host over the arena, which a card's arena is not";

struct RefLib {
    rad::Registry reg;
    bool ok = false;
    RefLib() { ok = !radiance_home().empty() &&
                    reg.load_plugin(radiance_home() + "/kernels/libref.so", 0) >= 0 &&
                    reg.schema("add") != nullptr; }
};
}  /* namespace */

/* The oracle checks a kernel against the reference library, not against whichever host library
 * won the band. Here the only host row is another library's and wrong (it adds one); the oracle
 * must report it, where comparing the issued row with the band's host row compares it with
 * itself and agrees. */
TEST(the_op_oracle_checks_against_the_reference_library_not_the_bands_host_row) {
    RefLib ref;
    if (!rad::device_is_host()) SKIP("%s", kOracleHostOnly);
    if (!ref.ok) ARTIFACT_MISSING(radiance_home() + "/kernels/libref.so");
    const OracleRun r = oracle_run(&ref.reg, "RADIANCE_OP_ORACLE=add", nullptr, k_add_f32<1>);
    CHECK_OK(r.status);
    CHECK(has(r.log, "oracle #"));
    CHECK(has(r.log, "op 'add' MISMATCH"));
    CHECK(has(r.log, "opd2"));
    CHECK(has(r.log, "host ref_add"));
    if (!has(r.log, "MISMATCH")) std::fprintf(stderr, "%s", r.log.c_str());
}

/* A device kernel that agrees with the reference is reported as agreeing, one that does not as a
 * mismatch naming the op and the operand; an oracle that cannot tell them apart reports
 * "agrees" for everything. */
TEST(the_op_oracle_tells_a_right_device_kernel_from_a_wrong_one) {
    RefLib ref;
    if (!rad::device_is_host()) SKIP("%s", kOracleHostOnly);
    if (!ref.ok) ARTIFACT_MISSING(radiance_home() + "/kernels/libref.so");
    const OracleRun good = oracle_run(&ref.reg, "RADIANCE_OP_ORACLE=add", k_add_f32<0>, nullptr);
    CHECK_OK(good.status);
    CHECK(has(good.log, "op 'add' agrees"));
    CHECK(!has(good.log, "MISMATCH"));
    const OracleRun bad = oracle_run(&ref.reg, "RADIANCE_OP_ORACLE=add", k_add_f32<1>, nullptr);
    CHECK(has(bad.log, "op 'add' MISMATCH"));
    CHECK(has(bad.log, "opd2"));
    /* Not armed for this op: nothing is checked. */
    const OracleRun off = oracle_run(&ref.reg, "RADIANCE_OP_ORACLE=mul", k_add_f32<1>, nullptr);
    CHECK(!has(off.log, "oracle #"));
}

/* A kernel issued on the second lane is read back after it has run: the snapshot drains that
 * lane, so a right kernel whose write is still queued is not reported as wrong. */
TEST(the_op_oracle_reads_a_second_lanes_output_after_its_kernel) {
    RefLib ref;
    if (!rad::device_is_host()) SKIP("%s", kOracleHostOnly);
    if (!ref.ok) ARTIFACT_MISSING(radiance_home() + "/kernels/libref.so");
    ord_setup();
    const OracleRun r = oracle_run(&ref.reg, "RADIANCE_OP_ORACLE=add", k_add_late, nullptr, 1);
    CHECK_OK(r.status);
    CHECK(has(r.log, "op 'add' agrees"));
}

/* Repeat mode runs the device kernel twice on the same bytes and flags one whose output depends
 * on anything but its inputs; a kernel that is a function of its inputs passes. */
TEST(repeat_mode_flags_a_kernel_that_is_not_a_function_of_its_inputs) {
    if (!rad::device_is_host()) SKIP("%s", kOracleHostOnly);
    const OracleRun bad = oracle_run(nullptr, "RADIANCE_OP_REPEAT=add", k_add_counter, nullptr);
    CHECK_OK(bad.status);
    CHECK(has(bad.log, "op 'add' (kernel add_device) is not a function of its inputs"));
    CHECK(has(bad.log, "opd2"));
    const OracleRun good = oracle_run(nullptr, "RADIANCE_OP_REPEAT=add", k_add_f32<0>, nullptr);
    CHECK_OK(good.status);
    CHECK(!has(good.log, "repeat #"));
    CHECK(has(good.log, "1 issue(s) run twice, 0 of them answered differently"));
}

RAD_TEST_MAIN()
