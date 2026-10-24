/* build_test.cpp -- the declare phase.
 *
 * Against a REAL Registry loading the real test plugin, because that is the only way this test
 * checks the thing it claims to: a fake selector would be a second implementation of the bucket
 * table, and the two would agree with each other and with nothing else. core/plugin/testplugin/
 * already carries spec §2.2's worked example -- t_gemm with rows at M<=16, M<=64 and unconstrained
 * -- a range with a hole in it, an op with no host kernel, and an init() that refuses.
 *
 * What is under test is the five things declare produces (spec §3.1) and the two properties they
 * rest on: that the phase always completes, and that everything it decides it decides once.
 */
#include "rad_test.h"

#include <cstring>

#include <nlohmann/json.hpp>

#include "build/rad_build.h"
#include "build/startup.h"
#include "rad_internal.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

using namespace rad;
using json = nlohmann::ordered_json;

namespace {

/* $RADIANCE_HOME is what ctest sets; RAD_TESTPLUGIN_HOME overrides it for a hand-built run. Same
 * spelling as plugin_test.cpp, deliberately: two tests that locate the plugins differently is one
 * more thing to fix when the layout moves. */
std::string home() {
    if (const char* d = std::getenv("RAD_TESTPLUGIN_HOME")) return d;
    const char* h = std::getenv("RADIANCE_HOME");
    return h ? h : "radiance_home";
}
std::string kernels_dir() { return home() + "/testkernels"; }

int load(Registry& r, std::vector<std::string> hierarchy = { "radtest_over", "radtest_base" }) {
    const int st = r.load_dir(kernels_dir(), hierarchy);
    if (st == RAD_E_IO)
        fprintf(stderr, "    (no test plugins in %s -- build the radtest_* targets)\n",
                kernels_dir().c_str());
    return st;
}

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

RadModelMeta make_meta() {
    RadModelMeta m{};
    m.arch_id  = "test_arch";
    m.name     = "test_model";
    m.quant    = "bf16";
    m.n_layers = 2;
    m.n_embd   = 4096;
    return m;
}

RadBuildCtx make_ctx() {
    RadBuildCtx c{};
    c.rank = 0;
    c.world_size = 1;
    c.max_tok  = 8192;
    c.max_seqs = 256;
    c.max_ctx  = 4096;
    c.scope = "";
    return c;
}

/* t_gemm's operands are a(in), b(weight), y(out), so one weight handle. */
rad_weight decl_w(Builder& b, const char* name, int64_t d0 = 4096, int64_t d1 = 4096) {
    RadWeightDecl d{};
    d.dtype = RAD_BF16;
    d.rank  = 2;
    d.shape[0] = d0;
    d.shape[1] = d1;
    d.access = RAD_ACCESS_PER_TOKEN;
    d.shard  = RAD_SHARD_ROW;
    d.group  = rad_group_layer(0);
    return b.decl_weight(name, &d);
}

/* The §2.2 worked example, as an architecture plugin would declare it. */
rad_op decl_gemm(Builder& b, rad_weight w, int64_t max_m = 8192) {
    const RadParam p[] = { RAD_RANGE("M", 1, max_m), RAD_INT("N", 4096), RAD_INT("K", 4096),
                           RAD_STR("dtype", "bf16") };
    return b.decl_op("t_gemm", p, 4, &w, 1);
}

BufferInfo make_buf(const char* name, int64_t bytes, int kind, int32_t lo, int32_t hi) {
    BufferInfo b;
    b.name = name;
    b.decl.dtype = RAD_I8;
    b.decl.rank  = 1;
    b.decl.shape[0] = bytes;
    b.decl.kind = kind;
    b.decl.domain = RAD_DOMAIN_DEVICE;
    b.bytes = bytes;
    b.first_def = lo;
    b.last_use  = hi;
    return b;
}

}  /* namespace */

/* A TEST THAT SEGFAULTED IS NOT A TEST THAT REPORTED. Every case below drives a REAL Registry
 * over the test plugin, and on a tree where the radtest_* targets have not been built there is
 * no op for declare to resolve -- so the assertions that follow walk a handle the builder never
 * issued, and the binary dies without naming the reason. ctest keys "Skipped" off this exact
 * prefix (tests/CMakeLists.txt), which is the treatment chat_test and text_test already give a
 * missing model file, for the same reason. */
#define NEED_PLUGIN(reg)                                                              \
    Registry reg;                                                                     \
    if (load(reg) < 0) {                                                              \
        fprintf(stderr, "  SKIP %s: no test plugins in %s\n",                         \
                ::radtest::current(), kernels_dir().c_str());                         \
        return;                                                                       \
    }

/* ================================================================== the bucket table */
TEST(bucket_table_resolves_per_band) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "blk.0.ffn_gate.weight");
    CHECK(w == 1);
    rad_op h = decl_gemm(b, w);

    CHECK(h == 1);
    CHECK_EQ(b.op_resolved(h), 1);
    CHECK_EQ((int)b.errors().size(), 0);

    const OpInfo* o = b.program().op(h);
    CHECK_EQ((int)o->bands.size(), 3);
    if (o->bands.size() != 3) return;
    CHECK_EQ(o->bands[0].hi, 16);
    CHECK_EQ(o->bands[1].hi, 64);
    CHECK_EQ(o->bands[2].hi, 8192);

    /* The band picks the kernel, not the range: resolving once at max_tok would have picked the
     * unconstrained prefill row and then run it at M = 1, which is exactly the loss §2.2 exists
     * to recover. */
    CHECK_EQ(std::string(o->bands[0].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_gemm_m16");
    CHECK_EQ(std::string(o->bands[1].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_gemm_m64");
    CHECK_EQ(std::string(o->bands[2].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_gemm_any");

    /* Both domains are resolved wherever both exist, so the core can change execution site at
     * runtime without re-resolving (spec §5.1). */
    for (const Band& band : o->bands)
        CHECK_EQ(std::string(band.dom[RAD_DOMAIN_HOST].row->info->name), "t_gemm_host");

    /* Each band is frozen at its UPPER BOUND, which is the worst case in it. */
    long long m = 0;
    CHECK(o->bands[1].dom[RAD_DOMAIN_DEVICE].geom.get_i("M", &m));
    CHECK_EQ(m, 64);

    CHECK_EQ((int)b.program().misses.size(), 0);
    CHECK_OK(b.finish());

    /* Weight use is what makes layer offload exact prefetch rather than a prediction. */
    CHECK_EQ(b.program().weight(w)->first_use_op, 1);
    CHECK_EQ(b.program().weight(w)->last_use_op, 1);
}

TEST(an_op_with_no_ranged_parameter_is_one_band) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "w");
    const RadParam p[] = { RAD_INT("M", 8), RAD_INT("n", 4096) };
    rad_op h = b.decl_op("t_tie", p, 2, &w, 1);
    CHECK(h == 1);

    /* build_bands returns exactly one band whose hi is INT64_MAX, so the issue path's band scan
     * always finds it and the no-range case needs no separate path anywhere. */
    const OpInfo* o = b.program().op(h);
    CHECK_EQ((int)o->bands.size(), 1);
    CHECK_EQ(o->bands[0].hi, INT64_MAX);
    CHECK(o->ranged_key.empty());
    /* Ties break on declaration order, and this is the record of it. */
    CHECK_EQ(std::string(o->bands[0].dom[RAD_DOMAIN_DEVICE].row->info->name), "t_tie_first");
}

/* ================================================================== the graph dump, exactly */
TEST(graph_dump_matches_the_spec) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "blk.12.ffn_gate.weight");
    decl_gemm(b, w);
    CHECK_OK(b.finish());

    const std::string s = dump_graph_str(b.program(), &b.errors());

    /* The bucket-table block, byte for byte the format in spec §2.2 -- the op, its fixed
     * parameters, then one line per band naming the kernel, the plugin and the priority. If this
     * fails because the format drifted, the spec is the thing that is right.
     *
     * t_gemm_m64 declares a `bn` axis, so §15's clause appears on that line and only that line:
     * a kernel with no declared axis has nothing to say about tuning, and printing a
     * placeholder for it would bury the rows that do. */
    const char* expect =
        "t_gemm  N=4096 K=4096 dtype=bf16\n"
        "    M <=   16  ->  t_gemm_m16              (radtest_base, prio 30)\n"
        "    M <=   64  ->  t_gemm_m64              (radtest_base, prio 20, bn=64 from axis defaults)\n"
        "    M <= 8192  ->  t_gemm_any              (radtest_base, prio 10)\n";
    if (!has(s, expect)) {
        fprintf(stderr, "---- dump was ----\n%s------------------\n", s.c_str());
        CHECK(has(s, expect));
    }

    /* A host kernel exists for this op, so its table is printed too and labelled -- the device
     * one is unlabelled, which is what makes the block above §2.2's exactly. */
    CHECK(has(s, "    host domain:\n"));
    CHECK(has(s, "    M <= 8192  ->  t_gemm_host             (radtest_base, prio 10)\n"));

    /* The constraints that matched, the weights it touches, and the manifest with tier and site
     * left for the planner to fill. */
    CHECK(has(s, "      matched  t_gemm_m16: M <= 16, K % 256 == 0, dtype in {bf16 w4a8}\n"));
    CHECK(has(s, "      matched  t_gemm_any: (unconstrained"));
    CHECK(has(s, "      weights  blk.12.ffn_gate.weight\n"));
    CHECK(has(s, "tier"));
    CHECK(has(s, "---- tuning requests: 1"));
    CHECK(!has(s, "---- unresolved"));
}

/* ================================================================== the buffer plan */
TEST(liveness_packing) {
    Program p;
    p.buffers.emplace_back();        /* the sentinel */
    p.ops.resize(6);                 /* five ops plus the sentinel */

    /* a and b cannot be alive at the same time, so they must share bytes.
     * c and d overlap at op 2 and 3, so they must not. */
    p.buffers.push_back(make_buf("a", 1024, RAD_BUF_TRANSIENT, 1, 2));
    p.buffers.push_back(make_buf("b", 1024, RAD_BUF_TRANSIENT, 3, 4));
    p.buffers.push_back(make_buf("c", 2048, RAD_BUF_TRANSIENT, 1, 4));
    p.buffers.push_back(make_buf("d", 2048, RAD_BUF_TRANSIENT, 2, 3));
    p.buffers.push_back(make_buf("keep", 512, RAD_BUF_PERSIST, -1, -1));

    BufPlanReport rep;
    CHECK_OK(plan_buffers(p, &rep));

    const BufferInfo& a = p.buffers[1];
    const BufferInfo& bb = p.buffers[2];
    const BufferInfo& c = p.buffers[3];
    const BufferInfo& d = p.buffers[4];
    const BufferInfo& k = p.buffers[5];

    CHECK_EQ(a.arena_offset, bb.arena_offset);
    CHECK(c.arena_offset != d.arena_offset);
    CHECK(c.arena_offset + c.bytes <= d.arena_offset || d.arena_offset + d.bytes <= c.arena_offset);

    /* The sum is what no sharing at all would have cost, the peak is what a perfect packer would
     * reach, and the arena has to sit between them. */
    CHECK_EQ(rep.sum_bytes, 6144);
    CHECK_EQ(rep.peak_live, 5120);
    CHECK_EQ(rep.transient_bytes, 5120);

    /* A PERSIST buffer survives across steps, so it gets its own region -- at the BOTTOM, where
     * no smaller step's plan moves it, with the transients above. */
    CHECK_EQ(k.arena_offset, 0);
    CHECK_EQ(p.arena_fixed_bytes, 512);
    for (const BufferInfo* t : { &a, &bb, &c, &d }) CHECK(t->arena_offset >= p.arena_fixed_bytes);
    CHECK_EQ(p.arena_bytes, 5632);
    CHECK_EQ(rep.persist_bytes, 512);
    CHECK_EQ(rep.n_unknown_live, 0);
}

/* A SMALLER STEP'S PLAN: the same buffers, rows scaled, the fixed region untouched, and the
 * transients packed lower -- so the arena past `end` is what a step that small never reaches. */
TEST(a_smaller_step_packs_its_transients_lower_and_keeps_the_fixed_region) {
    auto mk = [](int64_t rows) {
        Program p;
        p.buffers.emplace_back();
        p.ops.resize(6);
        auto rowbuf = [&](const char* n, int64_t w, int32_t lo, int32_t hi) {
            BufferInfo b = make_buf(n, rows * w, RAD_BUF_TRANSIENT, lo, hi);
            b.decl.rank = 2;
            b.decl.shape[0] = rows;
            b.decl.shape[1] = w;
            return b;
        };
        p.buffers.push_back(rowbuf("x", 256, 1, 2));
        p.buffers.push_back(rowbuf("y", 512, 2, 4));
        p.buffers.push_back(rowbuf("z", 256, 3, 5));
        p.buffers.push_back(make_buf("state", 4096, RAD_BUF_PERSIST, -1, -1));
        return p;
    };
    Program full = mk(2048);
    CHECK_OK(plan_buffers(full));
    Program probe = mk(256);
    ArenaLevel lv;
    std::string why;
    CHECK_OK(plan_level(full, probe, 256, &lv, &why));
    CHECK_EQ(lv.rows, 256);
    CHECK_EQ(lv.offset[4], full.buffers[4].arena_offset);      /* the fixed region stays */
    for (int i = 1; i <= 3; ++i) {
        CHECK(lv.offset[(size_t)i] >= full.arena_fixed_bytes);
        CHECK(lv.offset[(size_t)i] + buffer_bytes(lv.decl[(size_t)i]) <= lv.end);
        CHECK_EQ(lv.decl[(size_t)i].shape[0], 256);
    }
    CHECK(lv.end < full.arena_bytes);
    CHECK(lv.end - full.arena_fixed_bytes <= (full.arena_bytes - full.arena_fixed_bytes) / 4);
    /* Two transients alive at once never share bytes at the smaller size either. */
    CHECK(lv.offset[2] + buffer_bytes(lv.decl[2]) <= lv.offset[3] ||
          lv.offset[3] + buffer_bytes(lv.decl[3]) <= lv.offset[2]);

    /* ONLY THE ROWS MAY SHRINK: a narrower second extent is a different stride. */
    Program wide = mk(256);
    wide.buffers[1].decl.shape[1] = 128;
    CHECK_EQ(plan_level(full, wide, 256, &lv, &why), RAD_E_STATE);
    CHECK(why.find("extent") != std::string::npos);
    /* A different buffer list is a different graph. */
    Program other = mk(256);
    other.buffers.pop_back();
    CHECK_EQ(plan_level(full, other, 256, &lv, &why), RAD_E_STATE);
    /* Persistent state sized by the step keeps its full size and its place at a smaller one... */
    Program shrank = mk(256);
    shrank.buffers[4].decl.shape[0] = 1024;
    CHECK_OK(plan_level(full, shrank, 256, &lv, &why));
    CHECK_EQ(lv.decl[4].shape[0], full.buffers[4].decl.shape[0]);
    CHECK_EQ(lv.offset[4], full.buffers[4].arena_offset);
    /* ...and one LARGER at a smaller step was never sized by the step at all. */
    Program grew = mk(256);
    grew.buffers[4].decl.shape[0] = 8192;
    CHECK_EQ(plan_level(full, grew, 256, &lv, &why), RAD_E_STATE);
}

TEST(unknown_liveness_is_conservative) {
    Program p;
    p.buffers.emplace_back();
    p.ops.resize(4);

    /* Nothing declared a use, so nothing may share: an op may issue any handle obtained during
     * declare, in any order, and declaration position proves nothing about when it is read. */
    p.buffers.push_back(make_buf("x", 1024, RAD_BUF_TRANSIENT, -1, -1));
    p.buffers.push_back(make_buf("y", 1024, RAD_BUF_TRANSIENT, -1, -1));

    BufPlanReport rep;
    CHECK_OK(plan_buffers(p, &rep));
    CHECK_EQ(rep.n_unknown_live, 2);
    CHECK_EQ(rep.transient_bytes, 2048);
    CHECK(p.buffers[1].arena_offset != p.buffers[2].arena_offset);
}

TEST(host_and_device_buffers_get_separate_offset_spaces) {
    Program p;
    p.buffers.emplace_back();
    p.ops.resize(4);

    BufferInfo d0 = make_buf("dev_a", 4096, RAD_BUF_TRANSIENT, 1, 3);
    BufferInfo h0 = make_buf("host_a", 4096, RAD_BUF_TRANSIENT, 1, 3);
    h0.decl.domain = RAD_DOMAIN_HOST;
    BufferInfo h1 = make_buf("host_b", 2048, RAD_BUF_TRANSIENT, 1, 3);
    h1.decl.domain = RAD_DOMAIN_HOST;
    p.buffers.push_back(d0);
    p.buffers.push_back(h0);
    p.buffers.push_back(h1);

    BufPlanReport rep;
    CHECK_OK(plan_buffers(p, &rep));

    /* A host-site activation cannot index device memory, so the domains are coloured separately
     * and each offset is relative to its own arena. Two live buffers in DIFFERENT domains may sit
     * at the same offset; two in the same domain may not. */
    CHECK_EQ(p.buffers[1].arena_offset, 0);
    CHECK_EQ(p.buffers[2].arena_offset, 0);
    CHECK(p.buffers[3].arena_offset != p.buffers[2].arena_offset);
    CHECK_EQ(p.arena_bytes, 4096);
    CHECK_EQ(p.host_arena_bytes, 4096 + 2048);
    CHECK_EQ(rep.host_bytes, 4096 + 2048);
}

TEST(buffer_uses_drive_liveness) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    RadBufDecl bd{};
    bd.dtype = RAD_I8;
    bd.rank = 1;
    bd.shape[0] = 4096;
    bd.kind = RAD_BUF_TRANSIENT;
    bd.domain = RAD_DOMAIN_DEVICE;
    rad_buf x = b.decl_buffer("x", &bd);
    rad_buf y = b.decl_buffer("y", &bd);
    CHECK(x == 1 && y == 2);

    rad_weight w = decl_w(b, "w0");
    rad_op o1 = decl_gemm(b, w);
    rad_op o2 = decl_gemm(b, w);
    rad_op o3 = decl_gemm(b, w);

    CHECK_OK(b.op_writes(o1, &x, 1));
    CHECK_OK(b.op_reads (o2, &x, 1));
    CHECK_OK(b.op_writes(o3, &y, 1));
    CHECK_OK(b.finish());

    CHECK_EQ(b.program().buffer(x)->first_def, 1);
    CHECK_EQ(b.program().buffer(x)->last_use, 2);
    CHECK_EQ(b.program().buffer(y)->first_def, 3);
    CHECK_EQ(b.buf_report().n_unknown_live, 0);
    /* x dies at op 2 and y is born at op 3, so one 4 KiB slot serves both. */
    CHECK_EQ(b.program().buffer(x)->arena_offset, b.program().buffer(y)->arena_offset);
    CHECK_EQ(b.program().arena_bytes, 4096);
}

/* ================================================================== declare errors */
/* The parameter side of the schema check belongs to Registry::validate, and the sentence it writes
 * is what the report carries. These tests assert that declare refuses, and that the reason reaches
 * the caller intact -- not that this component phrased it. */
TEST(two_ranged_parameters_is_an_error) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "w");
    const RadParam p[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096),
                           RAD_RANGE("K", 1, 4096), RAD_STR("dtype", "bf16") };
    rad_op h = b.decl_op("t_gemm", p, 4, &w, 1);

    CHECK_EQ(h, RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_SCHEMA);
    CHECK_EQ((int)b.errors().size(), 1);
    /* Both are named: "one of them is wrong" is not a diagnostic. */
    CHECK(has(b.errors()[0], "'M'"));
    CHECK(has(b.errors()[0], "'K'"));
    CHECK(has(b.errors()[0], "at most one ranged parameter"));
    /* The op is not recorded, so the run phase cannot reach it. */
    CHECK_EQ((int)b.program().ops.size(), 1);
}

TEST(schema_mismatches_are_named) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    rad_weight w = decl_w(b, "w");

    /* Missing a required key. */
    const RadParam miss[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096),
                              RAD_STR("dtype", "bf16") };
    CHECK_EQ(b.decl_op("t_gemm", miss, 3, &w, 1), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "required parameter 'K' was not given"));

    /* A key nothing declared. */
    const RadParam extra[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                               RAD_STR("dtype", "bf16"), RAD_INT("tile", 128) };
    CHECK_EQ(b.decl_op("t_gemm", extra, 5, &w, 1), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "no parameter 'tile' in this op's schema"));

    /* The right key, the wrong type. */
    const RadParam wrong[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                               RAD_INT("dtype", 3) };
    CHECK_EQ(b.decl_op("t_gemm", wrong, 4, &w, 1), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "the schema says it is a string"));

    /* The wrong number of weight operands. Arguments are positional, so this is not a diagnosable
     * failure at issue -- it is silent numerical garbage, and it is caught here or not at all.
     * Registry::validate is handed the parameters and not the operands, so this half stays here. */
    const RadParam ok[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                            RAD_STR("dtype", "bf16") };
    const rad_weight two[] = { w, w };
    CHECK_EQ(b.decl_op("t_gemm", ok, 4, two, 2), RAD_NULL_HANDLE);
    CHECK(has(b.errors().back(), "2 weight operands declared"));

    CHECK_EQ(b.status(), RAD_E_SCHEMA);
}

TEST(no_schema_names_the_near_miss) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    const RadParam p[] = { RAD_INT("M", 8), RAD_INT("n", 4096) };
    CHECK_EQ(b.decl_op("t_tei", p, 2, nullptr, 0), RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_NOSCHEMA);
    /* Without a near miss, a typo and an unimplemented op are the same diagnostic (spec §2.3).
     * WHICH name it suggests is the registry's edit-distance policy and not this component's;
     * what is asserted here is that the sentence reaches the caller intact. */
    CHECK(has(b.errors().back(), "no loaded kernel plugin declares op 't_tei'"));
    CHECK(has(b.errors().back(), "did you mean"));
}

/* ================================================================== a weight list over its tables
 * A table operand consumes a RUN of the declared weights, and the declaration decides which run:
 * the fixed weight positions take one each, and the rest is divided over the table positions in
 * declaration order, the first positions taking one more where it does not divide. That map
 * (OpInfo::weight_opd) is what the layout pass, the converter and the issue path all read, so it
 * is asserted whole. t_moe is one fixed weight (operand 1) and eight tables -- class pairs at
 * operands 2/3, 5/6, 7/8 and 9/10 -- the way a four-class grouped GEMM lays its classes out. */
namespace {

/* `n` weights declared in a row under `tag`, and t_moe over them. */
rad_op decl_moe(Builder& b, const char* tag, int n, std::vector<rad_weight>* w) {
    w->clear();
    for (int i = 0; i < n; ++i)
        w->push_back(decl_w(b, (std::string(tag) + ".e" + std::to_string(i)).c_str(), 64, 64));
    const RadParam p[] = { RAD_INT("M", 8), RAD_INT("n", 64) };
    return b.decl_op("t_moe", p, 2, w->data(), (int)w->size());
}

/* The operand each declared weight went to, as runs: "1x1 2x3" is one weight at operand 1, then
 * three at operand 2. */
std::string runs(const OpInfo* o) {
    std::string s;
    for (size_t i = 0; i < o->weight_opd.size();) {
        size_t j = i;
        while (j < o->weight_opd.size() && o->weight_opd[j] == o->weight_opd[i]) ++j;
        s += (s.empty() ? "" : " ") + std::to_string(o->weight_opd[i]) + "x" +
             std::to_string(j - i);
        i = j;
    }
    return s;
}

}  /* namespace */

TEST(a_weight_list_divides_over_its_tables_the_first_taking_the_remainder) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    std::vector<rad_weight> w;

    /* Ten experts' codes and scales behind the fixed weight: twenty entries over eight tables is
     * two each and four left, so the first four positions -- classes 0 and 1, codes and scales --
     * hold three and the last four two. A four-class grouped GEMM over ten experts, whose first
     * two classes are one expert longer. */
    rad_op h = decl_moe(b, "ten", 21, &w);
    REQUIRE(h != RAD_NULL_HANDLE);
    const OpInfo* o = b.program().op(h);
    REQUIRE(o != nullptr);
    CHECK_EQ(runs(o), std::string("1x1 2x3 3x3 5x3 6x3 7x2 8x2 9x2 10x2"));
    /* The list is cut into runs in the order it was declared, not dealt out. */
    CHECK(o->weights == w);

    /* A list that divides is the plain equal division: sixteen entries, two a table. */
    h = decl_moe(b, "eight", 17, &w);
    REQUIRE(h != RAD_NULL_HANDLE);
    CHECK_EQ(runs(b.program().op(h)), std::string("1x1 2x2 3x2 5x2 6x2 7x2 8x2 9x2 10x2"));

    /* One entry a table is the least a declaration can give. */
    h = decl_moe(b, "four", 9, &w);
    REQUIRE(h != RAD_NULL_HANDLE);
    CHECK_EQ(runs(b.program().op(h)), std::string("1x1 2x1 3x1 5x1 6x1 7x1 8x1 9x1 10x1"));
    CHECK_EQ((int)b.errors().size(), 0);
}

TEST(a_weight_list_shorter_than_its_tables_is_refused_by_name) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());
    std::vector<rad_weight> w;
    const size_t ops = b.program().ops.size();

    /* Seven entries for eight tables leaves one empty, which no division can fill: refused at
     * declare, with the counts that say so. */
    CHECK_EQ(decl_moe(b, "short", 8, &w), RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_SCHEMA);
    REQUIRE(!b.errors().empty());
    CHECK(has(b.errors().back(), "op 't_moe': 8 weight operands declared"));
    CHECK(has(b.errors().back(), "1 fixed weight operands and 8 weight TABLE operands"));
    CHECK(has(b.errors().back(), "so 7 leaves a table empty"));
    /* Nothing is recorded for the run phase to reach. */
    CHECK_EQ(b.program().ops.size(), ops);
}

/* ================================================================== misses, reported whole */
TEST(a_hole_in_the_range_is_reported_as_that_band) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    /* t_skinny has rows at M <= 64 and M >= 4096 and nothing in between. */
    rad_weight w = decl_w(b, "w");
    const RadParam p[] = { RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                           RAD_STR("dtype", "bf16") };
    rad_op h = b.decl_op("t_skinny", p, 4, &w, 1);

    /* The op is not handed back, so the plugin can offer the unfused sequence instead -- and it is
     * still recorded, so the graph dump shows the hole. Declare always completes. */
    CHECK_EQ(h, RAD_NULL_HANDLE);
    CHECK_EQ((int)b.program().ops.size(), 2);
    CHECK_EQ(b.status(), RAD_OK);

    const OpInfo& o = b.program().ops[1];
    CHECK(o.bands.size() >= 3);
    if (o.bands.size() < 3) return;
    CHECK(o.bands[0].dom[RAD_DOMAIN_DEVICE]);          /* M <= 64 */
    CHECK(!o.bands[1].dom[RAD_DOMAIN_DEVICE]);         /* the hole */
    CHECK(o.bands[2].dom[RAD_DOMAIN_DEVICE]);          /* M >= 4096 */

    /* The missing-kernel report is per band: the hole is named, not the whole op. */
    bool named = false;
    for (const auto& m : b.program().misses)
        if (m.op == "t_skinny" && has(m.geom, "M=4095")) named = true;
    CHECK(named);

    const std::string s = dump_graph_str(b.program());
    CHECK(has(s, "(nothing)"));
    CHECK(has(s, "---- unresolved:"));
}

TEST(an_op_with_no_host_kernel_is_a_miss_per_band_not_a_failure) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    /* t_attn is device-only: the asymmetric-absence case. No host kernel removes host-site
     * placement for this op and nothing else, so declare succeeds and the report says so. */
    const RadParam p[] = { RAD_RANGE("q_len", 1, 16), RAD_INT("head_dim", 128),
                           RAD_INT("window", 0), RAD_INT("block_size", 16),
                           RAD_INT("causal", 1), RAD_INT("n_head", 8),
                           RAD_STR("kv_dtype", "bf16") };
    rad_op h = b.decl_op("t_attn", p, 7, nullptr, 0);
    CHECK(h == 1);
    CHECK_EQ(b.op_resolved(h), 1);
    CHECK_EQ(b.status(), RAD_OK);

    bool host_miss = false;
    for (const auto& m : b.program().misses) if (m.domain == RAD_DOMAIN_HOST) host_miss = true;
    CHECK(host_miss);
}

/* ================================================================== init, after the whole graph */
TEST(a_refusing_init_becomes_a_miss_and_not_an_abort) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    /* t_eq_m32's init always fails. Declaring it must still complete, and the op after it must
     * still resolve -- if init ran during resolution, this second op would never be reached and
     * the miss report would be assembled from half a graph. */
    rad_weight w = decl_w(b, "w");
    const RadParam eq[] = { RAD_INT("M", 32), RAD_INT("n", 4096) };
    rad_op h_eq = b.decl_op("t_eq", eq, 2, &w, 1);
    rad_op h_gemm = decl_gemm(b, w);
    CHECK(h_eq == 1);
    CHECK(h_gemm == 2);

    /* Resolution succeeded: the kernel matched the constraints. */
    CHECK(b.program().op(h_eq)->bands[0].dom[RAD_DOMAIN_DEVICE]);

    const int st = b.finish();
    CHECK(st < 0);
    CHECK(has(b.errors().back(), "init() refused the instance"));

    /* The band is now a miss, so nothing downstream can launch a kernel whose one-time setup
     * failed -- and the rest of the graph is intact and reported. */
    CHECK(!b.program().op(h_eq)->bands[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK(b.program().op(h_gemm)->bands[0].dom[RAD_DOMAIN_DEVICE]);
    CHECK_EQ(b.op_resolved(h_eq), 0);
    CHECK_EQ(b.op_resolved(h_gemm), 1);
}

/* ================================================================== scratch and tuning */
TEST(tune_requests_are_one_per_kernel_and_shape) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    rad_weight w = decl_w(b, "w");
    /* Sixty-four layers ask the same question sixty-four times; rad-tune benchmarks it once. */
    decl_gemm(b, w);
    decl_gemm(b, w);
    CHECK_OK(b.finish());

    /* Only t_gemm_m64 declares an axis, and only its band's shape needs measuring. */
    CHECK_EQ((int)b.program().tune_requests.size(), 1);
    CHECK_EQ(b.program().tune_requests[0].kernel, std::string("t_gemm_m64"));
    CHECK_EQ(b.program().tune_requests[0].plugin, std::string("radtest_base"));
    CHECK_EQ(b.program().tune_requests[0].info->n_tunables, 1);
    CHECK(has(b.program().tune_requests[0].geom.str(), "M=64"));
}

/* ================================================================== KV groups */
TEST(kv_block_size_comes_off_the_kernel) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_BF16;
    kd.n_head_kv = 8;
    kd.head_dim = 128;
    rad_kvgroup g = b.decl_kv_group("kv_full", &kd);
    CHECK(g == 1);

    /* Before the consuming op exists there is no answer, and a silent zero here becomes a division
     * by zero in the block manager three components later. */
    CHECK_EQ(b.kv_block_size(g), 0);
    CHECK_EQ(b.status(), RAD_E_STATE);
    CHECK(has(b.errors().back(), "no op that reads this group"));

    const RadParam p[] = { RAD_RANGE("q_len", 1, 16), RAD_INT("head_dim", 128),
                           RAD_INT("window", 0), RAD_INT("block_size", 16),
                           RAD_INT("causal", 1), RAD_INT("n_head", 8),
                           RAD_STR("kv_dtype", "bf16") };
    rad_op h = b.decl_op("t_attn", p, 7, nullptr, 0);
    CHECK(h == 1);

    /* 16 because the resolved kernel is compiled for 16 and says so in its constraints, not
     * because the core picked a number (spec §7.2). */
    CHECK_EQ(b.kv_block_size(g), 16);

    CHECK_OK(b.bind_layer_kv(0, g));
    CHECK_OK(b.bind_layer_kv(1, g));
    b.finish();

    const KVGroupInfo& gi = b.program().kv_groups[g];
    CHECK_EQ(gi.block_size, 16);
    /* PINS: ONE PAGE COVERS EVERY LAYER BOUND TO THE GROUP, which is what KVManager documents a
     * page to be and what its own fallback formula computes. The manager divides the builder's
     * figure by the layer count, so a builder that sized a page for ONE layer gives a KV pool
     * n_layers too small and a layer base pointer at a fractional stride -- which surfaces as an
     * alignment refusal from the attention kernel at the second attention layer of the first real
     * step. Hence the `* 2`: two layers are bound above, and the factor is the point of the
     * check. */
    CHECK_EQ(gi.bytes_per_block, 2 * 16 * 8 * 128 * 2 * 2);
    CHECK_EQ((int)gi.layers.size(), 2);
}

TEST(a_kv_group_with_no_layers_is_an_error) {
    NEED_PLUGIN(reg)
    Builder b(reg, make_meta(), make_ctx());

    RadKVGroupDecl kd{};
    kd.kind = RAD_KV_FULL;
    kd.dtype = RAD_BF16;
    kd.n_head_kv = 8;
    kd.head_dim = 128;
    b.decl_kv_group("kv_full", &kd);
    const RadParam p[] = { RAD_RANGE("q_len", 1, 16), RAD_INT("head_dim", 128),
                           RAD_INT("window", 0), RAD_INT("block_size", 16),
                           RAD_INT("causal", 1), RAD_INT("n_head", 8),
                           RAD_STR("kv_dtype", "bf16") };
    b.decl_op("t_attn", p, 7, nullptr, 0);

    /* How many state instances a sequence owns is the layer count, so a group nothing is bound to
     * makes the page size a guess. Declare refuses it where the name of the missing call is still
     * obvious. */
    CHECK(b.finish() < 0);
    CHECK(has(b.errors().back(), "kv group 'kv_full' has no layers bound"));
}

TEST(linear_state_is_sized_per_sequence) {
    NEED_PLUGIN(reg)
    RadBuildCtx ctx = make_ctx();
    ctx.max_spec = 4;
    Builder b(reg, make_meta(), ctx);

    RadKVGroupDecl ld{};
    ld.kind = RAD_KV_LINEAR;
    ld.dtype = RAD_F32;
    ld.n_head_kv = 32;
    ld.state_dim[0] = 128;
    ld.state_dim[1] = 128;
    rad_kvgroup gl = b.decl_kv_group("gdn_state", &ld);

    RadKVGroupDecl cd{};
    cd.kind = RAD_KV_CONV;
    cd.dtype = RAD_F32;
    cd.n_head_kv = 32;
    cd.head_dim = 128;
    cd.conv_width = 4;
    rad_kvgroup gc = b.decl_kv_group("gdn_conv", &cd);

    CHECK_OK(b.bind_layer_kv(0, gl));
    CHECK_OK(b.bind_layer_kv(0, gc));
    b.finish();

    /* 32 heads of 128x128 fp32 is 2 MiB a layer, which is why it cannot be snapshotted per block
     * (spec §7.3). Asking a per-sequence group for a block size is a category error, not a
     * failure: it warns and returns 0, because what the caller wants is bytes_per_state. */
    CHECK_EQ(b.program().kv_groups[gl].bytes_per_state, 32LL * 128 * 128 * 4);
    CHECK_EQ(b.program().kv_groups[gl].bytes_per_block, 0);

    /* A rolling window of conv_width-1 + n_spec entries, so a speculative rejection is a change of
     * read offset rather than a recompute (spec §10).
     *
     * EXACTLY THE STATE TENSOR'S EXTENT. The kernels lay a slot out from the tensor's last extent,
     * conv_width-1 + n_spec, and the KV manager clears, checkpoints and restores the slot this
     * figure describes; one entry more here put each of those over the start of the next
     * sequence's window. rad_builder.cpp's note on `entries` is the argument. */
    CHECK_EQ(b.program().kv_groups[gc].bytes_per_state, 32LL * 128 * (4 - 1 + 4) * 4);
    CHECK_EQ(b.program().kv_groups[gc].bytes_per_state,
             32LL * 128 * conv_window_entries(cd, 4) * 4);
}

/* ================================================================== names, notes, the C surface */

/* PINS `src_index` through decl_name_map, both halves: the index a plugin sets survives, and an
 * unset one arrives as -1 rather than as slice zero. Dropping it is an invisible failure, because
 * the field value-initialises to 0 while its documented default is -1 -- so every map in every
 * model would silently ask for sub-tensor 0 of its source. */
TEST(a_name_map_carries_its_sub_tensor_index) {
    NEED_PLUGIN(reg)
    RadBuildCtx ctx = make_ctx();
    Builder b(reg, make_meta(), ctx);

    {   /* One expert's slice of two stacked tensors, concatenated -- map_slice_concat's shape. */
        std::string declared = "blk.0.ffn_gate_up.3.weight";
        std::string s0 = "blk.0.ffn_gate_exps.weight";
        std::string s1 = "blk.0.ffn_up_exps.weight";
        RadNameMap m{};
        m.declared = declared.c_str();
        m.mode = RAD_MAP_CONCAT;
        m.n_src = 2;
        m.src[0] = s0.c_str();  m.src_index[0] = 3;
        m.src[1] = s1.c_str();  m.src_index[1] = 3;
        m.concat_dim = 0;
        CHECK_OK(b.decl_name_map(&m));
    }
    {   /* And an ordinary whole-tensor map, which must NOT come back as slice 0. */
        std::string declared = "token_embd.weight";
        std::string s0 = "model.embed_tokens.weight";
        RadNameMap m{};
        m.declared = declared.c_str();
        m.mode = RAD_MAP_COPY;
        m.n_src = 1;
        m.src[0] = s0.c_str();
        m.src_index[0] = -1;
        CHECK_OK(b.decl_name_map(&m));
    }

    CHECK_EQ((int)b.program().name_map.size(), 2);
    CHECK_EQ(b.program().name_map[0].src_index[0], 3);
    CHECK_EQ(b.program().name_map[0].src_index[1], 3);
    CHECK_EQ(b.program().name_map[1].src_index[0], -1);
    /* The unused tail is -1, not 0: a consumer that walks all eight must not see slice zero. */
    CHECK_EQ(b.program().name_map[1].src_index[1], -1);
    CHECK_EQ(b.program().name_map[1].src_index[7], -1);
}

/* A CONCAT MAP DECLARED IN PIECES, which is how a 128-shard tensor gets past `src[8]`, and the
 * two repeats that remain errors.
 *
 * The silent half is the one worth the test. A converter lookup built with `std::map::emplace`
 * KEEPS THE FIRST, so a weight declared twice converts from whichever spelling came first and says
 * nothing. Because CONCAT pieces are joined rather than dropped, the repeats that must NOT be
 * joined have to be refused here, at declare, rather than left to a reader of the conversion. */
TEST(a_concat_name_map_may_be_declared_in_pieces) {
    RadBuildCtx ctx = make_ctx();

    auto piece = [](const char* declared, std::initializer_list<const char*> src, int mode,
                    int dim) {
        RadNameMap m{};
        m.declared = declared;
        m.mode = mode;
        m.n_src = (int)src.size();
        int i = 0;
        for (const char* s : src) { m.src[i] = s; m.src_index[i] = -1; ++i; }
        m.concat_dim = dim;
        return m;
    };

    {   /* Twelve shards as two pieces: both are kept, in call order. */
        NEED_PLUGIN(reg)
        Builder b(reg, make_meta(), ctx);
        RadNameMap a = piece("ngram.weight", { "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7" },
                             RAD_MAP_CONCAT, 0);
        RadNameMap c = piece("ngram.weight", { "s8", "s9", "s10", "s11" }, RAD_MAP_CONCAT, 0);
        CHECK_OK(b.decl_name_map(&a));
        CHECK_OK(b.decl_name_map(&c));
        CHECK_EQ((int)b.program().name_map.size(), 2);
        CHECK_EQ(b.program().name_map[0].n_src, 8);
        CHECK_EQ(b.program().name_map[1].n_src, 4);
        /* ORDER IS THE CONCATENATION: shard 0 first, or the table is permuted and every hashed
         * row id addresses the wrong embedding. */
        CHECK(std::string(b.program().name_map[0].src[0]) == "s0");
        CHECK(std::string(b.program().name_map[1].src[0]) == "s8");
    }
    {   /* A repeated COPY map is the accident the merge rule has to keep refusing. */
        NEED_PLUGIN(reg)
        Builder b(reg, make_meta(), ctx);
        RadNameMap a = piece("w", { "x" }, RAD_MAP_COPY, 0);
        RadNameMap c = piece("w", { "y" }, RAD_MAP_COPY, 0);
        CHECK_OK(b.decl_name_map(&a));
        CHECK(b.decl_name_map(&c) != RAD_OK);
    }
    {   /* ...and so is a piece that joins along a different axis. */
        NEED_PLUGIN(reg)
        Builder b(reg, make_meta(), ctx);
        RadNameMap a = piece("w", { "x" }, RAD_MAP_CONCAT, 0);
        RadNameMap c = piece("w", { "y" }, RAD_MAP_CONCAT, 1);
        CHECK_OK(b.decl_name_map(&a));
        CHECK(b.decl_name_map(&c) != RAD_OK);
    }
}

TEST(name_map_and_notes_survive_the_plugin) {
    NEED_PLUGIN(reg)
    RadBuildCtx ctx = make_ctx();
    ctx.scope = "draft";
    Builder b(reg, make_meta(), ctx);

    {
        /* The strings the plugin passes are its own; the Program outlives the call, so they are
         * interned rather than borrowed. */
        std::string declared = "attn_qkv.weight";
        std::string s0 = "model.layers.0.q_proj.weight";
        std::string s1 = "model.layers.0.k_proj.weight";
        RadNameMap m{};
        m.declared = declared.c_str();
        m.mode = RAD_MAP_CONCAT;
        m.n_src = 2;
        m.src[0] = s0.c_str();
        m.src[1] = s1.c_str();
        m.concat_dim = 0;
        CHECK_OK(b.decl_name_map(&m));
    }

    CHECK_EQ((int)b.program().name_map.size(), 1);
    /* The scope prefixes declared names so a drafter and the target can share one builder. */
    CHECK_EQ(std::string(b.program().name_map[0].declared), std::string("draft.attn_qkv.weight"));
    CHECK_EQ(std::string(b.program().name_map[0].src[1]),
             std::string("model.layers.0.k_proj.weight"));

    rad_weight w = decl_w(b, "attn_qkv.weight", 16, 16);
    CHECK_EQ(b.program().weight(w)->name, std::string("draft.attn_qkv.weight"));

    rad_note(&b, "%d layers, gqa %d", 48, 8);
    CHECK_EQ(b.program().notes[0], std::string("48 layers, gqa 8"));
}

TEST(the_c_surface_drives_the_builder) {
    NEED_PLUGIN(reg)
    RadModelMeta meta = make_meta();
    Builder b(reg, meta, make_ctx());
    RadBuilder* h = &b;              /* the handle IS the builder */

    RadWeightDecl wd{};
    wd.dtype = RAD_BF16; wd.rank = 2; wd.shape[0] = 4096; wd.shape[1] = 4096;
    wd.access = RAD_ACCESS_PER_TOKEN;
    rad_weight w = rad_decl_weight(h, "blk.0.ffn_gate.weight", &wd);
    CHECK(w != RAD_NULL_HANDLE);

    rad_op op = RAD_OP(h, "t_gemm",
                       RAD_PARAMS(RAD_RANGE("M", 1, 8192), RAD_INT("N", 4096), RAD_INT("K", 4096),
                                  RAD_STR("dtype", "bf16")),
                       RAD_WEIGHTS(w));
    CHECK(op != RAD_NULL_HANDLE);
    CHECK_EQ(rad_op_resolved(h, op), 1);
    CHECK_EQ(rad_op_resolved(h, 99), 0);

    /* A weight declared twice cannot be diagnosed later: the container has one entry for it. */
    CHECK_EQ(rad_decl_weight(h, "blk.0.ffn_gate.weight", &wd), RAD_NULL_HANDLE);
    CHECK_EQ(b.status(), RAD_E_DUPLICATE);

    CHECK_EQ(rad_meta_geti(&meta, "nothing", 7), 7);
}

/* The program is moved out of the builder, and a short name lives inside the std::string object
 * itself -- so the decl's own pointer cannot name the program's copy of the string without
 * naming the builder's after the move. The name is drafter_name; the decl carries none. */
TEST(the_drafter_name_survives_the_program_leaving_the_builder) {
    Registry reg;
    std::unique_ptr<Builder> b(new Builder(reg, make_meta(), make_ctx()));

    std::string plugin_owned = "mtp";
    RadDrafterDecl d{};
    d.kind  = RAD_DRAFT_SERIAL;
    d.name  = plugin_owned.c_str();
    d.depth = 3;
    CHECK_OK(b->declare_drafter(&d));
    plugin_owned.assign("xxx");

    Program p = std::move(b->program());
    b.reset();
    CHECK_EQ(p.drafter_name, std::string("mtp"));
    CHECK(p.drafter.name == nullptr);
    CHECK_EQ(p.drafter.kind, (int)RAD_DRAFT_SERIAL);
    CHECK_EQ(p.drafter.depth, 3);
}

TEST(multistage_block_drafter_geometry_validation) {
    Registry reg;
    std::unique_ptr<Builder> b(new Builder(reg, make_meta(), make_ctx()));
    RadBufDecl bd{};
    bd.dtype = RAD_I32;
    bd.shape[0] = 8; bd.shape[1] = 12;
    bd.rank = 2;
    bd.kind = RAD_BUF_TRANSIENT;
    bd.domain = RAD_DOMAIN_DEVICE;
    rad_buf prop = b->decl_buffer("prop", &bd);

    // Multi-stage: block=7, depth=12 (2 * (7 - 1) == 12)
    RadDrafterDecl d{};
    d.kind = RAD_DRAFT_BLOCK;
    d.name = "fulldrafter";
    d.depth = 12;
    d.block = 7;
    d.proposal = prop;
    d.mask_token = 151643;
    CHECK_OK(b->declare_drafter(&d));

    // Multi-stage invalid depth: block=7, depth=13 (13 % 6 != 0)
    Builder b_inval(reg, make_meta(), make_ctx());
    rad_buf p_inval = b_inval.decl_buffer("prop", &bd);
    d.proposal = p_inval;
    d.depth = 13;
    d.block = 7;
    CHECK_EQ(b_inval.declare_drafter(&d), RAD_E_INVAL);

    // Multi-stage invalid block < 2
    Builder b_b1(reg, make_meta(), make_ctx());
    d.proposal = b_b1.decl_buffer("prop", &bd);
    d.depth = 12;
    d.block = 1;
    CHECK_EQ(b_b1.declare_drafter(&d), RAD_E_INVAL);

    // Single-stage block == depth + 1 (block=8, depth=7)
    Builder b_single(reg, make_meta(), make_ctx());
    d.proposal = b_single.decl_buffer("prop", &bd);
    d.depth = 7;
    d.block = 8;
    CHECK_OK(b_single.declare_drafter(&d));

    // Single-stage block == depth (block=7, depth=7)
    Builder b_anchor(reg, make_meta(), make_ctx());
    d.proposal = b_anchor.decl_buffer("prop", &bd);
    d.depth = 7;
    d.block = 7;
    CHECK_OK(b_anchor.declare_drafter(&d));

    // Multi-stage 3-stage: block=7, depth=18 (3 * (7 - 1) == 18)
    Builder b_s3(reg, make_meta(), make_ctx());
    d.proposal = b_s3.decl_buffer("prop", &bd);
    d.depth = 18;
    d.block = 7;
    CHECK_OK(b_s3.declare_drafter(&d));

    // Invalid geometry with depth < block: block=10, depth=3
    Builder b_b10(reg, make_meta(), make_ctx());
    d.proposal = b_b10.decl_buffer("prop", &bd);
    d.depth = 3;
    d.block = 10;
    CHECK_EQ(b_b10.declare_drafter(&d), RAD_E_INVAL);
}


/* ================================================================== the enum name tables
 *
 * These are printed into --debug-graph and into the Model view, which is where an operator reads
 * what a plugin declared. An "?" where a class name belongs is not a cosmetic fault: it is the
 * reader being told nothing about the axis they are looking at, and the case that produces it is
 * a value this core does not know -- which is exactly the case worth being able to see.
 */
TEST(the_declared_enums_all_have_names_and_an_unknown_value_says_so) {
    CHECK_EQ(std::string(bld_domain_name(RAD_DOMAIN_DEVICE)), std::string("device"));
    CHECK_EQ(std::string(bld_domain_name(RAD_DOMAIN_HOST)), std::string("host"));

    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_PER_TOKEN)), std::string("per-token"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_PER_REQUEST)), std::string("per-request"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_CONDITIONAL)), std::string("conditional"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_RARE)), std::string("rare"));
    CHECK_EQ(std::string(bld_access_name(RAD_ACCESS_VOCAB)), std::string("vocab"));

    CHECK_EQ(std::string(bld_buf_kind_name(RAD_BUF_TRANSIENT)), std::string("transient"));
    CHECK_EQ(std::string(bld_buf_kind_name(RAD_BUF_PERSIST)), std::string("persist"));
    CHECK_EQ(std::string(bld_buf_kind_name(RAD_BUF_DERIVED)), std::string("derived"));

    /* EVERY NAME IS DISTINCT, or the dump is ambiguous in a way a reader cannot detect: two
       classes printing the same word reads as one class. */
    const char* access[] = { bld_access_name(RAD_ACCESS_PER_TOKEN),
                             bld_access_name(RAD_ACCESS_PER_REQUEST),
                             bld_access_name(RAD_ACCESS_CONDITIONAL),
                             bld_access_name(RAD_ACCESS_RARE),
                             bld_access_name(RAD_ACCESS_VOCAB) };
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j) CHECK(std::strcmp(access[i], access[j]) != 0);

    /* An unknown value is "?" rather than a plausible name. */
    CHECK_EQ(std::string(bld_access_name(9999)), std::string("?"));
    CHECK_EQ(std::string(bld_buf_kind_name(-1)), std::string("?"));
    /* Every KV kind the ABI defines is named. */
    for (int k : { RAD_KV_FULL, RAD_KV_WINDOW, RAD_KV_LINEAR, RAD_KV_CONV }) {
        const char* n = bld_kv_kind_name(k);
        CHECK(n && *n);
        CHECK(std::strcmp(n, "?") != 0);
    }
}

/* ================================================================== sizing a declared buffer
 *
 * ZERO MEANS "THE CORE CANNOT SIZE IT", which is legal for a weight -- a kernel's layout hook
 * sizes those -- and is a declare-time failure for an activation. So zero has to be reachable
 * only for the reason it means, and it must not be what an ordinary buffer gets.
 */
TEST(buffer_bytes_sizes_a_declaration_or_says_it_cannot) {
    RadBufDecl d{};
    d.dtype = RAD_BF16;
    d.rank  = 2;
    d.shape[0] = 16;
    d.shape[1] = 32;
    CHECK_EQ(buffer_bytes(d), (int64_t)(16 * 32 * 2));

    d.dtype = RAD_F32;
    CHECK_EQ(buffer_bytes(d), (int64_t)(16 * 32 * 4));
    d.dtype = RAD_I8;
    CHECK_EQ(buffer_bytes(d), (int64_t)(16 * 32));

    /* A rank-0 declaration has no elements: a scalar buffer is not a buffer of one. */
    RadBufDecl z{};
    z.dtype = RAD_BF16;
    z.rank  = 0;
    CHECK_EQ(buffer_bytes(z), 0ll);

    /* A ZERO EXTENT IN ANY DIMENSION IS ZERO BYTES rather than the product of the others, which
       is what keeps an unused optional plane from reserving the arena its shape implies. */
    RadBufDecl e{};
    e.dtype = RAD_BF16;
    e.rank  = 2;
    e.shape[0] = 0;
    e.shape[1] = 4096;
    CHECK_EQ(buffer_bytes(e), 0ll);

    /* A dtype the core cannot measure is 0 -- the signal plan_buffers turns into a named
       refusal for an activation. */
    RadBufDecl p{};
    p.dtype = 0xFFFF;
    p.rank  = 1;
    p.shape[0] = 1024;
    CHECK_EQ(buffer_bytes(p), 0ll);

    /* The sizes here are the ones the arena is carved from, so the big shapes matter too. */
    RadBufDecl big{};
    big.dtype = RAD_BF16;
    big.rank  = 2;
    big.shape[0] = 200000;
    big.shape[1] = 4096;
    CHECK_EQ(buffer_bytes(big), 200000ll * 4096ll * 2ll);
}

/* ================================================================== the Model view's data
 *
 * dump_graph_json is what the dashboard's Model view parses, and its whole reason for existing is
 * the grouping: a 64-layer model declares one op sixty-four times with one bucket table, and
 * sixty-four identical rows are not more information than one row saying x64. The declared ORDER
 * is kept beside it as indices, because the grouping is exactly what throws the sequence away and
 * a diagram needs it back.
 */
TEST(the_graph_json_groups_identical_declarations_and_keeps_the_order) {
    Program p;
    p.meta = make_meta();
    p.ctx.rank = 0;
    p.ctx.world_size = 1;
    p.weights.emplace_back();
    p.buffers.emplace_back();
    p.ops.emplace_back();                 /* the sentinel */

    /* Two ops alternating, sixteen layers deep: two groups of sixteen, thirty-two in order. */
    for (int l = 0; l < 16; ++l) {
        for (const char* name : { "attn", "ffn" }) {
            OpInfo o;
            o.op = name;
            o.index = (int32_t)p.ops.size();
            o.base.set_i("N", 4096);
            p.ops.push_back(o);
        }
    }

    const std::string s = dump_graph_json(p);
    const json j = json::parse(s, nullptr, false);
    CHECK(!j.is_discarded());

    CHECK_EQ(j["counts"].value("ops", (int64_t)0), (int64_t)32);
    CHECK_EQ(j["ops"].size(), 2u);                 /* not 32 */
    CHECK_EQ(j["ops"][0].value("count", (int64_t)0), (int64_t)16);
    CHECK_EQ(j["ops"][1].value("count", (int64_t)0), (int64_t)16);
    CHECK_EQ(j["ops"][0].value("op", std::string()), std::string("attn"));
    CHECK_EQ(j["ops"][1].value("op", std::string()), std::string("ffn"));

    /* THE ORDER IS THE SEQUENCE THE GROUPING THREW AWAY, one index per declared op. */
    CHECK_EQ(j["order"].size(), 32u);
    CHECK_EQ(j["order"][0].get<int>(), 0);
    CHECK_EQ(j["order"][1].get<int>(), 1);
    CHECK_EQ(j["order"][2].get<int>(), 0);

    /* And where each group's declarations actually sit, because one op index would tell the
       reader about the first copy only. */
    CHECK_EQ(j["ops"][0].value("first_at", -1), 0);
    CHECK_EQ(j["ops"][0].value("last_at", -1), 30);
    CHECK_EQ(j["ops"][1].value("first_at", -1), 1);
    CHECK_EQ(j["ops"][1].value("last_at", -1), 31);

    /* The model block is what the view's header reads, and it comes from the container rather
       than from anything this walk computes. */
    CHECK_EQ(j["model"].value("arch", std::string()), std::string("test_arch"));
    CHECK_EQ(j["model"].value("n_layers", (int64_t)0), (int64_t)2);
    CHECK_EQ(j["model"].value("world", (int64_t)0), (int64_t)1);
}

/* TWO DECLARATIONS OF ONE OP THAT RESOLVED DIFFERENTLY ARE TWO GROUPS, because the band table is
 * part of the key -- collapsing them would hide the one thing a reader opened the view to see. */
TEST(the_graph_json_splits_a_group_whose_band_table_differs) {
    Program p;
    p.meta = make_meta();
    p.weights.emplace_back();
    p.buffers.emplace_back();
    p.ops.emplace_back();

    for (int i = 0; i < 2; ++i) {
        OpInfo o;
        o.op = "gemm";
        o.index = (int32_t)p.ops.size();
        Band b;
        b.hi = i ? 64 : 16;                        /* a different table on the second one */
        b.miss[RAD_DOMAIN_DEVICE] = "no kernel";
        b.miss[RAD_DOMAIN_HOST]   = "no kernel";
        o.bands.push_back(b);
        p.ops.push_back(o);
    }

    const json j = json::parse(dump_graph_json(p), nullptr, false);
    CHECK(!j.is_discarded());
    CHECK_EQ(j["ops"].size(), 2u);
    CHECK_EQ(j["ops"][0].value("count", (int64_t)0), (int64_t)1);
    CHECK_EQ(j["ops"][1].value("count", (int64_t)0), (int64_t)1);
    /* A band with no kernel on either domain reports the reason rather than being absent, which
       is the difference between "not resolved" and "not declared". */
    CHECK(j["ops"][0]["bands"][0].contains("device_miss"));
    CHECK(j["ops"][0]["bands"][0].contains("host_miss"));
}

TEST(the_graph_json_of_an_empty_program_is_still_valid_json) {
    Program p;
    p.weights.emplace_back();
    p.buffers.emplace_back();
    p.ops.emplace_back();
    const json j = json::parse(dump_graph_json(p), nullptr, false);
    CHECK(!j.is_discarded());
    CHECK(j["ops"].is_array());
    CHECK_EQ(j["ops"].size(), 0u);
    CHECK(j["order"].is_array());
    CHECK_EQ(j["order"].size(), 0u);
    CHECK_EQ(j["counts"].value("ops", (int64_t)-1), (int64_t)0);
}

/* ================================================================== the reference-kernel gate
 *
 * libref is BOTH the fallback that makes an unsupported model run and the oracle every
 * correctness number is measured against. The first is a hazard in production and the second is
 * not, so which one is happening has to be a decision rather than a discovery -- a model that
 * silently fell back to the reference for its GEMMs runs orders of magnitude below the machine
 * with nothing in the output to say so.
 *
 * ONE FUNCTION STANDS BETWEEN THAT AND SERVING. It is called from the engine's bringup and from
 * rad-convert, and a change that made it count zero ops would fail nothing else: every model
 * would start, and the only symptom would be the speed.
 */
namespace {

/* What the gate SAID, not only what it returned. The message is half the point of the function:
 * it names the ops and the caller so the operator reads what the consequence is rather than that
 * something generic went wrong. Captured rather than silenced -- there is no log level below
 * Error, and a refusal that prints its refusal in a passing test is only noise. */
std::string say(const std::function<void()>& fn) {
    std::fflush(stderr);
    const int saved = dup(2);
    std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                       "/radiance_build_test_stderr_XXXXXX";
    std::vector<char> tmpl(path.begin(), path.end());
    tmpl.push_back('\0');
    const int fd = mkstemp(tmpl.data());
    if (fd < 0 || saved < 0) { fn(); return {}; }
    dup2(fd, 2);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    lseek(fd, 0, SEEK_SET);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof buf)) > 0) out.append(buf, (size_t)n);
    close(fd);
    unlink(tmpl.data());
    return out;
}

/* Rows live in the test rather than in a Program because a Resolved points at one -- the plugin
 * tables they normally point into are static for the process. */
/* A library standing in for one that declared itself a reference implementation. Which library is
 * one is the declaration on its rows (KernelRow::reference), never its name. */
const char* const kRef = "aref";

KernelRow& row_from(const char* plugin) {
    static std::vector<std::unique_ptr<KernelRow>> rows;
    rows.push_back(std::make_unique<KernelRow>());
    rows.back()->plugin = plugin;
    rows.back()->reference = std::strcmp(plugin, kRef) == 0;
    return *rows.back();
}

/* An op with one band resolved in the device domain by `plugin`, or unresolved when null. */
OpInfo op_on(const char* op, const char* plugin) {
    OpInfo oi;
    oi.op = op;
    oi.bands.emplace_back();
    if (plugin) oi.bands[0].dom[RAD_DOMAIN_DEVICE].row = &row_from(plugin);
    return oi;
}

Program with_ops(std::vector<OpInfo> ops) {
    Program p;
    p.ops.emplace_back();            /* the sentinel: handles are 1-based */
    for (OpInfo& o : ops) p.ops.push_back(std::move(o));
    return p;
}

}  /* namespace */

TEST(an_op_a_real_library_serves_is_not_a_reference_fallback) {
    Program p = with_ops({ op_on("rmsnorm", "libr4d"), op_on("gemm", "libr4d") });
    std::vector<std::string> ops;
    CHECK_EQ(rad_reference_ops(p, &ops), 0);
    CHECK(ops.empty());
    CHECK_OK(rad_gate_reference_kernels(p, false, "serve"));
}

/* The sentinel at index 0 carries no bands and is not an op. Counted, it would make every
 * program in the tree look like a fallback and the gate would refuse everything. */
TEST(the_op_sentinel_is_not_counted_as_a_reference_fallback) {
    Program p;
    p.ops.emplace_back();
    std::vector<std::string> ops;
    CHECK_EQ(rad_reference_ops(p, &ops), 0);
    CHECK(ops.empty());
    CHECK_OK(rad_gate_reference_kernels(p, false, "serve"));
}

TEST(an_op_only_the_reference_serves_refuses_the_serve) {
    Program p = with_ops({ op_on("rmsnorm", "libr4d"),
                           op_on("gemm", kRef) });
    std::vector<std::string> ops;
    CHECK_EQ(rad_reference_ops(p, &ops), 1);
    CHECK_EQ(ops.size(), (size_t)1);
    if (!ops.empty()) CHECK_EQ(ops[0], std::string("gemm"));

    int st = RAD_OK;
    std::string msg = say([&] { st = rad_gate_reference_kernels(p, false, "serve"); });
    CHECK_EQ(st, RAD_E_NOKERNEL);
    CHECK(has(msg, "gemm"));
    CHECK(has(msg, "serve"));
    CHECK(has(msg, "--debug-accept-reference-kernels"));

    /* The flag is spelled with `debug` in it because this is what it is for: the cost is accepted
     * deliberately, the run continues, and it still says so. */
    msg = say([&] { st = rad_gate_reference_kernels(p, true, "serve"); });
    CHECK_OK(st);
    CHECK(has(msg, "gemm"));
    CHECK(has(msg, "not a production configuration"));
}

/* A LAYER MODEL DECLARES THE SAME OP SIXTY-FOUR TIMES. The count is every declaration, because
 * that is how much of the program is on the slow path; the names are deduplicated, because
 * "rmsnorm has no kernel" is the useful sentence and sixty-four copies of it is not. */
TEST(the_reference_count_is_per_declaration_and_the_names_are_not) {
    std::vector<OpInfo> ops;
    for (int i = 0; i < 8; ++i) {
        ops.push_back(op_on("rmsnorm", kRef));
        ops.push_back(op_on("gemm", kRef));
    }
    Program p = with_ops(std::move(ops));
    std::vector<std::string> names;
    CHECK_EQ(rad_reference_ops(p, &names), 16);
    CHECK_EQ(names.size(), (size_t)2);
    /* Reported in declaration order, so the message reads like the program. */
    if (names.size() == 2) {
        CHECK_EQ(names[0], std::string("rmsnorm"));
        CHECK_EQ(names[1], std::string("gemm"));
    }
}

/* AN OP IS ON THE REFERENCE ONLY IF NOTHING ELSE SERVES IT ANYWHERE. A band the reference covers
 * beside one a real library covers is a model whose long shapes are fast, and the gate must not
 * refuse it -- selection already prefers the real row wherever both apply. */
TEST(a_real_kernel_in_any_band_or_domain_clears_the_op) {
    OpInfo oi = op_on("gemm", kRef);
    oi.bands.emplace_back();
    oi.bands[1].dom[RAD_DOMAIN_DEVICE].row = &row_from("libr4d");
    Program p = with_ops({ oi });
    std::vector<std::string> names;
    CHECK_EQ(rad_reference_ops(p, &names), 0);

    /* And the same across domains rather than bands: a host row from a real library is still not
     * the oracle. */
    OpInfo oj = op_on("gemm", kRef);
    oj.bands[0].dom[RAD_DOMAIN_HOST].row = &row_from("libr4d");
    Program q = with_ops({ oj });
    names.clear();
    CHECK_EQ(rad_reference_ops(q, &names), 0);
}

/* An op nothing resolved at all is not serving on the reference, but it is not servable either.
 * It is counted here rather than passed over, because the alternative is a program that clears
 * this gate and fails at the first step that issues the op. */
TEST(an_op_nothing_resolved_is_counted_rather_than_waved_through) {
    Program p = with_ops({ op_on("gemm", nullptr) });
    std::vector<std::string> names;
    CHECK_EQ(rad_reference_ops(p, &names), 1);
    int st = RAD_OK;
    (void)say([&] { st = rad_gate_reference_kernels(p, false, "serve"); });
    CHECK_EQ(st, RAD_E_NOKERNEL);
}

/* The caller's own name goes into the message, so it says what the consequence is rather than
 * that something generic went wrong. */
TEST(the_gate_names_its_caller) {
    Program p = with_ops({ op_on("gemm", kRef) });
    int st = RAD_OK;
    const std::string msg =
        say([&] { st = rad_gate_reference_kernels(p, false, "the container's layout"); });
    CHECK_EQ(st, RAD_E_NOKERNEL);
    CHECK(has(msg, "the container's layout"));
}

RAD_TEST_MAIN()
