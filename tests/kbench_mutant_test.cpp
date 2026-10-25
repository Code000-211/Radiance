/* kbench_mutant_test.cpp -- rad-kbench end to end, against kernel libraries that are wrong on
 * purpose (core/plugin/testplugin/kbmutant.cc).
 *
 * The replay is the only correctness check most kernels get, and a check that cannot fail is
 * worse than none. Each defective library here must fail it with the right message: an output past
 * its tolerance, a NaN, an unwritten half, a write past an output's end, a refused geometry, an id
 * off by one, a write into an input. The correct library must pass on both domains, the reference
 * must replay its own recording exactly (and again from its cache), a row the engine would pass
 * over for its weight's encoding must not be the one that runs, and the coverage ratchet must
 * notice a row that is no longer reached. One case runs a real library: libavx's sampler stages
 * against libref on drawn requests.
 *
 * Everything runs on the host backend: device-domain rows compute on the backend's memory inside
 * the same guarded allocations a card would get. The fixture is built here from libref's own
 * operand descriptions and recorded by rad-kbench --rerecord. Each check runs the tool as its own
 * process, a few tens of milliseconds each, which is most of this binary's second.
 */
#include "rad_test.h"
#include "kbench_core.h"
#include "kfixture.h"
#include "opshapes.h"
#include "plugin/registry.h"
#include "format/radfile.h"
#include "rad_plugin.h"
#include "rad_device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace rad;

/* The mutants' device rows drain their stream through the engine's device API, which this binary
 * exports for the in-process registry test (tests/CMakeLists.txt). Referenced so the linker keeps
 * it. */
[[maybe_unused]] static int (*const kDeviceApi)(RadStream) = &rad_stream_sync;

namespace {

const char* const kMutants[] = { "ok", "scale", "nan", "half", "overrun", "refuse", "shape",
                                 "argmax", "inwrite", "enc", "pair" };

struct Out {
    int code = -1;
    std::string text;
};

/* The shared setup: a home holding libref and every mutant, a fixture described by libref's own
 * hooks, and that fixture recorded with libref's answers. */
struct Env {
    std::string dir, home, base, fx;
    bool ok = false;
};

Out run(const Env& e, const std::string& args) {
    const std::string cmd = "ROCR_VISIBLE_DEVICES=99 OMP_NUM_THREADS=2 OMP_WAIT_POLICY=passive " +
                            std::string(RAD_KBENCH_BIN) + " --home '" + e.home + "' " + args +
                            " 2>&1";
    Out o;
    FILE* p = ::popen(cmd.c_str(), "r");
    if (!p) return o;
    char buf[4096];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, p)) > 0) o.text.append(buf, got);
    const int st = ::pclose(p);
    o.code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    return o;
}

/* Prints the tool's output when a check on it fails, so the failure explains itself. */
#define CHECK_RUN(o, cond)                                                                \
    do { const bool _ok = (cond); CHECK(_ok);                                              \
         if (!_ok) std::fprintf(stderr, "    exit %d\n%s\n", (o).code, (o).text.c_str()); } \
    while (0)

/* One case of `op` at `g`, its operands described by `lib`'s hook for it: libref's, or the plugin
 * that implements an op libref does not have (plugin/qsatable's `ref_qsa_table`), whose description
 * is what a fixture of that op must carry. */
bool describe_lib(Registry& reg, const std::string& lib, const std::string& op, const Geometry& g,
                  uint64_t seed, int64_t M, KfCase* c) {
    const RadOpSchema* schema = reg.schema(op);
    const KernelRow* row = reg.select_in(lib, op, g, RAD_DOMAIN_HOST);
    if (!schema || !row) return false;
    std::vector<OpdShape> sh;
    if (rad_operand_shapes_hook(row->info, g, schema->n_operands, &sh) < 0) return false;
    c->op = op;
    c->seed = seed;
    c->m = M;
    for (int i = 0; i < g.n_params(); ++i) {
        const RadParam& p = g.params()[i];
        KfParam kp;
        kp.key = p.key;
        kp.kind = p.kind;
        kp.i = p.ival;
        kp.d = p.dval;
        kp.s = p.sval ? p.sval : "";
        c->params.push_back(kp);
    }
    for (size_t k = 0; k < sh.size(); ++k) {
        KfOperand d;
        d.role = schema->operands[k].role;
        d.dtype = sh[k].dtype;
        d.shape = sh[k].shape;
        d.absent = sh[k].absent;
        d.fill = sh[k].fill;
        d.fill_chunk = sh[k].fill_chunk;
        d.is_index = sh[k].is_index;
        d.idx_unique = sh[k].idx_unique;
        d.idx_cu = sh[k].idx_cu;
        d.idx_prefix = sh[k].idx_prefix;
        d.idx_history = sh[k].idx_history;
        d.idx_ordered = sh[k].idx_ordered;
        d.idx_max = sh[k].idx_max;
        d.idx_const = sh[k].idx_const;
        if ((d.role == RAD_OPD_WEIGHT || d.role == RAD_OPD_WTAB) && !d.absent) {
            /* A plain weight, one plane, as a container stores a norm's gain; a table records one
             * expert's (kf_draw_table). */
            d.has_enc = true;
            d.enc = rad_enc_plain(d.dtype);
            d.sel = { 0 };
            d.sel_rows = { 1 };
            d.sel_cols = { d.role == RAD_OPD_WTAB ? rad_numel(d.shape) / d.shape[0]
                                                  : rad_numel(d.shape) };
        }
        c->opd.push_back(d);
    }
    return true;
}

bool describe(Registry& reg, const std::string& op, const Geometry& g, uint64_t seed, int64_t M,
              KfCase* c) {
    return describe_lib(reg, "libref", op, g, seed, M, c);
}

Env build_env() {
    Env e;
    e.dir = radtest::tmpdir();
    e.home = e.dir + "/home";
    if (::mkdir(e.home.c_str(), 0755) != 0 || ::mkdir((e.home + "/kernels").c_str(), 0755) != 0 ||
        ::mkdir((e.home + "/architectures").c_str(), 0755) != 0)
        return e;
    const std::string ref = std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so";
    if (::symlink(ref.c_str(), (e.home + "/kernels/libref.so").c_str()) != 0) return e;
    for (const char* m : kMutants) {
        const std::string so = std::string(RAD_PLUGIN_DIR) + "/testkernels/kbmut/kbm_" + m + ".so";
        if (::symlink(so.c_str(), (e.home + "/kernels/kbm_" + m + ".so").c_str()) != 0) return e;
    }
    /* plugin/qsatable: the ops libref does not implement, each with the host reference beside it. */
    const std::string qsa = std::string(RAD_PLUGIN_DIR) + "/kernels/qsatable.so";
    if (::symlink(qsa.c_str(), (e.home + "/kernels/qsatable.so").c_str()) != 0 && errno != EEXIST)
        return e;
    const std::string arch = std::string(RAD_PLUGIN_DIR) + "/testarch/kbench/radtest_kbarch.so";
    if (::symlink(arch.c_str(), (e.home + "/architectures/radtest_kbarch.so").c_str()) != 0)
        return e;

    Registry reg;
    if (reg.load_plugin(ref, 0) < 0 || reg.check_complete() < 0) return e;
    Kfixture fx;
    fx.source = "kbench_mutant_test";
    fx.oracle = "libref";
    KfCase c;
    Geometry g;
    g.set_i("M", 8); g.set_i("n", 512); g.set_s("dtype", "bf16");
    if (!describe(reg, "add", g, 101, 8, &c)) return e;
    fx.cases.push_back(c);
    c = KfCase{};
    g = Geometry{};
    g.set_i("M", 4); g.set_i("n", 32); g.set_s("dtype", "bf16");
    if (!describe(reg, "gather_rows", g, 102, 4, &c)) return e;
    fx.cases.push_back(c);
    c = KfCase{};
    g = Geometry{};
    g.set_i("M", 3); g.set_i("n_vocab", 500);
    if (!describe(reg, "sample_argmax", g, 103, 3, &c)) return e;
    fx.cases.push_back(c);
    c = KfCase{};
    g = Geometry{};
    g.set_i("M", 4); g.set_i("n", 64); g.set_f("eps", 1e-6); g.set_s("dtype", "bf16");
    if (!describe(reg, "rmsnorm", g, 104, 4, &c)) return e;
    fx.cases.push_back(c);
    /* One case past a 1 MiB budget: 9 copies of a 512 KiB operand. */
    c = KfCase{};
    g = Geometry{};
    g.set_i("M", 8); g.set_i("n", 32768); g.set_s("dtype", "bf16");
    if (!describe(reg, "add", g, 105, 8, &c)) return e;
    fx.cases.push_back(c);
    e.base = e.dir + "/base.rkb";
    if (fx.save(e.base) != RAD_OK) return e;

    e.fx = e.dir + "/fx.rkb";
    const Out o = run(e, "--kernels libref --max-bytes 64 --rerecord '" + e.fx + "' --fixture '" +
                             e.base + "'");
    if (o.code != 0 || !has(o.text, "recorded 5 case(s)")) {
        std::fprintf(stderr, "    --rerecord failed (exit %d)\n%s\n", o.code, o.text.c_str());
        return e;
    }
    e.ok = true;
    return e;
}

const Env& env() {
    static const Env e = build_env();
    return e;
}

/* A fixture made of some of the recorded cases, for a test that needs a particular mix. */
std::string fixture_of(const std::string& leaf, const std::vector<KfCase>& cases,
                       const char* oracle = "libref") {
    Kfixture fx;
    fx.source = "kbench_mutant_test";
    fx.oracle = oracle;
    fx.cases = cases;
    const std::string path = env().dir + "/" + leaf;
    return fx.save(path) == RAD_OK ? path : std::string();
}

/* A one-layer container for radtest_kbarch's graph (a norm, a projection and an add), written once:
 * its gain and projection are constants; a recording draws its own. "" if it could not be written. */
std::string tiny_model() {
    static const std::string path = [] {
        const std::string model = env().dir + "/tiny.rad";
        RadWriter w;
        if (w.begin(model.c_str()) != RAD_OK) return std::string();
        w.set_arch("radtest.kbench");
        w.set_model_name("kbench-record");
        w.meta_i("n_layers", 1);
        w.meta_i("n_embd", 64);
        w.meta_i("n_ff", 96);
        RadWriter::Weight g;
        g.name = "norm.weight";
        g.enc = rad_enc_plain(RAD_BF16);
        g.rank = 1;
        g.shape[0] = 64;
        std::vector<uint16_t> gb(64, 0x3F80);
        const void* gp[] = { gb.data() };
        if (w.add_weight(g, gp) < 0) return std::string();
        RadWriter::Weight p = g;
        p.name = "proj.weight";
        p.rank = 2;
        p.shape[0] = 96;
        p.shape[1] = 64;
        std::vector<uint16_t> pb(96 * 64, 0x3C00);
        const void* pp[] = { pb.data() };
        if (w.add_weight(p, pp) < 0 || w.finish() != RAD_OK) return std::string();
        return model;
    }();
    return path;
}

Kfixture recorded() {
    Kfixture fx;
    fx.load(env().fx);
    return fx;
}

std::string replay(const std::string& lib, const std::string& extra = "") {
    return "--kernels " + lib + ":libref --no-ref --strict --max-bytes 64 --fixture '" + env().fx +
           "' " + extra;
}

/* A home holding only the libraries a test names. rad-kbench loads every plugin in its home, so "no
 * loaded library declares a reference for this op" is a property of the home: --kernels picks what
 * is under test, not what can answer in place of a recording's reference. */
Env home_with(const std::string& leaf, const std::vector<std::string>& kernels) {
    Env e = env();
    e.home = env().dir + "/" + leaf;
    for (const std::string& d : { "", "/kernels", "/architectures" })
        if (::mkdir((e.home + d).c_str(), 0755) != 0 && errno != EEXIST) { e.ok = false; return e; }
    for (const std::string& k : kernels) {
        /* The mutants live in testkernels/kbmut, as they do in the shared home. */
        const std::string so = std::string(RAD_PLUGIN_DIR) +
                               (k.rfind("kbm_", 0) == 0 ? "/testkernels/kbmut/" : "/kernels/") +
                               k + ".so";
        if (::symlink(so.c_str(), (e.home + "/kernels/" + k + ".so").c_str()) != 0 &&
            errno != EEXIST) { e.ok = false; return e; }
    }
    const std::string arch = std::string(RAD_PLUGIN_DIR) + "/testarch/kbench/radtest_kbarch.so";
    if (::symlink(arch.c_str(), (e.home + "/architectures/radtest_kbarch.so").c_str()) != 0 &&
        errno != EEXIST) e.ok = false;
    return e;
}

}  /* namespace */

/* ================================================================== the reference */

/* libref replays its own recording at tolerance zero, and a second run takes every answer from the
 * cache while still running the comparison: same verdicts, nothing launched. */
TEST(the_reference_replays_its_recording_exactly_and_again_from_its_cache) {
    REQUIRE(env().ok);
    const std::string args = "--kernels libref --strict --max-bytes 64 -v --fixture '" +
                             env().fx + "'";
    const Out first = run(env(), args);
    CHECK_RUN(first, first.code == 0);
    CHECK_RUN(first, has(first.text, "5 checked, 5 passed, 0 failed"));
    CHECK_RUN(first, has(first.text, "(exact)"));
    CHECK_RUN(first, !has(first.text, "(cached)"));
    const Out second = run(env(), args);
    CHECK_RUN(second, second.code == 0);
    CHECK_RUN(second, has(second.text, "5 checked, 5 passed, 0 failed"));
    CHECK_RUN(second, has(second.text, "5 reference result(s) came from the cache"));
    const Out fresh = run(env(), args + " --no-cache");
    CHECK_RUN(fresh, fresh.code == 0 && !has(fresh.text, "(cached)"));
}

/* One ulp moved in one recorded sample fails the reference, which must reproduce its recording bit
 * for bit, and passes a library judged at its tolerance. */
TEST(one_ulp_in_the_recording_fails_the_reference_and_no_other_library) {
    REQUIRE(env().ok);
    Kfixture fx = recorded();
    REQUIRE(!fx.cases.empty());
    KfOperand& y = fx.cases[0].opd[2];
    REQUIRE(y.have_ref && !y.sample.empty());
    y.sample[5] = std::nextafter(y.sample[5], 1e9f);
    const std::string path = fixture_of("ulp.rkb", fx.cases);
    REQUIRE(!path.empty());
    const Out ref = run(env(), "--kernels libref --no-cache --max-bytes 64 --op add --fixture '" +
                                   path + "'");
    CHECK_RUN(ref, ref.code == 1);
    const Out ok = run(env(), "--kernels kbm_ok:libref --no-ref --max-bytes 64 --op add "
                              "--fixture '" + path + "'");
    CHECK_RUN(ok, ok.code == 0);
}

/* ================================================================== the libraries */

/* The library without a defect passes every case, its device rows inside guarded allocations. */
TEST(a_correct_library_passes_on_both_domains) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_ok"));
    CHECK_RUN(o, o.code == 0);
    /* Two add cases and gather_rows on two rows each (host and device), sample_argmax, rmsnorm. */
    CHECK_RUN(o, has(o.text, "8 checked, 8 passed, 0 failed"));
    CHECK_RUN(o, has(o.text, "3 case(s) ran inside guarded allocations, 0 wrote outside one"));
    CHECK_RUN(o, has(o.text, "time: "));
}

TEST(an_output_past_its_tolerance_fails) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_scale"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "kbm_add ") && has(o.text, "FAIL"));
}

TEST(a_nan_output_fails_as_a_finiteness_disagreement) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_nan"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "finite on one side and not the other"));
}

TEST(an_unwritten_half_fails) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_half"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "FAIL"));
}

/* The red zones stand in for device ASan: a write one byte past an output fails the run even
 * though every compared value is right. */
TEST(a_write_past_an_output_fails_the_run) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_overrun"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "wrote 1 byte(s) past the end of operand 'y'"));
    CHECK_RUN(o, has(o.text, "0 failed"));
}

TEST(a_refused_geometry_its_constraints_accept_fails) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_refuse"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "refused a geometry its constraints accept"));
}

/* Two libraries describing one output differently implement two ops under one name: counted as a
 * divergence, not compared. */
TEST(an_output_described_differently_is_a_divergence) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_shape"));
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "describes an output differently"));
    CHECK_RUN(o, has(o.text, "4 diverged"));
}

/* At a large id, an id off by one is inside a 2e-2 rel_l2; ids are compared exactly. */
TEST(an_id_off_by_one_fails) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_argmax"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "id(s) differ from the reference's"));
}

/* Inputs are shared by every library, so a host kernel writing into one is caught, named, and
 * fails the run. */
TEST(a_host_kernel_that_writes_its_input_fails) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_inwrite"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "wrote into its input 'a'"));
}

/* kbm_enc prefers a row whose layout hook refuses the weight's encoding; the engine passes it over,
 * so the replay must too, and run the plain row (which passes). */
TEST(a_row_that_cannot_read_the_weights_encoding_is_not_the_one_run) {
    REQUIRE(env().ok);
    const Out o = run(env(), replay("kbm_enc", "-v --op rmsnorm"));
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "kbm_rmsnorm "));
    CHECK_RUN(o, !has(o.text, "kbm_rmsnorm_wide"));
}

/* ================================================================== what is not checked */

TEST(a_case_over_the_budget_is_skipped_by_name) {
    REQUIRE(env().ok);
    const Out o = run(env(), "--kernels kbm_ok:libref --no-ref --strict --max-bytes 1 "
                             "--fixture '" + env().fx + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "over the 1.00 MiB budget"));
    CHECK_RUN(o, has(o.text, "6 checked, 6 passed"));
}

TEST(identical_cases_are_capped_at_max_cases_one) {
    REQUIRE(env().ok);
    const Kfixture fx = recorded();
    const std::string path = fixture_of("twice.rkb", { fx.cases[0], fx.cases[0] });
    REQUIRE(!path.empty());
    const Out one = run(env(), "--kernels kbm_ok:libref --no-ref --max-cases 1 --max-bytes 64 "
                               "--fixture '" + path + "'");
    CHECK_RUN(one, one.code == 0 && has(one.text, "2 checked, 2 passed, 0 failed, 0 skipped, "
                                                  "0 diverged, 2 capped"));
    const Out all = run(env(), "--kernels kbm_ok:libref --no-ref --max-cases 0 --max-bytes 64 "
                               "--fixture '" + path + "'");
    CHECK_RUN(all, all.code == 0 && has(all.text, "4 checked, 4 passed"));
}

/* A case no library under test serves is named, and is not a failure. */
TEST(a_case_no_library_under_test_serves_is_named) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 0) >= 0);
    REQUIRE(reg.check_complete() >= 0);
    KfCase c;
    Geometry g;
    g.set_i("M", 2); g.set_i("n", 16); g.set_s("dtype", "f32");
    REQUIRE(describe(reg, "softmax", g, 7, 2, &c));
    const std::string path = fixture_of("unserved.rkb", { c });
    const Out o = run(env(), "--kernels kbm_ok:libref --no-ref --strict --fixture '" + path +
                             "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "no library under test has a row for it"));
}

/* An op libref does not implement (qsa_table, qsa_attn) has its definition in the plugin that ships
 * the reference beside the kernel, so a case of it must be answered by that row -- not skipped as
 * having no oracle -- and the ops must replay as checked. Named by the row's own reference flag,
 * which is how the tool finds it without knowing a plugin's name.
 *
 * Before this, every such case read "no libref kernel: no oracle exists for it" and a run reported
 * hundreds of skips with nothing checked, which a speed gate then read as a green run. */
TEST(the_table_ops_are_answered_by_their_plugins_reference_and_replay_checked) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/qsatable.so", 0) >= 0);
    KfCase c;
    Geometry g;
    g.set_i("M", 4); g.set_i("blocks", 64); g.set_i("topk", 8); g.set_i("tile", 16);
    g.set_i("ratio", 4); g.set_i("u_max", 64);
    REQUIRE(describe_lib(reg, "qsatable", "qsa_table", g, 201, 4, &c));
    /* `selb` is a block-order list and `pos` its row's place: the law the op needs, carried by the
     * fixture so the replay draws the same operand. */
    REQUIRE(c.opd.size() == 7);
    CHECK(c.opd[0].is_index && c.opd[0].idx_ordered);
    CHECK_EQ(c.opd[0].idx_max, (int64_t)64);
    CHECK_EQ(c.opd[1].idx_const, (int64_t)9);
    CHECK_EQ(c.opd[2].idx_const, (int64_t)31);

    KfCase a;
    Geometry ag;
    ag.set_i("M", 4); ag.set_i("n_head", 4); ag.set_i("head_dim", 64); ag.set_i("gqa", 1);
    ag.set_i("block_size", 4); ag.set_i("ratio", 4); ag.set_i("tile", 16); ag.set_i("topk", 4);
    ag.set_i("blocks", 64); ag.set_i("max_seqs", 2); ag.set_i("u_max", 16);
    ag.set_s("q_dtype", "f32"); ag.set_s("kv_dtype", "f32");
    /* The reader's own case: the table the op produced is a block-order list too, and `bt` is the
     * staged block table (bt_pitch wide) the row path indexes with the step's pitch. */
    REQUIRE(describe_lib(reg, "qsatable", "qsa_attn", ag, 202, 4, &a));
    REQUIRE(a.opd.size() == 10);
    CHECK(a.opd[2].is_index && a.opd[2].idx_ordered);
    CHECK(a.opd[6].idx_const == 16);              /* pos: the last token of a full page */

    Kfixture base;
    base.source = "kbench_mutant_test";
    base.oracle = "libref";
    base.cases.push_back(c);
    base.cases.push_back(a);
    const std::string bpath = env().dir + "/qsa_base.rkb";
    REQUIRE_EQ(base.save(bpath), RAD_OK);

    /* Recording takes the answer from the plugin's host reference, though the fixture names libref:
     * libref has no row for the op at all. */
    const std::string fx = env().dir + "/qsa.rkb";
    const Out rec = run(env(), "--kernels libref:qsatable --max-bytes 64 --rerecord '" + fx +
                                 "' --fixture '" + bpath + "'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded 2 case(s)"));

    /* Replaying it is a check, not a skip: the row runs and its answers are compared exactly (it is
     * the oracle of its own op). libref stays loaded as the run's reference and has no row for the
     * op, which is what used to leave the case unserved. */
    const Out o = run(env(), "--kernels qsatable:libref --strict --max-bytes 64 --fixture '" + fx +
                             "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "2 checked, 2 passed, 0 failed, 0 skipped, 0 diverged, 0 capped"));
    CHECK_RUN(o, !has(o.text, "no oracle"));
    CHECK_RUN(o, !has(o.text, "stale"));

    /* And the gate refuses a --bench run that timed nothing: a table of no measurements is not a
     * table of zeroes. */
    const Out gate = run(env(), "--kernels libref --bench --fixture '" + fx + "'");
    CHECK_RUN(gate, gate.code == 1);
    CHECK_RUN(gate, has(gate.text, "The measurement gate refused"));
}

/* The other half of the gate: a run that checked every case it timed keeps its times. Without this
 * a gate that refuses everything would look like it works. */
TEST(a_checked_and_timed_run_publishes_its_times_and_passes_the_gate) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 0) >= 0);
    auto softmax_case = [&](int64_t M, uint64_t seed, KfCase* c) {
        Geometry g;
        g.set_i("M", M); g.set_i("n", 16); g.set_s("dtype", "f32");
        return describe(reg, "softmax", g, seed, M, c);
    };
    KfCase small, big;
    REQUIRE(softmax_case(2, 11, &small));
    REQUIRE(softmax_case(4096, 12, &big));
    Kfixture base;
    base.source = "kbench_mutant_test";
    base.oracle = "libref";
    base.cases = { small, big };
    REQUIRE_EQ(base.save(env().dir + "/gate_base.rkb"), RAD_OK);
    const std::string fx = env().dir + "/gate.rkb";
    const Out rec = run(env(), "--kernels libref --max-bytes 1000000 --rerecord '" + fx +
                                 "' --fixture '" + env().dir + "/gate_base.rkb'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded 2 case(s)"));

    const Out o = run(env(), "--kernels libref --bench --no-cache --fixture '" + fx + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "2 case(s) timed"));
    CHECK_RUN(o, !has(o.text, "The measurement gate refused"));
    CHECK_RUN(o, has(o.text, "2 checked, 2 passed, 0 failed, 0 skipped"));
}

/* A fixture whose operand count no longer matches the schema checks nothing; --strict (the ctests)
 * makes that a failure instead of a quiet skip. */
TEST(a_stale_case_fails_only_under_strict) {
    REQUIRE(env().ok);
    KfCase c = recorded().cases[0];
    c.opd.pop_back();
    const std::string path = fixture_of("stale.rkb", { c });
    const Out loose = run(env(), "--kernels kbm_ok:libref --no-ref --fixture '" + path + "'");
    CHECK_RUN(loose, loose.code == 0 && has(loose.text, "re-record this fixture"));
    const Out strict = run(env(), "--kernels kbm_ok:libref --no-ref --strict --fixture '" + path +
                                  "'");
    CHECK_RUN(strict, strict.code == 1);
}

/* ================================================================== coverage */

TEST(the_coverage_ratchet_fails_when_a_row_is_no_longer_reached) {
    REQUIRE(env().ok);
    const std::string dir = env().dir + "/rows";
    REQUIRE(::mkdir(dir.c_str(), 0755) == 0);
    const std::string file = dir + "/" + kb::rows_file_name(env().fx, "kbm_ok");
    REQUIRE_EQ(kb::write_rows_file(file, { "kbm_add", "kbm_vanished" }), RAD_OK);
    const Out lost = run(env(), replay("kbm_ok", "--coverage '" + dir + "'"));
    CHECK_RUN(lost, lost.code == 1);
    CHECK_RUN(lost, has(lost.text, "kbm_ok/kbm_vanished is no longer reached"));
    CHECK_RUN(lost, has(lost.text, "newly reached"));
    const Out wrote = run(env(), replay("kbm_ok", "--coverage '" + dir + "' --coverage-write"));
    CHECK_RUN(wrote, wrote.code == 0);
    bool exists = false;
    const std::set<std::string> rows = kb::read_rows_file(file, &exists);
    CHECK(rows.count("kbm_gather_rows") && rows.count("kbm_add_device") &&
          !rows.count("kbm_vanished"));
    const Out again = run(env(), replay("kbm_ok", "--coverage '" + dir + "'"));
    CHECK_RUN(again, again.code == 0);
}

/* kbm_add_small serves only M <= 4, which no recorded case uses. --record-rows moves the add case
 * into its constraint and records libref's answer there, so a replay reaches it. */
TEST(a_row_driven_fixture_reaches_a_row_no_recorded_case_selects) {
    REQUIRE(env().ok);
    const Out before = run(env(), replay("kbm_ok", "-v"));
    CHECK_RUN(before, !has(before.text, "kbm_add_small"));
    const std::string rows = env().dir + "/rows.rkb";
    const Out rec = run(env(), "--kernels kbm_ok:libref --no-ref --max-bytes 64 --record-rows '" +
                               rows + "' --fixture '" + env().fx + "'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded"));
    const Out o = run(env(), "--kernels kbm_ok:libref --no-ref --strict --all-rows -v "
                             "--max-bytes 64 --fixture '" + rows + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "kbm_add_small"));
}

/* ================================================================== a real library
 *
 * A replay draws a case's inputs by the laws recorded with it, so a fixture's sampler cases check
 * what their recording's laws reach. Here libref describes every sampler stage at three batch
 * sizes, as requests (RAD_FILL_SAMPLE_PARAMS) over a history that repeats, and libavx is replayed
 * against libref's answers: every stage's arithmetic is compared on rows it acts on, at libavx's
 * tolerances, whatever the checked-in fixtures were recorded with. */
TEST(the_host_library_agrees_with_the_reference_on_drawn_sampler_requests) {
    REQUIRE(env().ok);
    const std::string avx = std::string(RAD_PLUGIN_DIR) + "/kernels/libavx.so";
    if (::access(avx.c_str(), R_OK) != 0) ARTIFACT_MISSING(avx);
    Env e = env();
    e.home = e.dir + "/avxhome";
    for (const std::string& d : { e.home, e.home + "/kernels", e.home + "/architectures" })
        REQUIRE(::mkdir(d.c_str(), 0755) == 0);
    REQUIRE(::symlink(avx.c_str(), (e.home + "/kernels/libavx.so").c_str()) == 0);
    REQUIRE(::symlink((std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so").c_str(),
                      (e.home + "/kernels/libref.so").c_str()) == 0);

    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 0) >= 0);
    REQUIRE(reg.check_complete() >= 0);
    const char* const vocab_ops[] = { "sample_penalties", "sample_dry", "sample_temp",
                                      "sample_mask", "sample_argmax", "sample_topk",
                                      "sample_logprobs" };
    const char* const cand_ops[] = { "sample_topp", "sample_minp", "sample_typical", "sample_xtc",
                                     "sample_pick" };
    std::vector<KfCase> cases;
    uint64_t seed = 300;
    for (int64_t M : { 1, 7, 64 }) {
        for (const char* op : vocab_ops) {
            Geometry g;
            g.set_i("M", M); g.set_i("n_vocab", 1000);
            if (!std::strcmp(op, "sample_logprobs")) g.set_i("k", 21);
            KfCase c;
            REQUIRE(describe(reg, op, g, ++seed, M, &c));
            cases.push_back(c);
        }
        for (const char* op : cand_ops) {
            Geometry g;
            g.set_i("M", M); g.set_i("n_cand", 1024);
            KfCase c;
            REQUIRE(describe(reg, op, g, ++seed, M, &c));
            cases.push_back(c);
        }
    }
    const std::string base = fixture_of("sampler_base.rkb", cases);
    REQUIRE(!base.empty());
    const std::string fx = e.dir + "/sampler.rkb";
    const Out rec = run(e, "--kernels libref --rerecord '" + fx + "' --fixture '" + base + "'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded 36 case(s)"));
    const Out o = run(e, "--kernels libavx:libref --no-ref --strict --max-cases 0 --fixture '" +
                         fx + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "36 checked, 36 passed, 0 failed"));
}

/* ================================================================== the clock
 *
 * The timing pass on the host backend: no card numbers, but every path a card takes. The flags are
 * the cheap end of each: a 1 MiB rotation footprint, three batches of two launches, no warm-up. */

std::string read_text(const std::string& path) {
    std::string s;
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return s;
    char buf[4096];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, got);
    std::fclose(f);
    return s;
}

/* The cells of the speed table's row for `kernel` (and `geom`, when given), trimmed: [1] op ... [5]
 * device us, [6] host us, [9] n, [13] notes. Empty when there is none. */
std::vector<std::string> speed_row(const std::string& md, const std::string& kernel,
                                   const std::string& geom = "") {
    const size_t sp = md.find("## Speed");
    if (sp == std::string::npos) return {};
    size_t at = sp;
    for (;;) {
        at = md.find("| `" + kernel + "` |", at);
        if (at == std::string::npos) return {};
        const size_t bol = md.rfind('\n', at) + 1, eol = md.find('\n', at);
        const std::string line = md.substr(bol, eol - bol);
        at = eol;
        if (!geom.empty() && !has(line, "`" + geom + "`")) continue;
        std::vector<std::string> cells;
        size_t i = 0;
        while (i < line.size()) {
            size_t j = line.find('|', i);
            if (j == std::string::npos) j = line.size();
            std::string c = line.substr(i, j - i);
            while (!c.empty() && c.front() == ' ') c.erase(0, 1);
            while (!c.empty() && c.back() == ' ') c.pop_back();
            cells.push_back(c);
            i = j + 1;
        }
        return cells;
    }
}

double num(const std::string& s) { return s.empty() || s == "-" ? -1.0 : std::atof(s.c_str()); }

const char* const kQuickBench = "--bench --bench-flush 1 --bench-iters 2 --bench-reps 3 "
                                "--bench-warm-ms 0";

/* ================================================================== the row, not the library */

/* One library can ship both: plugin/qsatable marks its host rows as the definitions of its ops and
 * its device rows as the kernels to be checked (`rad_kernel_reference`, per row). The container pass
 * asked whether a row's *library* was the reference, so once ref_for held the host row every row of
 * that library counted as the reference -- including the device row there is to check -- and was
 * skipped as if it had been checked. Row identity decides: the definition is skipped, the kernel
 * beside it runs. */
TEST(a_plugins_device_row_is_checked_against_the_definition_in_the_same_plugin) {
    const Env pair = home_with("pair", { "libref", "kbm_pair" });
    REQUIRE(pair.ok);
    const std::string model = tiny_model();
    REQUIRE(!model.empty());
    const std::string report = pair.home + "/pair.md";
    /* libref is loaded for its `add` schema and its norm and projection rows; `--ref kbm_pair` is
     * the recording's reference, and the model's add is its device row. */
    const Out o = run(pair, "-m '" + model + "' --kernels libref:kbm_pair --ref kbm_pair "
                             "--max-tok 4 --max-bytes 64 --report '" + report + "'");
    CHECK_RUN(o, o.code == 0);
    const std::string md = read_text(report);
    /* A row is only in the report when it was checked (record_only rows are not), so the device
     * row's presence is the whole claim; the host row beside it stayed the recording's. */
    CHECK(has(md, "| `add` | `kbm_pair_add_device` |"));
    CHECK(!has(md, "| `add` | `kbm_pair_add` |"));
}

/* The same shape in the fixture pass: `exact` was `lib == oracle`, true for the device row of the
 * library that holds the definition too, so it was held to the reference's bytes instead of its
 * dtype's bound. Bound to the row the recording came from, a rounding-sized difference passes and a
 * real divergence on the same row still fails. */
TEST(a_kernels_device_row_is_held_to_its_tolerance_not_its_librarys_exactness) {
    REQUIRE(env().ok);
    /* The add case libref recorded: host row and device row of kbm_pair both serve it. */
    const std::string base = fixture_of("pair_base.rkb", { recorded().cases[0] });
    const std::string fx = env().dir + "/pair.rkb";
    const Out rec = run(env(), "--kernels libref:kbm_pair --ref kbm_pair --max-bytes 64 "
                               "--rerecord '" + fx + "' --fixture '" + base + "'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded 1 case(s)"));

    /* Its own device row is 1e-4 off, inside bf16's 6e-3: it passes at that bound and would fail at
     * the reference's. */
    const Out ok = run(env(), "--kernels libref:kbm_pair --ref kbm_pair --strict --max-bytes 64 "
                              "--fixture '" + fx + "'");
    CHECK_RUN(ok, ok.code == 0);
    CHECK_RUN(ok, !has(ok.text, "FAIL"));
    CHECK_RUN(ok, has(ok.text, "3 checked, 3 passed, 0 failed"));

    /* The kernel that is 1.8% off, on the same recording and the same row: the tolerance is a bound,
     * not a bypass. */
    const Out bad = run(env(), "--kernels libref:kbm_pair:kbm_scale --ref kbm_pair --strict "
                               "--max-bytes 64 --fixture '" + fx + "'");
    CHECK_RUN(bad, bad.code == 1);
    CHECK_RUN(bad, has(bad.text, "kbm_add_device") && has(bad.text, "FAIL"));
}

/* A recording that names a library which does not serve its op, while another loaded library
 * declares a reference for it: not stale, and checked against that library's host row. `--rerecord`
 * writes the run's reference into the fixture, so a `qsa_table` case that `qsatable`'s host row
 * answered is written as `libref`'s -- and libref implements no `qsa_table`. The case used to be
 * reported unserved and lost its exactness; an unresolvable reference is the stale one. */
TEST(a_recording_whose_library_does_not_serve_it_runs_against_the_one_that_does) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/qsatable.so", 0) >= 0);
    KfCase c;
    Geometry g;
    g.set_i("M", 4); g.set_i("blocks", 64); g.set_i("topk", 8); g.set_i("tile", 16);
    g.set_i("ratio", 4); g.set_i("u_max", 64);
    REQUIRE(describe_lib(reg, "qsatable", "qsa_table", g, 211, 4, &c));
    const std::string base = fixture_of("gone_base.rkb", { c });
    const std::string fx = env().dir + "/gone.rkb";
    const Out rec = run(env(), "--kernels libref:qsatable --max-bytes 64 --rerecord '" + fx +
                                 "' --fixture '" + base + "'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded 1 case(s)"));

    /* The fixture now names libref, which has no row for the op; qsatable's host reference does. */
    const Out o = run(env(), "--kernels qsatable:libref --strict --max-bytes 64 --fixture '" +
                             fx + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, !has(o.text, "stale"));
    CHECK_RUN(o, has(o.text, "1 checked, 1 passed, 0 failed"));
    CHECK_RUN(o, has(o.text, "worst rel_l2 = 0.000e+00"));
}

/* The attention reader takes its entry count from `tlen` and reads `tab` and `mask` below it, so a
 * drawn count wider than the plane's declared `u_max` is a read past the row. The shape hook caps it
 * at the width, whatever `topk` says. */
TEST(a_drawn_table_length_never_leaves_the_planes_declared_width) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/qsatable.so", 0) >= 0);
    KfCase c;
    Geometry g;
    /* topk 4 into a plane two ids wide: the reader would walk four. */
    g.set_i("M", 1); g.set_i("n_head", 1); g.set_i("head_dim", 64); g.set_i("gqa", 1);
    g.set_i("block_size", 4); g.set_i("ratio", 4); g.set_i("tile", 1); g.set_i("topk", 4);
    g.set_i("blocks", 64); g.set_i("max_seqs", 1); g.set_i("u_max", 2);
    g.set_s("q_dtype", "f32"); g.set_s("kv_dtype", "f32");
    REQUIRE(describe_lib(reg, "qsatable", "qsa_attn", g, 212, 1, &c));
    REQUIRE(c.opd.size() == 10);
    CHECK_EQ(c.opd[2].shape.back(), (int64_t)2);       /* tab: the union's declared width */
    CHECK_EQ(c.opd[4].shape.back(), (int64_t)2);       /* mask */
    CHECK_EQ(c.opd[3].idx_const, (int64_t)2);          /* tlen: inside it */

    /* And the reference reads it without leaving the row. */
    const std::string base = fixture_of("narrow_base.rkb", { c });
    const std::string fx = env().dir + "/narrow.rkb";
    const Out rec = run(env(), "--kernels libref:qsatable --max-bytes 64 --rerecord '" + fx +
                                 "' --fixture '" + base + "'");
    CHECK_RUN(rec, rec.code == 0 && has(rec.text, "recorded 1 case(s)"));
    const Out o = run(env(), "--kernels qsatable:libref --strict --max-bytes 64 --fixture '" + fx +
                             "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "1 checked, 1 passed, 0 failed"));
}

/* The gate's negative half: a case of a timed op that was skipped or went stale withdraws every
 * time of that op and the run ends 1. Keeping the times of the cases that happened to be checked
 * would publish a measurement of part of an op as the op's. --timing-only compares nothing, so it
 * keeps what it timed. Both halves are stated on the report itself: a withdrawn time is a row with
 * no number in it, and a kept one is a row with a number -- a row that is simply missing proves
 * neither, so every row checked here is required to exist. */
TEST(a_timed_op_with_a_skipped_or_stale_case_publishes_no_time) {
    REQUIRE(env().ok);
    const std::vector<KfCase> all = recorded().cases;
    REQUIRE(all.size() == 5);          /* add, gather_rows, sample_argmax, rmsnorm, add 32768 */
    /* One add the replay takes and one it must skip: 3 x 8 x 32768 bf16 operands do not fit a 1 MiB
     * budget, the 24 KiB case does. */
    const std::string over = fixture_of("gate_over.rkb", { all[0], all[4] });
    const std::string report = env().dir + "/gate_over.md";
    const Out sk = run(env(), "--kernels kbm_ok:libref --no-ref --strict --max-bytes 1 --fixture '" +
                                 over + "' " + kQuickBench + " --report '" + report + "'");
    CHECK_RUN(sk, sk.code == 1);
    CHECK_RUN(sk, has(sk.text, "The measurement gate refused"));
    /* Both rows of the op are still listed -- the report names what was not timed -- with no
     * number in either column and the gate's reason beside them. */
    const std::string md = read_text(report);
    CHECK(has(md, "**Device timings withheld:**"));
    const std::vector<std::string> host = speed_row(md, "kbm_add");
    const std::vector<std::string> dev = speed_row(md, "kbm_add_device");
    REQUIRE(host.size() >= 14);
    REQUIRE(dev.size() >= 14);
    CHECK_EQ(host[5], std::string("-"));      /* device column: a host row has no device time */
    CHECK_EQ(host[6], std::string("-"));      /* the time the gate took away */
    CHECK_EQ(dev[5], std::string("-"));
    CHECK_EQ(dev[6], std::string("-"));
    CHECK_EQ(host[9], std::string("-"));      /* `n`: no batches survive either */
    CHECK(has(host[13], "withheld"));
    CHECK(has(dev[13], "withheld"));

    /* The same op registered as stale: its row was timed under the case that was checked and its
     * time is removed just the same -- a stale case of an op withdraws what the checked case of it
     * measured. A run that only ever reported a missing op would hide exactly this. */
    KfCase stale = all[0];
    stale.opd.pop_back();
    stale.seed ^= 1;
    const std::string sfile = fixture_of("gate_stale.rkb", { all[0], stale });
    const std::string sreport = env().dir + "/gate_stale.md";
    const Out st = run(env(), "--kernels kbm_ok:libref --no-ref --strict --max-bytes 64 "
                              "--fixture '" + sfile + "' " + kQuickBench + " --report '" +
                              sreport + "'");
    CHECK_RUN(st, st.code == 1);
    CHECK_RUN(st, has(st.text, "The measurement gate refused"));
    CHECK_RUN(st, has(st.text, "stale"));
    const std::string smd = read_text(sreport);
    CHECK(has(smd, "**Device timings withheld:**"));
    const std::vector<std::string> shost = speed_row(smd, "kbm_add");
    REQUIRE(shost.size() >= 14);
    for (size_t col : { 5u, 6u, 7u, 8u, 9u })   /* time, min, p90, n: every one gone */
        CHECK_EQ(shost[col], std::string("-"));
    CHECK(has(shost[13], "withheld"));

    /* A timing-only run has no comparison to lose: it publishes what it timed and stays 0. The
     * device row is required to carry a real number, so the absence above is evidence of the gate
     * and not of a report that cannot hold a time at all. */
    const std::string oreport = env().dir + "/gate_only.md";
    const Out only = run(env(), "--kernels kbm_ok:libref --no-ref --max-bytes 64 --fixture '" +
                                sfile + "' --timing-only " + kQuickBench + " --report '" +
                                oreport + "'");
    CHECK_RUN(only, only.code == 0);
    CHECK_RUN(only, has(only.text, "timed. Timing only"));
    const std::string omd = read_text(oreport);
    CHECK(!has(omd, "**Device timings withheld:**"));
    const std::vector<std::string> otimed = speed_row(omd, "kbm_add_device");
    REQUIRE(otimed.size() >= 14);
    CHECK(num(otimed[5]) > 0);                 /* a time that was actually taken */
    CHECK(num(otimed[9]) >= 1);                /* and the batches behind it */
}

/* The gate's negative half in a container pass, which books its skips itself: a container whose
 * cases are partly over the budget checks and times the rest, and every time it took is withheld --
 * the checked cases of an op do not stand for the op whose other cases were skipped. Before this
 * test the container path only ever proved that a refused gate was printed. */
TEST(a_container_pass_that_skipped_cases_withholds_the_op_times_it_took) {
    REQUIRE(env().ok);
    const std::string model = tiny_model();
    REQUIRE(!model.empty());
    const std::string report = env().dir + "/container_gate.md";
    /* 65536 tokens of a 64-wide bf16 activation is 25 MiB per copy, so most cases are skipped by
     * name against a 1 MiB budget and the smallest ones still run and are timed. */
    const Out c = run(env(), "-m '" + model + "' --kernels kbm_ok:libref --max-cases 1 "
                             "--max-tok 65536 --max-bytes 1 " + std::string(kQuickBench) +
                             " --report '" + report + "'");
    CHECK_RUN(c, c.code == 1);
    CHECK_RUN(c, has(c.text, "The measurement gate refused"));
    CHECK_RUN(c, has(c.text, "were skipped or are stale"));
    CHECK_RUN(c, has(c.text, "cannot stand for the op"));
    CHECK_RUN(c, has(c.text, "budget -- raise --max-bytes"));
    const std::string md = read_text(report);
    const size_t speed = md.find("## Speed");
    REQUIRE(speed != std::string::npos);
    /* Every row of the op is listed with the gate's reason and no number anywhere, so a reader
     * cannot mistake a withheld op for one that was not run. */
    int rows = 0;
    for (size_t at = md.find("| `rmsnorm` |", speed); at != std::string::npos;
         at = md.find("| `rmsnorm` |", at + 1)) {
        const size_t eol = md.find('\n', at);
        const std::string line = md.substr(at, eol - at);
        /* Every column from `device us` to `MHz` is a dash, `n` included: nothing was kept. */
        CHECK(has(line, "| - | - | - | - | - | - | - |"));
        CHECK(has(line, "withheld"));
        ++rows;
    }
    CHECK(rows > 0);
}

/* Device rows are timed after the checks, host rows where they are checked, each in its own
 * column with the calibration above the table. A device row that queues its work (the copy-based
 * gather) is timed behind the gate; one that drains its own stream when called cannot be, and is
 * timed without the gate rather than deadlocking, saying so. Timing changes no verdict. */
TEST(device_rows_are_timed_after_the_checks_and_host_rows_where_checked) {
    REQUIRE(env().ok);
    const std::string report = env().dir + "/bench.md";
    const Out o = run(env(), replay("kbm_ok", std::string("--op add --op gather_rows ") +
                                                  kQuickBench + " --report '" + report + "'"));
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "6 checked, 6 passed, 0 failed"));
    CHECK_RUN(o, has(o.text, "timing: 6 case(s) timed, 1 of the 3 on the device behind a gate"));
    CHECK_RUN(o, has(o.text, "launch floor:"));
    const std::string md = read_text(report);
    CHECK(has(md, "## Speed") && has(md, "launch floor:") && has(md, "cold: each launch reads"));
    const std::vector<std::string> gated = speed_row(md, "kbm_gather_rows_device");
    REQUIRE(gated.size() >= 14);
    CHECK(num(gated[5]) > 0);
    CHECK_EQ(gated[6], std::string("-"));
    CHECK_EQ(gated[9], std::string("3"));
    CHECK(!has(gated[13], "ungated"));
    const std::vector<std::string> inline_dev = speed_row(md, "kbm_add_device");
    REQUIRE(inline_dev.size() >= 14);
    CHECK(num(inline_dev[5]) > 0);
    CHECK(has(inline_dev[13], "ungated"));
    const std::vector<std::string> host = speed_row(md, "kbm_add");
    REQUIRE(host.size() >= 14);
    CHECK_EQ(host[5], std::string("-"));
    CHECK(num(host[6]) > 0);
    CHECK(has(host[13], "thread(s)"));
}

/* An in-place operand is restored from its pristine copy before every timed launch. The host
 * scatter refuses to write a row that already holds its value, so a second launch on an
 * unrestored operand would fail its timing; the device scatter's restores are timed and
 * subtracted, and their share named. On the host backend that kernel is a few row writes, no
 * longer than its restores, and on a loaded machine the restores' own batch can come out longer
 * than the timed one: the row is then reported restore-dominated, without a time or a share. */
TEST(an_in_place_operand_is_restored_before_every_timed_launch) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 0) >= 0);
    REQUIRE(reg.check_complete() >= 0);
    KfCase c;
    Geometry g;
    g.set_i("M", 4); g.set_i("n", 32); g.set_s("dtype", "bf16");
    REQUIRE(describe(reg, "scatter_rows", g, 201, 4, &c));
    const std::string base = fixture_of("scatter_base.rkb", { c });
    const std::string fx = env().dir + "/scatter.rkb";
    const Out rec = run(env(), "--kernels libref --max-bytes 64 --rerecord '" + fx + "' --fixture '" +
                                   base + "'");
    REQUIRE(rec.code == 0 && has(rec.text, "recorded 1 case(s)"));
    const std::string report = env().dir + "/scatter.md";
    const Out o = run(env(), "--kernels kbm_ok:libref --no-ref --strict --max-bytes 64 --fixture '" +
                                 fx + "' " + kQuickBench + " --report '" + report + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "2 checked, 2 passed, 0 failed"));
    const std::string md = read_text(report);
    const std::vector<std::string> host = speed_row(md, "kbm_scatter_rows");
    REQUIRE(host.size() >= 14);
    CHECK(num(host[6]) > 0);
    CHECK(!has(host[13], "failed"));
    const std::vector<std::string> dev = speed_row(md, "kbm_scatter_rows_device");
    REQUIRE(dev.size() >= 14);
    if (!has(dev[13], "restore-dominated")) {
        CHECK(num(dev[9]) >= 1);
        CHECK(has(dev[13], "restore "));
    }
}

/* --bench-flush 0 times warm: no rotation, and no read probe, which exists to check cold
 * reads. */
TEST(warm_timing_rereads_one_set_of_operands) {
    REQUIRE(env().ok);
    const std::string report = env().dir + "/warm.md";
    const Out o = run(env(), replay("kbm_ok", "--op gather_rows --bench --bench-flush 0 "
                                              "--bench-iters 2 --bench-reps 3 --bench-warm-ms 0 "
                                              "--report '" + report + "'"));
    CHECK_RUN(o, o.code == 0);
    const std::string md = read_text(report);
    CHECK(has(md, "warm: every launch rereads the same operands"));
    CHECK(has(md, "read bandwidth: not probed"));
    CHECK(has(md, "every operand is reused"));
    const std::vector<std::string> gated = speed_row(md, "kbm_gather_rows_device");
    REQUIRE(gated.size() >= 14);
    CHECK(num(gated[5]) > 0);
}

/* A weight table is timed through pointer tables over its rotated copies, each launch's experts
 * sent to other slots, on routing drawn skewed for timing: libref's expert GEMM reads both, and
 * its checked answer is still its recording's. */
TEST(a_weight_table_is_timed_through_rotated_pointer_tables) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 0) >= 0);
    REQUIRE(reg.check_complete() >= 0);
    KfCase c;
    Geometry g;
    g.set_i("M", 2); g.set_i("N", 64); g.set_i("K", 128); g.set_i("n_expert", 4);
    g.set_i("top_k", 2); g.set_i("group", 128); g.set_s("a_order", "sorted");
    g.set_s("dtype", "fp8a8");
    REQUIRE(describe(reg, "moe_gemm_q", g, 301, 2, &c));
    const std::string base = fixture_of("moe_base.rkb", { c });
    const std::string fx = env().dir + "/moe.rkb";
    const Out rec = run(env(), "--kernels libref --max-bytes 64 --rerecord '" + fx + "' --fixture '" +
                                   base + "'");
    REQUIRE(rec.code == 0 && has(rec.text, "recorded 1 case(s)"));
    const std::string report = env().dir + "/moe.md";
    const Out o = run(env(), "--kernels libref --strict --no-cache --max-bytes 64 -v --fixture '" +
                                 fx + "' " + kQuickBench + " --report '" + report + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "1 checked, 1 passed, 0 failed"));
    /* w and w_odd with their scale tables: four weight tables, every one rotated. */
    CHECK_RUN(o, has(o.text, "4 weight table(s) through pointer tables"));
    const std::string md = read_text(report);
    const std::vector<std::string> row = speed_row(md, "ref_moe_gemm_q");
    REQUIRE(row.size() >= 14);
    CHECK(num(row[6]) > 0);
    CHECK(!has(row[13], "failed"));
}

/* The container pass hands its passing cases to the same timing pass: its device rows are timed
 * after the checks, its host rows where they are checked, and the report carries both. */
TEST(the_container_pass_times_its_cases_too) {
    REQUIRE(env().ok);
    const std::string model = tiny_model();
    REQUIRE(!model.empty());
    const std::string report = env().dir + "/container.md";
    const Out o = run(env(), "-m '" + model + "' --kernels kbm_ok:libref --max-cases 1 --max-tok 4 "
                             "--max-bytes 64 " + std::string(kQuickBench) + " --report '" + report +
                             "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "timing: ") && has(o.text, "launch floor:"));
    const std::string md = read_text(report);
    const std::vector<std::string> dev = speed_row(md, "kbm_add_device");
    REQUIRE(dev.size() >= 14);
    CHECK(num(dev[5]) > 0);
    const std::vector<std::string> host = speed_row(md, "kbm_rmsnorm");
    REQUIRE(host.size() >= 14);
    CHECK(num(host[6]) > 0);
}

/* Attention's depth lives in its cache's extent, not its parameters: two cases with the same
 * parameters at depths 64 and 1024 are two report rows, each with its own time, not one row
 * holding the minimum of one and the median of the other. */
TEST(cases_that_differ_only_in_depth_are_reported_apart) {
    REQUIRE(env().ok);
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 0) >= 0);
    REQUIRE(reg.check_complete() >= 0);
    std::vector<KfCase> cases;
    for (const int64_t ctx : { (int64_t)64, (int64_t)1024 }) {
        KfCase c;
        Geometry g;
        g.set_i("q_len", 1); g.set_i("head_dim", 64); g.set_i("gqa", 1); g.set_i("n_head", 1);
        g.set_i("block_size", 16); g.set_i("causal", 1); g.set_i("window", 0);
        g.set_s("q_dtype", "bf16"); g.set_s("kv_dtype", "bf16"); g.set_i("ctx", ctx);
        REQUIRE(describe(reg, "attn_paged", g, 401, 1, &c));
        /* The depth is the description's input, as in a recording; the case does not carry it. */
        c.params.erase(std::remove_if(c.params.begin(), c.params.end(),
                                      [](const KfParam& p) { return p.key == "ctx"; }),
                       c.params.end());
        cases.push_back(c);
    }
    const std::string base = fixture_of("depth_base.rkb", cases);
    const std::string fx = env().dir + "/depth.rkb";
    const Out rec = run(env(), "--kernels libref --max-bytes 64 --rerecord '" + fx + "' --fixture '" +
                                   base + "'");
    REQUIRE(rec.code == 0 && has(rec.text, "recorded 2 case(s)"));
    const std::string report = env().dir + "/depth.md";
    const Out o = run(env(), "--kernels libref --strict --no-cache --max-cases 0 --max-bytes 64 "
                             "--fixture '" + fx + "' " + kQuickBench + " --report '" + report +
                             "'");
    CHECK_RUN(o, o.code == 0 && has(o.text, "2 checked, 2 passed"));
    const std::string md = read_text(report);
    const size_t speed = md.find("## Speed");
    REQUIRE(speed != std::string::npos);
    int correctness_rows = 0, speed_rows = 0;
    for (size_t at = md.find("| `ref_attn_paged` |"); at != std::string::npos;
         at = md.find("| `ref_attn_paged` |", at + 1))
        ++(at < speed ? correctness_rows : speed_rows);
    CHECK_EQ(correctness_rows, 2);
    CHECK_EQ(speed_rows, 2);
}

/* Only a case that passed is timed, in a replay and in a container pass alike: kbm_scale's add is
 * past its tolerance on both domains, so neither add row has a time, while the gather beside it
 * does. */
TEST(a_failing_case_is_not_timed) {
    REQUIRE(env().ok);
    const std::string report = env().dir + "/failing.md";
    const Out o = run(env(), replay("kbm_scale", std::string("--op add --op gather_rows ") +
                                                     kQuickBench + " --report '" + report + "'"));
    CHECK_RUN(o, o.code == 1);
    CHECK_RUN(o, has(o.text, "timing: 2 case(s) timed"));
    std::string md = read_text(report);
    CHECK(speed_row(md, "kbm_add").empty());
    CHECK(speed_row(md, "kbm_add_device").empty());
    CHECK(speed_row(md, "kbm_gather_rows_device").size() >= 14);

    const std::string model = tiny_model();
    REQUIRE(!model.empty());
    const std::string creport = env().dir + "/failing_container.md";
    const Out c = run(env(), "-m '" + model + "' --kernels kbm_scale:libref --max-cases 1 "
                             "--max-tok 4 --max-bytes 64 " + std::string(kQuickBench) +
                             " --report '" + creport + "'");
    CHECK_RUN(c, c.code == 1);
    md = read_text(creport);
    CHECK(has(md, "## Speed"));
    CHECK(speed_row(md, "kbm_add_device").empty());
    CHECK(speed_row(md, "kbm_rmsnorm").size() >= 14);
}

/* --timing-only times the device rows and nothing else: no check, no host row. */
TEST(timing_only_times_device_rows_and_checks_nothing) {
    REQUIRE(env().ok);
    const std::string report = env().dir + "/only.md";
    const Out o = run(env(), "--kernels kbm_ok:libref --no-ref --max-bytes 64 --op add --op "
                             "gather_rows --timing-only --bench-flush 1 --bench-iters 2 "
                             "--bench-reps 3 --bench-warm-ms 0 --fixture '" + env().fx +
                             "' --report '" + report + "'");
    CHECK_RUN(o, o.code == 0);
    CHECK_RUN(o, has(o.text, "3 timed. Timing only"));
    const std::string md = read_text(report);
    CHECK(speed_row(md, "kbm_add_device").size() >= 14);
    CHECK(speed_row(md, "kbm_add").empty());
}

/* ================================================================== recording without a card */

/* Recording reads a container's tables, draws every weight, and takes libref's answer at every
 * point whichever library resolved it. So a host with libref alone records, though every op
 * resolves to libref itself, and the container's skeleton (tables only, data a hole) records the
 * same fixture as the container. */
TEST(a_fixture_is_recorded_on_the_host_from_a_container_or_its_skeleton) {
    REQUIRE(env().ok);
    const std::string dir = env().dir;
    const std::string model = tiny_model();
    REQUIRE(!model.empty());
    const std::string a = dir + "/rec_full.rkb", skel = dir + "/tiny.skel.rad",
                      b = dir + "/rec_skel.rkb";
    const Out full = run(env(), "-m '" + model + "' --kernels libref --max-cases 1 --max-tok 64 "
                                "--max-bytes 64 --record '" + a + "'");
    CHECK_RUN(full, full.code == 0 && has(full.text, "recorded"));
    CHECK_RUN(full, has(full.text, "time: ") && has(full.text, "(libref "));
    const Out sk = run(env(), "-m '" + model + "' --skeleton '" + skel + "'");
    CHECK_RUN(sk, sk.code == 0);
    const Out fromskel = run(env(), "-m '" + skel + "' --kernels libref --max-cases 1 "
                                    "--max-tok 64 --max-bytes 64 --record '" + b + "'");
    CHECK_RUN(fromskel, fromskel.code == 0);

    Kfixture fa, fb;
    REQUIRE_EQ(fa.load(a), RAD_OK);
    REQUIRE_EQ(fb.load(b), RAD_OK);
    /* Every op at every point of its M sweep (1, 2, 4, ..., 64), once. */
    std::set<std::string> ops;
    for (const KfCase& c : fa.cases) ops.insert(c.op);
    CHECK((ops == std::set<std::string>{ "add", "gemm_nt", "rmsnorm" }));
    CHECK_EQ(fa.cases.size(), (size_t)21);
    REQUIRE_EQ(fa.cases.size(), fb.cases.size());
    for (size_t i = 0; i < fa.cases.size(); ++i) {
        const KfCase& x = fa.cases[i];
        const KfCase& y = fb.cases[i];
        CHECK_EQ(x.op, y.op);
        CHECK_EQ(x.seed, y.seed);
        REQUIRE_EQ(x.opd.size(), y.opd.size());
        for (size_t k = 0; k < x.opd.size(); ++k) {
            CHECK(x.opd[k].shape == y.opd[k].shape);
            CHECK_EQ(x.opd[k].has_enc, y.opd[k].has_enc);
            CHECK_EQ(x.opd[k].ref_sumsq, y.opd[k].ref_sumsq);
            CHECK(x.opd[k].sample == y.opd[k].sample);
            CHECK(x.opd[k].sketch == y.opd[k].sketch);
        }
    }
    /* And libref replays it exactly. */
    const Out rep = run(env(), "--kernels libref --strict --no-cache --fixture '" + a + "'");
    CHECK_RUN(rep, rep.code == 0 && has(rep.text, "21 checked, 21 passed"));
}

/* ================================================================== the selector */

/* The replay asks each library alone (Registry::select_in) with the builder's encoding filter;
 * with one library in the hierarchy that is the engine's own answer (Registry::select). */
TEST(select_in_answers_as_the_engine_does_for_one_library) {
    Registry reg;
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/testkernels/kbmut/kbm_enc.so", 0) >= 0);
    REQUIRE(reg.load_plugin(std::string(RAD_PLUGIN_DIR) + "/kernels/libref.so", 1) >= 0);
    REQUIRE(reg.check_complete() >= 0);
    Geometry g;
    g.set_i("M", 4); g.set_i("n", 64); g.set_f("eps", 1e-6); g.set_s("dtype", "bf16");
    const RadEncoding enc = rad_enc_plain(RAD_BF16);
    const int sel = 0;
    RadTensor plane{};
    plane.dtype = RAD_BF16;
    plane.rank = 2;
    plane.shape[0] = 1;
    plane.shape[1] = 64;
    const Registry::RowFilter accept = [&](const KernelRow& r, const Geometry& gg) {
        return weight_refusal(r.info, gg, 1, enc, &sel, &plane, 1, "w");
    };
    const KernelRow* engine = reg.select("rmsnorm", g, RAD_DOMAIN_HOST, nullptr, false, &accept);
    const KernelRow* replay_row = reg.select_in("kbm_enc", "rmsnorm", g, RAD_DOMAIN_HOST, &accept);
    REQUIRE(engine != nullptr);
    CHECK(engine == replay_row);
    CHECK_EQ(std::string(engine->info->name), std::string("kbm_rmsnorm"));
    /* Without the filter the preferred row wins: what a replay without it would run. */
    const KernelRow* unfiltered = reg.select_in("kbm_enc", "rmsnorm", g, RAD_DOMAIN_HOST);
    REQUIRE(unfiltered != nullptr);
    CHECK_EQ(std::string(unfiltered->info->name), std::string("kbm_rmsnorm_wide"));
    /* Restricted to a plugin, a row of another never answers. */
    const KernelRow* ref = reg.select_in("libref", "rmsnorm", g, RAD_DOMAIN_HOST);
    REQUIRE(ref != nullptr);
    CHECK_EQ(ref->plugin, std::string("libref"));
}

RAD_TEST_MAIN()
