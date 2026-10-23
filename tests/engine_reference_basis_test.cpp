/* engine_reference_basis_test.cpp -- the no-loan reference at the engine's own call sites.
 *
 * The reference basis (c0ced01, docs/ADAPTIVE-MTP.md) is defined by three engine decisions, and
 * this file pins each where the engine makes it rather than where the table consumes it:
 *
 *  1. A reference calibration is a *cost* measurement with an output path, and it runs with both
 *     lenders off. The startup contract is Engine::begin_draft_calibration's; the lender-free
 *     state is what the run leaves on its ranks, and the export's own guard
 *     (Engine::export_draft_cost_profile) is the engine's enforceable proof: a run that lent
 *     anything -- KV or arena -- must not write a file that claims a no-loan basis.
 *  2. A serving run keeps its loan dynamic and still admits the reference. The serving run's
 *     identity is derived by Engine::draft_cost_runtime from the loan it actually took, so it
 *     names a non-zero loan while the reference file has none. That file loads, stays active and
 *     is flagged reference(); a changed arena-level identity or a changed plan still refuses it,
 *     because a real placement change is not a loan move.
 *  3. A live profile keeps the strict treatment: the same file, loaded against a run whose loan
 *     differs, is inactive and names live_loan. The reference's exemption is the reference's.
 *
 * ADMISSION TEST, NOT A PROOF OF REAL DYNAMIC LOANS. The runtime fingerprint is derived by the
 * engine, but the profile *file* is hand-built from it (admission_profile_file below) and the
 * lending deployment is a fixture. What these cases prove is that a reference file is admitted
 * exactly when its identity matches a run that lends -- not that a real serving run, with its own
 * model, plan and card, borrows, returns and is then matched against its reference.
 *
 * The deployment around the ranks is a fixture (tests/engine_start_fill_test.cpp's shape: a
 * fixture program, the Planner's real plan, a real Mover and a real loan), so no model, plugin or
 * card is involved. Not covered, and not to be read as covered:
 *   - the actual lender proof: the in-start() part of "both lenders off" -- `Engine::start` skips
 *     `lend_from` for kLendKV/kLendArena under --draft-reference (core/engine_bringup.cpp) --
 *     needs a started engine with a real KV pool. What is proven here is the contract around it
 *     (a reference run must be a calibration, and an export refuses a run that lent) with a real
 *     Mover and real loans.
 *   - the arena-level identity against two non-empty level sets: arena_rows_ is built in start(),
 *     and this fixture has no start. The field is compared (a changed arena refuses, below); the
 *     rows themselves need a started deployment.
 *   - the table's own rules: immutability, Measured-only lookup, the depth-coverage rule R and
 *     live learning on the same bucket are tests/draft_reference_test.cpp's (with
 *     tests/draft_cost_test.cpp and tests/sched_draft_test.cpp). This file owns the engine-side
 *     admission only.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include "device/device.h"
#include "place/mover.h"
#include "place/planner.h"
#include "sched/draft_cost.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

/* Only FakeKV and FakePC: this binary runs on the real device layer, as the engine does. The
 * declaration is the scheduler fixture program the export case plays its cycles on. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "sched_program.h"

using namespace rad;

namespace {

constexpr int64_t MiB = 1024 * 1024;

/* The kernel rows the fixture program's one op is bound to, exactly as
 * tests/engine_start_fill_test.cpp builds them: the Planner only needs a well-formed program. */
const RadKernelInfo g_dev_info = { "fake_dev", "gemm", "gemm", "", "", "",
                                   RAD_DOMAIN_DEVICE, 0, nullptr, 0, nullptr, 0,
                                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
const RadKernelInfo g_host_info = { "fake_host", "gemm", "gemm", "", "", "",
                                    RAD_DOMAIN_HOST, 0, nullptr, 0, nullptr, 0,
                                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
KernelRow g_dev_row;
KernelRow g_host_row;

struct Fixture {
    Program prog;

    Fixture() {
        g_dev_row.info  = &g_dev_info;
        g_host_row.info = &g_host_info;
        prog.weights.resize(1);        /* index 0 is the sentinel */
        prog.ctx.rank = 0;
        prog.ctx.world_size = 1;
    }

    rad_weight weight(const std::string& name, int64_t bytes, int access, int32_t layer,
                      int32_t expert) {
        WeightInfo w;
        w.name = name;
        w.decl.dtype = RAD_U8;         /* one byte per element, so bytes are exact */
        w.decl.rank = 1;
        w.decl.shape[0] = bytes;
        w.decl.access = access;
        w.decl.group.layer = layer;
        w.decl.group.expert = expert;
        w.logical_bytes = bytes;
        w.align = RAD_ALIGN_UNIT;
        prog.weights.push_back(w);
        return (rad_weight)(prog.weights.size() - 1);
    }

    void op(const std::string& name, const std::vector<rad_weight>& ws) {
        OpInfo o;
        o.op = name;
        o.index = (int32_t)prog.ops.size();
        o.weights = ws;
        Band b;
        b.hi = 1 << 20;
        b.dom[RAD_DOMAIN_DEVICE].row = &g_dev_row;
        b.miss[RAD_DOMAIN_HOST] = "no host kernel for this geometry";
        o.bands.push_back(b);
        for (rad_weight w : ws) {
            WeightInfo& wi = prog.weights[w];
            if (wi.first_use_op < 0) wi.first_use_op = o.index;
            wi.last_use_op = o.index;
        }
        prog.ops.push_back(o);
    }
};

/* One MoE layer: `n` experts of `each` bytes, plus one dense model-level weight. */
void build_moe(Fixture& f, int n, int64_t each) {
    const rad_weight m = f.weight("token_embd.weight", 1 * MiB, RAD_ACCESS_VOCAB, -1, -1);
    f.op("embed", { m });
    for (int e = 0; e < n; ++e) {
        const rad_weight w = f.weight("blk.0.ffn_gate_exps." + std::to_string(e) + ".weight", each,
                                      RAD_ACCESS_CONDITIONAL, 0, e);
        f.op("moe_gemm", { w });
    }
}

/* The memory a lender loans, as tests/engine_start_fill_test.cpp builds it: `strips` stretches of
 * device memory, each paying `unit_tokens` tokens per `unit_bytes` of slab. The same strips serve
 * both lenders here -- they are memory ranges, and each lender carves its own slots out of them. */
struct Loan {
    static constexpr int64_t kStrip = 4 * MiB;
    static constexpr int64_t kUnitBytes = 64 * 1024;
    static constexpr int64_t kUnitTokens = 16;
    void* mem = nullptr;
    std::vector<LoanStrip> strips;
    explicit Loan(int n_strips) {
        mem = rad_dev_alloc(n_strips * kStrip, RAD_MEM_DEVICE);
        for (int k = 0; k < n_strips; ++k) {
            LoanStrip st;
            st.top = (char*)mem + (int64_t)(k + 1) * kStrip;
            st.bytes = kStrip;
            st.unit_bytes = kUnitBytes;
            st.unit_tokens = kUnitTokens;
            strips.push_back(st);
        }
    }
    ~Loan() { if (mem) rad_dev_free(mem, RAD_MEM_DEVICE); }
    int64_t all() const { return (int64_t)strips.size() * kStrip / kUnitBytes * kUnitTokens; }
};

/* Which lender a rank registers. A reference run registers neither; a serving run registers the
 * KV lender at start and takes its loan; the arena lender is the second one the reference must
 * also leave unregistered. */
enum class Lend { None, Kv, Arena };

/* The planner's own deployment (place_fill_episode_test.cpp's shape) plus the loan and the compute
 * stream the rank's mover runs on. One Rig builds one engine: it owns the stream the rank binds. */
struct Rig {
    Fixture   f;
    Plan      plan;
    Config    cfg;
    Loan      loan;
    RadStream compute = nullptr;

    Rig(int n_expert, int strips, int turnover) : loan(strips), turnover_(turnover) {
        build_moe(f, n_expert, 1 * MiB);
        cfg.placement = "expert_tiered";
        cfg.vram_weights_mib = 8;
        cfg.host_pool_mib = 64;
        cfg.draft_policy = DraftPolicyKind::Cost;
        cfg.n_spec = 6;
        PlannerInput in{ &f.prog, &cfg, 0, 1, nullptr, 0 };
        Planner(in).plan(&plan);
    }
    ~Rig() {
        if (compute) { rad_stream_sync(compute); rad_stream_destroy(compute); }
    }

    int build(Engine& e, Lend lend) {
        RAD_TRY(rad_stream_create(&compute, 0));
        const std::vector<LoanStrip> kv = lend == Lend::Kv ? loan.strips : std::vector<LoanStrip>{};
        RAD_TRY(EngineTestPeer::inject_plan_rank(e, f.prog, plan, turnover_, kv, compute));
        if (lend == Lend::Arena)
            RAD_TRY(EngineTestPeer::mover_of(e, 0).lend_from(Mover::kLendArena, loan.strips));
        return RAD_OK;
    }

private:
    int turnover_ = 0;
};

/* begin_draft_calibration's head/ceiling requirement: the profile names its rounds per row, which
 * only the serial head has. */
RadDrafterDecl serial_head(int depth) {
    RadDrafterDecl d{};
    d.kind  = RAD_DRAFT_SERIAL;
    d.depth = depth;
    return d;
}

/* This fixture loads no checkpoint and no plugin, so the identity's model name and kernel set are
 * empty. A profile file must name both, and the engine takes them from what it loaded -- so the
 * fixture names them here. Every field this file is about (basis, arena, live_loan, placement,
 * execution) is what Engine::draft_cost_runtime derived; nothing here changes those. */
void name_the_deployment(DraftCostRuntime* rt) {
    if (rt->fingerprint.model.empty()) rt->fingerprint.model = "fixture";
    if (rt->fingerprint.kernels.empty()) rt->fingerprint.kernels = "none";
}

/* One usable bucket, in the reader's shape (tests/engine_profile_test.cpp's clean cycle). It gives
 * the loaded table something a run could price, so reference() is reached with a profile that is
 * not structurally empty. */
const char* kBucket = R"({
  "ctx_class": 0,
  "shape": { "n_rows": 8, "verify_k": 3, "verify_tokens": 32, "history_tokens": 0,
             "prefill_tokens": 0, "pipeline_depth": 0,
             "rows": [0, 8, 8, 8], "draft_k": 3, "draft_rows": [0, 8, 8, 8],
             "origins": ["head", "head", "head", "head", "head", "head", "head", "none"],
             "sampling": "greedy", "replay": "play", "drafter": "serial" },
  "extent": { "rounds": 3, "draft_rows_total": 24, "draft_round_rows": [0, 8, 8, 8] },
  "time_ms": { "mean": 43.0, "unc": 0.6, "n": 5 },
  "components_ms": { "target": 25.9, "draft": 12.1, "host": 4.2, "valid": true },
  "flags": ["ahead"]
})";

/* An admission file: the profile a calibration of `rt` *would* write, laid out in the reader's
 * shape and pointed at the runtime identity `rt` derived. Building it here rather than writing it
 * from a calibration is what makes every case below an admission test -- the bytes are the test's,
 * only the identity is the engine's. `arena`/`placement` are the reader's view of a different
 * deployment: a changed one stands for a plan or level set this run does not have. */
std::string admission_profile_file(const DraftCostRuntime& rt, const std::string& leaf,
                         const std::string& arena = std::string(),
                         const std::string& placement = std::string()) {
    nlohmann::json fp;
    fp["model"]     = rt.fingerprint.model;
    fp["quant"]     = rt.fingerprint.quant;
    fp["tp"]        = rt.fingerprint.tp;
    fp["tp_wire"]   = rt.fingerprint.tp_wire;
    fp["gpus"]      = rt.fingerprint.gpus;
    fp["kernels"]   = rt.fingerprint.kernels;
    fp["kv_format"] = rt.fingerprint.kv_format;
    fp["placement"] = placement.empty() ? rt.fingerprint.placement : placement;
    fp["arena"]     = arena.empty() ? rt.fingerprint.arena : arena;
    fp["basis"]     = rt.fingerprint.basis == DraftCostBasis::NoLoan ? "no-loan" : "live";
    if (rt.fingerprint.basis == DraftCostBasis::Live)
        fp["live_loan"] = rt.fingerprint.live_loan;

    nlohmann::json j;
    j["format"]      = "rad-draft-profile/4";
    j["fingerprint"] = std::move(fp);
    j["execution"] =
        rt.execution == DraftCostExecution::Shrinking ? "shrinking" : "rectangular";
    j["buckets"] = nlohmann::json::array({ nlohmann::json::parse(kBucket) });

    const std::string path = radtest::tmp_path(leaf);
    std::ofstream out(path, std::ios::binary);
    out << j.dump(2);
    out.close();
    {
        std::ifstream check(path, std::ios::binary);
        CHECK(check.good());
    }
    return path;
}

}  /* namespace */

/* --draft-reference says what the run is for, and the flag alone is not enough. A serving run
 * keeps its loans dynamic and cannot measure a reference, so the flag without an output path is
 * refused; and the reference prices the cost planner's plans, so a legacy fixed-window calibration
 * is refused rather than writing a file for a plan the policy never asks for. Both refusals run
 * before the identity is derived, so a fixture with no model reaches them. */
TEST(a_reference_calibration_needs_an_output_path_and_the_cost_policy) {
    Config no_out;
    no_out.draft_reference = true;
    no_out.draft_policy    = DraftPolicyKind::Cost;
    Engine a(no_out);
    CHECK_EQ(EngineTestPeer::begin_calibration(a), RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(a).draft_costs().active());

    /* The default policy is not the cost one: the reference would measure the wrong plan. */
    Config fixed;
    fixed.draft_reference   = true;
    fixed.draft_profile_out = radtest::tmp_path("engine_reference_fixed.json");
    Engine b(fixed);
    CHECK_EQ(EngineTestPeer::begin_calibration(b), RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(b).draft_costs().active());

    /* With both satisfied the flag passes this gate; the refusal is then the deployment's, exactly
     * as for any calibration with no resolved plan. */
    Config cost;
    cost.draft_reference   = true;
    cost.draft_profile_out = radtest::tmp_path("engine_reference_noplan.json");
    cost.draft_policy      = DraftPolicyKind::Cost;
    cost.n_spec            = 6;
    Engine c(cost);
    EngineTestPeer::drafter(c, serial_head(6));
    CHECK_EQ(EngineTestPeer::begin_calibration(c), RAD_E_STATE);
}

/* --draft-calib-k forces the depths of a measurement, so the engine refuses it outside the
 * reference calibration -- before anything starts, and whether or not a plan resolves, so a
 * serving run cannot carry it. The refusals run before the identity is derived, as above. */
TEST(the_forced_calibration_sequence_is_refused_outside_a_reference_calibration) {
    /* A serving run: no output path, so no calibration at all. */
    Config serving;
    serving.draft_calib_k[0] = 3;
    serving.draft_calib_k[1] = 2;
    serving.draft_calib_n    = 2;
    Engine a(serving);
    CHECK_EQ(EngineTestPeer::begin_calibration(a), RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(a).draft_costs().active());

    /* An output path alone is not enough: the sequence belongs to the reference run that measures
     * the ladder, so a plain cost calibration is refused. */
    Config no_reference;
    no_reference.draft_calib_k[0]  = 3;
    no_reference.draft_calib_k[1]  = 2;
    no_reference.draft_calib_n     = 2;
    no_reference.draft_profile_out = radtest::tmp_path("engine_calib_noref.json");
    no_reference.draft_policy      = DraftPolicyKind::Cost;
    Engine b(no_reference);
    CHECK_EQ(EngineTestPeer::begin_calibration(b), RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(b).draft_costs().active());

    /* With both satisfied the flag passes this gate; the refusal is then the deployment's, exactly
     * as for any calibration with no resolved plan. */
    Config calib;
    calib.draft_calib_k[0]  = 3;
    calib.draft_calib_k[1]  = 2;
    calib.draft_calib_n     = 2;
    calib.draft_profile_out = radtest::tmp_path("engine_calib_noplan.json");
    calib.draft_reference   = true;
    calib.draft_policy      = DraftPolicyKind::Cost;
    calib.n_spec            = 6;
    Engine c(calib);
    EngineTestPeer::drafter(c, serial_head(6));
    CHECK_EQ(EngineTestPeer::begin_calibration(c), RAD_E_STATE);
}

/* The reference run's ranks. Both lenders stay off -- no strip is registered, no loan is offered
 * and nothing is filled -- while the run still arms its calibration, and the table it records into
 * is the writer, not a loaded reference. The identity says so: basis no-loan, no live_loan, and
 * the arena levels are the (empty here) level set, never the loan. */
TEST(a_reference_calibration_registers_no_lender_and_records_into_a_mutable_table) {
    Rig r(/*n_expert*/ 24, /*strips*/ 8, /*turnover*/ 2);
    Config cfg = r.cfg;
    cfg.draft_reference   = true;
    cfg.draft_profile_out = radtest::tmp_path("engine_reference_armed.json");
    Engine e(cfg);
    EngineTestPeer::drafter(e, serial_head(6));
    REQUIRE_EQ(r.build(e, Lend::None), RAD_OK);

    Mover& m = EngineTestPeer::mover_of(e, 0);
    CHECK_EQ(m.flex_capacity(), 0);
    CHECK_EQ(m.flex_offered_bytes(), 0);
    CHECK_EQ(m.flex_used_bytes(), 0);
    CHECK_EQ(m.loan(Mover::kLendKV), 0);
    CHECK_EQ(m.loan(Mover::kLendArena), 0);
    CHECK_EQ(m.loan_held(Mover::kLendKV), 0);
    CHECK_EQ(m.loan_held(Mover::kLendArena), 0);

    REQUIRE_EQ(EngineTestPeer::begin_calibration(e), RAD_OK);
    CHECK(EngineTestPeer::calibrating(e));
    const DraftCostTable& t = EngineTestPeer::sched(e).draft_costs();
    CHECK(t.active());
    /* The calibration's table carries the file's no-loan basis but is the writer: a reference
     * table records nothing, so it must not be flagged as one (DraftCostRuntime::recording). */
    CHECK(!t.reference());
    CHECK(t.records_only());
    CHECK_EQ((int)t.reference_skipped(), 0);

    DraftCostRuntime rt;
    REQUIRE_EQ(EngineTestPeer::runtime(e, &rt), RAD_OK);
    CHECK(rt.fingerprint.basis == DraftCostBasis::NoLoan);
    CHECK(rt.fingerprint.live_loan.empty());
    CHECK_EQ(rt.fingerprint.arena, std::string("none"));
}

/* The engine's enforceable half of "both lenders off": a reference file is only written by a run
 * that lent nothing. The guard reads the mover's real bookkeeping -- carved slots, bytes in use and
 * how far either lender's memory the slab still reaches -- not the target it was handed, so a rank
 * that registered a lender is refused whatever it was asked to hold. A run that lent is not a
 * reference, and the file is never written. */
TEST(a_reference_export_refuses_a_run_that_lent) {
    {
        Rig r(24, 8, 2);
        Config cfg = r.cfg;
        cfg.draft_reference   = true;
        cfg.draft_profile_out = radtest::tmp_path("engine_reference_kv_loan.json");
        Engine e(cfg);
        EngineTestPeer::drafter(e, serial_head(6));
        REQUIRE_EQ(r.build(e, Lend::Kv), RAD_OK);
        REQUIRE(EngineTestPeer::mover_of(e, 0).flex_capacity() > 0);
        REQUIRE_EQ(EngineTestPeer::begin_calibration(e), RAD_OK);

        CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_STATE);
        CHECK(!std::ifstream(cfg.draft_profile_out).good());
    }
    {
        /* The second lender alone is enough: the guard is not the KV lender's. */
        Rig r(24, 8, 2);
        Config cfg = r.cfg;
        cfg.draft_reference   = true;
        cfg.draft_profile_out = radtest::tmp_path("engine_reference_arena_loan.json");
        Engine e(cfg);
        EngineTestPeer::drafter(e, serial_head(6));
        REQUIRE_EQ(r.build(e, Lend::Arena), RAD_OK);
        REQUIRE(EngineTestPeer::mover_of(e, 0).flex_capacity() > 0);
        REQUIRE_EQ(EngineTestPeer::begin_calibration(e), RAD_OK);

        CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_STATE);
        CHECK(!std::ifstream(cfg.draft_profile_out).good());
    }
    {
        /* A live calibration of the same deployment is not held to the reference rule: the same
         * lender and the same empty table reach the writer's own refusal (nothing measured), not
         * the lender guard. */
        Rig r(24, 8, 2);
        Config cfg = r.cfg;
        cfg.draft_profile_out = radtest::tmp_path("engine_live_calib_loan.json");
        Engine e(cfg);
        EngineTestPeer::drafter(e, serial_head(6));
        REQUIRE_EQ(r.build(e, Lend::Kv), RAD_OK);
        REQUIRE(EngineTestPeer::mover_of(e, 0).flex_capacity() > 0);
        REQUIRE_EQ(EngineTestPeer::begin_calibration(e), RAD_OK);

        CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_INVAL);
        CHECK(!std::ifstream(cfg.draft_profile_out).good());
    }
}

/* Serving keeps its loan dynamic and the reference is still admitted. This is the admission case,
 * not a dynamic-loan proof: the run takes and holds a real loan, its identity is what the engine
 * derived from it, and the reference file is the hand-built one. The two identities differ in
 * exactly the fields the reference rule ignores (basis, live_loan) and agree on the ones it
 * compares -- so the file loads, is admitted and is flagged as a reference. What the rule does not
 * ignore still refuses: a changed arena-level identity and a changed plan are placement changes,
 * not loan moves. */
TEST(an_admission_reference_loads_on_a_serving_run_that_lends) {
    Rig cal(24, 8, 2);
    Config cal_cfg = cal.cfg;
    cal_cfg.draft_reference = true;
    Engine ce(cal_cfg);
    EngineTestPeer::drafter(ce, serial_head(6));
    REQUIRE_EQ(cal.build(ce, Lend::None), RAD_OK);
    DraftCostRuntime ref_rt;
    REQUIRE_EQ(EngineTestPeer::runtime(ce, &ref_rt), RAD_OK);
    name_the_deployment(&ref_rt);
    REQUIRE(ref_rt.fingerprint.basis == DraftCostBasis::NoLoan);
    REQUIRE(ref_rt.fingerprint.live_loan.empty());

    const std::string path = admission_profile_file(ref_rt, "engine_reference_serving.json");

    Rig srv(24, 8, 2);
    Config srv_cfg = srv.cfg;
    srv_cfg.draft_profile = path;
    Engine se(srv_cfg);
    EngineTestPeer::drafter(se, serial_head(6));
    REQUIRE_EQ(srv.build(se, Lend::Kv), RAD_OK);
    Mover& m = EngineTestPeer::mover_of(se, 0);
    REQUIRE(m.flex_capacity() > 0);
    REQUIRE_EQ(m.set_loan(Mover::kLendKV, srv.loan.all()), RAD_OK);
    REQUIRE(m.flex_offered_bytes() > 0);

    DraftCostRuntime live_rt;
    REQUIRE_EQ(EngineTestPeer::runtime(se, &live_rt), RAD_OK);
    name_the_deployment(&live_rt);
    CHECK(live_rt.fingerprint.basis == DraftCostBasis::Live);
    REQUIRE(live_rt.fingerprint.live_loan != ref_rt.fingerprint.live_loan);

    CHECK_EQ(EngineTestPeer::load_profile(se, live_rt), RAD_OK);
    const DraftCostTable& t = EngineTestPeer::sched(se).draft_costs();
    CHECK(t.active());
    CHECK(t.reference());
    CHECK_EQ((int)t.buckets().size(), 1);
    CHECK(t.has_usable_bucket());

    /* A different arena-level set is a different activation geometry, and a reference may not be
     * rescaled across it. */
    DraftCostRuntime other_arena = live_rt;
    other_arena.fingerprint.arena = ref_rt.fingerprint.arena + ",128";
    CHECK_EQ(EngineTestPeer::load_profile(se, other_arena), RAD_OK);
    CHECK(!t.active());
    CHECK(t.reason().find("arena") != std::string::npos);

    /* The plan too: the reference is immutable, not indifferent to where its numbers were
     * measured. */
    DraftCostRuntime other_plan = live_rt;
    other_plan.fingerprint.placement += ";x";
    CHECK_EQ(EngineTestPeer::load_profile(se, other_plan), RAD_OK);
    CHECK(!t.active());
    CHECK(t.reason().find("placement") != std::string::npos);
}

/* A live profile keeps the strict match it had before the reference existed: the loan a live run
 * took is part of its identity, and a file measured under another loan does not price this one.
 * Only the reference's basis is exempt from the loan comparison; here the same hand-built
 * admission file on the same plan, with a different loan, is refused by name and the table stays
 * inactive. */
TEST(an_admission_live_profile_keeps_the_strict_loan_match) {
    Rig cal(24, 8, 2);
    Engine ce(cal.cfg);
    EngineTestPeer::drafter(ce, serial_head(6));
    REQUIRE_EQ(cal.build(ce, Lend::Kv), RAD_OK);
    Mover& cm = EngineTestPeer::mover_of(ce, 0);
    /* The full carve against half of it: `all()` is the strips' whole token equivalent and covers
     * every carved slot in both runs, which would make the two loans indistinguishable. */
    const int64_t ceiling = cm.loan_ceiling(Mover::kLendKV);
    REQUIRE(ceiling > 0);
    REQUIRE_EQ(cm.set_loan(Mover::kLendKV, ceiling), RAD_OK);
    DraftCostRuntime live_rt;
    REQUIRE_EQ(EngineTestPeer::runtime(ce, &live_rt), RAD_OK);
    name_the_deployment(&live_rt);
    REQUIRE(live_rt.fingerprint.basis == DraftCostBasis::Live);
    REQUIRE(!live_rt.fingerprint.live_loan.empty());

    const std::string path = admission_profile_file(live_rt, "engine_live_profile_strict.json");

    Rig srv(24, 8, 2);
    Config srv_cfg = srv.cfg;
    srv_cfg.draft_profile = path;
    Engine se(srv_cfg);
    EngineTestPeer::drafter(se, serial_head(6));
    REQUIRE_EQ(srv.build(se, Lend::Kv), RAD_OK);
    Mover& sm = EngineTestPeer::mover_of(se, 0);
    REQUIRE_EQ(sm.set_loan(Mover::kLendKV, sm.loan_ceiling(Mover::kLendKV) / 2), RAD_OK);

    DraftCostRuntime half_rt;
    REQUIRE_EQ(EngineTestPeer::runtime(se, &half_rt), RAD_OK);
    name_the_deployment(&half_rt);
    REQUIRE(half_rt.fingerprint.live_loan != live_rt.fingerprint.live_loan);

    /* Its own identity: admitted, with buckets, and not a reference. */
    CHECK_EQ(EngineTestPeer::load_profile(se, live_rt), RAD_OK);
    const DraftCostTable& t = EngineTestPeer::sched(se).draft_costs();
    CHECK(t.active());
    CHECK(!t.reference());
    CHECK_EQ((int)t.buckets().size(), 1);

    /* The other loan: refused by name, and the table does not keep pricing the old one. */
    CHECK_EQ(EngineTestPeer::load_profile(se, half_rt), RAD_OK);
    CHECK(!t.active());
    CHECK(t.reason().find("live_loan") != std::string::npos);
}

/* The freezing half of the reference contract: --draft-reference measures the plan's start
 * placement, and a budgeted move during the run makes Scheduler::draft_drift_note mark every
 * later cycle `kDraftSamplePlacement`; the export then leaves those cycles out, which is what a
 * real calibration hit ("none of the 1 kept bucket(s) is priceable"). Engine::tiering_dispatch
 * therefore returns before its swap/fill tail under the flag. This case pins that at the seam the
 * scheduler reads -- Engine::residency_moves() -- with the fixture's real plan, heat engine and
 * Mover: a histogram primed exactly as a landed one credits heat, the dispatch runs, and the
 * mover's books stay at zero. The serving arms are the control: the same plan and heat move
 * (with and without a loan) and a serving run still lends, so the zero is the flag's freeze and
 * not a fixture that cannot move.
 *
 * The identity check is the reference rule's own: both runs derive their fingerprint from the same
 * plan, so placement and arena agree; they differ only in basis/live_loan, which the no-loan
 * reference is exempt from (core/sched/draft_cost.cpp).
 *
 * The flag `kDraftSamplePlacement` also covers a bracket whose step staged pooled experts
 * (`ev.staged`), which this freeze does not touch; the export case below keeps that exclusion as
 * its counter-check and still expects a clean bucket beside it. */
namespace {

/* A unit the heat engine can promote on this run: movable, pooled, and still in the pool after
 * the start fill (a serving run has promoted the hottest into lent slots). -1 when none is left,
 * which makes the case vacuous rather than green. */
int32_t pooled_candidate(Engine& e, const Plan& p) {
    HeatEngine& h = EngineTestPeer::heat_of(e, 0);
    for (int32_t u = 0; u < (int32_t)p.units.size(); ++u) {
        const MoveUnit& mu = p.units[(size_t)u];
        if (!mu.movable || mu.expert < 0 || mu.tier != Tier::HostPinned) continue;
        if (h.tier(u) == Tier::HostPinned) return u;
    }
    return -1;
}

/* The drain's cadence: one histogram crediting one expert heavily, as a routed pass does, then
 * the engine's tiering dispatch; the mover's stream is synced between dispatches so the copies
 * land and the next begin_dispatch retires them. */
int drive_heat(Engine& e, const Plan& p, int32_t hot, int n_expert, int steps) {
    HeatEngine& h = EngineTestPeer::heat_of(e, 0);
    Mover&      m = EngineTestPeer::mover_of(e, 0);
    std::vector<int32_t> counts((size_t)n_expert, 0);
    counts[(size_t)p.units[(size_t)hot].expert] = 500;
    for (int step = 0; step < steps; ++step) {
        h.observe(0, counts.data(), n_expert, 1.0f);
        EngineTestPeer::tiering_dispatch(e, 0, step);
        RAD_TRY(rad_stream_sync(m.transfer_stream()));
    }
    EngineTestPeer::tiering_dispatch(e, 0, steps);   /* retire the last copies */
    return rad_stream_sync(m.transfer_stream());
}

}  /* namespace */

TEST(a_reference_calibration_freezes_the_placement_bringup_settled) {
    constexpr int kExperts = 48;
    constexpr int kSteps   = 8;

    /* The reference arm is production's: no lender, so nothing is carved or filled, and heat
     * that the controls below show would move a unit on the same plan shape. */
    Rig ref(kExperts, /*strips*/ 4, /*turnover*/ 2);
    Config ref_cfg = ref.cfg;
    ref_cfg.draft_reference   = true;
    ref_cfg.draft_profile_out = radtest::tmp_path("engine_reference_freeze.json");
    Engine re(ref_cfg);
    REQUIRE_EQ(ref.build(re, Lend::None), RAD_OK);
    Mover&      rm = EngineTestPeer::mover_of(re, 0);
    HeatEngine& rh = EngineTestPeer::heat_of(re, 0);
    CHECK_EQ(rm.flex_capacity(), 0);
    CHECK_OK(EngineTestPeer::complete_start_fill(re));   /* serve's completion, as production */
    const int32_t ref_hot = pooled_candidate(re, ref.plan);
    REQUIRE(ref_hot >= 0);
    REQUIRE_EQ(drive_heat(re, ref.plan, ref_hot, kExperts, kSteps), RAD_OK);

    CHECK_EQ(rh.stats().proposed, (uint64_t)0);
    CHECK_EQ(rh.stats().promotions, (uint64_t)0);
    CHECK_EQ(rh.stats().demotions, (uint64_t)0);
    /* The count Scheduler::draft_residency receives: zero, so no bracket is marked placed. */
    CHECK_EQ(EngineTestPeer::residency_moves(re), (int64_t)0);
    /* And the plan's placement is still the one the fingerprint names. */
    for (int32_t u = 0; u < (int32_t)ref.plan.units.size(); ++u)
        CHECK_EQ((int)rh.tier(u), (int)ref.plan.units[(size_t)u].tier);

    /* Control 1: the same plan, heat and dispatch cadence on a serving run without a loan. The
     * heat engine proposes and the mover moves. */
    Rig srv(kExperts, 4, 2);
    Engine se(srv.cfg);
    REQUIRE_EQ(srv.build(se, Lend::None), RAD_OK);
    HeatEngine& sh = EngineTestPeer::heat_of(se, 0);
    CHECK_OK(EngineTestPeer::complete_start_fill(se));
    const int32_t srv_hot = pooled_candidate(se, srv.plan);
    REQUIRE(srv_hot >= 0);
    REQUIRE_EQ(drive_heat(se, srv.plan, srv_hot, kExperts, kSteps), RAD_OK);
    CHECK(sh.stats().proposed > 0);
    CHECK(EngineTestPeer::residency_moves(se) > 0);

    /* Control 2: a serving run still lends -- the KV strip fills the lent slots at the start
     * fill and the same heat moves the slab. */
    Rig live(kExperts, 4, 2);
    Engine le(live.cfg);
    REQUIRE_EQ(live.build(le, Lend::Kv), RAD_OK);
    Mover& lm = EngineTestPeer::mover_of(le, 0);
    REQUIRE(lm.flex_capacity() > 0);
    REQUIRE_EQ(lm.set_loan(Mover::kLendKV, live.loan.all()), RAD_OK);
    CHECK_OK(EngineTestPeer::complete_start_fill(le));
    CHECK(lm.flex_used_bytes() > 0);
    const int32_t live_hot = pooled_candidate(le, live.plan);
    REQUIRE(live_hot >= 0);
    REQUIRE_EQ(drive_heat(le, live.plan, live_hot, kExperts, kSteps), RAD_OK);
    CHECK(EngineTestPeer::residency_moves(le) > 0);

    /* Identity: the freeze changes no field the reference rule compares -- placement and arena
     * are the plan's in both bases; only basis/live_loan differ, and the reference is no-loan. */
    DraftCostRuntime ref_rt, live_rt;
    REQUIRE_EQ(EngineTestPeer::runtime(re, &ref_rt), RAD_OK);
    REQUIRE_EQ(EngineTestPeer::runtime(le, &live_rt), RAD_OK);
    CHECK(ref_rt.fingerprint.basis == DraftCostBasis::NoLoan);
    CHECK(ref_rt.fingerprint.live_loan.empty());
    CHECK(live_rt.fingerprint.basis == DraftCostBasis::Live);
    CHECK(!live_rt.fingerprint.live_loan.empty());
    CHECK_EQ(ref_rt.fingerprint.placement, live_rt.fingerprint.placement);
    CHECK_EQ(ref_rt.fingerprint.arena, live_rt.fingerprint.arena);
}

/* The scheduler's declared program (tests/sched_program.h), owned beside the engine: the export
 * reads the table the engine's own scheduler records into, so that scheduler has to run a real
 * program while rank 0 keeps the Planner's plan and the Mover this file's cases use. */
struct SchedProgram {
    Program            prog;
    Config             cfg;
    sched_fake::FakeKV kv;
    sched_fake::FakePC pc;
    std::vector<void*> owned;
    ~SchedProgram() { for (void* p : owned) std::free(p); }
};

/* One calibration run, played cycle by cycle on the declared fixture program: rank 0 is the real
 * plan, heat engine and mover, so the histogram a serving run moves on is running; the scheduler is
 * the engine's own and the export is serve()'s shutdown call. `reference` is --draft-reference:
 * with it the placement is frozen, without it this is the serving arm the freeze is measured
 * against. */
struct PlayedExport {
    static Config cfg_for(const Config& base, bool reference, const std::string& out) {
        Config c = base;
        c.draft_reference   = reference;
        c.draft_profile_out = out;
        return c;
    }

    int              experts;
    Rig              rig;
    Config           cfg;
    Engine           eng;
    SchedProgram     prog;
    DraftCostRuntime rt;
    int64_t          moves     = 0;       /* residency_moves() over the played cycles */
    int              export_rc = RAD_OK;
    int              played    = 0;       /* clean played cycles the exported file kept */
    int              staged_samples = 0;  /* samples the table kept with kDraftSamplePlacement */
    bool             inited    = false;

    PlayedExport(int n_experts, bool reference)
        : experts(n_experts),
          rig(n_experts, /*strips*/ 4, /*turnover*/ 2),
          cfg(cfg_for(rig.cfg, reference,
                      radtest::tmp_path(reference ? "engine_reference_played_ref.json"
                                                  : "engine_reference_played_live.json"))),
          eng(cfg) {}

    ~PlayedExport() { if (inited) EngineTestPeer::fini_sched(eng); }

    /* Decode buckets with no disturbing flag, by the samples they hold. */
    static int clean_played(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.good()) return 0;
        nlohmann::json j;
        in >> j;
        int n = 0;
        for (const auto& b : j["buckets"]) {
            if (b["shape"]["prefill_tokens"].get<int>() != 0) continue;
            if (!b["flags"].empty()) continue;
            n += b["time_ms"]["n"].get<int>();
        }
        return n;
    }

    int run(int cycles) {
        EngineTestPeer::drafter(eng, serial_head(6));
        RAD_TRY(rig.build(eng, Lend::None));
        /* serve()'s completion of the start fill, before the first bracket: the placement the
         * reference measures is the one bringup settled. */
        RAD_TRY(EngineTestPeer::complete_start_fill(eng));
        /* serve()'s startup call: it derives the identity from the resolved plan and arms the
         * calibration's recording table (record_only, the fallback baseline admitted). */
        RAD_TRY(EngineTestPeer::begin_calibration(eng));
        RAD_TRY(EngineTestPeer::runtime(eng, &rt));
        /* The two names this fixture has no checkpoint and no plugin to give (name_the_deployment);
         * every other identity field is the one the runtime derived. */
        name_the_deployment(&rt);
        EngineTestPeer::set_calibrating(eng, rt);

        sched_fake::build_program(prog.prog, prog.cfg, prog.kv, prog.pc, prog.owned,
                                  /*blocks*/ 4096, /*states*/ 64, /*max_tok*/ 64, /*max_ctx*/ 4096);
        prog.cfg.n_spec            = 6;
        prog.cfg.draft_policy      = DraftPolicyKind::Cost;
        prog.cfg.draft_profile_out = cfg.draft_profile_out;
        prog.prog.drafter.kind     = RAD_DRAFT_SERIAL;
        prog.prog.drafter.depth    = 6;
        prog.prog.drafter.name     = "mtp-test";
        RAD_TRY(EngineTestPeer::init_sched(eng, prog.prog, prog.cfg, &prog.kv, &prog.pc, rig.compute));
        inited = true;

        Scheduler& s = EngineTestPeer::sched(eng);
        auto req = std::make_unique<Request>();
        req->id         = 1;
        req->prompt.resize(160);
        for (size_t i = 0; i < req->prompt.size(); ++i) req->prompt[i] = (int32_t)(1000 + i);
        req->max_tokens = 64;
        req->ignore_eos = true;
        req->sp.temp    = 0.0f;
        RAD_TRY(EngineTestPeer::submit(eng, std::move(req)));

        /* The prompt: chunks only, as the scheduler plays them (no decode row yet, so no ring
         * record opens -- draft_fill_view has nothing to name). */
        for (int guard = 0; guard < 16; ++guard) {
            Request* q = s.find(1);
            if (!q) return RAD_E_STATE;
            if (q->n_computed >= (int32_t)q->prompt.size()) break;
            if (!s.step()) return RAD_E_STATE;
            RAD_TRY(EngineTestPeer::commit(eng, 11));
        }
        Request* q = s.find(1);
        if (!q || q->n_computed < (int32_t)q->prompt.size()) return RAD_E_STATE;

        /* The played cycles, in the order the step loop plays them: plan, commit, publish the
         * head's rounds, the rank's heat and tiering dispatch, then the bracket's evidence. The
         * histogram is the one the freeze case shows would promote a pooled unit on an unfrozen
         * mover. The last bracket is the stager's instead: the rank reports that the step armed
         * pooled experts for a prefill chunk, which the freeze does not touch and the export must
         * still leave out. */
        HeatEngine& heat  = EngineTestPeer::heat_of(eng, 0);
        Mover&      mover = EngineTestPeer::mover_of(eng, 0);
        /* The unit this plan lets the heat move: pooled after the start fill, as in the freeze
         * case's controls. Its expert is the one a landed histogram credits. */
        const int32_t hot = pooled_candidate(eng, rig.plan);
        if (hot < 0) return RAD_E_STATE;
        std::vector<int32_t> counts((size_t)experts, 0);
        counts[(size_t)rig.plan.units[(size_t)hot].expert] = 500;
        int32_t six[6] = { 21, 22, 23, 24, 25, 26 };
        const int64_t base = EngineTestPeer::residency_moves(eng);
        for (int step = 0; step <= cycles; ++step) {
            if (!s.step()) return RAD_E_STATE;
            const DraftPlan* dp = s.draft().plan();
            if (!dp || dp->n_seq != 1) return RAD_E_STATE;
            const int k_row = dp->k[0];
            RAD_TRY(EngineTestPeer::commit(eng, 11));
            RAD_TRY(s.set_draft(1, six, k_row));

            heat.observe(0, counts.data(), experts, 1.0f);
            EngineTestPeer::tiering_dispatch(eng, 0, step);
            RAD_TRY(rad_stream_sync(mover.transfer_stream()));

            /* The rank's own facts for this bracket, as step_evidence freezes them on a host
             * backend: a pass that established no tape, nothing staged -- and the cumulative move
             * count the engine actually derives, which is what Scheduler::step_finished marks on. */
            DraftStepEvidence ev;
            ev.replay_known     = true;
            ev.replay           = DraftReplay::None;
            ev.moves_since_bind = EngineTestPeer::residency_moves(eng) - base;
            ev.staged           = (step == cycles);
            s.step_finished(ev);
        }
        moves = EngineTestPeer::residency_moves(eng) - base;

        /* serve()'s shutdown call: it flushes the ring (the last cycle is still open, exactly as a
         * run that ends here would leave it) and writes the file. */
        export_rc = EngineTestPeer::export_calibration(eng);
        played    = clean_played(cfg.draft_profile_out);

        /* The staged bracket is not folded into a clean one: once the flush has closed it, the
         * table holds it with the placement flag, and the export has dropped it from the file
         * while the played cycles stayed. */
        for (const auto& b : s.draft_costs().buckets())
            if (b.flags & kDraftSamplePlacement) staged_samples += b.n;
        return RAD_OK;
    }
};

TEST(a_reference_calibration_exports_the_cycles_the_run_played) {
    constexpr int kExperts = 48;   /* the freeze case's plan: a pooled unit the heat promotes */
    constexpr int kCycles  = 10;   /* clean played cycles; one staged bracket follows them */

    {
        PlayedExport ref(kExperts, /*reference*/ true);
        REQUIRE_EQ(ref.run(kCycles), RAD_OK);
        REQUIRE_EQ(ref.export_rc, RAD_OK);
        CHECK_EQ(ref.moves, (int64_t)0);   /* the freeze: the placement never moved */
        CHECK_EQ(ref.played, kCycles);
        /* The stager's bracket is recorded with the flag and left out of the file: the freeze
         * narrows which cycles are marked, it does not make every played cycle exportable. */
        CHECK_EQ(ref.staged_samples, 1);
        std::ifstream in(ref.cfg.draft_profile_out, std::ios::binary);
        REQUIRE(in.good());
        nlohmann::json j;
        in >> j;
        CHECK_EQ(j["format"].get<std::string>(),
                 std::string("rad-draft-profile/") + std::to_string(RAD_DRAFT_PROFILE_VERSION));

        /* The file's identity is the armed runtime's: the plan's placement, the level set, no
         * loan. The freeze changes none of it. */
        const auto& fp = j["fingerprint"];
        CHECK_EQ(fp["basis"].get<std::string>(), std::string("no-loan"));
        CHECK(!fp.contains("live_loan"));
        CHECK_EQ(fp["model"].get<std::string>(), std::string("fixture"));
        CHECK_EQ(fp["placement"].get<std::string>(), ref.rt.fingerprint.placement);
        CHECK_EQ(fp["arena"].get<std::string>(), ref.rt.fingerprint.arena);
        CHECK_EQ(fp["arena"].get<std::string>(), std::string("none"));

        /* Every clean played decode cycle is in the file, none carrying a placement flag: the
         * samples fold by shape (the first cycle verified nothing, the following ones the window
         * the cycle before published). The staged bracket is not here -- the table's copy is the
         * CHECK above. */
        for (const auto& b : j["buckets"]) {
            if (b["shape"]["prefill_tokens"].get<int>() != 0) continue;
            CHECK(b["flags"].empty());
            CHECK_EQ(b["shape"]["n_rows"].get<int>(), 1);
            CHECK_EQ(b["extent"]["rounds"].get<int>(), 6);
        }
    }

    /* The same cycles serving: the heat does move the placement there, and the export drops the
     * brackets it disturbed -- which is what makes the case above a regression and not a
     * tautology. The export still has to succeed with its one clean cycle; an export that refused
     * or wrote nothing would otherwise leave both checks green at played == 0. */
    {
        PlayedExport srv(kExperts, /*reference*/ false);
        REQUIRE_EQ(srv.run(kCycles), RAD_OK);
        CHECK(srv.moves > 0);
        REQUIRE_EQ(srv.export_rc, RAD_OK);
        CHECK_EQ(srv.played, 1);
    }
}

RAD_TEST_MAIN()
