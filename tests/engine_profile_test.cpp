/* engine_profile_test.cpp -- --draft-profile at its production caller.
 *
 * Engine::load_draft_cost_profile is the read serve() performs immediately before Scheduler::init;
 * these cases call that function through the peer rather than through serve(), because a
 * default-constructed engine has no ranks and serve() refuses before reaching it. The runtime is
 * passed in (the peer's whole point here): it is what serve() derives, and handing it over lets a
 * fixture reach the mismatch and admission branches without a model, a plugin or a device.
 *
 * Not covered here, and not to be read as covered:
 *   - the derivation itself (Engine::draft_cost_runtime) needs ranks with a resolved plan. It has
 *     no test of its own in this file: with no ranks it refuses by design, and the plan/residency
 *     formatting is exercised only by a real start.
 *   - the Cost lock that turns an inactive table into a refused start lives in Scheduler::init
 *     (tests/sched_draft_test.cpp).
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include <fstream>
#include <string>
#include <utility>

namespace {

using rad::Config;
using rad::DraftCostRuntime;
using rad::Engine;
using rad::EngineTestPeer;

/* A profile whose fingerprint is the runtime below, with no buckets: enough to reach configure()
 * and the admission branch without repeating the bucket fixture from draft_cost_test. */
const char* kProfile = R"({
  "format": "rad-draft-profile/4",
  "fingerprint": {
    "model": "tiny", "quant": "bf16", "tp": 1, "tp_wire": "n/a",
    "gpus": "1x host dev=0 cu=1 vram=0", "kernels": "none", "kv_format": "bf16",
    "placement": "all_vram vram=0 host=0 file=0 dyn=1 res=0000000000000000",
    "basis": "live", "arena": "none", "live_loan": "flex=0 lend=0"
  },
  "execution": "rectangular",
  "buckets": []
})";

DraftCostRuntime matching_runtime() {
    DraftCostRuntime r;
    r.fingerprint.model     = "tiny";
    r.fingerprint.quant     = "bf16";
    r.fingerprint.tp        = 1;
    r.fingerprint.tp_wire   = "n/a";
    r.fingerprint.gpus      = "1x host dev=0 cu=1 vram=0";
    r.fingerprint.kernels   = "none";
    r.fingerprint.kv_format = "bf16";
    r.fingerprint.placement = "all_vram vram=0 host=0 file=0 dyn=1 res=0000000000000000";
    r.fingerprint.basis     = rad::DraftCostBasis::Live;
    r.fingerprint.arena     = "none";
    r.fingerprint.live_loan = "flex=0 lend=0";
    return r;
}

std::string write_case_file(const std::string& leaf, const std::string& text) {
    const std::string path = radtest::tmp_path(leaf);
    std::ofstream f(path, std::ios::binary);
    f << text;
    return path;
}

TEST(no_profile_path_is_not_an_error) {
    Engine e(Config{});
    const int rc = EngineTestPeer::load_profile(e, matching_runtime());
    CHECK_EQ(rc, RAD_OK);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
}

/* A path that cannot be read stops the start: serve() returns this error before Scheduler::init,
 * so a cost run never comes up with the table silently unconfigured. */
TEST(an_unreadable_profile_fails_the_read) {
    Config cfg;
    cfg.draft_profile = radtest::tmp_path("engine_profile_missing.json");
    Engine e(std::move(cfg));
    const int rc = EngineTestPeer::load_profile(e, matching_runtime());
    CHECK(rc < 0);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
}

TEST(a_malformed_profile_is_an_invalid_argument) {
    Config cfg;
    cfg.draft_profile = write_case_file("engine_profile_bad.json", "not a profile");
    Engine e(std::move(cfg));
    const int rc = EngineTestPeer::load_profile(e, matching_runtime());
    CHECK_EQ(rc, RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
}

/* A deployment mismatch is not a read error -- the file is fine, this machine is not the one it
 * was measured on -- so the table stays inactive and names the field. Under Cost the scheduler
 * refuses the start on exactly that state. */
TEST(a_fingerprint_mismatch_leaves_the_table_inactive_and_names_the_field) {
    Config cfg;
    cfg.draft_profile = write_case_file("engine_profile_other.json", kProfile);
    Engine e(std::move(cfg));
    DraftCostRuntime r = matching_runtime();
    r.fingerprint.placement = "expert_tiered vram=1 host=2 file=0 dyn=1 res=0000000000000001";
    const int rc = EngineTestPeer::load_profile(e, r);
    CHECK_EQ(rc, RAD_OK);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
    CHECK(EngineTestPeer::sched(e).draft_costs().reason().find("placement") != std::string::npos);
}

TEST(a_matching_profile_is_admitted_before_the_scheduler_starts) {
    Config cfg;
    cfg.draft_profile = write_case_file("engine_profile_ok.json", kProfile);
    Engine e(std::move(cfg));
    const int rc = EngineTestPeer::load_profile(e, matching_runtime());
    CHECK_EQ(rc, RAD_OK);
    CHECK(EngineTestPeer::sched(e).draft_costs().active());
}

/* --draft-profile-out's startup checks. Every refusal below runs before draft_cost_runtime, so a
 * host fixture reaches it; a run that cannot export must say so at startup, not after serving. */
RadDrafterDecl serial_head(int depth) {
    RadDrafterDecl d{};
    d.kind  = RAD_DRAFT_SERIAL;
    d.depth = depth;
    return d;
}

TEST(a_calibration_refuses_both_profile_flags_at_once) {
    Config cfg;
    cfg.draft_profile     = write_case_file("engine_calib_both_read.json", kProfile);
    cfg.draft_profile_out = radtest::tmp_path("engine_calib_both_out.json");
    Engine e(std::move(cfg));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
}

/* Cost calibrates like legacy does -- the policy is what makes a start cycle measurable, not a
 * reason to refuse the file. With no rank planned the identity cannot be derived, so the refusal
 * comes from the deployment, exactly as for the legacy case above. */
TEST(a_calibration_under_cost_reaches_the_identity_derivation) {
    Config cfg;
    cfg.draft_profile_out = radtest::tmp_path("engine_calib_cost.json");
    cfg.draft_policy      = rad::DraftPolicyKind::Cost;
    cfg.n_spec            = 6;
    Engine e(std::move(cfg));
    EngineTestPeer::drafter(e, serial_head(6));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_STATE);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
}

TEST(a_calibration_refuses_the_kl_mode) {
    Config cfg;
    cfg.draft_profile_out = radtest::tmp_path("engine_calib_kld.json");
    cfg.kld_record        = radtest::tmp_path("engine_calib_kld_dir");
    Engine e(std::move(cfg));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_INVAL);
}

/* The profile names its rounds per row, which only the serial head has: a block or n-gram drafter,
 * or a serial head with no depth, would write a file that misnames what produced it. */
TEST(a_calibration_refuses_a_run_without_a_serial_head) {
    Config cfg;
    cfg.draft_profile_out = radtest::tmp_path("engine_calib_nohead.json");
    cfg.n_spec            = 6;
    Engine e(std::move(cfg));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_INVAL);

    EngineTestPeer::drafter(e, serial_head(0));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_INVAL);

    /* A head and a ceiling pass that check and reach the identity: with no rank planned there is
     * no deployment to bind the profile to, and the refusal happens here, before the scheduler
     * starts -- not at export after a run that could never have written a valid header. */
    EngineTestPeer::drafter(e, serial_head(4));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_STATE);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());

    Config zero;
    zero.draft_profile_out = radtest::tmp_path("engine_calib_nospec.json");
    zero.n_spec            = 0;
    Engine z(std::move(zero));
    EngineTestPeer::drafter(z, serial_head(4));
    CHECK_EQ(EngineTestPeer::begin_calibration(z), RAD_E_INVAL);
}

TEST(a_calibration_refuses_an_existing_output_path) {
    Config cfg;
    cfg.draft_profile_out = write_case_file("engine_calib_exists.json", "{}");
    cfg.n_spec            = 6;
    Engine e(std::move(cfg));
    EngineTestPeer::drafter(e, serial_head(6));
    CHECK_EQ(EngineTestPeer::begin_calibration(e), RAD_E_INVAL);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
}

/* An empty run is not a calibration: the export refuses rather than write a file a reader would
 * take as a baseline. The writer has its own guard (tests/draft_cost_test.cpp); this is the
 * production caller's. */
TEST(the_export_refuses_a_run_that_measured_no_bucket) {
    const std::string path = radtest::tmp_path("engine_calib_empty.json");
    Config cfg;
    cfg.draft_profile_out = path;
    Engine e(std::move(cfg));
    EngineTestPeer::set_calibrating(e, matching_runtime());
    CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_INVAL);
    CHECK(!std::ifstream(path).good());
}

/* One bucket of the fingerprint above, `ctx` and `flags` being the only difference between the
 * clean cycle and the one the mover disturbed. Origins are Verify provenance, one name per row. */
std::string bucket_json(int ctx, const std::string& flags) {
    return R"({
        "ctx_class": )" + std::to_string(ctx) + R"(,
        "shape": { "n_rows": 8, "verify_k": 3, "verify_tokens": 32, "history_tokens": 0,
                   "prefill_tokens": 0, "pipeline_depth": 0,
                   "rows": [0, 8, 8, 8], "draft_k": 3, "draft_rows": [0, 8, 8, 8],
                   "origins": ["head", "head", "head", "head", "head", "head", "head", "none"],
                   "sampling": "greedy", "replay": "play", "drafter": "serial" },
        "extent": { "rounds": 3, "draft_rows_total": 24, "draft_round_rows": [0, 8, 8, 8] },
        "time_ms": { "mean": 43.0, "unc": 0.6, "n": 5 },
        "components_ms": { "target": 25.9, "draft": 12.1, "host": 4.2, "valid": true },
        "flags": [)" + flags + R"(]
    })";
}

std::string calibrated_profile(const std::string& buckets) {
    return std::string(R"({
      "format": "rad-draft-profile/4",
      "fingerprint": {
        "model": "tiny", "quant": "bf16", "tp": 1, "tp_wire": "n/a",
        "gpus": "1x host dev=0 cu=1 vram=0", "kernels": "none", "kv_format": "bf16",
        "placement": "all_vram vram=0 host=0 file=0 dyn=1 res=0000000000000000",
        "basis": "live", "arena": "none", "live_loan": "flex=0 lend=0"
      },
      "execution": "rectangular",
      "buckets": [)") + buckets + R"(]
    })";
}

/* The fingerprint names the plan resolved at start. A bucket the mover disturbed was measured
 * under a residency that plan does not describe, and the file has no field that could name the
 * drifted one -- so that observation is left out of the calibration export, not written as if the
 * cycle had run on the start placement. Only the affected bucket goes: the clean cycle before it
 * still measured the plan and is exported. */
TEST(the_export_leaves_out_the_bucket_the_mover_disturbed) {
    const std::string out = radtest::tmp_path("engine_calib_placement.json");
    Config cfg;
    cfg.draft_profile     = write_case_file("engine_calib_placement_in.json",
                                            calibrated_profile(bucket_json(0, R"("ahead")") + "," +
                                                               bucket_json(1, R"("placement")")));
    cfg.draft_profile_out = out;
    Engine e(std::move(cfg));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    REQUIRE(EngineTestPeer::sched(e).draft_costs().active());
    REQUIRE_EQ((int)EngineTestPeer::sched(e).draft_costs().buckets().size(), 2);

    EngineTestPeer::set_calibrating(e, matching_runtime());
    REQUIRE_EQ(EngineTestPeer::export_calibration(e), RAD_OK);

    rad::DraftCostProfile written;
    std::string err;
    REQUIRE_EQ(rad::draft_cost_profile_read(out, &written, &err), RAD_OK);
    REQUIRE_EQ((int)written.buckets.size(), 1);
    CHECK_EQ(written.buckets[0].shape.ctx_class, 0);
    CHECK_EQ((int)(written.buckets[0].flags & rad::kDraftSamplePlacement), 0);
}

/* And when every measured cycle was placement-affected there is nothing left that the header
 * could describe: writing a file of diagnostics would hand a reader a profile that prices
 * nothing. */
TEST(the_export_refuses_a_calibration_whose_every_bucket_was_placement_affected) {
    const std::string out = radtest::tmp_path("engine_calib_placement_only.json");
    Config cfg;
    cfg.draft_profile     = write_case_file("engine_calib_placement_only_in.json",
                                            calibrated_profile(bucket_json(0, R"("placement")")));
    cfg.draft_profile_out = out;
    Engine e(std::move(cfg));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    REQUIRE_EQ((int)EngineTestPeer::sched(e).draft_costs().buckets().size(), 1);

    EngineTestPeer::set_calibrating(e, matching_runtime());
    CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_INVAL);
    CHECK(!std::ifstream(out).good());
}

/* Diagnostic-only buckets are not a calibration either. A record/audit/warmup/admission bracket is
 * kept out of the price (draft_bucket_usable), so a file made only of them would report a
 * calibration whose every bucket the table refuses to price from. The export refuses instead --
 * the same rule estimate() and has_usable_bucket() apply, not a second one. */
TEST(the_export_refuses_a_calibration_with_no_priceable_bucket) {
    const std::string out = radtest::tmp_path("engine_calib_diagnostic_only.json");
    Config cfg;
    cfg.draft_profile     = write_case_file("engine_calib_diagnostic_only_in.json",
                                            calibrated_profile(bucket_json(0, R"("record")")));
    cfg.draft_profile_out = out;
    Engine e(std::move(cfg));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    REQUIRE_EQ((int)EngineTestPeer::sched(e).draft_costs().buckets().size(), 1);
    CHECK(!EngineTestPeer::sched(e).draft_costs().has_usable_bucket());

    EngineTestPeer::set_calibrating(e, matching_runtime());
    CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_INVAL);
    CHECK(!std::ifstream(out).good());
}

/* A live diagnostic (--draft-calib-k without --draft-reference) measures the residency serving
 * runs, which moves: it keeps the placement-affected bucket, flag and all, and the flag is what
 * keeps a reader from pricing it. A reference calibration with the same sequence still leaves it
 * out, and a live diagnostic made only of other diagnostics is still refused. */
Config live_diagnostic(const std::string& in, const std::string& out, bool reference) {
    Config cfg;
    cfg.draft_profile     = in;
    cfg.draft_profile_out = out;
    cfg.draft_policy      = rad::DraftPolicyKind::Cost;
    cfg.draft_calib_k[0]  = 3;
    cfg.draft_calib_k[1]  = 2;
    cfg.draft_calib_n     = 2;
    cfg.draft_reference   = reference;
    return cfg;
}

TEST(a_live_diagnostic_export_keeps_the_placement_affected_bucket_flagged) {
    const std::string out = radtest::tmp_path("engine_calib_live_diag.json");
    const std::string in = write_case_file("engine_calib_live_diag_in.json",
                                           calibrated_profile(bucket_json(0, R"("ahead")") + "," +
                                                              bucket_json(1, R"("placement")")));
    Engine e(live_diagnostic(in, out, /*reference=*/false));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    EngineTestPeer::set_calibrating(e, matching_runtime());
    REQUIRE_EQ(EngineTestPeer::export_calibration(e), RAD_OK);

    rad::DraftCostProfile written;
    std::string err;
    REQUIRE_EQ(rad::draft_cost_profile_read(out, &written, &err), RAD_OK);
    REQUIRE_EQ((int)written.buckets.size(), 2);
    int flagged = 0;
    for (const rad::DraftCostBucket& b : written.buckets) {
        if (!(b.flags & rad::kDraftSamplePlacement)) continue;
        ++flagged;
        CHECK_EQ(b.shape.ctx_class, 1);
        CHECK(!rad::draft_bucket_usable(b));   /* read back, it prices nothing */
    }
    CHECK_EQ(flagged, 1);
}

TEST(a_live_diagnostic_export_of_placement_affected_buckets_alone_is_written) {
    const std::string out = radtest::tmp_path("engine_calib_live_diag_only.json");
    const std::string in = write_case_file("engine_calib_live_diag_only_in.json",
                                           calibrated_profile(bucket_json(0, R"("placement")")));
    Engine e(live_diagnostic(in, out, /*reference=*/false));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    EngineTestPeer::set_calibrating(e, matching_runtime());
    REQUIRE_EQ(EngineTestPeer::export_calibration(e), RAD_OK);

    rad::DraftCostProfile written;
    std::string err;
    REQUIRE_EQ(rad::draft_cost_profile_read(out, &written, &err), RAD_OK);
    REQUIRE_EQ((int)written.buckets.size(), 1);
    CHECK(written.buckets[0].flags & rad::kDraftSamplePlacement);
    CHECK(!rad::draft_bucket_usable(written.buckets[0]));
}

TEST(a_reference_calibration_with_a_sequence_still_leaves_the_placement_bucket_out) {
    const std::string out = radtest::tmp_path("engine_calib_ref_seq.json");
    const std::string in = write_case_file("engine_calib_ref_seq_in.json",
                                           calibrated_profile(bucket_json(0, R"("ahead")") + "," +
                                                              bucket_json(1, R"("placement")")));
    Engine e(live_diagnostic(in, out, /*reference=*/true));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    EngineTestPeer::set_calibrating(e, matching_runtime());
    REQUIRE_EQ(EngineTestPeer::export_calibration(e), RAD_OK);

    rad::DraftCostProfile written;
    std::string err;
    REQUIRE_EQ(rad::draft_cost_profile_read(out, &written, &err), RAD_OK);
    REQUIRE_EQ((int)written.buckets.size(), 1);
    CHECK_EQ((int)(written.buckets[0].flags & rad::kDraftSamplePlacement), 0);
}

TEST(a_live_diagnostic_of_other_diagnostics_alone_is_still_refused) {
    const std::string out = radtest::tmp_path("engine_calib_live_diag_record.json");
    const std::string in = write_case_file("engine_calib_live_diag_record_in.json",
                                           calibrated_profile(bucket_json(0, R"("record")")));
    Engine e(live_diagnostic(in, out, /*reference=*/false));
    REQUIRE_EQ(EngineTestPeer::load_profile(e, matching_runtime()), RAD_OK);
    EngineTestPeer::set_calibrating(e, matching_runtime());
    CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_E_INVAL);
    CHECK(!std::ifstream(out).good());
}

TEST(the_export_of_an_unarmed_run_writes_nothing) {
    const std::string path = radtest::tmp_path("engine_calib_unarmed.json");
    Config cfg;
    cfg.draft_profile_out = path;
    Engine e(std::move(cfg));
    CHECK_EQ(EngineTestPeer::export_calibration(e), RAD_OK);
    CHECK(!std::ifstream(path).good());
}

}  /* namespace */

RAD_TEST_MAIN()
