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
  "format": "rad-draft-profile/2",
  "fingerprint": {
    "model": "tiny", "quant": "bf16", "tp": 1, "tp_wire": "n/a",
    "gpus": "1x host dev=0 cu=1 vram=0", "kernels": "none", "kv_format": "bf16",
    "placement": "all_vram vram=0 host=0 file=0 dyn=1 res=0000000000000000"
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

}  /* namespace */
