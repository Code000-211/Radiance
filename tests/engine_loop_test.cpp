/* engine_loop_test.cpp -- the serial drafter's history watermark, and the two facts a step carries
 * across it: the completion fact and the issuance one.
 *
 * One engine, started for real on the host backend over a tiny Gemma 4 checkpoint with an
 * assistant head (tests/tiny_model.h, `assistant = true`). The existing tiny fixtures are
 * untouched: the head is a `draft/` companion folder written only for this case, and the
 * `compressed-tensors` descriptor it adds is what selects `gemma4_bf16`, the plugin that has the
 * head at all. A headless run cannot pass this test: engine_info() must report a serial drafter
 * of depth 3.
 *
 * This binary is its own process: its signals, its threads and its crash are its own, and it is
 * the only one that defines the seam's strong hooks. Nothing listens beyond loopback:
 * `--host 127.0.0.1 --port 0`.
 *
 * The stop this driver sends is bound to the transport's running state, not to the signal
 * handler's disposition: the handler is installed before run(), so a stop raised between the two
 * lands in httplib's stop(), which closes nothing while the server is not running and drops it.
 * The engine's watcher reports Server::port() once it turns nonzero -- the state in which a stop is
 * accepted -- and only then is one SIGTERM raised. A report that never comes (the server failed to
 * start, or the deadline passes) ends this process as a failure rather than calling a possibly
 * dropped stop a clean cleanup.
 *
 * What it proves, and what it does not
 * ------------------------------------
 * finish_serial_drafter needs two facts before the history watermark may advance:
 * `hist_ran = serial_hist_issued_ && so.complete`. `serial_hist_issued_` is the *issuance* fact --
 * round 0's pass was built for this step. `so.complete` is the *completion* fact -- the wait ran
 * and collect_step set it (and only there). Rounds done and rows committed are neither; they are
 * recorded below as context and never asserted on.
 *
 * This case covers the shared head's half of that rule, and the issuance half:
 *
 *  - Mid-prompt, the shared head's half. A chunk short of the prompt end commits no logit row and
 *    issues no draft pass: the assistant head declares itself cache-less (`rad_drafter_shared_kv`),
 *    so `serial_shared_` is true and Scheduler::serial_begin skips every row that cannot draft
 *    (core/sched/scheduler.cpp:2266-2267). Round 0 has no batch, `serial_hist_issued_` stays false, and
 *    collect_step returns before any wait. Neither fact is claimed, so the watermark must stand --
 *    which the assertions below check. This fixture therefore cannot reach the row-less branch the
 *    repair added; engine_history_test.cpp owns that branch, with a serial head that keeps its own
 *    KV and so does owe a mid-prompt chunk its history.
 *  - The withheld step, the issuance half. Its row exists and its wait completes, but the record is
 *    withdrawn after the pass was built and before the finish reads it, so the finish leaves the
 *    watermark alone with the completion fact present (on the host backend every wait that ran ends
 *    complete: an unrecorded event is completed immediately).
 *
 * The seam runs after the step's draft passes were issued (`issue_serial_drafter`) and after
 * `issue_ahead`, immediately before `collect_step` -- the step's one wait. It therefore sees
 * `complete` as the reset left it (false, nothing has waited yet), the issuance record as the step
 * path set it, and the watermark *before* this step's finish. A step's own finish is judged by
 * comparing its value with the next seam's: a wait that ran shows up there as prior_complete, and a
 * watermark that moved shows up as a larger value. The trace is thus a sequence of pre-finish
 * values.
 *
 * Not covered: a completion fact that is pending or has failed while the issuance record stands.
 * That needs an end event that stays unrecorded, which the host backend cannot produce; it stays
 * open in the issue with the pending-end-event item. Also open: a second participating rank, and
 * the error paths a failed wait or a failed rank takes (fail_loop runs before the finish).
 *
 * Limits of this driver. One request: the seam records a per-step fact and reads a per-request
 * watermark, which name the same thing only while one request is in flight -- a step carrying
 * several rows moves each row's own watermark (Scheduler::serial_end), and "the" watermark would
 * be ambiguous. And the fixture runs `--draft-window fixed`, the policy under which every
 * drafting row keeps the full depth: the shrinking rounds of an adaptive or cost-planned cycle
 * are not exercised here.
 *
 * One fact the seam cannot give by itself, and which this driver therefore constructs:
 * before=true, after=false. A step whose wait ran leaves complete = true, so the sequence cannot
 * arise from a step's own end. The seam reports the value read before `so.complete = false`
 * beside the value the reset left -- the pair is observed, not assumed.
 */
#include "rad_test.h"
#include "tiny_model.h"

#include "device/device.h"
#include "engine.h"
#include "engine_test_peer.h"
#include "runtime/stager.h"
/* serve() takes the watch by value, so the call below destroys a temporary: the deleter needs
 * the complete type, which engine.h only forward-declares. */
#include "util/stallwatch.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using rad::Engine;
using rad::EngineTestPeer;

/* One step as the seam sees it, all of it read on the stepper thread. */
struct Rec {
    int     step           = 0;
    int     rows           = 0;      /* logit rows this step produced */
    bool    prior_complete = false;  /* the previous step's StepOut::complete, before the reset */
    bool    complete       = false;  /* immediately after the reset */
    bool    hist_issued    = false;  /* round 0's pass was built for this step */
    bool    withheld       = false;  /* this call withdrew that record before the finish read it */
    int64_t watermark      = -1;     /* the request's history watermark, before this step's finish */
    /* The step issued ahead by this step, as the rank loop's release hook has already recorded it
     * (`issue_ahead`'s second barrier join is before this seam). -1 when nothing was issued. */
    int32_t ahead_step     = -1;
    bool    tripped        = false;  /* this call published the injected residency move */

    /* What the finish will see: the pass was built and the issuance record still stands. */
    bool recorded() const { return hist_issued && !withheld; }
};

/* One step's evidence, as Scheduler::step_finished receives it. */
struct EvRec {
    int  step         = 0;
    bool known        = false;  /* a rank reported this step: not Unknown */
    int  replay       = 0;      /* DraftReplay */
    bool staged       = false;  /* the step's own trunk pass armed the stager */
    bool moved        = false;  /* placement moved a unit inside the step's window */
};

/* Written on the stepper thread from the hooks and read on the test's thread. Static storage: a
 * hook may still be in flight until serve() returns, which is after the last read. */
struct Driver {
    std::mutex              mu;
    std::condition_variable cv;
    std::vector<Rec>        trace;
    std::vector<EvRec>      ev;
    Engine*                 engine   = nullptr;
    uint64_t                req_id   = 1;
    bool                    withheld = false;
    bool                    serving  = false;  /* the engine reported the transport running */
    int                     submit_rc = RAD_OK;

    /* The ahead-shutdown case. `trip_ahead` arms the seam: the first row-carrying step then
     * publishes one residency move, and the loop's next top refuses the cycle it would open for
     * the step issued ahead. The rank loop's release hook records every trunk pass it issued
     * ahead (kIssueModel) so the seam can name that step. Both the hook and the reads below run
     * under this mutex; the test reads only after serve() joined every thread. */
    bool                 trip_ahead       = false;
    bool                 ahead_tripped    = false;
    std::vector<int32_t> model_passes;

    size_t live_before(size_t end) const {
        size_t n = 0;
        for (size_t i = 0; i < end && i < trace.size(); ++i)
            if (trace[i].rows > 0) ++n;
        return n;
    }
};

Driver g_driver;

/* The engine's cumulative residency count adds this to every rank's heat counters. Zero unless the
 * ahead-shutdown case armed it, so the watermark case runs no drift. */
std::atomic<int64_t> g_ahead_moves{0};

/* The fixture, restored on the way into and out of a case that arms it. REQUIRE returns from the
 * case body, so a reset written at the case's end runs only when every check passed; a failed one
 * would leave `serving`, the trace or an injected move to the case after it. Declared after the
 * case's Engine, so this runs before the engine it names dies. */
struct FixtureScope {
    FixtureScope() { reset(); }
    ~FixtureScope() { reset(); }
    FixtureScope(const FixtureScope&) = delete;
    FixtureScope& operator=(const FixtureScope&) = delete;
    static void reset() {
        g_ahead_moves.store(0, std::memory_order_release);
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine        = nullptr;
        g_driver.req_id        = 1;
        g_driver.withheld      = false;
        g_driver.serving       = false;
        g_driver.submit_rc     = RAD_OK;
        g_driver.trip_ahead    = false;
        g_driver.ahead_tripped = false;
        g_driver.model_passes.clear();
        g_driver.trace.clear();
        g_driver.ev.clear();
    }
};

std::string trace_text_locked() {
    std::string s = "   step  rows  prior  reset  hist  withheld  watermark\n";
    for (const Rec& r : g_driver.trace) {
        char line[128];
        std::snprintf(line, sizeof line, "  %4d  %4d  %5d  %5d  %4d  %8d  %9lld\n", r.step, r.rows,
                      (int)r.prior_complete, (int)r.complete, (int)r.hist_issued, (int)r.withheld,
                      (long long)r.watermark);
        s += line;
    }
    return s;
}

std::string trace_text() {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    return trace_text_locked();
}

/* A stager the tiny deployment never has: it arms for a prompt chunk and for nothing else, so the
 * step's `staged` bit is predictable per step. A draft pass returns early in every stager, so a
 * pass that followed the step must not be able to clear the answer. */
class PrefillStagerForTest final : public rad::OpStager {
public:
    bool begin_pass(const RadBatch* b) override {
        return b && b->draft_pass == 0 && b->phase == RAD_PHASE_PREFILL;
    }
    int before_op(rad_op, RadStream, bool* resync) override {
        *resync = false;
        return RAD_OK;
    }
    int end_pass(RadStream) override { return RAD_OK; }
};

/* The readiness proof for the stop, replacing an earlier one built on the signal disposition: the
 * engine reports Server::port() from its watcher once it is nonzero, which is exactly when the
 * transport is inside its accept loop and httplib's stop() closes the listening socket instead of
 * closing nothing. The disposition proves nothing here -- the handler is installed before run(), so
 * a stop raised on it alone can still land in the window where it is dropped.
 *
 * Bounded both ways: serve() returning first means the server never ran (a start failure) and there
 * is nothing to stop; the deadline bounds a server that never gets there. The case below ends this
 * process as a failure on both, so a stop that may have been dropped is never reported as a clean
 * cleanup. The report also implies the engine's handler is in place -- run_server() installs it
 * before the watcher that reports exists -- so the one raise cannot meet the default disposition. */
bool wait_serving(std::future<int>& done, std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        {
            std::lock_guard<std::mutex> lk(g_driver.mu);
            if (g_driver.serving) return true;
        }
        if (done.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) return false;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

}  /* namespace */

/* The seam's three strong definitions. The defaults in core/engine.cpp are weak and exist so every
 * other build of those units links; only this binary defines them. All three tolerate having no
 * driver attached (a null `engine`), and do nothing then. */
extern "C" void rad_test_step_point(int step, int rows, bool prior_complete, bool complete) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (!g_driver.engine) return;
    Rec r;
    r.step           = step;
    r.rows           = rows;
    r.prior_complete = prior_complete;
    r.complete       = complete;
    r.hist_issued    = EngineTestPeer::history_issued(*g_driver.engine);
    r.watermark      = EngineTestPeer::watermark(*g_driver.engine, g_driver.req_id);
    /* The pass `issue_ahead` just released is the next step's; the rank loop's hook recorded it
     * before the barrier join this seam follows. */
    for (int32_t st : g_driver.model_passes)
        if (st > step && st > r.ahead_step) r.ahead_step = st;
    /* The constructed follow-up step: the first row-carrying step after one that completed. Its
     * finish must not move the watermark, so the record the finish reads is withdrawn here --
     * after the pass was built, before collect_step and the commit. */
    if (rows > 0 && prior_complete && !g_driver.withheld && g_driver.live_before(g_driver.trace.size()) >= 1) {
        g_driver.withheld = true;
        r.withheld        = true;
        EngineTestPeer::history_issued(*g_driver.engine, false);
    }
    /* The ahead-shutdown case: one move, published only once an ahead pass is in flight (this
     * step's successor). The loop reads it after this step's commit -- step_finished notes the
     * drift -- and its next top refuses the cycle the ahead step would open. */
    if (g_driver.trip_ahead && rows > 0 && !g_driver.ahead_tripped) {
        g_driver.ahead_tripped = true;
        r.tripped              = true;
        g_ahead_moves.store(1, std::memory_order_release);
    }
    g_driver.trace.push_back(r);
    g_driver.cv.notify_all();
}

/* The engine's cumulative residency count. The ahead-shutdown case publishes one move through
 * this seam; nothing else in this binary arms it. */
extern "C" int64_t rad_test_residency_moves() {
    return g_ahead_moves.load(std::memory_order_acquire);
}

/* The rank loop's release hook. It injects no move here (0, the weak default's answer) and records
 * the trunk pass the loop issued ahead, on the rank thread that ran it -- after run_step, so a
 * step named here really executed. */
extern "C" int64_t rad_test_tiering_moves(int rank, int32_t step, int what) {
    if (what == EngineTestPeer::kIssueModel) {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        if (g_driver.engine) g_driver.model_passes.push_back(step);
    }
    (void)rank;
    return 0;
}

/* The evidence hook: what the engine hands Scheduler::step_finished for `step`. Recorded on the
 * stepper thread after the step's collect, which is after the next step's ahead pass has run --
 * so this is exactly the point where a cumulative counter delta would report the wrong pass. */
extern "C" void rad_test_step_evidence(int step, int replay_known, int replay, int staged,
                                       int moved) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (!g_driver.engine) return;
    g_driver.ev.push_back(EvRec{ step, replay_known != 0, replay, staged != 0, moved != 0 });
    g_driver.cv.notify_all();
}

/* One request, submitted inside serve() on the serve thread, after the scheduler is initialised and
 * before any rank or the stepper exists. Its sink is null: sink.h treats that as nobody listening,
 * so a benchmark-style request is safe. */
extern "C" void rad_test_serving() {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (!g_driver.engine) return;
    auto req = std::make_unique<rad::Request>();
    req->id         = g_driver.req_id;
    req->prompt.resize(192);
    for (size_t i = 0; i < req->prompt.size(); ++i)
        req->prompt[i] = (int32_t)(1 + (i * 37) % 318);
    req->max_tokens = 16;
    req->ignore_eos = true;
    req->sp.temp    = 0.0f;
    req->sink       = nullptr;
    g_driver.submit_rc = EngineTestPeer::submit(*g_driver.engine, std::move(req));
}

/* The engine's readiness report, from its watcher thread: the transport's port is nonzero, so a
 * stop is one the server accepts from here on. It records and nothing else -- the stop itself is
 * raised by the case, once. wait_serving() polls the flag; the report is one write and the wait's
 * slice is 5 ms, so there is no notification to lose. */
extern "C" void rad_test_serving_ready(int port) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (port > 0) g_driver.serving = true;
}

namespace {

/* The ahead-shutdown proof: a cost run stops on a residency drift while the next step's forward
 * pass is already issued, and nothing of that step is collected, committed or published.
 *
 * The drift is injected through the test build's cumulative-count seam (rad_test_residency_moves,
 * core/engine.cpp), as in engine_cost_drift_test.cpp. What this case adds is the shape the idle
 * drift case cannot produce: a real request, so `issue_ahead` actually releases the next step's
 * trunk pass (kIssueModel) before the current step is collected. The seam then publishes one move
 * on the first row-carrying step (the tail chunk). `Scheduler::step_finished` of that step notes
 * the drift; the next loop top refuses the cycle `promote_next()` would open for the step issued
 * ahead, and the engine takes `fail_loop` -- before that step's sampler, collect and commit.
 *
 * What the run proves:
 *  - the ahead pass really ran on the rank: the rank loop's release hook (which fires after
 *    run_step) named its step, and Engine::step_evidence still holds that step's frozen record;
 *  - it was never committed or published: no step point and no evidence hook names it, and the
 *    scheduler counted no commit for it;
 *  - the stop drained and joined: serve() returned only after joining the stepper and the rank
 *    threads (core/engine.cpp), with the request failed by Scheduler::shutdown;
 *  - nothing was released under a running pass: the seam that publishes the move runs after
 *    issue_ahead's second barrier join -- i.e. after the rank's pass returned -- while
 *    Scheduler::shutdown runs at the next loop top; the request's release therefore follows the
 *    pass that read it.
 *
 * The limit this host run cannot cross: the host backend executes run_step synchronously on the
 * rank thread, so "the pass completed before the release" is observable here, but an asynchronous
 * stream drain is not -- there is no unwaited device work to observe. On a device the completion
 * boundary is the teardown sync in Ctx::shutdown (core/runtime/ctx.cpp), which this test cannot
 * see; that half stays a code-path claim, not a measured one.
 *
 * This case raises no signal: the stop comes from the aborted step loop. It used to have to stay
 * ahead of the watermark case, which raises SIGTERM, because run_server never cleared the
 * process-global flag the handler sets -- a later serve would read the earlier run's stop before
 * its transport reached the accept loop, stop nothing, and serve a listener nothing could stop.
 * That lifecycle is repaired (core/engine.cpp clears the flag per run; core/server/http.cpp
 * remembers a stop that precedes its listener), and FixtureScope resets the driver on every path,
 * so neither case depends on the other's cleanliness or on the order. */
TEST(a_cost_drift_stop_leaves_the_step_issued_ahead_uncommitted_and_drains_it) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);

    tiny::TempDir dir("engine_loop_ahead");
    REQUIRE(dir.ok());
    tiny::Gemma4 s;
    s.assistant = true;
    REQUIRE(tiny::write_gemma4(dir.path, s));

    const std::string profile_path = dir / "ahead-profile.json";
    const int kChunk = 64;
    Argv argv({ "--model", dir.path,
                "--radiance-home", home(),
                "--kernels", "libref",
                "--debug-accept-reference-kernels",
                /* The cost policy's default tier ladder tops out at 6 (DraftPolicyConfig::tiers);
                 * a shallower ceiling is refused as unconfigurable, not clamped. */
                "--num-speculative-tokens", "6",
                "--draft-window", "fixed",
                "--draft-policy", "cost",
                "--draft-profile", profile_path,
                "--max-num-batched-tokens", std::to_string(kChunk),
                "--host", "127.0.0.1",
                "--port", "0" });
    rad::Config cfg;
    REQUIRE_EQ(rad::config_parse(argv.argc(), argv.argv.data(), &cfg), RAD_OK);
    REQUIRE(rad::device_is_host());

    Engine e(std::move(cfg));
    /* From here to the end of the case, every path -- including a REQUIRE's early return -- leaves
     * the global driver clean for the case after this one. */
    FixtureScope fixture;
    REQUIRE_EQ(e.start(), RAD_OK);
    {
        const nlohmann::json info = nlohmann::json::parse(EngineTestPeer::info(e));
        const nlohmann::json sp = info.value("speculation", nlohmann::json::object());
        REQUIRE(sp.value("enabled", false));
        REQUIRE_EQ(sp.value("kind", std::string()), std::string("serial"));
        REQUIRE_EQ(sp.value("depth", -1), 6);
    }

    /* The identity serve() will derive and read the profile against. start() resolved the plan,
     * so the runtime is derivable here; serve() reads the file only after this. The one bucket is
     * a legal key (measured cleanly), but no shape this fixture plans matches it, so every cycle
     * prices through the policy's named fallback -- the point here is the drift, not the price. */
    rad::DraftCostRuntime rt;
    REQUIRE_EQ(EngineTestPeer::runtime(e, &rt), RAD_OK);
    {
        rad::DraftCostProfile prof;
        prof.fingerprint = rt.fingerprint;
        prof.execution   = rt.execution;
        rad::DraftCostBucket b;
        b.shape.n_rows         = 1;
        b.shape.verify_k       = 1;
        b.shape.rows[1]        = 1;
        b.shape.verify_tokens  = 2;
        b.shape.history_tokens = 1;
        b.shape.pipeline_depth = 1;
        b.shape.draft_k        = 0;
        b.shape.sampling       = rad::DraftSampling::Greedy;
        b.shape.replay         = rad::DraftReplay::None;
        b.shape.drafter        = rad::DraftDrafter::Serial;
        b.origin_set[0]        = (uint8_t)(1u << (unsigned)rad::DraftOrigin::Head);
        b.rounds               = 0;
        b.draft_rows_total     = 0;
        b.draft_round_rows[0]  = 0;
        b.mean_ms              = 1.0;
        b.unc_ms               = 0.1;
        b.n                    = 1;
        prof.buckets.push_back(b);
        int32_t skipped = 0;
        std::string err;
        REQUIRE_EQ(rad::draft_cost_profile_write_file(profile_path, prof, &skipped, &err), RAD_OK);
        CHECK_EQ(skipped, 0);
    }

    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine        = &e;
        g_driver.req_id        = 1;
        g_driver.withheld      = false;
        g_driver.serving       = false;
        g_driver.submit_rc     = RAD_OK;
        g_driver.trace.clear();
        g_driver.ev.clear();
        g_driver.model_passes.clear();
        g_driver.ahead_tripped = false;
        g_driver.trip_ahead    = true;
    }
    g_ahead_moves.store(0, std::memory_order_release);

    std::promise<int> served;
    std::future<int>  done = served.get_future();
    std::thread serve_thread([&] {
        const int rc = e.serve(nullptr);
        served.set_value(rc);
    });

    /* No signal: the aborted step loop stops the transport itself (fail_loop's loop_status_). A
     * serve() that does not return means that stop failed; the engine owns threads nothing else
     * can join, so the process ends here as the failure it is. */
    if (done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        std::fprintf(stderr, "engine_loop_test: serve() did not return within 60 s after the cost "
                             "drift:\n%s\n", trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    serve_thread.join();

    std::vector<Rec>     tr;
    std::vector<EvRec>   ev;
    std::vector<int32_t> model_passes;
    bool                 tripped = false;
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine     = nullptr;
        g_driver.trip_ahead = false;
        tripped             = g_driver.ahead_tripped;
        tr                  = g_driver.trace;
        ev                  = g_driver.ev;
        model_passes        = g_driver.model_passes;
    }

    CHECK_EQ(done.get(), RAD_E_STATE);
    CHECK(tripped);
    REQUIRE(!tr.empty());
    std::fprintf(stderr, "engine_loop_test (ahead): %s\n", trace_text().c_str());

    /* The tail chunk: the first step that produced a logit row, and the seam's trip point. */
    size_t it = (size_t)-1;
    for (size_t i = 0; i < tr.size(); ++i)
        if (tr[i].tripped) { it = i; break; }
    REQUIRE(it != (size_t)-1);
    const Rec& tail = tr[it];
    CHECK(tail.rows > 0);

    /* The pass issued ahead: exactly one model release in this fixture (mid-prompt chunks issue
     * none), for the step after the tail. The seam saw it already run. */
    REQUIRE_EQ(model_passes.size(), (size_t)1);
    const int32_t ahead_step = model_passes[0];
    CHECK_EQ(ahead_step, tail.step + 1);
    CHECK_EQ(tail.ahead_step, ahead_step);

    /* The rank's frozen record for that step still names it: the pass really executed. */
    rad::DraftStepEvidence ahead_ev{};
    CHECK(EngineTestPeer::step_evidence(e, ahead_step, &ahead_ev));

    /* And no step ever collected or committed it: one step point per committed step, one evidence
     * record per commit, none for the ahead step. */
    for (const Rec& r : tr) CHECK(r.step != ahead_step);
    REQUIRE_EQ(ev.size(), tr.size());
    for (const EvRec& x : ev) CHECK(x.step != ahead_step);
    for (size_t i = 0; i < ev.size(); ++i) CHECK_EQ(ev[i].step, tr[i].step);
    const rad::SchedMetrics m = EngineTestPeer::sched(e).metrics();
    CHECK_EQ(m.steps, (int64_t)tr.size());
    CHECK_EQ(m.failed, 1);
    CHECK_EQ(m.finished, 0);
    CHECK_EQ(m.running, 0);
    CHECK_EQ(m.queue_depth, 0);

    /* The drift itself: the loaded profile stopped pricing, the cycle for the ahead step was
     * refused, and the run did not fall back to the ceiling. */
    CHECK(EngineTestPeer::sched(e).draft_cost_aborted());
    CHECK_EQ(EngineTestPeer::sched(e).draft_costs().buckets().size(), (size_t)1);
    CHECK(!EngineTestPeer::sched(e).draft_costs().active());
    CHECK_EQ(EngineTestPeer::sched(e).draft_costs().reason(),
             std::string("residency drifted from the fingerprint's start placement"));
    CHECK(!EngineTestPeer::sched(e).draft_costs().fallback_admitted());
    /* The fixture reset that used to stand here moved to FixtureScope: this path is the one that
     * reaches the end, and the failed ones had to leave the same state behind. */
}

/* A second serve in the same process is a second stop lifetime.
 *
 * The SIGTERM that ended the first run is answered by a process-global flag. If run_server kept
 * it, the second run's watcher would read the first run's stop before the second transport's
 * accept loop exists, stop nothing, and then wait on a listener nothing can stop -- the call
 * below would not return, and both bounded waits end this process as the failure that is. Both
 * stops are real SIGTERMs, raised only after the engine reported its transport running, so the
 * handler is in place and the first has completed before the second starts. The same engine
 * serves twice: serve() builds its own threads and scheduler state, so this is the repeat, not a
 * second engine.
 *
 * The driver is left detached -- the serving seam submits a request only when one is attached --
 * so the case is about the stop lifetime, not about what a stopped run leaves in the scheduler.
 * Its FixtureScope resets the driver on the way in and out, and no signal remains for the case
 * after it. */
TEST(a_second_serve_does_not_inherit_the_previous_runs_stop) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);

    tiny::TempDir dir("engine_loop_serve_again");
    REQUIRE(dir.ok());
    tiny::Gemma4 s;
    s.assistant = true;
    REQUIRE(tiny::write_gemma4(dir.path, s));

    Argv argv({ "--model", dir.path,
                "--radiance-home", home(),
                "--kernels", "libref",
                "--debug-accept-reference-kernels",
                "--num-speculative-tokens", "3",
                "--draft-window", "fixed",
                "--max-num-batched-tokens", "64",
                "--host", "127.0.0.1",
                "--port", "0" });
    rad::Config cfg;
    REQUIRE_EQ(rad::config_parse(argv.argc(), argv.argv.data(), &cfg), RAD_OK);
    REQUIRE(rad::device_is_host());

    Engine e(std::move(cfg));
    FixtureScope fixture;
    REQUIRE_EQ(e.start(), RAD_OK);

    for (int run = 0; run < 2; ++run) {
        {
            std::lock_guard<std::mutex> lk(g_driver.mu);
            g_driver.serving = false;
        }
        std::promise<int> served;
        std::future<int>  done = served.get_future();
        std::thread serve_thread([&] { served.set_value(e.serve(nullptr)); });

        /* Nothing is signalled before the server is known to be running: a start failure and a
         * stop inherited from the previous run both end this process here, with no stop sent. */
        if (!wait_serving(done, std::chrono::steady_clock::now() + std::chrono::seconds(60))) {
            const bool returned =
                done.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
            std::fprintf(stderr, "engine_loop_test (serve again, run %d): the server never "
                                 "reported a running transport%s, so no stop was sent:\n%s\n",
                         run + 1, returned ? " (serve() returned first)" : " within 60 s",
                         trace_text().c_str());
            std::fflush(stderr);
            std::_Exit(1);
        }
        if (done.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
            ::raise(SIGTERM) != 0) {
            std::fprintf(stderr, "engine_loop_test (serve again, run %d): raise(SIGTERM) failed "
                                 "although the server reported itself running and accepts a "
                                 "stop:\n%s\n", run + 1, trace_text().c_str());
            std::fflush(stderr);
            std::_Exit(1);
        }
        if (done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
            std::fprintf(stderr, "engine_loop_test (serve again, run %d): serve() did not return "
                                 "within 60 s after the stop:\n%s\n", run + 1,
                         trace_text().c_str());
            std::fflush(stderr);
            std::_Exit(1);
        }
        serve_thread.join();
        CHECK_EQ(done.get(), RAD_OK);
    }
}

/* A stop raised between two serve runs is not swallowed by the next run's start.
 *
 * Run 1 ends without a signal -- the drifted ahead pass aborts the step loop and fail_loop stops
 * the transport -- so the SIGTERM handler stays installed: SA_RESETHAND is spent only by a signal
 * that actually runs the handler. The SIGTERM below therefore arrives between the runs and sets
 * the process-global flag. The second run's watcher must consume it and stop; a start that
 * cleared the flag would swallow the stop and serve a listener nothing can stop, and the bounded
 * wait below ends this process as that failure. The driver is detached before run 2, so the
 * serving seam submits nothing: the second run can only end through the carried stop. */
TEST(a_stop_between_two_serve_runs_stops_the_second) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);

    tiny::TempDir dir("engine_loop_between_runs");
    REQUIRE(dir.ok());
    tiny::Gemma4 s;
    s.assistant = true;
    REQUIRE(tiny::write_gemma4(dir.path, s));

    const std::string profile_path = dir / "between-runs-profile.json";
    FixtureScope fixture;

    /* Run 1: the drifted ahead pass aborts the step loop, and fail_loop stops the transport. No
     * signal is raised, so the handler this run installed stays installed after the engine is
     * gone. The engine is scoped: the second run gets none of the failed run's state, so only the
     * carried stop can end it. */
    {
        Argv argv({ "--model", dir.path,
                    "--radiance-home", home(),
                    "--kernels", "libref",
                    "--debug-accept-reference-kernels",
                    "--num-speculative-tokens", "6",
                    "--draft-window", "fixed",
                    "--draft-policy", "cost",
                    "--draft-profile", profile_path,
                    "--max-num-batched-tokens", "64",
                    "--host", "127.0.0.1",
                    "--port", "0" });
        rad::Config cfg;
        REQUIRE_EQ(rad::config_parse(argv.argc(), argv.argv.data(), &cfg), RAD_OK);
        REQUIRE(rad::device_is_host());

        Engine e(std::move(cfg));
        REQUIRE_EQ(e.start(), RAD_OK);

        /* The one legal bucket the cost policy loads, in no shape this fixture plans: the drift,
         * not a price, is what this run needs. Same fixture as the ahead case. */
        rad::DraftCostRuntime rt;
        REQUIRE_EQ(EngineTestPeer::runtime(e, &rt), RAD_OK);
        {
            rad::DraftCostProfile prof;
            prof.fingerprint = rt.fingerprint;
            prof.execution   = rt.execution;
            rad::DraftCostBucket b;
            b.shape.n_rows         = 1;
            b.shape.verify_k       = 1;
            b.shape.rows[1]        = 1;
            b.shape.verify_tokens  = 2;
            b.shape.history_tokens = 1;
            b.shape.pipeline_depth = 1;
            b.shape.draft_k        = 0;
            b.shape.sampling       = rad::DraftSampling::Greedy;
            b.shape.replay         = rad::DraftReplay::None;
            b.shape.drafter        = rad::DraftDrafter::Serial;
            b.origin_set[0]        = (uint8_t)(1u << (unsigned)rad::DraftOrigin::Head);
            b.rounds               = 0;
            b.draft_rows_total     = 0;
            b.draft_round_rows[0]  = 0;
            b.mean_ms              = 1.0;
            b.unc_ms               = 0.1;
            b.n                    = 1;
            prof.buckets.push_back(b);
            int32_t skipped = 0;
            std::string err;
            REQUIRE_EQ(rad::draft_cost_profile_write_file(profile_path, prof, &skipped, &err),
                       RAD_OK);
            CHECK_EQ(skipped, 0);
        }

        {
            std::lock_guard<std::mutex> lk(g_driver.mu);
            g_driver.engine        = &e;
            g_driver.req_id        = 1;
            g_driver.withheld      = false;
            g_driver.serving       = false;
            g_driver.submit_rc     = RAD_OK;
            g_driver.trace.clear();
            g_driver.ev.clear();
            g_driver.model_passes.clear();
            g_driver.ahead_tripped = false;
            g_driver.trip_ahead    = true;
        }
        g_ahead_moves.store(0, std::memory_order_release);

        std::promise<int> first;
        std::future<int>  first_done = first.get_future();
        std::thread first_thread([&] { first.set_value(e.serve(nullptr)); });
        if (first_done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
            std::fprintf(stderr, "engine_loop_test (between runs): the first serve() did not "
                                 "return within 60 s after the cost drift:\n%s\n",
                         trace_text().c_str());
            std::fflush(stderr);
            std::_Exit(1);
        }
        first_thread.join();
        CHECK_EQ(first_done.get(), RAD_E_STATE);

        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine     = nullptr;
        g_driver.trip_ahead = false;
    }

    /* Between the runs: the handler run 1 installed is still in place -- SA_RESETHAND is spent
     * only by a signal that actually ran it -- so this SIGTERM sets the process-global flag. */
    REQUIRE_EQ(::raise(SIGTERM), 0);

    /* A fresh engine: nothing but the carried stop can end this serve. An entry that clears the
     * flag swallows the stop, the listener runs, and the bounded wait reports it. */
    Argv again({ "--model", dir.path,
                 "--radiance-home", home(),
                 "--kernels", "libref",
                 "--debug-accept-reference-kernels",
                 "--num-speculative-tokens", "3",
                 "--draft-window", "fixed",
                 "--max-num-batched-tokens", "64",
                 "--host", "127.0.0.1",
                 "--port", "0" });
    rad::Config cfg2;
    REQUIRE_EQ(rad::config_parse(again.argc(), again.argv.data(), &cfg2), RAD_OK);
    Engine e2(std::move(cfg2));
    REQUIRE_EQ(e2.start(), RAD_OK);

    std::promise<int> second;
    std::future<int>  second_done = second.get_future();
    std::thread second_thread([&] { second.set_value(e2.serve(nullptr)); });
    if (second_done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        std::fprintf(stderr, "engine_loop_test (between runs): the second serve() did not return "
                             "within 60 s after the stop raised between the runs: the stop was "
                             "swallowed at the run's start and the listener is waiting for "
                             "another one:\n%s\n", trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    second_thread.join();
    CHECK_EQ(second_done.get(), RAD_OK);
}

/* Both endpoints of the history watermark, on one engine started for real.
 *
 * Prompt 192 tokens against 64 a step: two mid-prompt chunks (no logit row and, on this shared head,
 * no draft pass to issue), then the tail -- the third chunk, which commits the first row and so
 * issues one -- then decode steps. The tail's finish moves the history watermark; the step after
 * the tail is the withheld one. */
TEST(engine_loop_watermark_does_not_move_when_the_issuance_record_is_withdrawn) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    /* Host execution has to hold *before* start(), not be checked after it: a HIP build probes the
     * runtime inside select_backend(), and an invisible card only makes the host likely -- a
     * device_is_host() after start() comes too late to rule that probe out. The guarantee this
     * case needs is the compile-time one: with -DRAD_WITH_HIP=OFF, select_backend() has no HIP
     * branch at all. */
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    /* Keeps ctest's runs from finding a card; the guarantee is the build, not this. */
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);

    tiny::TempDir dir("engine_loop");
    REQUIRE(dir.ok());
    tiny::Gemma4 s;
    s.assistant = true;
    REQUIRE(tiny::write_gemma4(dir.path, s));

    /* The chunk size the prompt below is cut into: 192 tokens against this many a step gives the
     * two row-less chunks the case asserts on, then the tail. */
    const int kChunk = 64;

    Argv argv({ "--model", dir.path,
                "--radiance-home", home(),
                "--kernels", "libref",
                "--debug-accept-reference-kernels",
                "--num-speculative-tokens", "3",
                "--draft-window", "fixed",
                "--max-num-batched-tokens", std::to_string(kChunk),
                "--host", "127.0.0.1",
                "--port", "0" });
    rad::Config cfg;
    REQUIRE_EQ(rad::config_parse(argv.argc(), argv.argv.data(), &cfg), RAD_OK);

    /* Selection runs exactly once, under std::call_once, and this call is what runs it: the backend
     * is fixed as host before the engine can reach the device layer, and no later call can change
     * it. In a HIP-free build it cannot be anything else; the call is here so the ordering is
     * checked rather than assumed. */
    REQUIRE(rad::device_is_host());

    Engine e(std::move(cfg));
    REQUIRE_EQ(e.start(), RAD_OK);

    /* A headless run is an intermediate state, not this case: without the head there is no serial
     * drafter, no history pass and no watermark to observe. */
    const nlohmann::json info = nlohmann::json::parse(EngineTestPeer::info(e));
    const nlohmann::json sp = info.value("speculation", nlohmann::json::object());
    REQUIRE(sp.value("enabled", false));
    CHECK_EQ(sp.value("kind", std::string()), std::string("serial"));
    CHECK_EQ(sp.value("depth", -1), 3);

    /* Attached for the rest of the case, and only now: nothing of the engine's runs before the
     * serve thread below, and every early return that follows the join releases it first. */
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = &e;
    }
    /* After start(), which attaches the deployment's own stager (none, on this fixture), and before
     * the stepper: every rank answers `staged` from this stager from now on. Static storage: the
     * rank threads hold the pointer until the join. */
    PrefillStagerForTest staging;
    EngineTestPeer::stager(e, &staging);

    std::promise<int> served;
    std::future<int>  done = served.get_future();
    std::thread serve_thread([&] {
        const int rc = e.serve(nullptr);
        served.set_value(rc);
    });

    /* Nothing is signalled before the server is known to be running: a start failure (bind, model
     * load in run_server) and a transport that never gets there both end this process here, as a
     * failure, with no stop sent at all. */
    if (!wait_serving(done, std::chrono::steady_clock::now() + std::chrono::seconds(60))) {
        const bool returned =
            done.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
        std::fprintf(stderr, "engine_loop_test: the server never reported a running transport%s, so "
                             "no stop was sent and this run is not a clean one:\n%s\n",
                     returned ? " (serve() returned first)" : " within 60 s", trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }

    /* Every wait below is bounded. `satisfied` asks for the three steps the comparison needs: the
     * withheld step, and two row-carrying steps after it (the second one's seam is where the first
     * one's finish is judged). A submission the scheduler refused never steps, so it ends the wait
     * at once and is reported below instead of burning the deadline. */
    auto satisfied = [&] {
        if (g_driver.submit_rc != RAD_OK) return true;
        size_t iw = (size_t)-1;
        for (size_t i = 0; i < g_driver.trace.size(); ++i)
            if (g_driver.trace[i].withheld) { iw = i; break; }
        if (iw == (size_t)-1) return false;
        size_t live = 0;
        for (size_t i = iw + 1; i < g_driver.trace.size(); ++i)
            if (g_driver.trace[i].rows > 0) ++live;
        return live >= 2;
    };

    bool observed = false;
    {
        std::unique_lock<std::mutex> lk(g_driver.mu);
        observed = g_driver.cv.wait_for(lk, std::chrono::seconds(60), satisfied);
    }

    /* One stop, one join, on every path from here -- including the ones that already failed. The
     * raise is bound to the running state wait_serving() proved, and a running server leaves that
     * state only by run() returning, so the ready check is exactly what keeps the raise inside it.
     * Nothing here ever signals a second time: the engine's handler is SA_RESETHAND, so a second
     * delivery kills the process. A join that does not complete ends the process too, since the
     * engine owns threads nothing else can join. */
    if (done.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
        ::raise(SIGTERM) != 0) {
        std::fprintf(stderr, "engine_loop_test: raise(SIGTERM) failed although the server reported "
                             "itself running and accepts a stop:\n%s\n", trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    if (done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        std::fprintf(stderr, "engine_loop_test: serve() did not return within 60 s:\n%s\n",
                     trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    serve_thread.join();

    /* serve() has joined the stepper, so the trace is final and no hook can be in flight; the
     * engine is released here rather than after the checks, so the early return below leaves no
     * global pointing at an engine that is about to die. */
    std::vector<Rec> tr;
    std::vector<EvRec> ev;
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = nullptr;
        tr = g_driver.trace;
        ev = g_driver.ev;
    }

    if (!observed || g_driver.submit_rc != RAD_OK) {
        std::fprintf(stderr, "engine_loop_test: the run did not reach the expected steps "
                             "(submit rc %d):\n%s\n", g_driver.submit_rc, trace_text().c_str());
        CHECK_EQ(g_driver.submit_rc, RAD_OK);
        CHECK(observed);
        return;
    }
    CHECK_EQ(done.get(), RAD_OK);

    std::fprintf(stderr, "engine_loop_test: %s\n", trace_text().c_str());

    size_t iw = (size_t)-1;
    for (size_t i = 0; i < tr.size(); ++i)
        if (tr[i].withheld) { iw = i; break; }
    REQUIRE(iw != (size_t)-1);
    REQUIRE(iw >= 1);
    const Rec& tail = tr[iw - 1];
    REQUIRE(tail.rows > 0);

    /* The mid-prompt chunks: no logit row and no draft pass either. The assistant head keeps no
     * cache of its own (rad_drafter_shared_kv), so serial_begin skips a row that cannot draft,
     * round 0 has no batch, and the issuance record stays false; collect_step returns before any
     * wait, so the completion fact stays false too. With neither fact claimed the watermark cannot
     * move, and the next seam shows that (prior_complete false, the same watermark). */
    for (size_t i = 0; i + 1 < iw; ++i) {
        CHECK_EQ(tr[i].rows, 0);
        CHECK(!tr[i].hist_issued);
        CHECK(!tr[i].complete);
        CHECK(!tr[i + 1].prior_complete);
        CHECK_EQ(tr[i].watermark, tr[0].watermark);
        CHECK_EQ(tr[i + 1].watermark, tr[0].watermark);
    }

    /* The tail step: its pass was built and its wait ran (`complete` after the wait is the
     * completion fact; the row count alone is not). Its finish therefore moves the watermark, read
     * at the next seam. */
    CHECK(tail.hist_issued);
    CHECK(tail.recorded());
    CHECK(!tail.complete);              /* at its own reset, before its wait */
    CHECK(tr[iw].watermark > tail.watermark);

    /* The constructed follow-up, at the reset, immediately after it: the previous step completed
     * and this one's flag is already cleared. */
    REQUIRE(tr[iw].prior_complete);
    CHECK(!tr[iw].complete);
    /* Its pass really ran, and the seam withdrew the issuance record the finish reads -- the
     * completion half is present, since that step's wait runs and completes as usual. */
    CHECK(tr[iw].hist_issued);
    CHECK(!tr[iw].recorded());

    /* The finish that ran without the issuance record left the watermark where it was. */
    size_t after = (size_t)-1;
    for (size_t i = iw + 1; i < tr.size(); ++i)
        if (tr[i].rows > 0) { after = i; break; }
    REQUIRE(after != (size_t)-1);
    CHECK_EQ(tr[after].watermark, tr[iw].watermark);
    CHECK(tr[after].hist_issued);

    /* With its record back, the next finish advances it again. */
    size_t resumed = (size_t)-1;
    for (size_t i = after + 1; i < tr.size(); ++i)
        if (tr[i].rows > 0) { resumed = i; break; }
    REQUIRE(resumed != (size_t)-1);
    CHECK(tr[resumed].watermark > tr[after].watermark);

    /* The cycle's evidence, as Scheduler::step_finished received it (rad_test_step_evidence reads
     * it there, after the step's collect and therefore after the next step's ahead pass has run).
     * One record per finished step, in step order, and tagged with the step that finished. */
    REQUIRE_EQ(ev.size(), tr.size());
    for (size_t i = 0; i < tr.size(); ++i) CHECK_EQ(ev[i].step, tr[i].step);
    /* No rank reported nothing: every step here really ran a pass, on the host backend, so the
     * state is a measured one and never Unknown. */
    for (const EvRec& x : ev) CHECK(x.known);
    /* The host backend has no tape (only the AQL one defines tape_begin), so every pass is issued
     * live and the class is the measured None. Play/Record/Audit are not producible here, and the
     * per-step freezing they need is the same record this checks. */
    for (const EvRec& x : ev) CHECK_EQ(x.replay, (int)rad::DraftReplay::None);
    /* Nothing in this fixture moves a unit between tiers (a static plan, no placement engine), and
     * a step that claims otherwise would price itself with kDraftSamplePlacement. */
    for (const EvRec& x : ev) CHECK(!x.moved);

    /* Every prompt chunk was staged: the stager armed for that pass, and the answer is the step's
     * own. The tail chunk is the one that matters -- the decode step behind it is issued ahead
     * before the tail is collected, so a stager latch read at collect time would answer with the
     * decode pass (false) instead of the tail's own (true). */
    for (size_t i = 0; i < iw; ++i) {
        CHECK_EQ(tr[i].step, ev[i].step);
        CHECK(ev[i].staged);
    }
    /* And the decode steps that follow are not staged: a decode step must not inherit the flag of
     * a prefill pass either. */
    for (size_t i = iw; i < ev.size(); ++i) CHECK(!ev[i].staged);
}

}  /* namespace */

RAD_TEST_MAIN()
