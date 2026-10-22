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
 * What it proves, and what it does not
 * ------------------------------------
 * finish_serial_drafter needs two facts before the history watermark may advance:
 * `hist_ran = serial_hist_issued_ && so.complete`. `serial_hist_issued_` is the *issuance* fact --
 * round 0's pass was built for this step. `so.complete` is the *completion* fact -- the wait ran
 * and collect_step set it (and only there). Rounds done and rows committed are neither; they are
 * recorded below as context and never asserted on.
 *
 * This case covers the issuance half. It withdraws the record after the pass really ran and shows
 * that the finish then leaves the watermark alone -- with the completion fact *present*, since on
 * the host backend every wait that ran ends complete (an unrecorded event is completed
 * immediately).
 *
 * It does not cover a completion fact that is pending or has failed while the issuance record
 * stands. That needs an end event that stays unrecorded, which the host backend cannot produce;
 * it stays open in the issue with the pending-end-event item.
 *
 * One fact the end-of-step observer cannot give by itself, and which this driver therefore
 * constructs: before=true, after=false. A step whose wait ran leaves complete = true, so the
 * sequence cannot arise from a step's own end. The seam sits at the issue site, immediately after
 * the reset, and reports the value read before `so.complete = false` beside the value the reset
 * left -- the pair is observed, not assumed.
 *
 * The watermark is read at the seam, before that step's finish runs. Therefore the value a finish
 * produced is read at the *next* step's seam: the trace is a sequence of pre-finish values, and a
 * step's own finish is judged by comparing its value with the next one's.
 *
 * Also not covered here, and open in the issue: a second participating rank, and the error paths a
 * failed wait or a failed rank takes (fail_loop runs before the finish).
 */
#include "rad_test.h"
#include "tiny_model.h"

#include "device/device.h"
#include "engine.h"
#include "engine_test_peer.h"

#include <nlohmann/json.hpp>

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

    /* What the finish will see: the pass was built and the issuance record still stands. */
    bool recorded() const { return hist_issued && !withheld; }
};

/* Written on the stepper thread from the hooks and read on the test's thread. Static storage: a
 * hook may still be in flight until serve() returns, which is after the last read. */
struct Driver {
    std::mutex              mu;
    std::condition_variable cv;
    std::vector<Rec>        trace;
    Engine*                 engine   = nullptr;
    uint64_t                req_id   = 1;
    bool                    withheld = false;
    int                     submit_rc = RAD_OK;

    size_t live_before(size_t end) const {
        size_t n = 0;
        for (size_t i = 0; i < end && i < trace.size(); ++i)
            if (trace[i].rows > 0) ++n;
        return n;
    }
};

Driver g_driver;

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

/* Our SIGTERM handler, installed before the serve thread exists. It is not a stop path: the
 * engine installs its own handler inside run_server() with SA_RESETHAND, so a delivery after that
 * one takes the default disposition and kills the process. This handler exists so the driver has
 * a disposition it can tell apart from the engine's, and so the previous one can be restored when
 * the case ends. */
void on_test_sigterm(int) {}

struct SigTermGuard {
    struct sigaction old {};
    bool installed = false;
    SigTermGuard() {
        struct sigaction sa {};
        sa.sa_handler = &on_test_sigterm;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;                 /* no SA_RESETHAND: this one stays in place */
        installed = sigaction(SIGTERM, &sa, &old) == 0;
    }
    ~SigTermGuard() { if (installed) sigaction(SIGTERM, &old, nullptr); }
};

/* The readiness proof for the engine's handler: the disposition is *read*, never inferred from a
 * delivery. A counter in our own handler cannot serve here -- handler swap and signal delivery
 * race, and a raise that finds the default disposition kills the process, while an unraised signal
 * increments nothing either way, so "the count did not move" does not mean "the engine has it".
 * sigaction with a null act changes nothing and is not itself a signal. SIG_DFL or SIG_IGN is a
 * failure, not readiness: SA_RESETHAND leaves exactly that behind after a delivery, and no stop is
 * sent then. */
bool engine_handler_installed(std::chrono::steady_clock::time_point deadline) {
    for (;;) {
        struct sigaction cur {};
        if (sigaction(SIGTERM, nullptr, &cur) != 0) return false;
        if (cur.sa_handler != &on_test_sigterm)
            return cur.sa_handler != SIG_DFL && cur.sa_handler != SIG_IGN;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

/* Exactly one stop, and none at all when the handler never appeared. No path here signals twice:
 * a second delivery under SA_RESETHAND would kill the process, so the fixed deadline in the case
 * below is the last safeguard instead. */
bool deliver_stop(std::chrono::steady_clock::time_point deadline) {
    if (!engine_handler_installed(deadline)) return false;
    return ::raise(SIGTERM) == 0;
}

}  /* namespace */

/* The seam's two strong definitions. The defaults in core/engine.cpp are weak and exist so every
 * other build of those units links; only this binary defines them. Both tolerate having no driver
 * attached (a null `engine`), and do nothing then. */
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
    /* The constructed follow-up step: the first row-carrying step after one that completed. Its
     * finish must not move the watermark, so the record the finish reads is withdrawn here --
     * after the pass was built, before collect_step and the commit. */
    if (rows > 0 && prior_complete && !g_driver.withheld && g_driver.live_before(g_driver.trace.size()) >= 1) {
        g_driver.withheld = true;
        r.withheld        = true;
        EngineTestPeer::history_issued(*g_driver.engine, false);
    }
    g_driver.trace.push_back(r);
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

namespace {

/* Both endpoints of the history watermark, on one engine started for real.
 *
 * Prompt 192 tokens against 64 a step: two mid-prompt chunks (no logit row, no pass issued, no
 * completion), then the tail (a row, the pass, the record), then decode steps. The tail is the
 * first step the watermark can move on; the step after it is the withheld one. */
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

    /* Selection runs exactly once, under std::call_once, and this call is what runs it: the backend
     * is fixed as host before the engine can reach the device layer, and no later call can change
     * it. In a HIP-free build it cannot be anything else; the call is here so the ordering is
     * checked rather than assumed. */
    REQUIRE(rad::device_is_host());

    Engine e(std::move(cfg));
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = &e;
    }
    REQUIRE_EQ(e.start(), RAD_OK);

    /* A headless run is an intermediate state, not this case: without the head there is no serial
     * drafter, no history pass and no watermark to observe. */
    const nlohmann::json info = nlohmann::json::parse(EngineTestPeer::info(e));
    const nlohmann::json sp = info.value("speculation", nlohmann::json::object());
    REQUIRE(sp.value("enabled", false));
    CHECK_EQ(sp.value("kind", std::string()), std::string("serial"));
    CHECK_EQ(sp.value("depth", -1), 3);

    /* The engine's own handler is installed inside run_server(), after the stepper starts. Until it
     * is, this one is the disposition the driver can recognise, and the stop below waits for the
     * engine's to replace it. */
    SigTermGuard sig;
    REQUIRE(sig.installed);

    std::promise<int> served;
    std::future<int>  done = served.get_future();
    std::thread serve_thread([&] {
        const int rc = e.serve(nullptr);
        served.set_value(rc);
    });

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
     * readiness proof before the raise is the installed handler; there is a residual window after
     * it, because run_server() does note_ready() and run() between installing the handler and
     * accepting, and a stop landing there is stored in the engine's flag for a watcher whose
     * srv.stop() is a no-op on a server that is not running yet. That window is why the deadline
     * below exists: it ends the process after printing the trace instead of hanging, and nothing
     * here ever signals a second time. A join that does not complete ends the process too: the
     * engine owns threads nothing else can join. */
    if (done.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
        !deliver_stop(std::chrono::steady_clock::now() + std::chrono::seconds(30))) {
        std::fprintf(stderr, "engine_loop_test: the engine's SIGTERM handler never appeared, so no "
                             "stop was sent:\n%s\n", trace_text().c_str());
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

    if (!observed || g_driver.submit_rc != RAD_OK) {
        std::fprintf(stderr, "engine_loop_test: the run did not reach the expected steps "
                             "(submit rc %d):\n%s\n", g_driver.submit_rc, trace_text().c_str());
        CHECK_EQ(g_driver.submit_rc, RAD_OK);
        CHECK(observed);
        return;
    }
    CHECK_EQ(done.get(), RAD_OK);

    /* serve() has joined the stepper, so the trace is final. */
    std::vector<Rec> tr;
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = nullptr;
        tr = g_driver.trace;
    }
    std::fprintf(stderr, "engine_loop_test: %s\n", trace_text().c_str());

    size_t iw = (size_t)-1;
    for (size_t i = 0; i < tr.size(); ++i)
        if (tr[i].withheld) { iw = i; break; }
    REQUIRE(iw != (size_t)-1);
    REQUIRE(iw >= 1);
    const Rec& tail = tr[iw - 1];
    REQUIRE(tail.rows > 0);

    /* The mid-prompt chunks: no row, no pass built, no completion, and the watermark untouched.
     * That is the natural case -- neither half is there, so nothing may advance. */
    for (size_t i = 0; i + 1 < iw; ++i) {
        CHECK_EQ(tr[i].rows, 0);
        CHECK(!tr[i].complete);
        CHECK(!tr[i].hist_issued);
        CHECK_EQ(tr[i].watermark, tr[0].watermark);
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
}

}  /* namespace */

RAD_TEST_MAIN()
