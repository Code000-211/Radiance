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
 * This case covers both halves of that rule, on the two paths this run produces:
 *
 *  - Mid-prompt, the row-less half. A chunk short of the prompt end commits no logit row, but its
 *    hidden states are history the head still needs, so round 0 is built for it anyway
 *    (Scheduler::serial_begin keeps one round when no row drafts, "because round 0 is the history
 *    pass the remaining rows still owe"). collect_step therefore takes the repaired
 *    `n_rows <= 0 && serial_hist_issued_` branch: it awaits the end and sets complete. The finish
 *    moves the watermark by exactly the chunk, and the next seam shows both facts -- prior_complete,
 *    and a watermark one chunk further on.
 *  - The withheld step, the issuance half. Its row exists and its wait completes, but the record is
 *    withdrawn after the pass was built and before the finish reads it, so the finish leaves the
 *    watermark alone with the completion fact present (on the host backend every wait that ran ends
 *    complete: an unrecorded event is completed immediately).
 *
 * The seam runs after the step's draft passes were issued (`issue_serial_drafter`) and after
 * `issue_ahead`, immediately before `collect_step` -- the step's one wait. It therefore sees
 * `complete` as the reset left it (false, nothing has waited yet), the issuance record as the step
 * path set it, and the watermark *before* this step's finish. The row-less branch's wait is exactly
 * what the seam is in front of; it is read at the next seam, as prior_complete and as the watermark
 * step that wait made possible. The trace is thus a sequence of pre-finish values, and a step's own
 * finish is judged by comparing its value with the next one's.
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
    bool                    serving  = false;  /* the engine reported the transport running */
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

/* The engine's readiness report, from its watcher thread: the transport's port is nonzero, so a
 * stop is one the server accepts from here on. It records and nothing else -- the stop itself is
 * raised by the case, once. wait_serving() polls the flag; the report is one write and the wait's
 * slice is 5 ms, so there is no notification to lose. */
extern "C" void rad_test_serving_ready(int port) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (port > 0) g_driver.serving = true;
}

namespace {

/* Both endpoints of the history watermark, on one engine started for real.
 *
 * Prompt 192 tokens against 64 a step: two mid-prompt chunks (no logit row, but each with the
 * history pass it owes), then the tail -- the third chunk, which commits the first row -- then
 * decode steps. Every chunk's finish moves the history watermark; the step after the tail is the
 * withheld one. */
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

    /* The chunk size the prefill assertions below rely on: a row-less chunk's finish moves the
     * watermark by exactly the history it covered, which is this many tokens. */
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
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = nullptr;
        tr = g_driver.trace;
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

    /* The mid-prompt chunks: no logit row, and yet round 0's pass was built -- the scheduler keeps
     * one round when no row drafts because that round is the history pass the chunk still owes. At
     * the seam the wait has not run (complete is the reset's false); the repaired row-less branch
     * awaits the end, and the *next* seam is where that shows: prior_complete is the completion it
     * set, and the watermark has moved by the chunk that pass covered. Asserting the flag without
     * this pair would only restate the seam's own reading. */
    for (size_t i = 0; i + 1 < iw; ++i) {
        CHECK_EQ(tr[i].rows, 0);
        CHECK(tr[i].hist_issued);
        CHECK(!tr[i].complete);
        CHECK(tr[i + 1].prior_complete);
        CHECK_EQ(tr[i + 1].watermark, tr[i].watermark + kChunk);
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
