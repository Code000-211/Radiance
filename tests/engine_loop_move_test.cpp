/* engine_loop_move_test.cpp -- the placement move a sampler-only release made belongs to the step
 * that release names.
 *
 * One engine, started for real on the host backend over a tiny Gemma 4 checkpoint with no drafter
 * (`--num-speculative-tokens 0`). Every decode step there is issued ahead: its forward pass runs
 * behind the previous step (`issue_ahead`, kIssueModel) and its tiering dispatch runs in the
 * following iteration's sampler-only release (kIssueSampler). With no drafting at all there is no
 * draft pass after that dispatch, so that release is the step's last chance to report the move --
 * the shared-KV/K0 shape, where a step has no draft pass of its own.
 *
 * The move is injected at the counter the rank loop reads (`rad_test_tiering_moves`): the host
 * fixture's plan has no tiered units, so no Mover can raise heat.stats() here. A note skipped in
 * the sampler release leaves `moved` false for a step that really moved a unit, and the step prices
 * itself as undisturbed (`kDraftSamplePlacement`).
 *
 * This binary is its own process: its signals, its threads and its crash are its own, and it is
 * the only one that defines the seam's strong hooks. Nothing listens beyond loopback:
 * `--host 127.0.0.1 --port 0`. The stop is bound to the transport's running state (the engine
 * reports Server::port() once it turns nonzero), exactly as engine_loop_test.cpp does it.
 */
#include "rad_test.h"
#include "tiny_model.h"

#include "device/device.h"
#include "engine.h"
#include "engine_test_peer.h"

/* serve() owns a StallWatch by unique_ptr; its deleter needs the complete type. */
#include "util/stallwatch.h"

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

/* One step as the seam sees it, and its evidence as Scheduler::step_finished received it. */
struct Rec {
    int  step        = 0;
    int  rows        = 0;      /* logit rows this step produced */
    bool hist_issued = false;  /* round 0's draft pass was built for this step */
};

struct EvRec {
    int  step  = 0;
    bool moved = false;        /* placement moved a unit inside the step's window */
};

struct Driver {
    std::mutex              mu;
    std::condition_variable cv;
    std::vector<Rec>        trace;
    std::vector<EvRec>      ev;
    /* Steps the tiering hook moved a unit for, at the release it happened in. */
    std::vector<int>        moved_steps;
    Engine*                 engine = nullptr;
    uint64_t                req_id = 1;
    bool                    serving = false;
    int                     submit_rc = RAD_OK;
};

Driver g_driver;

std::string trace_text() {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    std::string s = "   step  rows  hist\n";
    for (const Rec& r : g_driver.trace) {
        char line[64];
        std::snprintf(line, sizeof line, "  %4d  %4d  %4d\n", r.step, r.rows, (int)r.hist_issued);
        s += line;
    }
    return s;
}

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

/* The seam's strong definitions; the defaults in core/engine.cpp are weak so every other build of
 * those units links. */
extern "C" void rad_test_step_point(int step, int rows, bool, bool) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (!g_driver.engine) return;
    Rec r;
    r.step        = step;
    r.rows        = rows;
    r.hist_issued = EngineTestPeer::history_issued(*g_driver.engine);
    g_driver.trace.push_back(r);
    g_driver.cv.notify_all();
}

extern "C" void rad_test_step_evidence(int step, int, int, int, int moved) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (!g_driver.engine) return;
    g_driver.ev.push_back(EvRec{ step, moved != 0 });
}

/* The tiering hook: a placement move, injected at the counter the rank loop reads. The host
 * fixture's plan has no tiered units, so no Mover can raise heat.stats() here; this is the
 * controlled move. It lands only in the sampler-only release -- the release whose note used to be
 * skipped. */
extern "C" int64_t rad_test_tiering_moves(int rank, int32_t step, int what) {
    if (what != EngineTestPeer::kIssueSampler) return 0;
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (!g_driver.engine) return 0;
    g_driver.moved_steps.push_back(step);
    (void)rank;
    return 1;
}

/* One request, submitted on the serve thread after the scheduler is initialised and before any
 * rank or the stepper exists. Its sink is null: sink.h treats that as nobody listening. */
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

/* The engine's readiness report: the transport's port is nonzero, so a stop is one the server
 * accepts from here on. */
extern "C" void rad_test_serving_ready(int port) {
    std::lock_guard<std::mutex> lk(g_driver.mu);
    if (port > 0) g_driver.serving = true;
}

namespace {

/* A move the sampler-only release made is the step's own. Every decode step here is issued ahead
 * and none has a draft pass, so a note skipped in that release files the move under no step. */
TEST(a_move_the_sampler_only_release_made_belongs_to_its_step) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    radtest::skip("this build has HIP; the case needs the host backend guaranteed at compile time "
                  "(configure -DRAD_WITH_HIP=OFF)");
    return;
#endif
    setenv("ROCR_VISIBLE_DEVICES", "99", 1);
    setenv("RADIANCE_HOST_VRAM_MIB", "512", 1);

    tiny::TempDir dir("engine_loop_move");
    REQUIRE(dir.ok());
    tiny::Gemma4 s;                 /* no assistant head: no serial drafter, so no draft pass */
    REQUIRE(tiny::write_gemma4(dir.path, s));

    /* Prompt 192 tokens against 64 a step: two mid-prompt chunks, then the tail, then decode. */
    const int kChunk = 64;
    Argv argv({ "--model", dir.path,
                "--radiance-home", home(),
                "--kernels", "libref",
                "--debug-accept-reference-kernels",
                "--num-speculative-tokens", "0",
                "--max-num-batched-tokens", std::to_string(kChunk),
                "--host", "127.0.0.1",
                "--port", "0" });
    rad::Config cfg;
    REQUIRE_EQ(rad::config_parse(argv.argc(), argv.argv.data(), &cfg), RAD_OK);
    REQUIRE(rad::device_is_host());

    Engine e(std::move(cfg));
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = &e;
    }
    REQUIRE_EQ(e.start(), RAD_OK);

    std::promise<int> served;
    std::future<int>  done = served.get_future();
    std::thread serve_thread([&] {
        const int rc = e.serve(nullptr);
        served.set_value(rc);
    });
    if (!wait_serving(done, std::chrono::steady_clock::now() + std::chrono::seconds(60))) {
        std::fprintf(stderr, "engine_loop_move_test: the server never reported a running "
                             "transport%s:\n%s\n",
                     done.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready
                         ? " (serve() returned first)" : " within 60 s",
                     trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }

    /* Three decode steps are enough: every one of them is issued ahead and none has a draft pass. */
    {
        std::unique_lock<std::mutex> lk(g_driver.mu);
        g_driver.cv.wait_for(lk, std::chrono::seconds(60), [] {
            size_t live = 0;
            for (const Rec& r : g_driver.trace) if (r.rows > 0) ++live;
            return live >= 3 || g_driver.submit_rc != RAD_OK;
        });
    }
    if (done.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
        ::raise(SIGTERM) != 0) {
        std::fprintf(stderr, "engine_loop_move_test: raise(SIGTERM) failed although the server "
                             "reported itself running\n");
        std::fflush(stderr);
        std::_Exit(1);
    }
    if (done.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
        std::fprintf(stderr, "engine_loop_move_test: serve() did not return within 60 s:\n%s\n",
                     trace_text().c_str());
        std::fflush(stderr);
        std::_Exit(1);
    }
    serve_thread.join();

    std::vector<Rec>   tr;
    std::vector<EvRec> ev;
    std::vector<int>   moved;
    int                submit_rc = RAD_OK;
    {
        std::lock_guard<std::mutex> lk(g_driver.mu);
        g_driver.engine = nullptr;
        tr        = g_driver.trace;
        ev        = g_driver.ev;
        moved     = g_driver.moved_steps;
        submit_rc = g_driver.submit_rc;
    }

    if (submit_rc != RAD_OK) {
        std::fprintf(stderr, "engine_loop_move_test: the submission was refused (%s)\n",
                     rad_strerror(submit_rc));
        CHECK_EQ(submit_rc, RAD_OK);
        return;
    }
    REQUIRE(!moved.empty());
    REQUIRE(!ev.empty());

    size_t filed = 0;
    for (const EvRec& x : ev) {
        bool injected = false;
        for (int s : moved) injected = injected || s == x.step;
        /* Exactly the steps a sampler-only release moved a unit for carry `moved`; a move skipped
         * there is filed under no step at all. */
        CHECK_EQ(x.moved, injected);
        if (!injected) continue;
        ++filed;
        /* The shape that makes that note the only chance: a decode step with no draft pass. */
        const Rec* r = nullptr;
        for (const Rec& t : tr) if (t.step == x.step) { r = &t; break; }
        REQUIRE(r != nullptr);
        CHECK(r->rows > 0);
        CHECK(!r->hist_issued);
    }
    CHECK(filed >= 1);
}

}  /* namespace */

RAD_TEST_MAIN()
