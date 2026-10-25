/* engine_dtor_order_test.cpp -- ~Engine frees the buffers a drain may still be reading only after
 * the drain.
 *
 * Review finding at 8699a27: the destructor freed the pinned D2H targets (kld_host_,
 * pooled_host_, score_host) and the control-vector direction copies (cvec_views_[*].dir, device
 * memory and a kernel operand) before Ctx::shutdown synced the compute streams. A step aborted
 * between an asynchronous copy and its sync would free memory the card was still writing. The
 * repair moves both frees behind the drain; nothing else about teardown changes.
 *
 * What the case observes is the device allocator's registry, not a restatement of the new order.
 * The fixture queues an asynchronous copy from a pinned buffer on rank 0's compute stream and
 * hands both that buffer and a device direction to the destructor as their production owners do.
 * The destructor's test seam reports rad_dev_host_ptr() for both at two points: after the drain
 * returned (stage 0) and after the frees (stage 1). With the frees before the drain, stage 0
 * already finds them gone; with the repair they are live at stage 0 and gone only at stage 1. The
 * event recorded behind the copy must also read complete at stage 0, so the case shows the drain
 * really ran before the free point rather than assuming it.
 *
 * Host only: the host backend's streams are worker threads with a real sync, and its allocator
 * registry answers host_ptr after free. With HIP compiled in the backend may be a card, so the
 * case skips (configure -DRAD_WITH_HIP=OFF). No model, no plugin.
 */
#include "rad_test.h"

#include "engine.h"
#include "engine_test_peer.h"

#include "rad_device.h"
#include "build/rad_build.h"

#include <cstdlib>
#include <vector>

/* Only FakeKV and FakePC: this binary runs on the real device layer, as the engine does. */
#define SCHED_FAKES_KV_ONLY
#include "sched_fakes.h"
#include "sched_program.h"

namespace {

using rad::Engine;
using rad::EngineTestPeer;

struct HostProgram {
    rad::Program       prog;
    rad::Config        cfg;
    sched_fake::FakeKV kv;
    sched_fake::FakePC pc;
    std::vector<void*> owned;

    ~HostProgram() { for (void* p : owned) std::free(p); }
};

/* The fixture's one op is never issued here (no rank pass runs), but Ctx::bind_ops places an op
 * with no device launch on the host domain and then refuses it for having no host kernel. A
 * launch that never runs keeps it where sched_program.h declares it. */
int never_launched(const RadArgs*, RadStream) { return RAD_OK; }

/* Arena offsets are the buffer plan's, which this fixture does not run; Ctx::bind_arena refuses a
 * buffer without one. Lay them out in declaration order. */
void assign_arena(rad::Program& prog) {
    int64_t off = 0;
    for (size_t i = 1; i < prog.buffers.size(); ++i) {
        prog.buffers[i].arena_offset = off;
        off += (prog.buffers[i].bytes + 63) & ~(int64_t)63;
    }
    prog.arena_bytes = off;
}

/* The destructor seam's evidence, written inside ~Engine. */
void*    g_pinned = nullptr;   /* the rank's pinned score target */
void*    g_dir    = nullptr;   /* the rank's control-vector direction (device) */
RadEvent g_copied = nullptr;   /* recorded behind a copy from g_pinned */
int      g_stage[2] = {0, 0};
bool     g_live[2]  = {false, false};
bool     g_dir_live[2] = {false, false};
int      g_copy_at_drain = -1;

}  /* namespace */

/* Strong definition: rad_test_engine carries only the weak default. */
extern "C" void rad_test_dtor_stage(int stage) {
    if (stage < 0 || stage > 1) return;
    ++g_stage[stage];
    g_live[stage]     = g_pinned && rad_dev_host_ptr(g_pinned) != nullptr;
    g_dir_live[stage] = g_dir && rad_dev_host_ptr(g_dir) != nullptr;
    if (stage == 0) g_copy_at_drain = rad_event_query(g_copied);
}

TEST(the_destructor_frees_card_buffers_only_after_the_compute_drain) {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    /* The fixture allocates through the device layer. With HIP compiled in, select_backend() may
     * bind a card before this test can say otherwise; the guarantee this case needs is the
     * compile-time one. */
    SKIP("this build has HIP; the case needs the host backend guaranteed at compile time "
         "(configure -DRAD_WITH_HIP=OFF)");
#endif
    HostProgram h;
    sched_fake::build_program(h.prog, h.cfg, h.kv, h.pc, h.owned,
                              /*blocks*/ 4096, /*states*/ 64, /*max_tok*/ 512, /*max_ctx*/ 4096);
    sched_fake::kGdnKi.launch = &never_launched;
    assign_arena(h.prog);

    void* dst = nullptr;
    {
        Engine e(h.cfg);
        REQUIRE_EQ(EngineTestPeer::bind_rank(e, 0, h.prog), RAD_OK);
        RadStream s = EngineTestPeer::rank_stream(e, 0);
        REQUIRE(s != nullptr);

        g_pinned = rad_dev_alloc(4096, RAD_MEM_HOST_PINNED);
        REQUIRE(g_pinned != nullptr);
        EngineTestPeer::set_score_host(e, 0, (float*)g_pinned);

        g_dir = rad_dev_alloc(4096, RAD_MEM_DEVICE);
        REQUIRE(g_dir != nullptr);
        EngineTestPeer::set_cvec_dir(e, 0, (const float*)g_dir);

        dst = rad_dev_alloc(4096, RAD_MEM_DEVICE);
        REQUIRE(dst != nullptr);
        REQUIRE_EQ(rad_event_create(&g_copied), RAD_OK);
        REQUIRE_EQ(rad_memcpy_async(dst, g_pinned, 4096, s), RAD_OK);
        REQUIRE_EQ(rad_event_record(g_copied, s), RAD_OK);
    }   /* ~Engine runs here: drain, stage 0, frees, stage 1. */

    CHECK_EQ(g_stage[0], 1);
    CHECK_EQ(g_stage[1], 1);
    /* What the repair is for: the allocations outlive the drain ... */
    CHECK(g_live[0]);
    CHECK(g_dir_live[0]);
    CHECK_EQ(g_copy_at_drain, 1);
    /* ... and are actually freed after it. */
    CHECK(!g_live[1]);
    CHECK(!g_dir_live[1]);

    if (g_copied) rad_event_destroy(g_copied);
    if (dst) rad_dev_free(dst, RAD_MEM_DEVICE);
}

RAD_TEST_MAIN()
