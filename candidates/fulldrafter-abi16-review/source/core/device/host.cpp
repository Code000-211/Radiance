/* host.cpp -- the host device backend.
 *
 * This is a primary development target, not a fallback. On a machine with no ROCm the builder, the
 * planner, the mover, the scheduler, the KV manager, the tokeniser, the sampler's host reference
 * chain and the server all run and are tested through these fourteen calls
 * (docs/IMPLEMENTATION.md). It is also the code path a RAD_DOMAIN_HOST kernel runs on when the
 * planner puts a layer on the CPU, which is the fourth row of spec §5.1's table.
 *
 * The one design decision worth defending: **the streams are real.** Each stream is an in-order
 * work queue with its own worker thread, and an event is a sequence number that a stream reaches.
 * A backend whose "async" calls execute inline would let ordering bugs -- a missing
 * rad_event_wait, a buffer reused before the mover's copy landed -- pass every test here and
 * corrupt weights on a real device. The cost is one thread per stream and one condition-variable
 * wake per enqueue, which is nothing against the value of exercising the contract the HIP backend
 * actually has.
 *
 * The second: **no allocation on the step path.** The mover enqueues a transfer per weight per
 * layer dispatch, so the queue is a fixed-capacity ring of POD tasks sized at stream creation, not
 * a deque of std::function. A std::function holding a destination, a source and a length does not
 * fit libstdc++'s small-object buffer and would call operator new inside rad_memcpy_async, which
 * is exactly the thing docs/IMPLEMENTATION.md forbids on the step path.
 */
#include "device.h"
#include "../rad_internal.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace rad {
namespace {

using Clock = std::chrono::steady_clock;

/* Every allocation kind maps to page-aligned host memory. 4096 is not arbitrary: it is
 * RAD_ALIGN_UNIT, the .rad container's movement unit, and it is what O_DIRECT demands of a buffer
 * (directio.cpp), so a pinned pool allocated here is directly usable as a file-tier landing zone
 * with no bounce. */
constexpr int64_t kAlign = 4096;

/* A stream's queue depth. Deep enough that a layer dispatch's worth of mover transfers plus the
 * compute stream's issues never touch the backpressure path; shallow enough that the ring is a
 * few tens of KiB. Enqueue onto a full ring blocks, which is honest -- a saturated hardware queue
 * does the same thing -- and it is the only place in this file a caller can be made to wait. */
constexpr size_t kQueueDepth = 1024;

/* Tensor-parallel world sizes past 8 are not served: libr4d supplies 2/4/8-rank all-reduce and
 * nothing wider (spec §9), so a fake host backend that claimed 16 cards would only ever produce a
 * declare-time miss further down. */
constexpr int kMaxFakeDevices = 8;

/* ------------------------------------------------------------------ events */

/* An event is a pair of sequence numbers. `recorded` advances on the host, at the moment
 * rad_event_record is called; `completed` advances on the stream's worker, when the queue reaches
 * that record. The event is complete when the second has caught the first, which is exactly HIP's
 * rule that re-recording an event makes it pending again.
 *
 * `refs` is one for the caller's handle plus one for every queued task that names the event.
 * HIP lets a caller destroy an event whose record or wait is still queued -- the call returns and
 * the resources go once the queue is past them -- and teardown code written against a card does
 * exactly that. Freeing the object at rad_event_destroy instead leaves those tasks pointing at
 * freed memory, and the worker that reaches one locks a mutex whose bytes the allocator has
 * overwritten: the stream wedges and the rad_stream_destroy behind it never returns. */
struct HostEvent {
    std::mutex              m;
    std::condition_variable cv;
    uint64_t                recorded  = 0;
    uint64_t                completed = 0;
    Clock::time_point       stamp{};        /* when the worker reached the most recent record */
    std::atomic<int>        refs{1};
    bool                    host = false;   /* RAD_EVENT_HOST_WAIT */
};

void event_retain(HostEvent* e) { e->refs.fetch_add(1, std::memory_order_relaxed); }

/* The last reference out frees it, whether that is the handle or the last task naming it. */
void event_release(HostEvent* e) {
    if (e->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete e;
}

/* ------------------------------------------------------------------ tasks */

struct Task {
    enum Kind : uint8_t { Copy, Copy2D, Ranges, Memset, Record, Wait, Stop };
    Kind        kind = Stop;
    void*       dst  = nullptr;
    const void* src  = nullptr;
    int64_t     a = 0, b = 0, c = 0, d = 0;
    const void* aux = nullptr;
    HostEvent*  ev  = nullptr;
    uint64_t    seq = 0;
};

/* ------------------------------------------------------------------ streams */

struct HostStream {
    std::mutex              m;
    std::condition_variable cv_work;   /* worker waits on this for a task or a stop */
    std::condition_variable cv_room;   /* producers wait on this when the ring is full */
    std::condition_variable cv_done;   /* rad_stream_sync waits on this */

    std::vector<Task> ring;
    size_t            head = 0;        /* next slot to pop  */
    size_t            tail = 0;        /* next slot to push */
    size_t            count = 0;

    uint64_t submitted = 0;            /* tasks ever pushed */
    uint64_t retired   = 0;            /* tasks ever finished executing */

    std::thread worker;
};

void stream_run(HostStream* s);

/* Push one task, blocking while the ring is full; see kQueueDepth. Always succeeds -- there is no
 * "stream is closing" state to report, because rad_stream_destroy queues its stop behind the work
 * already submitted rather than racing it. */
int stream_push(HostStream* s, const Task& t) {
    std::unique_lock<std::mutex> lk(s->m);
    s->cv_room.wait(lk, [s] { return s->count < s->ring.size(); });
    s->ring[s->tail] = t;
    s->tail = (s->tail + 1) % s->ring.size();
    ++s->count;
    ++s->submitted;
    lk.unlock();
    s->cv_work.notify_one();
    return RAD_OK;
}

void stream_run(HostStream* s) {
    for (;;) {
        Task t;
        {
            std::unique_lock<std::mutex> lk(s->m);
            s->cv_work.wait(lk, [s] { return s->count > 0; });
            t = s->ring[s->head];
            s->head = (s->head + 1) % s->ring.size();
            --s->count;
        }
        s->cv_room.notify_one();

        bool done = false;
        switch (t.kind) {
            case Task::Copy:
                std::memcpy(t.dst, t.src, (size_t)t.a);
                break;

            case Task::Copy2D:
                /* a = dpitch, b = spitch, c = width bytes, d = height rows. Row by row, because
                 * the pitches are what make a 2D copy a 2D copy: a KV block's rows are strided by
                 * the pool's row pitch, not packed. */
                for (int64_t r = 0; r < t.d; ++r)
                    std::memcpy((uint8_t*)t.dst + r * t.a, (const uint8_t*)t.src + r * t.b,
                                (size_t)t.c);
                break;

            case Task::Ranges: {
                /* a = prefix bytes, b = range count; the list is read now, as the device's is. */
                std::memcpy(t.dst, t.src, (size_t)t.a);
                const RadCopyRange* r = (const RadCopyRange*)t.aux;
                for (int64_t i = 0; i < t.b; ++i)
                    std::memcpy((uint8_t*)t.dst + r[i].dst, (const uint8_t*)t.src + r[i].src,
                                (size_t)r[i].bytes);
                break;
            }

            case Task::Memset:
                std::memset(t.dst, (int)t.a, (size_t)t.b);
                break;

            case Task::Record: {
                std::lock_guard<std::mutex> lk(t.ev->m);
                if (t.seq > t.ev->completed) {
                    t.ev->completed = t.seq;
                    t.ev->stamp     = Clock::now();
                }
                t.ev->cv.notify_all();
                break;
            }

            case Task::Wait: {
                /* The worker blocks here, which is the whole point: no host thread synchronises,
                 * the *stream* does (spec §5.4). A wait whose event is only ever recorded later on
                 * this same stream wedges the stream -- which is precisely what the same construction
                 * does to a hardware queue, so it is a bug to catch, not a case to smooth over. */
                std::unique_lock<std::mutex> lk(t.ev->m);
                t.ev->cv.wait(lk, [&t] { return t.ev->completed >= t.seq; });
                break;
            }

            case Task::Stop:
                done = true;
                break;
        }
        /* After the task's lock has gone out of scope: this may be the last reference. */
        if (t.ev) event_release(t.ev);

        {
            std::lock_guard<std::mutex> lk(s->m);
            ++s->retired;
        }
        s->cv_done.notify_all();
        if (done) return;
    }
}

/* ------------------------------------------------------------------ allocation bookkeeping */

struct AllocRec {
    int64_t bytes = 0;
    int     kind  = 0;
    int     device = 0;
};

std::mutex&                               alloc_mutex() { static std::mutex m; return m; }
std::unordered_map<void*, AllocRec>&      alloc_table() {
    static std::unordered_map<void*, AllocRec> t;
    return t;
}

/* Per-device VRAM accounting, so rad_dev_props reports a vram_free that moves. The placement
 * planner's "did it fit" path is one of the things the host backend exists to exercise, and a
 * constant vram_free would let a plan that overcommits a real card pass here (spec §5.3). */
std::atomic<int64_t> g_used[kMaxFakeDevices];

/* A map lookup under a mutex on every alloc and free is fine and stays fine: allocation happens at
 * load and at pool construction, never inside rad_arch_step. If that ever changes, it is a
 * declare-phase bug and this mutex is where it will show up. */
void alloc_note(void* p, int64_t bytes, int kind, int device) {
    std::lock_guard<std::mutex> lk(alloc_mutex());
    alloc_table()[p] = AllocRec{bytes, kind, device};
}

bool alloc_take(void* p, AllocRec* out) {
    std::lock_guard<std::mutex> lk(alloc_mutex());
    auto it = alloc_table().find(p);
    if (it == alloc_table().end()) return false;
    *out = it->second;
    alloc_table().erase(it);
    return true;
}

bool alloc_find(void* p, AllocRec* out) {
    std::lock_guard<std::mutex> lk(alloc_mutex());
    auto it = alloc_table().find(p);
    if (it == alloc_table().end()) return false;
    *out = it->second;
    return true;
}

/* ------------------------------------------------------------------ devices */

int64_t env_i64(const char* key, int64_t dflt) {
    const char* v = std::getenv(key);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    long long n = std::strtoll(v, &end, 10);
    if (end == v || n < 0) {
        RAD_WARN("%s=\"%s\" is not a non-negative integer; using %lld", key, v, (long long)dflt);
        return dflt;
    }
    return (int64_t)n;
}

int host_count() {
    /* Memoised: the device count must not change under a rank thread that already called
     * rad_dev_set. RADIANCE_HOST_DEVICES is how tensor-parallel code gets tested without two
     * cards -- two ranks, two budgets, one address space (spec §9). */
    static int n = [] {
        int64_t v = env_i64("RADIANCE_HOST_DEVICES", 1);
        if (v < 1) v = 1;
        if (v > kMaxFakeDevices) {
            RAD_WARN("RADIANCE_HOST_DEVICES=%lld clamped to %d", (long long)v, kMaxFakeDevices);
            v = kMaxFakeDevices;
        }
        if (v > 1) RAD_INFO("device: host backend presenting %lld fake devices", (long long)v);
        return (int)v;
    }();
    return n;
}

/* Per-thread, matching hipSetDevice: spec §1 gives one thread per tensor-parallel rank, each
 * owning one device, so the current device is a property of the rank thread and never of the
 * process. */
thread_local int t_device = 0;

int64_t physical_ram() {
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 8ll << 30;
    return (int64_t)pages * (int64_t)page;
}

/* The budget the placement planner plans against. Read on every call rather than memoised: props
 * is not on the step path, and re-reading is what lets a test set the budget and see it.
 *
 * The default is half of physical RAM, divided by the fake device count -- a two-"card" host
 * backend that each claimed half the machine would let a plan pass here that cannot be loaded, and
 * the planner is exactly the component this budget exists to falsify. */
int64_t host_vram_bytes() {
    int64_t mib = env_i64("RADIANCE_HOST_VRAM_MIB", 0);
    if (mib > 0) return mib << 20;
    return (physical_ram() / 2) / host_count();
}

const char* cpu_model() {
    static std::string s = [] {
        std::string out = "host cpu";
        FILE* f = std::fopen("/proc/cpuinfo", "r");
        if (!f) return out;
        char line[512];
        while (std::fgets(line, sizeof line, f)) {
            if (std::strncmp(line, "model name", 10) != 0) continue;
            const char* c = std::strchr(line, ':');
            if (!c) break;
            ++c;
            while (*c == ' ' || *c == '\t') ++c;
            out.assign(c);
            while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
            break;
        }
        std::fclose(f);
        return out;
    }();
    return s.c_str();
}

/* ------------------------------------------------------------------ the fourteen calls */

int h_count(void) { return host_count(); }

int h_set(int device) {
    if (device < 0 || device >= host_count()) {
        device_set_error("rad_dev_set(%d): host backend has %d device(s)", device, host_count());
        return RAD_E_INVAL;
    }
    t_device = device;
    return RAD_OK;
}

int h_props(int device, RadDeviceProps* out) {
    if (!out) return RAD_E_INVAL;
    if (device < 0 || device >= host_count()) {
        device_set_error("rad_dev_props(%d): host backend has %d device(s)", device, host_count());
        return RAD_E_INVAL;
    }
    std::memset(out, 0, sizeof *out);
    snprintf(out->name, sizeof out->name, "%s", cpu_model());
    snprintf(out->arch, sizeof out->arch, "host");
    out->device_id = device;

    /* Threads, divided the same way the budget is: a fake two-device backend on a 32-core box is
     * two 16-"CU" devices, so a kernel plugin's occupancy arithmetic sees a coherent machine
     * rather than two copies of the whole one. */
    unsigned hw = std::thread::hardware_concurrency();
    out->n_cu = (int)((hw ? hw : 1u) / (unsigned)host_count());
    if (out->n_cu < 1) out->n_cu = 1;

    /* A host lane is one element wide. Any code that divides a tile by warp_size gets the honest
     * answer for a scalar backend. */
    out->warp_size = 1;

    out->vram_bytes = host_vram_bytes();
    int64_t used = g_used[device].load(std::memory_order_relaxed);
    out->vram_free = out->vram_bytes > used ? out->vram_bytes - used : 0;

    /* There is no LDS on a CPU, and this is a stand-in with one job. Budget arithmetic of the
     * form "how many tiles fit in LDS" divides by this, and a zero would trap on the backend
     * whose whole purpose is to run that arithmetic on a machine with no card. The figure is a
     * conventional per-workgroup scratch budget, not a property of this machine: host kernels are
     * not LDS-bound and must not read it as a limit. */
    out->lds_bytes = 64 * 1024;

    /* Every fake device shares one address space, so peer access is not merely available, it is
     * the only thing there is. */
    out->supports_p2p = 1;
    out->is_host_backend = 1;
    return RAD_OK;
}

int h_enable_peer(int self, int peer) {
    if (self < 0 || self >= host_count() || peer < 0 || peer >= host_count()) {
        device_set_error("rad_dev_enable_peer(%d, %d): host backend has %d device(s)",
                         self, peer, host_count());
        return RAD_E_INVAL;
    }
    return RAD_OK;   /* one address space; nothing to enable */
}

void* h_alloc(int64_t bytes, int kind) {
    if (bytes <= 0) {
        device_set_error("rad_dev_alloc(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    if (kind < RAD_MEM_DEVICE || kind > RAD_MEM_HOST_MAPPED) {
        device_set_error("rad_dev_alloc: unknown memory kind %d", kind);
        return nullptr;
    }
    void* p = nullptr;
    /* All four kinds are the same aligned host memory here. The distinction still matters and is
     * still recorded, because the planner reasons in kinds and on a real device rad_dev_free has
     * to give the right one back; the host backend's job is to make that code run, not to make the
     * distinction vanish. */
    /* Rounded up to a whole page as well as aligned to one, so a buffer handed to DirectFile can
     * receive the aligned superset of a misaligned extent without overrunning. That is the same
     * reason .rad pads its movement units to RAD_ALIGN_UNIT. */
    if (posix_memalign(&p, (size_t)kAlign, (size_t)align_up(bytes, kAlign)) != 0 || !p) {
        device_set_error("rad_dev_alloc(%lld, kind=%d): out of host memory",
                         (long long)bytes, kind);
        return nullptr;
    }
    int dev = t_device;
    alloc_note(p, bytes, kind, dev);
    if (kind == RAD_MEM_DEVICE) g_used[dev].fetch_add(bytes, std::memory_order_relaxed);
    return p;
}

void h_free(void* p, int kind) {
    if (!p) return;
    AllocRec r;
    if (!alloc_take(p, &r)) {
        /* Not ours. Freeing it would be a heap corruption three frames later, which is the class of
         * bug the plugin ABI's C-only rule exists to avoid; refuse loudly instead. */
        RAD_WARN("rad_dev_free(%p, kind=%d): pointer was not allocated by the device layer",
                 p, kind);
        return;
    }
    if (r.kind != kind)
        RAD_WARN("rad_dev_free(%p): freed as kind %d, allocated as kind %d", p, kind, r.kind);
    if (r.kind == RAD_MEM_DEVICE) g_used[r.device].fetch_sub(r.bytes, std::memory_order_relaxed);
    std::free(p);
}

/* Both views are the same pointer: on this backend the zero-copy execution site of spec §5.1 is
 * degenerate, and that is the correct answer rather than a shortcut -- the bytes really are
 * reachable from both sides without a transfer. Null for a pointer the layer did not allocate,
 * which is what the ABI's "null if the kind has no such view" means here. */
void* h_host_ptr(void* p) {
    if (!p) return nullptr;
    AllocRec r;
    return alloc_find(p, &r) ? p : nullptr;
}
void* h_device_ptr(void* p) { return h_host_ptr(p); }

int h_stream_create(RadStream* out, int high_priority) {
    if (!out) return RAD_E_INVAL;
    /* Priority is accepted and ignored. The mover and the compute stream ask for different
     * priorities on a real device and the same code has to run here; raising a thread's scheduling
     * class needs privileges this process does not have and would buy nothing on a queue that is
     * one thread deep. */
    (void)high_priority;
    auto* s = new (std::nothrow) HostStream();
    if (!s) { device_set_error("rad_stream_create: out of memory"); return RAD_E_NOMEM; }
    s->ring.resize(kQueueDepth);
    s->worker = std::thread(stream_run, s);
    *out = (RadStream)s;
    return RAD_OK;
}

void h_stream_destroy(RadStream st) {
    if (!st) return;
    auto* s = (HostStream*)st;
    Task t;
    t.kind = Task::Stop;
    /* Queued behind everything already submitted, so destroy drains rather than discards -- the
     * mover's last transfer must land before its stream goes away, or a weight is half-written and
     * nothing says so. */
    stream_push(s, t);
    if (s->worker.joinable()) s->worker.join();
    delete s;
}

int h_stream_sync(RadStream st) {
    if (!st) {
        device_set_error("rad_stream_sync: null stream");
        return RAD_E_INVAL;
    }
    auto* s = (HostStream*)st;
    std::unique_lock<std::mutex> lk(s->m);
    /* Snapshot: sync waits for work submitted *before* this call, not for work another thread
     * pushes while we wait. That is hipStreamSynchronize's contract and the scheduler depends on
     * it -- a sync that could never return under a busy mover would be a step-path stall. */
    const uint64_t target = s->submitted;
    s->cv_done.wait(lk, [s, target] { return s->retired >= target; });
    return RAD_OK;
}

/* RAD_EVENT_LOCAL means nothing in one memory; RAD_EVENT_HOST_WAIT is kept, so a host sync on an
 * event made without it is refused here as it is on a card. */
int h_event_create(RadEvent* out, unsigned flags) {
    if (!out) return RAD_E_INVAL;
    auto* e = new (std::nothrow) HostEvent();
    if (!e) { device_set_error("rad_event_create: out of memory"); return RAD_E_NOMEM; }
    e->host = (flags & RAD_EVENT_HOST_WAIT) != 0;
    *out = (RadEvent)e;
    return RAD_OK;
}

/* Drops the handle. A record or wait still queued keeps the event alive until the worker is past
 * it -- see HostEvent. */
void h_event_destroy(RadEvent ev) { if (ev) event_release((HostEvent*)ev); }

int h_event_record(RadEvent ev, RadStream st) {
    if (!ev || !st) {
        device_set_error("rad_event_record: null %s", ev ? "stream" : "event");
        return RAD_E_INVAL;
    }
    auto* e = (HostEvent*)ev;
    uint64_t seq;
    {
        std::lock_guard<std::mutex> lk(e->m);
        seq = ++e->recorded;      /* pending from this instant, on the host, before the push */
    }
    Task t;
    t.kind = Task::Record;
    t.ev   = e;
    t.seq  = seq;
    event_retain(e);
    return stream_push((HostStream*)st, t);
}

int h_event_wait(RadStream st, RadEvent ev) {
    if (!ev || !st) {
        device_set_error("rad_event_wait: null %s", ev ? "stream" : "event");
        return RAD_E_INVAL;
    }
    auto* e = (HostEvent*)ev;
    uint64_t target;
    {
        std::lock_guard<std::mutex> lk(e->m);
        target = e->recorded;
    }
    /* Waiting on an event that was never recorded is a no-op, matching hipStreamWaitEvent. The
     * mover relies on it: a weight with no transfer in flight has an event nobody recorded, and
     * making that case an error would put a branch on the issue path for nothing. */
    if (target == 0) return RAD_OK;
    Task t;
    t.kind = Task::Wait;
    t.ev   = e;
    t.seq  = target;
    event_retain(e);
    return stream_push((HostStream*)st, t);
}

int h_event_query(RadEvent ev) {
    if (!ev) return RAD_E_INVAL;
    auto* e = (HostEvent*)ev;
    std::lock_guard<std::mutex> lk(e->m);
    return e->completed >= e->recorded ? 1 : 0;
}

/* Host kernels read host memory directly, so there is nothing to make visible. */
void h_host_wrote(void) {}

int h_event_sync(RadEvent ev) {
    if (!ev) return RAD_E_INVAL;
    auto* e = (HostEvent*)ev;
    if (!e->host) {
        device_set_error("rad_event_sync: the event was made without RAD_EVENT_HOST_WAIT");
        return RAD_E_INVAL;
    }
    std::unique_lock<std::mutex> lk(e->m);
    const uint64_t target = e->recorded;
    e->cv.wait(lk, [e, target] { return e->completed >= target; });
    return RAD_OK;
}

int h_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    if (!a || !b || !out) return RAD_E_INVAL;
    auto* ea = (HostEvent*)a;
    auto* eb = (HostEvent*)b;
    Clock::time_point ta, tb;
    {
        std::lock_guard<std::mutex> lk(ea->m);
        if (ea->recorded == 0 || ea->completed < ea->recorded) return RAD_E_STATE;
        ta = ea->stamp;
    }
    {
        std::lock_guard<std::mutex> lk(eb->m);
        if (eb->recorded == 0 || eb->completed < eb->recorded) return RAD_E_STATE;
        tb = eb->stamp;
    }
    *out = std::chrono::duration<float, std::milli>(tb - ta).count();
    return RAD_OK;
}

/* A null stream is refused on both backends, and that is a decision rather than an oversight.
 * HIP's stream 0 carries legacy implicit-synchronisation semantics: work on it serialises against
 * every other blocking stream in the process. That is the exact opposite of spec §5.4's
 * requirement that a transfer involve no host synchronisation and not serialise the step, so the
 * engine never wants it, and letting it through here would let a caller pick it up by accident and
 * only find out on a real device. */
int check_stream(RadStream s, const char* who) {
    if (s) return RAD_OK;
    device_set_error("%s: null stream -- radiance does not use the default stream, "
                     "create one with rad_stream_create", who);
    return RAD_E_INVAL;
}

int h_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream st) {
    RAD_TRY(check_stream(st, "rad_memcpy_async"));
    if (bytes == 0) return RAD_OK;
    if (!dst || !src || bytes < 0) {
        device_set_error("rad_memcpy_async(%p <- %p, %lld): invalid argument",
                         dst, src, (long long)bytes);
        return RAD_E_INVAL;
    }
    Task t;
    t.kind = Task::Copy;
    t.dst = dst;
    t.src = src;
    t.a = bytes;
    return stream_push((HostStream*)st, t);
}

int h_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                      int64_t width, int64_t height, RadStream st) {
    RAD_TRY(check_stream(st, "rad_memcpy_2d_async"));
    if (width == 0 || height == 0) return RAD_OK;
    if (!dst || !src || width < 0 || height < 0 || dpitch < width || spitch < width) {
        device_set_error("rad_memcpy_2d_async: w=%lld h=%lld dpitch=%lld spitch=%lld "
                         "-- a pitch shorter than the row is a caller bug",
                         (long long)width, (long long)height,
                         (long long)dpitch, (long long)spitch);
        return RAD_E_INVAL;
    }
    Task t;
    t.kind = Task::Copy2D;
    t.dst = dst;
    t.src = src;
    t.a = dpitch;
    t.b = spitch;
    t.c = width;
    t.d = height;
    return stream_push((HostStream*)st, t);
}

int h_memcpy_ranges_async(void* dst, const void* src, int64_t prefix, const RadCopyRange* ranges,
                          int n, RadStream st) {
    RAD_TRY(check_stream(st, "rad_memcpy_ranges_async"));
    if (!dst || !src || prefix < 0 || n < 0 || (n > 0 && !ranges)) {
        device_set_error("rad_memcpy_ranges_async(%p <- %p, prefix %lld, %d ranges): invalid "
                         "argument", dst, src, (long long)prefix, n);
        return RAD_E_INVAL;
    }
    if (prefix == 0 && n == 0) return RAD_OK;
    Task t;
    t.kind = Task::Ranges;
    t.dst = dst;
    t.src = src;
    t.a = prefix;
    t.b = n;
    t.aux = ranges;
    return stream_push((HostStream*)st, t);
}

int h_memset_async(void* dst, int value, int64_t bytes, RadStream st) {
    RAD_TRY(check_stream(st, "rad_memset_async"));
    if (bytes == 0) return RAD_OK;
    if (!dst || bytes < 0) {
        device_set_error("rad_memset_async(%p, %lld): invalid argument", dst, (long long)bytes);
        return RAD_E_INVAL;
    }
    Task t;
    t.kind = Task::Memset;
    t.dst = dst;
    t.a = value & 0xff;
    t.b = bytes;
    return stream_push((HostStream*)st, t);
}

/* ------------------------------------------------------------------ elastic memory */
/* mmap is the host backend's answer to the card's virtual memory manager, and it is the same
 * mechanism rather than a simulation of one: PROT_NONE takes address space and no pages, MAP_FIXED
 * over it backs a granule, and PROT_NONE over that hands the pages back to the kernel. What a test
 * proves here -- that a decommitted range comes back zeroed, that the addresses never move, that
 * committing over a commit is the caller's bug -- is true of the HIP path for the same reasons.
 *
 * THE GRANULE IS THE POOL ALIGNMENT AND NOT THE PAGE SIZE. A host page is 4 KiB and a card's
 * allocation granularity is 2 MiB, so rounding to the page here would let a pool size that only
 * works on this backend pass its tests and then strand up to 2 MiB per stripe on the card. Matching
 * the coarser number is what makes the host suite able to hold the arithmetic down. */
constexpr int64_t kVmemGranule = RAD_ALIGN_POOL;

int64_t h_vmem_granularity(void) { return kVmemGranule; }

void* h_vmem_reserve(int64_t bytes, int64_t align) {
    if (bytes <= 0) return nullptr;
    if (align < kVmemGranule) align = kVmemGranule;
    /* Over-reserve by one alignment, then give back the head and the tail, so the range that
     * remains is exactly [base, base+bytes) and release() is a single munmap of what it was
     * handed. The alternative -- remembering the raw mapping in a side table -- is a table whose
     * only purpose is to survive until teardown. */
    const size_t raw_len = (size_t)(bytes + align);
    void* raw = ::mmap(nullptr, raw_len, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) {
        device_set_error("vmem_reserve(%lld): mmap failed", (long long)bytes);
        return nullptr;
    }
    char* base = (char*)align_up((int64_t)(uintptr_t)raw, align);
    const size_t head = (size_t)(base - (char*)raw);
    if (head) ::munmap(raw, head);
    const size_t tail = raw_len - head - (size_t)bytes;
    if (tail) ::munmap(base + bytes, tail);
    return base;
}

void h_vmem_release(void* base, int64_t bytes) {
    if (base && bytes > 0) ::munmap(base, (size_t)bytes);
}

int h_vmem_commit(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    void* p = ::mmap(addr, (size_t)bytes, PROT_READ | PROT_WRITE,
                     MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != addr) {
        device_set_error("vmem_commit(%p, %lld): mmap failed", addr, (long long)bytes);
        return RAD_E_NOMEM;
    }
    g_used[t_device].fetch_add(bytes, std::memory_order_relaxed);
    return RAD_OK;
}

int h_vmem_decommit(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    /* MAP_FIXED over the same range rather than munmap: munmap would leave a HOLE in the
     * reservation that an unrelated mmap could be handed, and the whole promise of this primitive
     * is that the address range stays the caller's until it releases it. */
    void* p = ::mmap(addr, (size_t)bytes, PROT_NONE,
                     MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p != addr) {
        device_set_error("vmem_decommit(%p, %lld): mmap failed", addr, (long long)bytes);
        return RAD_E_INVAL;
    }
    g_used[t_device].fetch_sub(bytes, std::memory_order_relaxed);
    return RAD_OK;
}

/* The host backend's commit already maps readable and writable, so this has nothing to grant. It
 * is implemented rather than left null because the run VMem computes for it is the same run the
 * card is handed, and a path that only ever runs on the machine with the card is a path whose
 * arithmetic the host suite cannot hold down. mprotect over an anonymous mapping that already has
 * these bits is a permission check and a no-op. */
int h_vmem_protect(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    if (::mprotect(addr, (size_t)bytes, PROT_READ | PROT_WRITE) != 0) {
        device_set_error("vmem_protect(%p, %lld): mprotect failed", addr, (long long)bytes);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int h_vram_used(int device, int64_t* out) {
    if (!out || device < 0 || device >= kMaxFakeDevices) return RAD_E_INVAL;
    *out = g_used[device].load(std::memory_order_relaxed);
    return RAD_OK;
}

int h_host_register(void*, int64_t, unsigned) { return RAD_OK; }
int h_host_unregister(void*) { return RAD_OK; }

const DeviceBackend g_host = {
    "host",
    h_count, h_set, h_props, h_enable_peer,
    h_alloc, h_free, h_host_ptr, h_device_ptr,
    h_host_register, h_host_unregister,
    h_stream_create, h_stream_destroy, h_stream_sync,
    h_event_create, h_event_destroy, h_event_record, h_event_wait,
    h_event_query, h_event_sync, h_event_elapsed_ms,
    h_host_wrote,
    h_memcpy_async, h_memcpy_2d_async, h_memset_async, h_memcpy_ranges_async,
    /* No recorded passes: a host stream runs its work as it is issued, so there is no queue to
     * write a recording into, and every pass is issued. */
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,

    /* Elastic memory, which the host backend DOES have: see above. */
    h_vmem_granularity, h_vmem_reserve, h_vmem_release, h_vmem_commit, h_vmem_decommit,
    h_vmem_protect,

    h_vram_used,
};

}  /* namespace */

const DeviceBackend* host_backend(void) { return &g_host; }

}  /* namespace rad */
