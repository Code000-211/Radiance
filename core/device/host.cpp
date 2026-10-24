/* host.cpp -- the host device backend.
 *
 * A primary development target, not a fallback: without ROCm the whole engine (builder, planner,
 * mover, scheduler, KV manager, tokeniser, sampler host reference, server) runs and is tested
 * through these calls (docs/IMPLEMENTATION.md). It is also where RAD_DOMAIN_HOST kernels run when
 * the planner puts a layer on the CPU (spec §5.1).
 *
 * The streams are real: each is an in-order queue with its own worker thread, and an event is a
 * sequence number the stream reaches. Inline "async" calls would let ordering bugs (a missing
 * rad_event_wait, a buffer reused before a copy landed) pass here and corrupt a real device.
 *
 * No allocation on the step path: the queue is a fixed-capacity ring of POD tasks, not
 * std::function, which would heap-allocate inside rad_memcpy_async.
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

/* Every allocation kind is page-aligned host memory. 4096 is RAD_ALIGN_UNIT and the O_DIRECT
 * buffer alignment (directio.cpp), so pools here are file-tier targets without a bounce. */
constexpr int64_t kAlign = 4096;

/* Per-stream queue depth: enough that a layer's mover transfers plus compute issues never hit
 * backpressure, small enough for a ring of a few tens of KiB. Enqueue onto a full ring blocks, as
 * on a saturated hardware queue. */
constexpr size_t kQueueDepth = 1024;

/* libr4d all-reduce supports at most 8 ranks (spec §9), so more fake devices are useless. */
constexpr int kMaxFakeDevices = 8;

/* ------------------------------------------------------------------ events */

/* An event is two sequence numbers: `recorded` advances on the host at rad_event_record,
 * `completed` on the stream worker when it reaches that record. Complete when completed has caught
 * up, so re-recording makes it pending again, as in HIP.
 *
 * `refs` counts the handle plus every queued task naming the event. HIP allows destroying an event
 * whose record or wait is still queued; freeing it at destroy would leave the worker locking a
 * freed mutex and wedge the stream. */
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

/* The last reference, handle or task, frees it. */
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
    int      gates     = 0;            /* closed gates (rad_stream_gate) queued on this stream */

    std::thread worker;
};

void stream_run(HostStream* s);

/* Push one task, blocking while the ring is full. Fails only behind a closed gate: the ring cannot
 * drain until the host opens it, and the host is the caller. Destroy queues its stop behind
 * submitted work, so there is no closing state. */
int stream_push(HostStream* s, const Task& t) {
    std::unique_lock<std::mutex> lk(s->m);
    if (s->count >= s->ring.size() && s->gates > 0) {
        device_set_error("a submission onto a full stream behind a closed gate: open the gate "
                         "(rad_gate_open) first");
        return RAD_E_STATE;
    }
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
                /* a = dpitch, b = spitch, c = width bytes, d = height rows. */
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
                /* The stream blocks, not a host thread (spec §5.4). Waiting on an event recorded
                 * later on this same stream wedges it, as it would a hardware queue. */
                std::unique_lock<std::mutex> lk(t.ev->m);
                t.ev->cv.wait(lk, [&t] { return t.ev->completed >= t.seq; });
                break;
            }

            case Task::Stop:
                done = true;
                break;
        }
        /* After the event lock is released: this may be the last reference. */
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

/* Per-device VRAM accounting, so vram_free moves and the planner's fit checks are exercised
 * (spec §5.3). */
std::atomic<int64_t> g_used[kMaxFakeDevices];

/* A mutex-guarded map is fine: allocation happens at load and pool construction, never inside
 * rad_arch_step. */
void alloc_note(void* p, int64_t bytes, int kind, int device) {
    std::lock_guard<std::mutex> lk(alloc_mutex());
    alloc_table()[p] = AllocRec{bytes, kind, device};
}

/* The record of allocation `p`, removed from the table when `take`. */
bool alloc_find(void* p, AllocRec* out, bool take = false) {
    std::lock_guard<std::mutex> lk(alloc_mutex());
    auto it = alloc_table().find(p);
    if (it == alloc_table().end()) return false;
    *out = it->second;
    if (take) alloc_table().erase(it);
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
    /* Memoised so the count cannot change under a rank thread. RADIANCE_HOST_DEVICES fakes
     * several devices in one address space for testing tensor parallel without cards (spec §9). */
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

/* Per-thread, matching hipSetDevice: one thread per rank, one device per rank (spec §1). */
thread_local int t_device = 0;

int64_t physical_ram() {
    long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return 8ll << 30;
    return (int64_t)pages * (int64_t)page;
}

/* The fake per-device VRAM budget. Read on every call so a test can change it. Default: half of
 * physical RAM, split across the fake devices so their sum is loadable. */
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

    /* Hardware threads split across fake devices, like the budget. */
    unsigned hw = std::thread::hardware_concurrency();
    out->n_cu = (int)((hw ? hw : 1u) / (unsigned)host_count());
    if (out->n_cu < 1) out->n_cu = 1;

    /* A host lane is one element wide. */
    out->warp_size = 1;

    out->vram_bytes = host_vram_bytes();
    int64_t used = g_used[device].load(std::memory_order_relaxed);
    out->vram_free = out->vram_bytes > used ? out->vram_bytes - used : 0;

    /* A CPU has no LDS. Nonzero because budget arithmetic divides by it; a conventional
     * per-workgroup figure, not a limit host kernels should honour. */
    out->lds_bytes = 64 * 1024;

    /* All fake devices share one address space. */
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
    /* All kinds are the same memory here, but the kind is recorded so mismatched frees are
     * caught. The size is rounded up to whole pages so DirectFile can read an aligned superset
     * into it without overrunning. */
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
    if (!alloc_find(p, &r, /*take=*/true)) {
        /* Not ours: freeing it would corrupt the heap, so refuse and warn. */
        RAD_WARN("rad_dev_free(%p, kind=%d): pointer was not allocated by the device layer",
                 p, kind);
        return;
    }
    if (r.kind != kind)
        RAD_WARN("rad_dev_free(%p): freed as kind %d, allocated as kind %d", p, kind, r.kind);
    if (r.kind == RAD_MEM_DEVICE) g_used[r.device].fetch_sub(r.bytes, std::memory_order_relaxed);
    std::free(p);
}

/* Host and device views are the same pointer (spec §5.1 zero-copy is trivial here). Null for a
 * pointer the device layer did not allocate. */
void* h_host_ptr(void* p) {
    if (!p) return nullptr;
    AllocRec r;
    return alloc_find(p, &r) ? p : nullptr;
}
void* h_device_ptr(void* p) { return h_host_ptr(p); }

int h_stream_create(RadStream* out, int high_priority) {
    if (!out) return RAD_E_INVAL;
    /* Priority is ignored: raising a thread's scheduling class needs privileges and buys nothing
     * on a one-thread queue. */
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
    /* Queued behind submitted work, so destroy drains rather than discards. */
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
    if (s->gates > 0) {
        device_set_error("rad_stream_sync: the stream waits at a closed gate, which only "
                         "rad_gate_open releases");
        return RAD_E_STATE;
    }
    /* Waits only for work submitted before this call, as hipStreamSynchronize does; the
     * scheduler relies on sync returning while another thread keeps pushing. */
    const uint64_t target = s->submitted;
    s->cv_done.wait(lk, [s, target] { return s->retired >= target; });
    return RAD_OK;
}

/* RAD_EVENT_LOCAL is meaningless here; RAD_EVENT_HOST_WAIT is kept so event_sync refuses events
 * created without it, as on a card. */
int h_event_create(RadEvent* out, unsigned flags) {
    if (!out) return RAD_E_INVAL;
    auto* e = new (std::nothrow) HostEvent();
    if (!e) { device_set_error("rad_event_create: out of memory"); return RAD_E_NOMEM; }
    e->host = (flags & RAD_EVENT_HOST_WAIT) != 0;
    *out = (RadEvent)e;
    return RAD_OK;
}

/* Drops the handle; queued tasks keep the event alive (see HostEvent). */
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
    /* A push the stream refuses queues nothing, so the task's reference is ours to drop again --
     * as rad_stream_gate does. Kept before the push so the worker cannot free the event first. */
    const int rc = stream_push((HostStream*)st, t);
    if (rc < 0) event_release(e);
    return rc;
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
    /* Waiting on a never-recorded event is a no-op, matching hipStreamWaitEvent; the mover relies
     * on it. */
    if (target == 0) return RAD_OK;
    Task t;
    t.kind = Task::Wait;
    t.ev   = e;
    t.seq  = target;
    event_retain(e);
    const int rc = stream_push((HostStream*)st, t);   /* the same drop as h_event_record */
    if (rc < 0) event_release(e);
    return rc;
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
    /* An event's stamp, if it was recorded and has completed. */
    auto stamp = [](RadEvent ev, Clock::time_point* t) {
        auto* e = (HostEvent*)ev;
        std::lock_guard<std::mutex> lk(e->m);
        if (e->recorded == 0 || e->completed < e->recorded) return false;
        *t = e->stamp;
        return true;
    };
    Clock::time_point ta, tb;
    if (!stamp(a, &ta) || !stamp(b, &tb)) return RAD_E_STATE;
    *out = std::chrono::duration<float, std::milli>(tb - ta).count();
    return RAD_OK;
}

/* A null stream is refused on both backends. HIP's stream 0 implicitly synchronises with every
 * blocking stream, which spec §5.4 forbids for transfers; accepting it here would hide such a
 * caller until it ran on a card. */
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
    /* The card's copy kernel moves 16-byte pieces (aql.cpp, a_memcpy_ranges_async), so the same
     * calls are refused here: a caller passing the host suite must run on a card. Whether the
     * memory is pinned cannot be told on this backend. */
    if (!dst || !src || prefix < 0 || (prefix & 15) || n < 0 || (n > 0 && !ranges)) {
        device_set_error("rad_memcpy_ranges_async(%p <- %p, prefix %lld (a multiple of 16), %d "
                         "ranges): invalid argument", dst, src, (long long)prefix, n);
        return RAD_E_INVAL;
    }
    if (prefix == 0 && n == 0) return RAD_OK;
    if (((uintptr_t)dst | (uintptr_t)src) & 15) {
        device_set_error("rad_memcpy_ranges_async(%p <- %p): both bases must be 16-byte aligned",
                         dst, src);
        return RAD_E_UNSUPPORTED;
    }
    Task t;
    t.kind = Task::Ranges;
    t.dst = dst;
    t.src = src;
    t.a = prefix;
    t.b = n;
    t.aux = ranges;
    return stream_push((HostStream*)st, t);
}

/* One copy task per segment, in list order. The list is read at the call, as on the device. */
int h_memcpy_list_async(const RadCopySeg* segs, int n, unsigned, RadStream st) {
    RAD_TRY(check_stream(st, "rad_memcpy_list_async"));
    if (n < 0 || (n > 0 && !segs)) {
        device_set_error("rad_memcpy_list_async: %d segments at %p", n, (const void*)segs);
        return RAD_E_INVAL;
    }
    for (int i = 0; i < n; ++i) RAD_TRY(h_memcpy_async(segs[i].dst, segs[i].src, segs[i].bytes, st));
    return RAD_OK;
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

/* ------------------------------------------------------------------ gates */
/* A gate is a wait task on an event whose one record only the host completes: recorded is 1 from
 * the start and rad_gate_open sets completed. The task holds a reference, so the event outlives
 * the handle until the worker has passed it. */
struct HostGate {
    HostEvent*  e = nullptr;
    HostStream* s = nullptr;
};

int h_stream_gate(RadStream st, void** out) {
    RAD_TRY(check_stream(st, "rad_stream_gate"));
    if (!out) return RAD_E_INVAL;
    auto* s = (HostStream*)st;
    auto* e = new (std::nothrow) HostEvent();
    auto* g = new (std::nothrow) HostGate();
    if (!e || !g) {
        delete e;
        delete g;
        device_set_error("rad_stream_gate: out of memory");
        return RAD_E_NOMEM;
    }
    e->recorded = 1;
    Task t;
    t.kind = Task::Wait;
    t.ev   = e;
    t.seq  = 1;
    event_retain(e);
    const int rc = stream_push(s, t);
    if (rc < 0) {
        event_release(e);
        event_release(e);
        delete g;
        return rc;
    }
    {
        std::lock_guard<std::mutex> lk(s->m);
        ++s->gates;
    }
    g->e = e;
    g->s = s;
    *out = g;
    return RAD_OK;
}

int h_gate_open(void* p) {
    auto* g = (HostGate*)p;
    if (!g || !g->e) return RAD_E_INVAL;
    {
        std::lock_guard<std::mutex> lk(g->e->m);
        g->e->completed = 1;
    }
    g->e->cv.notify_all();
    {
        std::lock_guard<std::mutex> lk(g->s->m);
        --g->s->gates;
    }
    event_release(g->e);
    delete g;
    return RAD_OK;
}

int64_t h_gate_room(RadStream st) {
    if (!st) return 0;
    auto* s = (HostStream*)st;
    std::lock_guard<std::mutex> lk(s->m);
    return (int64_t)(s->ring.size() - s->count);
}

/* ------------------------------------------------------------------ elastic memory */
/* Elastic memory via mmap, the same mechanism as the card's VMM: PROT_NONE reserves address space
 * without pages, MAP_FIXED read/write backs a granule, and PROT_NONE over it returns the pages.
 * Decommitted ranges come back zeroed and addresses never move, as on the HIP path.
 *
 * The granule is the pool alignment (the card's 2 MiB), not the 4 KiB page, so pool sizing that
 * passes here does not strand memory on a card. */
constexpr int64_t kVmemGranule = RAD_ALIGN_POOL;

int64_t h_vmem_granularity(void) { return kVmemGranule; }

void* h_vmem_reserve(int64_t bytes, int64_t align) {
    if (bytes <= 0) return nullptr;
    if (align < kVmemGranule) align = kVmemGranule;
    /* Over-reserve by one alignment and unmap the head and tail, so the range is exactly
     * [base, base+bytes) and release is one munmap with no side table. */
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
    /* MAP_FIXED rather than munmap: a hole in the reservation could be handed to an unrelated
     * mmap. */
    void* p = ::mmap(addr, (size_t)bytes, PROT_NONE,
                     MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p != addr) {
        device_set_error("vmem_decommit(%p, %lld): mmap failed", addr, (long long)bytes);
        return RAD_E_INVAL;
    }
    g_used[t_device].fetch_sub(bytes, std::memory_order_relaxed);
    return RAD_OK;
}

/* Commit already maps read/write, so this is a no-op mprotect. It is implemented anyway so the
 * host suite exercises the ranges VMem computes for the card. */
int h_vmem_protect(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    if (::mprotect(addr, (size_t)bytes, PROT_READ | PROT_WRITE) != 0) {
        device_set_error("vmem_protect(%p, %lld): mprotect failed", addr, (long long)bytes);
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int h_vram_used(int device, int64_t* out) {
    if (!out || device < 0 || device >= host_count()) return RAD_E_INVAL;
    *out = g_used[device].load(std::memory_order_relaxed);
    return RAD_OK;
}

const DeviceBackend g_host = {
    "host",
    h_count, h_set, h_props, h_enable_peer,
    h_alloc, h_free, h_host_ptr, h_device_ptr,
    h_stream_create, h_stream_destroy, h_stream_sync,
    h_event_create, h_event_destroy, h_event_record, h_event_wait,
    h_event_query, h_event_sync, h_event_elapsed_ms,
    h_host_wrote,
    h_memcpy_async, h_memcpy_2d_async, h_memset_async, h_memcpy_ranges_async,
    h_memcpy_list_async,
    nullptr, nullptr,   /* no DMA engine, so no link to hold */
    /* No recorded passes: host streams run work as issued. */
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,

    /* Elastic memory: see above. */
    h_vmem_granularity, h_vmem_reserve, h_vmem_release, h_vmem_commit, h_vmem_decommit,
    h_vmem_protect,

    h_vram_used,
    nullptr, nullptr,            /* no device code to register */
    nullptr, nullptr,            /* not integrated graphics; one address space */
    nullptr, nullptr, nullptr,   /* no PCI address, runtime or DMA engine */
    /* No expert prefetch. */
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,

    h_stream_gate, h_gate_open, h_gate_room,
};

}  /* namespace */

const DeviceBackend* host_backend(void) { return &g_host; }

}  /* namespace rad */
