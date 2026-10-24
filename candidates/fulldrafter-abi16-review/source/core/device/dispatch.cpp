/* dispatch.cpp -- picks a backend once, then forwards.
 *
 * The only definition of every rad_dev_* / rad_stream_* / rad_event_* symbol in the engine.
 * host.cpp and hip.cpp export vtables and nothing else, so there is no #ifdef anywhere above this
 * file and no way for the two to end up implementing different contracts.
 *
 * The selection rule: a build with HIP still runs the host backend when the machine shows no
 * device. Having ROCm installed and having a card are separate facts -- a CI container with the
 * headers and no `/dev/kfd` is the normal case -- and the honest response is to say which backend
 * won at Info and carry on, not to fail at the first hipMalloc.
 */
#include "device.h"
#include "../rad_internal.h"
#include "../../arch/common/rad_kv_compact.h"

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
extern "C" int rad_aql_kv_compact(const radiance::RadKVCompactParams&, RadStream);
#endif

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>

namespace rad {

/* ------------------------------------------------------------------ the error slot */
/* Thread-local and drained by reading, which is the ABI's own contract. One thread per rank plus
 * the mover (spec §1), so a fault reported on the wrong thread would name the wrong half of the
 * engine to an operator who has to decide whether to restart. */
static thread_local std::string t_last_error;

void device_set_error(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    t_last_error.assign(buf);
    /* Debug rather than Error: most of these are answered by the caller (a pool that tries an
     * allocation, finds it does not fit and asks the planner for a smaller one). The ones that are
     * fatal are reported by whoever decided they were fatal, with context this layer does not have. */
    RAD_DEBUG("device: %s", buf);
}

/* ------------------------------------------------------------------ selection */

/* One line, listing every visible device by index and arch.
 *
 * This is not decoration. A machine reports more devices than it has accelerators: a workstation
 * with two discrete cards and an APU reports three, and the integrated one claims tens of GiB of
 * "VRAM" that is really system RAM. A --tp matching that count, or a planner that sums vram_bytes
 * across devices, is wrong in a way that shows up as a memory error much later -- and the arch
 * printed here is what tells the two kinds apart. The device layer does not filter -- which
 * devices are usable is the engine's decision and it has rad_dev_props to make it with -- but it
 * does not let the inventory go unsaid either (spec §16). */
static std::string device_inventory(const DeviceBackend* b) {
    std::string s;
    const int n = b->count();
    for (int i = 0; i < n; ++i) {
        RadDeviceProps p{};
        if (b->props(i, &p) < 0) continue;
        s += fmt("%s%d:%s", i ? " " : "", i, p.arch);
    }
    return s;
}

static const DeviceBackend* select_backend() {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    if (const DeviceBackend* b = aql_backend_if_present()) {
        RAD_INFO("device: AQL backend, %d device(s) [%s]", b->count(),
                 device_inventory(b).c_str());
        return b;
    }
    RAD_INFO("device: built with HIP but no device is visible -- using the host backend");
#endif
    const DeviceBackend* b = host_backend();
    RAD_INFO("device: host backend, %d device(s) [%s]", b->count(), device_inventory(b).c_str());
    return b;
}

/* std::call_once rather than a function-local static, because the answer is read from every rank
 * thread and the guard variable's acquire load is the cheaper of the two on the paths that matter
 * -- and because a backend chosen twice would be a backend chosen differently twice. */
static const DeviceBackend* g_backend = nullptr;
static std::once_flag       g_once;

static inline const DeviceBackend* dev() {
    std::call_once(g_once, [] { g_backend = select_backend(); });
    return g_backend;
}

bool        device_is_host()      { return std::strcmp(dev()->name, "host") == 0; }
const char* device_backend_name() { return dev()->name; }

int kv_compact_winning_branch(const radiance::RadKVCompactParams& p, RadStream stream) {
    if (p.n_moves <= 0) return RAD_OK;
    if (p.n_moves > RAD_KV_COMPACT_MAX_MOVES) return RAD_E_INVAL;
    bool active = false;
    for (int m = 0; m < p.n_moves; ++m)
        active |= p.moves[m].src_slot >= 0 && p.moves[m].dst_slot >= 0 &&
                  p.moves[m].src_slot != p.moves[m].dst_slot;
    if (!active) return RAD_OK;
    if (!stream || !p.kv_pool_base || p.block_size <= 0 || p.tok_bytes <= 0 ||
        p.n_layers <= 0 || p.n_heads <= 0) return RAD_E_INVAL;
    // Host streams have real workers too: inline copies would race earlier queued KV writes.
    if (device_is_host())
        return radiance::rad_kv_compact_winning_branch(
            p.kv_pool_base, p.layer_stride, p.kv_block_stride, p.kv_head_stride,
            p.block_size, p.tok_bytes, p.n_layers, p.n_heads, p.moves, p.n_moves, stream);
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    return rad_aql_kv_compact(p, stream);
#else
    return RAD_E_UNSUPPORTED;
#endif
}

}  /* namespace rad */

/* ------------------------------------------------------------------ the ledger
 *
 * EVERY DEVICE ALLOCATION THIS ENGINE MAKES, COUNTED, because without it an unaccounted allocation
 * can only be found by sampling /sys/class/drm through startup and correlating the steps against
 * the log by eye.
 *
 * The VRAM budget (core/mem/vram_budget.cpp) spends the card down to --gpu-headroom-mib, which
 * only works if what lands outside the two pools is known. Most of it is known by construction:
 * the activation arena is computed from the buffer plan, the step batch staging from the run
 * configuration. Everything else has to be counted here, because an allocation nobody counts is an
 * allocation that eats the headroom and then fails somebody else's.
 *
 * WHAT THIS DOES NOT SEE, and it is stated here because the number is only honest with the caveat
 * attached: a kernel plugin that calls hipMalloc itself goes nowhere near this file. libr4d does,
 * for its split-tail workspaces and its all-reduce peer scratch, and the engine's own accounting
 * cannot include them. Engine::start reads the card back after everything has allocated for
 * exactly that reason -- the ledger says what the core spent, the card says what was spent.
 *
 * RADIANCE_TRACE_ALLOC=N logs every allocation of at least N bytes as it happens. That is the
 * instrument; the counters are the summary. */
namespace rad {
namespace {

struct Ledger {
    std::atomic<int64_t> bytes{0};
    std::atomic<int64_t> count{0};
};
Ledger g_ledger[4];        /* indexed by RAD_MEM_*; see rad_dev_alloc below for the bound check */

/* Event handles, counted here and not in a backend so the number means the same on both. An
 * event a component forgets at teardown is reclaimed at exit and reported by nothing else, so
 * this is the one place such a leak shows. */
std::atomic<int64_t> g_live_events{0};

int64_t trace_floor() {
    static const int64_t v = [] {
        const char* e = std::getenv("RADIANCE_TRACE_ALLOC");
        if (!e || !*e) return (int64_t)-1;
        const long long x = std::atoll(e);
        return x >= 0 ? (int64_t)x : (int64_t)0;
    }();
    return v;
}

const char* kind_name(int kind) {
    switch (kind) {
        case RAD_MEM_DEVICE:      return "device";
        case RAD_MEM_HOST:        return "host";
        case RAD_MEM_HOST_PINNED: return "host-pinned";
        case RAD_MEM_HOST_MAPPED: return "host-mapped";
        default:                  return "?";
    }
}

}  /* namespace */

void rad_dev_alloc_totals(int kind, int64_t* bytes, int64_t* count) {
    if (kind < 0 || kind >= 4) { if (bytes) *bytes = 0; if (count) *count = 0; return; }
    if (bytes) *bytes = g_ledger[kind].bytes.load(std::memory_order_relaxed);
    if (count) *count = g_ledger[kind].count.load(std::memory_order_relaxed);
}

int64_t rad_dev_live_events(void) { return g_live_events.load(std::memory_order_relaxed); }


/* ------------------------------------------------------------------ elastic memory */
/* See device.h for what this is for. What is here is the interval arithmetic and the two
 * roundings; the backends do the mapping. */

namespace {
std::atomic<int64_t> g_vmem_committed{0};
}  /* namespace */

int64_t rad_dev_vmem_committed(void) { return g_vmem_committed.load(std::memory_order_relaxed); }

/* BOTH HALVES OF THE QUESTION. A backend can carry the entries and still be running on a device
 * whose driver has no virtual memory manager -- d_vmem_granularity answers 0 there -- and a pool
 * that asked only whether the function pointers existed would reserve an address range it could
 * never back, then fail at the first commit instead of sizing itself whole at startup. */
bool    VMem::supported()   { return dev()->vmem_reserve != nullptr && VMem::granularity() > 0; }
int64_t VMem::granularity() {
    if (!dev()->vmem_granularity) return 0;
    const int64_t g = dev()->vmem_granularity();
    return g > 0 ? g : 0;
}

int VMem::reserve(int64_t bytes, int64_t align) {
    release();
    if (bytes <= 0) return RAD_E_INVAL;
    if (!supported()) {
        device_set_error("elastic memory: the %s backend has none", device_backend_name());
        return RAD_E_UNSUPPORTED;
    }
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    if (align <= 0) align = g;
    /* ROUNDED UP TO A GRANULE, because the last granule of a range that is not a whole number of
     * them can never be committed and the bytes in it are then reserved and unreachable. */
    const int64_t want = align_up(bytes, g);
    void* p = dev()->vmem_reserve(want, align);
    if (!p) return RAD_E_NOMEM;
    base_ = p;
    size_ = want;
    committed_ = 0;
    ext_.clear();
    return RAD_OK;
}

void VMem::release() {
    if (!base_) { size_ = 0; committed_ = 0; ext_.clear(); return; }
    /* A range released with pages still in it leaks the pages and, on some drivers, refuses the
     * address free as well -- so the teardown order is fixed here rather than asked of callers.
     * Granule at a time, because that is how each of them was mapped. */
    const int64_t g = granularity();
    for (const auto& e : ext_) {
        if (dev()->vmem_decommit && g > 0)
            for (int64_t o = e.first; o < e.first + e.second; o += g)
                (void)dev()->vmem_decommit((char*)base_ + o, g);
        g_vmem_committed.fetch_sub(e.second, std::memory_order_relaxed);
    }
    if (dev()->vmem_release) dev()->vmem_release(base_, size_);
    base_ = nullptr;
    size_ = 0;
    committed_ = 0;
    ext_.clear();
}

bool VMem::is_committed(int64_t off, int64_t bytes) const {
    if (bytes <= 0) return true;
    if (off < 0 || off + bytes > size_) return false;
    for (const auto& e : ext_)
        if (off >= e.first && off + bytes <= e.first + e.second) return true;
    return false;                       /* extents are merged, so one of them covers it or none does */
}
/* WHAT TO MAP IS COMPUTED BEFORE ANYTHING IS MAPPED, and that is not a style choice. add_extent
 * MERGES, so the list this walk is reading rearranges itself under the walk the moment a gap is
 * filled -- the cursor then lands past the end of a list that just got shorter and the tail of the
 * request is mapped a second time. On the host backend a double map silently succeeds and only
 * the accounting is wrong; on a card the driver refuses it and a pool fails to grow.
 *
 * ONE GRANULE PER CALL, AND THE ACCESS OVER THE RUN. Both halves of that are the driver's rule
 * rather than a preference, and they pull in opposite directions:
 *
 *   A mapping can only be unbacked exactly as it was backed, so a run mapped in one call is a run
 *   that can only be released whole. Mapping the granules separately is what lets a pool hand back
 *   the top of its range while it keeps the rest, which is the entire purpose of an elastic pool.
 *
 *   Access, though, cannot be granted per granule: a granule that sits against an already-
 *   accessible one is refused on its own and accepted as part of the contiguous run containing
 *   both. So the maps are named one at a time and the access is named once, over every extent the
 *   new granules ended up part of -- which add_extent has already merged by then. Re-granting
 *   access over a range that has it is free and is what makes that simple. */
int VMem::commit(int64_t off, int64_t bytes) {
    if (!base_ || bytes <= 0) return bytes <= 0 ? RAD_OK : RAD_E_INVAL;
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    /* OUTWARD: the caller asked for these bytes and must get all of them. */
    const int64_t lo = (off < 0 ? 0 : off / g * g);
    int64_t hi = align_up(off + bytes, g);
    if (hi > size_) hi = size_;
    if (lo >= hi) return RAD_OK;

    std::vector<int64_t> want;
    want.reserve((size_t)((hi - lo) / g));
    for (int64_t o = lo; o < hi; o += g)
        if (!is_committed(o, g)) want.push_back(o);
    if (want.empty()) return RAD_OK;

    /* A failure part way through leaves what already mapped mapped AND RECORDED. Half a grow is a
     * smaller pool, which the caller can live with; half a grow the bookkeeping has forgotten is
     * memory this process can neither use nor return. */
    int rc = RAD_OK;
    size_t took = 0;
    for (; took < want.size(); ++took) {
        rc = dev()->vmem_commit((char*)base_ + want[took], g);
        if (rc < 0) break;
        add_extent(want[took], g);
    }

    /* AND THE ACCESS IS GRANTED EVEN WHEN THE GROW STOPPED SHORT, for the same reason. A pool that
     * got half of what it asked for is one the caller can live with; one whose new granules are
     * mapped and unreachable is not, and the granules that did map are in the run either way. */
    int prc = RAD_OK;
    if (took > 0 && dev()->vmem_protect) {
        for (const auto& e : ext_) {
            if (e.first + e.second <= lo || e.first >= hi) continue;
            prc = dev()->vmem_protect((char*)base_ + e.first, e.second);
            if (prc < 0) break;
        }
    }
    if (prc < 0) {
        /* MAPPED BUT UNREACHABLE IS WORSE THAN NOT MAPPED: the VRAM is spent and the first kernel
         * to touch it faults. Give back exactly what this call took and leave the rest, which is
         * reachable and was before. */
        for (size_t i = 0; i < took; ++i) {
            if (dev()->vmem_decommit) (void)dev()->vmem_decommit((char*)base_ + want[i], g);
            drop_extent(want[i], g);
        }
        return prc;
    }
    return rc;
}

int VMem::decommit(int64_t off, int64_t bytes) {
    if (!base_ || bytes <= 0) return bytes <= 0 ? RAD_OK : RAD_E_INVAL;
    const int64_t g = granularity();
    if (g <= 0) return RAD_E_UNSUPPORTED;
    /* INWARD: a granule holding one byte the caller still wants stays. */
    const int64_t lo = align_up(off < 0 ? 0 : off, g);
    int64_t hi = (off + bytes) / g * g;
    if (hi > size_) hi = size_;
    if (lo >= hi) return RAD_OK;

    /* Gathered first, for the same reason as commit, and unmapped one granule at a time because
     * that is how each was mapped. */
    std::vector<int64_t> drop;
    for (int64_t o = lo; o < hi; o += g)
        if (is_committed(o, g)) drop.push_back(o);

    for (const int64_t o : drop) {
        const int rc = dev()->vmem_decommit((char*)base_ + o, g);
        if (rc < 0) return rc;
        drop_extent(o, g);
    }
    return RAD_OK;
}

void VMem::add_extent(int64_t off, int64_t bytes) {
    committed_ += bytes;
    g_vmem_committed.fetch_add(bytes, std::memory_order_relaxed);
    /* Insert in order, then absorb any neighbour it touches. Merging is not cosmetic: an extent
     * list that grows by one per commit turns is_committed() into a scan of every call ever made,
     * and the pool that uses this grows a granule at a time. */
    size_t i = 0;
    while (i < ext_.size() && ext_[i].first < off) ++i;
    ext_.insert(ext_.begin() + (long)i, { off, bytes });
    for (size_t j = 0; j + 1 < ext_.size();) {
        if (ext_[j].first + ext_[j].second >= ext_[j + 1].first) {
            const int64_t end = std::max(ext_[j].first + ext_[j].second,
                                         ext_[j + 1].first + ext_[j + 1].second);
            ext_[j].second = end - ext_[j].first;
            ext_.erase(ext_.begin() + (long)j + 1);
        } else {
            ++j;
        }
    }
}

/* The inverse, and it takes a piece that lies WHOLLY INSIDE one extent by construction -- decommit
 * clips every part it gathers to the extent it came from. What is left of that extent is nothing,
 * a head, a tail, or both. */
void VMem::drop_extent(int64_t off, int64_t bytes) {
    committed_ -= bytes;
    g_vmem_committed.fetch_sub(bytes, std::memory_order_relaxed);
    for (size_t i = 0; i < ext_.size(); ++i) {
        const int64_t es = ext_[i].first, ee = es + ext_[i].second;
        if (off < es || off + bytes > ee) continue;
        ext_.erase(ext_.begin() + (long)i);
        if (off + bytes < ee) ext_.insert(ext_.begin() + (long)i, { off + bytes, ee - off - bytes });
        if (es < off)         ext_.insert(ext_.begin() + (long)i, { es, off - es });
        return;
    }
}

}  /* namespace rad */

/* ------------------------------------------------------------------ the ABI surface */

using rad::dev;

extern "C" {

int  rad_dev_count(void)                          { return dev()->count(); }
int  rad_dev_set(int device)                      { return dev()->set(device); }
int  rad_dev_props(int device, RadDeviceProps* o)  { return dev()->props(device, o); }
int  rad_dev_enable_peer(int self, int peer)      { return dev()->enable_peer(self, peer); }

void* rad_dev_alloc(int64_t bytes, int kind) {
    void* p = dev()->alloc(bytes, kind);
    if (p && kind >= 0 && kind < 4) {
        rad::g_ledger[kind].bytes.fetch_add(bytes, std::memory_order_relaxed);
        rad::g_ledger[kind].count.fetch_add(1, std::memory_order_relaxed);
        const int64_t floor_ = rad::trace_floor();
        if (floor_ >= 0 && bytes >= floor_)
            RAD_INFO("alloc %s %s (%lld live in %lld allocations)", rad::kind_name(kind),
                     rad::humanb(bytes).c_str(),
                     (long long)rad::g_ledger[kind].bytes.load(std::memory_order_relaxed),
                     (long long)rad::g_ledger[kind].count.load(std::memory_order_relaxed));
    }
    return p;
}
void  rad_dev_free(void* p, int kind) {
    /* The ledger tracks what was TAKEN, not what is live: rad_dev_free has no size and the
     * backends do not keep one. Every caller in this engine frees at teardown and nowhere else,
     * so the two numbers differ only after the run is over. */
    dev()->free(p, kind);
}
void* rad_dev_host_ptr(void* p)                   { return dev()->host_ptr(p); }
void* rad_dev_device_ptr(void* p)                 { return dev()->device_ptr(p); }
int  rad_dev_host_register(void* p, int64_t bytes, unsigned flags) {
    return dev()->host_register ? dev()->host_register(p, bytes, flags) : RAD_OK;
}
int  rad_dev_host_unregister(void* p) {
    return dev()->host_unregister ? dev()->host_unregister(p) : RAD_OK;
}

int  rad_stream_create(RadStream* out, int hi)    { return dev()->stream_create(out, hi); }
void rad_stream_destroy(RadStream s)              { dev()->stream_destroy(s); }
int  rad_stream_sync(RadStream s)                 { return dev()->stream_sync(s); }

}  /* extern "C" */

namespace rad {

bool    tape_supported()       { return dev()->tape_begin != nullptr; }
int     dev_vram_used(int device, int64_t* out) {
    return dev()->vram_used ? dev()->vram_used(device, out) : RAD_E_UNSUPPORTED;
}
int     tape_begin()           { return dev()->tape_begin ? dev()->tape_begin() : RAD_E_UNSUPPORTED; }
int     tape_mark()            { return dev()->tape_mark ? dev()->tape_mark() : RAD_E_UNSUPPORTED; }
void    tape_pause(bool on)    { if (dev()->tape_pause) dev()->tape_pause(on ? 1 : 0); }
RadTape tape_end(bool keep, bool compare_only) {
    if (!dev()->tape_end) return nullptr;
    const int k = !keep ? 0 : (compare_only ? 2 : 1);
    return dev()->tape_end(k);
}
int     tape_len(RadTape t)    { return dev()->tape_len ? dev()->tape_len(t) : RAD_E_UNSUPPORTED; }
int     tape_play(RadTape t, int from, int to) {
    return dev()->tape_play ? dev()->tape_play(t, from, to) : RAD_E_UNSUPPORTED;
}
int     tape_diff(RadTape a, RadTape b, std::string* why) {
    if (!dev()->tape_diff) return RAD_E_UNSUPPORTED;
    char buf[512];
    buf[0] = 0;
    const int at = dev()->tape_diff(a, b, buf, (int)sizeof buf);
    if (why) *why = buf;
    return at;
}
void    tape_destroy(RadTape t) { if (dev()->tape_destroy) dev()->tape_destroy(t); }
int     tape_record(RadTape t, int i, int* kind, uint32_t* queue) {
    return dev()->tape_record ? dev()->tape_record(t, i, kind, queue) : RAD_E_UNSUPPORTED;
}
int     tape_set_overlap(RadTape t, int i, bool on) {
    return dev()->tape_set_overlap ? dev()->tape_set_overlap(t, i, on ? 1 : 0) : RAD_E_UNSUPPORTED;
}
bool    dev_integrated(int device) { return dev()->integrated && dev()->integrated(device) > 0; }
int     dev_peer_access(int self, int peer) {
    if (self == peer) return 1;
    return dev()->peer_access ? dev()->peer_access(self, peer) : 1;
}
size_t  dev_code_mark() { return dev()->code_mark ? dev()->code_mark() : 0; }
int     dev_code_runs(size_t from, size_t to, int device, std::string* why) {
    if (!dev()->code_runs || from >= to) return 1;
    char buf[1024];
    buf[0] = 0;
    const int r = dev()->code_runs(from, to, device, buf, (int)sizeof buf);
    if (why) *why = buf;
    return r;
}

}  /* namespace rad */

extern "C" {


int  rad_event_create_as(RadEvent* out, unsigned flags) {
    const int s = dev()->event_create(out, flags);
    if (s >= 0 && out && *out) rad::g_live_events.fetch_add(1, std::memory_order_relaxed);
    return s;
}
int  rad_event_create(RadEvent* out)              { return rad_event_create_as(out, RAD_EVENT_HOST_WAIT); }
int  rad_event_create_local(RadEvent* out)        { return rad_event_create_as(out, RAD_EVENT_LOCAL); }
void rad_event_destroy(RadEvent e) {
    if (!e) return;
    rad::g_live_events.fetch_sub(1, std::memory_order_relaxed);
    dev()->event_destroy(e);
}
int  rad_event_record(RadEvent e, RadStream s)    { return dev()->event_record(e, s); }
int  rad_event_wait(RadStream s, RadEvent e)      { return dev()->event_wait(s, e); }
int  rad_event_query(RadEvent e)                  { return dev()->event_query(e); }
int  rad_event_sync(RadEvent e)                   { return dev()->event_sync(e); }
void rad_dev_host_wrote(void)                     { dev()->host_wrote(); }
int  rad_event_elapsed_ms(RadEvent a, RadEvent b, float* out) {
    return dev()->event_elapsed_ms(a, b, out);
}

int rad_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream s) {
    return dev()->memcpy_async(dst, src, bytes, s);
}
int rad_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                        int64_t width, int64_t height, RadStream s) {
    return dev()->memcpy_2d_async(dst, dpitch, src, spitch, width, height, s);
}
int rad_memset_async(void* dst, int value, int64_t bytes, RadStream s) {
    return dev()->memset_async(dst, value, bytes, s);
}
int rad_memcpy_ranges_async(void* dst, const void* src, int64_t prefix,
                            const RadCopyRange* ranges, int n, RadStream s) {
    return dev()->memcpy_ranges_async(dst, src, prefix, ranges, n, s);
}

/* Cleared by reading it, per the header. Never null: a caller printing "%s" on a path where
 * nothing failed should get an empty line, not a segfault. */
const char* rad_dev_last_error(void) {
    static thread_local std::string drained;
    drained.swap(rad::t_last_error);
    rad::t_last_error.clear();
    return drained.c_str();
}

}  /* extern "C" */
