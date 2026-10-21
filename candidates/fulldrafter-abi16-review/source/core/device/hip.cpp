/* hip.cpp -- the HIP half of the device backend: devices, memory and elastic memory.
 *
 * Streams, events, copies and fills are not here. They are AQL packets the engine writes into
 * queues it owns (aql.cpp), which starts from this table and fills in the rest -- pinned host
 * memory included, because the copies have to recognise it.
 *
 * Thin by design. Every entry point is one hip* call plus an error capture, because anything
 * cleverer than that is a place where the host backend and this one can disagree, and the host
 * backend is where all of it is tested (docs/IMPLEMENTATION.md). If a call here grows a policy,
 * the policy belongs in the component that wanted it, not in the device layer.
 *
 * The whole file compiles to nothing without RAD_HAVE_HIP, so a no-ROCm configure builds it and
 * gets an empty object rather than needing the source list to know about the backend split.
 */
#include "device.h"

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP

#include "../rad_internal.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <new>
#include <vector>
#include <cstring>
#include <cstdio>
#include <strings.h>

#include <fcntl.h>
#include <unistd.h>

namespace rad {
namespace {

/* One place that turns a hipError_t into a RAD_E_*. The mapping is coarse on purpose: the string
 * carries the detail, the code carries the class, and a caller that wants to know exactly what the
 * driver said reads rad_dev_last_error(). */
int hip_fail(hipError_t e, const char* who) {
    device_set_error("%s: %s (%s)", who, hipGetErrorString(e), hipGetErrorName(e));
    switch (e) {
        case hipErrorOutOfMemory:      return RAD_E_NOMEM;
        case hipErrorInvalidValue:
        case hipErrorInvalidDevice:
        case hipErrorInvalidHandle:    return RAD_E_INVAL;
        case hipErrorNotSupported:     return RAD_E_UNSUPPORTED;
        default:                       return RAD_E_DEVICE;
    }
}

#define HIP_TRY(call)                                                            \
    do { hipError_t _e = (call);                                                 \
         if (_e != hipSuccess) return hip_fail(_e, #call); } while (0)

/* For the calls whose caller returns void -- the frees and the destroys. A failing teardown is
 * still worth naming: hipFree returning anything but success means the pointer was not what the
 * caller thought it was, and the next allocation of that class is the one that will look wrong. */
void hip_note(hipError_t e, const char* who) {
    if (e != hipSuccess) RAD_WARN("device: %s: %s", who, hipGetErrorString(e));
}
#define HIP_NOTE(call) hip_note((call), #call)

/* ------------------------------------------------------------------ devices */

int d_count(void) {
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess) return 0;
    return n;
}

int d_set(int device) {
    HIP_TRY(hipSetDevice(device));
    return RAD_OK;
}


/* ------------------------------------------------------------------ how much VRAM is really free
 *
 * hipMemGetInfo IS NOT THE ANSWER ON THIS DRIVER, and the error is large enough to pay for.
 *
 * On a card that also drives a display, hipMemGetInfo's "free" sits up to 0.7 GiB below what the
 * amdgpu driver's own per-card accounting reports as unused -- while two otherwise identical cards
 * differ by only tens of MiB in the driver's numbers. That gap is not memory anything holds: a run
 * budgeted past hipMemGetInfo's figure allocates all of it and serves, with sysfs still reporting
 * hundreds of MiB unused afterwards. Believing HIP's number costs that 0.7 GiB, which on a model
 * whose experts do not fit is most of a millisecond a decode step.
 *
 * amdgpu publishes its own accounting, in bytes, per card, counting every process on it. Match the
 * HIP device to a drm node by PCI slot and read it. Two guards: the node's total must equal the
 * total HIP reports, so the two accountings are known to be the same accounting, and any failure
 * at any step falls back to hipMemGetInfo rather than inventing a number.
 *
 * Returns -1 when sysfs cannot answer, which is every non-amdgpu platform.
 *
 * The node's `mem_info_vram_used` is also kept open, per device, for d_vram_used: an observer
 * reads the card through that one descriptor and never through the runtime. */
constexpr int kMaxVramNodes = 64;
std::atomic<int> g_vram_used_fd[kMaxVramNodes] = {};   /* 0: not opened; else the fd plus one */

static int64_t sysfs_vram_free(int device, int64_t total) {
    char bus[64] = {0};
    if (hipDeviceGetPCIBusId(bus, (int)sizeof bus, device) != hipSuccess || !bus[0]) return -1;
    const size_t buslen = std::strlen(bus);

    for (int c = 0; c < 64; ++c) {
        char path[160];
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/uevent", c);
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        bool match = false;
        char line[256];
        while (std::fgets(line, (int)sizeof line, f))
            if (!std::strncmp(line, "PCI_SLOT_NAME=", 14) &&
                !strncasecmp(line + 14, bus, buslen)) { match = true; break; }
        std::fclose(f);
        if (!match) continue;

        auto read_u64 = [&](const char* leaf, long long* out_v) {
            char p[160];
            std::snprintf(p, sizeof p, "/sys/class/drm/card%d/device/%s", c, leaf);
            FILE* g = std::fopen(p, "r");
            if (!g) return false;
            const bool ok = std::fscanf(g, "%lld", out_v) == 1 && *out_v >= 0;
            std::fclose(g);
            return ok;
        };
        long long tot = 0, used = 0;
        if (!read_u64("mem_info_vram_total", &tot) || !read_u64("mem_info_vram_used", &used))
            return -1;
        if (tot != (long long)total) return -1;      /* different accountings; do not mix them */
        if (device >= 0 && device < kMaxVramNodes &&
            g_vram_used_fd[device].load(std::memory_order_acquire) == 0) {
            char p[160];
            std::snprintf(p, sizeof p, "/sys/class/drm/card%d/device/mem_info_vram_used", c);
            const int fd = ::open(p, O_RDONLY | O_CLOEXEC);
            int none = 0;
            if (fd >= 0 && !g_vram_used_fd[device].compare_exchange_strong(none, fd + 1))
                ::close(fd);
        }
        return total - (int64_t)used;
    }
    return -1;
}
/* A sysfs attribute is regenerated by a read at offset 0, so the descriptor is read with pread and
 * never moved. */
int d_vram_used(int device, int64_t* out) {
    if (!out || device < 0 || device >= kMaxVramNodes) return RAD_E_INVAL;
    const int fd = g_vram_used_fd[device].load(std::memory_order_acquire) - 1;
    if (fd < 0) return RAD_E_UNSUPPORTED;
    char buf[32];
    const ssize_t n = ::pread(fd, buf, sizeof buf - 1, 0);
    if (n <= 0) return RAD_E_IO;
    buf[n] = '\0';
    char* end = nullptr;
    const long long v = std::strtoll(buf, &end, 10);
    if (end == buf || v < 0) return RAD_E_IO;
    *out = (int64_t)v;
    return RAD_OK;
}

int d_integrated(int device) {
    int v = 0;
    return hipDeviceGetAttribute(&v, hipDeviceAttributeIntegrated, device) == hipSuccess ? v : 0;
}

/* /proc/meminfo's MemAvailable: what the host could still hand out without swapping, -1 when it
 * cannot be read. */
static int64_t host_mem_available() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long long kb = -1;
    while (std::fgets(line, (int)sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1) break;
    std::fclose(f);
    return kb >= 0 ? (int64_t)kb * 1024 : -1;
}

int d_props(int device, RadDeviceProps* out) {
    if (!out) return RAD_E_INVAL;
    hipDeviceProp_t p{};
    HIP_TRY(hipGetDeviceProperties(&p, device));
    std::memset(out, 0, sizeof *out);

    /* TRUNCATION IS INTENDED AND IS SAID SO. HIP's name and gcnArchName are 256 bytes each and
     * these fields are 128 and 32; snprintf would cut them safely anyway, but silently enough that
     * gcc warns about it, and a build that carries a warning nobody intends to act on is a build
     * whose warnings stop being read. The explicit precision is the same cut, declared. Real
     * values are far inside both -- a product name and a gcnArchName run to a couple of dozen
     * characters -- so this is about the bound, not about the data. */
    snprintf(out->name, sizeof out->name, "%.*s", (int)(sizeof out->name) - 1, p.name);
    snprintf(out->arch, sizeof out->arch, "%.*s", (int)(sizeof out->arch) - 1, p.gcnArchName);
    /* gcnArchName carries the feature suffixes -- "gfx1201:sramecc-:xnack-". The arch a kernel
     * plugin's constraints match on is the bare target, so cut at the colon here rather than
     * making every consumer do it. */
    if (char* c = std::strchr(out->arch, ':')) *c = '\0';

    out->device_id  = device;
    out->n_cu       = p.multiProcessorCount;
    out->warp_size  = p.warpSize;
    out->vram_bytes = (int64_t)p.totalGlobalMem;
    out->lds_bytes  = (int64_t)p.sharedMemPerBlock;
    out->is_host_backend = 0;

    /* hipMemGetInfo reports the CURRENT device, so ask about `device` by visiting it. That costs
     * two context switches on a call nobody makes per step -- props is read at startup and by
     * rad-info -- and the alternative is a vram_free that silently describes the wrong card under
     * tensor parallel. */
    int prev = 0;
    bool moved = false;
    if (hipGetDevice(&prev) == hipSuccess && prev != device &&
        hipSetDevice(device) == hipSuccess)
        moved = true;
    size_t freeb = 0, totalb = 0;
    if (hipMemGetInfo(&freeb, &totalb) == hipSuccess) {
        out->vram_free = (int64_t)freeb;
        if (totalb) out->vram_bytes = (int64_t)totalb;
    }
    /* And then prefer the driver's own counter where it can be read -- see sysfs_vram_free. The
     * fallback above is what every non-amdgpu platform gets; it is conservative by up to a GiB on
     * a card that drives a display. */
    {
        const int64_t sf = sysfs_vram_free(device, out->vram_bytes);
        if (sf >= 0) {
            out->vram_free = sf;
        } else if (d_integrated(device)) {
            /* INTEGRATED GRAPHICS WHOSE "DEVICE MEMORY" IS THE HOST'S. When the runtime reports
             * more than the carve-out the driver counts as VRAM, the rest is system RAM mapped for
             * the GPU, and the runtime's free figure knows nothing of the other processes using
             * it: a budget derived from it claims memory the machine is already running on. What
             * the host can still give is the bound. A carve-out the runtime reports as it is --
             * the case sysfs answers above -- is dedicated memory and is left alone. */
            const int64_t avail = host_mem_available();
            if (avail >= 0 && avail < out->vram_free) out->vram_free = avail;
        }
    }
    /* A failed restore leaves this thread pointed at the wrong card, which under tensor parallel
     * is a rank quietly working on its neighbour's memory. Say so. */
    if (moved) HIP_NOTE(hipSetDevice(prev));

    /* Whether any peer is reachable at all. The engine asks for a specific pair through
     * rad_dev_enable_peer; this flag is for the report. */
    int n = d_count();
    for (int i = 0; i < n && !out->supports_p2p; ++i) {
        if (i == device) continue;
        int can = 0;
        if (hipDeviceCanAccessPeer(&can, device, i) == hipSuccess && can) out->supports_p2p = 1;
    }
    return RAD_OK;
}

int d_peer_access(int self, int peer) {
    int can = 0;
    HIP_TRY(hipDeviceCanAccessPeer(&can, self, peer));
    return can ? 1 : 0;
}

int d_enable_peer(int self, int peer) {
    if (self == peer) return RAD_OK;
    int can = 0;
    HIP_TRY(hipDeviceCanAccessPeer(&can, self, peer));
    if (!can) {
        device_set_error("rad_dev_enable_peer(%d, %d): the driver reports no peer path; "
                         "a kernel library's peer-to-peer all-reduce needs one (spec §9)", self,
                         peer);
        return RAD_E_UNSUPPORTED;
    }
    int prev = 0;
    HIP_TRY(hipGetDevice(&prev));
    HIP_TRY(hipSetDevice(self));
    hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
    HIP_NOTE(hipSetDevice(prev));
    /* Already enabled is success. The kernel plugin's init() hook runs per resolved instance
     * (spec §2.1) and several instances legitimately want the same pair. */
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
        return hip_fail(e, "hipDeviceEnablePeerAccess");
    return RAD_OK;
}

/* ------------------------------------------------------------------ memory */

void* d_alloc(int64_t bytes, int kind) {
    if (bytes <= 0) {
        device_set_error("rad_dev_alloc(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    void* p = nullptr;
    hipError_t e = hipSuccess;
    switch (kind) {
        case RAD_MEM_DEVICE:
            e = hipMalloc(&p, (size_t)bytes);
            break;
        case RAD_MEM_HOST_PINNED:
            /* The mover's staging pool and the O_DIRECT landing zone. Pinned so a transfer is a
             * DMA the driver does not have to stage through its own bounce buffer (spec §5.4).
             * hipHostMallocNonCoherent disables CPU cache snooping during GPU DMA transfers over PCIe,
             * delivering significantly higher PCIe throughput. */
#ifdef hipHostMallocNonCoherent
            e = hipHostMalloc(&p, (size_t)bytes, hipHostMallocNonCoherent);
#else
            e = hipHostMalloc(&p, (size_t)bytes, hipHostMallocDefault);
#endif
            break;
        case RAD_MEM_HOST_MAPPED:
            /* The AQL backend's, which pins host memory through ROCr in pieces the IOMMU can map
             * (aql.cpp, "pinned host memory, in pieces"). */
            device_set_error("rad_dev_alloc: pinned host memory is the AQL backend's");
            return nullptr;
        case RAD_MEM_HOST:
            /* Pageable. Aligned to a page anyway so the same buffer can be an O_DIRECT target. */
            if (posix_memalign(&p, 4096, (size_t)((bytes + 4095) / 4096 * 4096)) != 0) p = nullptr;
            if (!p) e = hipErrorOutOfMemory;
            break;
        default:
            device_set_error("rad_dev_alloc: unknown memory kind %d", kind);
            return nullptr;
    }
    if (e != hipSuccess || !p) {
        hip_fail(e == hipSuccess ? hipErrorOutOfMemory : e, "rad_dev_alloc");
        return nullptr;
    }
    return p;
}

void d_free(void* p, int kind) {
    if (!p) return;
    switch (kind) {
        case RAD_MEM_DEVICE:       HIP_NOTE(hipFree(p)); break;
        case RAD_MEM_HOST:         std::free(p); break;
        case RAD_MEM_HOST_PINNED:
        case RAD_MEM_HOST_MAPPED:
            RAD_WARN("rad_dev_free(%p): pinned host memory is the AQL backend's; leaked", p);
            break;
        default:
            RAD_WARN("rad_dev_free(%p): unknown memory kind %d; leaked rather than freed with the "
                     "wrong allocator", p, kind);
            break;
    }
}

void* d_host_ptr(void* p) {
    if (!p) return nullptr;
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) return nullptr;
    return a.hostPointer;
}

void* d_device_ptr(void* p) {
    if (!p) return nullptr;
    hipPointerAttribute_t a{};
    if (hipPointerGetAttributes(&a, p) != hipSuccess) return nullptr;
    if (a.devicePointer) return a.devicePointer;
    /* A mapped host allocation that the attribute query did not resolve: ask the explicit way. */
    void* d = nullptr;
    if (hipHostGetDevicePointer(&d, p, 0) == hipSuccess) return d;
    return nullptr;
}

int d_host_register(void* p, int64_t bytes, unsigned flags) {
    if (!p || bytes <= 0) return RAD_E_INVAL;
    hipError_t e = hipHostRegister(p, (size_t)bytes, hipHostRegisterMapped);
    if (e != hipSuccess && e != hipErrorHostMemoryAlreadyRegistered) {
        return hip_fail(e, "hipHostRegister");
    }
    return RAD_OK;
}

int d_host_unregister(void* p) {
    if (!p) return RAD_E_INVAL;
    hipError_t e = hipHostUnregister(p);
    if (e != hipSuccess) return hip_fail(e, "hipHostUnregister");
    return RAD_OK;
}

const DeviceBackend g_hip = {

    "hip",
    d_count, d_set, d_props, d_enable_peer,
    d_alloc, d_free, d_host_ptr, d_device_ptr,
    d_host_register, d_host_unregister,
    /* Streams, events, copies and fills: the AQL backend's (aql.cpp), which starts from this
     * table and fills these in. */
    nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr,
    nullptr,
    nullptr, nullptr, nullptr, nullptr,
    /* Recorded passes: likewise the AQL backend's. */
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,

    /* Elastic memory: likewise the AQL backend's, which reserves and backs ranges through HSA
     * (aql.cpp, "elastic memory, through HSA"). */
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,

    d_vram_used,
    /* Plugin device code: the AQL backend's, which owns the fat-binary registrations. */
    nullptr, nullptr,
    d_integrated,
    d_peer_access,
};

}  /* namespace */



const DeviceBackend* hip_backend_if_present(void) {
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess || n <= 0) {
        /* Not an error and not a warning: a HIP build on a machine with no visible card is a
         * supported configuration, it is just the host one. dispatch.cpp says so at Info. */
        RAD_DEBUG("device: hipGetDeviceCount -> %s, %d device(s)", hipGetErrorName(e), n);
        return nullptr;
    }
    return &g_hip;
}

}  /* namespace rad */

#endif /* RAD_HAVE_HIP */
