/* aql.cpp -- the AQL device backend: every stream is an HSA queue the engine owns, and every
 * kernel, copy, fill and fence on it is an AQL packet written here.
 *
 * Why own packets: hipLaunchKernel costs 0.5-1.3 us of host time per dispatch versus 6 ns to
 * write the packet, and HIP fences every kernel at system scope, which costs ~0.5 us of device
 * time between dependent kernels that an agent-scope fence does not. At ~1600 dispatches per rank
 * per decode step both add up to milliseconds. HIP also retires a Command object per launch on its
 * own thread, most of a core under load.
 *
 * What stays with HIP: allocation, device properties, peer access, IPC, elastic memory, and
 * compiling and registering kernels. A kernel library is built by hipcc and calls hipLaunchKernel
 * as usual; this file defines hipLaunchKernel in the executable, so the dynamic linker binds every
 * plugin's reference here ahead of the HIP runtime, and it records __hipRegisterFatBinary and
 * __hipRegisterFunction so a host stub maps to its code object. A launch onto a stream this
 * backend did not create is forwarded to HIP unchanged.
 *
 * Fences: kernels on one queue see each other's writes through the L2 with an agent-scope fence.
 * System scope is needed only for memory the L2 does not see:
 *   - the kernel argument ring is written by the host: into VRAM through the BAR when the BAR
 *     spans the card's memory, into pinned host memory otherwise. The L2 caches neither without
 *     an invalidation. A slot is rewritten one lap after it was last read, so a system-scope
 *     acquire is placed at least once per half lap; without it every kernel after the first lap
 *     reads its predecessor's arguments. Slots are 1 KiB so no argument fetch shares a cache line
 *     with the next slot;
 *   - host-written memory: the first dispatch after the host waits on any queue takes a
 *     system-scope acquire, since host writes for the next step follow such a wait;
 *   - copies touching host or peer memory carry the scope themselves, as do the barrier packets
 *     that implement events and synchronisation.
 */
#include "device.h"

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP

/* The standard headers first: HIP's defines __noinline__, which <format> (reached through
 * <chrono>) uses as an attribute name. */
#include <dirent.h>
#include <dlfcn.h>
#include <immintrin.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
/* ROCm ships 1.16, which carries signal_event_data that the kernel's 1.11 header lacks. */
#if __has_include(<rocm_smi/kfd_ioctl.h>)
#include <rocm_smi/kfd_ioctl.h>
#else
#include <linux/kfd_ioctl.h>
#endif
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "aql.h"
#include "rad_format.h"
#include "../rad_internal.h"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/amd_hsa_signal.h>


namespace rad {

int aql_vmem_device_of(const void* addr, int* dev);

namespace aql {
namespace {

constexpr uint64_t kStreamMagic = 0x5241442d41514c53ull;   /* "RAD-AQLS" */
constexpr uint64_t kEventMagic  = 0x5241442d41514c45ull;   /* "RAD-AQLE" */
constexpr uint64_t kTapeMagic   = 0x5241442d41514c54ull;   /* "RAD-AQLT" */
constexpr uint32_t kQueuePackets = 4096;
constexpr uint32_t kKargSlot     = 1024;
constexpr uint32_t kKargMax      = kKargSlot - 256;
constexpr int      kMaxDevices   = 64;
constexpr size_t   kBounceBytes  = 8u << 20;

/* ------------------------------------------------------------------ the HIP entry points */

struct Real {
    hipError_t (*launch)(const void*, dim3, dim3, void**, size_t, hipStream_t) = nullptr;
    hipError_t (*memcpy2d)(void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind,
                           hipStream_t) = nullptr;
    hipError_t (*memcpy)(void*, const void*, size_t, hipMemcpyKind, hipStream_t) = nullptr;
    hipError_t (*memset)(void*, int, size_t, hipStream_t) = nullptr;
    hipError_t (*memset_sync)(void*, int, size_t) = nullptr;
    hipError_t (*sync)(hipStream_t) = nullptr;
    hipError_t (*last)(void) = nullptr;
    hipError_t (*peek)(void) = nullptr;
    hipError_t (*ev_record)(hipEvent_t, hipStream_t) = nullptr;
    hipError_t (*wait_ev)(hipStream_t, hipEvent_t, unsigned) = nullptr;
    void** (*reg_fat)(const void*) = nullptr;
    void (*reg_fn)(void**, const void*, char*, const char*, unsigned, void*, void*, void*, void*,
                   int*) = nullptr;
    void (*unreg_fat)(void**) = nullptr;
};

template <typename F> void bind(F*& f, const char* name) {
    if (!f) f = reinterpret_cast<F*>(dlsym(RTLD_NEXT, name));
}

Real& real() {
    static Real* r = [] {
        Real* x = new Real();
        bind(x->launch, "hipLaunchKernel");
        bind(x->memcpy2d, "hipMemcpy2DAsync");
        bind(x->memcpy, "hipMemcpyAsync");
        bind(x->memset, "hipMemsetAsync");
        bind(x->memset_sync, "hipMemset");
        bind(x->sync, "hipStreamSynchronize");
        bind(x->last, "hipGetLastError");
        bind(x->peek, "hipPeekAtLastError");
        bind(x->ev_record, "hipEventRecord");
        bind(x->wait_ev, "hipStreamWaitEvent");
        bind(x->reg_fat, "__hipRegisterFatBinary");
        bind(x->reg_fn, "__hipRegisterFunction");
        bind(x->unreg_fat, "__hipUnregisterFatBinary");
        return x;
    }();
    return *r;
}

/* HIP's per-thread last-error slot. Every HIP call overwrites it, success included, and a
 * plugin's `launch; if (hipGetLastError())` relies on that: an earlier call's error must not be
 * reported against the launch. An entry point handled here records its result and marks itself
 * as the thread's latest call; hipGetLastError answers from it while the mark stands. */
thread_local hipError_t t_err = hipSuccess;
thread_local bool       t_ours = false;

inline hipError_t ret(hipError_t e) { t_err = e; t_ours = true; return e; }
thread_local uint8_t    t_next_fence = 0;

/* Whether any work was forwarded to a HIP stream (a stream this backend does not own). Until
 * then HIP's default stream is empty and synchronising it is answered here; asking HIP would
 * make it create the stream's queue first. */
std::atomic<bool> g_hip_streams_used{false};

/* ------------------------------------------------------------------ kernels */

struct KernelRec {
    uint64_t    object = 0;
    uint32_t    kernarg_size = 0;
    uint32_t    group_size = 0;
    uint32_t    private_size = 0;
    uint16_t    n_explicit = 0;
    std::vector<uint32_t> ex_off, ex_size;
    int32_t     hid[16];                      /* offset of each Hidden kind, or -1 */
    uint8_t     hid_size[16];
    std::string name;
};

struct ModuleImage {
    const void* co = nullptr;
    size_t      size = 0;
    std::string err;
    std::unordered_map<std::string, KernelMeta> meta;
    bool        parsed = false;
};

struct Module {
    const void* bundle = nullptr;
    std::mutex  mu;
    std::unordered_map<std::string, ModuleImage> image;   /* by isa */
    hsa_executable_t exec[kMaxDevices] = {};
    bool        loaded[kMaxDevices] = {};
    std::string load_err[kMaxDevices];
};

struct Func {
    Module*     mod = nullptr;
    std::string name;
    std::atomic<KernelRec*> rec[kMaxDevices];
    std::string err[kMaxDevices];
    Func() { for (auto& r : rec) r.store(nullptr, std::memory_order_relaxed); }
};

struct Registry {
    std::mutex mu;
    std::unordered_map<void**, Module*> by_handle;
    std::unordered_map<const void*, Func*> by_stub;
    /* Registrations in order, so one dlopen's fat binaries form a contiguous run the plugin
     * loader can check against the card (a_code_runs). */
    std::vector<Module*> order;
};

Registry& reg() {
    static Registry* r = new Registry();   /* never destroyed: unregistration runs from atexit */
    return *r;
}

/* ------------------------------------------------------------------ devices */

struct Dev {
    hsa_agent_t agent{};
    std::string isa;                    /* the agent's name, "gfx1201": for messages */
    std::vector<std::string> isas;      /* the targets it accepts, preferred first; see agent_isas */
    hsa_amd_memory_pool_t vram{};
    bool        has_vram = false;
    uint32_t    kfd_id = 0;             /* the driver's gpu_id: its sysfs and ioctls name it so */
    bool        host_kargs = false;     /* no large BAR: the argument ring lives in host memory */
};

struct Global {
    bool ok = false;
    std::string why;
    hsa_agent_t cpu{};
    hsa_amd_memory_pool_t host_fine{};
    bool has_host_fine = false;
    std::vector<Dev> dev;                     /* by HIP ordinal */
    uint64_t ts_freq = 1;
    /* Bumped whenever the host has waited on a queue: the next dispatch on every queue takes a
     * system-scope acquire, because what the host writes for the next step follows such a wait. */
    std::atomic<uint64_t> host_epoch{1};
    std::mutex streams_mu;
    std::vector<struct Stream*> streams;
    /* Host waits; see "host waits". */
    int kfd = -1;                             /* the driver, for its event waits */
    hsa_signal_t fault{0};                    /* fired by a queue fault, to end every wait */
    std::atomic<bool> faulted{false};
};

Global& G() {
    static Global* g = new Global();
    return *g;
}

std::string agent_isa(hsa_agent_t a) {
    char name[64] = {0};
    hsa_agent_get_info(a, HSA_AGENT_INFO_NAME, name);
    return name;
}

hsa_status_t collect_isa(hsa_isa_t isa, void* data) {
    uint32_t len = 0;
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len) != HSA_STATUS_SUCCESS || len == 0 ||
        len > 256)
        return HSA_STATUS_SUCCESS;
    std::vector<char> name((size_t)len + 1, '\0');
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name.data()) == HSA_STATUS_SUCCESS)
        static_cast<std::vector<std::string>*>(data)->emplace_back(name.data());
    return HSA_STATUS_SUCCESS;
}

/* The code-object targets an agent runs, in the runtime's order of preference: its own processor,
 * then its family's generic target, each with the agent's modes (an R9700: "gfx1201", then
 * "gfx12-generic"). Asking the runtime instead of keeping a family table means newer cards need
 * no edit here. */
std::vector<std::string> agent_isas(hsa_agent_t a) {
    std::vector<std::string> v;
    hsa_agent_iterate_isas(a, collect_isa, &v);
    if (v.empty()) v.push_back("amdgcn-amd-amdhsa--" + agent_isa(a));
    return v;
}

/* The driver's topology node of card `gpu_id`, -1 when no node names it. */
int topo_node(uint32_t gpu_id) {
    for (int n = 0; n < 256; ++n) {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", n);
        FILE* f = std::fopen(path, "r");
        if (!f) { if (n > 0) break; continue; }
        unsigned id = 0;
        const bool got = std::fscanf(f, "%u", &id) == 1;
        std::fclose(f);
        if (got && id == gpu_id) return n;
    }
    return -1;
}

/* Whether the CPU can write the card's VRAM: 1 yes, 0 no, -1 unknown. The driver maps VRAM for
 * the CPU only when the PCI BAR spans all of it. With a smaller BAR (Resizable BAR off or
 * unsupported) the topology reports a private frame buffer heap and the runtime maps every VRAM
 * allocation PROT_NONE, so argument rings go to host memory instead (init_devices). The heap type
 * is what the runtime itself decides from. */
int vram_host_writable(uint32_t gpu_id) {
    constexpr unsigned long long kHeapFbPrivate = 2;    /* HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE */
    const int n = gpu_id ? topo_node(gpu_id) : -1;
    if (n < 0) return -1;
    int banks = 0;
    bool priv = false;
    for (int b = 0; b < 64; ++b, ++banks) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/class/kfd/kfd/topology/nodes/%d/mem_banks/%d/properties",
                      n, b);
        FILE* f = std::fopen(path, "r");
        if (!f) break;
        char key[128];
        unsigned long long v = 0;
        while (std::fscanf(f, "%127s %llu", key, &v) == 2)
            if (!std::strcmp(key, "heap_type") && v == kHeapFbPrivate) priv = true;
        std::fclose(f);
    }
    return banks == 0 ? -1 : priv ? 0 : 1;
}

hsa_status_t collect_gpu(hsa_agent_t a, void* data) {
    hsa_device_type_t t;
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
    auto* v = static_cast<std::vector<hsa_agent_t>*>(data);
    if (t == HSA_DEVICE_TYPE_GPU) v->push_back(a);
    else if (t == HSA_DEVICE_TYPE_CPU && !G().cpu.handle) G().cpu = a;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t find_vram(hsa_amd_memory_pool_t p, void* data) {
    hsa_amd_segment_t seg;
    uint32_t flags = 0;
    bool alloc = false;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
    Dev* d = static_cast<Dev*>(data);
    if (alloc && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && !d->has_vram) {
        d->vram = p;
        d->has_vram = true;
    }
    return HSA_STATUS_SUCCESS;
}

hsa_status_t find_host_fine(hsa_amd_memory_pool_t p, void*) {
    hsa_amd_segment_t seg;
    uint32_t flags = 0;
    bool alloc = false;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
    if (alloc && (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) && !G().has_host_fine) {
        G().host_fine = p;
        G().has_host_fine = true;
    }
    return HSA_STATUS_SUCCESS;
}

/* Whether a wait on the signal can block: the packet that completes it posts an event and raises
 * an interrupt. */
bool interrupts(hsa_signal_t s) {
    return s.handle && reinterpret_cast<const amd_signal_t*>(s.handle)->event_mailbox_ptr != 0;
}

hsa_status_t on_system_event(const hsa_amd_event_t* e, void*);   /* the memory-fault report */

/* HIP's ordinals and HSA's agents are matched by PCI address rather than by position: the CPU's
 * integrated graphics is an HSA agent too, and HIP's visible-device mask can hide any of them. */
bool init_devices() {
    Global& g = G();
    if (hsa_init() != HSA_STATUS_SUCCESS) { g.why = "hsa_init failed"; return false; }
    g.kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (g.kfd < 0) { g.why = fmt("/dev/kfd could not be opened: %s", std::strerror(errno)); return false; }
    if (hsa_signal_create(0, 0, nullptr, &g.fault) != HSA_STATUS_SUCCESS || !interrupts(g.fault)) {
        g.why = "the runtime makes no interrupt signals, and every host wait blocks on one";
        return false;
    }
    std::vector<hsa_agent_t> gpus;
    hsa_iterate_agents(collect_gpu, &gpus);
    if (!g.cpu.handle) { g.why = "no CPU agent"; return false; }
    hsa_amd_agent_iterate_memory_pools(g.cpu, find_host_fine, nullptr);
    if (!g.has_host_fine) { g.why = "no fine-grained host memory pool"; return false; }
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &g.ts_freq);
    if (!g.ts_freq) g.ts_freq = 1;

    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n <= 0) { g.why = "no HIP devices"; return false; }
    if (n > kMaxDevices) { g.why = "more devices than the backend tracks"; return false; }
    g.dev.resize((size_t)n);
    for (int i = 0; i < n; ++i) {
        char bus[64] = {0};
        unsigned dom = 0, b = 0, d = 0, f = 0;
        if (hipDeviceGetPCIBusId(bus, (int)sizeof bus, i) != hipSuccess ||
            std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) {
            g.why = fmt("could not read the PCI address of HIP device %d", i);
            return false;
        }
        const uint32_t want = (b << 8) | (d << 3) | f;
        bool found = false;
        for (hsa_agent_t a : gpus) {
            uint32_t bdf = 0, adom = 0;
            hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf);
            hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &adom);
            if (bdf == want && adom == dom) {
                g.dev[(size_t)i].agent = a;
                g.dev[(size_t)i].isa = agent_isa(a);
                g.dev[(size_t)i].isas = agent_isas(a);
                hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DRIVER_UID,
                                   &g.dev[(size_t)i].kfd_id);
                hsa_amd_agent_iterate_memory_pools(a, find_vram, &g.dev[(size_t)i]);
                found = true;
                break;
            }
        }
        if (!found || !g.dev[(size_t)i].has_vram) {
            g.why = fmt("HIP device %d (%s) has no matching HSA agent with a VRAM pool", i, bus);
            return false;
        }
        /* The host writes every dispatch's arguments (a_stream_create), in VRAM when the BAR
         * spans it. A smaller BAR makes the runtime map VRAM for the CPU PROT_NONE, which would
         * fault at the first launch; the ring goes to pinned host memory instead, as HIP's own
         * launches do, at the cost of the card reading each dispatch's arguments over the link. */
        if (vram_host_writable(g.dev[(size_t)i].kfd_id) == 0) {
            g.dev[(size_t)i].host_kargs = true;
            RAD_WARN("device: card %d (%s) has a PCI BAR smaller than its VRAM (Resizable BAR "
                     "off, or the slot's window too small), so kernel arguments stay in pinned host "
                     "memory and every dispatch reads them across the link, which costs launch "
                     "rate. Above 4G Decoding and Resizable BAR in the firmware setup remove the "
                     "difference where the slot allows it",
                     i, bus);
        }
        /* The fallback on a card that needs none, to measure what reading each dispatch's
         * arguments across the link costs against the same card's VRAM rings. */
        const char* force_host_kargs = std::getenv("RADIANCE_DEBUG_AQL_HOST_KARGS");
        if (force_host_kargs && *force_host_kargs && *force_host_kargs != '0') {
            g.dev[(size_t)i].host_kargs = true;
            RAD_WARN("device: card %d (%s): RADIANCE_DEBUG_AQL_HOST_KARGS puts the kernel argument "
                     "rings in host memory", i, bus);
        }
    }
    /* Without the handler a fault falls back to the runtime's own handling, which the
     * memory-fault report below replaces; warn at startup. */
    if (hsa_amd_register_system_event_handler(on_system_event, nullptr) != HSA_STATUS_SUCCESS)
        RAD_WARN("device: the runtime refused the memory-fault report's handler; a GPU memory "
                 "fault will not be reported, and may hang the server instead of ending it");
    return true;
}

/* ------------------------------------------------------------------ code object loading */

const KernelMeta* module_meta(Module* m, const std::vector<std::string>& isas,
                              const ModuleImage** img, const std::string& sym, std::string* why) {
    /* Keyed by the card's preferred target with its modes, which is what decides the choice. */
    ModuleImage& im = m->image[isas.front()];
    if (!im.parsed) {
        im.parsed = true;
        if (!bundle_code_object(m->bundle, isas, &im.co, &im.size, &im.err)) return nullptr;
        std::vector<KernelMeta> ks;
        if (!code_object_kernels(im.co, im.size, &ks, &im.err)) { im.co = nullptr; return nullptr; }
        for (KernelMeta& k : ks) im.meta.emplace(k.symbol, std::move(k));
    }
    *img = &im;
    if (!im.co) { *why = im.err; return nullptr; }
    auto it = im.meta.find(sym);
    if (it == im.meta.end()) { *why = "the code object has no metadata for " + sym; return nullptr; }
    return &it->second;
}

bool module_load(Module* m, int ord, const ModuleImage& im, std::string* why) {
    if (m->loaded[ord]) {
        if (!m->load_err[ord].empty()) { *why = m->load_err[ord]; return false; }
        return true;
    }
    m->loaded[ord] = true;
    const Dev& d = G().dev[(size_t)ord];
    hsa_code_object_reader_t rd;
    hsa_executable_t ex;
    if (hsa_code_object_reader_create_from_memory(im.co, im.size, &rd) != HSA_STATUS_SUCCESS ||
        hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                  &ex) != HSA_STATUS_SUCCESS) {
        m->load_err[ord] = "could not create an executable for the code object";
    } else if (hsa_executable_load_agent_code_object(ex, d.agent, rd, nullptr, nullptr) !=
                   HSA_STATUS_SUCCESS ||
               hsa_executable_freeze(ex, nullptr) != HSA_STATUS_SUCCESS) {
        m->load_err[ord] = "the code object did not load on " + d.isa;
    } else {
        m->exec[ord] = ex;
    }
    if (!m->load_err[ord].empty()) { *why = m->load_err[ord]; return false; }
    return true;
}

/* Kernel objects by name, for the queue dump: a stalled queue is read by what it is stuck on. */
std::mutex g_names_mu;
std::unordered_map<uint64_t, std::string> g_names;

KernelRec* resolve(Func* fn, int ord) {
    if (KernelRec* k = fn->rec[ord].load(std::memory_order_acquire)) return k;
    Module* m = fn->mod;
    std::lock_guard<std::mutex> lk(m->mu);
    if (KernelRec* k = fn->rec[ord].load(std::memory_order_acquire)) return k;
    if (!fn->err[ord].empty()) return nullptr;

    const Dev& d = G().dev[(size_t)ord];
    const std::string sym = fn->name + ".kd";
    std::string why;
    const ModuleImage* im = nullptr;
    const KernelMeta* meta = module_meta(m, d.isas, &im, sym, &why);
    if (!meta || !module_load(m, ord, *im, &why)) {
        fn->err[ord] = why;
        return nullptr;
    }
    hsa_executable_symbol_t s;
    if (hsa_executable_get_symbol_by_name(m->exec[ord], sym.c_str(), &d.agent, &s) !=
        HSA_STATUS_SUCCESS) {
        fn->err[ord] = "the loaded executable has no symbol " + sym;
        return nullptr;
    }
    auto k = std::make_unique<KernelRec>();
    k->name = fn->name;
    hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &k->object);
    hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                                   &k->kernarg_size);
    hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                   &k->group_size);
    hsa_executable_symbol_get_info(s, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                   &k->private_size);
    /* The argument slot is fixed at 1 KiB; a larger segment would eat into the guard the fence
     * scheme needs between slots. No kernel in this tree comes near it. */
    if (k->kernarg_size > kKargMax) {
        fn->err[ord] = fmt("its kernel argument segment is %u bytes and the backend's slot holds %u",
                           k->kernarg_size, kKargMax);
        return nullptr;
    }
    /* A dynamic stack needs a scratch size the kernel cannot state; HIP guesses one, this backend
     * refuses. No kernel in this tree has one. */
    if (meta->dynamic_stack) {
        fn->err[ord] = "it uses a dynamic stack, which the AQL backend does not size";
        return nullptr;
    }
    for (int h = 0; h < 16; ++h) { k->hid[h] = -1; k->hid_size[h] = 0; }
    for (const KernelArg& a : meta->args) {
        if (a.hidden == Hidden::None) {
            k->ex_off.push_back(a.offset);
            k->ex_size.push_back(a.size);
        } else if (a.hidden == Hidden::Unsupported) {
            fn->err[ord] = "it takes the implicit argument " + a.kind + ", which the AQL backend "
                           "does not provide";
            return nullptr;
        } else if (a.hidden != Hidden::Pad) {
            k->hid[(int)a.hidden] = (int32_t)a.offset;
            k->hid_size[(int)a.hidden] = (uint8_t)a.size;
        }
    }
    k->n_explicit = (uint16_t)k->ex_off.size();
    {
        std::lock_guard<std::mutex> nl(g_names_mu);
        g_names[k->object] = k->name;
    }
    KernelRec* out = k.release();
    fn->rec[ord].store(out, std::memory_order_release);
    return out;
}

/* One lookup per launch: a thread-local direct-mapped cache in front of the registry. */
struct CacheEnt { const void* stub; int ord; KernelRec* k; };
thread_local CacheEnt t_cache[512];

KernelRec* kernel_for(const void* stub, int ord, std::string* why) {
    const size_t h = (((uintptr_t)stub >> 4) ^ ((uintptr_t)stub >> 13) ^ (size_t)ord) & 511u;
    CacheEnt& c = t_cache[h];
    if (c.stub == stub && c.ord == ord && c.k) return c.k;
    Func* fn = nullptr;
    {
        Registry& r = reg();
        std::lock_guard<std::mutex> lk(r.mu);
        auto it = r.by_stub.find(stub);
        if (it != r.by_stub.end()) fn = it->second;
    }
    if (!fn) { *why = "the kernel was never registered with HIP"; return nullptr; }
    KernelRec* k = resolve(fn, ord);
    if (!k) { *why = fn->name + ": " + fn->err[ord]; return nullptr; }
    c = { stub, ord, k };
    return k;
}

/* ------------------------------------------------------------------ streams */

struct Stream {
    uint64_t        magic = kStreamMagic;
    int             ord = 0;
    uint32_t        id = 0;                   /* index into g_queue_by_id, never reused */
    hsa_agent_t     agent{};
    hsa_queue_t*    q = nullptr;
    uint64_t        mask = 0;
    char*           karg = nullptr;           /* VRAM the CPU writes through the BAR, or host memory */
    std::atomic<bool> busy{false};
    uint64_t        wr = 0;                   /* next packet index */
    uint64_t        rd = 0;                   /* last read index seen */
    int64_t         last_sys_acq = -(int64_t)kQueuePackets;
    uint64_t        epoch = 0;                /* host epoch at the last system acquire */
    hsa_signal_t    ride{};                   /* rad_ride: the next dispatch's completion signal */
    bool            agent_only = false;       /* rad_xp_stream_create: never a system-scope acquire */
    std::atomic<int> gates{0};                /* closed gates (rad_stream_gate) on this queue */
    std::atomic<int> fault{0};
    char*           bounce = nullptr;         /* pinned, for pageable copies */
    std::mutex      bounce_mu;

    void lock() {
        while (busy.exchange(true, std::memory_order_acquire))
            while (busy.load(std::memory_order_relaxed)) _mm_pause();
    }
    void unlock() { busy.store(false, std::memory_order_release); }
};

/* Every queue this process has made, by an id that is never reused, so "packet i of queue n"
 * stays meaningful after the queue is destroyed. A destroyed queue's entry is null; everything
 * it held has been consumed. */
constexpr uint32_t kMaxQueues = 1024;
std::atomic<hsa_queue_t*> g_queue_by_id[kMaxQueues];
std::atomic<uint32_t>     g_next_queue_id{0};

inline Stream* ours(const void* s) {
    if ((uintptr_t)s < 4096) return nullptr;
    const Stream* p = static_cast<const Stream*>(s);
    return p->magic == kStreamMagic ? const_cast<Stream*>(p) : nullptr;
}

void on_queue_error(hsa_status_t st, hsa_queue_t*, void* data) {
    Stream* s = static_cast<Stream*>(data);
    const char* m = nullptr;
    hsa_status_string(st, &m);
    s->fault.store((int)st, std::memory_order_release);
    G().faulted.store(true, std::memory_order_release);
    hsa_signal_store_screlease(G().fault, 1);
    RAD_ERR("device %d: the AQL queue faulted: %s", s->ord, m ? m : "?");
}

/* Queue dump, printed once when a host wait has not returned for 5 s. Nothing on the step path
 * takes that long, so such a wait means a queue is stuck on a packet (a barrier on a signal
 * nothing will decrement, or a kernel that faulted silently); the packet at the read index says
 * which. */
/* Sum of every queue's read index. It only grows, so an unchanged sum means no queue moved. */
/* How long a host wait may see no queue advance before it reports the queues; see dump_queues. */
constexpr uint32_t kStallMs = 5000;

uint64_t read_progress() {
    std::lock_guard<std::mutex> lk(G().streams_mu);
    uint64_t n = 0;
    for (Stream* s : G().streams) n += hsa_queue_load_read_index_scacquire(s->q);
    return n;
}

/* The DMA lanes' part of the dump, set once a lane opens. */
void (*g_dump_lanes)() = nullptr;

void dump_queues(const char* where) {
    std::lock_guard<std::mutex> lk(G().streams_mu);
    RAD_ERR("device: %s has seen no queue advance for %u s; the queues:", where, kStallMs / 1000);
    for (Stream* s : G().streams) {
        const uint64_t rd = hsa_queue_load_read_index_scacquire(s->q);
        const uint64_t wr = hsa_queue_load_write_index_scacquire(s->q);
        std::string what = "idle";
        if (rd < wr) {
            auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) + (rd & s->mask);
            const uint16_t h = __atomic_load_n(&p->header, __ATOMIC_ACQUIRE);
            const int type = (h >> HSA_PACKET_HEADER_TYPE) & 0xff;
            if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
                std::lock_guard<std::mutex> nl(g_names_mu);
                auto it = g_names.find(p->kernel_object);
                what = "kernel " + (it != g_names.end() ? it->second : fmt("%#" PRIx64, p->kernel_object));
            } else if (type == HSA_PACKET_TYPE_BARRIER_AND) {
                auto* b = reinterpret_cast<hsa_barrier_and_packet_t*>(p);
                what = "barrier-and";
                for (int i = 0; i < 5; ++i)
                    if (b->dep_signal[i].handle)
                        what += fmt(" dep[%d]=%" PRId64, i, (int64_t)hsa_signal_load_relaxed(b->dep_signal[i]));
                if (b->completion_signal.handle)
                    what += fmt(" done=%" PRId64, (int64_t)hsa_signal_load_relaxed(b->completion_signal));
            } else {
                what = fmt("packet type %d", type);
            }
        }
        RAD_ERR("  device %d queue %p: written %" PRIu64 ", read %" PRIu64 " -> %s", s->ord,
                (void*)s->q, wr, rd, what.c_str());
    }
    if (g_dump_lanes) g_dump_lanes();
}

/* ------------------------------------------------------------------ the memory-fault report
 *
 * Explains a GPU memory fault in this process's terms. The runtime reports only the card and the
 * address, which cannot distinguish a use-after-free, an overrun of a live buffer, or a bogus
 * pointer. So every range this backend hands out is kept with its kind, size and order, along
 * with the latest frees, and on a fault the address and the running kernel's pointer arguments
 * are each located against both.
 *
 * The process then ends at once, without a core. A fault stops its queue for good, and the
 * runtime's next step depends on the core-size limit, uselessly either way. With no core limit,
 * HIP's handler claims the event (HIP_SKIP_ABORT_ON_GPU_ERROR defaults on) and holds the error
 * for a HIP call this backend never makes, so every wait spins on the stopped queue and the
 * server hangs. With a limit, the runtime writes a GPU core of every device mapping before
 * aborting, tens of GB for a served model. So the report is followed by fatal(): exit status 70,
 * no destructors (they would wait on that queue), no core; a supervisor restarts the server. */
struct AllocRec {
    uintptr_t base = 0;
    int64_t   bytes = 0;
    int       kind = 0;
    uint64_t  seq = 0;            /* the order it was handed out in, from 1 */
    uint64_t  freed_ns = 0;       /* when it was freed (CLOCK_MONOTONIC); 0 while it is live */
};
std::mutex                    g_alloc_mu;
std::map<uintptr_t, AllocRec> g_live;       /* by base */
std::deque<AllocRec>          g_freed;      /* the latest frees, oldest first */
uint64_t                      g_alloc_seq = 0;
constexpr size_t              kFreedKept = 1024;

uint64_t mono_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

const char* mem_kind(int kind) {
    switch (kind) {
        case RAD_MEM_DEVICE:      return "device";
        case RAD_MEM_HOST:        return "host";
        case RAD_MEM_HOST_PINNED: return "host-pinned";
        case RAD_MEM_HOST_MAPPED: return "host-mapped";
        default:                  return "backend";
    }
}

void note_alloc(const void* p, int64_t bytes, int kind) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_alloc_mu);
    g_live[(uintptr_t)p] = AllocRec{(uintptr_t)p, bytes, kind, ++g_alloc_seq, 0};
}

void note_free(const void* p) {
    if (!p) return;
    std::lock_guard<std::mutex> lk(g_alloc_mu);
    auto it = g_live.find((uintptr_t)p);
    if (it == g_live.end()) return;
    AllocRec r = it->second;
    g_live.erase(it);
    r.freed_ns = mono_ns();
    g_freed.push_back(r);
    if (g_freed.size() > kFreedKept) g_freed.pop_front();
}

std::string rec_str(const AllocRec& r, uintptr_t a) {
    return fmt("#%" PRIu64 " %s %s at %#" PRIxPTR " +%s", r.seq, mem_kind(r.kind),
               humanb(r.bytes).c_str(), r.base, humanb((int64_t)(a - r.base)).c_str());
}

/* Where address `a` lies: inside a live range, inside a freed one (newest first; a reallocated
 * range also holds it live), or in neither, with the live ranges on either side. Called with
 * g_alloc_mu held. */
std::string where_is(uintptr_t a) {
    auto it = g_live.upper_bound(a);
    if (it != g_live.begin()) {
        const AllocRec& r = std::prev(it)->second;
        if (a < r.base + (uintptr_t)r.bytes) return "inside live " + rec_str(r, a);
    }
    const uint64_t now = mono_ns();
    for (auto f = g_freed.rbegin(); f != g_freed.rend(); ++f)
        if (a >= f->base && a < f->base + (uintptr_t)f->bytes)
            return fmt("inside FREED %s, freed %.1f s ago", rec_str(*f, a).c_str(),
                       (double)(now - f->freed_ns) * 1e-9);
    std::string s = "in no range this process holds or recently freed";
    if (it != g_live.begin()) {
        const AllocRec& r = std::prev(it)->second;
        s += fmt("; %s past the end of #%" PRIu64 " %s %s at %#" PRIxPTR,
                 humanb((int64_t)(a - (r.base + (uintptr_t)r.bytes))).c_str(), r.seq,
                 mem_kind(r.kind), humanb(r.bytes).c_str(), r.base);
    }
    if (it != g_live.end())
        s += fmt("; %s before #%" PRIu64 " %s %s at %#" PRIxPTR,
                 humanb((int64_t)(it->second.base - a)).c_str(), it->second.seq,
                 mem_kind(it->second.kind), humanb(it->second.bytes).c_str(), it->second.base);
    return s;
}

/* The runtime allocation containing the address, if any. Covers what this backend did not
 * allocate (a kernel library's own hipMalloc, runtime internals). */
std::string runtime_view(uintptr_t a) {
    hsa_amd_pointer_info_t info{};
    info.size = sizeof info;
    if (hsa_amd_pointer_info((void*)a, &info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS)
        return "the runtime could not be asked";
    if (info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) return "the runtime holds no allocation there";
    const char* t = info.type == HSA_EXT_POINTER_TYPE_HSA ? "runtime memory"
                  : info.type == HSA_EXT_POINTER_TYPE_LOCKED ? "a locked host range"
                  : info.type == HSA_EXT_POINTER_TYPE_IPC ? "an IPC import"
                  : info.type == HSA_EXT_POINTER_TYPE_GRAPHICS ? "a graphics interop range"
                  : "a runtime range";
    return fmt("the runtime has %s of %s at %p", t, humanb((int64_t)info.sizeInBytes).c_str(),
               info.agentBaseAddress);
}

/* A word that could be an address in this process: canonical user space, past the null page. */
bool looks_like_pointer(uint64_t w) { return w >= 0x10000 && w < 0x0000800000000000ull; }

hsa_status_t on_system_event(const hsa_amd_event_t* e, void*) {
    if (!e || e->event_type != HSA_AMD_GPU_MEMORY_FAULT_EVENT) return HSA_STATUS_ERROR;
    const hsa_amd_gpu_memory_fault_info_t& f = e->memory_fault;
    const uintptr_t va = (uintptr_t)f.virtual_address;
    int card = -1;
    for (size_t i = 0; i < G().dev.size(); ++i)
        if (G().dev[i].agent.handle == f.agent.handle) card = (int)i;
    std::string why;
    const uint32_t m = f.fault_reason_mask;
    if (m & HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT) why += " page-not-present";
    if (m & HSA_AMD_MEMORY_FAULT_READ_ONLY)        why += " write-to-read-only";
    if (m & HSA_AMD_MEMORY_FAULT_NX)               why += " execute";
    if (m & HSA_AMD_MEMORY_FAULT_HOST_ONLY)        why += " host-only";
    if (m & HSA_AMD_MEMORY_FAULT_IMPRECISE)        why += " imprecise-address";
    {
        std::lock_guard<std::mutex> lk(g_alloc_mu);
        RAD_ERR("device %d: GPU memory fault at %#" PRIxPTR " (%s ): %s; %s", card, va,
                why.empty() ? " ?" : why.c_str(), where_is(va).c_str(), runtime_view(va).c_str());
    }
    /* The running kernel and its arguments. A queue's read index stays on a packet until it
     * completes, so the packet there is in flight. Every pointer-shaped argument word is located
     * like the address, which names the operand that got a stale or short buffer. Only card
     * `card`'s queues; arguments are read back through the BAR. */
    std::lock_guard<std::mutex> sl(G().streams_mu);
    for (Stream* s : G().streams) {
        if (s->ord != card) continue;
        const uint64_t rd = hsa_queue_load_read_index_scacquire(s->q);
        const uint64_t wr = hsa_queue_load_write_index_scacquire(s->q);
        if (rd >= wr) continue;
        auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) + (rd & s->mask);
        const uint16_t h = __atomic_load_n(&p->header, __ATOMIC_ACQUIRE);
        if (((h >> HSA_PACKET_HEADER_TYPE) & 0xff) != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;
        std::string name;
        {
            std::lock_guard<std::mutex> nl(g_names_mu);
            auto it = g_names.find(p->kernel_object);
            name = it != g_names.end() ? it->second : fmt("%#" PRIx64, p->kernel_object);
        }
        RAD_ERR("device %d: queue %p packet %" PRIu64 " (of %" PRIu64 " written) ran kernel %s", card,
                (void*)s->q, rd, wr, name.c_str());
        const uint64_t* ka = static_cast<const uint64_t*>(p->kernarg_address);
        if (!ka) continue;
        std::lock_guard<std::mutex> lk(g_alloc_mu);
        for (uint32_t i = 0; i < kKargMax / 8; ++i) {
            const uint64_t w = ka[i];
            if (!looks_like_pointer(w)) continue;
            RAD_ERR("  argument +%u = %#" PRIx64 ": %s", i * 8, w, where_is((uintptr_t)w).c_str());
        }
    }
    fatal("a GPU memory fault on device %d stopped its queue for good (the report is above); "
          "exiting so a restart can bring the server back", card);
}

/* Stall detector for a spin loop. tick() reads the clock every 64k iterations, a few thousand
 * times a second at most.
 *
 * A long wait is not a fault: a long prefill at deep context holds one wait for seconds, and a
 * video generator's denoising pass for tens of them. So the queues' read indices are sampled once
 * every stall period and only a period in which none advanced reports, with the packet each queue
 * is on: a hung kernel, or a queue the driver evicted and could not restore. A wait that keeps
 * moving says nothing. */
struct Stall {
    const char* where;
    uint32_t n = 0;
    bool told = false;
    bool sampled = false;
    uint64_t progress = 0;
    std::chrono::steady_clock::time_point next =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kStallMs);
    explicit Stall(const char* w) : where(w) {}
    void tick() {
        if (told || (++n & 0xffff)) return;
        check();
    }
    /* Checks on every call, for a loop that sleeps between calls. */
    void check() {
        if (told) return;
        const auto now = std::chrono::steady_clock::now();
        if (now < next) return;
        next = now + std::chrono::milliseconds(kStallMs);
        const uint64_t p = read_progress();
        if (sampled && p == progress) {
            told = true;
            dump_queues(where);
            RAD_ERR("device: no queue has advanced in the %u s since: the GPU is not running the "
                    "packet named above -- a hung kernel, or a queue the driver evicted and could "
                    "not restore (amdgpu cwsr_enable=0 is one cause)", kStallMs / 1000);
            return;
        }
        sampled = true;
        progress = p;
    }
};

/* Reserves packet `idx` once the packet processor has consumed far enough that its slot is free.
 * A launched barrier-bit packet means every packet before it has completed, which a kernel
 * argument slot (read for the kernel's whole run) needs. A played tape clears the bit on at most
 * kTapeOverlapRun consecutive packets (TapeRec::overlap), so the margin is that run plus one. */
constexpr uint64_t kReserveMargin = (uint64_t)kTapeOverlapRun + 2;

inline int reserve(Stream* s, uint64_t* out) {
    const uint64_t idx = s->wr;
    const uint64_t size = s->mask + 1;
    if (idx + kReserveMargin - s->rd > size) {
        Stall st("a full queue");
        for (;;) {
            s->rd = hsa_queue_load_read_index_scacquire(s->q);
            if (idx + kReserveMargin - s->rd <= size) break;
            if (s->fault.load(std::memory_order_relaxed)) return RAD_E_DEVICE;
            /* Behind a closed gate the queue drains only once the host opens it, and the host is
             * the one waiting here. */
            if (s->gates.load(std::memory_order_relaxed) > 0) {
                device_set_error("a submission onto a full queue behind a closed gate: open the "
                                 "gate (rad_gate_open) first");
                return RAD_E_STATE;
            }
            st.tick();
            _mm_pause();
        }
    }
    *out = idx;
    return RAD_OK;
}

inline void publish(Stream* s, uint64_t idx, uint16_t header, uint16_t setup) {
    s->wr = idx + 1;
    hsa_queue_store_write_index_relaxed(s->q, idx + 1);
    _mm_sfence();                       /* the argument slot, written before the header */
    __atomic_store_n(reinterpret_cast<uint32_t*>(
                         static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) +
                         (idx & s->mask)),
                     (uint32_t)header | ((uint32_t)setup << 16), __ATOMIC_RELEASE);
    hsa_signal_store_relaxed(s->q->doorbell_signal, (hsa_signal_value_t)idx);
}

/* ------------------------------------------------------------------ recorded passes
 *
 * A pass is recorded once and then written straight into the queues. The engine issues the same
 * kernels with the same arguments step after step, and nearly all of a dispatch's host cost is
 * producing it (block code, operand resolution, argument packing). A recorded dispatch is the
 * finished packet plus its argument block, kept in VRAM for the tape's lifetime, so playback is a
 * 60-byte copy into the queue and a header store.
 *
 * Recorded: what this thread submits while a capture is open -- dispatches (kernels and the
 * backend's own copies and fills) and event records and waits. Events are kept by event and
 * replayed through the ordinary calls, because each record takes a fresh signal (see "events").
 * A pageable copy stages through the host and waits, which a tape cannot reproduce, so it makes
 * the tape unusable. Other threads' submissions are not recorded. */
enum : uint8_t { kTapeDispatch = 0, kTapeRecord = 1, kTapeWait = 2 };

struct TapeRec {
    uint8_t  kind = kTapeDispatch;
    uint8_t  fence = 0;                 /* a dispatch's kFence* flags */
    uint32_t karg_off = 0;              /* a dispatch's argument block, in the tape's */
    uint32_t karg_len = 0;
    Stream*  s = nullptr;
    void*    ev = nullptr;              /* a record's or a wait's event */
    hsa_kernel_dispatch_packet_t pkt{}; /* a dispatch's packet, all but the header word */
    /* Play without the barrier bit, so the packet processor may launch it while earlier packets
     * on the queue still run. Set only where the dispatch touches nothing those packets write
     * (tape_set_overlap). */
    uint8_t  overlap = 0;
};

struct Tape {
    uint64_t             magic = kTapeMagic;
    std::vector<TapeRec> rec;
    std::vector<unsigned char> karg_host;   /* the argument blocks, kept to compare tapes by */
    char*                karg = nullptr;    /* placed at the end of the capture */
    std::string          bad;               /* why this tape cannot be played, if it cannot */
    /* Last packet the latest play wrote on each queue; the argument block is freed only after
     * all of them are consumed. */
    std::vector<std::pair<uint32_t, uint64_t>> last;
};

thread_local Tape* t_tape = nullptr;
thread_local int   t_tape_pause = 0;

inline bool capturing() { return t_tape && t_tape_pause == 0; }

/* A dispatch just written at `p` on `s`, with its argument block `ka`. */
void tape_note_dispatch(Stream* s, const hsa_kernel_dispatch_packet_t* p, uint16_t setup,
                        const unsigned char* ka, uint32_t n, uint8_t fence) {
    Tape* t = t_tape;
    TapeRec r;
    r.kind = kTapeDispatch;
    r.fence = fence;
    r.s = s;
    r.pkt = *p;
    r.pkt.header = 0;
    r.pkt.setup = setup;
    const size_t off = (t->karg_host.size() + 63) & ~size_t(63);
    t->karg_host.resize(off + n);
    std::memcpy(t->karg_host.data() + off, ka, n);
    r.karg_off = (uint32_t)off;
    r.karg_len = n;
    t->rec.push_back(r);
}

inline uint16_t header(hsa_packet_type_t type, int acq, int rel, bool barrier = true) {
    return (uint16_t)((type << HSA_PACKET_HEADER_TYPE) |
                      ((barrier ? 1u : 0u) << HSA_PACKET_HEADER_BARRIER) |
                      (acq << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                      (rel << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE));
}

/* Whether the packet at `idx` must acquire at system scope, and the bookkeeping if it does. */
inline int acquire_scope(Stream* s, uint64_t idx, bool forced) {
    const uint64_t ep = G().host_epoch.load(std::memory_order_acquire);
    const bool lap = (int64_t)idx - s->last_sys_acq >= (int64_t)(kQueuePackets / 2);
    if (forced || lap || s->epoch != ep) {
        s->last_sys_acq = (int64_t)idx;
        s->epoch = ep;
        /* The prefetch stream reads nothing the host writes; a system acquire there would only
         * invalidate the L2 under the compute stream. */
        if (s->agent_only && !forced) return HSA_FENCE_SCOPE_AGENT;
        return HSA_FENCE_SCOPE_SYSTEM;
    }
    return HSA_FENCE_SCOPE_AGENT;
}

int barrier(Stream* s, hsa_signal_t dep, hsa_signal_t done, int acq, int rel,
            uint64_t* out_idx = nullptr) {
    s->lock();
    uint64_t idx = 0;
    const int rc = reserve(s, &idx);
    if (rc < 0) { s->unlock(); return rc; }
    if (out_idx) *out_idx = idx;
    auto* p = reinterpret_cast<hsa_barrier_and_packet_t*>(
        static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) + (idx & s->mask));
    p->reserved0 = 0;
    p->reserved1 = 0;
    p->dep_signal[0] = dep;
    for (int i = 1; i < 5; ++i) p->dep_signal[i].handle = 0;
    p->reserved2 = 0;
    p->completion_signal = done;
    if (acq == HSA_FENCE_SCOPE_SYSTEM) {
        s->last_sys_acq = (int64_t)idx;
        s->epoch = G().host_epoch.load(std::memory_order_acquire);
    }
    publish(s, idx, header(HSA_PACKET_TYPE_BARRIER_AND, acq, rel), 0);
    s->unlock();
    return RAD_OK;
}

int dispatch(Stream* s, const KernelRec* k, dim3 g, dim3 b, void** args, size_t shm, uint8_t fence) {
    const uint64_t gx = (uint64_t)g.x * b.x, gy = (uint64_t)g.y * b.y, gz = (uint64_t)g.z * b.z;
    if (!gx || !gy || !gz) return RAD_OK;
    if (gx > UINT32_MAX || gy > UINT32_MAX || gz > UINT32_MAX) return RAD_E_INVAL;
    if ((uint64_t)k->group_size + shm > 65536) return RAD_E_INVAL;

    alignas(64) unsigned char ka[kKargMax];
    const uint32_t n = (k->kernarg_size + 15u) & ~15u;
    std::memset(ka, 0, n);
    for (uint16_t i = 0; i < k->n_explicit; ++i)
        std::memcpy(ka + k->ex_off[i], args[i], k->ex_size[i]);
    auto put = [&](Hidden h, uint64_t v) {
        const int32_t off = k->hid[(int)h];
        if (off >= 0) std::memcpy(ka + off, &v, k->hid_size[(int)h]);
    };
    put(Hidden::BlockCountX, g.x);  put(Hidden::BlockCountY, g.y);  put(Hidden::BlockCountZ, g.z);
    put(Hidden::GroupSizeX, b.x);   put(Hidden::GroupSizeY, b.y);   put(Hidden::GroupSizeZ, b.z);
    put(Hidden::GridDims, 3);
    put(Hidden::DynamicLds, shm);

    s->lock();
    uint64_t idx = 0;
    const int rc = reserve(s, &idx);
    if (rc < 0) { s->unlock(); return rc; }
    char* slot = s->karg + (size_t)(idx & s->mask) * kKargSlot;
    std::memcpy(slot, ka, n);
    auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) + (idx & s->mask);
    p->workgroup_size_x = (uint16_t)b.x;
    p->workgroup_size_y = (uint16_t)b.y;
    p->workgroup_size_z = (uint16_t)b.z;
    p->reserved0 = 0;
    p->grid_size_x = (uint32_t)gx;
    p->grid_size_y = (uint32_t)gy;
    p->grid_size_z = (uint32_t)gz;
    p->private_segment_size = k->private_size;
    p->group_segment_size = k->group_size + (uint32_t)shm;
    p->kernel_object = k->object;
    p->kernarg_address = slot;
    p->reserved2 = 0;
    p->completion_signal.handle = 0;
    constexpr uint16_t setup = 3u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    if (capturing()) tape_note_dispatch(s, p, setup, ka, n, fence);
    /* Attach the ride after the tape copied the packet: a recorded packet carries no signal, and
     * a play attaches whichever ride is pending then (PlayRun::put). */
    if (s->ride.handle) {
        p->completion_signal = s->ride;
        s->ride.handle = 0;
    }
    const int acq = acquire_scope(s, idx, (fence & kFenceAcquireSystem) != 0);
    const int rel = (fence & kFenceReleaseSystem) ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT;
    publish(s, idx, header(HSA_PACKET_TYPE_KERNEL_DISPATCH, acq, rel), setup);
    s->unlock();
    return RAD_OK;
}

/* A run of recorded dispatches onto one queue, under one hold of the queue lock, ringing the
 * doorbell once per kPlayRing packets: the doorbell is an uncached write across the link and a
 * pass is over a thousand dispatches. Each body is copied as recorded; the header is decided now
 * as in dispatch(), since its acquire scope depends on host writes since recording. Argument
 * blocks were placed when the tape was made, so no fence precedes the header. Ring
 * before a reserve that would wait: the packet processor only takes announced packets. */
constexpr int kPlayRing = 64;

/* Where this thread's plays last left each queue. If a queue has moved since, it holds packets
 * no recording saw (a weight-table patch or mover wait between played stretches, another thread's
 * copy), so the next played packet keeps its barrier bit: the overlap analysis only proved it
 * independent of the recorded packets before it. A queue not in the list is treated the same. */
struct PlayLeft { uint32_t q = UINT32_MAX; uint64_t wr = 0; };
constexpr int kPlayLeft = 8;
thread_local PlayLeft t_left[kPlayLeft];

struct PlayRun {
    Stream*  s = nullptr;
    uint64_t next = 0;        /* the write index after the last packet written */
    int      unrung = 0;
    bool     foreign = true;  /* the next packet's queue has packets this thread did not play */

    void ring() {
        if (!unrung) return;
        hsa_queue_store_write_index_relaxed(s->q, next);
        hsa_signal_store_relaxed(s->q->doorbell_signal, (hsa_signal_value_t)(next - 1));
        unrung = 0;
    }
    void end() {
        if (!s) return;
        ring();
        int i = 0, empty = -1;
        while (i < kPlayLeft && t_left[i].q != s->id) {
            if (t_left[i].q == UINT32_MAX && empty < 0) empty = i;
            ++i;
        }
        if (i == kPlayLeft) i = empty >= 0 ? empty : (int)(s->id % kPlayLeft);
        t_left[i] = PlayLeft{ s->id, s->wr };
        s->unlock();
        s = nullptr;
    }
    static bool moved(const Stream* q) {
        for (int i = 0; i < kPlayLeft; ++i)
            if (t_left[i].q == q->id) return t_left[i].wr != q->wr;
        return true;
    }
    /* `done`, when set, is the packet's completion signal and `rel_min` its minimum release
     * scope, for a record riding on this dispatch (see a_tape_play). */
    int put(const TapeRec& r, uint64_t* out_idx, hsa_signal_t done = hsa_signal_t{0},
            int rel_min = HSA_FENCE_SCOPE_NONE) {
        if (s != r.s) {
            end();
            s = r.s;
            s->lock();
            foreign = moved(s);
        }
        const uint64_t idx = s->wr;
        if (idx + kReserveMargin - s->rd > s->mask + 1) {
            ring();
            uint64_t got = 0;
            const int rc = reserve(s, &got);
            if (rc < 0) return rc;
        }
        auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) + (idx & s->mask);
        std::memcpy(reinterpret_cast<char*>(p) + 4, reinterpret_cast<const char*>(&r.pkt) + 4,
                    sizeof(hsa_kernel_dispatch_packet_t) - 4);
        /* A ride left pending by host work between played stretches (rad_ride) is completed by
         * the first played dispatch, as with an issued one. If a record already rides on this
         * dispatch, the ride waits for the next. */
        if (!done.handle && s->ride.handle) {
            done = s->ride;
            s->ride.handle = 0;
        }
        if (done.handle) p->completion_signal = done;
        const int acq = acquire_scope(s, idx, (r.fence & kFenceAcquireSystem) != 0);
        int rel = (r.fence & kFenceReleaseSystem) ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT;
        if (rel < rel_min) rel = rel_min;
        /* A dispatch carrying a record keeps the bit: its completion must cover everything before
         * it on the queue, which an unordered dispatch's does not. */
        const bool ordered = foreign || !r.overlap || done.handle;
        const uint16_t h = header(HSA_PACKET_TYPE_KERNEL_DISPATCH, acq, rel, ordered);
        __atomic_store_n(reinterpret_cast<uint32_t*>(p), (uint32_t)h | ((uint32_t)r.pkt.setup << 16),
                         __ATOMIC_RELEASE);
        foreign = false;
        s->wr = idx + 1;
        next = idx + 1;
        *out_idx = idx;
        if (++unrung >= kPlayRing) ring();
        return RAD_OK;
    }
};

/* One interrupt signal per thread for synchronous waits (see "host waits"). A signal is only
 * waited on by the thread that armed it, so two threads waiting on one queue do not share. */
hsa_signal_t thread_signal() {
    thread_local hsa_signal_t s{0};
    if (!s.handle && hsa_signal_create(0, 0, nullptr, &s) != HSA_STATUS_SUCCESS) s.handle = 0;
    return s;
}

/* ------------------------------------------------------------------ host waits
 *
 * A host wait blocks in the driver until the device completes. Every signal the host waits on is
 * an interrupt signal: when its packet completes, the packet processor posts the signal's event
 * and raises an interrupt, and the driver wakes the threads blocked on it. A wait costs one
 * syscall and one wake-up regardless of length, and never guesses the device's duration. The
 * cost is interrupt delivery, some tens of us; the engine hides it by keeping work queued behind
 * every point it waits on (the next step behind a step's end, independent layers behind a host
 * op's join), so the device is still busy when the host wakes.
 *
 * The wait goes to the driver directly. The runtime's blocking wait polls before it blocks,
 * burning a core for a short wait's whole length; the driver's event wait costs nothing until
 * the interrupt.
 *
 * No wake-up is lost. The driver counts each event's firings (its age), and a wait naming an age
 * returns at once if the event has fired since. Each thread keeps the age each event last woke it
 * at, so an interrupt between reading the signal and entering the driver ends the wait; a stale
 * age costs one early return.
 *
 * A fault ends every wait. A faulted queue never completes its signals, so every wait also names
 * the fault signal's event, which the queue-error callback fires. */

/* The age an event last woke this thread at; 1 is the driver's "never fired". */
uint64_t& seen_age(uint32_t ev) {
    thread_local std::vector<uint64_t> age;
    if (ev >= age.size()) age.resize((size_t)ev + 1, 1);
    return age[ev];
}

int host_wait(hsa_signal_t sig) {
    if (hsa_signal_load_scacquire(sig) != 0) {
        if (!interrupts(sig)) {
            device_set_error("a host wait on a signal that raises no interrupt");
            return RAD_E_DEVICE;
        }
        Stall st("a synchronisation");
        kfd_event_data ev[2] = {};
        ev[0].event_id = reinterpret_cast<const amd_signal_t*>(sig.handle)->event_id;
        ev[1].event_id = reinterpret_cast<const amd_signal_t*>(G().fault.handle)->event_id;
        do {
            if (G().faulted.load(std::memory_order_acquire)) return RAD_E_DEVICE;
            for (kfd_event_data& e : ev) e.signal_event_data.last_event_age = seen_age(e.event_id);
            kfd_ioctl_wait_events_args w{};
            w.events_ptr = (uint64_t)(uintptr_t)ev;
            w.num_events = 2;
            w.wait_for_all = 0;
            w.timeout = kStallMs;
            if (ioctl(G().kfd, AMDKFD_IOC_WAIT_EVENTS, &w) < 0) {
                if (errno == EINTR) continue;
                device_set_error("a host wait failed in the driver: %s", std::strerror(errno));
                return RAD_E_DEVICE;
            }
            for (const kfd_event_data& e : ev) seen_age(e.event_id) = e.signal_event_data.last_event_age;
            if (w.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT) st.check();
        } while (hsa_signal_load_scacquire(sig) != 0);
    }
    G().host_epoch.fetch_add(1, std::memory_order_acq_rel);
    return RAD_OK;
}

/* The arming store is silent: an ordinary store to an interrupt signal also fires its event,
 * which wakes nobody and costs a syscall. */
int stream_drain(Stream* s) {
    /* A pageable copy or a destroy behind a closed gate would wait for a barrier the host has not
     * released yet. */
    if (s->gates.load(std::memory_order_acquire) > 0) {
        device_set_error("a host wait on a queue at a closed gate: open the gate "
                         "(rad_gate_open) first");
        return RAD_E_STATE;
    }
    hsa_signal_t sig = thread_signal();
    if (!sig.handle) return RAD_E_DEVICE;
    hsa_signal_silent_store_relaxed(sig, 1);
    uint64_t idx = 0;
    RAD_TRY(barrier(s, hsa_signal_t{0}, sig, HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_SYSTEM, &idx));
    return host_wait(sig);
}

/* ------------------------------------------------------------------ where a pointer lives */

enum Loc : uint8_t { kDevLocal, kDevPeer, kHost, kPageable };

/* Per-thread cache of pointer locations. hsa_amd_pointer_info takes the runtime's global lock and
 * walks its allocation map; every copy asks about both ends, a few hundred queries per step on
 * the same dozen allocations, ~17 us of each rank thread's step.
 *
 * Only live allocations are cached, never "pageable": unknown memory may be registered later,
 * while a live allocation's kind cannot change. Which allocation an address belongs to changes
 * only by a free or unmap, and each of those in this backend bumps `g_loc_gen` first; entries of
 * an older generation are ignored. Ranges come from the runtime (sizeInBytes), so a hit equals the
 * call's answer. Reserved-but-unbacked addresses are not cached; their card is looked up per
 * page. */
struct LocHit {
    uintptr_t lo = 0, hi = 0;   /* the allocation's range, as the host addresses it */
    intptr_t  shift = 0;        /* agent address minus host address (pinned host memory) */
    uint64_t  owner = 0;        /* the owning card's agent handle; 0 for host memory */
    uint64_t  gen = 0;
};
constexpr int kLocWays = 16;
thread_local LocHit t_loc[kLocWays];
thread_local unsigned t_loc_next = 0;
std::atomic<uint64_t> g_loc_gen{1};

/* Invalidates every thread's cache. */
void loc_forget() { g_loc_gen.fetch_add(1, std::memory_order_acq_rel); }

void loc_keep(const void* host_base, size_t bytes, intptr_t shift, uint64_t owner, uint64_t gen) {
    if (!host_base || !bytes) return;
    LocHit& h = t_loc[t_loc_next++ % kLocWays];
    h.lo = (uintptr_t)host_base;
    h.hi = h.lo + bytes;
    h.shift = shift;
    h.owner = owner;
    h.gen = gen;
}

/* Host ranges this backend pinned itself (see "pinned host memory, in pieces"), ordered by base
 * so an interior address finds its range. `pool` is the host-pool allocation holding the range,
 * or null if it was locked in pieces. */
struct Pinned {
    int64_t len = 0;
    void* pool = nullptr;
};
std::mutex g_pinned_mu;
std::map<uintptr_t, Pinned> g_pinned;

bool pinned_range(const void* p, uintptr_t* lo, int64_t* len) {
    const uintptr_t a = (uintptr_t)p;
    std::lock_guard<std::mutex> lk(g_pinned_mu);
    auto it = g_pinned.upper_bound(a);
    if (it == g_pinned.begin()) return false;
    --it;
    if (a - it->first >= (uintptr_t)it->second.len) return false;
    if (lo) *lo = it->first;
    if (len) *len = it->second.len;
    return true;
}

Loc locate(const void* p, const Stream* s, const void** gpu) {
    *gpu = p;
    const uintptr_t a = (uintptr_t)p;
    const uint64_t gen = g_loc_gen.load(std::memory_order_acquire);
    for (const LocHit& h : t_loc) {
        if (h.gen != gen || a - h.lo >= h.hi - h.lo) continue;
        *gpu = (const void*)(a + (uintptr_t)h.shift);
        if (!h.owner) return kHost;
        return h.owner == s->agent.handle ? kDevLocal : kDevPeer;
    }
    /* Ranges pinned here are answered here: hsa_amd_pointer_info reports memory locked with
     * hsa_amd_memory_lock_to_pool as UNKNOWN (like pageable), although the lock mapped it on
     * every card at its own address. */
    uintptr_t plo = 0;
    int64_t plen = 0;
    if (pinned_range(p, &plo, &plen)) {
        loc_keep((const void*)plo, (size_t)plen, 0, 0, gen);
        return kHost;
    }
    hsa_amd_pointer_info_t info{};
    info.size = sizeof info;
    if (hsa_amd_pointer_info(const_cast<void*>(p), &info, nullptr, nullptr, nullptr) !=
        HSA_STATUS_SUCCESS)
        return kPageable;
    switch (info.type) {
        case HSA_EXT_POINTER_TYPE_LOCKED:
            *gpu = (const char*)info.agentBaseAddress + ((const char*)p - (const char*)info.hostBaseAddress);
            loc_keep(info.hostBaseAddress, info.sizeInBytes,
                     (intptr_t)((uintptr_t)info.agentBaseAddress - (uintptr_t)info.hostBaseAddress),
                     0, gen);
            return kHost;
        case HSA_EXT_POINTER_TYPE_HSA:
        case HSA_EXT_POINTER_TYPE_HSA_VMEM:
        case HSA_EXT_POINTER_TYPE_GRAPHICS: {
            hsa_device_type_t t;
            if (hsa_agent_get_info(info.agentOwner, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS)
                return kPageable;
            if (t == HSA_DEVICE_TYPE_CPU) {
                loc_keep(info.agentBaseAddress, info.sizeInBytes, 0, 0, gen);
                return kHost;
            }
            loc_keep(info.agentBaseAddress, info.sizeInBytes, 0, info.agentOwner.handle, gen);
            return info.agentOwner.handle == s->agent.handle ? kDevLocal : kDevPeer;
        }
        case HSA_EXT_POINTER_TYPE_IPC:
            return kDevPeer;
        case HSA_EXT_POINTER_TYPE_RESERVED_ADDR: {
            int d = 0;
            if (aql_vmem_device_of(p, &d) < 0) return kDevLocal;
            return d == s->ord ? kDevLocal : kDevPeer;
        }
        default:
            return kPageable;
    }
}

std::atomic<int64_t> g_pageable_bytes{0};

}  // namespace

void set_next_fence(uint8_t f) { t_next_fence = f; }

}  // namespace aql

/* ------------------------------------------------------------------ the backend */

namespace {

using namespace aql;

int a_stream_create(RadStream* out, int high_priority) {
    if (!out) return RAD_E_INVAL;
    int ord = 0;
    if (hipGetDevice(&ord) != hipSuccess || ord < 0 || ord >= (int)G().dev.size()) {
        device_set_error("rad_stream_create: no current device");
        return RAD_E_DEVICE;
    }
    const Dev& d = G().dev[(size_t)ord];
    auto s = std::make_unique<Stream>();
    s->ord = ord;
    s->agent = d.agent;
    if (hsa_queue_create(d.agent, kQueuePackets, HSA_QUEUE_TYPE_SINGLE, on_queue_error, s.get(),
                         UINT32_MAX, UINT32_MAX, &s->q) != HSA_STATUS_SUCCESS) {
        device_set_error("rad_stream_create: hsa_queue_create failed on device %d", ord);
        return RAD_E_DEVICE;
    }
    s->mask = s->q->size - 1;
    hsa_amd_profiling_set_profiler_enabled(s->q, 1);
    if (high_priority) hsa_amd_queue_set_priority(s->q, HSA_AMD_QUEUE_PRIORITY_HIGH);
    void* ka = nullptr;
    hsa_amd_memory_pool_t kpool = d.host_kargs ? G().host_fine : d.vram;
    hsa_agent_t kpeer = d.host_kargs ? d.agent : G().cpu;
    if (hsa_amd_memory_pool_allocate(kpool, (size_t)(s->mask + 1) * kKargSlot, 0, &ka) !=
            HSA_STATUS_SUCCESS ||
        hsa_amd_agents_allow_access(1, &kpeer, nullptr, ka) != HSA_STATUS_SUCCESS) {
        hsa_queue_destroy(s->q);
        device_set_error("rad_stream_create: the kernel argument ring could not be placed in %s",
                         d.host_kargs ? "pinned host memory" : "VRAM the host can write");
        return RAD_E_DEVICE;
    }
    s->karg = static_cast<char*>(ka);
    void* bb = nullptr;
    if (hsa_amd_memory_pool_allocate(G().host_fine, kBounceBytes, 0, &bb) != HSA_STATUS_SUCCESS ||
        hsa_amd_agents_allow_access(1, &d.agent, nullptr, bb) != HSA_STATUS_SUCCESS) {
        hsa_amd_memory_pool_free(ka);
        hsa_queue_destroy(s->q);
        device_set_error("rad_stream_create: no pinned bounce buffer");
        return RAD_E_NOMEM;
    }
    s->bounce = static_cast<char*>(bb);
    note_alloc(ka, (int64_t)(s->mask + 1) * kKargSlot, -1);
    note_alloc(bb, kBounceBytes, -1);
    s->id = g_next_queue_id.fetch_add(1, std::memory_order_relaxed);
    if (s->id >= kMaxQueues) {
        note_free(bb);
        note_free(ka);
        hsa_amd_memory_pool_free(bb);
        hsa_amd_memory_pool_free(ka);
        hsa_queue_destroy(s->q);
        device_set_error("rad_stream_create: more than %u streams in one process", kMaxQueues);
        return RAD_E_NOMEM;
    }
    g_queue_by_id[s->id].store(s->q, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(G().streams_mu);
        G().streams.push_back(s.get());
    }
    *out = (RadStream)s.release();
    return RAD_OK;
}

void a_stream_destroy(RadStream rs) {
    Stream* s = ours(rs);
    if (!s) return;
    (void)stream_drain(s);
    {
        std::lock_guard<std::mutex> lk(G().streams_mu);
        auto& v = G().streams;
        for (size_t i = 0; i < v.size(); ++i) if (v[i] == s) { v.erase(v.begin() + (long)i); break; }
    }
    g_queue_by_id[s->id].store(nullptr, std::memory_order_release);
    hsa_queue_destroy(s->q);
    note_free(s->karg);
    note_free(s->bounce);
    hsa_amd_memory_pool_free(s->karg);
    hsa_amd_memory_pool_free(s->bounce);
    s->magic = 0;
    delete s;
}

int a_stream_sync(RadStream rs) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_stream_sync: not a stream of this backend"); return RAD_E_INVAL; }
    if (s->gates.load(std::memory_order_acquire) > 0) {
        device_set_error("rad_stream_sync: the queue waits at a closed gate, which only "
                         "rad_gate_open releases");
        return RAD_E_STATE;
    }
    const int rc = stream_drain(s);
    if (rc < 0) device_set_error("rad_stream_sync: the queue on device %d faulted", s->ord);
    return rc;
}

/* ---- events.
 *
 * Each record takes a fresh signal. A signal is reused only when its record has completed and
 * every wait packet enqueued against it has been consumed. Reusing it earlier re-arms it under a
 * lagging waiter, which then waits on the newer record; that record can sit behind a wait on the
 * same lagging queue, a deadlock that two lanes joining each other every layer hit within a step.
 *
 * So each signal carries the (queue, packet index) of every wait naming it and returns to the
 * free list when both conditions hold. The host never blocks on this; the pool grows to the
 * number of records in flight. */
struct Sig {
    hsa_signal_t h{};
    bool         irq = false;                              /* an interrupt signal */
    hsa_agent_t  agent{};                                  /* where it was recorded */
    std::vector<std::pair<uint32_t, uint64_t>> waiters;    /* queue id, packet index */
};

struct SigPool {
    std::mutex        mu;
    std::vector<Sig*> free[2];                             /* by irq */
    std::deque<Sig*>  retiring;                            /* in the order they were retired */
};

SigPool& pool() {
    static SigPool* p = new SigPool();
    return *p;
}

bool reusable(const Sig* g) {
    if (hsa_signal_load_relaxed(g->h) != 0) return false;
    for (const auto& w : g->waiters) {
        hsa_queue_t* q = g_queue_by_id[w.first].load(std::memory_order_acquire);
        if (q && hsa_queue_load_read_index_relaxed(q) <= w.second) return false;
    }
    return true;
}

/* A signal for one record: an interrupt signal if the host waits on the event, else GPU_ONLY.
 * Only interrupt signals can end a host wait (see "host waits"), but each costs a driver event (a
 * process has a few thousand) and an interrupt per completion, wasted on events recorded dozens
 * of times a step and waited on only by queues. */
Sig* sig_get(bool irq) {
    SigPool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    /* Oldest first. Signals retire in record order, so the front is most likely done; popping
     * from the front costs a look or two, where a full walk reads every signal and waiter queue
     * from device-written memory. The walk is the fallback for when the front is not done (the
     * host plays a step while the card is still on the previous one). */
    while (p.free[irq].empty() && !p.retiring.empty() && reusable(p.retiring.front())) {
        Sig* g = p.retiring.front();
        p.retiring.pop_front();
        g->waiters.clear();
        p.free[g->irq].push_back(g);
    }
    if (p.free[irq].empty()) {
        /* In place and in order, so the front stays the oldest. */
        size_t kept = 0;
        for (size_t i = 0; i < p.retiring.size(); ++i) {
            Sig* g = p.retiring[i];
            if (reusable(g)) {
                g->waiters.clear();
                p.free[g->irq].push_back(g);
            } else {
                p.retiring[kept++] = g;
            }
        }
        p.retiring.resize(kept);
    }
    if (!p.free[irq].empty()) {
        Sig* g = p.free[irq].back();
        p.free[irq].pop_back();
        return g;
    }
    auto g = std::make_unique<Sig>();
    g->irq = irq;
    const hsa_status_t st = irq ? hsa_signal_create(0, 0, nullptr, &g->h)
                                : hsa_amd_signal_create(0, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &g->h);
    if (st != HSA_STATUS_SUCCESS || (irq && !interrupts(g->h))) return nullptr;
    return g.release();
}

void sig_retire(Sig* g) {
    SigPool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    p.retiring.push_back(g);
}

struct Event {
    uint64_t   magic = kEventMagic;
    Sig*       cur = nullptr;           /* the latest record; null before the first */
    bool       local = false;           /* RAD_EVENT_LOCAL: the record releases card-wide */
    bool       host = false;            /* RAD_EVENT_HOST_WAIT: the host may synchronise on it */
    std::mutex mu;
};

inline Event* ev_of(RadEvent e) {
    Event* p = static_cast<Event*>(e);
    return (p && p->magic == kEventMagic) ? p : nullptr;
}

int a_event_create(RadEvent* out, unsigned flags) {
    if (!out) return RAD_E_INVAL;
    Event* e = new (std::nothrow) Event();
    if (!e) return RAD_E_NOMEM;
    e->local = (flags & RAD_EVENT_LOCAL) != 0;
    e->host = (flags & RAD_EVENT_HOST_WAIT) != 0;
    *out = (RadEvent)e;
    return RAD_OK;
}

/* The signal stays in the pool until the waits naming it are consumed. */
void a_event_destroy(RadEvent re) {
    Event* e = ev_of(re);
    if (!e) return;
    Sig* g = nullptr;
    {
        std::lock_guard<std::mutex> lk(e->mu);
        g = e->cur;
        e->cur = nullptr;
        e->magic = 0;
    }
    if (g) sig_retire(g);
    delete e;
}

/* A fresh signal armed for one record of `e` on `s`, plus the release scope the record needs. The
 * caller puts it on a packet, then passes it to record_adopt. */
Sig* record_arm(Event* e, Stream* s, int* rel) {
    Sig* g = sig_get(e->host);
    if (!g) return nullptr;
    hsa_signal_silent_store_relaxed(g->h, 1);
    g->agent = s->agent;
    /* A local event releases at agent scope, which its L2 already satisfies. System scope waits
     * for every in-flight write to host memory: up to tens of us while the link streams weights. */
    *rel = e->local ? HSA_FENCE_SCOPE_AGENT : HSA_FENCE_SCOPE_SYSTEM;
    return g;
}

/* The record carrying `g` is enqueued: it becomes the event's latest, and the previous signal
 * retires to the pool. */
void record_adopt(Event* e, Sig* g) {
    Sig* old = nullptr;
    {
        std::lock_guard<std::mutex> lk(e->mu);
        old = e->cur;
        e->cur = g;
    }
    if (old) sig_retire(old);
}

int a_event_record(RadEvent re, RadStream rs) {
    Event* e = ev_of(re);
    Stream* s = ours(rs);
    if (!e || !s) { device_set_error("rad_event_record: null or foreign argument"); return RAD_E_INVAL; }
    if (capturing()) {
        TapeRec r;
        r.kind = kTapeRecord;
        r.s = s;
        r.ev = re;
        t_tape->rec.push_back(r);
    }
    int rel = HSA_FENCE_SCOPE_NONE;
    Sig* g = record_arm(e, s, &rel);
    if (!g) { device_set_error("rad_event_record: no signal could be made"); return RAD_E_DEVICE; }
    uint64_t idx = 0;
    const int rc = barrier(s, hsa_signal_t{0}, g->h, HSA_FENCE_SCOPE_NONE, rel, &idx);
    if (rc < 0) return rc;               /* the queue faulted; the signal stays armed and unused */
    record_adopt(e, g);
    return RAD_OK;
}

/* The event lock is held across the enqueue so the signal cannot retire before this wait is
 * noted against it. */
int a_event_wait(RadStream rs, RadEvent re) {
    Event* e = ev_of(re);
    Stream* s = ours(rs);
    if (!e || !s) { device_set_error("rad_event_wait: null or foreign argument"); return RAD_E_INVAL; }
    if (capturing()) {
        TapeRec r;
        r.kind = kTapeWait;
        r.s = s;
        r.ev = re;
        t_tape->rec.push_back(r);
    }
    std::lock_guard<std::mutex> lk(e->mu);
    if (!e->cur) return RAD_OK;               /* never recorded: nothing to wait for */
    uint64_t idx = 0;
    /* A producer on this card wrote through the shared L2, so agent scope suffices; another card
     * or the host needs system scope. */
    const int acq = e->cur->agent.handle == s->agent.handle ? HSA_FENCE_SCOPE_AGENT
                                                            : HSA_FENCE_SCOPE_SYSTEM;
    RAD_TRY(barrier(s, e->cur->h, hsa_signal_t{0}, acq, HSA_FENCE_SCOPE_NONE, &idx));
    e->cur->waiters.emplace_back(s->id, idx);
    return RAD_OK;
}

int a_event_query(RadEvent re) {
    Event* e = ev_of(re);
    if (!e) return RAD_E_INVAL;
    std::lock_guard<std::mutex> lk(e->mu);
    if (!e->cur) return 1;
    return hsa_signal_load_scacquire(e->cur->h) == 0 ? 1 : 0;
}

/* Only an event made for host waits can be synchronised: its records carry interrupt signals. */
int a_event_sync(RadEvent re) {
    Event* e = ev_of(re);
    if (!e) return RAD_E_INVAL;
    if (!e->host) {
        device_set_error("rad_event_sync: the event was made without RAD_EVENT_HOST_WAIT, so its "
                         "records raise no interrupt a host wait could block on");
        return RAD_E_INVAL;
    }
    hsa_signal_t sig;
    {
        std::lock_guard<std::mutex> lk(e->mu);
        if (!e->cur) return RAD_OK;
        sig = e->cur->h;
    }
    return host_wait(sig);
}

/* The next dispatch on every queue acquires at system scope, as after a host wait. */
void a_host_wrote(void) { G().host_epoch.fetch_add(1, std::memory_order_acq_rel); }

/* ---- recorded passes: the calls. See "recorded passes" above. */

inline Tape* tape_of(RadTape t) {
    Tape* p = static_cast<Tape*>(t);
    return (p && p->magic == kTapeMagic) ? p : nullptr;
}

/* Destroyed tapes whose plays may still be queued. */
std::mutex         g_tape_mu;
std::vector<Tape*> g_tape_retired;

/* Whether every packet of the tape's latest play has completed (or its queue is gone): a packet
 * kReserveMargin past it has launched, which proves it as in reserve(). */
bool tape_idle(const Tape* t) {
    for (const auto& l : t->last) {
        hsa_queue_t* q = g_queue_by_id[l.first].load(std::memory_order_acquire);
        if (q && hsa_queue_load_read_index_scacquire(q) < l.second + kReserveMargin) return false;
    }
    return true;
}

void tape_free(Tape* t) {
    note_free(t->karg);
    if (t->karg) hsa_amd_memory_pool_free(t->karg);
    t->magic = 0;
    delete t;
}

void tape_reap() {
    std::lock_guard<std::mutex> lk(g_tape_mu);
    for (size_t i = 0; i < g_tape_retired.size();) {
        if (tape_idle(g_tape_retired[i])) {
            tape_free(g_tape_retired[i]);
            g_tape_retired[i] = g_tape_retired.back();
            g_tape_retired.pop_back();
        } else {
            ++i;
        }
    }
}

int a_tape_begin(void) {
    if (t_tape) { device_set_error("rad tape: a capture is already open on this thread"); return RAD_E_STATE; }
    t_tape = new (std::nothrow) Tape();
    t_tape_pause = 0;
    return t_tape ? RAD_OK : RAD_E_NOMEM;
}

int a_tape_mark(void) { return t_tape ? (int)t_tape->rec.size() : RAD_E_STATE; }

void a_tape_pause(int on) {
    if (on) ++t_tape_pause;
    else if (t_tape_pause > 0) --t_tape_pause;
}

/* Moves the argument blocks to the tape's card, into the memory the stream ring uses there: VRAM
 * the host writes through the BAR, or pinned host memory without one. The packets are then final
 * except for the header. */
RadTape a_tape_end(int keep) {
    Tape* t = t_tape;
    t_tape = nullptr;
    t_tape_pause = 0;
    tape_reap();
    if (!t) return nullptr;
    if (!keep) { tape_free(t); return nullptr; }
    if (!t->bad.empty()) {
        device_set_error("the pass cannot be recorded: %s", t->bad.c_str());
        tape_free(t);
        return nullptr;
    }
    int ord = -1;
    for (const TapeRec& r : t->rec) {
        if (ord < 0) ord = r.s->ord;
        else if (r.s->ord != ord) {
            device_set_error("the pass cannot be recorded: it submits to devices %d and %d", ord,
                             r.s->ord);
            tape_free(t);
            return nullptr;
        }
    }
    if (!t->karg_host.empty()) {
        const Dev& d = G().dev[(size_t)ord];
        void* ka = nullptr;
        hsa_amd_memory_pool_t pool = d.host_kargs ? G().host_fine : d.vram;
        hsa_agent_t peer = d.host_kargs ? d.agent : G().cpu;
        if (hsa_amd_memory_pool_allocate(pool, t->karg_host.size(), 0, &ka) != HSA_STATUS_SUCCESS ||
            hsa_amd_agents_allow_access(1, &peer, nullptr, ka) != HSA_STATUS_SUCCESS) {
            if (ka) hsa_amd_memory_pool_free(ka);
            device_set_error("the pass cannot be recorded: %zu bytes of kernel arguments could "
                             "not be placed in %s", t->karg_host.size(),
                             d.host_kargs ? "pinned host memory" : "VRAM");
            tape_free(t);
            return nullptr;
        }
        std::memcpy(ka, t->karg_host.data(), t->karg_host.size());
        _mm_sfence();
        t->karg = static_cast<char*>(ka);
        note_alloc(ka, (int64_t)t->karg_host.size(), -1);
        for (TapeRec& r : t->rec)
            if (r.kind == kTapeDispatch) r.pkt.kernarg_address = t->karg + r.karg_off;
    }
    return t;
}

int a_tape_len(RadTape rt) {
    const Tape* t = tape_of(rt);
    return t ? (int)t->rec.size() : RAD_E_INVAL;
}

int a_tape_record(RadTape rt, int i, int* kind, uint32_t* queue) {
    const Tape* t = tape_of(rt);
    if (!t || i < 0 || i >= (int)t->rec.size()) return RAD_E_INVAL;
    const TapeRec& r = t->rec[(size_t)i];
    if (kind) *kind = r.kind == kTapeDispatch ? 0 : 1;
    if (queue) *queue = r.s->id;
    return RAD_OK;
}

int a_tape_set_overlap(RadTape rt, int i, int on) {
    Tape* t = tape_of(rt);
    if (!t || i < 0 || i >= (int)t->rec.size() || t->rec[(size_t)i].kind != kTapeDispatch)
        return RAD_E_INVAL;
    t->rec[(size_t)i].overlap = on ? 1 : 0;
    return RAD_OK;
}

int a_tape_play(RadTape rt, int from, int to) {
    Tape* t = tape_of(rt);
    if (!t || from < 0 || to > (int)t->rec.size() || from > to) {
        device_set_error("rad tape: play of a null tape or of records [%d, %d)", from, to);
        return RAD_E_INVAL;
    }
    ++t_tape_pause;
    int rc = RAD_OK;
    PlayRun run;
    for (int i = from; i < to && rc >= 0; ++i) {
        const TapeRec& r = t->rec[(size_t)i];
        if (r.kind == kTapeDispatch) {
            /* A record directly after a dispatch on the same queue rides on the dispatch. The
             * kernel packet has the barrier bit, so its completion already means everything before
             * it is done, which is all the record's barrier packet would say; the completion signal
             * is decremented after the release fence, raised here to the record's scope. This
             * saves a packet between two kernels and signals waiters sooner. */
            const TapeRec* nx = i + 1 < to ? &t->rec[(size_t)i + 1] : nullptr;
            Event* ne = (nx && nx->kind == kTapeRecord && nx->s == r.s) ? ev_of(nx->ev) : nullptr;
            Sig* g = nullptr;
            int rel = HSA_FENCE_SCOPE_NONE;
            if (ne && !(g = record_arm(ne, r.s, &rel))) {
                device_set_error("rad tape: no signal could be made for a record");
                rc = RAD_E_DEVICE;
                break;
            }
            uint64_t idx = 0;
            rc = run.put(r, &idx, g ? g->h : hsa_signal_t{0}, rel);
            if (rc < 0) break;
            bool found = false;
            for (auto& l : t->last)
                if (l.first == r.s->id) { l.second = idx; found = true; break; }
            if (!found) t->last.emplace_back(r.s->id, idx);
            if (g) {
                /* Ring before the event names the signal, as a_event_record does, since another
                 * queue may wait on it next. The event lock is taken with no queue held. */
                run.end();
                record_adopt(ne, g);
                ++i;
            }
            continue;
        }
        /* Event calls take the queue lock themselves, and a wait on another queue must not sit
         * behind packets this run has written but not announced. */
        run.end();
        rc = r.kind == kTapeRecord ? a_event_record(r.ev, (RadStream)r.s)
                                   : a_event_wait((RadStream)r.s, r.ev);
    }
    run.end();
    --t_tape_pause;
    return rc;
}

int a_tape_diff(RadTape ra, RadTape rb, char* why, int cap) {
    const Tape* a = tape_of(ra);
    const Tape* b = tape_of(rb);
    auto say = [&](int at, const std::string& m) {
        if (why && cap > 0) std::snprintf(why, (size_t)cap, "record %d: %s", at, m.c_str());
        return at;
    };
    if (!a || !b) return say(0, "a null tape");
    const size_t n = std::min(a->rec.size(), b->rec.size());
    auto kname = [](uint64_t obj) {
        std::lock_guard<std::mutex> nl(g_names_mu);
        auto it = g_names.find(obj);
        return it != g_names.end() ? it->second : fmt("%#" PRIx64, obj);
    };
    for (size_t i = 0; i < n; ++i) {
        const TapeRec& x = a->rec[i];
        const TapeRec& y = b->rec[i];
        if (x.kind != y.kind || x.s != y.s || x.ev != y.ev || x.fence != y.fence)
            return say((int)i, fmt("kind %d on queue %u against kind %d on queue %u", x.kind,
                                   x.s->id, y.kind, y.s->id));
        if (x.kind != kTapeDispatch) continue;
        hsa_kernel_dispatch_packet_t px = x.pkt, py = y.pkt;
        px.kernarg_address = py.kernarg_address = nullptr;
        if (std::memcmp(&px, &py, sizeof px) != 0)
            return say((int)i, fmt("kernel %s against %s, grid %ux%ux%u against %ux%ux%u",
                                   kname(x.pkt.kernel_object).c_str(),
                                   kname(y.pkt.kernel_object).c_str(), x.pkt.grid_size_x,
                                   x.pkt.grid_size_y, x.pkt.grid_size_z, y.pkt.grid_size_x,
                                   y.pkt.grid_size_y, y.pkt.grid_size_z));
        const size_t len = x.karg_len;
        if (len != y.karg_len ||
            std::memcmp(a->karg_host.data() + x.karg_off, b->karg_host.data() + y.karg_off, len) != 0) {
            size_t k = 0;
            while (k < len && k < y.karg_len &&
                   a->karg_host[x.karg_off + k] == b->karg_host[y.karg_off + k]) ++k;
            return say((int)i, fmt("kernel %s: its arguments differ at byte %zu",
                                   kname(x.pkt.kernel_object).c_str(), k));
        }
    }
    if (a->rec.size() != b->rec.size())
        return say((int)n, fmt("%zu records against %zu", a->rec.size(), b->rec.size()));
    return -1;
}

void a_tape_destroy(RadTape rt) {
    Tape* t = tape_of(rt);
    if (!t) return;
    {
        std::lock_guard<std::mutex> lk(g_tape_mu);
        g_tape_retired.push_back(t);
    }
    tape_reap();
}

int a_event_elapsed_ms(RadEvent ra, RadEvent rb, float* out) {
    Event* a = ev_of(ra);
    Event* b = ev_of(rb);
    if (!a || !b || !out) return RAD_E_INVAL;
    if (a == b) { *out = 0.0f; return a->cur ? RAD_OK : RAD_E_INVAL; }
    hsa_amd_profiling_dispatch_time_t ta{}, tb{};
    {
        std::scoped_lock lk(a->mu, b->mu);
        if (!a->cur || !b->cur) return RAD_E_INVAL;
        if (hsa_amd_profiling_get_dispatch_time(a->cur->agent, a->cur->h, &ta) != HSA_STATUS_SUCCESS ||
            hsa_amd_profiling_get_dispatch_time(b->cur->agent, b->cur->h, &tb) != HSA_STATUS_SUCCESS) {
            device_set_error("rad_event_elapsed_ms: no timestamps on the event packets");
            return RAD_E_DEVICE;
        }
    }
    *out = (float)((double)((int64_t)tb.end - (int64_t)ta.end) * 1e3 / (double)G().ts_freq);
    return RAD_OK;
}

/* ---- copies and fills */

int blit_copy(Stream* s, void* dst, const void* src, int64_t bytes, uint8_t fence) {
    set_next_fence(fence);
    const bool link = (fence & (kFenceAcquireSystem | kFenceReleaseSystem)) != 0;
    const int rc = rad_aql_blit_copy(dst, src, (uint64_t)bytes, s, link);
    set_next_fence(0);
    return rc < 0 ? RAD_E_DEVICE : RAD_OK;
}

/* Pageable memory is bounced through pinned memory and waited for, one bounce buffer at a time,
 * as HIP does. The bytes are counted (aql_pageable_bytes) because a pageable copy on the step
 * path is a host wait mid-step. */
int pageable_copy(Stream* s, void* dst, Loc ld, const void* src, Loc ls, int64_t bytes) {
    g_pageable_bytes.fetch_add(bytes, std::memory_order_relaxed);
    if (capturing() && t_tape->bad.empty())
        t_tape->bad = fmt("a copy of %lld bytes to or from pageable memory, which stages through "
                          "the host and waits", (long long)bytes);
    std::lock_guard<std::mutex> lk(s->bounce_mu);
    for (int64_t off = 0; off < bytes; off += (int64_t)kBounceBytes) {
        const int64_t n = std::min<int64_t>((int64_t)kBounceBytes, bytes - off);
        if (ls == kPageable && ld == kPageable) {
            std::memcpy((char*)dst + off, (const char*)src + off, (size_t)n);
        } else if (ls == kPageable) {
            /* The previous chunk's copy out of the bounce buffer must finish before the host
             * overwrites it. */
            RAD_TRY(stream_drain(s));
            std::memcpy(s->bounce, (const char*)src + off, (size_t)n);
            RAD_TRY(blit_copy(s, (char*)dst + off, s->bounce, n,
                              kFenceAcquireSystem | (ld == kDevLocal ? 0 : kFenceReleaseSystem)));
        } else {
            RAD_TRY(blit_copy(s, s->bounce, (const char*)src + off, n,
                              kFenceReleaseSystem | (ls == kDevLocal ? 0 : kFenceAcquireSystem)));
            RAD_TRY(stream_drain(s));
            std::memcpy((char*)dst + off, s->bounce, (size_t)n);
        }
    }
    if (ls == kPageable) RAD_TRY(stream_drain(s));
    return RAD_OK;
}

int a_memcpy_async(void* dst, const void* src, int64_t bytes, RadStream rs) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_memcpy_async: not a stream of this backend"); return RAD_E_INVAL; }
    if (bytes == 0) return RAD_OK;
    if (bytes < 0) return RAD_E_INVAL;
    const void* gd = nullptr;
    const void* gs = nullptr;
    const Loc ld = locate(dst, s, &gd);
    const Loc ls = locate(src, s, &gs);
    if (ld == kPageable || ls == kPageable)
        return pageable_copy(s, ld == kPageable ? dst : (void*)gd, ld,
                             ls == kPageable ? src : gs, ls, bytes);
    const uint8_t fence = (ls != kDevLocal ? kFenceAcquireSystem : 0) |
                          (ld != kDevLocal ? kFenceReleaseSystem : 0);
    return blit_copy(s, (void*)gd, gs, bytes, fence);
}

int a_memcpy_2d_async(void* dst, int64_t dpitch, const void* src, int64_t spitch, int64_t width,
                      int64_t height, RadStream rs) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_memcpy_2d_async: not a stream of this backend"); return RAD_E_INVAL; }
    if (width == 0 || height == 0) return RAD_OK;
    if (width < 0 || height < 0 || dpitch < width || spitch < width) return RAD_E_INVAL;
    const void* gd = nullptr;
    const void* gs = nullptr;
    const Loc ld = locate(dst, s, &gd);
    const Loc ls = locate(src, s, &gs);
    if (ld == kPageable || ls == kPageable) {
        /* Row by row through the 1-D path; no step-path caller copies pitched pageable memory. */
        for (int64_t r = 0; r < height; ++r)
            RAD_TRY(a_memcpy_async((char*)dst + r * dpitch, (const char*)src + r * spitch, width, rs));
        return RAD_OK;
    }
    const uint8_t fence = (ls != kDevLocal ? kFenceAcquireSystem : 0) |
                          (ld != kDevLocal ? kFenceReleaseSystem : 0);
    set_next_fence(fence);
    const int rc = rad_aql_blit_copy2d((void*)gd, (uint64_t)dpitch, gs, (uint64_t)spitch,
                                       (uint64_t)width, (uint64_t)height, s);
    set_next_fence(0);
    return rc < 0 ? RAD_E_DEVICE : RAD_OK;
}

/* No pageable form: the kernel reads the range list and the source in place, so both must be
 * card-reachable; bouncing them would put a host wait inside the copy. */
int a_memcpy_ranges_async(void* dst, const void* src, int64_t prefix, const RadCopyRange* ranges,
                          int n, RadStream rs) {
    Stream* s = ours(rs);
    if (!s) {
        device_set_error("rad_memcpy_ranges_async: not a stream of this backend");
        return RAD_E_INVAL;
    }
    if (prefix < 0 || (prefix & 15) || n < 0 || (n > 0 && !ranges)) {
        device_set_error("rad_memcpy_ranges_async: prefix %lld (a multiple of 16), %d ranges",
                         (long long)prefix, n);
        return RAD_E_INVAL;
    }
    if (prefix == 0 && n == 0) return RAD_OK;
    const void* gd = nullptr;
    const void* gs = nullptr;
    const void* gr = ranges;
    const Loc ld = locate(dst, s, &gd);
    const Loc ls = locate(src, s, &gs);
    const Loc lr = n > 0 ? locate(ranges, s, &gr) : kDevLocal;
    if (ld == kPageable || ls == kPageable || lr == kPageable ||
        (((uintptr_t)gd | (uintptr_t)gs) & 15)) {
        device_set_error("rad_memcpy_ranges_async(%p <- %p, ranges %p): every address must be one "
                         "the card reaches, and both bases 16-byte aligned", dst, src,
                         (const void*)ranges);
        return RAD_E_UNSUPPORTED;
    }
    set_next_fence((ls != kDevLocal || lr != kDevLocal ? kFenceAcquireSystem : 0) |
                   (ld != kDevLocal ? kFenceReleaseSystem : 0));
    const int rc = rad_aql_blit_ranges((void*)gd, gs, (uint64_t)prefix, (const RadCopyRange*)gr,
                                       n, s);
    set_next_fence(0);
    return rc < 0 ? RAD_E_DEVICE : RAD_OK;
}

/* ---- the DMA lane: a card's own SDMA queue ------------------------------------------------------
 *
 * Copy lists go to the card's DMA engine through this backend's own queue, not the runtime's copy
 * call. On gfx1201, 1.3 MB host->card copies beside a VRAM-saturating kernel stream: this queue
 * moves 28.7 GB/s (the link) at no cost to the kernels; the runtime's copy moves 22.0 GB/s and
 * slows them 8%; copy kernels get 2.3 GB/s. From gfx10 on the runtime brackets every copy with a
 * write-back and invalidation of the whole L2, two flushes per copy under the running GEMMs. Here
 * a list is one wait, its copies and one write; the two barriers ordering the list on its AQL
 * stream handle the caches.
 *
 * The queue is the driver's user-mode SDMA queue (AMDKFD_IOC_CREATE_QUEUE), whose buffers must
 * meet rules the runtime's allocations do not: the write pointer must start a one-page GTT buffer
 * of this card (the scheduler firmware pins it into the GART), and the ring must be a buffer of
 * exactly its own size. Both are allocated here with the thunk's driver calls, at an address
 * reserved in this process so CPU and card share it, and CPU-mapped through the render node the
 * thunk bound to the process's GPU address space (the only file the driver lets map such a
 * buffer).
 *
 * Engine 0 only: a second engine in the same direction adds nothing (the link is the bound) and
 * slowed a VRAM-bound kernel stream 40%. */
constexpr uint32_t kSdmaCopy     = 1;            /* COPY, sub-op LINEAR */
constexpr uint32_t kSdmaFence    = 5;
constexpr uint32_t kSdmaPoll     = 8;            /* POLL_REGMEM */
constexpr uint64_t kSdmaMaxCopy  = 0x3fffe0;     /* the largest count a linear copy carries */
constexpr uint64_t kDmaRingBytes = (uint64_t)1 << 20;
constexpr size_t   kDoorbellPage = 8192;         /* 1024 64-bit doorbells a process */
/* The link gate is a signal that only counts down: each hold and each release is a barrier packet
 * on the holding stream whose completion decrements it, so the value is odd exactly while held,
 * and the engine polls its low bit. Starts even and far from zero. */
constexpr int64_t  kGateStart    = (int64_t)1 << 62;
/* A yielding list checks the gate every 1 MiB. A check costs the engine ~2 us wherever the word
 * lives, 5% of its rate, and a hold starting mid-list shares the link for 36 us at most. 2 MiB
 * made prefill slower; 256 KiB made the checks cost 17%. */
constexpr uint64_t kYieldBytes   = (uint64_t)1 << 20;

/* The create-queue request in the current driver layout, not the build headers'. Selecting the
 * SDMA engine (queue type 4, sdma_engine_id) needs KFD ioctl 1.17, but build headers can be older
 * (Ubuntu 24.04's <linux/kfd_ioctl.h> is 1.14, which lacks both fields). The driver copies as many
 * bytes as the ioctl number encodes and zero-fills or drops the rest, so every driver gets this
 * form; one older than 1.17 refuses the queue type and the lane reports that. */
struct KfdCreateQueue {
    uint64_t ring_base_address, write_pointer_address, read_pointer_address, doorbell_offset;
    uint32_t ring_size, gpu_id, queue_type, queue_percentage, queue_priority, queue_id;
    uint64_t eop_buffer_address, eop_buffer_size, ctx_save_restore_address;
    uint32_t ctx_save_restore_size, ctl_stack_size, sdma_engine_id, metadata_ring_size;
};
static_assert(sizeof(KfdCreateQueue) == 96, "the KFD 1.22 create-queue layout");
constexpr uint32_t kKfdQueueSdmaByEngine = 0x4;
constexpr unsigned long kKfdCreateQueue = _IOWR(AMDKFD_IOCTL_BASE, 0x02, KfdCreateQueue);

struct DmaBo {
    char*    va = nullptr;
    size_t   size = 0;
    uint64_t handle = 0;
    char*    res = nullptr;                      /* the address reservation it sits in */
    size_t   res_size = 0;
};

struct DmaQueue {
    std::once_flag      once;
    std::mutex          mu;
    bool                ok = false;
    DmaBo               ring, ptrs;
    uint32_t*           rb = nullptr;
    volatile uint64_t*  rptr = nullptr;          /* bytes the engine has read, written by the engine */
    volatile uint64_t*  wptr = nullptr;          /* bytes written, for the firmware */
    volatile uint64_t*  db = nullptr;
    uint64_t            w = 0;                   /* bytes written, monotonic */
    uint32_t            qid = 0;
    std::once_flag      gate_once;
    hsa_signal_t        gate{};                  /* the link gate: odd while a stream holds the link */
    bool                held = false;            /* the last hold issued, for the alternation check */
    bool                ever_held = false;       /* a stream has held this card's link at least once */
    std::atomic<bool>   tried{false};            /* dma_open has run, for the startup report */
    std::string         why;                     /* why it failed */
    /* For a queue whose lists each end by writing their number to `fin` (the expert prefetch's),
     * the room check uses the end of the last finished list as a lower bound on what the engine
     * has read. The engine's written-back read pointer can stop moving on such a queue while the
     * engine runs on, and a ring the host thinks full would wait forever. */
    volatile uint32_t*  fin = nullptr;
    uint32_t            lists = 0;               /* lists handed over; list k ends at ends[k % kDmaEnds] */
    std::unique_ptr<uint64_t[]> ends;
};
constexpr uint32_t kDmaEnds = 4096;
DmaQueue g_dma[kMaxDevices];
DmaQueue g_dma_xp[kMaxDevices];   /* the expert prefetch's, on the card's second engine */

/* Minor of the render node (/dev/dri/renderD<minor>) a card's buffers are CPU-mapped through, from
 * the driver's topology entry for its gpu_id. */
int render_minor(uint32_t gpu_id) {
    const int n = topo_node(gpu_id);
    if (n < 0) return -1;
    char path[96];
    std::snprintf(path, sizeof path, "/sys/class/kfd/kfd/topology/nodes/%d/properties", n);
    FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    char key[128];
    unsigned long long v = 0;
    int minor = -1;
    while (std::fscanf(f, "%127s %llu", key, &v) == 2)
        if (!std::strcmp(key, "drm_render_minor")) minor = (int)v;
    std::fclose(f);
    return minor;
}

void dma_bo_free(uint32_t gpu_id, DmaBo* b) {
    if (b->handle) {
        uint32_t ids[1] = { gpu_id };
        kfd_ioctl_unmap_memory_from_gpu_args u{};
        u.handle = b->handle;
        u.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
        u.n_devices = 1;
        ioctl(G().kfd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &u);
        kfd_ioctl_free_memory_of_gpu_args f{};
        f.handle = b->handle;
        ioctl(G().kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &f);
    }
    if (b->res) munmap(b->res, b->res_size);
    *b = DmaBo{};
}

/* A GTT buffer of card `gpu_id`, at one address for the CPU and the card. */
bool dma_bo(uint32_t gpu_id, int minor, size_t size, DmaBo* b, std::string* why) {
    const size_t al = (size_t)2 << 20;
    b->res_size = size + al;
    void* r = mmap(nullptr, b->res_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) {
        b->res = nullptr;
        *why = fmt("no address range for a %zu-byte buffer: %s", size, std::strerror(errno));
        return false;
    }
    b->res = (char*)r;
    b->va = (char*)(((uintptr_t)r + al - 1) & ~(uintptr_t)(al - 1));
    b->size = size;
    kfd_ioctl_alloc_memory_of_gpu_args a{};
    a.va_addr = (uint64_t)(uintptr_t)b->va;
    a.size = size;
    a.gpu_id = gpu_id;
    a.flags = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
              KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
              KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
    if (ioctl(G().kfd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a) < 0) {
        *why = fmt("the driver refused a %zu-byte GTT buffer: %s", size, std::strerror(errno));
        dma_bo_free(gpu_id, b);
        return false;
    }
    b->handle = a.handle;
    uint32_t ids[1] = { gpu_id };
    kfd_ioctl_map_memory_to_gpu_args m{};
    m.handle = a.handle;
    m.device_ids_array_ptr = (uint64_t)(uintptr_t)ids;
    m.n_devices = 1;
    if (ioctl(G().kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &m) < 0) {
        *why = fmt("the driver would not map a GTT buffer to the card: %s", std::strerror(errno));
        dma_bo_free(gpu_id, b);
        return false;
    }
    char want[64];
    std::snprintf(want, sizeof want, "/dev/dri/renderD%d", minor);
    if (DIR* d = opendir("/proc/self/fd")) {
        while (dirent* e = readdir(d)) {
            char link[300], tgt[128];
            std::snprintf(link, sizeof link, "/proc/self/fd/%s", e->d_name);
            const ssize_t n = readlink(link, tgt, sizeof tgt - 1);
            if (n <= 0) continue;
            tgt[n] = 0;
            if (std::strcmp(tgt, want)) continue;
            const int fd = std::atoi(e->d_name);
            if (mmap(b->va, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, a.mmap_offset) !=
                MAP_FAILED) {
                closedir(d);
                return true;
            }
        }
        closedir(d);
    }
    *why = fmt("no file of this process maps a buffer of the card's address space (%s, the runtime's "
               "render node, is not open or refused)", want);
    dma_bo_free(gpu_id, b);
    return false;
}

/* The gate is created separately from the queue, on first use by either: a launch receives its
 * address when its op is bound, which must not open a DMA queue on a card nothing streams to. */
hsa_signal_t* gate_signal(int ord) {
    if (ord < 0 || ord >= kMaxDevices) return nullptr;
    DmaQueue& q = g_dma[ord];
    std::call_once(q.gate_once, [&] {
        if (hsa_amd_signal_create(kGateStart, 0, nullptr, HSA_AMD_SIGNAL_AMD_GPU_ONLY, &q.gate) !=
            HSA_STATUS_SUCCESS)
            q.gate.handle = 0;
    });
    return q.gate.handle ? &q.gate : nullptr;
}

/* The card's queue if it has been opened and is usable; never opens it. */
DmaQueue* dma_queue_open(int ord) {
    if (ord < 0 || ord >= kMaxDevices) return nullptr;
    DmaQueue& q = g_dma[ord];
    return q.tried.load(std::memory_order_acquire) && q.ok ? &q : nullptr;
}

bool dma_open(int ord, DmaQueue* q, std::string* why, uint32_t engine = 0, bool gated = true) {
    const uint32_t gpu_id = G().dev[(size_t)ord].kfd_id;
    const int minor = gpu_id ? render_minor(gpu_id) : -1;
    if (G().kfd < 0 || !gpu_id || minor < 0) {
        *why = "the driver's topology does not name the card's render node";
        return false;
    }
    if (!dma_bo(gpu_id, minor, kDmaRingBytes, &q->ring, why)) return false;
    if (!dma_bo(gpu_id, minor, 4096, &q->ptrs, why)) { dma_bo_free(gpu_id, &q->ring); return false; }
    std::memset(q->ring.va, 0, kDmaRingBytes);
    std::memset(q->ptrs.va, 0, 4096);
    q->rb = (uint32_t*)q->ring.va;
    q->rptr = (volatile uint64_t*)q->ptrs.va;
    q->wptr = (volatile uint64_t*)(q->ptrs.va + 64);
    KfdCreateQueue a{};
    a.ring_base_address = (uint64_t)(uintptr_t)q->ring.va;
    a.ring_size = (uint32_t)kDmaRingBytes;
    a.read_pointer_address = (uint64_t)(uintptr_t)q->rptr;
    a.write_pointer_address = (uint64_t)(uintptr_t)q->wptr;
    a.gpu_id = gpu_id;
    a.queue_type = kKfdQueueSdmaByEngine;
    a.sdma_engine_id = engine;
    a.queue_percentage = KFD_MAX_QUEUE_PERCENTAGE;
    a.queue_priority = KFD_MAX_QUEUE_PRIORITY;
    if (ioctl(G().kfd, kKfdCreateQueue, &a) < 0) {
        *why = fmt("the driver refused an SDMA queue: %s", std::strerror(errno));
        dma_bo_free(gpu_id, &q->ptrs);
        dma_bo_free(gpu_id, &q->ring);
        return false;
    }
    q->qid = a.queue_id;
    /* The offset names the process's doorbell page and, in its low bits, this queue's doorbell in it. */
    void* m = mmap(nullptr, kDoorbellPage, PROT_READ | PROT_WRITE, MAP_SHARED, G().kfd,
                   (off_t)(a.doorbell_offset & ~(uint64_t)(kDoorbellPage - 1)));
    if (m == MAP_FAILED) {
        *why = fmt("the SDMA queue's doorbell could not be mapped: %s", std::strerror(errno));
        kfd_ioctl_destroy_queue_args d{};
        d.queue_id = q->qid;
        ioctl(G().kfd, AMDKFD_IOC_DESTROY_QUEUE, &d);
        dma_bo_free(gpu_id, &q->ptrs);
        dma_bo_free(gpu_id, &q->ring);
        return false;
    }
    q->db = (volatile uint64_t*)((char*)m + (a.doorbell_offset & (kDoorbellPage - 1)));
    q->w = 0;
    if (!gated) return true;
    if (!gate_signal(ord)) {
        /* The queue stays open and unused: a list without a gate would not know when to yield. */
        *why = "no signal could be made for the link gate";
        return false;
    }
    return true;
}

inline uint64_t dma_read(const DmaQueue* q);

/* The DMA lanes' part of dump_queues. A list stops at the gate while held, so a gate left odd
 * shows as a ring written past what was read. Read without the lane lock, which a stuck writer
 * may hold. */
void dump_dma_lanes() {
    for (int c = 0; c < kMaxDevices; ++c) {
        DmaQueue& q = g_dma[c];
        if (!q.tried.load(std::memory_order_acquire) || !q.ok) continue;
        RAD_ERR("  device %d DMA lane: ring written %" PRIu64 " B, read %" PRIu64 " B; link gate %"
                PRId64 " (%s), last hold issued %s", c, (uint64_t)q.w, (uint64_t)*q.rptr,
                (int64_t)hsa_signal_load_relaxed(q.gate),
                (hsa_signal_load_relaxed(q.gate) & 1) ? "held" : "free", q.held ? "on" : "off");
    }
    for (int c = 0; c < kMaxDevices; ++c) {
        DmaQueue& q = g_dma_xp[c];
        if (!q.tried.load(std::memory_order_acquire) || !q.ok) continue;
        RAD_ERR("  device %d prefetch DMA queue: ring written %" PRIu64 " B, read %" PRIu64 " B; %u lists "
                "handed over, %u finished", c, (uint64_t)q.w, dma_read(&q), q.lists, q.fin ? *q.fin : 0u);
    }
}

/* The card's queue, opened on first use; null (after one warning) if the driver refuses. */
DmaQueue* dma_queue(int ord) {
    if (ord < 0 || ord >= kMaxDevices || ord >= (int)G().dev.size()) return nullptr;
    DmaQueue& q = g_dma[ord];
    std::call_once(q.once, [&] {
        std::string why;
        const auto t0 = std::chrono::steady_clock::now();
        q.ok = dma_open(ord, &q, &why);
        q.why = why;
        q.tried.store(true, std::memory_order_release);
        if (q.ok) aql::g_dump_lanes = dump_dma_lanes;
        if (q.ok)
            RAD_INFO("device: card %d copy lists go to DMA engine 0 on a queue of this process's own "
                     "(%.1f ms to open)", ord,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (!q.ok)
            RAD_WARN("device: card %d has no DMA queue of this process's own (%s); copy lists go as "
                     "copy kernels, which share the compute units with the running kernels and slow "
                     "the prefill stager down", ord, why.c_str());
    });
    return q.ok ? &q : nullptr;
}

/* Bytes the engine has read: its written-back read pointer, or the end of the last finished list
 * on a numbered queue if that is further (DmaQueue::fin). */
inline uint64_t dma_read(const DmaQueue* q) {
    uint64_t r = *q->rptr;
    if (q->fin) {
        const uint32_t k = *q->fin;
        if (k) r = std::max(r, q->ends[k % kDmaEnds]);
    }
    return r;
}

/* Writes `n` dwords to the ring; queue lock held. A packet never straddles the ring end: the tail
 * is padded with single-dword NOPs. The ring holds a few hundred layers' lists, so waiting for
 * room means the host is that far ahead of a stalled engine. */
int dma_put(DmaQueue* q, const uint32_t* dw, uint64_t n) {
    const uint64_t cap = kDmaRingBytes / 4;
    uint64_t at = (q->w / 4) & (cap - 1);
    const uint64_t pad = at + n > cap ? cap - at : 0;
    const uint64_t need = (pad + n) * 4;
    if (q->w + need - dma_read(q) > kDmaRingBytes) {
        Stall st("a full DMA ring");
        while (q->w + need - dma_read(q) > kDmaRingBytes) {
            if (G().faulted.load(std::memory_order_relaxed)) {
                device_set_error("rad_memcpy_list_async: a queue faulted while the DMA ring was full");
                return RAD_E_DEVICE;
            }
            st.tick();
            _mm_pause();
        }
    }
    for (uint64_t i = 0; i < pad; ++i) q->rb[at + i] = 0;
    q->w += pad * 4;
    at = (q->w / 4) & (cap - 1);
    for (uint64_t i = 0; i < n; ++i) q->rb[at + i] = dw[i];
    q->w += n * 4;
    return RAD_OK;
}

void dma_kick(DmaQueue* q) {
    _mm_sfence();
    *q->wptr = q->w;
    _mm_sfence();
    *q->db = q->w;
}

/* Where the engine reads and writes a signal's value: the 64-bit word the CP decrements and polls. */
inline uint64_t sig_value_addr(const Sig* g) {
    return (uint64_t)g->h.handle + offsetof(amd_signal_t, value);
}

/* ------------------------------------------------- the expert prefetch's device half (rad_xp_*) */

/* The card's prefetch queue on its second DMA engine (the lane has the first), opened on first
 * use; null if the driver refuses. Each list ends by writing its number to a signal's value,
 * which the room check and rad_xp_behind read. */
DmaQueue* xp_queue(int ord) {
    if (ord < 0 || ord >= kMaxDevices || ord >= (int)G().dev.size()) return nullptr;
    DmaQueue& q = g_dma_xp[ord];
    std::call_once(q.once, [&] {
        std::string why;
        Sig* g = sig_get(false);
        if (g) {
            hsa_signal_silent_store_relaxed(g->h, 0);
            q.ends.reset(new uint64_t[kDmaEnds]());
            q.ok = dma_open(ord, &q, &why, 1, false);
            if (q.ok) q.fin = (volatile uint32_t*)(uintptr_t)sig_value_addr(g);
        } else {
            why = "no signal could be made for its lists' numbers";
        }
        q.why = why;
        q.tried.store(true, std::memory_order_release);
        if (!q.ok)
            RAD_INFO("device: card %d has no DMA queue on the second engine for the expert "
                     "prefetch (%s)", ord, why.c_str());
    });
    return q.ok ? &q : nullptr;
}

int a_xp_supported(int device) { return xp_queue(device) != nullptr; }

int a_xp_stream_create(RadStream* out) {
    const int rc = a_stream_create(out, 0);
    if (rc < 0) return rc;
    ours(*out)->agent_only = true;
    return RAD_OK;
}

/* Refused while another ride on the stream is still waiting for its dispatch. */
int a_ride(RadStream rs, void** token) {
    Stream* s = ours(rs);
    if (!s || !token) return RAD_E_INVAL;
    *token = nullptr;
    Sig* g = sig_get(false);
    if (!g) return RAD_E_DEVICE;
    hsa_signal_silent_store_relaxed(g->h, 1);
    s->lock();
    const bool busy = s->ride.handle != 0;
    if (!busy) s->ride = g->h;
    s->unlock();
    if (busy) {
        hsa_signal_silent_store_relaxed(g->h, 0);
        sig_retire(g);
        return RAD_E_STATE;
    }
    *token = g;
    return RAD_OK;
}

int a_ride_wait(RadStream rs, void* token) {
    Stream* s = ours(rs);
    Sig* g = (Sig*)token;
    if (!s || !g) return RAD_E_INVAL;
    uint64_t idx = 0;
    const int rc = barrier(s, g->h, hsa_signal_t{0}, HSA_FENCE_SCOPE_AGENT, HSA_FENCE_SCOPE_NONE, &idx);
    if (rc == RAD_OK) g->waiters.emplace_back(s->id, idx);
    sig_retire(g);
    return rc;
}

void a_ride_release(void* token, RadStream after, int64_t packets) {
    Sig* g = (Sig*)token;
    if (!g) return;
    if (Stream* s = ours(after)) {
        s->lock();
        g->waiters.emplace_back(s->id, s->wr + (uint64_t)std::max<int64_t>(packets, 0));
        s->unlock();
    }
    sig_retire(g);
}

/* ---- gates (rad_stream_gate): a barrier on a signal only the host releases.
 *
 * The barrier acquires at system scope, which also takes the acquire the first dispatch after a
 * host wait would otherwise take (acquire_scope), so the packets behind it run at agent scope. The
 * signal is an interrupt signal from the event pool, the kind host-wait events record: created by
 * hsa_signal_create in host memory, where the host's store reaches the packet processor's poll as
 * any host write to fine-grained memory does. It returns to the pool once the barrier is
 * consumed. */
constexpr uint64_t kGateMagic = 0x5241442d41514c47ull;   /* "RAD-AQLG" */

struct Gate {
    uint64_t magic = kGateMagic;
    Stream*  s = nullptr;
    Sig*     g = nullptr;
};

int a_stream_gate(RadStream rs, void** out) {
    Stream* s = ours(rs);
    if (!s || !out) { device_set_error("rad_stream_gate: null or foreign argument"); return RAD_E_INVAL; }
    if (capturing()) {
        device_set_error("rad_stream_gate: a tape cannot record a gate");
        return RAD_E_STATE;
    }
    Sig* g = sig_get(true);
    if (!g) { device_set_error("rad_stream_gate: no signal could be made"); return RAD_E_DEVICE; }
    hsa_signal_silent_store_relaxed(g->h, 1);
    uint64_t idx = 0;
    const int rc = barrier(s, g->h, hsa_signal_t{0}, HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_NONE,
                           &idx);
    if (rc < 0) {
        hsa_signal_silent_store_relaxed(g->h, 0);
        sig_retire(g);
        return rc;
    }
    g->waiters.emplace_back(s->id, idx);
    s->gates.fetch_add(1, std::memory_order_acq_rel);
    Gate* gt = new Gate();
    gt->s = s;
    gt->g = g;
    *out = gt;
    return RAD_OK;
}

int a_gate_open(void* p) {
    Gate* gt = static_cast<Gate*>(p);
    if (!gt || gt->magic != kGateMagic) return RAD_E_INVAL;
    hsa_signal_store_screlease(gt->g->h, 0);
    gt->s->gates.fetch_sub(1, std::memory_order_acq_rel);
    sig_retire(gt->g);
    gt->magic = 0;
    delete gt;
    return RAD_OK;
}

/* The packets that fit before reserve() would wait for the read index, and before a dispatch would
 * take the half-lap system-scope acquire (acquire_scope). Barriers count against both. */
int64_t a_gate_room(RadStream rs) {
    Stream* s = ours(rs);
    if (!s) return 0;
    s->lock();
    const uint64_t rd = hsa_queue_load_read_index_scacquire(s->q);
    const int64_t cap = (int64_t)(rd + s->mask + 1) - (int64_t)kReserveMargin - (int64_t)s->wr;
    const int64_t lap = s->last_sys_acq + (int64_t)(kQueuePackets / 2) - (int64_t)s->wr;
    s->unlock();
    return std::max<int64_t>(0, std::min(cap, lap));
}

/* Waits for the mark, then for the card's link gate if a collective has ever held it (under TP
 * the all-reduce shares the link and the step waits on it, not on these copies), copies, writes
 * `done`, then the list's number. */
int a_xp_list(int ord, void* mark, int n, const uint64_t* src, const uint64_t* dst, const uint64_t* bytes,
              uint32_t* done, uint32_t done_val) {
    DmaQueue* dq = xp_queue(ord);
    if (!dq) return RAD_E_UNSUPPORTED;
    Sig* in = (Sig*)mark;
    DmaQueue* lane = dma_queue_open(ord);
    std::lock_guard<std::mutex> lk(dq->mu);
    int rc = RAD_OK;
    if (in) {
        const uint64_t ia = sig_value_addr(in);
        const uint32_t poll[6] = { kSdmaPoll | (3u << 28) /* equal */ | (1u << 31) /* memory */,
                                   (uint32_t)ia, (uint32_t)(ia >> 32), 0u, 0xffffffffu,
                                   10u /* interval */ | (0xfffu << 16) /* retry forever */ };
        rc = dma_put(dq, poll, 6);
    }
    if (rc == RAD_OK && n > 0 && lane && lane->ever_held && lane->gate.handle) {
        const uint64_t ga = (uint64_t)lane->gate.handle + offsetof(amd_signal_t, value);
        const uint32_t gate[6] = { kSdmaPoll | (3u << 28) /* equal */ | (1u << 31) /* memory */,
                                   (uint32_t)ga, (uint32_t)(ga >> 32), 0u, 1u /* the low bit */,
                                   10u /* interval */ | (0xfffu << 16) /* retry forever */ };
        rc = dma_put(dq, gate, 6);
    }
    for (int i = 0; rc == RAD_OK && i < n; ++i)
        for (uint64_t o = 0; rc == RAD_OK && o < bytes[i]; o += kSdmaMaxCopy) {
            const uint64_t len = std::min(kSdmaMaxCopy, bytes[i] - o);
            const uint64_t sa = src[i] + o, da = dst[i] + o;
            const uint32_t cp[7] = { kSdmaCopy, (uint32_t)(len - 1), 0u, (uint32_t)sa, (uint32_t)(sa >> 32),
                                     (uint32_t)da, (uint32_t)(da >> 32) };
            rc = dma_put(dq, cp, 7);
        }
    if (rc == RAD_OK && done) {
        const uint64_t a = (uint64_t)(uintptr_t)done;
        const uint32_t fence[4] = { kSdmaFence | (3u << 16) /* uncached */, (uint32_t)a, (uint32_t)(a >> 32),
                                    done_val };
        rc = dma_put(dq, fence, 4);
    }
    if (rc == RAD_OK) {
        const uint32_t k = dq->lists + 1;
        const uint64_t a = (uint64_t)(uintptr_t)dq->fin;
        const uint32_t fence[4] = { kSdmaFence | (3u << 16) /* uncached */, (uint32_t)a, (uint32_t)(a >> 32), k };
        rc = dma_put(dq, fence, 4);
        if (rc == RAD_OK) {
            dq->ends[k % kDmaEnds] = dq->w;
            dq->lists = k;
        }
    }
    dma_kick(dq);
    return rc;
}

int64_t a_xp_behind(int ord) {
    DmaQueue* dq = (ord >= 0 && ord < kMaxDevices) ? &g_dma_xp[ord] : nullptr;
    if (!dq || !dq->tried.load(std::memory_order_acquire) || !dq->ok) return 0;
    std::lock_guard<std::mutex> lk(dq->mu);
    return (int64_t)(uint32_t)(dq->lists - *dq->fin);
}

bool card_has_room(int ord, int64_t bytes, const char* what);
void* a_alloc_uncached(int64_t bytes) {
    if (bytes <= 0) return nullptr;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess || !card_has_room(dev, bytes, "rad_dev_alloc_uncached")) return nullptr;
    void* p = nullptr;
    if (hipExtMallocWithFlags(&p, (size_t)bytes, hipDeviceMallocUncached) != hipSuccess) return nullptr;
    return p;
}

void a_free_uncached(void* p) {
    if (p) (void)hipFree(p);
}

/* The engine waits for the stream, copies, and the stream waits for the engine (the two barriers
 * in `flush`). Copies the engine cannot take go as kernels, in list order: pageable memory
 * (bounced), a peer card's memory, and any copy while recording (a tape replays queue packets, and
 * a DMA submission is not one). Both signals retire behind the second barrier: once the engine
 * has written `out` it has stopped reading `in`. */
int a_memcpy_list_async(const RadCopySeg* segs, int n, unsigned flags, RadStream rs) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_memcpy_list_async: not a stream of this backend"); return RAD_E_INVAL; }
    if (n < 0 || (n > 0 && !segs)) {
        device_set_error("rad_memcpy_list_async: %d segments at %p", n, (const void*)segs);
        return RAD_E_INVAL;
    }
    if (n == 0) {   /* readies the card's queue; see device.h */
        (void)dma_queue(s->ord);
        return RAD_OK;
    }
    struct Dma { const RadCopySeg* seg; uint64_t dst, src; };
    std::vector<Dma> dma;
    dma.reserve((size_t)n);
    bool card = false;                       /* a copy reads or writes this card's memory */
    /* Stream order is kept by two barriers and a signal each way. The first barrier completes
     * `in` once everything before the list is done; the engine polls `in`, copies, and writes
     * `out`; the second barrier holds the stream on `out`. The first releases at system scope
     * when a copy touches this card's memory, so the engine sees kernel writes and no dirty L2
     * line is later written back over a copied destination. The second always acquires at system
     * scope: the engine writes past the L2, which may still hold the destination's old bytes. */
    auto flush = [&]() -> int {
        if (dma.empty()) return RAD_OK;
        DmaQueue* dq = dma_queue(s->ord);
        if (!dq) {
            for (const Dma& c : dma) RAD_TRY(a_memcpy_async(c.seg->dst, c.seg->src, c.seg->bytes, rs));
            dma.clear();
            card = false;
            return RAD_OK;
        }
        Sig* in  = sig_get(false);
        Sig* out = sig_get(false);
        if (!in || !out) {
            if (in) sig_retire(in);
            if (out) sig_retire(out);
            device_set_error("rad_memcpy_list_async: no signal could be made");
            return RAD_E_DEVICE;
        }
        hsa_signal_silent_store_relaxed(in->h, 1);
        hsa_signal_silent_store_relaxed(out->h, 1);
        int rc = barrier(s, hsa_signal_t{0}, in->h, HSA_FENCE_SCOPE_NONE,
                         card ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_NONE);
        if (rc != RAD_OK) {
            /* The queue has faulted and nothing was submitted: `in` will never complete. */
            sig_retire(in);
            sig_retire(out);
            dma.clear();
            return rc;
        }
        {
            std::lock_guard<std::mutex> lk(dq->mu);
            const uint64_t ia = sig_value_addr(in), oa = sig_value_addr(out);
            const uint32_t poll[6] = { kSdmaPoll | (3u << 28) /* equal */ | (1u << 31) /* memory */,
                                       (uint32_t)ia, (uint32_t)(ia >> 32), 0u, 0xffffffffu,
                                       10u /* interval */ | (0xfffu << 16) /* retry forever */ };
            rc = dma_put(dq, poll, 6);
            /* A yielding list checks the gate before its first copy and after every piece (copies
             * are cut to piece size), so a hold starting mid-list shares the link for one piece at
             * most. Skipped on a card whose link was never held (no collective, e.g. one card
             * alone), where the checks would cost 5% of a link-bound prefill. */
            const bool yield = (flags & RAD_COPY_YIELD) != 0 && dq->ever_held;
            const uint64_t piece = yield ? std::min(kYieldBytes, kSdmaMaxCopy) : kSdmaMaxCopy;
            const uint64_t ga = (uint64_t)dq->gate.handle + offsetof(amd_signal_t, value);
            const uint32_t gate[6] = { kSdmaPoll | (3u << 28) /* equal */ | (1u << 31) /* memory */,
                                       (uint32_t)ga, (uint32_t)(ga >> 32), 0u, 1u /* the low bit */,
                                       10u /* interval */ | (0xfffu << 16) /* retry forever */ };
            uint64_t since = piece;
            for (size_t i = 0; rc == RAD_OK && i < dma.size(); ++i) {
                const uint64_t bytes = (uint64_t)dma[i].seg->bytes;
                for (uint64_t o = 0; rc == RAD_OK && o < bytes; o += piece) {
                    const uint64_t len = std::min(piece, bytes - o);
                    const uint64_t sa = dma[i].src + o, da = dma[i].dst + o;
                    if (yield && since >= piece) {
                        rc = dma_put(dq, gate, 6);
                        since = 0;
                        if (rc != RAD_OK) break;
                    }
                    const uint32_t cp[7] = { kSdmaCopy, (uint32_t)(len - 1), 0u,
                                             (uint32_t)sa, (uint32_t)(sa >> 32),
                                             (uint32_t)da, (uint32_t)(da >> 32) };
                    rc = dma_put(dq, cp, 7);
                    since += len;
                }
            }
            if (rc == RAD_OK) {
                /* The low half of `out`'s value; the high half is already zero. */
                const uint32_t fence[4] = { kSdmaFence | (3u << 16) /* uncached */,
                                            (uint32_t)oa, (uint32_t)(oa >> 32), 0u };
                rc = dma_put(dq, fence, 4);
            }
            dma_kick(dq);
        }
        /* If the ring refused the list, release `out` so the stream does not wait on it forever. */
        if (rc != RAD_OK) hsa_signal_store_screlease(out->h, 0);
        uint64_t idx = 0;
        const int br = barrier(s, out->h, hsa_signal_t{0}, HSA_FENCE_SCOPE_SYSTEM,
                               HSA_FENCE_SCOPE_NONE, &idx);
        if (br == RAD_OK) {
            /* `in` is read by the engine, which is done with it once `out` is written. */
            in->waiters.emplace_back(s->id, idx);
            out->waiters.emplace_back(s->id, idx);
        }
        sig_retire(in);
        sig_retire(out);
        dma.clear();
        card = false;
        if (br != RAD_OK) return br;
        return rc;
    };
    for (int i = 0; i < n; ++i) {
        const RadCopySeg& g = segs[i];
        if (g.bytes == 0) continue;
        if (g.bytes < 0 || !g.src || !g.dst) {
            device_set_error("rad_memcpy_list_async: segment %d (%p <- %p, %lld)", i, g.dst, g.src,
                             (long long)g.bytes);
            return RAD_E_INVAL;
        }
        const void* gd = nullptr;
        const void* gs = nullptr;
        const Loc ld = locate(g.dst, s, &gd);
        const Loc ls = locate(g.src, s, &gs);
        /* The list stays on this card; a peer's memory is copied by kernel. */
        if (capturing() || ld == kPageable || ls == kPageable || ld == kDevPeer || ls == kDevPeer) {
            RAD_TRY(flush());
            RAD_TRY(a_memcpy_async(g.dst, g.src, g.bytes, rs));
            continue;
        }
        card = card || ld == kDevLocal || ls == kDevLocal;
        dma.push_back(Dma{&g, (uint64_t)(uintptr_t)gd, (uint64_t)(uintptr_t)gs});
    }
    return flush();
}

/* A hold or release is one barrier on the holding stream that decrements the gate. It takes
 * effect once everything before it on the stream is done: as the following collective starts, or
 * after the preceding one finishes. A recorded pass holds nothing; holds only pace lists, and a
 * play issues none. */
int a_link_hold(RadStream rs, int on) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_link_hold: not a stream of this backend"); return RAD_E_INVAL; }
    if (capturing()) return RAD_OK;
    /* Only the DMA engine polls the gate. */
    DmaQueue* dq = dma_queue_open(s->ord);
    if (!dq) return RAD_OK;
    std::lock_guard<std::mutex> lk(dq->mu);
    if (dq->held == (on == 1)) {
        device_set_error("rad_link_hold: card %d's link is %s held", s->ord,
                         on == 1 ? "already" : "not");
        return RAD_E_STATE;
    }
    /* Update the state only after the barrier is accepted. */
    if (on != 2)
        RAD_TRY(barrier(s, hsa_signal_t{0}, dq->gate, HSA_FENCE_SCOPE_NONE, HSA_FENCE_SCOPE_NONE));
    dq->held = on == 1;
    dq->ever_held = dq->ever_held || on == 1;
    return RAD_OK;
}

int a_dma_lane(int device, char* out, int cap) {
    if (device < 0 || device >= kMaxDevices || device >= (int)G().dev.size() || !out || cap <= 0)
        return RAD_E_INVAL;
    const DmaQueue& q = g_dma[device];
    if (!q.tried.load(std::memory_order_acquire))
        std::snprintf(out, (size_t)cap, "copy lists not used: nothing streams weights onto this card");
    else if (q.ok)
        std::snprintf(out, (size_t)cap, "copy lists on DMA engine 0, a queue of this process's own; "
                                        "they give way to the collectives");
    else
        std::snprintf(out, (size_t)cap, "copy lists as copy kernels: %s", q.why.c_str());
    return RAD_OK;
}

void* a_link_gate(RadStream rs) {
    Stream* s = ours(rs);
    hsa_signal_t* g = s ? gate_signal(s->ord) : nullptr;
    return g ? (void*)(uintptr_t)(g->handle + offsetof(amd_signal_t, value)) : nullptr;
}

int a_memset_async(void* dst, int value, int64_t bytes, RadStream rs) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_memset_async: not a stream of this backend"); return RAD_E_INVAL; }
    if (bytes == 0) return RAD_OK;
    const void* gd = nullptr;
    const Loc ld = locate(dst, s, &gd);
    if (ld == kPageable) {
        device_set_error("rad_memset_async(%p, %lld): the destination is not device-accessible "
                         "-- RAD_MEM_HOST is pageable and the GPU has never seen it. Use "
                         "RAD_MEM_HOST_PINNED or RAD_MEM_HOST_MAPPED, or memset it on the host.",
                         dst, (long long)bytes);
        return RAD_E_UNSUPPORTED;
    }
    set_next_fence(ld != kDevLocal ? kFenceReleaseSystem : 0);
    const int rc = rad_aql_blit_fill((void*)gd, value, (uint64_t)bytes, s);
    set_next_fence(0);
    return rc < 0 ? RAD_E_DEVICE : RAD_OK;
}

/* ---- VRAM: the room check.
 *
 * In ROCm 7.2 a VRAM allocation the driver refuses kills the process once any queue has been
 * destroyed. On a refusal ROCr frees its caches and retries (GpuAgent::Trim), walking every queue
 * created on the card, but it never removes destroyed queues from that list
 * (GpuAgent::aql_queues_, amd_gpu_agent.cpp), so the walk touches freed memory and SIGSEGVs in
 * libhsa-runtime64.so. Startup destroys queues before its large allocations (the two that measure
 * a queue's cost, and the loader's), so a card short on memory would crash in hipMalloc instead of
 * refusing the arena. Until a ROCr release removes the queue on destroy (develop does), every
 * device allocation is checked here first against what the driver will still give, and refused
 * with what the card has and which processes hold it.
 *
 * The figure is the driver's own (AMDKFD_IOC_AVAILABLE_MEMORY), which its allocation check uses;
 * requests are rounded up to 2 MiB so one the driver rounds up does not slip past. It is a check,
 * not a reservation: memory another process takes in between still reaches the runtime. */

/* What the driver will still allocate on card `ord`, -1 when it will not say. */
int64_t card_available(int ord) {
    if (ord < 0 || ord >= (int)G().dev.size() || G().kfd < 0) return -1;
    kfd_ioctl_get_available_memory_args a{};
    a.gpu_id = G().dev[(size_t)ord].kfd_id;
    if (!a.gpu_id || ioctl(G().kfd, AMDKFD_IOC_AVAILABLE_MEMORY, &a) < 0) return -1;
    return (int64_t)a.available;
}

/* Processes holding VRAM on card `ord`, largest first. The driver lists host pids, which differ
 * from this process's numbering inside a container, so this process is named only when they
 * agree. */
std::string card_holders(int ord) {
    const uint32_t id = G().dev[(size_t)ord].kfd_id;
    std::vector<std::pair<int64_t, long>> held;
    if (DIR* d = opendir("/sys/class/kfd/kfd/proc")) {
        while (dirent* e = readdir(d)) {
            char* end = nullptr;
            const long pid = std::strtol(e->d_name, &end, 10);
            if (end == e->d_name || *end) continue;
            char path[128];
            std::snprintf(path, sizeof path, "/sys/class/kfd/kfd/proc/%ld/vram_%u", pid, id);
            FILE* f = std::fopen(path, "r");
            if (!f) continue;
            long long v = 0;
            if (std::fscanf(f, "%lld", &v) == 1 && v > 0) held.push_back({ (int64_t)v, pid });
            std::fclose(f);
        }
        closedir(d);
    }
    if (held.empty()) return std::string();
    std::sort(held.begin(), held.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    std::string s = fmt(" %zu process(es) hold memory on it, by host process id:", held.size());
    bool self = false;
    for (size_t i = 0; i < held.size() && i < 8; ++i) {
        const bool me = held[i].second == (long)getpid();
        self = self || me;
        s += fmt("%s %ld %s%s", i ? "," : "", held[i].second, humanb(held[i].first).c_str(),
                 me ? " (this process)" : "");
    }
    if (held.size() > 8) s += fmt(", and %zu more", held.size() - 8);
    if (!self) s += "; this process is one of them under the host's numbering";
    return s + ".";
}

/* True when card `ord` can take `bytes` more; otherwise false with the refusal set, `what` naming
 * the call. If the driver gives no figure the allocation is allowed. */
bool card_has_room(int ord, int64_t bytes, const char* what) {
    const int64_t avail = card_available(ord);
    if (avail < 0) return true;
    constexpr int64_t kUnit = (int64_t)2 << 20;
    const int64_t need = (bytes + kUnit - 1) / kUnit * kUnit;
    if (need <= avail) return true;
    device_set_error("%s(%s) on card %d: the driver has %s left to give on it.%s Memory another "
                     "process took after the VRAM budget was measured is the usual cause: stop "
                     "that process, or start this one again once it has gone",
                     what, humanb(bytes).c_str(), ord, humanb(avail).c_str(),
                     card_holders(ord).c_str());
    return false;
}

/* ---- elastic memory, through HSA.
 *
 * HIP's virtual-memory calls create HIP's default-stream queue on the card the first time a range
 * is backed (hipMemCreate hands the pages to HIP's command machinery), a hardware queue this
 * process never submits to. Past a card's concurrent-queue limit it time-slices queues, every
 * kernel pays for the switch, and a rank's two lanes stop overlapping. The same operations through
 * HSA create no queue.
 *
 * A card whose pool reports no allocation granule reports granularity 0, and pools are then sized
 * once and allocated whole (VMem::supported()). A backed range is accessible only to its
 * reservation's card: each elastic pool belongs to one rank, and tensor parallel shares through
 * the all-reduce buffers, not these.
 *
 * The owning card is recorded at reserve time, because a commit must map pages on a device and
 * the vtable names none. Each rank reserves with its own card current, while the thread that later
 * grows a pool may have switched cards since. A mapping's handle is kept until the unmap releases
 * it. */
struct VRes { uintptr_t base; uintptr_t end; int ord; };
std::mutex                                                   g_vres_mu;
std::vector<VRes>                                            g_vres;
std::unordered_map<uintptr_t, hsa_amd_vmem_alloc_handle_t>   g_vmaps;

int vres_ord(const void* addr) {
    const uintptr_t a = (uintptr_t)addr;
    std::lock_guard<std::mutex> lk(g_vres_mu);
    for (const VRes& r : g_vres)
        if (a >= r.base && a < r.end) return r.ord;
    return -1;
}

int64_t a_vmem_granularity(void) {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= (int)G().dev.size()) return 0;
    size_t gran = 0;
    if (hsa_amd_memory_pool_get_info(G().dev[(size_t)dev].vram,
                                     HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &gran) !=
            HSA_STATUS_SUCCESS || gran == 0)
        return 0;
    /* The pool's granule is only a floor: a map costs tens of us whatever its size. Every pool is
     * already aligned to RAD_ALIGN_POOL, and using it matches the host backend's granule. */
    const int64_t g = (int64_t)gran < (int64_t)RAD_ALIGN_POOL ? (int64_t)RAD_ALIGN_POOL : (int64_t)gran;
    return align_up(g, (int64_t)RAD_ALIGN_POOL);
}

void* a_vmem_reserve(int64_t bytes, int64_t align) {
    if (bytes <= 0) return nullptr;
    /* The reservation's card decides where a_vmem_commit maps its pages. */
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        device_set_error("vmem: no current device to reserve %lld bytes for", (long long)bytes);
        return nullptr;
    }
    void* p = nullptr;
    if (hsa_amd_vmem_address_reserve_align(&p, (size_t)bytes, 0,
                                           (uint64_t)(align > 0 ? align : 0), 0) !=
            HSA_STATUS_SUCCESS || !p) {
        device_set_error("vmem: %lld bytes of address space could not be reserved",
                         (long long)bytes);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_vres_mu);
    g_vres.push_back({ (uintptr_t)p, (uintptr_t)p + (uintptr_t)bytes, dev });
    return p;
}

void a_vmem_release(void* base, int64_t bytes) {
    if (!base || bytes <= 0) return;
    loc_forget();
    {
        std::lock_guard<std::mutex> lk(g_vres_mu);
        for (size_t i = 0; i < g_vres.size(); ++i)
            if (g_vres[i].base == (uintptr_t)base) { g_vres.erase(g_vres.begin() + (long)i); break; }
    }
    (void)hsa_amd_vmem_address_free(base, (size_t)bytes);
}

/* Backs the range and grants the reservation's card read/write access at once. HSA grants access
 * per mapping, so vmem_protect has nothing to add. */
int a_vmem_commit(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    int dev = vres_ord(addr);
    if (dev < 0 && hipGetDevice(&dev) != hipSuccess) return RAD_E_DEVICE;
    if (dev < 0 || dev >= (int)G().dev.size()) return RAD_E_DEVICE;
    const Dev& d = G().dev[(size_t)dev];
    if (!card_has_room(dev, bytes, "vmem commit")) return RAD_E_NOMEM;
    hsa_amd_vmem_alloc_handle_t h{};
    if (hsa_amd_vmem_handle_create(d.vram, (size_t)bytes, MEMORY_TYPE_NONE, 0, &h) !=
        HSA_STATUS_SUCCESS) {
        device_set_error("vmem: %lld bytes could not be backed on device %d", (long long)bytes, dev);
        return RAD_E_NOMEM;
    }
    if (hsa_amd_vmem_map(addr, (size_t)bytes, 0, h, 0) != HSA_STATUS_SUCCESS) {
        (void)hsa_amd_vmem_handle_release(h);
        device_set_error("vmem: a map of %lld bytes at %p failed", (long long)bytes, addr);
        return RAD_E_DEVICE;
    }
    const hsa_amd_memory_access_desc_t acc{ HSA_ACCESS_PERMISSION_RW, d.agent };
    if (hsa_amd_vmem_set_access(addr, (size_t)bytes, &acc, 1) != HSA_STATUS_SUCCESS) {
        (void)hsa_amd_vmem_unmap(addr, (size_t)bytes);
        (void)hsa_amd_vmem_handle_release(h);
        device_set_error("vmem: device %d could not be given access to %p", dev, addr);
        return RAD_E_DEVICE;
    }
    std::lock_guard<std::mutex> lk(g_vres_mu);
    g_vmaps[(uintptr_t)addr] = h;
    loc_forget();   /* the range is backed, no longer only reserved */
    return RAD_OK;
}

int a_vmem_protect(void* addr, int64_t bytes) {
    return (!addr || bytes <= 0) ? RAD_E_INVAL : RAD_OK;
}

int a_vmem_decommit(void* addr, int64_t bytes) {
    if (!addr || bytes <= 0) return RAD_E_INVAL;
    loc_forget();
    hsa_amd_vmem_alloc_handle_t h{};
    bool have = false;
    {
        std::lock_guard<std::mutex> lk(g_vres_mu);
        auto it = g_vmaps.find((uintptr_t)addr);
        if (it != g_vmaps.end()) { h = it->second; have = true; g_vmaps.erase(it); }
    }
    if (hsa_amd_vmem_unmap(addr, (size_t)bytes) != HSA_STATUS_SUCCESS) {
        device_set_error("vmem: an unmap of %lld bytes at %p failed", (long long)bytes, addr);
        return RAD_E_DEVICE;
    }
    if (have) (void)hsa_amd_vmem_handle_release(h);
    return RAD_OK;
}

/* ------------------------------------------------------------------ pinned host memory, in pieces
 *
 * Every pinned or mapped host allocation is one address range locked in pieces of at most
 * kPinPiece, because of the IOMMU. Card-reachable host memory is a userptr buffer object that the
 * kernel DMA-maps per card as one scatterlist. On Intel VT-d in scalable mode that mapping fails
 * for any single buffer over ~2.5 GiB, so a large host pool allocated whole is refused far below
 * the driver's pinned-memory cap. iommu=pt allows the large pin but removes peer access, which
 * every multi-rank kernel needs, so translation must stay on.
 *
 * The lock is ROCr's hsa_amd_memory_lock_to_pool, the same userptr pin hipHostMalloc makes, one
 * buffer object per call. hsa_amd_pointer_info does not report locked ranges, so locate() answers
 * from g_pinned and HIP is never asked. (hipHostRegister would record them in HIP, which the
 * copies do not read.)
 *
 * The cards still see one range: each piece is mapped at its host address, so the pieces are
 * contiguous on the card too. This is checked per piece, because the pools, the mover and the
 * activation arena all hand kernels base + offset.
 *
 * In-place mapping needs the KFD's SVM API, through which ROCr's thunk registers the range.
 * Without it (e.g. amdgpu could not register card memory with HMM, seen in a KVM guest with
 * passed-through cards) each lock becomes a userptr buffer object at an address from the thunk's
 * aperture: below the host address, with a guard page between pieces, so pieces are neither in
 * place nor contiguous. hsa_amd_vmem cannot join them (ROCm 7.2 refuses a vmem handle on the host
 * pool), but a host-pool allocation can: the thunk maps it for the CPU at the cards' address. It
 * is one buffer object, so a host whose IOMMU refuses large pins and whose driver lacks SVM has no
 * large pinned range at all; the refusal names both. The path is chosen by
 * HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED, the capability the thunk uses, and logged once.
 *
 * Kept from hipHostMalloc: huge pages advised from 2 MiB up, preferred NUMA node nearest the
 * current card (ROCr allocates from that node's pool), fine-grained coherence (the pool locked
 * to), and access from every card. The memory is private anonymous, not hipHostMalloc's shared
 * kind: shared anonymous memory is shmem, whose huge pages follow
 * transparent_hugepage/shmem_enabled (`never` on stock Ubuntu), while private memory follows
 * transparent_hugepage/enabled (`madvise` there). Under IOMMU translation each page is its own
 * translation, and streaming from a 4 KiB-page pool made prefill 10-21% slower. MADV_DONTFORK
 * keeps a forked child from copying the pinned range whole.
 *
 * kPinPiece is a power of two well under the failing size; the cost is one buffer object per
 * piece per card, a few dozen for the largest pools. */
constexpr int64_t kPinPiece = (int64_t)1 << 30;

/* The NUMA node sysfs reports for the current card, -1 when it reports none. */
int current_card_numa_node() {
    int dev = 0;
    char bus[64] = {0};
    if (hipGetDevice(&dev) != hipSuccess ||
        hipDeviceGetPCIBusId(bus, (int)sizeof bus, dev) != hipSuccess || !bus[0])
        return -1;
    for (char* c = bus; *c; ++c) *c = (char)std::tolower((unsigned char)*c);
    char path[160];
    std::snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/numa_node", bus);
    FILE* f = std::fopen(path, "r");
    if (!f) return -1;
    int node = -1;
    if (std::fscanf(f, "%d", &node) != 1) node = -1;
    std::fclose(f);
    return node;
}

/* MPOL_PREFERRED | MPOL_F_STATIC_NODES, the policy ROCr's host allocations use. On refusal (seccomp
 * blocking mbind, a node without memory) pages land on the first-touching node, a placement cost
 * only: warn once and continue. */
void prefer_numa_node(void* p, int64_t len, int node) {
    constexpr int kMaxNodes = 1024;
    constexpr int kBits = 8 * (int)sizeof(unsigned long);
    if (node < 0 || node >= kMaxNodes) return;
    unsigned long mask[kMaxNodes / kBits] = {};
    mask[node / kBits] |= 1ul << (node % kBits);
    constexpr int kMpolPreferred = 1, kMpolStaticNodes = 1 << 15;
    if (::syscall(SYS_mbind, p, (unsigned long)len, kMpolPreferred | kMpolStaticNodes, mask,
                  (unsigned long)kMaxNodes + 1, 0u) == 0)
        return;
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        RAD_WARN("device: pinned host memory could not prefer NUMA node %d, the one nearest the "
                 "card (mbind: %s); its pages land on whichever node touches them first", node,
                 std::strerror(errno));
}

/* With transparent huge pages off, pinned ranges get 4 KiB pages whatever they advise. Warned
 * once, because under IOMMU translation that slows streaming from the host pool and nothing else
 * in the log would explain it. */
void warn_if_no_huge_pages() {
    static std::atomic<bool> said{false};
    if (said.exchange(true)) return;
    std::ifstream f("/sys/kernel/mm/transparent_hugepage/enabled");
    std::string mode;
    if (!std::getline(f, mode) || mode.find("[never]") == std::string::npos) return;
    RAD_WARN("device: transparent huge pages are off (/sys/kernel/mm/transparent_hugepage/enabled "
             "is never), so pinned host memory gets 4 KiB pages; where an IOMMU translates the "
             "cards' reads, streaming from the host pool is slower. 'madvise' gives the pinned "
             "ranges huge pages and nothing else.");
}

const char* hsa_why(hsa_status_t st) {
    const char* m = nullptr;
    return hsa_status_string(st, &m) == HSA_STATUS_SUCCESS && m ? m : "unknown HSA status";
}

/* Whether the driver offers the SVM API, so a locked piece is mapped in place. */
bool svm_in_place() {
    static const bool on = [] {
        bool svm = false;
        if (hsa_system_get_info((hsa_system_info_t)HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED, &svm) !=
            HSA_STATUS_SUCCESS)
            svm = false;
        if (!svm)
            RAD_INFO("device: the kernel driver offers no SVM API, so a pinned host range is one "
                     "allocation of the host pool rather than locked in pieces of %s",
                     humanb(kPinPiece).c_str());
        return svm;
    }();
    return on;
}

/* The range as one host-pool allocation, for a driver without the SVM API. The pool aligns only
 * a buffer's end to 2 MiB, so one extra alignment is requested and the base rounded up within it. */
void* pin_whole(int64_t bytes, const std::vector<hsa_agent_t>& cards) {
    const int64_t align = RAD_ALIGN_POOL;
    const int64_t len = align_up(bytes, 4096);
    const int64_t ask = align_up(len, align) + align;
    void* q = nullptr;
    hsa_status_t st = hsa_amd_memory_pool_allocate(G().host_fine, (size_t)ask, 0, &q);
    if (st != HSA_STATUS_SUCCESS) {
        device_set_error("rad_dev_alloc(%s): the kernel driver offers no SVM API, so a pinned "
                         "range has to be one allocation of the host pool (locked pieces would "
                         "land apart on the cards), and the pool refused %s: %s",
                         humanb(bytes).c_str(), humanb(ask).c_str(), hsa_why(st));
        return nullptr;
    }
    st = hsa_amd_agents_allow_access((uint32_t)cards.size(), cards.data(), nullptr, q);
    if (st != HSA_STATUS_SUCCESS) {
        device_set_error("rad_dev_alloc(%s): giving the cards access to the host-pool range: %s",
                         humanb(bytes).c_str(), hsa_why(st));
        (void)hsa_amd_memory_pool_free(q);
        return nullptr;
    }
    char* const p = (char*)(((uintptr_t)q + (uintptr_t)align - 1) & ~((uintptr_t)align - 1));
    std::lock_guard<std::mutex> lk(g_pinned_mu);
    g_pinned.emplace((uintptr_t)p, Pinned{len, q});
    return p;
}

void* pin_pieces(int64_t bytes) {
    std::vector<hsa_agent_t> cards;
    for (const Dev& d : G().dev) cards.push_back(d.agent);
    if (!svm_in_place()) return pin_whole(bytes, cards);

    const int64_t align = RAD_ALIGN_POOL;
    const int64_t len = align_up(bytes, 4096);
    /* Map one alignment extra and trim both ends so the range starts on a 2 MiB boundary: a
     * pool's fixed-stride slots inherit its base alignment, and huge pages need it. */
    void* m = ::mmap(nullptr, (size_t)(len + align), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        device_set_error("rad_dev_alloc(%s): mmap: %s", humanb(bytes).c_str(), std::strerror(errno));
        return nullptr;
    }
    const uintptr_t lo = (uintptr_t)m;
    const uintptr_t at = (lo + (uintptr_t)align - 1) & ~((uintptr_t)align - 1);
    if (at > lo) ::munmap(m, at - lo);
    if (lo + (uintptr_t)align > at) ::munmap((void*)(at + (uintptr_t)len), lo + (uintptr_t)align - at);
    char* const p = (char*)at;

    prefer_numa_node(p, len, current_card_numa_node());
    if (len >= ((int64_t)2 << 20)) {
        ::madvise(p, (size_t)len, MADV_HUGEPAGE);
        warn_if_no_huge_pages();
    }
    (void)::madvise(p, (size_t)len, MADV_DONTFORK);

    const int pieces = (int)((len + kPinPiece - 1) / kPinPiece);
    int done = 0;
    for (; done < pieces; ++done) {
        const int64_t off = (int64_t)done * kPinPiece;
        const int64_t n = std::min(kPinPiece, len - off);
        void* on_card = nullptr;
        const hsa_status_t st = hsa_amd_memory_lock_to_pool(p + off, (size_t)n, cards.data(),
                                                            (int)cards.size(), G().host_fine, 0,
                                                            &on_card);
        if (st != HSA_STATUS_SUCCESS) {
            device_set_error("rad_dev_alloc(%s): locking piece %d of %d (%s at +%s): %s",
                             humanb(bytes).c_str(), done + 1, pieces, humanb(n).c_str(),
                             humanb(off).c_str(), hsa_why(st));
            break;
        }
        if (on_card != p + off) {
            ++done;
            device_set_error("rad_dev_alloc(%s): piece %d of %d is mapped on the cards at %p, not "
                             "at its host address %p, so the pieces are not one range there, "
                             "although the kernel driver reports the SVM API that maps a lock in "
                             "place",
                             humanb(bytes).c_str(), done, pieces, on_card, (void*)(p + off));
            break;
        }
    }
    if (done < pieces) {
        for (int i = 0; i < done; ++i) (void)hsa_amd_memory_unlock(p + (int64_t)i * kPinPiece);
        ::munmap(p, (size_t)len);
        return nullptr;
    }
    std::lock_guard<std::mutex> lk(g_pinned_mu);
    g_pinned.emplace((uintptr_t)p, Pinned{len, nullptr});
    return p;
}

void unpin_pieces(void* p) {
    Pinned r;
    {
        std::lock_guard<std::mutex> lk(g_pinned_mu);
        auto it = g_pinned.find((uintptr_t)p);
        if (it == g_pinned.end()) {
            RAD_WARN("rad_dev_free(%p): not the base of a pinned range this backend handed out; "
                     "leaked rather than unpinned", p);
            return;
        }
        r = it->second;
        g_pinned.erase(it);
    }
    if (r.pool) {
        const hsa_status_t st = hsa_amd_memory_pool_free(r.pool);
        if (st != HSA_STATUS_SUCCESS)
            RAD_WARN("device: freeing pinned host-pool range %p: %s", r.pool, hsa_why(st));
        return;
    }
    const int64_t len = r.len;
    for (int64_t off = 0; off < len; off += kPinPiece) {
        const hsa_status_t st = hsa_amd_memory_unlock((char*)p + off);
        if (st != HSA_STATUS_SUCCESS)
            RAD_WARN("device: unlocking pinned piece %p: %s", (void*)((char*)p + off), hsa_why(st));
    }
    ::munmap(p, (size_t)len);
}

bool pinned_kind(int kind) { return kind == RAD_MEM_HOST_PINNED || kind == RAD_MEM_HOST_MAPPED; }

/* Device and pageable memory come from HIP's allocation calls; the pinned kinds are locked in
 * pieces. RAD_MEM_HOST_PINNED (mover staging and the O_DIRECT landing zone, spec §5.4) and
 * RAD_MEM_HOST_MAPPED (the zero-copy execution site, spec §5.1) are the same thing here: a locked
 * range every card addresses at its host address. */
void* (*g_hip_alloc)(int64_t, int) = nullptr;
void* a_alloc(int64_t bytes, int kind) {
    if (kind == RAD_MEM_DEVICE && bytes > 0) {
        int dev = -1;
        if (hipGetDevice(&dev) == hipSuccess && !card_has_room(dev, bytes, "rad_dev_alloc"))
            return nullptr;
    }
    if (!pinned_kind(kind)) {
        void* p = g_hip_alloc(bytes, kind);
        note_alloc(p, bytes, kind);
        return p;
    }
    if (bytes <= 0) {
        device_set_error("rad_dev_alloc(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    void* p = pin_pieces(bytes);
    note_alloc(p, bytes, kind);
    return p;
}

/* Every free invalidates locate()'s cache first. */
void (*g_hip_free)(void*, int) = nullptr;
void a_free(void* p, int kind) {
    loc_forget();
    if (!p) return;
    note_free(p);
    if (pinned_kind(kind)) unpin_pieces(p);
    else if (g_hip_free) g_hip_free(p, kind);
}

/* A pinned range has the same address on host and device, interior addresses included. */
void* (*g_hip_host_ptr)(void*) = nullptr;
void* (*g_hip_device_ptr)(void*) = nullptr;
void* a_host_ptr(void* p) { return pinned_range(p, nullptr, nullptr) ? p : g_hip_host_ptr(p); }
void* a_device_ptr(void* p) { return pinned_range(p, nullptr, nullptr) ? p : g_hip_device_ptr(p); }

/* ------------------------------------------------------------------ plugin device code */

size_t a_code_mark(void) {
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    return r.order.size();
}

/* Whether every fat binary registered in [from, to) has a code object `device` runs. Makes the
 * same choice as the first launch (module_meta), early enough that a library with nothing for
 * this card is left out of selection instead of failing at its first launch. */
int a_code_runs(size_t from, size_t to, int device, char* why, int cap) {
    if (device < 0 || (size_t)device >= G().dev.size()) return RAD_E_INVAL;
    const std::vector<std::string>& isas = G().dev[(size_t)device].isas;
    std::vector<const void*> bundles;
    {
        Registry& r = reg();
        std::lock_guard<std::mutex> lk(r.mu);
        for (size_t i = from; i < to && i < r.order.size(); ++i) bundles.push_back(r.order[i]->bundle);
    }
    for (const void* b : bundles) {
        const void* co = nullptr;
        size_t size = 0;
        std::string err;
        if (!bundle_code_object(b, isas, &co, &size, &err)) {
            if (why && cap > 0) std::snprintf(why, (size_t)cap, "%s", err.c_str());
            return 0;
        }
    }
    return 1;
}

DeviceBackend g_aql;

}  // namespace

size_t aql_code_registered(void) { return a_code_mark(); }

const DeviceBackend* aql_backend_if_present(void) {
    static const DeviceBackend* b = []() -> const DeviceBackend* {
        const DeviceBackend* hip = hip_backend_if_present();
        if (!hip) return nullptr;
        /* The interposition must be live, or plugin launches would hand HIP queues it does not
         * know. It is live when this executable exports hipLaunchKernel, as every target linking
         * the core does (core/CMakeLists.txt); otherwise it is a build error, reported as such. */
        void* bound = dlsym(RTLD_DEFAULT, "hipLaunchKernel");
        if (bound != reinterpret_cast<void*>(&::hipLaunchKernel)) {
            RAD_ERR("device: this executable does not export hipLaunchKernel, so kernel plugins "
                    "would launch through HIP onto queues HIP does not own. Link it with the "
                    "core's interface link options.");
            std::abort();
        }
        if (!init_devices()) {
            RAD_ERR("device: the AQL backend cannot start: %s", G().why.c_str());
            std::abort();
        }
        G().ok = true;
        g_aql = *hip;
        g_aql.name = "aql";
        g_hip_alloc = hip->alloc;
        g_hip_free = hip->free;
        g_hip_host_ptr = hip->host_ptr;
        g_hip_device_ptr = hip->device_ptr;
        g_aql.alloc = a_alloc;
        g_aql.free = a_free;
        g_aql.host_ptr = a_host_ptr;
        g_aql.device_ptr = a_device_ptr;
        g_aql.stream_create = a_stream_create;
        g_aql.stream_destroy = a_stream_destroy;
        g_aql.stream_sync = a_stream_sync;
        g_aql.event_create = a_event_create;
        g_aql.event_destroy = a_event_destroy;
        g_aql.event_record = a_event_record;
        g_aql.event_wait = a_event_wait;
        g_aql.event_query = a_event_query;
        g_aql.event_sync = a_event_sync;
        g_aql.host_wrote = a_host_wrote;
        g_aql.event_elapsed_ms = a_event_elapsed_ms;
        g_aql.memcpy_async = a_memcpy_async;
        g_aql.memcpy_2d_async = a_memcpy_2d_async;
        g_aql.memset_async = a_memset_async;
        g_aql.memcpy_ranges_async = a_memcpy_ranges_async;
        g_aql.memcpy_list_async = a_memcpy_list_async;
        g_aql.link_hold = a_link_hold;
        g_aql.link_gate = a_link_gate;
        g_aql.xp_supported = a_xp_supported;
        g_aql.xp_stream_create = a_xp_stream_create;
        g_aql.ride = a_ride;
        g_aql.ride_wait = a_ride_wait;
        g_aql.ride_release = a_ride_release;
        g_aql.xp_list = a_xp_list;
        g_aql.xp_behind = a_xp_behind;
        g_aql.alloc_uncached = a_alloc_uncached;
        g_aql.free_uncached = a_free_uncached;
        g_aql.dma_lane = a_dma_lane;
        g_aql.vmem_granularity = a_vmem_granularity;
        g_aql.vmem_reserve = a_vmem_reserve;
        g_aql.vmem_release = a_vmem_release;
        g_aql.vmem_commit = a_vmem_commit;
        g_aql.vmem_decommit = a_vmem_decommit;
        g_aql.vmem_protect = a_vmem_protect;
        g_aql.tape_begin = a_tape_begin;
        g_aql.tape_mark = a_tape_mark;
        g_aql.tape_pause = a_tape_pause;
        g_aql.tape_end = a_tape_end;
        g_aql.tape_len = a_tape_len;
        g_aql.tape_play = a_tape_play;
        g_aql.tape_diff = a_tape_diff;
        g_aql.tape_record = a_tape_record;
        g_aql.code_mark = a_code_mark;
        g_aql.code_runs = a_code_runs;
        g_aql.tape_set_overlap = a_tape_set_overlap;
        g_aql.tape_destroy = a_tape_destroy;
        g_aql.stream_gate = a_stream_gate;
        g_aql.gate_open = a_gate_open;
        g_aql.gate_room = a_gate_room;
        return &g_aql;
    }();
    return b;
}

int64_t aql_pageable_bytes(void) { return aql::g_pageable_bytes.load(std::memory_order_relaxed); }

int aql_vmem_device_of(const void* addr, int* dev) {
    const int d = vres_ord(addr);
    if (d < 0) return RAD_E_NOTFOUND;
    *dev = d;
    return RAD_OK;
}

}  // namespace rad

/* ================================================================== the HIP entry points
 *
 * Defined in the executable, so the dynamic linker binds plugin references here ahead of the HIP
 * runtime. Each forwards anything not on this backend's streams to HIP. */

using namespace rad::aql;

extern "C" {

void** __hipRegisterFatBinary(const void* data) {
    void** h = real().reg_fat ? real().reg_fat(data) : nullptr;
    /* The wrapper is { magic, version, binary, reserved }; `binary` is the offload bundle. */
    struct Wrapper { unsigned magic, version; const void* binary; void* reserved; };
    const Wrapper* w = static_cast<const Wrapper*>(data);
    if (w && w->binary) {
        Registry& r = reg();
        std::lock_guard<std::mutex> lk(r.mu);
        /* HIP returns no handle for a fat binary with nothing for any visible device. It is still
         * recorded (unlaunchable) so the plugin loader's check (a_code_runs) can catch it. */
        if (!h) {
            Module* m = new Module();
            m->bundle = w->binary;
            r.order.push_back(m);
            return h;
        }
        Module*& m = r.by_handle[h];
        /* HIP may reuse a handle after a plugin is unloaded (the loader closes refused ones); it
         * then names a new fat binary, and the old record's code object points into an unmapped
         * image. Records are never freed (see __hipUnregisterFatBinary), so the old one is
         * abandoned. */
        if (!m || m->bundle != w->binary) m = new Module();
        m->bundle = w->binary;
        r.order.push_back(m);
    }
    return h;
}

void __hipRegisterFunction(void** modules, const void* host_fn, char* device_fn,
                           const char* device_name, unsigned thread_limit, void* tid, void* bid,
                           void* block_dim, void* grid_dim, int* wsize) {
    if (real().reg_fn)
        real().reg_fn(modules, host_fn, device_fn, device_name, thread_limit, tid, bid, block_dim,
                      grid_dim, wsize);
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.by_handle.find(modules);
    if (it == r.by_handle.end() || !host_fn || !device_name) return;
    Func*& f = r.by_stub[host_fn];
    if (!f) f = new Func();
    f->mod = it->second;
    f->name = device_name;
}

void __hipUnregisterFatBinary(void** modules) {
    /* The module's executables and records stay: a process unloading a kernel plugin is exiting,
     * and a late launch would dereference them. */
    if (real().unreg_fat) real().unreg_fat(modules);
}

hipError_t hipLaunchKernel(const void* f, dim3 g, dim3 b, void** args, size_t shm, hipStream_t st) {
    Stream* s = ours(st);
    if (!s) {
        g_hip_streams_used.store(true, std::memory_order_relaxed);
        return real().launch ? real().launch(f, g, b, args, shm, st) : hipErrorNotSupported;
    }
    const uint8_t fence = t_next_fence;
    t_next_fence = 0;
    std::string why;
    KernelRec* k = kernel_for(f, s->ord, &why);
    if (!k) {
        RAD_ERR("device %d: cannot dispatch a kernel: %s", s->ord, why.c_str());
        return ret(hipErrorInvalidDeviceFunction);
    }
    return ret(dispatch(s, k, g, b, args, shm, fence) < 0 ? hipErrorLaunchFailure : hipSuccess);
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                            size_t height, hipMemcpyKind kind, hipStream_t st) {
    if (!ours(st)) {
        g_hip_streams_used.store(true, std::memory_order_relaxed);
        return real().memcpy2d ? real().memcpy2d(dst, dpitch, src, spitch, width, height, kind, st)
                               : hipErrorNotSupported;
    }
    return ret(rad_memcpy_2d_async(dst, (int64_t)dpitch, src, (int64_t)spitch, (int64_t)width,
                                   (int64_t)height, (RadStream)st) < 0 ? hipErrorInvalidValue
                                                                       : hipSuccess);
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind,
                          hipStream_t st) {
    if (!ours(st)) {
        g_hip_streams_used.store(true, std::memory_order_relaxed);
        return real().memcpy ? real().memcpy(dst, src, bytes, kind, st) : hipErrorNotSupported;
    }
    return ret(rad_memcpy_async(dst, src, (int64_t)bytes, (RadStream)st) < 0 ? hipErrorInvalidValue
                                                                             : hipSuccess);
}

hipError_t hipMemsetAsync(void* dst, int value, size_t bytes, hipStream_t st) {
    if (!ours(st)) {
        g_hip_streams_used.store(true, std::memory_order_relaxed);
        return real().memset ? real().memset(dst, value, bytes, st) : hipErrorNotSupported;
    }
    return ret(rad_memset_async(dst, value, (int64_t)bytes, (RadStream)st) < 0 ? hipErrorInvalidValue
                                                                              : hipSuccess);
}

/* A synchronous fill on one of the backend's own queues for the current device. Kernel libraries
 * zero small private blocks this way (an all-reduce's flags at init, a split-K GEMM's arrival
 * counters on first use), and HIP would create its default-stream queues on the card to serve
 * them: extra hardware queues that make the card time-slice, so every kernel pays a switch and a
 * rank's two lanes stop overlapping.
 *
 * The fill runs on the first queue this process made on the device (the owning rank's compute
 * stream) and is waited for. A device with no queue yet (a peer's buffers set up before its rank
 * starts) gets a temporary one for the call. Never recorded into a tape: it runs once. */
hipError_t hipMemset(void* dst, int value, size_t bytes) {
    if (!G().ok) return real().memset_sync ? real().memset_sync(dst, value, bytes) : hipErrorNotSupported;
    int dev = -1;
    if (hipGetDevice(&dev) != hipSuccess) return ret(hipErrorInvalidDevice);
    Stream* s = nullptr;
    {
        std::lock_guard<std::mutex> lk(G().streams_mu);
        for (Stream* x : G().streams)
            if (x->ord == dev) { s = x; break; }
    }
    RadStream own = nullptr;
    if (!s) {
        if (rad_stream_create(&own, 0) < 0) return ret(hipErrorOutOfMemory);
        s = ours(own);
    }
    ++t_tape_pause;
    int rc = rad_memset_async(dst, value, (int64_t)bytes, (RadStream)s);
    if (rc >= 0) rc = stream_drain(s);
    --t_tape_pause;
    if (own) rad_stream_destroy(own);
    return ret(rc < 0 ? hipErrorInvalidValue : hipSuccess);
}

hipError_t hipStreamSynchronize(hipStream_t st) {
    if (!st && !g_hip_streams_used.load(std::memory_order_relaxed)) return ret(hipSuccess);
    if (!ours(st)) return real().sync ? real().sync(st) : hipErrorNotSupported;
    return ret(rad_stream_sync((RadStream)st) < 0 ? hipErrorLaunchFailure : hipSuccess);
}

/* HIP events have no meaning on this backend's queues (events go through rad_event_*). Refused
 * instead of forwarded, since HIP would dereference the queue as one of its own streams. */
hipError_t hipEventRecord(hipEvent_t e, hipStream_t st) {
    if (!ours(st)) return real().ev_record ? real().ev_record(e, st) : hipErrorNotSupported;
    RAD_ERR("hipEventRecord on an engine stream: use rad_event_record");
    return ret(hipErrorNotSupported);
}

hipError_t hipStreamWaitEvent(hipStream_t st, hipEvent_t e, unsigned flags) {
    if (!ours(st)) return real().wait_ev ? real().wait_ev(st, e, flags) : hipErrorNotSupported;
    RAD_ERR("hipStreamWaitEvent on an engine stream: use rad_event_wait");
    return ret(hipErrorNotSupported);
}

hipError_t hipGetLastError(void) {
    if (t_ours) {
        const hipError_t e = t_err;
        t_ours = false;
        t_err = hipSuccess;
        /* Clear HIP's own slot: it holds an error from before this backend's call, which that
         * call superseded. */
        if (real().last) (void)real().last();
        return e;
    }
    return real().last ? real().last() : hipSuccess;
}

hipError_t hipPeekAtLastError(void) {
    if (t_ours) return t_err;
    return real().peek ? real().peek() : hipSuccess;
}

}  // extern "C"

#endif /* RAD_HAVE_HIP */
