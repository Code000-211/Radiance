/* aql.cpp -- the AQL device backend: every stream is an HSA queue the engine owns, and every
 * kernel, copy, fill and fence on it is an AQL packet this file writes.
 *
 * WHY THE ENGINE WRITES ITS OWN PACKETS. hipLaunchKernel costs 0.5-1.3 us of host time a
 * dispatch, where writing the packet is 6 ns; and HIP fences every kernel at SYSTEM scope, which
 * costs the device about half a microsecond between two dependent kernels that an AGENT-scope
 * fence does not. A decode step is ~1600 dispatches a rank, so both are milliseconds a step. HIP
 * also retires a Command object per launch on a thread of its own, which is most of a core under
 * load. None of that is needed to put a kernel on a queue.
 *
 * WHAT STAYS WITH HIP. Allocation, device properties, peer access, IPC and elastic memory are
 * HIP's, through the HIP backend's own entry points. So is compiling and REGISTERING kernels: a
 * kernel library is built by hipcc and calls hipLaunchKernel exactly as it would under HIP. This
 * file defines hipLaunchKernel in the executable, which the dynamic linker binds every plugin's
 * reference to ahead of the HIP runtime's, and it records what __hipRegisterFatBinary and
 * __hipRegisterFunction are told so a host stub maps to its own code object. A launch onto a
 * stream this backend did not create is forwarded to HIP unchanged.
 *
 * THE FENCES. Two kernels on one queue see each other's writes through the L2 with an agent-scope
 * fence; the system scope is only needed where memory the L2 does not see is involved:
 *
 *   - the kernel argument ring lives in VRAM and the host writes it through the BAR, which the L2
 *     does not snoop. A slot is rewritten one ring lap after it was last read, so a system-scope
 *     acquire is placed at least once per half lap: at agent scope alone, every kernel after the
 *     first lap reads its predecessor's arguments. Slots are 1 KiB so that no
 *     kernel's argument fetch shares a cache line with the next slot;
 *   - memory the host wrote. The first dispatch after the host has waited on any queue takes a
 *     system-scope acquire, since that wait is what host writes for the next step follow;
 *   - copies that read or write host or peer memory carry the scope themselves, and so do the
 *     barrier packets that implement events and synchronisation.
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

/* THE LAST-ERROR SLOT, AS HIP KEEPS IT. Every HIP call overwrites the thread's last error with
 * its own result, success included, and a plugin's `launch; if (hipGetLastError())` depends on
 * exactly that: an error left by some earlier call must not be reported against the launch. An
 * entry point handled here records its result and marks itself as the thread's latest call, and
 * hipGetLastError answers from it while the mark stands. */
thread_local hipError_t t_err = hipSuccess;
thread_local bool       t_ours = false;

inline hipError_t ret(hipError_t e) { t_err = e; t_ours = true; return e; }
thread_local uint8_t    t_next_fence = 0;

/* WHETHER ANYTHING WAS EVER HANDED TO HIP'S OWN STREAMS. Every submission this process makes goes
 * to the backend's queues; a HIP stream gets work only when a caller passes one the backend does
 * not own, and that is forwarded. Until then HIP's default stream is empty, so synchronising it is
 * a no-op answered here -- asking HIP would make it build the stream's queue first. */
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
    /* Every registration in the order it happened, so the fat binaries one dlopen brought in are a
     * contiguous run the plugin loader can check against the card (a_code_runs). */
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

/* The code-object targets an agent runs, in the runtime's order of preference: its own processor
 * first, then the generic target of its family, each with the modes the agent is in -- for one of
 * these R9700s "gfx1201" then "gfx12-generic". A kernel library built for either loads; asking the
 * runtime rather than keeping a family table here is what keeps a card released after this file
 * from needing an edit to it. */
std::vector<std::string> agent_isas(hsa_agent_t a) {
    std::vector<std::string> v;
    hsa_agent_iterate_isas(a, collect_isa, &v);
    if (v.empty()) v.push_back("amdgcn-amd-amdhsa--" + agent_isa(a));
    return v;
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
    }
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
    } else {
        hsa_status_t ld_st = hsa_executable_load_agent_code_object(ex, d.agent, rd, nullptr, nullptr);
        hsa_status_t fr_st = (ld_st == HSA_STATUS_SUCCESS) ? hsa_executable_freeze(ex, nullptr) : HSA_STATUS_ERROR;
        if (ld_st != HSA_STATUS_SUCCESS || fr_st != HSA_STATUS_SUCCESS) {
            m->load_err[ord] = fmt("the code object did not load on %s (load=%d freeze=%d)", d.isa.c_str(), (int)ld_st, (int)fr_st);
        } else {
            m->exec[ord] = ex;
        }
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
    /* THE ARGUMENT SLOT IS FIXED AT 1 KiB, and a kernel whose segment does not fit leaves less
     * than the guard the fence scheme needs between one slot and the next. Refused by name: no
     * kernel in this tree comes near it. */
    if (k->kernarg_size > kKargMax) {
        fn->err[ord] = fmt("its kernel argument segment is %u bytes and the backend's slot holds %u",
                           k->kernarg_size, kKargMax);
        return nullptr;
    }
    /* A dynamic stack needs a scratch size the kernel cannot state; HIP guesses one. This backend
     * does not guess, and no kernel in this tree has one. */
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

/* One lookup a launch, so a thread-local direct-mapped cache sits in front of the registry. */
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
    char*           karg = nullptr;           /* VRAM, CPU-mapped through the BAR */
    std::atomic<bool> busy{false};
    uint64_t        wr = 0;                   /* next packet index */
    uint64_t        rd = 0;                   /* last read index seen */
    int64_t         last_sys_acq = -(int64_t)kQueuePackets;
    uint64_t        epoch = 0;                /* host epoch at the last system acquire */
    std::atomic<int> fault{0};
    char*           bounce = nullptr;         /* pinned, for pageable copies */
    std::mutex      bounce_mu;

    void lock() {
        while (busy.exchange(true, std::memory_order_acquire))
            while (busy.load(std::memory_order_relaxed)) _mm_pause();
    }
    void unlock() { busy.store(false, std::memory_order_release); }
};

/* EVERY QUEUE THIS PROCESS HAS MADE, by a number that is never reused, so a record of "the wait
 * packet at index i of queue n" stays meaningful after the queue is destroyed: a destroyed queue's
 * entry is null, and everything it held has been consumed. */
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

/* WHAT EVERY QUEUE IS DOING, printed once when a host wait has not come back for five seconds.
 * Nothing on the step path takes that long, so a wait that does is a queue stuck on a packet --
 * a barrier on a signal nothing will decrement, or a kernel that faulted without a report -- and
 * the packet at the read index is the answer. */
void dump_queues(const char* where) {
    std::lock_guard<std::mutex> lk(G().streams_mu);
    RAD_ERR("device: %s has waited 5 s; the queues:", where);
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
}

/* How long a host wait goes before it reports the queues; see dump_queues. */
constexpr uint32_t kStallMs = 5000;

/* A spin that notices when it has gone on too long. Checked every 64k iterations, so the clock
 * is read a few thousand times a second at most. */
struct Stall {
    const char* where;
    uint32_t n = 0;
    bool told = false;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    explicit Stall(const char* w) : where(w) {}
    void tick() {
        if (told || (++n & 0xffff)) return;
        check();
    }
    /* Every call, for a loop that sleeps between calls and so makes few of them. */
    void check() {
        if (!told && std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(kStallMs)) {
            told = true;
            dump_queues(where);
        }
    }
};

/* The slot for packet `idx`, once the packet processor has consumed far enough that it is free.
 * A packet with the barrier bit having been LAUNCHED means every packet before it has COMPLETED --
 * which is what a kernel argument slot, read by the whole of its kernel, needs. A played tape
 * leaves the bit off on at most kTapeOverlapRun packets in a row (see TapeRec::overlap), so of any
 * kTapeOverlapRun + 1 consecutive packets one carries it; the margin is that run and one more. */
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
    _mm_sfence();                       /* the argument slot is write-combined VRAM */
    __atomic_store_n(reinterpret_cast<uint32_t*>(
                         static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) +
                         (idx & s->mask)),
                     (uint32_t)header | ((uint32_t)setup << 16), __ATOMIC_RELEASE);
    _mm_sfence();
    hsa_signal_store_screlease(s->q->doorbell_signal, (hsa_signal_value_t)idx);
}

/* ------------------------------------------------------------------ recorded passes
 *
 * A PASS RECORDED ONCE IS WRITTEN STRAIGHT INTO THE QUEUES AFTERWARDS. The engine issues the same
 * pass -- the same kernels with the same arguments -- step after step, and nearly all of a
 * dispatch's host cost is producing it: the architecture's block code, the runtime's operand
 * resolution, the kernel library's argument packing. A recorded dispatch is the finished packet and
 * its finished argument block, which lives in VRAM for as long as the tape does, so playing it back
 * is a 60-byte copy into the queue and a header store.
 *
 * WHAT IS RECORDED is what THIS THREAD submits while a capture is open: dispatches -- kernels, and
 * the backend's own copies and fills -- and event records and waits. Those are kept by EVENT and
 * replayed through the ordinary calls, because a record takes a fresh signal every time (see
 * "events"). A pageable copy stages through the host and waits, which a tape cannot reproduce, and
 * it makes the tape unusable. Other threads' submissions to the same queues are theirs and are not
 * recorded. */
enum : uint8_t { kTapeDispatch = 0, kTapeRecord = 1, kTapeWait = 2 };

struct TapeRec {
    uint8_t  kind = kTapeDispatch;
    uint8_t  fence = 0;                 /* a dispatch's kFence* flags */
    uint32_t karg_off = 0;              /* a dispatch's argument block, in the tape's */
    uint32_t karg_len = 0;
    Stream*  s = nullptr;
    void*    ev = nullptr;              /* a record's or a wait's event */
    hsa_kernel_dispatch_packet_t pkt{}; /* a dispatch's packet, all but the header word */
    /* A dispatch played WITHOUT the barrier bit, which lets the packet processor launch it while
     * the packets before it on its queue are still running. Set by the recorder only where it has
     * shown the dispatch reads and writes nothing those packets write (tape_set_overlap). */
    uint8_t  overlap = 0;
};

struct Tape {
    uint64_t             magic = kTapeMagic;
    std::vector<TapeRec> rec;
    std::vector<unsigned char> karg_host;   /* the argument blocks, kept to compare tapes by */
    char*                karg = nullptr;    /* VRAM, CPU-mapped, from the end of the capture on */
    std::string          bad;               /* why this tape cannot be played, if it cannot */
    /* The last packet the latest play wrote on each queue. The argument block is freed only once
     * each of them has been consumed. */
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
inline int acquire_scope(Stream* s, uint64_t idx, bool forced, uint64_t ep) {
    const bool lap = (int64_t)idx - s->last_sys_acq >= (int64_t)(kQueuePackets / 2);
    if (__builtin_expect(forced || lap || s->epoch != ep, 0)) {
        s->last_sys_acq = (int64_t)idx;
        s->epoch = ep;
        return HSA_FENCE_SCOPE_SYSTEM;
    }
    return HSA_FENCE_SCOPE_AGENT;
}

inline int acquire_scope(Stream* s, uint64_t idx, bool forced) {
    return acquire_scope(s, idx, forced, G().host_epoch.load(std::memory_order_acquire));
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
    const int acq = acquire_scope(s, idx, (fence & kFenceAcquireSystem) != 0);
    const int rel = (fence & kFenceReleaseSystem) ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT;
    publish(s, idx, header(HSA_PACKET_TYPE_KERNEL_DISPATCH, acq, rel), setup);
    s->unlock();
    return RAD_OK;
}

/* A RUN OF RECORDED DISPATCHES ONTO ONE QUEUE, under one hold of the queue's lock and with the
 * doorbell rung once for every kPlayRing packets rather than once a packet: the doorbell is an
 * uncached write across the link, and a pass is over a thousand dispatches. Each packet's body is
 * copied as recorded and its header -- whose acquire scope depends on what the host wrote since --
 * decided now, as dispatch() decides it. The argument blocks were written to VRAM when the tape
 * was made, so there is nothing to fence before the header. A reserve that would wait for room
 * rings first: the packet processor only takes packets the doorbell has announced. */
constexpr int kPlayRing = 32;

/* WHERE THIS THREAD'S PLAYS LAST LEFT EACH QUEUE. A queue found further on has had packets put on
 * it that no recording saw -- a weight-table patch or a mover wait between two played stretches,
 * another thread's copy -- and the next played packet on it keeps its barrier bit whatever the
 * recording's analysis said: the analysis proved it independent of the packets recorded before
 * it, and those are no longer the ones in front of it. A queue not in the list is treated the
 * same way. */
struct PlayLeft { uint32_t q = UINT32_MAX; uint64_t wr = 0; };
constexpr int kPlayLeft = 8;
thread_local PlayLeft t_left[kPlayLeft];

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static inline void copy_aql_payload_avx2(void* dst, const void* src) {
    const char* s = static_cast<const char*>(src);
    char* d = static_cast<char*>(dst);
    __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 4));
    __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + 32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 4), v0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 32), v1);
}
#endif

struct PlayRun {
    Tape*    tape = nullptr;
    Stream*  s = nullptr;
    uint64_t next = 0;        /* the write index after the last packet written */
    int      unrung = 0;
    bool     foreign = true;  /* the next packet's queue has packets this thread did not play */
    uint64_t epoch = 0;

    explicit PlayRun(Tape* t = nullptr) : tape(t) {}
    ~PlayRun() { end(); }

    void ring() {
        if (!unrung) return;
        _mm_sfence();
        hsa_queue_store_write_index_relaxed(s->q, next);
        hsa_signal_store_screlease(s->q->doorbell_signal, (hsa_signal_value_t)(next - 1));
        unrung = 0;
    }
    void end() {
        if (!s) return;
        ring();
        if (tape) {
            bool found = false;
            for (auto& l : tape->last) {
                if (l.first == s->id) { l.second = s->wr - 1; found = true; break; }
            }
            if (!found) tape->last.emplace_back(s->id, s->wr - 1);
        }
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
    /* `done`, when set, is the packet's completion signal and `rel_min` the least release scope
     * it must carry -- a record riding on this dispatch; see a_tape_play. */
    int put(const TapeRec& r, uint64_t* out_idx, hsa_signal_t done = hsa_signal_t{0},
            int rel_min = HSA_FENCE_SCOPE_NONE) {
        if (__builtin_expect(s != r.s, 0)) {
            end();
            s = r.s;
            s->lock();
            foreign = moved(s);
            epoch = G().host_epoch.load(std::memory_order_acquire);
        }
        const uint64_t idx = s->wr;
        if (__builtin_expect(idx + kReserveMargin - s->rd > s->mask + 1, 0)) {
            ring();
            uint64_t got = 0;
            const int rc = reserve(s, &got);
            if (rc < 0) return rc;
        }
        auto* p = static_cast<hsa_kernel_dispatch_packet_t*>(s->q->base_address) + (idx & s->mask);
#if defined(__x86_64__) || defined(_M_X64)
        static const bool has_avx2 = __builtin_cpu_supports("avx2");
        if (has_avx2) {
            copy_aql_payload_avx2(p, &r.pkt);
        } else {
            std::memcpy(reinterpret_cast<char*>(p) + 4, reinterpret_cast<const char*>(&r.pkt) + 4,
                        sizeof(hsa_kernel_dispatch_packet_t) - 4);
        }
#else
        std::memcpy(reinterpret_cast<char*>(p) + 4, reinterpret_cast<const char*>(&r.pkt) + 4,
                    sizeof(hsa_kernel_dispatch_packet_t) - 4);
#endif
        if (done.handle) p->completion_signal = done;
        const int acq = acquire_scope(s, idx, (r.fence & kFenceAcquireSystem) != 0, epoch);
        int rel = (r.fence & kFenceReleaseSystem) ? HSA_FENCE_SCOPE_SYSTEM : HSA_FENCE_SCOPE_AGENT;
        if (rel < rel_min) rel = rel_min;
        /* A record riding on the dispatch keeps the bit: its completion signal stands for
         * everything before it on the queue, which an unordered dispatch's completion does not. */
        const bool ordered = foreign || !r.overlap || done.handle;
        const uint16_t h = header(HSA_PACKET_TYPE_KERNEL_DISPATCH, acq, rel, ordered);
        __atomic_store_n(reinterpret_cast<uint32_t*>(p), (uint32_t)h | ((uint32_t)r.pkt.setup << 16),
                         __ATOMIC_RELEASE);
        foreign = false;
        s->wr = idx + 1;
        next = idx + 1;
        *out_idx = idx;
        if (__builtin_expect(++unrung >= kPlayRing, 0)) ring();
        return RAD_OK;
    }
};

/* One completion signal a thread for synchronous waits, an interrupt signal (see "host waits"):
 * two threads waiting on the same queue each get their own, and a signal is only ever waited on by
 * the thread that armed it. */
hsa_signal_t thread_signal() {
    thread_local hsa_signal_t s{0};
    if (!s.handle && hsa_signal_create(0, 0, nullptr, &s) != HSA_STATUS_SUCCESS) s.handle = 0;
    return s;
}

/* ------------------------------------------------------------------ host waits
 *
 * A HOST WAIT BLOCKS IN THE DRIVER UNTIL THE DEVICE SAYS IT IS DONE. Every signal the host waits
 * on is an interrupt signal: when the packet that carries it completes, the packet processor posts
 * the signal's event and raises an interrupt, and the driver wakes the threads blocked on that
 * event. A wait costs the host one system call and one wake-up whatever its length, and it makes
 * no guess at how long the device will take, so it can neither sleep past a device that finished
 * early nor spin through one that runs late. What it does cost is the interrupt's delivery: some
 * tens of microseconds between the packet completing and the thread running. That is hidden
 * rather than shortened. The engine keeps work queued behind every point it waits for -- the next
 * step behind a step's end, the layers that do not read a host op's output behind that op's join
 * -- so the device is still running when the host wakes.
 *
 * THE WAIT GOES TO THE DRIVER DIRECTLY. The runtime's own blocking wait polls the signal for a
 * stretch before it blocks, which spends a short wait's whole length on a core; the driver's wait
 * on the signal's event spends nothing until the interrupt arrives.
 *
 * NO WAKE-UP IS LOST. The driver counts each event's firings (its age), and a wait that names an
 * age returns at once if the event has fired since. A thread keeps the age each event last woke it
 * at, so an interrupt that lands between reading the signal and entering the driver ends the wait
 * instead of being missed; an age gone stale costs one early return and a second look.
 *
 * A FAULT ENDS EVERY WAIT. A faulted queue never completes the signal a wait is on, so every wait
 * also names the fault signal's event, which the queue-error callback fires. */

/* The age an event last woke this thread at; 1 is the driver's "never fired". */
uint64_t& seen_age(uint32_t ev) {
    thread_local std::vector<uint64_t> age;
    if (ev >= age.size()) age.resize((size_t)ev + 1, 1);
    return age[ev];
}

/* A SIGNAL WITH NO EVENT CANNOT BE WAITED ON THROUGH THE DRIVER, and one thing hands them out: a
 * profiler. With rocprofv3 attached, every hsa_signal_create after the first queue returns a signal
 * whose event_mailbox_ptr and event_id are 0 (measured with rocprofiler-sdk 1.3.5; before a queue
 * exists the same call returns an interrupt signal, which is why init_devices' own check passes).
 * Such a signal still completes -- the packet processor stores to it -- so the wait goes through the
 * runtime, which polls it, in slices short enough to look at the fault flag between them: a faulted
 * queue never completes the signal. Only a traced run gets here; an untraced one waits on events
 * exactly as before, and the warning says so once rather than letting a slow wait pass unremarked. */
int runtime_wait(hsa_signal_t sig) {
    static std::atomic<bool> told{false};
    if (!told.exchange(true, std::memory_order_relaxed))
        RAD_WARN("device: the runtime hands out signals with no interrupt (a profiler is attached); "
                 "host waits poll them instead of blocking in the driver");
    Stall st("a synchronisation");
    const uint64_t slice = std::max<uint64_t>(1, G().ts_freq / 100);   /* 10 ms of timestamp ticks */
    while (hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_EQ, 0, slice, HSA_WAIT_STATE_BLOCKED) != 0) {
        if (G().faulted.load(std::memory_order_acquire)) return RAD_E_DEVICE;
        st.check();
    }
    G().host_epoch.fetch_add(1, std::memory_order_acq_rel);
    return RAD_OK;
}

int host_wait(hsa_signal_t sig) {
    if (hsa_signal_load_scacquire(sig) != 0) {
        if (!interrupts(sig)) return runtime_wait(sig);
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

/* The store that arms a signal is silent: an interrupt signal's ordinary store also fires its
 * event, which would wake nobody and cost a system call. */
int stream_drain(Stream* s) {
    hsa_signal_t sig = thread_signal();
    if (!sig.handle) return RAD_E_DEVICE;
    hsa_signal_silent_store_relaxed(sig, 1);
    uint64_t idx = 0;
    RAD_TRY(barrier(s, hsa_signal_t{0}, sig, HSA_FENCE_SCOPE_SYSTEM, HSA_FENCE_SCOPE_SYSTEM, &idx));
    return host_wait(sig);
}

/* ------------------------------------------------------------------ where a pointer lives */

enum Loc : uint8_t { kDevLocal, kDevPeer, kHost, kPageable };

/* THE ALLOCATIONS A THREAD COPIES BETWEEN, REMEMBERED. hsa_amd_pointer_info takes the runtime's
 * global lock and walks its allocation map, and every copy asks it about both ends -- a few hundred
 * questions a step about the same dozen allocations, about 17 us of each rank thread's step. A hit
 * here answers from the allocation's own range.
 *
 * Only an answer about a live allocation is kept, never "pageable": memory the runtime does not
 * know may be registered later, and an allocation's kind cannot change while it lives. What can
 * change is WHICH allocation an address belongs to, and only by a free or an unmap -- every one of
 * those in this backend moves `g_loc_gen` first, and an entry of an older generation is ignored.
 * The ranges come back from the runtime itself (sizeInBytes), so a hit is exactly the answer the
 * call would give. A reserved-but-unbacked address is not kept: its card is the reservation's and
 * is looked up per page. */
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

/* Every entry in every thread's cache stops answering. */
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

/* The host ranges this backend pinned itself (see "pinned host memory, in pieces"): base ->
 * length, ordered so an inner address finds its range. */
std::mutex g_pinned_mu;
std::map<uintptr_t, int64_t> g_pinned;

bool pinned_range(const void* p, uintptr_t* lo, int64_t* len) {
    const uintptr_t a = (uintptr_t)p;
    std::lock_guard<std::mutex> lk(g_pinned_mu);
    auto it = g_pinned.upper_bound(a);
    if (it == g_pinned.begin()) return false;
    --it;
    if (a - it->first >= (uintptr_t)it->second) return false;
    if (lo) *lo = it->first;
    if (len) *len = it->second;
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
    /* PINNED HERE, ANSWERED HERE. hsa_amd_pointer_info reports memory locked with
     * hsa_amd_memory_lock_to_pool as UNKNOWN on this runtime -- indistinguishable from pageable --
     * although the lock mapped it on every card at its own address. */
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
    if (hsa_amd_memory_pool_allocate(d.vram, (size_t)(s->mask + 1) * kKargSlot, 0, &ka) !=
            HSA_STATUS_SUCCESS ||
        hsa_amd_agents_allow_access(1, &G().cpu, nullptr, ka) != HSA_STATUS_SUCCESS) {
        hsa_queue_destroy(s->q);
        device_set_error("rad_stream_create: the kernel argument ring could not be placed in VRAM "
                         "the host can write -- the AQL backend needs a large BAR");
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
    s->id = g_next_queue_id.fetch_add(1, std::memory_order_relaxed);
    if (s->id >= kMaxQueues) {
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
    hsa_amd_memory_pool_free(s->karg);
    hsa_amd_memory_pool_free(s->bounce);
    s->magic = 0;
    delete s;
}

int a_stream_sync(RadStream rs) {
    Stream* s = ours(rs);
    if (!s) { device_set_error("rad_stream_sync: not a stream of this backend"); return RAD_E_INVAL; }
    const int rc = stream_drain(s);
    if (rc < 0) device_set_error("rad_stream_sync: the queue on device %d faulted", s->ord);
    return rc;
}

/* ---- events.
 *
 * A RECORD TAKES A FRESH SIGNAL, and a signal is only reused once nothing can still name it. That
 * means two things: its own record has completed, AND every wait packet that was enqueued against
 * it has been consumed. Reusing a signal whose record has completed but whose waiters have not yet
 * been reached re-arms it under them: a lagging queue's wait for record r then waits for whatever
 * record reused the signal, which can sit behind a wait on that same lagging queue. Two lanes that
 * join each other every layer produce exactly that cycle within a step.
 *
 * So each signal carries the (queue, packet index) of every wait that names it, and returns to the
 * free list when both conditions hold. Nothing here ever blocks the host: a signal that cannot be
 * reused yet is simply not reused, and the pool grows to the number of records in flight. */
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
    static SigPool* p = [] {
        auto* sp = new SigPool();
        sp->free[0].reserve(1024);
        sp->free[1].reserve(1024);
        return sp;
    }();
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

/* A signal for one record: an interrupt signal for an event the host waits on, and a GPU_ONLY one
 * otherwise. Only the first kind can end a host wait (see "host waits"), and it costs what the
 * second does not: a driver event, of which a process has a few thousand, and an interrupt at every
 * completion, which an event recorded dozens of times a step and only ever waited on by other
 * queues has no use for. */
Sig* sig_get(bool irq) {
    SigPool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    /* THE OLDEST FIRST. A signal retires when its event is recorded again, so the retiring list is
     * in record order and its front is the signal most likely to be done: taking from the front
     * until one of the kind asked for is free costs a look or two, where a walk of the whole list
     * reads every signal's value and every waiter's queue out of memory the device writes. The
     * walk is kept for when the front is not done -- the host plays a step while the card is
     * still on the one before, and the records that step retired are then still in flight. */
    while (!p.retiring.empty() && reusable(p.retiring.front())) {
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
    /* A signal made after the first queue exists carries no event when a profiler is attached (see
     * runtime_wait). init_devices already proved the runtime makes interrupt signals, so a missing
     * event here is that case: the host wait on this signal polls, and says so once. */
    if (st != HSA_STATUS_SUCCESS) return nullptr;
    return g.release();
}

void sig_retire(Sig* g) {
    SigPool& p = pool();
    std::lock_guard<std::mutex> lk(p.mu);
    if (reusable(g)) {
        g->waiters.clear();
        p.free[g->irq].push_back(g);
    } else {
        p.retiring.push_back(g);
    }
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

/* The event goes; its signal stays with the pool until the waits that name it are consumed. */
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

/* A fresh signal armed for one record of `e` on `s`, and the release scope that record needs. The
 * caller puts it on a packet and then hands it to record_adopt. */
Sig* record_arm(Event* e, Stream* s, int* rel) {
    Sig* g = sig_get(e->host);
    if (!g) return nullptr;
    hsa_signal_silent_store_relaxed(g->h, 1);
    g->agent = s->agent;
    /* A LOCAL EVENT RELEASES TO THE CARD, which its own L2 already is. The system scope has to
     * wait out every write still on its way to host memory, and while the link is carrying
     * weights that is a stall of up to tens of microseconds in the recording stream. */
    *rel = e->local ? HSA_FENCE_SCOPE_AGENT : HSA_FENCE_SCOPE_SYSTEM;
    return g;
}

/* The record carrying `g` is on its queue: it becomes the event's latest, and the one before goes
 * back to the pool once the waits that name it are consumed. */
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

/* The event's lock is held across the enqueue so the signal cannot be retired between being read
 * and having this wait noted against it. */
int a_event_wait(RadStream rs, RadEvent re) {
    Event* e = ev_of(re);
    Stream* s = ours(rs);
    if (!e || !s) { device_set_error("rad_event_wait: null or foreign argument"); return RAD_E_INVAL; }
    std::lock_guard<std::mutex> lk(e->mu);
    if (!e->cur) return RAD_OK;               /* never recorded: nothing to wait for */
    for (const auto& w : e->cur->waiters) {
        if (w.first == s->id) return RAD_OK;
    }
    if (capturing()) {
        TapeRec r;
        r.kind = kTapeWait;
        r.s = s;
        r.ev = re;
        t_tape->rec.push_back(r);
    }
    uint64_t idx = 0;
    /* A producer on this stream's own card wrote through the L2 this stream reads, so the acquire
     * need only reach the card; one on another card, or the host, needs the system's. */
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

/* The host wrote memory the device is about to read: the next dispatch on every queue acquires at
 * system scope, exactly as after a host wait. */
void a_host_wrote(void) { G().host_epoch.fetch_add(1, std::memory_order_acq_rel); }

/* ---- recorded passes: the calls. See "recorded passes" above. */

inline Tape* tape_of(RadTape t) {
    Tape* p = static_cast<Tape*>(t);
    return (p && p->magic == kTapeMagic) ? p : nullptr;
}

/* Tapes destroyed while a play of theirs may still be in a queue. */
std::mutex         g_tape_mu;
std::vector<Tape*> g_tape_retired;

/* Whether every packet the tape's latest play wrote has completed, or its queue is gone: a packet
 * kReserveMargin past it has been launched, which proves it for the reason reserve() gives. */
bool tape_idle(const Tape* t) {
    for (const auto& l : t->last) {
        hsa_queue_t* q = g_queue_by_id[l.first].load(std::memory_order_acquire);
        if (q && hsa_queue_load_read_index_scacquire(q) < l.second + kReserveMargin) return false;
    }
    return true;
}

void tape_free(Tape* t) {
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

/* The argument blocks go to VRAM on the card the tape's queues belong to, where the packet
 * processor reads them as it reads the ring's; the packets are then final but for their header. */
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
    if (keep == 1 && !t->karg_host.empty()) {
        void* ka = nullptr;
        if (hsa_amd_memory_pool_allocate(G().dev[(size_t)ord].vram, t->karg_host.size(), 0, &ka) !=
                HSA_STATUS_SUCCESS ||
            hsa_amd_agents_allow_access(1, &G().cpu, nullptr, ka) != HSA_STATUS_SUCCESS) {
            if (ka) hsa_amd_memory_pool_free(ka);
            device_set_error("the pass cannot be recorded: %zu bytes of kernel arguments could "
                             "not be placed in VRAM", t->karg_host.size());
            tape_free(t);
            return nullptr;
        }
        std::memcpy(ka, t->karg_host.data(), t->karg_host.size());
        _mm_sfence();
        t->karg = static_cast<char*>(ka);
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
    if (!t->karg_host.empty() && !t->karg) {
        device_set_error("cannot play an audit-only tape");
        return RAD_E_STATE;
    }
    ++t_tape_pause;
    int rc = RAD_OK;
    PlayRun run(t);
    for (int i = from; i < to && rc >= 0; ++i) {
        const TapeRec& r = t->rec[(size_t)i];
        if (r.kind == kTapeDispatch) {
            /* A RECORD DIRECTLY BEHIND A DISPATCH ON THE SAME QUEUE RIDES ON THE DISPATCH. The
             * kernel packet carries the barrier bit, so its completion already means everything
             * before it on the queue is done -- which is all the record's own barrier packet says
             * -- and the packet processor decrements its completion signal after its release
             * fence, raised here to the scope the record needs. The separate packet costs the
             * recording queue a packet of its own between two kernels, and a queue waiting on the
             * record sees the signal that much later. */
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
            if (g) {
                /* Announced before the event names it, as the record's own path does: another
                 * queue is about to wait on it. The event's lock is taken with no queue held. */
                run.end();
                record_adopt(ne, g);
                ++i;
            }
            continue;
        }
        /* An event takes its queue's lock itself, and a wait on another queue must not sit behind
         * packets this run has written and not yet announced. */
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

/* PAGEABLE MEMORY IS BOUNCED THROUGH PINNED MEMORY AND WAITED FOR, one bounce buffer at a time,
 * which is what HIP does with it too. It is counted: a pageable copy on the step path is a host
 * wait in the middle of a step, and the counter is how such a caller is found. */
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
            /* The bounce buffer is reused by the next chunk, so the copy out of it has to have run
             * before the host overwrites it. */
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
        /* Row by row through the 1-D path: a pitched copy of pageable memory has no fast form and
         * no caller on the step path. */
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

/* NO PAGEABLE FORM: the kernel reads the range list and the source where they are, so both have
 * to be memory the card reaches, and bouncing them would be a host wait inside the copy. */
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
 * A VRAM ALLOCATION THE DRIVER REFUSES KILLS THE PROCESS IN THIS RUNTIME once a queue has been
 * destroyed. On a refusal ROCr releases its caches and asks again (GpuAgent::Trim), and the release
 * walks every queue the process created on the card. ROCm 7.2's runtime adds a queue to that list
 * (GpuAgent::aql_queues_, amd_gpu_agent.cpp) and never takes a destroyed one off it, so the walk
 * calls into freed memory and the process dies of SIGSEGV inside libhsa-runtime64.so instead of
 * being told no. Startup destroys queues before it allocates anything large -- the two it makes to
 * measure what a queue costs, and the loader's when a load ends -- so every refused device
 * allocation took that path: a card another process had taken memory from crashed startup in
 * hipMalloc rather than refusing its arena. ROCr's develop branch takes the queue off the list on
 * destroy; until a release carries that, every device allocation here is held to what the driver
 * will still give on the card and refused by this layer, naming what the card has and which
 * processes hold it, before the runtime is asked.
 *
 * The figure is the driver's own (AMDKFD_IOC_AVAILABLE_MEMORY), the one its allocation check
 * compares against, and the request is rounded up to 2 MiB so one the driver rounds up is not
 * passed by a hair. It is a check, not a reservation: memory another process takes between it and
 * the allocation still reaches the runtime. */

/* What the driver will still allocate on card `ord`, -1 when it will not say. */
int64_t card_available(int ord) {
    if (ord < 0 || ord >= (int)G().dev.size() || G().kfd < 0) return -1;
    kfd_ioctl_get_available_memory_args a{};
    a.gpu_id = G().dev[(size_t)ord].kfd_id;
    if (!a.gpu_id || ioctl(G().kfd, AMDKFD_IOC_AVAILABLE_MEMORY, &a) < 0) return -1;
    return (int64_t)a.available;
}

/* The processes holding VRAM on card `ord`, largest first, as the driver lists them: host process
 * ids, which inside a container are not this process's own numbering, so this process is named
 * only where the ids agree. */
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
 * the call. A card whose figure the driver will not give is let through, as before this check. */
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
 * HIP's virtual-memory calls build HIP's own default-stream queue on the card the first time a
 * range is backed -- hipMemCreate hands the new pages to HIP's command machinery -- and that queue
 * is a hardware queue this process never submits to. A card runs only so many queues at once; past
 * that it time-slices them, every kernel on the busy ones pays for the switch, and the two lanes of
 * a rank stop overlapping. The same reservations, backings and unmappings through HSA touch no
 * queue at all.
 *
 * A CARD WHOSE POOL REPORTS NO ALLOCATION GRANULE REPORTS A GRANULARITY OF ZERO, and every pool is
 * then sized once and allocated whole (VMem::supported()). A backed range is readable and writable
 * by its reservation's card only: every elastic pool belongs to one rank, and tensor parallel shares
 * through the all-reduce buffers, not these.
 *
 * WHICH CARD A RESERVATION BELONGS TO is recorded when it is made, because the pages a commit maps
 * go on a device and the vtable names none: the device current at RESERVE time is the rank's --
 * every rank reserves its pools with its own card selected -- while the thread that later grows a
 * pool may have visited another card since. A mapping's handle is kept until the unmap, which is
 * what releases it. */
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
    /* The pool's own answer is a floor and not the quantum to use: a map is tens of microseconds
     * whatever it covers. RAD_ALIGN_POOL is what every pool is aligned to already, and taking it
     * makes this backend's granule the host backend's. */
    const int64_t g = (int64_t)gran < (int64_t)RAD_ALIGN_POOL ? (int64_t)RAD_ALIGN_POOL : (int64_t)gran;
    return align_up(g, (int64_t)RAD_ALIGN_POOL);
}

void* a_vmem_reserve(int64_t bytes, int64_t align) {
    if (bytes <= 0) return nullptr;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) dev = 0;
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

/* Backed, and readable and writable by the reservation's card at once: HSA grants access per
 * mapping, so there is nothing for a later protect over a longer run to add. */
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
    /* R3-22: RESERVED_ADDR is never inserted into t_loc, so backing an unmapped range
     * cannot invalidate existing valid entries for other living allocations. */
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
 * EVERY PINNED OR MAPPED HOST ALLOCATION IS ONE ADDRESS RANGE LOCKED AS PIECES OF AT MOST
 * kPinPiece, BECAUSE OF THE IOMMU. Host memory a card reaches is a userptr buffer object, which the
 * kernel DMA-maps for each card as one scatterlist. On an Intel VT-d host in scalable mode (Linux
 * 7.0, two R9700) that mapping fails for any single buffer larger than about 2.5 GiB while smaller
 * ones succeed every time, so a 12 GiB host pool allocated whole was refused at startup far below
 * the driver's pinned-memory cap. Passthrough (iommu=pt) lets the large pin through but takes peer
 * access away, which every multi-rank kernel requires, so the configuration has to work with
 * translation on. A range locked in pieces never asks the IOMMU for more than one piece at a time.
 *
 * The lock is ROCr's, hsa_amd_memory_lock_to_pool: the same userptr pin hipHostMalloc makes, one
 * buffer object per call. hsa_amd_pointer_info does not report what it locked, so locate() answers
 * from this backend's own record of the ranges (g_pinned) and HIP is never asked about them.
 * hipHostRegister would be HIP's record instead, and HIP's record is not what the copies read.
 *
 * The cards still see one range: a locked piece is mapped at its own host address, so the pieces
 * of one reservation are consecutive on the card too. Each piece is checked rather than assumed,
 * because the pools, the mover and the activation arena all hand a kernel base + offset.
 *
 * What hipHostMalloc does besides the pin is kept, so the pages are the same pages: shared
 * anonymous memory, huge pages advised from 2 MiB up, a preferred NUMA node that is the one
 * nearest the current card (ROCr allocates from that node's pool), fine-grained coherence (the
 * pool locked to), and access from every card.
 *
 * kPinPiece is a power of two well under the failing size. The cost is one buffer object per piece
 * per card: a few dozen for the largest pools. */
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

/* MPOL_PREFERRED | MPOL_F_STATIC_NODES, the policy ROCr's own host allocations take. A refusal (a
 * seccomp profile that blocks mbind, a node with no memory) leaves the pages on whichever node
 * first touches them, which is a placement cost and not a correctness one: said once, then the
 * lock goes ahead. */
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

const char* hsa_why(hsa_status_t st) {
    const char* m = nullptr;
    return hsa_status_string(st, &m) == HSA_STATUS_SUCCESS && m ? m : "unknown HSA status";
}

void* pin_pieces(int64_t bytes) {
    const int64_t align = RAD_ALIGN_POOL;
    const int64_t len = align_up(bytes, 4096);
    /* One alignment over, trimmed at both ends, so the range starts on a 2 MiB boundary: a pool's
     * fixed-stride slots inherit its base alignment, and a huge page needs it. */
    void* m = ::mmap(nullptr, (size_t)(len + align), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
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
    if (len >= ((int64_t)2 << 20)) ::madvise(p, (size_t)len, MADV_HUGEPAGE);

    std::vector<hsa_agent_t> cards;
    for (const Dev& d : G().dev) cards.push_back(d.agent);
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
                             "at its host address %p, so the pieces are not one range there",
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
    g_pinned.emplace((uintptr_t)p, len);
    return p;
}

void unpin_pieces(void* p) {
    int64_t len = 0;
    {
        std::lock_guard<std::mutex> lk(g_pinned_mu);
        auto it = g_pinned.find((uintptr_t)p);
        if (it == g_pinned.end()) {
            RAD_WARN("rad_dev_free(%p): not the base of a pinned range this backend handed out; "
                     "leaked rather than unpinned", p);
            return;
        }
        len = it->second;
        g_pinned.erase(it);
    }
    for (int64_t off = 0; off < len; off += kPinPiece) {
        const hsa_status_t st = hsa_amd_memory_unlock((char*)p + off);
        if (st != HSA_STATUS_SUCCESS)
            RAD_WARN("device: unlocking pinned piece %p: %s", (void*)((char*)p + off), hsa_why(st));
    }
    ::munmap(p, (size_t)len);
}

bool pinned_kind(int kind) { return kind == RAD_MEM_HOST_PINNED || kind == RAD_MEM_HOST_MAPPED; }

/* HIP's own allocation calls for device and pageable memory; the pinned kinds are the pieces. The
 * two pinned kinds are one thing here: RAD_MEM_HOST_PINNED (the mover's staging and the O_DIRECT
 * landing zone, spec §5.4) and RAD_MEM_HOST_MAPPED (the zero-copy execution site, row three of
 * spec §5.1's table) are both a locked range every card addresses at its host address. */
void* (*g_hip_alloc)(int64_t, int) = nullptr;
void* a_alloc(int64_t bytes, int kind) {
    if (kind == RAD_MEM_DEVICE && bytes > 0) {
        int dev = -1;
        if (hipGetDevice(&dev) == hipSuccess && !card_has_room(dev, bytes, "rad_dev_alloc"))
            return nullptr;
    }
    if (!pinned_kind(kind)) return g_hip_alloc(bytes, kind);
    if (bytes <= 0) {
        device_set_error("rad_dev_alloc(%lld): size must be positive", (long long)bytes);
        return nullptr;
    }
    return pin_pieces(bytes);
}

/* Behind the forgetting every free needs (locate's cache). */
void (*g_hip_free)(void*, int) = nullptr;
void a_free(void* p, int kind) {
    loc_forget();
    if (!p) return;
    if (pinned_kind(kind)) unpin_pieces(p);
    else if (g_hip_free) g_hip_free(p, kind);
}

/* A pinned range is the same address on both sides, inner addresses included. */
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

/* Whether every fat binary registered in [from, to) holds a code object `device` runs -- the
 * same choice the first launch out of each will make (module_meta), made now, while a library
 * that carries nothing for this card can still be left out of selection rather than chosen and
 * failed at its first launch. */
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

const DeviceBackend* aql_backend_if_present(void) {
    static const DeviceBackend* b = []() -> const DeviceBackend* {
        const DeviceBackend* hip = hip_backend_if_present();
        if (!hip) return nullptr;
        /* THE INTERPOSITION HAS TO BE LIVE, or every plugin launch would hand HIP a queue it has
         * never heard of. It is live when this executable exports hipLaunchKernel, which every
         * target linking the core does (core/CMakeLists.txt); a binary that does not is a build
         * error, reported as one rather than run on the other backend. */
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
 * Defined in the executable, so the dynamic linker binds a plugin's references here ahead of the
 * HIP runtime's. Each forwards anything that is not this backend's to HIP. */

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
        /* HIP hands back no handle for a fat binary with nothing for any visible device. It is
         * still counted, under a record nothing can launch from, because that is exactly the
         * plugin the loader's check exists to catch. */
        if (!h) {
            Module* m = new Module();
            m->bundle = w->binary;
            r.order.push_back(m);
            return h;
        }
        Module*& m = r.by_handle[h];
        /* A handle HIP hands out again after a plugin was unloaded -- the loader closes one it
         * refuses -- names a new fat binary, and the old record's parsed code object points into a
         * mapping that is gone. Records are never freed (see __hipUnregisterFatBinary), so the old
         * one is simply left behind. */
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
    /* The module's executables stay loaded: a process that unloads a kernel plugin is exiting,
     * and the records above are what a late launch would otherwise dereference. */
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

/* A SYNCHRONOUS FILL, ON ONE OF THE BACKEND'S OWN QUEUES FOR THE CURRENT DEVICE. Kernel libraries
 * zero their small private blocks this way -- an all-reduce's flags at init, a split-K GEMM's
 * arrival counters on first use -- and HIP would serve each by building its default stream's
 * queues on that card. Those are hardware queues this process never otherwise uses, and a card
 * whose compute work is spread over more queues than it runs at once time-slices them: every
 * kernel on the busy ones then pays a switch, and the two lanes of a rank stop overlapping.
 *
 * The fill goes on the first queue this process made on the device, which is the compute stream of
 * the rank that owns it, and is waited for there. A device with no queue yet -- a peer's buffers
 * set up before its rank has started -- gets one for the fill and loses it again, so no queue
 * outlives the call. It is never part of a recorded pass: it runs once for a block that is then
 * kept. */
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

/* The two stream APIs a plugin could plausibly reach for that have no meaning on a queue of this
 * backend: events are the engine's, through rad_event_*. Refused by name rather than handed to
 * HIP, which would dereference the queue as one of its own streams. */
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
        /* HIP's own slot holds whatever came before this backend's call, which that call
         * overwrote. */
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
