/* engine_bringup.cpp -- everything the Engine does ONCE, in the order it does it.
 *
 * Devices, the container's metadata, the plugins, declare, the placement plan, the weights, the
 * KV pools, the ranks' threads, the vocabulary. Each numbered section is one component doing its
 * own job; this file is the order they happen in and nothing else.
 *
 * Kept apart from engine.cpp because none of it runs again: the step loop next door is on the hot
 * path and shares nothing with this but the Engine's fields. See engine_priv.h.
 */
#include "engine_priv.h"

#include "build/startup.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "device/directio.h"
#include "format/radfile.h"
#include "format/share.h"
#include "format/table_copy.h"
#include "rad_plugin.h"
#include "mem/vram_budget.h"
#include "runtime/arena.h"
#include "sample/gbnf.h"
#include "sched/scheduler.h"
#include "sched/tree_state.h"
#include "text/tokenizer.h"
#include "../arch/common/rad_mtp_tree_config.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <set>
#include <functional>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <chrono>

namespace rad {

/* ================================================================== the miss report */
/* WHAT DID NOT RESOLVE, in the three shapes the three answers want.
 *
 * A DEVICE miss WITH NO HOST KERNEL EITHER is a hole. The band exists, a request can land in it,
 * and the step that does fails at ISSUE rather than here -- declare completes either way -- so
 * every one gets a line of its own carrying the geometry and the constraint that refused it.
 *
 * A DEVICE miss whose op DID resolve on the host is not a hole at all. The planner puts that op's
 * arithmetic on the host and its weights in the container's mapping, which is how `embed_lookup_q`
 * serves a table too large to copy into VRAM when no device kernel for it is loaded. Reporting
 * that as fatal produces an error nobody believes, and an error nobody believes is how a real one
 * goes unread.
 *
 * A HOST miss removes ONE placement option for ONE op, and for a kernel library written for a GPU
 * it is the ordinary case rather than news: most ops have no host implementation and never will.
 * One line for all of them.
 *
 * The distinction is the whole point of this function. Printed alike, a large model at tp2 opens
 * with thousands of lines of `E` that are all host misses, and a real device hole is invisible
 * among them. */
struct MissReport {
    std::vector<std::string> device;   /* one line each; fatal at issue */
    std::string              host;     /* one line, or empty */
    /* Device bands whose op DOES have a host kernel. Not a hole: the planner runs such an op on
     * the host and pins its weights there (Tier::Mapped), which is the whole of how a table too
     * large to copy into VRAM is served. Reported apart because calling it fatal is a lie an
     * operator has to learn to ignore -- and an error nobody believes is how a real one goes
     * unread. */
    std::string              host_only;

    bool operator==(const MissReport& o) const {
        return device == o.device && host == o.host && host_only == o.host_only;
    }
};

static MissReport miss_report(const Program& p) {
    MissReport r;
    std::vector<std::string>              host_ops;    /* in first-seen order */
    std::map<std::string, std::string>    host_spans;  /* op -> "M <= 64, M in (64, 512]" */

    /* (op, band) pairs the HOST could not serve either. A device miss that is in this set is a
     * hole; one that is not has a host kernel standing behind it. */
    std::set<std::pair<std::string, std::string>> no_host;
    for (const Program::Miss& m : p.misses)
        if (m.domain == RAD_DOMAIN_HOST) no_host.emplace(m.op, m.span);

    std::vector<std::string> host_only_ops;
    for (const Program::Miss& m : p.misses) {
        if (m.domain != RAD_DOMAIN_HOST) {
            if (no_host.count(std::make_pair(m.op, m.span))) r.device.push_back(m.line());
            else if (std::find(host_only_ops.begin(), host_only_ops.end(), m.op) ==
                     host_only_ops.end()) host_only_ops.push_back(m.op);
            continue;
        }
        auto it = host_spans.find(m.op);
        if (it == host_spans.end()) {
            host_ops.push_back(m.op);
            host_spans.emplace(m.op, m.span);
        } else if (it->second.find(m.span) == std::string::npos) {
            it->second += ", " + m.span;
        }
    }
    if (!host_only_ops.empty()) {
        std::string names;
        for (size_t i = 0; i < host_only_ops.size(); ++i)
            names += (i ? ", " : "") + host_only_ops[i];
        r.host_only = fmt("host-only ops: %s resolved a host kernel and no device one, so the "
                          "planner places %s arithmetic on the host and its weights in the "
                          "container's mapping. Not a hole.",
                          names.c_str(), host_only_ops.size() == 1 ? "that op's" : "those ops'");
    }
    if (host_ops.empty()) return r;

    /* THE OP NAMES, ALL OF THEM UNTIL IT GETS SILLY, and not the bands: a reader who wants the
     * bands wants --debug-graph, and the bands are what would make this report thousands of lines
     * long. The names are what a reader can act on -- each one is a reference kernel somebody could
     * write -- and a short list is not enough to act on: whole-model host execution needs EVERY op
     * covered, so the op behind "and 1 more" is the one that stops the run and the one the line
     * does not name. Sixteen is past the point where the answer is a list rather than a shape. */
    std::string names;
    const size_t k_named = 16;
    for (size_t i = 0; i < host_ops.size() && i < k_named; ++i)
        names += (i ? ", " : "") + host_ops[i];
    if (host_ops.size() > k_named)
        names += fmt(", and %zu more", host_ops.size() - k_named);

    r.host = fmt("host-site execution: %zu of %zu declared ops have no host kernel (%s). Their "
                 "weights may still be tiered to host or SSD -- only the arithmetic is pinned to "
                 "the device. --debug-graph lists the bands.",
                 host_ops.size(), p.ops.size() - 1, names.c_str());
    return r;
}

/* ================================================================== 1. devices */
/* A HIP device is not necessarily a discrete GPU. A host with an integrated GPU reports it as a
 * device of its own, claiming a slice of system RAM as VRAM, and a --tp that lands a rank on one
 * runs slowly and wrongly with a failure that looks like a kernel bug. So the engine uses the
 * discrete cards when there are any -- integrated graphics only when it is all there is, which is
 * an APU machine -- and of those the ones sharing the first one's arch, and prints only what that
 * choice DID, rather than reprinting the backend's inventory. Which devices the backend shows at
 * all is the runtime's to say: ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES pick them. */
static int usable_devices(std::vector<int>* out, std::string* inventory, std::string* note) {
    const int n = rad_dev_count();
    if (n <= 0) return RAD_E_DEVICE;

    struct D { int i; std::string arch; bool igpu; };
    std::vector<D> all;
    bool any_discrete = false;
    std::string inv;
    for (int i = 0; i < n; ++i) {
        RadDeviceProps p{};
        if (rad_dev_props(i, &p) < 0) continue;
        const bool igpu = dev_integrated(i);
        inv += fmt("%s%d:%s%s", i ? " " : "", i, p.arch, igpu ? "(integrated)" : "");
        all.push_back({ i, p.arch, igpu });
        any_discrete |= !igpu;
    }

    std::string want;              /* the arch of the first device taken -- the ranks must all match */
    std::string use, skip;
    for (const D& d : all) {
        const bool take = (!any_discrete || !d.igpu) && (want.empty() || want == d.arch);
        if (take && want.empty()) want = d.arch;
        std::string& into = take ? use : skip;
        into += fmt("%s%d:%s", into.empty() ? "" : " ", d.i, d.arch.c_str());
        if (take) out->push_back(d.i);
    }
    *inventory = fmt("%s backend, %d device(s) [%s]", device_backend_name(), n, inv.c_str());

    /* THE BACKEND ALREADY PRINTED THAT INVENTORY (core/device/dispatch.cpp), so repeating it here
     * would only cost a reader two lines that say the same thing in different capitalisation.
     * What is the engine's to say is which of them it will USE -- and that is news only when the
     * arch filter dropped one, so on a uniform machine this stays empty and nothing is
     * printed. */
    *note = skip.empty()
          ? std::string()
          : fmt("using %zu of %d [%s], ignoring [%s]: integrated graphics beside a discrete card, "
                "or a different architecture from the first card used, and a rank placed on one "
                "runs slowly and wrongly. ROCR_VISIBLE_DEVICES chooses the devices explicitly",
                out->size(), n, use.c_str(), skip.c_str());
    return out->empty() ? RAD_E_DEVICE : RAD_OK;
}

/* WHY A POOL'S MEMORY WAS REFUSED, from what the machine holds now. The refusal carries the
 * driver's own answer; these are the facts that explain most of them.
 *
 * A card refusing its weights pool after its budget was measured had that memory taken in
 * between, and another process on the card is the usual taker -- the budget is measured once,
 * before any rank allocates. A host pool is PINNED memory, and the amdgpu driver caps what the
 * whole system may pin (ttm.pages_limit, half of RAM unless raised); every rank pins its own
 * --host-pool-mib, so a second rank asks for the same amount again. */
static void report_pool_refusal(int rank, int device, const Pools& pools, const Config& cfg) {
    bool card = false, host = false;
    for (const Shortfall& f : pools.shortfalls()) (f.pool == "host_pool" ? host : card) = true;
    if (card) {
        RadDeviceProps p{};
        if (rad_dev_props(device, &p) == RAD_OK)
            RAD_ERR("card %d has %s of %s free now. Its budget was measured before any rank "
                    "allocated, so memory gone since then is another process's; rocm-smi "
                    "--showpids names it", device, humanb(p.vram_free).c_str(),
                    humanb(p.vram_bytes).c_str());
    }
    if (host) {
        int64_t pinned = 0, mapped = 0;
        rad_dev_alloc_totals(RAD_MEM_HOST_PINNED, &pinned, nullptr);
        rad_dev_alloc_totals(RAD_MEM_HOST_MAPPED, &mapped, nullptr);
        int64_t cap = -1, avail = -1;
        {
            std::ifstream f("/sys/module/ttm/parameters/pages_limit");
            long long pages = 0;
            if (f >> pages && pages > 0) cap = (int64_t)pages * (int64_t)sysconf(_SC_PAGESIZE);
        }
        {
            std::ifstream f("/proc/meminfo");
            std::string line;
            while (std::getline(f, line)) {
                long long kb = 0;
                if (std::sscanf(line.c_str(), "MemAvailable: %lld kB", &kb) == 1) {
                    avail = (int64_t)kb * 1024;
                    break;
                }
            }
        }
        RAD_ERR("host: --host-pool-mib %lld is pinned once per rank, so %d rank(s) pin %s; this "
                "process had taken %s of pinned host memory before rank %d's pool. The amdgpu "
                "driver caps pinned memory system-wide at %s (ttm.pages_limit, half of RAM unless "
                "raised), and %s of RAM is available. Lower --host-pool-mib, or raise the cap",
                (long long)cfg.host_pool_mib, cfg.tp,
                humanb((int64_t)cfg.tp * cfg.host_pool_mib * 1024 * 1024).c_str(),
                humanb(pinned + mapped).c_str(), rank,
                cap > 0 ? humanb(cap).c_str() : "an unreadable limit",
                avail >= 0 ? humanb(avail).c_str() : "an unknown amount");
    }
}

/* ------------------------------------------------------------------ locked host memory */
/* WHAT EVERYTHING ELSE ON THE HOST STILL NEEDS beside a table locked whole: the process's own
 * unpooled allocations, the load's staging windows and the page cache the container's hot pages
 * live in. A copy that spends the last of it trades a faster gather for the kernel reclaiming
 * the files this engine reads, and then for the OOM killer. */
static const int64_t kLockMargin = (int64_t)4 << 30;

/* A cgroup limit and what its members use that reclaim cannot give back: memory.current less its
 * file pages (cgroup v2) or usage_in_bytes less its cache (v1), since page cache yields to a
 * locked allocation and anonymous memory does not. "max" and v1's near-INT64_MAX sentinel are no
 * limit. True with `room` set when `dir` holds one. */
static bool cgroup_room(const std::string& dir, bool v2, int64_t* room) {
    auto read_ll = [](const std::string& f, long long* v) {
        std::ifstream in(f);
        std::string t;
        if (!(in >> t)) return false;
        char* e = nullptr;
        *v = std::strtoll(t.c_str(), &e, 10);
        return e && *e == '\0';
    };
    auto stat_ll = [](const std::string& f, const char* key, long long* v) {
        std::ifstream in(f);
        std::string k;
        long long x = 0;
        while (in >> k >> x) if (k == key) { *v = x; return true; }
        return false;
    };
    long long lim = 0, used = 0, file = 0;
    if (v2) {
        if (!read_ll(dir + "/memory.max", &lim)) return false;           /* "max" fails to parse */
        if (!read_ll(dir + "/memory.current", &used)) return false;
        (void)stat_ll(dir + "/memory.stat", "file", &file);
    } else {
        if (!read_ll(dir + "/memory.limit_in_bytes", &lim) || lim >= ((long long)1 << 62))
            return false;
        if (!read_ll(dir + "/memory.usage_in_bytes", &used)) return false;
        if (!stat_ll(dir + "/memory.stat", "total_cache", &file))
            (void)stat_ll(dir + "/memory.stat", "cache", &file);
    }
    const long long hard = used > file ? used - file : 0;
    *room = lim > hard ? (int64_t)(lim - hard) : 0;
    return true;
}

/* WHETHER `need` MORE BYTES CAN BE HELD IN HOST MEMORY, after `still` more that this process will
 * allocate before the copy is made, or why not. Each limit is one a copy would otherwise meet part
 * way through: the memory the kernel reports available, every memory cgroup from the process's own
 * up to the root it can see, which is the limit Docker's --memory sets and the kernel enforces by
 * killing the process -- and, for a copy mlock()ed for the host gather (`mlocks`), RLIMIT_MEMLOCK
 * unless the process holds CAP_IPC_LOCK. Pinned memory a card reads is the driver's userptr
 * mapping, the host pools' kind, and RLIMIT_MEMLOCK does not count it. A limit that cannot be read
 * is not one; a limit that can is never guessed at. */
static bool lockable_room(int64_t need, int64_t still, bool mlocks, std::string* why) {
    long long locked_kb = 0;
    unsigned long long caps = 0;
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line)) {
            (void)std::sscanf(line.c_str(), "VmLck: %lld", &locked_kb);
            (void)std::sscanf(line.c_str(), "CapEff: %llx", &caps);
        }
    }
    struct rlimit rl {};
    if (mlocks && !((caps >> 14) & 1) && ::getrlimit(RLIMIT_MEMLOCK, &rl) == 0 &&
        rl.rlim_cur != RLIM_INFINITY && (int64_t)rl.rlim_cur < locked_kb * 1024 + need) {
        *why = fmt("RLIMIT_MEMLOCK allows %s locked and the copy locks %s (Docker: --ulimit "
                   "memlock=-1; systemd: LimitMEMLOCK=infinity)", humanb((int64_t)rl.rlim_cur).c_str(),
                   humanb(need).c_str());
        return false;
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        long long kb = -1;
        while (std::getline(f, line))
            if (std::sscanf(line.c_str(), "MemAvailable: %lld kB", &kb) == 1) break;
        if (kb >= 0 && kb * 1024 - still - kLockMargin < need) {
            *why = fmt("%s of memory is available, %s more is still to be allocated (the pinned "
                       "host pools and the prefix cache's host tier), and the copy wants %s with %s "
                       "to spare", humanb(kb * 1024).c_str(), humanb(still).c_str(),
                       humanb(need).c_str(), humanb(kLockMargin).c_str());
            return false;
        }
    }
    /* /proc/self/cgroup: "0::/path" on v2, "N:memory:/path" (or a comma list naming memory) on v1.
     * Each level up to the root of the hierarchy this process can see is a limit of its own. */
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    while (std::getline(f, line)) {
        const size_t a = line.find(':'), b = a == std::string::npos ? a : line.find(':', a + 1);
        if (b == std::string::npos) continue;
        const std::string ctrl = line.substr(a + 1, b - a - 1);
        std::string path = line.substr(b + 1);
        const bool v2 = ctrl.empty();
        bool memory = v2;
        for (size_t i = 0; !memory && i < ctrl.size();) {
            const size_t j = std::min(ctrl.find(',', i), ctrl.size());
            memory = ctrl.compare(i, j - i, "memory") == 0;
            i = j + 1;
        }
        if (!memory) continue;
        const std::string root = v2 ? "/sys/fs/cgroup" : "/sys/fs/cgroup/memory";
        for (;;) {
            const std::string dir = root + (path == "/" ? std::string() : path);
            int64_t room = 0;
            if (cgroup_room(dir, v2, &room) && room - still - kLockMargin < need) {
                *why = fmt("the memory cgroup at %s leaves %s, %s more is still to be allocated, "
                           "and the copy wants %s with %s to spare", dir.c_str(),
                           humanb(room).c_str(), humanb(still).c_str(), humanb(need).c_str(),
                           humanb(kLockMargin).c_str());
                return false;
            }
            if (path.empty() || path == "/") break;
            const size_t up = path.find_last_of('/');
            path = up == 0 || up == std::string::npos ? "/" : path.substr(0, up);
        }
    }
    return true;
}

/* --ngram-placement ram: the copy is table_copy_make's (core/format/table_copy.h), or for a card,
 * table_copy_fill's into pinned memory every card maps at its host address (rad_dev_alloc
 * RAD_MEM_HOST_MAPPED, locked in pieces). This keeps one a container address, shared by every rank,
 * and frees them after the ranks in ~Engine. */
int Engine::copy_table(const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                       std::string* why) {
    for (const auto& c : table_copy_of_)
        if (c.first == src) { *out = c.second; return RAD_OK; }
    const auto t0 = std::chrono::steady_clock::now();
    void* base = nullptr;
    if (card) {
        const char* shm_env = std::getenv("RADIANCE_PLE_SHM");
        const bool enable_shm = !shm_env || (std::strcmp(shm_env, "0") != 0 &&
                                             std::strcmp(shm_env, "off") != 0 &&
                                             std::strcmp(shm_env, "false") != 0);
        if (enable_shm) {
            uint64_t model_hash = 14695981039346656037ULL;
            for (char c : cfg_.model) {
                model_hash ^= (uint8_t)c;
                model_hash *= 1099511628211ULL;
            }
            char hash_str[32];
            std::snprintf(hash_str, sizeof(hash_str), "%016llx", (unsigned long long)model_hash);
            DIR* dir = opendir("/dev/shm");
            if (dir) {
                struct dirent* de;
                while ((de = readdir(dir)) != nullptr) {
                    std::string fn = de->d_name;
                    if (fn.rfind("rad_shm_", 0) == 0 && fn.find(hash_str) != std::string::npos &&
                        fn.size() > 6 && fn.rfind(".ready") == fn.size() - 6) {
                        std::string bin_name = fn.substr(0, fn.size() - 6);
                        std::string bin_path = "/dev/shm/" + bin_name;
                        struct stat st_bin{};
                        if (stat(bin_path.c_str(), &st_bin) == 0 && st_bin.st_size == (off_t)n) {
                            int fd = open(bin_path.c_str(), O_RDWR);
                            if (fd >= 0) {
                                void* p = mmap(nullptr, (size_t)n, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                                close(fd);
                                if (p != MAP_FAILED) {
                                    madvise(p, (size_t)n, MADV_HUGEPAGE);
                                    if (rad_dev_host_register(p, n, 0) == RAD_OK) {
                                        base = p;
                                        shm_copies_.push_back({ p, (size_t)n });
                                        RAD_INFO("placement: n-gram table attached from persistent shm cache in %.2f s (instant restart)",
                                                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                                        break;
                                    }
                                    munmap(p, (size_t)n);
                                }
                            }
                        }
                    }
                }
                closedir(dir);
            }
        }
        if (!base) {
            base = rad_dev_alloc(n, RAD_MEM_HOST_MAPPED);
            if (!base) {
                *why = fmt("pinning %s: %s", humanb(n).c_str(), rad_dev_last_error());
                return RAD_E_NOMEM;
            }
            const int rc = table_copy_fill(*file_, src, n, foff, static_cast<uint8_t*>(base), why);
            if (rc != RAD_OK) {
                rad_dev_free(base, RAD_MEM_HOST_MAPPED);
                return rc;
            }
            card_copies_.push_back(base);
        }
    } else {
        TableCopy tc;
        const int rc = table_copy_make(*file_, src, n, foff, &tc, why);
        if (rc != RAD_OK) return rc;
        table_copies_.push_back(tc);
        base = tc.base;
    }
    RAD_INFO("placement: n-gram table copied into %s, %s in %.1f s",
             card ? "pinned memory the cards read" : "locked memory", humanb(n).c_str(),
             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    table_copy_of_.push_back({ src, base });
    *out = base;
    return RAD_OK;
}

/* THE GATHER MOVES TO THE HOST WITH THE TABLE, when --ngram-placement auto chose a pinned copy for
 * a card and it could not be made. Every rank's n-gram weights take the host site, on the
 * WeightInfo Ctx::init reads to choose the op's domain and on the plan beside it, so the gather and
 * its scale move together; a copy some rank already published is host memory, which the host
 * gather reads as it reads RAM. False where no host kernel gathers the table. */
bool Engine::ngram_gather_to_host() {
    for (auto& r : ranks_) {
        WeightFootprint f;
        weight_footprint(r->program, &f, TablePlacement::Vram, TablePlacement::Disk);
        if (!f.ngram_host) return false;
    }
    for (auto& r : ranks_)
        for (size_t i = 1; i < r->program.weights.size(); ++i) {
            if (!is_ngram_table(r->program, (rad_weight)i)) continue;
            r->program.weights[i].site = Site::Host;
            if (i < r->plan.weight.size()) r->plan.weight[i].site = Site::Host;
        }
    return true;
}

/* WHERE THE TWO LOOKUP TABLES GO (--embedding-placement, --ngram-placement), decided once for every
 * rank before the card is split, and written into Config for the planner and the load phase.
 *
 * THE EMBEDDING: Auto keeps it on the card -- where every other weight a step reads lives --
 * unless that leaves the split unable to hold the fixed weights, or leaves a derived KV pool
 * unable to hold one --max-model-len sequence (`kv_floor`); then the pinned host pool takes it if
 * --host-pool-mib can, the card's lookup reading it over the link in place. Disk is refused: the
 * lookup writes the activations a card reads, which a host kernel reading the container cannot
 * reach -- an architecture declares a buffer's domain, and the token embedding's is the device.
 *
 * THE N-GRAM TABLE: Vram where a device kernel gathers it and the card holds it; Ram as a copy,
 * pinned for that kernel to read over the link or locked for the host gather; Disk in the
 * container, read by the host gather. Each tier is refused where no kernel reads it there. Auto
 * takes the first that holds the table, in that order: the card's test is the embedding's, and
 * RAM's is lockable_room's -- the memory and every cgroup after the pools still to be allocated,
 * and the memlock limit for a locked copy. */
int Engine::resolve_table_placement(const std::vector<VramFacts>& facts,
                                    const std::vector<int64_t>& emb_bytes, int64_t kv_floor,
                                    int64_t ngram_bytes, bool ngram_device, bool ngram_host) {
    const int64_t MiB = (int64_t)1 << 20;
    int64_t emb_max = 0;
    for (int64_t b : emb_bytes) emb_max = std::max(emb_max, b);

    TablePlacement emb = cfg_.embedding_placement;
    std::string emb_why;
    if (emb == TablePlacement::Disk) {
        RAD_ERR("--embedding-placement disk: the token embedding's lookup writes activations the "
                "card reads, and a host kernel reading the container cannot write them -- the "
                "architecture declares that buffer on the device. Use vram, ram or auto.");
        return RAD_E_INVAL;
    }
    if (emb_max == 0) {
        if (emb == TablePlacement::Ram) {
            RAD_ERR("--embedding-placement ram: this model has no token embedding table that only "
                    "embed_lookup reads. An output head tied to the table reads it too, and would "
                    "stream it over the link every step.");
            return RAD_E_INVAL;
        }
        emb = TablePlacement::Vram;
        emb_why = "no table only embed_lookup reads";
    } else if (emb == TablePlacement::Auto) {
        VramBudget tb;
        const int st = vram_budget_trial(cfg_, facts, &tb);
        const bool holds = st == RAD_OK && (!tb.derived_kv || tb.kv >= kv_floor);
        const bool host  = cfg_.host_pool_mib * MiB >= emb_max;
        if (holds) {
            emb = TablePlacement::Vram;
            emb_why = "the card holds it beside the other weights and a KV pool for one "
                      "--max-model-len sequence";
        } else if (host) {
            emb = TablePlacement::Ram;
            emb_why = st != RAD_OK
                ? "the card cannot hold the other weights with it, and the pinned host pool can"
                : fmt("with it on the card the KV pool is %s, short of the %s one --max-model-len "
                      "sequence needs, and the pinned host pool can hold it",
                      humanb(tb.kv).c_str(), humanb(kv_floor).c_str());
        } else {
            emb = TablePlacement::Vram;
            emb_why = fmt("the card is short of room for it, and --host-pool-mib %lld cannot hold "
                          "%s either", (long long)cfg_.host_pool_mib, humanb(emb_max).c_str());
        }
    } else {
        emb_why = "as asked";
    }
    cfg_.embedding_resolved = emb;

    TablePlacement ng = cfg_.ngram_placement;
    std::string ng_why;
    /* THE FASTEST TIER WITH THE ROOM, the card first: the table on the card, every rank holding
     * all of it (a lookup is not sharded), with the fixed weights and a KV pool for one
     * --max-model-len sequence beside it, as the embedding's test is; then a copy in RAM -- pinned
     * for a device kernel to read over the link, or locked for the host gather; then the
     * container, which only the host gather reads. Each is open only where a kernel reads it. */
    const int64_t still = (int64_t)cfg_.tp * cfg_.host_pool_mib * MiB +
                          cfg_.prefix_cache_host_mib * MiB;
    const int64_t need = align_up(ngram_bytes, (int64_t)2 << 20);
    const char* copy = ngram_device ? "a pinned copy the card reads" : "a locked copy";
    auto card_holds = [&](std::string* why) {
        std::vector<VramFacts> with = facts;
        int64_t fixed = 0;
        for (VramFacts& f : with) {
            f.statics += ngram_bytes;
            if (emb == TablePlacement::Ram) f.statics -= emb_max;
            fixed = std::max(fixed, f.statics);
        }
        /* `fixed` and not the trial's own statics: the split clamps those to what the card can give
         * and serves on, streaming every expert, so the trial alone cannot say they do not fit. */
        VramBudget tb;
        const int st = vram_budget_trial(cfg_, with, &tb);
        const bool over = st == RAD_OK && fixed >= tb.claimable;
        if (st == RAD_OK && !over && (!tb.derived_kv || tb.kv >= kv_floor)) return true;
        if (st != RAD_OK)
            *why = "the card cannot hold the other weights with it";
        else if (over)
            *why = fmt("with it the fixed weights want %s of a card that has %s to give",
                       humanb(fixed).c_str(), humanb(tb.claimable).c_str());
        else
            *why = fmt("with it on the card the KV pool is %s, short of the %s one "
                       "--max-model-len sequence needs", humanb(tb.kv).c_str(),
                       humanb(kv_floor).c_str());
        return false;
    };
    if (ngram_bytes == 0) {
        if (ng != TablePlacement::Auto)
            RAD_WARN("--ngram-placement %s: this model has no n-gram table, so there is nothing to "
                     "place", table_placement_name(ng));
        ng = TablePlacement::Disk;
    } else if (ng == TablePlacement::Vram) {
        std::string why;
        if (!ngram_device) {
            RAD_ERR("--ngram-placement vram: no loaded kernel plugin reads an n-gram table on a "
                    "card (embed_lookup_q has host kernels only), so it is read from host memory "
                    "or the container: ram or disk");
            return RAD_E_INVAL;
        }
        if (!card_holds(&why)) {
            RAD_ERR("--ngram-placement vram: the n-gram table (%s, all of it on every card) does "
                    "not fit: %s. Use ram, disk or auto.", humanb(ngram_bytes).c_str(),
                    why.c_str());
            return RAD_E_NOMEM;
        }
        ng_why = "as asked";
    } else if (ng == TablePlacement::Disk) {
        if (!ngram_host) {
            RAD_ERR("--ngram-placement disk: no loaded kernel plugin gathers an n-gram table on "
                    "the host, which is what reads it from the container. Use vram, ram or auto.");
            return RAD_E_INVAL;
        }
        ng_why = "as asked";
    } else {
        std::string card_why, ram_why;
        const bool card = ng == TablePlacement::Auto && ngram_device && card_holds(&card_why);
        const bool room = !card && lockable_room(need, still, !ngram_device, &ram_why);
        if (ng == TablePlacement::Ram) {
            if (!room) {
                RAD_ERR("--ngram-placement ram: the n-gram table (%s) cannot be held in memory as "
                        "%s: %s", humanb(ngram_bytes).c_str(), copy, ram_why.c_str());
                return RAD_E_NOMEM;
            }
            ng_why = "as asked";
        } else if (card) {
            ng = TablePlacement::Vram;
            ng_why = "the card holds it beside the other weights and a KV pool for one "
                     "--max-model-len sequence";
        } else if (room) {
            ng = TablePlacement::Ram;
            ng_why = fmt("%sthe memory%s and the cgroups allow %s",
                         ngram_device ? (card_why + "; ").c_str() : "",
                         ngram_device ? "" : ", the memlock limit", copy);
        } else if (ngram_host) {
            ng = TablePlacement::Disk;
            ng_why = fmt("%s%s does not fit: %s", ngram_device ? (card_why + "; ").c_str() : "",
                         copy, ram_why.c_str());
        } else {
            RAD_ERR("--ngram-placement auto: the n-gram table (%s) has nowhere to go: %s; %s does "
                    "not fit (%s); and no loaded kernel plugin gathers it on the host, which is "
                    "what reads it from the container", humanb(ngram_bytes).c_str(),
                    card_why.c_str(), copy, ram_why.c_str());
            return RAD_E_NOMEM;
        }
    }
    cfg_.ngram_resolved = ng;

    if (emb_max > 0)
        RAD_INFO("placement: token embedding (%s a rank) %s%s: %s", humanb(emb_max).c_str(),
                 table_placement_name(emb),
                 cfg_.embedding_placement == TablePlacement::Auto ? " (auto)" : "", emb_why.c_str());
    if (ngram_bytes > 0)
        RAD_INFO("placement: n-gram table (%s) %s%s: %s", humanb(ngram_bytes).c_str(),
                 table_placement_name(ng),
                 cfg_.ngram_placement == TablePlacement::Auto ? " (auto)" : "", ng_why.c_str());
    return RAD_OK;
}

/* ================================================================== 2. metadata */
/* The model's metadata ONLY -- architecture id, dimensions, quantisation, vocab. Not the weights:
 * which weights exist is the architecture plugin's answer, not the container's, and reading them
 * before declare would be reading them before we know what they are for. */
namespace {

/* A CONTAINER IS A FILE THAT SAYS SO: its first four bytes are the magic. Anything else --
 * a checkpoint directory, a .safetensors, an index, a .gguf -- is a checkpoint. A .rad that is not
 * one is still opened as a container, so its own reader names what is wrong with it. */
bool is_container(const std::string& path) {
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".rad") == 0) return true;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    const bool got = std::fread(&magic, 4, 1, f) == 1;
    std::fclose(f);
    return got && magic == RAD_MAGIC;
}

}  /* namespace */

WeightSourceFn Engine::sources() const {
    if (file_) return rad_container_sources(*file_);
    return ckpt_src_ ? ckpt_src_->fn() : WeightSourceFn{};
}

int Engine::read_metadata() {
    if (!is_container(cfg_.model)) return read_checkpoint_metadata();
    file_ = std::make_unique<RadFile>();
    RAD_TRY(file_->open(cfg_.model.c_str()));

    const RadFileHeader& h = file_->header();
    meta_ = RadModelMeta{};
    meta_.arch_id = file_->str(h.arch_id);
    meta_.name    = file_->str(h.model_name);
    meta_.quant   = file_->str(h.quant);

    auto geti = [&](const char* k, int64_t d) { int64_t v = d; file_->get_i(k, &v); return v; };
    auto getf = [&](const char* k, double d) { double v = d; file_->get_f(k, &v); return (float)v; };

    /* THE KEYS ARE RadModelMeta'S OWN FIELD NAMES, because that is what rad-convert writes and
     * what rad-kbench and rad-tune read. Reading GGUF's spellings instead -- `block_count`,
     * `embedding_length`, `attention.head_count` -- yields zero for every dimension, since no
     * writer in this project produces them; declare then refuses with "invalid argument" while
     * rad-info prints the correct shape off the same file.
     *
     * The GGUF names are NOT accepted as a fallback. A container carries these because
     * rad-convert normalises whatever it read -- safetensors or GGUF -- into RadModelMeta before
     * writing, so a file that lacks them is a file no version of this project wrote, and reading
     * it as if the absent keys were zeros turns a bad container into a wrong model instead of a
     * refusal. */
    meta_.n_layers        = geti("n_layers", 0);
    meta_.n_embd          = geti("n_embd", 0);
    meta_.n_head          = geti("n_head", 0);
    meta_.n_head_kv       = geti("n_head_kv", 0);
    /* Read head_dim; do NOT derive it. A model may declare a head_dim that is not n_embd/n_head,
     * and a derived one then makes every downstream shape silently wrong
     * (docs/ARCHITECTURES.md). */
    meta_.head_dim        = geti("head_dim", 0);
    meta_.n_ff            = geti("n_ff", 0);
    meta_.n_vocab         = geti("n_vocab", 0);
    meta_.n_ctx_train     = geti("n_ctx_train", 0);
    meta_.n_expert        = geti("n_expert", 0);
    meta_.n_expert_used   = geti("n_expert_used", 0);
    meta_.n_expert_shared = geti("n_expert_shared", 0);
    meta_.rms_eps         = getf("rms_eps", 1e-6);
    meta_.rope_theta      = getf("rope_theta", 10000.0);
    meta_.rope_scale      = getf("rope_scale", 1.0);

    /* A container that says nothing about its own shape is refused HERE, by name, rather than at
     * declare as an "invalid argument" from a plugin that was handed zeros. */
    if (meta_.n_layers <= 0 || meta_.n_embd <= 0 || meta_.n_head <= 0 || meta_.n_vocab <= 0) {
        RAD_ERR("%s carries no model dimensions (n_layers=%lld n_embd=%lld n_head=%lld "
                "n_vocab=%lld). A .rad written by rad-convert always does; rad-info prints what "
                "this one holds.", cfg_.model.c_str(), (long long)meta_.n_layers,
                (long long)meta_.n_embd, (long long)meta_.n_head, (long long)meta_.n_vocab);
        return RAD_E_FORMAT;
    }

    /* Everything the struct does not name, verbatim, for the plugin to read. The vectors own the
     * storage because RadModelMeta holds bare pointers. */
    meta_kv_k_.clear(); meta_kv_v_.clear();
    for (int64_t i = 0; i < file_->meta_count(); ++i) {
        const RadFileKV& kv = file_->meta_at(i);
        meta_kv_k_.push_back(file_->str(kv.key));
        switch (kv.type) {
            case RAD_P_INT:  meta_kv_v_.push_back(fmt("%lld", (long long)kv.v.i)); break;
            case RAD_KV_F64: meta_kv_v_.push_back(fmt("%.17g", kv.v.f)); break;
            default:         meta_kv_v_.push_back(file_->str(kv.v.s)); break;
        }
    }
    meta_kv_kp_.clear(); meta_kv_vp_.clear();
    for (auto& s : meta_kv_k_) meta_kv_kp_.push_back(s.c_str());
    for (auto& s : meta_kv_v_) meta_kv_vp_.push_back(s.c_str());
    meta_.n_kv    = (int)meta_kv_kp_.size();
    meta_.kv_key  = meta_kv_kp_.data();
    meta_.kv_val  = meta_kv_vp_.data();

    RAD_INFO("model: %s  arch=%s  quant=%s  %lld layers  %lld ctx",
             meta_.name && *meta_.name ? meta_.name : "(unnamed)",
             meta_.arch_id, meta_.quant && *meta_.quant ? meta_.quant : "bf16",
             (long long)meta_.n_layers, (long long)meta_.n_ctx_train);
    return RAD_OK;
}

/* A CHECKPOINT SERVED DIRECTLY (spec §4.3): its config.json -- or the GGUF header -- is the
 * metadata, read the way rad-convert reads it, and its tensors are the weights. Only an encoding
 * served as it stands is answered to declare (CheckpointSources), so a checkpoint whose served
 * form needs quantisation is refused by name after declare, with the container as the remedy. */
int Engine::read_checkpoint_metadata() {
    ckpt_ = std::make_unique<Checkpoint>();
    RAD_TRY(ckpt_->open(cfg_.model));
    ckpt_src_ = std::make_unique<CheckpointSources>(*ckpt_);
    const MetaOwn& mo = ckpt_->meta();
    meta_ = mo.m;
    if (!meta_.name || !*meta_.name) meta_.name = meta_.arch_id;
    if (meta_.n_layers <= 0 || meta_.n_embd <= 0 || meta_.n_head <= 0 || meta_.n_vocab <= 0) {
        RAD_ERR("%s: the checkpoint's configuration states no model dimensions (n_layers=%lld "
                "n_embd=%lld n_head=%lld n_vocab=%lld)", cfg_.model.c_str(),
                (long long)meta_.n_layers, (long long)meta_.n_embd, (long long)meta_.n_head,
                (long long)meta_.n_vocab);
        return RAD_E_FORMAT;
    }
    meta_kv_k_ = mo.keys;
    meta_kv_v_ = mo.vals;
    meta_kv_kp_.clear(); meta_kv_vp_.clear();
    for (auto& k : meta_kv_k_) meta_kv_kp_.push_back(k.c_str());
    for (auto& v : meta_kv_v_) meta_kv_vp_.push_back(v.c_str());
    meta_.n_kv    = (int)meta_kv_kp_.size();
    meta_.kv_key  = meta_kv_kp_.data();
    meta_.kv_val  = meta_kv_vp_.data();
    RAD_INFO("model: %s  arch=%s  quant=%s  %lld layers  %lld ctx  (checkpoint, %zu tensors, "
             "served as it stands)", meta_.name, meta_.arch_id,
             meta_.quant && *meta_.quant ? meta_.quant : "bf16", (long long)meta_.n_layers,
             (long long)meta_.n_ctx_train, ckpt_->tensors().size());
    return RAD_OK;
}

/* ================================================================== 3 + 4. plugins, declare */
int Engine::load_plugins() {
    int s = rad_tools_load_plugins(cfg_.radiance_home, cfg_.kernel_hierarchy,
                                   meta_.arch_id, meta_.quant ? meta_.quant : "", &plugins_);
    if (s < 0) {
        /* rad_tools_load_plugins has said why -- a plugin refused, or no architecture plugin for
         * this model -- each by name. What it does not know is that this was the engine's start. */
        RAD_ERR("the plugins under %s did not load (above)", cfg_.radiance_home.c_str());
        return s;
    }

    const Plugin* ap = rad_tools_registry().plugin(plugins_.arch_plugin);
    arch_step_ = ap ? ap->arch_step : nullptr;
    return RAD_OK;
}

RadBuildCtx Engine::build_ctx(int rank, int64_t max_tok) const {
    RadBuildCtx bc{};
    bc.rank       = rank;
    bc.world_size = cfg_.tp;
    bc.max_tok    = max_tok;
    bc.max_seqs   = cfg_.max_seqs;
    bc.max_ctx    = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
    bc.max_spec   = cfg_.n_spec;
    bc.tp_wire_lossy = cfg_.tp_wire_exact ? 0 : 1;
    /* One answer to "is the cache fp8", for every architecture. The engine always states it;
     * RAD_DT_INVALID stays meaningful in the ABI for an out-of-tree caller that leaves the field
     * zero, and means the plugin's own default. */
    bc.kv_dtype = cfg_.kv_cache_dtype == "fp8" ? RAD_F8E4M3 : RAD_BF16;
    bc.tp_wire_min_bytes = cfg_.tp_wire_min_kb * 1024;
    bc.scope      = "";
    bc.max_enc_patches = cfg_.mm_max_patches > 0 ? cfg_.mm_max_patches : 0;
    /* A row at every token of the step and the sampler's row a sequence before them
     * (core/kld.h); the KL mode has no speculation, so a sequence samples one. */
    bc.max_out_rows = cfg_.kld() ? max_tok + cfg_.max_seqs : 0;
    return bc;
}

int Engine::declare() {
    /* Declare runs ONCE PER RANK, with rank and world_size in RadBuildCtx, and the plugin declares
     * dimensions ALREADY DIVIDED. The core does not reason about sharding propagation and does not
     * insert collectives -- a sharding inference pass is a compiler pass whose failure mode is a
     * model that produces fluent wrong text on a rank count nobody tested (spec §9).
     *
     * EVERY RANK DECLARES AT THE SAME TIME, ON ITS OWN THREAD. That is a requirement and not a
     * speed-up. Kernel init() runs from here (registry.h says so), and a collective's init is a
     * RENDEZVOUS: libr4d's all-reduce publishes this rank's peer scratch pointer and then WAITS
     * for every other rank to publish before it can enable peer access. A serial loop never gets
     * a second rank there, so the first one times out after 60 seconds and --tp 2 fails startup
     * with `ar_oneshot_2rank_exact ... called in the wrong phase`. Registry::select() and
     * init_instance() are already behind one mutex for exactly this, and the arch plugin's
     * per-rank state is indexed by ctx->rank.
     *
     * The device is bound per thread because the backend's current device is thread-local and a
     * collective's init ALLOCATES on it -- unbound, both ranks would hand each other pointers into
     * card 0 and the reduce would read its own scratch twice. */
    /* Resolved HERE and not at argument parsing: it is the PLUGIN that says how deep this
     * container's drafter goes, and the plugin is not selected until the container is open. Every
     * consumer of `n_spec` -- buffer sizing, the kernel bands, the scheduler's window -- runs
     * after this line. The core reads no drafter configuration of its own; rad_arch_probe is the
     * whole of what it knows. */
    /* THE KL MODE MEASURES THE TRUNK (core/kld.h). A drafter adds passes that write no scored
     * row, and a prefix hit would start a document past positions the reference holds, so both
     * are off -- and an explicit depth is refused rather than dropped, because the operator asked
     * for something this run will not do. The tower stays off the card: the corpus is text. */
    if (cfg_.kld()) {
        if (cfg_.n_spec > 0) {
            RAD_ERR("--num-speculative-tokens %d: the KL mode scores the trunk's own rows and "
                    "drafts nothing; leave it at auto or 0", cfg_.n_spec);
            return RAD_E_INVAL;
        }
        cfg_.n_spec = 0;
        cfg_.prefix_cache = false;
        if (cfg_.mm_max_patches < 0) cfg_.mm_max_patches = 0;
    }
    if (cfg_.n_spec < 0) {
        RadArchProbe pb{};
        (void)rad_tools_probe(plugins_, meta_, &pb);
        cfg_.n_spec = pb.draft_depth > 0 ? pb.draft_depth : 0;
        if (cfg_.n_spec > 0)
            RAD_INFO("speculation: depth %d, this drafter's own operating point "
                     "(--num-speculative-tokens N overrides it)", cfg_.n_spec);
        else
            RAD_INFO("speculation: off -- this container carries no draft head");
    }
    /* THE ENCODER'S PASS SIZE, resolved the same way and for the same reason: whether there is a
     * tower to size is a property of the model file. A container says so (`radiance.encoder`,
     * written when its weights went in); a checkpoint served directly is the release itself, so a
     * vision_config is a tower whose weights are in it. 16384 patches is a 2048 x 2048 image, the
     * largest one pass takes; the reference processor allows sixteen times that area, which would
     * need an activation region sixteen times as large on every card for one picture. */
    if (cfg_.mm_max_patches < 0) {
        const char* enc = rad_meta_gets(&meta_, "radiance.encoder", "");
        const bool tower = rad_meta_geti(&meta_, "vision_config.patch_size", 0) > 0;
        cfg_.mm_max_patches = (enc && *enc) || (!file_ && tower) ? 16384 : 0;
        if (cfg_.mm_max_patches == 0 && tower)
            RAD_INFO("media: off -- the checkpoint has a vision tower but this container was "
                     "converted without its weights; convert again to serve images and video");
    }

    std::vector<int>         st(ranks_.size(), RAD_OK);
    std::vector<std::thread> th;
    th.reserve(ranks_.size());
    for (auto& rp : ranks_) {
        Rank* r = rp.get();
        th.emplace_back([this, r, &st] {
            const int ri = r->index;
            LogRank lr(ri);
            if (rad_dev_set(r->device) < 0) {
                RAD_ERR("rank %d: could not bind device %d to declare on", ri, r->device);
                st[(size_t)ri] = RAD_E_DEVICE;
                return;
            }
            const RadBuildCtx bc = build_ctx(ri, cfg_.max_tok);
            st[(size_t)ri] = rad_tools_declare(plugins_, meta_, bc, &r->program,
                                               [this, ri](RadBuilder* bb, int64_t lw) {
                                                   return declare_sampler(ri, bb, lw);
                                               },
                                               nullptr, sources());
        });
    }
    /* A message is a step's rows of the residual stream in bf16, so the floor is a ROW COUNT: a
     * decode step's rows are its sequences times one plus the speculative depth, and concurrency
     * alone decides whether a decode step is under it. */
    if (cfg_.tp > 1 && !cfg_.tp_wire_exact) {
        const int64_t row_bytes = meta_.n_embd > 0 ? meta_.n_embd * 2 : 1;
        const int64_t exact_rows = (cfg_.tp_wire_min_kb * 1024 + row_bytes - 1) / row_bytes - 1;
        RAD_WARN("tp wire: LOSSY (--tp-wire wht6) at %lld KiB a message and above. A cross-rank "
                 "all-reduce of that size carries a quantised payload instead of bf16 (libr4d's "
                 "is Walsh-Hadamard-rotated 6-bit at two ranks and tiered int8 at four), so this "
                 "process does not compute the same function the exact wire does. Steps of at "
                 "most %lld rows are exact: single-sequence decode, and a "
                 "decode step until sequences x (1 + speculative depth) passes that; prefill and "
                 "a busy decode step are lossy. --tp-wire-min-kb moves the floor.",
                 (long long)cfg_.tp_wire_min_kb, (long long)exact_rows);
    }

    for (auto& t : th) t.join();

    /* This phase ALWAYS completes: if anything failed to resolve, the FULL list is reported at
     * the end rather than the first failure at the top (spec §1 step 4). Reported after the join
     * and in rank order -- two ranks printing their lists concurrently interleaves them into one
     * that reads like neither.
     *
     * ONCE FOR EVERY RANK THAT AGREES, which is normally all of them: each rank declares the same
     * model against the same registry, so N ranks produce N identical copies of the same list. A
     * rank whose report differs is itself worth seeing, and gets its own. */
    {
        std::vector<MissReport> rep;
        for (auto& r : ranks_) rep.push_back(miss_report(r->program));
        bool same = true;
        for (size_t i = 1; i < rep.size(); ++i) if (!(rep[i] == rep[0])) same = false;

        for (size_t i = 0; i < rep.size(); ++i) {
            if (same && i > 0) break;
            const std::string who = same && rep.size() > 1 ? fmt("all %zu ranks", rep.size())
                                                           : fmt("rank %d", ranks_[i]->index);
            if (!rep[i].device.empty()) {
                RAD_ERR("%s: %zu device band(s) resolved to no kernel -- a request that lands in "
                        "one of them cannot run:", who.c_str(), rep[i].device.size());
                for (const std::string& l : rep[i].device) RAD_ERR("    %s", l.c_str());
            }
            if (!rep[i].host_only.empty()) RAD_INFO("%s", rep[i].host_only.c_str());
            if (!rep[i].host.empty()) RAD_INFO("%s", rep[i].host.c_str());
        }
    }
    /* A CHECKPOINT WEIGHT THAT IS NOT SERVED AS IT STANDS was declared as plain and nothing can
     * load it: every one is named, with the remedy, whatever declare made of the rest. */
    if (ckpt_src_) {
        const std::vector<std::string> no = ckpt_src_->refused();
        if (!no.empty()) {
            RAD_ERR("%zu weight(s) of %s are not in a form the engine serves as it stands -- "
                    "nothing is quantised at load. Convert the checkpoint to a container with a "
                    "recipe that says what they become (rad-convert --recipe):", no.size(),
                    cfg_.model.c_str());
            for (size_t i = 0; i < no.size() && i < 16; ++i) RAD_ERR("    %s", no[i].c_str());
            if (no.size() > 16) RAD_ERR("    ... and %zu more", no.size() - 16);
            return RAD_E_UNSUPPORTED;
        }
    }
    for (size_t i = 0; i < st.size(); ++i) if (st[i] < 0) return st[i];
    /* WHICH LINEAR GROUPS NEED NO SCRATCH BANK, decided once and here: the bands are resolved, and
     * plan()'s budget probes and configure_kv() both size the pools from this flag. Every rank or
     * none -- rank 0's manager names the slots for all of them (Engine::state_slots_of), so their
     * pools keep one geometry. */
    for (size_t g = 0; g < ranks_[0]->program.kv_groups.size(); ++g) {
        bool bound = true;
        for (const auto& r : ranks_)
            bound = bound && g < r->program.kv_groups.size() && tree_linear_bound(r->program, g);
        for (auto& r : ranks_)
            if (g < r->program.kv_groups.size()) r->program.kv_groups[g].tree_bound = bound;
    }
    RAD_INFO("declare: %zu ops, %zu weights, %zu buffers, %zu KV groups",
             ranks_[0]->program.ops.size() - 1, ranks_[0]->program.weights.size() - 1,
             ranks_[0]->program.buffers.size() - 1, ranks_[0]->program.kv_groups.size());

    /* IS THERE A DRAFT HEAD, AND WHAT SHAPE IS IT? The plugin declared it (rad_declare_drafter)
     * or it did not, and that is the whole of the question. There is no flag to read, no metadata
     * key to re-read and no second place for the two to disagree.
     *
     * Sniffing it out instead -- scanning the declared weights for known draft-head names, then
     * re-reading the metadata keys the plugin already read to build the head -- would put a list
     * of model names in the core, and a third party's drafter could not then exist without
     * editing this function. */
    const Program& p0 = ranks_[0]->program;
    drafter_       = p0.drafter;
    drafter_name_  = p0.drafter_name;
    drafter_.name  = drafter_name_.c_str();
    if (cfg_.n_spec <= 0) drafter_ = RadDrafterDecl{};
    const char* no_lookup = std::getenv("RADIANCE_NO_PROMPT_LOOKUP");
    const char* no_draft = std::getenv("RADIANCE_NO_DRAFT");
    const char* lookup_note = no_lookup && *no_lookup == '1'
        ? "prompt lookup disabled (RADIANCE_NO_PROMPT_LOOKUP=1)"
        : no_draft && *no_draft == '1' ? "prompt lookup disabled (RADIANCE_NO_DRAFT=1)"
        : "prompt lookup fills in for any sequence it declines";
    int serial_passes = drafter_.depth;
    if (drafter_.kind == RAD_DRAFT_SERIAL && p0.drafter_tree_capacity > 0) {
        const int tree_rounds = !std::strcmp(drafter_.name, "mtp") ? mtp_tree_config().rounds : 3;
        serial_passes = std::min(serial_passes, tree_rounds);
    }
    if (drafter_.kind != RAD_DRAFT_NONE) {
        /* The proposal buffer is a HANDLE the plugin passed, not a name this looked up. Zero is
         * legal and means the sampler's token buffer -- a head whose logits go through the
         * ordinary sampler chain, which the bf16 MTP head does and the 2-bit draft head does
         * not. */
        RAD_INFO("speculation: %s",
                 drafter_.kind == RAD_DRAFT_BLOCK
                 ? fmt("'%s' proposes %d token(s) in ONE block pass over %lld row(s) (mask id %d, "
                       "window %lld); %s",
                       drafter_.name, drafter_.depth, (long long)drafter_.block,
                       (int)drafter_.mask_token, (long long)drafter_.window, lookup_note).c_str()
                 : fmt("'%s' drafts %d token(s) a step in %d serial pass(es); %s",
                       drafter_.name, drafter_.depth, serial_passes, lookup_note).c_str());
    } else if (cfg_.n_spec > 0) {
        if ((no_lookup && *no_lookup == '1') || (no_draft && *no_draft == '1'))
            RAD_INFO("speculation: no declared draft head; %s", lookup_note);
        else
            RAD_INFO("speculation: prompt lookup only -- this container declares no draft head");
    }
    probe_arena_levels();
    return RAD_OK;
}

/* THE ACTIVATION ARENA'S SMALLER LEVELS, sized by the plugin itself.
 *
 * Every activation is declared for the largest step, and a decode step is a few dozen rows of the
 * 2048 a prefill chunk carries: most of the arena sits unread for as long as nothing prefills. The
 * plugin is asked to declare again at a smaller step (RadBuildCtx::shape_probe) -- its own shape
 * rules are the only ones that know how each buffer scales, so no rule here second-guesses them --
 * and each answer that describes the same buffers with only their rows smaller becomes a level:
 * the same fixed region, the transients packed lower, and the arena above them free while the
 * steps are that small.
 *
 * THE SMALLEST LEVEL IS 256 ROWS AT LEAST. A kernel that computes whole tiles may write the
 * padding rows of its last one; at the real size those land inside the buffer because the chunk is
 * a multiple of every tile, and a level is held to the same property so that padding never reaches
 * the next buffer or the memory past the level's end. Each level above is four times the one
 * below, up to the chunk.
 *
 * A refusal costs a level and nothing else: the arena is lent from the levels that were accepted,
 * or not at all. */
void Engine::probe_arena_levels() {
    for (auto& r : ranks_) r->arena_levels.clear();
    RadArchProbe pb{};
    (void)rad_tools_probe(plugins_, meta_, &pb);
    if (!pb.shape_probe_ok) {
        RAD_INFO("activation arena: not lent -- the '%s' plugin does not declare at a smaller step "
                 "size (RadArchProbe::shape_probe_ok)", plugins_.arch_plugin.c_str());
        return;
    }
    int64_t lo = 1;
    const int64_t rows_min = cfg_.max_seqs * (1 + std::max(cfg_.n_spec, 0));
    while (lo < rows_min) lo <<= 1;
    lo = std::max<int64_t>(lo, 256);
    std::vector<int64_t> want;
    for (int64_t R = lo; R < cfg_.max_tok; R *= 4) want.push_back(R);

    /* One sizing declare a (level, rank), all of them side by side: a sizing declare selects and
     * initialises nothing, so they share no state, and each is a whole declare of the model. */
    const size_t nl = want.size(), nr = ranks_.size();
    std::vector<ArenaLevel>  got(nl * nr);
    std::vector<std::string> why(nl * nr);
    std::vector<char>        ok(nl * nr, 0);
    {
        std::vector<std::thread> th;
        th.reserve(nl * nr);
        for (size_t l = 0; l < nl; ++l)
            for (size_t i = 0; i < nr; ++i)
                th.emplace_back([this, &want, &got, &why, &ok, l, i, nr] {
                    Rank& r = *ranks_[i];
                    LogRank lr(r.index);
                    const size_t k = l * nr + i;
                    if (rad_dev_set(r.device) < 0) {
                        why[k] = fmt("could not bind device %d", r.device);
                        return;
                    }
                    RadBuildCtx bc = build_ctx(r.index, want[l]);
                    bc.shape_probe = 1;
                    Program probe;
                    const int st = rad_tools_declare(plugins_, meta_, bc, &probe,
                                                     [this, &r](RadBuilder* bb, int64_t lw) {
                                                         return declare_sampler(r.index, bb, lw,
                                                                                true);
                                                     },
                                                     &r.program, sources());
                    if (st < 0) why[k] = fmt("the sizing declare failed: %s", rad_strerror(st));
                    else if (plan_level(r.program, probe, want[l], &got[k], &why[k]) >= 0)
                        ok[k] = 1;
                });
        for (auto& t : th) t.join();
    }

    std::string said;
    for (size_t l = 0; l < nl; ++l) {
        const int64_t R = want[l];
        size_t bad = nr;
        for (size_t i = 0; i < nr && bad == nr; ++i) if (!ok[l * nr + i]) bad = i;
        if (bad < nr) {
            RAD_WARN("activation arena: no level for %lld-token steps on rank %d -- %s",
                     (long long)R, ranks_[bad]->index, why[l * nr + bad].c_str());
            continue;
        }
        for (size_t i = 0; i < nr; ++i)
            ranks_[i]->arena_levels.push_back(std::move(got[l * nr + i]));
        said += fmt("%s%lld tokens use %s", said.empty() ? "" : ", ", (long long)R,
                    humanb(ranks_[0]->arena_levels.back().end).c_str());
    }
    if (!said.empty())
        RAD_INFO("activation arena: %s of %s (the fixed region is %s); the rest is lent to the "
                 "expert slab while the steps are that small", said.c_str(),
                 humanb(ranks_[0]->program.arena_bytes).c_str(),
                 humanb(ranks_[0]->program.arena_fixed_bytes).c_str());
}

/* ================================================================== 5. plan */
int Engine::plan() {
    /* THE BUDGET IS RESOLVED ONCE, FOR EVERY RANK, BEFORE ANY RANK ALLOCATES.
     *
     * It has to happen here and not inside the per-rank loop below for two reasons. One card's
     * answer is not another's -- the fold in vram_budget_resolve takes the smallest capacity and
     * the largest program so that one split fits every card it is applied to -- and the answer is
     * written back into Config, which the loop then reads. Doing it per rank would let rank 1
     * re-resolve against a card rank 0 has already filled.
     *
     * Every input is available by now and not before: declare() has run, so the buffer plan exists
     * and the arena can be sized; every kernel plugin's init has run, so the code objects are
     * resident and the free-VRAM reading is the steady-state one; and the KV groups are declared,
     * so a full cache's cost is computable. */
    {
        std::vector<VramFacts> facts;
        facts.reserve(ranks_.size());
        late_step_bytes_ = late_lazy_bytes_ = 0;
        /* The placement flags' tables, per rank where they are sharded (the embedding) and once
         * where they are not (the n-gram table every rank reads from the one mapping). */
        std::vector<int64_t> emb_bytes;
        int64_t ngram_bytes = 0, kv_floor = 0;
        bool ngram_device = false, ngram_host = false;
        const int64_t ctx = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
        for (auto& r : ranks_) {
            RAD_TRY(rad_dev_set(r->device));
            VramFacts f{};
            /* ONE HARDWARE QUEUE'S VRAM, MEASURED BY MAKING TWO. The engine's compute stream,
             * its mover's and a second compute lane are all made after this reading, and the
             * driver backs each queue with VRAM the ring alone does not account for. The second
             * queue is the unit, so a runtime that sets itself up on its first queue charges that
             * once and to `already held`, which is read after both are gone. */
            {
                RadStream q[2] = { nullptr, nullptr };
                RadDeviceProps a{}, b{};
                if (rad_stream_create(&q[0], 0) >= 0 && rad_dev_props(r->device, &a) >= 0 &&
                    rad_stream_create(&q[1], 0) >= 0 && rad_dev_props(r->device, &b) >= 0)
                    f.queue_bytes = std::max<int64_t>(a.vram_free - b.vram_free, 0);
                for (RadStream s : q) if (s) rad_stream_destroy(s);
            }
            f.n_queues = 2 + (r->program.scratch_regions > 1 ? 1 : 0);
            RadDeviceProps props{};
            rad_dev_props(r->device, &props);
            f.capacity = props.vram_bytes;
            f.free     = props.vram_free;
            /* AN EIGHTH OF THE HOST'S MEMORY STAYS THE HOST'S on integrated graphics. The card's
             * free figure is already bounded by what the host has available (core/device/hip.cpp);
             * this is the margin the rest of the machine keeps on top of that, because a budget
             * that spends every available byte leaves the operating system to reclaim its page
             * cache and then to kill something. */
            if (dev_integrated(r->device)) {
                const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
                if (pages > 0 && page > 0) f.host_reserve = (int64_t)pages * page / 8;
            }
            arena_plan_bytes(r->program, nullptr, &f.arena);
            /* WHAT THE WEIGHTS WOULD COST, split into the floor and the elastic plane -- see
             * weight_footprint in core/place/planner.h. Per RANK, which under tensor parallel is
             * already this card's even share of the static weights: declare ran once per rank
             * with every dimension divided. */
            WeightFootprint wf;
            weight_footprint(r->program, &wf, TablePlacement::Vram, TablePlacement::Disk);
            f.statics = wf.statics;
            f.experts = wf.experts;
            emb_bytes.push_back(wf.embedding);
            ngram_bytes = std::max(ngram_bytes, wf.ngram);
            ngram_device = ngram_device || wf.ngram_device;
            ngram_host = ngram_host || wf.ngram_host;
            /* ONE --max-model-len SEQUENCE'S CACHE: the same groups planned for a single
             * sequence, which is the least a KV pool can hold and still serve every request the
             * API accepts (configure_kv warns below it). --embedding-placement auto keeps the
             * table off the card only to stay above it. */
            {
                Config one = cfg_;
                one.max_seqs = 1;
                KVManager single;
                if (single.plan_groups(r->program.kv_groups, one) == RAD_OK)
                    kv_floor = std::max(kv_floor, single.ceiling(ctx));
            }
            /* THE STAGING REGION IS SIZED ONLY WHERE THE ARENA LENDS ITS TOP: with no smaller level
             * nothing gives the region back to the slab at decode, and a deterministic plan moves
             * nothing, so neither can stage or recover what a region would cost. */
            if (!r->arena_levels.empty() && !cfg_.deterministic) f.expert_layer_max = wf.expert_layer_max;
            /* A THROWAWAY MANAGER, because the question is what the groups WOULD cost and the
             * real one is configured against a pool that does not exist yet. plan_groups touches
             * no memory and no pool; it is the half of configure() that is pure arithmetic on the
             * declaration, split out for exactly this caller. */
            KVManager probe;
            if (probe.plan_groups(r->program.kv_groups, cfg_) == RAD_OK) {
                f.kv_ceiling = probe.ceiling(ctx);
                f.kv_fixed   = probe.fixed_bytes();
            }
            /* THE STEP BATCH'S DEVICE STAGING, which the scheduler reserves after this split: its
             * block tables are carved for max_seqs sequences at max_ctx and its media staging for
             * max_tok encoder rows. The state-index widths are the probe's, as the scheduler takes
             * them from the configured manager. */
            KVGeom kg = KVGeom::of(r->program.kv_groups);
            for (size_t i = 0; i < kg.g.size(); ++i) {
                const int64_t w = probe.state_index_width((int32_t)i);
                if (w > 0) kg.g[i].sidx_width = w;
            }
            f.staging = BatchBuilder::device_bytes(r->program, cfg_, kg);
            if (f.staging < 0) return RAD_E_INVAL;
            late_step_bytes_ = std::max(late_step_bytes_, f.staging);
            int64_t lazy = r->program.scratch_regions > 1 ? f.queue_bytes : 0;
            /* AND THE KV TIERS' COPY STAGING, which TierExec::device_init builds on the first
             * copy out. Which groups the prefix cache indexes is settled after this split, so
             * every paged group is counted: the entry it sizes can only be smaller. */
            if (cfg_.prefix_cache_host_mib > 0 || cfg_.prefix_cache_disk_mib > 0) {
                int64_t entry = 0, bs = 0;
                int ng = 0;
                for (size_t g = 0; g < r->program.kv_groups.size(); ++g) {
                    const KVGroupPlan* p = probe.plan((int32_t)g);
                    if (!p || !p->paged() || p->bytes_per_block <= 0) continue;
                    entry += p->bytes_per_block;
                    bs = bs > 0 ? std::min<int64_t>(bs, p->block_size) : p->block_size;
                    ++ng;
                }
                const int64_t tier = TierExec::device_bytes(entry, ng, TierExec::sizes(entry, ctx, bs));
                f.staging += tier;
                lazy += tier;
            }
            late_lazy_bytes_ = std::max(late_lazy_bytes_, lazy);
            facts.push_back(f);
        }
        RAD_TRY(resolve_table_placement(facts, emb_bytes, kv_floor, ngram_bytes, ngram_device,
                                        ngram_host));
        /* The n-gram table was counted off the card; on it, it is every rank's whole table. */
        if (cfg_.ngram_resolved == TablePlacement::Vram)
            for (VramFacts& f : facts) f.statics += ngram_bytes;
        if (cfg_.embedding_resolved == TablePlacement::Ram)
            for (size_t i = 0; i < facts.size(); ++i) facts[i].statics -= emb_bytes[i];
        VramBudget vb;
        RAD_TRY(vram_budget_resolve(cfg_, facts, &vb));
        if (!vb.report.empty()) RAD_INFO("%s", vb.report.c_str());
        /* The staging region the budget took out of the experts, above everything the buffer plan
         * placed -- so it is at the top of the arena, the part the smaller levels lend. */
        if (vb.stage > 0)
            for (auto& r : ranks_) {
                int64_t dev = 0;
                arena_plan_bytes(r->program, &dev, nullptr);
                r->program.arena_stage_off   = align_up(dev, RAD_ALIGN_UNIT);
                r->program.arena_stage_bytes = vb.stage;
                r->program.arena_bytes       = r->program.arena_stage_off + vb.stage;
            }
    }

    for (auto& r : ranks_) {
        LogRank lr(r->index);
        /* THE MAIN THREAD RUNS EVERY RANK'S STARTUP, one rank after another, and every allocator
         * reachable from here takes the CURRENT device -- rad_dev_alloc has no device argument.
         * Unbound, rank 1's pools, its weights and its KV cache all land on card 0. That fits by
         * accident at tp 1 and is silently wrong above it, so each phase binds the rank it is
         * working on before it allocates anything for it. */
        RAD_TRY(rad_dev_set(r->device));
        RadDeviceProps props{};
        rad_dev_props(r->device, &props);

        /* Every pool is budgeted explicitly. No profiling pass, no utilisation fraction, no
         * inference about what the workload will be (spec §6). Pools refuses by name and by how
         * much rather than shrinking anything. */
        int s = r->pools.configure(cfg_, props.vram_bytes);
        RAD_INFO("pools:\n%s", r->pools.fit_report().c_str());
        if (s < 0) {
            report_pool_refusal(r->index, r->device, r->pools, cfg_);
            return s;
        }

        /* Under tensor parallel the placement unit is a SHARD of a weight, so each rank plans
         * against its own card's budget (spec §5.3). */
        PlannerInput in{};
        in.program    = &r->program;
        in.config     = &cfg_;
        in.rank       = r->index;
        in.world_size = cfg_.tp;
        if (file_ && file_->header().prof_count) {
            in.profile   = file_->profile();
            in.n_profile = (int64_t)file_->header().prof_count;
        }
        in.container = file_ != nullptr;
        Planner planner(in);
        s = planner.plan(&r->plan);
        if (cfg_.debug_placement || s < 0)
            fputs(r->plan.report(r->program).c_str(), stderr);
        if (s < 0) return s;
    }
    return RAD_OK;
}

/* ================================================================== 6. load */
/* Weights are mapped and staged according to the plan. The mover has already reserved every slot
 * and published the address each weight will live at, so this is a fill: for every declared
 * weight, find its entry in the container and copy THIS RANK'S SHARD of it into place.
 *
 * The container holds whole weights unless it was converted pre-sharded, and the plugin declared
 * dimensions ALREADY DIVIDED (spec §9) -- so the slicing happens here, and it is the only place
 * in the engine that knows a weight is a piece of a larger one. Row sharding is a contiguous
 * range of dim 0. Column sharding is not contiguous and needs a 2D copy, which is exactly why
 * the two are different enum values rather than a bool. */
/* ONE RUN OF CONTAINER BYTES, AND WHERE THEY GO. Emitted by the walk below, consumed by the
 * sweep; `ny == 1` with `span == wb` is one contiguous range, and everything else is a column
 * share, whose pieces are `wb` bytes every `sp`. A run with `stage >= 0` fills that staged
 * weight's input at `dst` bytes in rather than landing where the weight lives. */
struct LoadRun {
    int64_t  foff;      /* first container byte the run reads */
    int64_t  span;      /* container bytes it covers, its own internal gaps included */
    uint8_t* dst;       /* the destination, or the offset into the staged input */
    int64_t  wb;        /* bytes in one contiguous piece */
    int64_t  sp;        /* container bytes from the start of one piece to the next */
    int64_t  ny;        /* how many pieces */
    int32_t  rank;      /* which rank's stream carries the copy */
    int32_t  dev;       /* the destination is device memory; 0 is a host pool and a plain memcpy */
    int32_t  stage;     /* the staged weight it fills, or -1 */
};

/* ================================================================== the relayout pool
 *
 * A WEIGHT THE LOADER HAS TO REARRANGE (spec §4.3): the resolved kernel stores something other
 * than the container's canonical planes -- a fragment order, a scale carried in each row's tail --
 * or the declaration widens a bf16 plane to f32. Its share of the planes is read out of the sweep
 * into host memory of its own, and when the last byte of it has arrived a worker produces the
 * stored form: straight into its host pool, or, for VRAM, into memory of its own and up through
 * that worker's pinned bounce buffers.
 *
 * IT HAS TO KEEP PACE WITH THE DISK, because Qwen3.8-27B-FP8 rearranges 24 GiB of its 29 at every
 * load and a single thread scattering bytes into fragment order runs at a fraction of 7 GB/s. So
 * it is a pool, and the sweep only reads: the copies out of its windows -- into a staged input,
 * or into a host pool slot -- are the pool's too, taken before any relayout, because a window
 * cannot be read into again until they are done. On the sweep thread those copies, and the first
 * touch of every fresh input, would run in series with the reads.
 *
 * AND IT HOLDS A BOUNDED AMOUNT, because a staged input is a copy of the weight in host memory and
 * a model is larger than that memory. A new input waits for room while there is finished work
 * that will free some -- and only then: a weight still arriving cannot be relaid, so waiting on
 * it would wait forever. Past the bound with nothing finished, the input is taken anyway. */
struct StagedWeight {
    const Program* prog = nullptr;
    size_t   wi = 0;
    int      device = 0;
    uint8_t* dst = nullptr;     /* where the stored form goes */
    bool     dev = false;       /* ...in device memory */
    int64_t  in_bytes = 0;
    int64_t  plane_off[RAD_ENC_MAX_PLANES] = {0};
    int64_t  remaining = 0;     /* bytes the sweep has yet to deliver */
    bool     reserved = false;  /* its input counts against the pool's budget */
    uint8_t* in = nullptr;      /* its input: in the pool's region, or its own allocation */
    bool     own = false;       /* ...its own, because the region had no hole that size */
    /* A checkpoint's weight has no sweep: the worker fills its input itself, from the mapped
     * tensors, before it lays it out -- so the reads run on every worker at once. */
    std::function<int(uint8_t* in)> fill;
};

class RelayoutPool {
public:
    explicit RelayoutPool(std::vector<StagedWeight>& items)
        : items_(items), once_(new std::once_flag[items.size()]) {}
    ~RelayoutPool() {
        finish();
        for (StagedWeight& s : items_) if (s.own) delete[] s.in;
        if (region_) ::munmap(region_, (size_t)budget_);
    }

    /* THE INPUTS SHARE ONE REGION THE SIZE OF THE BUDGET, made once and handed out again as each
     * weight is done with it. Memory made fresh for every input is a page fault every 4 KiB of
     * it, and over a model's worth of weights those faults are most of the load's CPU time; a
     * region the workers keep reusing faults once. An input with no hole its size -- larger than
     * the budget, or the region too cut up -- gets an allocation of its own. */
    void start(int threads, int64_t budget) {
        budget_ = budget;
        void* p = ::mmap(nullptr, (size_t)budget_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
            region_ = (uint8_t*)p;
            ::madvise(p, (size_t)budget_, MADV_HUGEPAGE);
            holes_[0] = budget_;
        }
        int devs = 1;
        for (const StagedWeight& w : items_) if (w.dev && w.device + 1 > devs) devs = w.device + 1;
        uplinks_.reset(new Uplink[(size_t)devs]);
        n_uplinks_ = devs;
        for (int t = 0; t < threads; ++t) threads_.emplace_back([this] { worker(); });
    }

    /* The sweep thread: room for a staged weight's input, taken the first time a run reaches it.
     * ROOM AND NOT MEMORY: the allocation is input()'s, on a worker. An allocation this size is
     * an mmap, which waits on every munmap and fault the workers are making in the same address
     * space -- milliseconds an input, and on the sweep thread that is a second of a load. */
    void reserve(int i) {
        StagedWeight& s = items_[(size_t)i];
        if (s.reserved) return;
        std::unique_lock<std::mutex> lk(mu_);
        room_.wait(lk, [&] { return held_ + s.in_bytes <= budget_ || pending_ == 0 || failed_; });
        held_ += s.in_bytes;
        if (held_ > peak_) peak_ = held_;
        s.reserved = true;
    }

    /* `n` more bytes of it are in -- from whichever worker copied them, so under the lock. */
    void arrived(int i, int64_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        StagedWeight& s = items_[(size_t)i];
        s.remaining -= n;
        if (s.remaining > 0) return;
        pending_ += s.in_bytes;
        queue_.push_back(i);
        work_.notify_one();
    }

    /* A COPY OUT OF A SWEEP WINDOW: the part of run `q` that lies in file range [j0, j1), out of
     * window `slot`, whose first byte is file offset `w0`. A staged run lands in its weight's
     * input, which the sweep has made room for with reserve() first; any other run is a host
     * destination. */
    struct Copy { const uint8_t* win; int64_t w0, j0, j1; LoadRun q; int slot; };

    /* The sweep thread: how many windows it reads into. */
    void windows(int n) {
        std::lock_guard<std::mutex> lk(mu_);
        out_.assign((size_t)n, 0);
    }
    void copy(const Copy& c) {
        std::lock_guard<std::mutex> lk(mu_);
        ++out_[(size_t)c.slot];
        copies_.push_back(c);
        work_.notify_one();
    }
    /* The sweep thread: until no copy out of window `slot` is outstanding. A failed pool still
     * retires its copies, so this returns either way. */
    void drain(int slot) {
        std::unique_lock<std::mutex> lk(mu_);
        drained_.wait(lk, [&] { return out_[(size_t)slot] == 0; });
    }

    int finish() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            done_ = true;
        }
        work_.notify_all();
        for (std::thread& t : threads_) t.join();
        threads_.clear();
        for (int d = 0; d < n_uplinks_; ++d)
            if (uplinks_[(size_t)d].st) {
                rad_dev_set(d);
                rad_stream_destroy(uplinks_[(size_t)d].st);
                uplinks_[(size_t)d].st = nullptr;
            }
        return status_;
    }

    bool failed() const { return failed_; }
    int64_t peak() const { return peak_; }

private:
    /* ONE STREAM A CARD FOR EVERY WORKER'S COPIES. A stream is a hardware queue of its own on
     * this backend, and a stream a worker a card made some twenty queues a card, which the card
     * time-slices: the workers spent three quarters of a Flash-Next load waiting on their copies.
     * One link copy fills the link by itself, so one queue a card loses nothing; the workers take
     * turns putting their copies on it. */
    struct Uplink {
        std::mutex mu;
        RadStream  st = nullptr;
    };

    /* A worker's way onto a card: a ring of pinned slots. A weight that fits one is laid out
     * straight into it, and a larger one goes up a slot's worth at a time; each slot is waited
     * for when the ring comes back round to it, not after every copy. */
    struct Bounce {
        static constexpr int     kSlots = 8;
        static constexpr int64_t kSlot  = 4ll << 20;
        explicit Bounce(Uplink* up) : up_(up) {}
        Uplink* up_;
        void* buf = nullptr;
        /* THE RELAYOUT'S OUTPUT for a weight larger than a slot, kept from one weight to the next
         * up to kKeep -- the same page faults as the inputs' region saves, and a worker's own. */
        static constexpr int64_t kKeep = 256ll << 20;
        std::unique_ptr<uint8_t[]> out;
        int64_t out_cap = 0;
        /* An event a device, since an event belongs to the device it was made on; `live` is the
         * device the slot's last copy went to. */
        struct Slot { std::vector<RadEvent> ev; int live = -1; };
        Slot slot[kSlots];
        int  next = 0;
        ~Bounce() {
            (void)flush();
            for (Slot& s : slot)
                for (size_t d = 0; d < s.ev.size(); ++d)
                    if (s.ev[d]) { rad_dev_set((int)d); rad_event_destroy(s.ev[d]); }
            if (buf) rad_dev_free(buf, RAD_MEM_HOST_PINNED);
        }
        /* The next slot, once the copy made out of it last time round has run. */
        int acquire(uint8_t** p) {
            if (!buf) {
                buf = rad_dev_alloc(kSlots * kSlot, RAD_MEM_HOST_PINNED);
                if (!buf) return RAD_E_NOMEM;
            }
            Slot& s = slot[next];
            if (s.live >= 0) {
                RAD_TRY(rad_event_sync(s.ev[(size_t)s.live]));
                s.live = -1;
            }
            *p = (uint8_t*)buf + next * kSlot;
            return RAD_OK;
        }
        /* The first `n` bytes of the slot acquire() handed out, up to `dst` on `device`. */
        int send(uint8_t* dst, int64_t n, int device) {
            Slot& s = slot[next];
            RAD_TRY(rad_dev_set(device));
            if (s.ev.size() <= (size_t)device) s.ev.resize((size_t)device + 1, nullptr);
            if (!s.ev[(size_t)device]) RAD_TRY(rad_event_create(&s.ev[(size_t)device]));
            Uplink& u = up_[device];
            {
                std::lock_guard<std::mutex> lk(u.mu);
                if (!u.st) RAD_TRY(rad_stream_create(&u.st, 0));
                RAD_TRY(rad_memcpy_async(dst, (uint8_t*)buf + next * kSlot, n, u.st));
                RAD_TRY(rad_event_record(s.ev[(size_t)device], u.st));
            }
            s.live = device;
            next = (next + 1) % kSlots;
            return RAD_OK;
        }
        int upload(uint8_t* dst, const uint8_t* src, int64_t n, int device) {
            for (int64_t off = 0; off < n;) {
                const int64_t c = n - off < kSlot ? n - off : kSlot;
                uint8_t* p = nullptr;
                RAD_TRY(acquire(&p));
                std::memcpy(p, src + off, (size_t)c);
                RAD_TRY(send(dst + off, c, device));
                off += c;
            }
            return RAD_OK;
        }
        /* Every copy this worker made has landed. */
        int flush() {
            int st = RAD_OK;
            for (Slot& s : slot)
                if (s.live >= 0) {
                    const int w = rad_event_sync(s.ev[(size_t)s.live]);
                    if (w < 0 && st == RAD_OK) st = w;
                    s.live = -1;
                }
            return st;
        }
    };

    void worker() {
        Bounce b(uplinks_.get());
        for (;;) {
            int i = -1;
            Copy c{};
            bool cp = false;
            {
                std::unique_lock<std::mutex> lk(mu_);
                work_.wait(lk, [&] { return !copies_.empty() || !queue_.empty() || done_; });
                if (!copies_.empty()) {
                    c = copies_.front();
                    copies_.pop_front();
                    cp = true;
                } else if (!queue_.empty()) {
                    i = queue_.front();
                    queue_.pop_front();
                } else {
                    break;
                }
            }
            if (cp) {
                if (!failed_) take(c);
                std::lock_guard<std::mutex> lk(mu_);
                --out_[(size_t)c.slot];
                drained_.notify_all();
                continue;
            }
            StagedWeight& s = items_[(size_t)i];
            if (!failed_) input(i);
            const int st = failed_ ? RAD_E_STATE : run(s, b);
            if (s.own) { delete[] s.in; s.in = nullptr; s.own = false; }
            std::lock_guard<std::mutex> lk(mu_);
            if (s.in) give_back(s.in - region_, align_up(s.in_bytes, kHole));
            s.in = nullptr;
            held_    -= s.in_bytes;
            pending_ -= s.in_bytes;
            if (st < 0 && !failed_) { failed_ = true; status_ = st; }
            room_.notify_all();
        }
        /* Its copies are still in flight: finish() returns once every worker's have landed. */
        const int fs = b.flush();
        std::lock_guard<std::mutex> lk(mu_);
        if (fs < 0 && !failed_) { failed_ = true; status_ = fs; }
    }

    /* A staged weight's input, made by the first worker that needs it: the first copy into it, or
     * the job that fills it from a checkpoint. The first hole that holds it. */
    uint8_t* input(int i) {
        StagedWeight& s = items_[(size_t)i];
        std::call_once(once_[(size_t)i], [&] {
            const int64_t n = align_up(s.in_bytes, kHole);
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (auto it = holes_.begin(); it != holes_.end(); ++it) {
                    if (it->second < n) continue;
                    s.in = region_ + it->first;
                    if (it->second > n) holes_[it->first + n] = it->second - n;
                    holes_.erase(it);
                    return;
                }
            }
            s.in = new uint8_t[(size_t)s.in_bytes];
            s.own = true;
        });
        return s.in;
    }

    /* Under the lock: [off, off + n) of the region is free again, merged with the holes either
     * side of it. */
    void give_back(int64_t off, int64_t n) {
        auto next = holes_.lower_bound(off);
        if (next != holes_.end() && off + n == next->first) {
            n += next->second;
            next = holes_.erase(next);
        }
        if (next != holes_.begin()) {
            auto prev = std::prev(next);
            if (prev->first + prev->second == off) { prev->second += n; return; }
        }
        holes_[off] = n;
    }

    /* One piece after another of the run, each the part of it the job's range holds: dense in
     * the destination, a row of a column share at a time. */
    void take(const Copy& c) {
        const LoadRun& q = c.q;
        uint8_t* base = q.stage >= 0 ? input(q.stage) + (intptr_t)q.dst : q.dst;
        int64_t got = 0;
        for (int64_t y = c.j0 > q.foff && q.sp > 0 ? (c.j0 - q.foff) / q.sp : 0; y < q.ny; ++y) {
            const int64_t ps = q.foff + y * q.sp;
            if (ps >= c.j1) break;
            const int64_t lo = ps > c.j0 ? ps : c.j0;
            const int64_t hi = ps + q.wb < c.j1 ? ps + q.wb : c.j1;
            if (hi <= lo) continue;
            std::memcpy(base + y * q.wb + (lo - ps), c.win + (lo - c.w0), (size_t)(hi - lo));
            got += hi - lo;
        }
        if (q.stage >= 0 && got) arrived(q.stage, got);
    }

    int run(StagedWeight& s, Bounce& b) {
        const WeightInfo& w = s.prog->weights[s.wi];
        if (s.fill) {
            const int fs = s.fill(s.in);
            if (fs < 0) return fs;
        }
        /* STORED AS IT IS, which only a weight with no sweep reaches: its share goes up as read. */
        if (w.identity && !w.widen) {
            const int64_t n = w.stored_bytes;
            if (!s.dev) { std::memcpy(s.dst, s.in, (size_t)n); return RAD_OK; }
            const int us = b.upload(s.dst, s.in, n, s.device);
            if (us < 0) RAD_ERR("load: '%s': the upload to device %d failed: %s", w.name.c_str(),
                                s.device, rad_strerror(us));
            return us;
        }
        RadTensor pt[RAD_ENC_MAX_PLANES];
        weight_planes(w, pt);
        for (int k = 0; k < w.n_sel; ++k) pt[k].data = s.in + s.plane_off[k];
        const int64_t n = w.stored_bytes;
        std::unique_ptr<uint8_t[]> tmp;
        uint8_t* out = s.dst;
        /* A weight that fits a slot is laid out in pinned memory the copy reads directly. */
        const bool in_slot = s.dev && n <= Bounce::kSlot;
        if (in_slot) {
            RAD_TRY(b.acquire(&out));
        } else if (s.dev && n > Bounce::kKeep) {
            tmp.reset(new uint8_t[(size_t)n]);
            out = tmp.get();
        } else if (s.dev) {
            if (b.out_cap < n) { b.out.reset(new uint8_t[(size_t)n]); b.out_cap = n; }
            out = b.out.get();
        }

        int st = RAD_OK;
        const char* who = "the loader";
        if (w.widen) {
            /* EXACT: every bf16 and f16 value is an f32. */
            const int64_t cnt = w.sel_rows[0] * w.sel_cols[0];
            if (cnt * 4 != n) st = RAD_E_SHAPE;
            else for (int64_t e = 0; e < cnt; ++e) ((float*)out)[e] = rad_load_f32(pt[0].data, pt[0].dtype, e);
        } else {
            const Resolved& r = s.prog->ops[(size_t)w.lay_op].bands[(size_t)w.lay_band].dom[w.lay_dom];
            const RadKernelInfo* ki = r.row->info;
            who = ki->name;
            st = ki->relayout ? ki->relayout(r.geom.params(), r.geom.n_params(), w.lay_operand,
                                             &w.enc, w.sel, pt, w.n_sel, out, n)
                              : RAD_E_UNSUPPORTED;
        }
        if (st < 0) {
            char e[256];
            rad_enc_format(&w.enc, e, sizeof e);
            RAD_ERR("load: %s could not produce '%s' (%s) from %s: %s", who, w.name.c_str(),
                    w.layout_tag.empty() ? "widened" : w.layout_tag.c_str(), e, rad_strerror(st));
            return st;
        }
        if (s.dev) {
            st = in_slot ? b.send(s.dst, n, s.device) : b.upload(s.dst, out, n, s.device);
            if (st < 0) RAD_ERR("load: '%s': the upload to device %d failed: %s", w.name.c_str(),
                                s.device, rad_strerror(st));
        }
        return st;
    }

    std::vector<StagedWeight>& items_;
    std::unique_ptr<std::once_flag[]> once_;   /* one an item: its input is made once */
    std::unique_ptr<Uplink[]>  uplinks_;       /* indexed by device */
    int                        n_uplinks_ = 0;
    static constexpr int64_t kHole = 4096;     /* every input starts on a page of its own */
    uint8_t*                   region_ = nullptr;
    std::map<int64_t, int64_t> holes_;        /* the region's free ranges, offset -> bytes */
    std::vector<std::thread>   threads_;
    std::mutex                 mu_;
    std::condition_variable    work_, room_, drained_;
    std::deque<int>            queue_;
    std::deque<Copy>           copies_;
    std::vector<int>           out_;      /* copies outstanding out of each sweep window */
    int64_t budget_ = 0, held_ = 0, pending_ = 0, peak_ = 0;
    bool done_ = false;
    std::atomic<bool> failed_{false};
    int status_ = RAD_OK;
};

/* THE ONE PASS OVER THE CONTAINER, SERVING EVERY RANK.
 *
 * The runs arrive in declaration order, grouped by rank. Sorted by FILE OFFSET they become a
 * single front-to-back sweep: read a window, hand it to every rank that asked for a piece of it,
 * move on. That is three separate wins over reading per rank as the walk goes:
 *
 *   - the file is read ONCE. A weight every rank wants -- every replicated weight, which is
 *     gigabytes on a large container -- would otherwise be read once per rank.
 *   - the reads are SEQUENTIAL. A declaration-order access pattern reads at a fraction of an
 *     NVMe disk's sequential rate; one thread reading front to back reaches it, and more threads
 *     do not help -- provided the reads go around the page cache. A buffered pread is the kernel
 *     copying every byte out of the cache into the window on this one thread, and that copy,
 *     not the disk, sets the rate: about two thirds of it alone, and half once the relayout
 *     pool's copies share the memory system. O_DIRECT lands the disk's DMA in the window itself.
 *   - the copies OVERLAP the reads. A ring of pinned windows, an event a rank on each: the read
 *     that fills one window runs while the copies out of the others are in flight -- the DMAs to
 *     the cards, and the pool's memcpys to host memory. Serialising them costs about a fifth of
 *     the load rate; waiting on the stream instead of on the event costs the same, because the
 *     stream also holds the copies this read was supposed to hide under.
 *
 * Host pinned memory is portable across devices in one process, so ONE window feeds every card:
 * copying each window to all of them still runs at the disk's rate rather than the link's. */
static int sweep_container(RadFile* file, std::vector<LoadRun>& runs,
                           const std::vector<int>& devices, RelayoutPool* pool,
                           int64_t* read_out) {
    *read_out = 0;
    if (runs.empty()) return RAD_OK;

    /* By where the bytes ARE, which is the whole idea: after this the runs are a reading order
     * and no longer a rank order. The tie-break on span keeps the order total, so two builds of
     * the same load issue their copies in the same sequence. */
    std::sort(runs.begin(), runs.end(), [](const LoadRun& a, const LoadRun& b) {
        if (a.foff != b.foff) return a.foff < b.foff;
        return a.span > b.span;
    });

    /* What the sweep will actually read, for the progress line: the union of the runs, which is
     * NOT the sum of them at --tp 2 and not the size of the file either -- the mapped tier is in
     * neither. */
    int64_t want_bytes = 0;
    {
        int64_t hi = -1;
        for (const LoadRun& q : runs) {
            const int64_t a = q.foff, b = a + q.span;
            if (a > hi)      { want_bytes += b - a; hi = b; }
            else if (b > hi) { want_bytes += b - hi; hi = b; }
        }
    }

    int64_t window = 64 << 20;
    if (const char* e = getenv("RADIANCE_LOAD_WINDOW_MIB")) {
        const long long v = atoll(e);
        if (v > 0) window = (int64_t)v << 20;
    }
    /* A window WIDE ENOUGH to hold one piece of a strided run, whatever the widest is. The
     * slicing below is correct at any window -- a piece cut by a chunk boundary is copied in the
     * two halves it falls in -- but a window narrower than a piece turns every piece into two
     * copies instead of one 2D copy for a whole row group at a time. Nothing in this tree comes
     * near 64 MiB in a piece: a column shard's piece is one row group of one rank's columns. */
    int64_t widest = 0;
    for (const LoadRun& q : runs) if (q.ny > 1 && q.wb > widest) widest = q.wb;
    if (widest > window) window = align_up(widest, 4096);

    /* READ THROUGH A GAP RATHER THAN BREAK THE STREAM FOR IT, up to a point. On an NVMe disk a
     * megabyte of sequential read costs about what starting a fresh read costs, so bytes nobody
     * asked for are cheaper than the break until there are a few million of them. The gaps that
     * matter are whole tiers -- a mapped n-gram table is tens of gigabytes of one -- and those
     * are nowhere near this bound, so the choice only ever costs a little and saves a lot. */
    const int64_t kGap = 4 << 20;

    /* SIX WINDOWS, because the host copies out of one are a worker's memcpy and not a DMA: a
     * window a staged weight's piece sits in is free only when a worker has taken it, and the
     * ring is how far the read may run ahead of them. Pieces are cut at kPiece so one window's
     * worth spreads over several workers. */
    constexpr int     kWin   = 6;
    constexpr int64_t kPiece = 8 << 20;
    constexpr int64_t kBlk   = DirectFile::alignment();
    DirectFile dio;
    const size_t nr = devices.size();
    std::vector<RadStream> st(nr, nullptr);
    std::vector<RadEvent>  ev(kWin * nr, nullptr);
    std::vector<char>      live(kWin * nr, 0), touched(nr, 0);
    void*    buf[kWin] = {};
    pool->windows(kWin);
    int64_t  read_bytes = 0;
    double   read_s = 0;     /* of the sweep's wall time, how much was the disk itself */

    const int rc = [&]() -> int {
        for (size_t k = 0; k < nr; ++k) {
            /* A stream and an event both belong to the device that was current when they were
             * made, and recording one on the other's stream is an error rather than a slow path.
             * The COPIES need no such bind: a stream carries its own device. */
            RAD_TRY(rad_dev_set(devices[k]));
            RAD_TRY(rad_stream_create(&st[k], 0));
            for (int b = 0; b < kWin; ++b) RAD_TRY(rad_event_create(&ev[(size_t)b * nr + k]));
        }
        /* O_DIRECT WANTS THE OFFSET, THE LENGTH AND THE BUFFER BLOCK-ALIGNED, and a chunk is
         * wherever its runs put it: a window is read from the block below the chunk to the block
         * above it, so it is two blocks wider than the chunk can be. A filesystem that refuses
         * O_DIRECT is read through the cache instead, at the cache's rate. */
        RAD_TRY(dio.open(file->path(), RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED));
        for (int b = 0; b < kWin; ++b) {
            buf[b] = rad_dev_alloc(window + 2 * kBlk, RAD_MEM_HOST_PINNED);
            if (!buf[b]) {
                RAD_ERR("load: no %s pinned staging window. The load needs %d of them; lower "
                        "RADIANCE_LOAD_WINDOW_MIB or free host memory.",
                        humanb(window).c_str(), kWin);
                return RAD_E_NOMEM;
            }
        }

        int cur = 0;

        /* Read the chunk [c0, c1) into window `b`; `*at` is where its first byte landed. The
         * blocks are whole up to the file's last full one, and a tail past that is the only read
         * the bounce in DirectFile serves. Read buffered, the cache goes with the window: nothing
         * comes back for a window the sweep has finished, and tens of gigabytes of dead pages
         * turn into reclaim the reads then wait on. */
        const int64_t whole = file->size() / kBlk * kBlk;
        auto fill = [&](int b, int64_t c0, int64_t c1, const uint8_t** at) -> int {
            const auto t = std::chrono::steady_clock::now();
            const int64_t a0 = c0 / kBlk * kBlk;
            const int64_t a1 = std::min(align_up(c1, kBlk), whole);
            uint8_t* w = (uint8_t*)buf[b];
            if (a1 > a0) RAD_TRY(dio.read_at(w, a0, a1 - a0));
            if (c1 > a1) RAD_TRY(dio.read_at(w + (a1 - a0), a1, c1 - a1));
            read_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
            read_bytes += c1 - c0;
            if (!dio.direct()) file->drop_range(c0, c1 - c0);
            *at = w + (c0 - a0);
            return RAD_OK;
        };

        /* Hand one run the part of itself that lies in the staged chunk [c0, c1).
         *
         * A run is SLICED BY THE CHUNK rather than given a chunk of its own, and that is the
         * whole reason the chunk is a file range and not a run. At --tp 2 a column-sharded weight
         * is two runs over the same entry, half a row apart: served one run at a time the entry
         * is read TWICE, so the sweep reads more bytes than the container holds; served as a
         * range, both runs take their columns out of one read.
         *
         * The strided case keeps its 2D copy for every piece the chunk holds WHOLE, which is all
         * of them but the two at the ends; a piece cut by a chunk boundary is the only thing that
         * becomes a copy of its own. */
        auto one = [&](const LoadRun& q, uint8_t* dst, const uint8_t* src, int64_t len) -> int {
            touched[(size_t)q.rank] = 1;
            return rad_memcpy_async(dst, src, len, st[(size_t)q.rank]);
        };
        auto many = [&](const LoadRun& q, uint8_t* dst, const uint8_t* src, int64_t n) -> int {
            touched[(size_t)q.rank] = 1;
            return rad_memcpy_2d_async(dst, q.wb, src, q.sp, q.wb, n, st[(size_t)q.rank]);
        };
        auto issue = [&](const LoadRun& q, const uint8_t* win, int64_t c0, int64_t c1,
                         int slot) -> int {
            /* A HOST DESTINATION IS A MEMCPY, and the pool's workers make it: a staged run into
             * its weight's input, which tells the pool how much of it has arrived, and any other
             * into its host pool slot. The sweep only reads. */
            if (q.stage >= 0 || !q.dev) {
                if (pool->failed()) return RAD_E_STATE;
                if (q.stage >= 0) pool->reserve(q.stage);
                const int64_t a = q.foff > c0 ? q.foff : c0;
                const int64_t z = q.foff + q.span < c1 ? q.foff + q.span : c1;
                for (int64_t j0 = a; j0 < z; j0 += kPiece)
                    pool->copy({ win, c0, j0, j0 + kPiece < z ? j0 + kPiece : z, q, slot });
                return RAD_OK;
            }
            if (q.ny <= 1) {
                const int64_t lo = q.foff > c0 ? q.foff : c0;
                const int64_t hi = q.foff + q.wb < c1 ? q.foff + q.wb : c1;
                if (hi <= lo) return RAD_OK;
                return one(q, q.dst + (lo - q.foff), win + (lo - c0), hi - lo);
            }
            int64_t y = c0 > q.foff && q.sp > 0 ? (c0 - q.foff) / q.sp : 0;
            while (y < q.ny) {
                const int64_t ps = q.foff + y * q.sp;
                if (ps >= c1) break;
                const int64_t lo = ps > c0 ? ps : c0;
                const int64_t hi = ps + q.wb < c1 ? ps + q.wb : c1;
                if (hi <= lo) { ++y; continue; }
                if (lo == ps && hi == ps + q.wb) {
                    int64_t n = 1;
                    while (y + n < q.ny && q.foff + (y + n) * q.sp + q.wb <= c1) ++n;
                    RAD_TRY(many(q, q.dst + y * q.wb, win + (ps - c0), n));
                    y += n;
                } else {
                    RAD_TRY(one(q, q.dst + y * q.wb + (lo - ps), win + (lo - c0), hi - lo));
                    ++y;
                }
            }
            return RAD_OK;
        };
        /* The window about to be overwritten is still the source of the copies issued out of it
         * a turn of the ring ago. Wait for THOSE, by event, and not for the stream: the stream
         * also holds the other windows' copies, which are exactly what this read is meant to hide
         * under. And for the pool's memcpys out of it. */
        auto acquire = [&](int b) -> int {
            for (size_t k = 0; k < nr; ++k)
                if (live[(size_t)b * nr + k]) {
                    RAD_TRY(rad_event_sync(ev[(size_t)b * nr + k]));
                    live[(size_t)b * nr + k] = 0;
                }
            pool->drain(b);
            return RAD_OK;
        };
        auto release = [&](int b) -> int {
            for (size_t k = 0; k < nr; ++k)
                if (touched[k]) {
                    RAD_TRY(rad_event_record(ev[(size_t)b * nr + k], st[k]));
                    live[(size_t)b * nr + k] = 1;
                    touched[k] = 0;
                }
            return RAD_OK;
        };

        const auto t0 = std::chrono::steady_clock::now();
        int64_t    last_note = 0;
        /* A LOAD THAT STOPS HAS TO SAY WHERE. This names a file offset, which `rad-info -v` turns
         * back into a weight -- the only way to localise a stall inside the sweep. Coarse by
         * default because the sweep is seconds long; RADIANCE_DEBUG_LOAD_EVERY_MIB=64 is the
         * fine-grained trail. */
        int64_t kNoteEvery = 8LL << 30;
        if (const char* e = getenv("RADIANCE_DEBUG_LOAD_EVERY_MIB")) {
            const long long v = atoll(e);
            if (v > 0) kNoteEvery = (int64_t)v << 20;
        }
        size_t i = 0;
        while (i < runs.size()) {
            /* THE GROUP IS A STRETCH OF FILE, NOT A RUN. Every run that chains to this one within
             * kGap belongs to the same read stream, however many ranks and however many weights
             * that is, and however far it reaches: a group larger than a window is read in
             * windows, not abandoned to a path of its own. */
            const int64_t g0 = runs[i].foff;
            int64_t g1 = g0 + runs[i].span;
            size_t  j  = i + 1;
            for (; j < runs.size(); ++j) {
                if (runs[j].foff - g1 > kGap) break;
                const int64_t end = runs[j].foff + runs[j].span;
                if (end > g1) g1 = end;
            }

            /* `lo` is the first run that has not finished. Sorted by offset, a run whose last
             * byte is behind the chunk is behind it for good, so the scan never goes backwards
             * and each run is looked at once per chunk it reaches into. */
            size_t lo = i;
            for (int64_t c0 = g0; c0 < g1; ) {
                const int64_t c1 = g1 < c0 + window ? g1 : c0 + window;
                RAD_TRY(acquire(cur));
                const uint8_t* win = nullptr;
                RAD_TRY(fill(cur, c0, c1, &win));
                while (lo < j && runs[lo].foff + runs[lo].span <= c0) ++lo;
                for (size_t k = lo; k < j && runs[k].foff < c1; ++k)
                    RAD_TRY(issue(runs[k], win, c0, c1, cur));
                RAD_TRY(release(cur));
                cur = (cur + 1) % kWin;
                c0 = c1;

                if (read_bytes - last_note >= kNoteEvery) {
                    last_note = read_bytes;
                    const double s = std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - t0).count();
                    RAD_INFO("load: %s of %s read, %.1f GB/s, at container offset %lld",
                             humanb(read_bytes).c_str(), humanb(want_bytes).c_str(),
                             read_bytes / (s > 0 ? s : 1) / 1e9, (long long)c0);
                }
            }
            i = j;
        }

        /* The last host synchronisation before serving: after this the step path never waits on
         * the host again (spec §5.4). */
        for (size_t k = 0; k < nr; ++k) RAD_TRY(rad_stream_sync(st[k]));
        for (int b = 0; b < kWin; ++b) pool->drain(b);

        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        /* THE SPLIT IS THE DIAGNOSTIC. If the disk time is the whole of the wall time the sweep
         * is at the drive's ceiling and there is nothing here to win; what is left over is the
         * copies that did not hide under a read -- the host-tier memcpy, which is synchronous,
         * and the per-run issue loop. */
        RAD_INFO("load: one pass over the container, %s read in %.1f s (%.1f GB/s) for %zu "
                 "rank(s); %.1f s of that was the disk (%.1f GB/s)",
                 humanb(read_bytes).c_str(), s, read_bytes / (s > 0 ? s : 1) / 1e9, nr,
                 read_s, read_bytes / (read_s > 0 ? read_s : 1) / 1e9);
        return RAD_OK;
    }();

    /* AND THE PAGE CACHE GOES BACK. Weights that will never be read from the file again are cache
     * the mapped tier cannot use: a mapped n-gram table is read from its mapping for the life of
     * the process, and it is the one thing here that wants the cache. What this frees is worth
     * more to the model than to the loader. */
    file->drop_blob_cache();
    /* NOT FREED UNDER A WORKER: a sweep that failed part-way may have left copies out of them. */
    for (int b = 0; b < kWin; ++b) pool->drain(b);
    for (int b = 0; b < kWin; ++b) if (buf[b]) rad_dev_free(buf[b], RAD_MEM_HOST_PINNED);
    for (size_t k = 0; k < ev.size(); ++k) if (ev[k]) rad_event_destroy(ev[k]);
    for (size_t k = 0; k < nr; ++k) if (st[k]) rad_stream_destroy(st[k]);
    *read_out = read_bytes;
    return rc;
}

/* One rank's share of a checkpoint weight's selected planes, read out of the mapped tensors into
 * `in` at the offsets the pool laid out: whole rows as one read, a column share a block of rows at
 * a time with the bytes it owns taken out of each. A whole plane declared under another shape (a
 * reshape) is the same bytes, so it is read in the checkpoint's own rows. */
static int fill_from_checkpoint(const CkptWeight& cw, const std::vector<PlaneShare>& shares,
                                const std::vector<int>& sel, const std::vector<int64_t>& offs,
                                const std::string& name, uint8_t* in) {
    std::vector<uint8_t> rows;
    std::string why;
    for (size_t k = 0; k < shares.size(); ++k) {
        const PlaneShare& sh = shares[k];
        const RadEncPlane& p = cw.enc.plane[sel[k]];
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, cw.rows, cw.cols, &pr, &pc);
        const int64_t crb = rad_enc_row_bytes(p.dtype, pc);
        uint8_t* at = in + offs[k];
        for (const PlaneShare::Rect& q : sh.rects) {
            if (q.byte0 == 0 && q.bytes == sh.row_bytes) {
                const int64_t off = q.row0 * sh.row_bytes, n = q.rows * sh.row_bytes;
                if (crb <= 0 || off % crb || n % crb) {
                    RAD_ERR("load: '%s': rows [%lld, %lld) of its '%s' plane do not fall on the "
                            "checkpoint's rows", name.c_str(), (long long)q.row0,
                            (long long)(q.row0 + q.rows), p.role);
                    return RAD_E_SHAPE;
                }
                if (ckpt_plane_rows(cw, sel[k], off / crb, n / crb, at, &why) != RAD_OK) {
                    RAD_ERR("load: '%s': %s", name.c_str(), why.c_str());
                    return RAD_E_FORMAT;
                }
                at += n;
                continue;
            }
            const int64_t step = std::max<int64_t>(1, (64ll << 20) / std::max<int64_t>(crb, 1));
            for (int64_t r0 = 0; r0 < q.rows; r0 += step) {
                const int64_t nr = std::min(step, q.rows - r0);
                rows.resize((size_t)(nr * crb));
                if (ckpt_plane_rows(cw, sel[k], q.row0 + r0, nr, rows.data(), &why) != RAD_OK) {
                    RAD_ERR("load: '%s': %s", name.c_str(), why.c_str());
                    return RAD_E_FORMAT;
                }
                for (int64_t r = 0; r < nr; ++r) {
                    std::memcpy(at, rows.data() + r * crb + q.byte0, (size_t)q.bytes);
                    at += q.bytes;
                }
            }
        }
    }
    return RAD_OK;
}

int Engine::load_weights() {
    int64_t total = 0, n_loaded = 0, n_missing = 0;
    /* Counted apart from `n_loaded` because it is not a load: mapped weights are published where
     * they already lie, and reporting their bytes as moved would credit the load with tens of
     * gigabytes of work it did not do. */
    int64_t n_mapped = 0, mapped_bytes = 0;

    /* THE WALK AND THE READING ARE TWO PHASES.
     *
     * A loop over RANKS, each walking the whole weight list and reading as it goes, reads the
     * container once per rank, in declaration order, one pread to a weight -- a fraction of the
     * disk's sequential rate, and more bytes read than placed, because every replicated weight is
     * read once per rank.
     *
     * So the walk below decides everything -- which rank wants which bytes, where they land,
     * what has to be refused -- and writes it down without touching the disk. sweep_container
     * then sorts what it wrote by file offset and reads the container once, front to back,
     * handing each window to every rank that asked for a piece of it. */
    std::vector<LoadRun> runs;
    std::vector<StagedWeight> staged;
    int64_t n_out = 0;      /* destination overruns reported, capped */

    /* ============================================================== the walk */
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        /* Bound before anything is allocated for this rank: rad_dev_alloc has no device argument
         * and takes whatever is current, so the mover's slabs would otherwise all land on card 0.
         * See plan() for the same bind and the same reason. */
        RAD_TRY(rad_dev_set(r->device));

        /* THE CONTAINER IS THE FILE TIER, and the mover has to be told so. `--weights-disk-tier` makes
         * the planner place units on Tier::SSD, and a bare `MoverConfig{}` then dies in
         * Mover::init with "the plan put weights on the file tier but no container path was
         * given" -- after the plan has already been made. The mover reads THIS file with O_DIRECT
         * rather than a copy of it: the weights are already
         * here, RAD_ALIGN_UNIT-aligned (rad_format.h), which is DirectFile's alignment exactly,
         * so a stage takes the no-bounce path. */
        MoverConfig mc{};
        mc.container_path = file_ ? cfg_.model : std::string();
        RAD_TRY(r->mover.init(r->program, r->plan, mc, &r->pools));
        /* WHAT THE MOVER TOOK, said before the first copy. The slabs are allocated in init, so by
         * here the pool is as full as it will get -- the copy loop writes into slots that already
         * exist and moves the cursor no further. A load that stops with the pool at zero free is
         * a placement failure wearing a loader's costume, and this line is what tells the two
         * apart from the outside. */
        RAD_INFO("load: rank %d, mover took %s, weights pool %s free of %s", r->index,
                 humanb(r->pools.weights().used()).c_str(),
                 humanb(r->pools.weights().avail()).c_str(),
                 humanb(r->pools.weights().used() + r->pools.weights().avail()).c_str());

        /* THE POLICY HALF, BOUND TO THE SAME PLAN. It returns RAD_OK on a static plan and reports
         * `enabled() == false`, which is a mode and not a failure -- all_vram and
         * --deterministic both land there -- so there is nothing to branch on here. */
        HeatParams hp;
        hp.min_residency_dispatches = 30;
        hp.readmit_penalty = 1.0f;
        if (const char* e = getenv("RADIANCE_HEAT_MOVES")) hp.moves_per_dispatch = std::atoi(e);
        if (const char* e = getenv("RADIANCE_HEAT_MIN_GAIN")) hp.min_gain = (float)std::atof(e);
        if (const char* e = getenv("RADIANCE_HEAT_HYSTERESIS")) hp.hysteresis = (float)std::atof(e);
        if (const char* e = getenv("RADIANCE_HEAT_DECAY")) hp.decay = (float)std::atof(e);
        if (const char* e = getenv("RADIANCE_HEAT_RESIDENCY")) hp.min_residency_dispatches = std::atoi(e);
        if (const char* e = getenv("RADIANCE_HEAT_READMIT_PENALTY")) hp.readmit_penalty = (float)std::atof(e);
        RAD_TRY(r->heat.init(r->plan, hp));

        /* A DESTINATION HAS TO LIE INSIDE THE POOL IT WAS CARVED FROM. A slot pointer that runs
         * past the end of the weights pool is a write to device memory the process does not own,
         * and on this driver that does not fault -- the copy is accepted, never completes, and
         * the host thread spins inside the runtime with the copy engine idle and no I/O. It looks
         * exactly like a hang, and it moves with --vram-weights-mib because a bigger pool is a
         * later overrun. Cheap enough to check on every copy, and checked before any of them
         * start rather than as each one is issued. */
        const uint8_t* wp_lo = (const uint8_t*)r->pools.weights().base();
        const uint8_t* wp_hi = wp_lo + r->pools.weights().bytes();
        auto dst_fits = [&](const void* p, int64_t len, const char* nm) {
            const uint8_t* q = (const uint8_t*)p;
            if (!wp_lo || (q >= wp_lo && q + len <= wp_hi)) return true;
            if (++n_out <= 8)
                RAD_ERR("rank %d: '%s' writes %lld bytes at %p, which is outside the %s weights "
                        "pool [%p, %p). The slot was carved from somewhere else, or past the end.",
                        r->index, nm, (long long)len, p,
                        humanb(r->pools.weights().bytes()).c_str(), (const void*)wp_lo,
                        (const void*)wp_hi);
            return false;
        };

        /* AND THE SOURCE OF A SHARE HAS TO LIE INSIDE ITS PLANE. Every offset below is computed
         * from the rank index, so rank 0 cannot exercise the arithmetic at all -- whatever the
         * shape is, it reads from the start of the plane. An offset that runs off the end is a
         * read of the wrong weight, or of nothing at all, so it is checked, not trusted. */
        const int64_t file_bytes = file_ ? file_->size() : 0;

        for (size_t i = 1; i < r->program.weights.size(); ++i) {
            WeightInfo& w = r->program.weights[i];
            /* WHERE ITS PLANES ARE: a container entry, or the checkpoint tensors declare resolved
             * it to. Either way the encoding and the logical shape the share is cut in. */
            const RadFileEntry* e = file_ ? file_->find(w.source) : nullptr;
            const CkptWeight* cw = ckpt_src_ ? ckpt_src_->find(w.source) : nullptr;
            if (!e && !cw) {
                if (w.decl.optional) continue;
                const std::string why = ckpt_src_ ? ckpt_src_->missing(w.source) : std::string();
                if (!why.empty())
                    RAD_ERR("rank %d: '%s' was declared over '%s', and %s", r->index,
                            w.name.c_str(), w.source.c_str(), why.c_str());
                else
                    RAD_ERR("rank %d: '%s' was declared over '%s', which is not in %s",
                            r->index, w.name.c_str(), w.source.c_str(), cfg_.model.c_str());
                ++n_missing;
                continue;
            }
            const RadEncoding& src_enc = e ? file_->encoding(*e) : cw->enc;
            const uint32_t src_rank = e ? e->rank : cw->rank;
            const int64_t* src_shape = e ? e->shape : cw->shape;

            /* THE ENCODING THE KERNELS WERE CHOSEN FOR IS THE ONE THE SOURCE HOLDS. Declare asked
             * it (rad_weight_encoding) and every layout hook answered for that encoding, so a
             * difference here is a declaration that was made against something else -- refused
             * by name rather than relaid as if it were the same weight. */
            if (!rad_enc_equal(&src_enc, &w.enc)) {
                char want[256], have[256];
                rad_enc_format(&w.enc, want, sizeof want);
                rad_enc_format(&src_enc, have, sizeof have);
                RAD_ERR("rank %d: '%s' was declared against '%s' as %s, and %s holds it as %s",
                        r->index, w.name.c_str(), w.source.c_str(), want, cfg_.model.c_str(),
                        have);
                return RAD_E_FORMAT;
            }

            /* THE PLANNER'S DECISION HAS TO LAND ON WeightInfo, and this is where it lands.
             *
             * WeightInfo::tier/site/slab_slot say "Filled by the planner (spec §5.3)" and the
             * planner's own finish() says it is "the component that fills these three" -- but
             * what it fills is Plan::weight[], a parallel array. Leaving the two apart is silent
             * in both directions:
             *
             *   - Ctx::bind_weights picks RAD_DOMAIN_HOST only when EVERY weight operand of an op
             *     has site == Site::Host. Left at the default Site::Device that test can never
             *     pass, so host-site execution is unreachable and the planner's whole
             *     "contiguous host runs" objective lands nowhere -- layer offload becomes dead
             *     code that still passes its tests.
             *   - a file-tier weight is skipped below on `tier == Tier::SSD` read off the PLAN,
             *     so the two spellings of the same fact would disagree right here.
             *
             * file_offset is the same story one layer down: Mover::stage reads it to pread the
             * container, and left unset every SSD stage reads offset 0 -- the file header -- for
             * every weight. It is the selected plane's, and only a weight read where it lies ever
             * uses it (weight_file_direct).
             *
             * Set before the residency lookup, not after: a file-tier weight has no address yet
             * and takes the `continue` below, and it is precisely the one that needs the offset. */
            w.tier        = r->plan.weight[i].tier;
            w.site        = r->plan.weight[i].site;
            w.slab_slot   = r->plan.weight[i].slab_slot;
            w.file_offset = e ? file_->plane(*e, w.sel[0]).offset : 0;
            const bool direct = weight_file_direct(w, cfg_.tp);
            if ((w.tier == Tier::Mapped || w.tier == Tier::SSD) && (!direct || !e)) {
                RAD_ERR("rank %d: '%s' was placed on the %s tier, which reads the container where "
                        "it lies, and its stored form is not the container's (%s)", r->index,
                        w.name.c_str(), tier_name(w.tier),
                        w.widen ? "widened at load" : w.identity ? "a shard of it"
                                                                 : w.layout_tag.c_str());
                return RAD_E_STATE;
            }

            /* THE MAPPED TIER IS A PUBLISH AND NOT A LOAD. The bytes are already at a host
             * address -- the container is mmapped and stays mapped for the life of the engine
             * (core/format/radfile.cpp) -- so there is nothing to copy and no pool to copy into.
             * This is the only weight class the load phase does not move, and the sweep below
             * never reads a byte of it -- which on a container with a large mapped table is most
             * of the file.
             *
             * IT HAS TO GO THROUGH THE RESIDENCY TABLE and not only onto WeightInfo::ptr, because
             * Ctx::resolve_weight reads the table whenever a mover exists and would otherwise
             * find the null the mover published for a unit with no slab class. */
            if (w.tier == Tier::Mapped) {
                const RadFilePlane& p0 = file_->plane(*e, w.sel[0]);
                const int64_t n = (int64_t)p0.bytes;
                if (n != w.stored_bytes) {
                    RAD_ERR("'%s': the container's '%s' plane has %lld bytes, the declaration "
                            "wants %lld", w.name.c_str(), file_->encoding(*e).plane[w.sel[0]].role,
                            (long long)n, (long long)w.stored_bytes);
                    return RAD_E_FORMAT;
                }
                w.ptr = (void*)file_->data(p0);
                w.host_ptr = w.ptr;
                /* --ngram-placement ram: the gather reads a copy instead of the mapping -- pinned
                 * memory a device kernel reads in place over the link, where the planner put the
                 * table on the zero-copy site, or locked memory for the host gather. A copy auto
                 * chose and could not make leaves the table in the container and says so, and a
                 * card's gather moves to the host with it; one the operator asked for refuses the
                 * start. */
                if (cfg_.ngram_resolved == TablePlacement::Ram &&
                    is_ngram_table(r->program, (rad_weight)i)) {
                    const bool card = w.site == Site::DeviceZeroCopy;
                    void* copy = nullptr;
                    std::string why;
                    const int cs = copy_table((const uint8_t*)w.ptr, n, (int64_t)p0.offset, card,
                                              &copy, &why);
                    if (cs == RAD_OK) {
                        void* on_card = card ? rad_dev_device_ptr(copy) : nullptr;
                        w.ptr = on_card ? on_card : copy;
                        w.host_ptr = copy;
                        r->mover.publish_mapped((rad_weight)i, w.ptr);
                        continue;   /* reported once, by copy_table, and not as mapped */
                    }
                    const char* into = card ? "pinned" : "locked";
                    if (cfg_.ngram_placement == TablePlacement::Ram) {
                        RAD_ERR("--ngram-placement ram: '%s' could not be copied into %s memory: "
                                "%s", w.name.c_str(), into, why.c_str());
                        return cs;
                    }
                    if (card && !ngram_gather_to_host()) {
                        RAD_ERR("--ngram-placement auto: '%s' could not be copied into pinned "
                                "memory (%s), and no loaded kernel plugin gathers it on the host, "
                                "which is what reads it from the container", w.name.c_str(),
                                why.c_str());
                        return cs;
                    }
                    RAD_WARN("--ngram-placement auto: '%s' could not be copied into %s memory "
                             "(%s); it is read from the container%s", w.name.c_str(), into,
                             why.c_str(), card ? " by the host gather" : "");
                    cfg_.ngram_resolved = TablePlacement::Disk;
                }
                /* NO READAHEAD ON A MAPPED WEIGHT. What reads one is a host kernel gathering rows
                 * out of it, and the only reason a weight is on this tier at all is that it is far
                 * too big to be resident -- so a fault is a real read from the file and the pages
                 * around it are pages nobody asked for. Linux's default 128 KiB readahead turns a
                 * row of a few hundred bytes into 32 pages of I/O, and at a dozen rows a token
                 * that is megabytes read to use kilobytes. MADV_RANDOM is the whole of the
                 * correction and it is per range, so the rest of the container keeps the
                 * sequential behaviour the load pass wants.
                 *
                 * IT SHOWS NOTHING ON A WARM PAGE CACHE. When the hot part of the table is
                 * already resident the serving process reads nothing from disk and the advice
                 * changes no timing at all; the regime it protects -- a table genuinely colder
                 * than RAM -- is the regime this tier exists for, and it costs one syscall at
                 * load. What shows its effect is a cold cache and a byte counter, not a
                 * stopwatch. */
                posix_madvise(w.ptr, (size_t)n, POSIX_MADV_RANDOM);
                if (w.site == Site::DeviceZeroCopy || w.site == Site::Device) {
                    /* A DEVICE KERNEL READS THIS TABLE, so it needs a GPU address. NOT by
                     * hipHostRegister on the file mapping: that pins page-cache pages as a KFD
                     * userptr, every page-cache eviction during the 65 GiB load sweep then fires
                     * an MMU-notifier eviction of the whole process's queues, and the copy queues
                     * stalled indefinitely (measured: aql_copy_link stuck on both cards).
                     *
                     * One anonymous pinned, device-mapped copy instead, made ONCE and shared by
                     * every rank -- they run in this process, and host memory is visible to every
                     * card -- so the resident footprint is one table, not one per rank. The
                     * page-cache pages the copy was read from are clean and reclaimable. */
                    static std::mutex mtx;
                    static std::map<const void*, void*> shared;   /* host source -> pinned copy */
                    std::lock_guard<std::mutex> lk(mtx);
                    void*& dst = shared[w.ptr];
                    if (!dst) {
                        const char* shm_env = std::getenv("RADIANCE_PLE_SHM");
                        const bool enable_shm = !shm_env || (std::strcmp(shm_env, "0") != 0 &&
                                                             std::strcmp(shm_env, "off") != 0 &&
                                                             std::strcmp(shm_env, "false") != 0);
                        std::string shm_id;
                        for (char c : w.name) {
                            if (std::isalnum((unsigned char)c) || c == '.' || c == '_') shm_id += c;
                            else shm_id += '_';
                        }
                        uint64_t model_hash = 14695981039346656037ULL;
                        for (char c : cfg_.model) {
                            model_hash ^= (uint8_t)c;
                            model_hash *= 1099511628211ULL;
                        }
                        char hash_str[32];
                        std::snprintf(hash_str, sizeof(hash_str), "%016llx", (unsigned long long)model_hash);
                        std::string bin_path = "/dev/shm/rad_shm_" + shm_id + "_" + hash_str + "_" + std::to_string(n) + ".bin";
                        std::string ready_path = bin_path + ".ready";

                        if (enable_shm) {
                            struct stat st_ready{}, st_bin{};
                            if (stat(ready_path.c_str(), &st_ready) == 0 && stat(bin_path.c_str(), &st_bin) == 0) {
                                if (st_bin.st_size == (off_t)n) {
                                    int rfd = open(ready_path.c_str(), O_RDONLY);
                                    if (rfd >= 0) {
                                        int64_t ready_size = 0;
                                        if (read(rfd, &ready_size, sizeof(ready_size)) == sizeof(ready_size) &&
                                            ready_size == n) {
                                            const auto t0 = std::chrono::steady_clock::now();
                                            int fd = open(bin_path.c_str(), O_RDWR);
                                            if (fd >= 0) {
                                                void* p = mmap(nullptr, (size_t)n, PROT_READ | PROT_WRITE,
                                                               MAP_SHARED, fd, 0);
                                                close(fd);
                                                if (p != MAP_FAILED) {
                                                    madvise(p, (size_t)n, MADV_HUGEPAGE);
                                                    int reg_err = rad_dev_host_register(p, n, 0);
                                                    if (reg_err == RAD_OK) {
                                                        dst = p;
                                                        RAD_INFO("load: '%s' attached from persistent shm cache in %.2f s (instant restart), shared by every rank",
                                                                 w.name.c_str(),
                                                                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                                                    } else {
                                                        RAD_WARN("load: '%s' persistent shm host_register failed (%d), falling back",
                                                                 w.name.c_str(), reg_err);
                                                        munmap(p, (size_t)n);
                                                    }
                                                }
                                            }
                                        }
                                        close(rfd);
                                    }
                                }
                            }
                        }

                        if (!dst && enable_shm) {
                            struct statvfs svfs{};
                            if (statvfs("/dev/shm", &svfs) == 0) {
                                uint64_t free_shm = (uint64_t)svfs.f_bavail * svfs.f_bsize;
                                if (free_shm >= (uint64_t)n + 1024ULL * 1024ULL * 1024ULL) {
                                    std::string tmp_path = bin_path + ".tmp";
                                    unlink(tmp_path.c_str());
                                    int fd = open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
                                    if (fd >= 0) {
                                        if (ftruncate(fd, (off_t)n) == 0) {
                                            void* p = mmap(nullptr, (size_t)n, PROT_READ | PROT_WRITE,
                                                           MAP_SHARED, fd, 0);
                                            close(fd);
                                            if (p != MAP_FAILED) {
                                                madvise(p, (size_t)n, MADV_HUGEPAGE);
                                                const auto t0 = std::chrono::steady_clock::now();
                                                const int nt = 16;
                                                const int64_t chunk = align_up((n + nt - 1) / nt, (int64_t)1 << 21);
                                                std::vector<std::thread> th;
                                                const uint8_t* src = (const uint8_t*)w.ptr;
                                                uint8_t* out = (uint8_t*)p;
                                                for (int k = 0; k < nt; ++k) {
                                                    const int64_t lo = k * chunk, hi = std::min<int64_t>(n, lo + chunk);
                                                    if (lo >= hi) break;
                                                    th.emplace_back([src, out, lo, hi] {
                                                        std::memcpy(out + lo, src + lo, (size_t)(hi - lo));
                                                    });
                                                }
                                                for (auto& t : th) t.join();

                                                rename(tmp_path.c_str(), bin_path.c_str());
                                                int rfd = open(ready_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
                                                if (rfd >= 0) {
                                                    write(rfd, &n, sizeof(n));
                                                    close(rfd);
                                                }

                                                int reg_err = rad_dev_host_register(p, n, 0);
                                                if (reg_err == RAD_OK) {
                                                    dst = p;
                                                    RAD_INFO("load: '%s' populated %s into persistent shm cache in %.1f s, shared by every rank",
                                                             w.name.c_str(), humanb(n).c_str(),
                                                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                                                } else {
                                                    RAD_WARN("load: '%s' persistent shm host_register failed (%d), falling back",
                                                             w.name.c_str(), reg_err);
                                                    munmap(p, (size_t)n);
                                                }
                                            }
                                        } else {
                                            close(fd);
                                            unlink(tmp_path.c_str());
                                        }
                                    }
                                }
                            }
                        }

                        if (!dst) {
                            dst = rad_dev_alloc(n, RAD_MEM_HOST_MAPPED);
                            if (!dst) {
                                RAD_ERR("'%s': no pinned host memory for the %s device-read table: %s",
                                        w.name.c_str(), humanb(n).c_str(), rad_dev_last_error());
                                return RAD_E_NOMEM;
                            }
                            const auto t0 = std::chrono::steady_clock::now();
                            const int nt = 16;
                            const int64_t chunk = align_up((n + nt - 1) / nt, (int64_t)1 << 21);
                            std::vector<std::thread> th;
                            const uint8_t* src = (const uint8_t*)w.ptr;
                            uint8_t* out = (uint8_t*)dst;
                            for (int k = 0; k < nt; ++k) {
                                const int64_t lo = k * chunk, hi = std::min<int64_t>(n, lo + chunk);
                                if (lo >= hi) break;
                                th.emplace_back([src, out, lo, hi] {
                                    std::memcpy(out + lo, src + lo, (size_t)(hi - lo));
                                });
                            }
                            for (auto& t : th) t.join();
                            RAD_INFO("load: '%s' copied into %s of pinned host memory for device "
                                     "gathers in %.1f s, shared by every rank", w.name.c_str(),
                                     humanb(n).c_str(),
                                     std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                                   t0).count());
                        }
                    }
                    void* dev_p = rad_dev_device_ptr(dst);
                    w.host_ptr = dst;
                    w.ptr = dev_p ? dev_p : dst;
                }
                r->mover.publish_mapped((rad_weight)i, w.ptr);
                ++n_mapped;
                mapped_bytes += n;
                continue;
            }

            const WeightSlot* slot = r->mover.residency()->lookup((rad_weight)i);
            if (!slot || !slot->ptr) {
                /* A file-tier unit has no address until the mover stages it, which is correct and
                 * not an error: RAD_ACCESS_RARE weights are read on demand (spec §5.2). */
                if (w.tier == Tier::SSD) continue;
                RAD_ERR("rank %d: '%s' has no address after mover init", r->index, w.name.c_str());
                return RAD_E_STATE;
            }

            /* THE ADDRESS HAS TO LAND ON WeightInfo, not only in the residency table. In the
             * all-in-VRAM case Ctx is built with a NULL residency -- the mover never runs, so
             * there is nothing to look up per issue -- and `Ctx::resolve_weight` collapses to
             * `WeightInfo::ptr` (core/runtime/issue.cpp says exactly that). Leaving it null
             * leaves every weight in the program unbound at issue, which the run phase reports as
             * "weight 'x' has no pointer at issue: the load phase never bound it" -- the load
             * having copied the whole model into addresses nothing afterwards can name.
             * `generation` is not bumped because `bind_weights` runs for the first time after
             * this, in Ctx::init. */
            w.ptr = slot->ptr;

            /* HOST-POOL DESTINATIONS STAY A memcpy. The host pool is pinned and device-mapped
             * (core/mem/pools.cpp), so the copy engine CAN write it, and the synchronous
             * host-tier memcpy is most of what separates this sweep from the disk's own rate --
             * which makes putting those copies on the stream look as though it would hide them
             * under the next read. It does not: an async host-to-host copy is not a DMA on this
             * stack, it is the same memcpy with a stream in front of it. */
            const bool dev = w.tier == Tier::VRAM;
            if (dev && !dst_fits(slot->ptr, w.stored_bytes, w.name.c_str())) return RAD_E_STATE;

            /* THIS RANK'S SHARE OF THE SELECTED PLANES, cut on the canonical form where a ROW
             * shard is rows of every plane and a COL shard is columns -- each on the plane's own
             * block boundary, and refused where a block or a byte would be split. */
            PlaneShare sh[RAD_ENC_MAX_PLANES];
            std::string why;
            if (weight_share(w, src_enc, src_rank, src_shape, cfg_.tp, r->index, sh, &why) !=
                RAD_OK) {
                RAD_ERR("rank %d: '%s' at --tp %d: %s", r->index, w.name.c_str(), cfg_.tp,
                        why.c_str());
                return RAD_E_SHAPE;
            }

            /* FROM A CHECKPOINT, every weight is staged and its worker reads it: there is no
             * sweep to deliver it, and the stored form is whatever the pool makes of the share --
             * as it is, widened, or relaid. */
            if (!e) {
                StagedWeight sw;
                sw.prog = &r->program;
                sw.wi = i;
                sw.device = r->device;
                sw.dst = (uint8_t*)slot->ptr;
                sw.dev = dev;
                int64_t in_bytes = 0;
                for (int k = 0; k < w.n_sel; ++k) {
                    sw.plane_off[k] = in_bytes;
                    in_bytes += align_up(sh[k].dense_bytes(), 64);
                    sw.remaining += sh[k].dense_bytes();
                    total += sh[k].dense_bytes();
                }
                sw.in_bytes = in_bytes;
                if (w.identity && !w.widen && sh[0].dense_bytes() != w.stored_bytes) {
                    RAD_ERR("rank %d: '%s': this rank's share is %lld bytes and the slot %lld",
                            r->index, w.name.c_str(), (long long)sh[0].dense_bytes(),
                            (long long)w.stored_bytes);
                    return RAD_E_FORMAT;
                }
                std::vector<PlaneShare> shares(sh, sh + w.n_sel);
                std::vector<int> sel(w.sel, w.sel + w.n_sel);
                std::vector<int64_t> offs(sw.plane_off, sw.plane_off + w.n_sel);
                const std::string name = w.name;
                sw.fill = [cw, shares, sel, offs, name](uint8_t* in) -> int {
                    return fill_from_checkpoint(*cw, shares, sel, offs, name, in);
                };
                staged.push_back(std::move(sw));
                ++n_loaded;
                continue;
            }

            /* STORED AS IT IS: the share goes from the file to the slot. Otherwise it is staged
             * whole for the relayout pool, plane after plane, and the pool writes the slot. */
            int32_t stage = -1;
            int64_t in_bytes = 0;
            if (!w.identity || w.widen) {
                stage = (int32_t)staged.size();
                staged.emplace_back();
                StagedWeight& sw = staged.back();
                sw.prog = &r->program;
                sw.wi = i;
                sw.device = r->device;
                sw.dst = (uint8_t*)slot->ptr;
                sw.dev = dev;
                for (int k = 0; k < w.n_sel; ++k) {
                    sw.plane_off[k] = in_bytes;
                    in_bytes += align_up(sh[k].dense_bytes(), 64);
                }
                sw.in_bytes = in_bytes;
                sw.remaining = 0;
                for (int k = 0; k < w.n_sel; ++k) sw.remaining += sh[k].dense_bytes();
            } else if (sh[0].dense_bytes() != w.stored_bytes) {
                RAD_ERR("rank %d: '%s': this rank's share is %lld bytes and the slot %lld",
                        r->index, w.name.c_str(), (long long)sh[0].dense_bytes(),
                        (long long)w.stored_bytes);
                return RAD_E_FORMAT;
            }
            for (int k = 0; k < w.n_sel; ++k) {
                const RadFilePlane& pk = file_->plane(*e, w.sel[k]);
                int64_t doff = stage >= 0 ? staged[(size_t)stage].plane_off[k] : 0;
                for (const PlaneShare::Rect& q : sh[k].rects) {
                    const int64_t foff = (int64_t)pk.offset + q.row0 * sh[k].row_bytes + q.byte0;
                    const int64_t len = (q.rows - 1) * sh[k].row_bytes + q.bytes;
                    if (q.rows <= 0) continue;
                    if (foff < 0 || foff + len > file_bytes ||
                        foff + len > (int64_t)(pk.offset + pk.bytes)) {
                        RAD_ERR("rank %d: '%s' would read [%lld, %lld), outside its '%s' plane "
                                "[%llu, %llu)", r->index, w.name.c_str(), (long long)foff,
                                (long long)(foff + len), file_->encoding(*e).plane[w.sel[k]].role,
                                (unsigned long long)pk.offset,
                                (unsigned long long)(pk.offset + pk.bytes));
                        return RAD_E_FORMAT;
                    }
                    uint8_t* dst = stage >= 0 ? (uint8_t*)(intptr_t)doff : (uint8_t*)slot->ptr + doff;
                    const bool dense = q.byte0 == 0 && q.bytes == sh[k].row_bytes;
                    if (dense)
                        runs.push_back(LoadRun{ foff, q.rows * q.bytes, dst, q.rows * q.bytes,
                                                q.rows * q.bytes, 1, r->index,
                                                stage >= 0 ? 0 : (dev ? 1 : 0), stage });
                    else
                        runs.push_back(LoadRun{ foff, len, dst, q.bytes, sh[k].row_bytes, q.rows,
                                                r->index, stage >= 0 ? 0 : (dev ? 1 : 0), stage });
                    total += q.rows * q.bytes;
                    doff += q.rows * q.bytes;
                }
            }
            ++n_loaded;
        }
    }

    if (n_missing) {
        RAD_ERR("%lld declared weights are absent from the container. Reported whole rather than "
                "first-failure-at-the-top; the model cannot run.", (long long)n_missing);
        return RAD_E_NOTFOUND;
    }

    /* ============================================================== the sweep */
    std::vector<int> devices;
    devices.reserve(ranks_.size());
    for (auto& r : ranks_) devices.push_back(r->device);
    int64_t read_bytes = 0;
    int64_t staged_bytes = 0;
    for (const StagedWeight& sw : staged) staged_bytes += sw.in_bytes;
    RelayoutPool pool(staged);
    const unsigned hw = std::thread::hardware_concurrency();
    /* Started whatever is staged: a host pool slot is filled by the same workers. */
    const int workers = (int)std::max(2u, std::min(16u, hw > 2 ? hw - 2 : 2u));
    pool.start(workers, std::max<int64_t>(4ll << 30, 0));
    const auto rt0 = std::chrono::steady_clock::now();
    int sw_st = RAD_OK;
    if (file_) {
        sw_st = sweep_container(file_.get(), runs, devices, &pool, &read_bytes);
    } else {
        /* every weight to the workers, in declaration order, as fast as the budget lets them in */
        for (size_t k = 0; k < staged.size() && !pool.failed(); ++k) {
            const int64_t n = staged[k].remaining;
            pool.reserve((int)k);
            pool.arrived((int)k, n);
            read_bytes += n;
        }
    }
    const int pool_st = pool.finish();
    RAD_TRY(sw_st);
    RAD_TRY(pool_st);
    if (!staged.empty() && file_)
        RAD_INFO("load: %zu weight(s), %s of planes, rearranged for their kernels by %d worker(s) "
                 "behind the read, %s held at most; done %.1f s after the read began",
                 staged.size(), humanb(staged_bytes).c_str(), workers, humanb(pool.peak()).c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - rt0).count());
    else if (!staged.empty())
        RAD_INFO("load: %zu weight(s), %s, read out of the checkpoint and laid out for their "
                 "kernels by %d worker(s), %s held at most, in %.1f s", staged.size(),
                 humanb(staged_bytes).c_str(), workers, humanb(pool.peak()).c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - rt0).count());

    RAD_INFO("load: %lld weights, %s placed from %s of %s read", (long long)n_loaded,
             humanb(total).c_str(), humanb(read_bytes).c_str(),
             file_ ? "container" : "checkpoint");
    if (n_mapped)
        /* THE BYTES ARE SHARED BETWEEN RANKS and the counters are not: every rank maps the same
         * file, so at --tp 2 this counts one mapped table twice. Said rather than halved,
         * because what the number IS is the sum of the mappings -- and an operator reading it as
         * memory would size a box for twice what the model needs. */
        RAD_INFO("load: %lld weight(s), %s, read in place from the container's mapping -- not "
                 "copied and charged to no pool%s", (long long)n_mapped,
                 humanb(mapped_bytes).c_str(),
                 cfg_.tp > 1 ? " (summed over ranks, which map the same file: the resident bytes "
                               "are that divided by --tp)" : "");
    return RAD_OK;
}

/* ================================================================== 6b. KV pools */
/* The block manager owns every KV group's memory -- paged attention caches, the GDN recurrent
 * states and the conv windows alike -- because two allocators disagreeing about what a sequence
 * owns is not a leak but the bug where a freed sequence's state slot is handed to the next
 * request while the first is still reading it (spec §7.2).
 *
 * This runs after load, not before, because it takes what the weight pool LEFT. It has to be
 * before Ctx::init: a Ctx built without these bindings resolves every RAD_OPK_KV operand to null,
 * and the first attention kernel of the first step dereferences it. */
int Engine::configure_kv() {
    /* THE ANCHOR IS A PROPERTY OF THE POOL, SO THE POOL'S OWNER TURNS IT ON. libr4d's recurrent
     * update can keep one committed state a sequence and replay the accepted prefix onto it out
     * of a scratch slot, instead of writing a full state per candidate token -- 2 slots a
     * sequence against 1 + n_spec, which at a realistic depth is several times less. It is off by
     * default because it is NOT PAGED: it assumes a sequence's state stays in the slot it was
     * given, which is true of radiance's linear pool and is not true of vLLM's, and the two
     * callers share the kernel. Set before any kernel runs; libr4d reads it once.
     *
     * The engine's half of the bargain is KVGroupPlan::state_copies (core/mem/kv.cpp): two slots
     * a sequence, both zeroed on admission, named by a state_index row wide enough to reach
     * them. Without speculation there is nothing to defer and the plain path is both correct and
     * one slot cheaper; a tree-bound group (KVGroupInfo::tree_bound, set in declare) never reaches
     * this kernel at decode and keeps one slot too. */
    /* Overwrite 0, so an operator can still set it -- but the two settings are NOT both runnable at
     * every depth. Per-candidate indexing needs `1 + n_spec` distinct slots and this pool has two,
     * so at a draft depth above one it gives several candidate tokens the same slot: a partially
     * accepted step then resumes from a state one or more tokens too far along, silently, in a
     * cache nothing reads back. The arch declares `state_form` on the op from the same condition
     * below, and libr4d refuses the form that disagrees with the declaration -- so an override that
     * would corrupt is a selection error naming the kernel and the geometry, rather than a
     * generation that drifts. */
    if (cfg_.n_spec > 0) setenv("RADIANCE_GDN_ANCHOR", "1", 0);
    /* Resolved once, for the cache below and for Scheduler::init later. Resolving it separately in
     * each is how the two come to disagree, and the interval is the field where that shows. */
    RAD_TRY(chunk_geometry_resolve(ranks_[0]->program, cfg_, &geo_));
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RAD_TRY(rad_dev_set(r->device));   /* allocates on the CURRENT device; see plan() */
        RAD_TRY(r->kv.configure(r->program.kv_groups, cfg_, r->pools.kv()));
        /* THE ONE CASE WHERE THE CACHE CANNOT GIVE ANYTHING BACK, said once and named. Every
         * caller is written against the elastic contract and runs unchanged here -- the pool
         * simply holds its whole carve -- so nothing fails, and the only symptom is an expert
         * plane that never grows and a split that stays where --expert-vs-cache-ratio put it.
         * That is worth a line, because it is indistinguishable from a working run in every
         * counter this engine reports. */
        if (r->index == 0 && !r->pools.kv().elastic())
            RAD_WARN("this device has no virtual memory management, so the KV cache holds its "
                     "whole carve and the expert slab has nothing to grow into. The split stays "
                     "where --expert-vs-cache-ratio put it.");
        if (cfg_.debug_placement) fputs(r->kv.report().c_str(), stderr);
        RAD_TRY(r->prefix.configure(cfg_, &r->kv, geo_.checkpoint_interval));
    }

    /* ---------------- the idle session tiers (spec §7.3) ---------------- */
    /* AFTER the prefix cache, because the tiers are sized in units of ONE CACHE ENTRY and only the
     * configured cache knows which groups an entry covers. No memory is committed unless an
     * operator gave a tier a size: a tier of zero bytes holds nothing, so the two inactivity
     * thresholds are ordinary defaults rather than a commitment of memory nobody asked to spend.
     *
     * CONFIGURED EVEN SO, because this object is also where a stored conversation is RECORDED,
     * and that is not a tiering fact. A server with no --prefix-cache-host-mib holds its whole prefix
     * cache in VRAM and holds it for exactly as long; skipping the configure would leave the
     * sessions view reading an unconfigured object and reporting an empty server, on every
     * deployment that has not opted into demotion. What the flags gate is the MOVER, below. */
    {
        const bool tier_asked = cfg_.prefix_cache_host_mib > 0 || cfg_.prefix_cache_disk_mib > 0;
        const std::vector<int32_t>& cg = ranks_[0]->prefix.cached_groups();
        int64_t entry_bytes = 0;
        for (int32_t g : cg) {
            const KVGroupPlan* p = ranks_[0]->kv.plan(g);
            if (p && p->paged()) entry_bytes += p->bytes_per_block;
        }
        /* ONE SLOT HOLDS EVERY RANK'S SHARD. An entry under tensor parallelism is n_ranks
         * pieces of KV that are not interchangeable, so the unit the tiers store and budget
         * in is the whole thing; each rank writes only its own window of it. */
        const int64_t slot_bytes = entry_bytes * (int64_t)ranks_.size();

        /* ---- how the tier is split between the two halves of a session ----
         *
         * ON THIS BUILD A RESTORED HYBRID SESSION WITH NO SNAPSHOT REUSES NOTHING AT ALL. The
         * scheduler clamps its reuse to the last reachable checkpoint and says why: RadBatch
         * carries one query length for every KV group, so the linear layers cannot be replayed
         * over a longer range than the attention layers, and the correct answer is to recompute
         * from the checkpoint (core/sched/scheduler.cpp, admit_one). A session whose blocks came
         * back and whose state did not therefore gets a hit of ZERO -- the blocks are not a
         * partial win, they are unusable -- and the snapshot store is not a nice-to-have beside
         * the block store, it is the thing that decides whether any of this works.
         *
         * So the split is DERIVED, from the session the operator sized the server for. At
         * --max-model-len M a session costs
         *
         *     blocks     (M / block_size) * slot_bytes
         *     snapshots  (log2(M / interval) + 1) * ck_slot_bytes     [geometric backoff]
         *
         * and splitting the cap in that ratio makes both halves hold the same NUMBER OF SESSIONS,
         * which is the only sense in which they can be balanced. The ratio is strongly length
         * dependent -- snapshots want 22% of the tier at 200K and more than half at 32K, because
         * the retained set grows logarithmically while the blocks grow linearly -- which is
         * exactly why a single fixed fraction is the wrong shape.
         *
         * Two thirds is the ceiling. Not because the arithmetic asks for one, but because the
         * clamp above is a property of THIS build rather than of the design, and a tier with
         * almost no room for blocks would be badly sized the day it is lifted. */
        const int64_t ck_slot_bytes = ranks_[0]->kv.checkpoint_bytes() * (int64_t)ranks_.size();
        int64_t ck_host_cap = 0, ck_disk_slots = 0;
        if (ck_slot_bytes > 0) {
            const int64_t bs = ranks_[0]->prefix.block_size();
            const int64_t iv = ranks_[0]->prefix.checkpoint_interval();
            /* The session the server was sized for: --max-model-len, or the training context
             * when it is unset -- the same reading the declare and the KV pool take. */
            const int64_t mlen = std::max<int64_t>(
                1, cfg_.max_ctx > 0 ? cfg_.max_ctx : meta_.n_ctx_train);
            int64_t retained = 1;
            for (int64_t n = (iv > 0 ? mlen / iv : 0); n > 1; n >>= 1) ++retained;
            const int64_t sess_blocks = bs > 0 ? (mlen / bs) * slot_bytes : slot_bytes;
            const int64_t sess_snaps  = retained * ck_slot_bytes;
            const int64_t denom = sess_blocks + sess_snaps;
            const int64_t host_cap = cfg_.prefix_cache_host_mib << 20;
            const int64_t disk_cap = cfg_.prefix_cache_disk_mib << 20;
            const int64_t host_ck  = std::min(host_cap * 2 / 3, denom > 0 ? host_cap / denom * sess_snaps
                                                                            + host_cap % denom * sess_snaps / denom
                                                                          : 0);
            const int64_t disk_ck  = std::min(disk_cap * 2 / 3, denom > 0 ? disk_cap / denom * sess_snaps
                                                                            + disk_cap % denom * sess_snaps / denom
                                                                          : 0);
            ck_host_cap   = (host_ck / ck_slot_bytes) * ck_slot_bytes;
            ck_disk_slots = disk_ck / ck_slot_bytes;
            /* One of each where there is room for one at all: a tier that holds a session's blocks
             * and not one snapshot holds nothing usable. */
            if (ck_host_cap == 0 && host_cap >= ck_slot_bytes * 2) ck_host_cap = ck_slot_bytes;
            if (ck_disk_slots == 0 && disk_cap >= ck_slot_bytes * 2) ck_disk_slots = 1;
        }
        if (cg.empty() || entry_bytes <= 0) {
            /* Said only to an operator who asked for a tier. With no flag there is nothing to
             * report: a cache that indexes no paged group stores no blocks to begin with. */
            if (tier_asked)
                RAD_WARN("kv tiers: the prefix cache indexes no paged group, so there is nothing "
                         "to move. --prefix-cache-host-mib and --prefix-cache-disk-mib do nothing here.");
        } else {
            if (cfg_.prefix_cache_disk_mib > 0) {
                if (cfg_.prefix_cache_dir.empty()) {
                    RAD_WARN("kv tiers: --prefix-cache-disk-mib needs --prefix-cache-dir to say WHERE. "
                             "The disk tier is off.");
                } else {
                    SSDTierConfig sc;
                    sc.dir = cfg_.prefix_cache_dir;
                    sc.block_payload = slot_bytes;
                    sc.block_tokens  = ranks_[0]->prefix.block_size();
                    sc.ckpt_payload  = ck_slot_bytes;
                    sc.n_ckpt_slots  = ck_disk_slots;
                    sc.n_block_slots = ((cfg_.prefix_cache_disk_mib << 20)
                                        - ck_disk_slots * ck_slot_bytes) / slot_bytes;
                    /* THE FINGERPRINT IS WHAT MAKES A PERSISTED STORE SAFE. A cached block is only
                     * meaningful under the container and layout that produced it; read back under
                     * another, it is a lossless function of somebody else's weights. */
                    /* THE FINGERPRINT IS WHAT MAKES A PERSISTED STORE SAFE, and it is built from
                     * everything a cached block's bytes actually depend on. Read back under a
                     * different container, layout or KV width, a block is a lossless function of
                     * somebody else's weights -- fluent, confident and wrong. There is no cheaper
                     * check than refusing the whole store, because a block carries no evidence of
                     * what produced it beyond the tokens it was keyed by. */
                    {
                        const std::string id =
                            std::string(meta_.name ? meta_.name : "") + "|" +
                            std::string(meta_.arch_id ? meta_.arch_id : "") + "|" +
                            std::to_string(meta_.n_layers) + "|" +
                            std::to_string(ranks_[0]->prefix.block_size()) + "|" +
                            std::to_string(slot_bytes) + "|" + std::to_string(ck_slot_bytes) + "|" +
                            cfg_.kv_cache_dtype + "|" +
                            std::to_string((long long)ranks_.size()) + "|" + RAD_VERSION;
                        uint64_t h = 1469598103934665603ull;
                        for (char c : id) { h ^= (unsigned char)c; h *= 1099511628211ull; }
                        sc.fingerprint = h;
                    }
                    sc.persistent  = true;
                    kv_disk_ = std::make_unique<SSDTier>();
                    RAD_INFO("%s", SSDTier::disclosure(sc.dir).c_str());
                    if (kv_disk_->open(sc) != RAD_OK) {
                        RAD_WARN("kv tiers: the disk tier at %s would not open; it is off.",
                                 sc.dir.c_str());
                        kv_disk_.reset();
                    }
                }
            }
            IdleTiers::Config tc;
            tc.host_cap_bytes  = cfg_.prefix_cache_host_mib << 20;
            tc.disk_cap_bytes  = kv_disk_ ? (cfg_.prefix_cache_disk_mib << 20) : 0;
            tc.entry_bytes     = slot_bytes;
            tc.ckpt_bytes      = ck_slot_bytes;
            tc.ckpt_cap_bytes  = ck_host_cap;
            tc.ckpt_disk_slots = kv_disk_ ? ck_disk_slots : 0;

            /* THE POLICY IS RANK 0'S AND THE TRANSFER IS EVERY RANK'S. A session is not sharded --
             * it is a conversation -- so one decision is taken about it; but each rank holds its
             * own shard of every block and gathers out of its own pool, so each rank moves its own
             * bytes when that decision is taken. */
            IdleTiers& T = ranks_[0]->tiers;
            RAD_TRY(T.configure(tc, kv_disk_.get()));
            T.set_model(meta_.name && meta_.name[0] ? meta_.name : "(unnamed)");
            if (T.any_on()) {
                /* HOW MUCH A PASS MOVES, SIZED IN BYTES AND NOT IN ENTRIES. A cache entry is one
                 * block per cached group, and that is 29 KiB on a 4-token-block hybrid and four
                 * times more on a 16-token one -- so a fixed entry count means a fixed batch on
                 * one model and a quarter of one on another. At a fixed 64 entries a 200K session
                 * on a 4-token-block hybrid takes about thirteen minutes to leave VRAM, which is
                 * most of the way to the disk threshold: a tier that works and is too slow to be
                 * worth having.
                 *
                 * 16 MiB of device staging a rank, which is about a millisecond of PCIe a pass
                 * and 0.02 ms a step of the VRAM it costs. The batch is one gather per (group,
                 * layer) however many entries are in it, so the launches are the same either
                 * way -- this buys the copies, which are what there are thousands of. */
                /* A COPY OUT IS FOUR STAGING BATCHES, about 64 MiB a rank. It shares the transfer
                 * stream with the expert mover, so what it queues is what a promotion issued
                 * behind it waits for -- a few milliseconds of link at this size -- and the next
                 * copy goes out on the tick after this one lands, so a long conversation still
                 * leaves at the rate the link carries it.
                 *
                 * A RESTORE IS A WHOLE CHAIN, as long as the longest context this server takes.
                 * TierExec::sizes holds the arithmetic, which the VRAM budget reserves from. */
                const TierExec::Sizes tz = TierExec::sizes(
                    entry_bytes, cfg_.max_ctx > 0 ? cfg_.max_ctx : meta_.n_ctx_train,
                    ranks_[0]->prefix.block_size());
                const int max_batch = tz.max_batch;
                const int64_t max_copy = tz.max_copy, max_chain = tz.max_chain;
                bool all = true;
                for (auto& r : ranks_) {
                    /* ON THAT RANK'S DEVICE. rad_dev_alloc takes the CURRENT device, and the
                     * staging buffer TierExec builds on its first pass is this rank's. The device
                     * is recorded in configure() and re-selected there, but selecting it here too
                     * keeps the rule that a rank is configured on its own card in one place: a
                     * gather reading a pointer that belongs to another card fails as a plain
                     * device error with nothing to say it was an addressing mistake.
                     *
                     * AND THE STREAM IS THE MOVER'S. See Mover::transfer_stream -- a fourth
                     * hardware queue costs an RDNA4 card 65% of its decode step. */
                    RAD_TRY(rad_dev_set(r->device));
                    if (r->tierx.configure(&rad_tools_registry(), &r->kv, &T, kv_disk_.get(), cg,
                                           max_batch, max_copy, max_chain,
                                           /*leader=*/&r == &ranks_[0],
                                           r->device,
                                           (int64_t)(&r - &ranks_[0]) * entry_bytes,
                                           (int64_t)(&r - &ranks_[0]) * ranks_[0]->kv.checkpoint_bytes(),
                                           /*shared=*/r->mover.transfer_stream())
                            != RAD_OK) {
                        all = false;
                        break;
                    }
                }
                RAD_TRY(rad_dev_set(ranks_[0]->device));
                if (!all) {
                    /* THE MOVER IS OFF AND THE BOOKKEEPING IS NOT. close() forgets the entry and
                     * snapshot sizes with everything else, and those are what a session's
                     * reported size is computed from, so the object is re-configured for the
                     * caps it can still honour -- none. */
                    T.close();
                    kv_disk_.reset();
                    tc.host_cap_bytes = tc.disk_cap_bytes = 0;
                    tc.ckpt_cap_bytes = tc.ckpt_disk_slots = 0;
                    RAD_TRY(T.configure(tc, nullptr));
                } else {
                    /* The tiers can lose an entry without the cache asking: a capacity drop, or a
                     * promotion that failed. The cache has to hear about it or it will go on
                     * serving block ids that are not its own any more. */
                    T.set_drop_sink([this](const BlockHash& h) {
                        ranks_[0]->prefix.drop_offdevice(h);
                    });
                    /* And the linear half on its own. The snapshot store is far smaller than the
                     * block store, so it overflows first and by itself; an index entry left
                     * behind would be a checkpoint the cache goes on offering whose bytes nobody
                     * holds, which is the recurrent-state form of the same corruption. */
                    T.set_snapshot_drop_sink([this](const BlockHash& h) {
                        ranks_[0]->prefix.drop_snapshot(h);
                    });
                    /* THE DISK HALF GETS A THREAD, and a plan holds about 512 MiB of it -- a
                     * fraction of a second of the store's write rate, so the disk copies keep up
                     * with the host copies in a handful of plans rather than one entry a second. */
                    pump_ = std::make_unique<TierPump>();
                    pump_->max_copy = max_copy;
                    pump_->slow_budget = (int)std::min<int64_t>(
                        1 << 20, std::max<int64_t>(64, (512ll << 20) / std::max<int64_t>(1, slot_bytes)));
                    if (kv_disk_) pump_->io.start(&ranks_[0]->tierx);
                    RAD_INFO("%s", ranks_[0]->tierx.report().c_str());
                }
            }
            /* ATTACHED WHETHER OR NOT ANYTHING MOVES. A null `tiers_` is the whole of the cache's
             * "off" path, and it turns off the session record along with the demotion it was
             * written for -- so detached, a server with no tier flag would publish no conversations
             * at all and the sessions view would have nothing to draw. With the caps at zero every
             * call the cache makes here is a record of what it is holding and nothing is ever
             * moved. */
            ranks_[0]->prefix.set_tiers(&T);
            /* AND SAID EITHER WAY. configure() builds this line whether the tiers are on or off;
             * printed only when they are on, a server that never moves a session would say
             * nothing about it at all -- leaving "why is the KV pool full with one conversation
             * open" with no answer in the log and a flag nobody knew was unset. */
            RAD_INFO("%s", T.report().c_str());
        }
    }

    RAD_INFO("kv: %s", ranks_[0]->kv.report().c_str());
    /* PrefixCache::configure builds a report saying whether it is on, which groups it covers and
     * which it had to skip, and this is where it is printed: unprinted, "why is cached_tokens
     * always zero" has no answer short of reading the source. It is one line at startup and names
     * the two things
     * that silently turn caching off for a hybrid model: a windowed group cannot be served a
     * prefix, and a linear model with no checkpoint interval replays from zero. */
    RAD_INFO("%s", ranks_[0]->prefix.report().c_str());

    /* CAN THE POOL ACTUALLY HOLD A MAX-LENGTH SEQUENCE? The context limit and the KV pool are set
     * by two unrelated flags, and this is where they are compared. Uncompared, a server started
     * with a context its cache cannot cover accepts the request at the door and refuses it at
     * admission seconds later, with a message about blocks rather than about the flag that was
     * wrong. A WINDOW group is exempt: its per-sequence cost is the window, not the context, which is
     * the whole point of it. Only paged groups can run out this way; a linear group is O(1) in
     * context and is sized per sequence. */
    {
        const int64_t want = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
        const KVManager& kv = ranks_[0]->kv;
        for (int32_t g = 0; g < kv.n_groups(); ++g) {
            const KVGroupPlan* p = kv.plan(g);
            if (!p || !p->paged() || p->window_blocks > 0) continue;
            const int64_t cap = p->n_blocks * p->block_size;
            if (cap >= want) continue;
            /* THIS ONE IS A REAL LIMIT AND NOT A BACKING RATIO. How much of the worst case
             * (every sequence at max_ctx at once) the pool backs is a provisioning choice and
             * well under half is normal for one-shot traffic. (It is NOT harmless for
             * multi-turn: prefix caching keeps a finished sequence's blocks so the next turn
             * can reuse them, so a pool that backs fewer live SESSIONS than it admits makes
             * every turn re-prefill. That is a provisioning choice too, and the budget line
             * in vram_budget.cpp says it.) What is never acceptable is a pool that cannot
             * hold ONE max-length sequence:
             * that request is accepted by the API and can never complete, whatever the load. */
            RAD_WARN("kv group '%s' holds %lld tokens but the context limit is %lld: ONE request "
                     "longer than %lld tokens is accepted by the API and then refused as too "
                     "large to ever fit -- this is a hard limit, not a backing ratio. Raise "
                     "--expert-vs-cache-ratio toward the cache (about %lld MiB more for the whole "
                     "context), halve the cost a token with --kv-cache-dtype fp8, or lower "
                     "--max-model-len to %lld.",
                     p->name.c_str(), (long long)cap, (long long)want, (long long)cap,
                     (long long)(((want - cap) / p->block_size * p->bytes_per_block) >> 20),
                     (long long)cap);
        }
    }
    return RAD_OK;
}

/* The group pools, as the run phase addresses them. The operand names an ABSOLUTE model layer and
 * the pool holds only the layers bound to this group, so the binding carries the POSITION of each
 * absolute layer within the group -- a group covering layers 3,7,...,63 is sixteen caches and layer
 * 7 is the second, not the fifth, and subtracting the first bound layer says the fifth. */
static void bind_kv_pools(const Program& p, const KVManager& kv, std::vector<KVPoolBinding>* out) {
    out->assign(p.kv_groups.size(), KVPoolBinding{});
    for (size_t g = 0; g < p.kv_groups.size(); ++g) {
        const KVGroupPlan* gp = kv.plan((int32_t)g);
        if (!gp) continue;
        KVPoolBinding& b = (*out)[g];
        b.base         = gp->base;
        b.layer_stride = gp->layer_stride();
        b.bytes        = gp->pool_bytes;
        b.dtype        = p.kv_groups[g].decl.dtype;

        /* The position of every bound layer, indexed by its absolute number. */
        const std::vector<int32_t>& ls = p.kv_groups[g].layers;
        int32_t hi = -1;
        for (int32_t l : ls) if (l > hi) hi = l;
        b.layer_slot.assign((size_t)(hi + 1), -1);
        for (size_t i = 0; i < ls.size(); ++i) b.layer_slot[(size_t)ls[i]] = (int32_t)i;

        /* The shape of ONE layer's cache, in the layout libref/ref_ops.h states for the whole
         * project. Composed here because this is the only place that holds both halves of it: the
         * plugin's declaration (heads, head_dim, state_dim, conv_width) and the block manager's
         * decisions (block_size, n_blocks, n_states, conv_slots). A kernel cannot derive the conv
         * window's depth or the slot count from any op parameter, so a flat span makes every shim
         * that needs the structure refuse. */
        const RadKVGroupDecl& d = p.kv_groups[g].decl;
        switch (gp->kind) {
        case RAD_KV_FULL:
        case RAD_KV_WINDOW:
            b.rank = 4;
            b.shape[0] = gp->n_blocks;
            b.shape[1] = d.n_head_kv;
            b.shape[2] = gp->block_size;
            b.shape[3] = 2 * d.head_dim;
            break;
        case RAD_KV_LINEAR:
            b.rank = 4;
            b.shape[0] = gp->n_states;
            b.shape[1] = d.n_head_kv;
            b.shape[2] = d.state_dim[0];
            b.shape[3] = d.state_dim[1];
            break;
        case RAD_KV_CONV:
            b.rank = 3;
            b.shape[0] = gp->n_states;
            b.shape[1] = d.n_head_kv * d.head_dim;
            b.shape[2] = gp->conv_slots;
            break;
        default:
            b.rank = 0;   /* the null sentinel at handle 0; nothing issues against it */
            break;
        }
    }
}

/* ================================================================== start */
void Engine::set_checkpoint_floors(int64_t keep) {
    for (auto& r : ranks_) {
        const int dev = r->device;
        r->kv.set_checkpoint_floor([dev](int64_t* free_now) -> int {
            RadDeviceProps p{};
            RAD_TRY(rad_dev_props(dev, &p));
            *free_now = p.vram_free;
            return RAD_OK;
        }, keep);
    }
}

int Engine::start() {
    std::vector<int> devs;
    std::string inventory;
    std::string devnote;
    /* Rank tags on the log from here down, so the pools, the planner, the mover and the sampler
     * are each attributed to the card they are about. At tp 1 there is nothing to disambiguate. */
    log_rank_tags(cfg_.tp > 1);

    int s = usable_devices(&devs, &inventory, &devnote);
    if (!devnote.empty()) RAD_WARN("device: %s", devnote.c_str());
    if (s < 0) { RAD_ERR("no usable device"); return s; }

    if ((int)devs.size() < cfg_.tp) {
        RAD_ERR("--tp %d needs %d devices of one architecture; %zu are usable [%s]",
                cfg_.tp, cfg_.tp, devs.size(), inventory.c_str());
        return RAD_E_DEVICE;
    }

    RAD_TRY(read_metadata());
    RAD_TRY(load_vocab());
    /* The cards are chosen, so a kernel library's device code is checked against the ones the
     * ranks will run on and not against whichever device the backend happens to list first. */
    rad_tools_registry().set_code_device(devs[0]);
    RAD_TRY(load_plugins());

    ranks_.clear();
    for (int i = 0; i < cfg_.tp; ++i) {
        auto r = std::make_unique<Rank>();
        r->index  = i;
        r->device = devs[i];
        ranks_.push_back(std::move(r));
    }

    /* PEER ACCESS, MEASURED BEFORE ANYTHING IS SELECTED. A kernel that reads or writes another
     * card's memory says so with a constraint on device.p2p (rad_types.h), and the selector reads
     * it from here: on a machine where some pair of the ranks' cards cannot reach each other such a
     * kernel is not selected, and declare names the constraint instead of the all-reduce faulting
     * or hanging on its first message. Every ordered pair, because access is granted one way at a
     * time. */
    {
        long long p2p = 1;
        std::string blocked;
        for (int i = 0; i < cfg_.tp; ++i)
            for (int j = 0; j < cfg_.tp; ++j) {
                if (i == j) continue;
                const int a = dev_peer_access(devs[i], devs[j]);
                if (a < 0) {
                    RAD_ERR("device: could not ask whether card %d can reach card %d's memory: %s",
                            devs[i], devs[j], rad_dev_last_error());
                    return a;
                }
                if (a == 0) {
                    p2p = 0;
                    blocked += fmt("%s%d->%d", blocked.empty() ? "" : ", ", devs[i], devs[j]);
                }
            }
        rad_tools_registry().set_device_fact(RAD_FACT_P2P, p2p);
        if (cfg_.tp > 1) {
            if (p2p) RAD_INFO("device: the %d cards reach each other's memory directly (peer to "
                              "peer)", cfg_.tp);
            else RAD_WARN("device: no peer-to-peer access between cards %s. A kernel that needs "
                          "it (device.p2p == 1) is not selected; if nothing else implements its "
                          "op, startup stops below naming it. IOMMU settings and the PCIe "
                          "topology decide this, not radiance", blocked.c_str());
        }
    }

    RAD_TRY(declare());

    /* --debug-graph stops here on purpose: everything it prints is known between declare and
     * plan, and a graph dump that first has to fit the model in VRAM is a graph dump you cannot
     * run on the machine where the model does not fit. */
    if (cfg_.debug_graph) return RAD_OK;

    /* Before anything is placed or mapped. A container whose ops resolve to the reference
     * implementation is a container that will serve at reference speed, and that is a decision the
     * operator makes rather than one the engine discovers three minutes into a load. */
    RAD_TRY(rad_gate_reference_kernels(ranks_[0]->program, cfg_.accept_reference_kernels, "serve"));

    RAD_TRY(plan());
    RAD_TRY(load_weights());
    RAD_TRY(configure_kv());

    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RAD_TRY(rad_dev_set(r->device));   /* allocates on the CURRENT device; see plan() */
        CtxDesc d{};
        d.program     = &r->program;
        d.rank        = r->index;
        d.world_size  = cfg_.tp;
        d.device      = r->device;
        /* Null in the all-in-VRAM case: the degenerate one where the planner assigned everything
         * to tier 0 and the mover never runs, and lookup collapses to WeightInfo::ptr. */
        d.residency   = r->plan.strategy == Strategy::AllVram ? nullptr : r->mover.residency();
        d.arch_step   = arch_step_;
        d.profile_ops = cfg_.profile_ops;
        bind_kv_pools(r->program, r->kv, &d.kv);
        RAD_TRY(r->ctx.init(d));

        /* THE MOVER NEEDS THE COMPUTE STREAM.
         *
         * Every transfer is ordered against the kernels that read the slot it targets, through an
         * event recorded on the compute stream once per dispatch (Mover::record_gate). With no
         * stream bound that function returns RAD_E_STATE, so a mover asked to move anything
         * refuses. The bind belongs here rather than in load_weights: Ctx owns the stream and does
         * not create it until init(), and the load runs on its own copy stream.
         *
         * Not under `if (d.residency)`: a plan can be static and still stage a file-tier weight
         * on demand, and a mover that is bound but idle costs one pointer. */
        RAD_TRY(r->mover.bind_compute_stream(r->ctx.stream()));

        /* THE PREFILL STAGER, where the budget made room for it: a step larger than every smaller
         * arena level is the only one that finds the region unlent, so that is the smallest pass
         * it arms for. */
        if (r->program.arena_stage_bytes > 0 && r->plan.dynamic && !r->arena_levels.empty()) {
            int64_t top_level = 0;
            for (const ArenaLevel& lv : r->arena_levels) top_level = std::max(top_level, lv.rows);
            RAD_TRY(r->stager.init(r->program, r->plan, &r->mover,
                                   (char*)r->ctx.arena_base() + r->program.arena_stage_off,
                                   r->program.arena_stage_bytes, top_level + 1));
            if (r->stager.enabled() && r->index == 0)
                RAD_INFO("prefill staging: %d %s buffers at the top of the arena; a step of "
                         "more than %lld tokens copies the next of %zu routed layers' pooled "
                         "experts onto the card while the one before it runs",
                         r->stager.n_bufs(), humanb(r->stager.buffer_bytes()).c_str(),
                         (long long)top_level, r->stager.n_layers());
        }
        /* THE FILE STAGER, wherever the plan put a routed expert on the file tier -- static plan
         * or dynamic, since a file-tier unit has no other address. */
        RAD_TRY(r->fstager.init(r->program, r->plan, &r->mover));
        if (r->fstager.enabled() && r->index == 0)
            RAD_INFO("file stager: %zu routed layers hold experts read from the container; each "
                     "pass reads a layer's (up to %s) into one of two VRAM buffers before its "
                     "first expert op, so every step is paced by the drive",
                     r->fstager.n_layers(), humanb(r->plan.file_layer_max).c_str());
        if (r->stager.enabled() && r->fstager.enabled()) {
            r->stagers.set(&r->stager, &r->fstager);
            r->ctx.set_stager(&r->stagers);
        } else if (r->stager.enabled()) {
            r->ctx.set_stager(&r->stager);
        } else if (r->fstager.enabled()) {
            r->ctx.set_stager(&r->fstager);
        }

        CedConfig ced_cfg;
        ced_cfg.enabled = cfg_.ced_prefetch;
        ced_cfg.lookahead = cfg_.ced_lookahead;
        ced_cfg.threshold = cfg_.ced_threshold;
        ced_cfg.top_k = cfg_.ced_top_k;
        ced_cfg.reuse_ridgefill = cfg_.ced_reuse_ridgefill;
        RAD_TRY(r->prefetcher.init(r->program, r->plan, &r->mover, &r->heat, ced_cfg));
        if (ced_cfg.enabled && r->index == 0) {
            RAD_INFO("CED prefetcher: lookahead=%d tau=%.2f top_k=%d (RDNA4 Wave32 async DMA on mover)",
                     ced_cfg.lookahead, ced_cfg.threshold, ced_cfg.top_k);
        }

        /* RADIANCE_HOST_DOMAIN puts EVERY op on the host domain, which is how you run the whole
         * model on libref. The oracle the entire correctness story leans on is a per-op
         * one -- it compares one kernel against ref on the operands the device handed it -- and
         * that can never answer "is the GRAPH right", because both sides read the same graph. A
         * whole forward pass on the reference implementation can, and it is the only thing that
         * can be diffed against another engine's.
         *
         * The site is a per-op property already (spec §5.1) and set_op_domain refuses by name
         * where a band has no host kernel, so this is the existing seam and not a new one. It
         * belongs with HIP_VISIBLE_DEVICES=-1, which drops the backend to host so the arena is
         * host memory a host kernel may dereference; without that this asks ref to read VRAM. */
        if (std::getenv("RADIANCE_HOST_DOMAIN")) {
            if (!device_is_host())
                RAD_WARN("RADIANCE_HOST_DOMAIN on the %s backend: the arena is device memory and "
                         "a host kernel cannot read it. Set HIP_VISIBLE_DEVICES=-1 as well.",
                         device_backend_name());
            int moved = 0;
            for (size_t i = 1; i < r->program.ops.size(); ++i) {
                const int st = r->ctx.set_op_domain((rad_op)i, RAD_DOMAIN_HOST);
                if (st < 0) return st;
                ++moved;
            }
            RAD_WARN("RADIANCE_HOST_DOMAIN: %d op(s) on the host. This is the reference "
                     "implementation running the whole model, and it is as slow as that sounds.",
                     moved);
        }
    }
    /* ---- THE LOAN: the cache's idle blocks, lent to the expert slab in place ----------------
     *
     * Here and not earlier because it needs both halves: the cache's carve fixes the addresses and
     * the mover's compute stream is what a recall drains. Every rank lends its own pool's strips --
     * the same geometry on every card -- and only a plan that moves units borrows at all.
     *
     * A STRIP IS ONE LAYER'S STRIPE OF ONE LENDABLE GROUP, top down, because that is how the cache
     * lends: the top blocks of a group are the top of every layer's stripe at once. */
    for (auto& r : ranks_) {
        if (!r->heat.enabled()) continue;
        RAD_TRY(rad_dev_set(r->device));
        std::vector<LoanStrip> strips;
        for (const KVGroupPlan& p : r->kv.plans()) {
            if (!r->kv.lendable(p.index) || !p.base) continue;
            const int64_t lbpb = p.layer_bytes_per_block(), stripe = p.layer_stride();
            for (int64_t L = 0; L < p.n_layers; ++L) {
                LoanStrip st;
                st.top = (char*)p.base + L * stripe + p.n_blocks * lbpb;
                st.bytes = p.n_blocks * lbpb;
                st.unit_bytes = lbpb;
                st.unit_tokens = p.block_size;
                strips.push_back(st);
            }
        }
        RAD_TRY(r->mover.lend_from(Mover::kLendKV, strips));
        if (r->mover.flex_capacity() > 0) lending_ = true;
    }
    (void)rad_dev_set(ranks_[0]->device);

    /* ---- AND THE TOP OF THE ACTIVATION ARENA, lent while the steps are small ----------------
     *
     * One stretch a rank: from the end of the full plan's transients down to the end of the
     * smallest level's, less a guard. A level lends everything above its own end and guard, so the
     * loan is stated in bytes from the top -- the measure every level can say in one number. Each
     * rank lends its own arena, whose plan may differ from rank 0's by the buffers only one rank
     * declares, so each rank's amounts are its own.
     *
     * THE GUARD is a mebibyte between a level's highest buffer and the lent memory: what a kernel
     * writes past the rows it was given lands there rather than in an expert's weights, which
     * would stay corrupt for as long as the unit stayed resident. */
    arena_rows_.clear();
    if (!ranks_.empty() && !ranks_[0]->arena_levels.empty() && ranks_[0]->heat.enabled()) {
        static constexpr int64_t kGuard = 1ll << 20;
        bool ok = true;
        for (auto& r : ranks_) {
            RAD_TRY(rad_dev_set(r->device));
            r->arena_lent.assign(1, 0);
            const Program& P = r->program;
            const int64_t top = P.arena_bytes;
            for (const ArenaLevel& lv : r->arena_levels) {
                if (r->ctx.add_level(lv.rows, lv.offset, lv.decl) < 0) { ok = false; break; }
                const int64_t edge = (lv.end + kGuard + RAD_ALIGN_UNIT - 1) /
                                     RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
                r->arena_lent.push_back(top > edge ? top - edge : 0);
            }
            if (!ok) break;
            int64_t most = 0;
            for (int64_t b : r->arena_lent) most = std::max(most, b);
            if (most <= 0) continue;
            LoanStrip st;
            st.top = (char*)r->ctx.arena_base() + top;
            st.bytes = most;
            st.unit_bytes = 1;
            st.unit_tokens = 1;
            RAD_TRY(r->mover.lend_from(Mover::kLendArena, { st }));
        }
        (void)rad_dev_set(ranks_[0]->device);
        if (ok) {
            arena_rows_.push_back(cfg_.max_tok);
            for (const ArenaLevel& lv : ranks_[0]->arena_levels) arena_rows_.push_back(lv.rows);
            std::string lv;
            for (size_t k = 1; k < arena_rows_.size(); ++k)
                lv += fmt("%s%s while steps are at most %lld tokens",
                          k > 1 ? ", " : "", humanb(ranks_[0]->arena_lent[k]).c_str(),
                          (long long)arena_rows_[k]);
            RAD_INFO("activation arena: lends %s", lv.c_str());
        } else {
            RAD_WARN("activation arena: a level would not bind; nothing is lent");
            for (auto& r : ranks_) r->arena_lent.clear();
        }
    }
    /* A RECALL THAT CANNOT WAIT. When a request needs more of the cache than it kept in hand, rank
     * 0's cache calls this from the scheduler, between dispatches, with every rank thread parked:
     * each rank's slab is drained to the loan the cache can still afford before the blocks are
     * handed out, and every follower's mark moves with rank 0's at once rather than at the next
     * step's build -- a tier restore on a follower picks the id rank 0 just chose. */
    if (lending_)
        ranks_[0]->kv.set_reclaim([this](int64_t tokens) -> int {
            int st = RAD_OK;
            for (auto& r : ranks_) {
                if (rad_dev_set(r->device) < 0) { st = RAD_E_DEVICE; break; }
                st = r->mover.reclaim(Mover::kLendKV, tokens, &r->heat);
                if (st < 0) break;
            }
            if (st >= 0)
                for (size_t i = 1; i < ranks_.size(); ++i) {
                    KVManager& f = ranks_[i]->kv;
                    if (tokens < f.loan_tokens()) (void)f.set_loan(tokens);
                }
            (void)rad_dev_set(ranks_[0]->device);
            return st;
        });

    /* WHAT THE CORE SPENT, beside what the card says was spent. The two differ by exactly the
     * allocations the core does not make: the driver's context, its code objects, and any kernel
     * plugin that calls hipMalloc for itself. Naming the difference is the only way to tell a
     * budget that is wrong from a plugin that is spending behind it. */
    {
        int64_t dev_bytes = 0, dev_count = 0;
        rad_dev_alloc_totals(RAD_MEM_DEVICE, &dev_bytes, &dev_count);
        int64_t pooled = 0;
        for (auto& r : ranks_)
            pooled += r->pools.weights().bytes() + r->pools.kv().bytes();
        RAD_INFO("device allocations: %lld totalling %s, of which %s is the two pools -- the rest "
                 "is the activation arena and the step staging. Anything on the card past this is "
                 "the driver or a kernel plugin's own hipMalloc, which this cannot see.",
                 (long long)dev_count, humanb(dev_bytes).c_str(), humanb(pooled).c_str());
    }

    RAD_TRY(step_state_init());

    /* ---- THE BUDGET, CHECKED AGAINST THE CARD IT WAS SPENT ON ----------------------------------
     *
     * Everything above this line allocated: the two pools, the activation arena, the step batch
     * staging, the weight tables a grouped GEMM walks. What is left on the card should now be
     * --gpu-headroom-mib and nothing else, and this is the only place that can say whether it is.
     *
     * Both directions are worth a line. TOO LITTLE means the terms the resolver could not compute
     * -- anything still allocated lazily, and whatever the driver took after the reading -- ate
     * into the memory the operator asked to keep, and the next thing to allocate on this card is
     * whatever fails. TOO MUCH means the budget left VRAM on the floor, which on a model whose
     * experts do not fit is milliseconds of decode step for every unspent GiB -- the exact
     * failure this resolver exists to remove.
     *
     * A slack of one arena is tolerated silently: MiB rounding and a kernel plugin's own
     * workspaces live in there, and naming a number this close to right would be noise. */
    /* DOES A DECOMMIT GIVE THE CARD BACK? The checkpoint slots are the one part of the KV pool
     * backed on demand, and releasing a freed slot's pages is only worth a driver call where the
     * card actually gets them back. That is a claim about the driver rather than about the API,
     * so it is measured here, once, on a range nothing else owns: commit it, read the card,
     * decommit it, read again. tools/vmemprobe.cpp asks the same question in more detail, and on
     * ROCm 7.2 / gfx1201 the answer is no -- hipMemUnmap returns nothing, and a reservation is
     * charged for its high-water commit until hipMemAddressFree.
     *
     * The expert slab does not depend on the answer. It borrows the cache's idle blocks at the
     * addresses the cache already owns, so no page ever changes hands (Engine::loan_tick). */
    if (!ranks_.empty() && VMem::supported()) {
        VMem      v;
        const int64_t g = VMem::granularity();
        const int64_t n = std::max<int64_t>(64ll << 20, g > 0 ? g * 16 : 0);
        RadDeviceProps a{}, b{};
        release_frees_card_ = false;
        if (rad_dev_set(ranks_[0]->device) >= 0 && v.reserve(n) >= 0 && v.commit(0, n) >= 0 &&
            rad_dev_props(ranks_[0]->device, &a) >= 0 && v.decommit(0, n) >= 0 &&
            rad_dev_props(ranks_[0]->device, &b) >= 0)
            release_frees_card_ = (b.vram_free - a.vram_free) > n / 2;
        for (auto& r : ranks_) r->kv.set_release_frees_card(release_frees_card_);
        if (!release_frees_card_)
            RAD_INFO("this driver does not return decommitted VRAM to the card (%s unmapped, the "
                     "free counter did not move), so a freed checkpoint slot keeps its pages. The "
                     "expert slab is unaffected: it borrows the cache's idle blocks in place.",
                     humanb(n).c_str());
    }

    /* A CHECKPOINT SLOT IS BACKED ON EVERY CARD BEFORE ITS ID IS HANDED OUT, AND NEVER INTO THE
     * HEADROOM. Rank 0's free list hands the ids out and every rank writes its shard of a snapshot
     * into the same slot of its own pool; backing a slot on one card and meeting a refusal on the
     * other at the snapshot copy would stop the engine mid-serve. See
     * KVManager::set_checkpoint_floor for why the floor is the headroom.
     *
     * AND NOT INTO WHAT THE BUDGET KEPT FOR LATER either: the step staging serve() allocates, and
     * what is built on first use. serve() lowers the floor once the staging exists. */
    const int64_t headroom = std::max<int64_t>(cfg_.gpu_headroom_mib, 0) * (1ll << 20);
    const int64_t keep_now = headroom + late_step_bytes_ + late_lazy_bytes_;
    if (!ranks_.empty()) {
        set_checkpoint_floors(keep_now);
        std::vector<KVManager*> peers;
        for (size_t i = 1; i < ranks_.size(); ++i) peers.push_back(&ranks_[i]->kv);
        ranks_[0]->kv.set_checkpoint_peers(std::move(peers));
    }
    /* AND ON A DRIVER THAT KEEPS WHAT IT MAPS, BACKED NOW. A slot backed later would cost the card
     * exactly what it costs now and could never be given back, so waiting would only move the
     * region's cost out of the reading below and into the middle of serving -- where the budget's
     * misses (allocations made after it was resolved) may already have taken part of it. Backed
     * here, the reading includes the slots, and a slot the cards have no room for is withdrawn
     * here and said so. */
    if (!release_frees_card_ && !ranks_.empty() && ranks_[0]->kv.free_checkpoint_slots() > 0) {
        KVManager& lead = ranks_[0]->kv;
        const int64_t asked = lead.free_checkpoint_slots();
        int64_t gone = 0;
        RAD_TRY(lead.back_free_checkpoints(&gone));
        if (gone > 0) {
            const int64_t stride = lead.checkpoint_stride();
            int64_t short_by = 0;
            for (auto& r : ranks_) {
                RadDeviceProps p{};
                if (rad_dev_props(r->device, &p) < 0) continue;
                short_by = std::max(short_by, gone * stride - (p.vram_free - keep_now));
            }
            RAD_WARN("checkpoint slots: %lld of %lld backed (%s each), %lld withdrawn -- one more "
                     "would take a card under the %s it has to keep free: the %s "
                     "--gpu-headroom-mib, the room a kernel's first launch allocates from, and %s "
                     "the budget kept for the step staging and for what is built on first use. "
                     "The prefix cache holds %lld linear-state snapshots on the card and its "
                     "retention policy decides which. The budget counted all %lld; the card is %s "
                     "short of them, which is allocations made after the budget was resolved.",
                     (long long)(asked - gone), (long long)asked, humanb(stride).c_str(),
                     (long long)gone, humanb(keep_now).c_str(), humanb(headroom).c_str(),
                     humanb(late_step_bytes_ + late_lazy_bytes_).c_str(), (long long)(asked - gone),
                     (long long)asked, humanb(short_by).c_str());
        }
    }

    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RadDeviceProps props{};
        if (rad_dev_props(r->device, &props) < 0 || props.vram_bytes <= 0) continue;

        /* AGAINST THE HEADROOM AND WHAT THE BUDGET KEPT FOR LATER, because both are still free
         * here: the step staging is allocated by serve(), and the KV tiers' staging and a second
         * compute lane are built on first use. A card at exactly the headroom now would be under
         * it once those exist. */
        const int64_t MiB  = 1024 * 1024;
        const int64_t late = late_step_bytes_ + late_lazy_bytes_;
        const int64_t want = headroom + late;
        const int64_t got  = props.vram_free;
        const std::string asked =
            late > 0 ? fmt("%s (the %s --gpu-headroom-mib and %s kept for the step staging and "
                           "for what is built on first use)", humanb(want).c_str(),
                           humanb(headroom).c_str(), humanb(late).c_str())
                     : fmt("%s --gpu-headroom-mib", humanb(want).c_str());
        if (got < want) {
            /* WHAT THIS SHORTFALL THREATENS is whatever allocates after startup: a kernel's code
             * object, loaded on its FIRST launch, and a spilling kernel's scratch, sized on its
             * first launch by the whole grid. Neither is knowable when the budget is resolved, and
             * running out for either is a fault inside the HIP runtime rather than an error. */
            RAD_WARN("card %d settled at %s free against %s. The difference is allocations "
                     "made after the budget was resolved, and it comes out of the room a kernel's "
                     "first launch allocates from. Raise --gpu-headroom-mib by at least %s so the "
                     "budget is resolved against the right number.",
                     r->device, humanb(got).c_str(), asked.c_str(), humanb(want - got).c_str());
        }
        else if (got > want + 256 * MiB) {
            /* AND "TOO MUCH" IS ONLY A FAULT WHERE SOMEBODY STATED THE BUDGET. Where the resolver
             * made the split, what is left is left because neither pool could address another
             * byte -- a small dense model already at the largest context its cache can hold
             * leaves most of the card -- and vram_budget.cpp printed it on the `unusable` line.
             * Warning here would contradict that receipt and send the reader looking for a flag
             * to set. */
            if (cfg_.vram_budget_derived)
                RAD_INFO("card %d settled at %s free against %s. The split accounted for it: "
                         "neither pool could address another byte, which is the `unusable` line of "
                         "the VRAM budget above.",
                         r->device, humanb(got).c_str(), asked.c_str());
            else
                RAD_WARN("card %d settled at %s free against %s, so %s is unused. On a model that "
                         "does not fit in VRAM every unused GiB is weight that has to be moved "
                         "instead of read, and that is decode step time.",
                         r->device, humanb(got).c_str(), asked.c_str(),
                         humanb(got - want).c_str());
        } else
            RAD_INFO("card %d settled at %s free, against %s",
                     r->device, humanb(got).c_str(), asked.c_str());
    }
    return RAD_OK;
}

const Program& Engine::program() const { return ranks_[0]->program; }

/* ================================================================== dump */
/* The answer to "it is unclear what even runs" (spec §16). It is a first-class mode rather than a
 * side effect of a verbose flag, because that is the complaint this project exists to answer. */
int Engine::dump() {
    Program& p = ranks_[0]->program;
    fputs(dump_graph_str(p, nullptr).c_str(), stdout);
    if (cfg_.debug_selection) fputs(rad_tools_registry().selection_table().c_str(), stdout);
    /* A HOST miss is not a failure -- see miss_report(). The exit code answers "is there a
     * hole a request can fall into", and a host miss is not one. NOR IS A DEVICE MISS WHOSE OP
     * HAS A HOST KERNEL: the planner places that op on the host and pins its weights in the
     * container's mapping, which is how a table too large to copy into VRAM is served at all. A
     * hole is a band NEITHER domain resolved. */
    std::set<std::pair<std::string, std::string>> no_host;
    for (const Program::Miss& m : p.misses)
        if (m.domain == RAD_DOMAIN_HOST) no_host.emplace(m.op, m.span);
    for (const Program::Miss& m : p.misses)
        if (m.domain != RAD_DOMAIN_HOST && no_host.count(std::make_pair(m.op, m.span)))
            return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* ================================================================== 6c. the vocabulary */
/* The sampler needs three facts about a token plus the ability to tokenise a short literal; the
 * text frontend has a whole Vocab. vocab_view.h says core/text implements the view over the real
 * vocabulary; core/text does not, so the adapter is here, where the two are wired together and
 * nowhere else. It belongs in core/text the moment a second caller wants it. */
namespace {
class RealVocabView final : public VocabView {
public:
    int init(std::shared_ptr<const Vocab> v) {
        v_ = std::move(v);
        RAD_TRY(tok_.init(v_));
        /* token_piece returns a reference, so the pieces have to outlive the call. Building them
         * once costs one pass over the vocabulary and saves a string construction per candidate
         * token per grammar step, which is host work on the step path.
         *
         * piece(), NOT Vocab::text(). text() is the vocab's OWN encoding -- for a byte-level BPE
         * that is the GPT-2 mapped alphabet, where a space is 'Ġ' (U+0120) and a newline is 'Ċ'.
         * piece() runs the declared decoder chain, which is what actually reaches the output
         * stream, and vocab_view.h asks for exactly that. Handing the grammar the mapped form
         * makes mask and accept consistently wrong, so nothing throws: every token containing a
         * space or a newline is simply forbidden by every grammar. render_special is TRUE
         * because the alternative gives a control token an empty piece, and an empty piece
         * consumes nothing and so satisfies any grammar position. */
        pieces_.resize((size_t)v_->n_tokens());
        for (int32_t i = 0; i < v_->n_tokens(); ++i)
            pieces_[(size_t)i] = tok_.piece(i, /*render_special=*/true);
        return RAD_OK;
    }
    int32_t n_tokens() const override { return v_ ? v_->n_tokens() : 0; }
    const std::string& token_piece(int32_t id) const override {
        static const std::string kEmpty;
        return (id >= 0 && (size_t)id < pieces_.size()) ? pieces_[(size_t)id] : kEmpty;
    }
    bool is_eog(int32_t id) const override { return v_ && v_->is_eog(id); }
    int  tokenize(const std::string& s, std::vector<int32_t>* out) const override {
        return out ? tok_.encode(s, *out, /*add_special=*/false, /*parse_special=*/false)
                   : RAD_E_INVAL;
    }
    int  tokenize_special(const std::string& s, std::vector<int32_t>* out) const override {
        return out ? tok_.encode(s, *out, /*add_special=*/false, /*parse_special=*/true)
                   : RAD_E_INVAL;
    }
private:
    std::shared_ptr<const Vocab> v_;
    Tokenizer                    tok_;
    std::vector<std::string>     pieces_;
};
}  /* namespace */

/* The container carries the vocabulary and nothing else is consulted. A .rad without one cannot
 * serve text, and says so here rather than at the first request (spec §4.1, §12). */
int Engine::load_vocab() {
    VocabBuild vb;
    if (file_) {
        VocabSection vs = file_->vocab();
        if (!vs) {
            RAD_ERR("%s carries no vocabulary section. A safetensors checkpoint has none of its "
                    "own, so rad-convert needs --tokenizer tokenizer.json.", cfg_.model.c_str());
            return RAD_E_FORMAT;
        }
        RAD_TRY(vocab_deserialize(file_->base(), (uint64_t)file_->size(), vs.h, vb));
    } else if (ckpt_->is_gguf()) {
        /* GGUF names its pre-tokeniser and the chain is reconstructed from the name, or refused
         * where it cannot be -- a container converted with --tokenizer bakes the declared one. */
        RAD_TRY(vocab_from_gguf(ckpt_->gguf(), vb));
    } else {
        /* beside config.json, with tokenizer_config.json and chat_template.jinja */
        const std::string tj = ckpt_->dir() + "/tokenizer.json";
        if (parse_tokenizer_json(tj.c_str(), vb) < 0) {
            RAD_ERR("%s: no readable tokenizer.json beside the checkpoint", cfg_.model.c_str());
            return RAD_E_FORMAT;
        }
    }

    /* THE OPERATOR'S TEMPLATE REPLACES THE CONTAINER'S before anything reads it, so the chat
     * endpoint and the reply format derived from the template agree about which one is served. An
     * unreadable or empty file is refused: serving the container's template in its place would be
     * the silent fallback the flag exists to rule out. */
    if (!cfg_.override_chat_template.empty()) {
        std::ifstream f(cfg_.override_chat_template);
        if (!f) {
            RAD_ERR("--override-chat-template %s: cannot read the file",
                    cfg_.override_chat_template.c_str());
            return RAD_E_IO;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string t = ss.str();
        /* Trailing newlines go, as they do for a checkpoint's own chat_template.jinja, so one file
         * renders the same prompt whichever way it reaches the server. */
        while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
        if (t.find_first_not_of(" \t\r\n") == std::string::npos) {
            RAD_ERR("--override-chat-template %s: the file is empty",
                    cfg_.override_chat_template.c_str());
            return RAD_E_INVAL;
        }
        RAD_INFO("chat: template from %s (%zu bytes), in place of the container's%s",
                 cfg_.override_chat_template.c_str(), t.size(),
                 vb.chat_template.empty() ? ", which carries none" : "");
        vb.chat_template = std::move(t);
    }

    auto v = std::make_shared<Vocab>();
    RAD_TRY(v->load(std::move(vb)));
    vocab_ = v;

    auto view = std::make_unique<RealVocabView>();
    RAD_TRY(view->init(vocab_));
    vocab_view_ = std::move(view);

    RAD_INFO("vocab: %d tokens, bos=%d eos=%d%s", vocab_->n_tokens(), vocab_->bos(), vocab_->eos(),
             vocab_->chat_template().empty() ? ", no chat template" : "");

    /* THE GRAMMAR ENGINE'S VOCABULARY INDEX, built here rather than by the first request that
     * carries a grammar -- a tool-calling template gives nearly every agent request one -- so
     * that request's admission pays only for its own grammar. */
    GbnfVocabInfo gi;
    const int gs = gbnf_prepare_vocab(vocab_view_.get(), &gi);
    if (gs < 0) {
        RAD_ERR("gbnf: the vocabulary index could not be built: %s", rad_strerror(gs));
        return gs;
    }
    RAD_INFO("gbnf: vocabulary index built in %.0f ms, %s", gi.build_ms,
             humanb((int64_t)gi.bytes).c_str());
    return RAD_OK;
}

}  /* namespace rad */
