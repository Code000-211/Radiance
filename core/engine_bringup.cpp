/* engine_bringup.cpp -- everything the Engine does once, in order: devices, the container's
 * metadata, the plugins, declare, the placement plan, the weights, the KV pools, the rank threads
 * and the vocabulary. Each numbered section is one component doing its own job.
 *
 * Separate from engine.cpp (the hot step loop), with which it shares only the Engine's fields.
 */
#include "engine_priv.h"
#include "rad_internal.h"
#include "engine_hostmem.h"
#include "gen/driver.h"
#include "gen/audio_driver.h"
#include "gen/video_driver.h"

#include "build/startup.h"
#include "build/rad_build.h"
#include "device/device.h"
#include "device/directio.h"
#include "format/radfile.h"
#include "format/rope_override.h"
#include "format/share.h"
#include "format/table_copy.h"
#include "rad_plugin.h"
#include "mem/vram_budget.h"
#include "runtime/arena.h"
#include "runtime/kvbind.h"
#include "runtime/xprefetch.h"
#include "sample/gbnf.h"
#include "sched/scheduler.h"
#include "text/token_view.h"
#include "text/tokenizer.h"

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

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
/* What did not resolve, in three classes reported differently:
 *
 *   - A device miss no host kernel stands in for is a hole (Program::device_holes). A request can
 *     land in it, so the engine refuses to start; each gets its own line with the geometry and
 *     refusing constraint.
 *   - A device miss a host kernel stands in for is not a hole: the planner runs the op on the host
 *     with its weights in the container's mapping (how `embed_lookup_q` serves a table too large
 *     for VRAM). An op with no weights has nothing to place, so on a card its host kernel cannot
 *     stand in (paged attention reads a VRAM cache).
 *   - A host miss only removes one placement option and is normal for a GPU library: one line for
 *     all of them.
 *
 * Printed alike, host misses would bury a real device hole in thousands of lines. */
struct MissReport {
    std::vector<std::string> device;   /* one line each; the engine does not start */
    std::string              host;     /* one line, or empty */
    /* Device misses covered by a host kernel (Tier::Mapped weights); informational. */
    std::string              host_only;
    /* A hole is an all-reduce declared on the lossy wire (`exact` 0, from --tp-wire;
     * rad_builder.h). The exact wire may be served at that world size, so the refusal names the
     * flag. */
    bool                     lossy_wire = false;

    bool operator==(const MissReport& o) const {
        return device == o.device && host == o.host && host_only == o.host_only &&
               lossy_wire == o.lossy_wire;
    }
};

static MissReport miss_report(const Program& p, bool host_backend) {
    MissReport r;
    std::vector<std::string>              host_ops;    /* in first-seen order */
    std::map<std::string, std::string>    host_spans;  /* op -> "M <= 64, M in (64, 512]" */

    const std::vector<const Program::Miss*> holes = p.device_holes(host_backend);
    std::set<std::pair<int32_t, std::string>> no_host;   /* (instance, band) with no host kernel */
    for (const Program::Miss& m : p.misses)
        if (m.domain == RAD_DOMAIN_HOST) no_host.emplace(m.index, m.span);
    for (const Program::Miss* m : holes) {
        long long exact = 1;
        if (m->index > 0 && (size_t)m->index < p.ops.size() &&
            p.ops[(size_t)m->index].base.get_i("exact", &exact) && exact == 0)
            r.lossy_wire = true;
        r.device.push_back(m->line());
        if (!no_host.count(std::make_pair(m->index, m->span)))
            r.device.back() += " (its host kernel cannot stand in: the op has no weights for "
                               "the planner to place on the host)";
    }

    std::vector<std::string> host_only_ops;
    for (const Program::Miss& m : p.misses) {
        if (m.domain != RAD_DOMAIN_HOST) {
            if (std::find(holes.begin(), holes.end(), &m) == holes.end() &&
                std::find(host_only_ops.begin(), host_only_ops.end(), m.op) ==
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

    /* Op names, not bands (--debug-graph has those), up to 16. Whole-model host execution needs
     * every op covered, so the names are actionable; past 16 the list stops being useful. */
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
/* Choose the devices. An integrated GPU appears as its own HIP device with system RAM as VRAM, and
 * a rank placed on one runs slowly and wrongly. So the engine uses the discrete cards if any
 * (integrated only on an APU-only machine), restricted to the first one's arch, and reports only
 * what it skipped. ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES control what the runtime exposes.
 *
 * `names` receives every device's product name by index ("" if unreadable), for
 * Engine::identify_devices, so the properties are read only once. */
static int usable_devices(std::vector<int>* out, std::string* inventory, std::string* note,
                          std::vector<std::string>* names) {
    const int n = rad_dev_count();
    if (n <= 0) return RAD_E_DEVICE;
    names->assign((size_t)n, std::string());

    struct D { int i; std::string arch; bool igpu; };
    std::vector<D> all;
    bool any_discrete = false;
    std::string inv;
    for (int i = 0; i < n; ++i) {
        RadDeviceProps p{};
        if (rad_dev_props(i, &p) < 0) continue;
        (*names)[(size_t)i] = p.name;
        const bool igpu = dev_integrated(i);
        inv += fmt("%s%d:%s%s", i ? " " : "", i, p.arch, igpu ? "(integrated)" : "");
        all.push_back({ i, p.arch, igpu });
        any_discrete |= !igpu;
    }

    std::string want;              /* the first taken device's arch; all ranks must match */
    std::string use, skip;
    for (const D& d : all) {
        const bool take = (!any_discrete || !d.igpu) && (want.empty() || want == d.arch);
        if (take && want.empty()) want = d.arch;
        std::string& into = take ? use : skip;
        into += fmt("%s%d:%s", into.empty() ? "" : " ", d.i, d.arch.c_str());
        if (take) out->push_back(d.i);
    }
    *inventory = fmt("%s backend, %d device(s) [%s]", device_backend_name(), n, inv.c_str());

    /* The backend already printed the inventory (core/device/dispatch.cpp); note only devices
     * skipped by the filter. */
    *note = skip.empty()
          ? std::string()
          : fmt("using %zu of %d [%s], ignoring [%s]: integrated graphics beside a discrete card, "
                "or a different architecture from the first card used, and a rank placed on one "
                "runs slowly and wrongly. ROCR_VISIBLE_DEVICES chooses the devices explicitly",
                out->size(), n, use.c_str(), skip.c_str());
    return out->empty() ? RAD_E_DEVICE : RAD_OK;
}

/* A /proc/meminfo field ("MemTotal", "MemAvailable") in bytes, or -1 when absent. */
static int64_t meminfo_bytes(const char* key) {
    std::ifstream f("/proc/meminfo");
    const std::string want = std::string(key) + ": %lld kB";
    std::string line;
    long long kb = 0;
    while (std::getline(f, line))
        if (std::sscanf(line.c_str(), want.c_str(), &kb) == 1) return (int64_t)kb * 1024;
    return -1;
}

/* The amdgpu driver's cap on the memory every process together pins: RAM less 1/64 and 1.5 GiB
 * (half of RAM less 1/64 below 3 GiB), unless amdgpu.no_system_mem_limit lifts it;
 * ttm.pages_limit does not apply. -1 when lifted (`*lifted` set) or MemTotal is unreadable. */
static int64_t amdgpu_pin_cap(bool* lifted) {
    {
        std::ifstream f("/sys/module/amdgpu/parameters/no_system_mem_limit");
        std::string v;
        *lifted = (f >> v) && (v == "Y" || v == "1");
    }
    const int64_t total = meminfo_bytes("MemTotal");
    if (*lifted || total <= 0) return -1;
    const int64_t reserve = (int64_t)3 << 29;
    const int64_t cap = total - (total >> 6);
    return cap < 2 * reserve ? cap >> 1 : cap - reserve;
}

/* Explain a refused pool allocation from the machine's current state.
 *
 * A card pool refused after the budget was measured (once, before any rank allocates) usually
 * lost the memory to another process. A host pool is pinned, once per rank, and counts against
 * amdgpu_pin_cap. */
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
        const int64_t avail = meminfo_bytes("MemAvailable");
        bool lifted = false;
        const int64_t cap = amdgpu_pin_cap(&lifted);
        /* With room under both limits the refusal is not about size; do not advise shrinking. */
        const int64_t pool = cfg.host_pool_total_mib() * 1024 * 1024;
        std::string tables, advice = "Lower --host-pool-mib";
        if (cfg.embedding_host_mib) {
            tables += fmt(" and the token embedding's %lld MiB", (long long)cfg.embedding_host_mib);
            advice += ", keep the embedding on the card with --embedding-placement vram";
        }
        if (cfg.tower_host_mib) {
            tables += fmt(" and the encoder towers' %lld MiB", (long long)cfg.tower_host_mib);
            advice += ", keep the encoder towers on the card with --vision-tower-placement vram";
        }
        const int64_t want = (int64_t)cfg.tp * pool + pinned + mapped;
        const bool fits = avail >= 0 && want <= avail && (lifted || (cap > 0 && want <= cap));
        RAD_ERR("host: --host-pool-mib %lld%s is pinned once per rank, so %d rank(s) pin %s; this "
                "process had taken %s of pinned host memory before rank %d's pool. The amdgpu "
                "driver caps what every process together pins at %s, and %s of RAM is available. %s",
                (long long)cfg.host_pool_mib, tables.c_str(), cfg.tp, humanb((int64_t)cfg.tp * pool).c_str(),
                humanb(pinned + mapped).c_str(), rank,
                lifted    ? "no limit (amdgpu.no_system_mem_limit)"
                : cap > 0 ? fmt("%s (RAM less 1/64 and 1.5 GiB)", humanb(cap).c_str()).c_str()
                          : "an unknown limit (MemTotal unreadable)",
                avail >= 0 ? humanb(avail).c_str() : "an unknown amount",
                fits ? "The pools fit under both, so the refusal is not one of size; the reason "
                       "the device gave is above"
                     : advice.c_str());
    }
}

/* ------------------------------------------------------------------ locked host memory */
/* The margin (--host-headroom-mib, Engine::host_headroom_bytes): host memory left free beside a
 * table locked whole -- the process's unpooled allocations, the load's staging windows, and the
 * page cache holding the container's hot pages. Without it the kernel reclaims the files the
 * engine reads, then the OOM killer runs. It is dead RAM by design; a host that can reclaim
 * cheaply trades it for pool, cache or slots. */

/* Every level of every memory cgroup the process is in, leaf first: `visit(dir, room)` until it
 * returns false (cgroup_levels). /proc/self/cgroup reads "0::/path" on v2 and "N:memory:/path"
 * (or a comma list naming memory) on v1; each level up to the visible root is a limit of its
 * own. */
static void memory_cgroups(const std::function<bool(const std::string&, int64_t)>& visit) {
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    bool go = true;
    while (go && std::getline(f, line)) {
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
        cgroup_levels(root, path, v2, [&](const std::string& dir, int64_t room) {
            return go = visit(dir, room);
        });
    }
}

/* Whether `need` more bytes fit in host memory after `still` more this process will allocate
 * first and `margin` (--host-headroom-mib) stays free, or why not. Checks MemAvailable; every
 * memory cgroup from the process's own up to the visible root (Docker's --memory, enforced by
 * the OOM killer); and, for an mlock()ed copy (`mlocks`), RLIMIT_MEMLOCK unless the process has
 * CAP_IPC_LOCK. Pinned memory a card reads is a driver userptr mapping, which RLIMIT_MEMLOCK
 * does not count. An unreadable limit is ignored, never guessed. */
static bool lockable_room(int64_t need, int64_t still, int64_t margin, bool mlocks,
                          std::string* why) {
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
    if (const int64_t avail = meminfo_bytes("MemAvailable");
        avail >= 0 && avail - still - margin < need) {
        *why = fmt("%s of memory is available, %s more is still to be allocated (the pinned host "
                   "pools, the prefix cache's host tier, the checkpoint slots and the host "
                   "activation arenas), and this wants %s with %s to spare "
                   "(--host-headroom-mib)",
                   humanb(avail).c_str(), humanb(still).c_str(), humanb(need).c_str(),
                   humanb(margin).c_str());
        return false;
    }
    bool fits = true;
    memory_cgroups([&](const std::string& dir, int64_t room) {
        if (room - still - margin >= need) return true;
        *why = fmt("the memory cgroup at %s leaves %s, %s more is still to be allocated, "
                   "and this wants %s with %s to spare (--host-headroom-mib)", dir.c_str(),
                   humanb(room).c_str(), humanb(still).c_str(), humanb(need).c_str(),
                   humanb(margin).c_str());
        fits = false;
        return false;
    });
    return fits;
}

/* The most lockable_room(need, still, margin, false) admits, and in `limit` the bound that
 * decides it. INT64_MAX when no limit can be read. */
static int64_t lockable_bytes(int64_t still, int64_t margin, std::string* limit) {
    int64_t most = INT64_MAX;
    *limit = "no readable memory limit";
    if (const int64_t avail = meminfo_bytes("MemAvailable"); avail >= 0) {
        most = avail - still - margin;
        *limit = fmt("%s of memory available, less %s still to be allocated and %s to spare",
                     humanb(avail).c_str(), humanb(still).c_str(), humanb(margin).c_str());
    }
    memory_cgroups([&](const std::string& dir, int64_t room) {
        if (room - still - margin < most) {
            most = room - still - margin;
            *limit = fmt("the memory cgroup at %s, which leaves %s, less %s still to be allocated "
                         "and %s to spare", dir.c_str(), humanb(room).c_str(),
                         humanb(still).c_str(), humanb(margin).c_str());
        }
        return true;
    });
    return std::max<int64_t>(most, 0);
}

/* The copy is table_copy_make's (core/format/table_copy.h), or for a card, table_copy_fill into
 * pinned memory every card maps (rad_dev_alloc RAD_MEM_HOST_MAPPED, locked in pieces). One copy
 * per container address, shared by every rank; freed in ~Engine. */
int Engine::copy_table(const uint8_t* src, int64_t n, int64_t foff, bool card, void** out,
                       std::string* why) {
    for (const auto& c : table_copy_of_)
        if (c.first == src) { *out = c.second; return RAD_OK; }
    const auto t0 = std::chrono::steady_clock::now();
    void* base = nullptr;
    if (card) {
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

/* When --ngram-placement auto chose a pinned copy for a card and it could not be made, move the
 * gather to the host. Every rank's n-gram weights take the host site, in both the WeightInfo
 * Ctx::init reads and the plan, so the gather and its scale move together; a copy already
 * published is host memory the host gather can read. False when no host kernel gathers the
 * table. */
bool Engine::ngram_gather_to_host() {
    for (auto& r : ranks_) {
        WeightFootprint f;
        weight_footprint(r->program, &f, TablePlacement::Vram, TablePlacement::Disk,
                         device_is_host(), file_ != nullptr);
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

/* Resolve the lookup-table placements (--embedding-placement, --vision-tower-placement,
 * --ngram-placement) once for every rank before the card is split, into Config for the planner
 * and the load phase.
 *
 * Embedding: Ram (default) pins it in its own host memory beside --host-pool-mib
 * (Config::embedding_host_mib); the card reads its M rows per step over the link, freeing VRAM for
 * KV and experts. If impossible (an output head tied to the table reads it whole every step, or
 * the host lacks memory) the default falls back to the card with a reason, while an explicit `ram`
 * stops the server. There is no disk tier: the lookup writes device-domain activations.
 *
 * Encoder towers: Vram by default; Ram pins them beside the pool (Config::tower_host_mib), read
 * over the link once per image.
 *
 * N-gram table: Vram where a device kernel gathers it and it fits; Ram as a copy (pinned for the
 * device kernel, or locked for the host gather); Disk in the container, read by the host gather.
 * Each tier is refused where no kernel reads it. Auto takes the first that fits, in that order.
 * The card test: the fixed weights plus a KV pool for one --max-model-len sequence (`kv_floor`).
 * The RAM test: lockable_room after the pools still to be allocated. */
int Engine::resolve_table_placement(const std::vector<VramFacts>& facts,
                                    const std::vector<int64_t>& emb_bytes,
                                    const std::vector<WeightFootprint>& wfs, int64_t kv_floor,
                                    int64_t ngram_bytes, bool ngram_device, bool ngram_host,
                                    int64_t host_late) {
    const int64_t MiB = (int64_t)1 << 20;
    int64_t emb_max = 0;
    for (int64_t b : emb_bytes) emb_max = std::max(emb_max, b);

    TablePlacement emb = cfg_.embedding_placement;
    const bool emb_asked = cfg_.embedding_placement_given;
    std::string emb_why;
    /* Host memory the table would pin (each rank its shard) after the pools and the prefix cache's
     * host tier still to be allocated; no memlock check, since card-read pins are not charged. */
    const int64_t emb_pin = (int64_t)cfg_.tp * align_up(emb_max, (int64_t)MiB);
    const int64_t pools_still = (int64_t)cfg_.tp * cfg_.host_pool_mib * MiB +
                                cfg_.prefix_cache_mib * MiB + host_late;
    std::string emb_room_why;
    auto emb_room = [&]() {
        return lockable_room(emb_pin, pools_still, host_headroom_bytes(), false, &emb_room_why);
    };
    if (emb_max == 0) {
        if (emb == TablePlacement::Ram && emb_asked) {
            RAD_ERR("--embedding-placement ram: this model has no token embedding table that only "
                    "embed_lookup reads. An output head tied to the table reads it too, and would "
                    "stream it over the link every step.");
            return RAD_E_INVAL;
        }
        emb = TablePlacement::Vram;
        emb_why = "no table only embed_lookup reads (an output head tied to it reads it too, "
                  "whole, every step)";
    } else if (emb == TablePlacement::Ram) {
        if (!emb_room()) {
            if (emb_asked) {
                RAD_ERR("--embedding-placement ram: the token embedding (%s a rank, pinned on "
                        "every rank) cannot be held in host memory: %s. Use vram.",
                        humanb(emb_max).c_str(), emb_room_why.c_str());
                return RAD_E_NOMEM;
            }
            emb = TablePlacement::Vram;
            emb_why = "the host cannot pin it: " + emb_room_why;
        } else {
            emb_why = emb_asked ? "as asked"
                                : "pinned in host memory and read over the link in place, so its "
                                  "VRAM goes to the KV cache and the experts";
        }
    } else {
        emb_why = "as asked";
    }
    cfg_.embedding_resolved = emb;
    cfg_.embedding_host_mib = emb == TablePlacement::Ram ? align_up(emb_max, MiB) / MiB : 0;

    /* Encoder towers: on the card unless ram was named; then pinned or the server stops. Every rank
     * pins a whole tower (encoders are not sharded). Decided before the n-gram table, whose tests
     * need the final statics and the pools still to be allocated. A rank's tower bytes leave out
     * the lookup tables the embedding pins (tower_pin_bytes); the pin, the card test below and
     * the split in Engine::plan all read that figure. */
    std::vector<int64_t> tower_bytes;
    for (const WeightFootprint& wf : wfs) tower_bytes.push_back(tower_pin_bytes(wf, emb));
    int64_t tower_max = 0;
    for (int64_t b : tower_bytes) tower_max = std::max(tower_max, b);
    const TablePlacement tw = cfg_.tower_placement;
    if (tw == TablePlacement::Ram) {
        if (tower_max == 0) {
            if (cfg_.mm_max_patches == 0)
                RAD_ERR("--vision-tower-placement ram: --mm-max-patches 0 asked for no media, so "
                        "this server declares no encoder tower to place");
            else
                RAD_ERR("--vision-tower-placement ram: this model has no encoder tower (no weight "
                        "its architecture reads once per request)");
            return RAD_E_INVAL;
        }
        std::string why;
        const int64_t pin = (int64_t)cfg_.tp * align_up(tower_max, (int64_t)MiB);
        if (!lockable_room(pin, pools_still + (int64_t)cfg_.tp * cfg_.embedding_host_mib * MiB,
                           host_headroom_bytes(), false, &why)) {
            RAD_ERR("--vision-tower-placement ram: the encoder towers (%s a rank, pinned on every "
                    "rank) cannot be held in host memory: %s. Use vram.", humanb(tower_max).c_str(),
                    why.c_str());
            return RAD_E_NOMEM;
        }
    }
    cfg_.tower_resolved = tw;
    cfg_.tower_host_mib = tw == TablePlacement::Ram ? align_up(tower_max, MiB) / MiB : 0;

    TablePlacement ng = cfg_.ngram_placement;
    std::string ng_why;
    /* The fastest tier with room: the card (every rank holds the whole table), then a RAM copy,
     * then the container. */
    const int64_t still = (int64_t)cfg_.tp * cfg_.host_pool_total_mib() * MiB +
                          cfg_.prefix_cache_mib * MiB + host_late;
    const int64_t need = align_up(ngram_bytes, (int64_t)2 << 20);
    const char* copy = ngram_device ? "a pinned copy the card reads" : "a locked copy";
    auto card_holds = [&](std::string* why) {
        std::vector<VramFacts> with = facts;
        int64_t fixed = 0;
        for (size_t i = 0; i < with.size(); ++i) {
            VramFacts& f = with[i];
            f.statics += ngram_bytes;
            if (emb == TablePlacement::Ram) f.statics -= emb_bytes[i];
            if (tw == TablePlacement::Ram) f.statics -= tower_bytes[i];
            fixed = std::max(fixed, f.statics);
        }
        /* Use `fixed`, not the trial's statics: the split clamps those to the card and would
         * stream every expert, so the trial alone cannot report a misfit. */
        VramBudget tb;
        const int st = vram_budget_trial(cfg_, with, &tb);
        const bool over = st == RAD_OK && fixed >= tb.claimable;
        /* The experts must keep a tier: when the cache lends, those the host pool cannot hold
         * have only the card. */
        const bool starved = st == RAD_OK && tb.lends && tb.expert_ceiling > 0 &&
                             tb.experts < tb.expert_floor;
        if (st == RAD_OK && !over && !starved && tb.kv >= kv_floor)
            return true;
        if (st != RAD_OK)
            *why = "the card cannot hold the other weights with it";
        else if (over)
            *why = fmt("with it the fixed weights want %s of a card that has %s to give",
                       humanb(fixed).c_str(), humanb(tb.claimable).c_str());
        else if (starved)
            *why = fmt("with it on the card the experts get %s, short of the %s the pinned host "
                       "pool cannot hold", humanb(tb.experts).c_str(),
                       humanb(tb.expert_floor).c_str());
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
        const bool room =
            !card && lockable_room(need, still, host_headroom_bytes(), !ngram_device, &ram_why);
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
                 table_placement_name(emb), emb_asked ? "" : " (default)", emb_why.c_str());
    if (ngram_bytes > 0)
        RAD_INFO("placement: n-gram table (%s) %s%s: %s", humanb(ngram_bytes).c_str(),
                 table_placement_name(ng),
                 cfg_.ngram_placement == TablePlacement::Auto ? " (auto)" : "", ng_why.c_str());
    decide("--embedding-placement", table_placement_name(emb),
           emb_max > 0 ? fmt("the table is %s a rank: %s", humanb(emb_max).c_str(), emb_why.c_str())
                       : emb_why);
    if (tower_max > 0)
        RAD_INFO("placement: encoder towers (%s a rank) %s%s", humanb(tower_max).c_str(),
                 table_placement_name(tw),
                 tw == TablePlacement::Ram ? ": pinned in host memory, read over the link once an "
                                             "image, their VRAM the KV cache's"
                                           : " (default)");
    decide("--vision-tower-placement", tower_max > 0 ? table_placement_name(tw) : "n/a",
           tower_max > 0 ? fmt("the towers are %s a rank", humanb(tower_max).c_str())
                         : std::string("this server declares no encoder tower"));
    has_ngram_ = ngram_bytes > 0;
    decide("--ngram-placement", ngram_bytes > 0 ? table_placement_name(ng) : "n/a",
           ngram_bytes > 0 ? fmt("the table is %s: %s", humanb(ngram_bytes).c_str(), ng_why.c_str())
                           : std::string("this model has no n-gram table"));
    return RAD_OK;
}

/* ================================================================== 2. metadata */
/* The model's metadata only: architecture id, dimensions, quantisation, vocab. Which weights exist
 * is the architecture plugin's answer at declare, not the container's. */
namespace {

/* A container is a file starting with the magic; anything else (a checkpoint directory, a
 * .safetensors, an index, a .gguf) is a checkpoint. A .rad path is always opened as a container,
 * so its reader reports what is wrong with it. */
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

    /* Keys are RadModelMeta's field names, which rad-convert writes (normalising safetensors or
     * GGUF input) and rad-kbench and rad-tune read. GGUF spellings (`block_count`, ...) are not a
     * fallback: a file lacking these keys was not written by rad-convert and is refused below. */
    meta_.n_layers        = geti("n_layers", 0);
    meta_.n_embd          = geti("n_embd", 0);
    meta_.n_head          = geti("n_head", 0);
    meta_.n_head_kv       = geti("n_head_kv", 0);
    /* Read head_dim, never derive it: it need not equal n_embd/n_head (docs/ARCHITECTURES.md). */
    meta_.head_dim        = geti("head_dim", 0);
    meta_.n_ff            = geti("n_ff", 0);
    meta_.n_vocab         = geti("n_vocab", 0);
    meta_.n_ctx_train     = geti("n_ctx_train", 0);
    meta_.n_expert        = geti("n_expert", 0);
    meta_.n_expert_used   = geti("n_expert_used", 0);
    meta_.n_expert_shared = geti("n_expert_shared", 0);
    meta_.rms_eps         = getf("rms_eps", 1e-6);
    meta_.rope_theta      = getf("rope_theta", 10000.0);
    /* 1 (abi/rad_builder.h): rotary scaling and its type come from the free-form keys. A
     * `rope_scale` key (the factor alone) is not read. */
    meta_.rope_scale      = 1.0f;

    /* Refuse a container with no dimensions here, by name, rather than at declare. */
    if (meta_.n_layers <= 0 || meta_.n_embd <= 0 || meta_.n_head <= 0 || meta_.n_vocab <= 0) {
        RAD_ERR("%s carries no model dimensions (n_layers=%lld n_embd=%lld n_head=%lld "
                "n_vocab=%lld). A .rad written by rad-convert always does; rad-info prints what "
                "this one holds.", cfg_.model.c_str(), (long long)meta_.n_layers,
                (long long)meta_.n_embd, (long long)meta_.n_head, (long long)meta_.n_vocab);
        return RAD_E_FORMAT;
    }

    /* All other keys verbatim for the plugin. The vectors own the storage; RadModelMeta holds
     * bare pointers. */
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
    RAD_TRY(apply_rope_scaling());
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

/* A checkpoint served directly (spec §4.3): config.json or the GGUF header is the metadata, read
 * as rad-convert reads it, and its tensors are the weights. Only encodings served as stored are
 * answered (CheckpointSources); weights needing quantisation are refused after declare. */
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
    RAD_TRY(apply_rope_scaling());
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

/* --rope-scaling (format/rope_override.h): write YaRN into the container's rotary keys at load,
 * where the architecture's reader takes them, so a container serves past its trained context
 * without reconversion. A drafter gets the same factor over its own trained length. Then log the
 * rotary and warn if the context exceeds its reach. */
int Engine::apply_rope_scaling() {
    RopeSpec spec;
    if (!rope_spec_parse(cfg_.rope_scaling.c_str(), &spec)) {
        RAD_ERR("--rope-scaling '%s': yarn:FACTOR (above 1) with an optional :ORIG, or none",
                cfg_.rope_scaling.c_str());
        return RAD_E_INVAL;
    }
    if (spec.kind != RopeSpec::Keep) {
        std::string why;
        if (!rope_override_group(meta_kv_k_, meta_kv_v_, spec, "", meta_.n_ctx_train, &why)) {
            RAD_ERR("--rope-scaling %s: %s", cfg_.rope_scaling.c_str(), why.c_str());
            return RAD_E_INVAL;
        }
        RopeKeys kv{ meta_kv_k_, meta_kv_v_ };
        if (kv.any_prefix("draft.")) {
            int64_t dtrain = meta_.n_ctx_train;
            if (const std::string* t = kv.get("draft.max_position_embeddings"))
                dtrain = std::atoll(t->c_str());
            if (!rope_override_group(meta_kv_k_, meta_kv_v_, spec, "draft.", dtrain, &why)) {
                RAD_ERR("--rope-scaling %s: %s", cfg_.rope_scaling.c_str(), why.c_str());
                return RAD_E_INVAL;
            }
        }
    }

    const RopeResolved r = rope_resolve(meta_kv_k_, meta_kv_v_, "");
    const long long want = cfg_.max_ctx > 0 ? (long long)cfg_.max_ctx : (long long)meta_.n_ctx_train;
    if (r.type == "yarn" && r.factor > 1.0) {
        const long long reach = (long long)(r.orig * r.factor);
        RAD_INFO("rope: YaRN factor %g over %lld trained positions -- the rotary reaches %lld; "
                 "--max-model-len %lld%s", r.factor, (long long)r.orig, reach, want,
                 spec.kind == RopeSpec::Yarn ? " (--rope-scaling)" : " (the container's own)");
        if (want > reach)
            RAD_WARN("--max-model-len %lld is past the %lld positions YaRN factor %g covers; "
                     "positions beyond them are extrapolated", want, reach, r.factor);
    } else if (meta_.n_ctx_train > 0 && want > meta_.n_ctx_train &&
               !(r.type == "linear" && r.factor > 1.0)) {
        RAD_WARN("--max-model-len %lld is past the %lld positions the model was trained on and "
                 "nothing scales its rotary, so positions beyond them are extrapolated; a model "
                 "card's YaRN extension is --rope-scaling yarn:FACTOR (docs/YARN.md)", want,
                 (long long)meta_.n_ctx_train);
    }
    return RAD_OK;
}

/* ================================================================== 3 + 4. plugins, declare */
int Engine::load_plugins() {
    int s = rad_tools_load_plugins(cfg_.radiance_home, cfg_.kernel_hierarchy,
                                   meta_.arch_id, meta_.quant ? meta_.quant : "", &plugins_,
                                   ckpt_ && ckpt_->imports() != Checkpoint::Import::None);
    if (s < 0) {
        /* rad_tools_load_plugins already logged why. A checkpoint whose quantisation the import
         * refused keeps the tool's descriptor, which no plugin claims; the refusal is the reason. */
        if (ckpt_ && !ckpt_->import_refusal().empty() && plugins_.arch_plugin.empty())
            RAD_ERR("quantisation '%s' cannot be imported: %s", meta_.quant ? meta_.quant : "",
                    ckpt_->import_refusal().c_str());
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
    /* The engine always states the KV dtype. RAD_DT_INVALID (zero) remains valid in the ABI for
     * other callers and means the plugin's default. */
    bc.kv_dtype = cfg_.kv_cache_dtype == "bf16" ? RAD_BF16 : RAD_F8E4M3;
    bc.kv_quant = cfg_.kv_cache_dtype == "fp8_v4" ? RAD_KVQ_V4
                : cfg_.kv_cache_dtype == "k6_v4"  ? RAD_KVQ_K6V4 : RAD_KVQ_NONE;
    bc.tp_wire_min_bytes = cfg_.tp_wire_min_kb * 1024;
    bc.scope      = "";
    bc.max_enc_patches = cfg_.mm_max_patches > 0 ? cfg_.mm_max_patches : 0;
    bc.max_audio_frames = cfg_.mm_audio_frames > 0 ? cfg_.mm_audio_frames : 0;
    /* A row at every token of the step plus the sampler's one row per sequence before them
     * (core/kld.h); KL mode has no speculation. */
    bc.max_out_rows = cfg_.kld() ? max_tok + cfg_.max_seqs : 0;
    /* Under --prefill-span every step but a lone prefill is planned at the shared budget
     * (Scheduler::step), so a wider step is one sequence; 0 without a span. */
    bc.max_mixed_tok = cfg_.prefill_span > 1 ? std::min(max_tok, prefill_shared_tok(cfg_)) : 0;
    /* The control vector, when one is loaded: this rank's view, or null. A plugin reads it at
     * declare only (the direction table never changes); the per-request switch rides the derived
     * mask, not this struct. */
    if (cvec_.loaded() && rank >= 0 && rank < (int) cvec_views_.size() &&
        cvec_views_[(size_t) rank].dir)
        bc.cvec = &cvec_views_[(size_t) rank];
    return bc;
}

int Engine::declare() {
    RAD_TRY(lora_targets());
    /* KL mode measures the trunk (core/kld.h): a drafter writes no scored rows and a prefix hit
     * would skip positions the reference holds, so both are off, and an explicit depth is refused
     * rather than ignored. The tower stays off the card; the corpus is text. */
    if (cfg_.kld()) {
        if (cfg_.n_spec > 0) {
            RAD_ERR("--num-speculative-tokens %d: the KL mode scores the trunk's own rows and "
                    "drafts nothing; leave it at auto or 0", cfg_.n_spec);
            return RAD_E_INVAL;
        }
        cfg_.n_spec = 0;
        cfg_.prefix_cache = false;
        if (cfg_.mm_max_patches < 0) cfg_.mm_max_patches = 0;
        decide("--num-speculative-tokens", "0", "the KL mode scores the trunk's own rows");
        prefix_off_why_ =
            "the KL mode: a prefix hit would start a document past positions the reference holds";
        decide("--no-prefix-cache", "prefix caching off", prefix_off_why_);
    }
    /* An image generator sizes the step itself (RadArchProbe::gen_rows): a denoise pass carries all
     * of an image's latent rows (twice with guidance), and a conditioning pass encodes every
     * reference picture of a request. Operator-given values are kept. No drafting (no tokens) and
     * no prefix cache (a conditioning is read only by its own request). */
    {
        RadArchProbe pb{};
        (void)rad_tools_probe(plugins_, meta_, &pb);
        if (pb.gen_rows > 0 && pb.gen_audio) {
            if (!cfg_.given.count("--max-num-batched-tokens")) {
                cfg_.max_tok = pb.gen_rows;
                decide("--max-num-batched-tokens", std::to_string(cfg_.max_tok),
                       "an audio generator: a window's denoising step, both branches, and a "
                       "prompt chunk of half as many rows a branch");
            }
            if (!cfg_.given.count("--max-num-seqs")) {
                cfg_.max_seqs = 8;
                decide("--max-num-seqs", "8", "an audio generator: four requests at once, two "
                       "branches each");
            }
            if (!cfg_.given.count("--max-model-len") && cfg_.max_ctx == 0) {
                cfg_.max_ctx = pb.gen_ctx > 0 ? pb.gen_ctx : pb.gen_rows;
                decide("--max-model-len", std::to_string(cfg_.max_ctx),
                       "an audio generator: the longest prompt and every frame of the longest "
                       "piece");
            }
            if (cfg_.mm_max_patches < 0) cfg_.mm_max_patches = 0;
            if (cfg_.n_spec < 0) cfg_.n_spec = 0;
            cfg_.prefix_cache = false;
        } else if (pb.gen_rows > 0 && pb.gen_video) {
            if (!cfg_.given.count("--max-num-batched-tokens")) {
                cfg_.max_tok = pb.gen_rows;
                decide("--max-num-batched-tokens", std::to_string(cfg_.max_tok),
                       "a video generator: a denoising step carries the prompt, the soundtrack "
                       "and every patch of the largest, longest video");
            }
            if (!cfg_.given.count("--max-num-seqs")) {
                cfg_.max_seqs = 1;
                decide("--max-num-seqs", "1", "a video generator: a pass is one request's");
            }
            if (!cfg_.given.count("--max-model-len") && cfg_.max_ctx == 0) {
                cfg_.max_ctx = pb.gen_rows;
                decide("--max-model-len", std::to_string(cfg_.max_ctx),
                       "a video generator: a step's rows");
            }
            if (cfg_.mm_max_patches < 0) cfg_.mm_max_patches = 0;
            if (cfg_.n_spec < 0) cfg_.n_spec = 0;
            cfg_.prefix_cache = false;
        } else if (pb.gen_rows > 0) {
            if (!cfg_.given.count("--max-num-batched-tokens")) {
                cfg_.max_tok = pb.gen_rows;
                decide("--max-num-batched-tokens", std::to_string(cfg_.max_tok),
                       "an image generator: a denoise pass carries the largest image's latent "
                       "rows twice (with guidance)");
            }
            if (!cfg_.given.count("--max-num-seqs")) {
                cfg_.max_seqs = 16;
                decide("--max-num-seqs", "16", "an image generator: images in one pass");
            }
            if (!cfg_.given.count("--max-model-len") && cfg_.max_ctx == 0) {
                cfg_.max_ctx = pb.gen_rows;
                decide("--max-model-len", std::to_string(cfg_.max_ctx),
                       "an image generator: the longest conditioning, a prompt and its pictures, "
                       "as the generator's rows");
            }
            if (cfg_.mm_max_patches < 0 && pb.gen_patches > 0) {
                cfg_.mm_max_patches = pb.gen_patches;
                decide("--mm-max-patches", std::to_string(cfg_.mm_max_patches),
                       "an image generator: every reference picture of a request at once");
            }
            if (cfg_.n_spec < 0) cfg_.n_spec = 0;
            cfg_.prefix_cache = false;
        }
    }
    /* fp8_v4 and k6_v4 are layouts only the plugin and libr4d's rows for them know. A plugin that
     * does not say it serves one (RadArchProbe::kv_quant_ok) would store plain E4M3 under a name
     * that promised a smaller cache, so the flag is refused rather than ignored. */
    if (cfg_.kv_cache_dtype == "fp8_v4" || cfg_.kv_cache_dtype == "k6_v4") {
        const int q = cfg_.kv_cache_dtype == "fp8_v4" ? RAD_KVQ_V4 : RAD_KVQ_K6V4;
        RadArchProbe pb{};
        (void)rad_tools_probe(plugins_, meta_, &pb);
        if (!(pb.kv_quant_ok & (1 << q))) {
            RAD_ERR("--kv-cache-dtype %s: the '%s' plugin does not serve it for this model "
                    "(fp8 does)", cfg_.kv_cache_dtype.c_str(), plugins_.arch_plugin.c_str());
            return RAD_E_UNSUPPORTED;
        }
    }
    /* Resolved here, not at argument parsing: the plugin states its drafter's depth, and it is
     * selected only once the container is open. Every consumer of `n_spec` (buffer sizing, kernel
     * bands, the scheduler's window) runs after this. rad_arch_probe is the core's only source of
     * drafter configuration. */
    if (cfg_.n_spec < 0) {
        RadArchProbe pb{};
        (void)rad_tools_probe(plugins_, meta_, &pb);
        cfg_.n_spec = pb.draft_depth > 0 ? pb.draft_depth : 0;
        if (cfg_.n_spec > 0)
            RAD_INFO("speculation: depth %d, this drafter's own operating point "
                     "(--num-speculative-tokens N overrides it)", cfg_.n_spec);
        else
            RAD_INFO("speculation: off -- this container carries no draft head");
        decide("--num-speculative-tokens", std::to_string(cfg_.n_spec),
               cfg_.n_spec > 0 ? "the depth the drafter in this container states as its operating point"
                               : "this container carries no draft head");
    }
    /* The audio encoder's pass: the plugin's longest clip, when the model file carries an audio
     * encoder (`radiance.encoder` names one) and the deployment is not text-only. */
    if (cfg_.mm_audio_frames < 0) {
        const char* enc = rad_meta_gets(&meta_, "radiance.encoder", "");
        const bool named = enc && std::strstr(enc, "audio") != nullptr;
        RadArchProbe pb{};
        (void)rad_tools_probe(plugins_, meta_, &pb);
        const bool text_only = cfg_.mm_max_patches == 0;
        cfg_.mm_audio_frames = !text_only && pb.audio_frames > 0 && (named || !file_)
                                   ? pb.audio_frames : 0;
    }
    /* The vision encoder's pass size. A container records its tower in `radiance.encoder` (written
     * when the tower's weights went in); a checkpoint served directly has one if it has a
     * vision_config. 16384 patches is a 2048 x 2048 image; the reference processor allows 16x that
     * area, which would need 16x the activation memory on every card. */
    if (cfg_.mm_max_patches < 0) {
        const char* enc = rad_meta_gets(&meta_, "radiance.encoder", "");
        const bool tower = rad_meta_geti(&meta_, "vision_config.patch_size", 0) > 0;
        cfg_.mm_max_patches = (enc && *enc) || (!file_ && tower) ? 16384 : 0;
        /* Whole output rows a pass: the plugin's merge, when it states one. */
        RadArchProbe pb{};
        (void)rad_tools_probe(plugins_, meta_, &pb);
        if (cfg_.mm_max_patches > 0 && pb.enc_merge > 1)
            cfg_.mm_max_patches -= cfg_.mm_max_patches % pb.enc_merge;
        if (cfg_.mm_max_patches == 0 && tower)
            RAD_INFO("media: off -- the checkpoint has a vision tower but this container was "
                     "converted without its weights; convert again to serve images and video");
        decide("--mm-max-patches", std::to_string(cfg_.mm_max_patches),
               cfg_.mm_max_patches > 0
                   ? "the model carries a vision tower: 16384 patches is a 2048 x 2048 image, the "
                     "largest one encoder pass takes"
               : tower ? "the checkpoint has a vision tower but this container was converted "
                         "without its weights"
                       : "the model carries no vision tower");
    }

    /* Declare once per rank (rank and world_size in RadBuildCtx; the plugin declares dimensions
     * already divided, spec §9), every rank concurrently on its own thread. Concurrency is
     * required: kernel init() runs during declare, and a collective's init is a rendezvous
     * (libr4d's all-reduce publishes its peer scratch pointer and waits for every other rank), so
     * a serial loop times out. Registry::select() and init_instance() share a mutex for this. Each
     * thread binds its device first, because the current device is thread-local and the init
     * allocates on it. */
    /* Control vectors are resolved here, before any rank declares: the plugin reads the direction
     * table from RadBuildCtx, so a table that failed validation must stop the server before a graph
     * is built around it.
     *
     * The feature is loaded only when the deployment asks: `--control-vector off|on`. Without it
     * nothing is looked for, the graph declares no projection op, and a request that asks for one
     * is refused by name -- so a deployment that does not use the feature pays nothing. With the
     * flag, the engine looks beside the container (or in the directory, or in
     * $RADIANCE_HOME/vectors) and uses the file if it is compatible; its operation parameters come
     * from the package's own JSON. A file that does not fit is reported and skipped -- a sidecar
     * must never stop a server nobody pointed at it. The flag's value is the startup switch: `off`
     * leaves requests off until one asks (or the dashboard flips the default), `on` starts them on. */
    const std::string arch = meta_.arch_id ? meta_.arch_id : "";
    if (cfg_.control_vector == "disabled") {
        RAD_INFO("control vectors: not loaded (no --control-vector); a request that asks for one is "
                 "refused");
    } else if (!arch.empty()) {
        std::string proj_path, tried;
        (void) control_vector_discover(cfg_.model, arch, cfg_.radiance_home, &proj_path, &tried);
        if (proj_path.empty()) {
            RAD_INFO("control vectors: none found (%s)", tried.c_str());
        } else if (arch != "qwen4exp") {
            RAD_WARN("control vectors: %s is for this family only (qwen4exp, whose gated residual "
                     "carries the projection op); this container is '%s' -- ignoring it",
                     proj_path.c_str(), arch.c_str());
        } else {
            std::string why;
            const int rc = control_vector_load(proj_path, arch, meta_.n_embd, meta_.n_layers,
                                               &cvec_, &why);
            if (rc != RAD_OK) {
                RAD_WARN("%s -- serving without it", why.c_str());
                cvec_ = ControlVector{};
            }
        }
    }
    if (cvec_.loaded()) {
        cvec_views_.assign(ranks_.size(), RadControlVector{});
        cvec_salt_ = control_vector_salt(cvec_);
        /* `--control-vector on` starts the request default on, which is also how `--kld-record`/
         * `--kld-ref` score the projection itself: those modes serve no request to carry the
         * switch. The switch still decides per request. */
        const bool default_on = cfg_.control_vector == "on";
        cvec_default_.store(default_on);
        RAD_INFO("control vectors: %s: %lld directions of %lld, layers %d..%d, scale %.3f -- "
                 "default %s%s",
                 cvec_.path.c_str(), (long long) cvec_.n_dirs,
                 (long long) cvec_.n_embd, cvec_.first, cvec_.last, (double) cvec_.user_scale,
                 default_on ? "ON" : "off",
                 default_on ? " (--control-vector on; a request can still say false)"
                            : " until a request asks for it (the dashboard's switch, or a request's "
                              "`control_vector` field)");
        decide("control vectors", cvec_.path,
               "found beside the container, in its directory, or in $RADIANCE_HOME/vectors; "
               "requests opt in");
    } else {
        cvec_default_.store(false);
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
            /* This rank's copy of the control vector's directions, on the device the thread has
             * just bound. Uploaded outside the program's arena: it is not a weight (it is not in
             * the container) and not an activation (it lives for the deployment). */
            if (cvec_.loaded()) {
                const float* d = control_vector_upload(cvec_, ri);
                if (!d) {
                    st[(size_t)ri] = RAD_E_DEVICE;
                    return;
                }
                cvec_views_[(size_t)ri] = control_vector_view(cvec_, d);
            }
            const RadBuildCtx bc = build_ctx(ri, cfg_.max_tok);
            LoraDeclare ld;
            ld.cfg = &cfg_.lora;
            ld.targets = lora_targets_;
            ld.sites_out = &r->lora_sites;
            st[(size_t)ri] = rad_tools_declare(plugins_, meta_, bc, &r->program,
                                               [this, ri](RadBuilder* bb, int64_t lw) {
                                                   return declare_sampler(ri, bb, lw);
                                               },
                                               nullptr, sources(), true,
                                               cfg_.lora.enabled() ? &ld : nullptr);
        });
    }
    /* A message is a step's bf16 residual rows, so the floor is a row count: a decode step has
     * sequences x (1 + speculative depth) rows. */
    if (cfg_.tp > 1 && !cfg_.tp_wire_exact) {
        const int64_t row_bytes = meta_.n_embd > 0 ? meta_.n_embd * 2 : 1;
        const int64_t exact_rows = (cfg_.tp_wire_min_kb * 1024 + row_bytes - 1) / row_bytes - 1;
        RAD_WARN("tp wire: LOSSY (--tp-wire wht6) at %lld KiB a message and above. A cross-rank "
                 "all-reduce of that size carries a quantised payload instead of bf16 (libr4d's "
                 "is Walsh-Hadamard-rotated 6-bit at two ranks and tiered int8 at three and "
                 "four), so this process does not compute the same function the exact wire does. "
                 "Steps of at most %lld rows are exact: single-sequence decode, and a "
                 "decode step until sequences x (1 + speculative depth) passes that; prefill and "
                 "a busy decode step are lossy. --tp-wire-min-kb moves the floor.",
                 (long long)cfg_.tp_wire_min_kb, (long long)exact_rows);
    }

    for (auto& t : th) t.join();

    /* Declare always completes and reports the full miss list (spec §1 step 4), after the join and
     * in rank order so lists do not interleave. Printed once when all ranks agree (the normal
     * case); a rank that differs gets its own report. */
    {
        std::vector<MissReport> rep;
        for (auto& r : ranks_) rep.push_back(miss_report(r->program, device_is_host()));
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
        /* A hole refuses startup: the bands come from this run's limits, so a request can land in
         * one, and that step would fail at issue and stop the engine. --debug-graph serves nothing,
         * so it continues to print the graph and reports the holes in its exit status
         * (Engine::dump). */
        for (const MissReport& m : rep)
            if (!m.device.empty() && !cfg_.debug_graph) {
                if (m.lossy_wire)
                    RAD_ERR("the band(s) above include the lossy wire --tp-wire wht6 asks for, "
                            "which the loaded kernel libraries do not implement at --tp %d; "
                            "--tp-wire exact asks for the exact one", cfg_.tp);
                RAD_ERR("not starting: the kernel libraries loaded have no kernel for the device "
                        "band(s) above, and this model needs them");
                return RAD_E_UNSUPPORTED;
            }
    }
    /* Checkpoint weights that cannot be served as stored cannot be loaded; name them all with the
     * remedy. */
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
    RAD_INFO("declare: %zu ops, %zu weights, %zu buffers, %zu KV groups",
             ranks_[0]->program.ops.size() - 1, ranks_[0]->program.weights.size() - 1,
             ranks_[0]->program.buffers.size() - 1, ranks_[0]->program.kv_groups.size());

    /* The draft head is exactly what the plugin declared (rad_declare_drafter). The core never
     * infers one from weight names or metadata, so third-party drafters need no core changes. */
    const Program& p0 = ranks_[0]->program;
    drafter_       = p0.drafter;
    drafter_name_  = p0.drafter_name;
    drafter_.name  = drafter_name_.c_str();
    if (cfg_.n_spec <= 0) drafter_ = RadDrafterDecl{};
    if (drafter_.kind != RAD_DRAFT_NONE) {
        /* A zero proposal handle means the sampler's token buffer: the head's logits go through
         * the ordinary sampler chain. */
        RAD_INFO("speculation: %s",
                 drafter_.kind == RAD_DRAFT_BLOCK
                 ? fmt("'%s' proposes %d token(s) in ONE block pass over %lld row(s) (mask id %d, "
                       "window %lld); prompt lookup fills in for any sequence it declines",
                       drafter_.name, drafter_.depth, (long long)drafter_.block,
                       (int)drafter_.mask_token, (long long)drafter_.window).c_str()
                 : fmt("'%s' drafts %d token(s) a step in %d serial pass(es); prompt lookup fills "
                       "in for any sequence it declines",
                       drafter_.name, drafter_.depth, drafter_.depth).c_str());
    } else if (cfg_.n_spec > 0) {
        RAD_INFO("speculation: prompt lookup only -- this container declares no draft head");
    }
    probe_arena_levels();
    return RAD_OK;
}

/* The mebibyte between a level's highest buffer and the memory above it, which is lent or holds
 * the stager's ring: what a kernel writes past the rows it was given lands there rather than in
 * an expert's weights. */
static constexpr int64_t kArenaGuard = 1ll << 20;

/* The activation arena's smaller levels, sized by the plugin itself.
 *
 * Activations are declared for the largest step, but a decode step uses a few dozen of a prefill
 * chunk's rows, leaving most of the arena idle. The plugin declares again at smaller step sizes
 * (RadBuildCtx::shape_probe), since only its shape rules know how each buffer scales; each answer
 * with the same buffers and fewer rows becomes a level: same fixed region, transients packed lower,
 * the arena above them free while steps stay that small.
 *
 * The smallest level is at least 256 rows. A tiled kernel may write padding rows of its last tile;
 * the real size is a multiple of every tile, and levels keep that property so padding never
 * reaches the next buffer or past the level's end. Each level is 4x the one below, up to the chunk.
 *
 * Under --prefill-span there is one level more, at the shared budget (prefill_shared_tok). The
 * full plan is sized for a spanned step, which only a prefill alone in its step takes; every other
 * step is at most the shared size, and at this level the span's extra rows are lent.
 *
 * A refused level is simply dropped. */
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
    arena_span_rows_ = 0;
    const int64_t span_rows = cfg_.prefill_span > 1 ? prefill_shared_tok(cfg_) : 0;
    if (span_rows >= lo && span_rows < cfg_.max_tok &&
        std::find(want.begin(), want.end(), span_rows) == want.end()) {
        want.push_back(span_rows);
        std::sort(want.begin(), want.end());
    }

    /* One sizing declare per (level, rank), all in parallel: they select and initialise nothing,
     * so they share no state. */
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
                    LoraDeclare ld;
                    ld.cfg = &cfg_.lora;
                    ld.targets = lora_targets_;
                    const int st = rad_tools_declare(plugins_, meta_, bc, &probe,
                                                     [this, &r](RadBuilder* bb, int64_t lw) {
                                                         return declare_sampler(r.index, bb, lw,
                                                                                true);
                                                     },
                                                     &r.program, sources(), true,
                                                     cfg_.lora.enabled() ? &ld : nullptr);
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
        if (R == span_rows) arena_span_rows_ = R;
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
    /* The budget is resolved once, for every rank, before any rank allocates. vram_budget_resolve
     * folds the smallest capacity with the largest program so one split fits every card, and
     * writes it into Config; per-rank resolution would let rank 1 measure a card rank 0 already
     * filled.
     *
     * All inputs exist by now: declare() has run (buffer plan, KV groups), and every kernel init
     * has run, so code objects are resident and free VRAM is the steady-state reading. */
    {
        std::vector<VramFacts> facts;
        facts.reserve(ranks_.size());
        late_step_bytes_ = late_lazy_bytes_ = 0;
        /* The lookup tables' sizes: per rank where sharded (the embedding; the towers are read
         * from `wfs`), once where not (the n-gram table, read by every rank from one mapping). */
        std::vector<int64_t> emb_bytes;
        std::vector<WeightFootprint> wfs;
        int64_t ngram_bytes = 0, kv_floor = 0;
        /* One checkpoint slot (every rank's shard) and the ranks' host activation arenas: pinned
         * after the tables are placed, outside every pool. */
        int64_t ck_slot = 0, host_arenas = 0;
        bool ngram_device = false, ngram_host = false;
        const int64_t ctx = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
        for (auto& r : ranks_) {
            RAD_TRY(rad_dev_set(r->device));
            VramFacts f{};
            /* One hardware queue's VRAM, measured as the difference between creating the first and
             * the second queue (the runtime's one-time setup lands on the first, and is charged to
             * `already held`). The compute stream, the mover's and a second lane are created after
             * this reading, and the driver backs each with VRAM beyond its ring. */
            {
                RadStream q[2] = { nullptr, nullptr };
                RadDeviceProps a{}, b{};
                if (rad_stream_create(&q[0], 0) >= 0 && rad_dev_props(r->device, &a) >= 0 &&
                    rad_stream_create(&q[1], 0) >= 0 && rad_dev_props(r->device, &b) >= 0)
                    f.queue_bytes = std::max<int64_t>(a.vram_free - b.vram_free, 0);
                for (RadStream s : q) if (s) rad_stream_destroy(s);
            }
            f.n_queues = 2 + (r->program.scratch_regions > 1 ? 1 : 0);
            /* The weights' cost, split into the fixed floor and the elastic plane
             * (weight_footprint in core/place/planner.h). Per rank, so already this card's share
             * under TP. */
            WeightFootprint wf;
            weight_footprint(r->program, &wf, TablePlacement::Vram, TablePlacement::Disk,
                             device_is_host(), file_ != nullptr);
            /* The expert prefetch's stream, ring and buffers, when it can run: experts in a pinned
             * pool, not --deterministic, and a spare DMA engine (asked before the free reading
             * below, so its queue is already counted there). */
            if (wf.routed && wf.expert_unit_max > 0 && !cfg_.deterministic &&
                (cfg_.placement == "auto" || cfg_.placement == "expert_tiered") && rad_xp_supported(r->device)) {
                /* Routed layers counted as Ctx::bind_routing does, including a head past the trunk. */
                int64_t nl = r->program.meta.n_layers;
                for (const WeightInfo& w : r->program.weights)
                    if (w.decl.group.expert >= 0) nl = std::max(nl, (int64_t)w.decl.group.layer + 1);
                f.n_queues += 1;
                f.prefetch = xp_device_bytes(wf.expert_unit_max, nl);
            }
            RadDeviceProps props{};
            rad_dev_props(r->device, &props);
            f.capacity = props.vram_bytes;
            f.free     = props.vram_free;
            /* On integrated graphics an eighth of host memory stays the host's, on top of the free
             * figure's own bound (core/device/hip.cpp), so the OS is not driven to reclaim and
             * kill. */
            if (dev_integrated(r->device)) {
                const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
                if (pages > 0 && page > 0) f.host_reserve = (int64_t)pages * page / 8;
            }
            arena_plan_bytes(r->program, nullptr, &f.arena);
            f.statics = wf.statics;
            f.experts = wf.experts;
            /* Kernel instances that hold device bytes of their own (the PLE VRAM cache). Their
             * init() ran before this reading, so the bytes are inside what `already held` measures
             * below; the figure only lets the budget report name that part of it. */
            for (const OpInfo& oi : r->program.ops)
                for (const Band& bd : oi.bands)
                    for (int d = 0; d < RAD_N_DOMAINS; ++d) {
                        const Resolved&            rs = bd.dom[d];
                        const RadKernelInfo* const ki = rs.row ? rs.row->info : nullptr;
                        if (ki && ki->mem_bytes && rs.instance)
                            f.plugin_held += ki->mem_bytes(rs.instance);
                    }
            wfs.push_back(wf);
            /* The cache lends where units move and the pool can return blocks in place. */
            f.lends = VMem::supported() && !cfg_.deterministic &&
                      (cfg_.placement == "auto" || cfg_.placement == "expert_tiered");
            emb_bytes.push_back(wf.embedding);
            ngram_bytes = std::max(ngram_bytes, wf.ngram);
            ngram_device = ngram_device || wf.ngram_device;
            ngram_host = ngram_host || wf.ngram_host;
            /* The staging region is sized only where the arena lends its top: without smaller
             * levels nothing returns it to the slab at decode, and a deterministic plan moves
             * nothing. */
            if (!r->arena_levels.empty() && !cfg_.deterministic) f.expert_layer_max = wf.expert_layer_max;
            /* A throwaway manager: plan_groups is configure()'s pure-arithmetic half and touches no
             * memory, and the real pool does not exist yet. */
            KVManager probe;
            host_arenas += arena_host_bytes(r->program);
            if (probe.plan_groups(r->program.kv_groups, cfg_) == RAD_OK) {
                ck_slot += probe.checkpoint_bytes();
                f.kv_ceiling = probe.ceiling(ctx);
                f.kv_fixed   = probe.fixed_bytes();
                /* One --max-model-len sequence's pool, as the inverse of the real plan: its
                 * recurrent and conv slots stay allocated with one request active, which a
                 * separate one-sequence plan would leave out. */
                const int64_t MiB = 1ll << 20;
                const int64_t need = probe.pool_mib_for_tokens(ctx, (f.kv_ceiling + MiB - 1) / MiB);
                if (need >= 0) f.kv_one = need * MiB;
                kv_floor = std::max(kv_floor, f.kv_one);
            }
            /* The step batch's device staging, reserved by the scheduler after this split: block
             * tables for max_seqs sequences at max_ctx, media staging for max_tok encoder rows.
             * State-index widths come from the probe, as the scheduler takes them from the
             * configured manager. */
            KVGeom kg = KVGeom::of(r->program.kv_groups);
            for (size_t i = 0; i < kg.g.size(); ++i) {
                const int64_t w = probe.state_index_width((int32_t)i);
                if (w > 0) kg.g[i].sidx_width = w;
            }
            f.staging = r->program.is_generator()       ? gen::Driver::device_bytes(r->program)
                      : r->program.is_audio_generator() ? gen::AudioDriver::device_bytes(r->program)
                      : r->program.is_video_generator() ? gen::VideoDriver::device_bytes(r->program)
                                                        : BatchBuilder::device_bytes(r->program, cfg_, kg);
            if (f.staging < 0) return RAD_E_INVAL;
            late_step_bytes_ = std::max(late_step_bytes_, f.staging);
            int64_t lazy = r->program.scratch_regions > 1 ? f.queue_bytes : 0;
            /* The KV tiers' copy staging, built by TierExec::device_init on the first copy out.
             * The cached groups are settled later, so every paged group is counted (an upper
             * bound). */
            if (cfg_.prefix_cache_mib > 0) {
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
        /* The pinned host pool is the experts' other home, less the file tier's staging, which
         * Planner::plan takes first. A host-placed embedding has its own room
         * (Config::embedding_host_mib). */
        for (size_t i = 0; i < facts.size(); ++i) {
            int64_t room = cfg_.host_pool_mib * ((int64_t)1 << 20);
            if (cfg_.weights_disk_tier)
                room -= file_stage_reserve(wfs[i].layer_unit_max, wfs[i].routed);
            facts[i].expert_floor = expert_vram_floor(wfs[i], std::max<int64_t>(room, 0));
        }
        /* Capped well below overflow, so the sums it joins stay defined: a store that large is
         * refused by name in configure_kv. Under --checkpoint-slots auto the store is sized in
         * configure_kv from the memory the tables leave, and the tables leave room for the fewest
         * slots auto resolves to. */
        const int64_t ck_n =
            cfg_.checkpoint_slots < 0 ? kCheckpointSlotsFloor : cfg_.checkpoint_slots;
        const int64_t ck_cap = INT64_MAX / 8;
        const int64_t ck_store = !cfg_.prefix_cache || ck_slot <= 0 ? 0
                                 : ck_n > ck_cap / ck_slot ? ck_cap : ck_slot * ck_n;
        /* The direction table was allocated in declare(); it is not part of the program, and
         * claiming its bytes back keeps a deployment that offers a vector planning against the
         * same card as one that does not (VramFacts::held_out). */
        const int64_t cvec_bytes = cvec_.loaded() ? (int64_t) cvec_.dir.size() * 4 : 0;
        for (VramFacts& f : facts) f.held_out = cvec_bytes;
        RAD_TRY(resolve_table_placement(facts, emb_bytes, wfs, kv_floor, ngram_bytes,
                                        ngram_device, ngram_host, ck_store + host_arenas));
        /* The n-gram table was counted off the card; on it, every rank holds the whole table. */
        if (cfg_.ngram_resolved == TablePlacement::Vram)
            for (VramFacts& f : facts) f.statics += ngram_bytes;
        if (cfg_.embedding_resolved == TablePlacement::Ram)
            for (size_t i = 0; i < facts.size(); ++i) facts[i].statics -= emb_bytes[i];
        if (cfg_.tower_resolved == TablePlacement::Ram)
            for (size_t i = 0; i < facts.size(); ++i)
                facts[i].statics -= tower_pin_bytes(wfs[i], cfg_.embedding_resolved);
        VramBudget vb;
        RAD_TRY(vram_budget_resolve(cfg_, facts, &vb));
        if (!vb.report.empty()) RAD_INFO("%s", vb.report.c_str());
        decide("vram split",
               fmt("weights %s (experts %s of it), KV cache %s, a card", humanb(vb.weights).c_str(),
                   humanb(vb.experts).c_str(), humanb(vb.kv).c_str()),
               vb.report);
        /* The staging region the budget took from the experts, placed above every planned buffer:
         * the arena top that smaller levels lend. */
        if (vb.stage > 0)
            for (auto& r : ranks_) {
                int64_t dev = 0;
                arena_plan_bytes(r->program, &dev, nullptr);
                r->program.arena_stage_off   = align_up(dev, RAD_ALIGN_UNIT);
                r->program.arena_stage_bytes = vb.stage;
                r->program.arena_bytes       = r->program.arena_stage_off + vb.stage;
            }

        /* ---- uneven worlds: each rank's own pools for the common token count ------------------
         *
         * The split above is the dearest rank's, applied to every card, and one block table serves
         * every rank (configure_kv caps each at the smallest paged token count). A rank whose cache
         * costs less per token (an attention-zero rank) would page only part of the common KV pool
         * while its extra experts sit in host memory. Such a rank gets a KV pool that pages exactly
         * the common token count, and the difference goes to its weights pool, up to all of its
         * weights resident. The token count, carve, loan and dearer ranks' pools are unchanged.
         *
         * Only expert_tiered re-splits: under `auto` a grown rank could pick all_vram while its
         * peers spill, and one strategy per world keeps the movers alike. A cache that does not
         * lend (--deterministic, or no virtual memory management) is still re-split, since its
         * unpaged bytes would otherwise be wasted. When all ranks cost the same, rank_cfg_ equals
         * cfg_. A qualifying rank that is not re-split is logged once with the reason. */
        rank_cfg_.assign(ranks_.size(), cfg_);
        kv_planned_tokens_ = 0;
        constexpr int64_t kMiB = (int64_t)1 << 20;
        int64_t kv_max = 0;
        for (const VramFacts& f : facts) kv_max = std::max(kv_max, f.kv_ceiling);
        std::vector<size_t> cheaper;
        for (size_t i = 0; i < facts.size() && i < ranks_.size(); ++i)
            if (facts[i].kv_ceiling < kv_max) cheaper.push_back(i);
        /* Why no cheaper rank is re-split, where the reason holds for the whole world. */
        std::string kept;
        std::vector<std::unique_ptr<KVManager>> probes(ranks_.size());
        int64_t T = -1;
        if (cheaper.empty()) {
            /* Every rank's cache costs the same per token: there is nothing to re-split. */
        } else if (cfg_.placement != "expert_tiered") {
            kept = fmt("--placement %s keeps one split for every card; only expert_tiered re-splits",
                       cfg_.placement.c_str());
        } else {
            /* The token count configure_kv will settle on: the least paged token count over ranks in
             * the common pool, from the same plan and bytes (nothing reserves from a KV pool before
             * its carve). */
            for (size_t i = 0; i < ranks_.size(); ++i) {
                probes[i] = std::make_unique<KVManager>();
                if (probes[i]->plan_groups(ranks_[i]->program.kv_groups, cfg_) != RAD_OK) {
                    kept = fmt("rank %d's KV groups did not plan", ranks_[i]->index);
                    break;
                }
                const int64_t t = probes[i]->paged_tokens(cfg_.vram_kv_mib * kMiB);
                if (t == KVManager::kUnboundedTokens) continue;
                if (T < 0 || t < T) T = t;
            }
            if (kept.empty() && T <= 0)
                kept = "no rank's cache pages a bounded token count in the common pool";
        }
        for (size_t i : cheaper) {
            const std::string costs =
                fmt("kv: rank %d's cache costs %s at the worst case where its dearest peer's costs "
                    "%s", ranks_[i]->index, humanb(facts[i].kv_ceiling).c_str(),
                    humanb(kv_max).c_str());
            if (!kept.empty()) {
                RAD_INFO("%s; its pools stay the common split: %s", costs.c_str(), kept.c_str());
                continue;
            }
            /* Exact, no margin: paged_split is monotone in the budget, so the least whole MiB that
             * pages T here pages at least T in configure_kv (same plan, same arithmetic). The
             * pool's alignment is already among the plan's fixed costs. */
            const int64_t need = probes[i]->pool_mib_for_tokens(T, cfg_.vram_kv_mib);
            if (need < 0) {
                RAD_INFO("%s; its pools stay the common split: it does not page the common %lld "
                         "tokens in the common %lld MiB", costs.c_str(), (long long)T,
                         (long long)cfg_.vram_kv_mib);
                continue;
            }
            const int64_t cap = facts[i].statics + expert_vram_floor(wfs[i], 0);
            const RankSplit rs =
                vram_budget_rank_split(cfg_.vram_weights_mib, cfg_.vram_kv_mib, need, cap);
            if (!rs.resplit) {
                RAD_INFO("%s; its pools stay the common split: it needs all %lld MiB of the common "
                         "pool to page the common %lld tokens", costs.c_str(),
                         (long long)cfg_.vram_kv_mib, (long long)T);
                continue;
            }
            rank_cfg_[i].vram_kv_mib      = rs.kv_mib;
            rank_cfg_[i].vram_weights_mib = rs.weights_mib;
            kv_planned_tokens_ = T;
            RAD_INFO("%s, so its KV pool pages the common %lld tokens in %lld MiB instead of %lld; "
                     "its weights pool grows %lld -> %lld MiB (every weight resident wants %s)%s",
                     costs.c_str(), (long long)T, (long long)rs.kv_mib, (long long)cfg_.vram_kv_mib,
                     (long long)cfg_.vram_weights_mib, (long long)rs.weights_mib,
                     humanb(cap).c_str(),
                     rs.unusable_mib > 0
                         ? fmt(", and %lld MiB stay unclaimed", (long long)rs.unusable_mib).c_str()
                         : "");
        }
    }

    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        auto& r = ranks_[ri];
        /* This rank's own split: cfg_ field for field unless the block above re-split it. */
        const Config& rc = rank_cfg_[ri];
        LogRank lr(r->index);
        /* This thread runs every rank's setup in turn, and rad_dev_alloc uses the current device,
         * so bind the rank's device before allocating anything for it. */
        RAD_TRY(rad_dev_set(r->device));
        RadDeviceProps props{};
        rad_dev_props(r->device, &props);

        /* Every pool is budgeted explicitly, with no profiling pass or utilisation fraction
         * (spec §6). Pools refuses by name and amount rather than shrinking anything. */
        int s = r->pools.configure(rc, props.vram_bytes);
        RAD_INFO("pools:\n%s", r->pools.fit_report().c_str());
        if (s < 0) {
            report_pool_refusal(r->index, r->device, r->pools, rc);
            return s;
        }

        /* Under TP the placement unit is a weight shard, so each rank plans against its own
         * card's budget (spec §5.3), the split its pools were given. */
        PlannerInput in{};
        in.program    = &r->program;
        in.config     = &rc;
        in.rank       = r->index;
        in.world_size = cfg_.tp;
        if (file_ && file_->header().prof_count) {
            in.profile   = file_->profile();
            in.n_profile = (int64_t)file_->header().prof_count;
        }
        in.container = file_ != nullptr;
        in.host_backend = device_is_host();
        Planner planner(in);
        s = planner.plan(&r->plan);
        if (cfg_.debug_placement || s < 0)
            fputs(r->plan.report(r->program).c_str(), stderr);
        if (s < 0) return s;
        /* The planner's slab must match the weights pool the mover fills. */
        RAD_TRY(plan_check_weights_pool(r->plan, r->pools.weights().bytes()));
        if (r->index == 0) {
            decide("--placement", strategy_name(r->plan.strategy),
                   r->plan.strategy_why.empty() ? std::string("as asked") : r->plan.strategy_why);
            const int64_t host = r->plan.totals.bytes[(int)Tier::HostPinned] +
                                 r->plan.totals.bytes[(int)Tier::HostPageable];
            decide("weights by tier",
                   fmt("VRAM %s, host pool %s, container on disk %s, mapped %s (rank 0)",
                       humanb(r->plan.totals.bytes[(int)Tier::VRAM]).c_str(), humanb(host).c_str(),
                       humanb(r->plan.totals.bytes[(int)Tier::SSD]).c_str(),
                       humanb(r->plan.totals.bytes[(int)Tier::Mapped]).c_str()),
                   r->plan.forced_host_used > 0
                       ? fmt("the host pool holds the %s the placement flags put there (the token "
                             "embedding, the encoder towers) beside the weights placed in it",
                             humanb(r->plan.forced_host_used).c_str())
                       : std::string());
        }
        /* --host-pool-mib is allocated whole on every rank, so a re-split rank whose weights all
         * fit its grown pool still holds an empty host pool; log it. */
        if (rc.vram_kv_mib < cfg_.vram_kv_mib && cfg_.host_pool_mib > 0 &&
            r->plan.totals.bytes[(int)Tier::HostPinned] == 0)
            RAD_INFO("kv: rank %d's weights all fit its grown weights pool, so its pinned host pool "
                     "holds no weights; its %lld MiB (--host-pool-mib) stay allocated, as on every "
                     "rank", r->index, (long long)cfg_.host_pool_mib);
    }
    return RAD_OK;
}

/* ================================================================== 6. load */
/* Weights are filled according to the plan. The mover has already reserved every slot and
 * published each weight's address, so this copies this rank's shard of each declared weight from
 * its container entry into place.
 *
 * The container holds whole weights (unless converted pre-sharded) and the plugin declared
 * dimensions already divided (spec §9), so slicing happens here, the only place that knows a
 * weight is part of a larger one. A row shard is a contiguous range of dim 0; a column shard
 * needs a 2D copy. */

/* One run of container bytes and its destination, emitted by the walk and consumed by the sweep.
 * `ny == 1` with `span == wb` is one contiguous range; otherwise a column share of `ny` pieces of
 * `wb` bytes every `sp`. With `stage >= 0` it fills that staged weight's input at offset `dst`
 * instead of the weight's home. */
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
 * Weights the loader must rearrange (spec §4.3): the resolved kernel stores something other than
 * the canonical planes (a fragment order, a per-row scale tail), or the declaration widens bf16 to
 * f32. The sweep copies the weight's planes into a host input; once all bytes arrive, a worker
 * produces the stored form, directly into its host pool slot or, for VRAM, through the worker's
 * pinned bounce buffers.
 *
 * It must keep pace with the disk (Qwen3.8-27B-FP8 rearranges 24 GiB of 29 per load; one thread
 * runs at a fraction of 7 GB/s), so it is a thread pool and the sweep only reads. The copies out
 * of sweep windows (into staged inputs or host pool slots) are also the pool's, done before any
 * relayout, because a window cannot be refilled until they finish.
 *
 * The pool holds a bounded amount, since staged inputs are host copies and a model exceeds host
 * memory. A new input waits for room only while finished work will free some (an incomplete
 * weight cannot be relaid, so waiting on it would deadlock); otherwise it is taken anyway. */
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
    /* A checkpoint weight has no sweep: the worker fills its input from the mapped tensors
     * itself, so reads run on all workers in parallel. */
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

    /* Inputs share one budget-sized region, reused as weights finish: fresh memory per input
     * would page-fault every 4 KiB, which dominates load CPU time. An input with no hole its size
     * (larger than the budget, or fragmentation) gets its own allocation. */
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

    /* Sweep thread: reserve budget for a staged weight's input when a run first reaches it. Only
     * the accounting happens here; input() allocates on a worker, because an mmap this size
     * contends with the workers' munmaps and faults and would stall the sweep. */
    void reserve(int i) {
        StagedWeight& s = items_[(size_t)i];
        if (s.reserved) return;
        std::unique_lock<std::mutex> lk(mu_);
        room_.wait(lk, [&] { return held_ + s.in_bytes <= budget_ || pending_ == 0 || failed_; });
        held_ += s.in_bytes;
        if (held_ > peak_) peak_ = held_;
        s.reserved = true;
    }

    /* `n` more bytes of input `i` arrived; called by any worker, so locked. */
    void arrived(int i, int64_t n) {
        std::lock_guard<std::mutex> lk(mu_);
        StagedWeight& s = items_[(size_t)i];
        s.remaining -= n;
        if (s.remaining > 0) return;
        pending_ += s.in_bytes;
        queue_.push_back(i);
        work_.notify_one();
    }

    /* A copy out of sweep window `slot` (first byte at file offset `w0`): the part of run `q` in
     * file range [j0, j1). A staged run lands in its weight's input (reserved first); any other
     * run has a host destination. */
    struct Copy { const uint8_t* win; int64_t w0, j0, j1; LoadRun q; int slot; };

    /* Sweep thread: the number of windows it reads into. */
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
    /* Sweep thread: wait until no copy from window `slot` is outstanding. A failed pool still
     * retires its copies, so this always returns. */
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
        /* Every staged weight must have been delivered; otherwise its slot holds whatever memory
         * held before (often zeros, so the bug can go unnoticed). */
        if (status_ == RAD_OK && !audited_) {
            audited_ = true;
            int64_t undelivered = 0;
            for (const StagedWeight& s : items_)
                if (s.remaining > 0) {
                    if (undelivered < 4)
                        RAD_ERR("load: '%s' was staged and nothing delivered its %s: its slot was "
                                "never written", s.prog->weights[s.wi].name.c_str(),
                                humanb(s.remaining).c_str());
                    ++undelivered;
                }
            if (undelivered) {
                RAD_ERR("load: %lld staged weight(s) were never delivered; the model cannot run",
                        (long long)undelivered);
                status_ = RAD_E_STATE;
            }
        }
        return status_;
    }

    bool failed() const { return failed_; }
    int64_t peak() const { return peak_; }

private:
    /* One stream per card shared by all workers. Each stream is a hardware queue on this backend,
     * and a stream per worker per card means ~20 time-sliced queues per card (workers spent three
     * quarters of a Flash-Next load waiting). One copy already saturates the link. */
    struct Uplink {
        std::mutex mu;
        RadStream  st = nullptr;
    };

    /* A worker's path to a card: a ring of pinned slots. A weight that fits is laid out directly
     * into a slot, a larger one goes up a slot at a time; a slot is waited for only when the ring
     * wraps back to it. */
    struct Bounce {
        static constexpr int     kSlots = 8;
        static constexpr int64_t kSlot  = 4ll << 20;
        explicit Bounce(Uplink* up) : up_(up) {}
        Uplink* up_;
        void* buf = nullptr;
        /* Relayout output for weights larger than a slot, reused across weights up to kKeep to
         * avoid page faults. */
        static constexpr int64_t kKeep = 256ll << 20;
        std::unique_ptr<uint8_t[]> out;
        int64_t out_cap = 0;
        /* One event per device (an event belongs to its device); `live` is the device of the
         * slot's last copy. */
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
        /* Wait for this worker's copies still in flight. */
        const int fs = b.flush();
        std::lock_guard<std::mutex> lk(mu_);
        if (fs < 0 && !failed_) { failed_ = true; status_ = fs; }
    }

    /* A staged weight's input, created once by the first worker to need it, in the first region
     * hole that fits. */
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

    /* Under the lock: free [off, off + n) of the region, merging with adjacent holes. */
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

    /* Copy each piece of the run that falls in the copy's file range: dense in the destination,
     * one row of a column share at a time. */
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
        /* Stored as read (only reached by a weight with no sweep). */
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
            /* Exact: every bf16 and f16 value is representable in f32. */
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
    std::unique_ptr<std::once_flag[]> once_;   /* per item: its input is made once */
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
    bool audited_ = false;   /* finish() runs its delivery audit once; the destructor calls it again */
    std::atomic<bool> failed_{false};
    int status_ = RAD_OK;
};

/* One pass over the container, serving every rank. The runs (declaration order, grouped by rank)
 * are sorted by file offset into one front-to-back sweep: read a window, hand it to every rank
 * that wants a piece, move on.
 *
 *   - The file is read once; replicated weights (gigabytes on a large container) are not read
 *     once per rank.
 *   - Reads are sequential, which one thread can sustain at the NVMe rate, provided they bypass
 *     the page cache: a buffered pread copies every byte on this thread and caps the rate at
 *     about two thirds (half with the relayout pool's copies). O_DIRECT DMAs into the window.
 *   - Copies overlap reads: a ring of pinned windows with one event per rank each, so reading one
 *     window overlaps the DMAs and pool memcpys out of the others. Serialising, or waiting on the
 *     stream instead of the event, costs about a fifth of the load rate.
 *
 * Pinned host memory is portable across devices in one process, so one window feeds every
 * card. */
static int sweep_container(RadFile* file, std::vector<LoadRun>& runs,
                           const std::vector<int>& devices, RelayoutPool* pool,
                           int64_t* read_out) {
    *read_out = 0;
    if (runs.empty()) return RAD_OK;

    /* Sort by file offset into a reading order; the span tie-break makes the order total, so the
     * same load always issues copies in the same sequence. */
    std::sort(runs.begin(), runs.end(), [](const LoadRun& a, const LoadRun& b) {
        if (a.foff != b.foff) return a.foff < b.foff;
        return a.span > b.span;
    });

    /* Bytes the sweep reads, for the progress line: the union of the runs (not their sum under TP,
     * nor the file size, since the mapped tier is excluded). */
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
    /* Widen the window to hold the widest strided piece. Slicing is correct at any width (a cut
     * piece is copied in two halves), but a narrower window doubles the copies. In practice no
     * piece approaches 64 MiB. */
    int64_t widest = 0;
    for (const LoadRun& q : runs) if (q.ny > 1 && q.wb > widest) widest = q.wb;
    if (widest > window) window = align_up(widest, 4096);

    /* Read through gaps up to kGap instead of breaking the stream: on NVMe a megabyte of
     * sequential read costs about as much as starting a new read. Large gaps (a mapped n-gram
     * table, tens of gigabytes) are far above this bound. */
    const int64_t kGap = 4 << 20;

    /* Six windows: host copies out of a window are worker memcpys, so a window is free only once
     * the workers have taken their pieces, and the ring bounds how far reads run ahead. Copies are
     * cut at kPiece so one window spreads over several workers. */
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
    double   read_s = 0;     /* the part of the sweep's wall time spent reading */

    const int rc = [&]() -> int {
        for (size_t k = 0; k < nr; ++k) {
            /* Streams and events belong to the device current at creation, so bind it here; the
             * copies themselves need no bind, since a stream carries its device. */
            RAD_TRY(rad_dev_set(devices[k]));
            RAD_TRY(rad_stream_create(&st[k], 0));
            for (int b = 0; b < kWin; ++b) RAD_TRY(rad_event_create(&ev[(size_t)b * nr + k]));
        }
        /* O_DIRECT needs block-aligned offset, length and buffer, while a chunk falls wherever its
         * runs do, so a window is read from the block below the chunk to the block above it (two
         * blocks wider). A filesystem that refuses O_DIRECT is read buffered. */
        RAD_TRY(dio.open(file->path(), RAD_DIO_READ | RAD_DIO_ALLOW_BUFFERED));
        for (int b = 0; b < kWin; ++b) {
            buf[b] = rad_dev_alloc(window + 2 * kBlk, RAD_MEM_HOST_PINNED);
            if (!buf[b]) {
                RAD_ERR("load: no %s pinned staging window (the load needs %d); free host memory",
                        humanb(window).c_str(), kWin);
                return RAD_E_NOMEM;
            }
        }

        int cur = 0;

        /* Read chunk [c0, c1) into window `b`; `*at` receives its first byte. Whole blocks up to
         * the file's last full block; only the tail past it uses DirectFile's bounce. When read
         * buffered, drop the window's pages afterwards, or tens of gigabytes of dead cache force
         * reclaim that the reads then wait on. */
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

        /* Issue the part of a run that lies in chunk [c0, c1). Chunks are file ranges, not runs, so
         * runs over the same entry (a column-sharded weight's per-rank runs, half a row apart)
         * share one read. Pieces wholly inside the chunk go as one 2D copy; only a piece cut by a
         * chunk boundary is copied separately. */
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
            /* Host destinations are memcpys done by the pool's workers: a staged run into its
             * weight's input (counted as arrived), any other into its host pool slot. */
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
        /* Before reusing a window, wait for the copies issued from it a ring turn ago: by event,
         * not stream sync (the stream also holds the other windows' copies this read should
         * overlap), and for the pool's memcpys out of it. */
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
        /* Progress lines name a container offset, which `rad-info -v` maps back to a weight, to
         * localise a stalled sweep. Every 8 GiB by default; RADIANCE_DEBUG_LOAD_EVERY_MIB sets a
         * finer interval. */
        int64_t kNoteEvery = 8LL << 30;
        if (const char* e = getenv("RADIANCE_DEBUG_LOAD_EVERY_MIB")) {
            const long long v = atoll(e);
            if (v > 0) kNoteEvery = (int64_t)v << 20;
        }
        size_t i = 0;
        while (i < runs.size()) {
            /* A group is a stretch of file: every run chaining to this one within kGap, across
             * ranks and weights, however long. A group larger than a window is read window by
             * window. */
            const int64_t g0 = runs[i].foff;
            int64_t g1 = g0 + runs[i].span;
            size_t  j  = i + 1;
            for (; j < runs.size(); ++j) {
                if (runs[j].foff - g1 > kGap) break;
                const int64_t end = runs[j].foff + runs[j].span;
                if (end > g1) g1 = end;
            }

            /* `lo` is the first unfinished run. Runs are sorted by offset, so one that ends before
             * the chunk is done for good and the scan never goes back. */
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

        /* The last host synchronisation before serving (spec §5.4). */
        for (size_t k = 0; k < nr; ++k) RAD_TRY(rad_stream_sync(st[k]));
        for (int b = 0; b < kWin; ++b) pool->drain(b);

        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count();
        /* Wall time vs disk time: when they match, the sweep is at the drive's ceiling; the
         * difference is copies that did not overlap a read (host memcpys, the issue loop). */
        RAD_INFO("load: one pass over the container, %s read in %.1f s (%.1f GB/s) for %zu "
                 "rank(s); %.1f s of that was the disk (%.1f GB/s)",
                 humanb(read_bytes).c_str(), s, read_bytes / (s > 0 ? s : 1) / 1e9, nr,
                 read_s, read_bytes / (read_s > 0 ? read_s : 1) / 1e9);
        return RAD_OK;
    }();

    /* Release the page cache of weights never read from the file again, leaving it to the mapped
     * tier (e.g. a mapped n-gram table, read from its mapping for the process lifetime). */
    file->drop_blob_cache();
    /* Drain before freeing: a sweep that failed part-way may have copies outstanding. */
    for (int b = 0; b < kWin; ++b) pool->drain(b);
    for (int b = 0; b < kWin; ++b) if (buf[b]) rad_dev_free(buf[b], RAD_MEM_HOST_PINNED);
    for (size_t k = 0; k < ev.size(); ++k) if (ev[k]) rad_event_destroy(ev[k]);
    for (size_t k = 0; k < nr; ++k) if (st[k]) rad_stream_destroy(st[k]);
    *read_out = read_bytes;
    return rc;
}

/* Read one rank's share of a checkpoint weight's selected planes from the mapped tensors into
 * `in` at the pool's offsets: whole rows in one read, a column share a block of rows at a time.
 * A reshaped whole plane has the same bytes, so it is read in the checkpoint's rows. */
static int fill_from_checkpoint(const CkptWeight& cw, const std::vector<PlaneShare>& shares,
                                const std::vector<int>& sel, const std::vector<int64_t>& offs,
                                const std::string& name, uint8_t* in) {
    std::vector<uint8_t> rows;
    std::string why;
    for (size_t k = 0; k < shares.size(); ++k) {
        const PlaneShare& sh = shares[k];
        uint8_t* at = in + offs[k];
        const RadEncPlane& p = cw.enc.plane[sel[k]];
        int64_t pr = 0, pc = 0;
        rad_enc_plane_dims(&p, cw.rows, cw.cols, &pr, &pc);
        const int64_t crb = rad_enc_row_bytes(p.dtype, pc);
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
    /* Counted apart from `n_loaded`: mapped weights are published in place, not loaded. */
    int64_t n_mapped = 0, mapped_bytes = 0;

    /* Two phases. The walk decides which rank wants which bytes, where they land, and what is
     * refused, without touching the disk; sweep_container then reads the container once in file
     * order (reading per rank in declaration order would run far below the sequential rate and
     * read replicated weights once per rank). */
    std::vector<LoadRun> runs;
    std::vector<StagedWeight> staged;
    int64_t n_out = 0;      /* destination overruns reported, capped */

    /* ============================================================== the walk */
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        /* Bind before allocating for this rank (as in plan()). */
        RAD_TRY(rad_dev_set(r->device));

        /* The container is the file tier (--weights-disk-tier places units on Tier::SSD), so the
         * mover needs its path. It reads the container with O_DIRECT: weights are RAD_ALIGN_UNIT-
         * aligned (rad_format.h), DirectFile's alignment, so stages take the no-bounce path. */
        MoverConfig mc{};
        mc.container_path = file_ ? cfg_.model : std::string();
        RAD_TRY(r->mover.init(r->program, r->plan, mc, &r->pools));
        /* What the mover took, before the first copy. Its slabs are allocated in init, so the pool
         * is as full as it will get; a load stopping with zero free is a placement failure, not
         * a loader one. */
        RAD_INFO("load: rank %d, mover took %s, weights pool %s free of %s", r->index,
                 humanb(r->pools.weights().used()).c_str(),
                 humanb(r->pools.weights().avail()).c_str(),
                 humanb(r->pools.weights().used() + r->pools.weights().avail()).c_str());

        /* The heat engine, bound to the same plan. On a static plan (all_vram, --deterministic) it
         * returns RAD_OK with `enabled() == false`. */
        RAD_TRY(r->heat.init(r->plan, HeatParams{}));

        /* Every destination must lie inside the weights pool. A copy past its end targets memory
         * the process does not own, which on this driver does not fault: the copy never completes
         * and the host spins in the runtime, like a hang. Checked for every copy before any
         * starts. */
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

        /* And every source range must lie inside its plane. Offsets depend on the rank index
         * (rank 0 always starts at the plane's start, so it never exercises the arithmetic), and
         * an overrun would read the wrong weight. */
        const int64_t file_bytes = file_ ? file_->size() : 0;

        for (size_t i = 1; i < r->program.weights.size(); ++i) {
            WeightInfo& w = r->program.weights[i];
            /* The planes' source: a container entry or the checkpoint tensors declare resolved,
             * giving the encoding and logical shape the share is cut in. */
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

            /* The source must hold the encoding declare saw (rad_weight_encoding) and every layout
             * hook answered for; otherwise refuse rather than relayout the wrong bytes. */
            if (!rad_enc_equal(&src_enc, &w.enc)) {
                char want[256], have[256];
                rad_enc_format(&w.enc, want, sizeof want);
                rad_enc_format(&src_enc, have, sizeof have);
                RAD_ERR("rank %d: '%s' was declared against '%s' as %s, and %s holds it as %s",
                        r->index, w.name.c_str(), w.source.c_str(), want, cfg_.model.c_str(),
                        have);
                return RAD_E_FORMAT;
            }

            /* Copy the planner's decision (Plan::weight[], a parallel array) onto WeightInfo.
             * Ctx::bind_weights picks RAD_DOMAIN_HOST only when every weight operand has
             * Site::Host, so without this host-site execution is unreachable. file_offset (the
             * selected plane's) is what Mover::stage preads for a weight read in place
             * (weight_file_direct); unset, it would read the file header. Set before the residency
             * lookup, because a file-tier weight has no address yet and `continue`s below. */
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

            /* The mapped tier is published, not loaded: the container stays mmapped for the
             * engine's life (core/format/radfile.cpp), and the sweep never reads these bytes.
             * Publish through the residency table too, not only WeightInfo::ptr, because
             * Ctx::resolve_weight reads the table whenever a mover exists. */
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
                /* --ngram-placement ram: the gather reads a copy instead of the mapping: pinned
                 * memory for a device kernel on the zero-copy site, or locked memory for the host
                 * gather. If auto's copy fails, the table stays in the container (a card's gather
                 * moves to the host); if `ram` was asked for, startup fails. */
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
                    decide("--ngram-placement", "disk",
                           fmt("auto chose a %s copy in RAM, which could not be made (%s); the "
                               "rows are read from the container%s", into, why.c_str(),
                               card ? " by the host gather" : ""));
                }
                /* No readahead on a mapped weight: a host kernel gathers a few rows from a table
                 * too large to be resident, so the default 128 KiB readahead would read 32 pages
                 * for a row of a few hundred bytes. MADV_RANDOM is per range, so the rest of the
                 * container keeps sequential readahead. The effect shows only with a cold cache. */
                posix_madvise(w.ptr, (size_t)n, POSIX_MADV_RANDOM);
                r->mover.publish_mapped((rad_weight)i, w.ptr);
                ++n_mapped;
                mapped_bytes += n;
                continue;
            }

            const WeightSlot* slot = r->mover.residency()->lookup((rad_weight)i);
            if (!slot || !slot->ptr) {
                /* A file-tier unit has no address until the mover stages it (spec §5.4). */
                if (w.tier == Tier::SSD) continue;
                RAD_ERR("rank %d: '%s' has no address after mover init", r->index, w.name.c_str());
                return RAD_E_STATE;
            }

            /* The address must also land on WeightInfo: with everything in VRAM, Ctx has no
             * residency table and Ctx::resolve_weight reads WeightInfo::ptr
             * (core/runtime/issue.cpp). `generation` is not bumped, since bind_weights first runs
             * after this, in Ctx::init. */
            w.ptr = slot->ptr;

            /* Host-pool destinations stay a memcpy. The pool is pinned and device-mapped
             * (core/mem/pools.cpp), but an async host-to-host copy on this stack is the same memcpy
             * behind a stream, not a DMA, so streaming it would not overlap it with reads. */
            const bool dev = w.tier == Tier::VRAM;
            if (dev && !dst_fits(slot->ptr, w.stored_bytes, w.name.c_str())) return RAD_E_STATE;

            /* This rank's share of the selected planes (format/share.h). */
            PlaneShare sh[RAD_ENC_MAX_PLANES];
            std::string why;
            if (weight_share(w, src_enc, src_rank, src_shape, cfg_.tp, r->index, sh, &why) !=
                RAD_OK) {
                RAD_ERR("rank %d: '%s' at --tp %d: %s", r->index, w.name.c_str(), cfg_.tp,
                        why.c_str());
                return RAD_E_SHAPE;
            }

            /* From a checkpoint every weight is staged and read by its worker (no sweep); the pool
             * stores it as is, widened, or relaid. */
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

            /* Stored as is: the share goes from the file to the slot. Otherwise it is staged plane
             * by plane for the relayout pool, which writes the slot. */
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
        RAD_ERR("%lld declared weights are absent from the container; the model cannot run",
                (long long)n_missing);
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
    /* Started even with nothing staged: the same workers fill host pool slots. */
    const int workers = (int)std::max(2u, std::min(16u, hw > 2 ? hw - 2 : 2u));
    pool.start(workers, 4ll << 30);
    const auto rt0 = std::chrono::steady_clock::now();
    int sw_st = RAD_OK;
    if (file_) {
        sw_st = sweep_container(file_.get(), runs, devices, &pool, &read_bytes);
    } else {
        /* Every weight to the workers in declaration order, as the budget allows. */
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
        /* Summed over ranks, which map the same file, so under TP the resident bytes are this
         * divided by --tp; the message says so. */
        RAD_INFO("load: %lld weight(s), %s, read in place from the container's mapping -- not "
                 "copied and charged to no pool%s", (long long)n_mapped,
                 humanb(mapped_bytes).c_str(),
                 cfg_.tp > 1 ? " (summed over ranks, which map the same file: the resident bytes "
                               "are that divided by --tp)" : "");
    return RAD_OK;
}

/* ================================================================== 6b. KV pools */
/* The block manager owns all KV group memory (paged attention caches, GDN recurrent states, conv
 * windows), so one owner decides what a sequence holds and a freed state slot is never handed out
 * while still in use (spec §7.2).
 *
 * Runs after load, since it takes what the weight pool left, and before Ctx::init, which would
 * otherwise resolve every RAD_OPK_KV operand to null. */
/* --checkpoint-slots auto, resolved into cfg_.checkpoint_slots once a slot's size is known
 * (checkpoint_slots_auto in core/mem/prefix.h). The store is pinned for the life of the process,
 * so it takes at most an eighth of host memory, leaving the rest to the page cache a model's
 * mapped tables read through and to the host's other work. It is pinned before the host
 * activation arenas and the prefix cache's host tier, so their bytes count as still to come;
 * on a card the amdgpu pin cap bounds it too. */
int Engine::size_checkpoint_store(int64_t slot, const char* what) {
    if (slot <= 0 || !cfg_.prefix_cache) {
        cfg_.checkpoint_slots = 0;
        decide("--checkpoint-slots", "0",
               slot <= 0 ? "the model keeps no linear or sliding-window state to snapshot"
                         : "prefix caching is off (--no-prefix-cache)");
        return RAD_OK;
    }
    const int64_t ctx = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
    const int64_t interval = geo_.checkpoint_interval;
    const std::unique_ptr<CheckpointPolicy> pol =
        make_checkpoint_policy(cfg_.checkpoint_policy, interval);

    int64_t still = cfg_.prefix_cache_mib * ((int64_t)1 << 20);
    for (const auto& r : ranks_) still += arena_host_bytes(r->program);
    std::string limit;
    int64_t room = lockable_bytes(still, host_headroom_bytes(), &limit);
    if (!device_is_host()) {
        bool lifted = false;
        if (const int64_t cap = amdgpu_pin_cap(&lifted); cap > 0) {
            int64_t pinned = 0, mapped = 0;
            rad_dev_alloc_totals(RAD_MEM_HOST_PINNED, &pinned, nullptr);
            rad_dev_alloc_totals(RAD_MEM_HOST_MAPPED, &mapped, nullptr);
            const int64_t left = std::max<int64_t>(cap - pinned - mapped - still, 0);
            if (left < room) {
                room = left;
                limit = fmt("the amdgpu driver's pin cap of %s, of which this process holds %s "
                            "and still pins %s", humanb(cap).c_str(),
                            humanb(pinned + mapped).c_str(), humanb(still).c_str());
            }
        }
    }
    const int64_t total = meminfo_bytes("MemTotal");
    const int64_t share = total > 0 ? total / 8 : room;
    const CheckpointSlots c = checkpoint_slots_auto(*pol, cfg_.max_seqs, ctx, interval, slot,
                                                    share, room);
    cfg_.checkpoint_slots = c.slots;

    const std::string need =
        fmt("%lld a sequence for --max-num-seqs %lld: %s keeps %lld for a %lld-token session at "
            "a %lld-token interval, and one more for a running turn", (long long)c.per_seq,
            (long long)cfg_.max_seqs, pol->name(), (long long)(c.per_seq - 1), (long long)ctx,
            (long long)interval);
    std::string bound;
    if (c.bound == CheckpointSlots::Bound::Share)
        bound = fmt("; %lld wanted, held to an eighth of host memory (%s); --checkpoint-slots N "
                    "sets more", (long long)c.want, humanb(share).c_str());
    else if (c.bound == CheckpointSlots::Bound::Room)
        bound = fmt("; %lld wanted, held to what the host can pin: %s", (long long)c.want,
                    limit.c_str());
    decide("--checkpoint-slots", std::to_string(c.slots), need + bound);
    if (c.slots == 0) {
        RAD_WARN("checkpoint slots: auto finds no room for one %s snapshot of %s (%s), so every "
                 "prefix hit on this model prefills from the start. Free host memory, or lower "
                 "--host-pool-mib or --prefix-cache-mib", what, humanb(slot).c_str(),
                 limit.c_str());
        return RAD_OK;
    }
    const std::string line =
        fmt("checkpoint slots: auto: %lld %s snapshot(s) of %s in pinned host memory (%s), none of "
            "it on the card; %s%s", (long long)c.slots, what, humanb(slot).c_str(),
            humanb(slot * c.slots).c_str(), need.c_str(), bound.c_str());
    if (c.bound == CheckpointSlots::Bound::Room) RAD_WARN("%s", line.c_str());
    else RAD_INFO("%s", line.c_str());
    return RAD_OK;
}

int Engine::configure_kv() {
    /* Resolved once, for the cache below and Scheduler::init later (see Engine::geo_). */
    RAD_TRY(chunk_geometry_resolve(ranks_[0]->program, cfg_, &geo_));
    if (geo_.checkpoint_interval != cfg_.checkpoint_interval)
        decide("--checkpoint-interval",
               geo_.checkpoint_interval ? std::to_string(geo_.checkpoint_interval) : "0 (none)",
               geo_.checkpoint_interval
                   ? fmt("rounded up to a multiple of %lld tokens, the step quantum (the attention "
                         "block and the linear-state chunk), so a checkpoint is a legal chunk split",
                         (long long)geo_.quantum)
                   : std::string("the model keeps no linear state to checkpoint"));
    /* One block table serves every rank (serve() checks it), so every rank carves the same number
     * of paged blocks. Under an uneven split the pools differ (more delta-net heads, or zero
     * attention weights), so each rank is planned first and all are capped at the least before
     * any carves. */
    int64_t least = -1;
    for (size_t i = 0; i < ranks_.size(); ++i) {
        auto& r = ranks_[i];
        LogRank lr(r->index);
        RAD_TRY(r->kv.plan_groups(r->program.kv_groups, cfg_));
        const int64_t t = r->kv.paged_tokens(r->pools.kv().avail());
        /* A rank plan() re-sized for the common token count must page at least that count (same
         * plan, same bytes); anything less is a broken invariant. */
        if (rank_cfg_[i].vram_kv_mib < cfg_.vram_kv_mib && t < kv_planned_tokens_) {
            RAD_ERR("kv: rank %d's KV pool was sized at %lld MiB to page the common %lld tokens "
                    "and pages %lld: the pool and the plan that sized it disagree", r->index,
                    (long long)rank_cfg_[i].vram_kv_mib, (long long)kv_planned_tokens_,
                    (long long)t);
            return RAD_E_STATE;
        }
        if (t == KVManager::kUnboundedTokens) continue;   /* holds no cache; limits nobody */
        if (least < 0 || t < least) least = t;
    }
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RAD_TRY(rad_dev_set(r->device));   /* allocates on the current device; see plan() */
        if (ranks_.size() > 1) {
            const int64_t own = r->kv.paged_tokens(r->pools.kv().avail());
            if (own > least && own != KVManager::kUnboundedTokens)
                RAD_INFO("kv: rank %d could page %lld tokens and carves %lld, the least rank's, so "
                         "that one block table serves every rank", r->index, (long long)own,
                         (long long)least);
            r->kv.cap_paged_tokens(least);
        }
        RAD_TRY(r->kv.carve(r->pools.kv()));
        /* Without virtual memory management the cache cannot lend: everything still works, but
         * the expert slab never grows, which no counter shows, so log it once. */
        if (r->index == 0 && !r->pools.kv().elastic())
            RAD_WARN("this device has no virtual memory management, so the KV cache holds its "
                     "whole carve and the expert slab has nothing to grow into. The experts were "
                     "given the card first and the cache one full-length sequence.");
        if (cfg_.debug_placement) fputs(r->kv.report().c_str(), stderr);
    }

    /* ---------------- the linear-state snapshot store ---------------- */
    /* Pinned host memory, not VRAM: no kernel reads a snapshot; it is copied out after the step
     * that produced it and back before a resume, a few milliseconds over the link each. One slot
     * holds every rank's shard at its own offset; rank 0's manager hands out ids. Opened only when
     * there is state to snapshot and a prefix cache. */
    {
        /* Shards can differ in size (heads need not divide evenly), so a slot is the sum of the
         * ranks' shards, each starting where the previous ends. */
        std::vector<int64_t> shard_off(ranks_.size(), 0);
        int64_t slot = 0;
        for (size_t i = 0; i < ranks_.size(); ++i) {
            shard_off[i] = slot;
            slot += ranks_[i]->kv.checkpoint_bytes();
        }
        /* A slot holds linear state or a trunk's window, never both (KVGroupPlan::win_snap). */
        const char* what = ranks_[0]->kv.has_window_snapshot() ? "sliding-window" : "linear-state";
        const bool sized_auto = cfg_.checkpoint_slots < 0;
        if (sized_auto) RAD_TRY(size_checkpoint_store(slot, what));
        const int64_t n = cfg_.checkpoint_slots > 0 ? cfg_.checkpoint_slots : 0;
        if (slot > 0 && n > 0 && cfg_.prefix_cache) {
            /* Pinned all at once and outside every pool. A store larger than the memory left would
             * push the process into reclaim, which stalls it without a word; one that cannot be had
             * would leave every prefix hit on this model prefilling from the start, which no counter
             * makes visible. Either stops startup. */
            const bool sized = n <= INT64_MAX / slot;
            auto refuse = [&](const std::string& reason) {
                RAD_ERR("checkpoint slots: %lld %s snapshot(s)%s of %s are %s of pinned host "
                        "memory, which %s. Lower --checkpoint-slots (a larger "
                        "--checkpoint-interval needs fewer for the same context), or pass "
                        "--checkpoint-slots 0 to serve with no %s snapshot, every prefix hit on "
                        "this model then prefilling from the start", (long long)n, what,
                        sized_auto ? " (auto)" : "",
                        humanb(slot).c_str(), sized ? humanb(slot * n).c_str() : "more than 8 EiB",
                        reason.c_str(), what);
                return RAD_E_NOMEM;
            };
            if (!sized) return refuse("no host has");
            std::string why;
            if (!lockable_room(slot * n, 0, host_headroom_bytes(), false, &why))
                return refuse("does not fit: " + why);
            if (ck_store_.open(slot, slot * n) != RAD_OK)
                return refuse(std::string("the host did not pin: ") + rad_dev_last_error());
            if (!sized_auto)
                RAD_INFO("checkpoint slots: %lld %s snapshot(s) of %s in pinned host memory (%s), "
                         "none of it on the card", (long long)n, what, humanb(slot).c_str(),
                         humanb(ck_store_.cap_bytes()).c_str());
        } else if (slot > 0 && cfg_.prefix_cache && !sized_auto) {
            RAD_INFO("checkpoint slots: --checkpoint-slots 0 keeps no %s snapshot, so every "
                     "prefix hit on this model prefills from the start", what);
        }
        for (auto& r : ranks_)
            r->kv.set_checkpoint_store(&ck_store_, shard_off[(size_t)(&r - &ranks_[0])]);
    }
    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RAD_TRY(r->prefix.configure(cfg_, &r->kv, geo_.checkpoint_interval));
    }

    /* ---------------- the idle session tiers (spec §7.3) ---------------- */
    /* After the prefix cache, since the tiers are sized in cache entries and only the configured
     * cache knows which groups an entry covers. No memory is committed unless the host tier has a
     * size, but the tiers are configured anyway because they also hold the session record the
     * sessions view reads. The flags gate only the mover, below. */
    {
        const bool tier_asked = cfg_.prefix_cache_mib > 0;
        const std::vector<int32_t>& cg = ranks_[0]->prefix.cached_groups();
        /* One slot holds every rank's shard of an entry; each rank writes only its window. Shards
         * can differ or be empty (uneven heads, rad_kv_group_zero), so a slot is the sum and each
         * window starts where the previous ends. TierExec::sizes uses the largest rank's entry,
         * since the staging ring is per rank. */
        std::vector<int64_t> rank_entry(ranks_.size(), 0), entry_off(ranks_.size(), 0);
        int64_t entry_bytes = 0, slot_bytes = 0;
        for (size_t i = 0; i < ranks_.size(); ++i) {
            for (int32_t g : cg) {
                const KVGroupPlan* p = ranks_[i]->kv.plan(g);
                if (p && p->paged()) rank_entry[i] += p->bytes_per_block;
            }
            entry_off[i] = slot_bytes;
            slot_bytes += rank_entry[i];
            entry_bytes = std::max(entry_bytes, rank_entry[i]);
        }

        /* Snapshots live in the checkpoint store (--checkpoint-slots); --prefix-cache-mib is for
         * blocks only. */
        const int64_t ck_slot_bytes = ck_store_.enabled() ? ck_store_.slot_bytes() : 0;
        if (cg.empty() || entry_bytes <= 0) {
            /* Warn only when a tier was asked for. */
            if (tier_asked)
                RAD_WARN("kv tiers: the prefix cache indexes no paged group, so there is nothing "
                         "to move. --prefix-cache-mib does nothing here.");
        } else {
            IdleTiers::Config tc;
            tc.host_cap_bytes  = cfg_.prefix_cache_mib << 20;
            tc.entry_bytes     = slot_bytes;
            tc.ckpt_bytes      = ck_slot_bytes;
            tc.ckpt_slots      = ck_store_.capacity();
            tc.block_tokens    = ranks_[0]->prefix.block_size();

            /* Policy on rank 0 (a session is not sharded); every rank moves its own shard. */
            IdleTiers& T = ranks_[0]->tiers;
            RAD_TRY(T.configure(tc));
            T.set_model(meta_.name && meta_.name[0] ? meta_.name : "(unnamed)");
            if (T.host_on()) {
                /* A conversation longer than the tier keeps only its start on the host; the rest
                 * is prefilled again once VRAM gives it up, which on a card whose experts do not
                 * fit is right after the turn. */
                const int64_t held = T.arena().capacity() * tc.block_tokens;
                const int64_t ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : meta_.n_ctx_train;
                if (held < ctx)
                    RAD_WARN("kv tiers: --prefix-cache-mib %lld holds %lld tokens of context (%s "
                             "per %lld-token block), less than one --max-model-len %lld "
                             "conversation: a longer one keeps its first %lld tokens in host "
                             "memory and is prefilled again past them once VRAM gives it up. "
                             "%lld MiB would hold one.",
                             (long long)cfg_.prefix_cache_mib, (long long)held,
                             humanb(slot_bytes).c_str(), (long long)tc.block_tokens,
                             (long long)ctx, (long long)held,
                             (long long)((ctx + tc.block_tokens - 1) / tc.block_tokens *
                                         slot_bytes / (1 << 20) + 1));
                /* Pass sizes are in bytes, not entries: an entry (one block per cached group) is
                 * 29 KiB on a 4-token-block hybrid and 4x that at 16 tokens, and a fixed 64-entry
                 * batch would take ~13 minutes to move a 200K session. A batch is 16 MiB of device
                 * staging per rank (about 1 ms of PCIe per pass), one gather per (group, layer)
                 * regardless of entry count. A copy out is four batches (~64 MiB per rank); it
                 * shares the transfer stream with the expert mover, so promotions wait at most a
                 * few milliseconds behind it. A restore is a whole chain up to the longest
                 * context. TierExec::sizes holds the arithmetic, which the VRAM budget uses. */
                const TierExec::Sizes tz = TierExec::sizes(
                    entry_bytes, cfg_.max_ctx > 0 ? cfg_.max_ctx : meta_.n_ctx_train,
                    ranks_[0]->prefix.block_size());
                const int max_batch = tz.max_batch;
                const int64_t max_copy = tz.max_copy, max_chain = tz.max_chain;
                bool all = true;
                for (auto& r : ranks_) {
                    /* Configure on the rank's own device (rad_dev_alloc uses the current one; a
                     * cross-card pointer fails as an opaque device error). The stream is the
                     * mover's: a fourth hardware queue costs an RDNA4 card 65% of its decode
                     * step (Mover::transfer_stream). */
                    RAD_TRY(rad_dev_set(r->device));
                    if (r->tierx.configure(&rad_tools_registry(), &r->kv, &T, cg,
                                           max_batch, max_copy, max_chain,
                                           /*leader=*/&r == &ranks_[0],
                                           r->device,
                                           entry_off[(size_t)(&r - &ranks_[0])],
                                           /*shared=*/r->mover.transfer_stream())
                            != RAD_OK) {
                        all = false;
                        break;
                    }
                }
                RAD_TRY(rad_dev_set(ranks_[0]->device));
                if (!all) {
                    /* The mover is off but the bookkeeping stays: close() forgets the entry and
                     * snapshot sizes that session sizes are computed from, so re-configure with
                     * zero caps. */
                    T.close();
                    tc.host_cap_bytes = 0;
                    RAD_TRY(T.configure(tc));
                } else {
                    /* The tiers can drop an entry on their own (capacity drop, failed promotion);
                     * the cache must hear of it or it would serve block ids it no longer owns. */
                    T.set_drop_sink([this](const BlockHash& h) {
                        ranks_[0]->prefix.drop_offdevice(h);
                    });
                    /* A capacity plan drops up to 512 MiB of the oldest host-only entries. */
                    pump_ = std::make_unique<TierPump>();
                    pump_->max_copy = max_copy;
                    pump_->slow_budget = (int)std::min<int64_t>(
                        1 << 20, std::max<int64_t>(64, (512ll << 20) / std::max<int64_t>(1, slot_bytes)));
                    RAD_INFO("%s", ranks_[0]->tierx.report().c_str());
                }
            }
            /* Attached even when nothing moves: a null `tiers_` would also disable the session
             * record. With zero caps every call is bookkeeping only. */
            ranks_[0]->prefix.set_tiers(&T);
            /* Logged whether the tiers are on or off. */
            RAD_INFO("%s", T.report().c_str());
        }
    }

    RAD_INFO("kv: %s", ranks_[0]->kv.report().c_str());
    /* The prefix cache's report: on or off, groups covered and skipped. It names the two things
     * that silently disable caching on a hybrid model: a windowed group cannot be served a prefix,
     * and a linear model with no checkpoint interval replays from zero. */
    RAD_INFO("%s", ranks_[0]->prefix.report().c_str());

    /* Can the pool hold one max-length sequence? The context limit and KV pool come from unrelated
     * flags; otherwise such a request is accepted and then refused at admission with a message
     * about blocks. Only full paged groups apply: a window group costs the window, and a linear
     * group is O(1) in context. */
    {
        const int64_t want = cfg_.max_ctx ? cfg_.max_ctx : meta_.n_ctx_train;
        const KVManager& kv = ranks_[0]->kv;
        for (int32_t g = 0; g < kv.n_groups(); ++g) {
            const KVGroupPlan* p = kv.plan(g);
            if (!p || !p->paged() || p->window_blocks > 0) continue;
            const int64_t cap = p->n_blocks * p->block_size;
            if (cap >= want) continue;
            /* A hard limit, unlike the backing ratio (how much of every-sequence-at-max_ctx the
             * pool backs, a provisioning choice reported in vram_budget.cpp): a pool that cannot
             * hold one max-length sequence accepts requests that can never complete.
             *
             * For an image generator the sequence is a conditioning, which its driver refuses at
             * admission; warn in those terms. */
            if (ranks_[0]->program.is_audio_generator()) {
                RAD_WARN("kv group '%s' holds %lld rows: a request needs its prompt and every "
                         "frame it may make, both branches, up to %lld rows a branch, so a long "
                         "piece is refused at once rather than cut short. Ask for shorter "
                         "pieces, use --kv-cache-dtype fp8, or add ranks.",
                         p->name.c_str(), (long long)cap, (long long)want);
                continue;
            }
            if (ranks_[0]->program.is_generator()) {
                RAD_WARN("kv group '%s' holds %lld rows: a conditioning longer than that -- a "
                         "prompt with its pictures, up to %lld rows -- is refused. A picture "
                         "costs a latent row per 16x16 pixels of the reference area; more ranks "
                         "give the cache more room.",
                         p->name.c_str(), (long long)cap, (long long)want);
                continue;
            }
            RAD_WARN("kv group '%s' holds %lld tokens but the context limit is %lld, so a "
                     "prompt longer than %lld tokens is refused (400) and a reply that grows past "
                     "it fails (kv_exhausted). Give the cache about %lld MiB "
                     "more (a larger --host-pool-mib takes experts off the card), use "
                     "--kv-cache-dtype fp8, or lower --max-model-len to %lld",
                     p->name.c_str(), (long long)cap, (long long)want, (long long)cap,
                     (long long)(((want - cap) / p->block_size * p->bytes_per_block) >> 20),
                     (long long)cap);
        }
    }
    return RAD_OK;
}

/* ================================================================== start */
int Engine::start() {
    startup_begin_ns_ = rad_mono_ns();
    std::vector<int> devs;
    std::string inventory;
    std::string devnote;
    /* Rank tags on log lines from here on (see LogRank). */
    log_rank_tags(cfg_.tp > 1);

    std::vector<std::string> dev_names;
    int s = usable_devices(&devs, &inventory, &devnote, &dev_names);
    /* Before any refusal, so a failed startup's report still identifies the cards. */
    identify_devices(dev_names);
    if (!devnote.empty()) RAD_WARN("device: %s", devnote.c_str());
    devnote_ = devnote;
    if (s < 0) { RAD_ERR("no usable device"); return s; }

    if ((int)devs.size() < cfg_.tp) {
        RAD_ERR("--tp %d needs %d devices of one architecture; %zu are usable [%s]",
                cfg_.tp, cfg_.tp, devs.size(), inventory.c_str());
        return RAD_E_DEVICE;
    }

    RAD_TRY(read_metadata());
    if (cfg_.max_ctx == 0)
        decide("--max-model-len", std::to_string(meta_.n_ctx_train),
               "the model's training context (--max-model-len 0)");
    RAD_TRY(load_vocab());
    /* Check kernel libraries' device code against the cards the ranks use, not device 0. */
    rad_tools_registry().set_code_device(devs[0]);
    rad_tools_registry().set_no_device_code(device_is_host());
    RAD_TRY(load_plugins());
    {
        /* The loaded libraries in selection order: as --kernels named them, or every library in
         * the home. */
        std::string libs;
        for (const PluginId& p : plugins_.kernels)
            libs += (libs.empty() ? "" : ",") + p.name + (p.version.empty() ? "" : "@" + p.version);
        decide("--kernels", libs.empty() ? "none" : libs,
               cfg_.kernel_hierarchy.empty()
                   ? "every kernel library under --radiance-home, in the loader's order (a reference "
                     "implementation last); architecture plugin " + plugins_.arch_plugin +
                         (plugins_.arch_version.empty() ? "" : "@" + plugins_.arch_version)
                   : "the libraries named, in that order; architecture plugin " + plugins_.arch_plugin +
                         (plugins_.arch_version.empty() ? "" : "@" + plugins_.arch_version));
    }

    ranks_.clear();
    for (int i = 0; i < cfg_.tp; ++i) {
        auto r = std::make_unique<Rank>();
        r->index  = i;
        r->device = devs[i];
        ranks_.push_back(std::move(r));
    }

    /* Peer access, measured before selection. Kernels that touch another card's memory constrain
     * device.p2p (rad_types.h); if any pair of the ranks' cards cannot reach each other, they are
     * not selected and a host-memory variant takes the op (or declare names the constraint),
     * rather than an all-reduce faulting or hanging. Every ordered pair is checked, since access is
     * granted one way at a time. --p2p off forces it off; --p2p on refuses to start without it. */
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
        if (cfg_.tp > 1 && cfg_.p2p == RAD_P2P_ON && !p2p) {
            RAD_ERR("device: --p2p on, but the driver reports no peer-to-peer access between cards "
                    "%s. IOMMU settings and the PCIe topology decide this, not radiance; --p2p auto "
                    "serves through host memory instead", blocked.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const bool forced_off = cfg_.p2p == RAD_P2P_OFF;
        if (forced_off) p2p = 0;
        rad_tools_registry().set_device_fact(RAD_FACT_P2P, p2p);
        p2p_measured_ = (int)p2p;
        p2p_blocked_ = blocked;
        if (cfg_.tp > 1)
            decide("--p2p", p2p ? "peer to peer" : "through pinned host memory",
                   p2p ? "the driver reports every pair of the ranks' cards can reach the other's memory"
                   : forced_off ? "--p2p off"
                                : "the driver reports no peer access between cards " + blocked);
        else
            decide("--p2p", "n/a", "one rank: no collective crosses a card");
        if (cfg_.tp > 1) {
            if (p2p) RAD_INFO("device: the %d cards reach each other's memory directly (peer to "
                              "peer)", cfg_.tp);
            else if (forced_off)
                RAD_INFO("device: --p2p off: the collectives exchange through pinned host memory%s",
                         blocked.empty() ? " although the driver reports peer-to-peer access"
                                         : "");
            else RAD_WARN("device: no peer-to-peer access between cards %s. The collectives "
                          "exchange through pinned host memory instead, and a kernel that needs "
                          "peer access (device.p2p == 1) is not selected; if nothing else "
                          "implements its op, startup stops below naming it. IOMMU settings and "
                          "the PCIe topology decide this, not radiance", blocked.c_str());
        }
    }

    RAD_TRY(declare());

    /* --debug-graph stops here: everything it prints is known after declare, and it must work on
     * a machine where the model does not fit. */
    if (cfg_.debug_graph) return RAD_OK;

    /* Before anything is placed or loaded, so serving at reference speed is refused up front. */
    RAD_TRY(rad_gate_reference_kernels(ranks_[0]->program, cfg_.accept_reference_kernels, "serve"));
    {
        /* Selection summary for the report: op bands served per library, and the attention rows by
         * name (the family whose choice varies with card, head width and context). */
        std::map<std::string, int> by_lib;
        std::set<std::string> attn;
        for (const OpInfo& op : ranks_[0]->program.ops)
            for (const Band& b : op.bands)
                for (int dm = 0; dm < RAD_N_DOMAINS; ++dm) {
                    const KernelRow* row = b.dom[dm].row;
                    if (!row || !row->info) continue;
                    ++by_lib[row->plugin + (dm == RAD_DOMAIN_HOST ? " (host)" : "")];
                    if (row->info->family && !std::strcmp(row->info->family, "attn"))
                        attn.insert(op.op + " -> " + row->info->name);
                }
        std::string libs, rows;
        for (const auto& [lib, n] : by_lib) libs += fmt("%s%s %d", libs.empty() ? "" : ", ", lib.c_str(), n);
        for (const std::string& a : attn) rows += (rows.empty() ? "" : "\n") + a;
        decide("kernel selection", libs.empty() ? "none" : libs + " op bands",
               rows.empty() ? std::string() : "attention:\n" + rows);
    }

    RAD_TRY(plan());
    RAD_TRY(load_weights());
    RAD_TRY(configure_kv());

    for (auto& r : ranks_) {
        LogRank lr(r->index);
        RAD_TRY(rad_dev_set(r->device));   /* allocates on the current device; see plan() */
        CtxDesc d{};
        d.program     = &r->program;
        d.rank        = r->index;
        d.world_size  = cfg_.tp;
        d.device      = r->device;
        /* Null for all-in-VRAM: the mover never runs and lookup uses WeightInfo::ptr. */
        d.residency   = r->plan.strategy == Strategy::AllVram ? nullptr : r->mover.residency();
        d.arch_step   = arch_step_;
        d.profile_ops = cfg_.profile_ops;
        bind_kv_pools(r->program, [&](int32_t g) { return r->kv.plan(g); }, &d.kv);
        d.registry    = &rad_tools_registry();
        RAD_TRY(r->ctx.init(d));

        /* The mover orders every transfer against the kernels reading its slot via an event on the
         * compute stream (Mover::record_gate), which exists only after Ctx::init. Bound even for a
         * static plan, which may still stage file-tier weights on demand. */
        RAD_TRY(r->mover.bind_compute_stream(r->ctx.stream()));

        /* The prefill stager, where the budget made room. Only a step larger than every smaller
         * arena level finds the region unlent, so that is the smallest pass it arms for. */
        if (r->program.arena_stage_bytes > 0 && r->plan.dynamic && !r->arena_levels.empty()) {
            /* The smallest pass is one past every level that lends the ring. The span's shared
             * level does not: it keeps its own place for it, right above its buffers and the
             * guard, wherever that is below the region at the top (stager.h). */
            int64_t top_level = 0;
            r->arena_low_off = -1;
            for (const ArenaLevel& lv : r->arena_levels) {
                if (lv.rows != arena_span_rows_) { top_level = std::max(top_level, lv.rows); continue; }
                const int64_t off = (lv.end + kArenaGuard + RAD_ALIGN_UNIT - 1) /
                                    RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
                r->arena_low_off = std::min(off, r->program.arena_stage_off);
            }
            char* const base = (char*)r->ctx.arena_base();
            RAD_TRY(r->stager.init(r->program, r->plan, &r->mover,
                                   base + r->program.arena_stage_off,
                                   r->arena_low_off >= 0 ? base + r->arena_low_off : nullptr,
                                   r->program.arena_stage_bytes, top_level + 1));
            if (r->stager.enabled() && r->index == 0)
                RAD_INFO("prefill staging: a %s ring at the top of the arena; a step of more "
                         "than %lld tokens copies the pooled experts of %zu routed layers onto "
                         "the card up to %d layers ahead of the layers that read them",
                         humanb(r->stager.ring_bytes()).c_str(), (long long)top_level,
                         r->stager.n_layers(), (int)PrefillStager::lead());
        }
        /* The file stager, wherever a routed expert is on the file tier (static or dynamic plan;
         * such a unit has no other address). */
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

    }
    /* Runtime LoRA: the startup adapters, copied into their slots now that every rank's pools
     * have an address and before any step can read them. */
    RAD_TRY(lora_start());

    /* ---- the loan: the cache's idle blocks, lent to the expert slab in place -----------------
     *
     * Here because it needs the cache's carve (the addresses) and the mover's compute stream
     * (which a recall drains). Every rank lends its own pool, same geometry on every card; only a
     * plan that moves units borrows. A strip is one layer's stripe of one lendable group, lent
     * from the top, since a group's top blocks are the top of every layer's stripe. */
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
    /* The next-layer expert prefetch, wherever experts sit in a pinned pool for routed GEMMs to
     * read over the link (core/runtime/xprefetch.h). Its stream and ring are in the VRAM budget
     * (xp_device_bytes, Engine::plan). Off under --deterministic. */
    if (!cfg_.deterministic)
        for (auto& r : ranks_) {
            RAD_TRY(rad_dev_set(r->device));
            uint64_t lo[kXpMaxPools], hi[kXpMaxPools], st[kXpMaxPools];
            const int n = r->mover.pool_ranges(lo, hi, st, kXpMaxPools);
            if (n > 0) r->ctx.xp_open(lo, hi, st, n);
        }
    (void)rad_dev_set(ranks_[0]->device);

    /* The most the cache can hold: the whole carve, including lent blocks (a request reclaims them
     * in the same step). Tokens are the smallest over the full paged groups; a windowed group
     * bounds history, not capacity. /stats and /metrics report the same numbers (kv_tokens_max,
     * radiance:kv_cache_max_tokens). */
    {
        const KVManager& kv = ranks_[0]->kv;
        int64_t bytes = 0, state = 0, tokens = 0;
        for (const KVGroupPlan& p : kv.plans()) {
            bytes += kv.pool_bytes_carved(p.index);
            if (p.stateful()) state += kv.pool_bytes_carved(p.index);
            if (!p.paged() || p.kind == RAD_KV_WINDOW || p.block_size <= 0) continue;
            const int64_t t = p.n_blocks * p.block_size;
            if (tokens == 0 || t < tokens) tokens = t;
        }
        const int64_t mlen = cfg_.max_ctx > 0 ? cfg_.max_ctx : meta_.n_ctx_train;
        const int64_t lent = lending_ ? ranks_[0]->mover.flex_capacity() : 0;
        if (bytes > 0)
            decide("kv cache",
                   fmt("%s a card: %lld tokens, %.1f sessions at %lld", humanb(bytes).c_str(),
                       (long long)tokens, mlen > 0 ? (double)tokens / (double)mlen : 0.0,
                       (long long)mlen),
                   fmt("%s of it is per-sequence state; %s", humanb(state).c_str(),
                       lent > 0 ? fmt("up to %s is lent to the experts while no request holds it",
                                      humanb(lent).c_str()).c_str()
                                : "nothing is lent: it is all the cache's"));
        if (bytes > 0)
            RAD_INFO("kv: the cache can grow to %s a card -- %lld tokens of context, %.1f sessions at "
                     "%lld -- with every block it lends the experts taken back; %s of that is "
                     "per-sequence state, and %s",
                     humanb(bytes).c_str(), (long long)tokens,
                     mlen > 0 ? (double)tokens / (double)mlen : 0.0, (long long)mlen,
                     humanb(state).c_str(),
                     lent > 0 ? fmt("up to %s is lent to the experts while no request holds it",
                                    humanb(lent).c_str()).c_str()
                              : "nothing is lent: it is all the cache's");
    }

    /* ---- the top of the activation arena, lent while steps are small ----------------------
     *
     * One stretch per rank, from the arena top down to the smallest level's end plus a guard. Each
     * level lends everything above its end and guard, stated in bytes from the top. Ranks' plans
     * can differ, so each rank's amounts are its own.
     *
     * The guard (1 MiB) catches a kernel writing past its rows, which would otherwise corrupt a
     * resident expert's weights. */
    arena_rows_.clear();
    if (!ranks_.empty() && !ranks_[0]->arena_levels.empty() && ranks_[0]->heat.enabled()) {
        bool ok = true;
        for (auto& r : ranks_) {
            RAD_TRY(rad_dev_set(r->device));
            r->arena_lent.assign(1, 0);
            const Program& P = r->program;
            const int64_t top = P.arena_bytes;
            for (const ArenaLevel& lv : r->arena_levels) {
                if (r->ctx.add_level(lv.rows, lv.offset, lv.decl) < 0) { ok = false; break; }
                /* The span's shared level keeps the stager's ring above its buffers: it lends
                 * what is above the ring. */
                const bool ring = lv.rows == arena_span_rows_ && r->arena_low_off >= 0 &&
                                  r->stager.enabled();
                const int64_t edge = ring
                    ? (r->arena_low_off + P.arena_stage_bytes + RAD_ALIGN_UNIT - 1) /
                          RAD_ALIGN_UNIT * RAD_ALIGN_UNIT
                    : (lv.end + kArenaGuard + RAD_ALIGN_UNIT - 1) / RAD_ALIGN_UNIT * RAD_ALIGN_UNIT;
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
    /* Immediate recall: when a request needs more cache than is in hand, rank 0's cache calls this
     * from the scheduler between dispatches, with every rank thread parked. Each rank's slab is
     * drained to the affordable loan before the blocks are handed out, and the followers' marks
     * move at once (a tier restore on a follower uses the id rank 0 just chose). */
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

    /* What the core allocated, to compare with the card's usage; the difference is the driver's
     * context and code objects and any plugin's own hipMalloc. */
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

    /* ---- the budget, checked against the card --------------------------------------------
     *
     * Everything is allocated now (pools, arena, step staging, grouped-GEMM weight tables), so the
     * card should have --gpu-headroom-mib plus what the budget kept for later.
     *
     * Too little means memory taken after the budget's reading (another process, a late driver
     * allocation). It comes out of the room a kernel's first launch needs for its code object and
     * scratch, and running out faults inside the HIP runtime.
     *
     * The warning tests the room (free less what is kept for later) against the engine's own
     * first-launch allowance (the flag's default), not free memory against the flag, which would
     * move both sides equally. A short card is warned with the flag value that fixes it.
     *
     * Too much means VRAM left unused, costly when experts do not fit; up to 256 MiB of slack
     * (MiB rounding, plugin workspaces) is silent. */
    const int64_t MiB = 1024 * 1024;
    const int64_t headroom = std::max<int64_t>(cfg_.gpu_headroom_mib, 0) * MiB;
    const int64_t allowance = Config{}.gpu_headroom_mib * MiB;
    for (size_t ri = 0; ri < ranks_.size(); ++ri) {
        auto& r = ranks_[ri];
        LogRank lr(r->index);
        RadDeviceProps props{};
        if (rad_dev_props(r->device, &props) < 0 || props.vram_bytes <= 0) continue;

        /* Both the headroom and the later allocations are still free here: serve() allocates
         * the step staging; the KV tiers' staging and a second lane are built on first use. */
        const int64_t late = late_step_bytes_ + late_lazy_bytes_;
        const int64_t want = headroom + late;
        const int64_t got  = props.vram_free;
        const int64_t room = got - late;
        const std::string asked =
            late > 0 ? fmt("%s (the %s --gpu-headroom-mib and %s kept for the step staging and "
                           "for what is built on first use)", humanb(want).c_str(),
                           humanb(headroom).c_str(), humanb(late).c_str())
                     : fmt("%s --gpu-headroom-mib", humanb(want).c_str());
        if (room < allowance) {
            const int64_t raise_mib = (allowance - room + MiB - 1) / MiB;
            RAD_WARN("card %d settled at %s free against %s, which leaves %s for a kernel's first "
                     "launch to load its code object and size its scratch in -- under the %s this "
                     "engine allows for its own. %s Set --gpu-headroom-mib to at least %lld: what "
                     "is left here rises with it one for one.",
                     r->device, humanb(got).c_str(), asked.c_str(),
                     humanb(std::max<int64_t>(room, 0)).c_str(), humanb(allowance).c_str(),
                     got < want
                         ? fmt("%s of the headroom was taken after the budget read the card, by "
                               "another process on it or an allocation the budget does not "
                               "count.", humanb(want - got).c_str()).c_str()
                         : "",
                     (long long)(std::max<int64_t>(cfg_.gpu_headroom_mib, 0) + raise_mib));
        } else if (got < want) {
            RAD_INFO("card %d settled at %s free against %s: %s of the headroom was taken after "
                     "the budget read the card, by another process on it or an allocation the "
                     "budget does not count, and %s is left for first launches",
                     r->device, humanb(got).c_str(), asked.c_str(), humanb(want - got).c_str(),
                     humanb(room).c_str());
        } else if (got > want + 256 * MiB) {
            /* Not a fault: neither pool could use more (e.g. a small dense model at its largest
             * context), reported on vram_budget.cpp's `unusable` line, or a re-split rank's
             * unusable remainder (the `kv: rank N ...` line). */
            const int64_t unclaimed_mib =
                (cfg_.vram_weights_mib + cfg_.vram_kv_mib) -
                (rank_cfg_[ri].vram_weights_mib + rank_cfg_[ri].vram_kv_mib);
            if (unclaimed_mib > 0)
                RAD_INFO("card %d settled at %s free against %s. The split accounted for it: %lld "
                         "MiB are what rank %d's weights could not use of what its KV pool did not "
                         "need, on the `kv: rank %d ...` line above, and the rest is the `unusable` "
                         "line of the VRAM budget.",
                         r->device, humanb(got).c_str(), asked.c_str(), (long long)unclaimed_mib,
                         r->index, r->index);
            else
                RAD_INFO("card %d settled at %s free against %s. The split accounted for it: "
                         "neither pool could address another byte, which is the `unusable` line of "
                         "the VRAM budget above.",
                         r->device, humanb(got).c_str(), asked.c_str());
        } else
            RAD_INFO("card %d settled at %s free, against %s",
                     r->device, humanb(got).c_str(), asked.c_str());
    }
    return RAD_OK;
}

const Program& Engine::program() const { return ranks_[0]->program; }

/* ================================================================== dump */
/* --debug-graph (spec §16); see Engine::dump in engine.h. */
int Engine::dump() {
    Program& p = ranks_[0]->program;
    fputs(dump_graph_str(p, nullptr).c_str(), stdout);
    if (cfg_.debug_selection) fputs(rad_tools_registry().selection_table().c_str(), stdout);
    /* The exit code reports device holes by declare()'s rule (Program::device_holes). */
    for (auto& r : ranks_)
        if (!r->program.device_holes(device_is_host()).empty()) return RAD_E_NOKERNEL;
    return RAD_OK;
}

/* ================================================================== 6c. the vocabulary */

/* The vocabulary comes from the model source: the container's vocab section, a GGUF header, or
 * tokenizer.json beside a checkpoint. A missing one is refused here rather than at the first
 * request (spec §4.2, §12). */
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
        /* GGUF names its pre-tokeniser; the chain is reconstructed from the name or refused (a
         * container converted with --tokenizer bakes in the declared one). */
        RAD_TRY(vocab_from_gguf(ckpt_->gguf(), vb));
    } else {
        /* With tokenizer_config.json and chat_template.jinja: beside config.json in a model
         * repository, in a diffusers pipeline's processor/ or tokenizer/ folder. */
        const std::string tj = ckpt_->tok_dir() + "/tokenizer.json";
        if (parse_tokenizer_json(tj.c_str(), vb) < 0) {
            RAD_ERR("%s: no readable tokenizer.json in %s", cfg_.model.c_str(),
                    ckpt_->tok_dir().c_str());
            return RAD_E_FORMAT;
        }
    }

    /* The operator's template replaces the container's before anything reads it, so the chat
     * endpoint and the derived reply format agree. An unreadable or empty file is refused rather
     * than silently falling back. */
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
        /* Strip trailing newlines, as for a checkpoint's chat_template.jinja, so the same file
         * renders the same prompt either way. */
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

    /* The checkpoint's end-of-generation ids, flattened from generation_config.json into metadata
     * as a space-separated list or one id. */
    if (const char* ids = rad_meta_gets(&meta_, "generation.eos_token_id", nullptr)) {
        const char* q = ids;
        while (*q) {
            char* end = nullptr;
            const long v = std::strtol(q, &end, 10);
            if (end == q) break;
            if (v >= 0 && v <= INT32_MAX) vb.eos_ids.push_back((int32_t)v);
            q = end;
        }
    }

    auto v = std::make_shared<Vocab>();
    RAD_TRY(v->load(std::move(vb)));
    vocab_ = v;
    if (file_ && file_->aux_vocab()) {
        VocabBuild ab;
        RAD_TRY(vocab_deserialize(file_->base(), (uint64_t)file_->size(), file_->aux_vocab().h, ab));
        auto av = std::make_shared<Vocab>();
        RAD_TRY(av->load(std::move(ab)));
        aux_vocab_ = av;
        RAD_INFO("vocab: a second vocabulary of %d tokens", aux_vocab_->n_tokens());
    }

    auto view = std::make_unique<TokenView>();
    RAD_TRY(view->init(vocab_));
    vocab_view_ = std::move(view);

    RAD_INFO("vocab: %d tokens, bos=%d eos=%d%s", vocab_->n_tokens(), vocab_->bos(), vocab_->eos(),
             vocab_->chat_template().empty() ? ", no chat template" : "");

    /* Build the grammar engine's vocabulary index now rather than on the first grammar request
     * (nearly every tool-calling request has one), so admission pays only for its own grammar. */
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
