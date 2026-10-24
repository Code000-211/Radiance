/* device.h -- the device layer's internal face.
 *
 * `abi/rad_device.h` is what the rest of the engine and every plugin calls. This header is
 * the seam between the two implementations of it, plus the two questions the rest of the core
 * actually needs to ask about which one won.
 *
 * The seam is a vtable rather than an `#ifdef` at every call site. That is a deliberate trade: an
 * indirect call costs a load and a jump, which is unmeasurable against a `hipMemcpyAsync` and
 * unmeasurable against a `memcpy` of anything worth enqueuing, and in exchange there is exactly one
 * definition of each `rad_dev_*` symbol and no way for the two backends to drift into implementing
 * different contracts. Two device layers that have silently diverged is the failure this avoids,
 * and it is the failure that matters, because the host backend is where everything above the
 * kernels is developed (docs/IMPLEMENTATION.md) and the HIP backend is where it has to still work.
 */
#pragma once
#include "rad_abi.h"
#include "rad_device.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

/* AN EVENT THAT ONLY ORDERS WORK ON ONE CARD. Its record makes the recording stream's writes
 * visible to the rest of that card and to nothing else, so it may be waited on by another stream
 * of the same card and queried or synchronised by the host for WHEN, but the host must not read
 * device-written memory on the strength of it, and no other card may wait on it. What that buys
 * is the release: making writes visible to the system waits for every write still in flight to
 * host memory, and with the link busy streaming weights that is most of a microsecond at best and
 * over ten at worst on every record, where the card-wide release costs the packet alone. */
extern "C" int rad_event_create_local(RadEvent* out);
/* WHAT AN EVENT IS FOR, fixed when it is made. RAD_EVENT_LOCAL is rad_event_create_local's scope.
 * RAD_EVENT_HOST_WAIT is an event the host synchronises on: its records complete with an interrupt
 * the waiting thread blocks on, where the other kind complete silently and can only be waited on by
 * a stream. rad_event_create is RAD_EVENT_HOST_WAIT, rad_event_create_local is RAD_EVENT_LOCAL
 * alone, and rad_event_sync refuses an event made without RAD_EVENT_HOST_WAIT. */
enum : unsigned {
    RAD_EVENT_LOCAL     = 1u,
    RAD_EVENT_HOST_WAIT = 2u,
};
extern "C" int rad_event_create_as(RadEvent* out, unsigned flags);
/* The host has just written memory a kernel issued after this call reads, and no host wait came
 * between: a host-site op's output, read over the link by the device op after it. */
extern "C" void rad_dev_host_wrote(void);

/* ONE RANGE OF A RANGED COPY: `bytes` from byte `src` of the source to byte `dst` of the
 * destination, all three multiples of four. */
struct RadCopyRange {
    int64_t src;
    int64_t dst;
    int64_t bytes;
};
/* A FIXED PREFIX AND A LIST OF RANGES, IN ONE DISPATCH: src[0, prefix) to dst[0, prefix) -- a
 * multiple of sixteen, and it may be zero -- and then every range. The copy reads `ranges` when it
 * runs, so the list lives in memory the device reaches (pinned host memory beside `src`) and stays
 * as written until the stream has passed the copy, like `src` itself. One workgroup copies one
 * range, so a caller cuts a long run into ranges of a few kilobytes. */
extern "C" int rad_memcpy_ranges_async(void* dst, const void* src, int64_t prefix,
                                       const RadCopyRange* ranges, int n, RadStream s);

namespace radiance { struct RadKVCompactParams; }

namespace rad {

/* A recorded pass, owned by the backend that recorded it. See DeviceBackend::tape_begin. */
typedef void* RadTape;

/* ------------------------------------------------------------------ who won */
/* Decided once, at first use, and cached. Callers: the planner (a host backend has no VRAM tier
 * worth planning against, only a budget), the selector (RAD_DOMAIN_HOST is the only domain that
 * can run), and every test that must skip a device-only assertion rather than fail it. */
bool        device_is_host();
const char* device_backend_name();      /* "host" or "aql" */
/* One stream-ordered dispatch per pool; moves execute in list order within each layer/head.
 * Host runtime selection remains valid even when the core was built with HIP. */
int kv_compact_winning_branch(const radiance::RadKVCompactParams& params, RadStream stream);

/* ------------------------------------------------------------------ two contract notes
 *
 * These are properties of rad_device.h that the frozen header does not state and that
 * tests/device_test.cpp pins against a real card. Written down here rather than left to be
 * rediscovered:
 *
 *  1. `rad_memset_async` requires a DEVICE-ACCESSIBLE destination: RAD_MEM_DEVICE,
 *     RAD_MEM_HOST_PINNED or RAD_MEM_HOST_MAPPED. Plain RAD_MEM_HOST is pageable memory the GPU
 *     has never been shown and hipMemsetAsync refuses it. The host backend cannot reproduce that
 *     refusal, so this is the one asymmetry between the two -- see the comment on d_memset_async
 *     in hip.cpp for why it is reported rather than papered over. `rad_memcpy_async` has no such
 *     restriction: hipMemcpyDefault handles pageable memory on both ends.
 *
 *  2. A null RadStream is refused by BOTH backends. HIP's stream 0 serialises against every other
 *     blocking stream in the process, which is the opposite of spec §5.4's requirement that a
 *     transfer involve no host synchronisation; the engine creates its streams and never uses the
 *     default one, so passing null is a bug and the host backend reports it as one as well.
 */

/* ------------------------------------------------------------------ the backend vtable */
/* One entry per call in rad_device.h. `name` is what device_backend_name() returns. */
struct DeviceBackend {
    const char* name;

    int   (*count)(void);
    int   (*set)(int device);
    int   (*props)(int device, RadDeviceProps* out);
    int   (*enable_peer)(int self, int peer);

    void* (*alloc)(int64_t bytes, int kind);
    void  (*free)(void* p, int kind);
    void* (*host_ptr)(void* p);
    void* (*device_ptr)(void* p);
    int   (*host_register)(void* p, int64_t bytes, unsigned flags);
    int   (*host_unregister)(void* p);

    int   (*stream_create)(RadStream* out, int high_priority);
    void  (*stream_destroy)(RadStream s);
    int   (*stream_sync)(RadStream s);

    /* `flags`: see rad_event_create_as. A backend with no scope to choose ignores RAD_EVENT_LOCAL. */
    int   (*event_create)(RadEvent* out, unsigned flags);
    void  (*event_destroy)(RadEvent e);
    int   (*event_record)(RadEvent e, RadStream s);
    int   (*event_wait)(RadStream s, RadEvent e);
    int   (*event_query)(RadEvent e);
    int   (*event_sync)(RadEvent e);
    int   (*event_elapsed_ms)(RadEvent a, RadEvent b, float* out);

    /* The host has written memory a kernel issued from here on will read, without a host wait
     * having come between: whatever makes the device see host writes after a wait has to be done
     * now as well. */
    void  (*host_wrote)(void);

    int   (*memcpy_async)(void* dst, const void* src, int64_t bytes, RadStream s);
    int   (*memcpy_2d_async)(void* dst, int64_t dpitch, const void* src, int64_t spitch,
                             int64_t width, int64_t height, RadStream s);
    int   (*memset_async)(void* dst, int value, int64_t bytes, RadStream s);
    int   (*memcpy_ranges_async)(void* dst, const void* src, int64_t prefix,
                                 const RadCopyRange* ranges, int n, RadStream s);

    /* ---- recorded passes. A backend with none leaves these null, and the run phase then issues
     * every pass; see Ctx::run_step for what a tape is for and when one is kept.
     *
     * `tape_begin` opens a capture of every submission THIS THREAD makes -- dispatches, event
     * records and waits -- while `tape_pause` is not in effect (it nests). `tape_mark` is the
     * number of records so far, which is how the caller cuts a tape into the stretches between
     * its own host work. `tape_end` closes the capture and returns the tape, or null when `keep`
     * is 0 or the pass did something a tape cannot reproduce (the reason is in
     * rad_dev_last_error). `tape_play` submits records [from, to) again, onto the queues they
     * were recorded on. `tape_destroy` may be called while a play is still queued; the backend
     * frees the tape once nothing can read it. */
    int     (*tape_begin)(void);
    int     (*tape_mark)(void);
    void    (*tape_pause)(int on);
    RadTape (*tape_end)(int keep);
    int     (*tape_len)(RadTape t);
    int     (*tape_play)(RadTape t, int from, int to);
    /* The first record at which two tapes differ, -1 when they agree: the same submissions to the
     * same queues, the same packets but for where their arguments live, the same argument bytes.
     * What differs is described into `why`. */
    int     (*tape_diff)(RadTape a, RadTape b, char* why, int cap);
    void    (*tape_destroy)(RadTape t);
    /* Record `i` of a tape: what it is (0 a dispatch, 1 an event record or wait) and the queue it
     * goes to. And the one edit a closed tape takes: whether dispatch `i` is played WITHOUT the
     * barrier bit, launching while the packets in front of it on its queue are still running.
     * The caller sets it only where it has shown the dispatch touches nothing those packets write,
     * and on at most kTapeOverlapRun dispatches in a row; see Ctx::haz_mark. */
    int     (*tape_record)(RadTape t, int i, int* kind, uint32_t* queue);
    int     (*tape_set_overlap)(RadTape t, int i, int on);


    /* ---- elastic memory: an address range whose SIZE IS FIXED AND WHOSE BACKING IS NOT.
     *
     * A pool allocated whole owns its bytes for the life of the process, which is the right trade
     * for a pool whose demand is known at startup and the wrong one for two pools that take
     * turns. The KV cache and the expert slab are that pair: the cache holds what the live
     * sessions hold and nothing more, and a byte it is not holding is a byte the expert plane
     * would rather have resident than stream over the link.
     *
     * WHY THIS IS A PRIMITIVE AND NOT FREE-THEN-REALLOCATE. The KV pool is layer-major
     * (core/mem/kv.h): layer i lives at `base + i * n_blocks * layer_bytes_per_block`, so the
     * block count is baked into every layer's base address and the unused blocks are a stripe
     * inside each layer rather than a tail anyone could hand back. Reserving the address range at
     * its maximum and backing only part of it leaves every stride where the kernels already
     * expect it while the physical pages come and go underneath.
     *
     * A backend without elastic memory leaves these null, and everything above it sizes its pools
     * once and lives with the split: the absence is a configuration rather than a degradation.
     *
     * The contract:
     *
     *   `vmem_reserve` takes address space and no pages.
     *
     *   `vmem_commit` backs ONE GRANULE and `vmem_decommit` unbacks one. A MAPPING IS UNMAPPED
     *   EXACTLY AS IT WAS MAPPED: a driver handed four granules in one call refuses to unmap the
     *   second of them on its own. VMem below therefore issues one call per granule and never a
     *   run, which is what makes the granule the unit a pool can grow and shrink by.
     *
     *   `vmem_protect` grants this device read/write access to a range that is already backed. It
     *   is separate from the commit because the range to name is not the one just mapped: access
     *   set over a granule adjacent to an already-accessible one is refused where the whole
     *   contiguous run containing both is accepted, so VMem names the run.
     *
     *   `vmem_release` takes a range with nothing committed in it.
     *
     * The bookkeeping that keeps a commit off a commit lives in VMem below, once, rather than in
     * each backend. */
    int64_t (*vmem_granularity)(void);                     /* commit quantum; 0 = unsupported */
    void*   (*vmem_reserve)(int64_t bytes, int64_t align);
    void    (*vmem_release)(void* base, int64_t bytes);
    int     (*vmem_commit)(void* addr, int64_t bytes);
    int     (*vmem_decommit)(void* addr, int64_t bytes);
    int     (*vmem_protect)(void* addr, int64_t bytes);

    /* ---- observation. The VRAM in use on a card, every process's, as the driver counts it, read
     * without entering the device runtime: a dashboard polls it, and a runtime call can wait on
     * a lock the launch path holds. Available once props() has been read for the device;
     * RAD_E_UNSUPPORTED before that, and from a backend with no such counter. Last in the table
     * so that a backend listing its entries in order may leave it out. */
    int     (*vram_used)(int device, int64_t* out);

    /* ---- plugin device code. A backend that loads device code out of the fat binaries kernel
     * plugins carry counts each registration; `code_mark` is how many there have been, and
     * `code_runs` answers whether every one in [from, to) holds a code object `device` runs (1),
     * not (0, with the reason in `why`), or cannot say (< 0). The plugin loader brackets each
     * dlopen with marks, so the run is exactly that plugin's. Null on a backend that loads no
     * device code. */
    size_t  (*code_mark)(void);
    int     (*code_runs)(size_t from, size_t to, int device, char* why, int cap);

    /* ---- 1 when the device is integrated graphics: its memory is the host's, so what it reports
     * as device memory is shared with every process on the machine. Null on a backend with no such
     * devices, which answers 0. */
    int     (*integrated)(int device);

    /* ---- 1 when `self` can read and write `peer`'s memory directly, 0 when it cannot, < 0 when
     * the backend cannot say. Asked, not enabled: rad_dev_enable_peer is the call that maps the
     * memory. Null on a backend whose devices share one address space, which answers 1. */
    int     (*peer_access)(int self, int peer);
};


const DeviceBackend* host_backend(void);

/* The current backend's recorded passes; RAD_E_UNSUPPORTED (or null) from one that has none. */
bool    tape_supported();
int     tape_begin();
int     tape_mark();
void    tape_pause(bool on);
RadTape tape_end(bool keep, bool compare_only = false);
int     tape_len(RadTape t);
int     tape_play(RadTape t, int from, int to);
int     tape_diff(RadTape a, RadTape b, std::string* why);
void    tape_destroy(RadTape t);
int     tape_record(RadTape t, int i, int* kind, uint32_t* queue);
int     tape_set_overlap(RadTape t, int i, bool on);

/* The current backend's DeviceBackend::peer_access. */
int     dev_peer_access(int self, int peer);

/* The current backend's DeviceBackend::vram_used. */
int     dev_vram_used(int device, int64_t* out);

/* The current backend's DeviceBackend::code_mark and code_runs. A backend without them has no
 * device code to check: the mark is 0 and every plugin runs (1). */
size_t  dev_code_mark();
int     dev_code_runs(size_t from, size_t to, int device, std::string* why);

/* The current backend's DeviceBackend::integrated. */
bool    dev_integrated(int device);

/* THE LONGEST RUN OF DISPATCHES A PLAYED TAPE LEAVES UNORDERED. Of any kTapeOverlapRun + 1
 * consecutive packets on a queue, one carries the barrier bit, so a packet having been launched
 * still proves that everything more than that many packets before it has completed -- which is
 * what the backend's argument slots and a tape's own lifetime are reclaimed on. */
constexpr int kTapeOverlapRun = 32;

/* ------------------------------------------------------------------ the allocation ledger
 * What rad_dev_alloc has handed out, by RAD_MEM_* kind, since the process started. See the
 * comment on the ledger in dispatch.cpp for what it is for and -- more to the point -- what it
 * cannot see: a kernel plugin that calls hipMalloc directly never reaches this file. */
void rad_dev_alloc_totals(int kind, int64_t* bytes, int64_t* count);

/* Event handles created and not yet destroyed, process-wide. */
int64_t rad_dev_live_events(void);

/* ------------------------------------------------------------------ elastic memory
 * One reserved address range and the record of which granules inside it are backed.
 *
 * The interval bookkeeping is here rather than in the backends because the driver refuses a
 * commit that overlaps a commit and a decommit that covers a hole, and two implementations of
 * "which parts of this range are already backed" is two places for that answer to drift. The
 * backends get granule-aligned, disjoint ranges and no decisions.
 *
 * THE TWO ROUNDINGS GO OPPOSITE WAYS, AND IT IS NOT A DETAIL. A commit rounds OUTWARD, so the
 * bytes the caller asked for are certainly backed; a decommit rounds INWARD, so a granule holding
 * one byte the caller still wants is never taken away. The alternative -- one rounding for both --
 * either strands memory or unmaps live data, and the second reads as a kernel fault at an address
 * that was valid a moment ago.
 *
 * Nothing here is thread-safe. A range is owned by whatever owns the pool it backs, and the
 * commit/decommit path is a policy decision, not a step-path one. */
class VMem {
public:
    /* Whether this process's backend has elastic memory at all. Callers size their pools whole
     * when it does not: see the DeviceBackend comment for why that is a supported configuration. */
    static bool    supported();
    static int64_t granularity();          /* 0 when unsupported */

    VMem() = default;
    ~VMem() { release(); }
    VMem(const VMem&) = delete;
    VMem& operator=(const VMem&) = delete;
    /* Movable, because an elastic range is owned by whatever structure holds it and those
     * structures live in vectors. The source is left naming nothing, so the destructor it still
     * gets does not unmap a range the destination now owns. */
    VMem(VMem&& o) noexcept
        : base_(o.base_), size_(o.size_), committed_(o.committed_), ext_(std::move(o.ext_)) {
        o.base_ = nullptr;
        o.size_ = o.committed_ = 0;
        o.ext_.clear();
    }
    VMem& operator=(VMem&& o) noexcept {
        if (this != &o) {
            release();
            base_ = o.base_;
            size_ = o.size_;
            committed_ = o.committed_;
            ext_ = std::move(o.ext_);
            o.base_ = nullptr;
            o.size_ = o.committed_ = 0;
            o.ext_.clear();
        }
        return *this;
    }

    /* Address space only: no pages, no VRAM, nothing the card notices. `align` defaults to the
     * granularity. Reserving large is the point -- the range is the MAXIMUM the pool may ever
     * reach, because it is the one thing that cannot be changed later without moving addresses. */
    int   reserve(int64_t bytes, int64_t align = 0);
    void  release();                       /* decommits whatever is still backed, then frees the VA */

    void*   base() const { return base_; }
    int64_t size() const { return size_; }
    int64_t committed() const { return committed_; }   /* bytes actually backed right now */

    /* Both are idempotent and both clamp to the range. commit() backs every granule touching
     * [off, off+bytes) that is not backed; decommit() unbacks every granule wholly inside it that
     * is. Either may do nothing and report success. */
    int  commit(int64_t off, int64_t bytes);
    int  decommit(int64_t off, int64_t bytes);

    /* Is every byte of [off, off+bytes) backed? The question a caller asks before handing the
     * range to a kernel. */
    bool is_committed(int64_t off, int64_t bytes) const;

private:
    void*   base_ = nullptr;
    int64_t size_ = 0;
    int64_t committed_ = 0;
    /* Sorted, disjoint and never adjacent: a commit that touches an existing extent merges into
     * it, so the list length tracks the number of live REGIONS and not the number of calls. */
    std::vector<std::pair<int64_t, int64_t>> ext_;

    /* Adds a newly mapped piece and merges it with whatever it touches. */
    void add_extent(int64_t off, int64_t bytes);
    void drop_extent(int64_t off, int64_t bytes);
};

/* What every VMem in this process has backed, for the receipt Engine::start prints. Device memory
 * committed this way never reaches the rad_dev_alloc ledger -- it is not an allocation -- so
 * without this line the card's own accounting and the engine's disagree by exactly the elastic
 * pools. */
int64_t rad_dev_vmem_committed(void);

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
/* Null when the build has HIP but the machine shows no device. A build that can talk to a GPU and
 * a machine that has one are separate facts, and conflating them is how a CI container that
 * happens to have ROCm headers installed fails at the first hipMalloc instead of running the host
 * backend it was always going to run. The configure decides which kernels are COMPILED; this
 * decides whether there is a card to run them on, and neither answer implies the other. */
const DeviceBackend* hip_backend_if_present(void);
/* The HIP backend with its streams, events, copies and fills replaced by AQL packets the engine
 * writes itself (aql.cpp). Aborts, naming why, on a machine it cannot run on. */
const DeviceBackend* aql_backend_if_present(void);
/* Bytes copied to or from pageable host memory so far: every such copy is a host wait. */
int64_t aql_pageable_bytes(void);
#endif

/* ------------------------------------------------------------------ the error slot */
/* Both backends write here; rad_dev_last_error() drains it. Thread-local, because spec §1 puts one
 * thread per tensor-parallel rank and the mover on its own: a fault on the mover's thread reported
 * as the compute rank's would send the operator to the wrong half of the engine. */
void device_set_error(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

}  /* namespace rad */
