#include "runtime/tierexec.h"

#include "device/device.h"
#include "rad_internal.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace rad {

TierExec::~TierExec() { close(); }

/* gather_rows and scatter_rows are resolved once here, at the width they are called at (`n` is the
 * per-group fragment in bytes). Nothing re-resolves on the maintenance path, so the kernel serving
 * a transfer cannot change in the middle of a batch. */
int TierExec::configure(Registry* reg, KVManager* kv, IdleTiers* tiers,
                        const std::vector<int32_t>& groups, int max_batch, int64_t max_copy,
                        int64_t max_chain, bool leader, int device, int64_t slot_off,
                        RadStream shared) {
    close();
    if (!reg || !kv || !tiers || groups.empty() || max_batch <= 0 || max_copy <= 0 ||
        max_chain <= 0)
        return RAD_E_INVAL;
    reg_ = reg; kv_ = kv; tiers_ = tiers; leader_ = leader; device_ = device;
    max_batch_ = max_batch; max_copy_ = max_copy; max_chain_ = max_chain;
    slot_off_ = slot_off;
    slot_bytes_    = tiers->entry_bytes();
    shared_        = shared;
    RAD_TRY(rad_dev_set(device_));

    int64_t off = 0;
    for (int32_t g : groups) {
        const KVGroupPlan* p = kv->plan(g);
        if (!p || !p->paged()) continue;
        GroupInfo gi;
        gi.g = g;
        gi.n_layers = p->n_layers;
        gi.frag = p->layer_bytes_per_block();
        gi.layer_stride = p->layer_stride();
        gi.n_blocks = p->n_blocks;
        gi.base = p->base;
        gi.off = off;
        if (p->zero) {
            gi.zero = true;
            gi_.push_back(gi);
            continue;
        }
        /* The kernel is picked on the fragment width (libr4d's byte-row pair needs n % 4 == 0).
         * An unsupported width fails here rather than at launch inside a batch. */
        Geometry q;
        q.set_s("dtype", "u8");
        q.set_i("n", gi.frag);
        q.set_i("M", max_batch);
        gi.gather  = reg->resolve("gather_rows",  q, RAD_DOMAIN_DEVICE);
        gi.scatter = reg->resolve("scatter_rows", q, RAD_DOMAIN_DEVICE);
        /* On the host backend the pool, the staging and the arena are all host memory, and the
         * kernel libraries (libavx, libref) register host rows only: run those, as ops with no
         * device kernel do there (Ctx's domain choice). */
        if ((!gi.gather || !gi.scatter) && device_is_host()) {
            gi.gather  = reg->resolve("gather_rows",  q, RAD_DOMAIN_HOST);
            gi.scatter = reg->resolve("scatter_rows", q, RAD_DOMAIN_HOST);
            gi.host = true;
        }
        if (!gi.gather || !gi.scatter) {
            RAD_WARN("kv tiers: no device gather/scatter for group %d at a %lld-byte fragment, "
                     "so a finished conversation cannot leave VRAM. The tiers are off.",
                     (int)g, (long long)gi.frag);
            close();
            return RAD_E_UNSUPPORTED;
        }
        off += gi.n_layers * gi.frag;
        gi_.push_back(gi);
    }
    if (gi_.empty()) return RAD_E_UNSUPPORTED;
    /* Every cached group is zero on this rank, so its entry would be empty. One rank that cannot
     * take part turns the tiers off for all. */
    if (off == 0) {
        RAD_WARN("kv tiers: this rank keeps no cache for any group the prefix cache holds "
                 "(rad_kv_group_zero), so it has no part of an entry to move. The tiers are off.");
        close();
        return RAD_E_UNSUPPORTED;
    }
    entry_bytes_ = off;

    /* Nothing is built on the device here; see device_init(). */
    verify_ = leader && std::getenv("RADIANCE_DEBUG_TIER_VERIFY") != nullptr;

    char buf[320];
    std::snprintf(buf, sizeof buf,
                  "kv tier transfer: %zu paged group(s), %s an entry, batches of %d through "
                  "gather_rows/scatter_rows at u8",
                  gi_.size(), humanb(entry_bytes_).c_str(), max_batch_);
    report_ = buf;
    enabled_ = true;
    return RAD_OK;
}

TierExec::Sizes TierExec::sizes(int64_t entry_bytes, int64_t max_ctx, int64_t block_size) {
    Sizes s;
    if (entry_bytes <= 0) return s;
    /* 16 MiB of staging per rank, clamped to 64..4096 entries (rationale at the bringup call site).
     * A copy out is four batches; a restore is a whole chain as long as the longest context. */
    s.max_batch = (int)std::min<int64_t>(4096, std::max<int64_t>(64, (16ll << 20) / entry_bytes));
    s.max_copy  = 4ll * s.max_batch;
    const int64_t bs = std::max<int64_t>(1, block_size);
    s.max_chain = (std::max<int64_t>(1, max_ctx) + bs - 1) / bs + 1;
    return s;
}

/* What device_init() allocates, each allocation rounded to the VRAM pool granule. */
int64_t TierExec::device_bytes(int64_t entry_bytes, int n_groups, const Sizes& s) {
    if (entry_bytes <= 0 || n_groups <= 0 || s.max_batch <= 0) return 0;
    const int64_t idx_n = (int64_t)n_groups * (s.max_copy + (int64_t)kRing * s.max_chain);
    return align_up((int64_t)s.max_batch * entry_bytes, (int64_t)RAD_ALIGN_POOL) +
           align_up(idx_n * (int64_t)sizeof(int32_t), (int64_t)RAD_ALIGN_POOL);
}

void TierExec::close() {
    /* Drain before freeing buffers the stream may still read. */
    if (stream_) (void)rad_stream_sync(stream_);
    if (stream_ && own_stream_) rad_stream_destroy(stream_);
    stream_ = nullptr; shared_ = nullptr; own_stream_ = false;
    release_device();
    copy_out_ = false;
    gi_.clear();
    order_.clear();
    verify_pending_.clear();
    entry_bytes_ = 0;
    enabled_ = false;
    leader_ = false;
    reg_ = nullptr; kv_ = nullptr; tiers_ = nullptr;
    report_.clear();
}

void TierExec::release_device() {
    if (stage_)    { rad_dev_free(stage_, RAD_MEM_DEVICE); stage_ = nullptr; }
    if (idx_dev_)  { rad_dev_free(idx_dev_, RAD_MEM_DEVICE); idx_dev_ = nullptr; }
    if (idx_host_) { rad_dev_free(idx_host_, RAD_MEM_HOST_PINNED); idx_host_ = nullptr; }
    for (int i = 0; i < kRing; ++i) {
        if (ring_ev_[i]) rad_event_destroy(ring_ev_[i]);
        ring_ev_[i] = nullptr;
        ring_used_[i] = false;
    }
    if (ev_compute_) { rad_event_destroy(ev_compute_); ev_compute_ = nullptr; }
    if (ev_copy_)    { rad_event_destroy(ev_copy_);    ev_copy_ = nullptr; }
    if (ev_restore_) { rad_event_destroy(ev_restore_); ev_restore_ = nullptr; }
}

/* Builds the device side on first use, never at bringup (same rule as Ctx::lane_init()). Built
 * eagerly, it makes every decode step 65% longer even when nothing ever moves: on two R9700s with
 * Flash-Next at 200K context, 20.5 -> 33.9 ms/step, all of it device time. The cost is not the
 * transfers, the pinned arena size or the staging VRAM; it is the extra hardware queue (see
 * stream_ in tierexec.h). */
int TierExec::device_init() {
    if (stream_) return RAD_OK;
    RAD_TRY(rad_dev_set(device_));
    const int64_t ng = (int64_t)gi_.size();
    copy_idx_  = max_copy_;
    chain_idx_ = max_chain_;
    const int64_t idx_n = ng * (copy_idx_ + (int64_t)kRing * chain_idx_);
    stage_    = rad_dev_alloc((int64_t)max_batch_ * entry_bytes_, RAD_MEM_DEVICE);
    idx_dev_  = (int32_t*)rad_dev_alloc(idx_n * (int64_t)sizeof(int32_t), RAD_MEM_DEVICE);
    idx_host_ = (int32_t*)rad_dev_alloc(idx_n * (int64_t)sizeof(int32_t), RAD_MEM_HOST_PINNED);
    bool ev = rad_event_create(&ev_compute_) == RAD_OK && rad_event_create(&ev_copy_) == RAD_OK &&
              rad_event_create(&ev_restore_) == RAD_OK;
    for (int i = 0; i < kRing && ev; ++i) ev = rad_event_create(&ring_ev_[i]) == RAD_OK;
    if (!stage_ || !idx_dev_ || !idx_host_ || !ev) { release_device(); return RAD_E_NOMEM; }
    /* Borrow the mover's maintenance stream if there is one; both move bytes nobody waits for,
     * and a fourth hardware queue costs about 13 ms per decode step. Never the step's own
     * stream, or a copy out would become a dependency of the next token. */
    if (shared_) {
        stream_ = shared_;
        own_stream_ = false;
    } else if (rad_stream_create(&stream_, /*high_priority=*/0) == RAD_OK) {
        own_stream_ = true;
    } else {
        stream_ = nullptr;
        release_device();
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* One (group, layer) launch. Staging rows are at stride entry_bytes_ from `stage_base` (the first
 * entry's slot), so each fragment lands inside its own contiguous staging entry. */
int TierExec::run_rows(const Resolved& r, bool scatter, const GroupInfo& gi, int64_t layer,
                       void* stage_base, const int32_t* d_idx, int64_t n_jobs) {
    RadTensor pool{};
    pool.dtype = RAD_U8;
    pool.rank = 2;
    pool.shape[0] = gi.n_blocks;  pool.shape[1] = gi.frag;
    pool.stride[0] = gi.frag;     pool.stride[1] = 1;
    pool.data = (char*)gi.base + layer * gi.layer_stride;

    RadTensor idx{};
    idx.dtype = RAD_I32; idx.rank = 1;
    idx.shape[0] = n_jobs; idx.stride[0] = 1;
    idx.data = (void*)d_idx;

    RadTensor pack{};
    pack.dtype = RAD_U8;
    pack.rank = 2;
    pack.shape[0] = n_jobs;      pack.shape[1] = gi.frag;
    pack.stride[0] = entry_bytes_;   /* one entry's stride, not one row's */
    pack.stride[1] = 1;
    pack.data = (char*)stage_base + gi.off + layer * gi.frag;

    RadTensor t3[3];
    if (scatter) { t3[0] = pack; t3[1] = idx; t3[2] = pool; }
    else         { t3[0] = pool; t3[1] = idx; t3[2] = pack; }

    /* Pass the resolved geometry, which carries the winning kernel's derived keys. Its M is the
     * band's upper bound; kernels take the row count from the operand extent. */
    RadArgs a{};
    a.t = t3; a.n_t = 3;
    a.p = r.geom.params(); a.n_p = r.geom.n_params();
    a.instance = r.instance;
    /* A host row runs now: what the stream still has queued (the index upload, the copy into the
     * staging, the wait on the step) must land first. What is enqueued after it is ordered behind
     * it, since it has finished by then. */
    if (gi.host) RAD_TRY(rad_stream_sync(stream_));
    return r.row->info->launch(&a, stream_);
}

/* ------------------------------------------------------------------ RADIANCE_DEBUG_TIER_VERIFY */

static uint64_t tier_fnv(const void* p, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* b = (const unsigned char*)p;
    for (int64_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

void TierExec::verify_note(const BlockHash& key, const void* p, int64_t n) {
    if (!verify_ || !leader_ || !p || n <= 0) return;
    out_sum_[key] = tier_fnv(p, n);
}

void TierExec::verify_check(const BlockHash& key, const void* p, int64_t n) {
    if (!verify_ || !leader_ || !p || n <= 0) return;
    auto it = out_sum_.find(key);
    if (it == out_sum_.end()) return;               /* never went out through this process */
    const uint64_t back = tier_fnv(p, n);
    if (back == it->second) return;
    RAD_ERR("kv tiers: the attention blocks for %s came back different (%016llx out, %016llx in); "
            "the tier moved the wrong bytes",
            key.str().c_str(), (unsigned long long)it->second, (unsigned long long)back);
}

/* ------------------------------------------------------------------ helpers */

/* Like RAD_TRY, but the first failure is logged with the call it came from. A failing tier is
 * otherwise invisible: entries stay put and the server only gets slower. */
#define TIER_TRY(call, what)                                                                  \
    do { const int _s = (call);                                                               \
         if (_s != RAD_OK) { warn_once(what, _s); return _s; } } while (0)

void TierExec::warn_once(const char* what, int st) {
    if (warned_) return;
    warned_ = true;
    RAD_WARN("kv tiers: %s failed with %s. The entry stays where it was; this is said once, and "
             "the running failure count is in /sessions.", what, rad_strerror(st));
}

bool TierExec::blocks_backed(const std::vector<int32_t>& blocks, const char* what) {
    if (!kv_) return false;
    for (size_t k = 0; k < gi_.size(); ++k) {
        if (k >= blocks.size()) continue;          /* an absent group is the gather's -1 row */
        const int32_t b = blocks[k];
        if (b < 0 || gi_[k].zero) continue;
        const int64_t live = kv_->live_blocks(gi_[k].g);
        if (b < live) continue;
        if (warned_unbacked_) return false;
        warned_unbacked_ = true;
        RAD_ERR("kv tier %s: block %d of group %d is at or past this rank's live mark of %lld, so "
                "it is lent to the expert slab and holds weights. The transfer is refused; the "
                "entry stays where it is.", what, (int)b, (int)gi_[k].g, (long long)live);
        return false;
    }
    return true;
}

int TierExec::copy_runs(const std::vector<TierJob*>& order, size_t first, int64_t n, bool out) {
    int64_t k = 0;
    while (k < n) {
        const int32_t s0 = order[first + (size_t)k]->host_slot;
        int64_t run = 1;
        while (k + run < n && order[first + (size_t)(k + run)]->host_slot == s0 + run) ++run;
        char* host = (char*)tiers_->arena().at(s0);
        if (!host) return RAD_E_STATE;
        host += slot_off_;
        char* dev = (char*)stage_ + k * entry_bytes_;
        const int st = out
            ? rad_memcpy_2d_async(host, slot_bytes_, dev, entry_bytes_, entry_bytes_, run, stream_)
            : rad_memcpy_2d_async(dev, entry_bytes_, host, slot_bytes_, entry_bytes_, run, stream_);
        if (st != RAD_OK) return st;
        k += run;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ copy on idle */

int TierExec::issue_copy(std::vector<TierJob>& jobs, RadStream compute) {
    if (!enabled_ || jobs.empty()) return RAD_OK;
    TIER_TRY(rad_dev_set(device_), "selecting the rank device");
    TIER_TRY(device_init(), "building the transfer staging");
    const int st = enqueue_copy(jobs, compute);
    /* Drain a partly enqueued copy before its host slots are freed by the report, or a queued
     * transfer would land on the slot's next owner. */
    if (st != RAD_OK) (void)rad_stream_sync(stream_);
    return st;
}

int TierExec::enqueue_copy(std::vector<TierJob>& jobs, RadStream compute) {
    order_.clear();
    for (TierJob& j : jobs) if (!j.blocks.empty()) order_.push_back(&j);
    const int64_t n = (int64_t)order_.size();
    if (n > copy_idx_) return RAD_E_INVAL;
    /* Check every id before any launch: a gather over an unbacked id faults the whole card. */
    for (TierJob* j : order_)
        if (!blocks_backed(j->blocks, "copy to host")) return RAD_E_STATE;

    /* Stream-ordered after everything given to compute, including the step that wrote these. */
    TIER_TRY(rad_event_record(ev_compute_, compute), "ordering the copy after the step");
    TIER_TRY(rad_event_wait(stream_, ev_compute_), "ordering the copy after the step");

    if (n > 0) {
        const int64_t ng = (int64_t)gi_.size();
        int32_t* h = idx_host_;                           /* the copy region */
        for (int64_t g = 0; g < ng; ++g)
            for (int64_t k = 0; k < n; ++k) {
                const std::vector<int32_t>& b = order_[(size_t)k]->blocks;
                h[g * n + k] = (size_t)g < b.size() ? b[(size_t)g] : -1;
            }
        TIER_TRY(rad_memcpy_async(idx_dev_, h, ng * n * (int64_t)sizeof(int32_t), stream_),
                 "the block-index upload");
        for (int64_t s = 0; s < n; s += max_batch_) {
            const int64_t c = std::min<int64_t>(max_batch_, n - s);
            for (size_t gidx = 0; gidx < gi_.size(); ++gidx)
                for (int64_t l = 0; l < gi_[gidx].n_layers && !gi_[gidx].zero; ++l)
                    TIER_TRY(run_rows(gi_[gidx].gather, /*scatter=*/false, gi_[gidx], l, stage_,
                                      idx_dev_ + (int64_t)gidx * n + s, c),
                             "the gather out of the pool");
            TIER_TRY(copy_runs(order_, (size_t)s, c, /*out=*/true), "the copy to host memory");
        }
        st_.bytes_out += n * entry_bytes_;
    }

    TIER_TRY(rad_event_record(ev_copy_, stream_), "marking the copy");
    copy_out_ = true;
    ++st_.passes;
    st_.jobs += (int64_t)jobs.size();
    if (verify_)
        for (const TierJob& j : jobs)
            verify_pending_.push_back({j.key, j.blocks.empty() ? -1 : j.host_slot});
    return RAD_OK;
}

int TierExec::copy_state() {
    if (!copy_out_) return 1;
    (void)rad_dev_set(device_);
    const int q = rad_event_query(ev_copy_);
    if (q == 0) return 0;
    copy_out_ = false;
    if (q > 0 && verify_) {
        for (const VerifyNote& v : verify_pending_) {
            if (v.host_slot >= 0)
                verify_note(v.key, (char*)tiers_->arena().at(v.host_slot) + slot_off_,
                            entry_bytes_);
        }
    }
    verify_pending_.clear();
    if (q < 0) warn_once("the copy to host memory", q);
    return q > 0 ? 1 : q;
}

int TierExec::wait_copy() {
    if (!copy_out_) return 1;
    (void)rad_dev_set(device_);
    const int st = rad_event_sync(ev_copy_);
    if (st != RAD_OK) warn_once("waiting for the copy to host memory", st);
    return copy_state();
}

/* ------------------------------------------------------------------ restore */

void TierExec::quiesce() {
    if (!enabled_ || !stream_) return;
    if (rad_dev_set(device_) < 0) return;
    (void)rad_stream_sync(stream_);
}

int TierExec::issue_restore(std::vector<TierJob>& jobs, RadStream compute,
                            std::vector<char>* ok) {
    ok->assign(jobs.size(), 0);
    if (!enabled_ || jobs.empty()) return RAD_OK;
    TIER_TRY(rad_dev_set(device_), "selecting the rank device");
    TIER_TRY(device_init(), "building the transfer staging");
    const size_t ng = gi_.size();

    /* Block ids are shared across ranks (one block table indexes every shard), so only the leader
     * chooses and reserves them; its allocator is the one the prefix cache (rank 0's) references.
     * Followers write into the ids they are handed.
     *
     * The whole chain is reserved before anything moves, with one loan recall. If the pool runs
     * short the chain is cut there: a hit needs a contiguous prefix, so entries past a gap are
     * useless. */
    if (leader_) {
        const int64_t n = (int64_t)jobs.size();
        std::vector<std::vector<int32_t>> ids(ng);
        int64_t m = n;
        for (size_t k = 0; k < ng; ++k) {
            ids[k].reserve((size_t)n);
            m = std::min(m, kv_->reserve_blocks(gi_[k].g, m, &ids[k]));
        }
        for (size_t k = 0; k < ng; ++k)
            for (size_t i = (size_t)m; i < ids[k].size(); ++i) kv_->release_block(gi_[k].g, ids[k][i]);
        for (size_t i = 0; i < jobs.size(); ++i) {
            jobs[i].blocks.clear();
            if ((int64_t)i >= m) continue;
            jobs[i].blocks.resize(ng);
            for (size_t k = 0; k < ng; ++k) jobs[i].blocks[k] = ids[k][i];
        }
    }
    size_t m = 0;
    while (m < jobs.size() && jobs[m].blocks.size() == ng) ++m;
    /* The leader chose ids against its own live mark and the recall should have moved every
     * follower's with it; verify that here. */
    for (size_t i = 0; i < m; ++i)
        if (!blocks_backed(jobs[i].blocks, "restore")) { m = i; break; }

    /* Followers find the chain by which jobs carry block ids, so every job past the cut releases
     * its ids and carries none; otherwise a follower would copy into blocks the cache does not own. */
    if (leader_) {
        for (size_t i = m; i < jobs.size(); ++i) {
            for (size_t k = 0; k < ng && k < jobs[i].blocks.size(); ++k)
                if (jobs[i].blocks[k] >= 0) kv_->release_block(gi_[k].g, jobs[i].blocks[k]);
            jobs[i].blocks.clear();
        }
    }
    if (m == 0) return RAD_OK;

    order_.clear();
    for (size_t i = 0; i < m; ++i) order_.push_back(&jobs[i]);
    for (size_t i = 0; i < m; ++i)
        verify_check(jobs[i].key, (char*)tiers_->arena().at(jobs[i].host_slot) + slot_off_,
                     entry_bytes_);

    /* From here a failure leaves blocks partly overwritten. Mark jobs started first, so report()
     * drops a failed started job instead of trusting its host copy. The compute stream waits on
     * whatever was enqueued, success or not, because a failed job's blocks return to the pool and
     * must not be reused before the queued transfers into them. */
    for (size_t i = 0; i < m; ++i) jobs[i].started = true;
    const int ri = ring_next_;
    ring_next_ = (ring_next_ + 1) % kRing;
    int32_t* h = idx_host_ + ng * copy_idx_ + (int64_t)ri * (int64_t)ng * chain_idx_;
    int32_t* d = idx_dev_  + ng * copy_idx_ + (int64_t)ri * (int64_t)ng * chain_idx_;
    int st = RAD_OK;
    size_t done = 0;
    for (size_t s = 0; s < m && st == RAD_OK; s += (size_t)chain_idx_) {
        const int64_t c = std::min<int64_t>(chain_idx_, (int64_t)(m - s));
        if (ring_used_[ri] && (st = rad_event_sync(ring_ev_[ri])) != RAD_OK) break;
        for (size_t g = 0; g < ng; ++g)
            for (int64_t k = 0; k < c; ++k) h[(int64_t)g * c + k] = jobs[s + (size_t)k].blocks[g];
        if ((st = rad_memcpy_async(d, h, (int64_t)ng * c * (int64_t)sizeof(int32_t), stream_)) != RAD_OK)
            break;
        if ((st = rad_event_record(ring_ev_[ri], stream_)) != RAD_OK) break;
        ring_used_[ri] = true;
        for (int64_t b = 0; b < c && st == RAD_OK; b += max_batch_) {
            const int64_t cc = std::min<int64_t>(max_batch_, c - b);
            if ((st = copy_runs(order_, s + (size_t)b, cc, /*out=*/false)) != RAD_OK) break;
            for (size_t gidx = 0; gidx < ng && st == RAD_OK; ++gidx)
                for (int64_t l = 0; l < gi_[gidx].n_layers && st == RAD_OK && !gi_[gidx].zero; ++l)
                    st = run_rows(gi_[gidx].scatter, /*scatter=*/true, gi_[gidx], l, stage_,
                                  d + (int64_t)gidx * c + b, cc);
        }
        if (st == RAD_OK) done = s + (size_t)c;
    }
    if (st != RAD_OK) warn_once("the restore into the pool", st);
    for (size_t i = 0; i < done; ++i) (*ok)[i] = 1;
    st_.bytes_in += (int64_t)done * entry_bytes_;

    /* The reading step waits on the device; if the event cannot be enqueued, fall back to a host
     * wait. */
    if (rad_event_record(ev_restore_, stream_) != RAD_OK ||
        rad_event_wait(compute, ev_restore_) != RAD_OK) {
        (void)rad_stream_sync(stream_);
    }
    ++st_.passes;
    st_.jobs += (int64_t)m;
    return st;
}

/* ------------------------------------------------------------------ the report */

void TierExec::report_pass(std::vector<TierJob>& jobs, const std::vector<char>& ok,
                           uint64_t now_ns) {
    if (!enabled_ || !leader_) return;
    bool quiet = false;
    for (size_t i = 0; i < jobs.size(); ++i) {
        TierJob& j = jobs[i];
        const bool good = i < ok.size() && ok[i];
        /* A failed promotion releases the leader's blocks once nothing queued against them can
         * still land; after report() nothing names them again. */
        if (j.kind == TierJob::Kind::ToDevice && !good) {
            if (!quiet) { quiesce(); quiet = true; }
            for (size_t k = 0; k < gi_.size() && k < j.blocks.size(); ++k)
                if (j.blocks[k] >= 0) kv_->release_block(gi_[k].g, j.blocks[k]);
            j.blocks.clear();
        }
        tiers_->report(j, good, now_ns);
    }
}

bool TierExec::plan_promote(const BlockHash& key, TierJob* out) {
    if (!enabled_ || !leader_) return false;
    return tiers_->plan_promote(key, out);
}

}  /* namespace rad */
