#include "mem/kvtier.h"

#include "rad_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace rad {

const char* kv_tier_name(KVTier t) {
    switch (t) {
        case KVTier::Device: return "device";
        case KVTier::Host:   return "host";
    }
    return "?";
}

/* ------------------------------------------------------------------ IdleTiers */

int IdleTiers::configure(const Config& c) {
    std::lock_guard lk(mu_);
    reset();
    cfg_ = c;

    if (cfg_.host_cap_bytes > 0 && cfg_.entry_bytes > 0) {
        const int st = arena_.open(cfg_.entry_bytes, cfg_.host_cap_bytes);
        if (st != RAD_OK) {
            RAD_WARN("kv tiers: no pinned host memory for %s, the host tier is off",
                     humanb(cfg_.host_cap_bytes).c_str());
            cfg_.host_cap_bytes = 0;
        }
    } else {
        cfg_.host_cap_bytes = 0;
    }

    char buf[512];
    if (!host_on()) {
        report_ = "kv tiers: off (no --prefix-cache-mib); a finished conversation's KV stays in "
                  "VRAM until the cache evicts it";
    } else {
        std::snprintf(buf, sizeof buf,
                      "kv tiers: a finished conversation's KV is copied to host memory at once "
                      "(%s, %lld entries, %lld tokens of context) and its VRAM is given up when "
                      "the expert loan or a request wants it; a full host tier drops its least "
                      "recently used entries, the end of a conversation before its start. One "
                      "entry is %s",
                      humanb(arena_.cap_bytes()).c_str(), (long long)arena_.capacity(),
                      (long long)(arena_.capacity() * cfg_.block_tokens),
                      humanb(cfg_.entry_bytes).c_str());
        report_ = buf;
        if (cfg_.ckpt_bytes > 0) {
            std::snprintf(buf, sizeof buf,
                          "\nkv tiers: a session's linear state is saved to host memory -- %lld "
                          "snapshot slot(s) of %s -- and a full store retires the least recently "
                          "used",
                          (long long)cfg_.ckpt_slots, humanb(cfg_.ckpt_bytes).c_str());
            report_ += buf;
        }
    }
    return RAD_OK;
}

void IdleTiers::close() {
    std::lock_guard lk(mu_);
    reset();
}

void IdleTiers::reset() {
    recs_.clear();
    pinned_.clear();
    sessions_.clear();
    roots_.clear();
    host_lru_.clear();
    wb_q_.clear();
    clean_q_.clear();
    room_.clear();
    wb_stuck_ = false;
    arena_.close();
    cfg_ = Config{};
    report_.clear();
}

std::string IdleTiers::report() const { return report_; }

void IdleTiers::unlist(Rec& r) {
    if (!r.listed) return;
    host_lru_.erase(r.it);
    r.listed = false;
}

/* Host records only: device records, clean or not, are ordered by the prefix cache's LRU (see the
 * one-record rule in the header). */
/* A use: stamps the record and moves a listed (host) one to the back of its list. */
void IdleTiers::touch(const BlockHash& key, Rec& r, uint64_t now_ns) {
    r.used_ns = now_ns;
    if (r.listed) {
        unlist(r);
        relist(key, r);
    }
}

void IdleTiers::relist(const BlockHash& key, Rec& r) {
    if (r.pending || r.tier == KVTier::Device || r.listed || pinned_.count(key)) return;
    r.it = host_lru_.emplace_hint(host_lru_.end(), LruKey{ r.used_ns, r.depth }, key);
    r.listed = true;
}

bool IdleTiers::wants_copy(const Rec& r) const {
    if (r.pending || r.doomed) return false;
    return host_on() && r.tier == KVTier::Device && r.host_slot < 0 && !r.blocks.empty();
}

void IdleTiers::queue_copy(const BlockHash& key, Rec& r) {
    if (r.queued || !wants_copy(r)) return;
    wb_q_.push_back(key);
    r.queued = true;
}

void IdleTiers::give_slot(int32_t slot) {
    if (slot < 0) return;
    arena_.free(slot);
    wb_stuck_ = false;
}

void IdleTiers::free_host(Rec& r) {
    give_slot(r.host_slot);
    r.host_slot = -1;
}

void IdleTiers::queue_clean(const BlockHash& key, Rec& r) {
    if (r.clean_q) return;
    clean_q_.push_back(key);
    r.clean_q = true;
}

/* ------------------------------------------------------------------ the linear half */

void IdleTiers::note_snapshot(const BlockHash& key, int32_t slot, int64_t pos) {
    if (slot < 0) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    r.ck_slot = slot;
    r.ck_pos = pos;
}

void IdleTiers::release_snapshot(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    r.ck_slot = -1;
    /* A record that only existed for the snapshot goes with it. note_snapshot creates one
     * mid-request; a request that never publishes (a rebased block table, a cancel before the
     * first whole block) would otherwise leave an unreachable empty record. */
    if (!r.pending && r.tier == KVTier::Device && r.blocks.empty() && r.host_slot < 0)
        recs_.erase(it);
}

/* ------------------------------------------------------------------ the cache's notices */

void IdleTiers::note_used(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns,
                          int64_t depth) {
    if (!host_on()) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    if (r.tier == KVTier::Device && !r.pending) r.blocks = blocks;
    /* Before touch, which files a host record under it. */
    r.depth = depth;
    /* For a device record this is a timestamp and a hash insert, cheap enough for the publish
     * path. */
    touch(key, r, now_ns);
    queue_copy(key, r);
}

void IdleTiers::note_hits(const BlockHash* keys, int64_t n, uint64_t now_ns) {
    if (!host_on() || !keys || n <= 0) return;
    std::lock_guard lk(mu_);
    for (int64_t i = 0; i < n; ++i) {
        auto it = recs_.find(keys[i]);
        if (it != recs_.end()) touch(keys[i], it->second, now_ns);
    }
}

void IdleTiers::rebound(const BlockHash& key, const std::vector<int32_t>& blocks, uint64_t now_ns,
                        int64_t depth) {
    if (!host_on()) return;
    std::lock_guard lk(mu_);
    Rec& r = recs_[key];
    if (r.pending) return;
    unlist(r);
    free_host(r);
    r.tier = KVTier::Device;
    r.blocks = blocks;
    r.used_ns = now_ns;
    r.depth = depth;
    queue_copy(key, r);
}

bool IdleTiers::doomed(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it == recs_.end() || it->second.doomed;
}

IdleTiers::Stats IdleTiers::stats() const {
    std::lock_guard lk(mu_);
    Stats s = stats_;
    /* Derived from the arena, clean copies included, so no separate counter can drift. */
    s.host_bytes = arena_.used_slots() * cfg_.entry_bytes;
    return s;
}

/* A chain whose first block has gone can never be hit (see roots_). Called wherever an entry goes
 * for good: forget(), and the sites here that drop an off-device entry and tell the cache. */
void IdleTiers::drop_rooted(const BlockHash& key) {
    auto ro = roots_.find(key);
    if (ro == roots_.end()) return;
    const std::vector<uint64_t> ids = ro->second;       /* drop_session rewrites the list */
    for (uint64_t id : ids) drop_session(id);
}

void IdleTiers::forget(const BlockHash& key) {
    std::lock_guard lk(mu_);
    drop_rooted(key);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    if (r.pending) {
        /* Mid-transfer: erasing now would be a use-after-free. report() cleans up. */
        r.doomed = true;
        return;
    }
    unlist(r);
    free_host(r);
    recs_.erase(it);
}

void IdleTiers::set_pinned(const BlockHash& key, bool on) {
    std::lock_guard lk(mu_);
    if (on) pinned_.insert(key);
    else pinned_.erase(key);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    if (on) unlist(it->second);
    else relist(key, it->second);
}

/* ------------------------------------------------------------------ archives on disk */

int64_t IdleTiers::host_copies(const BlockHash* keys, int64_t n) const {
    std::lock_guard lk(mu_);
    int64_t have = 0;
    for (int64_t i = 0; i < n; ++i) {
        auto it = recs_.find(keys[i]);
        if (it != recs_.end() && !it->second.pending && !it->second.doomed &&
            it->second.host_slot >= 0)
            ++have;
    }
    return have;
}

bool IdleTiers::read_host(const BlockHash* keys, int64_t n, uint8_t* dst) const {
    std::lock_guard lk(mu_);
    for (int64_t i = 0; i < n; ++i) {
        auto it = recs_.find(keys[i]);
        if (it == recs_.end() || it->second.pending || it->second.doomed ||
            it->second.host_slot < 0)
            return false;
        std::memcpy(dst + i * cfg_.entry_bytes, arena_.at(it->second.host_slot),
                    (size_t)cfg_.entry_bytes);
    }
    return true;
}

bool IdleTiers::adopt_host(const BlockHash& key, const void* src, int64_t depth, uint64_t now_ns) {
    if (!host_on()) return false;
    std::lock_guard lk(mu_);
    if (recs_.count(key)) return false;
    const int32_t slot = arena_.alloc();
    if (slot < 0) return false;
    std::memcpy(arena_.at(slot), src, (size_t)cfg_.entry_bytes);
    Rec& r = recs_[key];
    r.tier = KVTier::Host;
    r.host_slot = slot;
    r.used_ns = now_ns;
    r.depth = depth;
    relist(key, r);
    return true;
}

int64_t IdleTiers::free_host_slots() const {
    std::lock_guard lk(mu_);
    return host_on() ? arena_.free_slots() : 0;
}

KVTier IdleTiers::tier_of(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    return it == recs_.end() ? KVTier::Device : it->second.tier;
}

bool IdleTiers::resident_off_device(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    return !it->second.pending && it->second.tier != KVTier::Device;
}

/* ------------------------------------------------------------------ copy on idle */

bool IdleTiers::writeback_waiting() const {
    std::lock_guard lk(mu_);
    /* Not while nothing can move: an arena full of copies of entries still in VRAM offers nothing
     * to drop, so wait until a slot is freed or a VRAM copy is given up. */
    if (wb_stuck_ && arena_.full()) return false;
    return !wb_q_.empty() && host_on();
}

bool IdleTiers::starved() const {
    std::lock_guard lk(mu_);
    return !room_.empty();
}

void IdleTiers::plan_writeback(int64_t max_entries, std::vector<TierJob>* out) {
    if (!host_on() || max_entries <= 0) return;
    std::lock_guard lk(mu_);
    const size_t first = out->size();
    while (!wb_q_.empty() && (int64_t)(out->size() - first) < max_entries) {
        const BlockHash key = wb_q_.front();
        wb_q_.pop_front();
        auto it = recs_.find(key);
        if (it == recs_.end()) continue;
        Rec& r = it->second;
        r.queued = false;
        if (!wants_copy(r)) continue;
        TierJob j;
        j.kind = TierJob::Kind::WriteBack;
        j.key = key;
        j.bytes = cfg_.entry_bytes;
        j.blocks = r.blocks;
        out->push_back(std::move(j));
    }

    /* Allocate host slots for the whole batch as runs, so each run crosses the link as one copy.
     * Jobs past where the arena ran out are refused; see below. */
    const int64_t want = (int64_t)(out->size() - first);
    std::vector<int32_t> slots;
    slots.reserve((size_t)want);
    while ((int64_t)slots.size() < want) {
        int32_t got = 0;
        const int32_t s = arena_.alloc_run((int32_t)(want - (int64_t)slots.size()), &got);
        if (s < 0) break;
        for (int32_t k = 0; k < got; ++k) slots.push_back(s + k);
    }
    size_t next = 0;
    std::vector<BlockHash> refused;
    for (size_t i = first; i < out->size();) {
        TierJob& j = (*out)[i];
        if (next < slots.size()) {
            j.host_slot = slots[next++];
            ++i;
            continue;
        }
        refused.push_back(j.key);
        out->erase(out->begin() + (long)i);
    }
    if (want > 0 && slots.empty()) wb_stuck_ = true;
    for (size_t i = first; i < out->size(); ++i) {
        Rec& r = recs_[(*out)[i].key];
        unlist(r);
        r.pending = true;
    }
    if (refused.empty()) return;

    /* A refused copy is worth making room for only by dropping a host entry worth less than it.
     * The most valuable refused copies are matched with the least valuable host entries; one
     * that finds none leaves the queue, since any room made for it would cost more than it
     * saves: it would drop a conversation's start to keep its end, or a newer conversation to
     * keep an older one. Its entry stays in VRAM only and is copied again when a turn uses it. */
    const LruOrder less;
    std::vector<std::pair<LruKey, BlockHash>> by_value;
    by_value.reserve(refused.size());
    for (const BlockHash& k : refused) {
        const Rec& r = recs_[k];
        by_value.push_back({ LruKey{ r.used_ns, r.depth }, k });
    }
    std::sort(by_value.begin(), by_value.end(),
              [&less](const auto& a, const auto& b) { return less(b.first, a.first); });
    std::unordered_set<BlockHash, BlockHashHash> keep;
    int64_t held = 0;
    auto victim = host_lru_.begin();
    for (const auto& kv : by_value) {
        if (victim != host_lru_.end() && less(victim->first, kv.first)) {
            room_.push_back(kv.first);
            keep.insert(kv.second);
            ++victim;
        } else {
            ++held;
        }
    }
    /* Requeue at the front, in order, so the oldest is still copied first. */
    for (auto it = refused.rbegin(); it != refused.rend(); ++it) {
        if (!keep.count(*it)) continue;
        auto ri = recs_.find(*it);
        if (ri == recs_.end() || ri->second.queued) continue;
        wb_q_.push_front(*it);
        ri->second.queued = true;
    }
    if (held == 0) return;
    stats_.held_back += held;
    const uint64_t now = rad_mono_ns();
    if (held_back_warned_ns_ == 0 || now - held_back_warned_ns_ >= 60ull * 1000000000ull) {
        held_back_warned_ns_ = now;
        RAD_WARN("kv tiers: %lld entries (%lld tokens of context) got no host copy: the host "
                 "tier (--prefix-cache-mib, %s, %lld tokens) is full of context used more "
                 "recently or nearer the start of its conversation. They stay in VRAM only and "
                 "are prefilled again once VRAM gives them up; a larger --prefix-cache-mib holds "
                 "more. Counted in /sessions moved.held_back",
                 (long long)held, (long long)(held * cfg_.block_tokens),
                 humanb(arena_.cap_bytes()).c_str(),
                 (long long)(arena_.capacity() * cfg_.block_tokens));
    }
}

void IdleTiers::take_clean(std::vector<BlockHash>* out) {
    std::lock_guard lk(mu_);
    for (const BlockHash& k : clean_q_) {
        auto it = recs_.find(k);
        if (it == recs_.end()) continue;
        it->second.clean_q = false;
        out->push_back(k);
    }
    clean_q_.clear();
}

bool IdleTiers::droppable(const BlockHash& key) const {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    const Rec& r = it->second;
    return !r.pending && !r.doomed && r.tier == KVTier::Device && r.host_slot >= 0;
}

void IdleTiers::drop_commit(const BlockHash& key) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return;
    Rec& r = it->second;
    if (r.tier != KVTier::Device) return;
    r.tier = KVTier::Host;
    ++stats_.dropped;
    wb_stuck_ = false;       /* a host record is what a capacity drop takes */
    relist(key, r);
}

/* ------------------------------------------------------------------ capacity and expiry */

void IdleTiers::plan(int budget, std::vector<TierJob>* out) {
    if (!host_on() || budget <= 0) return;
    std::lock_guard lk(mu_);
    /* Drop one entry for each copy refused, least valuable first, and only one worth less than
     * the copy it makes room for (room_ runs most valuable first, the list least valuable first,
     * so the first pair that fails ends the plan). Looping until the arena has room would empty
     * the tier, since a queued drop frees its slot only when the engine reports it. */
    const int64_t want = std::min<int64_t>((int64_t)room_.size(), budget);
    const LruOrder less;
    for (int64_t k = 0; k < want && !host_lru_.empty();) {
        if (!less(host_lru_.begin()->first, room_[(size_t)k])) break;
        auto ri = recs_.find(host_lru_.begin()->second);
        if (ri == recs_.end()) { host_lru_.erase(host_lru_.begin()); continue; }
        Rec& r = ri->second;
        TierJob j;
        j.kind = TierJob::Kind::DropHost;
        j.key = ri->first;
        j.host_slot = r.host_slot;
        j.bytes = cfg_.entry_bytes;
        unlist(r);
        r.pending = true;
        out->push_back(std::move(j));
        ++k;
    }
    room_.clear();
}

void IdleTiers::expired(uint64_t cutoff_ns, size_t max, std::vector<BlockHash>* out) const {
    std::lock_guard lk(mu_);
    /* The list holds no pending record (relist), so every key here can go now. */
    for (auto it = host_lru_.begin();
         it != host_lru_.end() && it->first.used_ns < cutoff_ns && out->size() < max; ++it)
        out->push_back(it->second);
}

bool IdleTiers::plan_promote(const BlockHash& key, TierJob* out) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(key);
    if (it == recs_.end()) return false;
    Rec& r = it->second;
    if (r.pending || r.doomed || r.tier != KVTier::Host) return false;

    *out = TierJob{};
    out->kind = TierJob::Kind::ToDevice;
    out->key = key;
    out->host_slot = r.host_slot;
    out->bytes = cfg_.entry_bytes;
    ++stats_.host_hits;
    unlist(r);
    r.pending = true;
    return true;
}

void IdleTiers::report(const TierJob& job, bool ok, uint64_t now_ns) {
    std::lock_guard lk(mu_);
    auto it = recs_.find(job.key);
    if (it == recs_.end()) {
        /* Only the job names a write-back's slot. */
        if (job.kind == TierJob::Kind::WriteBack) give_slot(job.host_slot);
        return;
    }
    Rec& r = it->second;
    r.pending = false;
    /* An unstarted promotion is not a failure: nothing was tried. */
    if (!ok && (job.kind != TierJob::Kind::ToDevice || job.started)) ++stats_.failures;

    /* A record forgotten mid-transfer is finished off here, before the per-kind handling. */
    if (r.doomed) {
        /* The job's slot may or may not be the record's: a write-back's slot belongs to the job
         * until it reports. Free each distinct slot once. */
        if (job.host_slot >= 0 && job.host_slot != r.host_slot) give_slot(job.host_slot);
        free_host(r);
        recs_.erase(it);
        return;
    }

    switch (job.kind) {
        case TierJob::Kind::WriteBack: {
            if (job.host_slot >= 0) {
                if (ok && r.tier == KVTier::Device && r.host_slot < 0) {
                    r.host_slot = job.host_slot;
                    ++stats_.to_host;
                } else {
                    give_slot(job.host_slot);
                }
            }
            relist(job.key, r);
            /* Clean now: the cache is asked whether the VRAM copy can go. */
            if (r.tier == KVTier::Device && r.host_slot >= 0) queue_clean(job.key, r);
            /* A copy that failed is tried again. */
            queue_copy(job.key, r);
            return;
        }

        case TierJob::Kind::ToDevice:
            if (ok) {
                r.tier = KVTier::Device;
                r.blocks = job.blocks;
                /* The host copy is kept, so the entry is clean on landing. */
                r.host_slot = job.host_slot;
                /* A request is about to hit it. */
                if (now_ns) r.used_ns = now_ns;
                ++stats_.promoted;
                queue_clean(job.key, r);
                return;
            }
            if (!job.started) {
                /* Never started: the reservation ran out first. It stays on the host tier. */
                relist(job.key, r);
                return;
            }
            /* A started promotion that failed may have scattered some layers and not others, so
             * the entry is dropped to force a miss, and the cache is told so it stops naming the
             * blocks. */
            if (job.host_slot >= 0 && job.host_slot != r.host_slot) give_slot(job.host_slot);
            free_host(r);
            recs_.erase(it);
            drop_rooted(job.key);
            if (on_drop_) on_drop_(job.key);
            return;

        case TierJob::Kind::DropHost:
            free_host(r);
            ++stats_.host_drops;
            recs_.erase(it);
            drop_rooted(job.key);
            if (on_drop_) on_drop_(job.key);
            return;
    }
}


/* ------------------------------------------------------------------ the session view */

/* Two chains are the same conversation when one is a prefix of the other: blocks are content
 * addressed, so turn N's chain is turn N-1's plus more. */
static bool chain_extends(const std::vector<BlockHash>& shorter,
                          const std::vector<BlockHash>& longer) {
    if (shorter.size() > longer.size()) return false;
    for (size_t i = 0; i < shorter.size(); ++i) if (shorter[i] != longer[i]) return false;
    return true;
}

void IdleTiers::note_session(uint64_t id, const std::vector<BlockHash>& chain, int64_t n_tokens,
                             uint64_t now_ns, const std::string& label) {
    if (chain.empty()) return;
    std::lock_guard lk(mu_);

    /* A session is a leaf of the prefix trie. Each turn arrives under a new request id, so the
     * candidates are the sessions sharing this root (a handful at most):
     *
     *   - a chain this one extends is no longer a leaf and is absorbed;
     *   - a chain that extends this one means this is not a leaf either (a preempted request
     *     republishing shorter). Several conversations may extend it, so the most recent one is
     *     touched and no row is added. */
    uint64_t first_seen = 0;
    /* A label persists across later turns that send none. */
    std::string kept_label = label;
    bool absorbed = false;
    std::vector<uint64_t>& at_root = roots_[chain[0]];
    for (size_t i = 0; i < at_root.size();) {
        auto it = sessions_.find(at_root[i]);
        if (it == sessions_.end()) { at_root.erase(at_root.begin() + (long)i); continue; }
        if (!chain_extends(it->second.chain, chain)) { ++i; continue; }
        if (first_seen == 0 || it->second.first_seen_ns < first_seen)
            first_seen = it->second.first_seen_ns;
        if (kept_label.empty()) kept_label = it->second.label;
        sessions_.erase(it);
        at_root.erase(at_root.begin() + (long)i);
        absorbed = true;
    }

    if (!absorbed) {
        Session* newest = nullptr;
        for (uint64_t sid : at_root) {
            auto it = sessions_.find(sid);
            if (it == sessions_.end() || !chain_extends(chain, it->second.chain)) continue;
            if (!newest || it->second.last_used_ns > newest->last_used_ns) newest = &it->second;
        }
        if (newest) {
            newest->last_used_ns = now_ns;
            if (!label.empty()) newest->label = label;
            return;
        }
    }

    Session& s = sessions_[id];
    s.last_used_ns = now_ns;
    s.n_tokens = n_tokens;
    s.first_seen_ns = first_seen ? first_seen : now_ns;
    s.chain = chain;
    s.label = std::move(kept_label);
    at_root.push_back(id);
}

void IdleTiers::drop_session(uint64_t id) {
    auto it = sessions_.find(id);
    if (it == sessions_.end()) return;
    if (!it->second.chain.empty()) {
        auto ri = roots_.find(it->second.chain[0]);
        if (ri != roots_.end()) {
            std::vector<uint64_t>& v = ri->second;
            v.erase(std::remove(v.begin(), v.end(), id), v.end());
            if (v.empty()) roots_.erase(ri);
        }
    }
    sessions_.erase(it);
}

std::vector<IdleTiers::SessionInfo> IdleTiers::sessions(Occupancy* occ) const {
    /* One slice's hold on the lock (YieldingMutex). */
    struct Slice {
        YieldingMutex& m;
        explicit Slice(YieldingMutex& mu) : m(mu) { m.lock_after_others(); }
        ~Slice() { m.unlock(); }
    };
    /* Records per slice: a few microseconds of hash lookups, the longest an engine thread waits. */
    constexpr size_t kSliceRecords = 64;

    std::vector<SessionInfo> out;
    {
        Slice lk(mu_);
        out.reserve(sessions_.size());
        for (const auto& kv : sessions_) {
            SessionInfo si;
            si.id = kv.first;
            si.model = model_;
            si.n_tokens = kv.second.n_tokens;
            si.first_seen_ns = kv.second.first_seen_ns;
            si.label = kv.second.label;
            si.last_used_ns = kv.second.last_used_ns;
            si.bytes = (int64_t)kv.second.chain.size() * cfg_.entry_bytes;
            out.push_back(std::move(si));
        }
    }
    /* Distinct snapshots reached by the walk, for Occupancy. The store itself is the engine's and
     * is not read from this thread. One per checkpoint interval, so the set stays small. */
    std::unordered_set<BlockHash, BlockHashHash> host_snaps;
    size_t kept = 0;
    for (size_t k = 0; k < out.size(); ++k) {
        SessionInfo& si = out[k];
        /* The histogram is computed, never stored, since blocks move a few at a time. The session
         * is looked up again each slice: in between it may have been absorbed or dropped, and a
         * row that is gone is left out. */
        bool gone = false;
        for (size_t at = 0;;) {
            Slice lk(mu_);
            const auto it = sessions_.find(si.id);
            if (it == sessions_.end()) { gone = true; break; }
            const std::vector<BlockHash>& chain = it->second.chain;
            const size_t end = std::min(chain.size(), at + kSliceRecords);
            for (; at < end; ++at) {
                const BlockHash& h = chain[at];
                auto ri = recs_.find(h);
                if (ri == recs_.end()) { ++si.on_device; continue; }
                const Rec& rr = ri->second;
                if (rr.pending) si.moving = true;
                switch (rr.tier) {
                    case KVTier::Device: ++si.on_device; break;
                    case KVTier::Host:   ++si.on_host;   break;
                }
                /* Snapshots are in host memory whatever tier the blocks are on. */
                if (rr.ck_slot >= 0) {
                    ++si.snapshots;
                    if (occ) host_snaps.insert(h);
                    if (rr.ck_pos > si.resume_tokens) si.resume_tokens = rr.ck_pos;
                }
            }
            if (at >= chain.size()) break;
        }
        if (gone) continue;
        si.snap_bytes = si.snapshots * cfg_.ckpt_bytes;
        if (kept != k) out[kept] = std::move(si);
        ++kept;
    }
    out.resize(kept);
    std::sort(out.begin(), out.end(),
              [](const SessionInfo& a, const SessionInfo& b) {
                  return a.last_used_ns > b.last_used_ns;
              });
    if (occ) {
        Slice lk(mu_);
        occ->host_bytes = arena_.used_slots() * cfg_.entry_bytes +
                          (int64_t)host_snaps.size() * cfg_.ckpt_bytes;
        /* Block arena plus snapshot store, the same scope as `host_bytes`. */
        occ->host_cap_bytes = (host_on() ? cfg_.host_cap_bytes : 0) +
                              cfg_.ckpt_slots * cfg_.ckpt_bytes;
    }
    return out;
}
}  /* namespace rad */
