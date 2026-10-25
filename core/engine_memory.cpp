/* engine_memory.cpp -- archives on disk (--memory-dir): the engine half of core/mem/kvarchive.h.
 *
 * An archive (Scheduler::archive) pins its conversation's blocks and deepest snapshot in the
 * prefix cache. With --memory-dir it is also written to a file, so a restart does not lose it, and
 * every file there is read back at startup, before the server takes a request:
 *
 *   write   once a second the step loop looks for an archive not yet on disk whose pinned blocks
 *           all have host copies (the tiers copy each block to host memory as soon as its request
 *           finishes). The snapshot is copied out under the scheduler's lock, after the streams
 *           have drained so no save is still landing in its slot; the blocks are copied by the
 *           writer thread in batches under the tiers' own lock, and the file is written outside
 *           every lock.
 *   read    each file whose fingerprint is this deployment's and whose checksum holds is adopted:
 *           its blocks become off-device entries with host records (exactly what a demoted entry
 *           is), its snapshot takes a store slot, and the archive is pinned again. A request that
 *           hits the chain restores it as it restores any demoted conversation.
 *
 * Nothing here is on the step path: a write in progress costs the step loop nothing, and reading
 * back happens before the step loop starts. */
#include "engine_priv.h"
#include "format/radfile.h"
#include "mem/kvarchive.h"
#include "sched/scheduler.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>

namespace rad {

namespace fs = std::filesystem;

namespace {

uint64_t unix_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}

/* Sessions a read-back snapshot is filed under in the checkpoint index: outside any request id. */
constexpr uint64_t kRestoredSession = 1ull << 63;

}  /* namespace */

/* One write at a time: archives are made at compaction, minutes apart, and one at a time keeps
 * the disk's bandwidth for the one the agent is most likely to ask about first. */
struct Engine::MemWriter {
    struct Job {
        std::string            path;
        KvArchiveHeader        header;
        std::vector<int32_t>   tokens;
        std::vector<BlockHash> keys;
        std::vector<uint8_t>   snapshot;
    };
    std::mutex              mu;
    std::condition_variable cv;                /* a job, or stop */
    std::condition_variable done_cv;           /* a result */
    bool                    stop = false;
    bool                    has_job = false;   /* `job` waits for the thread */
    bool                    busy = false;      /* a job is out, until its result is taken */
    bool                    done = false;      /* `done_id` and `ok` hold a result */
    Job                     job;
    std::string             done_id;
    bool                    ok = false;
    std::thread             thread;
};

std::string Engine::memory_fingerprint() const {
    const Rank& lead = *ranks_[0];
    std::string fp = "radiance " RAD_VERSION;
    char buf[64];
    if (file_ && file_->is_open()) {
        /* The container's tables name every tensor, its encoding and its offset; samples of the
         * blob tell two containers with one layout apart (a finetune converted by the same
         * recipe). */
        const RadFileHeader& h = file_->header();
        uint64_t x = kva_fnv(&h, sizeof h);
        if (file_->meta_count() > 0)
            x = kva_fnv(&file_->meta_at(0), (size_t)file_->meta_count() * sizeof(RadFileKV), x);
        if (file_->entry_count() > 0)
            x = kva_fnv(&file_->entry(0), (size_t)file_->entry_count() * sizeof(RadFileEntry), x);
        if (h.str_bytes > 0 && h.str_off + h.str_bytes <= (uint64_t)file_->size())
            x = kva_fnv(file_->base() + h.str_off, (size_t)h.str_bytes, x);
        if (h.data_bytes > 0 && h.data_off + h.data_bytes <= (uint64_t)file_->size()) {
            const uint64_t take = std::min<uint64_t>(4096, h.data_bytes);
            for (uint64_t k = 0; k < 16; ++k) {
                const uint64_t at = h.data_off + (h.data_bytes - take) * k / 15;
                x = kva_fnv(file_->base() + at, (size_t)take, x);
            }
        }
        std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)x);
        fp += "; model " + std::string(buf) + " " + std::to_string((long long)file_->size());
    } else {
        fp += "; model " + cfg_.model;
    }
    fp += "; arch " + plugins_.arch_plugin + "@" + plugins_.arch_version;
    fp += "; kernels " + plugins_.libraries();
    fp += "; tp " + std::to_string(cfg_.tp);
    fp += "; kv " + cfg_.kv_cache_dtype;
    fp += "; block " + std::to_string((long long)lead.prefix.block_size());
    fp += "; entry " + std::to_string((long long)lead.tiers.entry_bytes());
    fp += "; snapshot " + std::to_string((long long)(ck_store_.enabled() ? ck_store_.slot_bytes() : 0));
    return fp;
}

int Engine::restore_archives() {
    if (cfg_.memory_dir.empty()) return RAD_OK;
    Rank& lead = *ranks_[0];
    if (!cfg_.prefix_cache || !lead.prefix.enabled()) {
        RAD_WARN("--memory-dir %s: this server keeps no prefix cache, so it has no archives to "
                 "write or read back", cfg_.memory_dir.c_str());
        return RAD_OK;
    }
    if (!lead.tiers.host_on()) {
        RAD_WARN("--memory-dir %s needs the host tier, which is off (%s): archives are written "
                 "from host memory and read back into it. They are kept in memory only",
                 cfg_.memory_dir.c_str(),
                 cfg_.prefix_cache_mib > 0 ? "see the kv tiers line above" : "no --prefix-cache-mib");
        return RAD_OK;
    }
    std::error_code ec;
    fs::create_directories(cfg_.memory_dir, ec);
    if (ec || !fs::is_directory(cfg_.memory_dir, ec)) {
        RAD_WARN("--memory-dir %s: not a directory this process can create (%s); archives are kept "
                 "in memory only", cfg_.memory_dir.c_str(), ec.message().c_str());
        return RAD_OK;
    }
    mem_on_ = true;
    mem_fp_ = memory_fingerprint();

    /* Newest first, so a smaller --memory-archives keeps the most recent. A `.tmp` is a write a
     * crash interrupted. */
    std::vector<std::pair<fs::file_time_type, fs::path>> files;
    for (const auto& de : fs::directory_iterator(cfg_.memory_dir, ec)) {
        if (!de.is_regular_file(ec)) continue;
        const fs::path& p = de.path();
        const std::string name = p.filename().string();
        if (name.size() > 8 && name.compare(name.size() - 8, 8, ".kva.tmp") == 0) {
            fs::remove(p, ec);
            continue;
        }
        if (p.extension() == ".kva") files.push_back({ de.last_write_time(ec), p });
    }
    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    const int64_t entry_bytes = lead.tiers.entry_bytes();
    const int64_t bs = lead.prefix.block_size();
    int restored = 0;
    uint64_t session = kRestoredSession;
    std::vector<uint8_t> buf((size_t)std::max<int64_t>(entry_bytes, 1));
    for (const auto& f : files) {
        const std::string path = f.second.string();
        if (restored >= cfg_.memory_archives) {
            RAD_INFO("memory: %s not read back: --memory-archives %lld are already held",
                     path.c_str(), (long long)cfg_.memory_archives);
            continue;
        }
        KvArchiveReader rd;
        if (rd.open(path) != RAD_OK) {
            RAD_WARN("memory: %s is not an archive this build reads; left in place", path.c_str());
            continue;
        }
        const KvArchiveHeader& h = rd.header();
        if (h.fingerprint != mem_fp_) {
            RAD_WARN("memory: archive '%s' (%s) was written by another deployment and is left in "
                     "place.\n  it was:  %s\n  this is: %s", h.id.c_str(), path.c_str(),
                     h.fingerprint.c_str(), mem_fp_.c_str());
            continue;
        }
        if (h.entry_bytes != entry_bytes || f.second.stem().string() != h.id) {
            RAD_WARN("memory: %s does not match its own header; left in place", path.c_str());
            continue;
        }
        if (rd.verify() != RAD_OK) {
            const std::string bad = path + ".bad";
            fs::rename(f.second, bad, ec);
            RAD_WARN("memory: archive '%s' failed its checksum (a torn or damaged file); set aside "
                     "as %s", h.id.c_str(), bad.c_str());
            continue;
        }
        /* The blocks must hash here as they did there: the same tokens, control vector and salt. */
        Request probe;
        probe.prompt = rd.tokens();
        probe.sp.use_control_vector = h.use_control_vector;
        std::vector<MMContentKey> mk;
        media_keys(probe, &mk, cvec_salt_for(h.use_control_vector));
        if (h.n_entries * bs > h.n_tokens ||
            lead.prefix.chain_hashes(rd.tokens(), mk, h.n_entries) != rd.keys()) {
            RAD_WARN("memory: archive '%s' hashes differently on this server (another control "
                     "vector?); left in place", h.id.c_str());
            continue;
        }
        if (lead.tiers.free_host_slots() < h.n_entries) {
            RAD_WARN("memory: archive '%s' needs %lld host-tier entries and %lld are free; not read "
                     "back (a larger --prefix-cache-mib holds it)", h.id.c_str(),
                     (long long)h.n_entries, (long long)lead.tiers.free_host_slots());
            continue;
        }

        int64_t adopted = 0;
        bool snap_ok = h.snapshot_bytes == 0;
        {
            const auto lk = sched_.hold();
            const uint64_t now = rad_mono_ns();
            for (; adopted < h.n_entries; ++adopted) {
                if (rd.next_entry(buf.data()) != RAD_OK) break;
                const BlockHash& key = rd.keys()[(size_t)adopted];
                const int32_t* toks = rd.tokens().data() + adopted * bs;
                if (!lead.prefix.adopt_entry(key, toks, [&] {
                        return lead.tiers.adopt_host(key, buf.data(), adopted, now);
                    }))
                    break;
            }
            int32_t slot = -1;
            if (h.snapshot_bytes > 0 && adopted == h.n_entries &&
                h.snapshot_bytes == ck_store_.slot_bytes() &&
                lead.prefix.reserve_checkpoint(session, h.snapshot_pos, h.snapshot_key, &slot) == RAD_OK) {
                snap_ok = rd.snapshot(ck_store_.at(slot)) == RAD_OK;
                /* Committed only once read whole: a slot whose read failed holds a partial state,
                 * which stays uncommitted (never offered) until a later reservation retires it. */
                if (snap_ok) lead.prefix.commit_checkpoint(h.snapshot_key);
            }
        }
        ++session;
        MemChain chain;
        chain.tokens = rd.tokens();
        chain.use_control_vector = h.use_control_vector;
        chain.updated_ns = rad_mono_ns();
        const uint64_t now_ms = unix_ms();
        const double age = now_ms > h.made_unix_ms ? (double)(now_ms - h.made_unix_ms) * 1e-3 : 0.0;
        MemArchiveStat st;
        const int rc = sched_.restore_archive(h.id, h.label, std::move(chain), age, &st);
        if (rc < 0) {
            RAD_WARN("memory: archive '%s' was read but not listed (%s)", h.id.c_str(),
                     rad_strerror(rc));
            continue;
        }
        ++restored;
        RAD_INFO("memory: archive '%s'%s%s read back from %s: %lld tokens, %lld of them cached, "
                 "snapshot at %lld%s", h.id.c_str(), h.label.empty() ? "" : " of session ",
                 h.label.c_str(), path.c_str(), (long long)st.tokens, (long long)st.pinned_tokens,
                 (long long)st.snapshot_pos,
                 adopted < h.n_entries ? " (the host tier took only part of it)"
                 : !snap_ok ? " (its snapshot was not read back; a fork prefills the chain)" : "");
    }
    if (restored > 0 || !files.empty())
        RAD_INFO("memory: %d of %zu archive(s) in %s read back", restored, files.size(),
                 cfg_.memory_dir.c_str());
    return RAD_OK;
}

void Engine::memory_tick(uint64_t now_ns) {
    if (!mem_on_ || now_ns - mem_last_ns_ < 1000000000ull) return;
    mem_last_ns_ = now_ns;
    (void)memory_step(now_ns);
}

int Engine::memory_step(uint64_t now_ns) {
    if (!mem_writer_) {
        mem_writer_ = std::make_shared<MemWriter>();
        MemWriter* w = mem_writer_.get();
        IdleTiers* tiers = &ranks_[0]->tiers;
        w->thread = std::thread([w, tiers] {
            std::unique_lock<std::mutex> lk(w->mu);
            for (;;) {
                w->cv.wait(lk, [w] { return w->stop || w->has_job; });
                /* A job handed before the stop is written first. */
                if (!w->has_job) return;
                MemWriter::Job j = std::move(w->job);
                w->has_job = false;
                lk.unlock();
                const int st = kva_write(j.path, j.header, j.tokens, j.keys,
                                         [&](int64_t first, int64_t n, uint8_t* dst) {
                                             return tiers->read_host(j.keys.data() + first, n, dst);
                                         },
                                         j.snapshot.empty() ? nullptr : j.snapshot.data());
                lk.lock();
                w->done_id = j.header.id;
                w->ok = st == RAD_OK;
                w->done = true;
                w->done_cv.notify_all();
            }
        });
    }
    MemWriter& w = *mem_writer_;
    {
        std::string id;
        bool ok = false, finished = false;
        {
            std::lock_guard<std::mutex> lk(w.mu);
            if (w.done) {
                id = w.done_id;
                ok = w.ok;
                finished = true;
                w.done = false;
                w.busy = false;
            }
            if (w.busy) return 1;
        }
        if (finished) sched_.archive_written(id, ok, now_ns);
    }

    MemArchive a;
    if (!sched_.archive_to_write(now_ns, &a)) return 0;
    Rank& lead = *ranks_[0];
    /* One not ready yet steps aside, so it does not hold back the ones behind it. */
    const uint64_t later = now_ns + 5ull * 1000000000ull;
    const std::vector<BlockHash>& keys = a.pinned.entries;
    const int64_t have = lead.tiers.host_copies(keys.data(), (int64_t)keys.size());
    if (have < (int64_t)keys.size()) {
        /* The copies land within moments of the request ending; one still missing after a minute
         * found the host tier full. */
        const double waited = (double)(now_ns - std::min(now_ns, a.made_ns)) * 1e-9;
        if (waited > 60 && std::find(mem_waiting_warned_.begin(), mem_waiting_warned_.end(), a.id) ==
                               mem_waiting_warned_.end()) {
            mem_waiting_warned_.push_back(a.id);
            RAD_WARN("memory: archive '%s' is not written yet: %lld of its %zu blocks have no host "
                     "copy after %.0f s (a full host tier holds them back; see --prefix-cache-mib)",
                     a.id.c_str(), (long long)((int64_t)keys.size() - have), keys.size(), waited);
        }
        sched_.archive_defer(a.id, later);
        return 3;
    }

    MemWriter::Job j;
    j.path = archive_path(cfg_.memory_dir, a.id);
    j.tokens = a.chain.tokens;
    j.keys = keys;
    KvArchiveHeader& h = j.header;
    h.id = a.id;
    h.label = a.label;
    h.fingerprint = mem_fp_;
    h.made_unix_ms = unix_ms() - (uint64_t)((now_ns - std::min(now_ns, a.made_ns)) / 1000000ull);
    h.use_control_vector = a.chain.use_control_vector;
    h.n_tokens = (int64_t)j.tokens.size();
    h.n_entries = (int64_t)keys.size();
    h.entry_bytes = lead.tiers.entry_bytes();
    if (a.pinned.snapshot_pos > 0 && ck_store_.enabled()) {
        int32_t slot = -1;
        bool settled = false;
        {
            const auto lk = sched_.hold();
            settled = lead.prefix.snapshot_settled(a.pinned.snapshot, &slot);
        }
        if (settled) {
            /* A save is issued on each rank's stream and committed before it lands; drained
             * streams leave none in flight. Once per archive. */
            for (auto& rk : ranks_) (void)rad_stream_sync(rk->ctx.stream());
            if (ranks_.size() > 1) (void)rad_dev_set(lead.device);
            const auto lk = sched_.hold();
            settled = lead.prefix.snapshot_settled(a.pinned.snapshot, &slot);
            if (settled) {
                const uint8_t* src = static_cast<const uint8_t*>(ck_store_.at(slot));
                j.snapshot.assign(src, src + ck_store_.slot_bytes());
            }
        }
        if (!settled) {
            sched_.archive_defer(a.id, later);
            return 3;
        }
        h.snapshot_bytes = ck_store_.slot_bytes();
        h.snapshot_pos = a.pinned.snapshot_pos;
        h.snapshot_key = a.pinned.snapshot;
    }
    std::lock_guard<std::mutex> lk(w.mu);
    w.job = std::move(j);
    w.has_job = true;
    w.busy = true;
    w.cv.notify_one();
    return 2;
}

void Engine::memory_stop() {
    if (!mem_on_) return;
    /* What is ready goes to disk before the process does: a conversation archived a moment before
     * a restart is the one its agent asks about first. One clock for the whole pass, so an archive
     * that steps aside is not offered again. */
    {
        /* The step loop has stopped, so nothing else lands the copies still waiting. */
        const auto lk = sched_.hold();
        (void)flush_copies();
    }
    const uint64_t now = rad_mono_ns();
    for (int round = 0; round < 4 * (int)cfg_.memory_archives + 4; ++round) {
        const int r = memory_step(now);
        if (r == 0) break;
        if (r == 1) {
            MemWriter& w = *mem_writer_;
            std::unique_lock<std::mutex> lk(w.mu);
            w.done_cv.wait(lk, [&] { return w.done; });
        }
    }
    if (!mem_writer_) return;
    {
        std::lock_guard<std::mutex> lk(mem_writer_->mu);
        mem_writer_->stop = true;
    }
    mem_writer_->cv.notify_one();
    /* A write in progress finishes first: its file is renamed into place or removed. */
    if (mem_writer_->thread.joinable()) mem_writer_->thread.join();
    if (mem_writer_->done) sched_.archive_written(mem_writer_->done_id, mem_writer_->ok, rad_mono_ns());
    mem_writer_.reset();
}

}  /* namespace rad */
