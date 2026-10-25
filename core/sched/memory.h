/* memory.h -- conversations by label, and the archives pinned from them: the engine side of the
 * memory endpoints (core/server, the /v1/memory routes).
 *
 * A client that compacts its conversation (an agent harness replacing old turns with a summary)
 * stops sending the transcript the cache still holds, and the cache lets it go. Two things keep
 * it reachable:
 *
 *   - Labels. Each finished request that carries a session label (`session_id`) leaves its exact
 *     token sequence, prompt ++ output, under that label. A fork must replay those tokens to hit
 *     the cache: the prefix cache keeps whole blocks only, and rendering the client's messages
 *     through the chat template again is not exact (a template may drop earlier turns'
 *     reasoning, and a client may serialise tool calls differently on the way back).
 *   - Archives. Pinning a chain under an id keeps its blocks and its deepest snapshot through
 *     eviction, expiry and retention (PrefixCache::pin) until the archive is deleted.
 *
 * Plain data. The scheduler owns it under its lock and does the pinning.
 */
#pragma once
#include "mem/prefix.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rad {

/* A conversation as the model saw it, and the keys its blocks were cached under: a fork that
 * runs without the same adapter or control vector computes different KV and must not hit. */
struct MemChain {
    std::vector<int32_t> tokens;           /* prompt ++ output */
    uint64_t lora_uid = 0;
    bool     use_control_vector = false;
    /* It carried pictures or audio: its placeholder tokens mean nothing without the media, which
     * a fork does not have, so it is refused. */
    bool     has_media = false;
    uint64_t updated_ns = 0;               /* monotonic */
};

struct MemArchive {
    std::string         id;
    std::string         label;             /* the session it was taken from, "" if none */
    MemChain            chain;
    PrefixCache::Pinned pinned;
    uint64_t            made_ns = 0;       /* monotonic */
};

/* What /v1/memory/archive reports about one archive. */
struct MemArchiveStat {
    std::string id;
    std::string label;
    int64_t     tokens = 0;                /* the chain's length */
    int64_t     pinned_tokens = 0;         /* of those, the whole blocks the cache still held */
    int64_t     snapshot_pos = 0;          /* where a fork resumes; 0 without recurrent state */
    double      age_s = 0;
};

class MemoryRegistry {
public:
    /* Labels kept at once. Each holds a conversation's tokens (4 bytes a token, 0.8 MB at 200K),
     * so the oldest label goes when a new one would pass this. */
    static constexpr size_t kMaxLabels = 64;

    void note(const std::string& label, MemChain c) {
        if (label.empty()) return;
        auto it = labels_.find(label);
        if (it == labels_.end() && labels_.size() >= kMaxLabels) {
            auto oldest = labels_.begin();
            for (auto j = labels_.begin(); j != labels_.end(); ++j)
                if (j->second.updated_ns < oldest->second.updated_ns) oldest = j;
            labels_.erase(oldest);
        }
        labels_[label] = std::move(c);
    }
    const MemChain* chain(const std::string& label) const {
        auto it = labels_.find(label);
        return it == labels_.end() ? nullptr : &it->second;
    }

    std::unordered_map<std::string, MemArchive>& archives() { return archives_; }
    const std::unordered_map<std::string, MemArchive>& archives() const { return archives_; }

    /* The cache was reset: every pin is gone with it. */
    void clear_archives() { archives_.clear(); }

private:
    std::unordered_map<std::string, MemChain>   labels_;
    std::unordered_map<std::string, MemArchive> archives_;
};

}  /* namespace rad */
