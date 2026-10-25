/* kvarchive.h -- an archived conversation on disk: the blocks and snapshot the prefix cache pins
 * for it (PrefixCache::pin), so a restart does not lose what /v1/memory/archive kept.
 *
 * One file per archive, in --memory-dir, written once every pinned block has a host copy (the
 * bytes are read from host memory, never from the card) and read back at startup, before the
 * server takes requests, into the host tier as off-device entries. A hit then restores them as it
 * restores any demoted entry.
 *
 *   header        magic, version, counts, the fingerprint, the archive's id and label
 *   tokens        n_tokens int32, the conversation as the model saw it (the fork's source)
 *   keys          n_entries BlockHash, the chain hash of each pinned block
 *   entries       n_entries x entry_bytes, each block's host slot (every rank's shard)
 *   snapshot      snapshot_bytes, the linear state at snapshot_pos (0 bytes on a pure attention
 *                 model, or when the conversation had none)
 *   trailer       checksum of everything above, end magic
 *
 * The bytes are the engine's own layout, so a file is valid only for the deployment that wrote
 * it: the fingerprint names the container, the plugins, tensor parallelism, the KV dtype and
 * every size, and a file whose fingerprint differs is left alone. A crash mid-write leaves a
 * `.tmp` file that the next start removes; a write is renamed into place only after it has
 * reached the disk, and a file whose checksum fails is set aside as `.bad`. */
#pragma once
#include "mem/prefix.h"
#include "rad_core.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace rad {

struct KvArchiveHeader {
    std::string id;
    std::string label;
    std::string fingerprint;
    uint64_t    made_unix_ms = 0;
    bool        use_control_vector = false;
    int64_t     n_tokens = 0;
    int64_t     n_entries = 0;
    int64_t     entry_bytes = 0;
    int64_t     snapshot_bytes = 0;     /* 0 without a snapshot */
    int64_t     snapshot_pos = 0;
    BlockHash   snapshot_key;
};

/* A streaming 64-bit checksum: fast enough that verifying a file costs about what reading it
 * does. Not cryptographic; it catches torn writes and bit rot. */
class KvChecksum {
public:
    void     add(const void* p, size_t n);
    uint64_t value() const;

private:
    /* Four independent lanes over 32-byte stripes, so the multiplies overlap. */
    uint64_t lane_[4] = { 0x243f6a8885a308d3ull, 0x13198a2e03707344ull, 0xa4093822299f31d0ull,
                          0x082efa98ec4e6c89ull };
    uint64_t total_ = 0;
    uint8_t  carry_[32] = {};
    size_t   n_carry_ = 0;
    void     stripe(const uint8_t* p);
};

/* `fill(first, n, dst)` copies entries [first, first + n) into dst, entry_bytes each; false
 * aborts the write. Written to `path`.tmp, flushed to the disk, renamed to `path` and the directory
 * flushed. RAD_OK, or RAD_E_IO / RAD_E_STATE (a fill refused) with the temporary file removed. */
using KvEntryFill = std::function<bool(int64_t first, int64_t n, uint8_t* dst)>;
int kva_write(const std::string& path, const KvArchiveHeader& h, const std::vector<int32_t>& tokens,
              const std::vector<BlockHash>& keys, const KvEntryFill& fill,
              const void* snapshot);

/* Reads an archive in order: the header and tokens and keys at open, then entries one at a time,
 * then the snapshot. verify() reads the whole file and checks its checksum without keeping
 * anything, for use before anything is adopted. */
class KvArchiveReader {
public:
    ~KvArchiveReader();
    /* RAD_OK, RAD_E_IO, or RAD_E_FORMAT (not an archive, another version, truncated). */
    int  open(const std::string& path);
    void close();
    const KvArchiveHeader&        header() const { return h_; }
    const std::vector<int32_t>&   tokens() const { return tokens_; }
    const std::vector<BlockHash>& keys() const { return keys_; }
    /* The checksum over the whole file; RAD_E_FORMAT on a mismatch. Leaves the reader at the
     * first entry. */
    int  verify();
    /* The next entry's bytes (entry_bytes) or, after the last entry, the snapshot's. */
    int  next_entry(void* dst);
    int  snapshot(void* dst);

private:
    FILE*                  f_ = nullptr;
    std::string            path_;
    KvArchiveHeader        h_;
    std::vector<int32_t>   tokens_;
    std::vector<BlockHash> keys_;
    int64_t                body_off_ = 0;       /* the first entry's offset */
    int64_t                next_ = 0;           /* entries read so far */
    int                    seek_body();
};

/* 64-bit FNV-1a, for fingerprints. */
uint64_t kva_fnv(const void* p, size_t n, uint64_t h = 0xcbf29ce484222325ull);

}  /* namespace rad */
