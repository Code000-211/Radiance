#include "mem/kvarchive.h"

#include "rad_internal.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace rad {

namespace {

constexpr char     kMagic[8]    = { 'R', 'A', 'D', 'K', 'V', 'A', '0', '1' };
constexpr char     kEndMagic[8] = { 'R', 'A', 'D', 'K', 'V', 'E', 'N', 'D' };
constexpr uint32_t kVersion     = 1;
/* Bounds a header is checked against before anything is allocated from it. */
constexpr uint32_t kMaxString   = 64 * 1024;
constexpr int64_t  kMaxEntry    = 1ll << 30;
constexpr int64_t  kMaxEntries  = 1ll << 28;
constexpr int64_t  kMaxSnapshot = 64ll << 30;

constexpr uint64_t P1 = 0x9e3779b185ebca87ull, P2 = 0xc2b2ae3d27d4eb4full,
                   P3 = 0x165667b19e3779f9ull, P4 = 0x85ebca77c2b2ae63ull;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t load64(const uint8_t* p) { uint64_t w; std::memcpy(&w, p, 8); return w; }

void put32(std::string* b, uint32_t v) { b->append((const char*)&v, 4); }
void put64(std::string* b, uint64_t v) { b->append((const char*)&v, 8); }
void put_str(std::string* b, const std::string& s) { put32(b, (uint32_t)s.size()); b->append(s); }
void pad8(std::string* b) { while (b->size() % 8) b->push_back('\0'); }

int64_t padded8(int64_t n) { return (n + 7) / 8 * 8; }

/* fwrite that names the file and the reason. */
bool write_all(FILE* f, const void* p, size_t n, const std::string& path) {
    if (n == 0 || std::fwrite(p, 1, n, f) == n) return true;
    RAD_ERR("memory: writing %s failed: %s", path.c_str(), std::strerror(errno));
    return false;
}

bool read_all(FILE* f, void* p, size_t n) { return n == 0 || std::fread(p, 1, n, f) == n; }

std::string dir_of(const std::string& path) {
    const size_t at = path.find_last_of('/');
    return at == std::string::npos ? std::string(".") : at == 0 ? std::string("/") : path.substr(0, at);
}

}  /* namespace */

/* ------------------------------------------------------------------ checksum */

void KvChecksum::stripe(const uint8_t* p) {
    for (int i = 0; i < 4; ++i) lane_[i] = rotl(lane_[i] + load64(p + 8 * i) * P2, 31) * P1;
}

void KvChecksum::add(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_ += n;
    if (n_carry_) {
        const size_t take = std::min(n, sizeof carry_ - n_carry_);
        std::memcpy(carry_ + n_carry_, p, take);
        n_carry_ += take;
        p += take;
        n -= take;
        if (n_carry_ < sizeof carry_) return;
        stripe(carry_);
        n_carry_ = 0;
    }
    for (; n >= 32; p += 32, n -= 32) stripe(p);
    std::memcpy(carry_, p, n);
    n_carry_ = n;
}

uint64_t KvChecksum::value() const {
    uint64_t h = rotl(lane_[0], 1) + rotl(lane_[1], 7) + rotl(lane_[2], 12) + rotl(lane_[3], 18);
    for (int i = 0; i < 4; ++i) h = (h ^ (rotl(lane_[i] * P2, 31) * P1)) * P1 + P4;
    h += total_;
    for (size_t i = 0; i < n_carry_; ++i) h = rotl(h ^ (carry_[i] * P3), 11) * P1;
    h ^= h >> 33;
    h *= P2;
    h ^= h >> 29;
    h *= P3;
    h ^= h >> 32;
    return h;
}

uint64_t kva_fnv(const void* data, size_t n, uint64_t h) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

/* ------------------------------------------------------------------ writing */

int kva_write(const std::string& path, const KvArchiveHeader& h, const std::vector<int32_t>& tokens,
              const std::vector<BlockHash>& keys, const KvEntryFill& fill, const void* snapshot) {
    if ((int64_t)tokens.size() != h.n_tokens || (int64_t)keys.size() != h.n_entries ||
        h.entry_bytes <= 0 || (h.snapshot_bytes > 0 && !snapshot))
        return RAD_E_INVAL;

    std::string head;
    head.append(kMagic, 8);
    put32(&head, kVersion);
    put32(&head, 0);                                  /* header_bytes, patched below */
    put64(&head, (uint64_t)h.n_tokens);
    put64(&head, (uint64_t)h.n_entries);
    put64(&head, (uint64_t)h.entry_bytes);
    put64(&head, (uint64_t)h.snapshot_bytes);
    put64(&head, (uint64_t)h.snapshot_pos);
    put64(&head, h.snapshot_key.hi);
    put64(&head, h.snapshot_key.lo);
    put64(&head, h.made_unix_ms);
    put32(&head, h.use_control_vector ? 1u : 0u);
    put_str(&head, h.fingerprint);
    put_str(&head, h.id);
    put_str(&head, h.label);
    pad8(&head);
    const uint32_t hb = (uint32_t)head.size();
    std::memcpy(&head[12], &hb, 4);

    std::string idx;
    idx.append((const char*)tokens.data(), tokens.size() * sizeof(int32_t));
    pad8(&idx);
    for (const BlockHash& k : keys) { put64(&idx, k.hi); put64(&idx, k.lo); }

    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        RAD_ERR("memory: cannot create %s: %s", tmp.c_str(), std::strerror(errno));
        return RAD_E_IO;
    }
    int st = RAD_OK;
    KvChecksum sum;
    auto put = [&](const void* p, size_t n) {
        if (st != RAD_OK) return;
        sum.add(p, n);
        if (!write_all(f, p, n, tmp)) st = RAD_E_IO;
    };
    put(head.data(), head.size());
    put(idx.data(), idx.size());

    /* Entries in batches of about 64 MiB: the fill copies them out of host memory under the
     * tiers' lock, the write happens outside it. */
    const int64_t per = std::max<int64_t>(1, (64ll << 20) / h.entry_bytes);
    std::vector<uint8_t> buf;
    for (int64_t first = 0; st == RAD_OK && first < h.n_entries; first += per) {
        const int64_t n = std::min(per, h.n_entries - first);
        buf.resize((size_t)(n * h.entry_bytes));
        if (!fill(first, n, buf.data())) { st = RAD_E_STATE; break; }
        put(buf.data(), buf.size());
    }
    if (h.snapshot_bytes > 0) put(snapshot, (size_t)h.snapshot_bytes);

    const uint64_t v = sum.value();
    if (st == RAD_OK && !(write_all(f, &v, 8, tmp) && write_all(f, kEndMagic, 8, tmp))) st = RAD_E_IO;
    if (st == RAD_OK && (std::fflush(f) != 0 || ::fsync(fileno(f)) != 0)) {
        RAD_ERR("memory: flushing %s failed: %s", tmp.c_str(), std::strerror(errno));
        st = RAD_E_IO;
    }
    if (std::fclose(f) != 0 && st == RAD_OK) st = RAD_E_IO;
    if (st == RAD_OK && std::rename(tmp.c_str(), path.c_str()) != 0) {
        RAD_ERR("memory: renaming %s into place failed: %s", tmp.c_str(), std::strerror(errno));
        st = RAD_E_IO;
    }
    if (st != RAD_OK) {
        (void)::unlink(tmp.c_str());
        return st;
    }
    /* The rename itself is durable only once the directory is. */
    const int d = ::open(dir_of(path).c_str(), O_RDONLY | O_DIRECTORY);
    if (d >= 0) {
        (void)::fsync(d);
        ::close(d);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ reading */

KvArchiveReader::~KvArchiveReader() { close(); }

void KvArchiveReader::close() {
    if (f_) std::fclose(f_);
    f_ = nullptr;
}

int KvArchiveReader::open(const std::string& path) {
    close();
    path_ = path;
    h_ = KvArchiveHeader{};
    tokens_.clear();
    keys_.clear();
    next_ = 0;
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) return RAD_E_IO;
    struct stat sb {};
    if (::fstat(fileno(f_), &sb) != 0) return RAD_E_IO;
    const int64_t file_bytes = (int64_t)sb.st_size;

    char magic[8];
    uint32_t version = 0, header_bytes = 0, flags = 0;
    uint64_t v[8];
    if (!read_all(f_, magic, 8) || std::memcmp(magic, kMagic, 8) != 0) return RAD_E_FORMAT;
    if (!read_all(f_, &version, 4) || version != kVersion) return RAD_E_FORMAT;
    if (!read_all(f_, &header_bytes, 4) || !read_all(f_, v, sizeof v) || !read_all(f_, &flags, 4))
        return RAD_E_FORMAT;
    h_.n_tokens = (int64_t)v[0];
    h_.n_entries = (int64_t)v[1];
    h_.entry_bytes = (int64_t)v[2];
    h_.snapshot_bytes = (int64_t)v[3];
    h_.snapshot_pos = (int64_t)v[4];
    h_.snapshot_key.hi = v[5];
    h_.snapshot_key.lo = v[6];
    h_.made_unix_ms = v[7];
    h_.use_control_vector = (flags & 1u) != 0;
    if (h_.n_tokens < 0 || h_.n_entries < 0 || h_.n_entries > kMaxEntries || h_.entry_bytes <= 0 ||
        h_.entry_bytes > kMaxEntry || h_.snapshot_bytes < 0 || h_.snapshot_bytes > kMaxSnapshot ||
        h_.n_tokens > file_bytes / 4)
        return RAD_E_FORMAT;
    for (std::string* s : { &h_.fingerprint, &h_.id, &h_.label }) {
        uint32_t n = 0;
        if (!read_all(f_, &n, 4) || n > kMaxString) return RAD_E_FORMAT;
        s->resize(n);
        if (!read_all(f_, s->data(), n)) return RAD_E_FORMAT;
    }

    const int64_t tok_off = header_bytes;
    const int64_t key_off = tok_off + padded8(h_.n_tokens * 4);
    body_off_ = key_off + h_.n_entries * 16;
    const int64_t want = body_off_ + h_.n_entries * h_.entry_bytes + h_.snapshot_bytes + 16;
    if (tok_off < 8 || want != file_bytes) return RAD_E_FORMAT;

    tokens_.resize((size_t)h_.n_tokens);
    keys_.resize((size_t)h_.n_entries);
    if (::fseeko(f_, tok_off, SEEK_SET) != 0 ||
        !read_all(f_, tokens_.data(), tokens_.size() * sizeof(int32_t)) ||
        ::fseeko(f_, key_off, SEEK_SET) != 0)
        return RAD_E_FORMAT;
    for (BlockHash& k : keys_)
        if (!read_all(f_, &k.hi, 8) || !read_all(f_, &k.lo, 8)) return RAD_E_FORMAT;
    return RAD_OK;
}

int KvArchiveReader::seek_body() {
    next_ = 0;
    return ::fseeko(f_, body_off_, SEEK_SET) == 0 ? RAD_OK : RAD_E_IO;
}

int KvArchiveReader::verify() {
    if (!f_) return RAD_E_STATE;
    const int64_t payload = body_off_ + h_.n_entries * h_.entry_bytes + h_.snapshot_bytes;
    if (::fseeko(f_, 0, SEEK_SET) != 0) return RAD_E_IO;
    KvChecksum sum;
    std::vector<uint8_t> buf(8u << 20);
    for (int64_t left = payload; left > 0;) {
        const size_t n = (size_t)std::min<int64_t>(left, (int64_t)buf.size());
        if (!read_all(f_, buf.data(), n)) return RAD_E_FORMAT;
        sum.add(buf.data(), n);
        left -= (int64_t)n;
    }
    uint64_t stored = 0;
    char end[8];
    if (!read_all(f_, &stored, 8) || !read_all(f_, end, 8) || std::memcmp(end, kEndMagic, 8) != 0 ||
        stored != sum.value())
        return RAD_E_FORMAT;
    return seek_body();
}

int KvArchiveReader::next_entry(void* dst) {
    if (!f_ || next_ >= h_.n_entries) return RAD_E_STATE;
    if (!read_all(f_, dst, (size_t)h_.entry_bytes)) return RAD_E_IO;
    ++next_;
    return RAD_OK;
}

int KvArchiveReader::snapshot(void* dst) {
    if (!f_) return RAD_E_STATE;
    if (h_.snapshot_bytes == 0) return RAD_OK;
    if (::fseeko(f_, body_off_ + h_.n_entries * h_.entry_bytes, SEEK_SET) != 0) return RAD_E_IO;
    next_ = h_.n_entries;
    return read_all(f_, dst, (size_t)h_.snapshot_bytes) ? RAD_OK : RAD_E_IO;
}

}  /* namespace rad */
