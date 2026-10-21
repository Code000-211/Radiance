/* avx_ngram.cpp -- where embed_lookup_q's rows come from, and the table's stored arrangement.
 * BASELINE-COMPILED: no -m flags, like avx_dispatch.cpp. Nothing here is vector arithmetic; the
 * decode that is arrives from the calling level (avx_rows.h).
 *
 * ============================== WHY THIS GATHER IS A HOST KERNEL ==============================
 *
 * An n-gram embedding of 320,001,536 rows of 160 E4M3 is 51.2 GB, read SIXTEEN ROWS A TOKEN, and
 * every mechanism for an oversized weight is built for a different shape of access:
 *
 *   Tier::SSD + DeviceStaged   stages a WHOLE movement unit into a slab slot and DMAs it. Staging
 *                              51.2 GB to use 5 KB is not a slow path, it is the wrong shape.
 *   Site::DeviceZeroCopy       wants it in pinned host RAM, which a table this size does not fit.
 *   hipHostRegister on the map pins, so it would force all 51.2 GB resident. Same wall.
 *
 * core/format/radfile.cpp mmaps the whole container -- "LOAD IS MAP-AND-GO" -- so the table is at
 * a HOST virtual address the moment the model opens, backed by a file on an NVMe. A sparse gather
 * of sixteen rows out of 320 million is a caching problem -- nothing can predict which rows are hot
 * -- but for a cache of ROWS, not of the 4 KiB pages the OS would keep; see the row cache below.
 * (core/place/mover.h rejects mmap for the WEIGHT POOL and that argument does not reach here -- it
 * is about SWEEPING a working set, where residency is controlled explicitly.)
 *
 * So the gather runs on the host, over the container's own mapping, and its [T, n_embd] bf16
 * result crosses the link once -- a few kilobytes a token at decode. The placement planner puts an
 * op on the host when every WEIGHT operand of it is Site::Host (core/runtime/ctx.cpp), which is why
 * the table has to be a weight operand: `gather_rows` takes a plain IN operand and would never
 * trigger it. A host op needs a host kernel in a production plugin -- libref is the oracle and
 * rad_gate_reference_kernels refuses to serve on it -- and host kernels are this plugin's.
 *
 * ============================== A SHARED ROW CACHE, AND DIRECT READS FOR ITS MISSES ==============================
 *
 * The gather is not arithmetic. 160 table bytes decode to 160 bf16 in a few dozen cycles; what a
 * row costs is getting its bytes, and at decode most of them have never been read before: eight
 * varied sequences want ~487 distinct rows a step, and about half of those are an n-gram hashed for
 * the first time (see the trace note below). No cache serves those; they are reads.
 *
 * THE PAGE CACHE IS THE WRONG CACHE FOR THIS TABLE, twice over. A buffered read of an uncached page
 * costs the submitting thread ~4 us -- page allocation, page-cache insertion, block mapping and the
 * bio all run inside the submission -- so a call's cold rows are started one after another: ~380 of
 * them take ~1.5 ms to submit, and the drive finishes the burst ~0.25 ms after the last one. And it
 * caches a 4 KiB page for a 160-byte row that shares its page with nothing else a gather wants, so
 * every miss evicts 4 KiB of some other file's cache to keep 160 bytes of this one.
 *
 * SO THE ROWS HAVE THEIR OWN CACHE, AND A MISS IS A DIRECT READ. RowCache below holds rows, not
 * pages, and every thread that gathers from the table shares it. A miss is an O_DIRECT read of the
 * aligned span holding the row: ~2 us to submit, nothing inserted anywhere, 512 bytes off the drive
 * where a page read takes 4096. At eight varied sequences a cache of 2^19 rows (80 MiB at 160 bytes)
 * misses 51.0% of rows, against 50.6% for an unbounded one.
 *
 * EACH MISS IS READ ONCE, BY WHICHEVER THREAD CLAIMS IT. Every rank gathers every row. A thread
 * claims a missing row by marking its slot LOADING, reads it and publishes it READY; a thread that
 * finds a row another thread is loading waits for it instead of reading it again. The ranks walk a
 * call's rows from different starting points, so each claims about its own share of the misses and
 * the cold work is divided between them rather than duplicated -- with no barrier between the
 * ranks: a late rank finds the rows already there.
 *
 * THE FILE COMES FROM THE MAPPING. The table operand is a pointer into the container's mmap and the
 * plugin ABI carries no file, so /proc/self/maps says which file and offset back that address, once
 * a table. A table no file backs -- a test's heap buffer -- is read as the memory it is. A file on a
 * filesystem that cannot do direct I/O is refused; direct_granule below finds the alignment the
 * reads keep, including on a filesystem that does not report one. */
#include "avx_rows.h"
#include "avx_common.h"

#include "rad_plugin.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

/* How many rows ahead of the one being looked up or decoded its cache lines are requested. The
 * rows are scattered over an 80 MiB cache and the set words over 4 MiB, so every row is a miss to
 * L3 or DRAM; a row's lines requested this far ahead arrive while the rows before it are worked on
 * instead of one after another. */
constexpr long long kPrefetchRows = 8;

/* One read of a row's aligned span: row m of the call, its key, and its cache slot (-1: the row is
 * decoded straight from the read, not kept). */
struct SpanRead {
    long long m;
    uint64_t  key;
    int64_t   slot;
};

/* A thread's working state for one call, kept between calls so a call allocates nothing once the
 * thread has seen its largest. */
struct Gather {
    std::vector<int64_t>   slot;     /* per row: its cache slot */
    std::vector<uint8_t>   kind;     /* per row: a RowCache::Found, or PAD / DONE below */
    std::vector<SpanRead>  reads;    /* the rows this thread reads */
    std::vector<long long> waits;    /* rows another thread is reading */
    std::vector<long long> again;    /* rows whose slot changed under the decode */
    /* One decode batch: each row's source and destination, and for a row decoded from the cache,
     * its call index and the slot word it was read under. */
    std::vector<const uint8_t*> src;
    std::vector<uint16_t*>      dst;
    std::vector<long long>      dm;
    std::vector<uint64_t>       dw;
    /* The same for rows decoded straight from a read's bounce buffer. */
    std::vector<const uint8_t*> bsrc;
    std::vector<uint16_t*>      bdst;
    uint8_t*              bounce = nullptr;
    size_t                bounce_bytes = 0;
    size_t                bounce_align = 0;

    enum : uint8_t { PAD = 16, DONE };

    /* A buffer of at least `bytes` on a page and on `align`, a power of two: a direct read's
     * buffer has to keep the file's granule, which a filesystem with large blocks puts above a
     * page. */
    uint8_t* bounce_for(size_t bytes, size_t align) {
        if (align < 4096) align = 4096;
        if (bytes > bounce_bytes || align > bounce_align) {
            std::free(bounce);
            bounce = static_cast<uint8_t*>(std::aligned_alloc(align, (bytes + align - 1) & ~(align - 1)));
            bounce_bytes = bounce ? bytes : 0;
            bounce_align = bounce ? align : 0;
        }
        return bounce;
    }
    ~Gather() { std::free(bounce); }
};

inline Gather& gather_state() {
    static thread_local Gather g;
    return g;
}

#if defined(__linux__)

/* The file behind a table: the address range of its mapping, the file offset of that range's first
 * byte, and the file opened for direct reads with the granule their offsets and lengths must keep.
 * `fd` is -1 for memory no file backs. */
struct TableFile {
    const uint8_t* lo    = nullptr;
    const uint8_t* hi    = nullptr;
    int            fd    = -1;
    long long      off   = 0;
    long long      align = 0;
};

/* THE GRANULE OF A DIRECT READ of `t`'s file, opened for them as `t.fd`: the alignment a read's
 * file offset, its length and its buffer's address must all keep. 0, with `*why` saying why, when
 * the file cannot be read that way. [p, p + n) is the table, inside the mapping, and `ino` the
 * mapped file's inode.
 *
 * STATX_DIOALIGN is the filesystem's own answer, and not every filesystem gives one: btrfs does
 * direct reads and reports nothing, and so does a tmpfs since Linux 6.6. Without the report the
 * granule is the file's block size, stx_blksize. No block filesystem's direct reads need more --
 * a filesystem's blocks are never smaller than its device's sectors -- and btrfs needs exactly
 * that: its sector size, below which it does not refuse a direct read but serves it through the
 * page cache instead.
 *
 * Reported or not, one read proves the granule before any row is served: a span of the table at
 * an offset and into a buffer each aligned to the granule and to no more, which has to return the
 * bytes the mapping holds there. That catches a filesystem refusing the granule and a file that is
 * not the one mapped. A btrfs serving the read through the page cache returns the right bytes and
 * passes it; the block size is what keeps btrfs off that path. */
long long direct_granule(const TableFile& t, const uint8_t* p, long long n, unsigned long long ino,
                         const char* path, const char** why) {
    struct statx sx;
    std::memset(&sx, 0, sizeof sx);
    if (::statx(t.fd, "", AT_EMPTY_PATH, STATX_INO | STATX_DIOALIGN, &sx) != 0) {
        *why = std::strerror(errno);
        return 0;
    }
    if (!(sx.stx_mask & STATX_INO) || sx.stx_ino != ino) {
        *why = "the file opened is not the one mapped";
        return 0;
    }
    const bool reported = (sx.stx_mask & STATX_DIOALIGN) != 0;
    if (reported && sx.stx_dio_offset_align == 0) {
        *why = "the filesystem says this file has no direct I/O";
        return 0;
    }
    const long long a = !reported ? (long long)sx.stx_blksize
                      : sx.stx_dio_offset_align > sx.stx_dio_mem_align
                      ? (long long)sx.stx_dio_offset_align : (long long)sx.stx_dio_mem_align;
    if (a <= 0 || (a & (a - 1))) {
        *why = reported ? "the filesystem's direct-I/O alignment is not a power of two"
                        : "the filesystem reports no direct-I/O alignment, and its block size is "
                          "not a power of two";
        return 0;
    }

    /* An odd multiple of the granule, so neither the offset nor the buffer keeps more than it. */
    const long long tab_lo = t.off + (long long)(p - t.lo);
    long long o = (tab_lo + a - 1) & ~(a - 1);
    if ((o / a) % 2 == 0) o += a;
    if (o + a > tab_lo + n) {
        *why = "the table is smaller than one direct read";
        return 0;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::aligned_alloc((size_t)(2 * a), (size_t)(2 * a)));
    if (!buf) {
        *why = "no memory for a read";
        return 0;
    }
    const ssize_t got = ::pread(t.fd, buf + a, (size_t)a, (off_t)o);
    const int e = errno;
    const bool same = got == a && std::memcmp(buf + a, t.lo + (o - t.off), (size_t)a) == 0;
    std::free(buf);
    if (!same) {
        *why = got < 0  ? std::strerror(e)
             : got != a ? "a read of one granule came back short"
                        : "a read returned other bytes than the mapping holds";
        return 0;
    }
    if (!reported)
        std::fprintf(stderr, "libavx: %s: the filesystem reports no direct-I/O alignment, so the "
                     "n-gram table is read in spans of its block size, %lld bytes; a direct read "
                     "of one returned the file's bytes\n", path, a);
    return a;
}

/* The file behind [p, p + n): the mapping holding `p`, extended over the mappings that continue it
 * in the same file (a madvise over a sub-range splits one mapping into several). Looked up once per
 * table and kept for the life of the process; every rank thread shares the entry and the fd, and a
 * read at an explicit offset has no file position to share. */
const TableFile* table_file(const uint8_t* p, long long n, int* err) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<TableFile>> known;
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& t : known)
        if (p >= t->lo && p + n <= t->hi) return t.get();

    auto t = std::make_unique<TableFile>();
    t->lo = p;
    t->hi = p + n;
    std::FILE* f = std::fopen("/proc/self/maps", "re");
    if (!f) { *err = RAD_E_IO; return nullptr; }
    char line[4352];
    bool found = false;
    uintptr_t end = 0;
    unsigned long long ino = 0, next_off = 0;
    std::string path;
    while (std::fgets(line, sizeof line, f)) {
        unsigned long long lo = 0, hi = 0, off = 0, inode = 0;
        char perms[8];
        unsigned dmaj = 0, dmin = 0;
        int at = 0;
        if (std::sscanf(line, "%llx-%llx %7s %llx %x:%x %llu %n", &lo, &hi, perms, &off, &dmaj,
                        &dmin, &inode, &at) < 7)
            continue;
        std::string name(line + at);
        while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
        if (!found) {
            if ((uintptr_t)p < lo || (uintptr_t)p >= hi) continue;
            if (inode == 0 || name.empty() || name[0] != '/') break;     /* no file: memory */
            found = true;
            t->lo = (const uint8_t*)(uintptr_t)lo;
            t->off = (long long)off;
            end = (uintptr_t)hi;
            ino = inode;
            next_off = off + (hi - lo);
            path = name;
        } else if ((uintptr_t)lo == end && inode == ino && name == path && off == next_off) {
            end = (uintptr_t)hi;
            next_off = off + (hi - lo);
        } else {
            break;
        }
        if ((uintptr_t)(p + n) <= end) break;
    }
    std::fclose(f);
    if (found) {
        if ((uintptr_t)(p + n) > end) { *err = RAD_E_IO; return nullptr; }
        t->hi = (const uint8_t*)end;
        t->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        const char* why = t->fd < 0 ? std::strerror(errno) : nullptr;
        if (t->fd >= 0) t->align = direct_granule(*t, p, n, ino, path.c_str(), &why);
        if (t->align == 0) {
            std::fprintf(stderr, "libavx: %s cannot be read with direct I/O (%s); the n-gram table "
                         "is read that way. A local filesystem with direct I/O -- ext4, XFS, "
                         "btrfs -- can hold the container\n", path.c_str(), why);
            if (t->fd >= 0) ::close(t->fd);
            *err = RAD_E_UNSUPPORTED;
            return nullptr;
        }
    }
    known.push_back(std::move(t));
    return known.back().get();
}

/* ============================== THE ROW CACHE ==============================
 *
 * Set-associative: a row's key -- its byte offset in the file -- hashes to a set of kWays slots, and
 * a slot is one atomic word plus the row's bytes. The word is the slot's whole state:
 *
 *      key << 10 | owner << 2 | state          state: EMPTY (word 0), LOADING, READY
 *
 * owner names the thread loading the row, so a thread that meets its own claim again (a row twice in
 * one call) does not wait on itself. A reader of a READY slot copies the bytes and then reads the
 * word again: a slot's bytes are only rewritten after its word has left READY, so an unchanged word
 * means the copy was that row's (a seqlock). The victim in a full set is chosen by CLOCK over
 * per-slot reference bits, kept outside the word so that marking a row used never fails another
 * reader's check; a LOADING slot is never a victim.
 *
 * A set's eight words are one 64-byte line. The arrays are anonymous mappings, so a slot costs
 * memory once a row lands in it. */
struct RowCache {
    static constexpr int      kWays = 8;
    static constexpr uint64_t kDefaultRows = 1ull << 27;
    enum : uint64_t { EMPTY = 0, LOADING = 1, READY = 2 };

    const TableFile*       tf       = nullptr;
    long long              n        = 0;
    uint64_t               rows     = 0;
    uint64_t               set_mask = 0;
    std::atomic<uint64_t>* word     = nullptr;
    std::atomic<uint8_t>*  ref      = nullptr;
    std::atomic<uint8_t>*  hand     = nullptr;
    uint8_t*               data     = nullptr;
    std::atomic<uint32_t>  published{0};    /* bumped, and waited on, as a batch of rows lands */

    static uint64_t ready(uint64_t key) { return key << 10 | READY; }

    static uint64_t mix(uint64_t x) {
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
        x ^= x >> 27; x *= 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    const void* set_line(uint64_t key) const { return &word[(mix(key) & set_mask) * kWays]; }

    enum Found { HIT, CLAIMED, MINE, THEIRS, NONE };

    /* The slot holding `key` -- READY (HIT), being loaded by this thread (MINE) or by another
     * (THEIRS) -- or a slot claimed for it now (CLAIMED), or none when every slot of its set is
     * being loaded. */
    Found find_or_claim(uint64_t key, uint64_t me, int64_t* slot) {
        const uint64_t set = mix(key) & set_mask;
        const uint64_t s0  = set * kWays;
        for (;;) {
            int64_t  victim = -1;
            uint64_t vw     = 0;
            for (int w = 0; w < kWays; ++w) {
                const uint64_t x = word[s0 + w].load(std::memory_order_acquire);
                if (x != EMPTY && (x >> 10) == key) {
                    *slot = (int64_t)(s0 + w);
                    if ((x & 3) == READY) {
                        ref[s0 + w].store(1, std::memory_order_relaxed);
                        return HIT;
                    }
                    return ((x >> 2) & 255) == me ? MINE : THEIRS;
                }
                if (x == EMPTY && victim < 0) victim = (int64_t)(s0 + w);
            }
            /* CLOCK: clear reference bits until a READY slot without one comes round. Two turns
             * clear them all, so a victim is found unless every slot is LOADING. */
            for (int k = 0; victim < 0 && k < 2 * kWays; ++k) {
                const uint8_t  h = hand[set].load(std::memory_order_relaxed);
                hand[set].store((uint8_t)((h + 1) % kWays), std::memory_order_relaxed);
                const uint64_t s = s0 + h % kWays;
                const uint64_t x = word[s].load(std::memory_order_acquire);
                if ((x & 3) != READY) continue;
                if (ref[s].load(std::memory_order_relaxed)) {
                    ref[s].store(0, std::memory_order_relaxed);
                    continue;
                }
                victim = (int64_t)s;
                vw = x;
            }
            if (victim < 0) { *slot = -1; return NONE; }
            if (word[victim].compare_exchange_strong(vw, key << 10 | me << 2 | LOADING,
                                                     std::memory_order_acq_rel)) {
                /* The bytes are rewritten only after this; see the seqlock note above. */
                std::atomic_thread_fence(std::memory_order_release);
                *slot = victim;
                return CLAIMED;
            }
            /* Another thread changed the set between the look and the claim: look again. */
        }
    }
};

template <class T>
T* anon_array(size_t count) {
    void* p = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<T*>(p);
}

/* The cache for rows of `n` bytes from `tf`, made on first use and kept for the life of the
 * process. */
RowCache* row_cache(const TableFile* tf, long long n) {
    static std::mutex mu;
    static std::vector<std::unique_ptr<RowCache>> caches;
    std::lock_guard<std::mutex> lk(mu);
    for (const auto& c : caches)
        if (c->tf == tf && c->n == n) return c.get();
    auto c = std::make_unique<RowCache>();
    c->tf       = tf;
    c->n        = n;
    uint64_t total_rows = RowCache::kDefaultRows;
    if (const char* env = std::getenv("RADIANCE_PLE_CACHE_ROWS")) {
        long long v = std::strtoll(env, nullptr, 0);
        if (v >= 65536) {
            uint64_t p = 65536;
            while (p < (uint64_t)v) p <<= 1;
            total_rows = p;
        }
    }
    c->rows     = total_rows;
    c->set_mask = total_rows / RowCache::kWays - 1;
    c->word     = anon_array<std::atomic<uint64_t>>(total_rows);
    c->ref      = anon_array<std::atomic<uint8_t>>(total_rows);
    c->hand     = anon_array<std::atomic<uint8_t>>(total_rows / RowCache::kWays);
    c->data     = anon_array<uint8_t>(total_rows * (size_t)n);
    if (!c->word || !c->ref || !c->hand || !c->data) return nullptr;
    caches.push_back(std::move(c));
    return caches.back().get();
}

/* This thread's owner tag in a LOADING word: distinct for every thread that gathers, 1..255. */
inline uint64_t owner_tag() {
    static std::atomic<uint32_t> next{0};
    static thread_local const uint32_t me = ++next;
    return me <= 255 ? me : 0;
}

struct ReadRing {
    static constexpr unsigned kEntries = 4096;
    int            fd       = -1;
    unsigned       entries  = 0;
    unsigned      *sq_tail  = nullptr, *sq_mask = nullptr, *sq_array = nullptr;
    unsigned      *cq_head  = nullptr, *cq_tail = nullptr, *cq_mask = nullptr;
    io_uring_sqe  *sqes     = nullptr;
    io_uring_cqe  *cqes     = nullptr;
    void          *sq_ring  = nullptr, *cq_ring = nullptr;
    size_t         sq_sz = 0, cq_sz = 0, sqe_sz = 0;
    int            err      = 0;     /* the errno that stopped init(), for ring_refusal() */

    /* One thread submits and reaps (SINGLE_ISSUER), and completions are run when it asks for them
     * rather than by interrupting it (DEFER_TASKRUN): the ring is thread-local and the thread only
     * ever waits for a batch it has just submitted. */
    bool init() {
        io_uring_params p;
        std::memset(&p, 0, sizeof p);
        p.flags = IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN;
        const long r = syscall(__NR_io_uring_setup, kEntries, &p);
        if (r < 0) { err = errno; return false; }
        fd      = (int)r;
        entries = p.sq_entries;
        sq_sz   = p.sq_off.array + (size_t)p.sq_entries * sizeof(unsigned);
        cq_sz   = p.cq_off.cqes  + (size_t)p.cq_entries * sizeof(io_uring_cqe);
        if (p.features & IORING_FEAT_SINGLE_MMAP) {
            if (cq_sz > sq_sz) sq_sz = cq_sz;
            cq_sz = sq_sz;
        }
        sq_ring = mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       fd, IORING_OFF_SQ_RING);
        if (sq_ring == MAP_FAILED) { err = errno; sq_ring = nullptr; shut(); return false; }
        cq_ring = (p.features & IORING_FEAT_SINGLE_MMAP)
                ? sq_ring
                : mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       fd, IORING_OFF_CQ_RING);
        if (cq_ring == MAP_FAILED) { err = errno; cq_ring = nullptr; shut(); return false; }
        sqe_sz = (size_t)p.sq_entries * sizeof(io_uring_sqe);
        sqes   = (io_uring_sqe*)mmap(nullptr, sqe_sz, PROT_READ | PROT_WRITE,
                                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sqes == MAP_FAILED) { err = errno; sqes = nullptr; shut(); return false; }
        sq_tail  = (unsigned*)((char*)sq_ring + p.sq_off.tail);
        sq_mask  = (unsigned*)((char*)sq_ring + p.sq_off.ring_mask);
        sq_array = (unsigned*)((char*)sq_ring + p.sq_off.array);
        cq_head  = (unsigned*)((char*)cq_ring + p.cq_off.head);
        cq_tail  = (unsigned*)((char*)cq_ring + p.cq_off.tail);
        cq_mask  = (unsigned*)((char*)cq_ring + p.cq_off.ring_mask);
        cqes     = (io_uring_cqe*)((char*)cq_ring + p.cq_off.cqes);
        return true;
    }

    void shut() {
        if (sqes    && sqes    != MAP_FAILED) munmap(sqes, sqe_sz);
        if (cq_ring && cq_ring != sq_ring)    munmap(cq_ring, cq_sz);
        if (sq_ring)                          munmap(sq_ring, sq_sz);
        if (fd >= 0)                          ::close(fd);
        sqes = nullptr; sq_ring = cq_ring = nullptr; fd = -1;
    }
    ~ReadRing() { shut(); }

    /* The aligned span holding each row of `r[0, count)` (count <= entries) into `bounce`, one span
     * every `stride` bytes; submitted with one call that also waits for all of them. Every read
     * must return its whole span. */
    int read_spans(const TableFile& tf, const SpanRead* r, unsigned count, long long n,
                   long long stride, uint8_t* bounce) {
        const long long A = tf.align;
        unsigned tail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
        for (unsigned i = 0; i < count; ++i) {
            const long long a = (long long)r[i].key & ~(A - 1);
            const long long e = ((long long)r[i].key + n + A - 1) & ~(A - 1);
            const unsigned idx = tail & *sq_mask;
            io_uring_sqe* s = &sqes[idx];
            std::memset(s, 0, sizeof *s);
            s->opcode    = IORING_OP_READ;
            s->fd        = tf.fd;
            s->off       = (uint64_t)a;
            s->addr      = (uint64_t)(uintptr_t)(bounce + (size_t)i * stride);
            s->len       = (unsigned)(e - a);
            s->user_data = (uint64_t)(e - a);
            sq_array[idx] = idx;
            ++tail;
        }
        __atomic_store_n(sq_tail, tail, __ATOMIC_RELEASE);
        int status = RAD_OK;
        unsigned to_submit = count, got = 0;
        while (got < count) {
            const long rc = syscall(__NR_io_uring_enter, fd, to_submit, count - got,
                                    IORING_ENTER_GETEVENTS, nullptr, (size_t)0);
            if (rc < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) return RAD_E_IO;
            if (rc > 0) to_submit -= (unsigned)rc < to_submit ? (unsigned)rc : to_submit;
            unsigned h = __atomic_load_n(cq_head, __ATOMIC_RELAXED);
            const unsigned tl = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
            for (; h != tl; ++h, ++got) {
                const io_uring_cqe& c = cqes[h & *cq_mask];
                if (c.res < 0 || (uint64_t)c.res != c.user_data) status = RAD_E_IO;
            }
            __atomic_store_n(cq_head, h, __ATOMIC_RELEASE);
        }
        return status;
    }
};

/* Thread-local: each rank thread issues its own step, so each gets its own ring and the two never
 * contend. Built on that thread's first gather over a file and torn down when the thread exits. */
/* This thread's ring, made on its first gather. Null if it could not be made, with the errno in
 * `*why`. */
inline ReadRing* read_ring(int* why) {
    static thread_local bool tried = false;
    static thread_local int  err   = 0;
    static thread_local std::unique_ptr<ReadRing> r;
    if (!tried) {
        tried = true;
        std::unique_ptr<ReadRing> p(new ReadRing());
        if (p->init()) r = std::move(p);
        else           err = p->err ? p->err : EIO;
    }
    if (!r && why) *why = err;
    return r.get();
}

/* WHY THIS PROCESS CANNOT MAKE A RING, said in terms of what to change. Docker's default seccomp
 * profile is by far the commonest: it answers EPERM to io_uring_setup, and so does the
 * kernel.io_uring_disabled sysctl. A kernel too old for SINGLE_ISSUER and DEFER_TASKRUN answers
 * EINVAL. */
std::string ring_refusal(int e) {
    std::string s = "libavx: the n-gram table is read through io_uring, and this process cannot "
                    "make a ring (" + std::string(std::strerror(e)) + "). ";
    if (e == EPERM || e == EACCES)
        s += "Docker's default seccomp profile refuses io_uring: run the container with "
             "--security-opt seccomp=unconfined (docs/DOCKER.md). Outside a container, the "
             "kernel.io_uring_disabled sysctl refuses it the same way.";
    else if (e == ENOSYS)
        s += "This kernel is built without io_uring.";
    else if (e == EINVAL)
        s += "The ring needs IORING_SETUP_SINGLE_ISSUER and IORING_SETUP_DEFER_TASKRUN, which "
             "Linux 6.1 added.";
    return s;
}

/* Whether this process can make the ring every gathering thread makes: 0, or the errno that
 * stopped it. Asked once, by making one and closing it. */
int ring_errno() {
    static const int e = [] {
        ReadRing r;
        return r.init() ? 0 : (r.err ? r.err : EIO);
    }();
    return e;
}

/* Read `reads` through the ring, a batch at a time: a row with a slot is copied into the cache and
 * published READY; a row without one is decoded straight into its output row. On a failed read,
 * every claim not yet published is given up (its word back to EMPTY) so no other thread waits on
 * it. */
int read_and_publish(const TableFile& tf, RowCache& rc, ReadRing& rr, Gather& g,
                     const std::vector<SpanRead>& reads, long long n, long long ne,
                     avx::RowDecode decode, const uint16_t* lut, uint16_t* out, long long x_ld) {
    const long long A      = tf.align;
    const long long stride = ((n + A - 1) / A + 1) * A;
    const size_t    total  = reads.size();
    for (size_t b = 0; b < total;) {
        const unsigned cnt = (unsigned)(total - b < rr.entries ? total - b : rr.entries);
        uint8_t* bounce = g.bounce_for((size_t)cnt * (size_t)stride, (size_t)A);
        const int rs = bounce ? rr.read_spans(tf, &reads[b], cnt, n, stride, bounce) : RAD_E_NOMEM;
        if (rs != RAD_OK) {
            for (size_t i = b; i < total; ++i)
                if (reads[i].slot >= 0)
                    rc.word[reads[i].slot].store(RowCache::EMPTY, std::memory_order_release);
            rc.published.fetch_add(1, std::memory_order_release);
            rc.published.notify_all();
            return rs;
        }
        g.bsrc.clear();
        g.bdst.clear();
        for (unsigned i = 0; i < cnt; ++i) {
            const SpanRead& r = reads[b + i];
            const uint8_t* row = bounce + (size_t)i * stride + (r.key & (uint64_t)(A - 1));
            if (r.slot < 0) {
                g.bsrc.push_back(row);
                g.bdst.push_back(out + r.m * x_ld);
                continue;
            }
            std::memcpy(rc.data + (size_t)r.slot * (size_t)n, row, (size_t)n);
            rc.ref[r.slot].store(1, std::memory_order_relaxed);
            rc.word[r.slot].store(RowCache::ready(r.key), std::memory_order_release);
        }
        rc.published.fetch_add(1, std::memory_order_release);
        rc.published.notify_all();
        if (!g.bsrc.empty()) decode(g.bsrc.data(), g.bdst.data(), (int64_t)g.bsrc.size(), ne, lut);
        b += cnt;
    }
    return RAD_OK;
}

/* out[m] = the decoded row ids[m] - voff for every m < M, through the cache: find or claim every
 * row starting at row `first`, read this thread's claims, wait for the rows other threads are
 * reading, then decode everything from the cache. A row whose slot is taken over while it is being
 * decoded is read again, without the cache. */
int gather_rows(const TableFile& tf, RowCache& rc, ReadRing& rr, const avx::RowGather& q,
                long long first, avx::RowDecode decode, const uint16_t* lut) {
    Gather& g = gather_state();
    const uint64_t me = owner_tag();
    if (!me) return RAD_E_UNSUPPORTED;
    /* `n` is a row's BYTES -- what the cache holds and the file offsets step by -- and `ne` its
     * elements, what the decode and a padding row write. */
    const long long M = q.M, n = q.row_bytes, ne = q.n_embd, x_ld = q.x_ld;
    const long long base = tf.off + (long long)(q.tab - tf.lo);
    auto row_of = [&](long long m) { return (long long)q.ids[m] - q.voff; };
    auto pad    = [&](long long id) { return id < 0 || id >= q.n_vocab; };
    auto key_of = [&](long long m) { return (uint64_t)(base + row_of(m) * n); };
    auto at     = [&](long long j) { return first + j < M ? first + j : first + j - M; };

    g.slot.resize((size_t)M);
    g.kind.resize((size_t)M);
    g.reads.clear();
    g.waits.clear();
    g.again.clear();

    for (long long j = 0; j < M; ++j) {
        if (j + kPrefetchRows < M) {
            const long long mp = at(j + kPrefetchRows);
            if (!pad(row_of(mp))) __builtin_prefetch(rc.set_line(key_of(mp)));
        }
        const long long m = at(j);
        if (pad(row_of(m))) { g.kind[m] = Gather::PAD; continue; }
        const uint64_t key = key_of(m);
        int64_t s = -1;
        const RowCache::Found f = rc.find_or_claim(key, me, &s);
        g.slot[m] = s;
        g.kind[m] = (uint8_t)f;
        if (f == RowCache::CLAIMED)     g.reads.push_back({m, key, s});
        else if (f == RowCache::NONE) { g.reads.push_back({m, key, -1}); g.kind[m] = Gather::DONE; }
        else if (f == RowCache::THEIRS) g.waits.push_back(m);
    }

    int rs = read_and_publish(tf, rc, rr, g, g.reads, n, ne, decode, lut, q.out, x_ld);
    if (rs != RAD_OK) return rs;

    /* The rows another thread claimed: wait for each to turn READY. `published` is read before the
     * words, so a batch that lands after the look changes it and the wait returns at once. A claim
     * given up (a failed read) or a slot that moved on is read here instead. */
    while (!g.waits.empty()) {
        const uint32_t e = rc.published.load(std::memory_order_acquire);
        size_t keep = 0;
        for (long long m : g.waits) {
            const uint64_t key = key_of(m);
            const uint64_t x   = rc.word[g.slot[m]].load(std::memory_order_acquire);
            if (x == RowCache::ready(key)) continue;
            if ((x & 3) == RowCache::LOADING && (x >> 10) == key) { g.waits[keep++] = m; continue; }
            g.again.push_back(m);
            g.kind[m] = Gather::DONE;
        }
        g.waits.resize(keep);
        if (keep) rc.published.wait(e, std::memory_order_acquire);
    }

    /* Every READY row, decoded in one batch: each row's word is read before its bytes and read
     * again after all of them (the seqlock above, over the batch), and a row whose word changed in
     * between is read again. */
    g.src.clear();
    g.dst.clear();
    g.dm.clear();
    g.dw.clear();
    for (long long m = 0; m < M; ++m) {
        uint16_t* dst = q.out + m * x_ld;
        const uint8_t k = g.kind[m];
        if (k == Gather::PAD)  { std::memset(dst, 0, (size_t)ne * sizeof(uint16_t)); continue; }
        if (k == Gather::DONE) continue;
        const int64_t  s  = g.slot[m];
        const uint64_t w1 = rc.word[s].load(std::memory_order_acquire);
        if (w1 != RowCache::ready(key_of(m))) { g.again.push_back(m); continue; }
        g.src.push_back(rc.data + (size_t)s * (size_t)n);
        g.dst.push_back(dst);
        g.dm.push_back(m);
        g.dw.push_back(w1);
    }
    if (!g.src.empty()) decode(g.src.data(), g.dst.data(), (int64_t)g.src.size(), ne, lut);
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t i = 0; i < g.dm.size(); ++i)
        if (rc.word[g.slot[g.dm[i]]].load(std::memory_order_relaxed) != g.dw[i])
            g.again.push_back(g.dm[i]);

    if (g.again.empty()) return RAD_OK;
    g.reads.clear();
    for (long long m : g.again) g.reads.push_back({m, key_of(m), -1});
    return read_and_publish(tf, rc, rr, g, g.reads, n, ne, decode, lut, q.out, x_ld);
}

/* ============================== THE READER THAT RUNS A STEP AHEAD ==============================
 *
 * A prefill chunk gathers `heads` rows a token -- 32,768 at a 2048-token chunk -- and on text the
 * cache has not seen, most of them are reads from the file: ~30 ms of random reads a chunk on an
 * NVMe drive, during which the step's issuing thread waits in this op and the card idles
 * behind it. The NEXT chunk's ids are known a step early (RadBatch::n_ahead, ngram_ids'
 * `ahead_ids`), so the call hands them to one process-wide reader thread and returns. The reader
 * claims each row the cache lacks, reads it and publishes it READY while the chunk runs on the
 * card; the next call finds its rows in the cache, or LOADING under the reader's tag, and waits
 * for those exactly as it waits for a row the other rank is reading.
 *
 * Both ranks hand over the same ids, and the second hand-over of a list the reader already has is
 * dropped. A failed read gives its claims up (read_and_publish), and the next call reads those
 * rows itself. The reader lives as long as the process: it is never joined, so neither it nor its
 * queue is ever destroyed under it. */
struct AheadJob {
    const TableFile*      tf = nullptr;
    RowCache*             rc = nullptr;
    long long             n  = 0;
    std::vector<uint64_t> keys;
};

struct AheadReader {
    std::mutex              mu;
    std::condition_variable cv;
    std::deque<AheadJob>    q;
    uint64_t                last = 0;     /* the last id list taken, hashed */

    void run() {
        Gather&        g  = gather_state();
        ReadRing*      rr = read_ring(nullptr);
        const uint64_t me = owner_tag();
        for (;;) {
            AheadJob j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return !q.empty(); });
                j = std::move(q.front());
                q.pop_front();
            }
            if (!rr || !me) continue;
            g.reads.clear();
            for (size_t i = 0; i < j.keys.size(); ++i) {
                if (i + kPrefetchRows < j.keys.size())
                    __builtin_prefetch(j.rc->set_line(j.keys[i + kPrefetchRows]));
                int64_t slot = -1;
                if (j.rc->find_or_claim(j.keys[i], me, &slot) == RowCache::CLAIMED)
                    g.reads.push_back({0, j.keys[i], slot});
            }
            /* Every read has a slot, so nothing is decoded and there is no output to write. */
            (void)read_and_publish(*j.tf, *j.rc, *rr, g, g.reads, j.n, 0, nullptr, nullptr,
                                   nullptr, 0);
        }
    }
};

AheadReader* ahead_reader() {
    static AheadReader* const r = [] {
        AheadReader* a = new AheadReader();
        std::thread([a] { a->run(); }).detach();
        return a;
    }();
    return r;
}

/* Hand the rows of `q.ahead` -- rows a later call will gather -- to the reader. */
void read_ahead(const TableFile& tf, RowCache& rc, const avx::RowGather& q) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t)(uintptr_t)&rc;
    for (long long m = 0; m < q.n_ahead; ++m) { h ^= (uint32_t)q.ahead[m]; h *= 1099511628211ull; }
    AheadReader* r = ahead_reader();
    {
        std::lock_guard<std::mutex> lk(r->mu);
        if (h == r->last) return;
        r->last = h;
    }
    AheadJob j;
    j.tf = &tf;
    j.rc = &rc;
    j.n  = q.row_bytes;
    const long long base = tf.off + (long long)(q.tab - tf.lo);
    j.keys.reserve((size_t)q.n_ahead);
    for (long long m = 0; m < q.n_ahead; ++m) {
        const long long id = (long long)q.ahead[m] - q.voff;
        if (id >= 0 && id < q.n_vocab) j.keys.push_back((uint64_t)(base + id * q.row_bytes));
    }
    {
        std::lock_guard<std::mutex> lk(r->mu);
        r->q.push_back(std::move(j));
    }
    r->cv.notify_one();
}

/* ============================== THE ROW-ID TRACE ==============================
 *
 * AVX_PLE_TRACE=<path> appends every row id this op gathers, one per line. Off unless the variable
 * is set, and it changes nothing the op computes. Two design questions -- "does the page cache ever
 * hit" and "would a userspace cache of 160-byte rows beat it, being 25x denser than a 4 KiB page
 * holding one useful row" -- are the SAME question about the n-gram reuse rate, and that is a
 * property of the text being generated rather than of this code. This is how to measure it on a
 * real generation.
 *
 * ---- WHAT IT SHOWS ON A PREFILL, WHICH IS WHERE THE ROWS ARE ----------------------------------
 *
 * A ~24k-token prefill of prose, both ranks traced, with the rows read through the container's
 * mapping rather than the row cache (fault counts from /proc/<pid>/stat):
 *
 *      762,432 rows gathered      277,911 distinct      450,608 MINOR faults      2 MAJOR
 *
 * THE PAGE CACHE HITS ESSENTIALLY ALWAYS once warm. Two major faults over a whole prefill: what a
 * read through the mapping pays instead is PAGE-TABLE POPULATION -- a minor fault to map an
 * already-cached page, about 1.2 of them per row gathered, at roughly a microsecond each.
 *
 * THE DENSITY ARGUMENT HOLDS AND IT IS THE WHOLE PRIZE. The distinct rows a prefill wants land on
 * 274,887 distinct 4 KiB pages: 1.011 USEFUL ROWS A PAGE, where a page holds 25. So 44.5 MB of
 * wanted row bytes is reached through 1.13 GB of page mapping. The packed row cache above keeps
 * only the rows touched and pays no mapping at all.
 *
 * THE REUSE RATE ALONE DOES NOT CARRY IT. Per rank the same prefill gathers 381,216 rows over
 * 277,911 distinct ones -- 1.37x, and 86% of rows are wanted exactly once. A cache justified by
 * reuse would be a poor bet; one justified by density is a good one, and those are different
 * arguments for the same structure.
 *
 * ---- AND AT DECODE, WHERE THE STEP WAITS ON IT ------------------------------------------------
 *
 * Eight varied sequences at a depth-3 verify, one rank traced: a call is 512 rows over ~487
 * distinct ones, and 50.6% of those are rows the generation has never asked for before -- the
 * floor for any cache. The rest recur within a short window: a row-granular LRU of 10^5 rows misses
 * 57.8%, of 4 x 10^5 rows 51.2%, and larger buys nothing. RowCache -- 2^19 rows, eight ways, CLOCK,
 * shared by the ranks -- misses 51.0% on the same trace.
 *
 * (The even split in the raw trace is the two RANKS, not reuse. Both hash the same ids and gather
 * the same rows -- see the replication note in arch/common/rad_block_ple.h, which prices that.) */
inline std::FILE* trace_file() {
    static thread_local bool tried = false;
    static thread_local std::FILE* f = nullptr;
    if (!tried) {
        tried = true;
        const char* p = std::getenv("AVX_PLE_TRACE");
        if (p && *p) f = std::fopen(p, "ae");
    }
    return f;
}

#else   /* not Linux: no file reads; a table is read as memory */
struct TableFile { int fd = -1; };
inline const TableFile* table_file(const uint8_t*, long long, int*) { static TableFile m; return &m; }
inline std::FILE* trace_file() { return nullptr; }
#endif

}  // namespace

int avx::row_gather(const RowGather& q, RowDecode decode) {
    if (!decode || !q.ids || !q.tab || !q.out || q.M <= 0 || q.n_embd <= 0 || q.n_vocab <= 0 ||
        q.row_bytes < q.n_embd)
        return RAD_E_INVAL;

    /* WHAT THIS OP COSTS HAS TWO REGIMES, and a benchmark that repeats one prompt only ever shows
     * the cheap one. Greedy repeat traffic walks the SAME n-gram rows every step, so every row is
     * already in the row cache and the call reads nothing. Varied serving makes about half of every
     * step's rows new, and the call waits on the NVMe for those. /proc/<pid>/io says which regime a
     * number came from, and AVX_PLE_TRACE records the rows themselves. */
    if (std::FILE* tf = trace_file()) {
        for (long long m = 0; m < q.M; ++m)
            std::fprintf(tf, "%lld\n", (long long)q.ids[m] - q.voff);
    }

    /* Every E4M3 code's bf16 at this call's scale, so the decode is a table lookup. The ABI's own
     * converters, so the answer is libref's by construction. */
    uint16_t lut[256];
    for (int b = 0; b < 256; ++b)
        lut[b] = rad_f32_to_bf16(rad_fp8e4m3_to_f32((uint8_t)b) * q.scale);

    /* The rows: through the shared cache from the file behind the table, or, for a table no file
     * backs, from the table itself. The caller has refused an id past the table already; an id
     * turned into a file offset has to be one. */
    int ferr = RAD_OK;
    const TableFile* tfile = table_file(q.tab, q.n_vocab * q.row_bytes, &ferr);
    if (!tfile) return ferr;
#if defined(__linux__)
    if (tfile->fd >= 0) {
        int why = 0;
        ReadRing* rr = read_ring(&why);
        if (!rr) {
            static std::once_flag said;
            std::call_once(said, [why] { std::fprintf(stderr, "%s\n", ring_refusal(why).c_str()); });
            return RAD_E_UNSUPPORTED;
        }
        RowCache* rc = row_cache(tfile, q.row_bytes);
        if (!rc) return RAD_E_NOMEM;
        const long long ws    = q.world_size > 1 ? q.world_size : 1;
        const long long first = q.rank > 0 && q.rank < ws ? q.M * q.rank / ws : 0;
        const int rs = gather_rows(*tfile, *rc, *rr, q, first, decode, lut);
        if (rs != RAD_OK) return rs;
        /* The rows a later call will gather, read while the step runs. After this call's own
         * rows, so they never compete with them. */
        if (q.ahead && q.n_ahead > 0) read_ahead(*tfile, *rc, q);
        return RAD_OK;
    }
#endif

    Gather& g = gather_state();
    g.src.clear();
    g.dst.clear();
    for (long long m = 0; m < q.M; ++m) {
        const long long id = (long long)q.ids[m] - q.voff;
        uint16_t* dst = q.out + m * q.x_ld;
        if (id < 0 || id >= q.n_vocab) {
            std::memset(dst, 0, (size_t)q.n_embd * sizeof(uint16_t));
            continue;
        }
        g.src.push_back(q.tab + id * q.row_bytes);
        g.dst.push_back(dst);
    }
    if (!g.src.empty()) decode(g.src.data(), g.dst.data(), (int64_t)g.src.size(), q.n_embd, lut);
    return RAD_OK;
}

/* ============================== THE TABLE'S ENCODING, AND ITS ONE SCALE ==============================
 *
 * Asked at declare, not by a step (spec §4.3).
 *
 * The table is E4M3, row-major, with ONE bf16 scale for all of it -- `fp8_e4m3*bf16[*x*]`. That
 * is not a simplification of a per-block scheme: it is the shipped format, and the structure of
 * the data is why it works. E4M3 carries its own exponent; a row of this table has a dynamic range
 * of only about 2.3x; and the error sits on a THREE-OCTAVE PLATEAU in the scale, so any value from
 * 2^-13 to 2^-10 gives the same relative L2 error to five digits. A per-row scale would buy a few
 * percent of that error for 1.25% more bytes, which is why the format has none. (INT8 in the same
 * byte would buy several times more, because it spends all eight bits on resolution rather than
 * four on an exponent this data does not need. That is a different format.)
 *
 * IT IS STORED AS IT IS: the gather reads the canonical rows, and that is what lets the table be
 * read in place from the container's mapping -- a relayout would need a copy of 51 GB. So this
 * hook only checks, and a table quantised any other way is refused at declare rather than
 * gathered as noise. A plain bf16 table -- the checkpoint's own, 102 GB -- is the one other form
 * it takes, read the same way with no scale. */
namespace {

enum { EQ_TOK = 0, EQ_WTE, EQ_SCALE, EQ_X, EQ_AHEAD };

}  /* namespace */

extern "C" int avx_layout_ngram(const RadParam* p, int n_p, int operand, const RadEncoding* enc,
                                const int* sel, const RadTensor* pl, int n, RadLayout* out) {
    (void)out;
    if (operand != EQ_WTE && operand != EQ_SCALE) return RAD_E_UNSUPPORTED;
    const long long rows = rad_param_geti(p, n_p, "n_vocab", 0);
    const long long dim  = rad_param_geti(p, n_p, "n_embd", 0);
    if (rows <= 0 || dim <= 0) return RAD_E_SHAPE;
    /* OR THE TABLE AS THE CHECKPOINT SHIPS IT: plain bf16, no scale -- the gather copies its rows
     * through the same cache. It has no scale plane, so only the table is ever asked about. */
    if (rad_enc_is(enc, "plain") && enc->n_planes == 1 && enc->plane[0].dtype == RAD_BF16) {
        if (operand != EQ_WTE) return RAD_E_DTYPE;
        if (n != 1 || !pl) return RAD_E_SHAPE;
        if (pl[0].rank != 2 || pl[0].shape[0] != rows || pl[0].shape[1] != dim) return RAD_E_SHAPE;
        return RAD_E_UNSUPPORTED;
    }
    const RadEncPlane* sc = rad_enc_plane(enc, "scale");
    if (!rad_enc_is(enc, "affine") || enc->n_planes != 2 || enc->transform[0] ||
        enc->plane[0].dtype != RAD_F8E4M3 || !sc || sc->dtype != RAD_BF16 || sc->block[0] != 0 ||
        sc->block[1] != 0)
        return RAD_E_DTYPE;
    if (n != 1 || !pl || !sel) return RAD_E_SHAPE;
    const char* want = operand == EQ_WTE ? "codes" : "scale";
    if (std::strncmp(enc->plane[sel[0]].role, want, RAD_ENC_STR) != 0) return RAD_E_SHAPE;
    const long long r = operand == EQ_WTE ? rows : 1, c = operand == EQ_WTE ? dim : 1;
    if (pl[0].rank != 2 || pl[0].shape[0] != r || pl[0].shape[1] != c) return RAD_E_SHAPE;
    return RAD_E_UNSUPPORTED;
}

/* ============================== THE INSTANCE: CAN THIS PROCESS READ THE TABLE ==============================
 *
 * Every miss of the gather into bf16 is an io_uring read, and a process that may not make a ring
 * fails every step that gathers: the first request fails, after the whole model has loaded, and
 * the engine stops. So the instance asks at declare, before a byte is loaded, and refuses with
 * the reason. A gather into any other dtype takes the general path, which reads the table as
 * memory and makes no ring. rad-convert declares without instances -- it launches nothing -- so a
 * sandbox that forbids io_uring still converts the model. */
extern "C" int avx_init_ngram(const RadParam* p, int n_p, int rank, int world_size, void** out) {
    (void)rank;
    (void)world_size;
    if (out) *out = nullptr;
#if defined(__linux__)
    const char* dt = rad_param_gets(p, n_p, "dtype", nullptr);
    if (!dt || std::strcmp(dt, "bf16") != 0) return RAD_OK;
    if (const int e = ring_errno()) {
        static std::once_flag said;
        std::call_once(said, [e] { std::fprintf(stderr, "%s\n", ring_refusal(e).c_str()); });
        return RAD_E_UNSUPPORTED;
    }
#else
    (void)p;
    (void)n_p;
#endif
    return RAD_OK;
}
