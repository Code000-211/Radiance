/* tests/r4d_emu: the PLE row cache's productive recheck (r4d_model_bf16.hip), on the gather's own
 * kernel entry -- the one every launch takes, not the probe entry r4d_selftest's forced window
 * uses.
 *
 * The gap this closes: r4d_selftest forces the reclaim/rewrite window through
 * ple_embed_lookup_q_probe_kernel, whose recheck is written as `now == g && same_id`
 * (the Probe instantiation). The gather's recheck is the other branch of the same template,
 * `load(gen+slot) == g && load(tags+slot) == id`, and a mutation of that expression alone leaves
 * the probe entry's assertions structurally green: the conflict count, the recheck generation and
 * the retire all run on the probe path. So the production expression had no test that reddens
 * when it is violated.
 *
 * This one does, and without a card. The emulator runs blocks one after another, so a cross-block
 * interleaving is not hoped for but placed: emu_park holds a block at a chosen cross-lane exchange
 * while the rest of the grid runs. The case forces the gather's own window with three readers of
 * one launch:
 *
 *   block 0  reads row A (a published hit at generation 2) and is held at its second exchange --
 *            after the lookup, before the payload load;
 *   block 1  reads row X, uncached, so it claims the coldest way of the same set, which is A's,
 *            rewrites it with X's payload, publishes generation 4, and finishes;
 *   block 2  reads row A again: A's way now carries tag X, so it is a miss, and its claim takes
 *            that same way (tag A, generation 5) -- and is held at its third exchange, between the
 *            tag write and the payload write.
 *
 * Block 0 is let go there: its payload load reads X's row while the way's generation went 2 -> 5
 * and its tag is A again. `hit` must therefore be false on the generation alone, and block 0's row
 * comes from its own decode of A. The bytes are compared against the same launch on the uncached
 * path (no instance: the generic kernel, one block a row, no cache), which is the oracle here.
 *
 * With the generation term dropped from the gather's recheck, block 0 hits and writes X's row into
 * A's output row: this case reports it, row 0, entry by entry. That is the mutation evidence; the
 * run here is the red one's complement, and any future change to that expression runs into it.
 *
 * The window is asserted, not assumed: both parks must have been reached (emu_park_hit) and the
 * launch must have taken the gather's entry (emu_last_kernel). A park the case never reached is a
 * dead fixture and fails as such. See docs/OPS.md. */
#include "rad_abi.h"
#include "rad_runtime.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "r4d.h"
#include "ref_host.h"
#include "hip/hip_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

extern "C" int r4d_rad_embed_lookup_q_init(const RadParam*, int, int, int, void** out);
extern "C" void r4d_rad_embed_lookup_q_fini(void*);

namespace {

/* The gather's geometry (r4d_model_bf16.hip): 160 E4M3 bytes a row, four ways a set. The case
 * turns on four, since three rows must fill a set and a fourth must then be its coldest: the warm
 * launches check that by being byte-identical whether or not they hit. */
constexpr int kWays = 4;
constexpr int kRow = 160;
constexpr int kSets = 32;
constexpr const char* kEntry = "(ple_embed_lookup_q_kernel)";   /* R4D_EMIT stringifies it so */

uint32_t set_of(int id) { return ((uint32_t)id * 2654435761u) & (uint32_t)(kSets - 1); }

/* Deterministic, distinct per (row, byte): a wrong row's payload is then visibly a wrong row. */
uint8_t code_of(int id, int j) {
    uint32_t h = (uint32_t)id * 0x9E3779B9u ^ (uint32_t)j * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2545F491u; h ^= h >> 13;
    return (uint8_t)(h & 0xFF);
}

struct Fx {
    int M = 0;
    int32_t n_vocab = 256;
    std::vector<uint8_t> wte;      /* [n_vocab][kRow] E4M3 */
    std::vector<uint16_t> scb{0};  /* one bf16 scale */
    std::vector<int32_t> tok;
    std::vector<uint16_t> xbuf;    /* 16-byte aligned base, M rows of kRow bf16 */
    uint16_t* x = nullptr;
    /* What the engine says about the table (RadTensor::space). RAD_SPACE_UNKNOWN leaves the plugin
     * to ask the driver, which is the ordinary in-tree case for a table it placed itself. */
    int wte_space = RAD_SPACE_UNKNOWN;

    static_assert(kWays == 4 && kRow == 160, "the constants above are the gather's own");

    void shape(const std::vector<int32_t>& t) {
        M = (int)t.size();
        tok = t;
        /* Filled once: the cache binds the first table it sees and leaves every other alone, so a
         * shape that reallocated the table would take the gathers off the cached path. */
        if (wte.empty()) {
            wte.assign((size_t)n_vocab * kRow, 0);
            for (int id = 0; id < n_vocab; ++id)
                for (int j = 0; j < kRow; ++j) wte[(size_t)id * kRow + j] = code_of(id, j);
        }
        scb[0] = f2b(0.03125f);
        xbuf.assign((size_t)M * kRow + 16, 0);
        x = (uint16_t*)(((uintptr_t)xbuf.data() + 15) & ~(uintptr_t)15);
    }
    void clear_out() { std::memset(x, 0, (size_t)M * kRow * sizeof(uint16_t)); }

    int launch(void* inst, int phase = RAD_PHASE_PREFILL) {
        RadTensor t[4]{};
        t[0].data = tok.data(); t[0].dtype = RAD_I32; t[0].rank = 1;
        t[0].shape[0] = M; t[0].stride[0] = 1;
        t[1].data = wte.data(); t[1].dtype = RAD_F8E4M3; t[1].rank = 2;
        t[1].shape[0] = n_vocab; t[1].shape[1] = kRow;
        t[1].stride[0] = kRow; t[1].stride[1] = 1;
        t[1].space = wte_space;
        t[2].data = scb.data(); t[2].dtype = RAD_BF16; t[2].rank = 1;
        t[2].shape[0] = 1; t[2].stride[0] = 1;
        t[3].data = x; t[3].dtype = RAD_BF16; t[3].rank = 2;
        t[3].shape[0] = M; t[3].shape[1] = kRow;
        t[3].stride[0] = kRow; t[3].stride[1] = 1;
        RadArgs a{};
        a.t = t; a.n_t = 4; a.instance = inst; a.phase = phase;
        return r4d_rad_embed_lookup_q(&a, nullptr);
    }
};

/* The rows `M` against the oracle, entry by entry; names the first wrong one. */
bool same(const Fx& f, const std::vector<uint16_t>& want, const char* what, int* bad_row) {
    for (int m = 0; m < f.M; ++m)
        for (int j = 0; j < kRow; ++j) {
            const size_t o = (size_t)m * kRow + (size_t)j;
            if (f.x[o] != want[o]) {
                if (bad_row) *bad_row = m;
                std::printf("  %s: row %d entry %d: got %04x, want %04x\n", what, m, j,
                            f.x[o], want[o]);
                return false;
            }
        }
    return true;
}

/* A reading of the cache's counters, so a case asserts what its own launches added: the counters
 * are a running total over the instance's whole life. */
struct Snap {
    unsigned long long hits[3]{}, misses[3]{}, bytes[3]{};
    static Snap of(void* inst) {
        Snap s;
        R4dPleCacheStats c{};
        if (r4d_ple_stats(inst, &c) != 0) { s.hits[0] = ~0ull; return s; }
        for (int ph = 0; ph < 3; ++ph) {
            s.hits[ph] = c.hits[ph]; s.misses[ph] = c.misses[ph]; s.bytes[ph] = c.host_bytes[ph];
        }
        return s;
    }
    Snap less(const Snap& b) const {   /* what happened between `b` and here */
        Snap d;
        for (int ph = 0; ph < 3; ++ph) {
            d.hits[ph] = hits[ph] - b.hits[ph]; d.misses[ph] = misses[ph] - b.misses[ph];
            d.bytes[ph] = bytes[ph] - b.bytes[ph];
        }
        return d;
    }
};

std::vector<uint16_t> oracle(Fx& f) {
    f.clear_out();
    if (f.launch(nullptr) != RAD_OK) std::printf("  oracle launch refused\n");
    return std::vector<uint16_t>(f.x, f.x + (size_t)f.M * kRow);
}

/* Everything a call wrote to stderr. The init diagnostic is the observable: a run that meant to
 * measure the cache must be told why it is off, and a run that never asked must stay quiet. */
template <class F>
std::string stderr_of(F fn) {
    std::fflush(stderr);
    FILE* tmp = std::tmpfile();
    const int saved = dup(fileno(stderr));
    dup2(fileno(tmp), fileno(stderr));
    fn();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::rewind(tmp);
    std::string s;
    char buf[512];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, tmp)) > 0) s.append(buf, n);
    std::fclose(tmp);
    return s;
}

}  // namespace

int main() {
    int fails = 0, cases = 0;

    /* Five ids of one set: A the reader, B/C/D the rows that keep it cold, X the evictor. */
    int ids[5] = {0, 0, 0, 0, 0};
    const uint32_t first = set_of(0);
    int n = 0;
    for (int id = 0; id < 4096 && n < 5; ++id) if (set_of(id) == first) ids[n++] = id;
    if (n < 5) { std::printf("ple_cache: no set of five ids\n"); return 1; }
    const int A = ids[0], B = ids[1], C = ids[2], D = ids[3], X = ids[4];

    Fx f;
    f.shape({A, B, C});

    /* The cache is opt-in through the environment, read by init. */
    (void)setenv("RADIANCE_PLE_VRAM_CACHE_SETS", "32", 1);
    void* cache = nullptr;
    const int ri = r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &cache);
    (void)unsetenv("RADIANCE_PLE_VRAM_CACHE_SETS");
    if (ri != RAD_OK || !cache) {
        std::printf("ple_cache: init gave no cache (%d): this case needs the cached path\n", ri);
        return 1;
    }
    std::printf("ple_cache: ids A=%d B=%d C=%d D=%d X=%d, set %u\n", A, B, C, D, X, set_of(A));

    /* Warm: B/C/D once (they claim three ways of the set), again (a hit each), then A, which takes
     * the set's empty way and so is its coldest. Every one of these is checked against the uncached
     * path, so a broken fixture cannot pass into the forced case. */
    for (const std::vector<int32_t>& t : std::vector<std::vector<int32_t>>{
             {B, C, D}, {B, C, D}, {A}}) {
        f.shape(t);
        const std::vector<uint16_t> want = oracle(f);
        f.clear_out();
        const int r = f.launch(cache);
        const bool ok = r == RAD_OK && same(f, want, "warm", nullptr);
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("  warm launch %d rows refused %d or not byte-identical\n", f.M, r);
        }
    }

    /* The forced window. Order: block 0 (row A) is held after its lookup; block 1 (row X) evicts
     * A's way; block 2 (row A) re-claims it and is held between its tag write and its payload
     * write; block 0 is then let go, so its load reads X's payload with the way's tag back to A. */
    f.shape({A, X, A});
    const std::vector<uint16_t> want = oracle(f);
    f.clear_out();
    emu_park_clear();
    const int p0 = emu_park(0, 2);      /* block 0: after `g = __shfl(g, 0)`, before the payload */
    const int p2 = emu_park(2, 3);      /* block 2: after ple_claim's tag, before its payload */
    const int r = f.launch(cache);
    const bool forced = emu_park_hit(p0) && emu_park_hit(p2);
    const bool entry = std::strcmp(emu_last_kernel, kEntry) == 0;
    emu_park_clear();
    ++cases;
    int bad = -1;
    const bool bytes = same(f, want, "forced", &bad);
    const bool ok = r == RAD_OK && forced && entry && bytes;
    if (!bytes)                 /* a wrong row is a wrong row's payload: name it by its entries */
        for (int m = 0; m < f.M; ++m) {
            int differ = 0;
            for (int j = 0; j < kRow; ++j)
                differ += f.x[(size_t)m * kRow + j] != want[(size_t)m * kRow + j];
            if (!differ) continue;
            std::printf("  row %d: %d of %d entries differ; its first 8 got/want:", m, differ, kRow);
            for (int j = 0; j < 8; ++j)
                std::printf(" %04x/%04x", f.x[(size_t)m * kRow + j], want[(size_t)m * kRow + j]);
            std::printf("\n");
        }
    if (!ok) {
        ++fails;
        if (r != RAD_OK) std::printf("  forced launch refused %d\n", r);
        if (!forced) std::printf("  the window was not forced: parks %d/%d\n",
                                 emu_park_hit(p0), emu_park_hit(p2));
        if (!entry) std::printf("  the gather's entry did not run: %s\n", emu_last_kernel);
    }
    std::printf("ple_cache: forced window %s (entry %s, row %d of %d)\n",
                ok ? "byte-identical to the uncached path" : "FAILED", emu_last_kernel,
                bad, f.M);

    /* The counters MAIN-60 needs readable, on the same instance: one set a phase, and the bytes a
     * miss actually fetched over the link rather than a figure derived from a mover. Each case is
     * the delta one launch added, against what the cache's state says it must be. The state is
     * settled here rather than inherited: the forced window above left three of the four ways
     * holding B/C/D and one holding A, whichever way its second claim chose. So each of B/C/D is
     * warmed twice first -- the second launch of a row is its hit -- and X, the row no warm launch
     * has ever published, is the miss.
     *
     * The phases come from RadArgs.phase, the step's, so the gather never guesses one. */
    {
        for (int id : {B, C, D}) {                 /* settle: the way holds this row, published */
            f.shape({id});
            (void)f.launch(cache);
            f.clear_out();
            (void)f.launch(cache);
        }
        struct Case { const char* name; std::vector<int32_t> tok; int phase;
                      unsigned long long h[3], m[3], by[3]; };
        const Case kCases[] = {
            /* decode, one published row: a hit, no link read */
            { "decode hit", {B}, RAD_PHASE_DECODE, {0, 1, 0}, {0, 0, 0}, {0, 0, 0} },
            /* decode, an out-of-range id and a published row: the -1 reads no table and is counted
             * in neither, so the phase shows the one row that reached the check */
            { "decode, -1 id", {-1, C}, RAD_PHASE_DECODE, {0, 1, 0}, {0, 0, 0}, {0, 0, 0} },
            /* mixed, a published row */
            { "mixed hit", {B}, RAD_PHASE_MIXED, {0, 0, 1}, {0, 0, 0}, {0, 0, 0} },
            /* prefill, a hit and an uncached row: one 160-byte link read, not two */
            { "prefill hit+miss", {D, X}, RAD_PHASE_PREFILL, {1, 0, 0}, {1, 0, 0}, {kRow, 0, 0} },
        };
        for (const Case& c : kCases) {
            f.shape(c.tok);
            const Snap before = Snap::of(cache);
            if (before.hits[0] == ~0ull) {
                ++fails;
                std::printf("  counters: r4d_ple_stats found no cache\n");
                break;
            }
            const int r = f.launch(cache, c.phase);
            const Snap d = Snap::of(cache).less(before);
            bool ok = r == RAD_OK;
            for (int ph = 0; ph < 3; ++ph)
                ok = ok && d.hits[ph] == c.h[ph] && d.misses[ph] == c.m[ph] &&
                     d.bytes[ph] == c.by[ph];
            ++cases;
            if (!ok) {
                ++fails;
                std::printf("  counters: %s (launch %d):", c.name, r);
                for (int ph = 0; ph < 3; ++ph)
                    std::printf(" phase %d h/m/b %llu/%llu/%llu want %llu/%llu/%llu", ph,
                                d.hits[ph], d.misses[ph], d.bytes[ph], c.h[ph], c.m[ph],
                                c.by[ph]);
                std::printf("\n");
            }
        }
        /* A launch with no instance takes the uncached path: it must not move the counters. */
        f.shape({B, C});
        const Snap before = Snap::of(cache);
        (void)f.launch(nullptr);
        const Snap d = Snap::of(cache).less(before);
        bool ok = true;
        for (int ph = 0; ph < 3; ++ph)
            ok = ok && !d.hits[ph] && !d.misses[ph] && !d.bytes[ph];
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("  counters: the uncached path moved the cache's counters\n");
        }
    }

    /* Off has to be said, and only when it was asked for: an explicit 0 or a value that is not a
     * number allocates nothing and prints why, so a run that meant to measure the cache cannot read
     * the tables' traffic as its misses. Unset is the default and stays silent. */
    {
        bool off_ok   = false;
        const std::string said = stderr_of([&] {
            (void)setenv("RADIANCE_PLE_VRAM_CACHE_SETS", "0", 1);
            void* off = nullptr;
            const int r = r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &off);
            off_ok = r == RAD_OK && off == nullptr;
            (void)unsetenv("RADIANCE_PLE_VRAM_CACHE_SETS");
        });
        const std::string quiet = stderr_of([] {
            void* off = nullptr;
            (void)r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &off);
        });
        const bool named = said.find("RADIANCE_PLE_VRAM_CACHE_SETS") != std::string::npos;
        ++cases;
        if (!off_ok || !named || !quiet.empty()) {
            ++fails;
            std::printf("  cache off: instance %s, explicit 0 %s, unset said %zu byte(s)\n",
                        off_ok ? "none" : "HANDED OUT", named ? "named" : "SILENT",
                        quiet.size());
        }
    }

    /* Off is the generic kernel, and it has to be the one that runs: every row read from the
     * table, nothing counted. The table here holds exactly two E4M3 codes with exact values --
     * 0x38 is 1.0, 0x00 is zero -- at scale 1.0, so the expectation is hand-computed instead of
     * being another launch of the same op: 160 entries of 1.0 (bf16 0x3f80) for the ones row and
     * 160 zeros for the zeros row. The host read volume of this path is one kPleRow a valid id,
     * 2 x 160 = 320 bytes here -- arithmetic, not a counter: nothing on the host observes a load,
     * which is why the cache counts at the load site when it is on (docs/OPS.md). */
    {
        const int one_id = 7, zero_id = 9;
        std::vector<uint8_t> wte2((size_t)32 * kRow, 0);
        std::memset(&wte2[(size_t)one_id * kRow], 0x38, kRow);
        std::vector<uint16_t> sc2{f2b(1.0f)};
        std::vector<int32_t> tok2{one_id, zero_id};
        std::vector<uint16_t> xb((size_t)tok2.size() * kRow + 16, 0xdead);
        uint16_t* xo = (uint16_t*)(((uintptr_t)xb.data() + 15) & ~(uintptr_t)15);
        RadTensor t[4]{};
        t[0].data = tok2.data(); t[0].dtype = RAD_I32; t[0].rank = 1;
        t[0].shape[0] = (int64_t)tok2.size(); t[0].stride[0] = 1;
        t[1].data = wte2.data(); t[1].dtype = RAD_F8E4M3; t[1].rank = 2;
        t[1].shape[0] = 32; t[1].shape[1] = kRow;
        t[1].stride[0] = kRow; t[1].stride[1] = 1;
        t[1].space = RAD_SPACE_HOST;
        t[2].data = sc2.data(); t[2].dtype = RAD_BF16; t[2].rank = 1;
        t[2].shape[0] = 1; t[2].stride[0] = 1;
        t[3].data = xo; t[3].dtype = RAD_BF16; t[3].rank = 2;
        t[3].shape[0] = (int64_t)tok2.size(); t[3].shape[1] = kRow;
        t[3].stride[0] = kRow; t[3].stride[1] = 1;
        RadArgs a{};
        a.t = t; a.n_t = 4; a.phase = RAD_PHASE_PREFILL;

        void* off = nullptr;
        (void)setenv("RADIANCE_PLE_VRAM_CACHE_SETS", "0", 1);
        const int r0 = r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &off);
        (void)unsetenv("RADIANCE_PLE_VRAM_CACHE_SETS");
        const int  r = r4d_rad_embed_lookup_q(&a, nullptr);
        R4dPleCacheStats none{};
        const bool counted = r4d_ple_stats(off, &none) == 0;   /* there is nothing to count */
        int bad = -1;
        for (int m = 0; m < 2 && bad < 0; ++m)
            for (int j = 0; j < kRow; ++j)
                if (xo[(size_t)m * kRow + (size_t)j] != (m == 0 ? 0x3f80 : 0x0000)) {
                    std::printf("  cache off: row %d entry %d: got %04x, want %04x\n", m, j,
                                xo[(size_t)m * kRow + (size_t)j], m == 0 ? 0x3f80 : 0x0000);
                    bad = m;
                    break;
                }
        ++cases;
        if (r0 != RAD_OK || off != nullptr || r != RAD_OK || counted || bad >= 0) {
            ++fails;
            std::printf("  cache off: init %d instance %s, launch %d, counters %s, rows %s\n", r0,
                        off ? "HANDED OUT" : "none", r, counted ? "MOVED" : "none",
                        bad < 0 ? "as computed" : "NOT as computed");
        }
    }

    /* A table in VRAM: a miss reads no host bytes, which is the whole of the link accounting. The
     * emulator's allocations read as host, so the range is marked device -- what
     * hipPointerGetAttributes reports for a table the engine placed in VRAM -- and a second cache
     * instance is asked, binding being per instance. The counters are the same, so a zero here is
     * the table's residence and not a dead statistic. */
    {
        R4dPleCacheStats host_side{};
        bool ok = r4d_ple_stats(cache, &host_side) == 0 && host_side.table_host == 1;
        f.shape({X});
        emu_mark_device(f.wte.data(), f.wte.size());
        (void)setenv("RADIANCE_PLE_VRAM_CACHE_SETS", "32", 1);
        void* vram = nullptr;
        const int ri2 = r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &vram);
        (void)unsetenv("RADIANCE_PLE_VRAM_CACHE_SETS");
        ok = ok && ri2 == RAD_OK && vram != nullptr;
        if (ok) {
            const Snap before = Snap::of(vram);
            f.clear_out();
            const int lr = f.launch(vram, RAD_PHASE_PREFILL);
            const Snap d  = Snap::of(vram).less(before);
            R4dPleCacheStats cs{};
            const bool stats = r4d_ple_stats(vram, &cs) == 0;
            const int64_t mem = r4d_ple_mem_bytes(vram);
            ok = lr == RAD_OK && d.misses[0] == 1 && d.hits[0] == 0 && d.bytes[0] == 0 &&
                 stats && cs.table_host == 0 && mem > 0;
            if (!ok)
                std::printf("  vram table (launch %d): h/m/b %llu/%llu/%llu, table_host %d, %lld "
                            "byte(s) reserved\n", lr, d.hits[0], d.misses[0], d.bytes[0],
                            cs.table_host, (long long)mem);
            r4d_rad_embed_lookup_q_fini(vram);
        }
        emu_clear_devices();
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("  the table in VRAM did not read as a device-resident table\n");
        }
    }

    /* The table the backend pinned: hipPointerGetAttributes cannot classify memory aql locked
     * into its host pool (core/device/aql.cpp keeps those ranges itself and the driver reports
     * UNKNOWN), so a plugin that fell back to "device" would count the miss and report that its
     * 160 bytes crossed no link. The engine knows where it placed the table and says so on the
     * operand; that answer is asked first and wins. */
    {
        f.shape({X});
        f.wte_space = RAD_SPACE_HOST;
        emu_mark_unknown(f.wte.data(), f.wte.size());
        (void)setenv("RADIANCE_PLE_VRAM_CACHE_SETS", "32", 1);
        void* pin = nullptr;
        const int ri3 = r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &pin);
        (void)unsetenv("RADIANCE_PLE_VRAM_CACHE_SETS");
        bool ok = ri3 == RAD_OK && pin != nullptr;
        if (ok) {
            const Snap before = Snap::of(pin);
            f.clear_out();
            const int lr = f.launch(pin, RAD_PHASE_PREFILL);
            const Snap d  = Snap::of(pin).less(before);
            R4dPleCacheStats cs{};
            const bool stats = r4d_ple_stats(pin, &cs) == 0;
            ok = lr == RAD_OK && d.misses[0] == 1 && d.hits[0] == 0 && d.bytes[0] == (unsigned)kRow &&
                 stats && cs.table_host == 1;
            if (!ok)
                std::printf("  pinned table (launch %d): h/m/b %llu/%llu/%llu want 0/1/%d, "
                            "table_host %d\n", lr, d.hits[0], d.misses[0], d.bytes[0], kRow,
                            cs.table_host);
            r4d_rad_embed_lookup_q_fini(pin);
        }
        emu_clear_devices();
        f.wte_space = RAD_SPACE_UNKNOWN;
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("  a table the backend pinned into host memory did not read as host\n");
        }
    }

    /* And when neither the engine nor the driver can say, the byte column is unmeasured, not 0:
     * the misses stand, the dump prints the figure as na and names the table unknown, so a reader
     * cannot mistake "not taken" for "nothing crossed the link". */
    {
        f.shape({X});
        f.wte_space = RAD_SPACE_UNKNOWN;
        emu_mark_unknown(f.wte.data(), f.wte.size());
        (void)setenv("RADIANCE_PLE_VRAM_CACHE_SETS", "32", 1);
        void* unk = nullptr;
        const int ri4 = r4d_rad_embed_lookup_q_init(nullptr, 0, 0, 1, &unk);
        (void)unsetenv("RADIANCE_PLE_VRAM_CACHE_SETS");
        bool ok = ri4 == RAD_OK && unk != nullptr;
        if (ok) {
            const Snap before = Snap::of(unk);
            f.clear_out();
            const int lr = f.launch(unk, RAD_PHASE_PREFILL);
            const Snap d  = Snap::of(unk).less(before);
            R4dPleCacheStats cs{};
            const bool stats = r4d_ple_stats(unk, &cs) == 0;
            const std::string said = stderr_of([&] { r4d_ple_stats_dump(unk, nullptr, 0); });
            const bool na = said.find("host_read_bytes=na") != std::string::npos &&
                            said.find("table=unknown") != std::string::npos;
            ok = lr == RAD_OK && d.misses[0] == 1 && d.hits[0] == 0 && d.bytes[0] == 0 && stats &&
                 cs.table_host == -1 && na;
            if (!ok)
                std::printf("  unknown table (launch %d): h/m/b %llu/%llu/%llu, table_host %d, "
                            "dump %s\n", lr, d.hits[0], d.misses[0], d.bytes[0], cs.table_host,
                            na ? "says na" : "DOES NOT SAY na");
            r4d_rad_embed_lookup_q_fini(unk);
        }
        emu_clear_devices();
        ++cases;
        if (!ok) {
            ++fails;
            std::printf("  an unclassifiable table reported its byte count as measured 0\n");
        }
    }

    r4d_rad_embed_lookup_q_fini(cache);
    std::printf("ple_cache: %d case(s), %d failure(s)\n", cases, fails);
    return fails != 0;
}
