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
#include "rad_plugin.h"
#include "r4d_plugin.h"
#include "ref_host.h"
#include "hip/hip_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    static_assert(kWays == 4 && kRow == 160, "the constants above are the gather's own");

    void shape(const std::vector<int32_t>& t) {
        M = (int)t.size();
        tok = t;
        wte.assign((size_t)n_vocab * kRow, 0);
        for (int id = 0; id < n_vocab; ++id)
            for (int j = 0; j < kRow; ++j) wte[(size_t)id * kRow + j] = code_of(id, j);
        scb[0] = f2b(0.03125f);
        xbuf.assign((size_t)M * kRow + 16, 0);
        x = (uint16_t*)(((uintptr_t)xbuf.data() + 15) & ~(uintptr_t)15);
    }
    void clear_out() { std::memset(x, 0, (size_t)M * kRow * sizeof(uint16_t)); }

    int launch(void* inst) {
        RadTensor t[4]{};
        t[0].data = tok.data(); t[0].dtype = RAD_I32; t[0].rank = 1;
        t[0].shape[0] = M; t[0].stride[0] = 1;
        t[1].data = wte.data(); t[1].dtype = RAD_F8E4M3; t[1].rank = 2;
        t[1].shape[0] = n_vocab; t[1].shape[1] = kRow;
        t[1].stride[0] = kRow; t[1].stride[1] = 1;
        t[2].data = scb.data(); t[2].dtype = RAD_BF16; t[2].rank = 1;
        t[2].shape[0] = 1; t[2].stride[0] = 1;
        t[3].data = x; t[3].dtype = RAD_BF16; t[3].rank = 2;
        t[3].shape[0] = M; t[3].shape[1] = kRow;
        t[3].stride[0] = kRow; t[3].stride[1] = 1;
        RadArgs a{};
        a.t = t; a.n_t = 4; a.instance = inst;
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

std::vector<uint16_t> oracle(Fx& f) {
    f.clear_out();
    if (f.launch(nullptr) != RAD_OK) std::printf("  oracle launch refused\n");
    return std::vector<uint16_t>(f.x, f.x + (size_t)f.M * kRow);
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

    r4d_rad_embed_lookup_q_fini(cache);
    std::printf("ple_cache: %d case(s), %d failure(s)\n", cases, fails);
    return fails != 0;
}
