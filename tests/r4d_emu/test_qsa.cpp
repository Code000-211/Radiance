/* tests/r4d_emu: the QSA block selection (r4d_qsa_select_bf16.hip) on the host SIMT emulator, each
 * of its four kernels against a port of libref's ref_qsa_select, byte for byte: the page-id table
 * (sentinel-filled before every launch, the row pitch's padding and a row past M included), nsel
 * and seqused.
 *
 * Every fixture runs three ways: pinned to the single-workgroup kernel and pinned to the wide one,
 * both without scratch, and the launcher's own choice with the split path's scratch given (the
 * local + merge pair for a decode-shaped launch over several chunks, the wide kernel above 64 rows,
 * else the single-workgroup one). The kernel that ran is checked by name.
 *
 * The fixtures follow r4d_selftest's qsa_select cases at sizes the emulator runs in seconds, plus a
 * seeded sweep of small odd shapes (NaN, +-inf, -0.0, denormal and -1e30 scores, pitches, a table
 * narrower than the scores, rows no sequence holds, topk 1, ratio 1). Which branch a row takes
 * inside the kernels is predicted from their constants, restated in `Path`; the run fails when the
 * fixtures stop reaching one of them. `sweep N` runs N seeded fixtures at serving sizes instead.
 *
 * One case is not a kernel against the port: `shared_table_case` feeds the kernel's own `selb` and
 * `nsel` for a tile of query rows to plugin/qsatable's `qsa_table` and checks that the union in block
 * order, filtered by a row's mask and followed by that row's partial page, is that row's own table
 * byte for byte (docs/QSA.md, "Table order"). The op's host reference is run on the same operands and
 * compared, so the case holds the kernel to the definition of the op too.
 *
 * Two cases hold the table path to what it replaces. `masked_read_case` is the read: a per-row table
 * (tile 1) and the tile's union (tile 16) give the same bytes for the same rows, and both are held
 * to libref's `attn_paged` -- the attention the architecture issues today, loaded from
 * $RADIANCE_HOME -- byte for byte as well, over 8K and 32K histories, draft depth 3 and 6, rows with
 * and without a partial page, ties at a coarse score grid and two prefill chunks. That comparison is
 * required, not opportunistic: a build whose `$RADIANCE_HOME/kernels/libref.so` is missing, refuses
 * to open, or declares no `attn_paged` fails the case with the reason, because a run without the
 * opponent would be a green run of half the claim. `parser_case` is the
 * batch: two sequences in one step, a declared capacity of 256 sequences and a block table with the
 * batch's two rows, so the parser's row check is held to the live sequence count rather than to the
 * capacity, and the same rows as a chunk show that `row0` keeps them in their own sequence.
 *
 * Fixtures marked signed run with the op's `signed` parameter: the kernels' signed key order,
 * against the port's signed ranking, over negative, mixed and the sweep's odd scores. The unsigned
 * fixtures are the same as without the parameter. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "qsaattn.h"
#include "qsatable.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

extern "C" void r4d_qsa_select_pin_form(int form);
extern int emu_grids;
extern const char* emu_last_kernel;

namespace {

constexpr int32_t kSentinel = 0x55555555;

struct Fix {
    std::string name;
    int S = 1, M = 1, nb_s = 1, nb_t = 1;   /* sequences, query rows, score and table widths */
    int s_ld = 0, t_ld = 0, o_ld = 0;       /* row pitches; 0 = tight */
    int topk = 1, ratio = 1;
    bool sgn = false;                       /* the op's `signed` parameter */
    std::vector<float> sc;                  /* [M][s_ld] */
    std::vector<int32_t> bt;                /* [S][t_ld] */
    std::vector<int32_t> nc, pos, cu;       /* [S], [M], [S + 1] */
    void shape(int s, int m, int nbs, int nbt, int tk, int r, int spad = 0, int tpad = 0, int opad = 0) {
        S = s; M = m; nb_s = nbs; nb_t = nbt; topk = tk; ratio = r;
        s_ld = nbs + spad; t_ld = nbt + tpad; o_ld = tk + 1 + opad;
        sc.assign((size_t)M * s_ld, -INFINITY);
        bt.assign((size_t)S * t_ld, -1);
        nc.assign((size_t)S, 0); pos.assign((size_t)M, 0); cu.assign((size_t)S + 1, 0);
    }
    float& score(int m, int b) { return sc[(size_t)m * s_ld + b]; }
    int32_t& page(int s, int b) { return bt[(size_t)s * t_ld + b]; }
    int nb() const { return std::min(nb_s, nb_t); }
    int seq_of(int m) const {
        for (int s = 0; s < S; ++s) if (m >= cu[s] && m < cu[s + 1]) return s;
        return 0;
    }
    int n_of(int m) const {
        const int s = seq_of(m);
        return (int)std::min<int64_t>(std::min<int64_t>(nc[s], nb()), ((int64_t)pos[m] + 1) / ratio);
    }
};

/* ---- libref's ref_qsa_select (libref/ref_qsa.cpp), restated over the fixture's arrays: the same
 * steps in the same order, both rankings. */
void ref_select(const Fix& f, std::vector<int32_t>& se, std::vector<int32_t>& ns,
                std::vector<int32_t>& su, std::vector<int32_t>& sb) {
    const int64_t topk = f.topk, ratio = f.ratio;
    se.assign((size_t)f.M * (topk + 1), 0); ns.assign((size_t)f.M, 0); su.assign((size_t)f.M, 0);
    sb.assign((size_t)f.M * (topk + 1), 0);
    for (int64_t m = 0; m < f.M; ++m) {
        const int64_t s = f.seq_of((int)m), p = f.pos[m];
        const int64_t n = std::min(std::min<int64_t>(f.nc[s], f.nb()), (p + 1) / ratio);
        std::vector<float> val((size_t)n);
        std::vector<int64_t> live;
        for (int64_t b = 0; b < n; ++b) {
            const float v = f.sc[(size_t)(m * f.s_ld + b)];
            if (f.bt[(size_t)(s * f.t_ld + b)] < 0) continue;
            if (f.sgn ? !std::isfinite(v) : !(v > -1e30f)) continue;
            val[(size_t)b] = f.sgn ? v : v > 0.0f ? v : 0.0f;
            live.push_back(b);
        }
        std::vector<int64_t> take;
        if ((int64_t)live.size() <= topk) {
            take = live;
        } else {
            std::vector<float> ord;
            for (int64_t b : live) ord.push_back(val[(size_t)b]);
            std::nth_element(ord.begin(), ord.begin() + (topk - 1), ord.end(), std::greater<float>());
            const float thr = ord[(size_t)(topk - 1)];
            /* The selection in block order, above the threshold and the ties alike: the ties are
             * the ones the top-k leaves slots for. */
            int64_t above = 0;
            for (int64_t b : live) above += val[(size_t)b] > thr;
            for (int64_t need = topk - above; int64_t b : live) {
                if (val[(size_t)b] > thr) take.push_back(b);
                else if (need > 0 && val[(size_t)b] == thr) { take.push_back(b); --need; }
            }
        }
        int32_t* row = &se[(size_t)(m * (topk + 1))];
        int32_t* brow = &sb[(size_t)(m * (topk + 1))];
        int64_t out = 0;
        for (int64_t b : take) { row[out] = f.bt[(size_t)(s * f.t_ld + b)]; brow[out++] = (int32_t)b; }
        if (n < f.nb() && (p + 1) % ratio != 0) {
            const int32_t part = f.bt[(size_t)(s * f.t_ld + n)];
            if (part >= 0) { row[out] = part; brow[out++] = (int32_t)n; }
        }
        for (int64_t i = out; i <= topk; ++i) { row[i] = -1; brow[i] = -1; }
        ns[(size_t)m] = (int32_t)out;
        su[(size_t)m] = (int32_t)(out == 0 ? 0 : (out - 1) * ratio + p % ratio + 1);
    }
}

/* ---- Which branch a row takes inside the kernels: the launcher's and kernels' constants
 * (r4d_qsa_select_bf16.hip) on a 64 KiB-LDS device, and their branch conditions. Coverage only: the
 * tables are checked against the port above whatever the prediction says. */
namespace Path {
constexpr int kCacheCap = 8192, kCacheMaxM = 64;                    /* the key cache */
constexpr int kSamp = 2048, kSampRun = 32, kSampRuns = 64, kSampMinN = 3 * kSamp, kSampCap = 1920;
constexpr int kWideShift = 19, kWideCand = 3840, kMergeAbove = 512, kMergeCand = 1664;
constexpr int kWideShiftSigned = 20;                                /* the signed key's 32 bits */
int wide_shift(bool sgn) { return sgn ? kWideShiftSigned : kWideShift; }
constexpr int kParM = 32, kParWMax = 16, kParCap = 8192;            /* the split path */
int par_chunks(int nb) { return std::max(1, std::min(kParWMax, (nb + 4095) / 4096)); }
}

uint32_t keyof(float v, int32_t page) {
    if (page < 0 || !(v > -1e30f)) return 0u;
    const float c = v > 0.0f ? v : 0.0f;
    uint32_t u;
    std::memcpy(&u, &c, 4);
    return u + 1u;
}

/* The signed order: the float's total order over 32 bits, -0 keyed as +0, nonfinite dead. */
uint32_t keyof_signed(float v, int32_t page) {
    if (page < 0 || !std::isfinite(v)) return 0u;
    uint32_t u = 0;
    if (v != 0.0f) std::memcpy(&u, &v, 4);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

enum Form { kSingle, kWide, kAuto };
const char* const kFormName[] = { "single", "wide", "auto" };

bool split_applies(const Fix& f) {
    const int nch = Path::par_chunks(f.nb()), chunk = (f.nb() + nch - 1) / nch;
    return f.M <= Path::kParM && nch > 1 && chunk <= Path::kParCap && nch * f.topk <= Path::kParCap;
}

std::map<std::string, int> g_paths;

/* The branches row m reaches under `form`. */
void note_paths(const Fix& f, int m, Form form, bool split) {
    const int s = f.seq_of(m), n = f.n_of(m), topk = f.topk;
    std::vector<uint32_t> k((size_t)std::max(n, 0));
    for (int b = 0; b < n; ++b) {
        const float v = f.sc[(size_t)m * f.s_ld + b];
        const int32_t pg = f.bt[(size_t)s * f.t_ld + b];
        k[(size_t)b] = f.sgn ? keyof_signed(v, pg) : keyof(v, pg);
    }
    std::vector<uint32_t> live;
    for (uint32_t x : k) if (x) live.push_back(x);
    std::sort(live.begin(), live.end(), std::greater<uint32_t>());
    const int L = (int)live.size();
    auto hit = [](const char* p) { ++g_paths[p]; };
    if (n <= topk) hit("rows of at most topk blocks");
    else if (L == topk) hit("exactly topk live over more blocks");
    else if (L < topk) hit("fewer than topk live over more blocks");
    else {
        hit("a threshold");
        const uint32_t T = live[(size_t)topk - 1];
        int above = 0, eq = 0;
        for (uint32_t x : live) { above += x > T; eq += x == T; }
        if (eq > topk - above) hit("a threshold whose ties are cut short");
        if (f.sgn && T < 0x80000000u) hit("signed: a negative threshold");
    }
    if (split) {
        const int nch = Path::par_chunks(f.nb()), chunk = (f.nb() + nch - 1) / nch;
        int tot = 0;
        bool over = false;
        for (int c = 0; c < nch && c * chunk < n; ++c) {
            const int lo = c * chunk, hi = std::min(lo + chunk, n);
            int lc = 0;
            for (int b = lo; b < hi; ++b) lc += k[(size_t)b] != 0;
            over |= hi - lo > topk && lc > topk;
            tot += std::min(lc, topk);
        }
        if (n <= topk || (!over && tot <= topk)) hit("split: topk or fewer live, every one taken");
        else if (!over) hit("split: no chunk over topk, more than topk live");
        else if (tot == topk) hit("split: exactly topk candidates, from a chunk over topk");
        else hit("split: a chunk over topk");
        return;
    }
    if (form == kSingle || (form == kAuto && f.M <= Path::kCacheMaxM)) {
        const int cap = f.M <= Path::kCacheMaxM ? std::min(Path::kCacheCap, f.nb()) : 0;
        hit(n <= cap ? "single: keys cached in LDS" : "single: keys read from global memory");
        return;
    }
    if (n <= topk) return;
    if (n >= Path::kSampMinN && topk <= Path::kMergeAbove) {
        std::vector<uint32_t> smp;
        for (int i = 0; i < Path::kSamp; ++i) {
            const long b = (long)(i / Path::kSampRun) * (n - Path::kSampRun) / (Path::kSampRuns - 1) + i % Path::kSampRun;
            if (k[(size_t)b]) smp.push_back(k[(size_t)b]);
        }
        std::sort(smp.begin(), smp.end(), std::greater<uint32_t>());
        const uint32_t ms = std::min<uint32_t>(
            (uint32_t)(((uint64_t)topk * 2u * Path::kSamp + (uint64_t)n - 1u) / (uint64_t)n) + 8u, Path::kSamp);
        const uint32_t lo = smp.size() > ms ? smp[ms - 1] : 0u;
        int cnt = 0;
        for (uint32_t x : live) cnt += x >= lo;
        const bool vec = (((uintptr_t)&f.sc[(size_t)m * f.s_ld] | (uintptr_t)&f.bt[(size_t)s * f.t_ld]) & 15u) == 0;
        hit(vec ? "wide: the sampled pass, rows 16-byte aligned" : "wide: the sampled pass, rows not 16-byte aligned");
        if (cnt > topk && cnt <= Path::kSampCap) {
            hit(lo ? "wide: the sampled list" : "wide: the sampled list, holding every live key");
            return;
        }
        hit("wide: a sample that fell back");
    }
    if (L <= topk) return;
    const int shift = Path::wide_shift(f.sgn);
    const uint32_t B = live[(size_t)topk - 1] >> shift;
    int in = 0;
    for (uint32_t x : live) in += (x >> shift) == B;
    hit(in <= Path::kMergeCand && topk <= Path::kMergeAbove ? "wide: merged lists"
        : in <= Path::kWideCand ? "wide: the threshold's bin in LDS" : "wide: the threshold's bin over global memory");
}

RadTensor tens(void* p, uint32_t dt, std::vector<int64_t> shape, int64_t ld = 0) {
    RadTensor t{}; t.data = p; t.dtype = dt; t.rank = (uint32_t)shape.size();
    int64_t st = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) { t.shape[i] = shape[i]; t.stride[i] = st; st *= shape[i]; }
    if (ld && t.rank == 2) t.stride[0] = ld;
    return t;
}

/* What a launch wrote, when the caller wants it (the shared-table case keeps the kernel's own
 * table, not the port's). */
struct SelOut {
    std::vector<int32_t> se, ns, su, sb;   /* sb: the same entries as block indices */
};

/* One launch of `form` against the port. Returns the failures (0 or 1). */
int run(Fix& f, Form form, SelOut* out = nullptr) {
    std::vector<int32_t> want, want_n, want_su, want_sb;
    ref_select(f, want, want_n, want_su, want_sb);
    const int M = f.M, W = f.topk + 1;
    std::vector<int32_t> se((size_t)(M + 1) * f.o_ld, kSentinel), ns((size_t)M + 1, kSentinel), su((size_t)M + 1, kSentinel);
    std::vector<int32_t> sb((size_t)(M + 1) * f.o_ld, kSentinel);
    RadTensor t[9] = {
        tens(f.sc.data(), RAD_F32, {M, f.nb_s}, f.s_ld), tens(f.bt.data(), RAD_I32, {f.S, f.nb_t}, f.t_ld),
        tens(f.nc.data(), RAD_I32, {f.S}),             tens(f.pos.data(), RAD_I32, {M}),
        tens(f.cu.data(), RAD_I32, {f.S + 1}),         tens(se.data(), RAD_I32, {M, W}, f.o_ld),
        tens(ns.data(), RAD_I32, {M}),                 tens(su.data(), RAD_I32, {M}),
        tens(sb.data(), RAD_I32, {M, W}, f.o_ld),
    };
    RadParam p[6]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"blocks", RAD_P_INT, f.nb()}; p[2] = {"topk", RAD_P_INT, f.topk};
    p[3] = {"ratio", RAD_P_INT, f.ratio}; p[4] = {"seqs", RAD_P_INT, f.S}; p[5] = {"signed", RAD_P_INT, 1};
    RadArgs a{}; a.t = t; a.n_t = 9; a.p = p; a.n_p = f.sgn ? 6 : 5;
    const int64_t need = r4d_rad_qsa_select_scratch(&a);
    std::vector<uint32_t> scratch((size_t)(need / 4 + 1), 0xA5A5A5A5u);
    if (form == kAuto) { a.scratch = scratch.data(); a.scratch_bytes = need; }
    r4d_qsa_select_pin_form(form == kSingle ? 1 : form == kWide ? 2 : 0);
    const int g0 = emu_grids;
    const int rc = r4d_rad_qsa_select(&a, nullptr);
    r4d_qsa_select_pin_form(0);
    const int grids = emu_grids - g0;
    const bool split = form == kAuto && split_applies(f);
    const char* expect = split ? "qsa_merge_kernel"
                       : form == kWide || (form == kAuto && f.M > Path::kCacheMaxM) ? "qsa_select_wide_kernel"
                       : "qsa_select_kernel";
    const char* fn = kFormName[form];
    if (rc != RAD_OK) { std::printf("FAIL %-48s %-6s launch returned %d\n", f.name.c_str(), fn, rc); return 1; }
    /* The launch site names the template, `kernel<Signed>`: the name is what precedes it. */
    const std::string ran(emu_last_kernel, std::strcspn(emu_last_kernel, "<"));
    if (ran != expect || grids != (split ? 2 : 1)) {
        std::printf("FAIL %-48s %-6s ran %s in %d grid(s), expected %s in %d\n", f.name.c_str(), fn,
                    emu_last_kernel, grids, expect, split ? 2 : 1);
        return 1;
    }
    int bad_rows = 0;
    for (int m = 0; m <= M; ++m) {
        if (m < M) note_paths(f, m, form, split);
        int diff = 0, first = -1;
        for (int i = 0; i < f.o_ld; ++i) {
            const int32_t w = m < M && i < W ? want[(size_t)m * W + i] : kSentinel;
            if (se[(size_t)m * f.o_ld + i] != w) { ++diff; if (first < 0) first = i; }
        }
        int bdiff = 0, bfirst = -1;
        for (int i = 0; i < f.o_ld; ++i) {
            const int32_t w = m < M && i < W ? want_sb[(size_t)m * W + i] : kSentinel;
            if (sb[(size_t)m * f.o_ld + i] != w) { ++bdiff; if (bfirst < 0) bfirst = i; }
        }
        const int32_t wn = m < M ? want_n[(size_t)m] : kSentinel, wsu = m < M ? want_su[(size_t)m] : kSentinel;
        if (!diff && !bdiff && ns[(size_t)m] == wn && su[(size_t)m] == wsu) continue;
        if (++bad_rows > 4) continue;
        std::printf("FAIL %-48s %-6s row %d (n %d): nsel %d want %d, seqused %d want %d", f.name.c_str(), fn, m,
                    m < M ? f.n_of(m) : -1, ns[(size_t)m], wn, su[(size_t)m], wsu);
        if (diff) {
            const int32_t w = m < M && first < W ? want[(size_t)m * W + first] : kSentinel;
            std::printf(", %d slot(s) differ, first slot %d: %d want %d", diff, first, se[(size_t)m * f.o_ld + first], w);
        }
        if (bdiff) {
            const int32_t w = m < M && bfirst < W ? want_sb[(size_t)m * W + bfirst] : kSentinel;
            std::printf(", %d block index(es) differ, first slot %d: %d want %d", bdiff, bfirst,
                        sb[(size_t)m * f.o_ld + bfirst], w);
        }
        std::printf("\n");
    }
    if (out) {
        out->se.assign(se.begin(), se.begin() + (size_t)M * f.o_ld);
        out->ns.assign(ns.begin(), ns.begin() + M);
        out->su.assign(su.begin(), su.begin() + M);
        out->sb.assign(sb.begin(), sb.begin() + (size_t)M * f.o_ld);
    }
    return bad_rows ? 1 : 0;
}

uint64_t g_x = 0;
uint64_t rnd() { g_x ^= g_x << 13; g_x ^= g_x >> 7; g_x ^= g_x << 17; return g_x; }

/* r4d_selftest's qsa_select_case: decode rows, ties on a coarse grid, a dead page with a winning
 * score, a tail on alternate rows; rows 0 and 1 have at most topk live. */
Fix fix_decode() {
    Fix f; f.name = "decode: ties, a tail and a dead page";
    const int M = 6, nb = 900, topk = 64, pages = 4096, ratio = 4;
    f.shape(M, M, nb, nb, topk, ratio);
    std::fill(f.sc.begin(), f.sc.end(), 0.0f);
    for (int m = 0; m < M; ++m) {
        const int live = m == 0 ? 9 : m == 1 ? topk : 100 + m * 137;
        f.nc[m] = live; f.cu[m + 1] = m + 1;
        for (int b = 0; b < nb; ++b) {
            if (b < live) {
                f.score(m, b) = (float)((b * 17 + m * 7) % 40) * 0.25f;
                f.page(m, b) = (b * 31 + m * 11) % pages;
                if (b == live / 3) { f.score(m, b) = 1e6f; f.page(m, b) = -1; }
            } else if (b == live) {
                f.page(m, b) = m % 2 == 0 ? (m * 97 + 5) % pages : -1;
            }
        }
        f.pos[m] = f.page(m, live) >= 0 ? live * ratio + m % 3 : live * ratio - 1;
    }
    return f;
}

/* r4d_selftest's qsa_select_prefill_case: two chunks over two histories, each query bounded by
 * its own position, the partial page present exactly when (pos + 1) % ratio != 0. */
Fix fix_prefill() {
    Fix f; f.name = "prefill: per-query bound and the partial page";
    const int S = 2, nb = 96, topk = 5, pages = 512, ratio = 4, q0 = 9, q1 = 14, M = q0 + q1;
    const int base[2] = { 137, 4 };
    f.shape(S, M, nb, nb, topk, ratio);
    f.cu = { 0, q0, M };
    for (int i = 0; i < q0; ++i) f.pos[i] = base[0] + i;
    for (int i = 0; i < q1; ++i) f.pos[q0 + i] = base[1] + i;
    f.nc = { (base[0] + q0) / ratio, (base[1] + q1) / ratio };
    for (int s = 0; s < S; ++s)
        for (int b = 0; b < nb; ++b)
            f.page(s, b) = b <= (base[s] + (s == 0 ? q0 : q1) - 1) / ratio ? (b * 29 + s * 7) % pages : -1;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 23 + m * 5) % 37) * 0.5f;
    return f;
}

/* r4d_selftest's qsa_select_wide_case at emulator size: the first sequence's rows see about 3000,
 * 5000, 7000 and 8800 blocks (the two-pass form below the sample's 6144, the sampled form above,
 * the key cache's 8192 crossed), the second's fewer than topk; dead pages and -inf scattered; five
 * score distributions. Over 4096 blocks at six rows, the automatic choice is the split path. */
Fix fix_wide(int dist) {
    static const char* const kDist[] = { "continuous", "coarse, heavy ties", "an eighth in one first-digit bin",
                                         "all in one first-digit bin", "fine ties at the threshold",
                                         "signed, heavy ties", "signed, all negative, heavy ties",
                                         "signed, continuous across zero" };
    Fix f; f.name = std::string("wide: ") + kDist[dist];
    f.sgn = dist >= 5;
    const int S = 2, nb = 9000, topk = 512, ratio = 4, pages = 1 << 20;
    f.shape(S, 6, nb, nb, topk, ratio);
    f.cu = { 0, 4, 6 };
    f.pos = { 4 * 3000 + 1, 4 * 5000 + 2, 4 * 7000 + 3, 4 * 8800 + 1, 4 * 300, 4 * 337 + 2 };
    f.nc = { 8900, 340 };
    g_x = 0x243F6A8885A308D3ull + (uint64_t)dist;
    for (auto& v : f.bt) v = rnd() % 97 == 0 ? -1 : (int32_t)(rnd() % pages);
    for (auto& v : f.sc) {
        const uint64_t u = rnd();
        if (u % 211 == 0) { v = -INFINITY; continue; }
        switch (dist) {
            case 0: v = (float)(u % 1000003) / 1000.0f; break;
            case 1: v = (float)(u % 23) * 0.25f; break;
            case 2: v = u % 8 == 0 ? 1.0f + (float)(u % 60000) * 1e-6f : (float)(u % 1000) / 2000.0f; break;
            case 3: v = 1.0f + (float)(u % 60000) * 1e-6f; break;
            case 5: v = (float)(u % 23) * 0.25f - 3.0f; break;
            case 6: v = -(float)(u % 23) * 0.25f - 1.0f; break;
            case 7: v = (float)(u % 1000003) / 1000.0f - 500.0f; break;
            default: v = (float)(u % 4999) * 0.01f; break;
        }
    }
    return f;
}

/* r4d_selftest's qsa_select_split_one at emulator size: three chunks, 96 distinct scores so the
 * tie rule decides, one page in 64 dead, `per` draft rows a sequence. */
Fix fix_split(int S, int per) {
    Fix f; f.name = "split: " + std::to_string(S) + " seq x " + std::to_string(per) + " rows, 96 scores";
    const int M = S * per, nb = 12001, topk = 512, ratio = 4, pages = 8192, n = 11000;
    f.shape(S, M, nb, nb, topk, ratio);
    for (int s = 0; s < S; ++s) {
        const int ns = n - s * 1300;
        f.nc[s] = ns; f.cu[s] = s * per;
        for (int b = 0; b < nb; ++b)
            f.page(s, b) = b >= ns ? -1 : b % 64 == 37 ? -1 : (b * 4099 + 11 + s * 71) % pages;
        for (int j = 0; j < per; ++j) f.pos[s * per + j] = ns * ratio + j - 2;
    }
    f.cu[S] = M;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 31 + m * 17) % 96) * 0.125f;
    return f;
}

/* Three chunks, none with more than topk live: the merge's candidates are every live key, more
 * than topk in the first row (its threshold the list's own, coarse scores tied at it), topk or
 * fewer in the second. The first row's sample holds fewer live keys than its bound, so the wide
 * kernel's list is every live key. */
Fix fix_split_sparse() {
    Fix f; f.name = "split: no chunk over topk";
    const int nb = 9000, topk = 512, ratio = 4;
    f.shape(2, 2, nb, nb, topk, ratio);
    f.cu = { 0, 1, 2 }; f.nc = { 8990, 8990 }; f.pos = { 4 * 8990 + 1, 4 * 8990 + 3 };
    for (int s = 0; s < 2; ++s)
        for (int b = 0; b < nb; ++b) f.page(s, b) = b % (s ? 20 : 9) == 3 ? (b * 131 + s) % 65536 : -1;
    for (int m = 0; m < 2; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 7 + m) % 13) * 0.5f;
    return f;
}

/* r4d_selftest's qsa_select_few_live_case: over more than topk blocks with topk or fewer live.
 *   0  few live pages, continuous scores           1  few live, most of them scoring zero or less
 *   2  live pages, most scores -inf                3  exactly topk live, a third tied at zero
 *   4  all live keys in the first chunk, more than topk, ties at the threshold
 *   5  nothing live                                6  at most topk blocks (control)
 *   7  everything live (control) */
Fix fix_few_live() {
    Fix f; f.name = "few live: topk or fewer live, in every way";
    const int S = 8, nb = 9000, topk = 512, ratio = 4, pages = 1 << 20;
    const int nrow[S] = { 8000, 8000, 8000, 8000, 2000, 8000, 300, 8000 };
    f.shape(S, S, nb, nb, topk, ratio);
    g_x = 0x9E3779B97F4A7C15ull;
    for (int s = 0; s < S; ++s) {
        const int n = nrow[s];
        f.nc[s] = n; f.cu[s] = s; f.pos[s] = n * ratio + s % 3;
        for (int b = 0; b < n; ++b) {
            float& v = f.score(s, b);
            int32_t& pg = f.page(s, b);
            const uint64_t u = rnd();
            const float cont = (float)(u % 1000003) / 1000.0f + 0.001f;
            pg = (int32_t)(u % pages);
            switch (s) {
                case 0: if (b % 80 != 3) pg = -1; v = cont; break;
                case 1: if (b % 20 != 7) pg = -1;
                        v = (b / 20) % 4 == 0 ? cont : (b / 20) % 3 == 0 ? -2.5f : (b / 20) % 3 == 1 ? -0.0f : 0.0f;
                        break;
                case 2: v = b % 40 == 1 ? cont : -INFINITY; break;
                case 3: if (b % 15 != 2 || b >= 15 * 512) pg = -1;
                        v = b % 45 == 2 ? 0.0f : cont; break;
                case 4: v = b % 7 == 0 ? cont : (b % 2 ? 0.0f : -0.0f); break;
                case 5: pg = -1; v = cont; break;
                case 6: if (b % 9 == 4) pg = -1; v = (float)(u % 13) * 0.5f; break;
                default: v = cont; break;
            }
        }
        f.page(s, n) = s % 2 == 0 ? (int32_t)(rnd() % pages) : -1;
    }
    f.cu[S] = S;
    return f;
}

/* ---- one block table per tile -------------------------------------------------------------------
 *
 * The lever this case guards: the sparse attention reads one block table per tile of query rows,
 * a stable mixture of the rows' own selections, with membership as a per-row mask, instead of one
 * table per row. The mixture is that tile's pages in block order, so every row's selection is a
 * subsequence of it and is read back unchanged -- the case checks that on the kernel's own output
 * rather than on the port, and reports how much the rows share.
 *
 * Two properties carry it. A row's list is in block order except for one run at the tail, the
 * threshold bin's ties (the tie rule picks them, so they cannot sit in place): a row whose list is
 * already in block order comes back byte for byte, and one that is not keeps its selection but is
 * read in the table's order, which is a different sum (docs/QSA.md, "The emit is one run in block
 * order"). And the partial page -- decided by the row's position -- stays out of the shared table
 * and is appended per row, so a page the chunk itself is filling cannot leak into a row that may
 * not see it. Both counts are reported; the tail run and the descents are also checked, since a
 * layout change in the emit would silently make the merge inexact. */

/* The score field a tile sees: one shared block field plus per-row noise, as the rows of a decode
 * step or of one prefill chunk do. `sigma` is the noise's width in thousandths of the field's, so
 * 0 is every row the same selection and larger values are rows that agree less; `grid` quantises
 * the sum, so scores tie exactly and the tie rule decides. */
Fix fix_shared_scores(const char* name, int M, int nb, int topk, int sigma, int grid, int seed) {
    Fix f; f.name = name;
    const int ratio = 4;
    f.shape(1, M, nb, nb, topk, ratio);
    f.cu = { 0, M }; f.nc = { nb };
    for (int b = 0; b < nb; ++b) f.page(0, b) = b;             /* page id = block index */
    g_x = 0x853C49E6748FEA9Bull + (uint64_t)seed * 0x9E3779B97F4A7C15ull;
    std::vector<float> base((size_t)nb);
    for (auto& v : base) v = (float)(rnd() % 1000003) / 1000003.0f;
    for (int m = 0; m < M; ++m) {
        /* One block short of the history's end, a row on and a row off the block boundary: every
         * block live, the last one partly filled, so half the rows carry a partial page. */
        f.pos[m] = 4 * (nb - 1) + 2 + m % 2;
        for (int b = 0; b < nb; ++b) {
            const float noise = (float)(rnd() % 1000003) / 1000003.0f - 0.5f;
            float v = base[(size_t)b] + noise * (float)sigma / 1000.0f;
            if (grid > 0) v = std::floor(v * (float)grid) / (float)grid;
            f.score(m, b) = v;
        }
    }
    return f;
}

/* A prefill tile: `rows` query rows of one chunk ending at the history's end, so the rows' visible
 * counts differ by a block and three rows in four carry a partial page -- the page the chunk is
 * filling, live in the table and belonging to one row only. */
Fix fix_shared_prefill(int nb, int rows, int sigma, int seed) {
    char name[96];
    std::snprintf(name, sizeof name, "shared: prefill, %d rows, %dK", rows, (nb * 4) / 1024);
    Fix f = fix_shared_scores(name, rows, nb, 512, sigma, 0, seed);
    const int base = 4 * nb - rows;
    for (int m = 0; m < rows; ++m) f.pos[m] = base + m;
    return f;
}

/* The row's last table entry is its partial page, exactly when the port appends one: the query's
 * count is short of the history and its position is off a block boundary, and the sequence's next
 * block is live (mid-chunk it is; at the end of the history it is not). */
bool row_has_partial(const Fix& f, int m) {
    const int n = f.n_of(m);
    return n < f.nb() && (f.pos[m] + 1) % f.ratio != 0 && f.bt[(size_t)(f.seq_of(m) * f.t_ld + n)] >= 0;
}

/* One table per tile, built by plugin/qsatable from the kernel's OWN `selb`/`nsel`, then read back
 * row by row. This is the measurement's premise checked in the same build as the selection: the
 * union of a tile's selections in block order, filtered by a row's membership mask and followed by
 * that row's partial page, has to be that row's own table -- byte for byte, `seqused` included, and
 * the row's selection is a subsequence of the union rather than a set the read path has to re-order.
 *
 * The rows are taken from `selb` (block indices), not from `sel`: a page id does not sort by block, a
 * dead block drops out of the block table and the prefix cache re-orders the rest, so a union built
 * from page ids would be the wrong order. The kernel fills the table; the op's host reference fills
 * a second set of buffers from the same inputs and the two are compared, so this case holds the
 * kernel to the definition of the op as well. */
int shared_table_case() {
    std::vector<Fix> fx = {
        fix_shared_scores("shared: decode depth 3, 8K", 4, 2048, 512, 300, 0, 1),
        fix_shared_scores("shared: decode depth 6, 8K", 8, 2048, 512, 300, 0, 2),
        fix_shared_scores("shared: decode depth 3, 32K", 4, 8192, 512, 300, 0, 3),
        fix_shared_scores("shared: decode depth 6, 32K", 8, 8192, 512, 300, 0, 4),
        fix_shared_scores("shared: depth 6, 32K, rows that agree less", 8, 8192, 512, 900, 0, 5),
        fix_shared_scores("shared: depth 6, 32K, a coarse grid of ties", 8, 8192, 512, 40, 64, 6),
        fix_shared_prefill(2048, 16, 300, 7),
        fix_shared_prefill(8192, 32, 300, 8),
    };
    int fails = 0;
    for (Fix& f : fx) {
        SelOut got;
        if (run(f, kAuto, &got)) { std::printf("FAIL %-46s launch\n", f.name.c_str()); ++fails; continue; }
        const int M = f.M, W = f.topk + 1;
        const int64_t tile = std::min(M, 16);
        const int64_t tiles = (M + tile - 1) / tile;
        const int64_t u_max = rad_qsa_table_u_max(tile, f.topk, f.nb());
        std::vector<int32_t> tab((size_t)(tiles * u_max), kSentinel), tlen((size_t)tiles, kSentinel);
        std::vector<int32_t> mask((size_t)M * u_max, kSentinel), roff((size_t)M, kSentinel);
        RadTensor t[7] = {
            tens(got.sb.data(), RAD_I32, {M, W}, f.o_ld), tens(got.ns.data(), RAD_I32, {M}),
            tens(f.pos.data(), RAD_I32, {M}),            tens(tab.data(), RAD_I32, {tiles, u_max}),
            tens(tlen.data(), RAD_I32, {tiles}),         tens(mask.data(), RAD_I32, {M, u_max}),
            tens(roff.data(), RAD_I32, {M}),
        };
        RadParam p[6]{};
        p[0] = {"M", RAD_P_INT, M}; p[1] = {"blocks", RAD_P_INT, f.nb()};
        p[2] = {"topk", RAD_P_INT, f.topk}; p[3] = {"tile", RAD_P_INT, (int)tile};
        p[4] = {"ratio", RAD_P_INT, f.ratio}; p[5] = {"u_max", RAD_P_INT, (int)u_max};
        RadArgs a{}; a.t = t; a.n_t = 7; a.p = p; a.n_p = 6;
        const int rc = rad_qsa_table_launch(&a, nullptr);
        if (rc != RAD_OK) {
            std::printf("FAIL %-46s qsa_table returned %d\n", f.name.c_str(), rc);
            ++fails;
            continue;
        }
        /* The same inputs through the op's host reference: the kernel above must agree byte for
         * byte, which is what makes the union a definition and not one implementation's choice. */
        std::vector<int32_t> rtab((size_t)(tiles * u_max), kSentinel), rtlen((size_t)tiles, kSentinel);
        std::vector<int32_t> rmask((size_t)M * u_max, kSentinel), rroff((size_t)M, kSentinel);
        RadTensor r[7] = {
            tens(got.sb.data(), RAD_I32, {M, W}, f.o_ld), tens(got.ns.data(), RAD_I32, {M}),
            tens(f.pos.data(), RAD_I32, {M}),            tens(rtab.data(), RAD_I32, {tiles, u_max}),
            tens(rtlen.data(), RAD_I32, {tiles}),        tens(rmask.data(), RAD_I32, {M, u_max}),
            tens(rroff.data(), RAD_I32, {M}),
        };
        RadArgs ra = a; ra.t = r;
        const int rrc = rad_ref_qsa_table(&ra);
        if (rrc != RAD_OK) {
            std::printf("FAIL %-46s ref_qsa_table returned %d\n", f.name.c_str(), rrc);
            ++fails;
            continue;
        }
        if (rtab != tab || rtlen != tlen || rmask != mask || rroff != roff) {
            std::printf("FAIL %-46s the kernel's table differs from the reference's\n", f.name.c_str());
            ++fails;
            continue;
        }
        int64_t total = 0, union_ = 0, off_rows = 0;
        bool unordered = false;
        int bad_rows = 0;
        for (int64_t r = 0; r < tiles; ++r) {
            total += 0;
            if (tlen[(size_t)r] < 0 || tlen[(size_t)r] > u_max) { ++bad_rows; continue; }
            union_ += tlen[(size_t)r];
            for (int64_t j = 1; j < tlen[(size_t)r]; ++j)
                if (tab[(size_t)(r * u_max + j)] <= tab[(size_t)(r * u_max + j - 1)]) unordered = true;
        }
        for (int m = 0; m < M; ++m) {
            const int64_t t0 = m / tile;
            if (roff[(size_t)m] != (int32_t)(t0 * u_max)) ++off_rows;
            total += got.ns[(size_t)m];
            std::vector<int32_t> rec;
            for (int64_t j = 0; j < tlen[(size_t)t0]; ++j)
                if (mask[(size_t)(m * u_max + j)])
                    rec.push_back(f.bt[(size_t)(f.seq_of(m) * f.t_ld + tab[(size_t)(t0 * u_max + j)])]);
            /* The partial page is position-decided, stays out of the table and is appended per row. */
            if (row_has_partial(f, m)) rec.push_back(got.se[(size_t)(m * f.o_ld + got.ns[(size_t)m] - 1)]);
            bool same = (int)rec.size() == got.ns[(size_t)m];
            for (int k = 0; same && k < (int)rec.size(); ++k)
                same = got.se[(size_t)(m * f.o_ld + k)] == rec[(size_t)k];
            for (int k = (int)rec.size(); same && k <= f.topk; ++k)
                same = got.se[(size_t)(m * f.o_ld + k)] == -1;
            if (!same && ++bad_rows <= 3)
                std::printf("FAIL %-46s row %d: the shared table does not recover this row\n",
                            f.name.c_str(), m);
        }
        if (unordered) { std::printf("FAIL %-46s a table is not in block order\n", f.name.c_str()); ++fails; }
        if (off_rows) { std::printf("FAIL %-46s %lld row(s) with a wrong row_off\n", f.name.c_str(), (long long)off_rows); ++fails; }
        const double factor = union_ ? (double)total / (double)union_ : 0.0;
        std::printf("  %-46s %d rows x %lld pages -> %lld blocks in %lld tile(s) (%.1fx fewer), "
                    "mask %lld B/row (a bitmask would be %lld), %d/%d rows recovered\n",
                    f.name.c_str(), M, (long long)(total / M), (long long)union_, (long long)tiles,
                    factor, (long long)(u_max * 4), (long long)((u_max + 7) / 8), M - bad_rows, M);
        fails += bad_rows ? 1 : 0;
    }
    std::printf("shared table: %zu tiles, %d failure(s)\n", fx.size(), fails);
    return fails;
}

/* ---- The masked read of the shared table (plugin/qsatable's `qsa_attn`). ----
 *
 * The premise of the measurement, checked from the other end: if a row's table is a subsequence of
 * the tile's union, then reading the union through the row's membership mask must give the row's own
 * result. The case runs the same rows through the same op twice -- once with a table per row (tile 1,
 * every entry the row's own, mask all ones) and once with the tile's union in block order (tile 16)
 * -- and requires the outputs to be equal **byte for byte**. Byte equality is the claim: softmax is
 * summed in table order and float addition is not associative, so anything less than the same keys
 * in the same order would show up here as a difference, not as a rounding.
 *
 * The keys are real: a KV cache of the fixtures' pages, a bf16 query, and the row's position from the
 * fixture. The op's f64 host reference is run on the per-row table too, so the case also holds the
 * read to the definition of the op -- the partial page included, which is position-decided and
 * appended by the reader rather than read off the table. */

/* Round-to-nearest-even bf16, so the query and the cache are values the op can actually read back. */
uint16_t bf16_of(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

struct SharedTab {
    int64_t tile = 1, tiles = 1, u_max = 1;
    std::vector<int32_t> tab, tlen, mask, roff;
};

/* One table for the tile of rows starting at `tile`, built by the op that builds it: the union of the
 * rows' `selb` in block order, the membership mask and `row_off`. */
int build_table(const Fix& f, const SelOut& got, int64_t tile, SharedTab& t) {
    const int M = f.M, W = f.topk + 1;
    t.tile = tile;
    t.tiles = (M + tile - 1) / tile;
    t.u_max = rad_qsa_table_u_max(tile, f.topk, f.nb());
    if (t.u_max < 1) return RAD_E_SHAPE;
    t.tab.assign((size_t)(t.tiles * t.u_max), kSentinel);
    t.tlen.assign((size_t)t.tiles, kSentinel);
    t.mask.assign((size_t)M * t.u_max, kSentinel);
    t.roff.assign((size_t)M, kSentinel);
    RadTensor a[7] = {
        tens((void*)got.sb.data(), RAD_I32, {M, W}, f.o_ld), tens((void*)got.ns.data(), RAD_I32, {M}),
        tens((void*)f.pos.data(), RAD_I32, {M}),  tens(t.tab.data(), RAD_I32, {t.tiles, t.u_max}),
        tens(t.tlen.data(), RAD_I32, {t.tiles}),  tens(t.mask.data(), RAD_I32, {M, t.u_max}),
        tens(t.roff.data(), RAD_I32, {M}),
    };
    RadParam p[6]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"blocks", RAD_P_INT, f.nb()};
    p[2] = {"topk", RAD_P_INT, f.topk}; p[3] = {"tile", RAD_P_INT, (int)tile};
    p[4] = {"ratio", RAD_P_INT, f.ratio}; p[5] = {"u_max", RAD_P_INT, (int)t.u_max};
    RadArgs args{}; args.t = a; args.n_t = 7; args.p = p; args.n_p = 6;
    return rad_qsa_table_launch(&args, nullptr);
}

/* The read: one launch of `qsa_attn` over `t`, or over its host reference when `ref` is set. */
int read_table(const Fix& f, const SharedTab& t, int64_t hd, int64_t nh, int64_t gqa, int64_t blocks,
                int64_t npages, const std::vector<uint16_t>& q, const std::vector<uint16_t>& kv,
                std::vector<uint16_t>& out, bool ref) {
    const int M = f.M;
    out.assign((size_t)(M * nh * hd), 0);
    RadTensor a[10] = {
        tens((void*)q.data(), RAD_BF16, {M, nh * hd}), tens((void*)kv.data(), RAD_BF16, {npages, nh / gqa, f.ratio, 2 * hd}),
        tens((void*)t.tab.data(), RAD_I32, {t.tiles, t.u_max}), tens((void*)t.tlen.data(), RAD_I32, {t.tiles}),
        tens((void*)t.mask.data(), RAD_I32, {M, t.u_max}),       tens((void*)t.roff.data(), RAD_I32, {M}),
        tens((void*)f.pos.data(), RAD_I32, {M}),                 tens((void*)f.bt.data(), RAD_I32, {f.S, f.nb_t}, f.t_ld),
        /* `cu_seqlens`: the row's sequence is the interval holding it, so no per-row id operand. */
        tens((void*)f.cu.data(), RAD_I32, {f.S + 1}),            tens(out.data(), RAD_BF16, {M, nh * hd}),
    };
    std::vector<RadParam> p;
    auto pi = [&](const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p.push_back(x); };
    auto ps = [&](const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p.push_back(x); };
    auto pf = [&](const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p.push_back(x); };
    pi("M", M); pi("n_head", nh); pi("head_dim", hd); pi("gqa", gqa);
    pi("block_size", f.ratio); pi("ratio", f.ratio); pi("tile", t.tile);
    pi("topk", f.topk); pi("blocks", blocks); pi("max_seqs", f.S); pi("u_max", (long long)t.u_max);
    pf("scale", 1.0 / std::sqrt((double)hd));
    ps("q_dtype", "bf16"); ps("kv_dtype", "bf16");
    RadArgs args{}; args.t = a; args.n_t = 10; args.p = p.data(); args.n_p = (int)p.size();
    return ref ? rad_ref_qsa_attn(&args) : rad_qsa_attn_launch(&args, nullptr);
}

/* The row path as this build resolves it: libref's `attn_paged`, through the plugin's own exports,
 * or null when $RADIANCE_HOME holds no libref (the case then reports that it covered only the
 * oracle). */
typedef int (*AttnLaunch)(const RadArgs*, RadStream);
/* The row path is the case's opponent, not an optional extra: `masked read:` claims the table's
 * read *is* the attention the architecture issues today, and a build that cannot load that
 * attention cannot make the claim. So the lookup reports why it failed and the case fails on it --
 * a missing reference must not turn the required comparison into a green run. */
AttnLaunch load_row_path(std::string* why) {
    const char* home = std::getenv("RADIANCE_HOME");
    if (!home || !*home) { *why = "RADIANCE_HOME is not set"; return nullptr; }
    const std::string p = std::string(home) + "/kernels/libref.so";
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* e = dlerror();                       /* cleared by the failed dlopen */
        *why = std::string("cannot load ") + p + ": " + (e ? e : "unknown");
        return nullptr;
    }
    auto open = (int (*)(void))dlsym(h, "rad_plugin_open");      /* as the loader does */
    if (open && open() < 0) { *why = p + ": rad_plugin_open refused"; return nullptr; }
    auto cnt = (int (*)(void))dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!cnt || !at) { *why = p + " has no rad_kernel_count/rad_kernel_at"; return nullptr; }
    for (int i = 0; i < cnt(); ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && !std::strcmp(k->op, "attn_paged")) return k->launch;
    }
    *why = p + " declares no attn_paged";
    return nullptr;
}

/* The row path's numbers for the same rows: `attn_paged` with `q_len` 1, so every query row is its
 * own sequence, its selection (`selb`, the block table of the table path) as the block table and
 * `seqused` as the row's key count. Block size is the QSA ratio, so key j of entry i is the cache's
 * slot i * ratio + j -- the mapping the masked read walks. */
int row_path_read(const Fix& f, const SelOut& got, int64_t hd, int64_t nh, int64_t gqa,
                  const std::vector<uint16_t>& q, const std::vector<uint16_t>& kv, int64_t npages,
                  AttnLaunch launch, std::vector<uint16_t>& out) {
    const int M = f.M, W = f.topk + 1;
    out.assign((size_t)(M * nh * hd), 0);
    RadTensor none{};
    RadTensor a[8] = {
        tens((void*)q.data(), RAD_BF16, {M, nh * hd}),
        tens((void*)kv.data(), RAD_BF16, {npages, nh / gqa, f.ratio, 2 * hd}),
        tens((void*)got.se.data(), RAD_I32, {M, W}, f.o_ld), tens((void*)got.su.data(), RAD_I32, {M}),
        none, none, none,                                       /* no descales, no cu_seqlens */
        tens(out.data(), RAD_BF16, {M, nh * hd}),
    };
    RadParam p[13]{};
    int n = 0;
    auto pi = [&](const char* k, long long v) { RadParam x{}; x.key = k; x.kind = RAD_P_INT; x.ival = v; p[n++] = x; };
    auto pf = [&](const char* k, double v) { RadParam x{}; x.key = k; x.kind = RAD_P_F64; x.dval = v; p[n++] = x; };
    auto ps = [&](const char* k, const char* v) { RadParam x{}; x.key = k; x.kind = RAD_P_STR; x.sval = v; p[n++] = x; };
    pi("q_len", 1); pi("head_dim", hd); pi("gqa", gqa); pi("block_size", f.ratio);
    pi("causal", 1); pi("window", 0); ps("q_dtype", "bf16"); ps("kv_dtype", "bf16");
    pf("scale", 1.0 / std::sqrt((double)hd)); pi("max_ctx", 0); pi("n_head", nh); pi("max_seqs", M);
    RadArgs args{}; args.t = a; args.n_t = 8; args.p = p; args.n_p = n;
    return launch(&args, nullptr);
}

int masked_read_case() {
    std::vector<Fix> fx = {
        fix_shared_scores("masked read: decode depth 3, 8K", 4, 2048, 512, 300, 0, 21),
        fix_shared_scores("masked read: decode depth 6, 8K", 8, 2048, 512, 300, 0, 22),
        fix_shared_scores("masked read: decode depth 3, 32K", 4, 8192, 512, 300, 0, 27),
        fix_shared_scores("masked read: depth 6, 32K, rows that agree less", 8, 8192, 512, 900, 0, 23),
        fix_shared_scores("masked read: depth 6, 32K, a coarse grid of ties", 8, 8192, 512, 40, 64, 24),
        fix_shared_prefill(2048, 16, 300, 25),
        fix_shared_prefill(8192, 32, 300, 26),
    };
    /* A small head so the f64 reference stays cheap: the op's arithmetic does not depend on its width. */
    const int64_t hd = 32, nh = 4, gqa = 1;
    std::string why;
    AttnLaunch rowp = load_row_path(&why);
    if (!rowp) {
        std::printf("FAIL masked read: the row path is unavailable: %s\n", why.c_str());
        std::printf("FAIL masked read: build libref and set RADIANCE_HOME to its prefix "
                    "($RADIANCE_HOME/kernels/libref.so); the byte claim needs it\n");
        return 1;
    }
    int fails = 0, partial_rows = 0;
    for (Fix& f : fx) {
        SelOut got;
        if (run(f, kAuto, &got)) { std::printf("FAIL %-52s the selection\n", f.name.c_str()); ++fails; continue; }
        const int M = f.M, ratio = f.ratio;
        const int64_t blocks = f.nb(), npages = f.nb() + 1;
        std::vector<uint16_t> kv((size_t)(npages * (nh / gqa) * ratio * 2 * hd));
        std::vector<uint16_t> q((size_t)(M * nh * hd));
        for (auto& v : kv) v = bf16_of((float)(rnd() % 2001) / 1000.0f - 1.0f);
        for (auto& v : q) v = bf16_of((float)(rnd() % 2001) / 1000.0f - 1.0f);

        SharedTab per_row, shared;
        int rc = build_table(f, got, 1, per_row);
        if (rc == RAD_OK) rc = build_table(f, got, 16, shared);  /* a tile, never clamped to M: a 4-row step is one short tile */
        if (rc != RAD_OK) { std::printf("FAIL %-52s build_table returned %d\n", f.name.c_str(), rc); ++fails; continue; }

        std::vector<uint16_t> own, uni, oracle;
        rc = read_table(f, per_row, hd, nh, gqa, blocks, npages, q, kv, own, false);
        if (rc == RAD_OK) rc = read_table(f, shared, hd, nh, gqa, blocks, npages, q, kv, uni, false);
        if (rc == RAD_OK) rc = read_table(f, per_row, hd, nh, gqa, blocks, npages, q, kv, oracle, true);
        if (rc != RAD_OK) { std::printf("FAIL %-52s qsa_attn returned %d\n", f.name.c_str(), rc); ++fails; continue; }

        /* The claim: the shared table's masked read is the per-row read, byte for byte. */
        if (own != uni) {
            int bad = 0;
            for (size_t i = 0; i < own.size(); ++i) bad += own[i] != uni[i];
            std::printf("FAIL %-52s the shared table's read differs in %d of %zu values\n",
                        f.name.c_str(), bad, own.size());
            ++fails;
            continue;
        }
        /* And both against the op's f64 oracle, which reads the per-row table. */
        double worst = 0.0;
        for (size_t i = 0; i < own.size(); ++i) {
            const double a = rad_bf16_to_f32(own[i]), b = rad_bf16_to_f32(oracle[i]);
            worst = std::max(worst, std::fabs(a - b));
        }
        if (!(worst < 2e-2)) {
            std::printf("FAIL %-52s the read is %.3g from the oracle\n", f.name.c_str(), worst);
            ++fails;
            continue;
        }
        /* The measurement's claim: this read is the row path's read. `libref`'s `attn_paged` over the
         * same rows, in this build, byte for byte -- not a tolerance, because both walk the row's keys
         * in block order with the same expressions. */
        {
            std::vector<uint16_t> path;
            rc = row_path_read(f, got, hd, nh, gqa, q, kv, npages, rowp, path);
            if (rc != RAD_OK) {
                std::printf("FAIL %-52s attn_paged returned %d\n", f.name.c_str(), rc);
                ++fails;
                continue;
            }
            if (path != own) {
                int bad = 0, first = -1;
                for (size_t i = 0; i < own.size(); ++i)
                    if (own[i] != path[i]) { ++bad; if (first < 0) first = (int)i; }
                std::printf("FAIL %-52s the row path differs in %d of %zu values, first %d: "
                            "%g against %g\n",
                            f.name.c_str(), bad, own.size(), first,
                            rad_bf16_to_f32(own[(size_t)first]), rad_bf16_to_f32(path[(size_t)first]));
                ++fails;
                continue;
            }
        }
        /* Report the sharing the read path actually walked, and how many rows carried a partial
         * page (the one page the table does not hold). */
        int partial = 0;
        for (int m = 0; m < M; ++m) partial += (f.pos[(size_t)m] + 1) % ratio != 0;
        partial_rows += partial;
        int64_t listed = 0, unioned = 0;
        for (int m = 0; m < M; ++m) listed += got.ns[(size_t)m];
        for (int64_t t = 0; t < shared.tiles; ++t) unioned += shared.tlen[(size_t)t];
        std::printf("  %-52s %d rows: %lld pages read -> %lld in %lld tile(s) (%.1fx fewer), "
                    "%d partial-page row(s), byte-identical, %.1e from the oracle, "
                    "byte-identical to the row path\n",
                    f.name.c_str(), M, (long long)listed, (long long)unioned,
                    (long long)shared.tiles, unioned ? (double)listed / (double)unioned : 0.0, partial,
                    worst);
    }
    std::printf("masked read: %zu fixtures, %d failure(s), %d partial-page row(s), "
                "row path: libref attn_paged, byte for byte\n",
                fx.size(), fails, partial_rows);
    return fails;
}

/* The architecture's own call, through the real argument parser and the real table op: two sequences
 * in one step (`cu` = 0, 256, 512), a declared capacity of 256 sequences and a block table with the
 * batch's two rows. The first sequence's blocks are all dead, so a row that resolves to the wrong
 * sequence reads no key and comes out zero -- which is the failure this case exists for, and the one
 * the capacity-for-live-count confusion produced (`max_seqs` is the chunk's row count, not the batch's
 * sequence count). The batch again as a chunk starting at row 256 then shows the other half: without
 * `row0` a chunk looks its rows up in the first sequence's interval. */
int parser_case() {
    const int64_t ratio = 4, tile = 16, topk = 4, blocks = 64, hd = 32, nh = 2, gqa = 1;
    const int M = 512, S = 2, cap = 256;      /* `cap`: the capacity the architecture declares */
    const int W = (int)topk + 1;
    std::vector<int32_t> cu = { 0, M / 2, M };
    std::vector<int32_t> bt((size_t)S * blocks, -1);       /* sequence 0 dead, sequence 1 identity */
    for (int64_t b = 0; b < blocks; ++b) bt[(size_t)blocks + b] = (int32_t)b;
    std::vector<int32_t> pos(M), selb((size_t)M * W, -1), nsel(M, 2);
    for (int m = 0; m < M; ++m) {
        pos[m] = (int32_t)(ratio * (30 + m % 5) + 2);      /* own 30..34, a partial page of 3 */
        selb[(size_t)m * W] = 0;
        selb[(size_t)m * W + 1] = 1;
    }
    const int64_t tiles = (M + tile - 1) / tile;
    const int64_t u_max = rad_qsa_table_u_max(tile, topk, blocks);
    std::vector<int32_t> tab((size_t)(tiles * u_max)), tlen((size_t)tiles),
        mask((size_t)M * u_max), roff((size_t)M);
    std::vector<uint16_t> q((size_t)(M * nh * hd)), kv((size_t)(blocks * (nh / gqa) * ratio * 2 * hd));
    for (auto& v : q) v = bf16_of((float)(rnd() % 2001) / 1000.0f - 1.0f);
    for (auto& v : kv) v = bf16_of((float)(rnd() % 2001) / 1000.0f - 1.0f);

    RadTensor ta[7] = {
        tens(selb.data(), RAD_I32, {M, W}), tens(nsel.data(), RAD_I32, {M}),
        tens(pos.data(), RAD_I32, {M}), tens(tab.data(), RAD_I32, {tiles, u_max}),
        tens(tlen.data(), RAD_I32, {tiles}), tens(mask.data(), RAD_I32, {M, u_max}),
        tens(roff.data(), RAD_I32, {M}),
    };
    RadParam tp[6]{};
    tp[0] = {"M", RAD_P_INT, M}; tp[1] = {"blocks", RAD_P_INT, (int)blocks};
    tp[2] = {"topk", RAD_P_INT, (int)topk}; tp[3] = {"tile", RAD_P_INT, (int)tile};
    tp[4] = {"ratio", RAD_P_INT, (int)ratio}; tp[5] = {"u_max", RAD_P_INT, (int)u_max};
    RadArgs targs{}; targs.t = ta; targs.n_t = 7; targs.p = tp; targs.n_p = 6;
    int rc = rad_qsa_table_launch(&targs, nullptr);
    if (rc != RAD_OK) { std::printf("FAIL parsed: qsa_table returned %d\n", rc); return 1; }

    /* A launch over `rows` rows starting at batch row `off`, with `cu` whole and the block table
     * handed over at `bts`/`btp` -- the width the step staged it at. */
    auto attn = [&](int rows, int off, bool with_row0, const int32_t* roffs, const int32_t* bts,
                    int64_t btp, std::vector<uint16_t>& o) -> int {
        o.assign((size_t)(rows * nh * hd), 0);
        const int64_t rt = (rows + tile - 1) / tile;
        RadTensor a[10] = {
            tens(q.data() + (size_t)off * nh * hd, RAD_BF16, {rows, nh * hd}),
            tens(kv.data(), RAD_BF16, {blocks, nh / gqa, ratio, 2 * hd}),
            tens(tab.data() + (size_t)(off / tile) * u_max, RAD_I32, {rt, u_max}),
            tens(tlen.data() + off / tile, RAD_I32, {rt}),
            tens(mask.data() + (size_t)off * u_max, RAD_I32, {rows, u_max}),
            tens((void*)roffs, RAD_I32, {rows}), tens(pos.data() + off, RAD_I32, {rows}),
            tens((void*)bts, RAD_I32, {S, btp}, btp), tens(cu.data(), RAD_I32, {S + 1}),
            tens(o.data(), RAD_BF16, {rows, nh * hd}),
        };
        RadParam p[16]{};
        int n = 0;
        p[n++] = RAD_INT("M", rows); p[n++] = RAD_INT("n_head", nh);
        p[n++] = RAD_INT("head_dim", hd); p[n++] = RAD_INT("gqa", gqa);
        p[n++] = RAD_INT("block_size", ratio); p[n++] = RAD_INT("ratio", ratio);
        p[n++] = RAD_INT("tile", tile); p[n++] = RAD_INT("topk", topk);
        p[n++] = RAD_INT("blocks", blocks); p[n++] = RAD_INT("max_seqs", cap);
        p[n++] = RAD_INT("u_max", u_max);
        if (with_row0) p[n++] = RAD_INT("row0", off);
        p[n++] = RAD_F64("scale", 1.0 / std::sqrt((double)hd));
        p[n++] = RAD_STR("q_dtype", "bf16");
        p[n++] = RAD_STR("kv_dtype", "bf16");
        RadArgs args{}; args.t = a; args.n_t = 10; args.p = p; args.n_p = n;
        return rad_qsa_attn_launch(&args, nullptr);
    };

    const int64_t rv = nh * hd;
    int fails = 0;
    auto nonzero = [&](const std::vector<uint16_t>& o, int m) {
        for (int i = 0; i < rv; ++i)
            if (o[(size_t)m * rv + i] != 0) return true;
        return false;
    };

    /* The whole step: the live sequence reads, the dead one does not. */
    std::vector<uint16_t> whole;
    rc = attn(M, 0, false, roff.data(), bt.data(), blocks, whole);
    if (rc != RAD_OK) { std::printf("FAIL parsed: the step's launch returned %d\n", rc); return 1; }
    int wrong_seq = 0, empty_live = 0;
    for (int m = 0; m < M; ++m) {
        const bool nz = nonzero(whole, m);
        if (m < M / 2 && nz) ++wrong_seq;        /* sequence 0 has no live block */
        if (m >= M / 2 && !nz) ++empty_live;     /* sequence 1 owns every page */
    }
    if (wrong_seq || empty_live) {
        std::printf("FAIL parsed: %d row(s) of the dead sequence read, %d row(s) of the live one did not\n",
                    wrong_seq, empty_live);
        ++fails;
    }

    /* The same rows as a chunk: `row0` is what keeps them in the second sequence, and `row_off` is
     * rebased because it counts offsets into the tab plane the caller hands over, which the chunk
     * hands over sliced (`qsaattn.h`). */
    std::vector<int32_t> chunk_roff(roff.begin() + M / 2, roff.end());
    for (auto& v : chunk_roff) v -= (int32_t)((M / 2 / tile) * u_max);
    std::vector<uint16_t> chunk, nochunk;

    rc = attn(M / 2, M / 2, true, chunk_roff.data(), bt.data(), blocks, chunk);
    if (rc != RAD_OK) { std::printf("FAIL parsed: the chunked launch returned %d\n", rc); ++fails; }
    else {
        int empty = 0;
        for (int m = 0; m < M / 2; ++m) empty += !nonzero(chunk, m);
        if (empty) { std::printf("FAIL parsed: %d chunk row(s) read nothing with row0\n", empty); ++fails; }
    }
    rc = attn(M / 2, M / 2, false, chunk_roff.data(), bt.data(), blocks, nochunk);
    if (rc != RAD_OK) { std::printf("FAIL parsed: the chunked launch without row0 returned %d\n", rc); ++fails; }
    else {
        int read = 0;
        for (int m = 0; m < M / 2; ++m) read += nonzero(nochunk, m);
        if (read) { std::printf("FAIL parsed: %d chunk row(s) read sequence 0 without row0\n", read); ++fails; }
    }
    /* The block table is the step's, not the block space's. `core/sched/batch.cpp` stages each
     * group's table at this step's reach -- "the pitch is per step, not max_ctx wide, so decode does
     * not stage megabytes of empty rows" -- while `blocks` is the group's whole capacity: a decode
     * step of 8K tokens stages a few hundred entries of a 50000-block space. The same rows read off
     * a table staged at that narrower pitch must read the same bytes, and the op must not refuse the
     * step over the difference (MAIN-55: every step of the table arm aborted at `M=1` with the band's
     * geometry, which is the whole reason a serve produced no A/B number). `own` is 30..34 here. */
    {
        const int64_t btp = 40;
        std::vector<int32_t> narrow((size_t)S * btp, -1);
        for (int64_t b = 0; b < btp; ++b) narrow[(size_t)btp + b] = (int32_t)b;
        std::vector<uint16_t> o;
        rc = attn(M, 0, false, roff.data(), narrow.data(), btp, o);
        if (rc != RAD_OK) {
            std::printf("FAIL parsed: a table staged at this step's pitch (%lld of %lld blocks) "
                        "returned %d\n", (long long)btp, (long long)blocks, rc);
            ++fails;
        } else if (o != whole) {
            int bad = 0;
            for (size_t i = 0; i < o.size(); ++i) bad += o[i] != whole[i];
            std::printf("FAIL parsed: the step's own block-table pitch reads %d of %zu values "
                        "differently\n", bad, o.size());
            ++fails;
        }
    }

    std::printf("  parsed: %d rows, %d sequences, capacity %d, block table %d row(s): %d row(s) of the "
                "live sequence read, the chunk reads with row0 and not without, the step's own "
                "block-table pitch reads the same bytes\n",
                M, S, cap, S, M / 2);
    std::printf("parsed: 1 batch of %d sequences, %d failure(s)\n", S, fails);
    return fails;
}

/* A 70-row prefill chunk: wider than the key cache's gate, so the automatic choice is the wide
 * kernel and the pinned single-workgroup kernel reads its keys from global memory. */
Fix fix_wide_launch() {
    Fix f; f.name = "a 70-row chunk: the wide kernel by default";
    const int M = 70, nb = 160, topk = 16, ratio = 4;
    f.shape(1, M, nb, nb, topk, ratio);
    f.cu = { 0, M }; f.nc = { 110 };
    for (int m = 0; m < M; ++m) f.pos[m] = 4 * 90 + m;
    for (int b = 0; b < nb; ++b) f.page(0, b) = b % 11 == 5 ? -1 : (b * 37 + 3) % 4096;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 13 + m * 3) % 9) * 0.5f;
    return f;
}

/* topk above the merged form's 512: the wide kernel's threshold bin goes to LDS alone, and the
 * split path still applies (two chunks of 2500, two lists of up to 1000 keys). */
Fix fix_big_topk() {
    Fix f; f.name = "topk 1000: above the merged form";
    const int nb = 5000, topk = 1000, ratio = 4;
    f.shape(1, 3, nb, nb, topk, ratio);
    f.cu = { 0, 3 }; f.nc = { 4990 }; f.pos = { 4 * 4990 + 2, 4 * 3000 - 1, 4 * 900 + 1 };
    for (int b = 0; b < nb; ++b) f.page(0, b) = b % 13 == 6 ? -1 : (b * 211 + 7) % 100000;
    for (int m = 0; m < 3; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 11 + m * 5) % 50) * 0.125f;
    return f;
}

/* Where the launcher leaves the split path: 33 query rows (one past its 32), and topk 3000, whose
 * three chunk lists overflow its 8192 keys. Both take the single-workgroup kernel. */
Fix fix_no_split(bool rows) {
    Fix f; f.name = rows ? "33 rows: past the split path's 32" : "topk 3000: past the split path's lists";
    const int M = rows ? 33 : 2, nb = rows ? 4500 : 9000, topk = rows ? 64 : 3000, ratio = 4;
    f.shape(1, M, nb, nb, topk, ratio);
    f.cu = { 0, M }; f.nc = { nb - 10 };
    for (int m = 0; m < M; ++m) f.pos[m] = 4 * (nb - 10 - 97 * m) + m % 4;
    for (int b = 0; b < nb; ++b) f.page(0, b) = b % 17 == 1 ? -1 : (b * 389 + 1) % 200000;
    for (int m = 0; m < M; ++m)
        for (int b = 0; b < nb; ++b) f.score(m, b) = (float)((b * 29 + m * 3) % 211) * 0.0625f;
    return f;
}

/* A seeded small fixture: odd values, pitches, widths and row-to-sequence maps. */
Fix fix_sweep(int it) {
    g_x = 0xD1B54A32D192ED03ull ^ ((uint64_t)(it + 1) * 0x9E3779B97F4A7C15ull);
    for (int i = 0; i < 4; ++i) rnd();
    static const int kTopk[] = { 1, 2, 3, 5, 8, 17, 33, 64 }, kRatio[] = { 1, 2, 3, 4, 7 };
    const int topk = kTopk[rnd() % 8], ratio = kRatio[rnd() % 5], S = 1 + (int)(rnd() % 3);
    const int nb_t = 1 + (int)(rnd() % 300), nb_s = rnd() % 3 ? nb_t : 1 + (int)(rnd() % 300);
    std::vector<int> rows((size_t)S);
    int M = 0;
    for (auto& r : rows) { r = (int)(rnd() % 5); M += r; }
    const int lead = rnd() % 4 == 0 ? 1 : 0, trail = rnd() % 4 == 0 ? 1 : 0;   /* rows no sequence holds */
    M += lead + trail;
    if (M == 0) { rows[0] = 1; M = 1; }
    Fix f; f.name = "sweep " + std::to_string(it);
    f.shape(S, M, nb_s, nb_t, topk, ratio, (int)(rnd() % 4), (int)(rnd() % 4), (int)(rnd() % 3));
    f.cu[0] = lead;
    for (int s = 0; s < S; ++s) f.cu[s + 1] = f.cu[s] + rows[(size_t)s];
    const int nb = f.nb();
    for (int s = 0; s < S; ++s) f.nc[s] = (int)(rnd() % (uint64_t)(nb + 6));
    for (int m = 0; m < M; ++m) f.pos[m] = (int)(rnd() % (uint64_t)(ratio * (nb + 3)));
    const unsigned dead = (unsigned)(rnd() % 4);      /* none, a tenth, half, nearly all */
    for (auto& v : f.bt) {
        const uint64_t u = rnd();
        v = (dead == 1 && u % 10 == 0) || (dead == 2 && u % 2) || (dead == 3 && u % 20) ? -1 : (int32_t)(u % 100000);
    }
    const unsigned grid = 1 + (unsigned)(rnd() % 12);  /* distinct values: ties when small */
    for (auto& v : f.sc) {
        const uint64_t u = rnd();
        switch (u % 23) {
            case 0: v = NAN; break;
            case 1: v = INFINITY; break;
            case 2: v = -INFINITY; break;
            case 3: v = -0.0f; break;
            case 4: v = 0.0f; break;
            case 5: v = 1e-40f * (float)(1 + (u >> 8) % 4); break;   /* denormal */
            case 6: v = -1e30f; break;                                /* not live: the bound is strict */
            case 7: v = -9e29f; break;                                /* live, ranked as zero */
            case 8: v = -(float)((u >> 8) % 100); break;
            default: v = (float)((u >> 8) % grid) * 0.75f; break;
        }
    }
    return f;
}

/* A seeded fixture at serving sizes for `sweep`: 500 to 14500 blocks, topk on both sides of the
 * merged form's 512 and of the split path's list budget, one to three sequences of a few draft
 * rows or of a prefill chunk past 64 rows, and dead pages, -inf, NaN and ties in varying amounts. */
Fix fix_random(int it) {
    g_x = 0xA0761D6478BD642Full ^ ((uint64_t)(it + 7) * 0xE7037ED1A0B428DBull);
    for (int i = 0; i < 4; ++i) rnd();
    static const int kTopk[] = { 1, 16, 64, 200, 511, 512, 513, 700, 1500, 2600 }, kRatio[] = { 1, 4, 4, 8 };
    const int topk = kTopk[rnd() % 10], ratio = kRatio[rnd() % 4], S = 1 + (int)(rnd() % 3);
    const int nb = 500 + (int)(rnd() % 14000);
    const bool chunk = rnd() % 6 == 0;
    std::vector<int> rows((size_t)S);
    int M = 0;
    for (auto& r : rows) { r = chunk ? 20 + (int)(rnd() % 30) : 1 + (int)(rnd() % 5); M += r; }
    Fix f; f.name = "sweep " + std::to_string(it) + ": " + std::to_string(M) + " rows, " + std::to_string(nb) +
                    " blocks, topk " + std::to_string(topk);
    f.shape(S, M, nb, nb + (int)(rnd() % 3), topk, ratio, (int)(rnd() % 5), (int)(rnd() % 5), (int)(rnd() % 3));
    for (int s = 0; s < S; ++s) f.cu[s + 1] = f.cu[s] + rows[(size_t)s];
    for (int s = 0; s < S; ++s) f.nc[s] = nb - (int)(rnd() % (uint64_t)(nb / 2 + 1)) + (int)(rnd() % 3);
    for (int m = 0; m < M; ++m)
        f.pos[m] = std::max(0, ratio * f.nc[f.seq_of(m)] - (int)(rnd() % (uint64_t)(ratio * nb / 3 + 1)) + (int)(rnd() % 7));
    const unsigned dead = (unsigned)(rnd() % 6), dist = (unsigned)(rnd() % 6), grid = 1 + (unsigned)(rnd() % 400);
    for (auto& v : f.bt) {
        const uint64_t u = rnd();
        const bool d = dead == 1 ? u % 50 == 0 : dead == 2 ? u % 2 == 0 : dead == 3 ? u % 16 != 0
                     : dead == 4 ? u % 100 != 0 : false;
        v = d ? -1 : (int32_t)(u % 1000000);
    }
    for (auto& v : f.sc) {
        const uint64_t u = rnd();
        if (u % 97 == 0) { v = -INFINITY; continue; }
        if (u % 389 == 0) { v = NAN; continue; }
        switch (dist) {
            case 0: v = (float)(u % 1000003) / 1000.0f; break;
            case 1: v = (float)((u >> 8) % grid) * 0.25f; break;
            case 2: v = u % 8 == 0 ? 1.0f + (float)(u % 60000) * 1e-6f : (float)(u % 1000) / 2000.0f; break;
            case 3: v = 1.0f + (float)((u >> 8) % 60000) * 1e-6f; break;
            case 4: v = (u >> 8) % 3 == 0 ? 0.0f : (u >> 8) % 3 == 1 ? -0.0f : -1.0f; break;
            default: v = (float)((u >> 8) % 4999) * 0.01f; break;
        }
    }
    return f;
}

/* A fixture under the signed order, its finite scores negated on odd rows when `flip` and moved
 * by `shift`, so live scores straddle zero or fall below it. */
Fix as_signed(Fix f, float shift, bool flip = false) {
    f.sgn = true;
    f.name = "signed " + f.name;
    if (shift == 0.0f && !flip) return f;          /* as built: keeps -0 apart from +0 */
    for (int m = 0; m < f.M; ++m)
        for (int b = 0; b < f.nb_s; ++b) {
            float& v = f.score(m, b);
            if (!std::isfinite(v)) continue;
            if (flip && m % 2) v = -v;
            v += shift;
        }
    return f;
}

}  // namespace

/* No argument: the fixtures above and the branch check. `sweep N`: N fix_random fixtures, about
 * 0.15 s each. */
int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool sweep = argc > 1 && std::string(argv[1]) == "sweep";
    std::vector<std::function<Fix()>> fixtures;
    if (sweep) {
        const int n = argc > 2 ? std::atoi(argv[2]) : 100;
        for (int i = 0; i < n; ++i) fixtures.push_back([i] { return fix_random(i); });
        for (int i = 0; i < n; ++i) fixtures.push_back([i] { return as_signed(fix_random(i), -1.0f, true); });
    } else {
        fixtures = { fix_decode, fix_prefill, fix_few_live, fix_wide_launch,
                     [] { return fix_split(1, 4); }, [] { return fix_split(2, 4); }, fix_split_sparse, fix_big_topk,
                     [] { return fix_no_split(true); }, [] { return fix_no_split(false); } };
        for (int d = 0; d < 5; ++d) fixtures.push_back([d] { return fix_wide(d); });
        for (int i = 0; i < 150; ++i) fixtures.push_back([i] { return fix_sweep(i); });
        fixtures.push_back([] { return as_signed(fix_decode(), -5.0f); });
        fixtures.push_back([] { return as_signed(fix_decode(), -20.0f); });
        fixtures.push_back([] { return as_signed(fix_few_live(), 0.0f); });
        fixtures.push_back([] { return as_signed(fix_split(2, 4), -6.0f); });
        fixtures.push_back([] { return as_signed(fix_wide_launch(), -1.0f, true); });
        for (int d = 5; d < 8; ++d) fixtures.push_back([d] { return fix_wide(d); });
        for (int i = 0; i < 150; ++i) fixtures.push_back([i] { return as_signed(fix_sweep(i), 0.0f); });
    }
    int fails = 0, launches = 0;
    for (auto& make : fixtures) {
        Fix f = make();
        for (Form form : { kSingle, kWide, kAuto }) { fails += run(f, form); ++launches; }
    }
    if (!sweep) fails += shared_table_case();
    if (!sweep) fails += parser_case();
    fails += masked_read_case();
    if (sweep) {
        for (const auto& [p, c] : g_paths) std::printf("  %5d row(s): %s\n", c, p.c_str());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("qsa_select sweep: %zu fixtures, %d launches, %d failure(s), %.1f s\n", fixtures.size(),
                    launches, fails, sec);
        return fails ? 1 : 0;
    }
    /* Every branch the fixtures are built to reach. */
    static const char* const kRequired[] = {
        "rows of at most topk blocks", "exactly topk live over more blocks", "fewer than topk live over more blocks",
        "a threshold", "a threshold whose ties are cut short",
        "single: keys cached in LDS", "single: keys read from global memory",
        "wide: the sampled pass, rows 16-byte aligned", "wide: the sampled pass, rows not 16-byte aligned",
        "wide: the sampled list", "wide: the sampled list, holding every live key",
        "wide: a sample that fell back", "wide: merged lists",
        "wide: the threshold's bin in LDS", "wide: the threshold's bin over global memory",
        "split: topk or fewer live, every one taken", "split: no chunk over topk, more than topk live",
        "split: a chunk over topk", "split: exactly topk candidates, from a chunk over topk",
        "signed: a negative threshold",
    };
    int missing = 0;
    for (const char* p : kRequired)
        if (!g_paths.count(p)) { std::printf("FAIL no row reaches: %s\n", p); ++missing; }
    for (const auto& [p, c] : g_paths) std::printf("  %5d row(s): %s\n", c, p.c_str());
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("qsa_select: %zu fixtures, %d launches, %d failure(s), %d branch(es) unreached, %.1f s\n",
                fixtures.size(), launches, fails, missing, sec);
    return fails || missing ? 1 : 0;
}
