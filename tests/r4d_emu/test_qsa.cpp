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
 * One case is not a kernel against the port: `shared_table_case` takes the kernel's own selections
 * for a tile of query rows and checks that their union in block order plus a per-row mask gives
 * every row back (docs/QSA.md, "The emit is one run in block order"), reporting how much the rows
 * share and where a row's list leaves block order.
 *
 * Fixtures marked signed run with the op's `signed` parameter: the kernels' signed key order,
 * against the port's signed ranking, over negative, mixed and the sweep's odd scores. The unsigned
 * fixtures are the same as without the parameter. */
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r4d_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
void ref_select(const Fix& f, std::vector<int32_t>& se, std::vector<int32_t>& ns, std::vector<int32_t>& su) {
    const int64_t topk = f.topk, ratio = f.ratio;
    se.assign((size_t)f.M * (topk + 1), 0); ns.assign((size_t)f.M, 0); su.assign((size_t)f.M, 0);
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
        int64_t out = 0;
        for (int64_t b : take) row[out++] = f.bt[(size_t)(s * f.t_ld + b)];
        if (n < f.nb() && (p + 1) % ratio != 0) {
            const int32_t part = f.bt[(size_t)(s * f.t_ld + n)];
            if (part >= 0) row[out++] = part;
        }
        for (int64_t i = out; i <= topk; ++i) row[i] = -1;
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
    std::vector<int32_t> se, ns, su;
};

/* One launch of `form` against the port. Returns the failures (0 or 1). */
int run(Fix& f, Form form, SelOut* out = nullptr) {
    std::vector<int32_t> want, want_n, want_su;
    ref_select(f, want, want_n, want_su);
    const int M = f.M, W = f.topk + 1;
    std::vector<int32_t> se((size_t)(M + 1) * f.o_ld, kSentinel), ns((size_t)M + 1, kSentinel), su((size_t)M + 1, kSentinel);
    RadTensor t[8] = {
        tens(f.sc.data(), RAD_F32, {M, f.nb_s}, f.s_ld), tens(f.bt.data(), RAD_I32, {f.S, f.nb_t}, f.t_ld),
        tens(f.nc.data(), RAD_I32, {f.S}),             tens(f.pos.data(), RAD_I32, {M}),
        tens(f.cu.data(), RAD_I32, {f.S + 1}),         tens(se.data(), RAD_I32, {M, W}, f.o_ld),
        tens(ns.data(), RAD_I32, {M}),                 tens(su.data(), RAD_I32, {M}),
    };
    RadParam p[6]{};
    p[0] = {"M", RAD_P_INT, M}; p[1] = {"blocks", RAD_P_INT, f.nb()}; p[2] = {"topk", RAD_P_INT, f.topk};
    p[3] = {"ratio", RAD_P_INT, f.ratio}; p[4] = {"seqs", RAD_P_INT, f.S}; p[5] = {"signed", RAD_P_INT, 1};
    RadArgs a{}; a.t = t; a.n_t = 8; a.p = p; a.n_p = f.sgn ? 6 : 5;
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
        const int32_t wn = m < M ? want_n[(size_t)m] : kSentinel, wsu = m < M ? want_su[(size_t)m] : kSentinel;
        if (!diff && ns[(size_t)m] == wn && su[(size_t)m] == wsu) continue;
        if (++bad_rows > 4) continue;
        std::printf("FAIL %-48s %-6s row %d (n %d): nsel %d want %d, seqused %d want %d", f.name.c_str(), fn, m,
                    m < M ? f.n_of(m) : -1, ns[(size_t)m], wn, su[(size_t)m], wsu);
        if (diff) {
            const int32_t w = m < M && first < W ? want[(size_t)m * W + first] : kSentinel;
            std::printf(", %d slot(s) differ, first slot %d: %d want %d", diff, first, se[(size_t)m * f.o_ld + first], w);
        }
        std::printf("\n");
    }
    if (out) {
        out->se.assign(se.begin(), se.begin() + (size_t)M * f.o_ld);
        out->ns.assign(ns.begin(), ns.begin() + M);
        out->su.assign(su.begin(), su.begin() + M);
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
    f.page(0, nb) = -1;                                      /* nothing past the history */
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

/* One table per tile, from the kernel's own selections, recovered row by row. Returns the failures.
 *
 * What is required: every row's selection is a subset of the shared table, so the table plus the
 * row's mask is that row's selection and nothing else -- its count, its pages, its `seqused`. What
 * is measured and reported instead: the rows' lists are in block order except for the threshold
 * bin, whose ties the emit places at the tail (the tie rule picks them, so they cannot sit in
 * place). A merged read walks the table, so those entries move; for a row already in block order
 * the table reproduces it byte for byte, and the count of moved entries is the size of the change
 * in the attention's accumulation order -- zero for every row below the budget. */
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
        /* Each row's table: its pages and the partial page the port appends last. */
        std::vector<std::vector<int32_t>> pages((size_t)f.M);
        std::vector<int32_t> part((size_t)f.M, -1);
        std::vector<char> already((size_t)f.M, 0);
        bool partial_seen = false, partial_absent = false;
        int64_t total = 0, tail = 0;
        int in_order = 0, breaks = 0, rows_broken = 0;
        for (int m = 0; m < f.M; ++m) {
            const int nsel = got.ns[(size_t)m];
            const int32_t* row = &got.se[(size_t)m * f.o_ld];
            const bool has = nsel > 0 && row_has_partial(f, m);
            partial_seen |= has;
            partial_absent |= !has;
            for (int i = 0; i < nsel; ++i) {
                if (has && i == nsel - 1) { part[(size_t)m] = row[i]; continue; }
                pages[(size_t)m].push_back(row[i]);
            }
            total += (int64_t)pages[(size_t)m].size();
            /* Block order, which the merge needs of a row: the emit is a run in block order and the
             * threshold bin's run after it, so a descent count says how far off it is and the last
             * run's length says how many entries the merge moves. */
            const std::vector<int32_t>& p = pages[(size_t)m];
            int last = -1, d = 0;
            for (size_t i = 0; i + 1 < p.size(); ++i)
                if (p[i] > p[i + 1]) { ++d; last = (int)i; }
            breaks += d;
            rows_broken += d > 1;
            already[(size_t)m] = last < 0;
            in_order += already[(size_t)m];
            tail += last < 0 ? 0 : (int64_t)p.size() - 1 - last;
            std::sort(pages[(size_t)m].begin(), pages[(size_t)m].end());
        }
        /* The shared table: the sorted union of the rows' pages -- the stable mixture, and the one
         * order in which every row's selection can be read. */
        std::vector<int32_t> tab;
        for (const auto& p : pages) tab.insert(tab.end(), p.begin(), p.end());
        std::sort(tab.begin(), tab.end());
        tab.erase(std::unique(tab.begin(), tab.end()), tab.end());
        /* Recover each row: walk the table, take its members, then its own partial page. */
        int bad_rows = 0;
        for (int m = 0; m < f.M; ++m) {
            std::vector<int32_t> rec;
            size_t i = 0;
            for (int32_t b : tab) {
                while (i < pages[(size_t)m].size() && pages[(size_t)m][i] < b) ++i;
                if (i < pages[(size_t)m].size() && pages[(size_t)m][i] == b) rec.push_back(b);
            }
            if (part[(size_t)m] >= 0) rec.push_back(part[(size_t)m]);
            const int32_t su = rec.empty() ? 0
                             : (int32_t)(((int)rec.size() - 1) * f.ratio + f.pos[m] % f.ratio + 1);
            bool same = (int)rec.size() == got.ns[(size_t)m] && got.su[(size_t)m] == su;
            /* The row's padding, and -- for a list already in block order -- the whole row, byte
             * for byte as the kernel wrote it. A row that was not keeps its selection, but the
             * merge reads it in the table's order. */
            for (int k = (int)rec.size(); same && k <= f.topk; ++k)
                same = got.se[(size_t)m * f.o_ld + k] == -1;
            for (int k = 0; already[(size_t)m] && same && k < (int)rec.size(); ++k)
                same = got.se[(size_t)m * f.o_ld + k] == rec[(size_t)k];
            if (!same && ++bad_rows <= 3)
                std::printf("FAIL %-46s row %d: the shared table does not recover this row\n",
                            f.name.c_str(), m);
        }
        const int64_t per_row = total / (int64_t)f.M;
        const double factor = tab.empty() ? 0.0 : (double)total / (double)tab.size();
        std::printf("  %-46s %d rows x %lld pages -> %zu blocks (%.1fx fewer), mask %zu B/row, "
                    "%d/%d rows recovered\n", f.name.c_str(), f.M, (long long)per_row, tab.size(),
                    factor, (tab.size() + 7) / 8, f.M - bad_rows, f.M);
        std::printf("  %-46s %d/%d rows already in block order, %.1f entries in the tail run, %d "
                    "break(s)\n", "", in_order, f.M, (double)tail / f.M, breaks);
        /* The emit is one run in block order and the threshold bin's run after it: that is what
         * makes the table's order the rows' order apart from one group, so check it. */
        if (rows_broken) {
            std::printf("FAIL %-46s %d row(s) not one run in block order plus the tail\n",
                        f.name.c_str(), rows_broken);
            ++fails;
        }
        if (!partial_seen || !partial_absent) {
            std::printf("FAIL %-46s partial page seen %d, absent %d\n", f.name.c_str(),
                        (int)partial_seen, (int)partial_absent);
            ++fails;
        }
        fails += bad_rows ? 1 : 0;
    }
    std::printf("shared table: %zu tiles, %d failure(s)\n", fx.size(), fails);
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
