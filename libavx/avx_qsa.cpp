/* avx_qsa.cpp -- the QSA indexer's score, selection, tail store and work list, and DFlash2's
 * grouped convolution (docs/OPS.md, "Sparse attention (QSA)" and "Draft heads").
 *
 * libref/ref_qsa.cpp is the contract: the same refusals in the same order, the same elements left
 * unwritten and the same order of every output. `cu` holds absolute row indices; a row belongs to
 * the first sequence whose span holds it, and cu[0] need not be 0.
 *
 *   qsa_score   A tile of up to kQT query rows of one sequence is scored against VF_N block keys
 *               at a time. Each (row, head, block) dot is a vector accumulator over the head's
 *               lanes, and a fixed transpose-and-add tree turns VF_N of them into one vector with
 *               one block per lane, so the relu, the sum over heads and the store are vector work.
 *               The keys are read once per tile, which at decode is once per sequence. The dots
 *               sum in vector order (libref's order at the scalar level); the heads in order.
 *   qsa_select  Linear time per row: an order key per block (the bits of max(score, 0) plus one,
 *               0 for a dead block; with `signed` the float's total order over all 32 bits), a
 *               histogram of the keys' top bits, an exact select among the keys in the bin that
 *               holds the topk-th largest, and one emitting pass in block order. The ids match
 *               libref's byte for byte.
 *   qsa_tail_store, qsa_work   Row copies through f32, as libref copies, so every value lands
 *               with the bits libref's would.
 *   dflash_conv Channels across lanes, in blocks of columns that walk their rows, so a block's base
 *               kernel and its slice of the row above stay in L1; one pass in registers a row.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <functional>
#include <type_traits>

using namespace avx;

namespace {

int64_t cu_at(const RadTensor* cu, int64_t i) { return ldi(cu, offlin(cu, i)); }

/* Offset of entry `i` of a rank-1 operand, or of column 0 of row `i` of a rank-2 one (a state
 * index row's committed slot). */
int64_t col0(const RadTensor* t, int64_t i) {
    return t->rank >= 2 ? i * t->stride[0] : offlin(t, i);
}

/* Whether the dimensions after the first are dense, so a row of a paged store is
 * `numel / shape[0]` contiguous values from `row * stride[0]`. */
bool rows_dense(const RadTensor* t) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return false;
        acc *= t->shape[i];
    }
    return true;
}

/* seq[m] for every m < M: the first sequence whose cu span holds row m, or `none` (libref's
 * seq_holding), filled from the spans instead of scanning every span for every row. */
void seq_rows(const RadTensor* cu, int64_t n_seq, int64_t M, int64_t none, int64_t* seq) {
    const int64_t unset = INT64_MIN;
    std::fill(seq, seq + M, unset);
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t lo = std::max<int64_t>(cu_at(cu, s), 0);
        const int64_t hi = std::min<int64_t>(cu_at(cu, s + 1), M);
        for (int64_t m = lo; m < hi; ++m)
            if (seq[m] == unset) seq[m] = s;
    }
    for (int64_t m = 0; m < M; ++m)
        if (seq[m] == unset) seq[m] = none;
}

/* `n` elements of `t` from `base`, `st` apart, as f32, and back. Whole-byte dtypes convert a row
 * at a time; the conversions are libref's bit for bit either way. */
void widen(const RadTensor* t, int64_t base, int64_t st, int64_t n, float* dst) {
    if (st == 1 && rad_dtype_bits(t->dtype) >= 8) {
        row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), dst, n);
        return;
    }
    for (int64_t i = 0; i < n; ++i) dst[i] = ldt(t, base + i * st);
}

void narrow(const RadTensor* t, int64_t base, int64_t st, int64_t n, const float* src) {
    if (st == 1 && rad_dtype_bits(t->dtype) >= 8) {
        row_from_f32(t->dtype, src, (void*)byte_at(t->data, t->dtype, base), n);
        return;
    }
    for (int64_t i = 0; i < n; ++i) stt(t, base + i * st, src[i]);
}

/* ================================================================== qsa_score */

/* Query rows a tile: the key history is read once per tile, and kQT * heads rows of f32 queries
 * stay in L1 beside VF_N widened keys (16 KiB and 8 KiB at Flash-Next's 4 heads of 128). */
enum { kQT = 8 };
/* Blocks a work unit. Units are ordered chunk-major and handed out in order, so the threads work
 * through one chunk's keys across the tiles of a prefill chunk while they are in L3. */
enum { kChunk = 256 };
/* Blocks the software prefetch runs ahead of the keys being widened. */
enum { kAhead = 32 };

/* lane_sums leaves the total of accumulator slot i in lane kSlot[i]; slot i is loaded with block
 * kSlot[i] of the group, so lane j holds block j. */
#if AVX_LEVEL == 3
const int kSlot[16] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };
#elif AVX_LEVEL >= 1
const int kSlot[8] = { 0, 4, 1, 5, 2, 6, 3, 7 };
#else
const int kSlot[1] = { 0 };
#endif

/* The VF_N horizontal sums of a[0 .. VF_N), one a lane, by a fixed tree: each level halves every
 * vector and adds pairs of them, so a level costs one vector of adds per output vector. */
static inline vf lane_sums(const vf* a) {
#if AVX_LEVEL == 3
    __m512 s[8], t[4];
    for (int i = 0; i < 8; ++i)
        s[i] = _mm512_add_ps(_mm512_shuffle_f32x4(a[2 * i], a[2 * i + 1], 0x44),
                             _mm512_shuffle_f32x4(a[2 * i], a[2 * i + 1], 0xEE));
    for (int i = 0; i < 4; ++i)
        t[i] = _mm512_add_ps(_mm512_shuffle_f32x4(s[2 * i], s[2 * i + 1], 0x88),
                             _mm512_shuffle_f32x4(s[2 * i], s[2 * i + 1], 0xDD));
    const __m512 u0 = _mm512_add_ps(_mm512_shuffle_ps(t[0], t[1], 0x44),
                                    _mm512_shuffle_ps(t[0], t[1], 0xEE));
    const __m512 u1 = _mm512_add_ps(_mm512_shuffle_ps(t[2], t[3], 0x44),
                                    _mm512_shuffle_ps(t[2], t[3], 0xEE));
    return _mm512_add_ps(_mm512_shuffle_ps(u0, u1, 0x88), _mm512_shuffle_ps(u0, u1, 0xDD));
#elif AVX_LEVEL >= 1
    __m256 s[4];
    for (int i = 0; i < 4; ++i)
        s[i] = _mm256_add_ps(_mm256_permute2f128_ps(a[2 * i], a[2 * i + 1], 0x20),
                             _mm256_permute2f128_ps(a[2 * i], a[2 * i + 1], 0x31));
    const __m256 t0 = _mm256_add_ps(_mm256_shuffle_ps(s[0], s[1], 0x44),
                                    _mm256_shuffle_ps(s[0], s[1], 0xEE));
    const __m256 t1 = _mm256_add_ps(_mm256_shuffle_ps(s[2], s[3], 0x44),
                                    _mm256_shuffle_ps(s[2], s[3], 0xEE));
    return _mm256_add_ps(_mm256_shuffle_ps(t0, t1, 0x88), _mm256_shuffle_ps(t0, t1, 0xDD));
#else
    return a[0];
#endif
}

/* Lane j: the dot of the query row `q` with block key j of the group in `kf` (slot order, rows
 * `np` apart). `np` is a multiple of 16 and the tails of both rows are zero. */
static inline vf dot_group(const float* q, const float* kf, int64_t np) {
    vf acc[VF_N];
    for (int i = 0; i < VF_N; ++i) acc[i] = vf_zero();
    for (int64_t c = 0; c < np; c += VF_N) {
        const vf vq = vf_load(q + c);
#pragma GCC unroll 16
        for (int i = 0; i < VF_N; ++i) acc[i] = vf_fma(vq, vf_load(kf + i * np + c), acc[i]);
    }
    return lane_sums(acc);
}

struct ScoreTile { int64_t m0, rows, s, nbs; };

struct ScoreJob {
    const RadTensor *q, *bk, *bt, *o;
    int64_t n, np, heads;
    int64_t q_rs, q_cs, pg_rs, bt_rs, bt_cs, o_rs, o_cs;
    int64_t esz;          /* bytes an element of bk, 0 for a packed dtype */
    float rn;
};

/* Scores blocks [b0, b1) of one tile. */
static void score_unit(const ScoreJob& j, const ScoreTile& t, int64_t b0, int64_t b1) {
    const int64_t n = j.n, np = j.np, R = t.rows * j.heads;
    float* qf = scratch_f32((size_t)(R * np));
    float* kf = scratch_f32_b((size_t)(VF_N * np));
    for (int64_t r = 0; r < R; ++r) {
        const int64_t m = t.m0 + r / j.heads, h = r % j.heads;
        float* d = qf + r * np;
        widen(j.q, m * j.q_rs + h * n * j.q_cs, j.q_cs, n, d);
        std::fill(d + n, d + np, 0.0f);
    }
    for (int i = 0; i < VF_N; ++i) std::fill(kf + i * np + n, kf + (i + 1) * np, 0.0f);
    const bool f32_out = j.o->dtype == RAD_F32;
    const uint8_t* bkb = (const uint8_t*)j.bk->data;
    const int64_t bt_row = t.s * j.bt_rs;
    alignas(64) float lanes[VF_N];
    /* Software prefetch kAhead blocks ahead of the group being widened, from the unit's first
     * block: a block table need not be in page order, so the hardware prefetcher cannot follow
     * it, and a unit's head would otherwise wait on DRAM. */
    int64_t pf = b0;
    auto prefetch_to = [&](int64_t lim) {
        for (lim = std::min(lim, b1); pf < lim; ++pf) {
            const int64_t p = ldi(j.bt, bt_row + pf * j.bt_cs);
            if (p < 0 || j.esz == 0) continue;
            const uint8_t* at = bkb + p * j.pg_rs * j.esz;
            for (int64_t off = 0; off < n * j.esz; off += 64) __builtin_prefetch(at + off);
        }
    };
    prefetch_to(b0 + kAhead);
    for (int64_t g = b0; g < b1; g += VF_N) {
        const int nv = (int)std::min<int64_t>(VF_N, b1 - g);
        prefetch_to(g + VF_N + kAhead);
        int64_t pg[VF_N];
        for (int L = 0; L < VF_N; ++L)
            pg[L] = L < nv ? ldi(j.bt, bt_row + (g + L) * j.bt_cs) : -1;
        for (int i = 0; i < VF_N; ++i) {
            const int64_t p = pg[kSlot[i]];
            float* d = kf + i * np;
            if (p >= 0) widen(j.bk, p * j.pg_rs, 1, n, d);
            else        std::fill(d, d + n, 0.0f);
        }
        for (int64_t qi = 0; qi < t.rows; ++qi) {
            vf s = vf_zero();
            for (int64_t h = 0; h < j.heads; ++h)
                s = vf_add(s, vf_max(dot_group(qf + (qi * j.heads + h) * np, kf, np), vf_zero()));
            vf_store(lanes, vf_div(s, vf_set1(j.rn)));
            const int64_t ob = (t.m0 + qi) * j.o_rs + g * j.o_cs;
            if (f32_out) {
                float* o = (float*)j.o->data;
                for (int L = 0; L < nv; ++L) o[ob + L * j.o_cs] = pg[L] < 0 ? -INFINITY : lanes[L];
            } else {
                for (int L = 0; L < nv; ++L)
                    stt(j.o, ob + L * j.o_cs, pg[L] < 0 ? -INFINITY : lanes[L]);
            }
        }
    }
}

}  /* namespace */

/* ------------------------------------------------------------------ qsa_score */
/* score[m][b] = (sum over heads h of relu(q[m][h] . bk[bt[seq(m)][b]])) / sqrt(n) for b below
 * min(nc[seq], the table's width, the score plane's width); a negative page id scores -inf.
 * Columns past the bound, and rows no sequence holds, are not written. */
AVX_KERNEL(qsa_score) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *bk = &a->t[1], *bt = &a->t[2], *cu = &a->t[3];
    const RadTensor *nc = &a->t[4], *o = &a->t[5];
    const int64_t n = p_int(a, "n", 0), heads = p_int(a, "heads", 0);
    if (n <= 0 || heads <= 0) return RAD_E_SHAPE;
    if (bt->rank != 2 || o->rank != 2 || bk->rank < 2) return RAD_E_SHAPE;
    const int64_t M = o->shape[0], nb = std::min(bt->shape[1], o->shape[1]);
    const int64_t n_seq = numel(cu) - 1, pages = bk->shape[0];
    if (M <= 0 || nb <= 0 || n_seq <= 0 || pages <= 0) return RAD_E_SHAPE;
    if (bt->shape[0] < n_seq || numel(nc) < n_seq || rows_of(q, heads * n) < M) return RAD_E_SHAPE;
    if (numel(bk) / pages != n) return RAD_E_SHAPE;
    if (!rows_dense(bk)) return RAD_E_STRIDE;
    const int64_t bt_rs = bt->stride[0], bt_cs = bt->stride[1];
    /* Every page id the launch will read must name a page of `bk`. */
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t nbs = std::min(ldi(nc, offlin(nc, s)), nb);
        for (int64_t b = 0; b < nbs; ++b)
            if (ldi(bt, s * bt_rs + b * bt_cs) >= pages) return RAD_E_INVAL;
    }

    ScoreJob j{};
    j.q = q; j.bk = bk; j.bt = bt; j.o = o;
    j.n = n; j.np = (n + 15) & ~(int64_t)15; j.heads = heads;
    j.q_rs = row_stride(q, heads * n); j.q_cs = laststride(q);
    j.pg_rs = bk->stride[0]; j.bt_rs = bt_rs; j.bt_cs = bt_cs;
    j.o_rs = o->stride[0]; j.o_cs = o->stride[1];
    j.esz = rad_dtype_bits(bk->dtype) / 8;
    j.rn = std::sqrt((float)n);

    /* Tiles: runs of up to kQT consecutive rows of one sequence. One allocation, as a second
     * scratch_raw call could move the first. */
    int64_t* seq = (int64_t*)scratch_raw((size_t)M * (sizeof(int64_t) + sizeof(ScoreTile)) + 64);
    ScoreTile* tiles = (ScoreTile*)(seq + M);
    seq_rows(cu, n_seq, M, -1, seq);
    int64_t nt = 0, widest = 0, work = 0;
    for (int64_t m = 0; m < M;) {
        const int64_t s = seq[m];
        int64_t e = m + 1;
        while (e < M && e - m < kQT && seq[e] == s) ++e;
        if (s >= 0) {
            const int64_t nbs = std::min(ldi(nc, offlin(nc, s)), nb);
            if (nbs > 0) {
                tiles[nt++] = ScoreTile{ m, e - m, s, nbs };
                widest = std::max(widest, nbs);
                work += (e - m) * heads * n * nbs;
            }
        }
        m = e;
    }
    const int64_t nch = (widest + kChunk - 1) / kChunk, units = nt * nch;
    /* Dynamic: a unit's arithmetic does not depend on the thread that runs it, so the output is
     * the same bytes for any schedule, and a preempted thread does not hold up the rest. */
#ifdef _OPENMP
    AVX_PRAGMA(omp parallel for schedule(dynamic, 1) if(work >= ((int64_t)1 << 19) ||
               avx_par_forced()) num_threads(avx_team_size()))
#endif
    for (int64_t u = 0; u < units; ++u) {
        const ScoreTile& t = tiles[u % nt];
        const int64_t b0 = (u / nt) * kChunk, b1 = std::min(b0 + kChunk, t.nbs);
        if (b0 < b1) score_unit(j, t, b0, b1);
    }
    return RAD_OK;
}

/* ================================================================== qsa_select */
namespace {

/* Bins of the first selection pass: the top 10 bits of a key, which is below 2^31 (a signed
 * key uses all 32 bits, so its bins sit one bit higher). */
enum { kBins = 1024, kBinShift = 21, kBinShiftSigned = 22 };

struct SelJob {
    const RadTensor *sc, *bt, *nc, *po, *se, *ns, *su;
    const RadTensor* sb;               /* `selb`: the same entries as block indices, optional */
    const int64_t* seq;
    int64_t nb, topk, ratio;
    int64_t s_rs, s_cs, bt_rs, bt_cs, o_rs, o_cs, b_rs, b_cs;
    bool sgn;                          /* the `signed` order */
};

/* The order key of each of the row's first `n` blocks: bits(max(score, 0)) + 1 for a live block
 * (page id >= 0 and score > -1e30), 0 for a dead one. Non-negative floats order as their bits, so
 * keys order as libref's values; max(score, 0) maps -0 to +0 as libref's `v > 0 ? v : 0` does.
 * The signed order makes a block with a finite score live and keys it by the float's total order
 * (sign bit flipped for a positive value, every bit for a negative one), -0 keyed as +0, so keys
 * order as libref's signed values; no finite score keys to 0. Returns the live count. */
static int64_t order_keys(const SelJob& j, int64_t m, int64_t s, int64_t n, uint32_t* key) {
    const int64_t sb = m * j.s_rs, bb = s * j.bt_rs;
    int64_t live = 0, b = 0;
#if AVX_LEVEL >= 2
    if (!j.sgn && j.sc->dtype == RAD_F32 && j.s_cs == 1 && j.bt->dtype == RAD_I32 && j.bt_cs == 1) {
        const float* v = (const float*)j.sc->data + sb;
        const int32_t* p = (const int32_t*)j.bt->data + bb;
#if AVX_LEVEL == 3
        const __m512 lo = _mm512_set1_ps(-1e30f);
        const __m512i one = _mm512_set1_epi32(1);
        for (; b + 16 <= n; b += 16) {
            const __m512 x = _mm512_loadu_ps(v + b);
            const __m512i pg = _mm512_loadu_si512((const void*)(p + b));
            const __mmask16 lv = _mm512_cmp_ps_mask(x, lo, _CMP_GT_OQ) &
                                 _mm512_cmpge_epi32_mask(pg, _mm512_setzero_si512());
            const __m512i k = _mm512_add_epi32(
                _mm512_castps_si512(_mm512_max_ps(x, _mm512_setzero_ps())), one);
            _mm512_storeu_si512((void*)(key + b), _mm512_maskz_mov_epi32(lv, k));
            live += __builtin_popcount((unsigned)lv);
        }
#else
        const __m256 lo = _mm256_set1_ps(-1e30f);
        const __m256i one = _mm256_set1_epi32(1), neg = _mm256_set1_epi32(-1);
        for (; b + 8 <= n; b += 8) {
            const __m256 x = _mm256_loadu_ps(v + b);
            const __m256i pg = _mm256_loadu_si256((const __m256i*)(p + b));
            const __m256 lv = _mm256_and_ps(_mm256_cmp_ps(x, lo, _CMP_GT_OQ),
                                            _mm256_castsi256_ps(_mm256_cmpgt_epi32(pg, neg)));
            const __m256i k = _mm256_add_epi32(
                _mm256_castps_si256(_mm256_max_ps(x, _mm256_setzero_ps())), one);
            _mm256_storeu_si256((__m256i*)(key + b),
                                _mm256_and_si256(k, _mm256_castps_si256(lv)));
            live += __builtin_popcount((unsigned)_mm256_movemask_ps(lv));
        }
#endif
    }
#endif
    if (j.sgn) {
        for (; b < n; ++b) {
            const float v = ldt(j.sc, sb + b * j.s_cs);
            if (ldi(j.bt, bb + b * j.bt_cs) < 0 || !std::isfinite(v)) { key[b] = 0; continue; }
            uint32_t u = 0;
            if (v != 0.0f) std::memcpy(&u, &v, 4);
            key[b] = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
            ++live;
        }
        return live;
    }
    for (; b < n; ++b) {
        const float v = ldt(j.sc, sb + b * j.s_cs);
        if (ldi(j.bt, bb + b * j.bt_cs) < 0 || !(v > -1e30f)) { key[b] = 0; continue; }
        const float x = v > 0.0f ? v : 0.0f;
        uint32_t u;
        std::memcpy(&u, &x, 4);
        key[b] = u + 1u;
        ++live;
    }
    return live;
}

/* The topk-th largest key of a row with more than `topk` live keys, and how many keys are above
 * it. A histogram of the top bits finds the bin holding it; the keys in that bin are gathered and
 * the exact value taken among them. `cand` holds up to n keys. `sgn`: the keys are signed-order
 * keys, which use all 32 bits. */
static uint32_t threshold(const uint32_t* key, int64_t n, int64_t topk, bool sgn, uint32_t* hist,
                          uint32_t* cand, int64_t* above_out) {
    const int shift = sgn ? (int)kBinShiftSigned : (int)kBinShift;
    /* Four interleaved histograms, so runs of keys in one bin do not serialise on one counter. */
    std::memset(hist, 0, 4 * kBins * sizeof(uint32_t));
    int64_t b = 0;
    for (; b + 4 <= n; b += 4) {
        ++hist[0 * kBins + (key[b] >> shift)];
        ++hist[1 * kBins + (key[b + 1] >> shift)];
        ++hist[2 * kBins + (key[b + 2] >> shift)];
        ++hist[3 * kBins + (key[b + 3] >> shift)];
    }
    for (; b < n; ++b) ++hist[key[b] >> shift];
    for (int i = 0; i < kBins; ++i)
        hist[i] += hist[kBins + i] + hist[2 * kBins + i] + hist[3 * kBins + i];
    /* Bin 0 is never counted: it is the last one left, and holds the dead keys (0) beside the
     * smallest live ones, which the gather below tells apart. */
    int64_t above = 0;
    int bin = kBins - 1;
    for (; bin > 0; --bin) {
        if (above + (int64_t)hist[bin] >= topk) break;
        above += hist[bin];
    }
    /* The keys of the bin, i.e. in [lo, hi], with key 0 (dead) left out of bin 0. Few keys fall in
     * one bin, so the vector forms test a vector and walk only its hits. */
    const uint32_t lo = std::max<uint32_t>((uint32_t)bin << shift, 1u);
    const uint32_t hi = (((uint32_t)bin + 1u) << shift) - 1u;
    int64_t nc = 0;
    b = 0;
#if AVX_LEVEL == 3
    const __m512i vlo = _mm512_set1_epi32((int)lo), vhi = _mm512_set1_epi32((int)hi);
    for (; b + 16 <= n; b += 16) {
        const __m512i k = _mm512_loadu_si512((const void*)(key + b));
        const __mmask16 in = _mm512_cmpge_epu32_mask(k, vlo) & _mm512_cmple_epu32_mask(k, vhi);
        _mm512_mask_compressstoreu_epi32((void*)(cand + nc), in, k);
        nc += __builtin_popcount((unsigned)in);
    }
#elif AVX_LEVEL == 2
    /* Keys are below 2^31, so the signed compares order them; signed-order keys take the scalar
     * loop. */
    const __m256i vlo = _mm256_set1_epi32((int)lo - 1), vhi = _mm256_set1_epi32((int)hi + 1);
    for (; !sgn && b + 8 <= n; b += 8) {
        const __m256i k = _mm256_loadu_si256((const __m256i*)(key + b));
        const __m256i in = _mm256_and_si256(_mm256_cmpgt_epi32(k, vlo), _mm256_cmpgt_epi32(vhi, k));
        unsigned hit = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(in));
        while (hit) {
            cand[nc++] = key[b + __builtin_ctz(hit)];
            hit &= hit - 1;
        }
    }
#endif
    for (; b < n; ++b) {
        const uint32_t k = key[b];
        if (k >= lo && k <= hi) cand[nc++] = k;
    }
    const int64_t want = topk - above;   /* 1 .. nc */
    std::nth_element(cand, cand + (want - 1), cand + nc, std::greater<uint32_t>());
    const uint32_t thr = cand[want - 1];
    for (int64_t i = 0; i < nc; ++i) above += cand[i] > thr;
    *above_out = above;
    return thr;
}

/* Writes the page ids of the keys above `thr` and the first `ties` keys equal to it in ONE
 * ascending pass. A tie can sit below a block above the threshold, so the two sets interleave; the
 * order is the contract (docs/QSA.md), because the attention sums a query's pages in table order
 * and a table shared by a query tile is their union in the same order. */
static void emit(const SelJob& j, const uint32_t* key, int64_t n, int64_t s, int64_t m,
                 uint32_t thr, int64_t gt, int64_t ties, int32_t* row) {
    const int64_t bb = s * j.bt_rs;
    int64_t o = 0, ot = 0, b = 0;
    auto one = [&](int64_t i) {
        const uint32_t k = key[i];
        if (k > thr || (k == thr && ot < ties)) {
            if (k == thr) ++ot;
            row[o++] = (int32_t)ldi(j.bt, bb + i * j.bt_cs);
            if (j.sb) st_int_dt(j.sb->dtype, j.sb->data, m * j.b_rs + (o - 1) * j.b_cs, i);
        }
    };
#if AVX_LEVEL == 3
    const __m512i vt = _mm512_set1_epi32((int)thr);
    for (; b + 16 <= n && o < gt + ties; b += 16) {
        const __m512i k = _mm512_loadu_si512((const void*)(key + b));
        unsigned hit = (unsigned)(_mm512_cmpge_epu32_mask(k, vt));
        while (hit) {
            one(b + __builtin_ctz(hit));
            hit &= hit - 1;
        }
    }
#elif AVX_LEVEL == 2
    /* Keys are below 2^31, so the signed compare orders them; signed-order keys take the scalar
     * loop. */
    const __m256i vt = _mm256_set1_epi32((int)thr - 1);
    for (; !j.sgn && b + 8 <= n && o < gt + ties; b += 8) {
        const __m256i k = _mm256_loadu_si256((const __m256i*)(key + b));
        unsigned hit = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(k, vt)));
        while (hit) {
            one(b + __builtin_ctz(hit));
            hit &= hit - 1;
        }
    }
#endif
    for (; b < n && o < gt + ties; ++b) one(b);
}

static void select_row(const SelJob& j, int64_t m) {
    const int64_t s = j.seq[m];
    const int64_t p = ldi(j.po, offlin(j.po, m));
    const int64_t n = std::min(std::min(ldi(j.nc, offlin(j.nc, s)), j.nb), (p + 1) / j.ratio);
    const int64_t topk = j.topk;
    uint32_t* key = (uint32_t*)scratch_f32((size_t)std::max<int64_t>(n, 1));
    int32_t* row = (int32_t*)scratch_f32_b((size_t)(4 * kBins + topk + 1));
    uint32_t* hist = (uint32_t*)(row + topk + 1);
    const int64_t live = n > 0 ? order_keys(j, m, s, n, key) : 0;
    int64_t out = 0;
    if (live <= topk) {
        const int64_t bb = s * j.bt_rs;
        for (int64_t b = 0; b < n && out < live; ++b)
            if (key[b]) {
                if (j.sb) st_int_dt(j.sb->dtype, j.sb->data, m * j.b_rs + out * j.b_cs, b);
                row[out++] = (int32_t)ldi(j.bt, bb + b * j.bt_cs);
            }
    } else {
        uint32_t* cand = (uint32_t*)scratch_f32_c((size_t)n);
        int64_t gt = 0;
        const uint32_t thr = threshold(key, n, topk, j.sgn, hist, cand, &gt);
        emit(j, key, n, s, m, thr, gt, topk - gt, row);
        out = topk;
    }
    if (n < j.nb && (p + 1) % j.ratio != 0) {
        const int64_t part = ldi(j.bt, s * j.bt_rs + n * j.bt_cs);
        if (part >= 0) {
            if (j.sb) st_int_dt(j.sb->dtype, j.sb->data, m * j.b_rs + out * j.b_cs, n);
            row[out++] = (int32_t)part;
        }
    }
    for (int64_t i = out; i <= topk; ++i) {
        row[i] = -1;
        if (j.sb) st_int_dt(j.sb->dtype, j.sb->data, m * j.b_rs + i * j.b_cs, -1);
    }
    const int64_t ob = m * j.o_rs;
    if (j.se->dtype == RAD_I32 && j.o_cs == 1) {
        std::memcpy((int32_t*)j.se->data + ob, row, (size_t)(topk + 1) * sizeof(int32_t));
    } else {
        for (int64_t i = 0; i <= topk; ++i)
            st_int_dt(j.se->dtype, j.se->data, ob + i * j.o_cs, row[i]);
    }
    st_int_dt(j.ns->dtype, j.ns->data, offlin(j.ns, m), out);
    st_int_dt(j.su->dtype, j.su->data, offlin(j.su, m),
              out == 0 ? 0 : (out - 1) * j.ratio + p % j.ratio + 1);
}

}  /* namespace */

/* ------------------------------------------------------------------ qsa_select */
/* The page-id table the sparse attention reads (docs/OPS.md): over the query's complete blocks
 * n = min(nc[seq], (pos + 1) / ratio, width), the live ones (page id >= 0, score > -1e30) ranked
 * by max(score, 0), or with `signed` 1 those with a finite score, ranked by it. All of them when
 * `topk` or fewer are live, else every block above the topk-th largest value and the blocks equal to
 * it until topk are taken, the whole selection in block order.
 * The partial page bt[n] follows when (pos + 1) % ratio != 0 and n < width, and -1 pads the row
 * to topk + 1. A row no sequence holds reads sequence 0, as libref's and libr4d's do. */
AVX_KERNEL(qsa_select) {
    if (!rad_args_have(a, 8)) return RAD_E_INVAL;
    const RadTensor *sc = &a->t[0], *bt = &a->t[1], *nc = &a->t[2], *po = &a->t[3];
    const RadTensor *cu = &a->t[4], *se = &a->t[5], *ns = &a->t[6], *su = &a->t[7];
    /* Optional, last: the same entries as block indices. */
    const RadTensor* sb = a->n_t > 8 && a->t[8].data ? &a->t[8] : nullptr;
    const int64_t topk = p_int(a, "topk", 0), ratio = p_int(a, "ratio", 0);
    const int64_t sgn = p_int(a, "signed", 0);
    if (topk <= 0 || ratio <= 0) return RAD_E_SHAPE;
    if (sgn != 0 && sgn != 1) return RAD_E_INVAL;
    if (sc->rank != 2 || bt->rank != 2 || se->rank != 2) return RAD_E_SHAPE;
    const int64_t M = sc->shape[0], nb = std::min(bt->shape[1], sc->shape[1]);
    const int64_t n_seq = numel(cu) - 1;
    if (M <= 0 || nb <= 0 || n_seq <= 0) return RAD_E_SHAPE;
    if (bt->shape[0] < n_seq || numel(nc) < n_seq) return RAD_E_SHAPE;
    if (se->shape[0] < M || se->shape[1] < topk + 1) return RAD_E_SHAPE;
    if (numel(ns) < M || numel(po) < M || numel(su) < M) return RAD_E_SHAPE;
    if (sb && (sb->rank != 2 || sb->shape[0] < M || sb->shape[1] < topk + 1)) return RAD_E_SHAPE;
    for (int64_t s = 0; s < n_seq; ++s)
        if (ldi(nc, offlin(nc, s)) < 0) return RAD_E_INVAL;
    for (int64_t m = 0; m < M; ++m)
        if (ldi(po, offlin(po, m)) < 0) return RAD_E_INVAL;

    int64_t* seq = (int64_t*)scratch_raw((size_t)M * sizeof(int64_t) + 64);
    seq_rows(cu, n_seq, M, 0, seq);
    SelJob j{};
    j.sc = sc; j.bt = bt; j.nc = nc; j.po = po; j.se = se; j.ns = ns; j.su = su; j.sb = sb;
    j.seq = seq; j.nb = nb; j.topk = topk; j.ratio = ratio;
    j.s_rs = sc->stride[0]; j.s_cs = sc->stride[1];
    j.bt_rs = bt->stride[0]; j.bt_cs = bt->stride[1];
    j.o_rs = se->stride[0]; j.o_cs = se->stride[1];
    j.b_rs = sb ? sb->stride[0] : 0; j.b_cs = sb ? sb->stride[1] : 0;
    j.sgn = sgn != 0;
    AVX_PARALLEL_FOR_IF(M * nb >= AVX_PAR_MIN && M > 1)
    for (int64_t m = 0; m < M; ++m) select_row(j, m);
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_tail_store */
/* state[sidx[s][0]][0][pos[t] % ring] = k[t] for the last `ring` rows of each sequence, rows in
 * ascending order, so a later row wins a shared target as in libref's serial loop (a row whose
 * slot is -1, or that no sequence holds, writes nothing). At most `ring` rows a sequence write, so
 * the copy is serial. */
AVX_KERNEL(qsa_tail_store) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *po = &a->t[1], *cu = &a->t[2], *st = &a->t[3];
    const RadTensor *sx = &a->t[4];
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), ring = p_int(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return RAD_E_INVAL;
    if (k->rank != 2 || st->rank != 4 || st->shape[1] != 1) return RAD_E_SHAPE;
    const int64_t T = k->shape[0], slots = st->shape[0], n_seq = numel(cu) - 1;
    if (T <= 0 || k->shape[1] != n || st->shape[2] < ring || st->shape[3] != n) return RAD_E_SHAPE;
    if (n_seq <= 0 || numel(po) < T) return RAD_E_SHAPE;
    if ((sx->rank >= 2 ? sx->shape[0] : numel(sx)) < n_seq) return RAD_E_SHAPE;

    /* The rows that write, as (row, slot), validated before any write. */
    int64_t* seq = (int64_t*)scratch_raw((size_t)T * 3 * sizeof(int64_t) + 64);
    int64_t* w = seq + T;
    seq_rows(cu, n_seq, T, -1, seq);
    int64_t nw = 0;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t s = seq[t];
        if (s < 0 || t < cu_at(cu, s + 1) - ring) continue;
        const int64_t si = ldi(sx, col0(sx, s));
        if (si < 0) continue;
        if (si >= slots || ldi(po, offlin(po, t)) < 0) return RAD_E_INVAL;
        w[2 * nw] = t;
        w[2 * nw + 1] = si;
        ++nw;
    }
    float* buf = scratch_f32((size_t)n);
    for (int64_t i = 0; i < nw; ++i) {
        const int64_t t = w[2 * i], si = w[2 * i + 1];
        const int64_t row = ldi(po, offlin(po, t)) % ring;
        widen(k, t * k->stride[0], k->stride[1], n, buf);
        narrow(st, si * st->stride[0] + row * st->stride[2], st->stride[3], n, buf);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_work */
/* The blocks that complete in this batch, in sequence order and ascending within one (libref's
 * ref_qsa_work has the full statement): page bt[s][b] (-1 without a tail slot), the first token's
 * position in bpos (three components with `rope`), and, with a slot, the block's `ratio` raw keys
 * in stage rows w * ratio + i, each from this step's rows or the tail's. Items past the count get
 * page -1 and position 0 and leave `stage` alone. nc[s] = (last position + 1) / ratio. Items are
 * independent, so they are split across threads. */
AVX_KERNEL(qsa_work) {
    if (a->n_t < 11) return RAD_E_INVAL;
    const RadTensor *k = rad_arg_in(a, 0), *po = rad_arg_in(a, 1), *cu = rad_arg_in(a, 2);
    const RadTensor *tl = rad_arg_in(a, 3), *sx = rad_arg_in(a, 4), *bt = rad_arg_in(a, 5);
    const RadTensor *rp = rad_arg_in(a, 6), *stg = rad_arg_in(a, 7), *pg = rad_arg_in(a, 8);
    const RadTensor *bp = rad_arg_in(a, 9), *nc = rad_arg_in(a, 10);
    if (!k || !po || !cu || !tl || !sx || !bt || !stg || !pg || !bp || !nc) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), ring = p_int(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return RAD_E_INVAL;
    if (k->rank != 2 || bt->rank != 2 || stg->rank != 2) return RAD_E_SHAPE;
    if (tl->rank != 4 || tl->shape[1] != 1 || tl->shape[2] < ring || tl->shape[3] != n)
        return RAD_E_SHAPE;
    const int64_t T = k->shape[0], n_seq = numel(cu) - 1, slots = tl->shape[0];
    if (T <= 0 || n_seq <= 0 || numel(po) < T || k->shape[1] != n || stg->shape[1] != n)
        return RAD_E_SHAPE;
    if (bt->shape[0] < n_seq || (sx->rank >= 2 ? sx->shape[0] : numel(sx)) < n_seq)
        return RAD_E_SHAPE;
    const int64_t mw = numel(pg);
    if (mw < n_seq || stg->shape[0] < mw * ratio || numel(nc) < n_seq) return RAD_E_SHAPE;
    int64_t bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < mw || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (rp && bp->shape[1] < 3))
            return RAD_E_SHAPE;
        bstep = bp->stride[0];
    } else if (rp || numel(bp) < mw) {
        return RAD_E_SHAPE;
    }
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T)) return RAD_E_SHAPE;
    for (int64_t t = 0; t < T; ++t)
        if (ldi(po, offlin(po, t)) < 0) return RAD_E_INVAL;

    /* The work list as (sequence, block), validated before any write. */
    int64_t* items = (int64_t*)scratch_raw((size_t)mw * 2 * sizeof(int64_t) + 64);
    int64_t ni = 0;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t t0 = cu_at(cu, s), t1 = cu_at(cu, s + 1);
        if (t1 <= t0) continue;
        if (t0 < 0 || t1 > T) return RAD_E_SHAPE;
        const int64_t p0 = ldi(po, offlin(po, t0)), p1 = ldi(po, offlin(po, t1 - 1));
        for (int64_t b = p0 / ratio; b <= (p1 + 1) / ratio - 1; ++b) {
            if (ni >= mw) break;
            if (b >= bt->shape[1]) return RAD_E_SHAPE;
            const int64_t si = ldi(sx, col0(sx, s));
            if (si >= slots) return RAD_E_INVAL;
            for (int64_t i = 0; si >= 0 && i < ratio; ++i) {
                const int64_t q = b * ratio + i;
                if (q >= p0 && t0 + (q - p0) >= T) return RAD_E_SHAPE;
            }
            items[2 * ni] = s;
            items[2 * ni + 1] = b;
            ++ni;
        }
    }

    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t t0 = cu_at(cu, s), t1 = cu_at(cu, s + 1);
        st_int_dt(nc->dtype, nc->data, offlin(nc, s),
                  t1 > t0 ? (ldi(po, offlin(po, t1 - 1)) + 1) / ratio : 0);
    }
    const int ncomp = rp ? 3 : 1;
    AVX_PARALLEL_FOR_IF(mw * ratio * n >= AVX_PAR_MIN)
    for (int64_t w = 0; w < mw; ++w) {
        auto bpos_at = [&](int c) { return bp->rank == 2 ? w * bstep + c : offlin(bp, w); };
        if (w >= ni) {
            st_int_dt(pg->dtype, pg->data, offlin(pg, w), -1);
            for (int c = 0; c < ncomp; ++c) st_int_dt(bp->dtype, bp->data, bpos_at(c), 0);
            continue;
        }
        const int64_t s = items[2 * w], b = items[2 * w + 1];
        const int64_t t0 = cu_at(cu, s), p0 = ldi(po, offlin(po, t0));
        const int64_t si = ldi(sx, col0(sx, s));
        st_int_dt(pg->dtype, pg->data, offlin(pg, w),
                  si < 0 ? -1 : ldi(bt, s * bt->stride[0] + b * bt->stride[1]));
        const int64_t f = b * ratio;
        if (!rp) {
            st_int_dt(bp->dtype, bp->data, bpos_at(0), f);
        } else {
            const int64_t rs = rp->stride[0], rc = rp->stride[1];
            for (int c = 0; c < 3; ++c)
                st_int_dt(bp->dtype, bp->data, bpos_at(c),
                          f >= p0 ? ldi(rp, c * rs + (t0 + f - p0) * rc)
                                  : f + (ldi(rp, t0 * rc) - p0));
        }
        if (si < 0) continue;
        float* buf = scratch_f32((size_t)n);
        for (int64_t i = 0; i < ratio; ++i) {
            const int64_t q = f + i;
            if (q >= p0) widen(k, (t0 + q - p0) * k->stride[0], k->stride[1], n, buf);
            else widen(tl, si * tl->stride[0] + (q % ring) * tl->stride[2], tl->stride[3], n, buf);
            narrow(stg, (w * ratio + i) * stg->stride[0], stg->stride[1], n, buf);
        }
    }
    return RAD_OK;
}

namespace {

/* VF_N bf16 values widened, and VF_N floats narrowed RTNE with the ABI's NaN rule: avx_vec.h's
 * row conversions at one vector, for a kernel that keeps the vector in a register. */
static inline vf load_bf16(const uint16_t* p) {
#if AVX_LEVEL == 3
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
#elif AVX_LEVEL == 2
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
#elif AVX_LEVEL == 1
    const __m128i h = _mm_loadu_si128((const __m128i*)p);
    const __m128i lo = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
    const __m128i hi = _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(h, 8)), 16);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(lo)),
                                _mm_castsi128_ps(hi), 1);
#else
    return rad_bf16_to_f32(*p);
#endif
}

static inline void store_bf16(uint16_t* p, vf v) {
#if AVX_LEVEL == 3
    _mm256_storeu_si256((__m256i*)p, _mm512_cvtepi32_epi16(bf16_round_bits(v)));
#elif AVX_LEVEL == 2
    const __m256i w = bf16_round_bits(v);
    _mm_storeu_si128((__m128i*)p, _mm_packus_epi32(_mm256_castsi256_si128(w),
                                                   _mm256_extracti128_si256(w, 1)));
#elif AVX_LEVEL == 1
    const __m128i lo = bf16_round_bits128(_mm256_castps256_ps128(v));
    const __m128i hi = bf16_round_bits128(_mm256_extractf128_ps(v, 1));
    _mm_storeu_si128((__m128i*)p, _mm_packus_epi32(lo, hi));
#else
    *p = rad_f32_to_bf16(v);
#endif
}

}  /* namespace */

/* ------------------------------------------------------------------ dflash_conv */
namespace {

struct ConvJob {
    const RadTensor *x, *dl, *bs, *o;
    int64_t T, H, taps, group, NG, bsz;
    int64_t x_rs, o_rs, d_rs, d_ts, d_gs;
};

/* Channels and rows a block: a block's widened base kernel (8 KiB a tap) and its slice of the row
 * above stay in L1 while the block walks its rows. */
enum { kConvCols = 2048, kConvRows = 16 };

/* Rows [t0, t1) of groups [g0, g1), for dense rows of bf16 (XB, OB) or f32 and groups that span
 * whole vectors. The rows a tap reads above are inputs, so any split of the rows is exact. */
template <bool XB, bool OB>
static void conv_block(const ConvJob& j, int64_t t0, int64_t t1, int64_t g0, int64_t g1) {
    const int64_t G = j.group, ng = g1 - g0, cb = ng * G, h0 = g0 * G, taps = j.taps;
    float* bf = scratch_f32((size_t)(taps * cb));
    float* dg = scratch_f32_b((size_t)(taps * ng));
    for (int64_t jt = 0; jt < taps; ++jt)
        widen(j.bs, rowoff(j.bs, jt, j.H) + h0 * laststride(j.bs), laststride(j.bs), cb,
              bf + jt * cb);
    /* Element pointers as locals: through `j` the compiler reloads them after every store. */
    using XT = typename std::conditional<XB, uint16_t, float>::type;
    using OT = typename std::conditional<OB, uint16_t, float>::type;
    const XT* const xd = (const XT*)j.x->data;
    OT* const od = (OT*)j.o->data;
    const int64_t x_rs = j.x_rs;
    auto load = [](const XT* p) {
        if constexpr (XB) return load_bf16(p);
        else              return vf_loadu(p);
    };
    auto store = [](OT* p, vf v) {
        if constexpr (OB) store_bf16(p, v);
        else              vf_storeu(p, v);
    };
    for (int64_t t = t0; t < t1; ++t) {
        /* Tap jt applies when jt == 0 or t % block_size >= jt, i.e. taps 0 .. nt - 1. */
        const int64_t nt = std::min<int64_t>(taps, t % j.bsz + 1);
        for (int64_t jt = 0; jt < nt; ++jt)
            widen(j.dl, t * j.d_rs + jt * j.d_ts + g0 * j.d_gs, j.d_gs, ng, dg + jt * ng);
        const XT* const x0 = xd + t * x_rs + h0;
        OT* const out = od + t * j.o_rs + h0;
        /* One and two taps (DFlash2's) straight-line; more in a loop. */
        for (int64_t k = 0; k < ng; ++k) {
            const vf d0 = vf_set1(dg[k]);
            const int64_t i1 = (k + 1) * G;
            if (nt == 1) {
                for (int64_t i = k * G; i < i1; i += VF_N)
                    store(out + i, vf_mul(vf_add(vf_loadu(bf + i), d0), load(x0 + i)));
            } else if (nt == 2) {
                const vf d1 = vf_set1(dg[ng + k]);
                const XT* const x1 = x0 - x_rs;
                const float* const b1 = bf + cb;
                for (int64_t i = k * G; i < i1; i += VF_N) {
                    const vf acc = vf_mul(vf_add(vf_loadu(bf + i), d0), load(x0 + i));
                    store(out + i, vf_fma(vf_add(vf_loadu(b1 + i), d1), load(x1 + i), acc));
                }
            } else {
                for (int64_t i = k * G; i < i1; i += VF_N) {
                    vf acc = vf_mul(vf_add(vf_loadu(bf + i), d0), load(x0 + i));
                    for (int64_t jt = 1; jt < nt; ++jt)
                        acc = vf_fma(vf_add(vf_loadu(bf + jt * cb + i), vf_set1(dg[jt * ng + k])),
                                     load(x0 - jt * x_rs + i), acc);
                    store(out + i, acc);
                }
            }
        }
    }
}

/* Any other rows: row t through f32 copies of x. */
static void conv_row(const ConvJob& j, const float* bf, int64_t t) {
    const int64_t H = j.H, NG = j.NG, G = j.group;
    const int64_t nt = std::min<int64_t>(j.taps, t % j.bsz + 1);
    float* acc = scratch_f32((size_t)H);
    float* xr = scratch_f32_b((size_t)H);
    float* dg = scratch_f32_c((size_t)NG);
    for (int64_t jt = 0; jt < nt; ++jt) {
        const float* xv = in_row(j.x, (t - jt) * j.x_rs, H, xr);
        const float* bj = bf + jt * H;
        widen(j.dl, t * j.d_rs + jt * j.d_ts, j.d_gs, NG, dg);
        for (int64_t g = 0; g < NG; ++g)
            for (int64_t h = g * G; h < (g + 1) * G; ++h) {
                const float term = (bj[h] + dg[g]) * xv[h];
                acc[h] = jt == 0 ? term : acc[h] + term;
            }
    }
    out_row(j.o, t * j.o_rs, H, acc);
}

}  /* namespace */

/* DFlash2's grouped dynamic depthwise convolution over the [T, H] rows of x, blocks of
 * `block_size` rows starting at row 0:
 *
 *   out[t][h] = sum over taps j with j == 0 or t % block_size >= j of
 *               (base[j][h] + delta[t][j][h / group_size]) * x[t - j][h]
 *
 * in tap order, the coefficient add rounded as libref's, the product and sum fused. `delta` is
 * [T, taps, NG], or [T, >= taps * NG] at its own row pitch. `out` must not alias `x`. */
AVX_KERNEL(dflash_conv) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *dl = &a->t[1], *bs = &a->t[2], *o = &a->t[3];
    const int64_t taps = p_int(a, "taps", 0), group = p_int(a, "group_size", 0);
    const int64_t NG = p_int(a, "NG", 0), bsz = p_int(a, "block_size", 0);
    if (taps <= 0 || group <= 0 || NG <= 0 || bsz <= 0) return RAD_E_INVAL;
    if (x->data == o->data) return RAD_E_INVAL;
    if (x->rank != 2 || o->rank != 2) return RAD_E_SHAPE;
    const int64_t T = x->shape[0], H = x->shape[1];
    if (T <= 0 || o->shape[0] != T || o->shape[1] != H || H != NG * group) return RAD_E_SHAPE;
    if (numel(bs) < taps * H) return RAD_E_SHAPE;
    ConvJob j{};
    j.x = x; j.dl = dl; j.bs = bs; j.o = o;
    j.T = T; j.H = H; j.taps = taps; j.group = group; j.NG = NG; j.bsz = bsz;
    j.x_rs = x->stride[0]; j.o_rs = o->stride[0];
    if (dl->rank == 3) {
        if (dl->shape[0] < T || dl->shape[1] < taps || dl->shape[2] < NG) return RAD_E_SHAPE;
        j.d_rs = dl->stride[0]; j.d_ts = dl->stride[1]; j.d_gs = dl->stride[2];
    } else if (dl->rank == 2) {
        if (dl->shape[0] < T || dl->shape[1] < taps * NG) return RAD_E_SHAPE;
        j.d_rs = dl->stride[0]; j.d_gs = dl->stride[1]; j.d_ts = NG * j.d_gs;
    } else {
        return RAD_E_SHAPE;
    }

    /* Dense bf16 and f32 rows, with groups of whole vectors so a group's delta is one broadcast:
     * blocks of columns and rows, converted in the loads and the store. */
    const bool fused = group % VF_N == 0 && laststride(x) == 1 && laststride(o) == 1 &&
                       (x->dtype == RAD_BF16 || x->dtype == RAD_F32) &&
                       (o->dtype == RAD_BF16 || o->dtype == RAD_F32);
    if (fused) {
        const int64_t gpb = std::max<int64_t>(1, kConvCols / group);
        const int64_t ncb = (NG + gpb - 1) / gpb, nrb = (T + kConvRows - 1) / kConvRows;
        const bool xb = x->dtype == RAD_BF16, ob = o->dtype == RAD_BF16;
        AVX_PARALLEL_FOR_IF(T * H * taps >= ((int64_t)1 << 18))
        for (int64_t u = 0; u < ncb * nrb; ++u) {
            const int64_t g0 = (u % ncb) * gpb, g1 = std::min(g0 + gpb, NG);
            const int64_t t0 = (u / ncb) * kConvRows, t1 = std::min(t0 + kConvRows, T);
            if (xb && ob)  conv_block<true, true>(j, t0, t1, g0, g1);
            else if (xb)   conv_block<true, false>(j, t0, t1, g0, g1);
            else if (ob)   conv_block<false, true>(j, t0, t1, g0, g1);
            else           conv_block<false, false>(j, t0, t1, g0, g1);
        }
        return RAD_OK;
    }
    /* The base kernel widened once; read-only inside the parallel loop. */
    float* bf = (float*)scratch_raw((size_t)(taps * H) * sizeof(float) + 64);
    for (int64_t jt = 0; jt < taps; ++jt)
        widen(bs, rowoff(bs, jt, H), laststride(bs), H, bf + jt * H);
    AVX_PARALLEL_FOR_IF(T * H * taps >= ((int64_t)1 << 18))
    for (int64_t t = 0; t < T; ++t) conv_row(j, bf, t);
    return RAD_OK;
}
