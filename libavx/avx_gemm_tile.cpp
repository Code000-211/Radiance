/* avx_gemm_tile.cpp -- the GEMM's tile drivers: an M x N tile loop over packed column panels.
 *
 * avx_gemm.cpp's register-blocked dot-product driver forks a parallel region per L2-sized block of
 * A rows and widens every weight row again in each, so at prefill B is streamed from DRAM once per
 * 6-16 rows of A. These drivers serve every M, one batch of products (avx_gemm.h: TileJob) a
 * parallel region: a dense product is one job per super-block of A rows (sized to stay in the L3),
 * a grouped GEMM one job per expert and part.
 *
 *   convert the batch's A rows once, in parallel
 *   parallel over tasks = (job, column panel or group of panels, row sub-block), dynamically
 *     for each K chunk (the packed panel chunk sized for the L2)
 *       pack the chunk: the panel's weight rows decoded and transposed to k-major order
 *       run the microkernel over every MR rows of the sub-block, into the task's C tile
 *     store the C tile (+ bias)
 *
 * At decode the same loop reads each weight row once, as a few long runs (the chunk is longer
 * where few rows share a panel, and the next rows are prefetched), and stays at the DRAM read
 * stream's rate for the byte-wide codes.
 *
 * The microkernels are outer products: per k, a broadcast of each A row against vectors of panel
 * columns, every lane one output, no horizontal reduction.
 *
 *   gemm_q_bf16   AVX512_BF16 parts, weight codes exact in bf16 (E4M3, int8, int4, e2m1) against
 *                 an E4M3, int8 or bf16 activation, which every quantised dense format here is.
 *                 Codes are packed as bf16 pairs ([k pair][48 columns][2], vdpbf16ps's operand
 *                 layout), and each scale group's sub-sum of raw codes is accumulated by
 *                 vdpbf16ps in f32 (products of two such codes are exact), then scaled once,
 *                 (sub * sa) * sb, as libref's quant_dot does. Plain bf16 weights take the same
 *                 path unscaled. 8 rows x 48 columns.
 *   gemm_tile_f32 every level and operand pair: A widened to f32 with its scale folded in, the
 *                 panel likewise (a weight code times its scale is exact in f32 for every format
 *                 here), one f32 FMA per element. 8 x 48 at AVX-512, 6 x 16 at AVX2; with eight
 *                 rows or fewer, dot products against the untransposed rows instead.
 *
 * An output's arithmetic does not depend on the blocking: a row's lane in the outer product is
 * the same whichever rows share its microkernel call, groups lie whole in a chunk, and the f32
 * chain runs over k in order through the C tile. So results do not depend on the team (avx_common.h's
 * threading rule) or on the chunk length.
 */
#include "avx_gemm.h"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>

namespace avx {
inline namespace AVX_SUFFIX {

namespace {

static int tile_team() {
#ifdef _OPENMP
    return avx_team_size();
#else
    return 1;
#endif
}

/* A super-block's tasks: `pg` panels by `sub` rows each. With at most eight panels and 64 rows
 * or more a task takes all the panels, iterated inside each K chunk, so its rows of A for a chunk
 * come into its L2 once rather than once per panel on other cores (a 144-channel convolution has
 * three); its sub-block is then at most 128 rows, which keeps that chunk of A, a panel's chunk and
 * the C tiles in the L2. More panels already give enough tasks one panel each, and fewer rows
 * (decode) take one a task for parallelism. The row sub-blocks give eight tasks per thread where
 * the panel groups are fewer, none shorter than 64 rows (a task packs its own panel chunks) and
 * none longer than 512. `rows` is a multiple of `mr` except in the last super-block. */
static void tile_split(int64_t rows, int64_t np, int64_t mr, int nth, int64_t* sub,
                       int64_t* nsub, int64_t* pg) {
    *pg = rows >= 64 && np <= 8 ? np : 1;
    const int64_t ngr = (np + *pg - 1) / *pg;
    int64_t ns = 1;
    const int64_t want = 8 * (int64_t)nth;
    if (ngr < want) ns = (want + ngr - 1) / ngr;
    const int64_t most = std::max<int64_t>(1, rows / 64);
    if (ns > most) ns = most;
    const int64_t cap = *pg > 1 ? 128 : 512;
    ns = std::max(ns, (rows + cap - 1) / cap);
    int64_t s = (rows + ns - 1) / ns;
    s = (s + mr - 1) / mr * mr;
    *sub = s;
    *nsub = (rows + s - 1) / s;
}

/* Columns [n0, n0 + w) of Y's row m from the f32 accumulators `c`, plus `bias` (w values, or null),
 * rounded once into Y's dtype by the row converter (bit-exact with the ABI's), element by element
 * where the row is strided or its dtype narrower than a byte. */
static void tile_store(const RadTensor* Y, int64_t ys_r, int64_t ys_c, int64_t m, int64_t n0,
                       int64_t w, const float* c, const float* bias, float* tmp) {
    const float* v = c;
    if (bias) {
        for (int64_t j = 0; j < w; ++j) tmp[j] = c[j] + bias[j];
        v = tmp;
    }
    const int64_t off = m * ys_r + n0 * ys_c;
    if (ys_c == 1 && rad_dtype_bits(Y->dtype) >= 8) {
        row_from_f32(Y->dtype, v, (void*)byte_at(Y->data, Y->dtype, off), w);
        return;
    }
    for (int64_t j = 0; j < w; ++j) stt(Y, off + j * ys_c, v[j]);
}

/* Bias values of columns [n0, n0 + w), zero past N; false when there is no bias. */
static bool tile_bias(const RadTensor* bias, int64_t n0, int64_t w, int64_t N, float* dst) {
    if (!bias) return false;
    const int64_t bs = laststride(bias);
    for (int64_t j = 0; j < w; ++j) dst[j] = n0 + j < N ? ldt(bias, (n0 + j) * bs) : 0.0f;
    return true;
}

#ifdef _OPENMP
/* The task loop's schedule, set for the calling thread's next region and restored after. One
 * product of few rows is a stream of B in uniform tasks, split statically so a thread takes
 * contiguous panels: the rows of B a static loop over them wrote (a convolution's patches, as y^T)
 * are then in its own core complex's L3. Otherwise tasks are handed out one at a time. */
struct TileSchedule {
    omp_sched_t kind;
    int chunk;
    explicit TileSchedule(int64_t njobs, int64_t rows) {
        omp_get_schedule(&kind, &chunk);
        if (njobs == 1 && rows <= 32) omp_set_schedule(omp_sched_static, 0);
        else omp_set_schedule(omp_sched_dynamic, 1);
    }
    ~TileSchedule() { omp_set_schedule(kind, chunk); }
};
#endif

/* The lines of [p, p + bytes), requested ahead of their read: the panel's weight rows are short
 * runs at a row's pitch, too short for the hardware's stream detection to cover. */
static inline void tile_prefetch(const void* p, int64_t bytes) {
    const uintptr_t a = (uintptr_t)p, e = a + (uintptr_t)bytes;
    for (uintptr_t q = a & ~(uintptr_t)63; q < e; q += 64) __builtin_prefetch((const void*)q, 0, 3);
}

}  /* namespace */

/* ================================================================== the bf16 grouped driver */
#if AVX_LEVEL == 3
#define AVX_QB_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))

namespace {

constexpr int64_t QB_COLS = 48;        /* panel columns: three zmm of f32 accumulators */
constexpr int64_t QB_MR = 8;           /* rows of A per microkernel call */
/* K chunk, in codes. Where few rows share a panel the weight rows stream from DRAM, and a long
 * chunk keeps each row one sequential read (8192: a 768 KiB panel); with many rows the chunk is
 * sized for the L2 (2048: 192 KiB). */
constexpr int64_t QB_KC_FEW = 8192, QB_KC = 2048, QB_FEW_ROWS = 32;

/* bf16 bit patterns of every code of the four weight formats and the two activation formats
 * that are not bf16 already. Each value has at most eight significant bits, so the top half of
 * its f32 pattern is the value exactly. */
struct QbLut {
    alignas(64) uint16_t e4m3[256];
    alignas(64) uint16_t i8[256];
    alignas(64) uint16_t i4[32];
    alignas(64) uint16_t e2m1[32];
};

static uint16_t qb_hi(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static const QbLut& qb_lut() {
    static const QbLut L = [] {
        QbLut l{};
        static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        for (int c = 0; c < 256; ++c) {
            l.e4m3[c] = qb_hi(rad_fp8e4m3_to_f32((uint8_t)c));
            l.i8[c] = qb_hi((float)(int8_t)c);
        }
        for (int c = 0; c < 32; ++c) {
            const int n = c & 15;
            l.i4[c] = qb_hi((float)((n ^ 8) - 8));
            l.e2m1[c] = qb_hi((n & 8) ? -kMag[n & 7] : kMag[n & 7]);
        }
        return l;
    }();
    return L;
}

/* A code format's table, loaded once per task: eight registers for a byte code (vpermt2w looks
 * up 64 words across a pair, the top two bits pick the pair), one for a nibble. */
struct QbTab {
    __m512i t[8];
    bool nib;
    bool copy;          /* the codes are bf16 already */
};

AVX_QB_TARGET static QbTab qb_tab(uint32_t dt) {
    const QbLut& L = qb_lut();
    QbTab q{};
    q.copy = dt == RAD_BF16;
    if (q.copy) return q;
    const uint16_t* src = dt == RAD_F8E4M3 ? L.e4m3 : dt == RAD_I8 ? L.i8
                        : dt == RAD_I4 ? L.i4 : L.e2m1;
    q.nib = dt == RAD_I4 || dt == RAD_FP4E2M1;
    for (int k = 0; k < (q.nib ? 1 : 8); ++k)
        q.t[k] = _mm512_loadu_si512((const void*)(src + 32 * k));
    return q;
}

AVX_QB_TARGET static inline __m512i qb_pick(const QbTab& q, __m512i ix) {
    if (q.nib) return _mm512_permutexvar_epi16(ix, q.t[0]);
    const __m512i q0 = _mm512_permutex2var_epi16(q.t[0], ix, q.t[1]);
    const __m512i q1 = _mm512_permutex2var_epi16(q.t[2], ix, q.t[3]);
    const __m512i q2 = _mm512_permutex2var_epi16(q.t[4], ix, q.t[5]);
    const __m512i q3 = _mm512_permutex2var_epi16(q.t[6], ix, q.t[7]);
    const __mmask32 h6 = _mm512_test_epi16_mask(ix, _mm512_set1_epi16(0x40));
    const __mmask32 h7 = _mm512_test_epi16_mask(ix, _mm512_set1_epi16(0x80));
    return _mm512_mask_blend_epi16(h7, _mm512_mask_blend_epi16(h6, q0, q1),
                                   _mm512_mask_blend_epi16(h6, q2, q3));
}

/* `n` codes from `src` (the byte holding the first) as bf16 into dst[0, npad), npad a multiple of
 * 32, the words past n zero. Zero is not code 0 in every format (int4's 0 is -8), so the padding
 * is applied to the values. With `ex`, each 32 codes are multiplied by the E8M0 scale ex[i / 32]
 * through their exponents: exact for an exponent byte in [2, 252] (qb_foldable), and 0xFF makes
 * them all NaN, as the scale's product would. */
AVX_QB_TARGET static void qb_codes(const QbTab& q, const uint8_t* src, int64_t n, int64_t npad,
                                   uint16_t* dst, const uint8_t* ex = nullptr) {
    for (int64_t i = 0; i < npad; i += 32) {
        const int64_t left = n - i;
        const __mmask32 keep = left >= 32 ? ~(__mmask32)0
                             : left <= 0 ? (__mmask32)0 : (__mmask32)((1u << left) - 1u);
        if (q.copy) {
            _mm512_storeu_si512((void*)(dst + i),
                                _mm512_maskz_loadu_epi16(keep, (const uint16_t*)src + i));
            continue;
        }
        __m256i x;
        if (!q.nib) {
            x = left >= 32 ? _mm256_loadu_si256((const __m256i*)(src + i))
                           : _mm256_maskz_loadu_epi8(keep, src + i);
        } else {
            const int64_t nb = left >= 32 ? 16 : left <= 0 ? 0 : (left + 1) / 2;
            const __m128i b = _mm_maskz_loadu_epi8((__mmask16)((1u << nb) - 1u), src + i / 2);
            const __m128i m4 = _mm_set1_epi8(0x0f);
            const __m128i lo = _mm_and_si128(b, m4);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m4);
            x = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        }
        __m512i v = qb_pick(q, _mm512_cvtepu8_epi16(x));
        if (ex) {
            const int e = ex[i / 32];
            if (e == 0xff) {
                v = _mm512_set1_epi16(0x7fc0);
            } else {
                const __mmask32 nz = _mm512_test_epi16_mask(v, _mm512_set1_epi16(0x7fff));
                v = _mm512_mask_add_epi16(v, nz, v, _mm512_set1_epi16((short)((e - 127) * 128)));
            }
        }
        _mm512_storeu_si512((void*)(dst + i), _mm512_maskz_mov_epi16(keep, v));
    }
}

/* `n` codes of `bits` (4 or 5) of a zero-point grid, from `row` (the byte holding the chunk's
 * first code), as code - zero in bf16 into dst[0, npad), the words past n zero. zr[j] is the zero
 * of the j-th 32 codes. A code minus its zero lies in [-31, 31], exact in bf16: `t` holds bf16(i -
 * 32) for i < 64, which one vpermt2w indexes by code - zero + 32. u5 codes are 40 bits for each
 * eight, deposited into bytes by pdep (BMI2, which the caller checked). */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16,bmi2"))) static void qb_codes_zp(
        const uint8_t* row, int bits, int64_t n, int64_t npad, uint16_t* dst, const int32_t* zr,
        const __m512i* t) {
    for (int64_t i = 0; i < npad; i += 32) {
        const int64_t left = n - i;
        const __mmask32 keep = left >= 32 ? ~(__mmask32)0
                             : left <= 0 ? (__mmask32)0 : (__mmask32)((1u << left) - 1u);
        __m256i x;
        if (bits == 4) {
            const int64_t nb = left >= 32 ? 16 : left <= 0 ? 0 : (left + 1) / 2;
            const __m128i b = _mm_maskz_loadu_epi8((__mmask16)((1u << nb) - 1u), row + i / 2);
            const __m128i m4 = _mm_set1_epi8(0x0f);
            const __m128i lo = _mm_and_si128(b, m4);
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m4);
            x = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
        } else {
            /* Four runs of eight codes, 5 bytes each; only the bytes the row holds are read. */
            const int64_t nb = left >= 32 ? 20 : left <= 0 ? 0 : (left * 5 + 7) / 8;
            alignas(8) uint8_t buf[24] = {};
            std::memcpy(buf, row + i / 8 * 5, (size_t)nb);
            const unsigned long long kLow5 = 0x1f1f1f1f1f1f1f1full;
            uint64_t q[4];
            for (int r = 0; r < 4; ++r) {
                uint64_t v = 0;
                std::memcpy(&v, buf + 5 * r, 8);
                q[r] = _pdep_u64(v & 0xffffffffffull, kLow5);
            }
            x = _mm256_set_epi64x((long long)q[3], (long long)q[2], (long long)q[1],
                                  (long long)q[0]);
        }
        const __m512i ix = _mm512_add_epi16(_mm512_cvtepu8_epi16(x),
                                            _mm512_set1_epi16((short)(32 - zr[i / 32])));
        const __m512i v = _mm512_permutex2var_epi16(t[0], ix, t[1]);
        _mm512_storeu_si512((void*)(dst + i), _mm512_maskz_mov_epi16(keep, v));
    }
}

/* E8M0 exponents whose products with every e2m1 code (0.5 to 6 in magnitude) are bf16 normals,
 * or the NaN. */
static inline bool qb_foldable(const uint8_t* ex, int64_t n) {
    for (int64_t i = 0; i < n; ++i)
        if ((ex[i] < 2 || ex[i] > 252) && ex[i] != 0xff) return false;
    return true;
}

/* r[j] <- column j of the 16 x 16 matrix of 32-bit elements whose row i is r[i]. */
AVX_QB_TARGET static inline void tr16(__m512i r[16]) {
    __m512i t[16], u[16], v[16];
    for (int i = 0; i < 8; ++i) {
        t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
        t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
    }
    /* u[4q + c], 128-bit lane L: column 4L + c of rows 4q .. 4q + 3. */
    for (int q = 0; q < 4; ++q) {
        u[4 * q + 0] = _mm512_unpacklo_epi64(t[4 * q], t[4 * q + 2]);
        u[4 * q + 1] = _mm512_unpackhi_epi64(t[4 * q], t[4 * q + 2]);
        u[4 * q + 2] = _mm512_unpacklo_epi64(t[4 * q + 1], t[4 * q + 3]);
        u[4 * q + 3] = _mm512_unpackhi_epi64(t[4 * q + 1], t[4 * q + 3]);
    }
    for (int c = 0; c < 4; ++c) {
        v[c] = _mm512_shuffle_i32x4(u[c], u[4 + c], 0x88);
        v[4 + c] = _mm512_shuffle_i32x4(u[c], u[4 + c], 0xdd);
        v[8 + c] = _mm512_shuffle_i32x4(u[8 + c], u[12 + c], 0x88);
        v[12 + c] = _mm512_shuffle_i32x4(u[8 + c], u[12 + c], 0xdd);
    }
    for (int c = 0; c < 4; ++c) {
        r[c] = _mm512_shuffle_i32x4(v[c], v[8 + c], 0x88);
        r[8 + c] = _mm512_shuffle_i32x4(v[c], v[8 + c], 0xdd);
        r[4 + c] = _mm512_shuffle_i32x4(v[4 + c], v[12 + c], 0x88);
        r[12 + c] = _mm512_shuffle_i32x4(v[4 + c], v[12 + c], 0xdd);
    }
}

/* MR rows of A against a packed chunk: per group, the sub-sums of 48 columns over its k pairs,
 * then C += (sub * sa) * sb. `A` is the rows' bf16 at the chunk's first column, `P` the chunk's
 * pairs [kp][48], `S` its scales [group][48], `gk` each group's k pairs, `sa` the rows'
 * activation scales of the chunk's groups (row i, group g at sa[i * sald + g * sastep]). */
template <int MR>
AVX_QB_TARGET static void qb_kernel(const uint16_t* A, int64_t lda, const uint32_t* P,
                                    const float* S, const int64_t* gk, int64_t ng, const float* sa,
                                    int64_t sald, int64_t sastep, float* C) {
    const uint16_t* a = A;
    const uint32_t* p = P;
    for (int64_t gi = 0; gi < ng; ++gi) {
        __m512 acc[MR][3];
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i)
            acc[i][0] = acc[i][1] = acc[i][2] = _mm512_setzero_ps();
        for (int64_t kp = 0; kp < gk[gi]; ++kp, p += QB_COLS, a += 2) {
            const __m512bh b0 = (__m512bh)_mm512_load_si512((const void*)p);
            const __m512bh b1 = (__m512bh)_mm512_load_si512((const void*)(p + 16));
            const __m512bh b2 = (__m512bh)_mm512_load_si512((const void*)(p + 32));
#pragma GCC unroll 8
            for (int i = 0; i < MR; ++i) {
                uint32_t pr;
                std::memcpy(&pr, a + i * lda, 4);
                const __m512bh av = (__m512bh)_mm512_set1_epi32((int)pr);
                acc[i][0] = _mm512_dpbf16_ps(acc[i][0], av, b0);
                acc[i][1] = _mm512_dpbf16_ps(acc[i][1], av, b1);
                acc[i][2] = _mm512_dpbf16_ps(acc[i][2], av, b2);
            }
        }
        const __m512 s0 = _mm512_loadu_ps(S + gi * QB_COLS);
        const __m512 s1 = _mm512_loadu_ps(S + gi * QB_COLS + 16);
        const __m512 s2 = _mm512_loadu_ps(S + gi * QB_COLS + 32);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const __m512 sv = _mm512_set1_ps(sa[i * sald + gi * sastep]);
            float* c = C + i * QB_COLS;
            _mm512_storeu_ps(c, _mm512_fmadd_ps(_mm512_mul_ps(acc[i][0], sv), s0,
                                                _mm512_loadu_ps(c)));
            _mm512_storeu_ps(c + 16, _mm512_fmadd_ps(_mm512_mul_ps(acc[i][1], sv), s1,
                                                     _mm512_loadu_ps(c + 16)));
            _mm512_storeu_ps(c + 32, _mm512_fmadd_ps(_mm512_mul_ps(acc[i][2], sv), s2,
                                                     _mm512_loadu_ps(c + 32)));
        }
    }
}

using QbKernel = void (*)(const uint16_t*, int64_t, const uint32_t*, const float*,
                          const int64_t*, int64_t, const float*, int64_t, int64_t, float*);
static const QbKernel kQbKernel[QB_MR + 1] = {
    nullptr, qb_kernel<1>, qb_kernel<2>, qb_kernel<3>, qb_kernel<4>,
    qb_kernel<5>, qb_kernel<6>, qb_kernel<7>, qb_kernel<8>,
};

/* The activation formats whose codes this path reads as bf16, and the weight formats. */
static bool qb_act(uint32_t dt) { return dt == RAD_BF16 || dt == RAD_F8E4M3 || dt == RAD_I8; }
static bool qb_weight(uint32_t dt) {
    return dt == RAD_F8E4M3 || dt == RAD_I8 || dt == RAD_I4 || dt == RAD_FP4E2M1 ||
           dt == RAD_BF16;
}

}  /* namespace */

/* A job's task layout, fixed before the parallel region. */
struct QbPlan {
    int64_t np, sub, nsub, pg, ngr, t0, KC;
};

/* The batch (avx_gemm.h: TileJob) on the bf16 grouped driver. Batch row r is A's row arow[r], or
 * abase + r unmapped; `ycols` is Y's logical width, for its row pitch. */
AVX_QB_TARGET static bool qb_core(const Mat& ma, int64_t abase, const int64_t* arow, int64_t arows,
                                  const Scales* SA, const TileJob* jobs, int64_t njobs,
                                  const RadTensor* Y, int64_t ycols) {
    if (njobs <= 0) return true;
    if (ma.dec || arows <= 0) return false;
    const uint32_t adt = ma.t->dtype, bdt = jobs[0].mb.t->dtype;
    const bool zpw = jobs[0].zero.p != nullptr;
    const int zbits = bdt == RAD_U4 ? 4 : bdt == RAD_U5 ? 5 : 0;
    static const bool bmi2 = __builtin_cpu_supports("bmi2");
    if (zpw && (!zbits || (zbits == 5 && !bmi2))) return false;
    if (!qb_act(adt) || (!zpw && !qb_weight(bdt)) || ma.cs != 1) return false;
    if (Y->dtype != RAD_BF16 && Y->dtype != RAD_F32) return false;
    const int64_t ys_r = row_stride(Y, ycols), ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    if (ys_c != 1) return false;
    const bool nib = bdt == RAD_I4 || bdt == RAD_FP4E2M1;
    const bool sb_on = jobs[0].SB.p != nullptr;
    int64_t Kmax = 0;
    bool even = true;
    for (int64_t j = 0; j < njobs; ++j) {
        const TileJob& jb = jobs[j];
        if (jb.mb.dec || jb.mb.t->dtype != bdt || jb.mb.cs != 1 || (nib && (jb.mb.rs & 1)))
            return false;
        if ((jb.SB.p != nullptr) != sb_on || (sb_on && jb.SB.group != jobs[0].SB.group))
            return false;
        /* A zero-point job: whole-byte rows of codes, a u8 zero per (row, scale group), groups of
         * whole 32s. */
        if ((jb.zero.p != nullptr) != zpw) return false;
        if (zpw && (!sb_on || jb.zero.dt != RAD_U8 || (jb.mb.rs * zbits) % 8 ||
                    jb.SB.group % 32 || jb.SB.rblk > 1))
            return false;
        Kmax = std::max(Kmax, jb.K);
        even = even && !(jb.K & 1);
    }
    const bool sa_on = SA && SA->p;
    /* Unscaled (bf16 x bf16), the group is only where a sub-sum joins the C tile: a chunk. */
    const int64_t G = sb_on && jobs[0].SB.group > 0 ? jobs[0].SB.group
                    : (!sa_on && Kmax > QB_KC) ? QB_KC : Kmax;
    const int64_t ngrp = (Kmax + G - 1) / G;
    /* Pairs must not straddle a group: every group but the last is even. */
    if (ngrp > 1 && (G & 1)) return false;
    if (sa_on && SA->group < G) return false;
    /* MXFP4's E8M0 scales fold into the packed codes (qb_codes), chunk by chunk where every scale
     * of the chunk allows it; the sub-sums then run over the activation's group (Gf), four of
     * the weight's, and close the vdpbf16ps chain a quarter as often. */
    const bool e8 = jobs[0].SB.e8m0 || jobs[0].SB.dt == RAD_E8M0;
    const int64_t Gf = sa_on ? SA->group : 0;
    const bool fold = bdt == RAD_FP4E2M1 && sb_on && e8 && jobs[0].SB.cs == 1 && G == 32 &&
                      (!sa_on || (Gf % G == 0 && QB_KC % Gf == 0));

    const int nth = tile_team();
    std::vector<QbPlan> plan((size_t)njobs);
    int64_t tasks = 0, sub_max = 0, KC_max = 0, work = 0;
    for (int64_t j = 0; j < njobs; ++j) {
        const TileJob& jb = jobs[j];
        QbPlan& pl = plan[(size_t)j];
        const int64_t kc_want = jb.M <= QB_FEW_ROWS ? QB_KC_FEW : QB_KC;
        pl.KC = G >= kc_want ? G : kc_want / G * G;
        pl.np = (jb.N + QB_COLS - 1) / QB_COLS;
        tile_split(jb.M, pl.np, QB_MR, nth, &pl.sub, &pl.nsub, &pl.pg);
        pl.ngr = (pl.np + pl.pg - 1) / pl.pg;
        pl.t0 = tasks;
        tasks += pl.ngr * pl.nsub;
        sub_max = std::max(sub_max, pl.sub * pl.pg);
        KC_max = std::max(KC_max, pl.KC);
        work += jb.M * jb.N * jb.K;
    }
    /* sub_max is the rows of a task's C tiles, a sub-block's for each panel of its group. */
    const int64_t kpad_max = (KC_max + 31) / 32 * 32;
    const int64_t ng_max = (KC_max + G - 1) / G;
    if (zpw && KC_max > QB_KC_FEW) return false;

    /* A as bf16: read in place when it is bf16, unmapped, with every pair inside its row; else
     * converted into rows of `lda`, zero past the width. */
    const bool in_place = adt == RAD_BF16 && !arow && even;
    const int64_t lda = in_place ? ma.rs : (Kmax + 1) / 2 * 2;
    const size_t a_bytes = in_place ? 0 : (size_t)(arows * lda) * sizeof(uint16_t);
    const size_t sa_floats = (size_t)(arows * ngrp);
    uint8_t* blk = (uint8_t*)scratch_raw(a_bytes + sa_floats * sizeof(float) + 128);
    uint16_t* abuf = (uint16_t*)blk;
    float* satab = (float*)(blk + (a_bytes + 63) / 64 * 64);
    const uint16_t* Ab = in_place ? (const uint16_t*)ma.t->data + abase * ma.rs : abuf;
    const QbTab qa = qb_tab(adt == RAD_BF16 ? (uint32_t)RAD_I8 : adt);

#ifdef _OPENMP
    const TileSchedule sched_scope(njobs, jobs[0].M);
#pragma omp parallel num_threads(nth) if(work >= 65536 || avx_par_forced())
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int64_t r = 0; r < arows; ++r) {
            const int64_t m = arow ? arow[r] : abase + r;
            float* sr = satab + r * ngrp;
            for (int64_t g = 0; g < ngrp; ++g)
                sr[g] = sa_on && m >= 0 ? SA->grp(m, (g * G) / SA->group) : 1.0f;
            if (in_place) continue;
            uint16_t* d = abuf + r * lda;
            if (m < 0) {
                std::memset(d, 0, (size_t)lda * 2);
                continue;
            }
            const uint8_t* src = (const uint8_t*)byte_at(ma.t->data, adt, m * ma.rs);
            if (adt == RAD_BF16) {
                std::memcpy(d, src, (size_t)Kmax * 2);
                for (int64_t k = Kmax; k < lda; ++k) d[k] = 0;
                continue;
            }
            /* lda is even and qb_codes writes whole 32s, so the last block goes through a small
             * buffer rather than past the row. */
            const int64_t full = lda / 32 * 32;
            qb_codes(qa, src, std::min(Kmax, full), full, d);
            if (full < lda) {
                alignas(64) uint16_t tail[32];
                qb_codes(qa, src + full, Kmax - full, 32, tail);
                std::memcpy(d + full, tail, (size_t)(lda - full) * 2);
            }
        }

#ifdef _OPENMP
#pragma omp for schedule(runtime)
#endif
        for (int64_t t = 0; t < tasks; ++t) {
            int64_t ji = 0;
            for (int64_t lo = 0, hi = njobs - 1; lo <= hi;) {
                const int64_t mid = (lo + hi) / 2;
                if (plan[(size_t)mid].t0 <= t) { ji = mid; lo = mid + 1; } else hi = mid - 1;
            }
            const TileJob& jb = jobs[ji];
            const QbPlan& pl = plan[(size_t)ji];
            const int64_t tl = t - pl.t0, N = jb.N, K = jb.K, KC = pl.KC, sub = pl.sub;
            const int64_t s = tl / pl.ngr, pe = std::min(pl.np, (tl % pl.ngr) * pl.pg + pl.pg);
            const int64_t r0 = s * sub, r1 = std::min(jb.M, r0 + sub);
            const uint8_t* bbase = (const uint8_t*)jb.mb.t->data;
            auto brow = [&](int64_t n, int64_t k0) {
                if (zpw) return bbase + ((jb.b0 + n) * jb.mb.rs + k0) * zbits / 8;
                return (const uint8_t*)byte_at(bbase, bdt, (jb.b0 + n) * jb.mb.rs + k0);
            };
            /* Task buffers: the C tile, the packed chunk, its scales, the decoded rows of one
             * 16-row block, the bias and the store's row, and the groups' k pairs. */
            const size_t f_c = (size_t)(sub_max * QB_COLS);
            const size_t f_p = (size_t)((kpad_max / 2) * QB_COLS);
            const size_t f_s = (size_t)(ng_max * QB_COLS);
            const size_t f_t = (size_t)(16 * (kpad_max + 32) / 2);
            const size_t f_g = (size_t)(2 * ng_max + 16);
            float* C = scratch_f32(f_c + f_p + f_s + f_t + f_g + 2 * QB_COLS + 64);
            uint32_t* P = (uint32_t*)(C + f_c);
            float* S = C + f_c + f_p;
            uint16_t* T = (uint16_t*)(S + f_s);
            float* bv = S + f_s + f_t;
            float* tmp = bv + QB_COLS;
            int64_t* gk = (int64_t*)(((uintptr_t)(tmp + QB_COLS) + 7) & ~(uintptr_t)7);
            const QbTab qb = qb_tab(zpw ? (uint32_t)RAD_I8 : bdt);
            /* A zero-point grid's table, bf16(i - 32), and one row's zeros per 32 codes. */
            __m512i zt[2];
            int32_t zr[QB_KC_FEW / 32];
            if (zpw)
                for (int h = 0; h < 2; ++h) {
                    alignas(64) uint16_t w[32];
                    for (int j = 0; j < 32; ++j) w[j] = qb_hi((float)(32 * h + j - 32));
                    zt[h] = _mm512_load_si512((const void*)w);
                }
            const int64_t p0 = (tl % pl.ngr) * pl.pg;
            for (int64_t p = p0; p < pe; ++p)
                std::memset(C + (p - p0) * sub * QB_COLS, 0,
                            (size_t)((r1 - r0) * QB_COLS) * sizeof(float));
            /* K chunks outermost and the group's panels within (tile_split). */
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                for (int64_t p = p0; p < pe; ++p) {
                    const int64_t n0 = p * QB_COLS;
                    float* Cp = C + (p - p0) * sub * QB_COLS;
                    const int64_t k1 = std::min(K, k0 + KC), kc = k1 - k0;
                    const int64_t kpc = (kc + 1) / 2, kpad = (kc + 31) / 32 * 32;
                    /* The decoded rows' pitch, one line past the chunk: a power-of-two pitch
                     * puts the transpose's sixteen loads in one L1 set. */
                    const int64_t tld = kpad + 32;
                    const int64_t g0 = k0 / G, ng = (k1 - k0 + G - 1) / G;
                    /* The chunk's scales fold into its codes when all of them can (one byte a
                     * group, the rows' own: rblk is 1 for MXFP4). */
                    auto exps = [&](int64_t n) {
                        const Scales& S8 = jb.SB;
                        return (const uint8_t*)S8.p +
                               ((jb.b0 + n) / (S8.rblk > 1 ? S8.rblk : 1)) * S8.rs + g0;
                    };
                    bool fc = fold;
                    for (int64_t c = 0; fc && c < QB_COLS && n0 + c < N; ++c)
                        fc = qb_foldable(exps(n0 + c), ng);
                    const int64_t gc = fc ? (sa_on ? Gf : kc) : G;
                    const int64_t ngc = (kc + gc - 1) / gc;
                    for (int64_t gi = 0; gi < ngc; ++gi) {
                        const int64_t a0 = k0 + gi * gc, a1 = std::min(k1, a0 + gc);
                        gk[gi] = (a1 - a0 + 1) / 2;
                    }
                    /* ---- the chunk's pairs, sixteen panel columns at a time */
                    for (int64_t rb = 0; rb < QB_COLS; rb += 16) {
                        for (int64_t r = 0; r < 16; ++r) {
                            const int64_t n = n0 + rb + r;
                            uint16_t* tr = T + r * tld;
                            if (n + 4 < N && rb + r + 4 < QB_COLS)
                                tile_prefetch(brow(n + 4, k0), rad_dtype_bytes(bdt, kc));
                            if (n < N && zpw) {
                                for (int64_t j = 0; j < kpad / 32; ++j)
                                    zr[j] = jb.zero.p[(jb.b0 + n) * jb.zero.row_bytes +
                                                      (k0 + 32 * j) / G];
                                qb_codes_zp(brow(n, k0), zbits, kc, kpad, tr, zr, zt);
                            } else if (n < N) {
                                qb_codes(qb, brow(n, k0), kc, kpad, tr, fc ? exps(n) : nullptr);
                            } else {
                                std::memset(tr, 0, (size_t)kpad * 2);
                            }
                        }
                        for (int64_t kb = 0; kb < kpad / 2; kb += 16) {
                            __m512i rr[16];
                            for (int r = 0; r < 16; ++r)
                                rr[r] = _mm512_loadu_si512((const void*)(T + r * tld + 2 * kb));
                            tr16(rr);
                            const int64_t lim = std::min<int64_t>(16, kpc - kb);
                            for (int64_t j = 0; j < lim; ++j)
                                _mm512_store_si512((void*)(P + (kb + j) * QB_COLS + rb), rr[j]);
                        }
                    }
                    /* ---- the chunk's weight scales, [group][column]: sixteen columns' rows of
                     * scales staged in T (free once the pairs are stored), then transposed; ones
                     * where they are folded. */
                    const int64_t ngp = (ng + 15) / 16 * 16, sld = tld / 2;
                    float* SR = (float*)T;
                    if (fc)
                        for (int64_t i = 0; i < ngc * QB_COLS; ++i) S[i] = 1.0f;
                    for (int64_t rb = 0; !fc && rb < QB_COLS; rb += 16) {
                        for (int64_t r = 0; r < 16; ++r) {
                            const int64_t n = n0 + rb + r;
                            float* sr = SR + r * sld;
                            if (n < N && sb_on) jb.SB.row_groups(jb.b0 + n, g0, ng, sr);
                            else for (int64_t gi = 0; gi < ng; ++gi) sr[gi] = n < N ? 1.0f : 0.0f;
                            for (int64_t gi = ng; gi < ngp; ++gi) sr[gi] = 0.0f;
                        }
                        for (int64_t gb = 0; gb < ngp; gb += 16) {
                            __m512i rr[16];
                            for (int r = 0; r < 16; ++r)
                                rr[r] = _mm512_loadu_si512((const void*)(SR + r * sld + gb));
                            tr16(rr);
                            const int64_t lim = std::min<int64_t>(16, ng - gb);
                            for (int64_t j = 0; j < lim; ++j)
                                _mm512_storeu_si512((void*)(S + (gb + j) * QB_COLS + rb), rr[j]);
                        }
                    }
                    /* ---- every MR rows of the sub-block */
                    for (int64_t m0 = r0; m0 < r1; m0 += QB_MR) {
                        const int64_t mm = std::min(QB_MR, r1 - m0), ar = jb.a0 + m0;
                        kQbKernel[mm](Ab + ar * lda + k0, lda, P, S, gk, ngc,
                                      satab + ar * ngrp + g0, ngrp, gc / G, Cp + (m0 - r0) * QB_COLS);
                    }
                }
            }
            for (int64_t p = p0; p < pe; ++p) {
                const int64_t n0 = p * QB_COLS, cols = std::min(QB_COLS, N - n0);
                const float* Cp = C + (p - p0) * sub * QB_COLS;
                const bool hb = tile_bias(jb.bias, n0, cols, N, bv);
                for (int64_t m = r0; m < r1; ++m)
                    tile_store(Y, ys_r, 1, jb.y0 + m, jb.yc0 + n0, cols, Cp + (m - r0) * QB_COLS,
                               hb ? bv : nullptr, tmp);
            }
        }
    }
    return true;
}

/* One product as a batch of one job per super-block of A rows: a super-block's rows stay in the
 * L3 (16 MiB), and a call has few super-blocks, since each is a parallel region of its own. */
AVX_QB_TARGET static bool qb_run(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                                 const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N,
                                 int64_t K) {
    const bool in_place = ma.t->dtype == RAD_BF16 && !(K & 1);
    const int64_t lda = in_place ? ma.rs : (K + 1) / 2 * 2;
    int64_t mcb = (int64_t)(16 << 20) / (2 * std::max<int64_t>(lda, 1));
    mcb = std::max<int64_t>(64, std::min<int64_t>(mcb, 8192)) / QB_MR * QB_MR;
    if (mcb > M) mcb = M;
    for (int64_t mb0 = 0; mb0 < M; mb0 += mcb) {
        TileJob jb{};
        jb.mb = mb;
        if (SB) jb.SB = *SB;
        jb.a0 = 0; jb.M = std::min(mcb, M - mb0); jb.N = N; jb.K = K;
        jb.y0 = mb0; jb.yc0 = 0; jb.bias = bias;
        if (!qb_core(ma, mb0, nullptr, jb.M, SA, &jb, 1, Y, N)) return false;
    }
    return true;
}

/* Entered only where the CPU has AVX512_BF16; the target-specific bodies are functions of their
 * own, since a declaration and definition that differ in `target` would be two versions. */
bool gemm_q_bf16(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                 const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N, int64_t K) {
    static const bool have = __builtin_cpu_supports("avx512bf16");
    if (!have) return false;
    return qb_run(ma, mb, SA, SB, Y, bias, M, N, K);
}

bool gemm_q_bf16_jobs(const Mat& ma, const int64_t* arow, int64_t arows, const Scales* SA,
                      const TileJob* jobs, int64_t njobs, const RadTensor* Y, int64_t ycols) {
    static const bool have = __builtin_cpu_supports("avx512bf16");
    if (!have) return false;
    return qb_core(ma, 0, arow, arows, SA, jobs, njobs, Y, ycols);
}

#else
bool gemm_q_bf16(const Mat&, const Mat&, const Scales*, const Scales*, const RadTensor*,
                 const RadTensor*, int64_t, int64_t, int64_t) {
    return false;
}
bool gemm_q_bf16_jobs(const Mat&, const int64_t*, int64_t, const Scales*, const TileJob*, int64_t,
                      const RadTensor*, int64_t) {
    return false;
}
#endif

/* ================================================================== the f32 driver */
namespace {

/* Register block: MR rows by NV vectors of columns. AVX-512: 24 accumulators, three panel loads
 * and eight broadcasts per k. AVX2: the 6 x 16 block of 12 accumulators that fits 16 registers
 * with its two panel vectors and a broadcast. */
#if AVX_LEVEL == 3
#define TF_MR_MAX 8
constexpr int TF_NV = 3;
#elif AVX_LEVEL == 2
#define TF_MR_MAX 6
constexpr int TF_NV = 2;
#elif AVX_LEVEL == 1
#define TF_MR_MAX 4
constexpr int TF_NV = 2;
#else
#define TF_MR_MAX 4
constexpr int TF_NV = 4;
#endif
constexpr int TF_MR = TF_MR_MAX;
constexpr int64_t TF_NR = TF_NV * VF_N;    /* panel columns */
/* K chunk: sized for the L2 (96 KiB of panel at AVX-512), and longer where few rows share a
 * panel, as the bf16 driver's. */
constexpr int64_t TF_KC = 512, TF_KC_FEW = 2048, TF_FEW_ROWS = 32;

template <int MR>
static void tf_kernel(const float* A, int64_t lda, const float* P, int64_t kc, float* C) {
    vf acc[MR][TF_NV];
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < TF_NV; ++v) acc[i][v] = vf_loadu(C + i * TF_NR + v * VF_N);
    for (int64_t k = 0; k < kc; ++k) {
        vf b[TF_NV];
        for (int v = 0; v < TF_NV; ++v) b[v] = vf_load(P + k * TF_NR + v * VF_N);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) {
            const vf a = vf_set1(A[i * lda + k]);
            for (int v = 0; v < TF_NV; ++v) acc[i][v] = vf_fma(a, b[v], acc[i][v]);
        }
    }
#pragma GCC unroll 8
    for (int i = 0; i < MR; ++i)
        for (int v = 0; v < TF_NV; ++v) vf_storeu(C + i * TF_NR + v * VF_N, acc[i][v]);
}

using TfKernel = void (*)(const float*, int64_t, const float*, int64_t, float*);
static const TfKernel kTfKernel[] = {
    nullptr, tf_kernel<1>, tf_kernel<2>, tf_kernel<3>, tf_kernel<4>,
#if TF_MR_MAX > 4
    tf_kernel<5>, tf_kernel<6>,
#endif
#if TF_MR_MAX > 6
    tf_kernel<7>, tf_kernel<8>,
#endif
};

/* dst[k * TF_NR + c] = src[c * ld + k] for c < TF_NR, k < kn: the panel's k-major order from its
 * rows, whole 16 x 16 (AVX-512) or 8 x 8 blocks in registers and the last k one at a time. */
static void tf_transpose(const float* src, int64_t ld, int64_t kn, float* dst) {
    int64_t k = 0;
#if AVX_LEVEL == 3
    for (; k + 16 <= kn; k += 16)
        for (int64_t c0 = 0; c0 < TF_NR; c0 += 16) {
            __m512i r[16];
            for (int i = 0; i < 16; ++i)
                r[i] = _mm512_castps_si512(_mm512_loadu_ps(src + (c0 + i) * ld + k));
            /* The same 16 x 16 transpose as the bf16 driver's, on f32 bits. */
            __m512i t[16], u[16], v[16];
            for (int i = 0; i < 8; ++i) {
                t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
                t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
            }
            for (int q = 0; q < 4; ++q) {
                u[4 * q + 0] = _mm512_unpacklo_epi64(t[4 * q], t[4 * q + 2]);
                u[4 * q + 1] = _mm512_unpackhi_epi64(t[4 * q], t[4 * q + 2]);
                u[4 * q + 2] = _mm512_unpacklo_epi64(t[4 * q + 1], t[4 * q + 3]);
                u[4 * q + 3] = _mm512_unpackhi_epi64(t[4 * q + 1], t[4 * q + 3]);
            }
            for (int c = 0; c < 4; ++c) {
                v[c] = _mm512_shuffle_i32x4(u[c], u[4 + c], 0x88);
                v[4 + c] = _mm512_shuffle_i32x4(u[c], u[4 + c], 0xdd);
                v[8 + c] = _mm512_shuffle_i32x4(u[8 + c], u[12 + c], 0x88);
                v[12 + c] = _mm512_shuffle_i32x4(u[8 + c], u[12 + c], 0xdd);
            }
            for (int c = 0; c < 4; ++c) {
                r[c] = _mm512_shuffle_i32x4(v[c], v[8 + c], 0x88);
                r[8 + c] = _mm512_shuffle_i32x4(v[c], v[8 + c], 0xdd);
                r[4 + c] = _mm512_shuffle_i32x4(v[4 + c], v[12 + c], 0x88);
                r[12 + c] = _mm512_shuffle_i32x4(v[4 + c], v[12 + c], 0xdd);
            }
            for (int j = 0; j < 16; ++j)
                _mm512_store_ps(dst + (k + j) * TF_NR + c0, _mm512_castsi512_ps(r[j]));
        }
#elif AVX_LEVEL >= 1
    for (; k + 8 <= kn; k += 8)
        for (int64_t c0 = 0; c0 < TF_NR; c0 += 8) {
            __m256 r[8], t[8], u[8];
            for (int i = 0; i < 8; ++i) r[i] = _mm256_loadu_ps(src + (c0 + i) * ld + k);
            for (int i = 0; i < 4; ++i) {
                t[2 * i] = _mm256_unpacklo_ps(r[2 * i], r[2 * i + 1]);
                t[2 * i + 1] = _mm256_unpackhi_ps(r[2 * i], r[2 * i + 1]);
            }
            /* u[4h + c], 128-bit lane L: column 4L + c of rows 4h .. 4h + 3. */
            for (int h = 0; h < 2; ++h) {
                u[4 * h + 0] = _mm256_shuffle_ps(t[4 * h], t[4 * h + 2], 0x44);
                u[4 * h + 1] = _mm256_shuffle_ps(t[4 * h], t[4 * h + 2], 0xee);
                u[4 * h + 2] = _mm256_shuffle_ps(t[4 * h + 1], t[4 * h + 3], 0x44);
                u[4 * h + 3] = _mm256_shuffle_ps(t[4 * h + 1], t[4 * h + 3], 0xee);
            }
            for (int c = 0; c < 4; ++c) {
                _mm256_store_ps(dst + (k + c) * TF_NR + c0, _mm256_permute2f128_ps(u[c], u[4 + c], 0x20));
                _mm256_store_ps(dst + (k + 4 + c) * TF_NR + c0,
                                _mm256_permute2f128_ps(u[c], u[4 + c], 0x31));
            }
        }
#endif
    for (; k < kn; ++k)
        for (int64_t c = 0; c < TF_NR; ++c) dst[k * TF_NR + c] = src[c * ld + k];
}

/* The dot-product form for a few rows: MRD rows of A against NRD decoded weight rows, each output
 * a vector of partial sums (lane l holds the k = l mod VF_N terms, in k order) carried in `V`
 * between chunks, [row][TF_NR] vectors. No transpose: with eight rows or fewer an outer product
 * would reuse each panel load too few times to pay for one. `kv` is a multiple of VF_N; both
 * operands are zero past the chunk. */
template <int MRD, int NRD>
static void td_kernel(const float* A, int64_t lda, const float* T, int64_t ldt, int64_t kv,
                      float* V) {
    vf acc[MRD][NRD];
    for (int i = 0; i < MRD; ++i)
        for (int j = 0; j < NRD; ++j) acc[i][j] = vf_loadu(V + (i * TF_NR + j) * VF_N);
    for (int64_t k = 0; k < kv; k += VF_N) {
        vf a[MRD];
        for (int i = 0; i < MRD; ++i) a[i] = vf_loadu(A + i * lda + k);
        for (int j = 0; j < NRD; ++j) {
            const vf b = vf_loadu(T + j * ldt + k);
            for (int i = 0; i < MRD; ++i) acc[i][j] = vf_fma(a[i], b, acc[i][j]);
        }
    }
    for (int i = 0; i < MRD; ++i)
        for (int j = 0; j < NRD; ++j) vf_storeu(V + (i * TF_NR + j) * VF_N, acc[i][j]);
}

/* Rows of A for which the dot-product form runs, and its blocks: two rows by four weight rows
 * (eight accumulators, two A loads and four weight loads a step) where the 16 registers allow,
 * four by four at AVX-512, one by eight for a last single row. */
constexpr int64_t TF_DOT_ROWS = 8;
#if AVX_LEVEL == 3
constexpr int TD_MR = 4;
#else
constexpr int TD_MR = 2;
#endif

}  /* namespace */

/* A job's task layout, fixed before the parallel region. */
struct TfPlan {
    int64_t np, sub, nsub, pg, ngr, t0, KC;
    bool dot;
};

/* The batch (avx_gemm.h: TileJob) on the f32 driver, as qb_core's. */
static bool tf_core(const Mat& ma, int64_t abase, const int64_t* arow, int64_t arows,
                    const Scales* SA, const TileJob* jobs, int64_t njobs, const RadTensor* Y,
                    int64_t ycols) {
    if (njobs <= 0) return true;
    if (arows <= 0) return false;
    const bool sa_on = SA && SA->p, sb_on = jobs[0].SB.p != nullptr;
    int64_t Kmax = 0;
    bool any_dot = false;
    for (int64_t j = 0; j < njobs; ++j) {
        const TileJob& jb = jobs[j];
        if ((jb.SB.p != nullptr) != sb_on || (sb_on && jb.mb.dec)) return false;
        Kmax = std::max(Kmax, jb.K);
        any_dot = any_dot || jb.M <= TF_DOT_ROWS;
    }
    /* A code times an E8M0 scale 2^(e - 127) is exact in f32 up to e = 252; at 253 and 254 it can
     * overflow where libref's grouped sum would not. No weight has such scales, so the tasks only
     * note one, and the call is then left to the grouped driver, which rewrites all of Y. */
    const bool e8 = sb_on && (jobs[0].SB.e8m0 || jobs[0].SB.dt == RAD_E8M0);
    std::atomic<bool> unfoldable{ false };
    const int64_t ys_r = row_stride(Y, ycols), ys_c = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;
    const int nth = tile_team();

    std::vector<TfPlan> plan((size_t)njobs);
    int64_t tasks = 0, sub_max = 0, kpad = 0, ng_max = 0, work = 0;
    for (int64_t j = 0; j < njobs; ++j) {
        const TileJob& jb = jobs[j];
        TfPlan& pl = plan[(size_t)j];
        pl.dot = jb.M <= TF_DOT_ROWS;
        /* The chunk is whole decoder spans (a joined head's scale block, a grid's group) and
         * starts on a byte of any packed code; the dot-product form's lanes start each chunk on a
         * multiple of 16 columns. */
        int64_t unit = jb.mb.dec ? jb.mb.dec->align() : 64;
        if (pl.dot) unit = unit / std::gcd<int64_t>(unit, 16) * 16;
        const int64_t kcw = jb.M <= TF_FEW_ROWS ? TF_KC_FEW : TF_KC;
        pl.KC = unit >= kcw ? unit : kcw / unit * unit;
        pl.np = (jb.N + TF_NR - 1) / TF_NR;
        tile_split(jb.M, pl.np, TF_MR, nth, &pl.sub, &pl.nsub, &pl.pg);
        pl.ngr = (pl.np + pl.pg - 1) / pl.pg;
        pl.t0 = tasks;
        tasks += pl.ngr * pl.nsub;
        sub_max = std::max(sub_max, pl.sub * pl.pg);    /* a task's C tiles */
        /* The widened rows' pitch: one line past the chunk, so the transpose's loads do not share
         * an L1 set (see the bf16 driver). */
        kpad = std::max(kpad, (pl.KC + 15) / 16 * 16 + 16);
        const int64_t gb = sb_on && jb.SB.group > 0 ? jb.SB.group : jb.K;
        ng_max = std::max(ng_max, pl.KC / gb + 2);
        work += jb.M * jb.N * jb.K;
    }

    /* A in place when it is dense f32, unmapped and unscaled; else widened (and scaled) into
     * rows zero to a multiple of 16, since the dot-product form reads whole vectors. */
    const bool in_place = ma.t->dtype == RAD_F32 && ma.cs == 1 && !sa_on && !ma.dec && !arow &&
                          !any_dot;
    const int64_t lda = in_place ? ma.rs : (Kmax + 15) / 16 * 16;
    float* abuf = in_place ? nullptr
                           : (float*)scratch_raw((size_t)(arows * lda) * sizeof(float) + 64);
    const float* Ab = in_place ? (const float*)ma.t->data + abase * ma.rs : abuf;
    const int64_t sag = sa_on && SA->group > 0 ? SA->group : Kmax;

#ifdef _OPENMP
    const TileSchedule sched_scope(njobs, jobs[0].M);
#pragma omp parallel num_threads(nth) if(work >= 65536 || avx_par_forced())
#endif
    {
        if (!in_place) {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t r = 0; r < arows; ++r) {
                const int64_t m = arow ? arow[r] : abase + r;
                float* d = abuf + r * lda;
                if (m < 0) {
                    std::memset(d, 0, (size_t)lda * sizeof(float));
                    continue;
                }
                if (ma.dec) ma.dec->row(m, d);
                else widen_cols(ma, m, 0, Kmax, d);
                for (int64_t k = Kmax; k < lda; ++k) d[k] = 0.0f;
                if (!sa_on) continue;
                for (int64_t k0 = 0; k0 < Kmax; k0 += sag) {
                    const float s = SA->grp(m, k0 / sag);
                    const int64_t k1 = std::min(Kmax, k0 + sag);
                    const vf vs = vf_set1(s);
                    int64_t k = k0;
                    for (; k + VF_N <= k1; k += VF_N) vf_storeu(d + k, vf_mul(vf_loadu(d + k), vs));
                    for (; k < k1; ++k) d[k] *= s;
                }
            }
        }

#ifdef _OPENMP
#pragma omp for schedule(runtime)
#endif
        for (int64_t t = 0; t < tasks; ++t) {
            int64_t ji = 0;
            for (int64_t lo = 0, hi = njobs - 1; lo <= hi;) {
                const int64_t mid = (lo + hi) / 2;
                if (plan[(size_t)mid].t0 <= t) { ji = mid; lo = mid + 1; } else hi = mid - 1;
            }
            const TileJob& jb = jobs[ji];
            const TfPlan& pl = plan[(size_t)ji];
            const Mat& mb = jb.mb;
            const int64_t tl = t - pl.t0, N = jb.N, K = jb.K, KC = pl.KC, sub = pl.sub;
            const bool dot = pl.dot;
            const int64_t gb = sb_on && jb.SB.group > 0 ? jb.SB.group : K;
            const int64_t s = tl / pl.ngr, pe = std::min(pl.np, (tl % pl.ngr) * pl.pg + pl.pg);
            const int64_t r0 = s * sub, r1 = std::min(jb.M, r0 + sub);
            const size_t f_c = (size_t)(sub_max * TF_NR * VF_N);
            const size_t f_p = (size_t)(kpad * TF_NR);
            const size_t f_t = (size_t)(TF_NR * kpad);
            float* C = scratch_f32(f_c + f_p + f_t + (size_t)ng_max + 2 * TF_NR + 64);
            float* P = C + (f_c + 15) / 16 * 16;
            float* T = P + f_p;
            float* gs = T + f_t;
            float* bv = gs + ng_max;
            float* tmp = bv + TF_NR;
            const int64_t p0 = (tl % pl.ngr) * pl.pg, ct = sub * TF_NR * (dot ? VF_N : 1);
            for (int64_t p = p0; p < pe; ++p)
                std::memset(C + (p - p0) * ct, 0,
                            (size_t)((r1 - r0) * TF_NR * (dot ? VF_N : 1)) * sizeof(float));
            /* K chunks outermost and the group's panels within (tile_split). */
            for (int64_t k0 = 0; k0 < K; k0 += KC) {
                for (int64_t p = p0; p < pe; ++p) {
                    const int64_t n0 = p * TF_NR;
                    float* Cp = C + (p - p0) * ct;
                    const int64_t k1 = std::min(K, k0 + KC), kc = k1 - k0;
                    /* ---- the panel's rows, widened (and scaled), then k-major */
                    for (int64_t c = 0; c < TF_NR; ++c) {
                        const int64_t n = n0 + c, wr = jb.b0 + n;
                        float* tr = T + c * kpad;
                        if (n >= N) { std::memset(tr, 0, (size_t)kpad * sizeof(float)); continue; }
                        if (c + 4 < TF_NR && n + 4 < N) {
                            const void* pp = nullptr;
                            int64_t pb = 0;
                            if (mb.dec) {
                                mb.dec->codes(wr + 4, k0, k1, &pp, &pb);
                            } else if (mb.cs == 1) {
                                pp = byte_at(mb.t->data, mb.t->dtype, (wr + 4) * mb.rs + k0);
                                pb = rad_dtype_bytes(mb.t->dtype, kc);
                            }
                            if (pp) tile_prefetch(pp, pb);
                        }
                        if (mb.dec) mb.dec->cols(wr, k0, k1, tr);
                        else widen_cols(mb, wr, k0, kc, tr);
                        for (int64_t k = kc; k < kpad; ++k) tr[k] = 0.0f;
                        if (!sb_on) continue;
                        const int64_t g0 = k0 / gb, g1 = (k1 - 1) / gb;
                        jb.SB.row_groups(wr, g0, g1 - g0 + 1, gs);
                        if (e8)
                            for (int64_t g = 0; g <= g1 - g0; ++g)
                                if (gs[g] >= 0x1p126f) unfoldable.store(true, std::memory_order_relaxed);
                        for (int64_t g = g0; g <= g1; ++g) {
                            const int64_t a0 = std::max(k0, g * gb) - k0;
                            const int64_t a1 = std::min(k1, (g + 1) * gb) - k0;
                            const vf vs = vf_set1(gs[g - g0]);
                            int64_t k = a0;
                            for (; k + VF_N <= a1; k += VF_N)
                                vf_storeu(tr + k, vf_mul(vf_loadu(tr + k), vs));
                            for (; k < a1; ++k) tr[k] *= gs[g - g0];
                        }
                    }
                    if (dot) {
                        const int64_t kv = (kc + VF_N - 1) / VF_N * VF_N;
                        int64_t m0 = r0;
                        for (; m0 + TD_MR <= r1; m0 += TD_MR)
                            for (int64_t c0 = 0; c0 < TF_NR; c0 += 4)
                                td_kernel<TD_MR, 4>(Ab + (jb.a0 + m0) * lda + k0, lda,
                                                    T + c0 * kpad, kpad, kv,
                                                    Cp + ((m0 - r0) * TF_NR + c0) * VF_N);
                        for (; m0 < r1; ++m0)
                            for (int64_t c0 = 0; c0 < TF_NR; c0 += 4)
                                td_kernel<1, 4>(Ab + (jb.a0 + m0) * lda + k0, lda, T + c0 * kpad,
                                                kpad, kv, Cp + ((m0 - r0) * TF_NR + c0) * VF_N);
                        continue;
                    }
                    tf_transpose(T, kpad, kc, P);
                    for (int64_t m0 = r0; m0 < r1; m0 += TF_MR) {
                        const int64_t mm = std::min<int64_t>(TF_MR, r1 - m0);
                        kTfKernel[mm](Ab + (jb.a0 + m0) * lda + k0, lda, P, kc,
                                      Cp + (m0 - r0) * TF_NR);
                    }
                }
            }
            for (int64_t p = p0; p < pe; ++p) {
                const int64_t n0 = p * TF_NR, cols = std::min(TF_NR, N - n0);
                float* Cp = C + (p - p0) * ct;
                if (dot) {
                    /* Each output's partial sums, reduced by the fixed tree of vf_hsum. */
                    for (int64_t i = 0; i < (r1 - r0) * TF_NR; ++i)
                        Cp[i] = vf_hsum(vf_loadu(Cp + i * VF_N));
                }
                const bool hb = tile_bias(jb.bias, n0, cols, N, bv);
                for (int64_t m = r0; m < r1; ++m)
                    tile_store(Y, ys_r, ys_c, jb.y0 + m, jb.yc0 + n0, cols, Cp + (m - r0) * TF_NR,
                               hb ? bv : nullptr, tmp);
            }
        }
    }
    return !unfoldable.load(std::memory_order_relaxed);
}

bool gemm_tile_f32(const Mat& ma, const Mat& mb, const Scales* SA, const Scales* SB,
                   const RadTensor* Y, const RadTensor* bias, int64_t M, int64_t N, int64_t K) {
    /* One job per super-block of A rows, which stays in the L3 (16 MiB), as qb_run's. */
    int64_t mcb = (int64_t)(16 << 20) / (4 * std::max<int64_t>(K, 1));
    mcb = std::max<int64_t>(64, std::min<int64_t>(mcb, 8192)) / TF_MR * TF_MR;
    if (mcb > M) mcb = M;
    bool ok = true;
    for (int64_t mb0 = 0; mb0 < M; mb0 += mcb) {
        TileJob jb{};
        jb.mb = mb;
        if (SB) jb.SB = *SB;
        jb.a0 = 0; jb.M = std::min(mcb, M - mb0); jb.N = N; jb.K = K;
        jb.y0 = mb0; jb.yc0 = 0; jb.bias = bias;
        ok = tf_core(ma, mb0, nullptr, jb.M, SA, &jb, 1, Y, N) && ok;
    }
    return ok;
}

bool gemm_tile_f32_jobs(const Mat& ma, const int64_t* arow, int64_t arows, const Scales* SA,
                        const TileJob* jobs, int64_t njobs, const RadTensor* Y, int64_t ycols) {
    return tf_core(ma, 0, arow, arows, SA, jobs, njobs, Y, ycols);
}

}  /* inline namespace AVX_SUFFIX */
}  /* namespace avx */
