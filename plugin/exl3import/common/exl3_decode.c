/* exl3_decode.c -- the oracle. See exl3_decode.h for the format and the conventions.
 *
 * Bit-level and scalar on purpose: this file is what every faster path is checked against, so it is
 * written to be read against exllamav3's codebook.cuh and Dylan's trellis.py line by line, not to
 * be fast. Compile with -ffp-contract=off (an FMA would change W's f32 roundings; W_hat is exact
 * either way).
 */
#include "exl3_decode.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ fp16 */

float exl3_half_to_float(uint16_t h) {
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);                 /* zero and subnormals */
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), e - 25);           /* (1024 + m) * 2^(e - 25) */
    return (h & 0x8000) ? -v : v;
}

uint16_t exl3_half_from_double(double x) {
    const uint16_t sign = signbit(x) ? 0x8000 : 0;
    const double a = fabs(x);
    double m;
    int e;
    if (isnan(x)) return 0x7E00;
    if (a >= 65520.0) return sign | 0x7C00;                /* past max + half an ulp: inf (RNE) */
    if (a < 0x1p-14) {                                     /* subnormal: units of 2^-24 */
        return sign | (uint16_t)nearbyint(a * 0x1p24);     /* 1024 lands on the smallest normal */
    }
    m = nearbyint(ldexp(frexp(a, &e), 11));                /* a = m * 2^(e - 11), m in [1024, 2048] */
    if (m == 2048.0) { m = 1024.0; ++e; }
    if (e + 14 >= 31) return sign | 0x7C00;
    return sign | (uint16_t)((e + 14) << 10) | (uint16_t)(m - 1024.0);
}

/* ------------------------------------------------------------------ codebook */

/* mul1's two fp16 constants, by their bits (codebook.cuh): K_INV ~ 1/147.7, K_BIAS ~ -10.39. */
#define EXL3_MUL1_K_INV  0x1EEEu
#define EXL3_MUL1_K_BIAS 0xC931u

uint16_t exl3_codebook_value(uint32_t state, int codebook) {
    uint32_t x = state & 0xFFFFu;
    if (codebook == EXL3_CB_MUL1) {
        /* h = x * 0x83DCD12D; fp16(1024 + byte sum of h) * K_INV + K_BIAS in ONE fp16 FMA. The
         * product and the sum are exact in double, so one rounding to fp16 is exactly __hfma. */
        uint32_t s;
        x *= EXL3_MUL1_MARKER;
        s = (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + (x >> 24);
        return exl3_half_from_double((1024.0 + (double)s) *
                                         (double)exl3_half_to_float(EXL3_MUL1_K_INV) +
                                     (double)exl3_half_to_float(EXL3_MUL1_K_BIAS));
    }
    if (codebook == EXL3_CB_MCG) x *= EXL3_MCG_MARKER;
    else if (codebook == EXL3_CB_3INST) x = x * 89226354u + 64248484u;
    else return 0x7E00;                                    /* not a codebook: nan, never a number */
    /* lop3 0x6a = (x & 0x8FFF8FFF) ^ 0x3B603B60, then __hadd of the two fp16 halves: exact sum in
     * double, one rounding. The mask keeps every half's exponent below 31: never inf or nan. */
    x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return exl3_half_from_double((double)exl3_half_to_float((uint16_t)(x & 0xFFFFu)) +
                                 (double)exl3_half_to_float((uint16_t)(x >> 16)));
}

/* ------------------------------------------------------------------ the tile */

void exl3_pos_to_tile(int p, int* r, int* c) {
    static const int off[4] = { 0, 1, 8, 9 };
    const int lane = p / 8, j = p % 8;
    *r = 2 * (lane % 4) + off[j % 4];
    *c = lane / 4 + 8 * (j / 4);
}

int exl3_tile_to_pos(int r, int c) {
    const int lane = 4 * (c % 8) + (r % 8) / 2;
    const int j = 4 * (c / 8) + ((r & 1) | ((r >> 3) << 1));
    return 8 * lane + j;
}

/* Stream bit b (MSB-first) of a tile: bit 31 - b%32 of little-endian u32 word b/32. */
static uint32_t stream_bit(const uint16_t* tile, int b) {
    const uint32_t w = (uint32_t)tile[2 * (b >> 5)] | ((uint32_t)tile[2 * (b >> 5) + 1] << 16);
    return (w >> (31 - (b & 31))) & 1u;
}

void exl3_tile_symbols(const uint16_t* tile, int bits, uint8_t sym[EXL3_TILE_POS]) {
    int p, i;
    for (p = 0; p < EXL3_TILE_POS; ++p) {
        uint32_t v = 0;
        for (i = 0; i < bits; ++i) v = (v << 1) | stream_bit(tile, p * bits + i);
        sym[p] = (uint8_t)v;
    }
}

void exl3_tile_pack(const uint8_t sym[EXL3_TILE_POS], int bits, uint16_t* tile) {
    int p, i;
    memset(tile, 0, (size_t)(16 * bits) * sizeof *tile);
    for (p = 0; p < EXL3_TILE_POS; ++p)
        for (i = 0; i < bits; ++i) {
            const int b = p * bits + i, shift = 31 - (b & 31);
            if (!((sym[p] >> (bits - 1 - i)) & 1)) continue;
            if (shift >= 16) tile[2 * (b >> 5) + 1] |= (uint16_t)(1u << (shift - 16));
            else tile[2 * (b >> 5)] |= (uint16_t)(1u << shift);
        }
}

void exl3_tile_states(const uint8_t sym[EXL3_TILE_POS], int bits, uint16_t state[EXL3_TILE_POS]) {
    const int back = (16 + bits - 1) / bits;          /* symbols a 16-bit window reaches into */
    const uint32_t mask = (1u << bits) - 1u;
    int p, i;
    for (p = 0; p < EXL3_TILE_POS; ++p) {
        uint32_t acc = 0;                             /* at most 16 + bits - 1 <= 23 bits */
        for (i = back - 1; i >= 0; --i)
            acc = (acc << bits) | (sym[(p - i + EXL3_TILE_POS) % EXL3_TILE_POS] & mask);
        state[p] = (uint16_t)(acc & 0xFFFFu);
    }
}

static int args_ok(int bits, int codebook) {
    return bits >= EXL3_BITS_MIN && bits <= EXL3_BITS_MAX && exl3_codebook_name(codebook) != 0;
}

int exl3_trellis_decode(const uint16_t* trellis, int64_t kt, int64_t nt, int bits, int codebook,
                        uint16_t* w_hat) {
    uint8_t sym[EXL3_TILE_POS];
    uint16_t st[EXL3_TILE_POS];
    int64_t tk, tn;
    int p, r, c;
    if (!trellis || !w_hat || kt < 0 || nt < 0 || !args_ok(bits, codebook)) return EXL3_E_ARG;
    for (tk = 0; tk < kt; ++tk)
        for (tn = 0; tn < nt; ++tn) {
            exl3_tile_symbols(trellis + (tk * nt + tn) * 16 * bits, bits, sym);
            exl3_tile_states(sym, bits, st);
            for (p = 0; p < EXL3_TILE_POS; ++p) {
                exl3_pos_to_tile(p, &r, &c);
                w_hat[(tk * 16 + r) * (nt * 16) + tn * 16 + c] = exl3_codebook_value(st[p], codebook);
            }
        }
    return EXL3_OK;
}

/* ------------------------------------------------------------------ the codes plane */

int64_t exl3_codes_row_bytes(int64_t cols, int bits) { return (cols * bits + 7) / 8; }

uint32_t exl3_codes_get(const uint8_t* codes, int64_t row_bytes, int bits, int64_t i, int64_t j) {
    const uint8_t* row = codes + i * row_bytes;
    int64_t b = j * bits;
    uint32_t v = 0;
    int k;
    for (k = 0; k < bits; ++k, ++b) v |= (uint32_t)((row[b >> 3] >> (b & 7)) & 1u) << k;
    return v;
}

void exl3_codes_set(uint8_t* codes, int64_t row_bytes, int bits, int64_t i, int64_t j,
                    uint32_t value) {
    uint8_t* row = codes + i * row_bytes;
    int64_t b = j * bits;
    int k;
    for (k = 0; k < bits; ++k, ++b) {
        const uint8_t m = (uint8_t)(1u << (b & 7));
        row[b >> 3] = (uint8_t)(((value >> k) & 1u) ? (row[b >> 3] | m) : (row[b >> 3] & ~m));
    }
}

/* ------------------------------------------------------------------ the rotation */

int exl3_had128(float* v, int64_t n) {
    const float scale = 0.08838834764831844055f;        /* 1/sqrt(128) */
    int64_t base, i, j;
    int h;
    if (!v || n < 0 || n % EXL3_HAD) return EXL3_E_SHAPE;
    for (base = 0; base < n; base += EXL3_HAD) {
        float* x = v + base;
        for (h = 1; h < EXL3_HAD; h <<= 1)
            for (i = 0; i < EXL3_HAD; i += 2 * h)
                for (j = i; j < i + h; ++j) {
                    const float a = x[j], b = x[j + h];
                    x[j] = a + b;
                    x[j + h] = a - b;
                }
        for (i = 0; i < EXL3_HAD; ++i) x[i] *= scale;
    }
    return EXL3_OK;
}

/* ------------------------------------------------------------------ one weight */

int exl3_decode_what_rows(const Exl3View* v, int64_t row0, int64_t n, float* out) {
    uint8_t sym[EXL3_TILE_POS];
    uint16_t st[EXL3_TILE_POS];
    int tr[EXL3_TILE_POS], tc[EXL3_TILE_POS];
    int64_t rb, tn, tk;
    int p;
    if (!v || !v->codes || !out || !args_ok(v->bits, v->codebook)) return EXL3_E_ARG;
    if (v->rows <= 0 || v->cols <= 0 || v->rows % EXL3_TILE || v->cols % EXL3_TILE ||
        row0 < 0 || n < 0 || row0 + n > v->rows)
        return EXL3_E_SHAPE;
    rb = v->codes_row_bytes ? v->codes_row_bytes : exl3_codes_row_bytes(v->cols, v->bits);
    for (p = 0; p < EXL3_TILE_POS; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (tn = row0 / EXL3_TILE; tn * EXL3_TILE < row0 + n; ++tn)
        for (tk = 0; tk < v->cols / EXL3_TILE; ++tk) {
            /* the tile's symbols, gathered from where the canonical plane keeps them:
             * position p decodes W_hat[k = tk*16 + r][n = tn*16 + c] = plane element (n, k) */
            for (p = 0; p < EXL3_TILE_POS; ++p)
                sym[p] = (uint8_t)exl3_codes_get(v->codes, rb, v->bits, tn * EXL3_TILE + tc[p],
                                                 tk * EXL3_TILE + tr[p]);
            exl3_tile_states(sym, v->bits, st);
            for (p = 0; p < EXL3_TILE_POS; ++p) {
                const int64_t i = tn * EXL3_TILE + tc[p];
                if (i < row0 || i >= row0 + n) continue;
                out[(i - row0) * v->cols + tk * EXL3_TILE + tr[p]] =
                    exl3_half_to_float(exl3_codebook_value(st[p], v->codebook));
            }
        }
    return EXL3_OK;
}

/* One 128-row block [b0, b0 + 128) of W into blk, f32 [128][cols], in exllamav3's order:
 * W_hat, had over k (along each row here), x suh, had over n (down each column), x svh. */
static int decode_block(const Exl3View* v, int64_t b0, float* blk, float* col) {
    const int64_t K = v->cols;
    int64_t r, k;
    int rc = exl3_decode_what_rows(v, b0, EXL3_HAD, blk);
    if (rc != EXL3_OK) return rc;
    for (r = 0; r < EXL3_HAD; ++r) {
        float* row = blk + r * K;
        const uint16_t* su = (v->suh1 && b0 + r >= v->rows / 2) ? v->suh1 : v->suh;
        exl3_had128(row, K);
        for (k = 0; k < K; ++k) row[k] *= exl3_half_to_float(su[k]);
    }
    for (k = 0; k < K; ++k) {
        for (r = 0; r < EXL3_HAD; ++r) col[r] = blk[r * K + k];
        exl3_had128(col, EXL3_HAD);
        for (r = 0; r < EXL3_HAD; ++r) blk[r * K + k] = col[r];
    }
    for (r = 0; r < EXL3_HAD; ++r) {
        const float s = exl3_half_to_float(v->svh[b0 + r]);
        for (k = 0; k < K; ++k) blk[r * K + k] *= s;
    }
    return EXL3_OK;
}

int exl3_decode_rows(const Exl3View* v, int64_t row0, int64_t n, float* out) {
    float col[EXL3_HAD];
    float* blk;
    int64_t b0;
    int rc = EXL3_OK;
    if (!v || !v->codes || !v->suh || !v->svh || !out || !args_ok(v->bits, v->codebook))
        return EXL3_E_ARG;
    if (v->rows <= 0 || v->cols <= 0 || v->rows % EXL3_HAD || v->cols % EXL3_HAD ||
        (v->suh1 && v->rows % (2 * EXL3_HAD)) || row0 < 0 || n < 0 || row0 + n > v->rows)
        return EXL3_E_SHAPE;
    if (n == 0) return EXL3_OK;
    blk = (float*)malloc((size_t)(EXL3_HAD * v->cols) * sizeof *blk);
    if (!blk) return EXL3_E_NOMEM;
    for (b0 = row0 / EXL3_HAD * EXL3_HAD; b0 < row0 + n && rc == EXL3_OK; b0 += EXL3_HAD) {
        const int64_t lo = b0 > row0 ? b0 : row0;
        const int64_t hi = b0 + EXL3_HAD < row0 + n ? b0 + EXL3_HAD : row0 + n;
        rc = decode_block(v, b0, blk, col);
        if (rc == EXL3_OK)
            memcpy(out + (lo - row0) * v->cols, blk + (lo - b0) * v->cols,
                   (size_t)((hi - lo) * v->cols) * sizeof *out);
    }
    free(blk);
    return rc;
}
