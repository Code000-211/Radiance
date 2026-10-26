/* exl3_decode.h -- THE ORACLE: what an EXL3 weight's numbers are, in plain C99, no dependencies.
 *
 * Everything another library checks itself against goes through here: the host reference kernels
 * (kernels/exl3ref), the quantiser's decode hook (quantizers/exl3) and the property tests. It is a
 * transcription of exllamav3's host-side decode as Dylan's exl3-moe recorded it (kernels/trellis.py,
 * from exllamav3 @ 12414d0) and is checked bit for bit against the golden decode vllm-exl3's host
 * code made (common/tests/data/exl3_golden.npz). Slow and readable on purpose.
 *
 * ============================================================ THE FORMAT (one linear, y = x @ W)
 *
 * exllamav3 stores a weight W of shape [k, n] (k = in_features, n = out_features) as
 *
 *     W = diag(suh) . Hb . W_hat . Hb . diag(svh)
 *
 * Hb the block-diagonal natural-order Sylvester Hadamard of block 128, scaled 1/sqrt(128); suh f16 [k]
 * and svh f16 [n] per-channel scales; W_hat [k, n] the trellis-coded values, cut into 16x16 tiles.
 * Tile (tk, tn) is 16*K int16 = 8*K little-endian u32 words, read as ONE MSB-first stream of 256*K
 * bits. Position p (0..255) owns stream bits [p*K, (p+1)*K) -- its K-bit SYMBOL -- and its 16-bit
 * STATE is the 16-bit window ending at bit (p+1)*K, circular within the tile. A codebook maps the
 * state to an fp16 value. Position p = 8*L + j (L 0..31, j 0..7) sits at tile row (k axis)
 * 2*(L%4) + {0,1,8,9}[j%4] and tile column (n axis) L/4 + 8*(j/4).
 *
 * ============================================================ RADIANCE'S VIEW (the canonical planes)
 *
 * radiance stores a weight as [rows, cols] = [n, k]: row i is OUTPUT feature n = i, column j is
 * INPUT feature k = j (the transpose of exllamav3's [k, n]). The canonical EXL3 encoding
 * (include/exl3_encoding.h, PLAN.md D4) holds, at codes-plane element (i, j), the K-bit SYMBOL of
 * the trellis position that decodes weight element W_hat[k = j][n = i]: tile (tk = j/16, tn = i/16),
 * position exl3_tile_to_pos(j % 16, i % 16). The plane is uK, row-major, each row a bit stream
 * lowest bits first and starting on a byte (rad_encoding.h), so element (i, j) is bits
 * [j*K, (j+1)*K) of row i and its VALUE is the symbol. Every symbol sequence is a valid trellis, so
 * this is a lossless permutation of the tile streams (exl3_layout.h converts both ways), and any
 * cut on 16 rows / 16 columns keeps tiles whole.
 *
 * "Decode one expert's weight to f32" is exl3_decode_rows(view, 0, view->rows, out): out is
 * f32 [n][k] -- radiance's [rows, cols], i.e. out[n*k_total + k] = W[k][n] in exllamav3's terms.
 *
 * ============================================================ NUMERICS
 *
 * Codebook values are exact fp16 (one rounding, as exllamav3's __hfma / __hadd). The Hadamard is the
 * butterfly in f32 then one multiply by (float)(1/sqrt(128)); W is formed in exllamav3's order
 * (had over k, x suh, had over n, x svh) in f32. Bit-exact claims are made for W_hat only; W agrees
 * with an f32 transcription to rounding (the tests state the bound). Build with -ffp-contract=off.
 *
 * Thread safety: every function is reentrant (no static state; scratch is malloc'd per call).
 */
#ifndef EXL3_DECODE_H
#define EXL3_DECODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any change to a declaration below; consumers check it at compile time. */
#define EXL3_DECODE_API_VERSION 1

enum {
    EXL3_TILE     = 16,    /* a tile is 16 x 16 weights */
    EXL3_TILE_POS = 256,   /* positions a tile */
    EXL3_HAD      = 128,   /* rotation block; both feature counts are multiples of it */
    EXL3_BITS_MIN = 1,     /* K, bits a weight, that the DECODER reads (exllamav3 1.4.4: 1..8) */
    EXL3_BITS_MAX = 8
};

/* Codebooks, numbered as exllamav3 numbers them (cb template argument, codebook.cuh). */
enum { EXL3_CB_3INST = 0, EXL3_CB_MCG = 1, EXL3_CB_MUL1 = 2 };

/* The scalar a checkpoint stores beside the trellis to name its codebook: `<name>.mul1` or
 * `<name>.mcg`, I32 [] holding these bits; neither present means 3inst. */
#define EXL3_MUL1_MARKER 0x83DCD12Du
#define EXL3_MCG_MARKER  0xCBAC1FEDu

/* Return codes. Negative is failure; consumers map them to their own (RAD_E_*). */
enum { EXL3_OK = 0, EXL3_E_ARG = -1, EXL3_E_SHAPE = -2, EXL3_E_NOMEM = -3 };

/* "3inst", "mcg", "mul1"; null for anything else. Inline so header-only consumers
 * (include/exl3_encoding.h) need nothing linked. */
static inline const char* exl3_codebook_name(int codebook) {
    return codebook == EXL3_CB_3INST ? "3inst"
         : codebook == EXL3_CB_MCG   ? "mcg"
         : codebook == EXL3_CB_MUL1  ? "mul1" : 0;
}
/* The inverse; -1 for a name that is not one of the three. */
static inline int exl3_codebook_from_name(const char* name) {
    int cb;
    for (cb = EXL3_CB_3INST; cb <= EXL3_CB_MUL1 && name; ++cb) {
        const char* a = exl3_codebook_name(cb);
        const char* b = name;
        while (*a && *a == *b) { ++a; ++b; }
        if (!*a && !*b) return cb;
    }
    return -1;
}

/* ------------------------------------------------------------------ fp16 */
float    exl3_half_to_float(uint16_t h);
/* Round to nearest even, ONE rounding of the exact double; overflow to inf, nan to 0x7E00. */
uint16_t exl3_half_from_double(double x);

/* ------------------------------------------------------------------ codebook */
/* A 16-bit trellis state (higher bits ignored) to its fp16 value, as fp16 bits. */
uint16_t exl3_codebook_value(uint32_t state, int codebook);

/* ------------------------------------------------------------------ the tile */
/* Position p (0..255) -> tile row r (k axis, 0..15) and tile column c (n axis, 0..15). */
void exl3_pos_to_tile(int p, int* r, int* c);
/* Tile (r, c) -> position. */
int  exl3_tile_to_pos(int r, int c);

/* One tile's stream (16*K uint16, i.e. 8*K little-endian u32 words, MSB-first) -> its 256 symbols,
 * symbol p being stream bits [p*K, (p+1)*K) read MSB first. */
void exl3_tile_symbols(const uint16_t* tile, int bits, uint8_t sym[EXL3_TILE_POS]);
/* The inverse: 256 symbols (low K bits used) -> the tile stream. Any symbols are a valid trellis. */
void exl3_tile_pack(const uint8_t sym[EXL3_TILE_POS], int bits, uint16_t* tile);
/* The 16-bit state of every position: the window of the symbol stream ending at symbol p's last
 * bit, circular within the tile (tail-biting). */
void exl3_tile_states(const uint8_t sym[EXL3_TILE_POS], int bits, uint16_t state[EXL3_TILE_POS]);

/* A whole trellis tensor as exllamav3 stores it -- int16 [kt][nt][16*K], tile (tk, tn) at
 * (tk*nt + tn)*16*K -- to W_hat [16*kt][16*nt] as fp16 bits (exllamav3's [k, n] order: the golden's).
 * EXL3_E_ARG for bits or codebook out of range. */
int exl3_trellis_decode(const uint16_t* trellis, int64_t kt, int64_t nt, int bits, int codebook,
                        uint16_t* w_hat);

/* ------------------------------------------------------------------ the codes plane */
/* Element (i, j) of a uK plane, rows `row_bytes` apart, LSB-first along the row. */
uint32_t exl3_codes_get(const uint8_t* codes, int64_t row_bytes, int bits, int64_t i, int64_t j);
void     exl3_codes_set(uint8_t* codes, int64_t row_bytes, int bits, int64_t i, int64_t j,
                        uint32_t value);
/* Bytes of one dense row of `cols` K-bit elements: ceil(cols*K/8). */
int64_t  exl3_codes_row_bytes(int64_t cols, int bits);

/* ------------------------------------------------------------------ the rotation */
/* The normalised natural-order Sylvester Hadamard over each block of 128 of v[0..n), in place:
 * butterfly (a+b low, a-b high) in f32, then x (float)(1/sqrt(128)). EXL3_E_SHAPE unless n % 128 == 0. */
int exl3_had128(float* v, int64_t n);

/* ------------------------------------------------------------------ one weight from its planes */
/* One weight -- a projection, or a TP slice of one -- in radiance's [rows = n, cols = k] view, as
 * the canonical planes hold it. A STACKED gate_up (gate rows then up rows, PLAN.md D4) has the up
 * half's input scales in `suh1`: rows [0, rows/2) are read with `suh`, rows [rows/2, rows) with
 * `suh1`. A TP slice of a stacked weight takes the same span of each half, so the rule holds for
 * the slice. */
typedef struct Exl3View {
    int64_t         rows, cols;       /* n, k */
    int             bits;             /* K */
    int             codebook;         /* EXL3_CB_* */
    const uint8_t*  codes;            /* uK [rows][cols], row-major, LSB-first */
    int64_t         codes_row_bytes;  /* the stride; 0 = dense, exl3_codes_row_bytes(cols, bits) */
    const uint16_t* suh;              /* f16 [cols] */
    const uint16_t* suh1;             /* f16 [cols] for the second half of a stacked weight, or null */
    const uint16_t* svh;              /* f16 [rows] */
} Exl3View;

/* W_hat (the rotated basis, the values a kernel multiplies by) of rows [row0, row0+n) as f32
 * [n][cols], each an exact fp16 value. Needs only codes, bits, codebook, rows, cols; rows and cols
 * multiples of 16. EXL3_E_ARG / EXL3_E_SHAPE (a range outside the view) / EXL3_E_NOMEM. */
int exl3_decode_what_rows(const Exl3View* v, int64_t row0, int64_t n, float* out);

/* The weight itself, rows [row0, row0+n) as f32 [n][cols] = W[k][n] transposed (see the header).
 * rows and cols multiples of 128 (both rotations), and of 256 rows when suh1 is set. Any row range
 * inside the view; the enclosing 128-row blocks are decoded. */
int exl3_decode_rows(const Exl3View* v, int64_t row0, int64_t n, float* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_DECODE_H */
