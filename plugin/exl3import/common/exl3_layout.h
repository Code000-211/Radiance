/* exl3_layout.h -- canonical codes plane <-> tile-stream record, both directions. Pure C99.
 *
 * The CANONICAL form is what a .rad holds (exl3_decode.h, PLAN.md D4): a uK plane over radiance's
 * [rows = n, cols = k] view whose element (i, j) is the K-bit symbol of the trellis position that
 * decodes W_hat[k = j][n = i]. The TILE-STREAM RECORD is exllamav3's own trellis tensor for the same
 * rectangle: int16 [cols/16][rows/16][16*K], tile (tk, tn) at (tk*(rows/16) + tn)*16*K, each tile
 * the MSB-first bit stream of its 256 symbols. It is what turboderp's checkpoint ships (the
 * quantiser reads it) and what a trellis-in-register kernel streams (its relayout writes it).
 *
 * The two are a permutation of the same bits. Both directions work on any rectangle whose corner and
 * extents are multiples of 16: pass the plane pointer at the rectangle's first row (and its first
 * column's byte, j0*K/8, which is whole because j0 % 16 == 0) with the WHOLE plane's row stride, and
 * the record comes back as that rectangle's tiles alone, densely. A stacked gate_up is two records,
 * gate rows then up rows -- exllamav3's [2][k/16][n/16][16*K] when the two are written back to back.
 *
 * Suh and svh are f16 vectors in both forms and are not touched here.
 */
#ifndef EXL3_LAYOUT_H
#define EXL3_LAYOUT_H

#include <stdint.h>

#include "exl3_decode.h"   /* EXL3_OK / EXL3_E_* */

#ifdef __cplusplus
extern "C" {
#endif

#define EXL3_LAYOUT_API_VERSION 1

/* uint16 values in the record of a rows x cols rectangle: (rows/16)*(cols/16)*16*K. */
int64_t exl3_record_elems(int64_t rows, int64_t cols, int bits);

/* Canonical plane -> record. `codes` points at the rectangle's (0, 0) element's byte, rows
 * `row_bytes` apart. EXL3_E_ARG for bits outside 1..8, EXL3_E_SHAPE unless rows, cols % 16 == 0. */
int exl3_codes_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                         int bits, uint16_t* record);

/* Record -> canonical plane. Writes exactly the rectangle's K-bit elements; the plane's other bits
 * (outside the rectangle) are left as they are. */
int exl3_record_to_codes(const uint16_t* record, int64_t rows, int64_t cols, int bits,
                         uint8_t* codes, int64_t row_bytes);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_LAYOUT_H */
