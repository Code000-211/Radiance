/* exl3_layout.c -- canonical codes plane <-> tile-stream record. See exl3_layout.h.
 *
 * Tile (tk, tn) of the rectangle holds W_hat[k = tk*16 + r][n = tn*16 + c] at position
 * p = exl3_tile_to_pos(r, c), whose symbol the canonical plane keeps at element (n, k). Both
 * directions walk the same map, so each is the other's inverse by construction; the property test
 * (tests/exl3_layout_test.c) and the numpy twin (exl3_ref.py) check it rather than trust it.
 */
#include "exl3_layout.h"

static int shape_ok(int64_t rows, int64_t cols) {
    return rows >= 0 && cols >= 0 && rows % EXL3_TILE == 0 && cols % EXL3_TILE == 0;
}

int64_t exl3_record_elems(int64_t rows, int64_t cols, int bits) {
    return (rows / EXL3_TILE) * (cols / EXL3_TILE) * 16 * bits;
}

int exl3_codes_to_record(const uint8_t* codes, int64_t row_bytes, int64_t rows, int64_t cols,
                         int bits, uint16_t* record) {
    uint8_t sym[EXL3_TILE_POS];
    int tr[EXL3_TILE_POS], tc[EXL3_TILE_POS];
    const int64_t nt = rows / EXL3_TILE, kt = cols / EXL3_TILE;
    int64_t tk, tn;
    int p;
    if (!codes || !record || bits < EXL3_BITS_MIN || bits > EXL3_BITS_MAX) return EXL3_E_ARG;
    if (!shape_ok(rows, cols) || row_bytes < exl3_codes_row_bytes(cols, bits)) return EXL3_E_SHAPE;
    for (p = 0; p < EXL3_TILE_POS; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (tk = 0; tk < kt; ++tk)
        for (tn = 0; tn < nt; ++tn) {
            for (p = 0; p < EXL3_TILE_POS; ++p)
                sym[p] = (uint8_t)exl3_codes_get(codes, row_bytes, bits, tn * EXL3_TILE + tc[p],
                                                 tk * EXL3_TILE + tr[p]);
            exl3_tile_pack(sym, bits, record + (tk * nt + tn) * 16 * bits);
        }
    return EXL3_OK;
}

int exl3_record_to_codes(const uint16_t* record, int64_t rows, int64_t cols, int bits,
                         uint8_t* codes, int64_t row_bytes) {
    uint8_t sym[EXL3_TILE_POS];
    int tr[EXL3_TILE_POS], tc[EXL3_TILE_POS];
    const int64_t nt = rows / EXL3_TILE, kt = cols / EXL3_TILE;
    int64_t tk, tn;
    int p;
    if (!codes || !record || bits < EXL3_BITS_MIN || bits > EXL3_BITS_MAX) return EXL3_E_ARG;
    if (!shape_ok(rows, cols) || row_bytes < exl3_codes_row_bytes(cols, bits)) return EXL3_E_SHAPE;
    for (p = 0; p < EXL3_TILE_POS; ++p) exl3_pos_to_tile(p, &tr[p], &tc[p]);
    for (tk = 0; tk < kt; ++tk)
        for (tn = 0; tn < nt; ++tn) {
            exl3_tile_symbols(record + (tk * nt + tn) * 16 * bits, bits, sym);
            for (p = 0; p < EXL3_TILE_POS; ++p)
                exl3_codes_set(codes, row_bytes, bits, tn * EXL3_TILE + tc[p],
                               tk * EXL3_TILE + tr[p], sym[p]);
        }
    return EXL3_OK;
}
