/* plugin/qsatable/ref_qsatable.cpp -- the host row: the definition of `qsa_table`.
 *
 * The kernel is checked against this, so it is written the plainest way that is right: walk the
 * blocks once per tile in ascending order, ask each row whether it holds the block, and count. No
 * prefix scans, no binary searches, no assumption about the rows' lists beyond their being ascending
 * (which is the contract `qsa_select` emits under), so it can be read as the statement of the op.
 *
 * Cost is O(tile * blocks * topk) per tile, which is the price of being obvious; the device kernel
 * pays a prefix scan instead of a re-walk. Nothing but rad-kbench and the emulator's tests call this.
 */
#include "qsatable.h"

#include <algorithm>
#include <vector>

namespace {

/* Is `b` one of row `r`'s selected blocks? The list is ascending and its only entry not below
 * `own` is the partial page, which no table holds. */
bool holds(const RadQsaTableCall& c, int64_t r, int64_t own, int64_t b) {
    if (b >= own) return false;
    const int32_t* row = c.selb + r * c.ld_selb;
    const int32_t* end = row + c.nsel[r];
    const int32_t* at = std::lower_bound(row, end, (int32_t)b);
    return at != end && *at == (int32_t)b;
}

}  /* namespace */

int rad_ref_qsa_table(const RadArgs* a) {
    RadQsaTableCall c{};
    const int rc = rad_qsa_table_args(a, &c);
    if (rc != RAD_OK) return rc;

    std::vector<int32_t> own((size_t)c.M);
    for (int64_t m = 0; m < c.M; ++m) {
        const int64_t n = c.nsel[m];
        own[(size_t)m] = n < 0 ? 0 : (c.pos[m] + 1) / c.ratio;
        if (n > c.topk + 1) return RAD_E_SHAPE;
        /* Ascending, block order, the partial page last: what makes the mask a filter. */
        for (int64_t i = 1; i < n; ++i)
            if (c.selb[m * c.ld_selb + i] <= c.selb[m * c.ld_selb + i - 1]) return RAD_E_INVAL;
    }

    for (int64_t t = 0; t < c.tiles; ++t) {
        const int64_t r0 = t * c.tile, r1 = r0 + c.tile < c.M ? r0 + c.tile : c.M;
        int32_t* tab = c.tab + t * c.u_max;
        int64_t u = 0;
        for (int64_t b = 0; b < c.blocks; ++b) {
            bool member = false;
            for (int64_t r = r0; r < r1 && !member; ++r) member = holds(c, r, own[(size_t)r], b);
            if (!member) continue;
            if (u >= c.u_max) return RAD_E_UNSUPPORTED;   /* the bound the shape hook declares */
            tab[u++] = (int32_t)b;
        }
        for (int64_t j = u; j < c.u_max; ++j) tab[j] = 0;
        c.tlen[t] = (int32_t)u;
        for (int64_t r = r0; r < r1; ++r) {
            c.row_off[r] = (int32_t)(t * c.u_max);
            int32_t* mask = c.mask + r * c.u_max;
            for (int64_t j = 0; j < c.u_max; ++j)
                mask[j] = j < u && holds(c, r, own[(size_t)r], tab[j]) ? 1 : 0;
        }
    }
    return RAD_OK;
}
