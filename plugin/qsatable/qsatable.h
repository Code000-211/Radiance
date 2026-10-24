/* plugin/qsatable/qsatable.h -- the one table op's contract between its host half and its kernel.
 *
 * Plugin-private ON PURPOSE: core/ must not include this. What a caller outside the plugin needs is
 * the schema (name, params, operands), which the plugin exports through rad_kernel_schema_at.
 *
 * THE OP. `qsa_table` builds the block table a query TILE shares: the union of the rows' own
 * selections, in block order, plus, per row, the membership mask that turns the union back into
 * exactly the row's own list. `qsa_select` hands out page ids; a page id does not sort by block (a
 * dead block drops out of the block table and the prefix cache re-orders the rest), which is why the
 * selection also carries `selb`, the same entries as block indices, and why the table is built from
 * `selb` and only then mapped to page ids.
 *
 * THE PARTIAL PAGE IS NOT IN THE TABLE. A row's last page is the partial one, chosen by the row's
 * position rather than by its scores, and its block index is the row's own complete-block count. Two
 * rows of a tile rarely agree on it, so it stays out of the union: an entry whose block index is not
 * below `(pos + 1) / ratio` belongs to no row's table. It is the only entry a row appends after the
 * table, and it is why a row's selection is the table filtered by the mask, in order, plus at most
 * one trailing page.
 *
 * WHY ONE TABLE AND A MASK IS THE SAME SUM. Attention sums the softmax over the pages in table
 * order and floating addition is not associative, so a shared table only reproduces a row's result
 * bit for bit when the row's own list is a subsequence of it. Block order makes that true by
 * construction: `qsa_select` emits the selection as one run in block order (docs/QSA.md, "Table
 * order"), so filtering the union by the row's mask gives back that row's list, in that order.
 */
#ifndef RAD_PLUGIN_QSATABLE_H
#define RAD_PLUGIN_QSATABLE_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>

/* The op's packed call, once for the device launch and once for the host reference. */
struct RadQsaTableCall {
    const int32_t* selb;    /* [M][topk + 1] the selection as BLOCK indices, block order */
    const int32_t* nsel;    /* [M] entries written, the partial page included */
    const int32_t* pos;     /* [M] the row's last token position */
    int32_t*       tab;     /* [tiles][u_max] the union in block order, filled to tlen[t] */
    int32_t*       tlen;    /* [tiles] the union's length */
    int32_t*       mask;    /* [M][u_max] 1 where the row's selection holds that entry */
    int32_t*       row_off; /* [M] the row's table offset into the flat tab plane, (m / tile) * u_max */
    int64_t        M, topk, tile, ratio, blocks, u_max, tiles;
    int64_t        ld_selb;
};

/* The op set: one schema, one device row and one host-reference row. */
int rad_qsatable_schema_count();
const RadOpSchema* rad_qsatable_schema_at(int i);
RadKernelInfo rad_qsatable_row(RadLaunchFn launch);

int rad_qsa_table_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_qsa_table_args(const RadArgs*, RadQsaTableCall*);

/* The launches: the device kernel (qsatable.hip) and the host oracle (ref_qsatable.cpp). The oracle
 * is the definition of the op; the kernel is checked against it. */
int rad_qsa_table_launch(const RadArgs*, RadStream);
int rad_ref_qsa_table(const RadArgs*);

/* The union width a shape hook and the kernels agree on: at most `blocks`, at most one entry per
 * (row, selection slot). */
int64_t rad_qsa_table_u_max(int64_t tile, int64_t topk, int64_t blocks);

#endif
