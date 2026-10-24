/* plugin/qsatable/qsaattn.h -- the masked read of the shared table: `qsa_attn`'s contract between its
 * host half and its kernel.
 *
 * Plugin-private on purpose, like qsatable.h: what a caller outside needs is the schema.
 *
 * THE OP. `qsa_attn` is paged attention over a shared table. Where `attn_paged` takes one list of
 * pages per row and a length, this one takes the tile's union (`tab`, `tlen`) and each row's
 * membership mask (`mask`, `row_off`), and walks the union in order, skipping the entries the row's
 * mask does not hold. The union is in block order and `tab` carries BLOCK indices, not page ids, so
 * the row's sequence's block table turns each entry into the page it reads -- a page id does not
 * follow its block (a dead block drops out of the block table and the prefix cache re-orders the
 * rest), and a table shared by rows of one tile cannot be re-sorted after the fact (qsatable.h).
 *
 * THE PARTIAL PAGE IS APPENDED, NOT READ OFF THE TABLE. It is decided by the row's position, not by
 * its scores: its block index is the row's own complete-block count `(pos + 1) / ratio`, and only
 * when `(pos + 1) % ratio != 0`. The op appends it after the mask-filtered union, with its tokens up
 * to `pos`, exactly where the row path puts it. That keeps a page the chunk itself is still filling
 * from reaching a row that may not see it.
 *
 * WHY THE SUM IS THE ROW PATH'S SUM. Softmax is summed in table order and float addition is not
 * associative, so the masked read reproduces a row's result only when the row's list is a
 * subsequence of the union -- which block order makes true by construction (docs/QSA.md, "Table
 * order"). The kernel accumulates in that order, one key at a time; `tests/r4d_emu/test_qsa.cpp`
 * (`masked read:`) runs the same rows through a per-row table (tile 1) and through the shared one and
 * requires the outputs to be equal **byte for byte**.
 *
 * PROTOTYPE. The device row is one workgroup of one lane per (row, kv head), so it is a statement of
 * the op, not a serving kernel: a real one keeps a row's union in LDS and processes the tile's rows
 * in one wave. It exists so the masked read has a definition, an oracle and a byte comparison.
 */
#ifndef RAD_PLUGIN_QSAATTN_H
#define RAD_PLUGIN_QSAATTN_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>

struct RadQsaAttnCall {
    const void*    q;       /* [M][n_head * head_dim], an activation dtype */
    const void*    kv;      /* [blocks][kv_heads][block_size][2 * head_dim] */
    const int32_t* tab;     /* [tiles][u_max] the union in block order, filled to tlen[t] */
    const int32_t* tlen;    /* [tiles] the union's length */
    const int32_t* mask;    /* [M][u_max] 1 where the row's selection holds that entry */
    const int32_t* row_off; /* [M] the row's table offset into the flat tab plane */
    const int32_t* pos;     /* [M] the row's last token position */
    const int32_t* bt;      /* [n_seq][bt_pitch] the sequence's block table, -1 for a dead block */
    const int32_t* seq;     /* [M] the row's sequence index */
    void*          out;     /* [M][n_head * head_dim] */

    int64_t M, n_head, kv_heads, gqa, head_dim, block_size, ratio;
    int64_t tile, u_max, tiles, topk, blocks;
    int64_t q_pitch, out_pitch, bt_pitch;              /* elements between rows */
    int64_t kv_sb, kv_sh, kv_so, kv_sd;                /* cache strides: block, head, slot, dim */
    float   scale;
    uint32_t q_dtype, kv_dtype, out_dtype;
};

int rad_qsaattn_schema_count();
const RadOpSchema* rad_qsaattn_schema_at(int i);
RadKernelInfo rad_qsaattn_row(RadLaunchFn launch);

int rad_qsa_attn_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_qsa_attn_args(const RadArgs*, RadQsaAttnCall*);

/* The device kernel (qsaattn.hip) and the host oracle (ref_qsaattn.cpp). The oracle is the
 * definition of the op; the kernel is checked against it. */
int rad_qsa_attn_launch(const RadArgs*, RadStream);
int rad_ref_qsa_attn(const RadArgs*);

/* One row's key order, in the shape every implementation walks: the union's entries the row's mask
 * holds, ascending, then the partial page's visible tokens. `visit(block, page, first, last)` is
 * called once per entry, `first..last` the tokens of that page the row sees; a page that maps to no
 * cache block is skipped, as `attn_paged` skips a -1 entry. Shared by the oracle and by the test's
 * own port so the three cannot drift. */
template <class Visit>
void rad_qsa_attn_keys(const RadQsaAttnCall& c, int64_t row, Visit visit) {
    const int64_t s = c.seq[row];
    const int64_t p = c.pos[row];
    const int64_t own = (p + 1) / c.ratio;
    const int64_t tile = c.row_off[row] / c.u_max;
    const int32_t* tab = c.tab + c.row_off[row];
    const int32_t* mask = c.mask + (int64_t)row * c.u_max;
    const int64_t tlen = c.tlen[tile];
    for (int64_t j = 0; j < tlen; ++j) {
        if (!mask[j]) continue;
        const int64_t b = tab[j];
        if (b < 0 || b >= own) continue;
        const int64_t page = c.bt[s * c.bt_pitch + b];
        if (page < 0) continue;
        visit(b, page, 0, c.ratio - 1);
    }
    /* The partial page, by position: only off a block boundary, and only if the sequence has the
     * block at all (mid-chunk it does; at the end of the history it does not). */
    if ((p + 1) % c.ratio != 0 && own < c.blocks) {
        const int64_t page = c.bt[s * c.bt_pitch + own];
        if (page >= 0) visit(own, page, 0, p % c.ratio);
    }
}

#endif /* RAD_PLUGIN_QSAATTN_H */
