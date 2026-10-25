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
 * order"). The kernel runs the row path's two passes over that key order -- the max, then the sum --
 * in the row path's own expressions, so two tables that give a row the same keys in the same order
 * give it the same bytes. `tests/r4d_emu/test_qsa.cpp` checks that
 * in two steps: `masked read:` holds the shared table to a table per row (tile 1) **of this op**
 * byte for byte, and `row path:` holds it to the attention the architecture issues today
 * (`libref`'s `attn_paged`, loaded from the same build) key for key, entry for entry. The second
 * one is the measurement's claim; the first is only the table's self-consistency.
 *
 * ONE LAUNCH IS THE STEP'S ROWS. `cu` is the batch's `cu_seqlens`, so the row numbering it answers
 * to is the batch's; `arch/common/rad_block_attn_gated_fp8.h` therefore issues one launch over the
 * step's `T` rows rather than a chunk apiece. A caller that chunks anyway slices q, positions and
 * the table's row planes but not `cu`, and must pass its chunk's first row in the batch's numbering
 * as `row0` -- otherwise the second chunk of a two-sequence batch looks its rows up in the first
 * sequence's interval and reads that sequence's pages.
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
    /* [seqs + 1] the batch's `cu_seqlens`, or null: row m belongs to the sequence holding it, so a
     * row set of several sequences needs no per-row id operand (the batch has none). */
    const int32_t* cu;
    void*          out;     /* [M][n_head * head_dim] */

    int64_t M, n_head, kv_heads, gqa, head_dim, block_size, ratio;
    int64_t tile, u_max, tiles, topk, blocks, seqs;
    int64_t row0;                   /* the chunk's first row in the batch's `cu` numbering; a chunk
                                     * also rebases `row_off`, which counts offsets into `tab` */
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

/* The row's sequence: the last `cu` entry at or before the row. A null `cu` is the caller's "every
 * row is this sequence" form -- it slices `bt` to that one. */
/* `inline` in both halves: the kernel reads a row's sequence through this too, so it carries the
 * device qualifier where there is one -- and this header also compiles as plain C++ (the emulator
 * and the host reference), where those keywords do not exist. */
#if defined(__HIPCC__)
#define RAD_QSA_HD __host__ __device__ inline
#else
#define RAD_QSA_HD inline
#endif
RAD_QSA_HD int64_t rad_qsa_attn_seq(const RadQsaAttnCall& c, int64_t row) {
    if (!c.cu) return 0;
    const int64_t r = row + c.row0;   /* the batch's row, not the chunk's */
    int64_t lo = 0, hi = c.seqs;   /* cu[lo] <= r, and r < cu[hi] when hi < seqs */
    while (lo + 1 < hi) {
        const int64_t mid = (lo + hi) / 2;
        if (c.cu[mid] <= r) lo = mid; else hi = mid;
    }
    return lo;
}

/* One row's key order, in the shape every implementation walks: the union's entries the row's mask
 * holds, ascending, then the partial page's visible tokens. `visit(block, page, first, last)` is
 * called once per entry, `first..last` the tokens of that page the row sees; a page that maps to no
 * cache block is skipped, as `attn_paged` skips a -1 entry. Shared by the oracle, the kernel and the
 * test's own port so they cannot drift. */
template <class Visit>
void rad_qsa_attn_keys(const RadQsaAttnCall& c, int64_t row, Visit visit) {
    const int64_t s = rad_qsa_attn_seq(c, row);
    const int64_t p = c.pos[row];
    const int64_t own = (p + 1) / c.ratio;
    /* `row_off` is an offset into the `tab` plane the caller handed over: a caller that slices the
     * plane rebases it (`qsa_attn`'s `row0`). */
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
    /* The partial page, by position: only off a block boundary, and only if the sequence's block
     * table has a row entry for it at all (mid-chunk it does; at the end of the history it does
     * not). The bound is the table's own width -- the step's `bt_pitch`, which the core stages per
     * step and which every block a row can name fits in -- and not the block space's `blocks`. */
    if ((p + 1) % c.ratio != 0 && own < c.bt_pitch) {
        const int64_t page = c.bt[s * c.bt_pitch + own];
        if (page >= 0) visit(own, page, 0, p % c.ratio);
    }
}

#endif /* RAD_PLUGIN_QSAATTN_H */
