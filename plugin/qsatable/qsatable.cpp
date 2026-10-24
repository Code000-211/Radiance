/* plugin/qsatable/qsatable.cpp -- the table op's schema, shapes, argument checks and export block.
 *
 * Shape hooks and geometry only; the union itself is in ref_qsatable.cpp (the definition) and
 * qsatable.hip (the kernel). Host C++ with no card: a card-less machine still checks the schema
 * and the shapes, and runs the reference row.
 */
#include "qsatable.h"

#include <climits>
#include <cstddef>

namespace {

#define PI(k) { k, RAD_P_INT, RAD_REQUIRED, 0 }
const RadParamSpec kTableParams[] = {
    PI("M"), PI("blocks"), PI("topk"), PI("tile"), PI("ratio"),
};
#undef PI

/* `selb`/`nsel`/`pos` are `qsa_select`'s own outputs, one row per query; the four outs describe the
 * table the tile shares. Every one is written on every launch, including the entries a row does not
 * hold (mask 0) and the tail of `tab` past `tlen` (0). */
const RadOperandSpec kTableOperands[] = {
    { "selb",    RAD_OPD_IN,  0 },
    { "nsel",    RAD_OPD_IN,  0 },
    { "pos",     RAD_OPD_IN,  0 },
    { "tab",     RAD_OPD_OUT, 0 },
    { "tlen",    RAD_OPD_OUT, 0 },
    { "mask",    RAD_OPD_OUT, 0 },
    { "row_off", RAD_OPD_OUT, 0 },
};

const RadOpSchema kSchemas[] = {
    { "qsa_table", kTableParams, 5, kTableOperands, 7,
      "The block table a query tile shares: the union of the rows' `selb` selections in block order "
      "(tab, tlen), and per row the membership mask that filters it back to that row's own list "
      "(mask, row_off). The partial page is not in the table; a row appends it. Union order is what "
      "makes a row's list a subsequence, so the row's attention sums the same pages in the same "
      "order." },
};

}  /* namespace */

int rad_qsatable_schema_count() { return 1; }
const RadOpSchema* rad_qsatable_schema_at(int i) {
    return i >= 0 && i < 1 ? &kSchemas[i] : nullptr;
}

int64_t rad_qsa_table_u_max(int64_t tile, int64_t topk, int64_t blocks) {
    /* One entry per (row, slot), and never more distinct blocks than exist. */
    int64_t u = tile * topk;
    if (u > blocks) u = blocks;
    return u < 1 ? 1 : u;
}

namespace {

/* The geometry both hooks and the argument readers agree on. `M`, `tile` and `ratio` are read as
 * their high bound in a shape hook: it sizes the buffers before any row count is known. */
int table_geometry(const RadParam* p, int n, bool range, RadQsaTableCall* c) {
    if (!p || !c) return RAD_E_INVAL;
    c->M      = range ? rad_param_getdim(p, n, "M", 0) : rad_param_geti(p, n, "M", 0);
    c->blocks = rad_param_geti(p, n, "blocks", 0);
    c->topk   = rad_param_geti(p, n, "topk", 0);
    c->tile   = range ? rad_param_getdim(p, n, "tile", 0) : rad_param_geti(p, n, "tile", 0);
    c->ratio  = rad_param_geti(p, n, "ratio", 0);
    if (c->M < 1 || c->M > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (c->topk < 1 || c->topk > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (c->tile < 1 || c->tile > c->M || c->tile > 32) return RAD_E_UNSUPPORTED;
    if (c->ratio < 1 || c->ratio > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (c->blocks < 1 || c->blocks > INT32_MAX) return RAD_E_UNSUPPORTED;
    c->tiles = (c->M + c->tile - 1) / c->tile;
    c->u_max = rad_qsa_table_u_max(c->tile, c->topk, c->blocks);
    return RAD_OK;
}

}  /* namespace */

int rad_qsa_table_shape(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (!out || i < 0 || i >= 7) return RAD_E_INVAL;
    RadQsaTableCall g{};
    const int rc = table_geometry(p, n, true, &g);
    if (rc != RAD_OK) return rc;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.dtype = RAD_I32;
    d.rank = 1;
    d.shape[0] = g.M;
    switch (i) {
    case 0: d.rank = 2; d.shape[0] = g.M; d.shape[1] = g.topk + 1; break;   /* selb */
    case 3: d.rank = 2; d.shape[0] = g.tiles; d.shape[1] = g.u_max; break;  /* tab */
    case 4: d.shape[0] = g.tiles; break;                                    /* tlen */
    case 5: d.rank = 2; d.shape[0] = g.M; d.shape[1] = g.u_max; break;      /* mask */
    default: break;                                                         /* nsel, pos, row_off */
    }
    *out = d;
    return RAD_OK;
}

/* The arena's `M` is the band's ceiling; this issue may carry fewer live rows, and the kernel takes
 * the row count from operand 0, so dim 0 narrows first and every other operand's extent is checked
 * against the result. Tile count follows from the live rows, so `tab`/`tlen` may be handed a shorter
 * plane than the declared one -- but only a whole number of tiles. */
int rad_qsa_table_args(const RadArgs* a, RadQsaTableCall* out) {
    if (!out || !a || !a->t || a->n_t != 7) return RAD_E_INVAL;
    RadQsaTableCall c{};
    int rc = table_geometry(a->p, a->n_p, false, &c);
    if (rc != RAD_OK) return rc;
    const int64_t live = a->t[0].rank ? a->t[0].shape[0] : 0;
    if (live < 1 || live > c.M) return RAD_E_SHAPE;
    c.M = live;
    c.tiles = (c.M + c.tile - 1) / c.tile;
    for (int i = 0; i < 7; ++i) {
        RadOpdDesc d{};
        if ((rc = rad_qsa_table_shape(a->p, a->n_p, i, &d)) != RAD_OK) return rc;
        d.shape[0] = i == 3 || i == 4 ? c.tiles : c.M;
        if (a->t[i].rank != d.rank || a->t[i].dtype != d.dtype) return RAD_E_SHAPE;
        if (!a->t[i].data) return RAD_E_INVAL;
        /* An input may cover more rows than are live and `selb` may carry a padded row pitch; an
         * output is written whole, so its rows are tight. */
        const bool out_opd = i >= 3;
        for (int j = 0; j < (int)d.rank; ++j) {
            if (out_opd ? a->t[i].shape[j] != d.shape[j] : a->t[i].shape[j] < d.shape[j])
                return RAD_E_SHAPE;
            /* Tight means each row starts where the next axis would put it. */
            const int64_t step = j + 1 < (int)d.rank ? d.shape[j + 1] : 1;
            if (out_opd ? a->t[i].stride[j] != step : a->t[i].stride[j] < step) return RAD_E_STRIDE;
        }
    }
    c.selb = (const int32_t*)a->t[0].data;
    c.nsel = (const int32_t*)a->t[1].data;
    c.pos  = (const int32_t*)a->t[2].data;
    c.tab  = (int32_t*)a->t[3].data;
    c.tlen = (int32_t*)a->t[4].data;
    c.mask = (int32_t*)a->t[5].data;
    c.row_off = (int32_t*)a->t[6].data;
    /* The outputs are written whole; the inputs are only read. An alias would let a write land on a
     * read and the union would depend on its own fill order. */
    for (int i = 0; i < 3; ++i)
        for (int j = 3; j < 7; ++j)
            if (a->t[i].data == a->t[j].data) return RAD_E_INVAL;
    c.ld_selb = a->t[0].stride[0];
    *out = c;
    return RAD_OK;
}

RadKernelInfo rad_qsatable_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "qsatable_qsa_table";
    r.op = "qsa_table";
    r.family = "attn";
    r.computes = "the union of a query tile's block selections in block order, with a per-row "
                 "membership mask";
    r.shape = "M,tile<=32,topk,blocks";
    r.dtypes = "i32";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.launch = launch;
    r.opd_shape = rad_qsa_table_shape;
    return r;
}
