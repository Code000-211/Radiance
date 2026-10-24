/* plugin/qsatable/qsaattn.cpp -- `qsa_attn`'s schema, shapes and argument checks.
 *
 * Shape hooks and geometry only; the read itself is in ref_qsaattn.cpp (the definition) and
 * qsaattn.hip (the prototype kernel).
 */
#include "qsaattn.h"
#include "qsatable.h"

#include <climits>
#include <cmath>
#include <cstddef>

namespace {

#define PI(k) { k, RAD_P_INT, RAD_REQUIRED, 0 }
const RadParamSpec kAttnParams[] = {
    { "M",          RAD_P_INT,   RAD_REQUIRED, 0 },
    { "n_head",     RAD_P_INT,   RAD_REQUIRED, 0 },
    { "head_dim",   RAD_P_INT,   RAD_REQUIRED, 0 },
    { "gqa",        RAD_P_INT,   RAD_REQUIRED, 0 },
    { "block_size", RAD_P_INT,   RAD_REQUIRED, 0 },
    { "ratio",      RAD_P_INT,   RAD_REQUIRED, 0 },
    { "tile",       RAD_P_INT,   RAD_REQUIRED, 0 },
    { "topk",       RAD_P_INT,   RAD_REQUIRED, 0 },
    { "blocks",     RAD_P_INT,   RAD_REQUIRED, 0 },
    { "max_seqs",   RAD_P_INT,   RAD_REQUIRED, 0 },
    { "u_max",      RAD_P_INT,   RAD_REQUIRED, 0 },
    { "scale",      RAD_P_F64,   0, 0 },
    { "q_dtype",    RAD_P_STR,   RAD_REQUIRED, 0 },
    { "kv_dtype",   RAD_P_STR,   RAD_REQUIRED, 0 },
};
#undef PI

/* q, the cache, the tile's table and the row's own descriptors, then the output. The operands the
 * table op produced are named as it names them, so a caller can hand them straight over. */
const RadOperandSpec kAttnOperands[] = {
    { "q",       RAD_OPD_IN,  0 },
    { "kv",      RAD_OPD_IN,  0 },
    { "tab",     RAD_OPD_IN,  0 },
    { "tlen",    RAD_OPD_IN,  0 },
    { "mask",    RAD_OPD_IN,  0 },
    { "row_off", RAD_OPD_IN,  0 },
    { "pos",     RAD_OPD_IN,  0 },
    { "bt",      RAD_OPD_IN,  0 },
    { "cu",      RAD_OPD_IN,  0 },
    { "out",     RAD_OPD_OUT, 0 },
};

const RadOpSchema kSchemas[] = {
    { "qsa_attn", kAttnParams, 14, kAttnOperands, 10,
      "Paged attention over a table a query tile shares: the union's entries the row's membership "
      "mask holds, in block order, then the row's partial page. Same keys in the same order as the "
      "row path, so the same sum." },
};

}  /* namespace */

int rad_qsaattn_schema_count() { return 1; }
const RadOpSchema* rad_qsaattn_schema_at(int i) {
    return i >= 0 && i < 1 ? &kSchemas[i] : nullptr;
}

namespace {

int attn_geometry(const RadParam* p, int n, RadQsaAttnCall* c) {
    if (!p || !c) return RAD_E_INVAL;
    c->M          = rad_param_getdim(p, n, "M", 0);
    c->n_head     = rad_param_geti(p, n, "n_head", 0);
    c->head_dim   = rad_param_geti(p, n, "head_dim", 0);
    c->gqa        = rad_param_geti(p, n, "gqa", 0);
    c->block_size = rad_param_geti(p, n, "block_size", 0);
    c->ratio      = rad_param_geti(p, n, "ratio", 0);
    c->topk       = rad_param_geti(p, n, "topk", 0);
    c->blocks     = rad_param_geti(p, n, "blocks", 0);
    const int64_t tile = rad_param_getdim(p, n, "tile", 0);
    if (c->M < 1 || c->M > INT32_MAX) return RAD_E_UNSUPPORTED;
    if (c->n_head < 1 || c->head_dim < 1 || c->gqa < 1) return RAD_E_UNSUPPORTED;
    if (c->n_head % c->gqa) return RAD_E_SHAPE;
    if (c->block_size < 1 || c->ratio < 1 || c->topk < 1 || c->blocks < 1) return RAD_E_UNSUPPORTED;
    /* `tab` holds block-table entries and the table is paged at the compress ratio, so a cache page
     * has to be exactly one block: a wider page would make the partial page's token range wrong. */
    if (c->block_size != c->ratio) return RAD_E_UNSUPPORTED;
    if (tile < 1 || tile > c->M || tile > 32) return RAD_E_UNSUPPORTED;
    c->kv_heads = c->n_head / c->gqa;
    c->tile     = tile;
    c->tiles    = (c->M + tile - 1) / tile;
    /* The table's declared width, read from the same `u_max` parameter `qsa_table` was issued
     * with: the reader cannot re-derive it, since a tile of fewer rows than the plane was sized
     * for still fills that width. */
    c->u_max    = rad_param_geti(p, n, "u_max", 0);
    if (c->u_max < 1 || c->u_max > c->tile * c->topk) return RAD_E_UNSUPPORTED;
    c->scale    = (float)rad_param_getf(p, n, "scale", 1.0 / std::sqrt((double)c->head_dim));
    return RAD_OK;
}

}  /* namespace */

/* Operand 0 is [M, n_head * head_dim]; the cache is [blocks, kv_heads, block_size, 2 * head_dim];
 * the four table operands are the ones `qsa_table` declares; `bt` is the sequence's block table at
 * its declared ceiling; `cu` is the batch's `cu_seqlens`. */
int rad_qsa_attn_shape(const RadParam* p, int n, int i, RadOpdDesc* out) {
    if (!out || i < 0 || i >= 10) return RAD_E_INVAL;
    RadQsaAttnCall g{};
    const int rc = attn_geometry(p, n, &g);
    if (rc != RAD_OK) return rc;
    RadOpdDesc d{};
    d.idx_const = -1;
    d.dtype = RAD_I32;
    d.rank = 1;
    d.shape[0] = g.M;
    switch (i) {
    case 0: case 9:
        d.dtype = rad_param_getdt(p, n, "q_dtype");
        d.rank = 2; d.shape[0] = g.M; d.shape[1] = g.n_head * g.head_dim; break;
    case 1:
        d.dtype = rad_param_getdt(p, n, "kv_dtype");
        d.rank = 4; d.shape[0] = g.blocks; d.shape[1] = g.kv_heads;
        d.shape[2] = g.block_size; d.shape[3] = 2 * g.head_dim; break;
    case 2: d.rank = 2; d.shape[0] = g.tiles; d.shape[1] = g.u_max; break;
    case 3: d.shape[0] = g.tiles; break;
    case 4: d.rank = 2; d.shape[0] = g.M; d.shape[1] = g.u_max; break;
    case 7: d.rank = 2; d.shape[0] = rad_param_geti(p, n, "max_seqs", 1); d.shape[1] = g.blocks; break;
    case 8: d.shape[0] = rad_param_geti(p, n, "max_seqs", 1) + 1; break;   /* cu_seqlens */
    default: break;
    }
    if (d.dtype == RAD_DT_INVALID) return RAD_E_UNSUPPORTED;
    *out = d;
    return RAD_OK;
}

/* `q` and three of the table's operands may cover more rows than are live, so their dim 0 narrows
 * first; everything else is checked against that count. `out` and `mask` are written whole and so
 * must be tight. The cache is checked for the extent its strides walk, not for a rank-4 shape it may
 * legitimately arrive without. */
int rad_qsa_attn_args(const RadArgs* a, RadQsaAttnCall* out) {
    if (!out || !a || !a->t || a->n_t != 10) return RAD_E_INVAL;
    RadQsaAttnCall c{};
    int rc = attn_geometry(a->p, a->n_p, &c);
    if (rc != RAD_OK) return rc;
    if (a->t[0].rank != 2 || a->t[0].shape[1] != c.n_head * c.head_dim) return RAD_E_SHAPE;
    const int64_t live = a->t[0].shape[0];
    if (live < 1 || live > c.M) return RAD_E_SHAPE;
    c.M = live;
    c.tiles = (c.M + c.tile - 1) / c.tile;
    for (int i = 0; i < 10; ++i)
        if (!a->t[i].data) return RAD_E_INVAL;
    if (a->t[0].dtype != rad_param_getdt(a->p, a->n_p, "q_dtype")) return RAD_E_SHAPE;
    if (a->t[1].dtype != rad_param_getdt(a->p, a->n_p, "kv_dtype")) return RAD_E_SHAPE;
    if (a->t[9].dtype != a->t[0].dtype) return RAD_E_SHAPE;
    if (a->t[2].rank != 2 || a->t[2].dtype != RAD_I32) return RAD_E_SHAPE;
    if (a->t[2].shape[0] < c.tiles || a->t[2].shape[1] < c.u_max) return RAD_E_SHAPE;
    if (a->t[3].rank != 1 || a->t[3].dtype != RAD_I32 || a->t[3].shape[0] < c.tiles) return RAD_E_SHAPE;
    if (a->t[4].rank != 2 || a->t[4].dtype != RAD_I32) return RAD_E_SHAPE;
    if (a->t[4].shape[0] < c.M || a->t[4].shape[1] != c.u_max) return RAD_E_SHAPE;
    if (a->t[4].stride[0] != c.u_max) return RAD_E_STRIDE;
    for (int i = 5; i <= 6; ++i)
        if (a->t[i].rank != 1 || a->t[i].dtype != RAD_I32 || a->t[i].shape[0] < c.M)
            return RAD_E_SHAPE;
    if (a->t[7].rank != 2 || a->t[7].dtype != RAD_I32) return RAD_E_SHAPE;
    if (a->t[7].shape[0] < rad_args_geti_or(a, "max_seqs", 1)) return RAD_E_SHAPE;
    /* `cu_seqlens`, one entry per sequence and one past the last, so its width is the sequence
     * count plus one and never the row count. */
    if (a->t[8].rank != 1 || a->t[8].dtype != RAD_I32) return RAD_E_SHAPE;
    if (a->t[8].shape[0] < 2 || a->t[8].shape[0] > rad_args_geti_or(a, "max_seqs", 1) + 1)
        return RAD_E_SHAPE;
    c.seqs = a->t[8].shape[0] - 1;
    if (a->t[9].rank != 2 || a->t[9].shape[0] != c.M ||
        a->t[9].shape[1] != c.n_head * c.head_dim) return RAD_E_SHAPE;
    if (a->t[9].stride[0] < c.n_head * c.head_dim) return RAD_E_STRIDE;
    if (a->t[1].rank != 4) return RAD_E_SHAPE;
    if (a->t[1].shape[0] < c.blocks || a->t[1].shape[1] < c.kv_heads ||
        a->t[1].shape[2] != c.block_size || a->t[1].shape[3] != 2 * c.head_dim) return RAD_E_SHAPE;

    c.q       = a->t[0].data;
    c.kv      = a->t[1].data;
    c.tab     = (const int32_t*)a->t[2].data;
    c.tlen    = (const int32_t*)a->t[3].data;
    c.mask    = (const int32_t*)a->t[4].data;
    c.row_off = (const int32_t*)a->t[5].data;
    c.pos     = (const int32_t*)a->t[6].data;
    c.bt      = (const int32_t*)a->t[7].data;
    c.cu      = (const int32_t*)a->t[8].data;
    c.out     = a->t[9].data;
    c.q_pitch   = a->t[0].stride[0];
    c.out_pitch = a->t[9].stride[0];
    c.bt_pitch  = a->t[7].stride[0];
    c.kv_sb = a->t[1].stride[0]; c.kv_sh = a->t[1].stride[1];
    c.kv_so = a->t[1].stride[2]; c.kv_sd = a->t[1].stride[3];
    c.q_dtype   = a->t[0].dtype;
    c.kv_dtype  = a->t[1].dtype;
    c.out_dtype = a->t[9].dtype;
    *out = c;
    return RAD_OK;
}

RadKernelInfo rad_qsaattn_row(RadLaunchFn launch) {
    RadKernelInfo r{};
    r.name = "qsatable_qsa_attn";
    r.op = "qsa_attn";
    r.family = "attn";
    r.computes = "paged attention over a shared table: the row's mask-filtered union, then its "
                 "partial page";
    r.shape = "M,tile,topk,blocks";
    r.dtypes = "q activation, kv bf16 or fp8_e4m3";
    r.domain = RAD_DOMAIN_DEVICE;
    r.priority = 100;
    r.launch = launch;
    r.opd_shape = rad_qsa_attn_shape;
    return r;
}
