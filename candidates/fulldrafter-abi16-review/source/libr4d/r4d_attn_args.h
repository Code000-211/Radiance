/* r4d_attn_args.h -- the paged-attention operand list, read once for four kernel geometries.
 *
 * The paged kernels -- h256/gqa6, h128/gqa4 and the gqa8/gqa12 instantiations, prefill and decode
 * -- take the SAME operands in the same order, because they implement the same op. The reader for
 * that list therefore belongs to the family rather than to any one of them, and this is where it
 * lives: a header beside the kernels, included by each unit's entry point.
 *
 * It is NOT a translation layer. libr4d's R4DArgs carries the same convention RadTensor does --
 * strides in ELEMENTS (spec 2.1) -- so this is field extraction plus the validation a constraint
 * cannot do: a constraint settles whether a kernel EXISTS for a geometry and says nothing about a
 * particular call, so contiguity, alignment and buffer sizes are checked here and refused with the
 * code that names the class.
 *
 * The kernels already return a negative code for a shape they do not serve (-1 head_dim or
 * block_size, -2 gqa, -4 too many decode rows, -100-hipError for a launch failure) and those are
 * passed through unchanged: a caller that gets one has hit a selection bug, and the exact value is
 * what says which.
 */
#ifndef R4D_ATTN_ARGS_H
#define R4D_ATTN_ARGS_H

#include "r4d.h"
#include "r4d_args.h"

#include <cmath>

/* libr4d's compiled-in geometry, checked against the row table at build time rather than trusted.
 * A libr4d rebuilt for a different head size would otherwise change what the rows mean without
 * changing the rows. */
static inline int check_dims_once() {
    int hd = 0, gqa = 0, bs = 0, rows = 0;
    r4d_attn_dims(&hd, &gqa, &bs, &rows);
    return (hd == 256 && gqa == 6 && bs == 16 && rows == 64) ? RAD_OK : RAD_E_SHAPE;
}

/* Operand indices, from oAttnPaged in r4d_rows.cpp. */
enum { AP_Q = 0, AP_KV, AP_BT, AP_SEQ, AP_KDS, AP_VDS, AP_CU, AP_OUT, AP_TREE };

/* Fill R4DArgs from a RadArgs. Returns RAD_OK, or the negative code naming what is wrong.
 *
 * RAGGED BATCHES. libr4d's paged kernels take a scalar q_len AND an optional cu_seqlens, and which
 * one this call uses is decided here. Without cu_seqlens every sequence in the batch has to present
 * the same query length, which is a restriction the scheduler does not honour and cannot be asked
 * to: core/sched/scheduler.cpp builds RAD_PHASE_MIXED steps deliberately, because a decode row
 * costs nothing once another sequence's prefill chunk is already reading the weights. A step that
 * mixes them has total_q that need not divide n_seq, and refusing that with RAD_E_SHAPE would be
 * reported by the run phase as a kernel that does not serve the shape, which aborts the engine. So
 * the operand is passed whenever it is there and the uniform arithmetic is kept only for the case
 * where it is not.
 *
 * q_len then means the batch MAXIMUM, and it comes from the PARAMETER, which is exactly what the
 * arch block issued the op with (RAD_ISSUE_N(..., batch->max_q_len, ...)). Deriving it from the
 * tensors is only possible in the uniform case, and that is what the code below does there --
 * total_q / num_seqs -- because the parameter is a band bound there, not a count.
 */
static inline int fill_paged(const RadArgs* a, R4DArgs* r, int expect_head_dim) {
    const RadTensor* q  = r4d_opt(a, AP_Q);
    const RadTensor* kv = r4d_opt(a, AP_KV);
    const RadTensor* bt = r4d_opt(a, AP_BT);
    const RadTensor* su = r4d_opt(a, AP_SEQ);
    const RadTensor* cu = r4d_opt(a, AP_CU);
    const RadTensor* o  = r4d_opt(a, AP_OUT);
    if (!q || !kv || !bt || !su || !o) return R4D_NO(RAD_E_INVAL);

    if (q->rank < 2 || o->rank < 2 || bt->rank != 2 || su->rank < 1) return R4D_NO(RAD_E_SHAPE);
    if (kv->rank != 4) return R4D_NO(RAD_E_SHAPE);

    /* THE HEAD AXIS MAY BE SPELLED OR FLATTENED, and both are the same dense block. An architecture
     * plugin declares the query buffer [tokens, n_head, head_dim] when something downstream needs
     * the head axis and [tokens, n_head*head_dim] when nothing does -- a gated attention block does
     * the latter, because its query comes out of a de-interleave and goes straight here. The kernel
     * indexes one dense [total_q, q_heads, head_dim] block either way, so the shim reads the rank
     * it was given rather than demanding three. `head_dim` is the op's parameter, which is what
     * makes the flattened form readable at all. */
    const long long head_dim = expect_head_dim;
    const long long total_q  = q->shape[0];
    const long long num_seqs = bt->shape[0];
    if (num_seqs <= 0 || total_q <= 0 || head_dim <= 0) return R4D_NO(RAD_E_SHAPE);
    if (r4d_numel(q) % (total_q * head_dim)) return R4D_NO(RAD_E_SHAPE);
    const long long q_heads = r4d_numel(q) / (total_q * head_dim);
    if (q_heads <= 0) return R4D_NO(RAD_E_SHAPE);
    /* Ragged unless cu_seqlens says otherwise; see the note above fill_paged. */
    const bool ragged = (cu != nullptr) && cu->data;
    if (ragged) {
        if (cu->rank != 1 || cu->shape[0] != num_seqs + 1) return R4D_NO(RAD_E_SHAPE);
        if (!r4d_contig(cu)) return R4D_NO(RAD_E_STRIDE);
    } else if (total_q % num_seqs) {
        return R4D_NO(RAD_E_SHAPE);
    }

    /* q and out are indexed as one dense [total_q, q_heads, head_dim] block; the kernel has no
     * pitch argument for either. */
    if (!r4d_contig(q) || !r4d_contig(o)) return R4D_NO(RAD_E_STRIDE);
    if (o->shape[0] != total_q || r4d_numel(o) != total_q * q_heads * head_dim)
        return R4D_NO(RAD_E_SHAPE);
    if (su->shape[0] != num_seqs) return R4D_NO(RAD_E_SHAPE);
    if (!r4d_contig(bt) || !r4d_contig(su)) return R4D_NO(RAD_E_STRIDE);

    /* The KV cache is the one operand that IS pitched: kv_block_stride and kv_head_stride are
     * taken as arguments, so a padded cache is legal. Everything inside a (block, head) slot is
     * not, because the kernel walks it with a vector load. */
    const long long block_size = kv->shape[2];
    if (kv->stride[3] != 1) return R4D_NO(RAD_E_STRIDE);
    if (kv->stride[2] != 2 * head_dim) return R4D_NO(RAD_E_STRIDE);
    if (kv->shape[3] != 2 * head_dim) return R4D_NO(RAD_E_SHAPE);

    /* 16-byte alignment: every one of these is read through a 128-bit load. */
    if (!r4d_aligned(q->data, 16) || !r4d_aligned(o->data, 16) || !r4d_aligned(kv->data, 16))
        return R4D_NO(RAD_E_ALIGN);

    const RadTensor* kds = r4d_opt(a, AP_KDS);
    const RadTensor* vds = r4d_opt(a, AP_VDS);
    const RadTensor* tree = r4d_opt(a, AP_TREE);

    std::memset(r, 0, sizeof(*r));
    r->q           = q->data;
    r->kv          = kv->data;
    r->block_table = (const int*)bt->data;
    r->seqused_k   = (const int*)su->data;
    r->out         = o->data;
    r->k_descale   = kds ? (const float*)kds->data : nullptr;
    r->v_descale   = vds ? (const float*)vds->data : nullptr;
    if (tree && tree->data) {
        r->q_descale = (const float*)tree->data;
    } else {
        long long tree_ptr = 0;
        if (r4d_geti(a, "tree_ancestors", &tree_ptr) && tree_ptr != 0) {
            r->q_descale = (const float*)(uintptr_t)tree_ptr;
        } else {
            r->q_descale = nullptr;
        }
    }
    r->scratch     = a->scratch;
    r->num_seqs    = (int)num_seqs;
    r->cu_q        = ragged ? (const int*)cu->data : nullptr;
    r->total_q     = (int)total_q;
    /* q_len is the exact count when the batch is uniform, and an UPPER BOUND on the per-sequence
     * lengths when it is ragged -- it sizes the prefill grid and the decode wave count, and the
     * kernels take the exact lengths from cu_seqlens.
     *
     * The bound is the smaller of two things that are both >= the true maximum and neither of
     * which is it. The `q_len` parameter is the resolved BAND (RadArgs carries the geometry with
     * ranges collapsed, not the value the op was issued with), and the band was selected by
     * max_q_len, so it is at least that. total_q is a sum of lengths, so it is at least the
     * largest of them. Taking the smaller keeps the decode band legal: its bound is 10 and
     * 10 * gqa is 60 rows, under the 64 a decode workgroup holds, while total_q alone would be
     * the whole batch and would be refused with -4.
     *
     * The cost of the bound not being tight is empty workgroups that exit on their first
     * comparison -- the prefill kernel's `qb * BLOCK_Q >= qlen` and the decode kernel's `live`. */
    r->q_len       = (int)(total_q / num_seqs);
    if (ragged) {
        const long long band = r4d_geti_or(a, "q_len", total_q);
        r->q_len = (int)(band > 0 && band < total_q ? band : total_q);
    }
    r->q_heads     = (int)q_heads;
    r->kv_heads    = (int)kv->shape[1];
    r->head_dim    = (int)head_dim;
    r->block_size  = (int)block_size;
    r->max_blocks  = (int)bt->shape[1];
    r->kv_block_stride = (long)kv->stride[0];
    r->kv_head_stride  = (long)kv->stride[1];
    r->window      = (int)r4d_geti_or(a, "window", 0);
    r->causal      = (int)r4d_geti_or(a, "causal", 1);
    r->causal_seq  = nullptr;

    /* The softmax scale. Absent, 1/sqrt(head_dim) -- right for every model in scope, and a
     * parameter rather than a hardcode for the ones that are not. */
    r->scale = (float)r4d_getf(a, "scale", 1.0 / std::sqrt((double)head_dim));

    /* max_ctx is the HOST-visible context bound; seqused_k is device-side and the split law
     * cannot read it. The block table's width times the block size is an exact upper bound on
     * what any sequence in this batch can have, so it is a safe default. */
    long long mc = r4d_geti_or(a, "max_ctx", 0);
    if (mc <= 0) mc = (long long)r->max_blocks * block_size;
    r->max_ctx = (int)mc;

    /* Set by the caller from the tuned `splits`; 0 means libr4d's own split law. */
    r->splits = 0;
    return RAD_OK;
}

/* The decode rows declare a `splits` axis; its default is 0, which is that same law. */
static inline int split_of(const RadArgs* a) {
    R4dSplitCfg c;
    r4d_cfg_split(a, &c);
    return c.splits;
}


#endif  /* R4D_ATTN_ARGS_H */
