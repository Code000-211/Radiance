/* ref_attn.cpp -- paged attention, dense attention, the KV write, and the two rotary forms.
 *
 * These ops are the easiest to get subtly wrong, so they favour legibility: the mask is an explicit
 * predicate per key, the softmax is two plain passes with the max subtracted, and nothing is fused.
 *
 * ============================== The paged layout ==============================
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]      K then V inside a slot
 *
 * The layout every kernel must match (ref_ops.h; libr4d's `R4DArgs::kv`). A flat slot index s is
 * block = s / block_size, offset = s % block_size; `block_table[seq, i]` holds the block id of the
 * sequence's i-th block and `slot_mapping[token]` the flat slot a token's KV goes to.
 *
 * ============================== The mask ==============================
 *
 * `seqused[s]` is the sequence's total key count including this step's query tokens, so query row
 * i of a q_len-row step sits at absolute position seqused[s] - q_len + i. One kernel thus serves
 * decode (q_len 1), speculative verify (q_len <= 64) and prefill chunks (q_len in the thousands).
 *
 *   causal = 1   key j is visible when j <= pos
 *   causal = 0   key j is visible when j <  seqused[s]
 *   window > 0   causal: additionally pos - j < window; non-causal: |pos - j| < window, which is
 *                the two-sided reading libr4d's h128 kernel documents.
 */
#include "ref_common.h"
#include "rad_yarn.h"
/* `row_stride`: every q/k/v/out row stride below goes through it rather than a rank-based
 * shortcut; see kv_store for where the shortcut is wrong. */
#include "ref_gemm.h"
#include "ref_ops.h"

#include <algorithm>

using namespace ref;

namespace {

/* The largest head this reference accumulates in one stack frame. The widest in use is an image
 * VAE's single-head mid-block attention at 1152 (Qwen-Image 2.1); above 2048 the op returns
 * RAD_E_UNSUPPORTED instead of overflowing the frame. */
enum { REF_HD_MAX = 2048 };

struct KVView {
    const RadTensor* t;
    int64_t sb, sh, so, sd;   /* strides: block, kv head, slot offset, element */
    int64_t head_dim;
};

static inline KVView kv_view(const RadTensor* c, int64_t head_dim, int64_t block_size,
                             int64_t kv_heads) {
    KVView v{};
    v.t = c;
    v.head_dim = head_dim;
    if (c->rank == 4) {
        v.sb = c->stride[0]; v.sh = c->stride[1]; v.so = c->stride[2]; v.sd = c->stride[3];
    } else {
        v.sd = 1;
        v.so = 2 * head_dim;
        v.sh = block_size * v.so;
        v.sb = kv_heads * v.sh;
    }
    return v;
}

/* Cache element loaders with the dtype resolved once per launch: the same floats as ldt, and the
 * per-element switch out of the inner loops. */
template <uint32_t DT> struct KvFix {
    float operator()(const void* p, int64_t i) const { return Cvt<DT>::ld(p, i); }
};
struct KvDyn  {
    uint32_t dt;
    float operator()(const void* p, int64_t i) const { return ld_dt(dt, p, i); }
};

}  /* namespace */

/* ------------------------------------------------------------------ attn_paged */
/* The shared body. `anc` null is attn_paged: causal over positions. Non-null is attn_paged_tree,
 * where each query row is a node of a candidate tree, op 8 is its uint16 ancestor set (bit b names
 * the node held in cache row root_pos + b) and the row sees the prefix below root_pos plus exactly
 * the rows its mask names. See ref_attn_paged_tree for the admitted form. */
static int ref_attn_paged_impl(const RadArgs* a, const RadTensor* anc, int64_t root_pos) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *kv = &a->t[1], *bt = &a->t[2], *su = &a->t[3], *o = &a->t[7];
    const RadTensor* kdes = t_in(a, 4);
    const RadTensor* vdes = t_in(a, 5);
    const RadTensor* cus  = t_in(a, 6);
    if (!q->data || !kv->data || !bt->data || !su->data || !o->data) return RAD_E_INVAL;

    const int64_t head_dim   = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa        = p_int(a, "gqa", 1);
    /* `block_size` is RAD_DERIVED here (docs/OPS.md), so a call resolved to ref's row may arrive
     * without it. Read it off the cache tensor, and cross-check below when the caller names one. */
    const int64_t block_size = p_int(a, "block_size", kv->rank == 4 ? kv->shape[2] : 0);
    const int     causal     = (int)p_int(a, "causal", 1);
    const int64_t window     = p_int(a, "window", 0);
    if (head_dim <= 0 || gqa <= 0 || block_size <= 0) return RAD_E_SHAPE;
    if (head_dim > REF_HD_MAX) return RAD_E_UNSUPPORTED;

    const int64_t kv_heads = kv->rank == 4 ? kv->shape[1] : 0;
    if (kv_heads <= 0) return RAD_E_SHAPE;
    const int64_t q_heads = gqa * kv_heads;
    /* gqa and the cache's head count fix the query head count; a rank-3 query that disagrees would
     * stride through the wrong rows. */
    if (q->rank >= 3 && q->shape[q->rank - 2] != q_heads) return RAD_E_SHAPE;
    const int64_t n_seq   = numel(su);
    const int64_t rows    = numel(q) / (q_heads * head_dim);
    if (n_seq <= 0 || rows <= 0) return RAD_E_SHAPE;
    /* Ragged batches. cu_seqlens is optional; absent, every sequence has the same query length.
     * Present, it is [n_seq + 1] with cu[0] == 0 and cu[n_seq] == rows, and sequence s owns query
     * rows [cu[s], cu[s+1]). Serving mixes a prefill chunk with decode rows in one step, so ragged
     * is the ordinary case. */
    if (!cus && rows % n_seq != 0) return RAD_E_SHAPE;
    if (cus && numel(cus) != n_seq + 1) return RAD_E_SHAPE;
    const int64_t q_len = cus ? 0 : rows / n_seq;
    if (numel(o) < rows * q_heads * head_dim) return RAD_E_SHAPE;
    if (kv->rank == 4 && kv->shape[2] != block_size) return RAD_E_SHAPE;
    if (kv->rank == 4 && kv->shape[3] < 2 * head_dim) return RAD_E_SHAPE;

    const int64_t max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : numel(bt) / n_seq;
    const int64_t bt_rs = bt->rank >= 2 ? bt->stride[bt->rank - 2] : max_blocks;
    const int64_t bt_cs = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;

    /* A tree step's masks, checked before a row is computed. The sequence's nodes are cache rows
     * root_pos .. root_pos + qlen - 1, one row a node, so siblings sit on distinct rows even
     * though they share a depth; `seqused` must reach the last of them or the mask would name a row
     * the step cannot read. A mask with a bit past its sequence's nodes, without the root, or
     * without the row's own node conditions a row on text that is not on its path. All refused,
     * none clamped. A tree step also needs every row of its path mapped: the block table has to
     * address the sequence's last needed row and hold a block for each one, or required rows would
     * silently drop out of the softmax and a shortened attention would answer RAD_OK. */
    if (anc) {
        if (anc->dtype != RAD_U16 && anc->dtype != RAD_U32) return RAD_E_INVAL;
        if (numel(anc) < rows) return RAD_E_SHAPE;
        if (causal != 1 || window != 0) return RAD_E_UNSUPPORTED;
        if (root_pos < 0) return RAD_E_INVAL;
        for (int64_t s = 0; s < n_seq; ++s) {
            const int64_t qlen = cus ? cu_at(cus, s + 1) - cu_at(cus, s) : q_len;
            if (qlen <= 0 || qlen > 32) return RAD_E_UNSUPPORTED;   /* one mask a row */
            const int64_t first = cus ? cu_at(cus, s) : s * q_len;
            if (ld_int(su->dtype, su->data, offlin(su, s)) < root_pos + qlen) return RAD_E_SHAPE;
            for (int64_t i = 0; i < qlen; ++i) {
                const int64_t m = ld_int(anc->dtype, anc->data, offlin(anc, first + i));
                if ((m >> qlen) != 0) return RAD_E_INVAL;         /* a node this step does not have */
                if ((m & 1) == 0) return RAD_E_INVAL;             /* the root is on every path */
                if (((m >> i) & 1) == 0) return RAD_E_INVAL;      /* a row attends to itself */
            }
            /* Every row the paths use is a block the table must name: the prefix rows 0 ..
             * root_pos - 1 and the node rows root_pos .. root_pos + qlen - 1. */
            const int64_t need = (root_pos + qlen - 1) / block_size + 1;
            if (need > max_blocks) return RAD_E_SHAPE;
            for (int64_t b = 0; b < need; ++b)
                if (ld_int(bt->dtype, bt->data, s * bt_rs + b * bt_cs) < 0) return RAD_E_INVAL;
        }
    }

    const KVView kvv = kv_view(kv, head_dim, block_size, kv_heads);

    /* Row strides of q and out read as [rows, q_heads, head_dim], at whatever rank they arrived. */
    const int64_t q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    const int64_t q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : head_dim * q_ds;
    const int64_t q_rs = row_stride(q, q_heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    const int64_t o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : head_dim * o_ds;
    const int64_t o_rs = row_stride(o, q_heads * head_dim);

    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));
    const int64_t total = rows * q_heads;
    const int64_t jcap = max_blocks * block_size;

    /* Every (query row, head) is one output row, computed as below in a fixed order: per visible
     * key, dot = sum over d of q[d] * (k[d] * k_descale), score = dot * scale; then the max over the
     * scores in key order; then sum and acc[d] accumulate exp(score - max) and
     * exp(...) * (v[d] * v_descale) in key order. The query row is widened once, each score is kept
     * for the second pass instead of recomputed, and four keys' dot chains run side by side; none
     * of that changes a value. */
    REF_PAR_REGION(total * head_dim * 64)
    {
        float* const sc = ws_get((size_t)std::max<int64_t>(jcap, 1));
        REF_FOR_DYN
        for (int64_t idx = 0; idx < total; ++idx) {
            auto unit = [&](auto kl) {
                const int64_t h   = idx % q_heads;
                const int64_t tok = idx / q_heads;          /* the dense query row */
                /* The owning sequence and the row's index within its query. */
                int64_t s = 0, qlen = q_len, i = 0;
                if (cus) {
                    s = seq_of(cus, n_seq, tok);
                    qlen = cu_at(cus, s + 1) - cu_at(cus, s);
                    i = tok - cu_at(cus, s);
                } else {
                    s = tok / q_len;
                    i = tok - s * q_len;
                }
                if (qlen <= 0) return;
                const int64_t kh = h / gqa;

                const int64_t ctx = ld_int(su->dtype, su->data, offlin(su, s));
                /* A tree row's position is not a key index: its keys are the prefix rows below
                 * root_pos plus the node rows its mask names, and siblings share a depth while
                 * holding distinct rows. So the causal rule over positions is replaced, and `pos`
                 * (unused by the tree branch of key_at, window 0 there) is only the causal row's. */
                int64_t mask = 0;
                if (anc) mask = ld_int(anc->dtype, anc->data, offlin(anc, tok));
                const int64_t pos = anc ? 0 : ctx - qlen + i;
                const int64_t qb  = tok * q_rs + h * q_hs;
                const int64_t ob  = tok * o_rs + h * o_hs;

                const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
                const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;

                /* Keys past the block table's width are never read. */
                const int64_t jend = std::min(ctx, jcap);
                /* Key j's slot in the cache, or -1 when the mask hides it or its block is
                 * unmapped. */
                auto key_at = [&](int64_t j) -> int64_t {
                    if (anc) {
                        /* The prefix below the root is on every path; past it, exactly the node
                         * rows the mask names. A clear bit is a sibling, not a later position.
                         * Rows past the step's nodes are hidden before the mask is shifted: a
                         * mask is one u16/u32 a row, so bit `qlen` and above is no bit of it, and
                         * shifting by it (`seqused` may run well past the nodes) is undefined. */
                        const int64_t rel = j - root_pos;
                        if (rel >= qlen) return -1;
                        if (rel >= 0 && ((mask >> rel) & 1) == 0) return -1;
                    } else if (causal && j > pos) return -1;
                    if (window > 0) {
                        const int64_t d = pos - j;
                        if (causal ? (d >= window) : (d >= window || d <= -window)) return -1;
                    }
                    const int64_t blk =
                        ld_int(bt->dtype, bt->data, s * bt_rs + (j / block_size) * bt_cs);
                    return blk < 0 ? -1 : blk * kvv.sb + kh * kvv.sh + (j % block_size) * kvv.so;
                };

                float qf[REF_HD_MAX];
                for (int64_t d = 0; d < head_dim; ++d) qf[d] = ldt(q, qb + d * q_ds);

                /* Pass one: every visible key's score, four keys at a time, and the maximum. */
                float mx = -INFINITY;
                for (int64_t j0 = 0; j0 < jend; j0 += 4) {
                    int64_t kb[4];
                    float dot[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    const int nk = (int)std::min<int64_t>(4, jend - j0);
                    for (int u = 0; u < 4; ++u) kb[u] = u < nk ? key_at(j0 + u) : -1;
                    if (kb[0] >= 0 && kb[1] >= 0 && kb[2] >= 0 && kb[3] >= 0) {
                        for (int64_t d = 0; d < head_dim; ++d) {
                            const float qd = qf[d];
                            dot[0] += qd * (kl(kv->data, kb[0] + d * kvv.sd) * kds);
                            dot[1] += qd * (kl(kv->data, kb[1] + d * kvv.sd) * kds);
                            dot[2] += qd * (kl(kv->data, kb[2] + d * kvv.sd) * kds);
                            dot[3] += qd * (kl(kv->data, kb[3] + d * kvv.sd) * kds);
                        }
                    } else {
                        for (int u = 0; u < nk; ++u) {
                            if (kb[u] < 0) continue;
                            for (int64_t d = 0; d < head_dim; ++d)
                                dot[u] += qf[d] * (kl(kv->data, kb[u] + d * kvv.sd) * kds);
                        }
                    }
                    for (int u = 0; u < nk; ++u) {
                        if (kb[u] < 0) continue;
                        const float sj = dot[u] * scale;
                        sc[j0 + u] = sj;
                        if (sj > mx) mx = sj;
                    }
                }

                /* No visible key (an empty sliding window) gives a zero row, not NaN, matching
                 * every kernel in the project. */
                if (mx == -INFINITY) {
                    for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
                    return;
                }

                /* Pass two: exponentiate and accumulate in ascending key order. */
                float acc[REF_HD_MAX];
                for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
                float sum = 0.0f;
                for (int64_t j = 0; j < jend; ++j) {
                    const int64_t kb = key_at(j);
                    if (kb < 0) continue;
                    const float w = ref_exp(sc[j] - mx);
                    sum += w;
                    const int64_t vb = kb + head_dim * kvv.sd;
                    for (int64_t d = 0; d < head_dim; ++d)
                        acc[d] += w * (kl(kv->data, vb + d * kvv.sd) * vds);
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
            };
            switch (kv->dtype) {
                case RAD_F32:    unit(KvFix<RAD_F32>{}); break;
                case RAD_BF16:   unit(KvFix<RAD_BF16>{}); break;
                case RAD_F8E4M3: unit(KvFix<RAD_F8E4M3>{}); break;
                default:         unit(KvDyn{ kv->dtype }); break;
            }
        }
    }
    return RAD_OK;
}

int ref_attn_paged(const RadArgs* a, RadStream) { return ref_attn_paged_impl(a, nullptr, 0); }

/* An ancestor-masked paged attention: one verify step of a candidate tree
 * (arch/common/rad_tree_candidate.h). Operands 0..7 are attn_paged's; operand 8 is `ancestors`, one
 * uint16 per query row, and the row is the tree node whose id is its index within its sequence.
 * `root_pos` is the row index of the root (the last committed token): the tree's nodes occupy cache
 * rows root_pos .. root_pos + nodes - 1, one row a node, which is what keeps siblings -- same
 * depth, different token -- on distinct slots. A row attends to the committed prefix (rows below
 * root_pos) plus exactly the node rows its mask names, itself included, so the mask decides which
 * text is on the row's path where a causal kernel would decide by position alone. `seqused` is the
 * sequence's total key count including every node of the tree. The row's RoPE position (its depth
 * past the root) is the rope stage's, not this op's.
 *
 * Admitted and refused, never approximated: `causal` 1 and `window` 0 (a verify is an unwindowed
 * causal step, and a window over tree rows would mask by position, not by path); at most 32 nodes
 * per sequence (one mask each); every mask with the root and the row's own node set and no bit past
 * the sequence's nodes; root_pos >= 0; a `seqused` reaching the last node's row; and a block table
 * that both addresses that row and maps every block the paths use, so no needed row can drop out of
 * the softmax.
 * The mask's agreement with the tree's topology is the caller's check (masks_consistent), because
 * the operand carries the sets and not the parents. */
int ref_attn_paged_tree(const RadArgs* a, RadStream) {
    if (a->n_t < 9) return RAD_E_INVAL;
    const RadTensor* anc = &a->t[8];
    if (!anc->data) return RAD_E_INVAL;
    return ref_attn_paged_impl(a, anc, p_int(a, "root_pos", -1));
}

/* ------------------------------------------------------------------ attn_dense */
/* For the vision tower: unpaged, unquantised, `cu_seqlens` bounding one image (or attention window)
 * per entry so a whole batch is one launch. Causal is a parameter so a dense decoder-only prefill
 * can use the same kernel. */
int ref_attn_dense(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[3], *o = &a->t[4];

    const int64_t head_dim = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa      = p_int(a, "gqa", 1);
    const int     causal   = (int)p_int(a, "causal", 0);
    /* `window` follows attn_paged's rule over a segment's own positions: causal, key j is visible
     * to query i when i - j < window; non-causal, when |i - j| < window. 0 is no window. `block`
     * cuts a segment into blocks of that many rows from its start, each attending only within
     * itself (a chunked encoder's windows). `cu_div` reads cu_seqlens in units of that many rows:
     * a segment is rows [cu[s] / cu_div, cu[s+1] / cu_div). */
    const int64_t window   = p_int(a, "window", 0);
    /* Non-causal only: the window's right side, key j visible when j - i < window_right; 0 is
     * `window`'s. A sliding encoder layer that sees `l` keys behind and `r` ahead is window l + 1,
     * window_right r + 1. */
    const int64_t wright   = p_int(a, "window_right", 0);
    const int64_t block    = p_int(a, "block", 0);
    const int64_t cdiv     = p_int(a, "cu_div", 1);
    if (head_dim <= 0 || gqa <= 0 || window < 0 || block < 0 || cdiv <= 0 || wright < 0 ||
        (wright > 0 && (causal || window == 0)))
        return RAD_E_SHAPE;
    if (head_dim > REF_HD_MAX) return RAD_E_UNSUPPORTED;

    /* [tokens, heads, head_dim], or [tokens, heads * head_dim] as a column slice of a fused
     * projection with the projection's row pitch. */
    const int64_t q_heads = q->rank >= 3 ? q->shape[q->rank - 2]
                          : (q->rank == 2 ? q->shape[1] / head_dim : 1);
    const int64_t kv_heads = q_heads / gqa;
    if (kv_heads <= 0 || q_heads % gqa != 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t total_q = numel(q) / (q_heads * head_dim);
    if (total_q <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */
    if (numel(o) < total_q * q_heads * head_dim) return RAD_E_SHAPE;
    if (numel(k) < total_q * kv_heads * head_dim) return RAD_E_SHAPE;
    if (numel(v) < total_q * kv_heads * head_dim) return RAD_E_SHAPE;

    const int64_t q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    const int64_t q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : head_dim * q_ds;
    const int64_t q_rs = row_stride(q, q_heads * head_dim);
    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t v_rs = row_stride(v, kv_heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    const int64_t o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : head_dim * o_ds;
    const int64_t o_rs = row_stride(o, q_heads * head_dim);

    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));

    REF_PAR_FOR(total_q * q_heads * head_dim * 64)
    for (int64_t sh = 0; sh < n_seq * q_heads; ++sh) {
        const int64_t s = sh / q_heads, h = sh % q_heads, kh = h / gqa;
        const int64_t b0 = cu_at(cu, s) / cdiv;
        const int64_t b1 = cu_at(cu, s + 1) / cdiv;
        for (int64_t i = b0; i < b1; ++i) {
            const int64_t qb = i * q_rs + h * q_hs, ob = i * o_rs + h * o_hs;
            int64_t jbeg = b0, jend = causal ? (i + 1) : b1;
            if (window > 0) {
                jbeg = std::max<int64_t>(b0, i - window + 1);
                if (!causal) jend = std::min<int64_t>(b1, i + (wright > 0 ? wright : window));
            }
            if (block > 0) {
                const int64_t blk = b0 + (i - b0) / block * block;
                jbeg = std::max<int64_t>(jbeg, blk);
                jend = std::min<int64_t>(jend, blk + block);
            }
            float mx = -INFINITY;
            for (int64_t j = jbeg; j < jend; ++j) {
                const int64_t kb = j * k_rs + kh * k_hs;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d)
                    dot += ldt(q, qb + d * q_ds) * ldt(k, kb + d * k_ds);
                dot *= scale;
                if (dot > mx) mx = dot;
            }
            float acc[REF_HD_MAX];
            for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            if (mx == -INFINITY) {
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
                continue;
            }
            float sum = 0.0f;
            for (int64_t j = jbeg; j < jend; ++j) {
                const int64_t kb = j * k_rs + kh * k_hs, vb = j * v_rs + kh * v_hs;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d)
                    dot += ldt(q, qb + d * q_ds) * ldt(k, kb + d * k_ds);
                const float w = ref_exp(dot * scale - mx);
                sum += w;
                for (int64_t d = 0; d < head_dim; ++d) acc[d] += w * ldt(v, vb + d * v_ds);
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_cross */
/* attn_dense with each entry's keys a run of their own: entry `s`'s queries are rows
 * [cu[s], cu[s+1]) of `q` and its keys and values rows [cu_k[s], cu_k[s+1]) of `k` and `v`. With
 * `causal` a query sees key j of its run when j <= its own index in its entry, both counted from
 * the start: a video's rows reading its prompt (cross attention), or a padded prompt's rows of which
 * the padding sees only the prompt. A query that sees no key is zero. */
int ref_attn_cross(const RadArgs* a, RadStream) {
    if (!have(a, 6)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[3], *cuk = &a->t[4];
    const RadTensor *o = &a->t[5];
    const int64_t head_dim = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa      = p_int(a, "gqa", 1);
    const int     causal   = (int)p_int(a, "causal", 0);
    if (head_dim <= 0 || gqa <= 0) return RAD_E_SHAPE;
    if (head_dim > REF_HD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t q_heads = q->rank >= 3 ? q->shape[q->rank - 2]
                          : (q->rank == 2 ? q->shape[1] / head_dim : 1);
    const int64_t kv_heads = q_heads / gqa;
    if (kv_heads <= 0 || q_heads % gqa != 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0 || numel(cuk) < n_seq + 1) return RAD_E_SHAPE;
    const int64_t total_q = numel(q) / (q_heads * head_dim);
    const int64_t total_k = numel(k) / (kv_heads * head_dim);
    if (total_q <= 0 || cu_at(cu, n_seq) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */
    if (numel(o) < total_q * q_heads * head_dim) return RAD_E_SHAPE;
    for (int64_t s = 0; s < n_seq; ++s)
        if (cu_at(cu, s + 1) < cu_at(cu, s) || cu_at(cuk, s + 1) < cu_at(cuk, s) ||
            cu_at(cu, s + 1) > total_q || cu_at(cuk, s + 1) > total_k)
            return RAD_E_SHAPE;

    const int64_t q_ds = q->rank >= 2 ? q->stride[q->rank - 1] : 1;
    const int64_t q_hs = q->rank >= 3 ? q->stride[q->rank - 2] : head_dim * q_ds;
    const int64_t q_rs = row_stride(q, q_heads * head_dim);
    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t v_rs = row_stride(v, kv_heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1;
    const int64_t o_hs = o->rank >= 3 ? o->stride[o->rank - 2] : head_dim * o_ds;
    const int64_t o_rs = row_stride(o, q_heads * head_dim);
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));

    REF_PAR_FOR(total_q * q_heads * head_dim * 64)
    for (int64_t sh = 0; sh < n_seq * q_heads; ++sh) {
        const int64_t s = sh / q_heads, h = sh % q_heads, kh = h / gqa;
        const int64_t b0 = cu_at(cu, s), b1 = cu_at(cu, s + 1);
        const int64_t k0 = cu_at(cuk, s), k1 = cu_at(cuk, s + 1);
        for (int64_t i = b0; i < b1; ++i) {
            const int64_t qb = i * q_rs + h * q_hs, ob = i * o_rs + h * o_hs;
            const int64_t jend = causal ? std::min<int64_t>(k1, k0 + (i - b0) + 1) : k1;
            float mx = -INFINITY;
            for (int64_t j = k0; j < jend; ++j) {
                const int64_t kb = j * k_rs + kh * k_hs;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d)
                    dot += ldt(q, qb + d * q_ds) * ldt(k, kb + d * k_ds);
                dot *= scale;
                if (dot > mx) mx = dot;
            }
            if (mx == -INFINITY) {
                for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
                continue;
            }
            float acc[REF_HD_MAX];
            for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            float sum = 0.0f;
            for (int64_t j = k0; j < jend; ++j) {
                const int64_t kb = j * k_rs + kh * k_hs, vb = j * v_rs + kh * v_hs;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d)
                    dot += ldt(q, qb + d * q_ds) * ldt(k, kb + d * k_ds);
                const float w = ref_exp(dot * scale - mx);
                sum += w;
                for (int64_t d = 0; d < head_dim; ++d) acc[d] += w * ldt(v, vb + d * v_ds);
            }
            const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
        }
    }
    return RAD_OK;
}

/* attn_na3d: 3-D neighborhood attention, NATTEN's na3d (LTX-2.5's diffusion decoder). The rows of
 * q, k, v and out are the voxels of the volume geom = (T, H, W), i32 on the device, (t, i, j)
 * row-major. A query attends the keys of a window `kt` x `kh` x `kw` (each extent at most its
 * axis) centred on it and shifted inward at the border: along an axis of L and window n the
 * window starts at clamp(c - n / 2, 0, L - n). Per head, in f32: each score the dot over d
 * ascending times `scale` (default 1 / sqrt(head_dim)), the window's keys in raster order, the
 * softmax as attn_cross takes it (the maximum, then the exponentials' sum), the output rounded to
 * its dtype. Rows past T H W are not written. */
int ref_attn_na3d(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *G = &a->t[3], *o = &a->t[4];
    const int64_t head_dim = p_int(a, "head_dim", 0);
    const int64_t kt = p_int(a, "kt", 0), kh = p_int(a, "kh", 0), kw = p_int(a, "kw", 0);
    if (head_dim <= 0 || kt <= 0 || kh <= 0 || kw <= 0 || numel(G) < 3 || q->rank != 2)
        return RAD_E_SHAPE;
    if (head_dim > REF_HD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t heads = q->shape[1] / head_dim;
    const int64_t T = cu_at(G, 0), H = cu_at(G, 1), W = cu_at(G, 2);
    if (heads <= 0 || q->shape[1] % head_dim || T <= 0 || H <= 0 || W <= 0) return RAD_E_SHAPE;
    const int64_t N = std::min(q->shape[0], T * H * W);
    if (N <= 0 || numel(k) < N * heads * head_dim || numel(v) < N * heads * head_dim ||
        numel(o) < N * heads * head_dim)
        return RAD_E_SHAPE;
    const int64_t nt = std::min(kt, T), nh = std::min(kh, H), nw = std::min(kw, W);
    const int64_t q_ds = q->stride[1], q_rs = row_stride(q, heads * head_dim);
    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1, k_rs = row_stride(k, heads * head_dim);
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1, v_rs = row_stride(v, heads * head_dim);
    const int64_t o_ds = o->rank >= 2 ? o->stride[o->rank - 1] : 1, o_rs = row_stride(o, heads * head_dim);
    const float scale = p_f32(a, "scale", 1.0f / std::sqrt((float)head_dim));
    auto start = [](int64_t c, int64_t n, int64_t L) {
        return std::min(std::max<int64_t>(c - n / 2, 0), L - n);
    };
    REF_PAR_FOR(N * heads * nt * nh * nw * head_dim)
    for (int64_t rh = 0; rh < N * heads; ++rh) {
        const int64_t r = rh / heads, h = rh % heads;
        const int64_t t = r / (H * W), i = (r / W) % H, j = r % W;
        const int64_t t0 = start(t, nt, T), i0 = start(i, nh, H), j0 = start(j, nw, W);
        const int64_t qb = r * q_rs + h * head_dim * q_ds, ob = r * o_rs + h * head_dim * o_ds;
        auto score = [&](int64_t key) {
            const int64_t kb = key * k_rs + h * head_dim * k_ds;
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) dot += ldt(q, qb + d * q_ds) * ldt(k, kb + d * k_ds);
            return dot * scale;
        };
        float mx = -INFINITY;
        for (int64_t a1 = t0; a1 < t0 + nt; ++a1)
            for (int64_t b1 = i0; b1 < i0 + nh; ++b1)
                for (int64_t c1 = j0; c1 < j0 + nw; ++c1) mx = std::max(mx, score((a1 * H + b1) * W + c1));
        float acc[REF_HD_MAX];
        for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
        float sum = 0.0f;
        for (int64_t a1 = t0; a1 < t0 + nt; ++a1)
            for (int64_t b1 = i0; b1 < i0 + nh; ++b1)
                for (int64_t c1 = j0; c1 < j0 + nw; ++c1) {
                    const int64_t key = (a1 * H + b1) * W + c1;
                    const float w = ref_exp(score(key) - mx);
                    sum += w;
                    const int64_t vb = key * v_rs + h * head_dim * v_ds;
                    for (int64_t d = 0; d < head_dim; ++d) acc[d] += w * ldt(v, vb + d * v_ds);
                }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ kv_store */
/* Writes this step's keys and values into the paged cache at `slot_mapping`. The cache is INOUT,
 * not OUT: only mapped slots are touched and every other slot must survive, which the buffer
 * planner needs to know (docs/OPS.md draws it as an output; this plugin deliberately differs).
 *
 * A negative slot is skipped: it marks a padded row with no cache slot. libr4d treats slot <= 0 as
 * null because vLLM reserves block 0, but a generic reference must not steal a valid slot.
 *
 * `k` and `v` rows are not dense. A de-interleaved projection gives [tokens, kv_heads, head_dim];
 * otherwise they are a column slice of the fused QKV plane, [tokens, kv_heads * head_dim] with the
 * projection's row stride (2560 vs 256 for 16 heads and 2 kv heads). Deriving the stride from the
 * shape would write every token after the first from the wrong rows, so `row_stride` reads the
 * operand. */
int ref_kv_store(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *k = &a->t[0], *v = &a->t[1], *sm = &a->t[2], *kv = &a->t[3];

    const int64_t head_dim   = p_int(a, "head_dim", k->rank >= 3 ? k->shape[k->rank - 1] : 0);
    const int64_t block_size = p_int(a, "block_size", 0);
    int64_t kv_heads = p_int(a, "n_head_kv", kv->rank == 4 ? kv->shape[1] : 0);
    if (head_dim <= 0 || block_size <= 0 || kv_heads <= 0) return RAD_E_SHAPE;
    if (kv->rank == 4 && (kv->shape[1] != kv_heads || kv->shape[2] != block_size ||
                          kv->shape[3] < 2 * head_dim)) return RAD_E_SHAPE;

    const int64_t M = numel(sm);
    if (M <= 0 || numel(k) < M * kv_heads * head_dim) return RAD_E_SHAPE;
    const int64_t n_blocks = kv->rank == 4 ? kv->shape[0] : 0;

    const int64_t k_ds = k->rank >= 2 ? k->stride[k->rank - 1] : 1;
    const int64_t k_hs = k->rank >= 3 ? k->stride[k->rank - 2] : head_dim * k_ds;
    const int64_t k_rs = row_stride(k, kv_heads * head_dim);
    const int64_t v_ds = v->rank >= 2 ? v->stride[v->rank - 1] : 1;
    const int64_t v_hs = v->rank >= 3 ? v->stride[v->rank - 2] : head_dim * v_ds;
    const int64_t v_rs = row_stride(v, kv_heads * head_dim);
    const KVView kvv = kv_view(kv, head_dim, block_size, kv_heads);
    /* An fp8 cache stores k / k_scale, where the scales are the reader's descales (default 1.0).
     * A wider cache stores the value itself. */
    const bool  fp8 = kv->dtype == RAD_F8E4M3 || kv->dtype == RAD_F8E5M2;
    const float ks = fp8 ? 1.0f / p_f32(a, "k_scale", 1.0f) : 1.0f;
    const float vs = fp8 ? 1.0f / p_f32(a, "v_scale", 1.0f) : 1.0f;

    for (int64_t t = 0; t < M; ++t) {
        const int64_t slot = ld_int(sm->dtype, sm->data, offlin(sm, t));
        if (slot >= 0 && n_blocks > 0 && slot / block_size >= n_blocks) return RAD_E_SHAPE;
    }

    REF_PAR_FOR(M * kv_heads * head_dim * 2)
    for (int64_t t = 0; t < M; ++t) {
        const int64_t slot = ld_int(sm->dtype, sm->data, offlin(sm, t));
        if (slot < 0) continue;
        const int64_t blk = slot / block_size, off = slot % block_size;
        for (int64_t h = 0; h < kv_heads; ++h) {
            const int64_t cb = blk * kvv.sb + h * kvv.sh + off * kvv.so;
            for (int64_t d = 0; d < head_dim; ++d) {
                stt(kv, cb + d * kvv.sd, ldt(k, t * k_rs + h * k_hs + d * k_ds) * ks);
                stt(kv, cb + (head_dim + d) * kvv.sd, ldt(v, t * v_rs + h * v_hs + d * v_ds) * vs);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rope */
namespace {

/* NeoX pairs element i with i + rotary_dim/2; GPT-J ("interleaved") pairs 2i with 2i+1. The
 * checkpoint decides, so it is a parameter: the wrong one yields fluent text with broken position
 * sense, which is hard to notice.
 *
 * The multi-component modes pair like NeoX and differ in which position component drives each
 * frequency (i indexes the rot/2 frequencies):
 *   mrope   contiguous sections: frequencies [0, s0) take component 0, the next s1 component 1...
 *           (Qwen2-VL), all on one ladder theta^(-2i/rot).
 *   imrope  interleaved (Qwen3-VL): component 1 where i % 3 == 1 and i < 3*s1, component 2 where
 *           i % 3 == 2 and i < 3*s2, component 0 everywhere else; one ladder.
 *   axial   contiguous sections, each on its own ladder theta^(-2j/(2*s_c)) for its j-th
 *           frequency: the vision tower's 2-D rotary embedding over (row, column).
 * With one position component per token (rank-1 `positions`) every component is the same, and
 * mrope and imrope reduce to neox. */
enum { ROPE_NEOX = 0, ROPE_GPTJ = 1, ROPE_MROPE = 2, ROPE_IMROPE = 3, ROPE_AXIAL = 4 };

struct RopeFreq {
    int   comp[REF_HD_MAX / 2];
    float inv[REF_HD_MAX / 2];
};

/* The component and inverse frequency of each of the rot/2 frequencies. `den` is the width the base
 * ladder is measured in: `rot` for a `rotary` ladder (a partial rotary is a full rotation of its
 * own width), `head_dim` for the proportional `head` variant. The loop count and pairing still come
 * from `rot`; `axial` overrides `inv`. */
static int rope_freqs(int mode, int64_t rot, int64_t den, float theta, const int64_t* sec,
                      int n_sec, RopeFreq* f) {
    const int64_t half = rot / 2;
    if (half > REF_HD_MAX / 2) return RAD_E_UNSUPPORTED;
    int64_t start[8] = {}, total = 0;
    for (int c = 0; c < n_sec; ++c) { start[c] = total; total += sec[c]; }
    if ((mode == ROPE_MROPE || mode == ROPE_AXIAL) && n_sec > 0 && total != half) return RAD_E_SHAPE;
    for (int64_t i = 0; i < half; ++i) {
        int c = 0;
        float inv = 1.0f / ref_pow(theta, (float)(2 * i) / (float)den);
        if (n_sec > 0 && (mode == ROPE_MROPE || mode == ROPE_AXIAL)) {
            while (c + 1 < n_sec && i >= start[c + 1]) ++c;
            if (mode == ROPE_AXIAL) {
                const int64_t j = i - start[c];
                inv = 1.0f / ref_pow(theta, (float)(2 * j) / (float)(2 * sec[c]));
            }
        } else if (n_sec >= 3 && mode == ROPE_IMROPE) {
            if (i % 3 == 1 && i < 3 * sec[1]) c = 1;
            else if (i % 3 == 2 && i < 3 * sec[2]) c = 2;
        }
        f->comp[i] = c;
        f->inv[i] = inv;
    }
    return RAD_OK;
}

static void rope_head(const RadTensor* t, int64_t base, int64_t ds, int64_t rot, int mode,
                      const RopeFreq& f, const float* pos, float magnitude = 1.0f) {
    const int64_t half = rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float ang = pos[f.comp[i]] * f.inv[i];
        const float c = ref_cos(ang) * magnitude, s = ref_sin(ang) * magnitude;
        const int64_t i0 = (mode == ROPE_GPTJ) ? 2 * i : i;
        const int64_t i1 = (mode == ROPE_GPTJ) ? 2 * i + 1 : i + half;
        const float x0 = ldt(t, base + i0 * ds), x1 = ldt(t, base + i1 * ds);
        stt(t, base + i0 * ds, x0 * c - x1 * s);
        stt(t, base + i1 * ds, x0 * s + x1 * c);
    }
}

/* A position as the rotation reads it, divided by the position scale in f32: an integer operand's
 * value, or a float operand's own value (a fractional coordinate, such as a video decoder's
 * normalised grid already multiplied by 2 pi). */
static inline float rope_pos(const RadTensor* pos, int64_t off, float div) {
    const uint32_t dt = pos->dtype;
    const bool flt = dt == RAD_F32 || dt == RAD_BF16 || dt == RAD_F16;
    return (flt ? ld_dt(dt, pos->data, off) : (float)ld_int(dt, pos->data, off)) / div;
}

/* "11 11 10" -> {11, 11, 10}. Absent is no sections. */
static int rope_sections(const RadArgs* a, int64_t* sec) {
    const char* v = p_str(a, "sections", nullptr);
    int n = 0;
    while (v && *v && n < 8) {
        char* end = nullptr;
        const long long x = std::strtoll(v, &end, 10);
        if (end == v) break;
        if (x <= 0) return -1;
        sec[n++] = x;
        v = end;
    }
    return n;
}

/* `ladder` names the width the base frequency ladder is measured in (see rope_freqs): absent or
 * `rotary` divides by `rot`, `head` by `head_dim`. An unknown value is refused, since a mistyped
 * ladder would silently give wrong angles. */
static int rope_ladder(const RadArgs* a, int64_t rot, int64_t head_dim, int64_t* den) {
    const char* lad = p_str(a, "ladder", nullptr);
    if (!lad) { *den = rot; return RAD_OK; }
    if (!std::strcmp(lad, "rotary")) { *den = rot; return RAD_OK; }
    if (!std::strcmp(lad, "head")) {
        if (head_dim <= 0) return RAD_E_SHAPE;
        *den = head_dim;
        return RAD_OK;
    }
    return RAD_E_UNSUPPORTED;
}

}  /* namespace */

int ref_rope(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *qkv = &a->t[0], *pos = &a->t[1];

    const int64_t head_dim  = p_int(a, "head_dim", 0);
    const int64_t n_head    = p_int(a, "n_head", 0);
    const int64_t n_head_kv = p_int(a, "n_head_kv", 0);
    const float   theta     = p_f32(a, "theta", 10000.0f);
    const float   rscale    = p_f32(a, "scale", 1.0f);
    static const char* kModes[] = { "neox", "gptj", "mrope", "imrope", "axial" };
    const int mode = p_enum(a, "mode", kModes, 5, -1);
    if (mode < 0) return RAD_E_UNSUPPORTED;
    int64_t rot = p_int(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0 || rot <= 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;
    int64_t den;
    RAD_REF_TRY(rope_ladder(a, rot, head_dim, &den));

    int64_t sec[8] = {};
    const int n_sec = rope_sections(a, sec);
    if (n_sec < 0) return RAD_E_SHAPE;
    /* Positions: [M] is one component per token; [C, M] is C components, component-major. */
    const bool multi = pos->rank >= 2;
    const int64_t n_comp = multi ? pos->shape[0] : 1;
    const int64_t M = multi ? pos->shape[1] : numel(pos);
    if (multi) {
        if (mode == ROPE_NEOX || mode == ROPE_GPTJ || n_sec <= 0) return RAD_E_SHAPE;
        const int64_t need = mode == ROPE_IMROPE ? 3 : n_sec;
        if (n_comp < need || n_comp > 8) return RAD_E_SHAPE;
    }
    /* The proportional ladder (`head`) is HF's: the rotation spans the whole head, pair i being
     * (i, i + head_dim/2), and frequencies past rotary_dim/2 are zero so those pairs do not turn.
     * A rotary of rotary_dim's own width, pairing (i, i + rot/2), would turn other channels. */
    const int64_t nfreq = rot / 2;
    if (den == head_dim && rot < head_dim && !multi && mode == ROPE_NEOX) rot = head_dim;
    RopeFreq f;
    RAD_REF_TRY(rope_freqs(mode, rot, den, theta, sec, multi ? n_sec : 0, &f));
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn)) return RAD_E_INVAL;
    if (yarn.factor != 0.0f) {
        if (mode == ROPE_AXIAL || rscale != 1.0f) return RAD_E_UNSUPPORTED;
        for (int64_t i = 0; i < rot / 2; ++i) f.inv[i] = yarn.frequency(f.inv[i], (int)i);
    }
    const RadLlama3 l3 = rad_llama3_args(a);
    if (!rad_llama3_valid(l3) || (l3.factor != 0.0f && yarn.factor != 0.0f)) return RAD_E_INVAL;
    if (l3.factor != 0.0f) {
        if (mode == ROPE_AXIAL || multi) return RAD_E_UNSUPPORTED;
        for (int64_t i = 0; i < rot / 2; ++i) f.inv[i] = l3.frequency(f.inv[i]);
    }
    for (int64_t i = nfreq; i < rot / 2; ++i) f.inv[i] = 0.0f;

    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    if (M <= 0 || numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    const int64_t rs = row_stride(qkv, stride_row);
    const int64_t ds = laststride(qkv);
    const int64_t pcs = multi ? pos->stride[0] : 0, pms = multi ? pos->stride[1] : 0;

    REF_PAR_FOR(M * stride_row)
    for (int64_t m = 0; m < M; ++m) {
        float p[8];
        const float div = rscale != 0.0f ? rscale : 1.0f;
        if (multi) {
            for (int64_t c = 0; c < n_comp; ++c) p[c] = rope_pos(pos, c * pcs + m * pms, div);
        } else {
            const float one = rope_pos(pos, offlin(pos, m), div);
            for (int c = 0; c < 8; ++c) p[c] = one;
        }
        const int64_t row = m * rs;
        /* Q heads then K heads; the V heads after them are left alone. */
        for (int64_t h = 0; h < n_head + n_head_kv; ++h)
            rope_head(qkv, row + h * head_dim * ds, ds, rot, mode, f, p, yarn.magnitude);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qk_norm_rope */
/* QK-norm and rotation in one op: an RMS norm over each head_dim-wide q and k head against its own
 * gain vector, then RoPE. `v_norm` 1 also RMS-norms each v head with unit gain and no rotation
 * (Gemma 4's weightless value norm); otherwise v heads are untouched. An absent gain is a unit
 * gain, without `wadd` (a video decoder's weightless q and k norms); `rotary_dim` 0 is the norm
 * alone, and then `positions` may be absent.
 *
 * The norm output is rounded to the tensor's dtype before the rotation, as the unfused pair does;
 * libr4d's r4d_qk_norm_rope_gate matches this. A fused kernel keeping f32 across the seam would be
 * half a ULP better and fail the comparison. */
int ref_qk_norm_rope(const RadArgs* a, RadStream) {
    if (a->n_t < 4 || !t_in(a, 0)) return RAD_E_INVAL;
    const RadTensor *qkv = &a->t[0], *pos = t_in(a, 1), *qw = t_in(a, 2), *kw = t_in(a, 3);

    const int64_t head_dim  = p_int(a, "head_dim", 0);
    const int64_t n_head    = p_int(a, "n_head", 0);
    const int64_t n_head_kv = p_int(a, "n_head_kv", 0);
    const float   theta     = p_f32(a, "theta", 10000.0f);
    const float   eps       = p_f32(a, "eps", 1e-6f);
    const float   rscale    = p_f32(a, "scale", 1.0f);
    const float   wadd      = p_f32(a, "wadd", 0.0f);
    const int     v_norm    = (int)p_int(a, "v_norm", 0);
    /* `gptj_blocks` is `blocks` with each section's pairs adjacent, (start + 2j, start + 2j + 1):
     * the complex-number form of Qwen-Image's 3-axis rotary. `imrope` is `rope`'s interleaved
     * M-RoPE over NeoX pairs, after the norm (Qwen3-VL text attention). `axial` is `rope`'s axial
     * form (a 3-axis video rotary over NeoX pairs), its section ladders applied with rank-1
     * positions too, every component then being that one position. */
    static const char* kModes[] = { "neox", "gptj", "blocks", "gptj_blocks", "imrope", "axial" };
    const int mode = p_enum(a, "mode", kModes, 6, 0);
    if (mode < 0) return RAD_E_UNSUPPORTED;
    const bool blocks = mode == 2 || mode == 3;
    const bool adjacent = mode == 3;
    const bool im = mode == 4;
    const bool axial = mode == 5;
    int64_t rot = p_int(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0 || rot < 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;
    if ((qw && numel(qw) < head_dim) || (kw && numel(kw) < head_dim)) return RAD_E_SHAPE;
    const bool turn = rot > 0;
    if (turn && !pos) return RAD_E_INVAL;

    /* Blocks: section c is `sec[c]` rotary pairs over its own run of 2 * sec[c] channels, paired
     * (start + j, start + sec[c] + j) and turned by position component comps[c] at
     * theta^(-j / sec[c]), i.e. a complete neox rotary of its own width. The sections fill the
     * rotated prefix exactly. Positions are [C, M], component-major. */
    int64_t sec[8] = {}, comp[8] = {};
    int n_sec = 0;
    if (turn && (im || axial)) {
        n_sec = rope_sections(a, sec);
        if (n_sec < (im ? 3 : 1)) return RAD_E_SHAPE;
    }
    if (turn && blocks) {
        n_sec = rope_sections(a, sec);
        if (n_sec <= 0) return RAD_E_SHAPE;
        int64_t total = 0;
        for (int c = 0; c < n_sec; ++c) { total += sec[c]; comp[c] = c; }
        if (2 * total != rot) return RAD_E_SHAPE;
        if (const char* cs = p_str(a, "comps", nullptr)) {
            int k = 0;
            for (const char* v = cs; *v && k < 8; ) {
                char* end = nullptr;
                const long long x = std::strtoll(v, &end, 10);
                if (end == v) break;
                comp[k++] = x;
                v = end;
            }
            if (k != n_sec) return RAD_E_SHAPE;
        }
    }
    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    const bool multi = pos && pos->rank >= 2;
    const int64_t n_comp = multi ? pos->shape[0] : 1;
    const int64_t M = !pos ? rows_of(qkv, stride_row) : multi ? pos->shape[1] : numel(pos);
    if (turn) {
        if (multi && !blocks && !im && !axial) return RAD_E_SHAPE;
        if (multi && im && n_comp < 3) return RAD_E_SHAPE;
        if (multi && axial && n_comp < n_sec) return RAD_E_SHAPE;
        for (int c = 0; blocks && c < n_sec; ++c)
            if (comp[c] < 0 || comp[c] >= n_comp) return RAD_E_SHAPE;
    }
    const int64_t pcs = multi ? pos->stride[0] : 0, pms = multi ? pos->stride[1] : 0;

    if (M <= 0 || numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    const int64_t rs = row_stride(qkv, stride_row);
    const int64_t ds = laststride(qkv);
    RopeFreq f;
    /* No `ladder` parameter here: the fused norm+rope measures by `rot`. */
    if (turn && im)
        RAD_REF_TRY(rope_freqs(ROPE_IMROPE, rot, rot, theta, sec, multi ? n_sec : 0, &f));
    else if (turn && axial)
        RAD_REF_TRY(rope_freqs(ROPE_AXIAL, rot, rot, theta, sec, n_sec, &f));
    else if (turn)
        RAD_REF_TRY(rope_freqs(blocks ? 0 : mode, rot, rot, theta, nullptr, 0, &f));
    const int pair_mode = im ? (int)ROPE_IMROPE : axial ? (int)ROPE_AXIAL : mode;

    /* RMS-norms one head in place, against gain `w` (+ wadd), or unit gain without one. */
    auto norm_head = [&](int64_t base, const RadTensor* w) {
        float ss = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
            const float x = ldt(qkv, base + d * ds);
            ss += x * x;
        }
        const float sc = 1.0f / std::sqrt(ss / (float)head_dim + eps);
        for (int64_t d = 0; d < head_dim; ++d) {
            const float x = ldt(qkv, base + d * ds) * sc;
            stt(qkv, base + d * ds, w ? x * (ldt(w, d * laststride(w)) + wadd) : x);
        }
    };

    REF_PAR_FOR(M * stride_row)
    for (int64_t m = 0; m < M; ++m) {
        float p[8];
        const float div = rscale != 0.0f ? rscale : 1.0f;
        if (turn && multi) {
            for (int64_t c = 0; c < n_comp && c < 8; ++c)
                p[c] = rope_pos(pos, c * pcs + m * pms, div);
        } else if (turn) {
            const float one = rope_pos(pos, offlin(pos, m), div);
            for (int c = 0; c < 8; ++c) p[c] = one;
        }
        const int64_t row = m * rs;
        for (int64_t h = 0; h < n_head + n_head_kv; ++h) {
            const int64_t base = row + h * head_dim * ds;
            norm_head(base, h < n_head ? qw : kw);
            if (!turn) continue;
            if (!blocks) {
                rope_head(qkv, base, ds, rot, pair_mode, f, p);
                continue;
            }
            int64_t start = 0;
            for (int c = 0; c < n_sec; ++c) {
                for (int64_t j = 0; j < sec[c]; ++j) {
                    const float inv = 1.0f / ref_pow(theta, (float)j / (float)sec[c]);
                    const float ang = p[comp[c]] * inv;
                    const float cs = ref_cos(ang), sn = ref_sin(ang);
                    const int64_t i0 = adjacent ? start + 2 * j : start + j;
                    const int64_t i1 = adjacent ? start + 2 * j + 1 : start + sec[c] + j;
                    const float x0 = ldt(qkv, base + i0 * ds), x1 = ldt(qkv, base + i1 * ds);
                    stt(qkv, base + i0 * ds, x0 * cs - x1 * sn);
                    stt(qkv, base + i1 * ds, x0 * sn + x1 * cs);
                }
                start += 2 * sec[c];
            }
        }
        for (int64_t h = 0; v_norm && h < n_head_kv; ++h)
            norm_head(row + (n_head + n_head_kv + h) * head_dim * ds, nullptr);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_local_rel */
/* Local attention with a relative-position term: Gemma 4's audio (USM conformer) attention.
 *
 * Query row i of a cu_seqlens segment sees the key rows j of the same segment with
 * 0 <= i - j <= left (`right` must be 0). Per head:
 *
 *   qf[d]  = q[i,d] * (head_dim^-0.5 / ln 2) * softplus(pds[d])
 *   logit  = sum_d qf[d] * k[j,d] * (ln(1 + e) / ln 2)  +  sum_d qf[d] * relk[i - j, d]
 *   logit  = softcap * tanh(logit / softcap)
 *
 * then a softmax over those keys and the weighted sum of v, all in f32 and rounded once. `relk`
 * is [left + 1, n_head * head_dim]: row d is the projected relative position for distance d.
 *
 * The upstream model computes this in 12-query blocks over 24-key windows with Transformer-XL's
 * rel_shift; keys outside 0 <= i - j <= left get a -1e9 logit and zero weight there, so this
 * sliding form is the same function. softplus(pds) is rounded to the weight's dtype before use,
 * as upstream evaluates it on the bf16 parameter. */
int ref_attn_local_rel(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *rk = &a->t[3];
    const RadTensor *pds = &a->t[4], *cu = &a->t[5], *o = &a->t[6];
    const int64_t D = p_int(a, "head_dim", 0), H = p_int(a, "n_head", 0);
    const int64_t L = p_int(a, "left", 0), R = p_int(a, "right", 0);
    const int64_t cdiv = p_int(a, "cu_div", 1);
    const float cap = p_f32(a, "softcap", 0.0f);
    if (D <= 0 || H <= 0 || L < 0 || cdiv <= 0 || !(cap > 0.0f)) return RAD_E_SHAPE;
    if (R != 0) return RAD_E_UNSUPPORTED;
    if (D > REF_HD_MAX || L + 1 > REF_HD_MAX) return RAD_E_UNSUPPORTED;
    const int64_t W = H * D;
    if (numel(pds) < D || numel(rk) < (L + 1) * W) return RAD_E_SHAPE;
    const int64_t S = numel(cu) - 1;
    if (S <= 0) return RAD_E_SHAPE;
    const int64_t Mq = rows_of(q, W), Mk = rows_of(k, W), Mv = rows_of(v, W), Mo = rows_of(o, W);
    if (Mq <= 0 || Mo <= 0 || cu_at(cu, S) <= cu_at(cu, 0)) return RAD_E_SHAPE;   /* zero rows */

    const float qs = (float)(1.0 / std::sqrt((double)D) / std::log(2.0));
    const float ks = (float)(std::log(1.0 + std::exp(1.0)) / std::log(2.0));
    float spd[REF_HD_MAX];
    for (int64_t d = 0; d < D; ++d)
        spd[d] = round_dt(pds->dtype, act_softplus(ldt(pds, d * laststride(pds)), 20.0f));

    for (int64_t s = 0; s < S; ++s) {
        const int64_t c0 = cu_at(cu, s);
        const int64_t c1 = cu_at(cu, s + 1);
        if (c0 % cdiv || c1 % cdiv || c1 < c0) return RAD_E_SHAPE;
        const int64_t t0 = c0 / cdiv, t1 = c1 / cdiv;
        if (t1 > Mq || t1 > Mk || t1 > Mv || t1 > Mo) return RAD_E_SHAPE;

        REF_PAR_FOR((t1 - t0) * H * D * (L + R + 1))
        for (int64_t ih = 0; ih < (t1 - t0) * H; ++ih) {
            const int64_t i = t0 + ih / H, h = ih % H;
            float qf[REF_HD_MAX], lg[REF_HD_MAX], acc[REF_HD_MAX];
            const int64_t qb = rowoff(q, i, W) + h * D * laststride(q);
            for (int64_t d = 0; d < D; ++d) qf[d] = ldt(q, qb + d * laststride(q)) * qs * spd[d];
            const int64_t j0 = i - L > t0 ? i - L : t0;
            float mx = -INFINITY;
            for (int64_t j = j0; j <= i; ++j) {
                const int64_t kb = rowoff(k, j, W) + h * D * laststride(k);
                const int64_t rb = rowoff(rk, i - j, W) + h * D * laststride(rk);
                float ac = 0.0f, bd = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    ac += qf[d] * (ldt(k, kb + d * laststride(k)) * ks);
                    bd += qf[d] * ldt(rk, rb + d * laststride(rk));
                }
                const float x = cap * ref_tanh((ac + bd) / cap);
                lg[j - j0] = x;
                mx = x > mx ? x : mx;
            }
            float sum = 0.0f;
            for (int64_t j = j0; j <= i; ++j) {
                lg[j - j0] = ref_exp(lg[j - j0] - mx);
                sum += lg[j - j0];
            }
            for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
            for (int64_t j = j0; j <= i; ++j) {
                const float p = lg[j - j0] / sum;
                const int64_t vb = rowoff(v, j, W) + h * D * laststride(v);
                for (int64_t d = 0; d < D; ++d) acc[d] += p * ldt(v, vb + d * laststride(v));
            }
            const int64_t ob = rowoff(o, i, W) + h * D * laststride(o);
            for (int64_t d = 0; d < D; ++d) stt(o, ob + d * laststride(o), acc[d]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rope_table */
/* The cos/sin plane a fused prologue reads instead of theta, written at each position's own row of
 * a [max_ctx, rot] table: cos in the first rot/2 values of a row, sin in the second.
 *
 * The angles must match `rope`'s to the last operation; any other construction gives subtly wrong
 * positions. Only rows named by `positions` are written; a negative position writes nothing. */
int ref_rope_table(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *pos = &a->t[0], *cs = &a->t[1];
    const int64_t rot      = p_int(a, "rot", 0);
    const int64_t head_dim = p_int(a, "head_dim", 0);
    const float   theta = p_f32(a, "theta", 10000.0f);
    const float   rscale = p_f32(a, "scale", 1.0f);
    if (rot < 2 || (rot & 1)) return RAD_E_SHAPE;
    int64_t den;
    RAD_REF_TRY(rope_ladder(a, rot, head_dim, &den));
    const int64_t M = numel(pos);
    if (M <= 0 || numel(cs) % rot) return RAD_E_SHAPE;
    const int64_t rows = numel(cs) / rot;
    const int64_t half = rot / 2;

    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn) || (yarn.factor != 0.0f && rscale != 1.0f)) return RAD_E_INVAL;
    REF_PAR_FOR(M * rot)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t pp = ld_int(pos->dtype, pos->data, offlin(pos, m));
        if (pp < 0 || pp >= rows) continue;
        const float p = (float)pp / (rscale != 0.0f ? rscale : 1.0f);
        for (int64_t i = 0; i < half; ++i) {
            const float inv = yarn.frequency(1.0f / ref_pow(theta, (float)(2 * i) / (float)den), (int)i);
            const float ang = p * inv;
            stt(cs, offlin(cs, pp * rot + i),        ref_cos(ang) * yarn.magnitude);
            stt(cs, offlin(cs, pp * rot + half + i), ref_sin(ang) * yarn.magnitude);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qsa_block_key */
/* The QSA indexer's compressed key (docs/OPS.md, docs/QSA.md): for each of M blocks,
 *
 *   pooled = mean of the block's `ratio` rows of k, accumulated in f32 and rounded to k's dtype
 *   normed = pooled * rsqrt(mean(pooled^2) + eps) * (wadd + w)
 *   out    = NeoX rotary of normed's first `rotary_dim` lanes at the block's position, the rest
 *            passed through, rounded once at the store
 *
 * The rounding after the mean is the upstream model's (`.float().mean(dim=1).to(dtype)`). The
 * block's position is `pos[b]` (a [>= M, 1 or 3] list holds a block's components side by side,
 * read under `mode` mrope/imrope and `sections`), or `pos0 + b*ratio` without one. `page[b]`, when
 * given, is the output row; a negative entry writes nothing. The rotary is `rope`'s ladder, with
 * YaRN's blend and magnitude when its coefficients are given. */
int ref_qsa_block_key(const RadArgs* a, RadStream) {
    const RadTensor* k = t_in(a, 0);
    const RadTensor* w = t_in(a, 1);
    const RadTensor* page = t_in(a, 2);
    const RadTensor* pos = t_in(a, 3);
    const RadTensor* out = t_in(a, 4);
    if (!k || !w || !out) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0);
    const int64_t rot = p_int(a, "rotary_dim", 0), pos0 = p_int(a, "pos0", -1);
    if (n <= 0 || n > REF_HD_MAX || ratio <= 0 || (pos0 < 0 && !pos)) return RAD_E_INVAL;
    if (rot < 0 || rot > n || (rot & 1)) return RAD_E_SHAPE;
    if (p_int(a, "block_size", ratio) != ratio) return RAD_E_SHAPE;
    if (k->rank != 2 || out->rank < 2 || out->shape[0] < 1) return RAD_E_SHAPE;
    const int64_t M = page ? numel(page) : out->shape[0];
    if (M < 1 || numel(out) / out->shape[0] != n) return RAD_E_SHAPE;
    if (k->shape[0] != M * ratio || k->shape[1] != n || numel(w) < n) return RAD_E_SHAPE;
    const bool pos2 = pos && pos->rank == 2;
    if (pos && (pos2 ? pos->shape[0] < M : numel(pos) < M)) return RAD_E_SHAPE;

    const float eps = p_f32(a, "eps", 1e-6f), wadd = p_f32(a, "wadd", 1.0f);
    const float theta = p_f32(a, "theta", 10000.0f), rscale = p_f32(a, "scale", 1.0f);
    static const char* kModes[] = { "neox", "mrope", "imrope" };
    const int mi = p_enum(a, "mode", kModes, 3, 0);
    if (mi < 0) return RAD_E_UNSUPPORTED;
    const int mode = mi == 1 ? ROPE_MROPE : mi == 2 ? ROPE_IMROPE : ROPE_NEOX;
    int64_t sec[8] = {};
    const int n_sec = rope_sections(a, sec);
    if (n_sec < 0 || n_sec > 3) return RAD_E_SHAPE;
    /* Multiple components need a mode, its sections and three columns of `pos`; otherwise every
     * frequency reads the block's single position, which reduces every mode to NeoX. */
    const bool mc = mode != ROPE_NEOX && n_sec > 0 && pos2 && pos->shape[1] >= 3;
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn) || (yarn.factor != 0.0f && (rot <= 0 || rscale != 1.0f)))
        return RAD_E_INVAL;
    RopeFreq f;
    RAD_REF_TRY(rope_freqs(mc ? mode : ROPE_NEOX, rot, rot, theta, sec, mc ? n_sec : 0, &f));
    for (int64_t i = 0; i < rot / 2; ++i) f.inv[i] = yarn.frequency(f.inv[i], (int)i);

    REF_PAR_FOR(M * n * ratio)
    for (int64_t b = 0; b < M; ++b) {
        const int64_t orow = page ? ld_int(page->dtype, page->data, offlin(page, b)) : b;
        if (orow < 0) continue;
        float x[REF_HD_MAX], y[REF_HD_MAX];
        double ss = 0.0;
        for (int64_t t = 0; t < n; ++t) {
            float acc = 0.0f;
            for (int64_t j = 0; j < ratio; ++j) acc += ldt(k, offlin(k, (b * ratio + j) * n + t));
            x[t] = round_dt(k->dtype, acc / (float)ratio);
            ss += (double)x[t] * x[t];
        }
        const float inv = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
        for (int64_t t = 0; t < n; ++t) y[t] = x[t] * inv * (wadd + ldt(w, offlin(w, t)));
        float p[3] = {};
        for (int c = 0; c < (mc ? 3 : 1); ++c) {
            const int64_t bp = !pos ? pos0 + b * ratio
                             : ld_int(pos->dtype, pos->data,
                                      offlin(pos, pos2 ? b * pos->shape[1] + c : b));
            p[c] = (float)bp / (rscale != 0.0f ? rscale : 1.0f);
        }
        const int64_t half = rot / 2;
        for (int64_t i = 0; i < half; ++i) {
            const float ang = p[f.comp[i]] * f.inv[i];
            const float c = ref_cos(ang) * yarn.magnitude, s = ref_sin(ang) * yarn.magnitude;
            const float x0 = y[i], x1 = y[half + i];
            y[i] = x0 * c - x1 * s;
            y[half + i] = x0 * s + x1 * c;
        }
        for (int64_t t = 0; t < n; ++t) stt(out, offlin(out, orow * n + t), y[t]);
    }
    return RAD_OK;
}
