/* ref_attn.cpp -- paged attention, dense attention, the KV write, and the two rotary forms.
 *
 * These are the ops most likely to be subtly wrong in any implementation, so they are written for
 * legibility over everything else: the mask is an explicit predicate per key, the softmax is two
 * plain passes with the max subtracted, and nothing is fused.
 *
 * ============================== THE PAGED LAYOUT ==============================
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]      K then V inside a slot
 *
 * This is the layout every kernel in the project has to match (libr4d's `R4DArgs::kv` names the
 * same one), so it is stated in ref_ops.h and obeyed here. A flat slot index
 * s decomposes as block = s / block_size, offset = s % block_size; `block_table[seq, i]` holds the
 * block id of the sequence's i-th block and `slot_mapping[token]` holds the flat slot a token's KV
 * goes to.
 *
 * ============================== THE MASK ==============================
 *
 * `seqused[s]` is the sequence's TOTAL key count INCLUDING this step's query tokens, so query row
 * i of a q_len-row step sits at absolute position seqused[s] - q_len + i. That is what makes one
 * kernel serve a decode (q_len 1), a speculative verify (q_len <= 64) and a prefill chunk
 * (q_len in the thousands) without the caller passing a second length.
 *
 *   causal = 1   key j is visible when j <= pos
 *   causal = 0   key j is visible when j <  seqused[s]
 *   window > 0   causal: additionally pos - j < window; non-causal: |pos - j| < window, which is
 *                the two-sided reading libr4d's h128 kernel documents.
 */
#include "ref_common.h"
/* `row_stride`: which axis carries a row of `cols` elements, at whatever rank the operand arrived
 * in. Every q/k/v/out below goes through it rather than through `rank >= 3 ? stride[rank-3] : cols`
 * -- the comment above kv_store shows where that shortcut is wrong. */
#include "ref_gemm.h"
#include "ref_ops.h"

using namespace ref;

namespace {

/* The largest head this reference accumulates in one stack frame. MLA's 576 is the widest head in
 * circulation and libr4d's widest is 256, so 1024 is headroom rather than a limit anyone will
 * meet; above it the op says RAD_E_UNSUPPORTED by name instead of overflowing a frame. */
enum { REF_HD_MAX = 1024 };

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

}  /* namespace */

/* ------------------------------------------------------------------ attn_paged */
int ref_attn_paged(const RadArgs* a, RadStream) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *kv = &a->t[1], *bt = &a->t[2], *su = &a->t[3], *o = &a->t[7];
    const RadTensor* kdes = t_in(a, 4);
    const RadTensor* vdes = t_in(a, 5);
    const RadTensor* cus  = t_in(a, 6);
    if (!q->data || !kv->data || !bt->data || !su->data || !o->data) return RAD_E_INVAL;

    const int64_t head_dim   = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa        = p_int(a, "gqa", 1);
    /* `block_size` is RAD_DERIVED on this op (docs/OPS.md), so a caller who did not pin it and
     * resolved to a kernel that declares no RAD_C_EQ for it -- ref's own row -- arrives here
     * without the key. The cache tensor says what the block size is; ref reads it off the operand,
     * which is this plugin's rule anyway, and cross-checks it below where the caller did name one. */
    const int64_t block_size = p_int(a, "block_size", kv->rank == 4 ? kv->shape[2] : 0);
    const int     causal     = (int)p_int(a, "causal", 1);
    const int64_t window     = p_int(a, "window", 0);
    if (head_dim <= 0 || gqa <= 0 || block_size <= 0) return RAD_E_SHAPE;
    if (head_dim > REF_HD_MAX) return RAD_E_UNSUPPORTED;

    const int64_t kv_heads = kv->rank == 4 ? kv->shape[1] : 0;
    if (kv_heads <= 0) return RAD_E_SHAPE;
    const int64_t q_heads = gqa * kv_heads;
    /* gqa and the cache's head count decide the query head count, so a rank-3 query that disagrees
     * is a caller bug that would otherwise stride through the wrong rows and return plausible
     * numbers. */
    if (q->rank >= 3 && q->shape[q->rank - 2] != q_heads) return RAD_E_SHAPE;
    const int64_t n_seq   = numel(su);
    const int64_t rows    = numel(q) / (q_heads * head_dim);
    if (n_seq <= 0 || rows <= 0) return RAD_E_SHAPE;
    /* RAGGED BATCHES. cu_seqlens is optional and absent means every sequence presents the same
     * query length, the uniform case. Present, it
     * is [n_seq + 1] with cu[0] == 0 and cu[n_seq] == rows, and sequence s owns query rows
     * [cu[s], cu[s+1]). A serving scheduler mixes one sequence prefill chunk with other single
     * decode rows in the same step, so the ragged case is the ordinary one. */
    if (!cus && rows % n_seq != 0) return RAD_E_SHAPE;
    if (cus && numel(cus) != n_seq + 1) return RAD_E_SHAPE;
    const int64_t q_len = cus ? 0 : rows / n_seq;
    if (numel(o) < rows * q_heads * head_dim) return RAD_E_SHAPE;
    if (kv->rank == 4 && kv->shape[2] != block_size) return RAD_E_SHAPE;
    if (kv->rank == 4 && kv->shape[3] < 2 * head_dim) return RAD_E_SHAPE;

    const int64_t max_blocks = bt->rank >= 2 ? bt->shape[bt->rank - 1] : numel(bt) / n_seq;
    const int64_t bt_rs = bt->rank >= 2 ? bt->stride[bt->rank - 2] : max_blocks;
    const int64_t bt_cs = bt->rank >= 2 ? bt->stride[bt->rank - 1] : 1;
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

    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < total; ++idx) {
        const int64_t h   = idx % q_heads;
        const int64_t tok = idx / q_heads;              /* the dense query row */
        /* Which sequence owns it, and where inside that sequence's query it sits. Linear rather than a
         * binary search: n_seq is a handful and this is the reference. */
        int64_t s = 0, qlen = q_len, i = 0;
        if (cus) {
            while (s + 1 < n_seq && ld_int(cus->dtype, cus->data, offlin(cus, s + 1)) <= tok) ++s;
            const int64_t beg = ld_int(cus->dtype, cus->data, offlin(cus, s));
            qlen = ld_int(cus->dtype, cus->data, offlin(cus, s + 1)) - beg;
            i = tok - beg;
        } else {
            s = tok / q_len;
            i = tok - s * q_len;
        }
        if (qlen <= 0) continue;
        const int64_t kh = h / gqa;

        const int64_t ctx = ld_int(su->dtype, su->data, offlin(su, s));
        const int64_t pos = ctx - qlen + i;
        const int64_t qb  = tok * q_rs + h * q_hs;
        const int64_t ob  = tok * o_rs + h * o_hs;

        const float kds = kdes ? ldt(kdes, s * kv_heads + kh) : 1.0f;
        const float vds = vdes ? ldt(vdes, s * kv_heads + kh) : 1.0f;

        /* Pass one: the maximum score, so the exponent below cannot overflow. */
        float mx = -INFINITY;
        for (int64_t j = 0; j < ctx; ++j) {
            if (causal ? (j > pos) : (j >= ctx)) continue;
            if (window > 0) {
                const int64_t d = pos - j;
                if (causal ? (d >= window) : (d >= window || d <= -window)) continue;
            }
            if (j / block_size >= max_blocks) break;   /* bounds first: the table is only this wide */
            const int64_t blk = ld_int(bt->dtype, bt->data, s * bt_rs + (j / block_size) * bt_cs);
            if (blk < 0) continue;
            const int64_t kb = blk * kvv.sb + kh * kvv.sh + (j % block_size) * kvv.so;
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d)
                dot += ldt(q, qb + d * q_ds) * (ldt(kv, kb + d * kvv.sd) * kds);
            dot *= scale;
            if (dot > mx) mx = dot;
        }

        float acc[REF_HD_MAX];
        for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;

        /* No visible key at all -- an empty sliding window on the first token of a sequence -- is a
         * zero row, not a NaN. A softmax over nothing has no defined answer and zero is the one
         * every kernel in this project produces. */
        if (mx == -INFINITY) {
            for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, 0.0f);
            continue;
        }

        /* Pass two: the scores again, exponentiated and accumulated in ascending key order. The
         * dot product is recomputed rather than kept, because keeping it would need a buffer the
         * size of the context and the reference has time it does not have memory. */
        float sum = 0.0f;
        for (int64_t j = 0; j < ctx; ++j) {
            if (causal ? (j > pos) : (j >= ctx)) continue;
            if (window > 0) {
                const int64_t d = pos - j;
                if (causal ? (d >= window) : (d >= window || d <= -window)) continue;
            }
            if (j / block_size >= max_blocks) break;   /* bounds first: the table is only this wide */
            const int64_t blk = ld_int(bt->dtype, bt->data, s * bt_rs + (j / block_size) * bt_cs);
            if (blk < 0) continue;
            const int64_t kb = blk * kvv.sb + kh * kvv.sh + (j % block_size) * kvv.so;
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d)
                dot += ldt(q, qb + d * q_ds) * (ldt(kv, kb + d * kvv.sd) * kds);
            const float w = ref_exp(dot * scale - mx);
            sum += w;
            for (int64_t d = 0; d < head_dim; ++d)
                acc[d] += w * (ldt(kv, kb + (head_dim + d) * kvv.sd) * vds);
        }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) stt(o, ob + d * o_ds, acc[d] * inv);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ attn_dense */
/* The vision tower's: nothing paged, nothing quantised, `cu_seqlens` bounding one image (or one
 * attention window) per entry so a whole batch is ONE launch rather than a per-segment loop and a
 * concatenate. Causal is a parameter because a dense decoder-only prefill wants the same kernel. */
int ref_attn_dense(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *v = &a->t[2], *cu = &a->t[3], *o = &a->t[4];

    const int64_t head_dim = p_int(a, "head_dim", q->rank >= 3 ? q->shape[q->rank - 1] : 0);
    const int64_t gqa      = p_int(a, "gqa", 1);
    const int     causal   = (int)p_int(a, "causal", 0);
    if (head_dim <= 0 || gqa <= 0) return RAD_E_SHAPE;
    if (head_dim > REF_HD_MAX) return RAD_E_UNSUPPORTED;

    /* [tokens, heads, head_dim], or [tokens, heads * head_dim] -- a column slice of a fused
     * projection, whose row pitch is the projection's. */
    const int64_t q_heads = q->rank >= 3 ? q->shape[q->rank - 2]
                          : (q->rank == 2 ? q->shape[1] / head_dim : 1);
    const int64_t kv_heads = q_heads / gqa;
    if (kv_heads <= 0 || q_heads % gqa != 0) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    if (n_seq <= 0) return RAD_E_SHAPE;
    const int64_t total_q = numel(q) / (q_heads * head_dim);
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

    #pragma omp parallel for schedule(static)
    for (int64_t sh = 0; sh < n_seq * q_heads; ++sh) {
        const int64_t s = sh / q_heads, h = sh % q_heads, kh = h / gqa;
        const int64_t b0 = ld_int(cu->dtype, cu->data, offlin(cu, s));
        const int64_t b1 = ld_int(cu->dtype, cu->data, offlin(cu, s + 1));
        for (int64_t i = b0; i < b1; ++i) {
            const int64_t qb = i * q_rs + h * q_hs, ob = i * o_rs + h * o_hs;
            const int64_t jend = causal ? (i + 1) : b1;
            float mx = -INFINITY;
            for (int64_t j = b0; j < jend; ++j) {
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
            for (int64_t j = b0; j < jend; ++j) {
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

/* ------------------------------------------------------------------ kv_store */
/* Writes this step's keys and values into the paged cache at `slot_mapping`. The cache operand is
 * INOUT and not OUT: this op touches only the mapped slots and every other slot in the pool must
 * survive it, which is a liveness fact the buffer planner has to know (docs/OPS.md draws it as an
 * output; that is the one role this plugin states differently, and deliberately).
 *
 * A NEGATIVE slot is skipped. That is how a padded batch row says "this token has no cache slot",
 * and it has to be a documented value rather than an accident: libr4d treats slot <= 0 as null
 * because vLLM reserves block 0, but a generic reference must not steal a valid slot, so ref skips
 * only the strictly negative.
 *
 * `k` AND `v` ARRIVE RANK 2 FROM THE ENGINE AND THEIR ROWS ARE NOT DENSE. An architecture that
 * de-interleaves the projection hands [tokens, kv_heads, head_dim]; one that does not hands a
 * COLUMN SLICE of the fused QKV plane -- [tokens, kv_heads * head_dim] whose row stride is the
 * whole projection's width, 2560 against 256 on a 16-head model with 2 kv heads. Deriving the row
 * stride from the shape instead of from the tensor puts token 0 in the right slot and every token
 * after it inside the previous token's QUERY rows, which is a cache that looks written and holds
 * the wrong keys from the second token on. `row_stride` asks the operand instead. */
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
    /* AN FP8 CACHE STORES k / k_scale: the scales are the descales the reader multiplies back by,
     * 1.0 unless the call names them. A wider cache stores the value itself. */
    const bool  fp8 = kv->dtype == RAD_F8E4M3 || kv->dtype == RAD_F8E5M2;
    const float ks = fp8 ? 1.0f / p_f32(a, "k_scale", 1.0f) : 1.0f;
    const float vs = fp8 ? 1.0f / p_f32(a, "v_scale", 1.0f) : 1.0f;

    for (int64_t t = 0; t < M; ++t) {
        const int64_t slot = ld_int(sm->dtype, sm->data, offlin(sm, t));
        if (slot >= 0 && n_blocks > 0 && slot / block_size >= n_blocks) return RAD_E_SHAPE;
    }

    #pragma omp parallel for schedule(static)
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

/* ------------------------------------------------------------------ query_fwht */
int ref_query_fwht(const RadArgs* a, RadStream) {
    if (!have(a, 1)) return RAD_E_INVAL;
    const RadTensor* q = &a->t[0];
    const int64_t total = numel(q);
    if (total <= 0 || total % 256 != 0) return RAD_E_SHAPE;
    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < total / 256; ++h) {
        float b[256];
        for (int i = 0; i < 256; ++i) b[i] = ldt(q, h * 256 + i);
        for (int len = 1; len < 256; len <<= 1) {
            for (int i = 0; i < 256; i += len << 1) {
                for (int j = 0; j < len; ++j) {
                    const float u = b[i + j], w = b[i + j + len];
                    b[i + j] = u + w;
                    b[i + j + len] = u - w;
                }
            }
        }
        constexpr float inv_16 = 1.0f / 16.0f;
        for (int i = 0; i < 256; ++i) stt(q, h * 256 + i, b[i] * inv_16);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rope */
namespace {

/* NeoX pairs element i with i + rotary_dim/2; GPT-J (the "interleaved" style) pairs 2i with 2i+1.
 * Which one a checkpoint wants is a property of the checkpoint, so it is a parameter and not a
 * guess -- and getting it wrong produces fluent text with a broken sense of position, which is the
 * hardest kind of wrong to notice (spec §12 makes the same point about token boundaries).
 *
 * THE MULTI-COMPONENT MODES pair like NeoX and differ in which position component drives each
 * frequency (i indexes the rot/2 frequencies):
 *   mrope   contiguous sections: frequencies [0, s0) take component 0, the next s1 component 1...
 *           (Qwen2-VL), all on one ladder theta^(-2i/rot).
 *   imrope  interleaved (Qwen3-VL): component 1 where i % 3 == 1 and i < 3*s1, component 2 where
 *           i % 3 == 2 and i < 3*s2, component 0 everywhere else; one ladder.
 *   axial   contiguous sections, EACH ON ITS OWN LADDER theta^(-2j/(2*s_c)) for its j-th
 *           frequency -- a 2-D rotary embedding over (row, column), the vision tower's.
 * With one position component a token -- a rank-1 `positions` -- every component is that one and
 * mrope and imrope ARE neox. */
enum { ROPE_NEOX = 0, ROPE_GPTJ = 1, ROPE_MROPE = 2, ROPE_IMROPE = 3, ROPE_AXIAL = 4 };

struct RopeFreq {
    int   comp[REF_HD_MAX / 2];
    float inv[REF_HD_MAX / 2];
};

/* Which component and inverse frequency each of the rot/2 frequencies uses. */
static int rope_freqs(int mode, int64_t rot, float theta, const int64_t* sec, int n_sec,
                      RopeFreq* f) {
    const int64_t half = rot / 2;
    if (half > REF_HD_MAX / 2) return RAD_E_UNSUPPORTED;
    int64_t start[8] = {}, total = 0;
    for (int c = 0; c < n_sec; ++c) { start[c] = total; total += sec[c]; }
    if ((mode == ROPE_MROPE || mode == ROPE_AXIAL) && n_sec > 0 && total != half) return RAD_E_SHAPE;
    for (int64_t i = 0; i < half; ++i) {
        int c = 0;
        float inv = 1.0f / std::pow(theta, (float)(2 * i) / (float)rot);
        if (n_sec > 0 && (mode == ROPE_MROPE || mode == ROPE_AXIAL)) {
            while (c + 1 < n_sec && i >= start[c + 1]) ++c;
            if (mode == ROPE_AXIAL) {
                const int64_t j = i - start[c];
                inv = 1.0f / std::pow(theta, (float)(2 * j) / (float)(2 * sec[c]));
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
                      const RopeFreq& f, const float* pos) {
    const int64_t half = rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float ang = pos[f.comp[i]] * f.inv[i];
        const float c = std::cos(ang), s = std::sin(ang);
        const int64_t i0 = (mode == ROPE_GPTJ) ? 2 * i : i;
        const int64_t i1 = (mode == ROPE_GPTJ) ? 2 * i + 1 : i + half;
        const float x0 = ldt(t, base + i0 * ds), x1 = ldt(t, base + i1 * ds);
        stt(t, base + i0 * ds, x0 * c - x1 * s);
        stt(t, base + i1 * ds, x0 * s + x1 * c);
    }
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

    int64_t sec[8] = {};
    const int n_sec = rope_sections(a, sec);
    if (n_sec < 0) return RAD_E_SHAPE;
    /* POSITIONS: [M] is one component a token; [C, M] is C components, component-major. */
    const bool multi = pos->rank >= 2;
    const int64_t n_comp = multi ? pos->shape[0] : 1;
    const int64_t M = multi ? pos->shape[1] : numel(pos);
    if (multi) {
        if (mode == ROPE_NEOX || mode == ROPE_GPTJ || n_sec <= 0) return RAD_E_SHAPE;
        const int64_t need = mode == ROPE_IMROPE ? 3 : n_sec;
        if (n_comp < need || n_comp > 8) return RAD_E_SHAPE;
    }
    RopeFreq f;
    RAD_REF_TRY(rope_freqs(mode, rot, theta, sec, multi ? n_sec : 0, &f));

    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    if (M <= 0 || numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    const int64_t rs = qkv->rank >= 2 ? qkv->stride[qkv->rank - 2] : stride_row;
    const int64_t ds = laststride(qkv);
    const int64_t pcs = multi ? pos->stride[0] : 0, pms = multi ? pos->stride[1] : 0;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        float p[8];
        const float div = rscale != 0.0f ? rscale : 1.0f;
        if (multi) {
            for (int64_t c = 0; c < n_comp; ++c)
                p[c] = (float)ld_int(pos->dtype, pos->data, c * pcs + m * pms) / div;
        } else {
            const float one = (float)ld_int(pos->dtype, pos->data, offlin(pos, m)) / div;
            for (int c = 0; c < 8; ++c) p[c] = one;
        }
        const int64_t row = m * rs;
        /* Q heads then K heads. The V heads that follow are left alone, which is the whole reason
         * this op takes the fused qkv rather than two tensors. */
        for (int64_t h = 0; h < n_head + n_head_kv; ++h)
            rope_head(qkv, row + h * head_dim * ds, ds, rot, mode, f, p);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ qk_norm_rope */
/* The QK-norm preamble and the rotation in one op: an RMS norm over each 128-wide (or whatever
 * head_dim is) q and k head against its own gain vector, then RoPE.
 *
 * THE NORM OUTPUT IS ROUNDED TO THE TENSOR'S DTYPE BEFORE THE ROTATION. That is not an accident of
 * writing it back in place -- it is what the unfused pair does, because the norm stores a tensor
 * and the rotation reads it, and libr4d's r4d_qk_norm_rope_gate matches the unfused reference for
 * exactly this reason. A fused kernel that kept f32 across the seam would be half a ULP better and
 * would fail a comparison against this. */
int ref_qk_norm_rope(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *qkv = &a->t[0], *pos = &a->t[1], *qw = &a->t[2], *kw = &a->t[3];

    const int64_t head_dim  = p_int(a, "head_dim", 0);
    const int64_t n_head    = p_int(a, "n_head", 0);
    const int64_t n_head_kv = p_int(a, "n_head_kv", 0);
    const float   theta     = p_f32(a, "theta", 10000.0f);
    const float   eps       = p_f32(a, "eps", 1e-6f);
    const float   rscale    = p_f32(a, "scale", 1.0f);
    const float   wadd      = p_f32(a, "wadd", 0.0f);
    static const char* kModes[] = { "neox", "gptj" };
    const int mode = p_enum(a, "mode", kModes, 2, ROPE_NEOX);
    int64_t rot = p_int(a, "rotary_dim", head_dim);
    if (head_dim <= 0 || n_head <= 0 || rot <= 0 || rot > head_dim || (rot & 1)) return RAD_E_SHAPE;
    if (numel(qw) < head_dim || numel(kw) < head_dim) return RAD_E_SHAPE;

    const int64_t stride_row = (n_head + 2 * n_head_kv) * head_dim;
    const int64_t M = numel(pos);
    if (M <= 0 || numel(qkv) < M * stride_row) return RAD_E_SHAPE;
    const int64_t rs = qkv->rank >= 2 ? qkv->stride[qkv->rank - 2] : stride_row;
    const int64_t ds = laststride(qkv);
    RopeFreq f;
    RAD_REF_TRY(rope_freqs(mode, rot, theta, nullptr, 0, &f));

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        float p[8];
        const float one = (float)ld_int(pos->dtype, pos->data, offlin(pos, m)) /
                          (rscale != 0.0f ? rscale : 1.0f);
        for (int c = 0; c < 8; ++c) p[c] = one;
        const int64_t row = m * rs;
        for (int64_t h = 0; h < n_head + n_head_kv; ++h) {
            const RadTensor* w = h < n_head ? qw : kw;
            const int64_t base = row + h * head_dim * ds, ws = laststride(w);
            float ss = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                float x = ldt(qkv, base + d * ds);
                ss += x * x;
            }
            const float sc = 1.0f / std::sqrt(ss / (float)head_dim + eps);
            for (int64_t d = 0; d < head_dim; ++d)
                stt(qkv, base + d * ds, ldt(qkv, base + d * ds) * sc * (ldt(w, d * ws) + wadd));
            rope_head(qkv, base, ds, rot, mode, f, p);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ rope_table */
/* The cos/sin plane a fused prologue reads instead of theta, written at each position's OWN row of
 * a [max_ctx, rot] table: `rot` values a row, cos for the first rot/2 and sin for the second.
 *
 * THE ANGLES ARE `rope`'s, TO THE LAST OPERATION, and that is the whole contract: a fused rotation
 * that read a table built any other way would be a different model, and the difference would be a
 * fraction of a ULP per element -- fluent text, subtly wrong positions, exactly the failure class
 * this file's rope_head comment is about. Only the rows `positions` names are written; a negative
 * position is a padded row and writes nothing. */
int ref_rope_table(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor *pos = &a->t[0], *cs = &a->t[1];
    const int64_t rot   = p_int(a, "rot", 0);
    const float   theta = p_f32(a, "theta", 10000.0f);
    const float   rscale = p_f32(a, "scale", 1.0f);
    if (rot < 2 || (rot & 1)) return RAD_E_SHAPE;
    const int64_t M = numel(pos);
    if (M <= 0 || numel(cs) % rot) return RAD_E_SHAPE;
    const int64_t rows = numel(cs) / rot;
    const int64_t half = rot / 2;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t pp = ld_int(pos->dtype, pos->data, offlin(pos, m));
        if (pp < 0 || pp >= rows) continue;
        const float p = (float)pp / (rscale != 0.0f ? rscale : 1.0f);
        for (int64_t i = 0; i < half; ++i) {
            const float inv = 1.0f / std::pow(theta, (float)(2 * i) / (float)rot);
            const float ang = p * inv;
            stt(cs, offlin(cs, pp * rot + i),        std::cos(ang));
            stt(cs, offlin(cs, pp * rot + half + i), std::sin(ang));
        }
    }
    return RAD_OK;
}
