/* ref_hc.cpp -- the GATED RESIDUAL, which is Qwen4-Exp's replacement for a pre-norm and a
 * residual add.
 *
 * ============================== WHAT IT IS ==============================
 *
 * The residual stream is `hc` times as wide as the model: hidden_size 2560 at hc_count 4 is a
 * 10240-wide stream carried through every layer, entered as four identical copies of the
 * embedding and collapsed once at the end. Every block reads a 2560 mix of the four and writes
 * its output back into all four, and BOTH the read's mix and the write's per-stream gain are
 * data-dependent -- which is what "gated" means here and what makes this a pair of ops rather
 * than a norm.
 *
 *   hc_read    N       = grouped_rmsnorm(h)              one rsqrt per 2560 stream, one gain of
 *                                                        10240 over all four, `1 + w`
 *              g       = sigmoid(W_up @ silu(W_down @ N / hc))
 *              x       = mean_over_streams(g * N)        the block's 2560 input
 *              inj[s]  = 2 * sigmoid(W_inj[s] . N / hc)  four scalars in (0, 2)
 *   hc_write   h[s]   += inj[s] * y                      for each stream s
 *
 * THE WRITE READS THE UNNORMED `h`, NOT `N`. The reference is `hyper_input + injection` where
 * `hyper_input` is the argument the read received, so the normalisation is consumed entirely by
 * the block input and the gate and never enters the stream. Writing back into `N` instead would
 * re-normalise the residual on every one of the 96 connections, which diverges within a few
 * layers and is the one mistake this pair can make that still produces fluent text.
 *
 * THE MIX IS A MEAN, NOT A SUM. `.mean(dim=-2)` over the hc dimension -- so the divide by hc
 * happens here AND, separately, inside both gate arguments (`/ self.hc_count` before the silu and
 * before the inject sigmoid). Three divisions by 4, all of them in the reference, none of them
 * derivable from the others.
 *
 * ============================== WHY THE QUANTISER IS IN THE READ ==============================
 *
 * `q` and `s` are OPTIONAL outputs. Present, the read also emits the fp8 block-scaled form of `x`
 * that the following projection's GEMM wants, which is what keeps a gated-residual model at the
 * same launch count as a pre-norm one: every launch costs a fixed floor whatever it computes, and a
 * model with 96 connections cannot afford a separate quantise op at each. Absent, `x` is bf16 and the
 * caller quantises it itself -- which is what the final mixer does, because its consumer is the
 * lm_head and that is bf16.
 *
 * IT QUANTISES WHAT WAS WRITTEN. `x` is narrowed to its own dtype first and the codes are taken
 * from the narrowed value, not from the f32 accumulator, because that is what a fused kernel
 * necessarily does and the two have to agree bit for bit -- the same epilogue-fold rule the GDN
 * out-projection's quantiser follows (libref/ref_gdn.cpp).
 *
 * `rq` and `rs` are the same argument one consumer further: had_quant_act_fp8 of the stored `x`,
 * for an expert plane stored with a 128-wide Hadamard, whose block would otherwise run that
 * quantiser as a launch of its own.
 */
#include "ref_common.h"
#include "ref_ops.h"

#include <cmath>

using namespace ref;

namespace {

/* A value narrowed to a tensor's own dtype and read back. The reference's arithmetic is f32
 * throughout, but the REFERENCE MODEL's is not: torch's grouped norm returns `.type_as(x)`, so the
 * two matrix products and the final multiply all read a bf16 `N`. Rounding once here is what makes
 * a fused kernel able to match this to a tolerance that means something. */
inline float narrow(uint32_t dtype, float v) {
    uint32_t buf[2] = {0, 0};
    st_dt(dtype, buf, 0, v);
    return ld_dt(dtype, buf, 0);
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }
inline float siluf(float v)    { return v * sigmoidf(v); }

}  /* namespace */

/* ------------------------------------------------------------------ hc_enter */
/* THE WIDE STREAM'S FIRST VALUE: hc identical copies of the embedding.
 *
 * It is one launch for the whole step and it exists rather than being folded into the embedding
 * lookup because the lookup's weight is [vocab, n] and its output width is n -- a wide output
 * would be a different op with a different weight shape. The four streams are identical here and
 * diverge at the first connection, because each block writes into all four with a DIFFERENT
 * per-branch gain. */
int ref_hc_enter(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* h = &a->t[1];
    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(h) < rows * hc * n) return RAD_E_SHAPE;
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), hb = rowoff(h, r, hc * n);
        const int64_t xs = laststride(x), hs = laststride(h);
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(x, xb + i * xs);
            for (int64_t c = 0; c < hc; ++c) stt(h, hb + (c * n + i) * hs, v);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_read */
int ref_hc_read(const RadArgs* a, RadStream) {
    /* NOT `have(a, 6)`: operand 4 is the OPTIONAL inject weight, and a prefix test would refuse
     * the final mixer -- the one connection that has none. The required ones are 0..3 and 5. */
    if (a->n_t < 6) return RAD_E_INVAL;
    if (!rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 5)) return RAD_E_INVAL;
    const RadTensor* h    = &a->t[0];
    const RadTensor* w    = &a->t[1];
    const RadTensor* down = &a->t[2];
    const RadTensor* up   = &a->t[3];
    const RadTensor* injw = t_in(a, 4);          /* absent on the final mixer */
    const RadTensor* x    = &a->t[5];
    const RadTensor* inj  = rad_arg_in(a, 6);    /* an OUT, but the optional test is the same */
    const RadTensor* q    = rad_arg_in(a, 7);
    const RadTensor* s    = rad_arg_in(a, 8);
    const RadTensor* rq   = rad_arg_in(a, 9);    /* the ROTATED fp8 form, for a w4a8h consumer */
    const RadTensor* rs   = rad_arg_in(a, 10);

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const int64_t lr = p_int(a, "lowrank", 0);
    const float  eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || hc <= 1 || lr <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(x) < rows * n || numel(w) < hn ||
        numel(down) < lr * hn || numel(up) < hn * lr) return RAD_E_SHAPE;
    if (injw && numel(injw) < hc * hn) return RAD_E_SHAPE;
    /* `inj` is what the paired hc_write consumes, so an inject WEIGHT with no inject OUTPUT is a
     * caller that computed four scalars and threw them away -- refused rather than ignored. */
    if (injw && (!inj || !inj->data || numel(inj) < rows * hc)) return RAD_E_SHAPE;

    int64_t g = 0;
    int64_t ngrp = 0;
    if (q && q->data) {
        g = p_int(a, "group", 0);
        if (g <= 0 || g > n) g = n;
        ngrp = (n + g - 1) / g;
        if (numel(q) < rows * n || !s || !s->data || numel(s) < rows * ngrp) return RAD_E_SHAPE;
    }
    /* The rotated form rotates each scale group whole, so the group must tile the row exactly and
     * be a power of two: a block that straddled the end of the row would mix real values with
     * padding, and no weight block was rotated that way. It is had_quant_act_fp8 of the stored `x`
     * -- the operand's consumer cannot tell which op wrote it. */
    int64_t rg = 0;
    if (rq) {
        rg = p_int(a, "group", 0);
        if (!pow2(rg) || n % rg != 0 || !rs || numel(rq) < rows * n || numel(rs) < rows * (n / rg))
            return RAD_E_SHAPE;
        if (rg > REF_HAD_MAX) return RAD_E_UNSUPPORTED;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(h, r, hn), xb = rowoff(x, r, n);
        const int64_t hs = laststride(h), xs = laststride(x), ws = laststride(w);

        /* ---- the grouped norm. One rsqrt per stream over its own n, then the full-width gain. */
        std::vector<float> N((size_t)hn);
        for (int64_t c = 0; c < hc; ++c) {
            double ss = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                const float v = ldt(h, hb + (c * n + i) * hs);
                ss += (double)v * (double)v;
            }
            const float sc = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
            for (int64_t i = 0; i < n; ++i) {
                const int64_t j = c * n + i;
                N[(size_t)j] = narrow(h->dtype,
                                      ldt(h, hb + j * hs) * sc * (ldt(w, j * ws) + wadd));
            }
        }

        /* ---- the read gate: 10240 -> lowrank -> 10240, silu between and sigmoid after. */
        std::vector<float> d((size_t)lr);
        for (int64_t j = 0; j < lr; ++j) {
            float acc = 0.0f;
            const int64_t db = rowoff(down, j, hn);
            const int64_t ds = laststride(down);
            for (int64_t i = 0; i < hn; ++i) acc += N[(size_t)i] * ldt(down, db + i * ds);
            d[(size_t)j] = narrow(down->dtype, siluf(acc / (float)hc));
        }

        /* ---- the mix, and the mean over the streams. */
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t c = 0; c < hc; ++c) {
                const int64_t row = c * n + i;
                const int64_t ub = rowoff(up, row, lr);
                const int64_t us = laststride(up);
                float gv = 0.0f;
                for (int64_t j = 0; j < lr; ++j) gv += d[(size_t)j] * ldt(up, ub + j * us);
                acc += narrow(up->dtype, sigmoidf(gv)) * N[(size_t)row];
            }
            stt(x, xb + i * xs, acc / (float)hc);
        }

        /* ---- the per-branch write gates. */
        if (injw && inj) {
            const int64_t ib = rowoff(inj, r, hc), is = laststride(inj);
            for (int64_t c = 0; c < hc; ++c) {
                float acc = 0.0f;
                const int64_t jb = rowoff(injw, c, hn), js = laststride(injw);
                for (int64_t i = 0; i < hn; ++i) acc += N[(size_t)i] * ldt(injw, jb + i * js);
                stt(inj, ib + c * is, 2.0f * sigmoidf(acc / (float)hc));
            }
        }

        /* ---- the optional fp8 form of x, read back from what was stored. */
        if (q && q->data) {
            const int64_t qb = rowoff(q, r, n), qs = laststride(q);
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
                float amax = 0.0f;
                for (int64_t i = k0; i < k1; ++i) {
                    const float m = std::fabs(ldt(x, xb + i * xs));
                    if (m > amax) amax = m;
                }
                const float sc  = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
                const float invs = 1.0f / sc;
                for (int64_t i = k0; i < k1; ++i) {
                    float v = ldt(x, xb + i * xs) * invs;
                    if (v >  448.0f) v =  448.0f;
                    if (v < -448.0f) v = -448.0f;
                    st_dt(q->dtype, q->data, qb + i * qs, v);
                }
                stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
            }
        }

        /* ---- the optional ROTATED fp8 form of x, from the same stored values. */
        if (rq) {
            float buf[REF_HAD_MAX];
            const int64_t qb = rowoff(rq, r, n), qs = laststride(rq), rgrp = n / rg;
            for (int64_t gi = 0; gi < rgrp; ++gi) {
                const int64_t k0 = gi * rg;
                for (int64_t i = 0; i < rg; ++i) buf[i] = ldt(x, xb + (k0 + i) * xs);
                fwht(buf, rg);
                float amax = 0.0f;
                for (int64_t i = 0; i < rg; ++i) {
                    const float m = std::fabs(buf[i]);
                    if (m > amax) amax = m;
                }
                const float sc   = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
                const float invs = 1.0f / sc;
                for (int64_t i = 0; i < rg; ++i) {
                    float v = buf[i] * invs;
                    if (v >  448.0f) v =  448.0f;
                    if (v < -448.0f) v = -448.0f;
                    st_dt(rq->dtype, rq->data, qb + (k0 + i) * qs, v);
                }
                stt(rs, rowoff(rs, r, rgrp) + gi * laststride(rs), sc);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_write */
/* `h[s] += inj[s] * y` for each of the hc streams. In place: `h` is the only output and it is the
 * stream the paired read was handed, unchanged by it. */
int ref_hc_write(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor* y   = &a->t[0];
    const RadTensor* inj = &a->t[1];
    const RadTensor* h   = &a->t[2];

    const int64_t n  = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(h) < rows * hn || numel(inj) < rows * hc) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(y, r, n), hb = rowoff(h, r, hn), ib = rowoff(inj, r, hc);
        const int64_t ys = laststride(y), hs = laststride(h), is = laststride(inj);
        for (int64_t c = 0; c < hc; ++c) {
            const float gain = ldt(inj, ib + c * is);
            for (int64_t i = 0; i < n; ++i) {
                const int64_t j = hb + (c * n + i) * hs;
                stt(h, j, ldt(h, j) + gain * ldt(y, yb + i * ys));
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_mean */
/* THE CAPTURE'S OWN REDUCTION, and nothing else: x[m,i] = mean over the hc streams of
 * h[m, c*n+i]. It is what a DFlash2 tap records -- the reference model's
 * `hidden_states.unflatten(-1, (hc, n)).mean(dim=-2)` -- and NOT the read half of a gated residual:
 * no RMS, no gate, no weight, no read of a block input. A caller that substituted `hc_read` would
 * divide by a different number, gate the streams and normalise them, and every tap would differ.
 *
 * THE ARITHMETIC IS f32 AND THE STORE IS THE ONLY ROUNDING. Sum over c in f32, divide by hc, store
 * the tensor's own dtype once. `* (1/hc)` and `/ hc` are the same value whenever hc is a power of
 * two (4 here), and both are exact there; the division is written out because it is also right for
 * an hc that is not.
 *
 * THE ROW PITCH IS HONOURED ON BOTH SIDES. `h` arrives as [M, hc*n] and `x` as a column view of a
 * wider tap plane -- width n at column t*n with the plane's n_taps*n kept as the stride -- so the
 * output row offset comes from x's own stride and adjacent taps are never touched. The two are
 * refused if they overlap: this op reads every stream before it stores any element of that row,
 * but a caller aliasing x into h would still be describing one buffer as two tensors, which is a
 * bug worth naming rather than a layout worth supporting. */
int ref_hc_mean(const RadArgs* a, RadStream) {
    if (!have(a, 2)) return RAD_E_INVAL;
    const RadTensor* h = &a->t[0];
    const RadTensor* x = &a->t[1];

    const int64_t n  = p_int(a, "N", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 0) return RAD_E_SHAPE;
    const char* dt = p_str(a, "dtype", nullptr);
    if (dt && std::strcmp(dt, "bf16") != 0) return RAD_E_DTYPE;
    /* ONE dtype and no cast: the whole op is bf16 in and bf16 out, so anything else is a caller
     * that meant a different op rather than one this one should widen for. */
    if (h->dtype != RAD_BF16 || x->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (h->rank != 2 || x->rank != 2) return RAD_E_SHAPE;
    if (!h->data || !x->data) return RAD_E_INVAL;

    const int64_t hn = hc * n;
    if (h->shape[1] != hn || x->shape[1] != n) return RAD_E_SHAPE;
    const int64_t rows = band_rows(h->shape[0], p_int(a, "M", 0));
    if (rows <= 0 || x->shape[0] != rows) return RAD_E_SHAPE;

    /* Disjoint outputs: see the header. A byte span is computed from each tensor's own extents and
     * strides, so a column view is measured over the columns it actually writes. */
    {
        auto span = [](const RadTensor* t) {
            uint64_t e = 1;
            for (uint32_t i = 0; i < t->rank; ++i)
                e += (uint64_t)(t->shape[i] - 1) * (uint64_t)t->stride[i];
            return e * 2u;
        };
        const uintptr_t hp = (uintptr_t)h->data, xp = (uintptr_t)x->data;
        const uint64_t hbytes = span(h), xbytes = span(x);
        if (hp < xp + xbytes && xp < hp + hbytes) return RAD_E_INVAL;
    }

    const int64_t hs = h->stride[1];   /* h's column step, in elements */
    const int64_t xs = x->stride[1];   /* x's column step; its row pitch is stride[0] */
    if (hs < 1 || xs < 1) return RAD_E_SHAPE;
    /* A ROW PITCH NARROWER THAN THE ROW IS NOT A VIEW. The stride of dim 0 is the tap plane's
     * width for a column slice, so it is at least the slice's own width; anything less makes
     * consecutive rows overlap, and a store to row m would land in row m+1's data. With one row
     * the pitch is never used, so it may be anything non-negative. */
    if (h->stride[0] < 0 || x->stride[0] < 0) return RAD_E_SHAPE;
    if (rows > 1 && (h->stride[0] < hn || x->stride[0] < n)) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hrow = r * h->stride[0], xrow = r * x->stride[0];
        const uint16_t* hsrc = (const uint16_t*)h->data;
        uint16_t*       xdst = (uint16_t*)x->data;
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t c = 0; c < hc; ++c) acc += bf16_to_f32(hsrc[hrow + (c * n + i) * hs]);
            xdst[xrow + i * xs] = f32_to_bf16(acc / (float)hc);
        }
    }
    return RAD_OK;
}

/* ================================================================== mtp_enter
 *
 * THE MULTI-TOKEN-PREDICTION HEAD'S ENTRY, for a model whose residual stream is `hc` wide.
 *
 * A pre-norm MTP head (Qwen3.5, Qwen3-Next, DeepSeek-V3) joins the trunk's hidden state and the
 * next token's embedding with ONE `fc` over their concatenation, and arch/common/rad_block_mtp_fp8.h
 * runs that as two rmsnorms into the two halves of one buffer and a single gemm_nt. Qwen4-Exp's
 * head cannot: its hidden state is the WIDE stream and the checkpoint carries TWO square matrices,
 *
 *     mtp.fc_hidden.weight        [n, n]        mtp.pre_fc_norm_hidden.weight       [hc * n]
 *     mtp.fc_embedding.weight     [n, n]        mtp.pre_fc_norm_embedding.weight    [n]
 *
 * so the hidden half is `fc_h` applied to EACH of the hc sub-streams -- the reference's own
 * `.unflatten(-1, (hc_count, hidden_size))` idiom, which is how every other 10240-wide tensor in
 * this architecture is handled -- and the embedding half is one 2560-wide product added into all
 * hc of them, which is the same broadcast the trunk enters the stream with (`hidden_states.repeat`).
 *
 *     rs_e      = rsqrt(mean(e^2) + eps)
 *     ne[j]     = e[j] * rs_e * (w_e[j] + wadd)
 *     fe[i]     = sum_j fc_e[i][j] * ne[j]
 *     rs[k]     = rsqrt(mean(h[k*n .. k*n+n)^2) + eps)          -- `ngroup` == hc
 *     nh[k][j]  = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
 *     x[k*n+i]  = sum_j fc_h[i][j] * nh[k][j] + fe[i]
 *
 * IT IS ONE OP AND NOT SIX because every intermediate above is a DIFFERENT VIEW of the same bytes
 * and this ABI has no reinterpreting view: `nh` is [M*hc, n] to the GEMM and [M, hc*n] to the norm,
 * and core/runtime/issue.cpp narrows dim 0 or the last dimension with every stride left alone --
 * which is the right rule and makes a row-pitch change inexpressible. Spelling the entry as one op
 * is what lets the wide buffers stay wide. It is also six launches off a draft round, and a draft
 * round is the latency this head exists to hide.
 *
 * `ngroup` IS THE ONE THING THE CHECKPOINT DOES NOT SAY. A [hc * n] gain fits both a grouped norm
 * (hc rsqrts, one per sub-stream -- what Qwen4ExpTextGatedResidual does at every one of its 97
 * connections) and a flat one (a single rsqrt over the whole wide row). Both are self-consistent
 * and no per-op oracle can tell them apart, exactly as rad_block_mtp_fp8.h's concat order cannot
 * be told apart -- and it has the same cheap empirical answer, the draft acceptance rate. `ngroup`
 * carries the choice so it is asked once, at declare, instead of being compiled in.
 */
int ref_mtp_enter(const RadArgs* a, RadStream) {
    if (!have(a, 7)) return RAD_E_INVAL;
    const RadTensor* h    = &a->t[0];
    const RadTensor* e    = &a->t[1];
    const RadTensor* w_h  = &a->t[2];
    const RadTensor* w_e  = &a->t[3];
    const RadTensor* fc_h = &a->t[4];
    const RadTensor* fc_e = &a->t[5];
    const RadTensor* x    = &a->t[6];

    const int64_t n   = p_int(a, "n", 0);
    const int64_t hc  = p_int(a, "hc", 0);
    const float  eps  = p_f32(a, "eps", 1e-6f);
    const float  wadd = p_f32(a, "wadd", 0.0f);
    int64_t ngroup    = p_int(a, "ngroup", hc);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    if (ngroup != 1 && ngroup != hc) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(e) < rows * n || numel(x) < rows * hn) return RAD_E_SHAPE;
    if (numel(w_h) < hn || numel(w_e) < n) return RAD_E_SHAPE;
    if (numel(fc_h) < n * n || numel(fc_e) < n * n) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(h, r, hn), eb = rowoff(e, r, n), xb = rowoff(x, r, hn);
        const int64_t hs = laststride(h), es = laststride(e), xs = laststride(x);
        const int64_t whs = laststride(w_h), wes = laststride(w_e);

        /* ---- the embedding half: one norm and one 2560-wide product, shared by every stream. */
        double ss = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const float v = ldt(e, eb + i * es);
            ss += (double)v * (double)v;
        }
        const float rse = 1.0f / std::sqrt((float)(ss / (double)n) + eps);
        std::vector<float> ne((size_t)n);
        for (int64_t i = 0; i < n; ++i)
            ne[(size_t)i] = narrow(e->dtype, ldt(e, eb + i * es) * rse * (ldt(w_e, i * wes) + wadd));

        std::vector<float> fe((size_t)n);
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            const int64_t fb = rowoff(fc_e, i, n);
            const int64_t fs = laststride(fc_e);
            for (int64_t j = 0; j < n; ++j) acc += ne[(size_t)j] * ldt(fc_e, fb + j * fs);
            fe[(size_t)i] = acc;
        }

        /* ---- the hidden half: the grouped norm, then the SAME square matrix over each stream.
         * The flat reading is one rsqrt over the whole wide row; `ngroup` says which. */
        float flat = 0.0f;
        if (ngroup == 1) {
            double sw = 0.0;
            for (int64_t j = 0; j < hn; ++j) {
                const float v = ldt(h, hb + j * hs);
                sw += (double)v * (double)v;
            }
            flat = 1.0f / std::sqrt((float)(sw / (double)hn) + eps);
        }

        std::vector<float> nh((size_t)n);
        for (int64_t c = 0; c < hc; ++c) {
            float sc = flat;
            if (ngroup != 1) {
                double sk = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    const float v = ldt(h, hb + (c * n + i) * hs);
                    sk += (double)v * (double)v;
                }
                sc = 1.0f / std::sqrt((float)(sk / (double)n) + eps);
            }
            for (int64_t i = 0; i < n; ++i) {
                const int64_t j = c * n + i;
                nh[(size_t)i] = narrow(h->dtype,
                                       ldt(h, hb + j * hs) * sc * (ldt(w_h, j * whs) + wadd));
            }
            for (int64_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                const int64_t fb = rowoff(fc_h, i, n);
                const int64_t fs = laststride(fc_h);
                for (int64_t j = 0; j < n; ++j) acc += nh[(size_t)j] * ldt(fc_h, fb + j * fs);
                stt(x, xb + (c * n + i) * xs, acc + fe[(size_t)i]);
            }
        }
    }
    return RAD_OK;
}
