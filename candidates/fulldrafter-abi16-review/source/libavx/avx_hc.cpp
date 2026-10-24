/* avx_hc.cpp -- the GATED RESIDUAL pair and the MTP head's entry.
 *
 * ============================== WHAT THE GATED RESIDUAL IS ==============================
 *
 * Qwen4-Exp's replacement for a pre-norm and a residual add. The residual stream is `hc` times as
 * wide as the model -- hidden 2560 at hc 4 is a 10240-wide stream carried through every layer,
 * entered as hc identical copies of the embedding and collapsed once at the end. Every block reads
 * a 2560 mix of the four and writes its output back into all four, and BOTH the read's mix and the
 * write's per-stream gain are data-dependent, which is what "gated" means here.
 *
 *   hc_read    N       = grouped_rmsnorm(h)              one rsqrt per n-wide stream, one gain of
 *                                                        hc*n over all of them, `1 + w`
 *              g       = sigmoid(W_up @ silu(W_down @ N / hc))
 *              x       = mean_over_streams(g * N)
 *              inj[s]  = 2 * sigmoid(W_inj[s] . N / hc)
 *   hc_write   h[s]   += inj[s] * y
 *
 * THREE FACTS THAT ARE NOT DERIVABLE FROM THE OTHERS, and a second implementation has to match all
 * three (docs/OPS.md states them and they are restated here because each is a silent failure):
 *
 *   THE WRITE READS THE UNNORMED `h`, NOT `N`. The normalisation is consumed by the block input
 *   and the gates and never enters the stream. Writing into `N` instead re-normalises the residual
 *   at every one of 96 connections, diverges within a few layers, and still produces fluent text.
 *
 *   THE MIX IS A MEAN, NOT A SUM -- `.mean(dim=-2)` over the hc dimension.
 *
 *   THERE IS A SEPARATE `/ hc` INSIDE EACH GATE ARGUMENT, before the silu and before the inject
 *   sigmoid. Three divisions by hc in total, none implied by the others.
 *
 * ============================== THE ROUNDING IS PART OF THE OP ==============================
 *
 * The arithmetic here is f32, but the REFERENCE MODEL's is not: torch's grouped norm returns
 * `.type_as(x)`, so both matrix products and the final multiply read a bf16 `N`. libref rounds
 * once per element for that reason and so does this -- through the row converter rather than
 * element by element, which is the same value and one pass. The same applies to the silu output,
 * the sigmoid gate, and to `x` before the optional fp8 quantiser reads it back.
 *
 * ============================== THE WEIGHTS ARE WIDENED ONCE PER LAUNCH ==============================
 *
 * This is the structural change from libref, and it is worth more here than vectorisation. Every
 * product below contracts a WEIGHT against a per-token vector -- `down` is [lowrank, hc*n], `up` is
 * [hc*n, lowrank], `inject_w` is [hc, hc*n] -- and libref reads those weights through `ldt` inside
 * the token loop, so a 2048-row prefill chunk converts the same 10240-wide weight row 2048 times.
 * Widening all three into one scratch block before the token loop makes every dot product a
 * straight f32 vector reduction and converts each weight element exactly once.
 *
 * ONE scratch_raw CALL FOR ALL OF THEM, because the arena grows: a second call at a larger size
 * frees the first block, and every worker thread would then be reading a dangling weight.
 */
#include "avx_vec.h"
#include "avx_common.h"

using namespace avx;

#include "avx_had.h"

/* A single value narrowed to a tensor's dtype and read back -- the scalar form of the rounding
 * rule above, for the handful of places where it lands on one number rather than a row. */
static inline float narrow1(uint32_t dt, float v) {
    uint32_t buf[2] = { 0, 0 };
    rad_store_f32(buf, dt, 0, v);
    return rad_load_f32(buf, dt, 0);
}

/* A weight plane widened into f32, row-major and contiguous. `rows` x `cols` of the operand read
 * the way libref reads it (rowoff / laststride), so a strided or rank-3 weight lands correctly. */
static void widen_plane(const RadTensor* t, int64_t rows, int64_t cols, float* dst) {
    const int64_t st = laststride(t);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t b = rowoff(t, r, cols);
        if (st == 1) row_to_f32(t->dtype, byte_at(t->data, t->dtype, b), dst + r * cols, cols);
        else for (int64_t i = 0; i < cols; ++i) dst[r * cols + i] = ldt(t, b + i * st);
    }
}

static inline float dotf(const float* a, const float* b, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t i = 0;
    for (; i + 4 * VF_N <= n; i += 4 * VF_N) {
        a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
        a1 = vf_fma(vf_loadu(a + i + VF_N), vf_loadu(b + i + VF_N), a1);
        a2 = vf_fma(vf_loadu(a + i + 2 * VF_N), vf_loadu(b + i + 2 * VF_N), a2);
        a3 = vf_fma(vf_loadu(a + i + 3 * VF_N), vf_loadu(b + i + 3 * VF_N), a3);
    }
    for (; i + VF_N <= n; i += VF_N) a0 = vf_fma(vf_loadu(a + i), vf_loadu(b + i), a0);
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* THE SUM OF SQUARES IS ACCUMULATED IN f64 HERE AND IN f32 IN avx_norm.cpp, and that is libref's
 * split reproduced rather than an inconsistency. `rmsnorm` accumulates in f32 because it is the
 * op every fast kernel is checked against and an oracle tighter than the thing it certifies turns
 * a tolerance into a guess; the gated residual's norm is an INTERNAL step of a fused op that no
 * kernel computes separately, so libref takes the tighter accumulation there and this matches it.
 * Not vectorised: it is one reduction per stream against two matrix products that dominate, and a
 * vector f64 path would be a fifth numeric convention in the file. */
static float rsqrt_meansq_f64(const float* v, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double)v[i] * (double)v[i];
    return 1.0f / std::sqrt((float)(ss / (double)n) + eps);
}

/* ------------------------------------------------------------------ hc_enter */
/* THE WIDE STREAM'S FIRST VALUE: hc identical copies of the embedding. One launch for the whole
 * step. It exists rather than being folded into the embedding lookup because the lookup's weight
 * is [vocab, n] and its output width is n -- a wide output would be a different op with a
 * different weight shape. The streams are identical here and diverge at the first connection,
 * because each block writes into all of them with a different per-branch gain. */
AVX_KERNEL(hc_enter) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor* x = &a->t[0];
    const RadTensor* h = &a->t[1];
    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t rows = numel(x) / n;
    if (rows <= 0 || numel(h) < rows * hc * n) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * hc * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t xb = rowoff(x, r, n), hb = rowoff(h, r, hc * n);
        const int64_t hs = laststride(h);
        float* buf = scratch_f32((size_t)n);
        const float* v = in_row(x, xb, n, buf);
        /* The copy is per STREAM and contiguous, where libref's inner loop is over i with the
         * stream inside it -- which writes with a stride of n and defeats the store buffer. Same
         * bytes, one stream at a time. */
        if (hs == 1) for (int64_t c = 0; c < hc; ++c) out_row(h, hb + c * n, n, v);
        else for (int64_t c = 0; c < hc; ++c)
                 for (int64_t i = 0; i < n; ++i) stt(h, hb + (c * n + i) * hs, v[i]);
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_read */
AVX_KERNEL(hc_read) {
    /* NOT `rad_args_have(a, 6)`: operand 4 is the OPTIONAL inject weight, and a prefix test would
     * refuse the final mixer -- the one connection that has none. The required ones are 0..3 and 5. */
    if (a->n_t < 6) return RAD_E_INVAL;
    if (!rad_arg_in(a, 0) || !rad_arg_in(a, 1) || !rad_arg_in(a, 2) || !rad_arg_in(a, 3) ||
        !rad_arg_in(a, 5)) return RAD_E_INVAL;
    const RadTensor* h = &a->t[0];
    const RadTensor* w = &a->t[1];
    const RadTensor* down = &a->t[2];
    const RadTensor* up = &a->t[3];
    const RadTensor* injw = rad_arg_in(a, 4);      /* absent on the final mixer */
    const RadTensor* x = &a->t[5];
    const RadTensor* inj = rad_arg_in(a, 6);
    const RadTensor* q = rad_arg_in(a, 7);
    const RadTensor* s = rad_arg_in(a, 8);
    const RadTensor* rq = rad_arg_in(a, 9);     /* the ROTATED fp8 form, for a w4a8h consumer */
    const RadTensor* rs = rad_arg_in(a, 10);

    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const int64_t lr = p_int(a, "lowrank", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    if (n <= 0 || hc <= 1 || lr <= 0) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(x) < rows * n || numel(w) < hn ||
        numel(down) < lr * hn || numel(up) < hn * lr) return RAD_E_SHAPE;
    if (injw && numel(injw) < hc * hn) return RAD_E_SHAPE;
    /* `inj` is what the paired hc_write consumes, so an inject WEIGHT with no inject OUTPUT is a
     * caller that computed hc scalars and threw them away -- refused rather than ignored. */
    if (injw && (!inj || !inj->data || numel(inj) < rows * hc)) return RAD_E_SHAPE;

    int64_t g = 0, ngrp = 0;
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
    if (rq && rq->data) {
        rg = p_int(a, "group", 0);
        if (!pow2(rg) || n % rg != 0 || !rs || !rs->data || numel(rq) < rows * n ||
            numel(rs) < rows * (n / rg))
            return RAD_E_SHAPE;
        if (rg > AVX_HAD_MAX) return RAD_E_UNSUPPORTED;
    }

    /* The gain and all three weight planes, widened once. See the header on why this is one call. */
    auto pad = [](size_t v) { return (v + 15) & ~(size_t)15; };
    const size_t o_gain = 0;
    const size_t o_down = pad(o_gain + (size_t)hn);
    const size_t o_up   = pad(o_down + (size_t)lr * (size_t)hn);
    const size_t o_inj  = pad(o_up + (size_t)hn * (size_t)lr);
    const size_t total  = o_inj + (injw ? (size_t)hc * (size_t)hn : 0) + 16;
    float* blk = (float*)scratch_raw(total * sizeof(float));
    float* gain = blk + o_gain;
    float* dn = blk + o_down;
    float* upw = blk + o_up;
    float* iw = blk + o_inj;
    {
        const int64_t ws = laststride(w);
        if (ws == 1) row_to_f32(w->dtype, w->data, gain, hn);
        else for (int64_t i = 0; i < hn; ++i) gain[i] = ldt(w, i * ws);
        if (wadd != 0.0f) for (int64_t i = 0; i < hn; ++i) gain[i] += wadd;
    }
    widen_plane(down, lr, hn, dn);
    widen_plane(up, hn, lr, upw);
    if (injw) widen_plane(injw, hc, hn, iw);

    AVX_PARALLEL_FOR_IF(rows >= 2)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = rowoff(h, r, hn), xb = rowoff(x, r, n);
        const int64_t xs = laststride(x);
        float* N = scratch_f32((size_t)hn);
        float* tmp = scratch_f32_b((size_t)hn);
        float* dv = scratch_f32_c((size_t)(lr + n) + 16);
        float* xv = dv + lr + 8;

        /* ---- the grouped norm: one rsqrt per stream over its own n, then the full-width gain,
         * then a single round trip through h's dtype for the whole wide row. */
        const float* hv = in_row(h, hb, hn, tmp);
        for (int64_t c = 0; c < hc; ++c) {
            const float sc = rsqrt_meansq_f64(hv + c * n, n, eps);
            const vf vs = vf_set1(sc);
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(N + c * n + i,
                          vf_mul(vf_mul(vf_loadu(hv + c * n + i), vs), vf_loadu(gain + c * n + i)));
            for (; i < n; ++i) N[c * n + i] = hv[c * n + i] * sc * gain[c * n + i];
        }
        if (h->dtype != RAD_F32) {
            row_from_f32(h->dtype, N, tmp, hn);
            row_to_f32(h->dtype, tmp, N, hn);
        }

        /* ---- the read gate: hc*n -> lowrank -> hc*n, silu between and sigmoid after. */
        for (int64_t j = 0; j < lr; ++j) {
            const float t = dotf(N, dn + j * hn, hn) / (float)hc;
            dv[j] = narrow1(down->dtype, t / (1.0f + std::exp(-t)));    /* silu */
        }

        /* ---- the mix, and the MEAN over the streams. */
        const float invhc = 1.0f / (float)hc;
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t c = 0; c < hc; ++c) {
                const int64_t row = c * n + i;
                const float gv = dotf(dv, upw + row * lr, lr);
                acc += narrow1(up->dtype, 1.0f / (1.0f + std::exp(-gv))) * N[row];
            }
            xv[i] = acc * invhc;
        }
        if (xs == 1) out_row(x, xb, n, xv);
        else for (int64_t i = 0; i < n; ++i) stt(x, xb + i * xs, xv[i]);

        /* ---- the per-branch write gates. */
        if (injw && inj) {
            const int64_t ib = rowoff(inj, r, hc), is = laststride(inj);
            for (int64_t c = 0; c < hc; ++c) {
                const float acc = dotf(N, iw + c * hn, hn) * invhc;
                stt(inj, ib + c * is, 2.0f / (1.0f + std::exp(-acc)));
            }
        }

        /* ---- the optional fp8 form of x, READ BACK FROM WHAT WAS STORED. `x` is narrowed to its
         * own dtype first and the codes come from the narrowed value, not from the accumulator,
         * because that is what a fused kernel necessarily does and the two have to agree bit for
         * bit. The round trip is done in this buffer rather than by re-reading the operand: same
         * arithmetic, and it does not depend on the store having retired. */
        if (((q && q->data) || rg) && x->dtype != RAD_F32) {
            row_from_f32(x->dtype, xv, tmp, n);
            row_to_f32(x->dtype, tmp, xv, n);
        }
        if (q && q->data) {
            const int64_t qb = rowoff(q, r, n), qs = laststride(q);
            float* qv = N;     /* N is dead from here; the codes reuse its storage. */
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < n) ? k0 + g : n;
                const float amax = row_amax(xv + k0, k1 - k0);
                const float sc = amax > 0.0f ? amax * (1.0f / 448.0f) : 1.0f;
                const float invs = 1.0f / sc;
                const vf vi = vf_set1(invs), hi = vf_set1(448.0f), lo = vf_set1(-448.0f);
                int64_t i = k0;
                for (; i + VF_N <= k1; i += VF_N)
                    vf_storeu(qv + i, vf_min(vf_max(vf_mul(vf_loadu(xv + i), vi), lo), hi));
                for (; i < k1; ++i) {
                    float v = xv[i] * invs;
                    qv[i] = v > 448.0f ? 448.0f : (v < -448.0f ? -448.0f : v);
                }
                stt(s, rowoff(s, r, ngrp) + gi * laststride(s), sc);
            }
            if (qs == 1) out_row(q, qb, n, qv);
            else for (int64_t i = 0; i < n; ++i) stt(q, qb + i * qs, qv[i]);
        }

        /* ---- the optional ROTATED fp8 form of x, from the same stored values. */
        if (rg) {
            float* rv = N;     /* N and tmp are dead once the plain codes are stored. */
            float* rc = tmp;
            std::memcpy(rv, xv, (size_t)n * sizeof(float));
            fwht_row(rv, n, rg);
            const int64_t rgrp = n / rg;
            for (int64_t gi = 0; gi < rgrp; ++gi) {
                float sc;
                fp8_group(rv, rc, gi * rg, gi * rg + rg, &sc);
                stt(rs, rowoff(rs, r, rgrp) + gi * laststride(rs), sc);
            }
            const int64_t qb = rowoff(rq, r, n), qs = laststride(rq);
            if (qs == 1) out_row(rq, qb, n, rc);
            else for (int64_t i = 0; i < n; ++i) stt(rq, qb + i * qs, rc[i]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_write */
/* `h[s] += inj[s] * y` for each of the hc streams, in place. `h` is the only output and it is the
 * stream the paired read was handed, UNNORMED and unchanged by it. */
AVX_KERNEL(hc_write) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor* y = &a->t[0];
    const RadTensor* inj = &a->t[1];
    const RadTensor* h = &a->t[2];

    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    const int64_t rows = numel(y) / n;
    if (rows <= 0 || numel(h) < rows * hn || numel(inj) < rows * hc) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(rows * hn >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t yb = rowoff(y, r, n), hb = rowoff(h, r, hn), ib = rowoff(inj, r, hc);
        const int64_t hs = laststride(h), is = laststride(inj);
        float* yv = scratch_f32((size_t)n);
        float* hv = scratch_f32_b((size_t)n);
        const float* yy = in_row(y, yb, n, yv);
        for (int64_t c = 0; c < hc; ++c) {
            const float gain = ldt(inj, ib + c * is);
            if (hs != 1) {
                for (int64_t i = 0; i < n; ++i) {
                    const int64_t j = hb + (c * n + i) * hs;
                    stt(h, j, ldt(h, j) + gain * yy[i]);
                }
                continue;
            }
            const float* cur = in_row(h, hb + c * n, n, hv);
            if (cur != hv) std::memcpy(hv, cur, (size_t)n * sizeof(float));
            const vf vg = vf_set1(gain);
            int64_t i = 0;
            for (; i + VF_N <= n; i += VF_N)
                vf_storeu(hv + i, vf_fma(vg, vf_loadu(yy + i), vf_loadu(hv + i)));
            for (; i < n; ++i) hv[i] += gain * yy[i];
            out_row(h, hb + c * n, n, hv);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ hc_mean */
/* THE CAPTURE'S REDUCTION. x[m,i] = mean over the hc streams of h[m, c*N+i], summed in f32 and
 * stored once. libref/ref_hc.cpp is the definition.
 *
 * ONE LOOP, NO SCRATCH. This is the host fallback: `hc` is small and the op is memory-bound, so a
 * per-row converted copy of the wide stream would allocate and grow a thread-local arena to save
 * arithmetic this op does not spend. `ldt`/`stt` already read and narrow through the operand's own
 * dtype and stride at the element, which is the same path every other generic op in this plugin
 * takes.
 *
 * THE ADDRESSING IS BOTH STRIDES. `h`'s column step is `stride[1]` and its streams sit `N` apart,
 * so element (m, c, i) is `h + m*stride[0] + (c*N + i)*stride[1]`; `x` is the tap plane's column
 * view, whose ROW pitch is `stride[0]` -- the plane's declared width -- and whose column step is
 * `stride[1]`. Nothing outside the named columns is read or written.
 *
 * THE ACCUMULATOR STARTS AT +0.0f AND THE STREAMS ARE ADDED IN ORDER, which is the oracle's
 * arithmetic including the sign of a zero: +0 + (-0) is +0, where seeding from stream 0 would keep
 * -0 through four negative zeros and store the other bf16 bit.
 *
 * `* (1/hc)` IS NOT `/ hc` for an hc that is not a power of two: the reciprocal rounds first. The
 * oracle divides, so the product form is taken only where it is exact (hc 4, the geometry this op
 * exists for) and the division is used otherwise. */
AVX_KERNEL(hc_mean) {
    if (!rad_args_have(a, 2)) return RAD_E_INVAL;
    const RadTensor* h = &a->t[0];
    const RadTensor* x = &a->t[1];

    const int64_t n = p_int(a, "N", 0);
    const int64_t hc = p_int(a, "hc", 0);
    if (n <= 0 || hc <= 0) return RAD_E_SHAPE;
    const char* dt = p_str(a, "dtype", nullptr);
    if (dt && std::strcmp(dt, "bf16") != 0) return RAD_E_DTYPE;
    if (h->dtype != RAD_BF16 || x->dtype != RAD_BF16) return RAD_E_DTYPE;
    if (h->rank != 2 || x->rank != 2) return RAD_E_SHAPE;
    const int64_t hn = hc * n;
    if (h->shape[1] != hn || x->shape[1] != n) return RAD_E_SHAPE;
    const int64_t rows = band_rows(h->shape[0], p_int(a, "M", 0));
    if (rows <= 0 || x->shape[0] != rows) return RAD_E_SHAPE;

    const int64_t hs = h->stride[1], xs = x->stride[1];
    if (hs < 1 || xs < 1) return RAD_E_SHAPE;
    /* The oracle's rule, so this kernel cannot accept a view the reference refuses: a row pitch
     * narrower than the row makes rows overlap. Unused with one row, where it may be zero. */
    if (h->stride[0] < 0 || x->stride[0] < 0) return RAD_E_SHAPE;
    if (rows > 1 && (h->stride[0] < hn || x->stride[0] < n)) return RAD_E_SHAPE;
    /* Aliasing is refused by the reference for the same reason; a caller that got here through a
     * different kernel library must not be the one that silently corrupts a tap. */
    {
        auto span = [](const RadTensor* t) {
            uint64_t e = 1;
            for (uint32_t i = 0; i < t->rank; ++i)
                e += (uint64_t)(t->shape[i] - 1) * (uint64_t)t->stride[i];
            return e * 2u;
        };
        const uintptr_t hp = (uintptr_t)h->data, xp = (uintptr_t)x->data;
        if (hp < xp + span(x) && xp < hp + span(h)) return RAD_E_INVAL;
    }

    const bool pow2_hc = (hc & (hc - 1)) == 0;
    const float invhc = pow2_hc ? 1.0f / (float)hc : 1.0f;
    const float divhc = (float)hc;
    AVX_PARALLEL_FOR_IF(rows * n >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = r * h->stride[0], xb = r * x->stride[0];
        for (int64_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (int64_t c = 0; c < hc; ++c) acc += ldt(h, hb + (c * n + i) * hs);
            stt(x, xb + i * xs, pow2_hc ? acc * invhc : acc / divhc);
        }
    }
    return RAD_OK;
}

/* ================================================================== mtp_enter
 *
 * THE MULTI-TOKEN-PREDICTION HEAD'S ENTRY, for a model whose residual stream is `hc` wide.
 *
 * A pre-norm MTP head joins the trunk's hidden state and the next token's embedding with ONE `fc`
 * over their concatenation. Qwen4-Exp's cannot: its hidden state is the WIDE stream and the
 * checkpoint carries TWO square matrices, so the hidden half is `fc_h` applied to EACH of the hc
 * sub-streams and the embedding half is one n-wide product added into all of them.
 *
 *     ne[j]     = e[j] * rsqrt(mean(e^2) + eps) * (w_e[j] + wadd)
 *     fe[i]     = sum_j fc_e[i][j] * ne[j]
 *     nh[k][j]  = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
 *     x[k*n+i]  = sum_j fc_h[i][j] * nh[k][j] + fe[i]
 *
 * `ngroup` IS THE ONE THING THE CHECKPOINT DOES NOT SAY: a [hc*n] gain fits both a grouped norm
 * (hc rsqrts) and a flat one (a single rsqrt over the wide row). Both are self-consistent and no
 * per-op oracle can tell them apart; the cheap empirical answer is the draft acceptance rate, so
 * the choice is carried as a parameter and asked once at declare.
 *
 * ============================== IT IS A GEMM AND IS RUN AS ONE ==============================
 *
 * libref computes this token by token, converting `fc_h` and `fc_e` out of their stored dtype
 * inside the innermost loop -- so an [n, n] weight is converted `rows * hc` times. Restructured
 * here as what it is: the norms are materialised into a PANEL first ([rows*hc, n] for the hidden
 * half, [rows, n] for the embedding), and then the loop runs over the OUTPUT COLUMN i, packing
 * `fc_h` row i and `fc_e` row i once and contracting them against every panel row.
 *
 * That is the same choice avx_gemm.cpp makes and for the same reason -- but note what it does NOT
 * do: it does not pre-widen the whole [n, n] weight, which at n = 2560 would be 26 MiB per matrix.
 * One row at a time, thread-local, is bounded by n floats and the weight is still read exactly
 * once. The panel is the thing that had to be materialised, and it is rows*hc*n -- 328 KiB on a
 * draft round, which is what this op is for.
 */
AVX_KERNEL(mtp_enter) {
    if (!rad_args_have(a, 7)) return RAD_E_INVAL;
    const RadTensor* h = &a->t[0];
    const RadTensor* e = &a->t[1];
    const RadTensor* w_h = &a->t[2];
    const RadTensor* w_e = &a->t[3];
    const RadTensor* fc_h = &a->t[4];
    const RadTensor* fc_e = &a->t[5];
    const RadTensor* x = &a->t[6];

    const int64_t n = p_int(a, "n", 0);
    const int64_t hc = p_int(a, "hc", 0);
    const float eps = p_f32(a, "eps", 1e-6f);
    const float wadd = p_f32(a, "wadd", 0.0f);
    const int64_t ngroup = p_int(a, "ngroup", hc);
    if (n <= 0 || hc <= 1) return RAD_E_SHAPE;
    if (ngroup != 1 && ngroup != hc) return RAD_E_SHAPE;
    const int64_t hn = hc * n;

    const int64_t rows = numel(h) / hn;
    if (rows <= 0 || numel(e) < rows * n || numel(x) < rows * hn) return RAD_E_SHAPE;
    if (numel(w_h) < hn || numel(w_e) < n) return RAD_E_SHAPE;
    if (numel(fc_h) < n * n || numel(fc_e) < n * n) return RAD_E_SHAPE;

    auto pad = [](size_t v) { return (v + 15) & ~(size_t)15; };
    const size_t o_gh = 0;
    const size_t o_ge = pad(o_gh + (size_t)hn);
    const size_t o_nh = pad(o_ge + (size_t)n);
    const size_t o_ne = pad(o_nh + (size_t)rows * (size_t)hn);
    const size_t total = o_ne + (size_t)rows * (size_t)n + 16;
    float* blk = (float*)scratch_raw(total * sizeof(float));
    float* gh = blk + o_gh;
    float* ge = blk + o_ge;
    float* nh = blk + o_nh;     /* [rows * hc, n] */
    float* ne = blk + o_ne;     /* [rows, n] */
    {
        const int64_t whs = laststride(w_h), wes = laststride(w_e);
        if (whs == 1) row_to_f32(w_h->dtype, w_h->data, gh, hn);
        else for (int64_t i = 0; i < hn; ++i) gh[i] = ldt(w_h, i * whs);
        if (wes == 1) row_to_f32(w_e->dtype, w_e->data, ge, n);
        else for (int64_t i = 0; i < n; ++i) ge[i] = ldt(w_e, i * wes);
        if (wadd != 0.0f) {
            for (int64_t i = 0; i < hn; ++i) gh[i] += wadd;
            for (int64_t i = 0; i < n; ++i) ge[i] += wadd;
        }
    }

    /* Phase one: the two norms, into the panel. Both are rounded to their source tensor's dtype,
     * which is what `.type_as(x)` does in the reference model and what the products then read. */
    AVX_PARALLEL_FOR_IF(rows * hn >= AVX_PAR_MIN)
    for (int64_t r = 0; r < rows; ++r) {
        float* buf = scratch_f32((size_t)hn);
        float* tmp = scratch_f32_b((size_t)hn);
        const float* ev = in_row(e, rowoff(e, r, n), n, buf);
        const float rse = rsqrt_meansq_f64(ev, n, eps);
        float* dst_e = ne + r * n;
        for (int64_t i = 0; i < n; ++i) dst_e[i] = ev[i] * rse * ge[i];
        if (e->dtype != RAD_F32) {
            row_from_f32(e->dtype, dst_e, tmp, n);
            row_to_f32(e->dtype, tmp, dst_e, n);
        }

        const float* hv = in_row(h, rowoff(h, r, hn), hn, buf);
        const float flat = (ngroup == 1) ? rsqrt_meansq_f64(hv, hn, eps) : 0.0f;
        float* dst_h = nh + r * hn;
        for (int64_t c = 0; c < hc; ++c) {
            const float sc = (ngroup == 1) ? flat : rsqrt_meansq_f64(hv + c * n, n, eps);
            for (int64_t i = 0; i < n; ++i)
                dst_h[c * n + i] = hv[c * n + i] * sc * gh[c * n + i];
        }
        if (h->dtype != RAD_F32) {
            row_from_f32(h->dtype, dst_h, tmp, hn);
            row_to_f32(h->dtype, tmp, dst_h, hn);
        }
    }

    /* Phase two: one output column at a time, so each weight row is widened exactly once. */
    const int64_t xs = laststride(x);
    AVX_PARALLEL_FOR_IF(rows * hc * n >= AVX_PAR_MIN)
    for (int64_t i = 0; i < n; ++i) {
        float* fh = scratch_f32((size_t)n);
        float* fe = scratch_f32_b((size_t)n);
        const int64_t fhb = rowoff(fc_h, i, n), fhs = laststride(fc_h);
        const int64_t feb = rowoff(fc_e, i, n), fes = laststride(fc_e);
        if (fhs == 1) row_to_f32(fc_h->dtype, byte_at(fc_h->data, fc_h->dtype, fhb), fh, n);
        else for (int64_t j = 0; j < n; ++j) fh[j] = ldt(fc_h, fhb + j * fhs);
        if (fes == 1) row_to_f32(fc_e->dtype, byte_at(fc_e->data, fc_e->dtype, feb), fe, n);
        else for (int64_t j = 0; j < n; ++j) fe[j] = ldt(fc_e, feb + j * fes);

        for (int64_t r = 0; r < rows; ++r) {
            const float fev = dotf(fe, ne + r * n, n);
            const int64_t xb = rowoff(x, r, hn);
            for (int64_t c = 0; c < hc; ++c)
                stt(x, xb + (c * n + i) * xs, dotf(fh, nh + r * hn + c * n, n) + fev);
        }
    }
    return RAD_OK;
}
