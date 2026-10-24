/* ref_moe.cpp -- routing, the scatter/gather pair, the grouped GEMM, and the exact row top-k.
 *
 * ============================== WHAT sorted_tok CARRIES ==============================
 *
 * docs/OPS.md names the operand and does not say what is in it, and the choice decides whether
 * `moe_gather` can work at all. It holds the FLATTENED (token, slot) index -- token * top_k + slot
 * -- grouped by expert, not the bare token id. With the bare id, gather would have no way to find
 * which of a token's top_k routing weights applies to a row, and would have to search. This is
 * also vLLM's `sorted_token_ids` convention, so a kernel lifted from there needs no translation.
 *
 * A slot whose expert id is outside [0, n_expert) is DROPPED: it is not counted, and the tail of
 * sorted_tok from expert_offset[n_expert] to M*top_k is filled with -1. A padded batch row has to
 * be expressible, and a negative id is how it is said.
 *
 * The histogram this produces is what rad_route_report hands the heat engine; it is read back
 * asynchronously and consumed on the NEXT step, and nothing here blocks the compute stream
 * (spec §5.5).
 */
#include "ref_common.h"
#include "ref_gemm.h"
#include "ref_ops.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace ref;

/* ------------------------------------------------------------------ router_topk */
/* softmax over ALL n_expert logits, then the top_k by probability, then -- if `norm` -- renormalise
 * the kept weights to sum to one. Softmax first and select second is the order Mixtral, Qwen3 and
 * every model in this engine's scope use; selecting first and normalising the raw logits would be
 * a different router, so the order is part of the op rather than an implementation choice. */
int ref_router_topk(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *ids = &a->t[1], *w = &a->t[2];
    const int64_t n_expert = p_int(a, "n_expert", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    const int64_t top_k    = p_int(a, "top_k", 1);
    const int     norm     = (int)p_int(a, "norm", 0);
    if (n_expert <= 0 || top_k <= 0 || top_k > n_expert) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_expert;
    if (M <= 0 || numel(ids) < M * top_k || numel(w) < M * top_k) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t lb = rowoff(lg, m, n_expert), ls = laststride(lg);
        float mx = -INFINITY;
        for (int64_t e = 0; e < n_expert; ++e) {
            float v = ldt(lg, lb + e * ls);
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int64_t e = 0; e < n_expert; ++e) sum += ref_exp(ldt(lg, lb + e * ls) - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

        /* The selection runs on the LOGITS, not the probabilities: softmax is monotone, so the
         * order is identical and the comparison is exact rather than through an exp. */
        float pv = INFINITY;
        int64_t pi = -1, kept = 0;
        float wsum = 0.0f;
        for (int64_t r = 0; r < top_k; ++r) {
            if (!topk_next(lg, lb, ls, n_expert, &pv, &pi)) break;
            const float p = ref_exp(pv - mx) * inv;
            st_int(ids->dtype, ids->data, rowoff(ids, m, top_k) + r * laststride(ids), pi);
            stt(w, rowoff(w, m, top_k) + r * laststride(w), p);
            wsum += p;
            ++kept;
        }
        if (norm && wsum > 0.0f) {
            const float n2 = 1.0f / wsum;
            for (int64_t r = 0; r < kept; ++r) {
                const int64_t off = rowoff(w, m, top_k) + r * laststride(w);
                stt(w, off, ldt(w, off) * n2);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_scatter */
/* `protect`: local experts the sort places LAST, in the order listed, the rest closed up -- the
 * layout a layer that serves a few experts with a second grouped GEMM needs (libr4d's sort_place).
 * Returns false on a list that is not strictly ascending inside [0, n). */
static bool sort_places(const char* s, int64_t n, std::vector<int64_t>* place) {
    std::vector<int64_t> prot;
    for (const char* c = s ? s : ""; *c;) {
        char* end = nullptr;
        const long long v = std::strtoll(c, &end, 10);
        if (end == c || v < 0 || v >= n || (!prot.empty() && v <= prot.back())) return false;
        prot.push_back(v);
        c = end;
        if (*c == ',') ++c;
        else if (*c) return false;
    }
    place->assign((size_t)n, 0);
    int64_t next = 0;
    for (int64_t e = 0; e < n; ++e)
        if (!std::binary_search(prot.begin(), prot.end(), e)) (*place)[(size_t)e] = next++;
    for (size_t j = 0; j < prot.size(); ++j) (*place)[(size_t)prot[j]] = next++;
    return true;
}

/* A stable counting sort of the (token, slot) pairs by expert id. Serial on purpose: it is O(M) on
 * a batch's worth of small integers, and a parallel version would need a per-thread histogram and
 * a prefix pass to stay stable, which is more code than the work it saves. */
int ref_moe_scatter(const RadArgs* a, RadStream) {
    if (!have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ids = &a->t[0], *sorted = &a->t[1], *off = &a->t[2], *cnt = &a->t[3];
    const int64_t n_expert = p_int(a, "n_expert", 0);
    const int64_t top_k    = p_int(a, "top_k", 1);
    /* THIS RANK'S FIRST EXPERT. Subtracted from every routed id, so the range test below is the
     * one that was always here and the sort is over local indices -- which is what the weight
     * table, the offsets and the GEMM downstream are indexed by. */
    const int64_t base     = p_int(a, "expert_base", 0);
    if (n_expert <= 0 || top_k <= 0 || base < 0) return RAD_E_SHAPE;
    const int64_t M = numel(ids) / top_k;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T) return RAD_E_SHAPE;
    if (numel(off) < n_expert + 1 || numel(cnt) < n_expert) return RAD_E_SHAPE;
    std::vector<int64_t> place;
    if (!sort_places(p_str(a, "protect", nullptr), n_expert, &place)) return RAD_E_SHAPE;
    /* A slot's place in the sort, or -1 for an expert outside this call's slice. */
    auto at = [&](int64_t i) -> int64_t {
        const int64_t e = ld_int(ids->dtype, ids->data, offlin(ids, i)) - base;
        return (e < 0 || e >= n_expert) ? -1 : place[(size_t)e];
    };

    for (int64_t e = 0; e < n_expert; ++e) st_int(cnt->dtype, cnt->data, offlin(cnt, e), 0);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e < 0) continue;
        const int64_t o = offlin(cnt, e);
        st_int(cnt->dtype, cnt->data, o, ld_int(cnt->dtype, cnt->data, o) + 1);
    }
    int64_t run = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        st_int(off->dtype, off->data, offlin(off, e), run);
        run += ld_int(cnt->dtype, cnt->data, offlin(cnt, e));
    }
    st_int(off->dtype, off->data, offlin(off, n_expert), run);

    /* A second cursor pass rather than mutating the offsets: the offsets are an output the caller
     * reads, and rewinding them afterwards is one more place to be wrong. */
    for (int64_t i = run; i < T; ++i) st_int(sorted->dtype, sorted->data, offlin(sorted, i), -1);
    for (int64_t e = 0, w = 0; e < n_expert; ++e) {
        w = ld_int(off->dtype, off->data, offlin(off, e));
        for (int64_t i = 0; i < T; ++i) {
            if (at(i) != e) continue;
            st_int(sorted->dtype, sorted->data, offlin(sorted, w), i);
            ++w;
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_gemm, moe_gemm_q */
/* y[i, :] = a[row(i), :] @ w[e]^T, where e is the expert whose offset range contains i and
 * `a_order` picks row(i) -- `sorted_tok[i] / top_k` when the activation has one row per TOKEN,
 * and i itself when it already has one per (token, slot). The output is in SORTED order -- one
 * row per (token, slot) pair, laid out the way moe_scatter grouped them -- because that is the
 * layout that makes the expert loop contiguous, which is the entire point of the scatter.
 * moe_gather puts it back.
 *
 * THE WEIGHT OPERAND IS A TABLE (RAD_OPD_WTAB), not a plane, and ref_gemm.h's expert_entry() is
 * where that is read. A layer's experts are separate movement units, so after placement one is in
 * VRAM and the next is streamed over the link; `base + e*stride` describes only the case where
 * nothing was offloaded. The stacked form is still accepted, because a caller with no engine
 * behind it -- rad-kbench replaying a fixture -- has nowhere to put a pointer array.
 *
 * The quantised arm is the same walk with ref_gemm.h's Scales over both operands, and the sub-sum
 * is accumulated from the RAW CODES and scaled once per group exactly as `gemm_nt_q` does: the
 * two references must agree in the last bits or rad-kbench blames a kernel for a choice made
 * here. Only the WEIGHT scale grid is block-shaped -- one scale per 128x128 tile of an expert --
 * while the activation's is per row per group of K, which is `rblk = 1`, which is what
 * make_scales derives on its own from the two extents. */
static int moe_gemm_impl(const RadArgs* a, bool quant, int ncls = 2) {
    const int n_opd = quant ? (ncls == 4 ? 13 : 9) : 5;
    if (!have(a, n_opd)) return RAD_E_INVAL;
    const RadTensor* A      = &a->t[0];
    const RadTensor* as_    = quant ? t_in(a, 1) : nullptr;
    const RadTensor* W      = &a->t[quant ? 2 : 1];
    const RadTensor* ws_    = quant ? t_in(a, 3) : nullptr;
    const RadTensor* sorted = &a->t[quant ? 4 : 2];
    const RadTensor* off    = &a->t[quant ? 5 : 3];
    const RadTensor* Y      = &a->t[quant ? (ncls == 4 ? 12 : 8) : 4];
    /* THE QUANTISED ARM'S EXPERTS ARE TWO TABLES BY PARITY -- see ref_registry.cpp -- or, for
     * `moe_gemm_q_mod4`, four by the expert's number mod 4; the plain arm's are one, which is the
     * same walk with every expert in the first class. */
    const RadTensor* W1     = quant ? &a->t[6] : W;
    const RadTensor* ws1_   = quant ? t_in(a, 7) : nullptr;
    const bool two = quant;
    const int  nc  = quant ? ncls : 1;
    const RadTensor* Wq[4]  = { W, W1, quant && nc == 4 ? &a->t[8] : W,
                                quant && nc == 4 ? &a->t[10] : W };
    const RadTensor* wsq[4] = { ws_, ws1_, quant && nc == 4 ? t_in(a, 9) : nullptr,
                                quant && nc == 4 ? t_in(a, 11) : nullptr };

    /* "sorted" means `a` already has one row per (token, slot) in the scatter's order, which is
     * what the DOWN projection of a routed block reads -- its input is the first projection's
     * output. "token" (the default, and the gate_up projection's case) means one row per token,
     * gathered through sorted_tok. See ref_registry.cpp for why this is said and not derived. */
    const char* ord = p_str(a, "a_order", "token");
    const bool  sorted_a = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !sorted_a && std::strcmp(ord, "token") != 0) return RAD_E_UNSUPPORTED;

    const int64_t N0 = p_int(a, "N", W->rank >= 3 ? W->shape[1] : 0);
    const int64_t K0 = p_int(a, "K", W->rank >= 3 ? W->shape[2] : 0);
    /* Each class's geometry, defaulting to class 0's: N_odd/K_odd at two classes, N_q/K_q at
     * four. */
    static const char* const kN4[4] = { "N", "N_1", "N_2", "N_3" };
    static const char* const kK4[4] = { "K", "K_1", "K_2", "K_3" };
    int64_t Nq[4] = { N0, N0, N0, N0 }, Kq[4] = { K0, K0, K0, K0 };
    for (int q = 1; q < nc; ++q) {
        Nq[q] = p_int(a, nc == 4 ? kN4[q] : "N_odd", N0);
        Kq[q] = p_int(a, nc == 4 ? kK4[q] : "K_odd", K0);
    }
    const int64_t parts = two ? p_int(a, "parts", 1) : 1;
    const int64_t n_expert = p_int(a, "n_expert", W->rank >= 3 ? W->shape[0] : 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    if (parts <= 0 || n_expert <= 0 || top_k <= 0) return RAD_E_SHAPE;
    /* The output is the widest class's width, the activation the longest class's K. Class q
     * holds ceil((n - q) / nc) experts. */
    int64_t N = 0, K = 0;
    for (int q = 0; q < nc; ++q) {
        if (Nq[q] <= 0 || Kq[q] <= 0 || Nq[q] % parts) return RAD_E_SHAPE;
        if (Nq[q] > N) N = Nq[q];
        if (Kq[q] > K) K = Kq[q];
        const int64_t neq = (n_expert + nc - 1 - q) / nc;
        if (neq > 0 && (Wq[q]->rank < 3 || Wq[q]->shape[0] < neq)) return RAD_E_SHAPE;
    }
    const int64_t T = numel(Y) / N;
    if (T <= 0 || numel(sorted) < T || numel(off) < n_expert + 1) return RAD_E_SHAPE;

    /* `y` HAS ONE ROW PER (token, slot) AND `a` HAS WHATEVER `a_order` SAYS. The token count is
     * recovered from T and top_k; the activation's row count is T under "sorted" and M under
     * "token", and it is checked against THAT rather than against either extent on its own -- a
     * sorted-order operand with only M rows is a caller that handed the down projection the
     * gate_up projection's buffer. */
    const int64_t M  = T / top_k;
    const int64_t AR = sorted_a ? T : M;
    if (M <= 0 || numel(A) < AR * K) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, AR, K);
    const Scales SA = make_scales(as_, AR, K, false);
    const int64_t gp = p_int(a, "group", 0);
    if (quant && gp > 0 && SA.p && SA.group != gp) return RAD_E_SHAPE;

    const int64_t y_rs = row_stride(Y, N);
    const int64_t y_cs = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    /* Per-expert readings, built once outside the row loop: a table lookup and a Scales are cheap,
     * and doing them per row would be T of them against n_expert. A null `data` means the operand
     * could not be read as a table at all, which is refused before the loop because an OpenMP body
     * cannot return. Expert e is entry e / nc of class e % nc. */
    std::vector<RadTensor> we((size_t)n_expert), wse((size_t)n_expert);
    std::vector<Mat>       mw((size_t)n_expert);
    std::vector<Scales>    SW((size_t)n_expert);
    std::vector<int64_t>   en((size_t)n_expert), ek((size_t)n_expert);
    for (int64_t e = 0; e < n_expert; ++e) {
        const int q = (int)(e % nc);
        const RadTensor* tw = Wq[q];
        const RadTensor* ts = wsq[q];
        const int64_t ix = e / nc;
        const int64_t n = Nq[q], k = Kq[q];
        en[(size_t)e] = n;
        ek[(size_t)e] = k;
        we[(size_t)e] = expert_entry(tw, ix);
        if (!we[(size_t)e].data) return RAD_E_STRIDE;
        if (we[(size_t)e].rank < 2 || we[(size_t)e].shape[0] < n ||
            we[(size_t)e].shape[1] < k) return RAD_E_SHAPE;
        mw[(size_t)e] = as_mat(&we[(size_t)e], n, k);
        if (quant && ts) {
            wse[(size_t)e] = expert_entry(ts, ix);
            if (!wse[(size_t)e].data) return RAD_E_STRIDE;
            SW[(size_t)e] = make_scales(&wse[(size_t)e], n, k, false);
        } else {
            SW[(size_t)e] = make_scales(nullptr, n, k, false);
        }
    }

    const int64_t fy = N / parts;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ld_int(sorted->dtype, sorted->data, offlin(sorted, i));
        if (flat < 0) {                     /* a dropped slot: a zero row, never stale arena bytes */
            for (int64_t n = 0; n < N; ++n) stt(Y, i * y_rs + n * y_cs, 0.0f);
            continue;
        }
        /* Which expert owns row i: a binary search over the offsets, so a 256-expert model does
         * not pay a linear scan per row. */
        int64_t lo = 0, hi = n_expert;
        while (lo + 1 < hi) {
            const int64_t mid = (lo + hi) / 2;
            if (ld_int(off->dtype, off->data, offlin(off, mid)) <= i) lo = mid; else hi = mid;
        }
        const int64_t e = lo;
        const int64_t tok = sorted_a ? i : flat / top_k;
        if (tok >= AR) { for (int64_t n = 0; n < N; ++n) stt(Y, i * y_rs + n * y_cs, 0.0f); continue; }

        const Mat&    B  = mw[(size_t)e];
        const Scales& SB = SW[(size_t)e];
        const int64_t fc = en[(size_t)e] / parts;
        /* One product, and it is `quant_dot`'s: the same grouping and the same order the dense
         * quantised GEMM folds its two scales in. A second copy of that arithmetic here would be a
         * second place for it to be right by accident -- the expert GEMM and gemm_nt_q are the
         * same function over different weights, and a reference that spelled them differently
         * would put an ulp between them that no tolerance is meant to absorb. Output column n is
         * weight row (n / fy) * fc + n % fy, and zero past a narrower class's part. */
        for (int64_t n = 0; n < N; ++n) {
            const int64_t pp = n / fy, j = n - pp * fy;
            stt(Y, i * y_rs + n * y_cs,
                j < fc ? quant_dot(ma, B, SA, SB, tok, pp * fc + j, ek[(size_t)e]) : 0.0f);
        }
    }
    return RAD_OK;
}

int ref_moe_gemm(const RadArgs* a, RadStream)   { return moe_gemm_impl(a, false); }
int ref_moe_gemm_q(const RadArgs* a, RadStream) { return moe_gemm_impl(a, true); }
int ref_moe_gemm_q_mod4(const RadArgs* a, RadStream) { return moe_gemm_impl(a, true, 4); }

/* ------------------------------------------------------------------ moe_gather */
/* y[t] = sum over the token's slots of expert_w[t, j] * y_expert[position of (t, j)].
 *
 * THE SUMMATION ORDER IS ASCENDING SLOT j, which is what makes this reproducible: an
 * implementation that walked the expert-sorted rows instead would sum a token's contributions in
 * expert-id order, and the two differ in the last bits.
 *
 * The inverse permutation is rebuilt in blocks rather than allocated: one pass over sorted_tok per
 * block of tokens, into a fixed stack array. That costs (M / block) passes, which at the sizes an
 * MoE layer runs is nothing, and it keeps the op free of both scratch and allocation. */
int ref_moe_gather(const RadArgs* a, RadStream) {
    /* Operands are POSITIONAL, so an absent shared pair is two NULL SLOTS rather than a shorter
     * array: `have` demands all six and would refuse every unfolded call. */
    if (a->n_t < 6) return RAD_E_INVAL;
    const RadTensor *ye = &a->t[0], *ew = &a->t[1], *sorted = &a->t[2], *Y = &a->t[5];
    if (!ye->data || !ew->data || !sorted->data || !Y->data) return RAD_E_INVAL;
    /* The shared arm, folded: y = bf16(shared * act(gate)) + bf16(gathered). A pair or neither. */
    const RadTensor *sh = t_in(a, 3), *sg = t_in(a, 4);
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    const char* const act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (sh && act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    if (n <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / n;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T || numel(ew) < T || numel(ye) < T * n) return RAD_E_SHAPE;

    enum { INV_MAX = 4096 };
    const int64_t tb = INV_MAX / top_k > 0 ? INV_MAX / top_k : 1;
    const int64_t nblk = (M + tb - 1) / tb;

    const int64_t ye_rs = ye->rank >= 2 ? ye->stride[ye->rank - 2] : n;
    const int64_t ye_cs = ye->rank >= 2 ? ye->stride[ye->rank - 1] : 1;
    const int64_t y_rs  = Y->rank >= 2 ? Y->stride[Y->rank - 2] : n;
    const int64_t y_cs  = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblk; ++b) {
        int32_t inv[INV_MAX];
        const int64_t t0 = b * tb, t1 = (t0 + tb < M) ? t0 + tb : M;
        const int64_t span = (t1 - t0) * top_k;
        for (int64_t i = 0; i < span; ++i) inv[i] = -1;
        for (int64_t i = 0; i < T; ++i) {
            const int64_t f = ld_int(sorted->dtype, sorted->data, offlin(sorted, i));
            if (f < t0 * top_k || f >= t1 * top_k) continue;
            inv[f - t0 * top_k] = (int32_t)i;
        }
        for (int64_t t = t0; t < t1; ++t) {
            float gate = 0.0f;
            if (sh) {
                const float v0 = ldt(sg, rowoff(sg, t, 1));
                gate = sig ? act_sigmoid(v0) : v0;
            }
            for (int64_t c = 0; c < n; ++c) {
                float acc = 0.0f;
                for (int64_t j = 0; j < top_k; ++j) {
                    const int32_t i = inv[(t - t0) * top_k + j];
                    if (i < 0) continue;
                    acc += ldt(ew, rowoff(ew, t, top_k) + j * laststride(ew)) *
                           ldt(ye, (int64_t)i * ye_rs + c * ye_cs);
                }
                const int64_t yo = t * y_rs + c * y_cs;
                /* THE NARROWS ARE THE UNFUSED PAIR'S, and the store is what performs them: the
                 * gather wrote a bf16 buffer and scale_rows narrowed its own product before the
                 * add, so both operands of the sum are rounded and the sum is rounded again. */
                stt(Y, yo, acc);
                if (!sh) continue;
                const float g = ldt(Y, yo);
                stt(Y, yo, ldt(sh, rowoff(sh, t, n) + c * laststride(sh)) * gate);
                stt(Y, yo, ldt(Y, yo) + g);
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ row_topk */
/* Exact per-row top-R, descending, ties to the lower column. The same selection the router uses,
 * exposed as its own op because a drafter's candidate set and a speculative tree both want it
 * without a softmax in front. */
int ref_row_topk(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    /* The vocab-parallel form. `off` shifts every column reported, so a rank's slice reports
     * global ids; `pairs` is the [M, R, 2] plane an all_gather moves, an int32 id in the first
     * float slot beside its value. Both absent is the ordinary single-rank top-R. */
    const RadTensor* pairs = t_in(a, 3);
    const int64_t off = p_int(a, "vocab_off", 0);
    const int64_t N = p_int(a, "N", x->rank >= 2 ? x->shape[x->rank - 1] : 0);
    const int64_t R = p_int(a, "R", 1);
    if (N <= 0 || R <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / N;
    if (M <= 0 || numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && (pairs->dtype != RAD_F32 || numel(pairs) < M * R * 2)) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t xb = rowoff(x, m, N), xs = laststride(x);
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < R; ++r) {
            const bool ok = topk_next(x, xb, xs, N, &pv, &pi);
            const int64_t id = ok ? pi + off : -1;
            st_int(idx->dtype, idx->data, rowoff(idx, m, R) + r * laststride(idx), id);
            stt(val, rowoff(val, m, R) + r * laststride(val), ok ? pv : -INFINITY);
            if (pairs) {
                float* const q = (float*)pairs->data + (m * R + r) * 2;
                const int32_t i32 = (int32_t)id;
                std::memcpy(q, &i32, sizeof i32);
                q[1] = ok ? pv : -INFINITY;
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ row_topk_merge */
/* The other half of the vocab-parallel top-R: the R largest of world_size all-gathered per-rank
 * candidate sets, in the same descending (idx, val) form the single-rank path produces, so
 * everything downstream is handed the answer it would have had from an unsharded plane.
 *
 * TIES BREAK ON THE LOWER ID, which is what makes that equality true rather than approximate: a
 * dense top-R over the whole vocabulary breaks ties on the lower column, so a merge that broke
 * them on rank order would disagree with it exactly when two shards hold the same logit.
 *
 * A slot the producing rank could not fill carries id -1 and is DROPPED -- a sentinel rather than
 * a -INFINITY value, because -inf is a value a masked logit legitimately takes. */
int ref_row_topk_merge(const RadArgs* a, RadStream) {
    if (!have(a, 3)) return RAD_E_INVAL;
    const RadTensor *g = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    const int64_t R = p_int(a, "R", 1);
    const int64_t W = p_int(a, "world_size", 1);
    const int64_t ld = (idx && idx->rank >= 2 && idx->shape[idx->rank - 1] > R) ? idx->shape[idx->rank - 1] : R;
    if (R <= 0 || W <= 0) return RAD_E_SHAPE;
    /* The rows are the OPERAND'S, not the parameter's: `M` is declared as a band ceiling and the
     * gathered plane is narrowed to the rows this step actually has. */
    const int64_t M = numel(g) / (W * R * 2);
    if (M <= 0) return RAD_E_SHAPE;
    if (g->dtype != RAD_F32 || numel(g) < M * W * R * 2) return RAD_E_SHAPE;
    if (numel(idx) < M * ld || numel(val) < M * ld) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        const float* const row = (const float*)g->data + m * W * R * 2;
        /* A selection sort over W*R entries: 32 of them at world 2 and R 16, so the quadratic
         * form is cheaper than anything with an allocation in it, and it is the reference. */
        std::vector<unsigned char> used((size_t)(W * R), 0);
        for (int64_t r = 0; r < R; ++r) {
            int64_t best = -1;
            float   bv   = 0.0f;
            int32_t bid  = 0;
            for (int64_t c = 0; c < W * R; ++c) {
                if (used[(size_t)c]) continue;
                int32_t id;
                std::memcpy(&id, row + c * 2, sizeof id);
                if (id < 0) continue;                      /* a pad, not a candidate */
                const float v = row[c * 2 + 1];
                if (best < 0 || v > bv || (v == bv && id < bid)) { best = c; bv = v; bid = id; }
            }
            if (best >= 0) used[(size_t)best] = 1;
            st_int(idx->dtype, idx->data, rowoff(idx, m, ld) + r * laststride(idx),
                   best >= 0 ? (int64_t)bid : -1);
            stt(val, rowoff(val, m, ld) + r * laststride(val), best >= 0 ? bv : -INFINITY);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ logit_rerank */
/* EXACTLY RESCORE R CANDIDATES A ROW AGAINST THE FULL-PRECISION HEAD.
 *
 * A speculative step can afford a COARSE draft head -- the target verifies every proposal, so a
 * worse draft costs acceptance and never a token -- but it cannot afford to be wrong about which
 * candidate the coarse head liked best. A heavily quantised head disagrees with the exact one
 * about the top-1 on a noticeable fraction of rows, and rescoring its top few candidates against
 * the exact weight recovers nearly all of that disagreement.
 *
 * So this is the second half of a coarse top-R: the coarse head proposes R, the exact weight
 * scores those R and only those R, and the answer is re-sorted on the exact values. R rows of the
 * head against the whole of it is the entire point -- a few hundred KB against a few hundred MB.
 *
 * `vocab_off` IS THE SHARD'S FIRST COLUMN, and the ids are GLOBAL, because that is the form
 * `row_topk` reports and `row_topk_merge` consumes. A candidate outside this rank's rows is not
 * this rank's to score and becomes a pad: under vocab-parallel drafting `row_topk` picked from
 * this rank's plane, so every id it reports is local and the guard is for a caller that mixes
 * them.
 *
 * ORDER, AND WHY IT IS THE SAME ORDER row_topk USES: descending by value, ties to the LOWER id.
 * The merge downstream breaks ties that way and the two have to agree, or a tie between shards
 * resolves differently from a tie inside one. A pad sorts last and keeps id -1. */
int ref_logit_rerank(const RadArgs* a, RadStream) {
    if (!have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *in = &a->t[2];
    const RadTensor *idx = &a->t[3], *val = &a->t[4];
    const RadTensor *pairs = (a->n_t > 5 && a->t[5].data) ? &a->t[5] : nullptr;
    const int64_t R    = p_int(a, "R", 1);
    const int64_t K    = p_int(a, "n_embd", 0);
    const int64_t NV   = p_int(a, "n_vocab", 0);
    const int64_t voff = p_int(a, "vocab_off", 0);
    const int64_t R_pairs = (pairs && pairs->rank >= 2 && pairs->shape[1] > 0) ? pairs->shape[1] : R;
    if (R <= 0 || K <= 0 || NV <= 0) return RAD_E_SHAPE;
    /* The rows are the OPERAND'S: `M` is a band ceiling and a step narrows it. */
    const int64_t M = numel(x) / K;
    if (M <= 0 || numel(in) < M * R) return RAD_E_SHAPE;
    if (numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && numel(pairs) < M * R_pairs * 2) return RAD_E_SHAPE;
    if (numel(w) < NV * K) return RAD_E_SHAPE;

    #pragma omp parallel for schedule(static)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<int32_t> id((size_t)R);
        std::vector<float>   sc((size_t)R);
        for (int64_t r = 0; r < R; ++r) {
            const int64_t g = ld_int(in->dtype, in->data, rowoff(in, m, R) + r * laststride(in));
            const int64_t local = g - voff;
            if (g < 0 || local < 0 || local >= NV) { id[(size_t)r] = -1; sc[(size_t)r] = -INFINITY; continue; }
            double acc = 0.0;                       /* the reference sums wide and in order */
            for (int64_t k = 0; k < K; ++k)
                acc += (double)ldt(x, rowoff(x, m, K) + k * laststride(x))
                     * (double)ldt(w, local * K + k);
            id[(size_t)r] = (int32_t)g;
            sc[(size_t)r] = (float)acc;
        }
        std::vector<unsigned char> used((size_t)R, 0);
        for (int64_t o = 0; o < R; ++o) {
            int64_t best = -1;
            for (int64_t c = 0; c < R; ++c) {
                if (used[(size_t)c] || id[(size_t)c] < 0) continue;
                if (best < 0 || sc[(size_t)c] > sc[(size_t)best] ||
                    (sc[(size_t)c] == sc[(size_t)best] && id[(size_t)c] < id[(size_t)best])) best = c;
            }
            const int32_t oid = best >= 0 ? id[(size_t)best] : -1;
            const float   ov  = best >= 0 ? sc[(size_t)best] : -INFINITY;
            if (best >= 0) used[(size_t)best] = 1;
            st_int(idx->dtype, idx->data, rowoff(idx, m, R) + o * laststride(idx), (int64_t)oid);
            stt(val, rowoff(val, m, R) + o * laststride(val), ov);
            if (pairs && o < R_pairs) {
                float* p = (float*)pairs->data + (m * R_pairs + o) * 2;
                std::memcpy(p, &oid, sizeof oid);
                p[1] = ov;
            }
        }
    }
    return RAD_OK;
}
