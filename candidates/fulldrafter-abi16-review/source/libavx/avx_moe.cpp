/* avx_moe.cpp -- routing, the scatter/gather pair, the grouped GEMM, and the exact row top-R.
 *
 * ============================== WHAT sorted_tok CARRIES ==============================
 *
 * The FLATTENED (token, slot) index -- token * top_k + slot -- grouped by expert, not the bare
 * token id. With the bare id, `moe_gather` would have no way to find which of a token's top_k
 * routing weights applies to a row and would have to search. It is also vLLM's
 * `sorted_token_ids` convention, so a kernel lifted from there needs no translation.
 *
 * A slot whose expert id is outside [0, n_expert) is DROPPED: not counted, and the tail from
 * expert_offset[n_expert] to M*top_k is filled with -1. A padded batch row has to be expressible.
 *
 * ============================== THREE ALGORITHMIC CHANGES, NOT JUST VECTORISATION ==============================
 *
 * This is the file where the reference's asymptotics, rather than its instruction mix, are the
 * thing to beat -- so three loops are rewritten rather than widened, and each produces the
 * identical answer:
 *
 *   moe_scatter is O(T + n_expert) instead of O(T * n_expert). libref places the rows by looping
 *   over experts and scanning the whole id array for each one, which is a stable sort and is
 *   quadratic in the expert count; at 256 experts that is 256 passes over the batch. A single pass
 *   with a per-expert write cursor is the same stable order -- rows are still appended in
 *   ascending i within an expert -- for one pass.
 *
 *   moe_gather builds the inverse permutation ONCE, O(T), instead of once per block of tokens.
 *   libref rebuilds it in blocks into a 4096-entry stack array specifically to avoid allocating,
 *   and pays (M / block) full passes over sorted_tok for it. The scratch arena removes the reason.
 *
 *   moe_gemm gathers the activation rows into a PANEL per expert and then runs the same
 *   register-blocked dot product avx_gemm.cpp uses. libref walks output rows and, for each, loops
 *   over N doing a fresh dot -- which means the expert's weight is re-read from memory once per
 *   row of that expert. Gathering first makes the weight pass contiguous and shared.
 *
 * The one thing NOT changed is the ORDER of moe_gather's summation. It is ascending slot j, and
 * that is what makes the op reproducible: an implementation that walked the expert-sorted rows
 * instead would sum a token's contributions in expert-id order and the two differ in the last bits.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace avx;

/* ================================================================== operand readings
 * avx_gemm.cpp's, and libref/ref_gemm.h's before that. Duplicated here rather than shared through
 * a header because these two translation units are compiled four times each and a shared header
 * would be a fifth definition to keep in step; the bodies are short and the contract they encode
 * is stated where it matters, in avx_gemm.cpp. `row_stride` and `grid_block` are avx_common.h's,
 * which every kernel here already shares. */

struct Mat { const RadTensor* t; int64_t rs, cs; };

static inline Mat as_mat(const RadTensor* t, int64_t cols) {
    Mat m{ t, cols, 1 };
    if (t->rank < 2) return m;
    m.cs = t->stride[t->rank - 1];
    m.rs = row_stride(t, cols);
    return m;
}

struct Scales {
    const void* p;
    uint32_t dt;
    int64_t rs, cs, group, rblk;
    inline float grp(int64_t row, int64_t g) const {
        return rad_load_f32(p, dt, (rblk > 1 ? row / rblk : row) * rs + g * cs);
    }
};

static Scales make_scales(const RadTensor* s, int64_t rows, int64_t K) {
    Scales q{};
    q.rblk = 1;
    if (!s || !s->data || rows <= 0 || numel(s) <= 0) {
        q.p = nullptr; q.group = K; q.dt = RAD_F32; return q;
    }
    q.p = s->data;
    q.dt = s->dtype;
    int64_t groups = 0;
    if (s->rank >= 2) {
        const int64_t srows = s->shape[s->rank - 2];
        q.rblk = srows >= rows ? 1 : grid_block(rows, srows);
        groups = s->shape[s->rank - 1];
        q.rs = s->stride[s->rank - 2];
        q.cs = s->stride[s->rank - 1];
    } else {
        const int64_t nel = numel(s);
        if (nel >= rows) { groups = nel / rows; q.rs = groups; }
        else             { groups = 1; q.rblk = grid_block(rows, nel); q.rs = 1; }
        q.cs = 1;
    }
    q.group = grid_block(K, groups);
    return q;
}

/* A RAD_OPD_WTAB operand: `data` is a host array of shape[0] pointers, one per expert, and
 * stride[0] is 0 to say so. The STACKED form -- a plain rank-3 [n_expert, N, K] with an ordinary
 * leading stride -- is accepted too and is what a caller outside the engine hands over, since a
 * checker drawing one dense plane has nowhere to put a pointer array. */
static RadTensor expert_entry(const RadTensor* w, int64_t e) {
    RadTensor t{};
    if (!w || !w->data || w->rank < 2 || e < 0 || e >= w->shape[0]) return t;
    t.dtype = w->dtype;
    t.rank = w->rank - 1;
    for (uint32_t d = 0; d + 1 < w->rank; ++d) {
        t.shape[d] = w->shape[d + 1];
        t.stride[d] = w->stride[d + 1];
    }
    if (w->stride[0] == 0) {
        void* const* tab = (void* const*)w->data;
        t.data = tab[e];
    } else {
        /* An element offset in a packed dtype can be mid-byte; the stacked form is only ever whole
         * planes, so the arithmetic is done in bytes off the leading stride. */
        const int64_t bits = rad_dtype_bits(w->dtype);
        const int64_t nbits = bits * w->stride[0] * e;
        if (bits <= 0 || (nbits & 7)) return t;      /* data stays null: unreadable */
        t.data = (void*)((const char*)w->data + (nbits >> 3));
    }
    return t;
}

static void pack_row(const Mat& m, int64_t r, int64_t K, float* dst) {
    const RadTensor* t = m.t;
    const int64_t base = r * m.rs;
    if (m.cs == 1) {
        const int sub = (t->dtype == RAD_I4 || t->dtype == RAD_FP4E2M1) ? 1
                      : (t->dtype == RAD_I2) ? 3 : 0;
        if (!sub || (base & sub) == 0) {
            row_to_f32(t->dtype, byte_at(t->data, t->dtype, base), dst, K);
            return;
        }
    }
    for (int64_t k = 0; k < K; ++k) dst[k] = ld_elem(t->data, t->dtype, base + k * m.cs);
}

static inline float dotf(const float* a, const float* b, int64_t k0, int64_t k1) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t k = k0;
    for (; k + 4 * VF_N <= k1; k += 4 * VF_N) {
        a0 = vf_fma(vf_loadu(a + k), vf_loadu(b + k), a0);
        a1 = vf_fma(vf_loadu(a + k + VF_N), vf_loadu(b + k + VF_N), a1);
        a2 = vf_fma(vf_loadu(a + k + 2 * VF_N), vf_loadu(b + k + 2 * VF_N), a2);
        a3 = vf_fma(vf_loadu(a + k + 3 * VF_N), vf_loadu(b + k + 3 * VF_N), a3);
    }
    for (; k + VF_N <= k1; k += VF_N) a0 = vf_fma(vf_loadu(a + k), vf_loadu(b + k), a0);
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; k < k1; ++k) s += a[k] * b[k];
    return s;
}

/* ------------------------------------------------------------------ router_topk */
/* Softmax over ALL n_expert logits, then the top_k by probability, then -- if `norm` --
 * renormalise the kept weights to sum to one. Softmax first and select second is the order
 * Mixtral, Qwen3 and every model in this engine's scope use; selecting first and normalising the
 * raw logits would be a different router, so the order is part of the op.
 *
 * The SELECTION runs on the LOGITS, not the probabilities: softmax is monotone, so the order is
 * identical and the comparison is exact rather than through an exp. */
AVX_KERNEL(router_topk) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *lg = &a->t[0], *ids = &a->t[1], *w = &a->t[2];
    const int64_t n_expert = p_int(a, "n_expert", lg->rank >= 2 ? lg->shape[lg->rank - 1] : 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    const int norm = (int)p_int(a, "norm", 0);
    if (n_expert <= 0 || top_k <= 0 || top_k > n_expert) return RAD_E_SHAPE;
    const int64_t M = numel(lg) / n_expert;
    if (M <= 0 || numel(ids) < M * top_k || numel(w) < M * top_k) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * n_expert >= 1024)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)n_expert);
        const float* v = in_row(lg, rowoff(lg, m, n_expert), n_expert, buf);
        const float mx = row_max(v, n_expert);
        float sum = 0.0f;
        for (int64_t e = 0; e < n_expert; ++e) sum += std::exp(v[e] - mx);
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

        float pv = INFINITY;
        int64_t pi = -1, kept = 0;
        float wsum = 0.0f;
        for (int64_t r = 0; r < top_k; ++r) {
            if (!topk_next_f32(v, n_expert, &pv, &pi)) break;
            const float p = std::exp(pv - mx) * inv;
            st_int_dt(ids->dtype, ids->data, rowoff(ids, m, top_k) + r * laststride(ids), pi);
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

/* A stable counting sort of the (token, slot) pairs by expert id, in ONE placement pass with a
 * per-expert write cursor rather than libref's expert-major rescan. The order is identical: within
 * an expert, rows are still appended in ascending i.
 *
 * Serial on purpose. It is O(T) over a batch's worth of small integers, and a parallel version
 * needs a per-thread histogram and a prefix pass to stay stable -- more code than the work it
 * saves, and a stability bug there routes a token to a different expert. */
AVX_KERNEL(moe_scatter) {
    if (!rad_args_have(a, 4)) return RAD_E_INVAL;
    const RadTensor *ids = &a->t[0], *sorted = &a->t[1], *off = &a->t[2], *cnt = &a->t[3];
    const int64_t n_expert = p_int(a, "n_expert", 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    /* THIS RANK'S FIRST EXPERT, subtracted from every routed id so the sort is over local indices
     * -- which is what the weight table, the offsets and the GEMM downstream are indexed by. */
    const int64_t base = p_int(a, "expert_base", 0);
    if (n_expert <= 0 || top_k <= 0 || base < 0) return RAD_E_SHAPE;
    const int64_t M = numel(ids) / top_k;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T) return RAD_E_SHAPE;
    if (numel(off) < n_expert + 1 || numel(cnt) < n_expert) return RAD_E_SHAPE;
    std::vector<int64_t> place;
    if (!sort_places(p_str(a, "protect", nullptr), n_expert, &place)) return RAD_E_SHAPE;
    /* A slot's place in the sort, or -1 for an expert outside this call's slice. */
    auto at = [&](int64_t i) -> int64_t {
        const int64_t e = ldi(ids, offlin(ids, i)) - base;
        return (e < 0 || e >= n_expert) ? -1 : place[(size_t)e];
    };

    int64_t* cur = (int64_t*)scratch_raw((size_t)(n_expert + 1) * sizeof(int64_t));
    for (int64_t e = 0; e < n_expert; ++e) cur[e] = 0;

    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e < 0) continue;
        ++cur[e];
    }
    for (int64_t e = 0; e < n_expert; ++e)
        st_int_dt(cnt->dtype, cnt->data, offlin(cnt, e), cur[e]);
    int64_t run = 0;
    for (int64_t e = 0; e < n_expert; ++e) {
        st_int_dt(off->dtype, off->data, offlin(off, e), run);
        const int64_t c = cur[e];
        cur[e] = run;                /* the cursor becomes the write position */
        run += c;
    }
    st_int_dt(off->dtype, off->data, offlin(off, n_expert), run);

    for (int64_t i = run; i < T; ++i)
        st_int_dt(sorted->dtype, sorted->data, offlin(sorted, i), -1);
    for (int64_t i = 0; i < T; ++i) {
        const int64_t e = at(i);
        if (e < 0) continue;
        st_int_dt(sorted->dtype, sorted->data, offlin(sorted, cur[e]), i);
        ++cur[e];
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ moe_gemm, moe_gemm_q */
/* y[i, :] = a[row(i), :] @ w[e]^T, where e is the expert whose offset range contains i and
 * `a_order` picks row(i): `sorted_tok[i] / top_k` when the activation has one row per TOKEN, and i
 * itself when it already has one per (token, slot). The output is in SORTED order, which is the
 * layout that makes the expert loop contiguous -- the entire point of the scatter. moe_gather puts
 * it back.
 *
 * `a_order` HAS TO BE SAID AND NOT DERIVED. The two grouped GEMMs of one routed block disagree
 * about it: gate_up reads the block's input, one row per token; down reads gate_up's output, one
 * row per (token, slot). The row count would give it away -- M against M*top_k -- but an operand
 * whose meaning changes with its extent is exactly the shape a checker cannot see: if both
 * implementations gather they agree with each other, and a harness that sizes the operand the way
 * the gather wants never exercises the other reading. */
static int moe_gemm_impl(const RadArgs* a, bool quant, int ncls = 2) {
    const int n_opd = quant ? (ncls == 4 ? 13 : 9) : 5;
    if (!rad_args_have(a, n_opd)) return RAD_E_INVAL;
    const RadTensor* A = &a->t[0];
    const RadTensor* as_ = quant ? rad_arg_in(a, 1) : nullptr;
    const RadTensor* W = &a->t[quant ? 2 : 1];
    const RadTensor* ws_ = quant ? rad_arg_in(a, 3) : nullptr;
    const RadTensor* sorted = &a->t[quant ? 4 : 2];
    const RadTensor* off = &a->t[quant ? 5 : 3];
    const RadTensor* Y = &a->t[quant ? (ncls == 4 ? 12 : 8) : 4];
    /* THE QUANTISED ARM'S EXPERTS ARE TWO TABLES BY PARITY, expert e at entry e / 2 of class
     * e % 2 -- or for `moe_gemm_q_mod4` four, entry e / 4 of class e % 4; the plain arm's are
     * one. */
    const bool two = quant;
    const int  nc  = quant ? ncls : 1;
    const RadTensor* W1 = two ? &a->t[6] : W;
    const RadTensor* ws1_ = two ? rad_arg_in(a, 7) : nullptr;
    const RadTensor* Wq[4]  = { W, W1, quant && nc == 4 ? &a->t[8] : W,
                                quant && nc == 4 ? &a->t[10] : W };
    const RadTensor* wsq[4] = { ws_, ws1_, quant && nc == 4 ? rad_arg_in(a, 9) : nullptr,
                                quant && nc == 4 ? rad_arg_in(a, 11) : nullptr };

    const char* ord = p_str(a, "a_order", "token");
    const bool sorted_a = ord && std::strcmp(ord, "sorted") == 0;
    if (ord && !sorted_a && std::strcmp(ord, "token") != 0) return RAD_E_UNSUPPORTED;

    const int64_t N0 = p_int(a, "N", W->rank >= 3 ? W->shape[1] : 0);
    const int64_t K0 = p_int(a, "K", W->rank >= 3 ? W->shape[2] : 0);
    /* Each class's geometry, defaulting to class 0's: N_odd/K_odd at two, N_q/K_q at four. */
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
    /* The output is the widest class's width, the activation the longest class's K; class q
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

    const int64_t M = T / top_k;
    const int64_t AR = sorted_a ? T : M;
    if (M <= 0 || numel(A) < AR * K) return RAD_E_SHAPE;

    const Mat ma = as_mat(A, K);
    const Scales SA = make_scales(as_, AR, K);
    const int64_t gp = p_int(a, "group", 0);
    if (quant && gp > 0 && SA.p && SA.group != gp) return RAD_E_SHAPE;

    const int64_t y_rs = row_stride(Y, N);
    const int64_t y_cs = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    /* ONE ARENA CALL FOR EVERYTHING THE PARALLEL REGIONS READ, and that is not tidiness. The
     * scratch arena GROWS -- a second scratch_raw at a larger size frees the first block and
     * returns a new one, so splitting this into two calls leaves every worker thread reading a
     * dangling expert table while the main thread walks a valid panel. The layout is computed
     * once, 64-aligned between sections so a vector load off the panel is legal. */
    const size_t ne = (size_t)n_expert;
    auto pad64 = [](size_t x) { return (x + 63) & ~(size_t)63; };
    const size_t off_we   = 0;
    const size_t off_wse  = pad64(off_we + ne * sizeof(RadTensor));
    const size_t off_mw   = pad64(off_wse + ne * sizeof(RadTensor));
    const size_t off_sw   = pad64(off_mw + ne * sizeof(Mat));
    const size_t off_pan  = pad64(off_sw + ne * sizeof(Scales));
    const size_t off_tok  = pad64(off_pan + (size_t)T * (size_t)K * sizeof(float));
    const size_t off_exp  = pad64(off_tok + (size_t)T * sizeof(int64_t));
    uint8_t* base = (uint8_t*)scratch_raw(off_exp + (size_t)T * sizeof(int64_t) + 64);
    RadTensor* we  = (RadTensor*)(base + off_we);
    RadTensor* wse = (RadTensor*)(base + off_wse);
    Mat*       mw  = (Mat*)(base + off_mw);
    Scales*    SW  = (Scales*)(base + off_sw);
    float*     apanel = (float*)(base + off_pan);
    int64_t*   atok = (int64_t*)(base + off_tok);
    int64_t*   aexp = (int64_t*)(base + off_exp);

    /* Per-expert readings, built once outside the loops. A null `data` means the operand could not
     * be read as a table at all, which is refused BEFORE any parallel region because an OpenMP
     * body cannot return. */
    for (int64_t e = 0; e < n_expert; ++e) {
        const int q = (int)(e % nc);
        const RadTensor* tw = Wq[q];
        const RadTensor* ts = wsq[q];
        const int64_t ix = e / nc;
        const int64_t n = Nq[q], k = Kq[q];
        we[e] = expert_entry(tw, ix);
        if (!we[e].data) return RAD_E_STRIDE;
        if (we[e].rank < 2 || we[e].shape[0] < n || we[e].shape[1] < k) return RAD_E_SHAPE;
        mw[e] = as_mat(&we[e], k);
        if (quant && ts) {
            wse[e] = expert_entry(ts, ix);
            if (!wse[e].data) return RAD_E_STRIDE;
            SW[e] = make_scales(&wse[e], n, k);
        } else {
            SW[e] = make_scales(nullptr, n, k);
        }
        /* An activation grid FINER than the weight's cannot have its scale hoisted out of the
         * group. No format in this project has one; refusing here, before any work, is the honest
         * answer and it cannot be done inside the loop because an OpenMP body cannot return. */
        const int64_t g = SW[e].p && SW[e].group > 0 ? SW[e].group : k;
        if (SA.p && SA.group < g) return RAD_E_UNSUPPORTED;
    }

    /* THE ACTIVATION PANEL. Every output row's source row is widened into f32 once, here, so the
     * dot products below never touch a dtype -- and so the N loop can read the same panel row for
     * all its outputs instead of re-converting it N times. Dropped slots are marked with a
     * negative token and their output row is zeroed rather than left as stale arena bytes. */
    AVX_PARALLEL_FOR_IF(T * K >= AVX_PAR_MIN)
    for (int64_t i = 0; i < T; ++i) {
        const int64_t flat = ldi(sorted, offlin(sorted, i));
        const int64_t tok = flat < 0 ? -1 : (sorted_a ? i : flat / top_k);
        atok[i] = (tok >= 0 && tok < AR) ? tok : -1;
        if (atok[i] >= 0) pack_row(ma, atok[i], K, apanel + i * K);
    }

    /* Which expert owns row i, walked once off the offsets rather than binary-searched per
     * (row, n) pair -- the rows are expert-sorted, so one linear sweep answers every row. */
    {
        int64_t e = 0;
        for (int64_t i = 0; i < T; ++i) {
            while (e + 1 < n_expert && ldi(off, offlin(off, e + 1)) <= i) ++e;
            aexp[i] = e;
        }
    }

    /* Output column n is weight row (n / fy) * fc + n % fy of its expert, and zero past a narrower
     * class's part. */
    const int64_t fy = N / parts;
    AVX_PARALLEL_FOR_IF(T * N * K >= AVX_PAR_MIN)
    for (int64_t n = 0; n < N; ++n) {
        float* bbuf = scratch_f32((size_t)K);
        int64_t packed_e = -1;
        const int64_t pp = n / fy, j = n - pp * fy;
        for (int64_t i = 0; i < T; ++i) {
            if (atok[i] < 0) { stt(Y, i * y_rs + n * y_cs, 0.0f); continue; }
            const int64_t e = aexp[i];
            const int q = (int)(e % nc);
            const int64_t fc = Nq[q] / parts, ke = Kq[q];
            if (j >= fc) { stt(Y, i * y_rs + n * y_cs, 0.0f); continue; }
            const int64_t wr = pp * fc + j;
            /* The expert's weight row is packed when the expert CHANGES, which -- because the
             * rows are expert-sorted -- is at most n_expert times over the whole i loop. That is
             * the win over walking (row, n) and re-reading the weight per row. */
            if (e != packed_e) { pack_row(mw[e], wr, ke, bbuf); packed_e = e; }
            const Scales& SB = SW[e];
            const int64_t g = SB.p && SB.group > 0 ? SB.group : ke;
            const int64_t ngrp = (ke + g - 1) / g;
            const float* av = apanel + i * K;
            float acc = 0.0f;
            for (int64_t gi = 0; gi < ngrp; ++gi) {
                const int64_t k0 = gi * g, k1 = (k0 + g < ke) ? k0 + g : ke;
                const float sb = SB.p ? SB.grp(wr, gi) : 1.0f;
                const float sa = (SA.p && SA.group > 0) ? SA.grp(atok[i], k0 / SA.group) : 1.0f;
                acc += dotf(av, bbuf, k0, k1) * sa * sb;
            }
            stt(Y, i * y_rs + n * y_cs, acc);
        }
    }
    return RAD_OK;
}

AVX_KERNEL(moe_gemm)   { return moe_gemm_impl(a, false); }
AVX_KERNEL(moe_gemm_q) { return moe_gemm_impl(a, true); }
AVX_KERNEL(moe_gemm_q_mod4) { return moe_gemm_impl(a, true, 4); }

/* ------------------------------------------------------------------ moe_gather */
/* y[t] = sum over the token's slots of expert_w[t, j] * y_expert[position of (t, j)].
 *
 * THE SUMMATION ORDER IS ASCENDING SLOT j, which is what makes this reproducible: walking the
 * expert-sorted rows instead would sum a token's contributions in expert-id order and the two
 * differ in the last bits.
 *
 * The inverse permutation is built ONCE over the whole batch. libref rebuilds it per block of
 * tokens into a fixed stack array specifically to avoid allocating, which costs (M / block) full
 * passes over sorted_tok; the scratch arena removes the reason for that. */
AVX_KERNEL(moe_gather) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *ye = &a->t[0], *ew = &a->t[1], *sorted = &a->t[2], *Y = &a->t[5];
    /* The shared arm, folded: y = narrow(shared * act(gate)) + narrow(gathered).
     * A pair or neither, exactly libref's opening lines. */
    const RadTensor *sh = rad_arg_in(a, 3), *sg = rad_arg_in(a, 4);
    if ((sh != nullptr) != (sg != nullptr)) return RAD_E_INVAL;
    const char* act = p_str(a, "act", "none");
    const bool sig = act && !std::strcmp(act, "sigmoid");
    if (sh && act && std::strcmp(act, "sigmoid") && std::strcmp(act, "none")) return RAD_E_INVAL;
    const int64_t n = p_int(a, "n", 0);
    const int64_t top_k = p_int(a, "top_k", 1);
    if (n <= 0 || top_k <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(Y) / n;
    const int64_t T = M * top_k;
    if (M <= 0 || numel(sorted) < T || numel(ew) < T || numel(ye) < T * n) return RAD_E_SHAPE;

    int32_t* inv = (int32_t*)scratch_raw((size_t)T * sizeof(int32_t) + 64);
    for (int64_t i = 0; i < T; ++i) inv[i] = -1;
    for (int64_t i = 0; i < T; ++i) {
        const int64_t f = ldi(sorted, offlin(sorted, i));
        if (f >= 0 && f < T) inv[f] = (int32_t)i;
    }

    const int64_t ye_rs = ye->rank >= 2 ? ye->stride[ye->rank - 2] : n;
    const int64_t ye_cs = ye->rank >= 2 ? ye->stride[ye->rank - 1] : 1;
    /* libref's Y addressing, for the fold below: stride[rank-2]/stride[rank-1], not the
     * rowoff/laststride reading out_row uses. The two agree on every rank-2 [M, n] this op is
     * described with; the fold uses this one so a hostile higher-rank Y still lands where the
     * reference puts it. */
    const int64_t y_rs = Y->rank >= 2 ? Y->stride[Y->rank - 2] : n;
    const int64_t y_cs = Y->rank >= 2 ? Y->stride[Y->rank - 1] : 1;

    AVX_PARALLEL_FOR_IF(M * n * top_k >= AVX_PAR_MIN)
    for (int64_t t = 0; t < M; ++t) {
        float* acc = scratch_f32((size_t)n);
        float* row = scratch_f32_b((size_t)n);
        for (int64_t c = 0; c < n; ++c) acc[c] = 0.0f;
        for (int64_t j = 0; j < top_k; ++j) {
            const int32_t i = inv[t * top_k + j];
            if (i < 0) continue;
            const float w = ldt(ew, rowoff(ew, t, top_k) + j * laststride(ew));
            const float* v;
            if (ye_cs == 1) {
                v = row_f32_in(ye->dtype, byte_at(ye->data, ye->dtype, (int64_t)i * ye_rs), row, n);
            } else {
                for (int64_t c = 0; c < n; ++c)
                    row[c] = rad_load_f32(ye->data, ye->dtype, (int64_t)i * ye_rs + c * ye_cs);
                v = row;
            }
            const vf vw = vf_set1(w);
            int64_t c = 0;
            for (; c + VF_N <= n; c += VF_N)
                vf_storeu(acc + c, vf_fma(vw, vf_loadu(v + c), vf_loadu(acc + c)));
            for (; c < n; ++c) acc[c] += w * v[c];
        }
        if (!sh) {
            out_row(Y, rowoff(Y, t, n), n, acc);
            continue;
        }
        /* THE NARROWS ARE THE UNFUSED PAIR'S, in libref's order: narrow the gathered row into Y,
         * narrow the shared product into Y, narrow their sum. The gate is one scalar per token,
         * through the same expf the oracle uses (see scale_rows on why not the vector form). */
        const float v0 = ldt(sg, t * laststride(sg));
        const float gate = sig ? 1.0f / (1.0f + std::exp(-v0)) : v0;
        const int64_t shb = rowoff(sh, t, n), shs = laststride(sh);
        for (int64_t c = 0; c < n; ++c) {
            const int64_t yo = t * y_rs + c * y_cs;
            stt(Y, yo, acc[c]);
            const float g = ldt(Y, yo);
            stt(Y, yo, ldt(sh, shb + c * shs) * gate);
            stt(Y, yo, ldt(Y, yo) + g);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ logit_rerank */
/* R candidate ids rescored against the full-precision head and re-sorted, the second half of a
 * coarse top-R. `idx_in` carries GLOBAL ids, `vocab_off` is this rank's first column,
 * and an id outside the head's rows becomes a pad that sorts last with -inf.
 *
 * THE DOTS ARE f32-VECTOR, the reference's are f64-scalar IN ORDER, and that is a tolerance
 * question rather than a defect: at K = 2048 the tree differs ~1e-6 relative and the op has no
 * tight entry, so it lands in the dtype bucket (1e-5 f32). What is EXACT is the selection --
 * descending by value, ties to the lower id, pads dropped -- copied rather than paraphrased,
 * because a tie resolved differently from row_topk is a silent routing divergence.
 *
 * `head` is read FLAT ([NV, K] row-major, unit stride), exactly as the reference reads it: the
 * row for candidate id g starts at g * K elements. That is the contract, not a convenience. */
static inline float dotf(const float* a, const float* b, int64_t n) {
    vf a0 = vf_zero(), a1 = vf_zero(), a2 = vf_zero(), a3 = vf_zero();
    int64_t k = 0;
    for (; k + 4 * VF_N <= n; k += 4 * VF_N) {
        a0 = vf_fma(vf_loadu(a + k), vf_loadu(b + k), a0);
        a1 = vf_fma(vf_loadu(a + k + VF_N), vf_loadu(b + k + VF_N), a1);
        a2 = vf_fma(vf_loadu(a + k + 2 * VF_N), vf_loadu(b + k + 2 * VF_N), a2);
        a3 = vf_fma(vf_loadu(a + k + 3 * VF_N), vf_loadu(b + k + 3 * VF_N), a3);
    }
    for (; k + VF_N <= n; k += VF_N) a0 = vf_fma(vf_loadu(a + k), vf_loadu(b + k), a0);
    float s = vf_hsum(vf_add(vf_add(a0, a1), vf_add(a2, a3)));
    for (; k < n; ++k) s += a[k] * b[k];
    return s;
}

AVX_KERNEL(logit_rerank) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *w = &a->t[1], *in = &a->t[2];
    const RadTensor *idx = &a->t[3], *val = &a->t[4];
    const RadTensor* pairs = (a->n_t > 5 && a->t[5].data) ? &a->t[5] : nullptr;
    const int64_t R    = p_int(a, "R", 1);
    const int64_t K    = p_int(a, "n_embd", 0);
    const int64_t NV   = p_int(a, "n_vocab", 0);
    const int64_t voff = p_int(a, "vocab_off", 0);
    const int64_t R_pairs = (pairs && pairs->rank >= 2 && pairs->shape[1] > 0) ? pairs->shape[1] : R;
    if (R <= 0 || K <= 0 || NV <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / K;
    if (M <= 0 || numel(in) < M * R) return RAD_E_SHAPE;
    if (numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && numel(pairs) < M * R_pairs * 2) return RAD_E_SHAPE;
    if (numel(w) < NV * K) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * R * K >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* xb = scratch_f32((size_t)K);
        float* wb = scratch_f32_b((size_t)K);
        float* sc = scratch_f32_c((size_t)R);
        const float* xv = in_row(x, rowoff(x, m, K), K, xb);
        /* ONE raw block, partitioned by hand: the arena has a single slot, so two get() calls
         * would hand back the SAME pointer and `used` would clobber `id`. */
        int32_t* id = (int32_t*)scratch_raw((size_t)R * sizeof(int32_t) + (size_t)R + 64);
        for (int64_t r = 0; r < R; ++r) {
            const int64_t g = ldi(in, rowoff(in, m, R) + r * laststride(in));
            const int64_t local = g - voff;
            if (g < 0 || local < 0 || local >= NV) {
                id[r] = -1;
                sc[r] = -INFINITY;
                continue;
            }
            row_to_f32(w->dtype, byte_at(w->data, w->dtype, local * K), wb, K);
            id[r] = (int32_t)g;
            sc[r] = dotf(xv, wb, K);
        }
        uint8_t* used = (uint8_t*)(id + R);
        for (int64_t r = 0; r < R; ++r) used[r] = 0;
        for (int64_t o = 0; o < R; ++o) {
            int64_t best = -1;
            for (int64_t c = 0; c < R; ++c) {
                if (used[c] || id[c] < 0) continue;
                if (best < 0 || sc[c] > sc[best] ||
                    (sc[c] == sc[best] && id[c] < id[best])) best = c;
            }
            const int32_t oid = best >= 0 ? id[best] : -1;
            const float ov = best >= 0 ? sc[best] : -INFINITY;
            if (best >= 0) used[best] = 1;
            st_int_dt(idx->dtype, idx->data, rowoff(idx, m, R) + o * laststride(idx), oid);
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

/* ------------------------------------------------------------------ row_topk */
/* Exact per-row top-R, descending, ties to the lower column. The same selection the router uses,
 * exposed as its own op because a drafter's candidate set and a speculative tree both want it
 * without a softmax in front.
 *
 * The vocab-parallel form: `vocab_off` shifts every column reported so a rank's slice reports
 * GLOBAL ids, and `pairs` is the [M, R, 2] plane an all_gather moves -- an int32 id bit-copied
 * into the first float slot beside its value. Both absent is the ordinary single-rank top-R. */
AVX_KERNEL(row_topk) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
    const RadTensor *x = &a->t[0], *idx = &a->t[1], *val = &a->t[2];
    const RadTensor* pairs = rad_arg_in(a, 3);
    const int64_t off = p_int(a, "vocab_off", 0);
    const int64_t N = p_int(a, "N", x->rank >= 2 ? x->shape[x->rank - 1] : 0);
    const int64_t R = p_int(a, "R", 1);
    if (N <= 0 || R <= 0) return RAD_E_SHAPE;
    const int64_t M = numel(x) / N;
    if (M <= 0 || numel(idx) < M * R || numel(val) < M * R) return RAD_E_SHAPE;
    if (pairs && (pairs->dtype != RAD_F32 || numel(pairs) < M * R * 2)) return RAD_E_SHAPE;

    AVX_PARALLEL_FOR_IF(M * N >= AVX_PAR_MIN)
    for (int64_t m = 0; m < M; ++m) {
        float* buf = scratch_f32((size_t)N);
        const float* v = in_row(x, rowoff(x, m, N), N, buf);
        float pv = INFINITY;
        int64_t pi = -1;
        for (int64_t r = 0; r < R; ++r) {
            const bool ok = topk_next_f32(v, N, &pv, &pi);
            const int64_t id = ok ? pi + off : -1;
            st_int_dt(idx->dtype, idx->data, rowoff(idx, m, R) + r * laststride(idx), id);
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
/* The R largest of world_size all-gathered per-rank candidate sets, in the same descending
 * (idx, val) form the single-rank path produces.
 *
 * TIES BREAK ON THE LOWER ID, which is what makes the merge EXACT rather than approximate: a dense
 * top-R over the whole vocabulary breaks ties on the lower column, so a merge that broke them on
 * rank order would disagree with it exactly when two shards hold the same logit.
 *
 * A slot the producing rank could not fill carries id -1 and is DROPPED -- a sentinel rather than
 * a -inf value, because -inf is a value a masked logit legitimately takes.
 *
 * LEFT SCALAR, AND THE REASON IS THE SIZE. W*R is 32 entries at world 2 and R 16; the selection
 * sort over it is 512 comparisons against a vector setup that would cost more than it saves, and
 * the tie rule is a lexicographic compare that no vector max expresses. */
AVX_KERNEL(row_topk_merge) {
    if (!rad_args_have(a, 3)) return RAD_E_INVAL;
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

    AVX_PARALLEL_FOR_IF(M * W * R * R >= 4096)
    for (int64_t m = 0; m < M; ++m) {
        const float* const row = (const float*)g->data + m * W * R * 2;
        unsigned char* used = (unsigned char*)scratch_f32((size_t)(W * R) / 4 + 4);
        for (int64_t c = 0; c < W * R; ++c) used[c] = 0;
        for (int64_t r = 0; r < R; ++r) {
            int64_t best = -1;
            float bv = 0.0f;
            int32_t bid = 0;
            for (int64_t c = 0; c < W * R; ++c) {
                if (used[c]) continue;
                int32_t id;
                std::memcpy(&id, row + c * 2, sizeof id);
                if (id < 0) continue;                      /* a pad, not a candidate */
                const float v = row[c * 2 + 1];
                if (best < 0 || v > bv || (v == bv && id < bid)) { best = c; bv = v; bid = id; }
            }
            if (best >= 0) used[best] = 1;
            st_int_dt(idx->dtype, idx->data, rowoff(idx, m, ld) + r * laststride(idx),
                      best >= 0 ? (int64_t)bid : -1);
            stt(val, rowoff(val, m, ld) + r * laststride(val), best >= 0 ? bv : -INFINITY);
        }
    }
    return RAD_OK;
}
