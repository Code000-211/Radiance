/* libref/ref_tree_state.h -- the CPU oracle for parent-indexed tree serving of the GDN block.
 *
 * This is NOT an op. It is a header-only, allocation-free scalar reference for wave 1 of tree
 * serving (reports/drafter_optimization_20261013/TREE_PLAN.md, "state" lane): given a candidate
 * tree of at most 16 nodes with an explicit root-inclusive parent topology, it computes each
 * node's depthwise-convolution outputs, rolling conv history and gated-delta-net state/output
 * FROM ITS PARENT'S STATE, and maps an accepted winning path back onto the committed state.
 * Nothing here is registered in ref_registry.cpp, has a schema, or is reachable by rad-kbench;
 * the GPU wave turns the factor contract below into a separately named op with its own libref
 * line or truthful replaces hook, exactly as the plan requires.
 *
 * ============================ THE ARITHMETIC (stated, not invented) ============================
 *
 * Every recurrence below is ref_gdn.cpp's decode path restated per node. Per token, per v head,
 * with the state S shaped [head_v, head_k]:
 *
 *     g    = -exp(A_log) . softplus(a + dt_bias)        beta = sigmoid(b)
 *     qn   = q . rsqrt(sum(q^2) + l2_eps) . scale       kn   = k . rsqrt(sum(k^2) + l2_eps)
 *     S   *= e^g
 *     u    = beta . (v - S kn)                          S += u kn^T          o = S qn
 *     o    = o . rsqrt(mean(o^2) + eps) . norm_w . act(z)
 *
 * The operation ORDER is ref_gdn_recurrent_update's own: decay and the dot in one pass per state
 * row, the rank-1 update and the output dot in a second, then the gated RMS norm over the head's
 * head_v channels. The convolution is gdn_conv_update's: a causal width-`conv_width` depthwise
 * window over RAW projection values (no activation before the taps), silu after, bias added
 * before the taps. The l2 norm is rsqrt(sum + l2_eps) with NO division by the head width. All
 * accumulation is f32; compile with -ffp-contract=off as libref does if bit-exactness against
 * the plugin matters (see the test's tolerance note).
 *
 * ======================= TOPOLOGY =======================
 *
 * `parent[i] < i` for every node but the root, whose `parent[root] == -1`. Nodes may appear in
 * any topological order -- sibling numbering, depth ordering and path contiguity are all free,
 * and NOTHING here may derive a node's parent from its index. The root is the anchor token
 * processed in this step from `state0`/`hist0` (the committed state entering the step).
 * Committing with zero accepted draft children still commits this root anchor (replay length 1),
 * advancing the committed state to the root's slot.
 *
 * ======================= BUFFERS (caller-owned, fixed size) =======================
 *
 * No function here allocates, syncs, or keeps state between calls. Scratch sizes:
 *
 *   q, k        n * n_head_k * head_k          v, o        n * n_head_v * head_v
 *   a, b        n * n_head_v                   A_log,dt_bias  n_head_v (or null)
 *   z           n * n_head_v * head_v (or null)  norm_w     head_v (or null)
 *   states      n * n_head_v * head_v * head_k   state0     n_head_v * head_v * head_k
 *   hist        n * conv_dim * (conv_width-1)    hist0      conv_dim * (conv_width-1)
 *   factors     n * n_head_v * (head_k+head_v+2)
 *
 * Inputs and outputs must not alias except where a function says so (the commit and replay
 * functions accept state_out == state0 and hist_out == hist0).
 *
 * ======================= THE FACTOR CONTRACT (for the GPU wave) =======================
 *
 * THE GPU DOES NOT NEED A STATE POOL PER NODE. The delta step's state effect factorises: a node
 * contributes, per v head, exactly
 *
 *     kn[head_k]  (k l2-normalised by rsqrt(sum k^2 + l2_eps), NO scale)
 *     v[head_v]   (raw)
 *     g           (the decay log-factor as computed, before exp)
 *     beta
 *
 * and folding a chain of such factor rows onto any state reproduces the per-node states
 * BITWISE, because rad_gdn_state_step() below is the one code path both the direct walk and the
 * replay call. This is the anchor-replay contract libr4d's
 * r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip already implements for the linear path:
 * its factor row stride is RU_FSTR = 4*head_k + head_v + 8 -- this row with kn written four
 * times (one copy per 32-row group, at offset grp*head_k), v at 4*head_k+row, g and beta at
 * 4*head_k+head_v+2*grp and +1, plus a one-based page-shift marker at 16*RU_FSTR. The MINIMAL
 * canonical row here is the un-duplicated one, len head_k + head_v + 2. A tree kernel replays
 * the accepted path's factors onto the one committed state; scratch per (node, head) is that
 * row, so 16 nodes cost 16*(K+V+2) floats a head against 16*K*V for a state pool -- and the
 * committed state stays ONE slot a sequence whatever the topology.
 *
 * The convolution does not factor: its "state" is the raw-x window, conv_dim*(conv_width-1)
 * floats, but only the accepted node's WINDOW is ever committed, so a tree kernel needs the
 * window beside the factors, not a pool per node.
 */
#ifndef RAD_REF_TREE_STATE_H
#define RAD_REF_TREE_STATE_H

#include "rad_abi.h"

#include <math.h>
#include <string.h>

extern "C" {

#define RAD_TREE_MAX_NODES  16      /* CandidateTree's existing capacity, root included */
#define RAD_TREE_HIST_MAX   16      /* deepest conv history held, ref_gdn.cpp's own bound */

struct RadTreeGdn {
    int   head_k, head_v;           /* K, V: the per-head widths */
    int   n_head_k, n_head_v;       /* Hg, Hv: head counts, Hv % Hg == 0 */
    float scale;                    /* the attention scale folded into qn */
    float l2_eps;                   /* the q/k l2 norm epsilon (1e-6f in ref) */
    float norm_eps;                 /* the gated RMS norm epsilon */
    float softplus_thr;             /* softplus linear region threshold (20.0f in ref) */
    int   act;                      /* output gate: 0 = silu, 1 = sigmoid (ref's p_enum) */
};

/* ------------------------------------------------------------------ topology */
/* RAD_OK, or RAD_E_INVAL/RAD_E_SHAPE by name: n in [1, RAD_TREE_MAX_NODES], exactly one root
 * (parent == -1), every other parent in [0, i). parent[i] >= i is a forward edge, which in a
 * parent-indexed tree is a cycle and is refused rather than walked. */
static inline int rad_tree_check_topo(const int* parent, int n) {
    if (!parent || n < 1 || n > RAD_TREE_MAX_NODES) return RAD_E_INVAL;
    int roots = 0;
    for (int i = 0; i < n; ++i) {
        if (parent[i] == -1) { ++roots; continue; }
        if (parent[i] < 0 || parent[i] >= i) return RAD_E_INVAL;
    }
    return roots == 1 ? RAD_OK : RAD_E_INVAL;
}

static inline int rad_tree_cfg_ok(const struct RadTreeGdn* g) {
    return g && g->head_k > 0 && g->head_v > 0 && g->n_head_k > 0 && g->n_head_v > 0 &&
           g->n_head_v % g->n_head_k == 0;
}

/* depth[i]: the root is 0. Caller owns depth[n]. */
static inline void rad_tree_depths(const int* parent, int n, int* depth) {
    for (int i = 0; i < n; ++i)
        depth[i] = parent[i] < 0 ? 0 : depth[parent[i]] + 1;
}

/* ------------------------------------------------------------------ activations */
/* ref_common.h's definitions, restated so this header stands alone. */
static inline float rad_tree_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float rad_tree_silu(float x)    { return x * rad_tree_sigmoid(x); }
static inline float rad_tree_softplus(float x, float thr) {
    if (x > thr) return x;
    return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
}

/* ------------------------------------------------------------------ the shared delta step */
/* THE ONE STATE UPDATE BOTH PATHS CALL, so the replay and the direct walk cannot drift. Per
 * (state row e): phase 1 decays and dots, phase 2 applies the rank-1 update; the output dot
 * rides phase 2 exactly as ref_gdn_recurrent_update's does. qn == null skips only the output,
 * never the state. `dec` and `bt` are passed precomputed so the factor path can supply them
 * from a stored g/beta and still produce the identical sequence. */
static inline void rad_gdn_state_step(float* S, const float* kn, const float* vraw,
                                      const float* qn, float* o_or_null, float dec, float bt,
                                      int K, int V) {
    for (int e = 0; e < V; ++e) {
        float sk2 = 0.0f;
        for (int d = 0; d < K; ++d) {
            S[e * K + d] *= dec;
            sk2 += S[e * K + d] * kn[d];
        }
        const float u = bt * (vraw[e] - sk2);
        float ss = 0.0f;
        for (int d = 0; d < K; ++d) {
            S[e * K + d] += u * kn[d];
            if (qn) ss += S[e * K + d] * qn[d];
        }
        if (o_or_null) o_or_null[e] = ss;
    }
}

/* qn/kn exactly as the decode kernel forms them: the l2 norm over the head, no width division,
 * the scale folded into q's factor only. */
static inline void rad_gdn_norm_qk(const float* q, const float* k, float* qn, float* kn,
                                   float scale, float l2_eps, int K) {
    float sq = 0.0f, sk = 0.0f;
    for (int d = 0; d < K; ++d) { sq += q[d] * q[d]; sk += k[d] * k[d]; }
    const float qs = 1.0f / sqrtf(sq + l2_eps) * scale;
    const float ks = 1.0f / sqrtf(sk + l2_eps);
    for (int d = 0; d < K; ++d) { qn[d] = q[d] * qs; kn[d] = k[d] * ks; }
}

/* ------------------------------------------------------------------ the convolution */
/* Per node: the depthwise causal conv over the PARENT's rolling window (the root's is `hist0`,
 * or zeros when hist0 == null -- ref's cold `has_init` case), silu applied, and the node's own
 * window written for its children. x rows are RAW projection values: the taps run on x itself,
 * the activation after. q/k/v land in ref's split layout. */
static inline int rad_tree_conv(const struct RadTreeGdn* g, int conv_width, int conv_dim,
                                const int* parent, int n,
                                const float* x, const float* conv_w, const float* conv_b,
                                const float* hist0,
                                float* q, float* k, float* v, float* hist) {
    if (!rad_tree_cfg_ok(g)) return RAD_E_SHAPE;
    const int topo = rad_tree_check_topo(parent, n);
    if (topo != RAD_OK) return topo;
    if (conv_width < 1 || conv_width - 1 > RAD_TREE_HIST_MAX) return RAD_E_UNSUPPORTED;
    const int64_t cd = 2 * (int64_t)g->n_head_k * g->head_k + (int64_t)g->n_head_v * g->head_v;
    if (conv_dim != cd) return RAD_E_SHAPE;
    if (!x || !conv_w || !q || !k || !v) return RAD_E_INVAL;
    if (conv_width > 1 && !hist) return RAD_E_INVAL;   /* no history to keep means no buffer;
                                                        * a warm step needs its window, and a
                                                        * cold one passes a zeroed one */
    if (conv_width > 1 && !hist0) return RAD_E_INVAL;   /* a warm step needs a window; use zeros
                                                         * by passing one, not by inventing it */

    const int hist_len = conv_width - 1;
    for (int i = 0; i < n; ++i) {
        const float* xrow = x + (size_t)i * conv_dim;
        const float* win = hist_len == 0 ? (const float*)0 : (parent[i] < 0
            ? hist0
            : hist + ((size_t)parent[i] * conv_dim) * hist_len);
        float* myhist = hist_len == 0 ? (float*)0 : hist + ((size_t)i * conv_dim) * hist_len;
        for (int c = 0; c < conv_dim; ++c) {
            const float bs = conv_b ? conv_b[c] : 0.0f;
            float acc = bs;
            for (int j = 0; j < conv_width; ++j) {
                /* tap j covers the value width-1-j places back: x itself for j == width-1 */
                const float val = j == conv_width - 1
                    ? xrow[c]
                    : (win ? win[(size_t)c * hist_len + j] : 0.0f);
                acc += val * conv_w[(size_t)c * conv_width + j];
            }
            const float o = rad_tree_silu(acc);
            float* dst;
            int64_t off;
            if (c < g->n_head_k * g->head_k) {
                dst = q; off = (size_t)i * g->n_head_k * g->head_k + c;
            } else if (c < 2 * g->n_head_k * g->head_k) {
                dst = k; off = (size_t)i * g->n_head_k * g->head_k + (c - g->n_head_k * g->head_k);
            } else {
                dst = v;
                off = (size_t)i * g->n_head_v * g->head_v +
                      (c - 2 * g->n_head_k * g->head_k);
            }
            dst[off] = o;
            if (hist_len > 0)
                myhist[(size_t)c * hist_len + hist_len - 1] = xrow[c];
        }
        /* the window in front of this node's own value is the parent's, shifted by one */
        for (int c = 0; c < conv_dim && hist_len > 1; ++c)
            for (int j = 0; j < hist_len - 1; ++j)
                hist[((size_t)i * conv_dim + c) * hist_len + j] =
                    win ? win[(size_t)c * hist_len + j + 1] : 0.0f;
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ the delta net */
/* Per node, per v head: the state comes from the PARENT's slot in `states` (the root's from
 * state0), the row is this node's, and the result lands in the node's own slot. `states` is
 * read and written in one pass, so a child's slot is only read after its parent wrote it --
 * which the topological parent order guarantees. */
static inline int rad_tree_gdn(const struct RadTreeGdn* g, const int* parent, int n,
                               const float* q, const float* k, const float* v,
                               const float* a, const float* b,
                               const float* A_log, const float* dt_bias,
                               const float* state0,
                               const float* z, const float* norm_w,
                               float* o, float* states) {
    if (!rad_tree_cfg_ok(g)) return RAD_E_SHAPE;
    const int topo = rad_tree_check_topo(parent, n);
    if (topo != RAD_OK) return topo;
    if (!q || !k || !v || !a || !b || !state0 || !o || !states) return RAD_E_INVAL;
    if (g->head_k > 512 || g->head_v > 512) return RAD_E_UNSUPPORTED;

    const int K = g->head_k, V = g->head_v, Hg = g->n_head_k, Hv = g->n_head_v;
    const int grp = Hv / Hg;
    const size_t head_state = (size_t)V * K, slot = (size_t)Hv * head_state;

    for (int i = 0; i < n; ++i) {
        const float* s_in = parent[i] < 0 ? state0 : states + (size_t)parent[i] * slot;
        float* s_out = states + (size_t)i * slot;
        memcpy(s_out, s_in, sizeof(float) * slot);
        for (int h = 0; h < Hv; ++h) {
            const int hk = h / grp;
            const float alog = A_log ? A_log[h] : 0.0f;
            const float dtb  = dt_bias ? dt_bias[h] : 0.0f;
            const float gv = -expf(alog) * rad_tree_softplus(a[(size_t)i * Hv + h] + dtb,
                                                             g->softplus_thr);
            const float bt = rad_tree_sigmoid(b[(size_t)i * Hv + h]);
            float qnv[512], knv[512];      /* head widths <= 512, refused by name above */
            rad_gdn_norm_qk(q + ((size_t)i * Hg + hk) * K, k + ((size_t)i * Hg + hk) * K,
                            qnv, knv, g->scale, g->l2_eps, K);
            float orow[512];
            rad_gdn_state_step(s_out + (size_t)h * head_state, knv,
                               v + ((size_t)i * Hv + h) * V, qnv, orow,
                               expf(gv), bt, K, V);
            /* the gated RMS norm, over the head's V channels, in ref's own order */
            float sq2 = 0.0f;
            for (int e = 0; e < V; ++e) sq2 += orow[e] * orow[e];
            const float sc = 1.0f / sqrtf(sq2 / (float)V + g->norm_eps);
            for (int e = 0; e < V; ++e) {
                float r = orow[e] * sc;
                if (norm_w) r *= norm_w[e];
                if (z) {
                    const float zz = z[((size_t)i * Hv + h) * V + e];
                    r *= g->act == 1 ? rad_tree_sigmoid(zz) : rad_tree_silu(zz);
                }
                o[((size_t)i * Hv + h) * V + e] = r;
            }
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ factors */
static inline int rad_tree_factor_len(const struct RadTreeGdn* g) {
    return g ? g->head_k + g->head_v + 2 : 0;
}

/* Record every node's factor row, per v head: [ kn | v | g | beta ]. The values are the SAME
 * floats the direct walk formed (the norm and the softplus chain are the shared helpers), which
 * is what makes the replay bitwise. */
static inline int rad_tree_pack_factors(const struct RadTreeGdn* g, const int* parent, int n,
                                        const float* k, const float* v,
                                        const float* a, const float* b,
                                        const float* A_log, const float* dt_bias,
                                        float* factors) {
    if (!rad_tree_cfg_ok(g)) return RAD_E_SHAPE;
    const int topo = rad_tree_check_topo(parent, n);
    if (topo != RAD_OK) return topo;
    if (!k || !v || !a || !b || !factors) return RAD_E_INVAL;
    if (g->head_k > 512 || g->head_v > 512) return RAD_E_UNSUPPORTED;
    const int K = g->head_k, V = g->head_v, Hg = g->n_head_k, Hv = g->n_head_v;
    const int grp = Hv / Hg, len = rad_tree_factor_len(g);

    for (int i = 0; i < n; ++i)
        for (int h = 0; h < Hv; ++h) {
            float* f = factors + ((size_t)i * Hv + h) * len;
            const int hk = h / grp;
            float qn[512], knv[512];                   /* qn unused here; the norm is shared */
            rad_gdn_norm_qk(k + ((size_t)i * Hg + hk) * K, k + ((size_t)i * Hg + hk) * K,
                            qn, knv, 1.0f, g->l2_eps, K);
            memcpy(f, knv, sizeof(float) * K);
            memcpy(f + K, v + ((size_t)i * Hv + h) * V, sizeof(float) * V);
            const float alog = A_log ? A_log[h] : 0.0f;
            const float dtb  = dt_bias ? dt_bias[h] : 0.0f;
            f[K + V] = -expf(alog) * rad_tree_softplus(a[(size_t)i * Hv + h] + dtb,
                                                       g->softplus_thr);
            f[K + V + 1] = rad_tree_sigmoid(b[(size_t)i * Hv + h]);
        }
    return RAD_OK;
}

/* Fold a chain of factor rows onto a state. `chain` is node ids, root first, each in
 * [0, n_nodes); state_out may alias state0. Invalid inputs leave state_out unchanged.
 * This is the commit path a GPU kernel runs against its ONE anchor state, and the reference
 * any per-parent reconstruction is checked against. */
static inline int rad_tree_replay_factors(const struct RadTreeGdn* g,
                                          const int* chain, int chain_len, int n_nodes,
                                          const float* factors, const float* state0,
                                          float* state_out) {
    if (!rad_tree_cfg_ok(g)) return RAD_E_SHAPE;
    if (chain_len < 0 || chain_len > n_nodes || n_nodes > RAD_TREE_MAX_NODES) return RAD_E_INVAL;
    if (!factors || !state0 || !state_out) return RAD_E_INVAL;
    if (chain_len > 0 && !chain) return RAD_E_INVAL;
    if (g->head_k > 512 || g->head_v > 512) return RAD_E_UNSUPPORTED;
    for (int c = 0; c < chain_len; ++c)
        if (chain[c] < 0 || chain[c] >= n_nodes) return RAD_E_INVAL;
    const int K = g->head_k, V = g->head_v, Hv = g->n_head_v;
    const int len = rad_tree_factor_len(g);
    const size_t head_state = (size_t)V * K, slot = (size_t)Hv * head_state;

    if (state_out != state0)
        memcpy(state_out, state0, sizeof(float) * slot);
    for (int c = 0; c < chain_len; ++c) {
        for (int h = 0; h < Hv; ++h) {
            const float* f = factors + ((size_t)chain[c] * Hv + h) * len;
            rad_gdn_state_step(state_out + (size_t)h * head_state, f, f + K,
                               (const float*)0, (float*)0, expf(f[K + V]), f[K + V + 1], K, V);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ commit */
struct RadTreeCommit {
    int node;        /* the node whose state becomes committed: path[accepted_children] */
    int replay_len;  /* replay length: accepted_children + 1 */
    int read_row;    /* the engine's 1-based num_accepted for the NEXT step under the
                      * anchor-row-inclusive layout: accepted_children + 1. Next step read offset is
                      * read_row - 1 = accepted_children. */
};

/* `accepted_children` is the number of winning-path draft children accepted, in [0, path_len - 1].
 * CRITICAL (TREE_PLAN.md): zero accepted draft children (accepted_children == 0) still commits
 * the root anchor (path[0]), with replay length 1 (accepted_children + 1).
 * Any newly sampled bonus token (or EOG promoted to bonus) has not been forwarded through the
 * recurrent net in this step and is excluded from commit/replay.
 * `path` must be a valid root-to-node chain: path[0] is root (parent == -1) and every subsequent
 * step's parent is the node before it in path. */
static inline int rad_tree_commit_index(const int* parent, int n,
                                        const int* path, int path_len, int accepted_children,
                                        struct RadTreeCommit* out) {
    const int topo = rad_tree_check_topo(parent, n);
    if (topo != RAD_OK) return topo;
    if (!out || !path || path_len < 1 || path_len > n ||
        accepted_children < 0 || accepted_children >= path_len)
        return RAD_E_INVAL;
    for (int i = 0; i < path_len; ++i) {
        if (path[i] < 0 || path[i] >= n) return RAD_E_INVAL;
        const int expect = i == 0 ? -1 : path[i - 1];
        if (parent[path[i]] != expect) return RAD_E_INVAL;
    }
    out->node = path[accepted_children];
    out->replay_len = accepted_children + 1;
    out->read_row = accepted_children + 1;
    return RAD_OK;
}

/* Copy the committed state and conv window the acceptance picked.
 * Zero accepted draft children (accepted_children == 0) commits the root anchor (path[0]),
 * with replay length 1 (accepted_children + 1).
 * New bonus / EOG promoted to bonus is not forwarded, excluded from committed state.
 * state_out may alias state0 and hist_out may alias hist0; both are fully written.
 * Width 1 has no history: hist0, hist and hist_out may all be null. */
static inline int rad_tree_commit(const struct RadTreeGdn* g, int conv_width, int conv_dim,
                                  const int* parent, int n,
                                  const int* path, int path_len, int accepted_children,
                                  const float* state0, const float* hist0,
                                  const float* states, const float* hist,
                                  float* state_out, float* hist_out) {
    if (!rad_tree_cfg_ok(g)) return RAD_E_SHAPE;
    struct RadTreeCommit ci;
    const int rc = rad_tree_commit_index(parent, n, path, path_len, accepted_children, &ci);
    if (rc != RAD_OK) return rc;
    if (conv_width < 1 || conv_width - 1 > RAD_TREE_HIST_MAX) return RAD_E_UNSUPPORTED;
    if (conv_dim < 1) return RAD_E_SHAPE;
    if (!states || !state0 || !state_out) return RAD_E_INVAL;
    const int hl = conv_width - 1;
    if (hl > 0 && (!hist0 || !hist || !hist_out)) return RAD_E_INVAL;

    const size_t slot = (size_t)g->n_head_v * g->head_v * g->head_k;
    const float* src_s = states + (size_t)ci.node * slot;
    if (state_out != src_s)
        memcpy(state_out, src_s, sizeof(float) * slot);
    if (hl > 0) {
        const float* src_h = hist + (size_t)ci.node * conv_dim * hl;
        if (hist_out != src_h)
            memcpy(hist_out, src_h, sizeof(float) * ((size_t)conv_dim * hl));
    }
    return RAD_OK;
}

}  /* extern "C" */

#endif  /* RAD_REF_TREE_STATE_H */
