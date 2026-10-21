/* bole_tree.hip -- the Bole tree-GDN state update for gfx1201, as the two device rows of one op,
 * plus the launches the host row table calls.
 *
 * WHY THIS OP DOES NOT EXIST IN libr4d. The engine's gdn_conv_recurrent_update is a serial
 * recurrence: it walks a sequence's tokens in order, keeps one FP32 state in registers and writes a
 * state per candidate. A speculative tree needs operands that op has no slot for -- a PARENT[] array
 * naming each candidate's parent, per-NODE factors and a per-node output -- so several candidates
 * that share an ancestor can be evaluated against that ancestor instead of against each other.
 * ABI 26 has no tree seam either (core/sched/batch.cpp refuses tree-layout speculation outright), so
 * no model reaches this op by accident: it is selected only by a caller that names this plugin. It
 * never shadows libr4d's row, because it does not name the same op.
 *
 * WHAT BOLE CHANGES. Only the recurrence solver. The paper (arXiv:2608.01651v1, IV-A) observes that a
 * node's state after a run of rank-one updates can be read off its ancestors without replaying them.
 * This is the `vb-exact-commit` variant of attic/2026-10-20_bole-vb-exact-commit (ABI 16): the newest
 * of the three archived variants and the only one whose commit was byte-exact. The README says why
 * adjoint and Neumann are superseded.
 *
 * THREAD SHAPE. Both kernels use libr4d's own: 256 threads, so thread tid owns value row tid >> 1 and
 * the 64-column half (tid & 1) * 64 of the k range, and a row's two halves close a dot with ONE
 * lane^1 exchange. This is not cosmetic. The source's tree kernel used a 512-thread quarter-row
 * shape with two exchanges; that associates every sum differently and would make this row's
 * normalized k -- and therefore its factors, and therefore the committed bytes -- differ from
 * libr4d's in the last bits. Keeping the engine's shape is what lets the commit row's byte claim be
 * true, and it is the one place the port deliberately departs from the source's layout.
 *
 * The expressions are libr4d's too, not just their order: `dot += h*k`, `h += u*k`, `sq += x*x` are
 * separate multiply and add, because the plugin is compiled -ffp-contract=off as libr4d's targets are
 * (cmake/RadAddPlugin.cmake, rad_hipify). Writing fmaf here would round once where the engine rounds
 * twice and change the last bits of u.
 *
 * Everything that is NOT the solver is the engine's arithmetic, restated from libr4d:
 *
 *     conv   : acc = bias; acc += w[i]*win[i], i = 0..3; one bf16 rounding of cp_silu(acc)
 *     norm   : q,k l2 with eps 1e-6, 64-term sum + lane^1, q scaled by `scale`
 *     decay  : g = -e^A_log * softplus(a + dt_bias, thr); beta = sigmoid(b)
 *     update : h *= e^g; dot = h.k; u = (v - dot)*beta; h += u k^T; o = h.q
 *     output : gated rmsnorm over 128 channels, one bf16 rounding, then the INT8 group-128 epilogue
 *
 * Below the root, Bole reorders ACCUMULATION: the walks, the ancestor contractions and the rounded-u
 * feedback are FP64. The root is a fixed one-node serial FP32 prefix, so the root's output and the
 * factors (which are exactly libr4d's) are the serial values; the non-root OUTPUTS are not, and the
 * verify row declares that. The commit row is byte-exact because it recomputes u from those factors
 * instead of replaying a stored U.
 */
#include "bole_device.h"
#include "bole_layout.h"

#include "rad_abi.h"
#include "rad_plugin.h"

#include <hip/hip_runtime.h>

/* libr4d's recurrent-block shape: 128 value rows, two k halves (BOLE_THR, in bole_layout.h so the
 * host row can bound the projection width the history arm covers). */
#define BOLE_HK 64

static __device__ __forceinline__ int64_t boff(const BoleCall& f, int t, int64_t a, int64_t b = 0,
                                               int64_t c = 0, int64_t d = 0) {
    return a * f.s[t][0] + b * f.s[t][1] + c * f.s[t][2] + d * f.s[t][3];
}
static __device__ __forceinline__ int bint(const BoleCall& f, int t, int a, int b = 0) {
    return static_cast<const int*>(f.t[t])[boff(f, t, a, b)];
}
static __device__ __forceinline__ float bflt(const BoleCall& f, int t, int a, int b = 0, int c = 0) {
    return static_cast<const float*>(f.t[t])[boff(f, t, a, b, c)];
}
static __device__ __forceinline__ float bsp(float x, float thr) {
    if (x > thr) return x;
    return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}
/* The group a value head belongs to: `hv / (H / Hg)` in libr4d, i.e. nhv/nkg value heads per key
 * head. The head WIDTHS are separate (both 128); using a width here would divide 128 by 1 and index
 * the projection row from channel 12800. */
static __device__ __forceinline__ int bole_khead(const BoleCall& f, int head) {
    return head / (f.nhv / f.nkg);
}
/* The linear chain: with no PARENT operand, node i's parent is i-1 and the root's is -1. */
static __device__ __forceinline__ int bparent(const BoleCall& f, int bos, int node) {
    return f.t[BPARENT] ? bint(f, BPARENT, bos + node) : node - 1;
}

/* ================================================================== the metadata gate
 * The engine validates a batch before submission, but a direct op caller can hand these rows any
 * bytes, and a malformed PARENT would walk off the array. Both rows check every sequence first and
 * publish the verdict to the whole block before any read or write. */
static __device__ __forceinline__ bool bole_meta(const BoleCall& f, int seq) {
    const int bos = bint(f, BCU, seq), end = bint(f, BCU, seq + 1);
    if (bos < 0 || end < bos || end > f.tokens || end - bos > f.q_len) return false;
    if (seq && bos < bint(f, BCU, seq - 1)) return false;
    for (int i = 0; i < end - bos; ++i) {
        const int p = bparent(f, bos, i);
        if ((i == 0 && p != -1) || (i > 0 && (p < 0 || p >= i))) return false;
    }
    /* Slot indices are checked HERE, against the caches' actual slot counts, not only by the host: an
     * index past the cache would make the state or the conv-history read walk off its tensor. A
     * negative index is refused too -- libr4d treats it as padding and skips the sequence, but this
     * op's conv read has no padding path, so serving it would read before the array. */
    const int ci = bint(f, BCIDX, seq), si = bint(f, BSIDX, seq);
    if (ci < 0 || ci >= f.cslots) return false;
    if (si < 0 || si >= f.slots) return false;
    if (bint(f, BCTX, seq) < 0) return false;
    /* num_accepted is the cache row the last accepted token left, and both the conv read and the
     * history shift index from it. It is a caller value, so it is gated: off = nacc - 1 must land
     * inside the cache, and the deepest node's oldest tap is off + q_len - 1, which the host has
     * already bounded against `cdepth - (conv_width - 2)`. */
    if (f.t[BNACC]) {
        const int na = bint(f, BNACC, seq);
        if (na < 1 || na - 1 + f.q_len > f.cdepth) return false;
    }
    return true;
}

/* ===================== one node's q, k and v, convolved along that node's own path =====================
 * libr4d's conv, with the sample positions generalised from a linear chain to a path.
 *
 * libr4d (r4d_gdn_recurrent_update..., the `conv` lambda) convoys a sequence's window of T tokens with
 * the three history taps first and then shifts each token in:
 *
 *     win[i] = hist[off + i], i = 0..2        off = num_accepted - 1, the slot the last accepted
 *                                            token left
 *     for t = 0..T-1:  emit cp_silu(bias + sum_i w[i]*win[i]);  win <- win[1], win[2], x[t]
 *
 * so token t's four taps are the samples at positions t-3..t of the window, and a position below 0
 * comes from the history at index off + 3 + position. A node of the tree sits at window position
 * depth-1 (its own index on the path), so:
 *
 *     lag L in 0..3, position p = depth - 4 + L
 *     p >= 0 : x at the ancestor that produced it, path[depth - 4 + L]   (path[0] is the root)
 *     p <  0 : hist[off + 3 + p] = hist[off + depth - 1 + L]
 *
 * Tap order, bias first, and the one BF16 rounding of cp_silu are libr4d's, so a node whose path is
 * the linear chain has exactly the convolution the serial row computes -- which is what makes the
 * verify row's factors the engine's factors.
 */
static __device__ __forceinline__ unsigned short bole_conv(const BoleCall& f, int bos, int ch,
                                                           int slot, int ctx, int off,
                                                           const signed short* path, int depth) {
    (void)ctx;
    float acc = f.t[BCB] ? bole_bf2f(
        static_cast<const unsigned short*>(f.t[BCB])[boff(f, BCB, ch)]) : 0.0f;
    /* libr4d's conv lambda, tap for tap: `win[CW-1] = x[t]; acc = bias + sum_i wt[i]*win[i]`, then
     * the window shifts. So tap i reads the sample at position t - (CW-1) + i, paired with weight i.
     * A node sits at t = depth - 1 on its path, so tap i reads position p = depth - CW + i: an
     * ancestor when p >= 0, else the history at off + 3 + p = off + depth - 1 + i.
     *
     * The tap index IS the lag. Pairing the weights with `CW-1-i` instead reverses the kernel and,
     * worse, moves the history read from off + depth - 1 + i (at most off + 2) to off + depth + 2,
     * which for a five-token window reads seven rows past the cache row -- the original fault. */
#pragma unroll
    for (int tap = 0; tap < BOLE_CONV_W; ++tap) {
        const int pos = depth - BOLE_CONV_W + tap;
        float x;
        /* `path` is stored NODE-FIRST (path[0] is the node, path[depth-1] the root), which is the
         * order the backward adjoint walk needs. A window position counts from the ROOT, so position
         * `pos` is path[depth - 1 - pos]. Indexing path[pos] instead mirrors the window: at depth 3
         * and beyond every node reads its own input where its ancestor's belongs, which is a wrong
         * number on the linear chain, not a fault. */
        if (pos >= 0)
            x = bole_bf2f(static_cast<const unsigned short*>(f.t[BX])[
                boff(f, BX, bos + path[depth - 1 - pos], ch)]);
        else
            x = bole_bf2f(static_cast<const unsigned short*>(f.t[BCST])[
                boff(f, BCST, slot, ch, off + depth - 1 + tap)]);
        const float w = bole_bf2f(
            static_cast<const unsigned short*>(f.t[BCW])[boff(f, BCW, ch, tap)]);
        acc += w * x;
    }
    return bole_f2bf(bole_silu(acc));
}

/* ================================================================== verify (Bole) */
/* One workgroup per (sequence, value head), 256 threads in libr4d's shape.
 *
 *   1. metadata verdict, published in LDS before any read;
 *   2. every node's q/k/v convolved along its path (bf16 in LDS, as the engine stores them);
 *   3. q,k l2 per node with the engine's 64-term sum and one lane^1 exchange;
 *   4. the root, serially, in the engine's FP32 order -> anchor = h1;
 *   5. nodes 1.. node-parallel, eight waves of two 16-lane walk rows, FP64, two passes;
 *   6. the contractions against h1 and the rounded-u feedback, one value row per thread pair;
 *   7. the factors the commit row recomputes u from (these are exactly the engine's normalized k);
 *   8. the engine's epilogue per node: gated rmsnorm, bf16 output, INT8 group-128 codes.
 */
__global__ __launch_bounds__(BOLE_THR) void bole_verify_kernel(BoleCall f) {
    const int seq = blockIdx.x, head = blockIdx.y;
    const int tid = threadIdx.x, row = tid >> 1, half = tid & 1, k0 = half * BOLE_HK;

    __shared__ char ok;
    __shared__ int par[BOLE_NODES];
    __shared__ signed short path[BOLE_NODES][BOLE_NODES], depth[BOLE_NODES];
    __shared__ unsigned short ancestors[BOLE_NODES], value[BOLE_NODES][128];
    __shared__ float qo[BOLE_NODES][128], kn[BOLE_NODES][128];
    __shared__ float gamma[BOLE_NODES], beta[BOLE_NODES], wave_sum[BOLE_THR / 32];
    /* [0] the q walk, [1] the gamma*k walk, over the non-root path of each node. */
    __shared__ double walk[2][BOLE_NODES][128], dots[2][BOLE_NODES][BOLE_NODES];

    if (!tid) {
        ok = 1;
        for (int s = 0; s < f.nseq; ++s)
            if (!bole_meta(f, s)) { ok = 0; break; }
    }
    __syncthreads();
    if (!ok) return;            /* no output, no factor, no state before the verdict */

    const int bos = bint(f, BCU, seq), nnode = bint(f, BCU, seq + 1) - bos;
    if (!nnode) return;         /* an empty sequence has no root: nothing to do */
    const int hslot = bint(f, BCIDX, seq), ctx = bint(f, BCTX, seq), slot = bint(f, BSIDX, seq);
    /* libr4d's history read offset: the cache row the last ACCEPTED token left. */
    const int off = f.t[BNACC] ? bint(f, BNACC, seq) - 1 : 0;
    const int kh = bole_khead(f, head);

    if (tid < nnode) par[tid] = bparent(f, bos, tid);
    __syncthreads();
    if (tid < nnode) {
        int n = 0;
        unsigned short mask = 0;
        for (int p = tid; p >= 0; p = par[p]) {
            path[tid][n++] = (signed short)p;
            if (p != tid && p) mask |= (unsigned short)(1u << p);   /* the root folds into h1 */
        }
        depth[tid] = (signed short)n;
        ancestors[tid] = mask;
    }
    __syncthreads();

    /* Every node's q, k and v: one channel per thread, three channel classes per node. */
    for (int e = tid; e < nnode * 384; e += BOLE_THR) {
        const int node = e / 384, channel = e % 384, col = channel & 127;
        const int ch = channel < 128 ? kh * f.kw + col :
                       channel < 256 ? (f.nkg + kh) * f.kw + col : (2 * f.nkg + head) * f.vw + col;
        const unsigned short v = bole_conv(f, bos, ch, hslot, ctx, off, path[node], depth[node]);
        if (channel < 128) qo[node][col] = bole_bf2f(v);
        else if (channel < 256) kn[node][col] = bole_bf2f(v);
        else value[node][col] = v;                              /* kept bf16 for the factor row */
    }
    __syncthreads();

    /* q,k l2 per node: thread pair (2n, 2n+1) owns node n's two k halves, each summing 64 products
     * and closing with one lane^1 exchange -- libr4d's terms, its order, its rounding. */
    if (tid < 2 * nnode) {
        const int node = tid >> 1;
        float sq = 0.0f, sk = 0.0f;
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) {
            sq += qo[node][k0 + j] * qo[node][k0 + j];
            sk += kn[node][k0 + j] * kn[node][k0 + j];
        }
        sq += bole_xor<1>(sq);
        sk += bole_xor<1>(sk);
        const float qs = __frsqrt_rn(sq + f.l2_eps) * f.scale, ks = __frsqrt_rn(sk + f.l2_eps);
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) {
            qo[node][k0 + j] *= qs;
            kn[node][k0 + j] *= ks;
        }
    }
    if (tid < nnode) {
        const int tok = bos + tid;
        const float av = bole_bf2f(
            static_cast<const unsigned short*>(f.t[BA])[boff(f, BA, tok, head)]);
        const float bv = bole_bf2f(
            static_cast<const unsigned short*>(f.t[BB])[boff(f, BB, tok, head)]);
        const float alog = __expf(bflt(f, BALOG, head)), dtb = bflt(f, BDTB, head);
        gamma[tid] = __expf(-alog * bsp(av + dtb, f.softplus_thr));
        beta[tid] = 1.0f / (1.0f + __expf(-bv));
    }
    __syncthreads();

    /* ---- the root: one node, serially, in the engine's FP32 order. `anchor` holds h1 afterwards,
     * and every other node's walk continues from it. */
    float anchor[BOLE_HK], on0 = 0.0f;
    {
        const float* p = static_cast<const float*>(f.t[BSTATE]) +
                         boff(f, BSTATE, slot, head, row, 0) + (int64_t)k0 * f.s[BSTATE][3];
        const int64_t cs = f.s[BSTATE][3];
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) anchor[j] = p[j * cs];
    }
    {
        const float decay = gamma[0];
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) {
            anchor[j] *= decay;
            dot += anchor[j] * kn[0][k0 + j];
        }
        dot += bole_xor<1>(dot);
        const float u = (bole_bf2f(value[0][row]) - dot) * beta[0];
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) {
            anchor[j] += u * kn[0][k0 + j];
            on0 += anchor[j] * qo[0][k0 + j];
        }
        on0 += bole_xor<1>(on0);
    }
    /* ---- nodes 1..: eight waves a pass, wave n of a pass walking node pass*8+n. Its two 16-lane
     * rows walk q and gamma*k over the node's non-root path, both over the same steps so every DPP
     * exchange sees a full wave.
     *
     * `path` is NODE-FIRST (path[node][0] is the node, path[node][depth-1] the root), so s = 0 is the
     * node itself and s = depth-2 is the child of the root. The root is excluded deliberately: the
     * anchor h1 is already the state AFTER the root's update, so folding the root in again would
     * apply it twice. dots[kind][node][a] is k_a . x just before a's update -- the coefficient the
     * contraction pairs with the rounding residual w[a].
     *
     * The q walk starts at the node (o_node needs u_node (k_node . q_node)); the gamma*k walk
     * describes h_parent . (gamma_node k_node), so it starts at the PARENT and skips the node's own
     * update. */
    for (int pass = 0; pass < (BOLE_NODES + 7) / 8; ++pass) {
        const int node = pass * 8 + (tid >> 5), kind = (tid >> 4) & 1, lane = tid & 15;
        if (node && node < nnode) {
            double x[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) {
                const int e = lane + 16 * m;
                x[m] = kind ? double(gamma[node]) * double(kn[node][e]) : double(qo[node][e]);
            }
            for (int s = 0; s < depth[node] - 1; ++s) {
                const int a = path[node][s];
                double dot = 0.0;
#pragma unroll
                for (int m = 0; m < 8; ++m) dot += double(kn[a][lane + 16 * m]) * x[m];
                dot += bole_xor_d<1>(dot); dot += bole_xor_d<2>(dot);
                dot += bole_xor_d<4>(dot); dot += bole_xor_d<8>(dot);
                if (kind && !s) continue;                 /* the gamma k walk starts at the parent */
                if (!lane) dots[kind][node][a] = dot;
                const double t = double(beta[a]) * dot, g = gamma[a];
#pragma unroll
                for (int m = 0; m < 8; ++m) x[m] = g * (x[m] - t * double(kn[a][lane + 16 * m]));
            }
#pragma unroll
            for (int m = 0; m < 8; ++m) walk[kind][node][lane + 16 * m] = x[m];
        }
        __syncthreads();     /* the pass's walk plane is complete before any contraction reads it */
    }

    /* h1 serves both contractions for every non-root node; the thread's own 64 columns, one lane^1
     * exchange, as the engine closes a dot. */
    double ao[BOLE_NODES], au[BOLE_NODES], w[BOLE_NODES];
#pragma unroll
    for (int node = 0; node < BOLE_NODES; ++node) {
        ao[node] = au[node] = 0.0;
        if (node && node < nnode) {
            double so = 0.0, su = 0.0;
#pragma unroll
            for (int j = 0; j < BOLE_HK; ++j) {
                so += double(anchor[j]) * walk[0][node][k0 + j];
                su += double(anchor[j]) * walk[1][node][k0 + j];
            }
            so += bole_xor_d<1>(so);
            su += bole_xor_d<1>(su);
            ao[node] = so;
            au[node] = su;
        }
    }
    /* Per value row the ancestors precede the descendants, so each u sees its ancestors' rounded
     * values. `w` carries the rounding residual, which is what makes the descendants agree. Both
     * halves of a row compute the same w; only half 0 writes the output row. */
    if (!half) qo[0][row] = on0;
#pragma unroll
    for (int i = 0; i < BOLE_NODES; ++i) {
        w[i] = 0.0;
        if (i && i < nnode) {
            const unsigned mask = ancestors[i];
            double r = au[i];
#pragma unroll
            for (int j = 0; j < i; ++j)
                if ((mask >> j) & 1u) r += w[j] * dots[1][i][j];
            const double v = double(bole_bf2f(value[i][row])), b = double(beta[i]);
            const double exact = b * (v - r);
            const float u = float(exact);              /* the FP32 value a serial step would form */
            w[i] = b * v + (double(u) - exact);
            double o = ao[i];
#pragma unroll
            for (int j = 0; j <= i; ++j)
                if (j == i || ((mask >> j) & 1u)) o += w[j] * dots[0][i][j];
            if (!half) qo[i][row] = float(o);
        }
    }
    /* The commit row's contract: [normalized k(128) | conv v as f32(128) | gamma | beta]. Written
     * before the epilogue overwrites qo with the output rows. */
    {
        float* const fs = static_cast<float*>(f.t[BFACT]);
        const int64_t fi = f.s[BFACT][2];
        for (int e = tid; e < nnode * BOLE_FACTOR_W; e += BOLE_THR) {
            const int node = e / BOLE_FACTOR_W, col = e % BOLE_FACTOR_W;
            const float x = col < 128 ? kn[node][col] :
                            col < 256 ? bole_bf2f(value[node][col - 128]) :
                            col == 256 ? gamma[node] : beta[node];
            fs[boff(f, BFACT, bos + node, head) + (int64_t)col * fi] = x;
        }
    }
    __syncthreads();

    /* ---- the engine's epilogue, one node at a time: gated rmsnorm over the 128-wide row (each half
     * squares only its own thread's row, then lanes 16, 8, 4, 2 and 1 across the wave, then the eight
     * wave sums), one bf16 rounding, the bf16 output, then the INT8 group-128 codes and their scale.
     * The quantiser reads the rounded bf16, never the f32 `on`, which is what keeps it equal to the
     * standalone op. */
    const float nw = bflt(f, BNW, row);
    for (int node = 0; node < nnode; ++node) {
        const int tok = bos + node;
        float on = qo[node][row];
        float rms = !half ? on * on : 0.0f;
        rms += bole_xor<16>(rms); rms += bole_xor<8>(rms); rms += bole_xor<4>(rms);
        rms += bole_xor<2>(rms);  rms += bole_xor<1>(rms);
        if (!(tid & 31)) wave_sum[tid >> 5] = rms;
        __syncthreads();
        float total = 0.0f;
#pragma unroll
        for (int i = 0; i < BOLE_THR / 32; ++i) total += wave_sum[i];
        const float sc = __frsqrt_rn(total * (1.0f / 128) + f.eps);
        const float z = bole_bf2f(static_cast<const unsigned short*>(f.t[BZ])[
            boff(f, BZ, tok, head * 128 + row)]);
        const float sig = 1.0f / (1.0f + __expf(-z));
        on *= sc * nw * (f.sigmoid ? sig : z * sig);
        const unsigned short ob = bole_f2bf(on);
        /* `o` is [tokens, H, V]: three indices. Passing `head*128 + row` as the second index
         * multiplies it by the V stride and writes (H*V) rows past the tensor. z and o_q are
         * [tokens, H*V], which is why they legitimately take the flattened column. */
        if (!half)
            static_cast<unsigned short*>(f.t[BO])[boff(f, BO, tok, head, row)] = ob;
        __syncthreads();
        if (f.t[BOQ]) {
            const float ov = bole_bf2f(ob);
            float mv = !half ? fabsf(ov) : 0.0f;
            mv = fmaxf(mv, bole_xor<16>(mv)); mv = fmaxf(mv, bole_xor<8>(mv));
            mv = fmaxf(mv, bole_xor<4>(mv));  mv = fmaxf(mv, bole_xor<2>(mv));
            mv = fmaxf(mv, bole_xor<1>(mv));
            if (!(tid & 31)) wave_sum[tid >> 5] = mv;
            __syncthreads();
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < BOLE_THR / 32; ++i) amax = fmaxf(amax, wave_sum[i]);
            const float qs = amax > 0.0f ? amax * (1.0f / bole_q8_top<true>()) : 1.0f;
            if (!tid) static_cast<float*>(f.t[BOS])[boff(f, BOS, tok, head)] = qs;
            if (!half) {
                const float cv = fminf(fmaxf(ov * (1.0f / qs), -127.0f), 127.0f);
                static_cast<uint8_t*>(f.t[BOQ])[boff(f, BOQ, tok, head * 128 + row)] =
                    (uint8_t)(int)rintf(cv);
            }
        }
        __syncthreads();                  /* quant[] and wave_sum[] are reused by the next node */
    }
}

/* ================================================================== commit (byte-exact) */
/* One selected-path node of the commit: the engine's recurrent step on the 64 k columns a thread
 * owns. The factors are the verify row's, which are the engine's normalized k, and the arithmetic
 * here is the serial step's, so the committed state is the serial state bit for bit. A stored U is
 * deliberately not replayed: that is the failure the archived adjoint variant had. */
static __device__ __forceinline__ void bole_recompute(float (&h)[BOLE_HK], const float* kcol,
                                                      float v, float decay, float beta) {
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < BOLE_HK; ++j) {
        h[j] *= decay;
        dot += h[j] * kcol[j];
    }
    dot += bole_xor<1>(dot);
    const float u = (v - dot) * beta;
#pragma unroll
    for (int j = 0; j < BOLE_HK; ++j) h[j] += u * kcol[j];
}

/* grid = (sequences, value heads + history workgroups), block = 256. Each value-head workgroup
 * replays the path's factors into one state (sequence, head); the history workgroups shift the conv
 * cache as gdn_conv_update does. Both gate on the same metadata verdict as the verify row, because a
 * commit that half-applies is worse than one that refuses. */
__global__ __launch_bounds__(BOLE_THR) void bole_commit_kernel(BoleCall f) {
    const int seq = blockIdx.x, unit = blockIdx.y, tid = threadIdx.x;
    __shared__ char ok;
    if (tid == 0) {
        char good = 1;
        for (int s = 0; s < f.nseq; ++s)
            if (!bole_meta(f, s)) { good = 0; break; }
        ok = good;
    }
    __syncthreads();
    if (!ok) return;

    const int bos = bint(f, BCU, seq), len = bint(f, BCU, seq + 1) - bos;

    if (unit < f.nhv) {
        const int slot = bint(f, BSIDX, seq), row = tid >> 1, k0 = (tid & 1) * BOLE_HK;
        const int64_t cs = f.s[BSTATE][3];
        float* const h0 = static_cast<float*>(f.t[BSTATE]) +
                          boff(f, BSTATE, slot, unit, row, 0) + (int64_t)k0 * cs;
        float h[BOLE_HK];
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) h[j] = h0[j * cs];

        const int64_t fi = f.s[BFACT][2];
        const float* const fs = static_cast<const float*>(f.t[BFACT]);
        /* Stage the path's factors: one node's 258 values are contiguous, so a linear index reads
         * them coalesced. The caller passes the accepted path's nodes in path order, root first, so
         * a tile is an ordinal range of it. */
        __shared__ __align__(16) float s_k[BOLE_COMMIT_TILE][128], s_v[BOLE_COMMIT_TILE][128];
        __shared__ float s_g[BOLE_COMMIT_TILE], s_b[BOLE_COMMIT_TILE];
        for (int first = 0; first < len; first += BOLE_COMMIT_TILE) {
            const int n = len - first < BOLE_COMMIT_TILE ? len - first : BOLE_COMMIT_TILE;
            if (first) __syncthreads();
            for (int e = tid; e < n * BOLE_FACTOR_W; e += BOLE_THR) {
                const int i = e / BOLE_FACTOR_W, j = e % BOLE_FACTOR_W;
                const float v = fs[boff(f, BFACT, bos + first + i, unit) + (int64_t)j * fi];
                if (j < 128) s_k[i][j] = v;
                else if (j < 256) s_v[i][j - 128] = v;
                else if (j == 256) s_g[i] = v;
                else s_b[i] = v;
            }
            __syncthreads();
            for (int i = 0; i < n; ++i)
                bole_recompute(h, s_k[i] + k0, s_v[i][row], s_g[i], s_b[i]);
        }
#pragma unroll
        for (int j = 0; j < BOLE_HK; ++j) h0[j * cs] = h[j];
        return;
    }

    /* The conv history arm, one packed [q|k|v] channel per thread: exactly libr4d's rewrite
     * (ru_conv_rewrite, NH = conv_width - 2). Drop the two oldest entries, keep the next two at
     * offsets 0 and 1, then append this step's raw convolution inputs from offset conv_width - 2.
     * Anything else would leave the cache in a layout the next decode step reads with a shifted
     * window. */
    const int ch = (unit - f.nhv) * BOLE_THR + tid;
    /* The packed width is the projection row's, not a stride: f.s[BCST][1] is the conv cache's
     * channel stride, which equals the width only when the cache happens to be dense. Reading it as
     * a count made a padded cache read past the row and a sparse one drop channels. */
    if (ch >= f.cdim) return;
    const int cslot = bint(f, BCIDX, seq);
    if (cslot < 0) return;                        /* padding: nothing to shift */
    const int64_t hi = f.s[BCST][2];
    unsigned short* const hist = static_cast<unsigned short*>(f.t[BCST]) +
                                 boff(f, BCST, cslot, ch, 0);
    const unsigned short* const rawp = static_cast<const unsigned short*>(f.t[BX]) +
                                        boff(f, BX, bos, ch);
    const int keep = f.conv_w - 2;                /* libr4d's NH */
    for (int i = 0; i < keep; ++i) hist[(size_t)i * hi] = hist[(size_t)(i + 1) * hi];
    for (int j = 0; j < len; ++j)
        hist[(size_t)(j + keep) * hi] = rawp[(int64_t)j * f.s[BX][0]];
}

/* ================================================================== the launches */
int bole_verify_launch(const BoleCall* f, void* stream) {
    const BoleCall c = *f;
    hipLaunchKernelGGL(bole_verify_kernel, dim3((unsigned)c.nseq, (unsigned)c.nhv, 1),
                       dim3(BOLE_THR), 0, (hipStream_t)stream, c);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int bole_commit_launch(const BoleCall* f, void* stream) {
    const BoleCall c = *f;
    hipLaunchKernelGGL(bole_commit_kernel,
                       dim3((unsigned)c.nseq, (unsigned)(c.nhv + BOLE_COMMIT_HIST_WG), 1),
                       dim3(BOLE_THR), 0, (hipStream_t)stream, c);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
