/* dflash_tree_select.hip -- DFlash2's bounded DarTree selector, one workgroup a sequence.
 *
 * The op definition is mine/dflash_tree_select.h. The EDGE ARITHMETIC IS `dflash_select`'s,
 * reproduced operation for operation: per candidate, a 32-lane accumulation over
 * `r = lane, lane+32, ...` seeded at zero, the same xor-butterfly fold (`16,8,4,2,1`) the
 * production kernel gets from `r4d_wave_sum`, and `unary` added LAST. Both are compiled with
 * `-ffp-contract=off`, so no product is fused into its add on either side. That is what makes the
 * single-path case reproduce the shipping selector's tokens rather than merely its formula.
 *
 * WHAT CHANGES IS WHAT HAPPENS AFTER A LEVEL'S SCORES ARE KNOWN: every parent of the frontier
 * scores every candidate, each parent's candidates are shifted so its own best sits at zero, and
 * the level's best `tree_width` pairs become nodes. The shift is monotone within the parent and
 * has no effect on which pair wins within it, but it makes scores comparable across parents, which
 * a tree needs and a chain does not.
 *
 * INVALID ROWS WRITE `count = 0`. A launch cannot be failed once issued, so a row whose data
 * cannot produce a tree writes zero and stops. The host oracle produces the same zero.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "dflash_tree_select.h"
#include "../libr4d/r4d_common.h"

namespace {

constexpr int kThreads = 128;
constexpr int kWave = 32;                              /* production's wave width */
constexpr int kMaxK = RAD_DFLASH_TREE_MAX_K;
constexpr int kMaxWidth = RAD_DFLASH_TREE_MAX_WIDTH;
constexpr int kMaxNodes = RAD_DFLASH_TREE_MAX_NODES;   /* 1 + 3*16 */
constexpr int kMaxPairs = kMaxK * kMaxWidth;
constexpr int kCols = RAD_DFLASH_TREE_OUT_COLS;
/* THE ROW'S NODE CAPACITY, which is what the packing pitch is derived from and NOT the selector's
 * `tree_budget`. The engine declares the tree with 16 node slots and reads a row of
 * `1 + 2*capacity` columns, so at capacity 16 that row is 33 columns wide with the anchor's
 * sentinel parent in between the two blocks. Keeping the pitch written as `1 + capacity + i`
 * rather than as a literal 18 is what makes this correct for a capacity other than 16. */
constexpr int kCapacity = (kCols - 1) / 2;              /* 16 */
static_assert(kCapacity <= RAD_DFLASH_TREE_MAX_WIDTH, "the row must hold every declared width");
static_assert(1 + 2 * kCapacity == kCols, "the row is exactly two capacity-blocks plus the count");

/* BF16 by shift, exactly as the production kernels widen it. */
__device__ __forceinline__ float bf16(uint16_t x) { return __uint_as_float((unsigned)x << 16); }

__device__ __forceinline__ bool finite(float x) { return x == x && x != INFINITY && x != -INFINITY; }

/* `dflash_select`'s edge sum for one (parent, candidate), in its order:
 *
 *   sa[r]     = pred[pid][r] * hp[row][r]        (or pred[pid][r] when hp is absent)
 *   acc[l]   += sa[r] * succ[token][r]           for r = l, l+32, ...   (l in [0, 32))
 *   fold      : stride 16, 8, 4, 2, 1  ->  acc[i] += acc[i ^ stride]
 *   score     = acc[0] + unary                    (the add comes LAST)
 *
 * Every lane participates, including zero lanes past an odd R, just as in the shipping fold. */
__device__ __forceinline__ float edge_sum(const uint16_t* pred_row, const uint16_t* hp_row,
                                         const uint16_t* succ_row, int R, int lane) {
    float acc = 0.0f;
    for (int64_t r = lane; r < R; r += kWave) {
        const float p = bf16(pred_row[r]);
        const float a = hp_row ? __fmul_rn(p, bf16(hp_row[r])) : p;
        acc = __fadd_rn(acc, __fmul_rn(a, bf16(succ_row[r])));
    }
    return r4d_wave_sum(acc);
}

struct Args {
    const int32_t* cand;
    const float*   unary;
    const uint16_t* hp;          /* nullptr: a plane of ones */
    const int32_t* anchor;
    const uint16_t* pred;
    const uint16_t* succ;
    int32_t*       out;
    int M, steps, K, R, V, astr, budget, width;
    float beta;
    int cand_ld, unary_ld, hp_ld, pred_ld, succ_ld, out_ld;
};

__global__ __launch_bounds__(kThreads) void tree_select_kernel(Args a) {
    const int s = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & (kWave - 1), wave = tid / kWave;
    int32_t* out = a.out + (int64_t)s * a.out_ld;
    if (tid == 0) {
        for (int i = 0; i < kCols; ++i) out[i] = -1;
        out[0] = 0;                          /* the invalid-row answer, before any early return */
    }

    __shared__ int tok[kMaxNodes], par[kMaxNodes], depth[kMaxNodes];
    __shared__ float score[kMaxNodes];
    __shared__ int32_t ctok[kMaxK];
    __shared__ float cun[kMaxK];
    __shared__ unsigned char live[kMaxK];
    __shared__ float pairScore[kMaxPairs], pairCorr[kMaxPairs];
    __shared__ int pairParent[kMaxPairs], pairCand[kMaxPairs], pairLive[kMaxPairs];
    __shared__ int keep[kMaxNodes + 1], remap[kMaxNodes + 1];
    __shared__ int invalid, total, first, frontier, chosen;

    if (tid == 0) {
        const int root = a.anchor[(int64_t)s * a.astr];
        invalid = root < 0 || root >= a.V;
        tok[0] = root;
        par[0] = -1;
        depth[0] = 0;
        score[0] = 0.0f;
        total = 1;
        first = 0;
        frontier = 1;
    }
    /* Cross-thread communication is LDS-only: inputs are read-only, and only tid 0 writes out.
     * Keep codebook cache lines across phases rather than globally fencing every barrier. */
    lds_barrier();
    if (invalid) return;

    for (int level = 1; level <= a.steps && frontier > 0; ++level) {
        const int64_t row = (int64_t)s * a.steps + level - 1;
        if (tid < a.K) {
            ctok[tid] = a.cand[row * a.cand_ld + tid];
            cun[tid] = a.unary[row * a.unary_ld + tid];
            /* An id outside the vocabulary is not a codebook row and cannot be a tree node.
             * Duplicates ARE live here and are deduplicated later, per parent, by score -- see the
             * pair loop below for why the choice of instance matters. */
            live[tid] = (ctok[tid] >= 0 && ctok[tid] < a.V) ? 1 : 0;
            if (live[tid] && !finite(cun[tid])) atomicExch(&invalid, 1);
        }
        lds_barrier();
        if (invalid) return;

        /* One wave owns a (parent, candidate) pair. Flattening the whole frontier keeps waves
         * busy even for K < the wave count, without a barrier for each parent. */
        const int pairs = frontier * a.K;
        for (int i = wave; i < pairs; i += kThreads / kWave) {
            const int j = i % a.K, node = first + i / a.K;
            if (live[j]) {
                const float corr = __fadd_rn(
                    edge_sum(a.pred + (int64_t)tok[node] * a.pred_ld,
                             a.hp ? a.hp + row * a.hp_ld : nullptr,
                             a.succ + (int64_t)ctok[j] * a.succ_ld, a.R, lane),
                    cun[j]);
                if (lane == 0) {
                    pairCorr[i] = corr;
                    if (!finite(corr)) atomicExch(&invalid, 1);
                }
            }
        }
        lds_barrier();
        if (invalid) return;

        /* Each parent owns a disjoint K-slot slice; dead and duplicate slots stay unoffered.
         * Keeping the holes preserves the old parent/candidate order without serial compaction. */
        if (tid < frontier) {
            const int begin = tid * a.K, node = first + tid;
            /* Shift by the parent's own maximum, exactly as before. Unary was added LAST above. */
            float peak = -INFINITY;
            for (int j = 0; j < a.K; ++j)
                if (live[j] && pairCorr[begin + j] > peak) peak = pairCorr[begin + j];
            for (int j = 0; j < a.K; ++j) {
                const int i = begin + j;
                pairLive[i] = 0;
                if (!live[j]) continue;
                /* Compare EVERY instance: a later, higher-scoring duplicate evicts an earlier
                 * one. Exact ties retain the lower candidate index, as in the linear walk. */
                bool beaten = false;
                for (int q = 0; q < a.K && !beaten; ++q) {
                    if (q == j || !live[q] || ctok[q] != ctok[j]) continue;
                    beaten = pairCorr[begin + q] > pairCorr[i] ||
                             (pairCorr[begin + q] == pairCorr[i] && q < j);
                }
                if (beaten) continue;
                pairScore[i] = __fadd_rn(score[node], pairCorr[i] - peak);
                pairParent[i] = node;
                pairCand[i] = j;
                pairLive[i] = 1;
            }
        }
        /* Snapshot before the fence: thread 0 may publish next level's total while other lanes
         * are still appending this one, and every writer must retain the same old-id base. */
        const int append = total;
        lds_barrier();

        /* The level's best `width` pairs. `pairScore` is the cross-parent order; then the PARENT old
         * id, which is a deterministic index rather than a comparison of two different parents'
         * logit scales; then `pairCorr`, so within one parent the winner is the linear walk's argmax
         * even when the shift rounds two neighbours together; then the candidate index. Total, and
         * identical on both sides. */
        /* Count strictly better pairs in parallel. The comparator is total, so rank is a unique
         * old-id offset: the same append order as width serial argmax rounds, without their
         * barriers. Stop counting once a pair cannot fit in this level's bounded prefix. */
        for (int i = tid; i < pairs; i += kThreads) {
            if (!pairLive[i]) continue;
            int rank = 0;
            for (int q = 0; q < pairs && rank < a.width; ++q) {
                if (!pairLive[q]) continue;
                const bool better =
                    pairScore[q] > pairScore[i] ||
                    (pairScore[q] == pairScore[i] &&
                     (pairParent[q] < pairParent[i] ||
                      (pairParent[q] == pairParent[i] &&
                       (pairCorr[q] > pairCorr[i] ||
                        (pairCorr[q] == pairCorr[i] && pairCand[q] < pairCand[i])))));
                if (better) ++rank;
            }
            if (rank >= a.width) continue;
            const int node = append + rank;
            tok[node] = ctok[pairCand[i]];
            par[node] = pairParent[i];
            depth[node] = level;
            score[node] = pairScore[i];
        }
        /* Pair liveness remains immutable throughout ranking; count it alongside the workers. */
        if (tid == 0) {
            chosen = 0;
            for (int i = 0; i < pairs && chosen < a.width; ++i)
                if (pairLive[i]) ++chosen;
            first = append;
            total = append + chosen;
            frontier = chosen;
        }
        lds_barrier();
    }

    /* ---- prune, then pack in old-id order ----
     * Peak-normalized deltas <= 0, validated finite beta <= 0 and increasing depth make each
     * penalized child <= its parent. Depth ties put the parent first even at -inf. The global
     * top-budget prefix is therefore parent-closed, exactly the eligible-
     * parent greedy walk. Rank it in parallel, reusing the now-dead pair score plane. */
    float* const value = pairScore;
    if (tid < total) value[tid] = score[tid] + a.beta * (float)depth[tid];
    lds_barrier();
    if (tid == 0) keep[0] = 1;
    if (tid > 0 && tid < total) {
        int rank = 0;
        for (int i = 1; i < total && rank < a.budget; ++i)
            if (value[i] > value[tid] ||
                (value[i] == value[tid] &&
                 (depth[i] < depth[tid] || (depth[i] == depth[tid] && i < tid))))
                ++rank;
        keep[tid] = rank < a.budget;
    }
    lds_barrier();
    if (tid != 0) return;

    for (int i = 0; i <= total; ++i) remap[i] = -1;
    remap[0] = 0;                       /* the anchor is node 0 */
    int node = 0;                       /* root-inclusive node id of the last emitted node */
    for (int i = 1; i < total; ++i) {
        if (!keep[i]) continue;
        const int parent = remap[par[i]];
        /* A level is appended whole, so a parent's old id is smaller and already remapped; the
         * check stays because a tree the verifier cannot read must not be published. */
        if (parent < 0 || par[i] >= i) {
            for (int c = 0; c < kCols; ++c) out[c] = -1;
            out[0] = 0;
            return;
        }
        remap[i] = ++node;
        /* THE PACKED ROW, capacity-generic, exactly as `dloop_tree.hip` and `ref_dloop.cpp` write
         * it and as `dloop_tree_from_packed` reads it:
         *   row[0]                    count = 1 + draft nodes
         *   row[1 + i]                node i's token, i = 0 (the anchor) .. count-1
         *   row[1 + kCols/2 + i]      node i's parent, the same range -- at the declared capacity
         *                             16 that is row[17 + i], and row[17] is the anchor's -1
         *                              SENTINEL the validator requires
         * `node` is the ROOT-INCLUSIVE id, so the token and parent of one node are written at the
         * same offset and the parent block starts exactly `capacity` columns after the token
         * block. That is why 16 tokens and 16 parents fit in 33 columns with the sentinel in
         * between, and why this form stays correct for a capacity other than 16. */
        out[1 + node] = tok[i];
        out[1 + kCapacity + node] = parent;
    }
    out[0] = node + 1;
    out[1] = tok[0];
    out[1 + kCapacity] = -1;           /* the anchor's sentinel parent */
}

}  /* namespace */

int rad_mine_dflash_tree_select_launch(const RadArgs* a, RadStream stream) {
    RadDflashTreeSelectCall call{};
    const int rc = rad_dflash_tree_select_args(a, &call);
    if (rc != RAD_OK) return rc;

    Args args{};
    args.cand = static_cast<const int32_t*>(call.cand);
    args.unary = static_cast<const float*>(call.unary);
    args.hp = static_cast<const uint16_t*>(call.hp);
    args.anchor = static_cast<const int32_t*>(call.anchor);
    args.pred = static_cast<const uint16_t*>(call.pred);
    args.succ = static_cast<const uint16_t*>(call.succ);
    args.out = static_cast<int32_t*>(call.out);
    args.M = (int)call.M;
    args.steps = (int)call.steps;
    args.K = (int)call.K;
    args.R = (int)call.R;
    args.V = (int)call.V;
    args.astr = (int)call.anchor_stride;
    args.budget = (int)call.budget;
    args.width = (int)call.width;
    args.beta = call.beta;
    args.cand_ld = (int)call.cand_ld;
    args.unary_ld = (int)call.unary_ld;
    args.hp_ld = (int)call.hp_ld;
    args.pred_ld = (int)call.pred_ld;
    args.succ_ld = (int)call.succ_ld;
    args.out_ld = (int)call.out_ld;

    void* kargs[] = { &args };
    const hipError_t e = hipLaunchKernel((const void*)tree_select_kernel, dim3((unsigned)call.M),
                                         dim3(kThreads), kargs, 0, (hipStream_t)stream);
    return e == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int rad_mine_dflash_tree_select_count(void) { return 1; }

const RadKernelInfo* rad_mine_dflash_tree_select_at(int i) {
    static const RadKernelInfo row = rad_dflash_tree_select_row(rad_mine_dflash_tree_select_launch);
    return i == 0 ? &row : nullptr;
}
