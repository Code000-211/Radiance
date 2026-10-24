/* ref_dflash_tree.cpp -- the host oracle for `dflash_tree_select`.
 *
 * It shares the schema and the call validation with the device row (mine/dflash_tree_select.cpp)
 * and nothing else, so a mistake in the kernel's shared-memory bookkeeping cannot be reproduced
 * here. What the two DO have to agree on is arithmetic and order, because the comparison is a byte
 * comparison:
 *
 *   * the edge sum is `dflash_select`'s, operation for operation -- a 32-lane accumulation over
 *     `r = lane, lane+32, ...` seeded at zero, the xor-butterfly fold (16,8,4,2,1), then `unary`
 *     added last, all FP32 with no contraction (`-ffp-contract=off` on this unit, and explicit
 *     temporaries so the compiler cannot fuse anyway);
 *   * a level keeps its best `width` pairs by (shifted score desc, raw score desc, parent old id
 *     asc, candidate index asc) -- the raw score second so a shift that rounds two neighbours
 *     together cannot change a single-parent pick;
 *   * a level's scores are shifted by that parent's own maximum; a node's score is its parent's
 *     score plus that shifted increment;
 *   * pruning keeps the best `budget` nodes by `score + depth_penalty * depth`, refusing a node
 *     whose parent is not kept;
 *   * packing is old-id order with root-inclusive parents; an invalid row writes count = 0.
 *
 * The tie cases in the tests use EXACTLY equal inputs, so both sides see the same bits and this
 * oracle's arithmetic is not being asked to adjudicate a 1-ulp difference. */
#include "ref_dflash_tree.h"

#include "dflash_tree_select.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace {

constexpr int kWave = 32;
constexpr int kMaxK = RAD_DFLASH_TREE_MAX_K;
constexpr int kMaxWidth = RAD_DFLASH_TREE_MAX_WIDTH;
constexpr int kMaxNodes = RAD_DFLASH_TREE_MAX_NODES;
constexpr int kCols = RAD_DFLASH_TREE_OUT_COLS;
/* The row's node capacity, which is what the packing pitch is derived from. See
 * mine/dflash_tree_select.hip: the engine declares 16 node slots, and the row is
 * 1 + 2*capacity columns with the anchor's sentinel in between. */
constexpr int kCapacity = (kCols - 1) / 2;

float bf(uint16_t x) {
    const uint32_t u = (uint32_t)x << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool finite_f(float x) {
    return x == x && x != std::numeric_limits<float>::infinity() &&
           x != -std::numeric_limits<float>::infinity();
}

/* The production edge sum: 32 lanes, each accumulating its own stride of `r`, folded by halves,
 * with `unary` added after the fold. `hp` null is the bigram form (a plane of ones). */
float edge_sum(const uint16_t* pred_row, const uint16_t* hp_row, const uint16_t* succ_row, int R) {
    float lane[kWave] = {};
    for (int r = 0; r < R; ++r) {
        const float p = bf(pred_row[r]);
        const float a = hp_row ? p * bf(hp_row[r]) : p;
        lane[r & (kWave - 1)] = lane[r & (kWave - 1)] + a * bf(succ_row[r]);
    }
    for (int stride = kWave / 2; stride > 0; stride >>= 1)
        for (int i = 0; i < stride; ++i) lane[i] = lane[i] + lane[i + stride];
    return lane[0];
}

struct Tree {
    int32_t tok[kMaxNodes], par[kMaxNodes], depth[kMaxNodes];
    float score[kMaxNodes];
    int total, first, frontier;
};

void run_row(const RadDflashTreeSelectCall& c, int64_t s, int32_t* out) {
    for (int i = 0; i < kCols; ++i) out[i] = -1;
    out[0] = 0;

    const auto* cand = static_cast<const int32_t*>(c.cand);
    const auto* unary = static_cast<const float*>(c.unary);
    const auto* hp = static_cast<const uint16_t*>(c.hp);
    const auto* anchor = static_cast<const int32_t*>(c.anchor);
    const auto* pred = static_cast<const uint16_t*>(c.pred);
    const auto* succ = static_cast<const uint16_t*>(c.succ);

    const int root = anchor[s * c.anchor_stride];
    if (root < 0 || root >= c.V) return;

    Tree t{};
    t.tok[0] = root;
    t.par[0] = -1;
    t.depth[0] = 0;
    t.score[0] = 0.0f;
    t.total = 1;
    t.first = 0;
    t.frontier = 1;

    for (int level = 1; level <= c.steps && t.frontier > 0; ++level) {
        const int64_t row = s * c.steps + level - 1;
        int32_t ctok[kMaxK];
        float cun[kMaxK], corr[kMaxK];
        bool live[kMaxK];
        for (int j = 0; j < c.K; ++j) {
            ctok[j] = cand[row * c.cand_ld + j];
            cun[j] = unary[row * c.unary_ld + j];
            live[j] = ctok[j] >= 0 && ctok[j] < c.V;
            /* A non-finite unary on a live candidate invalidates the row. */
            if (live[j] && !finite_f(cun[j])) return;
        }

        float pairScore[kMaxK * kMaxWidth], pairCorr[kMaxK * kMaxWidth];
        int pairParent[kMaxK * kMaxWidth], pairCand[kMaxK * kMaxWidth];
        int pairs = 0;
        for (int p = 0; p < t.frontier; ++p) {
            const int node = t.first + p;
            const int pid = t.tok[node];
            for (int j = 0; j < c.K; ++j) {
                corr[j] = -std::numeric_limits<float>::infinity();
                if (!live[j]) continue;
                corr[j] = edge_sum(pred + (int64_t)pid * c.pred_ld,
                                   hp ? hp + (int64_t)row * c.hp_ld : nullptr,
                                   succ + (int64_t)ctok[j] * c.succ_ld, int(c.R)) + cun[j];
                if (!finite_f(corr[j])) return;
            }
            float peak = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < c.K; ++j)
                if (live[j] && corr[j] > peak) peak = corr[j];
            for (int j = 0; j < c.K; ++j) {
                if (!live[j]) continue;
                /* One node per token under one parent: the scheduler refuses duplicate siblings.
                 * EVERY instance is compared, so a later one with a higher score evicts an earlier
                 * one and exactly one survives per token. Best raw score, ties to the lower index --
                 * the linear walk's own rule, so a width-1 pick is unchanged. */
                bool beaten = false;
                for (int q = 0; q < c.K && !beaten; ++q) {
                    if (q == j || !live[q] || ctok[q] != ctok[j]) continue;
                    beaten = corr[q] > corr[j] || (corr[q] == corr[j] && q < j);
                }
                if (beaten) continue;
                pairScore[pairs] = t.score[node] + (corr[j] - peak);
                pairCorr[pairs] = corr[j];
                pairParent[pairs] = node;
                pairCand[pairs] = j;
                ++pairs;
            }
        }

        bool used[kMaxK * kMaxWidth] = {};
        int chosen = 0;
        for (int w = 0; w < c.width; ++w) {
            int best = -1;
            for (int i = 0; i < pairs; ++i) {
                if (used[i]) continue;
                if (best < 0 || pairScore[i] > pairScore[best] ||
                    (pairScore[i] == pairScore[best] &&
                     (pairParent[i] < pairParent[best] ||
                      (pairParent[i] == pairParent[best] &&
                       (pairCorr[i] > pairCorr[best] ||
                        (pairCorr[i] == pairCorr[best] && pairCand[i] < pairCand[best]))))))
                    best = i;
            }
            if (best < 0) break;
            used[best] = true;
            const int node = t.total++;
            t.tok[node] = ctok[pairCand[best]];
            t.par[node] = pairParent[best];
            t.depth[node] = level;
            t.score[node] = pairScore[best];
            chosen = w + 1;
        }
        t.first = t.total - chosen;
        t.frontier = chosen;
    }

    bool keep[kMaxNodes + 1] = {};
    keep[0] = true;
    int accepted = 0;
    while (accepted < c.budget) {
        int best = -1;
        float bestValue = 0.0f;
        for (int i = 1; i < t.total; ++i) {
            if (keep[i] || !keep[t.par[i]]) continue;
            const float value = t.score[i] + c.beta * (float)t.depth[i];
            if (best < 0 || value > bestValue ||
                (value == bestValue &&
                 (t.depth[i] < t.depth[best] || (t.depth[i] == t.depth[best] && i < best)))) {
                best = i;
                bestValue = value;
            }
        }
        if (best < 0) break;
        keep[best] = true;
        ++accepted;
    }

    int remap[kMaxNodes + 1];
    for (int i = 0; i <= kMaxNodes; ++i) remap[i] = -1;
    remap[0] = 0;                      /* the anchor is node 0 */
    int node = 0;                      /* root-inclusive id of the last emitted node */
    for (int i = 1; i < t.total; ++i) {
        if (!keep[i]) continue;
        const int parent = remap[t.par[i]];
        if (parent < 0 || t.par[i] >= i) {
            for (int k = 0; k < kCols; ++k) out[k] = -1;
            out[0] = 0;
            return;
        }
        remap[i] = ++node;
        /* The packed row, capacity-generic, as `dloop_tree.hip`, `ref_dloop.cpp` and
         * `dloop_tree_from_packed` all agree: node i's token at 1+i, its root-inclusive parent at
         * 1+capacity+i, the anchor's sentinel at 1+capacity. At capacity 16 the parent columns are
         * 17..32 and the sentinel is column 17. */
        out[1 + node] = t.tok[i];
        out[1 + kCapacity + node] = parent;
    }
    out[0] = node + 1;
    out[1] = t.tok[0];
    out[1 + kCapacity] = -1;          /* the anchor's sentinel parent */
}

}  /* namespace */

int rad_ref_dflash_tree_select(const RadArgs* a, RadStream) {
    RadDflashTreeSelectCall c{};
    const int rc = rad_dflash_tree_select_args(a, &c);
    if (rc != RAD_OK) return rc;
    for (int64_t s = 0; s < c.M; ++s)
        run_row(c, s, static_cast<int32_t*>(c.out) + s * c.out_ld);
    return RAD_OK;
}

namespace {

RadKernelInfo host_row() {
    RadKernelInfo r = rad_dflash_tree_select_row(rad_ref_dflash_tree_select);
    r.name = "ref_dflash_tree_select";
    r.domain = RAD_DOMAIN_HOST;
    r.priority = 0;
    return r;
}

}  /* namespace */

int rad_ref_dflash_tree_select_count(void) { return 1; }

const RadKernelInfo* rad_ref_dflash_tree_select_at(int i) {
    static const RadKernelInfo row = host_row();
    return i == 0 ? &row : nullptr;
}
