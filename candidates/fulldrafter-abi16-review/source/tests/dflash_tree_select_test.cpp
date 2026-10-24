/* tests/dflash_tree_select_test.cpp -- the HOST behaviour and contract of `dflash_tree_select`.
 *
 * The op's interesting properties are discrete: which nodes survive, in which order, with which
 * parent, and whether the single-path case reproduces the shipping selector's tokens. None of that
 * needs a card. This drives the HOST ROW (`ref_dflash_tree_select`) directly, which is the op's
 * independent implementation and the side the device comparator is checked against.
 *
 * WHAT IT PINS:
 *
 *   * the packed layout -- count, root token, root parent -1, root-inclusive draft parents, unused
 *     slots -1 -- because the engine refuses a tree that does not have it;
 *   * `parent(i) < i` for every emitted node, which is what the verifier's row order needs;
 *   * the greedy child of a parent comes FIRST among its siblings, at a budget that forces real
 *     pruning -- the property the QSA exactness fallback relies on when it linearises the tree;
 *   * `steps` <= 3, `tree_budget` <= 15, `tree_width` <= top_k are refused, not clamped, and the
 *     boundary values are accepted on operands sized for them;
 *   * the single-path call reproduces `dflash_select`'s walk, computed here in the PRODUCTION
 *     reduction order (32 lanes, halving fold, unary last);
 *   * that reduction order is discriminating: a scene where the fold and an ascending sum rank two
 *     candidates OPPOSITELY, and the tree picks the fold's answer;
 *   * hp absent and hp present differ, a plane of exact ones is bit-identical to absence, and a
 *     non-finite unary on a live candidate or an out-of-vocabulary anchor writes count 0;
 *   * a tied score resolves to the lower candidate index, deterministically, on a second run;
 *   * `depth_penalty` must be finite and <= 0, and a negative one prefers shallow nodes;
 *   * sequences in one launch are independent, including a non-unit anchor stride;
 *   * an output overlapping an input is refused.
 *
 * NOTHING HERE CLAIMS DEVICE PARITY. tests/dflash_tree_gpu_test.cpp compares the device row's
 * bytes against this oracle.
 *
 * Link: the op definition and its host row only -- no core, no HIP, no device.
 */
#include "rad_test.h"

#include "mine/dflash_tree_select.h"
#include "mine/ref_dflash_tree.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int kCols = RAD_DFLASH_TREE_OUT_COLS;
/* The DECLARED node capacity of the drafter tree -- 16 slots -- which is what the row pitch is
 * derived from. A row is 1 + 2*kCap columns and node i's parent sits at 1 + kCap + i, so at 16 the
 * parent block is 17..32 and the anchor's sentinel is column 17. Every assertion uses this
 * constant rather than a literal so the test stays right for a different capacity. */
constexpr int kCap = (kCols - 1) / 2;

uint16_t bf(float v) { return rad_f32_to_bf16(v); }

struct Scene {
    int64_t M = 1, steps = 2, K = 4, R = 3, V = 8, astr = 1, budget = 15, width = 2;
    double beta = 0.0;

    std::vector<int32_t> cand;      /* [M*steps, K] */
    std::vector<float>   unary;
    std::vector<uint16_t> hp;       /* empty = absent */
    std::vector<int32_t> anchor;    /* [M*astr] */
    std::vector<uint16_t> pred;     /* [V, R] */
    std::vector<uint16_t> succ;
    std::vector<int32_t> out;       /* [M, 33] */

    int32_t& C(int64_t m, int64_t l, int64_t j) { return cand[size_t((m * steps + l) * K + j)]; }
    float&   U(int64_t m, int64_t l, int64_t j) { return unary[size_t((m * steps + l) * K + j)]; }
    int32_t* row(int64_t m) { return out.data() + m * kCols; }

    std::vector<RadParam> params() const {
        return { RAD_INT("M", M), RAD_INT("steps", steps), RAD_INT("top_k", K),
                 RAD_INT("rank", R), RAD_INT("n_vocab", V), RAD_INT("anchor_stride", astr),
                 RAD_INT("tree_budget", budget), RAD_INT("tree_width", width),
                 RAD_F64("depth_penalty", beta), RAD_STR("dtype", "bf16") };
    }

    void bind(RadTensor* t) const {
        auto shape2 = [](RadTensor& x, uint32_t dt, int64_t a, int64_t b) {
            x.dtype = dt; x.rank = 2; x.shape[0] = a; x.shape[1] = b;
            x.stride[0] = b; x.stride[1] = 1;
        };
        /* `RadTensor.data` is void*, so a const plane has to be laundered once here. Nothing in
         * this file writes through it: the op's only OUT operand is the tree. */
        auto p = [](const auto* q) { return const_cast<void*>(static_cast<const void*>(q)); };
        shape2(t[RAD_DFLASH_TREE_CAND], RAD_I32, M * steps, K);
        t[RAD_DFLASH_TREE_CAND].data = p(cand.data());
        shape2(t[RAD_DFLASH_TREE_UNARY], RAD_F32, M * steps, K);
        t[RAD_DFLASH_TREE_UNARY].data = p(unary.data());
        /* Absence is a null `data` on a present operand, which is the ABI's spelling. */
        shape2(t[RAD_DFLASH_TREE_HP], RAD_BF16, M * steps, R);
        t[RAD_DFLASH_TREE_HP].data = hp.empty() ? nullptr : p(hp.data());
        t[RAD_DFLASH_TREE_ANCHOR].dtype = RAD_I32;
        t[RAD_DFLASH_TREE_ANCHOR].rank = 1;
        t[RAD_DFLASH_TREE_ANCHOR].shape[0] = M * astr;
        t[RAD_DFLASH_TREE_ANCHOR].stride[0] = 1;
        t[RAD_DFLASH_TREE_ANCHOR].data = p(anchor.data());
        shape2(t[RAD_DFLASH_TREE_PRED], RAD_BF16, V, R);
        t[RAD_DFLASH_TREE_PRED].data = p(pred.data());
        shape2(t[RAD_DFLASH_TREE_SUCC], RAD_BF16, V, R);
        t[RAD_DFLASH_TREE_SUCC].data = p(succ.data());
        shape2(t[RAD_DFLASH_TREE_OUT], RAD_I32, M, kCols);
        t[RAD_DFLASH_TREE_OUT].data = p(out.data());
    }

    int run(std::vector<RadParam>* p = nullptr) {
        RadTensor t[RAD_DFLASH_TREE_OPERANDS]{};
        bind(t);
        std::vector<RadParam> own;
        if (!p) { own = params(); p = &own; }
        RadArgs a{};
        a.t = t;
        a.n_t = RAD_DFLASH_TREE_OPERANDS;
        a.p = p->data();
        a.n_p = int(p->size());
        return rad_ref_dflash_tree_select(&a, nullptr);
    }
};

/* ---- layout invariants every accepted call must satisfy ----
 * The canonical root-inclusive packing, capacity-generic, with `kCap` the declared node capacity:
 *   [0] count = 1 + draft nodes, [1+i] node i's token, [1+kCap+i] node i's parent, where 0 = the
 *   anchor, and [1+kCap] == -1 is the anchor's SENTINEL parent. At capacity 16 the parent columns
 *   are 17..32 and the sentinel is column 17.
 * Read-only: it validates a row the op produced. */
void check_tree(const Scene& s, const int32_t* out, int64_t m = 0) {
    const int count = out[0];
    CHECK(count >= 1 && count <= kCap);
    if (count < 1 || count > kCap) return;
    CHECK_EQ(out[1], s.anchor[size_t(m * s.astr)]);   /* node 0 is the anchor */
    CHECK_EQ(out[1 + kCap], -1);                     /* THE ROOT SENTINEL the validator demands */
    CHECK(count - 1 <= s.budget);
    /* Unused slots are -1 in both halves, from the last node to the end of each block. */
    for (int i = count; i < kCap; ++i) {
        CHECK_EQ(out[1 + i], -1);
        CHECK_EQ(out[1 + kCap + i], -1);
    }
    /* Every node after the root has a parent that is a real, EARLIER node (0 = the anchor), which
     * is what lets the validator build the tree in row order. */
    for (int i = 1; i < count; ++i) {
        const int parent = out[1 + kCap + i];
        CHECK(parent >= 0 && parent < i);
        CHECK(out[1 + i] >= 0 && out[1 + i] < s.V);
    }
    /* NO TWO SIBLINGS PROPOSE THE SAME TOKEN: `set_draft_tree` refuses such a tree. */
    for (int j = 1; j < count; ++j)
        for (int i = 1; i < j; ++i)
            CHECK(!(out[1 + kCap + i] == out[1 + kCap + j] && out[1 + i] == out[1 + j]));
}

/* ---- the production edge sum, computed here independently ---- */
/* `dflash_select`'s order, not a textbook sum: 32 lanes striding by 32, halving fold, unary last.
 * A tree that reproduced the FORMULA but not the order disagrees here on near-ties and
 * cancellation, which is what this file's order test is for. */
float edge_sum(const Scene& s, int64_t m, int64_t level, int32_t pid, int32_t token) {
    float lane[32] = {};
    for (int64_t r = 0; r < s.R; ++r) {
        const float p = rad_bf16_to_f32(s.pred[size_t(pid * s.R + r)]);
        const float h = s.hp.empty()
                            ? 1.0f
                            : rad_bf16_to_f32(s.hp[size_t((m * s.steps + level) * s.R + r)]);
        const float q = rad_bf16_to_f32(s.succ[size_t(token * s.R + r)]);
        const int l = int(r & 31);
        lane[l] = lane[l] + (p * h) * q;
    }
    for (int stride = 16; stride > 0; stride >>= 1)
        for (int i = 0; i < stride; ++i) lane[i] = lane[i] + lane[i + stride];
    return lane[0];
}

std::vector<int32_t> linear_walk(const Scene& s, int64_t m, int steps) {
    std::vector<int32_t> picked;
    int32_t pid = s.anchor[size_t(m * s.astr)];
    for (int64_t l = 0; l < steps; ++l) {
        float best = -std::numeric_limits<float>::infinity();
        int32_t bestTok = -1;
        for (int64_t j = 0; j < s.K; ++j) {
            const int32_t tok = s.cand[size_t((m * s.steps + l) * s.K + j)];
            if (tok < 0 || tok >= s.V) continue;
            const float sc =
                edge_sum(s, m, l, pid, tok) + s.unary[size_t((m * s.steps + l) * s.K + j)];
            /* Strictly greater: a tie goes to the lower INDEX, `row_topk`'s own rule. */
            if (sc > best) { best = sc; bestTok = tok; }
        }
        picked.push_back(bestTok);
        pid = bestTok;
    }
    return picked;
}

/* A small, non-degenerate scene: V = 8, R = 3, K = 4, three levels, a non-trivial hp plane. */
Scene make_scene() {
    Scene s;
    s.steps = 3;
    s.K = 4;
    s.R = 3;
    s.V = 8;
    s.M = 1;
    s.width = 2;
    s.budget = 15;
    s.anchor.assign(1, 1);
    s.cand.assign(size_t(s.M * s.steps * s.K), 0);
    s.unary.assign(size_t(s.M * s.steps * s.K), 0.0f);
    s.pred.assign(size_t(s.V * s.R), 0);
    s.succ.assign(size_t(s.V * s.R), 0);
    s.out.assign(size_t(s.M * kCols), -99);
    for (int64_t l = 0; l < s.steps; ++l)
        for (int64_t j = 0; j < s.K; ++j) {
            s.C(0, l, j) = int32_t((l * 3 + j * 2 + 1) % s.V);
            s.U(0, l, j) = 0.5f * float(j) - float(l) * 0.25f;
        }
    for (int64_t v = 0; v < s.V; ++v)
        for (int64_t r = 0; r < s.R; ++r) {
            s.pred[size_t(v * s.R + r)] = bf(0.25f * float(v + 1) - 0.5f * float(r));
            s.succ[size_t(v * s.R + r)] = bf(0.125f * float(r + 1) + 0.0625f * float(v));
        }
    /* A NON-TRIVIAL hp plane, so `hp` present and absent are different numbers and a kernel that
     * ignored the operand is caught: a plane of ones would make the two equal by construction. */
    s.hp.assign(size_t(s.M * s.steps * s.R), 0);
    for (int64_t l = 0; l < s.steps; ++l)
        for (int64_t r = 0; r < s.R; ++r)
            s.hp[size_t(l * s.R + r)] = bf(1.0f + 0.25f * float(r) - 0.125f * float(l));
    return s;
}

}  /* namespace */

TEST(single_path_reproduces_the_linear_walk) {
    Scene s = make_scene();
    s.width = 1;
    s.budget = RAD_DFLASH_TREE_MAX_BUDGET;
    REQUIRE_EQ(s.run(), RAD_OK);
    const int32_t* out = s.row(0);
    check_tree(s, out);
    /* A chain: `steps` nodes, each a child of the previous. */
    REQUIRE_EQ(out[0], int(s.steps) + 1);
    const std::vector<int32_t> want = linear_walk(s, 0, s.steps);
    for (int i = 1; i <= int(s.steps); ++i) {
        CHECK_EQ(out[1 + i], want[size_t(i - 1)]);
        /* Node i's parent is node i-1, root-inclusive: the anchor (0) for node 1, then the
         * previous draft node. A chain. */
        CHECK_EQ(out[1 + kCap + i], i - 1);
    }
}

TEST(multi_path_tree_is_prefix_closed_and_topo_ordered) {
    Scene s = make_scene();
    s.width = 2;
    s.budget = 4;                       /* forces real pruning over 1+2+4+8 candidate nodes */
    REQUIRE_EQ(s.run(), RAD_OK);
    const int32_t* out = s.row(0);
    check_tree(s, out);
    CHECK(out[0] >= 2);
    CHECK(out[0] <= s.budget + 1);
    CHECK_EQ(out[1 + kCap], -1);
    /* Not a chain: two nodes share a parent. */
    bool sibling = false;
    for (int i = 1; i < out[0] && !sibling; ++i)
        for (int j = i + 1; j < out[0] && !sibling; ++j) sibling = out[1 + kCap + i] == out[1 + kCap + j];
    CHECK(sibling);
}

TEST(hp_is_used_and_absent_hp_is_a_plane_of_ones) {
    /* A SCENE BUILT SO THAT THE SIGN OF hp DECIDES THE WINNER. With `rank = 1`, the edge is
     * `pred[pid] * hp[l][0] * succ[token][0]`; giving the two candidates succ terms of 1 and 2 and
     * equal unary terms makes hp = +1 pick the larger and hp = -1 pick the smaller. A fixture where
     * the codebook dominates, or where both candidates share a succ row, would pass whether or not
     * hp was read at all. */
    Scene s;
    s.steps = 1;
    s.K = 2;
    s.R = 1;
    s.V = 4;
    s.M = 1;
    s.width = 2;
    s.budget = 15;
    s.anchor.assign(1, 0);
    s.cand.assign(size_t(s.M * s.steps * s.K), 0);
    s.unary.assign(size_t(s.M * s.steps * s.K), 0.0f);
    s.hp.assign(size_t(s.M * s.steps * s.R), bf(1.0f));
    s.pred.assign(size_t(s.V * s.R), bf(1.0f));
    s.succ.assign(size_t(s.V * s.R), 0);
    s.out.assign(size_t(s.M * kCols), -99);
    s.C(0, 0, 0) = 0;
    s.C(0, 0, 1) = 1;
    s.succ[0] = bf(1.0f);              /* token 0 */
    s.succ[1] = bf(2.0f);              /* token 1 */

    REQUIRE_EQ(s.run(), RAD_OK);
    const std::vector<int32_t> positive(s.row(0), s.row(0) + kCols);
    check_tree(s, s.row(0));
    /* hp = +1: token 1 (edge 2) beats token 0 (edge 1), so node 1 is token 1. */
    REQUIRE_EQ(positive[0], 3);
    CHECK_EQ(positive[1 + 1], 1);

    Scene n = s;
    n.hp[0] = bf(-1.0f);
    REQUIRE_EQ(n.run(), RAD_OK);
    check_tree(n, n.row(0));
    const std::vector<int32_t> negative(n.row(0), n.row(0) + kCols);
    /* hp = -1: token 0 (edge -1) beats token 1 (edge -2), so the ORDER reverses. An ignored hp
     * would give the same row as `positive`. */
    CHECK(positive != negative);
    CHECK_EQ(negative[1 + 1], 0);

    /* A PLANE OF EXACT ONES IS THE ABSENT FORM, BIT FOR BIT: `p * 1.0f == p` in IEEE and the
     * absence path uses `p` directly. Fails if absence multiplies by something else or reads a
     * different row. */
    Scene b = s;
    b.hp.clear();
    REQUIRE_EQ(b.run(), RAD_OK);
    check_tree(b, b.row(0));
    CHECK(std::memcmp(s.row(0), b.row(0), size_t(kCols) * sizeof(int32_t)) == 0);
}

TEST(tied_scores_resolve_to_the_lower_candidate_index_deterministically) {
    Scene s = make_scene();
    s.steps = 1;
    s.K = 4;
    s.width = 2;
    s.budget = 15;
    s.hp.clear();
    /* Every candidate scores EXACTLY the same: same unary, and a succ plane constant across the
     * vocabulary, so every edge is the same float bit for bit. */
    for (int64_t j = 0; j < s.K; ++j) {
        s.C(0, 0, j) = int32_t(2 + j);
        s.U(0, 0, j) = 1.0f;
    }
    for (int64_t v = 0; v < s.V; ++v)
        for (int64_t r = 0; r < s.R; ++r) s.succ[size_t(v * s.R + r)] = bf(0.25f);
    REQUIRE_EQ(s.run(), RAD_OK);
    const int32_t* out = s.row(0);
    check_tree(s, out);
    REQUIRE_EQ(out[0], 3);
    /* Nodes 1 and 2, in index order, both children of the anchor (root-inclusive 0). */
    CHECK_EQ(out[1 + 1], 2);
    CHECK_EQ(out[1 + 2], 3);
    CHECK_EQ(out[1 + kCap + 1], 0);
    CHECK_EQ(out[1 + kCap + 2], 0);

    Scene again = s;
    REQUIRE_EQ(again.run(), RAD_OK);
    CHECK(std::memcmp(again.row(0), out, size_t(kCols) * sizeof(int32_t)) == 0);
}

TEST(repeated_candidate_ids_under_one_parent_become_one_best_scoring_node) {
    /* THE SCHEDULER REFUSES DUPLICATE SIBLINGS (`core/sched/scheduler.cpp`, `set_draft_tree`), so a
     * repeated id in `row_topk`'s output must collapse to a single node. Two properties matter:
     *
     *   1. NO TWO SIBLINGS SHARE A TOKEN, which is what the consumer checks; and
     *   2. THE INSTANCE KEPT IS THE HIGHEST-SCORING ONE, not merely the first occurrence -- so a
     *      later duplicate carrying a larger unary term must win.
     *
     * The scene below is built so that (2) is decidable: token 4 appears at indices 0 and 1 and its
     * SECOND instance scores higher; token 6 appears at 2 and 3 and its FIRST scores higher; token 7
     * appears at 4 and 5 and its SECOND scores higher. Width is 4 and the budget allows every
     * distinct token, so the tree shows one node per token and nothing is hidden by pruning. */
    Scene s;
    s.steps = 1;
    s.K = 6;
    s.R = 3;
    s.V = 8;
    s.M = 1;
    s.width = 4;
    s.budget = 15;
    s.anchor.assign(1, 0);
    s.cand.assign(size_t(s.M * s.steps * s.K), 0);
    s.unary.assign(size_t(s.M * s.steps * s.K), 0.0f);
    s.hp.assign(size_t(s.M * s.steps * s.R), bf(1.0f));
    s.pred.assign(size_t(s.V * s.R), bf(0.5f));
    s.succ.assign(size_t(s.V * s.R), bf(0.25f));
    s.out.assign(size_t(s.M * kCols), -99);
    s.anchor[0] = 0;

    /* tokens:  4, 4, 6, 6, 7, 7  -- three distinct tokens, each appearing twice. */
    const int32_t toks[6] = { 4, 4, 6, 6, 7, 7 };
    /* unary:    1, 3, 5, 2, 0, 9  -- for token 4 the SECOND wins, for 6 the FIRST wins, for 7 the
     * second wins. */
    const float un[6] = { 1.0f, 3.0f, 5.0f, 2.0f, 0.0f, 9.0f };
    for (int j = 0; j < 6; ++j) {
        s.C(0, 0, j) = toks[j];
        s.U(0, 0, j) = un[j];
    }

    REQUIRE_EQ(s.run(), RAD_OK);
    const int32_t* out = s.row(0);
    check_tree(s, out);

    /* Three distinct draft nodes plus the anchor. */
    REQUIRE_EQ(out[0], 4);
    std::vector<int32_t> got;
    for (int i = 1; i < out[0]; ++i) got.push_back(out[1 + i]);
    std::sort(got.begin(), got.end());
    CHECK(std::adjacent_find(got.begin(), got.end()) == got.end());   /* no duplicate siblings */
    CHECK_EQ(got[0], 4);
    CHECK_EQ(got[1], 6);
    CHECK_EQ(got[2], 7);

    /* THE INSTANCE RETAINED IS THE BEST ONE, and the evidence is the ORDER: nodes are emitted by
     * descending score, so the ranking reflects the retained instance's unary term. Token 7's
     * retained instance has 9, token 6's has 5, token 4's has 3 -- so the order is 7, 6, 4. Had the
     * kernel kept the first occurrence of each token the retained terms would be 1, 5, 0 and the
     * order would be 6, 4, 7. */
    CHECK_EQ(out[1 + 1], 7);
    CHECK_EQ(out[1 + 2], 6);
    CHECK_EQ(out[1 + 3], 4);

    /* AND THE SINGLE-PATH CASE STILL MATCHES THE LINEAR WALK ON THE SAME BYTES. Width 1 picks the
     * token with the highest edge score overall -- token 7 via its second instance -- which is what
     * `dflash_select` picks, so deduplication does not disturb the equivalence. */
    Scene w = s;
    w.width = 1;
    w.budget = 15;
    REQUIRE_EQ(w.run(), RAD_OK);
    const std::vector<int32_t> want = linear_walk(s, 0, s.steps);
    // `linear_walk` scores every in-range candidate independently and keeps the best token.
    CHECK_EQ(w.row(0)[0], 2);
    CHECK_EQ(w.row(0)[1 + 1], want[0]);
    CHECK_EQ(w.row(0)[1 + 1], 7);
}

TEST(candidate_ids_outside_the_vocabulary_are_never_emitted) {
    Scene s = make_scene();
    s.steps = 1;
    s.width = 4;
    /* Two impossible ids with the HIGHEST unary terms: if they were offered they would win. */
    s.C(0, 0, 0) = -1;
    s.U(0, 0, 0) = 100.0f;
    s.C(0, 0, 1) = int32_t(s.V);
    s.U(0, 0, 1) = 100.0f;
    REQUIRE_EQ(s.run(), RAD_OK);
    const int32_t* out = s.row(0);
    check_tree(s, out);
    for (int i = 1; i < out[0]; ++i) {
        CHECK(out[1 + i] >= 0);
        CHECK(out[1 + i] < s.V);
    }
    CHECK(out[0] >= 2);                 /* the valid candidates still built a tree */
}

TEST(an_invalid_anchor_writes_count_zero) {
    Scene s = make_scene();
    s.anchor[0] = int32_t(s.V);
    REQUIRE_EQ(s.run(), RAD_OK);
    CHECK_EQ(s.row(0)[0], 0);
    for (int i = 1; i < kCols; ++i) CHECK_EQ(s.row(0)[i], -1);

    s.anchor[0] = -1;
    REQUIRE_EQ(s.run(), RAD_OK);
    CHECK_EQ(s.row(0)[0], 0);
}

TEST(a_non_finite_unary_on_a_live_candidate_writes_count_zero) {
    Scene s = make_scene();
    s.U(0, 1, 2) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_EQ(s.run(), RAD_OK);
    CHECK_EQ(s.row(0)[0], 0);

    /* The same NaN on an id the op does not offer is not a live candidate, so the tree stands. */
    Scene t = make_scene();
    t.C(0, 1, 2) = -1;
    t.U(0, 1, 2) = std::numeric_limits<float>::infinity();
    REQUIRE_EQ(t.run(), RAD_OK);
    CHECK(t.row(0)[0] >= 1);
}

TEST(the_greedy_chain_survives_pruning_and_stays_first_child) {
    /* THE PROPERTY THE QSA EXACTNESS FALLBACK DEPENDS ON. With `width > 1` and a budget that prunes
     * branches, the width-1 tree's nodes must all appear, still connected, and each greedy node
     * must be the FIRST child of its parent in the packed output.
     *
     * It holds because a parent's own best candidate gets an increment of exactly zero, so every
     * node on the greedy path scores the same as the root -- and ranking by score descending puts
     * that whole path ahead of every branch, whose increments are negative. */
    Scene chain = make_scene();
    chain.width = 1;
    chain.budget = RAD_DFLASH_TREE_MAX_BUDGET;
    REQUIRE_EQ(chain.run(), RAD_OK);
    const int32_t* c = chain.row(0);
    REQUIRE_EQ(c[0], int(chain.steps) + 1);
    std::vector<int32_t> tokens;
    for (int i = 1; i < c[0]; ++i) tokens.push_back(c[1 + i]);

    Scene wide = make_scene();
    wide.width = 3;
    wide.budget = 5;      /* the level tree has 9 draft nodes, so branches really are dropped */
    REQUIRE_EQ(wide.run(), RAD_OK);
    const int32_t* w = wide.row(0);
    check_tree(wide, w);

    /* Walk the greedy chain through the wide tree. `previous` is a ROOT-INCLUSIVE node index (0 is
     * the anchor). Each chain token must be present under that parent, and it must be the FIRST
     * child of it, which is the property `dloop_greedy_chain` reads. */
    int previous = 0;
    for (size_t k = 0; k < tokens.size(); ++k) {
        int found = -1;
        for (int i = 1; i < w[0]; ++i)
            if (w[1 + i] == tokens[k] && w[1 + kCap + i] == previous) { found = i; break; }
        CHECK(found > 0);
        if (found <= 0) return;
        for (int i = 1; i < found; ++i) CHECK(w[1 + kCap + i] != previous);   /* no earlier sibling */
        previous = found;
    }
}

TEST(budget_level_and_width_bounds_are_refused_not_clamped) {
    Scene s = make_scene();
    auto refused = [&](std::vector<RadParam> p) {
        std::fill(s.out.begin(), s.out.end(), -99);
        return s.run(&p);
    };
    std::vector<RadParam> p = s.params();
    auto set = [&](const char* key, long long v) {
        for (auto& e : p)
            if (e.key && std::strcmp(e.key, key) == 0) e.ival = v;
    };

    set("steps", RAD_DFLASH_TREE_MAX_STEPS + 1);
    CHECK_EQ(refused(p), RAD_E_UNSUPPORTED);
    p = s.params(); set("tree_budget", RAD_DFLASH_TREE_MAX_BUDGET + 1);
    CHECK_EQ(refused(p), RAD_E_UNSUPPORTED);
    p = s.params(); set("top_k", RAD_DFLASH_TREE_MAX_K + 1);
    CHECK_EQ(refused(p), RAD_E_UNSUPPORTED);
    p = s.params(); set("tree_width", s.K + 1);
    CHECK_EQ(refused(p), RAD_E_UNSUPPORTED);
    p = s.params(); set("tree_budget", 0);
    CHECK_EQ(refused(p), RAD_E_UNSUPPORTED);
    p = s.params(); set("steps", 0);
    CHECK_EQ(refused(p), RAD_E_UNSUPPORTED);

    /* A refusal must not have written a tree. */
    CHECK_EQ(s.out[0], -99);

    /* THE BOUNDARY VALUES THEMSELVES ARE ACCEPTED, on operands sized for them: steps 3, top_k 16
     * and budget 15 together. A clamp passes this too; the refusals above are what a clamp fails. */
    Scene edge;
    edge.steps = RAD_DFLASH_TREE_MAX_STEPS;
    edge.K = RAD_DFLASH_TREE_MAX_K;
    edge.R = 3;
    edge.V = 32;
    edge.width = RAD_DFLASH_TREE_MAX_WIDTH;
    edge.budget = RAD_DFLASH_TREE_MAX_BUDGET;
    edge.anchor.assign(1, 1);
    edge.cand.assign(size_t(edge.M * edge.steps * edge.K), 0);
    edge.unary.assign(size_t(edge.M * edge.steps * edge.K), 0.0f);
    edge.hp.assign(size_t(edge.M * edge.steps * edge.R), bf(1.0f));
    edge.pred.assign(size_t(edge.V * edge.R), bf(0.25f));
    edge.succ.assign(size_t(edge.V * edge.R), bf(0.5f));
    edge.out.assign(size_t(edge.M * kCols), -99);
    for (int64_t l = 0; l < edge.steps; ++l)
        for (int64_t j = 0; j < edge.K; ++j) {
            edge.C(0, l, j) = int32_t((l * 7 + j + 2) % edge.V);
            edge.U(0, l, j) = 0.125f * float(j) - 0.5f * float(l);
        }
    REQUIRE_EQ(edge.run(), RAD_OK);
    const int32_t* e = edge.row(0);
    check_tree(edge, e);
    CHECK_EQ(e[0], RAD_DFLASH_TREE_MAX_BUDGET + 1);   /* 15 nodes plus the root fill the tree */
}

TEST(the_reduction_is_the_production_lane_fold_not_an_ascending_sum) {
    /* THE ORDER IS PART OF THE CONTRACT and needs a test a wrong order fails.
     *
     * `dflash_select` accumulates lane `r & 31` over `r = lane, lane+32, ...` and folds by halves,
     * which is a different summation TREE from an ascending walk over r. The two planes below are
     * chosen so that the two orders rank them OPPOSITELY -- verified numerically before this case
     * was written, not assumed:
     *
     *   candidate A   succ has 2 at r=21, 2^24 at r=27, 1 at r=42, 3 at r=61
     *                 fold = 2^24 + 4       ascending = 2^24 + 8
     *   candidate B   succ has 4 at r=44, 2 at r=53, 2^24 at r=60
     *                 fold = 2^24 + 6       ascending = 2^24 + 6
     *
     * so fold ranks A < B while ascending ranks A > B. Near 2^24 the ulp is 2, so a small term
     * survives or vanishes according to which lane it lands in and in what order -- exactly the
     * distinction being tested. The totals are recomputed here from the planes. */
    Scene s;
    s.steps = 1;
    s.K = 2;
    s.R = 64;
    s.V = 8;
    s.M = 1;
    s.width = 1;
    s.budget = 15;
    s.anchor.assign(1, 0);
    s.cand.assign(size_t(s.M * s.steps * s.K), 0);
    s.unary.assign(size_t(s.M * s.steps * s.K), 0.0f);
    s.hp.assign(size_t(s.M * s.steps * s.R), bf(1.0f));
    s.pred.assign(size_t(s.V * s.R), bf(1.0f));
    s.succ.assign(size_t(s.V * s.R), 0);
    s.out.assign(size_t(s.M * kCols), -99);
    s.C(0, 0, 0) = 0;
    s.C(0, 0, 1) = 1;

    const float big = 16777216.0f;                        /* 2^24; ulp is 2 */
    auto put = [&](int32_t token, int64_t r, float v) { s.succ[size_t(token * s.R + r)] = bf(v); };
    put(0, 21, 2.0f);  put(0, 27, big);  put(0, 42, 1.0f);  put(0, 61, 3.0f);
    put(1, 44, 4.0f);  put(1, 53, 2.0f);  put(1, 60, big);

    auto total = [&](int32_t token, bool folded) {
        if (folded) {
            float lane[32] = {};
            for (int64_t r = 0; r < s.R; ++r) {
                const int l = int(r & 31);
                lane[l] = lane[l] + rad_bf16_to_f32(s.succ[size_t(token * s.R + r)]);
            }
            for (int stride = 16; stride > 0; stride >>= 1)
                for (int i = 0; i < stride; ++i) lane[i] = lane[i] + lane[i + stride];
            return lane[0];
        }
        float acc = 0.0f;
        for (int64_t r = 0; r < s.R; ++r)
            acc = acc + rad_bf16_to_f32(s.succ[size_t(token * s.R + r)]);
        return acc;
    };
    CHECK(total(0, true) < total(1, true));      /* fold:      A loses, B wins */
    CHECK(total(0, false) > total(1, false));    /* ascending: A wins, B loses */

    REQUIRE_EQ(s.run(), RAD_OK);
    const int32_t* out = s.row(0);
    check_tree(s, out);
    REQUIRE_EQ(out[0], 2);
    CHECK_EQ(out[1 + 1], 1);                    /* B: the fold's answer, not the ascending one */
}

TEST(depth_penalty_must_be_finite_and_non_positive) {
    Scene s = make_scene();
    std::vector<RadParam> p = s.params();
    auto set_beta = [&](double v) {
        for (auto& e : p)
            if (e.key && std::strcmp(e.key, "depth_penalty") == 0) e.dval = v;
    };
    set_beta(0.5);
    CHECK_EQ(s.run(&p), RAD_E_INVAL);
    set_beta(std::numeric_limits<double>::quiet_NaN());
    CHECK_EQ(s.run(&p), RAD_E_INVAL);
    set_beta(-0.25);
    REQUIRE_EQ(s.run(&p), RAD_OK);
    check_tree(s, s.row(0));

    /* A large negative beta prefers SHALLOW nodes: with budget 2 the root's own children must
     * take the slots. */
    Scene d = make_scene();
    d.budget = 2;
    std::vector<RadParam> dp = d.params();
    for (auto& e : dp)
        if (e.key && std::strcmp(e.key, "depth_penalty") == 0) e.dval = -1.0;
    REQUIRE_EQ(d.run(&dp), RAD_OK);
    const int32_t* out = d.row(0);
    check_tree(d, out);
    CHECK(out[0] <= 3);
    if (out[0] >= 2) CHECK_EQ(out[1 + kCap + 1], 0);   /* node 1 is the anchor's child */
}

TEST(multiple_sequences_are_independent) {
    Scene s = make_scene();
    s.M = 2;
    s.steps = 2;
    s.anchor.assign(size_t(s.M * s.astr), 0);
    s.anchor[0] = 1;
    s.anchor[1] = 5;
    s.cand.assign(size_t(s.M * s.steps * s.K), 0);
    s.unary.assign(size_t(s.M * s.steps * s.K), 0.0f);
    s.hp.assign(size_t(s.M * s.steps * s.R), bf(1.0f));
    s.out.assign(size_t(s.M * kCols), -99);
    for (int64_t m = 0; m < s.M; ++m)
        for (int64_t l = 0; l < s.steps; ++l)
            for (int64_t j = 0; j < s.K; ++j) {
                s.C(m, l, j) = int32_t((m + l * 3 + j * 2 + 1) % s.V);
                s.U(m, l, j) = 0.5f * float(j) + float(m) - float(l) * 0.25f;
            }
    REQUIRE_EQ(s.run(), RAD_OK);
    check_tree(s, s.row(0), 0);
    check_tree(s, s.row(1), 1);
    CHECK_EQ(s.row(0)[1 + kCap], -1);
    CHECK_EQ(s.row(1)[1 + kCap], -1);
    /* The rows differ, so a kernel sharing state across sequences fails here. */
    CHECK(std::memcmp(s.row(0), s.row(1), size_t(kCols) * sizeof(int32_t)) != 0);

    /* A non-unit anchor stride reads at m * stride, not at m. */
    Scene t = s;
    t.astr = 4;
    t.anchor.assign(size_t(t.M * t.astr), 0);
    t.anchor[0] = 1;
    t.anchor[size_t(t.astr)] = 6;
    REQUIRE_EQ(t.run(), RAD_OK);
    check_tree(t, t.row(0), 0);
    check_tree(t, t.row(1), 1);
}

TEST(an_output_overlapping_an_input_is_refused) {
    Scene s = make_scene();
    RadTensor t[RAD_DFLASH_TREE_OPERANDS]{};
    s.bind(t);
    /* THE ALIAS IS THE ONLY FAULT. The output keeps its [M, 33] shape -- shrinking it to the
     * candidate plane's shape would trip the SHAPE check first and this would stop testing alias
     * detection -- and only its BASE POINTER is moved onto the candidate plane, so the two
     * genuinely share bytes. */
    t[RAD_DFLASH_TREE_OUT].data = s.cand.data();
    std::vector<RadParam> p = s.params();
    RadArgs a{};
    a.t = t;
    a.n_t = RAD_DFLASH_TREE_OPERANDS;
    a.p = p.data();
    a.n_p = int(p.size());
    RadDflashTreeSelectCall call{};
    CHECK_EQ(rad_dflash_tree_select_args(&a, &call), RAD_E_INVAL);
}

RAD_TEST_MAIN()
