/* fulldrafter_tree_test.cpp -- CPU tests for the root-fork tree builder over the FullDrafter
 * greedy spine (arch/common/rad_fulldrafter_tree.h).
 *
 * Holds the builder to its wave-1 contract: the greedy spine is preserved verbatim, siblings
 * fork only at the root, the node budget includes the root, invalid inputs are refused without
 * partial output, and the resulting tree verifies through the existing DARTree greedy accept
 * path identically to a linear tree when no alternates are requested.
 */

#include "rad_test.h"

#include "../arch/common/rad_fulldrafter_tree.h"

#include <cstdint>
#include <cstring>

using namespace radiance;
using r4d::FULLDRAFTER_SENTINEL_ID;

/* The drafter's full two-stage chain (12 = BLOCK_K * STAGES); token ids are arbitrary but
 * distinct so node identity and token identity never mask each other. */
static constexpr uint32_t kChain12[12] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};
static constexpr uint32_t kAnchor = 99;

static FulldrafterRootFork make_req(
    const uint32_t* chain, size_t chain_len,
    const uint32_t* ids, const float* scores, size_t n_cand,
    size_t alt_count, size_t node_budget)
{
    return FulldrafterRootFork{kAnchor, chain, chain_len, ids, scores, n_cand, alt_count, node_budget};
}

static bool trees_equal(const CandidateTree& a, const CandidateTree& b) {
    if (a.count != b.count) return false;
    for (size_t i = 0; i < a.count; ++i) {
        const TreeNode& x = a.nodes[i];
        const TreeNode& y = b.nodes[i];
        if (x.node_id != y.node_id || x.parent_id != y.parent_id ||
            x.token_id != y.token_id || x.depth != y.depth || x.score != y.score ||
            x.ancestor_bitmask != y.ancestor_bitmask || x.gdn_slot != y.gdn_slot)
            return false;
    }
    return true;
}

/* Brute-force ancestor by parent walk, independent of the bitmask the tree computes. */
static bool brute_is_ancestor(const CandidateTree& t, int32_t anc, int32_t node) {
    int32_t cur = node;
    while (cur >= 0) {
        if (cur == anc) return true;
        cur = t.nodes[cur].parent_id;
    }
    return false;
}

TEST(greedy_only_parity_with_linear) {
    CandidateTree built;
    FulldrafterRootFork r = make_req(kChain12, 12, nullptr, nullptr, 0, 0, 13);
    CHECK_EQ(fulldrafter_build_root_fork(r, &built), RAD_OK);
    CHECK_EQ(built.count, 13u);

    CandidateTree linear;
    linear.build_linear(kAnchor, kChain12, 12);
    CHECK(trees_equal(built, linear));

    /* A spine shorter than the full chain must match the linear tree of that length. */
    CandidateTree short_built;
    FulldrafterRootFork rs = make_req(kChain12, 3, nullptr, nullptr, 0, 0, 4);
    CHECK_EQ(fulldrafter_build_root_fork(rs, &short_built), RAD_OK);
    CandidateTree short_linear;
    short_linear.build_linear(kAnchor, kChain12, 3);
    CHECK(trees_equal(short_built, short_linear));
}

TEST(brute_force_ancestor_masks) {
    const uint32_t ids[6] = {200, 201, 202, 203, 204, 205};
    const float scores[6] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.5f};
    CandidateTree t;
    FulldrafterRootFork r = make_req(kChain12, 5, ids, scores, 6, 4, 10);
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);
    CHECK_EQ(t.count, 10u); /* 1 root + 5 spine + 4 alternates */

    for (int32_t a = 0; a < (int32_t)t.count; ++a) {
        for (int32_t b = 0; b < (int32_t)t.count; ++b) {
            const bool want = brute_is_ancestor(t, a, b);
            CHECK_EQ(t.is_ancestor(a, b), want);
            const bool bit = t.get_ancestor_bitmask(b).test(static_cast<size_t>(a));
            CHECK_EQ(bit, want);
        }
        /* The mask carries the node itself; nothing above the node is in it. */
        CHECK(t.get_ancestor_bitmask(a).test(static_cast<size_t>(a)));
    }
    /* Every alternate is a depth-1 leaf whose ancestry is exactly {root, self}. */
    for (size_t n = 6; n < t.count; ++n) {
        CHECK_EQ(t.nodes[n].parent_id, 0);
        CHECK_EQ(t.nodes[n].depth, 1);
        CHECK(t.get_ancestor_bitmask((int32_t)n) == ((1u << 0) | (1u << n)));
    }
    /* Out-of-range ids answer false instead of reading garbage. */
    CHECK(!t.is_ancestor(-1, 0));
    CHECK(!t.is_ancestor(0, (int32_t)t.count));
    CHECK(t.get_ancestor_bitmask((int32_t)t.count) == 0);
}

TEST(positions_and_depth_tensors) {
    const uint32_t ids[2] = {300, 301};
    const float scores[2] = {7.0f, 6.0f};
    CandidateTree t;
    FulldrafterRootFork r = make_req(kChain12, 12, ids, scores, 2, 2, 16);
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);
    CHECK_EQ(t.count, 15u);

    uint32_t toks[RAD_TREE_MAX_NODES] = {0};
    TreeMask anc[RAD_TREE_MAX_NODES] = {};
    uint32_t pos[RAD_TREE_MAX_NODES] = {0};
    t.populate_tensors(toks, anc, pos, 1000);
    for (size_t i = 0; i < t.count; ++i) {
        CHECK_EQ(toks[i], t.nodes[i].token_id);
        CHECK(anc[i] == t.nodes[i].ancestor_bitmask);
        /* Position is a RoPE coordinate derived from depth, never the physical row. */
        CHECK_EQ(pos[i], 1000u + (uint32_t)t.nodes[i].depth);
    }
    /* Tails beyond count stay zeroed: masked-off rows, not stale tokens. */
    for (size_t i = t.count; i < RAD_TREE_MAX_NODES; ++i) {
        CHECK_EQ(toks[i], 0u);
        CHECK(anc[i] == 0);
        CHECK_EQ(pos[i], 1000u);
    }
}

TEST(invalid_inputs_refused_without_partial_output) {
    const uint32_t ids[4] = {400, 401, 402, 403};
    const float scores[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    const float nan_scores[1] = {__builtin_nanf("")};
    const float inf_scores[1] = {__builtin_inff()};
    const float neg_inf_scores[1] = {-__builtin_inff()};

    struct Case { FulldrafterRootFork r; };
    const Case cases[] = {
        {make_req(nullptr, 12, ids, scores, 4, 0, 13)},      /* null chain */
        {make_req(kChain12, 0, ids, scores, 4, 0, 5)},       /* empty chain */
        {make_req(kChain12, 16, ids, scores, 4, 0, 17)},     /* chain longer than 15 */
        {make_req(kChain12, 12, ids, scores, 4, 0, 0)},      /* budget zero */
        {make_req(kChain12, 12, ids, scores, 4, 0, 17)},     /* budget above 16 */
        {make_req(kChain12, 12, ids, scores, 4, 0, 12)},     /* chain exceeds budget */
        {make_req(kChain12, 12, ids, scores, 4, 3, 15)},     /* alts exceed budget */
        {make_req(kChain12, 12, ids, scores, 17, 0, 13)},    /* n_cand above 16 */
        {make_req(kChain12, 12, nullptr, scores, 4, 1, 16)}, /* null cand ids */
        {make_req(kChain12, 12, ids, nullptr, 4, 1, 16)},    /* null cand scores */
        {make_req(kChain12, 12, nullptr, nullptr, 0, 1, 16)},/* alts without cands */
        {make_req(kChain12, 12, ids, scores, 2, 3, 16)},     /* not enough unique */
        {make_req(kChain12, 12, ids, nan_scores, 1, 1, 16)}, /* non-finite score: NaN */
        {make_req(kChain12, 12, ids, inf_scores, 1, 1, 16)}, /* non-finite score: +Inf */
        {make_req(kChain12, 12, ids, neg_inf_scores, 1, 1, 16)}, /* non-finite score: -Inf */
    };
    /* Each refusal names its error and leaves the caller's live tree untouched. */
    CandidateTree ref;
    ref.build_linear(kAnchor, kChain12, 12);
    for (const Case& c : cases) {
        CandidateTree t;
        t.build_linear(kAnchor, kChain12, 12);
        CHECK_EQ(fulldrafter_build_root_fork(c.r, &t), RAD_E_INVAL);
        CHECK(trees_equal(t, ref));
    }
    CHECK_EQ(fulldrafter_build_root_fork(cases[0].r, nullptr), RAD_E_INVAL);
}

TEST(sentinel_duplicate_tie_ordering) {
    /* 51 vs 52 tie at 7 -> lower id first; 50 vs 53 tie at 5 -> lower id first; the second 50
     * collapses into its better entry; the sentinel never becomes a node. */
    const uint32_t ids[6] = {50, 51, 50, FULLDRAFTER_SENTINEL_ID, 52, 53};
    const float scores[6] = {5.0f, 7.0f, 6.0f, 100.0f, 7.0f, 5.0f};
    CandidateTree t;
    FulldrafterRootFork r = make_req(kChain12, 4, ids, scores, 6, 3, 8);
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);
    CHECK_EQ(t.count, 8u); /* 1 + 4 spine + 3 alts */
    const uint32_t want_tok[3] = {51, 52, 50};
    const float want_score[3] = {7.0f, 7.0f, 6.0f};
    for (size_t a = 0; a < 3; ++a) {
        const TreeNode& n = t.nodes[5 + a];
        CHECK_EQ(n.token_id, want_tok[a]);
        CHECK_EQ(n.score, want_score[a]);
        CHECK_EQ(n.parent_id, 0);
        CHECK_EQ(n.depth, 1);
        CHECK(n.token_id != FULLDRAFTER_SENTINEL_ID);
    }

    /* A sentinel-only candidate list is fine when no alternates are requested. */
    const uint32_t sent[2] = {FULLDRAFTER_SENTINEL_ID, FULLDRAFTER_SENTINEL_ID};
    const float sent_scores[2] = {1.0f, 1.0f};
    CandidateTree t2;
    FulldrafterRootFork r2 = make_req(kChain12, 2, sent, sent_scores, 2, 0, 3);
    CHECK_EQ(fulldrafter_build_root_fork(r2, &t2), RAD_OK);
    CHECK_EQ(t2.count, 3u);

    /* But it cannot satisfy a request for alternates. */
    CandidateTree t3;
    FulldrafterRootFork r3 = make_req(kChain12, 2, sent, sent_scores, 2, 1, 4);
    CHECK_EQ(fulldrafter_build_root_fork(r3, &t3), RAD_E_INVAL);

    /* The first greedy token is deduped out of the root choices even at the top score. */
    const uint32_t ids4[3] = {kChain12[0], 60, 61};
    const float scores4[3] = {50.0f, 2.0f, 1.0f};
    CandidateTree t4;
    FulldrafterRootFork r4 = make_req(kChain12, 4, ids4, scores4, 3, 2, 7);
    CHECK_EQ(fulldrafter_build_root_fork(r4, &t4), RAD_OK);
    CHECK_EQ(t4.nodes[5].token_id, 60u);
    CHECK_EQ(t4.nodes[6].token_id, 61u);

    /* A candidate equal to a DEEPER spine token is allowed: different parent, different node. */
    const uint32_t ids5[1] = {kChain12[3]};
    const float scores5[1] = {1.0f};
    CandidateTree t5;
    FulldrafterRootFork r5 = make_req(kChain12, 4, ids5, scores5, 1, 1, 6);
    CHECK_EQ(fulldrafter_build_root_fork(r5, &t5), RAD_OK);
    CHECK_EQ(t5.nodes[5].token_id, kChain12[3]);
    CHECK_EQ(t5.nodes[5].parent_id, 0);
    CHECK_EQ(t5.nodes[4].parent_id, 3);
}

TEST(spine_preserved_root_only_forks) {
    const uint32_t ids[5] = {500, 501, 502, 503, 504};
    const float scores[5] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f};
    CandidateTree t;
    FulldrafterRootFork r = make_req(kChain12, 12, ids, scores, 5, 3, 16);
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);
    CHECK_EQ(t.count, 16u); /* root counts against the budget: 1 + 12 + 3 */

    /* The spine is verbatim, nodes 1..12. */
    for (size_t i = 0; i < 12; ++i) {
        CHECK_EQ(t.nodes[i + 1].token_id, kChain12[i]);
        CHECK_EQ(t.nodes[i + 1].parent_id, (int32_t)i);
        CHECK_EQ(t.nodes[i + 1].depth, (int32_t)i + 1);
    }
    /* Root. */
    CHECK_EQ(t.nodes[0].token_id, kAnchor);
    CHECK_EQ(t.nodes[0].parent_id, -1);
    CHECK_EQ(t.nodes[0].depth, 0);

    /* Children of the root are exactly the spine head plus the alternates. */
    size_t root_children = 0;
    for (size_t n = 1; n < t.count; ++n) {
        if (t.nodes[n].parent_id == 0) ++root_children;
        if (n > 12) {
            /* Stage 2 (nodes 8..12, chain[6..11]) stays on the greedy T6 path: no node past the
             * spine tail has a child, so an alternate can never grow a continuation. */
            for (size_t m = 1; m < t.count; ++m) {
                CHECK(t.nodes[m].parent_id != (int32_t)n);
            }
        }
    }
    CHECK_EQ(root_children, 4u); /* node 1 + 3 alternates */

    /* Budget below the full chain + alts is refused rather than trimmed. */
    CandidateTree t2;
    FulldrafterRootFork r2 = make_req(kChain12, 12, ids, scores, 5, 3, 15);
    CHECK_EQ(fulldrafter_build_root_fork(r2, &t2), RAD_E_INVAL);
}

TEST(sample_match_winner_and_bonus_paths) {
    const uint32_t ids[2] = {800, 801};
    const float scores[2] = {4.0f, 3.0f};
    CandidateTree t;
    FulldrafterRootFork r = make_req(kChain12, 12, ids, scores, 2, 2, 16);
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);
    /* Node map: 0=root(99) 1..12=100..111 spine 13=800 14=801 (alternates, parent root). */

    /* Path A: the teacher agrees with the greedy spine for three tokens, then diverges. */
    uint32_t preds[16];
    for (size_t i = 0; i < 16; ++i) preds[i] = 0;
    preds[0] = 100; /* after the anchor: the spine head */
    preds[1] = 101;
    preds[2] = 102;
    preds[3] = 999; /* teacher diverges after three accepts: bonus token */
    TreeVerificationResult a = DARTreeVerifier::verify_tree_greedy(t, preds, 16);
    CHECK_EQ(a.accepted_length, 3);
    CHECK(a.accepted_tokens[0] == 100 && a.accepted_tokens[1] == 101 && a.accepted_tokens[2] == 102);
    CHECK(a.has_bonus);
    CHECK_EQ(a.bonus_token, 999u);
    CHECK(a.best_path_node_ids[0] == 0 && a.best_path_node_ids[1] == 1 &&
          a.best_path_node_ids[2] == 2 && a.best_path_node_ids[3] == 3);

    /* Path B: the teacher's first pick is a root alternate. The alternate is a leaf, so the
     * next teacher prediction arrives as the bonus token. */
    preds[0] = 800;
    preds[13] = 777;
    TreeVerificationResult b = DARTreeVerifier::verify_tree_greedy(t, preds, 16);
    CHECK_EQ(b.accepted_length, 1);
    CHECK_EQ(b.accepted_tokens[0], 800u);
    CHECK(b.has_bonus);
    CHECK_EQ(b.bonus_token, 777u);
    CHECK_EQ(b.best_path_node_ids[1], 13);

    /* Path C: accepting through an alternate must not leak into the spine; the same tree with
     * the alternate rejected takes the spine. */
    preds[0] = 801;
    preds[14] = 100; /* a spine token is NOT a child of the alternate: bonus, not a walk */
    TreeVerificationResult c = DARTreeVerifier::verify_tree_greedy(t, preds, 16);
    CHECK_EQ(c.accepted_length, 1);
    CHECK_EQ(c.accepted_tokens[0], 801u);
    CHECK(c.has_bonus);
    CHECK_EQ(c.bonus_token, 100u);

    /* Path D: immediate divergence at root (zero accepted draft tokens).
     * Replay length is accepted children + 1 (root only). Bonus is the teacher's immediate pick. */
    preds[0] = 777; /* does not match spine head (100) or alternates (800, 801) */
    TreeVerificationResult d = DARTreeVerifier::verify_tree_greedy(t, preds, 16);
    CHECK_EQ(d.accepted_length, 0);
    CHECK_EQ(d.num_accepted(), 0u);
    CHECK_EQ(d.path_length(), 1u);
    CHECK_EQ(d.best_path_node_ids[0], 0);
    CHECK(d.has_bonus);
    CHECK_EQ(d.bonus_token, 777u);

    /* Linear parity through the whole accept path: with no alternates the built tree is not just
     * field-equal to build_linear's, it verifies identically. */
    CandidateTree t_lin;
    t_lin.build_linear(kAnchor, kChain12, 12);
    CandidateTree t_g;
    FulldrafterRootFork rg = make_req(kChain12, 12, ids, scores, 2, 0, 13);
    CHECK_EQ(fulldrafter_build_root_fork(rg, &t_g), RAD_OK);
    for (size_t i = 0; i < 16; ++i) preds[i] = kChain12[i % 12];
    TreeVerificationResult v_lin = DARTreeVerifier::verify_tree_greedy(t_lin, preds, 16);
    TreeVerificationResult v_g = DARTreeVerifier::verify_tree_greedy(t_g, preds, 16);
    CHECK_EQ(v_g.accepted_length, v_lin.accepted_length);
    CHECK_EQ(v_g.num_accepted(), v_lin.num_accepted());
    CHECK(v_g.has_bonus == v_lin.has_bonus);
    CHECK_EQ(v_g.bonus_token, v_lin.bonus_token);
    for (size_t i = 0; i <= v_g.num_accepted(); ++i) {
        CHECK_EQ(v_g.best_path_node_ids[i], v_lin.best_path_node_ids[i]);
    }
    for (size_t i = 0; i < v_g.num_accepted(); ++i) {
        CHECK_EQ(v_g.accepted_tokens[i], v_lin.accepted_tokens[i]);
    }
}

TEST(multi_sequence_and_reuse) {
    const uint32_t ids_a[2] = {600, 601};
    const float scores_a[2] = {5.0f, 4.0f};
    const uint32_t chain_b[6] = {900, 901, 902, 903, 904, 905};
    const uint32_t ids_b[3] = {700, 701, 702};
    const float scores_b[3] = {3.0f, 2.0f, 1.0f};

    CandidateTree ta, tb;
    FulldrafterRootFork ra = make_req(kChain12, 12, ids_a, scores_a, 2, 2, 16);
    FulldrafterRootFork rb = make_req(chain_b, 6, ids_b, scores_b, 3, 3, 10);
    CHECK_EQ(fulldrafter_build_root_fork(ra, &ta), RAD_OK);
    CHECK_EQ(fulldrafter_build_root_fork(rb, &tb), RAD_OK);
    CHECK_EQ(ta.count, 15u);
    CHECK_EQ(tb.count, 10u);
    CHECK_EQ(ta.nodes[13].token_id, 600u);
    CHECK_EQ(tb.nodes[7].token_id, 700u);

    /* Two sequences built back to back stay independent. */
    CHECK_EQ(ta.nodes[1].token_id, kChain12[0]);
    CHECK_EQ(tb.nodes[1].token_id, 900u);

    /* Reuse: rebuilding a slot's tree with another sequence's request reproduces a fresh build. */
    CandidateTree fresh_a;
    CHECK_EQ(fulldrafter_build_root_fork(ra, &fresh_a), RAD_OK);
    CHECK_EQ(fulldrafter_build_root_fork(ra, &tb), RAD_OK);
    CHECK(trees_equal(tb, fresh_a));

    /* Reset and rebuild deterministically: same request, same tree, every time. */
    tb.reset();
    CHECK_EQ(fulldrafter_build_root_fork(ra, &tb), RAD_OK);
    CHECK(trees_equal(tb, fresh_a));
    CandidateTree again;
    CHECK_EQ(fulldrafter_build_root_fork(ra, &again), RAD_OK);
    CHECK(trees_equal(again, fresh_a));
}

TEST(input_output_aliasing) {
    const uint32_t ids[3] = {650, 651, 652};
    const float scores[3] = {8.0f, 7.0f, 6.0f};
    CandidateTree t;
    FulldrafterRootFork r = make_req(kChain12, 12, ids, scores, 3, 2, 16);
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);

    /* Rebuilding into the same object reproduces the fresh build exactly (reset/reuse). */
    CHECK_EQ(fulldrafter_build_root_fork(r, &t), RAD_OK);
    CandidateTree fresh;
    CHECK_EQ(fulldrafter_build_root_fork(r, &fresh), RAD_OK);
    CHECK(trees_equal(t, fresh));

    /* Candidate ids overlapping the chain array: both are read before any write to *out, so a
     * caller publishing the drafter's chain and its top-16 out of one scratch buffer is safe.
     * (A CandidateTree cannot be aliased as an input: its nodes are structs, not a contiguous
     * token array, so no real caller can hand its storage back as greedy_chain.) */
    uint32_t shared[8];
    for (size_t i = 0; i < 6; ++i) shared[i] = kChain12[i]; /* chain = shared[0..5] */
    shared[6] = kChain12[0];                                /* spine head: must dedup out */
    shared[7] = 660;
    const float ov_scores[2] = {2.0f, 1.0f};
    CandidateTree t2;
    FulldrafterRootFork r2 = make_req(shared, 6, shared + 6, ov_scores, 2, 1, 8);
    CHECK_EQ(fulldrafter_build_root_fork(r2, &t2), RAD_OK);
    CHECK_EQ(t2.count, 8u);
    CHECK_EQ(t2.nodes[7].token_id, 660u); /* 660 beat nothing else; the spine head was skipped */
    CHECK_EQ(t2.nodes[7].parent_id, 0);

    /* Overlap at the chain's tail instead of past its end: same rule, offsets differ. */
    uint32_t shared2[7];
    for (size_t i = 0; i < 6; ++i) shared2[i] = kChain12[i];
    shared2[5] = 670; /* chain[5] and candidate[0] are the same slot */
    CandidateTree t3;
    FulldrafterRootFork r3 = make_req(shared2, 6, shared2 + 5, ov_scores, 1, 1, 8);
    CHECK_EQ(fulldrafter_build_root_fork(r3, &t3), RAD_OK);
    CHECK_EQ(t3.count, 8u);
    CHECK_EQ(t3.nodes[6].token_id, 670u); /* spine tail: chain[5] */
    CHECK_EQ(t3.nodes[6].parent_id, 5);
    CHECK_EQ(t3.nodes[6].depth, 6);
    CHECK_EQ(t3.nodes[7].token_id, 670u); /* the alternate: same token, different parent */
    CHECK_EQ(t3.nodes[7].parent_id, 0);
    CHECK_EQ(t3.nodes[7].depth, 1);

    /* Complete pointer aliasing between greedy_chain and root_cand_ids: */
    const float same_scores[6] = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f};
    CandidateTree t4;
    FulldrafterRootFork r4 = make_req(kChain12, 6, kChain12, same_scores, 6, 2, 9);
    CHECK_EQ(fulldrafter_build_root_fork(r4, &t4), RAD_OK);
    CHECK_EQ(t4.count, 9u); /* 1 root + 6 spine + 2 alts */
    /* kChain12[0] skipped because spine head owns it. Next highest scores are kChain12[1] and kChain12[2]. */
    CHECK_EQ(t4.nodes[7].token_id, kChain12[1]);
    CHECK_EQ(t4.nodes[7].parent_id, 0);
    CHECK_EQ(t4.nodes[8].token_id, kChain12[2]);
    CHECK_EQ(t4.nodes[8].parent_id, 0);
}

RAD_TEST_MAIN()
