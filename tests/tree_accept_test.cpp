/* tree_accept_test.cpp -- the ported tree geometry, masks and greedy winner walk
 * (arch/common/rad_tree_candidate.h) and the sampler's tree acceptance (accept_tree_greedy).
 *
 * What is checked, and why each one would fail loudly on a bad port:
 *   - the fixed topologies' ancestor masks, as the archived verifier asserted them;
 *   - the packed MTP8/64 tree: spine per round, alternates as siblings, dedup, capacity, negative
 *     terminators, and the round/capacity pairs the op only accepts;
 *   - masks_consistent()/fits(): a mutated mask or an oversized tree is refused, so no row is
 *     conditioned on a token that is not on its path;
 *   - the winner walk: on a chain it agrees with accept_greedy; on random trees its answer
 *     satisfies the acceptance contract (every step matches the parent's pick, no lower-id
 *     matching sibling exists, the stop node has no matching child, the bonus is that node's
 *     pick) and agrees with a second walk over a prebuilt child table; the lowest id wins a tie
 *     between siblings carrying the same token;
 *   - EOS: accept_cut_at_eog cuts a stop token inside the tree exactly as it cuts a chain.
 */
#include "rad_test.h"

#include "../arch/common/rad_tree_candidate.h"
#include "sample/accept.h"

#include <vector>

using namespace rad;

namespace {

bool is_eog_fixture(int32_t t) { return t == 2; }

/* Deterministic generator: the same trees on every host, so a failure reproduces. */
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed) {}
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    int32_t below(int32_t n) {
        return n <= 0 ? 0 : static_cast<int32_t>(next() % static_cast<uint32_t>(n));
    }
};

/* A second walk, written the other way round: build the child table once, then step through it.
 * Same contract, different indices -- a mismatch means one of the two is indexing wrong. */
int walk_from_child_table(const CandidateTree& t, const int32_t* pick, int32_t* path) {
    std::vector<std::vector<int32_t>> kids(static_cast<size_t>(t.size()));
    for (int32_t i = 1; i < t.size(); ++i) kids[static_cast<size_t>(t[i].parent)].push_back(i);
    int32_t cur = 0;
    int32_t acc = 0;
    path[0] = 0;
    for (;;) {
        int32_t hit = -1;
        for (int32_t c : kids[static_cast<size_t>(cur)]) {
            if (t[c].token == static_cast<uint32_t>(pick[cur])) { hit = c; break; }
        }
        if (hit < 0) return acc;
        path[++acc] = hit;
        cur = hit;
    }
}

}  // namespace

TEST(the_fixed_topologies_carry_the_ancestor_masks_the_verifier_asserted) {
    CandidateTree t;
    uint32_t cand[15];
    for (int i = 0; i < 15; ++i) cand[i] = 1000u + static_cast<uint32_t>(i);

    CHECK(t.build_top2(500u, cand, 6, false));
    CHECK_EQ(t.size(), 7);
    CHECK_EQ(tree_mask16(t[0].ancestors), 0x0001);
    CHECK_EQ(tree_mask16(t[1].ancestors), 0x0003);   /* child 1 of the root */
    CHECK_EQ(tree_mask16(t[2].ancestors), 0x0005);   /* child 2 of the root */
    CHECK_EQ(tree_mask16(t[3].ancestors), 0x000B);   /* child of 1 */
    CHECK_EQ(tree_mask16(t[4].ancestors), 0x0013);   /* child of 1 */
    CHECK_EQ(tree_mask16(t[5].ancestors), 0x0025);   /* child of 2 */
    CHECK_EQ(tree_mask16(t[6].ancestors), 0x0045);   /* child of 2 */
    CHECK_EQ(t[3].depth, 2);
    CHECK_EQ(t.position(3, 128), 130);
    CHECK(t.masks_consistent());
    CHECK(t.fits(4096, 64));

    /* Mode B branches at depths 1 and 2 only, then continues as a chain. */
    CHECK(t.build_top2(500u, cand, 9, true));
    CHECK_EQ(t.size(), 10);
    CHECK_EQ(t[7].parent, 3);
    CHECK_EQ(t[8].parent, 7);
    CHECK_EQ(t[8].depth, 4);
    CHECK(t.masks_consistent());

    /* Top-4: four children of the root, then each one's children. */
    CHECK(t.build_top4(500u, cand, 8));
    for (int i = 1; i <= 4; ++i) CHECK_EQ(t[i].parent, 0);
    CHECK_EQ(t[5].parent, 1);
    CHECK(t.masks_consistent());
}

TEST(the_packed_mtp_tree_is_a_spine_with_sibling_alternates) {
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 8; ++k) history[r][k] = -1;

    /* Four rounds of top-k, the shape the archived accumulator packed. */
    for (int r = 0; r < 4; ++r) {
        for (int k = 0; k < 8; ++k) history[r][k] = 100 + r * 8 + k;
    }

    CandidateTree t;
    CHECK(t.build_mtp_fixed(history, 8, kTreeMaxNodes, 7, 4096));
    /* Four spine rows (one per live round), then the 7 alternates of each round. */
    CHECK_EQ(t.size(), 1 + 4 + 4 * 7);
    CHECK_EQ(t[1].parent, 0);
    CHECK_EQ(t[2].parent, 1);
    CHECK_EQ(t[3].parent, 2);
    CHECK_EQ(t[4].parent, 3);
    CHECK_EQ(t[5].parent, 0);            /* round 0's alternates hang off the root */
    CHECK_EQ(t[4].depth, 4);
    CHECK(t.fits(4096, 64));
    CHECK(t.masks_consistent());

    /* Every alternate of round r is a sibling of spine row r, so it attends the spine only up to
     * its parent -- never a later spine row, never another alternate. */
    CHECK(t[5].ancestors[0]);
    CHECK(!t[5].ancestors[1]);
    CHECK(!t[5].ancestors[6]);

    /* A duplicate alternate (same token as the spine row above it, or as an earlier alternate of
     * the same round) is not added twice: two rows with one token and one ancestor set can never
     * be told apart by acceptance. */
    history[1][1] = history[1][0];
    CHECK(t.build_mtp_fixed(history, 8, kTreeMaxNodes, 7, 4096));
    CHECK_EQ(t.size(), 1 + 4 + 3 * 7 + 6);

    /* The budget truncates; it never overflows. */
    CHECK(t.build_mtp_fixed(history, 8, 12, 7, 4096));
    CHECK_EQ(t.size(), 12);
    CHECK(t.fits(4096, 64));

    /* A negative spine entry ends the tree at that round. */
    history[2][0] = -1;
    CHECK(t.build_mtp_fixed(history, 8, kTreeMaxNodes, 7, 4096));
    CHECK(t[1].token == 100 && t[2].token == 108);
    for (int32_t i = 1; i < t.size(); ++i) CHECK(t[i].token != 116);   /* round 2 never landed */

    /* The round/capacity pairs the op accepts: 3 rounds keep the legacy <=16 width, 8 rounds the
     * MTP8/64 width, and neither takes a capacity it cannot fill. */
    CHECK(!t.build_mtp_fixed(history, 4, 20, 7, 4096));
    CHECK(!t.build_mtp_fixed(history, 3, 17, 7, 4096));
    CHECK(t.build_mtp_fixed(history, 3, 16, 7, 4096));
    CHECK(!t.build_mtp_fixed(history, 8, 66, 7, 4096));
    CHECK(!t.build_mtp_fixed(history, 8, 1, 7, 4096));

    /* A token the sampler could never index empties the tree instead of building the row. */
    history[0][1] = 4096;
    CHECK(!t.build_mtp_fixed(history, 8, kTreeMaxNodes, 7, 4096));
    CHECK_EQ(t.size(), 0);
    /* A root the vocabulary cannot hold is refused the same way. */
    CHECK(!t.build_mtp_fixed(history, 8, kTreeMaxNodes, 4096, 4096));
    CHECK_EQ(t.size(), 0);
}

TEST(a_wrong_ancestor_mask_is_refused_not_scored) {
    CandidateTree t;
    uint32_t cand[15];
    for (int i = 0; i < 15; ++i) cand[i] = 2000u + static_cast<uint32_t>(i);
    CHECK(t.build_top2(500u, cand, 6, false));

    /* A row that attends a sibling's token: the failure the mask checks exist to stop. */
    CandidateTree bad = t;
    bad[3].ancestors.set(5);
    CHECK(!bad.masks_consistent());
    CHECK(!bad.fits(4096, 64));

    /* A mask missing its own bit is just as wrong. */
    bad = t;
    bad[4].ancestors.reset(4);
    CHECK(!bad.masks_consistent());

    /* Depth is derived from the parent, so a caller cannot hand in a depth that disagrees. */
    CHECK(t[6].depth == t[t[6].parent].depth + 1);

    /* The root is the only depth-0 row; a mutated root depth is refused rather than shifting
     * every position on the tree. */
    CandidateTree bad_root;
    CHECK(bad_root.build_linear(5, nullptr, 0));
    CHECK(bad_root.masks_consistent());
    bad_root[0].depth = -5;
    CHECK(!bad_root.masks_consistent());
    CHECK(!bad_root.fits(0, 64));

    /* The sampler's entry point checks the tree it was handed, so a bad mask cannot reach the
     * passes that read the accepted path. */
    int32_t out[8], path[8], acc = -1;
    std::vector<int32_t> pick(static_cast<size_t>(bad.size()), 1);
    CHECK_EQ(accept_tree_greedy(bad, pick.data(), bad.size(), out, &acc, path), RAD_E_INVAL);
}

TEST(the_legacy_uint16_mask_is_refused_above_sixteen_rows) {
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 8; ++k) history[r][k] = 100 + r * 8 + k;

    CandidateTree t;
    CHECK(t.build_mtp_fixed(history, 8, kTreeMaxNodes, 7, 4096));
    CHECK_EQ(t.size(), 65);

    /* Rows below 16 keep their exact form. */
    CHECK_EQ(tree_mask16(t[0].ancestors), 0x0001);
    CHECK_EQ(tree_mask16(t[9].ancestors), 0x0201);   /* root + itself */
    /* A row whose ancestor set reaches past bit 15 has no uint16 form: 0, never a truncated set
     * that would condition the row on the wrong tokens. */
    CHECK_EQ(tree_mask16(t[40].ancestors), 0x0000);
    CHECK_EQ(tree_mask16(t[64].ancestors), 0x0000);
}

TEST(the_geometry_matches_the_archived_reference_at_5c3bb1d) {
    /* Goldens dumped from the archived source
     * (candidates/fulldrafter-abi16-review/source @5c3bb1d): arch/common/rad_tree_candidate.h
     * builds the topologies, mine/ref_mtp_tree_accum.cpp packs the MTP tree. Values are literals
     * on purpose -- an expected value computed by the code under test proves nothing. */
    CandidateTree t;
    uint32_t cand[15];
    for (int i = 0; i < 15; ++i) cand[i] = 2000u + static_cast<uint32_t>(i);
    CHECK(t.build_top4(500u, cand, 8));
    const uint16_t top4[9] = {0x0001, 0x0003, 0x0005, 0x0009, 0x0011,
                              0x0023, 0x0043, 0x0083, 0x0103};
    CHECK_EQ(t.size(), 9);
    for (int32_t i = 0; i < t.size(); ++i) CHECK_EQ(tree_mask16(t[i].ancestors), top4[i]);

    /* The packed MTP tree the archived accumulator produced for three rounds of eight unique
     * candidates at the legacy capacity: root, three spine rows, then twelve alternates. */
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 8; ++k) history[r][k] = -1;
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 8; ++k) history[r][k] = 100 + r * 8 + k;

    const int32_t tokens3[16] = {7, 100, 108, 116, 101, 102, 103, 104,
                                 105, 106, 107, 109, 110, 111, 112, 113};
    const int32_t parents3[16] = {-1, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    const uint16_t masks3[16] = {0x0001, 0x0003, 0x0007, 0x000F, 0x0011, 0x0021, 0x0041, 0x0081,
                                  0x0101, 0x0201, 0x0401, 0x0803, 0x1003, 0x2003, 0x4003, 0x8003};
    CHECK(t.build_mtp_fixed(history, 3, 16, 7, 4096));
    CHECK_EQ(t.size(), 16);
    for (int32_t i = 0; i < 16; ++i) {
        CHECK_EQ(static_cast<int32_t>(t[i].token), tokens3[i]);
        CHECK_EQ(t[i].parent, parents3[i]);
        CHECK_EQ(tree_mask16(t[i].ancestors), masks3[i]);
    }

    /* The full MTP8/64 shape: eight spine rows, then each round's seven alternates. */
    for (int r = 0; r < 8; ++r)
        for (int k = 0; k < 8; ++k) history[r][k] = 100 + r * 8 + k;
    CHECK(t.build_mtp_fixed(history, 8, kTreeMaxNodes, 7, 4096));
    CHECK_EQ(t.size(), 65);
    const int32_t parents8[12] = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0};
    for (int i = 0; i < 12; ++i) CHECK_EQ(t[i].parent, parents8[i]);
    /* Masks of root, the first alternate, and one alternate per round up to the last row. */
    const int32_t probe[9] = {0, 9, 17, 25, 33, 41, 49, 57, 64};
    const uint64_t probe_low[9] = {0x1ull, 0x201ull, 0x20003ull, 0x2000007ull, 0x20000000Full,
                                   0x2000000001Full, 0x200000000003Full, 0x20000000000007Full,
                                   0x00000000000000FFull};
    const bool probe_high[9] = {false, false, false, false, false, false, false, false, true};
    for (int i = 0; i < 9; ++i) {
        uint64_t low = 0;
        for (size_t b = 0; b < 64; ++b)
            if (t[probe[i]].ancestors.test(b)) low |= 1ull << b;
        CHECK_EQ(low, probe_low[i]);
        CHECK_EQ(t[probe[i]].ancestors.test(64), probe_high[i]);
    }
}

TEST(a_chain_tree_accepts_exactly_what_accept_greedy_accepts) {
    int32_t draft[8] = {11, 12, 13, 14, 15, 16, 17, 18};
    const uint32_t root = 10;

    for (int mismatch = 0; mismatch <= 8; ++mismatch) {
        int32_t target[9];
        for (int i = 0; i < 8; ++i) target[i] = draft[i];
        target[8] = 99;
        if (mismatch < 8) target[mismatch] = 77;

        CandidateTree t;
        CHECK(t.build_linear(root, reinterpret_cast<const uint32_t*>(draft), 8));
        CHECK_EQ(t.size(), 9);

        /* Row j's pick is the target's token at position j, the root's row included. */
        int32_t out_chain[9], out_tree[9], path[9];
        int32_t acc_chain = -1, acc_tree = -1;
        CHECK_OK(accept_greedy(draft, 8, target, out_chain, &acc_chain));
        CHECK_OK(accept_tree_greedy(t, target, 9, out_tree, &acc_tree, path));
        CHECK_EQ(acc_tree, acc_chain);
        for (int32_t j = 0; j <= acc_chain; ++j) CHECK_EQ(out_tree[j], out_chain[j]);
        CHECK_EQ(path[0], 0);
        for (int32_t j = 0; j <= acc_tree; ++j) CHECK_EQ(path[j], j);
    }
}

TEST(the_winner_walk_satisfies_the_acceptance_contract_on_random_trees) {
    Lcg rng(0x5eed1234u);
    for (int iter = 0; iter < 60; ++iter) {
        CandidateTree t;
        const int32_t n = 2 + rng.below(18);
        /* Small token alphabet on purpose: sibling ties are the interesting case. */
        const int32_t alphabet = 2 + rng.below(4);
        t.add(-1, static_cast<uint32_t>(rng.below(alphabet)));
        for (int32_t i = 1; i < n; ++i) {
            const int32_t parent = rng.below(i);
            t.add(parent, static_cast<uint32_t>(rng.below(alphabet)));
        }
        CHECK(t.masks_consistent());

        std::vector<int32_t> pick(static_cast<size_t>(n));
        for (int32_t i = 0; i < n; ++i) pick[static_cast<size_t>(i)] = rng.below(alphabet);

        TreeWinner w;
        CHECK_OK(verify_tree_greedy(t, pick.data(), n, &w));

        /* Every accepted step is a child of the previous node carrying that node's pick. */
        CHECK_EQ(w.path[0], 0);
        for (int32_t j = 0; j < w.accepted; ++j) {
            const int32_t child = w.path[j + 1], parent = w.path[j];
            CHECK_EQ(t[child].parent, parent);
            CHECK_EQ(static_cast<int32_t>(t[child].token), pick[static_cast<size_t>(parent)]);
            CHECK_EQ(static_cast<int32_t>(w.tokens[j]), pick[static_cast<size_t>(parent)]);
        }
        /* No lower-id matching sibling was skipped. */
        for (int32_t j = 0; j < w.accepted; ++j) {
            const int32_t parent = w.path[j], child = w.path[j + 1];
            for (int32_t i = parent + 1; i < child; ++i) {
                if (t[i].parent != parent) continue;
                CHECK(t[i].token != static_cast<uint32_t>(pick[static_cast<size_t>(parent)]));
            }
        }
        /* The walk stopped because no child of the last node matches its pick, and the bonus is
         * that pick. */
        const int32_t last = w.path[w.accepted];
        for (int32_t i = last + 1; i < n; ++i) {
            if (t[i].parent != last) continue;
            CHECK(t[i].token != static_cast<uint32_t>(pick[static_cast<size_t>(last)]));
        }
        CHECK(w.has_bonus);
        CHECK_EQ(static_cast<int32_t>(w.bonus), pick[static_cast<size_t>(last)]);

        /* The second walk over a child table reaches the same path. */
        std::vector<int32_t> path2(static_cast<size_t>(n), -1);
        const int32_t acc2 = walk_from_child_table(t, pick.data(), path2.data());
        CHECK_EQ(acc2, w.accepted);
        for (int32_t j = 0; j <= w.accepted; ++j)
            CHECK_EQ(path2[static_cast<size_t>(j)], w.path[j]);
    }
}

TEST(the_lowest_id_wins_a_tie_between_siblings_with_the_same_token) {
    CandidateTree t;
    CHECK(t.add(-1, 5));      /* 0: root */
    CHECK(t.add(0, 9));       /* 1: matches the root's pick, and has no children */
    CHECK(t.add(0, 9));       /* 2: same token, and a deep chain behind it */
    CHECK(t.add(2, 9));       /* 3 */
    CHECK(t.add(3, 9));       /* 4 */

    const int32_t pick[5] = {9, 9, 9, 9, 9};
    TreeWinner w;
    CHECK_OK(verify_tree_greedy(t, pick, 5, &w));
    /* Node 1 wins the tie even though node 2's subtree is deeper: a sibling gets no second draw,
     * and the row order decides, so the winner does not depend on which branch is longer. */
    CHECK_EQ(w.accepted, 1);
    CHECK_EQ(w.path[1], 1);
    CHECK_EQ(static_cast<int32_t>(w.bonus), 9);

    int32_t out[6], path[6], acc = -1;
    CHECK_OK(accept_tree_greedy(t, pick, 5, out, &acc, path));
    CHECK_EQ(acc, 1);
    CHECK_EQ(out[0], 9);
    CHECK_EQ(out[1], 9);
    CHECK_EQ(path[1], 1);
}

TEST(an_immediate_mismatch_emits_the_bonus_alone) {
    CandidateTree t;
    CHECK(t.build_top2(500u, nullptr, 6, false));
    std::vector<int32_t> pick(static_cast<size_t>(t.size()), 0);   /* no child carries token 0 */

    int32_t out[8], path[8], acc = -1;
    CHECK_OK(accept_tree_greedy(t, pick.data(), t.size(), out, &acc, path));
    CHECK_EQ(acc, 0);
    CHECK_EQ(out[0], 0);      /* the target's own token at the root row */
    CHECK_EQ(path[0], 0);
}

TEST(a_stop_token_inside_the_tree_cuts_the_step_like_it_cuts_a_chain) {
    int32_t draft[5] = {11, 2, 13, 14, 15};   /* token 2 is the fixture's end-of-generation */
    const int32_t target[6] = {11, 2, 13, 14, 15, 99};   /* the chain agrees all the way */

    CandidateTree t;
    CHECK(t.build_linear(10, reinterpret_cast<const uint32_t*>(draft), 5));

    int32_t out[6], path[6], acc = -1;
    CHECK_OK(accept_tree_greedy(t, target, 6, out, &acc, path));
    CHECK_EQ(acc, 5);

    int32_t out_chain[6], acc_chain = -1;
    CHECK_OK(accept_greedy(draft, 5, target, out_chain, &acc_chain));
    CHECK_EQ(accept_cut_at_eog(out, acc, is_eog_fixture),
             accept_cut_at_eog(out_chain, acc_chain, is_eog_fixture));
    CHECK_EQ(accept_cut_at_eog(out, acc, is_eog_fixture), 1);
}

TEST(a_malformed_or_short_verify_is_refused) {
    CandidateTree t;
    CHECK(t.build_top2(500u, nullptr, 6, false));
    int32_t out[16], path[16], acc = -1;
    std::vector<int32_t> pick(static_cast<size_t>(t.size()), 1);

    CandidateTree empty;
    CHECK_EQ(verify_tree_greedy(empty, pick.data(), t.size(), nullptr), RAD_E_INVAL);
    CHECK_EQ(accept_tree_greedy(empty, pick.data(), t.size(), out, &acc, path), RAD_E_INVAL);

    TreeWinner w;
    /* One pick short: the walk may end on that row and needs its token. */
    CHECK_EQ(verify_tree_greedy(t, pick.data(), t.size() - 1, &w), RAD_E_INVAL);
    CHECK_EQ(accept_tree_greedy(t, nullptr, t.size(), out, &acc, path), RAD_E_INVAL);
    CHECK_EQ(accept_tree_greedy(t, pick.data(), t.size(), nullptr, &acc, path), RAD_E_INVAL);

    pick[0] = -1;   /* a row with no pick has no token to match */
    CHECK_EQ(verify_tree_greedy(t, pick.data(), t.size(), &w), RAD_E_INVAL);
}

TEST(fits_refuses_a_tree_the_step_has_no_room_for) {
    CandidateTree t;
    CHECK(t.build_top2(500u, nullptr, 9, true));
    CHECK(t.fits(1 << 20, 64));
    CHECK(!t.fits(1024, 64));         /* the built token ids are above this vocabulary */
    CHECK(!t.fits(1 << 20, 3));       /* depth 4 does not fit in 3 rows */
    CHECK(t.fits(0, 64));             /* vocabulary 0 skips that check, the rest still applies */
}

RAD_TEST_MAIN()
