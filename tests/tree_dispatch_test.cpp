/* tree_dispatch_test.cpp -- the host's tree-dispatch contract (core/sched/tree_dispatch.h): the
 * switch and temperature gate, the mask-width and capacity limits, the layout it stages for a
 * batch, and the commit it produces.
 *
 * What is checked, and why each one would fail loudly on a wrong dispatch:
 *   - RADIANCE_MTP_COUPLED off (coupled == false) and a temperature above 0 are refused before a
 *     single field of the admitted step is written, so a refused dispatch leaves no tree behind;
 *   - a chain tree commits token for token what the serial path accepts, end-of-generation cut
 *     included -- the tree may not change a greedy run's output;
 *   - the fixed MTP8 profile is admitted with the parents, positions and masks a batch stages, and
 *     a budget that would cut it short is refused instead of verifying a different tree;
 *   - a tree wider than the uint16 mask form is refused when the consumer is 16-bit, and admitted
 *     only when the caller stages a full-width mask;
 *   - a mutated ancestor mask, a depth past the token span and a token past the vocabulary are
 *     refused, so no row is ever conditioned on a token that is not on its path;
 *   - the commit names the winner path, counts the discarded branches and cuts at a stop token;
 *   - a commit the card's acceptance does not reproduce, a short pick buffer and a tampered staged
 *     layout are refused rather than emitted.
 */
#include "rad_test.h"

#include "sched/tree_dispatch.h"

#include <cstring>
#include <string>

using namespace rad;

namespace {

static constexpr int32_t kVocab = 4096;

bool is_eog(int32_t t) { return t == 2; }

/* A step carrying a tree of its own that no call will admit, and sentinels in every field it could
 * write, so a refusal that staged something shows up as a changed field rather than as a stale
 * zero that also looks untouched. */
TreeStep sentinel_step() {
    TreeStep s;
    for (int32_t j = 0; j < kTreeMaxNodes; ++j) {
        s.parents[j]   = -7;
        s.positions[j] = -7;
        s.masks16[j]   = 0xaaaa;
    }
    s.n_nodes     = -7;
    s.root_pos    = -7;
    s.has_mask16  = true;
    s.tree.add(-1, 1234);
    return s;
}

/* The tree itself: a refusal must not rewrite a node either. A token, parent, depth or ancestor
 * set the tree did not carry before is a half-staged tree that node_count() alone cannot show. */
bool tree_untouched(const CandidateTree& now, const CandidateTree& before) {
    if (now.node_count() != before.node_count()) return false;
    for (int32_t j = 0; j < before.node_count(); ++j)
        if (now[j].token != before[j].token || now[j].parent != before[j].parent ||
            now[j].depth != before[j].depth || now[j].ancestors != before[j].ancestors)
            return false;
    return true;
}

/* Every staged field the admit/commit pair writes: a refusal must leave all of them alone. */
bool staged_untouched(const TreeStep& now, const TreeStep& before) {
    if (now.n_nodes != before.n_nodes || now.root_pos != before.root_pos ||
        now.has_mask16 != before.has_mask16 || !tree_untouched(now.tree, before.tree))
        return false;
    for (int32_t j = 0; j < kTreeMaxNodes; ++j)
        if (now.parents[j] != before.parents[j] || now.positions[j] != before.positions[j] ||
            now.masks16[j] != before.masks16[j])
            return false;
    return true;
}

/* A chain of `n_draft` drafts past the root, ids 0..n_draft, and a pick vector that accepts all of
 * them. */
void chain_picks(const CandidateTree& t, int32_t* pick) {
    for (int32_t j = 0; j < t.node_count(); ++j) pick[j] = static_cast<int32_t>(t[j].token);
    pick[t.node_count() - 1] = 777;   /* the bonus at the last node */
}

}  // namespace

TEST(the_switch_and_the_temperature_contract_refuse_before_anything_is_staged) {
    CandidateTree t;
    uint32_t draft[4] = {10, 11, 12, 13};
    CHECK(t.build_linear(9, draft, 4));

    const TreeStep before = sentinel_step();
    TreeStep step = before;
    std::string why;

    /* The switch is off: refuse, and leave every staged field alone so no tree is allocated. */
    CHECK_EQ(tree_dispatch_admit(t, 100, kVocab, 512, /*temp0=*/true, /*coupled=*/false, 0,
                                 &step, &why), RAD_E_STATE);
    CHECK(why.find("RADIANCE_MTP_COUPLED") != std::string::npos);
    CHECK(staged_untouched(step, before));

    /* The switch is on but the sampler is not greedy: the tree cannot reproduce it. */
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, 100, kVocab, 512, /*temp0=*/false, /*coupled=*/true, 0,
                                 &step, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("temperature 0") != std::string::npos);
    CHECK(staged_untouched(step, before));

    /* The fixed profile takes the same gate, before it builds anything. */
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) history[r][c] = (c < 2) ? 100 + 10 * r + c : -1;
    why.clear();
    CHECK_EQ(tree_dispatch_fixed(history, 8, 65, 9, kVocab, 100, 512, false, true, 0, &step, &why),
             RAD_E_UNSUPPORTED);
    CHECK(staged_untouched(step, before));
    why.clear();
    CHECK_EQ(tree_dispatch_fixed(history, 8, 65, 9, kVocab, 100, 512, true, false, 0, &step, &why),
             RAD_E_STATE);
    CHECK(staged_untouched(step, before));

    /* An admitted step replaces all of it -- a sentinel left in place would be a half-staged tree. */
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, 100, kVocab, 512, true, true, 0, &step, &why), RAD_OK);
    CHECK_EQ(step.n_nodes, t.node_count());
    for (int32_t j = 0; j < step.n_nodes; ++j) {
        CHECK_EQ(step.parents[j], step.tree[j].parent);
        CHECK_EQ(step.positions[j], 100 + step.tree[j].depth);
    }
    CHECK(!staged_untouched(step, before));
}

TEST(a_chain_tree_commits_token_for_token_what_the_serial_path_accepts) {
    uint32_t draft[6] = {11, 12, 13, 14, 15, 16};
    CandidateTree t;
    CHECK(t.build_linear(10, draft, 6));

    TreeStep step;
    std::string why;
    CHECK_EQ(tree_dispatch_admit(t, 40, kVocab, 512, true, true, 0, &step, &why), RAD_OK);
    CHECK_EQ(step.n_nodes, 7);
    for (int32_t j = 0; j <= 6; ++j) {
        CHECK_EQ(step.parents[j], j - 1);
        CHECK_EQ(step.positions[j], 40 + j);
    }

    /* Three drafts agree, the fourth is a different token: the serial accept of the same six
     * drafts against the same targets stops at the same place. */
    int32_t target[7] = {11, 12, 13, 99, 15, 16, 0};
    int32_t pick[7];
    for (int32_t j = 0; j < 7; ++j) pick[j] = target[j];

    int32_t serial[8] = {};
    int32_t serial_acc = 0;
    CHECK_EQ(accept_greedy(reinterpret_cast<const int32_t*>(draft), 6, pick, serial, &serial_acc),
             RAD_OK);

    TreeCommit commit;
    CHECK_EQ(tree_dispatch_commit(step, pick, 7, is_eog, &commit, &why), RAD_OK);
    CHECK_EQ(commit.n_accepted, serial_acc);
    /* serial[0..acc] is the accepted prefix plus the verified token; the tree's emission is the
     * same sequence because a chain has exactly one child per node. */
    for (int32_t j = 0; j <= commit.n_accepted; ++j) CHECK_EQ(commit.tokens[j], serial[j]);
    CHECK_EQ(commit.n_discarded, 6 - commit.n_accepted);

    /* A stop token at the first draft ends the step with nothing accepted; one further in cuts
     * the accepted prefix where the serial cut does. The chain test would not otherwise reach an
     * end-of-generation cut at all -- its tokens are 11..16 and is_eog stops on 2. */
    auto stop_first = [](int32_t tok) { return tok == 11; };
    TreeCommit cut0;
    CHECK_EQ(tree_dispatch_commit(step, pick, 7, stop_first, &cut0, &why), RAD_OK);
    CHECK_EQ(cut0.n_accepted, 0);
    CHECK_EQ((int32_t)cut0.bonus, 11);
    CHECK_EQ(cut0.n_discarded, 6);

    auto stop_second = [](int32_t tok) { return tok == 12; };   /* the first accepted draft */
    TreeCommit cut1;
    CHECK_EQ(tree_dispatch_commit(step, pick, 7, stop_second, &cut1, &why), RAD_OK);
    CHECK_EQ(cut1.n_accepted, 1);
    CHECK_EQ((int32_t)cut1.bonus, 12);
    CHECK_EQ(cut1.n_discarded, 6 - 1);

    /* The card's acceptance, staged by advance_impl.h, must land on the same pair. */
    CHECK_EQ(tree_dispatch_confirm(commit, commit.n_accepted, (int32_t)commit.bonus, &why), RAD_OK);
    CHECK_EQ(tree_dispatch_confirm(commit, commit.n_accepted + 1, (int32_t)commit.bonus, &why),
             RAD_E_STATE);
    CHECK_EQ(tree_dispatch_confirm(commit, commit.n_accepted, (int32_t)commit.bonus + 1, &why),
             RAD_E_STATE);
}

TEST(the_fixed_mtp_profile_stages_its_parents_positions_and_masks) {
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) history[r][c] = (c < 8 && c < 2 + r) ? 200 + 10 * r + c : -1;

    TreeStep step;
    std::string why;
    CHECK_EQ(tree_dispatch_fixed(history, 8, kTreeMaxNodes, 5, kVocab, 64, 512, true, true, 0,
                                 &step, &why), RAD_OK);
    CHECK_EQ(step.n_nodes, step.tree.node_count());
    CHECK(step.n_nodes > kTreeLegacyMaskNodes);
    CHECK(!step.has_mask16);   /* node 16 and up have no exact uint16 ancestor set */
    CHECK_EQ(step.root_pos, 64);
    for (int32_t j = 0; j < step.n_nodes; ++j) {
        CHECK_EQ(step.parents[j], step.tree[j].parent);
        CHECK_EQ(step.positions[j], 64 + step.tree[j].depth);
        if (j < kTreeLegacyMaskNodes) CHECK(step.masks16[j] != 0);
        else CHECK_EQ(step.masks16[j], 0);   /* no exact uint16 form past 16 rows */
    }
    /* The spine is a chain, so the deepest node's position is the root plus eight rounds. */
    int32_t deepest = 0;
    for (int32_t j = 0; j < step.n_nodes; ++j) deepest = std::max(deepest, step.tree[j].depth);
    CHECK_EQ(deepest, 8);
    CHECK_EQ(step.positions[step.n_nodes - 1] >= step.root_pos + 1, true);
}

TEST(a_budget_that_cuts_the_fixed_profile_short_is_refused) {
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) history[r][c] = c < 2 ? 300 + 10 * r + c : -1;

    const TreeStep before = sentinel_step();
    TreeStep step = before;
    std::string why;
    /* kTreeLegacyMaskNodes rows is less than the profile's own node count. */
    CHECK_EQ(tree_dispatch_fixed(history, 8, kTreeLegacyMaskNodes, 5, kVocab, 64, 512, true, true,
                                 0, &step, &why), RAD_E_INVAL);
    CHECK(why.find("truncates") != std::string::npos);
    CHECK(staged_untouched(step, before));

    /* And a rounds/capacity pair the profile does not offer is refused by the profile itself. */
    why.clear();
    CHECK_EQ(tree_dispatch_fixed(history, 3, kTreeMaxNodes, 5, kVocab, 64, 512, true, true, 0,
                                 &step, &why), RAD_E_INVAL);
    CHECK(staged_untouched(step, before));
}

TEST(a_wide_tree_is_refused_for_a_sixteen_bit_mask_consumer_not_truncated) {
    int32_t history[8][8];
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) history[r][c] = c < 3 ? 400 + 10 * r + c : -1;

    const TreeStep before = sentinel_step();
    TreeStep step = before;
    std::string why;
    CHECK_EQ(tree_dispatch_fixed(history, 8, kTreeMaxNodes, 5, kVocab, 64, 512, true, true, 16,
                                 &step, &why), RAD_E_UNSUPPORTED);
    CHECK(why.find("uint16") != std::string::npos);
    CHECK(staged_untouched(step, before));

    /* A 16-row tree does have the legacy form and is admitted for that consumer. */
    uint32_t cand[14];
    for (int i = 0; i < 14; ++i) cand[i] = 500u + (uint32_t)i;
    CandidateTree t;
    CHECK(t.build_top2(4, cand, 14, false));
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, 20, kVocab, 512, true, true, 16, &step, &why), RAD_OK);
    CHECK(step.has_mask16);
    for (int32_t j = 0; j < step.n_nodes; ++j) CHECK(step.masks16[j] != 0);
    CHECK_EQ(step.masks16[0], 0x0001);
}

TEST(a_mutated_mask_an_out_of_span_depth_and_a_bad_vocabulary_are_refused) {
    uint32_t draft[3] = {21, 22, 23};
    const TreeStep before = sentinel_step();
    TreeStep step = before;
    std::string why;

    CandidateTree bad_mask;
    CHECK(bad_mask.build_linear(20, draft, 3));
    bad_mask[2].ancestors.reset(1);   /* node 2 no longer attends to node 1 */
    CHECK_EQ(tree_dispatch_admit(bad_mask, 7, kVocab, 512, true, true, 0, &step, &why), RAD_E_INVAL);
    CHECK(why.find("ancestor") != std::string::npos);
    CHECK(staged_untouched(step, before));

    CandidateTree t;
    CHECK(t.build_linear(20, draft, 3));
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, 7, kVocab, /*token_span=*/3, true, true, 0, &step, &why),
             RAD_E_INVAL);
    CHECK(staged_untouched(step, before));
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, 7, /*vocab=*/22, 512, true, true, 0, &step, &why), RAD_E_INVAL);
    CHECK(staged_untouched(step, before));
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, -1, kVocab, 512, true, true, 0, &step, &why), RAD_E_INVAL);
    CHECK(staged_untouched(step, before));
    why.clear();
    CHECK_EQ(tree_dispatch_admit(t, 7, kVocab, 512, true, true, /*mask_bits=*/32, &step, &why),
             RAD_E_INVAL);
    CHECK(staged_untouched(step, before));

    /* A tree of the root alone verifies nothing. */
    CandidateTree root_only;
    CHECK(root_only.build_linear(20, nullptr, 0));
    why.clear();
    CHECK_EQ(tree_dispatch_admit(root_only, 0, kVocab, 512, true, true, 0, &step, &why), RAD_E_INVAL);
    CHECK(staged_untouched(step, before));
}

TEST(the_commit_drops_every_branch_off_the_winner_and_cuts_at_the_stop_token) {
    /* Root 0, children 1 and 2; 1 has children 3 and 4; 2 has child 5. */
    uint32_t cand[6] = {31, 32, 33, 34, 35, 36};
    CandidateTree t;
    CHECK(t.build_top2(30, cand, 5, false));
    CHECK(t.node_count() >= 6);

    TreeStep step;
    std::string why;
    CHECK_EQ(tree_dispatch_admit(t, 12, kVocab, 512, true, true, 0, &step, &why), RAD_OK);

    /* Accept 0->1->3, then the bonus at 3. */
    int32_t pick[kTreeMaxNodes];
    for (int32_t j = 0; j < step.n_nodes; ++j) pick[j] = 0;
    pick[0] = (int32_t)t[1].token;
    pick[1] = (int32_t)t[3].token;
    pick[3] = 888;

    TreeCommit commit;
    CHECK_EQ(tree_dispatch_commit(step, pick, step.n_nodes, is_eog, &commit, &why), RAD_OK);
    CHECK_EQ(commit.n_accepted, 2);
    CHECK_EQ(commit.path[0], 0);
    CHECK_EQ(commit.path[1], 1);
    CHECK_EQ(commit.path[2], 3);
    CHECK_EQ((int32_t)commit.tokens[0], (int32_t)t[1].token);
    CHECK_EQ((int32_t)commit.tokens[1], (int32_t)t[3].token);
    CHECK_EQ((int32_t)commit.tokens[2], 888);
    CHECK_EQ(commit.n_discarded, step.n_nodes - 3);

    /* A stop token on the second accepted node ends the step there, as it does on a chain. */
    pick[1] = 2;
    t[3].token = 2;
    TreeStep stop_step;
    CHECK_EQ(tree_dispatch_admit(t, 12, kVocab, 512, true, true, 0, &stop_step, &why), RAD_OK);
    TreeCommit cut;
    CHECK_EQ(tree_dispatch_commit(stop_step, pick, stop_step.n_nodes, is_eog, &cut, &why), RAD_OK);
    CHECK_EQ(cut.n_accepted, 1);
    CHECK_EQ((int32_t)cut.bonus, 2);
    CHECK_EQ(cut.n_discarded, stop_step.n_nodes - 2);

    /* Nothing accepted: the root's pick is the step's own token and every node is discarded. */
    for (int32_t j = 0; j < step.n_nodes; ++j) pick[j] = 0;
    pick[0] = 999;
    TreeCommit none;
    CHECK_EQ(tree_dispatch_commit(step, pick, step.n_nodes, is_eog, &none, &why), RAD_OK);
    CHECK_EQ(none.n_accepted, 0);
    CHECK_EQ((int32_t)none.bonus, 999);
    CHECK_EQ(none.n_discarded, step.n_nodes - 1);
}

TEST(a_short_pick_buffer_a_tampered_layout_and_a_short_tree_are_refused) {
    uint32_t draft[3] = {41, 42, 43};
    CandidateTree t;
    CHECK(t.build_linear(40, draft, 3));

    TreeStep step;
    std::string why;
    CHECK_EQ(tree_dispatch_admit(t, 3, kVocab, 512, true, true, 0, &step, &why), RAD_OK);

    int32_t pick[8];
    chain_picks(t, pick);

    /* A refused commit writes nothing into the step it would have committed. */
    TreeCommit commit;
    for (int32_t j = 0; j < kTreeMaxNodes; ++j) {
        commit.tokens[j] = -9;
        commit.path[j]   = -9;
    }
    commit.n_accepted = -9;
    commit.n_discarded = -9;
    commit.bonus = 0xdeadbeefu;
    const TreeCommit commit_before = commit;
    CHECK_EQ(tree_dispatch_commit(step, pick, step.n_nodes - 1, is_eog, &commit, &why), RAD_E_INVAL);
    CHECK(why.find("shorter") != std::string::npos);
    CHECK_EQ(std::memcmp(&commit, &commit_before, sizeof commit), 0);

    pick[2] = -1;
    why.clear();
    CHECK_EQ(tree_dispatch_commit(step, pick, step.n_nodes, is_eog, &commit, &why), RAD_E_INVAL);
    CHECK_EQ(std::memcmp(&commit, &commit_before, sizeof commit), 0);
    chain_picks(t, pick);

    /* A staged parent that disagrees with the tree would make the card's walk take another edge. */
    TreeStep tampered = step;
    tampered.parents[2] = 0;
    why.clear();
    CHECK_EQ(tree_dispatch_commit(tampered, pick, tampered.n_nodes, is_eog, &commit, &why),
             RAD_E_INVAL);
    CHECK(why.find("staged layout") != std::string::npos);
    CHECK_EQ(std::memcmp(&commit, &commit_before, sizeof commit), 0);

    /* A step whose node count and tree disagree is refused before the walk. */
    TreeStep short_step = step;
    short_step.n_nodes = 2;
    why.clear();
    CHECK_EQ(tree_dispatch_commit(short_step, pick, 8, is_eog, &commit, &why), RAD_E_INVAL);
    CHECK_EQ(std::memcmp(&commit, &commit_before, sizeof commit), 0);

    /* And a tree of the root alone cannot be committed either. */
    TreeStep empty_step;
    CHECK_EQ(tree_dispatch_commit(empty_step, pick, 8, is_eog, &commit, &why), RAD_E_INVAL);
    CHECK_EQ(std::memcmp(&commit, &commit_before, sizeof commit), 0);
}

RAD_TEST_MAIN()
