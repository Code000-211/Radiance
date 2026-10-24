#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <thread>

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#elif __has_include("../arch/common/rad_tree_candidate.h")
#include "../arch/common/rad_tree_candidate.h"
#endif

#include "sample/accept.h"
#include "sched/draft.h"
#include <sstream>
#include <any>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include "diag.h"
#define private public
#include "sched/scheduler.h"
#include "engine.h"
#undef private
#include "rad_core.h"

#define RAD_ASSERT(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "Assertion failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

using namespace radiance;
using rad::accept_tree_greedy;
using rad::RAD_SCHED_MAX_SPEC;
using rad::Scheduler;
using rad::Engine;
using rad::Config;
using rad::Request;
using rad::ReqState;
using rad::StepEntry;

// =============================================================================
// Helper: Kernel visibility simulation
// Matches r4d_attn_is_visible in libr4d/r4d_attn_decode_h256_gqa6.hip
// =============================================================================
static inline bool r4d_attn_is_visible_ref(
    int k, int prefix_len, int ctx, int klimit, uint16_t mask, int wlo = 0, bool win = false, bool cz = true)
{
    if (win && (k < wlo || (!cz && k > klimit))) return false;
    if (k < prefix_len) return true;
    if (k >= ctx) return false;
    const int tree_k = k - prefix_len;
    return (tree_k < 16) && ((mask & (1u << tree_k)) != 0);
}

// =============================================================================
// TEST 1: Attention Mask Isolation & Causal Tree DAG Semantics
// =============================================================================
static void test_attention_mask_isolation() {
    std::printf("[TEST 1] Testing 2D Causal Tree Attention Mask Isolation & Prefix MSKIP...\n");

    const int prefix_len = 128;
    const int qlen = 16;
    const int ctx = prefix_len + qlen; // 144
    const int klimit = ctx - 1;

    // 1.1: Historical prefix tokens [0..prefix_len-1] must be unconditionally visible
    // for all candidate query rows regardless of their ancestor bitmasks.
    for (uint16_t mask = 0; mask < 256; mask += 17) {
        for (int k = 0; k < prefix_len; ++k) {
            bool vis = r4d_attn_is_visible_ref(k, prefix_len, ctx, klimit, mask);
            RAD_ASSERT(vis);
        }
    }

    // 1.2: Out-of-context tokens (k >= ctx) must be invisible (-inf score)
    for (int k = ctx; k < ctx + 32; ++k) {
        bool vis = r4d_attn_is_visible_ref(k, prefix_len, ctx, klimit, 0xFFFF);
        RAD_ASSERT(!vis);
    }

    // 1.3: Candidate DAG isolation across Top-2 Mode A tree
    CandidateTree tree;
    uint32_t draft_cands[15];
    for (size_t i = 0; i < 15; ++i) draft_cands[i] = 1000u + static_cast<uint32_t>(i);
    tree.build_top2(500u, draft_cands, false);

    // Verify ancestor bitmask correctness
    // Node 0: root, mask = 0x0001 (bit 0)
    RAD_ASSERT(tree[0].ancestor_bitmask == 0x0001);
    // Node 1: left child of 0, mask = bits {0, 1} = 0x0003
    RAD_ASSERT(tree[1].ancestor_bitmask == 0x0003);
    // Node 2: right child of 0, mask = bits {0, 2} = 0x0005
    RAD_ASSERT(tree[2].ancestor_bitmask == 0x0005);
    // Node 3: child of 1, mask = bits {0, 1, 3} = 0x000B
    RAD_ASSERT(tree[3].ancestor_bitmask == 0x000B);
    // Node 4: child of 1, mask = bits {0, 1, 4} = 0x0013
    RAD_ASSERT(tree[4].ancestor_bitmask == 0x0013);
    // Node 5: child of 2, mask = bits {0, 2, 5} = 0x0025
    RAD_ASSERT(tree[5].ancestor_bitmask == 0x0025);
    // Node 6: child of 2, mask = bits {0, 2, 6} = 0x0045
    RAD_ASSERT(tree[6].ancestor_bitmask == 0x0045);

    // Check visibility for Node 3 (query row qrow = 3, ancestor mask = 0x000B)
    const uint16_t mask_node3 = tree_mask16(tree[3].ancestor_bitmask);
    // Prefix keys [0..127]: visible
    RAD_ASSERT(r4d_attn_is_visible_ref(0, prefix_len, ctx, klimit, mask_node3));
    RAD_ASSERT(r4d_attn_is_visible_ref(127, prefix_len, ctx, klimit, mask_node3));

    // Candidate key 0 (node 0, k = 128): visible (ancestor)
    RAD_ASSERT(r4d_attn_is_visible_ref(prefix_len + 0, prefix_len, ctx, klimit, mask_node3));
    // Candidate key 1 (node 1, k = 129): visible (parent)
    RAD_ASSERT(r4d_attn_is_visible_ref(prefix_len + 1, prefix_len, ctx, klimit, mask_node3));
    // Candidate key 2 (node 2, k = 130): INVISIBLE (cross-branch sibling)
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 2, prefix_len, ctx, klimit, mask_node3));
    // Candidate key 3 (node 3, k = 131): visible (self)
    RAD_ASSERT(r4d_attn_is_visible_ref(prefix_len + 3, prefix_len, ctx, klimit, mask_node3));
    // Candidate key 4 (node 4, k = 132): INVISIBLE (sibling child of node 1)
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 4, prefix_len, ctx, klimit, mask_node3));
    // Candidate keys 5 & 6 (nodes 5 & 6, k = 133, 134): INVISIBLE (descendants of node 2)
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 5, prefix_len, ctx, klimit, mask_node3));
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 6, prefix_len, ctx, klimit, mask_node3));

    // Check visibility for Node 5 (query row qrow = 5, ancestor mask = 0x0025)
    const uint16_t mask_node5 = tree_mask16(tree[5].ancestor_bitmask);
    RAD_ASSERT(r4d_attn_is_visible_ref(prefix_len + 0, prefix_len, ctx, klimit, mask_node5)); // root
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 1, prefix_len, ctx, klimit, mask_node5)); // node 1 cross-branch
    RAD_ASSERT(r4d_attn_is_visible_ref(prefix_len + 2, prefix_len, ctx, klimit, mask_node5)); // node 2 parent
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 3, prefix_len, ctx, klimit, mask_node5)); // node 3 cross-branch
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 4, prefix_len, ctx, klimit, mask_node5)); // node 4 cross-branch
    RAD_ASSERT(r4d_attn_is_visible_ref(prefix_len + 5, prefix_len, ctx, klimit, mask_node5)); // self
    RAD_ASSERT(!r4d_attn_is_visible_ref(prefix_len + 6, prefix_len, ctx, klimit, mask_node5)); // node 6 sibling

    // 1.4: MSKIP tile-skip contract:
    // When (kbase + 7 < prefix_len), all 8 keys in the tile are strictly in the historical prefix,
    // so no per-key masking is performed.
    for (int kbase = 0; kbase + 7 < prefix_len; kbase += 8) {
        bool all_visible = true;
        for (int e = 0; e < 8; ++e) {
            if (!r4d_attn_is_visible_ref(kbase + e, prefix_len, ctx, klimit, 0)) {
                all_visible = false;
            }
        }
        RAD_ASSERT(all_visible);
    }

    std::printf("  PASS: Attention mask isolation and prefix MSKIP invariants verified.\n");
}

// =============================================================================
// TEST 2: RoPE Depth Positions Matching root_pos + depth(j)
// =============================================================================
static void test_rope_depth_positions() {
    std::printf("[TEST 2] Testing RoPE Depth Positions Across Topologies & TreeRegistry...\n");

    const uint32_t root_pos = 1024;
    CandidateTree tree;

    // 2.1: Linear Topology
    tree.build_linear(50, nullptr, 15);
    RAD_ASSERT(tree.node_count() == 16);
    uint32_t tokens[RAD_TREE_MAX_NODES] = {0};
    TreeMask ancestors[RAD_TREE_MAX_NODES] = {};
    uint32_t positions[RAD_TREE_MAX_NODES] = {0};
    tree.populate_tensors(tokens, ancestors, positions, root_pos);

    for (size_t i = 0; i < 16; ++i) {
        RAD_ASSERT(positions[i] == root_pos + static_cast<uint32_t>(i));
        RAD_ASSERT(tree[i].depth == static_cast<int32_t>(i));
    }

    // 2.2: Top-2 Mode A (Pure Markov Breadth Branching)
    static const int32_t kExpectedDepthsModeA[16] = {
        0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4
    };
    tree.build_top2(100, nullptr, false);
    RAD_ASSERT(tree.node_count() == 16);
    tree.populate_tensors(tokens, ancestors, positions, root_pos);
    for (size_t i = 0; i < 16; ++i) {
        RAD_ASSERT(tree[i].depth == kExpectedDepthsModeA[i]);
        RAD_ASSERT(positions[i] == root_pos + static_cast<uint32_t>(kExpectedDepthsModeA[i]));
    }
    // Nodes 1 and 2 must have identical depth 1 and position root_pos + 1
    RAD_ASSERT(positions[1] == root_pos + 1);
    RAD_ASSERT(positions[2] == root_pos + 1);
    // Nodes 3, 4, 5, 6 must have identical depth 2 and position root_pos + 2
    for (size_t i = 3; i <= 6; ++i) {
        RAD_ASSERT(positions[i] == root_pos + 2);
    }

    // 2.3: Top-2 Mode B (DLoop Best-Path Extension)
    static const int32_t kExpectedDepthsModeB[16] = {
        0, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    };
    tree.build_top2(200, nullptr, true);
    RAD_ASSERT(tree.node_count() == 16);
    tree.populate_tensors(tokens, ancestors, positions, root_pos);
    for (size_t i = 0; i < 16; ++i) {
        RAD_ASSERT(tree[i].depth == kExpectedDepthsModeB[i]);
        RAD_ASSERT(positions[i] == root_pos + static_cast<uint32_t>(kExpectedDepthsModeB[i]));
    }
    // Nodes 7..15 have monotonically increasing depth 3..11
    for (size_t i = 7; i < 16; ++i) {
        RAD_ASSERT(positions[i] == root_pos + static_cast<uint32_t>(i - 4));
    }

    // 2.4: Top-4 Topology
    static const int32_t kExpectedDepthsTop4[16] = {
        0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    };
    tree.build_top4(300, nullptr);
    RAD_ASSERT(tree.node_count() == 16);
    tree.populate_tensors(tokens, ancestors, positions, root_pos);
    for (size_t i = 0; i < 16; ++i) {
        RAD_ASSERT(tree[i].depth == kExpectedDepthsTop4[i]);
        RAD_ASSERT(positions[i] == root_pos + static_cast<uint32_t>(kExpectedDepthsTop4[i]));
    }

    // 2.5: TreeRegistry Thread Safety and Query
    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();
    const uint64_t seq_id = 4242;
    reg.set(seq_id, tree, root_pos);

    ActiveTreeRecord rec;
    bool found = reg.get(seq_id, &rec);
    RAD_ASSERT(found);
    RAD_ASSERT(rec.active);
    RAD_ASSERT(rec.root_pos == root_pos);
    for (size_t i = 0; i < 16; ++i) {
        RAD_ASSERT(rec.positions[i] == root_pos + static_cast<uint32_t>(kExpectedDepthsTop4[i]));
        RAD_ASSERT(rec.parents[i] == tree[i].parent_id);
    }

    reg.clear(seq_id);
    found = reg.get(seq_id, &rec);
    RAD_ASSERT(!found);

    std::printf("  PASS: RoPE depth position assignments verified across all topologies.\n");
}

// =============================================================================
// TEST 3: DARTree Greedy Verification Parity with Python dartree.py
// =============================================================================
static void test_dartree_greedy_verification() {
    std::printf("[TEST 3] Testing DARTree Greedy Argmax Verification Parity...\n");

    // 3.1: Linear Topology - Full Acceptance
    {
        CandidateTree tree;
        uint32_t draft[15];
        for (uint32_t i = 0; i < 15; ++i) draft[i] = 100u + i;
        tree.build_linear(50u, draft, 15);

        // Teacher predictions: root predicts 100, node 1 predicts 101, ..., node 15 predicts 999
        uint32_t preds[16];
        for (size_t i = 0; i < 15; ++i) preds[i] = 100u + static_cast<uint32_t>(i);
        preds[15] = 999u; // Bonus token from leaf

        TreeVerificationResult res = DARTreeVerifier::verify_tree_greedy(tree, preds, 16);
        RAD_ASSERT(res.accepted_length == 15);
        RAD_ASSERT(res.has_bonus);
        RAD_ASSERT(res.bonus_token == 999u);
        RAD_ASSERT(res.best_path_node_ids[0] == 0);
        for (int32_t j = 0; j < 15; ++j) {
            RAD_ASSERT(res.accepted_tokens[j] == 100u + static_cast<uint32_t>(j));
            RAD_ASSERT(res.best_path_node_ids[j + 1] == j + 1);
        }
    }

    // 3.2: Linear Topology - Partial Acceptance (divergence at depth 4)
    {
        CandidateTree tree;
        uint32_t draft[15];
        for (uint32_t i = 0; i < 15; ++i) draft[i] = 200u + i;
        tree.build_linear(50u, draft, 15);

        uint32_t preds[16];
        preds[0] = 200u; // matches node 1
        preds[1] = 201u; // matches node 2
        preds[2] = 202u; // matches node 3
        preds[3] = 777u; // diverges! (node 4 draft is 203)
        preds[4] = 204u;

        TreeVerificationResult res = DARTreeVerifier::verify_tree_greedy(tree, preds, 16);
        RAD_ASSERT(res.accepted_length == 3);
        RAD_ASSERT(res.has_bonus);
        RAD_ASSERT(res.bonus_token == 777u);
        RAD_ASSERT(res.best_path_node_ids[0] == 0);
        RAD_ASSERT(res.best_path_node_ids[1] == 1);
        RAD_ASSERT(res.best_path_node_ids[2] == 2);
        RAD_ASSERT(res.best_path_node_ids[3] == 3);
        RAD_ASSERT(res.accepted_tokens[0] == 200u);
        RAD_ASSERT(res.accepted_tokens[1] == 201u);
        RAD_ASSERT(res.accepted_tokens[2] == 202u);
    }

    // 3.3: Linear Topology - Zero Acceptance (root mismatch)
    {
        CandidateTree tree;
        uint32_t draft[15];
        for (uint32_t i = 0; i < 15; ++i) draft[i] = 300u + i;
        tree.build_linear(50u, draft, 15);

        uint32_t preds[16];
        preds[0] = 888u; // diverges at root! (draft[0] is 300)

        TreeVerificationResult res = DARTreeVerifier::verify_tree_greedy(tree, preds, 16);
        RAD_ASSERT(res.accepted_length == 0);
        RAD_ASSERT(res.has_bonus);
        RAD_ASSERT(res.bonus_token == 888u);
        RAD_ASSERT(res.best_path_node_ids[0] == 0);
    }

    // 3.4: Top-2 Mode A - Left Branch vs Right Branch Traversal
    {
        CandidateTree tree;
        uint32_t draft[15];
        for (uint32_t i = 0; i < 15; ++i) draft[i] = 400u + i;
        tree.build_top2(50u, draft, false);

        // Scenario: Right branch (node 2) chosen
        uint32_t preds[16] = {0};
        preds[0] = 401u; // Root selects node 2
        preds[2] = 405u; // Node 2 selects node 6
        preds[6] = 9999u; // Node 6 diverges/leaf, bonus = 9999

        TreeVerificationResult res = DARTreeVerifier::verify_tree_greedy(tree, preds, 16);
        RAD_ASSERT(res.accepted_length == 2);
        RAD_ASSERT(res.has_bonus);
        RAD_ASSERT(res.bonus_token == 9999u);
        RAD_ASSERT(res.best_path_node_ids[0] == 0);
        RAD_ASSERT(res.best_path_node_ids[1] == 2);
        RAD_ASSERT(res.best_path_node_ids[2] == 6);
        RAD_ASSERT(res.accepted_tokens[0] == 401u);
        RAD_ASSERT(res.accepted_tokens[1] == 405u);
    }

    // 3.5: Top-4 Topology - Branch 3 Selection
    {
        CandidateTree tree;
        uint32_t draft[15];
        for (uint32_t i = 0; i < 15; ++i) draft[i] = 500u + i;
        tree.build_top4(50u, draft);

        uint32_t preds[16] = {0};
        preds[0] = 502u;  // Root selects node 3
        preds[3] = 514u;  // Node 3 selects node 15
        preds[15] = 12345u; // Node 15 is leaf, bonus = 12345

        TreeVerificationResult res = DARTreeVerifier::verify_tree_greedy(tree, preds, 16);
        RAD_ASSERT(res.accepted_length == 2);
        RAD_ASSERT(res.has_bonus);
        RAD_ASSERT(res.bonus_token == 12345u);
        RAD_ASSERT(res.best_path_node_ids[0] == 0);
        RAD_ASSERT(res.best_path_node_ids[1] == 3);
        RAD_ASSERT(res.best_path_node_ids[2] == 15);
        RAD_ASSERT(res.accepted_tokens[0] == 502u);
        RAD_ASSERT(res.accepted_tokens[1] == 514u);
    }

    // 3.6: Deterministic Tie Breaking (Lowest node_id wins)
    {
        CandidateTree tree;
        tree.reset();
        tree.add_node(0, -1, 10u, 0);
        tree.add_node(1, 0, 99u, 1); // Child 1 with token 99
        tree.add_node(2, 0, 99u, 1); // Child 2 with token 99

        uint32_t preds[3] = {99u, 111u, 222u};
        TreeVerificationResult res = DARTreeVerifier::verify_tree_greedy(tree, preds, 3);
        RAD_ASSERT(res.accepted_length == 1);
        // Tie must be broken by lowest node_id: node 1
        RAD_ASSERT(res.best_path_node_ids[1] == 1);
        RAD_ASSERT(res.bonus_token == 111u);
    }

    std::printf("  PASS: DARTree greedy verification parity with dartree.py verified.\n");
}

// =============================================================================
// TEST 4: Divergence Bonus Token Emission & verify_with_logits
// =============================================================================
static void test_divergence_bonus_emission() {
    std::printf("[TEST 4] Testing Divergence Bonus Token Emission & Logits Matrix Verification...\n");

    // 4.1: Logits tensor argmax equivalence
    CandidateTree tree;
    uint32_t draft[15];
    for (uint32_t i = 0; i < 15; ++i) draft[i] = 10u + i;
    tree.build_linear(5u, draft, 15);

    const size_t vocab_size = 100;
    std::vector<float> logits(16 * vocab_size, 0.0f);

    // Set argmaxes:
    // Root -> token 10 (matches node 1)
    logits[0 * vocab_size + 10] = 5.0f;
    // Node 1 -> token 11 (matches node 2)
    logits[1 * vocab_size + 11] = 6.0f;
    // Node 2 -> token 77 (diverges! bonus = 77)
    logits[2 * vocab_size + 77] = 8.0f;

    TreeVerificationResult res = DARTreeVerifier::verify_with_logits(
        tree, logits.data(), vocab_size
    );

    RAD_ASSERT(res.accepted_length == 2);
    RAD_ASSERT(res.has_bonus);
    RAD_ASSERT(res.bonus_token == 77u);
    RAD_ASSERT(res.best_path_node_ids[0] == 0);
    RAD_ASSERT(res.best_path_node_ids[1] == 1);
    RAD_ASSERT(res.best_path_node_ids[2] == 2);
    RAD_ASSERT(res.accepted_tokens[0] == 10u);
    RAD_ASSERT(res.accepted_tokens[1] == 11u);

    // 4.2: Boundary cases
    // Empty tree
    CandidateTree empty_tree;
    TreeVerificationResult res_empty = DARTreeVerifier::verify_tree_greedy(
        empty_tree, static_cast<const uint32_t*>(nullptr), 0
    );
    RAD_ASSERT(res_empty.accepted_length == -1);
    RAD_ASSERT(!res_empty.has_bonus);

    // Single root node only
    CandidateTree single_tree;
    single_tree.add_node(0, -1, 50u, 0);
    uint32_t pred_single[1] = {42u};
    TreeVerificationResult res_single = DARTreeVerifier::verify_tree_greedy(single_tree, pred_single, 1);
    RAD_ASSERT(res_single.accepted_length == 0);
    RAD_ASSERT(res_single.bonus_token == 42u);
    RAD_ASSERT(res_single.best_path_node_ids[0] == 0);

    std::printf("  PASS: Divergence bonus emission and verify_with_logits verified.\n");
}

// =============================================================================
// TEST 5: accept_tree_greedy Adapter Function Contracts
// =============================================================================
static void test_accept_tree_greedy_adapter() {
    std::printf("[TEST 5] Testing accept_tree_greedy Adapter Function Contracts...\n");

    CandidateTree tree;
    uint32_t draft[15];
    for (uint32_t i = 0; i < 15; ++i) draft[i] = 100u + i;
    tree.build_linear(50u, draft, 15);

    int32_t target_argmax[16];
    target_argmax[0] = 100;
    target_argmax[1] = 101;
    target_argmax[2] = 102;
    target_argmax[3] = 999; // Bonus divergence

    int32_t out_accepted[RAD_SCHED_MAX_SPEC + 1] = {0};
    int32_t n_accepted = 0;
    int32_t out_bonus = 0;
    int32_t out_best_nodes[RAD_SCHED_MAX_SPEC + 1] = {0};

    // 5.1: Successful invocation
    int ret = accept_tree_greedy(
        tree,
        target_argmax,
        16,
        out_accepted,
        &n_accepted,
        &out_bonus,
        out_best_nodes
    );

    RAD_ASSERT(ret == RAD_OK);
    RAD_ASSERT(n_accepted == 3);
    RAD_ASSERT(out_bonus == 999);
    RAD_ASSERT(out_accepted[0] == 100);
    RAD_ASSERT(out_accepted[1] == 101);
    RAD_ASSERT(out_accepted[2] == 102);
    RAD_ASSERT(out_accepted[3] == 999); // Contract: out_accepted[n_accepted] is the bonus token!

    RAD_ASSERT(out_best_nodes[0] == 0);
    RAD_ASSERT(out_best_nodes[1] == 1);
    RAD_ASSERT(out_best_nodes[2] == 2);
    RAD_ASSERT(out_best_nodes[3] == 3);

    // 5.2: Error handling on invalid inputs
    RAD_ASSERT(accept_tree_greedy(tree, nullptr, 16, out_accepted, &n_accepted, &out_bonus) == RAD_E_INVAL);
    RAD_ASSERT(accept_tree_greedy(tree, target_argmax, 0, out_accepted, &n_accepted, &out_bonus) == RAD_E_INVAL);
    RAD_ASSERT(accept_tree_greedy(tree, target_argmax, 16, nullptr, &n_accepted, &out_bonus) == RAD_E_INVAL);
    RAD_ASSERT(accept_tree_greedy(tree, target_argmax, 16, out_accepted, nullptr, &out_bonus) == RAD_E_INVAL);

    CandidateTree empty_tree;
    RAD_ASSERT(accept_tree_greedy(empty_tree, target_argmax, 16, out_accepted, &n_accepted, &out_bonus) == RAD_E_INVAL);

    std::printf("  PASS: accept_tree_greedy adapter contracts verified.\n");
}

// =============================================================================
// TEST 6: Scheduler::set_draft_tree TreeRegistry Population & Query Contracts
// =============================================================================
static void test_scheduler_set_draft_tree_populates_registry() {
    std::printf("[TEST 6] Testing Scheduler::set_draft_tree Populates TreeRegistry...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    Scheduler sched;

    // 6.1: Submit an active request with may_speculate = true
    const uint64_t seq_id = 7001;
    auto req = std::make_unique<Request>();
    req->id = seq_id;
    req->prompt = {101, 102, 103, 104};
    req->may_speculate = true;
    req->max_tokens = 64;
    int rc = sched.add(std::move(req));
    RAD_ASSERT(rc == RAD_OK);

    // 6.2: Build Top-2 Candidate Tree
    CandidateTree tree;
    uint32_t draft_tokens[15];
    for (size_t i = 0; i < 15; ++i) draft_tokens[i] = 2000u + static_cast<uint32_t>(i);
    tree.build_top2(100u, draft_tokens, false);

    // Call Scheduler::set_draft_tree
    const uint32_t root_pos = 128;
    rc = sched.set_draft_tree(seq_id, tree, root_pos);
    RAD_ASSERT(rc == RAD_OK);

    // 6.3: Verify TreeRegistry contains the populated tree record
    ActiveTreeRecord trec;
    bool found = reg.get(seq_id, &trec);
    RAD_ASSERT(found);
    RAD_ASSERT(trec.active);
    RAD_ASSERT(trec.root_pos == root_pos);
    RAD_ASSERT(trec.tree.node_count() == 16);

    uint32_t exp_tokens[RAD_TREE_MAX_NODES] = {0};
    TreeMask exp_ancestors[RAD_TREE_MAX_NODES] = {};
    uint32_t exp_positions[RAD_TREE_MAX_NODES] = {0};
    tree.populate_tensors(exp_tokens, exp_ancestors, exp_positions, root_pos);

    for (size_t i = 0; i < 16; ++i) {
        RAD_ASSERT(trec.positions[i] == exp_positions[i]);
        RAD_ASSERT(trec.ancestors[i] == exp_ancestors[i]);
        RAD_ASSERT(trec.parents[i] == tree[i].parent_id);
    }

    // 6.4: Negative test: request with may_speculate = false must clear TreeRegistry
    const uint64_t seq_id_no_spec = 7002;
    auto req_no_spec = std::make_unique<Request>();
    req_no_spec->id = seq_id_no_spec;
    req_no_spec->prompt = {201, 202};
    req_no_spec->may_speculate = false;
    RAD_ASSERT(sched.add(std::move(req_no_spec)) == RAD_OK);

    rc = sched.set_draft_tree(seq_id_no_spec, tree, 200);
    RAD_ASSERT(rc == RAD_OK);
    RAD_ASSERT(!reg.get(seq_id_no_spec, &trec));

    // 6.5: Negative test: empty tree must clear TreeRegistry
    CandidateTree empty_tree;
    rc = sched.set_draft_tree(seq_id, empty_tree, root_pos);
    RAD_ASSERT(rc == RAD_OK);
    RAD_ASSERT(!reg.get(seq_id, &trec));

    // 6.6: Negative test: unknown sequence ID returns RAD_E_NOTFOUND
    RAD_ASSERT(sched.set_draft_tree(99999, tree, root_pos) == RAD_E_NOTFOUND);

    std::printf("  PASS: Scheduler::set_draft_tree TreeRegistry population and query verified.\n");
}

// =============================================================================
// TEST 7: Engine::collect_step Clears TreeRegistry for Completed Sequences
// =============================================================================
static void test_engine_collect_step_clears_completed_sequences() {
    std::printf("[TEST 7] Testing Engine::collect_step Clears TreeRegistry for Completed Sequences...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    const uint64_t seq_id = 8001;
    CandidateTree tree;
    uint32_t draft_tokens[15] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    tree.build_linear(50u, draft_tokens, 15);

    reg.set(seq_id, tree, 100);
    ActiveTreeRecord trec;
    RAD_ASSERT(reg.get(seq_id, &trec));

    // Initialize Engine
    Config cfg;
    Engine engine(cfg);

    // Set up mock StepPlan entry in engine's scheduler
    StepEntry entry{};
    entry.seq = seq_id;
    entry.n_spec = 15;
    entry.n_tokens = 16;
    Request req;
    req.id = seq_id;
    entry.req = &req;
    engine.sched_.plan_.e.push_back(entry);

    // Build mock StepBatch and StepOut
    RadBatch batch{};
    batch.step = 1;
    batch.n_seq = 1;
    batch.n_spec = 15;

    Engine::StepOut so;
    so.n_rows = 16;
    int32_t sample_rows[16];
    for (int i = 0; i < 15; ++i) sample_rows[i] = 10 + i;
    sample_rows[15] = 999; // leaf bonus
    so.rb = sample_rows;
    so.first_row = { 0 };

    // Before collect_step: record exists
    RAD_ASSERT(reg.get(seq_id, &trec));

    // Invoke collect_step
    int rc = engine.collect_step(&batch, so);
    RAD_ASSERT(rc == RAD_OK);

    // After collect_step: completed sequence tree must be cleared
    RAD_ASSERT(!reg.get(seq_id, &trec));

    std::printf("  PASS: Engine::collect_step clears TreeRegistry for completed sequences verified.\n");
}

// =============================================================================
// TEST 8: Engine::step_state_fini Clears All TreeRegistry Records
// =============================================================================
static void test_engine_step_state_fini_clears_records() {
    std::printf("[TEST 8] Testing Engine::step_state_fini Clears All Tree Records...\n");

    TreeRegistry& reg = TreeRegistry::instance();
    reg.clear_all();

    CandidateTree tree;
    uint32_t draft_tokens[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    tree.build_top4(100u, draft_tokens);

    reg.set(9001, tree, 10);
    reg.set(9002, tree, 20);
    reg.set(9003, tree, 30);

    ActiveTreeRecord rec;
    RAD_ASSERT(reg.get(9001, &rec));
    RAD_ASSERT(reg.get(9002, &rec));
    RAD_ASSERT(reg.get(9003, &rec));

    // Direct invocation of step_state_fini
    Config cfg;
    Engine engine(cfg);
    engine.step_state_fini();

    RAD_ASSERT(!reg.get(9001, &rec));
    RAD_ASSERT(!reg.get(9002, &rec));
    RAD_ASSERT(!reg.get(9003, &rec));

    // Destructor lifecycle invocation of step_state_fini
    reg.set(9004, tree, 40);
    RAD_ASSERT(reg.get(9004, &rec));
    {
        Engine scoped_engine(cfg);
    }
    RAD_ASSERT(!reg.get(9004, &rec));

    std::printf("  PASS: Engine::step_state_fini clears all tree records successfully.\n");
}

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================
int main() {
    std::printf("===================================================================\n");
    std::printf(" Radiance DARTree Verification Test Suite (Milestone 2 It2)\n");
    std::printf("===================================================================\n");

    test_attention_mask_isolation();
    test_rope_depth_positions();
    test_dartree_greedy_verification();
    test_divergence_bonus_emission();
    test_accept_tree_greedy_adapter();
    test_scheduler_set_draft_tree_populates_registry();
    test_engine_collect_step_clears_completed_sequences();
    test_engine_step_state_fini_clears_records();

    std::printf("===================================================================\n");
    std::printf(" ALL 8 DARTREE VERIFICATION TEST SUITES PASSED SUCCESSFULLY!\n");
    std::printf("===================================================================\n");
    return 0;
}
