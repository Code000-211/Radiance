#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <bitset>
#include <iterator>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace radiance {

// Root-inclusive host tree capacity: one anchor plus up to 64 proposals (MTP8/64). The GPU packed
// row stays count + token[capacity] + parent[capacity]; only the host's geometry is this wide.
constexpr size_t RAD_TREE_MAX_NODES = 65;
// Tree levels, root included: a full-capacity chain reaches depth RAD_TREE_MAX_DEPTH - 1.
constexpr size_t RAD_TREE_MAX_DEPTH = 65;
// Host-only ancestor set over node ids. Bit j is node j, so node 64 needs no shift past a word.
using TreeMask = std::bitset<RAD_TREE_MAX_NODES>;
// The legacy uint16 attention mask (RadBatch::tree_ancestors) describes at most this many rows.
constexpr size_t RAD_TREE_LEGACY_MASK_NODES = 16;

/* The legacy uint16 form of a node's ancestor set. Exact for nodes below 16 -- every ancestor
 * precedes its descendant -- and only staged for trees no wider than that. */
inline uint16_t tree_mask16(const TreeMask& m) {
    return static_cast<uint16_t>((m & TreeMask(0xFFFFu)).to_ullong());
}

/**
 * Speculative Candidate Tree Node.
 * Represents a single speculated token node within the DARTree DAG.
 */
struct TreeNode {
    int32_t  node_id;            // Node identifier [0, RAD_TREE_MAX_NODES), topologically ordered
    int32_t  parent_id;          // Parent node ID (-1 for root node)
    uint32_t token_id;           // Speculated token vocabulary ID
    int32_t  depth;              // Depth in candidate tree (root = 0)
    float    score;              // Cumulative proposal/transition confidence score
    TreeMask ancestor_bitmask;   // bit j == 1 iff node j is ancestor or j == node_id
    int32_t  gdn_slot;           // Allocated scratchpad slot (equal to node_id)
};

/**
 * Speculative Candidate Tree DAG Container.
 * Manages topological candidate token graphs, O(1) ancestor bitmask calculations,
 * and standard topologies (Linear bf=1, Top-2 bf=2 Mode A & B, Top-4 bf=4).
 */
class CandidateTree {
public:
    TreeNode nodes[RAD_TREE_MAX_NODES]{};
    size_t   count = 0;

    void reset() {
        std::fill(std::begin(nodes), std::end(nodes), TreeNode{});
        count = 0;
    }

    size_t size() const { return count; }
    size_t node_count() const { return count; }
    bool empty() const { return count == 0; }

    const TreeNode& operator[](size_t idx) const { return nodes[idx]; }
    TreeNode& operator[](size_t idx) { return nodes[idx]; }

    /**
     * Adds a node to the candidate tree, enforcing topological order and
     * incrementally computing the O(1) ancestor bitmask:
     *   mask[node_id] = (parent_id < 0) ? {node_id} : (mask[parent_id] | {node_id})
     */
    bool add_node(int32_t node_id, int32_t parent_id, uint32_t token_id, int32_t depth, float score = 1.0f) {
        if (count >= RAD_TREE_MAX_NODES || node_id < 0 || node_id >= static_cast<int32_t>(RAD_TREE_MAX_NODES)) {
            return false;
        }
        if (node_id != static_cast<int32_t>(count)) {
            // Must be added in strict topological order
            return false;
        }
        TreeMask mask;
        if (node_id == 0) {
            if (parent_id != -1 || depth != 0) return false; // Strictly enforce parent_id == -1 and depth == 0 for root
            mask.set(0);
        } else {
            if (parent_id < 0 || parent_id >= node_id) return false; // Non-root must have valid parent in [0, node_id)
            if (depth != nodes[parent_id].depth + 1) return false;
            mask = nodes[parent_id].ancestor_bitmask;
            mask.set(static_cast<size_t>(node_id));
        }

        nodes[count].node_id = node_id;
        nodes[count].parent_id = parent_id;
        nodes[count].token_id = token_id;
        nodes[count].depth = depth;
        nodes[count].score = score;
        nodes[count].ancestor_bitmask = mask;
        nodes[count].gdn_slot = node_id;

        count++;
        return true;
    }

    /**
     * Checks if ancestor_id is an ancestor of node_id (or equal to node_id).
     */
    bool is_ancestor(int32_t ancestor_id, int32_t node_id) const {
        if (node_id < 0 || node_id >= static_cast<int32_t>(count)) return false;
        if (ancestor_id < 0 || ancestor_id >= static_cast<int32_t>(count)) return false;
        return nodes[node_id].ancestor_bitmask.test(static_cast<size_t>(ancestor_id));
    }

    TreeMask get_ancestor_bitmask(int32_t node_id) const {
        if (node_id < 0 || node_id >= static_cast<int32_t>(count)) return TreeMask{};
        return nodes[node_id].ancestor_bitmask;
    }

    /**
     * Builds Linear Topology (branch_factor = 1).
     * Single linear path: node 0 is root, node i (1..N-1) has parent i-1.
     * Ancestor bitmask is lower triangular: mask[i] = bits {0..i}.
     */
    void build_linear(uint32_t root_token, const uint32_t* draft_tokens, size_t n_draft = 15) {
        reset();
        add_node(0, -1, root_token, 0, 1.0f);

        size_t n = std::min(n_draft, RAD_TREE_MAX_NODES - 1);
        for (size_t i = 0; i < n; ++i) {
            int32_t nid = static_cast<int32_t>(i + 1);
            int32_t pid = static_cast<int32_t>(i);
            int32_t dep = static_cast<int32_t>(i + 1);
            uint32_t tok = draft_tokens ? draft_tokens[i] : (root_token + static_cast<uint32_t>(i + 1));
            add_node(nid, pid, tok, dep, 1.0f);
        }
    }

    /**
     * Builds Top-2 Tree Topology (branch_factor = 2), at most 16 nodes: the tables below ARE the
     * topology, so a larger candidate count is clamped to them rather than read past them.
     * Mode A (dloop_best_path = false): Pure breadth branching:
     *   Depth 0 (node 0) -> Depth 1 (nodes 1,2 parent 0) -> Depth 2 (nodes 3,4 parent 1; 5,6 parent 2)
     *   Depth 3 (nodes 7..14) -> Depth 4 (node 15 parent 7). Total 16 nodes.
     * Mode B (dloop_best_path = true): Branch bf=2 up to depth 2 (nodes 0..6), then linearly
     *   extend highest-scoring node (node 3) through depths 3..11 (nodes 7..15).
     */
    void build_top2(uint32_t root_token, const uint32_t* draft_candidates, bool dloop_best_path = false, size_t n_candidates = 15) {
        reset();
        add_node(0, -1, root_token, 0, 1.0f);

        if (!dloop_best_path) {
            // Mode A: Pure Markov Breadth Branching
            // Parents array: [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
            static const int32_t kParentsModeA[16] = {
                -1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7
            };
            static const int32_t kDepthsModeA[16] = {
                0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4
            };
            for (size_t i = 1; i <= std::min(n_candidates, std::size(kParentsModeA) - 1); ++i) {
                uint32_t tok = draft_candidates ? draft_candidates[i - 1] : (root_token * 31u + static_cast<uint32_t>(i) * 7u);
                add_node(static_cast<int32_t>(i), kParentsModeA[i], tok, kDepthsModeA[i], 1.0f / static_cast<float>(i + 1));
            }
        } else {
            // Mode B: DLoop Best-Path Extension (branching at depths 1, 2, linear thereafter)
            static const int32_t kParentsModeB[16] = {
                -1, 0, 0, 1, 1, 2, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14
            };
            static const int32_t kDepthsModeB[16] = {
                0, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
            };
            for (size_t i = 1; i <= std::min(n_candidates, std::size(kParentsModeB) - 1); ++i) {
                uint32_t tok = draft_candidates ? draft_candidates[i - 1] : (root_token * 31u + static_cast<uint32_t>(i) * 7u);
                add_node(static_cast<int32_t>(i), kParentsModeB[i], tok, kDepthsModeB[i], 1.0f / static_cast<float>(i + 1));
            }
        }
    }

    /**
     * Builds Top-4 Tree Topology (branch_factor = 4), at most 16 nodes (clamped to the tables).
     * Depth 0: node 0
     * Depth 1: nodes 1, 2, 3, 4 (parent 0)
     * Depth 2: nodes 5..8 (parent 1), 9..12 (parent 2), 13..15 (parent 3; budget 16 reached).
     */
    void build_top4(uint32_t root_token, const uint32_t* draft_candidates, size_t n_candidates = 15) {
        reset();
        add_node(0, -1, root_token, 0, 1.0f);

        static const int32_t kParentsTop4[16] = {
            -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3
        };
        static const int32_t kDepthsTop4[16] = {
            0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
        };
        for (size_t i = 1; i <= std::min(n_candidates, std::size(kParentsTop4) - 1); ++i) {
            uint32_t tok = draft_candidates ? draft_candidates[i - 1] : (root_token * 17u + static_cast<uint32_t>(i) * 3u);
            add_node(static_cast<int32_t>(i), kParentsTop4[i], tok, kDepthsTop4[i], 1.0f / static_cast<float>(i + 1));
        }
    }

    /**
     * Populates host tensor arrays, each RAD_TREE_MAX_NODES long:
     * - out_tokens: token IDs
     * - out_ancestors: ancestor sets
     * - out_positions: RoPE positions: pos(i) = root_pos + depth(i)
     */
    void populate_tensors(uint32_t* out_tokens, TreeMask* out_ancestors, uint32_t* out_positions, uint32_t root_pos) const {
        for (size_t i = 0; i < RAD_TREE_MAX_NODES; ++i) {
            if (i < count) {
                if (out_tokens) out_tokens[i] = nodes[i].token_id;
                if (out_ancestors) out_ancestors[i] = nodes[i].ancestor_bitmask;
                if (out_positions) out_positions[i] = root_pos + static_cast<uint32_t>(nodes[i].depth);
            } else {
                if (out_tokens) out_tokens[i] = 0;
                if (out_ancestors) out_ancestors[i].reset();
                if (out_positions) out_positions[i] = root_pos;
            }
        }
    }
};

/**
 * Outcome of DARTree candidate tree verification at temperature 0.0.
 */
struct TreeVerificationResult {
    int32_t  best_path_node_ids[RAD_TREE_MAX_NODES + 1]; // [0, accepted_1, ..., accepted_m]
    uint32_t accepted_tokens[RAD_TREE_MAX_NODES];        // Verified draft tokens (excluding root)
    uint32_t bonus_token;                                // Greedy predicted bonus token
    int32_t  accepted_length;                            // Number of accepted candidate draft tokens (m)
    bool     has_bonus;                                  // True if valid bonus token was emitted
    bool     oracle_equivalent;                          // Parity conformance flag

    size_t num_accepted() const { return static_cast<size_t>(accepted_length >= 0 ? accepted_length : 0); }
    size_t path_length() const { return static_cast<size_t>(accepted_length >= 0 ? accepted_length + 1 : 0); }
};

/**
 * DARTree Greedy Argmax Verifier.
 * Bit-for-bit mathematical equivalent of DARTreeVerifier in dartree.py.
 */
class DARTreeVerifier {
public:
    static inline TreeVerificationResult verify_tree_greedy(
        const CandidateTree& tree,
        const uint32_t* teacher_preds,
        size_t n_preds)
    {
        TreeVerificationResult res{};
        std::memset(res.best_path_node_ids, 0, sizeof(res.best_path_node_ids));
        std::memset(res.accepted_tokens, 0, sizeof(res.accepted_tokens));
        res.bonus_token = 0;
        res.accepted_length = -1;
        res.has_bonus = false;
        res.oracle_equivalent = true;

        if (tree.empty() || !teacher_preds || n_preds == 0) {
            return res;
        }

        const size_t n_nodes = tree.node_count();
        int32_t curr_node_id = 0; // Root is node 0
        res.best_path_node_ids[0] = 0;
        int32_t acc_len = 0;
        uint32_t bonus = 0;
        bool found_bonus = false;

        while (curr_node_id >= 0 && static_cast<size_t>(curr_node_id) < n_nodes &&
               static_cast<size_t>(curr_node_id) < n_preds) {
            
            uint32_t greedy_target = teacher_preds[curr_node_id];

            // Search children of curr_node_id in topological order
            int32_t matching_child_id = -1;
            const size_t limit = std::min(n_nodes, n_preds);
            for (size_t i = static_cast<size_t>(curr_node_id + 1); i < limit; ++i) {
                if (tree.nodes[i].parent_id == curr_node_id) {
                    if (tree.nodes[i].token_id == greedy_target) {
                        matching_child_id = static_cast<int32_t>(i);
                        break; // Deterministic tie-break: lowest node_id
                    }
                }
            }

            if (matching_child_id >= 0) {
                res.accepted_tokens[acc_len] = tree.nodes[matching_child_id].token_id;
                res.best_path_node_ids[acc_len + 1] = matching_child_id;
                acc_len++;
                curr_node_id = matching_child_id;

                // EOS is vocabulary-specific; Engine truncates using Vocab::is_eog.
            } else {
                // Divergence point or leaf reached: greedy_target is bonus token
                bonus = greedy_target;
                found_bonus = true;
                break;
            }
        }

        res.accepted_length = acc_len;
        res.bonus_token = bonus;
        res.has_bonus = found_bonus;
        return res;
    }

    static inline TreeVerificationResult verify_tree_greedy(
        const CandidateTree& tree,
        const int32_t* teacher_preds,
        size_t n_preds)
    {
        if (!teacher_preds) return verify_tree_greedy(tree, static_cast<const uint32_t*>(nullptr), 0);
        uint32_t u_preds[RAD_TREE_MAX_NODES] = {0};
        size_t n = std::min(n_preds, RAD_TREE_MAX_NODES);
        for (size_t i = 0; i < n; ++i) {
            u_preds[i] = static_cast<uint32_t>(teacher_preds[i]);
        }
        return verify_tree_greedy(tree, u_preds, n);
    }

    static inline TreeVerificationResult verify_with_logits(
        const CandidateTree& tree,
        const float* teacher_logits,
        size_t vocab_size)
    {
        TreeVerificationResult res{};
        res.accepted_length = -1;
        res.has_bonus = false;
        res.oracle_equivalent = true;

        if (tree.empty() || !teacher_logits || vocab_size == 0) {
            return res;
        }

        const size_t n_nodes = tree.node_count();
        uint32_t greedy_preds[RAD_TREE_MAX_NODES] = {0};

        for (size_t i = 0; i < n_nodes; ++i) {
            const float* row = teacher_logits + i * vocab_size;
            size_t best_idx = 0;
            float best_val = row[0];
            for (size_t v = 1; v < vocab_size; ++v) {
                if (row[v] > best_val) {
                    best_val = row[v];
                    best_idx = v;
                }
            }
            greedy_preds[i] = static_cast<uint32_t>(best_idx);
        }

        return verify_tree_greedy(tree, greedy_preds, n_nodes);
    }
};

/**
 * Thread-safe registry for tracking active speculative tree DAGs and computed positions per sequence.
 */
struct ActiveTreeRecord {
    CandidateTree tree;
    uint32_t positions[RAD_TREE_MAX_NODES] = {};
    TreeMask ancestors[RAD_TREE_MAX_NODES] = {};
    int32_t  parents[RAD_TREE_MAX_NODES] = {};
    uint32_t root_pos = 0;
    bool active = false;
};

class TreeRegistry {
public:
    static TreeRegistry& instance() {
        static TreeRegistry s_inst;
        return s_inst;
    }

    void set(uint64_t seq_id, const CandidateTree& tree, uint32_t root_pos) {
        std::lock_guard<std::mutex> lk(mu_);
        ActiveTreeRecord rec;
        rec.tree = tree;
        rec.root_pos = root_pos;
        rec.active = true;
        tree.populate_tensors(nullptr, rec.ancestors, rec.positions, root_pos);
        for (size_t i = 0; i < tree.node_count() && i < RAD_TREE_MAX_NODES; ++i) {
            rec.parents[i] = tree[i].parent_id;
        }
        records_[seq_id] = rec;
    }

    bool get(uint64_t seq_id, ActiveTreeRecord* out) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = records_.find(seq_id);
        if (it != records_.end() && it->second.active) {
            if (out) *out = it->second;
            return true;
        }
        return false;
    }

    void clear(uint64_t seq_id) {
        std::lock_guard<std::mutex> lk(mu_);
        records_.erase(seq_id);
    }

    void clear_all() {
        std::lock_guard<std::mutex> lk(mu_);
        records_.clear();
    }

private:
    std::mutex mu_;
    std::unordered_map<uint64_t, ActiveTreeRecord> records_;
};

} // namespace radiance

