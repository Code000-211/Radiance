/* rad_tree_candidate.h -- the candidate tree a tree-layout verify step walks.
 *
 * Ported from the ABI-16 FullDrafter review snapshot
 * (candidates/fulldrafter-abi16-review/source/arch/common/rad_tree_candidate.h at 5c3bb1d), which
 * is the archived implementation of the tree geometries and the greedy winner walk. Only the host
 * model is ported: geometry, ancestor masks, the fixed MTP tree the archived MTP head accumulated,
 * and the walk that names the accepted path. Nothing here talks to a device, and nothing in the
 * engine selects a tree yet -- core/sched/batch.cpp still refuses `spec_is_tree`, because no
 * attention or state kernel honours an ancestor mask in this tree.
 *
 * ============================== THE UNITS ==============================
 *
 *   node        one verified row of a step: the root (the last committed token) plus one row per
 *               proposal. Node ids are topological: a parent is always a lower id than its child,
 *               so id order is also a valid row order for the passes that follow.
 *   depth       edges from the root; depth 0 is the root alone.
 *   position    root_pos + depth, the RoPE/KV position of a node's token (position()).
 *   capacity    node budget, root included: 2..65. RAD_TREE_MAX_NODES is 65 because the fixed
 *               MTP8 geometry is 8 spine rows + 56 alternates + the root.
 *   mask        ancestor set of a node, bit j set when node j is an ancestor OR the node itself.
 *               `tree_mask16` is the legacy uint16 form, exact only for trees of at most 16 rows.
 *
 * A tree is valid only if every parent id precedes its child, every token id is below the
 * vocabulary, no depth passes the token span the step has room for, and every stored mask equals
 * the one its parents imply. `fits` and `masks_consistent` are the checks a caller runs before it
 * dispatches: a wrong mask is not a wrong score, it conditions a row on a sibling's text.
 */
#pragma once

#include "rad_types.h"   /* RAD_OK, RAD_E_* */

#include <bitset>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace rad {

/* Root-inclusive host capacity: one anchor plus up to 64 proposals (MTP8/64). */
inline constexpr int32_t kTreeMaxNodes = 65;
/* Trees no wider than this have an exact uint16 ancestor-mask form. */
inline constexpr int32_t kTreeLegacyMaskNodes = 16;
/* Alternates the MTP head proposes per round, its top-k shortlist. */
inline constexpr int32_t kTreeRoundCandidates = 8;

using TreeMask = std::bitset<static_cast<size_t>(kTreeMaxNodes)>;

/* The legacy uint16 form. Exact for trees no wider than kTreeLegacyMaskNodes -- every ancestor
 * precedes its descendant. A mask with a bit above that width has no uint16 form, so it returns 0
 * instead of a silently truncated set: an ancestor set that lost its high bits conditions a row on
 * a token that is not on its path. Callers that can see wider trees use the full TreeMask. */
inline uint16_t tree_mask16(const TreeMask& m) {
    if ((m >> kTreeLegacyMaskNodes).any()) return 0;
    return static_cast<uint16_t>(m.to_ullong());
}

struct TreeNode {
    int32_t  parent = -1;   /* -1 for the root */
    int32_t  depth  = 0;    /* derived from the parent, never caller-supplied */
    uint32_t token  = 0;
    TreeMask ancestors;     /* bits of the nodes this one attends to, itself included */
};

class CandidateTree {
public:
    void reset() {
        for (int32_t i = 0; i < kTreeMaxNodes; ++i) nodes[i] = TreeNode{};
        count = 0;
    }

    int32_t size() const { return count; }
    int32_t node_count() const { return count; }
    bool empty() const { return count == 0; }
    bool full() const { return count >= kTreeMaxNodes; }

    const TreeNode& operator[](int32_t i) const { return nodes[i]; }
    TreeNode& operator[](int32_t i) { return nodes[i]; }

    /* Appends one node. Its id is `count`, so a caller must add strictly in topological order.
     * `parent` -1 is the root and is only legal as the first node; every other parent must
     * already be present. Returns false on capacity, ordering or a second root. */
    bool add(int32_t parent, uint32_t token) {
        if (count >= kTreeMaxNodes) return false;
        TreeNode n;
        n.parent = parent;
        n.token  = token;
        if (parent < 0) {
            if (count != 0) return false;
            n.depth = 0;
            n.ancestors.set(0);
        } else {
            if (count == 0 || parent >= count) return false;
            n.depth = nodes[parent].depth + 1;
            if (n.depth >= kTreeMaxNodes) return false;
            n.ancestors = nodes[parent].ancestors;
            n.ancestors.set(static_cast<size_t>(count));
        }
        nodes[count] = n;
        ++count;
        return true;
    }

    /* A chain: node i's parent is i-1. The linear topology a serial drafter emits, so a tree
     * batch and a chain batch have one construction path. `n_draft` past the capacity is
     * truncated to it (unlike build_mtp_fixed, which refuses a budget it cannot fill). */
    bool build_linear(uint32_t root, const uint32_t* draft, int32_t n_draft) {
        reset();
        if (!add(-1, root)) return false;
        const int32_t limit = kTreeMaxNodes - 1;
        const int32_t n = n_draft < 0 ? 0 : (n_draft > limit ? limit : n_draft);
        for (int32_t i = 0; i < n; ++i) {
            const uint32_t tok = draft ? draft[i] : root + static_cast<uint32_t>(i) + 1;
            if (!add(i, tok)) return false;
        }
        return true;
    }

    /* Top-2 topology, at most 16 nodes; `best_path` picks the archived Mode B (branch at depths 1
     * and 2, linear after) over Mode A (breadth branching throughout). */
    bool build_top2(uint32_t root, const uint32_t* cand, int32_t n_cand, bool best_path = false) {
        static const int32_t kParentsA[16] = {-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7};
        static const int32_t kParentsB[16] = {-1, 0, 0, 1, 1, 2, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14};
        const int32_t* parents = best_path ? kParentsB : kParentsA;
        reset();
        if (!add(-1, root)) return false;
        const int32_t n = n_cand < 0 ? 0 : (n_cand > 15 ? 15 : n_cand);
        for (int32_t i = 1; i <= n; ++i) {
            const uint32_t tok = cand ? cand[i - 1] : root * 31u + static_cast<uint32_t>(i) * 7u;
            if (!add(parents[i], tok)) return false;
        }
        return true;
    }

    /* Top-4 topology, at most 16 nodes: four children of the root, then their children. */
    bool build_top4(uint32_t root, const uint32_t* cand, int32_t n_cand) {
        static const int32_t kParents[16] = {-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
        reset();
        if (!add(-1, root)) return false;
        const int32_t n = n_cand < 0 ? 0 : (n_cand > 15 ? 15 : n_cand);
        for (int32_t i = 1; i <= n; ++i) {
            const uint32_t tok = cand ? cand[i - 1] : root * 17u + static_cast<uint32_t>(i) * 3u;
            if (!add(kParents[i], tok)) return false;
        }
        return true;
    }

    /* The fixed MTP profile: one spine row per draft round, then each round's alternate top-k as
     * siblings of that round's spine row, deduplicated against the tokens the round's parent
     * already carries. `history[r][k]` is row r's top-k token ids, column 0 the spine; a negative
     * entry ends the row. Ported unchanged from the archived host reference
     * (ref_mtp_tree_accum.cpp), which is what the archived MTP8/64 tree packed.
     *
     * `rounds` is 3 (capacity <= 16) or 8 (capacity <= 65); `capacity` is the node budget, root
     * included. A root or history entry at or above `vocab` empties the tree rather than building
     * a row the sampler could never index. */
    bool build_mtp_fixed(const int32_t (*history)[kTreeRoundCandidates], int32_t rounds,
                         int32_t capacity, int32_t root, int32_t vocab) {
        reset();
        if (!history || (rounds != 3 && rounds != 8)) return false;
        if (capacity < 2 || capacity > (rounds == 3 ? 16 : kTreeMaxNodes)) return false;
        if (root < 0 || (vocab > 0 && root >= vocab)) return false;
        for (int32_t r = 0; r < rounds; ++r) {
            for (int32_t k = 0; k < kTreeRoundCandidates && history[r][k] >= 0; ++k)
                if (vocab > 0 && history[r][k] >= vocab) return false;
        }

        int32_t tokens[kTreeMaxNodes];
        int32_t parents[kTreeMaxNodes];
        tokens[0] = root;
        parents[0] = -1;
        int32_t n = 1, spine = 0;
        /* Pass one: the spine, one row per round. */
        for (int32_t r = 0; r < rounds && n < capacity; ++r) {
            if (history[r][0] < 0) break;
            tokens[n] = history[r][0];
            parents[n] = n - 1;
            ++n;
            ++spine;
        }
        /* Pass two: the alternates, each a sibling of its round's spine row. A token the parent
         * already carries is not added twice: a duplicate sibling would be a second row with the
         * same token and the same ancestors, which acceptance can never distinguish. */
        for (int32_t r = 0; r < spine && n < capacity; ++r) {
            for (int32_t k = 1; k < kTreeRoundCandidates && n < capacity; ++k) {
                const int32_t alt = history[r][k];
                if (alt < 0) break;
                bool dup = (alt == history[r][0]);
                for (int32_t j = r + 1; j < n && !dup; ++j)
                    if (parents[j] == r && tokens[j] == alt) dup = true;
                if (dup) continue;
                tokens[n] = alt;
                parents[n] = r;
                ++n;
            }
        }
        reset();
        for (int32_t i = 0; i < n; ++i)
            if (!add(parents[i], static_cast<uint32_t>(tokens[i]))) return false;
        return true;
    }

    /* The RoPE/KV position of a node's token. */
    int32_t position(int32_t node, int32_t root_pos) const {
        if (node < 0 || node >= count) return root_pos;
        return root_pos + nodes[node].depth;
    }

    /* Every stored mask equals the set its parent's mask implies. A mask that disagrees conditions
     * a row on a token that is not on its path, which no tolerance can make right. */
    bool masks_consistent() const {
        if (count <= 0) return false;
        TreeMask expect;
        expect.set(0);
        if (nodes[0].ancestors != expect) return false;
        if (nodes[0].depth != 0) return false;   /* the root is the only depth-0 row */
        for (int32_t i = 1; i < count; ++i) {
            const int32_t p = nodes[i].parent;
            if (p < 0 || p >= i) return false;
            expect = nodes[p].ancestors;
            expect.set(static_cast<size_t>(i));
            if (nodes[i].ancestors != expect) return false;
            if (nodes[i].depth != nodes[p].depth + 1) return false;
        }
        return true;
    }

    /* Shape and capacity checks a caller runs before dispatch: ids in range, `token_span` rows
     * available past the root position, tokens below `vocab` (0 skips the vocabulary check). */
    bool fits(int32_t vocab, int32_t token_span) const {
        if (count <= 0 || count > kTreeMaxNodes) return false;
        for (int32_t i = 0; i < count; ++i) {
            if (nodes[i].parent >= i) return false;
            if (vocab > 0 && nodes[i].token >= static_cast<uint32_t>(vocab)) return false;
            if (nodes[i].depth >= token_span) return false;
        }
        return masks_consistent();
    }

    TreeNode nodes[kTreeMaxNodes];
    int32_t  count = 0;
};

/* The winner of a tree verify: the path acceptance walked and the target's own token past it.
 * `path[0..accepted]` are node ids (path[0] is the root); `tokens[0..accepted-1]` are the accepted
 * draft tokens; `bonus` is the target's pick at the last accepted node. */
struct TreeWinner {
    int32_t  path[kTreeMaxNodes]{};    /* node ids, [0, accepted] */
    uint32_t tokens[kTreeMaxNodes]{};  /* accepted draft tokens, [0, accepted) */
    int32_t  accepted = 0;
    uint32_t bonus    = 0;
    bool     has_bonus = false;
};

/* Acceptance by agreement over a tree, at every temperature. `pick[j]` is the target's own token
 * at node j's row -- its argmax at temperature 0, its sampled draw above it (see accept.h).
 *
 * At each visited node the walk takes the lowest-id child whose token equals the pick AT THAT
 * NODE, and continues from it; the first node with no such child ends the walk, and its pick is
 * the bonus token. Siblings other than the match get no second chance, and the lowest id wins a
 * tie between children carrying the same token, so the result is deterministic in a batch whose
 * row order is deterministic.
 *
 * `n_pick` must cover every row the walk can visit, which is the tree's node count: the bonus is
 * the pick at the last visited node, so a shorter buffer leaves the walk without the token the
 * step must emit. Returns RAD_OK and fills `*out`; RAD_E_INVAL on a malformed tree (bad masks or
 * topology, short buffer); RAD_E_STATE never occurs as written -- `node` grows strictly and leaves
 * have no children, so the walk always ends with a bonus -- and the clause is kept only so a future
 * tree shape cannot emit token 0. */
inline int verify_tree_greedy(const CandidateTree& tree, const int32_t* pick, int32_t n_pick,
                              TreeWinner* out) {
    if (!out || tree.count <= 0 || tree.count > kTreeMaxNodes) return RAD_E_INVAL;
    if (!pick || n_pick < tree.count) return RAD_E_INVAL;
    if (!tree.masks_consistent()) return RAD_E_INVAL;
    const int32_t n = tree.count;

    int32_t node = 0, acc = 0;
    out->path[0] = 0;
    while (node < n) {
        const int32_t greedy = pick[node];
        if (greedy < 0) return RAD_E_INVAL;   /* a row without a pick has no token to match */
        int32_t match = -1;
        for (int32_t i = node + 1; i < n; ++i) {
            if (tree.nodes[i].parent == node &&
                tree.nodes[i].token == static_cast<uint32_t>(greedy)) {
                match = i;
                break;
            }
        }
        if (match < 0) {
            out->bonus = static_cast<uint32_t>(greedy);
            out->has_bonus = true;
            break;
        }
        out->tokens[acc] = tree.nodes[match].token;
        out->path[acc + 1] = match;
        ++acc;
        node = match;
    }
    out->accepted = acc;
    return out->has_bonus ? RAD_OK : RAD_E_STATE;
}

}  /* namespace rad */
