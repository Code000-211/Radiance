/* tree_dispatch.h -- the host's dispatch contract for verifying a candidate tree.
 *
 * A tree verify replaces the linear chain a drafter currently emits (batch.h's spec_is_tree).
 * This module is the host side of that step and nothing else: it decides whether a tree may be
 * dispatched at all, what the batch has to stage for it, and which prefix the step commits. It
 * does not issue the attention or the state kernels -- the ancestors it hands out are only sound
 * if the attention kernel masks each row to its own ancestor set (see the "NOT WIRED" note below).
 *
 * ============================== THE CONTRACT ==============================
 *
 * Temperature. The tree accepts the target's own pick at each row and walks matching children,
 * exactly as accept_tree_greedy does on a chain (core/sample/accept.h). That is exact at
 * temperature 0 and not above it: a sampled row's pick is not the mode the walk would match. There
 * is no approximate tree sampling in this build, so `temperature_zero == false` is refused with
 * RAD_E_UNSUPPORTED and the caller keeps the serial path untouched. The refusal is the contract,
 * not an unimplemented corner.
 *
 * Switch. Tree dispatch runs only behind RADIANCE_MTP_COUPLED=1 (rad_spechub_config.h). With the
 * switch off or unset, `coupled == false` is refused before anything is written, so a caller that
 * gets the switch wrong cannot leave a half-built tree behind and the serial path stays
 * bit-identical with no tree allocation.
 *
 * Masks. The only ancestor form the card can read today is the legacy uint16 mask
 * (`tree_mask16`, exact for trees of at most kTreeLegacyMaskNodes rows). A wider tree is refused
 * with RAD_E_UNSUPPORTED rather than truncated: a mask that lost its high bits conditions a row on
 * a token that is not on its path. Callers that later stage a full-width mask pass mask_bits = 0.
 *
 * Capacity. The fixed MTP profile (`build_mtp_fixed`) truncates to a budget it cannot fill. A
 * truncated tree is a different tree, so tree_dispatch_fixed rebuilds it at the full budget and
 * refuses any node-count or geometry difference.
 *
 * ============================== NOT WIRED ==============================
 *
 * No attention kernel in this tree takes an ancestor mask: `attn_paged` (libr4d, libref) is causal
 * over positions, so a row of a branching tree would see its siblings. No GDN/QSA/PLE state
 * update can discard a losing branch either; the linear state absorbs every verified row in row
 * order (accept.h's header says why a rejection only moves a read offset there). This module
 * produces the parents, positions and masks such a step needs and the commit it must produce; the
 * dispatch that would consume them is refused by the engine, because a tree run against the
 * current kernels would be a silent miscomputation.
 */
#pragma once

#include "../../arch/common/rad_tree_candidate.h"
#include "../sample/accept.h"

#include <cstdint>
#include <string>

namespace rad {

/* One admitted tree: the geometry the batch stages, plus the checks its caller ran. */
struct TreeStep {
    CandidateTree tree;
    int32_t  root_pos = 0;                  /* position of the root (the last committed token) */
    int32_t  n_nodes  = 0;                 /* tree.node_count(), repeated for cheap range checks */
    int32_t  parents[kTreeMaxNodes]{};     /* node j's parent, -1 at the root */
    int32_t  positions[kTreeMaxNodes]{};   /* root_pos + depth */
    uint16_t masks16[kTreeMaxNodes]{};     /* the uint16 ancestor set, 0 where none exists */
    bool     has_mask16 = false;           /* every node's set has a uint16 form */
};

/* What the step commits: the winner path, the emitted tokens and the branches it drops. */
struct TreeCommit {
    int32_t  tokens[kTreeMaxNodes]{};   /* [0..n_accepted]: accepted drafts, then the bonus */
    int32_t  path[kTreeMaxNodes]{};     /* node ids, [0..n_accepted]; path[0] is the root */
    int32_t  n_accepted = 0;            /* accepted drafts after the end-of-generation cut */
    int32_t  n_discarded = 0;           /* nodes that are not on the committed path */
    uint32_t bonus = 0;                 /* the target's own token past the last accepted node */
};

inline void tree_step_reset(TreeStep* s) {
    if (s) *s = TreeStep{};
}

/* Admits an already built tree. Checks, in the order the caller can fail loudly on them:
 * the switch, the temperature contract, the mask width, capacity/topology (masks_consistent,
 * fits over the token span and the vocabulary) and the position arithmetic the batch stages.
 * Writes nothing to `out` on a refusal, so a refused dispatch leaves no staged state behind. */
inline int tree_dispatch_admit(const CandidateTree& tree, int32_t root_pos, int32_t vocab,
                               int32_t token_span, bool temperature_zero, bool coupled,
                               int32_t mask_bits, TreeStep* out, std::string* why) {
    auto refuse = [&](int code, const char* msg) {
        if (why) *why = msg;
        return code;
    };
    if (!out) return refuse(RAD_E_INVAL, "tree dispatch: no output for the admitted step");
    if (!coupled)
        return refuse(RAD_E_STATE,
                      "tree dispatch: RADIANCE_MTP_COUPLED is off; the tree runs only behind the "
                      "switch, and the serial path must not be replaced");
    if (!temperature_zero)
        return refuse(RAD_E_UNSUPPORTED,
                      "tree dispatch: the tree accepts at temperature 0 only; sampling a tree is "
                      "not implemented and the serial path must run unchanged");
    if (mask_bits != 0 && mask_bits != 16)
        return refuse(RAD_E_INVAL, "tree dispatch: an ancestor mask is 16 bits wide or full");
    if (!tree.masks_consistent())
        return refuse(RAD_E_INVAL,
                      "tree dispatch: the tree's ancestor masks disagree with its topology");
    if (!tree.fits(vocab, token_span))
        return refuse(RAD_E_INVAL,
                      "tree dispatch: the tree does not fit the vocabulary and the token span");
    if (tree.node_count() < 2)
        return refuse(RAD_E_INVAL, "tree dispatch: a tree of the root alone verifies nothing");
    if (root_pos < 0)
        return refuse(RAD_E_INVAL, "tree dispatch: the root position is before the context");
    if (mask_bits == 16) {
        for (int32_t j = 0; j < tree.node_count(); ++j)
            if (tree_mask16(tree[j].ancestors) == 0 && (tree[j].ancestors >> kTreeLegacyMaskNodes).any())
                return refuse(RAD_E_UNSUPPORTED,
                              "tree dispatch: a node's ancestor set has no uint16 form; a wider "
                              "tree needs a full-width mask consumer, not a truncated one");
    }

    out->tree      = tree;
    out->root_pos  = root_pos;
    out->n_nodes   = tree.node_count();
    out->has_mask16 = true;
    for (int32_t j = 0; j < out->n_nodes; ++j) {
        out->parents[j]   = tree[j].parent;
        out->positions[j] = tree.position(j, root_pos);
        out->masks16[j]   = tree_mask16(tree[j].ancestors);
        if (out->masks16[j] == 0) out->has_mask16 = false;
    }
    return RAD_OK;
}

/* The fixed MTP profile at a fixed budget: build, then admit. The full-budget tree is built a
 * second time and compared, so a budget that cuts the profile off is refused instead of silently
 * verifying a smaller tree than the head proposed. */
inline int tree_dispatch_fixed(const int32_t (*history)[kTreeRoundCandidates], int32_t rounds,
                               int32_t capacity, int32_t root, int32_t vocab, int32_t root_pos,
                               int32_t token_span, bool temperature_zero, bool coupled,
                               int32_t mask_bits, TreeStep* out, std::string* why) {
    auto refuse = [&](int code, const char* msg) {
        if (why) *why = msg;
        return code;
    };
    if (!out) return refuse(RAD_E_INVAL, "tree dispatch: no output for the admitted step");
    if (!coupled)
        return refuse(RAD_E_STATE,
                      "tree dispatch: RADIANCE_MTP_COUPLED is off; the tree runs only behind the "
                      "switch, and the serial path must not be replaced");
    if (!temperature_zero)
        return refuse(RAD_E_UNSUPPORTED,
                      "tree dispatch: the tree accepts at temperature 0 only; sampling a tree is "
                      "not implemented and the serial path must run unchanged");
    const int32_t budget = rounds == 3 ? 16 : kTreeMaxNodes;
    CandidateTree tree;
    if (!tree.build_mtp_fixed(history, rounds, capacity, root, vocab))
        return refuse(RAD_E_INVAL,
                      "tree dispatch: the fixed MTP profile refused its rounds, budget, root or "
                      "shortlist");
    CandidateTree full;
    if (!full.build_mtp_fixed(history, rounds, budget, root, vocab))
        return refuse(RAD_E_INVAL, "tree dispatch: the fixed MTP profile refused its full budget");
    bool same = tree.node_count() == full.node_count();
    for (int32_t j = 0; same && j < tree.node_count(); ++j)
        same = tree[j].parent == full[j].parent && tree[j].token == full[j].token;
    if (!same)
        return refuse(RAD_E_INVAL,
                      "tree dispatch: the budget truncates the fixed MTP profile; a cut tree is "
                      "not the tree the head proposed");
    return tree_dispatch_admit(tree, root_pos, vocab, token_span, temperature_zero, coupled,
                               mask_bits, out, why);
}

/* The commit a verified tree produces: accept_tree_greedy's walk, then collect_step's
 * end-of-generation cut. `pick[j]` is the target's own token at node j's row and must cover every
 * node. `n_discarded` counts the nodes the step drops -- the branches that lost, whose KV and
 * linear state must not survive the commit. Writes nothing to `out` on a refusal. */
template <class IsEog>
inline int tree_dispatch_commit(const TreeStep& s, const int32_t* pick, int32_t n_pick,
                                IsEog&& is_eog, TreeCommit* out, std::string* why) {
    auto refuse = [&](int code, const char* msg) {
        if (why) *why = msg;
        return code;
    };
    if (!out) return refuse(RAD_E_INVAL, "tree commit: no output for the committed step");
    if (s.n_nodes != s.tree.node_count() || s.n_nodes < 2 || s.n_nodes > kTreeMaxNodes)
        return refuse(RAD_E_INVAL, "tree commit: the admitted tree and its node count disagree");
    if (!s.tree.masks_consistent())
        return refuse(RAD_E_INVAL, "tree commit: the tree's ancestor masks disagree with its topology");
    if (!pick)
        return refuse(RAD_E_INVAL, "tree commit: no picks; every row the walk visits needs one");
    if (n_pick < s.n_nodes)
        return refuse(RAD_E_INVAL,
                      "tree commit: the pick buffer is shorter than the tree, so a visited row "
                      "would have no token to match");
    for (int32_t j = 0; j < s.n_nodes; ++j) {
        if (s.parents[j] != s.tree[j].parent || s.positions[j] != s.tree.position(j, s.root_pos))
            return refuse(RAD_E_INVAL, "tree commit: the staged layout and the tree disagree");
        if (pick[j] < 0)
            return refuse(RAD_E_INVAL, "tree commit: a row without a pick has no token to match");
    }

    TreeWinner w;
    const int rc = verify_tree_greedy(s.tree, pick, n_pick, &w);
    if (rc != RAD_OK) return refuse(rc, "tree commit: the winner walk refused the tree");

    int32_t emitted[kTreeMaxNodes]{};
    for (int32_t j = 0; j < w.accepted; ++j) emitted[j] = static_cast<int32_t>(w.tokens[j]);
    emitted[w.accepted] = static_cast<int32_t>(w.bonus);
    int32_t acc = w.accepted;
    for (int32_t j = 0; j <= acc; ++j)
        if (is_eog(emitted[j])) { acc = j; break; }

    /* The walk's own path must be a parent chain ending at the node whose pick is the bonus. A
     * path that is not one means the acceptance rule and the topology disagree, which no commit
     * can repair. */
    if (w.path[0] != 0) return refuse(RAD_E_STATE, "tree commit: the winner path does not start at the root");
    for (int32_t j = 1; j <= acc; ++j) {
        if (w.path[j] <= w.path[j - 1] || s.tree[w.path[j]].parent != w.path[j - 1])
            return refuse(RAD_E_STATE, "tree commit: the winner path is not a chain in the tree");
        if (s.tree[w.path[j]].token != static_cast<uint32_t>(emitted[j - 1]))
            return refuse(RAD_E_STATE, "tree commit: an accepted node's token is not its pick");
    }

    for (int32_t j = 0; j <= acc; ++j) out->tokens[j] = emitted[j];
    for (int32_t j = 0; j <= acc; ++j) out->path[j] = w.path[j];
    out->n_accepted  = acc;
    out->bonus       = static_cast<uint32_t>(emitted[acc]);
    out->n_discarded = s.n_nodes - (acc + 1);
    return RAD_OK;
}

/* The host's commit against the card's acceptance. The card walks the same tree in
 * advance_impl.h (ADV_TREE_ACCEPT) from the same staged parents and picks; a disagreement means
 * one of the two read different rows, and the step is refused rather than emitted. */
inline int tree_dispatch_confirm(const TreeCommit& c, int32_t card_accepted, int32_t card_token,
                                 std::string* why) {
    if (card_accepted != c.n_accepted || card_token != static_cast<int32_t>(c.bonus)) {
        if (why) {
            *why = "tree commit: the card accepted " + std::to_string(card_accepted) +
                   " node(s) ending in token " + std::to_string(card_token) +
                   " and the host " + std::to_string(c.n_accepted) + " ending in " +
                   std::to_string(c.bonus);
        }
        return RAD_E_STATE;
    }
    return RAD_OK;
}

}  /* namespace rad */
