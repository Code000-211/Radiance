/* rad_fulldrafter_tree.h -- root-fork candidate topology over the intact FullDrafter greedy spine.
 *
 * Wave-1 foundations only. This builds the safest tree the FullDrafter can propose today: the
 * 12-token discrete two-stage greedy chain stays exactly as the linear drafter emitted it, and
 * extra candidates fork ONLY at the root, where every top-16 entry is genuinely conditioned on
 * the same committed anchor. Deeper forks need per-parent Markov/selector conditioning that does
 * not exist yet, and stage-2 rows hang off the original greedy T6 path because that is the only
 * base representation the drafter computed. Nothing here claims the teacher can verify any
 * particular node count: node_budget is the caller's statement of what this step's verification
 * actually supports, and this builder never widens it.
 *
 * Host-only, allocation-free, fixed-size locals only: this is step-path code. Every input
 * byte is read before the first write to *out, so a caller may publish the chain and the
 * top-16 out of one overlapping scratch buffer.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

/* Same directory, quoted: resolves relative to this file however it was included, which is
 * why the fallback chain the heavier fulldrafter header needs is unnecessary here. */
#include "rad_tree_candidate.h"

/* RAD_OK / RAD_E_*: errors are values, never exceptions. Relative to this file, which is
 * how the plugin and test builds include it; a missing file is a loud compile error. */
#include "../../abi/rad_types.h"

/* The approved top-16 ordering (finite scores descending, ties to the lower token id, sentinel
 * ids always losing) is reused verbatim from the reviewed selection primitive rather than
 * re-implemented: a second ordering that could drift from it is exactly the kind of quiet fork
 * this file must not introduce. */
#include "../../libr4d/r4d_fulldrafter_select.h"

namespace radiance {

/* THE ROOT FORK'S OWN BUDGET, root included: the 16-wide verification it was built for, not the
 * host tree capacity (RAD_TREE_MAX_NODES), which grew for wider drafters and does not widen this. */
constexpr size_t FULLDRAFTER_TREE_MAX_NODES = 16;
static_assert(FULLDRAFTER_TREE_MAX_NODES <= RAD_TREE_MAX_NODES, "the root fork must fit a CandidateTree");

/* Explicit inputs for one sequence's root-fork tree. No defaults: every field is a decision the
 * caller (the engine lane) owns and states out loud. */
struct FulldrafterRootFork {
    uint32_t        anchor_token;    /* committed anchor: becomes root node 0 */
    const uint32_t* greedy_chain;    /* [chain_len] the drafter's greedy winners, spine order
                                      * (stage 1 slots 0..5 then stage 2 slots 6..11 for the
                                      * 2x6 drafter; chain[BLOCK_K-1] is the T6 boundary) */
    size_t          chain_len;       /* 1..15, the full two-stage chain (12 for FullDrafter) */
    const uint32_t* root_cand_ids;   /* [n_cand] slot-0 top-16 candidate token ids */
    const float*    root_cand_scores;/* [n_cand] reranked selector scores, parallel to ids */
    size_t          n_cand;          /* 0..16 valid entries; entries at or past this are ignored */
    size_t          alt_count;       /* alternate root children to add (0 = pure linear spine) */
    size_t          node_budget;     /* total node cap INCLUDING the root; 1..FULLDRAFTER_TREE_MAX_NODES */
};

/* Builds the tree into the caller-owned CandidateTree.
 *
 * Topology (spine-first node order):
 *   node 0                root, anchor token, depth 0
 *   nodes 1..chain_len    the greedy spine, verbatim: token chain[i-1], parent i-1, depth i,
 *                         score 1.0 (the builder receives no spine scores; 1.0 matches the
 *                         linear builder's convention and verification never reads .score)
 *   nodes chain_len+1..   chain_len+alt_count alternate children of the root ONLY: depth 1,
 *                         leaves, carrying their slot-0 selector score
 *
 * Candidate selection: ids equal to r4d::FULLDRAFTER_SENTINEL_ID are dropped deterministically
 * (unfilled top-16 slots); duplicate ids collapse to their best entry; the first greedy token is
 * already the spine head (node 1, a child of the root) and never duplicated as an alternate.
 * Remaining candidates are taken in the approved top-16 order, so an already-reranked input list
 * is reproduced exactly and an unordered one is sorted. Candidates whose id equals a DEEPER
 * spine token are allowed: they are conditioned on the anchor, not on their spine namesake, so
 * they are a different hypothesis at a different node, not a duplicate.
 *
 * Errors (RAD_E_INVAL) on: null out, null chain / zero-length chain (a request with no draft
 * builds no tree), chain longer than 15 or than the budget leaves room for, budget of 0 or above
 * FULLDRAFTER_TREE_MAX_NODES, n_cand above 16, null candidate arrays with n_cand > 0, alt_count the
 * budget cannot hold, alt_count the deduplicated candidates cannot fill, or a non-finite score.
 * On RAD_E_INVAL *out is left untouched (no partial output); the caller resets or reuses at its
 * own pace. Parents are derived, never read from input, so an invalid parent cannot arrive; the
 * add_node calls are still checked and a failure (a broken invariant, not a caller error) resets
 * *out and returns RAD_E_STATE.
 */
inline int fulldrafter_build_root_fork(const FulldrafterRootFork& r, CandidateTree* out) {
    if (!out) return RAD_E_INVAL;
    if (r.node_budget == 0 || r.node_budget > FULLDRAFTER_TREE_MAX_NODES) return RAD_E_INVAL;
    if (!r.greedy_chain || r.chain_len == 0) return RAD_E_INVAL;
    if (r.chain_len > FULLDRAFTER_TREE_MAX_NODES - 1) return RAD_E_INVAL;
    if (r.chain_len + 1 > r.node_budget) return RAD_E_INVAL;
    if (r.n_cand > FULLDRAFTER_TREE_MAX_NODES) return RAD_E_INVAL;
    if (r.n_cand > 0 && (!r.root_cand_ids || !r.root_cand_scores)) return RAD_E_INVAL;
    if (r.n_cand == 0 && r.alt_count > 0) return RAD_E_INVAL;
    if (r.alt_count > r.node_budget - 1 - r.chain_len) return RAD_E_INVAL;

    /* Read everything first, write *out last: inputs may alias *out and each other. */
    uint32_t chain[FULLDRAFTER_TREE_MAX_NODES - 1];
    for (size_t i = 0; i < r.chain_len; ++i) chain[i] = r.greedy_chain[i];

    for (size_t i = 0; i < r.n_cand; ++i) {
        if (!std::isfinite(r.root_cand_scores[i])) return RAD_E_INVAL;
    }

    r4d::FullDrafterThreadTop16 rank;
    rank.init();
    for (size_t i = 0; i < r.n_cand; ++i) {
        const uint32_t id = r.root_cand_ids[i];
        if (id == r4d::FULLDRAFTER_SENTINEL_ID) continue; /* unfilled slot, dropped */
        if (id == chain[0]) continue;                     /* spine head owns this token already */
        rank.insert(r.root_cand_scores[i], id);
    }

    uint32_t alt_id[FULLDRAFTER_TREE_MAX_NODES - 1];
    float    alt_score[FULLDRAFTER_TREE_MAX_NODES - 1];
    size_t   got = 0;
    for (int k = 0; k < r4d::FullDrafterThreadTop16::K && got < r.alt_count; ++k) {
        const uint32_t id = rank.id[k];
        if (id == r4d::FULLDRAFTER_SENTINEL_ID) break; /* sorted: the rest are empty too */
        bool dup = false;
        for (size_t j = 0; j < got; ++j) {
            if (alt_id[j] == id) { dup = true; break; } /* collapse to the best entry */
        }
        if (dup) continue;
        alt_id[got]   = id;
        alt_score[got] = rank.val[k];
        ++got;
    }
    if (got < r.alt_count) return RAD_E_INVAL; /* refuse; silently shrinking is a policy */

    out->reset();
    if (!out->add_node(0, -1, r.anchor_token, 0, 1.0f)) {
        out->reset();
        return RAD_E_STATE;
    }
    for (size_t i = 0; i < r.chain_len; ++i) {
        if (!out->add_node(static_cast<int32_t>(i + 1), static_cast<int32_t>(i), chain[i],
                           static_cast<int32_t>(i + 1), 1.0f)) {
            out->reset();
            return RAD_E_STATE;
        }
    }
    for (size_t a = 0; a < got; ++a) {
        if (!out->add_node(static_cast<int32_t>(r.chain_len + 1 + a), 0, alt_id[a], 1,
                           alt_score[a])) {
            out->reset();
            return RAD_E_STATE;
        }
    }
    return RAD_OK;
}

} /* namespace radiance */
