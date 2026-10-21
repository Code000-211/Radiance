#ifndef RAD_MINE_DFLASH_TREE_SELECT_H
#define RAD_MINE_DFLASH_TREE_SELECT_H

#include "rad_plugin.h"
#include <stdint.h>

/* dflash_tree_select -- DFlash2's candidate walk, widened into a bounded, pruned tree.
 *
 * THE SCORING IS THE SHIPPING SELECTOR'S. For one (parent, candidate) pair the edge is
 *
 *     unary[l][c] + sum_r pred[pid][r] * hp[l][r] * succ[cand[l][c]][r],   pid(0) = anchor,
 *
 * with `hp` absent meaning a plane of ones. The reduction is reproduced operation for operation
 * from libr4d/r4d_dflash_select_bf16.hip: a 32-lane accumulation over `r = lane, lane+32, ...`
 * seeded at zero, the xor-butterfly fold (16,8,4,2,1), and `unary` added last. Same order, same
 * FP32, `-ffp-contract=off` on both sides, so for the geometry the shipping row accepts
 * (`rank` a multiple of 32) the linear walk is bit-identical, not merely formula-identical. This
 * op also accepts `rank` that is not a multiple of 32 -- the fold is the natural extension and
 * nothing here depends on the multiple -- but parity with libr4d is only claimed where libr4d
 * runs at all. `top_k` is capped at 16 here rather than libr4d's 32, because a tree's frontier is
 * sized by the verifier's 16 rows.
 *
 * THE TREE HEURISTIC IS ONE SUBTRACTION. A chain compares candidates within one parent; a tree
 * must also compare ACROSS parents, and raw logits from different parents are not on one scale.
 * Each parent's candidate scores are therefore shifted by that parent's own maximum, so every
 * parent's best sits at zero and children carry their parent's accumulated score. The shift is
 * monotone inside a parent, so it changes no within-parent ordering; across parents it is a
 * normalisation by local maximum, not a probability. Pruning then keeps the `tree_budget` best
 * nodes by `score + depth_penalty * depth`, and never keeps a node whose parent it dropped.
 *
 * A subtraction rather than a log-softmax on purpose: `logf`/`expf` need not agree bit for bit
 * between the device and the host oracle, and this op's comparison is a byte comparison. Dropping
 * the log-sum-exp term removes a per-parent constant from a heuristic that is not a model of
 * anything, and keeps both sides exactly comparable.
 *
 * THREE DELIBERATE DIFFERENCES FROM `dflash_select`, all about ids:
 *
 *   * a candidate outside `[0, n_vocab)` is not offered. The chain may pick such a slot on its
 *     unary term alone because it emits an INDEX; a tree proposes a TOKEN the target must have.
 *   * an anchor outside `[0, n_vocab)` writes `count = 0`. The chain scores it as a predecessor
 *     with no codebook contribution; a tree cannot have an unverifiable root.
 *   * a REPEATED candidate id under one parent is deduplicated to a single node, keeping the
 *     instance with the highest raw edge score and, on an exact tie, the lower candidate index.
 *     The scheduler refuses a tree whose siblings propose the same token
 *     (`core/sched/scheduler.cpp`, `set_draft_tree`), and keeping the best-scoring instance is the
 *     linear walk's own argmax rule, so a `tree_width = 1` pick is unchanged by the rule.
 *
 * ============================== OUTPUT ==============================
 *
 * `packed_tree` is dloop_dartree's root-inclusive packing, `[M, 1 + 2*capacity]` I32, which is what
 * `mine/dloop_tree.hip` writes and `dloop_tree_from_packed` reads. The engine declares the drafter
 * tree with `capacity = 16` node slots, so a row is 33 columns:
 *
 *     col 0                  count = 1 + emitted draft nodes   (0 when the row is invalid)
 *     col 1 + i              node i's token, i = 0 (THE ANCHOR) .. count-1
 *     col 1 + capacity + i   node i's root-inclusive parent, the same range: 0 = the anchor
 *     col 1 + capacity       the anchor's parent: the -1 SENTINEL the validator requires
 *
 * At capacity 16 the parent block is columns 17..32, so the sentinel sits at 17 and the first
 * draft node's parent at 18. Writing the pitch as `1 + capacity` rather than as a literal is what
 * keeps this correct for a capacity other than 16 -- and `capacity` is the declared tree size, NOT
 * the selector's `tree_budget`. Unused slots are -1 in both blocks. Nodes are emitted in old-id
 * order, and an old id is assigned when its level is appended, so `parent(i) < i` holds for every
 * node after the root. Siblings of one parent appear by descending score, which puts that parent's
 * greedy child first -- the property `dloop_greedy_chain` reads.
 *
 * A row whose per-row data cannot produce a tree (an anchor outside the vocabulary, a non-finite
 * unary or edge on a live candidate) writes `count = 0`. That is the contract, not a convenience: a
 * device launch cannot be failed once issued, and the host oracle writes the same zero.
 *
 * ============================== BOUNDS ==============================
 *
 * `steps` is 1..3, `tree_budget` 1..15, `top_k` 1..16, `tree_width` 1..top_k. A call outside them is
 * refused rather than clamped: more than 15 draft nodes does not fit the verifier beside the
 * anchor, and more than 3 levels is past the depth the drafter was trained for. Growth is linear in
 * the level count (`1 + steps * tree_width <= 49` nodes), never a global beam. */
enum {
    RAD_DFLASH_TREE_CAND = 0,
    RAD_DFLASH_TREE_UNARY = 1,
    RAD_DFLASH_TREE_HP = 2,
    RAD_DFLASH_TREE_ANCHOR = 3,
    RAD_DFLASH_TREE_PRED = 4,
    RAD_DFLASH_TREE_SUCC = 5,
    RAD_DFLASH_TREE_OUT = 6,
    RAD_DFLASH_TREE_OPERANDS = 7
};

/* The declared limits. They are the verifier's, not a tuning knob: 15 draft nodes is the largest
 * tree that fits 16 verification rows beside the anchor. */
enum { RAD_DFLASH_TREE_MAX_STEPS = 3, RAD_DFLASH_TREE_MAX_K = 16,
       RAD_DFLASH_TREE_MAX_BUDGET = 15, RAD_DFLASH_TREE_MAX_WIDTH = RAD_DFLASH_TREE_MAX_K,
       RAD_DFLASH_TREE_MAX_NODES = 1 + RAD_DFLASH_TREE_MAX_STEPS * RAD_DFLASH_TREE_MAX_WIDTH,
       RAD_DFLASH_TREE_OUT_COLS = 33 };

/* Validated, device-free view of one call. Nothing here dereferences a pointer: at issue every
 * operand is device memory, and the values inside them are checked per row by the kernel and the
 * oracle, not here. */
struct RadDflashTreeSelectCall {
    const void* cand;
    const void* unary;
    const void* hp;          /* null when absent: a plane of ones */
    const void* anchor;
    const void* pred;
    const void* succ;
    void*       out;
    int64_t     M;           /* the output's rows: the sequences drafting in this pass */
    int64_t     rows;        /* M * steps: the candidate planes' rows, s*steps + level - 1 */
    int64_t     steps;
    int64_t     K;
    int64_t     R;
    int64_t     V;
    int64_t     anchor_stride;
    int64_t     budget;
    int64_t     width;
    float       beta;        /* depth penalty, <= 0 */
    int64_t     cand_ld;
    int64_t     unary_ld;
    int64_t     hp_ld;
    int64_t     pred_ld;
    int64_t     succ_ld;
    int64_t     out_ld;
};

int                 rad_dflash_tree_select_schema_count(void);
const RadOpSchema*  rad_dflash_tree_select_schema_at(int);
int                 rad_dflash_tree_select_shape(const RadParam*, int, int, RadOpdDesc*);
/* RAD_OK, or the reason this call is not the op: RAD_E_UNSUPPORTED outside the declared level /
 * top-k / budget band, RAD_E_DTYPE for a non-bf16 dtype, RAD_E_SHAPE for extents or pitches that
 * disagree with the parameters, RAD_E_STRIDE for a non-dense operand, RAD_E_INVAL for a required
 * operand absent or an output overlapping an input. */
int                 rad_dflash_tree_select_args(const RadArgs*, RadDflashTreeSelectCall*);
RadKernelInfo       rad_dflash_tree_select_row(RadLaunchFn);
int                 rad_mine_dflash_tree_select_count(void);
const RadKernelInfo* rad_mine_dflash_tree_select_at(int);
/* The launch entry point, declared here so the .hip defines its row from it. */
int                 rad_mine_dflash_tree_select_launch(const RadArgs*, RadStream);

#endif /* RAD_MINE_DFLASH_TREE_SELECT_H */
