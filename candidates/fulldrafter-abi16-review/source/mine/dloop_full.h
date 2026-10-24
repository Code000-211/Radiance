#ifndef RAD_MINE_DLOOP_FULL_H
#define RAD_MINE_DLOOP_FULL_H

#include "dloop_ops.h"

/* ============================================================ full-context DLoop drafter
 *
 * The L-layer, whole-verified-prefix drafter. Three ops, one contract, and the contract is
 * the reason this header exists: both the host oracle in ref_dloop_full.cpp and the device
 * kernels in dloop_kernels.hip read the same cache through the same arithmetic, and the engine
 * block that issues them sizes its buffers from the same hooks.
 *
 * WHAT IS DIFFERENT FROM THE RING OPS (dloop_ops.h), which stay exactly as they are:
 *
 *   THE WINDOW IS GONE. `model.context_window` is 0 and the validator refuses anything else.
 *   `context_capacity` (C) is an ALLOCATION BOUND: the number of verified positions each slot's
 *   cache can hold. It is never a sliding window, and exceeding it is an error. Nothing here
 *   truncates, wraps or drops a prefix row -- not at 32, not at C.
 *
 *   THE CACHE IS PER LAYER AND PRE-ROTATED. F32 [Q,L,C,2*D] with D = num_kv_heads*head_dim:
 *   for one (slot, layer, position) the first D values are K, already rotated at that position's
 *   absolute index, and the next D are V. F32, because a re-quantised cache is a different model.
 *   rad_dloop_full_kv_index is the only place the layout is spelled.
 *
 *   KV IS PROJECTED ONCE, AT APPEND. `dloop_full_append` projects each new verified tap row into
 *   every layer's K and V and writes it. base and tree then only READ the cache: no launch
 *   rebuilds historical K/V, and tree depth costs no context repacking.
 *
 *   ROPE POSITIONS ARE ABSOLUTE. The block's Q/K are rotated starting at length[slot], not at C.
 *   Attention sees exactly length[slot] cached positions plus the whole bidirectional block.
 *
 *   THE ROWS ARE VERIFIED TARGET TAPS, NOT DRAFTER OUTPUT. Only the target engine's accepted
 *   path is appended; a draft or tree step never writes historical storage.
 *
 * METADATA, VALIDATED IN FULL BEFORE THE FIRST WRITE:
 *
 *   length I32[Q]     verified prefix length of a slot. Slot-scoped, never batch-ordered.
 *   slots  I32[S]     the slot each sequence uses. Unique within one append.
 *   span   I32[S,2]   [start,count]. count >= 0 is a contiguous chunk of `count` tap rows
 *                     starting at `start`. count == -1 means the -1-terminated `path`
 *                     instead. Any other negative count is invalid.
 *   path   I32[S,16]  absolute tap-plane rows of one chunk, -1 terminated.
 *   reset  I32[S]     0/1. Only the first prefill chunk of a slot occupation sets 1; further
 *                     chunks append. An empty chunk is legal; a length-0 draft is not.
 *
 *   Every sequence of the batch is checked before any cache byte, length or output changes: a
 *   malformed slot, path, span or a capacity overflow leaves every operand untouched.
 *
 * SCRATCH. The host uses one F32 arena per sequence, sized by rad_dloop_full_scratch_layout.
 * Append reuses one projection row. Device hooks bound staging by 128 rows independently of
 * tap_rows; base/tree never copy the historical cache. Only the host's attention score row
 * grows with C; the device streams attention with bounded per-wave state.
 */
enum {
    RAD_DLOOP_FULL_APPEND = 0,
    RAD_DLOOP_FULL_BASE   = 1,
    RAD_DLOOP_FULL_DARTREE = 2,
    RAD_DLOOP_FULL_OPS    = 3
};

/* SPAN AND PATH, the two spellings of one chunk.
 *
 * `span` I32[S,2] is [start,count]. count >= 0 is `count` tap rows from `start`; count == -1
 * means the `path` operand instead; any other negative count is invalid.
 *
 * `path` I32[S,16] holds absolute tap-plane rows. A -1 fills the columns a chunk does not use;
 * a FULL sixteen columns needs no terminator, because root plus fifteen accepted draft nodes is a
 * legal complete path and the schema has exactly sixteen places for it. So the rule is: every
 * column up to the first -1 is a row index, every column from the first -1 on must be -1 (a
 * positive behind the terminator is a malformed path, not a silent tail to ignore), and sixteen
 * un-terminated columns are valid. */

/* `model` is the ring config with context_window == 0 and F == 3*target_width (three tap
 * planes). `context_capacity` is REQUIRED and is C. */
struct RadDLoopFullConfig {
    RadDLoopConfig model;
    int context_capacity;
};

/* The one place the cache layout is spelled out: element offset of column `d` (0 <= d < 2*D) of
 * (slot, layer, position). `position` is absolute and < context_capacity. */
static inline int64_t rad_dloop_full_kv_index(const RadDLoopFullConfig& c, int slot, int layer,
                                              int position, int d) {
    const int64_t D = (int64_t)c.model.num_kv_heads * c.model.head_dim;
    return ((((int64_t)slot * c.model.num_layers + layer) * c.context_capacity + position) * 2 * D) + d;
}

/* One F32 ARENA PER SEQUENCE, in 32-bit elements: every region below is an element OFFSET into
 * the sequence's slice, and `total` is the slice's size in elements (int32 regions occupy one
 * element each, so a single count sizes both). Unused fields are 0 for the op they do not
 * belong to. B = proposal_length+1, D = num_kv_heads*head_dim, G = hidden/conv_group,
 * I = intermediate, C = context_capacity, cap = rad_dloop_full_tree_capacity.
 *   block/z/pre/q/att/out   B*H     backbone activations
 *   kblk/vblk               B*D     the block's own K/V, roped at absolute positions
 *   dyn                     4*B*G   dynamic (grouped) convolution taps of one branch
 *   gate/up                 B*I     MLP
 *   score                   C+B     one attention row; the cache itself is never copied
 *   y                       M*W     draft-row output of one forward
 *   rank                    R       selector hidden projection of one candidate
 *   cand                    budget*K parent-major candidate scores of one depth
 *   node_score              cap+2   cumulative path score of every tree node
 *   nodes                   4*(cap+2) I32: token/parent/depth/source per node
 *   idx                     3*(cap+2) I32: prune order, keep flags, parent remap
 *   front                   2*budget  I32: the depth frontier and its successor
 *   stage                   append only: F+H -- one host projection row, reused for every tap.
 *                           Device staging has its own chunk-bounded layout and scratch hook.
 */
struct RadDLoopFullScratch {
    int64_t block, z, pre, q, att, out, kblk, vblk, dyn, gate, up, score, y, rank;
    int64_t cand, node_score;
    int64_t nodes, idx, front;
    int64_t stage;
    int64_t total;
};
RadDLoopFullScratch rad_dloop_full_scratch_layout(const RadDLoopFullConfig&, int op);
int64_t rad_dloop_full_scratch_elements(const RadDLoopFullConfig&, int op);
/* Nodes a pruned tree may hold before pruning -- width*proposal_length, or the budget when
 * unpruned -- and the largest frontier one depth can present. Both are bounds the arena is sized
 * from, and both are what the scratch hook is allowed to promise a kernel. */
int rad_dloop_full_tree_capacity(const RadDLoopFullConfig&);
int rad_dloop_full_frontier_max(const RadDLoopFullConfig&);

/* Parameters: the ring op's 29 names with context_window required to be 0, plus context_capacity.
 * Same weight blob order and offsets as dloop_ops.h: rad_dloop_weight_elements is unchanged. */
int rad_dloop_full_config(const RadParam*, int, RadDLoopFullConfig*);
int rad_dloop_full_args(const RadArgs*, int op, RadDLoopFullConfig*);

int rad_dloop_full_schema_count(void);
const RadOpSchema* rad_dloop_full_schema_at(int i);
/* The op-first form is the generic one; the three below are the per-op wrappers a RadShapeFn
 * pointer needs, since that type carries no op. */
int rad_dloop_full_shape(int op, const RadParam*, int np, int operand, RadOpdDesc*);
int rad_dloop_full_append_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_dloop_full_base_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_dloop_full_dartree_shape(const RadParam*, int, int, RadOpdDesc*);

/* Bytes, op-agnostic (the largest of the three) and per op. A launch is issued from a row, so the
 * per-op forms are what the engine allocates from. */
int64_t rad_dloop_full_scratch(const RadArgs*);
int64_t rad_dloop_full_append_scratch(const RadArgs*);
int64_t rad_dloop_full_base_scratch(const RadArgs*);
int64_t rad_dloop_full_tree_scratch(const RadArgs*);

RadKernelInfo rad_dloop_full_row(int op, int domain, RadLaunchFn launch);

int rad_ref_dloop_full_count(void);
const RadKernelInfo* rad_ref_dloop_full_at(int i);
int rad_ref_dloop_full_append(const RadArgs*, RadStream);
int rad_ref_dloop_full_base(const RadArgs*, RadStream);
int rad_ref_dloop_full_dartree(const RadArgs*, RadStream);

/* Device entry points, defined in dloop_kernels.hip / dloop_tree.hip and registered by
 * dloop_full.cpp. Production geometry first, then the five-layer fixture geometry. C linkage, as
 * every other mine_* entry point in this plugin: they are looked up by name, not linked by type. */
extern "C" {
int rad_mine_dloop_full_count(void);
const RadKernelInfo* rad_mine_dloop_full_at(int i);
/* The device side's OWN arena, for the same reason dloop_rows.cpp publishes
 * rad_mine_dloop_base_scratch: a device kernel keeps staging the host oracle has no use for --
 * projected tap rows, the head's candidate keys -- and the hook on a device row must cover it.
 * Defined by the device TU (see the RAD_DLOOP_FULL_DEVICE guard in dloop_full.cpp); the host
 * build never references it. */
int64_t rad_mine_dloop_full_scratch(const RadArgs*, int op);
int rad_mine_dloop_full_append_h768_l5_tp2(const RadArgs*, RadStream);
int rad_mine_dloop_full_base_h768_l5_tp2(const RadArgs*, RadStream);
int rad_mine_dloop_full_tree_h768_l5_tp2(const RadArgs*, RadStream);
int rad_mine_dloop_full_append_h64_l5_tp2(const RadArgs*, RadStream);
int rad_mine_dloop_full_base_h64_l5_tp2(const RadArgs*, RadStream);
int rad_mine_dloop_full_tree_h64_l5_tp2(const RadArgs*, RadStream);
}

/* Operand order, for the callers that build these calls by hand and for the harness:
 *
 *  append   0 tap0 BF16[T,W]      5 slots  I32[S]
 *           1 tap1 BF16[T,W]      6 reset  I32[S]
 *           2 tap2 BF16[T,W]      7 weights BF16[NW]
 *           3 path I32[S,16]      8 cache  F32[Q,L,C,2D]  inout
 *           4 span I32[S,2]       9 length I32[Q]         inout
 *
 *  base     0 cache F32[Q,L,C,2D]  5 head_shard BF16[Vs,W]
 *           1 length I32[Q]        6 y_base   F32[S,M,W]
 *           2 slots  I32[S]        7 local_ids I32[S,M,K]
 *           3 anchor_embed BF16[S,W] 8 local_logits F32[S,M,K]
 *           4 weights BF16[NW]      (no ctxh output: the cache IS the context)
 *
 *  dartree  0 cache, 1 length, 2 slots, 3 anchor_embed, 4 weights, 5 y_base, 6 pool_ids,
 *           7 unary, 8 pool_head, 9 pool_embed, 10 pool_pred, 11 pool_succ, 12 anchor_pred,
 *           13 tree_tok I32[S,16], 14 tree_parent I32[S,16], 15 n_nodes I32[S],
 *           16 anchor I32[S]?, 17 packed_tree I32[S,33]?   (16 and 17 both or neither)
 */
#endif
