#ifndef RAD_MINE_ATTN_PAGED_TREE_H
#define RAD_MINE_ATTN_PAGED_TREE_H

#include "rad_plugin.h"
#include <cstdint>

/* Model-owned, full-context attention: prefix followed by <=65 ordered tree nodes.
 * spec_parent is compact I32[T], sequence-local; root=-1, 0<=parent[j]<j.
 * Ordinary causal rows use parent[j]=j-1, never an all-minus-one sentinel.
 * Invalid device metadata produces zero output for the affected sequence.
 * No descales: the target stores unscaled OCP E4M3FN codes in its paged cache.
 * Numerik: verlustfrei-nicht-bitgenau relative to upstream decode's F16/PF16
 * path: conversions are exact, FP32 QK/softmax/PV and partials retain at least
 * its precision, but the dot reduction and split/merge order differ. */
enum { RAD_ATP_Q, RAD_ATP_KV, RAD_ATP_BT, RAD_ATP_SEQ, RAD_ATP_CU,
       RAD_ATP_PARENT, RAD_ATP_OUT, RAD_ATP_OPERANDS };
constexpr int RAD_ATP_SPLITS = 16;
constexpr int RAD_ATP_PARTIAL = 258;  // FP32 numerator[256], maximum, denominator
struct RadAttnPagedTreeCall {
    const uint16_t* q;
    const uint8_t* kv;
    const int32_t *bt, *seqused, *cu, *parent;
    uint16_t* out;
    int rows, seqs, pages, bt_width, max_ctx, q_bound;
    int64_t bt_pitch, kv_block_pitch, kv_head_pitch;
};
int rad_attn_paged_tree_schema_count();
const RadOpSchema* rad_attn_paged_tree_schema_at(int);
int rad_attn_paged_tree_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_attn_paged_tree_args(const RadArgs*, RadAttnPagedTreeCall*);
int64_t rad_attn_paged_tree_scratch(const RadArgs*);
RadKernelInfo rad_attn_paged_tree_row(RadLaunchFn);
int rad_ref_attn_paged_tree_launch(const RadArgs*, RadStream);
int rad_ref_attn_paged_tree_count();
const RadKernelInfo* rad_ref_attn_paged_tree_at(int);
int rad_mine_attn_paged_tree_count();
const RadKernelInfo* rad_mine_attn_paged_tree_at(int);
#endif
