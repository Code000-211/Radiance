#ifndef RAD_MINE_MTP_PRUNED_TREE_H
#define RAD_MINE_MTP_PRUNED_TREE_H
#include "rad_plugin.h"
#include <stdint.h>

// Persistent row: status,p,anchor,tokens[41],parents[41]. Scores include root.
struct RadMtpPrunedTreeCall {
    int64_t M, Q, token_count;
    int round, vocab;
    double beta;
    const uint32_t* gathered;
    const int32_t *request_ids, *tokens, *anchor_rows, *positions;
    int32_t *meta, *packed;
    float* score;
    int32_t *token_ids, *out_positions, *gather_ids, *ancestors, *slots, *prefix;
    int64_t gathered_ld, request_stride, token_stride, anchor_stride, position_stride;
    int64_t meta_ld, score_ld, packed_ld;
    int64_t token_ids_stride, out_positions_stride, gather_ids_stride;
    int64_t ancestors_ld, slots_stride, prefix_stride;
};
using RadMtpPrunedExpandCall = RadMtpPrunedTreeCall;
using RadMtpPrunedPrepareCall = RadMtpPrunedTreeCall;
using RadMtpPrunedPackCall = RadMtpPrunedTreeCall;
int rad_mtp_pruned_tree_schema_count();
const RadOpSchema* rad_mtp_pruned_tree_schema_at(int);
int rad_mine_mtp_pruned_tree_count();
const RadKernelInfo* rad_mine_mtp_pruned_tree_at(int);
int rad_ref_mtp_pruned_tree_count();
const RadKernelInfo* rad_ref_mtp_pruned_tree_at(int);
#define RAD_PRUNED_DECL(op, Call) \
int rad_mtp_pruned_##op##_shape(const RadParam*, int, int, RadOpdDesc*); \
int rad_mtp_pruned_##op##_args(const RadArgs*, Call*); \
RadKernelInfo rad_mtp_pruned_##op##_row(RadLaunchFn); \
int rad_mine_mtp_pruned_##op##_launch(const RadArgs*, RadStream); \
int rad_ref_mtp_pruned_##op(const RadArgs*, RadStream);
RAD_PRUNED_DECL(expand, RadMtpPrunedExpandCall)
RAD_PRUNED_DECL(prepare, RadMtpPrunedPrepareCall)
RAD_PRUNED_DECL(pack, RadMtpPrunedPackCall)
#undef RAD_PRUNED_DECL
#endif
