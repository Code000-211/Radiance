#ifndef RAD_MINE_MTP_SCORED_TREE_H
#define RAD_MINE_MTP_SCORED_TREE_H
#include "rad_plugin.h"
#include <stdint.h>

struct RadMtpHybridLseCall {
    const uint16_t* coarse;
    const int32_t* idx;
    const float* val;
    float* stats;
    int64_t M, N, coarse_ld, idx_ld, val_ld, stats_ld;
    int vocab, vocab_off;
};
/* Policy 0 is ID-only: stats/log_history are absent and never accessed.
 * Policies 1/2 require separate aligned TP metadata and persistent log_history.
 * Packed count 1 is a valid root-only proposal. Count 0 signals invalid input
 * or normalization; the serial consumer fails the step instead of hiding it. */
struct RadMtpScoredMergeTreeAccumCall {
    const uint32_t* gathered;
    const float* stats;
    int32_t* idx;
    uint32_t* val;
    const int32_t* tokens;
    const int32_t* ids;
    int32_t* history;
    float* log_history;
    int32_t* packed;
    int64_t M, token_count, gathered_ld, stats_ld, idx_ld, val_ld;
    int64_t token_stride, id_stride, hist_round_stride, hist_row_stride;
    int64_t log_round_stride, log_row_stride, out_ld;
    int capacity, max_rounds, vocab, round, policy;
    double costs[8];
};
int rad_mtp_scored_tree_schema_count();
const RadOpSchema* rad_mtp_scored_tree_schema_at(int);
int rad_mtp_hybrid_lse_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_hybrid_lse_args(const RadArgs*, RadMtpHybridLseCall*);
RadKernelInfo rad_mtp_hybrid_lse_row(RadLaunchFn);
int rad_mtp_scored_merge_tree_accum_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_scored_merge_tree_accum_args(const RadArgs*, RadMtpScoredMergeTreeAccumCall*);
RadKernelInfo rad_mtp_scored_merge_tree_accum_row(RadLaunchFn);
int rad_mine_mtp_hybrid_lse_launch(const RadArgs*, RadStream);
int rad_mine_mtp_scored_merge_tree_accum_launch(const RadArgs*, RadStream);
int rad_ref_mtp_hybrid_lse(const RadArgs*, RadStream);
int rad_ref_mtp_scored_merge_tree_accum(const RadArgs*, RadStream);
int rad_mine_mtp_scored_tree_count();
const RadKernelInfo* rad_mine_mtp_scored_tree_at(int);
int rad_ref_mtp_scored_tree_count();
const RadKernelInfo* rad_ref_mtp_scored_tree_at(int);
#endif
