#ifndef RAD_MINE_MTP_TREE_ACCUM_H
#define RAD_MINE_MTP_TREE_ACCUM_H

#include "rad_plugin.h"
#include <stdint.h>

/* candidates, token_ids?, draft_out_ids? -> history, packed.
 * Scalar round=1..max_rounds selects the issue; no new runtime ABI.
 * max_rounds=3 is the frozen legacy geometry (capacity<=16); max_rounds=8 is the
 * fixed MTP8 geometry (8 spine + 56 alternates + root, capacity<=65).
 * Round 1 resets all max_rounds history planes for each live row and gathers the root.
 * Later rounds reuse packed's root. Invalid roots/indices/live tokens emit count=0.
 * Negative candidates terminate a round, as in the old serial builder.
 * history is persistent [max_rounds,max_seqs,8]; packed is [M,1+2*capacity], root inclusive.
 * Each issue rebuilds every live packed row, spine first, then deduplicated siblings.
 * A new step MUST start at round 1; later issues MUST follow in order on one stream.
 */
struct RadMtpTreeAccumCall {
    const int32_t* cand;
    const int32_t* tokens;
    const int32_t* ids;
    int32_t* history;
    int32_t* packed;
    int64_t M, token_count;
    int64_t cand_ld, token_stride, id_stride, hist_round_stride, hist_row_stride, out_ld;
    int capacity, round, vocab, max_rounds;
};

/* TP=2 fusion: gathered [M,16,2] f32 (i32 id bits, f32 value) -> all eight
 * merged ids/values, then the same persistent history and packed tree.
 * Ordering is libr4d's unsigned 64-bit float-image/~id key, not float comparison.
 * Equal keys in different input slots disappear together, as in its 16-lane merge.
 * Negative ids are pads; exhausted outputs are (-1,-infinity).
 */
struct RadMtpMergeTreeAccumCall {
    RadMtpTreeAccumCall tree;
    const uint32_t* gathered;
    int32_t* idx;
    uint32_t* val;
    int64_t gathered_ld, val_ld;
};

int rad_mtp_tree_accum_schema_count(void);
const RadOpSchema* rad_mtp_tree_accum_schema_at(int);
int rad_mtp_tree_accum_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_tree_accum_args(const RadArgs*, RadMtpTreeAccumCall*);
RadKernelInfo rad_mtp_tree_accum_row(RadLaunchFn);
int rad_mtp_merge_tree_accum_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_merge_tree_accum_args(const RadArgs*, RadMtpMergeTreeAccumCall*);
RadKernelInfo rad_mtp_merge_tree_accum_row(RadLaunchFn);
int rad_mine_mtp_merge_tree_accum_launch(const RadArgs*, RadStream);
int rad_ref_mtp_merge_tree_accum(const RadArgs*, RadStream);
int rad_mine_mtp_tree_accum_launch(const RadArgs*, RadStream);
int rad_mine_mtp_tree_accum_count(void);
const RadKernelInfo* rad_mine_mtp_tree_accum_at(int);
int rad_ref_mtp_tree_accum(const RadArgs*, RadStream);
int rad_ref_mtp_tree_accum_count(void);
const RadKernelInfo* rad_ref_mtp_tree_accum_at(int);

#endif
