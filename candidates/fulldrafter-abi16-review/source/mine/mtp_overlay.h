#ifndef RAD_MINE_MTP_OVERLAY_H
#define RAD_MINE_MTP_OVERLAY_H
#include "rad_plugin.h"
#include <stdint.h>

/* Read-only committed prefix, then chronological provisional ancestors. Invalid
 * device metadata atomically invalidates request status and zeroes the query
 * before KV reads. max_ctx bounds committed positions; depth adds1..4 positions.
 * No scratch or prefix copies. GPU: exact dtype decode, FP32 products,
 * zero-padded 256-lane binary-tree QK
 * sum (offsets128..1), scale1/sqrt(head_dim), chronological online FP32 softmax
 * and PV, BF16 RTNE once at output. No intermediate probability narrowing.
 * Reference: independent ascending scalar FP32 QK and two-pass FP32 softmax/PV.
 * Neither path claims target WMMA arithmetic or bitwise target parity. */
struct RadMtpOverlayCall {
    const uint16_t* q;
    const void* committed_kv;
    const int32_t* committed_bt;
    const int32_t* prefix;
    const void* provisional_kv;
    const int32_t* ancestors;
    uint16_t* out;
    int64_t M, rows, pages, provisional_pages, bt_pitch;
    int32_t* meta;
    const int32_t* request_ids;
    int64_t N, meta_ld, request_stride;
    int64_t q_ld, out_ld, bt_ld, prefix_stride, ancestor_ld;
    int64_t committed_stride[4], provisional_stride[4];
    int head_dim, n_head, n_head_kv, block_size, depth, max_ctx;
    uint32_t kv_dtype;
};
int rad_mtp_overlay_schema_count();
const RadOpSchema* rad_mtp_overlay_schema_at(int);
int rad_mtp_overlay_attn_shape(const RadParam*, int, int, RadOpdDesc*);
int rad_mtp_overlay_attn_args(const RadArgs*, RadMtpOverlayCall*);
RadKernelInfo rad_mtp_overlay_attn_row(RadLaunchFn);
int rad_mine_mtp_overlay_attn_launch(const RadArgs*, RadStream);
int rad_ref_mtp_overlay_attn(const RadArgs*, RadStream);
int rad_mine_mtp_overlay_count();
const RadKernelInfo* rad_mine_mtp_overlay_at(int);
int rad_ref_mtp_overlay_count();
const RadKernelInfo* rad_ref_mtp_overlay_at(int);
#endif
