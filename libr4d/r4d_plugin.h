/* r4d_plugin.h -- what libr4d's translation units share.
 *
 * The row table (r4d_rows.cpp) and the weight layouts (r4d_layout.cpp) are pure host C++ and
 * include only this file and the ABI; the launch shims include r4d.h and so <hip/hip_runtime.h>.
 * The split lets the constraint table compile and be checked without ROCm, and is why every shim
 * is declared here rather than beside its definition.
 *
 * Nothing in this header may include r4d.h.
 */
#ifndef RAD_R4D_PLUGIN_H
#define RAD_R4D_PLUGIN_H

#include "r4d_args.h"


/* ================================================================== the priority ladder
 *
 * The selector ranks candidate rows by an integer priority (spec §2.1); where two rows of one op
 * can match a geometry, it decides which serves. Each group states the reason for its order.
 * Values are spaced by ten so a row can be slipped between two without renumbering.
 */
enum {
    /* A two-shot row's constraints are a superset of the one-shot's at the same world_size, so it
     * must rank higher to be reachable. Its lower bound on `numel` keeps small messages on the
     * one-shot. */
    R4D_PRIO_AR_TWOSHOT   = 70,
    R4D_PRIO_AR_ONESHOT   = 60,

    /* `attn_paged` is banded on q_len, and the decode row's bound (q_len<=10 at h256, q_len<=16 at
     * h128) is a bucket boundary, not a partition: both rows match in the low band, so decode must
     * outrank prefill. */
    R4D_PRIO_ATTN_DECODE  = 70,
    R4D_PRIO_ATTN_PREFILL = 60,
    /* The windowed kernels (r4d_attn_paged_window.hip) serve every q_len, so where their geometry
     * meets another leg's (h256 at gqa 2) they match in both of that leg's bands and rank below
     * both: on R9700 at the 31B's shapes (window 1024, fp8 cache) the h256/gqa2 decode kernel takes
     * 0.26-0.71 of the windowed kernel's time at q_len 1..32 and its prefill kernel 0.43-0.67 at
     * 33..512, bf16 cache similar. The windowed rows keep that geometry's non-causal band past
     * q_len 32, and attn_paged_qrow, which no other kernel serves. */
    R4D_PRIO_ATTN_WINDOW  = 50,

    /* At M<=16 with K%256==0 both bf16 skinny rows match and the m16 (dot2) kernel is taken. The
     * m64 WMMA kernel is faster on every shape measured, so a caller wanting it at low M asks with
     * M=64 to exclude m16. */
    R4D_PRIO_GEMM_M16     = 60,
    R4D_PRIO_GEMM_SKINNY  = 50,
    /* The MXFP4 decode row outranks the MXFP4 skinny row, whose band it shares up to N = 36864. */
    R4D_PRIO_GEMM_MXDEC   = 55,
    /* The skinny m64 row has no upper bound on M, so the tiled bf16 row (M > 64 only) must outrank
     * it, and stays below m16, whose band it never enters. */
    R4D_PRIO_GEMM_BF16_TILED = 55,

    /* Bands already separate these from the skinny kernels (skinny is M<=64, these unbounded), so
     * this pair only decides M>64. `pf` is the newer kernel and its tuned picks fall back to the
     * tiled one, so it outranks. Not expressible here: the optimum crosses between the two kernels
     * at a shape-dependent M, and tuning varies only within a kernel. */
    R4D_PRIO_GEMM_PREFILL = 40,
    R4D_PRIO_GEMM_TILED   = 30,

    /* The block-scaled fp8 family. fp8a16 is a separate dtype and needs no ordering against these;
     * the narrow fp8a8 row's M <= 16 is a subset of the tiled row's unbounded M, so it must
     * outrank. No dtype string reaches both these and the skinny int8 GEMMs. */
    R4D_PRIO_GEMM_FP8_M16   = 60,
    R4D_PRIO_GEMM_FP8_TILED = 40,
    /* The lm_head at fp8 (and int8), above the bf16 projection, which reads twice the bytes.
     * The head's encoding, fixed at conversion, decides which rows are candidates (spec §4.3):
     * these rows' layout hooks refuse a bf16 head, and the bf16 row has no hook for a scaled
     * one. */
    R4D_PRIO_LOGITS_FP8     = 60,

    /* Runtime LoRA: the tiled rows serve a subset of the scalar rows' dtypes at every M, so they
     * must outrank them to be reached; the scalar rows keep f16 / f32 activations (no exact WMMA)
     * and a coded activation whose scale group is not whole 32-wide steps, and stay reachable by
     * name. The choice never depends on M -- a row's bytes would then depend on its batch. */
    R4D_PRIO_LORA_TILED   = 60,
    R4D_PRIO_LORA_SCALAR  = 50,

    /* Exactly one candidate row per (op, geometry). The value carries no ordering; it leaves a
     * user's override plugin room above and below. */
    R4D_PRIO_ONLY         = 50,
};

/* ================================================================== the shims
 *
 * Declared here, defined in the HIP translation units. r4d_rows.cpp takes their addresses and so
 * stays free of r4d.h.
 */
extern "C" {

/* -- attention (r4d_attn_*.hip) -- */
int     r4d_rad_attn_prefill_h256_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_v4_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_k6_scratch(const RadArgs*);
int     r4d_rad_attn_decode_h256_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_scratch(const RadArgs*);
int64_t r4d_rad_attn_decode_h128_scratch(const RadArgs*);
/* Gemma 4 dense sliding window (r4d_attn_paged_h256_gqa2.hip): head_dim 256, 2 queries per KV
 * head. Both bands: decode (q_len*gqa <= 64, verify width 32 against the gqa6 leg's 10) and
 * windowed prefill (the gqa6 leg's r4d_attn_prefill_kernel with WIN on). */
int     r4d_rad_attn_prefill_h256_gqa2_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa2_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa2_fp8   (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa2_bf16  (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa2_scratch(const RadArgs*);
/* Gemma 3 sliding window (r4d_attn_paged_h128_gqa2w.hip): head_dim 128, 2 queries per KV head,
 * window >= 1. Decode (q_len*gqa <= 64) and windowed prefill, as the h256/gqa2 pair above. */
int     r4d_rad_attn_prefill_h128_gqa2w_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h128_gqa2w_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_gqa2w_fp8   (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_gqa2w_bf16  (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h128_gqa2w_scratch(const RadArgs*);
/* attn_paged_qrow on the split-KV decode kernel: the h256/gqa2 decode with q at its own pitch, and
 * head_dim 128 at gqa 8 as the gqa-4 kernel with two query groups a kv head (R4DArgs::kv_rep). */
int     r4d_rad_attn_qrow_decode_h256_gqa2_fp8   (const RadArgs*, RadStream);
int     r4d_rad_attn_qrow_decode_h256_gqa2_bf16  (const RadArgs*, RadStream);
int64_t r4d_rad_attn_qrow_decode_h256_gqa2_scratch(const RadArgs*);
int     r4d_rad_attn_qrow_h128_gqa8_fp8   (const RadArgs*, RadStream);
int     r4d_rad_attn_qrow_h128_gqa8_bf16  (const RadArgs*, RadStream);
int64_t r4d_rad_attn_qrow_h128_gqa8_scratch(const RadArgs*);
/* Paged attention at head_dim 256 and 512, 1/2/4/8 queries per KV head, any window, causal or not
 * (r4d_attn_paged_window.hip): one entry point per geometry and cache dtype, every q_len and any q
 * row pitch, so they serve both attn_paged and attn_paged_qrow. */
int     r4d_rad_attn_window_h256_gqa1_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa1_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa2_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa2_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa4_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa4_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa8_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h256_gqa8_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa1_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa1_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa2_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa2_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa4_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa4_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa8_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_window_h512_gqa8_fp8 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_window_h256_scratch(const RadArgs*);
int64_t r4d_rad_attn_window_h512_scratch(const RadArgs*);
/* The top-1 of a row of all-gathered (id, value) pairs, written one id a row: the vocab-sharded
 * draft head's merge (r4d_pair_argmax_f32.hip). */
int     r4d_rad_pair_argmax_f32(const RadArgs*, RadStream);
/* Gemma 4's two sandwich-norm joins, each one kernel for a chain of rmsnorm / add / mul_weight
 * launches (r4d_norm_join_bf16.hip), and expert_scale, the embed_lookup + mul pair that scales
 * each routing weight by its expert's scale (r4d_expert_scale_bf16.hip). */
int     r4d_rad_norm_add_norm_bf16      (const RadArgs*, RadStream);
int     r4d_rad_norm_add_norm_route_bf16(const RadArgs*, RadStream);
int     r4d_rad_join_add_scale_bf16     (const RadArgs*, RadStream);
int     r4d_rad_expert_scale_bf16       (const RadArgs*, RadStream);
/* Runtime LoRA's two halves, each row's adapter named by its slot (r4d_lora_bf16.hip): the scalar
 * rows, and the tiled rows (a WMMA shrink over row tiles with a K split fixed by K and R, its
 * partials in the op's scratch, and an expand that stages B[s] in LDS for a tile of rows). */
int     r4d_rad_lora_shrink             (const RadArgs*, RadStream);
int     r4d_rad_lora_expand             (const RadArgs*, RadStream);
int     r4d_rad_lora_shrink_wmma        (const RadArgs*, RadStream);
int64_t r4d_rad_lora_shrink_wmma_scratch(const RadArgs*);
int     r4d_rad_lora_expand_tiled       (const RadArgs*, RadStream);
int     r4d_rad_lora_expand_mm          (const RadArgs*, RadStream);
int     r4d_rad_attn_vit              (const RadArgs*, RadStream);
int     r4d_rad_attn_dense            (const RadArgs*, RadStream);
int     r4d_rad_seg_sinusoid          (const RadArgs*, RadStream);
/* The image generator's attention (r4d_attn_gen.hip): a paged prefix and the pass's own rows, the
 * causal-capable dense form at head_dim 128 (a text encoder's prompt), and one 1152-wide head
 * (an image decoder's mid block). */
int     r4d_rad_attn_prefix_self      (const RadArgs*, RadStream);
int     r4d_rad_attn_dense_h128       (const RadArgs*, RadStream);
int     r4d_rad_attn_cross            (const RadArgs*, RadStream);
int     r4d_rad_table_add             (const RadArgs*, RadStream);
int     r4d_rad_mix_rows              (const RadArgs*, RadStream);
int     r4d_rad_chan_stats            (const RadArgs*, RadStream);
int     r4d_rad_attn_dense_wide       (const RadArgs*, RadStream);
/* The image generator's other ops (r4d_gen.hip). */
int     r4d_rad_ln_mod                (const RadArgs*, RadStream);
int     r4d_rad_ln_mod_quant_fp8      (const RadArgs*, RadStream);
int     r4d_rad_gate_add              (const RadArgs*, RadStream);
int     r4d_rad_timestep_embed        (const RadArgs*, RadStream);
int     r4d_rad_gather_rows2          (const RadArgs*, RadStream);
int     r4d_rad_lane_gather           (const RadArgs*, RadStream);
int     r4d_rad_lane_store            (const RadArgs*, RadStream);
int     r4d_rad_randn_lanes           (const RadArgs*, RadStream);
int     r4d_rad_flow_euler            (const RadArgs*, RadStream);
int     r4d_rad_kd_input              (const RadArgs*, RadStream);
int     r4d_rad_kd_step               (const RadArgs*, RadStream);
int     r4d_rad_chan_affine           (const RadArgs*, RadStream);
int     r4d_rad_conv2d_nhwc           (const RadArgs*, RadStream);
int     r4d_rad_im2col_nhwc           (const RadArgs*, RadStream);
int     r4d_rad_l2norm_chan           (const RadArgs*, RadStream);
int     r4d_rad_d2s_add               (const RadArgs*, RadStream);
int     r4d_rad_s2d_avg_add           (const RadArgs*, RadStream);
int     r4d_rad_to_rgba8              (const RadArgs*, RadStream);
int     r4d_rad_gelu_tanh             (const RadArgs*, RadStream);
int     r4d_rad_gelu_quick            (const RadArgs*, RadStream);
int     r4d_rad_scatter_add_rows      (const RadArgs*, RadStream);
/* r4d_audio.hip */
int     r4d_rad_conv1d                (const RadArgs*, RadStream);
int     r4d_rad_conv_t1d              (const RadArgs*, RadStream);
int     r4d_rad_conv1d_f32            (const RadArgs*, RadStream);
int     r4d_rad_conv_t1d_f32          (const RadArgs*, RadStream);
int     r4d_rad_snake                 (const RadArgs*, RadStream);
int     r4d_rad_cfg_topk_sample       (const RadArgs*, RadStream);
int     r4d_rad_rvq_embed             (const RadArgs*, RadStream);
int     r4d_rad_frame_mix_store       (const RadArgs*, RadStream);
int     r4d_rad_end_status            (const RadArgs*, RadStream);
int     r4d_rad_frame_store           (const RadArgs*, RadStream);
int     r4d_rad_resample_nearest      (const RadArgs*, RadStream);
int     r4d_rad_fourier_embed         (const RadArgs*, RadStream);
int     r4d_rad_win_input             (const RadArgs*, RadStream);
int     r4d_rad_win_prep              (const RadArgs*, RadStream);
int     r4d_rad_win_finish            (const RadArgs*, RadStream);
int     r4d_rad_chan_split            (const RadArgs*, RadStream);
int     r4d_rad_interleave2           (const RadArgs*, RadStream);
int     r4d_rad_attn_dense_small_h256 (const RadArgs*, RadStream);
int     r4d_rad_iota                  (const RadArgs*, RadStream);
int     r4d_rad_cu_repeat             (const RadArgs*, RadStream);
int     r4d_rad_int_affine            (const RadArgs*, RadStream);
/* r4d_video.hip */
int     r4d_rad_rms_mod               (const RadArgs*, RadStream);
int     r4d_rad_rms_mod_quant_fp8     (const RadArgs*, RadStream);
int     r4d_rad_flow_x0_step          (const RadArgs*, RadStream);
int     r4d_rad_patch_unfold          (const RadArgs*, RadStream);
int     r4d_rad_vtile_gather          (const RadArgs*, RadStream);
int     r4d_rad_vtile_stitch          (const RadArgs*, RadStream);
int     r4d_rad_grid_pos3             (const RadArgs*, RadStream);
int     r4d_rad_snake_aa              (const RadArgs*, RadStream);
/* r4d_upscaler.hip: a video latent upscaler's 3D convolution, group norm, temporal depthwise
 * convolution and resize, the patch fold and the re-noise. */
int     r4d_rad_conv3d_ndhwc          (const RadArgs*, RadStream);
int     r4d_rad_group_norm            (const RadArgs*, RadStream);
int     r4d_rad_tconv_dw              (const RadArgs*, RadStream);
int     r4d_rad_resize_bilinear       (const RadArgs*, RadStream);
int     r4d_rad_patch_fold            (const RadArgs*, RadStream);
int     r4d_rad_flow_renoise          (const RadArgs*, RadStream);
int     r4d_rad_rope_split            (const RadArgs*, RadStream);
int     r4d_rad_head_gate             (const RadArgs*, RadStream);
int     r4d_rad_ltx_guide             (const RadArgs*, RadStream);
int     r4d_rad_pixel_norm            (const RadArgs*, RadStream);
int     r4d_rad_d2s3d                 (const RadArgs*, RadStream);
int     r4d_rad_s2d3d                 (const RadArgs*, RadStream);
int     r4d_rad_pix_patch             (const RadArgs*, RadStream);
int     r4d_rad_patch_pix8            (const RadArgs*, RadStream);
int     r4d_rad_vol_index             (const RadArgs*, RadStream);
int     r4d_rad_attn_na3d             (const RadArgs*, RadStream);
int     r4d_rad_tile_grid             (const RadArgs*, RadStream);
int     r4d_rad_tile_crop             (const RadArgs*, RadStream);
int     r4d_rad_tile_stitch           (const RadArgs*, RadStream);
int     r4d_rad_stft_logmel           (const RadArgs*, RadStream);
int     r4d_rad_bwe_up                (const RadArgs*, RadStream);
/* r4d_vsa.hip: video sparse attention's selection (scratch for the tile means) and attention. */
int     r4d_rad_vsa_select            (const RadArgs*, RadStream);
int64_t r4d_rad_vsa_select_scratch    (const RadArgs*);
int     r4d_rad_attn_vsa              (const RadArgs*, RadStream);
/* r4d_quant_act_fp8.hip: E4M3 codes with a scale per column over every row (an attention's V). */
int     r4d_rad_quant_cols_fp8        (const RadArgs*, RadStream);
int64_t r4d_rad_quant_cols_fp8_scratch(const RadArgs*);
/* Eight queries per KV head (r4d_attn_paged_gqa8.hip): the routed 35B-A3B at head_dim 256 and
 * MiniCPM5-2B at 128. A decode workgroup holds 64 rows of q_len*gqa, so the verify width (the
 * row's constraint value) is 8, against the gqa6 leg's 10. */
int     r4d_rad_attn_prefill_h256_gqa8_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa8_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa8_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa8_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa8_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h128_gqa8_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h128_gqa8_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_gqa8_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_gqa8_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h128_gqa8_scratch(const RadArgs*);
/* Four queries per KV head, head_dim 256 only (Qwen3.5-0.8B); same body, another instantiation
 * (r4d_attn_paged_h256_gqa4.hip). */
int     r4d_rad_attn_prefill_h256_gqa4_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa4_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa4_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa4_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa4_scratch(const RadArgs*);
/* ...and its head_dim 64 prefill leg, LFM2's geometry, serving every query length. */
int     r4d_rad_attn_prefill_h64_gqa4_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h64_gqa4_bf16(const RadArgs*, RadStream);
/* Two queries per KV head at head_dim 128 (Qwen3-1.7B, Qwen3-ASR's decoder): the gqa8 body at
 * gqa 2 (r4d_attn_paged_h128_gqa2.hip). */
int     r4d_rad_attn_prefill_h128_gqa2_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h128_gqa2_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_gqa2_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_gqa2_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h128_gqa2_scratch(const RadArgs*);
/* Twelve queries per KV head, head_dim 256 only: Qwen3.8-Flash-Next's QSA layers. Same body,
 * another instantiation (r4d_attn_paged_h256_gqa12.hip). */
int     r4d_rad_attn_prefill_h256_gqa12_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa12_fp8(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa12_bf16(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_scratch(const RadArgs*);
/* Six queries per KV head, head_dim 256, paged block 4: the same QSA layers at four ranks, with
 * the KV head replicated across a rank pair. The verify width is 64/6 = 10 against gqa12's 5,
 * the decode rows' constraint value. */
int     r4d_rad_attn_prefill_h256_gqa6_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa6_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_scratch(const RadArgs*);
/* The same two merges with the output gate and the quantiser, at the four-rank ratio. The body
 * instantiates these at every ratio (`R4D_ATTN_GQA` pastes them), so this needed a row, not a
 * kernel; see attn_decode_gq_h256_gqa6_fp8kv. */
int     r4d_rad_attn_decode_gq_h256_gqa6_fp8(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa6_bf16(const RadArgs*, RadStream);
/* The fp8_v4 and k6_v4 caches on both: prefill, decode, and decode with the gate and quantiser in
 * the merge (R4D_ATTN_V4 in r4d_attn_paged_gqa8.hip). */
int     r4d_rad_attn_prefill_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa12_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h256_gqa12_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa12_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa12_k6v4_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa6_fp8v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_fp8v4_scratch(const RadArgs*);
int     r4d_rad_attn_prefill_h256_gqa6_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_k6v4(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa6_k6v4(const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_gqa6_k6v4_scratch(const RadArgs*);

/* -- gated delta net (r4d_gdn_*.hip) -- */
int r4d_rad_gdn_chunk_scan      (const RadArgs*, RadStream);
int r4d_rad_gdn_conv_prep       (const RadArgs*, RadStream);
int r4d_rad_gdn_conv_update     (const RadArgs*, RadStream);
int r4d_rad_gdn_kkt_solve       (const RadArgs*, RadStream);
int r4d_rad_gdn_recurrent_update(const RadArgs*, RadStream);
/* gdn_conv_update folded into the recurrent update's prologue for the decode shape; q/k/v are
 * never written (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip). */
int r4d_rad_gdn_conv_recurrent_update(const RadArgs*, RadStream);
int r4d_rad_gdn_gated_rmsnorm   (const RadArgs*, RadStream);
int r4d_rad_dflash_conv         (const RadArgs*, RadStream);
int r4d_rad_dflash_select       (const RadArgs*, RadStream);

/* -- the gated residual (r4d_hc_bf16.hip) -- */
int     r4d_rad_hc_enter       (const RadArgs*, RadStream);
int     r4d_rad_hc_project     (const RadArgs*, RadStream);
int r4d_rad_mhc_pre(const RadArgs*, RadStream);
int r4d_rad_mhc_pre_partial(const RadArgs*, RadStream);
int r4d_rad_mhc_pre_finish(const RadArgs*, RadStream);
int r4d_rad_kda_update(const RadArgs*, RadStream);
int r4d_rad_moe_mxfp4(const RadArgs*, RadStream);
int r4d_rad_layout_moe_mxfp4(const RadParam*, int, int, const RadEncoding*, const int*,
                            const RadTensor*, int, RadLayout*);
int r4d_rad_mhc_post(const RadArgs*, RadStream);
int r4d_rad_mhc_contract(const RadArgs*, RadStream);
int     r4d_rad_mtp_enter      (const RadArgs*, RadStream);
int64_t r4d_rad_mtp_enter_scratch(const RadArgs*);
int     r4d_rad_hc_read        (const RadArgs*, RadStream);
int64_t r4d_rad_hc_read_scratch(const RadArgs*);
int     r4d_rad_hc_write       (const RadArgs*, RadStream);

/* -- all-reduce (r4d_ar_entry.hip) -- */
int  r4d_rad_ar_init(const RadParam*, int, int rank, int world, void** out);
void r4d_rad_ar_fini(void*);
/* The same three inits with every rank's exchange buffers in pinned host memory, for the host
 * twins of the collective rows (r4d_rows.cpp); launches and fini are shared. */
int  r4d_rad_ar_host_init(const RadParam*, int, int rank, int world, void** out);
int  r4d_rad_ar_ln_host_init(const RadParam*, int, int rank, int world, void** out);
int  r4d_rad_ar_hc_wide_host_init(const RadParam*, int, int rank, int world, void** out);
int  r4d_rad_ar_oneshot_2rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_gather_2rank       (const RadArgs*, RadStream);
/* The gather at a wide group's handshake (r4d_ar_gather_Nrank.hip launches it). */
int  r4d_rad_ar_gather_4rank       (const RadArgs*, RadStream);
int  r4d_rad_ar_gather_8rank       (const RadArgs*, RadStream);
int  r4d_rad_ar_oneshot_2rank_wht6 (const RadArgs*, RadStream);
int  r4d_rad_ar_oneshot_3rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_gather_3rank       (const RadArgs*, RadStream);
int  r4d_rad_ar_oneshot_4rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_oneshot_8rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_3rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_4rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_8rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_3rank_ti8  (const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_4rank_ti8  (const RadArgs*, RadStream);
/* The fused all-reduce + norm shares the IPC set, so it shares init/fini. */
int  r4d_rad_ar_ln_init(const RadParam*, int, int rank, int world, void** out);
int  r4d_rad_ar_ln_had_quant_i8(const RadArgs*, RadStream);
int  r4d_rad_ar_rmsnorm_quant_fp8(const RadArgs*, RadStream);
int  r4d_rad_ar_rmsnorm_had_quant_i8g(const RadArgs*, RadStream);
int  r4d_rad_ar_hc_write(const RadArgs*, RadStream);
/* ar_hc_write at four ranks, exact wire only, on an instance of the wide family keyed by its own
 * M bound. */
int  r4d_rad_ar_hc_wide_init(const RadParam*, int, int rank, int world, void** out);
int  r4d_rad_ar_hc_write_4rank(const RadArgs*, RadStream);
int  r4d_rad_ar_gather_hc_write(const RadArgs*, RadStream);
int64_t r4d_rad_ar_gather_hc_write_scratch(const RadArgs*);
/* ar_gather_hc_write at four ranks, exact wire only, on an instance from the same init. */
int  r4d_rad_ar_gather_hc_write_4rank(const RadArgs*, RadStream);

/* -- GEMM (r4d_gemm_*.hip) -- */
int r4d_rad_gemm_bf16_m16   (const RadArgs*, RadStream);
int r4d_rad_gemm_bf16_m64   (const RadArgs*, RadStream);
int r4d_rad_gemm_bf16_tiled (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a16_m64  (const RadArgs*, RadStream);
int r4d_rad_gemm_w5a16_m64  (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a16asym_m64(const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8i8_tiled(const RadArgs*, RadStream);
int r4d_rad_gemm_w5a8i8_tiled(const RadArgs*, RadStream);
int r4d_rad_givens_dense_quant(const RadArgs*, RadStream);
int r4d_rad_gemm_w8a16_m64  (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8_m64   (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8_tiled (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8_prefill(const RadArgs*, RadStream);
int r4d_rad_gemm_w8a8_m64   (const RadArgs*, RadStream);
int r4d_rad_gemm_w8a8_tiled (const RadArgs*, RadStream);
int r4d_rad_gemm_w2a8       (const RadArgs*, RadStream);
int r4d_rad_gemm_mxfp4a8_m64(const RadArgs*, RadStream);
int r4d_rad_gemm_mxfp4a8_tiled(const RadArgs*, RadStream);
int64_t r4d_rad_gemm_mxfp4a8_tiled_scratch(const RadArgs*);
int r4d_rad_gemm_mxfp4a8_decode(const RadArgs*, RadStream);
int64_t r4d_rad_gemm_mxfp4a8_decode_scratch(const RadArgs*);
int r4d_rad_dequant_w4      (const RadArgs*, RadStream);
int r4d_rad_rowtopk         (const RadArgs*, RadStream);
int64_t r4d_rad_rowtopk_scratch(const RadArgs*);
int r4d_rad_rowtopk_merge   (const RadArgs*, RadStream);
int r4d_rad_logit_rerank    (const RadArgs*, RadStream);
int r4d_rad_logit_rerank_i8 (const RadArgs*, RadStream);

/* -- the routed mixture of experts (r4d_moe.hip) --
 *
 * `moe_gemm_q`'s weight operands are tables (RAD_OPD_WTAB): the core resolves a run of declared
 * weights to a device array of per-expert pointers, since after placement a layer's experts are
 * not contiguous. See abi/rad_runtime.h. */
int     r4d_rad_router_topk        (const RadArgs*, RadStream);
int     r4d_rad_moe_scatter        (const RadArgs*, RadStream);
int     r4d_rad_router_topk_scatter(const RadArgs*, RadStream);
/* The router GEMM, selection and sort in one launch, optionally with the shared gate. init takes
 * the instance's arrival counter (r4d_moe.hip, "the arrival counter") and fini returns it; a
 * launch with no instance is refused. */
int  r4d_rad_router_gemm_topk_scatter_init(const RadParam*, int, int rank, int world, void** out);
void r4d_rad_router_gemm_topk_scatter_fini(void*);
int     r4d_rad_router_gemm_topk_scatter(const RadArgs*, RadStream);
int64_t r4d_rad_moe_scatter_scratch(const RadArgs*);
int     r4d_rad_moe_gemm_q         (const RadArgs*, RadStream);
/* Scratch for the run map that aligns the prefill form. Zero in the decode band and for a shape
 * whose parameters do not describe one, so a caller that never asks still gets a correct uniform
 * grid. */
int64_t r4d_rad_moe_gemm_q_scratch (const RadArgs*);
int64_t r4d_rad_moe_gemm_w4a8_scratch(const RadArgs*);
/* The same grouped GEMM with a 4-bit expert weight: same operands, schema and inner loop; only the
 * LDS staging and scale grid differ. See r4d_layout.cpp's moe w4 section for the stored plane. */
int     r4d_rad_moe_gemm_w4a8      (const RadArgs*, RadStream);
/* `moe_gemm_q_mod4`: the same two kernels over four classes of experts, by the expert's number
 * mod 4. The scratch hooks are the two-class ones. */
int     r4d_rad_moe_gemm_q_mod4    (const RadArgs*, RadStream);
int     r4d_rad_moe_gemm_w4a8_mod4 (const RadArgs*, RadStream);
/* `moe_gemm` at bf16: one weight table of the checkpoint's row-major planes, no scales. Same
 * grouping and three forms; its scratch hook sizes the same run map. */
int     r4d_rad_moe_gemm_bf16      (const RadArgs*, RadStream);
int64_t r4d_rad_moe_gemm_bf16_scratch(const RadArgs*);
int     r4d_rad_moe_gather         (const RadArgs*, RadStream);
int     r4d_rad_scale_rows         (const RadArgs*, RadStream);
int64_t r4d_rad_moe_gather_scratch (const RadArgs*);

/* -- quantisation, norms and the fusions (r4d_*quant*.hip, r4d_qk_norm_rope_gate.hip) -- */
int r4d_rad_quant_act_i8          (const RadArgs*, RadStream);
int r4d_rad_qkv_norm_rope         (const RadArgs*, RadStream);
int r4d_rad_add_rmsnorm           (const RadArgs*, RadStream);
int r4d_rad_qkv_norm_rope_kv_store(const RadArgs*, RadStream);
int r4d_rad_gelu_mul_quant_i8     (const RadArgs*, RadStream);
int r4d_rad_ar_rmsnorm_quant_i8   (const RadArgs*, RadStream);
int r4d_rad_ar_rmsnorm_quant_i8_4rank(const RadArgs*, RadStream);
int r4d_rad_rmsnorm_quant_i8      (const RadArgs*, RadStream);
int r4d_rad_had_quant_act_i8      (const RadArgs*, RadStream);
int r4d_rad_rmsnorm_had_quant_i8  (const RadArgs*, RadStream);
int r4d_rad_gated_had_quant_i8    (const RadArgs*, RadStream);
int r4d_rad_gdn_gated_norm_hq_i8  (const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate     (const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate_kv_store(const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate_kv_store_fp8v4(const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate_kv_store_k6v4(const RadArgs*, RadStream);
int r4d_rad_rope_table            (const RadArgs*, RadStream);

/* -- block-scaled fp8, and the device sampler (r4d_gemm_fp8*.hip / r4d_fp8_layout.cpp) --
 *
 * These return a negative code instead of throwing, unlike the rest of libr4d's GEMMs, so their
 * shims carry no R4D_TRY. */
int r4d_rad_gemm_fp8a16_m1  (const RadArgs*, RadStream);
int r4d_rad_gemm_fp8a8_m16  (const RadArgs*, RadStream);
int r4d_rad_gemm_fp8a8_gated_m16(const RadArgs*, RadStream);
int r4d_rad_gemm_fp8a8_bias_m16(const RadArgs*, RadStream);
int64_t r4d_rad_gemm_fp8a8_m16_scratch(const RadArgs*);
int r4d_rad_gemm_fp8a8_tiled(const RadArgs*, RadStream);
int r4d_rad_gemm_i8a8_m16  (const RadArgs*, RadStream);
int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs*);
int r4d_rad_logits_gemm_i8(const RadArgs*, RadStream);
int r4d_rad_gemm_i8a8_tiled(const RadArgs*, RadStream);
int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);
int r4d_rad_had_quant_act_i8g(const RadArgs*, RadStream);
int r4d_rad_quant_act_fp8   (const RadArgs*, RadStream);
int r4d_rad_had_quant_act_fp8 (const RadArgs*, RadStream);
int r4d_rad_givens_rotate (const RadArgs*, RadStream);
int r4d_rad_rmsnorm_quant_fp8(const RadArgs*, RadStream);
int r4d_rad_rmsnorm_had_quant_i8g(const RadArgs*, RadStream);
int r4d_rad_gated_quant_fp8 (const RadArgs*, RadStream);
int r4d_rad_gated_had_quant_fp8(const RadArgs*, RadStream);
int r4d_rad_gated_had_quant_i8g(const RadArgs*, RadStream);
int r4d_rad_gate_quant_fp8  (const RadArgs*, RadStream);
int r4d_rad_gram_accum      (const RadArgs*, RadStream);
/* The same accumulator over a plain bf16 activation, no scale operand. */
int r4d_rad_gram_accum_bf16 (const RadArgs*, RadStream);
int r4d_rad_qsa_block_key   (const RadArgs*, RadStream);
int r4d_rad_qsa_score       (const RadArgs*, RadStream);
int r4d_rad_qsa_select      (const RadArgs*, RadStream);
int64_t r4d_rad_qsa_select_scratch(const RadArgs*);
int r4d_rad_qsa_tail_store  (const RadArgs*, RadStream);
int r4d_rad_qsa_work        (const RadArgs*, RadStream);
int r4d_rad_sample_chain    (const RadArgs*, RadStream);
int r4d_rad_logits_gemm_fp8 (const RadArgs*, RadStream);

/* -- what each weight-reading kernel stores, from a weight's canonical planes --
 *
 * RadLayoutFn / RadRelayoutFn / RadUnrelayoutFn (rad_abi.h, spec §4.3). r4d_fp8_layout.cpp holds
 * the fp8 families (block-fp8 codes in fragment order; the lm_head, hyper-connection and MTP rows
 * with scales in each row's tail), r4d_layout.cpp the packed integer ones. Each refuses an
 * encoding its kernel does not read. A family whose kernel reads the canonical plane unchanged
 * (the row-major fp8 matvec, the fp8 experts) has only the layout hook, to check its input. */
#define R4D_HOOKS(name)                                                                         \
    int r4d_rad_layout_##name(const RadParam*, int, int, const RadEncoding*, const int*,       \
                              const RadTensor*, int, RadLayout*);                              \
    int r4d_rad_relayout_##name(const RadParam*, int, int, const RadEncoding*, const int*,     \
                                const RadTensor*, int, void*, int64_t);                        \
    int r4d_rad_unrelayout_##name(const RadParam*, int, int, const RadEncoding*, const int*,   \
                                  const void*, int64_t, const RadTensor*, int);
R4D_HOOKS(fp8_block)
R4D_HOOKS(i8_block)
R4D_HOOKS(fp8_logits)
R4D_HOOKS(i8_logits)
R4D_HOOKS(hc_e4m3)
R4D_HOOKS(mtp_e4m3)
R4D_HOOKS(w4)
R4D_HOOKS(w2a8)
R4D_HOOKS(moe_w4)
R4D_HOOKS(moe_w4_mod4)
#undef R4D_HOOKS
int r4d_rad_layout_fp8_rowmajor(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, RadLayout*);
int r4d_rad_layout_embed_q(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, RadLayout*);
int r4d_rad_layout_moe_fp8(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, RadLayout*);
int r4d_rad_layout_moe_fp8_mod4(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, RadLayout*);
int r4d_rad_layout_w8a16(const RadParam*, int, int, const RadEncoding*, const int*,
                         const RadTensor*, int, RadLayout*);
int r4d_rad_layout_w8a8(const RadParam*, int, int, const RadEncoding*, const int*,
                        const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_w8a16(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, void*, int64_t);
int r4d_rad_relayout_w8a8(const RadParam*, int, int, const RadEncoding*, const int*,
                          const RadTensor*, int, void*, int64_t);
int r4d_rad_layout_w5(const RadParam*, int, int, const RadEncoding*, const int*,
                      const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_w5(const RadParam*, int, int, const RadEncoding*, const int*,
                        const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_w5(const RadParam*, int, int, const RadEncoding*, const int*,
                          const void*, int64_t, const RadTensor*, int);
int r4d_rad_layout_gdense(const RadParam*, int, int, const RadEncoding*, const int*,
                          const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_gdense(const RadParam*, int, int, const RadEncoding*, const int*,
                            const RadTensor*, int, void*, int64_t);
/* conv2d_nhwc's weight, PyTorch's [cout][cin][k][k] stored tap-major [cout][k][k][cin]
 * (r4d_layout.cpp). */
int r4d_rad_layout_conv_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                            const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_conv_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                              const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_conv_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                                const void*, int64_t, const RadTensor*, int);
/* conv3d_ndhwc's weight, PyTorch's [cout][cin][kt][k][k] stored tap-major [cout][kt][k][k][cin]
 * (r4d_layout.cpp). */
int r4d_rad_layout_conv3d_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                              const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_conv3d_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_conv3d_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const void*, int64_t, const RadTensor*, int);
/* conv1d's weight, PyTorch's [cout][cin][k] stored [cout][k][cin], and conv_t1d's, PyTorch's
 * [cin][cout][2s] stored [s][cout][2][cin] (r4d_layout.cpp). */
int r4d_rad_layout_conv1d_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                              const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_conv1d_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_conv1d_tap(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const void*, int64_t, const RadTensor*, int);
int r4d_rad_layout_convt_phase(const RadParam*, int, int, const RadEncoding*, const int*,
                               const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_convt_phase(const RadParam*, int, int, const RadEncoding*, const int*,
                                 const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_convt_phase(const RadParam*, int, int, const RadEncoding*, const int*,
                                   const void*, int64_t, const RadTensor*, int);
/* The f32 rows' weights: conv1d's stored as bf16's, conv_t1d's [s][cout][ceil(k/s)][cin] with the
 * taps past k zero (r4d_layout.cpp). */
int r4d_rad_layout_conv1d_tap_f32(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_conv1d_tap_f32(const RadParam*, int, int, const RadEncoding*, const int*,
                                    const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_conv1d_tap_f32(const RadParam*, int, int, const RadEncoding*, const int*,
                                      const void*, int64_t, const RadTensor*, int);
int r4d_rad_layout_convt_phase_f32(const RadParam*, int, int, const RadEncoding*, const int*,
                                   const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_convt_phase_f32(const RadParam*, int, int, const RadEncoding*, const int*,
                                     const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_convt_phase_f32(const RadParam*, int, int, const RadEncoding*, const int*,
                                       const void*, int64_t, const RadTensor*, int);
int r4d_rad_layout_mxfp4(const RadParam*, int, int, const RadEncoding*, const int*,
                         const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_mxfp4(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, void*, int64_t);
int r4d_rad_layout_givens_pairs(const RadParam*, int, int, const RadEncoding*, const int*,
                                const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_givens_pairs(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, void*, int64_t);
int r4d_rad_unrelayout_givens_pairs(const RadParam*, int, int, const RadEncoding*, const int*,
                                    const void*, int64_t, const RadTensor*, int);

/* -- the small bf16 model ops (r4d_model_bf16.hip) --
 *
 * The embedding gather, norm, adds, rotary and KV scatter. Required because the buffers are in
 * VRAM and an op with no device launch is a fatal issue. `logits_gemm` is declared with them. */
int r4d_rad_embed_lookup    (const RadArgs*, RadStream);
/* The quantised gather (embed_lookup_q) for the n-gram table, read from VRAM or from pinned host
 * memory over the link. The L1 cache that keeps recurring rows on the card is opt-in
 * (RADIANCE_PLE_VRAM_CACHE_SETS) and is the same row's init/fini: with the variable unset or 0 the
 * row is allocated nothing and the launch is the one above, unchanged. */
int r4d_rad_embed_lookup_q  (const RadArgs*, RadStream);
int r4d_rad_embed_lookup_q_init(const RadParam*, int, int, int, void** out);
void r4d_rad_embed_lookup_q_fini(void*);
int r4d_rad_ngram_ids       (const RadArgs*, RadStream);
int r4d_rad_ple_gate        (const RadArgs*, RadStream);
int r4d_rad_ple_conv        (const RadArgs*, RadStream);
int r4d_rad_causal_conv1d   (const RadArgs*, RadStream);
int r4d_rad_cast            (const RadArgs*, RadStream);
int r4d_rad_gather_rows     (const RadArgs*, RadStream);
int r4d_rad_gather_rows_u8  (const RadArgs*, RadStream);
int r4d_rad_scatter_rows_u8 (const RadArgs*, RadStream);
int r4d_rad_scatter_rows_bf16(const RadArgs*, RadStream);
int r4d_rad_rmsnorm         (const RadArgs*, RadStream);
int r4d_rad_rmsnorm_p       (const RadArgs*, RadStream);
int r4d_rad_add             (const RadArgs*, RadStream);
int r4d_rad_mul             (const RadArgs*, RadStream);
int r4d_rad_mul_weight      (const RadArgs*, RadStream);
int r4d_rad_act_mul         (const RadArgs*, RadStream);
int r4d_rad_seg_mean        (const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope    (const RadArgs*, RadStream);
int r4d_rad_cast_widen      (const RadArgs*, RadStream);
int r4d_rad_cast_f16        (const RadArgs*, RadStream);
int r4d_rad_cast_bf16       (const RadArgs*, RadStream);
int r4d_rad_pool_rows       (const RadArgs*, RadStream);
int r4d_rad_relu            (const RadArgs*, RadStream);
int r4d_rad_silu            (const RadArgs*, RadStream);
int r4d_rad_gelu            (const RadArgs*, RadStream);
int r4d_rad_clamp           (const RadArgs*, RadStream);
int r4d_rad_conv2d          (const RadArgs*, RadStream);
int r4d_rad_conv1d_dw       (const RadArgs*, RadStream);
int r4d_rad_rel_sinusoid    (const RadArgs*, RadStream);
int r4d_rad_conv2d_unfold   (const RadArgs*, RadStream);
int r4d_rad_attn_local_rel  (const RadArgs*, RadStream);
int r4d_rad_sigmoid         (const RadArgs*, RadStream);
int r4d_rad_fill            (const RadArgs*, RadStream);
int r4d_rad_logit_softcap   (const RadArgs*, RadStream);
int r4d_rad_scale           (const RadArgs*, RadStream);
int r4d_rad_silu_mul        (const RadArgs*, RadStream);
int r4d_rad_swiglu_clamped(const RadArgs*, RadStream);
int r4d_rad_swiglu_clamped_quant_fp8(const RadArgs*, RadStream);
int r4d_rad_dsa_pool(const RadArgs*, RadStream);
int r4d_shape_dsa_pool(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_dsa_query(const RadArgs*, RadStream);
int r4d_rad_fp8_f32scale(const RadArgs*, RadStream);
int r4d_rad_moe_fp8_f32scale(const RadArgs*, RadStream);
int r4d_layout_fp8_f32scale(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, RadLayout*);
int r4d_shape_dsa_query(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_gemm_nt_f32(const RadArgs*, RadStream);
int r4d_shape_gemm_nt_f32(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_mla_project(const RadArgs*, RadStream);
int r4d_shape_mla_project(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_dsa_score(const RadArgs*, RadStream);
int r4d_shape_dsa_score(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_mla_attn(const RadArgs*, RadStream);
int64_t r4d_rad_mla_attn_scratch(const RadArgs*);
int r4d_rad_mla_attn_matrix(const RadArgs*, RadStream);
int64_t r4d_rad_mla_attn_matrix_scratch(const RadArgs*);
int r4d_shape_mla_attn(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_router_sigmoid_topk(const RadArgs*, RadStream);
int r4d_shape_router_sigmoid_topk(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_router_sigmoid_topk_scatter(const RadArgs*, RadStream);
int r4d_rad_moe_scatter_inv(const RadArgs*, RadStream);
int r4d_shape_moe_scatter_inv(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_moe_gather_inv(const RadArgs*, RadStream);
int r4d_shape_moe_gather_inv(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_router_sigmoid_topk_scatter(const RadParam*, int, int, RadOpdDesc*);
int r4d_rad_gelu_mul        (const RadArgs*, RadStream);
int r4d_rad_rope            (const RadArgs*, RadStream);
/* -- the vision tower's ops (r4d_vit_bf16.hip) -- */
int r4d_rad_layernorm       (const RadArgs*, RadStream);
int r4d_rad_grid_embed      (const RadArgs*, RadStream);
int r4d_rad_grid_embed_aa    (const RadArgs*, RadStream);
int r4d_rad_gemm_nt_bias    (const RadArgs*, RadStream);
/* gemm_nt_bias at f32 (r4d_gemm_f32_nt_bias.hip). */
int r4d_rad_gemm_f32_nt_bias(const RadArgs*, RadStream);
int r4d_rad_kv_store        (const RadArgs*, RadStream);
int r4d_rad_kv_store_fp8    (const RadArgs*, RadStream);
int r4d_rad_kv_store_fp8v4  (const RadArgs*, RadStream);
int r4d_rad_kv_store_k6v4   (const RadArgs*, RadStream);
int r4d_rad_logits_gemm     (const RadArgs*, RadStream);

/* -- the decomposed sampler chain (r4d_sample_stages_f32.hip) --
 *
 * docs/OPS.md's eleven ops, beside r4d_rad_sample_chain's fused form. The fused one suits a
 * speculative verify; these let an engine compose the chain per request and carry the stages the
 * fused one lacks. */
int r4d_rad_sample_penalties (const RadArgs*, RadStream);
int r4d_rad_sample_dry       (const RadArgs*, RadStream);
int r4d_rad_sample_temp      (const RadArgs*, RadStream);
int r4d_rad_sample_mask      (const RadArgs*, RadStream);
int r4d_rad_sample_topk      (const RadArgs*, RadStream);
int r4d_rad_sample_argmax    (const RadArgs*, RadStream);
int r4d_rad_sample_logprobs  (const RadArgs*, RadStream);
int r4d_rad_sample_topp      (const RadArgs*, RadStream);
int r4d_rad_sample_minp      (const RadArgs*, RadStream);
int r4d_rad_sample_typical   (const RadArgs*, RadStream);
int r4d_rad_sample_xtc       (const RadArgs*, RadStream);
int r4d_rad_sample_pick      (const RadArgs*, RadStream);
int r4d_rad_sample_merge_topk(const RadArgs*, RadStream);

/* -- operand descriptions (r4d_shapes.cpp, pure host) --
 *
 * `RadKernelInfo::opd_shape`: the size and allowed contents of operand i of this op at this
 * geometry. One function per op, not per row, because the description belongs to the operand list,
 * not the tiling: all six `attn_paged` rows share one and read the cache width off `kv_dtype`.
 * Where rows genuinely differ they take different functions, or carry NULL and are skipped by name
 * (`logits_gemm`'s fp8 row). r4d_shapes.cpp's header lists every decline and its reason.
 *
 * Pure host code called before allocation, so like r4d_rows.cpp and r4d_layout.cpp it includes
 * neither r4d.h nor <hip/hip_runtime.h>. */
int r4d_shape_embed_lookup   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_embed_lookup_q (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gather_rows    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gather_rows_u8 (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scatter_rows_u8(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_cast           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_enter       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_mhc_pre(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_mhc_pre_partial(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_mhc_pre_finish(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_kda_update(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_moe_mxfp4(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_mhc_post(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_mhc_contract(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_mtp_enter      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_read        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_write       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_project     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_binary         (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_mul_weight     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scatter_rows_bf16(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_layernorm      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_grid_embed     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_bias   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_unary          (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_fill           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_silu_mul       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rope           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_kv_store       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_logits_gemm    (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_sample_penalties (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_dry       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_temp      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_mask      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_topk      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_argmax    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_logprobs  (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_narrow    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_pick      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_chain     (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_attn_paged        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_attn_paged_gate_quant(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qk_norm_rope_gate (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qk_norm_rope_gate_kv_store(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qkv_norm_rope     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qkv_norm_rope_kv_store(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_add_rmsnorm(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rope_table        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gdn_gated_rmsnorm (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_all_reduce           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_all_gather           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_ln_had_quant_i8   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_rmsnorm_quant_fp8 (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_rmsnorm_had_quant_i8g (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_hc_write          (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_gather_hc_write   (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_gemm_nt        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_q      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_q_bias (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_q_gated(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_dequant_w4     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_row_topk       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_router_topk    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_router_topk_scatter(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_router_gemm_topk_scatter(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_moe_scatter    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_moe_gemm_q     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_moe_gemm_q_mod4(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_moe_gather     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scale_rows     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scale          (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_row_topk_merge (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_pair_argmax    (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_quant_act_i8       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_had_quant_act_i8   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm_had_quant_i8(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gated_had_quant_i8 (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gdn_gated_norm_had_quant_i8(const RadParam*, int, int operand,
                                          RadOpdDesc*);
int r4d_shape_quant_act_fp8      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm_quant_fp8  (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gated_quant_fp8    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm_had_quant_i8g (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gated_had_quant_i8g   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gate_quant_fp8     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gram_accum         (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qsa_block_key      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qsa_score          (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qsa_select         (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qsa_tail_store     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qsa_work           (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_dflash_conv    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_dflash_select  (const RadParam*, int, int operand, RadOpdDesc*);

/* -- declared fusions (r4d_fuse.cpp, pure host) --
 *
 * `RadKernelInfo::replaces`: the chain of conventional ops this kernel computes in one launch, so a
 * tool can run the chain on this plugin's own unfused rows and compare bytes. One function per
 * fused row, not per op: two rows of one op could fold different things.
 *
 * Only some rows declare one. r4d_fuse.cpp's header lists each decline with the libr4d line that
 * decides it: the int8 rotated family runs its Hadamard before the norm closes and keeps the
 * product in f32, so `silu_mul` + `had_quant_act_i8` is a different function, and
 * `ar_ln_had_quant_i8` reduces without the bf16 round its unfused pair stores. A chain that is only
 * close publishes no hook, because a false failure is worse than a skip.
 *
 * Pure host code, like r4d_rows.cpp, r4d_layout.cpp and r4d_shapes.cpp. */
int r4d_fuse_rmsnorm_quant_fp8   (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gated_quant_fp8     (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_swiglu_clamped_quant_fp8(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_router_sigmoid_topk_scatter(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gate_quant_fp8      (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gemm_nt_q_gated     (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gdn_conv_recurrent_update(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_qk_norm_rope_gate_kv_store(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_qkv_norm_rope_kv_store(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_add_rmsnorm(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_ar_rmsnorm_quant_fp8(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_ar_gather_hc_write  (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_attn_paged_gate_quant(const RadParam*, int, uint64_t, int, RadFuseStep*);

}  /* extern "C" */

#endif /* RAD_R4D_PLUGIN_H */
