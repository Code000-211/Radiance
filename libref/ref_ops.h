/* ref_ops.h -- the entry points ref_registry.cpp puts in the kernel table, and the layout
 * conventions they share.
 *
 * ============================== Shapes are read off the operands ==============================
 *
 * Every extent this plugin loops over comes from a RadTensor's shape, never from the parameters:
 *
 *   * A ranged parameter is collapsed to a band when the instance is resolved (spec §2.2), and
 *     `rad_issue` carries the actual value separately. A loop bounded by `M` would run to the
 *     band's upper bound (8192 rows for a one-row decode) and read past the arena. Ref takes M
 *     from the tensor.
 *   * Other parameters are either derivable from a shape (head_dim, N, K, n) or not shapes at all
 *     (causal, window, block_size, eps, group, top_k). Ref uses the latter directly and
 *     cross-checks the former against the shape, refusing with RAD_E_SHAPE where a disagreement
 *     would read out of bounds.
 *
 * ============================== The paged KV layout ==============================
 *
 * libr4d's layout, which every kernel must match:
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]
 *
 * K then V within a slot: key element d at [..., d], value element d at [..., head_dim + d]. A
 * flat slot index s is block = s / block_size, offset = s % block_size; `slot_mapping` carries such
 * indices and `block_table` maps to blocks.
 *
 * ============================== The speculative rollback contract ==============================
 *
 * The linear-attention pair (gdn_conv_update, gdn_recurrent_update) supports rejection without
 * recompute (spec §10); the contract is part of the schema:
 *
 *   conv state   [n_slots, conv_dim, state_len], state_len >= conv_width - 2 + q_len.
 *                A step reads its window at offset num_accepted[s] - 1 (the slot the last
 *                accepted token left) and rewrites the buffer shifted down by one: slots
 *                0 .. conv_width-3 take the tail of the old history, and this step's token t lands
 *                at slot t + conv_width - 2. Invariant: a step's last accepted token always sits at
 *                the slot its index implies, so only the next step's read offset changes.
 *   linear state [n_slots, n_head_v, head_v, head_k], addressed through state_idx[s, 0..q_len).
 *                The initial state is read from state_idx[s, num_accepted[s] - 1], and every
 *                candidate token t writes its own state to state_idx[s, t], since which survives
 *                verification is unknown until after this layer.
 *
 * num_accepted counts the previous step's accepted tokens, 1-based: 1 means no draft survived, only
 * the committed token. It is a required operand on both ops; a non-speculative decode passes all
 * ones. Ref still reads a null as 1, for direct callers with no declare to enforce the schema.
 *
 * Both ops also require `cu`: the number of sequences is cu's length minus one and nothing else
 * carries it. The fast kernels read `cu[n]` before any null test, so omitting it would fault.
 */
#ifndef RAD_REF_OPS_H
#define RAD_REF_OPS_H

#include "rad_abi.h"

extern "C" {

/* ---- elementwise and norms */
int ref_rmsnorm            (const RadArgs*, RadStream);
int ref_rmsnorm_p          (const RadArgs*, RadStream);
int ref_rmsnorm_add        (const RadArgs*, RadStream);
int ref_hc_enter           (const RadArgs*, RadStream);
int ref_mhc_pre            (const RadArgs*, RadStream);
int ref_mhc_pre_partial    (const RadArgs*, RadStream);
int ref_mhc_pre_finish     (const RadArgs*, RadStream);
int ref_mhc_post           (const RadArgs*, RadStream);
int ref_mhc_contract       (const RadArgs*, RadStream);
int ref_kda_update         (const RadArgs*, RadStream);
int ref_hc_read            (const RadArgs*, RadStream);
int ref_hc_write           (const RadArgs*, RadStream);
int ref_hc_project         (const RadArgs*, RadStream);
int ref_mtp_enter          (const RadArgs*, RadStream);
int ref_layernorm          (const RadArgs*, RadStream);
int ref_grid_embed         (const RadArgs*, RadStream);
int ref_grid_embed_aa      (const RadArgs*, RadStream);
int ref_add                (const RadArgs*, RadStream);
int ref_mul                (const RadArgs*, RadStream);
int ref_silu_mul           (const RadArgs*, RadStream);
int ref_swiglu_clamped     (const RadArgs*, RadStream);
int ref_router_sigmoid_topk(const RadArgs*, RadStream);
int ref_dsa_pool(const RadArgs*, RadStream);
int ref_dsa_query(const RadArgs*, RadStream);
int ref_gemm_nt_f32(const RadArgs*, RadStream);
int ref_mla_project(const RadArgs*, RadStream);
int ref_dsa_score(const RadArgs*, RadStream);
int ref_mla_attn(const RadArgs*, RadStream);
int ref_gelu_mul           (const RadArgs*, RadStream);
int ref_act_mul            (const RadArgs*, RadStream);
int ref_mul_weight         (const RadArgs*, RadStream);
int ref_seg_mean           (const RadArgs*, RadStream);
int ref_pool_rows          (const RadArgs*, RadStream);
int ref_relu               (const RadArgs*, RadStream);
int ref_clamp              (const RadArgs*, RadStream);
int ref_conv2d             (const RadArgs*, RadStream);
int ref_conv2d_unfold      (const RadArgs*, RadStream);
int ref_conv1d_dw          (const RadArgs*, RadStream);
int ref_rel_sinusoid       (const RadArgs*, RadStream);
int ref_seg_sinusoid       (const RadArgs*, RadStream);
int ref_attn_local_rel     (const RadArgs*, RadStream);
int ref_gelu               (const RadArgs*, RadStream);
int ref_silu               (const RadArgs*, RadStream);
int ref_sigmoid            (const RadArgs*, RadStream);
int ref_logit_softcap      (const RadArgs*, RadStream);
int ref_scale              (const RadArgs*, RadStream);
int ref_fill               (const RadArgs*, RadStream);
int ref_gather_rows        (const RadArgs*, RadStream);
int ref_scatter_rows       (const RadArgs*, RadStream);
int ref_ngram_ids          (const RadArgs*, RadStream);
int ref_ple_gate           (const RadArgs*, RadStream);
int ref_ple_conv           (const RadArgs*, RadStream);
int ref_causal_conv1d      (const RadArgs*, RadStream);
int ref_cast               (const RadArgs*, RadStream);
int ref_softmax            (const RadArgs*, RadStream);

/* ---- quantisation and rotation */
int ref_quant_act_i8       (const RadArgs*, RadStream);
int ref_quant_act_i8g      (const RadArgs*, RadStream);
int ref_had_quant_act_i8   (const RadArgs*, RadStream);
int ref_quant_act_fp8      (const RadArgs*, RadStream);
int ref_quant_cols_fp8     (const RadArgs*, RadStream);
int ref_had_quant_act_fp8  (const RadArgs*, RadStream);
int ref_had_quant_act_i8g  (const RadArgs*, RadStream);
int ref_givens_dense_quant(const RadArgs*, RadStream);
int ref_givens_rotate(const RadArgs*, RadStream);
int ref_rmsnorm_quant_fp8  (const RadArgs*, RadStream);
int ref_gated_quant_fp8    (const RadArgs*, RadStream);
int ref_gemm_nt_q_gated    (const RadArgs*, RadStream);
int ref_dflash_select      (const RadArgs*, RadStream);
int ref_gram_accum         (const RadArgs*, RadStream);
int ref_rmsnorm_had_quant_i8(const RadArgs*, RadStream);
int ref_gated_had_quant_i8 (const RadArgs*, RadStream);
int ref_dequant            (const RadArgs*, RadStream);

/* ---- gemm and the vocabulary edges */
int ref_gemm_nt            (const RadArgs*, RadStream);
int ref_gemm_nt_bias       (const RadArgs*, RadStream);
int ref_gemm_nt_q          (const RadArgs*, RadStream);
int ref_gemm_nt_q_bias     (const RadArgs*, RadStream);
/* Runtime LoRA's two halves, per-row adapter slot (ref_lora.cpp). */
int ref_lora_shrink        (const RadArgs*, RadStream);
int ref_lora_expand        (const RadArgs*, RadStream);
int ref_lora_expand_mm     (const RadArgs*, RadStream);
int ref_embed_lookup       (const RadArgs*, RadStream);
int ref_embed_lookup_q     (const RadArgs*, RadStream);
int ref_logits_gemm        (const RadArgs*, RadStream);

/* ---- attention */
int ref_attn_paged         (const RadArgs*, RadStream);
int ref_attn_paged_tree    (const RadArgs*, RadStream);
int ref_attn_dense         (const RadArgs*, RadStream);
int ref_attn_cross         (const RadArgs*, RadStream);
int ref_table_add          (const RadArgs*, RadStream);
int ref_mix_rows           (const RadArgs*, RadStream);
int ref_chan_stats         (const RadArgs*, RadStream);
int ref_kv_store           (const RadArgs*, RadStream);
int ref_rope               (const RadArgs*, RadStream);
int ref_rope_table         (const RadArgs*, RadStream);
int ref_qk_norm_rope       (const RadArgs*, RadStream);
/* image generation (ref_gen.cpp) */
int ref_ln_mod(const RadArgs*, RadStream);
int ref_ln_mod_quant_fp8(const RadArgs*, RadStream);
int ref_gate_add(const RadArgs*, RadStream);
int ref_timestep_embed(const RadArgs*, RadStream);
int ref_gather_rows2(const RadArgs*, RadStream);
int ref_lane_gather(const RadArgs*, RadStream);
int ref_lane_store(const RadArgs*, RadStream);
int ref_randn_lanes(const RadArgs*, RadStream);
int ref_flow_euler(const RadArgs*, RadStream);
int ref_kd_input(const RadArgs*, RadStream);
int ref_kd_step(const RadArgs*, RadStream);
int ref_rms_mod(const RadArgs*, RadStream);
int ref_rms_mod_quant_fp8(const RadArgs*, RadStream);
int ref_flow_x0_step(const RadArgs*, RadStream);
/* ref_video.cpp */
int ref_patch_unfold(const RadArgs*, RadStream);
int ref_vtile_gather(const RadArgs*, RadStream);
int ref_vtile_stitch(const RadArgs*, RadStream);
int ref_grid_pos3(const RadArgs*, RadStream);
int ref_patch_fold(const RadArgs*, RadStream);
int ref_conv3d_ndhwc(const RadArgs*, RadStream);
int ref_group_norm(const RadArgs*, RadStream);
int ref_tconv_dw(const RadArgs*, RadStream);
int ref_resize_bilinear(const RadArgs*, RadStream);
int ref_flow_renoise(const RadArgs*, RadStream);
/* ref_ltx.cpp */
int ref_rope_split(const RadArgs*, RadStream);
int ref_head_gate(const RadArgs*, RadStream);
int ref_ltx_guide(const RadArgs*, RadStream);
int ref_pixel_norm(const RadArgs*, RadStream);
int ref_d2s3d(const RadArgs*, RadStream);
int ref_s2d3d(const RadArgs*, RadStream);
int ref_pix_patch(const RadArgs*, RadStream);
int ref_patch_pix8(const RadArgs*, RadStream);
int ref_vol_index(const RadArgs*, RadStream);
int ref_attn_na3d(const RadArgs*, RadStream);
int ref_tile_grid(const RadArgs*, RadStream);
int ref_tile_crop(const RadArgs*, RadStream);
int ref_tile_stitch(const RadArgs*, RadStream);
int ref_stft_logmel(const RadArgs*, RadStream);
int ref_bwe_up(const RadArgs*, RadStream);
/* ltx_guide's scratch, in f32 values: the deviations' partial sums of a device implementation. */
enum { kLtxGuideStats = 8200 };
/* ref_vsa.cpp */
int ref_vsa_select(const RadArgs*, RadStream);
int ref_attn_vsa(const RadArgs*, RadStream);
/* ref_audio.cpp */
int ref_conv1d(const RadArgs*, RadStream);
int ref_conv_t1d(const RadArgs*, RadStream);
int ref_snake(const RadArgs*, RadStream);
int ref_snake_aa(const RadArgs*, RadStream);
int ref_cfg_topk_sample(const RadArgs*, RadStream);
int ref_rvq_embed(const RadArgs*, RadStream);
int ref_frame_mix_store(const RadArgs*, RadStream);
int ref_end_status(const RadArgs*, RadStream);
int ref_frame_store(const RadArgs*, RadStream);
int ref_resample_nearest(const RadArgs*, RadStream);
int ref_fourier_embed(const RadArgs*, RadStream);
int ref_win_input(const RadArgs*, RadStream);
int ref_win_prep(const RadArgs*, RadStream);
int ref_win_finish(const RadArgs*, RadStream);
int ref_chan_split(const RadArgs*, RadStream);
int ref_interleave2(const RadArgs*, RadStream);
int ref_iota(const RadArgs*, RadStream);
int ref_cu_repeat(const RadArgs*, RadStream);
int ref_int_affine(const RadArgs*, RadStream);
int ref_chan_affine(const RadArgs*, RadStream);
int ref_conv2d_nhwc(const RadArgs*, RadStream);
int ref_im2col_nhwc(const RadArgs*, RadStream);
int ref_l2norm_chan(const RadArgs*, RadStream);
int ref_d2s_add(const RadArgs*, RadStream);
int ref_s2d_avg_add(const RadArgs*, RadStream);
int ref_to_rgba8(const RadArgs*, RadStream);
int ref_gelu_tanh(const RadArgs*, RadStream);
int ref_gelu_quick(const RadArgs*, RadStream);
int ref_scatter_add_rows(const RadArgs*, RadStream);
int ref_attn_prefix_self(const RadArgs*, RadStream);
int ref_qsa_block_key      (const RadArgs*, RadStream);
/* the QSA indexer and DFlash2's convolution (ref_qsa.cpp) */
int ref_qsa_score          (const RadArgs*, RadStream);
int ref_qsa_select         (const RadArgs*, RadStream);
int ref_qsa_tail_store     (const RadArgs*, RadStream);
int ref_qsa_work           (const RadArgs*, RadStream);
int ref_dflash_conv        (const RadArgs*, RadStream);

/* ---- mixture of experts */
int ref_router_topk        (const RadArgs*, RadStream);
int ref_moe_scatter        (const RadArgs*, RadStream);
int ref_moe_gemm           (const RadArgs*, RadStream);
int ref_moe_gemm_q         (const RadArgs*, RadStream);
int ref_moe_gemm_q_mod4    (const RadArgs*, RadStream);
/* The layout hooks of moe_gemm_q and moe_gemm_q_mod4: a codebook expert's views of several planes
 * in the joined form of abi/rad_planes.h, checked against `dtype` (ref_moe.cpp). */
int ref_moe_cb_layout      (const RadParam*, int, int, const RadEncoding*, const int*,
                            const RadTensor*, int, RadLayout*);
int ref_moe_cb_relayout    (const RadParam*, int, int, const RadEncoding*, const int*,
                            const RadTensor*, int, void*, int64_t);
int ref_moe_cb_unrelayout  (const RadParam*, int, int, const RadEncoding*, const int*,
                            const void*, int64_t, const RadTensor*, int);
int ref_scale_rows         (const RadArgs*, RadStream);
int ref_moe_gather         (const RadArgs*, RadStream);
int ref_row_topk           (const RadArgs*, RadStream);
int ref_row_topk_merge     (const RadArgs*, RadStream);
int ref_pair_argmax        (const RadArgs*, RadStream);
int ref_logit_rerank       (const RadArgs*, RadStream);

/* ---- gated delta net */
int ref_gdn_conv_prep      (const RadArgs*, RadStream);
int ref_gdn_conv_update    (const RadArgs*, RadStream);
int ref_gdn_kkt_solve      (const RadArgs*, RadStream);
int ref_gdn_chunk_scan     (const RadArgs*, RadStream);
int ref_gdn_recurrent_update(const RadArgs*, RadStream);
int ref_gdn_conv_recurrent_update(const RadArgs*, RadStream);
int ref_gdn_gated_rmsnorm  (const RadArgs*, RadStream);

/* ---- collectives */
int     ref_all_reduce     (const RadArgs*, RadStream);
int     ref_all_gather     (const RadArgs*, RadStream);
int64_t ref_all_reduce_scratch(const RadArgs*);
int     ref_collective_init(const RadParam*, int, int, int, void**);
void    ref_collective_fini(void*);

/* ---- sampling */
int ref_sample_penalties   (const RadArgs*, RadStream);
int ref_sample_dry         (const RadArgs*, RadStream);
int ref_sample_temp        (const RadArgs*, RadStream);
int ref_sample_topk        (const RadArgs*, RadStream);
int ref_sample_topp        (const RadArgs*, RadStream);
int ref_sample_minp        (const RadArgs*, RadStream);
int ref_sample_typical     (const RadArgs*, RadStream);
int ref_sample_xtc         (const RadArgs*, RadStream);
int ref_sample_merge_topk  (const RadArgs*, RadStream);
int ref_sample_mask        (const RadArgs*, RadStream);
int ref_sample_pick        (const RadArgs*, RadStream);
int ref_sample_argmax      (const RadArgs*, RadStream);
int ref_sample_logprobs    (const RadArgs*, RadStream);

/* ============================== Operand description ==============================
 *
 * `RadKernelInfo::opd_shape` (rad_abi.h): the size of operand i at this geometry and what it may
 * legally hold. libref/ref_shapes.cpp implements one per entry point above. The naming is
 * load-bearing: the hook does not receive the op name, so ref_registry.cpp's ROW() macro pastes
 * `fn ## _shape` from the launch function, and an op added without a description fails to link.
 *
 * `p`/`n_p` is the resolved parameter list as `launch` receives it, `operand` the positional index
 * into the op's schema, and the result goes to `out`. Returns RAD_OK, or RAD_E_UNSUPPORTED when the
 * geometry lacks what the description needs; callers skip and count those rather than guess, since
 * an invented extent tests bounds handling, not correctness. */
#define REF_SHAPE_FN(fn) \
    int fn##_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out)

REF_SHAPE_FN(ref_rmsnorm);
REF_SHAPE_FN(ref_rmsnorm_p);
REF_SHAPE_FN(ref_rmsnorm_add);
REF_SHAPE_FN(ref_hc_enter);
REF_SHAPE_FN(ref_mhc_pre);
REF_SHAPE_FN(ref_mhc_pre_partial);
REF_SHAPE_FN(ref_mhc_pre_finish);
REF_SHAPE_FN(ref_mhc_post);
REF_SHAPE_FN(ref_mhc_contract);
REF_SHAPE_FN(ref_kda_update);
REF_SHAPE_FN(ref_hc_read);
REF_SHAPE_FN(ref_hc_write);
REF_SHAPE_FN(ref_hc_project);
REF_SHAPE_FN(ref_mtp_enter);
REF_SHAPE_FN(ref_layernorm);
REF_SHAPE_FN(ref_grid_embed);
REF_SHAPE_FN(ref_grid_embed_aa);
REF_SHAPE_FN(ref_add);
REF_SHAPE_FN(ref_mul);
REF_SHAPE_FN(ref_silu_mul);
REF_SHAPE_FN(ref_swiglu_clamped);
REF_SHAPE_FN(ref_router_sigmoid_topk);
REF_SHAPE_FN(ref_dsa_pool);
REF_SHAPE_FN(ref_dsa_query);
REF_SHAPE_FN(ref_gemm_nt_f32);
REF_SHAPE_FN(ref_mla_project);
REF_SHAPE_FN(ref_dsa_score);
REF_SHAPE_FN(ref_mla_attn);
REF_SHAPE_FN(ref_gelu_mul);
REF_SHAPE_FN(ref_act_mul);
REF_SHAPE_FN(ref_mul_weight);
REF_SHAPE_FN(ref_seg_mean);
REF_SHAPE_FN(ref_pool_rows);
REF_SHAPE_FN(ref_relu);
REF_SHAPE_FN(ref_clamp);
REF_SHAPE_FN(ref_conv2d);
REF_SHAPE_FN(ref_conv2d_unfold);
REF_SHAPE_FN(ref_conv1d_dw);
REF_SHAPE_FN(ref_rel_sinusoid);
REF_SHAPE_FN(ref_seg_sinusoid);
REF_SHAPE_FN(ref_attn_local_rel);
REF_SHAPE_FN(ref_gelu);
REF_SHAPE_FN(ref_silu);
REF_SHAPE_FN(ref_sigmoid);
REF_SHAPE_FN(ref_logit_softcap);
REF_SHAPE_FN(ref_scale);
REF_SHAPE_FN(ref_fill);
REF_SHAPE_FN(ref_gather_rows);
REF_SHAPE_FN(ref_scatter_rows);
REF_SHAPE_FN(ref_ngram_ids);
REF_SHAPE_FN(ref_ple_gate);
REF_SHAPE_FN(ref_ple_conv);
REF_SHAPE_FN(ref_causal_conv1d);
REF_SHAPE_FN(ref_cast);
REF_SHAPE_FN(ref_softmax);

REF_SHAPE_FN(ref_quant_act_i8);
REF_SHAPE_FN(ref_quant_act_i8g);
REF_SHAPE_FN(ref_had_quant_act_i8);
REF_SHAPE_FN(ref_quant_act_fp8);
REF_SHAPE_FN(ref_quant_cols_fp8);
REF_SHAPE_FN(ref_had_quant_act_fp8);
REF_SHAPE_FN(ref_had_quant_act_i8g);
REF_SHAPE_FN(ref_givens_dense_quant);
REF_SHAPE_FN(ref_givens_rotate);
REF_SHAPE_FN(ref_rmsnorm_quant_fp8);
REF_SHAPE_FN(ref_gated_quant_fp8);
REF_SHAPE_FN(ref_gemm_nt_q_gated);
REF_SHAPE_FN(ref_dflash_select);
REF_SHAPE_FN(ref_gram_accum);
REF_SHAPE_FN(ref_rmsnorm_had_quant_i8);
REF_SHAPE_FN(ref_gated_had_quant_i8);
REF_SHAPE_FN(ref_dequant);

REF_SHAPE_FN(ref_gemm_nt);
REF_SHAPE_FN(ref_gemm_nt_bias);
REF_SHAPE_FN(ref_gemm_nt_q);
REF_SHAPE_FN(ref_gemm_nt_q_bias);
REF_SHAPE_FN(ref_lora_shrink);
REF_SHAPE_FN(ref_lora_expand);
REF_SHAPE_FN(ref_lora_expand_mm);
REF_SHAPE_FN(ref_embed_lookup);
REF_SHAPE_FN(ref_embed_lookup_q);
REF_SHAPE_FN(ref_logits_gemm);

REF_SHAPE_FN(ref_attn_paged);
REF_SHAPE_FN(ref_attn_paged_tree);
REF_SHAPE_FN(ref_attn_dense);
REF_SHAPE_FN(ref_attn_cross);
REF_SHAPE_FN(ref_table_add);
REF_SHAPE_FN(ref_mix_rows);
REF_SHAPE_FN(ref_chan_stats);
REF_SHAPE_FN(ref_kv_store);
REF_SHAPE_FN(ref_rope);
REF_SHAPE_FN(ref_rope_table);
REF_SHAPE_FN(ref_qk_norm_rope);
REF_SHAPE_FN(ref_ln_mod);
REF_SHAPE_FN(ref_ln_mod_quant_fp8);
REF_SHAPE_FN(ref_gate_add);
REF_SHAPE_FN(ref_timestep_embed);
REF_SHAPE_FN(ref_gather_rows2);
REF_SHAPE_FN(ref_lane_gather);
REF_SHAPE_FN(ref_lane_store);
REF_SHAPE_FN(ref_randn_lanes);
REF_SHAPE_FN(ref_flow_euler);
REF_SHAPE_FN(ref_kd_input);
REF_SHAPE_FN(ref_kd_step);
REF_SHAPE_FN(ref_rms_mod);
REF_SHAPE_FN(ref_rms_mod_quant_fp8);
REF_SHAPE_FN(ref_flow_x0_step);
REF_SHAPE_FN(ref_patch_unfold);
REF_SHAPE_FN(ref_vtile_gather);
REF_SHAPE_FN(ref_vtile_stitch);
REF_SHAPE_FN(ref_grid_pos3);
REF_SHAPE_FN(ref_patch_fold);
REF_SHAPE_FN(ref_conv3d_ndhwc);
REF_SHAPE_FN(ref_group_norm);
REF_SHAPE_FN(ref_tconv_dw);
REF_SHAPE_FN(ref_resize_bilinear);
REF_SHAPE_FN(ref_flow_renoise);
REF_SHAPE_FN(ref_rope_split);
REF_SHAPE_FN(ref_head_gate);
REF_SHAPE_FN(ref_ltx_guide);
REF_SHAPE_FN(ref_pixel_norm);
REF_SHAPE_FN(ref_d2s3d);
REF_SHAPE_FN(ref_s2d3d);
REF_SHAPE_FN(ref_pix_patch);
REF_SHAPE_FN(ref_patch_pix8);
REF_SHAPE_FN(ref_vol_index);
REF_SHAPE_FN(ref_attn_na3d);
REF_SHAPE_FN(ref_tile_grid);
REF_SHAPE_FN(ref_tile_crop);
REF_SHAPE_FN(ref_tile_stitch);
REF_SHAPE_FN(ref_stft_logmel);
REF_SHAPE_FN(ref_bwe_up);
REF_SHAPE_FN(ref_vsa_select);
REF_SHAPE_FN(ref_attn_vsa);
REF_SHAPE_FN(ref_conv1d);
REF_SHAPE_FN(ref_conv_t1d);
REF_SHAPE_FN(ref_snake);
REF_SHAPE_FN(ref_snake_aa);
REF_SHAPE_FN(ref_cfg_topk_sample);
REF_SHAPE_FN(ref_rvq_embed);
REF_SHAPE_FN(ref_frame_mix_store);
REF_SHAPE_FN(ref_end_status);
REF_SHAPE_FN(ref_frame_store);
REF_SHAPE_FN(ref_resample_nearest);
REF_SHAPE_FN(ref_fourier_embed);
REF_SHAPE_FN(ref_win_input);
REF_SHAPE_FN(ref_win_prep);
REF_SHAPE_FN(ref_win_finish);
REF_SHAPE_FN(ref_chan_split);
REF_SHAPE_FN(ref_interleave2);
REF_SHAPE_FN(ref_iota);
REF_SHAPE_FN(ref_cu_repeat);
REF_SHAPE_FN(ref_int_affine);
REF_SHAPE_FN(ref_chan_affine);
REF_SHAPE_FN(ref_conv2d_nhwc);
REF_SHAPE_FN(ref_im2col_nhwc);
REF_SHAPE_FN(ref_l2norm_chan);
REF_SHAPE_FN(ref_d2s_add);
REF_SHAPE_FN(ref_s2d_avg_add);
REF_SHAPE_FN(ref_to_rgba8);
REF_SHAPE_FN(ref_gelu_tanh);
REF_SHAPE_FN(ref_gelu_quick);
REF_SHAPE_FN(ref_scatter_add_rows);
REF_SHAPE_FN(ref_attn_prefix_self);
REF_SHAPE_FN(ref_qsa_block_key);
REF_SHAPE_FN(ref_qsa_score);
REF_SHAPE_FN(ref_qsa_select);
REF_SHAPE_FN(ref_qsa_tail_store);
REF_SHAPE_FN(ref_qsa_work);
REF_SHAPE_FN(ref_dflash_conv);

REF_SHAPE_FN(ref_router_topk);
REF_SHAPE_FN(ref_moe_scatter);
REF_SHAPE_FN(ref_moe_gemm);
REF_SHAPE_FN(ref_moe_gemm_q);
REF_SHAPE_FN(ref_moe_gemm_q_mod4);
REF_SHAPE_FN(ref_scale_rows);
REF_SHAPE_FN(ref_moe_gather);
REF_SHAPE_FN(ref_row_topk);
REF_SHAPE_FN(ref_row_topk_merge);
REF_SHAPE_FN(ref_pair_argmax);
REF_SHAPE_FN(ref_logit_rerank);

REF_SHAPE_FN(ref_gdn_conv_prep);
REF_SHAPE_FN(ref_gdn_conv_update);
REF_SHAPE_FN(ref_gdn_kkt_solve);
REF_SHAPE_FN(ref_gdn_chunk_scan);
REF_SHAPE_FN(ref_gdn_recurrent_update);
REF_SHAPE_FN(ref_gdn_conv_recurrent_update);
REF_SHAPE_FN(ref_gdn_gated_rmsnorm);

REF_SHAPE_FN(ref_all_reduce);
REF_SHAPE_FN(ref_all_gather);

REF_SHAPE_FN(ref_sample_penalties);
REF_SHAPE_FN(ref_sample_dry);
REF_SHAPE_FN(ref_sample_temp);
REF_SHAPE_FN(ref_sample_topk);
REF_SHAPE_FN(ref_sample_topp);
REF_SHAPE_FN(ref_sample_minp);
REF_SHAPE_FN(ref_sample_typical);
REF_SHAPE_FN(ref_sample_xtc);
REF_SHAPE_FN(ref_sample_merge_topk);
REF_SHAPE_FN(ref_sample_mask);
REF_SHAPE_FN(ref_sample_pick);
REF_SHAPE_FN(ref_sample_argmax);
REF_SHAPE_FN(ref_sample_logprobs);

}  /* extern "C" */
#endif /* RAD_REF_OPS_H */
