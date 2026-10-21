/* ref_ops.h -- the entry points ref_registry.cpp puts in the kernel table, and the layout
 * conventions they share.
 *
 * ============================== SHAPES ARE READ OFF THE OPERANDS ==============================
 *
 * Every extent this plugin loops over comes from a RadTensor's shape, never from the op's
 * parameters. That is a deliberate rule and not laziness:
 *
 *   * A RANGED parameter is collapsed to a BAND when the instance is resolved (spec §2.2), and
 *     `rad_issue` carries the actual value as an integer rather than by rewriting the param list.
 *     A loop bounded by `M` would therefore run to the band's upper bound -- 8192 rows of a
 *     single-row decode -- and read off the end of the arena. Ref takes M from the tensor.
 *   * Everything else the parameters name is either derivable from a shape (head_dim, N, K, n)
 *     or is genuinely not a shape at all (causal, window, block_size, eps, group, top_k). Ref
 *     uses the parameter for the second kind and cross-checks it against the shape for the
 *     first, refusing with RAD_E_SHAPE where a disagreement would mean an out-of-bounds read.
 *
 * ============================== THE PAGED KV LAYOUT ==============================
 *
 * This is libr4d's and every kernel in the project has to match it, so it is written here once:
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]
 *
 * K first and V second inside a slot: element d of the key is at [..., d] and element d of the
 * value at [..., head_dim + d]. A flat slot index s decomposes as block = s / block_size and
 * offset = s % block_size, which is what `slot_mapping` carries and what `block_table` indexes
 * into. Nothing else in the file is allowed to invent a different one.
 *
 * ============================== THE SPECULATIVE ROLLBACK CONTRACT ==============================
 *
 * The linear-attention pair (gdn_conv_update, gdn_recurrent_update) must support rejection
 * without recompute (spec §10), and the contract is part of the schema rather than an assumption:
 *
 *   conv state   [n_slots, conv_dim, state_len], state_len >= conv_width - 2 + q_len.
 *                A step READS its window at offset num_accepted[s] - 1 -- the slot the last
 *                accepted token left -- and REWRITES the buffer shifted down by one: slots
 *                0 .. conv_width-3 take the tail of the old history, and this step's token t
 *                lands at slot t + conv_width - 2. The invariant that makes it work is that the
 *                last accepted token of a step always sits at the slot its own index implies, so
 *                the next step's read offset is the only thing that changes.
 *   linear state [n_slots, n_head_v, head_v, head_k], addressed through state_idx[s, 0..q_len).
 *                The initial state is read from state_idx[s, num_accepted[s] - 1] and EVERY
 *                candidate token t writes its own state to state_idx[s, t], because which of them
 *                survives verification is not known until after this layer has run.
 *
 * num_accepted counts the accepted tokens of the PREVIOUS step and is 1-based: 1 means "none of
 * the drafts survived, only the token that was already committed". It is a REQUIRED operand on
 * both ops -- an op that lets the caller omit it is an op whose rollback contract is optional --
 * and a non-speculative decode passes all ones. Ref still reads a null as 1 rather than faulting,
 * because a tool calling the launch directly has no declare in front of it to enforce the schema.
 *
 * BOTH OPS ALSO TAKE `cu`, and it is required. The two decode kernels run off the one cumulative-
 * lengths tensor on the one step; the number of sequences is cu's length minus one and nothing
 * else in either argument list carries it. The fast kernels read `cu[n]` before any null test, so
 * a schema that omitted it would not make it optional -- it would make the launch fault.
 */
#ifndef RAD_REF_OPS_H
#define RAD_REF_OPS_H

#include "rad_abi.h"

extern "C" {

/* ---- elementwise and norms */
int ref_rmsnorm            (const RadArgs*, RadStream);
int ref_rmsnorm_add        (const RadArgs*, RadStream);
int ref_hc_enter           (const RadArgs*, RadStream);
int ref_hc_read            (const RadArgs*, RadStream);
int ref_hc_write           (const RadArgs*, RadStream);
int ref_hc_mean            (const RadArgs*, RadStream);
int ref_mtp_enter          (const RadArgs*, RadStream);
int ref_layernorm          (const RadArgs*, RadStream);
int ref_grid_embed         (const RadArgs*, RadStream);
int ref_add                (const RadArgs*, RadStream);
int ref_mul                (const RadArgs*, RadStream);
int ref_silu_mul           (const RadArgs*, RadStream);
int ref_gelu               (const RadArgs*, RadStream);
int ref_silu               (const RadArgs*, RadStream);
int ref_sigmoid            (const RadArgs*, RadStream);
int ref_gather_rows        (const RadArgs*, RadStream);
int ref_scatter_rows       (const RadArgs*, RadStream);
int ref_ngram_ids          (const RadArgs*, RadStream);
int ref_ple_gate           (const RadArgs*, RadStream);
int ref_ple_conv           (const RadArgs*, RadStream);
int ref_cast               (const RadArgs*, RadStream);
int ref_softmax            (const RadArgs*, RadStream);

/* ---- quantisation and rotation */
int ref_quant_act_i8       (const RadArgs*, RadStream);
int ref_had_quant_act_i8   (const RadArgs*, RadStream);
int ref_quant_act_fp8      (const RadArgs*, RadStream);
int ref_had_quant_act_fp8  (const RadArgs*, RadStream);
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
int ref_embed_lookup       (const RadArgs*, RadStream);
int ref_embed_lookup_q     (const RadArgs*, RadStream);
int ref_logits_gemm        (const RadArgs*, RadStream);

/* ---- attention */
int ref_attn_paged         (const RadArgs*, RadStream);
int ref_attn_dense         (const RadArgs*, RadStream);
int ref_kv_store           (const RadArgs*, RadStream);
int ref_rope               (const RadArgs*, RadStream);
int ref_rope_table         (const RadArgs*, RadStream);
int ref_qk_norm_rope       (const RadArgs*, RadStream);

/* ---- mixture of experts */
int ref_router_topk        (const RadArgs*, RadStream);
int ref_moe_scatter        (const RadArgs*, RadStream);
int ref_moe_gemm           (const RadArgs*, RadStream);
int ref_moe_gemm_q         (const RadArgs*, RadStream);
int ref_moe_gemm_q_mod4    (const RadArgs*, RadStream);
int ref_scale_rows         (const RadArgs*, RadStream);
int ref_moe_gather         (const RadArgs*, RadStream);
int ref_row_topk           (const RadArgs*, RadStream);
int ref_row_topk_merge     (const RadArgs*, RadStream);
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

/* ============================== OPERAND DESCRIPTION ==============================
 *
 * `RadKernelInfo::opd_shape` (rad_abi.h): how big is operand i of this op at this geometry, and
 * what may legally be in it. libref/ref_shapes.cpp implements one for every entry point above,
 * and the NAMING IS PART OF THE MECHANISM rather than a convention: the hook does not receive the
 * op name, so a single shared function would have to recover it from a parameter and no op in this
 * vocabulary carries its own name as one. Instead ref_registry.cpp's ROW() macro pastes the shape
 * hook out of the launch it was already given -- `fn ## _shape` -- so every row gets its
 * description without a row being edited, and an op added later that forgets to describe itself
 * does not link.
 *
 * The signature is spelled once here because it is the same for every entry point above, and the
 * parameter names are what document it: `p`/`n_p` is the resolved parameter list exactly as
 * `launch` would receive it, `operand` is the positional index into the op's schema, and the
 * answer lands in `out`. RAD_OK, or RAD_E_UNSUPPORTED when the geometry
 * does not carry what the description needs -- which a caller SKIPS AND COUNTS rather than
 * guessing, because a tool that invents an extent tests the kernel's bounds handling and calls the
 * result correctness. */
#define REF_SHAPE_FN(fn) \
    int fn##_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out)

REF_SHAPE_FN(ref_rmsnorm);
REF_SHAPE_FN(ref_rmsnorm_add);
REF_SHAPE_FN(ref_hc_enter);
REF_SHAPE_FN(ref_hc_read);
REF_SHAPE_FN(ref_hc_write);
REF_SHAPE_FN(ref_hc_mean);
REF_SHAPE_FN(ref_mtp_enter);
REF_SHAPE_FN(ref_layernorm);
REF_SHAPE_FN(ref_grid_embed);
REF_SHAPE_FN(ref_add);
REF_SHAPE_FN(ref_mul);
REF_SHAPE_FN(ref_silu_mul);
REF_SHAPE_FN(ref_gelu);
REF_SHAPE_FN(ref_silu);
REF_SHAPE_FN(ref_sigmoid);
REF_SHAPE_FN(ref_gather_rows);
REF_SHAPE_FN(ref_scatter_rows);
REF_SHAPE_FN(ref_ngram_ids);
REF_SHAPE_FN(ref_ple_gate);
REF_SHAPE_FN(ref_ple_conv);
REF_SHAPE_FN(ref_cast);
REF_SHAPE_FN(ref_softmax);

REF_SHAPE_FN(ref_quant_act_i8);
REF_SHAPE_FN(ref_had_quant_act_i8);
REF_SHAPE_FN(ref_quant_act_fp8);
REF_SHAPE_FN(ref_had_quant_act_fp8);
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
REF_SHAPE_FN(ref_embed_lookup);
REF_SHAPE_FN(ref_embed_lookup_q);
REF_SHAPE_FN(ref_logits_gemm);

REF_SHAPE_FN(ref_attn_paged);
REF_SHAPE_FN(ref_attn_dense);
REF_SHAPE_FN(ref_kv_store);
REF_SHAPE_FN(ref_rope);
REF_SHAPE_FN(ref_rope_table);
REF_SHAPE_FN(ref_qk_norm_rope);

REF_SHAPE_FN(ref_router_topk);
REF_SHAPE_FN(ref_moe_scatter);
REF_SHAPE_FN(ref_moe_gemm);
REF_SHAPE_FN(ref_moe_gemm_q);
REF_SHAPE_FN(ref_moe_gemm_q_mod4);
REF_SHAPE_FN(ref_scale_rows);
REF_SHAPE_FN(ref_moe_gather);
REF_SHAPE_FN(ref_row_topk);
REF_SHAPE_FN(ref_row_topk_merge);
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

}  /* extern "C" */
#endif /* RAD_REF_OPS_H */
