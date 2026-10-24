/* r4d_plugin.h -- what libr4d's translation units share.
 *
 * The row table (r4d_rows.cpp) and the weight layouts (r4d_layout.cpp) are PURE HOST C++ and
 * include only this file and the ABI; the launch shims include r4d.h and therefore
 * <hip/hip_runtime.h>. That split is deliberate: it is what lets the constraint table -- the part
 * a transcription error hides in -- be compiled and read on a machine with no ROCm, and it is why
 * every shim is declared here rather than in the file that defines it.
 *
 * Nothing in this header may include r4d.h.
 */
#ifndef RAD_R4D_PLUGIN_H
#define RAD_R4D_PLUGIN_H

#include "r4d_args.h"


/* ================================================================== the priority ladder
 *
 * The selector ranks candidate rows by an integer priority (spec §2.1). Where two rows of one op
 * can both match a geometry, the value is what decides which serves it; each group below states
 * the reason its order is what it is.
 *
 * Values are spaced by ten so a row can be slipped between two of them without renumbering.
 */
enum {
    /* A two-shot row's constraints are a SUPERSET of the one-shot's at the same world_size, so a
     * two-shot that ranked below could never be reached: every query it serves the one-shot also
     * matches. Its lower bound on `numel` is what keeps a small message on the one-shot, and the
     * priority is what hands a large one to the two-shot. */
    R4D_PRIO_AR_TWOSHOT   = 70,
    R4D_PRIO_AR_ONESHOT   = 60,

    /* One op (`attn_paged`) banded on q_len, so the decode row's bound -- q_len<=10 at h256,
     * q_len<=16 at h128 -- is a bucket boundary and not a partition: inside the low band BOTH rows
     * match. The decode row therefore has to outrank the prefill one, which is the same choice a
     * caller selecting an entry point by hand would make. */
    R4D_PRIO_ATTN_DECODE  = 70,
    R4D_PRIO_ATTN_PREFILL = 60,

    /* At M<=16 with K%256==0 both bf16 skinny rows match and the m16 (dot2) kernel is taken.
     * The m64 WMMA kernel is faster on every shape measured, so a caller that wants it at low M
     * asks with M=64 to exclude m16. */
    R4D_PRIO_GEMM_M16     = 60,
    R4D_PRIO_GEMM_SKINNY  = 50,
    /* The skinny m64 row has no upper bound on M, so the tiled bf16 row -- M > 64 only -- has to
     * outrank it to be reached, and stays below m16 whose band it never enters. */
    R4D_PRIO_GEMM_BF16_TILED = 55,

    /* The bands already separate the tiled and prefill GEMMs from the skinny kernels -- skinny is
     * M<=64, these are unbounded -- so this pair only decides the M>64 band. `pf` is the newer
     * kernel and its tuned picks fall back to the tiled one where it is absent, so it outranks.
     * WHAT THIS CANNOT EXPRESS: the optimum crosses BETWEEN the two kernels at a shape-dependent
     * M, and the tuning axis is variants WITHIN a kernel, so a per-shape crossover has no home
     * here. */
    R4D_PRIO_GEMM_PREFILL = 40,
    R4D_PRIO_GEMM_TILED   = 30,

    /* The block-scaled fp8 family. fp8a16 is a separate dtype and needs no ordering against these
     * two; the narrow fp8a8 row's M <= 16 is a subset of the tiled row's unbounded M, so the
     * narrow row must outrank or it could not be reached. These sit above the skinny int8 GEMMs'
     * band only in the sense that no dtype string reaches both. */
    R4D_PRIO_GEMM_FP8_M16   = 60,
    R4D_PRIO_GEMM_FP8_TILED = 40,
    /* The lm_head at fp8, above the bf16 projection: both serve the same declared geometry, and
     * the one that reads half the bytes wins. The container follows the resolved row (spec 4.2),
     * so this choice is what decides the format the head is stored in. */
    R4D_PRIO_LOGITS_FP8     = 60,

    /* Everything with exactly one candidate row per (op, geometry). The value carries no ordering
     * information; it exists so a user's override plugin has room underneath and above. */
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
int     r4d_rad_attn_decode_h256_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h128_bf16 (const RadArgs*, RadStream);
int64_t r4d_rad_attn_decode_h256_scratch(const RadArgs*);
int64_t r4d_rad_attn_decode_h128_scratch(const RadArgs*);
int     r4d_rad_attn_vit              (const RadArgs*, RadStream);
/* Eight queries per KV head (r4d_attn_paged_gqa8.hip): the routed 35B-A3B at head_dim 256
 * and MiniCPM5-2B at 128. A decode workgroup holds 64 rows of q_len*gqa, so the verify
 * width is 8 here against the gqa6 leg's 10 -- which is the row's constraint value. */
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
/* Twelve queries a KV head, head_dim 256 only -- Qwen3.8-Flash-Next's QSA layers. Same body,
 * different instantiation; r4d_attn_paged_h256_gqa12.hip is four lines. */
int     r4d_rad_attn_prefill_h256_gqa12_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa12_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                    int, int*);
int     r4d_rad_attn_decode_h256_gqa12_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa12_bf16_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                     int, int*);
int     r4d_rad_attn_decode_gq_h256_gqa12_fp8(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa12_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                       int, int*);
int     r4d_rad_attn_decode_gq_h256_gqa12_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa12_bf16_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                        int, int*);
int64_t r4d_rad_attn_decode_h256_gqa12_scratch(const RadArgs*);
/* Six queries a KV head, head_dim 256, paged block 4 -- the same QSA layers at four ranks, where
 * the KV head is replicated across a rank pair. Same body again; the verify width is 64/6 = 10
 * against gqa12's 5, which is the decode rows' constraint value. */
int     r4d_rad_attn_prefill_h256_gqa6_fp8 (const RadArgs*, RadStream);
int     r4d_rad_attn_prefill_h256_gqa6_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_fp8  (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                   int, int*);
int     r4d_rad_attn_decode_h256_gqa6_bf16 (const RadArgs*, RadStream);
int     r4d_rad_attn_decode_h256_gqa6_bf16_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                    int, int*);
int64_t r4d_rad_attn_decode_h256_gqa6_scratch(const RadArgs*);
/* The same two merges with the output gate and the quantiser in them, at the four-rank ratio. The
 * body instantiates these at every ratio (`R4D_ATTN_GQA` pastes them), so unlike `all_gather` this
 * one needed a ROW and not a kernel — see attn_decode_gq_h256_gqa6_fp8kv. */
int     r4d_rad_attn_decode_gq_h256_gqa6_fp8(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa6_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                      int, int*);
int     r4d_rad_attn_decode_gq_h256_gqa6_bf16(const RadArgs*, RadStream);
int     r4d_rad_attn_decode_gq_h256_gqa6_bf16_describe(const RadArgs*, RadStream, RadLaunchDesc*,
                                                       int, int*);

/* -- gated delta net (r4d_gdn_*.hip) -- */
int r4d_rad_gdn_chunk_scan      (const RadArgs*, RadStream);
int64_t r4d_rad_gdn_chunk_scan_scratch(const RadArgs*);
int r4d_rad_gdn_conv_prep       (const RadArgs*, RadStream);
int r4d_rad_gdn_conv_update     (const RadArgs*, RadStream);
int r4d_rad_gdn_kkt_solve       (const RadArgs*, RadStream);
int r4d_rad_gdn_recurrent_update(const RadArgs*, RadStream);
/* gdn_conv_update folded into the recurrent update's prologue: the decode shape, q/k/v never
 * written (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip). */
int r4d_rad_gdn_conv_recurrent_update(const RadArgs*, RadStream);
int r4d_rad_gdn_gated_rmsnorm   (const RadArgs*, RadStream);
int r4d_rad_dflash_conv         (const RadArgs*, RadStream);
int r4d_rad_dflash_select       (const RadArgs*, RadStream);
int r4d_rad_fulldrafter_two_stage(const RadArgs*, RadStream);
int64_t r4d_rad_fulldrafter_two_stage_scratch(const RadArgs*);

/* -- the gated residual (r4d_hc_bf16.hip) -- */
int     r4d_rad_hc_enter       (const RadArgs*, RadStream);
int     r4d_rad_mtp_enter      (const RadArgs*, RadStream);
int     r4d_rad_mtp_enter_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int64_t r4d_rad_mtp_enter_scratch(const RadArgs*);
int     r4d_rad_hc_read        (const RadArgs*, RadStream);
int     r4d_rad_hc_read_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int64_t r4d_rad_hc_read_scratch(const RadArgs*);
int     r4d_rad_hc_write       (const RadArgs*, RadStream);
int r4d_rad_hc_write_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);

/* -- all-reduce (r4d_ar_entry.hip) -- */
int  r4d_rad_ar_init(const RadParam*, int, int rank, int world, void** out);
void r4d_rad_ar_fini(void*);
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
int  r4d_rad_ar_twoshot_4rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_8rank_exact(const RadArgs*, RadStream);
int  r4d_rad_ar_twoshot_4rank_ti8  (const RadArgs*, RadStream);
/* The fused all-reduce + norm shares the IPC set, so it shares init/fini. */
int  r4d_rad_ar_ln_init(const RadParam*, int, int rank, int world, void** out);
int  r4d_rad_ar_ln_had_quant_i8(const RadArgs*, RadStream);
int  r4d_rad_ar_rmsnorm_quant_fp8(const RadArgs*, RadStream);
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
int r4d_rad_gemm_w8a16_m64  (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8_m64   (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8_tiled (const RadArgs*, RadStream);
int r4d_rad_gemm_w4a8_prefill(const RadArgs*, RadStream);
int r4d_rad_gemm_w8a8_m64   (const RadArgs*, RadStream);
int r4d_rad_gemm_w8a8_tiled (const RadArgs*, RadStream);
int r4d_rad_gemm_w2a8       (const RadArgs*, RadStream);
int r4d_rad_gemm_mxfp4a8_m64(const RadArgs*, RadStream);
int r4d_rad_dequant_w4      (const RadArgs*, RadStream);
int r4d_rad_rowtopk         (const RadArgs*, RadStream);
int r4d_rad_rowtopk_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int64_t r4d_rad_rowtopk_scratch(const RadArgs*);
int r4d_rad_rowtopk_merge   (const RadArgs*, RadStream);
int r4d_rad_logit_rerank    (const RadArgs*, RadStream);
int r4d_rad_logit_rerank_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_logit_rerank_i8 (const RadArgs*, RadStream);
int r4d_rad_logit_rerank_i8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);

/* -- the routed mixture of experts (r4d_moe.hip) --
 *
 * `moe_gemm_q`'s weight operands are TABLES (RAD_OPD_WTAB): the core resolves a run of declared
 * weights to a device array of pointers, one per expert, because after placement a layer's
 * experts are not in one place. See abi/rad_runtime.h. */
int     r4d_rad_router_topk        (const RadArgs*, RadStream);
int     r4d_rad_moe_scatter        (const RadArgs*, RadStream);
int     r4d_rad_router_topk_scatter(const RadArgs*, RadStream);
int r4d_rad_router_topk_scatter_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int64_t r4d_rad_router_topk_scatter_scratch(const RadArgs*);
int64_t r4d_rad_moe_scatter_scratch(const RadArgs*);
int     r4d_rad_moe_gemm_q         (const RadArgs*, RadStream);
/* Scratch for the run map the prefill form is aligned by. Zero in the decode band and zero for a
 * shape whose parameters do not describe one, so a caller that never asks still gets a correct
 * uniform grid. */
int64_t r4d_rad_moe_gemm_q_scratch (const RadArgs*);
int64_t r4d_rad_moe_gemm_w4a8_scratch(const RadArgs*);
/* The same grouped GEMM with the expert weight at FOUR BITS: same operands, same schema, same
 * inner loop -- only the LDS staging and the scale grid differ. See r4d_layout.cpp's moe w4
 * section for the stored plane. */
int     r4d_rad_moe_gemm_w4a8      (const RadArgs*, RadStream);
int r4d_rad_moe_gemm_w4a8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
/* `moe_gemm_q_mod4`: the same two kernels over four classes of experts, by the expert's number
 * mod 4. The scratch hooks are the two-class ones. */
int     r4d_rad_moe_gemm_q_mod4    (const RadArgs*, RadStream);
int     r4d_rad_moe_gemm_w4a8_mod4 (const RadArgs*, RadStream);
int r4d_rad_moe_gemm_w4a8_mod4_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
/* `moe_gemm` at bf16: one weight table of the checkpoint's own row-major planes, no scales. The
 * same grouping and the same three forms; its scratch hook sizes the same run map. */
int     r4d_rad_moe_gemm_bf16      (const RadArgs*, RadStream);
int r4d_rad_moe_gemm_bf16_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int64_t r4d_rad_moe_gemm_bf16_scratch(const RadArgs*);
int     r4d_rad_moe_gather         (const RadArgs*, RadStream);
int     r4d_rad_scale_rows         (const RadArgs*, RadStream);
int64_t r4d_rad_moe_gather_scratch (const RadArgs*);

/* -- quantisation, norms and the fusions (r4d_*quant*.hip, r4d_qk_norm_rope_gate.hip) -- */
int r4d_rad_quant_act_i8          (const RadArgs*, RadStream);
int r4d_rad_quant_act_i8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_had_quant_act_i8      (const RadArgs*, RadStream);
int r4d_rad_rmsnorm_had_quant_i8  (const RadArgs*, RadStream);
int r4d_rad_gated_had_quant_i8    (const RadArgs*, RadStream);
int r4d_rad_gdn_gated_norm_hq_i8  (const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate     (const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_qk_norm_rope_gate_kv_store(const RadArgs*, RadStream);
int r4d_rad_qk_norm_rope_gate_kv_store_describe(const RadArgs*, RadStream, RadLaunchDesc*, int,
                                                int*);
int r4d_rad_rope_table            (const RadArgs*, RadStream);
int r4d_rad_rope_table_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);

/* -- block-scaled fp8, and the device sampler (r4d_gemm_fp8*.hip / r4d_fp8_layout.cpp) --
 *
 * These entry points RETURN a negative code rather than throwing, unlike the rest of libr4d's GEMM
 * family, so their shims carry no R4D_TRY. */
int r4d_rad_gemm_fp8a16_m1  (const RadArgs*, RadStream);
int r4d_rad_gemm_fp8a8_m16  (const RadArgs*, RadStream);
int r4d_rad_gemm_fp8a8_m16_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_gemm_fp8a8_gated_m16(const RadArgs*, RadStream);
int64_t r4d_rad_gemm_fp8a8_m16_scratch(const RadArgs*);
int r4d_rad_gemm_fp8a8_tiled(const RadArgs*, RadStream);
int r4d_rad_gemm_i8a8_m16  (const RadArgs*, RadStream);
int r4d_rad_gemm_i8a8_m16_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int64_t r4d_rad_gemm_i8a8_m16_scratch(const RadArgs*);
int r4d_rad_logits_gemm_i8(const RadArgs*, RadStream);
int r4d_rad_gemm_i8a8_tiled(const RadArgs*, RadStream);
int r4d_rad_quant_act_i8g(const RadArgs*, RadStream);
int r4d_rad_quant_act_i8g_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_had_quant_act_i8g(const RadArgs*, RadStream);
int r4d_rad_had_quant_act_i8g_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_quant_act_fp8   (const RadArgs*, RadStream);
int r4d_rad_quant_act_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_had_quant_act_fp8 (const RadArgs*, RadStream);
int r4d_rad_had_quant_act_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_rmsnorm_quant_fp8(const RadArgs*, RadStream);
int r4d_rad_gated_quant_fp8 (const RadArgs*, RadStream);
int r4d_rad_gated_quant_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_gated_had_quant_fp8(const RadArgs*, RadStream);
int r4d_rad_gated_had_quant_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_gate_quant_fp8  (const RadArgs*, RadStream);
int r4d_rad_gate_quant_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_gram_accum      (const RadArgs*, RadStream);
/* The same accumulator over a plain bf16 activation, no scale operand. */
int r4d_rad_gram_accum_bf16 (const RadArgs*, RadStream);
int r4d_rad_qsa_block_key_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_qsa_block_key   (const RadArgs*, RadStream);
int r4d_rad_qsa_score_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_qsa_score       (const RadArgs*, RadStream);
int r4d_rad_qsa_select_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_qsa_select      (const RadArgs*, RadStream);
int64_t r4d_rad_qsa_select_scratch(const RadArgs*);
int r4d_rad_qsa_tail_store_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_qsa_tail_store  (const RadArgs*, RadStream);
int r4d_rad_qsa_work_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_qsa_work        (const RadArgs*, RadStream);
int r4d_rad_sample_chain    (const RadArgs*, RadStream);
int r4d_rad_logits_gemm_fp8 (const RadArgs*, RadStream);

/* -- what each weight-reading kernel stores, from a weight's canonical planes --
 *
 * RadLayoutFn / RadRelayoutFn / RadUnrelayoutFn (rad_abi.h, spec §4.3). r4d_fp8_layout.cpp holds the
 * fp8 families -- block-fp8 codes in fragment order, the lm_head and the hyper-connection and MTP
 * rows with their scales in each row's tail -- and r4d_layout.cpp the packed integer ones. Each
 * refuses an encoding its kernel does not read. A family whose kernel reads the canonical plane as
 * it is -- the row-major fp8 matvec, the fp8 experts -- has only the layout hook, to check what it
 * is handed. */
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
int r4d_rad_layout_mxfp4(const RadParam*, int, int, const RadEncoding*, const int*,
                         const RadTensor*, int, RadLayout*);
int r4d_rad_relayout_mxfp4(const RadParam*, int, int, const RadEncoding*, const int*,
                           const RadTensor*, int, void*, int64_t);

/* -- the small bf16 model ops (r4d_model_bf16.hip) --
 *
 * The embedding gather, the norm, the adds, the rotary and the KV scatter. Uninteresting kernels
 * that the engine cannot run without: its buffers are in VRAM and an op with no DEVICE launch is a
 * fatal issue, not a slow one. `logits_gemm` is declared with them. */
int r4d_rad_embed_lookup    (const RadArgs*, RadStream);
int r4d_rad_embed_lookup_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
/* The quantised gather (embed_lookup_q): the n-gram table's, where a card reads the table. */
int r4d_rad_embed_lookup_q  (const RadArgs*, RadStream);
int r4d_rad_embed_lookup_q_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_ngram_ids       (const RadArgs*, RadStream);
/* ABI v10: the same dispatch, described. Generated with the launch above by R4D_ENTRY, from one
 * body, so the two cannot describe different work. */
int r4d_rad_ngram_ids_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_ple_gate        (const RadArgs*, RadStream);
int r4d_rad_ple_conv        (const RadArgs*, RadStream);
int r4d_rad_cast            (const RadArgs*, RadStream);
int r4d_rad_gather_rows     (const RadArgs*, RadStream);
int r4d_rad_gather_rows_u8  (const RadArgs*, RadStream);
int r4d_rad_gather_rows_u8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_scatter_rows_u8 (const RadArgs*, RadStream);
int r4d_rad_scatter_rows_u8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_scatter_rows_bf16(const RadArgs*, RadStream);
int r4d_rad_scatter_rows_bf16_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_gather_rows_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_rmsnorm         (const RadArgs*, RadStream);
int r4d_rad_add             (const RadArgs*, RadStream);
int r4d_rad_mul             (const RadArgs*, RadStream);
int r4d_rad_sigmoid         (const RadArgs*, RadStream);
int r4d_rad_silu_mul        (const RadArgs*, RadStream);
int r4d_rad_rope            (const RadArgs*, RadStream);
/* -- the vision tower's ops (r4d_vit_bf16.hip) -- */
int r4d_rad_layernorm       (const RadArgs*, RadStream);
int r4d_rad_layernorm_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_grid_embed      (const RadArgs*, RadStream);
int r4d_rad_grid_embed_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_gemm_nt_bias    (const RadArgs*, RadStream);
int r4d_rad_gemm_nt_bias_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_kv_store        (const RadArgs*, RadStream);
int r4d_rad_kv_store_fp8    (const RadArgs*, RadStream);
int r4d_rad_kv_store_fp8_describe(const RadArgs*, RadStream, RadLaunchDesc*, int, int*);
int r4d_rad_logits_gemm     (const RadArgs*, RadStream);

/* -- the decomposed sampler chain (r4d_sample_stages_f32.hip) --
 *
 * docs/OPS.md's eleven ops, as opposed to r4d_rad_sample_chain's fused form above. Both stay: the
 * fused one is what a speculative verify wants, these are what an engine that composes the chain
 * per request needs, and they carry the stages the fused one has no room for. */
int r4d_rad_sample_penalties (const RadArgs*, RadStream);
int r4d_rad_sample_dry       (const RadArgs*, RadStream);
int r4d_rad_sample_temp      (const RadArgs*, RadStream);
int r4d_rad_sample_mask      (const RadArgs*, RadStream);
int r4d_rad_sample_topk      (const RadArgs*, RadStream);
int r4d_rad_sample_argmax    (const RadArgs*, RadStream);
int r4d_rad_sample_topp      (const RadArgs*, RadStream);
int r4d_rad_sample_minp      (const RadArgs*, RadStream);
int r4d_rad_sample_typical   (const RadArgs*, RadStream);
int r4d_rad_sample_xtc       (const RadArgs*, RadStream);
int r4d_rad_sample_pick      (const RadArgs*, RadStream);
int r4d_rad_sample_merge_topk(const RadArgs*, RadStream);

/* -- operand descriptions (r4d_shapes.cpp, pure host) --
 *
 * `RadKernelInfo::opd_shape`: how big is operand i of this op at this geometry, and what may be in
 * it. One function per OP and not per row, because the description is a property of the operand
 * list and not of the tiling: all six `attn_paged` rows take one function and read the cache's
 * width off `kv_dtype`. Where two rows of one op genuinely differ, they take different functions
 * -- or the one that cannot be described carries a NULL and is skipped by name, which is what
 * `logits_gemm`'s fp8 row does. r4d_shapes.cpp's header lists every decline and its reason.
 *
 * Pure host code, called before anything is allocated, so it includes neither r4d.h nor
 * <hip/hip_runtime.h> -- the same split r4d_rows.cpp and r4d_layout.cpp keep. */
int r4d_shape_embed_lookup   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_embed_lookup_q (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gather_rows    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gather_rows_u8 (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scatter_rows_u8(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_cast           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_enter       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_mtp_enter      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_read        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_hc_write       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_binary         (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scatter_rows_bf16(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_layernorm      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_grid_embed     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_bias   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_unary          (const RadParam*, int, int operand, RadOpdDesc*);
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
int r4d_shape_sample_narrow    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_pick      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_sample_chain     (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_attn_paged        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_attn_paged_gate_quant(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qk_norm_rope_gate (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_qk_norm_rope_gate_kv_store(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rope_table        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gdn_gated_rmsnorm (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_all_reduce           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_all_gather           (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_ln_had_quant_i8   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_rmsnorm_quant_fp8 (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_hc_write          (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_ar_gather_hc_write   (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_gemm_nt        (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_q      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gemm_nt_q_gated(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_dequant_w4     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_row_topk       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_router_topk    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_router_topk_scatter(const RadParam*, int, int, RadOpdDesc*);
int r4d_shape_moe_scatter    (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_moe_gemm_q     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_moe_gemm_q_mod4(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_moe_gather     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_scale_rows     (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_row_topk_merge (const RadParam*, int, int operand, RadOpdDesc*);

int r4d_shape_quant_act_i8       (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_had_quant_act_i8   (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm_had_quant_i8(const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gated_had_quant_i8 (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gdn_gated_norm_had_quant_i8(const RadParam*, int, int operand,
                                          RadOpdDesc*);
int r4d_shape_quant_act_fp8      (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_rmsnorm_quant_fp8  (const RadParam*, int, int operand, RadOpdDesc*);
int r4d_shape_gated_quant_fp8    (const RadParam*, int, int operand, RadOpdDesc*);
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
 * `RadKernelInfo::replaces`: the chain of conventional ops this kernel computes in one launch, so
 * that a tool may run the chain out of this plugin's OWN unfused rows and compare the bytes. One
 * function per FUSED ROW and not per op, unlike the shape hooks: two rows of one op could fold
 * different things, and the chain is a claim about a kernel rather than about an operand list.
 *
 * FIVE ROWS DECLARE ONE AND THE REST DELIBERATELY DO NOT. r4d_fuse.cpp's header lists every
 * decline with the line of libr4d that decides it -- the int8 rotated family runs its Hadamard
 * before the norm closes and keeps the product in f32, so `silu_mul` + `had_quant_act_i8` is a
 * different function however it is spelled, and `ar_ln_had_quant_i8` reduces without the bf16
 * round its unfused pair stores. A chain that is merely close publishes no hook, because a false
 * failure is worse than a skip.
 *
 * Pure host code, called before anything is allocated, so it includes neither r4d.h nor
 * <hip/hip_runtime.h> -- the same split r4d_rows.cpp, r4d_layout.cpp and r4d_shapes.cpp keep. */
int r4d_fuse_rmsnorm_quant_fp8   (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gated_quant_fp8     (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gate_quant_fp8      (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gemm_nt_q_gated     (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_gdn_conv_recurrent_update(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_qk_norm_rope_gate_kv_store(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_ar_rmsnorm_quant_fp8(const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_ar_gather_hc_write  (const RadParam*, int, uint64_t, int, RadFuseStep*);
int r4d_fuse_attn_paged_gate_quant(const RadParam*, int, uint64_t, int, RadFuseStep*);

}  /* extern "C" */

#endif /* RAD_R4D_PLUGIN_H */
