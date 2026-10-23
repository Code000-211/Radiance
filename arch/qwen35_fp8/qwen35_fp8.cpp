/* qwen35_fp8 -- Qwen3.5 / 3.6 / 3.8 dense, block-scaled FP8 weights and FP8 activations.
 *
 * The sibling of qwen35_bf16: same architecture, graph, metadata and name maps for everything but
 * the projection matrices. The architecture is described in qwen35_bf16.cpp and the block headers;
 * this file covers what differs. One plugin implements one architecture id for one quantisation
 * scheme (spec §2.4).
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * A checkpoint whose config.json carries
 *
 *     "quantization_config": { "quant_method": "fp8", "fmt": "e4m3",
 *                              "activation_scheme": "dynamic",
 *                              "weight_block_size": [128, 128] }
 *
 * as in Qwen3.8-27B-FP8, Qwen3.6-27B-FP8 and other DeepSeek-style fp8 releases of this family.
 * Each large linear ships as `<name>.weight` in F8_E4M3 [N][K] beside `<name>.weight_scale_inv` in
 * BF16 [ceil(N/128)][ceil(K/128)]: the codes and scale plane of one weight,
 * `fp8_e4m3*bf16[128x128]`, which libr4d's block-scaled fp8 GEMMs (libr4d/r4d_fp8.cpp) read as
 * is. The checkpoint loads directly; widening to bf16 would double 24 GiB of weights to 48 GiB.
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * From the checkpoint's safetensors (the pattern is not simply "the big ones"):
 *
 *   F8_E4M3 + scale   self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
 *                     linear_attn.{in_proj_qkv, in_proj_z, out_proj}
 *   BF16, no scale    every norm, A_log, dt_bias, conv1d, embed_tokens, lm_head, AND
 *                     linear_attn.in_proj_a / in_proj_b
 *
 * in_proj_a and in_proj_b emit one column per value head (48), not a whole 128-row scale block,
 * so the checkpoint's quantiser left them bf16. That is why GdnFP8 splits the in-projection the
 * bf16 block fuses (rad_block_gdn_fp8.h).
 *
 * The lm_head is declared at whatever the model holds (bf16 as shipped, block fp8 where a recipe
 * quantised it), and the logits kernel for that encoding is selected.
 */
/* ============================== ONE SOURCE, TWO PLUGINS ==============================
 *
 * `qwen35moe` (Qwen3.6-35B-A3B-FP8) is this architecture with a routed feed-forward and
 * everything above it identical: the only structural difference is `mlp.experts.*` plus a router
 * and a shared expert in place of `mlp.{gate,up,down}_proj`.
 *
 * So one source builds two translation units; arch/qwen35moe_fp8/ defines the three macros below
 * and includes this file. Every difference is behind `#if QWEN35_MOE`: the expert-count check, the
 * routing geometry read from metadata, the block's buffers, the per-layer source names, the MTP
 * head's feed-forward, and the plugin identity. */
/* ============================== AND A THIRD: qwen35_w8a8 ==============================
 *
 * `qwen35_w8a8` is the dense build with an int8 trunk: every large linear `i8*bf16[1x128]/fwht128`,
 * rotated by a 128-wide Hadamard and served at i8a8h, each quantising its own input from the bf16
 * plane (LinearFP8's int8 branch). The producers already write that plane, as they do for a rotated
 * MXFP4 linear, so the graph is this one. It is a plugin of its own because the quantisation
 * descriptor is half of the selection key: `w8a8_fwht128`, which names the rotation, so a plain
 * int8 trunk is refused.
 *
 * Its differences are behind `#if QWEN35_W8A8`: check_format's method, the probe that learns the
 * trunk is int8 before the activations are declared (and refuses one that is not), and the plugin
 * identity. The int8 activation pairs follow from the probe. */
#ifndef QWEN35_MOE
#define QWEN35_MOE 0
#endif
#ifndef QWEN35_W8A8
#define QWEN35_W8A8 0
#endif
#if QWEN35_MOE && QWEN35_W8A8
#error "qwen35_w8a8 is the dense build; a routed int8 trunk is not a format this file serves"
#endif
/* The plugin's name for a message, expanded once more so it is QWEN35_NS's value. */
#define QWEN35_STR_X(x) #x
#define QWEN35_STR(x)   QWEN35_STR_X(x)
#ifndef QWEN35_NS
#define QWEN35_NS qwen35_fp8
#endif
#ifndef QWEN35_ARCH_ID
#define QWEN35_ARCH_ID "qwen35"
#endif

#include <cstdlib>
#include <arch/rad_arch.h>
#include <arch/rad_block_attn_gated_fp8.h>
#include <arch/rad_block_gdn_fp8.h>
#include <arch/rad_block_mlp_fp8.h>
#if QWEN35_MOE
#include <arch/rad_block_moe_fp8.h>
#endif
#include <arch/rad_block_mtp_fp8.h>
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_vit.h>

#include <cstring>
#include <vector>

namespace QWEN35_NS {

using namespace rad::arch;

/* The feed-forward this build declares, under one name. Both blocks take the same declare()
 * arity over their own Wire and Src (the routed one carries its geometry in its Src for this), so
 * every site below is spelled once. */
#if QWEN35_MOE
using Ffn    = MoeFP8;
using MtpHead = MtpBlockFP8<MoeFP8>;
#else
using Ffn    = MlpFP8;
using MtpHead = MtpFP8;
#endif

struct Layer {
    int          full = 0;
    AttnGatedFP8 attn;
    GdnFP8       gdn;
    Ffn          mlp;
};

struct Model {
    Names          nm{""};
    Geom           g{};
    GdnFP8::Config gcfg{};
    int64_t        n_rot  = 0;
    int64_t        n_main = 0;

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;

    /* The trunk's linears are int8 (LinearFP8::i8), learned before the activations are declared
     * because it decides whether they carry an int8 code pair. Only qwen35_w8a8 sets it. */
    bool trunk_i8 = false;

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_hout = 0, b_logits = 0;
    ActFP8  a_h{};
    /* full attention */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0;
    /* [rope_table_rows, n_rot] f32, the rope cos/sin every attention layer reads: one fill per
     * step instead of a rope launch per layer (rad_block_attn_gated_fp8.h, op_fused). */
    rad_buf b_rope_cs = 0;
    ActFP8  a_attn{};
    /* gated delta net */
    ActFP8  a_in{}, a_go{};
    rad_buf b_ab = 0, b_gq = 0, b_gk = 0, b_gv = 0;
    rad_buf b_gdec = 0, b_beta = 0, b_kkt = 0;
    /* feed-forward */
    rad_buf b_gate_up = 0;
    ActFP8  a_ffn{};
#if QWEN35_MOE
    /* The routed expert path's intermediates. `moe_rows` tall, not max_tok, because expert planes
     * are top_k times as tall as the step (rad_block_moe_fp8.h). */
    MoeFP8::Config moecfg{};
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_egu = 0, b_edn = 0, b_emoe = 0;
    ActFP8  a_eff{};
    rad_buf b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActFP8  a_sff{};
    /* Rotated codes of the expert input, for rotated four-bit experts (MoeFP8::Config::expert_rot). */
    ActFP8  a_hr{};
#endif

    /* The DFlash2 drafter, declared only when the container carries one and the deployment
     * speculates. An MTP head runs one chained forward per draft token; this runs one forward for
     * the whole block. If a container carries both, only DFlash2 is declared, being the better
     * drafter on this target. */
    bool          have_df = false;
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;

    /* the multi-token-prediction head, declared only when the deployment speculates */
    bool     have_mtp = false;
    MtpHead  mtp{};
    rad_buf  b_mtp_e = 0, b_mtp_c = 0, b_mtp_x = 0, b_mtp_h = 0;
    rad_buf  b_mtp_dq = 0, b_mtp_ds = 0, b_mtp_dsum = 0;
    rad_buf  b_mtp_dlog = 0, b_mtp_didx = 0, b_mtp_dval = 0;

    /* Auxiliary hidden-state taps, declared only when the container carries a drafter that wants
     * them. A DFlash2 drafter reads the residual stream at several intermediate layers and
     * projects the concatenation down (`fc` is [n_embd, n_taps * n_embd]), so its input must be one
     * contiguous row. The trunk's one residual buffer is overwritten by every layer, so each tap's
     * rows are copied into its column slice as they pass. */
    enum { MAX_TAPS = 8 };
    int      n_taps = 0;
    int      tap_layer[MAX_TAPS] = {};
    rad_buf  b_dfsrc = 0;                 /* [max_tok, n_taps * n_embd] */
    rad_op   op_tap = 0;

    rad_weight w_tok = 0, w_out_norm = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_final_norm = 0, op_gather = 0, op_logits = 0;
    rad_op op_rope_cs = 0;   /* rope_table, if the fused prologue is reachable */

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows,
     * declared only when the container carries a tower and the deployment asked for media. */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

    std::vector<Layer> layers;
};

static Model g_model[MAX_RANKS];

/* The sizing declare's own Model. A sizing declare (`RadBuildCtx::shape_probe`) runs after the
 * real one and would otherwise reset and refill `g_model`, which `step()` reads for the whole
 * run: the buffers, weights and per-layer handles the real declare resolved. It is thread-local
 * since the engine declares every rank's level on its own thread, and each such thread owns the
 * one Model its declares write. */
#if QWEN35_MOE
static thread_local Model g_probe;
#endif

/* Which Model a declare writes. Only the routed build answers a sizing declare (probe() below
 * sets shape_probe_ok under QWEN35_MOE); the dense and int8 builds keep the real one, so their
 * output is unchanged. */
static Model& declare_target(const RadBuildCtx* ctx) {
#if QWEN35_MOE
    if (ctx->shape_probe) return g_probe;
#else
    (void)ctx;
#endif
    return g_model[ctx->rank];
}

/* Checks the container's statement of its format, since every mismatch is silent: a
 * `weight_block_size` of [64,64] against kernels that shift by 7 reads the right bytes with the
 * wrong gain, and e5m2 decoded as e4m3 gives merely implausible numbers. A container that states
 * nothing (a .rad converted from a GGUF) is accepted; weight shapes and scale planes are checked
 * where they are read. */
static int check_format(const RadModelMeta* meta) {
    const char* method = rad_meta_gets(meta, "quantization_config.quant_method", nullptr);
    const char* fmt    = rad_meta_gets(meta, "quantization_config.fmt", nullptr);
#if QWEN35_W8A8
    /* The format the convert stated (the recipe's --set keys); the trunk probe in declare() holds
     * the weights to it. */
    if (!method || std::strcmp(method, "w8a8") != 0 || !fmt || std::strcmp(fmt, "fwht128") != 0) {
        fprintf(stderr, "radiance: qwen35_w8a8 serves a container converted for an int8 trunk "
                        "(quant_method 'w8a8', fmt 'fwht128': data/recipes/"
                        "q38-27b-w8a8-df2.recipe's --set keys); this one says '%s' / '%s'.\n",
                method ? method : "(none)", fmt ? fmt : "(none)");
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
#else
    const char* block  = rad_meta_gets(meta, "quantization_config.weight_block_size", nullptr);

    if (method && std::strcmp(method, "fp8") != 0) {
        fprintf(stderr, "radiance: qwen35_fp8 refuses a container quantised by '%s'. This plugin "
                        "serves block-scaled fp8 only; qwen35_bf16 serves an unquantised one.\n",
                method);
        return RAD_E_UNSUPPORTED;
    }
    if (fmt && std::strcmp(fmt, "e4m3") != 0) {
        fprintf(stderr, "radiance: qwen35_fp8 refuses fp8 format '%s'. libr4d's block-scaled GEMMs "
                        "decode E4M3 and nothing else; an e5m2 weight read as e4m3 is a different "
                        "number at every element.\n", fmt);
        return RAD_E_UNSUPPORTED;
    }
    if (block) {
        long long b0 = 0, b1 = 0;
        if (std::sscanf(block, "%lld %lld", &b0, &b1) != 2) { b0 = 0; b1 = 0; }
        if (b0 != RAD_FP8_BLOCK || b1 != RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: qwen35_fp8 refuses weight_block_size [%lld, %lld]. The "
                            "block is %d in both dimensions and is compiled into libr4d as a "
                            "shift, so another width reads the right bytes with the wrong gain.\n",
                    b0, b1, RAD_FP8_BLOCK);
            return RAD_E_UNSUPPORTED;
        }
    }
    return RAD_OK;
#endif
}

/* The DFlash2 drafter: its geometry, KV group, buffers and block.
 *
 * A second model, not a head: it has its own layer count, head count, feed-forward width, rope
 * base, norm convention and vocabulary claim. Its Geom starts as a copy of the trunk's (n_embd,
 * deployment bounds, rank, wire) and takes everything else from the drafter's config.
 *
 * Nothing is defaulted quietly: a `draft.` config that does not describe a drafter this engine
 * implements is refused by name, because a silently ignored scale factor would only lower
 * acceptance (spec §17). */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg{};
    /* Plain RMS norm: the drafter uses the ordinary norm where the trunk uses GemmaRMSNorm
     * (rad_block_dflash2.h). */
    RAD_ARCH_TRY(drafter_geom(dg, m.g, meta, "DFlash2"));

    Dflash2::Config dc{};
    dc.n_taps   = m.n_taps;
    dc.block    = ctx->max_spec + 1;
    dc.taps     = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group    = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    dc.causal   = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                  rad_meta_geti(meta, "draft.sliding_window", 0));
    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld). Without it the block of drafts has no path "
                        "through it.\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }

    /* The query block must be a power of two: the convolution masks a row's position with
     * `block_size - 1`, so any other value convolves one sequence's block into the next. The shim
     * also rejects it, but only after the drafter's weights have loaded. */
    const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (dc.block < 2 || (dc.block & (dc.block - 1))) {
        fprintf(stderr, "radiance: DFlash2's convolution masks the block position with "
                        "block_size - 1, so the query block has to be a power of two. This serve "
                        "asks for %d draft tokens, which makes it %lld; the checkpoint was trained "
                        "at %lld (%lld draft tokens).\n",
                ctx->max_spec, (long long)dc.block, trained, trained - 1);
        return RAD_E_UNSUPPORTED;
    }
    if (trained > 0 && trained != dc.block)
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and runs at "
                    "%lld; it accepts less away from its trained block length",
                 trained, trained - 1, (long long)dc.block);

    /* Three scalars this implementation does not apply. Each changes the proposals and would show
     * up only as lower acceptance, so a non-default value is refused. */
    const double escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    const double omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    const double scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    if (escale != 1.0 || omul != 1.0 || scap != 0.0) {
        fprintf(stderr, "radiance: this DFlash2 checkpoint scales its input embedding by %g and "
                        "its draft logits by %g, and soft-caps them at %g. None of the three is "
                        "implemented, and a drafter that ignored them would just accept less.\n",
                escale, omul, scap);
        return RAD_E_UNSUPPORTED;
    }

    /* The window bounds the cache: the drafter's 2048-token window, not the context length. */
    RAD_ARCH_TRY(drafter_kv_group(b, nm, ctx, dg, dc.window, base_layer, &m.kv_draft));

    /* Activations. The context pass has one row per committed token over a whole prefill chunk;
     * the query pass one row per draft position, `block` per sequence; the two are sized apart.
     * All are small: the drafter's cost is its weights. */
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();
    const int64_t spad = ((sr + 15) / 16) * 16;

    Dflash2::Wire& dw = m.dfw;
    dw.src     = m.b_dfsrc;
    dw.c       = decl_b(b, nm.f("df_c"),    dg.act_dtype, {dg.max_tok, dg.n_embd});
    dw.ck      = decl_b(b, nm.f("df_ck"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.cv      = decl_b(b, nm.f("df_cv"),   dg.act_dtype, {dg.max_tok, kvd});
    dw.x       = decl_b(b, nm.f("df_x"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h       = decl_b(b, nm.f("df_h"),    dg.act_dtype, {qr, dg.n_embd});
    dw.h2      = decl_b(b, nm.f("df_h2"),   dg.act_dtype, {qr, dg.n_embd});
    dw.coef    = decl_b(b, nm.f("df_coef"), dg.act_dtype, {qr, dc.coef_dim(dg.n_embd)});
    dw.q       = decl_b(b, nm.f("df_q"),    dg.act_dtype, {qr, qd});
    dw.k       = decl_b(b, nm.f("df_k"),    dg.act_dtype, {qr, kvd});
    dw.v       = decl_b(b, nm.f("df_v"),    dg.act_dtype, {qr, kvd});
    dw.attn    = decl_b(b, nm.f("df_attn"), dg.act_dtype, {qr, qd});
    dw.gate_up = decl_b(b, nm.f("df_gate_up"), dg.act_dtype, {qr, 2 * dg.n_ff});
    dw.ffn     = decl_b(b, nm.f("df_ffn"),  dg.act_dtype, {qr, dg.n_ff});
    /* FP8 copies. The drafter's six linears are block-scaled fp8 over a bf16 checkpoint
     * (rad_fp8.h's DenseFP8), so every activation feeding one has an E4M3 copy and an f32 scale
     * plane. `srcq` is the largest (50 MiB at a 2048-token chunk and five taps) and halves the
     * bytes of the context pass's dominant GEMM. */
    dw.srcq    = decl_b(b, nm.f("df_srcq"), RAD_F8E4M3,
                        {dg.max_tok, dc.n_taps * dg.n_embd});
    dw.srcs    = decl_b(b, nm.f("df_srcs"), RAD_F32,
                        {dg.max_tok, fp8_blocks(dc.n_taps * dg.n_embd)});
    dw.cq      = decl_b(b, nm.f("df_cq"),   RAD_F8E4M3, {dg.max_tok, dg.n_embd});
    dw.cs      = decl_b(b, nm.f("df_cs"),   RAD_F32,    {dg.max_tok, fp8_blocks(dg.n_embd)});
    dw.h2q     = decl_b(b, nm.f("df_h2q"),  RAD_F8E4M3, {qr, dg.n_embd});
    dw.h2s     = decl_b(b, nm.f("df_h2s"),  RAD_F32,    {qr, fp8_blocks(dg.n_embd)});
    dw.atq     = decl_b(b, nm.f("df_atq"),  RAD_F8E4M3, {qr, qd});
    dw.ats     = decl_b(b, nm.f("df_ats"),  RAD_F32,    {qr, fp8_blocks(qd)});
    dw.ffq     = decl_b(b, nm.f("df_ffq"),  RAD_F8E4M3, {qr, dg.n_ff});
    dw.ffs     = decl_b(b, nm.f("df_ffs"),  RAD_F32,    {qr, fp8_blocks(dg.n_ff)});
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    dw.dq      = decl_b(b, nm.f("df_dq"),   RAD_I8,  {sr, dg.n_embd});
    dw.ds      = decl_b(b, nm.f("df_ds"),   RAD_F32, {sr, 1});
    /* Group-major, [n_embd/group][ceil(M/16)*16]: quant_act_i8's asum plane, whose first
     * dimension is the K group, not the token. */
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    /* This rank's plane. `head_rows` is the block's own answer to whether the 2-bit head is
     * vocab-sharded; asking it keeps the buffer consistent with the op declared over it. */
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    /* Candidates are rescored against the target's head when logit_rerank can read its plane
     * (fp8 or int8 rows) and every candidate a rank proposes is one it owns; the planes are then
     * kRerankR wide and the selector reads the first sel_topk. */
    {
        const uint32_t hd = weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16);
        if ((hd == RAD_F8E4M3 || hd == RAD_I8) && (Dflash2::head_sharded(dg) || dg.world <= 1)) {
            m.df.w_lm  = m.w_lm_head;
            m.df.top_w = Dflash2::kRerankR;
        }
    }
    const int64_t tw = m.df.w_lm ? m.df.top_w : dc.sel_topk;
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, tw});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, tw});
    /* The vocab-parallel candidate exchange, and only when there is one: this rank's top-R as
     * (int32 id, float value) pairs and the all-gathered world_size * R of them. */
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, tw, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * tw, 2});
    }
    /* The proposal buffer (RadDrafterDecl::proposal): a DFlash2 pass proposes through the
     * selector, never through the sampler's token buffer. */
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;

    /* Block drafter: a context pass mirroring the trunk step, then one pass filling the whole
     * block. `mask_token_id` is read only here and travels in the drafter declaration. `window`
     * must be the one this plugin gated its own attention with, since the scheduler stages the
     * block pass against it. */
    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    /* The anchor row predicts nothing here, so the block is one row longer than the depth and the
     * head runs over rows 1..depth (rad_builder.h, RadDrafterDecl::block). */
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: this DFlash2 checkpoint declares no "
                        "dflash_config.mask_token_id, and every drafted position embeds it -- "
                        "there is nothing to run.\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));

    rad_note(b, "DFlash2 drafter: %lld layers at absolute %d..%d, %lld heads / %lld kv x %lld, "
                "ffn %lld, %s attention with a %lld-token window, %d trunk tap(s); one forward "
                "proposes %lld tokens through the candidate selector",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.steps());
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = declare_target(ctx);
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    /* "bf16" is the activation dtype every op but the fp8 linears matches on: norms, rope,
     * attention, delta-net kernels and collectives run on bf16, and E4M3 activations exist only
     * between quant_act_fp8 and the GEMM that reads them. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

#if QWEN35_MOE
    if (meta->n_expert <= 0) {
        fprintf(stderr, "radiance: qwen35moe_fp8 needs a routed checkpoint and this one declares "
                        "no experts; that is qwen35\n");
        return RAD_E_UNSUPPORTED;
    }
#else
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: " QWEN35_STR(QWEN35_NS) " refuses a checkpoint with %lld "
                        "experts; that is qwen35moe\n", (long long)meta->n_expert);
        return RAD_E_UNSUPPORTED;
    }
#endif

    Geom&  g  = m.g;
    Names& nm = m.nm;

    bool have_interval = false;
    const int64_t interval = meta_int2(meta, "qwen35.full_attention_interval",
                                       "full_attention_interval", 4, &have_interval);
    const int64_t nextn    = meta_int2(meta, "qwen35.nextn_predict_layers",
                                       "mtp_num_hidden_layers", 0);
    if (interval <= 0 || nextn < 0 || nextn >= g.n_layer) {
        fprintf(stderr, "radiance: qwen35: full_attention_interval %lld / nextn %lld out of range "
                        "for %lld layers\n",
                (long long)interval, (long long)nextn, (long long)g.n_layer);
        return RAD_E_INVAL;
    }
    RAD_ARCH_TRY(check_layer_schedule("qwen35", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval, {"full_attention"}));
    /* Every declared layer is a trunk layer: `mtp_num_hidden_layers` is not part of
     * `num_hidden_layers` in this family. Qwen3.8-27B-FP8 ships layers 0..63 under
     * `model.language_model.layers.N` and its MTP block separately under `mtp.`; subtracting it
     * would drop full-attention layer 63 and silently run a trunk one layer short. */
    m.n_main = g.n_layer;

    /* Qwen3.5's norms are GemmaRMSNorm, `x_hat * (1 + w)` (vLLM imports GemmaRMSNorm as
     * Qwen3_5RMSNorm). This covers input_layernorm, post_attention_layernorm, the final norm and
     * the q/k head norms, but not the delta net's ssm_norm, a plain RMSNormGated (gains ~0.88).
     * The checkpoint confirms it: layer norm weights of 0.07..0.29 and partly negative q/k norm
     * weights only make sense as `1 + w`. The final norm (0.68..0.96) is ambiguous and follows the
     * reference implementation. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;
    const float final_wadd = 1.0f;

    m.n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (m.n_rot <= 0) {
        /* Both spellings occur in this family: Qwen3.8-27B-FP8 states `partial_rotary_factor`
         * at the top level and inside `rope_parameters`, Qwen3.5-0.8B only inside
         * `rope_parameters` (rad-convert flattens nested config with a dot). */
        double prf = rad_meta_getf(meta, "partial_rotary_factor", 0.0);
        if (!(prf > 0.0)) prf = rad_meta_getf(meta, "rope_parameters.partial_rotary_factor", 0.0);
        if (prf > 0.0) m.n_rot = (int64_t)(g.head_dim * prf);
    }
    if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot & 1)) {
        fprintf(stderr, "radiance: qwen35: no usable rotary width -- give the container "
                        "qwen35.rope.dimension_count or partial_rotary_factor (head_dim %lld)\n",
                (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));
    GdnFP8::Config& gc = m.gcfg;
    const int64_t n_head_k_all = meta_int2(meta, "qwen35.ssm.group_count",
                                           "linear_num_key_heads", 0);
    const int64_t n_head_v_all = meta_int2(meta, "qwen35.ssm.time_step_rank",
                                           "linear_num_value_heads", 0);
    gc.head_k     = meta_int2(meta, "qwen35.ssm.state_size", "linear_key_head_dim", 0);
    gc.head_v     = meta_int2(meta, "qwen35.ssm.value_head_dim", "linear_value_head_dim", 0);
    gc.conv_width = meta_int2(meta, "qwen35.ssm.conv_kernel", "linear_conv_kernel_dim", 0);
    if (gc.head_v <= 0) {
        const int64_t inner = meta_int2(meta, "qwen35.ssm.inner_size", "linear_inner_size", 0);
        if (inner > 0 && n_head_v_all > 0) gc.head_v = inner / n_head_v_all;
    }
    if (n_head_k_all <= 0 || n_head_v_all <= 0 || gc.head_k <= 0 || gc.head_v <= 0 ||
        gc.conv_width <= 0) {
        fprintf(stderr, "radiance: qwen35: the container does not describe the linear layers "
                        "(key heads %lld, value heads %lld, key width %lld, value width %lld, "
                        "conv width %lld)\n",
                (long long)n_head_k_all, (long long)n_head_v_all, (long long)gc.head_k,
                (long long)gc.head_v, (long long)gc.conv_width);
        return RAD_E_INVAL;
    }
    gc.chunk = 64;

    int s;
    if ((s = divide_or_fail(n_head_k_all, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0)
        return s;
    if ((s = divide_or_fail(n_head_v_all, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0)
        return s;

    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = gc.v_dim();
    const int64_t conv_dim = gc.conv_dim();

    /* ---- the three KV groups. The attention cache is bf16 by default; `kv_dtype` follows it.
     *
     * --kv-cache-dtype fp8 makes it E4M3. The cache is `n_head_kv * head_dim * 2` bytes per token
     * per layer (2 KiB per rank here), paid only by the 16 attention layers; the 48 delta-net
     * layers carry fixed-size state. At eight sequences of 16k that is 4.3 GB of reads per step,
     * most of the cost of depth at that concurrency, and fp8 halves it along with each sequence's
     * KV pool footprint.
     *
     * Off by default for quality: greedy continuations from a long prompt agree for a few hundred
     * tokens and then diverge, and the fp8 side can produce syntactically broken code where bf16
     * stays clean, which matters for a coding model.
     *
     * The descales stay at libr4d's default of 1.0 (vLLM's default too); the kernels index them
     * per (sequence, head), so calibrated descales need no other mechanism. The fp8kv attention
     * rows and `kv_store_fp8` are in libr4d, the core sizes a page from the group's dtype, and the
     * drafter's cache takes the same dtype (declare_dflash2). */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    /* --kv-cache-dtype fp8_v4 / k6_v4 (RadBuildCtx::kv_quant): V as int4 in groups of 32 with f16
     * scales, K E4M3 as above or int6 the same way (libr4d's fp8_v4 / k6_v4 rows). A token's row
     * is 25 or 22 * head_dim / 16 bytes instead of 2 * head_dim, so the group is declared at half
     * that and the core sizes pages from it; only those kernels read the layout. The drafter's
     * group stays E4M3 (declare_dflash2). The engine asks only for a layout probe() offered. */
    const int kvq = fd.dtype == RAD_F8E4M3 ? ctx->kv_quant : RAD_KVQ_NONE;
    const char* kvq_name = kvq == RAD_KVQ_V4 ? "fp8_v4" : kvq == RAD_KVQ_K6V4 ? "k6_v4" : nullptr;
    if (kvq != RAD_KVQ_NONE && !kvq_name) {
        fprintf(stderr, "radiance: KV cache layout %d is not one this plugin knows\n", kvq);
        return RAD_E_UNSUPPORTED;
    }
    if (kvq_name && !(g.head_dim == 256 && g.n_head == 6 * g.n_head_kv)) {
        fprintf(stderr, "radiance: --kv-cache-dtype %s has libr4d rows at head_dim 256, gqa 6 "
                        "only; this model is head_dim %lld, gqa %lld\n", kvq_name,
                (long long)g.head_dim, (long long)(g.n_head_kv ? g.n_head / g.n_head_kv : 0));
        return RAD_E_UNSUPPORTED;
    }
    const int64_t row16 = kvq == RAD_KVQ_V4 ? 25 : kvq == RAD_KVQ_K6V4 ? 22 : 32;
    fd.head_dim = g.head_dim * row16 / 32;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    /* The op parameter must use the spelling the rows band on, which is rad_dtype_name's. */
    g.kv_dtype = kvq_name ? kvq_name : rad_dtype_name(fd.dtype);
    if (kvq_name)
        rad_note(b, "kv cache %s: %s K, int4 V, in groups of 32 with f16 scales, %lld bytes a token "
                    "and head instead of %lld", kvq_name, kvq == RAD_KVQ_V4 ? "E4M3" : "int6",
                 (long long)(g.head_dim * row16 / 16), (long long)(2 * g.head_dim));

    /* The SSM state cache is f32.
     *
     * State reads and writes dominate the two GDN kernels: one [128,128] state is 64 KiB at f32,
     * and a speculative step writes one per candidate token per head, since the surviving
     * candidate is unknown until after the layer. At depth 7 that is hundreds of MB per rank per
     * step; f16 would halve it. libr4d compiles both widths (`SDT`) and the f16 round-trip error is
     * ~5e-4 relative (elements are O(0.1)); vLLM's `--mamba-ssm-cache-dtype` is commonly float16.
     * It stays f32 because f16 changes the generated text. */
    RadKVGroupDecl sd{};
    sd.kind         = RAD_KV_LINEAR;
    sd.dtype        = RAD_F32;
    sd.n_head_kv    = gc.n_head_v;
    sd.state_dim[0] = gc.head_v;
    sd.state_dim[1] = gc.head_k;
    m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

    RadKVGroupDecl cd{};
    cd.kind       = RAD_KV_CONV;
    cd.dtype      = RAD_BF16;
    cd.n_head_kv  = 2 * gc.n_head_k + gc.n_head_v;
    cd.head_dim   = gc.head_k;
    cd.conv_width = gc.conv_width;
    m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);

#if QWEN35_W8A8
    /* ---- the trunk's stored format, read from the first delta-net layer's input projection as
     * qwen4exp_fp8 reads its own: the activations below carry an int8 code pair only for an int8
     * trunk, and this build serves only the rotated one its descriptor names. The probe maps the
     * weight through the layer's own LinearFP8, which then does not map it again; the resize
     * before the layer loop keeps the layers sized here. */
    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        if (is_full_attention(l, interval)) continue;
        const char* pre = "model.language_model.layers";
        RAD_ARCH_TRY(m.layers[(size_t)l].gdn.in.probe(
            b, nm, g, nm.f("blk.%lld.ssm_inz", (long long)l),
            { nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre, (long long)l),
              nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight", pre, (long long)l) }));
        RadEncoding enc{};
        m.trunk_i8 = weight_enc(b, nm.f("blk.%lld.ssm_inz.weight", (long long)l), &enc) &&
                     linear_is_w8h(enc);
        break;
    }
    if (!m.trunk_i8) {
        fprintf(stderr, "radiance: qwen35_w8a8 serves a rotated int8 trunk (i8 codes with a bf16 "
                        "scale a [1 x 128], /fwht128), and this container's first delta-net input "
                        "projection is not one; convert with "
                        "data/recipes/q38-27b-w8a8-df2.recipe\n");
        return RAD_E_DTYPE;
    }
#endif

    /* ---- activation buffers. Every activation that feeds a GEMM has an fp8 copy: an E4M3 plane
     * of the same shape and an f32 scale plane with one column per 128. They are transients that
     * liveness analysis can share, but not free: the largest, `ffn`'s, is 8192 x 17408 bytes
     * (143 MiB) beside its 285 MiB bf16 buffer. */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x     = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd, 0, m.trunk_i8));

    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x  = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    /* The rope table, shared by all sixteen attention layers (same positions, same theta). Only
     * the rows a step names are written. One row per absolute position, past `max_ctx` by the
     * speculative window (rope_table_rows), because the fused kernel indexes it by absolute
     * position. f32, so the fused rotation uses the same cosines `rope` computes.
     *
     * PERSIST: its rows outlive the step that wrote them, and it is read across the whole step, so
     * it must not share storage with transients. */
    m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32, {rope_table_rows(g), m.n_rot},
                         RAD_BUF_PERSIST);
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim, 0, m.trunk_i8));

    /* `gdn_in` is conv_dim + v_dim, not in_dim: a and b have their own buffer because the
     * checkpoint keeps them bf16, while the fp8 output gate is the tail of this one
     * (rad_block_gdn_fp8.h). The convolution input and the gate are column views of it. */
    m.a_in.x    = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab      = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {g.max_tok, gc.ab_dim()});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    /* bf16, not f32: gdn_kkt_solve writes this plane as `unsigned short*` and gdn_chunk_scan
     * reads it the same way. libref reads the declared dtype and is correct either way, so a wrong
     * declaration would not show up on the host path. */
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    /* Rank 3, as libr4d's scan and recurrent kernels require; the quantiser reads dim 0 as the
     * row. */
    m.a_go.x    = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim, 0, m.trunk_i8));

#if QWEN35_MOE
    /* ---- the routing geometry, read here so a refusal names the container's key. `n_ff` is
     * already the per-rank expert width: rad-convert falls `intermediate_size` back to
     * `moe_intermediate_size` and geom_from divided it by world_size.
     *
     * The expert count is not divided. Every rank holds every expert and routes every token the
     * same way; tensor parallelism splits each expert's intermediate width. */
    {
        MoeFP8::Config& mc = m.moecfg;
        mc.n_expert = meta->n_expert;
        mc.top_k    = meta->n_expert_used > 0 ? meta->n_expert_used
                                              : meta_int2(meta, "qwen35.expert_used_count",
                                                          "num_experts_per_tok", 0);
        mc.n_ff_exp = g.n_ff;
        /* `norm_topk_prob` defaults to true in this family (Qwen2-MoE onward). Without it the
         * top-k weights of a softmax over 256 experts sum to roughly 0.4 here, shrinking every
         * routed contribution by more than half. */
        mc.norm_topk = (int)meta_int2(meta, "qwen35.expert_weights_norm", "norm_topk_prob", 1);
        const int64_t sh_all = meta_int2(meta, "qwen35.expert_shared_feed_forward_length",
                                         "shared_expert_intermediate_size", 0);
        if (sh_all > 0) {
            int s2;
            if ((s2 = divide_or_fail(sh_all, g.world, "shared_expert_intermediate_size",
                                     &mc.n_ff_shared)) < 0) return s2;
        }
        /* The pass height. 2048 is the prefill chunk this family runs at, so the shipped
         * configuration is one pass; a deployment with a larger chunk slices rather than carrying
         * top_k times a whole chunk of expert intermediates. */
        const int64_t moe_rows = 2048;
        mc.rows = g.max_tok < moe_rows ? g.max_tok : moe_rows;
        if (mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: qwen35moe: top_k %lld is not in [1, %lld]; the container "
                            "must give num_experts_per_tok\n",
                    (long long)mc.top_k, (long long)mc.n_expert);
            return RAD_E_INVAL;
        }
        /* GPTQ calibration run (MoeFP8::Config::calib), on when the output directory is set. */
        mc.calib = calib_dir() != nullptr ? 1 : 0;
        /* The experts' stored format, read from layer 0's first expert as qwen4exp does, so a
         * container with four-bit experts is served in that format. */
        {
            MoeFP8::Src ps{};
            RAD_ARCH_TRY(moe_expert_sources(b, nm, "format",
                                            "model.language_model.layers.0.mlp.experts", &ps));
            RAD_ARCH_TRY(MoeFP8::probe(b, nm, 0, ps, &mc));
        }
        if (mc.expert_bf16) {
            fprintf(stderr, "radiance: qwen35moe: plain bf16 experts are not served by this "
                            "plugin; convert with a quantising recipe\n");
            return RAD_E_UNSUPPORTED;
        }
    }
    {
        /* The step's height for everything with one row per token, and a pass's height only for
         * the four planes with one row per (token, slot); see the rule at rad_block_moe_fp8.h,
         * Wire. */
        const MoeFP8::Config& mc = m.moecfg;
        const int64_t R = g.max_tok, K = mc.rows * mc.top_k;
        m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {R, mc.n_expert});
        m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {R, mc.top_k});
        m.b_ew      = decl_b(b, nm.f("moe_w"),      g.act_dtype, {R, mc.top_k});
        m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {K});
        m.b_eoff    = decl_b(b, nm.f("moe_off"),    RAD_I32,     {mc.n_expert + 1});
        /* PERSIST: the histogram's only reader is rad_route_report, not a declared op, so
         * liveness would see it dead as soon as moe_scatter wrote it. */
        m.b_ecnt    = decl_b(b, nm.f("moe_cnt"),    RAD_I32,     {mc.n_expert},
                             RAD_BUF_PERSIST);
        m.b_egu     = decl_b(b, nm.f("moe_gate_up"), g.act_dtype, {K, 2 * mc.n_ff_exp});
        m.a_eff.x   = decl_b(b, nm.f("moe_ffn"),    g.act_dtype, {K, mc.n_ff_exp});
        m.a_eff.q   = decl_b(b, nm.f("moe_ffn.q"),  RAD_F8E4M3,  {K, mc.n_ff_exp});
        m.a_eff.s   = decl_b(b, nm.f("moe_ffn.s"),  RAD_F32,     {K, fp8_blocks(mc.n_ff_exp)});
        m.b_edn     = decl_b(b, nm.f("moe_down"),   g.act_dtype, {K, g.n_embd});
        if (mc.n_ff_shared > 0) {
            m.b_emoe  = decl_b(b, nm.f("moe_out"),   g.act_dtype, {R, g.n_embd});
            m.b_sgu   = decl_b(b, nm.f("shexp_gate_up"), g.act_dtype, {R, 2 * mc.n_ff_shared});
            m.a_sff.x = decl_b(b, nm.f("shexp_ffn"),     g.act_dtype, {R, mc.n_ff_shared});
            m.a_sff.q = decl_b(b, nm.f("shexp_ffn.q"),   RAD_F8E4M3,  {R, mc.n_ff_shared});
            m.a_sff.s = decl_b(b, nm.f("shexp_ffn.s"),   RAD_F32, {R, fp8_blocks(mc.n_ff_shared)});
            m.b_sout  = decl_b(b, nm.f("shexp_out"),     g.act_dtype, {R, g.n_embd});
            m.b_sgate = decl_b(b, nm.f("shexp_gate"),    g.act_dtype, {R, 1});
        }
        if (mc.expert_rot) RAD_ARCH_TRY(m.a_hr.declare_qs(b, nm, g, "moe_hr", g.n_embd));
    }
#else
    m.b_gate_up = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.a_ffn.x   = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff, 0, m.trunk_i8));
#endif

#if QWEN35_W8A8
    /* ---- the rotated trunk's quantisers in their producers. Every reader of `h`'s codes is a
     * rotated int8 projection (q, k, v; the delta net's qkv|z; gate_up), and of `ffn`'s the down
     * projection, so where libr4d serves the `_had_quant_i8g` forms the norms and the gated
     * product write had_quant_act_i8g's codes themselves (ActFP8::q8_rot) and those projections
     * declare no quantiser: one launch for q, k and v instead of three, and none before down. The
     * codes are the same bytes. Where the forms are not declared (the CPU kernels), each
     * projection quantises its own input from the bf16 plane, as before. */
    if (rad_op_declared(b, "rmsnorm_had_quant_i8g") &&
        (g.world != 2 || rad_op_declared(b, "ar_rmsnorm_had_quant_i8g")))
        m.a_h.q8_fed = m.a_h.q8_rot = true;
    if (rad_op_declared(b, "gated_had_quant_i8g")) m.a_ffn.q8_fed = m.a_ffn.q8_rot = true;
#endif

    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* The auxiliary taps, only when the container carries the drafter and the deployment
     * speculates: one residual stream per tap, 100 MiB at a 2048-token chunk and five taps. `cast`
     * does the copy into a column slice (libr4d/r4d_rows.cpp). */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash")) {
            m.n_taps = parse_layer_ids(
                rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
                m.n_main, m.tap_layer, Model::MAX_TAPS);
            if (m.n_taps > 0) {
                m.b_dfsrc = decl_b(b, nm.f("dflash_src"), g.act_dtype,
                                   {g.max_tok, (int64_t)m.n_taps * g.n_embd});
                m.op_tap = rw(b, RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                             RAD_INT("n", g.n_embd),
                                             RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.b_x}, {m.b_dfsrc});
            }
        }
    }
    m.b_logits  = decl_b(b, nm.f("logits"),   RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- vocabulary edges. The embedding is bf16. The lm_head is declared at its codes' dtype
     * (bf16 as shipped, or block fp8 where the recipe made it so; the widest GEMM at half the
     * bytes), selecting the logits kernel for that encoding.
     *
     * Both are row-sharded with the vocabulary; replicating one would put 248320 x 5120 bf16
     * (2.43 GiB) on every rank, at the expense of KV cache. `embed_lookup` takes a `vocab_offset`:
     * a rank gathers the ids in its slice and writes zero rows for the rest, and the all_reduce
     * below sums the ranks into x.
     *
     * The name maps come first, since they are how the checkpoint is queried for each weight. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy(b, nm.f("output_norm.weight"),
                          nm.ckpt("model.language_model.norm.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));

    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_out_norm = decl_w(b, nm.f("output_norm.weight"), RAD_F32, {g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    /* Row-sharded with the vocabulary: g.n_vocab is this rank's share (rad_arch.h). */
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* The encoder's rows overwrite the lookup before its reduction, which layer 0's norm folds
     * in at two ranks, so this is the only place for them (see MediaRows). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));
    /* Always exact, not g.wire_exact: this all_reduce sums disjoint rows (each token's embedding
     * comes from one rank, the others contribute zero), so a lossy wire would only add error.
     * min_bytes stays 0 for the same reason. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});
    /* ---- the layers. */
    /* The rope table's fill, declared before the layers that read it. If it does not resolve,
     * `aw.cos_sin` is cleared and every attention block uses its unfused norm + rope pair. Its band
     * is the fused prologue's (qk_fuse_rows), so a step the prologue does not serve (a prefill
     * chunk, where the fill's grid is one block per token) issues no fill either. */
    m.op_rope_cs = rw(b, decl_rotary(b, g, "rope_table",
                         RAD_PARAMS(RAD_RANGE("M", 1, qk_fuse_rows(g)), RAD_INT("rot", m.n_rot),
                                    RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                    RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS),
                     {}, {m.b_rope_cs});

    AttnGatedFP8::Wire aw{};
    aw.x = m.b_x; aw.h = m.a_h; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.attn = m.a_attn;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;

    GdnFP8::Wire gw{};
    gw.x = m.b_x; gw.h = m.a_h; gw.in = m.a_in; gw.ab = m.b_ab;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.a_go;

    Ffn::Wire mw{};
#if QWEN35_MOE
    mw.x = m.b_x; mw.h = m.a_h;
    mw.logits = m.b_rlogits; mw.ids = m.b_eids; mw.ew = m.b_ew;
    mw.sorted = m.b_sorted;  mw.eoff = m.b_eoff; mw.ecnt = m.b_ecnt;
    mw.egu = m.b_egu; mw.eff = m.a_eff; mw.edn = m.b_edn; mw.moe = m.b_emoe;
    mw.sgu = m.b_sgu; mw.sff = m.a_sff; mw.sout = m.b_sout; mw.sgate = m.b_sgate;
    mw.hr = m.a_hr;
#else
    mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;
#endif

    /* ---- where the residual add happens (rad_fp8.h, layer_fold): each block folds its
     * predecessor's delta into its own norm, saving ~128 dispatches per step. The MTP head stays
     * unfolded: it is a separate chain over its own residual buffer and saves only two launches a
     * step. */
    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        const LayerFold f = layer_fold(g, l, m.n_main, m.tap_layer, m.n_taps, m.op_embed_ar != 0);

        const char* pre = "model.language_model.layers";
        if (lay.full) {
            AttnGatedFP8::Src as{};
            as.norm   = nm.ckpt("%s.%lld.input_layernorm.weight",     pre, (long long)l);
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight",    pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight",    pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight",    pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight",    pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight",    pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          f.mix_fold, false, f.mix_ar_in, f.mix_ar_out));
        } else {
            GdnFP8::Src gs{};
            gs.norm     = nm.ckpt("%s.%lld.input_layernorm.weight",           pre, (long long)l);
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight",   pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",     pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",     pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",     pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",        pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",                pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",              pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",          pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",      pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, gc, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         f.mix_fold, false, f.mix_ar_in, f.mix_ar_out));
        }

        Ffn::Src ms{};
#if QWEN35_MOE
        ms.cfg         = m.moecfg;
        ms.norm        = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.router      = nm.ckpt("%s.%lld.mlp.gate.weight",                 pre, (long long)l);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "layer",
                                        nm.ckpt("%s.%lld.mlp.experts", pre, (long long)l), &ms));
        if (m.moecfg.n_ff_shared > 0) {
            ms.shared      = nm.ckpt("%s.%lld.mlp.shared_expert",              pre, (long long)l);
            ms.shared_gate = nm.ckpt("%s.%lld.mlp.shared_expert_gate.weight",  pre, (long long)l);
        }
#else
        ms.norm = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.gate = nm.ckpt("%s.%lld.mlp.gate_proj.weight",            pre, (long long)l);
        ms.up   = nm.ckpt("%s.%lld.mlp.up_proj.weight",              pre, (long long)l);
        ms.down = nm.ckpt("%s.%lld.mlp.down_proj.weight",            pre, (long long)l);
#endif
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, true, f.mlp_add,
                                     f.mlp_ar_in, f.mlp_ar_out));
    }
#if QWEN35_MOE
    /* Each routed layer predicts the next one's experts at its router input (MoeFP8::xp_next). */
    for (int64_t l = 0; l + 1 < m.n_main; ++l) m.layers[(size_t)l].mlp.xp_next = &m.layers[(size_t)l + 1].mlp;
#endif

    m.op_final_norm = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                        RAD_F64("eps", g.eps), RAD_F64("wadd", final_wadd), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(m.w_out_norm)),
                         {m.b_x}, {m.a_h.x});

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {m.a_h.x}, {m.b_hout});

    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("n_vocab", g.n_vocab),
                                    RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(m.w_lm_head)),
                     {m.b_hout}, {m.b_logits});

    /* ---- the multi-token-prediction head (rad_block_mtp_fp8.h).
     *
     * Declared only when the deployment speculates, since its ~0.5 GB of weights would otherwise
     * be loaded and never issued. rad-convert declares at a non-zero max_spec so the container
     * carries everything the model can serve.
     *
     * Also only when there is no DFlash2 drafter (`n_taps` == 0): a deployment drafts with one,
     * and DFlash2 (one forward per block rather than one per token) is the better one here. */
    if (nextn > 0 && g.max_spec > 0 && m.n_taps == 0) {
        /* Sized at max_tok, not max_logit_rows: a draft round is one row per sequence, but the
         * history pass that fills the head's own K and V is one row per token over a whole prefill
         * chunk (26 MiB at a 512-token chunk, beside 477 MiB of head weights). */
        m.b_mtp_h = decl_b(b, nm.f("mtp_h"), g.act_dtype, {g.max_tok, g.n_embd});
        m.b_mtp_e = decl_b(b, nm.f("mtp_e"), g.act_dtype, {g.max_tok, g.n_embd});
        m.b_mtp_c = decl_b(b, nm.f("mtp_c"), g.act_dtype, {g.max_tok, 2 * g.n_embd});
        m.b_mtp_x = decl_b(b, nm.f("mtp_x"), g.act_dtype, {g.max_tok, g.n_embd});

        /* The 2-bit draft head's working set, at max_logit_rows, since only a draft round (one row
         * per sequence) uses it. `mtp_dsum` is quant_act_i8's group-major
         * [n_embd/128][ceil(M/16)*16] plane, first dimension the K group, not the token. */
        if (mtp_draft_head_q2(b, nm.f("mtp.draft_head.weight"), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,   {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32,  {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32,  {g.n_embd / 128, dmpad});
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype, {g.max_logit_rows, g.n_vocab_all});
            /* A draft round's proposal comes out of row_topk here, never through the sampler's
             * token buffer. */
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32,  {g.max_logit_rows, 1});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32,  {g.max_logit_rows, 1});
        }

        /* The head reuses the trunk's working buffers (`attn_qg`, `attn_k`, the ffn planes, `h`),
         * all dead by the time a draft round runs; a second set would be 150 MiB. Only the
         * residual stream is its own, because the trunk's `x` is still needed by the next step's
         * prefill. */
        AttnGatedFP8::Wire tw_attn = aw; tw_attn.x = m.b_mtp_x;
        Ffn::Wire          tw_mlp  = mw; tw_mlp.x  = m.b_mtp_x;

        MtpHead::Wire tw{};
        tw.src = m.a_h.x;   tw.hout = m.b_mtp_h; tw.logits = m.b_logits;
        tw.e = m.b_mtp_e;   tw.c = m.b_mtp_c;    tw.x = m.b_mtp_x;
        tw.dq = m.b_mtp_dq; tw.ds = m.b_mtp_ds;  tw.dsum = m.b_mtp_dsum;
        tw.dlog = m.b_mtp_dlog; tw.didx = m.b_mtp_didx; tw.dval = m.b_mtp_dval;
        tw.attn = tw_attn;  tw.mlp = tw_mlp;

        MtpHead::Src ts{};
        ts.fc                 = nm.ckpt("mtp.fc.weight");
        ts.norm               = nm.ckpt("mtp.norm.weight");
        ts.pre_norm_hidden    = nm.ckpt("mtp.pre_fc_norm_hidden.weight");
        ts.pre_norm_embedding = nm.ckpt("mtp.pre_fc_norm_embedding.weight");
        ts.attn.norm   = nm.ckpt("mtp.layers.0.input_layernorm.weight");
        ts.attn.qg     = nm.ckpt("mtp.layers.0.self_attn.q_proj.weight");
        ts.attn.k      = nm.ckpt("mtp.layers.0.self_attn.k_proj.weight");
        ts.attn.v      = nm.ckpt("mtp.layers.0.self_attn.v_proj.weight");
        ts.attn.q_norm = nm.ckpt("mtp.layers.0.self_attn.q_norm.weight");
        ts.attn.k_norm = nm.ckpt("mtp.layers.0.self_attn.k_norm.weight");
        ts.attn.o      = nm.ckpt("mtp.layers.0.self_attn.o_proj.weight");
        ts.mlp.norm    = nm.ckpt("mtp.layers.0.post_attention_layernorm.weight");
#if QWEN35_MOE
        /* The head's feed-forward is routed too, with its own 256 experts: `mtp.layers.0` carries
         * `mlp.experts.*`, `mlp.gate` and a shared expert like a trunk layer. That is why
         * MtpBlockFP8 is a template over the feed-forward. */
        ts.mlp.cfg     = m.moecfg;
        ts.mlp.router  = nm.ckpt("mtp.layers.0.mlp.gate.weight");
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "mtp", nm.ckpt("mtp.layers.0.mlp.experts"),
                                        &ts.mlp));
        if (m.moecfg.n_ff_shared > 0) {
            ts.mlp.shared      = nm.ckpt("mtp.layers.0.mlp.shared_expert");
            ts.mlp.shared_gate = nm.ckpt("mtp.layers.0.mlp.shared_expert_gate.weight");
        }
#else
        ts.mlp.gate    = nm.ckpt("mtp.layers.0.mlp.gate_proj.weight");
        ts.mlp.up      = nm.ckpt("mtp.layers.0.mlp.up_proj.weight");
        ts.mlp.down    = nm.ckpt("mtp.layers.0.mlp.down_proj.weight");
#endif
        /* The trunk maps this same tensor as `output.weight`; the draft head maps it again under a
         * name of its own, which the recipe quantises to two bits (mtp_draft_head_q2). */
        ts.lm          = nm.ckpt("lm_head.weight");

        /* Bound as one more layer of the trunk's full-attention group, at absolute index n_main,
         * so a draft round writes its K and V at the slot of the trunk's token for that position.
         * A rejected draft is then overwritten rather than rolled back (rad_block_mtp_fp8.h). */
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)m.n_main, m.kv_full));
        m.mtp.media = m.media;
        RAD_ARCH_TRY(m.mtp.declare(b, nm, g, (int)m.n_main, m.kv_full, m.n_rot, m.w_tok,
                                   m.w_lm_head, final_wadd, tw, ts));
        m.have_mtp = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal.
         * The 2-bit head writes its own argmax into `mtp_didx`; without it `mtp_didx` is 0 and the
         * bf16 head goes through the ordinary sampler chain. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = m.b_mtp_didx;
        dd.proposal_pitch = 1;
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
    }

    /* ---- the DFlash2 drafter (rad_block_dflash2.h), at absolute layers n_main.. of its own KV
     * group. `n_taps` is non-zero only when `draft.architectures` names a DFlash model and
     * max_spec is positive. */
    if (m.n_taps > 0) {
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m, (int)m.n_main));
        if (nextn > 0)
            rad_note(b, "the DFlash2 drafter replaces the %lld-layer MTP head, which is not "
                        "declared or carried; convert without --draft-model for the MTP head",
                     (long long)nextn);
    }

    rad_note(b, "rank %d/%d: %lld trunk layers, %lld full attention + %lld gated delta net, "
                "n_head=%lld n_head_kv=%lld head_dim=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main,
             (long long)(m.n_main / interval), (long long)(m.n_main - m.n_main / interval),
             (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "weights are block-scaled fp8: E4M3 [N,K] with a bf16 [N/%d,K/%d] scale plane, "
                "read by gemm_nt_q at fp8a8 as stored",
             RAD_FP8_BLOCK, RAD_FP8_BLOCK);
    rad_note(b, "activations are quantised to E4M3 every step (activation_scheme: dynamic), once "
                "per activation: the attention projections share one, as do the delta net's qkv "
                "and z");
    rad_note(b, "bf16 in the checkpoint: every norm, A_log, dt_bias, conv1d, the embedding, the "
                "lm_head (unless a recipe quantised it) and the delta net's in_proj_a / in_proj_b, "
                "whose 48 rows are not a whole %d-row scale block and run as a separate gemm_nt",
             RAD_FP8_BLOCK);
    rad_note(b, "gdn: %lld key heads x %lld, %lld value heads x %lld, conv width %lld, chunk %lld; "
                "conv row %lld channels, a|b row %lld",
             (long long)gc.n_head_k, (long long)gc.head_k, (long long)gc.n_head_v,
             (long long)gc.head_v, (long long)gc.conv_width, (long long)gc.chunk,
             (long long)conv_dim, (long long)gc.ab_dim());
    rad_note(b, "ssm_a_log holds A_log itself: the kernels form -exp(A_log) * softplus(a + dt), "
                "so a GGUF's ssm_a = -exp(A_log) must be converted back");
    rad_note(b, "rope: %s over %lld of %lld head dims%s%s", g.rope_mode, (long long)m.n_rot,
             (long long)g.head_dim, g.rope_mc ? ", sections " : "", g.rope_sections);
    if (!have_interval)
        rad_note(b, "the container did not state full_attention_interval; assuming %lld, which "
                    "makes layers 3,7,... full attention", (long long)interval);
    if (m.have_mtp)
        rad_note(b, "MTP head: the first of %lld NextN layer(s) under `mtp.`, as absolute layer "
                    "%lld of the full-attention KV group",
                 (long long)nextn, (long long)m.n_main);
    else if (nextn > 0)
        rad_note(b, "the %lld MTP/NextN layer(s) under `mtp.` are not loaded: this deployment "
                    "does not speculate", (long long)nextn);

    /* The vision tower, last: an encoder pass issues nothing else, so its activations share the
     * arena with every buffer above. Its weight groups start past every text layer, the MTP head
     * and a drafter's. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* A drafter pass runs only the drafter. For MTP, the trunk already left its hidden states in
     * `h`; the head turns the rows it wants into a draft token or one more position of its own
     * attention history, by the sign of `draft_pass` (rad_block_mtp_fp8.h). Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        /* Fill the drafter's own cos/sin rows first. A drafter pass runs past the committed end
         * (the MTP head's round k is at position head_i + k - 1), at rows the trunk step never
         * wrote; a stale row would silently cost acceptance only. A row depends only on its
         * position, so re-filling is idempotent. */
        if (m.op_rope_cs && batch->n_tok <= qk_fuse_rows(m.g) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, batch->n_tok,
                        rope_pos1(batch, batch->n_tok), RAD_B(m.b_rope_cs));
        /* For DFlash2, positive is the query pass, proposing the whole block at once, and
         * negative the context pass, turning the tokens the trunk just committed into the
         * drafter's own K and V. */
        if (m.have_df) {
            if (batch->draft_pass > 0) m.df.step(c, batch);
            else                      m.df.context(c, batch);
            return;
        }
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container has neither an `mtp.` block nor a `draft.` one, or "
                             "declare ran at max_spec 0");
            return;
        }
        m.mtp.step(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    m.mrows.step(c, batch, m.b_x, batch->n_tok);
    /* The vocab shard's reduction, only where layer 0's norm does not fold it. At tp1 there is no
     * op; at tp2 the fused norm normally takes it (every step for n_embd 5120). The standalone op
     * stays declared because ar_fused_ok depends on T and the width, and an n_embd the compressed
     * wire cannot tile falls back to it. */
    if (m.op_embed_ar && !ar_wire_is_exact(m.g, batch->n_tok))
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    /* One rope-table fill per step for all attention layers, writing only this step's rows, and
     * only when the fused prologue serves the step (a prefill chunk takes the unfused pair). The
     * band comes from `qk_fuse_rows`, so the fill and its reader agree. */
    if (m.op_rope_cs && batch->n_tok <= qk_fuse_rows(m.g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, batch->n_tok,
                    rope_pos1(batch, batch->n_tok), RAD_B(m.b_rope_cs));

    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
        if (l.full) l.attn.step(c, batch);
        else        l.gdn.step(c, batch);
        dbg_resid(c, (int)li, "mix", m.g.n_embd, m.b_x, m.a_h.x);
        l.mlp.step(c, batch);
        dbg_resid(c, (int)li, "ffn", m.g.n_embd, m.b_x, m.a_h.x);

        /* The tap, after the layer: the drafter conditions on this layer's output, which the next
         * layer overwrites, so the rows are copied into this tap's column slice now. */
        for (int t = 0; t < m.n_taps; ++t) {
            if (m.tap_layer[t] != (int)li) continue;
            RAD_ISSUE(c, m.op_tap, brows(m.b_x, batch->n_tok),
                      bcol(m.b_dfsrc, (int64_t)t * m.g.n_embd, m.g.n_embd, batch->n_tok));
        }
    }

    RAD_ISSUE(c, m.op_final_norm,
              brows(m.b_x, batch->n_tok), RAD_W(m.w_out_norm), brows(m.a_h.x, batch->n_tok));

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_h.x, batch->n_tok),
                    praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* ================================================================== the probe
 *
 * The speculation depth this container's drafter supports, answered before declare because
 * `max_spec` sizes the drafter's buffers, bands kernels and sets the scheduler's window.
 *
 * An MTP head is serial: depth d is d chained passes, so each extra token costs more as it gets
 * less likely. On Qwen3.8-27B-FP8 at two ranks throughput is flat across depths 2-4 and falls
 * after, so 3 is used.
 *
 * DFlash2 proposes the whole block in one pass, so its depth is the block length it was trained
 * at (block_size 8: one anchor and 7 mask positions), and any other depth is refused.
 *
 * 0 when the container carries no drafter. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    /* --kv-cache-dtype fp8_v4 / k6_v4: libr4d's rows for them are the h256/gqa6 attention
     * family's (Qwen3.8-27B), so other geometries are refused at startup rather than in declare. */
    {
        const long long hd  = meta->head_dim ? meta->head_dim
                                             : (meta->n_head ? meta->n_embd / meta->n_head : 0);
        const long long nkv = meta->n_head_kv ? meta->n_head_kv : meta->n_head;
        if (hd == 256 && nkv > 0 && meta->n_head == 6 * nkv)
            out->kv_quant_ok = (1 << RAD_KVQ_V4) | (1 << RAD_KVQ_K6V4);
    }
#if QWEN35_MOE
    /* The routed build answers a sizing declare: declare() takes the step's rows from the build
     * context alone and writes them into its own Model, so Engine::probe_arena_levels can hold the
     * answer against the real declare. plan_level decides per level -- a buffer set or an extent
     * that differs refuses it there, not here. Modes this build was not checked in are not claimed
     * to hold: the graph equality is a host-side check of the declares the tests build. */
    out->shape_probe_ok = 1;
#endif
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    if (da && std::strstr(da, "DFlash2")) {
        const long long bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 8);
        out->draft_depth = bs > 1 ? (int)(bs - 1) : 7;
        out->draft_depth_fixed = 1;   /* trained at exactly this block; any other is a refusal */
        return RAD_OK;
    }
    if (meta_int2(meta, "qwen35.nextn_predict_layers", "mtp_num_hidden_layers", 0) > 0)
        out->draft_depth = 3;
    return RAD_OK;
}

}  /* namespace QWEN35_NS */

#ifndef RAD_ARCH_NO_EXPORTS
/* One more level of expansion is required. RAD_ARCH_PLUGIN stringifies its NS argument to name
 * the plugin, and `#` applies to the parameter as written, so passing QWEN35_NS directly would
 * register "QWEN35_NS" for both builds. An argument not stringified in the intermediate macro is
 * expanded first, so these forward the expanded namespace. */
#define QWEN35_PROBE_X(NS)              RAD_ARCH_PROBE(NS)
#define QWEN35_PLUGIN_X(NS, ...)        RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
QWEN35_PROBE_X(QWEN35_NS)
#if QWEN35_MOE
QWEN35_PLUGIN_X(QWEN35_NS, QWEN35_ARCH_ID, "fp8_e4m3", "0.1.0",
                "Qwen3.5 / 3.6 MoE (35B-A3B), block-scaled fp8 weights and fp8 activations: the "
                "dense graph with a routed feed-forward (256 experts, 8 a token, and a shared "
                "expert), gated attention over paged KV every fourth layer and gated delta net "
                "between")
#elif QWEN35_W8A8
QWEN35_PLUGIN_X(QWEN35_NS, QWEN35_ARCH_ID, "w8a8_fwht128", "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), an int8 trunk: i8 weights with a bf16 "
                "scale a 128 of K, rotated by a 128-wide Hadamard, against int8 activations the "
                "same rotation quantises; gated attention over paged KV every fourth layer and "
                "gated delta net between")
/* An imported checkpoint is never a rotated int8 trunk: qwen35_fp8 serves those. */
RAD_ARCH_SERVES_IMPORTS(0)
#else
QWEN35_PLUGIN_X(QWEN35_NS, QWEN35_ARCH_ID, "fp8_e4m3", "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), block-scaled fp8 weights and fp8 "
                "activations, gated attention over paged KV every fourth layer and gated delta "
                "net between")
#endif
#endif
