/* qwen35_fp8 -- Qwen3.5 / 3.6 / 3.8 dense, BLOCK-SCALED FP8 weights and FP8 activations.
 *
 * The sibling of qwen35_bf16 and the one that serves the production checkpoint. Same architecture,
 * same graph, same metadata, same name maps for everything that is not a projection matrix -- so
 * the architecture prose lives in qwen35_bf16.cpp and in the three block headers, and this file
 * says only what makes it a different plugin. One plugin implements one architecture id for one
 * quantisation scheme (spec §2.4), and this is the second scheme.
 *
 * ============================== WHAT IT SERVES ==============================
 *
 * A checkpoint whose config.json carries
 *
 *     "quantization_config": { "quant_method": "fp8", "fmt": "e4m3",
 *                              "activation_scheme": "dynamic",
 *                              "weight_block_size": [128, 128] }
 *
 * which is Qwen3.8-27B-FP8, Qwen3.6-27B-FP8 and every other DeepSeek-style fp8 release of this
 * family. Each large linear ships as `<name>.weight` in F8_E4M3 [N][K] beside
 * `<name>.weight_scale_inv` in BF16 [ceil(N/128)][ceil(K/128)] -- the codes and the scale plane
 * of one weight, `fp8_e4m3*bf16[128x128]`, which libr4d's three block-scaled fp8 GEMMs read as
 * they are, so the checkpoint loads directly and a .rad of it costs what the checkpoint does.
 *
 * qwen35_bf16's header argues for widening to bf16 instead, and that argument holds only where
 * libr4d's registry exposes no fp8-weight GEMM. It does expose them (libr4d/r4d_fp8.cpp), so the
 * widening is not the cheaper path here: it doubles 24 GiB of weights to 48 GiB to buy nothing.
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * Read off `layers-0.safetensors` and `layers-3.safetensors` rather than assumed, because the
 * pattern is not "the big ones":
 *
 *   F8_E4M3 + scale   self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj,
 *                     linear_attn.{in_proj_qkv, in_proj_z, out_proj}
 *   BF16, no scale    every norm, A_log, dt_bias, conv1d, embed_tokens, lm_head, AND
 *                     linear_attn.in_proj_a / in_proj_b
 *
 * in_proj_a and in_proj_b emit one column per value head -- 48 of them -- and 48 is not a whole
 * 128-row scale block, so the quantiser that made the checkpoint left them alone. That is why
 * GdnFP8 splits the in-projection the bf16 block fuses; rad_block_gdn_fp8.h has the detail.
 *
 * The lm_head being bf16 in the checkpoint is worth stating too. It is declared at whatever the
 * model holds it as -- bf16 loaded directly, block fp8 where a recipe quantised it -- and the
 * logits kernel that reads that encoding is the one selected.
 */
/* ============================== ONE SOURCE, TWO PLUGINS ==============================
 *
 * `qwen35moe` is this architecture with the feed-forward ROUTED and everything above it identical:
 * the same gated attention, the same gated delta net, the same layer schedule, the same rope, the
 * same vocabulary edges, the same MTP head in the same position. Qwen3.6-35B-A3B-FP8 is the model,
 * and reading its config beside Qwen3.8-27B-FP8's the only structural difference is `mlp.experts.*`
 * plus a router and a shared expert where `mlp.{gate,up,down}_proj` was.
 *
 * So there is ONE source and two translation units, exactly as libr4d's attention dispatches are
 * one set of kernel templates instantiated at three geometries. arch/qwen35moe_fp8/ is four lines
 * that define the three macros below and include this file. The alternative is a second 1300-line
 * plugin kept in step by hand, and two hand-kept copies of one thing drift apart -- the reason
 * libr4d/r4d_rows.cpp keeps exactly one table of its kernels.
 *
 * EVERY DIFFERENCE IS BEHIND `#if QWEN35_MOE` AND THERE ARE SIX OF THEM: the expert-count check,
 * the routing geometry read out of metadata, the block's buffers, the per-layer source names, the
 * MTP head's feed-forward, and the plugin's own identity. Nothing else in this file knows. */
#ifndef QWEN35_MOE
#define QWEN35_MOE 0
#endif
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

/* The feed-forward this build declares, and the ONE name the rest of the file uses for it. Both
 * blocks take the same declare() arity over their own Wire and Src -- the routed one carries its
 * geometry inside its Src for exactly that reason -- so every site below is spelled once. */
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

    /* the residual stream and the two vocabulary edges */
    rad_buf b_x = 0, b_hout = 0, b_logits = 0;
    ActFP8  a_h{};
    /* full attention */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0;
    /* [rope_table_rows, n_rot] f32, the rope cos/sin every attention layer reads. One buffer and
     * one fill a step, against a rope launch a layer -- see rad_block_attn_gated_fp8.h at
     * op_fused. */
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
    /* ...and the routed one's, which are the expert path's intermediates. They are `moe_rows`
     * tokens tall and not max_tok, because the expert planes are top_k times as tall as the step;
     * rad_block_moe_fp8.h argues the slicing. */
    MoeFP8::Config moecfg{};
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_egu = 0, b_edn = 0, b_emoe = 0;
    ActFP8  a_eff{};
    rad_buf b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActFP8  a_sff{};
#endif

    /* THE DFLASH2 DRAFTER, declared only when the container carries one and the deployment
     * speculates. It is the second drafter and it is not an alternative spelling of the first: an
     * MTP head is one chained forward per draft token, this is one forward for the whole block.
     * Both can be in a container; only one is declared, and DFlash2 wins because it is the better
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

    /* THE AUXILIARY HIDDEN-STATE TAPS, declared only when the container carries a drafter that
     * asks for them. A DFlash2 drafter does not read the trunk's FINAL hidden state -- it reads
     * the residual stream at several intermediate layers and projects the concatenation down, so
     * `fc` is [n_embd, n_taps * n_embd] and its input has to be one contiguous row. The trunk
     * keeps one residual buffer that every layer overwrites, so the rows have to be copied out as
     * they pass, into the column slice that layer owns. Nothing else in the trunk changes, and on
     * a container with no drafter none of this is declared. */
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

/* "5 19 33 47 61" -> the layers whose output a drafter wants. rad-convert flattens a config.json
 * array into a space-separated string (tools/rad_convert.cpp, flatten_json), so this is the shape
 * every list-valued key arrives in. Out-of-range entries are dropped rather than clamped: a
 * drafter that names a layer this model does not have is paired with the wrong target, and
 * silently reading layer 63 instead of 71 would make that a quality mystery. */
static int parse_tap_layers(const char* s, int64_t n_layer, int* out, int max) {
    if (!s) return 0;
    int n = 0;
    for (const char* p = s; *p && n < max;) {
        while (*p == ' ' || *p == ',') ++p;
        if (!*p) break;
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        p = e;
        if (v >= 0 && v < n_layer) out[n++] = (int)v;
    }
    return n;
}

static Model g_model[MAX_RANKS];

static inline bool is_full_attention(int64_t l, int64_t interval) {
    return interval > 0 && ((l + 1) % interval == 0);
}

/* THE CONTAINER'S OWN LAYER SCHEDULE, CHECKED AGAINST THE ONE WE COMPUTE.
 *
 * `is_full_attention` derives the schedule from one integer: every `interval`-th layer is full
 * attention and the rest are linear. That is llama.cpp's rule and it is right for this model. It
 * is NOT a property of the family -- HuggingFace states the schedule explicitly, as a
 * `layer_types` array of "linear_attention" / "full_attention", and rad-convert carries that array
 * into the container as a space-separated string, which is what this function reads.
 *
 * A model whose array does not match the modular rule would be built with the WRONG LAYERS LINEAR.
 * It would convert, load, and generate fluent text from a different architecture than the one on
 * disk -- the exact silent-wrong-model failure this plugin's check_format exists to prevent one
 * layer down. So: where the container states the schedule, it is authoritative and disagreement
 * is fatal.
 *
 * A container that states nothing is accepted, for check_format's reason -- a .rad converted from
 * a GGUF carries no HuggingFace config, and the modular rule is then the only evidence there is.
 * `text_config.` is where it lands for a checkpoint whose config nests the language model. */
static int check_layer_schedule(const RadModelMeta* meta, int64_t n_layer, int64_t interval) {
    const char* s = rad_meta_gets(meta, "qwen35.layer_types", nullptr);
    if (!s) s = rad_meta_gets(meta, "text_config.layer_types", nullptr);
    if (!s) s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s || !*s) return RAD_OK;

    int64_t l = 0;
    for (const char* p = s; *p; ) {
        while (*p == ' ') ++p;
        if (!*p) break;
        const char* tok = p;
        while (*p && *p != ' ') ++p;
        const size_t len = (size_t)(p - tok);

        const bool says_full = len == 14 && std::strncmp(tok, "full_attention", 14) == 0;
        const bool says_lin  = len == 16 && std::strncmp(tok, "linear_attention", 16) == 0;
        if (!says_full && !says_lin) {
            fprintf(stderr, "radiance: qwen35: layer_types[%lld] is '%.*s'; this plugin knows "
                            "'full_attention' and 'linear_attention' and will not guess at a "
                            "third kind of layer.\n", (long long)l, (int)len, tok);
            return RAD_E_UNSUPPORTED;
        }
        /* Past the trunk is the MTP block, which the array may or may not include; it is not a
         * trunk layer and its kind is the drafter's business (see declare_draft). */
        if (l >= n_layer) break;
        if (says_full != is_full_attention(l, interval)) {
            fprintf(stderr, "radiance: qwen35: the container says layer %lld is %s, and "
                            "full_attention_interval %lld makes it %s. The interval rule is "
                            "llama.cpp's and it is not a property of this family -- a model whose "
                            "layer_types differ from it would be built with the wrong layers "
                            "linear, and would load and generate fluent text from an architecture "
                            "that is not the one in the checkpoint. Read layer_types directly "
                            "before serving this model.\n",
                    (long long)l, says_full ? "full_attention" : "linear_attention",
                    (long long)interval,
                    is_full_attention(l, interval) ? "full_attention" : "linear_attention");
            return RAD_E_UNSUPPORTED;
        }
        ++l;
    }
    if (l && l < n_layer) {
        fprintf(stderr, "radiance: qwen35: layer_types names %lld layers and the model has %lld. "
                        "A short schedule is not a schedule.\n", (long long)l, (long long)n_layer);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* The container's own statement about its format. Checked rather than assumed, because every
 * failure mode of getting it wrong is silent: a `weight_block_size` of [64,64] against kernels that
 * shift by 7 reads the right bytes with the wrong gain, and an e5m2 checkpoint decodes as e4m3 into
 * numbers that are merely implausible. A container that says nothing at all is accepted -- a .rad
 * converted from a GGUF carries no HuggingFace quantization_config -- because the weight shapes and
 * the scale planes are then the only evidence and they are checked where they are read. */
static int check_format(const RadModelMeta* meta) {
    const char* method = rad_meta_gets(meta, "quantization_config.quant_method", nullptr);
    const char* fmt    = rad_meta_gets(meta, "quantization_config.fmt", nullptr);
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
}

/* THE DFLASH2 DRAFTER: its geometry, its KV group, its buffers and its block.
 *
 * Split out of declare() because it is A SECOND MODEL and not a head. It has its own layer count,
 * head count, feed-forward width, rope base, norm convention and vocabulary claim, none of which
 * the trunk's Geom can carry -- so it starts from a COPY of the trunk's (n_embd, the deployment
 * bounds, the rank and the wire) and overwrites everything the drafter's own config states.
 *
 * Nothing here is defaulted quietly. A container whose `draft.` config does not describe a
 * drafter, or describes one this engine does not implement, is refused by name: a drafter that
 * silently ignored a scale factor would simply accept less, which reads as a quality mystery and
 * not as a missing feature (spec §17). */
static int declare_dflash2(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
                           Model& m, int base_layer) {
    Names& nm = m.nm;
    Geom   dg = m.g;   /* n_embd, the vocabulary, the deployment bounds, the rank, the wire */

    dg.n_layer       = rad_meta_geti(meta, "draft.num_hidden_layers", 0);
    dg.head_dim      = rad_meta_geti(meta, "draft.head_dim", 0);
    dg.n_head_all    = rad_meta_geti(meta, "draft.num_attention_heads", 0);
    dg.n_head_kv_all = rad_meta_geti(meta, "draft.num_key_value_heads", 0);
    dg.n_ff_all      = rad_meta_geti(meta, "draft.intermediate_size", 0);
    dg.eps           = (float)rad_meta_getf(meta, "draft.rms_norm_eps", 0.0);
    dg.theta         = (float)rad_meta_getf(meta, "draft.rope_parameters.rope_theta",
                                            rad_meta_getf(meta, "draft.rope_theta", 0.0));
    dg.rope_scale    = 1.0f;
    /* PLAIN RMS NORM. The drafter uses the ordinary norm where the trunk uses GemmaRMSNorm, and
     * the checkpoint agrees rather than only vLLM -- rad_block_dflash2.h has the numbers. Two
     * conventions in one process, which is exactly why this is a second Geom and not a flag. */
    dg.wadd = 0.0f;
    dg.wadd_qk = 0.0f;

    if (dg.n_layer <= 0 || dg.head_dim <= 0 || dg.n_head_all <= 0 || dg.n_head_kv_all <= 0 ||
        dg.n_ff_all <= 0 || !(dg.eps > 0.0f) || !(dg.theta > 0.0f)) {
        fprintf(stderr, "radiance: the container's `draft.` config does not describe a DFlash2 "
                        "drafter: layers %lld, heads %lld / kv %lld, head_dim %lld, ffn %lld, "
                        "rms_eps %g, rope_theta %g\n",
                (long long)dg.n_layer, (long long)dg.n_head_all, (long long)dg.n_head_kv_all,
                (long long)dg.head_dim, (long long)dg.n_ff_all, (double)dg.eps, (double)dg.theta);
        return RAD_E_INVAL;
    }
    int s;
    if ((s = divide_or_fail(dg.n_head_all,    dg.world, "draft num_attention_heads", &dg.n_head))    < 0) return s;
    if ((s = divide_or_fail(dg.n_head_kv_all, dg.world, "draft num_key_value_heads", &dg.n_head_kv)) < 0) return s;
    if ((s = divide_or_fail(dg.n_ff_all,      dg.world, "draft intermediate_size",   &dg.n_ff))      < 0) return s;

    /* THE DRAFTER BORROWS THE TARGET'S EMBEDDING TABLE AND lm_head -- it ships neither -- so a
     * vocabulary that disagrees is not a drafter for this target at all. */
    const long long dvocab = rad_meta_geti(meta, "draft.vocab_size", dg.n_vocab_all);
    if (dvocab != dg.n_vocab_all) {
        fprintf(stderr, "radiance: the DFlash2 drafter claims a %lld-token vocabulary and the "
                        "target has %lld. The drafter ships no embedding table and no lm_head; it "
                        "reads the target's, so the two have to be the same vocabulary.\n",
                dvocab, (long long)dg.n_vocab_all);
        return RAD_E_UNSUPPORTED;
    }

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

    /* THE QUERY BLOCK MUST BE A POWER OF TWO, and this is where that is said. The convolution
     * masks a row's position within the block with `block_size - 1`, so any other value silently
     * convolves one sequence's block into the next -- the shim rejects it, but by then the
     * deployment has loaded four gigabytes of drafter to be told no. */
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
        rad_note(b, "DFlash2 was trained at block_size %lld (%lld draft tokens) and this serve "
                    "asks for %lld. A block-diffusion drafter denoises a fixed number of mask "
                    "positions and was trained for one block length; it will run, and it will "
                    "accept less than the checkpoint can do",
                 trained, trained - 1, (long long)dc.block);

    /* Three scalars this drafter's family can carry that this implementation does not apply.
     * Each of them changes what the drafter proposes, and each would show up as acceptance that
     * is merely disappointing rather than as anything that looks like a bug. */
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

    /* THE DRAFTER'S OWN KV CACHE, a group of its own. It cannot share the trunk's: different head
     * count, different head_dim, five layers against sixteen, and -- the one that matters -- a
     * WINDOW. RAD_KV_WINDOW recycles the blocks that fall below it, so this cache is sized by the
     * drafter's 2048-token window and not by the model's context length. Its width is the trunk's,
     * --kv-cache-dtype: one flag says what every attention cache of the deployment is made of. */
    RadKVGroupDecl dd{};
    dd.kind      = dc.window > 0 ? RAD_KV_WINDOW : RAD_KV_FULL;
    dd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    dd.window    = dc.window;
    dd.n_head_kv = dg.n_head_kv;
    dd.head_dim  = dg.head_dim;
    m.kv_draft = rad_decl_kv_group(b, nm.f("kv_draft"), &dd);
    for (int64_t l = 0; l < dg.n_layer; ++l)
        RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, m.kv_draft));
    dg.kv_dtype = rad_dtype_name(dd.dtype);

    /* Activation buffers. The context pass is one row a COMMITTED TOKEN and runs over a whole
     * prefill chunk; everything after it is one row a DRAFT POSITION, `block` of them a sequence,
     * so the two halves are sized apart. None of it is large -- the drafter's cost is its
     * weights, not its activations. */
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
    /* THE FP8 TWINS. The drafter's six linears are block-scaled fp8 over a bf16 checkpoint
     * (rad_fp8.h's DenseFP8), so every activation that feeds one carries an E4M3 copy and a plane
     * of f32 scales beside it. `srcq` is the largest of them by a distance -- 50 MiB at a
     * 2048-token chunk and five taps, against the 100 MiB `dflash_src` it shadows -- and it buys
     * half the bytes of the one GEMM the context pass spends its time in. */
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
    /* GROUP-major, [n_embd/group][ceil(M/16)*16] -- quant_act_i8's asum plane, whose first
     * dimension is the K group and not the token. */
    dw.dsum    = decl_b(b, nm.f("df_dsum"), RAD_F32, {dg.n_embd / 128, spad});
    /* THIS RANK'S plane. `head_rows` is the block's own answer to whether the 2-bit head is
     * vocab-sharded, asked here rather than re-derived, because a buffer that disagreed with the
     * op declared over it is a read past its end. */
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, dc.sel_topk});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, dc.sel_topk});
    /* The vocab-parallel candidate exchange, and only when there is one: this rank's top-R as
     * (int32 id, float value) pairs and the all-gathered world_size * R of them. */
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, dc.sel_topk, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.sel_topk, 2});
    }
    /* NAMED, because the engine reads the draft out of it by name: a DFlash2 pass proposes through
     * the selector and never through the sampler's token buffer. */
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;

    /* WHAT THE ENGINE DRIVES. Block: a context pass that mirrors the trunk step, then ONE pass
     * that fills the whole block. `mask_token_id` is read HERE and nowhere else: it travels in the
     * drafter declaration, so no other component has to sniff the container for a drafter.
     *
     * `window` must be the number this plugin gated its OWN attention with, because the scheduler
     * stages the block pass against it: two readings of "how far back can the drafter see" is
     * a drafter attending to positions the scheduler did not stage. */
    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    /* THE ANCHOR ROW PREDICTS NOTHING here, so the block is one row longer than the depth and the
     * head runs over rows 1..depth. Stated rather than defaulted: it is the half of the row
     * alignment that lives in this plugin, and the other convention is a real drafter
     * (rad_builder.h, RadDrafterDecl::block). */
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

    rad_note(b, "the DFlash2 drafter is declared: %lld layers of its own at absolute %d..%d, "
                "%lld heads / %lld kv x %lld, ffn %lld, %s attention with a %lld-token window, "
                "reading the trunk at %d tap(s). ONE forward proposes %lld tokens -- there are no "
                "rounds -- and the candidate selector walks a path through the block",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.n_taps, (long long)dc.steps());
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    Model& m = g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope);
    RAD_ARCH_TRY(check_format(meta));
    /* "bf16" is the ACTIVATION dtype and it is what every op but the fp8 linears matches on: the
     * norms, rope, attention, the delta-net kernels and the collectives all run on bf16 here, and
     * the E4M3 activations exist only between quant_act_fp8 and the GEMM that reads them. Saying
     * "fp8" here would be saying it about the residual stream, which is false. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16));

#if QWEN35_MOE
    if (meta->n_expert <= 0) {
        fprintf(stderr, "radiance: qwen35moe_fp8 needs a routed checkpoint and this one declares "
                        "no experts; that is qwen35\n");
        return RAD_E_UNSUPPORTED;
    }
#else
    if (meta->n_expert > 0) {
        fprintf(stderr, "radiance: qwen35_fp8 refuses a checkpoint with %lld experts; that is "
                        "qwen35moe\n", (long long)meta->n_expert);
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
    RAD_ARCH_TRY(check_layer_schedule(meta, g.n_layer, interval));
    /* EVERY DECLARED LAYER IS A TRUNK LAYER. `mtp_num_hidden_layers` is NOT part of
     * `num_hidden_layers` in this family -- Qwen3.8-27B-FP8 ships layers-0..63 under
     * `model.language_model.layers.N` and its one MTP block separately, under an `mtp.` prefix of
     * its own, in mtp.safetensors. Subtracting it here would drop layer 63, which is a
     * full-attention layer, and the model would answer with the trunk one layer short: still a
     * distribution, still self-consistent op by op, and wrong. `nextn` is kept for the note below,
     * which is the only thing the count is good for until the MTP head is declared. */
    m.n_main = g.n_layer;

    /* Qwen3.5's norms are GemmaRMSNorm: `x_hat * (1 + w)`, not `x_hat * w`. vLLM says so by
     * importing it under the family's own name -- `from ...layernorm import GemmaRMSNorm as
     * Qwen3_5RMSNorm` -- and the checkpoint says so too: `input_layernorm.weight` sits around 0.07
     * to 0.29, which is a gain of 1.1 to 1.3 the right way round and a five-to-ten-times shrink
     * the wrong way. It applies to input_layernorm, post_attention_layernorm, the model's final
     * norm and the q/k head norms -- and NOT to the delta net's ssm_norm, which is a plain
     * RMSNormGated with gains around 0.88.
     *
     * The checkpoint answers two of the three families by itself: the layer norms (0.07..0.29)
     * and the q/k head norms (partly NEGATIVE) can only be `1 + w`. The model's FINAL norm sits at
     * 0.68..0.96, which reads as either, and takes `1 + w` because the reference implementation
     * does. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;
    const float final_wadd = 1.0f;

    m.n_rot = meta_int2(meta, "qwen35.rope.dimension_count", "rope_dimension_count", 0);
    if (m.n_rot <= 0) {
        /* BOTH SPELLINGS, because the family uses both. Qwen3.8-27B-FP8 states
         * `partial_rotary_factor` at the text_config level AND inside `rope_parameters`;
         * Qwen3.5-0.8B states it ONLY inside `rope_parameters`, and rad-convert flattens nested
         * config with a dot. Reading only one spelling refuses a container of the same
         * architecture with "no usable rotary width", which is a true statement about the key and
         * a false one about the model. */
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

    /* ---- the three KV groups. The attention cache is BF16 and `kv_dtype` follows it.
     *
     * --kv-cache-dtype fp8 MAKES IT E4M3, AND IT IS THE ONE KNOB HERE THAT IS WORTH REAL TIME
     * AT DEPTH. The attention cache is
     * `n_head_kv * head_dim * 2` bytes a token a layer -- 2 KiB a rank on this model -- and ONLY
     * the 16 attention layers pay it, because the 48 delta-net layers carry fixed-size state
     * instead. At eight sequences of 16k that is 4.3 GB of read traffic a step, which is most of
     * what depth costs at that concurrency, and halving the width halves it. It also halves what a
     * sequence costs in --vram-kv-mib, which is context and batch size as well as time.
     *
     * IT IS OFF BY DEFAULT ON A QUALITY ARGUMENT, not as a precaution. Greedy continuations from a
     * long prompt agree between bf16 and fp8 for the first few hundred tokens and then diverge,
     * and the fp8 side can produce syntactically broken code where the bf16 side stays clean --
     * and this model is a CODER, so that is the workload.
     *
     * Comparing the two formats takes more than a byte comparison: a few hundred tokens of
     * agreement is not evidence that survives a longer sample, and once the text moves the only
     * question left is whether what it moved to is WORSE, which needs reading and more than one
     * prompt. Alternate the two -- bf16 / fp8 / bf16 / fp8 -- so a difference between the FORMATS
     * can be told from a difference between two runs of one.
     *
     * `--kv-cache-dtype fp8` turns it on. The descales stay at libr4d's default of 1.0, which is
     * vLLM's default too; the kernels index them per (sequence, head), so a calibrated pair needs
     * no different mechanism.
     *
     * Nothing else moves with it: the fp8kv attention rows and their `k_descale` / `v_descale`
     * operands are in libr4d, the core sizes a page from the group's own dtype, and
     * `kv_store_fp8` is the write side. The drafter's cache takes the same width
     * (declare_dflash2). */
    RadKVGroupDecl fd{};
    fd.kind      = RAD_KV_FULL;
    fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
    fd.n_head_kv = g.n_head_kv;
    fd.head_dim  = g.head_dim;
    m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
    /* The op parameter has to be the SAME SPELLING the rows band on, and the rows band on
     * rad_dtype_name's, so it is taken from there rather than written out twice. */
    g.kv_dtype = rad_dtype_name(fd.dtype);

    /* THE SSM STATE CACHE IS f32.
     *
     * The state read and write ARE the two GDN kernels: one [128,128] state is 64 KiB at f32, and
     * a speculative step writes one PER CANDIDATE TOKEN PER HEAD, because which candidate survives
     * verification is not known until after the layer has run. At depth 7 that is hundreds of
     * megabytes a rank a step, and one of the largest single kernels after the fp8 GEMMs. An f16
     * state would halve it.
     *
     * libr4d compiles both widths (`SDT`), and it argues the numerics where the kernel lives: the
     * elements are O(0.1) with no range problem and f16's eleven mantissa bits are four more than
     * bf16's, which puts the round-trip error at 5e-4 relative. vLLM allocates this cache from
     * `--mamba-ssm-cache-dtype`, and reference serves of this model pass `float16`.
     *
     * f32 ALL THE SAME, because f16 changes the generated text, and nothing else in this plugin
     * does that without being asked to. */
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

    /* ---- activation buffers. Every activation that FEEDS A GEMM carries an fp8 twin: an E4M3
     * plane of the same shape and an f32 scale plane one column per 128. The twins are transients
     * like everything else here, so the liveness analysis shares their storage with whatever else
     * is dead at that point -- but they are not free, and the largest is `ffn`: 8192 x 17408 bytes
     * is 143 MiB against the 285 MiB the bf16 buffer it shadows already costs. */
    m.b_x       = decl_b(b, nm.f("x"),        g.act_dtype, {g.max_tok, g.n_embd});
    m.a_h.x     = decl_b(b, nm.f("h"),        g.act_dtype, {g.max_tok, g.n_embd});
    RAD_ARCH_TRY(m.a_h.declare_qs(b, nm, g, "h", g.n_embd));

    m.b_qg      = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {g.max_tok, g.n_head, 2 * g.head_dim});
    m.b_k       = decl_b(b, nm.f("attn_k"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_v       = decl_b(b, nm.f("attn_v"),   g.act_dtype, {g.max_tok, kv_dim});
    m.b_q       = decl_b(b, nm.f("attn_q"),   g.act_dtype, {g.max_tok, q_dim});
    m.a_attn.x  = decl_b(b, nm.f("attn_out"), g.act_dtype, {g.max_tok, q_dim});
    /* THE ROPE TABLE, and it is the model's rather than a layer's because sixteen layers rotate at
     * the same positions with the same theta. Only the rows a step names are written, so the fill
     * is kilobytes; the plane has a row per position, past `max_ctx` by the speculative window
     * (rope_table_rows), because the fused kernel indexes it by ABSOLUTE position. f32, so the
     * cosines the fused rotation multiplies are the ones `rope` computes. */
    /* PERSIST, not transient. A transient may share storage with another transient, and this one is
     * written once at the top of a step and read by sixteen layers spread across the whole of it --
     * so anything the planner overlaps it with lands in the middle of its live range. It is also
     * the only buffer here whose rows outlive the step that wrote them, which is what PERSIST is
     * for. */
    m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32, {rope_table_rows(g), m.n_rot},
                         RAD_BUF_PERSIST);
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim));

    /* `gdn_in` is conv_dim + v_dim and not in_dim: a and b are their own buffer here because the
     * checkpoint keeps them bf16, and the output gate is the tail of THIS one because it does not
     * (rad_block_gdn_fp8.h). The convolution and the gate are column views of it. */
    m.a_in.x    = decl_b(b, nm.f("gdn_in"),   g.act_dtype, {g.max_tok, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab      = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {g.max_tok, gc.ab_dim()});
    m.b_gq      = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gk      = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {g.max_tok, gc.n_head_k, gc.head_k});
    m.b_gv      = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    m.b_gdec    = decl_b(b, nm.f("gdn_g"),    RAD_F32,     {g.max_tok, gc.n_head_v});
    m.b_beta    = decl_b(b, nm.f("gdn_beta"), RAD_F32,     {g.max_tok, gc.n_head_v});
    /* BF16, not f32. gdn_kkt_solve WRITES this plane as `unsigned short*` and gdn_chunk_scan reads
     * it back the same way -- libr4d's rows say "bf16 k and A out" and "bf16 q/k/v/A". Declaring
     * it f32 sizes it at twice the bytes and makes the two kernels disagree about every element;
     * libref reads the operand's own dtype and is correct either way, which is exactly why nothing
     * on the host path would catch that. */
    m.b_kkt     = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16,    {g.max_tok, gc.n_head_v, gc.chunk});
    /* RANK 3, because libr4d's scan and recurrent kernels require it there -- and the quantiser
     * takes it that way, reading dim 0 as the row. */
    m.a_go.x    = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {g.max_tok, gc.n_head_v, gc.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim));

#if QWEN35_MOE
    /* ---- THE ROUTING GEOMETRY, read here rather than in the block so that a refusal names the
     * container's key and not an internal struct field. `n_ff` is already the PER-RANK expert
     * width: rad-convert falls `intermediate_size` back to `moe_intermediate_size`, which on a
     * routed checkpoint is the only one there is, and geom_from divided it by world_size.
     *
     * THE EXPERT COUNT IS NOT DIVIDED. Every rank holds every expert and routes every token the
     * same way; what tensor parallel splits is each expert's INTERMEDIATE width, which is the
     * division geom_from already made. Sharding the expert SET instead would make the two ranks
     * compute different tokens and the block's all-reduce would sum two models. */
    {
        MoeFP8::Config& mc = m.moecfg;
        mc.n_expert = meta->n_expert;
        mc.top_k    = meta->n_expert_used > 0 ? meta->n_expert_used
                                              : meta_int2(meta, "qwen35.expert_used_count",
                                                          "num_experts_per_tok", 0);
        mc.n_ff_exp = g.n_ff;
        /* `norm_topk_prob` defaults TRUE in this family (Qwen2-MoE onward), and a container that
         * does not say so gets the family's answer rather than an off switch. It is the difference
         * between routing weights that sum to one and weights that sum to whatever the top-k of a
         * softmax over 256 experts happened to be -- roughly 0.4 here, so reading it the other way
         * scales every routed contribution down by more than half. */
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
    }
    {
        /* THE STEP'S HEIGHT FOR EVERYTHING THAT IS ONE ROW A TOKEN, and a PASS's height only for
         * the four planes that carry one row per (token, slot). rad_block_moe_fp8.h states the
         * rule at Wire and says what breaks if it is not kept. */
        const MoeFP8::Config& mc = m.moecfg;
        const int64_t R = g.max_tok, K = mc.rows * mc.top_k;
        m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {R, mc.n_expert});
        m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {R, mc.top_k});
        m.b_ew      = decl_b(b, nm.f("moe_w"),      g.act_dtype, {R, mc.top_k});
        m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {K});
        m.b_eoff    = decl_b(b, nm.f("moe_off"),    RAD_I32,     {mc.n_expert + 1});
        /* PERSIST: the histogram's only reader is rad_route_report, which is not a declared op, so
         * liveness would see it dead the moment moe_scatter wrote it. See qwen4exp_fp8.cpp. */
        m.b_ecnt    = decl_b(b, nm.f("moe_cnt"),    RAD_I32,     {mc.n_expert},
                             RAD_BUF_PERSIST);
        m.b_egu     = decl_b(b, nm.f("moe_gate_up"), g.act_dtype, {K, 2 * mc.n_ff_exp});
        if (mc.expert_bf16)
            m.a_eff.x = decl_b(b, nm.f("moe_ffn"),    g.act_dtype, {K, mc.n_ff_exp});
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
    }
#else
    m.b_gate_up = decl_b(b, nm.f("gate_up"),  g.act_dtype, {g.max_tok, 2 * g.n_ff});
    m.a_ffn.x   = decl_b(b, nm.f("ffn"),      g.act_dtype, {g.max_tok, g.n_ff});
    RAD_ARCH_TRY(m.a_ffn.declare_qs(b, nm, g, "ffn", g.n_ff));
#endif

    m.b_hout    = decl_b(b, nm.f("h_out"),    g.act_dtype, {g.max_logit_rows, g.n_embd});

    /* The auxiliary taps. Gated on the DRAFTER being in the container AND on the deployment
     * intending to speculate: the buffer is one residual stream per tap, 100 MiB at a 2048-token
     * chunk and five taps, which a server that never drafts should not be carrying. `cast` is the
     * copy -- see libr4d/r4d_rows.cpp on why the destination is a column slice. */
    if (ctx->max_spec > 0) {
        const char* da = rad_meta_gets(meta, "draft.architectures", "");
        if (da && std::strstr(da, "DFlash")) {
            m.n_taps = parse_tap_layers(
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

    /* ---- vocabulary edges. The embedding is bf16 in this checkpoint. The lm_head is whatever the
     * model holds it as -- bf16 as the checkpoint ships it, or block fp8 where the recipe made it
     * so -- and it is declared at its codes' dtype, so the logits kernel that reads that encoding
     * is the one selected: the widest GEMM in the model at half the bytes when it is fp8.
     *
     * AND BOTH ARE ROW-SHARDED WITH THE VOCABULARY. Sharding one end and replicating the other
     * would put 248320 x 5120 in bf16 -- 2.43 GiB -- on EVERY rank, and that is KV cache.
     * `embed_lookup` takes a `vocab_offset`, so a rank gathers the ids in its own slice and writes
     * a zero row for the rest, and the all_reduce below sums the ranks back into x. Every token
     * lands on exactly one rank, so the sum is exact and `exact` is not the wire's choice to make
     * here.
     *
     * The name maps come first: they are how the checkpoint is asked what each weight is. */
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
    /* ROW-SHARDED with the vocabulary: g.n_vocab is THIS RANK'S share and rad_arch.h says what
     * the split is and what it buys. The two change together. */
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n_embd", g.n_embd),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                   RAD_INT("vocab_offset", g.vocab_off)),
                        RAD_WEIGHTS(m.w_tok)),
                    {}, {m.b_x});
    /* THE ENCODER'S ROWS, over the lookup and before its reduction -- which layer 0's norm folds
     * in at two ranks, so there is no other place to put them (MediaRows says why). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, g.max_tok, g.n_embd, m.b_x));
    /* EXACT, unconditionally, and not g.wire_exact. Every other all_reduce in this model sums
     * partial products where a lossy wire trades a little error for bandwidth; this one sums
     * DISJOINT rows -- each token's embedding comes from exactly one rank and every other rank
     * contributes a zero -- so a lossy wire would be pure loss on a value that is already the
     * answer. min_bytes is left at 0 for the same reason. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                          RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                               RAD_NOWEIGHTS),
                           {m.b_x}, {m.b_x});
    /* ---- the layers. */
    /* The rope table's fill, declared before the layers that read it. If it does not resolve the
     * buffer stays zero and every attention block falls back to its unfused norm + rope pair --
     * which is why `aw.cos_sin` is cleared rather than left pointing at a plane nothing fills. Its
     * band is the fused prologue's (qk_fuse_rows), so a step the prologue does not serve -- a
     * prefill chunk, where the fill's grid is one block a TOKEN -- issues no fill either. */
    m.op_rope_cs = rw(b, RAD_OP(b, "rope_table",
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
#else
    mw.x = m.b_x; mw.h = m.a_h; mw.gate_up = m.b_gate_up; mw.ffn = m.a_ffn;
#endif

    /* ---- WHERE THE RESIDUAL ADD HAPPENS, which is the one thing this chain has to agree on.
     *
     * Every block ends `x += h` and every block begins `norm(x)`, so the add can move to the front
     * of the NEXT block's norm and stop being a launch of its own (rad_fp8.h, NormQuantFP8): ~128
     * of them a step, each a full dispatch for 82 KiB of row. `fold` says a block adds its predecessor's
     * delta; `add_out` says the block after it will not, so this one must.
     *
     * TWO THINGS READ `x` BETWEEN BLOCKS and each keeps the add at the end of its block:
     *
     *   - THE DRAFTER'S TAPS. `op_tap` copies the residual stream after a named layer, and what
     *     DFlash2 conditions on has to be that layer's real output. So a tap layer's MLP keeps its
     *     own add, and the block after it has nothing to fold.
     *   - THE FINAL NORM, which is a plain `rmsnorm` and has no residual operand. The last layer
     *     keeps its add.
     *
     * The MTP head below is left unfolded for the same reason in reverse: it is a second chain
     * over a different residual buffer, it is not the shipped drafter, and two launches a step is
     * not worth a second copy of this argument. */
    auto is_tap = [&](int64_t l) {
        for (int t = 0; t < m.n_taps; ++t)
            if (m.tap_layer[t] == (int)l) return true;
        return false;
    };

    m.layers.resize((size_t)m.n_main);
    for (int64_t l = 0; l < m.n_main; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        /* The mixer folds unless it is first in the chain or its predecessor's MLP kept its add. */
        const bool mix_fold = l > 0 && !is_tap(l - 1);
        /* ...and never adds itself, because the MLP beside it always folds. */
        const bool mlp_fold = true;
        const bool mix_add  = !mlp_fold;
        const bool mlp_add  = is_tap(l) || l + 1 >= m.n_main;
        /* ...AND THE COLLECTIVE MOVES WITH THE ADD, one boundary at a time. A block's all-reduce
         * can become the next block's first pass exactly where that block folds -- the fold is
         * what says the delta in `h` is still there and that this norm is its only reader -- so
         * `ar_in` is the fold and `ar_out` is the successor's fold, which is the flag already
         * computed as `!*_add`. Two ranks or nothing: at tp1 there is no collective to fold, and
         * at four the kernel does not exist. See rad_fp8.h. */
        const bool tp2      = g.world == 2;
        /* ...AND LAYER 0 ABSORBS THE EMBEDDING'S. Its mixer does not FOLD -- there is no previous
         * block and so no delta to add -- but the embedding table is vocab-sharded, so there IS a
         * collective in front of it, and the fused norm takes one without the other. That is the
         * only place `ar_in` and `fold` come apart, and it is why they are separate flags.
         * rad_fp8.h::NormQuantFP8 declares the fused op without a residual. */
        const int  mix_ar_in  = !tp2               ? kArNone
                              : mix_fold           ? kArAny
                              : (l == 0 && m.op_embed_ar) ? kArExact   /* the embedding; see rad_fp8.h */
                                                   : kArNone;
        const bool mix_ar_out = !mix_add && tp2;
        const bool mlp_ar_in  = mlp_fold && tp2;
        const bool mlp_ar_out = !mlp_add && tp2;

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
                                          mix_fold, mix_add, mix_ar_in, mix_ar_out));
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
                                         mix_fold, mix_add, mix_ar_in, mix_ar_out));
        }

        Ffn::Src ms{};
#if QWEN35_MOE
        ms.cfg         = m.moecfg;
        ms.norm        = nm.ckpt("%s.%lld.post_attention_layernorm.weight", pre, (long long)l);
        ms.router      = nm.ckpt("%s.%lld.mlp.gate.weight",                 pre, (long long)l);
        ms.experts     = nm.ckpt("%s.%lld.mlp.experts",                     pre, (long long)l);
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
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, mlp_fold, mlp_add,
                                     mlp_ar_in, mlp_ar_out));
    }

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
     * DECLARED ONLY WHEN THE DEPLOYMENT SPECULATES. It is half a gigabyte of weights that a
     * non-speculative server would read off disk, place in VRAM and never issue, and `max_spec`
     * is the engine already saying whether it intends to draft. rad-convert declares at a
     * non-zero max_spec on purpose, because a CONTAINER should carry what the model can serve
     * even when the run that made it would not have used it.
     *
     * AND ONLY WHEN THERE IS NO DFLASH2 DRAFTER, which `n_taps` is the sign of. A container can
     * carry both; a deployment drafts with one, and DFlash2 is the better one on this target --
     * one forward a block against one chained forward a token. Declaring both would put half a
     * gigabyte of MTP weights on the card for a head nothing ever issues. */
    if (nextn > 0 && g.max_spec > 0 && m.n_taps == 0) {
        /* AT max_tok, NOT max_logit_rows. A draft round is one row a sequence, but the history
         * pass that fills the head's own K and V is one row a TOKEN and runs over a whole prefill
         * chunk. 26 MiB at a 512-token chunk, against the 477 MiB of head weights beside it. */
        m.b_mtp_h = decl_b(b, nm.f("mtp_h"), g.act_dtype, {g.max_tok, g.n_embd});
        m.b_mtp_e = decl_b(b, nm.f("mtp_e"), g.act_dtype, {g.max_tok, g.n_embd});
        m.b_mtp_c = decl_b(b, nm.f("mtp_c"), g.act_dtype, {g.max_tok, 2 * g.n_embd});
        m.b_mtp_x = decl_b(b, nm.f("mtp_x"), g.act_dtype, {g.max_tok, g.n_embd});

        /* The 2-bit draft head's working set, at max_logit_rows: only a DRAFT round reaches any of
         * it, and a draft round is one row a sequence. `mtp_dsum` is the odd one -- it is
         * quant_act_i8's group-major [n_embd/128][ceil(M/16)*16] plane, so its first dimension is
         * the K group and not the token. */
        if (mtp_draft_head_q2(b, nm.f("mtp.draft_head.weight"), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,   {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32,  {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32,  {g.n_embd / 128, dmpad});
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype, {g.max_logit_rows, g.n_vocab_all});
            /* NAMED, because the engine finds it by name: a draft round's proposal comes out of
             * row_topk and never through the sampler's token buffer. */
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32,  {g.max_logit_rows, 1});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32,  {g.max_logit_rows, 1});
        }

        /* The layer's working set is the trunk's: `attn_qg`, `attn_k`, the ffn planes and `h` are
         * all dead by the time a draft round runs, and a second set would be 150 MiB of transients
         * for one layer. Only the residual stream is the head's own -- it must not be the trunk's
         * `x`, which the next step's prefill still needs. */
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
        /* THE HEAD'S FEED-FORWARD IS ROUTED TOO, and it has its own 256 experts -- `mtp.layers.0`
         * carries `mlp.experts.*`, `mlp.gate` and a shared expert exactly as a trunk layer does.
         * That is what MtpBlockFP8 is a template for; a head hard-wired to the dense block could
         * not declare it, and the container would carry a drafter nothing could run. */
        ts.mlp.cfg     = m.moecfg;
        ts.mlp.router  = nm.ckpt("mtp.layers.0.mlp.gate.weight");
        ts.mlp.experts = nm.ckpt("mtp.layers.0.mlp.experts");
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
         * so a draft round writes its K and V at the slot the trunk's token for that position
         * owns. rad_block_mtp_fp8.h argues the case; the short version is that it makes a rejected
         * draft self-correcting instead of something to roll back. */
        RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)m.n_main, m.kv_full));
        m.mtp.media = m.media;
        RAD_ARCH_TRY(m.mtp.declare(b, nm, g, (int)m.n_main, m.kv_full, m.n_rot, m.w_tok,
                                   m.w_lm_head, final_wadd, tw, ts));
        m.have_mtp = true;

        /* WHAT THE ENGINE DRIVES. Serial: `max_spec` dependent passes, each embedding what the
         * last one proposed. `proposal` is the buffer a pass leaves its token in, and the two
         * heads differ -- the 2-bit head takes its own argmax into `mtp_didx`, the bf16 head goes
         * through the ordinary sampler chain and 0 says so. Declaring it is what keeps the engine
         * from having to work it out from buffer names. */
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
     * group. Gated on the same two things the taps are: a drafter in the container, and a
     * deployment that means to speculate. `n_taps` is already both -- it is non-zero only when
     * `draft.architectures` named a DFlash model AND max_spec was positive. */
    if (m.n_taps > 0) {
        RAD_ARCH_TRY(declare_dflash2(b, meta, ctx, m, (int)m.n_main));
        if (nextn > 0)
            rad_note(b, "the DFlash2 drafter DISPLACES the %lld-layer MTP head, which is not "
                        "declared and whose weights this container therefore does not carry. One "
                        "drafter a container: the two are 4.3 GB between them and a deployment "
                        "drafts with one of them. Convert without --draft-model for the MTP one",
                     (long long)nextn);
    }

    rad_note(b, "rank %d/%d: %lld trunk layers, %lld full attention + %lld gated delta net, "
                "n_head=%lld n_head_kv=%lld head_dim=%lld n_ff=%lld vocab=[%lld,%lld)",
             g.rank, g.world, (long long)m.n_main,
             (long long)(m.n_main / interval), (long long)(m.n_main - m.n_main / interval),
             (long long)g.n_head, (long long)g.n_head_kv, (long long)g.head_dim,
             (long long)g.n_ff, (long long)g.vocab_off, (long long)(g.vocab_off + g.n_vocab));
    rad_note(b, "weights are block-scaled fp8: E4M3 [N,K] with a bf16 [N/%d,K/%d] scale plane, "
                "one stored weight taken by gemm_nt_q at dtype fp8a8 as two operands -- no "
                "repacking and no dequantisation, so the model costs what the checkpoint does",
             RAD_FP8_BLOCK, RAD_FP8_BLOCK);
    rad_note(b, "activations are quantised to E4M3 every step (activation_scheme: dynamic): one "
                "quant_act_fp8 per activation that feeds a GEMM, not per GEMM -- the three "
                "attention projections share one, and so do the delta net's qkv and z");
    rad_note(b, "NOT fp8 in the checkpoint: every norm, A_log, dt_bias, conv1d, the embedding, the "
                "lm_head (unless a recipe made it so), and the delta net's in_proj_a / in_proj_b "
                "-- 48 columns each, which is not a whole %d-row scale block, so the checkpoint "
                "left them bf16 and this plugin runs them as a separate bf16 gemm_nt instead of "
                "fusing them into the qkv row", RAD_FP8_BLOCK);
    rad_note(b, "gdn: %lld key heads x %lld, %lld value heads x %lld, conv width %lld, chunk %lld; "
                "conv row %lld channels, a|b row %lld",
             (long long)gc.n_head_k, (long long)gc.head_k, (long long)gc.n_head_v,
             (long long)gc.head_v, (long long)gc.conv_width, (long long)gc.chunk,
             (long long)conv_dim, (long long)gc.ab_dim());
    rad_note(b, "ssm_a_log carries A_log AS A LOG: the kernels form -exp(A_log)*softplus(a+dt). "
                "A container built from a GGUF has ssm_a = -exp(A_log) already applied and must "
                "undo it, or every head's decay collapses");
    rad_note(b, "rope is mrope over %lld of %lld head dims, sections [11,11,10]; RadBatch carries "
                "one position component per token, so the three sections see the same position -- "
                "exact for text, wrong for an image span",
             (long long)m.n_rot, (long long)g.head_dim);
    if (!have_interval)
        rad_note(b, "the container did not state full_attention_interval; assuming %lld, which "
                    "makes layers 3,7,... full attention", (long long)interval);
    if (m.have_mtp)
        rad_note(b, "the MTP head is declared: %lld NextN layer(s) exist under an `mtp.` prefix of "
                    "their own and the FIRST is served, as absolute layer %lld of the "
                    "full-attention KV group. It drafts; the trunk verifies. mtp.fc reads the "
                    "normed embedding then the normed hidden state",
                 (long long)nextn, (long long)m.n_main);
    else if (nextn > 0)
        rad_note(b, "%lld MTP/NextN layer(s) sit BESIDE the %lld trunk layers -- under an `mtp.` "
                    "prefix of their own, not as layers %lld.. -- and are NOT declared, because "
                    "this deployment asked for no speculative window (max_spec 0). Half a "
                    "gigabyte of weights stays off the card",
                 (long long)nextn, (long long)m.n_main, (long long)m.n_main);

    /* THE VISION TOWER, LAST: an encoder pass issues nothing else, so its activations share the
     * arena with every buffer declared above. Its weight groups start past every text layer, the
     * MTP head and a drafter's. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];

    /* AN ENCODER PASS RUNS THE VISION TOWER AND NOTHING ELSE (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* AN MTP PASS runs the head and NOTHING ELSE. The trunk already ran this step and left its
     * hidden states in `h`; the head reads the rows it wants out of them and turns them into
     * either a draft token or one more position of its own attention history. Which of the two
     * is the SIGN of `draft_pass`, and rad_block_mtp_fp8.h says why there are two.
     *
     * Zero is a trunk step, so a zeroed RadBatch runs the model -- the same convention, and for
     * the same reason, as `num_accepted` (docs/OPS.md). */
    if (batch->draft_pass != 0) {
        /* THE DRAFTER'S OWN cos/sin ROWS FIRST, AND THE TRUNK DID NOT WRITE THEM. `rope_table`
         * fills the rows THIS STEP'S POSITIONS name, and a drafter pass runs PAST the committed
         * end -- the MTP head's round k is at position head_i + k - 1, which the trunk step
         * before it never reached. Without this the head rotates by whatever the plane held at a
         * row nothing had written, which costs only ACCEPTANCE (a mis-rotated draft is rejected
         * and the text is still the trunk's) and therefore has no symptom at all.
         *
         * A row is a function of its POSITION ALONE, so re-filling is idempotent and can never
         * write a row a later reader disagrees with. qwen4exp_fp8.cpp carries the same call for
         * the same reason and says more about it. */
        if (m.op_rope_cs && batch->n_tok <= qk_fuse_rows(m.g) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, batch->n_tok,
                        rope_pos1(batch, batch->n_tok), RAD_B(m.b_rope_cs));
        /* THE FIELD MEANS "A DRAFTER PASS, AND THE SIGN SAYS WHICH KIND", and both drafters read
         * it that way even though only the MTP head has rounds. For DFlash2 the two kinds are the
         * two passes: positive is the QUERY pass, which proposes the whole block in one go, and
         * negative is the CONTEXT pass, which turns the tokens the trunk just committed into the
         * drafter's own K and V. There is no third kind and no round counter -- a block-diffusion
         * drafter runs its layers once per step. */
        if (m.have_df) {
            if (batch->draft_pass > 0) m.df.step(c, batch);
            else                      m.df.context(c, batch);
            return;
        }
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container has neither an `mtp.` block nor a `draft.` one, or "
                             "declare ran at max_spec 0, which is the engine saying it would not "
                             "draft");
            return;
        }
        m.mtp.step(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, batch->n_tok), RAD_W(m.w_tok),
              brows(m.b_x, batch->n_tok));
    m.mrows.step(c, batch, m.b_x, batch->n_tok);
    /* The vocab shard's reduction -- ONLY WHERE LAYER 0'S NORM WILL NOT DO IT, which is the same
     * sentence every block in this model already carries about its own collective. At tp1 there
     * is no op and the graph is what it always was; at tp2 the fused norm takes it and this is
     * not issued at all, which for n_embd 5120 is every step. The standalone op stays DECLARED
     * because ar_fused_ok is a question about T and a width, and a model whose n_embd the
     * compressed wire cannot tile would fall back to it. */
    if (m.op_embed_ar && !ar_wire_is_exact(m.g, batch->n_tok))
        RAD_ISSUE_N(c, m.op_embed_ar, (int64_t)batch->n_tok * m.g.n_embd,
                    brows(m.b_x, batch->n_tok), RAD_NONE);

    /* ONE FILL A STEP, HERE, because every attention layer below reads it and they all rotate at
     * the same positions. Only the rows this step names are written -- and only on a step whose
     * width the fusion actually serves, because the table has exactly one consumer and a prefill
     * chunk takes the unfused pair. The band is asked of `qk_fuse_rows` rather than restated, so
     * the fill and the op that reads it cannot disagree about which steps have a table. */
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

        /* THE TAP, AFTER the layer, because what a drafter conditions on is this layer's OUTPUT.
         * The residual stream is one buffer the next layer overwrites, so the rows are copied into
         * the column slice this tap owns while they are still here. Linear in the tap count and
         * only over the layers named. */
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
 * WHAT THIS CONTAINER'S DRAFTER IS WORTH SPECULATING TO, answered before anything is declared
 * because `max_spec` is an input to declaration -- it sizes the drafter's buffers, bands the
 * kernels and sets the scheduler's window. The core asks; it reads none of these keys itself.
 *
 * THE DRAFT DEPTH IS A PROPERTY OF THE DRAFTER, not something an operator has to remember. The two
 * numbers below are far apart because the two drafters propose in different shapes.
 *
 * An MTP head is SERIAL: depth d is d chained forward passes, each one embedding the token the
 * last one proposed, so the marginal token gets more expensive exactly as it gets less likely. On
 * Qwen3.8-27B-FP8 at two ranks throughput is flat across depths 2, 3 and 4 and falls away after,
 * so 3 is the top of that plateau.
 *
 * DFlash2 is block-diffusion and proposes the WHOLE block in one pass, so its depth is not a
 * throughput trade at all: it is the block length the checkpoint was trained at, and asking for
 * anything else is a silently worse draft. This one ships trained at block_size 8, which is one
 * anchor and 7 mask positions.
 *
 * 0 when the container carries no drafter, which is the right answer to "speculate as deeply as
 * this model can" on a model that cannot. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
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
/* ONE MORE LEVEL OF EXPANSION, and it is load-bearing. RAD_ARCH_PLUGIN stringifies its NS argument
 * to name the plugin, and `#` applies to the parameter as WRITTEN -- so passing QWEN35_NS directly
 * would register a plugin called "QWEN35_NS", which the registry would then refuse the second copy
 * of by name. An argument that is not stringified in the intermediate macro IS macro-expanded
 * before substitution, so these two forward the expanded namespace. */
#define QWEN35_PROBE_X(NS)              RAD_ARCH_PROBE(NS)
#define QWEN35_PLUGIN_X(NS, ...)        RAD_ARCH_PLUGIN(NS, __VA_ARGS__)
QWEN35_PROBE_X(QWEN35_NS)
#if QWEN35_MOE
QWEN35_PLUGIN_X(QWEN35_NS, QWEN35_ARCH_ID, "fp8_e4m3", "0.1.0",
                "Qwen3.5 / 3.6 MoE (35B-A3B), block-scaled fp8 weights and fp8 activations: the "
                "dense graph with a routed feed-forward -- 256 experts, 8 a token, a shared expert "
                "beside them -- gated attention over paged KV every fourth layer and gated delta "
                "net between")
#else
QWEN35_PLUGIN_X(QWEN35_NS, QWEN35_ARCH_ID, "fp8_e4m3", "0.1.0",
                "Qwen3.5 / 3.6 / 3.8 dense (0.8B..27B), block-scaled fp8 weights and fp8 "
                "activations, gated attention over paged KV every fourth layer and gated delta "
                "net between")
#endif
#endif
