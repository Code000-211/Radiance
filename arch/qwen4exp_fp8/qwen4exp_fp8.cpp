/* qwen4exp_fp8 -- Qwen4-Exp (`model_type: qwen4_exp`), the architecture of Qwen3.8-Flash-Next.
 * The checkpoint is dense bf16, quantised ahead of time by a recipe
 * (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe is the served one): int8 or fp8 trunk linears with
 * matching activations, four-bit routed experts, or bf16 throughout.
 *
 * ============================== RELATION TO QWEN3.5 MOE ==============================
 *
 * In transformers it is `Qwen4ExpTextModel`, a subclass of `Qwen3_5MoeTextModel`, which this tree
 * already serves (arch/qwen35moe_fp8). Most of this file composes the same blocks:
 *
 *   reused                     the Gated DeltaNet (conv width 4, key/value head dim 128), gated
 *                              full attention (`attn_output_gate`: q_proj twice as wide, half of
 *                              it a sigmoid gate on the output), partial rotary (64 of 256 dims),
 *                              the routed feed-forward with a gated shared expert, the MTP
 *                              position.
 *   new                        the gated residual (arch/common/rad_block_hc.h), the n-gram / PLE
 *                              layer (arch/common/rad_block_ple.h), QSA's indexer
 *                              (arch/common/rad_qsa.h), stacked dense expert tensors
 *                              (tools/rad_convert.cpp, slice_stacked), gqa 12, and the delta net's
 *                              sigmoid output gate.
 *
 * Qwen3.8-Flash-Next: 48 layers of 12 x (3 x GatedDeltaNet -> MoE, then 1 x QSA -> MoE), hidden
 * 2560, hc_count 4 so the residual stream is 10240, 512 experts of 640 at top-10 plus a shared 640,
 * 24 query heads and 2 KV heads at head_dim 256, vocab 248320.
 *
 * ============================== THE RESIDUAL STREAM IS FOUR TIMES AS WIDE ==============================
 *
 * The checkpoint has no `input_layernorm`, `post_attention_layernorm` or final `norm`. Instead
 * every block has a gated residual connection: a grouped norm over the 10240-wide stream, a
 * low-rank data-dependent read gate that mixes the four sub-streams down to 2560, and four
 * data-dependent scalars that add the block's output back into all four. There are 96, plus a 97th
 * with no write half that collapses the stream for the lm_head.
 *
 * arch/common/rad_block_hc.h holds the arithmetic and docs/OPS.md's `hc_read` / `hc_write` rows
 * the schema. The blocks do not own their norms: this plugin passes `Src::norm = nullptr`, the fp8
 * blocks' "input already normed and quantised" mode, which forgoes folding the residual add and
 * the all-reduce into the next norm. The all-reduce rides in the gated residual's write instead,
 * where a kernel serves that (HyperConn::Config::reduce, see declare()).
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * The checkpoint is bf16 throughout (no `quantization_config`), so the quantisation descriptor is
 * empty and the served formats come from a recipe (data/recipes/qwen4exp-*.recipe). The plugin
 * declares each weight at its encoding, and blocks that read several formats choose their ops by
 * it:
 *
 *   block fp8 or int8  self_attn.{q,k,v,o}_proj, linear_attn.{in_proj_qkv, in_proj_z, out_proj},
 *                      the shared expert's three (LinearFP8); the indexer's qk projection
 *                      unless the recipe leaves it bf16, as the served one does
 *   fp8, four-bit      every routed expert's gate_up and down, each layer and projection in its
 *   or bf16            own form (MoeFP8::probe)
 *   E4M3 rows or bf16  each connection's input_mix_weight_{down,up}, whose lowrank of 320 is 2.5
 *                      scale blocks and so is scaled a row group at a time (HyperConn), and the
 *                      MTP head's fc_hidden / fc_embedding
 *   fp8, int8, bf16    the lm_head (whichever logits_gemm kernel reads it)
 *   BF16               every norm gain, A_log, dt_bias, conv1d, in_proj_a / in_proj_b (48 rows is
 *                      not a whole scale block), the router, shared_expert_gate, embed_tokens,
 *                      the PLE projections and inject weights
 *
 * ============================== HOW TENSOR PARALLEL SPLITS AN EXPERT ==============================
 *
 * `moe_intermediate_size` is 640 = 5 x 128. An even tp2 share, 320, is not a whole number of
 * 128-column blocks, on which the fp8 scale block, the 4-bit scale group and the rotation sit. So
 * each rank takes three blocks of the experts of one parity and two of the other's; the router is
 * replicated so both ranks route alike, each rank computes its slice of every routed expert, and
 * the block's existing end-of-block all-reduce sums the partials, with no all-to-all. See the
 * section in declare().
 *
 * ============================== PLE AND QSA ==============================
 *
 * PLE, the n-gram embedding, is 51B of the model's 180B parameters: a 320,001,536-row table of
 * 160-dim embeddings, gathered 16 rows per token by a hash of the last three token ids, read once
 * at layer 1 and injected into the wide stream.
 *
 * QSA's indexer picks the best `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens
 * per query and attends only to those. Below `indexer_budget + indexer_compress_ratio - 1` = 2051
 * tokens of context every complete block is selected, so QSA equals dense causal attention bit for
 * bit; the engine issues the cheaper dense attention there, and it serves as the oracle for QSA.
 * Above it the selection is per query on both prefill and decode (docs/QSA.md,
 * arch/common/rad_qsa.h).
 *
 * The vision tower is declared through arch/common/rad_block_vit.h, only when the container
 * carries one and the deployment asked for media; rad-convert writes only declared weights.
 */
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <arch/rad_arch.h>
#include <arch/rad_block_attn_gated_fp8.h>
#include <arch/rad_block_gdn_fp8.h>
#include <arch/rad_block_hc.h>
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_block_mtp_hc.h>
#include <arch/rad_block_ple.h>
#include <arch/rad_qsa.h>
#include <arch/rad_block_vit.h>

namespace qwen4exp_fp8 {

using namespace rad::arch;

struct Layer {
    int          full = 0;
    HyperConn    hc_mix;     /* in front of the attention or the delta net */
    HyperConn    hc_ffn;     /* in front of the routed feed-forward */
    AttnGatedFP8 attn;
    QsaIndexer   qsa;        /* full-attention layers only; inert when the checkpoint has none */
    GdnFP8       gdn;
    MoeFP8       mlp;
    /* The control vector's projection, applied to the wide stream right after this layer's
     * feed-forward write -- where llama.cpp's `build_cvec(res_hc, il)` sits in the `qwen4exp`
     * graph. Inert (no op) when the deployment has no vector or this layer is outside its range
     * (arch/common/rad_block_hc.h). */
    HcProject    proj;
};

struct Model {
    Names          nm{""};
    Geom           g{};
    GdnFP8::Config gcfg{};
    MoeFP8::Config moecfg{};
    /* The trunk's linears are int8 (LinearFP8::i8), as read from the first delta-net layer's input
     * projection before the activation buffers are declared. */
    bool           trunk_i8 = false;
    HyperConn::Config hccfg{};
    QsaIndexer::Config qsacfg{};
    QsaIndexer::Wire   qsaw{};
    rad_kvgroup        kv_qsa_bkey = 0;   /* created by the first QSA layer's declare() */

    int64_t n_rot = 0;

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;
    /* PLE lives on exactly one layer; -1 when the checkpoint has none or this build does not
     * declare that layer. */
    PleLayer    ple;
    int64_t     ple_layer = -1;

    /* The wide stream and the narrow block input. `b_h` is the residual stream, hc * n_embd per
     * token; `a_x` is what a connection's read produces and where its block writes the delta. One
     * of each for the whole model: a read is immediately followed by its block and its write, so
     * nothing in `a_x` or `b_inj` outlives the write that consumes it. Declaration order determines
     * the planner's liveness (see the note at declare()), so this holds by this file's order. */
    rad_buf b_h = 0;
    ActFP8  a_x{};
    rad_buf b_inj = 0;

    /* the attention block's own activations, shared across the 12 attention layers */
    rad_buf b_qg = 0, b_k = 0, b_v = 0, b_q = 0, b_rope_cs = 0, b_gate = 0;
    ActFP8  a_attn{};

    /* the delta net's */
    ActFP8  a_in{}, a_go{};
    rad_buf b_ab = 0, b_gq = 0, b_gk = 0, b_gv = 0, b_gdec = 0, b_beta = 0, b_kkt = 0;

    /* the routed feed-forward's */
    rad_buf b_rlogits = 0, b_eids = 0, b_ew = 0, b_sorted = 0, b_eoff = 0, b_ecnt = 0;
    rad_buf b_egu = 0, b_edn = 0, b_emoe = 0, b_sgu = 0, b_sout = 0, b_sgate = 0;
    ActFP8  a_eff{}, a_sff{};
    /* The rotated codes of the block input, for the routed arm only: `q` and `s` are used and `x`
     * stays zero, since the bf16 source is `a_x`, unrotated. Declared only when the container asks
     * for a rotation; zero handles tell the MoE block the pair is absent. */
    ActFP8  a_hr{};

    rad_buf b_hout = 0, b_logits = 0;

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h). Declared only when
     * the deployment speculates, since its ~2.5 GiB of weights would otherwise be loaded and never
     * issued. rad-convert declares at a non-zero max_spec so the container carries everything the
     * model can serve. */
    bool        have_mtp = false;
    MtpHcBlock  mtp{};
    rad_buf     b_mtp_hin = 0, b_mtp_hw = 0, b_mtp_e = 0, b_mtp_hout = 0;
    rad_buf     b_mtp_dq = 0, b_mtp_ds = 0, b_mtp_dsum = 0;
    rad_buf     b_mtp_dlog = 0, b_mtp_didx = 0, b_mtp_dval = 0;
    /* RADIANCE_MTP_COUPLED: [max_seqs] draws staged before the serial passes and [max_spec,
     * max_seqs] states read back after them. PERSIST: the engine writes them from the host and the
     * head writes the states on the card, across a step boundary. */
    rad_buf     b_mtp_couple_draws = 0, b_mtp_couple_state = 0;
    rad_buf     b_mtp_dpair = 0, b_mtp_dgath = 0;

    HcEnter   enter;
    HyperConn mixer;            /* the 97th connection: no write half */
    /* Whether this rank has said that it pipelines a prefill step (step_pipelined), once. */
    bool      pipe_noted = false;
    /* Whether the deployment spans (RadBuildCtx::max_mixed_tok below a max_tok wider than a MoE
     * pass): a step wider than a pass is then one sequence, the connections read it a pass at a
     * time and the indexer scores it in windows. `pipe_on` adds ranks whose blocks issue their own
     * all-reduce at such a step (declare): it is pipelined (step_pipelined) and its all-reduces
     * are declared at `overlap` 1. Without a span neither is set and none of these paths is
     * declared or taken. A sizing declare runs at a max_tok no wider than a pass and takes both
     * from the real declare, so it declares the same ops. */
    bool      span_on = false;
    bool      pipe_on = false;
    /* {0, pass, T} for a step wider than a pass (pipe_bounds): the windows of the indexer's score
     * plane and the delta net's pieces. A transient (declare), uploaded by every step that reads
     * it -- the first two from the pinned host copy `pipe_cu_h`, made once. */
    rad_buf   b_pipe_cu = 0;
    int32_t*  pipe_cu_h = nullptr;
    bool      pipe_cu_tried = false;

    std::vector<Layer> layers;

    rad_weight w_tok = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_rope_cs = 0, op_gather = 0, op_logits = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower and the scatter of its rows over
     * the token embeddings, declared only when the container carries a tower and the deployment
     * asked for media (RadBuildCtx::max_enc_patches). */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

};

static Model g_model[MAX_RANKS];

/* The n-gram table's row count, by prime search.
 *
 * Each of the `heads` hash heads has its own vocabulary: the 1st, 2nd, ... nth prime above
 * `ngram_vocab_size_base - 1`; the table is their sum, padded to `divisor`
 * (`Qwen4ExpTextNGramEmbedding.__init__`). The per-head sizes are in the checkpoint
 * (`ngram_heads_vocab_sizes`) and the served hash reads them there; only the total is derived,
 * because a weight's shape must be declared before any tensor is opened. rad-convert checks it:
 * a multi-source concat's declared leading extent must equal its sources'.
 *
 * Cost: 16 primes near 2e7 by trial division, once at declare. */
static bool ngram_is_prime(int64_t v) {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2)
        if (v % d == 0) return false;
    return true;
}

static int64_t ngram_table_rows(int64_t base, int64_t heads, int64_t divisor) {
    if (base <= 1 || heads <= 0) return 0;
    int64_t total = 0, p = base - 1;
    for (int64_t i = 0; i < heads; ++i) {
        do { ++p; } while (!ngram_is_prime(p));
        total += p;
    }
    return divisor > 1 ? ((total + divisor - 1) / divisor) * divisor : total;
}

/* A rank's place in the shared expert's uneven block split, where the first places take one
 * block more. MoeFP8::declare hands a plain expert's extra 64-wide tiles to the first ranks by
 * index; the shared expert's extra blocks start at the rank after the last of those, so the two
 * remainders land on different ranks when there are enough. (Routed experts leave no remainder:
 * the parity split at two ranks and the four-class split at four are even.) At four ranks and 640
 * columns (ten tiles, five blocks) ranks 0 and 1 take a tile more and rank 2 the block; at two the
 * place is the rank. */
static int64_t shared_place(int64_t n_ff_all, int world, int rank) {
    const int64_t first = (n_ff_all / 64) % world;
    return (rank - first + world) % world;
}

/* Columns [lo, hi) of the slice at `place` when `nb` fp8 blocks split over `world` places, the
 * first nb % world places taking one block more. */
static void block_slice(int64_t nb, int world, int64_t place, int64_t* lo, int64_t* hi) {
    const int64_t base = nb / world, extra = nb % world;
    const int64_t b0 = place * base + std::min<int64_t>(place, extra);
    *lo = b0 * RAD_FP8_BLOCK;
    *hi = (b0 + base + (place < extra ? 1 : 0)) * RAD_FP8_BLOCK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    /* A sizing declare writes scratch: the step reads g_model, filled by the real declare at the
     * real max_tok (RadBuildCtx::shape_probe). */
    static thread_local Model probe_model;
    Model& m = ctx->shape_probe ? probe_model : g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope ? ctx->scope : "");
    Names& nm = m.nm;

    /* arch_splits_ff: the plugin splits the routed experts' width (640 = five fp8 blocks) itself,
     * by whole blocks and parity, at any world up to the block count (including three, which 640
     * does not divide), so geom_from must not refuse it. attn_world_ok: this plugin's attention
     * blocks (rad_block_attn_gated_fp8.h) serve a smaller attention world when the two KV heads do
     * not divide the world. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16, /*arch_splits_ff=*/true,
                           /*attn_world_ok=*/true));
    Geom& g = m.g;

    /* Qwen4-Exp's norms are all `x_hat * (1 + w)` (`Qwen3NextRMSNorm.forward` is
     * `_norm(x.float()) * (1.0 + weight.float())`, weight initialised to zeros); the delta net's
     * gated output norm is plain `weight * x` (initialised to ones). Taken from the reference. */
    g.wadd    = 1.0f;
    g.wadd_qk = 1.0f;

    /* ---- the gated residual's geometry. */
    m.hccfg.hc      = meta_int2(meta, "qwen4exp.hc_count",   "hc_count",   0);
    m.hccfg.lowrank = meta_int2(meta, "qwen4exp.hc_lowrank", "hc_lowrank", 0);
    if (m.hccfg.hc <= 1 || m.hccfg.lowrank <= 0) {
        fprintf(stderr, "radiance: qwen4exp: the container does not describe the gated residual "
                        "(hc_count %lld, hc_lowrank %lld). Every norm in this architecture is one "
                        "of these connections, so there is nothing to fall back to.\n",
                (long long)m.hccfg.hc, (long long)m.hccfg.lowrank);
        return RAD_E_INVAL;
    }
    /* Every block's all-reduce rides in the following write where a kernel serves that
     * (HyperConn::Config::reduce): the block's partial sum is the write's input, so the pair is one
     * launch. Each block is told below whether its connection's fused op resolved. */
    m.hccfg.reduce = 1;
    const int64_t HN = m.hccfg.hc * g.n_embd;

    /* ---- partial rotary: 0.25 of head_dim 256 is 64; the other 192 dims pass through. */
    {
        const double pr = rad_meta_getf(meta, "text_config.rope_parameters.partial_rotary_factor",
                          rad_meta_getf(meta, "rope_parameters.partial_rotary_factor",
                          rad_meta_getf(meta, "partial_rotary_factor", 1.0)));
        m.n_rot = (int64_t)((double)g.head_dim * pr);
        if (m.n_rot <= 0 || m.n_rot > g.head_dim || (m.n_rot % 2)) {
            fprintf(stderr, "radiance: qwen4exp: partial_rotary_factor %.6g of head_dim %lld is "
                            "%lld rotary dims, which is not an even count in (0, head_dim]\n",
                    pr, (long long)g.head_dim, (long long)m.n_rot);
            return RAD_E_INVAL;
        }
    }
    RAD_ARCH_TRY(yarn_from(g, meta, "", m.n_rot));
    /* There is deliberately no serve-time layer slice. A model too large for the device needs a
     * converter flag that writes a named partial container, not a switch that silently serves less
     * than the model. */

    /* ---- the layer schedule. */
    const int64_t interval = meta_int2(meta, "qwen4exp.full_attention_interval",
                                       "full_attention_interval", 0);
    if (interval <= 1) {
        fprintf(stderr, "radiance: qwen4exp: full_attention_interval is %lld; this is a hybrid "
                        "architecture and the interval decides which layers are linear\n",
                (long long)interval);
        return RAD_E_INVAL;
    }
    /* The config file says "full_attention" for the QSA layers (HuggingFace's post_init rewrites
     * them to "qwen_sparse_attention" in memory), so both spellings mean a full layer. */
    RAD_ARCH_TRY(check_layer_schedule("qwen4exp", meta_text_gets(meta, "layer_types", nullptr),
                                      g.n_layer, interval,
                                      {"full_attention", "qwen_sparse_attention"}));

    /* ---- the delta net. */
    {
        GdnFP8::Config& gc = m.gcfg;
        const int64_t nk = meta_int2(meta, "qwen4exp.ssm.key_head_count",
                                     "linear_num_key_heads", 0);
        const int64_t nv = meta_int2(meta, "qwen4exp.ssm.value_head_count",
                                     "linear_num_value_heads", 0);
        gc.head_k     = meta_int2(meta, "qwen4exp.ssm.key_head_dim",   "linear_key_head_dim",   0);
        gc.head_v     = meta_int2(meta, "qwen4exp.ssm.value_head_dim", "linear_value_head_dim", 0);
        gc.conv_width = meta_int2(meta, "qwen4exp.ssm.conv_kernel",    "linear_conv_kernel_dim", 0);
        gc.chunk      = 64;
        /* `output_gate_type` is a checkpoint property: Qwen3.5/3.6 leave it unset and mean silu,
         * Qwen4-Exp states "sigmoid". libr4d's gated norm has both. */
        const char* ogt = rad_meta_gets(meta, "text_config.output_gate_type",
                          rad_meta_gets(meta, "output_gate_type", nullptr));
        if (!ogt || !*ogt) ogt = rad_meta_gets(meta, "text_config.hidden_act",
                                rad_meta_gets(meta, "hidden_act", "silu"));
        gc.act = ogt;
        if (nk <= 0 || nv <= 0 || gc.head_k <= 0 || gc.head_v <= 0 || gc.conv_width <= 0) {
            fprintf(stderr, "radiance: qwen4exp: the container does not describe the linear "
                            "layers (key heads %lld, value heads %lld, key width %lld, value "
                            "width %lld, conv width %lld)\n",
                    (long long)nk, (long long)nv, (long long)gc.head_k, (long long)gc.head_v,
                    (long long)gc.conv_width);
            return RAD_E_INVAL;
        }
        /* Heads split evenly when they can and unevenly otherwise: 16 K heads over three ranks
         * are 6, 5 and 5, each rank's V heads the same multiple of its K heads (48 / 16 = 3 ->
         * 18, 15, 15), declared as spans of the container's rows (rad_block_gdn_fp8.h,
         * Config::k_first). Nothing is padded; the kernels take any head count. A V count that is
         * not a multiple of the K count is refused. */
        gc.n_head_k_all = nk;
        gc.n_head_v_all = nv;
        if (g.world > 1 && (nk % g.world != 0 || nv % g.world != 0)) {
            if (nv % nk != 0) {
                fprintf(stderr, "radiance: qwen4exp: %lld value heads over %lld key heads do not "
                                "group, so an uneven split over %d ranks is not defined\n",
                        (long long)nv, (long long)nk, g.world);
                return RAD_E_INVAL;
            }
            const int64_t ratio = nv / nk, base = nk / g.world, rem = nk % g.world;
            /* Extra heads go first to ranks without attention (rad_arch.h remainder_place);
             * with none, this is the rank's own place. */
            const int64_t pl = remainder_place(g, g.rank, 0);
            gc.n_head_k = base + (pl < rem ? 1 : 0);
            gc.k_first  = pl * base + (pl < rem ? pl : rem);
            gc.n_head_v = ratio * gc.n_head_k;
            gc.v_first  = ratio * gc.k_first;
            rad_note(b, "rank %d takes delta-net K heads [%lld, %lld) of %lld and V heads "
                        "[%lld, %lld) of %lld", g.rank,
                    (long long)gc.k_first, (long long)(gc.k_first + gc.n_head_k), (long long)nk,
                    (long long)gc.v_first, (long long)(gc.v_first + gc.n_head_v), (long long)nv);
        } else {
            int s;
            if ((s = divide_or_fail(nk, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0) return s;
            if ((s = divide_or_fail(nv, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0) return s;
        }
    }

    /* ---- the routed feed-forward.
     *
     * n_ff is `moe_intermediate_size`, divided by world_size when the world divides it and whole
     * otherwise (geom_from's `arch_splits_ff`). At more than one rank the split below sets the
     * rank's width, by whole fp8 blocks and parity (see the file header). */
    {
        MoeFP8::Config& mc = m.moecfg;
        mc.n_expert    = meta_int2(meta, "qwen4exp.expert_count",      "num_experts", 0);
        mc.top_k       = meta_int2(meta, "qwen4exp.expert_used_count", "num_experts_per_tok", 0);
        mc.n_ff_exp    = g.n_ff;
        mc.n_ff_shared = meta_int2(meta, "qwen4exp.expert_shared_feed_forward_length",
                                   "shared_expert_intermediate_size", 0);
        /* The shared expert is split by whole fp8 scale blocks: evenly when possible, otherwise
         * as evenly as whole blocks allow, by the rank's place (shared_place). Its 640 columns are
         * five blocks: three on rank 0 and two on rank 1; at four ranks two on rank 2 and one on
         * each other. See MoeFP8::Config::shared_lo. */
        if (mc.n_ff_shared > 0 && g.world > 1 && (mc.n_ff_shared / g.world) % RAD_FP8_BLOCK == 0) {
            int s;
            if ((s = divide_or_fail(mc.n_ff_shared, g.world,
                                    "shared_expert_intermediate_size", &mc.n_ff_shared)) < 0)
                return s;
        } else if (mc.n_ff_shared > 0 && g.world > 1) {
            if (mc.n_ff_shared % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: the shared expert is %lld columns, which is not a whole "
                                "number of %d-column fp8 scale blocks, so it cannot be split "
                                "across %d ranks.\n", (long long)mc.n_ff_shared, RAD_FP8_BLOCK,
                        g.world);
                return RAD_E_INVAL;
            }
            const int64_t nb = mc.n_ff_shared / RAD_FP8_BLOCK;
            block_slice(nb, g.world, shared_place(g.n_ff_all, g.world, g.rank),
                        &mc.shared_lo, &mc.shared_hi);
            mc.n_ff_shared = mc.shared_hi - mc.shared_lo;
            rad_note(b, "the shared expert does not split evenly into %d-column fp8 scale blocks: "
                        "rank %d computes columns [%lld, %lld) of it, %lld of %lld blocks, and the "
                        "all-reduce sums the partials", RAD_FP8_BLOCK, g.rank,
                     (long long)mc.shared_lo, (long long)mc.shared_hi,
                     (long long)(mc.n_ff_shared / RAD_FP8_BLOCK), (long long)nb);
        }
        mc.norm_topk   = 1;
        /* GPTQ calibration run, enabled by the same setting that names the output directory (a
         * calibration with nowhere to write would be a 1.34 GB allocation and two launches per
         * layer for nothing). A separate run from serving; see MoeFP8::Config::calib. */
        mc.calib       = calib_dir() != nullptr ? 1 : 0;
        /* Tokens per pass. 2048 is the linear-state checkpoint interval the scheduler clamps the
         * prefill chunk to, so the shipped configuration is one pass. It also sizes four buffers at
         * `rows * top_k`, so a smaller max_tok lowers it. */
        mc.rows        = g.max_tok < 2048 ? g.max_tok : 2048;
        if (mc.n_expert <= 0 || mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: qwen4exp: the container describes %lld expert(s) with "
                            "top_k %lld\n", (long long)mc.n_expert, (long long)mc.top_k);
            return RAD_E_INVAL;
        }
        /* ---- the experts' stored format (block fp8, four-bit, four-bit rotated, or plain bf16),
         * read from one expert's encoding (MoeFP8::probe) before anything is sized by it: the
         * split between ranks, the rotated copy of the block input, the connection in front of
         * every routed block, and for bf16 every linear in the model. The encoding comes from the
         * container, checkpoint or recipe being applied, so a container is always served in the
         * format it holds.
         *
         * The probed expert is the first one this rank owns under expert parallelism: the answer
         * decides whether that split is used, and the expert is declared on this rank either way
         * (whole, or a slice under the parity split), so the block skips exactly the map the probe
         * made. */
        mc.probed_expert = (g.world > 1 && mc.n_expert % g.world == 0)
                               ? (int64_t)g.rank * (mc.n_expert / g.world) : 0;
        {
            MoeFP8::Src ps{};
            RAD_ARCH_TRY(moe_expert_sources(b, nm, "format",
                                            "model.language_model.layers.0.mlp.experts", &ps));
            RAD_ARCH_TRY(MoeFP8::probe(b, nm, 0, ps, &mc));
        }
        /* Plain experts mean a container converted without a quantising recipe, so every linear
         * is plain too: the blocks issue bf16 GEMMs over bf16 activations and nothing quantises
         * (Geom::w_bf16). Each linear refuses a quantised encoding then. The converse is allowed:
         * a quantised container may keep a linear plain (the QSA indexer's projection), which
         * LinearFP8 reads as bf16. */
        g.w_bf16 = mc.expert_bf16 != 0;
        /* The connections in front of the blocks then emit no codes, since nothing reads them. */
        m.hccfg.quant = g.w_bf16 ? 0 : 1;
        /* ============================== EVERY RANK HOLDS A SLICE OF EVERY EXPERT ==============================
         *
         * An expert is 640 columns, five 128-column blocks (the fp8 scale block, the 4-bit scale
         * group and the rotation width), so two ranks cannot halve one.
         *
         * The split is by parity: each parity's five blocks are cut contiguously over the ranks,
         * and the rank taking the extra block alternates with the expert's parity
         * (MoeFP8::Config::ff_lo). At two ranks, rank r's slice of expert e is the first three
         * blocks when e % 2 == r and the last two otherwise. The router is replicated, every rank
         * computes its slice of every routed expert, and the block's all-reduce sums the partials.
         *
         * At four ranks parity alone would give ranks 0 and 1 three blocks of every expert pair
         * and ranks 2 and 3 two, leaving 2 and 3 waiting in the all-reduce. So the extra block
         * rotates over four places by expert number mod 4 (four slice tables, `moe_gemm_q_mod4`),
         * and every rank computes five blocks of every four experts. A layer keeping two experts
         * plain bf16 has 510 quantised ones, and its first two classes are one expert longer.
         * Other remainders stay apart: the plain experts' extra tiles go to ranks 0 and 1
         * (MoeFP8::declare), the shared expert's extra block to rank 2 (shared_place).
         *
         * Not expert parallelism (rank r owning experts [r*E/W, (r+1)*E/W) whole): bytes per rank
         * are equal on average, but routing is random, and a verify step's forty routed slots leave
         * one rank about five experts ahead, costing about a millisecond per step at one sequence
         * in the all-reduce wait. With parity the ranks differ by a fifth of an expert per unit of
         * even/odd imbalance. Whole experts would also break the recipe's pairing of protected
         * experts, which keeps each rank's quantised-expert count even for the two parity tables.
         *
         * A calibration run does split whole experts per rank: the down projection's Hessian spans
         * an expert's whole K, and a slice would lack the cross terms. Whole experts give each rank
         * the full Hessian of every expert it holds, which the packer sums (rad_arch.h,
         * calib_dir). */
        mc.n_expert_all = mc.n_expert;
        /* bf16 experts are also split whole per rank: a slice of a bf16 expert is two row ranges
         * and a column range, which the file tier cannot read as one run, and a bf16 model is
         * served to calibrate and compare, not for decode speed. */
        if (g.world > 1 && (mc.calib || mc.expert_bf16)) {
            int s;
            int64_t per = 0;
            if ((s = divide_or_fail(mc.n_expert_all, g.world, "expert_count", &per)) < 0) return s;
            mc.n_ff_exp    = g.n_ff_all;
            mc.n_expert    = per;
            mc.expert_base = (int64_t)g.rank * per;
            rad_note(b, "%s: rank %d owns experts [%lld, %lld) of %lld whole.",
                     mc.calib ? "calibration (each expert's Hessian is over its whole width)"
                              : "bf16 experts", g.rank,
                     (long long)mc.expert_base, (long long)(mc.expert_base + mc.n_expert),
                     (long long)mc.n_expert_all);
        } else if (g.world > 1) {
            if (g.n_ff_all % RAD_FP8_BLOCK) {
                fprintf(stderr, "radiance: qwen4exp: an expert is %lld columns, which is not a whole "
                                "number of %d-column blocks and cannot be split across ranks.\n",
                        (long long)g.n_ff_all, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            const int64_t nb = g.n_ff_all / RAD_FP8_BLOCK;
            if (nb % g.world == 0) {
                mc.n_ff_exp = g.n_ff_all / g.world;
            } else if (g.world <= nb) {
                /* The uneven split is two slices per rank: each parity's blocks are cut
                 * contiguously across the ranks and the rank taking the odd block alternates with
                 * the parity, so slices are whole blocks and every block has exactly one owner.
                 * Widths are ceil(nb/W) and floor(nb/W). Every rank must get at least one block of
                 * each parity, since an empty slice still joins the all-reduce, so the world may
                 * not exceed nb. */
                for (int q = 0; q < 2; ++q) {
                    /* This rank's place in expert parity q's split: first for its own parity.
                     * Experts do not go remainder-first (rad_arch.h remainder_place): an expert
                     * block is a whole GEMM over every routed token, and a rank taking the extra
                     * block of every class becomes the slowest rank (at three ranks on
                     * Flash-Next, a 50/25/25 split cost 25% per step single-stream). Only the
                     * delta net's remainder, a head's worth of work, goes to the rank without
                     * attention. */
                    block_slice(nb, g.world, (g.rank - q + g.world) % g.world, &mc.ff_lo[q],
                                &mc.ff_hi[q]);
                }
                mc.n_ff_exp = std::max(mc.ff_hi[0] - mc.ff_lo[0], mc.ff_hi[1] - mc.ff_lo[1]);
                /* At four ranks, four classes by expert number mod 4 (MoeFP8::Config::ff_lo4),
                 * used by every layer instead: the same cut with the rank's place shifted by the
                 * class, so the extra block rotates over all four ranks. Every block still has one
                 * owner. */
                if (g.world == 4) {
                    for (int q = 0; q < 4; ++q) {
                        block_slice(nb, 4, (g.rank - q + 4) % 4, &mc.ff_lo4[q], &mc.ff_hi4[q]);
                        mc.n_ff_exp = std::max(mc.n_ff_exp, mc.ff_hi4[q] - mc.ff_lo4[q]);
                    }
                    rad_note(b, "routed experts at four classes: rank %d computes columns [%lld, "
                                "%lld) [%lld, %lld) [%lld, %lld) [%lld, %lld) of the experts "
                                "numbered 0, 1, 2 and 3 mod 4, %lld blocks per four experts like "
                                "every rank",
                             g.rank, (long long)mc.ff_lo4[0], (long long)mc.ff_hi4[0],
                             (long long)mc.ff_lo4[1], (long long)mc.ff_hi4[1],
                             (long long)mc.ff_lo4[2], (long long)mc.ff_hi4[2],
                             (long long)mc.ff_lo4[3], (long long)mc.ff_hi4[3], (long long)nb);
                } else {
                    rad_note(b, "routed experts: rank %d computes columns [%lld, %lld) of each "
                                "even expert and [%lld, %lld) of each odd one, of %lld; %lld "
                                "blocks of %d do not split evenly over %lld ranks, so the extra "
                                "block alternates with the expert's parity",
                             g.rank, (long long)mc.ff_lo[0], (long long)mc.ff_hi[0],
                             (long long)mc.ff_lo[1], (long long)mc.ff_hi[1],
                             (long long)g.n_ff_all, (long long)nb, RAD_FP8_BLOCK,
                             (long long)g.world);
                }
            } else {
                fprintf(stderr, "radiance: qwen4exp: an expert is %lld blocks of %d, which %d ranks "
                                "cannot each take a whole block of in both parities. Lower the rank "
                                "count, or serve whole experts (a bf16 container or a calibration "
                                "run does).\n",
                        (long long)nb, RAD_FP8_BLOCK, g.world);
                return RAD_E_INVAL;
            }
        }
    }

    /* ---- the n-gram embedding's layer and the indexer's geometry, each noted where it changes
     * what this build computes. */
    {
        const char* ple = rad_meta_gets(meta, "text_config.ple_layer_ids",
                          rad_meta_gets(meta, "ple_layer_ids", nullptr));
        if (ple && *ple) {
            /* One-indexed: `ple_layer_ids: [2]` is layer_idx 1 (the reference uses
             * `config.ple_layer_ids.index(layer_idx + 1)` and validates [1, num_hidden_layers]). */
            m.ple_layer = (int64_t)std::strtoll(ple, nullptr, 10) - 1;
            if (m.ple_layer < 0 || m.ple_layer >= g.n_layer) {
                /* Outside the declared layers: the n-gram embedding is not in this build, and the
                 * note says so rather than dropping it silently. */
                rad_note(b, "ple_layer_ids = %s is outside the %lld layers declared, so the n-gram "
                            "embedding is not in this build.", ple, (long long)g.n_layer);
                m.ple_layer = -1;
            }
        }
    }
    {
        const int64_t budget = meta_int2(meta, "qwen4exp.indexer_budget", "indexer_budget", 0);
        const int64_t ratio  = meta_int2(meta, "qwen4exp.indexer_compress_ratio",
                                         "indexer_compress_ratio", 0);
        m.qsacfg.budget   = budget;
        m.qsacfg.ratio    = ratio;
        m.qsacfg.heads    = meta_int2(meta, "qwen4exp.indexer_n_heads", "indexer_n_heads", 0);
        m.qsacfg.head_dim = meta_int2(meta, "qwen4exp.indexer_head_dim", "indexer_head_dim", 0);
        m.qsacfg.rot        = m.n_rot;
        m.qsacfg.topk       = (ratio > 0) ? budget / ratio : 0;
        m.qsacfg.max_blocks = (ratio > 0) ? (g.max_ctx + ratio - 1) / ratio : 0;
        m.qsacfg.max_work   = (ratio > 0) ? (g.max_tok + ratio - 1) / ratio + g.max_seqs : 0;
        /* The tail's ring (see QsaIndexer::Config::ring), sized for `max_spec`, the deepest
         * speculation any step can carry. */
        m.qsacfg.ring       = ratio + g.max_spec;
        if (budget > 0 && ratio > 0) {
            /* The exactness bound: with `pos + 1 <= budget + ratio - 1` tokens visible there are
             * at most budget/ratio complete blocks, the indexer's top-k keeps all of them, and the
             * tail is always attended, so QSA equals dense causal attention bit for bit. */
            const int64_t exact_to = budget + ratio - 1;
            /* The context bound is not capped: prefill selects per query exactly as decode does
             * (arch/common/rad_qsa.h, docs/QSA.md). There is deliberately no force-dense switch,
             * since above the bound dense attention is not the trained model. */
            if (g.max_ctx > exact_to)
                rad_note(b, "QSA: sparse past %lld tokens of context, %lld blocks of %lld per "
                            "query at prefill and decode; at or below that the engine issues the "
                            "bit-identical dense attention",
                         (long long)exact_to, (long long)(budget / ratio), (long long)ratio);
            else
                rad_note(b, "QSA: dense attention is exact to %lld tokens of context and "
                            "--max-model-len is %lld, so the indexer selects every visible block",
                         (long long)exact_to, (long long)g.max_ctx);
        }
    }

    rad_note(b, "gated residual: %lld streams of %lld (%lld wide), low-rank gate %lld; %lld "
                "connections plus the mixer",
             (long long)m.hccfg.hc, (long long)g.n_embd, (long long)HN,
             (long long)m.hccfg.lowrank, (long long)(2 * g.n_layer));

    /* ---- the KV groups: the same three as qwen35_fp8. */
    {
        RadKVGroupDecl fd{};
        fd.kind      = RAD_KV_FULL;
        fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
        fd.n_head_kv = g.n_head_kv;
        fd.head_dim  = g.head_dim;
        /* --kv-cache-dtype fp8_v4 / k6_v4 (RadBuildCtx::kv_quant): V as int4 in groups of 32 with
         * f16 scales, K E4M3 as above or int6 the same way (libr4d's gqa12 and four-rank gqa6
         * rows for these layers). A token's row is 25 or 22 * head_dim / 16 bytes instead of
         * 2 * head_dim, so the group is declared at half that and the core sizes pages from it;
         * only those kernels read the layout. The MTP head's layer shares this group. The engine
         * asks only for a layout probe() offered. */
        const int kvq = fd.dtype == RAD_F8E4M3 ? ctx->kv_quant : RAD_KVQ_NONE;
        const char* kvq_name = kvq == RAD_KVQ_V4 ? "fp8_v4" : kvq == RAD_KVQ_K6V4 ? "k6_v4"
                                                                                 : nullptr;
        if (kvq != RAD_KVQ_NONE && !kvq_name) {
            fprintf(stderr, "radiance: KV cache layout %d is not one this plugin knows\n", kvq);
            return RAD_E_UNSUPPORTED;
        }
        /* A rank's ratio: twelve at one and two ranks, six at four (each KV head replicated
         * across a rank pair). */
        if (kvq_name && !(g.head_dim == 256 && (g.gqa() == 12 || g.gqa() == 6))) {
            fprintf(stderr, "radiance: --kv-cache-dtype %s has libr4d rows at head_dim 256 and a "
                            "rank's gqa 12 or 6 only; this rank is head_dim %lld, gqa %lld\n",
                    kvq_name, (long long)g.head_dim, (long long)g.gqa());
            return RAD_E_UNSUPPORTED;
        }
        const int64_t row16 = kvq == RAD_KVQ_V4 ? 25 : kvq == RAD_KVQ_K6V4 ? 22 : 32;
        fd.head_dim = g.head_dim * row16 / 32;
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
        g.kv_dtype = kvq_name ? kvq_name : rad_dtype_name(fd.dtype);
        if (kvq_name)
            rad_note(b, "kv cache %s: %s K, int4 V, in groups of 32 with f16 scales, %lld bytes a "
                        "token and head instead of %lld", kvq_name,
                     kvq == RAD_KVQ_V4 ? "E4M3" : "int6", (long long)(g.head_dim * row16 / 16),
                     (long long)(2 * g.head_dim));

        RadKVGroupDecl sd{};
        sd.kind         = RAD_KV_LINEAR;
        sd.dtype        = RAD_F32;   /* f16 trades draft acceptance for a sliver of step time */
        sd.n_head_kv    = m.gcfg.n_head_v;
        sd.state_dim[0] = m.gcfg.head_v;
        sd.state_dim[1] = m.gcfg.head_k;
        m.kv_state = rad_decl_kv_group(b, nm.f("kv_gdn_state"), &sd);

        RadKVGroupDecl cd{};
        cd.kind       = RAD_KV_CONV;
        cd.dtype      = RAD_BF16;
        cd.n_head_kv  = 2 * m.gcfg.n_head_k + m.gcfg.n_head_v;
        cd.head_dim   = m.gcfg.head_k;
        cd.conv_width = m.gcfg.conv_width;
        m.kv_conv = rad_decl_kv_group(b, nm.f("kv_gdn_conv"), &cd);
        if (!m.kv_full || !m.kv_state || !m.kv_conv) return RAD_E_INVAL;

        /* ---- the QSA indexer's tail, which is state, not a buffer.
         *
         * It carries an unfinished block's raw index keys across steps, so it is indexed by
         * sequence and layer, which a step-local buffer cannot express (arch/common/rad_qsa.h,
         * Wire). As a linear group the engine keeps a slot stable for the sequence's life, zeroes
         * it on allocation, and saves and restores it with the linear state on prefix-cache
         * checkpoints, so a resume onto a half-built block reads defined data.
         *
         * `n_head_kv` 1 and `state_dim` {ring, head_dim}: `ring` raw keys, a few KiB per layer per
         * sequence. */
        if (m.qsacfg.heads > 0 && m.qsacfg.head_dim > 0 && m.qsacfg.ring > 0) {
            RadKVGroupDecl td{};
            td.kind         = RAD_KV_LINEAR;
            td.dtype        = g.act_dtype;
            td.n_head_kv    = 1;
            td.state_dim[0] = m.qsacfg.ring;
            td.state_dim[1] = m.qsacfg.head_dim;
            m.qsacfg.kv_tail = rad_decl_kv_group(b, nm.f("kv_qsa_tail"), &td);
            if (!m.qsacfg.kv_tail) return RAD_E_INVAL;
        }
    }

    /* ---- the trunk's stored format, read from the first delta-net layer's input projection as
     * the experts' is from one expert. The activation buffers below carry an int8 code pair only
     * when the trunk's linears are int8 (LinearFP8::i8), so it must be known first. The probe maps
     * the weight through the layer's own LinearFP8, which then does not map it again. */
    m.layers.resize((size_t)g.n_layer);
    {
        const char* pre0 = "model.language_model.layers";
        for (int64_t l = 0; l < g.n_layer; ++l) {
            if (is_full_attention(l, interval)) continue;
            RAD_ARCH_TRY(m.layers[(size_t)l].gdn.in.probe(
                b, nm, g, nm.f("blk.%lld.ssm_inz", (long long)l),
                { nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre0, (long long)l),
                  nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight", pre0, (long long)l) }));
            m.trunk_i8 = m.layers[(size_t)l].gdn.in.probed_i8;
            break;
        }
    }

    /* ---- buffers. */
    const int64_t q_dim    = g.q_dim();
    const int64_t kv_dim   = g.kv_dim();
    const int64_t v_dim    = m.gcfg.v_dim();
    const int64_t conv_dim = m.gcfg.conv_dim();
    const int64_t T        = g.max_tok;
    const int64_t ER       = m.moecfg.rows * m.moecfg.top_k;

    m.b_h   = decl_b(b, nm.f("h_wide"), g.act_dtype, {T, HN});
    m.a_x.x = decl_b(b, nm.f("x"),      g.act_dtype, {T, g.n_embd});
    if (ctx->shape_probe) {
        m.span_on = g_model[ctx->rank].span_on;
        m.pipe_on = g_model[ctx->rank].pipe_on;
    } else {
        m.span_on = ctx->max_mixed_tok > 0 && ctx->max_mixed_tok < g.max_tok &&
                    g.max_tok > m.moecfg.rows;
        /* Only where the blocks issue their own all-reduce at a step wider than a pass: at two
         * ranks the gated residual's fused write takes every step's (ar_exact_write_rows), so such
         * a step runs serially. */
        m.pipe_on = m.span_on && g.world > 1 && ar_exact_write_rows(g) < g.max_tok;
    }
    /* The pipelined step's all-reduces run on the second lane, in place over `x`, so it is the one
     * buffer that lane touches; it is live over the whole program anyway. */
    if (m.pipe_on) RAD_ARCH_TRY(rad_buf_concurrent(b, m.a_x.x));
    /* A transient although only its last entry changes per step: a persistent buffer sits in the
     * fixed region under every transient, and one more alignment unit there moves every
     * activation by 256 bytes, which on Flash-Next at four ranks slows the MoE gather by 73 % and
     * a prefill by 3 %. Every step that reads it rewrites it (pipe_bounds). */
    if (m.span_on)
        m.b_pipe_cu = decl_b(b, nm.f("pipe_cu"), RAD_I32, {4});
    /* Under a span every connection reads a step wider than a pass a pass at a time
     * (HyperConn::read_pieces, and the pipelined step's own reads), so the reads are declared no
     * wider than a pass and their scratch is sized for that band. */
    if (m.span_on) m.hccfg.read_rows = m.moecfg.rows;
    RAD_ARCH_TRY(m.a_x.declare_qs(b, nm, g, "x", g.n_embd, 0, m.trunk_i8));
    /* The connections write int8 codes themselves when every reader of their codes is an int8
     * linear: the trunk's are, and the routed experts then read the rotated copy or the bf16
     * plane. When experts read the plain E4M3 codes, the connections keep those and the int8
     * linears quantise for themselves. */
    m.hccfg.codes_i8 = (m.trunk_i8 && (m.moecfg.expert_rot || m.moecfg.expert_bf16)) ? 1 : 0;
    m.a_x.q8_fed     = m.hccfg.codes_i8 != 0;
    m.b_inj = decl_b(b, nm.f("hc_inj"), g.act_dtype, {T, m.hccfg.hc});

    m.b_qg     = decl_b(b, nm.f("attn_qg"),  g.act_dtype, {T, g.n_head, 2 * g.head_dim});
    m.b_k      = decl_b(b, nm.f("attn_k"),   g.act_dtype, {T, kv_dim});
    m.b_v      = decl_b(b, nm.f("attn_v"),   g.act_dtype, {T, kv_dim});
    m.b_q      = decl_b(b, nm.f("attn_q"),   g.act_dtype, {T, q_dim});
    m.a_attn.x = decl_b(b, nm.f("attn_out"), g.act_dtype, {T, q_dim});
    RAD_ARCH_TRY(m.a_attn.declare_qs(b, nm, g, "attn_out", q_dim, 0, m.trunk_i8));
    /* The gate epilogue in front of the int8 out projection writes its int8 codes itself. */
    m.a_attn.q8_fed = m.trunk_i8;
    /* The bf16 gate's own plane (AttnGatedFP8::Wire::gate); the fp8 path fuses it away. */
    if (g.w_bf16) m.b_gate = decl_b(b, nm.f("attn_gate"), g.act_dtype, {T, q_dim});
    /* The QSA indexer's planes, shared by every full-attention layer like `attn_out`. Declared
     * only when the checkpoint carries an indexer; otherwise the layer's declare() is a no-op. */
    if (m.qsacfg.heads > 0 && m.qsacfg.head_dim > 0) {
        m.qsaw.qk = decl_b(b, nm.f("qsa_qk"), g.act_dtype,
                           { g.max_tok, (m.qsacfg.heads + 1) * m.qsacfg.head_dim });
        m.qsaw.q  = decl_b(b, nm.f("qsa_q"),  g.act_dtype,
                           { g.max_tok, m.qsacfg.heads * m.qsacfg.head_dim });
        const int64_t hd = m.qsacfg.head_dim, rt = m.qsacfg.ratio, wk = m.qsacfg.max_work;
        m.qsaw.stage = decl_b(b, nm.f("qsa_stage"), g.act_dtype, { wk * rt, hd });
        m.qsaw.page  = decl_b(b, nm.f("qsa_page"),  RAD_I32, { wk });
        /* Three components a block under M-RoPE (arch/common/rad_qsa.h). */
        m.qsaw.bpos  = g.rope_mc ? decl_b(b, nm.f("qsa_bpos"), RAD_I32, { wk, 3 })
                                 : decl_b(b, nm.f("qsa_bpos"), RAD_I32, { wk });
        m.qsaw.nc    = decl_b(b, nm.f("qsa_nc"),    RAD_I32, { g.max_seqs });
        /* A pass tall under a span whose shared steps fit in a pass (RadBuildCtx::
         * max_mixed_tok): a wider step is one sequence and scores and selects in windows of a
         * pass (QsaIndexer::step, with the bounds in b_pipe_cu). The plane is the largest buffer
         * the span would otherwise grow, max_ctx / ratio floats a query row (512 MiB more at 4096
         * rows and 256K of context). */
        const int64_t qsa_rows = (m.span_on && ctx->max_mixed_tok <= m.moecfg.rows)
                                     ? m.moecfg.rows : g.max_tok;
        m.qsacfg.plane_rows = qsa_rows < g.max_tok ? qsa_rows : 0;
        /* One row per query, not per sequence (arch/common/rad_qsa.h). The score plane is the
         * only buffer whose size is the product of two deployment bounds, so an oversized one is
         * refused here, naming the two flags, rather than failing later as an allocation. */
        const int64_t sc_elems = qsa_rows * m.qsacfg.max_blocks;
        if (sc_elems > (int64_t)512 * 1024 * 1024) {
            fprintf(stderr,
                "radiance: qwen4exp: the QSA score plane would be %lld MiB -- %lld rows "
                "(--max-num-batched-tokens %lld) by --max-model-len %lld / compress ratio %lld. "
                "Lower either.\n",
                (long long)(sc_elems * 4 / (1024 * 1024)), (long long)qsa_rows,
                (long long)g.max_tok, (long long)g.max_ctx, (long long)m.qsacfg.ratio);
            return RAD_E_UNSUPPORTED;
        }
        m.qsaw.score = decl_b(b, nm.f("qsa_score"), RAD_F32,
                              { qsa_rows, m.qsacfg.max_blocks });
        m.qsaw.sel   = decl_b(b, nm.f("qsa_sel"),   RAD_I32,
                              { g.max_tok, m.qsacfg.topk + 1 });
        m.qsaw.nsel  = decl_b(b, nm.f("qsa_nsel"),  RAD_I32, { g.max_tok });
        m.qsaw.sequ  = decl_b(b, nm.f("qsa_sequ"),  RAD_I32, { g.max_tok });
        /* The shared-table read path, the prototype behind RADIANCE_QSA_TABLE=1
         * (arch/common/rad_qsa.h: qsa_table_on()). One union per tile of kQsaTile query rows and
         * the membership mask that turns it back into each row's own list, one buffer for the
         * whole model because every layer reads the same step's table.
         *
         * `u_max` is the plane's width and the two ops' `u_max` parameter; the union of `tile` rows
         * holds at most `tile * topk` entries and never more than the blocks there are. The mask is
         * the widest of the planes, `max_tok * u_max` int32 -- 64 MiB at a 2048-row step and 8 KiB
         * of union, which is the prototype's cost, and a bitmask (1 bit an entry) is the obvious
         * next step if it ever matters. */
        if (qsa_table_on()) {
            const int64_t tile = kQsaTile;
            const int64_t umax = std::min(tile * m.qsacfg.topk, m.qsacfg.max_blocks);
            const int64_t tiles = (g.max_tok + tile - 1) / tile;
            if (umax < 1) return RAD_E_INVAL;
            m.qsacfg.u_max = umax;
            m.qsaw.selb = decl_b(b, nm.f("qsa_selb"), RAD_I32,
                                 { g.max_tok, m.qsacfg.topk + 1 });
            m.qsaw.tab  = decl_b(b, nm.f("qsa_tab"),  RAD_I32, { tiles, umax });
            m.qsaw.tlen = decl_b(b, nm.f("qsa_tlen"), RAD_I32, { tiles });
            m.qsaw.mask = decl_b(b, nm.f("qsa_mask"), RAD_I32, { g.max_tok, umax });
            m.qsaw.roff = decl_b(b, nm.f("qsa_roff"), RAD_I32, { g.max_tok });
            if (!m.qsaw.selb || !m.qsaw.tab || !m.qsaw.tlen || !m.qsaw.mask || !m.qsaw.roff)
                return RAD_E_INVAL;
        }
        if (!m.qsaw.qk || !m.qsaw.q || !m.qsaw.stage || !m.qsaw.page ||
            !m.qsaw.bpos || !m.qsaw.nc || !m.qsaw.score || !m.qsaw.sel || !m.qsaw.nsel ||
            !m.qsaw.sequ)
            return RAD_E_INVAL;
    }

    m.a_in.x = decl_b(b, nm.f("gdn_in"), g.act_dtype, {T, conv_dim + v_dim});
    RAD_ARCH_TRY(m.a_in.declare_qs(b, nm, g, "gdn_in", conv_dim + v_dim));
    m.b_ab   = decl_b(b, nm.f("gdn_ab"),   g.act_dtype, {T, m.gcfg.ab_dim()});
    m.b_gq   = decl_b(b, nm.f("gdn_q"),    g.act_dtype, {T, m.gcfg.n_head_k, m.gcfg.head_k});
    m.b_gk   = decl_b(b, nm.f("gdn_k"),    g.act_dtype, {T, m.gcfg.n_head_k, m.gcfg.head_k});
    m.b_gv   = decl_b(b, nm.f("gdn_v"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    /* Both f32, not the activation dtype; `gdn_conv_prep` requires it and otherwise fails the
     * first step with "dtype this kernel does not serve". The log decay is a cumulative sum whose
     * differences the chunked scan exponentiates, and bf16's eight mantissa bits are not enough for
     * it or for beta. */
    m.b_gdec = decl_b(b, nm.f("gdn_gdec"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_beta = decl_b(b, nm.f("gdn_beta"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_kkt  = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16, {T, m.gcfg.n_head_v, m.gcfg.chunk});
    m.a_go.x = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim, 0, m.trunk_i8));
    /* The delta net's output epilogue also writes its int8 codes; its prefill quantiser is the
     * int8 one. */
    m.a_go.q8_fed = m.trunk_i8;

    /* The model's expert count, not this rank's: the router is replicated and scores every
     * expert. */
    m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {T, m.moecfg.n_expert_all});
    m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {T, m.moecfg.top_k});
    m.b_ew      = decl_b(b, nm.f("moe_ew"),     g.act_dtype, {T, m.moecfg.top_k});
    m.b_eoff    = decl_b(b, nm.f("moe_eoff"),   RAD_I32,     {m.moecfg.n_expert + 1});
    /* PERSIST, not transient. `moe_scatter` writes the per-expert histogram and no declared op
     * reads it (rad_route_report is a handoff, not an op), so as a transient liveness would let a
     * later buffer alias it, and a wide prefill chunk would corrupt the counts (even negative).
     * One int32 per expert. */
    m.b_ecnt    = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,     {m.moecfg.n_expert},
                         RAD_BUF_PERSIST);
    m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {ER});
    m.b_egu     = decl_b(b, nm.f("moe_egu"),    g.act_dtype, {ER, 2 * m.moecfg.n_ff_exp});
    m.a_eff.x   = decl_b(b, nm.f("moe_eff"),    g.act_dtype, {ER, m.moecfg.n_ff_exp});
    /* ER rows, not max_tok: an expert plane has one row per (token, slot). declare_qs's default
     * would make it `top_k` times too short and the gated quantiser would overrun it (see
     * ActFP8::declare_qs). */
    RAD_ARCH_TRY(m.a_eff.declare_qs(b, nm, g, "moe_eff", m.moecfg.n_ff_exp, ER));
    if (m.moecfg.expert_rot)
        RAD_ARCH_TRY(m.a_hr.declare_qs(b, nm, g, "moe_hr", g.n_embd));
    m.b_edn     = decl_b(b, nm.f("moe_edn"),    g.act_dtype, {ER, g.n_embd});
    m.b_emoe    = decl_b(b, nm.f("moe_out"),    g.act_dtype, {T, g.n_embd});
    if (m.moecfg.n_ff_shared > 0) {
        m.b_sgu   = decl_b(b, nm.f("moe_sgu"),   g.act_dtype, {T, 2 * m.moecfg.n_ff_shared});
        m.a_sff.x = decl_b(b, nm.f("moe_sff"),   g.act_dtype, {T, m.moecfg.n_ff_shared});
        RAD_ARCH_TRY(m.a_sff.declare_qs(b, nm, g, "moe_sff", m.moecfg.n_ff_shared, 0,
                                         m.trunk_i8));
        /* The shared expert's gated quantiser in front of its int8 down projection writes its codes
     * too. */
        m.a_sff.q8_fed = m.trunk_i8;
        m.b_sout  = decl_b(b, nm.f("moe_sout"),  g.act_dtype, {T, g.n_embd});
        m.b_sgate = decl_b(b, nm.f("moe_sgate"), g.act_dtype, {T, 1});
    }

    m.b_hout   = decl_b(b, nm.f("hout"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded as in qwen35_fp8. This checkpoint ships
     * `lm_head.weight` (tie_word_embeddings is false); the alternative source covers a smaller
     * family member that ties them.
     *
     * The head is declared over all its planes at the codes' dtype (bf16 as shipped, block fp8
     * where a recipe makes it so); the selector then picks a `logits_gemm` kernel whose layout
     * reads that encoding. The name maps come first, since they are how the head is queried. */
    RAD_ARCH_TRY(map_copy(b, nm.f("token_embd.weight"),
                          nm.ckpt("model.language_model.embed_tokens.weight")));
    RAD_ARCH_TRY(map_copy_alt(b, nm.f("output.weight"), nm.ckpt("lm_head.weight"),
                              nm.ckpt("model.language_model.embed_tokens.weight")));
    m.w_tok = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, {g.n_vocab, g.n_embd},
                     RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.w_lm_head = decl_w(b, nm.f("output.weight"),
                         weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16),
                         {g.n_vocab, g.n_embd}, RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    if (!m.w_tok || !m.w_lm_head) return RAD_E_INVAL;
    /* The rank's rows, stated explicitly: a vocabulary the world does not divide (248320 at three
     * ranks) is split by rad_arch.h's rule (ceil to a 128-row block, last rank short), not the
     * core's even share. Where it divides, the span equals the even share. */
    if (g.world > 1) {
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_tok,     g.vocab_off, g.vocab_off + g.n_vocab));
        RAD_ARCH_TRY(rad_weight_shard_span(b, m.w_lm_head, g.vocab_off, g.vocab_off + g.n_vocab));
    }

    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                       RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", g.n_embd),
                                  RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                  RAD_INT("vocab_offset", g.vocab_off)),
                       RAD_WEIGHTS(m.w_tok)),
                   {}, {m.a_x.x});
    /* The encoder's rows overwrite the lookup before its reduction (see MediaRows). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, T, g.n_embd, m.a_x.x));
    /* Always exact: this sums disjoint rows, a token's embedding coming from exactly one rank. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", g.n_embd, T * g.n_embd),
                                         RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                              RAD_NOWEIGHTS),
                          {m.a_x.x}, {m.a_x.x});

    /* The stream's initial value: hc identical copies of the embedding. */
    RAD_ARCH_TRY(m.enter.declare(b, g, m.hccfg.hc, m.a_x.x, m.b_h));

    /* The control vector's per-token switch, when this deployment found one (HcProject). All zeros
     * is the stock model, so a request that does not ask for the vector is unaffected. */
    const rad_buf proj_on = HcProject::declare_mask(b, nm.f("projection_on"), g, ctx->cvec);

    /* ---- the layers.
     *
     * Declaration order is the buffer planner's liveness: a transient lives over [first
     * definition, last use] in op indices, and transients with disjoint ranges may share bytes
     * (core/build/rad_bufplan.cpp). The buffers above are shared by every layer, so the order
     * below must match step()'s issue order (read, block, write, ...). Declaring a connection out
     * of order would alias buffers live in the same step, silently corrupting the output.
     */
    m.op_rope_cs = rw(b, decl_rotary(b, g, "rope_table",
                         RAD_PARAMS(RAD_RANGE("M", 1, qk_fuse_rows(g)),
                                    RAD_INT("rot", m.n_rot), RAD_F64("theta", g.theta),
                                    RAD_F64("scale", g.rope_scale), RAD_STR("dtype", "f32")),
                         RAD_NOWEIGHTS),
                     {}, {m.b_rope_cs = decl_b(b, nm.f("rope_cs"), RAD_F32,
                                               {rope_table_rows(g), m.n_rot})});

    AttnGatedFP8::Wire aw{};
    aw.x = m.b_h; aw.h = m.a_x; aw.qg = m.b_qg; aw.k = m.b_k; aw.v = m.b_v;
    aw.q = m.b_q; aw.attn = m.a_attn; aw.gate = m.b_gate;
    aw.cos_sin = m.op_rope_cs ? m.b_rope_cs : 0;
    /* The indexer reads the same rope table: its rotation uses `m.n_rot`, `g.theta` and
     * `g.rope_scale`, as the table does, which saves the trunk layers and the head's passes a norm
     * and rope each. Set here because the table is declared after `m.qsaw`'s buffers;
     * `tw.qsa = m.qsaw` below passes it on to the MTP head. */
    m.qsaw.cos_sin = aw.cos_sin;

    GdnFP8::Wire gw{};
    gw.x = m.b_h; gw.h = m.a_x; gw.in = m.a_in; gw.ab = m.b_ab;
    gw.q = m.b_gq; gw.k = m.b_gk; gw.v = m.b_gv;
    gw.gdec = m.b_gdec; gw.beta = m.b_beta; gw.kkt = m.b_kkt; gw.o = m.a_go;

    MoeFP8::Wire mw{};
    mw.x = m.b_h; mw.h = m.a_x; mw.hr = m.a_hr;
    mw.logits = m.b_rlogits; mw.ids = m.b_eids; mw.ew = m.b_ew;
    mw.sorted = m.b_sorted;  mw.eoff = m.b_eoff; mw.ecnt = m.b_ecnt;
    mw.egu = m.b_egu; mw.eff = m.a_eff; mw.edn = m.b_edn; mw.moe = m.b_emoe;
    mw.sgu = m.b_sgu; mw.sff = m.a_sff; mw.sout = m.b_sout; mw.sgate = m.b_sgate;

    HyperConn::Wire hw{};
    hw.h = m.b_h; hw.x = m.a_x; hw.inj = m.b_inj;

    m.layers.resize((size_t)g.n_layer);
    const char* pre = "model.language_model.layers";
    /* The pipelined step's all-reduces run beside the other piece's work, so each block declares
     * its collective a second time at `overlap` 1, for the rows that leave the CUs to that work
     * (the lane rows of libr4d/r4d_rows.cpp). */
    const int pipe_overlap = m.pipe_on ? 1 : 0;
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;
        lay.attn.ar_overlap = lay.gdn.ar_overlap = lay.mlp.ar_overlap = pipe_overlap;

        /* PLE comes first, before the attention connection reads the stream: the reference adds
         * it into `hidden_states` at the top of the decoder layer, so the following gated residual
         * normalises a stream that already includes the n-gram features. It is declared here, not
         * hoisted, so its buffers' liveness starts where they are used. */
        if (l == m.ple_layer) {
            PleLayer::Config pc{};
            pc.hc         = m.hccfg.hc;
            pc.embed_dim  = meta_int2(meta, "qwen4exp.ple_embed_dim", "ple_embed_dim", 0);
            pc.ngram      = meta_int2(meta, "qwen4exp.ngram_size", "ngram_size", 0);
            const int64_t per_ngram =
                meta_int2(meta, "qwen4exp.heads_per_ngram", "heads_per_ngram", 0);
            pc.heads      = (pc.ngram - 1) * per_ngram;
            pc.eos        = meta_int2(meta, "qwen4exp.eos_token_id", "eos_token_id", -1);
            pc.shards     = meta_int2(meta, "qwen4exp.split_ngram_parts", "split_ngram_parts", 0);
            pc.conv_width = meta_int2(meta, "qwen4exp.ple_conv_kernel_size",
                                      "ple_conv_kernel_size", 0);
            /* The dilation is the n-gram size (`conv_dilation = config.ngram_size` in the
             * reference), not a separate key and not 1. */
            pc.dilation   = pc.ngram;
            pc.rows       = ngram_table_rows(
                meta_int2(meta, "qwen4exp.ngram_vocab_size_base", "ngram_vocab_size_base", 0),
                pc.heads,
                meta_int2(meta, "qwen4exp.make_ngram_vocab_size_divisible_by",
                          "make_ngram_vocab_size_divisible_by", 1));

            PleLayer::Wire pw{};
            pw.h = m.b_h;

            PleLayer::Src ps{};
            ps.shard_prefix = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_embedding.shard_",
                                      pre, (long long)l);
            ps.shard_suffix = ".weight";
            ps.mult         = nm.ckpt("%s.%lld.ple.ple_embedding.layer_multipliers",
                                      pre, (long long)l);
            ps.vocab_sizes  = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_heads_vocab_sizes",
                                      pre, (long long)l);
            ps.offsets      = nm.ckpt("%s.%lld.ple.ple_embedding.ngram_heads_offsets",
                                      pre, (long long)l);
            ps.key_proj     = nm.ckpt("%s.%lld.ple.key_proj.weight",   pre, (long long)l);
            ps.value_proj   = nm.ckpt("%s.%lld.ple.value_proj.weight", pre, (long long)l);
            ps.norm_key     = nm.ckpt("%s.%lld.ple.norm_key.weight",   pre, (long long)l);
            ps.norm_query   = nm.ckpt("%s.%lld.ple.norm_query.weight", pre, (long long)l);
            ps.norm_conv    = nm.ckpt("%s.%lld.ple.norm_conv.weight",  pre, (long long)l);
            ps.conv1d       = nm.ckpt("%s.%lld.ple.conv1d.weight",     pre, (long long)l);
            RAD_ARCH_TRY(m.ple.declare(b, nm, g, pc, (int)l, pw, ps));
        }

        HyperConn::Src mixs{};
        mixs.norm     = nm.ckpt("%s.%lld.attn_hyper_connection.hc_norm.weight", pre, (long long)l);
        mixs.mix_down = nm.ckpt("%s.%lld.attn_hyper_connection.input_mix_weight_down.weight",
                                pre, (long long)l);
        mixs.mix_up   = nm.ckpt("%s.%lld.attn_hyper_connection.input_mix_weight_up.weight",
                                pre, (long long)l);
        mixs.inject_w = nm.ckpt("%s.%lld.attn_hyper_connection.block_inject_weight.weight",
                                pre, (long long)l);
        RAD_ARCH_TRY(lay.hc_mix.declare(b, nm, g, nm.f("blk.%lld.attn_hc", (long long)l),
                                        m.hccfg, hw, mixs));

        /* `Src::norm = nullptr` is the blocks' "input already normed and quantised" mode, and
         * `add_out = false` stops them adding into the stream; the connection's write half does
         * that with the four gains the read produced. `fold` and `ar_in` are off because both fuse
         * into a norm these blocks do not have. */
        if (lay.full) {
            AttnGatedFP8::Src as{};
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight", pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
            /* The KV page is the QSA compress block, so the indexer's selection is a paged block
             * table rather than a new attention kernel (docs/QSA.md); the attention rows must
             * declare the same 4 or nothing resolves. Page size only affects the block-table walk:
             * pages of 4 and 16 cost the same at long context and give identical text. */
            lay.attn.kv_block = 4;
            /* The selection must be set before the attention is declared: `attn_paged` declares
             * `sel` and `seqused` as reads, and a read declared after the op does not count
             * (rad_block_attn_gated_fp8.h). The indexer itself is declared after the attention
             * block, where the sparse attention reads its ops.
             *
             * Below the indexer's budget the selection is every visible block, so sparse and dense
             * paths must give identical text; any difference is a defect. */
            if (m.qsaw.qk) {
                lay.attn.qsa_sel      = m.qsaw.sel;
                lay.attn.qsa_sequ     = m.qsaw.sequ;
                lay.attn.qsa_topk     = m.qsacfg.topk;
                /* The shared table, when it was declared: the two ops are bound but only the gate
                 * and a library that declares them decide whether a step reads it
                 * (rad_block_attn_gated_fp8.h). */
                lay.attn.qsa_nsel     = m.qsaw.nsel;
                lay.attn.qsa_selb     = m.qsaw.selb;
                lay.attn.qsa_tab      = m.qsaw.tab;
                lay.attn.qsa_tlen     = m.qsaw.tlen;
                lay.attn.qsa_mask     = m.qsaw.mask;
                lay.attn.qsa_roff     = m.qsaw.roff;
                lay.attn.qsa_table    = m.qsaw.selb != 0;
                lay.attn.qsa_umax     = m.qsacfg.u_max;
                lay.attn.qsa_tile     = kQsaTile;
                lay.attn.qsa_ratio    = m.qsacfg.ratio;
                lay.attn.qsa_blocks   = m.qsacfg.max_blocks;
                /* The exactness bound, below which prefill issues the cheaper dense attention.
                 * Decode ignores it and always takes the sparse path, keeping that path exercised
                 * on every step. */
                lay.attn.qsa_exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            /* An attention-zero rank keeps no attention cache: its block is a fill of zeros
             * (rad_block_attn_gated_fp8.h), and the indexer reads only the shared block table. The
             * call is idempotent. */
            if (g.attn_zero && m.kv_full) RAD_ARCH_TRY(rad_kv_group_zero(b, m.kv_full));
            lay.attn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          false, false, kArNone, lay.hc_mix.op_ar_write != 0));
            /* The indexer, after the attention block so its ops sit where the sparse attention
             * reads them. It reads the same normed fp8 input as the projections
             * (arch/common/rad_qsa.h, docs/QSA.md). */
            if (m.qsaw.qk) {
                QsaIndexer::Config qc = m.qsacfg;
                qc.mode    = lay.attn.mode;
                qc.qk_proj = nm.ckpt("%s.%lld.self_attn.indexer.index_qk_proj.weight",
                                     pre, (long long)l);
                qc.q_norm  = nm.ckpt("%s.%lld.self_attn.indexer.q_layernorm.weight",
                                     pre, (long long)l);
                qc.k_norm  = nm.ckpt("%s.%lld.self_attn.indexer.k_layernorm.weight",
                                     pre, (long long)l);
                qc.kv_attn  = m.kv_full;
                qc.emit_sel = lay.attn.qsa_sel != 0;
                RAD_ARCH_TRY(lay.qsa.declare(b, nm, g, (int)l, qc, m.qsaw, aw.h,
                                             &m.kv_qsa_bkey));
            }
        } else {
            GdnFP8::Src gs{};
            gs.in_qkv   = nm.ckpt("%s.%lld.linear_attn.in_proj_qkv.weight", pre, (long long)l);
            gs.in_a     = nm.ckpt("%s.%lld.linear_attn.in_proj_a.weight",   pre, (long long)l);
            gs.in_b     = nm.ckpt("%s.%lld.linear_attn.in_proj_b.weight",   pre, (long long)l);
            gs.in_z     = nm.ckpt("%s.%lld.linear_attn.in_proj_z.weight",   pre, (long long)l);
            gs.conv1d   = nm.ckpt("%s.%lld.linear_attn.conv1d.weight",      pre, (long long)l);
            gs.a_log    = nm.ckpt("%s.%lld.linear_attn.A_log",              pre, (long long)l);
            gs.dt_bias  = nm.ckpt("%s.%lld.linear_attn.dt_bias",            pre, (long long)l);
            gs.out_norm = nm.ckpt("%s.%lld.linear_attn.norm.weight",        pre, (long long)l);
            gs.out      = nm.ckpt("%s.%lld.linear_attn.out_proj.weight",    pre, (long long)l);
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_state));
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_conv));
            lay.gdn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.gdn.declare(b, nm, g, m.gcfg, (int)l, m.kv_state, m.kv_conv, gw, gs,
                                         false, false, kArNone, lay.hc_mix.op_ar_write != 0));
        }

        HyperConn::Src ffns{};
        ffns.norm     = nm.ckpt("%s.%lld.mlp_hyper_connection.hc_norm.weight", pre, (long long)l);
        ffns.mix_down = nm.ckpt("%s.%lld.mlp_hyper_connection.input_mix_weight_down.weight",
                                pre, (long long)l);
        ffns.mix_up   = nm.ckpt("%s.%lld.mlp_hyper_connection.input_mix_weight_up.weight",
                                pre, (long long)l);
        ffns.inject_w = nm.ckpt("%s.%lld.mlp_hyper_connection.block_inject_weight.weight",
                                pre, (long long)l);
        /* The rotated expert plane's input codes come out of this read, beside the plain ones
         * the shared arm takes, so the routed arm declares no quantiser of its own. */
        HyperConn::Config fcfg = m.hccfg;
        HyperConn::Wire   fw   = hw;
        if (m.moecfg.expert_rot) { fcfg.rotate = 1; fw.rq = m.a_hr.q; fw.rs = m.a_hr.s; }
        RAD_ARCH_TRY(lay.hc_ffn.declare(b, nm, g, nm.f("blk.%lld.ffn_hc", (long long)l),
                                        fcfg, fw, ffns));

        MoeFP8::Src ms{};
        ms.cfg              = m.moecfg;
        ms.router           = nm.ckpt("%s.%lld.mlp.gate.weight", pre, (long long)l);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "layer",
                                        nm.ckpt("%s.%lld.mlp.experts", pre, (long long)l), &ms));
        if (m.moecfg.n_ff_shared > 0) {
            ms.shared      = nm.ckpt("%s.%lld.mlp.shared_expert", pre, (long long)l);
            ms.shared_gate = nm.ckpt("%s.%lld.mlp.shared_expert_gate.weight", pre, (long long)l);
        }
        lay.mlp.ar_out_take = lay.hc_ffn.ar_take();
        lay.mlp.hr_in = fcfg.rotate != 0;
        RAD_ARCH_TRY(lay.mlp.declare(b, nm, g, (int)l, mw, ms, false, false, kArNone,
                                     lay.hc_ffn.op_ar_write != 0));
        /* The write behind the block takes its gather where it can, declared after the block so
         * the buffers it reads are the block's outputs in declaration order. */
        {
            HyperConn::GatherSrc gs{};
            gs.ye     = lay.mlp.w.edn;
            gs.ew     = lay.mlp.w.ew;
            gs.sorted = lay.mlp.w.sorted;
            if (lay.mlp.gfold) { gs.sh = lay.mlp.w.sout; gs.sg = lay.mlp.w.sgate; }
            gs.top_k  = lay.mlp.c.top_k;
            gs.rows   = lay.mlp.c.rows;
            RAD_ARCH_TRY(lay.hc_ffn.declare_gather(b, gs));
            lay.mlp.gather_out_rows  = lay.hc_ffn.gather_rows;
            lay.mlp.gather_out_rows6 = lay.hc_ffn.gather_rows6;
        }

        /* The control vector's projection for this layer, if the vector steers it. */
        RAD_ARCH_TRY(lay.proj.declare(b, nm.f("blk.%lld.proj", (long long) l), g, l, ctx->cvec,
                                      proj_on, m.b_h, m.hccfg.hc));
    }

    /* Each routed layer predicts the next one's experts at its router input (MoeFP8::xp_next). */
    for (size_t l = 0; l + 1 < m.layers.size(); ++l) m.layers[l].mlp.xp_next = &m.layers[l + 1].mlp;

    /* ---- the 97th connection: no write half, no fp8 copy. It produces the bf16 hidden state the
     * lm_head reads; its `hc_norm` takes the place of a final norm (the checkpoint has no
     * `model.language_model.norm.weight`). */
    {
        HyperConn::Config mc = m.hccfg;
        mc.inject = 0;
        mc.quant  = 0;
        HyperConn::Wire mixw{};
        mixw.h = m.b_h; mixw.x.x = m.a_x.x;       /* q/s deliberately absent */
        HyperConn::Src mixs{};
        mixs.norm     = nm.ckpt("model.language_model.hyper_connection_mixer.hc_norm.weight");
        mixs.mix_down = nm.ckpt("model.language_model.hyper_connection_mixer."
                                "input_mix_weight_down.weight");
        mixs.mix_up   = nm.ckpt("model.language_model.hyper_connection_mixer."
                                "input_mix_weight_up.weight");
        RAD_ARCH_TRY(m.mixer.declare(b, nm, g, nm.f("output_hc"), mc, mixw, mixs));
    }

    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {m.a_x.x}, {m.b_hout});
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                   RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(m.w_lm_head)),
                    {m.b_hout}, {m.b_logits});
    if (!m.op_embed || !m.op_gather || !m.op_logits) return RAD_E_INVAL;

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h).
     *
     * Declared after op_logits; do not move it. Declaration order is the planner's liveness, and a
     * drafter declared before the trunk's tail ops would share a trunk buffer with one of the
     * head's, giving correct output with zero acceptance.
     *
     * Declared only when the checkpoint carries a head and the deployment speculates; at
     * max_spec 0 the weights stay off the card (the container still carries them, since
     * rad-convert declares at a non-zero max_spec). */
    const int64_t nextn = meta_int2(meta, "qwen4exp.mtp_num_hidden_layers",
                                    "mtp_num_hidden_layers", 0);
    if (nextn > 0 && g.max_spec > 0) {
        const char* mpre = "mtp";
        /* Sized at max_tok, not max_logit_rows: the history pass that fills the head's own K and V
         * has one row per token over a whole prefill chunk. */
        m.b_mtp_hin  = decl_b(b, nm.f("mtp_hin"),  g.act_dtype, {T, HN});
        m.b_mtp_hw   = decl_b(b, nm.f("mtp_hw"),   g.act_dtype, {T, HN});
        m.b_mtp_e    = decl_b(b, nm.f("mtp_e"),    g.act_dtype, {T, g.n_embd});
        m.b_mtp_hout = decl_b(b, nm.f("mtp_hout"), g.act_dtype, {g.max_logit_rows, g.n_embd});
        /* The 2-bit draft head, when the model holds one: asked once here, since the answer
         * decides the head's buffers and the block reads it back from them. */
        if (mtp_draft_head_q2(b, MtpHcBlock::head_name(nm), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,  {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32, {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32, {g.n_embd / 128, dmpad});
            /* This rank's rows, not the model's: the 2-bit head is vocab-parallel and the merge
             * reassembles it (rad_arch.h). */
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype,
                                  {g.max_logit_rows, vocab_head_rows(g)});
            if (vocab_head_sharded(g)) {
                m.b_mtp_dpair = decl_b(b, nm.f("mtp_dpair"), RAD_F32,
                                       {g.max_logit_rows, kDraftTopR, 2});
                m.b_mtp_dgath = decl_b(b, nm.f("mtp_dgath"), RAD_F32,
                                       {g.max_logit_rows, (int64_t)g.world * kDraftTopR, 2});
            }
            /* The proposal: a draft round's token comes from row_topk, never the sampler.
             * kDraftTopR columns, of which a draft reads column 0; the rest are the coarse head's
             * other candidates, which `logit_rerank` scores first, and the width also gives the
             * vocab-parallel all-gather a 16-byte row (rad_block_mtp_hc.h, kDraftTopR). */
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32, {g.max_logit_rows, kDraftTopR});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32, {g.max_logit_rows, kDraftTopR});

            /* RADIANCE_MTP_COUPLED. The coupling needs the 2-bit candidate head's reranked top-8,
             * because Q is that shortlist's own law; a plain bf16 head leaves one candidate and
             * there is nothing to couple over. The drafter SHAPE (a linear serial chain, never a
             * block drafter) is checked by the engine, which is where the declaration is in hand:
             * the switches that once named tree drafters do not exist in this tree, and testing
             * their environment variables here enforced nothing. */
            if (!rad::mtp_coupled_config_valid()) {
                rad_note(b, "RADIANCE_MTP_COUPLED accepts 0|1 (or unset); any other value is "
                            "refused, not treated as off");
                return RAD_E_INVAL;
            }
            if (rad::mtp_coupled_enabled()) {
                if (!m.b_mtp_didx || g.max_spec < 1 || g.max_spec > RAD_MTP_COUPLE_MAX_ROUNDS) {
                    rad_note(b, "RADIANCE_MTP_COUPLED requires the 2-bit candidate head's reranked "
                                "top8 and max_spec 1..%d",
                             RAD_MTP_COUPLE_MAX_ROUNDS);
                    return RAD_E_UNSUPPORTED;
                }
                m.b_mtp_couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                              {g.max_seqs, RAD_MTP_COUPLE_DRAW_WORDS},
                                              RAD_BUF_PERSIST);
                m.b_mtp_couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                              {g.max_spec, g.max_seqs,
                                               RAD_MTP_COUPLE_STATE_WORDS},
                                              RAD_BUF_PERSIST);
                if (!m.b_mtp_couple_draws || !m.b_mtp_couple_state) return RAD_E_INVAL;
                rad_note(b, "MTP coupled: the head samples x~Q from its reranked top8 each round "
                            "(FP64), and the sampler couples the target's own draw to it; %lld "
                            "round(s), fixed-seed tokens are NOT preserved",
                         (long long)g.max_spec);
            }
        }

        MtpHcBlock::Wire tw{};
        tw.src  = m.b_h;        tw.hin = m.b_mtp_hin;  tw.hw = m.b_mtp_hw;
        tw.e    = m.b_mtp_e;    tw.inj = m.b_inj;      tw.hout = m.b_mtp_hout;
        tw.logits = m.b_logits; tw.x = m.a_x;
        tw.dq   = m.b_mtp_dq;   tw.ds = m.b_mtp_ds;    tw.dsum = m.b_mtp_dsum;
        tw.dlog = m.b_mtp_dlog; tw.didx = m.b_mtp_didx; tw.dval = m.b_mtp_dval;
        tw.dpair = m.b_mtp_dpair; tw.dgath = m.b_mtp_dgath;
        tw.couple_draws = m.b_mtp_couple_draws; tw.couple_state = m.b_mtp_couple_state;
        /* The head reuses the trunk's working buffers, all dead by the time a draft round runs (a
         * second set would be 150 MiB). Only the wide stream and the carry are its own, since
         * they must survive the round that produced them. */
        tw.attn = aw; tw.attn.x = m.b_mtp_hw;
        tw.mlp  = mw; tw.mlp.x  = m.b_mtp_hw;
        /* The QSA wire is within-step scratch and a draft round is its own step, so the head
         * shares all of it. The QSA tail is per-layer state (`kv_qsa_tail`), not in the wire, and
         * declare() binds the head's layer to it like any other. */
        tw.qsa  = m.qsaw;

        MtpHcBlock::Src ts{};
        ts.fc_hidden          = nm.ckpt("%s.fc_hidden.weight", mpre);
        ts.fc_embedding       = nm.ckpt("%s.fc_embedding.weight", mpre);
        ts.pre_norm_hidden    = nm.ckpt("%s.pre_fc_norm_hidden.weight", mpre);
        ts.pre_norm_embedding = nm.ckpt("%s.pre_fc_norm_embedding.weight", mpre);

        ts.hc_attn.norm     = nm.ckpt("%s.layers.0.attn_hyper_connection.hc_norm.weight", mpre);
        ts.hc_attn.mix_down = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "input_mix_weight_down.weight", mpre);
        ts.hc_attn.mix_up   = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "input_mix_weight_up.weight", mpre);
        ts.hc_attn.inject_w = nm.ckpt("%s.layers.0.attn_hyper_connection."
                                      "block_inject_weight.weight", mpre);
        ts.hc_mlp.norm      = nm.ckpt("%s.layers.0.mlp_hyper_connection.hc_norm.weight", mpre);
        ts.hc_mlp.mix_down  = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "input_mix_weight_down.weight", mpre);
        ts.hc_mlp.mix_up    = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "input_mix_weight_up.weight", mpre);
        ts.hc_mlp.inject_w  = nm.ckpt("%s.layers.0.mlp_hyper_connection."
                                      "block_inject_weight.weight", mpre);
        /* The head's own mixer has no `block_inject_weight`, like the trunk's 97th connection:
         * nothing follows it to write back into. */
        ts.hc_mix.norm      = nm.ckpt("%s.hyper_connection_mixer.hc_norm.weight", mpre);
        ts.hc_mix.mix_down  = nm.ckpt("%s.hyper_connection_mixer.input_mix_weight_down.weight",
                                      mpre);
        ts.hc_mix.mix_up    = nm.ckpt("%s.hyper_connection_mixer.input_mix_weight_up.weight", mpre);

        ts.attn.qg     = nm.ckpt("%s.layers.0.self_attn.q_proj.weight", mpre);
        ts.attn.k      = nm.ckpt("%s.layers.0.self_attn.k_proj.weight", mpre);
        ts.attn.v      = nm.ckpt("%s.layers.0.self_attn.v_proj.weight", mpre);
        ts.attn.q_norm = nm.ckpt("%s.layers.0.self_attn.q_norm.weight", mpre);
        ts.attn.k_norm = nm.ckpt("%s.layers.0.self_attn.k_norm.weight", mpre);
        ts.attn.o      = nm.ckpt("%s.layers.0.self_attn.o_proj.weight", mpre);

        ts.mlp.cfg             = m.moecfg;
        ts.mlp.router          = nm.ckpt("%s.layers.0.mlp.gate.weight", mpre);
        RAD_ARCH_TRY(moe_expert_sources(b, nm, "mtp", nm.ckpt("%s.layers.0.mlp.experts", mpre),
                                        &ts.mlp));
        if (m.moecfg.n_ff_shared > 0) {
            ts.mlp.shared      = nm.ckpt("%s.layers.0.mlp.shared_expert", mpre);
            ts.mlp.shared_gate = nm.ckpt("%s.layers.0.mlp.shared_expert_gate.weight", mpre);
        }

        /* Set before declare(), as for the trunk: `attn_paged` declares `sel` and `seqused` as
         * reads, which must precede the op. */
        m.mtp.attn.kv_block = 4;
        if (m.qsaw.qk) {
            m.mtp.attn.qsa_sel      = m.qsaw.sel;
            m.mtp.attn.qsa_sequ     = m.qsaw.sequ;
            m.mtp.attn.qsa_topk     = m.qsacfg.topk;
            m.mtp.attn.qsa_nsel     = m.qsaw.nsel;
            m.mtp.attn.qsa_selb     = m.qsaw.selb;
            m.mtp.attn.qsa_tab      = m.qsaw.tab;
            m.mtp.attn.qsa_tlen     = m.qsaw.tlen;
            m.mtp.attn.qsa_mask     = m.qsaw.mask;
            m.mtp.attn.qsa_roff     = m.qsaw.roff;
            m.mtp.attn.qsa_table    = m.qsaw.selb != 0;
            m.mtp.attn.qsa_umax     = m.qsacfg.u_max;
            m.mtp.attn.qsa_tile     = kQsaTile;
            m.mtp.attn.qsa_ratio    = m.qsacfg.ratio;
            m.mtp.attn.qsa_blocks   = m.qsacfg.max_blocks;
            m.mtp.attn.qsa_exact_to = m.qsacfg.budget + m.qsacfg.ratio - 1;
            ts.qsa         = m.qsacfg;
            ts.qsa.mode    = m.mtp.attn.mode;
            ts.qsa.qk_proj = nm.ckpt("%s.layers.0.self_attn.indexer.index_qk_proj.weight", mpre);
            ts.qsa.q_norm  = nm.ckpt("%s.layers.0.self_attn.indexer.q_layernorm.weight", mpre);
            ts.qsa.k_norm  = nm.ckpt("%s.layers.0.self_attn.indexer.k_layernorm.weight", mpre);
            ts.qsa.kv_attn  = m.kv_full;
            ts.qsa.emit_sel = m.mtp.attn.qsa_sel != 0;
        }

        m.mtp.media = m.media;
        RAD_ARCH_TRY(m.mtp.declare(b, nm, g, (int)g.n_layer, m.kv_full, &m.kv_qsa_bkey, m.n_rot,
                                   m.w_tok, m.w_lm_head, m.hccfg, g.max_spec, tw, ts));
        /* A head pass refills the rope table ahead of the head's ops (step), while the carry
         * `mtp_hin` still holds the rows the previous pass left. The refill is the trunk's
         * `rope_table` op, so the table's planned life would end at its last declared reader,
         * which is in the trunk when no head op reads the table (no fused prologue), and the
         * planner could then put the carry in the table's bytes. Declared as written by the
         * head's first op, the table is live where the carry is. */
        if (m.op_rope_cs && rad_op_writes(b, m.mtp.op_gather, &m.b_rope_cs, 1) < 0)
            return RAD_E_INVAL;
        m.have_mtp = true;

        /* Serial drafter: `max_spec` dependent passes, each embedding the last one's proposal.
         * The 2-bit head writes its own argmax into `mtp_didx`; without it `mtp_didx` is 0 and the
         * bf16 head goes through the ordinary sampler chain. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec;
        dd.proposal       = m.b_mtp_didx;
        dd.proposal_pitch = kDraftTopR;   /* the pick is column 0; see kDraftTopR */
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        rad_note(b, "MTP head: one gated-attention layer with 512 routed experts over the wide "
                    "stream, %d serial draft round(s)", ctx->max_spec);
    } else if (nextn > 0) {
        rad_note(b, "the checkpoint's `mtp.` head is not loaded: this deployment does not "
                    "speculate");
    }

    /* The vision tower, last: an encoder pass issues nothing else, so its activations share the
     * arena with every buffer above. Its weight groups start past every text layer and the MTP
     * head. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}

/* ============================== A prefill step wider than one MoE pass ==============================
 *
 * Every layer of a prefill chunk makes two all-reduces, the mixer's and the MoE's, each in a hard
 * chain: mixer -> all-reduce -> write -> read -> MoE -> all-reduce -> write -> read. At four ranks
 * a chunk's 97 tiered-int8 two-shots are about a quarter of its time, with nothing beside them on
 * one stream.
 *
 * Under --prefill-span the scheduler hands a prefill alone in its step several checkpoint
 * intervals, each piece the interval-sized chunk a span of 1 makes. Everything a piece can run by
 * itself runs a piece at a time, and every all-reduce goes out on the second lane, beside the
 * other piece's work on the first:
 *
 *   the mixer       the delta net runs whole a piece at a time (projection, convolution, solve and
 *                   scan over the piece's bounds, the state handed on through the cache as between
 *                   two chunks), and the attention runs its gated sparse path a piece at a time;
 *                   each hands its first piece's all-reduce off behind that piece's out projection
 *                   (ArPieces), so it runs beside the second piece's mixer;
 *   the MoE         a pass a piece: the mixer's all-reduce of piece 1 runs beside piece 0's write,
 *                   read and pass, the MoE's of piece 0 beside piece 1's;
 *   the last one    the MoE's all-reduce of piece 1 runs beside piece 0's write and the next
 *                   layer's read.
 *
 * The indexer's selection and the embedding run over the whole step.
 *
 * A piece computes the bytes the narrower chunk does: every all-reduce message has a piece's rows,
 * so the tiered-int8 shards and their owners are the ones a span of 1 sends; every connection read
 * is a piece, so its down product splits K the same way (HyperConn::read_pieces); every MoE pass
 * and delta-net piece is a chunk's, and attention rows are their own. The first interval of a
 * prompt is never spanned (Scheduler::plan_chunk), so the dense attention below QSA's exactness
 * bound is taken where it was.
 *
 * The second lane's requirements (Ctx::lane1_) hold: every collective of the step is on it,
 * including the embedding's; none carries an arrival counter; and the one buffer it touches is out
 * of the sharing pool (declare). The collectives there are declared at `overlap` 1, whose rows
 * leave the CUs to the work beside them. */
static bool pipelined(const Model& m, int64_t T) {
    const int64_t R = m.moecfg.rows;
    if (!m.pipe_on || R <= 0 || T <= R) return false;
    if (m.moecfg.calib || dbg_resid_on() || !m.op_embed_ar) return false;
    for (const Layer& l : m.layers) {
        if (!(l.full ? l.attn.issues_ar(T) : l.gdn.issues_ar(T))) return false;
        if (!l.mlp.issues_ar(T, R)) return false;
    }
    return true;
}

/* The mixer's reduction of a piece the block has finished (ArPieces), on the second lane behind
 * the out projection that wrote it. */
struct MixAr {
    RadCtx*      c = nullptr;
    const Layer* l = nullptr;
};
static void mix_ar_piece(void* u, int64_t r0, int64_t n) {
    const MixAr& s = *(const MixAr*)u;
    rad_lane(s.c, 1);
    rad_lane_join(s.c, 0, 1);
    if (s.l->full) s.l->attn.issue_ar(s.c, r0, n);
    else           s.l->gdn.issue_ar(s.c, r0, n);
    rad_lane(s.c, 0);
}

/* The bounds a step wider than a pass is cut by, on the card: {0, pass, T}. The first two are the
 * window of the indexer's score plane (QsaIndexer::step, Config::plane_rows) and the delta net's
 * first piece (ArPieces::cu), copied from the pinned host copy; T is copied on the card from the
 * step's cu_seqlens. Both are issued on the step's stream ahead of every layer by each step of one
 * sequence wider than a pass, since the buffer is a transient (declare). Null where no step can be
 * wider than a pass (no b_pipe_cu) or the pinned copy could not be allocated. */
static const int32_t* pipe_bounds(RadCtx* c, const RadBatch* batch, Model& m) {
    if (!m.b_pipe_cu) return nullptr;
    if (!m.pipe_cu_tried) {
        m.pipe_cu_tried = true;
        m.pipe_cu_h = (int32_t*)rad_dev_alloc(2 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        if (m.pipe_cu_h) {
            m.pipe_cu_h[0] = 0;
            m.pipe_cu_h[1] = (int32_t)m.moecfg.rows;
        } else if (m.g.rank == 0) {
            fprintf(stderr, "radiance: qwen4exp: no pinned host memory for the window bounds; a "
                            "step wider than a pass cannot run\n");
        }
    }
    if (!m.pipe_cu_h) return nullptr;
    int32_t* d = (int32_t*)rad_buf_ptr(c, m.b_pipe_cu);
    if (batch->n_tok > m.moecfg.rows && batch->n_seq == 1) {
        const RadStream s = rad_stream(c);
        if (rad_memcpy_async(d, m.pipe_cu_h, 2 * sizeof(int32_t), s) != RAD_OK ||
            rad_memcpy_async(d + 2, batch->cu_seqlens + 1, sizeof(int32_t), s) != RAD_OK)
            return nullptr;
    }
    return d;
}

static void step_pipelined(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok, R = m.moecfg.rows;
    const int64_t K = (T + R - 1) / R;
    auto r0 = [&](int64_t p) { return p * R; };
    auto rn = [&](int64_t p) { return T - p * R < R ? T - p * R : R; };
    /* A collective on the second lane, ordered behind lane 0 where it reads what lane 0 just
     * wrote (a join to a lane not yet built is a no-op). */
    auto on1 = [&](bool behind0, auto&& issue) {
        rad_lane(c, 1);
        if (behind0) rad_lane_join(c, 0, 1);
        issue();
        rad_lane(c, 0);
    };
    auto wait1 = [&] { rad_lane_join(c, 1, 0); };
    if (!m.pipe_noted) {
        m.pipe_noted = true;
        if (g.rank == 0)
            fprintf(stderr, "radiance: qwen4exp: prefill steps wider than a MoE pass are pipelined: "
                            "%lld rows in %lld pieces of %lld, all-reduces on the second lane\n",
                    (long long)T, (long long)K, (long long)R);
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    on1(true, [&] { RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE); });
    wait1();
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);
    if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    /* The delta net runs a piece at a time over the bounds (ArPieces::cu) in a step of one
     * prefill sequence in two pieces. */
    const int32_t* gcu = pcu && K == 2 && batch->n_seq == 1 && batch->phase == RAD_PHASE_PREFILL
                       ? pcu : nullptr;

    const int64_t n = (int64_t)m.layers.size();
    if (n > 0) {
        if (m.ple_layer == 0) m.ple.step(c, batch);
        for (int64_t p = 0; p < K; ++p) m.layers[0].hc_mix.read(c, T, r0(p), rn(p));
    }
    for (int64_t li = 0; li < n; ++li) {
        const Layer& l = m.layers[(size_t)li];
        /* The mixer finishes its output a piece at a time where its last stages are per row and
         * hands the first pieces' reductions to the second lane as it goes (ArPieces), so the
         * first piece's all-reduce runs beside the next piece's attention or out projection. */
        MixAr ms{c, &l};
        ArPieces pc{};
        pc.rows = R; pc.done = mix_ar_piece; pc.u = &ms; pc.cu = gcu;
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch, false, &pc); }
        else        l.gdn.step(c, batch, false, &pc);
        auto ar_mix = [&](int64_t p) {
            if (l.full) l.attn.issue_ar(c, r0(p), rn(p));
            else        l.gdn.issue_ar(c, r0(p), rn(p));
        };
        /* A block that handed nothing leaves the first piece's all-reduce to be issued here. */
        if (pc.handed == 0) on1(true, [&] { ar_mix(0); });
        wait1();
        int64_t issued = pc.handed > 0 ? pc.handed : 1;
        bool behind = pc.handed == 0;     /* lane 1 already behind the whole mixer */
        for (int64_t p = 0; p < K; ++p) {
            if (p + 1 < K && p + 1 >= issued) {
                on1(!behind, [&] { ar_mix(p + 1); });
                behind = true;
                issued = p + 2;
            }
            l.hc_mix.write(c, T, r0(p), rn(p));
            l.hc_ffn.read(c, T, r0(p), rn(p));
            l.mlp.pass(c, T, r0(p), rn(p), false);
            /* Lane 0 waits here for the next piece's mixer all-reduce and the last piece's MoE
             * one, both issued before this pass; lane 1 then takes this piece's behind its
             * gather. */
            wait1();
            on1(true, [&] { l.mlp.issue_ar(c, r0(p), rn(p)); });
        }
        l.mlp.route_report(c);
        /* The writes, and the next layer's reads beside the last piece's all-reduce. PLE adds into
         * every row of the stream before the next layer reads it, so there the reads wait. */
        const Layer* nx = li + 1 < n ? &m.layers[(size_t)li + 1] : nullptr;
        const bool ple_next = nx && li + 1 == m.ple_layer;
        for (int64_t p = 0; p < K; ++p) {
            if (p == K - 1) wait1();
            l.hc_ffn.write(c, T, r0(p), rn(p));
            /* The projection of this piece, before the next layer reads it. */
            l.proj.step(c, r0(p), rn(p));
            if (nx && !ple_next) nx->hc_mix.read(c, T, r0(p), rn(p));
        }
        if (ple_next) {
            m.ple.step(c, batch);
            for (int64_t p = 0; p < K; ++p) nx->hc_mix.read(c, T, r0(p), rn(p));
        }
    }
}

/* The trunk one stream at a time over the whole step: every step a span of 1 makes, and every
 * other where step_pipelined's conditions do not hold. */
static void step_serial(RadCtx* c, const RadBatch* batch, Model& m, const int32_t* pcu) {
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;
    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    /* Unconditional, unlike qwen35_fp8's: the first block has no norm to fold it into, only a
     * gated-residual read. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE);
    /* The PLE's hash, ahead of every layer: its host gather waits for it, and the layers issued
     * between the two keep the device busy while the host gathers (rad_block_ple.h). */
    if (m.ple_layer >= 0) m.ple.ids(c, batch);
    m.enter.step(c, T);

    if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    for (int64_t li = 0; li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
        /* Before the attention connection reads the stream, which is where the reference puts it. */
        if (li == m.ple_layer) m.ple.step(c, batch);
        l.hc_mix.read_pieces(c, T);
        /* The indexer runs first, since the attention reads its selection. The block's input is
         * already normed and quantised by the connection (`ext_in`), so `w.h` is final. */
        if (l.full) { l.qsa.step(c, l.attn.w.h, batch, pcu); l.attn.step(c, batch); }
        else        l.gdn.step(c, batch);
        l.hc_mix.write(c, T, 0, T);
        dbg_resid(c, (int)li, "mix", g.n_embd, m.b_h, m.a_x.x);
        l.hc_ffn.read_pieces(c, T);
        l.mlp.step(c, batch);
        l.hc_ffn.write(c, T, 0, T);
        /* The request's control vector, on the stream the write just finished. */
        l.proj.step(c, 0, T);
        dbg_resid(c, (int)li, "ffn", g.n_embd, m.b_h, m.a_x.x);
    }
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;

    /* An encoder pass runs only the vision tower (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* An MTP pass runs only the head. The trunk already left its wide stream in `h`; the head
     * turns the rows it wants into a draft token or one more position of its own attention
     * history, by the sign of `draft_pass` (rad_block_mtp_hc.h). Zero is a trunk step. */
    if (batch->draft_pass != 0) {
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `mtp.` head, or declare ran at max_spec 0");
            return;
        }
        /* Fill the head's own cos/sin rows. A draft round runs past the committed end (round k
         * writes K and V at position head_i + k - 1), at rows the preceding trunk step never
         * wrote. Without this, later rounds would rotate by stale rows: only acceptance drops, and
         * because the indexer reads the same table, the head's QSA selection and so the accepted
         * text shift at tie level at long context. A row depends only on its position, so
         * re-filling is idempotent. */
        if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
        m.mtp.qsa_win_cu = pipe_bounds(c, batch, m);
        m.mtp.step(c, batch);
        return;
    }

    const int32_t* pcu = pipe_bounds(c, batch, m);
    if (pipelined(m, T)) step_pipelined(c, batch, m, pcu);
    else                 step_serial(c, batch, m, pcu);

    m.mixer.read_pieces(c, T);

    if (batch->n_out > 0) {
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
    }
}

/* The default draft depth, used by `--num-speculative-tokens auto` and by rad-convert.
 *
 * 3 is an operating point, not a model property (`mtp_num_hidden_layers` only says the one-layer
 * head exists), so `draft_depth_fixed` stays 0. Throughput peaks at 3 on this model: per-position
 * acceptance stays high at every depth, but each extra verify row costs real time because the
 * weights do not fit device memory (more experts streamed, more n-gram rows faulted from the
 * mapped table). */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    if (meta_int2(meta, "qwen4exp.mtp_num_hidden_layers", "mtp_num_hidden_layers", 0) > 0)
        out->draft_depth = 3;
    out->shape_probe_ok = 1;      /* declare() writes scratch under shape_probe */
    /* --kv-cache-dtype fp8_v4 / k6_v4: libr4d has their rows for the QSA layers' geometry
     * (head_dim 256, twelve queries per KV head; gqa6 with 4-token pages at four ranks), so other
     * geometries are refused at startup rather than in declare. */
    {
        const long long hd  = meta->head_dim ? meta->head_dim
                                             : (meta->n_head ? meta->n_embd / meta->n_head : 0);
        const long long nkv = meta->n_head_kv ? meta->n_head_kv : meta->n_head;
        if (hd == 256 && nkv > 0 && meta->n_head == 12 * nkv)
            out->kv_quant_ok = (1 << RAD_KVQ_V4) | (1 << RAD_KVQ_K6V4);
    }
    return RAD_OK;
}

}  /* namespace qwen4exp_fp8 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(qwen4exp_fp8)
RAD_ARCH_PLUGIN(qwen4exp_fp8, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next): hybrid Gated DeltaNet / gated attention, a gated "
                "residual four streams wide and 512 routed experts, from a dense bf16 checkpoint "
                "quantised ahead of time by a recipe")
#endif
