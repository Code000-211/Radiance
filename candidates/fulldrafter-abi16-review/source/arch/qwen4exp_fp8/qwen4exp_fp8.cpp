/* qwen4exp_fp8 -- Qwen4-Exp (`model_type: qwen4_exp`), the architecture Qwen3.8-Flash-Next ships
 * and the one Qwen says Qwen4 will be built on. DENSE bf16 checkpoint, quantised by a recipe ahead
 * of time (data/recipes/qwen4exp-w4nl64-i8-hc8m.recipe is the served one): int8 or fp8 trunk
 * linears with activations of the same kind, four-bit routed experts, or bf16 throughout.
 *
 * ============================== WHAT IT IS, RELATIVE TO WHAT ALREADY RUNS ==============================
 *
 * In transformers it is `Qwen4ExpTextModel`, a subclass of `Qwen3_5MoeTextModel` -- and this tree
 * already serves that (arch/qwen35moe_fp8, Qwen3.6-35B-A3B-FP8). So most of this file is
 * composition over the same blocks, and what is genuinely new is small and named:
 *
 *   SAME, reused as-is          the Gated DeltaNet (conv width 4, key/value head dim 128), the
 *                              GATED full attention (`attn_output_gate`, q_proj twice as wide,
 *                              half of it a sigmoid gate on the attention output), partial rotary
 *                              (64 of 256 dims), the routed feed-forward with a shared expert and
 *                              its own sigmoid gate, the MTP position.
 *   NEW, in this tree           the GATED RESIDUAL (arch/common/rad_block_hc.h), the n-gram / PLE
 *                              layer (arch/common/rad_block_ple.h), QSA's indexer
 *                              (arch/common/rad_qsa.h), the STACKED dense expert tensors
 *                              (tools/rad_convert.cpp's slice_stacked), gqa 12, and the delta
 *                              net's SIGMOID output gate. The last two sections of this comment
 *                              say what the first two of those are for.
 *
 * Qwen3.8-Flash-Next: 48 layers of 12 x (3 x GatedDeltaNet -> MoE, then 1 x QSA -> MoE), hidden
 * 2560, hc_count 4 so the residual stream is 10240, 512 experts of 640 at top-10 plus a shared 640,
 * 24 query heads and 2 KV heads at head_dim 256, vocab 248320.
 *
 * ============================== THE RESIDUAL STREAM IS FOUR TIMES AS WIDE ==============================
 *
 * There is no `input_layernorm`, no `post_attention_layernorm` and no final `norm` in this
 * checkpoint. In their place every block has a GATED RESIDUAL connection -- a grouped norm over a
 * 10240 stream, a low-rank data-dependent read gate that mixes the four sub-streams down to 2560,
 * and four data-dependent scalars that gain the block's output back into all four. 96 of them, plus
 * a 97th with no write half that collapses the stream for the lm_head.
 *
 * arch/common/rad_block_hc.h holds the arithmetic and docs/OPS.md's `hc_read` / `hc_write` rows
 * hold the schema. What matters HERE is that the blocks do not own their norms: this plugin
 * passes `Src::norm = nullptr`, which is the three fp8 blocks' "the input is already normed and
 * quantised" mode, and it therefore also gives up the two fusions that mode is built on -- the
 * residual add folded into the next norm, and the all-reduce folded into it. The all-reduce rides
 * in the gated residual's write instead, wherever a kernel serves that (HyperConn::Config::reduce,
 * see declare()).
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * The checkpoint is bf16 THROUGHOUT -- it carries no `quantization_config` at all -- so this
 * plugin's quantisation descriptor is the EMPTY one and the formats it serves are made by a
 * recipe (data/recipes/qwen4exp-*.recipe). The plugin reads each weight's encoding and declares
 * what that is; the blocks that read a weight in more than one format choose their ops by it:
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
 * `moe_intermediate_size` is 640 = 5 x 128. A rank's even share at tp2 is 320, which is NOT a whole
 * number of 128-column blocks -- the fp8 scale block, the 4-bit scale group and the rotation all
 * sit on them. So each rank takes three blocks of the experts of one parity and two of the
 * other's, the router stays replicated so both ranks agree on the routing, each rank computes its
 * slice of every routed expert, and THE BLOCK'S EXISTING END-OF-BLOCK ALL-REDUCE SUMS THE
 * PARTIALS. No all-to-all, no extra collective. The section in declare() carries the detail, and
 * why this is preferred to splitting the experts themselves between the ranks.
 *
 * ============================== PLE AND QSA ==============================
 *
 * PLE / THE N-GRAM EMBEDDING is 51B of the model's 180B parameters -- a 320,001,536-row table of
 * 160-dim embeddings, gathered 16 rows a token by a hash of the last three token ids, read once at
 * layer 1 and injected into the wide stream. Without it this is a 125B model missing a third of
 * its parameters.
 *
 * QSA'S INDEXER picks the best `indexer_budget / indexer_compress_ratio` = 512 blocks of 4 tokens
 * for each query and attends to those alone. Below `indexer_budget + indexer_compress_ratio - 1` =
 * 2051 tokens of context EVERY complete block is selected, so QSA is DENSE CAUSAL ATTENTION bit
 * for bit -- which is the oracle any change to it is checked against, and is also why the engine
 * issues the cheaper dense attention there. Above it the selection is per QUERY on both paths
 * (docs/QSA.md, arch/common/rad_qsa.h), so a long prompt is prefilled with the model's own
 * attention and not a denser stand-in.
 *
 * The vision tower is declared through arch/common/rad_block_vit.h, only when the container
 * carries one and the deployment asked for media; rad-convert writes only declared weights.
 */
#include <cstdlib>
#include <cstring>
#include <string>
#include <mutex>
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
#include <arch/rad_block_dflash2.h>
#include <arch/rad_block_dloop.h>
#if __has_include("arch/common/rad_block_fulldrafter.h")
#include "arch/common/rad_block_fulldrafter.h"
#elif __has_include("../common/rad_block_fulldrafter.h")
#include "../common/rad_block_fulldrafter.h"
#else
#include <arch/rad_block_fulldrafter.h>
#endif

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
};

struct Model {
    Names          nm{""};
    Geom           g{};
    GdnFP8::Config gcfg{};
    MoeFP8::Config moecfg{};
    /* The trunk's linears are int8 (LinearFP8::i8), read off the first delta-net layer's input
     * projection before the activation buffers are declared. */
    bool           trunk_i8 = false;
    HyperConn::Config hccfg{};
    QsaIndexer::Config qsacfg{};
    QsaIndexer::Wire   qsaw{};
    rad_kvgroup        kv_qsa_bkey = 0;   /* created by the first QSA layer's declare() */

    int64_t n_rot = 0;
    int64_t n_run = 0;          /* layers this step actually issues; the bisect truncates it */

    rad_kvgroup kv_full = 0, kv_state = 0, kv_conv = 0;
    /* PLE lives on exactly one layer; -1 when the checkpoint has none or this build does not
     * declare that layer. */
    PleLayer    ple;
    int64_t     ple_layer = -1;

    /* THE WIDE STREAM and the narrow block input. `b_h` is the residual stream -- hc * n_embd a
     * token -- and `a_x` is what a connection's read produces and what the block it fronts writes
     * its delta back into. One of each for the whole model, as qwen35_fp8 has one `x` and one `h`:
     * a connection's read is immediately followed by its block and then by its write, so nothing
     * in `a_x` or `b_inj` has to survive past the write that consumes it. DECLARATION ORDER IS
     * WHAT THE BUFFER PLANNER'S LIVENESS IS (see the note at declare()), so that sentence is a
     * statement about this file's order and not a hope. */
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
    /* The ROTATED codes of the block input, for the routed arm alone -- only `q` and `s` are used
     * and `x` stays zero, because the bf16 it was quantised from is `a_x`'s and is unrotated.
     * Declared only when the container asks for a rotation; zero otherwise, which is how the MoE block
     * knows the pair is absent. */
    ActFP8  a_hr{};

    rad_buf b_hout = 0, b_logits = 0;

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h). Declared only when
     * the deployment speculates -- it is ~2.5 GiB of weights a non-speculative server would read
     * off disk, place in VRAM and never issue, and `max_spec` is the engine already saying whether
     * it intends to draft. rad-convert declares at a non-zero max_spec on purpose, because a
     * CONTAINER should carry what the model can serve even when the run that made it would not. */
    bool        have_mtp = false;
    MtpHcBlock  mtp{};
    rad_buf     b_mtp_hin = 0, b_mtp_hw = 0, b_mtp_e = 0, b_mtp_hout = 0;
    rad_buf     b_mtp_dq = 0, b_mtp_ds = 0, b_mtp_dsum = 0;
    rad_buf     b_mtp_dlog = 0, b_mtp_didx = 0, b_mtp_dval = 0;
    rad_buf     b_mtp_dpair = 0, b_mtp_dgath = 0;
    rad_buf     b_mtp_cand_hist = 0, b_mtp_tree = 0;

    DLoopBlock dloop{};
    /* ---- FullDrafter Discrete Two-Stage Drafter (arch/common/rad_block_fulldrafter.h) */
    bool                         have_fulldrafter = false;
    radiance::FullDrafterWeights fulldrafter_weights{};
    rad_buf                      b_fd_hout = 0;
    rad_buf                      b_fd_logits = 0;
    rad_buf                      b_fd_cand = 0;
    rad_buf                      b_fd_unary = 0;
    rad_buf                      b_fd_dpair = 0;
    rad_buf                      b_fd_dgath = 0;
    rad_buf                      b_fd_tokens = 0;

    rad_buf                      b_fd_conv_x = 0;
    rad_buf                      b_fd_conv_delta = 0;
    rad_buf                      b_fd_conv_out = 0;
    rad_weight                   w_fd_conv_base = 0;
    rad_op                       op_fd_conv = 0;

    rad_op                       op_fd_logits = 0;
    rad_op                       op_fd_topk = 0;
    rad_op                       op_fd_gather = 0;
    rad_op                       op_fd_merge = 0;
    rad_op                       op_fulldrafter = 0;
    /* The trunk's taps at each drafting sequence's LAST ACCEPTED row, gathered on the query pass
     * through RadBatch::draft_out_ids into rows 0..n_seq-1: kld_tap0, kld_tap1 and the lm_head's
     * input (fd_ctx_src). Persistent like the taps, so the planner's liveness is untouched. */
    rad_buf                      b_fd_ctx[3] = {};
    rad_op                       op_fd_ctx[3] = {};

    HcEnter   enter;
    HyperConn mixer;            /* the 97th connection: no write half */

    /* ---- --kld-taps (core/kld.h): a drafter's training features, recorded and never served.
     * Tap k < n_ktap is the block input of layer ktap_layer[k] -- what that layer's attention
     * connection reads out of the wide stream, `a_x` right after hc_mix.read -- and tap n_ktap is
     * the hidden state the lm_head reads, `a_x` right after the mixer. Each is gathered at the
     * step's `out_ids` into its own PERSISTENT buffer, so the engine can copy the rows off after
     * the step without the buffer planner having given that storage to anything else. Declared
     * only when RAD_KLD_TAPS names layers, which only `--kld-record --kld-taps` sets. */
    int        n_ktap = 0;
    int        ktap_layer[8] = {};
    rad_buf    b_ktap[9] = {};
    rad_op     op_ktap[9] = {};
    rad_buf    b_ktap_anchor = 0;
    rad_op     op_ktap_anchor = 0;

    /* ---- THE DLOOP TAP PLANES, which are NOT the recording taps above.
     *
     * `kld_tap*` holds one row per SAMPLED position, gathered through `out_ids`, and is left
     * unwritten on a step that samples nothing -- the middle chunk of a long prefill. The
     * full-context drafter's persistent cache needs the opposite: ONE ROW PER VERIFIED TOKEN of
     * every chunk, in TRUNK-ROW order, so that the winning-path index the engine stages names the
     * same row here. They are therefore declared at `max_tok` and filled by `cast` (a strided bf16
     * row copy) with no index plane at all.
     *
     * THREE planes, in the op's concatenation order: [tap0,tap1,tap2]. Declared only for DLoop,
     * and only for the same two depths the recording uses -- a model that reads a different pair
     * of layers is a different drafter. */
    rad_buf    b_dtap[3] = {};
    rad_op     op_dtap[3] = {};

    /* ---- THE DFLASH2 DRAFTER (arch/common/rad_block_dflash2.h), the original
     * incoai/Qwen3.8-Flash-Next-DFlash2 checkpoint, served in its OWN precision.
     *
     * IT IS NOT THE DLOOP DRAFTER ABOVE AND THE TWO ARE NOT SPELLINGS OF ONE THING. DLoop is this
     * tree's custom full-context drafter over H768 taps; DFlash2 is a second, smaller transformer
     * that reads FIVE of this trunk's completed layer outputs and proposes a whole block in one
     * forward. Both can be named by a deployment and they conflict -- `declare` refuses the pair
     * by name rather than picking one, because the engine drives a single drafter.
     *
     * ITS TAPS ARE NOT `kld_tap*` OR `b_dtap*`. The capture patch that produced this checkpoint
     * (patches/qwen38-dflash2-capture.patch) takes the COMPLETED output of layers 5, 15, 25, 35
     * and 45 and averages the four hyper-connection sub-streams down to one 2560-wide row a tap:
     *
     *     aux = mean over hc_count of hidden.unflatten(-1, (hc, 2560))
     *
     * `kld_tap` is the block INPUT of a layer, gathered through `out_ids`, and `b_dtap` is the
     * mixer's output; neither is this. The mean is why `b_dfsrc` is 2560 wide a tap and not
     * 10240, and it is the one thing an op in this tree cannot express -- the refusal and the
     * precise missing primitive are at the tap declaration in `declare()`. */
    bool          have_df = false;
    Dflash2       df{};
    Dflash2::Wire dfw{};
    rad_kvgroup   kv_draft = 0;
    int           df_tap_layer[Dflash2::MAX_LAYERS] = {};
    int           df_n_taps = 0;
    rad_buf       b_dfsrc = 0;            /* [max_tok, n_taps * n_embd], one tap a column block */
    rad_op        op_df_tap[Dflash2::MAX_LAYERS] = {};  /* `hc_mean` a tap into its columns */

    std::vector<Layer> layers;

    rad_weight w_tok = 0, w_lm_head = 0;
    rad_op op_embed = 0, op_embed_ar = 0, op_rope_cs = 0, op_gather = 0, op_logits = 0;

    /* ---- media (arch/common/rad_block_vit.h): the vision tower an encoder pass runs, and the
     * scatter that puts its rows over the token embeddings. Declared only when the container
     * carries a tower and the deployment asked for media (RadBuildCtx::max_enc_patches). */
    bool        media = false;
    VisionTower vit{};
    MediaRows   mrows{};

};

static Model g_model[MAX_RANKS];
static radiance::FullDrafterWeights g_shared_fd_weights;
static std::mutex                   g_shared_fd_mu;
static std::string                  g_shared_fd_path;
static size_t                       g_shared_fd_ranks = 0;

/* Which tap each FullDrafter context input is: the two layer taps, then the lm_head's input. */
static inline rad_buf fd_ctx_src(const Model& m, int j) {
    return m.b_ktap[j < 2 ? j : m.n_ktap];
}

static inline bool is_full_attention(int64_t l, int64_t interval) {
    return interval > 0 && ((l + 1) % interval == 0);
}

/* THE CONTAINER'S OWN SCHEDULE, CHECKED AGAINST THE ONE WE DERIVE -- qwen35_fp8 argues this at
 * length and the argument is identical here, with one addition: Qwen4-Exp's config says
 * "full_attention" for the layers that actually run QSA (the HuggingFace post_init rewrites them to
 * "qwen_sparse_attention" in memory and the file on disk says the former), so BOTH spellings are
 * accepted for a full layer and neither is accepted for a linear one. */
static int check_layer_schedule(const RadModelMeta* meta, int64_t n_layer, int64_t interval) {
    const char* s = rad_meta_gets(meta, "text_config.layer_types", nullptr);
    if (!s) s = rad_meta_gets(meta, "layer_types", nullptr);
    if (!s || !*s) return RAD_OK;

    int64_t l = 0;
    const char* p = s;
    while (*p && l < n_layer) {
        while (*p == ' ') ++p;
        if (!*p) break;
        const char* e = p;
        while (*e && *e != ' ') ++e;
        const size_t n = (size_t)(e - p);
        const bool says_full = (n == 15 && std::strncmp(p, "full_attention", 14) == 0) ||
                               (n == 14 && std::strncmp(p, "full_attention", 14) == 0) ||
                               (n == 22 && std::strncmp(p, "qwen_sparse_attention", 21) == 0) ||
                               (n == 21 && std::strncmp(p, "qwen_sparse_attention", 21) == 0);
        const bool says_lin  = (n == 17 && std::strncmp(p, "linear_attention", 16) == 0) ||
                               (n == 16 && std::strncmp(p, "linear_attention", 16) == 0);
        if (!says_full && !says_lin) {
            fprintf(stderr, "radiance: qwen4exp: layer_types[%lld] is '%.*s', which is neither a "
                            "linear_attention nor a full/sparse attention layer\n",
                    (long long)l, (int)n, p);
            return RAD_E_UNSUPPORTED;
        }
        if (says_full != is_full_attention(l, interval)) {
            fprintf(stderr, "radiance: qwen4exp: the container says layer %lld is %s and the "
                            "interval rule (every %lld-th) says %s. The schedule decides which "
                            "layers are linear, so a disagreement is a different model.\n",
                    (long long)l, says_full ? "full attention" : "linear",
                    (long long)interval, is_full_attention(l, interval) ? "full" : "linear");
            return RAD_E_UNSUPPORTED;
        }
        ++l;
        p = e;
    }
    if (l != n_layer) {
        fprintf(stderr, "radiance: qwen4exp: layer_types names %lld layers and the container has "
                        "%lld\n", (long long)l, (long long)n_layer);
        return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}

/* An env switch that is OFF unless set to something other than "0". Spelled once because there are
 * four of them and each one is a statement that the model being run is not the model on disk. */
/* HOW MANY ROWS THE N-GRAM TABLE HAS, and it is a prime search because the checkpoint's own answer
 * arrives too late.
 *
 * Each of the `heads` hash heads gets its own vocabulary: the 1st, 2nd, ... nth prime above
 * `ngram_vocab_size_base - 1`. The table is their sum, padded to `divisor`. The reference builds
 * exactly this in `Qwen4ExpTextNGramEmbedding.__init__` and the resulting per-head sizes ARE in the
 * checkpoint -- `ngram_heads_vocab_sizes` -- which is where the served hash reads them from, and
 * nothing here re-derives those.
 *
 * WHAT IS DERIVED IS THE TOTAL, and only because a weight's shape must be declared before any
 * tensor has been opened. Getting it wrong would ship a container whose directory claims one row
 * count and carries another's bytes, so rad-convert refuses exactly that: the declared leading
 * extent of a multi-source concat has to equal its sources'. This search is checked against 128
 * shards of real bytes on every convert.
 *
 * The cost is 16 primes near 2e7 by trial division -- a few hundred thousand divisions, once, at
 * declare. */
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

/* A RANK'S PLACE WHERE THE SHARED EXPERT'S BLOCKS DO NOT DIVIDE over the ranks, the first places
 * taking one block more. A plain expert is a routed expert's width in 64-wide tiles, which need
 * not divide either, and MoeFP8::declare hands its extra tiles to the first ranks by index; the
 * shared expert's extra blocks start at the rank after the last of those, so the two remainders
 * land on different ranks while there are ranks enough for both. The routed experts leave no
 * remainder to avoid: at two ranks the parity split and at four the four-class split give every
 * rank the same width of them. At four ranks and this family's 640 columns -- ten tiles, five
 * blocks -- ranks 0 and 1 take a tile more and rank 2 the block; at two the ten tiles divide and
 * the place is the rank. */
static int64_t shared_place(int64_t n_ff_all, int world, int rank) {
    const int64_t first = (n_ff_all / 64) % world;
    return (rank - first + world) % world;
}

/* "5 15 25 35 45" -> the trunk layers whose COMPLETED output a DFlash2 drafter reads.
 *
 * rad-convert flattens a config.json array into a space-separated string, so this is the shape
 * every list-valued key arrives in -- qwen35_fp8's parse_tap_layers is the same reader, and the
 * two are kept separate because the two models' tap semantics are not the same thing (see
 * declare_df). Out-of-range entries are dropped rather than clamped: a drafter that names a layer
 * this model does not have is paired with the wrong target, and silently reading layer 7 instead
 * of 45 would make that a quality mystery. */
static int parse_target_layer_ids(const char* s, int64_t n_layer, int* out, int max) {
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

/* Whether this container carries an original DFlash2 drafter and the deployment means to spec it.
 *
 * The container says so the way qwen35_fp8 reads it: rad-convert carries the drafter's own
 * config.json under `draft.`, and `draft.architectures` names the family. `max_spec` is the
 * engine's own statement that it intends to draft; at 0 the drafter's weights stay off the card
 * and no tap is recorded. */
/* `RADIANCE_DARTREE=1` ASKS FOR THE BOUNDED TREE PROPOSAL in whichever built-in drafter this
 * program declares. Read in ONE place and handed down, because two reads of one environment
 * variable in one declare is how a graph ends up half on the tree path: the drafter's selector,
 * the target's stateful verifier and the scheduler's depth policy all have to agree about whether
 * this deployment verifies a tree, and the scheduler takes its answer from the DECLARED OPS
 * (core/sched/tree_state.h) rather than from this flag. */
static bool dartree_requested() {
    const char* v = std::getenv("RADIANCE_DARTREE");
    return v && *v && std::strcmp(v, "0") != 0;
}

static bool wants_dflash2(const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (ctx && ctx->max_spec <= 0) return false;
    const char* da = rad_meta_gets(meta, "draft.architectures", "");
    return da && std::strstr(da, "DFlash") != nullptr;
}

/* ---- THE ORIGINAL DFLASH2 DRAFTER (arch/common/rad_block_dflash2.h).
 *
 * Split out of declare() for the reason qwen35_fp8 splits its own: it is A SECOND MODEL and not a
 * head. It has its own layer count, head count, feed-forward width, rope base and norm convention,
 * none of which the trunk's Geom can carry, so it starts from a COPY of the trunk's -- n_embd, the
 * deployment bounds, the rank and the wire -- and overwrites everything its own config states.
 *
 * SERVED IN ITS OWN PRECISION. This is the checkpoint incoai/Qwen3.8-Flash-Next-DFlash2 ships:
 * bf16 weights, bf16 activations, and the TARGET's own bf16 lm_head. Nothing here is quantised,
 * no recipe touches it, and `Config::bf16` is what selects every one of the block's GEMM rows and
 * its head (rad_block_dflash2.h, dflash2_dense). The legacy qwen35 deployment leaves that flag
 * false and keeps the fp8 graph it always had; the two modes share every shape.
 *
 * ITS TAPS ARE NOT ANY OTHER DRAFTER'S. The capture patch that made this checkpoint
 * (patches/qwen38-dflash2-capture.patch) appends, at the END of a completed zero-based layer,
 *
 *     hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)
 *
 * -- the UNGATED arithmetic mean over the four hyper-connection streams of that layer's OUTPUT, at
 * 2560 wide. `kld_tap*` is a layer's block INPUT gathered through `out_ids`; `b_dtap*` is the
 * mixer's output at trunk width; `hc_read` is a gated, per-stream-RMS-normalised read of a block
 * input (libref/ref_hc.cpp). None of them is this, and substituting one would feed `fc` numbers
 * the drafter was never trained on -- which costs acceptance and reads as a quality mystery. */
static int declare_df(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx,
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
    /* PLAIN RMS NORM: `x_hat * w`, where this trunk's every norm is `x_hat * (1 + w)`. The block
     * header states the convention and the checkpoint agrees with it (rad_block_dflash2.h). Two
     * conventions in one process is why this is a second Geom and not a flag. */
    dg.wadd    = 0.0f;
    dg.wadd_qk = 0.0f;

    if (dg.n_layer <= 0 || dg.head_dim <= 0 || dg.n_head_all <= 0 || dg.n_head_kv_all <= 0 ||
        dg.n_ff_all <= 0 || !(dg.eps > 0.0f) || !(dg.theta > 0.0f)) {
        fprintf(stderr, "radiance: qwen4exp: the container's `draft.` config does not describe a "
                        "DFlash2 drafter: layers %lld, heads %lld / kv %lld, head_dim %lld, "
                        "ffn %lld, rms_eps %g, rope_theta %g\n",
                (long long)dg.n_layer, (long long)dg.n_head_all,
                (long long)dg.n_head_kv_all, (long long)dg.head_dim, (long long)dg.n_ff_all,
                (double)dg.eps, (double)dg.theta);
        return RAD_E_INVAL;
    }
    int s;
    if ((s = divide_or_fail(dg.n_head_all,    dg.world, "draft num_attention_heads",    &dg.n_head))    < 0) return s;
    if ((s = divide_or_fail(dg.n_head_kv_all, dg.world, "draft num_key_value_heads",   &dg.n_head_kv)) < 0) return s;
    if ((s = divide_or_fail(dg.n_ff_all,      dg.world, "draft intermediate_size",     &dg.n_ff))      < 0) return s;

    /* THE DRAFTER BORROWS THE TARGET'S EMBEDDING TABLE AND lm_head -- it ships neither -- so a
     * vocabulary that disagrees is not a drafter for this target at all. */
    const long long dvocab = rad_meta_geti(meta, "draft.vocab_size", dg.n_vocab_all);
    if (dvocab != dg.n_vocab_all) {
        fprintf(stderr, "radiance: qwen4exp: the DFlash2 drafter claims a %lld-token vocabulary "
                        "and the target has %lld. The drafter ships no embedding table and no "
                        "lm_head; it reads the target's, so the two have to be the same "
                        "vocabulary.\n", dvocab, (long long)dg.n_vocab_all);
        return RAD_E_UNSUPPORTED;
    }

    /* THE TRUNK LAYERS THE DRAFTER READS. Named by the container, and the count is a hard SHAPE
     * requirement as well as a semantic one: `fc` is [n_embd, n_taps * n_embd], so a list of the
     * wrong length cannot even be declared against the checkpoint's tensor. */
    m.df_n_taps = parse_target_layer_ids(
        rad_meta_gets(meta, "draft.dflash_config.target_layer_ids", nullptr),
        m.g.n_layer, m.df_tap_layer, Dflash2::MAX_LAYERS);
    if (m.df_n_taps <= 0) {
        fprintf(stderr, "radiance: qwen4exp: this DFlash2 container names no "
                        "`draft.dflash_config.target_layer_ids`. The drafter reads a fixed set of "
                        "COMPLETED trunk layer outputs averaged over the hyper-connection "
                        "streams; without them there is nothing to feed its `fc`.\n");
        return RAD_E_INVAL;
    }
    /* `fc` IS [n_embd, n_taps * n_embd]. A `target_layer_ids` entry outside 0..n_layer-1 is
     * dropped rather than clamped, so a shortened list would leave the tap plane narrower than
     * `fc`'s K. That is caught where it belongs -- `dflash2_dense` declares `fc` at exactly
     * {n_embd, n_taps * n_embd} and the builder checks the declaration against the checkpoint's
     * own tensor -- so the count is NOT re-derived here from a second reading of the same file.
     * What this file must say is that the list is non-empty; above it does. */

    Dflash2::Config dc{};
    dc.n_taps   = m.df_n_taps;
    dc.block    = ctx->max_spec + 1;
    /* THE BOUNDED DarTree, OPT-IN BY NAME. `RADIANCE_DARTREE=1` selects the tree selector for
     * whichever built-in drafter this program declares; unset or zero declares exactly the graph
     * it always did.
     *
     * THE TRAINED DEPTH IS NOT EXTENDED. A query pass runs `block` rows a sequence and the block
     * is the checkpoint's trained depth. The tree adds NO rows to it -- it is a second selector
     * over the candidate planes that pass already produced -- so `block` stays 4 (three proposal
     * levels) whatever RADIANCE_DARTREE asks for, and only the VERIFICATION row count moves.
     * NUM_SPECULATIVE_TOKENS is that count: the tree is pruned to it, never grown to match. */
    const bool dartree = dartree_requested();
    if (dartree) {
        const long long trained = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
        if (trained < 4) {
            fprintf(stderr, "radiance: qwen4exp: RADIANCE_DARTREE wants three proposal levels and"
                            " this container declares block_size %lld. A tree cannot propose"
                            " deeper than the drafter was trained; unset RADIANCE_DARTREE.\n",
                    trained);
            return RAD_E_UNSUPPORTED;
        }
        /* THE TWO NUMBERS THAT MUST AGREE, AND THE BUILDER IS WHAT MAKES THEM.
         *
         * `RadDrafterDecl::depth` has to equal RadBuildCtx::max_spec -- declare_drafter refuses
         * anything else, and it is right to: the plugin sized its buffers from max_spec and the
         * engine drives that depth. And a multi-stage block drafter (block < depth) has to satisfy
         * `depth % (block - 1) == 0`, because the block is how many tokens a draft pass proposes
         * from a fixed lattice. With block 4 that is `depth % 3 == 0`.
         *
         * SO max_spec IS THE VERIFICATION ROW BUDGET MINUS THE ROOT, and RADIANCE_DARTREE accepts
         * only the windows a 3-level tree can fill: 3, 6, 9, 12, 15. Anything else is refused here,
         * by name, rather than served at a depth the scheduler would then clamp -- a tree whose
         * rows exceed the verifier's is a proposal the target cannot check, and one whose rows
         * undershoot the lattice wastes the pass. */
        if (ctx->max_spec <= 0 || ctx->max_spec > 15 || (ctx->max_spec % 3) != 0) {
            fprintf(stderr, "radiance: qwen4exp: RADIANCE_DARTREE over DFlash2 needs a speculative "
                            "window that is a multiple of the 3 proposal levels -- "
                            "NUM_SPECULATIVE_TOKENS in {3, 6, 9, 12, 15} -- and this deployment "
                            "asked for %d.\n", ctx->max_spec);
            return RAD_E_UNSUPPORTED;
        }
        dc.block      = 4;            /* 3 proposal levels -- the trained depth, never more */
        dc.tree       = true;
        dc.tree_nodes = ctx->max_spec + 1;   /* the verifier's rows, root included */
    }
    dc.taps     = rad_meta_geti(meta, "draft.dflash_config.conv_kernel_size", 2);
    dc.group    = rad_meta_geti(meta, "draft.dflash_config.conv_group_size", 16);
    dc.sel_rank = rad_meta_geti(meta, "draft.dflash_config.selector_rank", 0);
    dc.sel_topk = rad_meta_geti(meta, "draft.dflash_config.selector_top_k", 0);
    dc.causal   = rad_meta_geti(meta, "draft.is_causal", 0) ? 1 : 0;
    if (rad_meta_geti(meta, "draft.use_sliding_window", 0))
        dc.window = rad_meta_geti(meta, "draft.dflash_config.swa_window_size",
                                  rad_meta_geti(meta, "draft.sliding_window", 0));
    /* THE WHOLE POINT OF THIS PATH: the checkpoint's own precision, unquantised. */
    dc.bf16     = true;
    /* AND THIS CALLER'S KERNEL HOME HAS `mine`. That is not an assumption: `want_df` below
     * declares `hc_mean` for the tap plane on this same path, and `hc_mean`'s schema is `mine`'s
     * alone -- so a kernel home without that plugin cannot boot this graph at all, and the
     * DFlash2 closing convolution's fused form (`dflash_conv_add`, also `mine`'s) adds no new
     * requirement. The block gates the fusion on this flag rather than probing, because the
     * builder ABI has no way to ask whether an op is known (rad_block_dflash2.h's Config says
     * why). qwen35_fp8 shares that header and leaves it false. */
    dc.mine_fusions = true;

    if (dc.sel_rank <= 0 || dc.sel_topk <= 0) {
        fprintf(stderr, "radiance: qwen4exp: the DFlash2 drafter states no selector geometry "
                        "(rank %lld, top_k %lld). Without it the block of drafts has no path "
                        "through it.\n", (long long)dc.sel_rank, (long long)dc.sel_topk);
        return RAD_E_INVAL;
    }

    /* THE QUERY BLOCK IS A RUNTIME ROW COUNT, not a property of the weights: the convolution's
     * position law is `t % block_size` and every linear's K is `n_embd`. Both references take the
     * block at serve time and the model card deploys block 4 (3 drafts), so the container's
     * declared `block_size` is the DEFAULT -- not training proof. A shorter block is served at
     * exactly the rows asked for; a longer one, or no declaration, is refused. */
    const long long declared = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
    if (declared < 2) {
        fprintf(stderr, "radiance: qwen4exp: this DFlash2 container declares no usable "
                        "`draft.dflash_config.block_size` (%lld), so the geometry it serves is "
                        "unknown.\n", declared);
        return RAD_E_FORMAT;
    }
    if (dc.block < 2 || dc.block > declared) {
        fprintf(stderr, "radiance: qwen4exp: this serve asks for block %lld (%d draft token(s)); "
                        "this container declares block_size %lld (%lld draft token(s)). Only "
                        "1..%lld query rows are served.\n",
                (long long)dc.block, ctx->max_spec, declared, declared - 1, declared);
        return RAD_E_UNSUPPORTED;
    }
    /* Three scalars this drafter's family can carry that this implementation does not apply. Each
     * would show up as acceptance that is merely disappointing rather than as anything that looks
     * like a bug, so each is refused. */
    const double escale = rad_meta_getf(meta, "draft.dflash_config.input_embedding_scale", 1.0);
    const double omul   = rad_meta_getf(meta, "draft.dflash_config.output_multiplier", 1.0);
    const double scap   = rad_meta_getf(meta, "draft.dflash_config.final_logit_softcapping", 0.0);
    if (escale != 1.0 || omul != 1.0 || scap != 0.0) {
        fprintf(stderr, "radiance: qwen4exp: this DFlash2 checkpoint scales its input embedding "
                        "by %g and its draft logits by %g, and soft-caps them at %g. None of the "
                        "three is implemented, and a drafter that ignored them would just accept "
                        "less.\n", escale, omul, scap);
        return RAD_E_UNSUPPORTED;
    }

    /* THE DRAFTER'S OWN KV CACHE, a group of its own: different head count, different head_dim,
     * five layers against the trunk's many, and a WINDOW. RAD_KV_WINDOW recycles the blocks below
     * it, so the cache is sized by the drafter's 2048-token window and not by the context length.
     * Keep the original drafter's BF16 cache even if the target uses another cache dtype. */
    RadKVGroupDecl dd{};
    dd.kind      = dc.window > 0 ? RAD_KV_WINDOW : RAD_KV_FULL;
    dd.dtype     = RAD_BF16;
    dd.window    = dc.window;
    dd.n_head_kv = dg.n_head_kv;
    dd.head_dim  = dg.head_dim;
    m.kv_draft = rad_decl_kv_group(b, nm.f("kv_draft"), &dd);
    for (int64_t l = 0; l < dg.n_layer; ++l)
        RAD_ARCH_TRY(rad_bind_layer_kv(b, base_layer + (int)l, m.kv_draft));
    dg.kv_dtype = rad_dtype_name(dd.dtype);

    /* Activation buffers. The context pass is one row a COMMITTED TOKEN and runs over a whole
     * prefill chunk; everything after it is one row a DRAFT POSITION, `block` of them a sequence.
     * None of it is large -- the drafter's cost is its weights, not its activations. */
    const int64_t qd  = dg.n_head * dg.head_dim;
    const int64_t kvd = dg.n_head_kv * dg.head_dim;
    int64_t qr = dg.max_seqs * dc.block;
    if (qr > dg.max_tok) qr = dg.max_tok;
    const int64_t sr   = dg.max_seqs * dc.steps();

    Dflash2::Wire& dw = m.dfw;
    /* THE TAP PLANE, [max_tok, n_taps * n_embd]. One row a token of this step, each tap occupying
     * its own n_embd columns, because `fc` reads one CONTIGUOUS [n_taps * n_embd] row and a tap is
     * n_embd of them. DECLARED HERE, BEFORE THE WIRE THAT CARRIES IT: `dw.src` is handed to the
     * block's own declare, which declares `fc` against it, and a zero there declares cleanly and
     * only fails at the first context pass. qwen35_fp8 orders its own `dflash_src` the same way. */
    m.b_dfsrc = decl_b(b, nm.f("dflash_src"), dg.act_dtype,
                       {dg.max_tok, (int64_t)m.df_n_taps * dg.n_embd});
    if (!m.b_dfsrc) return RAD_E_INVAL;
    dw.src     = m.b_dfsrc;
    /* THE GATHERED ACCEPTED ROWS, when a tree can be verified. A context pass after a TREE verify
     * may store only the accepted path's tap rows, and those are `start + node[j]` -- generally not
     * the contiguous run the pass would otherwise describe.
     *
     * AS TALL AS THE TAP PLANE, because the gather runs on EVERY context pass and a prefill chunk
     * passes its identity range: T rows of a gather is this pass's row count, which is `max_tok`
     * for a chunk and `n_seq * (1 + n_spec)` for a tree verify. A buffer sized for `block` rows a
     * sequence would be overflowed by the first prefill -- and by any verify window wider than the
     * trained block, which the tree flag asks for by construction (NUM_SPECULATIVE_TOKENS is the
     * verifier's row budget). So the destination is the same extent
     * `dflash_src` has, and `fc` reads a narrowed view of it. */
    if (dartree) {
        dw.srcg = decl_b(b, nm.f("dflash_src_gathered"), dg.act_dtype,
                         {dg.max_tok, (int64_t)m.df_n_taps * dg.n_embd});
        if (!dw.srcg) return RAD_E_INVAL;
    }
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
    /* THE FP8 TWINS ARE NOT DECLARED. `Config::bf16` declares no quantiser anywhere in the block
     * and issues nothing against them, so a plane nothing writes is a plane the planner would
     * still charge -- the Wire's own note says to leave all five pairs at 0. */
    dw.sel_h   = decl_b(b, nm.f("df_selh"), dg.act_dtype, {sr, dg.n_embd});
    dw.hp      = decl_b(b, nm.f("df_hp"),   dg.act_dtype, {sr, dc.sel_rank});
    /* `dq`, `ds` and `dsum` belong to the 2-bit head alone; the bf16 head's GEMM reads `sel_h` and
     * none of the three is written or read. `dlog` is the same plane in both modes. */
    dw.dlog    = decl_b(b, nm.f("df_dlog"), dg.act_dtype, {sr, Dflash2::head_rows(dg)});
    dw.cand    = decl_b(b, nm.f("df_cand"), RAD_I32, {sr, dc.sel_topk});
    dw.unary   = decl_b(b, nm.f("df_unary"), RAD_F32, {sr, dc.sel_topk});
    if (Dflash2::head_sharded(dg)) {
        dw.dpair = decl_b(b, nm.f("df_dpair"), RAD_F32, {sr, dc.sel_topk, 2});
        dw.dgath = decl_b(b, nm.f("df_dgath"), RAD_F32, {sr, (int64_t)dg.world * dc.sel_topk, 2});
    }
    /* NAMED, because the engine reads the draft out of it by name: a DFlash2 pass proposes through
     * the selector and never through the sampler's token buffer. */
    dw.tokens  = decl_b(b, nm.f("dflash_tokens"), RAD_I32, {dg.max_seqs, dc.steps()});
    /* THE PACKED TREE, [max_seqs, 33]: dloop_dartree's root-inclusive layout, which is the one
     * `rad_declare_drafter_tree` validates and the one the engine reads back. Declared only when
     * the tree is asked for, so a linear deployment carries no plane nothing writes. */
    if (dc.tree) {
        dw.packed = decl_b(b, nm.f("dflash_tree"), RAD_I32, {dg.max_seqs, 33});
        if (!dw.packed) return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(m.df.declare(b, nm, dg, dc, base_layer, m.kv_draft, m.w_tok, "draft",
                              nm.ckpt("lm_head.weight"), dw));
    m.have_df = true;

    /* WHAT THE ENGINE DRIVES. Block: a context pass that mirrors the trunk step, then ONE pass
     * that fills the whole block. `mask_token_id` is read HERE and nowhere else -- it travels in
     * the drafter declaration, so no other component has to sniff the container for a drafter.
     * `window` must be the number this plugin gated its OWN attention with, because the scheduler
     * stages the block pass against it: two readings of "how far back can the drafter see" is a
     * drafter attending to positions the scheduler did not stage. */
    RadDrafterDecl drf{};
    drf.name           = "dflash2";
    drf.kind           = RAD_DRAFT_BLOCK;
    drf.depth          = ctx->max_spec;
    /* THE ANCHOR ROW PREDICTS NOTHING here, so the block is one row longer than the depth and the
     * head runs over rows 1..depth. Stated rather than defaulted (RadDrafterDecl::block). */
    drf.block          = dc.block;
    drf.proposal       = dw.tokens;
    drf.proposal_pitch = dc.steps();
    drf.window         = dc.window;
    drf.mask_token     = (int32_t)rad_meta_geti(meta, "draft.dflash_config.mask_token_id", -1);
    if (drf.mask_token < 0) {
        fprintf(stderr, "radiance: qwen4exp: this DFlash2 checkpoint declares no "
                        "dflash_config.mask_token_id, and every drafted position embeds it -- "
                        "there is nothing to run.\n");
        return RAD_E_FORMAT;
    }
    RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
    /* THE PROPOSAL BESIDE THE LINEAR ONE. The tree is a TOPOLOGY, so it cannot travel in
     * `dflash_tokens`'s [max_seqs, steps] plane; `rad_declare_drafter_tree` names the packed
     * [max_seqs, 33] buffer instead and the engine reads its rows back after every query pass.
     * Declared AFTER the drafter, because the builder refuses a tree with no BLOCK drafter to
     * attach it to. */
    if (dc.tree) {
        RAD_ARCH_TRY(rad_declare_drafter_tree(b, dw.packed, 16));
        rad_note(b, "the DFlash2 drafter proposes through `dflash_tree_select`, a bounded DarTree"
                    " of at most %lld node(s) over the same %lld proposal level(s) -- width %lld"
                    " per level, pruned to the verification rows the deployment asked for.",
                 (long long)(dc.tree_nodes > 0 ? dc.tree_nodes : 16), (long long)dc.steps(),
                 (long long)dc.tree_width());
    }

    rad_note(b, "the DFlash2 drafter is declared in its OWN bf16 precision: %lld layers of its "
                "own at absolute %d..%d, %lld heads / %lld kv x %lld, ffn %lld, %s attention "
                "with a %lld-token window, reading the completed trunk layers %d..%d at %d "
                "tap(s). ONE forward proposes %lld tokens -- there are no rounds -- and it reads "
                "the target's bf16 lm_head%s",
             (long long)dg.n_layer, base_layer, base_layer + (int)dg.n_layer - 1,
             (long long)dg.n_head, (long long)dg.n_head_kv, (long long)dg.head_dim,
             (long long)dg.n_ff, dc.causal ? "causal" : "non-causal", (long long)dc.window,
             m.df_tap_layer[0], m.df_tap_layer[m.df_n_taps - 1], m.df_n_taps,
             (long long)dc.steps(), Dflash2::head_sharded(dg) ? ", vocab-sharded with the merge" : "");
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx) return RAD_E_INVAL;
    if (ctx->rank < 0 || ctx->rank >= MAX_RANKS) return RAD_E_INVAL;

    /* A SIZING DECLARE WRITES SCRATCH. The step reads g_model, which the real declare filled at
     * the real max_tok; one sized for a smaller step must not replace it (RadBuildCtx). */
    static thread_local Model probe_model;
    Model& m = ctx->shape_probe ? probe_model : g_model[ctx->rank];
    m = Model{};
    m.nm = Names(ctx->scope ? ctx->scope : "");
    Names& nm = m.nm;

    /* arch_splits_ff: the routed experts' width (640 = five fp8 blocks) is split by the plugin
     * below, by whole blocks and parity, at any world up to the block count -- three ranks included,
     * which 640 does not divide. geom_from must not refuse what the plugin then answers. */
    RAD_ARCH_TRY(geom_from(m.g, meta, ctx, "bf16", RAD_BF16, /*arch_splits_ff=*/true));
    Geom& g = m.g;
    const bool use_dloop = DLoopBlock::requested();
    /* WHO VERIFIES A TREE. The target's stateful layers verify a branched proposal only when their
     * tree kernels are declared, and the scheduler asks the DECLARED PROGRAM for that answer
     * (core/sched/tree_state.h): a `gdn_tree_commit`, `qsa_tree_*` or PLE tree op in the graph is
     * what makes a branched window admissible, and nothing else is. So a deployment that declares
     * a tree PROPOSAL must also declare the tree VERIFIER, or `set_draft_tree` refuses every
     * branched window and the proposal is silently downgraded to its greedy chain -- correct, and
     * no faster than the linear walk it replaced.
     *
     * DLoop needs it because that is the drafter it ships; the built-in drafters need it only when
     * RADIANCE_DARTREE asked for the tree, which is why the two are one predicate here and not a
     * second flag downstream. The two are ALSO mutually exclusive, which the DLoop gate below
     * enforces: one drafter a program. */
    const bool dartree = dartree_requested();
    const bool tree_state = use_dloop || dartree;
    if (use_dloop) {
        const char* fd=std::getenv("RADIANCE_FULLDRAFTER");
        const char* sc=std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
        if ((fd && *fd && std::strcmp(fd,"0")) || (sc && *sc) ||
            meta_int2(meta,"qwen4exp.fulldrafter","fulldrafter",0)>0) {
            rad_note(b,"RADIANCE_DLOOP_DRAFTER and FullDrafter cannot be enabled together");
            return RAD_E_INVAL;
        }
        if (g.n_embd!=2560 || g.n_vocab_all!=248320 || g.world!=2 ||
            g.max_spec!=DLoopBlock::tree_budget()) {
            rad_note(b,"DLoop requires teacher width 2560, vocabulary 248320, TP2 and max_spec equal to DARTree budget");
            return RAD_E_INVAL;
        }
    }

    /* Qwen4-Exp's norms are all `x_hat * (1 + w)` -- `Qwen3NextRMSNorm.forward` is
     * `_norm(x.float()) * (1.0 + weight.float())` with the weight initialised to ZEROS -- and its
     * delta net's gated output norm is the plain `weight * x` with weights initialised to ONES.
     * That is read off the reference rather than inferred from the checkpoint: one answer, from
     * the source. */
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
    /* EVERY BLOCK'S ALL-REDUCE RIDES IN THE WRITE THAT FOLLOWS IT, wherever a kernel serves that
     * (HyperConn::Config::reduce): the block's partial sum is the write's input, so the pair
     * becomes one launch. The blocks are told below, per connection, whether it resolved. */
    m.hccfg.reduce = 1;
    const int64_t HN = m.hccfg.hc * g.n_embd;

    /* ---- partial rotary. 0.25 of head_dim 256 is 64, and the other 192 dims pass through. */
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
    /* THERE IS NO BRING-UP SLICE, and the rule is worth stating: if a model ever arrives too large
     * for the target device, the answer is a converter flag that writes a NAMED partial container
     * -- not an environment variable that makes a full one silently mean something else. A slice
     * declared at serve time is a server computing something that is not the model while looking
     * exactly like one that is. */

    /* ---- the layer schedule. */
    const int64_t interval = meta_int2(meta, "qwen4exp.full_attention_interval",
                                       "full_attention_interval", 0);
    if (interval <= 1) {
        fprintf(stderr, "radiance: qwen4exp: full_attention_interval is %lld; this is a hybrid "
                        "architecture and the interval decides which layers are linear\n",
                (long long)interval);
        return RAD_E_INVAL;
    }
    /* The container's `layer_types` names every layer of the WHOLE model, and the count check
     * inside is against the layer count passed here rather than against the array's length. */
    RAD_ARCH_TRY(check_layer_schedule(meta, g.n_layer, interval));

    /* ---- the delta net. */
    {
        GdnFP8::Config& gc = m.gcfg;
        gc.tree_state = tree_state;
        const int64_t nk = meta_int2(meta, "qwen4exp.ssm.key_head_count",
                                     "linear_num_key_heads", 0);
        const int64_t nv = meta_int2(meta, "qwen4exp.ssm.value_head_count",
                                     "linear_num_value_heads", 0);
        gc.head_k     = meta_int2(meta, "qwen4exp.ssm.key_head_dim",   "linear_key_head_dim",   0);
        gc.head_v     = meta_int2(meta, "qwen4exp.ssm.value_head_dim", "linear_value_head_dim", 0);
        gc.conv_width = meta_int2(meta, "qwen4exp.ssm.conv_kernel",    "linear_conv_kernel_dim", 0);
        gc.chunk      = 64;
        /* `output_gate_type` is a CHECKPOINT property: Qwen3.5/3.6 leave it unset and mean silu,
         * Qwen4-Exp states "sigmoid". libr4d's gated norm has both arms. */
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
        int s;
        if ((s = divide_or_fail(nk, g.world, "linear_num_key_heads",   &gc.n_head_k)) < 0) return s;
        if ((s = divide_or_fail(nv, g.world, "linear_num_value_heads", &gc.n_head_v)) < 0) return s;
    }

    /* ---- the routed feed-forward.
     *
     * n_ff is `moe_intermediate_size`: divided by world_size when the world divides it, and left
     * whole when it does not (geom_from's `arch_splits_ff`). Either way it is only the starting
     * point -- at more than one rank the split below decides the rank's width, by whole fp8
     * blocks and parity, which is what the header's tensor-parallel section is about. */
    {
        MoeFP8::Config& mc = m.moecfg;
        mc.n_expert    = meta_int2(meta, "qwen4exp.expert_count",      "num_experts", 0);
        mc.top_k       = meta_int2(meta, "qwen4exp.expert_used_count", "num_experts_per_tok", 0);
        mc.n_ff_exp    = g.n_ff;
        mc.n_ff_shared = meta_int2(meta, "qwen4exp.expert_shared_feed_forward_length",
                                   "shared_expert_intermediate_size", 0);
        /* THE SHARED ARM IS SPLIT BY WHOLE fp8 SCALE BLOCKS: evenly when its width halves into
         * them, and otherwise as evenly as whole blocks allow, cut by the rank's place
         * (shared_place) -- this family's 640 columns are five blocks, three on rank 0 and two on
         * rank 1; at four ranks two on rank 2 and one on each of the others. See
         * MoeFP8::Config::shared_lo. */
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
            const int64_t base = nb / g.world, extra = nb % g.world;
            const int64_t place = shared_place(g.n_ff_all, g.world, g.rank);
            const int64_t b0 = place * base + std::min<int64_t>(place, extra);
            const int64_t b1 = b0 + base + (place < extra ? 1 : 0);
            mc.shared_lo   = b0 * RAD_FP8_BLOCK;
            mc.shared_hi   = b1 * RAD_FP8_BLOCK;
            mc.n_ff_shared = mc.shared_hi - mc.shared_lo;
            rad_note(b, "the shared expert does not halve into %d-column fp8 scale blocks, so rank "
                        "%d computes columns [%lld, %lld) of it -- %lld of %lld blocks -- and the "
                        "block's all-reduce sums the ranks' partials.", RAD_FP8_BLOCK, g.rank,
                     (long long)mc.shared_lo, (long long)mc.shared_hi, (long long)(b1 - b0),
                     (long long)nb);
        }
        mc.norm_topk   = 1;
        /* THE GPTQ CALIBRATION RUN, gated by the same variable that says where to write: there is
         * no second switch, because a calibration with nowhere to put its answer is a 1.34 GB
         * allocation and two extra launches a layer for nothing. It is a separate run from
         * serving -- see MoeFP8::Config::calib. */
        mc.calib       = calib_dir() != nullptr ? 1 : 0;
        /* Tokens a pass. 2048 is the linear-state checkpoint interval this family's scheduler
         * clamps the prefill chunk to, so the shipped configuration is ONE pass and the slicing is
         * inert -- but it also sizes four buffers at `rows * top_k`, so a deployment whose
         * max_tok is smaller must not pay for rows it cannot present. */
        mc.rows        = g.max_tok < 2048 ? g.max_tok : 2048;
        if (mc.n_expert <= 0 || mc.top_k <= 0 || mc.top_k > mc.n_expert) {
            fprintf(stderr, "radiance: qwen4exp: the container describes %lld expert(s) with "
                            "top_k %lld\n", (long long)mc.n_expert, (long long)mc.top_k);
            return RAD_E_INVAL;
        }
        /* ---- THE EXPERTS' STORED FORMAT -- block fp8, four-bit, four-bit rotated, or the
         * checkpoint's plain bf16 -- read off one expert's encoding (MoeFP8::probe) before anything
         * is sized by it: how the experts are split between the ranks, the rotated copy of the
         * block input, the connection in front of every routed block, and -- for bf16 -- every
         * linear in the model. It is the container's, the checkpoint's, or the recipe's that
         * rad-convert is applying, so a container cannot be served in a format other than the one
         * it holds.
         *
         * THE EXPERT ASKED IS THE FIRST ONE THIS RANK OWNS UNDER EXPERT PARALLELISM. The answer
         * decides whether the experts are split that way, so it has to be asked before; and the
         * expert is declared on this rank either way -- whole under expert parallelism, a slice of
         * it under the parity split -- so the block skips exactly the map the probe made. */
        mc.probed_expert = (g.world > 1 && mc.n_expert % g.world == 0)
                               ? (int64_t)g.rank * (mc.n_expert / g.world) : 0;
        {
            MoeFP8::Src ps{};
            ps.experts_gate_up = nm.ckpt("model.language_model.layers.0.mlp.experts.gate_up_proj");
            ps.experts_down    = nm.ckpt("model.language_model.layers.0.mlp.experts.down_proj");
            RAD_ARCH_TRY(MoeFP8::probe(b, nm, 0, ps, &mc));
        }
        /* A CHECKPOINT SERVED AS IT SHIPS IS bf16 THROUGHOUT. Plain experts mean a container
         * converted without a quantising recipe, and every linear of it is then plain too: the
         * blocks issue bf16 GEMMs over the bf16 activation and nothing in front of them quantises
         * (Geom::w_bf16). Each linear checks its own encoding against this and refuses a quantised
         * one. The other way round is allowed: a quantised container may keep a linear plain --
         * the QSA indexer's projection -- and LinearFP8 reads that one as bf16. */
        g.w_bf16 = mc.expert_bf16 != 0;
        /* ...and the connections in front of the blocks emit no codes, since nothing reads any. */
        m.hccfg.quant = g.w_bf16 ? 0 : 1;
        /* ============================== EVERY RANK HOLDS A SLICE OF EVERY EXPERT ==============================
         *
         * An expert of this model is 640 columns, five 128-column blocks -- the fp8 scale block,
         * the 4-bit plane's scale group and the rotation's width all at once -- so two ranks
         * cannot halve one.
         *
         * THE SPLIT IS BY PARITY: each parity's five blocks are cut contiguously over the ranks,
         * and the rank that takes the extra block alternates with the expert's parity
         * (MoeFP8::Config::ff_lo says why by parity). At two ranks, rank r's slice of expert e is
         * the first three blocks when e % 2 == r and the last two otherwise. The router stays
         * replicated, every rank computes its slice of every routed expert, and the block's
         * all-reduce sums the partials.
         *
         * AT FOUR RANKS PARITY IS NOT ENOUGH. It gives the extra block only two places to go, so
         * ranks 0 and 1 would compute three blocks of every even/odd pair of experts and ranks 2
         * and 3 two, and the all-reduce would have ranks 2 and 3 wait for the other two. So at four
         * the extra block is rotated over four places, by the expert's number mod 4: four slice
         * tables (`moe_gemm_q_mod4`), and every rank computes five blocks of every four experts.
         * A layer keeping two experts plain bf16 has 510 quantised ones, and its first two classes
         * are one expert longer. What does not divide elsewhere stays apart: the plain experts'
         * extra tiles are ranks 0 and 1's (MoeFP8::declare) and the shared expert's extra block
         * rank 2's (shared_place).
         *
         * WHY NOT EXPERT PARALLELISM, which splits the EXPERTS: rank r owns [r*E/W, (r+1)*E/W)
         * whole. The bytes a rank reads are the same on average either way, but which rank a
         * routed expert lands on is a coin flip, and a verify step's forty routed slots leave one
         * rank about five experts ahead -- which the all-reduce turns into the other ranks waiting,
         * about a millisecond a step at one sequence. Split by parity the ranks differ by a fifth
         * of an expert per unit of the even/odd difference instead. Whole experts would also break
         * the protected experts' pairing: the recipe protects them in pairs so each rank's count of
         * quantised experts stays even, which the two parity tables need.
         *
         * A CALIBRATION RUN SPLITS THE EXPERTS INSTEAD, whole experts per rank. The down
         * projection's Hessian is over an expert's whole K, and a rank holding a slice of each
         * expert has only its slice's block of it -- the cross terms live on the other rank. Whole
         * experts give each rank the Hessian of every expert it holds, which is what the packer
         * sums (rad_arch.h's calib_dir). */
        mc.n_expert_all = mc.n_expert;
        /* AND bf16 EXPERTS ARE SPLIT THE SAME WAY, whole experts per rank: a slice of a bf16 expert
         * is two row ranges and a column range of the container, which the file tier cannot read
         * as one run, and nothing about bf16 needs the parity split's balance -- a bf16 model is
         * served to calibrate and to compare, not for its decode rate. */
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
                /* The uneven split is TWO SLICES, not two RANKS: each parity's blocks are cut
                 * contiguously across the ranks and the rank taking the odd block alternates with
                 * the parity, so every rank's slices are whole blocks and every block of every
                 * expert has exactly one owner. At two ranks this is the split the tree has always
                 * made, byte for byte; at four it gives widths of ceil(nb/W) and floor(nb/W).
                 * A rank must come away with at least one block of each parity -- an empty slice
                 * is a rank that computes nothing and still joins the all-reduce -- which needs
                 * the ranks to fit inside the two parities' blocks together. */
                const int64_t base = nb / g.world, extra = nb % g.world;
                for (int q = 0; q < 2; ++q) {
                    /* This rank's place in expert parity q's split: first for its own parity. */
                    const int64_t p = (g.rank - q + g.world) % g.world;
                    const int64_t b0 = p * base + std::min<int64_t>(p, extra);
                    const int64_t b1 = b0 + base + (p < extra ? 1 : 0);
                    mc.ff_lo[q] = b0 * RAD_FP8_BLOCK;
                    mc.ff_hi[q] = b1 * RAD_FP8_BLOCK;
                }
                mc.n_ff_exp = std::max(mc.ff_hi[0] - mc.ff_lo[0], mc.ff_hi[1] - mc.ff_lo[1]);
                /* AND AT FOUR RANKS FOUR CLASSES, by the expert's number mod 4 (MoeFP8::Config::
                 * ff_lo4), which every layer then takes instead: the same cut with the rank's place
                 * shifted by the class, so the extra block goes round all four ranks instead of
                 * two. Every block of every expert still has one owner, and each slice is one a
                 * rank already takes of some parity. */
                if (g.world == 4) {
                    for (int q = 0; q < 4; ++q) {
                        const int64_t p = (g.rank - q + 4) % 4;
                        const int64_t b0 = p * base + std::min<int64_t>(p, extra);
                        const int64_t b1 = b0 + base + (p < extra ? 1 : 0);
                        mc.ff_lo4[q] = b0 * RAD_FP8_BLOCK;
                        mc.ff_hi4[q] = b1 * RAD_FP8_BLOCK;
                        mc.n_ff_exp  = std::max(mc.n_ff_exp, mc.ff_hi4[q] - mc.ff_lo4[q]);
                    }
                    rad_note(b, "routed experts at four classes: rank %d computes columns [%lld, "
                                "%lld) [%lld, %lld) [%lld, %lld) [%lld, %lld) of the experts of "
                                "number 0, 1, 2 and 3 mod 4 -- %lld blocks of every four experts, "
                                "as every rank does.",
                             g.rank, (long long)mc.ff_lo4[0], (long long)mc.ff_hi4[0],
                             (long long)mc.ff_lo4[1], (long long)mc.ff_hi4[1],
                             (long long)mc.ff_lo4[2], (long long)mc.ff_hi4[2],
                             (long long)mc.ff_lo4[3], (long long)mc.ff_hi4[3], (long long)nb);
                } else {
                    rad_note(b, "routed experts: rank %d computes columns [%lld, %lld) of each "
                                "even expert and [%lld, %lld) of each odd one, of %lld. The width "
                                "is %lld blocks of %d, which %lld ranks split unevenly, so the rank "
                                "taking the extra block alternates with the expert's parity; the "
                                "block's all-reduce sums the partials.",
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


    /* ---- the n-gram embedding's layer and the indexer's geometry, each stated in a note where it
     * changes what this build computes. */
    {
        const char* ple = rad_meta_gets(meta, "text_config.ple_layer_ids",
                          rad_meta_gets(meta, "ple_layer_ids", nullptr));
        if (ple && *ple) {
            /* ONE-INDEXED, and the whole layer hangs off getting that right: `ple_layer_ids: [2]`
             * is layer_idx 1. The reference says so twice -- `config.ple_layer_ids.index(layer_idx
             * + 1)` at construction and a [1, num_hidden_layers] bound in the validator. */
            m.ple_layer = (int64_t)std::strtoll(ple, nullptr, 10) - 1;
            if (m.ple_layer < 0 || m.ple_layer >= g.n_layer) {
                /* Out of the declared range, which a DECL_LAYERS slice makes ordinary: the layer
                 * simply is not in this build. Said rather than skipped, because a slice that
                 * silently drops PLE looks exactly like one that has it. */
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
        m.qsacfg.tree_state = tree_state;
        m.qsacfg.topk       = (ratio > 0) ? budget / ratio : 0;
        m.qsacfg.max_blocks = (ratio > 0) ? (g.max_ctx + ratio - 1) / ratio : 0;
        m.qsacfg.max_work   = (ratio > 0) ? (g.max_tok + ratio - 1) / ratio + g.max_seqs : 0;
        /* The tail's ring, and QsaIndexer::Config::ring is where the arithmetic is argued. It is
         * `max_spec` and not `n_spec`: a deployment that can speculate three deep sizes for
         * three, whatever a given step turns out to carry. */
        m.qsacfg.ring       = ratio + g.max_spec;
        if (budget > 0 && ratio > 0) {
            /* THE EXACTNESS BOUND, and it is the whole reason dense attention is not an
             * approximation here: with `pos + 1 <= topk*ratio + ratio - 1` tokens visible, the number
             * of COMPLETE blocks is at most budget/ratio, the indexer's top-k keeps all of them,
             * and the tail is attended unconditionally -- so the selected set is every visible
             * token and QSA is dense causal attention bit for bit. */
            const int64_t exact_to = m.qsacfg.exact_to();
            /* THERE IS NO CAP ON `--max-ctx` HERE. Prefill selects per query exactly as decode
             * does (arch/common/rad_qsa.h, docs/QSA.md), so above the bound this runs the model's
             * own attention rather than a denser stand-in -- and there is no force-dense switch,
             * because above the bound the dense path is NOT the trained attention and such a
             * switch would make a server quietly serve a different model. */
            if (g.max_ctx > exact_to)
                rad_note(b, "QSA: sparse past %lld tokens of context -- %lld blocks of %lld a "
                            "query, selected per query at prefill and at decode. At or below that "
                            "the selection is every visible block and the engine issues the "
                            "cheaper dense attention, which is the same answer to the bit.",
                         (long long)exact_to, (long long)(budget / ratio), (long long)ratio);
            else
                rad_note(b, "QSA: dense attention is EXACT to %lld tokens of context and "
                            "--max-ctx is %lld, so the indexer selects every visible block "
                            "and changes nothing.", (long long)exact_to, (long long)g.max_ctx);
        }
    }

    rad_note(b, "gated residual: %lld streams of %lld (%lld wide), low-rank gate %lld; %lld "
                "connections plus the mixer",
             (long long)m.hccfg.hc, (long long)g.n_embd, (long long)HN,
             (long long)m.hccfg.lowrank, (long long)(2 * g.n_layer));

    /* ---- the KV groups. Same three as qwen35_fp8 and for the same reasons. */
    {
        RadKVGroupDecl fd{};
        fd.kind      = RAD_KV_FULL;
        fd.dtype     = rad_kv_cache_dtype(ctx, RAD_BF16);
        fd.n_head_kv = g.n_head_kv;
        fd.head_dim  = g.head_dim;
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_attn"), &fd);
        g.kv_dtype = rad_dtype_name(fd.dtype);

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

        /* ---- the QSA indexer's TAIL, which is state and not a buffer.
         *
         * It carries an unfinished block's raw index keys across steps, so it has to be indexed
         * by the SEQUENCE and by the LAYER -- and a buffer can express neither. The batch row is
         * step-local and the keys are per-layer; arch/common/rad_qsa.h's Wire is the full
         * argument. As a linear group the engine keeps a slot stable for the life of a sequence,
         * zeroes it when it hands one out, and saves and restores it with the rest of the linear
         * state when the prefix cache checkpoints -- which is also the only thing that makes a
         * resume onto a half-built block read anything defined.
         *
         * `n_head_kv` 1 and `state_dim` {ring, head_dim}: one state is `ring` raw keys, a few
         * kilobytes a layer a sequence. */
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

    /* ---- THE TRUNK'S STORED FORMAT, read off the first delta-net layer's input projection the way
     * the experts' is read off one expert: the activation buffers below carry an int8 code pair
     * only when the trunk's linears are int8 (LinearFP8::i8), so it has to be known before they
     * are declared. The probe maps that weight through the layer's own LinearFP8, which then does
     * not map it again. */
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
    RAD_ARCH_TRY(m.a_x.declare_qs(b, nm, g, "x", g.n_embd, 0, m.trunk_i8));
    /* THE CONNECTIONS WRITE THE INT8 CODES THEMSELVES when every reader of their codes is an int8
     * linear: the trunk's are, and the routed experts read the rotated twin (or the bf16 plane),
     * never these. Experts that read the plain E4M3 codes keep them, and the int8 linears then
     * quantise for themselves. */
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
    /* THE QSA INDEXER'S TWO PLANES, shared by every full-attention layer the way `attn_out` is.
     * Declared only when the checkpoint carries an indexer, so a container without one allocates
     * nothing and the layer's declare() is a no-op. */
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
        /* ONE ROW A QUERY, not one a sequence -- QSA Stage B, arch/common/rad_qsa.h. The score
         * plane is the only buffer in this model whose size is a PRODUCT of the two deployment
         * bounds, so it is refused here by name rather than discovered as an allocation failure
         * halfway through the arena: 2 GiB is already an absurd answer and the message says which
         * two flags produced it. */
        const int64_t sc_elems = g.max_tok * m.qsacfg.max_blocks;
        if (sc_elems > (int64_t)512 * 1024 * 1024) {
            fprintf(stderr,
                "radiance: qwen4exp: the QSA score plane would be %lld MiB -- "
                "--max-num-batched-tokens %lld by --max-ctx %lld / compress ratio %lld. "
                "Lower either.\n",
                (long long)(sc_elems * 4 / (1024 * 1024)), (long long)g.max_tok,
                (long long)g.max_ctx, (long long)m.qsacfg.ratio);
            return RAD_E_UNSUPPORTED;
        }
        m.qsaw.score = decl_b(b, nm.f("qsa_score"), RAD_F32,
                              { g.max_tok, m.qsacfg.max_blocks });
        m.qsaw.sel   = decl_b(b, nm.f("qsa_sel"),   RAD_I32,
                              { g.max_tok, m.qsacfg.topk + 1 });
        m.qsaw.nsel  = decl_b(b, nm.f("qsa_nsel"),  RAD_I32, { g.max_tok });
        m.qsaw.sequ  = decl_b(b, nm.f("qsa_sequ"),  RAD_I32, { g.max_tok });
        if (g.world == 2) {
            m.qsaw.sel_shard  = decl_b(b, nm.f("qsa_sel_shard"),  RAD_I32,
                                       { g.max_tok / 2, m.qsacfg.topk + 1 });
            m.qsaw.sequ_shard = decl_b(b, nm.f("qsa_sequ_shard"), RAD_I32,
                                       { g.max_tok / 2 });
            if (!m.qsaw.sel_shard || !m.qsaw.sequ_shard)
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
    /* F32 AND NOT THE ACTIVATION DTYPE, both of them, and the kernels say so rather than tolerate
     * it: `gdn_conv_prep` ends `R4D_DT(g, RAD_F32); R4D_DT(beta, RAD_F32)`. The log decay is a
     * cumulative sum that the chunked scan exponentiates differences of, and beta is its
     * companion; bf16's eight mantissa bits are not enough for either. The refusal is
     * "dtype this kernel does not serve" at the first step, which is the right failure and not an
     * obvious one to read. */
    m.b_gdec = decl_b(b, nm.f("gdn_gdec"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_beta = decl_b(b, nm.f("gdn_beta"), RAD_F32,  {T, m.gcfg.n_head_v});
    m.b_kkt  = decl_b(b, nm.f("gdn_kkt"),  RAD_BF16, {T, m.gcfg.n_head_v, m.gcfg.chunk});
    m.a_go.x = decl_b(b, nm.f("gdn_o"),    g.act_dtype, {T, m.gcfg.n_head_v, m.gcfg.head_v});
    RAD_ARCH_TRY(m.a_go.declare_qs(b, nm, g, "gdn_o", v_dim, 0, m.trunk_i8));
    /* So does the delta net's output epilogue, and its prefill quantiser is the int8 one. */
    m.a_go.q8_fed = m.trunk_i8;

    /* THE MODEL'S expert count, not this rank's: the router is replicated under expert
     * parallelism and scores every expert. */
    m.b_rlogits = decl_b(b, nm.f("moe_logits"), g.act_dtype, {T, m.moecfg.n_expert_all});
    m.b_eids    = decl_b(b, nm.f("moe_ids"),    RAD_I32,     {T, m.moecfg.top_k});
    m.b_ew      = decl_b(b, nm.f("moe_ew"),     g.act_dtype, {T, m.moecfg.top_k});
    m.b_eoff    = decl_b(b, nm.f("moe_eoff"),   RAD_I32,     {m.moecfg.n_expert + 1});
    /* PERSIST, NOT TRANSIENT, and it is the one buffer here that has to be. `moe_scatter` writes
     * the per-expert histogram and NO DECLARED OP READS IT -- the reader is rad_route_report,
     * which is a handoff and not an op -- so liveness analysis sees it die the instant it is
     * written and is free to alias its storage with anything later in the step. Declared
     * transient, a wide enough prefill chunk gives the planner something to lay over it and the
     * histogram comes back holding values a sum of placements cannot be -- negative counts. A
     * small step may miss it entirely, so this is not a fault a short run reveals.
     *
     * The buffer is one int32 an expert. */
    m.b_ecnt    = decl_b(b, nm.f("moe_ecnt"),   RAD_I32,     {m.moecfg.n_expert},
                         RAD_BUF_PERSIST);
    m.b_sorted  = decl_b(b, nm.f("moe_sorted"), RAD_I32,     {ER});
    m.b_egu     = decl_b(b, nm.f("moe_egu"),    g.act_dtype, {ER, 2 * m.moecfg.n_ff_exp});
    if (m.moecfg.expert_bf16)
        m.a_eff.x = decl_b(b, nm.f("moe_eff"),    g.act_dtype, {ER, m.moecfg.n_ff_exp});
    /* ER ROWS, NOT max_tok: this is an EXPERT plane, one row per (token, slot). Taking
     * declare_qs's default declares it `top_k` times too short, and the gated quantiser then
     * writes that many times past the end of it -- see ActFP8::declare_qs, which is where the
     * guard and the argument live. */
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
        /* And the shared expert's gated quantiser in front of its int8 down projection. */
        m.a_sff.q8_fed = m.trunk_i8;
        m.b_sout  = decl_b(b, nm.f("moe_sout"),  g.act_dtype, {T, g.n_embd});
        m.b_sgate = decl_b(b, nm.f("moe_sgate"), g.act_dtype, {T, 1});
    }

    m.b_hout   = decl_b(b, nm.f("hout"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
    m.b_logits = decl_b(b, nm.f("logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
    RAD_ARCH_TRY(rad_declare_logits(b, m.b_logits));

    /* ---- the vocabulary edges, both row-sharded as qwen35_fp8 has them. `lm_head.weight` is
     * present in this checkpoint (tie_word_embeddings is false), but the alternative spelling
     * costs nothing and a smaller member of the family may tie them.
     *
     * THE HEAD IS DECLARED OVER EVERY PLANE IT HAS, at the codes' dtype: bf16 as the checkpoint
     * ships it, block fp8 where a recipe makes it so. Which `logits_gemm` kernel serves it is
     * then the selector's -- a kernel whose layout does not read the head's encoding is not a
     * candidate -- so one declaration serves both. The maps come first: they are how the model
     * is asked what the head is. */
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
    /* THE RANK'S ROWS, STATED: a vocabulary that does not divide by the world (248320 at three
     * ranks) is split by rad_arch.h's rule -- ceil to a 128-row block, last rank short -- and that
     * is a span, not the even share the core would otherwise cut. Harmless where it does divide
     * (the span IS the even share). */
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
    /* THE ENCODER'S ROWS, over the lookup and before its reduction (MediaRows says why). */
    m.media = VisionTower::present(meta) && ctx->max_enc_patches > 0;
    if (m.media) RAD_ARCH_TRY(m.mrows.declare(b, T, g.n_embd, m.a_x.x));
    /* EXACT unconditionally: every other collective here sums partial products, this one sums
     * DISJOINT rows -- a token's embedding comes from exactly one rank. qwen35_fp8 says the same. */
    if (g.world > 1)
        m.op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                              RAD_PARAMS(RAD_INT("world_size", g.world),
                                         RAD_RANGE("numel", g.n_embd, T * g.n_embd),
                                         RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                              RAD_NOWEIGHTS),
                          {m.a_x.x}, {m.a_x.x});

    /* THE STREAM'S FIRST VALUE: hc identical copies of the embedding. */
    RAD_ARCH_TRY(m.enter.declare(b, g, m.hccfg.hc, m.a_x.x, m.b_h));

    /* ---- the layers.
     *
     * DECLARATION ORDER IS WHAT THE BUFFER PLANNER'S LIVENESS IS: a transient's lifetime is
     * [first definition, last use] over OP INDICES in this order, and two transients whose ranges
     * are disjoint MAY share bytes (core/build/rad_bufplan.cpp). Every buffer above is shared by
     * every layer, so the order below has to be the order step() issues in -- read, block, write,
     * read, block, write -- and it is. A connection declared out of that order would make a false
     * statement about liveness and the planner would alias two buffers that are both live inside
     * one step; the symptom is fluent output with one arm of the model reading the other's
     * leftovers, which nothing downstream flags.
     */
    m.op_rope_cs = rw(b, RAD_OP(b, "rope_table",
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
    /* AND THE INDEXER READS THE SAME PLANE. Its rotation is at `m.n_rot`, `g.theta` and
     * `g.rope_scale` -- the three the table was built from -- so the rows this step's positions
     * name are already written, and the twelve trunk layers plus the head's four passes stop
     * issuing `heads` norms and a rope each. Set HERE rather than where `m.qsaw`'s buffers are
     * declared because the table is declared after them, and `tw.qsa = m.qsaw` below copies this
     * wire to the MTP head, which is how the head's passes get it too. */
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
    for (int64_t l = 0; l < g.n_layer; ++l) {
        Layer& lay = m.layers[(size_t)l];
        lay.full = is_full_attention(l, interval) ? 1 : 0;

        /* PLE COMES FIRST, BEFORE THE ATTENTION CONNECTION READS THE STREAM, because the reference
         * adds it into `hidden_states` at the top of the decoder layer's forward -- so the gated
         * residual that follows normalises a stream the n-gram features are already in.
         *
         * DECLARATION ORDER IS THE BUFFER PLANNER'S LIVENESS, so this has to be here and not
         * hoisted out of the loop: every buffer the block owns is live from here to its `add`, and
         * declaring it earlier would overstate that and cost the planner a sharing it could have
         * had. The same rule the connections below follow. */
        if (l == m.ple_layer) {
            PleLayer::Config pc{};
            pc.tree_state = tree_state;
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
            /* THE DILATION IS THE N-GRAM SIZE. `conv_dilation = config.ngram_size` in the
             * reference's PLE layer -- not a separate key, and not 1. */
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

        /* `Src::norm = nullptr` is the blocks' "the input is already normed and quantised" mode,
         * and `add_out = false` stops them adding into the stream -- the write half of the
         * connection does that, with the four gains the read produced. `fold` and `ar_in` are both
         * off for the same reason: both fuse into a norm this block does not have. */
        if (lay.full) {
            AttnGatedFP8::Src as{};
            as.qg     = nm.ckpt("%s.%lld.self_attn.q_proj.weight", pre, (long long)l);
            as.k      = nm.ckpt("%s.%lld.self_attn.k_proj.weight", pre, (long long)l);
            as.v      = nm.ckpt("%s.%lld.self_attn.v_proj.weight", pre, (long long)l);
            as.q_norm = nm.ckpt("%s.%lld.self_attn.q_norm.weight", pre, (long long)l);
            as.k_norm = nm.ckpt("%s.%lld.self_attn.k_norm.weight", pre, (long long)l);
            as.o      = nm.ckpt("%s.%lld.self_attn.o_proj.weight", pre, (long long)l);
            /* THE KV PAGE IS THE QSA COMPRESS BLOCK when that is asked for. A page of 4 makes the
             * indexer's selected set a paged block table rather than a new attention kernel
             * (docs/QSA.md); the attention kernel's rows have to declare the same 4 or nothing
             * resolves, which is what keeps this honest.
             *
             * A page size is pure storage, so the only thing it can move is the block-table walk:
             * a page of 4 and a page of 16 cost the same at long context and produce identical
             * text. */
            lay.attn.kv_block = 4;
            lay.attn.cfg.tree_state = tree_state;
            /* THE SELECTION IS SET BEFORE THE ATTENTION IS DECLARED, and the order is
             * load-bearing: `attn_paged` declares `sel` and `seqused` as READS, and a read
             * declared after the op is not a read at all -- see rad_block_attn_gated_fp8.h. The
             * indexer itself is declared after the attention block (its ops have to sit where the
             * sparse attention reads them), which is why the two halves of this are apart.
             *
             * Below the indexer's budget the selection is every visible block, so the sparse
             * and dense paths must produce identical text and a difference is a defect rather
             * than a model change. */
            if (m.qsaw.qk) {
                lay.attn.qsa_sel      = m.qsaw.sel;
                lay.attn.qsa_sequ     = m.qsaw.sequ;
                lay.attn.qsa_topk     = m.qsacfg.topk;
                /* THE BOUND, so the attention keeps the cheaper dense issue where the two paths
                 * are the same answer. Decode ignores it and always takes the sparse path, which
                 * is what keeps that path exercised on every step this engine runs. */
                lay.attn.qsa_exact_to = m.qsacfg.exact_to();
            }
            RAD_ARCH_TRY(rad_bind_layer_kv(b, (int)l, m.kv_full));
            lay.attn.ar_out_take = lay.hc_mix.ar_take();
            RAD_ARCH_TRY(lay.attn.declare(b, nm, g, (int)l, m.kv_full, m.n_rot, aw, as,
                                          false, false, kArNone, lay.hc_mix.op_ar_write != 0));
            /* THE INDEXER, after the attention block so its ops sit where the sparse attention
             * reads them. It reads the same normed fp8 input the projections do; arch/common/
             * rad_qsa.h is the declaration side of docs/QSA.md. */
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
        ms.experts_gate_up  = nm.ckpt("%s.%lld.mlp.experts.gate_up_proj", pre, (long long)l);
        ms.experts_down     = nm.ckpt("%s.%lld.mlp.experts.down_proj",    pre, (long long)l);
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
    }

    /* ---- the 97th connection. No write half, no fp8 twin: it produces the one hidden state the
     * lm_head reads, as a bf16 input. Its `hc_norm` is what a pre-norm model calls the
     * final norm -- this checkpoint has no `model.language_model.norm.weight` at all. */
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

    /* ---- THE DFLASH2 DRAFTER AND ITS TAPS.
     *
     * GATED ON THE DRAFTER BEING IN THE CONTAINER AND ON THE DEPLOYMENT MEANING TO SPECULATE: a
     * container that never drafts should not carry the graph, and `max_spec` is the engine already
     * saying whether it intends to.
     *
     * THE DRAFTER IS DECLARED WHOLE -- five layers of its own, its own bf16 linears, its own
     * windowed KV group and the target's own bf16 lm_head, vocab-sharded by the block's existing
     * merge. ITS INPUT IS THE CAPTURE'S OWN REDUCTION AND NOT A SUBSTITUTE: `hc_mean` over the
     * four hyper-connection streams of a COMPLETED layer's output
     * (`hidden_states.unflatten(-1, (hc, n)).mean(dim=-2)`, 2560 wide a tap). NOT `hc_read` --
     * that one is a GATED, per-stream-RMS-normalised read of a block INPUT -- and no scaling is
     * folded into `fc` or `hidden_norm`, which would be exact only up to `eps` in a drafter
     * served unquantised on purpose. */
    const char* fd_env = std::getenv("RADIANCE_FULLDRAFTER");
    const char* fd_sidecar = std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
    const int64_t fd_meta = meta_int2(meta, "qwen4exp.fulldrafter", "fulldrafter", 0);
    const bool use_fulldrafter = (fd_env && *fd_env && strcmp(fd_env, "0") != 0) ||
                                 (fd_sidecar && *fd_sidecar) ||
                                 (fd_meta > 0);

    const bool want_df = wants_dflash2(meta, ctx);
    if (want_df && use_fulldrafter) {
        rad_note(b, "DFlash2 and the FullDrafter drafter cannot be declared together: the engine "
                    "drives ONE drafter, and the two are different models with different taps. "
                    "Unset RADIANCE_FULLDRAFTER to serve the container's DFlash2.");
        return RAD_E_INVAL;
    }
    if (want_df) {
        /* `m.n_ktap` IS NOT PARSED YET at this point -- the recording taps are read below -- so the
         * test is the environment variable that fills it. Two taps with different meanings on one
         * serve is exactly the confusion this fence exists to prevent. */
        const char* kt = std::getenv("RAD_KLD_TAPS");
        if ((kt && *kt) || use_dloop) {
            rad_note(b, "the DFlash2 drafter and the recording taps / DLoop drafter do not share "
                        "a serve: the engine drives ONE drafter and its taps are a different "
                        "capture. Drop RAD_KLD_TAPS / RADIANCE_DLOOP_DRAFTER.");
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(declare_df(b, meta, ctx, m, (int)g.n_layer));
        /* THE TAP PLANE IS `declare_df`'S: it declares `dflash_src` and wires it before the block
         * reads it. Nothing is declared for it here -- a second `rad_decl_buffer` under the same
         * name would be a second arena plane the graph never reads, and declaring it here passed
         * the block a handle of 0. */

        /* THE TAP ITSELF: `hc_mean`, ONE LAUNCH A TAP, after that layer's completed output.
         *
         * It is the capture's own arithmetic -- `hidden_states.unflatten(-1,
         * (hc, n)).mean(dim=-2)` -- and nothing else: no RMS, no gate, no read of a block input,
         * and no scaling folded into `fc` or `hidden_norm`. Sum in f32 over the hc streams, divide
         * by hc, store bf16 once.
         *
         * THE OUTPUT IS A COLUMN VIEW of the tap plane: width n_embd at column t*n_embd with the
         * declared row pitch of n_taps*n_embd kept as the stride (bcol, rad_arch.h). That is why
         * the op must write through a row pitch rather than assume a packed [M, n_embd].
         *
         * IT IS ALSO `mine`'S SCHEMA ALONE, AND THIS CALL IS UNCONDITIONAL. libr4d, libref and
         * libavx publish no `hc_mean` (mine/hc_mean.cpp is its only declaration), so a kernel home
         * that loads libr4d without `mine` fails THIS declare with RAD_E_NOSCHEMA -- there is no
         * fallback and no probe, for the reason rad_block_dflash2.h's Config states. That is the
         * established prerequisite `dc.mine_fusions` above rests on: this caller already needs
         * the plugin for the tap plane, so the DFlash2 fusion it enables adds no new dependency. */
        for (int t = 0; t < m.df_n_taps; ++t) {
            m.op_df_tap[t] = rw(b, RAD_OP(b, "hc_mean",
                                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                                RAD_INT("N", g.n_embd),
                                                RAD_INT("hc", m.hccfg.hc),
                                                RAD_STR("dtype", g.dtype)),
                                     RAD_NOWEIGHTS),
                                 {m.b_h}, {m.b_dfsrc});
            if (!m.op_df_tap[t]) return RAD_E_INVAL;
        }
        rad_note(b, "the DFlash2 taps are produced by `hc_mean` over %lld hyper-connection "
                    "streams, one launch a tap, into the column view of `dflash_src` at that "
                    "tap's %lld columns",
                 (long long)m.hccfg.hc, (long long)g.n_embd);
    }

    /* ---- the recording taps (Model::n_ktap). After op_logits, so the gathers' reads of `a_x`
     * extend its life to the end of the step rather than shortening anything; their outputs are
     * persistent and outside the planner's liveness altogether. */
    const char* tl = std::getenv("RAD_KLD_TAPS");
    if (!tl || !*tl) {
        const char* tt = std::getenv("RADIANCE_TEACHER_TAPS");
        if (use_fulldrafter || use_dloop || (tt && *tt)) {
            tl = g.n_layer >= 24 ? "1,23" : (g.n_layer >= 2 ? "1" : "");
        }
    }
    if (tl && *tl) {
        for (const char* p = tl; *p;) {
            while (*p == ',') ++p;
            if (!*p) break;
            char* e = nullptr;
            const long l = std::strtol(p, &e, 10);
            if (e == p || l < 0 || l >= g.n_layer || m.n_ktap >= 8) {
                fprintf(stderr, "radiance: qwen4exp: RAD_KLD_TAPS '%s' names a layer outside "
                                "0..%lld or more than 8 of them\n", tl, (long long)g.n_layer - 1);
                return RAD_E_INVAL;
            }
            m.ktap_layer[m.n_ktap++] = (int)l;
            p = e;
        }
        if (g.act_dtype != RAD_BF16) return RAD_E_UNSUPPORTED;   /* the recording is bf16 */
        /* ONE ROW A SAMPLED POSITION, GATHERED THROUGH `out_ids`, exactly as before. The DLoop
         * drafter does NOT read these: its persistent cache needs a row per VERIFIED TOKEN of
         * every chunk, including the middle prefill chunks that sample nothing and therefore have
         * an empty `out_ids`. It declares and fills its own trunk-width planes below. */
        for (int k = 0; k <= m.n_ktap; ++k) {
            m.b_ktap[k] = decl_b(b, nm.f("kld_tap%d", k), g.act_dtype,
                                 {g.max_logit_rows, g.n_embd}, RAD_BUF_PERSIST);
            m.op_ktap[k] = rw(b, RAD_OP(b, "gather_rows",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                             RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {m.a_x.x}, {m.b_ktap[k]});
            if (!m.b_ktap[k] || !m.op_ktap[k]) return RAD_E_INVAL;
        }
        if (!use_dloop) {
            m.b_ktap_anchor = decl_b(b, nm.f("kld_tap_anchor"), g.act_dtype,
                                     {g.max_logit_rows, g.n_embd}, RAD_BUF_PERSIST);
            m.op_ktap_anchor = rw(b, RAD_OP(b, "gather_rows",
                                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                                 RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                                      RAD_NOWEIGHTS),
                                  {m.a_x.x}, {m.b_ktap_anchor});
            if (!m.b_ktap_anchor || !m.op_ktap_anchor) return RAD_E_INVAL;
        }
    }

    /* ---- the multi-token-prediction head (arch/common/rad_block_mtp_hc.h).
     *
     * AFTER op_logits, and that is load-bearing rather than tidy. Declaration order IS the buffer
     * planner's liveness, so a drafter declared before the trunk's tail ops would make the planner
     * read those ops as living AFTER the drafter is done -- and share a trunk buffer with one of
     * the head's. The symptom is fluent output with acceptance EXACTLY zero; arch/qwen35_fp8
     * declares its own drafter in this same position for the same reason.
     *
     * DECLARED ONLY WHEN THE DEPLOYMENT SPECULATES. `mtp_num_hidden_layers` says the checkpoint
     * carries a head and `max_spec` says the engine intends to use one; at max_spec 0 the weights
     * stay off the card and the container still carries them, because rad-convert declares at a
     * non-zero max_spec on purpose. */
    const int64_t nextn = meta_int2(meta, "qwen4exp.mtp_num_hidden_layers",
                                    "mtp_num_hidden_layers", 0);
    if (!rad::spechub_config_valid()) {
        rad_note(b, "SpecHub accepts RADIANCE_SPECHUB=0|1 (or unset), and "
                    "RADIANCE_SPECHUB_VERIFY=transport|match only with SpecHub enabled");
        return RAD_E_INVAL;
    }
    const bool spechub = rad::spechub_enabled();
    if (spechub && (nextn <= 0 || g.max_spec != 2 || !dartree || !tree_state ||
                    use_fulldrafter || use_dloop || m.have_df || g.world != 2 ||
                    !vocab_head_sharded(g))) {
        rad_note(b, "SpecHub requires Qwen MTP TP2, max_spec=2, DARTREE and TREE_STATE; "
                    "it is one position with two siblings, not MTP depth5");
        return RAD_E_UNSUPPORTED;
    }
    if (!rad::mtp_coupled_config_valid()) {
        rad_note(b, "RADIANCE_MTP_COUPLED accepts 0|1 (or unset) and excludes RADIANCE_SPECHUB");
        return RAD_E_INVAL;
    }
    const bool coupled = rad::mtp_coupled_enabled();
    if (coupled && (nextn <= 0 || g.max_spec < 1 || g.max_spec > RAD_MTP_COUPLE_MAX_ROUNDS ||
                    dartree || use_fulldrafter || use_dloop || m.have_df ||
                    (g.world > 1 && !vocab_head_sharded(g)))) {
        rad_note(b, "RADIANCE_MTP_COUPLED requires the linear builtin MTP chain (no DARTREE, DLoop, "
                    "FullDrafter or DFlash2), max_spec 1..%d, and a reranked draft head (world 1 or "
                    "a vocab-sharded head)", RAD_MTP_COUPLE_MAX_ROUNDS);
        return RAD_E_UNSUPPORTED;
    }
    if (nextn > 0 && g.max_spec > 0 && !use_fulldrafter && !use_dloop && !m.have_df) {
        const rad::MtpTreeConfig& tree_config = rad::mtp_tree_config();
        if (tree_config.error) {
            rad_note(b, "%s", tree_config.error);
            return RAD_E_INVAL;
        }
        if (spechub && (tree_config.policy != 0 || tree_config.rounds != 3)) {
            rad_note(b, "SpecHub cannot combine with MTP score/cost/pruned or five-round policies");
            return RAD_E_UNSUPPORTED;
        }
        const bool pruned_tree = tree_config.pruned();
        const bool scored_tree = !pruned_tree &&
            (tree_config.policy != 0 || tree_config.rounds == 5);
        if ((scored_tree || pruned_tree) && (!dartree || g.world != 2 ||
            !vocab_head_sharded(g) || g.max_spec > 7 || g.max_spec < tree_config.rounds ||
            (pruned_tree && (g.max_spec != 7 || g.max_tok < 8)))) {
            rad_note(b, "MTP scored/pruned tree requires RADIANCE_DARTREE, TP2 sharded "
                        "vocabulary; pruned additionally requires capacity7 and depth5");
            return RAD_E_UNSUPPORTED;
        }
        /* Fixed MTP8 tree: 8 serial rounds, spine + sibling-deduplicated alternates through the
         * base TP2 merge accumulator; up to 64 proposals + root = 65 packed nodes. */
        const bool wide_tree = tree_config.rounds == 8;
        if (wide_tree && (!dartree || g.max_spec < 8 || g.max_spec > 64)) {
            rad_note(b, "RADIANCE_MTP_TREE_ROUNDS=8 requires RADIANCE_DARTREE and max_spec 8..64");
            return RAD_E_UNSUPPORTED;
        }
        const char* mpre = "mtp";
        /* AT max_tok, NOT max_logit_rows. A draft round is one row a sequence, but the history
         * pass that fills the head's own K and V is one row a TOKEN over a whole prefill chunk. */
        m.b_mtp_hin  = decl_b(b, nm.f("mtp_hin"),  g.act_dtype, {T, HN});
        m.b_mtp_hw   = decl_b(b, nm.f("mtp_hw"),   g.act_dtype, {T, HN});
        m.b_mtp_e    = decl_b(b, nm.f("mtp_e"),    g.act_dtype, {T, g.n_embd});
        m.b_mtp_hout = decl_b(b, nm.f("mtp_hout"), g.act_dtype, {g.max_logit_rows, g.n_embd});
        /* THE 2-BIT DRAFT HEAD, when the model holds one: asked once, here, because the answer
         * declares the head's buffers and the block reads it back from them. */
        if (mtp_draft_head_q2(b, MtpHcBlock::head_name(nm), nm.ckpt("lm_head.weight"))) {
            const int64_t dmpad = ((g.max_logit_rows + 15) / 16) * 16;
            m.b_mtp_dq   = decl_b(b, nm.f("mtp_dq"),   RAD_I8,  {g.max_logit_rows, g.n_embd});
            m.b_mtp_ds   = decl_b(b, nm.f("mtp_ds"),   RAD_F32, {g.max_logit_rows, 1});
            m.b_mtp_dsum = decl_b(b, nm.f("mtp_dsum"), RAD_F32, {g.n_embd / 128, dmpad});
            /* AT THIS RANK'S ROWS, not the model's: the 2-bit head is vocab-parallel and the
             * merge below puts the halves back together. rad_arch.h owns the rule. */
            m.b_mtp_dlog = decl_b(b, nm.f("mtp_dlog"), g.act_dtype,
                                  {g.max_logit_rows, vocab_head_rows(g)});
            if (vocab_head_sharded(g)) {
                m.b_mtp_dpair = decl_b(b, nm.f("mtp_dpair"), RAD_F32,
                                       {g.max_logit_rows, kDraftTopR, 2});
                m.b_mtp_dgath = decl_b(b, nm.f("mtp_dgath"), RAD_F32,
                                       {g.max_logit_rows, (int64_t)g.world * kDraftTopR, 2});
            }
            /* NAMED and declared, because the engine reads the proposal straight out of it: a
             * draft round's token comes from row_topk and never through the sampler. */
            /* kDraftTopR COLUMNS, and a draft reads one. The rest are the coarse head's other
             * candidates, which `logit_rerank` scores before column 0 is taken; SD02 gathers and
             * merges only the top kDraftGatherR candidates (16-byte row for ar_gather_2rank). */
            m.b_mtp_didx = decl_b(b, nm.f("mtp_didx"), RAD_I32, {g.max_logit_rows, kDraftTopR});
            m.b_mtp_dval = decl_b(b, nm.f("mtp_dval"), RAD_F32, {g.max_logit_rows, kDraftTopR});
        }
        /* The device accumulator preserves each round's candidates and rebuilds a packed tree
         * with the complete autoregressive spine before its sibling-deduplicated leaves. */
        if (dartree && !m.b_mtp_didx) {
            fprintf(stderr, "radiance: qwen4exp: RADIANCE_DARTREE wants a bounded tree from the "
                            "MTP head and this container's `mtp.draft_head.weight` is not the 2-bit "
                            "candidate head. A serial drafter's tree is the alternate candidates "
                            "each round left beside its pick, and the plain bf16 head leaves none "
                            "-- unset RADIANCE_DARTREE.\n");
            return RAD_E_UNSUPPORTED;
        }
        const int64_t tnodes = std::min<int64_t>(wide_tree ? 65 : 16, (int64_t)g.max_spec + 1);
        rad_buf log_hist = 0, lse_local = 0, lse_gathered = 0;
        rad_buf spechub_draws = 0, spechub_state = 0;
        if (spechub) {
            spechub_draws = decl_b(b, nm.f("mtp_spechub_draws"), RAD_U32,
                                  {g.max_seqs, RAD_SPECHUB_DRAW_WORDS}, RAD_BUF_PERSIST);
            spechub_state = decl_b(b, nm.f("mtp_spechub_state"), RAD_U32,
                                  {g.max_seqs, RAD_SPECHUB_STATE_WORDS}, RAD_BUF_PERSIST);
            if (!spechub_draws || !spechub_state) return RAD_E_INVAL;
        }
        rad_buf couple_draws = 0, couple_state = 0;
        if (coupled) {
            if (!m.b_mtp_didx) {
                rad_note(b, "RADIANCE_MTP_COUPLED needs the 2-bit candidate head's reranked top8");
                return RAD_E_UNSUPPORTED;
            }
            couple_draws = decl_b(b, nm.f("mtp_couple_draws"), RAD_U32,
                                  {g.max_seqs, RAD_MTP_COUPLE_DRAW_WORDS}, RAD_BUF_PERSIST);
            couple_state = decl_b(b, nm.f("mtp_couple_state"), RAD_U32,
                                  {g.max_spec, g.max_seqs, RAD_MTP_COUPLE_STATE_WORDS},
                                  RAD_BUF_PERSIST);
            if (!couple_draws || !couple_state) return RAD_E_INVAL;
        }
        rad_buf root_carry = 0, pruned_meta = 0, pruned_score = 0;
        rad_buf frontier_gather = 0, frontier_kv = 0, frontier_slots = 0;
        rad_buf frontier_prefix = 0, frontier_ancestors = 0;
        const int64_t frontier_rows = pruned_tree
            ? 8 * std::min<int64_t>(g.max_seqs, std::min(g.max_tok, g.max_logit_rows) / 8) : 0;
        if (dartree) {
            if (!pruned_tree && !spechub)
                m.b_mtp_cand_hist = decl_b(b, nm.f("mtp_cand_hist"), RAD_I32,
                                          {tree_config.rounds, g.max_seqs, kDraftTopR}, RAD_BUF_PERSIST);
            m.b_mtp_tree = decl_b(b, nm.f("mtp_tree"), RAD_I32,
                                 {g.max_seqs, 1 + 2 * tnodes}, RAD_BUF_PERSIST);
            if (!m.b_mtp_tree || (!pruned_tree && !spechub && !m.b_mtp_cand_hist))
                return RAD_E_INVAL;
            if (pruned_tree) {
                root_carry = decl_b(b, nm.f("mtp_pruned_root"), g.act_dtype,
                                   {g.max_seqs, HN}, RAD_BUF_PERSIST);
                pruned_meta = decl_b(b, nm.f("mtp_pruned_meta"), RAD_I32,
                                    {g.max_seqs, 85}, RAD_BUF_PERSIST);
                pruned_score = decl_b(b, nm.f("mtp_pruned_score"), RAD_F32,
                                     {g.max_seqs, 41}, RAD_BUF_PERSIST);
                frontier_gather = decl_b(b, nm.f("mtp_frontier_gather"), RAD_I32, {frontier_rows});
                frontier_slots = decl_b(b, nm.f("mtp_frontier_slots"), RAD_I32, {frontier_rows});
                frontier_prefix = decl_b(b, nm.f("mtp_frontier_prefix"), RAD_I32, {frontier_rows});
                frontier_ancestors = decl_b(b, nm.f("mtp_frontier_ancestors"), RAD_I32,
                                          {frontier_rows, 4});
                frontier_kv = decl_b(b, nm.f("mtp_frontier_kv"), rad_dtype_parse(g.kv_dtype),
                                    {frontier_rows, g.n_head_kv, 4, 2 * g.head_dim}, RAD_BUF_PERSIST);
                if (!root_carry || !pruned_meta || !pruned_score || !frontier_gather ||
                    !frontier_slots || !frontier_prefix || !frontier_ancestors || !frontier_kv)
                    return RAD_E_INVAL;
            } else if (tree_config.policy != 0) {
                log_hist = decl_b(b, nm.f("mtp_log_hist"), RAD_F32,
                                 {tree_config.rounds, g.max_seqs, kDraftTopR}, RAD_BUF_PERSIST);
                lse_local = decl_b(b, nm.f("mtp_lse_local"), RAD_F32, {g.max_logit_rows, 4});
                lse_gathered = decl_b(b, nm.f("mtp_lse_gathered"), RAD_F32, {g.max_logit_rows, 8});
                if (!log_hist || !lse_local || !lse_gathered) return RAD_E_INVAL;
            }
        }

        MtpHcBlock::Wire tw{};
        tw.src  = m.b_h;        tw.hin = m.b_mtp_hin;  tw.hw = m.b_mtp_hw;
        tw.e    = m.b_mtp_e;    tw.inj = m.b_inj;      tw.hout = m.b_mtp_hout;
        tw.logits = m.b_logits; tw.x = m.a_x;
        tw.dq   = m.b_mtp_dq;   tw.ds = m.b_mtp_ds;    tw.dsum = m.b_mtp_dsum;
        tw.dlog = m.b_mtp_dlog; tw.didx = m.b_mtp_didx; tw.dval = m.b_mtp_dval;
        tw.dpair = m.b_mtp_dpair; tw.dgath = m.b_mtp_dgath;
        tw.cand_hist = m.b_mtp_cand_hist; tw.packed_tree = m.b_mtp_tree;
        tw.tree_capacity = dartree ? tnodes : 0;
        tw.log_hist = log_hist; tw.lse_local = lse_local; tw.lse_gathered = lse_gathered;
        tw.tree_config = tree_config;
        tw.spechub_draws = spechub_draws; tw.spechub_state = spechub_state;
        tw.couple_draws = couple_draws; tw.couple_state = couple_state;
        tw.root_carry = root_carry; tw.pruned_meta = pruned_meta; tw.pruned_score = pruned_score;
        tw.frontier_gather = frontier_gather; tw.frontier_kv = frontier_kv;
        tw.frontier_slots = frontier_slots; tw.frontier_prefix = frontier_prefix;
        tw.frontier_ancestors = frontier_ancestors; tw.frontier_rows = frontier_rows;
        /* The layer's working set is the TRUNK's: every one of these is dead by the time a draft
         * round runs, and a second set would be 150 MiB of activation planes for one layer. Only
         * the wide stream and the carry are the head's own, because those must survive the round
         * that produced them. */
        tw.attn = aw; tw.attn.x = m.b_mtp_hw;
        tw.mlp  = mw; tw.mlp.x  = m.b_mtp_hw;
        /* Everything in the wire is within-step scratch and a draft round is its own step, so the
         * head shares all of it. The QSA tail is the exception and it is not in the wire at all:
         * a head leaving its own index keys where a trunk layer pools them is a layer pooling
         * another layer's keys, so the tail is `kv_qsa_tail` -- per sequence and per layer -- and
         * declare() binds this layer to it like any other. */
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
        /* The head's own mixer, and it has no `block_inject_weight` for the same reason the
         * trunk's 97th connection has none: nothing follows it to write back into. */
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
        ts.mlp.experts_gate_up = nm.ckpt("%s.layers.0.mlp.experts.gate_up_proj", mpre);
        ts.mlp.experts_down    = nm.ckpt("%s.layers.0.mlp.experts.down_proj", mpre);
        if (m.moecfg.n_ff_shared > 0) {
            ts.mlp.shared      = nm.ckpt("%s.layers.0.mlp.shared_expert", mpre);
            ts.mlp.shared_gate = nm.ckpt("%s.layers.0.mlp.shared_expert_gate.weight", mpre);
        }

        /* THE ATTENTION'S TWO SETTINGS ARE SET BEFORE declare(), for the reason the trunk's are:
         * `attn_paged` declares `sel` and `seqused` as READS, and a read declared after the op is
         * not a read at all. */
        m.mtp.attn.kv_block = 4;
        /* THE TREE ADMISSION, AND IT IS NOT ONLY A VETO FLAG.
         *
         * `AttnGatedFP8::declare` refuses branching when the layer carries a QSA indexer and
         * `cfg.tree_state` is false (rad_block_attn_gated_fp8.h), because a branched verify needs
         * the indexer to retain raw keys until the host commits the winner. The trunk's own layers
         * get this at their declare; the head is a layer of its own and was the one place it was
         * missed, so a RADIANCE_DARTREE deployment published a tree the scheduler then rejected
         * with "a block declared branched trees unsupported" -- and the fallback that logged it
         * served a CHAIN, i.e. no tree at all while the flag said otherwise.
         *
         * It has to be set BEFORE `mtp.declare`, which is where the attention is declared. */
        m.mtp.attn.cfg.tree_state = tree_state;
        if (m.qsaw.qk) {
            m.mtp.attn.qsa_sel      = m.qsaw.sel;
            m.mtp.attn.qsa_sequ     = m.qsaw.sequ;
            m.mtp.attn.qsa_topk     = m.qsacfg.topk;
            m.mtp.attn.qsa_exact_to = m.qsacfg.exact_to();
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
                                   m.w_tok, m.w_lm_head, m.hccfg,
                                   spechub ? 1 : dartree ? std::min<int>(g.max_spec, tree_config.rounds)
                                                        : g.max_spec,
                                   tw, ts));
        m.have_mtp = true;

        /* WHAT THE ENGINE DRIVES. Serial: `max_spec` dependent passes, each embedding what the
         * last one proposed. `proposal` is the buffer a pass leaves its token in, and the two
         * heads differ -- the 2-bit head takes its own argmax into `mtp_didx`, the bf16 head goes
         * through the ordinary sampler chain and 0 says so. */
        RadDrafterDecl dd{};
        dd.name           = "mtp";
        dd.kind           = RAD_DRAFT_SERIAL;
        dd.depth          = ctx->max_spec; // ABI capacity; the driver selects five pruned rounds.
        dd.proposal       = m.b_mtp_didx;
        dd.proposal_pitch = kDraftTopR;   /* the pick is column 0; see kDraftTopR */
        dd.mask_token     = -1;
        RAD_ARCH_TRY(rad_declare_drafter(b, &dd));
        /* Capacity counts root-inclusive verification rows, not serial draft rounds. */
        if (dartree) {
            RAD_ARCH_TRY(rad_declare_drafter_tree(b, m.b_mtp_tree, tnodes));
            rad_note(b, "MTP device tree: at most %d serial rounds, policy %d, at most %lld "
                        "proposal nodes plus root; count selects actual target rows",
                     spechub ? 1 : std::min<int>(tree_config.rounds, g.max_spec), tree_config.policy,
                     (long long)tnodes - 1);
            if (spechub)
                rad_note(b, "SpecHub live k2: ONE proposal position, GPU sparse-Q sampling over "
                            "global reranked top8, paper-order transport, full filtered target "
                            "residual; FP64 normalization reorders existing FP32 sampler numerics");
            else if (pruned_tree)
                rad_note(b, "Pruned MTP: W8 K8 D5 B7 beta=-0.2; candidate-set normalization, "
                            "real branch hidden/KV evaluation, deferred top7. MTP attention "
                            "numerics reordered FP32; target precision unchanged; sparse "
                            "target batches fail rather than fall back");
            else if (tree_config.policy != 0)
                rad_note(b, "MTP scores use full coarse-vocabulary + local-8 FP32 replacement "
                            "proxy, not full exact head softmax. cost[n] verifies root + n "
                            "proposals (n=0..7)");
            else
                rad_note(b, "MTP legacy spine-first tree: no proxy normalization, metadata "
                            "collective, or score-history work");
        }
        if (coupled)
            rad_note(b, "MTP coupled: linear chain, every round draws x~Q from the final reranked "
                        "top8 (FP64 T1 softmax) and the target sampler applies the maximal coupling "
                        "in FP64; FP64 target law, not the FP32 pick's, and not seed-identical");
        rad_note(b, "MTP head: one gated-attention layer with 512 routed experts, over the WIDE "
                    "stream, %d serial draft round(s)", ctx->max_spec);
    } else if (nextn > 0) {
        rad_note(b, "the checkpoint carries an `mtp.` head and this deployment asked for no "
                    "speculative window (max_spec 0). Its weights stay off the card");
    }

    /* ---- FullDrafter Block Drafter (arch/common/rad_block_fulldrafter.h) */
    if (use_fulldrafter) {
        m.b_fd_hout   = decl_b(b, nm.f("fd_hout"),   g.act_dtype, {g.max_logit_rows, g.n_embd});
        m.b_fd_logits = decl_b(b, nm.f("fd_logits"), RAD_F32,     {g.max_logit_rows, g.n_vocab});
        m.b_fd_cand   = decl_b(b, nm.f("fd_cand"),   RAD_I32,     {g.max_logit_rows, 16});
        m.b_fd_unary  = decl_b(b, nm.f("fd_unary"),  RAD_F32,     {g.max_logit_rows, 16});
        if (g.world > 1) {
            m.b_fd_dpair = decl_b(b, nm.f("fd_dpair"), RAD_F32, {g.max_logit_rows, 16, 2});
            m.b_fd_dgath = decl_b(b, nm.f("fd_dgath"), RAD_F32, {g.max_logit_rows, (int64_t)g.world * 16, 2});
        }
        m.b_fd_tokens = decl_b(b, nm.f("fulldrafter_tokens"), RAD_I32, {g.max_seqs, 12});

        // Direct route through teacher LM-head WITHOUT additional final teacher RMSNorm
        m.op_fd_logits = rw(b, RAD_OP(b, "logits_gemm",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                           RAD_INT("n_vocab", g.n_vocab), RAD_INT("n_embd", g.n_embd),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_WEIGHTS(m.w_lm_head)),
                            {m.b_fd_hout}, {m.b_fd_logits});

        if (g.world > 1) {
            m.op_fd_topk = rw(b, RAD_OP(b, "row_topk",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                               RAD_INT("N", g.n_vocab),
                                               RAD_INT("R", 16),
                                               RAD_INT("vocab_off", g.vocab_off),
                                               RAD_STR("dtype", g.dtype)),
                                    RAD_NOWEIGHTS),
                              {m.b_fd_logits}, {m.b_fd_cand, m.b_fd_unary, m.b_fd_dpair});
            m.op_fd_gather = rw(b, RAD_OP(b, "all_gather",
                                     RAD_PARAMS(RAD_INT("world_size", g.world),
                                                RAD_INT("numel", g.max_logit_rows * 16 * 2),
                                                RAD_STR("dtype", "f32"),
                                                RAD_INT("row", 16 * 2)),
                                     RAD_NOWEIGHTS),
                                {m.b_fd_dpair}, {m.b_fd_dgath});
            m.op_fd_merge = rw(b, RAD_OP(b, "row_topk_merge",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                               RAD_INT("R", 16),
                                               RAD_INT("world_size", g.world),
                                               RAD_STR("dtype", g.dtype)),
                                    RAD_NOWEIGHTS),
                               {m.b_fd_dgath}, {m.b_fd_cand, m.b_fd_unary});
        } else {
            m.op_fd_topk = rw(b, RAD_OP(b, "row_topk",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                               RAD_INT("N", g.n_vocab),
                                               RAD_INT("R", 16),
                                               RAD_STR("dtype", g.dtype)),
                                    RAD_NOWEIGHTS),
                              {m.b_fd_logits}, {m.b_fd_cand, m.b_fd_unary});
        }

        /* DFlash2 dynamic convolution kernel hook: H=64, block size 7 (libr4d/r4d_dflash_conv_t2_g16_bf16.hip) */
        constexpr int64_t conv_h = 64;
        constexpr int64_t conv_taps = 2;
        constexpr int64_t conv_group = 16;
        constexpr int64_t conv_ng = conv_h / conv_group;
        constexpr int64_t conv_block = 7;
        const int64_t conv_t = g.max_seqs * conv_block;

        m.b_fd_conv_x     = decl_b(b, nm.f("fd_conv_x"),     g.act_dtype, {conv_t, conv_h});
        m.b_fd_conv_delta = decl_b(b, nm.f("fd_conv_delta"), g.act_dtype, {conv_t, 2, conv_taps, conv_ng});
        m.b_fd_conv_out   = decl_b(b, nm.f("fd_conv_out"),   g.act_dtype, {conv_t, conv_h});
        m.w_fd_conv_base  = decl_w(b, nm.f("conv.w_base"), g.act_dtype, {conv_taps, conv_h},
                                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model(), /*optional=*/1);

        m.op_fd_conv = rw(b, RAD_OP(b, "dflash_conv",
                                RAD_PARAMS(RAD_RANGE("T", 1, conv_t),
                                           RAD_INT("hidden_size", conv_h),
                                           RAD_INT("taps", conv_taps),
                                           RAD_INT("group_size", conv_group),
                                           RAD_INT("NG", conv_ng),
                                           RAD_INT("block_size", conv_block),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_WEIGHTS(m.w_fd_conv_base)),
                            {m.b_fd_conv_x, m.b_fd_conv_delta}, {m.b_fd_conv_out});

        /* THE TAPS AT THE ACCEPTED ROW. The trunk gathered its taps at every out_ids row of the
         * verify step; the drafter continues from each sequence's last accepted one, which the
         * query pass names in RadBatch::draft_out_ids. Gathered through that device index rather
         * than sliced at host-side row numbers, so the pass's operands are the same every step and
         * its recording replays. */
        if (m.n_ktap < 1 || !m.b_ktap_anchor) return RAD_E_INVAL;
        for (int j = 0; j < 3; ++j) {
            m.b_fd_ctx[j] = decl_b(b, nm.f("fd_ctx%d", j), g.act_dtype,
                                   {g.max_seqs, g.n_embd}, RAD_BUF_PERSIST);
            m.op_fd_ctx[j] = rw(b, RAD_OP(b, "gather_rows",
                                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                               RAD_INT("n", g.n_embd), RAD_STR("dtype", g.dtype)),
                                    RAD_NOWEIGHTS),
                                {fd_ctx_src(m, j)}, {m.b_fd_ctx[j]});
            if (!m.b_fd_ctx[j] || !m.op_fd_ctx[j]) return RAD_E_INVAL;
        }

        m.op_fulldrafter = rw(b, RAD_OP(b, "fulldrafter_two_stage",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                           RAD_INT("block_stride", 7),
                                           RAD_STR("dtype", "bf16")),
                                RAD_NOWEIGHTS),
                            {m.b_fd_ctx[0], m.b_fd_ctx[1], m.b_fd_ctx[2], m.b_ktap_anchor},
                            {m.b_fd_tokens});

        if (ctx && ctx->shape_probe) {
            m.have_fulldrafter = true;
        } else {
            std::lock_guard<std::mutex> lk(g_shared_fd_mu);
            std::string err;
            int requested_ranks = ctx ? ctx->world_size : 2;
            std::string sc_path = (fd_sidecar && *fd_sidecar) ? fd_sidecar : "";
            if (g_shared_fd_weights.num_ranks != static_cast<size_t>(requested_ranks) ||
                g_shared_fd_path != sc_path ||
                !g_shared_fd_weights[0].d_gpu_pointers) {
                const bool use_tp2_p2p = (requested_ranks == 2);
                if (radiance::FullDrafterLoader::load_weights(sc_path, &g_shared_fd_weights, nullptr, &err, requested_ranks)) {
                    if (!radiance::FullDrafterLoader::load_teacher_matrices("", &g_shared_fd_weights, &err, use_tp2_p2p)) {
                        rad_note(b, "FullDrafter teacher matrices loading note: %s", err.c_str());
                    }
                    g_shared_fd_path  = sc_path;
                    g_shared_fd_ranks = static_cast<size_t>(requested_ranks);
                } else {
                    rad_note(b, "FullDrafter sidecar loading note: %s", err.c_str());
                }
                if (ctx && ctx->rank >= 0) (void)hipSetDevice(ctx->rank);
            }
            m.have_fulldrafter = true;
        }

        RadDrafterDecl drf{};
        drf.name           = "fulldrafter";
        drf.kind           = RAD_DRAFT_BLOCK;
        drf.depth          = 12;
        drf.block          = 7;
        drf.proposal       = m.b_fd_tokens;
        drf.proposal_pitch = 12;
        drf.window         = 0;
        drf.mask_token     = 151643;
        RAD_ARCH_TRY(rad_declare_drafter(b, &drf));
        rad_note(b, "FullDrafter declared: 2 stages of 6 tokens (total 12), discrete chaining");
        if (nextn > 0) {
            rad_note(b, "the FullDrafter drafter DISPLACES the MTP head");
        }
    }

    if (use_dloop) {
        if (m.n_ktap!=2 || m.ktap_layer[0]!=1 || m.ktap_layer[1]!=23) {
            rad_note(b,"DLoop requires RAD_KLD_TAPS=1,23 in that order");
            return RAD_E_INVAL;
        }
        /* THE DRAfter's TAP PLANES, at trunk width and filled by a full-row `cast`.
         *
         * The planes are captured where the recording's two layer taps are captured -- right after
         * `hc_mix.read`, which is that layer's block input -- plus one for the mixer's output, and
         * they are filled for EVERY token of the step. That is what makes a middle prefill chunk
         * contribute its rows to the persistent cache, and it is why these are separate buffers
         * from `kld_tap*`: those are gathered through `out_ids` and are not written at all on a
         * step that samples nothing.
         *
         * `a_x.x` is the bf16 source for all three (`g.act_dtype` is checked below), and `cast` at
         * bf16-to-bf16 is libr4d's strided row copy. */
        if (g.act_dtype != RAD_BF16) {
            rad_note(b,"DLoop taps are bf16 recordings; this model's activations are not");
            return RAD_E_UNSUPPORTED;
        }
        for (int k = 0; k < 3; ++k) {
            m.b_dtap[k] = decl_b(b, nm.f("dloop_tap%d", k), RAD_BF16,
                                 {g.max_tok, g.n_embd}, RAD_BUF_PERSIST);
            m.op_dtap[k] = rw(b, RAD_OP(b, "cast",
                                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                              RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                                   RAD_NOWEIGHTS),
                              {m.a_x.x}, {m.b_dtap[k]});
            if (!m.b_dtap[k] || !m.op_dtap[k]) return RAD_E_INVAL;
        }
        if (!ctx->shape_probe) {
            std::string err;
            const int loaded=m.dloop.load(std::getenv("RADIANCE_DLOOP_SIDECAR"),ctx->rank,g.world,err);
            if (loaded<0) {
                rad_note(b,"DLoop loading failed: %s (%d)",err.c_str(),loaded);
                return loaded;
            }
            const char* teacher=std::getenv("RADIANCE_TEACHER_MATRICES");
            const std::string path=teacher && *teacher?teacher:
                "/home/orca/drafter_data/teacher/teacher-matrices.safetensors";
            const int teacher_loaded=m.dloop.load_teacher(path,ctx->rank,g.world,248320,2560,err);
            if (teacher_loaded<0) {
                rad_note(b,"DLoop teacher loading failed: %s (%d)",err.c_str(),teacher_loaded);
                return teacher_loaded;
            }
        }
        RAD_ARCH_TRY(m.dloop.declare(b,nm,g,m.b_dtap[0],m.b_dtap[1],m.b_dtap[2]));
        rad_note(b,"DLoop DARTree: at most %lld draft nodes plus anchor; committed host verification",
                 (long long)m.dloop.budget);
    }

    /* THERE IS NO LAYER BISECT AND THERE ARE NO BLOCK ABLATIONS HERE, which is a deliberate
     * absence rather than an omission.
     *
     * A switch that truncates the layer loop, or drops a block from it, is a real instrument for a
     * COMPOSITION fault -- the class an op oracle structurally cannot see, because it compares one
     * op's device result against the host's on the operands the device handed it, and a graph
     * wired wrongly hands both sides the same wrong operands.
     *
     * What such a switch costs is that it makes the server compute something that is NOT the model
     * while looking exactly like a server that is. It is read at declare, it is invisible in a bug
     * report, and a "THIS IS NOT THE MODEL" note is one line in a startup log nobody re-reads. A
     * deployment would be one stale shell export away from being fluent and wrong. A composition
     * bisect belongs in a tool that names what it is doing in its own output, not in an
     * environment variable read by a production path.
     *
     * The residual probe (RADIANCE_DEBUG_RESID) is the instrument that fits that rule: it MEASURES
     * the composition rather than breaking it, so a server carrying it still serves the model. */
    m.n_run = g.n_layer;
    /* THE VISION TOWER, LAST: an encoder pass issues nothing else, so its activations can share
     * the arena with every buffer the language model declared above. Its weight groups start past
     * every text layer and the MTP head. */
    if (m.media) RAD_ARCH_TRY(m.vit.declare(b, meta, ctx, nm, g.n_embd, (int)g.n_layer + 64));
    return RAD_OK;
}


static void step_fulldrafter_two_stage(RadCtx* c, const RadBatch* batch, Model& m) {
    if (!m.have_fulldrafter) return;
    int rank = rad_rank(c);
    int safe_rank = (rank >= 0 && rank < (int)g_shared_fd_weights.num_ranks) ? rank : 0;
    const auto& w = g_shared_fd_weights[safe_rank];
    if (!w.teacher_embedding_table || !w.teacher_lm_head) return;

    uint32_t* out_tokens = static_cast<uint32_t*>(rad_buf_ptr(c, m.b_fd_tokens));
    if (!out_tokens) return;

    const int64_t n_seq = batch->n_seq;
    constexpr size_t C = radiance::FullDrafterConfig::CONTEXT_DIM;
    constexpr size_t W = radiance::FullDrafterConfig::TARGET_WIDTH;

    auto copy_from_buf = [](void* dst, const void* src, size_t bytes) {
        if (!src || !dst || bytes == 0) return;
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, src) == hipSuccess && attr.type == hipMemoryTypeDevice) {
            (void)hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        } else {
            (void)hipGetLastError();
            std::memcpy(dst, src, bytes);
        }
    };

    auto copy_to_buf = [](void* dst, const void* src, size_t bytes) {
        if (!src || !dst || bytes == 0) return;
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, dst) == hipSuccess && attr.type == hipMemoryTypeDevice) {
            (void)hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        } else {
            (void)hipGetLastError();
            std::memcpy(dst, src, bytes);
        }
    };

    for (int64_t s = 0; s < n_seq; ++s) {
        /* step() has already gathered this sequence's accepted-row taps and its anchor embedding
         * into row s of fd_ctx* and kld_tap_anchor; the anchor token is its block's first row. */
        int64_t block_stride = (n_seq > 0) ? (batch->n_tok / n_seq) : 7;
        int64_t anchor_idx = s * block_stride;

        uint32_t anchor_tok = 0;
        if (batch->token_ids && anchor_idx < batch->n_tok) {
            hipPointerAttribute_t attr{};
            if (hipPointerGetAttributes(&attr, batch->token_ids) == hipSuccess && attr.type == hipMemoryTypeDevice) {
                int32_t dev_tok = 0;
                (void)hipMemcpy(&dev_tok, batch->token_ids + anchor_idx, sizeof(int32_t), hipMemcpyDeviceToHost);
                anchor_tok = static_cast<uint32_t>(dev_tok);
            } else {
                (void)hipGetLastError();
                anchor_tok = static_cast<uint32_t>(batch->token_ids[anchor_idx]);
            }
        }

        std::vector<radiance::rad_bf16_t> dummy_ctx(C, 0);
        for (int j = 0; j < 3; ++j) {
            const void* p = rad_buf_ptr(c, m.b_fd_ctx[j]);
            if (p)
                copy_from_buf(dummy_ctx.data() + j * W,
                              static_cast<const radiance::rad_bf16_t*>(p) + s * W,
                              W * sizeof(radiance::rad_bf16_t));
        }
        const radiance::rad_bf16_t* ctx_ptr = dummy_ctx.data();

        std::vector<radiance::rad_bf16_t> dummy_anc(W, 0);
        if (const void* anc_dev = rad_buf_ptr(c, m.b_ktap_anchor))
            copy_from_buf(dummy_anc.data(), static_cast<const radiance::rad_bf16_t*>(anc_dev) + s * W,
                          W * sizeof(radiance::rad_bf16_t));
        const radiance::rad_bf16_t* anc_ptr = dummy_anc.data();

        std::vector<uint32_t> local_tokens(12, 0);
        if (g_shared_fd_weights.num_ranks == 2) {
            const auto& w0 = g_shared_fd_weights[0];
            const auto& w1 = g_shared_fd_weights[1];
            radiance::FullDrafterExecutor::propose_candidates_tp2(
                ctx_ptr,
                anc_ptr,
                anchor_tok,
                w0.teacher_lm_head,
                w1.teacher_lm_head,
                w.teacher_embedding_table,
                w,
                local_tokens.data()
            );
        } else {
            radiance::FullDrafterExecutor::propose_candidates(
                ctx_ptr,
                anc_ptr,
                anchor_tok,
                w.teacher_lm_head,
                w.teacher_embedding_table,
                w,
                local_tokens.data()
            );
        }
        copy_to_buf(out_tokens + s * 12, local_tokens.data(), 12 * sizeof(uint32_t));
    }
}


static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_rank(c)];
    const Geom& g = m.g;
    const int64_t T = batch->n_tok;

    /* AN ENCODER PASS RUNS THE VISION TOWER AND NOTHING ELSE (RadBatch::enc). */
    if (batch->enc) {
        m.vit.step(c, batch);
        return;
    }

    /* THE DLOOP APPEND RIDES THE TREE-COMMIT RELEASE, AND IT RUNS ALONGSIDE THE GDN/PLE/QSA
     * COMMIT RATHER THAN INSTEAD OF IT.
     *
     * This release exists because acceptance has just been resolved on the host; the tree-state
     * blocks use it to commit the winner path into their state, and the full-context drafter uses
     * it to append the verified rows to its persistent cache. Both are consumers of the same event
     * and neither replaces the other, so the append is issued FIRST -- it depends only on the tap
     * planes this step recorded, which no commit touches -- and the commits then run as before. */
    if (batch->draft_pass == -2) {
        /* THE WINNER-PATH COMMIT RELEASE, AND IT IS NOT DLoop'S ALONE.
         *
         * This release exists because acceptance has just been resolved on the host, so the
         * stateful layers can commit the accepted node path into their recurrent state, their
         * convolution windows and their pooled index keys. `gdn_tree_commit`, `qsa_tree_*` and
         * `ple_tree_commit` are the ops that do it, and they are declared whenever a deployment
         * may verify a BRANCHED window -- which DLoop needs, and which the built-in drafters need
         * when RADIANCE_DARTREE asked them for a tree proposal.
         *
         * So gating this on `m.dloop.enabled` left those deployments with a tree PROPOSAL and no
         * winner-path commit: the layers' state advanced over the rejected siblings, and the next
         * step conditioned the model on text it never emitted. The append is DLoop's own and stays
         * behind its flag; the commits are every tree-state program's. */
        if (m.dloop.enabled) m.dloop.step(c,batch);
        if (m.ple_layer >= 0) m.ple.commit(c,batch);
        for (const Layer& l:m.layers) {
            if (l.full) l.qsa.commit(c,batch);
            else l.gdn.commit(c,batch);
        }
        /* THE HEAD HAS NO COMMIT HERE, DELIBERATELY. Its indexer declares `tree_state` -- it shares
         * the trunk's `m.qsacfg`, and the kv_qsa_tail group's binding is only complete with the
         * persistent `blk.<layer>.qsa_tree_qk` that flag declares -- but `QsaIndexer::defer_decode`
         * also requires `batch->spec_parent`, and the head only ever runs DRAFT passes, which are
         * planned without a parent array. So it never defers, never needs a replay, and a
         * `commit` call here would be a branch nothing can reach. The tree is verified by the
         * TRUNK, whose layers are committed in the loop above. */
        return;
    }
    if (batch->draft_pass != 0) {
        if (m.dloop.enabled) {
            /* ONE ARGUMENT: the tap planes and the span are buffers the block owns and reads, not
             * per-call parameters. The block distinguishes the append release (draft_pass == -2)
             * from the query pass itself. */
            m.dloop.step(c,batch);
            return;
        }
        /* ---- THE DFLASH2 DRAFTER. Block kind: a CONTEXT pass that mirrors the trunk step one row
         * a committed token, then ONE pass that fills the whole draft block. `declare` has already
         * refused every other drafter in this container, so there is no ambiguity about which one
         * the engine is driving.
         *
         * The context pass reads `b_dfsrc`, which the trunk step before it filled after each named
         * completed layer -- so the rows are the taps of the tokens this batch just committed, in
         * trunk-row order, which is what the pass's slot mapping addresses. */
        if (m.have_df) {
            if (batch->draft_pass > 0) m.df.step(c, batch);
            else                       m.df.context(c, batch);
            return;
        }
        if (m.have_fulldrafter) {
            if (batch->draft_pass > 0) {
                /* THE QUERY PASS'S INPUTS, both read through device indices, so the pass issues
                 * the same operands every step and its recording replays.
                 *
                 * The ANCHOR is the token the trunk just sampled, at the first row of each
                 * sequence's block (tok_shift 0) -- not the trunk's own input at its last row,
                 * the token before it, which is what kld_tap_anchor holds after the trunk step.
                 * Embedded here and gathered at cu_seqlens into kld_tap_anchor rows 0..n_seq-1;
                 * a --kld-record run copies that buffer after the trunk step and drafts nothing.
                 *
                 * The CONTEXT is the trunk's taps at each sequence's last accepted position, the
                 * row draft_out_ids names (core/sched/scheduler.cpp, block_begin) -- row s holds
                 * whatever position the verify step happened to put there. */
                const int64_t n_seq = batch->n_seq;
                if (n_seq <= 0 || batch->n_draft_out != n_seq || !batch->draft_out_ids) {
                    rad_step_fail(c, "FullDrafter query pass names no accepted trunk row a "
                                     "sequence (RadBatch::draft_out_ids)");
                    return;
                }
                int rank = rad_rank(c);
                int safe_rank = (rank >= 0 && rank < (int)g_shared_fd_weights.num_ranks) ? rank : 0;
                const auto& w = g_shared_fd_weights[safe_rank];
                const bool fast_local_anc = false;

                if (!fast_local_anc) {
                    if (n_seq == 1) {
                        RAD_ISSUE_N(c, m.op_embed, 1, praw(batch->token_ids, RAD_I32, 1),
                                    RAD_W(m.w_tok), brows(m.b_ktap_anchor, 1));
                        if (m.op_embed_ar)
                            RAD_ISSUE_N(c, m.op_embed_ar, g.n_embd, brows(m.b_ktap_anchor, 1), RAD_NONE);
                    } else {
                        RAD_ISSUE_N(c, m.op_embed, T, praw(batch->token_ids, RAD_I32, T),
                                    RAD_W(m.w_tok), brows(m.a_x.x, T));
                        if (m.op_embed_ar)
                            RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE);
                        RAD_ISSUE_N(c, m.op_ktap_anchor, n_seq,
                                    brows(m.a_x.x, T), praw(batch->cu_seqlens, RAD_I32, n_seq),
                                    brows(m.b_ktap_anchor, n_seq));
                    }
                }
                if (m.op_fulldrafter && w.d_gpu_pointers && m.b_fd_tokens) {
                    if (rank == 0 || w.tp2_p2p_active) {
                        for (int j = 0; j < 3; ++j)
                            RAD_ISSUE_N(c, m.op_fd_ctx[j], n_seq,
                                        RAD_B(fd_ctx_src(m, j)), praw(batch->draft_out_ids, RAD_I32, n_seq),
                                        brows(m.b_fd_ctx[j], n_seq));
                        /* The parameter operand's count carries scheduled slots (1..12) to the
                         * launcher; the pointer still names the full struct. Recorded with the
                         * issue, so tape replay preserves 3/4/7 without a host upload or sync. */
                        RAD_ISSUE_N(c, m.op_fulldrafter, n_seq,
                                    brows(m.b_fd_ctx[0], n_seq),
                                    brows(m.b_fd_ctx[1], n_seq),
                                    brows(m.b_fd_ctx[2], n_seq),
                                    brows(m.b_ktap_anchor, n_seq),
                                    praw(batch->token_ids, RAD_I32, T),
                                    praw(w.d_gpu_pointers, RAD_BF16, batch->n_spec),
                                    brows(m.b_fd_tokens, n_seq));
                    }
                } else {
                    for (int j = 0; j < 3; ++j)
                        RAD_ISSUE_N(c, m.op_fd_ctx[j], n_seq,
                                    RAD_B(fd_ctx_src(m, j)), praw(batch->draft_out_ids, RAD_I32, n_seq),
                                    brows(m.b_fd_ctx[j], n_seq));
                    step_fulldrafter_two_stage(c, batch, m);
                }
            }
            return;
        }
        if (!m.have_mtp) {
            rad_step_fail(c, "the batch asks for a drafter pass and no drafter is declared: the "
                             "container carries no `mtp.` head, or declare ran at max_spec 0, "
                             "which is the engine saying it would not draft");
            return;
        }
        /* THE HEAD'S OWN cos/sin ROWS, AND THE TRUNK DID NOT WRITE THEM. `rope_table` fills the
         * rows THIS STEP'S POSITIONS name, and a draft round runs PAST the committed end -- the
         * scheduler says so where it reserves the lookahead blocks: "round k writes the head's K
         * and V at position head_i + k - 1, which runs past the committed end". The trunk step
         * before this one covered head_i and no further, so without this call rounds past the
         * first would rotate by whatever the table held at a row nothing had written yet.
         *
         * LEAVING IT OUT COSTS ONLY ACCEPTANCE, WHICH IS WHY IT IS EASY TO MISS. A draft is
         * verified; a mis-rotated one is rejected and the text is still the trunk's, so the
         * failure has no symptom except a draft that agrees less often. It becomes visible only
         * once the indexer reads the same table: the head's QSA selection moves, its drafts move,
         * and the accepted text moves with them at tie level -- a transcript difference at long
         * context and none at all below the exactness bound, where every block is selected
         * regardless.
         *
         * A row is a function of its POSITION ALONE -- the angle for position P is the same angle
         * for every request and every step -- so re-filling costs nothing but the launch and can
         * never write a row a reader disagrees with. */
        if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
            RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));
        m.mtp.step(c, batch);
        return;
    }

    RAD_ISSUE(c, m.op_embed,
              praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_tok), brows(m.a_x.x, T));
    m.mrows.step(c, batch, m.a_x.x, T);
    /* Unconditional, unlike qwen35_fp8's: there is no norm in front of the first block to fold it
     * into, because the first thing that touches the stream is a gated-residual read. */
    if (m.op_embed_ar)
        RAD_ISSUE_N(c, m.op_embed_ar, T * g.n_embd, brows(m.a_x.x, T), RAD_NONE);
    if (m.op_ktap_anchor && batch->n_out > 0)
        RAD_ISSUE_N(c, m.op_ktap_anchor, batch->n_out,
                    brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_ktap_anchor, batch->n_out));
    /* The PLE's token-dependent ops (hash, gather, cast, key/value projections) run on Lane 1
     * concurrently with enter, rope, and Layer 0 (Block 0 GDN + MoE), hiding PLE gather latency.
     * ONLY FOR SMALL PASSES: a prefill chunk's PLE GEMMs compete with Block 0 for the CUs instead
     * of filling idle ones -- measured -6% at 64k prefill, while decode/verify gains. Large passes
     * run it inline at the PLE layer. T is a recorded shape, so the branch is replay-stable. */
    static const bool ple_nolane = getenv("RAD_PLE_NOLANE") != nullptr;  /* A/B switch */
    const bool ple_inline = ple_nolane || T > 128;
    if (m.ple_layer >= 0 && !ple_inline) {
        /* JOIN FIRST. Lane 1 is not ordered behind the previous pass's lane 0 by anything else,
         * and the host runs ahead of the device: without this the prep of THIS pass overwrites
         * ple_ids / ple_k / ple_v and the n-gram window state while the previous pass's gate on
         * lane 0 still reads them -- fluent, wrong text and a collapsed draft acceptance. */
        rad_lane(c, 1);              /* builds lane 1 on first use; lane_join is a no-op before */
        rad_lane_join(c, 0, 1);
        m.ple.prep(c, batch);
        rad_lane(c, 0);
    }
    m.enter.step(c, T);

    if (m.op_rope_cs && T <= qk_fuse_rows(g) && !rope_mixed(batch))
        RAD_ISSUE_N(c, m.op_rope_cs, T, rope_pos1(batch, T), RAD_B(m.b_rope_cs));

    for (int64_t li = 0; li < m.n_run && li < (int64_t)m.layers.size(); ++li) {
        const Layer& l = m.layers[(size_t)li];
        /* Before the attention connection reads the stream: join lane 1 and issue gate, conv, add. */
        if (li == m.ple_layer) {
            if (ple_inline) m.ple.prep(c, batch);
            else            rad_lane_join(c, 1, 0);
            m.ple.gate_conv_add(c, batch);
        }
        {
            l.hc_mix.read(c, T, 0, T);
            /* THE DLOOP TAPS TAKE THE SAME SOURCE AS THE LAYER TAPS and are filled for the whole
             * step: one plane a recorded layer, one row a token of this step. The recording taps
             * keep gathering through `out_ids` and stay empty on a step that samples nothing; these
             * must not, because the persistent cache is built from them. */
            if (m.dloop.enabled) {
                for (int j = 0; j < 2; ++j)
                    if (m.ktap_layer[j] == (int)li)
                        RAD_ISSUE_N(c, m.op_dtap[j], T,
                                    brow_slice(m.a_x.x, 0, T, g.n_embd),
                                    brow_slice(m.b_dtap[j], 0, T, g.n_embd));
            }
            for (int k = 0; k < m.n_ktap; ++k)
                if (m.ktap_layer[k] == (int)li && !m.dloop.enabled && batch->n_out > 0)
                    RAD_ISSUE_N(c, m.op_ktap[k], batch->n_out,
                                brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                                brows(m.b_ktap[k], batch->n_out));
            if (l.full) {
                l.qsa.step(c, l.attn.w.h, batch);
                l.attn.step(c, batch);
            }
            else        l.gdn.step(c, batch);
            l.hc_mix.write(c, T, 0, T);
        }
        dbg_resid(c, (int)li, "mix", g.n_embd, m.b_h, m.a_x.x);
        {
            l.hc_ffn.read(c, T, 0, T);
            l.mlp.step(c, batch);
            l.hc_ffn.write(c, T, 0, T);
        }
        dbg_resid(c, (int)li, "ffn", g.n_embd, m.b_h, m.a_x.x);
        /* THE DFLASH2 TAP, AFTER THE COMPLETED LAYER -- exactly the capture's site. Both
         * connections have written, so `b_h` holds that layer's finished output over all hc
         * streams, which is the only thing `hc_mean` may read. NOT `hc_mix.read`, not `a_x.x`
         * (the block input), and not any DLoop/kld plane: see declare().
         *
         * The output is tap t's COLUMN BLOCK of `dflash_src`: width n_embd at column t*n_embd,
         * with the plane's n_taps*n_embd row pitch kept as the stride. Rows are this step's
         * tokens, in trunk-row order, which is what the context pass's slot mapping addresses. */
        if (m.have_df) {
            for (int t = 0; t < m.df_n_taps; ++t) {
                if (m.df_tap_layer[t] != (int)li) continue;
                RAD_ISSUE_N(c, m.op_df_tap[t], T,
                            brow_slice(m.b_h, 0, T, m.hccfg.hc * g.n_embd),
                            bcol(m.b_dfsrc, (int64_t)t * g.n_embd, g.n_embd, T));
            }
        }
    }

    /* THE MIXER PRODUCES THE HIDDEN STATE, AND THE DLOOP TAPS NEED IT FOR THE WHOLE STEP.
     *
     * `mixer.read` collapses the wide gated-residual stream into `a_x.x`, which is what the
     * lm_head reads and what `kld_tapN` (the lm_head's own input) records. On a step that samples
     * nothing -- the middle chunk of a long prefill -- there is no logits row to produce, but the
     * DLoop tap planes still need every token's mixer output, in trunk-row order, because those
     * are the rows the persistent cache is built from. So the read runs over the whole step for
     * DLoop and over `T-1,1` only for the single-output fast path everything else uses. */
    if (m.dloop.enabled) {
        m.mixer.read(c, T, 0, T);
        RAD_ISSUE_N(c, m.op_dtap[2], T,
                    brow_slice(m.a_x.x, 0, T, g.n_embd),
                    brow_slice(m.b_dtap[2], 0, T, g.n_embd));
    }
    if (batch->n_out > 0) {
        if (!m.dloop.enabled) {
            if (batch->n_seq == 1 && batch->n_out == 1)
                m.mixer.read(c, T, T - 1, 1);
            else
                m.mixer.read(c, T, 0, T);
        }
        RAD_ISSUE_N(c, m.op_gather, batch->n_out,
                    brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(m.b_hout, batch->n_out));
        RAD_ISSUE_N(c, m.op_logits, batch->n_out,
                    brows(m.b_hout, batch->n_out), RAD_W(m.w_lm_head),
                    brows(m.b_logits, batch->n_out));
        if (m.n_ktap > 0 && !m.dloop.enabled)
            RAD_ISSUE_N(c, m.op_ktap[m.n_ktap], batch->n_out,
                        brows(m.a_x.x, T), praw(batch->out_ids, RAD_I32, batch->n_out),
                        brows(m.b_ktap[m.n_ktap], batch->n_out));
    }
}

/* WHAT THE PLUGIN WOULD DRAFT AT, which is what `--num-speculative-tokens auto` takes and what
 * rad-convert declares at when the run that makes a container would not itself have drafted.
 *
 * 3 and not `mtp_num_hidden_layers`: the checkpoint says the head EXISTS (one layer), never how
 * deep a chain of it is worth running. It is an operating point, not a property of the model,
 * which is why `draft_depth_fixed` stays 0 and the flag overrides it.
 *
 * THREE IS WHERE THROUGHPUT PEAKS on this model. Per-POSITION acceptance stays high at every
 * depth, so the head is not what runs out; what runs out is the VERIFY step. Each extra query row
 * costs real time here because the model's weights do not fit device memory -- more rows mean more
 * experts streamed and more n-gram rows faulted out of the mapped table -- and most of what a
 * deeper step adds is that, not the head's own ops. */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->draft_depth = 0;
    if (DLoopBlock::requested()) {
        const int depth=DLoopBlock::tree_budget();
        const char* fd=std::getenv("RADIANCE_FULLDRAFTER");
        const char* fd_sidecar=std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
        if (depth<0) {
            fprintf(stderr,"radiance: DLoop requires RADIANCE_DARTREE_BUDGET=1..15\n");
            return RAD_E_INVAL;
        }
        if ((fd && *fd && std::strcmp(fd,"0")) || (fd_sidecar && *fd_sidecar)) {
            fprintf(stderr,"radiance: RADIANCE_DLOOP_DRAFTER=1 conflicts with RADIANCE_FULLDRAFTER or RADIANCE_FULLDRAFTER_SIDECAR\n");
            return RAD_E_INVAL;
        }
        out->draft_depth=depth;
        out->shape_probe_ok=1;
        return RAD_OK;
    }
    const char* fd_env = std::getenv("RADIANCE_FULLDRAFTER");
    const char* fd_sidecar = std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
    const int64_t fd_meta = meta_int2(meta, "qwen4exp.fulldrafter", "fulldrafter", 0);
    if ((fd_env && *fd_env && strcmp(fd_env, "0") != 0) ||
        (fd_sidecar && *fd_sidecar) ||
        (fd_meta > 0)) {
        out->draft_depth = 12;
    } else if (wants_dflash2(meta, nullptr)) {
        /* THE DECLARED DEPTH, which is what `auto` takes. A block-diffusion drafter's checkpoint
         * names a `block_size` and that is the default operating point; the serve may ask for a
         * SHORTER block (`declare_df` allocates and issues exactly the requested rows) and a
         * longer one is refused there. Reporting the declared depth here is therefore the
         * default, not the only depth. */
        const int64_t bs = rad_meta_geti(meta, "draft.dflash_config.block_size", 0);
        if (bs < 2) {
            fprintf(stderr, "radiance: qwen4exp: this container names a DFlash2 drafter and no "
                            "`draft.dflash_config.block_size`, so the depth it drafts at is "
                            "unknown.\n");
            return RAD_E_FORMAT;
        }
        out->draft_depth = (int)(bs - 1);
    } else if (meta_int2(meta, "qwen4exp.mtp_num_hidden_layers", "mtp_num_hidden_layers", 0) > 0) {
        out->draft_depth = 3;
    }
    out->shape_probe_ok = 1;      /* declare() writes scratch under shape_probe */
    return RAD_OK;
}

}  /* namespace qwen4exp_fp8 */

#ifndef RAD_ARCH_NO_EXPORTS
RAD_ARCH_PROBE(qwen4exp_fp8)
RAD_ARCH_PLUGIN(qwen4exp_fp8, "qwen4exp", "", "0.1.0",
                "Qwen4-Exp (Qwen3.8-Flash-Next): hybrid Gated DeltaNet / gated attention, a GATED "
                "RESIDUAL four streams wide, and 512 routed experts -- from a DENSE bf16 "
                "checkpoint, quantised ahead of time by a recipe")
#endif
