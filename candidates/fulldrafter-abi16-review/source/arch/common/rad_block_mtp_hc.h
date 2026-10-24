/* rad_block_mtp_hc.h -- the multi-token-prediction head of a model whose residual stream is `hc`
 * times the model width.
 *
 * ============================== WHY THIS IS NOT rad_block_mtp_fp8.h ==============================
 *
 * That file serves Qwen3.5 / Qwen3-Next / Qwen3.6, where the head is a PRE-NORM transformer layer
 * that reads the trunk's [n_embd] hidden state. It is the same idea, the same indexing and the
 * same two kinds of pass -- and almost none of the same parts, because Qwen4-Exp's head lives in
 * the WIDE stream:
 *
 *   ITS INPUT IS 10240 WIDE      `mtp.pre_fc_norm_hidden.weight` is [hc * n_embd], so the hidden
 *                               state this head continues from is the trunk's wide stream BEFORE
 *                               the output mixer collapses it, not `last_hidden_state`.
 *   THERE ARE TWO fc MATRICES    `mtp.fc_hidden` and `mtp.fc_embedding`, both [n_embd, n_embd].
 *                               So there is no concatenation and no concat-ORDER question: the
 *                               hidden half is one square matrix applied to EACH sub-stream and
 *                               the embedding half is one product broadcast into all of them.
 *                               docs/OPS.md's `mtp_enter` row is that arithmetic.
 *   ITS LAYER HAS TWO GATED      `mtp.layers.0.attn_hyper_connection` and `.mlp_hyper_connection`,
 *   RESIDUALS, NOT TWO NORMS     exactly as a trunk layer does. There is no `input_layernorm` and
 *                               no `post_attention_layernorm` in this checkpoint at all.
 *   AND A MIXER OF ITS OWN       `mtp.hyper_connection_mixer` -- a connection with no write half,
 *                               which is what collapses the head's wide stream for the lm_head.
 *                               It is the 98th connection in the model and the trunk's 97th has
 *                               the same shape and the same absent `block_inject_weight`.
 *
 * What is NOT different is the part that is easy to get wrong, so it is restated here in full.
 *
 * ============================== THE INDEXING ==============================
 *
 *     head index i    reads  hidden(i) and embed(token at i + 1)
 *                     writes its K and V at KV position i, with rope position i
 *                     predicts the token at i + 2
 *
 * DeepSeek-V3's MTP layout, which this family follows: the head's transformer block runs over the
 * SAME position indices as the trunk and the shift lives entirely in which token gets embedded.
 * Getting it off by one costs nothing visible -- the drafts are simply wrong, acceptance collapses,
 * and the model's own output stays exactly correct.
 *
 * ============================== THE TWO KINDS OF PASS ==============================
 *
 * `RadBatch::draft_pass` says which and the SIGN is the whole distinction:
 *
 *   < 0   A HISTORY PASS, one row a token, continuing from the trunk's own wide stream. The
 *         head's attention reads its OWN K and V for every earlier position and nothing else in
 *         the engine writes those -- the trunk's prefill fills layers 0..n-1 and leaves the head's
 *         untouched. Without it the head attends to whatever the pool last held, which is not
 *         wrong output (a bad draft is only a rejected draft) but is a drafter that proposes noise.
 *         It is also DRAFT ROUND 1 for the sequences that draft: each one's last history row is
 *         the committed position with the token the trunk chose after it, which is round 1's
 *         whole input, so the pass ends in the lm_head over those rows (RadBatch::draft_out_ids)
 *         and hands their wide state on as the carry.
 *   > 1   A DRAFT ROUND, one row a sequence, continuing from the head's own stream. There is no
 *         round-1 pass; the history pass is it.
 *
 * ============================== THE CARRY, AND WHY IT IS A COPY ==============================
 *
 * `hin` is what a round continues FROM and `hw` is the head's own stream. They cannot be one
 * buffer: `mtp_enter` reads the whole wide row to produce every channel of its output, so a
 * workgroup writing channels 0..31 would clobber input another workgroup has not read yet. The
 * pre-norm head has no such problem because its carry is written by a DIFFERENT op at the end of
 * the round (`mtp.norm`), and here the end of the round is the mlp connection's write, in place.
 *
 * So one `cast` copies `hw` back to `hin` between rounds -- a 20 KB row copy at decode, whose cost
 * is its launch -- and it is SKIPPED on the last round because nothing reads it there. A history
 * pass that drafts takes its carry with a `gather_rows` instead, of the round-1 rows alone, and
 * copies them back to the front of `hw`, where the mixer reads a round's rows.
 *
 * ============================== THE KV CACHE ==============================
 *
 * The head's attention layer is bound to the TRUNK'S full-attention group as one more layer, at
 * absolute index n_layer. The slot mapping and the block table are the trunk's, so the head writes
 * its K and V at the slot the trunk's token for that position occupies and A REJECTED DRAFT'S
 * ENTRY IS OVERWRITTEN BY THE NEXT STEP'S CORRECT TOKEN AT THE SAME ADDRESS. Draft rounds run in
 * position order and each writes its own position before attending to it, so there is no rollback
 * to get wrong. The cost is that the group's page grows from 12 layers to 13 -- 8% more attention
 * KV -- and it is paid only when the deployment declares speculation.
 *
 * Pruned MTP is the exception: branch rounds store only into a fixed provisional overlay and
 * attend through explicit ancestor slots. Committed KV/QSA stay read-only until the next history
 * pass rebuilds newly accepted positions from target hidden states. No full prefix is copied.
 *
 * ============================== WHAT IS AND IS NOT DECLARED ==============================
 *
 * The head reuses the TRUNK'S working buffers for its layer -- `attn_qg`, the expert planes, the
 * narrow block input. Every one of them is dead by the time a draft round runs (the trunk step has
 * been sampled and committed) and a second set would be 150 MiB of activation planes for one
 * layer. Only the wide stream, the carry, the embedding and the head's own logits path are its
 * own, because those must survive the round that produced them.
 *
 * THIS BLOCK MUST THEREFORE BE DECLARED AFTER THE TRUNK'S LAST OP. Declaration order IS the buffer
 * planner's liveness (core/build/rad_bufplan.cpp) and a drafter declared mid-trunk shares bytes
 * with buffers the trunk still has live. The signature of getting it wrong is fluent output with
 * acceptance EXACTLY zero.
 */
#ifndef RAD_BLOCK_MTP_HC_H
#define RAD_BLOCK_MTP_HC_H

#include "rad_block_attn_gated_fp8.h"
#include "rad_block_hc.h"
#include "rad_block_moe_fp8.h"
#include "rad_block_mtp_fp8.h"     /* mtp_draft_head_q2(): one reading of the 2-bit head */
#include "rad_qsa.h"
#include "rad_block_vit.h"
#include "rad_mtp_tree_config.h"
#include "rad_spechub_config.h"
#include "rad_spechub.h"

#include <cstdlib>

namespace rad {
namespace arch {

/* HOW MANY CANDIDATES THE COARSE HEAD PROPOSES, and it is 8 because they are RESCORED.
 *
 * A draft needs one token. The 2-bit head picks a different top-1 than the exact head often enough
 * to cost several points of acceptance, so the coarse head proposes R candidates and
 * `logit_rerank` scores those R against the full-precision head before column 0 is read. 8 is the
 * reference implementation's number, which recovered a coarse head to 99.63% agreement with the
 * exact argmax; the cost is R rows of the weight a draft row, 40 KB against 318 MB. */
/* AND R HAS AN OPTIMUM, which is not the obvious thing. A wider candidate set can only raise the
 * EXACT score the rescore selects, so more candidates ought to be free -- and both sixteen and
 * four are reproducibly worse than eight on draft acceptance.
 *
 * Because acceptance is agreement with the TRUNK and the rescore maximises the MTP HEAD's own
 * preference over the candidate set. Past some width, taking that preference more seriously walks
 * away from the trunk's argmax -- so the coarse head's ranking is a useful restriction and not
 * just noise to be corrected. */
static constexpr int64_t kDraftTopR = 8;
/* Keep every rescored candidate for tree siblings and the serial column-0 winner.
 * Eight (i32 id, f32 value) pairs also meet the gather's 16-byte row alignment. */
static constexpr int64_t kDraftGatherR = kDraftTopR;

struct MtpHcBlock {
    struct Wire {
        /* [max_tok, hc * n_embd] the TRUNK's wide stream for the step just run. A pass names the
         * rows it wants through out_ids and never has to know how they were laid out. */
        rad_buf src  = 0;
        rad_buf hin  = 0;   /* [max_tok, hc * n_embd]  what this round continues from */
        rad_buf hw   = 0;   /* [max_tok, hc * n_embd]  the head's own wide stream */
        rad_buf e    = 0;   /* [max_tok, n_embd]       the input token's embedding */
        rad_buf inj  = 0;   /* [max_tok, hc]           both connections' write gains */
        rad_buf hout = 0;   /* [max_logit_rows, n_embd]  the mixer's output, the lm_head's input */
        rad_buf logits = 0; /* [max_logit_rows, n_vocab] the trunk's, shared */
        ActFP8  x{};        /* the narrow block input -- the TRUNK's */
        /* THE 2-BIT DRAFT HEAD's working set. Every one of these is a draft round's, so they are
         * sized at max_logit_rows: a history pass never reaches any of them. */
        rad_buf dq = 0, ds = 0, dsum = 0, dlog = 0, didx = 0, dval = 0;
        /* The vocab-parallel pair planes, declared only when the head is SHARDED:
         * [M,8,2] local pairs and [M,world*8,2] gathered pairs, with i32 id bits in f32. */
        rad_buf dpair = 0, dgath = 0;
        /* Persistent across the serial passes; round 1 resets the logical tree. */
        rad_buf cand_hist = 0, packed_tree = 0;
        rad_buf log_hist = 0, lse_local = 0, lse_gathered = 0;
        MtpTreeConfig tree_config{};
        int64_t tree_capacity = 0;
        rad_buf spechub_draws = 0, spechub_state = 0;
        /* RADIANCE_MTP_COUPLED: [max_seqs] draws and [depth, max_seqs] states, both PERSIST. */
        rad_buf couple_draws = 0, couple_state = 0;
        rad_buf root_carry = 0, pruned_meta = 0, pruned_score = 0;
        rad_buf frontier_gather = 0, frontier_kv = 0, frontier_slots = 0;
        rad_buf frontier_prefix = 0, frontier_ancestors = 0;
        int64_t frontier_rows = 0;
        AttnGatedFP8::Wire attn{};
        MoeFP8::Wire       mlp{};
        QsaIndexer::Wire   qsa{};
    };

    struct Src {
        const char* fc_hidden          = nullptr;
        const char* fc_embedding       = nullptr;
        const char* pre_norm_hidden    = nullptr;
        const char* pre_norm_embedding = nullptr;
        HyperConn::Src    hc_attn{}, hc_mlp{}, hc_mix{};
        AttnGatedFP8::Src attn{};
        MoeFP8::Src       mlp{};
        /* The indexer's names and geometry; `qk_proj` null declares no indexer at all. */
        QsaIndexer::Config qsa{};
    };

    Geom    g{};
    Wire    w{};
    int     layer  = 0;
    int64_t rows   = 0;
    int     depth  = 0;    /* rounds this deployment will run -- the carry's last-round test */
    int64_t ngroup = 0;    /* the pre-fc norm's groups: hc, one rsqrt a sub-stream */
    bool    q2     = false;
    /* The 2-bit draft head's logical weight: the checkpoint's lm_head under a name of its own,
     * which a recipe stores as 2-bit codes. The plugin asks mtp_draft_head_q2 about this name --
     * which also declares its map -- and declares the head's buffers when the answer is yes. */
    static const char* head_name(Names& nm) { return nm.f("mtp.draft_head.weight"); }

    rad_weight w_tok = 0, w_lm = 0, w_fc_h = 0, w_fc_e = 0, w_pre_h = 0, w_pre_e = 0;
    rad_weight w_dh = 0, w_dhs = 0;

    HyperConn    hc_attn{}, hc_mlp{}, mixer{};
    AttnGatedFP8 attn{};
    QsaIndexer   qsa{};
    MoeFP8       mlp{};

    rad_op op_gather = 0, op_embed = 0, op_embed_ar = 0, op_enter = 0, op_carry = 0;
    rad_op op_pruned_expand[5]{}, op_pruned_prepare[4]{}, op_pruned_pack = 0;
    rad_op op_root_carry = 0, op_root_gather = 0, op_parent_gather = 0;

    /* THE PROMPT'S MEDIA ROWS over the head's own embedding: its history pass embeds the prompt
     * too, one position on, and an image token's embedding is the encoder's there as well. Set
     * `media` before declare when the program has an encoder. */

    bool       media = false;

    MediaRows  mrows{};
    rad_op op_dsel = 0, op_dback = 0;
    rad_op op_logits = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0, op_drerank = 0;
    rad_op op_dgather = 0, op_dmerge = 0;
    rad_op op_mtp_tree[8] = {};
    rad_op op_mtp_merge_tree[8] = {};
    rad_op op_mtp_scored_tree[8] = {};
    rad_op op_hybrid_lse = 0, op_lse_gather = 0;
    rad_op op_spechub = 0;
    rad_op op_couple[RAD_MTP_COUPLE_MAX_ROUNDS] = {};

    /* rad_arch.h's, asked through the block so whoever sizes `dlog` and the pair planes does not
     * re-derive the rule. */
    static bool    head_sharded(const Geom& g) { return vocab_head_sharded(g); }
    static int64_t head_rows(const Geom& g)    { return vocab_head_rows(g); }

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup kv,
                 rad_kvgroup* qsa_bk, int64_t rotary_dim, rad_weight tok_embd, rad_weight lm_head,
                 const HyperConn::Config& hc, int draft_depth, const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int MtpHcBlock::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup kv,
                               rad_kvgroup* qsa_bk, int64_t rotary_dim, rad_weight tok_embd,
                               rad_weight lm_head, const HyperConn::Config& hc, int draft_depth,
                               const Wire& wire, const Src& src) {
    g = geom; w = wire; layer = l; w_tok = tok_embd; w_lm = lm_head;
    rows = g.max_tok; depth = draft_depth;
    /* The pre-fc norm over the wide stream is GROUPED, one rsqrt a sub-stream, which is what every
     * gated residual in this architecture does. A flat norm fits the same [hc*n] gain and no
     * per-op oracle can tell the two apart; the acceptance rate can -- docs/OPS.md's `mtp_enter`
     * row. */
    ngroup = hc.hc;
    if (rows <= 0 || !w_tok || !w_lm || hc.hc <= 1) return RAD_E_INVAL;
    if (!w.src || !w.hin || !w.hw || !w.e || !w.hout) return RAD_E_INVAL;
    const int64_t HN = hc.hc * g.n_embd;

    /* ---- the entry's four weights. SHARD_NONE on all of them: two are per-channel gains over a
     * replicated stream and the two fc matrices read and write that same stream, so a column shard
     * would need an all-reduce to save 13 MB a rank -- the wrong side of the trade at one launch
     * per draft round. */
    const char* nph = nm.f("mtp.pre_fc_norm_hidden.weight");
    const char* npe = nm.f("mtp.pre_fc_norm_embedding.weight");
    const char* nfh = nm.f("mtp.fc_hidden.weight");
    const char* nfe = nm.f("mtp.fc_embedding.weight");
    RAD_ARCH_TRY(map_copy(b, nph, src.pre_norm_hidden));
    RAD_ARCH_TRY(map_copy(b, npe, src.pre_norm_embedding));
    RAD_ARCH_TRY(map_copy(b, nfh, src.fc_hidden));
    RAD_ARCH_TRY(map_copy(b, nfe, src.fc_embedding));
    /* THE TWO ENTRY MATRICES ARE WHATEVER THEIR ENCODING IS -- bf16, or E4M3 rows with a scale a
     * group of each row (OPS.md's `fc` on mtp_enter) -- declared over every plane at the codes'
     * dtype. It only ever changes the DRAFTS: the head proposes and the trunk verifies. */
    const uint32_t fdt = weight_codes_dtype(b, nfh, RAD_BF16);
    if (weight_codes_dtype(b, nfe, RAD_BF16) != fdt || (fdt != RAD_BF16 && fdt != RAD_F8E4M3)) {
        fprintf(stderr, "radiance: the MTP head's fc_hidden is %s and fc_embedding %s; the entry "
                        "reads both as bf16 or both as E4M3 rows\n", rad_dtype_name(fdt),
                rad_dtype_name(weight_codes_dtype(b, nfe, RAD_BF16)));
        return RAD_E_DTYPE;
    }
    w_pre_h = decl_w(b, nph, RAD_F32, {HN}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_pre_e = decl_w(b, npe, RAD_F32, {g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(l));
    w_fc_h  = decl_w(b, nfh, fdt, {g.n_embd, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(l));
    w_fc_e  = decl_w(b, nfe, fdt, {g.n_embd, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(l));
    if (!w_pre_h || !w_pre_e || !w_fc_h || !w_fc_e) return RAD_E_INVAL;
    if (w.tree_config.pruned()) {
        if (!w.dq || !w.ds || !w.dsum || !w.dlog || !w.didx || !w.dval)
            return RAD_E_UNSUPPORTED;
        if (!w.root_carry || !w.pruned_meta || !w.pruned_score || !w.frontier_gather ||
            !w.frontier_kv || !w.frontier_slots || !w.frontier_prefix ||
            !w.frontier_ancestors || w.frontier_rows < 8) return RAD_E_INVAL;
        for (int r = 2; r <= 5; ++r) {
            op_pruned_prepare[r - 2] = rw(b, RAD_OP(b, "mtp_pruned_prepare",
                RAD_PARAMS(RAD_INT("M", g.max_seqs), RAD_INT("round", r),
                           RAD_INT("n_vocab", g.n_vocab_all)), RAD_NOWEIGHTS),
                {w.pruned_meta}, {w.frontier_gather, w.frontier_ancestors,
                                  w.frontier_slots, w.frontier_prefix});
            if (!op_pruned_prepare[r - 2]) return RAD_E_INVAL;
        }
        op_root_gather = rw(b, RAD_OP(b, "gather_rows",
            RAD_PARAMS(RAD_RANGE("M", 1, w.frontier_rows), RAD_INT("n", HN),
                       RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS),
            {w.root_carry, w.frontier_gather}, {w.hin});
        op_parent_gather = rw(b, RAD_OP(b, "gather_rows",
            RAD_PARAMS(RAD_RANGE("M", 1, w.frontier_rows), RAD_INT("n", HN),
                       RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS),
            {w.hw, w.frontier_gather}, {w.hin});
        if (!op_root_gather || !op_parent_gather) return RAD_E_INVAL;
        attn.frontier_kv = w.frontier_kv;
        attn.frontier_slots = w.frontier_slots;
        attn.frontier_prefix = w.frontier_prefix;
        attn.frontier_ancestors = w.frontier_ancestors;
        attn.frontier_meta = w.pruned_meta;
        attn.frontier_rows = w.frontier_rows;
    }

    /* A SECOND gather_rows and a SECOND embed_lookup over the trunk's own buffers and weights.
     * They are separate ops rather than re-issues of the trunk's because an op's read and write
     * sets are what the buffer planner reads for liveness, and these write somewhere else. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", HN),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                   {w.src}, {w.hin});
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.e});
    if (media) RAD_ARCH_TRY(mrows.declare(b, rows, g.n_embd, w.e));
    /* The trunk's table is vocab-sharded, so this gather is a partial one and needs the same exact
     * reduction the trunk's does. Its own op because it writes a different buffer. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, rows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.e}, {w.e});

    op_enter = rw(b, RAD_OP(b, "mtp_enter",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                RAD_INT("hc", hc.hc), RAD_F64("eps", g.eps),
                                RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd),
                                RAD_INT("ngroup", ngroup),
                                RAD_STR("fc", fdt == RAD_F8E4M3 ? "e4m3" : "bf16")),
                     RAD_WEIGHTS(w_pre_h, w_pre_e, w_fc_h, w_fc_e)),
                 {w.hin, w.e}, {w.hw});
    if (!op_gather || !op_embed || !op_enter) return RAD_E_INVAL;

    /* ---- the layer. One gated residual, a gated full-attention block with its indexer, a second
     * gated residual and the routed feed-forward -- a trunk full-attention layer, at layer index
     * n_layer, and declared in exactly the order step() issues it in. */
    RAD_ARCH_TRY(hc_attn.declare(b, nm, g, nm.f("mtp.attn_hc"), hc, HyperConn::Wire{w.hw, w.x, w.inj},
                                 src.hc_attn));
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv));
    attn.ar_out_take = hc_attn.ar_take();
    RAD_ARCH_TRY(attn.declare(b, nm, g, l, kv, rotary_dim, w.attn, src.attn,
                              false, false, kArNone, hc_attn.op_ar_write != 0));
    if (src.qsa.qk_proj)
        RAD_ARCH_TRY(qsa.declare(b, nm, g, l, src.qsa, w.qsa, w.attn.h, qsa_bk));
    /* ---- draft round 1's anchor selection (RadBatch::draft_out_ids):
     * Gathered into the carry, one row a drafting sequence in round order, and copied back
     * to the front of `hw` for the mixer and MoE.
     * Declared BEFORE hc_mlp and mlp so the buffer planner treats w.hin as live across the MoE
     * layer, preventing any MoE transient buffers from aliasing over the carry! */
    op_dsel = rw(b, RAD_OP(b, "gather_rows",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", HN),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.hw}, {w.hin});
    op_dback = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", HN),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.hin}, {w.hw});
    if (!op_dsel || !op_dback) return RAD_E_INVAL;

    /* The rotated expert input rides in this read, as it does on the trunk. */
    HyperConn::Config mcfg = hc;
    HyperConn::Wire   mwire{w.hw, w.x, w.inj};
    if (src.mlp.cfg.expert_rot) { mcfg.rotate = 1; mwire.rq = w.mlp.hr.q; mwire.rs = w.mlp.hr.s; }
    RAD_ARCH_TRY(hc_mlp.declare(b, nm, g, nm.f("mtp.ffn_hc"), mcfg, mwire, src.hc_mlp));
    mlp.ar_out_take = hc_mlp.ar_take();
    mlp.hr_in = mcfg.rotate != 0;
    RAD_ARCH_TRY(mlp.declare(b, nm, g, l, w.mlp, src.mlp, false, false, kArNone,
                             hc_mlp.op_ar_write != 0));
    /* The write behind the block takes its gather where it can, as on the trunk. */
    {
        HyperConn::GatherSrc gs{};
        gs.ye     = mlp.w.edn;
        gs.ew     = mlp.w.ew;
        gs.sorted = mlp.w.sorted;
        if (mlp.gfold) { gs.sh = mlp.w.sout; gs.sg = mlp.w.sgate; }
        gs.top_k  = mlp.c.top_k;
        gs.rows   = mlp.c.rows;
        RAD_ARCH_TRY(hc_mlp.declare_gather(b, gs));
        mlp.gather_out_rows  = hc_mlp.gather_rows;
        mlp.gather_out_rows6 = hc_mlp.gather_rows6;
    }

    /* ---- the carry. See the header: `mtp_enter` cannot be issued in place. */
    op_carry = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", HN),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.hw}, {w.hin});
    if (w.tree_config.pruned()) {
        op_root_carry = rw(b, RAD_OP(b, "cast",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n", HN),
                       RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)), RAD_NOWEIGHTS),
            {w.hin}, {w.root_carry});
        if (!op_root_carry) return RAD_E_INVAL;
    }

    /* ---- the head's own mixer: the connection with no write half, which collapses its wide
     * stream to the one hidden state the lm_head reads. No fp8 twin -- that input is bf16. */
    {
        HyperConn::Config mc = hc;
        mc.inject = 0;
        mc.quant  = 0;
        HyperConn::Wire mixw{};
        mixw.h = w.hw; mixw.x.x = w.hout;      /* q/s deliberately absent */
        RAD_ARCH_TRY(mixer.declare(b, nm, g, nm.f("mtp.output_hc"), mc, mixw, src.hc_mix));
    }

    /* The trunk's lm_head over the head's hidden state, banded at the logits buffer's own extent:
     * only a DRAFT round reaches this and a draft round is one row a sequence. */
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                 RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_lm)),
                  {w.hout}, {w.logits});
    if (!op_carry || !op_logits) return RAD_E_INVAL;

    /* ---- THE 2-BIT DRAFT HEAD. rad_block_mtp_fp8.h argues it in full; the short version is that
     * `logits_gemm` is already at the memory roofline, so its only lever is FEWER BYTES -- and
     * a speculative step runs the projection once to verify and once PER DRAFT ROUND. A draft is a
     * proposal the target verifies with its own head, so a worse draft costs acceptance and
     * cannot change a token the model emits. */
    enum { MTP_W2_GROUP = 128 };
    /* A DRAFT NEEDS ONE TOKEN AND THE TOP-R IS WIDER THAN THAT; kDraftTopR says why. There is also
     * a HARD LOWER BOUND of 2 from the all-gather: `ar_gather_2rank` places rank r's contribution
     * row-interleaved and requires the row to be a whole number of 16-byte words, and a pair is
     * (int32 id, float value) -- so R = 1 is an 8-byte row and is refused, while R = 2 is 16. The
     * engine reads column 0 through `proposal_pitch`, which is what that field is for. */
    q2 = w.dq && w.ds && w.dsum && w.dlog && w.didx && w.dval;
    if (q2) {
        /* AND IT IS VOCAB-PARALLEL, which is 159 MB a round rather than 318. The head does not go
         * through the sampler, so its argmax is a `row_topk` over one rank's plane -- and a
         * sharded plane alone would make every rank propose only tokens from its own half. That
         * is what `row_topk`'s `vocab_off` and `row_topk_merge` are for, and they are the same
         * three ops arch/common/rad_block_dspark.h already drives: a top-R reporting GLOBAL ids,
         * an all_gather of (id, value) pairs, and a merge that finds exactly the top-R an
         * unsharded head would have. */
        const bool dh_shard = head_sharded(g) && w.dpair && w.dgath;
        const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
        const int64_t hrows = dh_shard ? g.n_vocab : g.n_vocab_all;
        /* ITS OWN STORED WEIGHT, MADE FROM THE TRUNK'S lm_head by the recipe: two-bit codes with
         * an f16 scale and a u8 zero point a group of 128 (`u2*f16+u8[1x128]`), its name map
         * declared by mtp_draft_head_q2. The GEMM takes the codes as one operand and the scale
         * and zero planes as the other. */
        const char* dhn = head_name(nm);
        w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {hrows, g.n_embd},
                          RAD_ACCESS_PER_TOKEN, dh_split, grp_layer(l));
        w_dhs = decl_view(b, nm.f("mtp.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                          {hrows, g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split, grp_layer(l));
        if (!w_dh || !w_dhs) return RAD_E_INVAL;

        op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n", g.n_embd), RAD_INT("group", MTP_W2_GROUP),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.hout}, {w.dq, w.ds, w.dsum});
        op_dlogits = rw(b, RAD_OP(b, "gemm_nt_q",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                      RAD_INT("N", hrows), RAD_INT("K", g.n_embd),
                                      RAD_INT("group", MTP_W2_GROUP), RAD_STR("dtype", "w2a8")),
                           RAD_WEIGHTS(w_dh, w_dhs)),
                       {w.dq, w.ds, w.dsum}, {w.dlog});
        /* Sharded, the top-R runs over THIS RANK'S plane and reports GLOBAL ids, and it also
         * writes the pair plane the gather moves. `vocab_off` is omitted entirely when the head is
         * replicated: an optional parameter present with the value 0 is not the same declaration
         * as an absent one. */
        op_dtopk = dh_shard
            ? rw(b, RAD_OP(b, "row_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("N", hrows),
                               RAD_INT("R", kDraftTopR), RAD_INT("vocab_off", g.vocab_off),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                 {w.dlog}, {w.didx, w.dval})
            : rw(b, RAD_OP(b, "row_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("N", hrows),
                               RAD_INT("R", kDraftTopR), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                 {w.dlog}, {w.didx, w.dval});
        /* ---- THE COARSE HEAD'S CANDIDATES, RESCORED EXACTLY.
         *
         * `w_lm` is the trunk's lm_head: already declared, already resident, already this
         * block's (it is what the fallback's `op_logits` reads), so the rerank costs no weight.
         * It reads R rows of it a draft row, in the stored form the trunk's logits_gemm reads:
         * libr4d's rerank takes the same layout as its fp8 logits GEMM, so one stored head serves
         * both, and a kernel that reads neither form is not selected for either.
         *
         * IT NEEDS NO COLLECTIVE OF ITS OWN. Sharded, `row_topk` picked from THIS RANK'S plane and
         * reported global ids, so every candidate it hands over is one this rank owns and can
         * score; the existing all_gather and merge then find the global winner over exact values
         * instead of coarse ones. The op writes back over `didx`/`dval`/`dpair` in place, which is
         * what keeps that chain unchanged.
         *
         * AND THAT OWNERSHIP IS WHY A REPLICATED DRAFT HEAD ABOVE ONE RANK SKIPS IT. A vocabulary
         * that does not shard cleanly (vocab_head_sharded) replicates the DRAFT head while the
         * trunk's lm_head stays row-sharded, so `row_topk` then reports ids from the whole
         * vocabulary and half of them are not this rank's to score. The coarse head's own answer
         * stands, unreranked, and the rerank sits out.
         *
         * `vocab_off` is passed always and not only when sharded: unlike `row_topk`, this op has
         * one form, and at world 1 the offset is 0 and `n_vocab` is the whole vocabulary, which is
         * the same declaration either way. */
        const bool rerank = dh_shard || g.world <= 1;
        if (rerank) {
            op_drerank = RAD_OP(b, "logit_rerank",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                       RAD_INT("R", kDraftTopR), RAD_INT("n_vocab", g.n_vocab),
                                       RAD_INT("n_embd", g.n_embd),
                                       RAD_INT("vocab_off", g.vocab_off),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_lm));
            if (!op_drerank) return RAD_E_INVAL;
            /* Arrays and not rw()'s braced lists: the write set is conditional and a braced list
             * inside a ternary is a temporary -- rad_fp8.h's NormQuantFP8 records the same. */
            rad_buf rd[2] = { w.hout, w.didx };
            rad_buf wr[3] = { w.didx, w.dval, w.dpair };
            rad_op_reads (b, op_drerank, rd, 2);
            rad_op_writes(b, op_drerank, wr, dh_shard ? 3 : 2);
        }
        if (dh_shard) {
            /* ROW-INTERLEAVED: all eight exact pairs from every rank stay side by side. */
            op_dgather = rw(b, RAD_OP(b, "all_gather",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_INT("numel", g.max_logit_rows * kDraftGatherR * 2),
                                          RAD_STR("dtype", "f32"),
                                          RAD_INT("row", kDraftGatherR * 2)),
                               RAD_NOWEIGHTS),
                           {w.dpair}, {w.dgath});
            if (!w.packed_tree || g.world != 2) {
                op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                             RAD_INT("R", kDraftGatherR),
                                             RAD_INT("world_size", g.world),
                                             RAD_STR("dtype", g.dtype)),
                                  RAD_NOWEIGHTS),
                              {w.dgath}, {w.didx, w.dval});
                if (!op_dmerge) return RAD_E_INVAL;
            }
            if (!op_dgather) return RAD_E_INVAL;
        }
        if (!op_dquant || !op_dlogits || !op_dtopk) return RAD_E_INVAL;
    }
    if (spechub_enabled()) {
        if (!q2 || !op_dgather || !op_drerank || g.world != 2 || w.tree_capacity != 3 ||
            !w.spechub_draws || !w.spechub_state || w.tree_config.policy != 0)
            return RAD_E_UNSUPPORTED;
        op_spechub = rw(b, RAD_OP(b, "spechub_propose",
            RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n_vocab", g.n_vocab_all),
                       RAD_INT("world_size", 2)), RAD_NOWEIGHTS),
            {w.dgath, w.spechub_draws}, {w.spechub_state, w.packed_tree, w.didx, w.dval});
        if (!op_spechub) return RAD_E_INVAL;
    }
    /* THE LINEAR COUPLED PROPOSAL: one op a round (the ABI freezes `round`), after the final
     * rerank/merge, on the shortlist that would otherwise give column 0. Linear chain only. */
    if (mtp_coupled_enabled()) {
        if (!q2 || !op_drerank || (!op_dmerge && g.world > 1) || w.packed_tree ||
            !w.couple_draws || !w.couple_state || depth < 1 || depth > RAD_MTP_COUPLE_MAX_ROUNDS)
            return RAD_E_UNSUPPORTED;
        for (int round = 1; round <= depth; ++round) {
            op_couple[round - 1] = rw(b, RAD_OP(b, "mtp_couple_propose",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n_vocab", g.n_vocab_all),
                           RAD_INT("round", round), RAD_INT("max_rounds", depth)), RAD_NOWEIGHTS),
                {w.didx, w.dval, w.couple_draws, w.couple_state}, {w.didx, w.couple_state});
            if (!op_couple[round - 1]) return RAD_E_INVAL;
        }
    }
    if (w.packed_tree && !w.tree_config.pruned() && !spechub_enabled()) {
        /* Fixed 8-round policy: 8 spine + 56 alternates + root = 65 nodes. */
        const int64_t max_nodes = w.tree_config.rounds == 8 ? 65 : 16;
        if (!q2 || !w.cand_hist || w.tree_capacity < 2 || w.tree_capacity > max_nodes)
            return RAD_E_INVAL;
        const bool scored_op = w.tree_config.policy != 0 || w.tree_config.rounds == 5;
        if (scored_op && (!op_dgather || g.world != 2 || w.tree_capacity > 8 || !op_drerank))
            return RAD_E_UNSUPPORTED;
        if (w.tree_config.policy != 0) {
            if (!w.log_hist || !w.lse_local || !w.lse_gathered) return RAD_E_INVAL;
            op_hybrid_lse = rw(b, RAD_OP(b, "mtp_hybrid_lse",
                RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                           RAD_INT("N", head_rows(g)), RAD_INT("n_vocab", g.n_vocab_all),
                           RAD_INT("vocab_off", g.vocab_off)), RAD_NOWEIGHTS),
                {w.dlog, w.didx, w.dval}, {w.lse_local});
            /* Rerank's dense 64-byte pair row is frozen. A separate aligned 16-byte
             * row avoids changing its strides or the existing candidate collective. */
            op_lse_gather = rw(b, RAD_OP(b, "all_gather",
                RAD_PARAMS(RAD_INT("world_size", 2),
                           RAD_INT("numel", g.max_logit_rows * 4),
                           RAD_STR("dtype", "f32"), RAD_INT("row", 4)), RAD_NOWEIGHTS),
                {w.lse_local}, {w.lse_gathered});
            if (!op_hybrid_lse || !op_lse_gather) return RAD_E_INVAL;
        }
        /* The ABI freezes scalar geometry at declaration; each pass selects its round's op. */
        for (int round = 1; round <= w.tree_config.rounds; ++round) {
            if (scored_op) {
                const double* cost = w.tree_config.costs;
                op_mtp_scored_tree[round - 1] = RAD_OP(b, "mtp_scored_merge_tree_accum",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                               RAD_INT("pitch", kDraftTopR), RAD_INT("capacity", w.tree_capacity),
                               RAD_INT("max_rounds", w.tree_config.rounds),
                               RAD_INT("n_vocab", g.n_vocab_all), RAD_INT("round", round),
                               RAD_INT("world_size", 2), RAD_INT("policy", w.tree_config.policy),
                               RAD_F64("cost0", cost[0]), RAD_F64("cost1", cost[1]),
                               RAD_F64("cost2", cost[2]), RAD_F64("cost3", cost[3]),
                               RAD_F64("cost4", cost[4]), RAD_F64("cost5", cost[5]),
                               RAD_F64("cost6", cost[6]), RAD_F64("cost7", cost[7])),
                    RAD_NOWEIGHTS);
                const rad_op op = op_mtp_scored_tree[round - 1];
                if (w.tree_config.policy != 0)
                    op_mtp_scored_tree[round - 1] =
                        rw(b, op, {w.dgath, w.lse_gathered, w.cand_hist, w.log_hist, w.packed_tree},
                           {w.didx, w.dval, w.cand_hist, w.log_hist, w.packed_tree});
                else
                    op_mtp_scored_tree[round - 1] =
                        rw(b, op, {w.dgath, w.cand_hist, w.packed_tree},
                           {w.didx, w.dval, w.cand_hist, w.packed_tree});
                if (!op_mtp_scored_tree[round - 1]) return RAD_E_INVAL;
            } else if (op_dgather && g.world == 2) {
                op_mtp_merge_tree[round - 1] = rw(b, RAD_OP(b, "mtp_merge_tree_accum",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                               RAD_INT("pitch", kDraftTopR), RAD_INT("capacity", w.tree_capacity),
                               RAD_INT("max_rounds", w.tree_config.rounds == 8 ? 8 : 3),
                               RAD_INT("n_vocab", g.n_vocab_all),
                               RAD_INT("round", round), RAD_INT("world_size", 2)), RAD_NOWEIGHTS),
                    {w.dgath, w.cand_hist, w.packed_tree},
                    {w.didx, w.dval, w.cand_hist, w.packed_tree});
                if (!op_mtp_merge_tree[round - 1]) return RAD_E_INVAL;
            } else {
                op_mtp_tree[round - 1] = rw(b, RAD_OP(b, "mtp_tree_accum",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                               RAD_INT("pitch", kDraftTopR), RAD_INT("capacity", w.tree_capacity),
                               RAD_INT("max_rounds", w.tree_config.rounds == 8 ? 8 : 3),
                               RAD_INT("n_vocab", g.n_vocab_all),
                               RAD_INT("round", round)), RAD_NOWEIGHTS),
                    {w.didx, w.cand_hist, w.packed_tree}, {w.cand_hist, w.packed_tree});
                if (!op_mtp_tree[round - 1]) return RAD_E_INVAL;
            }
        }
    }
    if (w.tree_config.pruned()) {
        if (!q2 || !op_dgather || !op_drerank || g.world != 2 || w.tree_capacity != 8)
            return RAD_E_UNSUPPORTED;
        for (int r = 1; r <= 5; ++r) {
            op_pruned_expand[r - 1] = rw(b, RAD_OP(b, "mtp_pruned_expand",
                RAD_PARAMS(RAD_INT("M", g.max_seqs), RAD_INT("round", r),
                           RAD_INT("n_vocab", g.n_vocab_all)), RAD_NOWEIGHTS),
                {w.dgath, w.pruned_meta, w.pruned_score},
                {w.pruned_meta, w.pruned_score, w.packed_tree});
            if (!op_pruned_expand[r - 1]) return RAD_E_INVAL;
        }
        op_pruned_pack = rw(b, RAD_OP(b, "mtp_pruned_pack",
            RAD_PARAMS(RAD_INT("M", g.max_seqs), RAD_F64("beta", MtpTreeConfig::beta)),
            RAD_NOWEIGHTS),
            {w.pruned_meta, w.pruned_score}, {w.packed_tree});
        if (!op_pruned_pack) return RAD_E_INVAL;
    }
    /* THE CARRY STAYS LIVE THROUGH THE WHOLE HEAD. A pass writes `hin` for the round after it
     * before its mixer and lm_head run; declared as used by the carry alone, the buffer planner
     * would let the ops after it take its bytes, and the next round would continue from what they
     * wrote. Nothing fails: acceptance falls. Read by the head's last op as well, it survives
     * every pass of the step. */
    {
        rad_op last = op_logits;
        if (q2) last = op_dmerge ? op_dmerge : op_drerank ? op_drerank : op_dtopk;
        if (op_mtp_merge_tree[w.tree_config.rounds - 1])
            last = op_mtp_merge_tree[w.tree_config.rounds - 1];
        if (op_mtp_scored_tree[w.tree_config.rounds - 1])
            last = op_mtp_scored_tree[w.tree_config.rounds - 1];
        if (op_spechub) last = op_spechub;
        if (op_couple[0]) last = op_couple[depth - 1];
        if (rad_op_reads(b, last, &w.hin, 1) < 0) return RAD_E_INVAL;
        if (op_pruned_pack) {
            const rad_buf carry[] = {w.hin, w.hw};
            if (rad_op_reads(b, op_pruned_pack, carry, 2) < 0) return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

/* One pass of the head. The batch is the one the scheduler built for it: n_tok rows, `token_ids`
 * already shifted one position on (sched/batch.cpp, token_at), `positions` the head's own indices,
 * and the KV group batch carrying the slot each row writes at -- so every op below runs at n_tok
 * rows and the attention layer needs no special case at all. */
inline void MtpHcBlock::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int     r = batch->draft_pass;
    const bool pruned = w.tree_config.pruned();
    if (pruned && r > 0) {
        if (r < 2 || r > 5 || T != batch->n_seq * 8 || !batch->draft_out_ids ||
            batch->n_draft_out != batch->n_seq || T > w.frontier_rows) {
            rad_step_fail(c, "MTP pruned frontier has invalid round or request geometry");
            return;
        }
        RAD_ISSUE(c, op_pruned_prepare[r - 2],
                  praw(batch->draft_out_ids, RAD_I32, batch->n_seq), RAD_B(w.pruned_meta),
                  praw(batch->token_ids, RAD_I32, T), praw(batch->positions, RAD_I32, T),
                  brows(w.frontier_gather, T), brows(w.frontier_ancestors, T),
                  brows(w.frontier_slots, T), brows(w.frontier_prefix, T));
        RAD_ISSUE_N(c, r == 2 ? op_root_gather : op_parent_gather, T,
                    r == 2 ? RAD_B(w.root_carry) : RAD_B(w.hw),
                    brows(w.frontier_gather, T), brows(w.hin, T));
    }

    /* The history pass continues from the TRUNK's wide stream; the rounds after it continue from
     * the head's, which the carry left in the same buffer. */
    if (r == 1) {
        rad_step_fail(c, "the MTP head was handed a draft-round-1 pass: that round is carried by "
                         "the history pass (RadBatch::draft_out_ids) and has no pass of its own");
        return;
    }
    if (r < 0 && batch->n_out > 0)
        RAD_ISSUE_N(c, op_gather, batch->n_out,
                    RAD_B(w.src), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(w.hin, batch->n_out));

    RAD_ISSUE(c, op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.e, T));
    mrows.step(c, batch, w.e, T);
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.e, T), RAD_NONE);

    RAD_ISSUE(c, op_enter, brows(w.hin, T), brows(w.e, T), RAD_W(w_pre_h), RAD_W(w_pre_e),
              RAD_W(w_fc_h), RAD_W(w_fc_e), brows(w.hw, T));

    hc_attn.read(c, T, 0, T);
    /* THE INDEXER RUNS FIRST, because the attention reads its selection. It can: this block's
     * input is normed and quantised by the connection above, so `w.h` is already what it will be
     * when the attention runs. */
    if (!pruned || r < 0) qsa.step(c, attn.w.h, batch);
    attn.step(c, batch);

    /* A HISTORY PASS THAT DRAFTS NOTHING ENDS HERE. It exists to leave the head's state behind --
     * the K and V the attention just stored and the indexer's block keys and tail -- and both are
     * written above. Everything below computes the head's stream, which only a drafting row
     * reads: the experts, the two connections around them and the lm_head. Every prefill chunk
     * but a prompt's last is such a pass, so on those the head costs its attention alone.
     * `n_draft_out` is the batch's and the same on every rank, so the collectives skipped here
     * are skipped on all of them. (From Code000's opt33) */
    if (r < 0 && batch->n_draft_out <= 0) return;

    hc_attn.write(c, T, 0, T);

    hc_mlp.read(c, T, 0, T);
    mlp.step(c, batch);
    hc_mlp.write(c, T, 0, T);

    /* THE CARRY, and only where a round after this one will read it. The last round of a group
     * is followed by a trunk step. A history pass that drafts takes round 1's rows out of its own
     * -- into the carry, and back to the front of `hw` for the mixer below. */
    int64_t L = T;                      /* the rows the lm_head runs over */
    if (r < 0) {
        L = batch->n_draft_out;
        RAD_ISSUE_N(c, op_dsel, L, RAD_B(w.hw), praw(batch->draft_out_ids, RAD_I32, L),
                    brows(w.hin, L));
        RAD_ISSUE_N(c, op_dback, L, brows(w.hin, L), brows(w.hw, L));
        if (pruned)
            RAD_ISSUE_N(c, op_root_carry, L, brows(w.hin, L), brows(w.root_carry, L));
    } else if (!pruned && r < depth) {
        RAD_ISSUE_N(c, op_carry, T, brows(w.hw, T), brows(w.hin, T));
    }

    mixer.read(c, L, 0, L);
    if (q2) {
        /* `dsum` is GROUP-major [n_embd/group][ceil(L/16)*16] and not row-major, so it is passed
         * whole: there is no row slice of it that means anything. */
        RAD_ISSUE_N(c, op_dquant, L, brows(w.hout, L), brows(w.dq, L), brows(w.ds, L),
                    RAD_B(w.dsum));
        /* The trailing `b_ref` is mxfp4's and absent here; an absent optional operand is a null
         * tensor in a full-length list, not a shorter list (rad_fp8.h says why). */
        RAD_ISSUE_N(c, op_dlogits, L, brows(w.dq, L), brows(w.ds, L), RAD_W(w_dh), RAD_W(w_dhs),
                    brows(w.dlog, L), RAD_B(w.dsum), RAD_NONE);
        if (op_dgather) {
            RAD_ISSUE_N(c, op_dtopk, L, brows(w.dlog, L), brows(w.didx, L), brows(w.dval, L),
                        RAD_NONE);
            /* In place over didx/dval, and dpair carries the top kDraftGatherR pairs with EXACT
             * values for the all-gather. */
            if (op_drerank)
                RAD_ISSUE_N(c, op_drerank, L, brows(w.hout, L), RAD_W(w_lm), brows(w.didx, L),
                            brows(w.didx, L), brows(w.dval, L), brows(w.dpair, L));
            if (op_hybrid_lse) {
                RAD_ISSUE_N(c, op_hybrid_lse, L, brows(w.dlog, L), brows(w.didx, L),
                            brows(w.dval, L), brows(w.lse_local, L));
                RAD_ISSUE(c, op_lse_gather, brows(w.lse_local, L), brows(w.lse_gathered, L));
            }
            RAD_ISSUE(c, op_dgather, brows(w.dpair, L), brows(w.dgath, L));
            if (op_spechub) {
                if (r >= 0) {
                    rad_step_fail(c, "SpecHub only supports one history-carried MTP proposal round");
                    return;
                }
                RAD_ISSUE_N(c, op_spechub, L, brows(w.dgath, L), brows(w.spechub_draws, L),
                    praw(batch->token_ids, RAD_I32, T),
                    praw(batch->draft_out_ids, RAD_I32, L), brows(w.spechub_state, L),
                    brows(w.packed_tree, L), brows(w.didx, L), brows(w.dval, L));
            } else if (pruned) {
                const int round = r < 0 ? 1 : r;
                RAD_ISSUE(c, op_pruned_expand[round - 1], brows(w.dgath, L),
                    r < 0 ? RAD_NONE : praw(batch->draft_out_ids, RAD_I32, batch->n_seq),
                    r < 0 ? praw(batch->token_ids, RAD_I32, T) : RAD_NONE,
                    r < 0 ? praw(batch->draft_out_ids, RAD_I32, L) : RAD_NONE,
                    r < 0 ? praw(batch->positions, RAD_I32, T) : RAD_NONE,
                    RAD_B(w.pruned_meta), RAD_B(w.pruned_score), RAD_B(w.packed_tree));
                if (r == 5)
                    RAD_ISSUE(c, op_pruned_pack,
                        praw(batch->draft_out_ids, RAD_I32, batch->n_seq),
                        RAD_B(w.pruned_meta), RAD_B(w.pruned_score), RAD_B(w.packed_tree));
            } else if (op_mtp_scored_tree[0]) {
                const int round = r < 0 ? 1 : r;
                if (round < 1 || round > w.tree_config.rounds) {
                    rad_step_fail(c, "MTP scored tree round exceeds its startup history bound");
                    return;
                }
                RAD_ISSUE_N(c, op_mtp_scored_tree[round - 1], L, brows(w.dgath, L),
                            w.tree_config.policy != 0 ? brows(w.lse_gathered, L) : RAD_NONE,
                            brows(w.didx, L), brows(w.dval, L),
                            r < 0 ? praw(batch->token_ids, RAD_I32, T) : RAD_NONE,
                            r < 0 ? praw(batch->draft_out_ids, RAD_I32, L) : RAD_NONE,
                            RAD_B(w.cand_hist), w.tree_config.policy != 0 ? RAD_B(w.log_hist) : RAD_NONE,
                            brows(w.packed_tree, L));
            } else if (op_mtp_merge_tree[0]) {
                const int round = r < 0 ? 1 : r;
                if (round < 1 || round > w.tree_config.rounds) {
                    rad_step_fail(c, "MTP tree round exceeds its startup history bound");
                    return;
                }
                RAD_ISSUE_N(c, op_mtp_merge_tree[round - 1], L, brows(w.dgath, L),
                            brows(w.didx, L), brows(w.dval, L),
                            r < 0 ? praw(batch->token_ids, RAD_I32, T) : RAD_NONE,
                            r < 0 ? praw(batch->draft_out_ids, RAD_I32, L) : RAD_NONE,
                            RAD_B(w.cand_hist), brows(w.packed_tree, L));
            } else {
                RAD_ISSUE_N(c, op_dmerge, L, brows(w.dgath, L), brows(w.didx, L), brows(w.dval, L));
            }
        } else {
            RAD_ISSUE_N(c, op_dtopk, L, brows(w.dlog, L), brows(w.didx, L), brows(w.dval, L),
                        RAD_NONE);   /* the `pairs` slot, absent: operands are positional */
            if (op_drerank)
                RAD_ISSUE_N(c, op_drerank, L, brows(w.hout, L), RAD_W(w_lm), brows(w.didx, L),
                            brows(w.didx, L), brows(w.dval, L), RAD_NONE);
        }
        /* Rows are the drafting sequences in draft order in every pass, and seq_ids[row] is the
         * card the advance writes; the op refuses (status -1) a row whose staged draw names another. */
        if (op_couple[0]) {
            const int round = r < 0 ? 1 : r;
            if (round < 1 || round > depth) {
                rad_step_fail(c, "MTP coupled round exceeds the declared draft depth");
                return;
            }
            RAD_ISSUE_N(c, op_couple[round - 1], L, brows(w.didx, L), brows(w.dval, L),
                        praw(batch->seq_ids, RAD_I32, batch->n_seq), brows(w.couple_draws, L),
                        RAD_B(w.couple_state));
        }
    } else {
        RAD_ISSUE_N(c, op_logits, L, brows(w.hout, L), RAD_W(w_lm), brows(w.logits, L));
    }
    if (op_mtp_tree[0]) {
        const int round = r < 0 ? 1 : r;
        if (round < 1 || round > w.tree_config.rounds) {
            rad_step_fail(c, "MTP tree round exceeds its startup history bound");
            return;
        }
        RAD_ISSUE_N(c, op_mtp_tree[round - 1], L, brows(w.didx, L),
                    r < 0 ? praw(batch->token_ids, RAD_I32, T) : RAD_NONE,
                    r < 0 ? praw(batch->draft_out_ids, RAD_I32, L) : RAD_NONE,
                    RAD_B(w.cand_hist), brows(w.packed_tree, L));
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MTP_HC_H */
