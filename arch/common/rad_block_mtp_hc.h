/* rad_block_mtp_hc.h -- the multi-token-prediction head of a model whose residual stream is `hc`
 * times the model width.
 *
 * ============================== DIFFERENCES FROM rad_block_mtp_fp8.h ==============================
 *
 * That file serves Qwen3.5 / Qwen3-Next / Qwen3.6, whose head is a pre-norm layer reading the
 * trunk's [n_embd] hidden state. Qwen4-Exp's head has the same indexing and passes but lives in
 * the wide stream:
 *
 *   10240-wide input             `mtp.pre_fc_norm_hidden.weight` is [hc * n_embd]: the head
 *                               continues from the trunk's wide stream before the output mixer,
 *                               not `last_hidden_state`.
 *   two fc matrices              `mtp.fc_hidden` and `mtp.fc_embedding`, both [n_embd, n_embd], so
 *                               no concatenation: the hidden matrix is applied to each
 *                               sub-stream and the embedding product broadcast into all of them
 *                               (docs/OPS.md's `mtp_enter` row).
 *   gated residuals, no norms    `mtp.layers.0.attn_hyper_connection` and `.mlp_hyper_connection`,
 *                               as in a trunk layer; no `input_layernorm` or
 *                               `post_attention_layernorm`.
 *   its own mixer                `mtp.hyper_connection_mixer`, a connection with no write half
 *                               that collapses the wide stream for the lm_head (the model's 98th
 *                               connection, shaped like the trunk's final one).
 *
 * ============================== THE INDEXING ==============================
 *
 *     head index i    reads  hidden(i) and embed(token at i + 1)
 *                     writes its K and V at KV position i, with rope position i
 *                     predicts the token at i + 2
 *
 * DeepSeek-V3's layout: the head runs over the trunk's position indices and the shift is only in
 * which token is embedded. An off-by-one shows only as collapsed acceptance.
 *
 * ============================== THE TWO KINDS OF PASS ==============================
 *
 * By the sign of `RadBatch::draft_pass` (details in rad_block_mtp_fp8.h):
 *
 *   < 0   History pass, one row per token, continuing from the trunk's wide stream. It writes the
 *         head's own K and V, which nothing else writes. It is also draft round 1 for drafting
 *         sequences: it ends in the lm_head over their rows (RadBatch::draft_out_ids) and hands
 *         their wide state on as the carry.
 *   > 1   Draft round, one row per sequence, continuing from the head's own stream.
 *
 * ============================== THE CARRY ==============================
 *
 * `hin` is what a round continues from and `hw` the head's own stream. They cannot share a buffer:
 * `mtp_enter` reads the whole wide row for every output channel, so in-place writes would clobber
 * unread input. (The pre-norm head's carry is written by a different op, `mtp.norm`; here the
 * round ends with the mlp connection's in-place write.) A `cast` copies `hw` to `hin` between
 * rounds (a 20 KB row at decode), skipped on the last round. A drafting history pass instead
 * gathers its round-1 rows and copies them back to the front of `hw`, where the mixer reads them.
 *
 * ============================== THE KV CACHE ==============================
 *
 * The head's attention is bound to the trunk's full-attention group as one more layer, at
 * absolute index n_layer, sharing its slot mapping and block table; a rejected draft's entry is
 * overwritten by the next step's token at the same address (see rad_block_mtp_fp8.h). The group's
 * page grows from 12 layers to 13 (8% more attention KV), only when speculation is declared.
 *
 * ============================== BUFFERS ==============================
 *
 * The layer reuses the trunk's working buffers (`attn_qg`, the expert planes, the narrow block
 * input), which are dead once the trunk step is committed; a second set would be 150 MiB. Only
 * the wide stream, the carry, the embedding and the logits path are its own.
 *
 * Invariant: declare this block after the trunk's last op. Declaration order is the buffer
 * planner's liveness (core/build/rad_bufplan.cpp); declared mid-trunk, the head would share bytes
 * with live trunk buffers (symptom: correct output, zero acceptance).
 */
#ifndef RAD_BLOCK_MTP_HC_H
#define RAD_BLOCK_MTP_HC_H

#include "rad_block_attn_gated_fp8.h"
#include "rad_block_hc.h"
#include "rad_block_moe_fp8.h"
#include "rad_block_mtp_fp8.h"     /* mtp_draft_head_q2(): one reading of the 2-bit head */
#include "rad_spechub.h"           /* RAD_MTP_COUPLE_*: the coupling wire */
#include "rad_spechub_config.h"    /* RADIANCE_MTP_COUPLED */
#include "rad_qsa.h"
#include "rad_block_vit.h"

namespace rad {
namespace arch {

/* Candidates the coarse 2-bit head proposes for exact rescoring (`logit_rerank` against the
 * full-precision head) before column 0 is read. The 2-bit head's top-1 differs from the exact one
 * often enough to cost acceptance; 8 is the reference implementation's value, which recovered
 * 99.63% agreement with the exact argmax, at R rows of the weight per draft row (40 KB against
 * 318 MB).
 *
 * Both 4 and 16 give lower acceptance than 8: acceptance is agreement with the trunk, while the
 * rescore maximises the head's own preference, so beyond some width the coarse ranking is a useful
 * restriction. */
static constexpr int64_t kDraftTopR = 8;

struct MtpHcBlock {
    struct Wire {
        /* [max_tok, hc * n_embd] the trunk's wide stream for the step just run; a pass names the
         * rows it wants through out_ids. */
        rad_buf src  = 0;
        rad_buf hin  = 0;   /* [max_tok, hc * n_embd]  what this round continues from */
        rad_buf hw   = 0;   /* [max_tok, hc * n_embd]  the head's own wide stream */
        rad_buf e    = 0;   /* [max_tok, n_embd]       the input token's embedding */
        rad_buf inj  = 0;   /* [max_tok, hc]           both connections' write gains */
        rad_buf hout = 0;   /* [max_logit_rows, n_embd]  the mixer's output, the lm_head's input */
        rad_buf logits = 0; /* [max_logit_rows, n_vocab] the trunk's, shared */
        ActFP8  x{};        /* the narrow block input, the trunk's */
        /* The 2-bit draft head's working set, sized at max_logit_rows (only drafting rows reach
         * it). */
        rad_buf dq = 0, ds = 0, dsum = 0, dlog = 0, didx = 0, dval = 0;
        /* Vocab-parallel pairs, declared only when the head is sharded: (int32 id, f32 value)
         * pairs stored as f32, this rank's top-R and the all-gathered world * R. */
        rad_buf dpair = 0, dgath = 0;
        /* RADIANCE_MTP_COUPLED: [max_seqs] draws and [depth, max_seqs] states, both PERSIST across
         * the serial passes (round 1 resets the logical stream, not the plane). */
        rad_buf couple_draws = 0, couple_state = 0;
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
    int     depth  = 0;    /* rounds this deployment runs; decides whether the carry is needed */
    int64_t ngroup = 0;    /* the pre-fc norm's groups: hc, one rsqrt per sub-stream */
    bool    q2     = false;
    /* The 2-bit draft head's name. The plugin passes it to mtp_draft_head_q2 (which declares the
     * map) and declares the head's buffers if that returns true. */
    static const char* head_name(Names& nm) { return nm.f("mtp.draft_head.weight"); }

    rad_weight w_tok = 0, w_lm = 0, w_fc_h = 0, w_fc_e = 0, w_pre_h = 0, w_pre_e = 0;
    rad_weight w_dh = 0, w_dhs = 0;

    HyperConn    hc_attn{}, hc_mlp{}, mixer{};
    AttnGatedFP8 attn{};
    QsaIndexer   qsa{};
    /* The indexer's window bounds for a pass wider than its score plane (QsaIndexer::step), set by
     * the caller before step(); null where no pass is. */
    const int32_t* qsa_win_cu = nullptr;
    MoeFP8       mlp{};

    rad_op op_gather = 0, op_embed = 0, op_embed_ar = 0, op_enter = 0, op_carry = 0;

    /* Media rows over the head's embedding (see MtpBlockFP8::media). Set before declare when the
     * program has an encoder. */

    bool       media = false;

    MediaRows  mrows{};
    rad_op op_dsel = 0, op_dback = 0;
    rad_op op_logits = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0, op_drerank = 0;
    rad_op op_dgather = 0, op_dmerge = 0;
    /* RADIANCE_MTP_COUPLED: one op a round. The ABI freezes `round` as a parameter, so a round
     * cannot be an operand and each gets its own declared op. */
    rad_op op_couple[RAD_MTP_COUPLE_MAX_ROUNDS] = {};

    /* rad_arch.h's vocab-head rule, asked through the block by whoever sizes `dlog` and the pair
     * planes. */
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
    /* The pre-fc norm over the wide stream is grouped, one rsqrt per sub-stream, like every gated
     * residual here. A flat norm fits the same [hc*n] gain and passes per-op checks; only the
     * acceptance rate tells them apart (docs/OPS.md's `mtp_enter` row). */
    ngroup = hc.hc;
    if (rows <= 0 || !w_tok || !w_lm || hc.hc <= 1) return RAD_E_INVAL;
    if (!w.src || !w.hin || !w.hw || !w.e || !w.hout) return RAD_E_INVAL;
    const int64_t HN = hc.hc * g.n_embd;

    /* ---- the entry's four weights, all replicated: two gains over the replicated stream, and two
     * fc matrices whose column shard would save 13 MB per rank at an all-reduce per round. */
    const char* nph = nm.f("mtp.pre_fc_norm_hidden.weight");
    const char* npe = nm.f("mtp.pre_fc_norm_embedding.weight");
    const char* nfh = nm.f("mtp.fc_hidden.weight");
    const char* nfe = nm.f("mtp.fc_embedding.weight");
    RAD_ARCH_TRY(map_copy(b, nph, src.pre_norm_hidden));
    RAD_ARCH_TRY(map_copy(b, npe, src.pre_norm_embedding));
    RAD_ARCH_TRY(map_copy(b, nfh, src.fc_hidden));
    RAD_ARCH_TRY(map_copy(b, nfe, src.fc_embedding));
    /* The entry matrices are bf16 or E4M3 rows with a per-group scale (OPS.md's `fc` on
     * mtp_enter), declared over all planes at the codes' dtype; both must match. */
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

    /* Separate gather_rows and embed_lookup ops over the trunk's buffers and weights, because
     * they write different buffers and liveness comes from each op's read/write sets. */
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
    /* The trunk's table is vocab-sharded, so the lookup is partial and needs an exact
     * all-reduce. */
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

    /* ---- the layer, as a trunk full-attention layer at index n_layer: gated residual, gated
     * attention with its indexer, gated residual, routed feed-forward. Declared in step()'s issue
     * order. */
    RAD_ARCH_TRY(hc_attn.declare(b, nm, g, nm.f("mtp.attn_hc"), hc, HyperConn::Wire{w.hw, w.x, w.inj},
                                 src.hc_attn));
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv));
    attn.ar_out_take = hc_attn.ar_take();
    RAD_ARCH_TRY(attn.declare(b, nm, g, l, kv, rotary_dim, w.attn, src.attn,
                              false, false, kArNone, hc_attn.op_ar_write != 0));
    if (src.qsa.qk_proj)
        RAD_ARCH_TRY(qsa.declare(b, nm, g, l, src.qsa, w.qsa, w.attn.h, qsa_bk));
    /* ---- round 1's rows of a history pass (RadBatch::draft_out_ids), moved before the
     * feed-forward, which then runs on them only: gathered into `hin` and copied back to the
     * front of `hw` (an in-place gather could overwrite unread rows). */
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

    /* The connection's read also writes the rotated expert input, as on the trunk. */
    HyperConn::Config mcfg = hc;
    HyperConn::Wire   mwire{w.hw, w.x, w.inj};
    if (src.mlp.cfg.expert_rot) { mcfg.rotate = 1; mwire.rq = w.mlp.hr.q; mwire.rs = w.mlp.hr.s; }
    RAD_ARCH_TRY(hc_mlp.declare(b, nm, g, nm.f("mtp.ffn_hc"), mcfg, mwire, src.hc_mlp));
    mlp.ar_out_take = hc_mlp.ar_take();
    mlp.hr_in = mcfg.rotate != 0;
    RAD_ARCH_TRY(mlp.declare(b, nm, g, l, w.mlp, src.mlp, false, false, kArNone,
                             hc_mlp.op_ar_write != 0));
    /* The following write takes the MoE gather where it can, as on the trunk. */
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

    /* ---- the carry (see the header: `mtp_enter` cannot run in place). */
    op_carry = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", HN),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.hw}, {w.hin});

    /* ---- the head's mixer: a connection with no write half that collapses the wide stream for
     * the lm_head, which reads bf16, so no fp8 codes. */
    {
        HyperConn::Config mc = hc;
        mc.inject = 0;
        mc.quant  = 0;
        HyperConn::Wire mixw{};
        mixw.h = w.hw; mixw.x.x = w.hout;      /* q/s deliberately absent */
        RAD_ARCH_TRY(mixer.declare(b, nm, g, nm.f("mtp.output_hc"), mc, mixw, src.hc_mix));
    }

    /* The trunk's lm_head over the head's hidden state, banded at the logits buffer's extent (one
     * row per drafting sequence). */
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                 RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_lm)),
                  {w.hout}, {w.logits});
    if (!op_carry || !op_logits) return RAD_E_INVAL;

    /* ---- The 2-bit draft head (rationale in rad_block_mtp_fp8.h): fewer bytes per draft round,
     * at the cost of acceptance only. */
    enum { MTP_W2_GROUP = 128 };
    /* R is kDraftTopR. It must also be at least 2: `ar_gather_2rank` needs rows of whole 16-byte
     * words and a pair is 8 bytes. The engine reads column 0 through `proposal_pitch`. */
    q2 = w.dq && w.ds && w.dsum && w.dlog && w.didx && w.dval;
    if (q2) {
        /* Vocab-parallel when possible (159 MB a round instead of 318). Bypassing the sampler, it
         * uses the same three ops as rad_block_dspark.h: a top-R reporting global ids, an
         * all_gather of (id, value) pairs, and a merge giving exactly the unsharded top-R. */
        const bool dh_shard = head_sharded(g) && w.dpair && w.dgath;
        const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
        const int64_t hrows = dh_shard ? g.n_vocab : g.n_vocab_all;
        /* Its own stored weight made from the trunk's lm_head by the recipe (`u2*f16+u8[1x128]`),
         * name map declared by mtp_draft_head_q2. The GEMM takes codes as one operand and the
         * scale/zero planes as the other. */
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
        /* Sharded, the top-R runs over this rank's plane, reports global ids and also writes the
         * pair plane the gather moves. Replicated, `vocab_off` is omitted: an optional parameter
         * present with value 0 is a different declaration from an absent one. */
        op_dtopk = dh_shard
            ? rw(b, RAD_OP(b, "row_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("N", hrows),
                               RAD_INT("R", kDraftTopR), RAD_INT("vocab_off", g.vocab_off),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                 {w.dlog}, {w.didx, w.dval, w.dpair})
            : rw(b, RAD_OP(b, "row_topk",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("N", hrows),
                               RAD_INT("R", kDraftTopR), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                 {w.dlog}, {w.didx, w.dval});
        /* ---- Exact rescoring of the coarse candidates against `w_lm`, the trunk's lm_head
         * (already resident; the fallback `op_logits` reads it). It reads R rows per draft row in
         * the trunk's stored layout, which libr4d's rerank shares with its fp8 logits GEMM.
         *
         * No extra collective: sharded, every candidate is one this rank owns, and the existing
         * gather and merge then rank exact values. It overwrites `didx`/`dval`/`dpair` in place.
         *
         * Skipped for a replicated draft head above one rank: the vocabulary did not shard cleanly
         * (vocab_head_sharded) while the trunk's lm_head stays row-sharded, so candidates include
         * ids this rank cannot score.
         *
         * `vocab_off` is always passed: this op has one form, and at world 1 it is 0. */
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
            /* Arrays rather than rw(): the write set is conditional (see rad_fp8.h:
             * NormQuantFP8). */
            rad_buf rd[2] = { w.hout, w.didx };
            rad_buf wr[3] = { w.didx, w.dval, w.dpair };
            rad_op_reads (b, op_drerank, rd, 2);
            rad_op_writes(b, op_drerank, wr, dh_shard ? 3 : 2);
        }
        if (dh_shard) {
            /* Row-interleaved, so one draft row's pairs from every rank are adjacent for
             * `row_topk_merge`. */
            op_dgather = rw(b, RAD_OP(b, "all_gather",
                               RAD_PARAMS(RAD_INT("world_size", g.world),
                                          RAD_INT("numel", g.max_logit_rows * kDraftTopR * 2),
                                          RAD_STR("dtype", "f32"),
                                          RAD_INT("row", kDraftTopR * 2)),
                               RAD_NOWEIGHTS),
                           {w.dpair}, {w.dgath});
            op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                         RAD_INT("R", kDraftTopR),
                                         RAD_INT("world_size", g.world),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                          {w.dgath}, {w.didx, w.dval});
            if (!op_dgather || !op_dmerge) return RAD_E_INVAL;
        }
        if (!op_dquant || !op_dlogits || !op_dtopk) return RAD_E_INVAL;
        /* THE LINEAR COUPLED PROPOSAL: one op a round (the ABI freezes `round` as a parameter, so a
         * round cannot be an operand), after the final rerank/merge, on the shortlist that would
         * otherwise give column 0.
         *
         * LINEAR CHAIN ONLY. A coupled row's Q is the head's own top-8 at temperature 1, and x is a
         * TOKEN; a tree takes column 0 as a parent index, so coupling and branching cannot share a
         * pass. The engine refuses that combination before this runs; the test here keeps a
         * declaration that disagrees with it from silently proposing tokens into a tree plane. */
        if (mtp_coupled_enabled()) {
            if (!w.couple_draws || !w.couple_state || depth < 1 ||
                depth > RAD_MTP_COUPLE_MAX_ROUNDS || !op_drerank || (g.world > 1 && !op_dmerge))
                return RAD_E_UNSUPPORTED;
            for (int round = 1; round <= depth; ++round) {
                op_couple[round - 1] = rw(b, RAD_OP(b, "mtp_couple_propose",
                    /* The WHOLE vocabulary: `row_topk` reports global ids (its `vocab_off`) and
                     * the merge keeps them, while `g.n_vocab` is this rank's shard. */
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("n_vocab", g.n_vocab_all),
                               RAD_INT("round", round), RAD_INT("max_rounds", depth)),
                    RAD_NOWEIGHTS),
                    { w.didx, w.dval, w.couple_draws }, { w.didx, w.couple_state });
                if (!op_couple[round - 1]) return RAD_E_INVAL;
            }
        }
    }
    /* Keep the carry `hin` live through the head's last op: it is written before the mixer and
     * lm_head run, and the planner could otherwise give its bytes to them (silently lowering
     * acceptance). */
    {
        rad_op last = op_logits;
        if (q2) last = op_dmerge ? op_dmerge : op_drerank ? op_drerank : op_dtopk;
        if (op_couple[0]) last = op_couple[depth - 1];
        if (rad_op_reads(b, last, &w.hin, 1) < 0) return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* One pass of the head over the scheduler's batch for it: n_tok rows, `token_ids` already shifted
 * one position on (sched/batch.cpp, token_at), `positions` the head's indices, and the KV group
 * batch carrying each row's slot, so the attention layer needs no special case. */
inline void MtpHcBlock::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int     r = batch->draft_pass;

    /* The history pass continues from the trunk's wide stream; later rounds from the head's,
     * which the carry left in the same buffer. */
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

    hc_attn.read_pieces(c, T);

    /* A history pass with no drafting rows (every prefill chunk but a prompt's last) only stores
     * the head's state: K and V, and the indexer's block keys and tail. `n_draft_out` is the same
     * on every rank, so skipped collectives match. */
    if (r < 0 && batch->n_draft_out <= 0) {
        qsa.store(c, attn.w.h, batch);
        attn.store_kv(c, batch);
        return;
    }

    /* The indexer runs first because the attention reads its selection; `w.h` is already final,
     * written by the connection above. */
    qsa.step(c, attn.w.h, batch, qsa_win_cu);
    attn.step(c, batch);

    hc_attn.write(c, T, 0, T);

    /* A history pass with drafting rows continues on those rows alone (round 1): they are moved
     * to the front of `hw` and the rest of the head runs over them. */
    int64_t L = T;                      /* the rows the stream goes on with */
    if (r < 0) {
        L = batch->n_draft_out;
        RAD_ISSUE_N(c, op_dsel, L, RAD_B(w.hw), praw(batch->draft_out_ids, RAD_I32, L),
                    brows(w.hin, L));
        RAD_ISSUE_N(c, op_dback, L, brows(w.hin, L), brows(w.hw, L));
    }

    hc_mlp.read_pieces(c, L);
    mlp.step_rows(c, L);
    hc_mlp.write(c, L, 0, L);

    /* The carry, only if a later round reads it: a history pass is round 1, followed by round 2
     * when depth > 1; the last round is followed by a trunk step. */
    if (r < 0 ? depth > 1 : r < depth)
        RAD_ISSUE_N(c, op_carry, L, brows(w.hw, L), brows(w.hin, L));

    mixer.read(c, L, 0, L);
    if (q2) {
        /* `dsum` is group-major [n_embd/group][ceil(L/16)*16], so it has no row slice; pass it
         * whole. */
        RAD_ISSUE_N(c, op_dquant, L, brows(w.hout, L), brows(w.dq, L), brows(w.ds, L),
                    RAD_B(w.dsum));
        /* The trailing `b_ref` is mxfp4's and absent; operands are positional, so pass RAD_NONE
         * (rad_fp8.h). */
        RAD_ISSUE_N(c, op_dlogits, L, brows(w.dq, L), brows(w.ds, L), RAD_W(w_dh), RAD_W(w_dhs),
                    brows(w.dlog, L), RAD_B(w.dsum), RAD_NONE);
        if (op_dmerge) {
            RAD_ISSUE_N(c, op_dtopk, L, brows(w.dlog, L), brows(w.didx, L), brows(w.dval, L),
                        brows(w.dpair, L));
            /* In place over all three, so the gather and merge carry exact values. */
            if (op_drerank)
                RAD_ISSUE_N(c, op_drerank, L, brows(w.hout, L), RAD_W(w_lm), brows(w.didx, L),
                            brows(w.didx, L), brows(w.dval, L), brows(w.dpair, L));
            RAD_ISSUE(c, op_dgather, brows(w.dpair, L), brows(w.dgath, L));
            RAD_ISSUE_N(c, op_dmerge, L, brows(w.dgath, L), brows(w.didx, L), brows(w.dval, L));
        } else {
            RAD_ISSUE_N(c, op_dtopk, L, brows(w.dlog, L), brows(w.didx, L), brows(w.dval, L),
                        RAD_NONE);   /* the `pairs` slot, absent: operands are positional */
            if (op_drerank)
                RAD_ISSUE_N(c, op_drerank, L, brows(w.hout, L), RAD_W(w_lm), brows(w.didx, L),
                            brows(w.didx, L), brows(w.dval, L), RAD_NONE);
        }
        /* THE COUPLED PROPOSAL, after the FINAL reranked shortlist: column 0 is what ADV_TAKE reads,
         * and a sampled row replaces it with its own x. Round 1 is carried by the history pass
         * (r < 0), every later round by its own pass, exactly as the carry above is. */
        if (op_couple[0]) {
            const int round = r < 0 ? 1 : r;
            if (round < 1 || round > depth) {
                rad_step_fail(c, "MTP coupled round exceeds the declared draft depth");
                return;
            }
            RAD_ISSUE_N(c, op_couple[round - 1], L, brows(w.didx, L), brows(w.dval, L),
                        praw(batch->seq_ids, RAD_I32, batch->n_seq),
                        brows(w.couple_draws, L), RAD_B(w.couple_state));
        }
    } else {
        RAD_ISSUE_N(c, op_logits, L, brows(w.hout, L), RAD_W(w_lm), brows(w.logits, L));
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MTP_HC_H */
