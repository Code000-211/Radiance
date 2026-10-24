/* rad_block_mtp_fp8.h -- the multi-token-prediction head, with block-scaled FP8 projections.
 *
 * ============================== WHAT IT IS ==============================
 *
 * Qwen3.5 / Qwen3-Next ship ONE extra transformer layer beside the trunk, under an `mtp.` prefix
 * of its own, whose job is to predict token i+2 from the trunk's hidden state at i and the
 * embedding of the token at i+1. It is the drafter for speculative decoding, and it is a drafter
 * with no second checkpoint: the embedding table and the lm_head are the trunk's
 * (`mtp_use_dedicated_embeddings: false`, and no `mtp.lm_head` in the file).
 *
 * THE INDEXING IS THE WHOLE THING, so it is stated once here and everything else follows:
 *
 *     head index i    reads  hidden(i) and embed(token at i + 1)
 *                     writes its K and V at KV position i, with rope position i
 *                     predicts the token at i + 2
 *
 * That is DeepSeek-V3's MTP layout, which this family follows: the head's transformer block runs
 * over the SAME position indices as the trunk, and the shift lives entirely in which token gets
 * embedded. Getting it off by one costs nothing visible -- the drafts are simply wrong and the
 * acceptance rate collapses, while the model's own output stays exactly correct.
 *
 * The forward is short and every piece of it already exists in this library:
 *
 *     h = gather(the trunk's final hidden states, at the rows this pass continues from)
 *     e = embed(the token one position on)
 *     x = fc( concat( pre_fc_norm_embedding(e), pre_fc_norm_hidden(h) ) )
 *     x = one gated-GQA layer + one SwiGLU feed-forward, over x            <- AttnGatedFP8, MlpFP8
 *     h'= mtp.norm(x)
 *     logits = the TRUNK's lm_head over h'
 *
 * and h' is BOTH the logits input and the `h` of the next draft round. That is the reference's
 * shape too: a Qwen3-Next MTP forward returns one tensor and the proposer feeds the same tensor
 * back, unlike an EAGLE head which returns a (normed, unnormed) pair.
 *
 * ============================== THE TWO KINDS OF PASS ==============================
 *
 * RadBatch::draft_pass says which, and the sign is the whole distinction:
 *
 *   < 0   A HISTORY PASS, one row a token, gathering the trunk's hidden states. It exists
 *         because the head's attention reads its OWN K and V for every earlier position, and
 *         nothing else in the engine writes those: the trunk's prefill fills layers 0..n-1 and
 *         leaves the head's layer untouched. Without it the head attends to whatever the pool
 *         last held, which is not wrong output -- a bad draft is only a rejected draft -- but it
 *         is a drafter that proposes noise. The scheduler runs one over the positions the last
 *         step added, the verified ones included.
 *
 *         It is also DRAFT ROUND 1 for every sequence that drafts: that sequence's last history
 *         row is the committed position with the token the trunk chose after it -- round 1's
 *         whole input -- so the pass ends in the lm_head over those rows (RadBatch::draft_out_ids)
 *         and leaves the head's output at them where round 2 reads it.
 *
 *   > 1   A DRAFT ROUND, one row a sequence, reading the head's own output from the round before,
 *         which is already in the same buffer. Every round ends in the lm_head, because a draft is
 *         a token. There is no round-1 pass; the history pass is it.
 *
 * ============================== WHAT IS AND IS NOT FP8 ==============================
 *
 * Read off mtp.safetensors rather than assumed, exactly as qwen35_fp8.cpp reads the trunk:
 *
 *   F8_E4M3 + scale   layers.0.self_attn.{q,k,v,o}_proj, layers.0.mlp.{gate,up,down}_proj
 *   BF16, no scale    fc, norm, pre_fc_norm_embedding, pre_fc_norm_hidden, and every norm inside
 *                     the layer
 *
 * `fc` is [5120, 10240] and bf16, so it is a plain `gemm_nt`, issued ONCE over the whole chunk.
 * libr4d's bf16 GEMM row for this shape is C_LE("M", 65536) and K here is 10240, a multiple of 16,
 * so the wide row serves the whole chunk in one launch. That matters because the history pass runs
 * one row a TOKEN over a whole prefill chunk: banding this at 64 and slicing it would be dozens of
 * launches a chunk for a GEMM whose traffic is microseconds.
 *
 * ============================== THE CONCAT ORDER ==============================
 *
 * `fc` reads 10240 columns and the two halves are the normed embedding and the normed hidden
 * state. WHICH HALF IS FIRST IS NOT IN THE CHECKPOINT: both orders are one contiguous [5120,10240]
 * matrix and no shape, name or dtype distinguishes them. DeepSeek-V3's MTP -- the design this one
 * follows -- concatenates `enorm(embedding)` then `hnorm(hidden)`, and that is the order here.
 *
 * Both readings are self-consistent, libref shares whichever one the graph picks, and no per-op
 * oracle can reach it -- but it has a CHEAP EMPIRICAL ANSWER, the draft acceptance rate. A head
 * wired the right way round accepts most of what it proposes; wired the wrong way round it
 * proposes noise and accepts nothing, while the model's own output stays perfectly correct
 * because a rejected draft costs only time.
 *
 * ============================== THE NORM GAINS ==============================
 *
 * GemmaRMSNorm's `1 + w` (Geom::wadd) applies here and the checkpoint proves it rather than
 * suggesting it: pre_fc_norm_hidden sits at -0.06 .. -0.24 and pre_fc_norm_embedding at
 * -0.27 .. -0.61 -- NEGATIVE gains, which only `1 + w` explains. `mtp.norm` sits at 0.59 .. 1.31,
 * which reads as either, so it takes the same `final_wadd` the trunk's output norm takes -- one
 * question, asked once, answered in one place.
 *
 * ============================== THE KV CACHE ==============================
 *
 * The head's attention layer is bound to the TRUNK'S full-attention group as one more layer, at
 * absolute index n_main. Two reasons, and neither is thrift:
 *
 *   - the slot mapping and the block table are the trunk's. The head writes its K and V at the
 *     slot the trunk's token for that position occupies, so a rejected draft's entry is
 *     overwritten by the next step's correct token AT THE SAME ADDRESS. Draft rounds run in
 *     position order and each writes its own position before attending to it, so there is no
 *     rollback to get wrong -- which is why vLLM's EAGLE/MTP proposer needs no second block
 *     manager either.
 *   - a second group would need its own page geometry, its own free list and its own eviction,
 *     for a cache one seventeenth the size of the one already there.
 *
 * The cost is that the group's page grows from 16 layers to 17 -- 6% more attention KV -- and it
 * is paid only when the deployment declares speculation, because that is when the head is
 * declared at all.
 */
#ifndef RAD_BLOCK_MTP_FP8_H
#define RAD_BLOCK_MTP_FP8_H

#include "rad_block_attn_gated_fp8.h"
#include "rad_block_mlp_fp8.h"
#include "rad_block_vit.h"

#include <cstdlib>

namespace rad {
namespace arch {

/* THE 2-BIT DRAFT HEAD IS A WEIGHT THE MODEL HOLDS OR DOES NOT. `head` is the lm_head under a name
 * of its own, which a recipe quantises to the 2-bit grid -- u2 codes, an f16 scale and a u8 zero a
 * group of 128 -- so one checkpoint tensor makes two logical weights. This declares its name map
 * (to `lm`, the checkpoint's head) and answers whether the model holds it at two bits, which is
 * when the draft rounds read it. A checkpoint served as it stands, or a container converted
 * without the rule, holds the bf16 head alone, and the draft rounds read that.
 *
 * Called ONCE, by the plugin, before it declares the head's buffers -- a name map is declared once
 * -- and the block reads the answer back from whether those buffers exist. */
inline bool mtp_draft_head_q2(RadBuilder* b, const char* head, const char* lm) {
    if (!head || !lm) return false;
    if (map_copy(b, head, lm) < 0) return false;
    RadEncoding e{};
    return weight_enc(b, head, &e) && e.n_planes > 0 && e.plane[0].dtype == RAD_U2;
}

inline bool mtp_draft_head_reduced(RadBuilder* b, const char* head, const char* lm_reduced) {
    if (!head || !lm_reduced) return false;
    if (map_copy(b, head, lm_reduced) < 0) return false;
    RadEncoding e{};
    return weight_enc(b, head, &e) && e.n_planes > 0 && e.plane[0].dtype == RAD_U4;
}

/* TEMPLATED ON THE FEED-FORWARD, because the head is a TRUNK LAYER and a trunk layer's
 * feed-forward is dense in one member of this family and routed in the next. Qwen3.6-35B-A3B's
 * `mtp.layers.0` carries `mlp.experts.*`, `mlp.gate` and a shared expert, exactly as its trunk
 * layers do; a head hard-wired to MlpFP8 could not declare it, and the model would have a
 * container with a drafter in it and no way to run one.
 *
 * `Ffn` is MlpFP8 or MoeFP8. Both take the same declare() arity over their own Wire and Src -- the
 * MoE block's Config travels inside its Src for exactly this reason -- so nothing below has to
 * know which one it has. */
template <class Ffn>
struct MtpBlockFP8 {
    struct Wire {
        /* [max_tok, n_embd] the TRUNK's final hidden states for the step just run -- the plugin's
         * `h`, which gather_rows already reads for the trunk's own logits. A pass names the rows
         * it wants through out_ids and never has to know how they were laid out. */
        rad_buf src    = 0;
        /* [max_tok, n_embd] IN: the hidden state this pass predicts from. OUT: the head's own,
         * which is the next draft round's input. Sized for a whole prefill chunk because a
         * history pass is one row a token. */
        rad_buf hout   = 0;
        rad_buf e      = 0;   /* [max_tok, n_embd]      the input token's embedding */
        rad_buf c      = 0;   /* [max_tok, 2 * n_embd]  the two normed halves, fc's input */
        rad_buf x      = 0;   /* [max_tok, n_embd]      the head's own residual stream */
        rad_buf logits = 0;   /* [max_logit_rows, n_vocab]  the trunk's, shared */
        /* THE 2-BIT DRAFT HEAD's working set -- see the block comment at op_dlogits. Every one of
         * these is a draft round's, so they are sized at max_logit_rows and not at max_tok: a
         * history pass never reaches any of them. */
        rad_buf dq   = 0;   /* [max_logit_rows, n_embd]   int8 codes */
        rad_buf ds   = 0;   /* [max_logit_rows, 1]        f32, one scale a row */
        rad_buf dsum = 0;   /* [n_embd/group, mpad]       f32, GROUP-major (quant_act_i8's asum) */
        rad_buf dlog = 0;   /* [max_logit_rows, n_vocab]  bf16, the coarse logits */
        rad_buf didx = 0;   /* [max_logit_rows, 1]        i32, the proposed token */
        rad_buf dval = 0;   /* [max_logit_rows, 1]        f32, its logit */
        /* The layer's working set is the TRUNK's -- those buffers are dead by the time any of
         * this runs, and a second set would be 150 MiB of activation planes for one layer. */
        AttnGatedFP8::Wire attn{};
        typename Ffn::Wire mlp{};
    };

    struct Src {
        const char* fc                 = nullptr;
        const char* norm               = nullptr;
        const char* pre_norm_hidden    = nullptr;
        const char* pre_norm_embedding = nullptr;
        /* The checkpoint's lm_head, again. The trunk maps it as `output.weight`; the draft head is
         * a second logical weight over the SAME tensor, which the recipe quantises to 2-bit codes
         * (mtp_draft_head_q2). One source, two weights, no second checkpoint tensor. */
        const char* lm                 = nullptr;
        AttnGatedFP8::Src attn{};
        typename Ffn::Src mlp{};
    };

    Geom       g{};
    Wire       w{};
    int        layer   = 0;    /* ABSOLUTE, so the KV group's layer_slot lookup finds it */
    int64_t    rows    = 0;    /* the widest pass: one row a token of a prefill chunk */
    int64_t    off_e = 0, off_h = 0;   /* the two halves' column offsets inside `c` */
    rad_weight w_tok = 0, w_lm = 0, w_fc = 0, w_norm = 0, w_pre_h = 0, w_pre_e = 0;
    rad_weight w_dh = 0, w_dhs = 0;    /* the 2-bit lm_head and its (scale, zero) plane */
    bool       q2 = false;             /* is the draft round reading it? */
    AttnGatedFP8 attn{};
    Ffn          mlp{};
    rad_op     op_gather = 0, op_embed = 0, op_embed_ar = 0, op_pre_h = 0, op_pre_e = 0, op_fc = 0;
    /* THE PROMPT'S MEDIA ROWS over the head's own embedding: its history pass embeds the prompt
     * too, one position on, and an image token's embedding is the encoder's there as well. Set
     * `media` before declare when the program has an encoder. */
    bool       media = false;
    MediaRows  mrows{};
    rad_op     op_norm = 0, op_dsel = 0, op_dback = 0, op_logits = 0;
    rad_op     op_dquant = 0, op_dlogits = 0, op_dtopk = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup kv,
                 int64_t rotary_dim, rad_weight tok_embd, rad_weight lm_head, float out_wadd,
                 const Wire& wire, const Src& src);
    void step(RadCtx* c, const RadBatch* batch) const;
};

template <class Ffn>
inline int MtpBlockFP8<Ffn>::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                     rad_kvgroup kv, int64_t rotary_dim, rad_weight tok_embd,
                                     rad_weight lm_head, float out_wadd, const Wire& wire,
                                     const Src& src) {
    g = geom; w = wire; layer = l; w_tok = tok_embd; w_lm = lm_head;
    rows = g.max_tok;
    if (rows <= 0 || !w_tok || !w_lm) return RAD_E_INVAL;

    /* The normed embedding first, then the normed hidden state; see THE CONCAT ORDER above. */
    off_e = 0;
    off_h = g.n_embd;

    /* The head's own weights. SHARD_NONE on all four: three are per-channel norms over the
     * residual stream, which is replicated, and `fc` reads and writes that same stream -- a
     * column-sharded fc would need an all-reduce to buy 52 MiB a rank back, which is the wrong
     * side of the trade at one GEMM per draft round. */
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.pre_fc_norm_hidden.weight"),    src.pre_norm_hidden));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.pre_fc_norm_embedding.weight"), src.pre_norm_embedding));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.norm.weight"),                  src.norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("mtp.fc.weight"),                    src.fc));

    w_pre_h = decl_w(b, nm.f("mtp.pre_fc_norm_hidden.weight"),    RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_pre_e = decl_w(b, nm.f("mtp.pre_fc_norm_embedding.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_norm  = decl_w(b, nm.f("mtp.norm.weight"),                  RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_fc    = decl_w(b, nm.f("mtp.fc.weight"), RAD_BF16, {g.n_embd, 2 * g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* A SECOND gather_rows and a SECOND embed_lookup, over the trunk's own buffers and weights.
     * They are separate ops rather than re-issues of the trunk's because an op's read and write
     * sets are what the buffer planner reads for liveness, and these write somewhere else. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                       RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                  RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                   {w.src}, {w.hout});
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.e});
    if (media) RAD_ARCH_TRY(mrows.declare(b, rows, g.n_embd, w.e));
    /* The trunk's table is vocab-sharded, so this gather is a partial one and needs the same
     * exact reduction the trunk does. Its own op rather than the trunk's because it writes a
     * different buffer, which is the rule the gather above states. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, rows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.e}, {w.e});

    op_pre_h = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_pre_h)),
                  {w.hout}, {w.c});
    op_pre_e = rw(b, RAD_OP(b, "rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                 RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_pre_e)),
                  {w.e}, {w.c});

    op_fc = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", g.n_embd),
                              RAD_INT("K", 2 * g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_fc)),
               {w.c}, {w.x});

    RAD_ARCH_TRY(attn.declare(b, nm, g, l, kv, rotary_dim, w.attn, src.attn));
    RAD_ARCH_TRY(mlp.declare(b, nm, g, l, w.mlp, src.mlp));

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", out_wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.hout});

    /* DRAFT ROUND 1'S ROWS OUT OF A HISTORY PASS (RadBatch::draft_out_ids), one a drafting
     * sequence in round order, to the front of `hout` -- where the lm_head reads a round's rows
     * and round 2 reads its input. Through `x`, which is dead once the norm has read it, because a
     * gather in place could overwrite a row another sequence has yet to read. */
    op_dsel = rw(b, RAD_OP(b, "gather_rows",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.hout}, {w.x});
    op_dback = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", g.n_embd),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.x}, {w.hout});
    if (!op_dsel || !op_dback) return RAD_E_INVAL;

    /* The trunk's lm_head, over the head's hidden state. Banded at the logits buffer's own extent
     * rather than at `rows`: it runs over one row a drafting sequence. */
    op_logits = rw(b, RAD_OP(b, "logits_gemm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                 RAD_INT("n_vocab", g.n_vocab),
                                 RAD_INT("n_embd", g.n_embd), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_lm)),
                  {w.hout}, {w.logits});

    /* ---- THE 2-BIT DRAFT HEAD -------------------------------------------------------------
     *
     * `logits_gemm` IS ALREADY AT THE MEMORY ROOFLINE for the bf16 lm_head's 2.54 GB, so the only
     * lever it has is FEWER BYTES. That matters here and not at the trunk, because a speculative
     * step runs the projection THREE times: once to verify, and once per draft round.
     *
     * The two draft ones do not need the exact distribution. A draft is a PROPOSAL; the target
     * verifies every one of them with its own untouched bf16 head, and speculative decoding is
     * distribution-preserving -- so a worse draft costs ACCEPTANCE and cannot change a token the
     * model emits. That is the one place in the step where precision buys throughput outright.
     *
     * So the drafter reads a 2-bit copy: 318 MB of codes and 40 MB of (scale, zero) dwords against
     * 2.54 GB, an eighth of the traffic. It is a logical weight of its own over the checkpoint's
     * head (mtp_draft_head_q2), quantised by the recipe; libr4d's `gemm_w2a8_nt` takes its codes
     * and its scale and zero planes as two operands and lays them out as `r4d.w2a8.frag.g128` at
     * load (libr4d/r4d_layout.cpp). The grid is asymmetric -- four levels are not enough for a
     * symmetric one -- which is why the activation side is `quant_act_i8`'s three-output form: the
     * per-group row sums are what subtract the stored zero.
     *
     * AND IT DOES NOT GO THROUGH THE SAMPLER. The 2-bit GEMM writes bf16 and the sampler's chain
     * reads the f32 logits plane, so a draft round would need a cast of a 248K-wide row to reach
     * it -- for an argmax. `row_topk` at R=1 is that argmax, on the bf16 the GEMM already wrote,
     * and the engine reads the proposed id straight out of `didx`. The trunk's logits plane is
     * left alone, which is also what lets a draft round sample nothing at all.
     */
    enum { MTP_W2_GROUP = 128 };      /* libr4d's R4D_GEMM_W2A8_GROUP, and the layout's g128 tag */
    /* The plugin declared the working set only where mtp_draft_head_q2 found the head at two
     * bits, which declared its name map too. */
    q2 = src.lm && w.dq && w.ds && w.dsum && w.dlog && w.didx && w.dval;
    if (q2) {
        /* REPLICATED, AT THE WHOLE VOCABULARY, and that is a decision rather than an omission.
         *
         * The trunk's lm_head is row-sharded and its logits are merged across ranks by the
         * sampler. This head does NOT go through the sampler -- the comment above op_dlogits says
         * why -- so its argmax is a `row_topk` over one rank's plane, and a sharded plane would
         * make every rank propose only tokens from its own half. Replicating it costs 341 MiB a
         * card at 2 bits (against the 1.27 GiB a rank saves on the trunk head) and keeps the
         * draft path a single-rank computation, which is what it is.
         *
         * The stored bytes are also tile-major (16 output rows a packed block), so the relayout
         * would have to be handed a row share that is a whole number of tiles. */
        const char* dhn = nm.f("mtp.draft_head.weight");
        w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {g.n_vocab_all, g.n_embd},
                          RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        w_dhs = decl_view(b, nm.f("mtp.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                          {g.n_vocab_all, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_layer(l));
        if (!w_dh || !w_dhs) return RAD_E_INVAL;

        op_dquant = rw(b, RAD_OP(b, "quant_act_i8",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                     RAD_INT("n", g.n_embd), RAD_INT("group", MTP_W2_GROUP),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.hout}, {w.dq, w.ds, w.dsum});

        op_dlogits = rw(b, RAD_OP(b, "gemm_nt_q",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                      RAD_INT("N", g.n_vocab_all), RAD_INT("K", g.n_embd),
                                      RAD_INT("group", MTP_W2_GROUP), RAD_STR("dtype", "w2a8")),
                           RAD_WEIGHTS(w_dh, w_dhs)),
                       {w.dq, w.ds, w.dsum}, {w.dlog});

        op_dtopk = rw(b, RAD_OP(b, "row_topk",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows),
                                    RAD_INT("N", g.n_vocab_all), RAD_INT("R", 1),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {w.dlog}, {w.didx, w.dval});
    }

    /* THE HEAD'S OUTPUT STAYS LIVE THROUGH THE WHOLE HEAD. It is the next round's input, written
     * before this pass's lm_head runs; declared as read by the lm_head's first op alone, the
     * buffer planner would let the ops after it take its bytes, and the next round would continue
     * from what they wrote. Nothing fails: acceptance falls. Read by the head's last op as well,
     * it survives every pass of the step. */
    {
        const rad_op last = q2 ? op_dtopk : op_logits;
        if (rad_op_reads(b, last, &w.hout, 1) < 0) return RAD_E_INVAL;
    }
    return RAD_OK;
}

/* One pass of the head. The batch is the one the scheduler built for it: n_tok rows, `token_ids`
 * already shifted one position on (batch.cpp, token_at), `positions` the head's own indices, and
 * the KV group batch carrying the slot each row writes at -- so every op below runs at n_tok rows
 * and the attention layer needs no special case at all. */
template <class Ffn>
inline void MtpBlockFP8<Ffn>::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int r = batch->draft_pass;

    /* The history pass gathers; the rounds after it read the head's own output, which the pass
     * before left in the same buffer. */
    if (r == 1) {
        rad_step_fail(c, "the MTP head was handed a draft-round-1 pass: that round is carried by "
                         "the history pass (RadBatch::draft_out_ids) and has no pass of its own");
        return;
    }
    if (r < 0 && batch->n_out > 0)
        RAD_ISSUE_N(c, op_gather, batch->n_out,
                    RAD_B(w.src), praw(batch->out_ids, RAD_I32, batch->n_out),
                    brows(w.hout, batch->n_out));

    RAD_ISSUE(c, op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.e, T));
    mrows.step(c, batch, w.e, T);
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.e, T), RAD_NONE);
    RAD_ISSUE(c, op_pre_h, brows(w.hout, T), RAD_W(w_pre_h), bcol(w.c, off_h, g.n_embd, T));
    RAD_ISSUE(c, op_pre_e, brows(w.e, T),    RAD_W(w_pre_e), bcol(w.c, off_e, g.n_embd, T));

    RAD_ISSUE(c, op_fc, brows(w.c, T), RAD_W(w_fc), brows(w.x, T));

    attn.step(c, batch);

    /* A HISTORY PASS THAT DRAFTS NOTHING ENDS HERE. It exists to leave K and V behind, and the
     * attention above has stored them. The feed-forward, the norm and the lm_head compute the
     * head's output, which only a drafting row reads -- and every prefill chunk but a prompt's
     * last drafts nothing. `n_draft_out` is the batch's and the same on every rank, so a
     * collective the feed-forward would run is skipped on all of them. */
    if (r < 0 && batch->n_draft_out <= 0) return;

    mlp.step(c, batch);

    RAD_ISSUE(c, op_norm, brows(w.x, T), RAD_W(w_norm), brows(w.hout, T));

    /* A history pass that drafts brings round 1's rows to the front first. */
    int64_t L = T;                      /* the rows the lm_head runs over */
    if (r < 0) {
        L = batch->n_draft_out;
        RAD_ISSUE_N(c, op_dsel, L, RAD_B(w.hout), praw(batch->draft_out_ids, RAD_I32, L),
                    brows(w.x, L));
        RAD_ISSUE_N(c, op_dback, L, brows(w.x, L), brows(w.hout, L));
    }
    if (q2) {
        /* `dsum` is GROUP-major [n_embd/group][ceil(L/16)*16] and not row-major, so it is passed
         * whole: there is no row slice of it that means anything. */
        RAD_ISSUE_N(c, op_dquant, L, brows(w.hout, L), brows(w.dq, L), brows(w.ds, L),
                    RAD_B(w.dsum));
        /* The trailing `b_ref` is mxfp4's and absent here; an absent optional operand is a null
         * tensor in a full-length list, not a shorter list (rad_fp8.h says why). */
        RAD_ISSUE_N(c, op_dlogits, L, brows(w.dq, L), brows(w.ds, L), RAD_W(w_dh), RAD_W(w_dhs),
                    brows(w.dlog, L), RAD_B(w.dsum), RAD_NONE);
        RAD_ISSUE_N(c, op_dtopk, L, brows(w.dlog, L), brows(w.didx, L), brows(w.dval, L),
                    RAD_NONE);   /* the `pairs` slot, absent: operands are positional */
    } else {
        RAD_ISSUE_N(c, op_logits, L, brows(w.hout, L), RAD_W(w_lm), brows(w.logits, L));
    }
}

/* The dense head, which is the 27B's and every other member of this family whose feed-forward is
 * one SwiGLU. The routed one is MtpBlockFP8<MoeFP8>, named by the plugin that wants it -- this
 * header does not include rad_block_moe_fp8.h, because a dense plugin should not compile a routed
 * block it will never declare. */
using MtpFP8 = MtpBlockFP8<MlpFP8>;

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_MTP_FP8_H */
