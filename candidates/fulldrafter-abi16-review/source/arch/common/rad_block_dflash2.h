/* rad_block_dflash2.h -- the DFlash2 block-diffusion drafter, bf16.
 *
 * ============================== WHAT IT IS ==============================
 *
 * A SECOND, SMALLER TRANSFORMER that proposes a whole block of tokens in ONE forward pass. It is
 * the other drafter this engine serves, and it is a different shape of thing from the MTP head
 * beside it (rad_block_mtp_fp8.h):
 *
 *     MTP        depth d costs d CHAINED forwards. Round i+1 cannot start until round i has
 *                produced a token, because the token is what it embeds.
 *     DFlash2    depth d costs ONE forward over d+1 positions. The block is denoised together --
 *                the anchor token plus d mask tokens go in, d+1 hidden states come out.
 *
 * That is why depth is nearly free here and expensive there, and it is why this file has no
 * `draft_pass`: there are no rounds. Published acceptance for this drafter is 4.10-5.46 against
 * MTP's 3.74-5.02 at 7 drafts.
 *
 * The price of proposing the positions together is that position l+1 was computed WITHOUT KNOWING
 * what position l would choose. The candidate selector is what puts that dependence back: each
 * position keeps the draft head's top K, a low-rank bilinear form scores the edge between
 * consecutive positions, and a greedy walk picks one path. `dflash_select` is that kernel.
 *
 * ============================== THE TWO PASSES ==============================
 *
 * THE CONTEXT PASS, one row a committed token. The drafter never runs its layers over the context
 * -- it reads the TARGET's hidden states instead, at five intermediate layers, and turns them
 * straight into its own K and V:
 *
 *     c  = fc(concat of the trunk's residual stream at layers 5, 19, 33, 47, 61)   [n_taps*n_embd]
 *     c  = hidden_norm(c)
 *     for each draft layer:  k|v = kv_proj(c);  k = k_norm(k) then rope;  kv_store(k, v)
 *
 * The trunk's side of that -- the taps, and the `dflash_src` buffer they land in -- is in
 * qwen35_fp8.cpp. This file reads the buffer and never asks where the rows came from.
 *
 * THE QUERY PASS, one row a draft position, `block` = 1 + n_spec of them a sequence:
 *
 *     x = embed(anchor, mask, mask, ...)
 *     for each layer:
 *         h  = attn_norm(x);   coef = attn_conv_proj(h)
 *         h  = dflash_conv(h, coef[0:taps*NG], attn_conv_base[0])          <- prepare
 *         h  = o_proj(attention(q_proj(h), kv_proj(h)))                    <- non-causal, windowed
 *         h  = dflash_conv(h, coef[taps*NG:], attn_conv_base[1]);  x += h  <- finish
 *         h  = ffn_norm(x);    coef = ffn_conv_proj(h)
 *         h  = dflash_conv(h, coef[0:taps*NG], ffn_conv_base[0])
 *         h  = down(silu_mul(gate_up(h)))
 *         h  = dflash_conv(h, coef[taps*NG:], ffn_conv_base[1]);   x += h
 *     x = norm(x)
 *
 * IN Config::bf16 MODE, AND ONLY WHEN THE CALLER SAYS ITS KERNEL HOME HAS THE PLUGIN, the two
 * lines that end `x += h` are ONE op -- `dflash_conv_add`, the same convolution with the residual
 * add folded into its epilogue and the same bf16 rounding between them. The pair stays declared and
 * is what runs wherever that op does not resolve, and `Config::mine_fusions` (default false) is
 * what a caller that does not have the plugin leaves alone; see the declaration for both. Nothing
 * else in the layer changes, in either mode.
 *
 * and then, over the MASK rows of each sequence's block -- offsets 1..block-1, NOT 0..steps-1.
 * The anchor at offset 0 is a token the target has already chosen, so its row predicts nothing and
 * is there to condition the rest; each mask row predicts the token AT ITS OWN POSITION:
 *
 *     cand, unary = top_k(draft_head(x))          the candidates and their logits
 *     hp          = hidden_projection(x)          the selector's query, rank 256
 *     tokens      = dflash_select(cand, unary, hp, anchor, predecessor, successor)
 *
 * ============================== WHAT IS SHARED WITH THE TARGET ==============================
 *
 * The embedding table and the lm_head, both. The checkpoint ships neither -- 81 tensors, no
 * `embed_tokens` and no `lm_head` -- so a drafter without a target is not a model. `mask_token_id`
 * is an ordinary id in the target's vocabulary (248070 here), which is why there is no separate
 * mask embedding to load.
 *
 * IN Config::bf16 MODE THE HEAD IS THE TARGET'S lm_head AT THE PRECISION IT SHIPS AT -- one plain
 * bf16 `gemm_nt` over the same mapped tensor, the same vocabulary shard and the same candidate
 * merge. Nothing else in the block quantises: the linears are `gemm_nt` over their bf16 planes and
 * no `quant_act_*` is declared at all. What follows describes the fp8 mode, whose graph is what a
 * legacy deployment runs.
 *
 * The head reads the 2-BIT copy of the lm_head, exactly as the MTP head does and for exactly the
 * reason rad_block_mtp_fp8.h gives at op_dlogits: a draft is a proposal, the target verifies every
 * one of them with its own untouched head, and speculative decoding is distribution-preserving --
 * so a coarser draft costs acceptance and cannot change a token the model emits. Here it also buys
 * something MTP does not need: the head stays a SINGLE-RANK computation, so the candidate ids are
 * global without an all-gather and the selector's codebooks can be indexed directly.
 *
 * ============================== THE NORMS ARE PLAIN ==============================
 *
 * `w`, not `1 + w`, unlike every norm in the trunk this drafter is bolted to. vLLM says so by
 * using the ordinary RMSNorm rather than the family's GemmaRMSNorm, and the checkpoint agrees:
 * hidden_norm sits at 0.11..1.24 (mean 0.82), input_layernorm at 0.05..1.70, k_norm at 0.32..2.53
 * and the final norm at 2.45..5.25. All positive, all centred near one -- which is a plain gain
 * and is not what a `1 + w` tensor looks like. So `Geom::wadd` and `wadd_qk` are ZERO here while
 * the trunk's are one, and the two geometries are separate structs for this reason among others.
 *
 * ============================== SHARDING ==============================
 *
 * The layers shard the way any transformer's do: q/kv and gate_up by row, o and down by column
 * with an all-reduce. Ten collectives a draft pass, each 5120 * block * 2 bytes -- 80 KiB at the
 * shipped block of 8 -- which is below the lossy wire's floor and is served exactly.
 *
 * THREE THINGS ARE REPLICATED AND EACH IS A DECISION:
 *
 *   fc              [n_embd, n_taps*n_embd], 262 MiB. It could be COLUMN-sharded, since its input
 *                   is the replicated residual stream and its output would then be a partial sum:
 *                   131 MiB a rank and half the read bytes on the context pass's widest GEMM. Not
 *                   done, because it puts an all-reduce on the PREFILL path to save a small
 *                   fraction of it, and the real fix for the context pass's cost is not to run it
 *                   over the whole prompt at all (see below).
 *   conv weights    the projection and the base kernel, 26 MiB a layer. The convolution is over
 *                   the whole replicated hidden row and every rank needs every coefficient; a row
 *                   shard would give each rank a fraction of the groups.
 *   the codebooks   [n_vocab, 256] each, 127 MiB each. They are gathered by GLOBAL token id and
 *                   the ids come out of a whole-vocabulary draft head, so a vocabulary shard would
 *                   let a rank score only the candidates that happened to land in its own slice.
 *                   The gather is 128 rows a pass, so the bytes READ are nothing; it is the
 *                   residency that costs.
 *
 * ============================== THE CONTEXT PASS IS THE EXPENSIVE HALF ==============================
 *
 * In bytes: `fc` is 262 MiB and the five KV projections read 3.4 GB over a 2048-token prefill
 * chunk, which is a real fraction of what the chunk costs.
 *
 * THE FIX IS NOT A FASTER GEMM. The drafter's attention has a 2048-token window, so a context
 * position more than 2048 before the query is never read: for a prompt of any real length, all but
 * the last 2048 rows of the context pass are computed and thrown away. The scheduler can simply
 * not run it over them. That is scheduler work and it is noted here rather than done here.
 */
#ifndef RAD_BLOCK_DFLASH2_H
#define RAD_BLOCK_DFLASH2_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

/* A row slice of a WEIGHT, the same thing brow_slice() is for a buffer. The conv's base kernel is
 * [2, taps, n_embd] and the two sides are the prepare tap set and the finish tap set; there is one
 * checkpoint tensor and the op takes one side at a time. */
inline RadOperand wrow_slice(rad_weight h, int64_t first, int64_t n, int64_t pitch) {
    RadOperand o = RAD_W(h);
    o.offset = first * pitch;
    o.rows   = n;
    return o;
}

/* ============================== ONE LINEAR, IN EITHER MODE ==============================
 *
 * `Config::bf16` picks a linear's weight planes and its GEMM row and nothing else:
 *
 *     false   block fp8: E4M3 codes with a bf16 scale a 128x128 block, read by `gemm_nt_q` at
 *             `fp8a8` against the activation's E4M3 pair. The recipe makes the checkpoint so.
 *     true    the checkpoint's own bf16 tensor, read by `gemm_nt` against the bf16 activation.
 *             No quantiser is declared in front of it and none is issued.
 *
 * THE PAIR'S STRUCT IS THE SLOT IN BOTH MODES -- `s` stays 0 when there is no scale plane --
 * because the two are never both declared, and a second parallel set of members would be a second
 * graph nobody runs. It is called DenseFP8 because that is what it declares in the mode this
 * block was written for; the `rad_weight` field is the weight and that is what both modes need.
 *
 * `map_linear` owns the name map in both, so a bf16 gate_up is the same convert-time fusion (and
 * the same row parts) as an fp8 one: the fusion is a property of the checkpoint's spelling, not
 * of its encoding.
 */
inline int dflash2_dense(RadBuilder* b, Names& nm, DenseFP8& d, bool bf16, const char* base,
                         int64_t N, int64_t K, int shard, RadWeightGroup grp,
                         std::initializer_list<const char*> src,
                         std::initializer_list<int64_t> row_parts = {}) {
    if (!bf16)
        return d.declare(b, nm, base, N, K, RAD_ACCESS_PER_TOKEN, shard, grp, src, row_parts);
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    /* A QUANTISED SOURCE IS NAMED HERE, NOT SERVED AT bf16. `decl_w_n` would refuse it anyway with
     * the generic exact-widening message, which reads as a plugin bug rather than as "this
     * container was made by a recipe". Unknown is not refused: a builder with no model behind it
     * (a test, a graph dump) says nothing about the source, and the same silence LinearFP8 keeps.
     * A bf16 view of a quantised plane would be a dequantisation, and that lives in rad-convert. */
    RadEncoding enc{};
    if (weight_enc(b, wn, &enc) &&
        !(rad_enc_is(&enc, "plain") && enc.n_planes == 1 && enc.plane[0].dtype == RAD_BF16)) {
        char e[256];
        rad_enc_format(&enc, e, sizeof e);
        fprintf(stderr, "radiance: dflash2 is serving bf16 (Config::bf16) and '%s' holds '%s' as "
                        "%s. A bf16 drafter reads the checkpoint's own bf16 tensors; a quantised "
                        "container is served by the fp8 drafter, or converted without a recipe.\n",
                wn, src.size() == 1 ? *src.begin() : base, e);
        return RAD_E_DTYPE;
    }
    int64_t p[RAD_MAX_ROW_PARTS] = {0};
    int     np = 0;
    for (int64_t v : row_parts) if (np < RAD_MAX_ROW_PARTS) p[np++] = v;
    /* An uneven slice is not a thing this block asks for -- every one of its linears splits on a
     * dimension `divide_or_fail` has already divided -- so no span is set here in either mode. */
    d.w = decl_w_n(b, wn, RAD_BF16, {N, K}, RAD_ACCESS_PER_TOKEN, shard, grp, 0, p, np);
    d.s = 0;
    return d.w ? RAD_OK : RAD_E_INVAL;
}

/* The GEMM the linear is computed by, DECLARED WITH ITS READS. `gemm_nt` over the bf16 plane, or
 * `gemm_nt_q` at `fp8a8` over the E4M3 pair with the scale block the checkpoint is stored in. Both
 * are declared over `max_rows`, which is what selects between libr4d's bf16 rows -- narrow at
 * M <= 16, skinny to M 65536, tiled above 64 when N >= 128 -- and between the fp8a8 pair's narrow
 * and tiled instantiations. r4d_rows.cpp's cGemmNt / cGemmNt64 / cGemmNtTiled carry the exact
 * bands; every K in this block is a multiple of 128, so all three of them are reachable.
 *
 * `x` is the activation's bf16 plane and `q`/`s` its E4M3 pair. The mode picks which is READ, not
 * which is passed, so the caller hands over what the wire carries and gets one graph. */
inline rad_op dflash2_dense_decl(RadBuilder* b, const Geom& g, bool bf16, int64_t max_rows,
                                 int64_t N, int64_t K, rad_buf x, rad_buf q, rad_buf s,
                                 rad_buf y, rad_weight w, rad_weight ws) {
    if (bf16)
        return rw(b, RAD_OP(b, "gemm_nt",
                      RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w)),
                  {x}, {y});
    return rw(b, RAD_OP(b, "gemm_nt_q",
                  RAD_PARAMS(RAD_RANGE("M", 1, max_rows), RAD_INT("N", N), RAD_INT("K", K),
                             RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
                  RAD_WEIGHTS(w, ws)),
              {q, s}, {y});
}

/* And the issue. `x` is the activation's bf16 plane, `q`/`s` its E4M3 pair; the bf16 row reads
 * `x` and the fp8 row reads the pair, which is why both are passed and one is ignored. The two
 * trailing operand slots `gemm_nt_q` carries (a_sum, b_ref) are absent with an fp8 grid and are
 * passed as RAD_NONE rather than left off -- operands are positional. */
inline void dflash2_dense_step(RadCtx* c, bool bf16, rad_op op, int64_t rows, RadOperand x,
                               RadOperand q, RadOperand s, RadOperand y, rad_weight w,
                               rad_weight ws) {
    if (bf16) { RAD_ISSUE_N(c, op, rows, x, RAD_W(w), y); return; }
    RAD_ISSUE_N(c, op, rows, q, s, RAD_W(w), RAD_W(ws), y, RAD_NONE, RAD_NONE);
}

struct Dflash2 {
    enum { MAX_LAYERS = 8 };

    /* Everything the drafter's own config.json says that its geometry does not. Read by the
     * plugin from `draft.*` metadata keys and handed in whole, so this file reads no environment
     * and no container. */
    struct Config {
        int64_t n_taps   = 0;     /* how many trunk layers `fc` concatenates */
        /* QUERY ROWS A SEQUENCE: the anchor plus `n_spec` mask tokens. A RUNTIME ROW COUNT, not
         * a property of the weights -- the convolution's position law is `t % block_size` and
         * every linear's K is `n_embd`. The checkpoint's declared block is the default; a plugin
         * passes whatever the deployment asked for. */
        int64_t block    = 0;
        int64_t taps     = 2;     /* the convolution's */
        int64_t group    = 16;
        int64_t sel_rank = 256;   /* the codebooks' width */
        int64_t sel_topk = 16;    /* candidates kept per position */
        int64_t window   = 0;     /* sliding attention width, 0 for none */
        int     causal   = 0;
        /* SERVE THE CHECKPOINT'S OWN bf16 WEIGHTS AND ACTIVATIONS. False is the fp8 drafter this
         * block was written for: rad_fp8.h's DenseFP8 over a bf16 checkpoint, E4M3 activations and
         * the 2-bit lm_head. True is a container converted WITHOUT a quantising recipe -- the
         * linears are `gemm_nt` over their bf16 planes, the head is the target's lm_head at the
         * precision it ships at, and no quantiser is declared or issued anywhere in the block.
         * The two modes share every shape, the selector and both passes; only the weight planes,
         * the GEMM rows and the head differ. The LEGACY fp8 deployment leaves this false and its
         * graph is the one it always was.
         *
         * THE METADATA KEY IS `draft.bf16`, integer: `rad_meta_geti(meta, "draft.bf16", 0) != 0`.
         * It is a `--set` the converter writes, not something derived from the checkpoint -- a
         * bf16 drafter and an fp8 one have the same tensors and differ in whether a recipe ran,
         * and nothing in the tensors says which deployment wants which. A plugin that reads it
         * must ALSO state `draft.dtype`/its geometry at "bf16", which declare() checks. */
        bool    bf16     = false;
        /* WHETHER THIS CALLER'S KERNEL HOME CARRIES THE `mine` PLUGIN. A caller integration
         * policy, read once at declare; it selects the fused ops below and is not a runtime knob.
         *
         * It has to be a STATEMENT because the ABI cannot be asked: an op name with no loaded
         * schema is RAD_E_NOSCHEMA from rad_decl_op, which fails the entire declare rather than
         * returning a miss, and rad_op_resolved answers only for a handle already declared.
         *
         * FALSE BY DEFAULT, so a caller that never sets it declares only ops every kernel set
         * publishes. `arch/qwen4exp_fp8` sets it true: it already declares `mine`-only `hc_mean`
         * unconditionally on the same path. `arch/qwen35_fp8` shares this block and leaves it. */
        bool    mine_fusions = false;
        /* ============================== THE BOUNDED DarTree ==============================
         *
         * ONE BACKBONE PASS, MANY PROPOSED ROWS. `dflash_select` walks ONE path through the
         * candidate lattice and proposes `steps` tokens, so a verify step can accept at most one
         * a position. The lattice holds far more than one path -- level l's candidates are
         * scored against level l+1's unconditionally -- so a tree over the SAME edges proposes
         * several continuations a level and the target accepts the right one whichever it is.
         *
         * THE BACKBONE DOES NOT CHANGE. `block` stays 4 (3 proposal levels, the trained depth):
         * the tree is built entirely by `dflash_tree_select` over the candidate planes the query
         * pass already produced, so there is no extra forward pass, no extra row, and no
         * extension of the trained depth. `tree_nodes` is how many VERIFICATION rows this
         * deployment has -- RadBuildCtx::max_spec, i.e. NUM_SPECULATIVE_TOKENS -- and the tree
         * is PRUNED to that many nodes, which is why it is a node budget and not a width.
         *
         * `RADIANCE_DARTREE=1` is read by the CALLER (arch/qwen4exp_fp8), so a deployment that
         * does not set it declares exactly the graph it always did. */
        bool    tree       = false;
        /* TOTAL verification rows, root included: 1 + the tokens this step may verify. */
        int64_t tree_nodes = 0;
        int64_t steps() const { return block - 1; }
        /* THE FRONTIER WIDTH THAT FILLS THE BUDGET.
         *
         * `tree_width` is `dflash_tree_select`'s FRONTIER size per level -- how many nodes survive
         * at one level, summed over the level's parents -- and NOT a per-parent branching factor.
         * So the capacity before pruning is `1 + steps * tree_width`, linear in both the level
         * count and the width; a value that grew as w^steps would declare a lattice the selector
         * never builds and, at w = 3, would ask for 40 nodes under a 16-row verifier.
         *
         * The level count is always the trained `steps`; only the frontier moves. `tree_budget` is
         * what the pruner targets, so a width that overshoots the budget is normal and the surplus
         * is dropped -- which is why this returns the smallest width that can REACH the budget and
         * not the one that exactly meets it. Bounded by `sel_topk`: a width wider than the
         * candidate plane has nothing behind it. */
        int64_t tree_width() const {
            if (!tree || tree_nodes <= 1) return 1;
            const int64_t per_level = tree_nodes - 1;      /* draft nodes to place */
            const int64_t st = steps() > 0 ? steps() : 1;
            const int64_t w = (per_level + st - 1) / st;   /* ceil: the smallest sufficient width */
            return w < 1 ? 1 : (w > sel_topk ? sel_topk : w);
        }
        int64_t ng(int64_t n_embd) const { return group ? n_embd / group : 0; }
        /* The kernel projection emits [2, taps, NG] per token -- one coefficient set for `prepare`
         * and one for `finish` -- and `dflash_conv` reads a slice of it at the FULL row pitch. */
        int64_t coef_dim(int64_t n_embd) const { return 2 * taps * ng(n_embd); }
    };

    struct Wire {
        /* the context pass */
        rad_buf src  = 0;   /* [max_tok, n_taps * n_embd]  the trunk's taps, read only */
        /* THE ACCEPTED ROWS OF `src`, GATHERED, when a context pass follows a TREE verify.
         *
         * An accepted path is a set of NODE ids, and the trunk rows those nodes occupy are
         * `starts[i] + node[j]` -- NOT consecutive. A winner of nodes {0, 2, 5} sits on trunk rows
         * 0, 2 and 5, and the sibling on row 1 is a continuation the target rejected. The context
         * pass stores K and V into a cache that OUTLIVES the step, so it may read exactly the
         * accepted rows; running over the mirror and masking a suffix would put rows BETWEEN
         * accepted ones into that cache.
         *
         * As TALL AS THE TAP PLANE, and on every context pass. The gather names rows through
         * `out_ids`, one entry a row of the pass, and a pass's row count is `max_tok` for a prefill
         * chunk -- the identity range, which the gather copies -- and up to the verifier's window
         * for a tree verify. A buffer sized for `block` rows a sequence would be overflowed by the
         * first prefill. `fc` reads the first `T` rows of it. */
        rad_buf srcg = 0;   /* [max_tok, n_taps * n_embd]  gathered context taps */
        rad_buf c    = 0;   /* [max_tok, n_embd]           fc's output, then normed in place */
        /* K AND V ARE TWO BUFFERS, not one [max_tok, 2*kv_dim] with K in its left half. Both
         * halves are written by one projection and read by one kv_store, so fusing them is the
         * obvious thing -- and it is wrong, because of the PER-HEAD NORM in between. `rmsnorm`
         * takes one row pitch. K inside a fused row is `head_dim` apart within a token and
         * `2*kv_dim` apart between tokens, and no single pitch is both; answering with the
         * between-tokens stride normalises every eighth head against the wrong numbers. libr4d
         * refuses that operand (libr4d/r4d_model.cpp, rows_pitch), and the drafter gives it rows
         * that are what they claim to be. The trunk's attention block splits
         * k and v for the same reason -- see rad_block_attn_gated.h. */
        rad_buf ck   = 0;   /* [max_tok, kv_dim]           ONE layer's K, reused across layers */
        rad_buf cv   = 0;   /* [max_tok, kv_dim]           ONE layer's V, reused across layers */
        /* the query pass */
        rad_buf x    = 0;   /* [qrows, n_embd]  the drafter's residual stream */
        rad_buf h    = 0;   /* [qrows, n_embd]  normed input, o/down output, and the final norm */
        rad_buf h2   = 0;   /* [qrows, n_embd]  the convolution's output -- it may not alias x */
        rad_buf coef = 0;   /* [qrows, coef_dim] */
        rad_buf q    = 0;   /* [qrows, q_dim] */
        rad_buf k    = 0;   /* [qrows, kv_dim]  -- two buffers, and `ck` above says why */
        rad_buf v    = 0;   /* [qrows, kv_dim] */
        rad_buf attn = 0;   /* [qrows, q_dim] */
        rad_buf gate_up = 0;/* [qrows, 2 * n_ff] */
        rad_buf ffn  = 0;   /* [qrows, n_ff] */
        /* THE FP8 TWINS OF THE FOUR ACTIVATIONS A LINEAR READS. `quant_act_fp8` writes them and
         * the six `gemm_nt_q` linears read them; nothing else touches them, and the bf16 buffer
         * beside each stays the thing the rest of the graph reads and writes.
         *
         * ONE PAIR PER ACTIVATION, NOT PER LINEAR. q and k|v both read h2, so quantising it once
         * is the difference between two launches a layer and three -- the same argument rad_fp8.h
         * makes for keeping QuantFP8 a separate struct from LinearFP8. h2 is quantised twice a
         * layer all the same, because the convolution rewrites it between the attention half and
         * the feed-forward half.
         *
         * NONE OF THEM IS USED IN Config::bf16 MODE, and a plugin that is declaring a bf16 drafter
         * should leave all five pairs at 0: no quantiser is declared there, so a plane nothing
         * writes is a plane the planner still charges. The block issues nothing against them. */
        rad_buf srcq = 0, srcs = 0;  /* [max_tok, n_taps*n_embd] e4m3 | [.., /128] f32 */
        rad_buf cq   = 0, cs   = 0;  /* [max_tok, n_embd] */
        rad_buf h2q  = 0, h2s  = 0;  /* [qrows,   n_embd] */
        rad_buf atq  = 0, ats  = 0;  /* [qrows,   q_dim] */
        rad_buf ffq  = 0, ffs  = 0;  /* [qrows,   n_ff] */
        /* the head and the selector, at one row per DRAFT POSITION (block - 1 a sequence) */
        rad_buf sel_h = 0;  /* [srows, n_embd]     the gathered final hidden states */
        rad_buf hp    = 0;  /* [srows, sel_rank]   hidden_projection of them */
        rad_buf dq    = 0;  /* [srows, n_embd]  i8   the 2-bit head's quantised activation */
        rad_buf ds    = 0;  /* [srows, 1]       f32  one scale a row */
        rad_buf dsum  = 0;  /* [n_embd/128, pad] f32 GROUP-major, quant_act_i8's asum plane */
        /* `dq`, `ds` and `dsum` belong to the 2-bit head alone: in Config::bf16 mode the head's
         * GEMM reads `sel_h` and none of the three is written or read, so a bf16 plugin leaves
         * them 0. `dlog` is the same plane in both modes -- bf16, [srows, head_rows] -- because
         * that is what `row_topk` reads and what the unquantised reference matmul produces. */
        rad_buf dlog  = 0;  /* [srows, head_rows(g)] bf16  the logits */
        rad_buf cand  = 0;  /* [srows, sel_topk] i32  candidate token ids, global */
        rad_buf unary = 0;  /* [srows, sel_topk] f32  and their logits */
        /* The vocab-parallel pair, declared only when head_sharded(g). Both are f32 (int32 id,
         * float value) pairs: this rank's top-R, and the all-gathered world_size * R of them. */
        rad_buf dpair = 0;  /* [srows, sel_topk, 2]         f32 */
        rad_buf dgath = 0;  /* [srows, world * sel_topk, 2] f32 */
        rad_buf tokens = 0; /* [max_seqs, steps] i32  THE DRAFT -- the engine reads this by name */
        /* THE ROOT-INCLUSIVE PACKED TREE, `[max_seqs, 33]` i32 -- dloop_dartree's layout, which
         * is what rad_declare_drafter_tree validates and what the engine reads back when a
         * program declared one. Declared by the CALLER and named here so `dflash_tree_select`
         * can write it; 0 leaves this deployment on the linear selector. */
        rad_buf packed = 0;
    };

    /* WHETHER THE 2-BIT DRAFT HEAD IS VOCAB-SHARDED. The test itself is rad_arch.h's
     * `vocab_head_sharded` -- shared, because the second block-diffusion drafter in this library
     * asks it too and both the op and the buffer under it have to agree. These two names stay so
     * that whoever declares Dflash2's buffers can ask the BLOCK rather than re-derive the rule.
     *
     * Sharding halves the widest read on the draft path -- 341 MiB of codes and scales at 248320
     * rows when it is replicated -- at the price of one all_gather of R pairs a row. At 248320
     * over two ranks the shard is 124160, which is 7760 whole 16-row blocks. */
    static bool    head_sharded(const Geom& g) { return vocab_head_sharded(g); }
    /* Rows of the draft head, and therefore the width of `dlog`, on THIS rank. */
    static int64_t head_rows(const Geom& g)    { return vocab_head_rows(g); }

    Geom    g{};      /* the DRAFTER's geometry, not the trunk's */
    Config  cfg{};
    Wire    w{};
    int     first_layer = 0;   /* absolute, so the KV group's layer_slot lookup finds these five */
    int64_t qrows = 0, crows = 0, srows = 0;
    int64_t gated_rows = 0;    /* the band op_gug was declared over; 0 when there is no fold */
    rad_kvgroup kv = 0;

    /* model-scope weights.
     *
     * THE SIX LINEARS ARE FP8 IN THE SHIPPED MODE AND THE CHECKPOINT IS BF16. rad_fp8.h's
     * DenseFP8 declares each as block fp8 and the recipe makes it so -- and the reason is bytes:
     * the drafter is read WHOLE on every decode step, which makes it one of the largest single
     * items in the decode budget after the trunk's own GEMMs.
     * Nothing it computes can change an emitted token, because every draft is verified against the
     * trunk exactly; the only thing quantising it can move is acceptance.
     *
     * Config::bf16 KEEPS THE SAME MEMBERS AND THE SAME STRUCT: `w` holds the checkpoint's bf16
     * tensor and `s` stays 0 (dflash2_dense above says how). The scale plane is not declared, so
     * a reader of this struct must not assume `s` is live -- `cfg.bf16` is the test. */
    DenseFP8   w_fc{};
    rad_weight w_hnorm = 0, w_norm = 0;
    rad_weight w_selp = 0, w_pred = 0, w_succ = 0;
    rad_weight w_tok = 0;                    /* the TARGET's embedding table */
    rad_weight w_dh = 0, w_dhs = 0;          /* the 2-bit copy of the TARGET's lm_head */

    /* per-layer weights */
    rad_weight w_anorm[MAX_LAYERS] = {}, w_fnorm[MAX_LAYERS] = {};
    /* K AND V ARE TWO WEIGHTS, and the fp8 layout hook is why. One fused [2*kv_dim, n_embd]
     * tensor issued a half at a time through wrow_slice would cost a launch and no bytes -- and a
     * block-scaled weight cannot be used that way. The layout of an operand is a function of the
     * OP's N, and this op's N is kv_dim because it computes one half; the hook would be handed a
     * [2*kv_dim/128][K/128] plane for a [kv_dim, K] op and refuse it. Two weights make the op's N
     * and the planes' rows the same number again, shard independently at kv_dim, and issue the
     * same two launches. */
    DenseFP8   w_q[MAX_LAYERS]{},  w_k[MAX_LAYERS]{}, w_v[MAX_LAYERS]{}, w_o[MAX_LAYERS]{};
    rad_weight w_qn[MAX_LAYERS] = {}, w_kn[MAX_LAYERS] = {};
    DenseFP8   w_gu[MAX_LAYERS]{}, w_dn[MAX_LAYERS]{};
    rad_weight w_acp[MAX_LAYERS] = {}, w_acb[MAX_LAYERS] = {};
    rad_weight w_fcp[MAX_LAYERS] = {}, w_fcb[MAX_LAYERS] = {};

    /* context-pass ops */
    rad_op op_cfc = 0, op_cnorm = 0, op_crope = 0, op_cstore = 0;
    rad_op op_ck[MAX_LAYERS] = {}, op_cv[MAX_LAYERS] = {}, op_cknorm[MAX_LAYERS] = {};

    /* THE ACTIVATION QUANTISERS, one per activation that feeds a linear. Weightless and shape-only,
     * so one declaration each serves every layer -- the per-layer ops above are per-layer only
     * because a weight's prefetch interval comes from the op that declares it. */
    rad_op op_qsrc = 0, op_qc = 0;                  /* the context pass's two */
    rad_op op_qh2 = 0, op_qat = 0, op_qff = 0;      /* the query pass's three */

    /* query-pass ops. The weightless ones are declared once and issued per layer: an op's declared
     * read and write sets are what the buffer planner reads, and they are the same every layer. */
    rad_op op_embed = 0, op_embed_ar = 0;
    rad_op op_anorm[MAX_LAYERS] = {}, op_fnorm[MAX_LAYERS] = {};
    rad_op op_acoef[MAX_LAYERS] = {}, op_fcoef[MAX_LAYERS] = {};
    rad_op op_qp[MAX_LAYERS] = {}, op_kp[MAX_LAYERS] = {}, op_vp[MAX_LAYERS] = {};
    rad_op op_op[MAX_LAYERS] = {};
    rad_op op_qn[MAX_LAYERS] = {}, op_kn[MAX_LAYERS] = {};
    rad_op op_gu[MAX_LAYERS] = {}, op_dn[MAX_LAYERS] = {};
    rad_op op_gug[MAX_LAYERS] = {};   /* the gate_up|silu|quant fold; 0 when nothing serves it */
    rad_op op_aconv[MAX_LAYERS] = {}, op_fconv[MAX_LAYERS] = {};
    /* THE CLOSING CONVOLUTION AND ITS RESIDUAL ADD IN ONE LAUNCH -- `dflash_conv_add`, and 0 on
     * every path that keeps the pair (see the declaration). Only the FINISH halves fuse: the
     * prepare convolutions feed a GEMM rather than the residual stream, so there is no add there
     * to fold and nothing changes for them. */
    rad_op op_aconvadd[MAX_LAYERS] = {}, op_fconvadd[MAX_LAYERS] = {};
    /* K-NORM + ROPE + KV-STORE IN ONE LAUNCH -- `dflash_k_norm_rope_store`, per layer and 0 on
     * every path that keeps the chain. Context and query both, since both run the same three ops
     * over their own K/V buffers. */
    rad_op op_ckstore[MAX_LAYERS] = {}, op_kstore[MAX_LAYERS] = {};
    rad_op op_qrope = 0, op_krope = 0, op_store = 0, op_attn = 0;
    rad_op op_act = 0, op_ar = 0, op_add = 0, op_final = 0;

    /* head and selector */
    rad_op op_gather = 0, op_dquant = 0, op_dlogits = 0, op_dtopk = 0;
    rad_op op_cgather = 0;   /* the context pass's accepted-row gather; 0 without Wire::srcg */
    rad_op op_dgather = 0, op_dmerge = 0;   /* the vocab-parallel pair; 0 when the head is whole */
    rad_op op_hp = 0, op_sel = 0;
    /* THE TREE SELECTOR (`Config::tree`). A SECOND selector over the same candidate planes, and
     * not a replacement for `op_sel`: a deployment on `RADIANCE_DARTREE=1` issues this one and
     * leaves the linear walk undeclared, because two selectors writing two proposal buffers would
     * both be read and neither would be the answer. `dflash_select`'s schema, its shim and its
     * reference rows are untouched, which is why the two can coexist in one kernel home. */
    rad_op op_tsel = 0;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int base_layer,
                 rad_kvgroup group, rad_weight tok_embd, const char* ck_prefix,
                 const char* ck_lm, const Wire& wire);
    /* The context pass, one row a committed token. `batch` carries those rows and the draft KV
     * group's slot mapping; nothing about the query block is read. */
    void context(RadCtx* c, const RadBatch* batch) const;
    /* The query pass and the selector, one call, `block` rows a sequence. */
    void step(RadCtx* c, const RadBatch* batch) const;

private:
    int64_t q_dim()  const { return g.n_head * g.head_dim; }
    int64_t kv_dim() const { return g.n_head_kv * g.head_dim; }
};

/* ------------------------------------------------------------------------------ declare */

inline int Dflash2::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c,
                            int base_layer, rad_kvgroup group, rad_weight tok_embd,
                            const char* ck, const char* ck_lm, const Wire& wire) {
    g = geom; cfg = c; w = wire; first_layer = base_layer; kv = group; w_tok = tok_embd;
    if (!ck || !ck_lm || !w_tok) return RAD_E_INVAL;
    if (g.n_layer <= 0 || g.n_layer > MAX_LAYERS) {
        fprintf(stderr, "radiance: dflash2 has %lld layers; this block carries at most %d\n",
                (long long)g.n_layer, MAX_LAYERS);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.block < 2 || cfg.n_taps <= 0 || cfg.sel_rank <= 0 || cfg.sel_topk <= 0)
        return RAD_E_INVAL;
    /* THE TREE'S OWN BOUNDS, refused here rather than at the op. `dflash_tree_select` refuses a
     * call outside 1..3 levels / 1..15 nodes / 1..top_k width, and a REFUSAL AT ISSUE IS A DEAD
     * STEP -- the launch was already enqueued. So the numbers this block passes are the ones its
     * kernel accepts, checked once when they are still a declaration. */
    if (cfg.tree) {
        if (cfg.block > 4) {
            fprintf(stderr, "radiance: dflash2: the tree selector serves at most 3 proposal "
                            "levels and this block asks for %lld. A tree may not propose deeper "
                            "than the drafter was trained.\n", (long long)cfg.steps());
            return RAD_E_UNSUPPORTED;
        }
        if (!w.packed) {
            fprintf(stderr, "radiance: dflash2: Config::tree needs a packed tree buffer "
                            "(Wire::packed); there is nowhere for a topology to land.\n");
            return RAD_E_INVAL;
        }
        if (cfg.tree_nodes < 2 || cfg.tree_nodes > 16) {
            fprintf(stderr, "radiance: dflash2: the tree's row budget %lld is outside the "
                            "verifier's 2..16.\n", (long long)cfg.tree_nodes);
            return RAD_E_INVAL;
        }
    }
    /* THE BF16 MODE IS A BF16 MODEL. Every linear in it is `gemm_nt` over a bf16 weight and a
     * bf16 activation, and libr4d's rows for that op are `dtype bf16` and nothing else -- so a
     * geometry that says anything else cannot be served here and is refused by name rather than
     * resolved into a kernel miss with a band in it. */
    if (cfg.bf16 && std::strcmp(g.dtype, "bf16") != 0) {
        fprintf(stderr, "radiance: dflash2 asked for the bf16 drafter (Config::bf16) on a '%s' "
                        "geometry. Its linears are plain bf16 weights over bf16 activations, "
                        "which is one dtype and not a flag.\n", g.dtype);
        return RAD_E_DTYPE;
    }

    const int64_t NG   = cfg.ng(g.n_embd);
    const int64_t CD   = cfg.coef_dim(g.n_embd);
    const int64_t qd   = q_dim(), kvd = kv_dim();

    crows = g.max_tok;
    qrows = g.max_seqs * cfg.block;
    if (qrows > g.max_tok) qrows = g.max_tok;
    srows = g.max_seqs * cfg.steps();
    if (qrows <= 0 || srows <= 0) return RAD_E_INVAL;

    /* ---------------------------------------------------------------- model-scope weights */
    RAD_ARCH_TRY(dflash2_dense(b, nm, w_fc, cfg.bf16, nm.f("dflash.fc"), g.n_embd,
                               cfg.n_taps * g.n_embd, RAD_SHARD_NONE, grp_model(),
                               { nm.ckpt("%s.fc.weight", ck) }));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.hidden_norm.weight"), nm.ckpt("%s.hidden_norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.norm.weight"),        nm.ckpt("%s.norm.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_proj.weight"),
                          nm.ckpt("%s.candidate_selector.hidden_projection.weight", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_pred"),
                          nm.ckpt("%s.candidate_selector.predecessor_codebook", ck)));
    RAD_ARCH_TRY(map_copy(b, nm.f("dflash.sel_succ"),
                          nm.ckpt("%s.candidate_selector.successor_codebook", ck)));

    w_hnorm = decl_w(b, nm.f("dflash.hidden_norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_norm  = decl_w(b, nm.f("dflash.norm.weight"), RAD_F32, {g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_selp  = decl_w(b, nm.f("dflash.sel_proj.weight"), RAD_BF16, {cfg.sel_rank, g.n_embd},
                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    w_pred  = decl_w(b, nm.f("dflash.sel_pred"), RAD_BF16, {g.n_vocab_all, cfg.sel_rank},
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    w_succ  = decl_w(b, nm.f("dflash.sel_succ"), RAD_BF16, {g.n_vocab_all, cfg.sel_rank},
                     RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());

    /* THE DRAFT HEAD, IN THE MODE'S PRECISION.
     *
     * bf16: `dflash.draft_head.weight` IS the target's lm_head tensor -- the same name map, the
     * same vocab shard, the same ROW split -- declared at its own dtype. No scale plane and no
     * codes: `gemm_nt` reads the bf16 rows the checkpoint ships. The selector is unchanged,
     * because `row_topk` reads the same [srows, head_rows] bf16 logit plane in both modes and the
     * candidate merge below it is a property of the SHARD and not of the head's precision. This is
     * the "no approximation on the wire" mode: what the drafter proposes is exactly what the
     * target's own head would give for those rows.
     *
     * fp8: THE 2-BIT DRAFT HEAD. The same declaration the MTP head makes, for the same reasons,
     * argued at length in rad_block_mtp_fp8.h: `dflash.draft_head.weight` is the target's lm_head
     * under a name of its own, which the recipe quantises to the 2-bit grid (u2 codes, an f16
     * scale and a u8 zero a group of 128) -- one checkpoint tensor, two logical weights. The GEMM
     * takes the codes as one operand and the scale and zero planes, interleaved by its relayout,
     * as the other.
     *
     * IT IS SHARDED, AND THE MERGE IS WHAT MAKES THAT SAFE. The draft path never goes through the
     * sampler, so its argmax is a `row_topk` over one rank's plane -- and a sharded plane on its
     * own would let each rank propose only from its own half, while the selector's codebooks are
     * gathered by GLOBAL token id and would then see only the candidates that landed locally. The
     * merge runs BEFORE the selector: each rank's top-R comes back as global ids, an all_gather
     * moves R pairs a row, and `row_topk_merge` reduces them to the same global top-R an unsharded
     * head would have found -- exactly, ties included (docs/OPS.md). Both ranks then hold the same
     * candidate set and the codebooks are replicated, so everything downstream is unchanged.
     *
     * What it buys is the widest read on the draft path halved -- 341 MiB of codes and scales at
     * the whole vocabulary -- and that much device memory given back per rank. What it costs is
     * one all_gather of `world * R` pairs a row, which at R = 16 is 2 KiB: the link's latency and
     * essentially no bytes. In bf16 mode the same split halves the head's 1.27 GB (1.18 GiB) of
     * bf16 rows instead, at the same cost and with the same merge.
     *
     * SHARDING IS DECLINED RATHER THAN FORCED for a shard the packed layouts cannot express: a
     * rank whose share is not a whole number of rows for the head, or an unaligned vocabulary. The
     * test is rad_arch.h's `vocab_head_sharded` for both modes -- what it refuses is the SHARD, so
     * a bf16 head that fails it stays whole on every rank and the merge ops are not declared. */
    enum { DF_W2_GROUP = 128 };
    const bool dh_shard = head_sharded(g);
    const int  dh_split = dh_shard ? RAD_SHARD_ROW : RAD_SHARD_NONE;
    const char* dhn = nm.f("dflash.draft_head.weight");
    RAD_ARCH_TRY(map_copy(b, dhn, ck_lm));
    if (cfg.bf16) {
        /* THE bf16 HEAD IS THE TARGET'S TENSOR AS IT SHIPS. A target container whose recipe made
         * its lm_head block fp8 has no bf16 head rows to read, and dequantising them here would be
         * a lossy wire this mode exists to avoid -- rad-convert's job, or pair the drafter with a
         * bf16 target. Unknown is not refused (a builder with no model behind it), the same rule
         * dflash2_dense keeps. */
        RadEncoding enc{};
        if (weight_enc(b, dhn, &enc) &&
            !(rad_enc_is(&enc, "plain") && enc.n_planes == 1 && enc.plane[0].dtype == RAD_BF16)) {
            char e[256];
            rad_enc_format(&enc, e, sizeof e);
            fprintf(stderr, "radiance: dflash2 is serving bf16 (Config::bf16) and the target's "
                            "lm_head ('%s') is stored as %s. The bf16 drafter reads the target's "
                            "head at the precision it ships at; pair it with a bf16 target, or "
                            "convert one.\n", ck_lm, e);
            return RAD_E_DTYPE;
        }
        /* RAD_ACCESS_PER_TOKEN, the same class the 2-bit head and the MTP head declare. It is read
         * once per draft pass on every step, which is what that class means; the vocabulary class
         * beside it describes the TARGET's own lm_head, whose residency the sampler's path owns. */
        w_dh = decl_w(b, dhn, RAD_BF16, {head_rows(g), g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split,
                      grp_model());
        w_dhs = 0;   /* no scale plane: `gemm_nt` reads the weight itself */
        if (!w_dh) return RAD_E_INVAL;
    } else {
        w_dh  = decl_view(b, dhn, nullptr, "codes", RAD_U2, {head_rows(g), g.n_embd},
                          RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
        w_dhs = decl_view(b, nm.f("dflash.draft_head.scale"), dhn, "scale,zero", RAD_F16,
                          {head_rows(g), g.n_embd}, RAD_ACCESS_PER_TOKEN, dh_split, grp_model());
        if (!w_dh || !w_dhs) return RAD_E_INVAL;
    }

    /* ---------------------------------------------------------------- per-layer weights */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.input_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_norm.weight", i),
                              nm.ckpt("%s.layers.%d.post_attention_layernorm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.q_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i),
                              nm.ckpt("%s.layers.%d.self_attn.k_norm.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_conv_proj.weight", i),
                              nm.ckpt("%s.layers.%d.attention_conv.kernel_projection.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.attn_conv_base", i),
                              nm.ckpt("%s.layers.%d.attention_conv.base_kernel", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_conv_proj.weight", i),
                              nm.ckpt("%s.layers.%d.mlp_conv.kernel_projection.weight", ck, i)));
        RAD_ARCH_TRY(map_copy(b, nm.f("dflash.blk.%d.ffn_conv_base", i),
                              nm.ckpt("%s.layers.%d.mlp_conv.base_kernel", ck, i)));

        w_anorm[i] = decl_w(b, nm.f("dflash.blk.%d.attn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        w_fnorm[i] = decl_w(b, nm.f("dflash.blk.%d.ffn_norm.weight", i), RAD_F32, {g.n_embd},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        RAD_ARCH_TRY(dflash2_dense(b, nm, w_q[i], cfg.bf16, nm.f("dflash.blk.%d.attn_q", i),
                                   qd, g.n_embd, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                   { nm.ckpt("%s.layers.%d.self_attn.q_proj.weight", ck, i) }));
        /* K, V AND Q ARE THREE WEIGHTS, which is not the fusion every other attention block in
         * this library makes. Q is separate because of the context pass: it needs k and v over the
         * trunk's projected hidden state and has no use for q at all, so a fused qkv would make it
         * compute and discard a q_dim-wide GEMM per layer per context token. K and V are separate
         * because a block-scaled weight's layout follows its op's N, and both passes take one of
         * them at a time -- see the note beside the declarations. */
        RAD_ARCH_TRY(dflash2_dense(b, nm, w_k[i], cfg.bf16, nm.f("dflash.blk.%d.attn_k", i),
                                   kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                   { nm.ckpt("%s.layers.%d.self_attn.k_proj.weight", ck, i) }));
        RAD_ARCH_TRY(dflash2_dense(b, nm, w_v[i], cfg.bf16, nm.f("dflash.blk.%d.attn_v", i),
                                   kvd, g.n_embd, RAD_SHARD_ROW, grp_layer(first_layer + i),
                                   { nm.ckpt("%s.layers.%d.self_attn.v_proj.weight", ck, i) }));
        w_qn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_q_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        w_kn[i]    = decl_w(b, nm.f("dflash.blk.%d.attn_k_norm.weight", i), RAD_F32, {g.head_dim},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(first_layer + i));
        RAD_ARCH_TRY(dflash2_dense(b, nm, w_o[i], cfg.bf16, nm.f("dflash.blk.%d.attn_output", i),
                                   g.n_embd, qd, RAD_SHARD_COL, grp_layer(first_layer + i),
                                   { nm.ckpt("%s.layers.%d.self_attn.o_proj.weight", ck, i) }));
        RAD_ARCH_TRY(dflash2_dense(b, nm, w_gu[i], cfg.bf16, nm.f("dflash.blk.%d.ffn_gate_up", i),
                                   2 * g.n_ff, g.n_embd, RAD_SHARD_ROW,
                                   grp_layer(first_layer + i),
                                   { nm.ckpt("%s.layers.%d.mlp.gate_proj.weight", ck, i),
                                     nm.ckpt("%s.layers.%d.mlp.up_proj.weight", ck, i) },
                                   { g.n_ff, g.n_ff }));
        RAD_ARCH_TRY(dflash2_dense(b, nm, w_dn[i], cfg.bf16, nm.f("dflash.blk.%d.ffn_down", i),
                                   g.n_embd, g.n_ff, RAD_SHARD_COL, grp_layer(first_layer + i),
                                   { nm.ckpt("%s.layers.%d.mlp.down_proj.weight", ck, i) }));
        w_acp[i]   = decl_w(b, nm.f("dflash.blk.%d.attn_conv_proj.weight", i), RAD_BF16,
                            {CD, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_acb[i]   = decl_w(b, nm.f("dflash.blk.%d.attn_conv_base", i), RAD_BF16,
                            {2, cfg.taps, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_fcp[i]   = decl_w(b, nm.f("dflash.blk.%d.ffn_conv_proj.weight", i), RAD_BF16,
                            {CD, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));
        w_fcb[i]   = decl_w(b, nm.f("dflash.blk.%d.ffn_conv_base", i), RAD_BF16,
                            {2, cfg.taps, g.n_embd}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                            grp_layer(first_layer + i));

    }

    /* NON-CAUSAL, WINDOWED. The block is denoised as a whole, so every query row sees every other
     * row of its own block as well as the context -- `causal` 0 is the drafter's own config
     * (`is_causal: false`) and not a default. The window is the checkpoint's `sliding_window`, and
     * it is the reason the context pass has a bounded amount of real work in it.
     *
     * DECLARED FIRST, before either kv_store, even though it runs long after both. `block_size` is
     * RAD_DERIVED -- the KV block is a property of the resolved attention kernel and not a number
     * anyone gets to choose -- and the RAD_KV_BLOCK here only PINS the 16 the drafter's kernels
     * require. The two stores do not repeat the constant: they READ the resolved value back with
     * rad_kv_block_size() and declare it as their own parameter, and ask before the attention op is
     * declared and the answer is zero. */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                    RAD_PARAMS(RAD_RANGE("q_len", 1, cfg.block), RAD_INT("head_dim", g.head_dim),
                               RAD_INT("gqa", g.gqa()), RAD_INT("causal", cfg.causal),
                               RAD_INT("window", cfg.window), RAD_INT("n_head", g.n_head),
                               RAD_INT("block_size", RAD_KV_BLOCK),
                               RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                               RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                    RAD_NOWEIGHTS),
                {w.q}, {w.attn});

    /* ---------------------------------------------------------------- the context pass */
    /* IN THE SHIPPED MODE EVERY LINEAR HERE IS `fp8a8`, and every activation that feeds one gets a
     * `quant_act_fp8` in front of it. The weights are E4M3 with a 128x128 scale plane (rad_fp8.h's
     * DenseFP8 says why), and there is no `fp8a16` band anywhere in this block: that kernel is
     * M <= 1, and neither pass here ever runs at one row -- a context pass is a committed chunk
     * and a query pass is `block` rows by construction.
     *
     * IN Config::bf16 MODE every one of them is `gemm_nt` over the checkpoint's own bf16 weight and
     * the bf16 activation, and the quantisers above are not declared at all. dflash2_dense_decl
     * makes the choice, so this loop is the same loop in both modes.
     *
     * AND NOTHING HERE IS ROW-SLICED. The bf16 GEMM row covers M to 65536 and the fp8a8 pair covers
     * every M (narrow to 16, tiled above), so a context pass over a 2048-token prefill chunk is one
     * launch a layer rather than one per 64 rows.
     *
     * The convolutions' coefficient projections and the selector's hidden projection are bf16 in
     * both modes, and they are not sliced either: that GEMM row is C_LE("M", 65536) with
     * C_DIV("K", 16), and all three of these carry K = n_embd, so the wide row serves them whole.
     * Their output width is 640 (2*2*160 groups) and 256 (the selector rank), neither of which a
     * tiled row requires: the tiled row's N >= 128 is met and its M >= 65 is the prefill band. */
    /* IN EITHER MODE `op_cfc` is the same declaration over the same output buffer; the mode picks
     * whether it reads `src` or the E4M3 pair, and in bf16 mode no quantiser is declared at all. */
    if (!cfg.bf16) {
        op_qsrc = rw(b, RAD_OP(b, "quant_act_fp8",
                        RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", cfg.n_taps * g.n_embd),
                                   RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.src}, {w.srcq, w.srcs});
    }
    /* THE ACCEPTED-ROW GATHER, declared when the caller gave it a plane to land in. It runs over
     * the whole tap plane and emits rows at `batch->out_ids`, which is the one staging field the
     * engine already fills with exactly the rows a pass wants: for a context pass following a tree
     * verify those are the accepted path's TRUNK rows, in path order. Declared only against the
     * bf16 source plane because `fc` reads whichever buffer this op wrote.
     *
     * `M` IS THE TAP PLANE'S HEIGHT, not `max_seqs * block`. The gather runs on EVERY context pass,
     * and a prefill chunk emits its identity range -- as many rows as the chunk has tokens. A tree
     * verify is bounded by the verifier's own window, which the tree flag makes larger than `block`
     * by construction (NUM_SPECULATIVE_TOKENS is the row budget). The destination is declared to
     * the same extent below, and `fc` reads the first `T` rows of it. */
    if (w.srcg) {
        op_cgather = rw(b, RAD_OP(b, "gather_rows",
                         RAD_PARAMS(RAD_RANGE("M", 1, crows),
                                    RAD_INT("n", cfg.n_taps * g.n_embd), RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {w.src}, {w.srcg});
        if (!op_cgather) return RAD_E_UNSUPPORTED;
    }
    /* `fc` READS THE PLANE THE PASS ACTUALLY FILLED. An op's operands are fixed at declare, so a
     * gather that ran only sometimes would leave `fc` reading a stale plane on the passes that
     * skipped it. The gather therefore runs on EVERY context pass when a plane exists, naming the
     * rows that pass wants: the accepted path's trunk rows after a tree verify, and the identity
     * row range for a prefill chunk. One code path, and no pass can read a plane nothing wrote.
     *
     * The cost is one bandwidth-bound copy of the tap plane a context pass, against an `fc` that
     * streams the same bytes immediately after it. */
    const rad_buf csrc = w.srcg ? w.srcg : w.src;
    op_cfc = dflash2_dense_decl(b, g, cfg.bf16, crows, g.n_embd, cfg.n_taps * g.n_embd,
                                csrc, w.srcq, w.srcs, w.c, w_fc.w, w_fc.s);
    /* IN PLACE. Safe for any implementation that computes a row's scale before writing the row,
     * which none can avoid doing; the per-head q/k norms below run in place for the same reason. */
    op_cnorm = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_hnorm)),
                 {w.c}, {w.c});
    /* ONE QUANTISE FOR ALL FIVE LAYERS. `c` is the normed projection of the trunk's taps and every
     * layer's k|v reads that same buffer, so this is one launch a context pass and not five. */
    if (!cfg.bf16) {
        op_qc = rw(b, RAD_OP(b, "quant_act_fp8",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("n", g.n_embd),
                                 RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.c}, {w.cq, w.cs});
    }
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        /* TWO OPS OF THE SAME SHAPE, one per weight. An op's declared weight list is what gives a
         * weight its prefetch interval and its residency class, so two weights want two ops even
         * where one would issue both. */
        op_ck[i] = dflash2_dense_decl(b, g, cfg.bf16, crows, kvd, g.n_embd,
                                      w.c, w.cq, w.cs, w.ck, w_k[i].w, w_k[i].s);
        op_cv[i] = dflash2_dense_decl(b, g, cfg.bf16, crows, kvd, g.n_embd,
                                      w.c, w.cq, w.cs, w.cv, w_v[i].w, w_v[i].s);
        op_cknorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                             RAD_PARAMS(RAD_RANGE("M", 1, crows * g.n_head_kv),
                                        RAD_INT("n", g.head_dim), RAD_F64("eps", g.eps),
                                        RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(w_kn[i])),
                         {w.ck}, {w.ck});
    }
    /* K-ONLY ROPE. `rope` rotates the first n_head + n_head_kv heads of its operand and leaves the
     * V heads that follow alone, so a K-only tensor is described by n_head = n_head_kv and
     * n_head_kv = 0: all the heads there are rotate, none follow them. Weightless, so one op
     * serves every layer. */
    op_crope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.ck}, {w.ck});
    op_cstore = rw(b, RAD_OP(b, "kv_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, crows), RAD_INT("head_dim", g.head_dim),
                                 RAD_INT("n_head_kv", g.n_head_kv),
                                 RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                 RAD_STR("kv_dtype", g.kv_dtype)),
                      RAD_NOWEIGHTS),
                  {w.ck, w.cv}, {});

    /* ============================== K NORM + ROPE + STORE, IN ONE LAUNCH
     *
     * `dflash_k_norm_rope_store` is `rmsnorm` -> `rope` -> `kv_store` over the SAME buffers: K is
     * normed and rotated IN PLACE, then K and V are scattered into the paged cache. Both bf16
     * roundings are kept -- the norm's store, then the rotation's -- and the rotation is computed
     * from `theta` with the same `sincosf`, so there is no cos/sin plane. kv_store's slot rule is
     * unchanged: a negative or past-the-end slot writes nothing, and K still leaves normed and
     * rotated.
     *
     * SAME TWO CONDITIONS AS THE CLOSING CONVOLUTION: `Config::mine_fusions` (schema is `mine`'s),
     * and the row's geometry -- head_dim 128, NeoX over the full head, bf16 K/V/CACHE (the cache
     * because the op's layout is the bf16 paged one). Anything else leaves the three ops above.
     *
     * DECLARED AFTER `attn_paged` because `block_size` is read back from the resolved attention
     * kernel, the same dependency `kv_store` above has. */
    const bool kstore_geom = cfg.mine_fusions && g.head_dim == 128 &&
                             std::strcmp(g.dtype, "bf16") == 0 &&
                             std::strcmp(g.kv_dtype, "bf16") == 0;
    if (kstore_geom) {
        const RadParam tp[] = {
            RAD_RANGE("M", 1, crows), RAD_INT("n_head_kv", g.n_head_kv),
            RAD_INT("head_dim", g.head_dim), RAD_F64("eps", g.eps),
            RAD_F64("wadd", g.wadd_qk), RAD_F64("theta", g.theta),
            RAD_F64("scale", g.rope_scale), RAD_INT("block_size", rad_kv_block_size(b, kv)),
            RAD_STR("kv_dtype", g.kv_dtype)
        };
        const int np = (int)(sizeof(tp) / sizeof(tp[0]));
        for (int64_t l = 0; l < g.n_layer; ++l)
            op_ckstore[(int)l] = rw(b, rad_decl_op(b, "dflash_k_norm_rope_store", tp, np,
                                                   &w_kn[(int)l], 1),
                                    {w.ck, w.cv}, {w.ck});
    }

    /* ---------------------------------------------------------------- the query pass */
    op_embed = rw(b, RAD_OP(b, "embed_lookup",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n_embd", g.n_embd),
                                RAD_INT("n_vocab", g.n_vocab), RAD_STR("dtype", g.dtype),
                                RAD_INT("vocab_offset", g.vocab_off)),
                     RAD_WEIGHTS(w_tok)),
                 {}, {w.x});
    /* The trunk's embedding is vocab-sharded, so this gather is partial and takes the same exact
     * reduction. It is on the DRAFT path, which runs once a step ahead of the trunk: a second
     * small collective. */
    if (g.world > 1)
        op_embed_ar = rw(b, RAD_OP(b, "all_reduce",
                            RAD_PARAMS(RAD_INT("world_size", g.world),
                                       RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                       RAD_STR("dtype", g.dtype), RAD_INT("exact", 1)),
                            RAD_NOWEIGHTS),
                        {w.x}, {w.x});
    /* The query pass's three quantisers, declared once and issued per layer -- weightless ops with
     * the same shape every layer, exactly like the ropes and the residual add below. h2 is
     * quantised TWICE a layer because the convolution rewrites it between the two halves; attn and
     * ffn once each. NOT DECLARED IN bf16 MODE: nothing reads their codes there. */
    if (!cfg.bf16) {
        op_qh2 = rw(b, RAD_OP(b, "quant_act_fp8",
                       RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                  RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                   {w.h2}, {w.h2q, w.h2s});
        op_qat = rw(b, RAD_OP(b, "quant_act_fp8",
                       RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", qd),
                                  RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                   {w.attn}, {w.atq, w.ats});
        op_qff = rw(b, RAD_OP(b, "quant_act_fp8",
                       RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                                  RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
                       RAD_NOWEIGHTS),
                   {w.ffn}, {w.ffq, w.ffs});
    }

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_anorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_anorm[i])),
                        {w.x}, {w.h});
        op_acoef[i] = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", CD),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_acp[i])),
                        {w.h}, {w.coef});
        op_qp[i] = dflash2_dense_decl(b, g, cfg.bf16, qrows, qd, g.n_embd,
                                      w.h2, w.h2q, w.h2s, w.q, w_q[i].w, w_q[i].s);
        op_kp[i] = dflash2_dense_decl(b, g, cfg.bf16, qrows, kvd, g.n_embd,
                                      w.h2, w.h2q, w.h2s, w.k, w_k[i].w, w_k[i].s);
        op_vp[i] = dflash2_dense_decl(b, g, cfg.bf16, qrows, kvd, g.n_embd,
                                      w.h2, w.h2q, w.h2s, w.v, w_v[i].w, w_v[i].s);
        op_qn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_qn[i])),
                     {w.q}, {w.q});
        op_kn[i] = rw(b, RAD_OP(b, "rmsnorm",
                         RAD_PARAMS(RAD_RANGE("M", 1, qrows * g.n_head_kv), RAD_INT("n", g.head_dim),
                                    RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_WEIGHTS(w_kn[i])),
                     {w.k}, {w.k});
        op_op[i] = dflash2_dense_decl(b, g, cfg.bf16, qrows, g.n_embd, qd,
                                      w.attn, w.atq, w.ats, w.h, w_o[i].w, w_o[i].s);
        op_fnorm[i] = rw(b, RAD_OP(b, "rmsnorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                       RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fnorm[i])),
                        {w.x}, {w.h});
        op_fcoef[i] = rw(b, RAD_OP(b, "gemm_nt",
                            RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("N", CD),
                                       RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fcp[i])),
                        {w.h}, {w.coef});
        op_gu[i] = dflash2_dense_decl(b, g, cfg.bf16, qrows, 2 * g.n_ff, g.n_embd,
                                      w.h2, w.h2q, w.h2s, w.gate_up, w_gu[i].w, w_gu[i].s);
        /* THE SAME FOLD THE TRUNK'S MLP HAS, over the same weights and into the same activation;
         * rad_block_mlp_fp8.h declares it beside the pair for the same reason. Unfused, a layer
         * runs gate_up -> silu_mul -> quant, which is two extra launches and two extra passes over
         * the [T, n_ff] product. The gated kernel is also the faster of the two on the SAME shape,
         * because the fold saves the bf16 round trip.
         *
         * NOT RAD_ARCH_TRY: a kernel set without the row is the ordinary case and the pair below
         * still runs. `gated_rows` is the band it was declared over and the issue site checks it,
         * because M > 64 is a refusal in the kernel and not a slower path.
         *
         * The fold writes quantised planes; the pair below writes `ffn`. */
        if (!cfg.bf16) {
            const int64_t hi = qrows < 64 ? qrows : 64;
            if (hi > 0) {
                op_gug[i] = rw(b, RAD_OP(b, "gemm_nt_q_gated",
                                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", 2 * g.n_ff),
                                              RAD_INT("K", g.n_embd),
                                              RAD_INT("group", RAD_FP8_BLOCK),
                                              RAD_STR("dtype", "fp8a8"), RAD_STR("act", "silu")),
                                   RAD_WEIGHTS(w_gu[i].w, w_gu[i].s)),
                               {w.h2q, w.h2s}, {w.ffq, w.ffs});
                if (op_gug[i]) gated_rows = hi;
            }
        }
        op_dn[i] = dflash2_dense_decl(b, g, cfg.bf16, qrows, g.n_embd, g.n_ff,
                                      w.ffn, w.ffq, w.ffs, w.h, w_dn[i].w, w_dn[i].s);
    }

    /* The convolutions. `block_size` is the QUERY BLOCK -- the mask that kills the shifted tap at
     * the first row of each block is what keeps one sequence's block from convolving with the
     * previous sequence's, so it is the batch's per-sequence row count and not a tuning number.
     *
     * ONE OP PER (LAYER, CONVOLUTION) even though every one of them is the same shape, because the
     * base kernel is a WEIGHT and an op's declared weight list is what gives a weight its prefetch
     * interval and its residency class. A single shared declaration would leave ten weights that
     * no op admits to using. Each is issued twice -- `prepare` and `finish` -- against the two
     * halves of one [2, taps, n_embd] tensor. */
    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        op_aconv[i] = rw(b, RAD_OP(b, "dflash_conv",
                            RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                       RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                       RAD_INT("NG", NG), RAD_INT("block_size", cfg.block),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_acb[i])),
                        {w.h, w.coef}, {w.h2});
        op_fconv[i] = rw(b, RAD_OP(b, "dflash_conv",
                            RAD_PARAMS(RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
                                       RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group),
                                       RAD_INT("NG", NG), RAD_INT("block_size", cfg.block),
                                       RAD_STR("dtype", g.dtype)),
                            RAD_WEIGHTS(w_fcb[i])),
                        {w.h, w.coef}, {w.h2});
    }

    /* ============================== THE CLOSING CONVOLUTION AND ITS ADD, IN ONE LAUNCH
     *
     * `dflash_conv_add` is `dflash_conv` followed by `add`, keeping the pair's bf16 rounding
     * between them: the f32 accumulator is rounded to bf16 and THEN added to the residual
     * (mine/dflash_conv_add.h). BOTH closing sites of a layer -- attention and feed-forward -- and
     * nothing else: the PREPARE convolutions feed a GEMM, so there is no add to fold there.
     *
     * BF16 MODE AND `Config::mine_fusions` ONLY. That op's schema is `mine`'s alone, and it has to
     * be the caller's statement rather than a probe: an unknown op name is RAD_E_NOSCHEMA from
     * Builder::decl_op, which fails the whole declare, and nothing exported answers "is this op
     * known" before declaring. Default false, so a caller without the plugin runs the pair;
     * `arch/qwen4exp_fp8` sets it true because it already declares `mine`-only `hc_mean`.
     *
     * RESOLVE MISSES STILL FALL BACK: the pair above is always declared, an unresolvable band here
     * returns RAD_NULL_HANDLE rather than an error, and step() issues the pair. GEOMETRY is the
     * row's specialisation (taps 2, group 16, hidden_size % 64) and is asked here because a
     * refusal AT ISSUE fails the step instead of falling back; it is also what makes the
     * activations and the coefficient slice 16-byte addressable.
     *
     * ONE DECLARATION A LAYER AND SITE, each naming its OWN base kernel weight, so its prefetch
     * interval and residency class attach to it. OPERANDS: x, delta, base, residual, out, with
     * `out` IS `residual`. The delta is the same `bcol(coef, half, half, T)` slice, whose FULL row
     * pitch `2 * taps * NG` is the one number a shape check cannot derive. */
    const bool convadd_geom = cfg.bf16 && cfg.mine_fusions && cfg.taps == 2 &&
                              cfg.group == 16 && g.n_embd > 0 && (g.n_embd % 64) == 0;
    if (convadd_geom) {
        const RadParam tp[] = {
            RAD_RANGE("T", 1, qrows), RAD_INT("hidden_size", g.n_embd),
            RAD_INT("taps", cfg.taps), RAD_INT("group_size", cfg.group), RAD_INT("NG", NG),
            RAD_INT("block_size", cfg.block), RAD_STR("dtype", g.dtype)
        };
        const int np = (int)(sizeof(tp) / sizeof(tp[0]));
        for (int64_t l = 0; l < g.n_layer; ++l) {
            const int i = (int)l;
            rad_op*      dst[2] = { &op_aconvadd[i], &op_fconvadd[i] };
            rad_weight   wb[2]  = { w_acb[i], w_fcb[i] };
            for (int s = 0; s < 2; ++s)
                *dst[s] = rw(b, rad_decl_op(b, "dflash_conv_add", tp, np, &wb[s], 1),
                             {w.h, w.coef, w.x}, {w.x});
        }
    }
    op_qrope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.q});
    op_krope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head_kv), RAD_INT("n_head_kv", 0),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", "neox")),
                     RAD_NOWEIGHTS),
                 {w.k}, {w.k});
    op_store = rw(b, RAD_OP(b, "kv_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.k, w.v}, {});
    /* The query pass's own rows, the same geometry test: the band is `qrows` here, and an op's M
     * range is its band, so this is a second declaration and not a reuse. */
    if (kstore_geom) {
        const RadParam tp[] = {
            RAD_RANGE("M", 1, qrows), RAD_INT("n_head_kv", g.n_head_kv),
            RAD_INT("head_dim", g.head_dim), RAD_F64("eps", g.eps),
            RAD_F64("wadd", g.wadd_qk), RAD_F64("theta", g.theta),
            RAD_F64("scale", g.rope_scale), RAD_INT("block_size", rad_kv_block_size(b, kv)),
            RAD_STR("kv_dtype", g.kv_dtype)
        };
        const int np = (int)(sizeof(tp) / sizeof(tp[0]));
        for (int64_t l = 0; l < g.n_layer; ++l)
            op_kstore[(int)l] = rw(b, rad_decl_op(b, "dflash_k_norm_rope_store", tp, np,
                                                  &w_kn[(int)l], 1),
                                   {w.k, w.v}, {w.k});
    }
    op_act = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_ff),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.gate_up}, {w.ffn});
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                      RAD_PARAMS(RAD_INT("world_size", g.world),
                                 RAD_RANGE("numel", g.n_embd, qrows * g.n_embd),
                                 RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                 RAD_INT("min_bytes", g.wire_min_bytes)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.h});
    op_add = rw(b, RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.x, w.h2}, {w.x});
    op_final = rw(b, RAD_OP(b, "rmsnorm",
                     RAD_PARAMS(RAD_RANGE("M", 1, qrows), RAD_INT("n", g.n_embd),
                                RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_norm)),
                 {w.x}, {w.h});

    /* ---------------------------------------------------------------- the head and the selector
     *
     * THE MASK ROWS, offsets 1..block-1 of each sequence's block. Offset 0 is the ANCHOR -- the
     * token the target has already chosen -- and its row predicts nothing: it is there so the mask
     * rows have something to attend to. Each mask row predicts the token at ITS OWN position, so
     * the head runs over `steps` rows a sequence and not `block`. Getting this off by one is the
     * kind of mistake that costs nothing visible: the drafts are simply wrong, acceptance
     * collapses, and the model's own output stays exactly correct.
     *
     * The gather is what makes those rows contiguous, and it is also what fixes the head's row
     * numbering at s * steps + l -- which is the numbering `dflash_select` reads. */
    op_gather = rw(b, RAD_OP(b, "gather_rows",
                      RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                                 RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.h}, {w.sel_h});
    op_dquant = cfg.bf16
        ? 0
        : rw(b, RAD_OP(b, "quant_act_i8",
                  RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("n", g.n_embd),
                             RAD_INT("group", DF_W2_GROUP), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS),
              {w.sel_h}, {w.dq, w.ds, w.dsum});
    /* THE HEAD'S GEMM.
     *
     * bf16 mode: `gemm_nt` over `sel_h` itself -- the target's lm_head rows at the dtype the
     * checkpoint stores, no codes and no plane, one launch. `dlog` is bf16 in both modes; that is
     * what SGLang's unquantised candidate path produces too (`torch.matmul(hidden.to(weight.dtype),
     * weight[:num_org].T)`) and it is what `row_topk` reads. Only the SELECTED values leave the
     * head -- they are widened to f32 into `unary`, and the selector adds its f32 edge score to
     * them -- so the logits that reach a decision are f32 and the bf16 plane is the same narrowing
     * the reference's own bf16 matmul makes.
     *
     * fp8 mode: the 2-bit GEMM over the codebook pair, unchanged.
     *
     * THE VOCABULARY IS `n_vocab_all` ROWS AND THE HEAD IS THAT WIDE. SGLang's unquantised path
     * masks a padded tail (`logits[:, num_org:] = -inf`) because its lm_head is wider than the
     * tokenizer's vocabulary; Radiance has no such concept -- `rad_arch.h`'s `n_vocab_all` is the
     * active vocabulary and the container's head is exactly that many rows -- so there is no tail
     * to mask and no `-inf` pass. A container whose head IS padded has to declare the padded width
     * and the padded rows become candidate ids the codebooks also hold; that is the same behaviour
     * the fp8 head has and it is a property of the container, not of this block. */
    op_dlogits = cfg.bf16
        ? rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_dh)),
               {w.sel_h}, {w.dlog})
        : rw(b, RAD_OP(b, "gemm_nt_q",
                   RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                              RAD_INT("K", g.n_embd), RAD_INT("group", DF_W2_GROUP),
                              RAD_STR("dtype", "w2a8")),
                   RAD_WEIGHTS(w_dh, w_dhs)),
               {w.dq, w.ds, w.dsum}, {w.dlog});
    /* Sharded, the top-R runs over THIS RANK'S plane and reports global ids, and it also writes
     * the pair plane the gather moves. `vocab_off` is omitted entirely when the head is
     * replicated, so that graph is the one it always was -- an optional parameter present with
     * the value 0 is not the same declaration as an absent one. */
    op_dtopk = dh_shard
        ? rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.sel_topk), RAD_INT("vocab_off", g.vocab_off),
                           RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary, w.dpair})
        : rw(b, RAD_OP(b, "row_topk",
                RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", head_rows(g)),
                           RAD_INT("R", cfg.sel_topk), RAD_STR("dtype", g.dtype)),
                RAD_NOWEIGHTS),
             {w.dlog}, {w.cand, w.unary});
    if (dh_shard) {
        /* ROW-INTERLEAVED, so one draft position's pairs from every rank land side by side --
         * `row_topk_merge` reads one row at one row stride, the same reason the sampler's own
         * candidate gather carries `row`. */
        op_dgather = rw(b, RAD_OP(b, "all_gather",
                           RAD_PARAMS(RAD_INT("world_size", g.world),
                                      RAD_INT("numel", srows * cfg.sel_topk * 2),
                                      RAD_STR("dtype", "f32"),
                                      RAD_INT("row", cfg.sel_topk * 2)),
                           RAD_NOWEIGHTS),
                       {w.dpair}, {w.dgath});
        op_dmerge = rw(b, RAD_OP(b, "row_topk_merge",
                          RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("R", cfg.sel_topk),
                                     RAD_INT("world_size", g.world),
                                     RAD_STR("dtype", g.dtype)),
                          RAD_NOWEIGHTS),
                      {w.dgath}, {w.cand, w.unary});
    }
    op_hp = rw(b, RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, srows), RAD_INT("N", cfg.sel_rank),
                             RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w_selp)),
              {w.sel_h}, {w.hp});
    /* `anchor_stride` is `block`, because the operand is the step's own token_ids: a draft pass
     * carries `block` ids a sequence and the anchor is the first of them. */
    if (cfg.tree && w.packed) {
        /* THE SAME SIX OPERANDS, THE SAME PARAMETERS, PLUS THE TREE'S THREE. `hp` is operand index
         * 2 in both schemas and stays optional; the output is the packed tree instead of
         * `tokens`, because a tree is a topology and a `[max_seqs, steps]` token plane cannot
         * carry one. `tree_budget` is the verification row budget minus the root, and it is
         * bounded BELOW by one node: a call that could build nothing would propose nothing, and
         * the caller's linear path is what such a deployment should use instead. */
        const int64_t budget = cfg.tree_nodes > 1 ? cfg.tree_nodes - 1
                                                  : cfg.steps();
        op_tsel = rw(b, RAD_OP(b, "dflash_tree_select",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                                   RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                                   RAD_INT("n_vocab", g.n_vocab_all),
                                   RAD_INT("anchor_stride", cfg.block),
                                   RAD_INT("tree_budget", budget > 0 ? budget : 1),
                                   RAD_INT("tree_width", cfg.tree_width()),
                                   RAD_F64("depth_penalty", 0.0), RAD_STR("dtype", g.dtype)),
                        RAD_WEIGHTS(w_pred, w_succ)),
                    {w.cand, w.unary, w.hp}, {w.packed});
        /* A MISSING TREE SELECTOR IS A REFUSAL, NOT A FALLBACK. `rad_decl_op` fails the whole
         * declare for a name no loaded plugin publishes, so reaching here with 0 means the op was
         * rejected for its PARAMETERS -- a width or budget the kernel's band does not take. Either
         * way there is nothing to issue, and a deployment that returned OK would run a drafter
         * whose every pass silently proposes nothing: the tokens stay correct and speculation
         * simply stops, which reads as a policy decision rather than as a wiring fault. */
        if (!op_tsel) return RAD_E_UNSUPPORTED;
    } else {
        op_sel = rw(b, RAD_OP(b, "dflash_select",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs), RAD_INT("steps", cfg.steps()),
                                  RAD_INT("top_k", cfg.sel_topk), RAD_INT("rank", cfg.sel_rank),
                                  RAD_INT("n_vocab", g.n_vocab_all),
                                  RAD_INT("anchor_stride", cfg.block), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_pred, w_succ)),
                   {w.cand, w.unary, w.hp}, {w.tokens});
    }

    return RAD_OK;
}

/* ------------------------------------------------------------------------------ the context pass */

inline void Dflash2::context(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;

    /* THE ACCEPTED ROWS FIRST, WHEN THE PASS NAMES THEM.
     *
     * After a TREE verify the accepted path's rows in the tap plane are `start + node[j]`, which is
     * generally not the contiguous run this pass would read. `out_ids` carries exactly that list --
     * one entry a row of THIS pass -- so the gather turns it into a dense plane of `T` rows and
     * everything below reads it. The rows the gather does not name are the siblings the target
     * rejected, and they must not reach a cache that outlives the step.
     *
     * A prefill chunk names its own identity range, so the gather is a copy and the arithmetic below
     * is bit-identical to reading `src` directly; it runs anyway, because `fc`'s operand is fixed at
     * declare and one path is one behaviour to reason about. */
    if (op_cgather && w.srcg) {
        const int64_t nidx = batch->n_out;
        if (nidx < T) {
            rad_step_fail(c, "DFlash2 context pass has fewer source row indices than context rows");
            return;
        }
        /* THE SOURCE IS THE WHOLE TAP PLANE, NOT THIS PASS'S ROWS. `idx` holds ABSOLUTE rows of
         * `src` -- `start + node[j]` -- so the operand's extent is what every index is validated
         * against. Narrowing it to `T` makes the gather read `src` as if the pass's first `T` rows
         * were the whole buffer: an accepted path of nodes {0, 2, 5} at T = 3 would ask for row 5
         * and be refused (or, on a kernel that does not bounds-check, read the next sequence's
         * taps). The destination is the narrowed one -- `T` rows is exactly what the gather emits.
         *
         * The declared height of `src` is `max_tok`, and the rows past this step were never
         * written; an index there is a producer bug the reference rejects, which is the answer we
         * want rather than a zero row that looks like a real hidden state. */
        RAD_ISSUE_N(c, op_cgather, T, RAD_B(w.src), praw(batch->out_ids, RAD_I32, T),
                    brows(w.srcg, T));
    }


    /* NOT SLICED. A context pass is one row a committed token, so it is as wide as the prefill
     * chunk; the bf16 GEMM's own band covers every M to 65536 and the fp8a8 pair covers every M,
     * so each op here is one launch and not one per 64 rows. */
    const rad_buf csrc = (op_cgather && w.srcg) ? w.srcg : w.src;
    if (op_qsrc) RAD_ISSUE_N(c, op_qsrc, T, brows(csrc, T), brows(w.srcq, T), brows(w.srcs, T));
    dflash2_dense_step(c, cfg.bf16, op_cfc, T, brows(csrc, T), brows(w.srcq, T),
                       brows(w.srcs, T), brows(w.c, T), w_fc.w, w_fc.s);
    RAD_ISSUE_N(c, op_cnorm, T, brows(w.c, T), RAD_W(w_hnorm), brows(w.c, T));
    if (op_qc) RAD_ISSUE_N(c, op_qc, T, brows(w.c, T), brows(w.cq, T), brows(w.cs, T));

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;
        dflash2_dense_step(c, cfg.bf16, op_ck[i], T, brows(w.c, T), brows(w.cq, T),
                           brows(w.cs, T), brows(w.ck, T), w_k[i].w, w_k[i].s);
        dflash2_dense_step(c, cfg.bf16, op_cv[i], T, brows(w.c, T), brows(w.cq, T),
                           brows(w.cs, T), brows(w.cv, T), w_v[i].w, w_v[i].s);
        /* THE FUSED CHAIN WHEN IT RESOLVED, THE THREE OPS OTHERWISE. Operands are the chain's own,
         * in its order: k (in place), v, the K-norm gain, positions, slots, the cache. */
        if (op_ckstore[i]) {
            RAD_ISSUE_N(c, op_ckstore[i], T,
                        brows(w.ck, T), brows(w.cv, T), RAD_W(w_kn[i]),
                        praw(batch->positions, RAD_I32, T), praw(slot, RAD_I32, T),
                        kv_cache(kv, first_layer + i));
        } else {
            RAD_ISSUE_N(c, op_cknorm[i], T * g.n_head_kv,
                        brows(w.ck, T), RAD_W(w_kn[i]), brows(w.ck, T));
            RAD_ISSUE_N(c, op_crope, T, brows(w.ck, T), praw(batch->positions, RAD_I32, T));
            RAD_ISSUE_N(c, op_cstore, T, brows(w.ck, T), brows(w.cv, T),
                        praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        }
    }
}

/* ------------------------------------------------------------------------------ the query pass */

inline void Dflash2::step(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (T <= 0) return;
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;


    const int64_t NG = cfg.ng(g.n_embd);
    const int64_t half = cfg.taps * NG;      /* prepare's coefficients, then finish's */

    RAD_ISSUE_N(c, op_embed, T, praw(batch->token_ids, RAD_I32, T), RAD_W(w_tok), brows(w.x, T));
    if (op_embed_ar)
        RAD_ISSUE_N(c, op_embed_ar, (int64_t)T * g.n_embd, brows(w.x, T), RAD_NONE);

    for (int64_t l = 0; l < g.n_layer; ++l) {
        const int i = (int)l;

        /* ---- attention, wrapped in the two halves of one convolution */
        RAD_ISSUE_N(c, op_anorm[i], T, brows(w.x, T), RAD_W(w_anorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_acoef[i], T, brows(w.h, T), RAD_W(w_acp[i]), brows(w.coef, T));
        RAD_ISSUE_N(c, op_aconv[i], T, brows(w.h, T), bcol(w.coef, 0, half, T),
                    wrow_slice(w_acb[i], 0, 1, cfg.taps * g.n_embd), brows(w.h2, T));

        if (op_qh2) RAD_ISSUE_N(c, op_qh2, T, brows(w.h2, T), brows(w.h2q, T), brows(w.h2s, T));
        dflash2_dense_step(c, cfg.bf16, op_qp[i], T, brows(w.h2, T), brows(w.h2q, T),
                           brows(w.h2s, T), brows(w.q, T), w_q[i].w, w_q[i].s);
        dflash2_dense_step(c, cfg.bf16, op_kp[i], T, brows(w.h2, T), brows(w.h2q, T),
                           brows(w.h2s, T), brows(w.k, T), w_k[i].w, w_k[i].s);
        dflash2_dense_step(c, cfg.bf16, op_vp[i], T, brows(w.h2, T), brows(w.h2q, T),
                           brows(w.h2s, T), brows(w.v, T), w_v[i].w, w_v[i].s);
        RAD_ISSUE_N(c, op_qn[i], T * g.n_head, brows(w.q, T), RAD_W(w_qn[i]), brows(w.q, T));
        RAD_ISSUE_N(c, op_qrope, T, brows(w.q, T), praw(batch->positions, RAD_I32, T));
        /* Q's rotation stays its own op: the fused kernel is the K chain only. K's norm, rotation
         * and store are one launch when it resolved, three when it did not. */
        if (op_kstore[i]) {
            RAD_ISSUE_N(c, op_kstore[i], T,
                        brows(w.k, T), brows(w.v, T), RAD_W(w_kn[i]),
                        praw(batch->positions, RAD_I32, T), praw(slot, RAD_I32, T),
                        kv_cache(kv, first_layer + i));
        } else {
            RAD_ISSUE_N(c, op_kn[i], T * g.n_head_kv,
                        brows(w.k, T), RAD_W(w_kn[i]), brows(w.k, T));
            RAD_ISSUE_N(c, op_krope, T, brows(w.k, T), praw(batch->positions, RAD_I32, T));
            RAD_ISSUE_N(c, op_store, T, brows(w.k, T), brows(w.v, T),
                        praw(slot, RAD_I32, T), kv_cache(kv, first_layer + i));
        }
        /* ALL NINE POSITIONAL SLOTS, ABSENT ONES INCLUDED. `k_descale`/`v_descale` are absent
         * because the drafter's cache is bf16, and `tree_ancestors` is absent because this block
         * is NON-CAUSAL over a linear draft block: libr4d's decode kernels read the ancestor mask
         * only when `causal` is set, and its query-tiled prefill kernel never reads it -- the
         * block's own visibility is the two-sided window, not a tree. Absent is the contract's
         * RAD_NONE at the slot; a missing slot is a miscount and the engine refuses the op. */
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, first_layer + i),
                    praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn, T),
                    RAD_NONE);
        if (op_qat) RAD_ISSUE_N(c, op_qat, T, brows(w.attn, T), brows(w.atq, T), brows(w.ats, T));
        dflash2_dense_step(c, cfg.bf16, op_op[i], T, brows(w.attn, T), brows(w.atq, T),
                           brows(w.ats, T), brows(w.h, T), w_o[i].w, w_o[i].s);
        /* BEFORE the closing convolution, not after. The convolution is linear in its input and
         * its coefficients come from the replicated pre-attention norm, so either order is the
         * same arithmetic -- and reducing first keeps the residual stream's contract ("what the
         * ranks add to x has already been agreed") in one place. */
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        /* THE FUSED FORM WHEN THE ROW RESOLVED, THE PAIR OTHERWISE. Same operands in the same
         * order -- x, delta, base, residual, out -- and `out` is the residual stream itself, which
         * is the in-place add the pair performs. The all-reduce above is untouched and still
         * precedes both: the fusion neither reads nor writes a collective, so the residual
         * contract ("what the ranks add to x has already been agreed") stays where it was. */
        if (op_aconvadd[i]) {
            RAD_ISSUE_N(c, op_aconvadd[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                        wrow_slice(w_acb[i], 1, 1, cfg.taps * g.n_embd),
                        brows(w.x, T), brows(w.x, T));
        } else {
            RAD_ISSUE_N(c, op_aconv[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                        wrow_slice(w_acb[i], 1, 1, cfg.taps * g.n_embd), brows(w.h2, T));
            RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h2, T), brows(w.x, T));
        }

        /* ---- the feed-forward, wrapped in the other convolution */
        RAD_ISSUE_N(c, op_fnorm[i], T, brows(w.x, T), RAD_W(w_fnorm[i]), brows(w.h, T));
        RAD_ISSUE_N(c, op_fcoef[i], T, brows(w.h, T), RAD_W(w_fcp[i]), brows(w.coef, T));
        RAD_ISSUE_N(c, op_fconv[i], T, brows(w.h, T), bcol(w.coef, 0, half, T),
                    wrow_slice(w_fcb[i], 0, 1, cfg.taps * g.n_embd), brows(w.h2, T));
        if (op_qh2) RAD_ISSUE_N(c, op_qh2, T, brows(w.h2, T), brows(w.h2q, T), brows(w.h2s, T));
        if (op_gug[i] && T <= gated_rows) {
            RAD_ISSUE_N(c, op_gug[i], T, brows(w.h2q, T), brows(w.h2s, T),
                        RAD_W(w_gu[i].w), RAD_W(w_gu[i].s), brows(w.ffq, T), brows(w.ffs, T),
                        RAD_NONE);
        } else {
            dflash2_dense_step(c, cfg.bf16, op_gu[i], T, brows(w.h2, T), brows(w.h2q, T),
                               brows(w.h2s, T), brows(w.gate_up, T), w_gu[i].w, w_gu[i].s);
            RAD_ISSUE_N(c, op_act, T, brows(w.gate_up, T), brows(w.ffn, T));
            if (op_qff) RAD_ISSUE_N(c, op_qff, T, brows(w.ffn, T), brows(w.ffq, T),
                                    brows(w.ffs, T));
        }
        dflash2_dense_step(c, cfg.bf16, op_dn[i], T, brows(w.ffn, T), brows(w.ffq, T),
                           brows(w.ffs, T), brows(w.h, T), w_dn[i].w, w_dn[i].s);
        if (op_ar) RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h, T), RAD_NONE);
        /* The same choice as the attention half: the fused conv+add where the row resolved, the
         * pair where it did not. The all-reduce stays in front of both. */
        if (op_fconvadd[i]) {
            RAD_ISSUE_N(c, op_fconvadd[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                        wrow_slice(w_fcb[i], 1, 1, cfg.taps * g.n_embd),
                        brows(w.x, T), brows(w.x, T));
        } else {
            RAD_ISSUE_N(c, op_fconv[i], T, brows(w.h, T), bcol(w.coef, half, half, T),
                        wrow_slice(w_fcb[i], 1, 1, cfg.taps * g.n_embd), brows(w.h2, T));
            RAD_ISSUE_N(c, op_add, T, brows(w.x, T), brows(w.h2, T), brows(w.x, T));
        }
    }

    RAD_ISSUE_N(c, op_final, T, brows(w.x, T), RAD_W(w_norm), brows(w.h, T));

    /* The rows that predict something: `steps` of every sequence's `block`. The scheduler names
     * them through out_ids, the same field the trunk's own logits gather reads. */
    const int64_t S = batch->n_out;
    if (S <= 0) return;
    RAD_ISSUE_N(c, op_gather, S, RAD_B(w.h), praw(batch->out_ids, RAD_I32, S), brows(w.sel_h, S));

    /* `dsum` is quant_act_i8's GROUP-major plane, [n_embd/group][ceil(M/16)*16], so it is passed
     * whole: there is no row slice of it that means anything. Only the fp8 head has one. */
    if (op_dquant)
        RAD_ISSUE_N(c, op_dquant, S, brows(w.sel_h, S), brows(w.dq, S), brows(w.ds, S),
                    RAD_B(w.dsum));
    /* THE HEAD'S GEMM, one issue site for both modes. bf16 reads `sel_h` and the weight; fp8 reads
     * the int8 pair and carries the two absent optional slots `gemm_nt_q`'s schema has after `y`
     * (a_sum, b_ref), which are POSITIONAL and must be passed as RAD_NONE rather than left off. */
    if (cfg.bf16) {
        RAD_ISSUE_N(c, op_dlogits, S, brows(w.sel_h, S), RAD_W(w_dh), brows(w.dlog, S));
    } else {
        RAD_ISSUE_N(c, op_dlogits, S, brows(w.dq, S), brows(w.ds, S), RAD_W(w_dh), RAD_W(w_dhs),
                    brows(w.dlog, S), RAD_B(w.dsum), RAD_NONE);
    }
    /* RAD_NONE is the `pairs` slot: operands are POSITIONAL, so an optional one that the graph
     * does not use is still passed, absent. Omitting it is a miscount and the engine refuses the
     * op rather than guessing which of the four was left out.
     *
     * SHARDED, the top-R over this rank's plane is not the answer -- it is this rank's CANDIDATES,
     * and `cand`/`unary` are overwritten by the merge two ops below. Writing them here anyway
     * costs nothing (they are the same R entries the pairs plane holds) and keeps one issue site
     * for both paths. */
    if (op_dmerge) {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    brows(w.dpair, S));
        RAD_ISSUE(c, op_dgather, brows(w.dpair, S), brows(w.dgath, S));
        RAD_ISSUE_N(c, op_dmerge, S, brows(w.dgath, S), brows(w.cand, S), brows(w.unary, S));
    } else {
        RAD_ISSUE_N(c, op_dtopk, S, brows(w.dlog, S), brows(w.cand, S), brows(w.unary, S),
                    RAD_NONE);
    }

    RAD_ISSUE_N(c, op_hp, S, brows(w.sel_h, S), RAD_W(w_selp), brows(w.hp, S));
    /* ONE ISSUE SITE, TWO SELECTORS. `M` is the sequences in this pass and the candidate planes
     * are indexed `s*steps + l`, which is the layout the pass just wrote; the anchor is read out
     * of the same `token_ids` plane at `anchor_stride`, exactly as the linear walk reads it. */
    if (op_tsel)
        RAD_ISSUE_N(c, op_tsel, batch->n_seq, brows(w.cand, S), brows(w.unary, S),
                    brows(w.hp, S), praw(batch->token_ids, RAD_I32, T),
                    RAD_W(w_pred), RAD_W(w_succ), brows(w.packed, batch->n_seq));
    else
        RAD_ISSUE_N(c, op_sel, batch->n_seq, brows(w.cand, S), brows(w.unary, S), brows(w.hp, S),
                    praw(batch->token_ids, RAD_I32, T), RAD_W(w_pred), RAD_W(w_succ),
                    brows(w.tokens, batch->n_seq));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_DFLASH2_H */
