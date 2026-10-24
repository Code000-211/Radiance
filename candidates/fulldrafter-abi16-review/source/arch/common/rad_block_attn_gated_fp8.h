/* rad_block_attn_gated_fp8.h -- the gated GQA block with block-scaled FP8 projections.
 *
 * THE ARCHITECTURE IS rad_block_attn_gated.h's AND IS DOCUMENTED THERE: the interleaved query/gate
 * projection and why it is declared rank 3, the partial mrope and what is not expressible about it,
 * why k and v are not fused, why the fused qk_norm_rope_gate probe is withdrawn, and why the gate
 * is a sigmoid. None of that changes with the weight format and none of it is repeated here.
 *
 * What differs:
 *
 *   - q|gate, k, v and o are LinearFP8 -- an E4M3 weight with a bf16 [N/128][K/128] scale plane
 *     declared as a weight of its own, through `gemm_nt_q` at dtype fp8a8.
 *   - two `quant_act_fp8` launches a layer, not four: the three input projections all read the
 *     post-norm hidden state, so one quantise feeds all three, and the output projection gets its
 *     own because it reads the gated attention output.
 *   - everything between the GEMMs -- the per-head norms, rope, kv_store, attn_paged, the sigmoid
 *     and the gate multiply -- is UNCHANGED and still bf16.
 *   - THE KV CACHE'S WIDTH IS `g.kv_dtype` AND NOT `g.dtype`. `dtype` follows the WEIGHTS;
 *     "the cache is bf16 because the weights are fp8" is true of this checkpoint and is not a
 *     rule. `kv_dtype` follows the CACHE, which is a deployment choice: the fp8kv attention family
 *     and its descale operands are in libr4d, and the cache dtype is what selects them.
 *     Both the attention op and kv_store take it, so the two cannot disagree about the width of
 *     the thing one writes and the other reads.
 *
 * The q|gate projection is [2*n_head*head_dim, n_embd] = [12288, 5120] at 27B, and its scale plane
 * is [96, 40]. Row-sharding by head keeps whole 128-row scale blocks together only because
 * 2*head_dim is 512, a multiple of 128 -- so a rank's slice of the head range is a whole number of
 * scale rows. LinearFP8 checks that rather than trusting it.
 */
#ifndef RAD_BLOCK_ATTN_GATED_FP8_H
#define RAD_BLOCK_ATTN_GATED_FP8_H

#include "rad_fp8.h"

#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

struct AttnGatedFP8 {
    struct Config {
        bool tree_state = false;
    };
    Config cfg{};
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        /* [rope_table_rows, rot] f32, the model's, shared by every layer and filled once a step.
         * Zero means the plugin has not been taught to build one, and then the fused prologue is
         * not asked for at all. */
        rad_buf cos_sin = 0;
        ActFP8  attn{};     /* [max_tok, q_dim]   attention output, then gated in place */
        /* [max_tok, q_dim] the de-interleaved sigmoid gate, in a model whose linears are bf16
         * (Geom::w_bf16) only: there the gate is `sigmoid` then `mul`, rad_block_attn_gated.h's
         * pair, and the fp8 path's fused gate-and-quantiser has no form that skips its codes. */
        rad_buf gate = 0;
    };

    struct Src {
        const char* norm   = nullptr;
        const char* qg     = nullptr;   /* self_attn.q_proj.weight -- query AND gate */
        const char* k      = nullptr;
        const char* v      = nullptr;
        const char* q_norm = nullptr;
        const char* k_norm = nullptr;
        const char* o      = nullptr;
    };

    Geom        g{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    /* THE KV PAGE, IN TOKENS. 0 means RAD_KV_BLOCK, which is 16 and is what every geometry in
     * this tree has wanted. It is a field rather than that constant because Qwen4-Exp's QSA
     * selects 4-TOKEN BLOCKS, and a page of 4 makes the selected set a paged block table instead
     * of a new attention kernel -- see docs/QSA.md. The attention kernel's own rows must carry a
     * matching C_EQ("block_size", ...) or nothing resolves, which is the check that keeps this
     * from being a number anyone can set. */
    int64_t     kv_block = 0;
    /* THE QSA SELECTION, when this layer has an indexer. Set, and on a step where one query a
     * sequence makes the selection per-SEQUENCE, the attention is issued with these in place of
     * the sequence's own block table and context length -- which is the whole of what makes it
     * sparse. Both are ordinary operands (the block table has always been one), so this is the
     * SAME op and the SAME kernel reading a different list.
     *
     * `sel` is `topk + 1` wide and the kernel takes its block-table pitch from the operand, so
     * nothing else has to be told. See docs/QSA.md; arch/common/rad_qsa.h produces them. */
    rad_buf     qsa_sel  = 0;
    rad_buf     qsa_sequ = 0;
    /* THE CONTEXT UP TO WHICH THE DENSE PATH IS THE SAME ANSWER, `budget + ratio - 1`. Below it
     * every complete block is selected and the two paths agree to the bit, and the dense one is
     * cheaper at prefill -- it reads each key once a query TILE where the sparse one reads a
     * table a query. Above it the dense path is a different model. Zero means "always sparse",
     * which is what a caller that has not been told the bound should get. */
    int64_t     qsa_exact_to = 0;
    /* `sel`'s row pitch, `topk + 1`. The extra slot is the partial page and is not spare --
     * r4d_qsa_select_bf16.hip says why. */
    int64_t     qsa_topk = 0;
    /* Model-owned MTP frontier overlay; zero handles leave every ordinary path unchanged. */
    rad_buf frontier_kv = 0, frontier_slots = 0, frontier_prefix = 0;
    rad_buf frontier_ancestors = 0, frontier_meta = 0;
    int64_t frontier_rows = 0;
    rad_op op_frontier_attn[4]{};
    /* HOW MANY QUERY ROWS ONE SPARSE ATTENTION LAUNCH CARRIES, and it is a scratch budget rather
     * than a preference. The decode kernel's split-KV partials are sized AT DECLARE from the
     * `max_seqs` and `q_len` parameters, so the only way to issue more rows than `max_seqs` is to
     * declare a larger one -- which grows the scratch arena by `rows * q_len_band * n_head *
     * (head_dim + 2) * 4`, about 32 MiB at 256 rows on this model, ONCE for the whole program
     * because the arena is the maximum over ops.
     *
     * It matters because the alternative is `max_seqs` itself. At `--max-num-seqs 4` a 2048-token
     * prefill chunk would be 512 launches a layer, and the dispatch cost of that dominates what
     * the sparse path saves. 256 makes it 8.
     *
     * Raising it further is pure scratch: 512 rows is 63 MiB, 2048 is 253. 256 is the point at
     * which the launch cost has stopped mattering. */
    static constexpr int64_t kQsaRows = 256;
    int64_t qsa_rows() const { return g.max_seqs > kQsaRows ? g.max_seqs : kQsaRows; }
    /* THE GATED FORM TAKES THE WHOLE STEP IN ONE LAUNCH, because its band is q_len 1 and so its
     * partials are only `rows * n_head * (2 * head_dim + 8)` bytes -- 12.8 MiB at a 2048-row step
     * on this model, where the ungated op's q_len band makes the same rows cost the figures
     * above. Fewer launches still pay here: one 2048-row launch takes about 2.16 ms a layer
     * against about 3.0 ms for eight 256-row ones, each of which is 256 one-wave workgroups -- two
     * waves a SIMD -- with its own tail. */
    int64_t qsa_gq_rows() const { return g.max_tok > qsa_rows() ? g.max_tok : qsa_rows(); }
    /* THE ROTATION'S PAIRING, a question the checkpoint cannot answer by inspection and that no
     * per-op check can reach, because both readings -- halves (NeoX, what HF's rotate_half does
     * over the partial rotary width) and adjacent pairs (GPT-J) -- are self-consistent and libref
     * shares whichever one the graph picks. Either answer is a DIFFERENT MODEL. "mrope" is what the
     * model does, and over one position component a token it is NeoX (rad_block_attn_gated.h).
     *
     * THE q|gate ORDER IS FIXED TOO. q is the FIRST half of each head's pair and the gate the
     * second, which is how vLLM chunks q_gate, and every consumer of the projection has to agree
     * on it: the unfused q norm reads column 0 of each head, the gate quantiser column head_dim,
     * and the fused prologue splits the pair the same way inside its kernel. A swap would be
     * arithmetically silent -- q_norm renormalises either half and the sigmoid accepts either --
     * so an order that only some of them followed would be fluent and wrong. */
    const char* mode  = "mrope";

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    LinearFP8  qg{}, kp{}, vp{}, o{};
    NormQuantFP8 nq_h{};

    rad_op     op_fused = 0;                                 /* qk_norm_rope_gate, if it resolves */
    rad_op     op_fused_kv = 0;                              /* ...with kv_store folded on, likewise */
    int64_t    fused_rows = 0;                               /* ...and the widest T it is issued at */
    rad_op     op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op     op_attn = 0, op_kv_store = 0, op_gq = 0;
    rad_op     op_attn_gq = 0;                               /* attn_paged + gate_quant_fp8 */
    rad_op     op_attn_tree = 0;                             /* wide target trees, I32 parents */
    rad_op     op_sig = 0, op_mul = 0;                       /* the bf16 gate, Geom::w_bf16 */
    rad_op     op_ar = 0, op_add = 0;
    /* ...AND WHETHER THE COLLECTIVE AT THE END OF THIS BLOCK IS THE NEXT BLOCK'S FIRST PASS.
     * `ar_in` says this block's own norm absorbs its PREDECESSOR's all-reduce; `ar_out` says the
     * block after it absorbs THIS one's, so this one must not issue it. They are the residual
     * fold's two flags again, one boundary later, and they are separate for the same reason: a
     * block is declared without knowing what follows it. On EITHER wire -- the fused kernel
     * carries both payloads -- and both sides ask ar_fused_ok(g, T), so the two cannot disagree
     * about whether the collective has already been issued. */
    bool ar_out = false;
    /* ...and WHICH WIRES that consumer takes (ArOut, rad_fp8.h): the norm's by default, and a gated
     * residual's write says its own (HyperConn::ar_take). Set before declare; both sides then ask
     * ar_taken(g, T, ar_out, ar_out_take). */
    int  ar_out_take = kArOutNorm;

    /* `fold` says the PREVIOUS block left its delta in `w.h.x` for this norm to add; `add_out`
     * says the NEXT one will not, so this block issues the trailing add itself. See rad_fp8.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnGatedFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                 rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                 const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    /* DLoop's scheduler admits branches only on the batch-wide dense-exact
     * QSA route and commits the indexer's accepted raw-key path separately.
     * Other drafters retain the existing veto. */
    if (qsa_sel && !cfg.tree_state) {
        const int rc = rad_disable_branched_tree(b);
        if (rc != RAD_OK) return rc;
    }
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();
    const int64_t qg_dim = 2 * q_dim;

    /* THE KV ROW SPAN, AND IT ONLY EXISTS ABOVE THE KV HEAD COUNT.
     * A world size larger than `n_head_kv_all` cannot divide the KV heads, so rad_arch.h's
     * `geom_from` replicates them in PAIRS: two ranks serve the same head, `g.n_head_kv` is 1 on
     * each, and the checkpoint entry they both read is still `n_head_kv_all * head_dim` rows -- 512
     * for 2 KV heads at head_dim 256. This rank wants ONE head's row-span of that entry, which the
     * even shard would cut up instead (at 4 ranks it would hand rank 2 rows [256, 384), half of one
     * head), so the span `[g.kv_off, g.kv_off + kv_dim)` is declared in the CONTAINER's unsharded
     * units -- [0, 256) on ranks 0 and 1, [256, 512) on ranks 2 and 3 -- and the scale plane takes
     * the same span in 128-blocks. `kv_off` is a multiple of head_dim, so it is whole blocks and
     * never cuts one, and share.cpp's `shard_hi <= cpart` holds with cpart = 512 / 1 part.
     * AT WORLD <= n_head_kv_all the ordinary even shard already selects exactly these rows, so the
     * span stays 0/0 and the default path runs as before -- TP2 declares identically. */
    const bool    kv_spanned = g.world > g.n_head_kv_all;
    const int64_t kv_span_lo = kv_spanned ? g.kv_off : 0;
    const int64_t kv_span_hi = kv_spanned ? g.kv_off + kv_dim : 0;

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* THE INPUT MAY BE PREPARED BY THE ARCHITECTURE, and `src.norm == nullptr` is how it says so.
     *
     * A pre-norm model's block owns its own norm: one op that adds the previous block's delta into
     * the residual stream, normalises it and quantises it (rad_fp8.h's NormQuantFP8). A GATED
     * RESIDUAL model's does not -- arch/common/rad_block_hc.h's `hc_read` produces the block input
     * from a stream four times as wide, with a data-dependent mix the block has no way to compute,
     * and there is nothing left for a norm here to do. So this block takes `wire.h` as ALREADY
     * normed and quantised and declares no norm weight at all.
     *
     * `ar_in` is refused in that mode rather than ignored: the fused all-reduce lives INSIDE the
     * norm, so with no norm there is nowhere to fold it and a caller that asked would silently get
     * an unreduced input. It emits a standalone collective instead (`add_out`/`ar_out`). */
    ext_in = (src.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
        if (g.w_bf16 && !w.gate) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
        w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_q_norm.weight", l), src.q_norm));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_k_norm.weight", l), src.k_norm));
    w_q_norm = decl_w(b, nm.f("blk.%d.attn_q_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k_norm = decl_w(b, nm.f("blk.%d.attn_k_norm.weight", l), RAD_F32, {g.head_dim},
                      RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* One norm, one quantise, three consumers -- and one op, see rad_fp8.h's NormQuantFP8. */
    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));
    RAD_ARCH_TRY(qg.declare(b, nm, g, nm.f("blk.%d.attn_qg", l), qg_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.qg }, w.h, w.qg));
    RAD_ARCH_TRY(kp.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.k }, w.h, w.k,
                            0, {}, kv_span_lo, kv_span_hi));
    RAD_ARCH_TRY(vp.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.v }, w.h, w.v,
                            0, {}, kv_span_lo, kv_span_hi));

    /* THE FUSED PROLOGUE, AND IT IS A PROBE: the four ops below are declared either way and are
     * what runs when nothing serves this one. rad_block_attn_gated.h does not make the probe,
     * because the schema takes a PRECOMPUTED cos/sin table rather than `theta` and that block has
     * none; `rope_table` builds one here, once a step for every layer, and `w.cos_sin` is where
     * the model puts it. A block handed no table cannot ask for the fusion, which is what keeps a
     * plugin that does not build the table on the unfused path.
     *
     * NEOX ONLY. The fused kernel rotates NeoX pairs and has no `mode`, so the probe is made only
     * where mrope reduces to neox -- one position component a token, which is every text batch and
     * is argued in rad_block_attn_gated.h. A vision span would need the sections first.
     *
     * NO `gate_out`. `gate_quant_fp8` below already reads the gate at its column offset inside the
     * interleaved projection, so a contiguous copy would be q_dim of bf16 a layer written for
     * nothing. The operand is optional and this passes none.
     *
     * IT IS BIT-IDENTICAL TO THE FOUR, checked in r4d_selftest (`qk_norm_rope_gate ... matches the
     * unfused pair`) rather than argued: the kernel rounds its norm output to bf16 before rotating
     * because that is what a norm that stores a tensor and a rope that reads it back do. It
     * reproduces `r4d_rmsnorm_bf16`'s sum-of-squares fold element by element rather than folding
     * eight elements a lane, and r4d_selftest asserts byte equality. A merely RELATIVE bound is
     * not enough here: a fold order that agrees to a ULP and not to the last bit moves the
     * engine's transcript within a few dozen steps. */
    /* M-RoPE qualifies too: the table is filled at one component a token, which is exact on every
     * pass whose tokens' components agree, and the step takes the fused path only then. */
    const bool rope_neox = !std::strcmp(mode, "neox") || !std::strcmp(mode, "mrope") || g.rope_mc;
    fused_rows = qk_fuse_rows(g);
    if (w.cos_sin && rope_neox && fused_rows > 0)
        op_fused = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, fused_rows), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head", g.n_head),
                                    RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_q_norm, w_k_norm)),
                     {w.qg, w.k, w.cos_sin}, {w.q, w.k});

    op_q_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_q_norm)),
                   {w.qg}, {w.q});
    op_k_norm = rw(b, RAD_OP(b, "rmsnorm",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head_kv),
                                  RAD_INT("n", g.head_dim),
                                  RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk), RAD_STR("dtype", g.dtype)),
                       RAD_WEIGHTS(w_k_norm)),
                   {w.k}, {w.k});
    op_rope_q = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head, 0, n_rot, mode),
                   {w.q}, {w.q});
    op_rope_k = rw(b, decl_rope(b, g, g.max_tok, g.head_dim, g.n_head_kv, 0, n_rot, mode),
                   {w.k}, {w.k});

    /* Declared before kv_store because block_size is RAD_DERIVED and kv_store reads it back --
     * rad_block_attn_gated.h explains why the declared order differs from the issue order. */
    op_attn = RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                                /* The sparse path issues one row a QUERY, so the bound the split-KV
                                 * scratch is sized from is the widest such launch and not the
                                 * sequence count -- see kQsaRows. */
                                RAD_INT("max_seqs", (qsa_sel && qsa_sequ) ? qsa_rows()
                                                                          : g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS);
    /* THE QSA SELECTION IS A READ OF THIS OP AND HAS TO BE DECLARED AS ONE. The block table has
     * always been a raw batch pointer, so `attn_paged`'s read set was just the query -- and
     * `sel`/`seqused` are BUFFERS, whose lifetimes the Builder plans from exactly these lists. Left
     * out, their last use is `qsa_select` and the arena is free to lay another buffer over them;
     * anything written between the selection and the attention then arrives as a page id, and a
     * garbage page id is an address far outside the cache -- an aperture violation at the first
     * step whose plan happens to overlap them, which need not be a step that ran before. Spelled
     * with rad_op_reads rather than rw() because the list is conditional -- a layer with no
     * indexer has no such read.
     *
     * THE DECLARATION ORDER IS THE BUFFER PLAN, and this is one of the shapes that rule takes. */
    if (op_attn) {
        const rad_buf rd[3] = { w.q, qsa_sel, qsa_sequ };
        rad_op_reads(b, op_attn, rd, (qsa_sel && qsa_sequ) ? 3 : 1);
        rad_op_writes(b, op_attn, &w.attn.x, 1);
    }
    /* Small trees also need masked attention when the whole batch selects the
     * ordinary prefill row. Keep other geometries and the correct <=5-row decode
     * path unchanged; this private row serves the TP2 Qwen geometry only. */
    const bool small_tree_attn = g.max_spec > 0 && g.head_dim == 256 &&
        g.gqa() == 12 && g.n_head == 12 &&
        (kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK) == 4 &&
        g.dtype && !std::strcmp(g.dtype, "bf16") &&
        g.kv_dtype && !std::strcmp(g.kv_dtype, "fp8_e4m3");
    if (cfg.tree_state && (g.max_spec > 15 || small_tree_attn)) {
        const int64_t tree_capacity = g.max_spec > 15 ? 65 : 16;
        const int64_t tree_tokens = g.max_seqs > 0 && g.max_seqs <= g.max_tok / tree_capacity
                                  ? g.max_seqs * tree_capacity : g.max_tok;
        op_attn_tree = rw(b, RAD_OP(b, "attn_paged_tree",
            RAD_PARAMS(RAD_RANGE("q_len", 1, tree_capacity), RAD_INT("max_tokens", tree_tokens),
                       RAD_INT("head_dim", g.head_dim), RAD_INT("gqa", g.gqa()),
                       RAD_INT("causal", 1), RAD_INT("window", 0),
                       RAD_INT("n_head", g.n_head),
                       RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                       RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                       RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
            RAD_NOWEIGHTS), {w.q}, {w.attn.x});
        if (!op_attn_tree) return RAD_E_UNSUPPORTED;
    }
    if (frontier_kv) {
        if (!frontier_slots || !frontier_prefix || !frontier_ancestors || !frontier_meta ||
            frontier_rows < 8 || rad_kv_block_size(b, kv) != 4) return RAD_E_INVAL;
        for (int d = 1; d <= 4; ++d) {
            op_frontier_attn[d - 1] = rw(b, RAD_OP(b, "mtp_overlay_attn",
                RAD_PARAMS(RAD_INT("M", frontier_rows), RAD_INT("N", g.max_seqs),
                           RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                           RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("block_size", 4),
                           RAD_INT("depth", d), RAD_STR("q_dtype", g.dtype),
                           RAD_STR("kv_dtype", g.kv_dtype), RAD_INT("max_ctx", g.max_ctx)),
                RAD_NOWEIGHTS),
                {w.q, frontier_kv, frontier_prefix, frontier_ancestors, frontier_meta},
                {w.attn.x, frontier_meta});
            if (!op_frontier_attn[d - 1]) return RAD_E_INVAL;
        }
    }

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* THE PROLOGUE AND THE CACHE STORE AS ONE LAUNCH, wherever the fused prologue is taken: a k
     * head's block already holds its token's rotated k, one element a lane, so it writes that and
     * the head's v into the cache too, byte-for-byte what kv_store would (the fold's row says why).
     * Declared here and not beside op_fused because the cache's block size is read back from the
     * attention op above. It resolves for an E4M3 and a bf16 cache alike. */
    if (op_fused)
        op_fused_kv = rw(b, RAD_OP(b, "qk_norm_rope_gate_kv_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, fused_rows),
                                       RAD_INT("head_dim", g.head_dim), RAD_INT("n_head", g.n_head),
                                       RAD_INT("n_head_kv", g.n_head_kv), RAD_INT("rot", n_rot),
                                       RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                       RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_STR("kv_dtype", g.kv_dtype)),
                            RAD_WEIGHTS(w_q_norm, w_k_norm)),
                         {w.qg, w.k, w.cos_sin, w.v}, {w.q, w.k});
    if (frontier_kv) {
        const rad_op stores[] = {op_kv_store, op_fused_kv};
        for (rad_op op : stores) if (op) {
            if (rad_op_reads(b, op, &frontier_slots, 1) < 0 ||
                rad_op_writes(b, op, &frontier_kv, 1) < 0) return RAD_E_INVAL;
        }
    }

    /* THE OUTPUT GATE, THE MULTIPLY AND THE QUANTISER ARE ONE PASS. Unfused they are three --
     * sigmoid into a de-interleaved buffer, a multiply in place, then quant_act_fp8 -- and at
     * decode each is small enough that the dispatch gap between them is comparable to the work
     * itself, once per layer. THE ROW IS A (token, head) PAIR, which is what makes the fusion
     * expressible: at that shape the gate is a fixed stride inside the [q|gate] projection, the
     * attention output is contiguous, and the fp8 scale plane of [T * n_head, head_dim / 128] is
     * byte-for-byte the [T, q_dim / 128] the out projection reads. No separate gate buffer is
     * declared, because nothing reads one.
     *
     * The bf16 product is still written, because that is what the unfused pair left in `w.attn.x`
     * and the codes are taken over it -- see r4d_fused_quant_fp8.hip on why that is a requirement.
     * The codes are the int8 pair when the model feeds the out projection int8 (ActFP8::q8_fed),
     * and the same op writes them: the codes operand's dtype picks the quantiser.
     * PREFILL AND DECODE BOTH TAKE IT: unlike the gdn fold this one has no reduction across the
     * row, so there is no shape it cannot serve. */
    if (g.w_bf16) {
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {w.gate});
        op_mul = rw(b, RAD_OP(b, "mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.attn.x, w.gate}, {w.attn.x});
        if (!op_sig || !op_mul) return RAD_E_INVAL;
    } else
    op_gq = rw(b, RAD_OP(b, "gate_quant_fp8",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                              RAD_INT("n", g.head_dim), RAD_INT("group", RAD_FP8_BLOCK),
                              RAD_STR("dtype", g.dtype), RAD_STR("act", "sigmoid")),
                   RAD_NOWEIGHTS),
               {w.qg, w.attn.x}, {w.attn.cq(), w.attn.cs(), w.attn.x});

    /* AND ON THE SPARSE PATH THE GATE RIDES IN THE ATTENTION'S OWN MERGE. There every launch is
     * one query row a (token, head) pair at q_len 1, and the split-KV merge that closes each row
     * holds exactly the 256 values the gate and the quantiser read next -- so the fused op merges,
     * gates, rounds and quantises the row where it is, byte-for-byte the pair (its row says how),
     * and the launch between them is gone. Declared only beside a selection: the dense path's
     * query rows are the band the prefill kernel serves as well, and that kernel has no merge. */
    if (op_attn && qsa_sel && qsa_sequ && !g.w_bf16) {
        op_attn_gq = RAD_OP(b, "attn_paged_gate_quant",
                            RAD_PARAMS(RAD_INT("q_len", 1), RAD_INT("head_dim", g.head_dim),
                                       RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                       RAD_INT("window", 0), RAD_INT("n_head", g.n_head),
                                       RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                       RAD_INT("max_seqs", qsa_gq_rows()),
                                       RAD_INT("max_ctx", g.max_ctx),
                                       RAD_STR("q_dtype", g.dtype),
                                       RAD_STR("kv_dtype", g.kv_dtype),
                                       RAD_INT("group", RAD_FP8_BLOCK),
                                       RAD_STR("act", "sigmoid")),
                            RAD_NOWEIGHTS);
        if (op_attn_gq) {
            const rad_buf rd[4] = { w.q, qsa_sel, qsa_sequ, w.qg };
            const rad_buf wr[3] = { w.attn.x, w.attn.cq(), w.attn.cs() };
            RAD_ARCH_TRY(rad_op_reads (b, op_attn_gq, rd, 4));
            RAD_ARCH_TRY(rad_op_writes(b, op_attn_gq, wr, 3));
        }
    }

    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x));

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    /* Only where the NEXT block's norm will not do it; see rad_fp8.h's NormQuantFP8. */
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h.x}, {w.x});

    return RAD_OK;
}

inline void AttnGatedFP8::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;
    const bool frontier = frontier_kv && batch->draft_pass >= 2 && batch->draft_pass <= 5;
    const RadOperand store_slots = frontier ? brows(frontier_slots, T) : praw(slot, RAD_I32, T);
    const RadOperand store_cache = frontier ? RAD_B(frontier_kv) : kv_cache(kv, layer);

    if (!ext_in) nq_h.step(c, T);
    qg.step(c, w.h, w.qg, T);
    kp.step(c, w.h, w.k,  T);
    vp.step(c, w.h, w.v,  T);

    /* Four launches or one. The fused kernel takes the [q|gate] projection WHOLE and splits it
     * itself -- so `w.qg` goes in as it stands and not as the column slice the unfused norm needs
     * -- and it rotates k in place, which is what the unfused pair does too. */
    /* THE FUSED PATH READS ONE COMPONENT A TOKEN, so a pass carrying media tokens -- whose
     * components differ -- takes the unfused pair with the whole [3, T] instead. */
    const bool fuse_ok = T <= fused_rows && !rope_mixed(batch);
    const bool fused_kv = op_fused_kv && fuse_ok;
    if (fused_kv) {
        RAD_ISSUE_N(c, op_fused_kv, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), store_slots, store_cache);
    } else if (op_fused && fuse_ok) {
        RAD_ISSUE_N(c, op_fused, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE);
    } else {
        RAD_ISSUE_N(c, op_q_norm, T * g.n_head,
                    bcol(w.qg, 0, g.head_dim, T), RAD_W(w_q_norm), brows(w.q, T));
        RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv,
                    brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
        RAD_ISSUE(c, op_rope_q, brows(w.q, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    }

    if (!fused_kv)
        RAD_ISSUE(c, op_kv_store,
                  brows(w.k, T), brows(w.v, T), store_slots, store_cache);

    /* ONE ROW A QUERY IS WHAT MAKES THE SPARSE PATH WORK AT PREFILL, and it needs no new kernel
     * for the same reason the decode path needed none: `attn_paged` indexes its block table and
     * its length by the ROW, and nothing says a row has to be a sequence. A chunk of T queries
     * issued as T rows of q_len 1, each with its own selection and its own `seqused`, is exactly
     * per-query sparse attention -- and at decode, where T is the sequence count, it is the same
     * issue it has always been.
     *
     * DENSE BELOW THE BOUND, because there the two are the same answer and dense is cheaper: it
     * reads each key once a query TILE where this reads a table a query. `max_ctx_len` is the
     * host-side bound the batch already carries, so the test costs nothing.
     *
     * CHUNKED AT max_seqs, and that is not tidiness. The decode kernel's split-KV scratch is
     * sized AT DECLARE from the `max_seqs` and `q_len` parameters -- max_seqs * q_len_max * heads
     * rows -- and handing it T rows past that is refused as RAD_E_SCRATCH at issue. max_seqs rows
     * of q_len 1 is strictly inside what q_len_max >= 1 already bought, so the bound holds
     * whatever the kernel's band turns out to be. */
    const int64_t reach  = (int64_t)batch->max_ctx_len + batch->max_q_len;
    /* Branches reach this block only when reach <= qsa_exact_to. Every
     * visible block then fits the QSA budget, so dense attention with ancestor
     * masks is the same model, not an above-budget dense approximation.
     * A sparse batch was linearized before staging; its masks are chains. */
    const bool    sparse = !frontier && qsa_sel && qsa_sequ &&
                           (T == batch->n_seq || qsa_exact_to <= 0 || reach > qsa_exact_to);
    const bool gated = sparse && op_attn_gq;
    int64_t D = 0, DT = 0;
    if (op_attn_tree && !frontier && !sparse) batch_split(batch, &D, &DT);
    const int64_t decode_rows = batch->phase == RAD_PHASE_MIXED
                              ? batch->max_q_len_decode : batch->max_q_len;
    /* The ordinary prefill row does not consume ancestor masks. Its band is
     * selected by the whole batch, including any longer prefill suffix. */
    const bool wide_tree = D > 0 && batch->max_q_len > 64 / g.gqa() &&
                           batch->spec_parent;
    if (frontier) {
        RAD_ISSUE(c, op_frontier_attn[batch->draft_pass - 2],
                  brows(w.q, T), kv_cache(kv, layer),
                  praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                  brows(frontier_prefix, T), RAD_B(frontier_kv),
                  brows(frontier_ancestors, T), brows(w.attn.x, T),
                  RAD_B(frontier_meta), praw(batch->draft_out_ids, RAD_I32, batch->n_seq));
    } else if (wide_tree) {
        RAD_ISSUE_N(c, op_attn_tree, decode_rows,
                    brows(w.q, DT), kv_cache(kv, layer),
                    praw2(table, RAD_I32, D, kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    praw(batch->spec_parent, RAD_I32, DT), brows(w.attn.x, DT));
        if (D < batch->n_seq) {
            /* cu retains absolute query offsets. Keep full q/out views so the
             * ordinary causal prefill writes only its suffix, not the tree rows. */
            RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                        brows(w.q, T), kv_cache(kv, layer),
                        praw2(table + D * kvb->block_table_pitch, RAD_I32,
                              batch->n_seq - D, kvb->block_table_pitch),
                        praw(used + D, RAD_I32, batch->n_seq - D),
                        RAD_NONE, RAD_NONE,
                        praw(batch->cu_seqlens + D, RAD_I32, batch->n_seq - D + 1),
                        brows(w.attn.x, T), RAD_NONE);
        }
    } else if (gated) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            RAD_ISSUE_N(c, op_attn_gq, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw),
                        bcol_at(w.qg, off, g.n_head * 2 * hd, hd, hd, rows),
                        brow_slice(w.attn.cq(), off, rows, qw),
                        brow_slice(w.attn.cs(), off, rows, fp8_blocks(qw)));
        }
    } else if (sparse) {
        const int64_t chunk = qsa_rows();
        const int64_t qw = g.q_dim();
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            RAD_ISSUE_N(c, op_attn, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                        brow_slice(qsa_sequ, off, rows, 1),
                        RAD_NONE, RAD_NONE, RAD_NONE,
                        brow_slice(w.attn.x, off, rows, qw));
        }
    } else {
        RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                    brows(w.q, T), kv_cache(kv, layer),
                    praw2(table, RAD_I32, batch->n_seq,
                          kvb ? kvb->block_table_pitch : 0),
                    praw(used, RAD_I32, batch->n_seq),
                    RAD_NONE, RAD_NONE,
                    /* cu_seqlens. The batch may be ragged -- one prefill chunk beside other
                     * sequences one decode row each -- and without it the kernel has to
                     * assume a single q_len for the whole step, which it cannot. */
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn.x, T),
                    batch->tree_ancestors ? praw(batch->tree_ancestors, RAD_U16, batch->n_seq * 16) : RAD_NONE);
    }

    if (op_sig) {
        /* The gate is the strided second half of each head's [q|gate] pair. */
        RAD_ISSUE_N(c, op_sig, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.gate, T));
        RAD_ISSUE(c, op_mul, brows(w.attn.x, T), brows(w.gate, T), brows(w.attn.x, T));
    } else if (!gated)
        RAD_ISSUE_N(c, op_gq, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.attn.x, T),
                    brows(w.attn.cq(), T), brows(w.attn.cs(), T), brows(w.attn.x, T));

    o.step(c, w.attn, w.h.x, T);
    if (op_ar && !ar_taken(g, T, ar_out, ar_out_take))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_FP8_H */
