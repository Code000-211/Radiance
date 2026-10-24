/* rad_block_attn_gated_fp8.h -- gated GQA attention with block-scaled FP8 projections.
 *
 * The architecture (interleaved q|gate projection, partial M-RoPE, separate k and v, sigmoid
 * gate) is documented in rad_block_attn_gated.h. What differs here:
 *
 *   - q|gate, k, v and o are LinearFP8: an E4M3 weight with a bf16 [N/128][K/128] scale plane,
 *     through `gemm_nt_q` at dtype fp8a8.
 *   - One input quantise feeds the three input projections (they all read the post-norm hidden
 *     state); the gate pass quantises the out-projection's input.
 *   - Per-head norms, rope, kv_store and attn_paged stay bf16.
 *   - The KV cache dtype is `g.kv_dtype`, not `g.dtype`: it is a deployment choice that selects
 *     libr4d's fp8kv attention family. The attention op and kv_store both take it, so they agree.
 *   - Optional fused prologue, QSA sparse attention and an attention-zero mode (see declare).
 *
 * The q|gate projection is [2*n_head*head_dim, n_embd] = [12288, 5120] at 27B, scale plane
 * [96, 40]. Row-sharding by head keeps whole 128-row scale blocks only because 2*head_dim = 512 is
 * a multiple of 128; LinearFP8 checks this.
 */
#ifndef RAD_BLOCK_ATTN_GATED_FP8_H
#define RAD_BLOCK_ATTN_GATED_FP8_H

#include "rad_fp8.h"

#include <cstring>

namespace rad {
namespace arch {

struct AttnGatedFP8 {
    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]              the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]              normed input, then the o output */
        rad_buf qg   = 0;   /* [max_tok, n_head, 2*head_dim]  query and gate, interleaved */
        rad_buf k    = 0;   /* [max_tok, kv_dim] */
        rad_buf v    = 0;   /* [max_tok, kv_dim] */
        rad_buf q    = 0;   /* [max_tok, q_dim]   de-interleaved, normed, rotated */
        /* [rope_table_rows, rot] f32 cos/sin table, shared by every layer, filled once a step. Zero
         * if the plugin builds none; the fused prologue is then not requested. */
        rad_buf cos_sin = 0;
        ActFP8  attn{};     /* [max_tok, q_dim]   attention output, then gated in place */
        /* [max_tok, q_dim] de-interleaved sigmoid gate, used only with bf16 linears
         * (Geom::w_bf16), where the gate is `sigmoid` then `mul`; the fused gate quantiser always
         * writes codes. */
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
    /* The KV page in tokens; 0 means RAD_KV_BLOCK (16). Qwen4-Exp's QSA selects 4-token blocks,
     * and a page of 4 turns the selected set into a paged block table, so no new attention kernel
     * is needed (docs/QSA.md). The attention kernel's rows must carry a matching
     * C_EQ("block_size", ...) or nothing resolves. */
    int64_t     kv_block = 0;
    /* The QSA selection, when this layer has an indexer (produced by arch/common/rad_qsa.h; see
     * docs/QSA.md). On a sparse step the attention is issued with these in place of each row's
     * block table and context length: same op, same kernel, different list. `sel` is `topk + 1`
     * wide and the kernel takes its block-table pitch from the operand. */
    rad_buf     qsa_sel  = 0;
    rad_buf     qsa_sequ = 0;
    /* The shared-table read path (docs/QSA.md, "The masked read"), the prototype behind
     * `RADIANCE_QSA_TABLE=1` (arch/common/rad_qsa.h: qsa_table_on()). `qsa_nsel`/`qsa_selb` are the
     * table op's inputs -- `qsa_select`'s other two outputs, as block indices -- and
     * `qsa_tab`/`qsa_tlen`/`qsa_mask`/`qsa_roff` the planes it writes and this block's attention
     * reads. All zero, or `qsa_table` false, leaves every step on the per-row path, which is the
     * measurement's control arm. */
    rad_buf     qsa_nsel = 0, qsa_selb = 0;
    rad_buf     qsa_tab  = 0, qsa_tlen = 0, qsa_mask = 0, qsa_roff = 0;
    bool        qsa_table = false;
    int64_t     qsa_umax  = 0;    /* the planes' width, `min(tile * topk, blocks)` */
    int64_t     qsa_tile  = 16;   /* query rows one table covers */
    int64_t     qsa_ratio = 0;    /* the indexer's compress ratio, one page per block */
    int64_t     qsa_blocks = 0;   /* the widest block table */
    /* Context length up to which dense attention gives the same result, `budget + ratio - 1`.
     * Below it every complete block is selected, the paths agree to the bit, and dense is cheaper
     * at prefill (it reads each key once per query tile, sparse reads a table per query). Zero
     * means always sparse. */
    int64_t     qsa_exact_to = 0;
    /* `sel`'s row pitch is `topk + 1`; the extra slot is the partial page
     * (r4d_qsa_select_bf16.hip). */
    int64_t     qsa_topk = 0;
    /* Query rows per sparse attention launch, a scratch budget. The decode kernel's split-KV
     * partials are sized at declare from `max_seqs` and `q_len`, so issuing more rows than
     * `max_seqs` means declaring a larger value, which grows the scratch arena (the maximum over
     * ops) by `rows * q_len_band * n_head * (head_dim + 2) * 4` bytes: about 32 MiB at 256 rows on
     * this model, 63 MiB at 512, 253 MiB at 2048. With max_seqs alone, `--max-num-seqs 4` would
     * split a 2048-token prefill chunk into 512 launches per layer; 256 makes it 8, past which
     * launch cost no longer matters. */
    static constexpr int64_t kQsaRows = 256;
    int64_t qsa_rows() const { return g.max_seqs > kQsaRows ? g.max_seqs : kQsaRows; }
    /* The same rounded up to a whole number of tiles: a chunk boundary is a tile boundary, which
     * is what lets a chunk's table rows be sliced by `off / qsa_tile`. */
    int64_t qsa_chunk() const {
        return (qsa_rows() + qsa_tile - 1) / qsa_tile * qsa_tile;
    }
    /* The gated form takes the whole step in one launch: its band is q_len 1, so partials are only
     * `rows * n_head * (2 * head_dim + 8)` bytes (12.8 MiB for 2048 rows on this model). One
     * 2048-row launch is about 2.16 ms per layer versus 3.0 ms for eight 256-row launches, each
     * with its own tail. */
    int64_t qsa_gq_rows() const { return g.max_tok > qsa_rows() ? g.max_tok : qsa_rows(); }
    /* The rotation's pairing. Halves (NeoX, HF's rotate_half over the partial rotary width) and
     * adjacent pairs (GPT-J) are both self-consistent and pass every per-op check, so it must be
     * stated. "mrope" over one position component per token is NeoX (rad_block_attn_gated.h).
     *
     * The q|gate order is fixed as well: q is the first half of each head's pair, the gate the
     * second (as vLLM chunks q_gate). The unfused q norm reads column 0 of each head, the gate
     * quantiser column head_dim, and the fused prologue splits the same way. A swap would pass
     * silently, since q_norm and the sigmoid accept either half. */
    const char* mode  = "mrope";

    rad_weight w_norm = 0, w_q_norm = 0, w_k_norm = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    LinearFP8  qg{}, kp{}, vp{}, o{};
    NormQuantFP8 nq_h{};

    rad_op     op_fused = 0;                                 /* qk_norm_rope_gate, if it resolves */
    rad_op     op_fused_kv = 0;                              /* the same with kv_store folded in */
    int64_t    fused_rows = 0;                               /* widest T the fused ops take */
    rad_op     op_q_norm = 0, op_k_norm = 0, op_rope_q = 0, op_rope_k = 0;
    rad_op     op_attn = 0, op_kv_store = 0, op_gq = 0;
    rad_op     op_attn_gq = 0;                               /* attn_paged + gate_quant_fp8 */
    rad_op     op_tab = 0, op_attn_tab = 0;                  /* qsa_table + qsa_attn */
    /* The output gate as `sigmoid` then `mul`: the bf16-linear form (Geom::w_bf16), and with fp8
     * linears the chain gate_quant_fp8 fuses, where it is not served. `gate` is the sigmoid's
     * output (the wire's with bf16 linears, the block's own otherwise) and `q_gate` the chain's
     * quantiser (no op with bf16 linears). */
    rad_op     op_sig = 0, op_mul = 0;
    rad_buf    gate = 0;
    QuantFP8   q_gate{};
    rad_op     op_ar = 0, op_add = 0;
    rad_op     op_fill = 0;                                  /* the attention-zero rank: see declare */
    /* The next block absorbs this block's closing all-reduce, so this one must not issue it.
     * Counterpart of `ar_in` (this block's norm absorbs the predecessor's); separate because a
     * block is declared without knowing what follows. Both sides ask ar_fused_ok(g, T), so they
     * agree on whether the collective was issued. */
    bool ar_out = false;
    /* Which wires that consumer takes (ArOut, rad_fp8.h): the norm's by default, or a gated
     * residual's own (HyperConn::ar_take). Set before declare; both sides then ask
     * ar_taken(g, T, ar_out, ar_out_take). */
    int  ar_out_take = kArOutNorm;

    /* `fold`: the previous block left its delta in `w.h.x` for this norm to add. `add_out`: the
     * next block will not add, so this one issues the trailing add. See rad_fp8.h. */
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    /* Stores the pass's K and V exactly as step() does, with no query, attention or output. Used
     * by passes that only leave layer state behind, such as the MTP head's history pass over a
     * prompt chunk with no drafting row. */
    void store_kv(RadCtx* c, const RadBatch* batch) const;

    /* The block's tail, declared last by every rank: the all-reduce and the trailing add. */
    int  declare_out(RadBuilder* b, bool add_out);
    /* `own_ar` false: the caller issues the all-reduce itself (GdnFP8::step says when). */
    void step(RadCtx* c, const RadBatch* batch, bool own_ar = true,
              ArPieces* pieces = nullptr) const;
    bool issues_ar(int64_t T) const { return op_ar && !ar_taken(g, T, ar_out, ar_out_take); }
    void issue_ar(RadCtx* c, int64_t r0, int64_t rows) const {
        RAD_ISSUE_N(c, op_ar_ov ? op_ar_ov : op_ar, rows * g.n_embd,
                    brow_slice(w.h.x, r0, rows, g.n_embd), RAD_NONE);
    }
    /* Set before declare: the `overlap` issue_ar's collective is declared at, 0 for none. */
    int    ar_overlap = 0;
    rad_op op_ar_ov = 0;
};

inline int AttnGatedFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                                 rad_kvgroup group, int64_t rotary_dim, const Wire& wire,
                                 const Src& src, bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();
    const int64_t qg_dim = 2 * q_dim;

    /* Row spans, in the container's unsharded rows, used where the even shard is the wrong cut.
     *
     * Above the KV head count geom_from replicates KV heads (rad_arch.h): with 2 KV heads at
     * head_dim 256 and 4 ranks, the k/v entry is 512 rows and ranks 0-1 take [0, 256), ranks 2-3
     * [256, 512), where the even shard would hand rank 2 half a head. The scale plane takes the
     * same span in 128-blocks; kv_off is a multiple of head_dim, so it never cuts a block. At
     * world <= n_head_kv_all the span stays 0/0 and the even shard applies.
     *
     * Under a smaller attention world (Geom::attn_world) q|gate and the output also take this
     * rank's heads [attn_rank * n_head, +n_head) as a span. An attention-zero rank declares none
     * of them (see the `fill` below). */
    const bool    a_spanned  = g.attn_world != g.world;
    const bool    kv_spanned = g.world > g.n_head_kv_all || a_spanned;
    const int64_t kv_span_lo = kv_spanned ? g.kv_off : 0;
    const int64_t kv_span_hi = kv_spanned ? g.kv_off + kv_dim : 0;
    const int64_t qg_span_lo = a_spanned ? g.attn_rank * qg_dim : 0;
    const int64_t qg_span_hi = a_spanned ? (g.attn_rank + 1) * qg_dim : 0;
    const int64_t o_span_lo  = a_spanned ? g.attn_rank * q_dim : 0;
    const int64_t o_span_hi  = a_spanned ? (g.attn_rank + 1) * q_dim : 0;

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    /* `src.norm == nullptr`: the architecture prepares the input. A pre-norm block owns its norm
     * (rad_fp8.h's NormQuantFP8: add the previous delta, normalise, quantise). In a gated-residual
     * model, rad_block_hc.h's `hc_read` produces the block input from a 4x wider stream, so this
     * block takes `wire.h` as already normed and quantised and declares no norm weight.
     *
     * `ar_in` is refused in that mode: the fused all-reduce lives inside the norm, so it would be
     * silently dropped. The caller emits a standalone collective instead (`add_out`/`ar_out`). */
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

    /* One norm+quantise op feeds all three input projections (rad_fp8.h: NormQuantFP8). */
    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));

    /* An attention-zero rank holds no heads, so its output projection contributes zero to the
     * reduce. Its output wire is zeroed by one `fill` instead of running projections on zero
     * weights (and it keeps no KV pool, rad_kv_group_zero). The all-reduce and residual add run as
     * usual, and the norm above stays because a pre-norm block's `fold` and `ar_in` live in it. */
    if (g.attn_zero) {
        op_fill = rw(b, RAD_OP(b, "fill",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                    RAD_STR("dtype", g.dtype)),
                         RAD_NOWEIGHTS),
                     {}, {w.h.x});
        return declare_out(b, add_out);
    }
    RAD_ARCH_TRY(qg.declare(b, nm, g, nm.f("blk.%d.attn_qg", l), qg_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.qg }, w.h, w.qg,
                            0, {}, qg_span_lo, qg_span_hi));
    RAD_ARCH_TRY(kp.declare(b, nm, g, nm.f("blk.%d.attn_k", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.k }, w.h, w.k,
                            0, {}, kv_span_lo, kv_span_hi));
    RAD_ARCH_TRY(vp.declare(b, nm, g, nm.f("blk.%d.attn_v", l), kv_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { src.v }, w.h, w.v,
                            0, {}, kv_span_lo, kv_span_hi));

    /* The fused prologue (qk norms + rope + q|gate split) is a probe; the four unfused ops below
     * are declared either way and run when it does not resolve. Its schema takes a precomputed
     * cos/sin table instead of `theta`: `rope_table` fills `w.cos_sin` once a step for every
     * layer, and without a table the fusion is not requested.
     *
     * The kernel rotates NeoX pairs and has no `mode`. M-RoPE qualifies because the table is
     * filled at one component per token, which is exact when the components agree; step() takes
     * the fused path only then.
     *
     * No `gate_out` is passed: `gate_quant_fp8` reads the gate in place in the interleaved
     * projection.
     *
     * Bit-identical to the unfused ops, asserted in r4d_selftest: the kernel rounds the norm output
     * to bf16 before rotating and reproduces `r4d_rmsnorm_bf16`'s sum-of-squares order element by
     * element. A ULP-level difference would shift the engine's transcript within a few dozen
     * steps. */
    const bool rope_neox = !std::strcmp(mode, "neox") || !std::strcmp(mode, "mrope") || g.rope_mc;
    fused_rows = qk_fuse_rows(g);
    if (w.cos_sin && rope_neox && fused_rows > 0 && rad_op_declared(b, "qk_norm_rope_gate"))
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

    /* Declared before kv_store, which reads back the resolved `block_size`
     * (rad_block_attn_gated.h). */
    op_attn = RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", 0),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK),
                                /* The sparse path issues one row per query, so the split-KV
                                 * scratch is sized for the widest launch (kQsaRows). */
                                RAD_INT("max_seqs", (qsa_sel && qsa_sequ) ? qsa_chunk()
                                                                          : g.max_seqs),
                                RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS);
    /* The QSA selection buffers must be declared as reads of this op. Otherwise their last use is
     * `qsa_select`, the arena may overlay another buffer, and the attention reads garbage page
     * ids: an aperture violation outside the cache. rad_op_reads rather than rw() because the list
     * depends on whether the layer has an indexer. */
    if (op_attn) {
        const rad_buf rd[3] = { w.q, qsa_sel, qsa_sequ };
        rad_op_reads(b, op_attn, rd, (qsa_sel && qsa_sequ) ? 3 : 1);
        rad_op_writes(b, op_attn, &w.attn.x, 1);
    }

    /* The shared table a sparse step reads instead of one list per row (docs/QSA.md, "The masked
     * read"). Declared only when the caller declares the planes, the prototype's switch is on, and
     * a loaded library declares both ops; otherwise step() keeps the row path, and a fake builder
     * (tests/arch_fakes.h) whose vocabulary lacks them takes the row path too.
     *
     * `block_size` must equal the compress ratio: `selb` holds block-table entries and the table is
     * paged at the ratio, so a page holds exactly one block. Any other geometry keeps the row
     * path, which maps a block to its page like any other list. */
    const int64_t qsa_page = kv_block > 0 ? kv_block : (int64_t)RAD_KV_BLOCK;
    if (qsa_table && qsa_tab && qsa_tlen && qsa_mask && qsa_roff && qsa_nsel && qsa_selb &&
        qsa_umax > 0 && qsa_ratio > 0 && qsa_blocks > 0 && qsa_page == qsa_ratio &&
        rad_op_declared(b, "qsa_table") && rad_op_declared(b, "qsa_attn")) {
        op_tab = RAD_OP(b, "qsa_table",
                        RAD_PARAMS(RAD_RANGE("M", 1, qsa_chunk()), RAD_INT("blocks", qsa_blocks),
                                   RAD_INT("topk", qsa_topk), RAD_INT("tile", qsa_tile),
                                   RAD_INT("ratio", qsa_ratio), RAD_INT("u_max", qsa_umax)),
                        RAD_NOWEIGHTS);
        op_attn_tab = RAD_OP(b, "qsa_attn",
                             RAD_PARAMS(RAD_RANGE("M", 1, qsa_chunk()),
                                        RAD_INT("n_head", g.n_head),
                                        RAD_INT("head_dim", g.head_dim),
                                        RAD_INT("gqa", g.gqa()), RAD_INT("block_size", qsa_page),
                                        RAD_INT("ratio", qsa_ratio), RAD_INT("tile", qsa_tile),
                                        RAD_INT("topk", qsa_topk), RAD_INT("blocks", qsa_blocks),
                                        RAD_INT("max_seqs", qsa_chunk()),
                                        RAD_INT("u_max", qsa_umax),
                                        RAD_STR("q_dtype", g.dtype),
                                        RAD_STR("kv_dtype", g.kv_dtype)),
                             RAD_NOWEIGHTS);
        /* The planes outlive `qsa_select`, so the table op must declare them as its inputs and the
         * attention as its own: otherwise the arena may overlay another buffer and the read walks
         * garbage entries -- an aperture violation outside the cache. */
        if (op_tab) {
            const rad_buf in[2]  = { qsa_selb, qsa_nsel };
            const rad_buf out[4] = { qsa_tab, qsa_tlen, qsa_mask, qsa_roff };
            RAD_ARCH_TRY(rad_op_reads (b, op_tab, in, 2));
            RAD_ARCH_TRY(rad_op_writes(b, op_tab, out, 4));
        }
        if (op_attn_tab) {
            const rad_buf in[5] = { w.q, qsa_tab, qsa_tlen, qsa_mask, qsa_roff };
            RAD_ARCH_TRY(rad_op_reads (b, op_attn_tab, in, 5));
            RAD_ARCH_TRY(rad_op_writes(b, op_attn_tab, &w.attn.x, 1));
        }
    }

    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.k, w.v}, {});

    /* The fused prologue with the cache store folded in: a k head's block already holds its
     * token's rotated k, so it also writes k and v into the cache, byte-identical to kv_store.
     * Declared here because the cache block size is read back from the attention op above.
     * Resolves for E4M3 and bf16 caches. */
    if (op_fused && rad_op_declared(b, "qk_norm_rope_gate_kv_store"))
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

    /* Output gate, multiply and quantiser in one pass (`gate_quant_fp8`); at decode the dispatch
     * gaps between three small ops are comparable to the work. Each row is a (token, head) pair:
     * the gate is a fixed stride inside [q|gate], the attention output is contiguous, and the
     * [T * n_head, head_dim / 128] scale plane is byte-identical to the [T, q_dim / 128] the out
     * projection reads.
     *
     * The bf16 product is still written to `w.attn.x` and the codes are taken over it
     * (r4d_fused_quant_fp8.hip says why). The codes operand's dtype picks the quantiser, int8 when
     * the out projection is fed int8 (ActFP8::q8_fed). Serves prefill and decode: there is no
     * cross-row reduction.
     *
     * With bf16 linears (Geom::w_bf16) the gate is the plain `sigmoid` + `mul` pair instead.
     *
     * Where no loaded library declares gate_quant_fp8, or it does not resolve, the fp8 block runs
     * the chain it fuses: the gate into a buffer of its own, the multiply, then the activation
     * quantiser, which write what the fused op writes. A head of whole 128-column blocks makes
     * the quantiser's [T, q_dim / 128] scale plane the fused op's [T * n_head, head_dim / 128]. */
    auto gate_pair = [&](rad_buf out) {
        gate = out;
        op_sig = rw(b, RAD_OP(b, "sigmoid",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                   RAD_INT("n", g.head_dim), RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.qg}, {gate});
        op_mul = rw(b, RAD_OP(b, "mul",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.attn.x, gate}, {w.attn.x});
        return op_sig && op_mul ? RAD_OK : RAD_E_INVAL;
    };
    if (g.w_bf16) {
        RAD_ARCH_TRY(gate_pair(w.gate));
    } else {
        if (rad_op_declared(b, "gate_quant_fp8"))
            op_gq = rw(b, RAD_OP(b, "gate_quant_fp8",
                           RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * g.n_head),
                                      RAD_INT("n", g.head_dim), RAD_INT("group", RAD_FP8_BLOCK),
                                      RAD_STR("dtype", g.dtype), RAD_STR("act", "sigmoid")),
                           RAD_NOWEIGHTS),
                       {w.qg, w.attn.x}, {w.attn.cq(), w.attn.cs(), w.attn.x});
        if (!op_gq && g.head_dim % RAD_FP8_BLOCK == 0) {
            RAD_ARCH_TRY(gate_pair(decl_b(b, nm.f("blk.%d.attn_gate", l), g.act_dtype,
                                          {g.max_tok, q_dim})));
            RAD_ARCH_TRY(q_gate.declare(b, g, w.attn, q_dim));
            if (!q_gate.op) return RAD_E_INVAL;
        }
    }

    /* On the sparse path the gate is applied inside the attention's split-KV merge: each launch
     * row is one (token, head) at q_len 1, and the merge already holds the head_dim values the gate
     * and quantiser read, so one op merges, gates, rounds and quantises (byte-identical to the
     * pair). Declared only with a selection: the dense path also uses the prefill kernel, which
     * has no merge. */
    if (op_attn && qsa_sel && qsa_sequ && !g.w_bf16 &&
        rad_op_declared(b, "attn_paged_gate_quant")) {
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
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x,
                           0, {}, o_span_lo, o_span_hi));
    return declare_out(b, add_out);
}

inline int AttnGatedFP8::declare_out(RadBuilder* b, bool add_out) {
    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                  RAD_INT("min_bytes", g.wire_min_bytes)),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});
    /* The same collective for a caller that runs it beside other work (issue_ar, a pipelined
     * prefill): at `overlap` 1 it gets the rows that hold fewer CUs while waiting on the link. */
    if (g.world > 1 && ar_overlap > 0)
        op_ar_ov = rw(b, RAD_OP(b, "all_reduce",
                          RAD_PARAMS(RAD_INT("world_size", g.world),
                                     RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                     RAD_STR("dtype", g.dtype), RAD_INT("exact", g.wire_exact),
                                     RAD_INT("min_bytes", g.wire_min_bytes),
                                     RAD_INT("overlap", ar_overlap)),
                          RAD_NOWEIGHTS),
                      {w.h.x}, {w.h.x});

    /* Only when the next block's norm will not add (rad_fp8.h: NormQuantFP8). */
    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.x, w.h.x}, {w.x});

    return RAD_OK;
}

inline void AttnGatedFP8::store_kv(RadCtx* c, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    if (!ext_in) nq_h.step(c, T);
    if (op_fill) return;                /* the attention-zero rank holds no heads to store */
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot = kvb ? kvb->slot_mapping : nullptr;
    kp.step(c, w.h, w.k, T);
    vp.step(c, w.h, w.v, T);
    RAD_ISSUE_N(c, op_k_norm, T * g.n_head_kv, brows(w.k, T), RAD_W(w_k_norm), brows(w.k, T));
    RAD_ISSUE(c, op_rope_k, brows(w.k, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, op_kv_store,
              brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
}

inline void AttnGatedFP8::step(RadCtx* c, const RadBatch* batch, bool own_ar,
                               ArPieces* pieces) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;

    if (!ext_in) nq_h.step(c, T);
    if (op_fill) {
        /* Attention-zero rank: zero output, then the reduce and the residual. */
        RAD_ISSUE_N(c, op_fill, T, brows(w.h.x, T));
        if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
            RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
        if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
        return;
    }
    qg.step(c, w.h, w.qg, T);
    kp.step(c, w.h, w.k,  T);
    vp.step(c, w.h, w.v,  T);

    /* Fused or unfused prologue. The fused kernel takes the whole [q|gate] projection and splits it
     * itself, and rotates k in place like the unfused ops. It reads one rotary component per
     * token, so a pass with media tokens (differing components) takes the unfused path with the
     * full [3, T] positions. */
    const bool fuse_ok = T <= fused_rows && !rope_mixed(batch);
    const bool fused_kv = op_fused_kv && fuse_ok;
    if (fused_kv) {
        RAD_ISSUE_N(c, op_fused_kv, T,
                    brows(w.qg, T), brows(w.k, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_W(w_q_norm), RAD_W(w_k_norm),
                    brows(w.q, T), brows(w.k, T), RAD_NONE,
                    brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));
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
                  brows(w.k, T), brows(w.v, T), praw(slot, RAD_I32, T), kv_cache(kv, layer));

    /* Sparse attention issues one row per query. `attn_paged` indexes its block table and length
     * by row, and a row need not be a sequence, so T queries issued as T rows of q_len 1, each with
     * its own selection and `seqused`, is per-query sparse attention; at decode T is the sequence
     * count.
     *
     * Below `qsa_exact_to` the dense path gives the same result more cheaply (see the field);
     * `max_ctx_len` is a host-side bound the batch already carries.
     *
     * Launches are chunked because the split-KV scratch is sized at declare from `max_seqs` and
     * `q_len` (rows = max_seqs * q_len_max * heads); more rows are refused as RAD_E_SCRATCH at
     * issue. Rows of q_len 1 up to the declared max_seqs always fit. */
    const int64_t reach  = (int64_t)batch->max_ctx_len + batch->max_q_len;
    const bool    sparse = qsa_sel && qsa_sequ &&
                           (T == batch->n_seq || qsa_exact_to <= 0 || reach > qsa_exact_to);
    /* The shared-table path takes the ungated attention and runs the gate epilogue below unfused:
     * the fused op has no table form, and the two epilogues write the same bytes. */
    const bool tabulated = sparse && op_tab && op_attn_tab;
    const bool gated = sparse && op_attn_gq && !tabulated;
    /* A piece at a time where the caller takes the all-reduce that way (ArPieces). On the gated
     * sparse path every query row is its own (q_len 1) and the gate rides in the merge, so the
     * attention and out projection can finish a piece's rows and hand off its reduction while the
     * next piece's attention runs. Each piece is launched as a step of the piece's width would
     * launch it. */
    if (gated && !own_ar && pieces && pieces->cuts(T) && !op_add) {
        const int64_t chunk = qsa_gq_rows();
        const int64_t qw = g.q_dim(), hd = g.head_dim;
        for (int64_t p0 = 0; p0 < T; p0 += pieces->rows) {
            const int64_t pn = T - p0 < pieces->rows ? T - p0 : pieces->rows;
            for (int64_t off = p0; off < p0 + pn; off += chunk) {
                const int64_t rows = (p0 + pn - off) < chunk ? (p0 + pn - off) : chunk;
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
            o.step(c, w.attn, w.h.x, T, p0, pn);
            if (p0 + pn < T) pieces->hand(p0, pn);
        }
        return;
    }
    if (gated) {
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
    } else if (tabulated) {
        /* One union per tile of `qsa_tile` rows, then the attention reads it through each row's
         * membership mask. A last chunk narrower than a tile runs on the row path below: the table
         * needs one whole tile, and the two paths are byte-identical (tests/r4d_emu/test_qsa.cpp,
         * "masked read"), so the arm a row lands in cannot move its numbers.
         *
         * `bt` is the sequence's own block table here, not the row's selection: the table holds
         * block indices and the reader maps them, in block order, to the pages the prefix cache
         * gave them. */
        const int64_t chunk = qsa_chunk();
        const int64_t qw = g.q_dim(), tile = qsa_tile;
        const int64_t btp = kvb ? kvb->block_table_pitch : 0;
        for (int64_t off = 0; off < T; off += chunk) {
            const int64_t rows = (T - off) < chunk ? (T - off) : chunk;
            if (rows < tile) {
                RAD_ISSUE_N(c, op_attn, 1,
                            brow_slice(w.q, off, rows, qw),
                            kv_cache(kv, layer),
                            brow_slice(qsa_sel, off, rows, qsa_topk + 1),
                            brow_slice(qsa_sequ, off, rows, 1),
                            RAD_NONE, RAD_NONE, RAD_NONE,
                            brow_slice(w.attn.x, off, rows, qw));
                continue;
            }
            const int64_t tiles = (rows + tile - 1) / tile;
            const int64_t t0 = off / tile;
            RAD_ISSUE_N(c, op_tab, rows,
                        brow_slice(qsa_selb, off, rows, qsa_topk + 1),
                        brow_slice(qsa_nsel, off, rows, 1),
                        praw(batch->positions + off, RAD_I32, rows),
                        brow_slice(qsa_tab, t0, tiles, qsa_umax),
                        brow_slice(qsa_tlen, t0, tiles, 1),
                        brow_slice(qsa_mask, off, rows, qsa_umax),
                        brow_slice(qsa_roff, off, rows, 1));
            RAD_ISSUE_N(c, op_attn_tab, 1,
                        brow_slice(w.q, off, rows, qw),
                        kv_cache(kv, layer),
                        brow_slice(qsa_tab, t0, tiles, qsa_umax),
                        brow_slice(qsa_tlen, t0, tiles, 1),
                        brow_slice(qsa_mask, off, rows, qsa_umax),
                        brow_slice(qsa_roff, off, rows, 1),
                        praw(batch->positions + off, RAD_I32, rows),
                        praw2(table, RAD_I32, batch->n_seq, btp),
                        praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                        brow_slice(w.attn.x, off, rows, qw));
        }
    } else if (sparse) {
        const int64_t chunk = qsa_chunk();
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
                    /* cu_seqlens: the batch may be ragged (a prefill chunk beside decode rows). */
                    praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                    brows(w.attn.x, T));
    }

    if (gated) {
        /* the sparse attention gated and quantised in its merge */
    } else if (op_sig) {
        /* The gate is the strided second half of each head's [q|gate] pair. */
        RAD_ISSUE_N(c, op_sig, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(gate, T));
        RAD_ISSUE(c, op_mul, brows(w.attn.x, T), brows(gate, T), brows(w.attn.x, T));
        q_gate.step(c, T);
    } else {
        RAD_ISSUE_N(c, op_gq, T * g.n_head,
                    bcol(w.qg, g.head_dim, g.head_dim, T), brows(w.attn.x, T),
                    brows(w.attn.cq(), T), brows(w.attn.cs(), T), brows(w.attn.x, T));
    }

    o.step(c, w.attn, w.h.x, T);
    if (own_ar && op_ar && !ar_taken(g, T, ar_out, ar_out_take))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_ATTN_GATED_FP8_H */
