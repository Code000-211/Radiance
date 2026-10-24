/* rad_block_gdn_fp8.h -- the gated-delta-net block with block-scaled FP8 projections.
 *
 * THE ARCHITECTURE IS rad_block_gdn.h's AND IS DOCUMENTED THERE: that `A_log` is a log and the
 * kernels take the exponent themselves, that q and k are L2-normalised rather than RMS-normalised
 * and that this happens inside the conv/recurrent kernels, why the prefill and decode paths are two
 * declared sequences and why that is not the branch spec §2.4 forbids, and why both take
 * `num_accepted`. None of it changes with the weight format.
 *
 * ============================== a AND b SPLIT OUT; z DOES NOT ==============================
 *
 * rad_block_gdn.h fuses in_proj_qkv, in_proj_a and in_proj_b into ONE weight and one GEMM, because
 * [q|k|v|a|b] is the row gdn_conv_prep reads. THAT FUSION IS NOT AVAILABLE HERE, and the reason is
 * the checkpoint rather than a limitation:
 *
 *   in_proj_qkv.weight   F8_E4M3 [10240, 5120]   with weight_scale_inv [80, 40]
 *   in_proj_z.weight     F8_E4M3 [ 6144, 5120]   with weight_scale_inv [48, 40]
 *   in_proj_a.weight     BF16    [48, 5120]      -- no scale plane
 *   in_proj_b.weight     BF16    [48, 5120]      -- no scale plane
 *
 * a and b emit ONE COLUMN PER VALUE HEAD, 48 of them, and 48 is not a whole 128-row scale block --
 * so the quantiser that produced this checkpoint left them in bf16, and no concatenation of an
 * E4M3 tensor with a BF16 one is a tensor. They become a second, tiny bf16 `gemm_nt`: [96, 5120],
 * 0.5% of the layer's projection bytes, and the price of it is one extra launch.
 *
 * a and b are separate operands of gdn_conv_prep and gdn_recurrent_update in both blocks -- the
 * bf16 block passes them as column views of the fused row, this one as column views of their own
 * buffer -- so nothing downstream changes shape. Ref sizes its `x` read as
 * `conv_dim + (a present ? 0 : 2*H)` for precisely this case.
 *
 * THE OUTPUT GATE IS THE OPPOSITE CASE and it IS fused, into `ssm_inz`: in_proj_z is E4M3 with its
 * own scale plane, its rows are a whole number of blocks, and it reads the same normed activation.
 * So [q|k|v|z] is one weight, one launch and the same bytes -- see declare() for why one call is
 * cheaper than two. What that costs is that `w.in.x` is conv_dim + v_dim wide, and the convolution
 * and the gate are COLUMN VIEWS of it rather than buffers of their own: gdn_conv_prep,
 * gdn_conv_update and gdn_recurrent_update all take a row pitch on the operand concerned.
 *
 * ============================== THE OUT-PROJECTION'S INPUT IS RANK 3 ==============================
 *
 * `o` is [tokens, value heads, head width] because libr4d's chunk-scan and recurrent kernels
 * require rank 3 there, and it is the out-projection's activation -- so the quantiser has to accept
 * it. It does: rows are dim 0 and everything after is the row, which is the reading r4d_row_pitch
 * already takes and which the consuming GEMM's K forces. Reading it as one row per HEAD would put a
 * scale on a grid the GEMM does not read.
 */
#ifndef RAD_BLOCK_GDN_FP8_H
#define RAD_BLOCK_GDN_FP8_H

#include "rad_fp8.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

struct GdnFP8 {
    /* The same geometry the bf16 block takes, with one addition: `ab_dim` is what the split costs
     * in declared width, and it is 2 * n_head_v. */
    struct Config {
        int64_t n_head_k   = 0;
        int64_t head_k     = 0;
        int64_t n_head_v   = 0;
        int64_t head_v     = 0;
        int64_t conv_width = 4;
        int64_t chunk      = 64;
        /* THE OUTPUT GATE'S ACTIVATION, and it is a CHECKPOINT PROPERTY rather than a family
         * one. `RMSNormGated` defaults to silu and every Qwen3.5/3.6 release leaves it there,
         * but Qwen4-Exp states `output_gate_type: "sigmoid"` and means it -- a gate of
         * sigmoid(z) where this said silu(z) scales every value head's output by roughly z
         * again. libr4d has compiled both arms all along (r4d_gdn_gated_rmsnorm's `gn_act`) and
         * `gdn_recurrent_update` composes the same parameter, so this is a string to pass and
         * not a kernel to write. "silu" or "sigmoid"; anything else is refused below. */
        const char* act    = "silu";
        bool tree_state = false;  /* DLoop: immediate winner commit for every decode batch. */

        int64_t k_dim()    const { return n_head_k * head_k; }
        int64_t v_dim()    const { return n_head_v * head_v; }
        int64_t conv_dim() const { return 2 * k_dim() + v_dim(); }
        int64_t ab_dim()   const { return 2 * n_head_v; }
    };

    struct Wire {
        rad_buf x    = 0;   /* [max_tok, n_embd]           the residual stream */
        ActFP8  h{};        /* [max_tok, n_embd]           normed input, then the out projection */
        /* ONE BUFFER, TWO PROJECTIONS: [q|k|v] and the output gate come out of one GEMM (see
         * ssm_inz in declare()), so the gate is this buffer's right-hand `v_dim` columns and the
         * convolution reads the left-hand `conv_dim`. Both are `bcol` views, which is why every
         * consumer of either takes a row pitch. */
        ActFP8  in{};       /* [max_tok, conv_dim+v_dim]   [q|k|v] then the output gate */
        rad_buf ab   = 0;   /* [max_tok, 2*n_head_v]       the decay and beta inputs, bf16 */
        rad_buf q    = 0;   /* [max_tok, n_head_k, head_k] post-convolution, l2-normed */
        rad_buf k    = 0;   /* [max_tok, n_head_k, head_k] */
        rad_buf v    = 0;   /* [max_tok, n_head_v, head_v] */
        rad_buf gdec = 0;   /* [max_tok, n_head_v] */
        rad_buf beta = 0;   /* [max_tok, n_head_v] */
        rad_buf kkt  = 0;   /* [max_tok, n_head_v, chunk] */
        ActFP8  o{};        /* [max_tok, n_head_v, head_v] -- RANK 3, see the header */
    };

    struct Src {
        const char* norm     = nullptr;
        const char* in_qkv   = nullptr;   /* fp8 */
        const char* in_a     = nullptr;   /* bf16 */
        const char* in_b     = nullptr;   /* bf16 */
        const char* in_z     = nullptr;   /* fp8 */
        const char* conv1d   = nullptr;
        const char* a_log    = nullptr;
        const char* dt_bias  = nullptr;
        const char* out_norm = nullptr;
        const char* out      = nullptr;   /* fp8 */
    };

    Geom        g{};
    Config      cfg{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv_state = 0, kv_conv = 0;

    rad_weight w_norm = 0, w_ab = 0, w_conv = 0;
    /* The input was normed and quantised by the caller; this block declares no norm. */
    bool       ext_in = false;
    rad_weight w_a_log = 0, w_dt_bias = 0, w_out_norm = 0;
    LinearFP8  in{}, out{};
    NormQuantFP8 nq_h{};
    QuantFP8   q_o{};

    rad_op op_ab = 0;
    rad_op op_conv_prep = 0, op_kkt = 0, op_scan = 0, op_gnorm = 0;   /* prefill */
    rad_op op_conv_update = 0, op_recur = 0;                          /* decode */
    rad_op op_tree_conv = 0, op_tree_recur = 0, op_tree_commit = 0, op_tree_i8g = 0;
    rad_buf tree_raw = 0, tree_factors = 0;
    /* `gdn_conv_recurrent_update`, if it resolves: the pair above as one launch, the convolution
     * computed in the recurrent update's prologue. 0 leaves the pair, which is declared either
     * way; cr_rows is the widest decode window the fold serves. */
    rad_op  op_cr = 0;
    int64_t cr_rows = 0;
    rad_op op_ar = 0, op_add = 0;
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
    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                 rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s,
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
    void commit(RadCtx* c, const RadBatch* batch) const;
};

inline int GdnFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& c, int l,
                           rad_kvgroup state, rad_kvgroup conv, const Wire& wire, const Src& s,
                           bool fold, bool add_out,
                           int ar_in, bool ar_out_) {
    g = geom;
    ar_out = ar_out_; cfg = c; w = wire; layer = l; kv_state = state; kv_conv = conv;
    if (cfg.tree_state && g.max_spec > 64) {
        fprintf(stderr, "radiance: GDN tree-state verification supports at most 65 rows "
                        "(64 draft nodes); max_spec=%lld is unsupported\n",
                (long long)g.max_spec);
        return RAD_E_UNSUPPORTED;
    }
    if (cfg.n_head_k <= 0 || cfg.head_k <= 0 || cfg.n_head_v <= 0 || cfg.head_v <= 0 ||
        cfg.conv_width <= 0 || cfg.chunk <= 0)
        return RAD_E_INVAL;
    if (cfg.n_head_v % cfg.n_head_k != 0) {
        fprintf(stderr, "radiance: %lld value heads is not a multiple of %lld key heads\n",
                (long long)cfg.n_head_v, (long long)cfg.n_head_k);
        return RAD_E_INVAL;
    }

    /* THE QUERY SCALE, AND IT IS NOT OPTIONAL. `gdn_chunk_scan` and `gdn_recurrent_update` both
     * carry a `scale` that multiplies the output -- o = scale * e^g Q S^T + ... -- and it defaults to
     * 1. FLA's chunk_gated_delta_rule defaults it to head_k^-0.5 and vllm-radiance passes exactly
     * that; llama.cpp's fused GDN does the same.
     *
     * IT LOOKS LIKE IT CANCELS AND IT DOES NOT. The scan's output goes straight into the gated RMS
     * norm, and an RMS norm divides a constant factor back out -- so scaling `o` by c should change
     * nothing. That argument is wrong here for one reason: the scan output of this model is around
     * 1e-6, `mean(o^2)` is then around 1e-12, and the norm's eps is 1e-6. eps DOMINATES the variance,
     * the norm degenerates into a fixed division by sqrt(eps), and the factor passes straight through
     * to the residual stream.
     *
     * Leaving it at 1 is visible end to end: the scan output comes out larger than the reference
     * implementations' by exactly sqrt(head_k), and with forty-eight of this model's sixty-four
     * layers delta net the residual stream then carries a delta-net contribution that many times
     * too large -- fluent grammar with no content. */
    const double q_scale = 1.0 / std::sqrt((double)cfg.head_k);

    if (!cfg.act || (std::strcmp(cfg.act, "silu") != 0 && std::strcmp(cfg.act, "sigmoid") != 0)) {
        fprintf(stderr, "radiance: the delta net's output gate is '%s'; libr4d's gated norm "
                        "implements silu and sigmoid and nothing else\n", cfg.act ? cfg.act : "(null)");
        return RAD_E_INVAL;
    }
    const int64_t v_dim    = cfg.v_dim();
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t ab_dim   = cfg.ab_dim();

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
    ext_in = (s.norm == nullptr);
    if (ext_in) {
        if (!w.h.x || (!g.w_bf16 && (!w.h.q || !w.h.s)) || ar_in != kArNone) return RAD_E_INVAL;
    } else {
        RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), s.norm));
        w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    }
    /* The name maps ahead of the weights they name: a checkpoint is asked what a weight is
     * through its map, at the declaration. */
    RAD_ARCH_TRY(map_concat(b, nm.f("blk.%d.ssm_ab.weight", l),     { s.in_a, s.in_b }, 0));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_conv1d.weight", l), s.conv1d));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_a_log", l),         s.a_log));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_dt.bias", l),       s.dt_bias));
    RAD_ARCH_TRY(map_copy  (b, nm.f("blk.%d.ssm_norm.weight", l),   s.out_norm));
    /* The decay and beta projections, kept bf16 because the checkpoint kept them bf16 -- see the
     * header. Row-sharded with the value heads they belong to: a and b are concatenated along dim 0
     * so rank r owns rows [r*n_head_v, (r+1)*n_head_v) of EACH half, which is what the two-part
     * row split says and what a contiguous slice of the concatenation would not give. */
    w_ab       = decl_w(b, nm.f("blk.%d.ssm_ab.weight", l),     RAD_BF16, {ab_dim, g.n_embd},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                        {cfg.n_head_v, cfg.n_head_v});
    /* The depthwise convolution is per channel, so it shards with the channels it convolves --
     * and those channels are [q|k|v], three groups in one stored tensor, so the split is theirs. */
    w_conv     = decl_w(b, nm.f("blk.%d.ssm_conv1d.weight", l), RAD_BF16,
                        {conv_dim, cfg.conv_width},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l), 0,
                        {cfg.k_dim(), cfg.k_dim(), cfg.v_dim()});
    w_a_log    = decl_w(b, nm.f("blk.%d.ssm_a_log", l),         RAD_F32, {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    w_dt_bias  = decl_w(b, nm.f("blk.%d.ssm_dt.bias", l),       RAD_F32, {cfg.n_head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp_layer(l));
    w_out_norm = decl_w(b, nm.f("blk.%d.ssm_norm.weight", l),   RAD_F32, {cfg.head_v},
                        RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));


    /* One norm and one quantise of the normed hidden state, in one op; the qkv and z
     * projections both read it. rad_fp8.h's NormQuantFP8 says why they are not two. */
    if (!ext_in)
        RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                                  fold ? w.x : 0, 0, ar_in));
    /* TWO CHECKPOINT TENSORS, FOUR PROJECTIONS, ONE GEMM. in_proj_qkv is [q|k|v] stacked along
     * dim 0 and in_proj_z is the output gate; both read the SAME normed, quantised hidden state
     * and differ only in their rows, so they are one weight here and one launch at step time.
     *
     * WHY ONE CALL AND NOT TWO. The decode GEMM is bandwidth-bound but not at small N: two calls
     * at 5120 and 3072 rows cost more between them than one call at 8192, because each pays its
     * own ramp and tail on top of its launch. Over this model's delta-net layers that adds up to a
     * measurable share of a decode step. It is bit-identical: nothing about a row's arithmetic
     * changes, and both extents are already past the split-K threshold so the summation order does
     * not move either.
     *
     * THE ROW PARTS ARE THE WHOLE POINT. Rank r must own its slice of EACH of q, k, v and z, so
     * the shard is declared as four parts and not as one contiguous cut -- which would give rank 0
     * all of q, all of k and a third of v, a well-formed wrong model. */
    RAD_ARCH_TRY(in.declare(b, nm, g, nm.f("blk.%d.ssm_inz", l), conv_dim + v_dim, g.n_embd,
                            RAD_SHARD_ROW, grp_layer(l), { s.in_qkv, s.in_z }, w.h, w.in.x, 0,
                            { cfg.k_dim(), cfg.k_dim(), cfg.v_dim(), cfg.v_dim() }));
    /* NOT SLICED, and why it could need to be is worth keeping. This is a 96-column bf16 GEMM --
     * half a percent of the layer's projection bytes -- and a bf16 kernel for this shape that
     * stopped at M = 64 would leave a hole in its bucket table above 64. Since every op in a step
     * is issued at the same token count, that one hole would cap the prefill chunk of the entire
     * 27B model at 64 tokens, and the workaround is a launch per 64 rows in every GDN layer. The
     * cap was a launcher guard the kernel does not need: MT row tiles live in a block and grid.y
     * covers the rest, so M is general, and with the guard raised this is one launch.
     *
     * AT DECODE THE COST IS THE LAUNCH COUNT, not the arithmetic. `gemm_nt` is the most-dispatched
     * kernel in a decode step and it is almost entirely fixed cost, spread over three separate
     * small per-layer projections that all read the same activation: the MoE router (one a layer),
     * the shared expert's gate (one a layer, RANK 0 ONLY, so on the rank the all-reduce is already
     * waiting for) and this a|b (one a GDN layer).
     *
     * Fusing them is the lever and it is not a small one: they read the same activation but hold
     * different weights, so a merge wants either contiguous weights -- a re-convert, which is
     * ruled out -- or a GROUPED gemm_nt taking a table of weight pointers the way moe_gemm_q
     * does. */
    op_ab = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", ab_dim),
                              RAD_INT("K", g.n_embd), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_ab)), {w.h.x}, {w.ab});

    /* ---- the chunked path. Unchanged from the bf16 block except that `a` and `b_gate` are views
     * of `ab` rather than of the projection row's tail. */
    op_conv_prep = rw(b, RAD_OP(b, "gdn_conv_prep",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                     RAD_INT("head_v", cfg.head_v), RAD_INT("chunk", cfg.chunk),
                                     RAD_INT("conv_width", cfg.conv_width)),
                          RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias)),
                      {w.in.x, w.ab}, {w.q, w.k, w.v, w.gdec, w.beta});
    op_kkt  = rw(b, RAD_OP(b, "gdn_kkt_solve",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k)),
                     RAD_NOWEIGHTS),
                 {w.k, w.beta, w.gdec}, {w.kkt});
    op_scan = rw(b, RAD_OP(b, "gdn_chunk_scan",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                RAD_INT("head_v", cfg.head_v), RAD_F64("scale", q_scale)),
                     RAD_NOWEIGHTS),
                 {w.q, w.k, w.v, w.kkt, w.gdec, w.beta}, {w.o.x});
    op_gnorm = rw(b, RAD_OP(b, "gdn_gated_rmsnorm",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("channels", cfg.head_v),
                                 RAD_F64("eps", g.eps), RAD_STR("act", cfg.act)),
                      RAD_WEIGHTS(w_out_norm)),
                  {w.o.x, w.in.x}, {w.o.x});

    /* ---- the recurrent path. */
    op_conv_update = rw(b, RAD_OP(b, "gdn_conv_update",
                            RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok),
                                       RAD_INT("head_k", cfg.head_k),
                                       RAD_INT("head_v", cfg.head_v),
                                       RAD_INT("conv_width", cfg.conv_width)),
                            RAD_WEIGHTS(w_conv)),
                        {w.in.x}, {w.q, w.k, w.v});
    /* WHAT THE STATE CACHE MEANS, and it is not derivable from the operands. libr4d's recurrent
     * update has two state contracts. PER-CANDIDATE writes one full state per candidate token and
     * needs `1 + n_spec` distinct slots; ANCHOR keeps one committed state a sequence and replays
     * the accepted prefix onto it out of a scratch slot, which is two slots whatever the depth.
     * Same `o` either way, and a completely different `state` -- so a reference that implements one
     * of them and is handed the other computes a correct answer to the other question.
     *
     * The engine already chooses: KVGroupPlan::state_copies is 2 when the deployment speculates
     * and 1 when it does not, which is exactly this condition. Saying it here is what lets a
     * reference DECLINE the form it does not implement instead of disagreeing with it, and what
     * lets the kernel refuse a caller whose pool was sized for the other one. Under tree_state
     * neither op is issued (step() routes decode to the tree kernels), and a group whose every
     * layer serves them keeps one copy (KVGroupInfo::tree_bound). */
    op_recur = rw(b, RAD_OP(b, "gdn_recurrent_update",
                      RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_k", cfg.head_k),
                                 RAD_INT("head_v", cfg.head_v),
                                 RAD_INT("n_head_v", cfg.n_head_v),
                                 RAD_F64("scale", q_scale),
                                 RAD_F64("eps", g.eps), RAD_STR("act", cfg.act),
                                 RAD_STR("state_form",
                                         g.max_spec > 0 ? "anchor" : "per_candidate")),
                      RAD_WEIGHTS(w_a_log, w_dt_bias, w_out_norm)),
                  {w.q, w.k, w.v, w.ab, w.in.x}, {w.o.x});
    /* The epilogue's codes, for an fp8 or int8 out-projection -- the int8 pair when the model feeds
     * it (ActFP8::q8_fed), whose dtype picks the kernel's quantiser; a bf16 one reads `o.x` and
     * nothing else, and the operands are left absent. */
    if (op_recur && w.o.cq()) {
        const rad_buf oq[2] = { w.o.cq(), w.o.cs() };
        RAD_ARCH_TRY(rad_op_writes(b, op_recur, oq, 2));
    }
    /* THE FOLD, over the same weights and caches. Its window is a register preload, so the band
     * stops at eight tokens a sequence; not RAD_ARCH_TRY, since a plugin without the row is the
     * ordinary case and the pair runs. */
    {
        const int64_t hi = g.max_tok < 16 ? g.max_tok : 16;
        op_cr = rw(b, RAD_OP(b, "gdn_conv_recurrent_update",
                       RAD_PARAMS(RAD_RANGE("q_len", 1, hi), RAD_INT("head_k", cfg.head_k),
                                  RAD_INT("head_v", cfg.head_v),
                                  RAD_INT("conv_width", cfg.conv_width),
                                  RAD_INT("n_head_v", cfg.n_head_v), RAD_F64("scale", q_scale),
                                  RAD_F64("eps", g.eps), RAD_STR("act", cfg.act),
                                  RAD_STR("state_form",
                                          g.max_spec > 0 ? "anchor" : "per_candidate")),
                       RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias, w_out_norm)),
                   {w.in.x, w.ab}, {w.o.x});
        if (op_cr && w.o.cq()) {
            const rad_buf oq[2] = { w.o.cq(), w.o.cs() };
            RAD_ARCH_TRY(rad_op_writes(b, op_cr, oq, 2));
        }
        if (op_cr) cr_rows = hi;
    }

    if (cfg.tree_state) {
        const int64_t capacity = g.max_spec > 15 ? 65 : 16;
        /* Only decode rows reach these buffers: DT <= D*qd, D <= max_seqs (batch builder), and
         * qd <= capacity (the tree ops' q_len band); prefill and commit-without-decode skip them. */
        const int64_t tree_tokens = g.max_seqs>0 && g.max_seqs<=g.max_tok/capacity
                                  ? g.max_seqs*capacity : g.max_tok;
        tree_raw = decl_b(b, nm.f("blk.%d.gdn_tree_raw", l), RAD_BF16,
                            {tree_tokens, conv_dim}, RAD_BUF_PERSIST);
        tree_factors = decl_b(b, nm.f("blk.%d.gdn_tree_factors", l), RAD_F32,
                                {tree_tokens, cfg.n_head_v, 257}, RAD_BUF_PERSIST);
        if (!tree_raw || !tree_factors) return RAD_E_INVAL;
        const RadParam tp[] = {
            RAD_RANGE("q_len", 1, capacity), RAD_INT("head_k", cfg.head_k),
            RAD_INT("head_v", cfg.head_v), RAD_INT("n_head_k", cfg.n_head_k),
            RAD_INT("n_head_v", cfg.n_head_v), RAD_INT("conv_width", cfg.conv_width),
            RAD_INT("node_capacity", capacity), RAD_INT("history_ring", 0),
            RAD_F64("scale", q_scale), RAD_F64("eps", g.eps), RAD_STR("act", cfg.act),
            RAD_INT("state_group", state), RAD_INT("conv_group", conv), RAD_INT("layer", l),
            RAD_INT("raw_buffer", tree_raw), RAD_INT("factor_buffer", tree_factors)
        };
        if (cfg.head_k==128 && cfg.head_v==128 && cfg.conv_width==4 &&
            w.o.q8_fed && w.o.q8 && w.o.s8) {
            RadParam tf[17]{};
            for (int i=0;i<16;++i) tf[i]=tp[i];
            tf[16]=RAD_INT("max_tokens",tree_tokens);
            op_tree_i8g = rw(b, rad_decl_op(b, "gdn_conv_recurrent_tree_i8g", tf, 17,
                                           RAD_WEIGHTS(w_conv, w_a_log, w_dt_bias, w_out_norm)),
                             {w.in.x, w.ab}, {tree_raw, tree_factors, w.o.q8, w.o.s8});
            if (!op_tree_i8g) return RAD_E_UNSUPPORTED;
        } else {
            op_tree_conv = rw(b, rad_decl_op(b, "gdn_conv_tree", tp, 16, RAD_WEIGHTS(w_conv)),
                              {w.in.x}, {w.q, w.k, w.v, tree_raw});
            op_tree_recur = rw(b, rad_decl_op(b, "gdn_recurrent_tree", tp, 16,
                                            RAD_WEIGHTS(w_a_log, w_dt_bias, w_out_norm)),
                               {w.q, w.k, w.v, w.ab, w.in.x}, {w.o.x, tree_factors});
            if (!op_tree_conv || !op_tree_recur) return RAD_E_UNSUPPORTED;
        }
        op_tree_commit = rw(b, rad_decl_op(b, "gdn_tree_commit", tp, 16, RAD_NOWEIGHTS),
                            {tree_raw, tree_factors}, {});
        if (!op_tree_commit) return RAD_E_UNSUPPORTED;
    }
    RAD_ARCH_TRY(q_o.declare(b, g, w.o, v_dim));
    RAD_ARCH_TRY(out.declare(b, nm, g, nm.f("blk.%d.ssm_out", l), g.n_embd, v_dim,
                             RAD_SHARD_COL, grp_layer(l), { s.out }, w.o, w.h.x));

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

inline void GdnFP8::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* st = kv_batch(batch, kv_state);
    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* st_idx = st ? st->state_index : nullptr;
    /* HOW MANY SLOTS THE ROW NAMES. One is the state; a linear kernel serving speculation
     * asks for scratch beside it and the manager sizes the row to suit (RadKVGroupBatch).
     * The block passes the row through and does not read it. */
    const int64_t st_w = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;
    const int64_t H = cfg.n_head_v;
    /* The two halves of `w.in.x`: the convolution's [q|k|v] and the output gate behind it. */
    const int64_t conv_dim = cfg.conv_dim();
    const int64_t v_dim    = cfg.v_dim();

    if (!ext_in) nq_h.step(c, T);
    in.step(c, w.h, w.in.x, T);
    RAD_ISSUE_N(c, op_ab, T, brows(w.h.x, T), RAD_W(w_ab), brows(w.ab, T));

    /* THE TWO PATHS, EACH OVER ITS OWN RUN OF SEQUENCES.
     *
     * The step is sorted decode-first, so the decode rows are sequences [0, D) holding tokens
     * [0, DT) and the prefill chunks are sequences [D, S) holding the rest. Each path is issued
     * over its own half and a pure step issues exactly one of them.
     *
     * THE PER-SEQUENCE ARRAYS MOVE AND THE TOKEN OPERANDS DO NOT. Every kernel here takes
     * cu_seqlens and derives both its grid and its rows from it -- N is cu's own length minus one
     * and row zero of a sequence is the ABSOLUTE cu[n] -- so the prefill half is addressed by
     * handing it `cu + D` while leaving q, k, v and the rest at their base. Offsetting those too
     * would count the displacement twice. The decode half is the prefix, so it narrows instead:
     * its cu ends at DT and its token operands are cut to DT rows, which agree because they are
     * the same number.
     *
     * The two ops that take no cu -- the gated norm and the standalone quantiser -- are per TOKEN,
     * so they take a row slice and are the only place DT appears as an offset.
     *
     * What this buys is that a mixed step does not run the chunked path over a sequence that is
     * only decoding. That path commits the state directly and has no num_accepted operand, so a
     * draft verified on it could not be rolled back and the scheduler would have to refuse one in
     * every mixed step -- on an agentic load, a large share of all decode rows. */
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    /* And the band the decode half selects on. max_q_len over a mixed step is the prefill chunk;
     * over a pure decode step it is already the right number, so a batch that leaves
     * max_q_len_decode zero still selects correctly there. */
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    const int64_t P  = S - D;
    const int64_t PT = T - DT;
    const int64_t in_w = conv_dim + v_dim;      /* w.in.x's declared row width */

    if (D > 0 && cfg.tree_state && op_tree_i8g) {
        RAD_ISSUE_N(c, op_tree_i8g, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->cu_seqlens, RAD_I32, D + 1),
                    batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE,
                    praw(batch->ctx_lens, RAD_I32, D),
                    bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), bcol(w.in.x, conv_dim, v_dim, DT),
                    RAD_W(w_out_norm), brows(tree_raw, DT), brows(tree_factors, DT),
                    brows(w.o.q8, DT), brows(w.o.s8, DT));
    } else if (D > 0 && cfg.tree_state) {
        RAD_ISSUE_N(c, op_tree_conv, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->cu_seqlens, RAD_I32, D + 1),
                    batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE,
                    praw(batch->ctx_lens, RAD_I32, D),
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT), brows(tree_raw, DT));
        RAD_ISSUE_N(c, op_tree_recur, qd,
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT),
                    bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->cu_seqlens, RAD_I32, D + 1),
                    batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE,
                    bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(w_out_norm), brows(w.o.x, DT),
                    brows(tree_factors, DT));
        q_o.step(c, T, 0, DT);
    } else if (D > 0 && op_cr && qd <= cr_rows) {
        RAD_ISSUE_N(c, op_cr, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D),
                    bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(w_out_norm), brows(w.o.x, DT),
                    w.o.cq() ? brows(w.o.cq(), DT) : RAD_NONE,
                    w.o.cq() ? brows(w.o.cs(), DT) : RAD_NONE);
    } else if (D > 0) {
        RAD_ISSUE_N(c, op_conv_update, qd,
                    bcol(w.in.x, 0, conv_dim, DT), RAD_W(w_conv), RAD_NONE, kv_cache(kv_conv, layer),
                    praw(cv_idx, RAD_I32, D), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT));
        RAD_ISSUE_N(c, op_recur, qd,
                    brows(w.q, DT), brows(w.k, DT), brows(w.v, DT),
                    bcol(w.ab, 0, H, DT), bcol(w.ab, H, H, DT),
                    RAD_W(w_a_log), RAD_W(w_dt_bias), kv_cache(kv_state, layer),
                    praw2(st_idx, RAD_I32, D, st_w), praw(batch->num_accepted, RAD_I32, D),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    bcol(w.in.x, conv_dim, v_dim, DT), RAD_W(w_out_norm), brows(w.o.x, DT),
                    w.o.cq() ? brows(w.o.cq(), DT) : RAD_NONE,
                    w.o.cq() ? brows(w.o.cs(), DT) : RAD_NONE);
    }

    if (P > 0) {
        RAD_ISSUE(c, op_conv_prep,
                  bcol(w.in.x, 0, conv_dim, T), RAD_W(w_conv), RAD_NONE, RAD_W(w_a_log), RAD_W(w_dt_bias),
                  kv_cache(kv_conv, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  bcol(w.ab, 0, H, T), bcol(w.ab, H, H, T),
                  praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                  praw(batch->ctx_lens + D, RAD_I32, P),
                  brows(w.q, T), brows(w.k, T), brows(w.v, T),
                  brows(w.gdec, T), brows(w.beta, T));
        RAD_ISSUE(c, op_kkt,
                  brows(w.k, T), brows(w.beta, T), brows(w.gdec, T),
                  praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(w.kkt, T));
        RAD_ISSUE(c, op_scan,
                  brows(w.q, T), brows(w.k, T), brows(w.v, T), brows(w.kkt, T),
                  brows(w.gdec, T), brows(w.beta, T),
                  kv_cache(kv_state, layer), praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                  brows(w.o.x, T), kv_cache(kv_state, layer),
                  praw2(st_idx ? st_idx + D * st_w : nullptr, RAD_I32, P, st_w));
        /* PER TOKEN, so this one is a row slice and not a sequence sub-array. Over the decode
         * rows the recurrent kernel has already applied the gate in its own epilogue. */
        RAD_ISSUE_N(c, op_gnorm, PT,
                    brow_slice(w.o.x, DT, PT, v_dim),
                    bcol_at(w.in.x, DT, in_w, conv_dim, v_dim, PT),
                    RAD_W(w_out_norm), brow_slice(w.o.x, DT, PT, v_dim));
        /* PREFILL STILL QUANTISES SEPARATELY, and only prefill. The decode kernel folds the
         * quantiser into its own epilogue -- see the operand list on gdn_recurrent_update -- so
         * issuing this over those rows would overwrite the same bytes with the same values for a
         * launch a layer, in every delta-net layer, for nothing. The chunked
         * scan has no such epilogue: its workgroup is a chunk, not a row, and the group's amax is
         * not a reduction it can do. */
        q_o.step(c, T, DT, PT);
    }

    out.step(c, w.o, w.h.x, T);
    if (op_ar && !ar_taken(g, T, ar_out, ar_out_take))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

inline void GdnFP8::commit(RadCtx* c, const RadBatch* batch) const {
    if (!cfg.tree_state) return;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    if (!D) return;  /* prefill scan already materialized state and canonical raw history */
    const auto* st = kv_batch(batch, kv_state);
    const auto* cv = kv_batch(batch, kv_conv);
    const int64_t pitch = st && st->state_index_pitch > 0 ? st->state_index_pitch : 1;
    const int qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    // Commit batches carry the physical path pitch independently of the active node capacity.
    const int64_t path_pitch = batch->n_seq > 0 && batch->n_draft_out % batch->n_seq == 0
                                 ? batch->n_draft_out / batch->n_seq : 0;
    RAD_ISSUE_N(c, op_tree_commit, qd,
                kv_cache(kv_state, layer), praw2(st ? st->state_index : nullptr, RAD_I32, D, pitch),
                brows(tree_factors, DT), kv_cache(kv_conv, layer),
                praw(cv ? cv->state_index : nullptr, RAD_I32, D), brows(tree_raw, DT),
                praw2(batch->draft_out_ids, RAD_I32, D, path_pitch),
                praw(batch->num_accepted, RAD_I32, D), praw(batch->ctx_lens, RAD_I32, D),
                praw(batch->cu_seqlens, RAD_I32, D + 1),
                batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE);
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_GDN_FP8_H */
