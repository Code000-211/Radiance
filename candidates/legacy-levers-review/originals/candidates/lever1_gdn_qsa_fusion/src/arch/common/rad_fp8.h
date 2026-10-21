/* rad_fp8.h -- the two pieces every block-scaled FP8 block is built from.
 *
 * One plugin per (architecture, quantisation) is spec §2.4, and the file-count mitigation is that
 * the blocks are COMPOSITION (rad_arch.h). This is the composed piece the FP8 blocks share and the
 * bf16 ones have no use for: a linear layer whose weight is E4M3 with a block scale plane, and the
 * activation quantiser that feeds it.
 *
 * ============================== WHAT "BLOCK-SCALED FP8" IS ==============================
 *
 * The format is the checkpoint's, not this project's. Qwen3.8-27B-FP8 -- and every other
 * `quantization_config: {quant_method: fp8, weight_block_size: [128,128]}` release -- stores each
 * linear as
 *
 *   <linear>.weight             F8_E4M3, row-major [N][K]
 *   <linear>.weight_scale_inv   BF16,    row-major [ceil(N/128)][ceil(K/128)]
 *
 * and libr4d's gemm_fp8a16_nt_m1 / gemm_fp8a8_nt_m16 / gemm_fp8a8_tiled read exactly those bytes
 * (libr4d/r4d_fp8_layout.cpp verified the grid against the real file). So rad-convert COPIES:
 * no permutation, no repacking, no dequantisation, and a .rad costs what the checkpoint does. That
 * is the whole reason this plugin family exists -- widening a 27B FP8 checkpoint to bf16 doubles
 * the weight bytes, which does not fit the device memory this engine is built for.
 *
 * ============================== WHY THE SCALE PLANE IS A WEIGHT ==============================
 *
 * `gemm_nt_q`'s operand list is `a`, `a_scale`, `b`(w), `b_scale`(w) -> `y`, so the scale plane is
 * a first-class WEIGHT OPERAND and is declared here as its own rad_weight. It is not an `aux`
 * stream on the weight, and that is not a stylistic choice:
 *
 *   - RadLayout can declare an aux stream beside a weight, but RadArgs has NO WAY TO DELIVER one
 *     to a launch (libr4d/r4d_fp8_layout.cpp reports this). An aux scale would be unreachable
 *     at run time by construction.
 *   - a weight can be placed, moved and load-time SHARDED on its own. The engine slices a
 *     row-sharded [136,40] plane at dim 0 exactly as it slices the [17408,5120] weight, and an aux
 *     stream has no shard axis at all.
 *
 * The cost is one extra declared weight per linear -- 2.1 KiB of scales against 89 MiB of weight at
 * gate_proj's shape -- and one extra name-map entry, which is `<src>_scale_inv` and is derived
 * here rather than spelled at every call site.
 *
 * ============================== WHY THE ACTIVATIONS ARE FP8 TOO ==============================
 *
 * `dtype` is ONE string per declared op, so a linear is `fp8a16` or `fp8a8` and cannot be both.
 * fp8a16 reads a bf16 activation directly and needs no quantise launch, which is strictly better --
 * and its kernel is `M <= 1`. One row. That serves a single-sequence decode and nothing else: at
 * two concurrent sequences M is 2 and there is no kernel. So the shipped path is `fp8a8`, whose two
 * kernels between them cover every M (`nt_m16` to 16, `tiled` above), and the price is one
 * `quant_act_fp8` launch per activation that feeds a GEMM.
 *
 * PER ACTIVATION, NOT PER LINEAR. The three attention projections all read the post-norm hidden
 * state, and the GDN block's in and z projections both read it; quantising once and pointing every
 * linear at the same pair is the difference between one launch a layer and five. That is why
 * QuantFP8 is a separate struct from LinearFP8 rather than a step inside it.
 */
#ifndef RAD_FP8_H
#define RAD_FP8_H

#include "rad_arch.h"

#include <cstdlib>
#include <cstring>

namespace rad {
namespace arch {

/* The scale block, in both dimensions. A property of the CHECKPOINT (`weight_block_size`), not a
 * knob: libr4d compiles it in as a shift and its constraint rows carry C_EQ("group", 128), so a
 * container quantised at any other width resolves to no kernel and says so by name. Declared as a
 * named constant because three files have to agree about it. */
enum { RAD_FP8_BLOCK = 128 };

/* Rows of the scale plane for an N-row weight, and columns for a K-wide one. */
inline int64_t fp8_blocks(int64_t n) { return (n + RAD_FP8_BLOCK - 1) / RAD_FP8_BLOCK; }

/* An activation and its fp8 twin. The bf16 buffer is what the rest of the graph reads and writes;
 * the other two exist only between the quantiser and the GEMMs that consume them. */
struct ActFP8 {
    rad_buf x = 0;   /* [max_tok, n]                bf16 */
    rad_buf q = 0;   /* [max_tok, n]                e4m3 */
    rad_buf s = 0;   /* [max_tok, ceil(n/block)]    f32 */
    /* THE SAME ACTIVATION AT INT8 CODES, for an int8 linear (LinearFP8::i8): [max_tok, n] i8 and
     * its f32 scale per (row, block). Declared only where a model's trunk is int8 (declare_qs's
     * `i8`), and written either by the linear's own quant_act_i8g from `x` or, when `q8_fed`, by
     * the producer itself. */
    rad_buf q8 = 0;
    rad_buf s8 = 0;
    /* THE PRODUCER WRITES q8/s8 ITSELF -- a connection read (HyperConn::Config::codes_i8), a gated
     * quantiser, the attention's gate epilogue, the delta net's output epilogue -- in place of
     * the E4M3 pair, so an int8 linear reading this input declares no quantiser of its own. Set by
     * the model when it declares the pair, before the wires copy it. */
    bool    q8_fed = false;

    /* The code pair the producer writes: int8 when the model fed it, E4M3 otherwise. */
    rad_buf cq() const { return q8_fed ? q8 : q; }
    rad_buf cs() const { return q8_fed ? s8 : s; }

    /* Declare all three at once, since a caller that declared two of them is a caller with a bug.
     * `x` is passed in rather than made here: it is usually a buffer the block already had.
     *
     * `rows` IS NOT OPTIONAL FOR A BUFFER THAT IS NOT ONE ROW A TOKEN, and leaving it at the
     * default for one that is not is a silent arena overrun rather than a refusal. The default is
     * `g.max_tok` because that is what an activation is; a MoE expert plane is `rows * top_k` and
     * an encoder tower's is its own patch count. Getting it wrong is hard to see: a routed expert
     * pair declared at max_tok while the gated quantiser writes `rows * top_k` overruns onto a
     * buffer that may already be dead, so the model stays correct and only a later reader of that
     * buffer sees it. core/runtime/issue.cpp refuses the issue by name, which is the guard; this
     * is the honest declaration. */
    int declare_qs(RadBuilder* b, Names& nm, const Geom& g, const char* name, int64_t n,
                   int64_t rows = 0, bool i8 = false) {
        /* A bf16 model's linears read `x` and nothing else (Geom::w_bf16), so there are no codes
         * to declare: an arena plane nothing writes is a plane the planner still charges. */
        if (g.w_bf16) { q = s = q8 = s8 = 0; return RAD_OK; }
        const int64_t m = rows > 0 ? rows : g.max_tok;
        q = decl_b(b, nm.f("%s.q", name), RAD_F8E4M3, { m, n });
        s = decl_b(b, nm.f("%s.s", name), RAD_F32,    { m, fp8_blocks(n) });
        if (i8) {
            q8 = decl_b(b, nm.f("%s.q8", name), RAD_I8,  { m, n });
            s8 = decl_b(b, nm.f("%s.s8", name), RAD_F32, { m, fp8_blocks(n) });
            if (!q8 || !s8) return RAD_E_INVAL;
        }
        return (q && s) ? RAD_OK : RAD_E_INVAL;
    }
};

/* HOW MANY TOKEN ROWS THE fp8a16 MATVEC COVERS, which is the one number that decides both which
 * GEMM a linear issues and whether the quantiser in front of it runs at all.
 *
 * IT IS ONE ROW. The multi-row matvec reads the weight once for all M rows, so widening the band
 * looks free; it is not. The matvec is issue-bound above one row: the per-row work is 27
 * instructions per 16 weight bytes and eight rows of it do not fit in the issue budget, so
 * effective weight bandwidth falls as M rises on identical weight traffic
 * (r4d_gemm_fp8a16_nt_m1.hip has the ISA counts). The fp8a8 tile's cost is flat in M to 16, so it
 * wins from two rows up and the crossover is not near eight.
 *
 * SO THE SHIPPED PATH IS W8A8, which is also what the checkpoint is: `quant=fp8_e4m3` with
 * `activation_scheme: dynamic`, and prefill always quantises. A deployment that speculates
 * runs T = 1 + n_spec on every trunk step, so it is W8A8 end to end and decode computes the
 * same function prefill does.
 *
 * WHAT ONCE KEPT A ROW FOR THE MATVEC was the standalone `quant_act_fp8` in front of every linear
 * -- one launch per activation per layer -- which the fp8a8 path needs and this one does not. The
 * fusions below remove it: the producer that feeds a linear quantises in its own epilogue
 * (`rmsnorm_quant_fp8`, `gated_quant_fp8`, `gate_quant_fp8`), so the fp8a8 path pays no extra
 * launch and the last argument for a wider matvec band goes with it. */
/* ...AND IT IS DECLINED, because of where the fp8 weight is stored and what the matvec can read.
 *
 * The fp8 weight is stored in WMMA FRAGMENT ORDER (libr4d/r4d_fp8_frag.h): a (16 row, 64 k) tile
 * laid out as 32 lanes x 32 bytes, which is what lets the two fp8a8 GEMMs read a fragment straight
 * from a coalesced global load instead of transposing 16 rows through LDS. Both of them do, and
 * both are checked against the host reference.
 *
 * The matvec is built on the opposite premise -- "a lane reads 16 CONSECUTIVE k of one row", in
 * three separate read paths with their own RB blocking -- and 16 consecutive k of a row are not
 * contiguous in fragment order. Converting it is a rewrite of a tuned kernel, not an edit, so it
 * is not done here and the op is not declared rather than declared and wrong.
 *
 * WHAT THAT COSTS is the one-row band falling back to the fp8a8 tile, which serves M <= 16 and
 * therefore covers it. A deployment that speculates never takes this band at all: T is
 * 1 + n_spec on every trunk step. */
inline int64_t fp8_matvec_rows(const Geom& g) {
    (void)g;
    return 0;
}

/* THE FUSED ATTENTION PROLOGUE'S BAND, asked here rather than restated because the block that
 * issues the fused op, the op that fills the rope table, the QSA indexer that reads that table and
 * the step that decides whether to fill it all have to agree on it.
 *
 * The fusion is bit-exact against the unfused path: the kernel reproduces `r4d_rmsnorm_bf16`'s
 * sum-of-squares fold element by element rather than folding eight elements a lane, and
 * r4d_selftest asserts byte equality over 1.8M elements at both gain widths. A fold order that
 * merely agrees to a ULP is not enough here -- it moves the engine's transcript.
 *
 * IT STOPS AT THE DECODE BAND, because a fusion that replaces four launches with one pays where
 * the launch is the cost and nowhere else. At a 2048-token chunk it is slightly slower than the
 * four separate kernels: those are already wide enough to fill the device, and the fused form
 * does strictly more arithmetic -- a rotary lane re-derives its partner's normed value rather than
 * taking it from a register. 64 is the same decode band the narrow GEMM and the gated fold use,
 * and for the same reason. */
inline int64_t qk_fuse_rows(const Geom& g) {
    return g.max_tok < 64 ? g.max_tok : 64;
}

/* THE ROPE TABLE'S HEIGHT. `rope_table` writes each position at its OWN row, and the fused
 * prologue and the QSA indexer read it there, so the table needs a row for every position a step
 * can name -- and a speculative step names positions past the context bound. The last verify of a
 * request that fills its context feeds position max_ctx - 2 and one more row a draft token, and a
 * draft round rotates at positions ahead of the committed end as well: `max_spec` rows past
 * `max_ctx` cover both, and the one after them keeps the bound from resting on that arithmetic
 * being exact. The extra rows are (max_spec + 1) * rot floats. */
inline int64_t rope_table_rows(const Geom& g) {
    return g.max_ctx + (g.max_spec > 0 ? (int64_t)g.max_spec : 0) + 1;
}

/* THE SMALLEST TOKEN COUNT AT WHICH A QUANTISED ACTIVATION IS ACTUALLY READ.
 *
 * `LinearFP8::step` takes the matvec at T <= fp8_matvec_rows(), and the matvec does not read
 * `a.q` or `a.s` -- libr4d says so itself about that kernel: "the ONLY member of this family that
 * does not need r4d_quant_act_fp8". Left unguarded, the quantiser in front of it runs anyway, so a
 * single-sequence decode step issues 256 launches -- one per activation per layer -- to produce
 * bytes nobody reads. A decode step's budget IS launches.
 *
 * The buffer PLAN is unaffected. Liveness comes from the declared op list, not from what a step
 * issues, so `q` and `s` keep their arena slots and an op that does read them at a larger T still
 * finds them where it expects. */
inline int64_t fp8_quant_min_rows(const Geom& g) {
    return fp8_matvec_rows(g) + 1;
}

/* ================================================================== the activation quantiser */
struct QuantFP8 {
    ActFP8  a{};
    int64_t n  = 0;
    /* The smallest T at which some linear reads what this produces; see fp8_quant_min_rows. */
    int64_t min_rows = 1;
    rad_op  op = 0;

    int  declare(RadBuilder* b, const Geom& g, const ActFP8& act, int64_t width);
    /* `r0`/`rows` narrow it to a RANGE of rows, the same spelling every other helper here takes.
     * A mixed step needs it: only the prefill rows want the standalone quantiser, because the
     * recurrent path folds it into its own epilogue. */
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

inline int QuantFP8::declare(RadBuilder* b, const Geom& g, const ActFP8& act, int64_t width) {
    a = act;
    n = width;
    /* Nothing to quantise for a bf16 model: no op, and step() issues nothing. */
    if (g.w_bf16) { op = 0; return a.x ? RAD_OK : RAD_E_INVAL; }
    min_rows = fp8_quant_min_rows(g);
    if (!a.x || !a.cq() || !a.cs() || n <= 0) return RAD_E_INVAL;
    /* `dtype` here is the INPUT's, which is bf16 -- the output width is what the op name says.
     * libr4d's row constrains it to bf16 for that reason and ref reads each operand's own dtype.
     * An input the model feeds at int8 codes is quantised by quant_act_i8g into its int8 pair. */
    op = rw(b, RAD_OP(b, a.q8_fed ? "quant_act_i8g" : "quant_act_fp8",
               RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", n),
                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype)),
               RAD_NOWEIGHTS),
           { a.x }, { a.cq(), a.cs() });
    return RAD_OK;
}

inline void QuantFP8::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (!op) return;
    if (rows <= 0) rows = T - r0;
    if (rows < min_rows) return;
    RAD_ISSUE(c, op, brow_slice(a.x, r0, rows, n), brow_slice(a.cq(), r0, rows, n),
              brow_slice(a.cs(), r0, rows, n / RAD_FP8_BLOCK));
}

/* ================================== the quantiser folded into the op that PRODUCES the row
 *
 * QuantFP8 above is the standalone launch, and two thirds of them do not have to be. A decode
 * step issues ~300 quantisers across both ranks, and each is dominated by launch cost rather than
 * arithmetic: the work is 82 KiB of row. What it costs is a launch, a grid fill, and a second
 * pass over a row the previous kernel had in registers -- and for ~190 of the ~300 the previous
 * kernel is right there in the same block:
 *
 *     rmsnorm -> quant     the input of every attention, GDN and MLP linear
 *     silu_mul -> quant    down_proj's input
 *
 * libr4d's `rmsnorm_quant_fp8` and `gated_quant_fp8` are those two producers with the quantiser on
 * the same pass. They still write the bf16 output the unfused producer wrote -- the fp8a16 matvec
 * reads it at one row -- and they take the codes over the BF16-ROUNDED value, so the bytes are the
 * unfused pair's exactly and generated text does not move.
 *
 * AND THE RESIDUAL ADD FOLDS IN TOO, which is another ~130 launches. `add` is the cheapest kernel
 * in the model and is almost entirely launch cost: 82 KiB of row is negligible traffic beside it.
 * Every block in this architecture ends the same way -- out-projection into
 * the shared `h` buffer, all-reduce, `x += h` -- and every block BEGINS with a norm of `x`. So the
 * add moves from the end of one block to the front of the next, where a kernel is already reading
 * both buffers, and `fold` is what says a block has a predecessor to add.
 *
 * The kernel rounds the sum to bf16 and STORES it before normalising, which is not what the int8
 * sibling does: that one keeps the f32 sum, which is more accurate. This one reproduces the pair
 * it replaces to the byte, and byte-identical output across builds is the property this engine is
 * checked against.
 *
 * THE FOLDED FORM HAS NO min_rows. QuantFP8 declines to run below the matvec's band
 * because it would be producing bytes nobody reads; there is nothing to decline when the write is
 * two stores on a pass that happens anyway. */
/* WHETHER THE ALL-REDUCE AT `T` ROWS WILL GO OUT ON THE EXACT WIRE, which is the one question the
 * fused form below turns on, and it is answered here rather than guessed because the plugin's
 * `all_reduce` row answers it per call from the same three facts: an install that forbids a lossy
 * wire, a message the rotation's group of 64 does not tile, or a message under the floor. Get this
 * wrong in the optimistic direction and a step issues the fused kernel where the collective needed
 * the compressed payload; get it wrong the other way and the pair runs, which merely costs the
 * unfused launch. Both ranks evaluate it on the same T, so neither can take a different branch.
 *
 * The 6-bit wire pushes 2.6x fewer bytes than the exact one, and the fused kernel carries BOTH
 * payloads -- so the question below decides which wire the fused op is issued with rather than
 * whether it is issued at all, and `ar_fused_ok` is the separate question of whether a fused form
 * exists for that answer.
 *
 * WHY THE FLOOR IS A FLOOR AND NOT A SWITCH: the trade is the message size, and it changes sign
 * across the band. At a single sequence the message is small -- T * n_embd * 2 bytes with T on the
 * order of 8 -- and the rotation's own arithmetic costs more than the bytes it saves, so the exact
 * wire wins. From two concurrent sequences up the message is large enough that the compressed wire
 * wins, and the margin grows monotonically with it, so there is no second crossover inside the
 * band. M is C * (1 + n_spec), and 128 KiB sits between the two regimes.
 */
inline bool ar_wire_is_exact(const Geom& g, int64_t T) {
    if (g.world != 2) return false;
    const int64_t numel = T * g.n_embd;
    if (numel <= 0) return false;
    if (g.wire_exact) return true;
    if (numel % 64) return true;
    return numel * 2 < g.wire_min_bytes;   /* bf16 on the wire */
}

/* WHAT THE COMPRESSED FUSED FORM NEEDS ON TOP OF THE WIRE ITSELF: the fused kernel gives a block
 * whole ROWS, because a sum of squares closes only over one, and it packs the wire in chunks of
 * this many elements -- so a chunk must not straddle a row, or the two ranks would disagree about
 * which bytes a block owns. Stated here rather than asked of the plugin for the same reason the
 * rotation's group of 64 is stated above: the architecture has to decide, at declare time and
 * identically on both ranks, whether the op exists at all. */
static const int64_t kArWire6Chunk = 512;

/* Whether the all-reduce at `T` rows can be folded into the norm that follows it. The exact wire
 * always can; the compressed one can when a chunk tiles the row. Both ranks evaluate it on the
 * same T, so neither can take a different branch, and the block in FRONT asks the same question
 * before deciding to skip its own collective. */
inline bool ar_fused_ok(const Geom& g, int64_t T) {
    if (g.world != 2 || T * g.n_embd <= 0) return false;
    return ar_wire_is_exact(g, T) || (g.n_embd % kArWire6Chunk) == 0;
}

/* THE FOUR-RANK LOSSY WIRE'S TILE: tiered int8 in groups of 32, and a group may not straddle one
 * of the four shards, so a message whose element count is not a multiple of 128 goes out exact.
 * Stated here for the reason the rotation's group of 64 is stated in ar_wire_is_exact: the
 * architecture has to decide, identically on every rank, which wire a step's message takes. */
static const int64_t kArWideWireTile = 128;

/* THE MOST ROWS A DECODE OR SPECULATIVE-VERIFY STEP CARRIES: one a sequence, and 1 + max_spec a
 * sequence at a verify (max_spec is 0 with speculation off), never more than max_tok. It is a fact
 * about this engine's own steps, which is why the architecture holds it. The four-rank fused write
 * and its gather are declared over M up to this, and ar_wide_write_exact takes no step above it, so
 * a step sent to them is always inside the range they were resolved for. */
inline int64_t ar_wide_write_rows(const Geom& g) {
    const int64_t r = g.max_seqs * (1 + (g.max_spec > 0 ? g.max_spec : 0));
    return r < g.max_tok ? r : g.max_tok;
}

/* THE ROWS THE EXACT FUSED WRITE IS DECLARED FOR: every step at two ranks, the decode and verify
 * steps at four (ar_wide_write_rows), and none at any other width. */
inline int64_t ar_exact_write_rows(const Geom& g) {
    if (g.world == 2) return g.max_tok;
    if (g.world == 4) return ar_wide_write_rows(g);
    return 0;
}

/* THE FOUR-RANK FUSED WRITE'S STEPS. At four ranks ar_hc_write exists on the exact wire only, and
 * it is the exact ONE-SHOT with the write behind it. A step takes it where both of these hold:
 *
 *   rows      T is at most ar_wide_write_rows. The fused kernel is a one-shot, and the messages a
 *             one-shot is for are the decode and verify steps'. A larger step -- a prefill chunk --
 *             keeps the pair, and there the standalone `all_reduce`'s own size bands choose the
 *             one-shot or the two-shot: where that choice falls is the kernel library's to say,
 *             so the architecture holds no copy of it. The fused one-shot serves any message it was
 *             declared for with the same bytes, so this bound is a cost, not a correctness line.
 *   exact     --tp-wire exact, or a message the four-rank lossy row serves exactly itself: one
 *             kArWideWireTile does not tile, or one under `min_bytes`, both of which that row hands
 *             to the exact sum. ar_wire_is_exact's three facts, with that wire's tile.
 *
 * ar_wire_is_exact itself stays false off two ranks: the embedding's collective asks it whether
 * layer 0's fused norm absorbs that reduction, and at four no norm does. Every rank evaluates this
 * on the same T, so none can take a different branch. */
inline bool ar_wide_write_exact(const Geom& g, int64_t T) {
    if (g.world != 4 || T < 1 || T > ar_wide_write_rows(g)) return false;
    const int64_t numel = T * g.n_embd;
    if (numel <= 0) return false;
    if (g.wire_exact) return true;
    if (numel % kArWideWireTile) return true;
    return numel * 2 < g.wire_min_bytes;   /* bf16 on the wire */
}

/* WHETHER A GATED RESIDUAL'S EXACT FUSED WRITE TAKES THE ALL-REDUCE AT T ROWS: at two ranks on every
 * step whose wire is exact, at four where ar_wide_write_exact says so. The write
 * (HyperConn::ar_write) and the block in front (ar_taken with kArOutExact) both ask exactly this. */
inline bool ar_exact_write_takes(const Geom& g, int64_t T) {
    return g.world == 2 ? ar_wire_is_exact(g, T) : ar_wide_write_exact(g, T);
}

/* WHICH WIRES THE CONSUMER OF A BLOCK'S ALL-REDUCE TAKES. A block is declared without knowing what
 * follows it, so the architecture tells it:
 *
 *   kArOutNorm   the next block's norm, which takes either wire it has a fused form for
 *                (ar_fused_ok);
 *   kArOutExact  a gated residual's write with the exact fused form only, so a message that would
 *                go out compressed stays with the block (ar_exact_write_takes -- at four ranks
 *                also one of a step larger than a decode or verify step);
 *   kArOutBoth   a gated residual's write with a fused form on each wire, which takes every
 *                message (HyperConn::ar_take). */
enum ArOut : int {
    kArOutNorm  = 0,
    kArOutExact = 1,
    kArOutBoth  = 2,
};

/* WHETHER A BLOCK SKIPS ITS OWN ALL-REDUCE because its consumer takes it. The consumer asks the
 * same question of the same T, so the two cannot both skip it or both issue it. */
inline bool ar_taken(const Geom& g, int64_t T, bool out, int take) {
    if (!out) return false;
    if (take == kArOutBoth)  return true;
    if (take == kArOutExact) return ar_exact_write_takes(g, T);
    return ar_fused_ok(g, T);
}

/* WHETHER A MoE BLOCK'S GATHER RIDES IN FRONT OF THE REDUCTION THAT FOLLOWS IT (ar_gather_hc_write).
 * `rows` and `rows6` are the bounds the fused op was declared for on the exact and on the rotated
 * wire, 0 where it was not: at most the tokens one MoE pass carries, so a step it serves is one
 * pass whose gather is the whole step's. The wire the step's message goes out on picks the bound:
 * the exact one wherever the exact fused write takes the step (ar_exact_write_takes -- at two
 * ranks ar_wire_is_exact itself, at four the decode and verify steps on the exact wire, where
 * `rows6` is always 0).
 * The block that would issue the gather and the write that would absorb it ask this of the same T
 * and the same bounds, so neither can skip it while the other issues it. */
inline bool gather_taken(const Geom& g, int64_t T, int64_t rows, int64_t rows6) {
    const int64_t r = ar_exact_write_takes(g, T) ? rows : rows6;
    return r > 0 && T <= r;
}

/* WHAT A NORM MAY ABSORB FROM THE COLLECTIVE IN FRONT OF IT.
 *
 * `kArAny` is every layer boundary: the predecessor reduced a PARTIAL SUM, both ranks contributed
 * real values, and the compressed wire is the trade this model already makes there.
 *
 * `kArExact` is the EMBEDDING, and the difference is not stylistic. That collective sums DISJOINT
 * rows -- one rank gathered the token and the other contributed zeros -- so the compressed wire
 * takes a value that is already exactly right and degrades it, at the model's INPUT, before any
 * layer runs. It also disagrees with the DRAFTER, whose own embedding reduction is exact, and a
 * trunk and a drafter that see different embeddings reject each other's tokens.
 *
 * It is visible in the output, not just in principle: allowing the compressed wire here changes
 * the greedy text, because a prompt of a few dozen tokens crosses --tp-wire-min-kb where an 8-row
 * decode step does not. Exact-only keeps the decode fold -- which is the whole win -- and leaves
 * prefill on the standalone collective. */
enum ArFold : int {
    kArNone  = 0,
    kArAny   = 1,
    kArExact = 2,
};

struct NormQuantFP8 {
    ActFP8     a{};
    rad_weight w = 0;
    /* What the norm is taken of. Without a fold that is the residual stream itself; with one it is
     * the PREVIOUS block's delta, which this op adds into the stream first. */
    rad_buf    src = 0;
    rad_buf    res = 0;
    rad_op     op = 0;
    /* ...AND THE SAME OP WITH THE PRECEDING ALL-REDUCE FOLDED INTO IT. Declared only where the
     * predecessor has an all-reduce whose output this norm is the sole consumer of, which is
     * exactly where `residual` is non-null: `src` is then the delta buffer the collective wrote
     * and nothing between the two reads it. BOTH are declared and the choice is made at step time,
     * because the wire the collective picks depends on T and the fused kernel serves one of them.
     * The predecessor asks the same question and skips its own all_reduce on the same answer. */
    rad_op     op_arn = 0;
    /* ...and the same fused op again with the ROTATED 6-BIT payload, which is the one a served
     * step actually issues: the message is T * n_embd * 2 bytes and T is n_seq * (1 + n_spec), so
     * it crosses --tp-wire-min-kb at the second concurrent sequence and stays across it. Declared
     * separately rather than parameterised at step time because a wire is a declaration-time
     * parameter of the op, and the two are chosen between exactly as `op_arn` and `op` are. */
    rad_op     op_arn6 = 0;
    Geom       gm{};
    /* The row width, kept so a ROW SLICE can turn a row offset into an element offset. */
    int64_t    n = 0;

    int  declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x, const ActFP8& act,
                 int64_t n, rad_buf residual = 0, int64_t max_rows = 0, int fuse_ar = kArNone);
    void step(RadCtx* c, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
    /* Which of the three this step runs. The exact wire and the compressed one each have a fused
     * form; without one, the pair. */
    rad_op pick(int64_t T) const {
        if (op_arn && ar_wire_is_exact(gm, T)) return op_arn;
        if (op_arn6 && ar_fused_ok(gm, T)) return op_arn6;
        return op;
    }
    /* What the block in FRONT of this one asks before issuing its own collective. */
    bool takes_ar(int64_t T) const { return pick(T) != op; }
};

inline int NormQuantFP8::declare(RadBuilder* b, const Geom& g, rad_weight gain, rad_buf x,
                                 const ActFP8& act, int64_t n, rad_buf residual,
                                 int64_t max_rows, int fuse_ar) {
    a = act;
    w = gain;
    src = x;
    res = residual;
    gm = g;
    this->n = n;                    /* the parameter shadows the member; the member is the pitch */
    /* A bf16 model of this family takes every block input from its gated residual (hc_read) and
     * declares no block norm; a pre-norm model served bf16 is the bf16 blocks' (rad_block_*.h), not
     * these. Refused by name rather than declared as a quantiser nothing reads. */
    if (g.w_bf16) {
        fprintf(stderr, "radiance: a fused norm and quantiser was declared on a model whose "
                        "linears are bf16; its blocks take a prepared input instead\n");
        return RAD_E_UNSUPPORTED;
    }
    if (!src || !a.x || !a.q || !a.s || !w || n <= 0) return RAD_E_INVAL;
    op = RAD_OP(b, "rmsnorm_quant_fp8",
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_F64("eps", g.eps),
                           RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", g.dtype),
                           RAD_F64("wadd", g.wadd)),
                RAD_WEIGHTS(w));
    if (!op) return RAD_E_INVAL;
    /* Not rw()'s two initializer_lists, because the sets are conditional and a braced list in a
     * ternary has no lifetime extension. `residual` is read AND written, so it is in both: the
     * buffer planner reads these for liveness, and an INOUT that claimed only one of them is a
     * use-after-free waiting for a plan that reorders around it. */
    rad_buf rd[2] = { src, res };
    rad_buf wr[4] = { a.q, a.s, a.x, res };
    rad_op_reads (b, op, rd, res ? 2 : 1);
    rad_op_writes(b, op, wr, res ? 4 : 3);

    /* THE FUSED FORM'S EXTRA OUTPUT IS THE REDUCED MESSAGE IN `src`, and whether that is new
     * liveness depends on which form this is. A FOLDED norm's `src` IS `a.x` -- the same buffer,
     * already in the write set. An UNFOLDED one's `src` is the residual stream itself and is NOT,
     * so it is added: the buffer planner reads these sets, and a collective writing a buffer it is
     * not recorded as writing is a plan free to reorder a reader in front of it.
     *
     * `res` is NOT required. A fused collective mostly makes sense where the predecessor left a
     * delta, but the EMBEDDING is a predecessor too, and with a vocab-sharded table its gather is
     * partial and needs exactly this reduction. That case has no residual: `x` is the stream, not
     * a delta to add to one. The kernel compiles the arm (r4d_fused_quant_fp8.hip dispatches
     * `res ? true : false`) and the schema marks `residual` optional. */
    if (fuse_ar && g.world == 2) {
        op_arn = RAD_OP(b, "ar_rmsnorm_quant_fp8",
                        RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                   RAD_INT("n", n), RAD_F64("eps", g.eps),
                                   RAD_INT("group", RAD_FP8_BLOCK),
                                   RAD_INT("world_size", g.world),
                                   RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                        RAD_WEIGHTS(w));
        if (!op_arn) return RAD_E_INVAL;
        rad_buf wr_ar[4] = { a.q, a.s, a.x, res ? res : src };
        rad_op_reads (b, op_arn, rd, res ? 2 : 1);
        rad_op_writes(b, op_arn, wr_ar, 4);

        /* Declared only where the compressed wire can actually be reached and served: an install
         * that forbids a lossy payload never picks it, and a width the wire's chunk does not tile
         * would be a step-time shape refusal -- which in this engine is fatal -- rather than a
         * choice not taken. */
        if (fuse_ar != kArExact && !g.wire_exact && (n % kArWire6Chunk) == 0) {
            op_arn6 = RAD_OP(b, "ar_rmsnorm_quant_fp8",
                             RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                                        RAD_INT("n", n), RAD_F64("eps", g.eps),
                                        RAD_INT("group", RAD_FP8_BLOCK),
                                        RAD_INT("world_size", g.world),
                                        RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd),
                                        RAD_INT("wire", 6)),
                             RAD_WEIGHTS(w));
            if (!op_arn6) return RAD_E_INVAL;
            rad_op_reads (b, op_arn6, rd, res ? 2 : 1);
            rad_op_writes(b, op_arn6, wr_ar, 4);
        }
    }
    return RAD_OK;
}

/* `r0` and `rows` are the ROW SLICE this issue covers, and they default to the whole step. At
 * offset 0 `brow_slice` builds exactly the operand `brows` did -- same rows, offset zero -- so the
 * unsliced path is unchanged by construction rather than by inspection.
 *
 * The slice is what lets a block run its rows in passes -- the routed feed-forward's, whose
 * expert planes are one pass tall. Its correctness rests on the band rule in core/runtime/ctx.h:
 * every slice must land in the same bucket-table band as the whole, or the op resolves to a
 * different kernel with a different summation order. */
inline void NormQuantFP8::step(RadCtx* c, int64_t T, int64_t r0, int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    RAD_ISSUE_N(c, pick(rows), rows, brow_slice(src, r0, rows, n),
                res ? brow_slice(res, r0, rows, n) : RAD_NONE, RAD_W(w),
                brow_slice(a.q, r0, rows, n),
                brow_slice(a.s, r0, rows, n / RAD_FP8_BLOCK),
                brow_slice(a.x, r0, rows, n));
}

/* `act` is the vocabulary's name for the gate and it is a NAME: "silu" is down_proj's input,
 * "sigmoid" is a gated attention o_proj's. */
struct GateQuantFP8 {
    ActFP8 a{};
    rad_op op = 0;
    /* WHETHER THE bf16 PRODUCT IS WRITTEN AT ALL. `out_bf16` is not an optimisation in general --
     * docs/OPS.md says a layer with a bf16 consumer of the fused value must ask for it or the
     * fusion is a different function -- but a block-fp8 down_proj reads the E4M3 codes and their
     * scale, and only a down_proj that reads the bf16 plane instead (LinearFP8::reads_x) reads the
     * bf16 silu product. It is [T, n_ff] a layer: 18.2 GB of a 16k prefill and 71 MB of an
     * 8-sequence decode step, written for one reader that is not there. Declared out of the write
     * list rather than merely passed absent, so the plan says what actually happens. */
    bool   want_x = true;
    /* The row width. `gate_up` is 2n wide, the quantised output n; see NormQuantFP8::n. */
    int64_t n = 0;
    /* `silu_mul` into the bf16 plane alone, for a model whose linears are bf16 (Geom::w_bf16). */
    bool   bf16 = false;

    int  declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActFP8& act, int64_t n,
                 const char* act_name = "silu", int64_t max_rows = 0, bool bf16_out = true,
                 bool rot = false);
    void step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0 = 0, int64_t rows = 0) const;
};

/* `rot` PICKS A DIFFERENT OP, not a parameter on the same one: the codes are not the same codes,
 * so the consumer is not free either -- only a weight stored with the matching rotation reads
 * them. It exists for the routed MoE arm's down projection at four bits and nothing else, and
 * `out_bf16` still receives the UNROTATED product in both cases. */
inline int GateQuantFP8::declare(RadBuilder* b, const Geom& g, rad_buf gate_up, const ActFP8& act,
                                 int64_t n, const char* act_name, int64_t max_rows,
                                 bool bf16_out, bool rot) {
    a = act;
    want_x = bf16_out;
    this->n = n;
    /* A bf16 model's down projection reads the product itself, so the producer is `silu_mul` and
     * writes the bf16 plane alone. A rotation has no meaning without the codes it rotates. */
    bf16 = g.w_bf16;
    if (bf16) {
        if (!gate_up || !a.x || n <= 0 || rot || std::strcmp(act_name, "silu") != 0) {
            fprintf(stderr, "radiance: a bf16 gate is silu over [gate|up] into a bf16 plane; "
                            "asked for '%s'%s\n", act_name, rot ? " rotated" : "");
            return RAD_E_INVAL;
        }
        want_x = true;
        op = rw(b, RAD_OP(b, "silu_mul",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               { gate_up }, { a.x });
        return op ? RAD_OK : RAD_E_INVAL;
    }
    if (!gate_up || (want_x && !a.x) || !a.cq() || !a.cs() || n <= 0) return RAD_E_INVAL;
    /* The rotated codes are a four-bit expert's operand, which reads E4M3 only. */
    if (rot && a.q8_fed) return RAD_E_INVAL;
    const char* opname = rot ? "gated_had_quant_fp8" : "gated_quant_fp8";
    op = want_x
        ? rw(b, RAD_OP(b, opname,
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_INT("group", RAD_FP8_BLOCK),
                           RAD_STR("dtype", g.dtype), RAD_STR("act", act_name)),
                RAD_NOWEIGHTS),
            { gate_up }, { a.cq(), a.cs(), a.x })
        : rw(b, RAD_OP(b, opname,
                RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                           RAD_INT("n", n), RAD_INT("group", RAD_FP8_BLOCK),
                           RAD_STR("dtype", g.dtype), RAD_STR("act", act_name)),
                RAD_NOWEIGHTS),
            { gate_up }, { a.cq(), a.cs() });
    return op ? RAD_OK : RAD_E_INVAL;
}

inline void GateQuantFP8::step(RadCtx* c, rad_buf gate_up, int64_t T, int64_t r0,
                               int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n), brow_slice(a.x, r0, rows, n));
        return;
    }
    RAD_ISSUE_N(c, op, rows, brow_slice(gate_up, r0, rows, 2 * n),
                brow_slice(a.cq(), r0, rows, n),
                brow_slice(a.cs(), r0, rows, n / RAD_FP8_BLOCK),
                want_x ? brow_slice(a.x, r0, rows, n) : RAD_NONE);
}

/* ================================================================== one block-scaled linear */
struct LinearFP8 {
    rad_weight w = 0, ws = 0;
    rad_op     op = 0;
    int64_t    N = 0, K = 0;
    /* A PLAIN bf16 WEIGHT read by `gemm_nt` over the activation's bf16 plane, in a model whose
     * linears are bf16 (Geom::w_bf16): one weight and no scale plane, and no quantiser in front. */
    bool       bf16 = false;
    /* AN INT8 WEIGHT, read by `gemm_nt_q` at dtype i8a8 against an INT8 activation: i8 codes and
     * a bf16 scale per row per 128 of K (libquant `codes=i8 group=128 scale=bf16`), the activation
     * i8 with an f32 scale per (row, 128) -- the E4M3 activation's grid at int8 codes, in the same
     * two buffers (`a.q`, `a.s`). The weight carries a quarter of E4M3's error at the same bytes and
     * the iu8 WMMA runs at the fp8 one's rate. `op_q` writes those codes from the bf16 plane
     * (quant_act_i8g) when the producer in front writes E4M3; it is declared and issued before the
     * GEMM. Read off the weight's encoding like `bf16`. */
    bool       i8 = false;
    rad_op     op_q = 0;
    /* What probe() read: the weight is int8. Lets a model learn its trunk's format from one linear
     * before it declares the activation buffers that format decides (ActFP8::declare_qs's `i8`). */
    bool       probed_i8 = false;
    /* THE DECODE MATVEC, DECLARED BESIDE THE GEMM. libr4d has gemm_fp8a16_nt_m1 -- an M <= 1
     * kernel that reads the bf16 activation directly, with no quantise and no tiling -- and it is
     * written for exactly the shape a single-sequence decode step issues. Declaring only fp8a8
     * leaves every decode on the TILED kernel at M = 1, which is the shape it is worst at.
     *
     * `dtype` is one string per declared op, so an op cannot be both: this is two ops over the
     * same weights, banded so their bucket tables do not overlap, and step() picks by the token
     * count. That is the same shape as the delta net's prefill/decode pair and for the same
     * reason -- it is a choice between two DECLARED sequences, not a branch inside one. */
    rad_op     op_m1 = 0;
    /* The widest T that op_m1 serves. 0 when the matvec is declined. */
    int64_t    mv_rows = 0;
    /* WHETHER THIS LINEAR READS THE INPUT'S bf16 PLANE, known before its producer is declared.
     * In a quantised model a producer writes the bf16 plane only when something reads it -- a
     * gated quantiser writes the codes alone -- so the block asks probe() first and has the
     * producer write the plane when this is set. Without that a plain weight reads arena bytes
     * nothing wrote: finite nonsense on the first step, NaN once they hold one. */
    bool       reads_x = false;
    bool       mapped  = false;   /* probe() made the name map; declare() does not make it again */

    /* The name map, and from the encoding behind it whether the weight is plain bf16. `base` and
     * `src` are what declare() is then given. */
    int  probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
               std::initializer_list<const char*> src);

    /* `base` is the DECLARED name without a suffix ("blk.7.ffn_gate_up"); the logical weight is
     * `<base>.weight` and its two views are `.weight` (the codes) and `.scale`. `src` is the
     * checkpoint side, one name or several to concatenate along dim 0 -- and a block-fp8
     * checkpoint's scale planes join with them, which is what makes a fusion of two fp8 linears
     * expressible at all: [17408,5120]+[17408,5120] concatenates to [34816,5120] and
     * [136,40]+[136,40] to [272,40], and 34816/128 is 272. A fusion whose parts are not whole
     * multiples of the block would not line up, and none here are. */
    /* `span_lo`/`span_hi`, when hi is set, is this rank's uneven slice of the split dimension in
     * the model's units -- rows of each row part for RAD_SHARD_ROW, columns for RAD_SHARD_COL --
     * and n_out / n_in is already its width; see rad_weight_shard_span. Whole scale blocks. */
    int  declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                 int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                 std::initializer_list<const char*> src,
                 const ActFP8& a, rad_buf y, int64_t max_rows = 0,
                 std::initializer_list<int64_t> row_parts = {},
                 int64_t span_lo = 0, int64_t span_hi = 0);
    void step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0 = 0,
              int64_t rows = 0) const;
};

/* THE NAME MAP OF A LINEAR, DECLARED BEFORE ITS WEIGHT: it is how the checkpoint -- or the recipe
 * rad-convert applies -- is asked what the weight is (rad_weight_encoding), so it has to exist
 * first. One source is copied; several are a convert-time fusion (gate and up into gate_up),
 * concatenated along dim 0 -- and a block-fp8 checkpoint's scale planes join with them, which is
 * why a fused part has to be a whole number of 128-row blocks. */
inline int map_linear(RadBuilder* b, const char* wn, std::initializer_list<const char*> src) {
    if (src.size() == 1) return map_copy(b, wn, *src.begin());
    RadNameMap mw = map_base(wn, RAD_MAP_CONCAT);
    int i = 0;
    for (const char* q : src) {
        if (i >= 8) return RAD_E_INVAL;
        mw.src[i++] = q;
    }
    mw.n_src = i;
    mw.concat_dim = 0;
    return rad_decl_name_map(b, &mw);
}

/* The int8 weight LinearFP8 serves at i8a8: i8 codes one a value, a bf16 scale per [1 x 128], no
 * transform and no other plane -- what libr4d's i8 block layout hook accepts. */
inline bool linear_is_w8(const RadEncoding& e) {
    if (!rad_enc_is(&e, "affine") || e.n_planes != 2 || e.transform[0]) return false;
    const RadEncPlane* sc = rad_enc_plane(&e, "scale");
    return e.plane[0].dtype == RAD_I8 && e.plane[0].block[1] == 1 && sc && sc->dtype == RAD_BF16 &&
           sc->block[0] == 1 && sc->block[1] == RAD_FP8_BLOCK;
}

inline int LinearFP8::probe(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                            std::initializer_list<const char*> src) {
    if (src.size() == 0) return RAD_E_INVAL;
    const char* wn = nm.f("%s.weight", base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    mapped = true;
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;
    probed_i8 = known && linear_is_w8(enc);
    reads_x = g.w_bf16 || plain || probed_i8;
    return RAD_OK;
}

inline int LinearFP8::declare(RadBuilder* b, Names& nm, const Geom& g, const char* base,
                              int64_t n_out, int64_t n_in, int shard, RadWeightGroup grp,
                              std::initializer_list<const char*> src,
                              const ActFP8& a, rad_buf y, int64_t max_rows,
                              std::initializer_list<int64_t> row_parts,
                              int64_t span_lo, int64_t span_hi) {
    N = n_out;
    K = n_in;
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;

    /* THE NAME MAP FIRST: it is how the checkpoint -- or the recipe rad-convert applies -- is
     * asked what the weight is, and that answer picks the branch below. */
    const char* wn = nm.f("%s.weight", base);
    if (!mapped) RAD_ARCH_TRY(map_linear(b, wn, src));
    RadEncoding enc{};
    const bool known = weight_enc(b, wn, &enc);
    const bool plain = known && rad_enc_is(&enc, "plain") && enc.n_planes == 1 &&
                       enc.plane[0].dtype == RAD_BF16;

    /* ---- A bf16 WEIGHT, `gemm_nt` over `a.x`: every linear of a bf16 model, and one a quantised
     * container keeps at the checkpoint's precision -- the QSA indexer's projection, whose top-k is
     * a hard choice between blocks and costs 20 MB to keep exact. The row parts and an uneven
     * slice apply to it as they would to the codes; there is no block to respect. */
    bf16 = g.w_bf16 || plain;
    if (bf16) {
        if (known && !plain) {
            char en[256];
            rad_enc_format(&enc, en, sizeof en);
            fprintf(stderr, "radiance: '%s' is %s, and this model's linears are plain bf16 -- a "
                            "bf16 container is served bf16 throughout\n", wn, en);
            return RAD_E_DTYPE;
        }
        /* The input's bf16 plane is what a bf16 weight multiplies. A quantised model's producer
         * may write only the codes -- a gated quantiser does -- and then there is nothing here to
         * read; the weight has to be quantised with the rest. */
        if (!a.x) {
            fprintf(stderr, "radiance: '%s' is plain bf16 and its input has no bf16 plane in this "
                            "model; quantise it with its neighbours\n", wn);
            return RAD_E_DTYPE;
        }
        int64_t parts[RAD_MAX_ROW_PARTS] = {0};
        int     np = 0;
        for (int64_t p : row_parts) {
            if (np >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
            parts[np++] = p;
        }
        w = decl_w_n(b, wn, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp, 0, parts, np);
        if (!w) return RAD_E_INVAL;
        if (span_hi > 0) RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        op = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, max_rows > 0 ? max_rows : g.max_tok),
                              RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w)),
               { a.x }, { y });
        return op ? RAD_OK : RAD_E_INVAL;
    }

    /* ---- AN INT8 WEIGHT, `gemm_nt_q` at i8a8 over the input's int8 pair `a.q8` / `a.s8`. The
     * scale plane is one bf16 per row per 128 of K, so it takes the weight's row parts and row
     * slice as they are, and a column slice in groups of 128 -- which the per-rank K being a
     * whole number of groups guarantees. The codes are the producer's own when the model feeds
     * the pair (ActFP8::q8_fed); otherwise op_q writes them from `a.x`, so the producer in front
     * has to write the bf16 plane (probe() set reads_x for it). Either way the model has to have
     * declared the pair (ActFP8::declare_qs's `i8`). */
    i8 = known && linear_is_w8(enc);
    if (i8) {
        if (!a.x || !a.q8 || !a.s8) {
            fprintf(stderr, "radiance: '%s' is an int8 weight, quantised from its input's bf16 "
                            "plane into its int8 code pair, and its input lacks one of them -- the "
                            "model did not learn its trunk is int8 before declaring it\n", wn);
            return RAD_E_DTYPE;
        }
        if (N % 16 || K % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an int8 weight takes rows in "
                            "16s and K in whole %d-column scale groups.\n",
                    base, (long long)N, (long long)K, RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        int64_t parts[RAD_MAX_ROW_PARTS] = {0};
        int     np = 0;
        for (int64_t p : row_parts) {
            if (np >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
            parts[np++] = p;
        }
        const char* sn = nm.f("%s.scale", base);
        w  = decl_view(b, wn, nullptr, "codes", RAD_I8, { N, K }, RAD_ACCESS_PER_TOKEN, shard, grp,
                       parts, np);
        ws = decl_view(b, sn, wn, "scale", RAD_BF16, { N, K / RAD_FP8_BLOCK }, RAD_ACCESS_PER_TOKEN,
                       shard, grp, parts, np);
        if (!w || !ws) return RAD_E_INVAL;
        if (span_hi > 0) {
            const bool col = shard == RAD_SHARD_COL;
            if (col && (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK)) {
                fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a "
                                "%d-column scale group.\n", base, (long long)span_lo,
                        (long long)span_hi, RAD_FP8_BLOCK);
                return RAD_E_INVAL;
            }
            RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
            RAD_ARCH_TRY(rad_weight_shard_span(b, ws, col ? span_lo / RAD_FP8_BLOCK : span_lo,
                                               col ? span_hi / RAD_FP8_BLOCK : span_hi));
        }
        const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
        if (!a.q8_fed)
            op_q = rw(b, RAD_OP(b, "quant_act_i8g",
                     RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("n", K),
                                RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "bf16")),
                     RAD_NOWEIGHTS),
                  { a.x }, { a.q8, a.s8 });
        op = rw(b, RAD_OP(b, "gemm_nt_q",
                   RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                              RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "i8a8")),
                   RAD_WEIGHTS(w, ws)),
               { a.q8, a.s8 }, { y });
        return (op_q || a.q8_fed) && op ? RAD_OK : RAD_E_INVAL;
    }

    /* An E4M3 weight reads the E4M3 pair, and an input the model feeds at int8 codes has none --
     * a container whose trunk mixes the two formats, which the model's one probe did not see. */
    if (a.q8_fed) {
        fprintf(stderr, "radiance: '%s' is not an int8 weight, and its input is fed int8 codes "
                        "because this model's trunk is int8; quantise it with its neighbours\n", wn);
        return RAD_E_DTYPE;
    }

    /* The one geometric precondition, checked here so it is reported once per weight with the
     * weight named rather than as a kernel miss with a band in it. These are PER-RANK extents, so
     * this is also the check that a tensor-parallel width does not cut a scale block in half. */
    if (N <= 0 || K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an fp8 weight requires "
                        "K in whole %d-column scale groups.\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }

    /* THE SCALE PLANE SPLITS WHERE THE WEIGHT DOES, in units of RAD_FP8_BLOCK rows. A fused
     * [q|k|v] whose parts are 1024/1024/3072 rows per rank is 8/8/24 scale rows per rank, and the
     * scale would otherwise be sliced contiguously while the weight it scales was not -- a
     * mismatch that survives every shape check and shows up only as wrong numbers. A part that is
     * not a whole number of blocks cannot be expressed either way, so it is refused by name. */
    int64_t wparts[RAD_MAX_ROW_PARTS] = {0}, sparts[RAD_MAX_ROW_PARTS] = {0};
    int     np = 0;
    int64_t accum = 0;
    for (int64_t p : row_parts) {
        if (np >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
        if (p <= 0) return RAD_E_INVAL;
        wparts[np] = p;
        int64_t b0 = fp8_blocks(accum);
        accum += p;
        int64_t b1 = fp8_blocks(accum);
        sparts[np] = b1 - b0;
        ++np;
    }

    const char* sn = nm.f("%s.scale",  base);
    /* ONE STORED WEIGHT, TWO OPERANDS: the codes and the scale plane of `<base>.weight`, which is
     * block-fp8 in the checkpoint or made so by the recipe. A checkpoint that is not (a dense
     * bf16 release) declares the same two views; what makes the planes exist is rad-convert. */
    w  = decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, { N, K },
                   RAD_ACCESS_PER_TOKEN, shard, grp, wparts, np);
    ws = decl_view(b, sn, wn, "scale", RAD_BF16, { fp8_blocks(N), fp8_blocks(K) },
                   RAD_ACCESS_PER_TOKEN, shard, grp, sparts, np);
    if (!w || !ws) return RAD_E_INVAL;
    /* AN UNEVEN SLICE, and the scale plane takes the same slice in blocks. */
    if (span_hi > 0) {
        if (span_lo % RAD_FP8_BLOCK || span_hi % RAD_FP8_BLOCK) {
            fprintf(stderr, "radiance: '%s' is sliced [%lld, %lld) on this rank, which cuts a %d-"
                            "wide scale block.\n", base, (long long)span_lo, (long long)span_hi,
                    RAD_FP8_BLOCK);
            return RAD_E_INVAL;
        }
        RAD_ARCH_TRY(rad_weight_shard_span(b, w, span_lo, span_hi));
        RAD_ARCH_TRY(rad_weight_shard_span(b, ws, span_lo / RAD_FP8_BLOCK,
                                           span_hi / RAD_FP8_BLOCK));
    }

    /* `M` is ranged over the token count, which is what selects between libr4d's narrow (M<=16)
     * and tiled instantiations. A caller with a tighter bound -- the lm_head, which runs only on
     * the sampled rows -- passes it, because a band ceiling of max_tok would size the narrow
     * kernel's split-K scratch for a step that cannot happen. */
    const int64_t hi = max_rows > 0 ? max_rows : g.max_tok;
    /* The matvec first, up to the width its kernel serves, so a VERIFY STEP still takes it.
     * fp8_matvec_rows is that width and says how it is chosen; at 0 the matvec is declined and
     * every T goes to the fp8a8 GEMM below. */
    mv_rows = fp8_matvec_rows(g);
    if (mv_rows > 0)
        op_m1 = rw(b, RAD_OP(b, "gemm_nt_q",
                      RAD_PARAMS(RAD_RANGE("M", 1, mv_rows), RAD_INT("N", N), RAD_INT("K", K),
                                 RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a16")),
                      RAD_WEIGHTS(w, ws)),
                  { a.x }, { y });
    op = rw(b, RAD_OP(b, "gemm_nt_q",
               RAD_PARAMS(RAD_RANGE("M", 1, hi), RAD_INT("N", N), RAD_INT("K", K),
                          RAD_INT("group", RAD_FP8_BLOCK), RAD_STR("dtype", "fp8a8")),
               RAD_WEIGHTS(w, ws)),
           { a.q, a.s }, { y });
    return RAD_OK;
}

/* THE TWO TRAILING OPTIONAL OPERANDS ARE PASSED, as RAD_NONE. `gemm_nt_q`'s schema is seven long
 * -- `a_sum` and `b_ref` sit after `y` for the asymmetric integer and mxfp4 grids -- and operands
 * are POSITIONAL, so the runtime takes an operand count that disagrees with the schema as a miscount
 * rather than as an omission (core/runtime/issue.cpp). An absent optional operand is a null tensor
 * in the list, not a shorter list; an fp8 grid is symmetric and needs neither. */
inline void LinearFP8::step(RadCtx* c, const ActFP8& a, rad_buf y, int64_t T, int64_t r0,
                            int64_t rows) const {
    if (rows <= 0) rows = T - r0;
    if (bf16) {
        RAD_ISSUE_N(c, op, rows, brow_slice(a.x, r0, rows, K), RAD_W(w), brow_slice(y, r0, rows, N));
        return;
    }
    if (i8) {
        if (op_q)
            RAD_ISSUE_N(c, op_q, rows, brow_slice(a.x, r0, rows, K), brow_slice(a.q8, r0, rows, K),
                        brow_slice(a.s8, r0, rows, K / RAD_FP8_BLOCK));
        RAD_ISSUE_N(c, op, rows, brow_slice(a.q8, r0, rows, K),
                    brow_slice(a.s8, r0, rows, K / RAD_FP8_BLOCK),
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
        return;
    }
    if (op_m1 && rows <= mv_rows)
        RAD_ISSUE_N(c, op_m1, rows, brow_slice(a.x, r0, rows, K), RAD_NONE,
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
    else
        RAD_ISSUE_N(c, op, rows, brow_slice(a.q, r0, rows, K),
                    brow_slice(a.s, r0, rows, K / RAD_FP8_BLOCK),
                    RAD_W(w), RAD_W(ws), brow_slice(y, r0, rows, N), RAD_NONE, RAD_NONE);
}


/* ================================================== an fp8 weight with no op of its own
 *
 * LinearFP8's two views without the GEMM: a weight a block declares here and reads through an op
 * of its own -- the DFlash2 and DSpark drafters' projections, which ship bf16 and are made block
 * fp8 by the recipe, since they are read whole on every decode step and only have to be good
 * enough to PROPOSE (every draft is verified against the trunk exactly).
 *
 * BOTH VIEWS ARE DECLARED AT WHAT THEY SELECT -- [N][K] E4M3 codes, [N/128][K/128] BF16 scales --
 * because that is the shape the runtime hands a kernel, and libr4d checks the scale plane's
 * extents against ceil(N/128) x ceil(K/128) before it will run. The row parts follow it: the
 * weight's are rows and the plane's are BLOCKS of 128 rows, the same division LinearFP8 makes. A
 * part that is not a whole number of blocks cannot be sharded and is refused by name.
 */
struct DenseFP8 {
    rad_weight w = 0, s = 0;

    int declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                int access, int shard, RadWeightGroup grp,
                std::initializer_list<const char*> src,
                std::initializer_list<int64_t> row_parts = {});
};

inline int DenseFP8::declare(RadBuilder* b, Names& nm, const char* base, int64_t N, int64_t K,
                             int access, int shard, RadWeightGroup grp,
                             std::initializer_list<const char*> src,
                             std::initializer_list<int64_t> row_parts) {
    if (N <= 0 || K <= 0 || src.size() == 0) return RAD_E_INVAL;
    if (K % RAD_FP8_BLOCK) {
        fprintf(stderr, "radiance: '%s' is %lldx%lld per rank; an fp8 weight requires "
                        "K in whole %d-column scale groups.\n",
                base, (long long)N, (long long)K, RAD_FP8_BLOCK);
        return RAD_E_INVAL;
    }

    int64_t wp[RAD_MAX_ROW_PARTS] = {0}, sp[RAD_MAX_ROW_PARTS] = {0};
    int     np = 0;
    int64_t accum = 0;
    for (int64_t p : row_parts) {
        if (np >= RAD_MAX_ROW_PARTS) return RAD_E_INVAL;
        if (p <= 0) return RAD_E_INVAL;
        wp[np] = p;
        int64_t b0 = fp8_blocks(accum);
        accum += p;
        int64_t b1 = fp8_blocks(accum);
        sp[np] = b1 - b0;
        ++np;
    }

    const char* wn = nm.f("%s.weight", base);
    const char* sn = nm.f("%s.scale",  base);
    RAD_ARCH_TRY(map_linear(b, wn, src));
    w = decl_view(b, wn, nullptr, "codes", RAD_F8E4M3, { N, K }, access, shard, grp, wp, np);
    s = decl_view(b, sn, wn, "scale", RAD_BF16, { fp8_blocks(N), fp8_blocks(K) }, access, shard,
                  grp, sp, np);
    return w && s ? RAD_OK : RAD_E_INVAL;
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_FP8_H */
