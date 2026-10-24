/* rad_qsa.h -- Qwen4-Exp's QSA indexer: the part of a full-attention layer that decides WHICH
 * blocks of key history that layer's attention reads.
 *
 * docs/QSA.md is the decomposition and the argument; this is the declaration side of it. The
 * kernels are libr4d's `qsa_block_key`, `qsa_score` and `qsa_select`, and the sparse attention is
 * the ORDINARY `attn_paged` issued with the selection in place of the sequence's own block table
 * -- both of those are plain operands, so nothing in core/ learns what QSA is.
 *
 * WHAT IS HERE is the whole indexer: the projection and the query path, the block-key path with
 * its two states, and the score and selection the sparse attention is issued with. Below
 * `budget + ratio - 1` tokens of context the selection is EVERY visible block, so the sparse path
 * must reproduce the dense path's text byte for byte -- which is the oracle any change to this
 * file is checked against.
 */
#ifndef RAD_QSA_H
#define RAD_QSA_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

struct QsaIndexer {
    struct Config {
        int64_t     heads    = 0;   /* indexer_n_heads      -- 4 */
        int64_t     head_dim = 0;   /* indexer_head_dim     -- 128 */
        int64_t     ratio    = 0;   /* indexer_compress_ratio -- 4 tokens a block */
        int64_t     budget   = 0;   /* indexer_budget       -- 2048 tokens, so 512 blocks */
        int64_t     rot      = 0;   /* the LAYER's rotary width, which the indexer shares */
        const char* mode     = "neox";
        /* Checkpoint names. Absent means this layer has no indexer and declare() is a no-op. */
        const char* qk_proj  = nullptr;
        const char* q_norm   = nullptr;
        const char* k_norm   = nullptr;
        /* Derived, and all four are the caller's because they size things it declares. */
        int64_t     topk       = 0;   /* budget / ratio -- blocks the selection keeps */
        int64_t     max_blocks = 0;   /* max_ctx / ratio -- the widest block table */
        int64_t     max_work   = 0;   /* ceil(max_tok / ratio) + max_seqs */
        /* THE TAIL'S RING LENGTH, `ratio + max_spec`, and the `max_spec` is the whole of it.
         *
         * A tail of exactly `ratio` rows indexed by `pos % ratio` is right when a step's tokens
         * are the tokens that happened. Under speculation they are not: a verify step runs
         * 1 + n_spec positions and the next step RESTARTS at the first rejected one, so positions
         * the tail has already overwritten are needed again. Concretely at ratio 4, depth 3,
         * positions 9..12 -- the store writes slots 1,2,3,0 and position 12 lands on the slot
         * position 8 is still owed to. Accept one and the next step rebuilds block 2 (tokens
         * 8..11) out of a slot holding token 12. That block key is then wrong in the cache for
         * the rest of the sequence, and it happens on any step that straddles a block boundary
         * and does not accept everything -- which is most of them.
         *
         * The deepest rollback is to the first draft, so the widest window a step can still owe
         * is [p0 - ratio + 1, p0 + n_spec]: ratio + n_spec positions, each needing its own
         * residue. Under `ratio + max_spec` a slot is only ever overwritten by a position that
         * far ahead, which is past the end of the step that could ask for it. */
        int64_t     ring       = 0;   /* ratio + max_spec */
        /* THE ATTENTION GROUP, because the SELECTION's output is what the attention reads and so
         * it is that group's page ids that come out. The block-key group is this block's own and
         * is declared on first use -- see declare(). */
        rad_kvgroup kv_attn    = 0;
        /* THE TAIL'S GROUP: one RAD_KV_LINEAR state a sequence a layer, `ring` rows of
         * `head_dim`. It is a KV group and not a buffer because a buffer has no way to name a
         * SEQUENCE -- see the note on Wire. */
        rad_kvgroup kv_tail    = 0;
        /* DOES ANYTHING READ THE SELECTION? False on the DENSE path, where the attention is
         * issued with the sequence's own block table and `sel` is written for nobody. The block-key
         * half still runs -- it is STATEFUL, and a history built without it is a history the sparse
         * path could not later read -- but the score and the selection are then dead work, and on
         * this model they are the largest op in the indexer. Declared either way, so the graph and
         * the buffer plan do not depend on it. */
        bool        emit_sel   = true;
        /* DOES ANYTHING READ THE SELECTION ON *THIS* STEP? Two questions, and this one is the
         * cheaper half of the first. `emit_sel` is fixed at declare: a layer with no selection
         * wired to its attention never selects at all. But a wired layer still serves most of its
         * steps on the DENSE path -- decode below the exactness bound and every prefill chunk
         * there -- and on those steps the attention reads the sequence's own block table and
         * `sel`, `nsel`, `score` and `sequ` are written for nobody.
         *
         * RADIANCE_QSA_SKIP_DENSE_SELECTION=1 skips the score, the select and the two gathers on
         * exactly the steps the attention's own `sparse` predicate rejects. READ ONCE AT DECLARE
         * and carried here, because a per-step getenv would put a string comparison in the
         * dispatch loop and, worse, would let two ranks read two different answers.
         *
         * THE PREDICATE IS THE ATTENTION'S, NOT A NEAR MISS. rad_block_attn_gated_fp8.h issues
         * the selection only when
         *
         *     !frontier && sel && sequ && (T == n_seq || exact_to <= 0 || reach > exact_to)
         *
         * with `reach` = max_ctx_len + max_q_len. Below that line the selection is dead. The
         * block-key half is NOT skipped with it: it is STATEFUL, and a history built without it
         * is a history the sparse path could not later read. Neither is the tail store, the work
         * list or the projection. */
        bool        skip_dense_sel = false;
        int64_t exact_to() const { return topk * ratio + ratio - 1; }
    };

    /* The two planes the front half produces. `qk` is the projection whole -- `heads` query heads
     * then ONE raw key head -- and `q` is the query heads normed and rotated, contiguous.
     *
     * THE RAW KEY IS NEVER NORMED OR ROTATED HERE. It is pooled over `ratio` tokens FIRST and the
     * norm and the rotation happen to the pooled result, at the position of the block's first
     * token (`qsa_block_key`). Reading the reference quickly gives the opposite impression,
     * because the q path right above it does both. */
    struct Wire {
        rad_buf qk = 0;   /* [max_tok, (heads + 1) * head_dim] */
        rad_buf q  = 0;   /* [max_tok, heads * head_dim] */
        /* THE TAIL IS NOT IN THIS WIRE AND CANNOT BE. Every other buffer here is within-step
         * scratch, which is why one set serves the whole model; the tail carries an unfinished
         * block's RAW INDEX KEYS ACROSS STEPS, and that makes it state with two axes a buffer
         * cannot express.
         *
         *   THE LAYER. The keys are `index_qk_proj`'s output and that weight is PER LAYER, so a
         *   shared tail has layer L pooling layer L-1's keys -- at decode three of every four
         *   keys in a completing block come from the wrong layer. Every layer needs its own,
         *   including the MTP head's.
         *
         *   THE SEQUENCE. A per-layer buffer indexed by the BATCH ROW is no better across
         *   requests: the row is step-local -- the scheduler builds the step from `running_` in
         *   priority order and then stable-sorts decode first -- so a sequence that outlives an
         *   earlier one shifts down a row and inherits its keys. A buffer has no per-sequence
         *   axis to index instead; a KV group does, and that is the one the engine keeps stable
         *   for the life of a sequence, zeroes when it hands a slot out, and checkpoints with
         *   the rest of the linear state when the prefix cache saves one.
         *
         * So the tail is `Config::kv_tail`, a RAD_KV_LINEAR group, and declare() binds this
         * layer to it. Nothing about it belongs to a wire shared by every layer. */
        rad_buf stage = 0;   /* [max_work * ratio, head_dim] */
        rad_buf page  = 0;   /* [max_work]      i32 */
        rad_buf bpos  = 0;   /* [max_work] i32, or [max_work, 3] under M-RoPE */
        rad_buf nc    = 0;   /* [max_seqs]      i32 */
        /* The selection, ONE ROW A QUERY and not one a sequence. At decode the two are the same
         * thing; at prefill a chunk's queries share one history and each picks its own set out of
         * it, which is the whole of Stage B (docs/QSA.md). `sel` and `seqused` are what the sparse
         * attention is issued with in place of the sequence's own block table and length.
         *
         * `score` IS THE BIG ONE: max_tok * max_ctx/ratio floats, 134 MB at a 2048-token chunk and
         * 64K of context. It is one buffer for the whole model rather than one a layer, and
         * declare() refuses a geometry whose product is unreasonable rather than discovering it as
         * an allocation failure. Tiling the query axis would cap it, and the reason it is not done
         * is that `cu` is absolute over the batch: a tile would need its rows rebased, which means
         * a per-token sequence id operand -- qsa_work is where it would come from. */
        rad_buf score = 0;   /* [max_tok, max_blocks] f32 */
        rad_buf sel   = 0;   /* [max_tok, topk + 1]   i32 */
        rad_buf nsel  = 0;   /* [max_tok]             i32 */
        rad_buf sequ  = 0;   /* [max_tok]             i32 */
        rad_buf sel_shard  = 0;   /* [max_tok / 2, topk + 1] i32 */
        rad_buf sequ_shard = 0;   /* [max_tok / 2]           i32 */
        /* THE MODEL'S cos/sin PLANE, [rope_table_rows, rot], or 0 if it builds none. It is the
         * attention block's -- `rope_table` fills it once a step for every layer -- and the
         * indexer's own rotation is at the same `rot`, `theta` and `scale`, so the rows are
         * already there. A block handed no table asks for no fusion and runs the per-head norms
         * below, which is what keeps a plugin that does not build the table on the unfused
         * path. */
        rad_buf cos_sin = 0;
    };

    Geom       g{};
    Config     c{};
    Wire       w{};
    int        layer = 0;
    rad_weight  w_qn = 0, w_kn = 0;
    LinearFP8   proj{};
    rad_kvgroup kv_bk = 0;
    rad_op      op_norm = 0, op_rope = 0, op_qprep = 0;
    /* The most rows `op_qprep` may be issued over, 0 when it is not declared. It is the model's
     * own `qk_fuse_rows` and it is NOT a tuning: the cos/sin table is filled only for steps at or
     * under it, so above it the rows this op would read were never written. */
    int64_t     qprep_rows = 0;
    rad_op      op_work = 0, op_bkey = 0, op_tail = 0, op_score = 0, op_select = 0;
    rad_op      op_tree_commit = 0;
    rad_op      op_score_shard = 0, op_select_shard = 0;
    rad_op      op_gather_sel = 0, op_gather_sequ = 0;

    bool on() const { return op_rope != 0; }
    bool defer_decode(const RadBatch* batch) const {
        return c.tree_state && batch->spec_parent &&
            static_cast<int64_t>(batch->max_ctx_len) + batch->max_q_len <= c.exact_to();
    }

    /* `bk` is the block-key KV group, created on FIRST USE and shared by every QSA layer after
     * that -- which is why it is an in/out parameter rather than a member. It cannot be declared
     * with the model's other groups: `Builder::find_kv_consumer` binds a group to the first op
     * declared AFTER it that carries a `block_size`, and between a model-level declaration and
     * this block's `qsa_block_key` there is a whole layer of other ops. Declared immediately
     * before its own consumer, the binding cannot be anything else. */
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, const ActFP8& in, rad_kvgroup* bk);
    void step(RadCtx* c, const ActFP8& in, const RadBatch* batch) const;
    void commit(RadCtx* c, const RadBatch* batch) const;
};

inline int QsaIndexer::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                               const Config& cfg, const Wire& wire, const ActFP8& in,
                               rad_kvgroup* bk) {
    g = geom; c = cfg; w = wire; layer = l;
    if (!c.qk_proj || !c.q_norm) return RAD_OK;          /* no indexer on this layer */
    if (c.heads <= 0 || c.head_dim <= 0 || c.ratio <= 0 || c.budget <= 0) return RAD_E_INVAL;
    if (!w.qk || !w.q) return RAD_E_INVAL;
    /* The rotation is the LAYER's and it has to fit inside the index head, which the checkpoint's
     * own validator also requires -- so a container that violated it is refused here by name
     * rather than rotating past the end of a head. */
    if (c.rot < 0 || c.rot > c.head_dim || (c.rot & 1)) return RAD_E_SHAPE;

    const int64_t nw = (c.heads + 1) * c.head_dim;       /* the projection width */
    if (c.tree_state) {
        w.qk = decl_b(b, nm.f("blk.%d.qsa_tree_qk", l), RAD_BF16,
                      {g.max_tok, nw}, RAD_BUF_PERSIST);
        if (!w.qk) return RAD_E_INVAL;
    }

    RAD_ARCH_TRY(proj.declare(b, nm, g, nm.f("blk.%d.qsa_qk", l), nw, g.n_embd,
                              RAD_SHARD_NONE, grp_layer(l), { c.qk_proj }, in, w.qk));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.qsa_q_norm.weight", l), c.q_norm));
    w_qn = decl_w(b, nm.f("blk.%d.qsa_q_norm.weight", l), RAD_F32, { c.head_dim },
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_qn) return RAD_E_INVAL;

    /* ONE DECLARED NORM, ISSUED ONCE A HEAD, and the reason is the projection's layout rather
     * than a preference. `qk` is `heads + 1` heads wide, so the query heads are 128 apart WITHIN
     * a token and 640 apart ACROSS tokens -- which is not a stride any rank-2 view can express,
     * and the ABI narrows dim 0 or the last dimension and nothing between. A column slice of ONE
     * head is expressible, so the norm runs `heads` times and writes into a contiguous `q` that
     * the rotation can then take whole.
     *
     * Four extra launches a layer, and for a 128-wide row the launch IS the cost: the norms and
     * the ropes across the model's indexer layers are a real slice of a decode step against
     * microseconds of arithmetic. `op_qprep` is the fused form that replaces them.
     *
     * THE FUSED FORM IS `qk_norm_rope_gate`'S K SIDE, not a new op. That kernel already reads a
     * plain [tokens, heads, head_dim] input at the operand's OWN row pitch and writes the heads
     * contiguously -- which is exactly this projection's query half inside its `heads + 1` width
     * -- and it already rounds the norm to bf16 before rotating, because that is what a norm that
     * stores a tensor and a rope that reads it back do. The only thing its q side adds is the
     * [q|gate] split, and there is no gate here, so `n_head` 0 asks for the half that fits and
     * passes no q operands at all. Five launches a layer become one.
     *
     * IT NEEDS THE MODEL'S cos/sin TABLE and asks for nothing if it has none. The rotation is at
     * the same `rot`, `theta` and `scale` as the attention block's, so `rope_table` has already
     * written the rows this step's positions name -- and a plugin that builds no table leaves
     * `w.cos_sin` at 0 and runs the per-head norms below unchanged.
     *
     * AND ONLY UP TO `qk_fuse_rows`, WHICH IS A CORRECTNESS BOUND AND NOT A TUNING. The model
     * issues `rope_table` only for steps at or under that row count -- the fusion pays where the
     * launch is the cost, and a prefill chunk's kernels already fill the card -- so ABOVE it the
     * table rows this op would read were never written THIS step. It would rotate by another
     * step's angles: fluent text with a subtly wrong sense of position, which is the failure
     * `rope_table`'s own note calls out and which nothing downstream would flag. The bound is the
     * attention block's, off the same function, so the two cannot drift apart. */
    const bool qprep_neox = !std::strcmp(c.mode, "neox") || !std::strcmp(c.mode, "mrope") ||
                            g.rope_mc;
    qprep_rows = qk_fuse_rows(g);
    if (w.cos_sin && qprep_neox && qprep_rows > 0 && (c.head_dim % 32) == 0)
        op_qprep = rw(b, RAD_OP(b, "qk_norm_rope_gate",
                         RAD_PARAMS(RAD_RANGE("M", 1, qprep_rows),
                                    RAD_INT("head_dim", c.head_dim), RAD_INT("n_head", 0),
                                    RAD_INT("n_head_kv", c.heads), RAD_INT("rot", c.rot),
                                    RAD_STR("q_dtype", g.dtype), RAD_F64("eps", g.eps),
                                    RAD_F64("wadd", g.wadd_qk), RAD_INT("cs_f32", 1)),
                         RAD_WEIGHTS(w_qn)),
                     {w.qk, w.cos_sin}, {w.q});

    op_norm = rw(b, RAD_OP(b, "rmsnorm",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_F64("eps", g.eps), RAD_F64("wadd", g.wadd_qk),
                               RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_qn)),
                {w.qk}, {w.q});
    op_rope = rw(b, decl_rope(b, g, g.max_tok, c.head_dim, c.heads, 0, c.rot, c.mode),
                {w.q}, {w.q});

    /* ---- the block-key path. Nothing below reads anything above except `w.qk`'s raw-key half. */
    if (!c.k_norm || !w.stage || !c.topk || !c.max_blocks || !c.max_work)
        return RAD_E_INVAL;
    if (!c.kv_tail || c.ring < c.ratio) return RAD_E_INVAL;
    /* THIS LAYER'S TAIL STATE. The group is the caller's -- it holds one state per sequence per
     * BOUND LAYER, so binding is how this layer gets its own, and it is the same call the
     * block-key group needs for the same reason. */
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, c.kv_tail));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.qsa_k_norm.weight", l), c.k_norm));
    w_kn = decl_w(b, nm.f("blk.%d.qsa_k_norm.weight", l), RAD_F32, { c.head_dim },
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_kn) return RAD_E_INVAL;

    op_work = rw(b, RAD_OP(b, "qsa_work",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_INT("ratio", c.ratio), RAD_INT("ring", c.ring),
                               RAD_INT("seqs", g.max_seqs),
                               RAD_INT("work", c.max_work), RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.qk}, {w.stage, w.page, w.bpos, w.nc});

    /* THE GROUP IS DECLARED HERE, immediately before the op that sizes it -- see declare()'s
     * note. `kv_heads` 1, `block_size` the compress ratio and `head_dim` n/(2*ratio) make a page
     * hold exactly `n` contiguous values, which is one block key, and make the manager allocate
     * exactly one page per `ratio` tokens. The geometry is a container, not an attention shape. */
    if (*bk == 0) {
        if ((c.head_dim % (2 * c.ratio)) != 0) return RAD_E_SHAPE;
        RadKVGroupDecl d{};
        d.kind      = RAD_KV_FULL;
        d.dtype     = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim  = c.head_dim / (2 * c.ratio);
        *bk = rad_decl_kv_group(b, nm.f("kv_qsa_bkey"), &d);
        if (!*bk) return RAD_E_INVAL;
    }
    kv_bk = *bk;
    /* UNDER M-RoPE a block key rotates at its first token's three components, which qsa_work
     * writes into a [work, 3] `bpos` when it is handed the step's rotary planes. */
    op_bkey = g.rope_mc
        ? RAD_OP(b, "qsa_block_key",
                 RAD_PARAMS(RAD_RANGE("M", 1, c.max_work), RAD_INT("n", c.head_dim),
                            RAD_INT("ratio", c.ratio), RAD_INT("rotary_dim", c.rot),
                            RAD_INT("block_size", c.ratio),
                            RAD_F64("eps", g.eps), RAD_F64("theta", g.theta),
                            RAD_F64("scale", g.rope_scale), RAD_F64("wadd", g.wadd_qk),
                            RAD_STR("dtype", g.dtype), RAD_STR("mode", g.rope_mode),
                            RAD_STR("sections", g.rope_sections)),
                 RAD_WEIGHTS(w_kn))
        : RAD_OP(b, "qsa_block_key",
                 RAD_PARAMS(RAD_RANGE("M", 1, c.max_work), RAD_INT("n", c.head_dim),
                            RAD_INT("ratio", c.ratio), RAD_INT("rotary_dim", c.rot),
                            RAD_INT("block_size", c.ratio),
                            RAD_F64("eps", g.eps), RAD_F64("theta", g.theta),
                            RAD_F64("scale", g.rope_scale), RAD_F64("wadd", g.wadd_qk),
                            RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(w_kn));
    op_bkey = rw(b, op_bkey, {w.stage, w.page, w.bpos}, {});
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv_bk));

    op_tail = rw(b, RAD_OP(b, "qsa_tail_store",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                               RAD_INT("ratio", c.ratio), RAD_INT("ring", c.ring),
                               RAD_INT("seqs", g.max_seqs),
                               RAD_STR("dtype", g.dtype)),
                    RAD_NOWEIGHTS),
                {w.qk}, {});

    /* Winner replay owns one sequence per workgroup and stages its ring in LDS.
     * Larger rings/heads retain the existing replay. M-RoPE geometry alone does
     * not require component positions: commit without rotary planes uses NeoX. */
    if (c.tree_state && c.head_dim <= 256 && c.ratio <= 32 &&
        c.ring <= 32 && g.max_seqs <= 256)
        op_tree_commit = rw(b, RAD_OP(b, "qsa_tree_commit",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_seqs),
                                           RAD_INT("n", c.head_dim), RAD_INT("ratio", c.ratio),
                                           RAD_INT("ring", c.ring), RAD_INT("rotary_dim", c.rot),
                                           RAD_INT("block_size", c.ratio),
                                           RAD_INT("max_tokens", g.max_tok),
                                           RAD_F64("eps", g.eps), RAD_F64("theta", g.theta),
                                           RAD_F64("scale", g.rope_scale),
                                           RAD_F64("wadd", g.wadd_qk)),
                                RAD_WEIGHTS(w_kn)),
                            {w.qk}, {});

    /* M IS QUERIES ON BOTH, up to a whole prefill chunk. A band stopping at max_seqs would be
     * right only for a per-SEQUENCE selection; this one selects per QUERY. */
    op_score = rw(b, RAD_OP(b, "qsa_score",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.head_dim),
                                RAD_INT("heads", c.heads), RAD_INT("blocks", c.max_blocks),
                                RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype)),
                     RAD_NOWEIGHTS),
                 {w.q, w.nc}, {w.score});
    op_select = rw(b, RAD_OP(b, "qsa_select",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok),
                                 RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                 RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs)),
                      RAD_NOWEIGHTS),
                  {w.score, w.nc}, {w.sel, w.nsel, w.sequ});

    if (g.world == 2 && w.sel_shard && w.sequ_shard) {
        op_score_shard = rw(b, RAD_OP(b, "qsa_score",
                                RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok / 2), RAD_INT("n", c.head_dim),
                                           RAD_INT("heads", c.heads), RAD_INT("blocks", c.max_blocks),
                                           RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype),
                                           RAD_INT("tp_partition", 1)),
                                RAD_NOWEIGHTS),
                            {w.q, w.nc}, {w.score});
        op_select_shard = rw(b, RAD_OP(b, "qsa_select",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok / 2),
                                            RAD_INT("blocks", c.max_blocks), RAD_INT("topk", c.topk),
                                            RAD_INT("ratio", c.ratio), RAD_INT("seqs", g.max_seqs),
                                            RAD_INT("tp_partition", 1)),
                                 RAD_NOWEIGHTS),
                             {w.score, w.nc}, {w.sel_shard, w.nsel, w.sequ_shard});
        op_gather_sel = rw(b, RAD_OP(b, "all_gather",
                               RAD_PARAMS(RAD_INT("world_size", 2),
                                          RAD_RANGE("numel", c.topk + 1, (g.max_tok / 2) * (c.topk + 1)),
                                          RAD_STR("dtype", "i32")),
                               RAD_NOWEIGHTS),
                           {w.sel_shard}, {w.sel});
        op_gather_sequ = rw(b, RAD_OP(b, "all_gather",
                                RAD_PARAMS(RAD_INT("world_size", 2),
                                           RAD_RANGE("numel", 1, g.max_tok / 2),
                                           RAD_STR("dtype", "i32")),
                                RAD_NOWEIGHTS),
                            {w.sequ_shard}, {w.sequ});
    }
    return RAD_OK;
}

inline void QsaIndexer::step(RadCtx* c_, const ActFP8& in, const RadBatch* batch) const {
    if (!on()) return;
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const int64_t qw = c.heads * c.head_dim;
    const int64_t nw = (c.heads + 1) * c.head_dim;
    const int64_t koff = qw;                     /* the raw-key head, after the query heads */

    proj.step(c_, in, w.qk, T);
    /* Dense-exact verification needs no indexer selection. Leave both stateful
     * caches immutable: siblings cannot share a linear tail or a pooled block.
     * The host later replays only the accepted key path into those caches. */
    const bool defer = defer_decode(batch);
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t first = defer ? D : 0, NS = S - first;
    if (!NS) return;
    /* ONE LAUNCH OR `heads` + 1. The fused form takes the projection WHOLE -- the kernel picks
     * head `h` out of it at the operand's own row pitch, so the strided view the unfused norm
     * needs is not taken at all -- and it writes the query heads contiguous and rotated, which is
     * what the pair below leaves behind. `n_head` 0 is the k-only form: no q_gate, no q_w, no
     * q_out, and the ninth operand (`gate_out`) is absent as it is everywhere else.
     *
     * `T <= qprep_rows` IS THE TABLE'S BOUND and declare() says why it is correctness. */
    if (op_qprep && T <= qprep_rows && !rope_mixed(batch)) {
        RAD_ISSUE_N(c_, op_qprep, T,
                    RAD_NONE, brows(w.qk, T), RAD_B(w.cos_sin), rope_pos1(batch, T),
                    RAD_NONE, RAD_W(w_qn),
                    RAD_NONE, brows(w.q, T), RAD_NONE);
    } else {
        for (int64_t h = 0; h < c.heads; ++h)
            RAD_ISSUE_N(c_, op_norm, T,
                        bcol_at(w.qk, 0, nw, h * c.head_dim, c.head_dim, T), RAD_W(w_qn),
                        bcol_at(w.q,  0, qw, h * c.head_dim, c.head_dim, T));
        RAD_ISSUE_N(c_, op_rope, T, brows(w.q, T), rope_posmc(g, batch, T));
    }

    const RadKVGroupBatch* bkb = kv_batch(batch, kv_bk);
    const RadKVGroupBatch* atb = kv_batch(batch, c.kv_attn);
    const RadKVGroupBatch* tlb = kv_batch(batch, c.kv_tail);
    if (!bkb || !atb || !tlb) return;
    /* COLUMN 0 OF THE STATE ROW IS THE COMMITTED SLOT. The row is wider than one whenever the
     * deployment speculates -- the manager gives every linear group a second copy then, because
     * the kernel that needed one asked for it -- and this op wants the committed slot in every
     * case. The pitch is the batch's, not a constant. */
    const int64_t tw = tlb->state_index_pitch > 0 ? tlb->state_index_pitch : 1;

    /* THE WORK LIST BEFORE THE TAIL STORE, and the order is load-bearing rather than tidy: the
     * gather reads the raw keys of the block that is completing, and up to `ratio - 1` of them
     * are in the tail from earlier steps -- which the store is about to overwrite with this
     * step's. Reversed, every block completing at decode would pool three of this step's keys.
     *
     * The ring makes the store's overwrite land `ring` positions later rather than `ratio`, so
     * the two do not collide within a step either -- but the order still decides what a
     * straddling block reads, and it is the cheaper of the two guarantees to keep. */
    const int64_t work = (T + c.ratio - 1) / c.ratio + NS;
    /* THE BLOCK POSITIONS: one component a block, or -- under M-RoPE with the planes -- three.
     * Under M-RoPE `bpos` is declared [max_work, 3], a block a row, so a smaller step is fewer
     * rows and the arena's sizing declare can shrink it; this step hands [work, 3], or its first
     * column alone when the pass has no planes -- qsa_work then writes component 0 and the block
     * key reads the one-component form. */
    const bool mc = g.rope_mc && batch->rope_pos;
    RadOperand bpos = g.rope_mc ? RAD_B_COL(w.bpos, 0, mc ? 3 : 1) : brows(w.bpos, work);
    if (g.rope_mc) bpos.rows = work;
    RAD_ISSUE_N(c_, op_work, T,
                bcol_at(w.qk, 0, nw, koff, c.head_dim, T),
                praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens + first, RAD_I32, NS + 1),
                kv_cache(c.kv_tail, layer),
                praw2(tlb->state_index + first * tw, RAD_I32, NS, tw),
                praw2(bkb->block_table + first * bkb->block_table_pitch, RAD_I32, NS, bkb->block_table_pitch),
                mc ? praw2(batch->rope_pos, RAD_I32, 3, T) : RAD_NONE,
                brows(w.stage, work * c.ratio), brows(w.page, work), bpos,
                brows(w.nc, NS));
    RAD_ISSUE_N(c_, op_bkey, work,
                brows(w.stage, work * c.ratio), RAD_W(w_kn),
                brows(w.page, work), bpos,
                kv_cache(kv_bk, layer));
    RAD_ISSUE_N(c_, op_tail, T,
                bcol_at(w.qk, 0, nw, koff, c.head_dim, T),
                praw(batch->positions, RAD_I32, T),
                praw(batch->cu_seqlens + first, RAD_I32, NS + 1),
                kv_cache(c.kv_tail, layer),
                praw2(tlb->state_index + first * tw, RAD_I32, NS, tw));
    if (defer) return; /* prefill subrange is materialized; decode waits for winner replay */

    /* SCORE AND SELECT OVER EVERY QUERY IN THE STEP. A selection does not have to be per-SEQUENCE
     * even though a block table is: both ops take `cu`, the block table stays the sequence's, and
     * `m` is a query row. So a prefill chunk's queries each get their own selection out of the one
     * history they share, which is what the checkpoint does, and there is no cap on the context a
     * step may index. */
    if (!c.emit_sel) return;

    /* HOW MUCH HISTORY THERE ACTUALLY IS. The score plane is declared for `max_ctx`, so scoring
     * and selecting to its width means a 2000-token prompt walks 50000 block slots -- and both
     * kernels size their GRID from the declared width, because the live count is device data.
     * Narrowing the operand is how the host tells them: the pitch stays the declaration's, only
     * the column count moves, which is the same trick the attention block plays with `reach`.
     *
     * IT IS A CEILING AND EVERY TERM IS SLACK, because an UNDER-estimate silently drops blocks
     * from the selection and nothing would say so. A query's complete-block count is at most
     * (position + 1) / ratio; the highest position this step can reach is bounded by the
     * host-side context bound plus every token in the step plus the draft window; and the +1
     * keeps `n < nb`, which is what makes the partial page -- `bt[n]` -- readable at all. */
    const int64_t reach = (int64_t)batch->max_ctx_len + batch->n_tok + batch->n_spec + c.ratio;
    int64_t live = reach / c.ratio + 1;
    if (live > c.max_blocks) live = c.max_blocks;
    if (live < 1) live = 1;

    /* Each rank owns ceil(T / 2) rows. Odd batches gather one padded row into
     * the existing max_tok capacity; downstream attention still consumes T rows. */
    const int64_t Th = (T + 1) / 2;
    const bool do_part = (g.world == 2) && (T >= 4) && (2 * Th <= g.max_tok) &&
                         op_score_shard && op_select_shard && op_gather_sel && op_gather_sequ;
    if (do_part) {
        const int rank = rad_rank(c_);
        const int64_t m_off = (rank == 1) ? Th : 0;
        RAD_ISSUE_N(c_, op_score_shard, Th,
                    brow_slice(w.q, m_off, Th, qw), kv_cache(kv_bk, layer),
                    praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                    bcol_at(w.score, m_off, c.max_blocks, 0, live, Th));
        RAD_ISSUE_N(c_, op_select_shard, Th,
                    bcol_at(w.score, m_off, c.max_blocks, 0, live, Th),
                    praw2(atb->block_table, RAD_I32, S, atb->block_table_pitch),
                    brows(w.nc, S),
                    praw((const int32_t*)batch->positions + m_off, RAD_I32, Th),
                    praw(batch->cu_seqlens, RAD_I32, S + 1),
                    brows(w.sel_shard, Th), brows(w.nsel, Th), brows(w.sequ_shard, Th));
        RAD_ISSUE_N(c_, op_gather_sel, Th * (c.topk + 1), brows(w.sel_shard, Th), brows(w.sel, 2 * Th));
        RAD_ISSUE_N(c_, op_gather_sequ, Th, brows(w.sequ_shard, Th), brows(w.sequ, 2 * Th));
    } else {
        RAD_ISSUE_N(c_, op_score, T,
                    brows(w.q, T), kv_cache(kv_bk, layer),
                    praw2(bkb->block_table, RAD_I32, S, bkb->block_table_pitch),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                    bcol_at(w.score, 0, c.max_blocks, 0, live, T));
        RAD_ISSUE_N(c_, op_select, T,
                    bcol_at(w.score, 0, c.max_blocks, 0, live, T),
                    praw2(atb->block_table, RAD_I32, S, atb->block_table_pitch),
                    brows(w.nc, S), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1),
                    brows(w.sel, T), brows(w.nsel, T), brows(w.sequ, T));
    }

    /* RADIANCE_DEBUG_QSA: the only window into a selection. Nothing else in the tree reports one,
     * and the numbers it prints are what tell a divergence from the dense path apart from QSA
     * doing its job -- `nc` above `topk` is blocks being dropped ON PURPOSE. It SYNCHRONISES THE
     * STREAM, so it is an environment variable, off, and costs nothing when it is not set; the
     * first few decode steps are enough to see the shape. */
    if (getenv("RADIANCE_DEBUG_QSA")) {
        static int seen = 0;
        if (seen++ < 6) {
            rad_stream_sync(rad_stream(c_));
            std::vector<int32_t> hnc(1), hns(1), hsu(1), hsel(8);
            std::vector<float>   hsc(8);
            rad_memcpy_async(hnc.data(), rad_buf_ptr(c_, w.nc),   4, rad_stream(c_));
            rad_memcpy_async(hns.data(), rad_buf_ptr(c_, w.nsel), 4, rad_stream(c_));
            rad_memcpy_async(hsu.data(), rad_buf_ptr(c_, w.sequ), 4, rad_stream(c_));
            rad_memcpy_async(hsel.data(), rad_buf_ptr(c_, w.sel), 32, rad_stream(c_));
            rad_memcpy_async(hsc.data(), rad_buf_ptr(c_, w.score), 32, rad_stream(c_));
            rad_stream_sync(rad_stream(c_));
            std::fprintf(stderr, "D qsa L%d nc=%d nsel=%d sequ=%d sel[%d %d %d %d] "
                                 "score[%.4g %.4g %.4g %.4g]\n", layer,
                         hnc[0], hns[0], hsu[0], hsel[0], hsel[1], hsel[2], hsel[3],
                         hsc[0], hsc[1], hsc[2], hsc[3]);
        }
    }
}

inline void QsaIndexer::commit(RadCtx* ctx, const RadBatch* batch) const {
    if (!on() || !defer_decode(batch)) return;
    const auto* bkb = kv_batch(batch, kv_bk);
    const auto* tlb = kv_batch(batch, c.kv_tail);
    if (!bkb || !tlb || !batch->draft_out_ids || !batch->num_accepted) return;
    /* These two arrays are explicitly HOST_MAPPED in the commit-only pass.
     * The suffix is the common cu={0,1} for each single-token replay. */
    const int32_t* paths = batch->draft_out_ids;
    const int64_t path_pitch = batch->n_seq > 0 && batch->n_draft_out % batch->n_seq == 0
        ? batch->n_draft_out / batch->n_seq : 0;
    const int32_t* cu = paths + batch->n_seq * path_pitch;
    const int64_t tw = tlb->state_index_pitch > 0 ? tlb->state_index_pitch : 1;
    const int64_t nw = (c.heads + 1) * c.head_dim, koff = c.heads * c.head_dim;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    (void)DT;
    /* Logical path width bounds node IDs, independently of physical row stride.
     * Rotary planes retain the exact token replay below.
     * One launch replaces 3 * sum(accepted lengths) launches for this indexer. */
    if (op_tree_commit && D > 0 && (path_pitch == 16 || path_pitch == 65) && !batch->rope_pos) {
        const int32_t* lengths = batch->num_accepted;
        const int32_t* starts = lengths + batch->n_seq;
        RAD_ISSUE_N(ctx, op_tree_commit, D,
                    bcol_at(w.qk, 0, nw, koff, c.head_dim, batch->n_tok),
                    praw(batch->positions, RAD_I32, batch->n_tok),
                    praw2(paths, RAD_I32, D, path_pitch), praw(lengths, RAD_I32, D),
                    praw(starts, RAD_I32, D), kv_cache(c.kv_tail, layer),
                    praw2(tlb->state_index, RAD_I32, D, tw),
                    praw2(bkb->block_table, RAD_I32, D, bkb->block_table_pitch),
                    RAD_W(w_kn), kv_cache(kv_bk, layer));
        return;
    }
    /* qsa_work with RAD_NONE writes ONLY component 0, not three broadcasts.
     * Without planes, expose that scalar column to qsa_block_key: mrope/imrope
     * both use NeoX for one column. A full [work,3] view would read stale columns.
     * Plane-bearing replay is deliberately unchanged; it is not fused above. */
    RadOperand bpos = g.rope_mc && !batch->rope_pos
        ? RAD_B_COL(w.bpos, 0, 1) : brows(w.bpos, 2);
    bpos.rows = 2;
    for (int64_t s = 0; s < D; ++s) {
        const int32_t* row = paths + s * path_pitch;
        /* Prefix row offsets follow the mapped path lengths in commit metadata. */
        const int32_t* lengths = batch->num_accepted;
        const int32_t* starts = lengths + batch->n_seq;
        for (int j = 0; j < lengths[s]; ++j) {
            const int64_t tok = starts[s] + row[j];
            RadOperand pos = praw(batch->positions + tok, RAD_I32, 1);
            RAD_ISSUE_N(ctx, op_work, 1,
                        bcol_at(w.qk, tok, nw, koff, c.head_dim, 1), pos,
                        praw(cu, RAD_I32, 2), kv_cache(c.kv_tail, layer),
                        praw2(tlb->state_index + s * tw, RAD_I32, 1, tw),
                        praw2(bkb->block_table + s * bkb->block_table_pitch, RAD_I32, 1,
                              bkb->block_table_pitch),
                        RAD_NONE, brows(w.stage, 2 * c.ratio), brows(w.page, 2),
                        bpos, brows(w.nc, 1));
            RAD_ISSUE_N(ctx, op_bkey, 2,
                        brows(w.stage, 2 * c.ratio), RAD_W(w_kn), brows(w.page, 2),
                        bpos, kv_cache(kv_bk, layer));
            RAD_ISSUE_N(ctx, op_tail, 1,
                        bcol_at(w.qk, tok, nw, koff, c.head_dim, 1), pos,
                        praw(cu, RAD_I32, 2), kv_cache(c.kv_tail, layer),
                        praw2(tlb->state_index + s * tw, RAD_I32, 1, tw));
        }
    }
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_QSA_H */
