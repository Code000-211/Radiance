/* rad_block_ple.h -- Qwen4-Exp's Per-Layer Embedding, declared and issued.
 *
 * ============================== WHAT IT IS ==============================
 *
 * ONE layer of the model gets a second embedding table, addressed not by the token but by a HASH of
 * the last three. Qwen3.8-Flash-Next's is 320,001,536 rows of 160 -- FOUR FIFTHS OF THE CHECKPOINT,
 * 51 of its 125 billion parameters -- read sixteen rows a token and mixed into the residual stream
 * by a dot product against the stream itself:
 *
 *   ids  = ngram_ids(tokens)                     [T, 16]   a hash of the last 3 ids, per head
 *   e    = embed_lookup_q(ids, table, scale)     [T, 2560]  16 x 160, flattened
 *   k    = key_proj(e)    [T, hc*n]              v = value_proj(e)    [T, n]
 *   gv,
 *   gvn  = ple_gate(k, h, v, ...)                the dot, the signed sqrt, the broadcast
 *   out  = ple_conv(gvn, resid = gv, ...)        dilated depthwise conv, silu, and the add
 *   h   += out
 *
 * and it happens BEFORE the attention hyper-connection reads the stream, so `h` here is the wide
 * un-normed residual.
 *
 * ============================== THE TABLE DECIDES THE SHAPE OF ALL OF THIS ======================
 *
 * 51.2 GB cannot be on the card and cannot be staged onto it, so `embed_lookup_q` runs on the HOST
 * over the container's own mmap and its result crosses the link once -- 5 KB a token at decode.
 * libavx/avx_ngram.cpp carries that argument in full. Three consequences reach this file:
 *
 *   THE TABLE IS A WEIGHT OPERAND, because the planner puts an op on the host only when every
 *   weight operand of it is Site::Host. A gather whose source were an activation could never be
 *   placed there.
 *
 *   ITS BUFFERS ARE HOST DOMAIN. `ids` and `e` are read and written by a host kernel, so they live
 *   in the host arena; everything downstream of `e` is ordinary device work.
 *
 *   IT IS 128 CHECKPOINT TENSORS. RadNameMap holds eight sources, so the map is declared in pieces
 *   by map_concat_shards() and the order IS the concatenation -- a permuted table is a model that
 *   loads, runs, and looks up the wrong embeddings.
 *
 * ============================== TWO STATES, TWO HISTORY LAYOUTS ==============================
 *
 * Both groups are RAD_KV_CONV. Token history needs ngram-1 ids; convolution
 * needs (width-1)*dilation inputs. Cold history is EOS for tokens, zero for conv.
 * Legacy ngram_ids decode uses num_accepted-1 and a shifted rolling window;
 * prefill leaves its committed tail at zero. The production GPU ple_conv uses
 * an absolute-position ring instead: ctx_lens plus a relative tap, modulo the
 * state's last extent (r4d_ple_conv_bf16.hip), not the canonical host ref layout.
 *
 * DLoop opts in through Config::tree_state. Decode reads immutable committed
 * canonical token history and ring convolution history, follows each candidate's
 * parents, and retains raw conv inputs until the -2 winner commit. Prefill
 * hashes with existing ngram_ids and materializes the ring with the additive
 * conv's prefill mode. No legacy decode or prefill launch changes without opt-in.
 */
#ifndef RAD_BLOCK_PLE_H
#define RAD_BLOCK_PLE_H

#include "rad_arch.h"
#include "rad_fp8.h"

namespace rad {
namespace arch {

struct PleLayer {
    struct Config {
        int64_t hc         = 0;    /* hyper-connection streams */
        int64_t embed_dim  = 0;    /* ple_embed_dim: heads * head_dim */
        int64_t heads      = 0;    /* (ngram - 1) * heads_per_ngram */
        int64_t ngram      = 0;
        int64_t eos        = -1;
        int64_t rows       = 0;    /* the table's row count, padded */
        int64_t shards     = 0;    /* how many checkpoint tensors it is stored as */
        int64_t conv_width = 0;
        int64_t dilation   = 0;    /* == ngram, in this model */
        bool tree_state = false;  /* DLoop: immutable candidates, then winner-path commit. */
        int64_t head_dim() const { return heads > 0 ? embed_dim / heads : 0; }
        int64_t hist()     const { return (conv_width - 1) * dilation; }
    };

    struct Wire {
        rad_buf h = 0;   /* [max_tok, hc * n_embd] THE WIDE STREAM, read and added into */
    };

    struct Src {
        /* The table's shards are named `<prefix><i><suffix>`; everything else is one tensor. */
        const char* shard_prefix = nullptr;
        const char* shard_suffix = nullptr;
        const char* mult         = nullptr;
        const char* vocab_sizes  = nullptr;
        const char* offsets      = nullptr;
        const char* key_proj     = nullptr;
        const char* value_proj   = nullptr;
        const char* norm_key     = nullptr;
        const char* norm_query   = nullptr;
        const char* norm_conv    = nullptr;
        const char* conv1d       = nullptr;
    };

    Geom   g{};
    Config c{};
    Wire   w{};
    int    layer_ = 0;

    rad_weight w_table = 0, w_scale = 0, w_mult = 0, w_vsz = 0, w_off = 0;
    /* The table as the checkpoint ships it -- plain bf16, no scale weight -- rather than E4M3. */
    bool       table_bf16 = false;
    rad_weight w_key = 0, w_value = 0, w_nk = 0, w_nq = 0, w_nc = 0, w_conv = 0;

    rad_kvgroup kv_tok = 0, kv_conv = 0;
    rad_buf b_ids = 0, b_emb = 0, b_embd = 0, b_k = 0, b_v = 0, b_gv = 0, b_gvn = 0, b_out = 0;
    rad_op  op_ids = 0, op_gather = 0, op_embd = 0, op_key = 0, op_value = 0;
    rad_op  op_gate = 0, op_conv = 0, op_add = 0;
    rad_buf tree_raw = 0;
    rad_op op_tree_ids = 0, op_tree_conv = 0, op_tree_prefill = 0, op_tree_commit = 0;

    int64_t hn() const { return c.hc * g.n_embd; }
    /* The prompt that follows the step, bounded by what `b_ids` holds after the step's rows. */
    int64_t ahead_rows(const RadBatch* batch) const {
        const int64_t a = batch->n_ahead;
        return a <= 0 ? 0 : (a < g.max_tok ? a : g.max_tok);
    }

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& cfg, int layer,
                 const Wire& wire, const Src& src);
    /* The hash, issued at the TOP of the step; the rest, at the layer. See step(). */
    void ids(RadCtx* c_, const RadBatch* batch) const;
    void prep(RadCtx* c_, const RadBatch* batch) const;
    void prep_after_ids(RadCtx* c_, const RadBatch* batch) const;
    void gate_conv_add(RadCtx* c_, const RadBatch* batch) const;
    void step(RadCtx* c_, const RadBatch* batch) const;
    void commit(RadCtx* c_, const RadBatch* batch) const;
};

inline int PleLayer::declare(RadBuilder* b, Names& nm, const Geom& geom, const Config& cfg,
                             int layer, const Wire& wire, const Src& src) {
    g = geom; c = cfg; w = wire;
    if (c.tree_state && (g.max_spec > 64 || g.act_dtype != RAD_BF16)) {
        fprintf(stderr, "radiance: PLE tree state requires BF16 and at most 65 rows "
                        "(64 draft nodes)\n");
        return RAD_E_UNSUPPORTED;
    }

    if (c.hc <= 1 || c.heads <= 0 || c.ngram < 2 || c.eos < 0 || c.rows <= 0 ||
        c.shards <= 0 || c.conv_width <= 1 || c.dilation <= 0 || c.embed_dim <= 0) {
        fprintf(stderr, "radiance: ple: geometry refused -- hc %lld heads %lld ngram %lld eos "
                        "%lld rows %lld shards %lld width %lld dilation %lld embed %lld\n",
                (long long)c.hc, (long long)c.heads, (long long)c.ngram, (long long)c.eos,
                (long long)c.rows, (long long)c.shards, (long long)c.conv_width,
                (long long)c.dilation, (long long)c.embed_dim);
        return RAD_E_INVAL;
    }
    /* `heads` is `ngram - 1` equal blocks, one per n-gram order, and `embed_dim` splits evenly
     * across them. Both are the reference's own constructor checks. */
    if (c.heads % (c.ngram - 1) || c.embed_dim % c.heads) {
        fprintf(stderr, "radiance: ple: %lld heads over %lld n-gram orders, %lld embed dims -- "
                        "neither divides\n", (long long)c.heads, (long long)(c.ngram - 1),
                (long long)c.embed_dim);
        return RAD_E_INVAL;
    }
    if (!w.h) return RAD_E_INVAL;
    if (!src.shard_prefix || !src.shard_suffix || !src.mult || !src.vocab_sizes ||
        !src.offsets || !src.key_proj || !src.value_proj || !src.norm_key || !src.norm_query ||
        !src.norm_conv || !src.conv1d) return RAD_E_INVAL;
    /* ============================== THE WHOLE LAYER IS REPLICATED UNDER TP ==============================
     *
     * Replication is the RIGHT answer here, not a gap: sharding would be the mistake.
     *
     * THE TABLE IS ADDRESSED BY A HASH OF THE TOKENS, which every rank computes identically, so a
     * vocabulary shard would give each rank rows it can never be asked for. And it costs NOTHING
     * to replicate: it is Tier::Mapped, read from the container's own file, so both ranks point at
     * the same table on the same machine and gather through one row cache they share.
     *
     * WHAT IS LEFT IS 63 MiB A RANK -- key_proj 50, value_proj 12.5, three norm gains and a conv
     * weight -- against the 112 GiB of experts TP exists to split here. Column-sharding the
     * projections would shard `gv` and `gvn` across the hyper-connection streams, and then the
     * convolution and the residual add would need the full width back: an all-gather a step, to
     * save 31 MiB a card. Geom::n_embd is "the residual stream, replicated on every rank", and
     * this layer reads and writes exactly that.
     *
     * SO EVERY RANK RUNS THE WHOLE LAYER AND THERE IS NO COLLECTIVE IN IT. Both ranks see the same
     * token ids, hash them the same way, gather the same rows and add the same `out` into their
     * own copy of a stream that was already identical -- bit for bit, because the all-reduce that
     * produced it delivers the same sum to both. The duplicated work is two small GEMMs and one
     * host gather per rank.
     *
     * ---- AND THAT LAST SENTENCE IS TRUE AT DECODE AND EXPENSIVE AT PREFILL ---------------------
     *
     * The duplicated gather is `heads` rows a TOKEN, so what is sixteen rows at a decode step is
     * sixteen thousand at a 1024-token chunk and thirty-two thousand at a 2048 one. A gather that
     * reads the table through the mapping costs each rank about 70 ms at a 2048-token chunk --
     * about 7.6% of a prefill step each. The RANKS ARE THREADS IN ONE PROCESS, so the page faults
     * are shared, but each rank still walks every row, and the step waits for both.
     *
     * THAT COST IS PAGE-TABLE POPULATION, NOT I/O. The distinct rows a prefill wants land on
     * distinct 4 KiB pages -- about ONE USEFUL ROW A PAGE, where a page holds 25 of them -- so a
     * gather through the mapping pays a fault per row. At decode, with varied text, it is the
     * other way round: half of every step's rows are new, and the page cache makes the waiting
     * thread start each of those reads itself. So the kernel keeps the rows in a cache of its own
     * and reads a miss straight from the file (libavx/avx_ngram.cpp).
     *
     * AND THAT CACHE SPLITS THE COLD WORK WITHOUT A COLLECTIVE. Dividing the rows between ranks,
     * or having one rank gather for both, would need a cross-rank HOST barrier inside a block that
     * deliberately has no collective in it. The shared cache does not: each rank claims the misses
     * it reaches first, reads only those, and waits only for rows the other rank has already
     * claimed -- so a miss is read once, and a late rank finds the rows there. What each rank still
     * does alone is decode the rows into its own output, which is a table lookup a byte. A
     * vocabulary shard is still wrong for the reason given above.
     */

    const int64_t HN = hn();
    const int64_t HD = c.head_dim();

    /* ---- the table, its scale, and the three hash constants -------------------------------- */
    /* E4M3 with ONE scale for all 320 million rows -- Qwen3.8-Flash-Next-FP8's own choice, and the
     * table's statistics say why that works: E4M3 carries its own exponent, a row's dynamic range
     * is 2.3x, and the scale sits on a three-octave plateau. ONE STORED WEIGHT, `fp8_e4m3*bf16[*x*]`
     * -- its codes and a one-element scale plane, made by the recipe from the dense checkpoint
     * (data/recipes/qwen4exp-w4.recipe) -- and the gather takes the two planes as two operands.
     * The shards are joined along rows by the name map, declared first because it is how the
     * model is asked what the table is. RAD_ACCESS_VOCAB because it IS a vocabulary table, just
     * not the token one. */
    const char* tn = nm.f("blk.%d.ple_ngram.weight", layer);
    RAD_ARCH_TRY(map_concat_shards(b, tn, src.shard_prefix, src.shard_suffix, (int)c.shards, 0));
    /* OR THE CHECKPOINT'S OWN bf16 TABLE, 102 GB, when the container holds it as it ships: the
     * same op with no scale (embed_lookup_q's absent scale is one), so it stays a host gather over
     * the mapping like the E4M3 one -- the gather's row cache serves either row size. */
    RadEncoding te{};
    table_bf16 = weight_enc(b, tn, &te) && rad_enc_is(&te, "plain") && te.n_planes == 1 &&
                 te.plane[0].dtype == RAD_BF16;
    if (table_bf16) {
        w_table = decl_w(b, tn, RAD_BF16, {c.rows, HD}, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                         grp_model());
    } else {
        w_table = decl_view(b, tn, nullptr, "codes", RAD_F8E4M3, {c.rows, HD},
                            RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
        w_scale = decl_view(b, nm.f("blk.%d.ple_ngram.scale", layer), tn, "scale", RAD_BF16, {1},
                            RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    }

    /* I64 AND NOT I32. The multipliers are ~2.4e13 -- the hash's whole point is that a token id
     * times one of them fills sixty-three bits -- so narrowing them would change every id. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_mult", layer), src.mult));
    w_mult = decl_w(b, nm.f("blk.%d.ple_mult", layer), RAD_I64, {c.ngram},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_vocab_sizes", layer), src.vocab_sizes));
    w_vsz = decl_w(b, nm.f("blk.%d.ple_vocab_sizes", layer), RAD_I64, {c.heads},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_offsets", layer), src.offsets));
    w_off = decl_w(b, nm.f("blk.%d.ple_offsets", layer), RAD_I64, {c.heads},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* ---- the projections and the three gains ------------------------------------------------ */
    /* BF16 AND NOT fp8: `modules_to_not_convert` in the FP8 checkpoint names both projections, so
     * these arrive dense whichever checkpoint is the source. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_key.weight", layer), src.key_proj));
    w_key = decl_w(b, nm.f("blk.%d.ple_key.weight", layer), RAD_BF16, {HN, c.embed_dim},
                   RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_value.weight", layer), src.value_proj));
    w_value = decl_w(b, nm.f("blk.%d.ple_value.weight", layer), RAD_BF16,
                     {g.n_embd, c.embed_dim}, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_norm_key.weight", layer), src.norm_key));
    w_nk = decl_w(b, nm.f("blk.%d.ple_norm_key.weight", layer), RAD_F32, {HN},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_norm_query.weight", layer), src.norm_query));
    w_nq = decl_w(b, nm.f("blk.%d.ple_norm_query.weight", layer), RAD_F32, {HN},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_norm_conv.weight", layer), src.norm_conv));
    w_nc = decl_w(b, nm.f("blk.%d.ple_norm_conv.weight", layer), RAD_F32, {HN},
                  RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    /* The checkpoint stores it [hc*n, 1, width]; the op wants [hc*n, width], which is the same
     * bytes with the depthwise group axis dropped. */
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.ple_conv.weight", layer), src.conv1d));
    w_conv = decl_w(b, nm.f("blk.%d.ple_conv.weight", layer), RAD_BF16, {HN, c.conv_width},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());

    if (!w_table || (!table_bf16 && !w_scale) || !w_mult || !w_vsz || !w_off || !w_key || !w_value ||
        !w_nk || !w_nq || !w_nc || !w_conv) return RAD_E_INVAL;

    /* ---- the two rolling windows ------------------------------------------------------------ */
    /* RAD_KV_CONV is "per-sequence, a rolling window of width-1 + num_spec entries", which is both
     * of these. The token window's `conv_width` is ngram, so its window is ngram - 1 + n_spec.
     *
     * THE CHANNEL COUNT IS n_head_kv * head_dim AND NOT state_dim, which is the one field pair a
     * reader of RadKVGroupDecl gets wrong: `state_dim` is RAD_KV_LINEAR's per-head geometry and
     * the CONV kind never looks at it. Declared into state_dim, both of these size to zero bytes a
     * sequence and the KV manager refuses the model at bring-up -- which is the good outcome, but
     * rad-kbench cannot find it: it synthesises the state operand from the op's own shape function
     * and never asks the manager for one. */
    {
        RadKVGroupDecl td{};
        td.kind = RAD_KV_CONV;
        td.dtype = RAD_I32;
        td.conv_width = c.ngram;
        td.n_head_kv = 1;
        td.head_dim = 1;            /* one token id a slot */
        kv_tok = rad_decl_kv_group(b, nm.f("kv_ple_tok"), &td);

        RadKVGroupDecl cd{};
        cd.kind = RAD_KV_CONV;
        cd.dtype = g.act_dtype;
        /* The conv's history is (width-1)*dilation deep, so the GROUP's width -- which the manager
         * reads as "window = conv_width - 1 + n_spec" -- is that plus one. Passing the raw
         * conv_width here would size the window for an undilated convolution and the deepest tap
         * would read a value that was never written. */
        cd.conv_width = c.hist() + 1;
        cd.n_head_kv = 1;
        cd.head_dim = HN;           /* [n_states, hc*n_embd, hist + n_spec] */
        kv_conv = rad_decl_kv_group(b, nm.f("kv_ple_conv"), &cd);
    }
    if (!kv_tok || !kv_conv) return RAD_E_INVAL;
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv_tok));
    RAD_ARCH_TRY(rad_bind_layer_kv(b, layer, kv_conv));

    /* ---- buffers ---------------------------------------------------------------------------- */
    /* HOST DOMAIN for the two the gather touches. `ids` is written by ngram_ids and read by the
     * host gather; `e` is written by the host gather and read by one device copy onto the card
     * (`ple_emb_dev`, below). The host arena is mapped, so the device side of that hand-off is a
     * link read (core/runtime/ctx.cpp) -- 5 KB a token at decode.
     *
     * `ids` IS PERSIST because it is written at the top of the step and read at this layer: the
     * buffer plan computes liveness in DECLARED order, where the two ops are neighbours, and a
     * transient would lend its bytes to anything issued in between. */
    {
        const char* ple_env = std::getenv("RADIANCE_PLE_BACKEND");
        if (!ple_env) ple_env = std::getenv("RADIANCE_PLE_STORAGE");
        if (!ple_env) ple_env = std::getenv("RADIANCE_PLE_MODE");
        bool ple_on_device = true;
        if (ple_env && *ple_env) {
            if (!std::strcmp(ple_env, "host") || !std::strcmp(ple_env, "cpu") ||
                !std::strcmp(ple_env, "nvme") || !std::strcmp(ple_env, "mmap")) {
                ple_on_device = false;
            }
        }

        /* TWICE THE STEP'S ROWS: the step's own ids, then the ids of the prompt that follows it
         * (RadBatch::n_ahead), which the gather hands to its reader to fetch while the step runs. */
        RadBufDecl d{};
        d.dtype = RAD_I32; d.rank = 2; d.shape[0] = 2 * g.max_tok; d.shape[1] = c.heads;
        d.kind = RAD_BUF_PERSIST;
        d.domain = ple_on_device ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST;
        b_ids = rad_decl_buffer(b, nm.f("ple_ids"), &d);

        /* [max_tok, embed_dim] AND NOT [max_tok * heads, head_dim], because ONE declared shape has
         * to serve two readings and only this one does. The gather writes `heads` rows a token and
         * the projections read one row a token of `embed_dim` -- the same bytes either way -- and
         * the direction that works is decided by HOW EACH KERNEL FINDS ITS ROW COUNT.
         *
         * The gather derives it: r4d_rows_pitch(x, n_embd) is numel / the width the kernel asks
         * for, so [T, 2560] read at 160 IS T*16 rows of 160. The GEMM does not: gemm_ops takes
         * M = a->shape[0] and K = a->shape[1] straight off the operand, because a strided rank-3
         * activation has to be refused rather than folded. So the wide shape serves both and the
         * tall one serves only the gather: declared tall, the key projection is handed M = T*16,
         * K = 160 against a [T, 10240] output and refuses the step. */
        RadBufDecl e{};
        e.dtype = g.act_dtype; e.rank = 2;
        e.shape[0] = g.max_tok; e.shape[1] = c.embed_dim;
        e.kind = RAD_BUF_TRANSIENT;
        e.domain = ple_on_device ? RAD_DOMAIN_DEVICE : RAD_DOMAIN_HOST;
        b_emb = rad_decl_buffer(b, nm.f("ple_emb"), &e);
    }
    /* AND ITS COPY ON THE CARD, which is what the two projections read. */
    b_embd = decl_b(b, nm.f("ple_emb_dev"), g.act_dtype, {g.max_tok, c.embed_dim});
    b_k   = decl_b(b, nm.f("ple_k"),   g.act_dtype, {g.max_tok, HN});
    b_v   = decl_b(b, nm.f("ple_v"),   g.act_dtype, {g.max_tok, g.n_embd});
    b_gv  = decl_b(b, nm.f("ple_gv"),  g.act_dtype, {g.max_tok, HN});
    b_gvn = decl_b(b, nm.f("ple_gvn"), g.act_dtype, {g.max_tok, HN});
    b_out = decl_b(b, nm.f("ple_out"), g.act_dtype, {g.max_tok, HN});
    if (!b_ids || !b_emb || !b_embd || !b_k || !b_v || !b_gv || !b_gvn || !b_out)
        return RAD_E_INVAL;
    if (c.tree_state) {
        tree_raw = decl_b(b, nm.f("blk.%d.ple_tree_raw", layer), RAD_BF16,
                          {g.max_tok, HN}, RAD_BUF_PERSIST);
        if (!tree_raw) return RAD_E_INVAL;
    }

    rad_buf_concurrent(b, b_ids);
    rad_buf_concurrent(b, b_emb);
    rad_buf_concurrent(b, b_embd);
    rad_buf_concurrent(b, b_k);
    rad_buf_concurrent(b, b_v);

    /* NO DERIVED BUFFER FOR THE STATE SLOTS, and that is worth saying because the obvious reading
     * of the core says otherwise. `conv_state_index` exists, computes exactly the cursor these two
     * states want, and is called by NO architecture plugin: rad_block_gdn_fp8.h takes
     * `cv->state_index` straight off RadKVGroupBatch, which the step batch fills PER GROUP. This
     * block does the same -- one fewer per-step computation, and the same number the working
     * precedent uses.
     *
     * ---- the ops, in the order step() issues them ------------------------------------------- */
    layer_ = layer;

    op_ids = rw(b, RAD_OP(b, "ngram_ids",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("heads", c.heads),
                              RAD_INT("ngram", c.ngram), RAD_INT("eos", c.eos)),
                   RAD_WEIGHTS(w_mult, w_vsz, w_off)),
               {}, {b_ids});

    /* M IS T * heads HERE, not T: the gather fetches one table row per HEAD per token, and the
     * band this op resolves on is that count. */
    if (table_bf16)
        op_gather = rw(b, RAD_OP(b, "embed_lookup_q",
                          RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * c.heads), RAD_INT("n_embd", HD),
                                     RAD_INT("n_vocab", c.rows), RAD_STR("dtype", g.dtype)),
                          RAD_WEIGHTS(w_table)),
                      {b_ids}, {b_emb});
    else
    op_gather = rw(b, RAD_OP(b, "embed_lookup_q",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * c.heads), RAD_INT("n_embd", HD),
                                 RAD_INT("n_vocab", c.rows), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(w_table, w_scale)),
                  {b_ids}, {b_emb});

    op_embd = rw(b, RAD_OP(b, "cast",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.embed_dim),
                                RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                     RAD_NOWEIGHTS),
                 {b_emb}, {b_embd});

    op_key = rw(b, RAD_OP(b, "gemm_nt",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", HN),
                              RAD_INT("K", c.embed_dim), RAD_STR("dtype", g.dtype)),
                   RAD_WEIGHTS(w_key)),
               {b_embd}, {b_k});
    op_value = rw(b, RAD_OP(b, "gemm_nt",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", g.n_embd),
                                RAD_INT("K", c.embed_dim), RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_value)),
                 {b_embd}, {b_v});

    /* `q` IS THE INCOMING STREAM ITSELF, un-normed: the op owns all three norms because the third
     * is taken over bytes it wrote. */
    op_gate = rw(b, RAD_OP(b, "ple_gate",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                               RAD_INT("hc", c.hc), RAD_F64("eps", g.eps),
                               RAD_STR("dtype", g.dtype), RAD_F64("wadd", g.wadd)),
                    RAD_WEIGHTS(w_nk, w_nq, w_nc)),
                {b_k, w.h, b_v}, {b_gv, b_gvn});

    if (c.tree_state) {
        const int64_t capacity = g.max_spec <= 15 ? 16 : 65;
        RadParam tp[] = {
            RAD_RANGE("q_len", 1, capacity), RAD_INT("n", HN), RAD_INT("heads", c.heads),
            RAD_INT("ngram", c.ngram), RAD_INT("eos", c.eos),
            RAD_INT("width", c.conv_width), RAD_INT("dilation", c.dilation),
            RAD_INT("node_capacity", capacity), RAD_INT("history_ring", 1),
            RAD_INT("tok_group", kv_tok), RAD_INT("conv_group", kv_conv),
            RAD_INT("layer", layer), RAD_INT("raw_buffer", tree_raw), RAD_STR("mode", "verify")
        };
        const int np = sizeof(tp) / sizeof(tp[0]);
        op_tree_ids = rw(b, rad_decl_op(b, "ngram_ids_tree", tp, np,
                                       RAD_WEIGHTS(w_mult, w_vsz, w_off)), {}, {b_ids});
        op_tree_conv = rw(b, rad_decl_op(b, "ple_conv_tree", tp, np, RAD_WEIGHTS(w_conv)),
                         {b_gvn, b_gv}, {b_out, tree_raw});
        op_tree_commit = rw(b, rad_decl_op(b, "ple_tree_commit", tp, np, RAD_NOWEIGHTS),
                           {tree_raw}, {});
        /* DLoop prefill materializes the same position-addressed ring as decode
         * commit. Chunk boundaries and restored checkpoint slots need no bridge. */
        tp[0] = RAD_RANGE("q_len", 1, g.max_tok);
        tp[np - 1] = RAD_STR("mode", "prefill");
        op_tree_prefill = rw(b, rad_decl_op(b, "ple_conv_tree", tp, np, RAD_WEIGHTS(w_conv)),
                            {b_gvn, b_gv}, {b_out});
        if (!op_tree_ids || !op_tree_conv || !op_tree_commit || !op_tree_prefill)
            return RAD_E_UNSUPPORTED;
    } else {
    op_conv = rw(b, RAD_OP(b, "ple_conv",
                    RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", HN),
                               RAD_INT("width", c.conv_width), RAD_INT("dilation", c.dilation),
                               RAD_STR("dtype", g.dtype)),
                    RAD_WEIGHTS(w_conv)),
                {b_gvn, b_gv}, {b_out});
    }

    /* AND THE ADD INTO THE STREAM. The reference is `hidden_states = hidden_states + ple(...)`,
     * and `ple(...)` is what ple_conv already produced -- the convolution's residual operand
     * folded the layer's own `gv + silu(conv)`, so what is left is one add of the whole thing. */
    op_add = rw(b, RAD_OP(b, "add",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", HN),
                              RAD_STR("dtype", g.dtype)),
                   RAD_NOWEIGHTS),
               {w.h, b_out}, {w.h});

    if (!op_ids || !op_gather || !op_embd || !op_key || !op_value || !op_gate ||
        (!c.tree_state && !op_conv) || !op_add)
        return RAD_E_INVAL;
    return RAD_OK;
}

/* THE SLOTS COME OFF THE GROUP BATCH, per group, exactly as the gated delta net reads them. And
 * `has_init` is `ctx_lens`: a sequence with computed tokens behind it has a history, one without
 * has none, which is the same test rad_block_gdn_fp8.h makes of the same field.
 *
 * THE TWO WINDOWED OPS ARE ISSUED ONCE PER HALF OF THE STEP. The step is sorted decode-first, so
 * the decode rows are sequences [0, D) holding tokens [0, DT) and the prefill chunks are sequences
 * [D, S) holding the rest (batch_split). The decode half passes `num_accepted`, which is what
 * makes the op leave its window in the layout a partly rejected verify can be read back from; the
 * prefill half passes none, so each chunk leaves its tail at offset zero. A pure step issues one
 * half, so a pure decode step is one launch and only a mixed step pays a second.
 *
 * The prefill half is addressed the way the GDN's is: the per-sequence arrays move to `+ D` and
 * the token operands stay at their base, because each kernel takes a sequence's rows from the
 * ABSOLUTE offsets in `cu`. The decode half is the prefix, so it narrows to DT rows instead. */
/* THE HASH GOES FIRST IN THE STEP, AHEAD OF EVERY LAYER. It reads the batch's token ids and its
 * own rolling window and nothing the trunk computes, so it can run as soon as the batch has been
 * uploaded. The gather that consumes it runs on the host and has to wait for it; issued here, that
 * wait is for one small kernel, and the layers issued between the two keep the device busy while
 * the host gathers. Issued at the layer, the wait would be for every layer in front of it, and the
 * device would sit idle through the gather. */
inline void PleLayer::ids(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;

    const RadKVGroupBatch* tk = kv_batch(batch, kv_tok);
    const int32_t* tk_idx = tk ? tk->state_index : nullptr;

    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;

    if (D > 0 && c.tree_state) {
        const int32_t qd = batch->phase == RAD_PHASE_MIXED ?
                           batch->max_q_len_decode : batch->max_q_len;
        RAD_ISSUE_N(c_, op_tree_ids, qd,
                    praw(batch->token_ids, RAD_I32, DT), kv_cache(kv_tok, layer_),
                    praw(tk_idx, RAD_I32, D), praw(batch->cu_seqlens, RAD_I32, D + 1),
                    batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE,
                    praw(batch->ctx_lens, RAD_I32, D),
                    RAD_W(w_mult), RAD_W(w_vsz), RAD_W(w_off), brows(b_ids, DT));
    } else if (D > 0)
        RAD_ISSUE_N(c_, op_ids, DT,
                    praw(batch->token_ids, RAD_I32, DT), kv_cache(kv_tok, layer_),
                    praw(batch->cu_seqlens, RAD_I32, D + 1),
                    RAD_W(w_mult), RAD_W(w_vsz), RAD_W(w_off),
                    praw(tk_idx, RAD_I32, D),
                    praw(batch->ctx_lens, RAD_I32, D),
                    praw(batch->num_accepted, RAD_I32, D),
                    brows(b_ids, DT), RAD_NONE);
    /* THE PROMPT THAT FOLLOWS the last prefill chunk is hashed in the same launch, into the rows
     * after the step's: `tok` then reaches past the step by that many tokens, and the window the
     * op leaves behind is the step's alone. */
    const int64_t A = P > 0 ? ahead_rows(batch) : 0;
    if (P > 0)
        RAD_ISSUE_N(c_, op_ids, T,
                    praw(batch->token_ids, RAD_I32, T + A), kv_cache(kv_tok, layer_),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1),
                    RAD_W(w_mult), RAD_W(w_vsz), RAD_W(w_off),
                    praw(tk_idx ? tk_idx + D : nullptr, RAD_I32, P),
                    praw(batch->ctx_lens + D, RAD_I32, P),
                    RAD_NONE,
                    brows(b_ids, T),
                    A > 0 ? brow_slice(b_ids, T, A, c.heads) : RAD_NONE);
}

inline void PleLayer::prep(RadCtx* c_, const RadBatch* batch) const {
    ids(c_, batch);
    prep_after_ids(c_, batch);
}

inline void PleLayer::prep_after_ids(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;

    /* T ROWS, not T * heads: the buffer is [max_tok, embed_dim] and the gather derives its own
     * 16-rows-a-token reading out of numel. The op's M band is still T * heads -- that is how many
     * table rows it fetches -- but the OPERAND is the token count. */
    /* The ids of the prompt that follows, which the gather's reader fetches while this step
     * runs, so the next step finds its rows already read. */
    const int64_t A = P > 0 ? ahead_rows(batch) : 0;
    RAD_ISSUE_N(c_, op_gather, T * c.heads,
                brows(b_ids, T), RAD_W(w_table), table_bf16 ? RAD_NONE : RAD_W(w_scale),
                brows(b_emb, T),
                A > 0 ? brow_slice(b_ids, T, A, c.heads) : RAD_NONE);

    RAD_ISSUE_N(c_, op_embd,  T, brows(b_emb, T), brows(b_embd, T));
    RAD_ISSUE_N(c_, op_key,   T, brows(b_embd, T), RAD_W(w_key),   brows(b_k, T));
    RAD_ISSUE_N(c_, op_value, T, brows(b_embd, T), RAD_W(w_value), brows(b_v, T));
}

inline void PleLayer::gate_conv_add(RadCtx* c_, const RadBatch* batch) const {
    const int64_t T = batch->n_tok;
    const int64_t S = batch->n_seq;

    const RadKVGroupBatch* cv = kv_batch(batch, kv_conv);
    const int32_t* cv_idx = cv ? cv->state_index : nullptr;

    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    const int64_t P = S - D;

    RAD_ISSUE_N(c_, op_gate, T,
                brows(b_k, T), brows(w.h, T), brows(b_v, T),
                RAD_W(w_nk), RAD_W(w_nq), RAD_W(w_nc),
                brows(b_gv, T), brows(b_gvn, T));

    const int32_t qd = batch->phase == RAD_PHASE_MIXED ? batch->max_q_len_decode : batch->max_q_len;
    if (D > 0 && c.tree_state) {
        RAD_ISSUE_N(c_, op_tree_conv, qd,
                    brows(b_gvn, DT), RAD_W(w_conv), kv_cache(kv_conv, layer_),
                    praw(cv_idx, RAD_I32, D), praw(batch->cu_seqlens, RAD_I32, D + 1),
                    batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE,
                    praw(batch->ctx_lens, RAD_I32, D), brows(b_gv, DT),
                    brows(b_out, DT), brows(tree_raw, DT));
    } else if (D > 0)
        RAD_ISSUE_N(c_, op_conv, qd > 0 ? qd : DT,
                    brows(b_gvn, DT), RAD_W(w_conv), kv_cache(kv_conv, layer_),
                    praw(batch->cu_seqlens, RAD_I32, D + 1), brows(b_gv, DT),
                    praw(cv_idx, RAD_I32, D),
                    praw(batch->ctx_lens, RAD_I32, D),
                    praw(batch->num_accepted, RAD_I32, D),
                    brows(b_out, DT));
    const int32_t qp = batch->max_q_len_prefill > 0 ? batch->max_q_len_prefill : batch->max_q_len;
    if (P > 0 && c.tree_state) {
        RAD_ISSUE_N(c_, op_tree_prefill, qp > 0 ? qp : T,
                    brows(b_gvn, T), RAD_W(w_conv), kv_cache(kv_conv, layer_),
                    praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1), RAD_NONE,
                    praw(batch->ctx_lens + D, RAD_I32, P), brows(b_gv, T),
                    brows(b_out, T), RAD_NONE);
    } else if (P > 0)
        RAD_ISSUE_N(c_, op_conv, qp > 0 ? qp : T,
                    brows(b_gvn, T), RAD_W(w_conv), kv_cache(kv_conv, layer_),
                    praw(batch->cu_seqlens + D, RAD_I32, P + 1), brows(b_gv, T),
                    praw(cv_idx ? cv_idx + D : nullptr, RAD_I32, P),
                    praw(batch->ctx_lens + D, RAD_I32, P),
                    RAD_NONE,
                    brows(b_out, T));

    /* M IS T, not T * HN. `add` counts ROWS of `n` -- every other block in this tree issues it
     * that way -- and the declared band is 1..max_tok. Passing the element count instead lands
     * outside every band of the op's own bucket table and the run phase refuses the step. */
    RAD_ISSUE_N(c_, op_add, T, brows(w.h, T), brows(b_out, T), brows(w.h, T));
}

inline void PleLayer::step(RadCtx* c_, const RadBatch* batch) const {
    prep_after_ids(c_, batch);
    gate_conv_add(c_, batch);
}

/* The commit-only pass keeps the original trunk token/parent rows alive.
 * Token history is canonical; convolution history is an absolute-position ring.
 * Only decode sequences replay: prefill (including mixed/chunked prefill) has
 * already materialized both windows, ready to checkpoint or restore as-is. */
inline void PleLayer::commit(RadCtx* c_, const RadBatch* batch) const {
    if (!c.tree_state) return;
    int64_t D = 0, DT = 0;
    batch_split(batch, &D, &DT);
    if (!D) return;
    const auto* tk = kv_batch(batch, kv_tok);
    const auto* cv = kv_batch(batch, kv_conv);
    const int32_t qd = batch->phase == RAD_PHASE_MIXED ?
                       batch->max_q_len_decode : batch->max_q_len;
    const int64_t path_pitch = batch->n_seq > 0 && batch->n_draft_out % batch->n_seq == 0
                                 ? batch->n_draft_out / batch->n_seq : 0;
    RAD_ISSUE_N(c_, op_tree_commit, qd,
                kv_cache(kv_tok, layer_), praw(tk ? tk->state_index : nullptr, RAD_I32, D),
                kv_cache(kv_conv, layer_), praw(cv ? cv->state_index : nullptr, RAD_I32, D),
                praw(batch->token_ids, RAD_I32, DT), brows(tree_raw, DT),
                praw(batch->cu_seqlens, RAD_I32, D + 1),
                batch->spec_parent ? praw(batch->spec_parent, RAD_I32, DT) : RAD_NONE,
                praw(batch->ctx_lens, RAD_I32, D),
                praw2(batch->draft_out_ids, RAD_I32, D, path_pitch),
                praw(batch->num_accepted, RAD_I32, D));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_BLOCK_PLE_H */
