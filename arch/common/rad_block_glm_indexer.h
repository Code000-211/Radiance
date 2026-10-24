/* GLM-5.3's DSA indexer: the stateful compressed-index core over its paged key cache (DsaCache),
 * which scores every complete block of a sequence's history against each query and selects the
 * 512 the MLA attends to, and the front end around it (GlmIndexer), whose casts bridge ordinary
 * linear rows to the flattened per-head quantiser and join normalized keys with learned pool
 * gates. No RoPE; the only rotation is the Hadamard inside dsa_query. Selected pages belong to
 * the MLA.
 *
 * Under tensor parallelism the indexer is replicated whole: every rank pools into the whole index
 * cache, scores every query row against it and selects for every row. Only the head gates' f32
 * projection splits its columns from GlmTp::kSplitRows (SplitF32Proj). */
#ifndef RAD_BLOCK_GLM_INDEXER_H
#define RAD_BLOCK_GLM_INDEXER_H
#include "rad_fp8.h"
#include "rad_glm_tp.h"

namespace rad {
namespace arch {
struct DsaCache {
    struct Config {
        int64_t blocks = 0, work = 0, ring = 0, topk = 512;
    };
    struct Wire {
        rad_buf query = 0, codes = 0, scales = 0, gate = 0, kg = 0;
        rad_buf stage = 0, page = 0, bpos = 0, nc = 0, pooled = 0, pool_scale = 0;
        rad_buf score = 0, selected = 0, nsel = 0, length = 0;
    };
    Geom g{};
    Config c{};
    Wire w{};
    int layer = 0;
    rad_kvgroup index = 0, tail = 0, attention = 0;
    rad_weight ape = 0;
    rad_op op_query = 0, op_work = 0, op_pool = 0, op_tail = 0, op_score = 0, op_select = 0;

    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, rad_kvgroup attn, rad_kvgroup tail_group,
                rad_kvgroup* index_group, const char* ape_source) {
        g = geom;
        c = cfg;
        w = wire;
        layer = l;
        attention = attn;
        tail = tail_group;
        if (!b || !index_group || !ape_source || !attention || !tail || c.blocks < 1 ||
            c.work < (g.max_tok + 3) / 4 + g.max_seqs || c.ring < 4 + g.max_spec || c.topk != 512 ||
            !w.query || !w.codes || !w.scales || !w.gate || !w.kg || !w.stage || !w.page ||
            !w.bpos || !w.nc || !w.pooled || !w.pool_scale || !w.score || !w.selected || !w.nsel ||
            !w.length)
            return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, tail));
        const char* name = nm.f("blk.%d.index_ape", l);
        RAD_ARCH_TRY(map_copy(b, name, ape_source));
        ape = decl_w(b, name, RAD_F32, { 4, 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                     grp_layer(l));
        if (!ape) return RAD_E_INVAL;
        // Query buffers are [tokens*32,128]; scoring uses their rank-three view.
        op_query = rw(b,
                      RAD_OP(b, "dsa_query",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * 32), RAD_INT("n", 128),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                      { w.query }, { w.codes, w.scales });
        op_work = rw(
            b,
            RAD_OP(b, "qsa_work",
                   RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 256), RAD_INT("ratio", 4),
                              RAD_INT("ring", c.ring), RAD_INT("seqs", g.max_seqs),
                              RAD_INT("work", c.work), RAD_STR("dtype", "bf16")),
                   RAD_NOWEIGHTS),
            { w.kg }, { w.stage, w.page, w.bpos, w.nc });
        // Declare immediately before the cache consumer: page size comes from that op.
        if (!*index_group) {
            RadKVGroupDecl d{};
            d.kind = RAD_KV_FULL;
            d.dtype = RAD_U8;
            d.n_head_kv = 1;
            d.head_dim = 17;
            *index_group = rad_decl_kv_group(b, nm.f("kv_dsa_index"), &d);
        }
        index = *index_group;
        if (!index) return RAD_E_INVAL;
        op_pool =
            rw(b,
               RAD_OP(b, "dsa_pool",
                      RAD_PARAMS(RAD_RANGE("M", 1, c.work), RAD_INT("n", 128), RAD_INT("ratio", 4),
                                 RAD_INT("block_size", 4), RAD_STR("dtype", "bf16")),
                      RAD_WEIGHTS(ape)),
               { w.stage, w.page }, { w.pooled, w.pool_scale });
        if (!op_pool || rad_kv_block_size(b, index) != 4) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_bind_layer_kv(b, l, index));
        op_tail = rw(b,
                     RAD_OP(b, "qsa_tail_store",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 256),
                                       RAD_INT("ratio", 4), RAD_INT("ring", c.ring),
                                       RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "bf16")),
                            RAD_NOWEIGHTS),
                     { w.kg }, {});
        op_score = rw(b,
                      RAD_OP(b, "dsa_score",
                             RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                        RAD_INT("heads", 32), RAD_INT("blocks", c.blocks),
                                        RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", "fp8_e4m3")),
                             RAD_NOWEIGHTS),
                      { w.codes, w.scales, w.gate, w.nc }, { w.score });
        op_select = rw(b,
                       RAD_OP(b, "qsa_select",
                              RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("blocks", c.blocks),
                                         RAD_INT("topk", 512), RAD_INT("ratio", 4),
                                         RAD_INT("seqs", g.max_seqs), RAD_INT("signed", 1)),
                              RAD_NOWEIGHTS),
                       { w.score, w.nc }, { w.selected, w.nsel, w.length });
        return op_query && op_work && op_tail && op_score && op_select ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const auto* ib = kv_batch(batch, index);
        const auto* tb = kv_batch(batch, tail);
        const auto* ab = kv_batch(batch, attention);
        if (!ib || !tb || !ab || !ib->block_table || !ab->block_table || !tb->state_index) {
            rad_step_fail(ctx, "DSA requires index/attention block tables and tail state indices");
            return;
        }
        const int64_t T = batch->n_tok, S = batch->n_seq;
        const int64_t work = (T + 3) / 4 + S,
                      pitch = tb->state_index_pitch > 0 ? tb->state_index_pitch : 1;
        RAD_ISSUE_N(ctx, op_query, T * 32, brows(w.query, T * 32), brows(w.codes, T * 32),
                    brows(w.scales, T * 32));
        RAD_ISSUE_N(ctx, op_work, T, brows(w.kg, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch),
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch), RAD_NONE,
                    brows(w.stage, work * 4), brows(w.page, work), brows(w.bpos, work),
                    brows(w.nc, S));
        RAD_ISSUE_N(ctx, op_pool, work, brows(w.stage, work * 4), RAD_W(ape), brows(w.pooled, work),
                    brows(w.pool_scale, work), brows(w.page, work), kv_cache(index, layer));
        // Gather before overwriting the tail; the wider ring preserves rejected-token rollback.
        RAD_ISSUE_N(ctx, op_tail, T, brows(w.kg, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(tail, layer),
                    praw2(tb->state_index, RAD_I32, S, pitch));
        int64_t live = ((int64_t)batch->max_ctx_len + T + batch->n_spec + 4) / 4 + 1;
        if (live > c.blocks) live = c.blocks;
        // T is the total across sequences, so the conservative bound can exceed the
        // per-sequence page table. Columns without a table entry cannot be scored.
        if (live > ib->block_table_pitch) live = ib->block_table_pitch;
        if (live < 1) live = 1;
        RAD_ISSUE_N(ctx, op_score, T, brows(w.codes, T * 32), brows(w.scales, T * 32),
                    brows(w.gate, T), kv_cache(index, layer), RAD_NONE,
                    praw2(ib->block_table, RAD_I32, S, ib->block_table_pitch),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), brows(w.nc, S),
                    bcol_at(w.score, 0, c.blocks, 0, live, T));
        RAD_ISSUE_N(ctx, op_select, T, bcol_at(w.score, 0, c.blocks, 0, live, T),
                    praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch), brows(w.nc, S),
                    praw(batch->positions, RAD_I32, T), praw(batch->cu_seqlens, RAD_I32, S + 1),
                    brows(w.selected, T), brows(w.nsel, T), brows(w.length, T),
                    /* DSA reads the page ids; `selb` is the schema's ninth operand and an absent
                     * optional operand is a null one, not a shorter list (abi/rad_plugin.h). */
                    RAD_NONE);
    }
};

struct GlmIndexer {
    struct Wire {
        rad_buf input = 0, q_low = 0, q_wide = 0, kg_raw = 0;
        DsaCache::Wire cache{};
    };
    Geom g{};
    Wire w{};
    LinearFP8 query{}, key_gate{};
    DsaCache cache{};
    rad_weight gain = 0, bias = 0, head_weight = 0;
    rad_op op_norm = 0, op_query_copy = 0, op_gate_copy = 0, op_head = 0;
    SplitF32Proj head_split{};
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Wire& wire,
                const char* prefix, const DsaCache::Config& cfg, rad_kvgroup attention,
                rad_kvgroup tail, rad_kvgroup* index) {
        g = geom;
        w = wire;
        if (!b || !prefix || g.n_embd != 4096 || !w.input || !w.q_low || !w.q_wide || !w.kg_raw)
            return RAD_E_INVAL;
        Geom bg = g;
        bg.w_bf16 = true;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        RAD_ARCH_TRY(query.declare(b, nm, bg, nm.f("blk.%d.index_query", l), 4096, 1536,
                                   RAD_SHARD_NONE, grp_layer(l),
                                   { nm.ckpt("%s.wq_b.weight", prefix) }, act(w.q_low), w.q_wide));
        RAD_ARCH_TRY(key_gate.declare(
            b, nm, bg, nm.f("blk.%d.index_key_gate", l), 256, 4096, RAD_SHARD_NONE, grp_layer(l),
            { nm.ckpt("%s.wk.weight", prefix), nm.ckpt("%s.index_kpool_compress_gate", prefix) },
            act(w.input), w.kg_raw, 0, { 128, 128 }));
        const char *gn = nm.f("blk.%d.index_norm.weight", l),
                   *bn = nm.f("blk.%d.index_norm.bias", l),
                   *hn = nm.f("blk.%d.index_head.weight", l);
        RAD_ARCH_TRY(map_copy(b, gn, nm.ckpt("%s.k_norm.weight", prefix)));
        RAD_ARCH_TRY(map_copy(b, bn, nm.ckpt("%s.k_norm.bias", prefix)));
        RAD_ARCH_TRY(map_copy(b, hn, nm.ckpt("%s.weights_proj.weight", prefix)));
        gain = decl_w(b, gn, RAD_F32, { 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        bias = decl_w(b, bn, RAD_F32, { 128 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
        head_weight = decl_w(b, hn, RAD_F32, { 32, 4096 }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                             grp_layer(l));
        if (!gain || !bias || !head_weight) return RAD_E_INVAL;
        op_norm = rw(b,
                     RAD_OP(b, "layernorm",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                       RAD_F64("eps", 1e-6), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(gain, bias)),
                     { w.kg_raw }, { w.cache.kg });
        op_query_copy = rw(b,
                           RAD_OP(b, "cast",
                                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok * 32), RAD_INT("n", 128),
                                             RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                  RAD_NOWEIGHTS),
                           { w.q_wide }, { w.cache.query });
        op_gate_copy = rw(b,
                          RAD_OP(b, "cast",
                                 RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", 128),
                                            RAD_STR("from", "bf16"), RAD_STR("to", "bf16")),
                                 RAD_NOWEIGHTS),
                          { w.kg_raw }, { w.cache.kg });
        op_head = rw(b,
                     RAD_OP(b, "gemm_nt_f32",
                            RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("N", 32),
                                       RAD_INT("K", 4096), RAD_STR("dtype", "bf16")),
                            RAD_WEIGHTS(head_weight)),
                     { w.input }, { w.cache.gate });
        RAD_ARCH_TRY(head_split.declare(b, nm, g, nm.f("blk.%d.index_head", l), g.max_tok, 32, 4096,
                                        head_weight, w.input, w.cache.gate));
        RAD_ARCH_TRY(cache.declare(b, nm, g, l, cfg, w.cache, attention, tail, index,
                                   nm.ckpt("%s.index_kpool_compress_ape", prefix)));
        return op_norm && op_query_copy && op_gate_copy && op_head ? RAD_OK : RAD_E_INVAL;
    }
    void step(RadCtx* ctx, const RadBatch* batch) const {
        const int64_t T = batch->n_tok;
        auto act = [](rad_buf x) {
            ActFP8 a{};
            a.x = x;
            return a;
        };
        query.step(ctx, act(w.q_low), w.q_wide, T);
        RAD_ISSUE_N(ctx, op_query_copy, T * 32, brows(w.q_wide, T), brows(w.cache.query, T * 32));
        key_gate.step(ctx, act(w.input), w.kg_raw, T);
        RAD_ISSUE_N(ctx, op_norm, T, bcol(w.kg_raw, 0, 128, T), RAD_W(gain), RAD_W(bias),
                    bcol(w.cache.kg, 0, 128, T));
        RAD_ISSUE_N(ctx, op_gate_copy, T, bcol(w.kg_raw, 128, 128, T),
                    bcol(w.cache.kg, 128, 128, T));
        if (head_split.takes(T))
            head_split.step(ctx, T, brows(w.input, T), head_weight, brows(w.cache.gate, T));
        else
            RAD_ISSUE_N(ctx, op_head, T, brows(w.input, T), RAD_W(head_weight),
                        brows(w.cache.gate, T));
        cache.step(ctx, batch);
    }
};
}
} // namespace rad::arch
#endif
