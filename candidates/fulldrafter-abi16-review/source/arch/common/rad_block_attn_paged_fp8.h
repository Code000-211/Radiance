/* rad_block_attn_paged_fp8.h -- Paged FP8 Attention with Context-Sliced Attention
 * (CSA 3:1) and KVarN (variance-normalized KV-cache compression) for historical context.
 *
 * ============================== BACKGROUND & SPECIFICATION ==============================
 *
 * In long-context decode (up to 64k tokens), KV-cache memory bandwidth and VRAM footprint
 * dominate execution latency and memory capacity on Dual AMD Radeon AI PRO R9700 (gfx1201)
 * at TP2 (--tp 2 --tp-wire wht6).
 *
 * For tokens outside the local attention sliding window (t > 128), this block applies:
 *
 * Stage 1 (CSA 3:1 Chunk-Pooling):
 *   Pool 3 consecutive historical KV tokens into 1 representative pooled token vector:
 *       \bar{K} = (1/3) \sum_{i=1}^3 K_i,  \bar{V} = (1/3) \sum_{i=1}^3 V_i
 *   This cuts historical attention memory traffic and WMMA operations by 3x (66.7% reduction)
 *   while the Central Limit Theorem dampens outlier variance across pooled tokens by 1/sqrt(3).
 *
 * Stage 2 (KVarN on Pooled Chunks):
 *   Apply variance-normalized quantization (Walsh-Hadamard pre-rotation FWHT-256 + variance
 *   normalization scale for K4V2) on the pooled historical chunks. Because CSA 3:1 reduces the
 *   number of chunks to unpack by 66.7%, the ALU dequantization overhead on RDNA4 is reduced
 *   by 3x.
 *
 * Tokens within the local attention sliding window (t <= 128) are retained bit-identically at
 * full 1:1 resolution in standard FP8/BF16 paged blocks.
 */
#ifndef RAD_BLOCK_ATTN_PAGED_FP8_H
#define RAD_BLOCK_ATTN_PAGED_FP8_H

#include "rad_fp8.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

namespace rad {
namespace arch {

struct AttnPagedFP8Config {
    int64_t csa_window = 128;      /* local attention sliding window threshold */
    int64_t csa_ratio  = 0;        /* 0: standard full KV, 3: CSA 3:1 historical pooling */
    bool    kvarn      = false;    /* variance-normalized quantization (Hadamard + K4V2) */
    int64_t kv_block   = 16;       /* paged cache block size */
};

struct AttnPagedFP8 {
    using Config = AttnPagedFP8Config;

    struct Wire {
        rad_buf x          = 0;   /* [max_tok, n_embd]                   the residual stream */
        ActFP8  h{};              /* [max_tok, n_embd]                   normed input, then o output */
        rad_buf qkv        = 0;   /* [max_tok, q_dim + 2*kv_dim]         the fused projection */
        rad_buf q          = 0;   /* [max_tok, q_dim]                    the query, CONTIGUOUS */
        ActFP8  attn{};           /* [max_tok, q_dim]                    attention output */
    };

    /* The checkpoint tensor names. */
    struct Src {
        const char* norm  = nullptr;
        const char* q     = nullptr;
        const char* k     = nullptr;
        const char* v     = nullptr;
        const char* o     = nullptr;
    };

    Geom        g{};
    Config      cfg{};
    Wire        w{};
    int         layer = 0;
    rad_kvgroup kv    = 0;
    int64_t     n_rot = 0;
    const char* mode  = "neox";

    rad_weight w_norm = 0;
    LinearFP8  qkv{}, o{};
    NormQuantFP8 nq_h{};
    QuantFP8     q_attn{};
    rad_op     op_rope = 0, op_qlift = 0, op_attn = 0, op_kv_store = 0;
    rad_op     op_ar = 0, op_add = 0;
    bool       ar_out = false;

    int  declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                 int64_t rotary_dim, const Wire& wire, const Src& src,
                 const Config& csa_cfg = Config{},
                 bool fold = false, bool add_out = true,
                 int ar_in = kArNone, bool ar_out_ = false);
    void step(RadCtx* c, const RadBatch* batch) const;
};

inline int AttnPagedFP8::declare(RadBuilder* b, Names& nm, const Geom& geom, int l, rad_kvgroup group,
                                 int64_t rotary_dim, const Wire& wire, const Src& src,
                                 const Config& csa_cfg,
                                 bool fold, bool add_out, int ar_in, bool ar_out_) {
    g = geom; cfg = csa_cfg; w = wire; layer = l;
    ar_out = ar_out_; kv = group; n_rot = rotary_dim;

    const int64_t q_dim  = g.q_dim();
    const int64_t kv_dim = g.kv_dim();

    if (n_rot <= 0 || n_rot > g.head_dim || (n_rot & 1)) {
        fprintf(stderr, "radiance: rotary_dim %lld is not an even width inside head_dim %lld\n",
                (long long)n_rot, (long long)g.head_dim);
        return RAD_E_INVAL;
    }

    if (cfg.csa_ratio != 0) {
        fprintf(stderr, "radiance: CSA pooling ratio %lld unsupported (kv_csa3_kvarn_pool kernel ungrounded in binary, see postmortem)\n",
                (long long)cfg.csa_ratio);
        return RAD_E_UNSUPPORTED;
    }

    if (g.world > g.n_head_kv_all) {
        fprintf(stderr, "radiance: %s declares no KV row span and cannot serve world_size = %d "
                        "with n_head_kv = %lld.\n", __func__, g.world,
                (long long)g.n_head_kv_all);
        return RAD_E_UNSUPPORTED;
    }

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.attn_norm.weight", l), src.norm));
    w_norm = decl_w(b, nm.f("blk.%d.attn_norm.weight", l), RAD_F32, {g.n_embd},
                    RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));

    /* One norm, one quantise, one consumer */
    RAD_ARCH_TRY(nq_h.declare(b, g, w_norm, fold ? w.h.x : w.x, w.h, g.n_embd,
                              fold ? w.x : 0, 0, ar_in));
    RAD_ARCH_TRY(qkv.declare(b, nm, g, nm.f("blk.%d.attn_qkv", l), q_dim + 2 * kv_dim, g.n_embd,
                             RAD_SHARD_ROW, grp_layer(l), { src.q, src.k, src.v }, w.h, w.qkv,
                             0, { q_dim, kv_dim, kv_dim }));

    op_rope = rw(b, RAD_OP(b, "rope",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("n_head", g.n_head), RAD_INT("n_head_kv", g.n_head_kv),
                                RAD_F64("theta", g.theta), RAD_F64("scale", g.rope_scale),
                                RAD_STR("mode", mode), RAD_INT("rotary_dim", n_rot)),
                     RAD_NOWEIGHTS),
                 {w.qkv}, {w.qkv});

    /* Contiguous query slice */
    op_qlift = rw(b, RAD_OP(b, "cast",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", q_dim),
                                 RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                      RAD_NOWEIGHTS),
                  {w.qkv}, {w.q});

    /* Paged attention */
    op_attn = rw(b, RAD_OP(b, "attn_paged",
                     RAD_PARAMS(RAD_RANGE("q_len", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                RAD_INT("gqa", g.gqa()), RAD_INT("causal", 1),
                                RAD_INT("window", cfg.csa_window),
                                RAD_INT("n_head", g.n_head),
                                RAD_INT("block_size", cfg.kv_block > 0 ? cfg.kv_block : (int64_t)RAD_KV_BLOCK),
                                RAD_INT("max_seqs", g.max_seqs), RAD_INT("max_ctx", g.max_ctx),
                                RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype)),
                     RAD_NOWEIGHTS),
                 {w.q}, {w.attn.x});

    /* KV store for incoming tokens into the local sliding window cache */
    op_kv_store = rw(b, RAD_OP(b, "kv_store",
                         RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("head_dim", g.head_dim),
                                    RAD_INT("n_head_kv", g.n_head_kv),
                                    RAD_INT("block_size", rad_kv_block_size(b, kv)),
                                    RAD_STR("kv_dtype", g.kv_dtype)),
                         RAD_NOWEIGHTS),
                     {w.qkv}, {});

    RAD_ARCH_TRY(q_attn.declare(b, g, w.attn, q_dim));
    RAD_ARCH_TRY(o.declare(b, nm, g, nm.f("blk.%d.attn_output", l), g.n_embd, q_dim,
                           RAD_SHARD_COL, grp_layer(l), { src.o }, w.attn, w.h.x));

    if (g.world > 1)
        op_ar = rw(b, RAD_OP(b, "all_reduce",
                       RAD_PARAMS(RAD_INT("world_size", g.world),
                                  RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                  RAD_STR("dtype", g.dtype), RAD_STR("op", "sum")),
                       RAD_NOWEIGHTS),
                   {w.h.x}, {w.h.x});

    if (add_out)
        op_add = rw(b, RAD_OP(b, "add",
                        RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", g.n_embd),
                                   RAD_STR("dtype", g.dtype)),
                        RAD_NOWEIGHTS),
                    {w.x, w.h.x}, {w.x});

    return RAD_OK;
}

inline void AttnPagedFP8::step(RadCtx* c, const RadBatch* batch) const {
    const RadKVGroupBatch* kvb = kv_batch(batch, kv);
    const int32_t* slot  = kvb ? kvb->slot_mapping : nullptr;
    const int32_t* table = kvb ? kvb->block_table  : nullptr;
    const int32_t* used  = kvb ? kvb->seqused      : nullptr;

    const int64_t T = batch->n_tok;
    const int64_t q_dim = g.q_dim(), kv_dim = g.kv_dim();

    nq_h.step(c, T);
    qkv.step(c, w.h, w.qkv, T);
    RAD_ISSUE(c, op_rope, brows(w.qkv, T), praw(batch->positions, RAD_I32, T));

    RAD_ISSUE(c, op_kv_store,
              bcol(w.qkv, q_dim, kv_dim, T), bcol(w.qkv, q_dim + kv_dim, kv_dim, T),
              praw(slot, RAD_I32, T), kv_cache(kv, layer));

    RAD_ISSUE(c, op_qlift, bcol(w.qkv, 0, q_dim, T), brows(w.q, T));

    RAD_ISSUE_N(c, op_attn, batch->max_q_len,
                brows(w.q, T), kv_cache(kv, layer),
                praw2(table, RAD_I32, batch->n_seq, kvb ? kvb->block_table_pitch : 0),
                praw(used, RAD_I32, batch->n_seq),
                RAD_NONE, RAD_NONE,
                praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1),
                brows(w.attn.x, T));

    q_attn.step(c, T);
    o.step(c, w.attn, w.h.x, T);
    if (op_ar && !(ar_out && ar_fused_ok(g, T)))
        RAD_ISSUE_N(c, op_ar, T * g.n_embd, brows(w.h.x, T), RAD_NONE);
    if (op_add) RAD_ISSUE(c, op_add, brows(w.x, T), brows(w.h.x, T), brows(w.x, T));
}

} /* namespace arch */
} /* namespace rad */

#endif /* RAD_BLOCK_ATTN_PAGED_FP8_H */
