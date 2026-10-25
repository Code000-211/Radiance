/* opshapes.cpp -- operand shapes, dtype widening and the seeded fill (opshapes.h).
 *
 * Shared by rad-kbench and rad-tune so the tuner and the checker generate identical bytes.
 */
#include "opshapes.h"

#include <cmath>
#include <cstring>
#include <thread>

namespace rad {

int rad_operand_shapes_hook(const RadKernelInfo* k, const Geometry& g, int n_opd,
                            std::vector<OpdShape>* out) {
    if (!k || !k->opd_shape || n_opd <= 0) return RAD_E_UNSUPPORTED;
    out->assign((size_t)n_opd, OpdShape{});
    for (int i = 0; i < n_opd; ++i) {
        RadOpdDesc d{};
        d.idx_const = -1;   /* the ABI's "draw" default */
        const int st = k->opd_shape(g.params(), g.n_params(), i, &d);
        if (st < 0) return st;
        /* An inexpressible rank is a hook bug, not a decline; it must not become a zero-extent
         * buffer that passes every comparison vacuously. */
        if (d.rank == 0 || d.rank > RAD_MAX_RANK) {
            if (!(d.flags & RAD_OPD_F_ABSENT)) return RAD_E_SHAPE;
        }
        OpdShape& o = (*out)[(size_t)i];
        o.dtype = d.dtype;
        o.shape.assign(d.shape, d.shape + (d.rank <= RAD_MAX_RANK ? d.rank : 0));
        o.absent     = (d.flags & RAD_OPD_F_ABSENT)     != 0;
        o.idx_unique = (d.flags & RAD_OPD_F_IDX_UNIQUE) != 0;
        o.idx_cu     = (d.flags & RAD_OPD_F_IDX_CU)     != 0;
        o.idx_prefix = (d.flags & RAD_OPD_F_IDX_PREFIX) != 0;
        o.idx_history = (d.flags & RAD_OPD_F_IDX_HISTORY) != 0;
        o.idx_ordered = (d.flags & RAD_OPD_F_IDX_ORDERED) != 0;
        o.is_index   = d.fill == RAD_FILL_INDEX;
        o.idx_max    = d.idx_max;
        o.idx_const  = d.idx_const;
        o.fill_chunk = d.fill_chunk;
        switch (d.fill) {
            case RAD_FILL_SIGMOID:       o.fill = OpdShape::FillSigmoid;      break;
            case RAD_FILL_GATE_CUMSUM:   o.fill = OpdShape::FillGateCumsum;   break;
            case RAD_FILL_TRI_LOWER:     o.fill = OpdShape::FillTriLower;     break;
            case RAD_FILL_UNIT_ROW:      o.fill = OpdShape::FillUnitRow;      break;
            case RAD_FILL_SAMPLE_PARAMS: o.fill = OpdShape::FillSampleParams; break;
            default:                     o.fill = OpdShape::FillNormal;       break;
        }
        for (int64_t e : o.shape) if (e <= 0 && !o.absent) return RAD_E_SHAPE;
    }
    return RAD_OK;
}

int rad_operand_shapes(const std::string& op, const Geometry& g, int n_opd, int64_t M,
                       std::vector<OpdShape>* out, const ShapeCtx* ctx) {
    auto gi = [&](const char* k, int64_t dflt) {
        long long v = 0;
        return g.get_i(k, &v) ? (int64_t)v : dflt;
    };
    /* Activation dtype: `dtype`, else attention's `q_dtype` / `kv_dtype`, else bf16 (the
     * project's activation type). Some ops, e.g. `rope` and `qk_norm_rope`, carry no dtype; an
     * f32 fallback would make a bf16 kernel refuse the case. */
    const char* dts = g.get_s("dtype");
    if (!dts) dts = g.get_s("q_dtype");
    if (!dts) dts = g.get_s("kv_dtype");
    const uint32_t dt = dts ? rad_dtype_parse(dts) : (uint32_t)RAD_BF16;
    const uint32_t D  = dt == RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;

    const int64_t n     = gi("n", 0);
    const int64_t N     = gi("N", 0);
    const int64_t K     = gi("K", 0);
    const int64_t group = gi("group", 128);

    out->clear();
    auto add = [&](uint32_t d, std::vector<int64_t> s, bool idx = false) {
        OpdShape o;
        o.dtype = d; o.shape = std::move(s); o.is_index = idx;
        out->push_back(std::move(o));
    };
    auto add_fill = [&](uint32_t d, std::vector<int64_t> s, int f, int64_t chunk = 0) {
        OpdShape o;
        o.dtype = d; o.shape = std::move(s); o.fill = f; o.fill_chunk = chunk;
        out->push_back(std::move(o));
    };
    auto add_idx = [&](std::vector<int64_t> s, int64_t max, int64_t konst = -1,
                       bool uniq = false) {
        OpdShape o;
        o.dtype = (uint32_t)RAD_I32; o.shape = std::move(s); o.is_index = true;
        o.idx_max = max; o.idx_const = konst; o.idx_unique = uniq;
        out->push_back(std::move(o));
    };
    /* Cumulative sequence lengths for S sequences (OpdShape::idx_cu). */
    auto add_cu = [&](int64_t S) { add_idx({S + 1}, 0, -1); out->back().idx_cu = true; };
    /* One optional operand this case does not pass. See OpdShape::absent. */
    auto skip_opd = [&]() { OpdShape o; o.absent = true; out->push_back(std::move(o)); };

    const int64_t head_dim  = gi("head_dim", 0);
    const int64_t n_head    = gi("n_head", 0);
    const int64_t n_head_kv = gi("n_head_kv", n_head);

    if (op == "rmsnorm")           { add(D, {M, n}); add(D, {n}); add(D, {M, n}); }
    else if (op == "rope") {
        /* One inout operand plus positions. qkv is the fused row: n_head queries and n_head_kv
         * each of k and v, all head_dim wide. */
        if (head_dim <= 0 || n_head <= 0) return RAD_E_UNSUPPORTED;
        add(D, {M, (n_head + 2 * n_head_kv) * head_dim});
        add((uint32_t)RAD_I32, {M}, true);
    }
    else if (op == "rmsnorm_add")  { add(D, {M, n}); add(D, {M, n}); add(D, {n});
                                     add(D, {M, n}); add(D, {M, n}); }
    else if (op == "layernorm")    { add(D, {M, n}); add(D, {n}); add(D, {n}); add(D, {M, n}); }
    else if (op == "add" || op == "mul") { add(D, {M, n}); add(D, {M, n}); add(D, {M, n}); }
    else if (op == "silu_mul")     { add(D, {M, 2 * n}); add(D, {M, n}); }
    else if (op == "gelu" || op == "silu" || op == "sigmoid") { add(D, {M, n}); add(D, {M, n}); }
    else if (op == "softmax")      { add(D, {M, n}); add(D, {M, n}); }
    else if (op == "gather_rows") {
        /* idx picks M of 4M rows, so the gather is not the identity (which passes while broken). */
        add(D, {4 * M, n}); add((uint32_t)RAD_I32, {M}, true); add(D, {M, n});
    }
    else if (op == "cast") {
        const uint32_t from = rad_dtype_parse(g.get_s("from") ? g.get_s("from") : "f32");
        const uint32_t to   = rad_dtype_parse(g.get_s("to")   ? g.get_s("to")   : "f32");
        add(from ? from : (uint32_t)RAD_F32, {M, n});
        add(to   ? to   : (uint32_t)RAD_F32, {M, n});
    }
    else if (op == "gemm_nt")      { add(D, {M, K}); add(D, {N, K}); add(D, {M, N}); }
    else if (op == "gemm_nt_bias") {
        /* The optional residual is passed, to cover the epilogue, as libref's description does. */
        add(D, {M, K}); add(D, {N, K}); add(D, {N}); add(D, {M, N}); add(D, {M, N});
    }
    else if (op == "embed_lookup") {
        add((uint32_t)RAD_I32, {M}, true);
        add(D, {gi("n_vocab", 0), gi("n_embd", 0)});
        add(D, {M, gi("n_embd", 0)});
    }
    else if (op == "logits_gemm") {
        add(D, {M, gi("n_embd", 0)});
        add(D, {gi("n_vocab", 0), gi("n_embd", 0)});
        add((uint32_t)RAD_F32, {M, gi("n_vocab", 0)});
    }
    else if (op == "quant_act_i8" || op == "had_quant_act_i8") {
        /* One scale per row. quant_act_i8's optional `asum` (group-major per-group code sums, rows
         * padded to 16) is passed when `group` divides n, as libref describes it. */
        add(D, {M, n}); add((uint32_t)RAD_I8, {M, n}); add((uint32_t)RAD_F32, {M, 1});
        const int64_t g = gi("group", 0);
        if (op == "quant_act_i8") {
            if (g > 0 && n % g == 0) add((uint32_t)RAD_F32, {n / g, (M + 15) / 16 * 16});
            else skip_opd();
        }
    }
    else if (op == "quant_act_fp8") {
        /* quant_act_i8's operands at fp8 width. `dtype` names the (bf16) input. */
        if (n <= 0 || group <= 0) return RAD_E_UNSUPPORTED;
        add(D, {M, n});
        add((uint32_t)RAD_F8E4M3, {M, n});
        add((uint32_t)RAD_F32, {M, (n + group - 1) / group});
    }
    else if (op == "gemm_nt_q") {
        /* Block-scaled fp8 and mxfp4 only. `gemm_nt_q` covers six weight formats whose scale grids
         * (per row, per row-group, per output-row-block, e8m0 exponents, zero points, reference
         * exponents) belong to the reading kernel, not the schema. The fp8 grid is fixed by the
         * checkpoint: E4M3 [N][K] with a bf16 [ceil(N/128)][ceil(K/128)] scale plane, `group`
         * being the 128. Other formats are skipped and counted.
         *
         * `a_sum` and `b_ref` belong to the asymmetric-integer and mxfp4 grids. */
        const char* q = g.get_s("dtype");
        /* OCP-MXFP4: e2m1 codes with one E8M0 exponent per 32 of K (`group` 32), against the fp8
         * activation pair (E4M3 plus an f32 scale per row and 128 of K, as LinearFP8 feeds it).
         * Weights are in libr4d's stored layout and sized in bytes: codes K/2 per row in fragment
         * order, exponents [K/32][N], per-row reference [N]. Random bytes suffice for timing. */
        if (q && std::strcmp(q, "mxfp4a8") == 0) {
            if (N <= 0 || K <= 0 || group != 32) return RAD_E_UNSUPPORTED;
            add((uint32_t)RAD_F8E4M3, {M, K});
            add((uint32_t)RAD_F32,    {M, (K + 127) / 128});
            add((uint32_t)RAD_U8,     {N, K / 2});     /* b       -- e2m1 codes, fragment order */
            add((uint32_t)RAD_U8,     {K / 32, N});    /* b_scale -- E8M0, [K/32][N] */
            add((uint32_t)RAD_BF16,   {M, N});         /* y */
            skip_opd();                                /* a_sum */
            add((uint32_t)RAD_U8,     {N});            /* b_ref   -- the per-row reference */
        } else {
            const bool a8  = q && std::strcmp(q, "fp8a8")  == 0;
            const bool a16 = q && std::strcmp(q, "fp8a16") == 0;
            if ((!a8 && !a16) || N <= 0 || K <= 0 || group <= 0) return RAD_E_UNSUPPORTED;
            const int64_t kb = (K + group - 1) / group;
            const int64_t nb = (N + group - 1) / group;
            /* a, a_scale */
            if (a8) { add((uint32_t)RAD_F8E4M3, {M, K}); add((uint32_t)RAD_F32, {M, kb}); }
            else    { add((uint32_t)RAD_BF16,   {M, K}); skip_opd(); }
            add((uint32_t)RAD_F8E4M3, {N, K});    /* b       -- the checkpoint's own bytes */
            add((uint32_t)RAD_BF16,   {nb, kb});  /* b_scale -- weight_scale_inv, one per 128x128 */
            add((uint32_t)RAD_BF16,   {M, N});    /* y */
            skip_opd();                           /* a_sum */
            skip_opd();                           /* b_ref */
        }
    }
    else if (op == "dequant") {
        add((uint32_t)RAD_I8, {M, n}); add((uint32_t)RAD_F32, {M, group ? n / group : 1});
        add(D, {M, n});
    }
    else if (op == "all_reduce" || op == "all_gather") {
        /* all_gather's second operand is [world_size * numel]. all_reduce's is the optional second
         * destination (absent means in place, the usual form); it is passed to exercise the
         * out-of-place form. */
        const int64_t numel = gi("numel", n ? n : M);
        add(D, {numel});
        add(D, {op == "all_gather" ? numel * gi("world_size", 1) : numel});
    }
    /* ---- the paged pair, sharing the layout stated in libref/ref_ops.h:
     *
     *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]   K first, V second in a slot
     *
     * The block count and context length are this tool's choice; any cache large enough for the
     * case works as long as the indices land inside it. */
    else if (op == "kv_store") {
        const int64_t bs = gi("block_size", 0);
        const int64_t kvh = gi("n_head_kv", 0);
        if (bs <= 0 || kvh <= 0 || head_dim <= 0) return RAD_E_UNSUPPORTED;
        /* Two blocks of slack so the slot mapping is not a dense prefix, which would not test the
         * block index. */
        const int64_t nb = (M + bs - 1) / bs + 2;
        /* Only the cache takes `kv_dtype`; k and v are always bf16 (D would fall back to
         * kv_dtype). */
        const char*    kvs = g.get_s("kv_dtype");
        const uint32_t kvd = kvs ? rad_dtype_parse(kvs) : (uint32_t)RAD_DT_INVALID;
        add((uint32_t)RAD_BF16, {M, kvh, head_dim});
        add((uint32_t)RAD_BF16, {M, kvh, head_dim});
        add_idx({M}, nb * bs, -1, /*uniq*/ true);
        add(kvd == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : kvd, {nb, kvh, bs, 2 * head_dim});
    }
    else if (op == "attn_paged") {
        /* `M` is q_len (decode and prefill are different kernels, docs/OPS.md). One sequence whose
         * context is exactly this query: causal attention over its own prefix, as a prefill chunk
         * runs. */
        const int64_t bs  = gi("block_size", 16);
        const int64_t gqa = gi("gqa", 1);
        const int64_t nh  = gi("n_head", gqa);
        if (bs <= 0 || gqa <= 0 || nh <= 0 || head_dim <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
        const int64_t kvh = nh / gqa;
        const int64_t n_seq = 1, ctx = M;
        const int64_t max_blocks = (ctx + bs - 1) / bs;
        add(D, {n_seq * M, nh, head_dim});
        add(D, {max_blocks, kvh, bs, 2 * head_dim});
        add_idx({n_seq, max_blocks}, max_blocks);
        /* seqused is fixed, not drawn (see OpdShape::idx_const). */
        add_idx({n_seq}, 0, ctx);
        skip_opd();                           /* k_descale -- this cache is not quantised */
        skip_opd();                           /* v_descale */
        skip_opd();                           /* cu_seqlens -- one sequence, read rectangular */
        add(D, {n_seq * M, nh, head_dim});
    }
    /* ---- gated delta net. The layouts are libref/ref_gdn.cpp's, stated there once:
     *
     *     q, k     [T, Hk, head_k]         v, o  [T, Hv, head_v]
     *     g, beta  [T, Hv]  fp32           A     [T, Hv, chunk]
     *     h0, ht   [N, Hv, head_v, head_k] fp32  cu    [N+1]
     *     conv w   [conv_dim, conv_width]        conv_state [n_slots, conv_dim, state_len]
     *
     * with conv_dim = 2 * Hk * head_k + Hv * head_v.
     *
     * Head counts are not op parameters; they come from container metadata via `ctx`. Without
     * them the op is skipped, not guessed. */
    else if (op.rfind("gdn_", 0) == 0 && op != "gdn_gated_rmsnorm") {
        if (!ctx || ctx->n_head_k <= 0 || ctx->n_head_v <= 0) return RAD_E_UNSUPPORTED;
        const int64_t Hk = ctx->n_head_k, Hv = ctx->n_head_v;
        const int64_t hk = gi("head_k", 0), hv = gi("head_v", hk);
        const int64_t chunk = gi("chunk", 64);
        const int64_t width = gi("conv_width", 4);
        if (hk <= 0 || hv <= 0 || chunk <= 0 || width <= 1) return RAD_E_UNSUPPORTED;
        const int64_t conv_dim = 2 * Hk * hk + Hv * hv;
        const int64_t S = ctx->n_seq > 0 ? ctx->n_seq : 1;

        if (op == "gdn_conv_prep") {
            /* T tokens over S sequences, one conv-state slot each. The state length is
             * conv_width - 1 + n_spec, i.e. conv_width - 1 without speculation. */
            add(D, {M, conv_dim});                     /* x -- the fused projection row */
            add(D, {conv_dim, width});                 /* w */
            skip_opd();                                /* b -- this checkpoint has no conv bias */
            add((uint32_t)RAD_F32, {Hv});              /* A_log */
            add((uint32_t)RAD_F32, {Hv});              /* dt_bias */
            add(D, {S, conv_dim, width - 1});          /* conv_state, INOUT */
            add_cu(S);                                 /* cu */
            add(D, {M, Hv});                           /* a */
            add(D, {M, Hv});                           /* b_gate */
            add_idx({S}, S, -1, /*uniq*/ true);        /* cache_idx -- one slot a sequence */
            add_idx({S}, 0, 0);                        /* has_init -- cold prefill, passed
                                                        * explicitly: ref reads a null as warm,
                                                        * libr4d as cold. */
            add(D, {M, Hk, hk});                       /* q */
            add(D, {M, Hk, hk});                       /* k */
            add(D, {M, Hv, hv});                       /* v */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillGateCumsum, chunk);   /* g */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillSigmoid);             /* beta */
        }
        else if (op == "gdn_kkt_solve") {
            add(D, {M, Hk, hk});                       /* k */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillSigmoid);             /* beta */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillGateCumsum, chunk);   /* g */
            add_cu(S);                                 /* cu */
            add_fill(D, {M, Hv, chunk}, OpdShape::FillTriLower, chunk);              /* A */
        }
        else if (op == "gdn_chunk_scan") {
            add(D, {M, Hk, hk});                       /* q */
            add(D, {M, Hk, hk});                       /* k */
            add(D, {M, Hv, hv});                       /* v */
            add_fill(D, {M, Hv, chunk}, OpdShape::FillTriLower, chunk);              /* A */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillGateCumsum, chunk);   /* g */
            add_fill((uint32_t)RAD_F32, {M, Hv}, OpdShape::FillSigmoid);             /* beta */
            add((uint32_t)RAD_F32, {S, Hv, hv, hk});   /* h0 */
            add_cu(S);                                 /* cu */
            add(D, {M, Hv, hv});                       /* o */
            add((uint32_t)RAD_F32, {S, Hv, hv, hk});   /* ht */
            add_idx({S}, S, -1, /*uniq*/ true);        /* state_idx */
        }
        else if (op == "gdn_conv_update") {
            /* Decode. `M` is q_len, one row per candidate token; the window is
             * conv_width - 2 + q_len deep so a rejection only moves the read offset (ref_ops.h's
             * rollback contract). num_accepted is 1-based; all ones is non-speculative decode. */
            const int64_t sl = width - 2 + M;
            add(D, {S * M, conv_dim});                 /* x */
            add(D, {conv_dim, width});                 /* w */
            skip_opd();                                /* b */
            add(D, {S, conv_dim, sl});                 /* conv_state, INOUT */
            add_idx({S, M}, S, -1);                    /* state_idx */
            add_idx({S}, 0, 1);                        /* num_accepted -- none rejected */
            add_cu(S);                                 /* cu */
            add(D, {S * M, Hk, hk});                   /* q */
            add(D, {S * M, Hk, hk});                   /* k */
            add(D, {S * M, Hv, hv});                   /* v */
        }
        else if (op == "gdn_recurrent_update") {
            add(D, {S * M, Hk, hk});                   /* q */
            add(D, {S * M, Hk, hk});                   /* k */
            add(D, {S * M, Hv, hv});                   /* v */
            add(D, {S * M, Hv});                       /* a */
            add(D, {S * M, Hv});                       /* b */
            add((uint32_t)RAD_F32, {Hv});              /* A_log */
            add((uint32_t)RAD_F32, {Hv});              /* dt_bias */
            add((uint32_t)RAD_F32, {S * M, Hv, hv, hk});  /* state, one per candidate token */
            add_idx({S, M}, S * M, -1, /*uniq*/ true); /* state_idx */
            add_idx({S}, 0, 1);                        /* num_accepted */
            add_cu(S);                                 /* cu */
            add(D, {S * M, Hv, hv});                   /* z */
            add((uint32_t)RAD_F32, {hv});              /* norm_w */
            add(D, {S * M, Hv, hv});                   /* o */
            skip_opd();                                /* o_q -- the folded fp8 quantiser */
            skip_opd();                                /* o_scale */
        }
        else return RAD_E_UNSUPPORTED;
    }
    else if (op == "gdn_gated_rmsnorm") {
        /* One row per (token, head): `M` is already tokens x heads and `channels` is head_v. */
        const int64_t ch = gi("channels", 0);
        if (ch <= 0) return RAD_E_UNSUPPORTED;
        add(D, {M, ch}); add(D, {M, ch}); add((uint32_t)RAD_F32, {ch}); add(D, {M, ch});
    }
    /* Deliberately absent: ssm_conv and qk_norm_rope_gate take extents from the KV group's state
     * geometry and from interleaved strided views (attn_q is [n_embd, 2*head_dim*n_head] with query
     * and gate alternating per head). Only the kernel's hook can describe them. */
    else return RAD_E_UNSUPPORTED;

    if ((int)out->size() != n_opd) {
        /* Recipe and schema disagree: a bug in one of them. Do not pad with a guess. */
        RAD_DEBUG("%s: shape recipe produces %zu operands, the schema declares %d",
                  op.c_str(), out->size(), n_opd);
        return RAD_E_UNSUPPORTED;
    }
    for (const OpdShape& s : *out)
        for (int64_t d : s.shape)
            if (d <= 0) return RAD_E_UNSUPPORTED;
    return RAD_OK;
}

void rad_widen(const void* src, uint32_t dtype, int64_t n, std::vector<float>* out) {
    out->assign((size_t)n, 0.0f);
    switch (dtype) {
        case RAD_F32:
            std::memcpy(out->data(), src, (size_t)n * 4);
            break;
        case RAD_BF16:
            for (int64_t i = 0; i < n; ++i) {
                const uint32_t b = (uint32_t)((const uint16_t*)src)[i] << 16;
                float f;
                std::memcpy(&f, &b, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        case RAD_F16:
            for (int64_t i = 0; i < n; ++i) {
                const uint16_t h = ((const uint16_t*)src)[i];
                const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
                uint32_t bits;
                if (e == 0) {
                    /* Subnormal: renormalise rather than flush; a flushed value would pass a
                     * norm-based tolerance while wrong. */
                    if (m == 0) bits = sign;
                    else {
                        int sh = -1;
                        uint32_t mm = m;
                        do { mm <<= 1; ++sh; } while ((mm & 0x400) == 0);
                        bits = sign | ((uint32_t)(112 - sh) << 23) | ((mm & 0x3FF) << 13);
                    }
                } else if (e == 0x1F) {
                    bits = sign | 0x7F800000u | (m << 13);
                } else {
                    bits = sign | ((e + 112) << 23) | (m << 13);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        /* OCP fp8. Without these the fp8 family would pass vacuously: inputs and outputs would
         * all read as zero. e4m3 has no infinity and S1111111 is its only NaN; e5m2 is
         * IEEE-shaped. */
        case RAD_F8E4M3:
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t b = ((const uint8_t*)src)[i];
                const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
                const uint32_t e = (b >> 3) & 0xFu, m = b & 0x7u;
                uint32_t bits;
                if (e == 0) {
                    if (m == 0) bits = sign;
                    else {                                  /* subnormal: m * 2^-9 */
                        int sh = -1;
                        uint32_t mm = m;
                        do { mm <<= 1; ++sh; } while ((mm & 0x8u) == 0);
                        bits = sign | ((uint32_t)(127 - 6 - sh) << 23) | ((mm & 0x7u) << 20);
                    }
                } else if (e == 0xFu && m == 0x7u) {
                    bits = sign | 0x7FC00000u;              /* the one NaN encoding */
                } else {
                    bits = sign | ((e + 127 - 7) << 23) | (m << 20);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        case RAD_F8E5M2:
            for (int64_t i = 0; i < n; ++i) {
                const uint8_t b = ((const uint8_t*)src)[i];
                const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
                const uint32_t e = (b >> 2) & 0x1Fu, m = b & 0x3u;
                uint32_t bits;
                if (e == 0) {
                    if (m == 0) bits = sign;
                    else {
                        int sh = -1;
                        uint32_t mm = m;
                        do { mm <<= 1; ++sh; } while ((mm & 0x4u) == 0);
                        bits = sign | ((uint32_t)(127 - 14 - sh) << 23) | ((mm & 0x3u) << 21);
                    }
                } else if (e == 0x1Fu) {
                    bits = sign | 0x7F800000u | (m << 21);
                } else {
                    bits = sign | ((e + 127 - 15) << 23) | (m << 21);
                }
                float f;
                std::memcpy(&f, &bits, 4);
                (*out)[(size_t)i] = f;
            }
            break;
        case RAD_I8:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int8_t*)src)[i];
            break;
        case RAD_U8:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const uint8_t*)src)[i];
            break;
        case RAD_I32:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int32_t*)src)[i];
            break;
        /* Integer types must be readable here: an unhandled dtype widens to zeros on both the
         * recorder's and the reader's side, so any output would pass. The set matches what
         * libref's element accessor reads. i64 is exact only up to 2^24 in float, enough for
         * what these operands carry (sizes, offsets, token ids). */
        case RAD_I16:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int16_t*)src)[i];
            break;
        case RAD_I64:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const int64_t*)src)[i];
            break;
        case RAD_U32:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = (float)((const uint32_t*)src)[i];
            break;
        case RAD_BOOL:
            for (int64_t i = 0; i < n; ++i) (*out)[(size_t)i] = ((const uint8_t*)src)[i] ? 1.0f : 0.0f;
            break;
        default:
            /* Plugin-private dtype: left zeroed; the caller reports a zero reference as such. */
            break;
    }
}

/* Fixture inputs are redrawn from recorded seeds (kfixture.h), so every rounding below is part of
 * the recorded contract: changing one invalidates the recorded cases of that dtype. */
void rad_narrow(const float* src, uint32_t dtype, int64_t n, void* dst) {
    switch (dtype) {
        case RAD_F32:
            std::memcpy(dst, src, (size_t)n * 4);
            break;
        case RAD_BF16:
            /* Round-to-nearest-even, as rad-convert does; truncation would bias inputs toward
             * zero. */
            for (int64_t i = 0; i < n; ++i) {
                uint32_t b;
                std::memcpy(&b, &src[i], 4);
                ((uint16_t*)dst)[i] = (uint16_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
            }
            break;
        /* f16 and e5m2 truncate and flush subnormals to zero. */
        case RAD_F16:
            for (int64_t i = 0; i < n; ++i) {
                uint32_t x;
                std::memcpy(&x, &src[i], 4);
                const uint32_t sign = (x >> 16) & 0x8000u;
                int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
                const uint32_t m = x & 0x7FFFFF;
                uint16_t h;
                if (e >= 0x1F)      h = (uint16_t)(sign | 0x7C00u);
                else if (e <= 0)    h = (uint16_t)sign;
                else                h = (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
                ((uint16_t*)dst)[i] = h;
            }
            break;
        /* Unit normals fit E4M3 (max 448) unscaled. Round-to-nearest-even on the 3-bit mantissa
         * with a subnormal path, matching libref/ref_common.h. */
        case RAD_F8E4M3:
            for (int64_t i = 0; i < n; ++i) {
                float a = src[i] < 0 ? -src[i] : src[i];
                uint32_t sb;
                std::memcpy(&sb, &src[i], 4);
                const uint8_t sign = (uint8_t)((sb >> 24) & 0x80u);
                if (!(a > 0.0f)) { ((uint8_t*)dst)[i] = sign; continue; }
                if (a > 448.0f) a = 448.0f;
                if (a >= 0x1p-6f) {
                    uint32_t u;
                    std::memcpy(&u, &a, 4);
                    int e = (int)((u >> 23) & 0xFFu) - 127;
                    uint32_t mm = (u & 0x7FFFFFu) >> 20;
                    const uint32_t rem = u & 0xFFFFFu, half = 1u << 19;
                    if (rem > half || (rem == half && (mm & 1u))) { if (++mm == 8u) { mm = 0; ++e; } }
                    if (e > 8) { e = 8; mm = 7; }
                    ((uint8_t*)dst)[i] = (uint8_t)(sign | ((uint32_t)(e + 7) << 3) | mm);
                } else {
                    const float scaled = a * 512.0f;
                    uint32_t m = (uint32_t)(scaled + 0.5f);
                    if ((float)m - scaled == 0.5f && (m & 1u)) --m;
                    ((uint8_t*)dst)[i] = m > 7u ? (uint8_t)(sign | (1u << 3)) : (uint8_t)(sign | m);
                }
            }
            break;
        case RAD_F8E5M2:
            for (int64_t i = 0; i < n; ++i) {
                uint32_t x;
                std::memcpy(&x, &src[i], 4);
                const uint32_t sign = (x >> 24) & 0x80u;
                int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
                const uint32_t m = x & 0x7FFFFF;
                uint8_t h;
                if (e >= 0x1F)      h = (uint8_t)(sign | 0x7Cu);
                else if (e <= 0)    h = (uint8_t)sign;
                else                h = (uint8_t)(sign | ((uint32_t)e << 2) | (m >> 21));
                ((uint8_t*)dst)[i] = h;
            }
            break;
        /* Unit normals scaled by 32 to spread over the int8 range. */
        case RAD_I8:
            for (int64_t i = 0; i < n; ++i) {
                float v = src[i] * 32.0f;
                v = v < -127.f ? -127.f : (v > 127.f ? 127.f : v);
                ((int8_t*)dst)[i] = (int8_t)std::lrintf(v);
            }
            break;
        case RAD_U8:
            for (int64_t i = 0; i < n; ++i) {
                float v = src[i];
                v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
                ((uint8_t*)dst)[i] = (uint8_t)std::lrintf(v);
            }
            break;
        case RAD_I32:
            for (int64_t i = 0; i < n; ++i) ((int32_t*)dst)[i] = (int32_t)std::lrintf(src[i]);
            break;
        /* Integer types must be handled here, since `default` zeroes silently; e.g. ngram_ids
         * refuses `vocab_size <= 0`, so zeroed sizes would skip it at every geometry. Keep this
         * set in step with rad_widen.
         *
         * `llrintf`, not `lrintf`: long is 32-bit on some targets. Values past 2^24 do not
         * round-trip through float; shapes needing a large exact value use idx_const. */
        case RAD_I64:
            for (int64_t i = 0; i < n; ++i) ((int64_t*)dst)[i] = (int64_t)std::llrintf(src[i]);
            break;
        case RAD_I16:
            for (int64_t i = 0; i < n; ++i) ((int16_t*)dst)[i] = (int16_t)std::lrintf(src[i]);
            break;
        case RAD_U32:
            for (int64_t i = 0; i < n; ++i) {
                const float v = src[i] < 0.f ? 0.f : src[i];
                ((uint32_t*)dst)[i] = (uint32_t)std::llrintf(v);
            }
            break;
        case RAD_BOOL:
            for (int64_t i = 0; i < n; ++i) ((uint8_t*)dst)[i] = src[i] != 0.0f ? 1u : 0u;
            break;
        default:
            std::memset(dst, 0, (size_t)rad_dtype_bytes(dtype, n));
            break;
    }
}

uint64_t rad_case_seed(uint64_t base, std::string_view op, int band, int domain, int operand) {
    uint64_t h = base ^ 0xCBF29CE484222325ull;
    for (char c : op) { h ^= (unsigned char)c; h *= 0x100000001B3ull; }
    h ^= (uint64_t)band * 0x9E37u;      h *= 0x100000001B3ull;
    h ^= (uint64_t)domain * 0x85EBu;    h *= 0x100000001B3ull;
    h ^= (uint64_t)operand * 0xC2B2u;   h *= 0x100000001B3ull;
    return h ? h : 1;
}

/* Standard normals by Box-Muller: transformer activations are roughly Gaussian, and the tails
 * (outliers that break a per-row scale) matter, which a uniform fill lacks.
 *
 * This is the replay's hot loop (an fp8 GEMM weight at N=34816 K=5120 is 178M elements), so it is
 * threaded. One splitmix chain cannot be split, so the stream is cut into fixed blocks, each seeded
 * from (seed, block index). Values are then independent of thread count, which the contract
 * requires: fixtures recorded on one machine are replayed on another. */
const int64_t RAD_FILL_BLOCK = 1 << 16;

static void fill_block(float* p, int64_t n, uint64_t seed, float sigma) {
    uint64_t s = seed;
    for (int64_t i = 0; i < n; i += 2) {
        const double u1 = ((double)(rad_splitmix(s) >> 11) + 1.0) * (1.0 / 9007199254740992.0);
        const double u2 = ((double)(rad_splitmix(s) >> 11)) * (1.0 / 9007199254740992.0);
        const double r = std::sqrt(-2.0 * std::log(u1)), t = 6.283185307179586 * u2;
        p[i] = (float)((double)sigma * r * std::cos(t));
        if (i + 1 < n) p[i + 1] = (float)((double)sigma * r * std::sin(t));
    }
}

static uint64_t block_seed(uint64_t seed, int64_t b) {
    uint64_t s = seed ^ (0x9E3779B97F4A7C15ull * (uint64_t)(b + 1));
    return rad_splitmix(s);
}

/* One block of that stream, for callers drawing a large operand block by block to bound the float
 * intermediate. Same seed and block index give the same bytes. */
void rad_fill_normal_block(float* p, int64_t n, uint64_t seed, int64_t block, float sigma) {
    fill_block(p, n, block_seed(seed, block), sigma);
}

void rad_fill_normal(float* p, int64_t n, uint64_t seed, float sigma) {
    if (n <= 0) return;
    rad_for_blocks((n + RAD_FILL_BLOCK - 1) / RAD_FILL_BLOCK, [&](int64_t b) {
        const int64_t lo = b * RAD_FILL_BLOCK;
        fill_block(p + lo, std::min<int64_t>(RAD_FILL_BLOCK, n - lo), block_seed(seed, b), sigma);
    });
}

void rad_for_blocks(int64_t nb, const std::function<void(int64_t)>& fn) {
    const unsigned hw = std::thread::hardware_concurrency();
    const int64_t nt = std::min<int64_t>(nb, hw ? (int64_t)hw : 1);
    if (nt <= 1) {
        for (int64_t b = 0; b < nb; ++b) fn(b);
        return;
    }
    std::vector<std::thread> th;
    th.reserve((size_t)nt);
    for (int64_t t = 0; t < nt; ++t)
        th.emplace_back([&, t]() { for (int64_t b = t; b < nb; b += nt) fn(b); });
    for (auto& x : th) x.join();
}

}  /* namespace rad */
