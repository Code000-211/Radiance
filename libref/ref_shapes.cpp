/* ref_shapes.cpp -- how big is operand i of this op at this geometry, and what may legally be in
 * it. libref's implementation of `RadKernelInfo::opd_shape` (rad_abi.h), for every op this
 * plugin registers.
 *
 * ============================== Why this lives here ==============================
 *
 * A tool that calls an op cold (a correctness check, a benchmark) must size its buffers, and a
 * schema names operands and roles but not extents. A hand-written table in `tools/` (opshapes.h)
 * could only describe ops its author knew and could not be extended without patching the engine.
 * The component that reads an operand knows its extent; this file answers for the whole
 * conventional vocabulary, so any op in docs/OPS.md is describable even when the kernel under test
 * is not.
 *
 * ============================== One function per row, named after the launch ==============
 *
 * `RadShapeFn` gets only the resolved parameters and an operand index, not the op name. So every
 * entry point `ref_X` has a `ref_X_shape`, which ref_registry.cpp's ROW() macro pastes as
 * `fn ## _shape`; a new op without a description fails to link.
 *
 * ============================== The count is checked ==============================
 *
 * A description must emit exactly as many operands as the schema declares; that catches a
 * description that drifted from its schema (which would otherwise leave the op silently skipped).
 * The hook is called per operand, so each function fills a local array with the whole list and
 * hands out the requested entry, and `Opd::finish` compares the length with
 * `RadOpSchema::n_operands` from this plugin's schema table. A mismatch returns RAD_E_SHAPE, a bug
 * here or in the schema; RAD_E_UNSUPPORTED means the geometry lacks what the op needs, which a
 * caller skips and counts.
 *
 * ============================== What is in an operand ==============================
 *
 * The fill domain matters as much as the extent (see rad_abi.h):
 *
 *   * A gated-delta-net `g` is an intra-chunk cumulative sum of log-decays: non-increasing and
 *     never positive. The scan's e^{g_i - g_j}, j <= i, is <= 1 only because g is monotone; normal
 *     draws overflow the oracle to 2e38.
 *   * `beta` is a sigmoid output in (0, 1). `A` is a chunk's triangular inverse, zero above the
 *     diagonal within a chunk; a dense random A exposes a convention (whether the upper part is
 *     read) that never matters on real data.
 *   * An index operand's range is part of its meaning: a KV slot indexes n_blocks * block_size
 *     positions and a block-table entry n_blocks, neither being any operand's dim 0. Drawing from
 *     the wrong range tests out-of-range handling instead.
 *   * A scatter destination must be unique: two tokens on one slot are a race that ref (OpenMP over
 *     tokens) and a device kernel resolve differently. Read-side indices (gather rows, block-table
 *     entries) may repeat.
 *   * `seqused` is a constant, not a draw: a context shorter than the query makes causal attention
 *     read a nonexistent prefix.
 *   * A sampler's `params` rows are requests: a stage gated by a flag, a history window or a mask
 *     row does work only where those integer fields are what a request sets. Its token history
 *     must repeat itself, or DRY never finds a repeat to penalise.
 *   * An optional operand deliberately not passed still needs an entry for the count check; a null
 *     RadTensor.data marks absence, as in the ABI.
 *
 * ============================== What this file may use ==============================
 *
 * A kernel plugin links only rad_abi (libref/CMakeLists.txt), so core symbols outside headers
 * (e.g. RAD_DEBUG) are unreachable. Header-only ABI helpers are available: `rad_dtype_parse` and
 * `rad_dtype_bytes` (rad_types.h) and `rad_param_*` (rad_plugin.h). Use the ABI's version first;
 * `gi` below is the one local variant, and says why.
 *
 * `rad_kernel_schema_count` / `rad_kernel_schema_at` are this plugin's own exports
 * (ref_registry.cpp), so the count check adds no coupling.
 */
#include "ref_ops.h"

/* rad_param_find / rad_param_getdim / rad_param_gets / rad_dtype_parse. */
#include "rad_plugin.h"
#include "rad_runtime.h"   /* RAD_VGEOM_N */
#include "rad_sample.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <initializer_list>

namespace {

/* ================================================================== parameters
 * Not ref_common.h's p_int(), which takes a RadArgs; this hook gets a bare RadParam list before
 * anything is allocated. rad_param_getdim gives the same int-or-range reading (an uncollapsed range
 * describes the band's upper bound, the extent the arena is sized for).
 *
 * The one local leniency: a string spelling of an integer is accepted, so the oracle checks rather
 * than skips such a geometry. libr4d's shape hooks deliberately refuse it, since its launches read
 * RAD_P_INT only; the difference is intended. */
static int64_t gi(const RadParam* p, int n_p, const char* k, int64_t dflt) {
    const RadParam* q = rad_param_find(p, n_p, k);
    if (q && q->kind == RAD_P_STR && q->sval) return (int64_t)std::strtoll(q->sval, nullptr, 10);
    return (int64_t)rad_param_getdim(p, n_p, k, (long long)dflt);
}

static const char* gs(const RadParam* p, int n_p, const char* k) {
    return rad_param_gets(p, n_p, k, nullptr);
}

/* ================================================================== dtype names
 * The ABI's parser, so a dtype added to the ABI is understood here too. */
static uint32_t dt_parse(const char* n) { return rad_dtype_parse(n); }

/* The activation dtype: `dtype`, else attention's `q_dtype` / `kv_dtype`, else bf16. Some ops
 * (e.g. `rope`) carry no dtype parameter, and an f32 fallback would make a bf16-only kernel fail on
 * dtype. A quantisation-scheme string ("w4a8", "fp8a8") also lands on bf16. */
static uint32_t act_dt(const RadParam* p, int n_p) {
    const char* s = gs(p, n_p, "dtype");
    if (!s) s = gs(p, n_p, "q_dtype");
    if (!s) s = gs(p, n_p, "kv_dtype");
    const uint32_t dt = dt_parse(s);
    return dt == (uint32_t)RAD_DT_INVALID ? (uint32_t)RAD_BF16 : dt;
}

/* ================================================================== the operand list
 * Each function builds the whole list and hands out the requested entry, so the count can be
 * checked against the schema (see the header). */
enum { REF_MAX_OPD = 24 };   /* the widest schema here is gdn_conv_prep's sixteen */

struct Opd {
    RadOpdDesc v[REF_MAX_OPD];
    int        n = 0;
    bool       over = false;
    RadOpdDesc sink{};       /* somewhere for an overflowing push to land; `over` is the answer */

    RadOpdDesc* push() {
        if (n >= REF_MAX_OPD) { over = true; return &sink; }
        RadOpdDesc& d = v[n++];
        std::memset(&d, 0, sizeof d);
        d.fill = (uint32_t)RAD_FILL_NORMAL;
        /* -1 is the ABI's "draw"; set explicitly because a zeroed field means "the constant 0". */
        d.idx_const = -1;
        return &d;
    }

    static void shape(RadOpdDesc* d, uint32_t dt, std::initializer_list<int64_t> s) {
        d->dtype = dt;
        d->rank  = 0;
        for (int64_t e : s) {
            if (d->rank >= (uint32_t)RAD_MAX_RANK) break;
            d->shape[d->rank++] = e;
        }
    }

    /* An ordinary float operand: standard normals. */
    void t(uint32_t dt, std::initializer_list<int64_t> s) { shape(push(), dt, s); }

    /* A float operand whose domain is narrower than a normal draw (a sigmoid gate, a monotone
     * log-decay cumsum, a chunk-triangular factor). `chunk` is the reset width. */
    void t_fill(uint32_t dt, std::initializer_list<int64_t> s, uint32_t fill, int64_t chunk) {
        RadOpdDesc* d = push();
        shape(d, dt, s);
        d->fill = fill;
        d->fill_chunk = chunk;
    }

    /* An index operand of dtype `dt`. `max` is the exclusive upper bound of the draw; 0 means the
     * previous operand's leading extent. `konst` >= 0 makes every element that value. A dtype
     * other than i32 is for weights whose values are sizes or offsets (those are i64 in every
     * container and plugins refuse any other dtype) and for a byte plane of marks. */
    void t_idx_dt(uint32_t dt, std::initializer_list<int64_t> s, int64_t max, int64_t konst,
                  uint32_t flags) {
        RadOpdDesc* d = push();
        shape(d, dt, s);
        d->fill      = (uint32_t)RAD_FILL_INDEX;
        d->idx_max   = max;
        d->idx_const = konst;
        d->flags     = flags;
    }
    void t_idx(std::initializer_list<int64_t> s, int64_t max, int64_t konst, uint32_t flags) {
        t_idx_dt((uint32_t)RAD_I32, s, max, konst, flags);
    }

    /* An optional operand this description deliberately does not pass. */
    void skip() {
        RadOpdDesc* d = push();
        d->flags = (uint32_t)RAD_OPD_F_ABSENT;
    }

    int finish(const char* op, int operand, RadOpdDesc* out) const;
};

/* The operand count this plugin's schema declares for `op`, or -1 if the op is not ours. */
static int schema_operands(const char* op) {
    const int n = rad_kernel_schema_count();
    for (int i = 0; i < n; ++i) {
        const RadOpSchema* s = rad_kernel_schema_at(i);
        if (s && s->op && std::strcmp(s->op, op) == 0) return s->n_operands;
    }
    return -1;
}

int Opd::finish(const char* op, int operand, RadOpdDesc* out) const {
    if (over) return RAD_E_SHAPE;
    const int want = schema_operands(op);
    /* Description and schema disagree: a bug in one of them. */
    if (want != n) return RAD_E_SHAPE;
    if (operand < 0 || operand >= n || !out) return RAD_E_SHAPE;
    /* A non-positive extent means the geometry lacks a parameter this description needs: a
     * decline (the caller skips and counts), not a bug. */
    for (int i = 0; i < n; ++i) {
        if (v[i].flags & RAD_OPD_F_ABSENT) continue;
        if (v[i].rank == 0) return RAD_E_UNSUPPORTED;
        for (uint32_t d = 0; d < v[i].rank; ++d)
            if (v[i].shape[d] <= 0) return RAD_E_UNSUPPORTED;
    }
    *out = v[operand];
    return RAD_OK;
}

/* ================================================================== shared readings */

/* The scale-stream group count shared by `quant_act_i8g`, `quant_act_fp8` and `dequant`, matching
 * their launches (ref_quant.cpp): group <= 0 or wider than the row is one scale per row, otherwise
 * ceil(n / group). Not the `had_*` reading, where `group` is the rotation width and the scale is
 * per row. */
static int64_t scale_groups(int64_t n, int64_t group) {
    int64_t g = group;
    if (g <= 0 || g > n) g = n;
    return g > 0 ? (n + g - 1) / g : 0;
}

/* The gated-delta-net head counts. No gdn op has them as parameters (kernels read them off the
 * operands), so the family declines unless the caller passes `n_head_k` / `n_head_v`, taken from
 * the container's `linear_num_key_heads` / `linear_num_value_heads`. */
static bool gdn_heads(const RadParam* p, int n_p, int64_t* Hk, int64_t* Hv) {
    *Hk = gi(p, n_p, "n_head_k", 0);
    *Hv = gi(p, n_p, "n_head_v", 0);
    return *Hk > 0 && *Hv > 0;
}

/* Sequences in the described batch: 1 by default; a caller may pass `n_seq` for a ragged batch,
 * which the gdn and attention descriptions follow. */
static int64_t n_seq_of(const RadParam* p, int n_p) {
    const int64_t s = gi(p, n_p, "n_seq", 1);
    return s > 0 ? s : 1;
}

/* The sampler's per-row parameter plane, one RadSampleParams per sampled position
 * (abi/rad_sample.h): a byte blob viewed through a 32-bit dtype; this plugin and libr4d accept u32,
 * i32 or f32.
 *
 * Drawn with RAD_FILL_SAMPLE_PARAMS: each row is a request's settings, its flags agreeing with its
 * fields, its history window inside the history plane and its `mask_row` a row of the bitmask
 * plane, which the description passes as the fill's `fill_chunk` and `idx_max`. Row 0 switches
 * every stage on, so a one-row case does work in each; every other row switches each stage on or
 * leaves it at its identity at random, and a minority is greedy (rad_abi.h has the law). The
 * history repeats itself (RAD_OPD_F_IDX_HISTORY), so the frequency penalty counts tokens more than
 * once and DRY finds repeated suffixes to penalise.
 *
 * Described as f32: a tool that does not know the law draws normals, under which the float fields
 * land in plausible ranges and the narrowing stages still work on some rows. A word-typed normal
 * draw would make them denormals or NaN and `1 / temp` infinite.
 *
 * Word-typed operands here are i32, never u32: opshapes.cpp's rad_widen has no RAD_U32 case, so a
 * u32 operand would compare as zeroes and pass vacuously. */
static int64_t sample_params_words() {
    return (int64_t)(sizeof(RadSampleParams) / 4);   /* 96 bytes, 24 words */
}

}  /* namespace */

/* ================================================================== elementwise and norms */

int ref_rmsnorm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { n });             /* w */
    o.t(D, { M, n });          /* y */
    return o.finish("rmsnorm", operand, out);
}

int ref_rmsnorm_p_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, n });          /* y */
    return o.finish("rmsnorm_p", operand, out);
}

int ref_rmsnorm_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, n });          /* residual */
    o.t(D, { n });             /* w */
    o.t(D, { M, n });          /* y */
    o.t(D, { M, n });          /* residual_out */
    return o.finish("rmsnorm_add", operand, out);
}

/* The gated residual. `inject` and `group` default on (every trunk connection); 0 describes the
 * final mixer or a bf16-consumer read (see the schema in ref_registry.cpp). */
int ref_hc_read_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0), lr = gi(p, n_p, "lowrank", 0);
    const int64_t hn = hc * n;
    const int64_t inject = gi(p, n_p, "inject", 1);
    const int64_t group  = gi(p, n_p, "group", 0);
    const int64_t rotate = gi(p, n_p, "rotate", 0);
    Opd o;
    o.t(D, { M, hn });                          /* h */
    /* The gain is f32 in the container, not the activation dtype. */
    o.t((uint32_t)RAD_F32, { hn });             /* w */
    o.t(D, { lr, hn });                         /* mix_down */
    o.t(D, { hn, lr });                         /* mix_up */
    if (inject) o.t(D, { hc, hn }); else o.skip();   /* inject_w */
    o.t(D, { M, n });                           /* x */
    if (inject) o.t(D, { M, hc }); else o.skip();    /* inj */
    if (group > 0) {
        o.t((uint32_t)RAD_F8E4M3, { M, n });                       /* q */
        o.t((uint32_t)RAD_F32, { M, scale_groups(n, group) });     /* scale */
    } else {
        o.skip();
        o.skip();
    }
    if (group > 0 && rotate) {
        o.t((uint32_t)RAD_F8E4M3, { M, n });                       /* rq */
        o.t((uint32_t)RAD_F32, { M, scale_groups(n, group) });     /* rscale */
    } else {
        o.skip();
        o.skip();
    }
    return o.finish("hc_read", operand, out);
}

int ref_hc_enter_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, hc * n });     /* h */
    return o.finish("hc_enter", operand, out);
}

int ref_kda_update_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const char* form = gs(p, n_p, "state_form");
    // Replay records and aliased engine index columns require a hand-built fixture.
    if (form && !std::strcmp(form,"anchor")) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), H = gi(p, n_p, "heads", 0);
    const int64_t K = gi(p, n_p, "head_k", 0), V = gi(p, n_p, "head_v", 0);
    if (M < 1 || H < 1 || K < 1 || K > 128 || V < 1 || V > 128) return RAD_E_UNSUPPORTED;
    const char* mode = gs(p, n_p, "mode");
    const int64_t history = mode && !std::strcmp(mode, "verify") ? M : 1;
    Opd o;
    o.t(D, {M, H, K}); o.t(D, {M, H, K}); o.t(D, {M, H, V});
    o.t(D, {M, H, K}); o.t(D, {M, H});
    o.t((uint32_t)RAD_F32, {H}); o.t((uint32_t)RAD_F32, {H * K});
    o.t((uint32_t)RAD_F32, {history + 1, H, V, K});
    o.t_idx({1, history}, history + 1, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);
    o.t_idx({2}, M, -1, (uint32_t)RAD_OPD_F_IDX_CU);
    o.t_idx({1}, 0, 1, 0); o.t_idx({1}, 0, 1, 0);
    o.t(D, {M, H, V});
    return o.finish("kda_update", operand, out);
}

int ref_mhc_pre_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, {M, hc * n});
    o.t((uint32_t)RAD_F32, {hc * (hc + 2), hc * n});
    o.t((uint32_t)RAD_F32, {3});
    o.t((uint32_t)RAD_F32, {hc * (hc + 2)});
    o.t(D, {M, n});
    o.t((uint32_t)RAD_F32, {M, hc});
    o.t((uint32_t)RAD_F32, {M, hc * hc});
    return o.finish("mhc_pre", operand, out);
}

static int64_t mhc_part_stride(int64_t hc) { return (hc * (hc + 2) + 1 + 3) / 4 * 4; }

int ref_mhc_pre_partial_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    const int64_t chunk = gi(p, n_p, "chunk", 0), chunks = gi(p, n_p, "chunks", 0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8 || chunk <= 0 || (hc * n) % chunk || chunks <= 0)
        return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, {M, hc * n});
    o.t((uint32_t)RAD_F32, {hc * (hc + 2), hc * n});
    o.t((uint32_t)RAD_F32, {M, chunks * mhc_part_stride(hc)});
    return o.finish("mhc_pre_partial", operand, out);
}

int ref_mhc_pre_finish_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    const int64_t chunk = gi(p, n_p, "chunk", 0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8 || chunk <= 0 || (hc * n) % chunk)
        return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, {M, hc * n / chunk * mhc_part_stride(hc)});
    o.t(D, {M, hc * n});
    o.t((uint32_t)RAD_F32, {3});
    o.t((uint32_t)RAD_F32, {hc * (hc + 2)});
    o.t(D, {M, n});
    o.t((uint32_t)RAD_F32, {M, hc});
    o.t((uint32_t)RAD_F32, {M, hc * hc});
    return o.finish("mhc_pre_finish", operand, out);
}

int ref_mhc_post_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, {M, n}); o.t(D, {M, hc * n});
    o.t((uint32_t)RAD_F32, {M, hc}); o.t((uint32_t)RAD_F32, {M, hc * hc});
    o.t(D, {M, hc * n});
    return o.finish("mhc_post", operand, out);
}

int ref_mhc_contract_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    if (M <= 0 || n <= 0 || hc < 2 || hc > 8) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, {M, hc * n}); o.t(D, {M, n});
    return o.finish("mhc_contract", operand, out);
}

int ref_hc_write_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, n });          /* y */
    o.t(D, { M, hc });         /* inj */
    o.t(D, { M, hc * n });     /* h, in place */
    return o.finish("hc_write", operand, out);
}

int ref_hc_project_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, hc * n });                 /* h, in place */
    o.t((uint32_t) RAD_F32, { n });        /* v: one layer's unit direction */
    o.t((uint32_t) RAD_I32, { M });        /* on: 1 = this row takes the projection */
    return o.finish("hc_project", operand, out);
}

int ref_mtp_enter_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M  = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t hc = gi(p, n_p, "hc", 0);
    Opd o;
    o.t(D, { M, hc * n });                      /* h, the wide stream */
    o.t(D, { M, n });                           /* e, the next token's embedding */
    /* Both gains are f32 in the container, as hc_read's. */
    o.t((uint32_t)RAD_F32, { hc * n });         /* w_h */
    o.t((uint32_t)RAD_F32, { n });              /* w_e */
    o.t(D, { n, n });                           /* fc_h */
    o.t(D, { n, n });                           /* fc_e */
    o.t(D, { M, hc * n });                      /* x */
    return o.finish("mtp_enter", operand, out);
}

int ref_grid_embed_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), side = gi(p, n_p, "side", 0);
    if (M <= 0 || n <= 0 || side <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M, n });              /* x, INOUT */
    o.t(D, { side * side, n });    /* table */
    /* row, column, height, width, drawn over the table's side so some rows exercise the clamped
     * taps. */
    o.t_idx({ 4, M }, side, -1, 0);
    return o.finish("grid_embed", operand, out);
}

/* grid_embed's operands; the coordinates are drawn over the table's side, as there. */
int ref_grid_embed_aa_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), side = gi(p, n_p, "side", 0);
    if (M <= 0 || n <= 0 || side <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M, n });              /* x, INOUT */
    o.t(D, { side * side, n });    /* table */
    o.t_idx({ 4, M }, side, -1, 0);
    return o.finish("grid_embed_aa", operand, out);
}

int ref_layernorm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { n });             /* w */
    /* `b` is optional but passed: the biased form covers strictly more code. */
    o.t(D, { n });             /* b */
    o.t(D, { M, n });          /* y */
    return o.finish("layernorm", operand, out);
}

/* add / mul share a schema and a description. `b` is described at full extent; the row-broadcast
 * form is not described, since its buffers cannot be told from a shape bug. */
static int binary_shape(const RadParam* p, int n_p, int operand, const char* op, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* a */
    o.t(D, { M, n });          /* b */
    o.t(D, { M, n });          /* y */
    return o.finish(op, operand, out);
}

int ref_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, "add", out);
}
int ref_mul_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, "mul", out);
}

int ref_mul_weight_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    if (gi(p, n_p, "scalar", 0)) o.t(D, { 1 });   /* w -- the scalar form */
    else                          o.t(D, { n });   /* w -- one row */
    o.t(D, { M, n });          /* y */
    return o.finish("mul_weight", operand, out);
}

int ref_act_mul_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return binary_shape(p, n_p, operand, "act_mul", out);
}

int ref_seg_mean_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = n_seq_of(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* x */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t(D, { S, n });                                         /* y */
    return o.finish("seg_mean", operand, out);
}

int ref_pool_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t n = gi(p, n_p, "n", 0), k = gi(p, n_p, "k", 0);
    /* A band's bound need not be whole windows, but issued rows always are; round down. */
    int64_t M = gi(p, n_p, "M", 0);
    if (k > 0) M -= M % k;
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M / k, n });      /* y */
    return o.finish("pool_rows", operand, out);
}

int ref_relu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, n });          /* y */
    return o.finish("relu", operand, out);
}

int ref_clamp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { 1 });             /* lo */
    o.t(D, { 1 });             /* hi */
    o.t(D, { M, n });          /* y */
    return o.finish("clamp", operand, out);
}

/* The audio tower's segmented ops are described as one segment. `cu_seqlens` counts in units
 * `cu_div` times finer than the operand's rows (a subsampled layer reads the pass's frame
 * boundaries), so the rows are M / cu_div; a band that does not divide (into whole output rows, for
 * a strided convolution) is declined. */
int ref_conv2d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), W = gi(p, n_p, "width", 0);
    const int64_t cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    const int64_t kh = gi(p, n_p, "kh", 0), kw = gi(p, n_p, "kw", 0);
    const int64_t st = gi(p, n_p, "stride", 1), pad = gi(p, n_p, "pad", 0);
    const int64_t cdiv = gi(p, n_p, "cu_div", 1), vst = gi(p, n_p, "valid_stride", 1);
    if (M <= 0 || W <= 0 || st <= 0 || cdiv <= 0 || vst <= 0 || M % (cdiv * st))
        return RAD_E_UNSUPPORTED;
    const int64_t T = M / cdiv;
    const int64_t To = (T + 2 * pad - kh) / st + 1, Wo = (W + 2 * pad - kw) / st + 1;
    Opd o;
    o.t(D, { T, W * cin });                                   /* x */
    if (gi(p, n_p, "t_last", 0)) o.t(D, { cout, cin, kw, kh });   /* w, time last */
    else                         o.t(D, { cout, cin, kh, kw });   /* w */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t_idx({ T * vst }, 2, -1, 0);                           /* valid: 0 or 1 a row */
    o.t(D, { To, Wo * cout });                                /* y */
    o.t(D, { cout });                                         /* b */
    return o.finish("conv2d", operand, out);
}

int ref_conv2d_unfold_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), W = gi(p, n_p, "width", 0), cin = gi(p, n_p, "cin", 0);
    const int64_t kh = gi(p, n_p, "kh", 0), kw = gi(p, n_p, "kw", 0);
    const int64_t st = gi(p, n_p, "stride", 1), pad = gi(p, n_p, "pad", 0);
    const int64_t cdiv = gi(p, n_p, "cu_div", 1), vst = gi(p, n_p, "valid_stride", 1);
    if (M <= 0 || W <= 0 || st <= 0 || cdiv <= 0 || vst <= 0 || M % (cdiv * st))
        return RAD_E_UNSUPPORTED;
    const int64_t T = M / cdiv;
    const int64_t To = (T + 2 * pad - kh) / st + 1, Wo = (W + 2 * pad - kw) / st + 1;
    Opd o;
    o.t(D, { T, W * cin });                                   /* x */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t_idx({ T * vst }, 2, -1, 0);                           /* valid */
    o.t(D, { To * Wo, cin * kh * kw });                       /* y */
    return o.finish("conv2d_unfold", operand, out);
}

int ref_conv1d_dw_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "n", 0), K = gi(p, n_p, "k", 0);
    const int64_t cdiv = gi(p, n_p, "cu_div", 1);
    if (M <= 0 || cdiv <= 0 || M % cdiv) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M / cdiv, C });                                  /* x */
    o.t(D, { C, 1, K });                                      /* w */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M / cdiv, C });                                  /* y */
    o.skip();                                                 /* b */
    return o.finish("conv1d_dw", operand, out);
}

int ref_rel_sinusoid_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* y */
    return o.finish("rel_sinusoid", operand, out);
}

int ref_seg_sinusoid_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (gi(p, n_p, "cu_div", 1) != 1) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M, n });                                         /* y */
    return o.finish("seg_sinusoid", operand, out);
}

int ref_attn_local_rel_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), hd = gi(p, n_p, "head_dim", 0);
    const int64_t H = gi(p, n_p, "n_head", 0), L = gi(p, n_p, "left", 0);
    const int64_t cdiv = gi(p, n_p, "cu_div", 1);
    if (M <= 0 || cdiv <= 0 || M % cdiv || L < 0) return RAD_E_UNSUPPORTED;
    const int64_t T = M / cdiv;
    Opd o;
    o.t(D, { T, H * hd });                                    /* q */
    o.t(D, { T, H * hd });                                    /* k */
    o.t(D, { T, H * hd });                                    /* v */
    o.t(D, { L + 1, H * hd });                                /* relk */
    o.t(D, { hd });                                           /* pds */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { T, H * hd });                                    /* out */
    return o.finish("attn_local_rel", operand, out);
}

int ref_swiglu_clamped_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return ref_silu_mul_shape(p, n_p, operand, out);
}
int ref_router_sigmoid_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), ne = gi(p, n_p, "n_expert", 0),
                  k = gi(p, n_p, "top_k", 0);
    if (M < 1 || ne < 1 || ne > 512 || k < 1 || k > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });
    o.t(RAD_F32, { ne });
    o.t_idx({ M, k }, ne, -1, 0);
    o.t(D, { M, k });
    return o.finish("router_sigmoid_topk", operand, out);
}
int ref_mla_project_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), h = gi(p, n_p, "heads", 0), d = gi(p, n_p, "head_dim", 0),
                  l = gi(p, n_p, "latent_dim", 0);
    const char* mode = gs(p, n_p, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"),
               expand = mode && !std::strcmp(mode, "expand");
    if (M < 1 || h < 1 || d < 1 || l < 1 || (!absorb && !expand)) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(RAD_BF16, { M, h, absorb ? d : l });
    o.t(RAD_BF16, { h * 2 * d, l });
    o.t(RAD_BF16, { M, h, absorb ? l : d });
    return o.finish("mla_project", operand, out);
}
int ref_dsa_query_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M < 1 || n < 1 || n > 256 || (n & (n - 1))) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(RAD_BF16, { M, n });
    o.t(RAD_F8E4M3, { M, n });
    o.t(RAD_F32, { M, 1 });
    return o.finish("dsa_query", operand, out);
}
int ref_dsa_pool_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), ratio = gi(p, n_p, "ratio", 0);
    if (M < 1 || n < 1 || n > 256 || (n & (n - 1)) || ratio < 2 || ratio > 16)
        return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(RAD_BF16, { M * ratio, 2 * n });
    o.t(RAD_F32, { ratio, n });
    o.t(RAD_F8E4M3, { M, n });
    o.t(RAD_F32, { M, 1 });
    o.skip();
    o.skip();   // Optional paged cache writes use a hand-built unique page list.
    return o.finish("dsa_pool", operand, out);
}
int ref_dsa_score_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), H = gi(p, n_p, "heads", 0);
    const int64_t B = gi(p, n_p, "blocks", 0), S = gi(p, n_p, "seqs", 0), pages = S * B;
    if (M < 1 || n < 1 || H < 1 || H > 64 || B < 1 || S < 1 || S > M) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(RAD_F8E4M3, { M, H, n });
    o.t_fill(RAD_F32, { M, H }, RAD_FILL_SIGMOID, 0);
    o.t(RAD_F32, { M, H });
    o.t(RAD_F8E4M3, { pages, n });
    o.t_fill(RAD_F32, { pages }, RAD_FILL_SIGMOID, 0);
    o.t_idx({ S, B }, pages, -1, 0);
    o.t_idx({ S + 1 }, M, -1, RAD_OPD_F_IDX_CU);
    o.t_idx({ S }, 0, B, 0);
    o.t(RAD_F32, { M, B });
    return o.finish("dsa_score", operand, out);
}
int ref_mla_attn_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), D = gi(p, n_p, "latent_dim", 0),
                  H = gi(p, n_p, "heads", 0);
    const int64_t B = gi(p, n_p, "blocks", 0), bs = gi(p, n_p, "block_size", 0),
                  pages = std::max((int64_t)2, B * M);
    if (M < 1 || D < 1 || D > 2048 || H < 1 || B < 1 || bs < 1) return RAD_E_UNSUPPORTED;
    const uint32_t dtype = act_dt(p, n_p);
    Opd o;
    o.t(dtype, { M, H, D });
    o.t(dtype, { pages, 1, bs, D });
    o.t_idx({ M, B }, pages, -1, 0);
    o.t_idx({ M }, 0, B * bs, 0);
    o.t(dtype, { M, H, D });
    return o.finish("mla_attn", operand, out);
}
int ref_silu_mul_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, 2 * n });      /* gate_up -- the gate is the FIRST half */
    o.t(D, { M, n });          /* y */
    return o.finish("silu_mul", operand, out);
}

int ref_gelu_mul_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, 2 * n });      /* gate_up -- the gate is the FIRST half */
    o.t(D, { M, n });          /* y */
    return o.finish("gelu_mul", operand, out);
}

static int unary_shape(const RadParam* p, int n_p, int operand, const char* op, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { M, n });          /* y */
    return o.finish(op, operand, out);
}

int ref_gelu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "gelu", out);
}
int ref_silu_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "silu", out);
}
int ref_sigmoid_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "sigmoid", out);
}

int ref_fill_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });          /* y */
    return o.finish("fill", operand, out);
}

int ref_logit_softcap_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "logit_softcap", out);
}
int ref_scale_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                /* x */
    o.t(D, { 1 });                                   /* s -- one scalar, broadcast over everything */
    o.t(D, { M, n });                                /* y */
    return o.finish("scale", operand, out);
}
int ref_softmax_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return unary_shape(p, n_p, operand, "softmax", out);
}

int ref_gather_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    /* The source is 4x the gathered height so the gather is not the identity (which would pass
     * while broken); `idx` draws over those 4M rows. */
    o.t(D, { 4 * M, n });                                  /* x */
    o.t_idx({ M }, 0, -1, 0);                              /* idx */
    o.t(D, { M, n });                                      /* y */
    return o.finish("gather_rows", operand, out);
}


/* scatter_rows. The destination is 4x the scattered height: onto exactly M rows a scatter is a
 * permutation, which a kernel ignoring `idx` could pass. Unnamed rows must come back unchanged. */
int ref_scatter_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                                      /* v */
    o.t_idx({ M }, 4 * M, -1, RAD_OPD_F_IDX_UNIQUE);       /* idx, distinct destinations */
    o.t(D, { 4 * M, n });                                  /* x, inout */
    return o.finish("scatter_rows", operand, out);
}
/* PLE's gate. `v` is the one narrow operand (n wide against hc*n): a single value row gated
 * differently into each of the hc streams. */
int ref_ple_gate_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D  = act_dt(p, n_p);
    const int64_t M   = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), hc = gi(p, n_p, "hc", 0);
    const int64_t hn  = hc * n;
    Opd o;
    o.t(D, { M, hn });        /* k */
    o.t(D, { M, hn });        /* q */
    o.t(D, { M, n });         /* v */
    /* The three gains are f32 in the container, as hc_read's. */
    o.t((uint32_t)RAD_F32, { hn });   /* w_key */
    o.t((uint32_t)RAD_F32, { hn });   /* w_query */
    o.t((uint32_t)RAD_F32, { hn });   /* w_conv */
    o.t(D, { M, hn });        /* gv */
    o.t(D, { M, hn });        /* gvn */
    return o.finish("ple_gate", operand, out);
}

/* PLE's dilated convolution, described as a decode step: `num_accepted` present, so the state
 * must be (width-1)*dilation - 1 + M deep for the shifted window (ref_ple.cpp), as the manager's
 * (width-1)*dilation + n_spec is for a 1 + n_spec step.
 *
 * Warm and read at the deepest offset (`num_accepted` = M), so the offset arithmetic and shifted
 * history are exercised. One sequence. */
int ref_ple_conv_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D  = act_dt(p, n_p);
    const int64_t M   = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t wid = gi(p, n_p, "width", 0), dil = gi(p, n_p, "dilation", 1);
    if (M <= 0 || n <= 0 || wid <= 0 || dil <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = 1, hist = (wid - 1) * dil, depth = hist - 1 + M;
    Opd o;
    o.t(D, { M, n });                                /* x   -- gvn */
    o.t(D, { n, wid });                              /* w */
    o.t(D, { S, n, depth });                         /* conv_state, INOUT */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);           /* cu */
    o.t(D, { M, n });                                /* resid -- gv */
    o.t_idx({ S }, S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);           /* cache_idx */
    /* Warm, stated explicitly rather than relying on a null `has_init` default. */
    o.t_idx({ S }, 0, 1, 0);                         /* has_init */
    o.t_idx({ S }, 0, M, 0);                         /* num_accepted: read at M - 1 */
    o.t(D, { M, n });                                /* y */
    return o.finish("ple_conv", operand, out);
}

/* ple_conv's description without `resid` and at dilation one. */
int ref_causal_conv1d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D  = act_dt(p, n_p);
    const int64_t M   = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), wid = gi(p, n_p, "width", 0);
    if (M <= 0 || n <= 0 || wid <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = 1, depth = wid - 2 + M;
    Opd o;
    o.t(D, { M, n });                                /* x */
    o.t(D, { n, wid });                              /* w */
    o.t(D, { S, n, depth });                         /* conv_state, INOUT */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);           /* cu */
    o.t_idx({ S }, S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);           /* cache_idx */
    o.t_idx({ S }, 0, 1, 0);                         /* has_init: warm */
    o.t_idx({ S }, 0, M, 0);                         /* num_accepted: read at M - 1 */
    o.t(D, { M, n });                                /* y */
    return o.finish("causal_conv1d", operand, out);
}

/* PLE's n-gram ids. The three i64 weights are not index-like: `mult` is ~10^13, `vocab_sizes` a
 * prime near 2*10^7 and `offsets` its prefix sum. Index-range draws would give a modulo by a small
 * number, landing every id in the first rows of the table.
 *
 * `tok` and `state` are unbounded index operands (a token id is bounded by the vocabulary, which
 * this op never sees); drawing over the whole i32 range also exercises the wrapping multiply. */
int ref_ngram_ids_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M     = gi(p, n_p, "M", 0);
    const int64_t heads = gi(p, n_p, "heads", 0);
    const int64_t ngram = gi(p, n_p, "ngram", 0);
    const int64_t eos   = gi(p, n_p, "eos", 0);
    const int64_t ctx   = ngram > 0 ? ngram - 1 : 0;
    /* One sequence, a decode step as for ple_conv: `num_accepted` present, a warm window read at
     * the deepest offset, and a state `ctx - 1 + M` deep for the shifted window. */
    const int64_t S = 1, depth = ctx - 1 + M;
    Opd o;
    o.t_idx({ M }, 0, -1, 0);                                        /* tok */
    o.t_idx({ S, depth }, 0, -1, 0);                                 /* state, INOUT */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);           /* cu */
    o.t((uint32_t)RAD_I64, { ngram });                               /* mult */
    /* Not normal draws: the launch refuses `v <= 0 || off[h] < 0`, which a normal cast to i64
     * fails on most heads, and the refusal would look like an unserved geometry.
     *
     * The vocabulary size is a constant eos + 1 (positive, and the vocabulary must contain eos).
     * The offsets are drawn non-negative over [0, heads); they give the heads of one n-gram block,
     * which share `mixed`, different answers. */
    o.t_idx_dt((uint32_t)RAD_I64, { heads }, 0, eos + 1, 0);         /* vocab_sizes */
    o.t_idx_dt((uint32_t)RAD_I64, { heads }, 0, -1, 0);              /* offsets */
    o.t_idx({ S }, S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);           /* cache_idx */
    o.t_idx({ S }, 0, 1, 0);                                         /* has_init: warm */
    o.t_idx({ S }, 0, M, 0);                                         /* num_accepted: at M - 1 */
    o.t((uint32_t)RAD_I32, { M, heads });                            /* ids */
    o.skip();                                                        /* ahead_ids */
    return o.finish("ngram_ids", operand, out);
}

int ref_cast_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    /* Operand dtypes come from `from` and `to`; an unparseable spelling falls back to f32, the
     * widest it could mean. */
    uint32_t from = dt_parse(gs(p, n_p, "from"));
    uint32_t to   = dt_parse(gs(p, n_p, "to"));
    if (from == (uint32_t)RAD_DT_INVALID) from = (uint32_t)RAD_F32;
    if (to   == (uint32_t)RAD_DT_INVALID) to   = (uint32_t)RAD_F32;
    Opd o;
    o.t(from, { M, n });       /* x */
    o.t(to,   { M, n });       /* y */
    return o.finish("cast", operand, out);
}

/* ================================================================== quantisation and rotation */

/* `asum` is described whenever `group` divides n, so a case checks the row sums as well. */
int ref_quant_act_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), g = gi(p, n_p, "group", 0);
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_I8, { M, n });        /* q, in the step order */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale */
    if (g > 0 && n % g == 0)
        o.t((uint32_t)RAD_F32, { n / g, (M + 15) / 16 * 16 });   /* asum, group-major */
    else
        o.skip();
    return o.finish("quant_act_i8", operand, out);
}

int ref_quant_act_i8g_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    return o.finish("quant_act_i8g", operand, out);
}

int ref_quant_act_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    /* `dtype` names the input; the output dtype is in the op name. */
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale, a dequant multiplier (amax / 448) */
    return o.finish("quant_act_fp8", operand, out);
}

int ref_quant_cols_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* y */
    o.t((uint32_t)RAD_F32, { n });          /* scale, one a column (amax / 448) */
    return o.finish("quant_cols_fp8", operand, out);
}

/* `n` is K on both axes of the accumulator: h is [n, n], not [M, n]. */
int ref_gram_accum_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const char* dt = gs(p, n_p, "dtype");
    Opd o;
    if (dt && std::strcmp(dt, "bf16") == 0) {
        o.t((uint32_t)RAD_BF16, { M, n });  /* a, read as it stands */
        o.skip();                           /* a_scale: a plain activation has none */
    } else {
        const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
        o.t((uint32_t)RAD_F8E4M3, { M, n });    /* a */
        o.t((uint32_t)RAD_F32, { M, ng });      /* a_scale, a dequant multiplier (amax / 448) */
    }
    o.t((uint32_t)RAD_F32, { n, n });       /* h, read and added to */
    return o.finish("gram_accum", operand, out);
}

/* Declined: `pairs` must pair each group's columns (every column once per stage, partner in the
 * group), and a drawn index operand has repeats, so it would not be a rotation. r4d_selftest builds
 * real pairings. */
int ref_givens_dense_quant_shape(const RadParam*, int, int, RadOpdDesc*) {
    return RAD_E_UNSUPPORTED;               /* a composed rotation cannot be drawn: it is a weight, built from real pairings */
}

int ref_givens_rotate_shape(const RadParam*, int, int, RadOpdDesc*) {
    return RAD_E_UNSUPPORTED;               /* a pairing cannot be drawn: a described operand can only be drawn, and an index draw has repeats */
}

/* The same extents as quant_act_fp8: `group` is both rotation width and scale granularity here,
 * unlike the int8 rotated forms, so the rotation changes the values, not the shapes. */
int ref_had_quant_act_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale, a dequant multiplier (amax / 448) */
    return o.finish("had_quant_act_fp8", operand, out);
}

/* quant_act_i8g's extents: the rotation changes the values, not the shapes. */
int ref_had_quant_act_i8g_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale (amax / 127) */
    return o.finish("had_quant_act_i8g", operand, out);
}

/* The int8 rotated forms: `group` is the Walsh-Hadamard rotation width (libr4d's `had`) and the
 * scale is one per row, carrying the 1/sqrt(group) normalisation (ref_quant.cpp). Describing it as
 * [M, n/group] would leave the tail of every scale buffer unwritten by both sides, comparing two
 * zero fills. */

int ref_had_quant_act_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale -- one per row */
    return o.finish("had_quant_act_i8", operand, out);
}

int ref_rmsnorm_had_quant_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                       /* x */
    /* `residual` passed, so the fused_add_rms_norm rounding (the norm reads the narrowed sum
     * written back through the INOUT operand) is exercised. */
    o.t(D, { M, n });                       /* residual, INOUT */
    o.t(D, { n });                          /* w */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale -- one per row */
    /* `out_bf16` (post-norm, pre-rotation) is written by both plugins when present, so described. */
    o.t(D, { M, n });                       /* out_bf16 */
    return o.finish("rmsnorm_had_quant_i8", operand, out);
}

int ref_gated_had_quant_i8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    /* The packed form the MLP issues: `b` absent, `a` [M, 2n] with the gate first. */
    o.t(D, { M, 2 * n });                   /* a */
    o.skip();                               /* b */
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, 1 });       /* scale -- one per row */
    return o.finish("gated_had_quant_i8", operand, out);
}

int ref_dequant_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t((uint32_t)RAD_I8, { M, n });        /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t(D, { M, n });                       /* y */
    return o.finish("dequant", operand, out);
}

/* ================================================================== gemm */

int ref_gemm_nt_f32_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    if (M < 1 || N < 1 || K < 1) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(RAD_BF16, { M, K });
    o.t(RAD_F32, { N, K });
    o.t(RAD_F32, { M, N });
    return o.finish("gemm_nt_f32", operand, out);
}
int ref_gemm_nt_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    Opd o;
    o.t(D, { M, K });          /* a */
    o.t(D, { N, K });          /* b -- stored transposed, so both operands are k-contiguous */
    o.t(D, { M, N });          /* y */
    return o.finish("gemm_nt", operand, out);
}

int ref_gemm_nt_bias_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    Opd o;
    o.t(D, { M, K });          /* a */
    o.t(D, { N, K });          /* b */
    o.t(D, { N });             /* bias */
    /* `res` is optional but passed, to cover the residual epilogue. */
    o.t(D, { M, N });          /* res */
    o.t(D, { M, N });          /* y */
    return o.finish("gemm_nt_bias", operand, out);
}

/* Block-scaled fp8, the symmetric int4 and int8 weights against quant_act_i8's activation, the
 * int8 trunk (`i8a8`) and OCP-MXFP4 (`mxfp4a8`). The zero-point grids (`w2a8`, `w4a8_asym`,
 * `w4a16_asym`, `w5a16`, `w4a8i8_asym`, `w5a8i8`) read their scale and zero joined in one operand
 * (ref_planes.h), which a description cannot state; they and any other format are declined rather
 * than guessed.
 *
 * The fp8 grid is stated by the checkpoint: E4M3 [N][K] against a bf16 [ceil(N/128)][ceil(K/128)]
 * plane, `group` carrying the 128. w4a8 and w8a8 are the affine encoding's two planes, codes and
 * an f16 scale per (row, `group` of K), against int8 codes in the step order with one f32 scale a
 * row. `a_sum` (asymmetric grids) is absent, and so is `b_ref` except at mxfp4a8. Activation
 * scales are drawn in (0, 1): a quantiser's are never negative, and an MXFP4 kernel treats a
 * block whose scale is not positive as empty (docs/OPS.md). */
static int gemm_q_shape(const RadParam* p, int n_p, int operand, bool bias, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const int64_t group = gi(p, n_p, "group", 0);
    const char* q = gs(p, n_p, "dtype");
    auto is = [&](const char* s) { return q && std::strcmp(q, s) == 0; };
    const bool f32s = is("fp8a8f32");
    const bool a8 = is("fp8a8") || f32s, a16 = is("fp8a16"), w4 = is("w4a8"), w8 = is("w8a8");
    /* i8a8h is i8a8's operands; its rotation is in the values. */
    const bool i8 = is("i8a8") || is("i8a8h"), mx = is("mxfp4a8");
    if ((!a8 && !a16 && !w4 && !w8 && !i8 && !mx) || N <= 0 || K <= 0 || group <= 0)
        return RAD_E_UNSUPPORTED;
    /* Neither has a bias form in any architecture. */
    if (bias && (i8 || mx)) return RAD_E_UNSUPPORTED;
    /* The bias form requires `a_scale`, so the fp8a16 reading (bf16 activation, no scale) cannot
     * be expressed there. */
    if (bias && a16) return RAD_E_UNSUPPORTED;

    const int64_t kb = (K + group - 1) / group;
    const int64_t nb = (N + group - 1) / group;
    Opd o;
    if (i8 || mx) {
        /* i8a8: quant_act_i8g's row-major codes and f32 scale per (row, `group` of K) against int8
         * codes with a bf16 scale per (row, `group` of K). mxfp4a8: quant_act_fp8's pair at 128
         * against e2m1 codes with an E8M0 exponent per (row, `group` = 32 of K); `b_ref` is the same
         * exponent plane, which an mxfp4 kernel reduces to a per-row reference at load. */
        const int64_t akb = mx ? (K + 127) / 128 : kb;
        o.t((uint32_t)(mx ? RAD_F8E4M3 : RAD_I8), { M, K });          /* a */
        o.t_fill((uint32_t)RAD_F32, { M, akb }, (uint32_t)RAD_FILL_SIGMOID, 0);  /* a_scale */
        o.t((uint32_t)(mx ? RAD_FP4E2M1 : RAD_I8), { N, K });          /* b */
        o.t((uint32_t)(mx ? RAD_E8M0 : RAD_BF16), { N, kb });          /* b_scale */
        o.t((uint32_t)RAD_BF16, { M, N });                             /* y */
        o.skip();                                                      /* a_sum */
        if (mx) o.t((uint32_t)RAD_E8M0, { N, kb });                    /* b_ref */
        else    o.skip();
        return o.finish("gemm_nt_q", operand, out);
    }
    if (w4 || w8) {
        o.t((uint32_t)RAD_I8, { M, K });            /* a, in the step order */
        o.t_fill((uint32_t)RAD_F32, { M, 1 }, (uint32_t)RAD_FILL_SIGMOID, 0);     /* a_scale */
        o.t((uint32_t)(w4 ? RAD_I4 : RAD_I8), { N, K });   /* b */
        o.t((uint32_t)RAD_F16, { N, kb });          /* b_scale */
    } else {
        if (a8) {
            o.t((uint32_t)RAD_F8E4M3, { M, K });    /* a */
            o.t_fill((uint32_t)RAD_F32, { M, kb }, (uint32_t)RAD_FILL_SIGMOID, 0);  /* a_scale */
        } else {
            o.t((uint32_t)RAD_BF16, { M, K });      /* a */
            o.skip();                               /* a_scale */
        }
        o.t((uint32_t)RAD_F8E4M3, { N, K });        /* b -- the checkpoint's own bytes */
        /* b_scale -- weight_scale_inv, per 128x128; f32 at fp8a8f32 */
        o.t(f32s ? (uint32_t)RAD_F32 : (uint32_t)RAD_BF16, { nb, kb });
    }
    if (bias) {
        o.t((uint32_t)RAD_BF16, { N });             /* bias */
        o.t((uint32_t)RAD_BF16, { M, N });          /* y */
        return o.finish("gemm_nt_q_bias", operand, out);
    }
    o.t((uint32_t)RAD_BF16, { M, N });              /* y */
    o.skip();                                       /* a_sum */
    o.skip();                                       /* b_ref */
    return o.finish("gemm_nt_q", operand, out);
}

int ref_gemm_nt_q_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gemm_q_shape(p, n_p, operand, false, out);
}
int ref_gemm_nt_q_bias_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return gemm_q_shape(p, n_p, operand, true, out);
}

/* Runtime LoRA. `slot` is drawn in [0, S], S itself being a base row one past the pool, so a cold
 * draw covers the rows neither op may touch as well as the adapter rows. A draw never yields -1,
 * the other spelling of a base row, which is a hand case (tests/lora_ops_test.cpp, ref_test). A
 * single row is drawn in [0, S): a one-row call whose row drew S would write nothing, which
 * rad-avx-check refuses as vacuous. The scale stream exists only beside a coded activation; a float
 * `a` is read as it stands, as gemm_nt_q's fp8a16 form reads it. `group` defaults to 128, the
 * block every block-scaled checkpoint and quantiser here uses, when the geometry states none. */
int ref_lora_shrink_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), K = gi(p, n_p, "K", 0);
    const int64_t R = gi(p, n_p, "R", 0), S = gi(p, n_p, "S", 0);
    int64_t group = gi(p, n_p, "group", 0);
    const bool coded = D == (uint32_t)RAD_F8E4M3 || D == (uint32_t)RAD_F8E5M2 ||
                       D == (uint32_t)RAD_I8;
    if (coded && group <= 0) group = 128;
    Opd o;
    o.t(D, { M, K });                                           /* a */
    if (coded) o.t((uint32_t)RAD_F32, { M, (K + group - 1) / group });   /* a_scale */
    else o.skip();                                              /* a_scale */
    o.t_idx({ M }, M > 1 ? S + 1 : S, -1, 0);                   /* slot, S = base */
    o.t((uint32_t)RAD_BF16, { S, R, K });                       /* lora_a */
    o.t((uint32_t)RAD_F32, { M, R });                           /* t */
    return o.finish("lora_shrink", operand, out);
}

int ref_lora_expand_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0);
    const int64_t R = gi(p, n_p, "R", 0), S = gi(p, n_p, "S", 0);
    Opd o;
    o.t((uint32_t)RAD_F32, { M, R });                           /* t */
    o.t_idx({ M }, M > 1 ? S + 1 : S, -1, 0);                   /* slot, S = base */
    o.t((uint32_t)RAD_BF16, { S, N, R });                       /* lora_b */
    o.t((uint32_t)RAD_F32, { S, N });                           /* scale */
    o.t(D, { M, N });                                           /* y */
    return o.finish("lora_expand", operand, out);
}

int ref_lora_expand_mm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return ref_lora_expand_shape(p, n_p, operand, out);
}

/* ================================================================== attention
 *
 * The paged layout (ref_ops.h):
 *
 *     kv_cache[num_blocks, kv_heads, block_size, 2 * head_dim]   K first, V second in a slot
 *
 * The block count and context length are this description's choice; any cache large enough for
 * the indices will do. */

int ref_attn_paged_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    /* `q_len` is the ranged parameter, a query length (one row at decode, thousands in a prefill
     * chunk; different kernels, docs/OPS.md). */
    const int64_t M   = gi(p, n_p, "q_len", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 16);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    const int64_t nh  = gi(p, n_p, "n_head", gqa);
    if (M <= 0 || hd <= 0 || bs <= 0 || gqa <= 0 || nh <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
    const int64_t kvh = nh / gqa;

    /* Context depth, not the query length. By default one sequence whose context is exactly its
     * query (a prefill chunk's case), which never exercises deep paging, KV splits or sliding
     * windows. A harness sweeping depth passes `ctx` (no op declares it; kernels read the context
     * from `seqused`). It is at least the query length, or causal attention would read a
     * nonexistent prefix. */
    const int64_t S = 1;
    int64_t ctx = gi(p, n_p, "ctx", 0);
    if (ctx < M) ctx = M;
    const int64_t max_blocks = (ctx + bs - 1) / bs;

    /* `q_dtype` and `kv_dtype` are read separately: the engine issues fp8 caches beside bf16
     * queries. */
    const uint32_t Q  = act_dt(p, n_p);
    const uint32_t kv = dt_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? Q : kv;

    Opd o;
    o.t(Q,  { S * M, nh, hd });                      /* q */
    o.t(KV, { max_blocks, kvh, bs, 2 * hd });        /* kv_cache */
    o.t_idx({ S, max_blocks }, max_blocks, -1, 0);   /* block_table -- indexes blocks, may repeat */
    /* A constant, not a draw (see the file header). */
    o.t_idx({ S }, 0, ctx, 0);                       /* seqused */
    o.skip();                                        /* k_descale -- no per-(seq,head) descales */
    o.skip();                                        /* v_descale */
    /* `cu_seqlens` absent: the rectangular reading, the only one at one sequence. Present, it
     * would have to agree with a tool's `M` rather than this description's. */
    o.skip();                                        /* cu_seqlens */
    o.t(Q,  { S * M, nh, hd });                      /* out */
    return o.finish("attn_paged", operand, out);
}

/* attn_paged over a candidate tree: attn_paged's operands and one mask a node row, the rows being
 * `q_len` nodes that follow the committed prefix (root_pos rows of it) in the cache. A mask is a
 * topology, not a drawable tensor, so the description fills it with the root alone and ref_test
 * writes its real cases out (see `undrawable`). */
int ref_attn_paged_tree_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M    = gi(p, n_p, "q_len", 0);
    const int64_t root = gi(p, n_p, "root_pos", -1);
    if (root < 0) return RAD_E_UNSUPPORTED;
    if (operand == 8) {
        Opd o;
        o.t_idx({ M }, 0, 1, 0);                        /* ancestors: the root alone */
        return o.finish("attn_paged_tree", operand, out);
    }
    /* Operands 0..7 are attn_paged's, at the same geometry: the nodes are cache rows root_pos.., and
     * whoever asks for a tree passes a `ctx` that reaches them. */
    return ref_attn_paged_shape(p, n_p, operand, out);
}

int ref_attn_dense_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    if (M <= 0 || hd <= 0 || gqa <= 0) return RAD_E_UNSUPPORTED;
    /* A segment read in units finer than the rows has no [0, ..., T] description. */
    if (gi(p, n_p, "cu_div", 1) != 1) return RAD_E_UNSUPPORTED;
    /* The head count is not in the schema (`gqa` is a ratio; the count is read off `q`), so this
     * picks `gqa` query heads over one KV head, the smallest geometry exercising the grouping. */
    const int64_t nh = gqa, kvh = 1;
    const uint32_t D = act_dt(p, n_p);
    const int64_t S = n_seq_of(p, n_p);

    Opd o;
    o.t(D, { M, nh, hd });                           /* q */
    o.t(D, { M, kvh, hd });                          /* k */
    o.t(D, { M, kvh, hd });                          /* v */
    /* [0, ..., T], the batch's shape, not a draw. */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* cu_seqlens */
    o.t(D, { M, nh, hd });                           /* out */
    return o.finish("attn_dense", operand, out);
}

int ref_attn_cross_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    if (M <= 0 || hd <= 0 || gqa <= 0) return RAD_E_UNSUPPORTED;
    const int64_t nh = gqa, kvh = 1;
    const uint32_t D = act_dt(p, n_p);
    const int64_t S = n_seq_of(p, n_p);

    /* As many key rows as query rows, each side bounded by its own [0, ..., T]. */
    Opd o;
    o.t(D, { M, nh, hd });                           /* q */
    o.t(D, { M, kvh, hd });                          /* k */
    o.t(D, { M, kvh, hd });                          /* v */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* cu_seqlens */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* cu_k */
    o.t(D, { M, nh, hd });                           /* out */
    return o.finish("attn_cross", operand, out);
}

int ref_table_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), L = gi(p, n_p, "L", 0);
    if (M <= 0 || n <= 0 || L <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M, n });          /* x */
    o.t(D, { L, n });          /* t */
    o.t(D, { M, L * n });      /* y */
    return o.finish("table_add", operand, out);
}

int ref_mix_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), k = gi(p, n_p, "k", 0);
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M * k, n });      /* x */
    o.t(D, { k });             /* w */
    o.t(D, { M, n });          /* y */
    return o.finish("mix_rows", operand, out);
}

int ref_chan_stats_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(D, { M, n });                      /* x */
    o.t(D, { n });                         /* mean */
    o.t_fill(D, { n }, (uint32_t)RAD_FILL_SIGMOID, 0);   /* s -- a deviation or variance, > 0 */
    o.t(D, { M, n });                      /* y */
    return o.finish("chan_stats", operand, out);
}

int ref_kv_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t kvh = gi(p, n_p, "n_head_kv", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 0);
    if (M <= 0 || hd <= 0 || kvh <= 0 || bs <= 0) return RAD_E_UNSUPPORTED;
    /* Two blocks of slack so the slot mapping is not a dense prefix and the block index matters. */
    const int64_t nb = (M + bs - 1) / bs + 2;
    /* Only the cache takes `kv_dtype`: k and v are bf16 activations whatever the cache stores
     * (act_dt would fall back on kv_dtype and describe an fp8 k). */
    const uint32_t D  = (uint32_t)RAD_BF16;
    const uint32_t kv = dt_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? D : kv;

    Opd o;
    o.t(D, { M, kvh, hd });                          /* k */
    o.t(D, { M, kvh, hd });                          /* v */
    /* A scatter destination: distinct (see the file header), over the whole cache in slots. */
    o.t_idx({ M }, nb * bs, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* slot_mapping */
    o.t(KV, { nb, kvh, bs, 2 * hd });                /* kv_cache, INOUT */
    return o.finish("kv_store", operand, out);
}

int ref_rope_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t nh  = gi(p, n_p, "n_head", 0);
    const int64_t nkv = gi(p, n_p, "n_head_kv", nh);
    if (M <= 0 || hd <= 0 || nh <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);   /* no `dtype` on this op: bf16, see act_dt */
    Opd o;
    /* The fused row: n_head q heads, then n_head_kv each of k and v, head_dim wide. */
    o.t(D, { M, (nh + 2 * nkv) * hd });              /* qkv, INOUT */
    /* A multi-component mode with sections takes one position per component, component-major:
     * the section count, or three for imrope. */
    const char* mode = gs(p, n_p, "mode");
    if (!mode) mode = "neox";
    const char* sec  = gs(p, n_p, "sections");
    int64_t nc = 0;
    for (const char* q = sec; q && *q; ) {
        char* e = nullptr;
        (void)std::strtoll(q, &e, 10);
        if (e == q) break;
        ++nc;
        q = e;
    }
    const bool multi = nc > 0 && (!std::strcmp(mode, "mrope") || !std::strcmp(mode, "imrope") ||
                                  !std::strcmp(mode, "axial"));
    if (multi) o.t_idx({ !std::strcmp(mode, "imrope") ? 3 : nc, M }, M, -1, 0);
    else       o.t_idx({ M }, 0, -1, 0);             /* positions */
    return o.finish("rope", operand, out);
}

int ref_qk_norm_rope_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t nh  = gi(p, n_p, "n_head", 0);
    const int64_t nkv = gi(p, n_p, "n_head_kv", nh);
    if (M <= 0 || hd <= 0 || nh <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    /* The contiguous fused [q|k|v] row, as for `rope`. (libr4d's `qk_norm_rope_gate` reads an
     * interleaved q/gate view that parameters cannot describe; opshapes.h declines it.) */
    o.t(D, { M, (nh + 2 * nkv) * hd });              /* qkv, INOUT */
    /* `blocks` takes positions [C, M], component-major, C one past the largest `comps` entry (the
     * section count without `comps`); `axial` one component a section. Without a rotation
     * (`rotary_dim` 0) the positions are not passed. */
    const char* mode = gs(p, n_p, "mode");
    const bool blocks = mode && !std::strcmp(mode, "blocks");
    const bool axial = mode && !std::strcmp(mode, "axial");
    if (gi(p, n_p, "rotary_dim", hd) == 0) {
        o.skip();                                    /* positions */
    } else if (blocks || axial) {
        int64_t nc = 0;
        for (const char* q = gs(p, n_p, "sections"); q && *q; ) {
            char* e = nullptr;
            (void)std::strtoll(q, &e, 10);
            if (e == q) break;
            ++nc;
            q = e;
        }
        if (const char* cs = blocks ? gs(p, n_p, "comps") : nullptr) {
            int64_t mx = -1;
            for (const char* q = cs; *q; ) {
                char* e = nullptr;
                const long long x = std::strtoll(q, &e, 10);
                if (e == q) break;
                mx = x > mx ? x : mx;
                q = e;
            }
            nc = mx + 1;
        }
        if (nc <= 0) return RAD_E_UNSUPPORTED;
        o.t_idx({ nc, M }, M, -1, 0);                /* positions */
    } else {
        o.t_idx({ M }, 0, -1, 0);                    /* positions */
    }
    o.t(D, { hd });                                  /* q_w -- one gain per head channel */
    o.t(D, { hd });                                  /* k_w */
    return o.finish("qk_norm_rope", operand, out);
}

int ref_rope_table_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "M", 0);
    const int64_t rot = gi(p, n_p, "rot", 0);
    if (M <= 0 || rot < 2 || (rot & 1)) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    /* Each position writes its own row of a [max_ctx, rot] plane and max_ctx is not a parameter,
     * so the table is M rows with positions drawn from [0, M). The bound must be explicit:
     * `positions` is the first operand, so the default (previous operand's extent) does not apply. */
    o.t_idx({ M }, M, -1, 0);                        /* positions */
    o.t(D, { M, rot });                              /* cos_sin -- cos in the first half, sin in the second */
    return o.finish("rope_table", operand, out);
}

int ref_qsa_block_key_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    const int64_t n = gi(p, n_p, "n", 0);
    const int64_t r = gi(p, n_p, "ratio", 0);
    if (M <= 0 || n <= 0 || r <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M * r, n });                            /* k -- `ratio` rows a block */
    o.t((uint32_t)RAD_F32, { n });                   /* w -- the norm gain */
    /* page: absent, so the write is contiguous; a drawn page list could repeat and race. */
    o.skip();
    o.t_idx({ M }, M * r, -1, 0);                    /* pos -- each block's first-token position */
    o.t(D, { M, n });                                /* out */
    return o.finish("qsa_block_key", operand, out);
}

/* The indexer's other four ops. The fill laws have no per-sequence position run, so the
 * descriptions pick batches whose answers are defined and race-free on every implementation:
 * `cu` splits the M rows evenly over S sequences; the tail store gives every row a sequence and a
 * distinct slot of its own (two rows of one sequence could share a ring row); the work list puts
 * every row of a sequence at position ratio - 1, so each sequence completes block 0 from one key
 * of the batch and ratio - 1 of the tail. */
static int64_t qsa_seqs(const RadParam* p, int n_p, int64_t M) {
    const int64_t s = gi(p, n_p, "seqs", 1);
    return std::max<int64_t>(1, std::min(s, M));
}

int ref_qsa_score_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t heads = gi(p, n_p, "heads", 0), blocks = gi(p, n_p, "blocks", 0);
    if (M <= 0 || n <= 0 || heads <= 0 || blocks <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const int64_t S = qsa_seqs(p, n_p, M), pages = S * blocks;
    Opd o;
    o.t(D, { M, heads * n });                                     /* q */
    o.t(D, { pages, n });                                         /* bk -- one block key a page */
    o.t_idx({ S, blocks }, pages, -1, 0);                         /* bt -- page ids */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu */
    o.t_idx({ S }, blocks + 1, -1, 0);                            /* nc -- 0 .. blocks */
    o.t((uint32_t)RAD_F32, { M, blocks });                        /* score */
    return o.finish("qsa_score", operand, out);
}

int ref_qsa_select_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), blocks = gi(p, n_p, "blocks", 0);
    const int64_t topk = gi(p, n_p, "topk", 0), ratio = gi(p, n_p, "ratio", 0);
    if (M <= 0 || blocks <= 0 || topk <= 0 || ratio <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = qsa_seqs(p, n_p, M);
    Opd o;
    o.t((uint32_t)RAD_F32, { M, blocks });                        /* score */
    o.t_idx({ S, blocks }, S * blocks, -1, 0);                    /* bt -- page ids */
    o.t_idx({ S }, blocks + 1, -1, 0);                            /* nc -- 0 .. blocks */
    o.t_idx({ M }, (blocks + 1) * ratio, -1, 0);                  /* pos */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu */
    o.t((uint32_t)RAD_I32, { M, topk + 1 });                      /* sel */
    o.t((uint32_t)RAD_I32, { M });                                /* nsel */
    o.t((uint32_t)RAD_I32, { M });                                /* seqused */
    return o.finish("qsa_select", operand, out);
}

int ref_qsa_tail_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ratio = gi(p, n_p, "ratio", 0), ring = gi(p, n_p, "ring", ratio);
    if (M <= 0 || n <= 0 || ratio <= 0 || ring < ratio) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                             /* k */
    o.t_idx({ M }, 1 << 16, -1, 0);                               /* pos */
    o.t_idx({ M + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu -- a row a sequence */
    o.t(D, { M, 1, ring, n });                                    /* state */
    o.t_idx({ M, 1 }, M, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);     /* sidx -- a slot each */
    return o.finish("qsa_tail_store", operand, out);
}

int ref_qsa_work_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ratio = gi(p, n_p, "ratio", 0), ring = gi(p, n_p, "ring", ratio);
    const int64_t work = gi(p, n_p, "work", 0);
    if (M <= 0 || n <= 0 || ratio <= 0 || ring < ratio || work <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const int64_t S = std::min(qsa_seqs(p, n_p, M), work);
    Opd o;
    o.t(D, { M, n });                                             /* k */
    o.t_idx({ M }, 0, ratio - 1, 0);                              /* pos -- see above */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu */
    o.t(D, { S, 1, ring, n });                                    /* state -- the tail */
    o.t_idx({ S, 1 }, S, -1, 0);                                  /* sidx */
    o.t_idx({ S, 2 }, 1 << 16, -1, 0);                            /* bt -- block-key pages */
    o.skip();                                                     /* rope */
    o.t(D, { work * ratio, n });                                  /* stage */
    o.t((uint32_t)RAD_I32, { work });                             /* page */
    o.t((uint32_t)RAD_I32, { work });                             /* bpos */
    o.t((uint32_t)RAD_I32, { S });                                /* nc */
    return o.finish("qsa_work", operand, out);
}

/* `delta` is described as the whole [T, 2 * taps * NG] coefficient projection, of which the op
 * reads the first taps * NG columns at the row pitch: the form the architecture issues, and the
 * pitch libr4d requires. */
int ref_dflash_conv_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t T = gi(p, n_p, "T", 0), H = gi(p, n_p, "hidden_size", 0);
    const int64_t taps = gi(p, n_p, "taps", 0), NG = gi(p, n_p, "NG", 0);
    if (T <= 0 || H <= 0 || taps <= 0 || NG <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { T, H });                                             /* x */
    o.t(D, { T, 2 * taps * NG });                                 /* delta */
    o.t(D, { taps, H });                                          /* base */
    o.t(D, { T, H });                                             /* out */
    return o.finish("dflash_conv", operand, out);
}

/* ================================================================== mixture of experts */

int ref_scale_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                /* x */
    o.t(D, { M });                                   /* s -- one scalar a row */
    o.t(D, { M, n });                                /* add -- optional, described to exercise the
                                                      * fused form */
    o.t(D, { M, n });                                /* y */
    return o.finish("scale_rows", operand, out);
}

int ref_router_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    if (M <= 0 || ne <= 0 || k <= 0 || k > ne) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });                               /* logits */
    o.t((uint32_t)RAD_I32, { M, k });                /* expert_ids */
    o.t(D, { M, k });                                /* expert_w */
    return o.finish("router_topk", operand, out);
}

int ref_moe_scatter_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    if (M <= 0 || ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    /* Expert ids draw over [0, n_expert), exercising the sort rather than the dropping of
     * out-of-range slots. */
    o.t_idx({ M, k }, ne, -1, 0);                    /* expert_ids */
    o.t((uint32_t)RAD_I32, { M * k });               /* sorted_tok -- token*top_k + slot, by expert */
    o.t((uint32_t)RAD_I32, { ne + 1 });              /* expert_offset */
    o.t((uint32_t)RAD_I32, { ne });                  /* expert_count */
    return o.finish("moe_scatter", operand, out);
}

/* `a_order` decides the activation's row count: one per token, or one per (token, slot) in
 * scatter order. */
static inline bool sorted_a_order(const RadParam* p, int n_p) {
    const char* s = gs(p, n_p, "a_order");
    return s && std::strcmp(s, "sorted") == 0;
}

int ref_moe_gemm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t N  = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    if (M <= 0 || N <= 0 || K <= 0 || ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    const int64_t AR = sorted_a_order(p, n_p) ? M * k : M;
    o.t(D, { AR, K });                               /* a -- see `a_order` */
    o.t(D, { ne, N, K });                            /* w */
    /* A valid routing, as moe_scatter produces it: `sorted_tok` a permutation of [0, M*top_k) and
     * `expert_offset` a non-decreasing prefix array from 0 to M*top_k. A random offset array is a
     * malformed histogram that implementations searching it differently (binary against linear)
     * may read differently without either being wrong. */
    o.t_idx({ M * k }, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* sorted_tok */
    o.t_idx({ ne + 1 }, M * k + 1, -1, (uint32_t)RAD_OPD_F_IDX_PREFIX);   /* expert_offset */
    o.t(D, { M * k, N });                            /* y -- in sorted order */
    return o.finish("moe_gemm", operand, out);
}

/* The weight scale grid `dtype` names, as weight rows and K elements per scale. Block-fp8 experts
 * (fp8a8) carry the checkpoint's one scale per 128x128 tile (`group` = 128). The 4-bit formats
 * (docs/MOE-W4.md) carry one per row per K group: all of `group` for w4a8, w4a8h and w4nla8h, half
 * for w4nl64a8h and w5nl64a8h, a quarter for w4nl32a8h. These are the served format names
 * (arch/common/rad_block_moe_fp8.h, kServed). Any other dtype returns { 0, 0 } and the description
 * declines rather than guess a grid.
 *
 * The grid matters beyond extents: re-laid weights are inverted into the tensor described here
 * before the oracle reads them (core/runtime/oracle.cpp), so the wrong grid leaves the scales
 * nothing to invert into. The arithmetic (ref_gemm.h) derives the group from the extents. */
struct MoeScaleGrid { int64_t rows, k; };
static MoeScaleGrid moe_scale_grid(const RadParam* p, int n_p, int64_t g) {
    const char* d = gs(p, n_p, "dtype");
    auto is = [&](const char* s) { return d && std::strcmp(d, s) == 0; };
    if (is("fp8a8") || is("fp8a8f32")) return { g, g };
    if (is("mxfp4a8")) return { 1, 32 };
    if (is("w4a8") || is("w4a8h") || is("w4nla8h")) return { 1, g };
    if (is("w4nl64a8h") || is("w5nl64a8h")) return { 1, g / 2 };
    if (is("w4nl32a8h")) return { 1, g / 4 };
    return { 0, 0 };
}

/* The quantised arm: the same planes plus two scale streams, the activation's per row per K group
 * and the weight's one plane per expert on the grid above ([ceil(N/g), ceil(K/g)] for block fp8).
 * `w` is described stacked [ne, N, K], since a fixture cannot hold a pointer table; expert_entry()
 * reads both forms. */
int ref_moe_gemm_q_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t N0 = gi(p, n_p, "N", 0), K0 = gi(p, n_p, "K", 0);
    const int64_t N1 = gi(p, n_p, "N_odd", N0), K1 = gi(p, n_p, "K_odd", K0);
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    int64_t g = gi(p, n_p, "group", 0);
    if (M <= 0 || N0 <= 0 || K0 <= 0 || N1 <= 0 || K1 <= 0 || ne <= 0 || k <= 0)
        return RAD_E_UNSUPPORTED;
    /* Output width and activation K are the larger class's; the tables hold even and odd
     * experts. */
    const int64_t N = N0 > N1 ? N0 : N1, K = K0 > K1 ? K0 : K1;
    const int64_t ne0 = (ne + 1) / 2, ne1 = ne / 2;
    if (g <= 0) g = K;
    const MoeScaleGrid sg = moe_scale_grid(p, n_p, g);
    if (sg.rows <= 0 || sg.k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    const int64_t AR = sorted_a_order(p, n_p) ? M * k : M;
    const char* kind = gs(p, n_p, "dtype");
    const bool mx = kind && !std::strcmp(kind, "mxfp4a8");
    const auto SD = kind && !std::strcmp(kind, "fp8a8f32") ? (uint32_t)RAD_F32 : D;
    o.t((uint32_t)RAD_F8E4M3, { AR, K });             /* a -- E4M3 codes, see `a_order` */
    o.t((uint32_t)RAD_F32, { AR, (K + g - 1) / g });   /* a_scale */
    /* w -- the even experts; mxfp4a8 carries e2m1 codes with an E8M0 byte per (row, 32 of K) */
    o.t(mx ? (uint32_t)RAD_FP4E2M1 : (uint32_t)RAD_F8E4M3, { ne0, N0, K0 });
    if (mx) o.t_idx_dt(RAD_U8, { ne0, N0, (K0 + 31) / 32 }, 0, 127, 0);    /* w_scale */
    else    o.t(SD, { ne0, (N0 + sg.rows - 1) / sg.rows, (K0 + sg.k - 1) / sg.k });
    o.t_idx({ M * k }, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* sorted_tok */
    o.t_idx({ ne + 1 }, M * k + 1, -1, (uint32_t)RAD_OPD_F_IDX_PREFIX);   /* expert_offset */
    /* w_odd -- the odd experts */
    o.t(mx ? (uint32_t)RAD_FP4E2M1 : (uint32_t)RAD_F8E4M3, { ne1, N1, K1 });
    if (mx) o.t_idx_dt(RAD_U8, { ne1, N1, (K1 + 31) / 32 }, 0, 127, 0);    /* w_odd_scale */
    else    o.t(SD, { ne1, (N1 + sg.rows - 1) / sg.rows, (K1 + sg.k - 1) / sg.k });
    o.t(D, { M * k, N });                             /* y -- in sorted order */
    return o.finish("moe_gemm_q", operand, out);
}

/* The four-class form: class q at [N_q, K_q] (default class 0's) holds experts e % 4 == q,
 * ceil((n - q) / 4) of them; `y` follows the last class. */
int ref_moe_gemm_q_mod4_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t N0 = gi(p, n_p, "N", 0), K0 = gi(p, n_p, "K", 0);
    const int64_t Nq[4] = { N0, gi(p, n_p, "N_1", N0), gi(p, n_p, "N_2", N0), gi(p, n_p, "N_3", N0) };
    const int64_t Kq[4] = { K0, gi(p, n_p, "K_1", K0), gi(p, n_p, "K_2", K0), gi(p, n_p, "K_3", K0) };
    const int64_t ne = gi(p, n_p, "n_expert", 0);
    const int64_t k  = gi(p, n_p, "top_k", 0);
    int64_t g = gi(p, n_p, "group", 0);
    if (M <= 0 || ne <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    int64_t N = 0, K = 0;
    for (int q = 0; q < 4; ++q) {
        if (Nq[q] <= 0 || Kq[q] <= 0) return RAD_E_UNSUPPORTED;
        if (Nq[q] > N) N = Nq[q];
        if (Kq[q] > K) K = Kq[q];
    }
    if (g <= 0) g = K;
    const MoeScaleGrid sg = moe_scale_grid(p, n_p, g);
    if (sg.rows <= 0 || sg.k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    const int64_t AR = sorted_a_order(p, n_p) ? M * k : M;
    const char* kind = gs(p, n_p, "dtype");
    const bool mx = kind && !std::strcmp(kind, "mxfp4a8");
    auto cls = [&](int q) {
        const int64_t nq = (ne + 3 - q) / 4;
        o.t(mx ? (uint32_t)RAD_FP4E2M1 : (uint32_t)RAD_F8E4M3, { nq, Nq[q], Kq[q] });
        if (mx) o.t_idx_dt(RAD_U8, {nq, Nq[q], (Kq[q] + 31) / 32}, 0, 127, 0);
        else o.t(D, { nq, (Nq[q] + sg.rows - 1) / sg.rows, (Kq[q] + sg.k - 1) / sg.k });
    };
    o.t((uint32_t)RAD_F8E4M3, { AR, K });             /* a -- E4M3 codes, see `a_order` */
    o.t((uint32_t)RAD_F32, { AR, (K + g - 1) / g });   /* a_scale */
    cls(0);                                           /* w, w_scale -- class 0 */
    o.t_idx({ M * k }, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* sorted_tok */
    o.t_idx({ ne + 1 }, M * k + 1, -1, (uint32_t)RAD_OPD_F_IDX_PREFIX);   /* expert_offset */
    cls(1);
    cls(2);
    cls(3);
    o.t(D, { M * k, N });                             /* y -- in sorted order */
    return o.finish("moe_gemm_q_mod4", operand, out);
}

int ref_moe_gather_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t k = gi(p, n_p, "top_k", 0);
    if (M <= 0 || n <= 0 || k <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M * k, n });                            /* y_expert -- one row per (token, slot) */
    o.t(D, { M, k });                                /* expert_w */
    /* Distinct although read-side: gather rebuilds the inverse permutation, so a repeat would let
     * two rows claim one (token, slot) with an implementation-defined winner. A full permutation of
     * [0, M*top_k), as moe_scatter produces. */
    o.t_idx({ M * k }, M * k, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* sorted_tok */
    o.t(D, { M, n });                                /* shared -- optional, described anyway */
    o.t(D, { M });                                   /* shared_gate */
    o.t(D, { M, n });                                /* y */
    return o.finish("moe_gather", operand, out);
}

int ref_row_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), R = gi(p, n_p, "R", 0);
    if (M <= 0 || N <= 0 || R <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, N });                                /* x */
    o.t((uint32_t)RAD_I32, { M, R });                /* idx */
    o.t((uint32_t)RAD_F32, { M, R });                /* val */
    /* `pairs`, an output, is described to check the packing: an int32 id's bits in the first float
     * slot beside its value, id -1 for an unfilled slot. It must be f32 to hold the value. */
    o.t((uint32_t)RAD_F32, { M, R, 2 });             /* pairs */
    return o.finish("row_topk", operand, out);
}

int ref_row_topk_merge_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), R = gi(p, n_p, "R", 0);
    const int64_t W = gi(p, n_p, "world_size", 1);
    if (M <= 0 || R <= 0 || W <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    /* A fillable packed pair plane (unlike sample_merge_topk's): as f32 normals the value lanes are
     * proper floats, and the id lanes are float bit patterns read as int32, negative ones dropped as
     * pads and positive ones legal ids that are only compared and copied. No ties occur. */
    o.t((uint32_t)RAD_F32, { M, W * R, 2 });         /* gathered */
    o.t((uint32_t)RAD_I32, { M, R });                /* idx */
    o.t((uint32_t)RAD_F32, { M, R });                /* val */
    return o.finish("row_topk_merge", operand, out);
}

/* The same pair plane, fillable for the same reason as row_topk_merge's. */
int ref_pair_argmax_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, n, 2 });             /* pairs */
    o.t((uint32_t)RAD_I32, { M });                   /* idx */
    o.t((uint32_t)RAD_F32, { M });                   /* val */
    return o.finish("pair_argmax", operand, out);
}



/* The head is described in f32: an E4M3 code times its bf16 block scale needs more mantissa than
 * bf16 has, and a reference reading rounded weights reorders candidates whose exact scores are
 * close. At a full vocabulary that puts a case over rad-kbench's default --max-bytes, so it is
 * checked only where the budget is raised; r4d_selftest checks the kernels over their stored head
 * either way. Candidate ids are distinct, as row_topk's are: a repeated id is one entry of the
 * kernel's top-R chain but two of a plain sort.
 *
 * The op oracle needs the description too: it un-relays the device weight through
 * RadUnrelayoutFn and dequantises it into this logical form, giving the reference exactly what the
 * kernel saw. ref_logits_gemm_shape describes the same weight. */
int ref_logit_rerank_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0), R = gi(p, n_p, "R", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0), ne = gi(p, n_p, "n_embd", 0);
    const int64_t vo = gi(p, n_p, "vocab_off", 0);
    if (M <= 0 || R <= 0 || nv <= 0 || ne <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });                               /* x      */
    o.t((uint32_t)RAD_F32, { nv, ne });              /* head   */
    /* Candidate ids are global, drawn over [0, vo + nv); those below this rank's slice become pads
     * (-1). */
    o.t_idx({ M, R }, vo + nv, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* idx_in */
    o.t_idx({ M, R }, vo + nv, -1, 0);                /* idx    */
    o.t((uint32_t)RAD_F32, { M, R });                /* val    */
    /* f32 [M, R, 2], as row_topk's. */
    o.t((uint32_t)RAD_F32, { M, R, 2 });             /* pairs  */
    return o.finish("logit_rerank", operand, out);
}
/* ================================================================== the fp8 fusions
 *
 * The unfused pair's operands minus the bf16 intermediate, which `out_bf16` restores (the codes are
 * taken over its value).
 */
int ref_rmsnorm_quant_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, n });                       /* x */
    o.t(D, { M, n });                       /* residual, inout -- the fused_add_rms_norm form */
    o.t(D, { n });                          /* w, the gain */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t((uint32_t)RAD_BF16, { M, n });      /* out_bf16 -- what the codes were taken over */
    return o.finish("rmsnorm_quant_fp8", operand, out);
}

int ref_gated_quant_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    Opd o;
    o.t(D, { M, 2 * n });                   /* gate_up, gate half first */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t((uint32_t)RAD_BF16, { M, n });      /* out_bf16 */
    return o.finish("gated_quant_fp8", operand, out);
}

/* `N` is the weight's rows, twice the fused width: the quantised output is [M, N/2]. */
int ref_gemm_nt_q_gated_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), N = gi(p, n_p, "N", 0), K = gi(p, n_p, "K", 0);
    const int64_t group = gi(p, n_p, "group", 0);
    const char* q = gs(p, n_p, "dtype");
    const bool a8 = q && std::strcmp(q, "fp8a8") == 0;
    if (!a8 || N <= 0 || K <= 0 || group <= 0 || (N & 1)) return RAD_E_UNSUPPORTED;
    const int64_t n  = N / 2;
    const int64_t kb = (K + group - 1) / group;
    const int64_t nb = (N + group - 1) / group;
    const int64_t ng = scale_groups(n, group);
    Opd o;
    o.t((uint32_t)RAD_F8E4M3, { M, K });    /* a */
    o.t((uint32_t)RAD_F32, { M, kb });      /* a_scale */
    o.t((uint32_t)RAD_F8E4M3, { N, K });    /* b */
    o.t((uint32_t)RAD_BF16, { nb, kb });    /* b_scale */
    o.t((uint32_t)RAD_F8E4M3, { M, n });    /* q */
    o.t((uint32_t)RAD_F32, { M, ng });      /* scale */
    o.t((uint32_t)RAD_BF16, { M, n });      /* out_bf16 */
    return o.finish("gemm_nt_q_gated", operand, out);
}

/* DFlash2's candidate walk. `cand` and `anchor` hold token ids, bounded by n_vocab so the codebook
 * gather stays in the plane. `cand`/`unary`/`hp` have one row per (sequence, step), M * steps. */
int ref_dflash_select_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M     = gi(p, n_p, "M", 0);
    const int64_t steps = gi(p, n_p, "steps", 0);
    const int64_t K     = gi(p, n_p, "top_k", 0);
    const int64_t R     = gi(p, n_p, "rank", 0);
    const int64_t nv    = gi(p, n_p, "n_vocab", 0);
    const int64_t astr  = gi(p, n_p, "anchor_stride", 1);
    if (M <= 0 || steps <= 0 || K <= 0 || R <= 0 || nv <= 0 || astr <= 0) return RAD_E_UNSUPPORTED;
    const int64_t rows = M * steps;
    Opd o;
    o.t_idx_dt((uint32_t)RAD_I32, { rows, K }, nv, -1, 0);   /* cand */
    o.t((uint32_t)RAD_F32, { rows, K });                     /* unary */
    o.t((uint32_t)RAD_BF16, { rows, R });                    /* hp, absent is a plane of ones */
    o.t_idx_dt((uint32_t)RAD_I32, { M * astr }, nv, -1, 0);  /* anchor */
    o.t((uint32_t)RAD_BF16, { nv, R });                      /* pred */
    o.t((uint32_t)RAD_BF16, { nv, R });                      /* succ */
    o.t_idx_dt((uint32_t)RAD_I32, { M, steps }, nv, -1, 0);  /* tokens */
    return o.finish("dflash_select", operand, out);
}

/* ================================================================== gated delta net
 *
 * The layouts are libref/ref_gdn.cpp's:
 *
 *     q, k     [T, Hk, head_k]         v, o  [T, Hv, head_v]
 *     g, beta  [T, Hv]  fp32           A     [T, Hv, chunk]
 *     h0, ht   [N, Hv, head_v, head_k] fp32  cu    [N+1]
 *     conv w   [conv_dim, conv_width]        conv_state [n_slots, conv_dim, state_len]
 *
 * with conv_dim = 2 * Hk * head_k + Hv * head_v. The family declines unless the caller supplies
 * the head counts (see gdn_heads). */

namespace {
/* What the gdn descriptions share, resolved once; gdn_geom returns false to decline. */
struct GdnGeom {
    int64_t Hk, Hv, hk, hv, chunk, width, conv_dim, S, M;
    uint32_t D;
};

static bool gdn_geom(const RadParam* p, int n_p, const char* mkey, GdnGeom* g) {
    if (!gdn_heads(p, n_p, &g->Hk, &g->Hv)) return false;
    g->hk = gi(p, n_p, "head_k", 0);
    g->hv = gi(p, n_p, "head_v", g->hk);
    g->chunk = gi(p, n_p, "chunk", 64);
    g->width = gi(p, n_p, "conv_width", 4);
    g->M = gi(p, n_p, mkey, 0);
    g->S = n_seq_of(p, n_p);
    g->D = act_dt(p, n_p);
    if (g->hk <= 0 || g->hv <= 0 || g->chunk <= 0 || g->width <= 1 || g->M <= 0) return false;
    g->conv_dim = 2 * g->Hk * g->hk + g->Hv * g->hv;
    return true;
}

/* The recurrence's state, state_idx and num_accepted, by `state_form` (docs/OPS.md).
 *
 * per_candidate: a slot per candidate token, a [S, M] table of distinct slots, and every token
 * accepted.
 *
 * anchor: the kernels read six columns a sequence (the anchor's three, then the factor block's
 * three), so the table is at least six wide, every entry a distinct slot. The state is drawn
 * whole, so the factor block holds random factors and a random stamp; every implementation reads
 * a block the same way (each 32-row group of V from its own copy of k, g and beta, the anchor
 * column from the rounded stamp), so that is a defined input. One accepted token is replayed: with
 * random k the replay scales the state by about |k|^2, which one token keeps finite. A step
 * longer than the factor block holds is refused by the op, so it is not described. */
bool gdn_state(const RadParam* p, int n_p, const GdnGeom& g, Opd& o) {
    const char* form = gs(p, n_p, "state_form");
    if (form && std::strcmp(form, "anchor") == 0) {
        const int64_t groups = (g.hv + 31) / 32, fstr = groups * g.hk + g.hv + 2 * groups;
        const int64_t tmax = std::min<int64_t>(16, (g.hv * g.hk - 1) / fstr);
        if (tmax < 1 || g.M > tmax) return false;
        const int64_t cols = std::max<int64_t>(g.M, 6), slots = g.S * cols;
        o.t((uint32_t)RAD_F32, { slots, g.Hv, g.hv, g.hk });             /* state, INOUT */
        o.t_idx({ g.S, cols }, slots, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);  /* state_idx */
        o.t_idx({ g.S }, 0, 1, 0);                                      /* num_accepted */
        return true;
    }
    if (form && std::strcmp(form, "per_candidate") != 0) return false;
    const int64_t T = g.S * g.M;
    /* One state slot per candidate token (ref_ops.h's rollback contract). */
    o.t((uint32_t)RAD_F32, { T, g.Hv, g.hv, g.hk });                     /* state, INOUT */
    o.t_idx({ g.S, g.M }, T, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);        /* state_idx */
    o.t_idx({ g.S }, 0, 1, 0);                              /* num_accepted: none rejected */
    return true;
}

}  /* namespace */

int ref_gdn_conv_prep_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "M", &g)) return RAD_E_UNSUPPORTED;
    Opd o;
    /* T tokens over S sequences, one conv-state slot each, `state_len` conv_width - 1 (the
     * manager's conv_width - 1 + n_spec with no speculation). */
    o.t(g.D, { g.M, g.conv_dim });                   /* x -- the fused projection row */
    o.t(g.D, { g.conv_dim, g.width });               /* w */
    o.skip();                                        /* b -- this checkpoint has no conv bias */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* A_log */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* dt_bias */
    o.t(g.D, { g.S, g.conv_dim, g.width - 1 });      /* conv_state, INOUT */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    /* `a` and `b_gate` passed: the separate-projection path every current caller takes. */
    o.t(g.D, { g.M, g.Hv });                         /* a */
    o.t(g.D, { g.M, g.Hv });                         /* b_gate */
    o.t_idx({ g.S }, g.S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* cache_idx, one a sequence */
    /* A cold prefill, stated explicitly: ref reads a null `has_init` as warm, libr4d as cold. */
    o.t_idx({ g.S }, 0, 0, 0);                       /* has_init */
    o.t(g.D, { g.M, g.Hk, g.hk });                   /* q */
    o.t_fill(g.D, { g.M, g.Hk, g.hk }, (uint32_t)RAD_FILL_UNIT_ROW, 0);                  /* k */
    o.t(g.D, { g.M, g.Hv, g.hv });                   /* v */
    /* g and beta are outputs, but their domains are stated: they are the inputs of gdn_kkt_solve
     * and gdn_chunk_scan. */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_GATE_CUMSUM, g.chunk);  /* g */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_SIGMOID, 0);            /* beta */
    return o.finish("gdn_conv_prep", operand, out);
}

int ref_gdn_conv_update_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "q_len", &g)) return RAD_E_UNSUPPORTED;
    /* Decode: `M` is q_len, one row per candidate token, and the window is conv_width - 2 + q_len
     * deep (ref_ops.h's rollback contract). */
    const int64_t sl = g.width - 2 + g.M;
    Opd o;
    o.t(g.D, { g.S * g.M, g.conv_dim });             /* x */
    o.t(g.D, { g.conv_dim, g.width });               /* w */
    o.skip();                                        /* b */
    o.t(g.D, { g.S, g.conv_dim, sl });               /* conv_state, INOUT */
    o.t_idx({ g.S, g.M }, g.S, -1, 0);               /* state_idx */
    /* num_accepted is 1-based; all ones is the non-speculative decode. A draw could move the read
     * offset outside the window. */
    o.t_idx({ g.S }, 0, 1, 0);                       /* num_accepted */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { g.S * g.M, g.Hk, g.hk });             /* q */
    o.t(g.D, { g.S * g.M, g.Hk, g.hk });             /* k */
    o.t(g.D, { g.S * g.M, g.Hv, g.hv });             /* v */
    return o.finish("gdn_conv_update", operand, out);
}

int ref_gdn_kkt_solve_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "M", &g)) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_fill(g.D, { g.M, g.Hk, g.hk }, (uint32_t)RAD_FILL_UNIT_ROW, 0);                  /* k */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_SIGMOID, 0);            /* beta */
    /* Non-increasing, never positive, reset at each chunk boundary (see the file header). */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_GATE_CUMSUM, g.chunk);  /* g */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t_fill(g.D, { g.M, g.Hv, g.chunk }, (uint32_t)RAD_FILL_TRI_LOWER, g.chunk);         /* A */
    return o.finish("gdn_kkt_solve", operand, out);
}

int ref_gdn_chunk_scan_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "M", &g)) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(g.D, { g.M, g.Hk, g.hk });                   /* q */
    o.t_fill(g.D, { g.M, g.Hk, g.hk }, (uint32_t)RAD_FILL_UNIT_ROW, 0);                  /* k */
    o.t(g.D, { g.M, g.Hv, g.hv });                   /* v */
    /* Zero above the diagonal within a chunk, as kkt_solve produces (see the file header). */
    o.t_fill(g.D, { g.M, g.Hv, g.chunk }, (uint32_t)RAD_FILL_TRI_LOWER, g.chunk);         /* A */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_GATE_CUMSUM, g.chunk);  /* g */
    o.t_fill((uint32_t)RAD_F32, { g.M, g.Hv }, (uint32_t)RAD_FILL_SIGMOID, 0);            /* beta */
    o.t((uint32_t)RAD_F32, { g.S, g.Hv, g.hv, g.hk });               /* h0 */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { g.M, g.Hv, g.hv });                   /* o */
    o.t((uint32_t)RAD_F32, { g.S, g.Hv, g.hv, g.hk });               /* ht */
    o.t_idx({ g.S }, g.S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* state_idx */
    return o.finish("gdn_chunk_scan", operand, out);
}

int ref_gdn_recurrent_update_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "q_len", &g)) return RAD_E_UNSUPPORTED;
    const int64_t T = g.S * g.M;
    Opd o;
    o.t(g.D, { T, g.Hk, g.hk });                     /* q */
    o.t(g.D, { T, g.Hk, g.hk });                     /* k */
    o.t(g.D, { T, g.Hv, g.hv });                     /* v */
    /* `a` and `b` are the raw gate and beta projections (the op applies softplus and sigmoid
     * itself), so normals are the right domain. */
    o.t(g.D, { T, g.Hv });                           /* a */
    o.t(g.D, { T, g.Hv });                           /* b */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* A_log */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* dt_bias */
    if (!gdn_state(p, n_p, g, o)) return RAD_E_UNSUPPORTED;  /* state, state_idx, num_accepted */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { T, g.Hv, g.hv });                     /* z -- the output gate */
    o.t((uint32_t)RAD_F32, { g.hv });                /* norm_w */
    o.t(g.D, { T, g.Hv, g.hv });                     /* o */
    /* The folded fp8 out-projection quantiser (all-or-nothing) is skipped; the unfused pair (this
     * op, then quant_act_fp8) covers the same arithmetic. */
    o.skip();                                        /* o_q */
    o.skip();                                        /* o_scale */
    return o.finish("gdn_recurrent_update", operand, out);
}

int ref_gdn_conv_recurrent_update_shape(const RadParam* p, int n_p, int operand,
                                        RadOpdDesc* out) {
    GdnGeom g;
    if (!gdn_geom(p, n_p, "q_len", &g)) return RAD_E_UNSUPPORTED;
    /* gdn_conv_update's inputs, then gdn_recurrent_update's minus q, k and v. `conv_idx` is one
     * slot per sequence. */
    const int64_t T = g.S * g.M, sl = g.width - 2 + g.M;
    Opd o;
    o.t(g.D, { T, g.conv_dim });                     /* x */
    o.t(g.D, { g.conv_dim, g.width });               /* conv_w */
    o.skip();                                        /* conv_b */
    o.t(g.D, { g.S, g.conv_dim, sl });               /* conv_state, INOUT */
    o.t_idx({ g.S }, g.S, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* conv_idx */
    o.t(g.D, { T, g.Hv });                           /* a */
    o.t(g.D, { T, g.Hv });                           /* b */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* A_log */
    o.t((uint32_t)RAD_F32, { g.Hv });                /* dt_bias */
    if (!gdn_state(p, n_p, g, o)) return RAD_E_UNSUPPORTED;  /* state, state_idx, num_accepted */
    o.t_idx({ g.S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);         /* cu */
    o.t(g.D, { T, g.Hv, g.hv });                     /* z */
    o.t((uint32_t)RAD_F32, { g.hv });                /* norm_w */
    o.t(g.D, { T, g.Hv, g.hv });                     /* o */
    o.skip();                                        /* o_q -- see gdn_recurrent_update */
    o.skip();                                        /* o_scale */
    return o.finish("gdn_conv_recurrent_update", operand, out);
}

int ref_gdn_gated_rmsnorm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    /* One row per (token, head): `M` is tokens x heads and `channels` is head_v, so no head count
     * is needed. */
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ch = gi(p, n_p, "channels", 0);
    if (M <= 0 || ch <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ch });                               /* x */
    o.t(D, { M, ch });                               /* z -- the gate */
    o.t((uint32_t)RAD_F32, { ch });                  /* w */
    o.t(D, { M, ch });                               /* o */
    return o.finish("gdn_gated_rmsnorm", operand, out);
}

/* ================================================================== collectives */

int ref_all_reduce_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t numel = gi(p, n_p, "numel", 0);
    if (numel <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { numel });                               /* x, INOUT */
    /* `y`, the optional separate destination, is passed: libr4d's two-shot rows are not safe in
     * place and need it (see ref_all_reduce). */
    o.t(D, { numel });                               /* y */
    return o.finish("all_reduce", operand, out);
}

int ref_all_gather_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t numel = gi(p, n_p, "numel", 0);
    const int64_t W = gi(p, n_p, "world_size", 1);
    if (numel <= 0 || W <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { numel });                               /* x */
    o.t(D, { numel * W });                           /* y -- rank r's contribution at r * numel */
    return o.finish("all_gather", operand, out);
}

/* ================================================================== vocabulary edges */

int ref_embed_lookup_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t vo = gi(p, n_p, "vocab_offset", 0);
    if (M <= 0 || ne <= 0 || nv <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    /* The range is explicit since `tokens` is operand 0. Ids are global, so the bound is
     * `vocab_offset + nv`: under a shard the draw crosses the slice boundary and exercises the
     * zero-row path. */
    o.t_idx({ M }, vo + nv, -1, 0);                  /* tokens */
    o.t(D, { nv, ne });                              /* wte */
    o.t(D, { M, ne });                               /* x */
    return o.finish("embed_lookup", operand, out);
}

/* The quantised gather: the table is E4M3 and the output the activation dtype (`dtype` names the
 * combination, e.g. "w8a16"). `scale` is one element; a normal draw is fine, since the op only
 * multiplies by it. */
int ref_embed_lookup_q_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t vo = gi(p, n_p, "vocab_offset", 0);
    if (M <= 0 || ne <= 0 || nv <= 0 || vo < 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t_idx({ M }, vo + nv, -1, 0);                  /* tokens */
    o.t((uint32_t)RAD_F8E4M3, { nv, ne });           /* wte */
    o.t(D, { 1 });                                   /* scale */
    o.t(D, { M, ne });                               /* x */
    o.skip();                                        /* ahead: a later call's ids, optional */
    return o.finish("embed_lookup_q", operand, out);
}

int ref_logits_gemm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nv = gi(p, n_p, "n_vocab", 0);
    const int64_t ne = gi(p, n_p, "n_embd", 0);
    if (M <= 0 || nv <= 0 || ne <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, ne });                               /* x */
    o.t(D, { nv, ne });                              /* lm_head */
    /* f32 logits, as every sampler stage reads them. */
    o.t((uint32_t)RAD_F32, { M, nv });               /* logits */
    return o.finish("logits_gemm", operand, out);
}

/* ================================================================== sampling
 *
 * Per-row scalars arrive in the `params` operand (see sample_params_words() for its draw). The
 * other operands:
 *
 *   logits     f32 [M, n_vocab]     cand_idx  i32 [M, n_cand]    token   i32 [M]
 *   history    i32 [M, max_hist]    cand_val  f32 [M, n_cand]    bitmask i32 [M, ceil(nv/32)]
 *   breakers   u8  [M, max_hist]
 *
 * libr4d's shim requires exactly these dtypes (libr4d/r4d_sample_stages_f32.hip). Word-typed
 * operands are i32, never u32 (see sample_params_words()). */

/* The history window width: the deployment sizes the history plane, so this picks a plausible 64. */
enum { REF_SAMPLE_HIST = 64 };
/* The candidate width for `sample_topk`, which reads it off the operand (its parameters carry only
 * `n_vocab`): narrower than the vocabulary, wide enough for the narrowing stages. */
enum { REF_SAMPLE_CAND = 64 };

static int sample_vocab(const RadParam* p, int n_p, int64_t* M, int64_t* nv) {
    *M  = gi(p, n_p, "M", 0);
    *nv = gi(p, n_p, "n_vocab", 0);
    return (*M > 0 && *nv > 0) ? RAD_OK : RAD_E_UNSUPPORTED;
}

/* The `params` plane, drawn as requests (see sample_params_words()). The law places each row's
 * history window inside a plane `hist` wide and its `mask_row` among `mask_rows` bitmask rows; 0
 * for an op without that plane. */
static void sample_params(Opd& o, int64_t M, int64_t hist, int64_t mask_rows) {
    RadOpdDesc* d = o.push();
    Opd::shape(d, (uint32_t)RAD_F32, { M, sample_params_words() });
    d->fill       = (uint32_t)RAD_FILL_SAMPLE_PARAMS;
    d->fill_chunk = hist;
    d->idx_max    = mask_rows;
}

/* sample_penalties and sample_dry share logits, params and history. The history is real token ids
 * in [0, n_vocab) that repeat as text does (RAD_OPD_F_IDX_HISTORY). DRY's optional breaker plane is
 * passed with about half its positions marked, so the exemption runs as well as the penalty. */
static int penalty_shape(const RadParam* p, int n_p, int operand, const char* op,
                         bool breakers, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits, INOUT */
    sample_params(o, M, REF_SAMPLE_HIST, 0);                        /* params */
    o.t_idx({ M, (int64_t)REF_SAMPLE_HIST }, nv, -1,
            (uint32_t)RAD_OPD_F_IDX_HISTORY);                       /* history */
    if (breakers)                                                   /* breakers, optional */
        o.t_idx_dt((uint32_t)RAD_U8, { M, (int64_t)REF_SAMPLE_HIST }, 2, -1, 0);
    return o.finish(op, operand, out);
}

int ref_sample_penalties_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return penalty_shape(p, n_p, operand, "sample_penalties", false, out);
}
int ref_sample_dry_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return penalty_shape(p, n_p, operand, "sample_dry", true, out);
}

int ref_sample_temp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits, INOUT */
    sample_params(o, M, 0, 0);                                      /* params */
    return o.finish("sample_temp", operand, out);
}

int ref_sample_mask_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits, INOUT */
    sample_params(o, M, 0, M);                                      /* params */
    /* A set bit means allowed, word w bit b covering token 32w + b. One row per sequence is the
     * widest the plane gets (unconstrained rows share it with `mask_row` -1). */
    o.t((uint32_t)RAD_I32, { M, (nv + 31) / 32 });                  /* bitmask */
    return o.finish("sample_mask", operand, out);
}

int ref_sample_argmax_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits */
    sample_params(o, M, 0, 0);                                      /* params -- unread by argmax */
    o.t((uint32_t)RAD_I32, { M });                                  /* token */
    return o.finish("sample_argmax", operand, out);
}

int ref_sample_logprobs_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const int64_t k = gi(p, n_p, "k", 0);
    if (k <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits */
    sample_params(o, M, 0, 0);                                      /* params */
    o.t((uint32_t)RAD_F32, { M, 2 + 2 * k });                       /* stats */
    return o.finish("sample_logprobs", operand, out);
}

int ref_sample_topk_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    int64_t M = 0, nv = 0;
    if (sample_vocab(p, n_p, &M, &nv) < 0) return RAD_E_UNSUPPORTED;
    const int64_t nc = nv < (int64_t)REF_SAMPLE_CAND ? nv : (int64_t)REF_SAMPLE_CAND;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, nv });                              /* logits */
    sample_params(o, M, 0, 0);                                      /* params */
    o.t((uint32_t)RAD_I32, { M, nc });                              /* cand_idx */
    o.t((uint32_t)RAD_F32, { M, nc });                              /* cand_val -- logits, not probs */
    /* The vocab-parallel pair plane, an output, described to check the packing: (u32 global id,
     * f32 logit), 0xffffffff for a hole; `vocab_off` is 0 here. */
    o.t((uint32_t)RAD_I32, { M, 2 * nc });                          /* pairs */
    return o.finish("sample_topk", operand, out);
}

/* The four narrowing stages share a schema (cand_idx, cand_val, params, in place) and one
 * description.
 *
 * The candidate set is drawn, not sorted: the ABI has no descending fill domain. min-p, typical and
 * XTC are per-candidate predicates. Top-p is a prefix rule in candidate order, which applies equally
 * to a random order; an implementation that sorted internally would genuinely disagree with the
 * schema. */
static int narrow_shape(const RadParam* p, int n_p, int operand, const char* op, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nc = gi(p, n_p, "n_cand", 0);
    if (M <= 0 || nc <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    /* Drawn indices are >= 0, so every candidate is live. The range is the candidate width; ids
     * are only copied and compared here. */
    o.t_idx({ M, nc }, nc, -1, 0);                                  /* cand_idx, INOUT */
    o.t((uint32_t)RAD_F32, { M, nc });                              /* cand_val, INOUT */
    sample_params(o, M, 0, 0);                                      /* params */
    return o.finish(op, operand, out);
}

int ref_sample_topp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_topp", out);
}
int ref_sample_minp_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_minp", out);
}
int ref_sample_typical_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_typical", out);
}
int ref_sample_xtc_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    return narrow_shape(p, n_p, operand, "sample_xtc", out);
}

int ref_sample_pick_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M  = gi(p, n_p, "M", 0);
    const int64_t nc = gi(p, n_p, "n_cand", 0);
    if (M <= 0 || nc <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ M, nc }, nc, -1, 0);                                  /* cand_idx */
    o.t((uint32_t)RAD_F32, { M, nc });                              /* cand_val */
    sample_params(o, M, 0, 0);                                      /* params -- seed and pos */
    o.t((uint32_t)RAD_I32, { M });                                  /* token */
    return o.finish("sample_pick", operand, out);
}

/* sample_merge_topk is declined although its extents are known. `gathered` packs (u32 global id,
 * f32 logit) pairs in a word-typed buffer, so logits are read by bit pattern and no fill domain
 * yields sensible floats: an integer draw makes every logit a denormal, which a flushing device
 * and a non-flushing host order differently (the failure r4d_selftest's two-rank all-gather
 * checks). Also, `world_size` is not a parameter, so the gathered width would be invented.
 *
 * row_topk_merge is not declined: its plane is f32, so the value lane is a real float. */
int ref_sample_merge_topk_shape(const RadParam*, int, int, RadOpdDesc*) {
    return RAD_E_UNSUPPORTED;
}

/* ================================================================== image generation (ref_gen.cpp)
 *
 * Lane ops are described over REF_LANES lanes and image ops at a chosen picture shape (below);
 * gather_rows2 declines. */
int ref_ln_mod_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* x */
    o.t(D, { 1, n });                                         /* s: one sequence */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M, n });                                         /* y */
    return o.finish("ln_mod", operand, out);
}
int ref_ln_mod_quant_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* y */
    o.t(D, { 1, n });                                         /* g: one sequence */
    o.t(D, { 1, n });                                         /* s */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M, n });                                         /* x, INOUT */
    o.t((uint32_t)RAD_F8E4M3, { M, n });                      /* q */
    o.t((uint32_t)RAD_F32, { M, ng });                        /* scale */
    o.t(D, { M, n });                                         /* z */
    return o.finish("ln_mod_quant_fp8", operand, out);
}
/* rms_mod's branch and gate are passed when `gated`, and absent otherwise. */
int ref_rms_mod_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    if (gi(p, n_p, "gated", 0)) {
        o.t(D, { M, n });                                     /* y */
        o.t(D, { 1, n });                                     /* g: one sequence */
    } else {
        o.skip();
        o.skip();
    }
    o.t(D, { 1, 2 * n });                                     /* s: [shift | scale] */
    /* A LayerNorm without a gain passes none, and no norm takes none. */
    const char* nk = gs(p, n_p, "norm");
    if (nk && (std::strcmp(nk, "layer") == 0 || std::strcmp(nk, "none") == 0)) o.skip();  /* w */
    else                                     o.t(D, { n });  /* w */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M, n });                                         /* x, INOUT */
    o.t(D, { M, n });                                         /* z */
    return o.finish("rms_mod", operand, out);
}
int ref_rms_mod_quant_fp8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t ng = scale_groups(n, gi(p, n_p, "group", 0));
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    if (gi(p, n_p, "gated", 0)) {
        o.t(D, { M, n });                                     /* y */
        o.t(D, { 1, n });                                     /* g */
    } else {
        o.skip();
        o.skip();
    }
    o.t(D, { 1, 2 * n });                                     /* s */
    const char* nk = gs(p, n_p, "norm");
    if (nk && std::strcmp(nk, "layer") == 0) o.skip();       /* w: a LayerNorm without a gain */
    else                                     o.t(D, { n });  /* w */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M, n });                                         /* x, INOUT */
    o.t((uint32_t)RAD_F8E4M3, { M, n });                      /* q */
    o.t((uint32_t)RAD_F32, { M, ng });                        /* scale */
    o.t(D, { M, n });                                         /* z */
    return o.finish("rms_mod_quant_fp8", operand, out);
}
/* The sampler's three scalars in (0, 1), the state f32. */
int ref_flow_x0_step_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t F = (uint32_t)RAD_F32;
    Opd o;
    o.t(act_dt(p, n_p), { M, n });                            /* v */
    o.t_fill(F, { 1 }, (uint32_t)RAD_FILL_SIGMOID, 0);        /* t */
    o.t_fill(F, { 1 }, (uint32_t)RAD_FILL_SIGMOID, 0);        /* sigma */
    o.t_fill(F, { 1 }, (uint32_t)RAD_FILL_SIGMOID, 0);        /* sigma_next */
    o.t(F, { M, n });                                         /* x, INOUT */
    return o.finish("flow_x0_step", operand, out);
}
int ref_gate_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* y */
    o.t(D, { 1, n });                                         /* g */
    o.t_idx({ 2 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);        /* cu_seqlens */
    o.t(D, { M, n });                                         /* x */
    return o.finish("gate_add", operand, out);
}
int ref_timestep_embed_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t_fill((uint32_t)RAD_F32, { M }, (uint32_t)RAD_FILL_SIGMOID, 0);   /* t: a sigma in (0, 1) */
    o.t(act_dt(p, n_p), { M, n });                                    /* y */
    return o.finish("timestep_embed", operand, out);
}
int ref_gather_rows2_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    (void)p; (void)n_p; (void)operand; (void)out;
    return RAD_E_UNSUPPORTED;   /* a signed index: half its draw would have to be negative */
}
/* ---- the per-lane ops. A lane is one image's slot of a state buffer, `stride` rows apart; each
 * entry reads or writes its own lane. Four lanes leaves some untouched, so a write into the wrong
 * lane shows up. Entries are as many as M needs at `stride` rows each, split by cu_seqlens. */
enum { REF_LANES = 4 };
static int64_t lane_entries(const RadParam* p, int n_p, int64_t M, int64_t stride) {
    const int64_t s = n_seq_of(p, n_p), need = (M + stride - 1) / stride;
    return std::min<int64_t>(std::max(s, need), REF_LANES);
}

int ref_lane_gather_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || n <= 0 || st <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = lane_entries(p, n_p, M, st);
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { REF_LANES * st, n });                            /* src: every lane */
    o.t_idx({ S }, REF_LANES, -1, 0);                         /* lane: may repeat */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t(D, { M, n });                                         /* y */
    return o.finish("lane_gather", operand, out);
}

int ref_lane_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || n <= 0 || st <= 0 || M > st) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* src */
    /* Distinct, two of the four: one image stored to the lanes of its guided pair. */
    o.t_idx({ 2 }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* lane */
    o.t(D, { REF_LANES * st, n });                            /* dst, INOUT */
    return o.finish("lane_store", operand, out);
}

int ref_randn_lanes_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || n <= 0 || st <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = lane_entries(p, n_p, M, st);
    if (M > S * st) return RAD_E_UNSUPPORTED;
    Opd o;
    /* Flags drawn over all three bits, so only some entries are first steps. */
    o.t_idx({ S }, 8, -1, 0);                                 /* flags */
    o.t_idx({ 2 * S }, (int64_t)1 << 30, -1, 0);              /* seed: two words an entry */
    /* Distinct: two first steps in one lane would be a race between their seeds. */
    o.t_idx({ S }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* lane */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t((uint32_t)RAD_BF16, { REF_LANES * st, n });           /* dst, INOUT */
    return o.finish("randn_lanes", operand, out);
}

int ref_flow_euler_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || n <= 0 || st <= 0) return RAD_E_UNSUPPORTED;
    /* Two equal-length entries where M allows, so a pair drawn from {0, 1} is always legal. -1
     * (unguided) and -2 (parked) are not drawable and are left to the engine's tests. */
    const int64_t S = (M % 2 == 0 && M / 2 <= st) ? 2 : 1;
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t F = (uint32_t)RAD_F32, B = (uint32_t)RAD_BF16;
    Opd o;
    o.t(B, { M, n });                                         /* v */
    o.t(F, { S });                                            /* sigma */
    o.t(F, { S });                                            /* sigma_next */
    o.t(F, { S });                                            /* cfg */
    o.t_idx({ S }, S, -1, 0);                                 /* pair */
    o.t_idx({ S }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* lane */
    o.t_idx({ S }, 8, -1, 0);                                 /* flags: preview and repaint drawn */
    o.t_idx({ 2 * S }, (int64_t)1 << 30, -1, 0);              /* seed */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t(B, { REF_LANES * st, n });                            /* inp_lat */
    o.t(F, { REF_LANES * st });                               /* inp_mask */
    o.t(B, { REF_LANES * st, n });                            /* state, INOUT */
    o.t(B, { REF_LANES * st, n });                            /* preview, INOUT */
    return o.finish("flow_euler", operand, out);
}

/* The k-diffusion sampler's two ops over flow_euler's entries: two equal-length entries where M
 * allows, the lanes f32, sigmas drawn in (0, 1) so every logarithm and root is of a positive
 * value. Flags draw only the first step (bit 1): the preview needs its optional operand and
 * neither op repaints; the hand-written cases take the rest. */
int ref_kd_input_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || n <= 0 || st <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = (M % 2 == 0 && M / 2 <= st) ? 2 : 1;
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t F = (uint32_t)RAD_F32;
    Opd o;
    o.t_fill(F, { S }, (uint32_t)RAD_FILL_SIGMOID, 0);       /* sigma */
    o.t_idx({ S }, S, -1, 0);                                 /* pair */
    o.t_idx({ S }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* lane */
    o.t_idx({ S }, 2, -1, 0);                                 /* flags */
    o.t_idx({ 2 * S }, (int64_t)1 << 30, -1, 0);              /* seed */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t(F, { REF_LANES * st, n });                            /* state, INOUT */
    o.t(act_dt(p, n_p), { M, n });                            /* y */
    return o.finish("kd_input", operand, out);
}

int ref_kd_step_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || n <= 0 || st <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = (M % 2 == 0 && M / 2 <= st) ? 2 : 1;
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t F = (uint32_t)RAD_F32;
    Opd o;
    o.t(act_dt(p, n_p), { M, n });                            /* out */
    o.t_fill(F, { S }, (uint32_t)RAD_FILL_SIGMOID, 0);       /* sigma */
    o.t_fill(F, { S }, (uint32_t)RAD_FILL_SIGMOID, 0);       /* sigma_next */
    o.t_fill(F, { S }, (uint32_t)RAD_FILL_SIGMOID, 0);       /* sigma_prev */
    o.t(F, { S });                                            /* cfg */
    o.t_idx({ S }, S, -1, 0);                                 /* pair */
    o.t_idx({ S }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* lane */
    o.t_idx({ S }, 2, -1, 0);                                 /* flags */
    o.t_idx({ 2 * S }, (int64_t)1 << 30, -1, 0);              /* seed */
    o.t_idx({ S }, 3, -1, 0);                                 /* sampler */
    o.t_idx({ S }, 64, -1, 0);                                /* step */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t(F, { REF_LANES * st, n });                            /* state, INOUT */
    o.t(F, { REF_LANES * st, n });                            /* hist, INOUT */
    o.t(F, { REF_LANES * st, n });                            /* preview, INOUT */
    return o.finish("kd_step", operand, out);
}

/* ---- an image decoder's convolutions and shortcuts. Sizes are read off the operands and M counts
 * output pixels, so this picks eight columns where M allows (a row with both edges and an interior)
 * and the input that produces that output; otherwise it declines. */
static int64_t cols_for(int64_t M, int64_t mult) {
    for (int64_t w = 8; w >= 1; w /= 2)
        if (w % mult == 0 && M % w == 0 && (M / w) % mult == 0) return w;
    return 0;
}

int ref_conv2d_nhwc_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    const int64_t k = gi(p, n_p, "k", 0), stride = gi(p, n_p, "stride", 1);
    const int64_t up = gi(p, n_p, "up", 1);
    const char* pad = gs(p, n_p, "pad");
    if (M <= 0 || cin <= 0 || cout <= 0 || k <= 0 || stride <= 0 || up <= 0)
        return RAD_E_UNSUPPORTED;
    const bool down = pad && !std::strcmp(pad, "down");
    const int64_t pads = down ? 1 : 2 * (k / 2);
    const int64_t Wo = cols_for(M, 1), Ho = Wo ? M / Wo : 0;
    if (!Wo) return RAD_E_UNSUPPORTED;
    /* The smallest upsampled input the output needs, which must be a whole input at `up`. */
    const int64_t Hv = (Ho - 1) * stride + k - pads, Wv = (Wo - 1) * stride + k - pads;
    if (Hv <= 0 || Wv <= 0 || Hv % up || Wv % up) return RAD_E_UNSUPPORTED;
    if ((Hv + pads - k) / stride + 1 != Ho || (Wv + pads - k) / stride + 1 != Wo)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { Hv / up, (Wv / up) * cin });                     /* x */
    o.t(D, { cout, cin, k, k });                              /* w */
    o.t(D, { cout });                                         /* b */
    o.t(D, { Ho, Wo * cout });                                /* y */
    return o.finish("conv2d_nhwc", operand, out);
}

/* conv2d_nhwc's geometry with the receptive fields as y's rows: M output pixels of cin k k values. */
int ref_im2col_nhwc_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), k = gi(p, n_p, "k", 0);
    const int64_t stride = gi(p, n_p, "stride", 1), up = gi(p, n_p, "up", 1);
    const char* pad = gs(p, n_p, "pad");
    if (M <= 0 || cin <= 0 || k <= 0 || stride <= 0 || up <= 0) return RAD_E_UNSUPPORTED;
    const bool down = pad && !std::strcmp(pad, "down");
    const int64_t pads = down ? 1 : 2 * (k / 2);
    const int64_t Wo = cols_for(M, 1), Ho = Wo ? M / Wo : 0;
    if (!Wo) return RAD_E_UNSUPPORTED;
    const int64_t Hv = (Ho - 1) * stride + k - pads, Wv = (Wo - 1) * stride + k - pads;
    if (Hv <= 0 || Wv <= 0 || Hv % up || Wv % up) return RAD_E_UNSUPPORTED;
    if ((Hv + pads - k) / stride + 1 != Ho || (Wv + pads - k) / stride + 1 != Wo)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { Hv / up, (Wv / up) * cin });                     /* x */
    o.t(D, { Ho * Wo, cin * k * k });                         /* y */
    return o.finish("im2col_nhwc", operand, out);
}

int ref_d2s_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    if (M <= 0 || cin <= 0 || cout <= 0) return RAD_E_UNSUPPORTED;
    const int64_t Wo = cols_for(M, 2);
    if (!Wo) return RAD_E_UNSUPPORTED;
    const int64_t Ho = M / Wo;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { Ho / 2, (Wo / 2) * cin });                       /* x */
    o.t(D, { Ho, Wo * cout });                                /* y, INOUT */
    return o.finish("d2s_add", operand, out);
}

int ref_s2d_avg_add_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    if (M <= 0 || cin <= 0 || cout <= 0) return RAD_E_UNSUPPORTED;
    const int64_t Wo = cols_for(M, 1), Ho = Wo ? M / Wo : 0;
    if (!Wo) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { 2 * Ho, 2 * Wo * cin });                         /* x */
    o.t(D, { Ho, Wo * cout });                                /* y, INOUT */
    return o.finish("s2d_avg_add", operand, out);
}

/* ---- attention over a paged prefix and the step's own rows. The prefix is `ctx` deep, else 37
 * (two whole blocks and part of a third at block size 16), exercising the block walk and the
 * partial block. */
int ref_attn_prefix_self_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M   = gi(p, n_p, "q_len", 0);
    const int64_t hd  = gi(p, n_p, "head_dim", 0);
    const int64_t bs  = gi(p, n_p, "block_size", 16);
    const int64_t gqa = gi(p, n_p, "gqa", 1);
    const int64_t nh  = gi(p, n_p, "n_head", gqa);
    if (M <= 0 || hd <= 0 || bs <= 0 || gqa <= 0 || nh <= 0 || nh % gqa) return RAD_E_UNSUPPORTED;
    const int64_t kvh = nh / gqa, S = n_seq_of(p, n_p);
    int64_t P = gi(p, n_p, "ctx", 0);
    if (P <= 0) P = 37;
    const int64_t max_blocks = (P + bs - 1) / bs;
    const uint32_t Q  = act_dt(p, n_p);
    const uint32_t kv = dt_parse(gs(p, n_p, "kv_dtype"));
    const uint32_t KV = kv == (uint32_t)RAD_DT_INVALID ? Q : kv;
    /* `max_ctx` 0 declares the prefix-free form: no cache, block table or prefix lengths. */
    const bool pre = gi(p, n_p, "max_ctx", -1) != 0;
    const char* pv = gs(p, n_p, "pv");
    const bool pv8 = pv && !std::strcmp(pv, "fp8");
    if (pv8 && pre) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(Q,  { M, nh, hd });                                   /* q */
    o.t(Q,  { M, kvh, hd });                                  /* k */
    o.t(pv8 ? (uint32_t)RAD_F8E4M3 : Q, { M, kvh, hd });      /* v */
    if (pre) {
        o.t(KV, { S * max_blocks, kvh, bs, 2 * hd });         /* kv_cache */
        o.t_idx({ S, max_blocks }, S * max_blocks, -1, 0);    /* block_table: may repeat */
        o.t_idx({ S }, 0, P, 0);                              /* seqused: the prefix depth */
    } else {
        o.skip();
        o.skip();
        o.skip();
    }
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.skip();                                                 /* k_descale */
    o.skip();                                                 /* v_descale */
    const char* qk = gs(p, n_p, "qk");
    if (qk && !std::strcmp(qk, "i8")) {
        o.t((uint32_t)RAD_I8, { M, kvh, hd });                /* k_i8 */
        o.t((uint32_t)RAD_F32, { M, kvh });                   /* k_scale */
    } else {
        o.skip();
        o.skip();
    }
    o.t(Q,  { M, nh, hd });                                   /* out */
    if (pv8) o.t((uint32_t)RAD_F32, { kvh * hd });            /* v_scale: one a column */
    else     o.skip();
    return o.finish("attn_prefix_self", operand, out);
}

int ref_chan_affine_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });
    o.t(D, { M, n });
    return o.finish("chan_affine", operand, out);
}
int ref_l2norm_chan_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* x */
    o.t(D, { n });                                            /* gamma */
    o.t(D, { M, n });                                         /* y */
    return o.finish("l2norm_chan", operand, out);
}
int ref_to_rgba8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(act_dt(p, n_p), { M, n });
    o.t((uint32_t)RAD_U8, { M, n });
    return o.finish("to_rgba8", operand, out);
}
int ref_gelu_tanh_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });
    o.t(D, { M, n });
    return o.finish("gelu_tanh", operand, out);
}
int ref_gelu_quick_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });
    o.t(D, { M, n });
    return o.finish("gelu_quick", operand, out);
}
int ref_scatter_add_rows_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const uint32_t D = act_dt(p, n_p);
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    Opd o;
    o.t(D, { M, n });                                      /* v */
    o.t_idx({ M }, 4 * M, -1, RAD_OPD_F_IDX_UNIQUE);       /* idx, distinct destinations */
    o.t(D, { 4 * M, n });                                  /* x, inout */
    return o.finish("scatter_add_rows", operand, out);
}

/* ================================================================== video generation
 *
 * The request's geometry is a device operand, and a description can only draw an index operand
 * as one value throughout, so every axis of `geom` is one extent c: a c x c x c grid (and, for
 * vtile_stitch, a chunk that is its video's first and last). The ranged M is the output's rows;
 * an M below what the grid writes declines, and rows past it are left as drawn. */

/* The tiles split_tiles (ref_video.cpp) lays over `len` latent units, 0 where none fit. */
static int64_t tile_count(int64_t len, int64_t tile, int64_t overlap) {
    if (len <= 0 || tile <= 0 || overlap < 0 || overlap >= tile) return 0;
    if (tile >= len) return len == tile ? 1 : 0;
    int64_t nt = (len + tile - 1) / tile;
    while (tile * nt - overlap * (nt - 1) - len < 0) ++nt;
    return nt;
}

/* A 2 x 2 x 2 grid of patches. */
int ref_patch_unfold_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "C", 0);
    const int64_t pt = gi(p, n_p, "pt", 0), ph = gi(p, n_p, "ph", 0), pw = gi(p, n_p, "pw", 0);
    const int64_t c = 2, vol = pt * ph * pw;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || M < c * c * c * vol) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, C * vol });                           /* x: patch rows */
    o.t_idx({ 3 }, 0, c, 0);                                  /* geom: (F, Hp, Wp) */
    o.t(D, { M, C });                                         /* y: voxel rows */
    return o.finish("patch_unfold", operand, out);
}

/* A latent of `frames` frames, as tall and as wide as it is long. */
int ref_vtile_gather_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), tile = gi(p, n_p, "tile", 0);
    const int64_t ov = gi(p, n_p, "overlap", 0), c = gi(p, n_p, "frames", 0);
    const int64_t period = gi(p, n_p, "period", 0), n_reg = gi(p, n_p, "n_reg", 0);
    const int64_t nt = tile_count(c, tile, ov);
    if (n <= 0 || nt <= 0 || n_reg < 0 || period < c * tile * tile + n_reg ||
        M < nt * nt * period)
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, n });                                 /* x: voxel rows */
    o.t(D, { n_reg > 0 ? n_reg : 1, n });                     /* reg */
    o.t_idx({ 3 }, 0, c, 0);                                  /* geom: (frames, H, W) */
    o.t(D, { M, n });                                         /* y */
    return o.finish("vtile_gather", operand, out);
}

/* The smallest grid the chunk's frames and the tile fit. */
int ref_vtile_stitch_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), tile = gi(p, n_p, "tile", 0);
    const int64_t ov = gi(p, n_p, "overlap", 0), S = gi(p, n_p, "scale", 0);
    const int64_t pt = gi(p, n_p, "pt", 0), skip = gi(p, n_p, "skip", 0);
    const int64_t keep = gi(p, n_p, "keep", 0), blend = gi(p, n_p, "blend", 0);
    const int64_t period = gi(p, n_p, "period", 0);
    if (tile <= 0 || S <= 0 || pt <= 0 || skip < 0 || keep <= 0 || blend < 0 || blend > keep)
        return RAD_E_UNSUPPORTED;
    const int64_t c = std::max(tile, (2 * skip + keep + blend + pt - 1) / pt);
    const int64_t nt = tile_count(c, tile, ov), npx = c * S * c * S;
    if (nt <= 0 || period < c * tile * tile || M < (keep + blend) * npx) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { nt * nt * period, 3 * pt * S * S });            /* x: the decoded tiles */
    o.t_idx({ 5 }, 0, c, 0);                                  /* geom: first and last set */
    o.t(D, { blend > 0 ? blend * npx : 1, 3 });               /* prev, INOUT */
    o.t((uint32_t)RAD_U8, { M, 3 });                          /* y */
    return o.finish("vtile_stitch", operand, out);
}

int ref_grid_pos3_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t((uint32_t)RAD_F32, { 3, M });                         /* y: one row a component */
    return o.finish("grid_pos3", operand, out);
}

/* ---- the latent upscaler. A volume is c x c x c voxels, c the largest whose cube M holds (the
 * ranged M is the output's rows), so a 3 x 3 x 3 volume and up has an interior voxel. Every geom
 * entry being c, resize_bilinear's description is the identity resize; its hand cases resize. */
static int64_t cube_side(int64_t M) {
    int64_t c = 0;
    while ((c + 1) * (c + 1) * (c + 1) <= M) ++c;
    return c;
}

int ref_conv3d_ndhwc_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    const int64_t k = gi(p, n_p, "k", 0), kt = gi(p, n_p, "kt", 0), c = cube_side(M);
    if (c <= 0 || cin <= 0 || cout <= 0 || k <= 0 || kt <= 0 || !(k & 1) || !(kt & 1))
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, cin });                               /* x: voxel rows */
    o.t(D, { cout, cin, kt, k, k });                          /* w */
    o.t(D, { cout });                                         /* b */
    o.t_idx({ 3 }, 0, c, 0);                                  /* geom: (T, H, W) */
    o.t(D, { M, cout });                                      /* res */
    o.t(D, { M, cout });                                      /* y */
    return o.finish("conv3d_ndhwc", operand, out);
}

/* The modulated form; `stats` at the size the op requires. */
int ref_group_norm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), G = gi(p, n_p, "groups", 0);
    if (M <= 0 || n <= 0 || G <= 0 || n % G) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                         /* x */
    o.t(D, { n });                                            /* w */
    o.t(D, { n });                                            /* b */
    o.t(D, { 2 * n });                                        /* mod: [scale | shift] */
    o.t((uint32_t)RAD_F32, { G * (2 + 3 * 1024) });           /* stats, INOUT */
    o.t(D, { M, n });                                         /* y */
    return o.finish("group_norm", operand, out);
}

int ref_tconv_dw_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), k = gi(p, n_p, "k", 0);
    const int64_t c = cube_side(M);
    if (c <= 0 || n <= 0 || k <= 0 || !(k & 1)) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, n });                                 /* x */
    o.t(D, { n, 1, k, 1, 1 });                                /* w */
    o.t(D, { n });                                            /* b */
    o.t_idx({ 3 }, 0, c, 0);                                  /* geom: (T, H, W) */
    o.t(D, { M, n });                                         /* y */
    return o.finish("tconv_dw", operand, out);
}

int ref_resize_bilinear_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), c = cube_side(M);
    if (c <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, n });                                 /* x */
    o.t_idx({ 5 }, 0, c, 0);                                  /* geom: (T, h, w, H, W) */
    o.t(D, { M, n });                                         /* y */
    return o.finish("resize_bilinear", operand, out);
}

/* patch_unfold's 2 x 2 x 2 grid of patches, folded. */
int ref_patch_fold_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "C", 0);
    const int64_t pt = gi(p, n_p, "pt", 0), ph = gi(p, n_p, "ph", 0), pw = gi(p, n_p, "pw", 0);
    const int64_t c = 2, vol = pt * ph * pw;
    if (C <= 0 || pt <= 0 || ph <= 0 || pw <= 0 || M < c * c * c) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c * vol, C });                           /* x: voxel rows */
    o.t_idx({ 3 }, 0, c, 0);                                  /* geom: (F, Hp, Wp) */
    o.t((uint32_t)RAD_F32, { M, C * vol });                   /* y: patch rows */
    return o.finish("patch_fold", operand, out);
}

/* sigma in (0, 1); the latent f32. */
int ref_flow_renoise_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t F = (uint32_t)RAD_F32;
    Opd o;
    o.t_fill(F, { 1 }, (uint32_t)RAD_FILL_SIGMOID, 0);        /* sigma */
    o.t_idx({ 2 }, (int64_t)1 << 30, -1, 0);                  /* seed: two words */
    o.t(F, { M, n });                                         /* x, INOUT */
    return o.finish("flow_renoise", operand, out);
}

/* ================================================================== video sparse attention
 *
 * The tiles come from the values of `segs` and `geom`, which a description draws only as
 * cumulative lengths and as one value c in every entry. So the grid is a cube of c frames and
 * c / patch rows and columns, c the smallest that gives every axis an edge box (more than one box,
 * the last short), and `segs` cuts M into M / (F Hp Wp) segments of the grid's rows each, the last
 * the video; an M not a multiple of the grid declines. `n_head` (default 2) and `dtype` (default
 * bf16) are the description's, as is `coarse`: 0 leaves the optional operands out.
 *
 * attn_vsa's `sel` is drawn as one value, the first video tile, which is a selection only at
 * k = 1; any other sparsity declines (the hand-built cases build selections,
 * libavx/avx_check_hand.cpp). */
namespace {

struct VsaGeo {
    int64_t M, hd, nh, c, n_seg, P, V, N, k;
    uint32_t D;
    bool coarse;
};

/* An axis of `len` units over boxes of `t` has more than one box, the last short; a box of one
 * unit needs only a second unit. */
bool vsa_edge(int64_t len, int64_t t) { return t > 1 ? len > t && len % t != 0 : len > 1; }

int vsa_geo(const RadParam* p, int n_p, VsaGeo* g) {
    g->M = gi(p, n_p, "M", 0);
    g->hd = gi(p, n_p, "head_dim", 0);
    g->nh = gi(p, n_p, "n_head", 2);
    const int64_t tt = gi(p, n_p, "tile_t", 0), th = gi(p, n_p, "tile_h", 0);
    const int64_t tw = gi(p, n_p, "tile_w", 0), patch = gi(p, n_p, "patch", 0);
    const double sparsity = rad_param_getf(p, n_p, "sparsity", -1.0);
    if (g->M <= 0 || g->hd <= 0 || g->nh <= 0 || tt <= 0 || th <= 0 || tw <= 0 || patch <= 0 ||
        !(sparsity >= 0.0 && sparsity < 1.0))
        return RAD_E_UNSUPPORTED;
    int64_t c = 1;
    while (c <= 4096 && !(vsa_edge(c, tt) && vsa_edge(c / patch, th) && vsa_edge(c / patch, tw)))
        ++c;
    const int64_t hp = c / patch, rows = c * hp * hp, TT = tt * th * tw;
    if (c > 4096 || g->M % rows != 0) return RAD_E_UNSUPPORTED;
    g->c = c;
    g->n_seg = g->M / rows;
    g->P = (g->n_seg - 1) * ((rows + TT - 1) / TT);
    g->V = ((c + tt - 1) / tt) * ((hp + th - 1) / th) * ((hp + tw - 1) / tw);
    g->N = g->P + g->V;
    const double keep = (1.0 - sparsity) * (double)g->V;
    g->k = std::max<int64_t>(1, std::min<int64_t>((int64_t)std::ceil(keep), g->V));
    g->D = act_dt(p, n_p);
    g->coarse = gi(p, n_p, "coarse", 1) != 0;
    return RAD_OK;
}

}  /* namespace */

int ref_vsa_select_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    VsaGeo g{};
    if (vsa_geo(p, n_p, &g) != RAD_OK) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(g.D, { g.M, g.nh, g.hd });                                 /* q */
    o.t(g.D, { g.M, g.nh, g.hd });                                 /* k */
    if (g.coarse) o.t(g.D, { g.M, g.nh, g.hd }); else o.skip();    /* v */
    o.t_idx({ g.n_seg + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* segs */
    o.t_idx({ RAD_VGEOM_N }, 0, g.c, 0);                           /* geom: c everywhere */
    /* A tile row and a slot past the selection's, which nothing writes. */
    o.t((uint32_t)RAD_I32, { g.nh, g.N + 1, g.k + 1 });            /* sel */
    if (g.coarse) o.t(g.D, { g.nh, g.N + 1, g.hd }); else o.skip();   /* coarse */
    return o.finish("vsa_select", operand, out);
}

int ref_attn_vsa_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    VsaGeo g{};
    if (vsa_geo(p, n_p, &g) != RAD_OK || g.k != 1) return RAD_E_UNSUPPORTED;
    const char* pv = gs(p, n_p, "pv");
    const bool pv8 = pv && !std::strcmp(pv, "fp8");
    Opd o;
    o.t(g.D, { g.M, g.nh, g.hd });                                 /* q */
    o.t(g.D, { g.M, g.nh, g.hd });                                 /* k */
    o.t(pv8 ? (uint32_t)RAD_F8E4M3 : g.D, { g.M, g.nh, g.hd });    /* v */
    o.t_idx({ g.n_seg + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* segs */
    o.t_idx({ RAD_VGEOM_N }, 0, g.c, 0);                           /* geom: c everywhere */
    o.t_idx({ g.nh, g.N, 1 }, g.N, g.P, 0);                        /* sel: the first video tile */
    if (g.coarse) {
        o.t(g.D, { g.nh, g.N, g.hd });                             /* coarse */
        o.t(g.D, { g.M, g.nh, g.hd });                             /* gate */
    } else {
        o.skip();
        o.skip();
    }
    o.t(g.D, { g.M, g.nh, g.hd });                                 /* out */
    const char* qk = gs(p, n_p, "qk");
    if (qk && !std::strcmp(qk, "i8")) {
        o.t((uint32_t)RAD_I8, { g.M, g.nh, g.hd });                 /* k_i8 */
        o.t((uint32_t)RAD_F32, { g.M, g.nh });                      /* k_scale */
    } else {
        o.skip();
        o.skip();
    }
    if (pv8) o.t((uint32_t)RAD_F32, { g.nh * g.hd });              /* v_scale: one a column */
    else     o.skip();
    return o.finish("attn_vsa", operand, out);
}

/* ================================================================== audio generation
 * Sequenced ops (the convolutions) are described over two sequences where M allows; lane ops over
 * REF_LANES lanes, as the image generator's are. */
static int64_t two_seqs(int64_t M) { return M >= 4 ? 2 : 1; }

int ref_conv1d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    const int64_t k = gi(p, n_p, "k", 0), causal = gi(p, n_p, "causal", 0);
    if (M <= 0 || cin <= 0 || cout <= 0 || k <= 0 || (!causal && (k & 1) == 0))
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const int64_t S = two_seqs(M);
    Opd o;
    o.t(D, { M, cin });                                       /* x */
    o.t(D, { cout, cin, k });                                 /* w */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* cu_seqlens */
    o.t(D, { M, cout });                                      /* res */
    o.t(D, { M, cout });                                      /* y */
    o.t(D, { cout });                                         /* b */
    return o.finish("conv1d", operand, out);
}

int ref_conv_t1d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), cin = gi(p, n_p, "cin", 0), cout = gi(p, n_p, "cout", 0);
    const int64_t s = gi(p, n_p, "stride", 0), k = gi(p, n_p, "k", 2 * s);
    const int64_t pad = gi(p, n_p, "pad", -1);
    if (M <= 0 || cin <= 0 || cout <= 0 || s <= 0 || k < s || (pad < 0 && ((k - s) & 1)))
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, cin });                                       /* x */
    o.t(D, { cin, cout, k });                                 /* w */
    o.t(D, { cout });                                         /* b */
    o.t_idx({ two_seqs(M) + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);   /* cu_seqlens */
    o.t(D, { M * s, cout });                                  /* y */
    return o.finish("conv_t1d", operand, out);
}

int ref_snake_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                                     /* x */
    /* alpha in (0, 1): a trained alpha is positive, and 1 / alpha near zero is no test. */
    o.t_fill((uint32_t)RAD_F32, { n }, (uint32_t)RAD_FILL_SIGMOID, 0);    /* alpha */
    o.t(D, { M, n });                                                     /* y */
    o.skip();                                                             /* beta */
    return o.finish("snake", operand, out);
}

int ref_snake_aa_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    const int64_t R = gi(p, n_p, "ratio", 0), k = gi(p, n_p, "k", 0);
    if (M <= 0 || n <= 0 || R <= 0 || k < R) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p), F = (uint32_t)RAD_F32;
    Opd o;
    o.t(D, { M, n });                                                     /* x */
    o.t_fill(F, { n }, (uint32_t)RAD_FILL_SIGMOID, 0);                    /* alpha */
    o.t_fill(F, { n }, (uint32_t)RAD_FILL_SIGMOID, 0);                    /* beta */
    o.t(F, { 1, 1, k });                                                  /* f_up */
    o.t(F, { 1, 1, k });                                                  /* f_down */
    o.t_idx({ two_seqs(M) + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);      /* cu_seqlens */
    o.t(D, { M, n });                                                     /* y */
    return o.finish("snake_aa", operand, out);
}

int ref_cfg_topk_sample_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), V = gi(p, n_p, "n_vocab", 0), rep = gi(p, n_p, "rep", 1);
    if (M <= 0 || V <= 0 || rep < 1 || M % rep) return RAD_E_UNSUPPORTED;
    const int64_t n = M / rep;
    Opd o;
    o.t((uint32_t)RAD_F32, { M, V });                         /* logits */
    o.t_idx({ 2 * n }, (int64_t)1 << 30, -1, 0);              /* seed */
    o.t_idx({ n }, 9000, -1, 0);                              /* frame */
    o.t_idx({ M }, V, -1, 0);                                 /* ids */
    o.skip();                                                 /* cfg */
    o.skip();                                                 /* lane */
    o.skip();                                                 /* hist */
    return o.finish("cfg_topk_sample", operand, out);
}

int ref_rvq_embed_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), D = gi(p, n_p, "n_embd", 0);
    const int64_t K = gi(p, n_p, "n_codes", 0), voc = gi(p, n_p, "vocab", 0);
    const int64_t off0 = gi(p, n_p, "off0", 0);
    if (M <= 0 || D <= 0 || K < 1 || voc <= 0 || off0 < 0) return RAD_E_UNSUPPORTED;
    const uint32_t B = (uint32_t)RAD_BF16;
    Opd o;
    o.t_idx({ K, M }, voc, -1, 0);                            /* codes */
    o.t(B, { off0 + voc, D });                                /* t0 */
    o.t(B, { (K > 1 ? K - 1 : 1) * voc, D });                 /* t1 */
    o.t(act_dt(p, n_p), { M, D });                            /* y */
    return o.finish("rvq_embed", operand, out);
}

int ref_frame_mix_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), D = gi(p, n_p, "n_embd", 0), L = gi(p, n_p, "L", 0);
    const int64_t st = gi(p, n_p, "stride", 0);
    if (M <= 0 || D <= 0 || L <= 0 || st <= 0 || M > REF_LANES) return RAD_E_UNSUPPORTED;
    const uint32_t A = act_dt(p, n_p);
    Opd o;
    o.t(A, { L * 2 * M, D });                                 /* hid: two branch rows a lane */
    o.t((uint32_t)RAD_F32, { L });                            /* logits */
    o.t_fill((uint32_t)RAD_F32, { 1 }, (uint32_t)RAD_FILL_SIGMOID, 0);   /* scale */
    o.t_idx({ M }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);       /* lane */
    o.t_idx({ M }, st + 1, -1, 0);                            /* frame, 0 included */
    o.t_idx({ REF_LANES }, 8, -1, 0);                         /* status */
    o.t(A, { REF_LANES * st, D });                            /* dst, INOUT */
    return o.finish("frame_mix_store", operand, out);
}

int ref_end_status_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), rep = gi(p, n_p, "rep", 1), end = gi(p, n_p, "end", 0);
    if (M <= 0 || rep < 1 || M > REF_LANES) return RAD_E_UNSUPPORTED;
    Opd o;
    /* Ids over [0, 2 end): about half the lanes end. */
    o.t_idx({ rep * M }, end > 0 ? 2 * end : 2, -1, 0);       /* ids */
    o.t_idx({ M }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* lane */
    o.t_idx({ REF_LANES }, 8, -1, 0);                         /* status, INOUT */
    return o.finish("end_status", operand, out);
}

int ref_frame_store_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), K = gi(p, n_p, "n_codes", 0);
    const int64_t st = gi(p, n_p, "stride", 0), rep = gi(p, n_p, "rep", 1);
    if (M <= 0 || K <= 0 || st <= 0 || rep < 1 || M > REF_LANES) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ K, rep * M }, 2048, -1, 0);                             /* codes */
    o.t_idx({ M }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);    /* lane */
    o.t_idx({ M }, st, -1, 0);                                        /* frame */
    o.t_idx({ REF_LANES }, 8, -1, 0);                                 /* status */
    o.t_idx({ REF_LANES * K, st }, 2048, -1, 0);                      /* dst, INOUT */
    return o.finish("frame_store", operand, out);
}

int ref_resample_nearest_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M / 3 + 1, n });                                 /* x: fewer rows than y */
    o.t(D, { M, n });                                         /* y */
    return o.finish("resample_nearest", operand, out);
}

int ref_fourier_embed_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0 || (n & 1)) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_fill((uint32_t)RAD_F32, { M }, (uint32_t)RAD_FILL_SIGMOID, 0);   /* t in (0, 1) */
    o.t((uint32_t)RAD_F32, { n / 2 });                        /* w */
    o.t(act_dt(p, n_p), { M, n });                            /* y */
    return o.finish("fourier_embed", operand, out);
}

int ref_win_input_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "n_lat", 0), Z = gi(p, n_p, "n_zero", 0);
    const int64_t Dc = gi(p, n_p, "n_cond", 0), st = gi(p, n_p, "stride", 0);
    if (M <= 0 || C <= 0 || Z < 0 || Dc <= 0 || st <= 0) return RAD_E_UNSUPPORTED;
    const int64_t S = lane_entries(p, n_p, M, st);
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t A = act_dt(p, n_p);
    Opd o;
    o.t(A, { REF_LANES * st, C });                            /* lat */
    o.t(A, { REF_LANES * st, Dc });                           /* cond */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* lat_cu */
    o.t_idx({ S }, REF_LANES, -1, 0);                         /* win_lane */
    o.t_idx({ S }, S, -1, 0);                                 /* pair */
    o.t(A, { M, C + Z + Dc });                                /* y */
    return o.finish("win_input", operand, out);
}

int ref_win_prep_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), st = gi(p, n_p, "stride", 0);
    const int64_t ov = gi(p, n_p, "ov", 0);
    if (M <= 0 || n <= 0 || st <= 0 || ov <= 0 || ov > st) return RAD_E_UNSUPPORTED;
    const int64_t S = lane_entries(p, n_p, M, st);
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t B = (uint32_t)RAD_BF16;
    Opd o;
    o.t_idx({ S }, 32, -1, 0);                                /* flags: every bit drawn */
    o.t_fill((uint32_t)RAD_F32, { S }, (uint32_t)RAD_FILL_SIGMOID, 0);   /* t */
    o.t_idx({ 2 * S }, (int64_t)1 << 30, -1, 0);              /* seed */
    o.t_idx({ S }, 64, -1, 0);                                /* win_index */
    /* Distinct: two entries moving one lane would race. */
    o.t_idx({ S }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* win_lane */
    o.t_idx({ S }, S, -1, 0);                                 /* pair */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* lat_cu */
    o.t(B, { REF_LANES * ov, n });                            /* prev */
    o.t(B, { REF_LANES * ov, n });                            /* noise, INOUT */
    o.t(B, { REF_LANES * st, n });                            /* lat, INOUT */
    return o.finish("win_prep", operand, out);
}

int ref_win_finish_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), Dc = gi(p, n_p, "n_cond", 0);
    const int64_t st = gi(p, n_p, "stride", 0), ov = gi(p, n_p, "ov", 0);
    if (M <= 0 || n <= 0 || Dc <= 0 || st <= 0 || ov <= 0 || ov > st) return RAD_E_UNSUPPORTED;
    const int64_t S = lane_entries(p, n_p, M, st);
    if (M > S * st) return RAD_E_UNSUPPORTED;
    const uint32_t B = (uint32_t)RAD_BF16;
    Opd o;
    o.t_idx({ S }, 32, -1, 0);                                /* flags */
    o.t_idx({ S }, S, -1, 0);                                 /* pair */
    o.t_idx({ S }, REF_LANES, -1, (uint32_t)RAD_OPD_F_IDX_UNIQUE);   /* win_lane */
    o.t_idx({ S + 1 }, 0, -1, (uint32_t)RAD_OPD_F_IDX_CU);    /* lat_cu */
    o.t(B, { REF_LANES * st, Dc });                           /* cond */
    o.t(B, { REF_LANES * st, n });                            /* lat, INOUT */
    o.t(B, { REF_LANES * ov, n });                            /* prev, INOUT */
    o.t(B, { REF_LANES * ov, Dc });                           /* prev_cond, INOUT */
    return o.finish("win_finish", operand, out);
}

int ref_chan_split_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), P = gi(p, n_p, "parts", 0);
    if (M <= 0 || n <= 0 || P <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, P * n });                                     /* x */
    o.t(D, { P * M, n });                                     /* y */
    return o.finish("chan_split", operand, out);
}

int ref_interleave2_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M });                                            /* a */
    o.t(D, { M });                                            /* b */
    o.t((uint32_t)RAD_F32, { M, 2 });                         /* y */
    return o.finish("interleave2", operand, out);
}

int ref_iota_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ M }, 1, -1, 0);                                 /* y */
    return o.finish("iota", operand, out);
}

int ref_cu_repeat_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ 1 }, 1000, -1, 0);                              /* len */
    o.t_idx({ M }, 1, -1, 0);                                 /* y */
    return o.finish("cu_repeat", operand, out);
}

int ref_int_affine_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0);
    if (M <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ 16 }, 1000, -1, 0);                             /* x */
    o.t_idx({ M }, 1, -1, 0);                                 /* y */
    return o.finish("int_affine", operand, out);
}

/* ================================================================== LTX-2 (ref_ltx.cpp) */
int ref_rope_split_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), nh = gi(p, n_p, "n_head", 0);
    const int64_t hd = gi(p, n_p, "head_dim", 0), nd = gi(p, n_p, "n_dims", 0);
    if (M <= 0 || nh <= 0 || hd <= 0 || nd <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(act_dt(p, n_p), { M, nh * hd });                       /* x, INOUT */
    o.t((uint32_t)RAD_F32, { nd, M });                         /* positions */
    return o.finish("rope_split", operand, out);
}

int ref_head_gate_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), nh = gi(p, n_p, "n_head", 0);
    const int64_t hd = gi(p, n_p, "head_dim", 0);
    if (M <= 0 || nh <= 0 || hd <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, nh });                                         /* g */
    o.t(D, { M, nh * hd });                                    /* x, INOUT */
    return o.finish("head_gate", operand, out);
}

/* The four guidance passes bound, scales and sigmas in (0, 1): every term taken. */
int ref_ltx_guide_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0);
    if (M <= 0 || n <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p), F = (uint32_t)RAD_F32;
    Opd o;
    for (int i = 0; i < 4; ++i) o.t(D, { M, n });              /* vc, vu, vs, vm */
    o.t_fill(F, { 4 }, (uint32_t)RAD_FILL_SIGMOID, 0);         /* scales */
    o.t_fill(F, { 2 }, (uint32_t)RAD_FILL_SIGMOID, 0);         /* sched */
    o.t(F, { kLtxGuideStats });                                /* stats, INOUT */
    o.t(F, { M, n });                                          /* x, INOUT */
    return o.finish("ltx_guide", operand, out);
}

/* The rest of LTX-2's: its VAE's norm, resampling and patches, its vocoder's spectrogram and
 * resampler. A volume op's `geom` is the cube of side cube_side(M), M its input's rows. */
int ref_pixel_norm_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), n = gi(p, n_p, "n", 0), L = gi(p, n_p, "ileave", 1);
    if (M <= 0 || n <= 0 || L <= 0 || M % L) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M, n });                                          /* x */
    o.t(D, { M / L, n * L });                                  /* y */
    return o.finish("pixel_norm", operand, out);
}

int ref_d2s3d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "C", 0), c = cube_side(M);
    const int64_t st = gi(p, n_p, "st", 1), sh = gi(p, n_p, "sh", 1), sw = gi(p, n_p, "sw", 1);
    if (c <= 0 || C <= 0 || st <= 0 || sh <= 0 || sw <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, C * st * sh * sw });                   /* x */
    o.t_idx({ 3 }, 0, c, 0);                                   /* geom */
    o.t(D, { (c * st - st + 1) * c * sh * c * sw, C });        /* y */
    return o.finish("d2s3d", operand, out);
}

int ref_s2d3d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "C", 0), c = cube_side(M);
    const int64_t st = gi(p, n_p, "st", 1), sh = gi(p, n_p, "sh", 1), sw = gi(p, n_p, "sw", 1);
    const int64_t tpre = gi(p, n_p, "tpre", 0), g = gi(p, n_p, "group", 1);
    const char* mode = gs(p, n_p, "mode");
    const bool mean = mode && !std::strcmp(mode, "mean");
    const int64_t P = C * st * sh * sw;
    if (c <= 0 || C <= 0 || st <= 0 || sh <= 0 || sw <= 0 || tpre < 0 || g <= 0 ||
        (c + tpre) % st || c % sh || c % sw || (mean && P % g))
        return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { c * c * c, C });                                  /* x */
    o.t_idx({ 3 }, 0, c, 0);                                   /* geom */
    o.t(D, { ((c + tpre) / st) * (c / sh) * (c / sw), mean ? P / g : P });   /* y, INOUT */
    return o.finish("s2d3d", operand, out);
}

int ref_pix_patch_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), pp = gi(p, n_p, "p", 0), C = gi(p, n_p, "C", 0);
    const int64_t w = cols_for(M, 1), h = w ? M / w : 0;
    if (!w || pp <= 0 || C <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { h * pp, w * pp * C });                            /* x: the image */
    o.t(D, { h, w * C * pp * pp });                            /* y: patch rows */
    return o.finish("pix_patch", operand, out);
}

/* attn_na3d over a c x c x c volume, two heads. */
int ref_attn_na3d_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), hd = gi(p, n_p, "head_dim", 0), c = cube_side(M);
    if (c <= 0 || hd <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    const int64_t V = c * c * c;
    Opd o;
    o.t(D, { V, 2 * hd });                                     /* q */
    o.t(D, { V, 2 * hd });                                     /* k */
    o.t(D, { V, 2 * hd });                                     /* v */
    o.t_idx({ 3 }, 0, c, 0);                                   /* geom */
    o.t(D, { V, 2 * hd });                                     /* out */
    return o.finish("attn_na3d", operand, out);
}

int ref_vol_index_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), c = cube_side(M);
    if (c <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ 3 }, 0, c, 0);                                   /* geom: c x c x c */
    o.t_idx({ 3, c * c * c }, 1, -1, 0);                       /* y */
    return o.finish("vol_index", operand, out);
}

/* The tiled decode's ops: a c x c x c volume (every geom entry c), tiles and boxes inside it. */
int ref_tile_grid_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), stride = gi(p, n_p, "stride", 0);
    int64_t k = 0;
    while ((k + 1) * (k + 1) <= M) ++k;
    if (k <= 0 || stride <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t_idx({ 3 }, 0, k * stride, 0);                          /* geom: k x k tiles */
    o.t_idx({ M, 8 }, 1, -1, 0);                               /* y */
    return o.finish("tile_grid", operand, out);
}

int ref_tile_crop_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), C = gi(p, n_p, "C", 0);
    int64_t k = 0;
    while ((k + 1) * (k + 1) * (k + 1) <= M) ++k;
    if (k <= 0 || C <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { 8 * k * k * k, C });                              /* x: 2k x 2k x 2k */
    o.t_idx({ 3 }, 0, 2 * k, 0);                               /* geom */
    o.t_idx({ 6 }, 0, k, 0);                                   /* box: k x k x k at (k, k, k) */
    o.t(D, { k * k * k, C });                                  /* y */
    return o.finish("tile_crop", operand, out);
}

int ref_tile_stitch_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), tile = gi(p, n_p, "tile", 0);
    const int64_t stride = gi(p, n_p, "stride", 0), sc = gi(p, n_p, "scale", 0);
    const int64_t pp = gi(p, n_p, "p", 0), C = gi(p, n_p, "C", 0);
    if (tile <= 0 || stride <= 0 || stride > tile || sc <= 0 || pp <= 0 || C <= 0)
        return RAD_E_UNSUPPORTED;
    int64_t c = 0;
    while ((c + 1) * (c + 1) * sc * pp <= M) ++c;
    if (c <= 0) return RAD_E_UNSUPPORTED;
    int64_t span = 0;                                          /* the tiles' rows along an axis */
    for (int64_t s0 = 0; s0 < c; s0 += stride) span += std::min(tile, c - s0);
    Opd o;
    o.t(act_dt(p, n_p), { c * sc * sc * span * span, C * pp * pp });   /* x: c frames of tiles */
    o.t_idx({ 3 }, 0, c, 0);                                   /* geom */
    o.t_idx({ 1 }, 0, 0, 0);                                   /* first: frame 0 */
    o.t((uint32_t)RAD_U8, { c * c * sc * pp, c * sc * pp * C });  /* y: every frame */
    return o.finish("tile_stitch", operand, out);
}

int ref_patch_pix8_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), pp = gi(p, n_p, "p", 0), C = gi(p, n_p, "C", 0);
    const int64_t w = cols_for(M, 1), R = w ? M / w : 0;
    if (!w || pp <= 0 || C <= 0) return RAD_E_UNSUPPORTED;
    Opd o;
    o.t(act_dt(p, n_p), { R, w * C * pp * pp });              /* x: patch rows */
    o.t((uint32_t)RAD_U8, { R * pp, w * pp * C });            /* y: bytes */
    return o.finish("patch_pix8", operand, out);
}

int ref_stft_logmel_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), hop = gi(p, n_p, "hop", 0), nfft = gi(p, n_p, "nfft", 0);
    const int64_t nm = gi(p, n_p, "n_mel", 0), ch = gi(p, n_p, "ch", 0), F = nfft / 2 + 1;
    if (M <= 0 || hop <= 0 || nfft < hop || nm <= 0 || ch <= 0) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M * hop, ch });                                   /* x: samples */
    o.t(D, { 2 * F, 1, nfft });                                /* basis */
    o.t(D, { nm, F });                                         /* mel */
    o.t(D, { M, ch * nm });                                    /* y */
    return o.finish("stft_logmel", operand, out);
}

int ref_bwe_up_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    const int64_t M = gi(p, n_p, "M", 0), r = gi(p, n_p, "ratio", 0), ch = gi(p, n_p, "ch", 0);
    if (M <= 0 || r <= 0 || ch <= 0 || M % r) return RAD_E_UNSUPPORTED;
    const uint32_t D = act_dt(p, n_p);
    Opd o;
    o.t(D, { M / r, ch });                                     /* x: the low-rate waveform */
    o.t(D, { M, ch });                                         /* res */
    o.t(D, { M, ch });                                         /* y */
    return o.finish("bwe_up", operand, out);
}
