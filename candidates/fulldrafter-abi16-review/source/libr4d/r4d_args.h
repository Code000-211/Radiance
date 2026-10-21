/* r4d_args.h -- how a libr4d entry point reads its arguments.
 *
 * THERE IS NO ADAPTER LAYER. An entry point takes `(const RadArgs*, RadStream)` and returns a
 * status, which is the SAME contract RadLaunchFn publishes to every other plugin. An adapter
 * between the core and a first-party library would be a statement that the interface is wrong: an
 * out-of-tree author could not write the same kernel without also writing the same adapter, and
 * that adapter would live in this repository where they cannot reach it.
 *
 * What this header holds is the part that is not translation: reading a parameter, finding an
 * operand, asking whether a tensor is contiguous. Every plugin needs it, none of it is libr4d's,
 * and most of it is one line over rad_abi.h -- which is the shape a shared helper should have.
 *
 * Nothing here includes r4d.h or HIP: a host translation unit (the row table, the layouts, the
 * shape hooks) includes this too, and that split is what lets the tables a transcription error
 * hides in be compiled and read on a machine with no ROCm.
 */
#ifndef R4D_ARGS_H
#define R4D_ARGS_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ================================================================== parameter access
 *
 * ONE-LINE FORWARDS. rad_args_geti and rad_args_gets are static inline in rad_abi.h -- the header
 * a third-party plugin gets -- so every plugin shares ONE definition rather than writing its own.
 * These keep libr4d's spelling for its own call sites and forward. */
static inline int r4d_geti(const RadArgs* a, const char* k, long long* out) {
    return rad_args_geti(a, k, out);
}

static inline long long r4d_geti_or(const RadArgs* a, const char* k, long long dflt) {
    return rad_args_geti_or(a, k, dflt);
}

static inline const char* r4d_gets(const RadArgs* a, const char* k) {
    return rad_args_gets(a, k);
}

/* A float parameter is RAD_P_F64 (rad_types.h, docs/OPS.md "The four rules"), so `dval` is what
 * arrives and the step path runs no strtod. This one does NOT forward, because it
 * is deliberately laxer than rad_args_getf_or: the string and integer spellings are still read,
 * since a tool or a selftest may call a launch directly with no declare in front of it to enforce
 * the type, and failing such a caller over a spelling tells it nothing useful. */
static inline double r4d_getf(const RadArgs* a, const char* k, double dflt) {
    if (!a || !k) return dflt;
    for (int i = 0; i < a->n_p; ++i) {
        if (!a->p[i].key || std::strcmp(a->p[i].key, k) != 0) continue;
        if (a->p[i].kind == RAD_P_F64) return a->p[i].dval;
        if (a->p[i].kind == RAD_P_INT) return (double)a->p[i].ival;
        if (a->p[i].kind == RAD_P_STR && a->p[i].sval && a->p[i].sval[0]) {
            char* end = nullptr;
            const double v = std::strtod(a->p[i].sval, &end);
            return (end && end != a->p[i].sval) ? v : dflt;
        }
        return dflt;
    }
    return dflt;
}

/* An enumerated parameter. `act` is "silu" or "sigmoid" and NOT 0 or 1: a --debug-graph dump
 * reading `act=1` is exactly the "unclear what actually runs" complaint this engine exists to
 * answer. The mapping to libr4d's integer belongs here, in the shim, and not in the caller.
 * Returns -1 for a name that is not in the list, which every caller turns into RAD_E_INVAL rather
 * than running the other activation. `alt` is libr4d's own integer spelling of the same key --
 * `mode` on gated_had_quant_i8 -- consulted only when the name is absent. */
static inline int r4d_act_code(const RadArgs* a, const char* key, const char* alt, int dflt) {
    if (const char* s = r4d_gets(a, key)) {
        if (!std::strcmp(s, "silu") || !std::strcmp(s, "swish")) return 0;
        if (!std::strcmp(s, "sigmoid")) return 1;
        return -1;
    }
    long long v = 0;
    if (alt && r4d_geti(a, alt, &v)) return (v == 0 || v == 1) ? (int)v : -1;
    return dflt;
}

/* Same, off a RadParam list rather than RadArgs -- init() and the layout hooks get the bare list.
 * rad_plugin.h calls these rad_param_*; these keep libr4d's spelling for its call sites. */
static inline int r4d_pgeti(const RadParam* p, int n_p, const char* k, long long* out) {
    const RadParam* f = rad_param_find(p, n_p, k);
    if (!f || f->kind != RAD_P_INT) return 0;
    if (out) *out = f->ival;
    return 1;
}

static inline long long r4d_pgeti_or(const RadParam* p, int n_p, const char* k, long long dflt) {
    return rad_param_geti(p, n_p, k, dflt);
}

static inline const char* r4d_pgets(const RadParam* p, int n_p, const char* k) {
    return rad_param_gets(p, n_p, k, nullptr);
}

/* ================================================================== operand access
 *
 * An absent optional operand is a null RadTensor.data (rad_abi.h). r4d_opt() returns null for
 * both "index past the end" and "present but null", so a shim's optional-operand test is one
 * pointer test. r4d_req() is the same lookup for a required operand and the caller turns a null
 * into RAD_E_INVAL -- a missing required operand is a bug in the issue, not a shape this kernel
 * declines.
 */
static inline const RadTensor* r4d_opt(const RadArgs* a, int i) { return rad_arg_in(a, i); }

static inline const void* r4d_p(const RadTensor* t) { return t ? t->data : nullptr; }

/* ...and the same for an operand a kernel WRITES. RadTensor::data is `void*` already; this exists
 * so an output cast reads as an output cast rather than as a pointer laundered through `long`. */
static inline void* r4d_wp(const RadTensor* t) { return t ? t->data : nullptr; }
static inline long r4d_addr(const RadTensor* t) { return t ? (long)(uintptr_t)t->data : 0L; }

static inline long long r4d_dim(const RadTensor* t, int i) {
    if (!t || i < 0 || (uint32_t)i >= t->rank) return 0;
    return t->shape[i];
}

static inline long long r4d_stride(const RadTensor* t, int i) {
    if (!t || i < 0 || (uint32_t)i >= t->rank) return 0;
    return t->stride[i];
}

static inline long long r4d_numel(const RadTensor* t) { return rad_tensor_numel(t); }

/* [rows..., K] flattened to (M, K). Every per-row kernel in libr4d owns whole ROWS -- one
 * workgroup per row -- so the leading axes are only a row count and just the last axis is
 * structural. A rank-3 activation is a rank-2 one here by construction, which is the same fold
 * the GEMM has to make explicitly (see gemm_ops). */

static inline int r4d_rows_of(const RadTensor* t, long long* M, long long* K) {
    if (!t || t->rank < 1) return RAD_E_SHAPE;
    *K = t->shape[t->rank - 1];
    *M = 1;
    for (uint32_t i = 0; i + 1 < t->rank; ++i) *M *= t->shape[i];
    return (*M >= 1 && *K >= 1) ? RAD_OK : RAD_E_SHAPE;
}

/* Dense row-major over the whole tensor. libr4d's entry points take raw pointers and, where they
 * take a pitch at all, take exactly ONE -- so anything they do not pitch has to be tight, and the
 * shim is where that is checked. This is the "necessary condition only" boundary from spec §2.1:
 * the constraint table settles the geometry, contiguity is a property of the call. */
static inline int r4d_contig(const RadTensor* t) { return rad_tensor_is_contiguous(t); }

/* Contiguous in every axis but the leading one, whose stride is the row pitch libr4d takes as an
 * explicit argument (`xpitch`, `a_stride`, `xrow`, ...). Returns the pitch in ELEMENTS, or -1. */
static inline long long r4d_row_pitch(const RadTensor* t) {
    if (!t || t->rank == 0) return -1;
    if (t->rank == 1) return 1;
    long long acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return -1;
        acc *= t->shape[i];
    }
    return t->stride[0];
}

/* [rows..., n] read as (rows, n): the leading axes are a row count and only the last is
 * structural. Returns the pitch in ELEMENTS, or -1.
 *
 * THE COLUMN-SLICE CASE MUST REFUSE. A slice of a wider buffer -- k out of a fused
 * [tok, 2*kv_dim] projection, read as [tok*n_head_kv, head_dim] for the per-head norm -- is rows
 * of `n` that are `n` apart INSIDE a token and a whole buffer row apart BETWEEN tokens. No single
 * pitch describes that, and answering with the leading stride gives the between-tokens one: the
 * consumer then reads every n_head_kv-th head against the right numbers and the rest against
 * garbage, silently, from an op whose own arithmetic is correct. So the trailing axes must be the
 * WHOLE of the row before their pitch means anything. */
static inline long long r4d_rows_pitch(const RadTensor* t, long long n, long long* rows) {
    if (!t || t->rank < 1 || n <= 0) return -1;
    const long long nel = rad_tensor_numel(t);
    if (nel % n) return -1;
    *rows = nel / n;
    if (t->rank == 1) return n;
    if (t->shape[t->rank - 1] == n) {
        if (t->stride[t->rank - 1] != 1) return -1;
        return t->rank >= 2 ? t->stride[t->rank - 2] : n;
    }
    if (rad_tensor_is_contiguous(t)) return n;
    long long trail = 1;
    for (int i = 1; i < (int)t->rank; ++i) trail *= t->shape[i];
    return trail == n ? r4d_row_pitch(t) : -1;
}

/* 16-byte alignment. Every libr4d kernel that pushes over P2P or issues a global_load_b128 needs
 * it, and the kernels themselves do not check, so the shim is where it is checked. */
static inline int r4d_aligned(const void* p, unsigned bytes) {
    return p ? (((uintptr_t)p & (uintptr_t)(bytes - 1)) == 0) : 1;
}

/* ================================================================== which check refused
 *
 * A shim rejects a call in a dozen places and returns the same three or four codes from all of
 * them, so the engine's report -- "kernel X refused op Y at M=5: shape this kernel does not serve"
 * -- names the kernel and the geometry and says nothing about WHICH condition failed. Distinct
 * causes are then indistinguishable in the report: an activation declared [tokens, heads, width]
 * where a shim expected [tokens, width] prints identically to any other refusal at that geometry.
 *
 * So: wrap a refusal in R4D_NO() and it names its own line when RAD_R4D_TRACE is set in the
 * environment. Off, it is the bare return it replaces -- one load of a cached flag, on a path that
 * is about to fail the step anyway. */
static inline int r4d_trace_on(void) {
    static int on = -1;
    if (on < 0) { const char* e = std::getenv("RAD_R4D_TRACE"); on = (e && *e && *e != '0'); }
    return on;
}
static inline int r4d_refused(int code, const char* file, int line) {
    if (r4d_trace_on()) std::fprintf(stderr, "[r4d] %s:%d refused -> %d\n", file, line, code);
    return code;
}
#define R4D_NO(code) r4d_refused((code), __FILE__, __LINE__)

/* ================================================================== dtype helpers */
/* Widest world any libr4d all-reduce serves. Fixed by the 8-rank entry points' argument lists,
 * which name seven peers explicitly. */
enum { R4D_MAX_RANKS = 8, R4D_MAX_PEERS = 7 };

/* libr4d's all-reduce dtype code: 0 bf16, 1 fp16, 2 fp32. Returns -1 for anything else. */
static inline int r4d_ar_dtype_code(const char* s) {
    if (!s) return -1;
    if (!std::strcmp(s, "bf16")) return 0;
    if (!std::strcmp(s, "fp16") || !std::strcmp(s, "f16")) return 1;
    if (!std::strcmp(s, "fp32") || !std::strcmp(s, "f32")) return 2;
    if (!std::strcmp(s, "i32") || !std::strcmp(s, "u32")) return 2;
    return -1;
}

static inline int r4d_ar_esize(int code) { return code == 2 ? 4 : 2; }


/* ================================================================== weight layout constants
 *
 * These are libr4d's compile-time constants, restated here because r4d_layout.cpp must not include
 * r4d.h (it is host code compiled without ROCm) and because the .rad container records them in the
 * layout tag. THE BUILD CHECKS THEM: libr4d/CMakeLists.txt passes the same
 * -DR4D_GEMM_*_GROUP=128 to libr4d's translation units, and r4d_gemm_args.h static_asserts the
 * group constants against libr4d's own macros, so a divergence is a compile error rather than a
 * silently wrong weight file.
 */
enum {
    R4D_W4_GROUP = 128,   /* K per (scale, zero) dword; libr4d R4D_GEMM_W4_GROUP  */
    R4D_W4_KPB   = 64,    /* K per packed block: 4 k steps, 16 bytes per lane     */
    R4D_W8_GROUP = 128,   /* libr4d R4D_GEMM_W8_GROUP and R4D_GEMM_W8A8_GROUP     */
    R4D_W8_KPB   = 32,    /* K per packed block: 2 k steps, 16 bytes per lane     */
    R4D_W2_GROUP = 128,   /* libr4d R4D_GEMM_W2A8_GROUP                           */
    R4D_MXFP4_GROUP = 32, /* K per E8M0 exponent, fixed by the OCP MX format      */
    /* The block-scaled fp8 family's block, on BOTH axes. Unlike every other constant here it is not
     * a choice this project's quantiser made: it is `quantization_config.weight_block_size` in the
     * checkpoint's own config.json, and the weight bytes on disk are laid out to it. It cannot be
     * static_asserted against libr4d's `r4d_gemm_fp8a8_block()` because that is a function and not
     * a macro, so r4d_gemm_fp8a8.hip checks it at launch -- a scalar compare against a constant,
     * which is nothing, and the alternative is a silently wrong scale grid. */
    R4D_FP8_BLOCK = 128,
    R4D_WAVE     = 32,
    R4D_NTILE    = 16     /* output rows per WMMA n-tile                          */
};

/* Plugin-private dtypes. The core never interprets these: rad_types.h reserves everything at or
 * above RAD_DT_PLUGIN_BASE for exactly this, and the layout hook is what sizes them. */
enum {
    R4D_DT_W4_FRAG    = RAD_DT_PLUGIN_BASE + 0,  /* 4-bit codes in WMMA fragment order      */
    R4D_DT_W4_SZ      = RAD_DT_PLUGIN_BASE + 1,  /* (f16 scale, f16 -(1024+zero)) dwords    */
    R4D_DT_W8_FRAG    = RAD_DT_PLUGIN_BASE + 2,  /* 8-bit codes in the same fragment order  */
    R4D_DT_W8_S       = RAD_DT_PLUGIN_BASE + 3,  /* f16-only scale plane (w8a8)             */
    R4D_DT_W2_FRAG    = RAD_DT_PLUGIN_BASE + 4,
    R4D_DT_MXFP4_FRAG = RAD_DT_PLUGIN_BASE + 5,
    R4D_DT_E8M0       = RAD_DT_PLUGIN_BASE + 6,  /* [K/32][N] shared exponents, and Wref[N]  */
    /* The grouped MoE GEMM's 4-bit expert weight: ROW-MAJOR packed nibbles, each a code into a
     * sixteen-entry table, expanded to E4M3 on the way into LDS. Not R4D_DT_W4_FRAG -- that plane
     * is a WMMA fragment permutation for a kernel that loads its weight straight from global, and
     * this one stages through LDS because its rows are gathered. See r4d_layout.cpp. */
    R4D_DT_MOEW4      = RAD_DT_PLUGIN_BASE + 7,
    /* The same GEMM's 5-bit expert weight: each row is R4D_DT_MOEW4's nibble plane of the codes'
     * low four bits -- a magnitude -- followed by one bit a code, the sign, a dword a 32
     * elements in the order the kernel merges them (r4d_layout.cpp, moew5_row). Five bits an
     * element, so a row is 5K/8 bytes. */
    R4D_DT_MOEW5      = RAD_DT_PLUGIN_BASE + 8
};

/* THE GROUPED MoE GEMM'S CODE TABLES, as the four dwords w4_e4m3's byte permutes read
 * (r4d_moe.hip): entries 0..3, 4..7, 8..11 and 12..15, one E4M3 byte an entry, the lowest entry in
 * the low byte. `w4a8` and `w4a8h` decode a two's complement nibble, -8..7. `w4nla8h` decodes
 * libquant's `table=w4nl` -- magnitudes ascending, then their negatives -- and kW4nlTable is that
 * table as the floats a container stores, which the layout hook compares the container's own table
 * plane with before it serves a byte of it. */
static constexpr uint32_t kW4IntPerm[4] = { 0x44403800u, 0x4E4C4A48u, 0xCACCCED0u, 0xB8C0C4C8u };
static constexpr uint32_t kW4nlPerm[4]  = { 0x504B4539u, 0x5B595653u, 0xD0CBC5B9u, 0xDBD9D6D3u };
static constexpr float kW4nlTable[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                                          -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f,
                                          -22.0f };

/* `w5nl64a8h` DECODES libquant's `table=w5nl`: thirty-two levels, the sixteen magnitudes
 * ascending and then their negatives, so a code's high bit is its sign and its low four bits index
 * the magnitudes. The kernel expands the magnitude as it does a w4nl code, out of kW5nlMagPerm --
 * those sixteen magnitudes' E4M3 bytes in w4_e4m3's dword order -- and sets the byte's sign bit
 * from the sign plane. Every level is exact in E4M3. */
static constexpr uint32_t kW5nlMagPerm[4] = { 0x4B484235u, 0x5453514Fu, 0x5A595856u, 0x5F5D5C5Bu };
static constexpr float kW5nlTable[32] = {
    0.8125f, 2.5f, 4.0f, 5.5f, 7.5f, 9.0f, 11.0f, 12.0f,
    14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 30.0f,
    -0.8125f, -2.5f, -4.0f, -5.5f, -7.5f, -9.0f, -11.0f, -12.0f,
    -14.0f, -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -30.0f };

/* `w4nl64a8h`'S, `w4nl32a8h`'S AND `w5nl64a8h`'S FIXED SECOND LEVEL: a weight's E4M3 scale a (row,
 * 64 or 32 of K) multiplies it, and a container states it (libquant's scale2_value=, one f32 for
 * each weight), which the layout hook compares with this before serving a byte. It is fixed rather
 * than per row because the experts' scales -- with the rotation's 1/128 folded in -- sit between
 * 2^-15 and 2^-10.6 on Qwen3.8-Flash-Next at a 64 (a 32's sit a little lower, and w5nl's, over a
 * top level of 30 rather than 22, half a binade lower), so under 2^-13 every one is a normal E4M3
 * value with binades to spare either side, and E4M3's three mantissa bits are the same in every
 * binade a row level would have put it in. */
static constexpr float kW4G64Scale = 0x1p-13f;


/* ================================================================== skinny-GEMM variants
 *
 * libr4d's five skinny-GEMM knobs. Exposed as RadVariant.config so `rad-tune` can walk them and
 * `--debug-graph` can name the pick (spec §15). The axis earns its keep because the best setting
 * is per (N, K) AND per M band, and the spread between the best and a single fixed default is
 * large enough to matter on every shape in a step.
 *
 * sk == 0 means "choose the largest legal split count at issue". The split-K constraint is
 * K % (SK * group) == 0, which depends on K, so no fixed default is legal at every shape -- and a
 * default that is illegal at a shape would make an UNTUNED install fail rather than be slow, which
 * spec §15 says is the wrong way round. The sentinel is the plugin's own rule, declared here and
 * resolved in one place (r4d_gemm_args.h:pick_sk), not a hidden fallback.
 */
struct R4dGemmCfg { int wv, sk, mb, npw, nt; };

/* The tiled and prefill GEMMs take a single opaque variant index into libr4d's own switch. Only
 * arms that compute a CORRECT result are listed. Most arms of that switch are cost ablations that
 * compute wrong answers by construction, and a tuner handed one would benchmark a kernel that does
 * not compute the GEMM and then pick it. */
struct R4dTileCfg { int variant; int bm; int bn; };

/* Split-KV segment count for the paged decode kernels. 0 is libr4d's own split law, which is
 * measured (r4d_attn_paged_h256_gqa6.hip carries the table) and reads num_seqs and max_ctx that
 * only exist at issue -- so it stays the default and the fixed counts are there for a tuner that
 * has a concrete shape. RadScratchFn is handed the same variant index, so the arena is sized for
 * whatever this picks rather than for the law's answer at some other shape. */
struct R4dSplitCfg { int splits; };

/* The all-reduce launch geometry. libr4d takes nblocks/nthreads/drain/acq (and `pub` at width) as
 * arguments rather than compiling them in. The defaults are nthreads=1024, drain=3
 * (s_wait_storecnt, which is what fine-grained uncached scratch requires), acq=0 and pub=1, with
 * nblocks scaled by message size and clamped. nblocks == 0 selects that scaling rule. */
struct R4dArCfg { int nblocks; int nthreads; int drain; int acq; int pub; };

/* rowtopk: RC is how many candidates a chunk keeps (>= R), NCH the requested chunk width (a
 * REQUEST -- libr4d rounds it up to a multiple of eight columns, so the scratch hook asks
 * r4d_rowtopk_bf16_chunks() what the launch will actually use), PT1 the stage-1 thread count. */
struct R4dTopkCfg { int rc; int nch; int pt1; };

/* The block-scaled fp8 matvec's rows a wave. Blocking trades waves for memory-level parallelism, so
 * where it lands depends on whether the kernel is waiting on loads or on the ALU -- a property of
 * the shape AND of the inner loop. THE RULE IS FITTED, so it holds only for the loop it was fitted
 * against: any change to the matvec's inner loop (the width of the scale load, the activation
 * dtype) invalidates it and it has to be re-fitted. 0 selects libr4d's default rule, so an untuned
 * install is fast rather than broken; the rest pin it for rad-tune. */
struct R4dMvCfg { int rb; };



extern "C" {
/* -- the tuned configuration (r4d_rows.cpp) --
 *
 * The core resolved one instantiation, picked a point in the space the row declared, and appended
 * the choices to RadArgs::p. These read that point back, falling to the axis's own default when a
 * cold caller supplied none -- so the default is defined once, in the RadTunable table, and a
 * checker calling the kernel with no tuned key gets exactly what an untuned install runs.
 *
 * They cannot fail: every axis has a default and the loader refuses a plugin whose default is not
 * one of that axis's own values, so there is no "out of range" to report. */
void r4d_cfg_gemm   (const RadArgs*, struct R4dGemmCfg*);
void r4d_cfg_tiled  (const RadArgs*, struct R4dTileCfg*);
void r4d_cfg_prefill(const RadArgs*, struct R4dTileCfg*);
void r4d_cfg_split  (const RadArgs*, struct R4dSplitCfg*);
void r4d_cfg_ar     (const RadArgs*, struct R4dArCfg*);
void r4d_cfg_topk   (const RadArgs*, struct R4dTopkCfg*);
void r4d_cfg_mv     (const RadArgs*, struct R4dMvCfg*);

}  /* extern "C" */

#endif  /* R4D_ARGS_H */
