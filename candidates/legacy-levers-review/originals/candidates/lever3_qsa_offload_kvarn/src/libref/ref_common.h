/* ref_common.h -- libref's numeric bedrock: dtype conversion, strided element access, and
 * the parameter readers every op shares.
 *
 * ============================ THE NUMERICS ARE THE REFERENCE DEFINITION ============================
 *
 * rad-kbench compares a resolved kernel against this plugin to a tolerance, so what this plugin
 * computes is not "one correct answer" -- it is THE answer, and every rule below is part of the
 * contract rather than an implementation detail:
 *
 *   * EVERY accumulation is f32. Not double, because a device kernel accumulates in f32 and an
 *     oracle that is more accurate than the thing it certifies turns a tolerance into a guess
 *     about which of the two is drifting. Not the operand dtype either: a bf16 accumulator would
 *     make the reference worse than every kernel it judges.
 *   * IEEE round-to-nearest-even everywhere, including the hand-written f32 -> bf16 / f16 / fp8
 *     rounders below. There is no bf16 convert instruction anywhere on this project's target, so
 *     a software RTNE is the right reference and not a workaround.
 *   * NO FUSED MULTIPLY-ADD AND NO REASSOCIATION. `a*b + c` must round twice. A reference that
 *     contracts is a reference that disagrees with itself between two compilers, and one that
 *     reassociates cannot be trusted to a tolerance at all. The pragma below asks the compiler for
 *     this and libref/CMakeLists.txt passes -ffp-contract=off, because the pragma is advisory
 *     in C++ and the flag is not.
 *   * SUMMATION ORDER IS FIXED AND INDEPENDENT OF THREAD COUNT. OpenMP parallelism in this plugin
 *     is only ever over whole output elements -- one thread owns a row, a column block, a head --
 *     and there is not one `reduction()` clause in the plugin. Two runs at different thread counts
 *     are bit-identical, which is what makes a diff against a device kernel mean something.
 *
 * The cost of all of this is stated plainly: this plugin is slow. It is a nest of scalar loops
 * with a per-element dtype switch outside the few paths that matter, and it is meant to be. Its
 * jobs are to make an unsupported model run at reduced speed instead of not at all, and to be the
 * thing every fast kernel is checked against (spec §17).
 */
#ifndef RAD_REF_COMMON_H
#define RAD_REF_COMMON_H

/* THE CONTRACTION RULE IS A BUILD FLAG, NOT A PRAGMA. `#pragma STDC FP_CONTRACT` is C, and g++
 * ignores it with a warning on every translation unit that includes this header; clang's
 * equivalent is spelled differently and is scoped. libref/CMakeLists.txt therefore passes
 * -ffp-contract=off on the target, which every compiler this project builds with honours, and the
 * top-level build already sets it for device code (docs/OPS.md). If you compile these sources by
 * hand, pass it. */

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace ref {

/* ================================================================== narrowing conversions
 *
 * ONE DEFINITION, IN rad_plugin.h, and these are the names ref's kernels call it by.
 *
 * These conversions ARE the reference definition, which is why they belong in the header every
 * plugin gets rather than in this file. This plugin's contract is f32 accumulation, IEEE
 * round-to-nearest-ties-to-even, no FMA and no reassociation, and a plugin whose bf16 narrowing
 * truncates instead of rounding to even cannot reproduce it and therefore cannot be checked
 * against it. A convention every implementer has to rediscover is a convention that drifts. */
static inline float    bf16_to_f32(uint16_t h)   { return rad_bf16_to_f32(h); }
static inline uint16_t f32_to_bf16(float f)      { return rad_f32_to_bf16(f); }
static inline float    f16_to_f32(uint16_t h)    { return rad_f16_to_f32(h); }
static inline uint16_t f32_to_f16(float f)       { return rad_f32_to_f16(f); }
static inline float    fp8e4m3_to_f32(uint8_t b) { return rad_fp8e4m3_to_f32(b); }
static inline uint8_t  f32_to_fp8e4m3(float f)   { return rad_f32_to_fp8e4m3(f); }
static inline float    fp8e5m2_to_f32(uint8_t b) { return rad_fp8e5m2_to_f32(b); }
static inline uint8_t  f32_to_fp8e5m2(float f)   { return rad_f32_to_fp8e5m2(f); }

/* OCP MXFP4 element (e2m1): sign in bit 3, exponent in 2:1, mantissa in bit 0. Eight magnitudes,
 * and no infinity or NaN -- the format has no room for either. */
static inline float mxfp4_elem_to_f32(uint8_t nib) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float v = kMag[nib & 0x7u];
    return (nib & 0x8u) ? -v : v;
}

/* OCP e8m0 block scale: a raw exponent byte, value 2^(e-127). 255 is NaN by definition. */
static inline float e8m0_to_f32(uint8_t e) {
    if (e == 0xff) return std::nanf("");
    return std::ldexp(1.0f, (int)e - 127);
}

/* ================================================================== packed weight codes
 * The packing conventions are libr4d's, because a .rad file is specific to a kernel set (spec
 * §4.2) and ref has to read the same bytes the fast kernel does:
 *
 *   RAD_I4  two nibbles per byte, element 2i in the LOW nibble; symmetric two's-complement code,
 *           value = (n ^ 8) - 8, so 0..15 maps to -8..7. This is what radiance_w4.pack writes and
 *           what r4d_dequant_w4_bf16's `twos=1` path reads.
 *   RAD_I2  four codes per byte, element 4i in the LOW two bits; value = (c ^ 2) - 2, the same
 *           construction one radix down, so 0..3 maps to -2..1.
 *   RAD_FP4E2M1 two e2m1 nibbles per byte, element 2i low, with one e8m0 scale per 32 elements
 *           carried in a separate stream (the `b_scale` operand).
 *
 * All three return the RAW CODE's value. The group scale is applied by the caller, because only
 * the caller knows where the scale stream is. */
static inline float w4_code(const void* p, int64_t i) {
    uint8_t b = ((const uint8_t*)p)[i >> 1];
    uint8_t n = (i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf);
    return (float)((int)(n ^ 0x8u) - 8);
}
static inline float w2_code(const void* p, int64_t i) {
    uint8_t b = ((const uint8_t*)p)[i >> 2];
    uint8_t c = (uint8_t)((b >> (2 * (i & 3))) & 0x3u);
    return (float)((int)(c ^ 0x2u) - 2);
}
static inline float mxfp4_code(const void* p, int64_t i) {
    uint8_t b = ((const uint8_t*)p)[i >> 1];
    return mxfp4_elem_to_f32((i & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xf));
}

/* ================================================================== element access
 * One switch per element, which is the price of being generic. The GEMM family lifts it out with
 * the templated loaders below; nothing else runs often enough to care. */
static inline float ld_dt(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_F32:    return ((const float*)p)[i];
        case RAD_F16:    return f16_to_f32(((const uint16_t*)p)[i]);
        case RAD_BF16:   return bf16_to_f32(((const uint16_t*)p)[i]);
        case RAD_F8E4M3: return fp8e4m3_to_f32(((const uint8_t*)p)[i]);
        case RAD_F8E5M2: return fp8e5m2_to_f32(((const uint8_t*)p)[i]);
        case RAD_I8:     return (float)((const int8_t*)p)[i];
        case RAD_U8:     return (float)((const uint8_t*)p)[i];
        case RAD_BOOL:   return ((const uint8_t*)p)[i] ? 1.0f : 0.0f;
        case RAD_I16:    return (float)((const int16_t*)p)[i];
        case RAD_I32:    return (float)((const int32_t*)p)[i];
        case RAD_U32:    return (float)((const uint32_t*)p)[i];
        case RAD_I64:    return (float)((const int64_t*)p)[i];
        case RAD_I4:     return w4_code(p, i);
        case RAD_I2:     return w2_code(p, i);
        case RAD_FP4E2M1:  return mxfp4_code(p, i);
        default:         return 0.0f;
    }
}

static inline void st_dt(uint32_t dt, void* p, int64_t i, float v) {
    switch (dt) {
        case RAD_F32:    ((float*)p)[i] = v; break;
        case RAD_F16:    ((uint16_t*)p)[i] = f32_to_f16(v); break;
        case RAD_BF16:   ((uint16_t*)p)[i] = f32_to_bf16(v); break;
        case RAD_F8E4M3: ((uint8_t*)p)[i] = f32_to_fp8e4m3(v); break;
        case RAD_F8E5M2: ((uint8_t*)p)[i] = f32_to_fp8e5m2(v); break;
        /* Integer stores round-to-nearest and saturate. A reference that wrapped would turn one
         * overflowing activation into a plausible small number, which is the failure this plugin
         * exists to catch rather than to commit. */
        case RAD_I8:  { float r = std::nearbyintf(v); r = r < -128.f ? -128.f : (r > 127.f ? 127.f : r);
                        ((int8_t*)p)[i] = (int8_t)r; break; }
        case RAD_U8:  { float r = std::nearbyintf(v); r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
                        ((uint8_t*)p)[i] = (uint8_t)r; break; }
        case RAD_BOOL:   ((uint8_t*)p)[i] = v != 0.0f ? 1u : 0u; break;
        case RAD_I16: { float r = std::nearbyintf(v); r = r < -32768.f ? -32768.f : (r > 32767.f ? 32767.f : r);
                        ((int16_t*)p)[i] = (int16_t)r; break; }
        case RAD_I32:    ((int32_t*)p)[i] = (int32_t)std::nearbyintf(v); break;
        case RAD_U32:    ((uint32_t*)p)[i] = (uint32_t)std::nearbyintf(v); break;
        case RAD_I64:    ((int64_t*)p)[i] = (int64_t)std::nearbyintf(v); break;
        case RAD_I4: {
            float r = std::nearbyintf(v);
            r = r < -8.f ? -8.f : (r > 7.f ? 7.f : r);
            const uint8_t nib = (uint8_t)((int8_t)r & 0x0F);
            uint8_t* bp = (uint8_t*)p + (i / 2);
            if ((i & 1) == 0) {
                *bp = (uint8_t)((*bp & 0xF0) | nib);
            } else {
                *bp = (uint8_t)((*bp & 0x0F) | (nib << 4));
            }
            break;
        }
        default:         break;
    }
}

/* Integer reads that must NOT go through f32: a token id, a slot index or a block id above 2^24
 * would come back rounded. Index tensors are read with this and never with ld_dt. */
static inline int64_t ld_int(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32:  return (int64_t)((const int32_t*)p)[i];
        case RAD_U32:  return (int64_t)((const uint32_t*)p)[i];
        case RAD_I64:  return ((const int64_t*)p)[i];
        case RAD_I16:  return (int64_t)((const int16_t*)p)[i];
        case RAD_I8:   return (int64_t)((const int8_t*)p)[i];
        case RAD_U8:   return (int64_t)((const uint8_t*)p)[i];
        case RAD_BOOL: return ((const uint8_t*)p)[i] ? 1 : 0;
        case RAD_F32:  return (int64_t)((const float*)p)[i];
        default:       return 0;
    }
}

static inline void st_int(uint32_t dt, void* p, int64_t i, int64_t v) {
    switch (dt) {
        case RAD_I32: ((int32_t*)p)[i] = (int32_t)v; break;
        case RAD_U32: ((uint32_t*)p)[i] = (uint32_t)v; break;
        case RAD_I64: ((int64_t*)p)[i] = v; break;
        case RAD_I16: ((int16_t*)p)[i] = (int16_t)v; break;
        case RAD_I8:  ((int8_t*)p)[i]  = (int8_t)v; break;
        case RAD_U8:  ((uint8_t*)p)[i] = (uint8_t)v; break;
        default:      st_dt(dt, p, i, (float)v); break;
    }
}

/* ================================================================== templated loaders
 * The GEMM family reads one element per FLOP, so the switch above would be most of its cost. These
 * let a caller hoist the decision to the top of the op and pay it once. */
template <uint32_t DT> struct Cvt;
template <> struct Cvt<RAD_F32>    { static inline float ld(const void* p, int64_t i) { return ((const float*)p)[i]; } };
template <> struct Cvt<RAD_F16>    { static inline float ld(const void* p, int64_t i) { return f16_to_f32(((const uint16_t*)p)[i]); } };
template <> struct Cvt<RAD_BF16>   { static inline float ld(const void* p, int64_t i) { return bf16_to_f32(((const uint16_t*)p)[i]); } };
template <> struct Cvt<RAD_F8E4M3> { static inline float ld(const void* p, int64_t i) { return fp8e4m3_to_f32(((const uint8_t*)p)[i]); } };
template <> struct Cvt<RAD_I8>     { static inline float ld(const void* p, int64_t i) { return (float)((const int8_t*)p)[i]; } };
template <> struct Cvt<RAD_I32>    { static inline float ld(const void* p, int64_t i) { return (float)((const int32_t*)p)[i]; } };
template <> struct Cvt<RAD_I4>     { static inline float ld(const void* p, int64_t i) { return w4_code(p, i); } };
template <> struct Cvt<RAD_I2>     { static inline float ld(const void* p, int64_t i) { return w2_code(p, i); } };
template <> struct Cvt<RAD_FP4E2M1>  { static inline float ld(const void* p, int64_t i) { return mxfp4_code(p, i); } };

/* ------------------------------------------------------------------ the ranged key
 *
 * A RANGED PARAMETER NAMES THE BAND AND THE OPERAND NAMES THE CALL, and every kernel below that
 * wants a row count asks the operand.
 *
 * `M`, `numel` and `q_len` are declared RAD_RANGE by the architectures, and the core freezes the
 * band's UPPER BOUND into the geometry a kernel reads -- max_tok, max_seqs, the whole residual
 * stream. It is what the selector matched on and what a kernel sizes a persistent buffer from;
 * it is not this issue's extent. A decode step presents a few rows of a max_tok buffer, so a
 * reference that reads the parameter as the length either REFUSES the call outright or, where the
 * buffers are large enough to survive it, walks rows nobody wrote and reports a mismatch against
 * a kernel that is right.
 *
 * libr4d reads the operand for the same reason -- the all-reduce shim takes r4d_numel(x) and uses
 * its `numel` parameter only to size the peer scratch at init -- and the engine binds every
 * operand with its real shape at issue, which is what makes the operand the honest source.
 *
 * The parameter is kept as a CEILING where one is stated: a call above the band was resolved
 * against constraints it does not satisfy, and that is a caller bug worth refusing. */
static inline int64_t band_rows(int64_t from_operand, int64_t band) {
    if (from_operand <= 0) return 0;
    return (band > 0 && from_operand > band) ? 0 : from_operand;
}

/* ================================================================== tensors */
static inline int64_t numel(const RadTensor* t) {
    if (!t || !t->data || t->rank == 0) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

static inline bool contig(const RadTensor* t) { return rad_tensor_is_contiguous(t) != 0; }

/* Element offset of the `lin`-th element in row-major logical order. Contiguous is the norm and
 * costs nothing; a strided view -- a slice of a wider projection, which the GDN block hands its
 * conv in every real model -- costs a decomposition, and is supported rather than refused because
 * "no geometry constraints" has to mean strides too. */
static inline int64_t offlin(const RadTensor* t, int64_t lin) {
    if (t->rank == 0) return lin;
    int64_t acc = 1;
    bool c = true;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) { c = false; break; }
        acc *= t->shape[i];
    }
    if (c) return lin;
    int64_t off = 0;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        int64_t s = t->shape[i];
        if (s <= 0) return 0;
        off += (lin % s) * t->stride[i];
        lin /= s;
    }
    return off;
}

/* The element offset of logical row `r` when a tensor is read as [rows, cols]: rows is everything
 * but the last dimension and cols is the last. This is the shape almost every op here wants. */
static inline int64_t rowoff(const RadTensor* t, int64_t r, int64_t cols) {
    if (t->rank >= 2 && t->shape[t->rank - 1] == cols) {
        /* Walk the leading dimensions with their own strides, so a [T,H,V] tensor read as
         * [T*H, V] lands correctly even when H's stride is not V. */
        int64_t off = 0;
        for (int i = (int)t->rank - 2; i >= 0; --i) {
            int64_t s = t->shape[i];
            if (s <= 0) return 0;
            off += (r % s) * t->stride[i];
            r /= s;
        }
        return off;
    }
    return r * cols;
}

static inline int64_t laststride(const RadTensor* t) {
    return t->rank ? t->stride[t->rank - 1] : 1;
}

static inline float ldt(const RadTensor* t, int64_t off) { return ld_dt(t->dtype, t->data, off); }
static inline void  stt(const RadTensor* t, int64_t off, float v) { st_dt(t->dtype, t->data, off, v); }

/* ================================================================== parameters
 *
 * p_find FORWARDS to the ABI's own lookup, and the property that makes that safe is that
 * rad_args_geti/gets and the rest are `static inline` in the headers: this .so carries its own
 * copy and has no undefined symbol from rad_abi.h. That matters because a kernel plugin links only
 * the ABI (spec 18) -- a plugin referencing a symbol defined inside the engine would fail to
 * dlopen unless the host exported its whole static core dynamically, which is a build coupling
 * this side of the boundary must not have.
 *
 * The getters below stay ref's own, because they are deliberately LAXER than the ABI's: they
 * accept a RANGE and a string spelling of an int, since a tool or a test may call a kernel
 * directly with no declare in front of it to enforce the type. */
static inline const RadParam* p_find(const RadArgs* a, const char* k) {
    return a ? rad_param_find(a->p, a->n_p, k) : nullptr;
}

static inline long long p_int(const RadArgs* a, const char* k, long long dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_INT)   return p->ival;
    if (p->kind == RAD_P_RANGE) return p->ihi;      /* see p_range_note below */
    if (p->kind == RAD_P_STR && p->sval) return std::strtoll(p->sval, nullptr, 10);
    return dflt;
}

static inline const char* p_str(const RadArgs* a, const char* k, const char* dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) return p->sval;
    return dflt;
}

/* A real-valued parameter. These are RAD_P_F64 in the schema (docs/OPS.md), and the core refuses
 * any other spelling for a float key at declare -- so `dval` is what arrives. The other kinds are
 * still read here because a tool or a test may call a kernel directly, with no declare in front of
 * it to enforce the type, and failing such a caller over a spelling would tell it nothing useful. */
static inline float p_f32(const RadArgs* a, const char* k, float dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_F64)   return (float)p->dval;
    if (p->kind == RAD_P_STR && p->sval) return (float)std::strtod(p->sval, nullptr);
    if (p->kind == RAD_P_INT)   return (float)p->ival;
    if (p->kind == RAD_P_RANGE) return (float)p->ihi;
    return dflt;
}

/* An enumerated parameter that a caller may spell either way: act="silu" or act=0. Names win;
 * the integer form is libr4d's ordering. */
static inline int p_enum(const RadArgs* a, const char* k, const char* const* names, int n,
                         int dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) {
        for (int i = 0; i < n; ++i)
            if (names[i] && std::strcmp(names[i], p->sval) == 0) return i;
        return dflt;
    }
    if (p->kind == RAD_P_INT) return (int)p->ival;
    return dflt;
}

/* ================================================================== operand checks */
/* Errors are values (docs/IMPLEMENTATION.md). Nothing in this plugin throws, and nothing that did
 * not do the work returns RAD_OK. */
#define RAD_REF_TRY(expr) do { int _s = (expr); if (_s < 0) return _s; } while (0)

static inline const RadTensor* t_in(const RadArgs* a, int i) { return rad_arg_in(a, i); }
static inline bool have(const RadArgs* a, int n) { return rad_args_have(a, n) != 0; }

/* ================================================================== per-thread workspace
 * The gated-delta-net ops carry state that is too big for a stack frame -- a [128,128] fp32
 * recurrent state alone is 64 KB, and the chunked scan wants three more matrices beside it. The
 * fast kernel keeps all of that in WMMA accumulators; the reference cannot, so it needs somewhere
 * to put it.
 *
 * A per-thread vector that GROWS ONCE and is then reused. This is the one place in the plugin that
 * allocates, and it is worth naming why it is allowed to: the engine's no-allocation rule is about
 * the arena and the step path (docs/IMPLEMENTATION.md), and this allocates on the first launch of
 * a shape and never again -- there is no per-step malloc. The honest alternative is a scratch hook,
 * and it does not work here: scratch is sized at declare from the geometry, and the amount this
 * needs depends on the OpenMP thread count, which declare does not know.
 *
 * ONE CALL PER LAUNCH. A second call may reallocate and invalidate the first pointer, so every op
 * below asks for its whole workspace at once and carves it up itself. */
inline float* ws_get(size_t n_floats) {
    static thread_local std::vector<float> v;
    if (v.size() < n_floats) v.resize(n_floats);
    return v.data();
}

/* ================================================================== exact top-k
 * One step of an exact selection over a row, descending by value and, ON A TIE, by the LOWER
 * index. That tie rule is libr4d's (r4d_rowtopk_bf16: "descending, ties to the lower column") and
 * it has to be stated rather than left to a sort's stability, because a router that breaks ties
 * differently from the kernel it is checked against routes a token to a different expert and the
 * diff is a whole layer wide.
 *
 * `pv`/`pi` carry the previously selected (value, index); seed them with +infinity and -1. The
 * cost is O(N) per selected element, i.e. O(N*R) for the whole row -- the straightforward
 * definition, and the one that needs no scratch buffer to be exact. */
static inline bool topk_next(const RadTensor* x, int64_t base, int64_t st, int64_t N,
                             float* pv, int64_t* pi) {
    float best_v = 0.0f;
    int64_t best_i = -1;
    for (int64_t i = 0; i < N; ++i) {
        const float v = ld_dt(x->dtype, x->data, base + i * st);
        if (v > *pv) continue;                       /* strictly better than the last pick */
        if (v == *pv && i <= *pi) continue;          /* the last pick itself, or one before it */
        if (best_i < 0 || v > best_v) { best_v = v; best_i = i; }
    }
    if (best_i < 0) return false;
    *pv = best_v;
    *pi = best_i;
    return true;
}

/* ================================================================== small maths
 * Written out rather than taken from <cmath>'s fast paths, because the reference's exp is the one
 * every softmax and every gate is checked against. std::expf is correctly rounded to well under a
 * ULP on glibc, which is the accuracy an oracle wants; a device kernel's __expf is not, and the
 * gap is what rad-kbench's tolerance is for. */
static inline float ref_exp(float x)  { return std::exp(x); }
/* Used by locally-typical sampling, which is a threshold on |surprisal - entropy| and therefore
 * needs both. Same reasoning as ref_exp: the oracle takes the correctly-rounded libm. */
static inline float ref_log(float x)  { return std::log(x); }
static inline float ref_abs(float x)  { return std::fabs(x); }
static inline float act_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float act_silu(float x) { return x * act_sigmoid(x); }

/* The exact erf-based GELU, not tanh. A model trained with the tanh approximation should ask for
 * it by name rather than have the reference guess; until an op says so, "gelu" is the real one. */
static inline float act_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

/* softplus in the two-sided form, with the large-x cutoff FLA and libr4d both use: above the
 * threshold log1p(e^x) IS x to every bit f32 holds, and evaluating it costs an overflow. */
static inline float act_softplus(float x, float thr) {
    if (x > thr) return x;
    return x > 0.0f ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}

/* ================================================================== the Hadamard rotation
 * The widest rotation this reference performs in one stack frame. libr4d caps `had` at 512 -- a
 * power of two dividing the per-rank K, and 8704 admits no more -- so 4096 is four doublings of
 * headroom. A wider request returns RAD_E_UNSUPPORTED by name rather than silently rotating a
 * narrower block, because a half-applied rotation is numerically plausible and completely wrong. */
enum { REF_HAD_MAX = 4096 };

/* The unnormalised in-place transform over `g` values, g a power of two. */
static inline void fwht(float* v, int64_t g) {
    for (int64_t h = 1; h < g; h <<= 1)
        for (int64_t i = 0; i < g; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

static inline bool pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

}  /* namespace ref */
#endif /* RAD_REF_COMMON_H */
