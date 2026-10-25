/* ref_common.h -- libref's numeric core: dtype conversion, strided element access, and the
 * parameter readers every op shares.
 *
 * The numerics are the reference definition. rad-kbench compares each resolved kernel against
 * this plugin to a tolerance, so these rules are part of the contract:
 *
 *   * Every accumulation is f32. Not double: device kernels accumulate in f32, and an oracle more
 *     accurate than what it certifies makes a tolerance ambiguous. Not the operand dtype either: a
 *     bf16 accumulator would be worse than every kernel it judges. The exceptions sum in double
 *     and are part of their op's definition (docs/OPS.md):
 *       - seg_mean's segment sums, which OPS.md states;
 *       - gram_accum's Gram matrix, a calibration statistic read once at convert time;
 *       - logit_rerank's dot products, an exact rescoring of a coarse head's candidates;
 *       - sample_logprobs' normaliser, the sum of exp over the whole vocabulary;
 *       - givens_dense_quant's dense rotation and ple_gate's key-query dot;
 *       - the sums of squares of hc_read's, mtp_enter's, ple_gate's and qsa_block_key's norms. An
 *         f32 sum of n squares differs by at most about n * 2^-24 relative, which at these widths
 *         (n <= 10240) stays below one bf16 rounding of the normed value.
 *   * IEEE round-to-nearest-even everywhere, including the f32 -> bf16 / f16 / fp8 rounders. The
 *     target has no bf16 convert instruction, so software RTNE is the right reference.
 *   * No fused multiply-add and no reassociation: `a*b + c` rounds twice. A contracting reference
 *     disagrees with itself across compilers. libref/CMakeLists.txt passes -ffp-contract=off.
 *   * Summation order is fixed and independent of thread count. OpenMP parallelism is only over
 *     whole output elements (a row, a column block, a head), with no `reduction()` clause anywhere,
 *     so runs at different thread counts are bit-identical.
 *
 * This plugin is deliberately simple: scalar loops with a per-element dtype switch outside the few
 * hot paths (the GEMM tiles, ref_gemm.h, and the delta-net scans, ref_gdn.cpp, which run
 * independent output chains side by side without changing any chain). Its jobs are to run an
 * unsupported model at reduced speed instead of not at all, and to be what every fast kernel is
 * checked against (spec §17).
 *
 * An operand of a dtype these conversions cannot read or store is refused at launch
 * (ref_registry.cpp), never read as zeros.
 */
#ifndef RAD_REF_COMMON_H
#define RAD_REF_COMMON_H

/* The no-contraction rule is a build flag, not a pragma: `#pragma STDC FP_CONTRACT` is C and g++
 * ignores it with a warning, and clang's equivalent is spelled differently and scoped. The target
 * gets -ffp-contract=off from libref/CMakeLists.txt (the top-level build sets it for device code,
 * docs/OPS.md). Pass it when compiling these sources by hand. */

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
 * Local names for rad_plugin.h's conversions. They are part of the reference definition, so they
 * live in the header every plugin gets: a plugin whose bf16 narrowing truncates instead of rounding
 * to even cannot be checked against this one. */
static inline float    bf16_to_f32(uint16_t h)   { return rad_bf16_to_f32(h); }
static inline uint16_t f32_to_bf16(float f)      { return rad_f32_to_bf16(f); }
static inline float    f16_to_f32(uint16_t h)    { return rad_f16_to_f32(h); }
static inline uint16_t f32_to_f16(float f)       { return rad_f32_to_f16(f); }
static inline float    fp8e4m3_to_f32(uint8_t b) { return rad_fp8e4m3_to_f32(b); }
static inline uint8_t  f32_to_fp8e4m3(float f)   { return rad_f32_to_fp8e4m3(f); }
static inline float    fp8e5m2_to_f32(uint8_t b) { return rad_fp8e5m2_to_f32(b); }
static inline uint8_t  f32_to_fp8e5m2(float f)   { return rad_f32_to_fp8e5m2(f); }

/* E4M3 decoded through a table of all 256 codes, filled by fp8e4m3_to_f32 itself: the same
 * floats without the decode's branches, which otherwise dominate an fp8 GEMM or KV-cache read. */
struct Fp8Table {
    float v[256];
    Fp8Table() { for (int b = 0; b < 256; ++b) v[b] = fp8e4m3_to_f32((uint8_t)b); }
};
inline const Fp8Table kFp8Table;
static inline float fp8e4m3_lut(uint8_t b) { return kFp8Table.v[b]; }

/* OCP MXFP4 element (e2m1): sign in bit 3, exponent in 2:1, mantissa in bit 0. Eight magnitudes;
 * no infinity or NaN. */
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
 * The canonical planes a container stores (abi/rad_encoding.h): sub-byte codes are a bit stream
 * along the row, lowest bits first, and integer codes are two's complement (abi/rad_types.h).
 * libref rearranges no plane (its relayouts only join whole planes, ref_planes.h), so it reads
 * these bytes as stored; the op oracle compares a device kernel that relayouts a weight at load
 * after RadUnrelayoutFn restores this form.
 *
 *   RAD_I4  two nibbles per byte, element 2i in the low nibble; two's-complement code,
 *           value = (n ^ 8) - 8, so 0..15 maps to -8..7.
 *   RAD_I2  four codes per byte, element 4i in the low two bits; value = (c ^ 2) - 2, so 0..3 maps
 *           to -2..1.
 *   RAD_FP4E2M1 two e2m1 nibbles per byte, element 2i low, with one e8m0 scale per 32 elements in a
 *           separate stream (the `b_scale` operand).
 *
 * All three return the raw code's value; the caller applies the group scale, since only it knows
 * where the scale stream is. */
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
 * One switch per element, the price of being generic. The GEMM family hoists it with the templated
 * loaders below; nothing else runs often enough to matter. */
static inline float ld_dt(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_F32:    return ((const float*)p)[i];
        case RAD_F16:    return f16_to_f32(((const uint16_t*)p)[i]);
        case RAD_BF16:   return bf16_to_f32(((const uint16_t*)p)[i]);
        case RAD_F8E4M3: return fp8e4m3_lut(((const uint8_t*)p)[i]);
        case RAD_F8E5M2: return fp8e5m2_to_f32(((const uint8_t*)p)[i]);
        case RAD_I8:     return (float)((const int8_t*)p)[i];
        case RAD_U8:     return (float)((const uint8_t*)p)[i];
        case RAD_BOOL:   return ((const uint8_t*)p)[i] ? 1.0f : 0.0f;
        case RAD_I16:    return (float)((const int16_t*)p)[i];
        case RAD_U16:    return (float)((const uint16_t*)p)[i];
        case RAD_I32:    return (float)((const int32_t*)p)[i];
        case RAD_U32:    return (float)((const uint32_t*)p)[i];
        case RAD_I64:    return (float)((const int64_t*)p)[i];
        case RAD_I4:     return w4_code(p, i);
        case RAD_I2:     return w2_code(p, i);
        case RAD_FP4E2M1:  return mxfp4_code(p, i);
        case RAD_E8M0:   return e8m0_to_f32(((const uint8_t*)p)[i]);
        /* Unreachable through a launch: the registry refuses an operand of any other dtype
         * (ref_registry.cpp, ref_check_dtypes). */
        default:         return 0.0f;
    }
}

/* A packed code's bits `c` (already masked to `bits`) into element `i` of a bit stream, lowest
 * bits first. The two atomics touch only this element's bits, so rows written by different threads
 * may share a byte. */
static inline void st_packed(void* p, int64_t i, int bits, unsigned c) {
    const int per = 8 / bits;
    uint8_t* b = (uint8_t*)p + i / per;
    const int sh = (int)(i % per) * bits;
    const uint8_t mask = (uint8_t)(((1u << bits) - 1u) << sh);
    __atomic_fetch_and(b, (uint8_t)~mask, __ATOMIC_RELAXED);
    __atomic_fetch_or(b, (uint8_t)((c << sh) & mask), __ATOMIC_RELAXED);
}

/* The e2m1 code nearest `v` (ties to the even code), saturating at 6; NaN is +0. */
static inline unsigned e2m1_code(float v) {
    static const float kMag[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    if (v != v) return 0u;
    const unsigned sign = std::signbit(v) ? 8u : 0u;
    const float a = std::fabs(v);
    unsigned best = 0;
    for (unsigned c = 1; c < 8; ++c) {
        const float d = std::fabs(kMag[c] - a), db = std::fabs(kMag[best] - a);
        if (d < db || (d == db && (c & 1u) == 0u)) best = c;
    }
    return sign | best;
}

/* `v` rounded to nearest even and clamped to [lo, hi]; NaN is 0. */
static inline float sat_round(float v, float lo, float hi) {
    if (v != v) return 0.0f;
    const float r = std::nearbyintf(v);
    return r < lo ? lo : (r > hi ? hi : r);
}

static inline void st_dt(uint32_t dt, void* p, int64_t i, float v) {
    switch (dt) {
        case RAD_F32:    ((float*)p)[i] = v; break;
        case RAD_F16:    ((uint16_t*)p)[i] = f32_to_f16(v); break;
        case RAD_BF16:   ((uint16_t*)p)[i] = f32_to_bf16(v); break;
        case RAD_F8E4M3: ((uint8_t*)p)[i] = f32_to_fp8e4m3(v); break;
        case RAD_F8E5M2: ((uint8_t*)p)[i] = f32_to_fp8e5m2(v); break;
        /* Integer stores round to nearest even, saturate to the type's range and store NaN as 0.
         * Wrapping would turn an overflowing activation into a plausible small number, the kind
         * of failure this plugin must catch. */
        case RAD_I8:     ((int8_t*)p)[i]   = (int8_t)sat_round(v, -128.0f, 127.0f); break;
        case RAD_U8:     ((uint8_t*)p)[i]  = (uint8_t)sat_round(v, 0.0f, 255.0f); break;
        case RAD_BOOL:   ((uint8_t*)p)[i] = v != 0.0f ? 1u : 0u; break;
        case RAD_I16:    ((int16_t*)p)[i]  = (int16_t)sat_round(v, -32768.0f, 32767.0f); break;
        case RAD_U16:    ((uint16_t*)p)[i] = (uint16_t)sat_round(v, 0.0f, 65535.0f); break;
        /* Bounds that are powers of two, exact in f32: at and past 2^31 (2^32, 2^63) the store
         * saturates to the largest value, which f32 cannot hold. */
        case RAD_I32: {
            const float r = sat_round(v, -0x1p31f, 0x1p31f);
            ((int32_t*)p)[i] = r >= 0x1p31f ? INT32_MAX : (int32_t)r;
            break;
        }
        case RAD_U32: {
            const float r = sat_round(v, 0.0f, 0x1p32f);
            ((uint32_t*)p)[i] = r >= 0x1p32f ? UINT32_MAX : (uint32_t)r;
            break;
        }
        case RAD_I64: {
            const float r = sat_round(v, -0x1p63f, 0x1p63f);
            ((int64_t*)p)[i] = r >= 0x1p63f ? INT64_MAX : (int64_t)r;
            break;
        }
        /* Packed codes: the nearest code, saturated, so a code read by ld_dt stores back as
         * itself (a gather or scatter of packed rows moves them exactly). */
        case RAD_I4:     st_packed(p, i, 4, (unsigned)(int)sat_round(v, -8.0f, 7.0f) & 0xFu); break;
        case RAD_I2:     st_packed(p, i, 2, (unsigned)(int)sat_round(v, -2.0f, 1.0f) & 0x3u); break;
        case RAD_FP4E2M1: st_packed(p, i, 4, e2m1_code(v)); break;
        default:         break;   /* refused at launch, as in ld_dt */
    }
}

/* Integer reads that must not go through f32, where an id above 2^24 (token, slot, block) would
 * be rounded. Index tensors are read with this, never with ld_dt. */
static inline int64_t ld_int(uint32_t dt, const void* p, int64_t i) {
    switch (dt) {
        case RAD_I32:  return (int64_t)((const int32_t*)p)[i];
        case RAD_U32:  return (int64_t)((const uint32_t*)p)[i];
        case RAD_U16:  return (int64_t)((const uint16_t*)p)[i];   /* attn_paged_tree's masks */
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
        case RAD_U16: ((uint16_t*)p)[i] = (uint16_t)v; break;
        case RAD_I64: ((int64_t*)p)[i] = v; break;
        case RAD_I16: ((int16_t*)p)[i] = (int16_t)v; break;
        case RAD_I8:  ((int8_t*)p)[i]  = (int8_t)v; break;
        case RAD_U8:  ((uint8_t*)p)[i] = (uint8_t)v; break;
        default:      st_dt(dt, p, i, (float)v); break;
    }
}

/* ================================================================== templated loaders
 * The GEMM family reads one element per FLOP, so the switch above would dominate its cost. These
 * let a caller resolve the dtype once at the top of the op. */
template <uint32_t DT> struct Cvt;
template <> struct Cvt<RAD_F32>    { static inline float ld(const void* p, int64_t i) { return ((const float*)p)[i]; } };
template <> struct Cvt<RAD_F16>    { static inline float ld(const void* p, int64_t i) { return f16_to_f32(((const uint16_t*)p)[i]); } };
template <> struct Cvt<RAD_BF16>   { static inline float ld(const void* p, int64_t i) { return bf16_to_f32(((const uint16_t*)p)[i]); } };
template <> struct Cvt<RAD_F8E4M3> { static inline float ld(const void* p, int64_t i) { return fp8e4m3_lut(((const uint8_t*)p)[i]); } };
template <> struct Cvt<RAD_I8>     { static inline float ld(const void* p, int64_t i) { return (float)((const int8_t*)p)[i]; } };
template <> struct Cvt<RAD_I32>    { static inline float ld(const void* p, int64_t i) { return (float)((const int32_t*)p)[i]; } };
template <> struct Cvt<RAD_I4>     { static inline float ld(const void* p, int64_t i) { return w4_code(p, i); } };
template <> struct Cvt<RAD_I2>     { static inline float ld(const void* p, int64_t i) { return w2_code(p, i); } };
template <> struct Cvt<RAD_FP4E2M1>  { static inline float ld(const void* p, int64_t i) { return mxfp4_code(p, i); } };

/* ------------------------------------------------------------------ the ranged key
 *
 * A ranged parameter names the band; the operand names the call. Every kernel that wants a row
 * count reads it from the operand.
 *
 * `M`, `numel` and `q_len` are declared RAD_RANGE, and the core freezes the band's upper bound
 * (max_tok, max_seqs, the whole residual stream) into the geometry. That is what the selector
 * matched and what persistent buffers are sized from, not this issue's extent: a decode step
 * presents a few rows of a max_tok buffer. Reading the parameter as the length would refuse the
 * call or walk unwritten rows and report a false mismatch. The engine binds every operand with its
 * real shape at issue, and libr4d reads the operand too (its all-reduce uses `numel` only to size
 * peer scratch at init).
 *
 * The parameter stays a ceiling where one is stated: a call above the band was resolved against
 * constraints it does not satisfy, a caller bug worth refusing. */
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

/* Element offset of the `lin`-th element in row-major logical order. Contiguous costs nothing; a
 * strided view (a slice of a wider projection, as the GDN block passes its conv) costs a
 * decomposition. Strides are supported because the reference has no geometry constraints. */
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

/* Element offset of logical row `r` when a tensor is read as [rows, cols], cols being the last
 * dimension and rows everything before it. Most ops here use this view. */
static inline int64_t rowoff(const RadTensor* t, int64_t r, int64_t cols) {
    if (t->rank >= 2 && t->shape[t->rank - 1] == cols) {
        /* Walk the leading dimensions with their own strides, so a [T,H,V] tensor read as
         * [T*H, V] is correct even when H's stride is not V. */
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

/* `v` rounded through a store and load of dtype `dt`, as an unfused pair rounds the tensor between
 * them. Whole-byte dtypes only. */
static inline float round_dt(uint32_t dt, float v) {
    uint64_t buf = 0;
    st_dt(dt, &buf, 0, v);
    return ld_dt(dt, &buf, 0);
}

/* Rows of a tensor read as [rows, n]: `n` is a parameter, the row count comes from the operand
 * (ref_ops.h). */
static inline int64_t rows_of(const RadTensor* t, int64_t n) { return n > 0 ? numel(t) / n : 0; }

/* Entry `i` of a cu_seqlens operand. */
static inline int64_t cu_at(const RadTensor* cu, int64_t i) {
    return ld_int(cu->dtype, cu->data, offlin(cu, i));
}

/* The sequence that owns dense row `r`; a linear scan, since there are few. */
static inline int64_t seq_of(const RadTensor* cu, int64_t n_seq, int64_t r) {
    int64_t s = 0;
    while (s + 1 < n_seq && cu_at(cu, s + 1) <= r) ++s;
    return s;
}

/* ================================================================== parameters
 *
 * p_find forwards to the ABI's lookup. That is safe because the rad_args_* helpers are `static
 * inline` in the headers, so this .so has no undefined symbol from rad_abi.h. A kernel plugin links
 * only the ABI (spec 18); referencing an engine symbol would break dlopen unless the host exported
 * its static core.
 *
 * The getters are ref's own and deliberately laxer than the ABI's: they accept a RANGE and a string
 * spelling of an int, since a tool or test may call a kernel directly with no declare to enforce
 * the type. */
static inline const RadParam* p_find(const RadArgs* a, const char* k) {
    return a ? rad_param_find(a->p, a->n_p, k) : nullptr;
}

static inline long long p_int(const RadArgs* a, const char* k, long long dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_INT)   return p->ival;
    if (p->kind == RAD_P_RANGE) return p->ihi;      /* the band's ceiling; see band_rows */
    if (p->kind == RAD_P_STR && p->sval) return std::strtoll(p->sval, nullptr, 10);
    return dflt;
}

static inline const char* p_str(const RadArgs* a, const char* k, const char* dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) return p->sval;
    return dflt;
}

/* A real-valued parameter. The schema types these RAD_P_F64 (docs/OPS.md) and declare refuses any
 * other spelling, so `dval` is what normally arrives. Other kinds are still accepted for direct
 * callers (tools, tests) with no declare in front. */
static inline float p_f32(const RadArgs* a, const char* k, float dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_F64)   return (float)p->dval;
    if (p->kind == RAD_P_STR && p->sval) return (float)std::strtod(p->sval, nullptr);
    if (p->kind == RAD_P_INT)   return (float)p->ival;
    if (p->kind == RAD_P_RANGE) return (float)p->ihi;
    return dflt;
}

/* An enumerated parameter spelled either way, act="silu" or act=0. Names win; the integer form
 * follows libr4d's ordering. An unknown name is -1, for the caller to refuse. */
static inline int p_enum(const RadArgs* a, const char* k, const char* const* names, int n,
                         int dflt) {
    const RadParam* p = p_find(a, k);
    if (!p) return dflt;
    if (p->kind == RAD_P_STR && p->sval) {
        for (int i = 0; i < n; ++i)
            if (names[i] && std::strcmp(names[i], p->sval) == 0) return i;
        return -1;
    }
    if (p->kind == RAD_P_INT) return (int)p->ival;
    return dflt;
}

/* ================================================================== operand checks */
/* Errors are values (docs/IMPLEMENTATION.md): nothing here throws, and nothing returns RAD_OK
 * without doing the work. */
#define RAD_REF_TRY(expr) do { int _s = (expr); if (_s < 0) return _s; } while (0)

/* quant_act_i8g's rule for one row of `n` (ref_gen.cpp): returns the scale (amax/127, 1 for an
 * all-zero row) and writes the codes rint(clamp(x * (1/scale), +-127)) to `code`, which may be `x`. */
float ref_q8_row(const float* x, int64_t n, float* code);

static inline const RadTensor* t_in(const RadArgs* a, int i) { return rad_arg_in(a, i); }
static inline bool have(const RadArgs* a, int n) { return rad_args_have(a, n) != 0; }

/* ================================================================== parallel loops
 * OpenMP over whole outputs, with an if() clause: a loop with fewer than REF_PAR_MIN units of
 * work (a unit is about one multiply-add or element access) runs on the calling thread, so a
 * small op does not wake a thread team. Which thread computes an output never changes its value.
 *
 *   REF_PAR_FOR(work)      a parallel for, static schedule (REF_PAR_FOR_DYN: dynamic)
 *   REF_PAR_REGION(work)   a parallel region, for a loop that wants per-thread setup (ws_get)
 *   REF_FOR / REF_FOR_DYN  the for inside such a region, static or dynamic schedule */
#define REF_PRAGMA(x) _Pragma(#x)
#define REF_PAR_MIN ((int64_t)1 << 16)
#define REF_PAR_FOR(work) \
    REF_PRAGMA(omp parallel for schedule(static) if((int64_t)(work) >= REF_PAR_MIN))
#define REF_PAR_FOR_DYN(work) \
    REF_PRAGMA(omp parallel for schedule(dynamic) if((int64_t)(work) >= REF_PAR_MIN))
#define REF_PAR_REGION(work) REF_PRAGMA(omp parallel if((int64_t)(work) >= REF_PAR_MIN))
#define REF_FOR     REF_PRAGMA(omp for schedule(static))
#define REF_FOR_DYN REF_PRAGMA(omp for schedule(dynamic))

/* ================================================================== per-thread workspace
 * The gated-delta-net ops carry state too big for a stack frame: a [128,128] fp32 recurrent state
 * is 64 KB, and the chunked scan needs three more matrices. The fast kernel keeps that in WMMA
 * accumulators; the reference needs memory.
 *
 * A per-thread vector that grows once and is then reused, so it allocates only on a shape's first
 * launch. A scratch hook cannot replace it: scratch is sized at declare, but this depends on the
 * OpenMP thread count, which declare does not know. (This plugin is not the step path the engine's
 * no-allocation rule covers, docs/IMPLEMENTATION.md; other ops allocate small per-row vectors.)
 *
 * One call per launch: a second call may reallocate and invalidate the first pointer, so each op
 * requests its whole workspace at once and carves it up. */
inline float* ws_get(size_t n_floats) {
    static thread_local std::vector<float> v;
    if (v.size() < n_floats) v.resize(n_floats);
    return v.data();
}

/* ================================================================== exact top-k
 * One step of an exact selection over a row: descending by value, ties to the lower index. That is
 * libr4d's rule (r4d_rowtopk_bf16) and it must be explicit, not left to sort stability: a router
 * that breaks ties differently sends a token to a different expert and the whole layer diverges.
 *
 * `pv`/`pi` carry the previous pick (value, index); seed them with +infinity and -1. O(N) per
 * selected element, O(N*R) per row, with no scratch buffer. */
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
 * Every transcendental is evaluated in double and rounded once to float. glibc's float routines
 * are not the same function from one release to the next (2.41 made log1pf, tanhf, erff and
 * others correctly rounded), so a float call answers differently on two hosts and an oracle held
 * to its recordings exactly would fail on the one that did not record them. The double routines
 * have been stable for many releases and are accurate to well under a ULP of double, so the
 * rounded float is the same everywhere but for a double-rounding tie. A device kernel's __expf is
 * further off; rad-kbench's tolerance covers that gap. */
static inline float ref_exp(float x)   { return (float)std::exp((double)x); }
static inline float ref_log(float x)   { return (float)std::log((double)x); }
static inline float ref_log1p(float x) { return (float)std::log1p((double)x); }
static inline float ref_erf(float x)   { return (float)std::erf((double)x); }
static inline float ref_tanh(float x)  { return (float)std::tanh((double)x); }
static inline float ref_sin(float x)   { return (float)std::sin((double)x); }
static inline float ref_cos(float x)   { return (float)std::cos((double)x); }
static inline float ref_pow(float b, float e) { return (float)std::pow((double)b, (double)e); }
static inline float ref_abs(float x)   { return std::fabs(x); }
static inline float act_sigmoid(float x) { return 1.0f / (1.0f + ref_exp(-x)); }
static inline float act_silu(float x) { return x * act_sigmoid(x); }

/* The exact erf-based GELU. "gelu" means this; the tanh approximation must be requested by name. */
static inline float act_gelu(float x) {
    return 0.5f * x * (1.0f + ref_erf(x * 0.70710678118654752f));
}

/* The tanh approximation (PyTorch's gelu_tanh, Gemma's gelu_pytorch_tanh), which gated
 * feed-forwards are trained with. */
static inline float act_gelu_tanh(float x) {
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + ref_tanh(k0 * (x + k1 * x * x * x)));
}

/* softplus in the two-sided form, with FLA's and libr4d's large-x cutoff: above the threshold
 * log1p(e^x) equals x in f32, and evaluating it would overflow. */
static inline float act_softplus(float x, float thr) {
    if (x > thr) return x;
    return x > 0.0f ? x + ref_log1p(ref_exp(-x)) : ref_log1p(ref_exp(x));
}

/* ================================================================== the Hadamard rotation
 * The widest rotation this reference performs in one stack frame. libr4d caps `had` at 512 (a
 * power of two dividing the per-rank K; 8704 admits no more), so 4096 leaves headroom. A wider
 * request returns RAD_E_UNSUPPORTED rather than rotating a narrower block, since a half-applied
 * rotation looks plausible and is wrong. */
enum { REF_HAD_MAX = 4096 };

/* The unnormalised in-place transform over `g` values, g a power of two. */
static inline void fwht(float* v, int64_t g) {
    for (int64_t h = 1; h < g; h <<= 1)
        for (int64_t i = 0; i < g; i += 2 * h)
            for (int64_t j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

static inline bool pow2(int64_t v) { return v > 0 && (v & (v - 1)) == 0; }

/* ================================================================== the 8-bit group
 * One group of the block-scaled 8-bit quantisers (constants explained at ref_quant.cpp's
 * quant_act_fp8): `v(i)` for i < n is the group's values and `top` the largest code, 448 for E4M3
 * and 127 for int8. The codes clamp(v(i) / scale, +-top) go to q[qb + i * qs] in q's dtype (int8
 * rounded to nearest even), and the scale amax / top (1 for an all-zero group) is returned. */
template <class V>
static inline float q8_group(const V& v, int64_t n, float top, const RadTensor* q, int64_t qb,
                             int64_t qs) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; ++i) { const float m = std::fabs(v(i)); if (m > amax) amax = m; }
    const float sc  = amax > 0.0f ? amax * (1.0f / top) : 1.0f;
    const float inv = 1.0f / sc;
    for (int64_t i = 0; i < n; ++i) {
        float c = v(i) * inv;
        if (c >  top) c =  top;
        if (c < -top) c = -top;
        st_dt(q->dtype, q->data, qb + i * qs, c);
    }
    return sc;
}

/* ================================================================== the int8 step order
 * quant_act_i8 stores a row's codes in the order libr4d's int8 WMMA reads an A fragment: within
 * each sixteen-byte step the middle two dwords are exchanged, so column c sits at byte
 * i8_step_pos(c) (bits 2 and 3 of c swapped). The map is its own inverse. gemm_nt_q reads its
 * activation in this order at the dtypes i8_steps_dtype names. */
static inline int64_t i8_step_pos(int64_t c) {
    return (c & ~(int64_t)12) | ((c & 4) << 1) | ((c & 8) >> 1);
}
static inline bool i8_steps_dtype(const char* dt) {
    static const char* const kDtypes[] = { "w4a8", "w4a8_asym", "w8a8", "w2a8" };
    for (const char* d : kDtypes)
        if (dt && std::strcmp(dt, d) == 0) return true;
    return false;
}

}  /* namespace ref */
#endif /* RAD_REF_COMMON_H */
