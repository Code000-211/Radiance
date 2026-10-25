/* rad_types.h -- the primitive vocabulary shared by the core, the plugins and the tools.
 *
 * Pure C. Everything here crosses an .so boundary, so nothing in this file may be a C++ type,
 * have a non-trivial layout, or depend on a compiler flag. See spec.md §2.
 */
#ifndef RAD_TYPES_H
#define RAD_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ status */
/* Zero is success. Negative is failure, and the value names its class so a caller can report it.
 * A kernel returning negative at issue is a selection bug, not a runtime condition (spec.md §17). */
enum {
    RAD_OK              =  0,
    RAD_E_SHAPE         = -1,   /* geometry this instantiation does not serve */
    RAD_E_DTYPE         = -2,
    RAD_E_ALIGN         = -3,   /* pointer or stride alignment the kernel requires */
    RAD_E_STRIDE        = -4,   /* non-contiguous where contiguity is required */
    RAD_E_SCRATCH       = -5,   /* scratch buffer absent or too small */
    RAD_E_NOMEM         = -6,
    RAD_E_NOKERNEL      = -7,   /* nothing in the hierarchy resolved */
    RAD_E_NOSCHEMA      = -8,   /* op name has no declared schema */
    RAD_E_SCHEMA        = -9,   /* params do not satisfy the op's schema */
    RAD_E_ABI           = -10,  /* plugin ABI version mismatch */
    RAD_E_DUPLICATE     = -11,  /* two plugins claim the same architecture id / op schema */
    RAD_E_INVAL         = -12,
    RAD_E_IO            = -13,
    RAD_E_FORMAT        = -14,  /* container is not a .rad, or is a version we do not read */
    RAD_E_DEVICE        = -15,  /* the device layer refused or faulted */
    RAD_E_STATE         = -16,  /* called in the wrong phase */
    RAD_E_NOTFOUND      = -17,
    RAD_E_UNSUPPORTED   = -18,
    RAD_E_FULL          = -19,  /* a pool, arena or table is exhausted */
    RAD_E_CANCELLED     = -20
};

/* ------------------------------------------------------------------ the helpers, inline
 *
 * Every free function in this header is a `static inline` definition, not a declaration. A plugin
 * links these headers and nothing of the engine's, so a function defined in the core would
 * resolve at dlopen only if the host happened to export it (ENABLE_EXPORTS/-rdynamic); otherwise
 * RTLD_NOW fails the plugin.
 *
 * The helpers are therefore part of the ABI by value: a plugin carries the dtype table it was
 * built against, and RAD_ABI_VERSION enforces the match (the loader refuses a mismatch by name,
 * with both versions printed).
 *
 * Everything here is pure, small and total: nothing allocates or holds state, and nothing may
 * start to. */
static inline const char* rad_strerror(int status) {
    switch (status) {
        case RAD_OK:            return "ok";
        case RAD_E_SHAPE:       return "shape this kernel does not serve";
        case RAD_E_DTYPE:       return "dtype this kernel does not serve";
        case RAD_E_ALIGN:       return "alignment the kernel requires was not met";
        case RAD_E_STRIDE:      return "non-contiguous operand where contiguity is required";
        case RAD_E_SCRATCH:     return "scratch buffer absent or too small";
        case RAD_E_NOMEM:       return "out of memory";
        case RAD_E_NOKERNEL:    return "no kernel resolved";
        case RAD_E_NOSCHEMA:    return "op has no declared schema";
        case RAD_E_SCHEMA:      return "parameters do not satisfy the op schema";
        case RAD_E_ABI:         return "plugin ABI version mismatch";
        case RAD_E_DUPLICATE:   return "duplicate registration";
        case RAD_E_INVAL:       return "invalid argument";
        case RAD_E_IO:          return "io error";
        case RAD_E_FORMAT:      return "container format error";
        case RAD_E_DEVICE:      return "device error";
        case RAD_E_STATE:       return "called in the wrong phase";
        case RAD_E_NOTFOUND:    return "not found";
        case RAD_E_UNSUPPORTED: return "unsupported";
        case RAD_E_FULL:        return "pool or table exhausted";
        case RAD_E_CANCELLED:   return "cancelled";
        default:                return "unknown error";
    }
}

/* ------------------------------------------------------------------ dtypes */
/* The core understands dtypes (spec §19.6): libref computes on tensors, rad-kbench compares them
 * elementwise, the buffer planner sizes buffers from shapes, and rad-convert writes bytes.
 *
 * The enum is open at the top: values >= RAD_DT_PLUGIN_BASE are kernel-plugin private, sized by
 * the plugin's own layout hook and never interpreted by the core. A format invented for one kernel
 * lives there (spec §19.6). */
enum {
    RAD_DT_INVALID = 0,

    /* float */
    RAD_F32   = 1,
    RAD_F16   = 2,
    RAD_BF16  = 3,
    RAD_F8E4M3 = 4,
    RAD_F8E5M2 = 5,

    /* integer */
    RAD_I8    = 8,
    RAD_U8    = 9,
    RAD_I16   = 10,
    RAD_I32   = 11,
    RAD_I64   = 12,
    RAD_U32   = 13,
    RAD_BOOL  = 14,

    /* Sub-byte codes and the exponent scale. Element counts are logical elements; byte sizes
     * come from rad_dtype_bits(). They pack along a row, lowest bits first (element 2j of a 4-bit
     * row in bits 0-3 of byte j), and a row starts on a byte (rad_encoding.h). What a code means
     * is its encoding's business; these give only its width and signedness. */
    RAD_I4      = 16,   /* two's complement, -8..7 */
    RAD_I2      = 17,   /* two's complement, -2..1 */
    RAD_FP4E2M1 = 18,   /* OCP FP4: 1-2-1, no inf or nan, max 6 */
    RAD_U4      = 19,   /* 0..15 */
    RAD_U2      = 20,   /* 0..3 */
    RAD_E8M0    = 21,   /* OCP E8M0: 2^(e-127), 0xFF is nan -- an MX block's shared scale */
    RAD_U1      = 22,   /* a bit: a sign plane, a binary code */
    RAD_I3      = 23,
    RAD_U3      = 24,
    RAD_I5      = 25,
    RAD_U5      = 26,
    RAD_I6      = 27,   /* also the K-quants' 6-bit scale with its offset removed */
    RAD_U6      = 28,
    RAD_I7      = 29,
    RAD_U7      = 30,
    RAD_U9      = 31,   /* 9- to 12-bit unsigned: lattice grid indices (IQ3_S, IQ2_S, IQ1_S) */
    RAD_U10     = 32,
    RAD_U11     = 33,
    RAD_U12     = 34,
    RAD_FP6E2M3 = 35,   /* OCP FP6: 1-2-3, bias 1, no inf or nan, max 7.5 */
    RAD_FP6E3M2 = 36,   /* OCP FP6: 1-3-2, bias 3, no inf or nan, max 28 */
    RAD_U16     = 37,

    RAD_DT_PLUGIN_BASE = 256
};
/* The dtype table, in the header for the reason given above rad_strerror.
 *
 * Sub-byte formats carry their packed width here. A shared scale is never folded into a code's
 * width (it is a plane of its own, rad_encoding.h), so every width below is exact. */
typedef struct RadDtRow { uint32_t dt; const char* name; int bits; } RadDtRow;

static const RadDtRow rad_dt_table[] = {
    { RAD_F32,    "f32",      32 },
    { RAD_F16,    "f16",      16 },
    { RAD_BF16,   "bf16",     16 },
    { RAD_F8E4M3, "fp8_e4m3",  8 },
    { RAD_F8E5M2, "fp8_e5m2",  8 },
    { RAD_I8,     "i8",        8 },
    { RAD_U8,     "u8",        8 },
    { RAD_I16,    "i16",      16 },
    { RAD_I32,    "i32",      32 },
    { RAD_I64,    "i64",      64 },
    { RAD_U32,    "u32",      32 },
    { RAD_BOOL,   "bool",      8 },
    { RAD_I4,      "i4",        4 },
    { RAD_I2,      "i2",        2 },
    { RAD_FP4E2M1, "fp4_e2m1",  4 },
    { RAD_U4,      "u4",        4 },
    { RAD_U2,      "u2",        2 },
    { RAD_E8M0,    "e8m0",      8 },
    { RAD_U1,      "u1",        1 },
    { RAD_I3,      "i3",        3 },
    { RAD_U3,      "u3",        3 },
    { RAD_I5,      "i5",        5 },
    { RAD_U5,      "u5",        5 },
    { RAD_I6,      "i6",        6 },
    { RAD_U6,      "u6",        6 },
    { RAD_I7,      "i7",        7 },
    { RAD_U7,      "u7",        7 },
    { RAD_U9,      "u9",        9 },
    { RAD_U10,     "u10",      10 },
    { RAD_U11,     "u11",      11 },
    { RAD_U12,     "u12",      12 },
    { RAD_FP6E2M3, "fp6_e2m3",  6 },
    { RAD_FP6E3M2, "fp6_e3m2",  6 },
    { RAD_U16,     "u16",      16 },
};
enum { RAD_DT_TABLE_N = (int)(sizeof(rad_dt_table) / sizeof(rad_dt_table[0])) };

/* Bits per logical element. Returns 0 for a plugin-private dtype -- the plugin sizes those. */
static inline int rad_dtype_bits(uint32_t dtype) {
    for (int i = 0; i < RAD_DT_TABLE_N; ++i)
        if (rad_dt_table[i].dt == dtype) return rad_dt_table[i].bits;
    return 0;
}

static inline const char* rad_dtype_name(uint32_t dtype) {
    for (int i = 0; i < RAD_DT_TABLE_N; ++i)
        if (rad_dt_table[i].dt == dtype) return rad_dt_table[i].name;
    return dtype >= RAD_DT_PLUGIN_BASE ? "plugin-private" : "invalid";
}

/* RAD_DT_INVALID if unknown. The alternate spellings are those found in checkpoints (safetensors)
 * and kernel dtype strings. */
static inline uint32_t rad_dtype_parse(const char* name) {
    if (!name) return RAD_DT_INVALID;
    for (int i = 0; i < RAD_DT_TABLE_N; ++i)
        if (strcmp(rad_dt_table[i].name, name) == 0) return rad_dt_table[i].dt;
    if (!strcmp(name, "float32") || !strcmp(name, "fp32")) return RAD_F32;
    if (!strcmp(name, "float16") || !strcmp(name, "fp16") || !strcmp(name, "half")) return RAD_F16;
    if (!strcmp(name, "bfloat16")) return RAD_BF16;
    if (!strcmp(name, "e4m3") || !strcmp(name, "fp8")) return RAD_F8E4M3;
    if (!strcmp(name, "e5m2")) return RAD_F8E5M2;
    if (!strcmp(name, "int8"))  return RAD_I8;
    if (!strcmp(name, "int32")) return RAD_I32;
    return RAD_DT_INVALID;
}

/* Byte size of n elements, rounding up. 0 if the dtype is plugin-private. */
static inline int64_t rad_dtype_bytes(uint32_t dtype, int64_t n) {
    const int bits = rad_dtype_bits(dtype);
    if (bits == 0) return 0;
    return ((int64_t)bits * n + 7) / 8;
}

/* ------------------------------------------------------------------ tensors */
enum { RAD_MAX_RANK = 6 };

/* Where an operand's bytes lie, as the engine's placement knows it: a buffer's declared domain,
 * the site the planner gave a weight (WeightInfo::site, so a table placed in host memory and read
 * over the link says HOST), a KV operand the block manager's device pool holds (DEVICE). UNKNOWN
 * is the engine declining to say -- a raw pointer it did not allocate, a weight a mover relocated
 * (it republishes the pointer, not the site) -- and means a plugin may not assume either: it must
 * either ask what it can itself or report the figure as unmeasured. */
enum { RAD_SPACE_UNKNOWN = 0, RAD_SPACE_DEVICE = 1, RAD_SPACE_HOST = 2 };

/* Strides are in elements, not bytes, since a byte stride would be fractional for sub-byte
 * dtypes. */
typedef struct RadTensor {
    void*    data;
    uint32_t dtype;
    uint32_t rank;
    int64_t  shape [RAD_MAX_RANK];
    int64_t  stride[RAD_MAX_RANK];
    int      space;   /* RAD_SPACE_*: where data lies, when the engine knows */
} RadTensor;

static inline int64_t rad_tensor_numel(const RadTensor* t) {
    if (!t || t->rank == 0) return 0;
    int64_t n = 1;
    for (uint32_t i = 0; i < t->rank; ++i) n *= t->shape[i];
    return n;
}

/* A rank-0 tensor and a tensor whose non-unit axes are dense are both contiguous. A shape-1 axis
 * may carry any stride and still describe the same bytes, so its stride is not checked. */
static inline int rad_tensor_is_contiguous(const RadTensor* t) {
    if (!t || t->rank == 0) return 1;
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return 0;
        acc *= t->shape[i];
    }
    return 1;
}

/* Fill strides for a dense row-major tensor whose shape and rank are already set. */
static inline void rad_tensor_pack(RadTensor* t) {
    if (!t || t->rank == 0) return;
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) { t->stride[i] = acc; acc *= t->shape[i]; }
}

/* ------------------------------------------------------------------ parameters */
/* An op's parameters are named scalars: the geometry the selector matches constraints against.
 * A parameter may be a fixed value or a range, which turns one declared op into a bucket table
 * (spec §2.2). At most one parameter per op may be ranged. */
enum { RAD_P_INT = 1, RAD_P_STR = 2, RAD_P_RANGE = 3, RAD_P_F64 = 4 };

typedef struct RadParam {
    const char* key;
    int         kind;      /* RAD_P_INT | RAD_P_STR | RAD_P_RANGE | RAD_P_F64 */
    long long   ival;      /* RAD_P_INT, and the low bound of RAD_P_RANGE */
    long long   ihi;       /* RAD_P_RANGE: the high bound, inclusive */
    const char* sval;      /* RAD_P_STR */
    double      dval;      /* RAD_P_F64: eps, rope theta, an attention scale. A double rather
                            * than a "%.9g" string, which would be exact for an IEEE single but
                            * unreadable in a graph dump. Constraints never match on a float:
                            * a tolerance is not a predicate. */
} RadParam;

/* Builders. RAD_PARAMS below wraps these into a positional list. */
#define RAD_INT(k, v)      ((RadParam){ (k), RAD_P_INT,   (long long)(v), 0, 0, 0.0 })
#define RAD_STR(k, v)      ((RadParam){ (k), RAD_P_STR,   0, 0, (v), 0.0 })
#define RAD_F64(k, v)      ((RadParam){ (k), RAD_P_F64,   0, 0, 0, (double)(v) })
#define RAD_RANGE(k, lo, hi) ((RadParam){ (k), RAD_P_RANGE, (long long)(lo), (long long)(hi), 0, 0.0 })

/* ------------------------------------------------------------------ constraints */
/* A kernel's necessary condition on a geometry, deliberately not sufficient: strides,
 * contiguity, alignment and buffer sizes are properties of a call rather than of a model, and the
 * entry point still rejects what it cannot run (spec §2.1). */
enum { RAD_C_EQ = 0, RAD_C_LE = 1, RAD_C_GE = 2, RAD_C_DIV = 3, RAD_C_IN = 4 };

typedef struct RadConstraint {
    const char* key;
    int         op;
    long long   ival;   /* EQ / LE / GE / DIV */
    const char* sval;   /* IN: space-separated set of admissible strings */
} RadConstraint;

/* A constraint on the machine rather than the geometry. A key starting "device." is matched
 * against what the engine measured about the ranks' devices at launch, not against op parameters,
 * so architecture plugins and schemas need not carry machine capabilities. Such a key is in no
 * geometry, so it changes no tuning key and no selection fold. Where nothing measured it (a tool
 * selecting without an engine) it is absent, and a constraint on an absent key does not hold.
 *
 *   device.p2p   1 when every rank's device can read and write every other rank's memory
 *                directly; 0 when some pair cannot. One rank has no peer to reach, and is 1. */
#define RAD_DEVICE_FACT_PREFIX "device."
#define RAD_FACT_P2P           "device.p2p"

#define RAD_C(k, o, v)  { (k), (o), (long long)(v), 0 }
#define RAD_CEQ(k, v)   { (k), RAD_C_EQ,  (long long)(v), 0 }
#define RAD_CLE(k, v)   { (k), RAD_C_LE,  (long long)(v), 0 }
#define RAD_CGE(k, v)   { (k), RAD_C_GE,  (long long)(v), 0 }
#define RAD_CDIV(k, v)  { (k), RAD_C_DIV, (long long)(v), 0 }
#define RAD_CIN(k, s)   { (k), RAD_C_IN,  0, (s) }

/* ------------------------------------------------------------------ handles */
/* Opaque, dense, small. A handle is an index into a core-owned table; zero is never a valid
 * handle, so a zeroed struct is an unset one and a failed decl_* is falsy. */
typedef uint32_t rad_weight;
typedef uint32_t rad_buf;
typedef uint32_t rad_op;
/* A KV group handle. Here with the other handles because RadBufDecl in rad_builder.h carries
 * one. */
typedef uint32_t rad_kvgroup;

#define RAD_NULL_HANDLE 0u

/* ------------------------------------------------------------------ streams */
/* Opaque: the core never dereferences one. On a card it is the engine's own queue, not a
 * hipStream_t. A kernel plugin may use it only with hipLaunchKernel (and so <<<>>>),
 * hipMemcpyAsync, hipMemcpy2DAsync and hipMemsetAsync, which the engine implements itself, and
 * must pass it to nothing else in HIP (docs/PLUGIN.md, "What a launch may do"). Under the host
 * backend the host kernels ignore it. */
typedef void* RadStream;

#ifdef __cplusplus
}   /* extern "C" */
#endif
#endif /* RAD_TYPES_H */
