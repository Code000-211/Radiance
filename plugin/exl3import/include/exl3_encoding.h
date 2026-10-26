/* exl3_encoding.h -- THE canonical EXL3 encoding (PLAN.md D4), as radiance encoding data, and the
 * GEMM dtype strings the architecture and kernels name it by. Header-only; owner Q.
 *
 * include/exl3_abi.h (owner R) includes this file; nothing else defines the encoding. A weight in it:
 *
 *   scheme     "exl3"                           a quantiser's own scheme: the core checks plane 0 and
 *                                               role uniqueness only (rad_enc_valid) and decodes it
 *                                               through the exl3 quantiser's decode hook
 *   transform  "<codebook>/h128"                "mul1/h128", "mcg/h128" or "3inst/h128": the trellis
 *                                               codebook, and the rotation (normalised Hadamard,
 *                                               block 128) on BOTH sides of the weight
 *   plane 0    codes  uK  TILED [1x1]           element (i, j) = the K-bit trellis symbol of W_hat at
 *                                               (k = j, n = i), position-major (common/exl3_decode.h)
 *   plane 1    suh    f16 TILED [0x1]           input scales, one a column (k); cut with the columns
 *   plane 2    svh    f16 TILED [1x0]           output scales, one a row (n); cut with the rows
 *   plane 3    suh.1  f16 TILED [0x1]           STACKED gate_up only: the up half's input scales;
 *                                               rows [rows/2, rows) read suh.1, the first half suh
 *
 * so a stacked 4-bit gate_up spells `exl3:codes=u4[1x1],suh=f16[*x1],svh=f16[1x*],suh.1=f16[*x1]/mul1/h128`.
 * The plane ORDER is fixed (the indices below) so down and gate_up share planes 0..2; read by role
 * where a role is enough. Encodings are identical byte for byte (rad_enc_equal) or different:
 * exl3_encoding_read() refuses anything exl3_encoding_make() would not have written.
 *
 * K is per weight and per encoding: two weights of different K are two encodings. A recipe's bits
 * option, the trellis tensor's last dimension (16*K) and the codes dtype must all agree.
 */
#ifndef EXL3_ENCODING_H
#define EXL3_ENCODING_H

#include "rad_encoding.h"
#include "../common/exl3_decode.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any change to the encoding or a declaration below. */
#define EXL3_ENCODING_VERSION 1

#define EXL3_SCHEME   "exl3"
#define EXL3_ROTATION "h128"   /* the second half of the transform string */

enum { EXL3_PLANE_CODES = 0, EXL3_PLANE_SUH = 1, EXL3_PLANE_SVH = 2, EXL3_PLANE_SUH1 = 3 };

/* The K a GEMM dtype string exists for: "exl3k2" .. "exl3k8". The DECODER reads K 1..8
 * (EXL3_BITS_MIN..MAX); K=1 has no dtype string, so no kernel can be named for it and the quantiser
 * refuses it. */
enum { EXL3_GEMM_BITS_MIN = 2, EXL3_GEMM_BITS_MAX = 8 };

/* The core dtype of a K-bit codes plane: RAD_U1..RAD_U8; RAD_DT_INVALID outside 1..8. */
static inline uint32_t exl3_codes_dtype(int bits) {
    static const uint32_t dt[9] = { RAD_DT_INVALID, RAD_U1, RAD_U2, RAD_U3, RAD_U4,
                                    RAD_U5, RAD_U6, RAD_U7, RAD_U8 };
    return bits >= EXL3_BITS_MIN && bits <= EXL3_BITS_MAX ? dt[bits] : (uint32_t)RAD_DT_INVALID;
}

/* The inverse: K of a codes dtype, or -1. */
static inline int exl3_bits_of_dtype(uint32_t dtype) {
    int k;
    for (k = EXL3_BITS_MIN; k <= EXL3_BITS_MAX; ++k)
        if (exl3_codes_dtype(k) == dtype) return k;
    return -1;
}

/* The GEMM dtype string an op constraint names for K-bit EXL3 weights; null when K has none. It
 * names K only: the codebook is in the encoding, and a kernel's layout hook refuses a codebook it
 * does not decode (RAD_E_DTYPE) -- the gfx1201 rows decode mul1 alone (PLAN.md D9). */
static inline const char* exl3_gemm_dtype(int bits) {
    static const char* const s[9] = { 0, 0, "exl3k2", "exl3k3", "exl3k4",
                                      "exl3k5", "exl3k6", "exl3k7", "exl3k8" };
    return bits >= EXL3_GEMM_BITS_MIN && bits <= EXL3_GEMM_BITS_MAX ? s[bits] : 0;
}

/* K of a GEMM dtype string, or -1. */
static inline int exl3_gemm_bits(const char* dtype) {
    int k;
    for (k = EXL3_GEMM_BITS_MIN; k <= EXL3_GEMM_BITS_MAX && dtype; ++k)
        if (strcmp(dtype, exl3_gemm_dtype(k)) == 0) return k;
    return -1;
}

/* The encoding of a K-bit weight in `codebook`; `stacked` nonzero for a gate_up (adds suh.1).
 * RAD_OK, or RAD_E_INVAL for K outside 1..8 or an unknown codebook. */
static inline int exl3_encoding_make(int bits, int codebook, int stacked, RadEncoding* out) {
    char t[RAD_ENC_STR];
    const char* cb = exl3_codebook_name(codebook);
    const uint32_t dt = exl3_codes_dtype(bits);
    if (!out || !cb || dt == RAD_DT_INVALID) return RAD_E_INVAL;
    rad_enc_clear(out);
    rad_enc_copy_str(out->scheme, EXL3_SCHEME);
    snprintf(t, sizeof t, "%s/%s", cb, EXL3_ROTATION);
    rad_enc_copy_str(out->transform, t);
    rad_enc_add_plane(out, "codes", dt, 1, 1);
    rad_enc_add_plane(out, "suh", RAD_F16, 0, 1);
    rad_enc_add_plane(out, "svh", RAD_F16, 1, 0);
    if (stacked) rad_enc_add_plane(out, "suh.1", RAD_F16, 0, 1);
    return RAD_OK;
}

/* Read an encoding back. RAD_E_UNSUPPORTED when the scheme is not "exl3" (not ours: a decoder
 * answers that and the core asks the next one); RAD_E_FORMAT when it says "exl3" but is not exactly
 * what exl3_encoding_make writes for some K, codebook and stacking; RAD_OK otherwise, with the
 * three filled in (any pointer may be null). */
static inline int exl3_encoding_read(const RadEncoding* e, int* bits, int* codebook, int* stacked) {
    RadEncoding want;
    char cb[RAD_ENC_STR];
    const char* slash;
    size_t n;
    int k, c, s;
    if (!e || !rad_enc_is(e, EXL3_SCHEME)) return RAD_E_UNSUPPORTED;
    if (e->n_planes < 3 || e->n_planes > 4) return RAD_E_FORMAT;
    slash = (const char*)memchr(e->transform, '/', RAD_ENC_STR);
    if (!slash) return RAD_E_FORMAT;
    n = (size_t)(slash - e->transform);
    memcpy(cb, e->transform, n);
    cb[n] = 0;
    k = exl3_bits_of_dtype(e->plane[EXL3_PLANE_CODES].dtype);
    c = exl3_codebook_from_name(cb);
    s = e->n_planes == 4;
    if (k < 0 || c < 0 || exl3_encoding_make(k, c, s, &want) != RAD_OK || !rad_enc_equal(e, &want))
        return RAD_E_FORMAT;
    if (bits) *bits = k;
    if (codebook) *codebook = c;
    if (stacked) *stacked = s;
    return RAD_OK;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* EXL3_ENCODING_H */
