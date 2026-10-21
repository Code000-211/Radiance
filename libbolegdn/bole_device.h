/* bole_device.h -- the gfx1201 helpers the Bole rows need, restated from libr4d and cited.
 *
 * libr4d's r4d_common.h, r4d_gdn_wmma.h, r4d_gdn_conv_tap.h, r4d_q8.h and r4d_emit.h are private to
 * libr4d: an out-of-tree plugin sees only abi/, so the conversions, the DPP exchanges, the conv
 * activation and the INT8 code packer are written out here. Each is a copy of the frozen libr4d
 * definition, because a result is defined by its rounding: a bf16 narrowing that truncates where
 * libr4d rounds would make this row's output differ from the engine's own GDN row by one ulp per
 * element, silently, and the byte comparison in test/bole_gdn_gpu_test would be comparing the wrong
 * thing.
 *
 *   bole_f2bf       libr4d/r4d_gdn_wmma.h:16      round-half-away, (u + 0x8000) >> 16
 *   bole_bf2f       libr4d/r4d_gdn_wmma.h:39      widen, exact (a shift)
 *   bole_xor        libr4d/r4d_common.h:88        DPP row_xmask / v_permlanex16, no LDS round trip
 *   bole_xor_d      libr4d/r4d_common.h:94        the same over a 64-bit value, halves separate
 *   bole_silu       libr4d/r4d_gdn_conv_tap.h:15  z * rcpf(1 + exp(-z)), not an IEEE divide
 *   bole_q8_pack4   libr4d/r4d_q8.h:25            four codes in one dword, value 0 low
 *   bole_q8_top     libr4d/r4d_q8.h:16            127 (int8) / 448 (E4M3)
 *
 * Deviced-only by construction: a host compile of this header fails on the first builtin.
 */
#ifndef BOLE_DEVICE_H
#define BOLE_DEVICE_H

#include "bole_layout.h"

#include <hip/hip_runtime.h>

/* Round-half-away rather than RTNE: the two differ only when the discarded bits are exactly 0x8000,
 * and libr4d's conv and output boundaries both use this form. */
__device__ __forceinline__ unsigned short bole_f2bf(float f) {
    return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}

/* A shift, since every bf16 is exactly a float. */
__device__ __forceinline__ float bole_bf2f(unsigned short h) {
    return __builtin_bit_cast(float, (unsigned)h << 16);
}

/* `lane ^ N` through DPP instead of ds_bpermute_b32. Every caller runs it on a whole wave, so the
 * row and bank masks are all ones and bound_ctrl is moot. */
template <int N>
__device__ __forceinline__ uint32_t bole_xor_u32(uint32_t v) {
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16, "a lane xor within the wave");
    if constexpr (N < 16)
        return (uint32_t)__builtin_amdgcn_update_dpp(0, (int)v, 0x160 | N, 0xf, 0xf, true);
    else
        return (uint32_t)__builtin_amdgcn_permlanex16((int)v, (int)v, 0x76543210u, 0xFEDCBA98u,
                                                     false, false);
}

template <int N>
__device__ __forceinline__ float bole_xor(float v) {
    return __builtin_bit_cast(float, bole_xor_u32<N>(__builtin_bit_cast(uint32_t, v)));
}

/* A 64-bit value's partner, the halves exchanged separately, as libr4d does: the Bole walk rows
 * contract in FP64 and still close their dot with one of these. */
template <int N>
__device__ __forceinline__ double bole_xor_d(double v) {
    const unsigned long long u = __builtin_bit_cast(unsigned long long, v);
    const uint32_t lo = bole_xor_u32<N>((uint32_t)u);
    const uint32_t hi = bole_xor_u32<N>((uint32_t)(u >> 32));
    return __builtin_bit_cast(double, ((unsigned long long)hi << 32) | lo);
}

/* The reciprocal form on purpose: the result is rounded to bf16 immediately, so an IEEE divide buys
 * nothing against bf16's 2^-8 quantum, and this expression is libr4d's conv boundary. */
__device__ __forceinline__ float bole_silu(float z) {
    return z * __builtin_amdgcn_rcpf(1.0f + __expf(-z));
}

template <bool I8>
__device__ __forceinline__ constexpr float bole_q8_top() {
    return I8 ? 127.0f : 448.0f;
}

/* Four values times `inv` as four codes in one dword, value 0 in the low byte. The E4M3 clamp is
 * required: v_cvt_pk_fp8_f32 returns NaN out of range instead of saturating. */
template <bool I8>
__device__ __forceinline__ uint32_t bole_q8_pack4(float a, float b, float c, float d, float inv) {
    if constexpr (I8) {
        const int ia = (int)rintf(fminf(fmaxf(a * inv, -127.0f), 127.0f));
        const int ib = (int)rintf(fminf(fmaxf(b * inv, -127.0f), 127.0f));
        const int ic = (int)rintf(fminf(fmaxf(c * inv, -127.0f), 127.0f));
        const int id = (int)rintf(fminf(fmaxf(d * inv, -127.0f), 127.0f));
        return ((uint32_t)ia & 0xFFu) | (((uint32_t)ib & 0xFFu) << 8) |
               (((uint32_t)ic & 0xFFu) << 16) | (((uint32_t)id & 0xFFu) << 24);
    } else {
        int pk = 0;
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(a * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(b * inv, -448.0f), 448.0f), pk, false);
        pk = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(c * inv, -448.0f), 448.0f),
                                             fminf(fmaxf(d * inv, -448.0f), 448.0f), pk, true);
        return (uint32_t)pk;
    }
}

#endif /* BOLE_DEVICE_H */
