/* plugin/qsatable/qsaattn.hip -- the device launch for `qsa_attn`, the masked read of a shared table.
 *
 * ONE LANE PER (query row, query head). A lane walks the tile's union from its row's `row_off`, skips
 * the entries its mask does not hold, maps each block index through its sequence's block table,
 * appends its own partial page and sums the softmax online in that order. Nothing is shared between
 * lanes, so the summation order is the key order and two tables that give a row the same keys in the
 * same order give the same bytes -- which is what `tests/r4d_emu/test_qsa.cpp` (`masked read:`)
 * requires of a per-row table (tile 1) and of the shared one.
 *
 * PROTOTYPE. A lane per head recomputes the dot product for every dim and reads the union per head
 * instead of once per tile in LDS; a serving form keeps a row's union in LDS, scores keys with the
 * wave and prefetches V across the accumulation. What this row exists for is a statement of the op a
 * host oracle and a byte comparison can hold: the read path behind the architecture's switch.
 */
#include <hip/hip_runtime.h>

#include "qsaattn.h"
#include "../../libr4d/r4d_emit.h"

namespace {

/* The widest head this prototype accumulates in one frame. */
constexpr int kMaxHeadDim = 512;

/* The three dtypes the QSA path holds: an activation read as bf16, a cache of bf16 or fp8_e4m3. The
 * ABI's conversions are host functions, so the device side states its own; both round as round-to-
 * nearest-even, which is the point of the byte comparison (`abi/rad_plugin.h`). */
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    const uint32_t s = (uint32_t)(b >> 7) << 31, e = (b >> 3) & 0xfu, m = b & 0x7u;
    if (e == 0) {                     /* subnormal */
        float v = (float)m;
        for (int k = 0; k < 9; ++k) v *= 0.5f;
        return s ? -v : v;
    }
    return __builtin_bit_cast(float, s | ((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ float ld(const void* base, uint32_t dt, int64_t i) {
    if (dt == RAD_BF16) {
        const uint16_t h = *(const uint16_t*)((const char*)base + i * 2);
        return __builtin_bit_cast(float, (uint32_t)h << 16);
    }
    if (dt == RAD_F32) return ((const float*)base)[i];
    if (dt == RAD_F8E4M3) return e4m3_to_f32(((const uint8_t*)base)[i]);
    return 0.0f;
}

__device__ __forceinline__ void st(void* base, uint32_t dt, int64_t i, float v) {
    if (dt == RAD_BF16) {
        const uint32_t u = __builtin_bit_cast(uint32_t, v);
        ((uint16_t*)base)[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    } else if (dt == RAD_F32) {
        ((float*)base)[i] = v;
    }
}

/* One page's visible tokens, `first..last`, in key order. `m`, `l` and `acc` are the running online
 * softmax: scores are read with a dot per key and the maximum is folded in as it advances, which
 * rescales the running sum and the accumulator by exp(m_old - m_new) -- the same values in the same
 * order as a two-pass form, in one pass. */
__device__ void take_page(const RadQsaAttnCall& c, int64_t kh, int64_t page, int64_t first,
                          int64_t last, const float* qf, float* m, float* l, float* acc) {
    for (int64_t t = first; t <= last; ++t) {
        const int64_t kb = page * c.kv_sb + kh * c.kv_sh + t * c.kv_so;
        float dot = 0.0f;
        for (int64_t d = 0; d < c.head_dim; ++d)
            dot += qf[d] * ld(c.kv, c.kv_dtype, kb + d * c.kv_sd);
        const float sj = dot * c.scale;
        if (sj > *m) {
            const float a = expf(*m - sj);
            *l *= a;
            for (int64_t d = 0; d < c.head_dim; ++d) acc[d] *= a;
            *m = sj;
        }
        const float w = expf(sj - *m);
        *l += w;
        for (int64_t d = 0; d < c.head_dim; ++d)
            acc[d] += w * ld(c.kv, c.kv_dtype, kb + c.head_dim + d * c.kv_sd);
    }
}

__global__ void qsa_attn_kernel(RadQsaAttnCall c) {
    const int64_t unit = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (unit >= c.M * c.n_head) return;
    const int64_t row = unit / c.n_head;
    const int64_t h   = unit % c.n_head;
    const int64_t kh  = h / c.gqa;

    const int64_t p    = c.pos[row];
    const int64_t own  = (p + 1) / c.ratio;
    const int64_t tile = c.row_off[row] / c.u_max;
    const int64_t tlen = c.tlen[tile];
    const int32_t* tab  = c.tab + c.row_off[row];
    const int32_t* mask = c.mask + row * c.u_max;
    const int32_t* bts  = c.bt + rad_qsa_attn_seq(c, row) * c.bt_pitch;

    float qf[kMaxHeadDim];
    for (int64_t d = 0; d < c.head_dim; ++d)
        qf[d] = ld(c.q, c.q_dtype, row * c.q_pitch + h * c.head_dim + d);

    float acc[kMaxHeadDim];
    for (int64_t d = 0; d < c.head_dim; ++d) acc[d] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    for (int64_t j = 0; j < tlen; ++j) {
        if (!mask[j]) continue;
        const int64_t b = tab[j];
        if (b < 0 || b >= own) continue;
        const int64_t page = bts[b];
        if (page < 0) continue;
        take_page(c, kh, page, 0, c.ratio - 1, qf, &m, &l, acc);
    }
    /* The row's partial page, by position, appended last: the page the chunk itself may be filling,
     * visible up to the row's own token. */
    const int64_t off = (p + 1) % c.ratio;
    if (off != 0 && own < c.blocks) {
        const int64_t page = bts[own];
        if (page >= 0) take_page(c, kh, page, 0, off - 1, qf, &m, &l, acc);
    }

    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    for (int64_t d = 0; d < c.head_dim; ++d)
        st(c.out, c.out_dtype, row * c.out_pitch + h * c.head_dim + d, acc[d] * inv);
}

}  /* namespace */

int rad_qsa_attn_launch(const RadArgs* a, RadStream stream) {
    RadQsaAttnCall c{};
    const int rc = rad_qsa_attn_args(a, &c);
    if (rc != RAD_OK) return rc;
    if (c.head_dim > kMaxHeadDim) return RAD_E_UNSUPPORTED;
    const int64_t units = c.M * c.n_head;
    const int64_t threads = 64;
    const unsigned grid = (unsigned)((units + threads - 1) / threads);
    return R4D_EMIT(stream, qsa_attn_kernel, dim3(grid), dim3((unsigned)threads), 0, c);
}
