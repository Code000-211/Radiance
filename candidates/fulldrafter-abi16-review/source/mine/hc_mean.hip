/* hc_mean.hip -- the DFlash2 tap reduction on the device: x[m,i] = mean over the hc streams of
 * h[m, c*N+i], summed in f32 and stored bf16 ONCE. The definition is libref/ref_hc.cpp; mine/hc_mean.h
 * states the contract and the geometry this row declares.
 *
 * ============================== WHAT IT IS NOT ==============================
 *
 * NOT `hc_read`. That op normalises each stream (one rsqrt per N), applies a low-rank gate,
 * multiplies by a per-stream gain and then takes the mean; substituting it here would give a tap
 * that is a different function of the same bytes, and the DFlash2 capture is
 * `hidden_states.unflatten(-1, (hc, n)).mean(dim=-2)` with nothing else on it.
 *
 * ============================== THE LAYOUT ==============================
 *
 * A lane owns eight consecutive output elements and reads the matching element of each stream:
 * element (m, c, i) is at h + m*h_pitch + c*N + i, so the hc pointers are `N` apart and
 * consecutive lanes touch consecutive addresses inside each stream.
 *
 * THE OUTPUT'S PITCH IS ITS OWN. `x` arrives as a column view of the tap plane: width N, at column
 * t*N, with the plane's n_taps*N as stride[0]. The store is `x + m*x_pitch + i`, so the columns of
 * the other taps are untouched and no separate parameter has to say so.
 *
 * NO TAIL BRANCH, because the row is declared at N % 8 == 0: a lane's group of 8 sits wholly inside
 * the row, so the bounds test is the lane's own group index and there is no partial group to hand
 * to a scalar path. That is what lets the entry point refuse any other width instead of carrying a
 * second code path the oracle would have to agree with.
 *
 * NOTHING IS ALLOCATED AND NOTHING SYNCHRONISES: one launch, one argument struct by value, and the
 * only host work is the validation below (which dereferences no device memory).
 */
#include <hip/hip_runtime.h>

#include "hc_mean.h"

namespace {

constexpr int kThreads = 128;

/* f32 -> bf16, round-to-nearest-even with the tie on the surviving bit's parity, NaN forced quiet.
 * Spelled out rather than including libr4d/r4d_common.h: this plugin does not link libr4d, and a
 * device copy of a rounding rule that lives in another plugin's header is exactly how two kernels
 * come to disagree about the last bit. rad_plugin.h's rad_f32_to_bf16 is the definition; this is
 * the same arithmetic in device code. */
__device__ __forceinline__ uint16_t to_bf16(float f) {
    const uint32_t u = __builtin_bit_cast(uint32_t, f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* bf16 -> f32 is a shift: every bf16 is exactly a float. */
__device__ __forceinline__ float to_f32(uint16_t h) {
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

/* The row declares hc 4, so the streams are unrolled rather than looped: `rad_hc_mean_args` has
 * already refused any other count, and a loop bound the compiler cannot see is a loop it cannot
 * flatten into four adds. `Args` carries no hc for that reason. */
struct Args {
    const uint16_t* __restrict__ h;
    uint16_t* __restrict__ x;
    int64_t rows, N, h_pitch, x_pitch;
};

__global__ __launch_bounds__(kThreads) void mean_kernel(Args a) {
    /* grid.x indexes rows and grid.y the group of 8 columns within a row: the column dimension is
     * what keeps a prefill chunk's row count from changing the arithmetic per element. A 2560-wide
     * tap is 320 groups, so 2.5 blocks of 128 threads per row and the last one is partial. */
    const int64_t m = blockIdx.x;
    const int64_t g = blockIdx.y * (int64_t)(kThreads * 8) + threadIdx.x * 8;
    if (m >= a.rows || g >= a.N) return;

    const uint16_t* hp = a.h + m * a.h_pitch + g;
    const uint16_t* s0 = hp;
    const uint16_t* s1 = hp + a.N;
    const uint16_t* s2 = hp + 2 * a.N;
    const uint16_t* s3 = hp + 3 * a.N;

    uint16_t* xp = a.x + m * a.x_pitch + g;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // Match the reference's +0 accumulator, including four negative-zero inputs.
        const float acc = 0.0f + to_f32(s0[k]) + to_f32(s1[k]) + to_f32(s2[k]) + to_f32(s3[k]);
        xp[k] = to_bf16(acc * 0.25f);
    }
}

}  /* namespace */

/* The launch. `rad_hc_mean_args` has already refused every geometry this row was not given, so
 * there is no second validation here and no path that computes something the oracle would not:
 * either the call is the op or it returned a reason. */
int rad_mine_hc_mean_launch(const RadArgs* a, RadStream stream) {
    RadHcMeanCall call{};
    const int rc = rad_hc_mean_args(a, &call);
    if (rc != RAD_OK) return rc;

    Args args{};
    args.h = static_cast<const uint16_t*>(call.h);
    args.x = static_cast<uint16_t*>(call.x);
    args.rows = call.rows;
    args.N = call.N;
    args.h_pitch = call.h_pitch;
    args.x_pitch = call.x_pitch;

    const unsigned gx = (unsigned)call.rows;
    const unsigned gy = (unsigned)((call.N + kThreads * 8 - 1) / (kThreads * 8));
    void* kargs[] = { &args };
    const hipError_t e = hipLaunchKernel((const void*)mean_kernel, dim3(gx, gy), dim3(kThreads),
                                         kargs, 0, (hipStream_t)stream);
    return e == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int rad_mine_hc_mean_count(void) { return 1; }

const RadKernelInfo* rad_mine_hc_mean_at(int i) {
    static const RadKernelInfo row = rad_hc_mean_row(rad_mine_hc_mean_launch);
    return i == 0 ? &row : nullptr;
}
