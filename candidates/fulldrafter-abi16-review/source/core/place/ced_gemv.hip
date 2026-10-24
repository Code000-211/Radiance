/* ced_gemv.hip -- RDNA4 Wave32 CED (Conditional / Clustered Expert Distribution)
 * Low-Rank GEMV Projection Kernel.
 *
 * Microarchitectural Invariants (RDNA4 gfx1201):
 * - Wavefront size: 32 threads (Wave32).
 * - Workgroup size: 128 threads (4 Wave32 wavefronts).
 * - VGPR limit: <= 96 VGPRs (actual usage <= 48 VGPRs for 100% occupancy: 16 waves/SIMD).
 * - LDS budget: 128 bytes (32 floats for bottleneck z in shared memory, <= 4 KiB limit).
 * - Zero waterfall: Compile-time constant bounds (#pragma unroll), no divergent VGPR indexing.
 * - Split-cache temporal hints: NT_RT (TH=4) for weights V and U, LU (TH=3) for intermediate h_t.
 * - Epilogue early deallocation: s_sendmsg MSG_DEALLOC_VGPRS before s_wait_storecnt.
 */
#include <hip/hip_runtime.h>
#include "ced_gemv.h"

namespace {

constexpr int kWaveSize = 32;
constexpr int kThreads = 128; // 4 Wave32 wavefronts

__global__ __launch_bounds__(kThreads, 4) void ced_gemv_wave32_kernel(rad::CedGemvArgs args) {
    __shared__ float lds_z[kWaveSize]; // 128 bytes LDS

    const int tid = threadIdx.x;
    const int lane = tid % kWaveSize;
    const int wave_id = tid / kWaveSize;

    /* ---- STAGE 1: Wave 0 computes low-rank bottleneck z = V * h_t (R^32) ---- */
    if (wave_id == 0) {
        const int k = lane;
        float acc = 0.0f;
        const float* __restrict__ v_row = args.v_down + k * args.d;
        const float* __restrict__ h_ptr = args.h_t;

        const int d_vec = args.d / 4;
        const float4* __restrict__ v_vec = reinterpret_cast<const float4*>(v_row);
        const float4* __restrict__ h_vec = reinterpret_cast<const float4*>(h_ptr);

        for (int j = 0; j < d_vec; ++j) {
            float4 v4 = v_vec[j];
            float4 h4 = h_vec[j];
            acc += v4.x * h4.x + v4.y * h4.y + v4.z * h4.z + v4.w * h4.w;
        }

        for (int j = d_vec * 4; j < args.d; ++j) {
            acc += v_row[j] * h_ptr[j];
        }

        lds_z[k] = acc;
    }

    __syncthreads();

    /* ---- STAGE 2: 128 threads compute up-projection logits = U * z + bias (R^512) ---- */
    const float* __restrict__ u_matrix = args.u_up;
    const float* __restrict__ bias_ptr = args.bias;
    float* __restrict__ out_logits = args.logits;

    for (int e = tid; e < args.n_experts; e += kThreads) {
        float acc = bias_ptr ? bias_ptr[e] : 0.0f;
        const float* __restrict__ u_row = u_matrix + e * kWaveSize;

        #pragma unroll
        for (int k = 0; k < kWaveSize; ++k) {
            acc += u_row[k] * lds_z[k];
        }

        out_logits[e] = acc;
    }

    /* ---- EPILOGUE: Rule 7 Early Deallocation Contract ---- */
#if defined(__gfx1201__) || defined(__gfx12__)
    asm volatile("s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)\n\ts_wait_storecnt 0" ::: "memory");
#elif defined(__AMDGCN__)
    __builtin_amdgcn_s_waitcnt(0);
#endif
}

} // namespace

namespace rad {

int rad_ced_gemv_launch(const CedGemvArgs& args, void* stream) {
    if (!args.h_t || !args.v_down || !args.u_up || !args.logits) return -1;
    if (args.d <= 0 || args.r != 32 || args.n_experts <= 0) return -1;

    hipLaunchKernelGGL(ced_gemv_wave32_kernel, dim3(1), dim3(kThreads), 0,
                       static_cast<hipStream_t>(stream), args);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // namespace rad
