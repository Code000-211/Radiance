// r4d_kv_anchor_pool4.hip -- Asymmetric Anchor-Pooling for 4:1 KV-Cache Compression (DeepSeek-CSA + KVarN)
//
// In hybrid Mamba/Attention models (such as Qwen Flash Next), Attention layers serve primarily
// for pinpoint needle-in-a-haystack retrieval, while GDN/Mamba layers process continuous context.
// An unweighted arithmetic mean over keys smudges high-frequency directional vectors.
//
// This kernel implements Asymmetric Anchor-Pooling:
// 1. Dominant Key Anchor Selection: In every 4-token temporal chunk B = {t_0, t_1, t_2, t_3},
//    the key with the highest L2 norm (i* = argmax ||K_i||_2) is retained bit-identically,
//    preserving exact directional sharpness and original RoPE phase.
// 2. Norm-Weighted Value Aggregation: Values are pooled via norm-weighted soft mixture:
//    w_i = ||K_i||_2 / sum(||K_j||_2), V_pooled = sum(w_i * V_i).
//
// Operates on native FP8 (E4M3) KV caches with HEAD_DIM=256 and GQA 6:1 (1 KV head per rank at TP2).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cmath>

#include "r4d.h"
#include "r4d_common.h"

#define HEAD_DIM 256
#define TOK_ELEMS (2 * HEAD_DIM) // K (256B) + V (256B) = 512B per token

__global__ void r4d_kv_anchor_pool4_kernel(
    uint8_t* __restrict__ kv,
    const int32_t* __restrict__ src_blocks,
    const int32_t* __restrict__ dst_blocks,
    int64_t kv_block_stride,
    int64_t kv_head_stride,
    int n_chunks
) {
    const int out_tok = blockIdx.x;      // 0..15 (16 output tokens per chunk)
    const int chunk   = blockIdx.y;      // 0..n_chunks-1
    const int head    = blockIdx.z;      // 0..n_heads-1
    const int lane    = threadIdx.x;     // 0..31 (32 lanes per wave)

    if (chunk >= n_chunks || out_tok >= 16) return;

    // Output destination pointers
    const int32_t dst_b = dst_blocks[chunk];
    uint8_t* out_k = kv + (size_t)dst_b * kv_block_stride + (size_t)head * kv_head_stride + (size_t)out_tok * TOK_ELEMS;
    uint8_t* out_v = out_k + HEAD_DIM;

    // 4 input candidate tokens: t0, t1, t2, t3
    const uint8_t* in_k[4];
    const uint8_t* in_v[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int in_tok = out_tok * 4 + i;
        const int b_idx = in_tok / 16;
        const int t_sub = in_tok % 16;
        const int32_t src_b = src_blocks[chunk * 4 + b_idx];
        in_k[i] = kv + (size_t)src_b * kv_block_stride + (size_t)head * kv_head_stride + (size_t)t_sub * TOK_ELEMS;
        in_v[i] = in_k[i] + HEAD_DIM;
    }

    // Each lane processes 8 elements (8 bytes) out of 256 elements
    const int d_offset = lane * 8;
    float norm_sq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint2 k_raw[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        k_raw[i] = *reinterpret_cast<const uint2*>(in_k[i] + d_offset);
        // Unpack 8 FP8 elements to 8 floats
        const auto f01 = __builtin_amdgcn_cvt_pk_f32_fp8((int)k_raw[i].x, false);
        const auto f23 = __builtin_amdgcn_cvt_pk_f32_fp8((int)k_raw[i].x, true);
        const auto f45 = __builtin_amdgcn_cvt_pk_f32_fp8((int)k_raw[i].y, false);
        const auto f67 = __builtin_amdgcn_cvt_pk_f32_fp8((int)k_raw[i].y, true);

        float local_sq = f01[0] * f01[0] + f01[1] * f01[1]
                       + f23[0] * f23[0] + f23[1] * f23[1]
                       + f45[0] * f45[0] + f45[1] * f45[1]
                       + f67[0] * f67[0] + f67[1] * f67[1];
        norm_sq[i] = r4d_wave_sum(local_sq);
    }

    // Step 1: Find Anchor Key index i*
    int i_star = 0;
    float max_n = norm_sq[0];
    if (norm_sq[1] > max_n) { max_n = norm_sq[1]; i_star = 1; }
    if (norm_sq[2] > max_n) { max_n = norm_sq[2]; i_star = 2; }
    if (norm_sq[3] > max_n) { max_n = norm_sq[3]; i_star = 3; }

    // Store bit-exact Anchor Key directly to output
    *reinterpret_cast<uint2*>(out_k + d_offset) = k_raw[i_star];

    // Step 2: Compute norm-weighted Value mixture
    float s[4];
    float sum_s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        s[i] = sqrtf(fmaxf(norm_sq[i], 0.0f)) + 1e-6f;
        sum_s += s[i];
    }
    const float inv_sum = 1.0f / sum_s;
    const float w[4] = { s[0] * inv_sum, s[1] * inv_sum, s[2] * inv_sum, s[3] * inv_sum };

    // Unpack 4 candidate values and blend
    float v_blended[8] = {0.0f};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint2 v_raw = *reinterpret_cast<const uint2*>(in_v[i] + d_offset);
        const auto v01 = __builtin_amdgcn_cvt_pk_f32_fp8((int)v_raw.x, false);
        const auto v23 = __builtin_amdgcn_cvt_pk_f32_fp8((int)v_raw.x, true);
        const auto v45 = __builtin_amdgcn_cvt_pk_f32_fp8((int)v_raw.y, false);
        const auto v67 = __builtin_amdgcn_cvt_pk_f32_fp8((int)v_raw.y, true);

        v_blended[0] += w[i] * v01[0];
        v_blended[1] += w[i] * v01[1];
        v_blended[2] += w[i] * v23[0];
        v_blended[3] += w[i] * v23[1];
        v_blended[4] += w[i] * v45[0];
        v_blended[5] += w[i] * v45[1];
        v_blended[6] += w[i] * v67[0];
        v_blended[7] += w[i] * v67[1];
    }

    // Pack blended floats back to FP8 (E4M3) via native gfx1201 instructions
    const int p01 = __builtin_amdgcn_cvt_pk_fp8_f32(v_blended[0], v_blended[1], 0, false);
    const int p23 = __builtin_amdgcn_cvt_pk_fp8_f32(v_blended[2], v_blended[3], 0, false);
    const int p45 = __builtin_amdgcn_cvt_pk_fp8_f32(v_blended[4], v_blended[5], 0, false);
    const int p67 = __builtin_amdgcn_cvt_pk_fp8_f32(v_blended[6], v_blended[7], 0, false);

    const uint32_t lo_word = ((uint32_t)(p01 & 0xFFFF)) | (((uint32_t)(p23 & 0xFFFF)) << 16);
    const uint32_t hi_word = ((uint32_t)(p45 & 0xFFFF)) | (((uint32_t)(p67 & 0xFFFF)) << 16);

    *reinterpret_cast<uint2*>(out_v + d_offset) = make_uint2(lo_word, hi_word);
}

extern "C" int r4d_kv_anchor_pool4(
    uint8_t* kv,
    const int32_t* src_blocks,
    const int32_t* dst_blocks,
    int64_t kv_block_stride,
    int64_t kv_head_stride,
    int n_chunks,
    int n_heads,
    hipStream_t stream
) {
    if (!kv || !src_blocks || !dst_blocks || n_chunks <= 0 || n_heads <= 0) return 0;
    dim3 grid(16, n_chunks, n_heads);
    dim3 block(32, 1, 1);
    hipLaunchKernelGGL(r4d_kv_anchor_pool4_kernel, grid, block, 0, stream,
                       kv, src_blocks, dst_blocks, kv_block_stride, kv_head_stride, n_chunks);
    return (int)hipGetLastError();
}
