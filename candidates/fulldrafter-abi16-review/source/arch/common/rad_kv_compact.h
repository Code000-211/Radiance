#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_runtime.h>
#endif

#if __has_include("abi/rad_device.h")
#include "abi/rad_device.h"
#elif __has_include("../abi/rad_device.h")
#include "../abi/rad_device.h"
#elif __has_include("../../abi/rad_device.h")
#include "../../abi/rad_device.h"
#else
extern "C" int rad_memcpy_async(void* dst, const void* src, int64_t bytes, void* s);
#endif

namespace radiance {

#define RAD_KV_COMPACT_MAX_MOVES 16

/**
 * RadKVSlotMove: A single physical slot copy operation.
 * Hazard-free guarantee: when executed in ascending depth order k = 1, ..., acc,
 * target index k is strictly less than all future source indices n_j (j > k).
 */
struct RadKVSlotMove {
    int32_t src_slot;
    int32_t dst_slot;
};

/**
 * RadKVCompactParams: Compact POD parameter structure passed to device kernel
 * or host dispatcher by value (zero host-device memory allocation overhead).
 */
struct RadKVCompactParams {
    void*         kv_pool_base;       // Base pointer to paged KV pool on local rank
    int64_t       layer_stride;       // Byte stride between attention layers
    int64_t       kv_block_stride;    // Byte stride per physical block
    int64_t       kv_head_stride;     // Byte stride per KV head within a block
    int32_t       block_size;         // Tokens per block (4 or 16)
    int32_t       tok_bytes;          // Byte size of (K, V) per token per head (2 * HEAD_DIM * sizeof(dtype))
    int32_t       n_layers;           // Number of full-attention layers
    int32_t       n_heads;            // Number of KV heads on this rank
    int32_t       n_moves;            // Number of slot relocations to execute
    RadKVSlotMove moves[RAD_KV_COMPACT_MAX_MOVES];
};

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
/**
 * rad_kv_compact_winning_branch_kernel:
 * In-place GPU slot compaction kernel for AMD RDNA4 (gfx1201).
 *
 * Parallelization geometry:
 * - Grid: (n_heads, n_layers)
 * - Block: 32 threads (1 wave on gfx1201)
 *
 * Uses 128-bit vectorized (uint4) transfers for peak GDDR6 memory bandwidth.
 * Moves are evaluated sequentially in ascending depth order k = 1..acc, ensuring
 * strictly hazard-free execution without scratchpad buffers.
 */
__global__ __launch_bounds__(32) void rad_kv_compact_winning_branch_kernel(
    RadKVCompactParams p)
{
    const int h = blockIdx.x; // KV head index on this rank
    const int l = blockIdx.y; // Layer index
    const int lane = threadIdx.x; // 0..31

    if (h >= p.n_heads || l >= p.n_layers) return;

    uint8_t* const layer_base = reinterpret_cast<uint8_t*>(p.kv_pool_base) + (size_t)l * p.layer_stride;

    #pragma nounroll
    for (int m = 0; m < p.n_moves; ++m) {
        const int32_t src = p.moves[m].src_slot;
        const int32_t dst = p.moves[m].dst_slot;

        if (src == dst || src < 0 || dst < 0) continue;

        const size_t src_blk = static_cast<size_t>(src / p.block_size);
        const size_t src_off = static_cast<size_t>(src % p.block_size);
        const size_t dst_blk = static_cast<size_t>(dst / p.block_size);
        const size_t dst_off = static_cast<size_t>(dst % p.block_size);

        const size_t src_byte_off = src_blk * static_cast<size_t>(p.kv_block_stride)
                                  + static_cast<size_t>(h) * static_cast<size_t>(p.kv_head_stride)
                                  + src_off * static_cast<size_t>(p.tok_bytes);

        const size_t dst_byte_off = dst_blk * static_cast<size_t>(p.kv_block_stride)
                                  + static_cast<size_t>(h) * static_cast<size_t>(p.kv_head_stride)
                                  + dst_off * static_cast<size_t>(p.tok_bytes);

        const uint8_t* const s_ptr = layer_base + src_byte_off;
        uint8_t* const d_ptr       = layer_base + dst_byte_off;

        // Opaque rows can include metadata and need not have 16-byte-aligned token strides.
        if (((reinterpret_cast<uintptr_t>(s_ptr) | reinterpret_cast<uintptr_t>(d_ptr)) & 15u) == 0) {
            const int vector_bytes = p.tok_bytes & ~15;
            for (int b = lane * 16; b < vector_bytes; b += 32 * 16)
                *reinterpret_cast<uint4*>(d_ptr + b) = *reinterpret_cast<const uint4*>(s_ptr + b);
            for (int b = vector_bytes + lane; b < p.tok_bytes; b += 32)
                d_ptr[b] = s_ptr[b];
        } else {
            for (int b = lane; b < p.tok_bytes; b += 32)
                d_ptr[b] = s_ptr[b];
        }
        // A later move may read a slot this move just wrote, including another lane's bytes.
        if (m + 1 < p.n_moves) __syncthreads();
    }
}
#endif

/**
 * Compares winning node indices n_k against target contiguous slot positions k
 * for k = 1, ..., acc and builds the hazard-free move list in ascending order.
 *
 * Fast-path check: returns 0 if n_k == k for all k (e.g. root-only or linear path).
 */
inline int32_t rad_kv_compact_build_moves(
    const int32_t* best_nodes,
    int32_t acc,
    int64_t ctx_len,
    const std::vector<int32_t>& block_table,
    int64_t first_pos,
    int32_t block_size,
    RadKVSlotMove* out_moves,
    int32_t max_moves)
{
    if (!best_nodes || acc <= 0 || !out_moves || max_moves <= 0 || block_size <= 0) {
        return 0;
    }

    int32_t n_moves = 0;
    for (int32_t k = 1; k <= acc && n_moves < max_moves; ++k) {
        const int32_t node_id = best_nodes[k];
        if (node_id == k) {
            // Already in contiguous slot: no relocation needed
            continue;
        }

        const int64_t target_pos = ctx_len + k;
        const int64_t source_pos = ctx_len + node_id;

        const int64_t bi_dst = (target_pos - first_pos) / block_size;
        const int64_t bi_src = (source_pos - first_pos) / block_size;

        if (bi_dst >= 0 && bi_dst < static_cast<int64_t>(block_table.size()) &&
            bi_src >= 0 && bi_src < static_cast<int64_t>(block_table.size())) {
            out_moves[n_moves].dst_slot = block_table[static_cast<size_t>(bi_dst)] * block_size +
                                          static_cast<int32_t>(target_pos % block_size);
            out_moves[n_moves].src_slot = block_table[static_cast<size_t>(bi_src)] * block_size +
                                          static_cast<int32_t>(source_pos % block_size);
            n_moves++;
        }
    }
    return n_moves;
}

/**
 * rad_kv_compact_winning_branch:
 * Standalone wrapper for header-only regressions. The engine uses the runtime-selected
 * core/device entry, not this compile-time dispatch.
 *
 * No moves or identity moves bypass dispatch. HIP translation units launch the kernel;
 * ordinary C++ uses stream-ordered copies, or CPU copies when stream is null.
 */
static inline int rad_kv_compact_winning_branch(
    void* kv_pool_base,
    int64_t layer_stride,
    int64_t kv_block_stride,
    int64_t kv_head_stride,
    int32_t block_size,
    int32_t tok_bytes,
    int32_t n_layers,
    int32_t n_heads,
    const RadKVSlotMove* moves,
    int32_t n_moves,
    void* stream = nullptr)
{
    // Host fast-path: 0 moves needed
    if (n_moves <= 0 || !moves || !kv_pool_base || block_size <= 0 || tok_bytes <= 0) {
        return 0;
    }

    bool has_active_moves = false;
    for (int i = 0; i < n_moves; ++i) {
        if (moves[i].src_slot != moves[i].dst_slot && moves[i].src_slot >= 0 && moves[i].dst_slot >= 0) {
            has_active_moves = true;
            break;
        }
    }
    if (!has_active_moves) {
        return 0;
    }

    RadKVCompactParams params{};
    params.kv_pool_base    = kv_pool_base;
    params.layer_stride    = layer_stride;
    params.kv_block_stride = kv_block_stride;
    params.kv_head_stride  = kv_head_stride;
    params.block_size      = block_size;
    params.tok_bytes       = tok_bytes;
    params.n_layers        = n_layers;
    params.n_heads         = std::max(1, n_heads);
    params.n_moves         = std::min(n_moves, RAD_KV_COMPACT_MAX_MOVES);

    for (int i = 0; i < params.n_moves; ++i) {
        params.moves[i] = moves[i];
    }

#if defined(__HIPCC__)
    dim3 grid(static_cast<uint32_t>(params.n_heads), static_cast<uint32_t>(params.n_layers));
    dim3 block(32);
    hipLaunchKernelGGL(
        rad_kv_compact_winning_branch_kernel,
        grid,
        block,
        0,
        reinterpret_cast<hipStream_t>(stream),
        params
    );
    // A successful enqueue is required before the accepted branch may be committed.
    return hipGetLastError() == hipSuccess ? 0 : -1;
#else
    if (stream != nullptr) {
        // Asynchronous stream-ordered relocations via 128-bit vectorized device blits
        for (int l = 0; l < params.n_layers; ++l) {
            uint8_t* const layer_base = reinterpret_cast<uint8_t*>(params.kv_pool_base) + (size_t)l * params.layer_stride;
            for (int h = 0; h < params.n_heads; ++h) {
                for (int m = 0; m < params.n_moves; ++m) {
                    const int32_t src = params.moves[m].src_slot;
                    const int32_t dst = params.moves[m].dst_slot;
                    if (src == dst || src < 0 || dst < 0) continue;

                    const size_t src_blk = static_cast<size_t>(src / params.block_size);
                    const size_t src_off = static_cast<size_t>(src % params.block_size);
                    const size_t dst_blk = static_cast<size_t>(dst / params.block_size);
                    const size_t dst_off = static_cast<size_t>(dst % params.block_size);

                    const size_t src_byte_off = src_blk * static_cast<size_t>(params.kv_block_stride)
                                              + static_cast<size_t>(h) * static_cast<size_t>(params.kv_head_stride)
                                              + src_off * static_cast<size_t>(params.tok_bytes);

                    const size_t dst_byte_off = dst_blk * static_cast<size_t>(params.kv_block_stride)
                                              + static_cast<size_t>(h) * static_cast<size_t>(params.kv_head_stride)
                                              + dst_off * static_cast<size_t>(params.tok_bytes);

                    const int status = rad_memcpy_async(
                        layer_base + dst_byte_off,
                        layer_base + src_byte_off,
                        params.tok_bytes,
                        reinterpret_cast<RadStream>(stream)
                    );
                    if (status != 0) return status;
                }
            }
        }
        return 0;
    } else {
        // Host CPU execution with 128-bit vectorized transfers (for unit tests / CPU mock)
        for (int l = 0; l < params.n_layers; ++l) {
            uint8_t* const layer_base = reinterpret_cast<uint8_t*>(params.kv_pool_base) + (size_t)l * params.layer_stride;
            for (int h = 0; h < params.n_heads; ++h) {
                for (int m = 0; m < params.n_moves; ++m) {
                    const int32_t src = params.moves[m].src_slot;
                    const int32_t dst = params.moves[m].dst_slot;
                    if (src == dst || src < 0 || dst < 0) continue;

                    const size_t src_blk = static_cast<size_t>(src / params.block_size);
                    const size_t src_off = static_cast<size_t>(src % params.block_size);
                    const size_t dst_blk = static_cast<size_t>(dst / params.block_size);
                    const size_t dst_off = static_cast<size_t>(dst % params.block_size);

                    const size_t src_byte_off = src_blk * static_cast<size_t>(params.kv_block_stride)
                                              + static_cast<size_t>(h) * static_cast<size_t>(params.kv_head_stride)
                                              + src_off * static_cast<size_t>(params.tok_bytes);

                    const size_t dst_byte_off = dst_blk * static_cast<size_t>(params.kv_block_stride)
                                              + static_cast<size_t>(h) * static_cast<size_t>(params.kv_head_stride)
                                              + dst_off * static_cast<size_t>(params.tok_bytes);

                    uint8_t* const d_ptr = layer_base + dst_byte_off;
                    const uint8_t* const s_ptr = layer_base + src_byte_off;

                    // memcpy permits unaligned packed rows without typed-load aliasing.
                    std::memcpy(d_ptr, s_ptr, static_cast<size_t>(params.tok_bytes));
                }
            }
        }
        return 0;
    }
#endif
}

} // namespace radiance
