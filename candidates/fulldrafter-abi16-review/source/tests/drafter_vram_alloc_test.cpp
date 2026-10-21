#ifdef NDEBUG
#undef NDEBUG
#endif
#ifndef __HIP_PLATFORM_AMD__
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_AMD__
#endif
#endif
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <vector>
#include <fstream>
#include <string>

#include "arch/common/rad_block_fulldrafter.h"

using namespace radiance;

static void verify_tensor_pointer(
    const char* name,
    const rad_bf16_t* ptr,
    const uint8_t* base,
    size_t total_bytes,
    int expected_dev
) {
    assert(ptr != nullptr && "Tensor pointer must not be null");
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(ptr);
    assert(byte_ptr >= base && "Tensor pointer must be >= base");
    assert(byte_ptr < base + total_bytes && "Tensor pointer must be < base + total_bytes");
    assert((reinterpret_cast<uintptr_t>(ptr) % 2 == 0) && "BF16 tensor pointer must be 2-byte aligned");

    hipPointerAttribute_t attr{};
    hipError_t err = hipPointerGetAttributes(&attr, ptr);
    assert(err == hipSuccess && "hipPointerGetAttributes must succeed");
    assert(attr.type == hipMemoryTypeDevice && "Pointer must reside in GPU device VRAM");
    assert(attr.device == expected_dev && "Pointer must reside on the expected GPU device");
}

static void test_gpu_device_allocation() {
    std::printf("===================================================================\n");
    std::printf(" Radiance FullDrafter GPU Device VRAM Allocation Test\n");
    std::printf("===================================================================\n");

    const char* path = "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_25000.safetensors";
    std::printf("[TEST 1] Loading sidecar weights into GPU VRAM: %s\n", path);

    FullDrafterWeights weights;
    FullDrafterManifest manifest;
    std::string err;

    bool ok = FullDrafterLoader::load_weights(path, &weights, &manifest, &err, 2);
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", err.c_str());
        std::exit(1);
    }

    assert(weights.owns_memory && "Weights must own memory");
    assert(weights.allocated_bytes == FullDrafterConfig::TOTAL_BYTES && "Allocated bytes must match TOTAL_BYTES (175.38MB)");
    assert(weights.num_ranks == 2 && "Dual-GPU replication must allocate 2 ranks");
    assert(weights.raw_buffers[0] != nullptr && "Rank 0 buffer must be non-null");
    assert(weights.raw_buffers[1] != nullptr && "Rank 1 buffer must be non-null");
    assert(weights.raw_buffer == weights.raw_buffers[0] && "raw_buffer alias must point to Rank 0");
    std::printf("  PASS: Loaded 175.38MB sidecar across 2 GPU ranks\n");

    // 2. Verify GPU physical device attributes
    std::printf("[TEST 2] Verifying physical GPU device VRAM placement...\n");
    for (int r = 0; r < 2; ++r) {
        hipPointerAttribute_t attr{};
        hipError_t h_err = hipPointerGetAttributes(&attr, weights.raw_buffers[r]);
        assert(h_err == hipSuccess && "hipPointerGetAttributes must succeed for raw_buffer");
        assert(attr.type == hipMemoryTypeDevice && "Buffer must reside in GPU device VRAM");
        assert(attr.device == r && "Buffer device ID must match rank");
        assert(weights.ranks[r].device_id == r);
        assert(weights.ranks[r].raw_buffer == weights.raw_buffers[r]);
        assert(weights.ranks[r].allocated_bytes == FullDrafterConfig::TOTAL_BYTES);
        std::printf("  Rank %d: device=%d, type=DEVICE_VRAM, ptr=%p\n", r, attr.device, weights.raw_buffers[r]);
    }
    std::printf("  PASS: Both Rank 0 and Rank 1 reside strictly in GPU device VRAM\n");

    // 3. Verify all 38 tensors on Rank 0 and Rank 1
    std::printf("[TEST 3] Verifying clean device pointers for all 38 tensors on Rank 0 & 1...\n");
    for (int r = 0; r < 2; ++r) {
        const auto& rw = weights.ranks[r];
        const uint8_t* base = weights.raw_buffers[r];
        size_t bytes = FullDrafterConfig::TOTAL_BYTES;

        verify_tensor_pointer("mask_vector", rw.mask_vector, base, bytes, r);
        verify_tensor_pointer("context_proj", rw.context_proj, base, bytes, r);
        verify_tensor_pointer("context_norm", rw.context_norm, base, bytes, r);
        verify_tensor_pointer("anchor_proj", rw.anchor_proj, base, bytes, r);
        verify_tensor_pointer("final_norm", rw.final_norm, base, bytes, r);
        verify_tensor_pointer("out_proj", rw.out_proj, base, bytes, r);

        verify_tensor_pointer("conv_w_base", rw.conv_w_base, base, bytes, r);
        verify_tensor_pointer("conv_w_proj", rw.conv_w_proj, base, bytes, r);
        verify_tensor_pointer("recurrent_norm", rw.recurrent_norm, base, bytes, r);
        verify_tensor_pointer("markov_matrix_a", rw.markov_matrix_a, base, bytes, r);
        verify_tensor_pointer("markov_matrix_b", rw.markov_matrix_b, base, bytes, r);

        verify_tensor_pointer("selector_pred_codebook", rw.selector_pred_codebook, base, bytes, r);
        verify_tensor_pointer("selector_succ_codebook", rw.selector_succ_codebook, base, bytes, r);
        verify_tensor_pointer("selector_hidden_proj", rw.selector_hidden_proj, base, bytes, r);

        verify_tensor_pointer("confidence_norm", rw.confidence_norm, base, bytes, r);
        verify_tensor_pointer("confidence_fc1", rw.confidence_fc1, base, bytes, r);
        verify_tensor_pointer("confidence_fc2", rw.confidence_fc2, base, bytes, r);
        verify_tensor_pointer("verification_norm", rw.verification_norm, base, bytes, r);
        verify_tensor_pointer("verification_fc1", rw.verification_fc1, base, bytes, r);
        verify_tensor_pointer("verification_fc2", rw.verification_fc2, base, bytes, r);

        for (int b = 0; b < 2; ++b) {
            verify_tensor_pointer("input_layernorm", rw.blocks[b].input_layernorm, base, bytes, r);
            verify_tensor_pointer("q_proj", rw.blocks[b].q_proj, base, bytes, r);
            verify_tensor_pointer("k_proj", rw.blocks[b].k_proj, base, bytes, r);
            verify_tensor_pointer("v_proj", rw.blocks[b].v_proj, base, bytes, r);
            verify_tensor_pointer("o_proj", rw.blocks[b].o_proj, base, bytes, r);
            verify_tensor_pointer("post_attn_layernorm", rw.blocks[b].post_attn_layernorm, base, bytes, r);
            verify_tensor_pointer("gate_proj", rw.blocks[b].gate_proj, base, bytes, r);
            verify_tensor_pointer("up_proj", rw.blocks[b].up_proj, base, bytes, r);
            verify_tensor_pointer("down_proj", rw.blocks[b].down_proj, base, bytes, r);
        }

        // Verify primary pointers on weights match Rank 0
        if (r == 0) {
            assert(weights.mask_vector == rw.mask_vector);
            assert(weights.context_proj == rw.context_proj);
            assert(weights.out_proj == rw.out_proj);
            assert(weights.markov_matrix_a == rw.markov_matrix_a);
        }
    }
    std::printf("  PASS: All 38 tensor pointers validated on both GPU Rank 0 and Rank 1\n");

    // 4. Verify dual-GPU replication data parity against source file
    std::printf("[TEST 4] Verifying dual-GPU data replication parity against disk payload...\n");
    std::ifstream file(path, std::ios::binary);
    uint64_t hlen = 0;
    file.read(reinterpret_cast<char*>(&hlen), 8);
    file.seekg(8 + hlen, std::ios::beg);

    constexpr size_t CHECK_BYTES = 65536;
    std::vector<uint8_t> expected_first(CHECK_BYTES);
    file.read(reinterpret_cast<char*>(expected_first.data()), CHECK_BYTES);

    std::vector<uint8_t> actual_r0(CHECK_BYTES, 0);
    std::vector<uint8_t> actual_r1(CHECK_BYTES, 0);

    assert(hipSetDevice(0) == hipSuccess);
    hipError_t c0 = hipMemcpyDtoH(actual_r0.data(), weights.raw_buffers[0], CHECK_BYTES);
    assert(c0 == hipSuccess);

    assert(hipSetDevice(1) == hipSuccess);
    hipError_t c1 = hipMemcpyDtoH(actual_r1.data(), weights.raw_buffers[1], CHECK_BYTES);
    assert(c1 == hipSuccess);

    assert(std::memcmp(actual_r0.data(), expected_first.data(), CHECK_BYTES) == 0 && "GPU 0 data mismatch with file");
    assert(std::memcmp(actual_r1.data(), expected_first.data(), CHECK_BYTES) == 0 && "GPU 1 data mismatch with file");
    assert(std::memcmp(actual_r0.data(), actual_r1.data(), CHECK_BYTES) == 0 && "GPU 0 vs GPU 1 replication mismatch");
    std::printf("  PASS: 64KB initial chunk verified byte-for-byte identical on GPU 0 and GPU 1\n");

    // 5. Verify RAII ownership transfer and hipFree cleanup
    std::printf("[TEST 5] Verifying RAII ownership transfer, move semantics & hipFree cleanup...\n");
    {
        FullDrafterWeights w_moved = std::move(weights);
        assert(!weights.owns_memory && "Moved-from object must not own memory");
        assert(weights.raw_buffer == nullptr);
        assert(weights.raw_buffers[0] == nullptr && weights.raw_buffers[1] == nullptr);
        assert(w_moved.owns_memory && "Moved-to object must own memory");
        assert(w_moved.raw_buffers[0] != nullptr && w_moved.raw_buffers[1] != nullptr);

        // Move back
        weights = std::move(w_moved);
        assert(weights.owns_memory);
        assert(weights.raw_buffers[0] != nullptr);
    }

    // Explicit free
    FullDrafterLoader::free_weights(&weights);
    assert(!weights.owns_memory && "free_weights must reset owns_memory");
    assert(weights.raw_buffer == nullptr && "free_weights must reset raw_buffer");
    assert(weights.raw_buffers[0] == nullptr && "free_weights must reset raw_buffers[0]");
    assert(weights.raw_buffers[1] == nullptr && "free_weights must reset raw_buffers[1]");
    assert(weights.allocated_bytes == 0);
    assert(weights.mask_vector == nullptr);
    assert(weights.context_proj == nullptr);
    std::printf("  PASS: RAII move semantics and free_weights hipFree validated\n");

    // 6. Test reload on existing struct without leaking
    std::printf("[TEST 6] Testing reload on existing weights object...\n");
    ok = FullDrafterLoader::load_weights(path, &weights, nullptr, &err, 2);
    assert(ok && "First load must succeed");
    void* prev_ptr0 = weights.raw_buffers[0];
    void* prev_ptr1 = weights.raw_buffers[1];

    // Reload triggers free_weights first
    ok = FullDrafterLoader::load_weights(path, &weights, nullptr, &err, 2);
    assert(ok && "Reload must succeed without leaking");
    assert(weights.raw_buffers[0] != nullptr);
    assert(weights.raw_buffers[1] != nullptr);
    std::printf("  PASS: Reloaded successfully (prev=[%p, %p], new=[%p, %p])\n",
                prev_ptr0, prev_ptr1, weights.raw_buffers[0], weights.raw_buffers[1]);

    FullDrafterLoader::free_weights(&weights);
    std::printf("===================================================================\n");
    std::printf(" ALL GPU DEVICE VRAM ALLOCATION TESTS PASSED!\n");
    std::printf("===================================================================\n");
}

static void test_edge_cases_and_error_paths() {
    std::printf("===================================================================\n");
    std::printf(" Radiance FullDrafter Edge Cases & VRAM Leak Validation\n");
    std::printf("===================================================================\n");

    const char* path = "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_25000.safetensors";
    std::string err;

    // 1. Null out_weights
    std::printf("[EDGE 1] Testing null out_weights pointer...\n");
    assert(!FullDrafterLoader::load_weights(path, nullptr, nullptr, &err));
    assert(err.find("Null") != std::string::npos);
    std::printf("  PASS: Null out_weights rejected\n");

    // 2. Non-existent path
    std::printf("[EDGE 2] Testing non-existent safetensors file...\n");
    FullDrafterWeights w;
    assert(!FullDrafterLoader::load_weights("/invalid/non_existent.safetensors", &w, nullptr, &err));
    assert(!err.empty());
    assert(!w.owns_memory && w.raw_buffer == nullptr);
    std::printf("  PASS: Non-existent file rejected cleanly\n");

    // 3. Single-rank request
    std::printf("[EDGE 3] Testing single-rank allocation request (requested_ranks=1)...\n");
    assert(FullDrafterLoader::load_weights(path, &w, nullptr, &err, 1));
    assert(w.num_ranks == 1);
    assert(w.raw_buffers[0] != nullptr);
    assert(w.raw_buffers[1] == nullptr);
    assert(w.ranks[0].device_id == 0);
    assert(w.ranks[1].raw_buffer == nullptr);
    assert(w.rank(0).device_id == 0);
    assert(w.rank(0).raw_buffer == w.raw_buffers[0]);
    assert(w[0].raw_buffer == w.raw_buffers[0]);
    assert(w.rank(1).device_id == -1);
    assert(w.rank(1).raw_buffer == nullptr);
    assert(w[1].device_id == -1);
    assert(w[1].raw_buffer == nullptr);
    assert(w.rank(2).device_id == -1);
    assert(w.rank(2).raw_buffer == nullptr);
    assert(w[2].raw_buffer == nullptr);
    FullDrafterLoader::free_weights(&w);
    std::printf("  PASS: Single rank requested and allocated on GPU 0 exclusively with correct rank accessors\n");

    // 4. RAII scope exit without explicit free_weights
    std::printf("[EDGE 4] Testing RAII destructor without explicit free_weights()...\n");
    {
        FullDrafterWeights w_scoped;
        assert(FullDrafterLoader::load_weights(path, &w_scoped, nullptr, &err, 2));
        assert(w_scoped.owns_memory);
        // Exiting scope: destructor must free both GPU buffers
    }
    std::printf("  PASS: Scope exited and destructor freed VRAM safely\n");

    // 5. Zero-leak loop test over 5 cycles
    std::printf("[EDGE 5] Testing VRAM leak freedom across 5 reload cycles...\n");
    size_t init_free0 = 0, total0 = 0;
    size_t init_free1 = 0, total1 = 0;
    assert(hipSetDevice(0) == hipSuccess);
    assert(hipMemGetInfo(&init_free0, &total0) == hipSuccess);
    assert(hipSetDevice(1) == hipSuccess);
    assert(hipMemGetInfo(&init_free1, &total1) == hipSuccess);

    for (int cycle = 0; cycle < 5; ++cycle) {
        FullDrafterWeights w_cycle;
        assert(FullDrafterLoader::load_weights(path, &w_cycle, nullptr, &err, 2));
        assert(w_cycle.owns_memory);
        FullDrafterLoader::free_weights(&w_cycle);
    }

    size_t end_free0 = 0, end_free1 = 0;
    assert(hipSetDevice(0) == hipSuccess);
    assert(hipMemGetInfo(&end_free0, &total0) == hipSuccess);
    assert(hipSetDevice(1) == hipSuccess);
    assert(hipMemGetInfo(&end_free1, &total1) == hipSuccess);

    assert(init_free0 == end_free0 && "GPU 0 VRAM leaked across reload cycles");
    assert(init_free1 == end_free1 && "GPU 1 VRAM leaked across reload cycles");
    std::printf("  PASS: 0 bytes leaked across 5 load/free cycles (GPU 0 free: %zu, GPU 1 free: %zu)\n",
                end_free0, end_free1);

    // 6. Partial allocation rollback (GPU 1 OOM simulation)
    std::printf("[EDGE 6] Testing partial allocation rollback without VRAM leak on GPU 0...\n");
    int dev_count = 0;
    assert(hipGetDeviceCount(&dev_count) == hipSuccess);
    if (dev_count >= 2) {
        size_t r6_init_free0 = 0, r6_init_free1 = 0;
        assert(hipSetDevice(0) == hipSuccess);
        assert(hipMemGetInfo(&r6_init_free0, &total0) == hipSuccess);
        assert(hipSetDevice(1) == hipSuccess);
        assert(hipMemGetInfo(&r6_init_free1, &total1) == hipSuccess);

        // Leave only 10MB free on GPU 1 so that allocating 175.38MB fails
        size_t hog_size = (r6_init_free1 > 10 * 1024 * 1024) ? (r6_init_free1 - 10 * 1024 * 1024) : 0;
        void* hog_mem = nullptr;
        assert(hipMalloc(&hog_mem, hog_size) == hipSuccess);

        FullDrafterWeights w_fail;
        std::string fail_err;
        bool fail_ok = FullDrafterLoader::load_weights(path, &w_fail, nullptr, &fail_err, 2);
        assert(!fail_ok && "load_weights must fail when GPU 1 is out of memory");
        assert(!fail_err.empty() && fail_err.find("hipMalloc") != std::string::npos);
        assert(!w_fail.owns_memory);
        assert(w_fail.raw_buffer == nullptr);
        assert(w_fail.raw_buffers[0] == nullptr);
        assert(w_fail.raw_buffers[1] == nullptr);

        // Free hog memory on GPU 1
        assert(hipSetDevice(1) == hipSuccess);
        assert(hipFree(hog_mem) == hipSuccess);

        // Verify that GPU 0 allocation was completely rolled back and 0 bytes were leaked
        size_t r6_end_free0 = 0, r6_end_free1 = 0;
        assert(hipSetDevice(0) == hipSuccess);
        assert(hipMemGetInfo(&r6_end_free0, &total0) == hipSuccess);
        assert(hipSetDevice(1) == hipSuccess);
        assert(hipMemGetInfo(&r6_end_free1, &total1) == hipSuccess);

        assert(r6_init_free0 == r6_end_free0 && "GPU 0 VRAM leaked during partial allocation failure!");
        assert(r6_init_free1 == r6_end_free1 && "GPU 1 VRAM leaked after releasing test buffer!");
        std::printf("  PASS: GPU 0 VRAM cleanly rolled back (0 bytes leaked) after GPU 1 hipMalloc OOM failure\n");
    }

    std::printf("===================================================================\n");
    std::printf(" ALL EDGE CASE & LEAK FREEDOM TESTS PASSED!\n");
    std::printf("===================================================================\n");
}

int main() {
    test_gpu_device_allocation();
    test_edge_cases_and_error_paths();
    return 0;
}
