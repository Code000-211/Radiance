#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#elif __has_include("vendor/nlohmann/json.hpp")
#include "vendor/nlohmann/json.hpp"
#elif __has_include("../../vendor/nlohmann/json.hpp")
#include "../../vendor/nlohmann/json.hpp"
#endif

#if __has_include("arch/common/rad_tree_candidate.h")
#include "arch/common/rad_tree_candidate.h"
#else
#include "rad_tree_candidate.h"
#endif

#if __has_include("libr4d/r4d_fulldrafter.h")
#include "libr4d/r4d_fulldrafter.h"
#elif __has_include("../libr4d/r4d_fulldrafter.h")
#include "../libr4d/r4d_fulldrafter.h"
#elif __has_include("../../libr4d/r4d_fulldrafter.h")
#include "../../libr4d/r4d_fulldrafter.h"
#endif

#if __has_include(<hip/hip_runtime.h>)
#ifndef __HIP_PLATFORM_AMD__
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_AMD__
#endif
#endif
#include <hip/hip_runtime.h>
#else
typedef int hipError_t;
enum { hipSuccess = 0 };
enum hipMemoryType { hipMemoryTypeHost = 0, hipMemoryTypeDevice = 1 };
struct hipPointerAttribute_t { hipMemoryType type = hipMemoryTypeHost; int device = 0; };
inline const char* hipGetErrorString(hipError_t) { return "hip_stub"; }
inline hipError_t hipGetDevice(int* d) { if (d) *d = 0; return 0; }
inline hipError_t hipSetDevice(int) { return 0; }
inline hipError_t hipGetDeviceCount(int* c) { if (c) *c = 0; return 0; }
inline hipError_t hipDeviceCanAccessPeer(int* can, int, int) { if (can) *can = 0; return 0; }
inline hipError_t hipDeviceEnablePeerAccess(int, unsigned int) { return 0; }
inline hipError_t hipMalloc(void** p, size_t) { if (p) *p = nullptr; return 0; }
inline hipError_t hipFree(void*) { return 0; }
inline hipError_t hipMemset(void*, int, size_t) { return 0; }
inline hipError_t hipMemcpyHtoD(void*, const void*, size_t) { return 0; }
inline hipError_t hipMemcpyDtoH(void*, const void*, size_t) { return 0; }
enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
};
inline hipError_t hipMemcpy(void*, const void*, size_t, int = 0) { return 0; }
inline hipError_t hipDeviceSynchronize() { return 0; }
inline hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*) { return 0; }
inline hipError_t hipGetLastError() { return 0; }
#endif

namespace radiance {

using rad_bf16_t = uint16_t;

// Architectural hyperparameters for FullDrafter (K=12, DLoop 2x6, Qwen3.8-Flash-Next)
struct FullDrafterConfig {
    static constexpr size_t HIDDEN_SIZE          = 768;
    static constexpr size_t INTERMEDIATE_SIZE    = 2304; // 3 * 768 (SwiGLU)
    static constexpr size_t NUM_BLOCKS           = 2;
    static constexpr size_t NUM_Q_HEADS          = 12;
    static constexpr size_t NUM_KV_HEADS         = 4;    // 3:1 GQA
    static constexpr size_t HEAD_DIM             = 64;   // 12 * 64 = 768
    static constexpr size_t TARGET_WIDTH         = 2560; // Teacher hidden dim
    static constexpr size_t CONTEXT_DIM          = 7680; // 3 * 2560 (Layers 1, 23, mixer)
    static constexpr size_t BLOCK_K              = 6;    // Proposals per DLoop stage
    static constexpr size_t STAGES               = 2;    // DLoop recurrent stages
    static constexpr size_t TOTAL_K              = 12;   // 2 * 6 = 12 proposals
    static constexpr size_t VOCAB_SIZE           = 248320;
    static constexpr size_t ACTIVE_VOCAB         = 248077;
    static constexpr size_t TP_SHARD_VOCAB       = 124160; // V / 2 columns per GPU under TP=2
    static constexpr size_t MARKOV_RANK          = 64;
    static constexpr size_t SELECTOR_RANK        = 64;
    static constexpr size_t TOP_K                = 16;
    static constexpr size_t CONV_KERNEL_SIZE     = 2;
    static constexpr size_t CONV_GROUP_SIZE      = 16;
    static constexpr size_t CONV_NUM_GROUPS      = 48;   // 768 / 16 = 48
    static constexpr float  RMS_NORM_EPS         = 1e-6f;
    static constexpr float  ROPE_THETA           = 10000000.0f;

    static constexpr size_t TOTAL_PARAMS         = 87688704;
    static constexpr size_t TOTAL_BYTES          = 175377408; // 87,688,704 * 2 (BF16)
    static constexpr size_t ALODLM_TOTAL_PARAMS  = 87491845;
    static constexpr size_t ALODLM_TOTAL_BYTES   = 174983690; // 87,491,845 * 2 (BF16)
};

// Per-block Transformer weights
struct FullDrafterBlockWeights {
    const rad_bf16_t* input_layernorm = nullptr;       // [768]
    const rad_bf16_t* q_proj = nullptr;                // [768, 768]
    const rad_bf16_t* k_proj = nullptr;                // [256, 768]
    const rad_bf16_t* v_proj = nullptr;                // [256, 768]
    const rad_bf16_t* o_proj = nullptr;                // [768, 768]
    const rad_bf16_t* post_attn_layernorm = nullptr;   // [768]
    const rad_bf16_t* gate_proj = nullptr;             // [2304, 768]
    const rad_bf16_t* up_proj = nullptr;               // [2304, 768]
    const rad_bf16_t* down_proj = nullptr;             // [768, 2304]
};

/**
 * FullDrafterRankWeights: Provides typed device pointers for all 38 tensors on a specific GPU rank.
 */
struct FullDrafterRankWeights {
    int      device_id = -1;
    uint8_t* raw_buffer = nullptr;
    size_t   allocated_bytes = 0;

    // 1. Drafter Backbone
    const rad_bf16_t* mask_vector = nullptr;       // [768]
    const rad_bf16_t* context_proj = nullptr;      // [768, 7680]
    const rad_bf16_t* context_norm = nullptr;      // [768]
    const rad_bf16_t* anchor_proj = nullptr;       // [768, 2560]
    FullDrafterBlockWeights blocks[2];
    const rad_bf16_t* final_norm = nullptr;        // [768]
    const rad_bf16_t* out_proj = nullptr;          // [2560, 768]

    // 2. DFlash2 Dynamic Convolution
    const rad_bf16_t* conv_w_base = nullptr;       // [2, 768]
    const rad_bf16_t* conv_w_proj = nullptr;       // [96, 768]

    // 3. Recurrent Transition
    const rad_bf16_t* recurrent_norm = nullptr;    // [768]

    // 4. Markov Bigram Correction
    const rad_bf16_t* markov_matrix_a = nullptr;   // [248320, 64]
    const rad_bf16_t* markov_matrix_b = nullptr;   // [248320, 64]

    // 5. Top-16 Selector
    const rad_bf16_t* selector_pred_codebook = nullptr; // [248320, 64]
    const rad_bf16_t* selector_succ_codebook = nullptr; // [248320, 64]
    const rad_bf16_t* selector_hidden_proj = nullptr;   // [64, 768]

    // 6. Auxiliary Heads
    const rad_bf16_t* confidence_norm = nullptr;   // [768]
    const rad_bf16_t* confidence_fc1 = nullptr;    // [256, 768]
    const rad_bf16_t* confidence_fc2 = nullptr;    // [1, 256]
    const rad_bf16_t* verification_norm = nullptr; // [768]
    const rad_bf16_t* verification_fc1 = nullptr;  // [256, 768]
    const rad_bf16_t* verification_fc2 = nullptr;  // [1, 256]
    const rad_bf16_t* exit_gate_net_weight = nullptr; // [1, 768] (ALoDLM-4)
    const rad_bf16_t* exit_gate_net_bias = nullptr;   // [1] (ALoDLM-4)
    const rad_bf16_t* exit_gate_depth_bias = nullptr; // [4] (ALoDLM-4)

    // 7. Quarantined Teacher Matrices (mmap or external reference, not owned by sidecar)
    const rad_bf16_t* teacher_embedding_table = nullptr; // [248320, 2560]
    const rad_bf16_t* teacher_lm_head = nullptr;         // [248320, 2560]

    // 8. Device Pointers & Host Mirrors
    const FullDrafterGpuPointers* d_gpu_pointers = nullptr;
    const rad_bf16_t* host_teacher_lm_head = nullptr;
    const rad_bf16_t* host_teacher_embedding_table = nullptr;

    // 9. Execution Geometry
    size_t lm_head_dim = FullDrafterConfig::TARGET_WIDTH;
    bool is_fused = false;
    bool tp2_p2p_active = false;
    bool is_alodlm_loop = false;
    uint32_t alodlm_max_depth = 4;
    float alodlm_tau = 0.4f;
    float alodlm_q = 0.5f;
    float alodlm_pos_penalty = 0.0f;

    void bind_tensor(const std::string& name, const rad_bf16_t* ptr) {
        if (name == "drafter.mask_vector") mask_vector = ptr;
        else if (name == "drafter.context_proj.weight") context_proj = ptr;
        else if (name == "drafter.context_norm.weight") context_norm = ptr;
        else if (name == "drafter.anchor_proj.weight") anchor_proj = ptr;
        else if (name == "drafter.blocks.0.input_layernorm.weight") blocks[0].input_layernorm = ptr;
        else if (name == "drafter.blocks.0.self_attn.q_proj.weight") blocks[0].q_proj = ptr;
        else if (name == "drafter.blocks.0.self_attn.k_proj.weight") blocks[0].k_proj = ptr;
        else if (name == "drafter.blocks.0.self_attn.v_proj.weight") blocks[0].v_proj = ptr;
        else if (name == "drafter.blocks.0.self_attn.o_proj.weight") blocks[0].o_proj = ptr;
        else if (name == "drafter.blocks.0.post_attention_layernorm.weight") blocks[0].post_attn_layernorm = ptr;
        else if (name == "drafter.blocks.0.mlp.gate_proj.weight") blocks[0].gate_proj = ptr;
        else if (name == "drafter.blocks.0.mlp.up_proj.weight") blocks[0].up_proj = ptr;
        else if (name == "drafter.blocks.0.mlp.down_proj.weight") blocks[0].down_proj = ptr;
        else if (name == "drafter.blocks.1.input_layernorm.weight") blocks[1].input_layernorm = ptr;
        else if (name == "drafter.blocks.1.self_attn.q_proj.weight") blocks[1].q_proj = ptr;
        else if (name == "drafter.blocks.1.self_attn.k_proj.weight") blocks[1].k_proj = ptr;
        else if (name == "drafter.blocks.1.self_attn.v_proj.weight") blocks[1].v_proj = ptr;
        else if (name == "drafter.blocks.1.self_attn.o_proj.weight") blocks[1].o_proj = ptr;
        else if (name == "drafter.blocks.1.post_attention_layernorm.weight") blocks[1].post_attn_layernorm = ptr;
        else if (name == "drafter.blocks.1.mlp.gate_proj.weight") blocks[1].gate_proj = ptr;
        else if (name == "drafter.blocks.1.mlp.up_proj.weight") blocks[1].up_proj = ptr;
        else if (name == "drafter.blocks.1.mlp.down_proj.weight") blocks[1].down_proj = ptr;
        else if (name == "drafter.final_norm.weight") final_norm = ptr;
        else if (name == "drafter.out_proj.weight") out_proj = ptr;
        else if (name == "conv.w_base") conv_w_base = ptr;
        else if (name == "conv.w_proj.weight") conv_w_proj = ptr;
        else if (name == "recurrent_norm.weight") recurrent_norm = ptr;
        else if (name == "markov.matrix_a.weight") markov_matrix_a = ptr;
        else if (name == "markov.matrix_b.weight") markov_matrix_b = ptr;
        else if (name == "selector.pred_codebook.weight") selector_pred_codebook = ptr;
        else if (name == "selector.succ_codebook.weight") selector_succ_codebook = ptr;
        else if (name == "selector.hidden_proj.weight") selector_hidden_proj = ptr;
        else if (name == "confidence.norm.weight") confidence_norm = ptr;
        else if (name == "confidence.fc1.weight") confidence_fc1 = ptr;
        else if (name == "confidence.fc2.weight") confidence_fc2 = ptr;
        else if (name == "verification.norm.weight") verification_norm = ptr;
        else if (name == "verification.fc1.weight") verification_fc1 = ptr;
        else if (name == "verification.fc2.weight") verification_fc2 = ptr;
        else if (name == "exit_gate.net.weight") exit_gate_net_weight = ptr;
        else if (name == "exit_gate.net.bias") exit_gate_net_bias = ptr;
        else if (name == "exit_gate.depth_bias") exit_gate_depth_bias = ptr;
        else if (name == "model.language_model.embed_tokens.weight" || name == "teacher_embedding_table") teacher_embedding_table = ptr;
        else if (name == "lm_head.weight" || name == "teacher_lm_head") teacher_lm_head = ptr;
    }

    void reset() {
        device_id = -1;
        raw_buffer = nullptr;
        allocated_bytes = 0;
        mask_vector = nullptr;
        context_proj = nullptr;
        context_norm = nullptr;
        anchor_proj = nullptr;
        blocks[0] = FullDrafterBlockWeights{};
        blocks[1] = FullDrafterBlockWeights{};
        final_norm = nullptr;
        out_proj = nullptr;
        conv_w_base = nullptr;
        conv_w_proj = nullptr;
        recurrent_norm = nullptr;
        markov_matrix_a = nullptr;
        markov_matrix_b = nullptr;
        selector_pred_codebook = nullptr;
        selector_succ_codebook = nullptr;
        selector_hidden_proj = nullptr;
        confidence_norm = nullptr;
        confidence_fc1 = nullptr;
        confidence_fc2 = nullptr;
        verification_norm = nullptr;
        verification_fc1 = nullptr;
        verification_fc2 = nullptr;
        exit_gate_net_weight = nullptr;
        exit_gate_net_bias = nullptr;
        exit_gate_depth_bias = nullptr;
        is_alodlm_loop = false;
        alodlm_max_depth = 4;
        alodlm_tau = 0.4f;
        alodlm_q = 0.5f;
        alodlm_pos_penalty = 0.0f;
        teacher_embedding_table = nullptr;
        teacher_lm_head = nullptr;
        d_gpu_pointers = nullptr;
        host_teacher_lm_head = nullptr;
        host_teacher_embedding_table = nullptr;
        lm_head_dim = FullDrafterConfig::TARGET_WIDTH;
        is_fused = false;
    }
};

/**
 * FullDrafterWeights: Manages contiguous GPU device VRAM allocation for FullDrafter weights
 * across dual R9700 GPUs (Rank 0 and Rank 1), providing typed device pointers for all 38 tensors
 * on each rank and enforcing strict RAII cleanup via hipFree.
 */
struct FullDrafterWeights : public FullDrafterRankWeights {
    static constexpr size_t MAX_RANKS = 2;

    bool     owns_memory = false;
    size_t   num_ranks = 0;
    uint8_t* raw_buffers[MAX_RANKS] = {nullptr, nullptr};
    FullDrafterRankWeights ranks[MAX_RANKS];

    void*    teacher_mmap_ptr = nullptr;
    size_t   teacher_mmap_len = 0;

    void*                   d_lm_head_shard0[MAX_RANKS] = {nullptr, nullptr};
    void*                   d_lm_head_shard1[MAX_RANKS] = {nullptr, nullptr};
    // q8 names retain the byte-pointer interface; storage is packed INT4.
    int8_t*                 d_lm_head_q8_shard0[MAX_RANKS] = {nullptr, nullptr};
    int8_t*                 d_lm_head_q8_shard1[MAX_RANKS] = {nullptr, nullptr};
    float*                  d_lm_head_scale_shard0[MAX_RANKS] = {nullptr, nullptr};
    float*                  d_lm_head_scale_shard1[MAX_RANKS] = {nullptr, nullptr};
    float*                  d_lm_head_col_scale[MAX_RANKS] = {nullptr, nullptr};
    void*                   d_embedding_tables[MAX_RANKS] = {nullptr, nullptr};
    void*                   d_p2p_comm[MAX_RANKS] = {nullptr, nullptr};
    uint32_t*               d_step_seq[MAX_RANKS] = {nullptr, nullptr};
    float*                  d_markov_b_norm[MAX_RANKS] = {nullptr, nullptr};
    FullDrafterGpuPointers* d_gpu_params[MAX_RANKS] = {nullptr, nullptr};
    const rad_bf16_t*       host_teacher_lm_head = nullptr;
    const rad_bf16_t*       host_teacher_embedding_table = nullptr;

    FullDrafterWeights() = default;

    ~FullDrafterWeights() {
        release();
    }

    // Move constructor & move assignment for RAII ownership transfer
    FullDrafterWeights(FullDrafterWeights&& other) noexcept {
        move_from(std::move(other));
    }

    FullDrafterWeights& operator=(FullDrafterWeights&& other) noexcept {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    // Explicitly delete copy constructor & assignment to prevent double-free
    FullDrafterWeights(const FullDrafterWeights&) = delete;
    FullDrafterWeights& operator=(const FullDrafterWeights&) = delete;

    const FullDrafterRankWeights& rank(size_t r) const {
        if (r < MAX_RANKS) {
            return ranks[r];
        }
        static const FullDrafterRankWeights s_empty{};
        return s_empty;
    }

    FullDrafterRankWeights& rank(size_t r) {
        if (r < MAX_RANKS) {
            return ranks[r];
        }
        static FullDrafterRankWeights s_empty{};
        s_empty.reset();
        return s_empty;
    }

    const FullDrafterRankWeights& operator[](size_t r) const {
        return rank(r);
    }

    FullDrafterRankWeights& operator[](size_t r) {
        return rank(r);
    }

    void release() {
        if (teacher_mmap_ptr && teacher_mmap_len > 0) {
            ::munmap(teacher_mmap_ptr, teacher_mmap_len);
            teacher_mmap_ptr = nullptr;
            teacher_mmap_len = 0;
        }
        int prev_dev = 0;
        bool have_prev = (hipGetDevice(&prev_dev) == hipSuccess);
        for (size_t r = 0; r < MAX_RANKS; ++r) {
            int dev = (ranks[r].device_id >= 0) ? ranks[r].device_id : static_cast<int>(r);
            if (raw_buffers[r] != nullptr) {
                if (owns_memory) {
                    (void)hipSetDevice(dev);
                    (void)hipFree(raw_buffers[r]);
                }
                raw_buffers[r] = nullptr;
            }
            if (d_lm_head_shard0[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_lm_head_shard0[r]);
                d_lm_head_shard0[r] = nullptr;
            }
            if (d_lm_head_shard1[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_lm_head_shard1[r]);
                d_lm_head_shard1[r] = nullptr;
            }
            for (void* ptr : {static_cast<void*>(d_lm_head_q8_shard0[r]),
                              static_cast<void*>(d_lm_head_q8_shard1[r]),
                              static_cast<void*>(d_lm_head_scale_shard0[r]),
                              static_cast<void*>(d_lm_head_scale_shard1[r]),
                              static_cast<void*>(d_lm_head_col_scale[r])}) {
                if (ptr) {
                    (void)hipSetDevice(dev);
                    (void)hipFree(ptr);
                }
            }
            d_lm_head_q8_shard0[r] = nullptr;
            d_lm_head_q8_shard1[r] = nullptr;
            d_lm_head_scale_shard0[r] = nullptr;
            d_lm_head_scale_shard1[r] = nullptr;
            d_lm_head_col_scale[r] = nullptr;
            if (d_embedding_tables[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_embedding_tables[r]);
                d_embedding_tables[r] = nullptr;
            }
            if (d_p2p_comm[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_p2p_comm[r]);
                d_p2p_comm[r] = nullptr;
            }
            if (d_step_seq[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_step_seq[r]);
                d_step_seq[r] = nullptr;
            }
            if (d_markov_b_norm[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_markov_b_norm[r]);
                d_markov_b_norm[r] = nullptr;
            }
            if (d_gpu_params[r] != nullptr) {
                (void)hipSetDevice(dev);
                (void)hipFree(d_gpu_params[r]);
                d_gpu_params[r] = nullptr;
            }
            ranks[r].reset();
        }
        if (have_prev) {
            (void)hipSetDevice(prev_dev);
        }
        owns_memory = false;
        num_ranks = 0;
        host_teacher_lm_head = nullptr;
        host_teacher_embedding_table = nullptr;
        FullDrafterRankWeights::reset();
    }

private:
    void move_from(FullDrafterWeights&& other) noexcept {
        static_cast<FullDrafterRankWeights&>(*this) = static_cast<const FullDrafterRankWeights&>(other);
        owns_memory = other.owns_memory;
        num_ranks = other.num_ranks;
        teacher_mmap_ptr = other.teacher_mmap_ptr;
        teacher_mmap_len = other.teacher_mmap_len;
        host_teacher_lm_head = other.host_teacher_lm_head;
        host_teacher_embedding_table = other.host_teacher_embedding_table;
        other.teacher_mmap_ptr = nullptr;
        other.teacher_mmap_len = 0;
        other.host_teacher_lm_head = nullptr;
        other.host_teacher_embedding_table = nullptr;
        for (size_t r = 0; r < MAX_RANKS; ++r) {
            raw_buffers[r] = other.raw_buffers[r];
            ranks[r] = other.ranks[r];
            d_lm_head_shard0[r] = other.d_lm_head_shard0[r];
            d_lm_head_shard1[r] = other.d_lm_head_shard1[r];
            d_lm_head_q8_shard0[r] = other.d_lm_head_q8_shard0[r];
            d_lm_head_q8_shard1[r] = other.d_lm_head_q8_shard1[r];
            d_lm_head_scale_shard0[r] = other.d_lm_head_scale_shard0[r];
            d_lm_head_scale_shard1[r] = other.d_lm_head_scale_shard1[r];
            d_lm_head_col_scale[r] = other.d_lm_head_col_scale[r];
            d_embedding_tables[r] = other.d_embedding_tables[r];
            d_p2p_comm[r] = other.d_p2p_comm[r];
            d_step_seq[r] = other.d_step_seq[r];
            d_markov_b_norm[r] = other.d_markov_b_norm[r];
            d_gpu_params[r] = other.d_gpu_params[r];
            other.raw_buffers[r] = nullptr;
            other.d_lm_head_shard0[r] = nullptr;
            other.d_lm_head_shard1[r] = nullptr;
            other.d_lm_head_q8_shard0[r] = nullptr;
            other.d_lm_head_q8_shard1[r] = nullptr;
            other.d_lm_head_scale_shard0[r] = nullptr;
            other.d_lm_head_scale_shard1[r] = nullptr;
            other.d_lm_head_col_scale[r] = nullptr;
            other.d_embedding_tables[r] = nullptr;
            other.d_p2p_comm[r] = nullptr;
            other.d_step_seq[r] = nullptr;
            other.d_markov_b_norm[r] = nullptr;
            other.d_gpu_params[r] = nullptr;
            other.ranks[r].reset();
        }
        other.owns_memory = false;
        other.num_ranks = 0;
        other.FullDrafterRankWeights::reset();
    }
};

// Sidecar Manifest representation
struct FullDrafterManifest {
    std::string format;
    std::string version;
    size_t      parameter_count = 0;
    size_t      sidecar_size_bytes = 0;
    std::string dtype;
    std::string target_model_id;
    std::string target_embedding_sha256;
    std::string target_head_sha256;
    std::string safetensors_file_sha256;
};

// Numerical conversion utilities
inline float bf16_to_float(rad_bf16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline rad_bf16_t float_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even
    uint32_t lsb = (u >> 16) & 1;
    uint32_t round_bias = 0x7FFF + lsb;
    u += round_bias;
    return static_cast<rad_bf16_t>(u >> 16);
}

inline float round_bf16(float f) {
    return bf16_to_float(float_to_bf16(f));
}

namespace rad_crypto {
inline std::string sha256(const void* data, size_t len) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    auto rotr = [](uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); };
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t total_bits = (uint64_t)len * 8ULL;
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> buf(padded_len, 0);
    std::memcpy(buf.data(), p, len);
    buf[len] = 0x80;
    for (int i = 0; i < 8; ++i) {
        buf[padded_len - 1 - i] = static_cast<uint8_t>(total_bits >> (i * 8));
    }
    for (size_t chunk = 0; chunk < padded_len; chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(buf[chunk + i * 4 + 0]) << 24) |
                   (static_cast<uint32_t>(buf[chunk + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(buf[chunk + i * 4 + 2]) << 8)  |
                   (static_cast<uint32_t>(buf[chunk + i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h_val = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h_val + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h_val = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;
    }
    char hex[65];
    std::snprintf(hex, sizeof(hex),
        "%08x%08x%08x%08x%08x%08x%08x%08x",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
    return std::string(hex);
}
} // namespace rad_crypto

/**
 * FullDrafterLoader: Parses safetensors sidecar, enforces teacher quarantine,
 * validates target SHA-256 bindings, and populates FullDrafterWeights.
 */
class FullDrafterLoader {
public:
    static bool validate_tensor_header(const std::string& header, std::string* err = nullptr) {
        try {
            const auto root = nlohmann::json::parse(header);
            if (!root.is_object()) throw std::runtime_error("Header must be an object");
            const bool is_alodlm = root.contains("exit_gate.net.weight");
            const size_t expected_total_bytes = is_alodlm ? FullDrafterConfig::ALODLM_TOTAL_BYTES
                                                          : FullDrafterConfig::TOTAL_BYTES;
            std::map<std::string, std::vector<size_t>> expected = {
                {"drafter.mask_vector", {768}},
                {"drafter.context_proj.weight", {768, 7680}},
                {"drafter.context_norm.weight", {768}},
                {"drafter.anchor_proj.weight", {768, 2560}},
                {"drafter.final_norm.weight", {768}},
                {"drafter.out_proj.weight", {2560, 768}},
                {"conv.w_base", {2, 768}},
                {"conv.w_proj.weight", {96, 768}},
                {"recurrent_norm.weight", {768}},
                {"markov.matrix_a.weight", {248320, 64}},
                {"markov.matrix_b.weight", {248320, 64}},
                {"selector.pred_codebook.weight", {248320, 64}},
                {"selector.succ_codebook.weight", {248320, 64}},
                {"selector.hidden_proj.weight", {64, 768}},
                {"confidence.norm.weight", {768}},
                {"confidence.fc1.weight", {256, 768}},
                {"confidence.fc2.weight", {1, 256}},
                {"drafter.blocks.0.input_layernorm.weight", {768}},
                {"drafter.blocks.0.post_attention_layernorm.weight", {768}},
                {"drafter.blocks.0.self_attn.q_proj.weight", {768, 768}},
                {"drafter.blocks.0.self_attn.k_proj.weight", {256, 768}},
                {"drafter.blocks.0.self_attn.v_proj.weight", {256, 768}},
                {"drafter.blocks.0.self_attn.o_proj.weight", {768, 768}},
                {"drafter.blocks.0.mlp.gate_proj.weight", {2304, 768}},
                {"drafter.blocks.0.mlp.up_proj.weight", {2304, 768}},
                {"drafter.blocks.0.mlp.down_proj.weight", {768, 2304}},
                {"drafter.blocks.1.input_layernorm.weight", {768}},
                {"drafter.blocks.1.post_attention_layernorm.weight", {768}},
                {"drafter.blocks.1.self_attn.q_proj.weight", {768, 768}},
                {"drafter.blocks.1.self_attn.k_proj.weight", {256, 768}},
                {"drafter.blocks.1.self_attn.v_proj.weight", {256, 768}},
                {"drafter.blocks.1.self_attn.o_proj.weight", {768, 768}},
                {"drafter.blocks.1.mlp.gate_proj.weight", {2304, 768}},
                {"drafter.blocks.1.mlp.up_proj.weight", {2304, 768}},
                {"drafter.blocks.1.mlp.down_proj.weight", {768, 2304}},
            };
            if (is_alodlm) {
                expected["exit_gate.net.weight"]   = {1, 768};
                expected["exit_gate.net.bias"]     = {1};
                expected["exit_gate.depth_bias"]   = {4};
            } else {
                expected["verification.norm.weight"] = {768};
                expected["verification.fc1.weight"]  = {256, 768};
                expected["verification.fc2.weight"]  = {1, 256};
            }
            size_t count = 0;
            std::vector<std::pair<size_t, size_t>> ranges;
            for (auto it = root.begin(); it != root.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                ++count;
                auto want = expected.find(it.key());
                if (want == expected.end()) throw std::runtime_error("Unexpected tensor: " + it.key());
                const auto& tensor = it.value();
                if (tensor.at("dtype") != "BF16" ||
                    tensor.at("shape").get<std::vector<size_t>>() != want->second)
                    throw std::runtime_error("Invalid dtype/shape: " + it.key());
                const auto offsets = tensor.at("data_offsets").get<std::vector<size_t>>();
                size_t bytes = 2;
                for (size_t dim : want->second) bytes *= dim;
                if (offsets.size() != 2 || offsets[0] > expected_total_bytes ||
                    offsets[1] > expected_total_bytes || offsets[1] < offsets[0] ||
                    offsets[1] - offsets[0] != bytes || offsets[0] % 2 != 0)
                    throw std::runtime_error("Invalid tensor byte range: " + it.key());
                ranges.emplace_back(offsets[0], offsets[1]);
            }
            if (count != expected.size()) throw std::runtime_error("Missing FullDrafter tensors");
            std::sort(ranges.begin(), ranges.end());
            size_t end = 0;
            for (const auto& range : ranges) {
                if (range.first != end) throw std::runtime_error("Overlapping or gapped tensor ranges");
                end = range.second;
            }
            if (end != expected_total_bytes) throw std::runtime_error("Incorrect payload size");
            return true;
        } catch (const std::exception& ex) {
            if (err) *err = ex.what();
            return false;
        }
    }

    static size_t find_tensor_offset(const std::string& header, const std::string& name, size_t* out_bytes = nullptr) {
        std::string needle = "\"" + name + "\":";
        size_t pos = header.find(needle);
        if (pos == std::string::npos) return static_cast<size_t>(-1);
        size_t do_pos = header.find("\"data_offsets\":[", pos);
        if (do_pos == std::string::npos) return static_cast<size_t>(-1);
        do_pos += 16;
        size_t comma_pos = header.find(',', do_pos);
        if (comma_pos == std::string::npos) return static_cast<size_t>(-1);
        size_t close_bracket = header.find(']', comma_pos);
        if (close_bracket == std::string::npos) return static_cast<size_t>(-1);
        try {
            std::string start_str = header.substr(do_pos, comma_pos - do_pos);
            size_t start_off = std::stoull(start_str);
            if (out_bytes) {
                std::string end_str = header.substr(comma_pos + 1, close_bracket - comma_pos - 1);
                size_t end_off = std::stoull(end_str);
                if (end_off < start_off) return static_cast<size_t>(-1);
                *out_bytes = end_off - start_off;
            }
            return start_off;
        } catch (...) {
            return static_cast<size_t>(-1);
        }
    }

    static bool load_weights(
        const std::string& safetensors_path,
        FullDrafterWeights* out_weights,
        FullDrafterManifest* out_manifest = nullptr,
        std::string* err = nullptr,
        int requested_ranks = 2
    ) {
        if (!out_weights) {
            if (err) *err = "Null out_weights pointer";
            return false;
        }

        // Memory leak on reload: free existing weights if already loaded
        if (out_weights->owns_memory) {
            out_weights->release();
        }

        std::string path = safetensors_path;
        if (path.empty()) {
            const char* env = std::getenv("RADIANCE_FULLDRAFTER_SIDECAR");
            if (env && *env) path = env;
        }
        if (path.empty()) {
            static const char* kDefaultPaths[] = {
                "/home/orca/drafter_data/checkpoints/flash_next_k12_exp/drafter_step_25000.safetensors",
                "/home/orca/drafter_data/checkpoints/flash_next_k12/drafter_final.safetensors"
            };
            for (const char* p : kDefaultPaths) {
                if (std::ifstream(p).good()) {
                    path = p;
                    break;
                }
            }
        }

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            if (err) *err = "Failed to open safetensors file: " + path;
            return false;
        }

        std::streamsize file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (file_size < 8) {
            if (err) *err = "File size smaller than safetensors header length field";
            return false;
        }

        uint64_t header_len = 0;
        file.read(reinterpret_cast<char*>(&header_len), 8);
        if (header_len > static_cast<uint64_t>(file_size - 8)) {
            if (err) *err = "Invalid safetensors header length: " + std::to_string(header_len);
            return false;
        }

        std::vector<char> header_buf(header_len + 1, 0);
        file.read(header_buf.data(), header_len);

        // 1. Extract tensor names and enforce strict Teacher Weight Quarantine
        std::string header_str(header_buf.data(), header_len);
        if (!validate_tensor_header(header_str, err)) return false;
        header_str = nlohmann::json::parse(header_str).dump();

        static const char* forbidden_patterns[] = {
            "target_embedding",
            "target_lm_head",
            "target_endnorm",
            "target_final_norm",
            "target_trunk",
            "teacher.",
            "frozen_teacher"
        };

        size_t scan_pos = 0;
        size_t found_tensors = 0;
        while ((scan_pos = header_str.find("\":{\"dtype\":", scan_pos)) != std::string::npos) {
            size_t key_end = scan_pos;
            size_t key_start = header_str.rfind('"', key_end - 1);
            if (key_start != std::string::npos) {
                std::string tname = header_str.substr(key_start + 1, key_end - key_start - 1);
                found_tensors++;
                for (const char* pat : forbidden_patterns) {
                    if (tname.find(pat) != std::string::npos) {
                        if (err) *err = std::string("Teacher weight quarantine violation! Forbidden pattern: ") + pat + " in tensor " + tname;
                        return false;
                    }
                }
            }
            scan_pos += 11;
        }

        // Helper to extract metadata fields
        auto extract_meta = [&](const std::string& key) -> std::string {
            std::string needle = "\"" + key + "\":\"";
            size_t kpos = header_str.find(needle);
            if (kpos == std::string::npos) return "";
            size_t vstart = kpos + needle.length();
            size_t vend = header_str.find('"', vstart);
            if (vend == std::string::npos) return "";
            return header_str.substr(vstart, vend - vstart);
        };

        const bool is_alodlm = (header_str.find("\"exit_gate.net.weight\"") != std::string::npos);
        const size_t payload_bytes = is_alodlm ? FullDrafterConfig::ALODLM_TOTAL_BYTES
                                               : FullDrafterConfig::TOTAL_BYTES;
        const size_t payload_params = is_alodlm ? FullDrafterConfig::ALODLM_TOTAL_PARAMS
                                                : FullDrafterConfig::TOTAL_PARAMS;

        uint32_t alodlm_max_depth = 4;
        float alodlm_tau = 0.4f;
        float alodlm_q = 0.5f;
        float alodlm_pos_penalty = 0.0f;
        if (is_alodlm) {
            int arch_stages = -1;
            int arch_stage_block_k = -1;
            try {
                const auto h_json = nlohmann::json::parse(header_str);
                if (h_json.contains("__metadata__") && h_json["__metadata__"].contains("architecture")) {
                    const auto arch_json = nlohmann::json::parse(h_json["__metadata__"]["architecture"].get<std::string>());
                    if (arch_json.contains("max_depth")) alodlm_max_depth = arch_json["max_depth"].get<uint32_t>();
                    if (arch_json.contains("stages")) arch_stages = arch_json["stages"].get<int>();
                    if (arch_json.contains("stage_block_k")) arch_stage_block_k = arch_json["stage_block_k"].get<int>();
                    if (arch_json.contains("model_config") && arch_json["model_config"].is_object()) {
                        const auto& mc = arch_json["model_config"];
                        if (mc.contains("max_depth")) alodlm_max_depth = mc["max_depth"].get<uint32_t>();
                        if (mc.contains("tau")) alodlm_tau = mc["tau"].get<float>();
                        if (mc.contains("q")) alodlm_q = mc["q"].get<float>();
                        if (mc.contains("position_penalty")) alodlm_pos_penalty = mc["position_penalty"].get<float>();
                    }
                }
            } catch (...) {}
            if (alodlm_max_depth == 1 &&
                ((arch_stages != -1 && arch_stages != 2) ||
                 (arch_stage_block_k != -1 && arch_stage_block_k != 4))) {
                if (err) *err = "Incompatible ALoDLM max_depth=1 topology in sidecar metadata (expected 2x4 DLoop: stages=2, stage_block_k=4)";
                return false;
            }
        }

        // Truncated file handling: validate physical file size before memory allocation
        if (static_cast<uint64_t>(file_size) < 8 + header_len + payload_bytes) {
            if (err) *err = "Safetensors file is truncated: total size " + std::to_string(file_size) +
                            " bytes < required header (8+" + std::to_string(header_len) +
                            ") + payload (" + std::to_string(payload_bytes) + ") bytes";
            return false;
        }

        // Validate all 38 tensors and collect their offsets
        static const char* kFullTensorNames[] = {
            "drafter.mask_vector", "drafter.context_proj.weight", "drafter.context_norm.weight",
            "drafter.anchor_proj.weight", "drafter.blocks.0.input_layernorm.weight",
            "drafter.blocks.0.self_attn.q_proj.weight", "drafter.blocks.0.self_attn.k_proj.weight",
            "drafter.blocks.0.self_attn.v_proj.weight", "drafter.blocks.0.self_attn.o_proj.weight",
            "drafter.blocks.0.post_attention_layernorm.weight", "drafter.blocks.0.mlp.gate_proj.weight",
            "drafter.blocks.0.mlp.up_proj.weight", "drafter.blocks.0.mlp.down_proj.weight",
            "drafter.blocks.1.input_layernorm.weight", "drafter.blocks.1.self_attn.q_proj.weight",
            "drafter.blocks.1.self_attn.k_proj.weight", "drafter.blocks.1.self_attn.v_proj.weight",
            "drafter.blocks.1.self_attn.o_proj.weight", "drafter.blocks.1.post_attention_layernorm.weight",
            "drafter.blocks.1.mlp.gate_proj.weight", "drafter.blocks.1.mlp.up_proj.weight",
            "drafter.blocks.1.mlp.down_proj.weight", "drafter.final_norm.weight",
            "drafter.out_proj.weight", "conv.w_base", "conv.w_proj.weight",
            "recurrent_norm.weight", "markov.matrix_a.weight", "markov.matrix_b.weight",
            "selector.pred_codebook.weight", "selector.succ_codebook.weight",
            "selector.hidden_proj.weight", "confidence.norm.weight",
            "confidence.fc1.weight", "confidence.fc2.weight",
            "verification.norm.weight", "verification.fc1.weight", "verification.fc2.weight"
        };
        static const char* kAlodlmTensorNames[] = {
            "drafter.mask_vector", "drafter.context_proj.weight", "drafter.context_norm.weight",
            "drafter.anchor_proj.weight", "drafter.blocks.0.input_layernorm.weight",
            "drafter.blocks.0.self_attn.q_proj.weight", "drafter.blocks.0.self_attn.k_proj.weight",
            "drafter.blocks.0.self_attn.v_proj.weight", "drafter.blocks.0.self_attn.o_proj.weight",
            "drafter.blocks.0.post_attention_layernorm.weight", "drafter.blocks.0.mlp.gate_proj.weight",
            "drafter.blocks.0.mlp.up_proj.weight", "drafter.blocks.0.mlp.down_proj.weight",
            "drafter.blocks.1.input_layernorm.weight", "drafter.blocks.1.self_attn.q_proj.weight",
            "drafter.blocks.1.self_attn.k_proj.weight", "drafter.blocks.1.self_attn.v_proj.weight",
            "drafter.blocks.1.self_attn.o_proj.weight", "drafter.blocks.1.post_attention_layernorm.weight",
            "drafter.blocks.1.mlp.gate_proj.weight", "drafter.blocks.1.mlp.up_proj.weight",
            "drafter.blocks.1.mlp.down_proj.weight", "drafter.final_norm.weight",
            "drafter.out_proj.weight", "conv.w_base", "conv.w_proj.weight",
            "recurrent_norm.weight", "markov.matrix_a.weight", "markov.matrix_b.weight",
            "selector.pred_codebook.weight", "selector.succ_codebook.weight",
            "selector.hidden_proj.weight", "confidence.norm.weight",
            "confidence.fc1.weight", "confidence.fc2.weight",
            "exit_gate.net.weight", "exit_gate.net.bias", "exit_gate.depth_bias"
        };

        struct TensorOffsetInfo {
            const char* name;
            size_t offset;
            size_t bytes;
        };
        constexpr size_t num_tensors = sizeof(kFullTensorNames) / sizeof(kFullTensorNames[0]);
        const char* const* active_names = is_alodlm ? kAlodlmTensorNames : kFullTensorNames;
        std::vector<TensorOffsetInfo> tensor_infos;
        tensor_infos.reserve(num_tensors);

        for (size_t ti = 0; ti < num_tensors; ++ti) {
            const char* tname = active_names[ti];
            size_t tensor_bytes = 0;
            size_t off = find_tensor_offset(header_str, tname, &tensor_bytes);
            if (off == static_cast<size_t>(-1)) {
                if (err) *err = std::string("Missing tensor in safetensors header: ") + tname;
                *out_weights = FullDrafterWeights{};
                return false;
            }
            if (off > payload_bytes ||
                tensor_bytes > payload_bytes ||
                off + tensor_bytes > payload_bytes) {
                if (err) *err = std::string("Tensor offset/size out of bounds for ") + tname +
                                ": offset=" + std::to_string(off) + ", bytes=" + std::to_string(tensor_bytes);
                *out_weights = FullDrafterWeights{};
                return false;
            }
            tensor_infos.push_back({tname, off, tensor_bytes});
        }

        // Query available HIP devices
        int dev_count = 0;
        hipError_t err_dev = hipGetDeviceCount(&dev_count);
        if (err_dev != hipSuccess || dev_count <= 0) {
            if (err) *err = "No HIP-capable GPU devices found: " + std::string(hipGetErrorString(err_dev));
            *out_weights = FullDrafterWeights{};
            return false;
        }

        int prev_dev = 0;
        (void)hipGetDevice(&prev_dev);

        size_t target_ranks = (requested_ranks > 0) ? static_cast<size_t>(requested_ranks) : 2;
        if (target_ranks > static_cast<size_t>(dev_count)) {
            target_ranks = static_cast<size_t>(dev_count);
        }
        if (target_ranks > FullDrafterWeights::MAX_RANKS) {
            target_ranks = FullDrafterWeights::MAX_RANKS;
        }

        // Read payload into temporary host staging buffer
        std::vector<uint8_t> host_staging(payload_bytes);
        file.seekg(8 + header_len, std::ios::beg);
        file.read(reinterpret_cast<char*>(host_staging.data()), payload_bytes);
        if (file.gcount() != static_cast<std::streamsize>(payload_bytes) || !file.good()) {
            if (err) *err = "Truncated file: failed to read full payload (expected " +
                            std::to_string(payload_bytes) + ", read " +
                            std::to_string(file.gcount()) + ")";
            *out_weights = FullDrafterWeights{};
            return false;
        }

        // Precompute exact L2 norms AND Rank-1 Orthogonal Cauchy-Schwarz decomposition (beta_v, ||b_v_perp||_2 * 1.001f, u_vec)
        std::vector<float> host_mb_norm(3 * FullDrafterConfig::VOCAB_SIZE + FullDrafterConfig::MARKOV_RANK, 0.0f);
        for (const auto& info : tensor_infos) {
            if (info.name == "markov.matrix_b.weight") {
                const rad_bf16_t* mb_host = reinterpret_cast<const rad_bf16_t*>(host_staging.data() + info.offset);
                double mean_vec[FullDrafterConfig::MARKOV_RANK] = {0.0};
                for (size_t v = 0; v < FullDrafterConfig::ACTIVE_VOCAB; ++v) {
                    const rad_bf16_t* row = mb_host + v * FullDrafterConfig::MARKOV_RANK;
                    for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                        uint32_t u = static_cast<uint32_t>(row[k]) << 16;
                        float fv = 0.0f;
                        std::memcpy(&fv, &u, sizeof(float));
                        mean_vec[k] += static_cast<double>(fv);
                    }
                }
                double mean_sq = 0.0;
                for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                    mean_vec[k] /= static_cast<double>(FullDrafterConfig::ACTIVE_VOCAB);
                    mean_sq += mean_vec[k] * mean_vec[k];
                }
                float u_vec[FullDrafterConfig::MARKOV_RANK] = {0.0f};
                if (mean_sq > 1e-20) {
                    const double inv_norm = 1.0 / std::sqrt(mean_sq);
                    for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                        u_vec[k] = static_cast<float>(mean_vec[k] * inv_norm);
                    }
                    // Re-normalize in FP32 so ||u_vec||_2 == 1.0f in FP32
                    double u_sq = 0.0;
                    for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                        u_sq += static_cast<double>(u_vec[k]) * static_cast<double>(u_vec[k]);
                    }
                    const float u_inv = static_cast<float>(1.0 / std::sqrt(u_sq));
                    for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                        u_vec[k] *= u_inv;
                    }
                } else {
                    u_vec[0] = 1.0f;
                }
                for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                    host_mb_norm[3 * FullDrafterConfig::VOCAB_SIZE + k] = u_vec[k];
                }
                for (size_t v = 0; v < FullDrafterConfig::ACTIVE_VOCAB; ++v) {
                    const rad_bf16_t* row = mb_host + v * FullDrafterConfig::MARKOV_RANK;
                    float sq = 0.0f;
                    float beta = 0.0f;
                    float fvals[FullDrafterConfig::MARKOV_RANK];
                    for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                        uint32_t u = static_cast<uint32_t>(row[k]) << 16;
                        float fv = 0.0f;
                        std::memcpy(&fv, &u, sizeof(float));
                        fvals[k] = fv;
                        sq += fv * fv;
                        beta += fv * u_vec[k];
                    }
                    float sq_perp = 0.0f;
                    for (size_t k = 0; k < FullDrafterConfig::MARKOV_RANK; ++k) {
                        float d = fvals[k] - beta * u_vec[k];
                        sq_perp += d * d;
                    }
                    host_mb_norm[v] = std::sqrt(sq) * 1.001f;
                    host_mb_norm[FullDrafterConfig::VOCAB_SIZE + 2 * v + 0] = beta;
                    host_mb_norm[FullDrafterConfig::VOCAB_SIZE + 2 * v + 1] = std::sqrt(sq_perp) * 1.001f + 1e-4f;
                }
                break;
            }
        }

        // Allocate directly in GPU VRAM and copy via hipMemcpyHtoD on each rank
        out_weights->release();
        out_weights->owns_memory = true;
        for (size_t r = 0; r < target_ranks; ++r) {
            hipError_t h_err = hipSetDevice(static_cast<int>(r));
            if (h_err != hipSuccess) {
                if (err) *err = "hipSetDevice(" + std::to_string(r) + ") failed: " + hipGetErrorString(h_err);
                out_weights->release();
                (void)hipSetDevice(prev_dev);
                return false;
            }

            void* d_mem = nullptr;
            h_err = hipMalloc(&d_mem, payload_bytes);
            if (h_err != hipSuccess || !d_mem) {
                if (err) *err = "hipMalloc failed on GPU " + std::to_string(r) + " (" +
                                std::to_string(payload_bytes) + " bytes): " +
                                hipGetErrorString(h_err);
                out_weights->release();
                (void)hipSetDevice(prev_dev);
                return false;
            }

            // Immediately track in raw_buffers so release() can free it on any error
            out_weights->raw_buffers[r] = reinterpret_cast<uint8_t*>(d_mem);
            out_weights->ranks[r].device_id = static_cast<int>(r);
            out_weights->ranks[r].raw_buffer = reinterpret_cast<uint8_t*>(d_mem);
            out_weights->ranks[r].allocated_bytes = payload_bytes;
            out_weights->ranks[r].is_alodlm_loop = is_alodlm;
            out_weights->ranks[r].alodlm_max_depth = alodlm_max_depth;
            out_weights->ranks[r].alodlm_tau = alodlm_tau;
            out_weights->ranks[r].alodlm_q = alodlm_q;
            out_weights->ranks[r].alodlm_pos_penalty = alodlm_pos_penalty;

            h_err = hipMemcpyHtoD(d_mem, host_staging.data(), payload_bytes);
            if (h_err != hipSuccess) {
                if (err) *err = "hipMemcpyHtoD failed on GPU " + std::to_string(r) + ": " +
                                hipGetErrorString(h_err);
                out_weights->release();
                (void)hipSetDevice(prev_dev);
                return false;
            }

            void* d_mbn = nullptr;
            const size_t mbn_bytes = host_mb_norm.size() * sizeof(float);
            if (hipMalloc(&d_mbn, mbn_bytes) == hipSuccess && d_mbn) {
                (void)hipMemcpyHtoD(d_mbn, host_mb_norm.data(), mbn_bytes);
                out_weights->d_markov_b_norm[r] = reinterpret_cast<float*>(d_mbn);
            }

            h_err = hipDeviceSynchronize();
            if (h_err != hipSuccess) {
                if (err) *err = "hipDeviceSynchronize failed on GPU " + std::to_string(r) + ": " +
                                hipGetErrorString(h_err);
                out_weights->release();
                (void)hipSetDevice(prev_dev);
                return false;
            }

            for (const auto& info : tensor_infos) {
                out_weights->ranks[r].bind_tensor(
                    info.name,
                    reinterpret_cast<const rad_bf16_t*>(out_weights->ranks[r].raw_buffer + info.offset)
                );
            }
        }

        (void)hipSetDevice(prev_dev);

        out_weights->num_ranks = target_ranks;
        out_weights->device_id = out_weights->ranks[0].device_id;
        out_weights->raw_buffer = out_weights->ranks[0].raw_buffer;
        out_weights->allocated_bytes = out_weights->ranks[0].allocated_bytes;
        out_weights->is_alodlm_loop = is_alodlm;
        out_weights->alodlm_max_depth = alodlm_max_depth;
        out_weights->alodlm_tau = alodlm_tau;
        out_weights->alodlm_q = alodlm_q;
        out_weights->alodlm_pos_penalty = alodlm_pos_penalty;

        for (const auto& info : tensor_infos) {
            out_weights->bind_tensor(
                info.name,
                reinterpret_cast<const rad_bf16_t*>(out_weights->raw_buffer + info.offset)
            );
        }

        if (out_weights->teacher_embedding_table) {
            for (size_t r = 0; r < out_weights->num_ranks && r < FullDrafterWeights::MAX_RANKS; ++r) {
                out_weights->ranks[r].teacher_embedding_table = out_weights->teacher_embedding_table;
                if (out_weights->teacher_lm_head) {
                    if (out_weights->num_ranks == 2) {
                        constexpr size_t V_SHARD = FullDrafterConfig::TP_SHARD_VOCAB;
                        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
                        out_weights->ranks[r].teacher_lm_head = out_weights->teacher_lm_head + r * V_SHARD * W;
                    } else {
                        out_weights->ranks[r].teacher_lm_head = out_weights->teacher_lm_head;
                    }
                }
            }
        }

        if (out_manifest) {
            out_manifest->format = extract_meta("format");
            if (out_manifest->format.empty()) out_manifest->format = "safetensors";
            out_manifest->version = extract_meta("version");
            if (out_manifest->version.empty()) out_manifest->version = "1.0";
            out_manifest->parameter_count = payload_params;
            out_manifest->sidecar_size_bytes = static_cast<size_t>(file_size);
            out_manifest->dtype = extract_meta("dtype");
            out_manifest->target_model_id = extract_meta("target_model_id");
            out_manifest->target_embedding_sha256 = extract_meta("target_embedding_sha256");
            out_manifest->target_head_sha256 = extract_meta("target_head_sha256");
        }

        return true;
    }

    static bool load_teacher_matrices(
        const std::string& teacher_safetensors_path,
        FullDrafterWeights* out_weights,
        std::string* err = nullptr,
        bool tp2_p2p_sharded = false
    ) {
        if (!out_weights) {
            if (err) *err = "Null out_weights pointer";
            return false;
        }
        std::string path = teacher_safetensors_path;
        if (path.empty()) {
            const char* env = std::getenv("RADIANCE_TEACHER_MATRICES");
            if (env && *env) path = env;
            else path = "/home/orca/drafter_data/teacher/teacher-matrices.safetensors";
        }
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (err) *err = "Failed to open teacher safetensors file: " + path;
            return false;
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size < 8) {
            ::close(fd);
            if (err) *err = "Invalid teacher safetensors file size: " + path;
            return false;
        }
        size_t file_size = static_cast<size_t>(st.st_size);
        void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (mapped == MAP_FAILED) {
            if (err) *err = "Failed to mmap teacher safetensors file: " + path;
            return false;
        }

        uint64_t header_len = *reinterpret_cast<const uint64_t*>(mapped);
        if (8 + header_len > file_size) {
            ::munmap(mapped, file_size);
            if (err) *err = "Corrupt safetensors header length in teacher file";
            return false;
        }

        std::string header_str(static_cast<const char*>(mapped) + 8, header_len);
        size_t lm_head_bytes = 0, emb_bytes = 0;
        size_t lm_head_off = find_tensor_offset(header_str, "lm_head.weight", &lm_head_bytes);
        size_t emb_off = find_tensor_offset(header_str, "model.language_model.embed_tokens.weight", &emb_bytes);

        if (lm_head_off == static_cast<size_t>(-1) || emb_off == static_cast<size_t>(-1)) {
            ::munmap(mapped, file_size);
            if (err) *err = "Missing lm_head.weight or model.language_model.embed_tokens.weight in teacher file";
            return false;
        }

        const uint8_t* payload_base = static_cast<const uint8_t*>(mapped) + 8 + header_len;
        if (8 + header_len + lm_head_off + lm_head_bytes > file_size ||
            8 + header_len + emb_off + emb_bytes > file_size) {
            ::munmap(mapped, file_size);
            if (err) *err = "Teacher tensor offsets exceed file size";
            return false;
        }

        const rad_bf16_t* lm_head_ptr = reinterpret_cast<const rad_bf16_t*>(payload_base + lm_head_off);
        const rad_bf16_t* emb_ptr = reinterpret_cast<const rad_bf16_t*>(payload_base + emb_off);

        if (out_weights->teacher_mmap_ptr && out_weights->teacher_mmap_len > 0) {
            ::munmap(out_weights->teacher_mmap_ptr, out_weights->teacher_mmap_len);
        }
        out_weights->teacher_mmap_ptr = mapped;
        out_weights->teacher_mmap_len = file_size;

        out_weights->teacher_lm_head = lm_head_ptr;
        out_weights->teacher_embedding_table = emb_ptr;
        out_weights->host_teacher_lm_head = lm_head_ptr;
        out_weights->host_teacher_embedding_table = emb_ptr;

        constexpr size_t V_SHARD = FullDrafterConfig::TP_SHARD_VOCAB;
        constexpr size_t V_TOTAL = FullDrafterConfig::VOCAB_SIZE;
        size_t W = lm_head_bytes / (V_TOTAL * sizeof(rad_bf16_t));
        if (W != 768 && W != FullDrafterConfig::TARGET_WIDTH) {
            ::munmap(mapped, file_size);
            if (err) *err = "Invalid teacher LM head width: " + std::to_string(W) + " (expected 768 or 2560)";
            return false;
        }

        bool is_fused = (W == 768);
        if (is_fused) {
            auto extract_teacher_meta = [&](const std::string& key) -> std::string {
                std::string needle = "\"" + key + "\":\"";
                size_t kpos = header_str.find(needle);
                if (kpos == std::string::npos) return "";
                size_t vstart = kpos + needle.length();
                size_t vend = header_str.find('"', vstart);
                if (vend == std::string::npos) return "";
                return header_str.substr(vstart, vend - vstart);
            };

            std::string bound_out_proj_hash = extract_teacher_meta("drafter_out_proj_sha256");
            if (out_weights->ranks[0].out_proj && !bound_out_proj_hash.empty()) {
                size_t op_bytes = FullDrafterConfig::TARGET_WIDTH * FullDrafterConfig::HIDDEN_SIZE * sizeof(rad_bf16_t);
                std::string actual_hash = rad_crypto::sha256(out_weights->ranks[0].out_proj, op_bytes);
                if (actual_hash != bound_out_proj_hash) {
                    ::munmap(mapped, file_size);
                    if (err) *err = "Cryptographic binding mismatch: Fused LM-head was compiled for out_proj SHA-256 " +
                                    bound_out_proj_hash + " but active drafter has " + actual_hash;
                    return false;
                }
            }
        }

        out_weights->lm_head_dim = W;
        out_weights->is_fused = is_fused;

        const size_t shard_bytes = V_SHARD * W * sizeof(rad_bf16_t);
        const size_t emb_table_bytes = V_TOTAL * FullDrafterConfig::TARGET_WIDTH * sizeof(rad_bf16_t);

        for (size_t r = 0; r < out_weights->num_ranks && r < FullDrafterWeights::MAX_RANKS; ++r) {
            out_weights->ranks[r].lm_head_dim = W;
            out_weights->ranks[r].is_fused = is_fused;
            if (out_weights->num_ranks == 2) {
                out_weights->ranks[r].teacher_lm_head = lm_head_ptr + r * V_SHARD * W;
            } else {
                out_weights->ranks[r].teacher_lm_head = lm_head_ptr;
            }
            out_weights->ranks[r].teacher_embedding_table = emb_ptr;
            out_weights->ranks[r].host_teacher_lm_head = out_weights->ranks[r].teacher_lm_head;
            out_weights->ranks[r].host_teacher_embedding_table = emb_ptr;
        }

        // Allocate teacher matrices directly into GPU VRAM if GPU ranks are active
        int prev_dev = 0;
        bool have_prev = (hipGetDevice(&prev_dev) == hipSuccess);
        int dev_count = 0;
        if (hipGetDeviceCount(&dev_count) == hipSuccess && dev_count >= 1 && out_weights->num_ranks > 0) {
            size_t n_ranks = out_weights->num_ranks;
            if (n_ranks > 2) n_ranks = 2;

            // Keep BF16 for exact reranking; quantize to symmetric per-row INT8 for the coarse W8A8 pass.
            std::vector<float> host_lm_head_col_scale(W, 1.0f);
            std::vector<int8_t> host_lm_head_q8(V_TOTAL * W);
            std::vector<float> host_lm_head_scale(V_TOTAL);
            const size_t n_workers = 8;
            {
                std::vector<std::thread> workers;
                workers.reserve(n_workers);
                for (size_t tid = 0; tid < n_workers; ++tid) {
                    workers.emplace_back([&, tid]() {
                        const size_t v0 = (V_TOTAL * tid) / n_workers;
                        const size_t v1 = (V_TOTAL * (tid + 1)) / n_workers;
                        for (size_t v = v0; v < v1; ++v) {
                            const rad_bf16_t* row = lm_head_ptr + v * W;
                            float maxabs = 0.0f;
                            for (size_t k = 0; k < W; ++k) {
                                uint32_t bits = static_cast<uint32_t>(row[k]) << 16;
                                float value;
                                std::memcpy(&value, &bits, sizeof(value));
                                maxabs = std::max(maxabs, std::fabs(value));
                            }
                            const float scale = maxabs > 1e-8f ? maxabs / 127.0f : 1.0f;
                            const float inv_scale = 1.0f / scale;
                            host_lm_head_scale[v] = scale;
                            int8_t* dst = host_lm_head_q8.data() + v * W;
                            for (size_t k = 0; k < W; ++k) {
                                uint32_t bits = static_cast<uint32_t>(row[k]) << 16;
                                float value;
                                std::memcpy(&value, &bits, sizeof(value));
                                dst[k] = static_cast<int8_t>(
                                    std::lrintf(std::clamp(value * inv_scale, -127.0f, 127.0f)));
                            }
                        }
                    });
                }
                for (auto& th : workers) th.join();
            }
            const size_t q8_shard_bytes = V_SHARD * W * sizeof(int8_t);
            const size_t scale_shard_bytes = V_SHARD * sizeof(float);
            const size_t col_scale_bytes = W * sizeof(float);

            // Enable P2P access between GPU 0 and GPU 1 if dual GPUs
            bool p2p_ok = false;
            if (n_ranks >= 2 && dev_count >= 2) {
                (void)hipSetDevice(0);
                int can01 = 0;
                if (hipDeviceCanAccessPeer(&can01, 0, 1) == hipSuccess && can01) {
                    (void)hipDeviceEnablePeerAccess(1, 0);
                    (void)hipGetLastError();
                }
                (void)hipSetDevice(1);
                int can10 = 0;
                if (hipDeviceCanAccessPeer(&can10, 1, 0) == hipSuccess && can10) {
                    (void)hipDeviceEnablePeerAccess(0, 0);
                    (void)hipGetLastError();
                }
                p2p_ok = (can01 != 0) && (can10 != 0);
            }
            const bool use_tp2_p2p = tp2_p2p_sharded && (n_ranks == 2) && p2p_ok;
            const size_t p2p_comm_bytes = FULLDRAFTER_P2P_MAX_SEQS * FullDrafterConfig::TOTAL_K * sizeof(FullDrafterP2PSlotPacket);

            for (size_t r = 0; r < n_ranks; ++r) {
                int dev = (out_weights->ranks[r].device_id >= 0) ? out_weights->ranks[r].device_id : static_cast<int>(r);
                (void)hipSetDevice(dev);

                if (!out_weights->d_lm_head_col_scale[r]) {
                    void* d_mem = nullptr;
                    if (hipMalloc(&d_mem, col_scale_bytes) == hipSuccess && d_mem) {
                        if (hipMemcpyHtoD(d_mem, host_lm_head_col_scale.data(), col_scale_bytes) == hipSuccess) {
                            out_weights->d_lm_head_col_scale[r] = static_cast<float*>(d_mem);
                        } else {
                            (void)hipFree(d_mem);
                        }
                    }
                }

                // 1. In TP2 P2P mode, Rank 0 allocates only Shard 0 and Rank 1 allocates only Shard 1 (saving 635.7 MB VRAM/GPU).
                if (!use_tp2_p2p || r == 0) {
                    if (!out_weights->d_lm_head_shard0[r]) {
                        void* d_s0 = nullptr;
                        if (hipMalloc(&d_s0, shard_bytes) == hipSuccess && d_s0) {
                            (void)hipMemcpyHtoD(d_s0, lm_head_ptr, shard_bytes);
                            out_weights->d_lm_head_shard0[r] = d_s0;
                        }
                    }
                }
                if (!use_tp2_p2p || r == 1) {
                    if (!out_weights->d_lm_head_shard1[r]) {
                        void* d_s1 = nullptr;
                        if (hipMalloc(&d_s1, shard_bytes) == hipSuccess && d_s1) {
                            (void)hipMemcpyHtoD(d_s1, lm_head_ptr + V_SHARD * W, shard_bytes);
                            out_weights->d_lm_head_shard1[r] = d_s1;
                        }
                    }
                }
                for (size_t shard = 0; shard < 2; ++shard) {
                    if (use_tp2_p2p && r != shard) continue;
                    int8_t*& d_q8 = shard == 0 ? out_weights->d_lm_head_q8_shard0[r]
                                              : out_weights->d_lm_head_q8_shard1[r];
                    float*& d_scale = shard == 0 ? out_weights->d_lm_head_scale_shard0[r]
                                                : out_weights->d_lm_head_scale_shard1[r];
                    if (!d_q8) {
                        void* d_mem = nullptr;
                        if (hipMalloc(&d_mem, q8_shard_bytes) == hipSuccess && d_mem) {
                            if (hipMemcpyHtoD(d_mem, host_lm_head_q8.data() + shard * V_SHARD * W, q8_shard_bytes) == hipSuccess) {
                                d_q8 = static_cast<int8_t*>(d_mem);
                            } else {
                                (void)hipFree(d_mem);
                            }
                        }
                    }
                    if (!d_scale) {
                        void* d_mem = nullptr;
                        if (hipMalloc(&d_mem, scale_shard_bytes) == hipSuccess && d_mem) {
                            if (hipMemcpyHtoD(d_mem, host_lm_head_scale.data() + shard * V_SHARD, scale_shard_bytes) == hipSuccess) {
                                d_scale = static_cast<float*>(d_mem);
                            } else {
                                (void)hipFree(d_mem);
                            }
                        }
                    }
                }

                // 2. Allocate teacher embedding table on rank r: 50/50 row-sharded in TP2 P2P mode (saving 606.25 MiB VRAM/GPU).
                if (!out_weights->d_embedding_tables[r]) {
                    const size_t r_emb_bytes = use_tp2_p2p ? shard_bytes : emb_table_bytes;
                    const rad_bf16_t* r_emb_src = use_tp2_p2p ? (emb_ptr + r * V_SHARD * FullDrafterConfig::TARGET_WIDTH) : emb_ptr;
                    void* d_emb = nullptr;
                    if (hipMalloc(&d_emb, r_emb_bytes) == hipSuccess && d_emb) {
                        (void)hipMemcpyHtoD(d_emb, r_emb_src, r_emb_bytes);
                        out_weights->d_embedding_tables[r] = d_emb;
                    }
                }

                if (use_tp2_p2p) {
                    if (!out_weights->d_p2p_comm[r]) {
                        void* d_comm = nullptr;
                        if (hipMalloc(&d_comm, p2p_comm_bytes) == hipSuccess && d_comm) {
                            (void)hipMemset(d_comm, 0, p2p_comm_bytes);
                            out_weights->d_p2p_comm[r] = d_comm;
                        }
                    }
                    if (!out_weights->d_step_seq[r]) {
                        void* d_seq = nullptr;
                        if (hipMalloc(&d_seq, sizeof(uint32_t)) == hipSuccess && d_seq) {
                            (void)hipMemset(d_seq, 0, sizeof(uint32_t));
                            out_weights->d_step_seq[r] = reinterpret_cast<uint32_t*>(d_seq);
                        }
                    }
                }
            }

            // Sync devices after copy
            for (size_t r = 0; r < n_ranks; ++r) {
                int dev = (out_weights->ranks[r].device_id >= 0) ? out_weights->ranks[r].device_id : static_cast<int>(r);
                (void)hipSetDevice(dev);
                (void)hipDeviceSynchronize();
            }

            // 3. Build FullDrafterGpuPointers on each rank and upload to GPU
            for (size_t r = 0; r < n_ranks; ++r) {
                int dev = (out_weights->ranks[r].device_id >= 0) ? out_weights->ranks[r].device_id : static_cast<int>(r);
                (void)hipSetDevice(dev);

                const auto& rw = out_weights->ranks[r];
                FullDrafterGpuPointers h_gpu{};
                h_gpu.teacher_lm_head_col_scale = out_weights->d_lm_head_col_scale[r];
                h_gpu.mask_vector = rw.mask_vector;
                h_gpu.context_proj = rw.context_proj;
                h_gpu.context_norm = rw.context_norm;
                h_gpu.anchor_proj = rw.anchor_proj;

                h_gpu.b0_input_layernorm = rw.blocks[0].input_layernorm;
                h_gpu.b0_q_proj = rw.blocks[0].q_proj;
                h_gpu.b0_k_proj = rw.blocks[0].k_proj;
                h_gpu.b0_v_proj = rw.blocks[0].v_proj;
                h_gpu.b0_o_proj = rw.blocks[0].o_proj;
                h_gpu.b0_post_attn_layernorm = rw.blocks[0].post_attn_layernorm;
                h_gpu.b0_gate_proj = rw.blocks[0].gate_proj;
                h_gpu.b0_up_proj = rw.blocks[0].up_proj;
                h_gpu.b0_down_proj = rw.blocks[0].down_proj;

                h_gpu.b1_input_layernorm = rw.blocks[1].input_layernorm;
                h_gpu.b1_q_proj = rw.blocks[1].q_proj;
                h_gpu.b1_k_proj = rw.blocks[1].k_proj;
                h_gpu.b1_v_proj = rw.blocks[1].v_proj;
                h_gpu.b1_o_proj = rw.blocks[1].o_proj;
                h_gpu.b1_post_attn_layernorm = rw.blocks[1].post_attn_layernorm;
                h_gpu.b1_gate_proj = rw.blocks[1].gate_proj;
                h_gpu.b1_up_proj = rw.blocks[1].up_proj;
                h_gpu.b1_down_proj = rw.blocks[1].down_proj;

                h_gpu.conv_w_base = rw.conv_w_base;
                h_gpu.conv_w_proj = rw.conv_w_proj;

                h_gpu.final_norm = rw.final_norm;
                h_gpu.out_proj = is_fused ? nullptr : rw.out_proj;
                h_gpu.recurrent_norm = rw.recurrent_norm;

                h_gpu.markov_matrix_a = rw.markov_matrix_a;
                h_gpu.markov_matrix_b = rw.markov_matrix_b;

                h_gpu.selector_pred_codebook = rw.selector_pred_codebook;
                h_gpu.selector_succ_codebook = rw.selector_succ_codebook;
                h_gpu.selector_hidden_proj = rw.selector_hidden_proj;

                h_gpu.confidence_norm = rw.confidence_norm;
                h_gpu.confidence_fc1 = rw.confidence_fc1;
                h_gpu.confidence_fc2 = rw.confidence_fc2;
                h_gpu.verification_norm = rw.verification_norm;
                h_gpu.verification_fc1 = rw.verification_fc1;
                h_gpu.verification_fc2 = rw.verification_fc2;

                const bool active_tp2_p2p = use_tp2_p2p &&
                                            out_weights->d_p2p_comm[0] && out_weights->d_p2p_comm[1] &&
                                            out_weights->d_step_seq[r];
                out_weights->ranks[r].tp2_p2p_active = active_tp2_p2p;
                if (active_tp2_p2p) {
                    h_gpu.teacher_lm_head_rank0 = (r == 0) ? out_weights->d_lm_head_shard0[0]
                                                           : out_weights->d_lm_head_shard1[1];
                    h_gpu.teacher_lm_head_rank1 = nullptr;
                    h_gpu.teacher_lm_head_q8_rank0 = (r == 0) ? out_weights->d_lm_head_q8_shard0[0]
                                                              : out_weights->d_lm_head_q8_shard1[1];
                    h_gpu.teacher_lm_head_scale_rank0 = (r == 0) ? out_weights->d_lm_head_scale_shard0[0]
                                                                 : out_weights->d_lm_head_scale_shard1[1];
                    h_gpu.teacher_lm_head_q8_rank1 = nullptr;
                    h_gpu.teacher_lm_head_scale_rank1 = nullptr;
                    h_gpu.rank_id               = static_cast<uint32_t>(r);
                    h_gpu.tp2_p2p_enabled       = 1u;
                    h_gpu.my_p2p_comm           = out_weights->d_p2p_comm[r];
                    h_gpu.peer_p2p_comm         = out_weights->d_p2p_comm[1 - r];
                    h_gpu.d_step_seq            = out_weights->d_step_seq[r];
                } else {
                    h_gpu.teacher_lm_head_rank0 = out_weights->d_lm_head_shard0[r];
                    h_gpu.teacher_lm_head_rank1 = out_weights->d_lm_head_shard1[r];
                    h_gpu.teacher_lm_head_q8_rank0 = out_weights->d_lm_head_q8_shard0[r];
                    h_gpu.teacher_lm_head_q8_rank1 = out_weights->d_lm_head_q8_shard1[r];
                    h_gpu.teacher_lm_head_scale_rank0 = out_weights->d_lm_head_scale_shard0[r];
                    h_gpu.teacher_lm_head_scale_rank1 = out_weights->d_lm_head_scale_shard1[r];
                    h_gpu.rank_id               = static_cast<uint32_t>(r);
                    h_gpu.tp2_p2p_enabled       = 0u;
                    h_gpu.my_p2p_comm           = nullptr;
                    h_gpu.peer_p2p_comm         = nullptr;
                    h_gpu.d_step_seq            = nullptr;
                }
                // Incomplete coarse uploads retain the exact BF16 fallback.
                if (!h_gpu.teacher_lm_head_col_scale || !h_gpu.teacher_lm_head_q8_rank0 ||
                    !h_gpu.teacher_lm_head_scale_rank0 ||
                    (!active_tp2_p2p && (!h_gpu.teacher_lm_head_q8_rank1 ||
                                        !h_gpu.teacher_lm_head_scale_rank1))) {
                    h_gpu.teacher_lm_head_q8_rank0 = nullptr;
                    h_gpu.teacher_lm_head_q8_rank1 = nullptr;
                }
                h_gpu.teacher_embedding_table = out_weights->d_embedding_tables[r];

                h_gpu.lm_head_dim = static_cast<uint32_t>(out_weights->lm_head_dim);
                h_gpu.is_fused = out_weights->is_fused ? 1 : 0;
                float dloop_theta = 0.20f;
                if (const char* env_th = std::getenv("RADIANCE_DLOOP_THETA")) {
                    if (*env_th) dloop_theta = std::strtof(env_th, nullptr);
                }
                uint32_t fixed_k = 0u;
                h_gpu.dloop_theta = dloop_theta;
                h_gpu.dloop_pad = 0u;
                h_gpu.markov_b_norm = out_weights->d_markov_b_norm[r];
                h_gpu.exit_gate_net_weight = rw.exit_gate_net_weight;
                h_gpu.exit_gate_net_bias   = rw.exit_gate_net_bias;
                h_gpu.exit_gate_depth_bias = rw.exit_gate_depth_bias;
                h_gpu.is_alodlm_loop       = rw.is_alodlm_loop ? 1u : 0u;
                h_gpu.alodlm_max_depth     = rw.alodlm_max_depth > 0 ? rw.alodlm_max_depth : 4u;
                h_gpu.alodlm_tau           = rw.alodlm_tau;
                h_gpu.alodlm_q             = rw.alodlm_q;
                h_gpu.alodlm_pos_penalty   = rw.alodlm_pos_penalty;
                h_gpu.alodlm_pad           = fixed_k;

                if (!out_weights->d_gpu_params[r]) {
                    void* d_par = nullptr;
                    if (hipMalloc(&d_par, sizeof(FullDrafterGpuPointers)) == hipSuccess && d_par) {
                        out_weights->d_gpu_params[r] = reinterpret_cast<FullDrafterGpuPointers*>(d_par);
                    }
                }
                if (out_weights->d_gpu_params[r]) {
                    (void)hipMemcpyHtoD(out_weights->d_gpu_params[r], &h_gpu, sizeof(FullDrafterGpuPointers));
                    out_weights->ranks[r].d_gpu_pointers = out_weights->d_gpu_params[r];
                }
            }

            if (have_prev) {
                (void)hipSetDevice(prev_dev);
            }
        }

        return true;
    }

    static void free_weights(FullDrafterWeights* weights) {
        if (!weights) return;
        weights->release();
    }
};

/**
 * HostStagedWeights: Transparently stages device weights to a temporary host buffer
 * if needed, enabling the CPU reference forward pass (FullDrafterExecutor) to run at
 * host RAM speeds without un-cached PCIe MMIO transaction stalls.
 */
class HostStagedWeights {
public:
    explicit HostStagedWeights(const FullDrafterRankWeights& src) {
        hipPointerAttribute_t attr{};
        bool is_device = false;
        if (src.raw_buffer && hipPointerGetAttributes(&attr, src.raw_buffer) == hipSuccess) {
            is_device = (attr.type == hipMemoryTypeDevice);
        }
        (void)hipGetLastError();
        if (is_device && src.raw_buffer && src.allocated_bytes > 0) {
            int prev_dev = 0;
            bool have_prev = (hipGetDevice(&prev_dev) == hipSuccess);
            if (src.device_id >= 0) {
                (void)hipSetDevice(src.device_id);
            }
            host_buf_.resize(src.allocated_bytes);
            (void)hipMemcpyDtoH(host_buf_.data(), src.raw_buffer, src.allocated_bytes);
            if (have_prev) {
                (void)hipSetDevice(prev_dev);
            }
            ptrdiff_t diff = host_buf_.data() - src.raw_buffer;
            rebind(src, diff);
        } else {
            view_ = src;
        }
    }

    const FullDrafterRankWeights& get() const { return view_; }

private:
    std::vector<uint8_t> host_buf_;
    FullDrafterRankWeights view_;

    static const rad_bf16_t* shift_ptr(const rad_bf16_t* ptr, ptrdiff_t diff) {
        if (!ptr) return nullptr;
        return reinterpret_cast<const rad_bf16_t*>(reinterpret_cast<const uint8_t*>(ptr) + diff);
    }

    void rebind(const FullDrafterRankWeights& src, ptrdiff_t diff) {
        view_ = src;
        view_.device_id = -1;
        view_.raw_buffer = host_buf_.data();
        view_.mask_vector = shift_ptr(src.mask_vector, diff);
        view_.context_proj = shift_ptr(src.context_proj, diff);
        view_.context_norm = shift_ptr(src.context_norm, diff);
        view_.anchor_proj = shift_ptr(src.anchor_proj, diff);
        for (size_t b = 0; b < 2; ++b) {
            view_.blocks[b].input_layernorm = shift_ptr(src.blocks[b].input_layernorm, diff);
            view_.blocks[b].q_proj = shift_ptr(src.blocks[b].q_proj, diff);
            view_.blocks[b].k_proj = shift_ptr(src.blocks[b].k_proj, diff);
            view_.blocks[b].v_proj = shift_ptr(src.blocks[b].v_proj, diff);
            view_.blocks[b].o_proj = shift_ptr(src.blocks[b].o_proj, diff);
            view_.blocks[b].post_attn_layernorm = shift_ptr(src.blocks[b].post_attn_layernorm, diff);
            view_.blocks[b].gate_proj = shift_ptr(src.blocks[b].gate_proj, diff);
            view_.blocks[b].up_proj = shift_ptr(src.blocks[b].up_proj, diff);
            view_.blocks[b].down_proj = shift_ptr(src.blocks[b].down_proj, diff);
        }
        view_.final_norm = shift_ptr(src.final_norm, diff);
        view_.out_proj = shift_ptr(src.out_proj, diff);
        view_.conv_w_base = shift_ptr(src.conv_w_base, diff);
        view_.conv_w_proj = shift_ptr(src.conv_w_proj, diff);
        view_.recurrent_norm = shift_ptr(src.recurrent_norm, diff);
        view_.markov_matrix_a = shift_ptr(src.markov_matrix_a, diff);
        view_.markov_matrix_b = shift_ptr(src.markov_matrix_b, diff);
        view_.selector_pred_codebook = shift_ptr(src.selector_pred_codebook, diff);
        view_.selector_succ_codebook = shift_ptr(src.selector_succ_codebook, diff);
        view_.selector_hidden_proj = shift_ptr(src.selector_hidden_proj, diff);
        view_.confidence_norm = shift_ptr(src.confidence_norm, diff);
        view_.confidence_fc1 = shift_ptr(src.confidence_fc1, diff);
        view_.confidence_fc2 = shift_ptr(src.confidence_fc2, diff);
        view_.verification_norm = shift_ptr(src.verification_norm, diff);
        view_.verification_fc1 = shift_ptr(src.verification_fc1, diff);
        view_.verification_fc2 = shift_ptr(src.verification_fc2, diff);
        view_.exit_gate_net_weight = shift_ptr(src.exit_gate_net_weight, diff);
        view_.exit_gate_net_bias = shift_ptr(src.exit_gate_net_bias, diff);
        view_.exit_gate_depth_bias = shift_ptr(src.exit_gate_depth_bias, diff);
    }
};

/**
 * FullDrafterExecutor: Executes the complete FullDrafter forward pass:
 * 1. 7680 BF16 context projection (b_ktap[0..2]) + anchor embedding projection
 * 2. 2x6 recurrent DLoop stages with DFlash2 dynamic convolution
 * 3. 2 Transformer decoder blocks (GQA 12:4, RoPE theta 1e7, SwiGLU 2304)
 * 4. Transition norm and boundary recirculating
 * 5. Output projection to 12 x 2560 proposal tokens
 */
class FullDrafterExecutor {
public:
    static void rms_norm(const float* in, const rad_bf16_t* weight, float* out, size_t dim, float eps = 1e-6f) {
        float sum_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            sum_sq += in[i] * in[i];
        }
        float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + eps);
        for (size_t i = 0; i < dim; ++i) {
            float w = weight ? bf16_to_float(weight[i]) : 1.0f;
            out[i] = round_bf16(in[i] * inv_rms * w);
        }
    }

    static void matmul_bf16(
        const float* x,             // [in_dim]
        const rad_bf16_t* weight,   // [out_dim, in_dim]
        float* out,                 // [out_dim]
        size_t in_dim,
        size_t out_dim
    ) {
        for (size_t o = 0; o < out_dim; ++o) {
            float sum = 0.0f;
            const rad_bf16_t* w_row = weight + o * in_dim;
            for (size_t i = 0; i < in_dim; ++i) {
                sum += x[i] * bf16_to_float(w_row[i]);
            }
            out[o] = round_bf16(sum);
        }
    }

    static void dflash2_conv(
        float* h_seq,               // [seq_len, 768]
        size_t seq_len,
        const rad_bf16_t* w_base,   // [2, 768]
        const rad_bf16_t* w_proj    // [96, 768]
    ) {
        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t G = FullDrafterConfig::CONV_GROUP_SIZE;
        constexpr size_t NG = FullDrafterConfig::CONV_NUM_GROUPS;

        std::vector<float> prev_x(seq_len * H, 0.0f);
        // Preceding token
        for (size_t t = 1; t < seq_len; ++t) {
            std::memcpy(&prev_x[t * H], &h_seq[(t - 1) * H], H * sizeof(float));
        }

        std::vector<float> out(seq_len * H, 0.0f);
        for (size_t t = 0; t < seq_len; ++t) {
            // Dynamic delta_t = w_proj @ x_t
            float delta[96] = {0.0f};
            if (w_proj) {
                for (size_t o = 0; o < 96; ++o) {
                    float s = 0.0f;
                    const rad_bf16_t* w_row = w_proj + o * H;
                    for (size_t i = 0; i < H; ++i) {
                        s += h_seq[t * H + i] * bf16_to_float(w_row[i]);
                    }
                    delta[o] = round_bf16(s);
                }
            }

            for (size_t c = 0; c < H; ++c) {
                size_t g_idx = c / G;
                float w0 = round_bf16(bf16_to_float(w_base[0 * H + c]) + delta[0 * NG + g_idx]);
                float w1 = round_bf16(bf16_to_float(w_base[1 * H + c]) + delta[1 * NG + g_idx]);

                float tap0 = round_bf16(w0 * h_seq[t * H + c]);
                float tap1 = (t >= 1) ? round_bf16(w1 * prev_x[t * H + c]) : 0.0f;
                out[t * H + c] = round_bf16(tap0 + tap1);
            }
        }
        std::memcpy(h_seq, out.data(), seq_len * H * sizeof(float));
    }

    static void transformer_block(
        float* h_seq,
        size_t seq_len,
        const FullDrafterBlockWeights& block
    ) {
        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t I = FullDrafterConfig::INTERMEDIATE_SIZE;
        constexpr size_t N_Q = FullDrafterConfig::NUM_Q_HEADS;
        constexpr size_t N_KV = FullDrafterConfig::NUM_KV_HEADS;
        constexpr size_t D_H = FullDrafterConfig::HEAD_DIM;

        std::vector<float> norm_in(H);
        std::vector<float> q(seq_len * H), k(seq_len * (N_KV * D_H)), v(seq_len * (N_KV * D_H));

        // 1. Pre-RMSNorm & Q, K, V projections
        for (size_t t = 0; t < seq_len; ++t) {
            rms_norm(&h_seq[t * H], block.input_layernorm, norm_in.data(), H);
            matmul_bf16(norm_in.data(), block.q_proj, &q[t * H], H, H);
            matmul_bf16(norm_in.data(), block.k_proj, &k[t * (N_KV * D_H)], H, N_KV * D_H);
            matmul_bf16(norm_in.data(), block.v_proj, &v[t * (N_KV * D_H)], H, N_KV * D_H);
        }

        // 2. Rotary Position Embedding (RoPE theta 1e7)
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t i = 0; i < D_H / 2; ++i) {
                float freq = 1.0f / std::pow(FullDrafterConfig::ROPE_THETA, static_cast<float>(2 * i) / static_cast<float>(D_H));
                float theta = static_cast<float>(t) * freq;
                float cos_th = std::cos(theta);
                float sin_th = std::sin(theta);

                // Apply to all Q heads
                for (size_t h = 0; h < N_Q; ++h) {
                    float* q_ptr = &q[t * H + h * D_H];
                    float q0 = q_ptr[i];
                    float q1 = q_ptr[i + D_H / 2];
                    q_ptr[i]           = round_bf16(q0 * cos_th - q1 * sin_th);
                    q_ptr[i + D_H / 2] = round_bf16(q0 * sin_th + q1 * cos_th);
                }
                // Apply to all KV heads
                for (size_t h = 0; h < N_KV; ++h) {
                    float* k_ptr = &k[t * (N_KV * D_H) + h * D_H];
                    float k0 = k_ptr[i];
                    float k1 = k_ptr[i + D_H / 2];
                    k_ptr[i]           = round_bf16(k0 * cos_th - k1 * sin_th);
                    k_ptr[i + D_H / 2] = round_bf16(k0 * sin_th + k1 * cos_th);
                }
            }
        }

        // 3. Causal Attention (12:4 GQA)
        std::vector<float> attn_out(seq_len * H, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(D_H));

        for (size_t q_h = 0; q_h < N_Q; ++q_h) {
            size_t kv_h = q_h / (N_Q / N_KV);
            for (size_t t = 0; t < seq_len; ++t) {
                std::vector<float> scores(t + 1);
                float max_score = -1e9f;
                for (size_t p = 0; p <= t; ++p) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < D_H; ++d) {
                        dot += q[t * H + q_h * D_H + d] * k[p * (N_KV * D_H) + kv_h * D_H + d];
                    }
                    scores[p] = dot * scale;
                    if (scores[p] > max_score) max_score = scores[p];
                }
                float exp_sum = 0.0f;
                for (size_t p = 0; p <= t; ++p) {
                    scores[p] = std::exp(scores[p] - max_score);
                    exp_sum += scores[p];
                }
                for (size_t p = 0; p <= t; ++p) {
                    float prob = round_bf16(scores[p] / exp_sum);
                    for (size_t d = 0; d < D_H; ++d) {
                        attn_out[t * H + q_h * D_H + d] = round_bf16(
                            attn_out[t * H + q_h * D_H + d] + prob * v[p * (N_KV * D_H) + kv_h * D_H + d]
                        );
                    }
                }
            }
        }

        // 4. Output projection and residual add
        std::vector<float> proj_attn(H);
        for (size_t t = 0; t < seq_len; ++t) {
            matmul_bf16(&attn_out[t * H], block.o_proj, proj_attn.data(), H, H);
            for (size_t i = 0; i < H; ++i) {
                h_seq[t * H + i] = round_bf16(h_seq[t * H + i] + proj_attn[i]);
            }
        }

        // 5. Post-attention RMSNorm and SwiGLU MLP
        std::vector<float> gate(I), up(I), down(H);
        for (size_t t = 0; t < seq_len; ++t) {
            rms_norm(&h_seq[t * H], block.post_attn_layernorm, norm_in.data(), H);
            matmul_bf16(norm_in.data(), block.gate_proj, gate.data(), H, I);
            matmul_bf16(norm_in.data(), block.up_proj, up.data(), H, I);

            // SiLU(gate) * up
            for (size_t i = 0; i < I; ++i) {
                float silu = gate[i] / (1.0f + std::exp(-gate[i]));
                gate[i] = round_bf16(silu * up[i]);
            }
            matmul_bf16(gate.data(), block.down_proj, down.data(), I, H);
            for (size_t i = 0; i < H; ++i) {
                h_seq[t * H + i] = round_bf16(h_seq[t * H + i] + down[i]);
            }
        }
    }

    /**
     * Executes one DLoop stage (1 base representation + 6 proposal masks -> 6 proposal outputs).
     */
    static void execute_stage(
        const float* base_repr,             // [768]
        const FullDrafterRankWeights& weights,
        float* out_hidden,                  // [6, 768]
        float* out_projected                // [6, 2560]
    ) {
        HostStagedWeights staged(weights);
        const auto& w = staged.get();

        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
        constexpr size_t Kb = FullDrafterConfig::BLOCK_K;
        constexpr size_t seq_len = 1 + Kb; // 1 + 6 = 7

        std::vector<float> h_seq(seq_len * H);
        std::memcpy(&h_seq[0], base_repr, H * sizeof(float));

        // Fill slots 1..6 with mask_vector
        for (size_t s = 1; s <= Kb; ++s) {
            for (size_t i = 0; i < H; ++i) {
                h_seq[s * H + i] = bf16_to_float(w.mask_vector[i]);
            }
        }

        // Pass through 2 blocks with DFlash2Conv in front of each
        for (size_t b = 0; b < 2; ++b) {
            if (w.conv_w_base) {
                dflash2_conv(h_seq.data(), seq_len, w.conv_w_base, w.conv_w_proj);
            }
            transformer_block(h_seq.data(), seq_len, w.blocks[b]);
        }

        // Final RMSNorm
        std::vector<float> norm_out(H);
        for (size_t s = 1; s <= Kb; ++s) {
            rms_norm(&h_seq[s * H], w.final_norm, norm_out.data(), H);
            if (out_hidden) {
                std::memcpy(&out_hidden[(s - 1) * H], norm_out.data(), H * sizeof(float));
            }
            if (out_projected) {
                matmul_bf16(norm_out.data(), w.out_proj, &out_projected[(s - 1) * W], H, W);
            }
        }
    }

    /**
     * Applies Markov bigram logit adjustment: logits += matrix_b @ matrix_a[prev_token].
     * Masks tokens outside active vocabulary to -INFINITY.
     * Supports TP=2 split vocabulary ($V/2 = 124,160$ per GPU) via vocab_offset and shard_size.
     */
    static void apply_markov_correction(
        float* logits,                       // [shard_size or VOCAB_SIZE] in-place
        uint32_t prev_token,
        const FullDrafterRankWeights& weights,
        size_t vocab_size = FullDrafterConfig::VOCAB_SIZE,
        size_t active_vocab = FullDrafterConfig::ACTIVE_VOCAB,
        size_t rank = FullDrafterConfig::MARKOV_RANK,
        size_t vocab_offset = 0,
        size_t shard_size = 0                // 0 means vocab_size
    ) {
        if (!logits) return;
        HostStagedWeights staged(weights);
        const auto& w = staged.get();
        if (!w.markov_matrix_a || !w.markov_matrix_b) return;

        // Bounds clamp
        if (prev_token >= vocab_size) prev_token = 0;

        const size_t effective_shard = (shard_size > 0) ? shard_size : vocab_size;

        // 1. Gather predecessor rank-64 representation h_a = matrix_a[prev]
        float h_a[64];
        const rad_bf16_t* a_row = w.markov_matrix_a + static_cast<size_t>(prev_token) * rank;
        for (size_t r = 0; r < rank; ++r) {
            h_a[r] = bf16_to_float(a_row[r]);
        }

        // 2. Project through matrix_b: delta[v] = sum_r matrix_b[v_global, r] * h_a[r]
        // Tokens outside active vocabulary (>= active_vocab) are masked to -INFINITY.
        for (size_t v = 0; v < effective_shard; ++v) {
            const size_t v_global = vocab_offset + v;
            if (v_global < active_vocab) {
                const rad_bf16_t* b_row = w.markov_matrix_b + v_global * rank;
                float sum = 0.0f;
                for (size_t r = 0; r < rank; ++r) {
                    sum += bf16_to_float(b_row[r]) * h_a[r];
                }
                logits[v] += sum;
            } else {
                logits[v] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    /**
     * Scores top-16 proposal candidates using trilinear codebook scoring:
     *   score[c] = unary[c] + E_succ[c] @ (hidden_proj @ h * E_pred[prev])
     */
    static void score_top16_selector(
        const float* hidden,                // [768] Drafter slot hidden state
        uint32_t prev_token,                // Predecessor token ID
        const uint32_t* candidate_ids,      // [16] Top-16 candidate token IDs
        const float* unary_logits,          // [16] Unary logits for candidates
        float* out_scores,                  // [16] Output reranked scores
        const FullDrafterRankWeights& weights,
        size_t hidden_size = FullDrafterConfig::HIDDEN_SIZE,
        size_t vocab_size = FullDrafterConfig::VOCAB_SIZE,
        size_t rank = FullDrafterConfig::SELECTOR_RANK,
        size_t top_k = FullDrafterConfig::TOP_K
    ) {
        if (!out_scores) return;
        if (!hidden || !candidate_ids || !unary_logits) {
            for (size_t i = 0; i < top_k; ++i) out_scores[i] = unary_logits ? unary_logits[i] : 0.0f;
            return;
        }
        HostStagedWeights staged(weights);
        const auto& w = staged.get();
        if (!w.selector_pred_codebook || !w.selector_succ_codebook || !w.selector_hidden_proj) {
            for (size_t i = 0; i < top_k; ++i) out_scores[i] = unary_logits[i];
            return;
        }

        if (prev_token >= vocab_size) prev_token = 0;

        // 1. Hidden projection: h_proj = hidden_proj @ hidden  [64]
        float h_proj[64];
        for (size_t r = 0; r < rank; ++r) {
            float sum = 0.0f;
            const rad_bf16_t* w_row = w.selector_hidden_proj + r * hidden_size;
            for (size_t i = 0; i < hidden_size; ++i) {
                sum += bf16_to_float(w_row[i]) * hidden[i];
            }
            h_proj[r] = sum;
        }

        // 2. Predecessor codebook lookup & interaction vector: q[r] = h_proj[r] * pred_emb[r]
        float q[64];
        const rad_bf16_t* pred_row = w.selector_pred_codebook + static_cast<size_t>(prev_token) * rank;
        for (size_t r = 0; r < rank; ++r) {
            q[r] = h_proj[r] * bf16_to_float(pred_row[r]);
        }

        // 3. Bilinear dot product with each candidate successor codebook row
        for (size_t c = 0; c < top_k; ++c) {
            uint32_t cand_id = candidate_ids[c];
            float bias = 0.0f;
            if (cand_id < vocab_size) {
                const rad_bf16_t* succ_row = w.selector_succ_codebook + static_cast<size_t>(cand_id) * rank;
                for (size_t r = 0; r < rank; ++r) {
                    bias += bf16_to_float(succ_row[r]) * q[r];
                }
            }
            out_scores[c] = unary_logits[c] + bias;
        }
    }

    struct Candidate {
        float logit;
        uint32_t id;
    };

    /**
     * Extracts top-K candidates from local logits shard with global token IDs.
     * Descending order, tie-breaking to lower token ID.
     */
    static void extract_shard_topk(
        const float* shard_logits,
        size_t shard_size,
        size_t vocab_offset,
        std::vector<Candidate>& out_topk,
        size_t K = FullDrafterConfig::TOP_K
    ) {
        out_topk.clear();
        if (!shard_logits || shard_size == 0) return;
        out_topk.reserve(K);

        for (size_t v = 0; v < shard_size; ++v) {
            float val = shard_logits[v];
            uint32_t global_id = static_cast<uint32_t>(vocab_offset + v);
            if (out_topk.size() < K) {
                out_topk.push_back({val, global_id});
                if (out_topk.size() == K) {
                    std::sort(out_topk.begin(), out_topk.end(), [](const Candidate& a, const Candidate& b) {
                        if (a.logit != b.logit) return a.logit > b.logit;
                        return a.id < b.id;
                    });
                }
            } else if (val > out_topk.back().logit ||
                       (val == out_topk.back().logit && global_id < out_topk.back().id)) {
                out_topk.back() = {val, global_id};
                for (int i = static_cast<int>(K) - 2; i >= 0; --i) {
                    if (out_topk[i + 1].logit > out_topk[i].logit ||
                        (out_topk[i + 1].logit == out_topk[i].logit && out_topk[i + 1].id < out_topk[i].id)) {
                        std::swap(out_topk[i], out_topk[i + 1]);
                    } else {
                        break;
                    }
                }
            }
        }
        if (out_topk.size() < K && out_topk.size() > 1) {
            std::sort(out_topk.begin(), out_topk.end(), [](const Candidate& a, const Candidate& b) {
                if (a.logit != b.logit) return a.logit > b.logit;
                return a.id < b.id;
            });
        }
    }

    /**
     * Merges top-K candidate lists from Rank 0 and Rank 1 into global top-K.
     * Descending order, tie-breaking to lower token ID.
     */
    static void merge_topk_tp2(
        const std::vector<Candidate>& rank0_cands,
        const std::vector<Candidate>& rank1_cands,
        std::vector<Candidate>& out_merged,
        size_t K = FullDrafterConfig::TOP_K
    ) {
        out_merged.clear();
        out_merged.reserve(K);
        size_t i0 = 0, i1 = 0;
        while (out_merged.size() < K && (i0 < rank0_cands.size() || i1 < rank1_cands.size())) {
            if (i0 < rank0_cands.size() && i1 < rank1_cands.size()) {
                const auto& c0 = rank0_cands[i0];
                const auto& c1 = rank1_cands[i1];
                if (c0.logit > c1.logit || (c0.logit == c1.logit && c0.id < c1.id)) {
                    out_merged.push_back(c0);
                    i0++;
                } else {
                    out_merged.push_back(c1);
                    i1++;
                }
            } else if (i0 < rank0_cands.size()) {
                out_merged.push_back(rank0_cands[i0++]);
            } else {
                out_merged.push_back(rank1_cands[i1++]);
            }
        }
    }

    /**
     * Gathers split vocabulary logits from Rank 0 and Rank 1 into full vocabulary logits.
     * Supports M rows (row-interleaved). Rank 0 columns [0 .. V/2-1], Rank 1 columns [V/2 .. V-1].
     */
    static void gather_tp2_logits(
        const float* shard_rank0,
        const float* shard_rank1,
        float* out_full_logits,
        size_t shard_size = FullDrafterConfig::TP_SHARD_VOCAB,
        size_t num_rows = 1
    ) {
        if (!shard_rank0 || !shard_rank1 || !out_full_logits || shard_size == 0) return;
        for (size_t m = 0; m < num_rows; ++m) {
            std::memcpy(out_full_logits + m * (2 * shard_size),
                        shard_rank0 + m * shard_size,
                        shard_size * sizeof(float));
            std::memcpy(out_full_logits + m * (2 * shard_size) + shard_size,
                        shard_rank1 + m * shard_size,
                        shard_size * sizeof(float));
        }
    }

    /**
     * Extracts top-16 candidates from proposal logits, reranks them using the selector,
     * and selects the winning proposal token.
     */
    static void extract_and_rerank_slot(
        const float* logits,                // [VOCAB_SIZE] proposal logits with Markov adjustment
        const float* hidden,                // [768] drafter hidden state for slot
        uint32_t prev_token,                // predecessor token ID
        const FullDrafterRankWeights& weights,
        uint32_t* out_cand_ids,             // [16] optional candidate IDs output
        float* out_scores,                  // [16] optional scores output
        uint32_t* selected_token            // [1] output winning token ID
    ) {
        constexpr size_t K = FullDrafterConfig::TOP_K;
        constexpr size_t V_ACTIVE = FullDrafterConfig::ACTIVE_VOCAB;

        std::vector<Candidate> top_k;
        extract_shard_topk(logits, V_ACTIVE, 0, top_k, K);

        uint32_t cand_ids[K] = {0};
        float unary[K] = {0.0f};
        for (size_t i = 0; i < top_k.size(); ++i) {
            cand_ids[i] = top_k[i].id;
            unary[i] = top_k[i].logit;
            if (out_cand_ids) out_cand_ids[i] = cand_ids[i];
        }

        float scores[K] = {0.0f};
        score_top16_selector(
            hidden,
            prev_token,
            cand_ids,
            unary,
            scores,
            weights
        );

        if (out_scores) {
            for (size_t i = 0; i < K; ++i) out_scores[i] = scores[i];
        }

        size_t best_idx = 0;
        float best_score = scores[0];
        for (size_t i = 1; i < K; ++i) {
            if (scores[i] > best_score || (scores[i] == best_score && cand_ids[i] < cand_ids[best_idx])) {
                best_score = scores[i];
                best_idx = i;
            }
        }

        if (selected_token) {
            *selected_token = cand_ids[best_idx];
        }
    }

    /**
     * Performs candidate extraction, cross-GPU gathering/merge, Top-16 Selector scoring,
     * and argmax token selection under TP=2.
     */
    static void extract_and_rerank_slot_tp2(
        const float* shard_logits_rank0,    // [124160] Rank 0 Markov-adjusted logits
        const float* shard_logits_rank1,    // [124160] Rank 1 Markov-adjusted logits
        const float* hidden,                // [768] Drafter hidden state for slot
        uint32_t prev_token,                // Predecessor token ID
        const FullDrafterRankWeights& weights,
        uint32_t* out_cand_ids,             // [16] optional candidate IDs output
        float* out_scores,                  // [16] optional scores output
        uint32_t* selected_token            // [1] output winning token ID
    ) {
        if (!shard_logits_rank0 || !shard_logits_rank1 || !hidden) return;
        constexpr size_t K = FullDrafterConfig::TOP_K;
        constexpr size_t SHARD = FullDrafterConfig::TP_SHARD_VOCAB;

        // 1. Extract local top-K on each rank.
        // Rank 1 only extracts over active columns to eliminate padding tokens (248077..248319).
        size_t active_shard1 = (FullDrafterConfig::ACTIVE_VOCAB > SHARD) ? (FullDrafterConfig::ACTIVE_VOCAB - SHARD) : 0;
        if (active_shard1 > SHARD) active_shard1 = SHARD;

        std::vector<Candidate> top_rank0, top_rank1;
        extract_shard_topk(shard_logits_rank0, SHARD, 0, top_rank0, K);
        extract_shard_topk(shard_logits_rank1, active_shard1, SHARD, top_rank1, K);

        // 2. Cross-GPU merge top-K candidates
        std::vector<Candidate> merged;
        merge_topk_tp2(top_rank0, top_rank1, merged, K);

        uint32_t cand_ids[K] = {0};
        float unary[K] = {0.0f};
        for (size_t i = 0; i < merged.size(); ++i) {
            cand_ids[i] = merged[i].id;
            unary[i] = merged[i].logit;
            if (out_cand_ids) out_cand_ids[i] = cand_ids[i];
        }

        // 3. Score candidates with Top-16 Selector
        float scores[K] = {0.0f};
        score_top16_selector(hidden, prev_token, cand_ids, unary, scores, weights);

        if (out_scores) {
            for (size_t i = 0; i < K; ++i) out_scores[i] = scores[i];
        }

        // 4. Argmax selection with lower token ID tie-breaking
        size_t best_idx = 0;
        float best_score = scores[0];
        for (size_t i = 1; i < K; ++i) {
            if (scores[i] > best_score || (scores[i] == best_score && cand_ids[i] < cand_ids[best_idx])) {
                best_score = scores[i];
                best_idx = i;
            }
        }

        if (selected_token) {
            *selected_token = cand_ids[best_idx];
        }
    }

    /**
     * Computes Stage 2 base representation using discrete token embedding lookup:
     *   base2 = recurrent_norm(last_hidden_s1) + anchor_proj(teacher_embedding[boundary_token])
     *
     * Strictly eliminates continuous fallback representations.
     * Throws std::invalid_argument if teacher_embedding_table is null.
     */
    static void compute_stage2_base(
        const float* last_hidden_s1,               // [768] terminal hidden state h1[5]
        uint32_t boundary_token,                   // discrete token proposal from slot 5 (T_6)
        const rad_bf16_t* teacher_embedding_table, // [248320, 2560] teacher token embeddings (MANDATORY)
        const FullDrafterRankWeights& weights,
        float* out_base2                           // [768] output base representation for Stage 2
    ) {
        if (!last_hidden_s1 || !out_base2) {
            throw std::invalid_argument("compute_stage2_base: last_hidden_s1 and out_base2 cannot be null");
        }
        const rad_bf16_t* emb_tbl = teacher_embedding_table ? teacher_embedding_table : (weights.host_teacher_embedding_table ? weights.host_teacher_embedding_table : weights.teacher_embedding_table);
        if (!emb_tbl) {
            throw std::invalid_argument("compute_stage2_base: teacher_embedding_table cannot be null (continuous fallback eliminated)");
        }
        constexpr size_t V = FullDrafterConfig::VOCAB_SIZE;
        if (boundary_token >= V) {
            throw std::out_of_range("compute_stage2_base: boundary_token " + std::to_string(boundary_token) + " exceeds vocabulary size " + std::to_string(V));
        }
        HostStagedWeights staged(weights);
        const auto& w = staged.get();
        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;

        std::vector<float> norm_r1(H);
        rms_norm(last_hidden_s1, w.recurrent_norm, norm_r1.data(), H);

        hipPointerAttribute_t attr_emb{};
        std::vector<rad_bf16_t> dev_row;
        const rad_bf16_t* emb_row = nullptr;
        if (hipPointerGetAttributes(&attr_emb, emb_tbl) == hipSuccess && attr_emb.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_embedding_table) {
                emb_row = weights.host_teacher_embedding_table + static_cast<size_t>(boundary_token) * W;
            } else {
                dev_row.resize(W);
                const rad_bf16_t* src_row = emb_tbl + static_cast<size_t>(boundary_token) * W;
                (void)hipMemcpyDtoH(dev_row.data(), const_cast<rad_bf16_t*>(src_row), W * sizeof(rad_bf16_t));
                emb_row = dev_row.data();
            }
        } else {
            (void)hipGetLastError();
            emb_row = emb_tbl + static_cast<size_t>(boundary_token) * W;
        }

        std::vector<float> emb_f(W);
        for (size_t i = 0; i < W; ++i) {
            emb_f[i] = bf16_to_float(emb_row[i]);
        }

        std::vector<float> proj_bound(H);
        matmul_bf16(emb_f.data(), w.anchor_proj, proj_bound.data(), W, H);

        for (size_t i = 0; i < H; ++i) {
            out_base2[i] = round_bf16(norm_r1[i] + proj_bound[i]);
        }
    }

    /**
     * Computes Stage 2 base representation using an already-extracted discrete token embedding:
     *   base2 = recurrent_norm(last_hidden_s1) + anchor_proj(boundary_token_embedding)
     */
    static void compute_stage2_base_with_embedding(
        const float* last_hidden_s1,               // [768] terminal hidden state h1[5]
        const rad_bf16_t* boundary_token_embedding,// [2560] teacher embedding for boundary token (MANDATORY)
        const FullDrafterRankWeights& weights,
        float* out_base2                           // [768] output base representation for Stage 2
    ) {
        if (!last_hidden_s1 || !out_base2) {
            throw std::invalid_argument("compute_stage2_base_with_embedding: last_hidden_s1 and out_base2 cannot be null");
        }
        if (!boundary_token_embedding) {
            throw std::invalid_argument("compute_stage2_base_with_embedding: boundary_token_embedding cannot be null");
        }
        HostStagedWeights staged(weights);
        const auto& w = staged.get();
        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;

        std::vector<float> norm_r1(H);
        rms_norm(last_hidden_s1, w.recurrent_norm, norm_r1.data(), H);

        std::vector<float> emb_f(W);
        for (size_t i = 0; i < W; ++i) {
            emb_f[i] = bf16_to_float(boundary_token_embedding[i]);
        }

        std::vector<float> proj_bound(H);
        matmul_bf16(emb_f.data(), w.anchor_proj, proj_bound.data(), W, H);

        for (size_t i = 0; i < H; ++i) {
            out_base2[i] = round_bf16(norm_r1[i] + proj_bound[i]);
        }
    }

    /**
     * Executes full algorithmic proposal generation over 12 slots (2 DLoop stages of 6 slots),
     * producing discretized candidate tokens, DAG tree structures, and projected representations.
     * Enforces Discrete Two-Stage Chaining (2x6=12) with NO continuous fallback and NO null-logits.
     */
    static void propose_candidates(
        const rad_bf16_t* context_features,        // [7680]
        const rad_bf16_t* anchor_embedding,       // [2560]
        uint32_t anchor_token,                     // Committed anchor token ID
        const rad_bf16_t* teacher_lm_head,         // [248320, 2560] (MANDATORY)
        const rad_bf16_t* teacher_embedding_table, // [248320, 2560] (MANDATORY)
        const FullDrafterRankWeights& weights,
        uint32_t* out_tokens,                      // [12] proposed token IDs
        rad_bf16_t* out_proposals = nullptr,       // [12, 2560] optional continuous representations
        uint32_t* out_cand_tree = nullptr,         // [12, 16] optional top-16 candidate IDs
        float* out_score_tree = nullptr            // [12, 16] optional selector scores
    ) {
        if (!context_features || !anchor_embedding || !out_tokens) {
            throw std::invalid_argument("propose_candidates requires non-null context_features, anchor_embedding, and out_tokens");
        }
        const rad_bf16_t* lm_head = teacher_lm_head ? teacher_lm_head : (weights.host_teacher_lm_head ? weights.host_teacher_lm_head : weights.teacher_lm_head);
        const rad_bf16_t* wte = teacher_embedding_table ? teacher_embedding_table : (weights.host_teacher_embedding_table ? weights.host_teacher_embedding_table : weights.teacher_embedding_table);
        hipPointerAttribute_t attr_lm{}, attr_wte{};
        if (lm_head && hipPointerGetAttributes(&attr_lm, lm_head) == hipSuccess && attr_lm.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_lm_head) lm_head = weights.host_teacher_lm_head;
        } else {
            (void)hipGetLastError();
        }
        if (wte && hipPointerGetAttributes(&attr_wte, wte) == hipSuccess && attr_wte.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_embedding_table) wte = weights.host_teacher_embedding_table;
        } else {
            (void)hipGetLastError();
        }
        if (!lm_head || !wte) {
            throw std::invalid_argument("propose_candidates requires non-null teacher_lm_head and teacher_embedding_table (continuous fallback and null-logits eliminated)");
        }
        HostStagedWeights staged(weights);
        const auto& w = staged.get();
        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
        constexpr size_t C = FullDrafterConfig::CONTEXT_DIM;
        constexpr size_t Kb = FullDrafterConfig::BLOCK_K;
        constexpr size_t V = FullDrafterConfig::VOCAB_SIZE;
        constexpr size_t K = FullDrafterConfig::TOP_K;

        std::vector<float> ctx_f(C), anc_f(W);
        for (size_t i = 0; i < C; ++i) ctx_f[i] = bf16_to_float(context_features[i]);
        for (size_t i = 0; i < W; ++i) anc_f[i] = bf16_to_float(anchor_embedding[i]);

        // Stage 1 Base: RMSNorm(context_proj(ctx)) + anchor_proj(anc)
        std::vector<float> proj_ctx(H), norm_ctx(H), proj_anc(H), base1(H);
        matmul_bf16(ctx_f.data(), w.context_proj, proj_ctx.data(), C, H);
        rms_norm(proj_ctx.data(), w.context_norm, norm_ctx.data(), H);
        matmul_bf16(anc_f.data(), w.anchor_proj, proj_anc.data(), W, H);
        for (size_t i = 0; i < H; ++i) base1[i] = round_bf16(norm_ctx[i] + proj_anc[i]);

        std::vector<float> h1(Kb * H), o1(Kb * W);
        execute_stage(base1.data(), w, h1.data(), o1.data());

        uint32_t prev_tok = anchor_token;
        std::vector<float> logits(V, 0.0f);

        for (size_t j = 0; j < Kb; ++j) {
            const float* o_slot = &o1[j * W];
            const float* h_slot = &h1[j * H];

            matmul_bf16(o_slot, lm_head, logits.data(), W, V);
            apply_markov_correction(logits.data(), prev_tok, w);

            uint32_t chosen = 0;
            uint32_t* slot_cands = out_cand_tree ? &out_cand_tree[j * K] : nullptr;
            float* slot_scores = out_score_tree ? &out_score_tree[j * K] : nullptr;
            extract_and_rerank_slot(logits.data(), h_slot, prev_tok, w, slot_cands, slot_scores, &chosen);

            out_tokens[j] = chosen;
            prev_tok = chosen;
        }

        // Stage 2 Transition: Discrete Recurrent Boundary (slot 5 winning token T_6)
        // base2 = recurrent_norm(h1[5]) + anchor_proj(wte(T_6))
        uint32_t boundary_tok = out_tokens[Kb - 1];
        std::vector<float> base2(H);
        compute_stage2_base(
            &h1[(Kb - 1) * H],
            boundary_tok,
            wte,
            w,
            base2.data()
        );

        std::vector<float> h2(Kb * H), o2(Kb * W);
        execute_stage(base2.data(), w, h2.data(), o2.data());

        for (size_t j = 0; j < Kb; ++j) {
            const float* o_slot = &o2[j * W];
            const float* h_slot = &h2[j * H];

            matmul_bf16(o_slot, lm_head, logits.data(), W, V);
            apply_markov_correction(logits.data(), prev_tok, w);

            uint32_t chosen = 0;
            uint32_t* slot_cands = out_cand_tree ? &out_cand_tree[(Kb + j) * K] : nullptr;
            float* slot_scores = out_score_tree ? &out_score_tree[(Kb + j) * K] : nullptr;
            extract_and_rerank_slot(logits.data(), h_slot, prev_tok, w, slot_cands, slot_scores, &chosen);

            out_tokens[Kb + j] = chosen;
            prev_tok = chosen;
        }

        if (out_proposals) {
            for (size_t s = 0; s < Kb; ++s) {
                for (size_t i = 0; i < W; ++i) {
                    out_proposals[s * W + i] = float_to_bf16(o1[s * W + i]);
                    out_proposals[(Kb + s) * W + i] = float_to_bf16(o2[s * W + i]);
                }
            }
        }
    }

    /**
     * Executes proposal generation under TP=2 (split vocabulary V/2 = 124,160 per GPU).
     * Routes drafter representations directly through the teacher LM-head shards WITHOUT
     * additional final teacher RMSNorm, applies Markov correction, and evaluates the
     * Top-16 Selector with cross-GPU gathering/argmax.
     * Enforces Discrete Two-Stage Chaining (2x6=12) with NO continuous fallback.
     */
    static void propose_candidates_tp2(
        const rad_bf16_t* context_features,        // [7680]
        const rad_bf16_t* anchor_embedding,       // [2560]
        uint32_t anchor_token,                     // Committed anchor token ID
        const rad_bf16_t* teacher_lm_head_rank0,   // [124160, 2560] Rank 0 LM-head shard
        const rad_bf16_t* teacher_lm_head_rank1,   // [124160, 2560] Rank 1 LM-head shard
        const rad_bf16_t* teacher_embedding_table, // [248320, 2560]
        const FullDrafterRankWeights& weights,
        uint32_t* out_tokens,                      // [12] proposed token IDs
        rad_bf16_t* out_proposals = nullptr,       // [12, 2560] optional continuous representations
        uint32_t* out_cand_tree = nullptr,         // [12, 16] optional top-16 candidate IDs
        float* out_score_tree = nullptr            // [12, 16] optional selector scores
    ) {
        if (!context_features || !anchor_embedding || !out_tokens) {
            throw std::invalid_argument("propose_candidates_tp2 requires non-null context_features, anchor_embedding, and out_tokens");
        }
        constexpr size_t V_SHARD = FullDrafterConfig::TP_SHARD_VOCAB;
        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
        const rad_bf16_t* head0 = teacher_lm_head_rank0 ? teacher_lm_head_rank0 : (weights.host_teacher_lm_head ? weights.host_teacher_lm_head : weights.teacher_lm_head);
        const rad_bf16_t* head1 = teacher_lm_head_rank1 ? teacher_lm_head_rank1 : (weights.host_teacher_lm_head ? weights.host_teacher_lm_head + V_SHARD * W : nullptr);
        const rad_bf16_t* wte = teacher_embedding_table ? teacher_embedding_table : (weights.host_teacher_embedding_table ? weights.host_teacher_embedding_table : weights.teacher_embedding_table);
        hipPointerAttribute_t attr0{}, attr1{}, attr_wte{};
        if (head0 && hipPointerGetAttributes(&attr0, head0) == hipSuccess && attr0.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_lm_head) head0 = weights.host_teacher_lm_head;
        } else {
            (void)hipGetLastError();
        }
        if (head1 && hipPointerGetAttributes(&attr1, head1) == hipSuccess && attr1.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_lm_head) head1 = weights.host_teacher_lm_head + V_SHARD * W;
        } else {
            (void)hipGetLastError();
        }
        if (wte && hipPointerGetAttributes(&attr_wte, wte) == hipSuccess && attr_wte.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_embedding_table) wte = weights.host_teacher_embedding_table;
        } else {
            (void)hipGetLastError();
        }
        if (!head0 || !head1 || !wte) {
            throw std::invalid_argument("propose_candidates_tp2 requires non-null teacher LM head shards and embedding table (continuous fallback eliminated)");
        }
        HostStagedWeights staged(weights);
        const auto& w = staged.get();

        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t C = FullDrafterConfig::CONTEXT_DIM;
        constexpr size_t Kb = FullDrafterConfig::BLOCK_K;
        constexpr size_t V = FullDrafterConfig::VOCAB_SIZE;
        constexpr size_t V_ACTIVE = FullDrafterConfig::ACTIVE_VOCAB;
        constexpr size_t K = FullDrafterConfig::TOP_K;

        std::vector<float> ctx_f(C), anc_f(W);
        for (size_t i = 0; i < C; ++i) ctx_f[i] = bf16_to_float(context_features[i]);
        for (size_t i = 0; i < W; ++i) anc_f[i] = bf16_to_float(anchor_embedding[i]);

        // Stage 1 Base: RMSNorm(context_proj(ctx)) + anchor_proj(anc)
        std::vector<float> proj_ctx(H), norm_ctx(H), proj_anc(H), base1(H);
        matmul_bf16(ctx_f.data(), w.context_proj, proj_ctx.data(), C, H);
        rms_norm(proj_ctx.data(), w.context_norm, norm_ctx.data(), H);
        matmul_bf16(anc_f.data(), w.anchor_proj, proj_anc.data(), W, H);
        for (size_t i = 0; i < H; ++i) base1[i] = round_bf16(norm_ctx[i] + proj_anc[i]);

        std::vector<float> h1(Kb * H), o1(Kb * W);
        execute_stage(base1.data(), w, h1.data(), o1.data());

        uint32_t prev_tok = anchor_token;
        std::vector<float> logits0(V_SHARD, 0.0f), logits1(V_SHARD, 0.0f);

        for (size_t j = 0; j < Kb; ++j) {
            const float* o_slot = &o1[j * W];
            const float* h_slot = &h1[j * H];

            matmul_bf16(o_slot, head0, logits0.data(), W, V_SHARD);
            matmul_bf16(o_slot, head1, logits1.data(), W, V_SHARD);

            // Bilinear Markov correction on each rank's shard
            apply_markov_correction(logits0.data(), prev_tok, w, V, V_ACTIVE, FullDrafterConfig::MARKOV_RANK, 0, V_SHARD);
            apply_markov_correction(logits1.data(), prev_tok, w, V, V_ACTIVE, FullDrafterConfig::MARKOV_RANK, V_SHARD, V_SHARD);

            // Cross-GPU candidate gathering, Selector evaluation, and argmax
            uint32_t chosen = 0;
            uint32_t* slot_cands = out_cand_tree ? &out_cand_tree[j * K] : nullptr;
            float* slot_scores = out_score_tree ? &out_score_tree[j * K] : nullptr;
            extract_and_rerank_slot_tp2(logits0.data(), logits1.data(), h_slot, prev_tok, w, slot_cands, slot_scores, &chosen);

            out_tokens[j] = chosen;
            prev_tok = chosen;
        }

        // Stage 2 Transition: Discrete Recurrent Boundary (slot 5 winning token T_6)
        uint32_t boundary_tok = out_tokens[Kb - 1];
        std::vector<float> base2(H);
        compute_stage2_base(
            &h1[(Kb - 1) * H],
            boundary_tok,
            wte,
            w,
            base2.data()
        );

        std::vector<float> h2(Kb * H), o2(Kb * W);
        execute_stage(base2.data(), w, h2.data(), o2.data());

        for (size_t j = 0; j < Kb; ++j) {
            const float* o_slot = &o2[j * W];
            const float* h_slot = &h2[j * H];

            matmul_bf16(o_slot, head0, logits0.data(), W, V_SHARD);
            matmul_bf16(o_slot, head1, logits1.data(), W, V_SHARD);

            apply_markov_correction(logits0.data(), prev_tok, w, V, V_ACTIVE, FullDrafterConfig::MARKOV_RANK, 0, V_SHARD);
            apply_markov_correction(logits1.data(), prev_tok, w, V, V_ACTIVE, FullDrafterConfig::MARKOV_RANK, V_SHARD, V_SHARD);

            uint32_t chosen = 0;
            uint32_t* slot_cands = out_cand_tree ? &out_cand_tree[(Kb + j) * K] : nullptr;
            float* slot_scores = out_score_tree ? &out_score_tree[(Kb + j) * K] : nullptr;
            extract_and_rerank_slot_tp2(logits0.data(), logits1.data(), h_slot, prev_tok, w, slot_cands, slot_scores, &chosen);

            out_tokens[Kb + j] = chosen;
            prev_tok = chosen;
        }

        if (out_proposals) {
            for (size_t s = 0; s < Kb; ++s) {
                for (size_t i = 0; i < W; ++i) {
                    out_proposals[s * W + i] = float_to_bf16(o1[s * W + i]);
                    out_proposals[(Kb + s) * W + i] = float_to_bf16(o2[s * W + i]);
                }
            }
        }
    }

    /**
     * Executes FullDrafter forward pass over batch of context features and anchor embeddings,
     * using Discrete Two-Stage Chaining with boundary condition wte(T_6).
     * Strictly eliminates continuous fallback representations.
     */
    static void forward(
        const rad_bf16_t* context_features, // [batch, seq_len, 7680]
        const rad_bf16_t* anchor_embeddings,// [batch, seq_len, 2560]
        rad_bf16_t* output_proposals,       // [batch, seq_len, 12, 2560]
        int batch_size,
        int seq_len,
        const FullDrafterRankWeights& weights,
        const rad_bf16_t* teacher_lm_head = nullptr,
        const rad_bf16_t* teacher_embedding_table = nullptr,
        uint32_t* out_tokens = nullptr,
        const uint32_t* anchor_tokens = nullptr
    ) {
        if (!context_features || !anchor_embeddings || !output_proposals) {
            throw std::invalid_argument("FullDrafterExecutor::forward requires non-null context_features, anchor_embeddings, and output_proposals");
        }
        if (batch_size <= 0 || seq_len <= 0) {
            throw std::invalid_argument("FullDrafterExecutor::forward requires positive batch_size and seq_len");
        }
        const rad_bf16_t* lm_head = teacher_lm_head ? teacher_lm_head : (weights.host_teacher_lm_head ? weights.host_teacher_lm_head : weights.teacher_lm_head);
        const rad_bf16_t* wte = teacher_embedding_table ? teacher_embedding_table : (weights.host_teacher_embedding_table ? weights.host_teacher_embedding_table : weights.teacher_embedding_table);
        hipPointerAttribute_t attr_lm{}, attr_wte{};
        if (lm_head && hipPointerGetAttributes(&attr_lm, lm_head) == hipSuccess && attr_lm.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_lm_head) lm_head = weights.host_teacher_lm_head;
        } else {
            (void)hipGetLastError();
        }
        if (wte && hipPointerGetAttributes(&attr_wte, wte) == hipSuccess && attr_wte.type == hipMemoryTypeDevice) {
            if (weights.host_teacher_embedding_table) wte = weights.host_teacher_embedding_table;
        } else {
            (void)hipGetLastError();
        }
        if (!lm_head || !wte) {
            throw std::invalid_argument("FullDrafterExecutor::forward requires teacher_lm_head and teacher_embedding_table (continuous fallback eliminated)");
        }

        HostStagedWeights staged(weights);
        const auto& w = staged.get();
        constexpr size_t H = FullDrafterConfig::HIDDEN_SIZE;
        constexpr size_t W = FullDrafterConfig::TARGET_WIDTH;
        constexpr size_t C = FullDrafterConfig::CONTEXT_DIM;
        constexpr size_t Kb = FullDrafterConfig::BLOCK_K;
        constexpr size_t K = FullDrafterConfig::TOTAL_K;
        constexpr size_t V = FullDrafterConfig::VOCAB_SIZE;

        size_t total_anchors = static_cast<size_t>(batch_size * seq_len);

        std::vector<float> ctx_f(C), anc_f(W);
        std::vector<float> proj_ctx(H), norm_ctx(H), proj_anc(H), base1(H);
        std::vector<float> h1(Kb * H), o1(Kb * W);
        std::vector<float> base2(H);
        std::vector<float> h2(Kb * H), o2(Kb * W);
        std::vector<float> logits(V);

        for (size_t a = 0; a < total_anchors; ++a) {
            const rad_bf16_t* ctx_ptr = context_features + a * C;
            const rad_bf16_t* anc_ptr = anchor_embeddings + a * W;

            for (size_t i = 0; i < C; ++i) ctx_f[i] = bf16_to_float(ctx_ptr[i]);
            for (size_t i = 0; i < W; ++i) anc_f[i] = bf16_to_float(anc_ptr[i]);

            // Stage 1 Base: RMSNorm(context_proj(ctx)) + anchor_proj(anc)
            matmul_bf16(ctx_f.data(), w.context_proj, proj_ctx.data(), C, H);
            rms_norm(proj_ctx.data(), w.context_norm, norm_ctx.data(), H);
            matmul_bf16(anc_f.data(), w.anchor_proj, proj_anc.data(), W, H);
            for (size_t i = 0; i < H; ++i) base1[i] = round_bf16(norm_ctx[i] + proj_anc[i]);

            // Execute Stage 1
            execute_stage(base1.data(), w, h1.data(), o1.data());

            // Slot 0..5 discrete selection to find slot 5 (T_6)
            uint32_t prev_tok = (anchor_tokens ? anchor_tokens[a] : 0);
            uint32_t chosen_t6 = 0;
            for (size_t j = 0; j < Kb; ++j) {
                const float* o_slot = &o1[j * W];
                const float* h_slot = &h1[j * H];
                matmul_bf16(o_slot, lm_head, logits.data(), W, V);
                apply_markov_correction(logits.data(), prev_tok, w);
                uint32_t chosen = 0;
                extract_and_rerank_slot(logits.data(), h_slot, prev_tok, w, nullptr, nullptr, &chosen);
                if (out_tokens) out_tokens[a * K + j] = chosen;
                prev_tok = chosen;
                if (j == Kb - 1) chosen_t6 = chosen;
            }

            // Stage 2 Transition: Discrete Recurrent Boundary (wte(T_6) + recurrent_norm(h1[5]))
            compute_stage2_base(
                &h1[(Kb - 1) * H],
                chosen_t6,
                wte,
                w,
                base2.data()
            );

            // Execute Stage 2
            execute_stage(base2.data(), w, h2.data(), o2.data());

            // Slot 6..11 discrete selection
            for (size_t j = 0; j < Kb; ++j) {
                const float* o_slot = &o2[j * W];
                const float* h_slot = &h2[j * H];
                matmul_bf16(o_slot, lm_head, logits.data(), W, V);
                apply_markov_correction(logits.data(), prev_tok, w);
                uint32_t chosen = 0;
                extract_and_rerank_slot(logits.data(), h_slot, prev_tok, w, nullptr, nullptr, &chosen);
                if (out_tokens) out_tokens[a * K + Kb + j] = chosen;
                prev_tok = chosen;
            }

            // Pack 12 proposals into BF16 output
            rad_bf16_t* out_ptr = output_proposals + a * K * W;
            for (size_t s = 0; s < Kb; ++s) {
                for (size_t i = 0; i < W; ++i) {
                    out_ptr[s * W + i] = float_to_bf16(o1[s * W + i]);
                }
            }
            for (size_t s = 0; s < Kb; ++s) {
                for (size_t i = 0; i < W; ++i) {
                    out_ptr[(Kb + s) * W + i] = float_to_bf16(o2[s * W + i]);
                }
            }
        }
    }
};

} // namespace radiance
