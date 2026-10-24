#pragma once
#include <cstdint>
#include <cstddef>

#define FULLDRAFTER_HIDDEN_SIZE       768
#define FULLDRAFTER_INTERMEDIATE_SIZE 2304
#define FULLDRAFTER_TARGET_WIDTH      2560
#define FULLDRAFTER_CONTEXT_DIM       7680
#define FULLDRAFTER_BLOCK_K           6
#define FULLDRAFTER_STAGES            2
#define FULLDRAFTER_TOTAL_K           12
#define FULLDRAFTER_VOCAB_SIZE        248320
#define FULLDRAFTER_ACTIVE_VOCAB      248077
#define FULLDRAFTER_TP_SHARD_VOCAB    124160
#define FULLDRAFTER_MARKOV_RANK       64
#define FULLDRAFTER_SELECTOR_RANK     64
#define FULLDRAFTER_TOP_K             16
#define FULLDRAFTER_CONV_KERNEL_SIZE  2
#define FULLDRAFTER_CONV_GROUP_SIZE   16
#define FULLDRAFTER_CONV_NUM_GROUPS   48
#define FULLDRAFTER_RMS_NORM_EPS      1e-6f
#define FULLDRAFTER_ROPE_THETA        10000000.0f
#define FULLDRAFTER_HEAD_DIM          64
#define FULLDRAFTER_NUM_Q_HEADS       12
#define FULLDRAFTER_NUM_KV_HEADS      4
#define FULLDRAFTER_SCR_ARRIVAL_CTR_OFF 220160
#define FULLDRAFTER_SCR_PROP_ID_OFF     220264
#define FULLDRAFTER_SCR_PROP_VAL_OFF    221032
#define FULLDRAFTER_SCR_PER_SEQ         8388608LL
#define FULLDRAFTER_P2P_MAX_SEQS        64

struct alignas(16) FullDrafterP2PSlotPacket {
    float    vals[16];
    uint32_t ids[16];
    uint32_t flag;
    float    shard_max_l;
    float    shard_sum_e;
    float    shard_sum_xe;
    uint32_t emb_flag;
    alignas(16) uint16_t emb_inbox[FULLDRAFTER_TARGET_WIDTH];
};
static_assert(sizeof(FullDrafterP2PSlotPacket) % 16 == 0 &&
              offsetof(FullDrafterP2PSlotPacket, vals) % 16 == 0 &&
              offsetof(FullDrafterP2PSlotPacket, ids) % 16 == 0 &&
              offsetof(FullDrafterP2PSlotPacket, emb_inbox) % 16 == 0,
              "P2P vector copies require 16-byte-aligned packets and payloads");

struct FullDrafterGpuPointers {
    // 1. Drafter Backbone
    const void* mask_vector;            // [768] bf16
    const void* context_proj;           // [768, 7680] bf16
    const void* context_norm;           // [768] bf16
    const void* anchor_proj;            // [768, 2560] bf16

    // 2. Transformer Block 0
    const void* b0_input_layernorm;     // [768] bf16
    const void* b0_q_proj;              // [768, 768] bf16
    const void* b0_k_proj;              // [256, 768] bf16
    const void* b0_v_proj;              // [256, 768] bf16
    const void* b0_o_proj;              // [768, 768] bf16
    const void* b0_post_attn_layernorm; // [768] bf16
    const void* b0_gate_proj;           // [2304, 768] bf16
    const void* b0_up_proj;             // [2304, 768] bf16
    const void* b0_down_proj;           // [768, 2304] bf16

    // 3. Transformer Block 1
    const void* b1_input_layernorm;     // [768] bf16
    const void* b1_q_proj;              // [768, 768] bf16
    const void* b1_k_proj;              // [256, 768] bf16
    const void* b1_v_proj;              // [256, 768] bf16
    const void* b1_o_proj;              // [768, 768] bf16
    const void* b1_post_attn_layernorm; // [768] bf16
    const void* b1_gate_proj;           // [2304, 768] bf16
    const void* b1_up_proj;             // [2304, 768] bf16
    const void* b1_down_proj;           // [768, 2304] bf16

    // 4. DFlash2 Dynamic Conv
    const void* conv_w_base;            // [2, 768] bf16
    const void* conv_w_proj;            // [96, 768] bf16

    // 5. Final Norm & Projection
    const void* final_norm;             // [768] bf16
    const void* out_proj;               // [2560, 768] bf16
    const void* recurrent_norm;         // [768] bf16

    // 6. Markov Bigram Correction
    const void* markov_matrix_a;        // [248320, 64] bf16
    const void* markov_matrix_b;        // [248320, 64] bf16

    // 7. Top-16 Selector
    const void* selector_pred_codebook; // [248320, 64] bf16
    const void* selector_succ_codebook; // [248320, 64] bf16
    const void* selector_hidden_proj;   // [64, 768] bf16

    // 8. Auxiliary Heads (optional)
    const void* confidence_norm;        // [768] bf16
    const void* confidence_fc1;         // [256, 768] bf16
    const void* confidence_fc2;         // [1, 256] bf16
    const void* verification_norm;      // [768] bf16
    const void* verification_fc1;       // [256, 768] bf16
    const void* verification_fc2;       // [1, 256] bf16

    // 9. Teacher Matrices in GPU VRAM
    const void* teacher_lm_head_rank0;   // [124160, D] bf16 (GPU 0, or active shard in TP2 P2P)
    const void* teacher_lm_head_rank1;   // [124160, D] bf16 (GPU 1, or nullptr in TP2 P2P)
    const int8_t* teacher_lm_head_q8_rank0; // [124160, D/2] packed INT4 bytes, same shard as BF16
    const int8_t* teacher_lm_head_q8_rank1; // [124160, D/2] packed INT4 bytes, or nullptr in TP2 P2P
    const float* teacher_lm_head_scale_rank0; // [124160] FP32 row scales for smoothed INT4 weights
    const float* teacher_lm_head_scale_rank1; // [124160] FP32 row scales, or nullptr in TP2 P2P
    const float* teacher_lm_head_col_scale; // [D] SmoothQuant factors: weights divide, activations multiply
    const void* teacher_embedding_table; // [248320, 2560] bf16

    // 10. Execution Geometry & TP=2 P2P Split-Vocabulary Sharding
    uint32_t  lm_head_dim;               // 2560 (standard) or 768 (fused)
    uint32_t  is_fused;                  // 0 (unfused) or 1 (fused)
    uint32_t  rank_id;                   // 0 on GPU 0, 1 on GPU 1
    uint32_t  tp2_p2p_enabled;           // 1 when TP=2 P2P vocabulary sharding is active, 0 otherwise
    void*     my_p2p_comm;               // Local P2P inbox buffer: FullDrafterP2PSlotPacket[MAX_SEQS * 12]
    void*     peer_p2p_comm;             // Peer P2P inbox buffer: FullDrafterP2PSlotPacket[MAX_SEQS * 12]
    uint32_t* d_step_seq;                // Device-resident monotonic step sequence counter
    float        dloop_theta;            // DLoop Verification Head Stage-2 survival threshold (0 = always 2 stages, >1 = Stage 1 only)
    uint32_t     dloop_pad;              // 8-byte struct alignment
    const float* markov_b_norm;          // [248320] FP32 precomputed L2 row norms of markov_matrix_b (* 1.001f) for Cauchy-Schwarz pruning
    const void*  exit_gate_net_weight;   // [1, 768] bf16 (ALoDLM-4)
    const void*  exit_gate_net_bias;     // [1] bf16 (ALoDLM-4)
    const void*  exit_gate_depth_bias;   // [4] bf16 (ALoDLM-4)
    uint32_t     is_alodlm_loop;         // 1 when model_type == "alodlm_loop", 0 when "full"
    uint32_t     alodlm_max_depth;       // default 4
    float        alodlm_tau;             // default 0.4f
    float        alodlm_q;               // default 0.5f
    float        alodlm_pos_penalty;     // default 0.0f
    uint32_t     alodlm_pad;             // requested slots; 0 uses standalone defaults; launch count overrides
};

#if __has_include(<hip/hip_runtime.h>)
#ifndef __HIP_PLATFORM_AMD__
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_AMD__
#endif
#endif
#include <hip/hip_runtime.h>
#else
typedef struct ihipStream_t* hipStream_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Milestone 1: Batched 128-bit Vectorized LM-Head Multi-GEMV
int r4d_fulldrafter_lmhead_gemv6(
    const void* d_hidden,
    const void* d_lm_head0,
    const void* d_lm_head1,
    float* d_logits_base,
    uint32_t V_shard,
    uint32_t V_total,
    uint32_t D,
    hipStream_t stream
);

// Milestone 2: Fused LDS Markov Bigram & In-LDS Top-16 Selector
int r4d_fulldrafter_slot_markov_select(
    int stage,
    int slot,
    const FullDrafterGpuPointers* p,
    char* scratch,
    uint32_t* tokens_out,
    uint32_t M,
    hipStream_t stream
);

// Milestone 3: High-Occupancy Fused Transformer Backbone (H=64)
int r4d_fulldrafter_stage1_backbone(
    const void* ktap0,
    const void* ktap1,
    const void* ktap2,
    const void* ktap_anc,
    const int32_t* token_ids,
    int block_stride,
    int total_tok,
    const FullDrafterGpuPointers* p,
    char* scratch,
    uint32_t M,
    hipStream_t stream
);

int r4d_fulldrafter_stage2_backbone(
    const FullDrafterGpuPointers* p,
    char* scratch,
    const uint32_t* tokens_out,
    uint32_t M,
    hipStream_t stream
);

// Milestone 4: Pipeline Assembly, Stream Dispatch & Latency Pass
struct RadArgs;
#ifndef RAD_TYPES_H
typedef void* RadStream;
#endif

int r4d_rad_fulldrafter_two_stage(const struct RadArgs* a, RadStream s);
int64_t r4d_rad_fulldrafter_two_stage_scratch(const struct RadArgs* a);

#ifdef __cplusplus
}
#endif
