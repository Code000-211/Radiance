// r4d_fulldrafter.hip -- FullDrafter discrete two-stage block drafter GPU implementation.
// Executes the complete 2x6=12 proposal pipeline on RDNA4 / gfx1201 device hardware.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <type_traits>

#include "r4d.h"
#include "r4d_args.h"
#include "r4d_common.h"
#include "r4d_dflash_conv_body.h"
#include "r4d_fulldrafter.h"
#include "r4d_fulldrafter_select.h"

// ==============================================================================
// Constants & Offsets
// ==============================================================================
[[maybe_unused]] static constexpr int H_DIM       = FULLDRAFTER_HIDDEN_SIZE;       // 768
[[maybe_unused]] static constexpr int W_DIM       = FULLDRAFTER_TARGET_WIDTH;      // 2560
[[maybe_unused]] static constexpr int C_DIM       = FULLDRAFTER_CONTEXT_DIM;       // 7680
[[maybe_unused]] static constexpr int INTERM_DIM  = FULLDRAFTER_INTERMEDIATE_SIZE; // 2304
[[maybe_unused]] static constexpr int BLOCK_K     = FULLDRAFTER_BLOCK_K;           // 6
[[maybe_unused]] static constexpr int V_TOTAL     = FULLDRAFTER_VOCAB_SIZE;        // 248320
[[maybe_unused]] static constexpr int V_ACT       = FULLDRAFTER_ACTIVE_VOCAB;      // 248077
[[maybe_unused]] static constexpr int V_SHARD     = FULLDRAFTER_TP_SHARD_VOCAB;    // 124160
[[maybe_unused]] static constexpr int M_RANK      = FULLDRAFTER_MARKOV_RANK;       // 64
[[maybe_unused]] static constexpr int S_RANK      = FULLDRAFTER_SELECTOR_RANK;     // 64
[[maybe_unused]] static constexpr int K_TOP       = FULLDRAFTER_TOP_K;             // 16
[[maybe_unused]] static constexpr int NUM_BLOCKS  = 240;                           // 120 per shard
[[maybe_unused]] static constexpr int SLICE_ITEMS = 1035;
static constexpr uint32_t SEL_HPROJ_READY_TAG     = 0x53454C36u;

// Per-sequence scratch layout (total 8MB per sequence):
// base1:      [768] float         offset 0
// h1:         [7, 768] float      offset 4096
// o1:         [6, 2560] float     offset 32768
// base2:      [768] float         offset 98304
// h2:         [7, 768] float      offset 102400
// o2:         [6, 2560] float     offset 131072
// prev_tokens:[16] uint32_t       offset 196608
// block_cands:[64, 16] uint32_t   offset 200704
// block_unary:[64, 16] float      offset 216064 (ends at 220160)
// arrival_ctr:[1] uint32_t        offset 220160 (ends at 220164)
// prop_ids:   [12, 16] uint32_t   offset 220264 (ends at 221032)
// prop_vals:  [12, 16] float      offset 221032 (ends at 221800)
// temp:       remainder (~1.8MB)  offset 232000
// base_logits:[6, 248320] float   offset 2097152 (ends at 8056832)

#define SCR_BASE1_OFF       0
#define SCR_H1_OFF          4096
#define SCR_O1_OFF          32768
#define SCR_BASE2_OFF       98304
#define SCR_H2_OFF          102400
#define SCR_O2_OFF          131072
#define SCR_PREV_TOK_OFF    196608
#define SCR_BLK_CANDS_OFF   200704
#define SCR_BLK_UNARY_OFF   216064
#define SCR_ARRIVAL_CTR_OFF 220160
#define SCR_PROP_ID_OFF     220264
#define SCR_PROP_VAL_OFF    221032
#define SCR_TEMP_OFF        232000
#define SCR_BASE_LOGITS_OFF 2097152LL
#define SCR_PER_SEQ         8388608LL

static constexpr size_t   DLOOP_VERIF_SURV_OFF    = SCR_TEMP_OFF + 90112; // float[12] survival probabilities (past 86KB s_temp)
static constexpr size_t   DLOOP_STAGE2_ACTIVE_OFF = SCR_TEMP_OFF + 90160; // uint32_t: 1 = run Stage 2, 0 = skip Stage 2
static constexpr size_t   DLOOP_VERIF_READY_OFF   = SCR_TEMP_OFF + 90164; // uint32_t magic tag
static constexpr uint32_t DLOOP_VERIF_MAGIC       = 0xD1009A7Eu;
static constexpr size_t   UNARY_BUCKET_OFF        = SCR_TEMP_OFF + 92160; // uint64_t[12 * 256] per-slot unary bucket maxima (24 KB)
static constexpr size_t   UNARY_BUCKET_READY_OFF  = SCR_TEMP_OFF + 116736; // uint32_t magic tag
static constexpr uint32_t UNARY_BUCKET_MAGIC      = 0x554E4136u;

// ALoDLM-4 scratch offsets (inside unused [363072 .. 2097152) region of 8 MB per-sequence scratch)
static constexpr int      ALODLM_L                = 13; // 1 base + 12 proposal slots
static constexpr size_t   ALODLM_NORM_OFF         = SCR_TEMP_OFF + 12288;  // float[13 * 768] = 39,936 B (ends at SCR_TEMP_OFF + 52224)
static constexpr size_t   ALODLM_LATENT_OFF       = SCR_TEMP_OFF + 52224;  // float[12 * 768] = 36,864 B (ends at SCR_TEMP_OFF + 89088 < 90112)
static constexpr size_t   ALODLM_H_OFF            = SCR_TEMP_OFF + 120832; // float[13 * 768] = 39,936 B (ends at SCR_TEMP_OFF + 160768)
static constexpr size_t   ALODLM_QKV_OFF          = SCR_TEMP_OFF + 160768; // float[13 * 2304] = 119,808 B (ends at SCR_TEMP_OFF + 280576)
static constexpr size_t   ALODLM_IDS_OFF          = SCR_TEMP_OFF + 280576; // uint32_t[12] committed token IDs
static constexpr size_t   ALODLM_PRED_OFF         = SCR_TEMP_OFF + 280624; // uint32_t[12] current depth predicted token IDs
static constexpr size_t   ALODLM_EXITS_OFF        = SCR_TEMP_OFF + 280672; // uint32_t[12] exit depth (0 = unresolved, 1..4 = committed)
static constexpr size_t   ALODLM_SURV_OFF         = SCR_TEMP_OFF + 280720; // float[12] survival probabilities
static constexpr size_t   ALODLM_ENTROPY_OFF      = SCR_TEMP_OFF + 280768; // float[12] per-slot entropy
static constexpr size_t   ALODLM_DEPTHS_OFF       = SCR_TEMP_OFF + 280816; // uint32_t[1] halted depth (1..4)
static constexpr size_t   ALODLM_BLK_ENT_OFF      = SCR_TEMP_OFF + 280832; // float[64 * 4] per-block (max_l, sum_e, sum_xe, pad)
static constexpr size_t   ALODLM_EMB_CACHE_OFF    = SCR_TEMP_OFF + 281856; // float[12 * 768] cached anchor_proj(teacher_embed(ids[j])) across depths
static constexpr size_t   ALODLM_SLOT_EPOCH_OFF   = SCR_TEMP_OFF + 318720; // uint32_t[1] persistent multi-slot grid barrier epoch
static constexpr size_t   ALODLM_MCACHE_TOK_OFF   = SCR_TEMP_OFF + 318752; // uint32_t[3] cached prev_tok IDs for Markov dot-product cache (Slots 0, 1, 2)
static constexpr size_t   ALODLM_MCACHE_VAL_OFF   = SCR_TEMP_OFF + 318848; // float[3 * 124160] (1,489,920 B, ends at 2,040,768 < SCR_BASE_LOGITS_OFF)
// Packed LM-head inputs live after the Markov cache, before base logits.
static constexpr size_t LMHEAD_IN_OFF = ALODLM_MCACHE_VAL_OFF + 3 * V_SHARD * sizeof(float);
static constexpr size_t LMHEAD_SCALE_OFF = LMHEAD_IN_OFF + FULLDRAFTER_TOTAL_K * W_DIM;
static_assert(LMHEAD_SCALE_OFF + FULLDRAFTER_TOTAL_K * sizeof(float) <= SCR_BASE_LOGITS_OFF);
static constexpr size_t ALODLM_CONFIDENCE_OFF = LMHEAD_SCALE_OFF + FULLDRAFTER_TOTAL_K * sizeof(float);
static constexpr size_t ALODLM_CONFIDENCE_ARRIVAL_OFF = ALODLM_CONFIDENCE_OFF + 4 * sizeof(float);
static_assert(ALODLM_CONFIDENCE_ARRIVAL_OFF + sizeof(uint32_t) <= SCR_BASE_LOGITS_OFF);

__device__ __forceinline__ bool dloop_stage2_skipped(const char* s_base) {
    return (*(const uint32_t*)(s_base + DLOOP_VERIF_READY_OFF) == DLOOP_VERIF_MAGIC) &&
           (*(const uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF) == 0u);
}

// ==============================================================================
// Math Helpers
// ==============================================================================
typedef unsigned int r4d_u32x4 __attribute__((ext_vector_type(4)));
typedef unsigned int r4d_u32x2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float round_bf16_val(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    uint16_t h = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    return __builtin_bit_cast(float, (uint32_t)h << 16);
}

__device__ __forceinline__ float r4d_dot2_f32_bf16(uint32_t a, uint32_t b, float c = 0.0f) {
    float d;
    asm("v_dot2_f32_bf16 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}

__device__ __forceinline__ uint32_t pack_f32_to_bf16x2(float f0, float f1) {
    uint32_t u0 = __builtin_bit_cast(uint32_t, f0);
    uint32_t u1 = __builtin_bit_cast(uint32_t, f1);
    uint16_t h0 = (uint16_t)((u0 + 0x7fffu + ((u0 >> 16) & 1u)) >> 16);
    uint16_t h1 = (uint16_t)((u1 + 0x7fffu + ((u1 >> 16) & 1u)) >> 16);
    return (uint32_t)h0 | ((uint32_t)h1 << 16);
}

__device__ __forceinline__ uint32_t float_to_ordered_u32(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
}

__device__ __forceinline__ float ordered_u32_to_float(uint32_t ord) {
    uint32_t u = (ord & 0x80000000u) ? (ord ^ 0x80000000u) : (~ord);
    return __builtin_bit_cast(float, u);
}

__device__ inline float block_reduce_sum(float val) {
    __shared__ float s_sum[8];
    int lane = threadIdx.x & 31;
    int wave = threadIdx.x >> 5;
    val = r4d_wave_sum(val);
    if (lane == 0) s_sum[wave] = val;
    __syncthreads();
    float sum = 0.0f;
    if (wave == 0) {
        float wval = (lane < 8) ? s_sum[lane] : 0.0f;
        sum = r4d_wave_sum(wval);
    }
    if (lane == 0 && wave == 0) s_sum[0] = sum;
    __syncthreads();
    return s_sum[0];
}

__device__ inline void dev_rmsnorm(
    const float* x, const void* w_bf16, float* out, int N
) {
    const uint16_t* w = (const uint16_t*)w_bf16;
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = x[i];
        local_sum += v * v;
    }
    float sum = block_reduce_sum(local_sum);
    float scale = rsqrtf(sum / static_cast<float>(N) + FULLDRAFTER_RMS_NORM_EPS);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = x[i] * scale;
        if (w) val *= bf16_to_f32(w[i]);
        out[i] = val;
    }
    __syncthreads();
}

__device__ inline void dev_gemv_dense(
    const float* in, const void* w_bf16, float* out, int K, int M
) {
    const uint16_t* w = (const uint16_t*)w_bf16;
    for (int m = threadIdx.x; m < M; m += blockDim.x) {
        const uint16_t* w_row = w + (size_t)m * K;
        float dot = 0.0f;
        #pragma unroll 4
        for (int k = 0; k < K; k += 8) {
            uint4 raw = *reinterpret_cast<const uint4*>(w_row + k);
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot += in[k + i] * bf16_to_f32(h[i]);
            }
        }
        out[m] = round_bf16_val(dot);
    }
    __syncthreads();
}

// ==============================================================================
// Kernel 1: Stage 1 Multi-CU Grid Parallelization (All 64 CUs on Dual R9700)
// ==============================================================================

// 1. Stage 1 Tap Prep: Context Proj (7680 -> 768) and Anchor Proj (2560 -> 768)
// Grid: dim3(96, M), Block: dim3(256) -> 96 * 8 = 768 waves across 64 CUs (1.5 WGs/CU)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fulldrafter_stage1_backbone_kernel(
    const uint16_t* __restrict__ ktap0,
    const uint16_t* __restrict__ ktap1,
    const uint16_t* __restrict__ ktap2,
    const uint16_t* __restrict__ ktap_anc,
    const int32_t*  __restrict__ token_ids,
    int block_stride,
    int total_tok,
    const FullDrafterGpuPointers* __restrict__ p,
    char*           __restrict__ scratch
) {
    const int s = blockIdx.y;
    const int global_row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int tid = threadIdx.x;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;

    const int anc_idx = s * block_stride;
    const uint32_t anc_tok = (token_ids != nullptr && anc_idx >= 0 && anc_idx < total_tok)
                                 ? (uint32_t)token_ids[anc_idx]
                                 : 0u;
    if (tid == 0 && blockIdx.x == 0) {
        *(uint32_t*)(s_base + SCR_PREV_TOK_OFF) = anc_tok;
        *(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u;
        *(uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF) = 1u;
        *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF) = 0u;
        if (s == 0 && p->d_step_seq != nullptr) {
            *p->d_step_seq += 1u;
        }
    }

    if (global_row >= H_DIM) return;

    // Context projection: row global_row of [768, 7680]
    const uint16_t* k0 = ktap0 + (size_t)s * W_DIM;
    const uint16_t* k1 = ktap1 + (size_t)s * W_DIM;
    const uint16_t* k2 = ktap2 + (size_t)s * W_DIM;
    const uint16_t* w_ctx = (const uint16_t*)p->context_proj;
    const uint16_t* w_row = w_ctx + (size_t)global_row * C_DIM;
    const uint16_t* w_k0 = w_row;
    const uint16_t* w_k1 = w_row + W_DIM;
    const uint16_t* w_k2 = w_row + 2 * W_DIM;

    float dot_ctx = 0.0f;
    for (int col = lane * 8; col < W_DIM; col += 32 * 8) {
        r4d_u32x4 rk0 = *reinterpret_cast<const r4d_u32x4*>(k0 + col);
        r4d_u32x4 rw0 = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_k0 + col));
        r4d_u32x4 rk1 = *reinterpret_cast<const r4d_u32x4*>(k1 + col);
        r4d_u32x4 rw1 = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_k1 + col));
        r4d_u32x4 rk2 = *reinterpret_cast<const r4d_u32x4*>(k2 + col);
        r4d_u32x4 rw2 = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_k2 + col));
        #pragma unroll
        for (int p_i = 0; p_i < 4; ++p_i) {
            float p0 = r4d_dot2_f32_bf16(rk0[p_i], rw0[p_i], 0.0f);
            float p1 = r4d_dot2_f32_bf16(rk1[p_i], rw1[p_i], 0.0f);
            float p2 = r4d_dot2_f32_bf16(rk2[p_i], rw2[p_i], 0.0f);
            dot_ctx += (p0 + p1) + p2;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_ctx += __shfl_down(dot_ctx, offset, 32);
    }

    // Anchor projection: in TP2 P2P mode, ktap_anc holds the exact anchor embedding from Trunk m.w_tok + op_embed_ar
    const uint16_t* k_anc = (p->tp2_p2p_enabled == 0u && p->teacher_embedding_table != nullptr && token_ids != nullptr && anc_tok < (uint32_t)V_TOTAL)
                                ? (((const uint16_t*)p->teacher_embedding_table) + (size_t)anc_tok * W_DIM)
                                : (ktap_anc + (size_t)s * W_DIM);
    const uint16_t* w_anc = (const uint16_t*)p->anchor_proj;
    const uint16_t* w_a_row = w_anc + (size_t)global_row * W_DIM;
    float dot_anc = 0.0f;
    for (int col = lane * 8; col < W_DIM; col += 32 * 8) {
        r4d_u32x4 rk = *reinterpret_cast<const r4d_u32x4*>(k_anc + col);
        r4d_u32x4 rw = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_a_row + col));
        #pragma unroll
        for (int p_idx = 0; p_idx < 4; ++p_idx) {
            dot_anc = r4d_dot2_f32_bf16(rk[p_idx], rw[p_idx], dot_anc);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_anc += __shfl_down(dot_anc, offset, 32);
    }

    if (lane == 0) {
        float* s_temp = (float*)(s_base + SCR_TEMP_OFF);
        s_temp[global_row] = round_bf16_val(dot_ctx);
        s_temp[2 * H_DIM + global_row] = round_bf16_val(dot_anc);
    }
}

// 2. Stage 1 Base Commit: Context RMSNorm + Anchor Sum + Mask Vector Broadcast
// Grid: dim3(1, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fulldrafter_stage1_base_commit_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch
) {
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;

    float* s_base1 = (float*)(s_base + SCR_BASE1_OFF);
    float* s_h1    = (float*)(s_base + SCR_H1_OFF);
    float* s_temp  = (float*)(s_base + SCR_TEMP_OFF);

    // Context RMSNorm: s_temp[0..H_DIM-1] -> s_temp[H_DIM..2*H_DIM-1]
    dev_rmsnorm(s_temp, p->context_norm, s_temp + H_DIM, H_DIM);

    // Base1 = norm_ctx + proj_anc
    for (int h = tid; h < H_DIM; h += blockDim.x) {
        s_base1[h] = round_bf16_val(s_temp[H_DIM + h] + s_temp[2 * H_DIM + h]);
        s_h1[0 * H_DIM + h] = s_base1[h];
    }
    // Fill learned mask vector into proposal slots 1..6
    const uint16_t* w_mask = (const uint16_t*)p->mask_vector;
    for (int t = 1; t <= BLOCK_K; ++t) {
        for (int h = tid; h < H_DIM; h += blockDim.x) {
            s_h1[t * H_DIM + h] = bf16_to_f32(w_mask[h]);
        }
    }
}

// 3. DFlash2 Dynamic 1D Grouped Depthwise Convolution + Optional Fused Pre-RMSNorm
// Parallelizes all 7*96=672 (token, channel-group) projections across all 256 threads in LDS.
// Grid: dim3(1, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_block_dflash_conv_kernel(
    float*          __restrict__ s_h,         // [7, 768] in scratch
    float*          __restrict__ prev_buf,    // [7, 768] temp/norm_out buffer in scratch
    const uint16_t* __restrict__ conv_base,   // [2, 768] bf16
    const uint16_t* __restrict__ conv_proj,   // [96, 768] bf16
    int M,
    const void*     __restrict__ in_norm_w = nullptr
) {
    (void)M;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    float* h_base = (float*)((char*)s_h + (size_t)s * SCR_PER_SEQ);
    float* p_base = (float*)((char*)prev_buf + (size_t)s * SCR_PER_SEQ);

    __shared__ float s_p[7 * H_DIM];
    __shared__ float s_delta[7 * 96];

    for (int i = tid; i < 7 * H_DIM; i += blockDim.x) {
        s_p[i] = h_base[i];
    }
    __syncthreads();

    if (tid < 96) {
        const uint16_t* w_row = conv_proj + (size_t)tid * H_DIM;
        float dot[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma nounroll
        for (int i = 0; i < H_DIM; i += 8) {
            uint4 r_w = *reinterpret_cast<const uint4*>(w_row + i);
            const uint16_t* hw = reinterpret_cast<const uint16_t*>(&r_w);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float wv = bf16_to_f32(hw[j]);
                int col = i + j;
                #pragma unroll
                for (int t = 0; t < 7; ++t) {
                    dot[t] += s_p[t * H_DIM + col] * wv;
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            s_delta[t * 96 + tid] = round_bf16_val(dot[t]);
        }
    }
    __syncthreads();

    for (int c = tid; c < H_DIM; c += blockDim.x) {
        int g = c / FULLDRAFTER_CONV_GROUP_SIZE;
        float cb0 = bf16_to_f32(conv_base[0 * H_DIM + c]);
        float cb1 = bf16_to_f32(conv_base[1 * H_DIM + c]);
        float p_prev = 0.0f;
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            float p_curr = s_p[t * H_DIM + c];
            float w0 = round_bf16_val(cb0 + s_delta[t * 96 + 0 * 48 + g]);
            float w1 = round_bf16_val(cb1 + s_delta[t * 96 + 1 * 48 + g]);
            float tap0 = round_bf16_val(w0 * p_curr);
            float tap1 = (t >= 1) ? round_bf16_val(w1 * p_prev) : 0.0f;
            float val = round_bf16_val(tap0 + tap1);
            h_base[t * H_DIM + c] = val;
            s_p[t * H_DIM + c] = val;
            p_prev = p_curr;
        }
    }

    if (in_norm_w) {
        __syncthreads();
        for (int t = 0; t < 7; ++t) {
            dev_rmsnorm(s_p + t * H_DIM, in_norm_w, p_base + t * H_DIM, H_DIM);
        }
    }
}

// 4. Token RMSNorm: 1 Workgroup per Token
// Grid: dim3(N_TOKENS, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_token_rmsnorm_kernel(
    const float*    __restrict__ in_x,   // [N_TOKENS, 768]
    const void*     __restrict__ w_bf16, // [768] bf16
    float*          __restrict__ out_y,  // [N_TOKENS, 768]
    int N_TOKENS,
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    const int t = blockIdx.x;
    if (t >= N_TOKENS) return;

    const float* x = (const float*)((const char*)in_x + (size_t)s * SCR_PER_SEQ) + (size_t)t * H_DIM;
    float* y       = (float*)((char*)out_y + (size_t)s * SCR_PER_SEQ) + (size_t)t * H_DIM;
    const uint16_t* w = (const uint16_t*)w_bf16;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < H_DIM; i += blockDim.x) {
        float v = x[i];
        local_sum += v * v;
    }
    float sum = block_reduce_sum(local_sum);
    float scale = rsqrtf(sum / static_cast<float>(H_DIM) + FULLDRAFTER_RMS_NORM_EPS);

    for (int i = threadIdx.x; i < H_DIM; i += blockDim.x) {
        float val = x[i] * scale;
        if (w) val *= bf16_to_f32(w[i]);
        y[i] = val;
    }
}

// 5. Token-Batched QKV Multi-GEMV across 160 Workgroups / 64 CUs (2.5 WGs/CU)
// Grid: dim3(160, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_block_qkv_kernel(
    const float*    __restrict__ in_norm_x, // [7, 768] in scratch
    const uint16_t* __restrict__ w_q,       // [768, 768] bf16
    const uint16_t* __restrict__ w_k,       // [256, 768] bf16
    const uint16_t* __restrict__ w_v,       // [256, 768] bf16
    float*          __restrict__ out_qkv,   // [7, 1280] in scratch (q:0..767, k:768..1023, v:1024..1279)
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    const int global_row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (global_row >= 1280) return;

    const uint16_t* w_row;
    if (global_row < 768) {
        w_row = w_q + (size_t)global_row * 768;
    } else if (global_row < 1024) {
        w_row = w_k + (size_t)(global_row - 768) * 768;
    } else {
        w_row = w_v + (size_t)(global_row - 1024) * 768;
    }

    __shared__ float s_x[7 * 768]; // 21 KB LDS -> 16 Waves/SIMD, eliminates 8x redundant global reads per WG
    const float* x_base = (const float*)((const char*)in_norm_x + (size_t)s * SCR_PER_SEQ);
    for (int i = threadIdx.x; i < 7 * 768; i += 256) {
        s_x[i] = x_base[i];
    }
    __syncthreads();

    float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        int col = (it * 32 + lane) * 8;
        r4d_u32x4 raw = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_row + col));
        const uint16_t* hw = reinterpret_cast<const uint16_t*>(&raw);
        for (int i = 0; i < 8; ++i) {
            float w_val = bf16_to_f32(hw[i]);
            int idx = col + i;
            #pragma unroll
            for (int t = 0; t < 7; ++t) {
                acc[t] += w_val * s_x[t * 768 + idx];
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int t = 0; t < 7; ++t) {
            acc[t] += __shfl_down(acc[t], offset, 32);
        }
    }

    if (lane == 0) {
        float* qkv_s = (float*)((char*)out_qkv + (size_t)s * SCR_PER_SEQ);
        for (int t = 0; t < 7; ++t) {
            qkv_s[t * 1280 + global_row] = round_bf16_val(acc[t]);
        }
    }
}

// 6. In-Register Attention (H=64, N_q=12, N_kv=4, L=7)
// 1 Wave per Head, Native Wave32 Lane Mapping (d0=l, d1=l+32), V_DOT2_F32_BF16 + __shfl_xor Wave Reduction
// Zero DRAM / LDS Score Writes, In-Register Softmax & Value Accumulation
// Grid: dim3(12, M), Block: dim3(32)
template <bool CAUSAL = true>
__global__ __launch_bounds__(32) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_block_attention_kernel(
    const float* __restrict__ in_qkv,   // [7, 1280] in scratch
    float*       __restrict__ out_attn, // [7, 768] in scratch
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    const int q_h = blockIdx.x; // 0..11
    if (q_h >= 12) return;
    const int kv_h = q_h / 3;   // 0..3
    const int lane = threadIdx.x; // 0..31

    const float* qkv_s = (const float*)((const char*)in_qkv + (size_t)s * SCR_PER_SEQ);
    float* attn_s      = (float*)((char*)out_attn + (size_t)s * SCR_PER_SEQ);

    uint32_t q_pk[7], k_pk[7];
    float v0[7], v1[7];
    const float freq = 1.0f / powf(FULLDRAFTER_ROPE_THETA, (2.0f * (float)lane) / 64.0f);

    for (int t = 0; t < 7; ++t) {
        const float* q_tok = qkv_s + t * 1280 + q_h * 64;
        const float* k_tok = qkv_s + t * 1280 + 768 + kv_h * 64;
        const float* v_tok = qkv_s + t * 1280 + 1024 + kv_h * 64;

        float q0 = q_tok[lane], q1 = q_tok[lane + 32];
        float k0 = k_tok[lane], k1 = k_tok[lane + 32];
        v0[t]    = v_tok[lane]; v1[t] = v_tok[lane + 32];

        float theta = (float)t * freq;
        float cos_th = cosf(theta), sin_th = sinf(theta);
        float qr0 = round_bf16_val(q0 * cos_th - q1 * sin_th);
        float qr1 = round_bf16_val(q0 * sin_th + q1 * cos_th);
        float kr0 = round_bf16_val(k0 * cos_th - k1 * sin_th);
        float kr1 = round_bf16_val(k0 * sin_th + k1 * cos_th);

        q_pk[t] = pack_f32_to_bf16x2(qr0, qr1);
        k_pk[t] = pack_f32_to_bf16x2(kr0, kr1);
    }

    for (int t = 0; t < 7; ++t) {
        float scores[7];
        float max_score = -1e9f;
        int p_max = CAUSAL ? t : 6;
        for (int p_idx = 0; p_idx <= p_max; ++p_idx) {
            float dot = r4d_dot2_f32_bf16(q_pk[t], k_pk[p_idx], 0.0f);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                dot += __shfl_xor(dot, mask, 32);
            }
            dot = dot * 0.125f; // 1 / sqrt(64)
            scores[p_idx] = dot;
            if (dot > max_score) max_score = dot;
        }

        float exp_sum = 0.0f;
        for (int p_idx = 0; p_idx <= p_max; ++p_idx) {
            scores[p_idx] = expf(scores[p_idx] - max_score);
            exp_sum += scores[p_idx];
        }

        float acc0 = 0.0f, acc1 = 0.0f;
        for (int p_idx = 0; p_idx <= p_max; ++p_idx) {
            float prob = round_bf16_val(scores[p_idx] / exp_sum);
            acc0 += prob * v0[p_idx];
            acc1 += prob * v1[p_idx];
        }

        attn_s[t * 768 + q_h * 64 + lane]      = round_bf16_val(acc0);
        attn_s[t * 768 + q_h * 64 + lane + 32] = round_bf16_val(acc1);
    }
}

// 7. Token-Batched Attention Output Projection Multi-GEMV + Residual Add
// Grid: dim3(96, M), Block: dim3(256) -> 96 * 8 = 768 waves across 64 CUs (1.5 WGs/CU)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_block_out_proj_kernel(
    const float*    __restrict__ in_attn, // [7, 768] in scratch
    const uint16_t* __restrict__ w_o,     // [768, 768] bf16
    float*          __restrict__ out_h,   // [7, 768] in scratch (residual in-place)
    int M,
    const void*     __restrict__ post_norm_w = nullptr,
    float*          __restrict__ out_norm_y  = nullptr
) {
    (void)M;
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int global_row = blockIdx.x * 8 + (tid >> 5);
    const int lane = tid & 31;

    __shared__ uint32_t s_a_pk[7 * 384];
    const float* a_base = (const float*)((const char*)in_attn + (size_t)s * SCR_PER_SEQ);
    for (int i = tid; i < 7 * 384; i += 256) {
        uint32_t u0 = __builtin_bit_cast(uint32_t, a_base[2 * i + 0]);
        uint32_t u1 = __builtin_bit_cast(uint32_t, a_base[2 * i + 1]);
        s_a_pk[i] = (u0 >> 16) | (u1 & 0xFFFF0000u);
    }
    __syncthreads();

    float* h_s = (float*)((char*)out_h + (size_t)s * SCR_PER_SEQ);
    if (global_row < 768) {
        const r4d_u32x4* w_row = reinterpret_cast<const r4d_u32x4*>(w_o + (size_t)global_row * 768);
        float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        const uint16_t* s_a_bf16 = reinterpret_cast<const uint16_t*>(s_a_pk);
        #pragma nounroll
        for (int it = 0; it < 3; ++it) {
            int vec_idx = it * 32 + lane;
            r4d_u32x4 raw = __builtin_nontemporal_load(&w_row[vec_idx]);
            const uint16_t* hw = reinterpret_cast<const uint16_t*>(&raw);
            int col = vec_idx * 8;
            for (int i = 0; i < 8; ++i) {
                float w_val = bf16_to_f32(hw[i]);
                int idx = col + i;
                #pragma unroll
                for (int t = 0; t < 7; ++t) {
                    acc[t] += w_val * bf16_to_f32(s_a_bf16[t * 768 + idx]);
                }
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            for (int t = 0; t < 7; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }

        if (lane == 0) {
            for (int t = 0; t < 7; ++t) {
                h_s[t * 768 + global_row] = round_bf16_val(h_s[t * 768 + global_row] + acc[t]);
            }
        }
    }

    if (post_norm_w != nullptr && out_norm_y != nullptr) {
        __threadfence();
        __syncthreads();
        __shared__ uint32_t s_is_last;
        if (tid == 0) {
            char* s_base = (char*)in_attn - SCR_TEMP_OFF + (size_t)s * SCR_PER_SEQ;
            unsigned int* ctr = (unsigned int*)(s_base + SCR_ARRIVAL_CTR_OFF);
            unsigned int arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (arrived == (unsigned int)(gridDim.x - 1)) {
                *ctr = 0u;
                s_is_last = 1u;
            } else {
                s_is_last = 0u;
            }
        }
        __syncthreads();
        if (s_is_last != 0u) {
            float* norm_s = (float*)((char*)out_norm_y + (size_t)s * SCR_PER_SEQ);
            for (int t = 0; t < 7; ++t) {
                dev_rmsnorm(h_s + t * H_DIM, post_norm_w, norm_s + t * H_DIM, H_DIM);
            }
        }
    }
}

// 8. Fused SwiGLU MLP Gate & Up Multi-GEMV + In-Register SiLU Activation
// Fusing Gate and Up projections in a single pass across 288 Workgroups / 64 CUs (4.5 WGs/CU)
// Grid: dim3(288, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_block_mlp_gate_up_kernel(
    const float*    __restrict__ in_norm_y, // [7, 768] in scratch
    const uint16_t* __restrict__ w_gate,    // [2304, 768] bf16
    const uint16_t* __restrict__ w_up,      // [2304, 768] bf16
    float*          __restrict__ out_act,   // [7, 2304] in scratch
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    const int wave = threadIdx.x >> 5;
    const int global_m = blockIdx.x * 8 + wave;
    const int lane = threadIdx.x & 31;

    __shared__ float s_y[7 * 768];
    __shared__ float s_silu[8][7];
    const float* y_base = (const float*)((const char*)in_norm_y + (size_t)s * SCR_PER_SEQ);
    for (int i = threadIdx.x; i < 7 * 768; i += 256) {
        s_y[i] = y_base[i];
    }
    __syncthreads();

    if (global_m >= 2304) return;
    const uint16_t* g_row = w_gate + (size_t)global_m * 768;
    const uint16_t* u_row = w_up   + (size_t)global_m * 768;

    float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        int col = (it * 32 + lane) * 8;
        r4d_u32x4 raw_g = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(g_row + col));
        const uint16_t* hg = reinterpret_cast<const uint16_t*>(&raw_g);
        for (int i = 0; i < 8; ++i) {
            float g_val = bf16_to_f32(hg[i]);
            int idx = col + i;
            #pragma unroll
            for (int t = 0; t < 7; ++t) {
                acc[t] += g_val * s_y[t * 768 + idx];
            }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            acc[t] += __shfl_down(acc[t], offset, 32);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            float g = acc[t];
            s_silu[wave][t] = g / (1.0f + expf(-g));
        }
    }

    #pragma unroll
    for (int t = 0; t < 7; ++t) acc[t] = 0.0f;

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        int col = (it * 32 + lane) * 8;
        r4d_u32x4 raw_u = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(u_row + col));
        const uint16_t* hu = reinterpret_cast<const uint16_t*>(&raw_u);
        for (int i = 0; i < 8; ++i) {
            float u_val = bf16_to_f32(hu[i]);
            int idx = col + i;
            #pragma unroll
            for (int t = 0; t < 7; ++t) {
                acc[t] += u_val * s_y[t * 768 + idx];
            }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            acc[t] += __shfl_down(acc[t], offset, 32);
        }
    }
    if (lane == 0) {
        float* act_s = (float*)((char*)out_act + (size_t)s * SCR_PER_SEQ);
        #pragma unroll
        for (int t = 0; t < 7; ++t) {
            act_s[t * 2304 + global_m] = round_bf16_val(s_silu[wave][t] * acc[t]);
        }
    }
}

// 9. Token-Batched SwiGLU MLP Down Projection Multi-GEMV + Residual Add
// Grid: dim3(96, M), Block: dim3(256) -> 96 * 8 = 768 waves across 64 CUs (1.5 WGs/CU)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_block_mlp_down_kernel(
    const float*    __restrict__ in_act,   // [7, 2304] in scratch
    const uint16_t* __restrict__ w_down,   // [768, 2304] bf16
    float*          __restrict__ out_h,    // [7, 768] in scratch (residual in-place)
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    const int global_row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (global_row >= 768) return;

    __shared__ uint32_t s_act_pk[7 * 1152]; // 31.5 KB LDS -> 16 Waves/SIMD, 0 ULP bit-identical with v_dot2_f32_bf16
    const float* act_base = (const float*)((const char*)in_act + (size_t)s * SCR_PER_SEQ);
    for (int i = threadIdx.x; i < 7 * 1152; i += 256) {
        uint32_t u0 = __builtin_bit_cast(uint32_t, act_base[2 * i + 0]);
        uint32_t u1 = __builtin_bit_cast(uint32_t, act_base[2 * i + 1]);
        s_act_pk[i] = (u0 >> 16) | (u1 & 0xFFFF0000u);
    }
    __syncthreads();

    if (global_row >= 768) return;
    const r4d_u32x4* w_row = reinterpret_cast<const r4d_u32x4*>(w_down + (size_t)global_row * 2304);
    float acc[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const uint16_t* s_act_bf16 = reinterpret_cast<const uint16_t*>(s_act_pk);
    #pragma nounroll
    for (int it = 0; it < 9; ++it) {
        int vec_idx = it * 32 + lane;
        r4d_u32x4 raw = __builtin_nontemporal_load(&w_row[vec_idx]);
        const uint16_t* hw = reinterpret_cast<const uint16_t*>(&raw);
        int col = vec_idx * 8;
        for (int i = 0; i < 8; ++i) {
            float w_val = bf16_to_f32(hw[i]);
            int idx = col + i;
            #pragma unroll
            for (int t = 0; t < 7; ++t) {
                acc[t] += w_val * bf16_to_f32(s_act_bf16[t * 2304 + idx]);
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        for (int t = 0; t < 7; ++t) {
            acc[t] += __shfl_down(acc[t], offset, 32);
        }
    }

    if (lane == 0) {
        float* h_s = (float*)((char*)out_h + (size_t)s * SCR_PER_SEQ);
        for (int t = 0; t < 7; ++t) {
            h_s[t * 768 + global_row] = round_bf16_val(h_s[t * 768 + global_row] + acc[t]);
        }
    }
}

// 10. Final RMSNorm for Proposal Slots 1..6 (supports in-place slot 1..6 -> 0..5 when s_temp == nullptr)
// Grid: dim3(6, M) when s_temp != nullptr, or dim3(1, M) when s_temp == nullptr
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_final_norm_kernel(
    float*       __restrict__ s_h,        // [7, 768] in scratch
    const void*  __restrict__ final_norm, // [768] bf16
    float*       __restrict__ s_temp,     // [6, 768] temp buffer or nullptr for in-place
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    float* h_base = (float*)((char*)s_h + (size_t)s * SCR_PER_SEQ);

    if (!s_temp) {
        for (int slot = 0; slot < BLOCK_K; ++slot) {
            dev_rmsnorm(h_base + (1 + slot) * H_DIM, final_norm, h_base + slot * H_DIM, H_DIM);
        }
        return;
    }

    const int slot = blockIdx.x; // 0..5
    if (slot >= BLOCK_K) return;
    float* t_base = (float*)((char*)s_temp + (size_t)s * SCR_PER_SEQ);
    dev_rmsnorm(h_base + (1 + slot) * H_DIM, final_norm, t_base + slot * H_DIM, H_DIM);
}

// 11. Final Copy of Proposal Slots 0..5 into Target Hidden Activations
// Grid: dim3(6, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_copy_final_norm_kernel(
    float*       __restrict__ s_h,    // [7, 768] in scratch
    const float* __restrict__ s_temp, // [6, 768] temp buffer
    int M
) {
    (void)M;
    const int s = blockIdx.y;
    const int slot = blockIdx.x; // 0..5
    if (slot >= BLOCK_K) return;

    float* h_base       = (float*)((char*)s_h + (size_t)s * SCR_PER_SEQ);
    const float* t_base = (const float*)((const char*)s_temp + (size_t)s * SCR_PER_SEQ);

    for (int i = threadIdx.x; i < H_DIM; i += blockDim.x) {
        h_base[slot * H_DIM + i] = t_base[slot * H_DIM + i];
    }
}

// 12. Host Dispatch: Stage 1 Multi-CU Transformer Backbone
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
) {
    if (!p || !scratch) return -1;
    dim3 block_256(256);

    // 1. Stage 1 Tap Prep across 96 blocks / 64 CUs
    dim3 grid_prep(96, M);
    fulldrafter_stage1_backbone_kernel<<<grid_prep, block_256, 0, stream>>>(
        (const uint16_t*)ktap0, (const uint16_t*)ktap1, (const uint16_t*)ktap2,
        (const uint16_t*)ktap_anc, token_ids, block_stride, total_tok, p, scratch
    );

    // 2. Base commit: context RMSNorm + anchor proj sum + mask broadcast
    dim3 grid_commit(1, M);
    fulldrafter_stage1_base_commit_kernel<<<grid_commit, block_256, 0, stream>>>(p, scratch);

    float* s_h1   = (float*)(scratch + SCR_H1_OFF);
    float* s_temp = (float*)(scratch + SCR_TEMP_OFF);

    // 3. Two Transformer Blocks (batched across all M sequences, fused conv + pre-RMSNorm)
    for (int b = 0; b < 2; ++b) {
        const void* conv_base   = p->conv_w_base;
        const void* conv_proj   = p->conv_w_proj;
        const void* in_norm_w   = (b == 0) ? p->b0_input_layernorm : p->b1_input_layernorm;
        const void* q_w         = (b == 0) ? p->b0_q_proj : p->b1_q_proj;
        const void* k_w         = (b == 0) ? p->b0_k_proj : p->b1_k_proj;
        const void* v_w         = (b == 0) ? p->b0_v_proj : p->b1_v_proj;
        const void* o_w         = (b == 0) ? p->b0_o_proj : p->b1_o_proj;
        const void* post_norm_w = (b == 0) ? p->b0_post_attn_layernorm : p->b1_post_attn_layernorm;
        const void* gate_w      = (b == 0) ? p->b0_gate_proj : p->b1_gate_proj;
        const void* up_w        = (b == 0) ? p->b0_up_proj : p->b1_up_proj;
        const void* down_w      = (b == 0) ? p->b0_down_proj : p->b1_down_proj;

        r4d_fulldrafter_block_dflash_conv_kernel<<<dim3(1, M), block_256, 0, stream>>>(
            s_h1, s_temp, (const uint16_t*)conv_base, (const uint16_t*)conv_proj, (int)M, in_norm_w
        );
        r4d_fulldrafter_block_qkv_kernel<<<dim3(160, M), block_256, 0, stream>>>(
            s_temp, (const uint16_t*)q_w, (const uint16_t*)k_w, (const uint16_t*)v_w,
            s_temp + 7 * H_DIM, (int)M
        );
        r4d_fulldrafter_block_attention_kernel<true><<<dim3(12, M), dim3(32), 0, stream>>>(
            s_temp + 7 * H_DIM, s_temp, (int)M
        );
        r4d_fulldrafter_block_out_proj_kernel<<<dim3(96, M), block_256, 0, stream>>>(
            s_temp, (const uint16_t*)o_w, s_h1, (int)M, post_norm_w, s_temp
        );
        r4d_fulldrafter_block_mlp_gate_up_kernel<<<dim3(288, M), block_256, 0, stream>>>(
            s_temp, (const uint16_t*)gate_w, (const uint16_t*)up_w, s_temp + 7 * H_DIM, (int)M
        );
        r4d_fulldrafter_block_mlp_down_kernel<<<dim3(96, M), block_256, 0, stream>>>(
            s_temp + 7 * H_DIM, (const uint16_t*)down_w, s_h1, (int)M
        );
    }

    // 4. Final RMSNorm for proposal slots 1..6 -> slots 0..5 in-place in s_h1
    r4d_fulldrafter_final_norm_kernel<<<dim3(1, M), block_256, 0, stream>>>(
        s_h1, p->final_norm, nullptr, (int)M
    );
    return 0;
}

// ==============================================================================
// Kernel: Fast Coalesced Out Projection + Fused Selector Hidden Projection
// Blocks 0..319 compute out_proj [6, 2560]; Block 320 computes selector_hidden_proj [6, 64]
// ==============================================================================
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fast_out_proj_kernel(
    const void* __restrict__ out_proj_w,
    char* __restrict__ scratch,
    size_t in_h_offset,
    size_t out_o_offset,
    const void* __restrict__ sel_hproj_w = nullptr,
    const FullDrafterGpuPointers* __restrict__ p = nullptr,
    uint32_t* __restrict__ tokens_out = nullptr,
    int M_batch = 1
) {
    const int s = blockIdx.y; // Sequence index
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    if ((in_h_offset == SCR_H2_OFF || (p && p->is_alodlm_loop != 0u)) && dloop_stage2_skipped(s_base)) {
        return;
    }
    const float* s_h = (const float*)(s_base + in_h_offset);
    const float* h0 = s_h + 0 * H_DIM;
    const float* h1 = s_h + 1 * H_DIM;
    const float* h2 = s_h + 2 * H_DIM;
    const float* h3 = s_h + 3 * H_DIM;
    const float* h4 = s_h + 4 * H_DIM;
    const float* h5 = s_h + 5 * H_DIM;

    // Block 321: Native DLoop Early-Exit Verification Head (768 -> RMSNorm -> FC1(256) -> SiLU -> FC2(1) -> Sigmoid)
    if (blockIdx.x == 321) {
        const int tid = threadIdx.x;
        const int stage_idx = (in_h_offset == SCR_H2_OFF) ? 1 : 0;
        float* surv_out = (float*)(s_base + DLOOP_VERIF_SURV_OFF) + stage_idx * BLOCK_K;
        if (!p || !p->verification_norm || !p->verification_fc1 || !p->verification_fc2) {
            if (tid < BLOCK_K) surv_out[tid] = 1.0f;
            if (stage_idx == 0 && tid == 0) {
                const bool skip_s2 = (M_batch >= 5);
                *(uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF) = skip_s2 ? 0u : 1u;
                *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF)  = DLOOP_VERIF_MAGIC;
                if (skip_s2 && tokens_out != nullptr) {
                    for (int k = BLOCK_K; k < FULLDRAFTER_TOTAL_K; ++k) {
                        tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + k] = r4d::FULLDRAFTER_SENTINEL_ID;
                    }
                }
            }
            return;
        }
        __shared__ uint32_t s_vnorm_pk[BLOCK_K * (H_DIM / 2)];
        const uint16_t* v_norm = (const uint16_t*)p->verification_norm;
        for (int slot = 0; slot < BLOCK_K; ++slot) {
            const float* hs = s_h + slot * H_DIM;
            float sq = 0.0f;
            for (int i = tid; i < H_DIM; i += 256) {
                float v = hs[i];
                sq += v * v;
            }
            float sum_sq = block_reduce_sum(sq);
            float scale = rsqrtf(sum_sq / static_cast<float>(H_DIM) + FULLDRAFTER_RMS_NORM_EPS);
            for (int p_idx = tid; p_idx < (H_DIM / 2); p_idx += 256) {
                int c0 = 2 * p_idx;
                float x0 = hs[c0 + 0] * scale * bf16_to_f32(v_norm[c0 + 0]);
                float x1 = hs[c0 + 1] * scale * bf16_to_f32(v_norm[c0 + 1]);
                s_vnorm_pk[slot * (H_DIM / 2) + p_idx] = pack_f32_to_bf16x2(x0, x1);
            }
            __syncthreads();
        }
        // tid (0..255) computes row `tid` of verification_fc1 [256, 768] across all 6 slots
        const r4d_u32x4* fc1_row = (const r4d_u32x4*)((const uint16_t*)p->verification_fc1 + (size_t)tid * H_DIM);
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f;
        #pragma nounroll
        for (int k4 = 0; k4 < (H_DIM / 8); ++k4) {
            r4d_u32x4 wv = fc1_row[k4];
            int p_base = k4 * 4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t w_pk = wv[j];
                acc0 = r4d_dot2_f32_bf16(w_pk, s_vnorm_pk[0 * 384 + p_base + j], acc0);
                acc1 = r4d_dot2_f32_bf16(w_pk, s_vnorm_pk[1 * 384 + p_base + j], acc1);
                acc2 = r4d_dot2_f32_bf16(w_pk, s_vnorm_pk[2 * 384 + p_base + j], acc2);
                acc3 = r4d_dot2_f32_bf16(w_pk, s_vnorm_pk[3 * 384 + p_base + j], acc3);
                acc4 = r4d_dot2_f32_bf16(w_pk, s_vnorm_pk[4 * 384 + p_base + j], acc4);
                acc5 = r4d_dot2_f32_bf16(w_pk, s_vnorm_pk[5 * 384 + p_base + j], acc5);
            }
        }
        const float w2 = bf16_to_f32(((const uint16_t*)p->verification_fc2)[tid]);
        float p_slot[BLOCK_K];
        p_slot[0] = (acc0 / (1.0f + expf(-acc0))) * w2;
        p_slot[1] = (acc1 / (1.0f + expf(-acc1))) * w2;
        p_slot[2] = (acc2 / (1.0f + expf(-acc2))) * w2;
        p_slot[3] = (acc3 / (1.0f + expf(-acc3))) * w2;
        p_slot[4] = (acc4 / (1.0f + expf(-acc4))) * w2;
        p_slot[5] = (acc5 / (1.0f + expf(-acc5))) * w2;

        float sv[BLOCK_K];
        for (int slot = 0; slot < BLOCK_K; ++slot) {
            float logit = block_reduce_sum(p_slot[slot]);
            sv[slot] = 1.0f / (1.0f + expf(-logit));
        }
        if (tid == 0) {
            for (int slot = 0; slot < BLOCK_K; ++slot) {
                surv_out[slot] = sv[slot];
            }
            if (stage_idx == 0) {
                const float theta = p->dloop_theta;
                const bool skip_s2 = (M_batch >= 5) ||
                                     (theta > 0.0f && (sv[0] < theta || sv[1] < theta || sv[2] < theta ||
                                                       sv[3] < theta || sv[4] < theta || sv[5] < theta));
                *(uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF) = skip_s2 ? 0u : 1u;
                *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF)  = DLOOP_VERIF_MAGIC;
                if (skip_s2 && tokens_out != nullptr) {
                    for (int k = BLOCK_K; k < FULLDRAFTER_TOTAL_K; ++k) {
                        tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + k] = r4d::FULLDRAFTER_SENTINEL_ID;
                    }
                }
            }
        }
        return;
    }

    // Block 320: Precompute selector_hidden_proj ([64, 768] @ [6, 768]^T -> [6, 64]) + zero unary buckets
    if (blockIdx.x == 320) {
        const int tid = threadIdx.x;
        uint64_t* u_buckets = (uint64_t*)(s_base + UNARY_BUCKET_OFF);
        #pragma unroll
        for (int s_slot = 0; s_slot < BLOCK_K; ++s_slot) {
            u_buckets[s_slot * 256 + tid] = 0ULL;
        }
        if (tid == 0) {
            *(uint32_t*)(s_base + UNARY_BUCKET_READY_OFF) = UNARY_BUCKET_MAGIC;
        }
        if (tid < S_RANK && sel_hproj_w != nullptr) {
            const uint16_t* hp_row = (const uint16_t*)sel_hproj_w + (size_t)tid * H_DIM;
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
            float sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;
            #pragma nounroll
            for (int i = 0; i < H_DIM; i += 8) {
                uint4 raw = *reinterpret_cast<const uint4*>(hp_row + i);
                const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float wv = bf16_to_f32(h[j]);
                    int idx = i + j;
                    sum0 += wv * h0[idx];
                    sum1 += wv * h1[idx];
                    sum2 += wv * h2[idx];
                    sum3 += wv * h3[idx];
                    sum4 += wv * h4[idx];
                    sum5 += wv * h5[idx];
                }
            }
            float* s_hp = (float*)(s_base + SCR_TEMP_OFF);
            s_hp[0 * S_RANK + tid] = sum0;
            s_hp[1 * S_RANK + tid] = sum1;
            s_hp[2 * S_RANK + tid] = sum2;
            s_hp[3 * S_RANK + tid] = sum3;
            s_hp[4 * S_RANK + tid] = sum4;
            s_hp[5 * S_RANK + tid] = sum5;
            if (tid == 0) {
                *(uint32_t*)(s_base + SCR_TEMP_OFF + 6 * S_RANK * sizeof(float)) = SEL_HPROJ_READY_TAG;
            }
        }
        return;
    }

    __shared__ float s_h_lds[6 * H_DIM]; // 18 KB LDS -> eliminates 8x redundant global reads per WG
    for (int i = threadIdx.x; i < 6 * H_DIM; i += 256) {
        s_h_lds[i] = s_h[i];
    }
    __syncthreads();

    const int global_row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane = threadIdx.x % 32;
    if (global_row >= W_DIM) return;

    const uint16_t* w_row = (const uint16_t*)out_proj_w + (size_t)global_row * H_DIM;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    float sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f;

    #pragma nounroll
    for (int col = lane * 8; col < H_DIM; col += 32 * 8) {
        uint4 raw = *reinterpret_cast<const uint4*>(w_row + col);
        const uint16_t* hw = reinterpret_cast<const uint16_t*>(&raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t u = static_cast<uint32_t>(hw[i]) << 16;
            float w_val = *reinterpret_cast<const float*>(&u);
            int idx = col + i;
            sum0 += w_val * s_h_lds[0 * H_DIM + idx];
            sum1 += w_val * s_h_lds[1 * H_DIM + idx];
            sum2 += w_val * s_h_lds[2 * H_DIM + idx];
            sum3 += w_val * s_h_lds[3 * H_DIM + idx];
            sum4 += w_val * s_h_lds[4 * H_DIM + idx];
            sum5 += w_val * s_h_lds[5 * H_DIM + idx];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
        sum2 += __shfl_down(sum2, offset, 32);
        sum3 += __shfl_down(sum3, offset, 32);
        sum4 += __shfl_down(sum4, offset, 32);
        sum5 += __shfl_down(sum5, offset, 32);
    }

    if (lane == 0) {
        float* s_out = (float*)(s_base + out_o_offset);
        s_out[0 * W_DIM + global_row] = round_bf16_val(sum0);
        s_out[1 * W_DIM + global_row] = round_bf16_val(sum1);
        s_out[2 * W_DIM + global_row] = round_bf16_val(sum2);
        s_out[3 * W_DIM + global_row] = round_bf16_val(sum3);
        s_out[4 * W_DIM + global_row] = round_bf16_val(sum4);
        s_out[5 * W_DIM + global_row] = round_bf16_val(sum5);
    }
}

// ==============================================================================
// Kernel 2A: Milestone 1 Batched 6-Slot Vectorized LM-Head Multi-GEMV
// 2-Half LDS Tiling (15 KB LDS -> 100% WGP Occupancy) + 1 Row/Wave + v_dot2_f32_bf16
// ==============================================================================
template <uint32_t D>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_lmhead_gemv6_kernel(
    const float*    __restrict__ d_hidden_base, // [6, D] FP32 in scratch (seq 0 base)
    const uint16_t* __restrict__ d_lm_head0,     // [124160, D] BF16 in VRAM (Rank 0)
    const uint16_t* __restrict__ d_lm_head1,     // [124160, D] BF16 in VRAM (Rank 1, optional)
    float*          __restrict__ d_logits_base0, // [6, V_total] FP32 in scratch (seq 0 SCR_BASE_LOGITS_OFF)
    uint32_t V_shard,                            // 124160
    uint32_t V_total,                            // 124160 or 248320
    char*           __restrict__ scratch_base = nullptr,
    int stage_id = 1
) {
    const int seq = blockIdx.y;
    char* seq_scr = scratch_base ? (scratch_base + (size_t)seq * SCR_PER_SEQ) : nullptr;
    if (stage_id >= 2 && seq_scr != nullptr && dloop_stage2_skipped(seq_scr)) {
        return;
    }
    __shared__ uint32_t s_unres6;
    const int tid = threadIdx.x;
    if (tid == 0) {
        uint32_t m6 = 0x3Fu;
        if (stage_id >= 3 && seq_scr != nullptr) {
            const uint32_t* exits = (const uint32_t*)(seq_scr + ALODLM_EXITS_OFF) + ((stage_id == 4) ? BLOCK_K : 0);
            m6 = 0u;
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                if (exits[k] == 0u) m6 |= (1u << k);
            }
        }
        s_unres6 = m6;
    }
    __syncthreads();
    const uint32_t unres6 = s_unres6;
    if (unres6 == 0u) return;

    const float* d_hidden = (const float*)((const char*)d_hidden_base + (size_t)seq * SCR_PER_SEQ);
    float* d_logits_base  = (float*)((char*)d_logits_base0 + (size_t)seq * SCR_PER_SEQ);

    constexpr int K_STEPS = D / 256;
    constexpr int NUM_HALVES = (D == 2560) ? 2 : 1;
    constexpr int HALF_STEPS = K_STEPS / NUM_HALVES;
    constexpr int HALF_PAIRS = HALF_STEPS * 128;
    __shared__ uint32_t s_hidden[6 * HALF_PAIRS];

    const int lane_id = tid & 31;
    const int wave_id = tid >> 5;
    const uint32_t global_row = blockIdx.x * 8 + wave_id;
    const bool valid_row = global_row < V_total;
    const bool is_shard1 = (d_lm_head1 != nullptr) && (global_row >= V_shard);
    const uint32_t row_in_shard = is_shard1 ? (global_row - V_shard) : global_row;
    const uint16_t* shard_mat = is_shard1 ? d_lm_head1 : d_lm_head0;
    const r4d_u32x4* lm_row = valid_row
        ? (const r4d_u32x4*)(shard_mat + (size_t)row_in_shard * D) : nullptr;
    float acc[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma nounroll
    for (int half = 0; half < NUM_HALVES; ++half) {
        if (half > 0) __syncthreads();
        #pragma unroll 4
        for (int i = tid; i < 6 * HALF_PAIRS; i += blockDim.x) {
            const int slot = i / HALF_PAIRS;
            const int pair = half * HALF_PAIRS + i % HALF_PAIRS;
            const int f_idx = slot * (int)D + pair * 2;
            uint32_t u0 = __builtin_bit_cast(uint32_t, d_hidden[f_idx + 0]);
            uint32_t u1 = __builtin_bit_cast(uint32_t, d_hidden[f_idx + 1]);
            s_hidden[i] = (u0 >> 16) | (u1 & 0xFFFF0000u);
        }
        __syncthreads();
        if (!valid_row) continue;

        #pragma nounroll
        for (int hs = 0; hs < HALF_STEPS; ++hs) {
            int vec_offset = (half * HALF_STEPS + hs) * 32 + lane_id;
            r4d_u32x4 w_vec = __builtin_nontemporal_load(&lm_row[vec_offset]);
            int lds_base = (hs * 32 + lane_id) * 4;

            #pragma unroll
            for (int s = 0; s < 6; ++s) {
                uint32_t o0 = s_hidden[s * HALF_PAIRS + lds_base + 0];
                uint32_t o1 = s_hidden[s * HALF_PAIRS + lds_base + 1];
                uint32_t o2 = s_hidden[s * HALF_PAIRS + lds_base + 2];
                uint32_t o3 = s_hidden[s * HALF_PAIRS + lds_base + 3];

                float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                acc[s] += (((0.0f + p0) + p1) + p2) + p3;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            acc[s] += __shfl_down(acc[s], offset, 32);
        }
    }

    __shared__ uint64_t s_wg_u[6][8];
    if (lane_id == 0) {
        const bool valid_act = valid_row && (global_row < (uint32_t)V_ACT);
        #pragma unroll
        for (int s = 0; s < 6; ++s) {
            if ((unres6 >> s) & 1u) {
                float val = acc[s];
                if (valid_row) d_logits_base[(size_t)s * V_total + global_row] = val;
                s_wg_u[s][wave_id] = valid_act
                    ? ((static_cast<uint64_t>(float_to_ordered_u32(val)) << 32) | static_cast<uint64_t>(global_row))
                    : 0ULL;
            }
        }
    }
    if (seq_scr != nullptr) {
        __syncthreads();
        if (tid < 6 && ((unres6 >> tid) & 1u)) {
            uint64_t best_p = s_wg_u[tid][0];
            #pragma unroll
            for (int w = 1; w < 8; ++w) {
                uint64_t cand_p = s_wg_u[tid][w];
                if (cand_p > best_p) best_p = cand_p;
            }
            if (best_p != 0ULL) {
                const int u_slot = (stage_id == 4) ? (6 + tid) : tid;
                unsigned long long* u_buckets = (unsigned long long*)(seq_scr + UNARY_BUCKET_OFF);
                atomicMax(&u_buckets[u_slot * 256 + (blockIdx.x & 255u)], (unsigned long long)best_p);
            }
        }
    }
}

extern "C" int r4d_fulldrafter_lmhead_gemv6(
    const void* d_hidden,
    const void* d_lm_head0,
    const void* d_lm_head1,
    float* d_logits_base,
    uint32_t V_shard,
    uint32_t V_total,
    uint32_t D,
    hipStream_t stream
) {
    if (!d_hidden || !d_lm_head0 || !d_logits_base) return -1;

    dim3 block(256);
    dim3 grid((V_total + 7) / 8);

    if (D == 768) {
        r4d_fulldrafter_lmhead_gemv6_kernel<768><<<grid, block, 0, stream>>>(
            (const float*)d_hidden,
            (const uint16_t*)d_lm_head0,
            (const uint16_t*)d_lm_head1,
            d_logits_base,
            V_shard,
            V_total
        );
    } else {
        r4d_fulldrafter_lmhead_gemv6_kernel<2560><<<grid, block, 0, stream>>>(
            (const float*)d_hidden,
            (const uint16_t*)d_lm_head0,
            (const uint16_t*)d_lm_head1,
            d_logits_base,
            V_shard,
            V_total
        );
    }

    return 0;
}


// ==============================================================================
// Milestone 2: Fused Markov Bigram & In-LDS Top-16 Selector
// 1. 8-Lane Cooperative Row-Coalescing on markov_matrix_b (512B contiguous wave loads)
// 2. Zero-Scratch Register Candidate Array + Register Final Merge
// 3. TP=2 P2P Push Split-Vocabulary Sharding (124,160 rows/GPU + 128B P2P Push Merge)
// ==============================================================================

static constexpr int M2_GRID_BLOCKS   = 64;   // 1 workgroup per CU on R9700
static constexpr int M2_THREADS       = 256;  // 8 Wave32 wavefronts
static constexpr int M2_CANDS_PER_BLK = 16;   // Top-16 candidates per block
[[maybe_unused]] static constexpr int M2_TOTAL_CANDS = M2_GRID_BLOCKS * M2_CANDS_PER_BLK; // 1024 candidates

__device__ __forceinline__ bool is_better(float va, uint32_t ia, float vb, uint32_t ib) {
    return r4d::fulldrafter_is_better(va, ia, vb, ib);
}

__device__ __forceinline__ int dev_get_alodlm_max_mcache_slots(
    int M, int64_t total_scratch_bytes, int total_vocab, int active_k
) {
    if (total_vocab <= V_SHARD) {
        if (total_scratch_bytes >= (int64_t)(2 * M) * SCR_PER_SEQ) return 12;
        if (active_k <= 6) return 6;
        return 3;
    } else {
        if (total_scratch_bytes >= (int64_t)(2 * M) * SCR_PER_SEQ) return 9;
        return 1;
    }
}

__device__ __forceinline__ float* dev_get_alodlm_mcache_slot_ptr(
    char* scratch, char* s_base, int s, int M, int64_t total_scratch_bytes, int total_vocab, int active_k, int c
) {
    const int base_slots = (total_vocab <= V_SHARD) ? 3 : 1;
    if (c < base_slots) {
        return (float*)(s_base + ALODLM_MCACHE_VAL_OFF) + (size_t)c * total_vocab;
    }
    if (total_scratch_bytes >= (int64_t)(2 * M) * SCR_PER_SEQ) {
        char* ext_base = scratch + (size_t)(M + s) * SCR_PER_SEQ;
        return (float*)ext_base + (size_t)(c - base_slots) * total_vocab;
    }
    return (float*)(s_base + SCR_BASE_LOGITS_OFF) + (size_t)(3 + c) * total_vocab;
}

__device__ __forceinline__ void dev_alodlm_commit_step(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ s_base,
    uint32_t* __restrict__ tokens_out,
    int s,
    int max_depth,
    int active_k = FULLDRAFTER_TOTAL_K
) {
    uint32_t* ids        = (uint32_t*)(s_base + ALODLM_IDS_OFF);
    const uint32_t* pred = (const uint32_t*)(s_base + ALODLM_PRED_OFF);
    uint32_t* exits      = (uint32_t*)(s_base + ALODLM_EXITS_OFF);
    const float* surv    = (const float*)(s_base + ALODLM_SURV_OFF);
    const float* ent     = (const float*)(s_base + ALODLM_ENTROPY_OFF);
    const int depth      = (int)(*(const uint32_t*)(s_base + ALODLM_DEPTHS_OFF));
    if (max_depth == 1) {
        for (int j = 0; j < FULLDRAFTER_TOTAL_K; ++j) {
            const uint32_t token = (j < active_k) ? pred[j] : r4d::FULLDRAFTER_SENTINEL_ID;
            ids[j] = token;
            exits[j] = 1u;
            tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + j] = token;
        }
        *(uint32_t*)(s_base + ALODLM_DEPTHS_OFF) = 1u;
        *(uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF) = 0u;
        *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF) = DLOOP_VERIF_MAGIC;
        return;
    }


    const float tau     = p->alodlm_tau;
    const float q       = p->alodlm_q;
    const float pos_pen = p->alodlm_pos_penalty;

    uint32_t commit_mask = 0u;
    uint32_t rem_mask    = 0u;
    int unres_cum = 0;
    int rem_count = 0;
    float cdf_sum = 0.0f;

    #pragma nounroll
    for (int j = 0; j < active_k; ++j) {
        bool unres = (exits[j] == 0u);
        if (unres) ++unres_cum;
        float pen = (unres_cum > 1) ? ((float)(unres_cum - 1) * pos_pen) : 0.0f;
        bool c = unres && ((ent[j] + pen) < tau);
        bool rem = unres && !c;
        if (c) commit_mask |= (1u << j);
        if (rem) {
            rem_mask |= (1u << j);
            ++rem_count;
            cdf_sum += (1.0f - surv[j]);
        }
    }

    float mean_cdf = cdf_sum / (float)(rem_count > 0 ? rem_count : 1);
    bool stop = (rem_count == 0) || (mean_cdf >= q) || (depth >= max_depth - 1);
    uint32_t write_mask = commit_mask | (stop ? rem_mask : 0u);

    #pragma nounroll
    for (int j = 0; j < active_k; ++j) {
        if ((write_mask >> j) & 1u) {
            ids[j]   = pred[j];
            exits[j] = (uint32_t)(depth + 1);
            tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + j] = pred[j];
        }
    }
    *(uint32_t*)(s_base + ALODLM_DEPTHS_OFF) = (uint32_t)(depth + 1);
    if (stop) {
        *(uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF) = 0u;
        *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF)   = DLOOP_VERIF_MAGIC;
    }
}

__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_slot_markov_select_kernel(
    int stage,               // 1..4 single-slot, or 5 = persistent 12-slot ALoDLM-4 (M==1)
    int slot,                // 0..5 (or 0 when stage == 5)
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch,
    uint32_t* __restrict__ tokens_out
) {
    const int b = blockIdx.x; // 0..63 (64 CUs)
    const int s = blockIdx.y; // Sequence index 0..M-1
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 3; // 0..3 (4 groups of 8 lanes per wave)
    const int sub  = lane & 7;  // 0..7 (8 lanes per 128B row)

    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    if ((stage == 2 || p->is_alodlm_loop != 0u) && dloop_stage2_skipped(s_base)) return;

    const bool gemv12_mode = (stage >= 3);
    const int eff_stage = gemv12_mode ? (stage - 2) : stage;
    const int local_slot = slot;
    const int global_slot_idx = (eff_stage == 1) ? slot : (BLOCK_K + slot);
    int max_depth = (p->alodlm_max_depth > 0 && p->alodlm_max_depth <= 8) ? (int)p->alodlm_max_depth : 4;
    if (gridDim.y >= 5) max_depth = 1;

    if (p->is_alodlm_loop != 0u) {
        const uint32_t* exits = (const uint32_t*)(s_base + ALODLM_EXITS_OFF);
        if (exits[global_slot_idx] != 0u) {
            if (b == 0 && tid == 0) {
                uint32_t committed_tok = ((const uint32_t*)(s_base + ALODLM_IDS_OFF))[global_slot_idx];
                ((uint32_t*)(s_base + ALODLM_PRED_OFF))[global_slot_idx] = committed_tok;
                ((uint32_t*)(s_base + SCR_PREV_TOK_OFF))[global_slot_idx + 1] = committed_tok;
                if (global_slot_idx == FULLDRAFTER_TOTAL_K - 1) {
                    dev_alodlm_commit_step(p, s_base, tokens_out, s, max_depth);
                }
            }
            return;
        }
    }

    const bool tp2_p2p = (p->tp2_p2p_enabled != 0u) && (p->my_p2p_comm != nullptr) &&
                         (p->peer_p2p_comm != nullptr) && (p->d_step_seq != nullptr);
    const bool is_dual_rank = (!tp2_p2p) && (p->teacher_lm_head_rank1 != nullptr);
    const int total_vocab = is_dual_rank ? V_TOTAL : V_SHARD;
    const uint32_t vocab_offset = (tp2_p2p && p->rank_id == 1u) ? (uint32_t)V_SHARD : 0u;
    const int logit_slot_idx = gemv12_mode ? global_slot_idx : local_slot;
    const float* s_base_logits = (const float*)(s_base + SCR_BASE_LOGITS_OFF + (size_t)logit_slot_idx * total_vocab * sizeof(float));

    const uint32_t prev_tok = ((const uint32_t*)(s_base + SCR_PREV_TOK_OFF))[global_slot_idx];

    __shared__ uint32_t s_ha_pk[M_RANK / 2];
    __shared__ uint32_t s_ha_nz;
    __shared__ float    s_a_norm;
    __shared__ float    s_guaranteed_16th;
    __shared__ float    s_wave_dot[8][32];
    __shared__ float    s_q[S_RANK];
    __shared__ uint32_t s_top16_id[K_TOP];
    __shared__ float    s_top16_val[K_TOP];
    __shared__ float    s_top16_score[K_TOP];
    __shared__ unsigned int s_arrived;

        // 1. Cooperative load of h_markov as packed BF16 pairs into LDS
        uint32_t ptok_clamp = (prev_tok < V_TOTAL) ? prev_tok : 0;
    if (tid == 0) {
        s_ha_nz = 0u;
        s_a_norm = 0.0f;
        s_guaranteed_16th = r4d::FULLDRAFTER_SENTINEL_SCORE;
    }
    __syncthreads();
    if (tid < 8) {
        const uint4* a_vec = reinterpret_cast<const uint4*>(
            (const uint16_t*)p->markov_matrix_a + (size_t)ptok_clamp * M_RANK
        );
        uint4 raw = a_vec[tid];
        s_ha_pk[tid * 4 + 0] = raw.x;
        s_ha_pk[tid * 4 + 1] = raw.y;
        s_ha_pk[tid * 4 + 2] = raw.z;
        s_ha_pk[tid * 4 + 3] = raw.w;
        uint32_t nz = (raw.x | raw.y | raw.z | raw.w) & 0x7FFF7FFFu;
        if (nz) atomicOr(&s_ha_nz, 1u);
    }
    __syncthreads();

    const uint16_t* mb_matrix = (const uint16_t*)p->markov_matrix_b;
    const float*    mb_norm   = p->markov_b_norm;
    const bool has_markov = (mb_matrix != nullptr) && (s_ha_nz != 0u);
    const uint32_t ha_sub0 = s_ha_pk[sub * 4 + 0];
    const uint32_t ha_sub1 = s_ha_pk[sub * 4 + 1];
    const uint32_t ha_sub2 = s_ha_pk[sub * 4 + 2];
    const uint32_t ha_sub3 = s_ha_pk[sub * 4 + 3];

    // 1B. Compute L2 norm of a_prev and exact Cauchy-Schwarz lower bound s_guaranteed_16th
    // (Disabled for is_alodlm_loop != 0u so full-vocab Shannon entropy includes exact Markov logits on all v < V_ACT)
    if (p->is_alodlm_loop == 0u && has_markov && mb_norm != nullptr &&
        *(const uint32_t*)(s_base + UNARY_BUCKET_READY_OFF) == UNARY_BUCKET_MAGIC) {
        if (tid < 32) {
            uint32_t pk = s_ha_pk[tid];
            float f0 = bf16_to_f32((uint16_t)(pk & 0xFFFFu));
            float f1 = bf16_to_f32((uint16_t)(pk >> 16));
            float sq = f0 * f0 + f1 * f1;
            sq = r4d_wave_sum(sq);
            if (tid == 0) s_a_norm = sqrtf(sq) * 1.001f;
        }
        // Each wave (0..7) picks its Top-2 unary bucket winners -> 16 candidates across the 8 waves
        const uint64_t packed = ((const uint64_t*)(s_base + UNARY_BUCKET_OFF))[local_slot * 256 + tid];
        float u_val = (packed != 0ULL) ? ordered_u32_to_float((uint32_t)(packed >> 32))
                                       : r4d::FULLDRAFTER_SENTINEL_SCORE;
        uint32_t u_id = (packed != 0ULL) ? (uint32_t)(packed & 0xFFFFFFFFu) : r4d::FULLDRAFTER_SENTINEL_ID;
        #pragma unroll
        for (int pick = 0; pick < 2; ++pick) {
            float bv = u_val;
            uint32_t bi = u_id;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float ov = __shfl_down(bv, offset, 32);
                uint32_t oi = __shfl_down(bi, offset, 32);
                if (is_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            uint32_t win_i = __shfl(bi, 0, 32);
            if (lane == 0) {
                s_top16_val[wave * 2 + pick] = bv;
                s_top16_id[wave * 2 + pick]  = bi;
            }
            if (u_id == win_i) {
                u_val = r4d::FULLDRAFTER_SENTINEL_SCORE;
                u_id  = r4d::FULLDRAFTER_SENTINEL_ID;
            }
        }
        __syncthreads();

        // First 16 8-lane groups (tid < 128) evaluate the exact Markov score on the 16 unary winners
        if (tid < 128) {
            const int grp16 = tid >> 3; // 0..15
            const uint32_t uid = s_top16_id[grp16];
            const uint32_t ug_row = vocab_offset + uid;
            float part = 0.0f;
            if (uid < (uint32_t)total_vocab && ug_row < (uint32_t)V_ACT) {
                const r4d_u32x4* mb_row_u4 = reinterpret_cast<const r4d_u32x4*>(
                    mb_matrix + (size_t)ug_row * M_RANK
                );
                r4d_u32x4 raw = __builtin_nontemporal_load(&mb_row_u4[sub]);
                float p0 = r4d_dot2_f32_bf16(raw[0], ha_sub0, 0.0f);
                float p1 = r4d_dot2_f32_bf16(raw[1], ha_sub1, 0.0f);
                float p2 = r4d_dot2_f32_bf16(raw[2], ha_sub2, 0.0f);
                float p3 = r4d_dot2_f32_bf16(raw[3], ha_sub3, 0.0f);
                part = (((0.0f + p0) + p1) + p2) + p3;
            }
            part += __shfl_down(part, 4, 8);
            part += __shfl_down(part, 2, 8);
            part += __shfl_down(part, 1, 8);
            if (sub == 0) {
                s_top16_score[grp16] = (uid < (uint32_t)total_vocab && ug_row < (uint32_t)V_ACT)
                                           ? (s_top16_val[grp16] + part)
                                           : r4d::FULLDRAFTER_SENTINEL_SCORE;
            }
        }
        __syncthreads();
        if (tid < 16) {
            float sc = s_top16_score[tid];
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                float other = __shfl_down(sc, offset, 32);
                if (other < sc) sc = other;
            }
            if (tid == 0 && sc > -1e20f) {
                s_guaranteed_16th = sc - 1e-5f;
            }
        }
        __syncthreads();
    }

    // 2. 8-Lane Cooperative Row-Coalesced Markov Dot Product with Predecessor-Invariance Cache & 1-Cycle Wave-Ballot CS Skip
    const int v_per_block = (total_vocab + gridDim.x - 1) / gridDim.x;
    const int v_start = b * v_per_block;
    const int v_end = min(v_start + v_per_block, total_vocab);

    int mcache_hit   = -1;
    int mcache_write = -1;
    float* mcache_val = (float*)(s_base + ALODLM_MCACHE_VAL_OFF);
    if (p->is_alodlm_loop != 0u && has_markov) {
        const int max_mcache_slots = (total_vocab <= V_SHARD) ? 3 : 1;
        const uint32_t* mcache_tok = (const uint32_t*)(s_base + ALODLM_MCACHE_TOK_OFF);
        for (int c = 0; c < max_mcache_slots; ++c) {
            if (mcache_tok[c] == ptok_clamp) {
                mcache_hit = c;
                break;
            }
        }
        if (mcache_hit < 0) {
            if (global_slot_idx < max_mcache_slots) {
                mcache_write = global_slot_idx;
            } else {
                for (int c = 0; c < max_mcache_slots; ++c) {
                    if (mcache_tok[c] == 0xFFFFFFFFu) {
                        mcache_write = c;
                        break;
                    }
                }
            }
        }
    }

    float cand_v[16];
    uint32_t cand_i[16];
    const float a_norm = s_a_norm;
    const float guar_16th = s_guaranteed_16th;
    const bool use_cs_prune = (p->is_alodlm_loop == 0u) && (mb_norm != nullptr) && (guar_16th > -1e20f);

    #pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int v_wave_base = v_start + m * 256 + wave * 32;
        const int v = v_wave_base + lane;
        const uint32_t v_global = vocab_offset + (uint32_t)v;
        const bool valid_v = (v < v_end) && (v_global < (uint32_t)V_ACT);
        const float my_base = valid_v ? s_base_logits[v] : r4d::FULLDRAFTER_SENTINEL_SCORE;

        if (!has_markov) {
            cand_v[m] = my_base;
            cand_i[m] = valid_v ? v_global : r4d::FULLDRAFTER_SENTINEL_ID;
            continue;
        }

        if (mcache_hit >= 0) {
            if (valid_v) {
                cand_v[m] = my_base + mcache_val[(size_t)mcache_hit * total_vocab + v];
                cand_i[m] = v_global;
            } else {
                cand_v[m] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                cand_i[m] = r4d::FULLDRAFTER_SENTINEL_ID;
            }
            continue;
        }

        const bool my_active = valid_v && (!use_cs_prune || ((my_base + a_norm * mb_norm[v_global]) >= guar_16th));
        const uint32_t active_mask = __builtin_amdgcn_ballot_w32(my_active);
        if (active_mask == 0u) {
            cand_v[m] = r4d::FULLDRAFTER_SENTINEL_SCORE;
            cand_i[m] = r4d::FULLDRAFTER_SENTINEL_ID;
            continue;
        }

        #pragma unroll
        for (int s_step = 0; s_step < 8; ++s_step) {
            const uint32_t step_bits = (active_mask >> (s_step * 4)) & 0xFu;
            if (step_bits == 0u) continue;
            const int v_row = v_wave_base + s_step * 4 + grp;
            const uint32_t vg_row = vocab_offset + (uint32_t)v_row;
            float part = 0.0f;
            if ((step_bits >> grp) & 1u) {
                const r4d_u32x4* mb_row_u4 = reinterpret_cast<const r4d_u32x4*>(
                    mb_matrix + (size_t)vg_row * M_RANK
                );
                r4d_u32x4 raw = __builtin_nontemporal_load(&mb_row_u4[sub]);
                float p0 = r4d_dot2_f32_bf16(raw[0], ha_sub0, 0.0f);
                float p1 = r4d_dot2_f32_bf16(raw[1], ha_sub1, 0.0f);
                float p2 = r4d_dot2_f32_bf16(raw[2], ha_sub2, 0.0f);
                float p3 = r4d_dot2_f32_bf16(raw[3], ha_sub3, 0.0f);
                part = (((0.0f + p0) + p1) + p2) + p3;
                part += __shfl_down(part, 4, 8);
                part += __shfl_down(part, 2, 8);
                part += __shfl_down(part, 1, 8);
            }
            if (sub == 0) {
                s_wave_dot[wave][s_step * 4 + grp] = part;
            }
        }

        if (my_active) {
            float m_dot = s_wave_dot[wave][lane];
            if (mcache_write >= 0) {
                mcache_val[(size_t)mcache_write * total_vocab + v] = m_dot;
            }
            cand_v[m] = my_base + m_dot;
            cand_i[m] = v_global;
        } else {
            cand_v[m] = r4d::FULLDRAFTER_SENTINEL_SCORE;
            cand_i[m] = r4d::FULLDRAFTER_SENTINEL_ID;
        }
    }

    uint32_t* blk_cands = (uint32_t*)(s_base + SCR_BLK_CANDS_OFF + (size_t)b * K_TOP * sizeof(uint32_t));
    float*    blk_unary = (float*)(s_base + SCR_BLK_UNARY_OFF + (size_t)b * K_TOP * sizeof(float));

    // 3. Two-Stage Wave-Local Top-16 -> Wave-0 Block Top-16 Reduction (2 barriers instead of 32!)
    // Reuses s_wave_dot[8][32] (1 KB LDS) as [8 waves x 16 vals] + [8 waves x 16 ids].
    __syncthreads();
    float*    s_wtop_val = &s_wave_dot[0][0];
    uint32_t* s_wtop_id  = reinterpret_cast<uint32_t*>(&s_wave_dot[4][0]);

    #pragma nounroll
    for (int k = 0; k < K_TOP; ++k) {
        float my_val = cand_v[0];
        uint32_t my_id = cand_i[0];
        #pragma unroll
        for (int m = 1; m < 16; ++m) {
            if (is_better(cand_v[m], cand_i[m], my_val, my_id)) {
                my_val = cand_v[m];
                my_id  = cand_i[m];
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down(my_val, offset, 32);
            uint32_t oi = __shfl_down(my_id, offset, 32);
            if (is_better(ov, oi, my_val, my_id)) {
                my_val = ov;
                my_id  = oi;
            }
        }
        uint32_t win_id = __shfl(my_id, 0, 32);
        if (lane == 0) {
            s_wtop_val[wave * K_TOP + k] = my_val;
            s_wtop_id[wave * K_TOP + k]  = my_id;
        }
        #pragma unroll
        for (int m = 0; m < 16; ++m) {
            if (cand_i[m] == win_id) {
                cand_v[m] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                cand_i[m] = r4d::FULLDRAFTER_SENTINEL_ID;
            }
        }
    }
    __syncthreads();

    if (tid < 32) {
        float wv4[4];
        uint32_t wi4[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int idx = tid * 4 + c;
            wv4[c] = s_wtop_val[idx];
            wi4[c] = s_wtop_id[idx];
        }
        #pragma nounroll
        for (int k = 0; k < K_TOP; ++k) {
            float bv = wv4[0];
            uint32_t bi = wi4[0];
            #pragma unroll
            for (int c = 1; c < 4; ++c) {
                if (is_better(wv4[c], wi4[c], bv, bi)) { bv = wv4[c]; bi = wi4[c]; }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float ov = __shfl_down(bv, offset, 32);
                uint32_t oi = __shfl_down(bi, offset, 32);
                if (is_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            uint32_t win_i = __shfl(bi, 0, 32);
            if (tid == 0) {
                blk_unary[k] = bv;
                blk_cands[k] = bi;
                if (k == 0) s_top16_val[0] = bv;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (wi4[c] == win_i) {
                    wv4[c] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                    wi4[c] = r4d::FULLDRAFTER_SENTINEL_ID;
                }
            }
        }
    }
    __syncthreads();

    // 3B. ALoDLM-4 Per-Block Partition Function (max_l, sum_e, sum_xe) for Exact Entropy
    if (p->is_alodlm_loop != 0u) {
        const float blk_max = s_top16_val[0];
        float my_se = 0.0f, my_sxe = 0.0f;
        if (blk_max > -1e20f) {
            if (tid < 128) {
                float wv = s_wtop_val[tid];
                if (wv > -1e20f) {
                    float x = wv - blk_max;
                    if (x > -80.0f) {
                        float e = expf(x);
                        my_se  += e;
                        my_sxe += x * e;
                    }
                }
            }
            #pragma unroll
            for (int m = 0; m < 16; ++m) {
                if (cand_i[m] != r4d::FULLDRAFTER_SENTINEL_ID) {
                    float lv = cand_v[m];
                    float x = lv - blk_max;
                    if (x > -80.0f) {
                        float e = expf(x);
                        my_se  += e;
                        my_sxe += x * e;
                    }
                }
            }
        }
        float tot_se  = block_reduce_sum(my_se);
        float tot_sxe = block_reduce_sum(my_sxe);
        if (tid == 0) {
            float* bent = (float*)(s_base + ALODLM_BLK_ENT_OFF) + b * 4;
            bent[0] = blk_max;
            bent[1] = tot_se;
            bent[2] = tot_sxe;
        }
    }

    // 4. In-kernel Atomic Arrival & Merge Elimination
    __threadfence();
    __syncthreads();

    if (tid == 0) {
        unsigned int* ctr = (unsigned int*)(s_base + SCR_ARRIVAL_CTR_OFF);
        s_arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();

    if (s_arrived != (unsigned int)(gridDim.x - 1)) return;

    // Reset arrival counter for next slot and publish Markov cache tag if written
    if (tid == 0) {
        unsigned int* ctr = (unsigned int*)(s_base + SCR_ARRIVAL_CTR_OFF);
        *ctr = 0u;
        if (mcache_write >= 0) {
            ((uint32_t*)(s_base + ALODLM_MCACHE_TOK_OFF))[mcache_write] = ptok_clamp;
        }
    }
    __threadfence();

    // 5. Final Workgroup Cooperative Merge in Registers (4 candidates per thread, 0 LDS array)
    const uint32_t* all_cands = (const uint32_t*)(s_base + SCR_BLK_CANDS_OFF);
    const float*    all_unary = (const float*)(s_base + SCR_BLK_UNARY_OFF);

    float mv[4];
    uint32_t mi[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        int idx = tid + c * 256;
        mv[c] = all_unary[idx];
        mi[c] = all_cands[idx];
    }

    #pragma nounroll
    for (int k = 0; k < K_TOP; ++k) {
        float my_best_val = mv[0];
        uint32_t my_best_id = mi[0];
        #pragma unroll
        for (int c = 1; c < 4; ++c) {
            if (is_better(mv[c], mi[c], my_best_val, my_best_id)) {
                my_best_val = mv[c];
                my_best_id  = mi[c];
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down(my_best_val, offset, 32);
            uint32_t oi = __shfl_down(my_best_id, offset, 32);
            if (is_better(ov, oi, my_best_val, my_best_id)) {
                my_best_val = ov;
                my_best_id  = oi;
            }
        }
        uint32_t win_id = __shfl(my_best_id, 0, 32);
        if (lane == 0) {
            s_wtop_val[wave * K_TOP + k] = my_best_val;
            s_wtop_id[wave * K_TOP + k]  = my_best_id;
        }
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            if (mi[c] == win_id) {
                mv[c] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                mi[c] = r4d::FULLDRAFTER_SENTINEL_ID;
            }
        }
    }
    __syncthreads();

    if (tid < 32) {
        float wv4[4];
        uint32_t wi4[4];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int idx = tid * 4 + c;
            wv4[c] = s_wtop_val[idx];
            wi4[c] = s_wtop_id[idx];
        }
        #pragma nounroll
        for (int k = 0; k < K_TOP; ++k) {
            float bv = wv4[0];
            uint32_t bi = wi4[0];
            #pragma unroll
            for (int c = 1; c < 4; ++c) {
                if (is_better(wv4[c], wi4[c], bv, bi)) { bv = wv4[c]; bi = wi4[c]; }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float ov = __shfl_down(bv, offset, 32);
                uint32_t oi = __shfl_down(bi, offset, 32);
                if (is_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
            }
            uint32_t win_i = __shfl(bi, 0, 32);
            if (tid == 0) {
                s_top16_val[k] = bv;
                s_top16_id[k]  = bi;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (wi4[c] == win_i) {
                    wv4[c] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                    wi4[c] = r4d::FULLDRAFTER_SENTINEL_ID;
                }
            }
        }
    }
    __syncthreads();

    // 5A-2. ALoDLM-4 64-Block Partition Reduction into Shard (shard_max_l, shard_sum_e, shard_sum_xe)
    __shared__ float s_shard_ent[3];
    if (p->is_alodlm_loop != 0u) {
        const float* bent = (const float*)(s_base + ALODLM_BLK_ENT_OFF);
        float bm   = (tid < 64) ? bent[tid * 4 + 0] : r4d::FULLDRAFTER_SENTINEL_SCORE;
        float bse  = (tid < 64) ? bent[tid * 4 + 1] : 0.0f;
        float bsxe = (tid < 64) ? bent[tid * 4 + 2] : 0.0f;
        const float sh_max = s_top16_val[0];
        float scaled_se = 0.0f, scaled_sxe = 0.0f;
        if (tid < 64 && bm > -1e20f && sh_max > -1e20f) {
            float dm = bm - sh_max;
            if (dm > -80.0f) {
                float sc = expf(dm);
                scaled_se  = bse * sc;
                scaled_sxe = (bsxe + bse * dm) * sc;
            }
        }
        float tot_se  = block_reduce_sum(scaled_se);
        float tot_sxe = block_reduce_sum(scaled_sxe);
        if (tid == 0) {
            s_shard_ent[0] = sh_max;
            s_shard_ent[1] = tot_se;
            s_shard_ent[2] = tot_sxe;
            if (!tp2_p2p) {
                float ent = (tot_se > 0.0f) ? (logf(tot_se) - tot_sxe / tot_se) : 0.0f;
                ((float*)(s_base + ALODLM_ENTROPY_OFF))[global_slot_idx] = ent;
            }
        }
        __syncthreads();
    }

    // 5B. TP=2 P2P Push Candidate Exchange & Wave-0 32->16 Merge (128B pushed over PCIe)
    if (tp2_p2p) {
        const uint32_t step_seq  = *p->d_step_seq;
        const uint32_t cur_depth = (p->is_alodlm_loop != 0u) ? *(const uint32_t*)(s_base + ALODLM_DEPTHS_OFF) : (uint32_t)(eff_stage - 1);
        const uint32_t pkt_idx   = (uint32_t)s * FULLDRAFTER_TOTAL_K + (uint32_t)global_slot_idx;
        const uint32_t tag       = (step_seq * 8u + cur_depth) * 1024u + pkt_idx + 1u;
        FullDrafterP2PSlotPacket* my_pkt   = (FullDrafterP2PSlotPacket*)p->my_p2p_comm + pkt_idx;
        FullDrafterP2PSlotPacket* peer_pkt = (FullDrafterP2PSlotPacket*)p->peer_p2p_comm + pkt_idx;

        if (tid < K_TOP) {
            peer_pkt->vals[tid] = s_top16_val[tid];
            peer_pkt->ids[tid]  = s_top16_id[tid];
            if (tid == 0 && p->is_alodlm_loop != 0u) {
                peer_pkt->shard_max_l  = s_shard_ent[0];
                peer_pkt->shard_sum_e  = s_shard_ent[1];
                peer_pkt->shard_sum_xe = s_shard_ent[2];
            }
        }
        __threadfence_system();
        __syncthreads();

        if (tid == 0) {
            __hip_atomic_store(&peer_pkt->flag, tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            uint64_t spin = 0;
            while (__hip_atomic_load(&my_pkt->flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != tag) {
                if (++spin > 200000000ULL) break;
            }
        }
        __syncthreads();
        __threadfence_system();

        if (tid < 32) {
            float cv = (tid < 16) ? s_top16_val[tid] : my_pkt->vals[tid - 16];
            uint32_t ci = (tid < 16) ? s_top16_id[tid] : my_pkt->ids[tid - 16];
            #pragma unroll
            for (int k = 0; k < K_TOP; ++k) {
                float best_v = cv;
                uint32_t best_i = ci;
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    float ov = __shfl_down(best_v, offset, 32);
                    uint32_t oi = __shfl_down(best_i, offset, 32);
                    if (is_better(ov, oi, best_v, best_i)) {
                        best_v = ov;
                        best_i = oi;
                    }
                }
                uint32_t win_i = __shfl(best_i, 0, 32);
                if (tid == 0) {
                    s_top16_val[k] = best_v;
                    s_top16_id[k]  = best_i;
                }
                if (ci == win_i) {
                    cv = r4d::FULLDRAFTER_SENTINEL_SCORE;
                    ci = r4d::FULLDRAFTER_SENTINEL_ID;
                }
            }
            if (tid == 0 && p->is_alodlm_loop != 0u) {
                float m0   = (p->rank_id == 0u) ? s_shard_ent[0] : my_pkt->shard_max_l;
                float se0  = (p->rank_id == 0u) ? s_shard_ent[1] : my_pkt->shard_sum_e;
                float sxe0 = (p->rank_id == 0u) ? s_shard_ent[2] : my_pkt->shard_sum_xe;
                float m1   = (p->rank_id == 0u) ? my_pkt->shard_max_l  : s_shard_ent[0];
                float se1  = (p->rank_id == 0u) ? my_pkt->shard_sum_e  : s_shard_ent[1];
                float sxe1 = (p->rank_id == 0u) ? my_pkt->shard_sum_xe : s_shard_ent[2];
                float mg   = fmaxf(m0, m1);
                float c0   = (m0 - mg > -80.0f) ? expf(m0 - mg) : 0.0f;
                float c1   = (m1 - mg > -80.0f) ? expf(m1 - mg) : 0.0f;
                float seg  = se0 * c0 + se1 * c1;
                float sxeg = (sxe0 + se0 * (m0 - mg)) * c0 + (sxe1 + se1 * (m1 - mg)) * c1;
                float ent  = (seg > 0.0f) ? (logf(seg) - sxeg / seg) : 0.0f;
                ((float*)(s_base + ALODLM_ENTROPY_OFF))[global_slot_idx] = ent;
            }
        }
        __syncthreads();
    }

    // 6. Top-16 Selector Scoring (uses precomputed fast_out_proj h_proj when available)
    const uint16_t* w_hproj = (const uint16_t*)p->selector_hidden_proj;
    const uint16_t* w_pred  = (const uint16_t*)p->selector_pred_codebook;
    const uint16_t* w_succ  = (const uint16_t*)p->selector_succ_codebook;
    const int hproj_slots = gemv12_mode ? FULLDRAFTER_TOTAL_K : BLOCK_K;
    const uint32_t hproj_tag = *(const uint32_t*)(s_base + SCR_TEMP_OFF + hproj_slots * S_RANK * sizeof(float));

    if (hproj_tag == SEL_HPROJ_READY_TAG && w_hproj && w_pred) {
        if (tid < S_RANK) {
            float sum = ((const float*)(s_base + SCR_TEMP_OFF))[logit_slot_idx * S_RANK + tid];
            float pred_emb = bf16_to_f32(w_pred[(size_t)ptok_clamp * S_RANK + tid]);
            s_q[tid] = sum * pred_emb;
        }
    } else {
        __shared__ float s_h_slot[H_DIM];
        const float* h_slot = (eff_stage == 1) ? (const float*)(s_base + SCR_H1_OFF + (size_t)local_slot * H_DIM * sizeof(float))
                                               : (const float*)(s_base + SCR_H2_OFF + (size_t)local_slot * H_DIM * sizeof(float));
        #pragma unroll
        for (int i = tid; i < H_DIM; i += 256) {
            s_h_slot[i] = h_slot[i];
        }
        __syncthreads();

        if (tid < S_RANK) {
            if (w_hproj && w_pred) {
                const uint16_t* hp_row = w_hproj + (size_t)tid * H_DIM;
                float sum = 0.0f;
                #pragma unroll 4
                for (int i = 0; i < H_DIM; i += 8) {
                    uint4 raw = *reinterpret_cast<const uint4*>(hp_row + i);
                    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sum += bf16_to_f32(h[j]) * s_h_slot[i + j];
                    }
                }
                float pred_emb = bf16_to_f32(w_pred[(size_t)ptok_clamp * S_RANK + tid]);
                s_q[tid] = sum * pred_emb;
            } else {
                s_q[tid] = 0.0f;
            }
        }
    }
    __syncthreads();

    // 7 & 8. Fused Selector Scoring, Proposal Store, and 4-Step Warp-Shuffle Argmax (0 barriers!)
    if (tid < K_TOP) {
        uint32_t cid = s_top16_id[tid];
        float bias = 0.0f;
        if (w_succ && cid < V_TOTAL) {
            const uint16_t* succ_row = w_succ + (size_t)cid * S_RANK;
            #pragma unroll
            for (int r = 0; r < S_RANK; r += 8) {
                uint4 raw = *reinterpret_cast<const uint4*>(succ_row + r);
                const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    bias += bf16_to_f32(h[j]) * s_q[r + j];
                }
            }
        }
        float my_score = s_top16_val[tid] + bias;
        uint32_t* out_pids  = (uint32_t*)(s_base + SCR_PROP_ID_OFF + (size_t)global_slot_idx * K_TOP * sizeof(uint32_t));
        float*    out_pvals = (float*)(s_base + SCR_PROP_VAL_OFF + (size_t)global_slot_idx * K_TOP * sizeof(float));
        out_pids[tid]  = cid;
        out_pvals[tid] = my_score;

        float best_score = my_score;
        uint32_t best_id = cid;
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
            float ov = __shfl_down(best_score, offset, 32);
            uint32_t oi = __shfl_down(best_id, offset, 32);
            if (is_better(ov, oi, best_score, best_id)) {
                best_score = ov;
                best_id = oi;
            }
        }
        if (tid == 0) {
            uint32_t winning_tok = best_id;
            if (p->is_alodlm_loop != 0u) {
                const uint32_t* exits = (const uint32_t*)(s_base + ALODLM_EXITS_OFF);
                const uint32_t* ids   = (const uint32_t*)(s_base + ALODLM_IDS_OFF);
                uint32_t* pred        = (uint32_t*)(s_base + ALODLM_PRED_OFF);
                uint32_t chosen       = (exits[global_slot_idx] == 0u) ? winning_tok : ids[global_slot_idx];
                pred[global_slot_idx] = chosen;
                ((uint32_t*)(s_base + SCR_PREV_TOK_OFF))[global_slot_idx + 1] = chosen;
                if (global_slot_idx == FULLDRAFTER_TOTAL_K - 1) {
                    dev_alodlm_commit_step(p, s_base, tokens_out, s, max_depth);
                }
            } else {
                tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + global_slot_idx] = winning_tok;
                ((uint32_t*)(s_base + SCR_PREV_TOK_OFF))[global_slot_idx + 1] = winning_tok;
            }
        }
    }
}


// ALoDLM-4 Single-Launch Persistent 12-Slot Markov + Entropy + P2P + Selector + Commit Kernel (M == 1, 64 CUs)
// Replaces 12 serial slot_markov_select launches + 1 commit launch per depth with ONE single kernel launch!
// Zero local register arrays -> 0 Scratch, 0 Spills, 16 Waves/SIMD (100% Occupancy).
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_all_slots_select_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch,
    uint32_t* __restrict__ tokens_out,
    int M,
    int depth,
    int max_depth,
    int active_k,
    int slot_range_start,
    int slot_range_end,
    int64_t total_scratch_bytes,
    bool fast_selector = false
) {
    const int b = blockIdx.x; // 0..63 (1 WG per CU across all 64 CUs, 100% resident for any M >= 1!)
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 3;
    const int sub  = lane & 7;

    __shared__ float    s_cand_v[16][256]; // 16 KB LDS (bank-conflict-free by tid)
    __shared__ __align__(16) uint32_t s_cand_i[16][256]; // 16 KB LDS (bank-conflict-free by tid)
    __shared__ uint32_t s_exits_init[FULLDRAFTER_TOTAL_K];
    __shared__ int      s_last_unres;
    __shared__ uint32_t s_epoch_base;
    __shared__ uint32_t s_ha_pk[M_RANK / 2];
    __shared__ uint32_t s_ha_nz;
    __shared__ float    s_a_alpha;
    __shared__ float    s_a_norm;
    __shared__ float    s_guaranteed_16th;
    __shared__ float    s_wave_dot[8][32];
    __shared__ float    s_q[S_RANK];
    __shared__ __align__(16) uint32_t s_top16_id[K_TOP];
    __shared__ __align__(16) float s_top16_val[K_TOP];
    __shared__ float    s_top16_score[K_TOP];
    __shared__ float    s_shard_ent[3];
    __shared__ unsigned int s_arrived;
    __shared__ uint32_t s_wave_active[8];

    const bool tp2_p2p = (p->tp2_p2p_enabled != 0u) && (p->my_p2p_comm != nullptr) &&
                         (p->peer_p2p_comm != nullptr) && (p->d_step_seq != nullptr);
    const bool is_dual_rank = (!tp2_p2p) && (p->teacher_lm_head_rank1 != nullptr);
    const int total_vocab = is_dual_rank ? V_TOTAL : V_SHARD;
    const uint32_t vocab_offset = (tp2_p2p && p->rank_id == 1u) ? (uint32_t)V_SHARD : 0u;
    const int v_per_block = (total_vocab + 63) / 64;
    const int v_start = b * v_per_block;
    const int v_end = min(v_start + v_per_block, total_vocab);
    const uint16_t* mb_matrix = (const uint16_t*)p->markov_matrix_b;
    const float*    mb_norm   = p->markov_b_norm;
    const int max_mcache_slots = dev_get_alodlm_max_mcache_slots(M, total_scratch_bytes, total_vocab, active_k);
    const bool is_final_range = (slot_range_end >= active_k - 1);
    const bool need_entropy   = (depth < max_depth - 1);

    #pragma nounroll
    for (int s = 0; s < M; ++s) {
        char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
        if (tid < FULLDRAFTER_TOTAL_K) {
            s_exits_init[tid] = ((const uint32_t*)(s_base + ALODLM_EXITS_OFF))[tid];
        }
        if (tid == 0) {
            s_epoch_base = __hip_atomic_load((const uint32_t*)(s_base + ALODLM_SLOT_EPOCH_OFF), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();

        if (tid == 0) {
            int lu = -1;
            for (int k = slot_range_start; k <= slot_range_end; ++k) {
                if (s_exits_init[k] == 0u) lu = k;
            }
            s_last_unres = lu;
        }
        __syncthreads();

        const int last_unres = s_last_unres;
        if (last_unres < 0) {
            if (is_final_range && b == 0 && tid == 0 && !dloop_stage2_skipped(s_base)) {
                dev_alodlm_commit_step(p, s_base, tokens_out, s, max_depth, active_k);
            }
            __syncthreads();
            continue;
        }

        uint32_t my_epoch = s_epoch_base;

        #pragma nounroll
        for (int global_slot_idx = slot_range_start; global_slot_idx <= last_unres; ++global_slot_idx) {
            if (s_exits_init[global_slot_idx] != 0u) continue;

            __shared__ uint32_t s_ptok_clamp;
            __shared__ int      s_mcache_hit;
            __shared__ int      s_mcache_write;
            const int logit_slot_idx = (total_vocab <= V_SHARD) ? global_slot_idx : (global_slot_idx - slot_range_start);
            const float* s_base_logits = (const float*)(s_base + SCR_BASE_LOGITS_OFF + (size_t)logit_slot_idx * total_vocab * sizeof(float));

        if (tid == 0) {
            s_ha_nz = 0u;
            s_a_alpha = 0.0f;
            s_a_norm = 0.0f;
            s_guaranteed_16th = r4d::FULLDRAFTER_SENTINEL_SCORE;
            uint32_t prev_tok = __hip_atomic_load(
                (const uint32_t*)(s_base + SCR_PREV_TOK_OFF) + global_slot_idx,
                __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT
            );
            uint32_t ptc = (prev_tok < V_TOTAL) ? prev_tok : 0u;
            s_ptok_clamp = ptc;

            int m_hit = -1;
            int m_write = -1;
            const uint32_t* mcache_tok = (const uint32_t*)(s_base + ALODLM_MCACHE_TOK_OFF);
            for (int c = 0; c < max_mcache_slots; ++c) {
                uint32_t ctok = __hip_atomic_load(&mcache_tok[c], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                if (ctok == ptc) { m_hit = c; break; }
            }
            if (m_hit < 0) {
                if (global_slot_idx < max_mcache_slots &&
                    __hip_atomic_load(&mcache_tok[global_slot_idx], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0xFFFFFFFFu) {
                    m_write = global_slot_idx;
                } else {
                    for (int c = 0; c < max_mcache_slots; ++c) {
                        uint32_t ctok = __hip_atomic_load(&mcache_tok[c], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                        if (ctok == 0xFFFFFFFFu) { m_write = c; break; }
                    }
                    if (m_write < 0) {
                        m_write = (global_slot_idx < max_mcache_slots) ? global_slot_idx : (global_slot_idx % max_mcache_slots);
                    }
                }
            }
            s_mcache_hit   = m_hit;
            s_mcache_write = m_write;
        }
        __syncthreads();
        const uint32_t ptok_clamp = s_ptok_clamp;

        if (tid < 8) {
            const uint4* a_vec = reinterpret_cast<const uint4*>(
                (const uint16_t*)p->markov_matrix_a + (size_t)ptok_clamp * M_RANK
            );
            uint4 raw = a_vec[tid];
            s_ha_pk[tid * 4 + 0] = raw.x;
            s_ha_pk[tid * 4 + 1] = raw.y;
            s_ha_pk[tid * 4 + 2] = raw.z;
            s_ha_pk[tid * 4 + 3] = raw.w;
            uint32_t nz = (raw.x | raw.y | raw.z | raw.w) & 0x7FFF7FFFu;
            if (nz) atomicOr(&s_ha_nz, 1u);
        }
        __syncthreads();

        const bool has_markov = (mb_matrix != nullptr) && (s_ha_nz != 0u);
        const uint32_t ha_sub0 = s_ha_pk[sub * 4 + 0];
        const uint32_t ha_sub1 = s_ha_pk[sub * 4 + 1];
        const uint32_t ha_sub2 = s_ha_pk[sub * 4 + 2];
        const uint32_t ha_sub3 = s_ha_pk[sub * 4 + 3];
        const int mcache_hit   = has_markov ? s_mcache_hit   : -1;
        const float* mcache_hit_ptr = (mcache_hit >= 0)
            ? dev_get_alodlm_mcache_slot_ptr(scratch, s_base, s, M, total_scratch_bytes, total_vocab, active_k, mcache_hit)
            : nullptr;

        // 1B. Compute L2 norm of a_prev and exact Cauchy-Schwarz lower bound s_guaranteed_16th on final depth (!need_entropy)
        const bool can_cs = !need_entropy && (mcache_hit_ptr == nullptr) && has_markov && (mb_norm != nullptr) &&
                            (*(const uint32_t*)(s_base + UNARY_BUCKET_READY_OFF) == UNARY_BUCKET_MAGIC);
        if (can_cs) {
            if (tid < 32) {
                uint32_t pk = s_ha_pk[tid];
                float f0 = bf16_to_f32((uint16_t)(pk & 0xFFFFu));
                float f1 = bf16_to_f32((uint16_t)(pk >> 16));
                const float* mb_u = mb_norm + 3 * V_TOTAL;
                float u0 = mb_u[2 * tid + 0];
                float u1 = mb_u[2 * tid + 1];
                float alpha = r4d_wave_sum(f0 * u0 + f1 * u1);
                alpha = __shfl(alpha, 0, 32);
                float d0 = f0 - alpha * u0;
                float d1 = f1 - alpha * u1;
                float sq_perp = r4d_wave_sum(d0 * d0 + d1 * d1);
                if (tid == 0) {
                    s_a_alpha = alpha;
                    s_a_norm  = sqrtf(sq_perp) * 1.001f;
                }
            }
            const uint64_t packed = ((const uint64_t*)(s_base + UNARY_BUCKET_OFF))[global_slot_idx * 256 + tid];
            float u_val = (packed != 0ULL) ? ordered_u32_to_float((uint32_t)(packed >> 32))
                                           : r4d::FULLDRAFTER_SENTINEL_SCORE;
            uint32_t u_id = (packed != 0ULL) ? (uint32_t)(packed & 0xFFFFFFFFu) : r4d::FULLDRAFTER_SENTINEL_ID;
            #pragma unroll
            for (int pick = 0; pick < 2; ++pick) {
                float bv = u_val;
                uint32_t bi = u_id;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float ov = __shfl_down(bv, offset, 32);
                    uint32_t oi = __shfl_down(bi, offset, 32);
                    if (is_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
                }
                uint32_t win_i = __shfl(bi, 0, 32);
                if (lane == 0) {
                    s_top16_val[wave * 2 + pick] = bv;
                    s_top16_id[wave * 2 + pick]  = bi;
                }
                if (u_id == win_i) {
                    u_val = r4d::FULLDRAFTER_SENTINEL_SCORE;
                    u_id  = r4d::FULLDRAFTER_SENTINEL_ID;
                }
            }
            __syncthreads();

            if (tid < 128) {
                const int grp16 = tid >> 3;
                const uint32_t uid = s_top16_id[grp16];
                const uint32_t ug_row = vocab_offset + uid;
                float part = 0.0f;
                if (uid < (uint32_t)total_vocab && ug_row < (uint32_t)V_ACT) {
                    const r4d_u32x4* mb_row_u4 = reinterpret_cast<const r4d_u32x4*>(
                        mb_matrix + (size_t)ug_row * M_RANK
                    );
                    r4d_u32x4 raw = __builtin_nontemporal_load(&mb_row_u4[sub]);
                    float p0 = r4d_dot2_f32_bf16(raw[0], ha_sub0, 0.0f);
                    float p1 = r4d_dot2_f32_bf16(raw[1], ha_sub1, 0.0f);
                    float p2 = r4d_dot2_f32_bf16(raw[2], ha_sub2, 0.0f);
                    float p3 = r4d_dot2_f32_bf16(raw[3], ha_sub3, 0.0f);
                    part = (((0.0f + p0) + p1) + p2) + p3;
                }
                part += __shfl_down(part, 4, 8);
                part += __shfl_down(part, 2, 8);
                part += __shfl_down(part, 1, 8);
                if (sub == 0) {
                    s_top16_score[grp16] = (uid < (uint32_t)total_vocab && ug_row < (uint32_t)V_ACT)
                                               ? (s_top16_val[grp16] + part)
                                               : r4d::FULLDRAFTER_SENTINEL_SCORE;
                }
            }
            __syncthreads();
            if (tid < 16) {
                float sc = s_top16_score[tid];
                #pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1) {
                    float other = __shfl_down(sc, offset, 32);
                    if (other < sc) sc = other;
                }
                if (tid == 0 && sc > -1e20f) {
                    s_guaranteed_16th = sc - 1e-5f;
                }
            }
            __syncthreads();
        }

        const float a_alpha = s_a_alpha;
        const float a_norm = s_a_norm;
        const float guar_16th = s_guaranteed_16th;
        const bool use_cs_prune = can_cs && (guar_16th > -1e20f);
        const int mcache_write = (has_markov && !use_cs_prune) ? s_mcache_write : -1;
        float* mcache_write_ptr = (mcache_write >= 0)
            ? dev_get_alodlm_mcache_slot_ptr(scratch, s_base, s, M, total_scratch_bytes, total_vocab, active_k, mcache_write)
            : nullptr;

        const int m_limit = (v_end - v_start + 255) >> 8;
        uint32_t wave_active_m = 0u;
        #pragma nounroll
        for (int m = 0; m < m_limit; ++m) {
            const int v_wave_base = v_start + m * 256 + wave * 32;
            const int v = v_wave_base + lane;
            const uint32_t v_global = vocab_offset + (uint32_t)v;
            const bool valid_v = (v < v_end) && (v_global < (uint32_t)V_ACT);
            const float my_base = valid_v ? s_base_logits[v] : r4d::FULLDRAFTER_SENTINEL_SCORE;

            if (!has_markov) {
                wave_active_m |= (1u << m);
                s_cand_v[m][tid] = my_base;
                s_cand_i[m][tid] = valid_v ? v_global : r4d::FULLDRAFTER_SENTINEL_ID;
                continue;
            }
            if (mcache_hit_ptr != nullptr) {
                wave_active_m |= (1u << m);
                s_cand_v[m][tid] = valid_v ? (my_base + mcache_hit_ptr[v]) : r4d::FULLDRAFTER_SENTINEL_SCORE;
                s_cand_i[m][tid] = valid_v ? v_global : r4d::FULLDRAFTER_SENTINEL_ID;
                continue;
            }
            const float2 mb_bn = (valid_v && use_cs_prune)
                ? reinterpret_cast<const float2*>(mb_norm + V_TOTAL)[v_global]
                : make_float2(0.0f, 0.0f);
            const bool my_active = valid_v && (!use_cs_prune || ((my_base + a_alpha * mb_bn.x + a_norm * mb_bn.y) >= guar_16th));
            const uint32_t active_mask = __builtin_amdgcn_ballot_w32(my_active);
            if (active_mask != 0u) wave_active_m |= (1u << m);
            if (active_mask == 0u) {
                if (need_entropy) {
                    s_cand_v[m][tid] = valid_v ? my_base : r4d::FULLDRAFTER_SENTINEL_SCORE;
                    s_cand_i[m][tid] = valid_v ? 0x80000000u : r4d::FULLDRAFTER_SENTINEL_ID;
                }
                continue;
            }
            if (active_mask == 0xFFFFFFFFu) {
                #pragma nounroll
                for (int s_base_step = 0; s_base_step < 8; s_base_step += 4) {
                    const uint32_t vg0 = vocab_offset + (uint32_t)(v_wave_base + (s_base_step + 0) * 4 + grp);
                    const uint32_t vg1 = vocab_offset + (uint32_t)(v_wave_base + (s_base_step + 1) * 4 + grp);
                    const uint32_t vg2 = vocab_offset + (uint32_t)(v_wave_base + (s_base_step + 2) * 4 + grp);
                    const uint32_t vg3 = vocab_offset + (uint32_t)(v_wave_base + (s_base_step + 3) * 4 + grp);
                    r4d_u32x4 raw0 = __builtin_nontemporal_load(&reinterpret_cast<const r4d_u32x4*>(mb_matrix + (size_t)vg0 * M_RANK)[sub]);
                    r4d_u32x4 raw1 = __builtin_nontemporal_load(&reinterpret_cast<const r4d_u32x4*>(mb_matrix + (size_t)vg1 * M_RANK)[sub]);
                    r4d_u32x4 raw2 = __builtin_nontemporal_load(&reinterpret_cast<const r4d_u32x4*>(mb_matrix + (size_t)vg2 * M_RANK)[sub]);
                    r4d_u32x4 raw3 = __builtin_nontemporal_load(&reinterpret_cast<const r4d_u32x4*>(mb_matrix + (size_t)vg3 * M_RANK)[sub]);

                    float pt0 = (((0.0f + r4d_dot2_f32_bf16(raw0[0], ha_sub0, 0.0f)) + r4d_dot2_f32_bf16(raw0[1], ha_sub1, 0.0f)) + r4d_dot2_f32_bf16(raw0[2], ha_sub2, 0.0f)) + r4d_dot2_f32_bf16(raw0[3], ha_sub3, 0.0f);
                    float pt1 = (((0.0f + r4d_dot2_f32_bf16(raw1[0], ha_sub0, 0.0f)) + r4d_dot2_f32_bf16(raw1[1], ha_sub1, 0.0f)) + r4d_dot2_f32_bf16(raw1[2], ha_sub2, 0.0f)) + r4d_dot2_f32_bf16(raw1[3], ha_sub3, 0.0f);
                    float pt2 = (((0.0f + r4d_dot2_f32_bf16(raw2[0], ha_sub0, 0.0f)) + r4d_dot2_f32_bf16(raw2[1], ha_sub1, 0.0f)) + r4d_dot2_f32_bf16(raw2[2], ha_sub2, 0.0f)) + r4d_dot2_f32_bf16(raw2[3], ha_sub3, 0.0f);
                    float pt3 = (((0.0f + r4d_dot2_f32_bf16(raw3[0], ha_sub0, 0.0f)) + r4d_dot2_f32_bf16(raw3[1], ha_sub1, 0.0f)) + r4d_dot2_f32_bf16(raw3[2], ha_sub2, 0.0f)) + r4d_dot2_f32_bf16(raw3[3], ha_sub3, 0.0f);

                    pt0 += __shfl_down(pt0, 4, 8); pt0 += __shfl_down(pt0, 2, 8); pt0 += __shfl_down(pt0, 1, 8);
                    pt1 += __shfl_down(pt1, 4, 8); pt1 += __shfl_down(pt1, 2, 8); pt1 += __shfl_down(pt1, 1, 8);
                    pt2 += __shfl_down(pt2, 4, 8); pt2 += __shfl_down(pt2, 2, 8); pt2 += __shfl_down(pt2, 1, 8);
                    pt3 += __shfl_down(pt3, 4, 8); pt3 += __shfl_down(pt3, 2, 8); pt3 += __shfl_down(pt3, 1, 8);

                    if (sub == 0) {
                        s_wave_dot[wave][(s_base_step + 0) * 4 + grp] = pt0;
                        s_wave_dot[wave][(s_base_step + 1) * 4 + grp] = pt1;
                        s_wave_dot[wave][(s_base_step + 2) * 4 + grp] = pt2;
                        s_wave_dot[wave][(s_base_step + 3) * 4 + grp] = pt3;
                    }
                }
            } else {
                #pragma nounroll
                for (int s_step = 0; s_step < 8; ++s_step) {
                    const uint32_t step_bits = (active_mask >> (s_step * 4)) & 0xFu;
                    if (step_bits == 0u) continue;
                    const int v_row = v_wave_base + s_step * 4 + grp;
                    const uint32_t vg_row = vocab_offset + (uint32_t)v_row;
                    float part = 0.0f;
                    if ((step_bits >> grp) & 1u) {
                        const r4d_u32x4* mb_row_u4 = reinterpret_cast<const r4d_u32x4*>(
                            mb_matrix + (size_t)vg_row * M_RANK
                        );
                        r4d_u32x4 raw = __builtin_nontemporal_load(&mb_row_u4[sub]);
                        float p0 = r4d_dot2_f32_bf16(raw[0], ha_sub0, 0.0f);
                        float p1 = r4d_dot2_f32_bf16(raw[1], ha_sub1, 0.0f);
                        float p2 = r4d_dot2_f32_bf16(raw[2], ha_sub2, 0.0f);
                        float p3 = r4d_dot2_f32_bf16(raw[3], ha_sub3, 0.0f);
                        part = (((0.0f + p0) + p1) + p2) + p3;
                        part += __shfl_down(part, 4, 8);
                        part += __shfl_down(part, 2, 8);
                        part += __shfl_down(part, 1, 8);
                    }
                    if (sub == 0) s_wave_dot[wave][s_step * 4 + grp] = part;
                }
            }
            if (my_active) {
                float m_dot = s_wave_dot[wave][lane];
                if (mcache_write_ptr != nullptr) mcache_write_ptr[v] = m_dot;
                s_cand_v[m][tid] = my_base + m_dot;
                s_cand_i[m][tid] = v_global;
            } else {
                s_cand_v[m][tid] = valid_v ? my_base : r4d::FULLDRAFTER_SENTINEL_SCORE;
                s_cand_i[m][tid] = valid_v ? 0x80000000u : r4d::FULLDRAFTER_SENTINEL_ID;
            }
        }

        uint32_t* blk_cands = (uint32_t*)(s_base + SCR_BLK_CANDS_OFF + (size_t)b * K_TOP * sizeof(uint32_t));
        float*    blk_unary = (float*)(s_base + SCR_BLK_UNARY_OFF + (size_t)b * K_TOP * sizeof(float));
        if (lane == 0) s_wave_active[wave] = wave_active_m;
        __syncthreads();
        float*    s_wtop_val = &s_wave_dot[0][0];
        uint32_t* s_wtop_id  = reinterpret_cast<uint32_t*>(&s_wave_dot[4][0]);
        uint32_t wg_active = 0u;
        #pragma unroll
        for (int w = 0; w < 8; ++w) wg_active |= s_wave_active[w];
        const bool empty_block = !need_entropy && wg_active == 0u;
        if (empty_block && tid < K_TOP) {
            __hip_atomic_store((uint32_t*)&blk_unary[tid], __builtin_bit_cast(uint32_t, r4d::FULLDRAFTER_SENTINEL_SCORE), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&blk_cands[tid], r4d::FULLDRAFTER_SENTINEL_ID, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        if (!empty_block && !need_entropy && wave_active_m == 0u && lane < K_TOP) {
            s_wtop_val[wave * K_TOP + lane] = r4d::FULLDRAFTER_SENTINEL_SCORE;
            s_wtop_id[wave * K_TOP + lane] = r4d::FULLDRAFTER_SENTINEL_ID;
        }
        if (!empty_block && (need_entropy || wave_active_m != 0u)) {
            float my_val = r4d::FULLDRAFTER_SENTINEL_SCORE;
            uint32_t my_id = r4d::FULLDRAFTER_SENTINEL_ID;
            int best_m = 0;
            #pragma nounroll
            for (int m = 0; m < m_limit; ++m) {
                if (!need_entropy && ((wave_active_m >> m) & 1u) == 0u) continue;
                uint32_t cim = s_cand_i[m][tid];
                float cv = (cim < 0x80000000u) ? s_cand_v[m][tid] : r4d::FULLDRAFTER_SENTINEL_SCORE;
                uint32_t ci = (cim < 0x80000000u) ? cim : r4d::FULLDRAFTER_SENTINEL_ID;
                if (is_better(cv, ci, my_val, my_id)) { my_val = cv; my_id = ci; best_m = m; }
            }
            #pragma nounroll
            for (int k = 0; k < K_TOP; ++k) {
                float w_val = my_val;
                uint32_t w_id = my_id;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float ov = __shfl_down(w_val, offset, 32);
                    uint32_t oi = __shfl_down(w_id, offset, 32);
                    if (is_better(ov, oi, w_val, w_id)) { w_val = ov; w_id = oi; }
                }
                uint32_t win_id = __shfl(w_id, 0, 32);
                if (!need_entropy && win_id == r4d::FULLDRAFTER_SENTINEL_ID) {
                    if (lane < K_TOP - k) {
                        s_wtop_val[wave * K_TOP + k + lane] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                        s_wtop_id[wave * K_TOP + k + lane]  = r4d::FULLDRAFTER_SENTINEL_ID;
                    }
                    break;
                }
                if (lane == 0) {
                    s_wtop_val[wave * K_TOP + k] = w_val;
                    s_wtop_id[wave * K_TOP + k]  = w_id;
                }
                if (my_id == win_id) {
                    s_cand_v[best_m][tid] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                    s_cand_i[best_m][tid] = r4d::FULLDRAFTER_SENTINEL_ID;
                    my_val = r4d::FULLDRAFTER_SENTINEL_SCORE;
                    my_id  = r4d::FULLDRAFTER_SENTINEL_ID;
                    best_m = 0;
                    #pragma nounroll
                    for (int m = 0; m < m_limit; ++m) {
                        if (!need_entropy && ((wave_active_m >> m) & 1u) == 0u) continue;
                        uint32_t cim = s_cand_i[m][tid];
                        float cv = (cim < 0x80000000u) ? s_cand_v[m][tid] : r4d::FULLDRAFTER_SENTINEL_SCORE;
                        uint32_t ci = (cim < 0x80000000u) ? cim : r4d::FULLDRAFTER_SENTINEL_ID;
                        if (is_better(cv, ci, my_val, my_id)) { my_val = cv; my_id = ci; best_m = m; }
                    }
                }
            }
        }
        __syncthreads();

        if (need_entropy) {
            if (tid == 0) {
                float bmax = s_wtop_val[0];
                for (int w = 1; w < 8; ++w) {
                    float w0 = s_wtop_val[w * K_TOP];
                    if (w0 > bmax) bmax = w0;
                }
                s_top16_val[0] = bmax;
            }
            __syncthreads();

            const float blk_max = s_top16_val[0];
            float my_se = 0.0f, my_sxe = 0.0f;
            if (blk_max > -1e20f) {
                if (tid < 128) {
                    float wv = s_wtop_val[tid];
                    if (wv > -1e20f) {
                        float x = wv - blk_max;
                        if (x > -80.0f) { float e = expf(x); my_se += e; my_sxe += x * e; }
                    }
                }
                #pragma nounroll
                for (int m = 0; m < m_limit; ++m) {
                    if (s_cand_i[m][tid] != r4d::FULLDRAFTER_SENTINEL_ID) {
                        float x = s_cand_v[m][tid] - blk_max;
                        if (x > -80.0f) { float e = expf(x); my_se += e; my_sxe += x * e; }
                    }
                }
            }
            float tot_se  = block_reduce_sum(my_se);
            float tot_sxe = block_reduce_sum(my_sxe);
            if (tid == 0) {
                uint32_t* bent_u = (uint32_t*)(s_base + ALODLM_BLK_ENT_OFF) + b * 4;
                __hip_atomic_store(&bent_u[0], __builtin_bit_cast(uint32_t, blk_max), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&bent_u[1], __builtin_bit_cast(uint32_t, tot_se),  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&bent_u[2], __builtin_bit_cast(uint32_t, tot_sxe), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            __syncthreads();
        }

        // Wave-0 Top-16 in-place on s_wtop_val / s_wtop_id (0 local arrays!)
        if (!empty_block && tid < 32) {
            float v0 = s_wtop_val[tid * 4 + 0], v1 = s_wtop_val[tid * 4 + 1], v2 = s_wtop_val[tid * 4 + 2], v3 = s_wtop_val[tid * 4 + 3];
            uint32_t i0 = s_wtop_id[tid * 4 + 0], i1 = s_wtop_id[tid * 4 + 1], i2 = s_wtop_id[tid * 4 + 2], i3 = s_wtop_id[tid * 4 + 3];
            #pragma nounroll
            for (int k = 0; k < K_TOP; ++k) {
                float bv = v0; uint32_t bi = i0; int best_c = 0;
                if (is_better(v1, i1, bv, bi)) { bv = v1; bi = i1; best_c = 1; }
                if (is_better(v2, i2, bv, bi)) { bv = v2; bi = i2; best_c = 2; }
                if (is_better(v3, i3, bv, bi)) { bv = v3; bi = i3; best_c = 3; }
                float w_bv = bv;
                uint32_t w_bi = bi;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float ov = __shfl_down(w_bv, offset, 32);
                    uint32_t oi = __shfl_down(w_bi, offset, 32);
                    if (is_better(ov, oi, w_bv, w_bi)) { w_bv = ov; w_bi = oi; }
                }
                uint32_t win_i = __shfl(w_bi, 0, 32);
                if (!need_entropy && win_i == r4d::FULLDRAFTER_SENTINEL_ID) {
                    if (tid < K_TOP - k) {
                        __hip_atomic_store((uint32_t*)&blk_unary[k + tid], __builtin_bit_cast(uint32_t, r4d::FULLDRAFTER_SENTINEL_SCORE), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                        __hip_atomic_store(&blk_cands[k + tid], r4d::FULLDRAFTER_SENTINEL_ID, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    }
                    break;
                }
                if (tid == 0) {
                    __hip_atomic_store((uint32_t*)&blk_unary[k], __builtin_bit_cast(uint32_t, w_bv), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                    __hip_atomic_store(&blk_cands[k], w_bi, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
                if (bi == win_i) {
                    if (best_c == 0) { v0 = r4d::FULLDRAFTER_SENTINEL_SCORE; i0 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    else if (best_c == 1) { v1 = r4d::FULLDRAFTER_SENTINEL_SCORE; i1 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    else if (best_c == 2) { v2 = r4d::FULLDRAFTER_SENTINEL_SCORE; i2 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    else { v3 = r4d::FULLDRAFTER_SENTINEL_SCORE; i3 = r4d::FULLDRAFTER_SENTINEL_ID; }
                }
            }
        }

        ++my_epoch;
        __threadfence();
        __syncthreads();

        if (tid == 0) {
            unsigned int* ctr = (unsigned int*)(s_base + SCR_ARRIVAL_CTR_OFF);
            s_arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();

        const bool is_winner = (s_arrived == (unsigned int)(gridDim.x - 1));
        if (is_winner && tid == 0) {
            unsigned int* ctr = (unsigned int*)(s_base + SCR_ARRIVAL_CTR_OFF);
            *ctr = 0u;
            if (mcache_write >= 0) {
                __hip_atomic_store(
                    (uint32_t*)(s_base + ALODLM_MCACHE_TOK_OFF) + mcache_write,
                    ptok_clamp, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT
                );
            }
        }
        if (is_winner) __threadfence();

        // Step 5: Stage 1024 candidates into s_cand_v[0..3][tid] / s_cand_i[0..3][tid] in LDS (0 local arrays!)
        if (is_winner) {
            const uint32_t* all_cands = (const uint32_t*)(s_base + SCR_BLK_CANDS_OFF);
            const uint32_t* all_unary_u = (const uint32_t*)(s_base + SCR_BLK_UNARY_OFF);
            float cv0 = __builtin_bit_cast(float, __hip_atomic_load(&all_unary_u[tid + 0 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
            float cv1 = __builtin_bit_cast(float, __hip_atomic_load(&all_unary_u[tid + 1 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
            float cv2 = __builtin_bit_cast(float, __hip_atomic_load(&all_unary_u[tid + 2 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
            float cv3 = __builtin_bit_cast(float, __hip_atomic_load(&all_unary_u[tid + 3 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
            uint32_t ci0 = __hip_atomic_load(&all_cands[tid + 0 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            uint32_t ci1 = __hip_atomic_load(&all_cands[tid + 1 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            uint32_t ci2 = __hip_atomic_load(&all_cands[tid + 2 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            uint32_t ci3 = __hip_atomic_load(&all_cands[tid + 3 * 256], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            const bool lane_active = (ci0 < 0x80000000u) || (ci1 < 0x80000000u) || (ci2 < 0x80000000u) || (ci3 < 0x80000000u);
            const bool empty_wave = !need_entropy && use_cs_prune &&
                __builtin_amdgcn_ballot_w32(lane_active) == 0u;
            if (empty_wave && lane < K_TOP) {
                s_wtop_val[wave * K_TOP + lane] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                s_wtop_id[wave * K_TOP + lane] = r4d::FULLDRAFTER_SENTINEL_ID;
            }
            if (!empty_wave) {
                #pragma nounroll
                for (int k = 0; k < K_TOP; ++k) {
                    float my_best_val = cv0; uint32_t my_best_id = ci0; int best_c = 0;
                    if (is_better(cv1, ci1, my_best_val, my_best_id)) { my_best_val = cv1; my_best_id = ci1; best_c = 1; }
                    if (is_better(cv2, ci2, my_best_val, my_best_id)) { my_best_val = cv2; my_best_id = ci2; best_c = 2; }
                    if (is_better(cv3, ci3, my_best_val, my_best_id)) { my_best_val = cv3; my_best_id = ci3; best_c = 3; }
                    float w_val = my_best_val;
                    uint32_t w_id = my_best_id;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        float ov = __shfl_down(w_val, offset, 32);
                        uint32_t oi = __shfl_down(w_id, offset, 32);
                        if (is_better(ov, oi, w_val, w_id)) { w_val = ov; w_id = oi; }
                    }
                    uint32_t win_id = __shfl(w_id, 0, 32);
                    if (!need_entropy && win_id == r4d::FULLDRAFTER_SENTINEL_ID) {
                        if (lane < K_TOP - k) {
                            s_wtop_val[wave * K_TOP + k + lane] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                            s_wtop_id[wave * K_TOP + k + lane]  = r4d::FULLDRAFTER_SENTINEL_ID;
                        }
                        break;
                    }
                    if (lane == 0) {
                        s_wtop_val[wave * K_TOP + k] = w_val;
                        s_wtop_id[wave * K_TOP + k]  = w_id;
                    }
                    if (my_best_id == win_id) {
                        if (best_c == 0) { cv0 = r4d::FULLDRAFTER_SENTINEL_SCORE; ci0 = r4d::FULLDRAFTER_SENTINEL_ID; }
                        else if (best_c == 1) { cv1 = r4d::FULLDRAFTER_SENTINEL_SCORE; ci1 = r4d::FULLDRAFTER_SENTINEL_ID; }
                        else if (best_c == 2) { cv2 = r4d::FULLDRAFTER_SENTINEL_SCORE; ci2 = r4d::FULLDRAFTER_SENTINEL_ID; }
                        else { cv3 = r4d::FULLDRAFTER_SENTINEL_SCORE; ci3 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    }
                }
            }
        }
        __syncthreads();

        if (is_winner && tid < 32) {
            float v0 = s_wtop_val[tid * 4 + 0], v1 = s_wtop_val[tid * 4 + 1], v2 = s_wtop_val[tid * 4 + 2], v3 = s_wtop_val[tid * 4 + 3];
            uint32_t i0 = s_wtop_id[tid * 4 + 0], i1 = s_wtop_id[tid * 4 + 1], i2 = s_wtop_id[tid * 4 + 2], i3 = s_wtop_id[tid * 4 + 3];
            #pragma nounroll
            for (int k = 0; k < K_TOP; ++k) {
                float bv = v0; uint32_t bi = i0; int best_c = 0;
                if (is_better(v1, i1, bv, bi)) { bv = v1; bi = i1; best_c = 1; }
                if (is_better(v2, i2, bv, bi)) { bv = v2; bi = i2; best_c = 2; }
                if (is_better(v3, i3, bv, bi)) { bv = v3; bi = i3; best_c = 3; }
                float w_bv = bv;
                uint32_t w_bi = bi;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    float ov = __shfl_down(w_bv, offset, 32);
                    uint32_t oi = __shfl_down(w_bi, offset, 32);
                    if (is_better(ov, oi, w_bv, w_bi)) { w_bv = ov; w_bi = oi; }
                }
                uint32_t win_i = __shfl(w_bi, 0, 32);
                if (!need_entropy && win_i == r4d::FULLDRAFTER_SENTINEL_ID) {
                    if (tid < K_TOP - k) {
                        s_top16_val[k + tid] = r4d::FULLDRAFTER_SENTINEL_SCORE;
                        s_top16_id[k + tid]  = r4d::FULLDRAFTER_SENTINEL_ID;
                    }
                    break;
                }
                if (tid == 0) {
                    s_top16_val[k] = w_bv;
                    s_top16_id[k]  = w_bi;
                }
                if (bi == win_i) {
                    if (best_c == 0) { v0 = r4d::FULLDRAFTER_SENTINEL_SCORE; i0 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    else if (best_c == 1) { v1 = r4d::FULLDRAFTER_SENTINEL_SCORE; i1 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    else if (best_c == 2) { v2 = r4d::FULLDRAFTER_SENTINEL_SCORE; i2 = r4d::FULLDRAFTER_SENTINEL_ID; }
                    else { v3 = r4d::FULLDRAFTER_SENTINEL_SCORE; i3 = r4d::FULLDRAFTER_SENTINEL_ID; }
                }
            }
        }
        __syncthreads();

        if (need_entropy) {
            float scaled_se = 0.0f, scaled_sxe = 0.0f;
            if (is_winner && tid < 64) {
                const uint32_t* bent_u = (const uint32_t*)(s_base + ALODLM_BLK_ENT_OFF);
                float bm   = __builtin_bit_cast(float, __hip_atomic_load(&bent_u[tid * 4 + 0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                float bse  = __builtin_bit_cast(float, __hip_atomic_load(&bent_u[tid * 4 + 1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                float bsxe = __builtin_bit_cast(float, __hip_atomic_load(&bent_u[tid * 4 + 2], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                const float sh_max = s_top16_val[0];
                if (bm > -1e20f && sh_max > -1e20f) {
                    float dm = bm - sh_max;
                    if (dm > -80.0f) {
                        float sc = expf(dm);
                        scaled_se  = bse * sc;
                        scaled_sxe = (bsxe + bse * dm) * sc;
                    }
                }
            }
            float tot_se2  = block_reduce_sum(scaled_se);
            float tot_sxe2 = block_reduce_sum(scaled_sxe);
            if (is_winner && tid == 0) {
                s_shard_ent[0] = s_top16_val[0];
                s_shard_ent[1] = tot_se2;
                s_shard_ent[2] = tot_sxe2;
                if (!tp2_p2p) {
                    float ent = (tot_se2 > 0.0f) ? (logf(tot_se2) - tot_sxe2 / tot_se2) : 0.0f;
                    ((float*)(s_base + ALODLM_ENTROPY_OFF))[global_slot_idx] = ent;
                }
            }
            __syncthreads();
        } else if (is_winner && tid == 0) {
            s_shard_ent[0] = s_top16_val[0];
            s_shard_ent[1] = 1.0f;
            s_shard_ent[2] = (float)(*(const uint32_t*)(s_base + DLOOP_VERIF_READY_OFF));
        }

        // Replace only coarse unary; retain Markov and the coarse entropy partition.
        if (is_winner && p->teacher_lm_head_q8_rank0 != nullptr && p->lm_head_dim == 2560u) {
            const int hidden_slot = global_slot_idx;
            const float* d_h_slot = (const float*)(s_base + (hidden_slot < BLOCK_K ? SCR_O1_OFF : SCR_O2_OFF))
                + (size_t)(hidden_slot % BLOCK_K) * W_DIM;
            uint32_t* s_hidden_pk = &s_cand_i[4][0];
            static_assert(W_DIM % 8 == 0, "Hidden packing requires complete BF16 vectors");
            for (int vec = tid; vec < W_DIM / 8; vec += 256) {
                const uint4 h0 = reinterpret_cast<const uint4*>(d_h_slot)[vec * 2];
                const uint4 h1 = reinterpret_cast<const uint4*>(d_h_slot)[vec * 2 + 1];
                reinterpret_cast<uint4*>(s_hidden_pk)[vec] = make_uint4(
                    (h0.x >> 16) | (h0.y & 0xFFFF0000u),
                    (h0.z >> 16) | (h0.w & 0xFFFF0000u),
                    (h1.x >> 16) | (h1.y & 0xFFFF0000u),
                    (h1.z >> 16) | (h1.w & 0xFFFF0000u));
            }
            __syncthreads();

            // Share hidden LDS reads and overlap both rows, retaining each BF16 sum's order.
            const uint32_t cid0 = s_top16_id[wave * 2 + 0];
            const uint32_t cid1 = s_top16_id[wave * 2 + 1];
            const uint32_t local_id0 = cid0 - vocab_offset;
            const uint32_t local_id1 = cid1 - vocab_offset;
            const bool valid0 = cid0 < (uint32_t)V_ACT && local_id0 < (uint32_t)total_vocab;
            const bool valid1 = cid1 < (uint32_t)V_ACT && local_id1 < (uint32_t)total_vocab;
            const bool shard1_0 = is_dual_rank && cid0 >= (uint32_t)V_SHARD;
            const bool shard1_1 = is_dual_rank && cid1 >= (uint32_t)V_SHARD;
            const uint16_t* shard0 = (const uint16_t*)(shard1_0
                ? p->teacher_lm_head_rank1 : p->teacher_lm_head_rank0);
            const uint16_t* shard1 = (const uint16_t*)(shard1_1
                ? p->teacher_lm_head_rank1 : p->teacher_lm_head_rank0);
            const uint32_t row0 = shard1_0 ? cid0 - (uint32_t)V_SHARD : local_id0;
            const uint32_t row1 = shard1_1 ? cid1 - (uint32_t)V_SHARD : local_id1;
            const r4d_u32x4* lm_row0 = reinterpret_cast<const r4d_u32x4*>(
                shard0 + (size_t)(valid0 ? row0 : 0u) * W_DIM);
            const r4d_u32x4* lm_row1 = reinterpret_cast<const r4d_u32x4*>(
                shard1 + (size_t)(valid1 ? row1 : 0u) * W_DIM);
            float exact0 = 0.0f, exact1 = 0.0f;
            #pragma unroll 2
            for (int step = 0; step < W_DIM / 256; ++step) {
                const int vec = step * 32 + lane;
                const uint32_t h0 = s_hidden_pk[vec * 4 + 0];
                const uint32_t h1 = s_hidden_pk[vec * 4 + 1];
                const uint32_t h2 = s_hidden_pk[vec * 4 + 2];
                const uint32_t h3 = s_hidden_pk[vec * 4 + 3];
                const r4d_u32x4 w0 = valid0 ? __builtin_nontemporal_load(&lm_row0[vec])
                    : r4d_u32x4{0u, 0u, 0u, 0u};
                const r4d_u32x4 w1 = valid1 ? __builtin_nontemporal_load(&lm_row1[vec])
                    : r4d_u32x4{0u, 0u, 0u, 0u};
                const float p00 = r4d_dot2_f32_bf16(w0[0], h0, 0.0f);
                const float p01 = r4d_dot2_f32_bf16(w0[1], h1, 0.0f);
                const float p02 = r4d_dot2_f32_bf16(w0[2], h2, 0.0f);
                const float p03 = r4d_dot2_f32_bf16(w0[3], h3, 0.0f);
                exact0 += (((0.0f + p00) + p01) + p02) + p03;
                const float p10 = r4d_dot2_f32_bf16(w1[0], h0, 0.0f);
                const float p11 = r4d_dot2_f32_bf16(w1[1], h1, 0.0f);
                const float p12 = r4d_dot2_f32_bf16(w1[2], h2, 0.0f);
                const float p13 = r4d_dot2_f32_bf16(w1[3], h3, 0.0f);
                exact1 += (((0.0f + p10) + p11) + p12) + p13;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                exact0 += __shfl_down(exact0, offset, 32);
                exact1 += __shfl_down(exact1, offset, 32);
            }
            if (lane == 0) {
                if (valid0) s_top16_val[wave * 2 + 0] = exact0
                    + (s_top16_val[wave * 2 + 0] - s_base_logits[local_id0]);
                if (valid1) s_top16_val[wave * 2 + 1] = exact1
                    + (s_top16_val[wave * 2 + 1] - s_base_logits[local_id1]);
            }
            __syncthreads();

        }

        if (is_winner && tp2_p2p) {
            const uint32_t step_seq = *p->d_step_seq;
            const uint32_t pkt_idx  = (uint32_t)s * FULLDRAFTER_TOTAL_K + (uint32_t)global_slot_idx;
            const uint32_t tag      = (step_seq * 8u + (uint32_t)depth) * 1024u + pkt_idx + 1u;
            FullDrafterP2PSlotPacket* my_pkt   = (FullDrafterP2PSlotPacket*)p->my_p2p_comm + pkt_idx;
            FullDrafterP2PSlotPacket* peer_pkt = (FullDrafterP2PSlotPacket*)p->peer_p2p_comm + pkt_idx;
            static_assert(K_TOP % 4 == 0, "P2P candidates require complete vectors");
            if (tid < K_TOP / 4) {
                reinterpret_cast<float4*>(peer_pkt->vals)[tid] =
                    reinterpret_cast<const float4*>(s_top16_val)[tid];
                reinterpret_cast<uint4*>(peer_pkt->ids)[tid] =
                    reinterpret_cast<const uint4*>(s_top16_id)[tid];
            }
            if (tid == 0) {
                peer_pkt->shard_max_l  = s_shard_ent[0];
                peer_pkt->shard_sum_e  = s_shard_ent[1];
                peer_pkt->shard_sum_xe = s_shard_ent[2];
            }
            __threadfence_system();
        }
        __syncthreads();
        if (is_winner && tp2_p2p && tid == 0) {
            const uint32_t step_seq = *p->d_step_seq;
            const uint32_t pkt_idx  = (uint32_t)s * FULLDRAFTER_TOTAL_K + (uint32_t)global_slot_idx;
            const uint32_t tag      = (step_seq * 8u + (uint32_t)depth) * 1024u + pkt_idx + 1u;
            FullDrafterP2PSlotPacket* my_pkt   = (FullDrafterP2PSlotPacket*)p->my_p2p_comm + pkt_idx;
            FullDrafterP2PSlotPacket* peer_pkt = (FullDrafterP2PSlotPacket*)p->peer_p2p_comm + pkt_idx;
            __hip_atomic_store(&peer_pkt->flag, tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            uint64_t spin = 0;
            while (__hip_atomic_load(&my_pkt->flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != tag) {
                if (++spin > 200000000ULL) break;
            }
        }
        __syncthreads();
        if (is_winner && tp2_p2p) {
            __threadfence_system();
            const uint32_t pkt_idx = (uint32_t)s * FULLDRAFTER_TOTAL_K + (uint32_t)global_slot_idx;
            FullDrafterP2PSlotPacket* my_pkt = (FullDrafterP2PSlotPacket*)p->my_p2p_comm + pkt_idx;
            if (tid < 32) {
                float cv = (tid < 16) ? s_top16_val[tid] : my_pkt->vals[tid - 16];
                uint32_t ci = (tid < 16) ? s_top16_id[tid] : my_pkt->ids[tid - 16];
                int rank = 0;
                #pragma unroll
                for (int other = 0; other < 32; ++other) {
                    const float ov = __shfl(cv, other, 32);
                    const uint32_t oi = __shfl(ci, other, 32);
                    // Sentinel IDs can repeat when pruning leaves fewer than 16 candidates.
                    rank += is_better(ov, oi, cv, ci) ||
                        (ov == cv && oi == ci && other < tid);
                }
                if (rank < K_TOP) {
                    s_top16_val[rank] = cv;
                    s_top16_id[rank] = ci;
                }
                if (tid == 0) {
                    float m0   = (p->rank_id == 0u) ? s_shard_ent[0] : my_pkt->shard_max_l;
                    float se0  = (p->rank_id == 0u) ? s_shard_ent[1] : my_pkt->shard_sum_e;
                    float sxe0 = (p->rank_id == 0u) ? s_shard_ent[2] : my_pkt->shard_sum_xe;
                    float m1   = (p->rank_id == 0u) ? my_pkt->shard_max_l  : s_shard_ent[0];
                    float se1  = (p->rank_id == 0u) ? my_pkt->shard_sum_e  : s_shard_ent[1];
                    float sxe1 = (p->rank_id == 0u) ? my_pkt->shard_sum_xe : s_shard_ent[2];
                    float mg   = fmaxf(m0, m1);
                    float c0   = (m0 - mg > -80.0f) ? expf(m0 - mg) : 0.0f;
                    float c1   = (m1 - mg > -80.0f) ? expf(m1 - mg) : 0.0f;
                    float seg  = se0 * c0 + se1 * c1;
                    float sxeg = (sxe0 + se0 * (m0 - mg)) * c0 + (sxe1 + se1 * (m1 - mg)) * c1;
                    float ent  = (seg > 0.0f) ? (logf(seg) - sxeg / seg) : 0.0f;
                    ((float*)(s_base + ALODLM_ENTROPY_OFF))[global_slot_idx] = ent;
                }
            }
        }
        __syncthreads();

        if (is_winner && tid < S_RANK) {
            const uint16_t* w_pred = (const uint16_t*)p->selector_pred_codebook;
            if (w_pred) {
                float sum = ((const float*)(s_base + SCR_TEMP_OFF))[global_slot_idx * S_RANK + tid];
                float pred_emb = bf16_to_f32(w_pred[(size_t)ptok_clamp * S_RANK + tid]);
                s_q[tid] = sum * pred_emb;
            } else {
                s_q[tid] = 0.0f;
            }
        }
        __syncthreads();

        if (is_winner && fast_selector && tid < K_TOP * 8) {
            static_assert(K_TOP == 16 && S_RANK == 64, "Eight-lane selector requires 16 x 64");
            const int candidate = tid >> 3;
            const int part = tid & 7;
            const uint32_t cid = s_top16_id[candidate];
            const uint16_t* w_succ = (const uint16_t*)p->selector_succ_codebook;
            float bias = 0.0f;
            if (w_succ && cid < V_TOTAL) {
                const int r = part * 8;
                const uint4 raw = *reinterpret_cast<const uint4*>(
                    w_succ + (size_t)cid * S_RANK + r);
                #pragma unroll
                for (int z = 0; z < 4; ++z) {
                    const uint32_t pk = z == 0 ? raw.x : (z == 1 ? raw.y : (z == 2 ? raw.z : raw.w));
                    bias += __builtin_bit_cast(float, pk << 16) * s_q[r + 2 * z];
                    bias += __builtin_bit_cast(float, pk & 0xFFFF0000u) * s_q[r + 2 * z + 1];
                }
            }
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1)
                bias += __shfl_down(bias, offset, 8);
            if (part == 0) s_top16_score[candidate] = s_top16_val[candidate] + bias;
        }
        if (fast_selector) __syncthreads();

        if (is_winner && tid < K_TOP) {
            const uint16_t* w_succ = (const uint16_t*)p->selector_succ_codebook;
            uint32_t cid = s_top16_id[tid];
            float bias = 0.0f;
            // Standalone golden callers retain the original FP32 accumulation order.
            if (!fast_selector && w_succ && cid < V_TOTAL) {
                const uint16_t* succ_row = w_succ + (size_t)cid * S_RANK;
                #pragma nounroll
                for (int r = 0; r < S_RANK; r += 8) {
                    uint4 raw = *reinterpret_cast<const uint4*>(succ_row + r);
                    uint32_t r0 = raw.x, r1 = raw.y, r2 = raw.z, r3 = raw.w;
                    #pragma unroll
                    for (int p_i = 0; p_i < 4; ++p_i) {
                        uint32_t pk = (p_i == 0) ? r0 : ((p_i == 1) ? r1 : ((p_i == 2) ? r2 : r3));
                        bias += __builtin_bit_cast(float, pk << 16) * s_q[r + 2 * p_i + 0];
                        bias += __builtin_bit_cast(float, pk & 0xFFFF0000u) * s_q[r + 2 * p_i + 1];
                    }
                }
            }
            float my_score = fast_selector ? s_top16_score[tid] : s_top16_val[tid] + bias;
            uint32_t* out_pids  = (uint32_t*)(s_base + SCR_PROP_ID_OFF + (size_t)global_slot_idx * K_TOP * sizeof(uint32_t));
            float*    out_pvals = (float*)(s_base + SCR_PROP_VAL_OFF + (size_t)global_slot_idx * K_TOP * sizeof(float));
            out_pids[tid]  = cid;
            out_pvals[tid] = my_score;

            float best_score = my_score;
            uint32_t best_id = cid;
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                float ov = __shfl_down(best_score, offset, 32);
                uint32_t oi = __shfl_down(best_id, offset, 32);
                if (is_better(ov, oi, best_score, best_id)) { best_score = ov; best_id = oi; }
            }
            if (tid == 0) {
                uint32_t chosen = best_id;
                __hip_atomic_store((uint32_t*)(s_base + ALODLM_PRED_OFF) + global_slot_idx, chosen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                if (max_depth == 1) {
                    ((uint32_t*)(s_base + ALODLM_IDS_OFF))[global_slot_idx] = chosen;
                }
                __hip_atomic_store((uint32_t*)(s_base + SCR_PREV_TOK_OFF) + global_slot_idx + 1, chosen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                if (is_final_range && global_slot_idx == last_unres) {
                    dev_alodlm_commit_step(p, s_base, tokens_out, s, max_depth, active_k);
                } else if (max_depth == 1 && global_slot_idx == 3 && active_k > 4) {
                    const uint32_t my_want = *(const uint32_t*)(s_base + DLOOP_VERIF_READY_OFF);
                    const uint32_t peer_want = tp2_p2p
                        ? (uint32_t)(((const FullDrafterP2PSlotPacket*)p->my_p2p_comm)[
                            (uint32_t)s * FULLDRAFTER_TOTAL_K + 3u].shard_sum_xe)
                        : my_want;
                    const int agreed_k = min(active_k, (int)min(my_want, peer_want));
                    if (agreed_k <= 4) {
                        dev_alodlm_commit_step(p, s_base, tokens_out, s, max_depth, agreed_k);
                    }
                }
            }
        }
        if (is_winner && tp2_p2p && max_depth == 1 && global_slot_idx == 3 && active_k > 4) {
            __syncthreads();
            if (!dloop_stage2_skipped(s_base)) {
                uint32_t btok = ((const uint32_t*)(s_base + ALODLM_IDS_OFF))[3];
                if (btok >= V_TOTAL) btok = 0u;
                const uint32_t owner_rank = (btok >= V_SHARD) ? 1u : 0u;
                const uint32_t local_tok  = (btok >= V_SHARD) ? (btok - V_SHARD) : btok;
                const uint32_t step_seq   = *p->d_step_seq;
                const uint32_t pkt_idx    = (uint32_t)s * FULLDRAFTER_TOTAL_K + 3u;
                const uint32_t emb_tag    = (step_seq * 8u + (uint32_t)depth) * 1024u + pkt_idx + 513u;
                FullDrafterP2PSlotPacket* my_pkt   = (FullDrafterP2PSlotPacket*)p->my_p2p_comm + pkt_idx;
                FullDrafterP2PSlotPacket* peer_pkt = (FullDrafterP2PSlotPacket*)p->peer_p2p_comm + pkt_idx;
                const uint16_t* emb_shard = (const uint16_t*)p->teacher_embedding_table;
                if (p->rank_id == owner_rank && emb_shard != nullptr) {
                    const uint4* src_u4 = reinterpret_cast<const uint4*>(emb_shard + (size_t)local_tok * W_DIM);
                    uint4* my_dst_u4    = reinterpret_cast<uint4*>(my_pkt->emb_inbox);
                    uint4* peer_dst_u4  = reinterpret_cast<uint4*>(peer_pkt->emb_inbox);
                    for (int idx = tid; idx < W_DIM / 8; idx += (int)blockDim.x) {
                        const uint4 v = src_u4[idx];
                        my_dst_u4[idx]   = v;
                        peer_dst_u4[idx] = v;
                    }
                    __threadfence_system();
                    __syncthreads();
                    if (tid == 0) {
                        __hip_atomic_store(&peer_pkt->emb_flag, emb_tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
                    }
                } else {
                    if (tid == 0) {
                        uint64_t spin = 0;
                        while (__hip_atomic_load(&my_pkt->emb_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != emb_tag) {
                            if (++spin > 200000000ULL) break;
                        }
                    }
                    __syncthreads();
                    __threadfence_system();
                }
            }
        }

        if (is_winner) {
            __threadfence();
            if (tid == 0) {
                __hip_atomic_store((uint32_t*)(s_base + ALODLM_SLOT_EPOCH_OFF), my_epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        } else if (tid == 0) {
            uint32_t* ep_ptr = (uint32_t*)(s_base + ALODLM_SLOT_EPOCH_OFF);
            uint64_t spin = 0;
            while (__hip_atomic_load(ep_ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != my_epoch) {
                if (++spin > 200000000ULL) break;
            }
        }
        __syncthreads();
        } // end global_slot_idx loop
    } // end s loop
}

extern "C" int r4d_fulldrafter_slot_markov_select(
    int stage,
    int slot,
    const FullDrafterGpuPointers* p,
    char* scratch,
    uint32_t* tokens_out,
    uint32_t M,
    hipStream_t stream
) {
    if (!p || !scratch || !tokens_out) return -1;
    dim3 grid(M2_GRID_BLOCKS, M);
    dim3 block(M2_THREADS);
    r4d_fulldrafter_slot_markov_select_kernel<<<grid, block, 0, stream>>>(
        stage, slot, p, scratch, tokens_out
    );
    return 0;
}

// ==============================================================================
// Kernel 4: Stage 2 Backbone (T6 Boundary Prep & Anchor Proj across 96 blocks / 64 CUs)
// Grid: dim3(96, M), Block: dim3(256) -> 96 * 8 = 768 waves across 64 CUs
// ==============================================================================
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fulldrafter_stage2_backbone_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch,
    const uint32_t* __restrict__ tokens_out
) {
    const int s = blockIdx.y;
    const int global_row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int tid = threadIdx.x;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    // Re-initialize arrival counter for Stage 2 slot selection
    if (blockIdx.x == 0 && tid == 0) {
        *(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u;
    }

    // Discrete T6 boundary token from slot 5
    uint32_t boundary_tok = tokens_out[(size_t)s * FULLDRAFTER_TOTAL_K + (BLOCK_K - 1)];
    if (boundary_tok >= V_TOTAL) boundary_tok = 0;

    // Anchor projection on teacher_embedding[T6]: row global_row of [768, 2560]
    const uint16_t* emb_row = ((const uint16_t*)p->teacher_embedding_table) + (size_t)boundary_tok * W_DIM;
    const uint16_t* w_anc   = (const uint16_t*)p->anchor_proj;
    const uint16_t* w_a_row = w_anc + (size_t)global_row * W_DIM;

    float dot_anc = 0.0f;
    for (int col = lane * 8; col < W_DIM; col += 32 * 8) {
        r4d_u32x4 rk = *reinterpret_cast<const r4d_u32x4*>(emb_row + col);
        r4d_u32x4 rw = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_a_row + col));
        #pragma unroll
        for (int p_idx = 0; p_idx < 4; ++p_idx) {
            dot_anc = r4d_dot2_f32_bf16(rk[p_idx], rw[p_idx], dot_anc);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_anc += __shfl_down(dot_anc, offset, 32);
    }

    if (lane == 0) {
        float* s_temp = (float*)(s_base + SCR_TEMP_OFF);
        s_temp[H_DIM + global_row] = round_bf16_val(dot_anc);
    }
}

// Stage 2 Base Commit: Recurrent RMSNorm on h1[5] + Anchor Sum + Mask Vector Broadcast
// Grid: dim3(1, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fulldrafter_stage2_base_commit_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch
) {
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    float* s_base2 = (float*)(s_base + SCR_BASE2_OFF);
    float* s_h1    = (float*)(s_base + SCR_H1_OFF);
    float* s_h2    = (float*)(s_base + SCR_H2_OFF);
    float* s_temp  = (float*)(s_base + SCR_TEMP_OFF);

    // 1. Recurrent RMSNorm on terminal state h1[5] -> s_temp[0..H_DIM-1]
    dev_rmsnorm(s_h1 + (BLOCK_K - 1) * H_DIM, p->recurrent_norm, s_temp, H_DIM);
    __syncthreads();

    // 2. Base2 = recurrent_norm(h1[5]) + anchor_proj(emb[T6])
    for (int h = tid; h < H_DIM; h += blockDim.x) {
        s_base2[h] = round_bf16_val(s_temp[h] + s_temp[H_DIM + h]);
        s_h2[0 * H_DIM + h] = s_base2[h];
    }

    // 3. Mask vector broadcast for tokens 1..6
    const uint16_t* w_mask = (const uint16_t*)p->mask_vector;
    for (int t = 1; t <= BLOCK_K; ++t) {
        for (int h = tid; h < H_DIM; h += blockDim.x) {
            s_h2[t * H_DIM + h] = bf16_to_f32(w_mask[h]);
        }
    }
}

// Host Dispatch: Stage 2 Multi-CU Transformer Backbone
int r4d_fulldrafter_stage2_backbone(
    const FullDrafterGpuPointers* p,
    char* scratch,
    const uint32_t* tokens_out,
    uint32_t M,
    hipStream_t stream
) {
    if (!p || !scratch || !tokens_out) return -1;
    dim3 block_256(256);

    // 1. Stage 2 Anchor Proj & Arrival Ctr Init across 96 blocks / 64 CUs
    dim3 grid_prep(96, M);
    fulldrafter_stage2_backbone_kernel<<<grid_prep, block_256, 0, stream>>>(
        p, scratch, tokens_out
    );

    // 2. Base2 commit: recurrent RMSNorm + anchor sum + mask broadcast
    dim3 grid_commit(1, M);
    fulldrafter_stage2_base_commit_kernel<<<grid_commit, block_256, 0, stream>>>(p, scratch);

    float* s_h2   = (float*)(scratch + SCR_H2_OFF);
    float* s_temp = (float*)(scratch + SCR_TEMP_OFF);

    // 3. Two Transformer Blocks for Stage 2 on s_h2 (batched across all M sequences, fused conv + pre-RMSNorm)
    for (int b = 0; b < 2; ++b) {
        const void* conv_base   = p->conv_w_base;
        const void* conv_proj   = p->conv_w_proj;
        const void* in_norm_w   = (b == 0) ? p->b0_input_layernorm : p->b1_input_layernorm;
        const void* q_w         = (b == 0) ? p->b0_q_proj : p->b1_q_proj;
        const void* k_w         = (b == 0) ? p->b0_k_proj : p->b1_k_proj;
        const void* v_w         = (b == 0) ? p->b0_v_proj : p->b1_v_proj;
        const void* o_w         = (b == 0) ? p->b0_o_proj : p->b1_o_proj;
        const void* post_norm_w = (b == 0) ? p->b0_post_attn_layernorm : p->b1_post_attn_layernorm;
        const void* gate_w      = (b == 0) ? p->b0_gate_proj : p->b1_gate_proj;
        const void* up_w        = (b == 0) ? p->b0_up_proj : p->b1_up_proj;
        const void* down_w      = (b == 0) ? p->b0_down_proj : p->b1_down_proj;

        r4d_fulldrafter_block_dflash_conv_kernel<<<dim3(1, M), block_256, 0, stream>>>(
            s_h2, s_temp, (const uint16_t*)conv_base, (const uint16_t*)conv_proj, (int)M, in_norm_w
        );
        r4d_fulldrafter_block_qkv_kernel<<<dim3(160, M), block_256, 0, stream>>>(
            s_temp, (const uint16_t*)q_w, (const uint16_t*)k_w, (const uint16_t*)v_w,
            s_temp + 7 * H_DIM, (int)M
        );
        r4d_fulldrafter_block_attention_kernel<true><<<dim3(12, M), dim3(32), 0, stream>>>(
            s_temp + 7 * H_DIM, s_temp, (int)M
        );
        r4d_fulldrafter_block_out_proj_kernel<<<dim3(96, M), block_256, 0, stream>>>(
            s_temp, (const uint16_t*)o_w, s_h2, (int)M, post_norm_w, s_temp
        );
        r4d_fulldrafter_block_mlp_gate_up_kernel<<<dim3(288, M), block_256, 0, stream>>>(
            s_temp, (const uint16_t*)gate_w, (const uint16_t*)up_w, s_temp + 7 * H_DIM, (int)M
        );
        r4d_fulldrafter_block_mlp_down_kernel<<<dim3(96, M), block_256, 0, stream>>>(
            s_temp + 7 * H_DIM, (const uint16_t*)down_w, s_h2, (int)M
        );
    }

    // 4. Final RMSNorm for proposal slots 1..6 -> slots 0..5 in-place in s_h2
    r4d_fulldrafter_final_norm_kernel<<<dim3(1, M), block_256, 0, stream>>>(
        s_h2, p->final_norm, nullptr, (int)M
    );
    return 0;
}


// ==============================================================================
// ALoDLM-4 (model_type == "alodlm_loop", L=13 = 1 + 12, max_depth=4) GPU Kernels
// ==============================================================================

// ALoDLM-4 Depth > 0 Step Prep across 96 Workgroups / 64 CUs with Cross-Depth Anchor-Proj Caching:
// - Blocks 0..11 handle recurrent_norm(latent[j]) for unresolved slots (exits[j] == 0) and cached copies for earlier-resolved slots (exits[j] < depth).
// - All 96 blocks cooperatively compute anchor_proj(teacher_embed(ids[j])) in 1 pass over anchor_proj ONLY for newly-committed slots (exits[j] == depth) and cache the result in ALODLM_EMB_CACHE_OFF!
// Grid: dim3(96, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fulldrafter_alodlm_step_prep_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch,
    int depth
) {
    (void)depth;
    const int bx = blockIdx.x; // 0..active_k-1
    const int s  = blockIdx.y; // 0..M-1
    const int tid = threadIdx.x;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;

    if (dloop_stage2_skipped(s_base)) return;

    float* s_h         = (float*)(s_base + ALODLM_H_OFF);
    const float* s_b1  = (const float*)(s_base + SCR_BASE1_OFF);
    const float* s_lat = (const float*)(s_base + ALODLM_LATENT_OFF);
    float* s_slots     = (float*)(s_base + ALODLM_EMB_CACHE_OFF);
    const float* surv  = (const float*)(s_base + ALODLM_SURV_OFF);

    if (bx == 0) {
        if (tid == 0) *(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u;
        for (int h = tid; h < H_DIM; h += 256) {
            s_h[0 * H_DIM + h] = s_b1[h];
        }
    }

    if (bx < FULLDRAFTER_TOTAL_K) {
        const int j = bx;
        float* dst_slot    = s_h + (1 + j) * H_DIM;
        float* slot_state  = s_slots + j * H_DIM;
        const float* lat_j = s_lat + j * H_DIM;
        const float stay   = surv[j];
        float sq = 0.0f;
        for (int h = tid; h < H_DIM; h += 256) {
            float v = lat_j[h];
            sq += v * v;
        }
        float sum_sq = block_reduce_sum(sq);
        float scale  = rsqrtf(sum_sq / static_cast<float>(H_DIM) + FULLDRAFTER_RMS_NORM_EPS);
        const uint16_t* rn_w = (const uint16_t*)p->recurrent_norm;
        for (int h = tid; h < H_DIM; h += 256) {
            float rn = round_bf16_val(lat_j[h] * scale * bf16_to_f32(rn_w[h]));
            float ns = round_bf16_val(slot_state[h] + round_bf16_val(stay * rn));
            slot_state[h] = ns;
            dst_slot[h]   = ns;
        }
    }
}


// Stage 2 boundary is global slot 3; backbone rows remain local to this stage.
// Grid: dim3(96, M), Block: dim3(256), stage2_k <= 4.
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fulldrafter_alodlm_stage2_prep_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch,
    int stage2_k
) {
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    float* s_h = (float*)(s_base + ALODLM_H_OFF);
    __shared__ float norm[H_DIM];
    const float* h1 = (const float*)(s_base + SCR_H1_OFF);
    dev_rmsnorm(h1 + 3 * H_DIM, p->recurrent_norm, norm, H_DIM);

    uint32_t boundary_tok = ((const uint32_t*)(s_base + ALODLM_IDS_OFF))[3];
    if (boundary_tok >= V_TOTAL) boundary_tok = 0;
    const uint16_t* emb_row = (p->tp2_p2p_enabled != 0u && p->my_p2p_comm != nullptr)
        ? ((const FullDrafterP2PSlotPacket*)p->my_p2p_comm)[(uint32_t)s * FULLDRAFTER_TOTAL_K + 3u].emb_inbox
        : ((const uint16_t*)p->teacher_embedding_table + (size_t)boundary_tok * W_DIM);
    const uint16_t* w_anc = (const uint16_t*)p->anchor_proj;
    const int row = blockIdx.x * 8 + wave;
    if (row < H_DIM) {
        const uint16_t* w_row = w_anc + (size_t)row * W_DIM;
        float dot = 0.0f;
        for (int col = lane * 8; col < W_DIM; col += 32 * 8) {
            r4d_u32x4 rk = *reinterpret_cast<const r4d_u32x4*>(emb_row + col);
            r4d_u32x4 rw = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_row + col));
            #pragma unroll
            for (int p_idx = 0; p_idx < 4; ++p_idx) {
                dot = r4d_dot2_f32_bf16(rk[p_idx], rw[p_idx], dot);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down(dot, offset, 32);
        }
        if (lane == 0) s_h[row] = round_bf16_val(norm[row] + round_bf16_val(dot));
    }
    if (blockIdx.x != 0) return;

    const uint16_t* mask = (const uint16_t*)p->mask_vector;
    for (int t = 1; t <= stage2_k; ++t) {
        for (int h = tid; h < H_DIM; h += 256) {
            s_h[t * H_DIM + h] = bf16_to_f32(mask[h]);
        }
    }
    if (tid < FULLDRAFTER_TOTAL_K) {
        ((uint32_t*)(s_base + ALODLM_EXITS_OFF))[tid] =
            (tid >= 4 && tid < 4 + stage2_k) ? 0u : 1u;
    }
    if (tid == 0) *(uint32_t*)(s_base + SCR_ARRIVAL_CTR_OFF) = 0u;
}

// ALoDLM-4 L=13 DFlash2 Conv + Fused Pre-RMSNorm (uses only 5 KB LDS -> 16 Waves/SIMD = 100% Occupancy)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_dflash_conv_kernel(
    float*          __restrict__ s_h,         // [13, 768] in scratch (ALODLM_H_OFF)
    float*          __restrict__ out_norm,    // [13, 768] in scratch (ALODLM_NORM_OFF)
    const uint16_t* __restrict__ conv_base,   // [2, 768] bf16
    const uint16_t* __restrict__ conv_proj,   // [96, 768] bf16
    const void*     __restrict__ in_norm_w,   // [768] bf16
    int             active_L = 7,
    const FullDrafterGpuPointers* __restrict__ p_init = nullptr,
    uint32_t*       __restrict__ tokens_out_init      = nullptr
) {
    const int s = blockIdx.y;
    const int tid = threadIdx.x;
    char* s_base = (char*)s_h - ALODLM_H_OFF + (size_t)s * SCR_PER_SEQ;
    float* h_base = (float*)((char*)s_h + (size_t)s * SCR_PER_SEQ);
    float* n_base = (float*)((char*)out_norm + (size_t)s * SCR_PER_SEQ);

    if (p_init != nullptr) {
        float* s_base1 = (float*)(s_base + SCR_BASE1_OFF);
        float* s_temp  = (float*)(s_base + SCR_TEMP_OFF);
        float* s_slots = (float*)(s_base + ALODLM_EMB_CACHE_OFF);
        dev_rmsnorm(s_temp, p_init->context_norm, s_temp, H_DIM);
        __syncthreads();
        const uint16_t* w_mask = (const uint16_t*)p_init->mask_vector;
        for (int h = tid; h < H_DIM; h += 256) {
            float bval = round_bf16_val(s_temp[h] + s_temp[2 * H_DIM + h]);
            s_base1[h] = bval;
            h_base[0 * H_DIM + h] = bval;
            float mval = bf16_to_f32(w_mask[h]);
            #pragma unroll
            for (int t = 1; t < ALODLM_L; ++t) {
                h_base[t * H_DIM + h] = mval;
                s_slots[(t - 1) * H_DIM + h] = mval;
            }
        }
        const int init_active_k = active_L - 1;
        if (tid < FULLDRAFTER_TOTAL_K) {
            uint32_t anc_tok = *(const uint32_t*)(s_base + SCR_PREV_TOK_OFF);
            ((uint32_t*)(s_base + ALODLM_IDS_OFF))[tid]        = anc_tok;
            ((uint32_t*)(s_base + ALODLM_PRED_OFF))[tid]       = anc_tok;
            ((uint32_t*)(s_base + ALODLM_EXITS_OFF))[tid]      = (tid >= init_active_k) ? 1u : 0u;
            ((float*)(s_base + ALODLM_SURV_OFF))[tid]          = 1.0f;
            ((float*)(s_base + ALODLM_ENTROPY_OFF))[tid]       = 100.0f;
            ((uint32_t*)(s_base + ALODLM_MCACHE_TOK_OFF))[tid] = 0xFFFFFFFFu;
            if (tokens_out_init) {
                tokens_out_init[(size_t)s * FULLDRAFTER_TOTAL_K + tid] =
                    (tid >= init_active_k) ? r4d::FULLDRAFTER_SENTINEL_ID : anc_tok;
            }
        }
        if (tid == 0) {
            *(uint32_t*)(s_base + ALODLM_DEPTHS_OFF)          = 0u;
            *(uint32_t*)(s_base + ALODLM_SLOT_EPOCH_OFF)      = 0u;
            *(uint32_t*)(s_base + DLOOP_STAGE2_ACTIVE_OFF)    = 1u;
            *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF)      = 1u;
            __hip_atomic_store((uint32_t*)(s_base + ALODLM_CONFIDENCE_ARRIVAL_OFF), 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
    } else if (dloop_stage2_skipped(s_base)) {
        return;
    }

    __shared__ float s_delta[ALODLM_L * 96];
    __shared__ float s_h_all[5 * H_DIM]; // 15 KB LDS (holds full [5, 768] in 1 barrier when active_L <= 5)

    const int wave = tid >> 5;
    const int lane = tid & 31;

    if (active_L <= 5) {
        for (int i = tid; i < active_L * H_DIM; i += 256) {
            s_h_all[i] = h_base[i];
        }
        __syncthreads();

        #pragma unroll 2
        for (int g = 0; g < 12; ++g) {
            const int row = g * 8 + wave;
            const uint16_t* w_row = conv_proj + (size_t)row * H_DIM;
            const int col0 = 0 * 256 + lane * 8;
            const int col1 = 1 * 256 + lane * 8;
            const int col2 = 2 * 256 + lane * 8;
            const r4d_u32x4 r_w0 = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_row + col0));
            const r4d_u32x4 r_w1 = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_row + col1));
            const r4d_u32x4 r_w2 = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_row + col2));
            float dot[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const r4d_u32x4 r_ws[3] = {r_w0, r_w1, r_w2};
            #pragma unroll
            for (int chunk = 0; chunk < 3; ++chunk) {
                const int col = chunk * 256 + lane * 8;
                const r4d_u32x4 r_w = r_ws[chunk];
                uint32_t r0 = r_w[0], r1 = r_w[1], r2 = r_w[2], r3 = r_w[3];
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    uint32_t w_pk = (p == 0) ? r0 : ((p == 1) ? r1 : ((p == 2) ? r2 : r3));
                    float w0 = __builtin_bit_cast(float, w_pk << 16);
                    float w1 = __builtin_bit_cast(float, w_pk & 0xFFFF0000u);
                    int idx = col + 2 * p;
                    #pragma unroll
                    for (int t = 0; t < 5; ++t) {
                        if (t < active_L) {
                            dot[t] += s_h_all[t * H_DIM + idx + 0] * w0;
                            dot[t] += s_h_all[t * H_DIM + idx + 1] * w1;
                        }
                    }
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int t = 0; t < 5; ++t) {
                    dot[t] += __shfl_down(dot[t], offset, 32);
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int t = 0; t < 5; ++t) {
                    if (t < active_L) s_delta[t * 96 + row] = round_bf16_val(dot[t]);
                }
            }
        }
    } else if (tid < 96) {
        const uint16_t* w_row = conv_proj + (size_t)tid * H_DIM;
        float dot[ALODLM_L];
        #pragma unroll
        for (int t = 0; t < ALODLM_L; ++t) dot[t] = 0.0f;
        #pragma nounroll
        for (int i = 0; i < H_DIM; i += 8) {
            uint4 r_w = *reinterpret_cast<const uint4*>(w_row + i);
            const uint16_t* hw = reinterpret_cast<const uint16_t*>(&r_w);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float wv = bf16_to_f32(hw[j]);
                int col = i + j;
                #pragma unroll
                for (int t = 0; t < 5; ++t) {
                    dot[t] += h_base[t * H_DIM + col] * wv;
                }
                if (active_L > 5) {
                    #pragma unroll
                    for (int t = 5; t < 7; ++t) {
                        dot[t] += h_base[t * H_DIM + col] * wv;
                    }
                }
                if (active_L > 7) {
                    #pragma unroll
                    for (int t = 7; t < ALODLM_L; ++t) {
                        dot[t] += h_base[t * H_DIM + col] * wv;
                    }
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < ALODLM_L; ++t) {
            s_delta[t * 96 + tid] = round_bf16_val(dot[t]);
        }
    }
    __syncthreads();

    for (int c = tid; c < H_DIM; c += 256) {
        int g = c / FULLDRAFTER_CONV_GROUP_SIZE;
        float cb0 = bf16_to_f32(conv_base[0 * H_DIM + c]);
        float cb1 = bf16_to_f32(conv_base[1 * H_DIM + c]);
        float p_prev = 0.0f;
        for (int t = 0; t < active_L; ++t) {
            float p_curr = (active_L <= 5) ? s_h_all[t * H_DIM + c] : h_base[t * H_DIM + c];
            float w0 = round_bf16_val(cb0 + s_delta[t * 96 + 0 * 48 + g]);
            float w1 = round_bf16_val(cb1 + s_delta[t * 96 + 1 * 48 + g]);
            float tap0 = round_bf16_val(w0 * p_curr);
            float tap1 = (t >= 1) ? round_bf16_val(w1 * p_prev) : 0.0f;
            float val = round_bf16_val(tap0 + tap1);
            h_base[t * H_DIM + c] = val;
            p_prev = p_curr;
        }
    }
    __syncthreads();

    if (in_norm_w) {
        for (int t = 0; t < active_L; ++t) {
            dev_rmsnorm(h_base + t * H_DIM, in_norm_w, n_base + t * H_DIM, H_DIM);
        }
    }
}

// ALoDLM-4 L=13 Token-Batched QKV Multi-GEMV (Grid: dim3(160, M), Block: dim3(256))
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_qkv_kernel(
    const float*    __restrict__ in_norm_x, // [13, 768]
    const uint16_t* __restrict__ w_q,
    const uint16_t* __restrict__ w_k,
    const uint16_t* __restrict__ w_v,
    float*          __restrict__ out_qkv,   // [13, 1280]
    int             active_L = 7
) {
    const int s = blockIdx.y;
    const char* s_base = (const char*)in_norm_x - ALODLM_NORM_OFF + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    const int global_row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const uint16_t* w_row = (global_row < 768) ? (w_q + (size_t)global_row * 768)
                          : (global_row < 1024) ? (w_k + (size_t)(global_row - 768) * 768)
                                                : (w_v + (size_t)(global_row - 1024) * 768);
    __shared__ float s_x[ALODLM_L * 256]; // 13,312 B FP32 LDS (<= 16 KiB -> 16 Waves/SIMD true R9700 occupancy!)
    const float* x_base = (const float*)((const char*)in_norm_x + (size_t)s * SCR_PER_SEQ);

    float acc[ALODLM_L];
    #pragma unroll
    for (int t = 0; t < ALODLM_L; ++t) acc[t] = 0.0f;

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        if (it > 0) __syncthreads();
        const int col_off = it * 256 + threadIdx.x;
        for (int t = 0; t < active_L; ++t) {
            s_x[t * 256 + threadIdx.x] = x_base[t * 768 + col_off];
        }
        __syncthreads();
        if (global_row >= 1280) continue;

        int col = (it * 32 + lane) * 8;
        int lds_col = lane * 8;
        r4d_u32x4 raw = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(w_row + col));
        uint32_t r0 = raw[0], r1 = raw[1], r2 = raw[2], r3 = raw[3];
        #pragma nounroll
        for (int p = 0; p < 4; ++p) {
            uint32_t w_pk = r0;
            r0 = r1; r1 = r2; r2 = r3;
            float w0 = __builtin_bit_cast(float, w_pk << 16);
            float w1 = __builtin_bit_cast(float, w_pk & 0xFFFF0000u);
            int idx = lds_col + 2 * p;
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += w0 * s_x[t * 256 + idx + 0];
                acc[t] += w1 * s_x[t * 256 + idx + 1];
            }
            if (active_L > 5) {
                #pragma unroll
                for (int t = 5; t < 7; ++t) {
                    acc[t] += w0 * s_x[t * 256 + idx + 0];
                    acc[t] += w1 * s_x[t * 256 + idx + 1];
                }
            }
            if (active_L > 7) {
                #pragma unroll
                for (int t = 7; t < ALODLM_L; ++t) {
                    acc[t] += w0 * s_x[t * 256 + idx + 0];
                    acc[t] += w1 * s_x[t * 256 + idx + 1];
                }
            }
        }
    }
    if (global_row >= 1280) return;
    if (active_L <= 5) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* qkv_s = (float*)((char*)out_qkv + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                qkv_s[t * 1280 + global_row] = round_bf16_val(acc[t]);
            }
        }
    } else {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* qkv_s = (float*)((char*)out_qkv + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                qkv_s[t * 1280 + global_row] = round_bf16_val(acc[t]);
            }
        }
    }
}

// ALoDLM-4 L=13 Causal GQA Attention with Shared KV-Head LDS Staging (Grid: dim3(4, M), Block: dim3(96))
// 1 WG per KV-head (kv_h = 0..3), 3 waves (q_sub = 0..2) share s_k_pk/s_v0/s_v1 in 5.1 KB LDS -> 16/16 Waves/SIMD!
__global__ __launch_bounds__(96) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_attention_kernel(
    const float* __restrict__ in_qkv,   // [13, 1280]
    float*       __restrict__ out_attn, // [13, 768]
    int          active_L = 7
) {
    const int s = blockIdx.y;
    const char* s_base = (const char*)out_attn - ALODLM_NORM_OFF + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    const int kv_h  = blockIdx.x; // 0..3
    const int q_sub = threadIdx.x >> 5; // 0..2
    const int lane  = threadIdx.x & 31;
    const int q_h   = kv_h * 3 + q_sub;
    if (kv_h >= 4 || q_sub >= 3) return;

    const float* qkv_s = (const float*)((const char*)in_qkv + (size_t)s * SCR_PER_SEQ);
    float* attn_s      = (float*)((char*)out_attn + (size_t)s * SCR_PER_SEQ);

    __shared__ uint32_t s_k_pk[ALODLM_L][32];
    __shared__ float    s_v0[ALODLM_L][32];
    __shared__ float    s_v1[ALODLM_L][32];
    __shared__ float    s_scores[3][ALODLM_L];

    const float freq = 1.0f / powf(FULLDRAFTER_ROPE_THETA, (2.0f * (float)lane) / 64.0f);

    // All 3 waves cooperatively load and RoPE-rotate K & V once per KV head (3x less KV work!)
    for (int t = q_sub; t < active_L; t += 3) {
        const float* k_tok = qkv_s + t * 1280 + 768 + kv_h * 64;
        const float* v_tok = qkv_s + t * 1280 + 1024 + kv_h * 64;
        float k0 = k_tok[lane], k1 = k_tok[lane + 32];
        s_v0[t][lane] = v_tok[lane];
        s_v1[t][lane] = v_tok[lane + 32];
        float theta = (float)t * freq;
        float cos_th = cosf(theta), sin_th = sinf(theta);
        float kr0 = round_bf16_val(k0 * cos_th - k1 * sin_th);
        float kr1 = round_bf16_val(k0 * sin_th + k1 * cos_th);
        s_k_pk[t][lane] = pack_f32_to_bf16x2(kr0, kr1);
    }
    __syncthreads();

    for (int t = 0; t < active_L; ++t) {
        const float* q_tok = qkv_s + t * 1280 + q_h * 64;
        float q0 = q_tok[lane], q1 = q_tok[lane + 32];
        float theta = (float)t * freq;
        float cos_th = cosf(theta), sin_th = sinf(theta);
        float qr0 = round_bf16_val(q0 * cos_th - q1 * sin_th);
        float qr1 = round_bf16_val(q0 * sin_th + q1 * cos_th);
        uint32_t q_pk = pack_f32_to_bf16x2(qr0, qr1);

        float max_score = -1e9f;
        for (int p_idx = 0; p_idx <= t; ++p_idx) {
            float dot = r4d_dot2_f32_bf16(q_pk, s_k_pk[p_idx][lane], 0.0f);
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                dot += __shfl_xor(dot, mask, 32);
            }
            dot = dot * 0.125f;
            if (lane == 0) s_scores[q_sub][p_idx] = dot;
            if (dot > max_score) max_score = dot;
        }
        float exp_sum = 0.0f;
        for (int p_idx = 0; p_idx <= t; ++p_idx) {
            float ex = expf(s_scores[q_sub][p_idx] - max_score);
            if (lane == 0) s_scores[q_sub][p_idx] = ex;
            exp_sum += ex;
        }
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int p_idx = 0; p_idx <= t; ++p_idx) {
            float prob = round_bf16_val(s_scores[q_sub][p_idx] / exp_sum);
            acc0 += prob * s_v0[p_idx][lane];
            acc1 += prob * s_v1[p_idx][lane];
        }
        attn_s[t * 768 + q_h * 64 + lane]      = round_bf16_val(acc0);
        attn_s[t * 768 + q_h * 64 + lane + 32] = round_bf16_val(acc1);
    }
}

// ALoDLM-4 L=13 Out-Proj + Residual Add with Cooperative LDS BF16x2 Staging & v_dot2_f32_bf16
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_out_proj_kernel(
    const float*    __restrict__ in_attn, // [13, 768] (already BF16-rounded by attention kernel)
    const uint16_t* __restrict__ w_o,     // [768, 768]
    float*          __restrict__ out_h,   // [13, 768]
    const void*     __restrict__ post_norm_w = nullptr,
    float*          __restrict__ out_norm_y  = nullptr,
    int             active_L = 7
) {
    const int s = blockIdx.y;
    const char* s_base = (const char*)out_h - ALODLM_H_OFF + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    __shared__ uint32_t s_a_pk[ALODLM_L * 128]; // 6,656 B LDS (<= 16 KiB -> 16 Waves/SIMD true R9700 occupancy!)
    const float* a_base = (const float*)((const char*)in_attn + (size_t)s * SCR_PER_SEQ);
    const int global_row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    const r4d_u32x4* w_row = (global_row < 768) ? reinterpret_cast<const r4d_u32x4*>(w_o + (size_t)global_row * 768) : nullptr;
    float acc[ALODLM_L];
    #pragma unroll
    for (int t = 0; t < ALODLM_L; ++t) acc[t] = 0.0f;

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        if (it > 0) __syncthreads();
        if (threadIdx.x < 128) {
            int col_pair = it * 256 + 2 * threadIdx.x;
            for (int t = 0; t < active_L; ++t) {
                uint32_t u0 = __builtin_bit_cast(uint32_t, a_base[t * 768 + col_pair + 0]);
                uint32_t u1 = __builtin_bit_cast(uint32_t, a_base[t * 768 + col_pair + 1]);
                s_a_pk[t * 128 + threadIdx.x] = (u0 >> 16) | (u1 & 0xFFFF0000u);
            }
        }
        __syncthreads();
        if (!w_row) continue;

        int vec_idx = it * 32 + lane;
        r4d_u32x4 raw = __builtin_nontemporal_load(&w_row[vec_idx]);
        int p_base = lane * 4;
        uint32_t r0 = raw[0], r1 = raw[1], r2 = raw[2], r3 = raw[3];
        #pragma nounroll
        for (int p = 0; p < 4; ++p) {
            uint32_t w_pk = r0;
            r0 = r1; r1 = r2; r2 = r3;
            float w0 = __builtin_bit_cast(float, w_pk << 16);
            float w1 = __builtin_bit_cast(float, w_pk & 0xFFFF0000u);
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                uint32_t a_pk = s_a_pk[t * 128 + p_base + p];
                acc[t] += w0 * __builtin_bit_cast(float, a_pk << 16);
                acc[t] += w1 * __builtin_bit_cast(float, a_pk & 0xFFFF0000u);
            }
            if (active_L > 5) {
                #pragma unroll
                for (int t = 5; t < 7; ++t) {
                    uint32_t a_pk = s_a_pk[t * 128 + p_base + p];
                    acc[t] += w0 * __builtin_bit_cast(float, a_pk << 16);
                    acc[t] += w1 * __builtin_bit_cast(float, a_pk & 0xFFFF0000u);
                }
            }
            if (active_L > 7) {
                #pragma unroll
                for (int t = 7; t < ALODLM_L; ++t) {
                    uint32_t a_pk = s_a_pk[t * 128 + p_base + p];
                    acc[t] += w0 * __builtin_bit_cast(float, a_pk << 16);
                    acc[t] += w1 * __builtin_bit_cast(float, a_pk & 0xFFFF0000u);
                }
            }
        }
    }
    if (global_row >= 768) return;
    float* h_s = (float*)((char*)out_h + (size_t)s * SCR_PER_SEQ);
    if (active_L <= 5) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                h_s[t * 768 + global_row] = round_bf16_val(h_s[t * 768 + global_row] + acc[t]);
            }
        }
    } else {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                h_s[t * 768 + global_row] = round_bf16_val(h_s[t * 768 + global_row] + acc[t]);
            }
        }
    }

    if (post_norm_w != nullptr && out_norm_y != nullptr) {
        __threadfence();
        __syncthreads();
        __shared__ uint32_t s_is_last;
        if (threadIdx.x == 0) {
            unsigned int* ctr = (unsigned int*)((char*)s_base + SCR_ARRIVAL_CTR_OFF);
            unsigned int arrived = __hip_atomic_fetch_add(ctr, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (arrived == (unsigned int)(gridDim.x - 1)) {
                *ctr = 0u;
                s_is_last = 1u;
            } else {
                s_is_last = 0u;
            }
        }
        __syncthreads();
        if (s_is_last != 0u) {
                    float* norm_s = (float*)((char*)out_norm_y + (size_t)s * SCR_PER_SEQ);
            for (int t = 0; t < active_L; ++t) {
                dev_rmsnorm(h_s + t * H_DIM, post_norm_w, norm_s + t * H_DIM, H_DIM);
            }
        }
    }
}

// ALoDLM-4 L=13 SwiGLU MLP Gate+Up (Grid: dim3(288, M), Block: dim3(256), 0 Scratch)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_mlp_gate_up_kernel(
    const float*    __restrict__ in_norm_y, // [13, 768]
    const uint16_t* __restrict__ w_gate,
    const uint16_t* __restrict__ w_up,
    float*          __restrict__ out_act,   // [13, 2304]
    int             active_L = 7
) {
    const int s = blockIdx.y;
    const char* s_base = (const char*)in_norm_y - ALODLM_NORM_OFF + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    const int wave = threadIdx.x >> 5;
    const int global_m = blockIdx.x * 8 + wave;
    const int lane = threadIdx.x & 31;

    __shared__ float s_y[ALODLM_L * 256]; // 13,312 B FP32 LDS (<= 16 KiB -> 16 Waves/SIMD true R9700 occupancy!)
    __shared__ float s_silu[8][ALODLM_L];
    const float* y_base = (const float*)((const char*)in_norm_y + (size_t)s * SCR_PER_SEQ);
    const uint16_t* g_row = (global_m < 2304) ? (w_gate + (size_t)global_m * 768) : nullptr;
    const uint16_t* u_row = (global_m < 2304) ? (w_up   + (size_t)global_m * 768) : nullptr;

    if (active_L <= 5) {
        float acc_g[5] = {0}, acc_u[5] = {0};
        #pragma nounroll
        for (int it = 0; it < 3; ++it) {
            if (it > 0) __syncthreads();
            const int col_off = it * 256 + threadIdx.x;
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                if (t < active_L) {
                    s_y[t * 256 + threadIdx.x] = y_base[t * 768 + col_off];
                }
            }
            __syncthreads();
            if (!g_row) continue;

            int col = (it * 32 + lane) * 8;
            int lds_col = lane * 8;
            r4d_u32x4 raw_g = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(g_row + col));
            r4d_u32x4 raw_u = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(u_row + col));
            uint32_t g0_pk = raw_g[0], g1_pk = raw_g[1], g2_pk = raw_g[2], g3_pk = raw_g[3];
            uint32_t u0_pk = raw_u[0], u1_pk = raw_u[1], u2_pk = raw_u[2], u3_pk = raw_u[3];
            #pragma nounroll
            for (int p = 0; p < 4; ++p) {
                uint32_t g_pk = g0_pk, u_pk = u0_pk;
                g0_pk = g1_pk; g1_pk = g2_pk; g2_pk = g3_pk;
                u0_pk = u1_pk; u1_pk = u2_pk; u2_pk = u3_pk;
                float g0 = __builtin_bit_cast(float, g_pk << 16);
                float g1 = __builtin_bit_cast(float, g_pk & 0xFFFF0000u);
                float u0 = __builtin_bit_cast(float, u_pk << 16);
                float u1 = __builtin_bit_cast(float, u_pk & 0xFFFF0000u);
                int idx = lds_col + 2 * p;
                #pragma unroll
                for (int t = 0; t < 5; ++t) {
                    if (t < active_L) {
                        float y0 = s_y[t * 256 + idx + 0];
                        float y1 = s_y[t * 256 + idx + 1];
                        acc_g[t] += g0 * y0;
                        acc_g[t] += g1 * y1;
                        acc_u[t] += u0 * y0;
                        acc_u[t] += u1 * y1;
                    }
                }
            }
        }
        if (global_m >= 2304) return;
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc_g[t] += __shfl_down(acc_g[t], offset, 32);
                acc_u[t] += __shfl_down(acc_u[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* act_s = (float*)((char*)out_act + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                if (t < active_L) {
                    float g = acc_g[t];
                    act_s[t * 2304 + global_m] = round_bf16_val((g / (1.0f + expf(-g))) * acc_u[t]);
                }
            }
        }
        return;
    }

    float acc[ALODLM_L];
    #pragma unroll
    for (int t = 0; t < ALODLM_L; ++t) acc[t] = 0.0f;

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        if (it > 0) __syncthreads();
        const int col_off = it * 256 + threadIdx.x;
        for (int t = 0; t < active_L; ++t) {
            s_y[t * 256 + threadIdx.x] = y_base[t * 768 + col_off];
        }
        __syncthreads();
        if (!g_row) continue;

        int col = (it * 32 + lane) * 8;
        int lds_col = lane * 8;
        r4d_u32x4 raw_g = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(g_row + col));
        uint32_t r0 = raw_g[0], r1 = raw_g[1], r2 = raw_g[2], r3 = raw_g[3];
        #pragma nounroll
        for (int p = 0; p < 4; ++p) {
            uint32_t g_pk = r0;
            r0 = r1; r1 = r2; r2 = r3;
            float g0 = __builtin_bit_cast(float, g_pk << 16);
            float g1 = __builtin_bit_cast(float, g_pk & 0xFFFF0000u);
            int idx = lds_col + 2 * p;
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += g0 * s_y[t * 256 + idx + 0];
                acc[t] += g1 * s_y[t * 256 + idx + 1];
            }
            if (active_L > 5) {
                #pragma unroll
                for (int t = 5; t < 7; ++t) {
                    acc[t] += g0 * s_y[t * 256 + idx + 0];
                    acc[t] += g1 * s_y[t * 256 + idx + 1];
                }
            }
            if (active_L > 7) {
                #pragma unroll
                for (int t = 7; t < ALODLM_L; ++t) {
                    acc[t] += g0 * s_y[t * 256 + idx + 0];
                    acc[t] += g1 * s_y[t * 256 + idx + 1];
                }
            }
        }
    }
    if (g_row) {
        if (active_L <= 5) {
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int t = 0; t < 5; ++t) {
                    acc[t] += __shfl_down(acc[t], offset, 32);
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int t = 0; t < 5; ++t) {
                    float g = acc[t];
                    s_silu[wave][t] = g / (1.0f + expf(-g));
                }
            }
        } else {
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int t = 0; t < ALODLM_L; ++t) {
                    acc[t] += __shfl_down(acc[t], offset, 32);
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int t = 0; t < ALODLM_L; ++t) {
                    float g = acc[t];
                    s_silu[wave][t] = g / (1.0f + expf(-g));
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < ALODLM_L; ++t) acc[t] = 0.0f;

    #pragma nounroll
    for (int it = 0; it < 3; ++it) {
        __syncthreads();
        const int col_off = it * 256 + threadIdx.x;
        for (int t = 0; t < active_L; ++t) {
            s_y[t * 256 + threadIdx.x] = y_base[t * 768 + col_off];
        }
        __syncthreads();
        if (!u_row) continue;

        int col = (it * 32 + lane) * 8;
        int lds_col = lane * 8;
        r4d_u32x4 raw_u = __builtin_nontemporal_load(reinterpret_cast<const r4d_u32x4*>(u_row + col));
        uint32_t r0 = raw_u[0], r1 = raw_u[1], r2 = raw_u[2], r3 = raw_u[3];
        #pragma nounroll
        for (int p = 0; p < 4; ++p) {
            uint32_t u_pk = r0;
            r0 = r1; r1 = r2; r2 = r3;
            float u0 = __builtin_bit_cast(float, u_pk << 16);
            float u1 = __builtin_bit_cast(float, u_pk & 0xFFFF0000u);
            int idx = lds_col + 2 * p;
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += u0 * s_y[t * 256 + idx + 0];
                acc[t] += u1 * s_y[t * 256 + idx + 1];
            }
            if (active_L > 5) {
                #pragma unroll
                for (int t = 5; t < 7; ++t) {
                    acc[t] += u0 * s_y[t * 256 + idx + 0];
                    acc[t] += u1 * s_y[t * 256 + idx + 1];
                }
            }
            if (active_L > 7) {
                #pragma unroll
                for (int t = 7; t < ALODLM_L; ++t) {
                    acc[t] += u0 * s_y[t * 256 + idx + 0];
                    acc[t] += u1 * s_y[t * 256 + idx + 1];
                }
            }
        }
    }
    if (global_m >= 2304) return;
    if (active_L <= 5) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* act_s = (float*)((char*)out_act + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                act_s[t * 2304 + global_m] = round_bf16_val(s_silu[wave][t] * acc[t]);
            }
        }
    } else {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* act_s = (float*)((char*)out_act + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                act_s[t * 2304 + global_m] = round_bf16_val(s_silu[wave][t] * acc[t]);
            }
        }
    }
}

// ALoDLM-4 L=13 SwiGLU MLP Down-Proj + Residual Add with 3-Tile LDS BF16x2 Staging & v_dot2_f32_bf16
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_mlp_down_kernel(
    const float*    __restrict__ in_act, // [13, 2304] (already BF16-rounded)
    const uint16_t* __restrict__ w_down,
    float*          __restrict__ out_h,  // [13, 768]
    int             active_L = 7
) {
    const int s = blockIdx.y;
    const char* s_base = (const char*)out_h - ALODLM_H_OFF + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    __shared__ uint32_t s_act_pk[ALODLM_L * 128]; // 6,656 B LDS (<= 16 KiB -> 16 Waves/SIMD true R9700 occupancy!)
    const int tid = threadIdx.x;
    const int global_row = blockIdx.x * 8 + (tid >> 5);
    const int lane = tid & 31;
    const float* act_base = (const float*)((const char*)in_act + (size_t)s * SCR_PER_SEQ);
    const r4d_u32x4* w_row = (global_row < 768)
                                 ? reinterpret_cast<const r4d_u32x4*>(w_down + (size_t)global_row * 2304)
                                 : nullptr;

    float acc[ALODLM_L];
    #pragma unroll
    for (int t = 0; t < ALODLM_L; ++t) acc[t] = 0.0f;

    #pragma nounroll
    for (int it = 0; it < 9; ++it) {
        if (it > 0) __syncthreads();
        if (tid < 128) {
            int col_pair = it * 256 + 2 * tid;
            for (int t = 0; t < active_L; ++t) {
                uint32_t u0 = __builtin_bit_cast(uint32_t, act_base[t * 2304 + col_pair + 0]);
                uint32_t u1 = __builtin_bit_cast(uint32_t, act_base[t * 2304 + col_pair + 1]);
                s_act_pk[t * 128 + tid] = (u0 >> 16) | (u1 & 0xFFFF0000u);
            }
        }
        __syncthreads();

        if (!w_row) continue;

        int vec_idx = it * 32 + lane;
        r4d_u32x4 raw = __builtin_nontemporal_load(&w_row[vec_idx]);
        int p_base = lane * 4;
        uint32_t r0 = raw[0], r1 = raw[1], r2 = raw[2], r3 = raw[3];
        #pragma nounroll
        for (int p = 0; p < 4; ++p) {
            uint32_t w_pk = r0;
            r0 = r1; r1 = r2; r2 = r3;
            float w0 = __builtin_bit_cast(float, w_pk << 16);
            float w1 = __builtin_bit_cast(float, w_pk & 0xFFFF0000u);
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                uint32_t a_pk = s_act_pk[t * 128 + p_base + p];
                acc[t] += w0 * __builtin_bit_cast(float, a_pk << 16);
                acc[t] += w1 * __builtin_bit_cast(float, a_pk & 0xFFFF0000u);
            }
            if (active_L > 5) {
                #pragma unroll
                for (int t = 5; t < 7; ++t) {
                    uint32_t a_pk = s_act_pk[t * 128 + p_base + p];
                    acc[t] += w0 * __builtin_bit_cast(float, a_pk << 16);
                    acc[t] += w1 * __builtin_bit_cast(float, a_pk & 0xFFFF0000u);
                }
            }
            if (active_L > 7) {
                #pragma unroll
                for (int t = 7; t < ALODLM_L; ++t) {
                    uint32_t a_pk = s_act_pk[t * 128 + p_base + p];
                    acc[t] += w0 * __builtin_bit_cast(float, a_pk << 16);
                    acc[t] += w1 * __builtin_bit_cast(float, a_pk & 0xFFFF0000u);
                }
            }
        }
    }
    if (global_row >= 768) return;
    if (active_L <= 5) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* h_s = (float*)((char*)out_h + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < 5; ++t) {
                h_s[t * 768 + global_row] = round_bf16_val(h_s[t * 768 + global_row] + acc[t]);
            }
        }
    } else {
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                acc[t] += __shfl_down(acc[t], offset, 32);
            }
        }
        if (lane == 0) {
            float* h_s = (float*)((char*)out_h + (size_t)s * SCR_PER_SEQ);
            #pragma unroll
            for (int t = 0; t < ALODLM_L; ++t) {
                h_s[t * 768 + global_row] = round_bf16_val(h_s[t * 768 + global_row] + acc[t]);
            }
        }
    }
}

// ALoDLM-4 Final RMSNorm (latent[0..11] -> SCR_H1_OFF[0..5] & SCR_H2_OFF[0..5]) + Exit Gate Hazard Update
// Grid: dim3(12, M), Block: dim3(256)
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_alodlm_final_norm_gate_kernel(
    const FullDrafterGpuPointers* __restrict__ p,
    char* __restrict__ scratch,
    int depth,
    int slot_offset = 0,
    bool confidence_gate = false,
    int active_k = FULLDRAFTER_TOTAL_K,
    bool adaptive_slots = false
) {
    const int local_j = blockIdx.x;
    const int j = local_j + slot_offset; // Global proposal slot.
    const int s = blockIdx.y; // 0..M-1
    const int tid = threadIdx.x;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    if (dloop_stage2_skipped(s_base)) return;

    const float* s_h = (const float*)(s_base + ALODLM_H_OFF) + (1 + local_j) * H_DIM;
    float* s_lat     = (float*)(s_base + ALODLM_LATENT_OFF) + j * H_DIM;
    float* dst_h     = (j < BLOCK_K) ? ((float*)(s_base + SCR_H1_OFF) + j * H_DIM)
                                     : ((float*)(s_base + SCR_H2_OFF) + (j - BLOCK_K) * H_DIM);
    ((uint64_t*)(s_base + UNARY_BUCKET_OFF))[j * 256 + tid] = 0ULL;
    if (local_j == 0 && tid == 0) {
        *(uint32_t*)(s_base + UNARY_BUCKET_READY_OFF) = UNARY_BUCKET_MAGIC;
        // Every sequence resets the packed-head tag before its projection launch.
        *(uint32_t*)(s_base + SCR_TEMP_OFF + FULLDRAFTER_TOTAL_K * S_RANK * sizeof(float)) = 0u;
    }

    float sq = 0.0f;
    for (int i = tid; i < H_DIM; i += 256) {
        float v = s_h[i];
        s_lat[i] = v;
        sq += v * v;
    }
    float sum_sq = block_reduce_sum(sq);
    float scale  = rsqrtf(sum_sq / static_cast<float>(H_DIM) + FULLDRAFTER_RMS_NORM_EPS);
    const uint16_t* fn_w = (const uint16_t*)p->final_norm;
    const uint16_t* gw   = (const uint16_t*)p->exit_gate_net_weight;

    float g_dot = 0.0f;
    for (int i = tid; i < H_DIM; i += 256) {
        float hn = round_bf16_val(s_h[i] * scale * bf16_to_f32(fn_w[i]));
        dst_h[i] = hn;
        if (gw) g_dot += hn * bf16_to_f32(gw[i]);
    }
    float gate_sum = block_reduce_sum(g_dot);
    if (tid == 0) {
        float g_bias = p->exit_gate_net_bias ? bf16_to_f32(((const uint16_t*)p->exit_gate_net_bias)[0]) : 0.0f;
        float d_bias = (p->exit_gate_depth_bias && depth >= 0 && depth < 4)
                           ? bf16_to_f32(((const uint16_t*)p->exit_gate_depth_bias)[depth])
                           : 0.0f;
        float logit  = gate_sum + g_bias + d_bias;
        float hazard = 1.0f / (1.0f + expf(-logit));
        float* surv  = (float*)(s_base + ALODLM_SURV_OFF);
        surv[j] = round_bf16_val(1.0f - hazard);
        if (local_j == 0) {
            *(uint32_t*)(s_base + ALODLM_DEPTHS_OFF) = (uint32_t)depth;
        }
    }
    if (confidence_gate && local_j < 4 && p->dloop_theta > 0.0f &&
        p->confidence_norm && p->confidence_fc1 && p->confidence_fc2) {
        __shared__ uint32_t normalized[H_DIM / 2];
        const uint16_t* norm = (const uint16_t*)p->confidence_norm;
        // Confidence normalizes the BF16-rounded final-norm output, not s_h.
        __syncthreads();
        float conf_sq = 0.0f;
        for (int i = tid; i < H_DIM; i += 256) conf_sq += dst_h[i] * dst_h[i];
        const float conf_scale = rsqrtf(block_reduce_sum(conf_sq) / (float)H_DIM + 1e-6f);
        for (int i = tid; i < H_DIM / 2; i += 256)
            normalized[i] = pack_f32_to_bf16x2(
                dst_h[2 * i] * conf_scale * bf16_to_f32(norm[2 * i]),
                dst_h[2 * i + 1] * conf_scale * bf16_to_f32(norm[2 * i + 1]));
        __syncthreads();
        const r4d_u32x4* fc1_row = (const r4d_u32x4*)(
            (const uint16_t*)p->confidence_fc1 + (size_t)tid * H_DIM);
        float acc = 0.0f;
        #pragma nounroll
        for (int k = 0; k < H_DIM / 8; ++k) {
            const r4d_u32x4 w = fc1_row[k];
            #pragma unroll
            for (int z = 0; z < 4; ++z)
                acc = r4d_dot2_f32_bf16(w[z], normalized[k * 4 + z], acc);
        }
        const float w2 = bf16_to_f32(((const uint16_t*)p->confidence_fc2)[tid]);
        const float logit = block_reduce_sum((acc / (1.0f + expf(-acc))) * w2);
        if (tid == 0) {
            __hip_atomic_store((uint32_t*)(s_base + ALODLM_CONFIDENCE_OFF) + local_j,
                               __builtin_bit_cast(uint32_t, 1.0f / (1.0f + expf(-logit))),
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            // Last producer combines all four slots; no extra launch or grid spin.
            const uint32_t arrived = __hip_atomic_fetch_add(
                (uint32_t*)(s_base + ALODLM_CONFIDENCE_ARRIVAL_OFF), 1u,
                __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            if (arrived == 3u) {
                const uint32_t* prob_u = (const uint32_t*)(s_base + ALODLM_CONFIDENCE_OFF);
                const float p0 = __builtin_bit_cast(float, __hip_atomic_load(&prob_u[0], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                const float p1 = __builtin_bit_cast(float, __hip_atomic_load(&prob_u[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                const float p2 = __builtin_bit_cast(float, __hip_atomic_load(&prob_u[2], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                const float p3 = __builtin_bit_cast(float, __hip_atomic_load(&prob_u[3], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
                const float s2 = (p0 * p1) * p2;
                const float s3 = s2 * p3;
                const int want_k = s3 >= p->dloop_theta
                    ? active_k
                    : (adaptive_slots ? min(active_k, s2 >= 0.14f ? 4 : (p0 * p1 >= 0.18f ? 3 : 2)) : min(active_k, 4));
                *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF) = (uint32_t)want_k;
            }
        }
    }
    else if (confidence_gate && local_j == 0 && tid == 0) {
        *(uint32_t*)(s_base + DLOOP_VERIF_READY_OFF) = (uint32_t)active_k;
    }
}

// Pack up to four sequences into the existing 12 LDS vectors; each weight row is read once.
// Blocks 0..319 compute out_proj; blocks 320..327 compute selector_hidden_proj.
template <int SLOTS_PER_SEQ = FULLDRAFTER_TOTAL_K>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void fast_out_proj12_kernel(
    const void* __restrict__ out_proj_w,
    char* __restrict__ scratch,
    const void* __restrict__ sel_hproj_w,
    int total_M
) {
    constexpr int SEQS_PER_WG = FULLDRAFTER_TOTAL_K / SLOTS_PER_SEQ;
    const int s = blockIdx.y * SEQS_PER_WG;
    char* s_base = scratch + (size_t)s * SCR_PER_SEQ;
    bool all_skipped = true;
    #pragma unroll
    for (int i = 0; i < SEQS_PER_WG; ++i)
        if (s + i < total_M && !dloop_stage2_skipped(scratch + (size_t)(s + i) * SCR_PER_SEQ))
            all_skipped = false;
    if (all_skipped) return;

    const float* s_h1 = (const float*)(s_base + SCR_H1_OFF);
    const float* s_h2 = (const float*)(s_base + SCR_H2_OFF);
    const int tid = threadIdx.x;

    __shared__ float s_h_lds[FULLDRAFTER_TOTAL_K * 256]; // 12,288 B FP32 LDS (<= 16 KiB -> 16 Waves/SIMD true R9700 occupancy!)
    __shared__ uint32_t s_unres_mask;
    if (tid == 0) {
        uint32_t m = 0u;
        #pragma unroll
        for (int v = 0; v < FULLDRAFTER_TOTAL_K; ++v) {
            const int seq = s + v / SLOTS_PER_SEQ;
            if (seq < total_M) {
                const uint32_t* exits = (const uint32_t*)(scratch + (size_t)seq * SCR_PER_SEQ + ALODLM_EXITS_OFF);
                if (exits[v % SLOTS_PER_SEQ] == 0u) m |= (1u << v);
            }
        }
        s_unres_mask = m;
    }
    __syncthreads();
    const uint32_t unres_mask = s_unres_mask;
    if (unres_mask == 0u) return;

    const int global_row = blockIdx.x * 8 + (tid >> 5);
    const int lane = tid & 31;
    const uint16_t* w_row = global_row < W_DIM
        ? (const uint16_t*)out_proj_w + (size_t)global_row * H_DIM
        : (global_row < W_DIM + S_RANK && sel_hproj_w != nullptr
            ? (const uint16_t*)sel_hproj_w + (size_t)(global_row - W_DIM) * H_DIM
            : nullptr);
    float acc[FULLDRAFTER_TOTAL_K];
    #pragma unroll
    for (int k = 0; k < FULLDRAFTER_TOTAL_K; ++k) acc[k] = 0.0f;

    const bool active_q0 = (unres_mask & 0x007u) != 0u;
    const bool active_q1a = (unres_mask & 0x008u) != 0u;
    const bool active_q1b = (unres_mask & 0x030u) != 0u;
    const bool active_q2a = (unres_mask & 0x0C0u) != 0u;
    const bool active_q2b = (unres_mask & 0x100u) != 0u;
    const bool active_q3 = (unres_mask & 0xE00u) != 0u;
    #pragma nounroll
    for (int chunk = 0; chunk < 3; ++chunk) {
        if (chunk > 0) __syncthreads();
        const int c_off = chunk * 256 + tid;
        if constexpr (SEQS_PER_WG > 1) {
            #pragma unroll
            for (int v = 0; v < FULLDRAFTER_TOTAL_K; ++v) {
                float h = 0.0f;
                if ((unres_mask >> v) & 1u) {
                    const float* hk = (const float*)(scratch + (size_t)(s + v / SLOTS_PER_SEQ) * SCR_PER_SEQ + SCR_H1_OFF);
                    h = hk[(v % SLOTS_PER_SEQ) * H_DIM + c_off];
                }
                s_h_lds[v * 256 + tid] = h;
            }
        } else {
            if (active_q0) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) s_h_lds[k * 256 + tid] = s_h1[k * H_DIM + c_off];
            }
            if (active_q1a) s_h_lds[3 * 256 + tid] = s_h1[3 * H_DIM + c_off];
            if (active_q1b) {
                #pragma unroll
                for (int k = 4; k < 6; ++k) s_h_lds[k * 256 + tid] = s_h1[k * H_DIM + c_off];
            }
            if (active_q2a) {
                #pragma unroll
                for (int k = 0; k < 2; ++k) s_h_lds[(6 + k) * 256 + tid] = s_h2[k * H_DIM + c_off];
            }
            if (active_q2b) s_h_lds[8 * 256 + tid] = s_h2[2 * H_DIM + c_off];
            if (active_q3) {
                #pragma unroll
                for (int k = 3; k < 6; ++k) s_h_lds[(6 + k) * 256 + tid] = s_h2[k * H_DIM + c_off];
            }
        }
        __syncthreads();
        if (!w_row) continue;

        int col = chunk * 256 + lane * 8;
        int lds_col = lane * 8;
        uint4 raw = *reinterpret_cast<const uint4*>(w_row + col);
        uint32_t r0 = raw.x, r1 = raw.y, r2 = raw.z, r3 = raw.w;
        #pragma nounroll
        for (int p = 0; p < 4; ++p) {
            uint32_t w_pk = r0;
            r0 = r1; r1 = r2; r2 = r3;
            float w0 = __builtin_bit_cast(float, w_pk << 16);
            float w1 = __builtin_bit_cast(float, w_pk & 0xFFFF0000u);
            int idx = lds_col + 2 * p;
            if (active_q0) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    acc[k] += w0 * s_h_lds[k * 256 + idx + 0];
                    acc[k] += w1 * s_h_lds[k * 256 + idx + 1];
                }
            }
            if (active_q1a) {
                acc[3] += w0 * s_h_lds[3 * 256 + idx + 0];
                acc[3] += w1 * s_h_lds[3 * 256 + idx + 1];
            }
            if (active_q1b) {
                #pragma unroll
                for (int k = 4; k < 6; ++k) {
                    acc[k] += w0 * s_h_lds[k * 256 + idx + 0];
                    acc[k] += w1 * s_h_lds[k * 256 + idx + 1];
                }
            }
            if (active_q2a) {
                #pragma unroll
                for (int k = 6; k < 8; ++k) {
                    acc[k] += w0 * s_h_lds[k * 256 + idx + 0];
                    acc[k] += w1 * s_h_lds[k * 256 + idx + 1];
                }
            }
            if (active_q2b) {
                acc[8] += w0 * s_h_lds[8 * 256 + idx + 0];
                acc[8] += w1 * s_h_lds[8 * 256 + idx + 1];
            }
            if (active_q3) {
                #pragma unroll
                for (int k = 9; k < FULLDRAFTER_TOTAL_K; ++k) {
                    acc[k] += w0 * s_h_lds[k * 256 + idx + 0];
                    acc[k] += w1 * s_h_lds[k * 256 + idx + 1];
                }
            }
        }
    }
    if (global_row >= W_DIM + S_RANK) return;

    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int k = 0; k < FULLDRAFTER_TOTAL_K; ++k) {
            if ((unres_mask >> k) & 1u) acc[k] += __shfl_down(acc[k], offset, 32);
        }
    }

    if (lane == 0) {
        if (global_row < W_DIM) {
            if constexpr (SEQS_PER_WG > 1) {
                #pragma unroll
                for (int v = 0; v < FULLDRAFTER_TOTAL_K; ++v) {
                    if ((unres_mask >> v) & 1u) {
                        float* out = (float*)(scratch + (size_t)(s + v / SLOTS_PER_SEQ) * SCR_PER_SEQ + SCR_O1_OFF);
                        out[(v % SLOTS_PER_SEQ) * W_DIM + global_row] = round_bf16_val(acc[v]);
                    }
                }
            } else {
                float* s_o1 = (float*)(s_base + SCR_O1_OFF);
                float* s_o2 = (float*)(s_base + SCR_O2_OFF);
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    if ((unres_mask >> k) & 1u) s_o1[k * W_DIM + global_row] = round_bf16_val(acc[k]);
                    if ((unres_mask >> (6 + k)) & 1u) s_o2[k * W_DIM + global_row] = round_bf16_val(acc[6 + k]);
                }
            }
        } else if (sel_hproj_w != nullptr) {
            const int sel_r = global_row - W_DIM;
            #pragma unroll
            for (int v = 0; v < FULLDRAFTER_TOTAL_K; ++v) {
                if ((unres_mask >> v) & 1u) {
                    float* s_hp = (float*)(scratch + (size_t)(s + v / SLOTS_PER_SEQ) * SCR_PER_SEQ + SCR_TEMP_OFF);
                    s_hp[(v % SLOTS_PER_SEQ) * S_RANK + sel_r] = acc[v];
                }
            }
        }
    }
    // The next kernel consumes this tag only after all eight projection blocks finish.
    if (blockIdx.x == 320 && tid == 0 && sel_hproj_w != nullptr) {
        #pragma unroll
        for (int sub_s = 0; sub_s < SEQS_PER_WG; ++sub_s) {
            if (s + sub_s < total_M) {
                char* seq_base = scratch + (size_t)(s + sub_s) * SCR_PER_SEQ;
                *(uint32_t*)(seq_base + SCR_TEMP_OFF + FULLDRAFTER_TOTAL_K * S_RANK * sizeof(float)) = SEL_HPROJ_READY_TAG;
            }
        }
    }
}

// Quantize each active vector once, not once per vocabulary tile.
template <int SLOTS_PER_SEQ, int ACTIVE_VECS = FULLDRAFTER_TOTAL_K>
__global__ __launch_bounds__(256) void r4d_fulldrafter_quant_lmhead_in_kernel(
    char* scratch, size_t h1_off, size_t h2_off, int total_M, int slot_base = 0
) {
    constexpr int SEQS_PER_WG = ACTIVE_VECS / SLOTS_PER_SEQ > 0 ? ACTIVE_VECS / SLOTS_PER_SEQ : 1;
    const int first_seq = blockIdx.y * SEQS_PER_WG;
    bool all_skipped = true;
    #pragma unroll
    for (int i = 0; i < SEQS_PER_WG; ++i)
        if (first_seq + i < total_M && !dloop_stage2_skipped(scratch + (size_t)(first_seq + i) * SCR_PER_SEQ))
            all_skipped = false;
    if (all_skipped) return;
    const int v = blockIdx.x;
    const int seq = first_seq + v / SLOTS_PER_SEQ;
    if (seq >= total_M) return;
    const int slot = slot_base + (v % SLOTS_PER_SEQ);
    char* base = scratch + (size_t)seq * SCR_PER_SEQ;
    if (dloop_stage2_skipped(base) || ((const uint32_t*)(base + ALODLM_EXITS_OFF))[slot] != 0u) return;
    const float* hk = (const float*)(base + (slot < 6 ? h1_off : h2_off)) + (slot % 6) * W_DIM;
    const int tid = threadIdx.x;
    __shared__ float maxima[8];
    __shared__ float x_scale;
    float vmax = 0.0f;
    for (int c = tid; c < W_DIM; c += 256) vmax = fmaxf(vmax, fabsf(hk[c]));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) vmax = fmaxf(vmax, __shfl_down(vmax, off, 32));
    if ((tid & 31) == 0) maxima[tid >> 5] = vmax;
    __syncthreads();
    if (tid == 0) {
        vmax = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) vmax = fmaxf(vmax, maxima[w]);
        x_scale = vmax > 1e-8f ? vmax / 127.0f : 1.0f;
        ((float*)(base + LMHEAD_SCALE_OFF))[slot] = x_scale;
    }
    __syncthreads();
    uint32_t* packed_out = (uint32_t*)(base + LMHEAD_IN_OFF) + slot * (W_DIM / 4);
    for (int i = tid; i < W_DIM / 4; i += 256) {
        uint32_t packed = 0u;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float x = hk[4 * i + p] / x_scale;
            const int q = (int)nearbyintf(fminf(fmaxf(x, -127.0f), 127.0f));
            packed |= (uint32_t)(uint8_t)(int8_t)q << (8 * p);
        }
        packed_out[i] = packed;
    }
}

// Packed LM-Head: W4A8 stages four vectors once, or twelve vectors in two passes.
template <uint32_t D, int SLOTS_PER_SEQ = FULLDRAFTER_TOTAL_K, int ACTIVE_VECS = FULLDRAFTER_TOTAL_K>
__global__ __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(16, 16))) void r4d_fulldrafter_lmhead_gemv12_kernel(
    const uint16_t* __restrict__ d_lm_head0,     // [124160, D] BF16 in VRAM
    const uint16_t* __restrict__ d_lm_head1,     // optional
    uint32_t V_shard,
    uint32_t V_total,
    char*    __restrict__ scratch_base,
    size_t   s1_hidden_off,
    size_t   s2_hidden_off,
    int      logit_slot_offset,
    int      total_M,
    const int8_t* q8_rank0,
    const int8_t* q8_rank1,
    const float* scale_rank0,
    const float* scale_rank1,
    const float* col_scale,
    int      slot_base = 0
) {
    static_assert(ACTIVE_VECS == 4 || ACTIVE_VECS == FULLDRAFTER_TOTAL_K);
    constexpr int SEQS_PER_WG = ACTIVE_VECS / SLOTS_PER_SEQ > 0 ? ACTIVE_VECS / SLOTS_PER_SEQ : 1;
    const int seq = blockIdx.y * SEQS_PER_WG;
    char* seq_scr = scratch_base + (size_t)seq * SCR_PER_SEQ;
    bool all_skipped = true;
    #pragma unroll
    for (int i = 0; i < SEQS_PER_WG; ++i)
        if (seq + i < total_M && !dloop_stage2_skipped(scratch_base + (size_t)(seq + i) * SCR_PER_SEQ))
            all_skipped = false;
    if (all_skipped) return;

    const float* d_h1 = (const float*)(seq_scr + s1_hidden_off);
    const float* d_h2 = (const float*)(seq_scr + s2_hidden_off);

    constexpr int K_STEPS = D / 256;
    constexpr int NUM_HALVES = (D == 2560) ? ((ACTIVE_VECS == 4) ? 2 : 5) : 3;
    constexpr int HALF_STEPS = K_STEPS / NUM_HALVES;
    constexpr int HALF_PAIRS = HALF_STEPS * 128;

    constexpr int Q8_HALVES = ACTIVE_VECS == 4 ? 1 : 2;
    constexpr int Q8_WORDS = D == 2560 ? D / (4 * Q8_HALVES) : 1;
    __shared__ union {
        uint32_t s_hidden[ACTIVE_VECS * HALF_PAIRS];
        uint32_t s_hidden_i8[ACTIVE_VECS * Q8_WORDS];
    } staging;
    uint32_t* s_hidden = staging.s_hidden;
    uint32_t* s_hidden_i8 = staging.s_hidden_i8;
    __shared__ float s_x_scale[ACTIVE_VECS];
    __shared__ uint64_t s_wg_u[ACTIVE_VECS][8];
    __shared__ uint32_t s_unres_mask;

    const int tid = threadIdx.x;
    if (tid == 0) {
        uint32_t m = 0u;
        #pragma unroll
        for (int v = 0; v < ACTIVE_VECS; ++v) {
            const int s = seq + v / SLOTS_PER_SEQ;
            if (s < total_M) {
                const uint32_t* exits = (const uint32_t*)(scratch_base + (size_t)s * SCR_PER_SEQ + ALODLM_EXITS_OFF);
                if (exits[slot_base + (v % SLOTS_PER_SEQ)] == 0u) m |= (1u << v);
            }
        }
        s_unres_mask = m;
    }
    __syncthreads();
    const uint32_t unres_mask = s_unres_mask;
    if (unres_mask == 0u) return;
    const int lane_id = tid & 31;
    const int wave_id = tid >> 5;
    const uint32_t global_row = blockIdx.x * 8 + wave_id;

    const bool is_shard1 = (d_lm_head1 != nullptr) && (global_row >= V_shard);
    const uint32_t row_in_shard = is_shard1 ? (global_row - V_shard) : global_row;
    const uint16_t* shard_mat = is_shard1 ? d_lm_head1 : d_lm_head0;
    const r4d_u32x4* lm_row = (global_row < V_total)
                                  ? (const r4d_u32x4*)(shard_mat + (size_t)row_in_shard * D)
                                  : nullptr;

    float acc[ACTIVE_VECS];
    #pragma unroll
    for (int k = 0; k < ACTIVE_VECS; ++k) acc[k] = 0.0f;

    const bool active_q0 = (unres_mask & 0x007u) != 0u;
    const bool active_q1a = (unres_mask & 0x008u) != 0u;
    const bool active_q1b = ACTIVE_VECS > 4 && (unres_mask & 0x030u) != 0u;
    const bool active_q2a = ACTIVE_VECS > 4 && (unres_mask & 0x0C0u) != 0u;
    const bool active_q2b = ACTIVE_VECS > 4 && (unres_mask & 0x100u) != 0u;
    const bool active_q3 = ACTIVE_VECS > 4 && (unres_mask & 0xE00u) != 0u;

    const bool use_q8 = D == 2560 && q8_rank0 != nullptr;
    if (use_q8) {
        if (tid < ACTIVE_VECS) {
            s_x_scale[tid] = ((unres_mask >> tid) & 1u)
                ? ((const float*)(scratch_base + (size_t)(seq + tid / SLOTS_PER_SEQ) * SCR_PER_SEQ + LMHEAD_SCALE_OFF))[slot_base + (tid % SLOTS_PER_SEQ)]
                : 1.0f;
        }
        int32_t iacc[ACTIVE_VECS] = {};
        const int8_t* qmat = is_shard1 ? q8_rank1 : q8_rank0;
        #pragma unroll
        for (int half = 0; half < Q8_HALVES; ++half) {
            if (half != 0) __syncthreads();
            #pragma unroll
            for (int k = 0; k < ACTIVE_VECS; ++k) {
                for (int i = tid; i < Q8_WORDS; i += 256) {
                    uint32_t packed = 0u;
                    if ((unres_mask >> k) & 1u) {
                        const uint32_t* packed_in = (const uint32_t*)(scratch_base
                            + (size_t)(seq + k / SLOTS_PER_SEQ) * SCR_PER_SEQ + LMHEAD_IN_OFF);
                        packed = packed_in[(slot_base + (k % SLOTS_PER_SEQ)) * (D / 4) + half * Q8_WORDS + i];
                    }
                    s_hidden_i8[k * Q8_WORDS + i] = packed;
                }
            }
            __syncthreads();
            if (lm_row) {
                const r4d_u32x4* row = (const r4d_u32x4*)(qmat + (size_t)row_in_shard * D);
                #pragma unroll
                for (int idx = lane_id; idx < Q8_WORDS / 4; idx += 32) {
                    const r4d_u32x4 w = __builtin_nontemporal_load(&row[half * (Q8_WORDS / 4) + idx]);
                    #pragma unroll
                    for (int k = 0; k < ACTIVE_VECS; ++k) {
                        if ((unres_mask >> k) & 1u) {
                            #pragma unroll
                            for (int p = 0; p < 4; ++p) {
                                const int x_idx = k * Q8_WORDS + idx * 4 + p;
                                iacc[k] = __builtin_amdgcn_sudot4(true, w[p], true, s_hidden_i8[x_idx], iacc[k], false);
                            }
                        }
                    }
                }
            }
        }
        if (lm_row) {
            const float row_scale = (is_shard1 ? scale_rank1 : scale_rank0)[row_in_shard];
            #pragma unroll
            for (int k = 0; k < ACTIVE_VECS; ++k) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) iacc[k] += __shfl_down(iacc[k], off, 32);
                acc[k] = (float)iacc[k] * (s_x_scale[k] * row_scale);
            }
        }
    } else {
    for (int half = 0; half < NUM_HALVES; ++half) {
        if (half > 0) __syncthreads();
        if constexpr (SEQS_PER_WG > 1) {
            for (int i = tid; i < ACTIVE_VECS * HALF_PAIRS; i += 256) {
                const int v = i / HALF_PAIRS;
                const int pair = i % HALF_PAIRS;
                uint32_t packed = 0u;
                if ((unres_mask >> v) & 1u) {
                    const float* hk = (const float*)(scratch_base + (size_t)(seq + v / SLOTS_PER_SEQ) * SCR_PER_SEQ + s1_hidden_off);
                    const int f_idx = (v % SLOTS_PER_SEQ) * (int)D + 2 * (half * HALF_PAIRS + pair);
                    uint32_t u0 = __builtin_bit_cast(uint32_t, hk[f_idx + 0]);
                    uint32_t u1 = __builtin_bit_cast(uint32_t, hk[f_idx + 1]);
                    packed = (u0 >> 16) | (u1 & 0xFFFF0000u);
                }
                s_hidden[i] = packed;
            }
        } else if constexpr (ACTIVE_VECS == 4) {
            for (int i = tid; i < 4 * HALF_PAIRS; i += 256) {
                int s = i / HALF_PAIRS;
                int slot = slot_base + s;
                int p_idx = half * HALF_PAIRS + (i % HALF_PAIRS);
                const float* dh = (slot < 6) ? (d_h1 + slot * (int)D) : (d_h2 + (slot - 6) * (int)D);
                if ((unres_mask >> s) & 1u) {
                    uint32_t u0 = __builtin_bit_cast(uint32_t, dh[2 * p_idx + 0]);
                    uint32_t u1 = __builtin_bit_cast(uint32_t, dh[2 * p_idx + 1]);
                    s_hidden[i] = (u0 >> 16) | (u1 & 0xFFFF0000u);
                }
            }
        } else {
            for (int i = tid; i < (ACTIVE_VECS < 6 ? ACTIVE_VECS : 6) * HALF_PAIRS; i += 256) {
                int s = i / HALF_PAIRS;
                int p_idx = half * HALF_PAIRS + (i % HALF_PAIRS);
                int f_idx = s * (int)D + 2 * p_idx;

                if ((unres_mask >> s) & 1u) {
                    uint32_t u0 = __builtin_bit_cast(uint32_t, d_h1[f_idx + 0]);
                    uint32_t u1 = __builtin_bit_cast(uint32_t, d_h1[f_idx + 1]);
                    s_hidden[s * HALF_PAIRS + (i % HALF_PAIRS)] = (u0 >> 16) | (u1 & 0xFFFF0000u);
                }
                if (ACTIVE_VECS > 6 && ((unres_mask >> (6 + s)) & 1u)) {
                    uint32_t v0 = __builtin_bit_cast(uint32_t, d_h2[f_idx + 0]);
                    uint32_t v1 = __builtin_bit_cast(uint32_t, d_h2[f_idx + 1]);
                    s_hidden[(6 + s) * HALF_PAIRS + (i % HALF_PAIRS)] = (v0 >> 16) | (v1 & 0xFFFF0000u);
                }
            }
        }
        __syncthreads();

        if (!lm_row) continue;

        #pragma nounroll
        for (int hs = 0; hs < HALF_STEPS; ++hs) {
            int k_step = half * HALF_STEPS + hs;
            int vec_offset = k_step * 32 + lane_id;
            r4d_u32x4 w_vec = __builtin_nontemporal_load(&lm_row[vec_offset]);
            int lds_base = (hs * 32 + lane_id) * 4;

            if (active_q0) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    uint32_t o0 = s_hidden[k * HALF_PAIRS + lds_base + 0];
                    uint32_t o1 = s_hidden[k * HALF_PAIRS + lds_base + 1];
                    uint32_t o2 = s_hidden[k * HALF_PAIRS + lds_base + 2];
                    uint32_t o3 = s_hidden[k * HALF_PAIRS + lds_base + 3];
                    float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                    float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                    float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                    float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                    acc[k] += (((0.0f + p0) + p1) + p2) + p3;
                }
            }
            if (active_q1a) {
                const int k = 3;
                uint32_t o0 = s_hidden[k * HALF_PAIRS + lds_base + 0];
                uint32_t o1 = s_hidden[k * HALF_PAIRS + lds_base + 1];
                uint32_t o2 = s_hidden[k * HALF_PAIRS + lds_base + 2];
                uint32_t o3 = s_hidden[k * HALF_PAIRS + lds_base + 3];
                float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                acc[k] += (((0.0f + p0) + p1) + p2) + p3;
            }
            if (active_q1b) {
                #pragma unroll
                for (int k = 4; k < 6; ++k) {
                    uint32_t o0 = s_hidden[k * HALF_PAIRS + lds_base + 0];
                    uint32_t o1 = s_hidden[k * HALF_PAIRS + lds_base + 1];
                    uint32_t o2 = s_hidden[k * HALF_PAIRS + lds_base + 2];
                    uint32_t o3 = s_hidden[k * HALF_PAIRS + lds_base + 3];
                    float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                    float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                    float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                    float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                    acc[k] += (((0.0f + p0) + p1) + p2) + p3;
                }
            }
            if (active_q2a) {
                #pragma unroll
                for (int k = 6; k < 8; ++k) {
                    uint32_t o0 = s_hidden[k * HALF_PAIRS + lds_base + 0];
                    uint32_t o1 = s_hidden[k * HALF_PAIRS + lds_base + 1];
                    uint32_t o2 = s_hidden[k * HALF_PAIRS + lds_base + 2];
                    uint32_t o3 = s_hidden[k * HALF_PAIRS + lds_base + 3];
                    float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                    float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                    float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                    float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                    acc[k] += (((0.0f + p0) + p1) + p2) + p3;
                }
            }
            if (active_q2b) {
                const int k = 8;
                uint32_t o0 = s_hidden[k * HALF_PAIRS + lds_base + 0];
                uint32_t o1 = s_hidden[k * HALF_PAIRS + lds_base + 1];
                uint32_t o2 = s_hidden[k * HALF_PAIRS + lds_base + 2];
                uint32_t o3 = s_hidden[k * HALF_PAIRS + lds_base + 3];
                float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                acc[k] += (((0.0f + p0) + p1) + p2) + p3;
            }
            if (active_q3) {
                #pragma unroll
                for (int k = 9; k < ACTIVE_VECS; ++k) {
                    uint32_t o0 = s_hidden[k * HALF_PAIRS + lds_base + 0];
                    uint32_t o1 = s_hidden[k * HALF_PAIRS + lds_base + 1];
                    uint32_t o2 = s_hidden[k * HALF_PAIRS + lds_base + 2];
                    uint32_t o3 = s_hidden[k * HALF_PAIRS + lds_base + 3];
                    float p0 = r4d_dot2_f32_bf16(w_vec[0], o0, 0.0f);
                    float p1 = r4d_dot2_f32_bf16(w_vec[1], o1, 0.0f);
                    float p2 = r4d_dot2_f32_bf16(w_vec[2], o2, 0.0f);
                    float p3 = r4d_dot2_f32_bf16(w_vec[3], o3, 0.0f);
                    acc[k] += (((0.0f + p0) + p1) + p2) + p3;
                }
            }
        }
    }
    }

    if (lm_row && !use_q8) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int k = 0; k < ACTIVE_VECS; ++k) {
                if ((unres_mask >> k) & 1u) acc[k] += __shfl_down(acc[k], offset, 32);
            }
        }
    }

    if (lane_id == 0) {
        const bool valid_act = (global_row < V_total) && (global_row < (uint32_t)V_ACT);
        #pragma unroll
        for (int k = 0; k < ACTIVE_VECS; ++k) {
            if ((unres_mask >> k) & 1u) {
                float val = acc[k];
                if (global_row < V_total) {
                    float* logits = (float*)(scratch_base + (size_t)(seq + k / SLOTS_PER_SEQ) * SCR_PER_SEQ + SCR_BASE_LOGITS_OFF);
                    logits[(size_t)(slot_base + (k % SLOTS_PER_SEQ) - logit_slot_offset) * V_total + global_row] = val;
                }
                s_wg_u[k][wave_id] = valid_act
                    ? ((static_cast<uint64_t>(float_to_ordered_u32(val)) << 32) | static_cast<uint64_t>(global_row))
                    : 0ULL;
            }
        }
    }
    __syncthreads();
    if (tid < ACTIVE_VECS && ((unres_mask >> tid) & 1u)) {
        uint64_t best_p = s_wg_u[tid][0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            uint64_t cand_p = s_wg_u[tid][w];
            if (cand_p > best_p) best_p = cand_p;
        }
        if (best_p != 0ULL) {
            unsigned long long* u_buckets = (unsigned long long*)(scratch_base + (size_t)(seq + tid / SLOTS_PER_SEQ) * SCR_PER_SEQ + UNARY_BUCKET_OFF);
            atomicMax(&u_buckets[(slot_base + (tid % SLOTS_PER_SEQ)) * 256 + (blockIdx.x & 255u)], (unsigned long long)best_p);
        }
    }
}

static int r4d_fulldrafter_alodlm_loop(
    const uint16_t* k0_ptr,
    const uint16_t* k1_ptr,
    const uint16_t* k2_ptr,
    const uint16_t* anc_ptr,
    const int32_t*  tok_ptr,
    int block_stride,
    int total_tok,
    const FullDrafterGpuPointers* p_ptr,
    char* scr_ptr,
    uint32_t* out_ptr,
    uint32_t M,
    hipStream_t stream,
    int64_t total_scratch_bytes = 0,
    hipEvent_t* prof_events = nullptr,
    int requested_k = 0
) {
    dim3 block_256(256);
    const bool is_fused = (p_ptr->is_fused == 1) || (p_ptr->lm_head_dim == 768) || (p_ptr->out_proj == nullptr);
    const uint32_t head_dim = is_fused ? 768 : W_DIM;
    const size_t s1_hidden_off = is_fused ? SCR_H1_OFF : SCR_O1_OFF;
    const size_t s2_hidden_off = is_fused ? SCR_H2_OFF : SCR_O2_OFF;
    const uint32_t active_v_total = (p_ptr->tp2_p2p_enabled != 0u)
                                        ? V_SHARD
                                        : ((p_ptr->teacher_lm_head_rank1 != nullptr) ? V_TOTAL : V_SHARD);
    const void* active_lm_head1 = (p_ptr->tp2_p2p_enabled != 0u) ? nullptr : p_ptr->teacher_lm_head_rank1;
    const int blocks_out_proj = (W_DIM + S_RANK + 7) / 8;

    int max_depth = (p_ptr->alodlm_max_depth > 0 && p_ptr->alodlm_max_depth <= 8) ? (int)p_ptr->alodlm_max_depth : 4;
    // Standalone callers omit pad; engine tensor metadata supplies scheduled depth.
    int req_k = 8;
    if (p_ptr->alodlm_pad > 0u && p_ptr->alodlm_pad <= (uint32_t)FULLDRAFTER_TOTAL_K) {
        req_k = (int)p_ptr->alodlm_pad;
    }
    if (requested_k > 0 && requested_k <= FULLDRAFTER_TOTAL_K) {
        req_k = requested_k;
    }
    const int active_k = (max_depth == 1) ? min(req_k, 8) : req_k;

    fulldrafter_stage1_backbone_kernel<<<dim3(96, M), block_256, 0, stream>>>(
        k0_ptr, k1_ptr, k2_ptr, anc_ptr, tok_ptr, block_stride, total_tok, p_ptr, scr_ptr
    );

    float* s_h    = (float*)(scr_ptr + ALODLM_H_OFF);
    float* s_norm = (float*)(scr_ptr + ALODLM_NORM_OFF);
    float* s_qkv  = (float*)(scr_ptr + ALODLM_QKV_OFF);

    const int stages = (max_depth == 1 && active_k > 4) ? 2 : 1;
    for (int stage = 0; stage < stages; ++stage) {
    const int slot_start = stage * 4;
    const bool adaptive_slots = false;
    const int stage_k = (max_depth == 1) ? min(active_k - slot_start, 4) : active_k;
    const int active_L = stage_k + 1;
    const int slot_end = slot_start + stage_k - 1;
    if (stage != 0) {
        fulldrafter_alodlm_stage2_prep_kernel<<<dim3(96, M), block_256, 0, stream>>>(
            p_ptr, scr_ptr, stage_k
        );
    }

    // 1. Run max_depth Latent-Residual Transformer Loops in 768-dim space (~0.22 ms per loop!)
    for (int depth = 0; depth < max_depth; ++depth) {
        if (depth > 0) {
            fulldrafter_alodlm_step_prep_kernel<<<dim3(active_k, M), block_256, 0, stream>>>(
                p_ptr, scr_ptr, depth
            );
        }

        for (int b = 0; b < 2; ++b) {
            const void* in_norm_w   = (b == 0) ? p_ptr->b0_input_layernorm : p_ptr->b1_input_layernorm;
            const void* q_w         = (b == 0) ? p_ptr->b0_q_proj : p_ptr->b1_q_proj;
            const void* k_w         = (b == 0) ? p_ptr->b0_k_proj : p_ptr->b1_k_proj;
            const void* v_w         = (b == 0) ? p_ptr->b0_v_proj : p_ptr->b1_v_proj;
            const void* o_w         = (b == 0) ? p_ptr->b0_o_proj : p_ptr->b1_o_proj;
            const void* post_norm_w = (b == 0) ? p_ptr->b0_post_attn_layernorm : p_ptr->b1_post_attn_layernorm;
            const void* gate_w      = (b == 0) ? p_ptr->b0_gate_proj : p_ptr->b1_gate_proj;
            const void* up_w        = (b == 0) ? p_ptr->b0_up_proj : p_ptr->b1_up_proj;
            const void* down_w      = (b == 0) ? p_ptr->b0_down_proj : p_ptr->b1_down_proj;

            r4d_fulldrafter_alodlm_dflash_conv_kernel<<<dim3(1, M), block_256, 0, stream>>>(
                s_h, s_norm, (const uint16_t*)p_ptr->conv_w_base, (const uint16_t*)p_ptr->conv_w_proj, in_norm_w,
                active_L,
                (stage == 0 && depth == 0 && b == 0) ? p_ptr : nullptr,
                (stage == 0 && depth == 0 && b == 0) ? out_ptr : nullptr
            );
            r4d_fulldrafter_alodlm_qkv_kernel<<<dim3(160, M), block_256, 0, stream>>>(
                s_norm, (const uint16_t*)q_w, (const uint16_t*)k_w, (const uint16_t*)v_w, s_qkv, active_L
            );
            r4d_fulldrafter_alodlm_attention_kernel<<<dim3(4, M), dim3(96), 0, stream>>>(
                s_qkv, s_norm, active_L
            );
            r4d_fulldrafter_alodlm_out_proj_kernel<<<dim3(96, M), block_256, 0, stream>>>(
                s_norm, (const uint16_t*)o_w, s_h, post_norm_w, s_norm, active_L
            );
            r4d_fulldrafter_alodlm_mlp_gate_up_kernel<<<dim3(288, M), block_256, 0, stream>>>(
                s_norm, (const uint16_t*)gate_w, (const uint16_t*)up_w, s_qkv, active_L
            );
            r4d_fulldrafter_alodlm_mlp_down_kernel<<<dim3(96, M), block_256, 0, stream>>>(
                s_qkv, (const uint16_t*)down_w, s_h, active_L
            );
        }

        r4d_fulldrafter_alodlm_final_norm_gate_kernel<<<dim3(stage_k, M), block_256, 0, stream>>>(
            p_ptr, scr_ptr, depth, slot_start,
            stage == 0 && stage_k == 4 && (stages > 1 || adaptive_slots), active_k, adaptive_slots
        );
        if (prof_events != nullptr && depth < 3) {
            (void)hipEventRecord(prof_events[depth], stream);
        }
    }
    if (prof_events != nullptr) {
        for (int d = max_depth; d < 3; ++d) {
            (void)hipEventRecord(prof_events[d], stream);
        }
    }

    // Pack stage-0 slots only: later stages retain their independent skip state and slot offsets.
    const int final_depth = max_depth - 1;
    auto launch_heads = [&](auto slots) {
        constexpr int slots_per_seq = decltype(slots)::value;
        constexpr int seqs_per_wg = FULLDRAFTER_TOTAL_K / slots_per_seq;
        const uint32_t groups = (M + seqs_per_wg - 1) / seqs_per_wg;
        if (!is_fused) {
            fast_out_proj12_kernel<slots_per_seq><<<dim3(blocks_out_proj, groups), block_256, 0, stream>>>(
                p_ptr->out_proj, scr_ptr, p_ptr->selector_hidden_proj, (int)M
            );
        }
        if (active_v_total <= V_SHARD || max_depth == 1) {
            const int logit_slot_offset = (active_v_total > V_SHARD) ? slot_start : 0;
            dim3 gemv_grid((active_v_total + 7) / 8, groups);
            if (head_dim == 2560 && p_ptr->teacher_lm_head_q8_rank0 != nullptr) {
                if (M == 1 && max_depth == 1 && stage_k <= 4) {
                    r4d_fulldrafter_quant_lmhead_in_kernel<12, 4><<<dim3(4, groups), block_256, 0, stream>>>(
                        scr_ptr, s1_hidden_off, s2_hidden_off, (int)M, slot_start);
                } else {
                    r4d_fulldrafter_quant_lmhead_in_kernel<slots_per_seq><<<dim3(FULLDRAFTER_TOTAL_K, groups), block_256, 0, stream>>>(
                        scr_ptr, s1_hidden_off, s2_hidden_off, (int)M, 0);
                }
            }
            if (head_dim == 768) {
                r4d_fulldrafter_lmhead_gemv12_kernel<768, slots_per_seq><<<gemv_grid, block_256, 0, stream>>>(
                    (const uint16_t*)p_ptr->teacher_lm_head_rank0, (const uint16_t*)active_lm_head1,
                    V_SHARD, active_v_total, scr_ptr, s1_hidden_off, s2_hidden_off, logit_slot_offset, (int)M,
                    nullptr, nullptr, nullptr, nullptr, nullptr
                );
            } else if (M == 1 && max_depth == 1 && stage_k <= 4 && (stage == 0 || p_ptr->teacher_lm_head_q8_rank0 != nullptr)) {
                r4d_fulldrafter_lmhead_gemv12_kernel<2560, 12, 4><<<gemv_grid, block_256, 0, stream>>>(
                    (const uint16_t*)p_ptr->teacher_lm_head_rank0, (const uint16_t*)active_lm_head1,
                    V_SHARD, active_v_total, scr_ptr, s1_hidden_off, s2_hidden_off, logit_slot_offset, (int)M,
                    p_ptr->teacher_lm_head_q8_rank0, active_lm_head1 ? p_ptr->teacher_lm_head_q8_rank1 : nullptr,
                    p_ptr->teacher_lm_head_scale_rank0, p_ptr->teacher_lm_head_scale_rank1,
                    p_ptr->teacher_lm_head_col_scale, slot_start
                );
            } else {
                r4d_fulldrafter_lmhead_gemv12_kernel<2560, slots_per_seq><<<gemv_grid, block_256, 0, stream>>>(
                    (const uint16_t*)p_ptr->teacher_lm_head_rank0, (const uint16_t*)active_lm_head1,
                    V_SHARD, active_v_total, scr_ptr, s1_hidden_off, s2_hidden_off, logit_slot_offset, (int)M,
                    p_ptr->teacher_lm_head_q8_rank0, active_lm_head1 ? p_ptr->teacher_lm_head_q8_rank1 : nullptr,
                    p_ptr->teacher_lm_head_scale_rank0, p_ptr->teacher_lm_head_scale_rank1,
                    p_ptr->teacher_lm_head_col_scale
                );
            }
        }
    };
    if (stage == 0 && max_depth == 1 && M > 1 && stage_k <= 3) {
        launch_heads(std::integral_constant<int, 3>{});
    } else if (stage == 0 && max_depth == 1 && M > 1 && stage_k == 4) {
        launch_heads(std::integral_constant<int, 4>{});
    } else {
        launch_heads(std::integral_constant<int, FULLDRAFTER_TOTAL_K>{});
    }

    if (active_v_total <= V_SHARD || max_depth == 1) {
        r4d_fulldrafter_alodlm_all_slots_select_kernel<<<dim3(M2_GRID_BLOCKS, 1), dim3(M2_THREADS), 0, stream>>>(
            p_ptr, scr_ptr, out_ptr, (int)M, final_depth, max_depth,
            active_k,
            slot_start, slot_end, total_scratch_bytes, requested_k > 0
        );
    } else {
        dim3 gemv_grid((active_v_total + 7) / 8, M);
        if (head_dim == 768) {
            r4d_fulldrafter_lmhead_gemv6_kernel<768><<<gemv_grid, block_256, 0, stream>>>(
                (const float*)(scr_ptr + s1_hidden_off), (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                (const uint16_t*)active_lm_head1, (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                V_SHARD, active_v_total, scr_ptr, 3
            );
        } else {
            r4d_fulldrafter_lmhead_gemv6_kernel<2560><<<gemv_grid, block_256, 0, stream>>>(
                (const float*)(scr_ptr + s1_hidden_off), (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                (const uint16_t*)active_lm_head1, (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                V_SHARD, active_v_total, scr_ptr, 3
            );
        }
        r4d_fulldrafter_alodlm_all_slots_select_kernel<<<dim3(M2_GRID_BLOCKS, 1), dim3(M2_THREADS), 0, stream>>>(
            p_ptr, scr_ptr, out_ptr, (int)M, final_depth, max_depth, active_k, 0, min(active_k, BLOCK_K) - 1, total_scratch_bytes
        );

        if (active_k > BLOCK_K) {
            if (head_dim == 768) {
                r4d_fulldrafter_lmhead_gemv6_kernel<768><<<gemv_grid, block_256, 0, stream>>>(
                    (const float*)(scr_ptr + s2_hidden_off), (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                    (const uint16_t*)active_lm_head1, (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                    V_SHARD, active_v_total, scr_ptr, 4
                );
            } else {
                r4d_fulldrafter_lmhead_gemv6_kernel<2560><<<gemv_grid, block_256, 0, stream>>>(
                    (const float*)(scr_ptr + s2_hidden_off), (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                    (const uint16_t*)active_lm_head1, (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                    V_SHARD, active_v_total, scr_ptr, 4
                );
            }
            r4d_fulldrafter_alodlm_all_slots_select_kernel<<<dim3(M2_GRID_BLOCKS, 1), dim3(M2_THREADS), 0, stream>>>(
                p_ptr, scr_ptr, out_ptr, (int)M, final_depth, max_depth, active_k, BLOCK_K, active_k - 1, total_scratch_bytes
            );
        }
    }
    }
    if (prof_events != nullptr) {
        (void)hipEventRecord(prof_events[3], stream);
    }

    hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? RAD_OK : RAD_E_DEVICE;
}

// ==============================================================================
// Entry Points for Libr4d Plugin
// ==============================================================================
extern "C" int64_t r4d_rad_fulldrafter_two_stage_scratch(const RadArgs* a) {
    long long M = 8;
    if (a && a->p) {
        for (int i = 0; i < a->n_p; ++i) {
            if (a->p[i].key && strcmp(a->p[i].key, "M") == 0) {
                if (a->p[i].kind == RAD_P_RANGE) {
                    M = a->p[i].ihi;
                } else if (a->p[i].kind == RAD_P_INT) {
                    M = a->p[i].ival;
                }
                break;
            }
        }
    }
    if (M < 1) M = 1;
    return (int64_t)M * SCR_PER_SEQ; // 8 MB per sequence
}

extern "C" int r4d_rad_fulldrafter_two_stage(const RadArgs* a, RadStream s) {
    if (!a) return RAD_E_INVAL;

    // Expected operands (strictly 7 required):
    // 0: ktap0, 1: ktap1, 2: ktap2, 3: ktap_anc, 4: tokens_in, 5: gpu_params, 6: tokens_out
    if (a->n_t < 7) {
        std::fprintf(stderr, "[FULLDRAFTER_HIP] ERROR: fulldrafter_two_stage requires 7 operands (got n_t=%d)\n", a->n_t);
        return RAD_E_INVAL;
    }
    const RadTensor* t_k0   = &a->t[0];
    const RadTensor* t_k1   = &a->t[1];
    const RadTensor* t_k2   = &a->t[2];
    const RadTensor* t_anc  = &a->t[3];
    const RadTensor* t_toks = &a->t[4];
    const RadTensor* t_par  = &a->t[5];
    const RadTensor* t_out  = &a->t[6];

    if (!t_k0 || !t_k1 || !t_k2 || !t_anc || !t_toks || !t_par || !t_out) return RAD_E_INVAL;
    if (!t_k0->data || !t_k1->data || !t_k2->data || !t_anc->data || !t_toks->data || !t_par->data || !t_out->data) {
        return RAD_E_INVAL;
    }

    const FullDrafterGpuPointers* p_ptr = (const FullDrafterGpuPointers*)t_par->data;

    long long M = 1;
    if (t_out->shape[0] > 0) {
        M = t_out->shape[0];
    } else if (t_k0->shape[0] > 0) {
        M = t_k0->shape[0];
    } else {
        rad_args_geti(a, "M", &M);
    }
    if (M <= 0) return RAD_OK;

    long long block_stride = 7;
    rad_args_geti(a, "block_stride", &block_stride);
    if (block_stride <= 0) block_stride = 7;

    const int64_t need_scratch = (int64_t)M * SCR_PER_SEQ;
    if (!a->scratch || (a->scratch_bytes > 0 && a->scratch_bytes < need_scratch && a->scratch_bytes < (int64_t)SCR_PER_SEQ)) {
        return RAD_E_SCRATCH;
    }

    hipStream_t stream = (hipStream_t)s;
    const uint16_t* k0_ptr  = (const uint16_t*)t_k0->data;
    const uint16_t* k1_ptr  = (const uint16_t*)t_k1->data;
    const uint16_t* k2_ptr  = (const uint16_t*)t_k2->data;
    const uint16_t* anc_ptr = (const uint16_t*)t_anc->data;
    const int32_t*  tok_ptr = (const int32_t*)t_toks->data;
    int total_tok = (t_toks->shape[0] > 0) ? (int)t_toks->shape[0] : (int)(M * block_stride);
    if (!tok_ptr || total_tok <= 0) {
        std::fprintf(stderr, "[FULLDRAFTER_HIP] ERROR: invalid or empty token_ids buffer (total_tok=%d)\n", total_tok);
        return RAD_E_INVAL;
    }
    uint32_t* out_ptr       = (uint32_t*)t_out->data;
    char* scr_ptr           = (char*)a->scratch;



    static int profile_check = -1;
    if (profile_check == -1) {
        profile_check = getenv("DRAFTER_PROFILE") ? 1 : 0;
    }

    hipEvent_t e0, e1, e2, e3, e4, e5, e6;
    if (profile_check) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2);
        (void)hipEventCreate(&e3); (void)hipEventCreate(&e4); (void)hipEventCreate(&e5);
        (void)hipEventCreate(&e6);
        (void)hipEventRecord(e0, stream);
    }

    if (p_ptr->is_alodlm_loop != 0u) {
        // The engine tags the opaque parameter tensor's host extent with its scheduled depth.
        // Legacy callers retain the full struct extent and use alodlm_pad/defaults.
        const int requested_k = (t_par->shape[0] >= 1 && t_par->shape[0] <= FULLDRAFTER_TOTAL_K)
                                    ? (int)t_par->shape[0] : 0;
        int eff_stride = (M > 0 && total_tok > 0 && (total_tok % (int)M) == 0) ? (total_tok / (int)M) : (int)block_stride;
        hipEvent_t prof_ev[4] = {e1, e2, e3, e4};
        int rc = r4d_fulldrafter_alodlm_loop(
            k0_ptr, k1_ptr, k2_ptr, anc_ptr, tok_ptr, eff_stride, total_tok,
            p_ptr, scr_ptr, out_ptr, (uint32_t)M, stream,
            a->scratch_bytes,
            profile_check ? prof_ev : nullptr,
            requested_k
        );
        if (!profile_check) return rc;
    } else {

    bool is_fused = (p_ptr->is_fused == 1) || (p_ptr->lm_head_dim == 768) || (p_ptr->out_proj == nullptr);
    uint32_t head_dim = is_fused ? 768 : W_DIM;
    size_t s1_hidden_off = is_fused ? SCR_H1_OFF : SCR_O1_OFF;
    size_t s2_hidden_off = is_fused ? SCR_H2_OFF : SCR_O2_OFF;

    // Launch 1: Stage 1 Backbone
    r4d_fulldrafter_stage1_backbone(
        k0_ptr, k1_ptr, k2_ptr, anc_ptr, tok_ptr, (int)block_stride, total_tok, p_ptr, scr_ptr, (uint32_t)M, stream
    );

    int threads = 256;
    int blocks_out_proj = ((W_DIM + (threads / 32) - 1) / (threads / 32)) + 2; // 320 + Block 320 (sel_hproj) + Block 321 (DLoop verif head)
    if (!is_fused) {
        fast_out_proj_kernel<<<dim3(blocks_out_proj, (uint32_t)M), dim3(threads), 0, stream>>>(
            p_ptr->out_proj, scr_ptr, SCR_H1_OFF, SCR_O1_OFF, p_ptr->selector_hidden_proj,
            p_ptr, out_ptr, (int)M
        );
    }

    if (profile_check) (void)hipEventRecord(e1, stream);

    const uint32_t active_v_total = (p_ptr->tp2_p2p_enabled != 0u)
                                        ? V_SHARD
                                        : ((p_ptr->teacher_lm_head_rank1 != nullptr) ? V_TOTAL : V_SHARD);
    const void* active_lm_head1 = (p_ptr->tp2_p2p_enabled != 0u) ? nullptr : p_ptr->teacher_lm_head_rank1;

    // Launch 2: Stage 1 Batched 6-Slot Vectorized Multi-GEMV across all M sequences
    {
        dim3 gemv_grid((active_v_total + 7) / 8, (uint32_t)M);
        dim3 gemv_block(256);
        if (head_dim == 768) {
            r4d_fulldrafter_lmhead_gemv6_kernel<768><<<gemv_grid, gemv_block, 0, stream>>>(
                (const float*)(scr_ptr + s1_hidden_off),
                (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                (const uint16_t*)active_lm_head1,
                (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                V_SHARD, active_v_total, scr_ptr, 1
            );
        } else {
            r4d_fulldrafter_lmhead_gemv6_kernel<2560><<<gemv_grid, gemv_block, 0, stream>>>(
                (const float*)(scr_ptr + s1_hidden_off),
                (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                (const uint16_t*)active_lm_head1,
                (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                V_SHARD, active_v_total, scr_ptr, 1
            );
        }
    }

    if (profile_check) (void)hipEventRecord(e2, stream);

    // Launch 3: Stage 1 Slots 0..5 (Milestone 2 Fused In-LDS Reduction & Markov Select)
    for (int slot = 0; slot < BLOCK_K; ++slot) {
        r4d_fulldrafter_slot_markov_select(
            1, slot, p_ptr, scr_ptr, out_ptr, (uint32_t)M, stream
        );
    }

    if (profile_check) (void)hipEventRecord(e3, stream);

    // Launch 4: Stage 2 Backbone (automatically skipped on-GPU per sequence when DLoop verification < theta)
    r4d_fulldrafter_stage2_backbone(
        p_ptr, scr_ptr, out_ptr, (uint32_t)M, stream
    );

    if (!is_fused) {
        fast_out_proj_kernel<<<dim3(blocks_out_proj, (uint32_t)M), dim3(threads), 0, stream>>>(
            p_ptr->out_proj, scr_ptr, SCR_H2_OFF, SCR_O2_OFF, p_ptr->selector_hidden_proj,
            p_ptr, out_ptr, (int)M
        );
    }

    if (profile_check) (void)hipEventRecord(e4, stream);

    // Launch 5: Stage 2 Batched 6-Slot Vectorized Multi-GEMV across all M sequences
    {
        dim3 gemv_grid((active_v_total + 7) / 8, (uint32_t)M);
        dim3 gemv_block(256);
        if (head_dim == 768) {
            r4d_fulldrafter_lmhead_gemv6_kernel<768><<<gemv_grid, gemv_block, 0, stream>>>(
                (const float*)(scr_ptr + s2_hidden_off),
                (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                (const uint16_t*)active_lm_head1,
                (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                V_SHARD, active_v_total, scr_ptr, 2
            );
        } else {
            r4d_fulldrafter_lmhead_gemv6_kernel<2560><<<gemv_grid, gemv_block, 0, stream>>>(
                (const float*)(scr_ptr + s2_hidden_off),
                (const uint16_t*)p_ptr->teacher_lm_head_rank0,
                (const uint16_t*)active_lm_head1,
                (float*)(scr_ptr + SCR_BASE_LOGITS_OFF),
                V_SHARD, active_v_total, scr_ptr, 2
            );
        }
    }

    if (profile_check) (void)hipEventRecord(e5, stream);

    // Launch 6: Stage 2 Slots 0..5 (Milestone 2 Fused In-LDS Reduction & Markov Select)
    for (int slot = 0; slot < BLOCK_K; ++slot) {
        r4d_fulldrafter_slot_markov_select(
            2, slot, p_ptr, scr_ptr, out_ptr, (uint32_t)M, stream
        );
    }
    } // end else (!is_alodlm_loop)

    if (profile_check) {
        (void)hipEventRecord(e6, stream);
        (void)hipStreamSynchronize(stream);
        if (p_ptr->is_alodlm_loop != 0u) {
            float t_d0 = 0.0f, t_d1 = 0.0f, t_d2 = 0.0f, t_d3 = 0.0f, t_tot = 0.0f;
            (void)hipEventElapsedTime(&t_d0, e0, e1);
            (void)hipEventElapsedTime(&t_d1, e1, e2);
            (void)hipEventElapsedTime(&t_d2, e2, e3);
            (void)hipEventElapsedTime(&t_d3, e3, e4);
            (void)hipEventElapsedTime(&t_tot, e0, e6);
            std::printf("[PROFILE] ALoDLM-4 (M=%lld): d0: %.3f ms | d1: %.3f ms | d2: %.3f ms | d3: %.3f ms | Total: %.3f ms\n",
                        M, t_d0, t_d1, t_d2, t_d3, t_tot);
        } else {
            float t_bb1 = 0, t_gemv1 = 0, t_slots1 = 0;
            float t_bb2 = 0, t_gemv2 = 0, t_slots2 = 0;
            (void)hipEventElapsedTime(&t_bb1, e0, e1);
            (void)hipEventElapsedTime(&t_gemv1, e1, e2);
            (void)hipEventElapsedTime(&t_slots1, e2, e3);
            (void)hipEventElapsedTime(&t_bb2, e3, e4);
            (void)hipEventElapsedTime(&t_gemv2, e4, e5);
            (void)hipEventElapsedTime(&t_slots2, e5, e6);
            std::printf("[PROFILE] S1_BB: %.3f ms | S1_GEMV: %.3f ms | S1_SLOTS: %.3f ms | S2_BB: %.3f ms | S2_GEMV: %.3f ms | S2_SLOTS: %.3f ms | Total: %.3f ms\n",
                        t_bb1, t_gemv1, t_slots1, t_bb2, t_gemv2, t_slots2,
                        t_bb1 + t_gemv1 + t_slots1 + t_bb2 + t_gemv2 + t_slots2);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);
        (void)hipEventDestroy(e3); (void)hipEventDestroy(e4); (void)hipEventDestroy(e5);
        (void)hipEventDestroy(e6);
    }

    hipError_t err = hipGetLastError();
    return (err == hipSuccess) ? RAD_OK : RAD_E_DEVICE;
}
