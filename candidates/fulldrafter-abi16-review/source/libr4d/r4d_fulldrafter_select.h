#pragma once

#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define R4D_FDS_HD __host__ __device__
#define R4D_FDS_INLINE __forceinline__
#else
#define R4D_FDS_HD
#define R4D_FDS_INLINE inline
#endif

namespace r4d {

static constexpr float FULLDRAFTER_SENTINEL_SCORE = -1e30f;
static constexpr uint32_t FULLDRAFTER_SENTINEL_ID  = 0xFFFFFFFFu;

// Finite scores sort descending, then by lower token ID. Empty slots must lose
// even to valid scores below the scratch buffer's -1e30 initialization value.
R4D_FDS_HD R4D_FDS_INLINE bool fulldrafter_is_better(float va, uint32_t ia, float vb, uint32_t ib) {
    if (ia == FULLDRAFTER_SENTINEL_ID) return false;
    if (ib == FULLDRAFTER_SENTINEL_ID) return true;
    if (va > vb) return true;
    if (va < vb) return false;
    return ia < ib;
}

// Fixed-capacity sorted top-K candidate tracker (K = 16).
// Maintained in descending order under fulldrafter_is_better.
struct FullDrafterThreadTop16 {
    static constexpr int K = 16;
    float val[K];
    uint32_t id[K];

    R4D_FDS_HD R4D_FDS_INLINE void init() {
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            val[i] = FULLDRAFTER_SENTINEL_SCORE;
            id[i]  = FULLDRAFTER_SENTINEL_ID;
        }
    }

    R4D_FDS_HD R4D_FDS_INLINE void insert(float v, uint32_t i) {
        if (!fulldrafter_is_better(v, i, val[K - 1], id[K - 1])) {
            return;
        }
        int pos = K - 1;
        #pragma unroll
        for (int p = K - 1; p > 0; --p) {
            if (fulldrafter_is_better(v, i, val[p - 1], id[p - 1])) {
                val[p] = val[p - 1];
                id[p]  = id[p - 1];
                pos = p - 1;
            } else {
                break;
            }
        }
        val[pos] = v;
        id[pos]  = i;
    }

    R4D_FDS_HD R4D_FDS_INLINE float current_val(int local_idx) const {
        return (static_cast<unsigned>(local_idx) < K) ? val[local_idx] : FULLDRAFTER_SENTINEL_SCORE;
    }

    R4D_FDS_HD R4D_FDS_INLINE uint32_t current_id(int local_idx) const {
        return (static_cast<unsigned>(local_idx) < K) ? id[local_idx] : FULLDRAFTER_SENTINEL_ID;
    }
};

// Host CPU simulation helpers for shared regression verification.
inline void fulldrafter_block_select_step(
    const FullDrafterThreadTop16* threads,
    int* local_idx,
    int num_threads,
    float* out_val,
    uint32_t* out_id,
    int* winning_tid
) {
    float best_v = FULLDRAFTER_SENTINEL_SCORE;
    uint32_t best_i = FULLDRAFTER_SENTINEL_ID;
    int best_t = -1;
    for (int t = 0; t < num_threads; ++t) {
        float tv = threads[t].current_val(local_idx[t]);
        uint32_t ti = threads[t].current_id(local_idx[t]);
        if (fulldrafter_is_better(tv, ti, best_v, best_i)) {
            best_v = tv;
            best_i = ti;
            best_t = t;
        }
    }
    *out_val = best_v;
    *out_id = best_i;
    *winning_tid = best_t;
    if (best_t >= 0 && local_idx[best_t] < FullDrafterThreadTop16::K) {
        local_idx[best_t]++;
    }
}

inline void fulldrafter_block_top16(
    const FullDrafterThreadTop16* threads,
    int* local_idx,
    int num_threads,
    float* out_vals,
    uint32_t* out_ids
) {
    for (int k = 0; k < FullDrafterThreadTop16::K; ++k) {
        int winner = -1;
        fulldrafter_block_select_step(
            threads, local_idx, num_threads,
            &out_vals[k], &out_ids[k], &winner
        );
    }
}

inline void fulldrafter_merge_top16(
    float* cands_val,
    uint32_t* cands_id,
    int total_cands,
    float* out_vals,
    uint32_t* out_ids
) {
    for (int k = 0; k < FullDrafterThreadTop16::K; ++k) {
        float best_v = FULLDRAFTER_SENTINEL_SCORE;
        uint32_t best_i = FULLDRAFTER_SENTINEL_ID;
        int best_idx = -1;
        for (int i = 0; i < total_cands; ++i) {
            if (fulldrafter_is_better(cands_val[i], cands_id[i], best_v, best_i)) {
                best_v = cands_val[i];
                best_i = cands_id[i];
                best_idx = i;
            }
        }
        out_vals[k] = best_v;
        out_ids[k] = best_i;
        if (best_idx >= 0) {
            cands_val[best_idx] = FULLDRAFTER_SENTINEL_SCORE;
            cands_id[best_idx]  = FULLDRAFTER_SENTINEL_ID;
        }
    }
}

} // namespace r4d
