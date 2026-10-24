#include <hip/hip_runtime.h>
#include "mtp_tree_accum.h"

namespace {
/* One lane per sequence. Rounds is the validated max_rounds: 3 keeps the legacy
 * <=16-node private arrays; 8 is the fixed MTP8 tree with <=65 nodes. No host candidate
 * read, allocation, synchronization, or auxiliary launch. */
template <int Rounds>
__device__ __forceinline__ void accumulate_row(RadMtpTreeAccumCall c, int64_t s,
                                               const int32_t* merged = nullptr) {
    constexpr int Cap = Rounds == 3 ? 16 : 65;
    int32_t* const out = c.packed+s*c.out_ld;
    int root = -1;
    if (c.round == 1) {
        const int idx = c.ids[s*c.id_stride];
        if (idx >= 0 && idx < c.token_count) root = c.tokens[static_cast<int64_t>(idx)*c.token_stride];
        // History is reset below while snapshotting this issue's candidate rows.
    } else if (out[0] >= 1 && out[0] <= c.capacity) {
        root = out[1];
    }
    int32_t candidates[Rounds][8];
    for (int r=0;r<Rounds;++r) {
        int32_t* const row = c.history+r*c.hist_round_stride+s*c.hist_row_stride;
        if (r == c.round-1) {
            for (int k=0;k<8;++k) {
                const int32_t tok = merged ? merged[k] : c.cand[s*c.cand_ld+k];
                candidates[r][k] = tok;
                row[k] = tok;
            }
        } else if (c.round == 1) {
            for (int k=0;k<8;++k) row[k] = -1;
        } else if (r < c.round) {
            for (int k=0;k<8;++k) candidates[r][k] = row[k];
        }
    }
    bool valid = root >= 0 && root < c.vocab;
    for (int r=0;r<c.round;++r)
        for (int k=0;k<8 && candidates[r][k]>=0;++k)
            valid = valid && candidates[r][k] < c.vocab;

    // Construct privately: reading just-written packed entries in each sibling scan
    // creates a dependent device-memory chain for an otherwise tiny integer walk.
    int32_t tokens[Cap], parents[Cap];
    for (int k=0;k<c.capacity;++k) { tokens[k] = -1; parents[k] = -1; }
    int n = 0;
    if (valid) {
        tokens[0] = root;
        n = 1;
        int spine = 0;
        for (int r=0;r<c.round && n<c.capacity;++r) {
            const int tok = candidates[r][0];
            if (tok < 0) break;
            tokens[n] = tok;
            parents[n] = n-1;
            ++n; ++spine;
        }
        for (int r=0;r<spine && n<c.capacity;++r) {
            for (int k=1;k<8 && n<c.capacity;++k) {
                const int tok = candidates[r][k];
                if (tok < 0) break;
                bool duplicate = tok == candidates[r][0];
                for (int j=r+1;j<n && !duplicate;++j)
                    duplicate = parents[j] == r && tokens[j] == tok;
                if (duplicate) continue;
                tokens[n] = tok;
                parents[n] = r;
                ++n;
            }
        }
    }
    out[0] = n;
    for (int k=0;k<c.capacity;++k) {
        out[1+k] = tokens[k];
        out[1+c.capacity+k] = parents[k];
    }
}

template <int Rounds>
__global__ void accumulate(RadMtpTreeAccumCall c) {
    const int64_t s = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (s < c.M) accumulate_row<Rounds>(c,s);
}

template <int Rounds>
__global__ void merge_accumulate(RadMtpMergeTreeAccumCall c) {
    const int64_t s = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (s >= c.tree.M) return;
    const uint32_t* const row = c.gathered+s*c.gathered_ld;
    uint64_t keys[16];
    for (int k=0;k<16;++k) {
        const int32_t id = static_cast<int32_t>(row[2*k]);
        uint32_t bits = row[2*k+1];
        bits = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
        keys[k] = id < 0 ? 0 : (static_cast<uint64_t>(bits)<<32) | (0xffffffffu-row[2*k]);
    }
    int32_t merged[8];
    for (int k=0;k<8;++k) {
        uint64_t winner = 0;
        for (int j=0;j<16;++j) if (keys[j] > winner) winner = keys[j];
        uint32_t bits = static_cast<uint32_t>(winner>>32);
        bits = winner ? (bits & 0x80000000u ? bits & 0x7fffffffu : ~bits) : 0xff800000u;
        merged[k] = winner ? static_cast<int32_t>(0xffffffffu-static_cast<uint32_t>(winner)) : -1;
        c.idx[s*c.tree.cand_ld+k] = merged[k];
        c.val[s*c.val_ld+k] = bits;
        // libr4d's wave removes every lane equal to the key, not just one slot.
        for (int j=0;j<16;++j) if (keys[j] == winner) keys[j] = 0;
    }
    accumulate_row<Rounds>(c.tree,s,merged);
}
}

int rad_mine_mtp_tree_accum_launch(const RadArgs* a, RadStream stream) {
    RadMtpTreeAccumCall c{};
    const int rc = rad_mtp_tree_accum_args(a,&c);
    if (rc != RAD_OK) return rc;
    void* args[] = {&c};
    // max_rounds is validated to 3 or 8 by rad_mtp_tree_accum_args.
    const void* kernel = c.max_rounds == 8 ? (const void*)accumulate<8> : (const void*)accumulate<3>;
    return hipLaunchKernel(kernel, dim3(static_cast<unsigned>((c.M+63)/64)),
                           dim3(64), args, 0, reinterpret_cast<hipStream_t>(stream))
               == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int rad_mine_mtp_merge_tree_accum_launch(const RadArgs* a, RadStream stream) {
    RadMtpMergeTreeAccumCall c{};
    const int rc = rad_mtp_merge_tree_accum_args(a,&c);
    if (rc != RAD_OK) return rc;
    void* args[] = {&c};
    const void* kernel = c.tree.max_rounds == 8 ? (const void*)merge_accumulate<8>
                                                : (const void*)merge_accumulate<3>;
    return hipLaunchKernel(kernel,
                           dim3(static_cast<unsigned>((c.tree.M+63)/64)), dim3(64), args, 0,
                           reinterpret_cast<hipStream_t>(stream)) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int rad_mine_mtp_tree_accum_count(void) { return 2; }
const RadKernelInfo* rad_mine_mtp_tree_accum_at(int i) {
    static const RadKernelInfo rows[] = {
        rad_mtp_tree_accum_row(rad_mine_mtp_tree_accum_launch),
        rad_mtp_merge_tree_accum_row(rad_mine_mtp_merge_tree_accum_launch)};
    return i >= 0 && i < 2 ? &rows[i] : nullptr;
}
