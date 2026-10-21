#include "mtp_tree_accum.h"

#include <algorithm>
#include <cstring>

int rad_ref_mtp_tree_accum(const RadArgs* a, RadStream) {
    RadMtpTreeAccumCall c{};
    const int rc = rad_mtp_tree_accum_args(a,&c);
    if (rc != RAD_OK) return rc;
    for (int64_t s=0;s<c.M;++s) {
        int32_t* const packed = c.packed+s*c.out_ld;
        int32_t root = -1;
        if (c.round == 1) {
            const int32_t index = c.ids[s*c.id_stride];
            if (index >= 0 && index < c.token_count) root = c.tokens[index*c.token_stride];
            for (int r=0;r<c.max_rounds;++r)
                std::fill_n(c.history+r*c.hist_round_stride+s*c.hist_row_stride,8,-1);
        } else if (packed[0] > 0 && packed[0] <= c.capacity) root = packed[1];
        std::fill_n(packed,1+2*c.capacity,-1);
        packed[0] = 0;
        std::copy_n(c.cand+s*c.cand_ld,8,
                    c.history+(c.round-1)*c.hist_round_stride+s*c.hist_row_stride);
        bool valid = root >= 0 && root < c.vocab;
        for (int r=0;r<c.round;++r) {
            const int32_t* cr = c.history+r*c.hist_round_stride+s*c.hist_row_stride;
            for (int k=0;k<8 && cr[k]>=0;++k) valid = valid && cr[k]<c.vocab;
        }
        if (!valid) continue;
        /* Kept independent from device code: the old engine builder's two passes,
         * using local token/parent arrays before root-inclusive serialization. */
        int32_t tokens[65], parents[65];
        tokens[0] = root; parents[0] = -1;
        int n = 1, spine = 0;
        for (int r=0;r<c.round && n<c.capacity;++r) {
            const int32_t* cr = c.history+r*c.hist_round_stride+s*c.hist_row_stride;
            if (cr[0] < 0) break;
            tokens[n] = cr[0]; parents[n] = n-1;
            ++n; ++spine;
        }
        for (int r=0;r<spine && n<c.capacity;++r) {
            const int32_t* cr = c.history+r*c.hist_round_stride+s*c.hist_row_stride;
            for (int k=1;k<8 && n<c.capacity;++k) {
                const int32_t alt = cr[k];
                if (alt < 0) break;
                bool dup = alt == cr[0];
                for (int j=r+1;j<n && !dup;++j)
                    if (parents[j] == r && tokens[j] == alt) dup = true;
                if (dup) continue;
                tokens[n] = alt; parents[n] = r; ++n;
            }
        }
        packed[0] = n;
        std::copy_n(tokens,n,packed+1);
        std::copy_n(parents,n,packed+1+c.capacity);
    }
    return RAD_OK;
}
int rad_ref_mtp_merge_tree_accum(const RadArgs* a, RadStream stream) {
    RadMtpMergeTreeAccumCall c{};
    const int rc = rad_mtp_merge_tree_accum_args(a,&c);
    if (rc != RAD_OK) return rc;
    for (int64_t s=0;s<c.tree.M;++s) {
        uint64_t keys[16]{};
        for (int j=0;j<16;++j) {
            int32_t id;
            uint32_t bits;
            std::memcpy(&id,c.gathered+s*c.gathered_ld+2*j,4);
            std::memcpy(&bits,c.gathered+s*c.gathered_ld+2*j+1,4);
            if (id < 0) continue;
            const uint32_t ordered = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
            keys[j] = (static_cast<uint64_t>(ordered)<<32) | (0xffffffffu-static_cast<uint32_t>(id));
        }
        // Independent host algorithm: sort once, then compact identical keys.
        std::sort(keys,keys+16,[](uint64_t x, uint64_t y) { return x > y; });
        const auto end = std::unique(keys,keys+16);
        for (int k=0;k<8;++k) {
            const uint64_t key = k < end-keys ? keys[k] : 0;
            uint32_t bits = static_cast<uint32_t>(key>>32);
            bits = key ? (bits & 0x80000000u ? bits & 0x7fffffffu : ~bits) : 0xff800000u;
            c.idx[s*c.tree.cand_ld+k] =
                key ? static_cast<int32_t>(0xffffffffu-static_cast<uint32_t>(key)) : -1;
            std::memcpy(c.val+s*c.val_ld+k,&bits,4);
        }
    }
    RadTensor tree_t[] = {a->t[1],a->t[3],a->t[4],a->t[5],a->t[6]};
    RadArgs tree_a = *a; tree_a.t = tree_t; tree_a.n_t = 5;
    return rad_ref_mtp_tree_accum(&tree_a,stream);
}

int rad_ref_mtp_tree_accum_count(void) { return 2; }
const RadKernelInfo* rad_ref_mtp_tree_accum_at(int i) {
    static const RadKernelInfo rows[] = {
        [] {
            RadKernelInfo r = rad_mtp_tree_accum_row(rad_ref_mtp_tree_accum);
            r.name = "ref_mtp_tree_accum"; r.domain = RAD_DOMAIN_HOST; r.priority = 0;
            return r;
        }(),
        [] {
            RadKernelInfo r = rad_mtp_merge_tree_accum_row(rad_ref_mtp_merge_tree_accum);
            r.name = "ref_mtp_merge_tree_accum"; r.domain = RAD_DOMAIN_HOST; r.priority = 0;
            return r;
        }()};
    return i >= 0 && i < 2 ? &rows[i] : nullptr;
}
