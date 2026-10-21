#include "attn_paged_tree.h"
#include <algorithm>
#include <cmath>
#include <limits>

int rad_ref_attn_paged_tree_launch(const RadArgs* a,RadStream) {
    RadAttnPagedTreeCall c{}; const int rc=rad_attn_paged_tree_args(a,&c); if(rc) return rc;
    std::fill_n(c.out,int64_t(c.rows)*3072,uint16_t(0));
    // Validate the ragged map before deriving any row address.
    if(c.cu[0]!=0 || c.cu[c.seqs]!=c.rows) return RAD_OK;
    for(int s=0;s<c.seqs;++s)
        if(c.cu[s]<0 || c.cu[s+1]<c.cu[s] || c.cu[s+1]>c.rows) return RAD_OK;
    for(int seq=0;seq<c.seqs;++seq) {
        const int begin=c.cu[seq], count=c.cu[seq+1]-begin, ctx=c.seqused[seq];
        if(count==0) continue;
        bool valid=count<=c.q_bound && ctx>=count && ctx<=c.max_ctx &&
            (int64_t(ctx)+3)/4<=c.bt_width && c.parent[begin]==-1;
        for(int j=1;j<count && valid;++j) valid=c.parent[begin+j]>=0 && c.parent[begin+j]<j;
        for(int64_t b=0;b<(int64_t(ctx)+3)/4 && valid;++b) {
            const int page=c.bt[int64_t(seq)*c.bt_pitch+b];
            valid=page>=0 && page<c.pages;
        }
        if(!valid) continue;
        const int prefix=ctx-count;
        for(int node=0;node<count;++node) {
            bool ancestors[65]{};
            int cur=node;
            for(int step=0;step<65 && cur>=0;++step) {
                ancestors[cur]=true; cur=c.parent[begin+cur];
            }
            for(int head=0;head<12;++head) {
                auto element=[&](int pos,int col) {
                    const int page=c.bt[int64_t(seq)*c.bt_pitch+pos/4];
                    return rad_fp8e4m3_to_f32(c.kv[int64_t(page)*c.kv_block_pitch+(pos%4)*512+col]);
                };
                auto logit=[&](int pos) {
                    double dot=0;
                    for(int d=0;d<256;++d)
                        dot+=double(rad_bf16_to_f32(c.q[(int64_t(begin+node)*12+head)*256+d]))*element(pos,d);
                    return dot/16.0;
                };
                // Independent two-pass FP64 softmax, not the device online/split recurrence.
                double maximum=-std::numeric_limits<double>::infinity();
                for(int pos=0;pos<ctx;++pos)
                    if(pos<prefix || ancestors[pos-prefix]) maximum=std::max(maximum,logit(pos));
                double numerator[256]{}, denominator=0;
                for(int pos=0;pos<ctx;++pos) {
                    if(pos>=prefix && !ancestors[pos-prefix]) continue;
                    const double weight=std::exp(logit(pos)-maximum);
                    denominator+=weight;
                    for(int d=0;d<256;++d) numerator[d]+=weight*element(pos,256+d);
                }
                for(int d=0;d<256;++d)
                    c.out[(int64_t(begin+node)*12+head)*256+d]=rad_f32_to_bf16(float(numerator[d]/denominator));
            }
        }
    }
    return RAD_OK;
}
int rad_ref_attn_paged_tree_count() { return 1; }
const RadKernelInfo* rad_ref_attn_paged_tree_at(int i) {
    static const RadKernelInfo row=[] {
        auto r=rad_attn_paged_tree_row(rad_ref_attn_paged_tree_launch);
        r.name="ref_attn_paged_tree"; r.domain=RAD_DOMAIN_HOST; r.priority=0; r.scratch=nullptr;
        r.computes="Independent two-pass FP64 QK/softmax/PV oracle, BF16 output; ordered tree visibility.";
        return r;
    }();
    return i==0 ? &row : nullptr;
}
