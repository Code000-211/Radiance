#include "mtp_overlay.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
float load(const void* p,int64_t offset,uint32_t dtype) {
    return dtype==RAD_BF16 ? rad_bf16_to_f32(static_cast<const uint16_t*>(p)[offset])
        : rad_fp8e4m3_to_f32(static_cast<const uint8_t*>(p)[offset]);
}
/* Separate scalar host implementation: no GPU metadata helper, reduction tree,
 * online recurrence, or temporary copy of either KV cache. */
bool metadata(const RadMtpOverlayCall& c,int64_t row,int& prefix) {
    const int64_t request=row/8;
    const int id=c.request_ids[request*c.request_stride];
    bool valid=id>=0 && id<c.N;
    if(valid) valid=c.meta[int64_t(id)*c.meta_ld]==1;
    prefix=c.prefix[row*c.prefix_stride];
    valid=valid && prefix>=1 && prefix<=c.max_ctx && (int64_t(prefix)+3)/4<=c.bt_pitch;
    if(valid) for(int64_t b=0;b<(int64_t(prefix)+3)/4;++b) {
        const int page=c.committed_bt[request*c.bt_ld+b];
        if(page<0 || page>=c.pages) { valid=false; break; }
    }
    for(int d=0;d<4;++d) {
        const int slot=c.ancestors[row*c.ancestor_ld+d];
        if(d<c.depth) {
            const int64_t lower=request*32+d*8;
            if(slot<lower || slot>=lower+8 || int64_t(slot)/4>=c.provisional_pages) valid=false;
            if(d==c.depth-1 && slot!=lower+row%8) valid=false;
        } else if(slot!=-1) valid=false;
    }
    if(!valid) {
        if(id>=0 && id<c.N) c.meta[int64_t(id)*c.meta_ld]=0;
        else for(int64_t i=0;i<c.N;++i) c.meta[i*c.meta_ld]=0;
    }
    return valid;
}
}
int rad_ref_mtp_overlay_attn(const RadArgs* a,RadStream) {
    RadMtpOverlayCall c{}; const int rc=rad_mtp_overlay_attn_args(a,&c); if(rc) return rc;
    const float scale=1.0f/std::sqrt(float(c.head_dim));
    for(int64_t row=0;row<c.rows;++row) {
        int prefix=0;
        if(!metadata(c,row,prefix)) {
            std::fill_n(c.out+row*c.out_ld,int64_t(c.n_head)*c.head_dim,uint16_t(0));
            continue;
        }
        for(int head=0;head<c.n_head;++head) {
            const int kvhead=head/(c.n_head/c.n_head_kv);
            auto element=[&](int64_t pos,int col) {
                if(pos<prefix) {
                    const int page=c.committed_bt[(row/8)*c.bt_ld+pos/4];
                    const int64_t offset=int64_t(page)*c.committed_stride[0]+int64_t(kvhead)*c.committed_stride[1]+
                        (pos%4)*c.committed_stride[2]+col;
                    return load(c.committed_kv,offset,c.kv_dtype);
                }
                const int slot=c.ancestors[row*c.ancestor_ld+pos-prefix];
                const int64_t offset=int64_t(slot/4)*c.provisional_stride[0]+int64_t(kvhead)*c.provisional_stride[1]+
                    int64_t(slot%4)*c.provisional_stride[2]+col;
                return load(c.provisional_kv,offset,c.kv_dtype);
            };
            auto logit=[&](int64_t pos) {
                float dot=0;
                for(int d=0;d<c.head_dim;++d) {
                    const float q=rad_bf16_to_f32(c.q[row*c.q_ld+int64_t(head)*c.head_dim+d]);
                    const float product=q*element(pos,d);
                    dot=dot+product;
                }
                return dot*scale;
            };
            float maximum=-std::numeric_limits<float>::infinity();
            for(int64_t pos=0;pos<int64_t(prefix)+c.depth;++pos) maximum=std::max(maximum,logit(pos));
            float numerator[256]{};
            float denominator=0;
            for(int64_t pos=0;pos<int64_t(prefix)+c.depth;++pos) {
                const float weight=std::exp(logit(pos)-maximum);
                denominator=denominator+weight;
                for(int d=0;d<c.head_dim;++d) {
                    const float product=weight*element(pos,c.head_dim+d);
                    numerator[d]=numerator[d]+product;
                }
            }
            for(int d=0;d<c.head_dim;++d)
                c.out[row*c.out_ld+int64_t(head)*c.head_dim+d]=rad_f32_to_bf16(numerator[d]/denominator);
        }
    }
    return RAD_OK;
}
int rad_ref_mtp_overlay_count() { return 1; }
const RadKernelInfo* rad_ref_mtp_overlay_at(int i) {
    static const RadKernelInfo row=[] {
        RadKernelInfo r=rad_mtp_overlay_attn_row(rad_ref_mtp_overlay_attn);
        r.name="ref_mtp_overlay_attn"; r.domain=RAD_DOMAIN_HOST; r.priority=0;
        return r;
    }();
    return i==0 ? &row : nullptr;
}
