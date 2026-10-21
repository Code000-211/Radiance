#include <hip/hip_runtime.h>
#include "mtp_overlay.h"
#include <cmath>

namespace {
/* rad_plugin's converters are host-only. These device bit conversions implement
 * the same BF16 RTNE and OCP E4M3FN encodings, without target-private helpers. */
__device__ float load_kv(const void* p,int64_t offset,uint32_t dtype) {
    if(dtype==RAD_BF16) return __uint_as_float(uint32_t(static_cast<const uint16_t*>(p)[offset])<<16);
    const uint8_t x=static_cast<const uint8_t*>(p)[offset];
    const int e=(x>>3)&15, m=x&7;
    if(e==15 && m==7) return NAN;
    const float mag=e ? ldexpf(float(8+m),e-10) : ldexpf(float(m),-9);
    return (x&128) ? -mag : mag;
}
__device__ uint16_t narrow(float f) {
    uint32_t u=__float_as_uint(f);
    if((u&0x7fffffffU)>0x7f800000U) return uint16_t((u>>16)|0x40U);
    u+=0x7fffU+((u>>16)&1U); return uint16_t(u>>16);
}
__global__ void overlay(RadMtpOverlayCall c) {
    const int64_t r=blockIdx.x, request=r/8;
    const int head=blockIdx.y, lane=threadIdx.x;
    __shared__ float work[256];
    __shared__ float alpha, weight, denominator;
    __shared__ int valid, length;
    if(lane==0) {
        const int id=c.request_ids[request*c.request_stride];
        bool ok=id>=0 && id<c.N;
        if(ok) ok=atomicAdd(c.meta+int64_t(id)*c.meta_ld,0)==1;
        const int prefix=c.prefix[r*c.prefix_stride];
        length=prefix;
        ok=ok && prefix>=1 && prefix<=c.max_ctx && (int64_t(prefix)+3)/4<=c.bt_pitch;
        if(ok) {
            for(int64_t b=0;b<(int64_t(prefix)+3)/4;++b) {
                const int page=c.committed_bt[request*c.bt_ld+b];
                if(page<0 || page>=c.pages) { ok=false; break; }
            }
        }
        for(int d=0;d<4;++d) {
            const int slot=c.ancestors[r*c.ancestor_ld+d];
            if(d<c.depth) {
                const int64_t lo=request*32+d*8;
                if(slot<lo || slot>=lo+8 || int64_t(slot)/4>=c.provisional_pages) ok=false;
                if(d==c.depth-1 && slot!=lo+r%8) ok=false;
            } else if(slot!=-1) ok=false;
        }
        if(!ok) {
            if(id>=0 && id<c.N) atomicExch(c.meta+int64_t(id)*c.meta_ld,0);
            else for(int64_t i=0;i<c.N;++i) atomicExch(c.meta+i*c.meta_ld,0);
        }
        valid=ok; denominator=0;
    }
    __syncthreads();
    if(!valid) {
        if(lane<c.head_dim) c.out[r*c.out_ld+int64_t(head)*c.head_dim+lane]=0;
        return;
    }
    const int kh=head/(c.n_head/c.n_head_kv);
    const float query=lane<c.head_dim ? __uint_as_float(uint32_t(c.q[r*c.q_ld+int64_t(head)*c.head_dim+lane])<<16) : 0;
    float accum=0, maximum=-INFINITY;
    const float scale=1.0f/sqrtf(float(c.head_dim));
    for(int64_t pos=0;pos<int64_t(length)+c.depth;++pos) {
        const bool committed=pos<length;
        const int64_t slot=committed ? pos : c.ancestors[r*c.ancestor_ld+pos-length];
        const int64_t page=committed ? c.committed_bt[request*c.bt_ld+slot/4] : slot/4;
        const int64_t* stride=committed ? c.committed_stride : c.provisional_stride;
        const void* kv=committed ? c.committed_kv : c.provisional_kv;
        const int64_t base=page*stride[0]+int64_t(kh)*stride[1]+(slot%4)*stride[2];
        const float key=lane<c.head_dim ? load_kv(kv,base+lane,c.kv_dtype) : 0;
        work[lane]=query*key; __syncthreads();
        for(int d=128;d;d/=2) { if(lane<d) work[lane]=work[lane]+work[lane+d]; __syncthreads(); }
        if(lane==0) {
            const float logit=work[0]*scale;
            const float next=fmaxf(maximum,logit);
            alpha=expf(maximum-next); weight=expf(logit-next);
            denominator=denominator*alpha+weight; maximum=next;
        }
        __syncthreads();
        if(lane<c.head_dim) {
            const float value=load_kv(kv,base+c.head_dim+lane,c.kv_dtype);
            accum=accum*alpha+weight*value;
        }
        __syncthreads();
    }
    if(lane<c.head_dim) c.out[r*c.out_ld+int64_t(head)*c.head_dim+lane]=narrow(accum/denominator);
}
}
int rad_mine_mtp_overlay_attn_launch(const RadArgs* a,RadStream stream) {
    RadMtpOverlayCall c{}; const int rc=rad_mtp_overlay_attn_args(a,&c); if(rc) return rc;
    void* args[]={&c};
    return hipLaunchKernel(reinterpret_cast<const void*>(overlay),dim3(static_cast<unsigned>(c.rows),static_cast<unsigned>(c.n_head)),
        dim3(256),args,0,reinterpret_cast<hipStream_t>(stream))==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int rad_mine_mtp_overlay_count() { return 1; }
const RadKernelInfo* rad_mine_mtp_overlay_at(int i) {
    static const RadKernelInfo row=rad_mtp_overlay_attn_row(rad_mine_mtp_overlay_attn_launch);
    return i==0 ? &row : nullptr;
}
