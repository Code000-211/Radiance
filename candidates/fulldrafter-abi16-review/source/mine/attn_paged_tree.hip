#include <hip/hip_runtime.h>
#include "attn_paged_tree.h"
#include <cmath>

namespace {
__device__ float fp8(uint8_t x) {
    const int e=(x>>3)&15, m=x&7;
    if(e==15 && m==7) return NAN;
    const float mag=e ? ldexpf(float(8+m),e-10) : ldexpf(float(m),-9);
    return x&128 ? -mag : mag;
}
__device__ uint16_t bf16(float value) {
    uint32_t u=__float_as_uint(value);
    if((u&0x7fffffffU)>0x7f800000U) return uint16_t((u>>16)|0x40U);
    u+=0x7fffU+((u>>16)&1U); return uint16_t(u>>16);
}
// One wave owns one query/head/split. Every intermediate and partial is FP32,
// unlike upstream decode's F16 operands/probabilities and PF16 output partials.
// Numerik: verlustfrei-nicht-bitgenau; reductions and packing are not upstream's.
__global__ __launch_bounds__(32) void paged_tree_partial(RadAttnPagedTreeCall c,float* scratch) {
    const int job=blockIdx.x, split=blockIdx.y, lane=threadIdx.x;
    const int row=job/12, head=job%12;
    int seq=-1, prefix=0, ctx=0, valid=1;
    uint32_t mask0=0, mask1=0, mask2=0;
    if(lane==0) {
        if(c.cu[0]!=0 || c.cu[c.seqs]!=c.rows) valid=0;
        for(int s=0;s<c.seqs;++s) {
            const int begin=c.cu[s], end=c.cu[s+1];
            if(begin<0 || end<begin || end>c.rows) valid=0;
            if(row>=begin && row<end) seq=s;
        }
        if(seq<0) valid=0;
        if(valid) {
            const int begin=c.cu[seq], count=c.cu[seq+1]-begin;
            ctx=c.seqused[seq]; prefix=ctx-count;
            valid=count<=c.q_bound && ctx>=count && ctx<=c.max_ctx &&
                (int64_t(ctx)+3)/4<=c.bt_width && c.parent[begin]==-1;
            for(int j=1;j<count && valid;++j) valid=c.parent[begin+j]>=0 && c.parent[begin+j]<j;
            for(int64_t b=0;b<(int64_t(ctx)+3)/4 && valid;++b) {
                const int page=c.bt[int64_t(seq)*c.bt_pitch+b];
                valid=page>=0 && page<c.pages;
            }
            if(valid) {
                int cur=row-begin;
                // Ordered parents strictly decrease; the hard bound prevents malformed loops.
                for(int step=0;step<65 && cur>=0;++step) {
                    if(cur<32) mask0|=uint32_t(1)<<cur;
                    else if(cur<64) mask1|=uint32_t(1)<<(cur-32);
                    else mask2|=1u;
                    cur=c.parent[begin+cur];
                }
            }
        }
    }
    valid=__shfl(valid,0); seq=__shfl(seq,0); ctx=__shfl(ctx,0); prefix=__shfl(prefix,0);
    mask0=__shfl(mask0,0); mask1=__shfl(mask1,0); mask2=__shfl(mask2,0);
    float query[8], accum[8]{};
    #pragma unroll
    for(int d=0;d<8;++d)
        query[d]=__uint_as_float(uint32_t(c.q[(int64_t(row)*12+head)*256+lane+32*d])<<16);
    float maximum=-INFINITY, denominator=0;
    if(valid) {
        const int64_t lo=int64_t(ctx)*split/RAD_ATP_SPLITS;
        const int64_t hi=int64_t(ctx)*(split+1)/RAD_ATP_SPLITS;
        for(int64_t pos=lo;pos<hi;++pos) {
            if(pos>=prefix) {
                const int n=int(pos)-prefix;
                const bool visible=n<32 ? ((mask0>>n)&1u) : n<64 ? ((mask1>>(n-32))&1u) : (mask2&1u);
                if(!visible) continue;
            }
            const int page=c.bt[int64_t(seq)*c.bt_pitch+pos/4];
            const int64_t base=int64_t(page)*c.kv_block_pitch+(pos%4)*512;
            float dot=0;
            #pragma unroll
            for(int d=0;d<8;++d) dot=dot+query[d]*fp8(c.kv[base+lane+32*d]);
            #pragma unroll
            for(int delta=16;delta;delta/=2) dot+=__shfl_xor(dot,delta);
            const float logit=dot*(1.0f/16.0f), next=fmaxf(maximum,logit);
            const float alpha=expf(maximum-next), weight=expf(logit-next);
            denominator=denominator*alpha+weight; maximum=next;
            #pragma unroll
            for(int d=0;d<8;++d)
                accum[d]=accum[d]*alpha+weight*fp8(c.kv[base+256+lane+32*d]);
        }
    }
    float* partial=scratch+(int64_t(job)*RAD_ATP_SPLITS+split)*RAD_ATP_PARTIAL;
    #pragma unroll
    for(int d=0;d<8;++d) partial[lane+32*d]=accum[d];
    if(lane==0) { partial[256]=maximum; partial[257]=denominator; }
}
__global__ __launch_bounds__(32) void paged_tree_merge(RadAttnPagedTreeCall c,const float* scratch) {
    const int job=blockIdx.x, lane=threadIdx.x;
    const float* partial=scratch+int64_t(job)*RAD_ATP_SPLITS*RAD_ATP_PARTIAL;
    float maximum=-INFINITY;
    for(int s=0;s<RAD_ATP_SPLITS;++s)
        if(partial[s*RAD_ATP_PARTIAL+257]>0) maximum=fmaxf(maximum,partial[s*RAD_ATP_PARTIAL+256]);
    float numerator[8]{}, denominator=0;
    for(int s=0;s<RAD_ATP_SPLITS;++s) {
        const float* p=partial+s*RAD_ATP_PARTIAL;
        if(p[257]==0) continue; // includes invalid metadata and splits with no visible keys
        const float weight=expf(p[256]-maximum);
        denominator+=weight*p[257];
        #pragma unroll
        for(int d=0;d<8;++d) numerator[d]+=weight*p[lane+32*d];
    }
    #pragma unroll
    for(int d=0;d<8;++d)
        c.out[int64_t(job)*256+lane+32*d]=denominator==0 ? uint16_t(0) : bf16(numerator[d]/denominator);
}
int launch(const RadArgs* a,RadStream stream) {
    RadAttnPagedTreeCall c{}; const int rc=rad_attn_paged_tree_args(a,&c); if(rc) return rc;
    const int64_t need=rad_attn_paged_tree_scratch(a);
    if(!a->scratch || a->scratch_bytes<need) return RAD_E_SCRATCH;
    if(reinterpret_cast<uintptr_t>(a->scratch)%8) return RAD_E_ALIGN;
    const uintptr_t start=reinterpret_cast<uintptr_t>(a->scratch);
    if(uint64_t(need)>UINTPTR_MAX-start) return RAD_E_INVAL;
    // Scratch must be disjoint from every operand; allocation is the caller's responsibility.
    for(int i=0;i<a->n_t;++i) {
        const auto& t=a->t[i]; int64_t span=1;
        for(uint32_t j=0;j<t.rank;++j) span+=(t.shape[j]-1)*t.stride[j];
        const uint64_t size=t.dtype==RAD_BF16 ? 2 : t.dtype==RAD_I32 ? 4 : 1;
        const uintptr_t x=reinterpret_cast<uintptr_t>(t.data);
        if(start<x+uint64_t(span)*size && x<start+uint64_t(need)) return RAD_E_INVAL;
    }
    float* scratch=static_cast<float*>(a->scratch);
    void* args[]={&c,&scratch};
    const dim3 jobs(static_cast<unsigned>(c.rows*12));
    const auto s=reinterpret_cast<hipStream_t>(stream);
    hipError_t status=hipLaunchKernel(reinterpret_cast<const void*>(paged_tree_partial),
        dim3(jobs.x,RAD_ATP_SPLITS),dim3(32),args,0,s);
    if(status!=hipSuccess) return RAD_E_DEVICE;
    status=hipLaunchKernel(reinterpret_cast<const void*>(paged_tree_merge),jobs,dim3(32),args,0,s);
    return status==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}
int rad_mine_attn_paged_tree_count() { return 1; }
const RadKernelInfo* rad_mine_attn_paged_tree_at(int i) {
    static const RadKernelInfo row=rad_attn_paged_tree_row(launch);
    return i==0 ? &row : nullptr;
}
