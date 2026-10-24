#include <hip/hip_runtime.h>
#include "ple_tree.h"
#include "../libr4d/r4d_common.h"

#pragma clang fp contract(off)

namespace {
constexpr int kThreads=128;

__device__ __forceinline__ uint16_t raw(const RadPleTreeTensor& t,int a,int b=0,int d=0) {
    return static_cast<const uint16_t*>(t.data)[rad_ple_off(t,a,b,d)];
}
__device__ __forceinline__ float bf(const RadPleTreeTensor& t,int a,int b=0,int d=0) {
    return bf16_to_f32(raw(t,a,b,d));
}
__device__ __forceinline__ int64_t weight(const RadPleTreeTensor& t,int i) {
    return static_cast<const int64_t*>(t.data)[rad_ple_off(t,i)];
}
__device__ __forceinline__ void store(const RadPleTreeTensor& t,uint16_t value,int a,int b=0,int d=0) {
    static_cast<uint16_t*>(t.data)[rad_ple_off(t,a,b,d)]=value;
}

__global__ __launch_bounds__(kThreads) void ids_kernel(RadPleTreeCall c) {
    const int seq=blockIdx.x,node=threadIdx.x;
    if (!rad_ple_tree_metadata(c,seq)) return;
    const int bos=rad_ple_int(c.t[3],seq),end=rad_ple_int(c.t[3],seq+1);
    if (node>=end-bos) return;
    const int slot=rad_ple_int(c.t[2],seq),ctx=rad_ple_int(c.t[5],seq);
    for (int h=0;h<c.heads;++h) {
        const int64_t divisor=weight(c.t[7],h),offset=weight(c.t[8],h);
        if (divisor<=0 || offset<0 || offset>INT32_MAX || divisor-1>INT32_MAX-offset) return;
    }
    int64_t shifted[8];
    shifted[0]=rad_ple_int(c.t[0],bos+node);
    bool cut=false;
    for (int k=1;k<c.ngram;++k) {
        if (cut || shifted[k-1]==c.eos) { cut=true;shifted[k]=c.eos;continue; }
        const int source=rad_ple_tree_source(c,bos,node,k);
        shifted[k]=source>=0 ? rad_ple_int(c.t[0],bos+source) :
            ctx+source>=0 ? rad_ple_int(c.t[1],slot,c.ngram-1+source) : c.eos;
    }
    uint64_t mixed=0;
    const int per=c.heads/(c.ngram-1);
    for (int k=0;k<c.ngram;++k) {
        mixed^=static_cast<uint64_t>(shifted[k])*static_cast<uint64_t>(weight(c.t[6],k));
        if (!k) continue;
        const int64_t signed_mix=__builtin_bit_cast(int64_t,mixed);
        for (int j=0;j<per;++j) {
            const int head=(k-1)*per+j;
            const int64_t divisor=weight(c.t[7],head),remainder=signed_mix%divisor;
            const int64_t mod=remainder<0 ? remainder+divisor : remainder;
            const uint64_t id=static_cast<uint64_t>(mod)+static_cast<uint64_t>(weight(c.t[8],head));
            static_cast<int32_t*>(c.t[9].data)[rad_ple_off(c.t[9],bos+node,head)]=static_cast<int32_t>(id);
        }
    }
}

/* A lane owns one channel for the entire sequence. All prefill history reads
 * finish before that lane updates its window, without a second dispatch or a
 * cross-workgroup barrier, including arbitrary prefill lengths and hist<=64. */
__global__ __launch_bounds__(kThreads) void conv_kernel(RadPleTreeCall c) {
    const int seq=blockIdx.x,ch=blockIdx.y*kThreads+threadIdx.x;
    if (ch>=c.n) return;
    if (c.prefill) {
        for (int s=0;s<c.nseq;++s) if (!rad_ple_tree_metadata(c,s)) return;
    } else if (!rad_ple_tree_metadata(c,seq)) return;
    const int bos=rad_ple_int(c.t[4],seq),end=rad_ple_int(c.t[4],seq+1),length=end-bos;
    const int slot=rad_ple_int(c.t[3],seq),ctx=rad_ple_int(c.t[6],seq);
    float taps[8];
    for (int j=0;j<c.width;++j) taps[j]=bf(c.t[1],ch,j);
    for (int node=0;node<length;++node) {
        float acc=0;
        for (int j=0;j<c.width;++j) {
            const int lag=c.hist-j*c.dilation,source=rad_ple_tree_source(c,bos,node,lag);
            float value=0;
            if (source>=0) value=bf(c.t[0],bos+source,ch);
            else if (ctx+source>=0) value=bf(c.t[2],slot,ch,c.ring ? (ctx+source)%c.conv_len : c.hist+source);
            acc=acc+value*taps[j];
        }
        const float residual=c.t[7].data ? bf(c.t[7],bos+node,ch) : 0;
        store(c.t[8],f32_to_bf16(residual+acc/(1.0f+expf(-acc))),bos+node,ch);
        if (c.t[9].data) store(c.t[9],raw(c.t[0],bos+node,ch),bos+node,ch);
    }
    if (!c.prefill) return;
    if (c.ring) {
        const int first=length>c.conv_len ? length-c.conv_len : 0;
        for (int i=first;i<length;++i) store(c.t[2],raw(c.t[0],bos+i,ch),slot,ch,(ctx+i)%c.conv_len);
    } else {
        for (int i=0;i<c.hist;++i) {
            const int source=length-c.hist+i;
            const uint16_t value=source>=0 ? raw(c.t[0],bos+source,ch) :
                ctx+source>=0 ? raw(c.t[2],slot,ch,c.hist+source) : 0;
            store(c.t[2],value,slot,ch,i);
        }
    }
}

__global__ __launch_bounds__(kThreads) void commit_kernel(RadPleTreeCall c) {
    const int seq=blockIdx.x,ch=blockIdx.y*kThreads+threadIdx.x;
    if (ch>=c.n) return;
    // Reject the whole batch before any store, including another sequence's invalid path or slot.
    for (int s=0;s<c.nseq;++s) if (!rad_ple_tree_metadata(c,s)) return;
    const int bos=rad_ple_int(c.t[6],seq),ctx=rad_ple_int(c.t[8],seq),length=rad_ple_int(c.t[10],seq);
    if (ch==0) {
        const int slot=rad_ple_int(c.t[1],seq),window=c.ngram-1;
        for (int i=0;i<window;++i) {
            const int source=length-window+i;
            const int32_t value=source>=0 ? rad_ple_int(c.t[4],bos+rad_ple_int(c.t[9],seq,source)) :
                ctx+source>=0 ? rad_ple_int(c.t[0],slot,window+source) : c.eos;
            static_cast<int32_t*>(c.t[0].data)[rad_ple_off(c.t[0],slot,i)]=value;
        }
    }
    const int slot=rad_ple_int(c.t[3],seq);
    if (c.ring) {
        for (int i=0;i<length;++i) store(c.t[2],raw(c.t[5],bos+rad_ple_int(c.t[9],seq,i),ch),slot,ch,(ctx+i)%c.conv_len);
    } else {
        // Sources from old history lie strictly after each destination, so ascending stores are safe.
        for (int i=0;i<c.hist;++i) {
            const int source=length-c.hist+i;
            const uint16_t value=source>=0 ? raw(c.t[5],bos+rad_ple_int(c.t[9],seq,source),ch) :
                ctx+source>=0 ? raw(c.t[2],slot,ch,c.hist+source) : 0;
            store(c.t[2],value,slot,ch,i);
        }
    }
}

int ids_launch(const RadArgs* a,RadStream stream) {
    RadPleTreeCall c{};const int rc=rad_ple_tree_args(a,RAD_PLE_TREE_IDS,&c);if (rc!=RAD_OK) return rc;
    void* args[]={&c};
    return hipLaunchKernel((const void*)ids_kernel,dim3(c.nseq),dim3(kThreads),args,0,(hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int conv_launch(const RadArgs* a,RadStream stream) {
    RadPleTreeCall c{};const int rc=rad_ple_tree_args(a,RAD_PLE_TREE_CONV,&c);if (rc!=RAD_OK) return rc;
    void* args[]={&c};
    return hipLaunchKernel((const void*)conv_kernel,dim3(c.nseq,(c.n-1)/kThreads+1),dim3(kThreads),args,0,(hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int commit_launch(const RadArgs* a,RadStream stream) {
    RadPleTreeCall c{};const int rc=rad_ple_tree_args(a,RAD_PLE_TREE_COMMIT,&c);if (rc!=RAD_OK) return rc;
    void* args[]={&c};
    return hipLaunchKernel((const void*)commit_kernel,dim3(c.nseq,(c.n-1)/kThreads+1),dim3(kThreads),args,0,(hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
} // namespace

int rad_mine_ple_tree_count(void) { return RAD_PLE_TREE_OPS; }
const RadKernelInfo* rad_mine_ple_tree_at(int i) {
    static const RadKernelInfo rows[]={rad_ple_tree_row(RAD_PLE_TREE_IDS,RAD_DOMAIN_DEVICE,ids_launch),
        rad_ple_tree_row(RAD_PLE_TREE_CONV,RAD_DOMAIN_DEVICE,conv_launch),rad_ple_tree_row(RAD_PLE_TREE_COMMIT,RAD_DOMAIN_DEVICE,commit_launch)};
    return i>=0 && i<RAD_PLE_TREE_OPS ? rows+i : nullptr;
}
