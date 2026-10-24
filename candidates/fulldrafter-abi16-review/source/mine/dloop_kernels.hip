#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "dloop_ops.h"
#include "dloop_full.h"
#include "dloop_full_device.h"
#include "dloop_skinny.h"

namespace {
using namespace rad_dloop;
/* Batched matrix tiles. One wave covers ALL rows of a row chunk with one lane per row and a
 * register tile of `mt_cols` output columns; a CTA is `Waves` such waves over the same rows,
 * so the staged activation tile is shared and the weight tile covers Waves*mt_cols columns.
 * LDS strides are padded so every staged access is bank-conflict free, and each is a multiple
 * of four floats so the inner-loop LDS.128 reads stay 16-byte aligned. */
constexpr int mt_rows=32, mt_k=32, mt_pad=36;

// One persistent CTA owns four output columns, retaining their ENTIRE weight
// rows in LDS while every M row is processed. There is no per-frontier weight
// pass and no cross-CTA dependency. FP32 inputs never pass through BF16 WMMA.
struct MatArgs {
    const float* x;
    const uint16_t *w0,*w1,*w2,*ring,*anchor,*embed;
    const int *valid,*slots,*heads,*lengths,*path,*frontier;
    /* Optional uniform row-tile exit: null for every ring and full base/tree call, set by the full
     * append to the live count of the chunk being staged. A row tile that begins at or past that count
     * has no live row, so the whole tile may return before any barrier or weight load. */
    const int* live;
    float* y;
    int rows,in,out,xstride,rows_per_seq,kind,split,source;
    int front,slot,m,k,cw,f,vocab_lo,active;
};
/* Row resolution of the batched input source. Called once per row chunk and not per element: an
 * invalid sequence, a masked prefix row, and a row past the end all come back as a null source, so
 * the staging loop needs no branch beyond that test, and the wrapped ring offset is resolved here. */
struct RowSrc { const void* p; };
__device__ RowSrc matrix_row(const MatArgs& a,int row) {
    if(row>=a.rows)return {nullptr};
    const int s=row/a.rows_per_seq;
    if(!a.valid[s])return {nullptr};
    if(a.source==1) {
        const int t=row%a.cw,slot=a.slots[s],len=a.lengths[slot];
        if(t<a.cw-len)return {nullptr};
        return {a.ring+(int64_t(slot)*a.cw+(a.heads[slot]+t)%a.cw)*a.f};
    }
    if(a.source==2) {
        const int b=a.m+1,job=row/b,pos=row%b,seq=job/a.front;
        if(pos==0)return {a.anchor+int64_t(seq)*a.in};
        if(pos>a.slot)return {nullptr};
        const int parent=a.frontier[seq*max_frontier+job%a.front];
        if(parent<0||parent>=max_nodes)return {nullptr};
        const int index=a.path[(seq*max_nodes+parent)*a.m+pos-1];
        if(index<0||index>=a.m*a.k)return {nullptr};
        return {a.embed+(int64_t(seq)*a.m*a.k+index)*a.in};
    }
    return {a.x+int64_t(row)*a.xstride};
}
/* The weight row of an output column, before its stride is applied. `kind` only picks which operand
 * holds it: the packed QKV/KV projections split the column range across the three weight operands. */
__device__ const uint16_t* matrix_weight(const MatArgs& a,int col) {
    if(a.kind==1) {
        if(col<a.split)return a.w0+int64_t(col)*a.in;
        col-=a.split;
        const int kv=(a.out-a.split)/2;
        return col<kv?a.w1+int64_t(col)*a.in:a.w2+int64_t(col-kv)*a.in;
    }
    if(a.kind==2)return col<a.split?a.w0+int64_t(col)*a.in:a.w1+int64_t(col-a.split)*a.in;
    return a.w0+int64_t(col)*a.in;
}
/* Tiled batched matrix, one kernel for every shape of this op. The grid is
 * (column blocks, row chunks): a CTA owns BN=4*C output columns for one 32-row chunk and streams K
 * in 32-wide steps, so a CTA's weight slice crosses LDS once and one accumulator chain per output
 * sums the whole K. A lane holds one row and C columns in registers, so a weight read out of LDS is
 * wave-uniform and one LDS.128 of weights feeds 32 lanes x 4 k values (128 FMA).
 *
 * Both weight tiles are staged as 16-byte loads with the per-column row pointer resolved ONCE per
 * CTA into a shared table (`matrix_weight` has branches and 64-bit multiplies that cost more than
 * the 16 bytes it feeds, so it must not sit in the inner loop). The activation tile is staged from
 * the per-row source pointers resolved once per row chunk. Both tiles use pitch 36, which makes every
 * staged and every consumed float4 bank-conflict free and 16-byte aligned.
 *
 * K streams through LDS, so no LDS size depends on a matrix shape; the column block only widens with
 * the output width. FP32 activations and FP32 accumulation are unchanged; only the order within each
 * output's K-sum differs from the reference. Each active weight element is read from global exactly
 * once, and the grid covers every (column block, row chunk) pair. */
template<int C,bool Gate>
__global__ __launch_bounds__(threads) void dloop_matmul_tiled(MatArgs a) {
    constexpr int BN=threads/32*C,PAD=mt_pad;
    __shared__ float sx[mt_rows*PAD];
    __shared__ float sw[BN*PAD];
    __shared__ float sup[Gate?BN*PAD:1];
    __shared__ RowSrc srow[mt_rows];
    __shared__ const uint16_t* wrow[Gate?2*BN:BN];
    const int lane=threadIdx.x%32,cg=threadIdx.x/32,cfirst=blockIdx.x*BN,r0=blockIdx.y*mt_rows;
    const int cbase=cfirst+cg*C;
    /* A row tile that begins at or past the live count has no live row anywhere in it (live rows form a
     * prefix of the staged chunk), and nothing outside reads this tile: the store kernel re-tests each
     * row's own index, and the append's norm skips a fully dead chunk. The test is uniform across the
     * CTA and sits BEFORE the first barrier and the first weight load, so the tile costs one load and a
     * branch. `live` is null on every ring and full base/tree call, which is why this is a null test
     * rather than a second kernel. */
    if(a.live&&r0>=*a.live)return;
    if(a.active&&a.vocab_lo+cfirst>=a.active) {
        for(int r=0;r<mt_rows;++r)if(r0+r<a.rows)for(int c=0;c<C;++c) {
            const int col=cbase+c;
            if(col<a.out)a.y[int64_t(r0+r)*a.out+col]=-INFINITY;
        }
        return;
    }
    for(int i=threadIdx.x;i<mt_rows;++i)srow[i]=matrix_row(a,r0+i);
    for(int c=threadIdx.x;c<BN;c+=threads) {
        const int col=cfirst+c;
        const bool present=col<a.out&&(a.active==0||a.vocab_lo+col<a.active);
        wrow[c]=present?matrix_weight(a,col):nullptr;
        if constexpr(Gate)wrow[BN+c]=present?a.w1+int64_t(col)*a.in:nullptr;
    }
    __syncthreads();
    const bool srcf32=a.source==0;
    float acc[C],other[C];
    for(int c=0;c<C;++c){acc[c]=0;other[c]=0;}
    for(int k0=0;k0<a.in;k0+=mt_k) {
        for(int i=threadIdx.x;i<mt_rows*mt_k;i+=threads) {
            const int row=i/mt_k,d=i%mt_k,at=k0+d;
            const RowSrc r=srow[row];
            float v=0;
            if(r.p&&at<a.in)
                v=srcf32?static_cast<const float*>(r.p)[at]:bf(*(static_cast<const uint16_t*>(r.p)+at));
            sx[row*PAD+d]=v;
        }
        for(int i=threadIdx.x;i<BN*(mt_k/8);i+=threads) {
            const int c=i/(mt_k/8),q=(i%(mt_k/8))*8,at=k0+q;
            const uint16_t* wr=wrow[c];
            uint4 w{0,0,0,0},g{0,0,0,0};
            if(wr&&at+8<=a.in) {
                w=*reinterpret_cast<const uint4*>(wr+at);
                if constexpr(Gate)g=*reinterpret_cast<const uint4*>(wrow[BN+c]+at);
            }
            float* dst=sw+c*PAD+q;
            #pragma unroll
            for(int j=0;j<8;j+=4) {
                const uint32_t lo=j==0?w.x:w.z,hi=j==0?w.y:w.w;
                *reinterpret_cast<float4*>(dst+j)={bf(uint16_t(lo&0xffffu)),bf(uint16_t(lo>>16)),
                    bf(uint16_t(hi&0xffffu)),bf(uint16_t(hi>>16))};
                if constexpr(Gate) {
                    const uint32_t gl=j==0?g.x:g.z,gh=j==0?g.y:g.w;
                    *reinterpret_cast<float4*>(sup+c*PAD+q+j)={bf(uint16_t(gl&0xffffu)),bf(uint16_t(gl>>16)),
                        bf(uint16_t(gh&0xffffu)),bf(uint16_t(gh>>16))};
                }
            }
        }
        __syncthreads();
        if(r0+lane<a.rows)for(int kq=0;kq<mt_k;kq+=4) {
            const float4 xv=*reinterpret_cast<const float4*>(sx+lane*PAD+kq);
            #pragma unroll
            for(int c=0;c<C;++c) {
                const float4 wv=*reinterpret_cast<const float4*>(sw+(cg*C+c)*PAD+kq);
                acc[c]=fmaf(xv.x,wv.x,acc[c]);acc[c]=fmaf(xv.y,wv.y,acc[c]);
                acc[c]=fmaf(xv.z,wv.z,acc[c]);acc[c]=fmaf(xv.w,wv.w,acc[c]);
                if constexpr(Gate) {
                    const float4 uv=*reinterpret_cast<const float4*>(sup+(cg*C+c)*PAD+kq);
                    other[c]=fmaf(xv.x,uv.x,other[c]);other[c]=fmaf(xv.y,uv.y,other[c]);
                    other[c]=fmaf(xv.z,uv.z,other[c]);other[c]=fmaf(xv.w,uv.w,other[c]);
                }
            }
        }
        __syncthreads();
    }
    const int row=r0+lane;
    if(row<a.rows) {
        const bool seqok=a.valid[row/a.rows_per_seq]!=0;
        float* outrow=a.y+int64_t(row)*a.out;
        for(int c=0;c<C;++c) {
            const int col=cbase+c;
            if(col>=a.out)continue;
            float v=acc[c];
            if constexpr(Gate)v=(v/(1.0f+expf(-v)))*other[c];
            if(a.active&&(a.vocab_lo+col>=a.active||!seqok))v=-INFINITY;
            outrow[col]=v;
        }
    }
}

/* Append accepted tap rows to the per-slot ring. The body is a pure gather-and-copy: every row is
 * written to a consecutive ring offset, so the copy is a vectorised uint4 gather with no
 * per-element modulo and no barrier inside the loop (the old version synced the block 16 times per
 * sequence for a 192 KiB copy). f is a multiple of 8 elements in every supported geometry, which is
 * what makes the uint4 form exact; the tail loop covers any other multiple of two.
 *
 * Parallelism: one CTA per (sequence, tap row) instead of one CTA per sequence, because a single
 * CTA of 128 threads copying a whole 192 KiB ring segment is latency bound -- the grid was 4 CTAs
 * of 128 threads for the entire GPU. Each row's ring offset is a closed form of the row index, so
 * the CTAs need no ordering; only the (sequence, 0) CTA publishes the ring head. */
__global__ __launch_bounds__(threads) void dloop_ctx_append_dense(DeviceArgs a) {
    const auto& c=a.c;const int s=blockIdx.x,f=c.context_features_dim,cw=c.context_window;
    const int slot=static_cast<const int*>(a.t[2])[s];
    const int reset=static_cast<const int*>(a.t[3])[s];
    if(slot<0||slot>=c.max_seqs||(reset!=0&&reset!=1))return;
    // Duplicate slots are invalid: only their first owner may touch the ring.
    for(int prior=0;prior<s;++prior)if(static_cast<const int*>(a.t[2])[prior]==slot)return;
    int* heads=static_cast<int*>(a.t[5]);int* lens=static_cast<int*>(a.t[6]);
    if(!reset&&(heads[slot]<0||heads[slot]>=cw||lens[slot]<0||lens[slot]>cw))return;
    int rows[16],count_rows=0;
    bool ended=false;
    for(int i=0;i<16;++i) {
        const int row=static_cast<const int*>(a.t[1])[s*16+i];
        if(row==-1)ended=true;
        else if(ended||row<0||row>=c.tap_rows)return;
        else rows[count_rows++]=row;
    }
    const bool split=a.t[7]!=nullptr,vec4=(f%8)==0;
    const int width=split?c.target_width:f;
    const int first=reset?0:heads[slot];
    const int tap=int(blockIdx.y);
    if(tap<count_rows) {
        const int row=rows[tap];
        auto* dst=static_cast<uint16_t*>(a.t[4])+(int64_t(slot)*cw+(first+tap)%cw)*f;
        if(vec4) {
            const int quads=f/8;
            for(int q=threadIdx.x;q<quads;q+=threads) {
                uint4 v;
                if(!split) {
                    v=*reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.t[0])+int64_t(row)*f+q*8);
                }else {
                    auto* out=reinterpret_cast<uint16_t*>(&v);
                    #pragma unroll
                    for(int j=0;j<8;++j) {
                        const int d=q*8+j,plane=d>=width?(d<2*width?7:8):0;
                        out[j]=static_cast<const uint16_t*>(a.t[plane])[int64_t(row)*width+d%width];
                    }
                }
                *reinterpret_cast<uint4*>(dst+q*8)=v;
            }
        }else {
            for(int d=threadIdx.x;d<f;d+=threads) {
                const int plane=split&&d>=width?(d<2*width?7:8):0;
                dst[d]=static_cast<const uint16_t*>(a.t[plane])[int64_t(row)*width+(split?d%width:d)];
            }
        }
    }
    if(tap==0&&threadIdx.x==0) {
        const int grown=count_rows;
        heads[slot]=(first+grown)%cw;lens[slot]=reset?grown:((lens[slot]+grown<cw)?lens[slot]+grown:cw);
    }
}
__global__ __launch_bounds__(threads) void dloop_initialize(DeviceArgs a) {
    if(threadIdx.x!=0)return;
    const auto& c=a.c;const int s=blockIdx.x;Work w=workspace(a);
    const int si=a.op==RAD_DLOOP_BASE?3:2,li=a.op==RAD_DLOOP_BASE?2:1;
    const int slot=static_cast<const int*>(a.t[si])[s];
    bool valid=slot>=0&&slot<c.max_seqs;
    if(valid) {
        const int length=static_cast<const int*>(a.t[li])[slot];valid=length>=1&&length<=c.context_window;
        if(a.op==RAD_DLOOP_BASE) {
            const int head=static_cast<const int*>(a.t[1])[slot];valid=valid&&head>=0&&head<c.context_window;
        }
    }
    w.valid[s]=valid?1:0;w.count[s]=valid?1:0;w.frontier[s*max_frontier]=0;
    const int root=s*max_nodes;w.node_score[root]=0;w.tok[root]=-1;w.parent[root]=-1;w.depth[root]=0;
    for(int i=0;i<c.proposal_length;++i)w.path[root*c.proposal_length+i]=-1;
}
__global__ __launch_bounds__(threads) void dloop_context_norm(DeviceArgs a) {
    const int s=blockIdx.x;const auto& c=a.c;
    float* ctx=static_cast<float*>(a.t[7])+int64_t(s)*c.context_window*c.hidden_size;
    rms(ctx,weight(a,DL_CONTEXT_NORM),ctx,c.context_window,c.hidden_size,c.eps);
}
__global__ __launch_bounds__(threads) void dloop_context_rope(DeviceArgs a,int layer) {
    const auto& c=a.c;const int s=blockIdx.x,kv=c.num_kv_heads*c.head_dim,dh=c.head_dim;
    if(!seq_valid(a,s))return;
    float* cache=workspace(a).ctxkv+(int64_t(layer)*c.n_seq+s)*c.context_window*2*kv;
    for(int i=threadIdx.x;i<c.context_window*(kv/2);i+=threads) {
        const int row=i/(kv/2),head=(i/(dh/2))%c.num_kv_heads,d=i%(dh/2),at=row*2*kv+head*dh+d;
        const float angle=float(row)*powf(c.rope_theta,-2.0f*float(d)/float(dh));
        const float co=cosf(angle),sn=sinf(angle),lo=cache[at],hi=cache[at+dh/2];
        cache[at]=lo*co-hi*sn;cache[at+dh/2]=hi*co+lo*sn;
    }
}
__global__ __launch_bounds__(threads) void dloop_initial_norm(DeviceArgs a,int front,int slot) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);
    float* block=w.block+int64_t(job)*b*h;
    for(int i=threadIdx.x;i<b*h;i+=threads)if(i/h>slot)block[i]=bf(weight(a,DL_MASK_TOKEN)[i%h]);
    __syncthreads();rms(block,weight(a,DL_INPUT_NORM,0),w.norm+int64_t(job)*b*h,b,h,c.eps);
}
__global__ __launch_bounds__(threads) void dloop_conv_pre(DeviceArgs a,int front,int layer,int mlp) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);
    convolve(w.norm+int64_t(job)*b*h,w.delta+int64_t(job)*b*4*(h/c.conv_group_size),
        weight(a,mlp?DL_MLP_BASE:DL_ATTN_BASE,layer),w.pre+int64_t(job)*b*h,b,h,c.conv_group_size,0);
}
__global__ __launch_bounds__(threads) void dloop_attention(DeviceArgs a,int front,int layer) {
    __shared__ float probability[9*12*41];
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);
    const int kv=c.num_kv_heads*c.head_dim,dh=c.head_dim,pitch=h+2*kv,mem=c.context_window+b;
    float* qkv=w.qkv+int64_t(job)*b*pitch;
    const float* cache=w.ctxkv+(int64_t(layer)*c.n_seq+s)*c.context_window*2*kv;
    const int seqslot=static_cast<const int*>(a.t[a.op==RAD_DLOOP_BASE?3:2])[s];
    const int length=static_cast<const int*>(a.t[a.op==RAD_DLOOP_BASE?2:1])[seqslot];
    for(int i=threadIdx.x;i<b*((h+kv)/2);i+=threads) {
        const int row=i/((h+kv)/2),channel=i%((h+kv)/2);
        const bool key=channel>=h/2;const int rel=key?channel-h/2:channel;
        const int head=rel/(dh/2),d=rel%(dh/2),at=row*pitch+(key?h:0)+head*dh+d;
        const float angle=float(c.context_window+row)*powf(c.rope_theta,-2.0f*float(d)/float(dh));
        const float co=cosf(angle),sn=sinf(angle),lo=qkv[at],hi=qkv[at+dh/2];
        qkv[at]=lo*co-hi*sn;qkv[at+dh/2]=hi*co+lo*sn;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<b*c.num_q_heads*mem;i+=threads) {
        const int key=i%mem,qh=(i/mem)%c.num_q_heads,row=i/(mem*c.num_q_heads),kh=qh/(c.num_q_heads/c.num_kv_heads);
        float sum=-INFINITY;
        if(key>=c.context_window-length) {
            sum=0;
            for(int d=0;d<dh;++d) {
                const float kval=key<c.context_window?cache[key*2*kv+kh*dh+d]:qkv[(key-c.context_window)*pitch+h+kh*dh+d];
                sum+=qkv[row*pitch+qh*dh+d]*kval;
            }
            sum/=sqrtf(float(dh));
        }
        probability[i]=sum;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<b*c.num_q_heads;i+=threads) {
        float mx=-INFINITY,sum=0;for(int j=0;j<mem;++j)mx=fmaxf(mx,probability[i*mem+j]);
        for(int j=0;j<mem;++j){const float e=expf(probability[i*mem+j]-mx);probability[i*mem+j]=e;sum+=e;}
        for(int j=0;j<mem;++j)probability[i*mem+j]/=sum;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<b*h;i+=threads) {
        const int row=i/h,qh=(i%h)/dh,d=i%dh,kh=qh/(c.num_q_heads/c.num_kv_heads);float sum=0;
        for(int j=0;j<mem;++j) {
            const float v=j<c.context_window?cache[j*2*kv+kv+kh*dh+d]:qkv[(j-c.context_window)*pitch+h+kv+kh*dh+d];
            sum+=probability[(row*c.num_q_heads+qh)*mem+j]*v;
        }
        w.norm[int64_t(job)*b*h+i]=sum;
    }
}
// Stage 0: attention finish/residual/post norm. Stage 1: MLP finish/residual,
// then the next layer's input norm, or just the required final output rows.
__global__ __launch_bounds__(threads) void dloop_finish_norm(DeviceArgs a,int front,int layer,int stage,int slot) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);const int64_t off=int64_t(job)*b*h;
    convolve(w.out+off,w.delta+int64_t(job)*b*4*(h/c.conv_group_size),
        weight(a,stage?DL_MLP_BASE:DL_ATTN_BASE,layer),w.pre+off,b,h,c.conv_group_size,1);
    for(int i=threadIdx.x;i<b*h;i+=threads)w.block[off+i]+=w.pre[off+i];
    __syncthreads();
    if(!stage)rms(w.block+off,weight(a,DL_POST_NORM,layer),w.norm+off,b,h,c.eps);
    else if(layer+1<c.num_layers)rms(w.block+off,weight(a,DL_INPUT_NORM,layer+1),w.norm+off,b,h,c.eps);
    else if(a.op==RAD_DLOOP_BASE)rms(w.block+off+h,weight(a,DL_FINAL_NORM),w.final+int64_t(job)*c.proposal_length*h,c.proposal_length,h,c.eps);
    else rms(w.block+off+(slot+1)*h,weight(a,DL_FINAL_NORM),w.final+int64_t(job)*h,1,h,c.eps);
}

/* Monotone rank key: a bigger key is the better candidate -- the higher score, and at equal scores
 * the LOWER token id, which is exactly the reference's `better`. Every key is unique because the
 * token id is, so a zero key marks an empty slot and ranks below every real candidate. */
__device__ unsigned long long rank_key(float v,int token) {
    const uint32_t bits=__float_as_uint(v),order=bits&0x80000000u?~bits:bits|0x80000000u;
    return (uint64_t(order)<<32)|uint64_t(~uint32_t(token));
}
__device__ float rank_value(unsigned long long k) {
    const uint32_t order=uint32_t(k>>32);
    return __uint_as_float(order&0x80000000u?order&0x7FFFFFFFu:~order);
}
__device__ int rank_id(unsigned long long k) {return int(~uint32_t(k));}
/* Fused LM-head pass: one CTA computes the shard logits of `ht_tile` consecutive vocabulary columns
 * for EVERY batch row, so each BF16 head row is streamed from DRAM exactly once, with 16-byte
 * coalesced loads, and the CTA emits only the top-64 candidate keys of each row. That replaces the
 * old [rows x shard] FP32 logits round trip and its separate 32-CTA rescan, which re-read every
 * logit once and was purely latency bound.
 *
 * Geometry: a warp owns eight rows (rr = 8*wave+rr) and four columns per lane (c = lane+32*j), so
 * one LDS weight quad feeds 32 FMAs and one weight element is read from LDS once per eight rows it
 * multiplies. The activation tile (32 rows x ht_kc) and the weight tile (ht_tile x ht_kc) are
 * staged in FP32 (BF16 converted once per element at staging, not once per row), pitch ht_pad = 36
 * floats, which keeps every staged store and every staged float4 read bank-conflict free: a staged
 * row is 36 words, so lanes 0..7 of a phase land on words 4L+const, covering all 32 banks.
 *
 * Top-k: each lane sorts its four keys (descending rank key, so equal scores prefer the LOWER token
 * id, exactly the reference's `better`), then the warp extracts its 64 winners with repeated warp
 * argmax -- the winner rotates its list and lane 0 records the round, so the emitted list is the
 * exact top-64 of the CTA's candidates in rank order. Keys are unique, so no tie can split a pair
 * of equal keys across lists.
 *
 * Rows are processed in chunks of 32 (grid.y); the key buffer is row-major over ALL rows, so the
 * per-row list count is uniform and the buffer needs exactly `rows*shard*4` bytes -- the size of
 * the FP32 logits it replaces. */
constexpr int ht_tile=128, ht_kc=32, ht_pad=36;
struct HeadArgs {
    const float* y;
    const uint16_t* w;
    unsigned long long* keys;
    int* ids;                   /* local_ids / pool_ids output */
    float* logits;              /* local_logits / unary output */
    int in,shard,active,lists,rows;
};
__device__ __forceinline__ void bf8(const uint4 r,float4& lo,float4& hi) {
    lo=make_float4(__uint_as_float(r.x<<16),__uint_as_float(r.x&0xffff0000u),
                   __uint_as_float(r.y<<16),__uint_as_float(r.y&0xffff0000u));
    hi=make_float4(__uint_as_float(r.z<<16),__uint_as_float(r.z&0xffff0000u),
                   __uint_as_float(r.w<<16),__uint_as_float(r.w&0xffff0000u));
}
/* DECODE serves the single-sequence decode shape (rows <= 8): every wave owns 32 columns of all 8
 * rows instead of 8 rows of all 128 columns, so no wave multiplies the zero-padded rows 8..31. Each
 * (row, column) keeps the same ascending-k fmaf chain and the same top-k extraction, so the keys are
 * bit-identical to the general mapping. */
template<bool DECODE>
__global__ __launch_bounds__(threads) void dloop_head_topk(DeviceArgs a,HeadArgs h) {
    /* DECODE stages the weight tile as raw BF16 (two per word), so a 64-wide K chunk costs less LDS
     * than the general path's 32-wide FP32 chunk and each row is read in 128-byte segments. */
    constexpr int KC=DECODE?64:ht_kc,P=DECODE?KC+4:ht_pad,WP=DECODE?KC/2+4:ht_pad,T=ht_tile,Q=KC/8;   /* 128 measured slower */
    __shared__ float sw[T*WP];
    __shared__ float sy[(DECODE?8:32)*P];
    const auto& c=a.c;
    const int lane=threadIdx.x%32,wave=threadIdx.x/32,cta=blockIdx.x;
    const int r0=blockIdx.y*32,lo=cta*T,glo=c.vocab_lo+lo;
    float acc[8][4];
    #pragma unroll
    for(int rr=0;rr<8;++rr)for(int j=0;j<4;++j)acc[rr][j]=0;
    for(int k0=0;k0<h.in;k0+=KC) {
        for(int i=threadIdx.x;i<T*Q;i+=threads) {
            const int col=i/Q,q=(i%Q)*8;
            uint4 r{0,0,0,0};
            if(glo+col<h.active&&k0+q+8<=h.in)r=*reinterpret_cast<const uint4*>(h.w+int64_t(lo+col)*h.in+k0+q);
            if constexpr(DECODE) {
                *reinterpret_cast<uint4*>(sw+col*WP+q/2)=r;
            } else {
                float4 x,z;bf8(r,x,z);
                *reinterpret_cast<float4*>(sw+col*P+q)=x;
                *reinterpret_cast<float4*>(sw+col*P+q+4)=z;
            }
        }
        for(int i=threadIdx.x;i<(DECODE?8:32)*(KC/4);i+=threads) {
            const int r=i/(KC/4),j=(i%(KC/4))*4;
            float4 v={0,0,0,0};
            if(r0+r<h.rows&&k0+j+4<=h.in)v=*reinterpret_cast<const float4*>(h.y+int64_t(r0+r)*h.in+k0+j);
            *reinterpret_cast<float4*>(sy+r*P+j)=v;
        }
        __syncthreads();
        if constexpr(DECODE) {
            #pragma unroll
            for(int q=0;q<KC;q+=8) {
                const int col=32*wave+lane;
                float4 w0,w1;bf8(*reinterpret_cast<const uint4*>(sw+col*WP+q/2),w0,w1);
                #pragma unroll
                for(int rr=0;rr<8;++rr) {
                    const float4 y0=*reinterpret_cast<const float4*>(sy+rr*P+q);
                    const float4 y1=*reinterpret_cast<const float4*>(sy+rr*P+q+4);
                    float& s=acc[rr][0];
                    s=fmaf(y0.x,w0.x,s);s=fmaf(y0.y,w0.y,s);s=fmaf(y0.z,w0.z,s);s=fmaf(y0.w,w0.w,s);
                    s=fmaf(y1.x,w1.x,s);s=fmaf(y1.y,w1.y,s);s=fmaf(y1.z,w1.z,s);s=fmaf(y1.w,w1.w,s);
                }
            }
        } else {
        #pragma unroll
        for(int q=0;q<KC;q+=8) {
            float4 yv[8][2],wv[4][2];
            #pragma unroll
            for(int rr=0;rr<8;++rr) {
                yv[rr][0]=*reinterpret_cast<const float4*>(sy+(wave*8+rr)*P+q);
                yv[rr][1]=*reinterpret_cast<const float4*>(sy+(wave*8+rr)*P+q+4);
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                const int c=lane+32*j;
                wv[j][0]=*reinterpret_cast<const float4*>(sw+c*P+q);
                wv[j][1]=*reinterpret_cast<const float4*>(sw+c*P+q+4);
            }
            #pragma unroll
            for(int rr=0;rr<8;++rr) {
                #pragma unroll
                for(int j=0;j<4;++j) {
                    acc[rr][j]=fmaf(yv[rr][0].x,wv[j][0].x,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][0].y,wv[j][0].y,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][0].z,wv[j][0].z,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][0].w,wv[j][0].w,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][1].x,wv[j][1].x,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][1].y,wv[j][1].y,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][1].z,wv[j][1].z,acc[rr][j]);
                    acc[rr][j]=fmaf(yv[rr][1].w,wv[j][1].w,acc[rr][j]);
                }
            }
        }
        }
        __syncthreads();
    }
    /* DECODE regathers through LDS so each wave holds two rows at the general mapping's columns. */
    constexpr int ROWS=DECODE?2:8;
    if constexpr(DECODE) {
        #pragma unroll
        for(int rr=0;rr<8;++rr)sw[rr*T+32*wave+lane]=acc[rr][0];
        __syncthreads();
        #pragma unroll
        for(int rr=0;rr<2;++rr)for(int j=0;j<4;++j)acc[rr][j]=sw[(2*wave+rr)*T+lane+32*j];
    }
    const int rbase=DECODE?r0+2*wave:r0+8*wave;
    unsigned long long kk[8][4];
    #pragma unroll
    for(int rr=0;rr<ROWS;++rr) {
        const int row=rbase+rr;
        const bool ok=row<h.rows&&seq_valid(a,row/a.c.proposal_length);
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int tok=glo+lane+32*j;float v=acc[rr][j];
            kk[rr][j]=ok&&tok<h.active&&isfinite(v)?rank_key(v==0.0f?0.0f:v,tok):0;
        }
        // descending 4-element sort: (0,1),(2,3),(0,2),(1,3),(1,2)
        if(kk[rr][1]>kk[rr][0]){const unsigned long long t=kk[rr][0];kk[rr][0]=kk[rr][1];kk[rr][1]=t;}
        if(kk[rr][3]>kk[rr][2]){const unsigned long long t=kk[rr][2];kk[rr][2]=kk[rr][3];kk[rr][3]=t;}
        if(kk[rr][2]>kk[rr][0]){const unsigned long long t=kk[rr][0];kk[rr][0]=kk[rr][2];kk[rr][2]=t;}
        if(kk[rr][3]>kk[rr][1]){const unsigned long long t=kk[rr][1];kk[rr][1]=kk[rr][3];kk[rr][3]=t;}
        if(kk[rr][2]>kk[rr][1]){const unsigned long long t=kk[rr][1];kk[rr][1]=kk[rr][2];kk[rr][2]=t;}
    }
    /* CTA merge: for each row the owning warp extracts its 64 winners with a warp-uniform repeated
     * argmax -- every lane ends each round holding the same winning key, so the two entries of that
     * round (i and i+32) are written straight to global by lanes `i%32`, no LDS staging and no
     * barrier. Keys are unique, so argmax order is the total order. */
    #pragma unroll
    for(int rr=0;rr<ROWS;++rr) {
        const int row=rbase+rr;
        if(row>=h.rows)continue;
        unsigned long long* dst=h.keys+(int64_t(row)*h.lists+cta)*64;
        unsigned long long t0=kk[rr][0],t1=kk[rr][1],t2=kk[rr][2],t3=kk[rr][3];
        for(int i=0;i<64;++i) {
            unsigned long long best=t0;
            #pragma unroll
            for(int o=16;o;o>>=1) {
                const unsigned long long other=__shfl_xor(best,o);
                if(other>best)best=other;
            }
            if(t0==best){t0=t1;t1=t2;t2=t3;t3=0;}
            if(i==lane||i==lane+32)dst[i]=best;
        }
    }
}
/* Exact shard top-k from the per-CTA candidate lists: ONE k-way tournament over the list heads.
 *
 * Every list `dloop_head_topk` writes is ALREADY sorted descending, and that is what makes the
 * merge cheap: the global top-64 of the union is found by repeatedly taking the largest remaining
 * head, so the merge never examines a key that is not a winner. Each lane owns eight lists
 * (threadIdx.x + 128*j covers up to 1024 >= lists), keeps their eight heads in registers, and
 * offers the largest of them; the warp butterfly picks the wave-wide largest, the winner advances
 * the cursor of the list it came from, and lane 0 records the round. After 64 rounds the four wave
 * lists hold their top-64 in descending order, and wave 0 merges those four the same way.
 *
 * The previous kernel re-read all `lists*64` keys per row into a 64-entry register list kept sorted
 * by a compare-and-shift chain: the same answer for 64x the work, and 256 VGPRs with 14 spills
 * (the list has no compile-time index once the insert position is dynamic). The candidate multiset
 * is identical, keys are unique so every comparison is decisive, and the emitted (ids, logits)
 * pairs come from the same code as before -> bit-identical output. */
__global__ __launch_bounds__(threads) void dloop_topk_merge(DeviceArgs a,const unsigned long long* keys,HeadArgs h) {
    __shared__ unsigned long long wlist[threads/32][64];
    const auto& c=a.c;const int row=blockIdx.x,lane=threadIdx.x%32,wave=threadIdx.x/32;
    const int lists=h.lists;
    const unsigned long long* base=keys+int64_t(row)*lists*64;
    constexpr int PER=8;
    unsigned long long head[PER];
    int hcur[PER];
    #pragma unroll
    for(int j=0;j<PER;++j) {
        const int l=threadIdx.x+threads*j;
        hcur[j]=0;
        head[j]=l<lists?base[int64_t(l)*64]:0;
    }
    for(int r=0;r<64;++r) {
        unsigned long long best=head[0];
        #pragma unroll
        for(int j=1;j<PER;++j)if(head[j]>best)best=head[j];
        #pragma unroll
        for(int o=16;o;o>>=1) {
            const unsigned long long other=__shfl_xor(best,o);
            if(other>best)best=other;
        }
        #pragma unroll
        for(int j=0;j<PER;++j)if(head[j]==best) {
            const int l=threadIdx.x+threads*j;
            if(l<lists&&hcur[j]<63) {
                ++hcur[j];
                head[j]=base[int64_t(l)*64+hcur[j]];
            }else head[j]=0;
            break;
        }
        if(lane==0)wlist[wave][r]=best;
    }
    __syncthreads();
    if(wave)return;
    const int nlists=threads/32;
    int cursor=lane<nlists?0:64;
    for(int r=0;r<64;++r) {
        const unsigned long long mine=lane<nlists?wlist[lane][cursor]:0;
        unsigned long long best=mine;int owner=lane;
        #pragma unroll
        for(int o=16;o;o>>=1) {
            const unsigned long long other=__shfl_xor(best,o);const int ol=__shfl_xor(owner,o);
            if(other>best||(other==best&&ol<owner)){best=other;owner=ol;}
        }
        if(owner==lane)++cursor;
        if(lane==0) {
            h.ids[int64_t(row)*c.pool_size+r]=best?rank_id(best):-1;
            h.logits[int64_t(row)*c.pool_size+r]=best?rank_value(best):-INFINITY;
        }
    }
}
/* Merge the two ranks' packets into the token-sorted pool and gather the owned teacher/codebook rows.
 *
 * The candidate set is world_size*S*M*K = 4096 entries per row and the pool keeps the best K=64. One
 * CTA per row: each of the 128 threads reads exactly ONE candidate (rank r, slot j), so the whole
 * packet is consumed in a single coalesced pass; a bitonic sort over the 128 rank keys puts the
 * candidates in the reference's exact order, and a single walk takes the best 64 DISTINCT token ids
 * (a repeated id is inserted once by the reference's `insert`, and only as a repeated same-score
 * packet). The 64 winners are then sorted ascending by token id, which is the order the reference
 * emits. Rebuilding the score from the key keeps the emitted value bit-identical to the candidate's
 * float, and the guard `n<K` reproduces the reference's `ids[K-1]<0` rejection.
 *
 * This replaces a single-thread O(K^2) rescan of all 4096 entries per row (a serial dependency chain
 * on one lane of each of 32 CTAs), which cost 0.9-1.3 ms per call. */
__device__ void bitonic_asc(unsigned long long* x,int n) {
    for(int k=2;k<=n;k<<=1)for(int j=k>>1;j;j>>=1) {
        const int i=threadIdx.x;
        if(i<n) {
            const int l=i^j;
            if(l>i) {
                const unsigned long long A=x[i],B=x[l];
                if((i&k)==0?(A>B):(A<B)){x[i]=B;x[l]=A;}
            }
        }
        __syncthreads();
    }
}
__global__ __launch_bounds__(threads) void dloop_pool_dense(DeviceArgs a) {
    constexpr int K=64,ENTRIES=2*K;
    __shared__ unsigned long long cand[ENTRIES],pick[K],order[K];
    __shared__ int selected[K];__shared__ float values[K];__shared__ int valid;
    const auto& c=a.c;const int row=blockIdx.x,s=row/c.proposal_length;
    const auto* ids=static_cast<const int*>(a.t[0]);const auto* logits=static_cast<const float*>(a.t[1]);
    if(threadIdx.x==0) {
        const int anchor=static_cast<const int*>(a.t[2])[s];valid=anchor>=0&&anchor<c.active_vocab;
    }
    if(threadIdx.x<K)order[threadIdx.x]=threadIdx.x;
    if(threadIdx.x<ENTRIES) {
        const int rank=threadIdx.x/K,j=threadIdx.x%K;
        const int at=(rank*c.n_seq*c.proposal_length+row)*c.pool_size+j;
        const int token=ids[at];const float v=logits[at];
        // Ids outside the active vocabulary and non-finite scores are dropped (the reference rejects
        // such a packet; the device guard skips the entry and lets the K-count decide validity).
        cand[threadIdx.x]=token>=0&&token<c.active_vocab&&isfinite(v)?rank_key(v==0.0f?0.0f:v,token):0;
    }
    __syncthreads();
    bitonic_asc(cand,ENTRIES);          // ascending: empty (0) keys first, the best candidate last
    if(threadIdx.x==0) {
        int n=0,last=-1;
        for(int i=ENTRIES-1;i>=0&&n<K;--i) {
            const unsigned long long kk=cand[i];
            if(!kk)break;               // every remaining entry is empty
            const int token=rank_id(kk);
            if(n&&token==last)continue; // repeated same-score packet: the reference inserts it once
            pick[n]=kk;order[n]=uint64_t(uint32_t(token))<<32|uint64_t(n);last=token;++n;
        }
        if(n<K)valid=0;                 // fewer than K distinct candidates: the reference rejects it
    }
    __syncthreads();
    bitonic_asc(order,K);               // ascending by token id == the reference's insertion sort
    if(threadIdx.x<K) {
        if(valid) {
            const uint64_t o=order[threadIdx.x];
            selected[threadIdx.x]=int(o>>32);
            values[threadIdx.x]=rank_value(pick[o&uint64_t(K-1)]);
        } else {selected[threadIdx.x]=-1;values[threadIdx.x]=-INFINITY;}
        static_cast<int*>(a.t[7])[row*K+threadIdx.x]=selected[threadIdx.x];
        static_cast<float*>(a.t[8])[row*K+threadIdx.x]=values[threadIdx.x];
    }
    __syncthreads();
    /* Gather the owned teacher/codebook rows. Both widths (target 2560/32, selector rank 256/8) are
     * multiples of 8, so a row is copied in aligned 16-byte `uint4` blocks; the previous per-element
     * loop wasted 8x of every transaction on 2-byte accesses and dominated the call (0.64 ms). */
    for(int i=threadIdx.x;i<c.pool_size*(c.target_width/8);i+=threads) {
        const int k=i/(c.target_width/8),q=(i%(c.target_width/8))*8;
        const int64_t off=int64_t(row)*c.pool_size*c.target_width+int64_t(k)*c.target_width+q;
        /* Ownership is per row start, exactly like the reference's memcpy: a row that BEGINS inside
         * this shard is copied whole, and the width is a multiple of 8 so the blocks always cover it. */
        const int local=selected[k]-c.vocab_lo;const bool own=valid&&local>=0&&local<c.vocab_size_shard;
        uint4 h4{0,0,0,0},e4{0,0,0,0};
        if(own) {
            h4=*reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.t[3])+int64_t(local)*c.target_width+q);
            e4=*reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.t[4])+int64_t(local)*c.target_width+q);
        }
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(a.t[9])+off)=h4;
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(a.t[10])+off)=e4;
    }
    for(int i=threadIdx.x;i<c.pool_size*(c.selector_rank/8);i+=threads) {
        const int k=i/(c.selector_rank/8),q=(i%(c.selector_rank/8))*8;
        const int64_t off=int64_t(row)*c.pool_size*c.selector_rank+int64_t(k)*c.selector_rank+q;
        const int local=selected[k]-c.vocab_lo;const bool own=valid&&local>=0&&local<c.vocab_size_shard;
        uint4 p4{0,0,0,0},s4{0,0,0,0};
        if(own) {
            p4=*reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.t[5])+int64_t(local)*c.selector_rank+q);
            s4=*reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(a.t[6])+int64_t(local)*c.selector_rank+q);
        }
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(a.t[11])+off)=p4;
        *reinterpret_cast<uint4*>(static_cast<uint16_t*>(a.t[12])+off)=s4;
    }
    if(row%c.proposal_length==0)for(int d=threadIdx.x;d<c.selector_rank;d+=threads) {
        const int64_t local=int64_t(static_cast<const int*>(a.t[2])[s])-c.vocab_lo;
        static_cast<uint16_t*>(a.t[13])[s*c.selector_rank+d]=valid&&local>=0&&local<c.vocab_size_shard?static_cast<const uint16_t*>(a.t[5])[int64_t(local)*c.selector_rank+d]:0;
    }
}
bool supported(const RadDLoopConfig& c,bool tiny) {
    if(c.world_size!=2||c.num_layers!=2||c.proposal_length!=8||c.context_window!=32||
       c.conv_kernel_size!=2||c.conv_group_size!=16||c.pool_size!=64||c.width>12||
       c.vocab_size_shard*2!=c.vocab_size||(c.vocab_lo!=0&&c.vocab_lo!=c.vocab_size_shard))return false;
    if(tiny)return c.hidden_size==64&&c.intermediate_size==128&&c.num_q_heads==4&&c.num_kv_heads==2&&
        c.head_dim==16&&c.context_features_dim==48&&c.target_width==32&&c.selector_rank==8&&c.vocab_size==512&&c.active_vocab==509;
    return c.hidden_size==768&&c.intermediate_size==2304&&c.num_q_heads==12&&c.num_kv_heads==4&&
        c.head_dim==64&&c.context_features_dim==7680&&c.target_width==2560&&c.selector_rank==256&&c.vocab_size==248320&&c.active_vocab==248077;
}
int64_t scratch_bytes(const RadArgs* a,int op) {
    RadDLoopConfig c{};if(!a||rad_dloop_config(a->p,a->n_p,&c)!=RAD_OK)return 0;
    if(op!=RAD_DLOOP_BASE&&op!=RAD_DLOOP_DARTREE)return 0;
    return 4*(ring_work_elements(c,op)+(op==RAD_DLOOP_BASE?int64_t(c.n_seq)*c.proposal_length*c.vocab_size_shard:0));
}
/* Kernel choice by output width: wider column blocks amortize the staged activation tile, so wide
 * matrices (the head pass, gate/up) take 16 columns per lane while narrow ones keep a finer column
 * split for CTA count. LDS is 13.5/22.5/40.5 KiB for C=4/8/16 (doubled for gate), inside 64 KiB. */
constexpr int tiled_cols(int out) {return out>=16384?16:(out>=1024?8:4);}
inline const void* tiled_kernel(int C,bool gate) {
    switch(C) {
    case 16:return gate?reinterpret_cast<const void*>(dloop_matmul_tiled<16,true>):reinterpret_cast<const void*>(dloop_matmul_tiled<16,false>);
    case 8:return gate?reinterpret_cast<const void*>(dloop_matmul_tiled<8,true>):reinterpret_cast<const void*>(dloop_matmul_tiled<8,false>);
    default:return gate?reinterpret_cast<const void*>(dloop_matmul_tiled<4,true>):reinterpret_cast<const void*>(dloop_matmul_tiled<4,false>);
    }
}

struct Launcher {
    hipStream_t stream;
    int send(const void* kernel,dim3 grid,void** args)const {
        return hipLaunchKernel(kernel,grid,dim3(threads),args,0,stream)==hipSuccess?RAD_OK:RAD_E_DEVICE;
    }
    int simple(const void* kernel,DeviceArgs& a,int blocks)const {void* args[]={&a};return send(kernel,dim3(blocks),args);}
    int matrix(MatArgs m)const {
        const int C=tiled_cols(m.out),BN=(threads/32)*C,blocks=(m.out+BN-1)/BN;
        void* args[]={&m};
        return send(tiled_kernel(C,m.kind==3),dim3(blocks,(m.rows+mt_rows-1)/mt_rows),args);
    }
};
MatArgs matrix_args(const DeviceArgs& a,const float* x,const uint16_t* weights,float* y,
                    int rows,int in,int out,int perseq) {
    MatArgs m{};m.x=x;m.w0=weights;m.y=y;m.rows=rows;m.in=in;m.out=out;m.xstride=in;
    m.rows_per_seq=perseq;m.valid=workspace(a).valid;return m;
}
int context_cache(DeviceArgs& a,const Launcher& launch) {
    const auto& c=a.c;Work w=workspace(a);const int kv=c.num_kv_heads*c.head_dim;
    const float* ctx=static_cast<const float*>(a.t[a.op==RAD_DLOOP_BASE?7:0]);
    for(int layer=0;layer<c.num_layers;++layer) {
        MatArgs m=matrix_args(a,ctx,weight(a,DL_K_PROJ,layer),w.ctxkv+int64_t(layer)*c.n_seq*c.context_window*2*kv,
            c.n_seq*c.context_window,c.hidden_size,2*kv,c.context_window);
        m.kind=2;m.split=kv;m.w1=weight(a,DL_V_PROJ,layer);
        int rc=launch.matrix(m);if(rc)return rc;
        void* argv[]={&a,&layer};rc=launch.send(reinterpret_cast<const void*>(dloop_context_rope),dim3(c.n_seq),argv);if(rc)return rc;
    }
    return RAD_OK;
}
int backbone_batch(DeviceArgs& a,const Launcher& launch,int front,int slot) {
    const auto& c=a.c;Work w=workspace(a);const int b=c.proposal_length+1,h=c.hidden_size,kv=c.num_kv_heads*c.head_dim;
    const int jobs=c.n_seq*front,rows=jobs*b;
    MatArgs embed=matrix_args(a,nullptr,weight(a,DL_TOKEN_PROJ),w.block,rows,c.target_width,h,front*b);
    embed.source=2;embed.anchor=static_cast<const uint16_t*>(a.t[a.op==RAD_DLOOP_BASE?4:3]);
    embed.embed=a.op==RAD_DLOOP_DARTREE?static_cast<const uint16_t*>(a.t[9]):nullptr;
    embed.path=w.path;embed.frontier=w.frontier;embed.front=front;embed.slot=slot;embed.m=c.proposal_length;embed.k=c.pool_size;
    int rc=launch.matrix(embed);if(rc)return rc;
    void* initial[]={&a,&front,&slot};rc=launch.send(reinterpret_cast<const void*>(dloop_initial_norm),dim3(jobs),initial);if(rc)return rc;
    for(int layer=0;layer<c.num_layers;++layer) {
        for(int mlp=0;mlp<2;++mlp) {
            MatArgs dyn=matrix_args(a,w.norm,weight(a,mlp?DL_MLP_KERNEL:DL_ATTN_KERNEL,layer),w.delta,rows,h,4*(h/c.conv_group_size),front*b);
            rc=launch.matrix(dyn);if(rc)return rc;
            void* conv[]={&a,&front,&layer,&mlp};rc=launch.send(reinterpret_cast<const void*>(dloop_conv_pre),dim3(jobs),conv);if(rc)return rc;
            if(!mlp) {
                MatArgs qkv=matrix_args(a,w.pre,weight(a,DL_Q_PROJ,layer),w.qkv,rows,h,h+2*kv,front*b);
                qkv.kind=1;qkv.split=h;qkv.w1=weight(a,DL_K_PROJ,layer);qkv.w2=weight(a,DL_V_PROJ,layer);
                rc=launch.matrix(qkv);if(rc)return rc;
                void* attn[]={&a,&front,&layer};rc=launch.send(reinterpret_cast<const void*>(dloop_attention),dim3(jobs),attn);if(rc)return rc;
                rc=launch.matrix(matrix_args(a,w.norm,weight(a,DL_O_PROJ,layer),w.out,rows,h,h,front*b));if(rc)return rc;
            }else {
                MatArgs gate=matrix_args(a,w.pre,weight(a,DL_GATE_PROJ,layer),w.ff,rows,h,c.intermediate_size,front*b);
                gate.kind=3;gate.w1=weight(a,DL_UP_PROJ,layer);rc=launch.matrix(gate);if(rc)return rc;
                rc=launch.matrix(matrix_args(a,w.ff,weight(a,DL_DOWN_PROJ,layer),w.out,rows,c.intermediate_size,h,front*b));if(rc)return rc;
            }
            void* finish[]={&a,&front,&layer,&mlp,&slot};rc=launch.send(reinterpret_cast<const void*>(dloop_finish_norm),dim3(jobs),finish);if(rc)return rc;
        }
    }
    const bool base=a.op==RAD_DLOOP_BASE;
    return launch.matrix(matrix_args(a,w.final,weight(a,DL_OUTPUT_PROJ),base?static_cast<float*>(a.t[8]):w.y,
        base?c.n_seq*c.proposal_length:jobs,h,c.target_width,base?c.proposal_length:front));
}
int run(const RadArgs* args,RadStream stream,int op,bool tiny) {
    DeviceArgs a{};int rc=rad_dloop_args(args,op,&a.c);if(rc)return rc;
    if(!supported(a.c,tiny)||args->world_size!=2||args->rank<0||args->rank>=2||a.c.vocab_lo!=args->rank*a.c.vocab_size_shard)return RAD_E_UNSUPPORTED;
    if(args->scratch_bytes<scratch_bytes(args,op))return RAD_E_SCRATCH;
    for(int i=0;i<args->n_t;++i)a.t[i]=args->t[i].data;
    for(int l=0;l<2;++l)for(int w=0;w<20;++w)a.offset[l][w]=int(rad_dloop_weight_offset(a.c,w,l));
    a.scratch=static_cast<float*>(args->scratch);a.op=op;const auto& c=a.c;
    const Launcher launch{reinterpret_cast<hipStream_t>(stream)};
    void* append_args[]={&a};
    if(op==RAD_DLOOP_CTX_APPEND)return launch.send(reinterpret_cast<const void*>(dloop_ctx_append_dense),dim3(c.n_seq,16),append_args);

    if(op==RAD_DLOOP_POOL)return launch.simple(reinterpret_cast<const void*>(dloop_pool_dense),a,c.n_seq*c.proposal_length);
    rc=launch.simple(reinterpret_cast<const void*>(dloop_initialize),a,c.n_seq);if(rc)return rc;
    if(op==RAD_DLOOP_BASE) {
        MatArgs ctx=matrix_args(a,nullptr,weight(a,DL_CONTEXT_PROJ),static_cast<float*>(a.t[7]),c.n_seq*c.context_window,
            c.context_features_dim,c.hidden_size,c.context_window);
        ctx.source=1;ctx.ring=static_cast<const uint16_t*>(a.t[0]);ctx.heads=static_cast<const int*>(a.t[1]);
        ctx.lengths=static_cast<const int*>(a.t[2]);ctx.slots=static_cast<const int*>(a.t[3]);ctx.cw=c.context_window;ctx.f=c.context_features_dim;
        rc=launch.matrix(ctx);if(rc)return rc;
        rc=launch.simple(reinterpret_cast<const void*>(dloop_context_norm),a,c.n_seq);if(rc)return rc;
        rc=context_cache(a,launch);if(rc)return rc;
        rc=backbone_batch(a,launch,1,0);if(rc)return rc;
        const int rows=c.n_seq*c.proposal_length,lists=(c.vocab_size_shard+ht_tile-1)/ht_tile;
        auto* keys=reinterpret_cast<unsigned long long*>(a.scratch+ring_work_elements(c,op));
        HeadArgs head{static_cast<const float*>(a.t[8]),static_cast<const uint16_t*>(a.t[6]),keys,
            static_cast<int*>(a.t[9]),static_cast<float*>(a.t[10]),
            c.target_width,c.vocab_size_shard,c.active_vocab,lists,rows};
        void* hargs[]={&a,&head};
        rc=launch.send(reinterpret_cast<const void*>(dloop_head_topk<false>),dim3(lists,(rows+31)/32),hargs);if(rc)return rc;
        void* top[]={&a,&keys,&head};return launch.send(reinterpret_cast<const void*>(dloop_topk_merge),dim3(rows),top);
    }
    return RAD_E_UNSUPPORTED;
}
static_assert(sizeof(DeviceArgs)<=768);
static_assert(sizeof(MatArgs)<=768);

/* ====================================================== full-context path (dloop_full.h)
 *
 * The base path keeps the tiled GEMM: it retains each output column's whole weight row in LDS and
 * streams K, which is what the wide projections (F7680, FF2304, the LM head) want. The append's
 * projections are dense and narrow (chunk_rows x H), so the same kernel serves them.
 *
 * Only the launch policy and the entry points live here; every stage sequence, every arena offset and
 * both shared kernels (streaming attention, the block rope) come from dloop_full_device.h, so the
 * tree path uses the SAME source with its own GEMM.
 */
struct FullPolicy {
    hipStream_t stream;
    hipStream_t stream_of() const {return stream;}
    int dyn(const DeviceArgs& a,const GemmReq& r,int layer,int mlp) const {
        if(!skinny_dyn_fits(r,a.c))return RAD_E_UNSUPPORTED;
        return skinny_dyn_launch(r,a,layer,mlp,true,stream);
    }
    /* GEMM request -> the tiled kernel. `kind` picks the weight mapping (0 one matrix, 1 Q|K|V, 2 K|V,
     * 3 gate/up with silu(gate)*up), `source` the activation source: 0 a dense F32 matrix, 2 the
     * anchor/path token embeddings. rows_per_seq drives the per-row validity test the kernel has.
     *
     * Same dispatch as the tree file's policy: the decode shape goes to the lane-per-row kernel in
     * dloop_skinny.hip, everything else (the append's staged chunks, rows past the threshold) keeps
     * the tiled kernel below. The base path's tiled kernel accumulated with explicit fmaf, so the
     * skinny kernel runs with FUSE=true here to reproduce its rounding (see dloop_skinny.hip). */
    int gemm(const GemmReq& r) const {
        if(skinny_fits(r))return skinny_gemm_launch(r,true,stream);
        MatArgs m{};
        m.x=r.x;m.w0=r.w0;m.w1=r.w1;m.w2=r.w2;m.anchor=r.anchor;m.embed=r.embed;m.y=r.y;
        m.valid=r.valid;m.path=r.path;m.frontier=r.frontier;m.live=r.live;
        m.rows=r.rows;m.in=r.in;m.out=r.out;m.xstride=r.xstride;m.rows_per_seq=r.rows_per_seq;
        m.kind=r.kind;m.source=r.source;m.split=r.split;
        m.front=r.front;m.slot=r.slot;m.m=r.m;m.k=r.k;
        const int C=tiled_cols(m.out),BN=(threads/32)*C,blocks=(m.out+BN-1)/BN;
        void* args[]={&m};
        return send(tiled_kernel(C,m.kind==3),dim3(blocks,(m.rows+mt_rows-1)/mt_rows),threads,args,stream);
    }
};
int run_full(const RadArgs* args,RadStream stream,int op,bool tiny) {
    RadDLoopFullConfig f{};
    int rc=rad_dloop_full_args(args,op,&f);if(rc)return rc;
    DeviceArgs a{};a.c=f.model;a.capacity=f.context_capacity;a.op=FULL_APPEND+op;
    if(!full_supported(a,tiny)||args->world_size!=2||args->rank<0||args->rank>=2)return RAD_E_UNSUPPORTED;
    if(a.c.vocab_lo!=args->rank*a.c.vocab_size_shard)return RAD_E_UNSUPPORTED;
    /* Fail closed on both bounds: the device layout is the floor because only the device knows its
     * own staging, and the row's published hook is the other floor because that is what the engine
     * allocated from. The larger wins; a short buffer refuses instead of running truncated. */
    const int64_t hook=op==RAD_DLOOP_FULL_APPEND?rad_dloop_full_append_scratch(args):
        op==RAD_DLOOP_FULL_BASE?rad_dloop_full_base_scratch(args):rad_dloop_full_tree_scratch(args);
    const int64_t need=full_scratch_bytes(a,a.op);
    if(args->scratch_bytes<need||args->scratch_bytes<hook)return RAD_E_SCRATCH;
    for(int i=0;i<args->n_t;++i)a.t[i]=args->t[i].data;
    const int layers=a.c.num_layers<5?a.c.num_layers:5;
    for(int l=0;l<layers;++l)for(int w=0;w<20;++w)a.offset[l][w]=int(rad_dloop_weight_offset(a.c,w,l));
    a.scratch=static_cast<float*>(args->scratch);
    const FullPolicy policy{reinterpret_cast<hipStream_t>(stream)};
    if(a.op==FULL_APPEND)return full_append_run<0>(a,policy);
    void* plain[]={&a};
    rc=send(reinterpret_cast<const void*>(full_initialize<0>),dim3(a.c.n_seq),threads,plain,policy.stream);
    if(rc)return rc;
    rc=full_backbone<0>(a,policy,1,0,1,static_cast<float*>(a.t[6]));if(rc)return rc;
    /* The head pass is the ring base's, unchanged: the same shard top-k over the same y rows, only
     * the operand indices differ (the full base has no ctxh output). */
    Work w=workspace(a);
    const int rows=a.c.n_seq*a.c.proposal_length,lists=int(head_lists(a.c.vocab_size_shard));
    auto* keys=reinterpret_cast<unsigned long long*>(w.keys);
    HeadArgs head{static_cast<const float*>(a.t[6]),static_cast<const uint16_t*>(a.t[5]),keys,
        static_cast<int*>(a.t[7]),static_cast<float*>(a.t[8]),
        a.c.target_width,a.c.vocab_size_shard,a.c.active_vocab,lists,rows};
    void* hargs[]={&a,&head};
    const void* head_kernel=rows<=8?reinterpret_cast<const void*>(dloop_head_topk<true>):
        reinterpret_cast<const void*>(dloop_head_topk<false>);
    rc=send(head_kernel,dim3(lists,(rows+31)/32),threads,hargs,policy.stream);
    if(rc)return rc;
    void* top[]={&a,&keys,&head};
    return send(reinterpret_cast<const void*>(dloop_topk_merge),dim3(rows),threads,top,policy.stream);
}
}   /* anonymous namespace: everything above is this translation unit's own */

/* ==================================================== the full-context entry points
 *
 * OUTSIDE the anonymous namespace: the registration in dloop_full.cpp names these, and the hook has
 * to be resolvable from another object file. They are the only three-argument wrappers in this file;
 * every stage sequence and every arena offset lives in dloop_full_device.h.
 *
 * The hook is the DEVICE layout's requirement. run_full also compares it against the row's published
 * hook (rad_dloop_full_*_scratch) and takes the larger, so an arena sized by either formula is safe
 * and a short one refuses with RAD_E_SCRATCH rather than running truncated. */
int64_t rad_mine_dloop_full_scratch(const RadArgs* args,int op) {
    RadDLoopFullConfig f{};
    if(!args||op<0||op>=RAD_DLOOP_FULL_OPS||rad_dloop_full_config(args->p,args->n_p,&f)!=RAD_OK)return 0;
    DeviceArgs a{};a.c=f.model;a.capacity=f.context_capacity;
    return 4*std::max(full_elements(a,FULL_APPEND+op),
        int64_t(f.model.n_seq)*rad_dloop_full_scratch_elements(f,op));
}
#define FULL_ENTRY(name,op,tiny) \
    int name(const RadArgs* a,RadStream s){return run_full(a,s,op,tiny);}
FULL_ENTRY(rad_mine_dloop_full_append_h768_l5_tp2,RAD_DLOOP_FULL_APPEND,false)
FULL_ENTRY(rad_mine_dloop_full_base_h768_l5_tp2,RAD_DLOOP_FULL_BASE,false)
FULL_ENTRY(rad_mine_dloop_full_append_h64_l5_tp2,RAD_DLOOP_FULL_APPEND,true)
FULL_ENTRY(rad_mine_dloop_full_base_h64_l5_tp2,RAD_DLOOP_FULL_BASE,true)
#undef FULL_ENTRY

extern "C" int64_t rad_mine_dloop_base_scratch(const RadArgs* a){return scratch_bytes(a,RAD_DLOOP_BASE);}
#define ENTRY(name,op,tiny) extern "C" int name(const RadArgs* a,RadStream s){return run(a,s,op,tiny);}
ENTRY(rad_mine_dloop_ctx_append_c32_f7680_tp2,RAD_DLOOP_CTX_APPEND,false)
ENTRY(rad_mine_dloop_base_h768_m8_tp2,RAD_DLOOP_BASE,false)
ENTRY(rad_mine_dloop_pool_h768_k64_tp2,RAD_DLOOP_POOL,false)
ENTRY(rad_mine_dloop_ctx_append_c32_f48_tp2,RAD_DLOOP_CTX_APPEND,true)
ENTRY(rad_mine_dloop_base_h64_m8_tp2,RAD_DLOOP_BASE,true)
ENTRY(rad_mine_dloop_pool_h64_k64_tp2,RAD_DLOOP_POOL,true)
#undef ENTRY
