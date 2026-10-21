/* Dartree device path, split out of dloop_kernels.hip.
 *
 * The tree op is the only DLoop op that walks seven dependent depths, each with
 * S*frontier*(M+1) backbone rows. The generic kernel in dloop_kernels.hip keeps
 * four output columns per CTA in LDS, so every one of its N/4 column CTAs reads
 * the whole activation tile again: at width 12 that is ~5 GB of activation
 * re-reads per layer launch. This file replaces the matrix kernel for the tree
 * op with a register-tiled GEMM (64x128 CTA tile, 32 FP32 accumulators per
 * thread) whose traffic is 2*K*N*ceil(rows/64) weight bytes plus
 * 4*rows*K*ceil(N/128) activation bytes, and keeps every elementwise stage
 * numerically identical.
 *
 * Accumulation order is unchanged: one FP32 accumulator per output, advanced in
 * ascending K one multiply-add at a time, with -ffp-contract=off, exactly like
 * rad_ref_dloop_forward and the generic kernel. Discrete outputs (token/parent/
 * depth arrays and the packed tree) therefore stay bit-exact.
 *
 * The workspace layout below MUST stay identical to arena_elements/workspace in
 * dloop_kernels.hip: rad_mine_dloop_tree_scratch is defined here and has to
 * agree with every buffer offset used by run().
 */
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include "dloop_ops.h"
#include "dloop_full.h"
#include "dloop_full_device.h"
#include "dloop_skinny.h"

namespace {
using namespace rad_dloop;
/* Register-tiled GEMM: CTA tile 64 rows x 64 columns, K chunk 32, 256 threads with a 4x8 thread tile
 * (32 FP32 accumulators). This file owns it because the tree op is the one that walks seven dependent
 * depths, where the generic kernel's N/4 column CTAs would re-read the whole activation tile per
 * column block. */
constexpr int gemm_threads=256, gemm_tm=64, gemm_tn=64, gemm_kc=32;
constexpr int a_pitch=gemm_kc+4;          /* FP32 activation rows, 16B aligned */
constexpr int b_pitch=gemm_tn+8;          /* FP32 weight rows, 16B aligned */
constexpr int gate_tn=64, gate_pitch=gate_tn+16;
struct MatArgs {
    const float* x;
    const uint16_t *w0,*w1,*w2,*anchor,*embed;
    const int *valid,*path,*frontier;
    /* Optional uniform row-tile exit: null for the ring tree and the full base/tree, set only by the
     * full append to the live count of the chunk being staged (see the tiled kernel in dloop_kernels.hip
     * for the full reasoning; live rows form a prefix, so a tile beginning past the count is dead). */
    const int* live;
    float* y;
    int rows,in,out,xstride,rows_per_seq,kind,source,split;
    int front,slot,m,k;
};
/* Two BF16 pairs as four F32 lanes: the tree GEMM's embedding gather reads 8 bytes at a time. */
__device__ float4 bf4(const uint16_t* p) {
    const uint2 u=*reinterpret_cast<const uint2*>(p);
    return make_float4(bf(uint16_t(u.x)),bf(uint16_t(u.x>>16)),bf(uint16_t(u.y)),bf(uint16_t(u.y>>16)));
}
/* Same column-to-weight mapping as the generic kernel: kind 1 splits Q|K|V,
 * kind 2 splits K|V, everything else is one matrix. */
__device__ const uint16_t* wrow(const MatArgs& a,int col) {
    if(a.kind==1) {
        if(col<a.split)return a.w0+int64_t(col)*a.in;
        col-=a.split;const int kv=(a.out-a.split)/2;
        return col<kv?a.w1+int64_t(col)*a.in:a.w2+int64_t(col-kv)*a.in;
    }
    if(a.kind==2)return col<a.split?a.w0+int64_t(col)*a.in:a.w1+int64_t(col-a.split)*a.in;
    return a.w0+int64_t(col)*a.in;
}
/* Activation tile [gemm_tm x gemm_kc] into LDS, row major with a padded pitch.
 * SRC 0 reads a plain FP32 matrix, SRC 2 gathers anchor/path token embeddings with exactly the bounds
 * and mask rules of the generic matrix_input(). */
template<int SRC>
__device__ void gemm_a(const MatArgs& a,float* la,int row0,int k0) {
    for(int t=threadIdx.x;t<gemm_tm*(gemm_kc/4);t+=gemm_threads) {
        const int r=t/(gemm_kc/4),q=t%(gemm_kc/4),row=row0+r,d=k0+q*4;
        float4 v=make_float4(0.0f,0.0f,0.0f,0.0f);
        if(row<a.rows&&d<a.in&&(SRC!=0||a.valid[row/a.rows_per_seq])) {
            const bool in_bounds=d+4<=a.in;
            if constexpr(SRC==0) {
                const float* src=a.x+int64_t(row)*a.xstride+d;
                if(in_bounds&&((reinterpret_cast<uintptr_t>(src)&15)==0))v=*reinterpret_cast<const float4*>(src);
                else for(int j=0;j<4&&d+j<a.in;++j)reinterpret_cast<float*>(&v)[j]=src[j];
            } else {
                const int b=a.m+1,pos=row%b,job=row/b,seq=job/a.front;
                const uint16_t* e=nullptr;
                if(a.valid[seq]) {
                    if(pos==0)e=a.anchor+int64_t(seq)*a.in;
                    else if(pos<=a.slot) {
                        const int parent=a.frontier[seq*max_frontier+job%a.front];
                        const int index=(parent>=0&&parent<max_nodes)?a.path[(seq*max_nodes+parent)*a.m+pos-1]:-1;
                        if(index>=0&&index<a.m*a.k)e=a.embed+(int64_t(seq)*a.m*a.k+index)*a.in;
                    }
                }
                if(e) {
                    if(in_bounds&&((reinterpret_cast<uintptr_t>(e+d)&7)==0))v=bf4(e+d);
                    else for(int j=0;j<4&&d+j<a.in;++j)reinterpret_cast<float*>(&v)[j]=bf(e[d+j]);
                }
            }
        }
        *reinterpret_cast<float4*>(la+r*a_pitch+q*4)=v;
    }
}
/* Weight tile [gemm_tn x gemm_kc] into LDS, stored K-major (pitch b_pitch) so a
 * thread can read eight FP32 columns of one K coordinate as two b128 loads. The
 * BF16 weights are expanded to FP32 once here, so the compute loop has no
 * conversion: bf(x) is exactly the value the reference multiplies with. */
__device__ void gemm_b(const MatArgs& a,float* lb,int col0,int k0) {
    for(int t=threadIdx.x;t<gemm_tn*(gemm_kc/8);t+=gemm_threads) {
        const int c=t/(gemm_kc/8),q=t%(gemm_kc/8),col=col0+c,d=k0+q*8;
        float* dst=lb+q*8*b_pitch+c;
        if(col<a.out&&d<a.in) {
            const uint16_t* w=wrow(a,col);
            #pragma unroll
            for(int j=0;j<8;++j)dst[j*b_pitch]=d+j<a.in?bf(w[d+j]):0.0f;
        } else {
            #pragma unroll
            for(int j=0;j<8;++j)dst[j*b_pitch]=0.0f;
        }
    }
}
template<int SRC>
__global__ __launch_bounds__(gemm_threads) void tree_gemm(MatArgs a) {
    __shared__ float la[gemm_tm*a_pitch];
    __shared__ float lb[gemm_kc*b_pitch];
    const int col0=blockIdx.x*gemm_tn,row0=blockIdx.y*gemm_tm;
    const int ty=threadIdx.x/16,tx=threadIdx.x%16;
    /* Uniform dead-tile exit, before the first barrier and the first weight load. */
    if(a.live&&row0>=*a.live)return;
    float acc[4][4]={};
    for(int k0=0;k0<a.in;k0+=gemm_kc) {
        if(k0)__syncthreads();
        gemm_a<SRC>(a,la,row0,k0);
        gemm_b(a,lb,col0,k0);
        __syncthreads();
        const int kc=a.in-k0<gemm_kc?a.in-k0:gemm_kc;
        /* Deliberately not unrolled: unrolling the K loop quadruples the live
         * activation/weight registers and drops occupancy from 16 to 8 waves
         * per SIMD, which costs more than the loop overhead. */
        #pragma unroll 1
        for(int kq=0;kq<kc;kq+=4) {
            float av[4][4];float4 bw[4];
            #pragma unroll
            for(int r=0;r<4;++r) {
                const float4 t=*reinterpret_cast<const float4*>(la+(ty*4+r)*a_pitch+kq);
                av[r][0]=t.x;av[r][1]=t.y;av[r][2]=t.z;av[r][3]=t.w;
            }
            #pragma unroll
            for(int j=0;j<4;++j)bw[j]=*reinterpret_cast<const float4*>(lb+(kq+j)*b_pitch+tx*4);
            #pragma unroll
            for(int j=0;j<4;++j) {
                const float ws[4]={bw[j].x,bw[j].y,bw[j].z,bw[j].w};
                #pragma unroll
                for(int c=0;c<4;++c) {
                    #pragma unroll
                    for(int r=0;r<4;++r)acc[r][c]+=av[r][j]*ws[c];
                }
            }
        }
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        const int row=row0+ty*4+r;
        if(row>=a.rows)continue;
        float* dst=a.y+int64_t(row)*a.out+col0+tx*4;
        if(col0+tx*4+4<=a.out)*reinterpret_cast<float4*>(dst)=make_float4(acc[r][0],acc[r][1],acc[r][2],acc[r][3]);
        else {
            #pragma unroll
            for(int c=0;c<4;++c)if(col0+tx*4+c<a.out)dst[c]=acc[r][c];
        }
    }
}
/* Gate/up pair: one activation tile feeds two weight matrices, so the CTA keeps
 * gate and up accumulators for the same 64 output columns and writes
 * silu(gate)*up. a.w1 is the up projection, exactly as rad_ref_dloop_forward
 * applies it; both weight tiles are expanded to FP32 in LDS once. */
__device__ void gemm_b_pair(const MatArgs& a,float* lg,float* lu,int col0,int k0) {
    for(int t=threadIdx.x;t<gate_tn*(gemm_kc/8);t+=gemm_threads) {
        const int c=t/(gemm_kc/8),q=t%(gemm_kc/8),col=col0+c,d=k0+q*8;
        float* dg=lg+q*8*gate_pitch+c;float* du=lu+q*8*gate_pitch+c;
        if(col<a.out&&d<a.in) {
            const uint16_t* wg=a.w0+int64_t(col)*a.in;const uint16_t* wu=a.w1+int64_t(col)*a.in;
            #pragma unroll
            for(int j=0;j<8;++j){const bool live=d+j<a.in;dg[j*gate_pitch]=live?bf(wg[d+j]):0.0f;du[j*gate_pitch]=live?bf(wu[d+j]):0.0f;}
        } else {
            #pragma unroll
            for(int j=0;j<8;++j){dg[j*gate_pitch]=0.0f;du[j*gate_pitch]=0.0f;}
        }
    }
}
template<int SRC>
__global__ __launch_bounds__(gemm_threads) void tree_gemm_gate(MatArgs a) {
    __shared__ float la[gemm_tm*a_pitch];
    __shared__ float lg[gemm_kc*gate_pitch];
    __shared__ float lu[gemm_kc*gate_pitch];
    const int col0=blockIdx.x*gate_tn,row0=blockIdx.y*gemm_tm;
    const int ty=threadIdx.x/16,tx=threadIdx.x%16;
    /* Uniform dead-tile exit, before the first barrier and the first weight load. */
    if(a.live&&row0>=*a.live)return;
    float ag[4][4]={},au[4][4]={};
    for(int k0=0;k0<a.in;k0+=gemm_kc) {
        if(k0)__syncthreads();
        gemm_a<SRC>(a,la,row0,k0);
        gemm_b_pair(a,lg,lu,col0,k0);
        __syncthreads();
        const int kc=a.in-k0<gemm_kc?a.in-k0:gemm_kc;
        for(int kq=0;kq<kc;kq+=4) {
            float av[4][4];float4 bg[4],bu[4];
            #pragma unroll
            for(int r=0;r<4;++r) {
                const float4 t=*reinterpret_cast<const float4*>(la+(ty*4+r)*a_pitch+kq);
                av[r][0]=t.x;av[r][1]=t.y;av[r][2]=t.z;av[r][3]=t.w;
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                bg[j]=*reinterpret_cast<const float4*>(lg+(kq+j)*gate_pitch+tx*4);
                bu[j]=*reinterpret_cast<const float4*>(lu+(kq+j)*gate_pitch+tx*4);
            }
            #pragma unroll
            for(int j=0;j<4;++j) {
                const float gs[4]={bg[j].x,bg[j].y,bg[j].z,bg[j].w};
                const float us[4]={bu[j].x,bu[j].y,bu[j].z,bu[j].w};
                #pragma unroll
                for(int c=0;c<4;++c) {
                    #pragma unroll
                    for(int r=0;r<4;++r){ag[r][c]+=av[r][j]*gs[c];au[r][c]+=av[r][j]*us[c];}
                }
            }
        }
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        const int row=row0+ty*4+r;
        if(row>=a.rows)continue;
        #pragma unroll
        for(int c=0;c<4;++c) {
            const int col=col0+tx*4+c;
            if(col<a.out) {
                const float gv=ag[r][c];
                a.y[int64_t(row)*a.out+col]=(gv/(1.0f+expf(-gv)))*au[r][c];
            }
        }
    }
}

/* ---------------------------------------------------------------- elementwise */
/* The tree metadata gate, used by BOTH the ring tree and the full-context tree: their operand tables
 * agree on slots (2), length (1), pool_ids (6) and the optional anchor (16), so one validation serves
 * both. A sequence is refused unless its slot, its verified length (bounded by the ring window or, in
 * the full path, the slot cache capacity), the optional anchor id, and the WHOLE sorted pool are in
 * range -- checked before any kernel reads a data-dependent address. Every CTA validates its own
 * sequence and reports through workspace().valid, so one malformed sequence refuses only itself. */
__global__ __launch_bounds__(block_threads) void tree_initialize(DeviceArgs a) {
    __shared__ int bad;
    const auto& c=a.c;const int s=blockIdx.x;
    if(threadIdx.x==0)bad=0;
    __syncthreads();
    if(s>=c.n_seq)return;
    const int bound=a.capacity?a.capacity:c.context_window;
    const int slot=static_cast<const int*>(a.t[2])[s];
    bool valid=slot>=0&&slot<c.max_seqs;
    if(valid) {
        const int length=static_cast<const int*>(a.t[1])[slot];
        valid=length>=1&&length<=bound;
    }
    if(valid&&a.t[16]) {
        const int anchor=static_cast<const int*>(a.t[16])[s];
        valid=anchor>=0&&anchor<c.active_vocab;
    }
    if(valid) {
        const auto* pool=static_cast<const int*>(a.t[6])+int64_t(s)*c.proposal_length*c.pool_size;
        for(int i=threadIdx.x;i<c.proposal_length*c.pool_size;i+=block_threads) {
            const int id=pool[i],prev=i%c.pool_size?pool[i-1]:-1;
            if(id<0||id>=c.active_vocab||(prev>=0&&id<=prev))bad=1;
        }
    }
    __syncthreads();
    if(bad)valid=false;
    if(valid&&a.capacity&&c.head_dim==64)full_rope_table(a,s);
    if(threadIdx.x!=0)return;
    Work w=workspace(a);const int root=s*max_nodes;
    w.valid[s]=valid?1:0;w.count[s]=valid?1:0;w.frontier[s*max_frontier]=0;
    w.node_score[root]=0;w.tok[root]=-1;w.parent[root]=-1;w.depth[root]=0;
    for(int i=0;i<c.proposal_length;++i)w.path[root*c.proposal_length+i]=-1;
}
/* The masked suffix is substituted while normalizing instead of in a separate
 * fill pass; every stored value is the one the generic path writes. */
__global__ __launch_bounds__(block_threads) void tree_initial_norm(DeviceArgs a,int front,int slot) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);
    const uint16_t* mask=weight(a,DL_MASK_TOKEN);
    const uint16_t* norm=weight(a,DL_INPUT_NORM,0);
    float* block=w.block+int64_t(job)*b*h;
    float* y=w.norm+int64_t(job)*b*h;
    __shared__ float scale[32];
    for(int row=threadIdx.x;row<b;row+=block_threads) {
        float sum=0;
        for(int d=0;d<h;++d) {
            float v=block[row*h+d];
            if(row>slot){v=bf(mask[d]);block[row*h+d]=v;}
            sum+=v*v;
        }
        scale[row]=1.0f/sqrtf(sum/float(h)+c.eps);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<b*h;i+=block_threads) {
        const int row=i/h,d=i%h;
        y[i]=block[i]*scale[row]*bf(norm[d]);
    }
    __syncthreads();
}
__global__ __launch_bounds__(block_threads) void tree_conv_pre(DeviceArgs a,int front,int layer,int mlp) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);
    convolve(w.norm+int64_t(job)*b*h,w.delta+int64_t(job)*b*4*(h/c.conv_group_size),
        weight(a,mlp?DL_MLP_BASE:DL_ATTN_BASE,layer),w.pre+int64_t(job)*b*h,b,h,c.conv_group_size,0);
}
__global__ __launch_bounds__(block_threads) void tree_context_rope(DeviceArgs a,int layer) {
    const auto& c=a.c;const int s=blockIdx.x,kv=c.num_kv_heads*c.head_dim,dh=c.head_dim;
    if(!seq_valid(a,s))return;
    float* cache=workspace(a).ctxkv+(int64_t(layer)*c.n_seq+s)*c.context_window*2*kv;
    for(int i=threadIdx.x;i<c.context_window*(kv/2);i+=block_threads) {
        const int row=i/(kv/2),head=(i/(dh/2))%c.num_kv_heads,d=i%(dh/2),at=row*2*kv+head*dh+d;
        const float angle=float(row)*powf(c.rope_theta,-2.0f*float(d)/float(dh));
        const float co=cosf(angle),sn=sinf(angle),lo=cache[at],hi=cache[at+dh/2];
        cache[at]=lo*co-hi*sn;cache[at+dh/2]=hi*co+lo*sn;
    }
}
__global__ __launch_bounds__(block_threads) void tree_attention(DeviceArgs a,int front,int layer) {
    __shared__ float probability[9*12*41];
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);
    const int kv=c.num_kv_heads*c.head_dim,dh=c.head_dim,pitch=h+2*kv,mem=c.context_window+b;
    float* qkv=w.qkv+int64_t(job)*b*pitch;
    const float* cache=w.ctxkv+(int64_t(layer)*c.n_seq+s)*c.context_window*2*kv;
    const int seqslot=static_cast<const int*>(a.t[2])[s];
    const int length=static_cast<const int*>(a.t[1])[seqslot];
    for(int i=threadIdx.x;i<b*((h+kv)/2);i+=block_threads) {
        const int row=i/((h+kv)/2),channel=i%((h+kv)/2);
        const bool key=channel>=h/2;const int rel=key?channel-h/2:channel;
        const int head=rel/(dh/2),d=rel%(dh/2),at=row*pitch+(key?h:0)+head*dh+d;
        const float angle=float(c.context_window+row)*powf(c.rope_theta,-2.0f*float(d)/float(dh));
        const float co=cosf(angle),sn=sinf(angle),lo=qkv[at],hi=qkv[at+dh/2];
        qkv[at]=lo*co-hi*sn;qkv[at+dh/2]=hi*co+lo*sn;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<b*c.num_q_heads*mem;i+=block_threads) {
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
    for(int i=threadIdx.x;i<b*c.num_q_heads;i+=block_threads) {
        float mx=-INFINITY,sum=0;for(int j=0;j<mem;++j)mx=fmaxf(mx,probability[i*mem+j]);
        for(int j=0;j<mem;++j){const float e=expf(probability[i*mem+j]-mx);probability[i*mem+j]=e;sum+=e;}
        for(int j=0;j<mem;++j)probability[i*mem+j]/=sum;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<b*h;i+=block_threads) {
        const int row=i/h,qh=(i%h)/dh,d=i%dh,kh=qh/(c.num_q_heads/c.num_kv_heads);float sum=0;
        for(int j=0;j<mem;++j) {
            const float v=j<c.context_window?cache[j*2*kv+kv+kh*dh+d]:qkv[(j-c.context_window)*pitch+h+kv+kh*dh+d];
            sum+=probability[(row*c.num_q_heads+qh)*mem+j]*v;
        }
        w.norm[int64_t(job)*b*h+i]=sum;
    }
}
/* Stage 0: attention finish/residual/post norm. Stage 1: MLP finish/residual,
 * then the next layer's input norm, or just the required final output row. */
__global__ __launch_bounds__(block_threads) void tree_finish_norm(DeviceArgs a,int front,int layer,int stage,int slot) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;Work w=workspace(a);const int64_t off=int64_t(job)*b*h;
    convolve(w.out+off,w.delta+int64_t(job)*b*4*(h/c.conv_group_size),
        weight(a,stage?DL_MLP_BASE:DL_ATTN_BASE,layer),w.pre+off,b,h,c.conv_group_size,1);
    for(int i=threadIdx.x;i<b*h;i+=block_threads)w.block[off+i]+=w.pre[off+i];
    __syncthreads();
    if(!stage)rms(w.block+off,weight(a,DL_POST_NORM,layer),w.norm+off,b,h,c.eps);
    else if(layer+1<c.num_layers)rms(w.block+off,weight(a,DL_INPUT_NORM,layer+1),w.norm+off,b,h,c.eps);
    else rms(w.block+off+(slot+1)*h,weight(a,DL_FINAL_NORM),w.final+int64_t(job)*h,1,h,c.eps);
}

/* ------------------------------------------------------------------- scoring */
__device__ bool better(float v,int id,float best,int best_id){return v>best||(v==best&&id<best_id);}
/* One depth's scoring. Threads below k take the unary dot of candidate t, the next k the transition
 * one -- one accumulator per candidate walking ASCENDING d, exactly the reference's sum, and the same
 * thread then adds the two into the candidate value. What the old loop did not have was run-up: it
 * issued one 2-byte weight load and one float load per element and consumed both immediately, so every
 * add waited out the memory round trip of its own operand. That single fact is the whole 125.8 us.
 *
 * The ILP version walks sixteen elements per step with the loads issued as wide vectors (four float4
 * of the vector and two uint4 of the BF16 row) and unrolls four such steps, so sixty-four elements of
 * run-up are in flight while the adds of the previous step retire. The order of the additions is
 * untouched: the same single accumulator, the same ascending d, the same two-rounding mul/add. */
__global__ __launch_bounds__(block_threads) void tree_scores(DeviceArgs a,int front,int slot) {
    __shared__ float values[64];
    __shared__ float sunary[64],sedge[64];
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,k=c.pool_size,r=c.selector_rank,m=c.proposal_length;
    if(!seq_valid(a,s))return;Work w=workspace(a);const int parent=w.frontier[s*max_frontier+job%front];
    const float* y=slot?w.y+int64_t(job)*c.target_width:static_cast<const float*>(a.t[5])+int64_t(s)*m*c.target_width;
    const uint16_t* pred=nullptr;
    if(slot) {
        const int source=w.path[(s*max_nodes+parent)*m+slot-1];
        if(source<0||source>=m*k)return;
        pred=static_cast<const uint16_t*>(a.t[10])+(int64_t(s)*m*k+source)*r;
    } else pred=static_cast<const uint16_t*>(a.t[12])+s*r;
    for(int d=threadIdx.x;d<r;d+=block_threads)w.rank[int64_t(job)*r+d]*=bf(pred[d]);
    __syncthreads();
    if(int(threadIdx.x)<2*k) {
        const int cand=int(threadIdx.x)%k,index=(s*m+slot)*k+cand;
        if(int(threadIdx.x)<k) {
            const int W=c.target_width;
            const uint16_t* uw=static_cast<const uint16_t*>(a.t[8])+int64_t(index)*W;
            /* The bias is the reference's seed value, not an accumulation: `acc = bias` and then the
             * products are added to it in ascending d. `0` is exact when the reference starts at 0. */
            float acc=slot?0.0f:static_cast<const float*>(a.t[7])[index];
            if(slot)acc=dot_bf16(y,uw,W);
            sunary[cand]=acc;
        } else {
            const float* rk=w.rank+int64_t(job)*r;
            const uint16_t* ew=static_cast<const uint16_t*>(a.t[11])+int64_t(index)*r;
            sedge[cand]=dot_bf16(rk,ew,r);
        }
    }
    __syncthreads();
    for(int c0=threadIdx.x;c0<k;c0+=block_threads)values[c0]=sunary[c0]+sedge[c0];
    __syncthreads();
    /* Maximum and sum in ascending candidate order, exactly as the reference. */
    if(threadIdx.x==0) {
        float mx=-INFINITY,sum=0;for(int i=0;i<k;++i)mx=fmaxf(mx,values[i]);
        for(int i=0;i<k;++i)sum+=expf(values[i]-mx);
        const float logsum=logf(sum);
        for(int i=0;i<k;++i)w.scores[job*k+i]=w.node_score[s*max_nodes+parent]+((values[i]-mx)-logsum);
    }
}
/* Selection: same rule, same tie order and same node bookkeeping as the generic
 * kernel, but the per-quota argmax runs over the whole block. The reference
 * keeps the lowest index among equal scores, and better() reproduces that. */
__global__ __launch_bounds__(block_threads) void tree_select(DeviceArgs a,int front,int slot,int quota) {
    const auto& c=a.c;const int s=blockIdx.x,k=c.pool_size,m=c.proposal_length;
    if(!seq_valid(a,s))return;
    __shared__ unsigned char used[max_frontier*64];
    __shared__ float bval[block_threads];__shared__ int bidx[block_threads];__shared__ int bail;
    Work w=workspace(a);const int root=s*max_nodes;const int count=w.count[s];
    const int total=front*k;
    for(int i=threadIdx.x;i<total;i+=block_threads)used[i]=0;
    for(int q=0;q<quota;++q) {
        if(threadIdx.x==0)bail=0;
        __syncthreads();
        float value=-INFINITY;int index=-1;
        for(int i=threadIdx.x;i<total;i+=block_threads) {
            if(used[i])continue;
            const float v=w.scores[(s*front)*k+i];
            if(v>value||(v==value&&index>=0&&i<index)){value=v;index=i;}
        }
        bval[threadIdx.x]=value;bidx[threadIdx.x]=index>=0?index:INT32_MAX;
        __syncthreads();
        for(unsigned step=block_threads/2;step;step/=2) {
            if(threadIdx.x<step&&better(bval[threadIdx.x+step],bidx[threadIdx.x+step],bval[threadIdx.x],bidx[threadIdx.x])) {
                bval[threadIdx.x]=bval[threadIdx.x+step];bidx[threadIdx.x]=bidx[threadIdx.x+step];
            }
            __syncthreads();
        }
        if(threadIdx.x==0) {
            const int best=bidx[0]==INT32_MAX?-1:bidx[0];
            w.remap[root+q]=best;
            const int node=count+q,parent=best<0?-1:w.frontier[s*max_frontier+best/k];
            if(best<0||node>=max_nodes||parent<0||parent>=count) {bail=1;return;}
            const int index=root+node,source=slot*k+best%k;
            w.tok[index]=static_cast<const int*>(a.t[6])[s*m*k+source];
            w.parent[index]=parent;w.depth[index]=slot+1;w.node_score[index]=bval[0];
            for(int j=0;j<m;++j)w.path[index*m+j]=w.path[(root+parent)*m+j];
            w.path[index*m+slot]=source;
            if(bidx[0]!=INT32_MAX)used[bidx[0]]=1;
        }
        __syncthreads();
        if(bail)return;
    }
    if(threadIdx.x==0) {
        for(int q=0;q<quota;++q)w.frontier[s*max_frontier+q]=count+q;
        w.count[s]=count+quota;
    }
}
__global__ __launch_bounds__(block_threads) void tree_write_tree(DeviceArgs a) {
    if(threadIdx.x!=0)return;
    const auto& c=a.c;const int s=blockIdx.x,root=s*max_nodes;Work w=workspace(a);
    int* tokens=static_cast<int*>(a.t[13])+s*16;int* parents=static_cast<int*>(a.t[14])+s*16;
    for(int i=0;i<16;++i){tokens[i]=-1;parents[i]=-1;}static_cast<int*>(a.t[15])[s]=0;
    int* packed=a.t[17]?static_cast<int*>(a.t[17])+s*33:nullptr;
    if(packed)for(int i=0;i<33;++i)packed[i]=-1;
    if(!w.valid[s]){if(packed)packed[0]=0;return;}
    const int count=w.count[s];int kept=count-1;
    for(int i=0;i<count;++i)w.remap[root+i]=0;w.remap[root]=1;
    if(c.pruned&&kept>c.budget) {
        kept=c.budget;
        for(int q=0;q<kept;++q) {
            int best=-1;float priority=-INFINITY;
            for(int i=1;i<count;++i)if(!w.remap[root+i]) {
                const float p=w.node_score[root+i]+c.depth_bonus*float(w.depth[root+i]);
                if(best<0||p>priority||(p==priority&&w.depth[root+i]<w.depth[root+best])){best=i;priority=p;}
            }
            w.remap[root+best]=1;
        }
    }else for(int i=1;i<count;++i)w.remap[root+i]=1;
    int next=0;for(int i=0;i<count;++i)w.remap[root+i]=w.remap[root+i]?next++:-1;
    // Nonpositive transitions and depth_bonus make pruning ancestor-closed;
    // retain the bounds guard rather than producing an invalid parent -2.
    for(int i=1;i<count;++i)if(w.remap[root+i]>=0) {
        const int parent=w.parent[root+i];
        if(parent<0||parent>=i||w.remap[root+parent]<0){if(packed)packed[0]=0;return;}
    }
    for(int i=1;i<count;++i)if(w.remap[root+i]>=0) {
        const int index=w.remap[root+i]-1;tokens[index]=w.tok[root+i];parents[index]=w.remap[root+w.parent[root+i]]-1;
    }
    static_cast<int*>(a.t[15])[s]=kept;
    if(packed) {
        packed[0]=kept+1;packed[1]=static_cast<const int*>(a.t[16])[s];
        for(int i=0;i<kept;++i){packed[2+i]=tokens[i];packed[18+i]=parents[i]+1;}
    }
}

/* ------------------------------------------------------------------ host side */
/* The ring op's own validity test, unchanged: it is the geometry the OLD dartree rows still serve. */
bool supported(const RadDLoopConfig& c,bool tiny) {
    if(c.world_size!=2||c.num_layers!=2||c.proposal_length!=8||c.context_window!=32||
       c.conv_kernel_size!=2||c.conv_group_size!=16||c.pool_size!=64||c.width>12||
       c.vocab_size_shard*2!=c.vocab_size||(c.vocab_lo!=0&&c.vocab_lo!=c.vocab_size_shard))return false;
    if(tiny)return c.hidden_size==64&&c.intermediate_size==128&&c.num_q_heads==4&&c.num_kv_heads==2&&
        c.head_dim==16&&c.context_features_dim==48&&c.target_width==32&&c.selector_rank==8&&c.vocab_size==512&&c.active_vocab==509;
    return c.hidden_size==768&&c.intermediate_size==2304&&c.num_q_heads==12&&c.num_kv_heads==4&&
        c.head_dim==64&&c.context_features_dim==7680&&c.target_width==2560&&c.selector_rank==256&&c.vocab_size==248320&&c.active_vocab==248077;
}
int64_t scratch_bytes(const RadArgs* a) {
    RadDLoopConfig c{};if(!a||rad_dloop_config(a->p,a->n_p,&c)!=RAD_OK)return 0;
    return 4*ring_work_elements(c,RAD_DLOOP_DARTREE);
}
struct Launcher {
    hipStream_t stream;
    int send(const void* kernel,dim3 grid,int nthr,void** args)const {
        return hipLaunchKernel(kernel,grid,dim3(nthr),args,0,stream)==hipSuccess?RAD_OK:RAD_E_DEVICE;
    }
    int simple(const void* kernel,DeviceArgs& a,int blocks)const {void* args[]={&a};return send(kernel,dim3(blocks),block_threads,args);}
    int block(const void* kernel,DeviceArgs& a,int blocks,int n,void** extra)const {
        void* args[8];args[0]=&a;for(int i=0;i<n;++i)args[i+1]=extra[i];
        return send(kernel,dim3(blocks),block_threads,args);
    }
    int gemm(MatArgs m)const {
        const void* kernel=nullptr;
        if(m.kind==3)kernel=m.source==2?reinterpret_cast<const void*>(tree_gemm_gate<2>):reinterpret_cast<const void*>(tree_gemm_gate<0>);
        else kernel=m.source==2?reinterpret_cast<const void*>(tree_gemm<2>):reinterpret_cast<const void*>(tree_gemm<0>);
        const int tn=m.kind==3?gate_tn:gemm_tn;
        void* args[]={&m};
        return send(kernel,dim3((m.out+tn-1)/tn,(m.rows+gemm_tm-1)/gemm_tm),gemm_threads,args);
    }
};
MatArgs matrix_args(const DeviceArgs& a,const float* x,const uint16_t* weights,float* y,
                    int rows,int in,int out,int perseq) {
    MatArgs m{};m.x=x;m.w0=weights;m.y=y;m.rows=rows;m.in=in;m.out=out;m.xstride=in;
    m.rows_per_seq=perseq;m.valid=workspace(a).valid;return m;
}
int context_cache(DeviceArgs& a,const Launcher& launch) {
    const auto& c=a.c;Work w=workspace(a);const int kv=c.num_kv_heads*c.head_dim;
    const float* ctx=static_cast<const float*>(a.t[0]);
    for(int layer=0;layer<c.num_layers;++layer) {
        MatArgs m=matrix_args(a,ctx,weight(a,DL_K_PROJ,layer),w.ctxkv+int64_t(layer)*c.n_seq*c.context_window*2*kv,
            c.n_seq*c.context_window,c.hidden_size,2*kv,c.context_window);
        m.kind=2;m.split=kv;m.w1=weight(a,DL_V_PROJ,layer);
        int rc=launch.gemm(m);if(rc)return rc;
        void* argv[]={&layer};rc=launch.block(reinterpret_cast<const void*>(tree_context_rope),a,c.n_seq,1,argv);if(rc)return rc;
    }
    return RAD_OK;
}
int backbone_batch(DeviceArgs& a,const Launcher& launch,int front,int slot) {
    const auto& c=a.c;Work w=workspace(a);const int b=c.proposal_length+1,h=c.hidden_size,kv=c.num_kv_heads*c.head_dim;
    const int jobs=c.n_seq*front,rows=jobs*b;
    Launcher l=launch;
MatArgs embed=matrix_args(a,nullptr,weight(a,DL_TOKEN_PROJ),w.block,rows,c.target_width,h,front*b);
    embed.source=2;embed.anchor=static_cast<const uint16_t*>(a.t[3]);
    embed.embed=static_cast<const uint16_t*>(a.t[9]);
    embed.path=w.path;embed.frontier=w.frontier;embed.front=front;embed.slot=slot;embed.m=c.proposal_length;embed.k=c.pool_size;
    int rc=l.gemm(embed);if(rc)return rc;
    void* initial[]={&front,&slot};rc=l.block(reinterpret_cast<const void*>(tree_initial_norm),a,jobs,2,initial);if(rc)return rc;
    for(int layer=0;layer<c.num_layers;++layer) {
        for(int mlp=0;mlp<2;++mlp) {
            MatArgs dyn=matrix_args(a,w.norm,weight(a,mlp?DL_MLP_KERNEL:DL_ATTN_KERNEL,layer),w.delta,rows,h,4*(h/c.conv_group_size),front*b);
            rc=l.gemm(dyn);if(rc)return rc;
            void* conv[]={&front,&layer,&mlp};rc=l.block(reinterpret_cast<const void*>(tree_conv_pre),a,jobs,3,conv);if(rc)return rc;
            if(!mlp) {
                MatArgs qkv=matrix_args(a,w.pre,weight(a,DL_Q_PROJ,layer),w.qkv,rows,h,h+2*kv,front*b);
                qkv.kind=1;qkv.split=h;qkv.w1=weight(a,DL_K_PROJ,layer);qkv.w2=weight(a,DL_V_PROJ,layer);
                rc=l.gemm(qkv);if(rc)return rc;
                void* attn[]={&front,&layer};rc=l.block(reinterpret_cast<const void*>(tree_attention),a,jobs,2,attn);if(rc)return rc;
                rc=l.gemm(matrix_args(a,w.norm,weight(a,DL_O_PROJ,layer),w.out,rows,h,h,front*b));if(rc)return rc;
            } else {
                MatArgs gate=matrix_args(a,w.pre,weight(a,DL_GATE_PROJ,layer),w.ff,rows,h,c.intermediate_size,front*b);
                gate.kind=3;gate.w1=weight(a,DL_UP_PROJ,layer);rc=l.gemm(gate);if(rc)return rc;
                rc=l.gemm(matrix_args(a,w.ff,weight(a,DL_DOWN_PROJ,layer),w.out,rows,c.intermediate_size,h,front*b));if(rc)return rc;
            }
            void* finish[]={&front,&layer,&mlp,&slot};rc=l.block(reinterpret_cast<const void*>(tree_finish_norm),a,jobs,4,finish);if(rc)return rc;
        }
    }
    return l.gemm(matrix_args(a,w.final,weight(a,DL_OUTPUT_PROJ),w.y,jobs,h,c.target_width,front));
}
int score_select(DeviceArgs& a,const Launcher& launch,int front,int slot,int quota) {
    Launcher l=launch;
    const auto& c=a.c;Work w=workspace(a);
    const float* y=slot?w.y:static_cast<const float*>(a.t[5]);
    MatArgs m=matrix_args(a,y,weight(a,DL_SELECTOR_PROJ),w.rank,c.n_seq*front,c.target_width,c.selector_rank,front);
    if(!slot)m.xstride=c.proposal_length*c.target_width;
    int rc=l.gemm(m);if(rc)return rc;
    void* scores[]={&front,&slot};rc=l.block(reinterpret_cast<const void*>(tree_scores),a,c.n_seq*front,2,scores);if(rc)return rc;
    void* select[]={&front,&slot,&quota};return l.block(reinterpret_cast<const void*>(tree_select),a,c.n_seq,3,select);
}
int run(const RadArgs* args,RadStream stream,bool tiny) {
    DeviceArgs a{};int rc=rad_dloop_args(args,RAD_DLOOP_DARTREE,&a.c);if(rc)return rc;
    if(!supported(a.c,tiny)||args->world_size!=2||args->rank<0||args->rank>=2||a.c.vocab_lo!=args->rank*a.c.vocab_size_shard)return RAD_E_UNSUPPORTED;
    if(args->scratch_bytes<scratch_bytes(args))return RAD_E_SCRATCH;
    for(int i=0;i<args->n_t;++i)a.t[i]=args->t[i].data;
    for(int l=0;l<2;++l)for(int w=0;w<20;++w)a.offset[l][w]=int(rad_dloop_weight_offset(a.c,w,l));
    a.scratch=static_cast<float*>(args->scratch);a.op=RAD_DLOOP_DARTREE;const auto& c=a.c;
    const Launcher launch{reinterpret_cast<hipStream_t>(stream)};
    rc=launch.simple(reinterpret_cast<const void*>(tree_initialize),a,c.n_seq);if(rc)return rc;
    rc=context_cache(a,launch);if(rc)return rc;
    const int width=frontier_bound(c),capacity=c.pruned?width*c.proposal_length:c.budget;
    int count=1,front=1;
    for(int slot=0;slot<c.proposal_length&&count<=capacity;++slot) {
        if(slot) {
            rc=backbone_batch(a,launch,front,slot);if(rc)return rc;
        }
        const int remaining=capacity+1-count;
        int quota=c.pruned?width:(remaining+c.proposal_length-slot-1)/(c.proposal_length-slot);
        quota=quota<remaining?quota:remaining;quota=quota<front*c.pool_size?quota:front*c.pool_size;
        rc=score_select(a,launch,front,slot,quota);if(rc)return rc;
        count+=quota;front=quota;
    }
    return launch.simple(reinterpret_cast<const void*>(tree_write_tree),a,c.n_seq);
}

/* ====================================================== full-context path (dloop_full.h)
 *
 * The tree op's full-context form keeps this file's register-tiled GEMM -- the depth-batched walk at
 * width<=12 is exactly the shape that kernel was written for -- and takes every stage sequence, the
 * arena and both shared kernels from dloop_full_device.h. The context K/V is NOT rebuilt: the op
 * reads the persistent cache the append wrote, at the block's absolute positions, at every depth.
 */
namespace {
struct FullPolicy {
    hipStream_t stream;
    hipStream_t stream_of() const {return stream;}
    int dyn(const DeviceArgs& a,const GemmReq& r,int layer,int mlp) const {
        if(!skinny_dyn_fits(r,a.c))return RAD_E_UNSUPPORTED;
        return skinny_dyn_launch(r,a,layer,mlp,false,stream);
    }
    /* GEMM request -> this file's register-tiled kernel. `split` is the column at which a combined
     * weight operand switches from w0 to w1/w2, exactly as the ring path's MatArgs uses it. */
    int gemm(const GemmReq& r) const {
        /* The decode shape (rows <= skinny_rows_max, no dead-row exit, aligned operands) goes to the
         * lane-per-row kernel in dloop_skinny.hip; everything else -- a deeper tree, the append's
         * staging -- keeps this file's register-tiled kernel. */
        if(skinny_fits(r))return skinny_gemm_launch(r,false,stream);
        MatArgs m{};
        m.x=r.x;m.w0=r.w0;m.w1=r.w1;m.w2=r.w2;m.anchor=r.anchor;m.embed=r.embed;m.y=r.y;
        m.valid=r.valid;m.path=r.path;m.frontier=r.frontier;m.live=r.live;
        m.rows=r.rows;m.in=r.in;m.out=r.out;m.xstride=r.xstride;m.rows_per_seq=r.rows_per_seq;
        m.kind=r.kind;m.source=r.source;m.split=r.split;
        m.front=r.front;m.slot=r.slot;m.m=r.m;m.k=r.k;
        const void* kernel=nullptr;
        if(r.kind==3)kernel=r.source==2?reinterpret_cast<const void*>(tree_gemm_gate<2>):reinterpret_cast<const void*>(tree_gemm_gate<0>);
        else kernel=r.source==2?reinterpret_cast<const void*>(tree_gemm<2>):reinterpret_cast<const void*>(tree_gemm<0>);
        const int tn=r.kind==3?gate_tn:gemm_tn;
        void* args[]={&m};
        return send(kernel,dim3((m.out+tn-1)/tn,(m.rows+gemm_tm-1)/gemm_tm),gemm_threads,args,stream);
    }
};
/* The tree's per-depth draft rows are gathered from the frontier and the path, so the shared
 * backbone's embed stage carries the frontier and path pointers the tree GEMM already reads. The
 * final rows are the single row slot+1 per job, which `base_out=false` selects.
 *
 * One depth's scoring and selection: the selector projection, the candidate log-probability, then the
 * same stable argmax the ring path uses -- including its tie order (descending score, lowest index)
 * and its node bookkeeping. Only the operands differ (the full tree carries the cache first), so the
 * stage is the ring path's own kernel bodies applied to the full arena. */
int score_select_full(DeviceArgs& a,const FullPolicy& p,int front,int slot,int quota) {
    const auto& c=a.c;Work w=workspace(a);
    GemmReq m{};
    m.x=slot?w.y:static_cast<const float*>(a.t[5]);
    m.w0=full_weight(a,DL_SELECTOR_PROJ);m.y=w.rank;
    m.rows=c.n_seq*front;m.in=c.target_width;m.out=c.selector_rank;
    m.xstride=slot?c.target_width:c.proposal_length*c.target_width;m.rows_per_seq=front;
    m.valid=w.valid;
    int rc=p.gemm(m);if(rc)return rc;
    int32_t e[]={front,slot};
    void* scores[]={&a,&e[0],&e[1]};
    rc=send(reinterpret_cast<const void*>(tree_scores),dim3(c.n_seq*front),block_threads,scores,p.stream);
    if(rc)return rc;
    void* select[]={&a,&e[0],&e[1],&quota};
    return send(reinterpret_cast<const void*>(tree_select),dim3(c.n_seq),block_threads,select,p.stream);
}
}
int run_full_tree(const RadArgs* args,RadStream stream,bool tiny) {
    RadDLoopFullConfig f{};
    int rc=rad_dloop_full_args(args,RAD_DLOOP_FULL_DARTREE,&f);if(rc)return rc;
    DeviceArgs a{};a.c=f.model;a.capacity=f.context_capacity;
    if(!full_supported(a,tiny)||args->world_size!=2||args->rank<0||args->rank>=2)return RAD_E_UNSUPPORTED;
    if(a.c.vocab_lo!=args->rank*a.c.vocab_size_shard)return RAD_E_UNSUPPORTED;
    /* Fail closed on both bounds: the device layout is the floor only the device knows, and the row's
     * published hook is the other floor because that is what the engine allocated from. */
    const int64_t need=full_scratch_bytes(a,FULL_TREE);
    const int64_t hook=rad_dloop_full_tree_scratch(args);
    if(args->scratch_bytes<need||args->scratch_bytes<hook)return RAD_E_SCRATCH;
    for(int i=0;i<args->n_t;++i)a.t[i]=args->t[i].data;
    const int layers=a.c.num_layers<5?a.c.num_layers:5;
    for(int l=0;l<layers;++l)for(int w=0;w<20;++w)a.offset[l][w]=int(rad_dloop_weight_offset(a.c,w,l));
    a.scratch=static_cast<float*>(args->scratch);a.op=FULL_TREE;
    const FullPolicy policy{reinterpret_cast<hipStream_t>(stream)};
    const auto& c=a.c;
    /* The full tree's operands 0..17 line up with the ring tree's 0..17, so this file's own metadata
     * gate (slots 2, length 1, pool_ids 6, optional anchor 16) applies unchanged; only the length bound
     * differs, and that comes from `capacity` when it is set. */
    void* plain[]={&a};
    rc=send(reinterpret_cast<const void*>(tree_initialize),dim3(c.n_seq),threads,plain,policy.stream);
    if(rc)return rc;
    const int width=frontier_bound(c),capacity=c.pruned?width*c.proposal_length:c.budget;
    int count=1,front=1;
    for(int slot=0;slot<c.proposal_length&&count<=capacity;++slot) {
        /* Slot 0 scores the anchor row directly from y_base, exactly as the ring tree does: the first
         * depth has no parent to gather a draft row from. Deeper slots run the full backbone over the
         * frontier, and every layer's attention reads the PERSISTENT cache: no launch in this loop
         * projects or packs historical K/V. */
        if(slot) {
            rc=full_backbone<1>(a,policy,front,slot,0,nullptr);if(rc)return rc;
        }
        const int remaining=capacity+1-count;
        int quota=c.pruned?width:(remaining+c.proposal_length-slot-1)/(c.proposal_length-slot);
        quota=quota<remaining?quota:remaining;quota=quota<front*c.pool_size?quota:front*c.pool_size;
        rc=score_select_full(a,policy,front,slot,quota);if(rc)return rc;
        count+=quota;front=quota;
    }
    return send(reinterpret_cast<const void*>(tree_write_tree),dim3(c.n_seq),threads,plain,policy.stream);
}
}   /* anonymous namespace: everything above is this translation unit's own */

/* The full-context tree entry points, OUTSIDE the anonymous namespace because the registration in
 * dloop_full.cpp names them. */
#define FULL_ENTRY(name,tiny) \
    int name(const RadArgs* a,RadStream s){return run_full_tree(a,s,tiny);}
FULL_ENTRY(rad_mine_dloop_full_tree_h768_l5_tp2,false)
FULL_ENTRY(rad_mine_dloop_full_tree_h64_l5_tp2,true)
#undef FULL_ENTRY

extern "C" int64_t rad_mine_dloop_tree_scratch(const RadArgs* a){return scratch_bytes(a);}
#define ENTRY(name,tiny) extern "C" int name(const RadArgs* a,RadStream s){return run(a,s,tiny);}
ENTRY(rad_mine_dloop_dartree_h768_m8_tp2,false)
ENTRY(rad_mine_dloop_dartree_h64_m8_tp2,true)
#undef ENTRY
