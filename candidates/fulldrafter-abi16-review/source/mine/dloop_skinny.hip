#include <hip/hip_runtime.h>
#include <cstdint>
#include "dloop_full.h"
#include "dloop_full_device.h"
#include "dloop_skinny.h"

namespace {
using namespace rad_dloop;
/* One CTA = 128 threads = 4 waves; a wave owns skinny_tn adjacent output columns, a lane owns one row.
 * skinny_tn is 2 because that is where the prototype's measured curve flattens: the wave's activation
 * registers feed two columns' weight loads, so the K loop is driven by two independent FP32 chains per
 * lane instead of one, and the accumulator footprint stays far below the 96-VGPR budget. The 1/2/4/8
 * comparison and the row crossover are measured in docs/dloop_speed/W1_skinny_gemm.md.
 *
 * WHY NO fmaf. This plugin builds with -ffp-contract=off, so `a*b+c` is one multiply and one add --
 * two roundings, and the chain over ascending k is element for element the host reference's
 * `z+=x[k]*bf(w[k])` and the tree GEMM's `acc[r][c]+=av[r][j]*ws[c]`. fmaf would round once and drift
 * from both. */
constexpr int skinny_threads=128;
constexpr int skinny_tn=2;

/* The wave-uniform weight row of one output column, resolved once per column group and never inside
 * the K loop: same column-to-weight mapping as the tree GEMM's wrow() and the tiled kernel's
 * matrix_weight() (kind 1 splits Q|K|V at `split`, kind 2 splits K|V, everything else is one matrix).
 * The gate call is kind 3, which is the one-matrix case here: gate is w0 and the up projection is w1. */
__device__ __forceinline__ const uint16_t* skinny_wrow(const GemmReq& a,int col) {
    if(a.kind==1) {
        if(col<a.split)return a.w0+int64_t(col)*a.in;
        col-=a.split;const int kv=(a.out-a.split)/2;
        return col<kv?a.w1+int64_t(col)*a.in:a.w2+int64_t(col-kv)*a.in;
    }
    if(a.kind==2)return col<a.split?a.w0+int64_t(col)*a.in:a.w1+int64_t(col-a.split)*a.in;
    return a.w0+int64_t(col)*a.in;
}
/* The activation row of one lane, or null when the row contributes nothing: past the row count, an
 * invalid sequence, or a path position the tree has no token for.
 *
 * NULL IS A ZERO ROW. That is the whole validity contract this kernel inherits: a dead row's
 * accumulator stays exactly 0.0f, so it reaches y as +0 (as gemm_a's zeroed activation tile does) and
 * never as uninitialised scratch. Only rows at or past `rows` are left unwritten, which is the tiled
 * kernel's own store guard.
 *
 * The embedding gather carries gemm_a<2>'s anchor/path/mask rules: pos 0 is the anchor row, pos 1..slot
 * walks the pooled path token, anything past the mask is a zero row. */
__device__ __forceinline__ const void* skinny_row_src(const GemmReq& a,int row) {
    if(row>=a.rows)return nullptr;
    if(!a.valid[row/a.rows_per_seq])return nullptr;
    if(a.source!=2)return a.x+int64_t(row)*a.xstride;
    const int b=a.m+1,pos=row%b,job=row/b,seq=job/a.front;
    if(pos==0)return a.anchor+int64_t(seq)*a.in;
    if(pos>a.slot)return nullptr;
    const int parent=a.frontier[seq*max_frontier+job%a.front];
    if(parent<0||parent>=max_nodes)return nullptr;
    const int index=a.path[(seq*max_nodes+parent)*a.m+pos-1];
    if(index<0||index>=a.m*a.k)return nullptr;
    return a.embed+(int64_t(seq)*a.m*a.k+index)*a.in;
}
/* The j-th BF16 of a packed 32-bit word, widened exactly the way bf() and the BF16 operand widening of
 * both other kernels do it: the high half IS the float. */
__device__ __forceinline__ float bfj(uint32_t u,int j) {return bf(uint16_t(u>>(16*(j&1))));}

/* Eight ascending k of one row: one 8-value activation vector in registers, and one weight vector
 * for each of the FOUR columns the two accumulator pairs belong to. GATE is off in the GEMM path
 * (the pair is then (column 0, column 1)); the fused conv producer in this file turns it on and
 * hands the two pairs its group's four delta columns `{g, ng+g, 2ng+g, 3ng+g}`.
 *
 * THE ROUNDING IS A PROPERTY OF THE KERNEL THIS ONE REPLACES, NOT A FREE CHOICE. The tree file's
 * register-tiled kernel accumulates with `acc+=a*b` (two roundings, no contraction because the plugin
 * builds with -ffp-contract=off); the base file's tiled kernel accumulates with an explicit `fmaf`
 * (one rounding) -- 213 v_fma_f32 in dloop_kernels.hip.o against 0 in dloop_tree.hip.o. A skinny
 * kernel that picked one of the two would change the bits of the path whose kernel did it the other
 * way, which is what the parity harness compares. FUSE therefore reproduces the replaced kernel's own
 * arithmetic: false for the tree policy, true for the base policy. Ascending k with a single
 * accumulator per output column is the summation order in both cases. */
template<bool GATE,bool FUSE>
__device__ __forceinline__ void skinny_k8(const uint16_t* p0,const uint16_t* p1,const uint16_t* q0,const uint16_t* q1,
    int k,const float xv[8],float& y0,float& y1,float& z0,float& z1) {
    const uint4 A=*reinterpret_cast<const uint4*>(p0+k),B=*reinterpret_cast<const uint4*>(p1+k);
    const uint32_t a[4]={A.x,A.y,A.z,A.w},b[4]={B.x,B.y,B.z,B.w};
    uint4 C{},D{};
    if constexpr(GATE) {C=*reinterpret_cast<const uint4*>(q0+k);D=*reinterpret_cast<const uint4*>(q1+k);}
    const uint32_t c[4]={C.x,C.y,C.z,C.w},d[4]={D.x,D.y,D.z,D.w};
    #pragma unroll
    for(int j=0;j<8;++j) {
        const float x=xv[j];
        if constexpr(FUSE) {
            y0=fmaf(x,bfj(a[j>>1],j),y0);y1=fmaf(x,bfj(b[j>>1],j),y1);
            if constexpr(GATE) {z0=fmaf(x,bfj(c[j>>1],j),z0);z1=fmaf(x,bfj(d[j>>1],j),z1);}
        } else {
            y0+=x*bfj(a[j>>1],j);y1+=x*bfj(b[j>>1],j);
            if constexpr(GATE) {z0+=x*bfj(c[j>>1],j);z1+=x*bfj(d[j>>1],j);}
        }
    }
}
template<int SRC,bool GATE,bool FUSE>
__global__ __launch_bounds__(skinny_threads) void skinny_gemm(GemmReq a) {
    const int wave=threadIdx.x>>5,lane=threadIdx.x&31;
    const int row=blockIdx.y*32+lane;
    const int c0=(blockIdx.x*4+wave)*skinny_tn,c1=c0+1;
    if(c0>=a.out)return;                       /* uniform, and before the first weight load */
    /* A ragged last column group keeps its second column's ADDRESS in bounds instead of predicating the
     * load: the dead column walks the live column's weight row and its store is dropped. */
    const uint16_t* p0=skinny_wrow(a,c0);
    const uint16_t* p1=c1<a.out?skinny_wrow(a,c1):p0;
    const uint16_t* q0=GATE?a.w1+int64_t(c0)*a.in:nullptr;
    const uint16_t* q1=GATE?(c1<a.out?a.w1+int64_t(c1)*a.in:q0):nullptr;
    float y0=0.0f,y1=0.0f,z0=0.0f,z1=0.0f;
    /* Unrolling lets later iterations' loads overlap the current chain; same arithmetic order.
     * Measured 2026-10-18: 4 beats 8 and 1, ties 2; gate-only 2 changed nothing. */
    constexpr int unroll=4;
    if(const void* src=skinny_row_src(a,row)) {
        if constexpr(SRC==0) {
            const float* fx=static_cast<const float*>(src);
            #pragma unroll unroll
            for(int k=0;k<a.in;k+=8) {
                float xv[8];
                *reinterpret_cast<float4*>(xv)=*reinterpret_cast<const float4*>(fx+k);
                *reinterpret_cast<float4*>(xv+4)=*reinterpret_cast<const float4*>(fx+k+4);
                skinny_k8<GATE,FUSE>(p0,p1,q0,q1,k,xv,y0,y1,z0,z1);
            }
        } else {
            const uint16_t* hx=static_cast<const uint16_t*>(src);
            #pragma unroll unroll
            for(int k=0;k<a.in;k+=8) {
                const uint2 u=*reinterpret_cast<const uint2*>(hx+k),v=*reinterpret_cast<const uint2*>(hx+k+4);
                const float xv[8]={bf(uint16_t(u.x)),bf(uint16_t(u.x>>16)),bf(uint16_t(u.y)),bf(uint16_t(u.y>>16)),
                    bf(uint16_t(v.x)),bf(uint16_t(v.x>>16)),bf(uint16_t(v.y)),bf(uint16_t(v.y>>16))};
                skinny_k8<GATE,FUSE>(p0,p1,q0,q1,k,xv,y0,y1,z0,z1);
            }
        }
    }
    if(row>=a.rows)return;
    float* dst=a.y+int64_t(row)*a.out+c0;
    if constexpr(GATE) {
        /* The silu epilogue, expression for expression from tree_gemm_gate: the same expf, the same
         * operation order. */
        dst[0]=(y0/(1.0f+expf(-y0)))*z0;
        if(c1<a.out)dst[1]=(y1/(1.0f+expf(-y1)))*z1;
    } else {
        dst[0]=y0;
        if(c1<a.out)dst[1]=y1;
    }
}
/* conv_pre's dyn GEMM and group convolution in ONE launch; the header carries the contract. Wave =
 * group g (4 waves/CTA), lane = row. The delta chain walks the FULL row (r.x + row*r.xstride) and
 * keeps the group's four columns in registers; only the convolution reads the group's own 16
 * elements (xc = xr + g*16). `two` is convolve_column's own taps>1, so a job's row 0 skips the
 * second tap pair instead of adding a zero term. */
template<bool FUSE>
__global__ __launch_bounds__(skinny_threads) void skinny_dyn(GemmReq r,const uint16_t* base,float* pre,int h,int b) {
    const int wave=threadIdx.x>>5,lane=threadIdx.x&31;
    const int ng=h/skinny_dyn_group;
    const int g=blockIdx.x*4+wave,row=blockIdx.y*32+lane;
    if(g>=ng||row>=r.rows)return;              /* the wave's group, and the tiled kernel's row guard */
    const uint16_t* const bw0=base+int64_t(g)*skinny_dyn_group;   /* tap 0, this group's 16 columns */
    const uint16_t* const bw1=bw0+h;                             /* tap 1: the base matrix is 2 rows of h */
    const uint16_t* dw[4]={r.w0+int64_t(g)*h,r.w0+int64_t(ng+g)*h,r.w0+int64_t(2*ng+g)*h,r.w0+int64_t(3*ng+g)*h};
    const float* const xr=r.x+int64_t(row)*r.xstride;             /* full row: what the delta chain reads */
    const float* const xc=xr+int64_t(g)*skinny_dyn_group;         /* this group: what the conv reads */
    const bool ok=r.valid[row/r.rows_per_seq]!=0;
    float dl[4]={0.0f,0.0f,0.0f,0.0f};
    if(ok) {
    #pragma unroll 4
    for(int k=0;k<h;k+=8) {
        float xv[8];
        *reinterpret_cast<float4*>(xv)=*reinterpret_cast<const float4*>(xr+k);
        *reinterpret_cast<float4*>(xv+4)=*reinterpret_cast<const float4*>(xr+k+4);
        skinny_k8<true,FUSE>(dw[0],dw[1],dw[2],dw[3],k,xv,dl[0],dl[1],dl[2],dl[3]);
    }
    }
    /* Every covered row stores all four columns, an invalid sequence's row as the zeros its zeroed
     * accumulator gives today; the finish stage reads rows 2,3 back out of w.delta. */
    float* const drow=r.y+int64_t(row)*int64_t(4*ng);
    drow[g]=dl[0];drow[ng+g]=dl[1];drow[2*ng+g]=dl[2];drow[3*ng+g]=dl[3];
    if(!ok)return;                             /* w.pre untouched: full_conv_pre returns before its store */
    const bool two=(row%b)!=0;
    const uint4 t0=ldu4(bw0),t1=ldu4(bw0+8),u0=ldu4(bw1),u1=ldu4(bw1+8);
    const uint32_t pl[4]={t0.x,t0.z,t1.x,t1.z},ph[4]={t0.y,t0.w,t1.y,t1.w};
    const uint32_t ql[4]={u0.x,u0.z,u1.x,u1.z},qh[4]={u0.y,u0.w,u1.y,u1.w};
    float* const prow=pre+int64_t(row)*h+int64_t(g)*skinny_dyn_group;
    #pragma unroll
    for(int q=0;q<4;++q) {
        const float4 x0=ld4(xc+q*4);
        const float4 x1=two?ld4(xc-h+q*4):x0;   /* never read when `two` is false */
        float4 o;
        o.x=conv_one(bf(uint16_t(pl[q])),x0.x,dl[0],bf(uint16_t(ql[q])),x1.x,dl[1],two);
        o.y=conv_one(bf(uint16_t(uint32_t(pl[q])>>16)),x0.y,dl[0],bf(uint16_t(uint32_t(ql[q])>>16)),x1.y,dl[1],two);
        o.z=conv_one(bf(uint16_t(ph[q])),x0.z,dl[0],bf(uint16_t(qh[q])),x1.z,dl[1],two);
        o.w=conv_one(bf(uint16_t(uint32_t(ph[q])>>16)),x0.w,dl[0],bf(uint16_t(uint32_t(qh[q])>>16)),x1.w,dl[1],two);
        st4(prow+q*4,o);
    }
}
}   /* anonymous namespace: this translation unit's own kernels */

namespace rad_dloop {
/* FUSE selects the accumulation of the kernel this one replaces: see skinny_k8's comment. The tree
 * policy (dloop_tree.hip) passes false, the base policy (dloop_kernels.hip) passes true. */
int skinny_gemm_launch(const GemmReq& r,bool fuse,hipStream_t stream) {
    GemmReq a=r;
    void* args[]={&a};
    const int bn=4*skinny_tn;
    const dim3 grid((a.out+bn-1)/bn,(a.rows+31)/32);
    const void* kernel;
    if(a.source==2)
        kernel=a.kind==3?(fuse?reinterpret_cast<const void*>(skinny_gemm<2,true,true>):reinterpret_cast<const void*>(skinny_gemm<2,true,false>))
                        :(fuse?reinterpret_cast<const void*>(skinny_gemm<2,false,true>):reinterpret_cast<const void*>(skinny_gemm<2,false,false>));
    else
        kernel=a.kind==3?(fuse?reinterpret_cast<const void*>(skinny_gemm<0,true,true>):reinterpret_cast<const void*>(skinny_gemm<0,true,false>))
                        :(fuse?reinterpret_cast<const void*>(skinny_gemm<0,false,true>):reinterpret_cast<const void*>(skinny_gemm<0,false,false>));
    return hipLaunchKernel(kernel,grid,dim3(skinny_threads),args,0,stream)==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
/* The fused conv producer. Grid (ceil(ng/4), ceil(rows/32)): one wave per group column, one lane per
 * row, exactly skinny_dyn's mapping. The predicate is re-tested here as well as in the caller, so an
 * unguarded call cannot launch a geometry the kernel would read out of bounds. */
int skinny_dyn_launch(const GemmReq& r,const DeviceArgs& a,int layer,int mlp,bool fuse,hipStream_t stream) {
    if(!skinny_dyn_fits(r,a.c))return RAD_E_UNSUPPORTED;
    GemmReq gr=r;
    const uint16_t* base=full_weight(a,mlp?DL_MLP_BASE:DL_ATTN_BASE,layer);
    float* pre=workspace(a).pre;
    int h=a.c.hidden_size,b=a.c.proposal_length+1;
    const int ng=r.in/skinny_dyn_group;
    void* args[]={&gr,&base,&pre,&h,&b};
    const dim3 grid((ng+3)/4,(r.rows+31)/32);
    const void* kernel=fuse?reinterpret_cast<const void*>(skinny_dyn<true>):reinterpret_cast<const void*>(skinny_dyn<false>);
    return hipLaunchKernel(kernel,grid,dim3(skinny_threads),args,0,stream)==hipSuccess?RAD_OK:RAD_E_DEVICE;
}
}   /* namespace rad_dloop */
