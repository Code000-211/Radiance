#include <hip/hip_runtime.h>
#include "tree_gdn.h"
#include "../libr4d/r4d_common.h"
#include "../libr4d/r4d_gdn_wmma.h"
#include "../libr4d/r4d_gdn_conv_tap.h"
#include "../libr4d/r4d_q8.h"
#include <cstdlib>
#include <cstring>

/* Cross-thread handoffs are LDS-only: no thread consumes another thread's
 * global writes in the same dispatch. In particular, recurrent replays read
 * LDS factors, not their global output. Factors, outputs and committed state/
 * history reach later kernels through ordered dispatch release/acquire fences,
 * not an intrakernel cache invalidate. */

namespace {
__device__ __forceinline__ float ld(const RadGdnTreeTensor& t,int a,int b=0,int c=0,int d=0) {
    return static_cast<const float*>(t.data)[rad_gdn_off(t,a,b,c,d)];
}
__device__ __forceinline__ float softplus(float x,float thr) {
    if (x>thr) return x;
    return x>0 ? x+__logf(1.0f+__expf(-x)) : __logf(1.0f+__expf(x));
}

/* THE BLOCK PROLOGUE: ONE VALIDATION, ONE ANCESTOR TABLE, ONE BARRIER EACH.
 *
 * rad_gdn_tree_metadata bounds every access, but it is the same answer for
 * every thread of the block, and it is a chain of scalar loads. Running it in
 * all 128 or 512 threads of every block costs a dependent-load chain on each
 * thread's critical path before any work starts; running it in thread 0 and
 * broadcasting through LDS costs one barrier. The check still precedes every
 * write, which is what the comment on the op demands.
 *
 * Walking PARENT[] upward is a second chain of dependent loads whose length is
 * the node's depth. Building the table once per block costs one parallel walk
 * and turns every later use into a plain shared-memory read. path[n][0] is node
 * n itself, path[n][i] its i-th ancestor; plen[n] counts both. par[] holds the
 * block's copy of PARENT[cu[seq]..+capacity).
 *
 * The first LDS barrier publishes the metadata verdict and par[]; the second
 * publishes path[] and plen[]. Neither handoff needs a global-memory fence. */
__device__ __forceinline__ int check_and_paths(const RadGdnTreeCall& c,int seq,char* okp,
                                               const RadGdnTreeTensor& pt,int bos,int nnode,int tid,
                                               signed char (&path)[RAD_GDN_TREE_CAP][RAD_GDN_TREE_CAP],
                                               signed char (&plen)[RAD_GDN_TREE_CAP],
                                               int (&par)[RAD_GDN_TREE_CAP]) {
    if (tid==0) *okp=(char)rad_gdn_tree_metadata(c,seq);
    /* NO PARENT READ BEFORE THE VERDICT. A rejected batch was never validated,
     * so every range can be hostile: bos can be -1 (cu=-1) or past `tokens`
     * (cu clipped only by the caller's own refusal), nnode can be negative or
     * past capacity, and rad_gdn_parent would read the parent tensor anywhere
     * in that range. The load happens only when the WHOLE range is in bounds,
     * and even then a negative verdict stops the path walk before it reads. */
    const bool bounded = bos>=0 && bos<=c.tokens && nnode>=0 && nnode<=RAD_GDN_TREE_CAP
                         && bos<=c.tokens-nnode;
    if (bounded && tid<nnode) par[tid]=rad_gdn_parent(pt,bos,tid);
    else if (tid<RAD_GDN_TREE_CAP) par[tid]=-1;
    /* ONE fence for par[] AND the answer: ok is written before it and read after
     * it, and the ancestor walk below already sees the table it needs. */
    lds_barrier();
    const int ok=*okp;
    if (!ok) return 0;                 /* rejected: no path walk, no path writes */
    if (tid<nnode) {
        int n=0,p=tid;
        while (p>=0 && n<RAD_GDN_TREE_CAP) { path[tid][n++]=(signed char)p;p=par[p]; }
        plen[tid]=(signed char)n;
    }
    lds_barrier();
    return ok;
}

/* ANCESTOR LISTS, BUILT ONCE PER BLOCK.
 *
 * Walking PARENT[] upward is a chain of dependent loads whose length is the
 * node's depth. Building the list once per block costs one walk in parallel
 * across the block and turns every later use into a plain shared-memory read.
 * path[n][0] is node n itself, path[n][i] its i-th ancestor; plen[n] counts
 * both. par[] holds the block's copy of PARENT[cu[seq]..+capacity). */
/* 128-bit accesses where a run of columns is contiguous and 16-byte aligned,
 * scalar otherwise. Dense operands hit the vector path; a strided projection
 * slice still has contiguous 32-float runs per (head, quarter). */
__device__ __forceinline__ bool vec32(const void* p,int64_t stride) {
    return stride==1 && (reinterpret_cast<uintptr_t>(p)&15)==0;
}
__device__ __forceinline__ void cp32(float (&h)[32],const float (&src)[32]) {
#pragma unroll
    for (int j=0;j<32;++j) h[j]=src[j];
}
/* One saved node of the rank-one update, applied to this thread's 32 columns:
 * h = h * decay + u k^T, from the block's LDS copy of that node's factors.
 * kcol is the node's 128 normalized-k columns, urow the owning row's u. */
__device__ __forceinline__ void replay_lds(float (&h)[32],const float (&kcol)[128],float u,
                                           float decay,int qtr) {
    const float4* const kp=reinterpret_cast<const float4*>(kcol+qtr*32);
#pragma unroll
    for (int j=0;j<8;++j) {
        const float4 w=kp[j];
        h[4*j]=fmaf(u,w.x,h[4*j]*decay);h[4*j+1]=fmaf(u,w.y,h[4*j+1]*decay);
        h[4*j+2]=fmaf(u,w.z,h[4*j+2]*decay);h[4*j+3]=fmaf(u,w.w,h[4*j+3]*decay);
    }
}
/* The state row slice a thread owns, indexed by the tensor's column stride. */
__device__ __forceinline__ void ld32(float (&h)[32],const float* p,int64_t stride) {
    if (vec32(p,stride)) {
#pragma unroll
        for (int j=0;j<32;j+=4) {
            const float4 w=*reinterpret_cast<const float4*>(p+j*stride);
            h[j]=w.x;h[j+1]=w.y;h[j+2]=w.z;h[j+3]=w.w;
        }
    } else {
#pragma unroll
        for (int j=0;j<32;++j) h[j]=p[j*stride];
    }
}
__device__ __forceinline__ void st32(float* p,int64_t stride,const float (&h)[32]) {
    if (vec32(p,stride)) {
#pragma unroll
        for (int j=0;j<32;j+=4)
            *reinterpret_cast<float4*>(p+j*stride)=make_float4(h[j],h[j+1],h[j+2],h[j+3]);
    } else {
#pragma unroll
        for (int j=0;j<32;++j) p[j*stride]=h[j];
    }
}
/* A row of BF16 in, 32 floats out. */
__device__ __forceinline__ void ldbf32(float (&h)[32],const uint16_t* p,int64_t stride) {
    if (vec32(p,stride)) {
#pragma unroll
        for (int j=0;j<32;j+=8) {
            const uint4 w=*reinterpret_cast<const uint4*>(p+j*stride);
            const uint16_t* e=reinterpret_cast<const uint16_t*>(&w);
#pragma unroll
            for (int b=0;b<8;++b) h[j+b]=bf2f(e[b]);
        }
    } else {
#pragma unroll
        for (int j=0;j<32;++j) h[j]=bf2f(p[j*stride]);
    }
}

/* Parent-aware BF16 causal convolution, one thread per channel.
 *
 * TAP OUTER, NODE INNER, WITH THE NODE LOOP FULLY UNROLLED. That order reads
 * each of the width weights once for all sixteen nodes instead of once per
 * node, and the unrolled node loop keeps the sixteen accumulators in registers
 * with static indices instead of a roll of dynamically indexed ones. */
__global__ __launch_bounds__(128) void conv_kernel(RadGdnTreeCall c) {
    const int seq=blockIdx.x,ch=blockIdx.y*128+threadIdx.x,d=128*(2*c.hk+c.hv);
    __shared__ signed char s_path[RAD_GDN_TREE_CAP][RAD_GDN_TREE_CAP],s_plen[RAD_GDN_TREE_CAP];
    __shared__ char s_ok;
    __shared__ int s_par[RAD_GDN_TREE_CAP];
    const int bos=rad_gdn_int(c.t[5],seq),end=rad_gdn_int(c.t[5],seq+1);
    const int slot=rad_gdn_int(c.t[4],seq),ctx=rad_gdn_int(c.t[7],seq);
    const int nnode=end-bos;
    if (!check_and_paths(c,seq,&s_ok,c.t[6],bos,nnode,threadIdx.x,s_path,s_plen,s_par)) return;
    if (ch>=d) return;
    const uint16_t* const xp=static_cast<const uint16_t*>(c.t[0].data)+rad_gdn_off(c.t[0],bos,ch);
    const uint16_t* const wp=static_cast<const uint16_t*>(c.t[1].data)+rad_gdn_off(c.t[1],ch,0);
    const uint16_t* const hp=static_cast<const uint16_t*>(c.t[3].data)+rad_gdn_off(c.t[3],slot,ch,0);
    const int64_t xs=c.t[0].s[0],wi=c.t[1].s[1],hi=c.t[3].s[2];
    const float bias=c.t[2].data ? bf2f(static_cast<const uint16_t*>(c.t[2].data)[rad_gdn_off(c.t[2],ch)]) : 0;
    /* Which of q/k/v this channel feeds, and the raw-input copy beside it. */
    const int which=ch<c.hk*128 ? 8 : ch<2*c.hk*128 ? 9 : 10;
    const RadGdnTreeTensor& ot=c.t[which];
    uint16_t* const op=const_cast<uint16_t*>(static_cast<const uint16_t*>(ot.data))+
        rad_gdn_off(ot,bos,which==8 ? ch : which==9 ? ch-c.hk*128 : ch-2*c.hk*128);
    uint16_t* const rp=static_cast<uint16_t*>(c.t[11].data)+rad_gdn_off(c.t[11],bos,ch);
    const int64_t os=ot.s[0],rs=c.t[11].s[0];
    float acc[RAD_GDN_TREE_CAP];
#pragma unroll
    for (int node=0;node<RAD_GDN_TREE_CAP;++node) acc[node]=bias;
    for (int tap=0;tap<c.width;++tap) {
        const float w=bf2f(wp[(int64_t)tap*wi]);
        const int lag=c.width-1-tap;
        /* PREDICATED, NOT BROKEN OUT OF. A break on a runtime node count makes
         * the sixteen loads of one tap a serial chain -- LDS path read, then a
         * global load that waits on it, then the FMA -- and a channel only
         * walks four taps, so that chain is most of the kernel. Predicating on
         * the same counts instead leaves the compiler sixteen independent
         * chains to overlap, and the store loop below still skips the nodes a
         * sequence does not have, so nothing extra is written. */
#pragma unroll
        for (int node=0;node<RAD_GDN_TREE_CAP;++node) {
            const int n=s_plen[node],pos=ctx+n-1-lag;
            float x=0;
            if (node<nnode) {
                if (lag<n) x=bf2f(xp[(int64_t)s_path[node][lag]*xs]);
                else if (pos>=0) x=bf2f(hp[(int64_t)(c.ring ? pos%c.hist_len : c.width-1+n-1-lag)*hi]);
            }
            acc[node]=fmaf(w,x,acc[node]);
        }
    }
#pragma unroll
    for (int node=0;node<RAD_GDN_TREE_CAP;++node) {
        if (node>=nnode) break;
        op[(int64_t)node*os]=f2bf(cp_silu(acc[node]));
        rp[(int64_t)node*rs]=xp[(int64_t)node*xs];
    }
}

/* One block owns one head of one sequence, and every thread owns a QUARTER of a
 * state row: 128 rows x 128 columns over 512 threads is 32 floats a thread, so
 * the whole F32 state stays in registers with no spill and no per-node state
 * traffic. The old shape put 64 floats of state, 64 of q and 64 of k in one
 * thread -- a 256-VGPR kernel at 5 waves/SIMD that spilled 596 bytes to scratch.
 *
 * THE COMMITTED STATE IS READ ONCE, AND EVERY REPLAY READS LDS, NOT MEMORY.
 * That is what this kernel is about. The committed state is immutable during
 * verification, so the anchor it holds is the base for EVERY branch, and each
 * branch copies the registers and replays its ancestors' saved factors. The
 * golden tree's 16 nodes contain 14 branches whose ancestor chains total 63
 * replays, and a replay that reads its factors from global memory -- which is
 * what this kernel used to do -- is 63 x 64 KB = 4 MB per block: 387 MB per
 * GDN layer per step, at the last-level cache's bandwidth rather than a
 * register's. Each node's normalized k, its u and its exp(g) therefore stay in
 * shared memory as this block walks the tree, so a replay costs 32 FMAs and a
 * short LDS read instead of an L1/L2 round trip.
 *
 * The factors still go to their declared global output, unchanged: commit reads
 * them there, and no value differs for having been read back from LDS. */
__global__ __launch_bounds__(512) void recurrent_kernel(RadGdnTreeCall c) {
    const int seq=blockIdx.x,head=blockIdx.y,tid=threadIdx.x,row=tid>>2,qtr=tid&3;
    /* TWO PARTIAL BUFFERS, ONE BARRIER A NODE. The node's barrier must both
     * publish this node's factors and fence the sixteen wave partials between
     * their writer and their readers. A single buffer needs a SECOND barrier
     * after the reads, or the next node's writes land on partials a slow wave
     * has not read yet. Alternating the buffer by node parity removes that
     * hazard: node n+2 rewrites the buffer node n read, and every thread has
     * passed node n+1's barrier by then, so all of node n's reads are done. */
    __shared__ float wave_sum[2][RAD_GDN_TREE_CAP];
    /* replay_lds reads the k columns as float4, so the staging bands are 16-byte aligned. */
    __shared__ __align__(16) float s_k[RAD_GDN_TREE_CAP][128],s_u[RAD_GDN_TREE_CAP][128],s_g[RAD_GDN_TREE_CAP];
    __shared__ signed char s_path[RAD_GDN_TREE_CAP][RAD_GDN_TREE_CAP],s_plen[RAD_GDN_TREE_CAP];
    __shared__ char s_ok;
    __shared__ int s_par[RAD_GDN_TREE_CAP];
    const int bos=rad_gdn_int(c.t[9],seq),end=rad_gdn_int(c.t[9],seq+1),slot=rad_gdn_int(c.t[8],seq);
    const int nnode=end-bos,key_head=head/(c.hv/c.hk);
    if (!check_and_paths(c,seq,&s_ok,c.t[10],bos,nnode,tid,s_path,s_plen,s_par)) return;
    const float alog=__expf(ld(c.t[5],head)),dtb=ld(c.t[6],head),nw=ld(c.t[12],row);
    const uint16_t* const qrow=static_cast<const uint16_t*>(c.t[0].data)+rad_gdn_off(c.t[0],0,key_head*128+qtr*32);
    const uint16_t* const krow=static_cast<const uint16_t*>(c.t[1].data)+rad_gdn_off(c.t[1],0,key_head*128+qtr*32);
    const uint16_t* const vrow=static_cast<const uint16_t*>(c.t[2].data)+rad_gdn_off(c.t[2],0,head*128+row);
    const uint16_t* const arow=static_cast<const uint16_t*>(c.t[3].data)+rad_gdn_off(c.t[3],0,head);
    const uint16_t* const brow=static_cast<const uint16_t*>(c.t[4].data)+rad_gdn_off(c.t[4],0,head);
    const uint16_t* const zrow=static_cast<const uint16_t*>(c.t[11].data)+rad_gdn_off(c.t[11],0,head*128+row);
    uint16_t* const orow=static_cast<uint16_t*>(c.t[13].data)+rad_gdn_off(c.t[13],0,head*128+row);
    const int64_t qs=c.t[0].s[0],ks=c.t[1].s[0],vs=c.t[2].s[0],as=c.t[3].s[0],bs=c.t[4].s[0],
        zs=c.t[11].s[0],os=c.t[13].s[0],fi=c.t[14].s[2];
    const int64_t qi=c.t[0].s[1],ki=c.t[1].s[1];
    float* const fs=static_cast<float*>(c.t[14].data);
    /* The one committed-state read of this block. */
    float anchor[32],h[32];
    ld32(anchor,static_cast<const float*>(c.t[7].data)+rad_gdn_off(c.t[7],slot,head,row,qtr*32),c.t[7].s[3]);
    for (int node=0;node<nnode;++node) {
        const int tok=bos+node;
        if (node==0) cp32(h,anchor);
        else if (s_par[node]!=node-1) {
            /* A branch: back to the anchor in registers, then its ancestors
             * farthest first, which is the order they were carried in. */
            cp32(h,anchor);
            for (int i=s_plen[node]-1;i>=1;--i) {
                const int a=s_path[node][i];
                replay_lds(h,s_k[a],s_u[a][row],s_g[a],qtr);
            }
        }
        const uint16_t* const qr=qrow+(int64_t)tok*qs;
        const uint16_t* const kr=krow+(int64_t)tok*ks;
        float q[32],k[32],sq=0,sk=0;
        ldbf32(q,qr,qi);ldbf32(k,kr,ki);
#pragma unroll
        for (int j=0;j<32;++j) { sq=fmaf(q[j],q[j],sq);sk=fmaf(k[j],k[j],sk); }
        /* Four lanes hold one row's columns, so the norm closes with two
         * exchanges instead of the old half-to-half one. */
        sq+=r4d_xor_lane<1>(sq);sq+=r4d_xor_lane<2>(sq);
        sk+=r4d_xor_lane<1>(sk);sk+=r4d_xor_lane<2>(sk);
        const float qscl=__frsqrt_rn(sq+c.l2_eps)*c.scale,kscl=__frsqrt_rn(sk+c.l2_eps);
#pragma unroll
        for (int j=0;j<32;++j) { q[j]*=qscl;k[j]*=kscl; }
        const float g=-alog*softplus(bf2f(arow[tok*as])+dtb,c.softplus_thr);
        const float decay=__expf(g),beta=1.0f/(1.0f+__expf(-bf2f(brow[tok*bs])));
        float dot=0;
#pragma unroll
        for (int j=0;j<32;++j) { h[j]*=decay;dot=fmaf(h[j],k[j],dot); }
        dot+=r4d_xor_lane<1>(dot);dot+=r4d_xor_lane<2>(dot);
        const float u=(bf2f(vrow[tok*vs])-dot)*beta;
        /* The declared factor output, and this block's own copy for its later
         * replays. Both hold the identical F32 values. */
        float* const f=fs+rad_gdn_off(c.t[14],tok,head,0);
        if (row==0) {
#pragma unroll
            for (int j=0;j<32;++j) { f[(qtr*32+j)*fi]=k[j];s_k[node][qtr*32+j]=k[j]; }
        }
        if (qtr==0) { f[(128+row)*fi]=u;s_u[node][row]=u; }
        if (tid==0) { f[256*fi]=decay;s_g[node]=decay; }
        float on=0;
#pragma unroll
        for (int j=0;j<32;++j) { h[j]=fmaf(u,k[j],h[j]);on=fmaf(h[j],q[j],on); }
        on+=r4d_xor_lane<1>(on);on+=r4d_xor_lane<2>(on);
        /* One lane a row, then the warp's rows, then the block's sixteen
         * warps: the same descending exchange the committed norm uses. */
        float rms=(qtr==0) ? on*on : 0;
        rms+=r4d_xor_lane<4>(rms);rms+=r4d_xor_lane<8>(rms);rms+=r4d_xor_lane<16>(rms);
        if ((tid&31)==0) wave_sum[node&1][tid>>5]=rms;
        lds_barrier();
        float total=0;
#pragma unroll
        for (int i=0;i<RAD_GDN_TREE_CAP;++i) total+=wave_sum[node&1][i];
        const float sc=__frsqrt_rn(total*(1.0f/128)+c.eps);
        const float z=bf2f(zrow[tok*zs]),sig=1.0f/(1.0f+__expf(-z));
        on*=sc*nw*(c.sigmoid ? sig : z*sig);
        if (qtr==0) orow[(int64_t)tok*os]=f2bf(on);
        /* ONE SYNC A NODE. This is the barrier that publishes this node's
         * factors to the branches that replay them and this node's sixteen wave
         * partials to the readers below. The partial buffer alternates with the
         * node, so by the time node n+2 overwrites what node n read, every
         * thread has passed node n+1's barrier and every read of node n's
         * partials is done -- the second barrier the single buffer needed is
         * gone, sixteen block-wide syncs saved over the walk. */
    }
}

#include "tree_gdn_fused_i8g.h"

/* Root-inclusive winner replay and raw convolution history commit. The state
 * arm uses the same quarter-row shape as the recurrent kernel, so a committed
 * path costs one state read and one state write per sequence and head. */
__global__ __launch_bounds__(512) void commit_kernel(RadGdnTreeCall c) {
    const int seq=blockIdx.x,unit=blockIdx.y,tid=threadIdx.x;
    /* Commit is all-or-none even for invalid metadata in a different sequence.
     * Normal submission already validated it; this bounds direct op callers.
     *
     * Thread 0 checks every sequence before publishing the LDS verdict. Every
     * thread observes that verdict before any state/history write; global
     * metadata is already visible through the dispatch acquire fence. */
    __shared__ char s_ok;
    if (tid==0) {
        char ok=1;
        for (int s=0;s<c.nseq;++s) if (!rad_gdn_tree_metadata(c,s)) { ok=0;break; }
        s_ok=ok;
    }
    lds_barrier();
    if (!s_ok) return;
    const int bos=rad_gdn_int(c.t[9],seq),len=rad_gdn_int(c.t[7],seq);
    if (unit<c.hv) {
        const int slot=rad_gdn_int(c.t[1],seq),row=tid>>2,qtr=tid&3;
        const RadGdnTreeTensor& st_state=c.t[0];
        const int64_t col=st_state.s[3];
        float* const base=static_cast<float*>(st_state.data)+rad_gdn_off(st_state,slot,unit,row,0)+
            (int64_t)qtr*32*col;
        const int64_t fi=c.t[2].s[2];
        const float* const fs=static_cast<const float*>(c.t[2].data);
        /* The winner path's factors, staged in tiles of RAD_GDN_TREE_CAP nodes:
         * the 257 values of one node are contiguous, so staging by linear index
         * reads them coalesced. Tiles are indexed by tile-local ordinal, never by
         * node ID, and h stays in registers across tiles, so a wide path replays
         * in the same FP32 order without a per-node state. A path of at most
         * RAD_GDN_TREE_CAP nodes is one tile: one stage and one barrier, as before;
         * each further tile adds a barrier so no thread restages a bank another
         * thread is still replaying. */
        __shared__ signed char s_win[RAD_GDN_TREE_WIDE_CAP];
        __shared__ __align__(16) float s_k[RAD_GDN_TREE_CAP][128],s_u[RAD_GDN_TREE_CAP][128],s_g[RAD_GDN_TREE_CAP];
        if (tid<len) s_win[tid]=(signed char)rad_gdn_int(c.t[6],seq,tid);
        lds_barrier();
        float h[32];
        ld32(h,base,col);
        for (int first=0;first<len;first+=RAD_GDN_TREE_CAP) {
            const int n=len-first<RAD_GDN_TREE_CAP ? len-first : (int)RAD_GDN_TREE_CAP;
            if (first) lds_barrier();
            for (int e=tid;e<n*257;e+=512) {
                const int i=e/257,j=e%257;
                const float v=fs[rad_gdn_off(c.t[2],bos+s_win[first+i],unit,0)+(int64_t)j*fi];
                if (j<128) s_k[i][j]=v;
                else if (j<256) s_u[i][j-128]=v;
                else s_g[i]=v;
            }
            lds_barrier();
            for (int i=0;i<n;++i) replay_lds(h,s_k[i],s_u[i][row],s_g[i],qtr);
        }
        st32(base,col,h);
    } else {
        const int ch=(unit-c.hv)*512+tid,d=128*(2*c.hk+c.hv);
        if (ch>=d) return;
        const int slot=rad_gdn_int(c.t[4],seq),ctx=rad_gdn_int(c.t[8],seq),w=c.width-1;
        const int64_t hi=c.t[3].s[2],rs=c.t[5].s[0];
        uint16_t* const hist=static_cast<uint16_t*>(c.t[3].data)+rad_gdn_off(c.t[3],slot,ch,0);
        const uint16_t* const rawp=static_cast<const uint16_t*>(c.t[5].data)+rad_gdn_off(c.t[5],bos,ch);
        if (c.ring) {
            for (int i=0;i<len;++i)
                hist[((ctx+i)%c.hist_len)*hi]=rawp[(int64_t)rad_gdn_int(c.t[6],seq,i)*rs];
        } else if (w) {
            uint16_t win[16];
            for (int i=0;i<w;++i) win[i]=ctx-w+i<0 ? 0 : hist[i*hi];
            for (int j=0;j<len;++j) {
                for (int i=0;i<w-1;++i) win[i]=win[i+1];
                win[w-1]=rawp[(int64_t)rad_gdn_int(c.t[6],seq,j)*rs];
            }
            for (int i=0;i<w;++i) hist[i*hi]=win[i];
        }
    }
}

/* The serial conv/recurrent kernels hold RAD_GDN_TREE_CAP-node LDS tables;
 * wider geometry is refused before launch, including direct launcher calls. */
int conv_launch(const RadArgs* a,RadStream stream) {
    RadGdnTreeCall c{};const int rc=rad_gdn_tree_args(a,0,&c);if (rc!=RAD_OK) return rc;
    if (c.capacity>RAD_GDN_TREE_CAP) return RAD_E_UNSUPPORTED;
    void* args[]={&c};
    return hipLaunchKernel((const void*)conv_kernel,dim3(c.nseq,2*c.hk+c.hv),dim3(128),
                           args,0,(hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int recurrent_launch(const RadArgs* a,RadStream stream) {
    RadGdnTreeCall c{};const int rc=rad_gdn_tree_args(a,1,&c);if (rc!=RAD_OK) return rc;
    if (c.capacity>RAD_GDN_TREE_CAP) return RAD_E_UNSUPPORTED;
    void* args[]={&c};
    return hipLaunchKernel((const void*)recurrent_kernel,dim3(c.nseq,c.hv),dim3(512),
                           args,0,(hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int commit_launch(const RadArgs* a,RadStream stream) {
    RadGdnTreeCall c{};const int rc=rad_gdn_tree_args(a,2,&c);if (rc!=RAD_OK) return rc;
    const int d=128*(2*c.hk+c.hv);
    void* args[]={&c};
    return hipLaunchKernel((const void*)commit_kernel,dim3(c.nseq,c.hv+(d+511)/512),dim3(512),
                           args,0,(hipStream_t)stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
} // namespace
int rad_mine_gdn_tree_count(void) { return RAD_GDN_TREE_OPS; }
const RadKernelInfo* rad_mine_gdn_tree_at(int i) {
    // Resolve once when the plugin manifest is loaded, never on the decode step path.
    const char* mode=std::getenv("RADIANCE_GDN_BOLE_EXACT");
    if (mode && std::strcmp(mode,"0") && std::strcmp(mode,"1")) return nullptr;
    if (i==RAD_GDN_TREE_FUSED_I8G && mode && !std::strcmp(mode,"1")) {
        static const RadKernelInfo exact=[] {
            auto r=rad_gdn_tree_row(RAD_GDN_TREE_FUSED_I8G,RAD_DOMAIN_DEVICE,fused_i8g_launch<true>,true);
            r.name="mine_gdn_bole_exact_i8g";
            r.computes="Exact node-parallel FP32 path replay; fused convolution, recurrence and INT8 group128";
            return r;
        }();
        return &exact;
    }
    static const RadKernelInfo rows[]={rad_gdn_tree_row(0,RAD_DOMAIN_DEVICE,conv_launch,false),
        rad_gdn_tree_row(1,RAD_DOMAIN_DEVICE,recurrent_launch,false),
        rad_gdn_tree_row(2,RAD_DOMAIN_DEVICE,commit_launch,true),
        rad_gdn_tree_row(RAD_GDN_TREE_FUSED_I8G,RAD_DOMAIN_DEVICE,fused_i8g_launch<false>,false)};
    return i>=0 && i<RAD_GDN_TREE_OPS ? rows+i : nullptr;
}
