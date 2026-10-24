#ifndef RAD_MINE_DLOOP_FULL_DEVICE_H
#define RAD_MINE_DLOOP_FULL_DEVICE_H

/* Private device-side core of the DLoop drafter, shared by mine/dloop_kernels.hip and
 * mine/dloop_tree.hip. NOT a public interface: nothing in abi/ or the engine includes it. The op
 * contracts are mine/dloop_ops.h (ring ops, unchanged) and mine/dloop_full.h (full-context ops,
 * owned by worker FullContract).
 *
 * Why this file exists:
 *   - the ring operand table, arena and elementwise helpers used to be kept identical in the two
 *     translation units by a comment. They are defined once now, so the full-context ops cannot
 *     drift from the ring ops.
 *   - the whole full-context stage sequence (embed -> per-layer conv/QKV/attention/MLP -> finish)
 *     and the whole append stage sequence are written ONCE and parameterised over a small policy
 *     that each translation unit implements with its own optimized GEMM. The five-layer tree path
 *     therefore adds no second backbone and no second attention.
 *
 * FULL-CONTEXT RULES (docs/DLOOP_FULL_CONTEXT_PLAN.md, mine/dloop_full.h):
 *   - `capacity` (C) is an ALLOCATION bound, never a sliding window; nothing is truncated at C.
 *   - the cache is F32 [Q,L,C,2D] in rad_dloop_full_kv_index's layout: for one (slot, layer,
 *     position) the first D values are K, already rotated at that position's ABSOLUTE index, and the
 *     next D are V. Never BF16, never requantised.
 *   - attention streams over the cache operand. There is NO C-sized register array, no C-sized LDS
 *     array, and no copy of the cache into scratch anywhere in this file.
 *   - block Q/K are rotated starting at length[slot]; a cached K was rotated when it was appended.
 *   - the append writes only its own new rows per layer; base and tree never write the cache. A
 *     refused batch leaves EVERY operand untouched: the batch refusal flag is published before the
 *     first mutating kernel runs, and every one of them tests it before its first store.
 */

#include <cmath>
#include <cstdint>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

#include "dloop_ops.h"
#include "dloop_full.h"

namespace rad_dloop {

constexpr int threads=128;          /* ring CTA width; also the tree's block_threads */
constexpr int block_threads=threads;
constexpr int max_nodes=121, max_frontier=15;
/* New verified tap rows the full append stages per internal step. The batch's whole new-row set is
 * walked in these chunks, so the append's arena is bounded by the chunk and NOT by tap_rows: a
 * 4096-row prefill costs the same scratch as a 1-row decode step, it just takes more steps. */
constexpr int chunk_rows=128;

/* The full-context ops run under their own tags, distinct from every ring op id, so no kernel can
 * mistake one path's operand table for the other's. `capacity != 0` marks the same thing and is what
 * selects the arena layout. */
enum { FULL_APPEND=8, FULL_BASE=9, FULL_TREE=10 };

struct DeviceArgs {
    RadDLoopConfig c;
    int capacity;               /* C: slot cache capacity; 0 for the ring ops */
    void* t[18];
    int offset[5][20];          /* weight blob offsets per layer (the full path has 5 layers) */
    float* scratch;             /* F32 arena, >= full_elements / ring_work_elements for the op */
    int op;
};

__host__ __device__ inline int frontier_bound(const RadDLoopConfig& c) {
    if(!c.pruned)return (c.budget+c.proposal_length-1)/c.proposal_length;
    int n=c.width<c.budget?c.width:c.budget;
    return n<c.pool_size?n:c.pool_size;
}

/* ---------------------------------------------------------------- per-sequence arena
 *
 * Field order and meaning are identical for both paths, so a kernel body never branches on which one
 * it runs under. `stage`/`ctl` exist for the full append, `keys` for the full base. */
struct Work {
    float *block,*norm,*pre,*out,*qkv,*ff,*delta,*final,*y,*rank,*scores,*ctxkv,*node_score;
    int *tok,*parent,*depth,*path,*remap,*frontier,*count,*valid;
    float* stage;               /* full append: staged tap features, projected rows, then their K|V */
    float* ctl;                 /* full append: the batch control block (see AppendCtl) */
    float* keys;                /* full base: the head pass's per-row candidate rank keys */
    float* rope;                /* full base/tree: per-sequence absolute-position cos/sin pairs */
    int64_t total;              /* F32 elements the layout consumed: the arena size, not a field */
};

__host__ __device__ inline int64_t ring_arena_elements(const RadDLoopConfig& c,int op) {
    const int64_t jobs=int64_t(c.n_seq)*(op==RAD_DLOOP_BASE?1:frontier_bound(c)),b=c.proposal_length+1;
    const int64_t h=c.hidden_size,kv=c.num_kv_heads*c.head_dim;
    return jobs*(4*b*h+b*(h+2*kv)+b*c.intermediate_size+b*4*(h/c.conv_group_size)+
        c.proposal_length*h+c.target_width+c.selector_rank+c.pool_size)+
        int64_t(c.n_seq)*(c.num_layers*c.context_window*2*kv+max_nodes*(5+c.proposal_length)+max_frontier+2);
}
__host__ __device__ inline Work ring_workspace(const DeviceArgs& a) {
    const auto& c=a.c;
    const int64_t jobs=int64_t(c.n_seq)*(a.op==RAD_DLOOP_BASE?1:frontier_bound(c)),b=c.proposal_length+1,h=c.hidden_size;
    const int64_t kv=c.num_kv_heads*c.head_dim,n=int64_t(c.n_seq)*max_nodes;
    float* p=a.scratch;Work w{};
    w.block=p;p+=jobs*b*h;w.norm=p;p+=jobs*b*h;w.pre=p;p+=jobs*b*h;w.out=p;p+=jobs*b*h;
    w.qkv=p;p+=jobs*b*(h+2*kv);w.ff=p;p+=jobs*b*c.intermediate_size;
    w.delta=p;p+=jobs*b*4*(h/c.conv_group_size);w.final=p;p+=jobs*c.proposal_length*h;
    w.y=p;p+=jobs*c.target_width;w.rank=p;p+=jobs*c.selector_rank;w.scores=p;p+=jobs*c.pool_size;
    w.ctxkv=p;p+=int64_t(c.n_seq)*c.num_layers*c.context_window*2*kv;
    w.node_score=p;p+=n;w.tok=reinterpret_cast<int*>(p);p+=n;
    w.parent=reinterpret_cast<int*>(p);p+=n;w.depth=reinterpret_cast<int*>(p);p+=n;
    w.path=reinterpret_cast<int*>(p);p+=n*c.proposal_length;w.remap=reinterpret_cast<int*>(p);p+=n;
    w.frontier=reinterpret_cast<int*>(p);p+=c.n_seq*max_frontier;
    w.count=reinterpret_cast<int*>(p);p+=c.n_seq;w.valid=reinterpret_cast<int*>(p);p+=c.n_seq;
    w.stage=nullptr;w.ctl=nullptr;w.keys=nullptr;
    w.total=p-a.scratch;
    return w;
}
__host__ __device__ inline int64_t ring_work_elements(const RadDLoopConfig& c,int op) {
    const int64_t own=ring_arena_elements(c,op),oracle=int64_t(c.n_seq)*rad_dloop_scratch_elements(c);
    return own>oracle?own:oracle;
}

/* Head CTAs per row: the head pass emits one 64-key list per CTA and the merge reads exactly
 * rows*lists*64 keys, so that product is what the full base's arena has to hold. */
__host__ __device__ inline int64_t head_lists(int64_t shard) {return (shard+127)/128;}

/* Ints the append's control block needs: six n-sized arrays (ylo, yhi, len0, slot, limit, ok),
 * rowoff (n+1), bad (1), live (1) and the per-staged-row validity table (chunk_rows+1).
 *   6n + (n+1) + 1 + 1 + (chunk_rows+1) = 7n + chunk_rows + 4
 * append_ctl advances the cursor by exactly those increments and records where it stopped in
 * AppendCtl::used, so a field added there without this formula being updated is caught by the
 * regression (used <= append_ctl_ints) instead of silently overwriting the next arena region. */
__host__ __device__ inline int64_t append_ctl_ints(int n_seq) {
    return 7*int64_t(n_seq)+int64_t(chunk_rows)+4;
}

/* Full-context arena, F32 elements for the WHOLE BATCH -- the ring path's convention, where one arena
 * holds every sequence and every per-sequence state array is dimensioned by n_seq. The launch
 * requires scratch_bytes >= 4 * full_elements(...); the wrapper also checks the row's own hook.
 *
 * The layout in full_workspace is the ONLY definition of this size: full_elements runs that same
 * function and returns the bytes it advanced, so a field added to one can never be missed by the
 * other. The append's staging is bounded by `chunk_rows`; base and tree allocate nothing that grows
 * with C -- in particular there is NO num_layers*C*2*D context-KV block, because the cache operand
 * IS the storage. base's head-pass candidate keys are the one term the host reference does not have.
 *
 * The control block needs no alignment: the cache offsets in it are split into low/high ints rather
 * than held as int64, so every field is an int32 and the float arena's own alignment suffices. */
__host__ __device__ inline Work full_workspace(const DeviceArgs& a) {
    const auto& c=a.c;
    const int64_t H=c.hidden_size,D=int64_t(c.num_kv_heads)*c.head_dim,n=int64_t(c.n_seq);
    int64_t used=0;Work w{};
    auto take=[&](int64_t count) {
        // Sizing calls have no arena; integer offsets avoid arithmetic on a null pointer.
        auto* region=reinterpret_cast<float*>(uintptr_t(a.scratch)+uintptr_t(used)*sizeof(float));
        used+=count;return region;
    };
    if(a.op==FULL_APPEND) {
        w.stage=take(int64_t(chunk_rows)*c.context_features_dim);
        w.norm=take(int64_t(chunk_rows)*H);
        w.qkv=take(int64_t(chunk_rows)*2*D);
        w.ctl=take(append_ctl_ints(n));
        w.total=used;
        return w;
    }
    const int64_t B=int64_t(c.proposal_length)+1;
    const int64_t front=frontier_bound(c),jobs=n*(a.op==FULL_BASE?1:front),nodes=n*max_nodes;
    w.block=take(jobs*B*H);w.norm=take(jobs*B*H);w.pre=take(jobs*B*H);w.out=take(jobs*B*H);
    w.qkv=take(jobs*B*(H+2*D));w.ff=take(jobs*B*c.intermediate_size);
    w.delta=take(jobs*B*4*(H/c.conv_group_size));w.final=take(jobs*c.proposal_length*H);
    w.y=take(jobs*c.target_width);w.rank=take(jobs*c.selector_rank);w.scores=take(jobs*c.pool_size);
    w.rope=take(n*B*c.head_dim);
    w.ctxkv=nullptr;
    w.node_score=take(nodes);w.tok=reinterpret_cast<int*>(take(nodes));
    w.parent=reinterpret_cast<int*>(take(nodes));w.depth=reinterpret_cast<int*>(take(nodes));
    w.path=reinterpret_cast<int*>(take(nodes*c.proposal_length));w.remap=reinterpret_cast<int*>(take(nodes));
    w.frontier=reinterpret_cast<int*>(take(n*max_frontier));
    w.count=reinterpret_cast<int*>(take(n));w.valid=reinterpret_cast<int*>(take(n));
    w.stage=nullptr;w.ctl=nullptr;w.keys=nullptr;
    if(a.op==FULL_BASE) {
        /* rows * head_lists * 64 candidate rank keys, each a 64-bit unsigned value: two F32 words per
         * key. The ring base places the same buffer past its own arena and sizes its hook the same
         * way, so both paths reserve one key-list block of the same shape. */
        w.keys=take(2*(int64_t(c.n_seq)*c.proposal_length*head_lists(c.vocab_size_shard)*64));
    }
    w.total=used;
    return w;
}
/* The arena's size in F32 elements, taken from the layout itself rather than restated, and its byte
 * count. */
__host__ __device__ inline int64_t full_elements(const DeviceArgs& a,int op) {
    DeviceArgs b=a;b.op=op;
    return full_workspace(b).total;
}
__host__ inline int64_t full_scratch_bytes(const DeviceArgs& a,int op) {
    return 4*full_elements(a,op);
}

__host__ __device__ inline Work workspace(const DeviceArgs& a) {
    return a.capacity?full_workspace(a):ring_workspace(a);
}

__device__ inline float bf(uint16_t x){return __uint_as_float(uint32_t(x)<<16);}
__device__ inline bool seq_valid(const DeviceArgs& a,int s) {return workspace(a).valid[s]!=0;}
__host__ __device__ inline const uint16_t* weight(const DeviceArgs& a,int id,int layer=0) {
    /* The weight blob's operand index differs per op: the ring base at 5, the full base/tree at 4, and
     * the full append at 7. One lookup, so no kernel body can read the wrong operand. */
    const int k=a.op==RAD_DLOOP_BASE?5:(a.op==FULL_APPEND?7:4);
    return static_cast<const uint16_t*>(a.t[k])+a.offset[layer][id];
}

/* Layer norm over `rows` rows of width h. The per-row work is one warp-sized group of at most 32 rows:
 * the ring path never passes more than 9, and the full append's staging passes a whole chunk, so the
 * same body is applied group by group. For rows <= 32 there is exactly one group and the arithmetic
 * and order are the ring path's exactly.
 *
 * The kernel was pure latency: one lane per row walked d = 0..h-1 with a load and its dependent add
 * interleaved, so every add waited the whole memory round trip of its own operand. Both passes now
 * carry four float4 loads per lane, so the memory system gets 16 elements of run-up instead of one.
 * The sum is still ONE accumulator per row advancing in strictly ascending d -- only the arrival time
 * of the operands changed -- and `x*s` then `*weight` is the same pair of roundings as before.
 *
 * `vec` is a uniform predicate on the pointers and the width: both are 16B aligned in every caller
 * (the arena and the weight blob are device allocations, h is a multiple of 4), and the scalar path
 * stays for the widths that are not. */
__device__ inline float4 ld4(const float* p) {return *reinterpret_cast<const float4*>(p);}
__device__ inline uint4 ldu4(const void* p) {return *reinterpret_cast<const uint4*>(p);}
__device__ inline void st4(float* p,const float4 v) {*reinterpret_cast<float4*>(p)=v;}
__device__ inline bool vec16(const void* p) {return (reinterpret_cast<uintptr_t>(p)&15)==0;}
/* One dot product of a fp32 vector with a BF16 row of `n`: ONE accumulator, ascending d, `x*w` per
 * element with the product added to the accumulator -- the reference's own sum, not a reassociated
 * tree. What the old loop lacked was run-up: it issued one 2-byte weight load and one float load per
 * element and consumed both at once, so each of the 2560 adds of a unary dot waited out the whole
 * memory round trip of its own operand. Here the loads of the NEXT thirty-two elements are issued
 * while the adds of the current thirty-two retire, in vector form (eight float4 and four uint4). */
__device__ inline float dot_bf16(const float* x,const uint16_t* w,int n) {
    float acc=0;int d=0;
    if((n&7)==0&&vec16(x)&&vec16(w)) {
        for(;d+32<=n;d+=32) {
            const float4 a0=ld4(x+d),a1=ld4(x+d+4),a2=ld4(x+d+8),a3=ld4(x+d+12);
            const float4 a4=ld4(x+d+16),a5=ld4(x+d+20),a6=ld4(x+d+24),a7=ld4(x+d+28);
            const uint4 b0=ldu4(w+d),b1=ldu4(w+d+8),b2=ldu4(w+d+16),b3=ldu4(w+d+24);
            acc+=a0.x*bf(uint16_t(b0.x));acc+=a0.y*bf(uint16_t(b0.x>>16));
            acc+=a0.z*bf(uint16_t(b0.y));acc+=a0.w*bf(uint16_t(b0.y>>16));
            acc+=a1.x*bf(uint16_t(b0.z));acc+=a1.y*bf(uint16_t(b0.z>>16));
            acc+=a1.z*bf(uint16_t(b0.w));acc+=a1.w*bf(uint16_t(b0.w>>16));
            acc+=a2.x*bf(uint16_t(b1.x));acc+=a2.y*bf(uint16_t(b1.x>>16));
            acc+=a2.z*bf(uint16_t(b1.y));acc+=a2.w*bf(uint16_t(b1.y>>16));
            acc+=a3.x*bf(uint16_t(b1.z));acc+=a3.y*bf(uint16_t(b1.z>>16));
            acc+=a3.z*bf(uint16_t(b1.w));acc+=a3.w*bf(uint16_t(b1.w>>16));
            acc+=a4.x*bf(uint16_t(b2.x));acc+=a4.y*bf(uint16_t(b2.x>>16));
            acc+=a4.z*bf(uint16_t(b2.y));acc+=a4.w*bf(uint16_t(b2.y>>16));
            acc+=a5.x*bf(uint16_t(b2.z));acc+=a5.y*bf(uint16_t(b2.z>>16));
            acc+=a5.z*bf(uint16_t(b2.w));acc+=a5.w*bf(uint16_t(b2.w>>16));
            acc+=a6.x*bf(uint16_t(b3.x));acc+=a6.y*bf(uint16_t(b3.x>>16));
            acc+=a6.z*bf(uint16_t(b3.y));acc+=a6.w*bf(uint16_t(b3.y>>16));
            acc+=a7.x*bf(uint16_t(b3.z));acc+=a7.y*bf(uint16_t(b3.z>>16));
            acc+=a7.z*bf(uint16_t(b3.w));acc+=a7.w*bf(uint16_t(b3.w>>16));
        }
    }
    for(;d<n;++d)acc+=x[d]*bf(w[d]);
    return acc;
}

__device__ inline void rms(const float* x,const uint16_t* weight,float* y,int rows,int h,float eps) {
    __shared__ float scale[32];
    const bool vec=(h&3)==0&&vec16(x)&&vec16(y)&&(reinterpret_cast<uintptr_t>(weight)&7)==0;
    const int n4=h>>2;
    for(int base=0;base<rows;base+=32) {
        const int n=rows-base<32?rows-base:32;
        for(int row=threadIdx.x;row<n;row+=threads) {
            const float* xr=x+(base+row)*h;
            float sum=0;int d=0;
            if(vec) {
                for(;d+16<=h;d+=16) {
                    const float4 v0=ld4(xr+d),v1=ld4(xr+d+4),v2=ld4(xr+d+8),v3=ld4(xr+d+12);
                    sum+=v0.x*v0.x;sum+=v0.y*v0.y;sum+=v0.z*v0.z;sum+=v0.w*v0.w;
                    sum+=v1.x*v1.x;sum+=v1.y*v1.y;sum+=v1.z*v1.z;sum+=v1.w*v1.w;
                    sum+=v2.x*v2.x;sum+=v2.y*v2.y;sum+=v2.z*v2.z;sum+=v2.w*v2.w;
                    sum+=v3.x*v3.x;sum+=v3.y*v3.y;sum+=v3.z*v3.z;sum+=v3.w*v3.w;
                }
                for(;d+4<=h;d+=4) {
                    const float4 v=ld4(xr+d);
                    sum+=v.x*v.x;sum+=v.y*v.y;sum+=v.z*v.z;sum+=v.w*v.w;
                }
            }
            for(;d<h;++d)sum+=xr[d]*xr[d];
            scale[row]=1.0f/sqrtf(sum/float(h)+eps);
        }
        __syncthreads();
        /* The apply pass walked one flat index and paid a division for the row, the column and the
         * weight index of every element; row major with the scale hoisted costs none of them. One
         * float4 per lane keeps every lane in the group busy at every row count -- the per-row stride
         * form left 112 of 128 lanes idle when the ring path's 9 rows of 64 met a 128-thread block. */
        if(vec) {
            for(int i=threadIdx.x;i<n*n4;i+=threads) {
                const int row=i/n4,d=(i-row*n4)*4;
                const float s=scale[row];
                const float4 v=ld4(x+(base+row)*h+d);
                const uint32_t w01=*reinterpret_cast<const uint32_t*>(weight+d);
                const uint32_t w23=*reinterpret_cast<const uint32_t*>(weight+d+2);
                st4(y+(base+row)*h+d,make_float4(v.x*s*bf(uint16_t(w01)),v.y*s*bf(uint16_t(w01>>16)),
                                                v.z*s*bf(uint16_t(w23)),v.w*s*bf(uint16_t(w23>>16))));
            }
        } else {
            for(int row=0;row<n;++row) {
                const float s=scale[row];
                const float* xr=x+(base+row)*h;float* yr=y+(base+row)*h;
                #pragma unroll 4
                for(int d=threadIdx.x;d<h;d+=threads)yr[d]=xr[d]*s*bf(weight[d]);
            }
        }
        __syncthreads();
    }
}
/* One convolution output element, as the reference writes it: the tap loop ascending, base tap then
 * delta tap, each product added to a sum that starts at zero. `two` is the tap loop's own bound
 * (`tap<=row`), so the row-0 element never sees the second pair -- adding a zero product there would
 * turn a -0.0 result into +0.0. */
__device__ inline float conv_one(float b0,float v0,float dlo,float b1,float v1,float dhi,bool two) {
    float sum=0;
    sum+=b0*v0;
    sum+=dlo*v0;
    if(two) {
        sum+=b1*v1;
        sum+=dhi*v1;
    }
    return sum;
}
/* The convolution is elementwise, so WHICH lane computes an element is free; the per-element sum is
 * the ring path's own and each lane keeps its own accumulator, so the order inside an element is
 * untouched. A lane takes one (row, group) column, and a column is exactly one group of 16 elements:
 * the group's two delta taps are invariants there and the tap weights are 16 contiguous u16 each.
 *
 * The old walk was a flat element index: two 2-byte weight loads, two float loads and two divisions
 * per element. The column walks the SAME elements with the group's weight pairs and vector rows, so
 * sixteen elements cost four float4, four more for the tap-1 row, two uint4 of weights per tap, and
 * one float4 of stores -- and the index arithmetic is a division per column instead of per element.
 * RESIDUAL folds the caller's separate `block += pre` pass into the element that produced it. */
template<bool RESIDUAL>
__device__ inline void convolve_column(const float* x,const float* delta,const uint16_t* base,float* y,
                                       float* block,int row0,int row1,int h,int group,int phase) {
    const int ng=h/group;
    const bool vec=group==16&&(h&3)==0&&vec16(x)&&vec16(y)&&(!RESIDUAL||vec16(block));
    for(int t=threadIdx.x;t<(row1-row0)*ng;t+=threads) {
        const int row=row0+t/ng,g=t%ng,d0=g*group;
        const int taps=row?2:1;
        const float* xr=x+int64_t(row)*h+d0;
        const float* dr=delta+int64_t(row)*4*ng+phase*2*ng+g;
        float* yr=y+int64_t(row)*h+d0;
        /* Hoisted by hand: a store into y may alias delta for all the compiler knows, so an invariant
         * load left in the body would be re-issued for every element. */
        const float dlo=dr[0],dhi=taps>1?dr[ng]:0.0f;
        if(vec) {
            const uint16_t* w0=base+int64_t(phase*2)*h+d0;
            const uint4 p0=ldu4(w0),p1=ldu4(w0+8);
            const uint4 q0=taps>1?ldu4(w0+h):make_uint4(0u,0u,0u,0u);
            const uint4 q1=taps>1?ldu4(w0+h+8):make_uint4(0u,0u,0u,0u);
            const uint32_t pl[4]={p0.x,p0.z,p1.x,p1.z},ph[4]={p0.y,p0.w,p1.y,p1.w};
            const uint32_t ql[4]={q0.x,q0.z,q1.x,q1.z},qh[4]={q0.y,q0.w,q1.y,q1.w};
            const float* x1=taps>1?xr-h:nullptr;
            float4 o[4];
            /* Element 4q+0 and 4q+1 are the two halves of the pair word pl[q], 4q+2 and 4q+3 those
             * of ph[q]: the low half is the even element, the high half the odd one. */
            #pragma unroll
            for(int q=0;q<4;++q) {
                const float4 a=ld4(xr+q*4);
                const float4 c=taps>1?ld4(x1+q*4):a;
                o[q].x=conv_one(bf(uint16_t(pl[q])),a.x,dlo,bf(uint16_t(ql[q])),c.x,dhi,taps>1);
                o[q].y=conv_one(bf(uint16_t(uint32_t(pl[q])>>16)),a.y,dlo,
                                bf(uint16_t(uint32_t(ql[q])>>16)),c.y,dhi,taps>1);
                o[q].z=conv_one(bf(uint16_t(ph[q])),a.z,dlo,bf(uint16_t(qh[q])),c.z,dhi,taps>1);
                o[q].w=conv_one(bf(uint16_t(uint32_t(ph[q])>>16)),a.w,dlo,
                                bf(uint16_t(uint32_t(qh[q])>>16)),c.w,dhi,taps>1);
            }
            #pragma unroll
            for(int q=0;q<4;++q) {
                if constexpr(RESIDUAL) {
                    const float4 r=ld4(block+int64_t(row)*h+d0+q*4);
                    o[q]=make_float4(r.x+o[q].x,r.y+o[q].y,r.z+o[q].z,r.w+o[q].w);
                    st4(block+int64_t(row)*h+d0+q*4,o[q]);
                }
                st4(yr+q*4,o[q]);
            }
        } else {
            #pragma unroll 4
            for(int dd=0;dd<group;++dd) {
                float sum=0;
                for(int tap=0;tap<taps;++tap) {
                    const float v=xr[dd-tap*h];
                    sum+=bf(base[int64_t(phase*2+tap)*h+d0+dd])*v;
                    sum+=tap?dhi*v:dlo*v;
                }
                yr[dd]=sum;
                if constexpr(RESIDUAL)block[int64_t(row)*h+d0+dd]+=sum;
            }
        }
    }
    __syncthreads();
}
__device__ inline void convolve(const float* x,const float* delta,const uint16_t* base,float* y,
                                int b,int h,int group,int phase) {
    convolve_column<false>(x,delta,base,y,nullptr,0,b,h,group,phase);
}

/* ---------------------------------------------------------------- full-context helpers */

/* The cache element offset of (slot, layer, 0, 0): the same expression as
 * rad_dloop_full_kv_index in dloop_full.h, which is not __host__ __device__ qualified. */
__host__ __device__ inline int64_t kv_base(const DeviceArgs& a,int slot,int layer) {
    const int64_t D=int64_t(a.c.num_kv_heads)*a.c.head_dim;
    return ((int64_t(slot)*a.c.num_layers+layer)*a.capacity)*2*D;
}
__host__ __device__ inline int64_t kv_layer_stride(const DeviceArgs& a) {
    return int64_t(a.capacity)*2*a.c.num_kv_heads*a.c.head_dim;
}
__host__ __device__ inline const float* full_cache(const DeviceArgs& a) {
    return static_cast<const float*>(a.t[a.op==FULL_APPEND?8:0]);
}
__host__ __device__ inline float* full_cache_mut(const DeviceArgs& a) {
    return const_cast<float*>(full_cache(a));
}
__host__ __device__ inline const int* full_length(const DeviceArgs& a) {
    return static_cast<const int*>(a.t[a.op==FULL_APPEND?9:1]);
}
__host__ __device__ inline int* full_length_mut(const DeviceArgs& a) {
    return const_cast<int*>(full_length(a));
}
__host__ __device__ inline const int* full_slots(const DeviceArgs& a) {
    return static_cast<const int*>(a.t[a.op==FULL_APPEND?5:2]);
}
/* Weight blob of the full ops: the append carries it at operand 7, base and tree at 4. */
__host__ __device__ inline const uint16_t* full_weight(const DeviceArgs& a,int id,int layer=0) {
    return static_cast<const uint16_t*>(a.t[a.op==FULL_APPEND?7:4])+a.offset[layer][id];
}

/* The append's control block. EVERY field is int32, and int32 and float32 are the same size in this
 * arena, so the block's F32 word count is exactly its int count.
 *
 *   bad         whole-batch refusal. ONE malformed sequence makes the call a no-op -- not one cache
 *               byte, not one length, not one output is touched on a rejected call.
 *   ok[s]       this sequence passed validation
 *   limit[s]    new rows of this sequence (0 both when refused and for a legal empty chunk)
 *   rowoff[s]   prefix sum of limit over s, PLUS rowoff[n_seq]: the batch's total new-row count, which
 *               is why this array is n+1 long and must not share an element with any other field
 *   len0[s]     first written position; yoff(s) is the cache offset of (slot, 0, len0, 0)
 *   ylo[s],yhi[s]  low and high half of that offset (no int64 anywhere, so no alignment requirement)
 *   ones[r]     all-ones row-validity table for a staged chunk, r in [0, chunk_rows]; both GEMMs index
 *               it by a staged row's position within the chunk, so it is chunk_rows+1 entries
 *
 * The offsets are laid out once, here, by a running cursor. Seven n-sized arrays plus rowoff (n+1),
 * bad (1) and ones (chunk_rows+1) is exactly what append_ctl_ints returns; nothing is hand-counted a
 * second time and no field starts inside another.
 */
struct AppendCtl {
    int *ylo,*yhi,*len0,*slot,*limit,*ok,*rowoff,*bad,*live,*rowseen;
    int used;                     /* ints the cursor consumed: must be <= append_ctl_ints(n_seq) */
    __host__ __device__ int64_t yoff(int s) const {
        return (int64_t(yhi[s])<<32)|int64_t(uint32_t(ylo[s]));
    }
};
__host__ __device__ inline AppendCtl append_ctl(const DeviceArgs& a) {
    const int n=a.c.n_seq;
    int* p=reinterpret_cast<int*>(workspace(a).ctl);
    AppendCtl k{};int* c=p;
    k.ylo=c;c+=n;                 /* n */
    k.yhi=c;c+=n;                 /* n */
    k.len0=c;c+=n;                /* n */
    k.slot=c;c+=n;                /* n */
    k.limit=c;c+=n;               /* n */
    k.ok=c;c+=n;                  /* n */
    k.rowoff=c;c+=n+1;            /* n+1: rowoff[n_seq] is the batch row total, a field of its own */
    k.bad=c;c+=1;                 /* 1 */
    k.live=c;c+=1;                /* 1: live rows in the chunk being staged (see full_append_count) */
    k.rowseen=c;c+=chunk_rows+1;  /* chunk_rows+1: per-staged-row validity, filled by the count kernel */
    k.used=int(c-p);
    return k;
}
/* The absolute tap-plane row of the i-th new row of a sequence: a contiguous span, or the
 * -1-terminated path. One definition, so the gather and any host-side reader agree. */
__host__ __device__ inline int append_tap_row(const DeviceArgs& a,int s,int i) {
    const int* span=static_cast<const int*>(a.t[4]);
    return span[s*2+1]>=0?span[s*2]+i:static_cast<const int*>(a.t[3])[s*16+i];
}
/* The sequence owning global new-row r: the last one whose prefix offset is <= r. An empty sequence
 * cannot be selected, because it shares its offset with the next non-empty one. */
__host__ __device__ inline int append_row_seq(const AppendCtl& k,int n_seq,int64_t r) {
    int lo=0,hi=n_seq-1;
    while(lo<hi) {
        const int mid=(lo+hi+1)/2;
        if(int64_t(k.rowoff[mid])<=r)lo=mid;else hi=mid-1;
    }
    return lo;
}

/* The geometries the device code is compiled and validated for: the ring ops' two rows with
 * num_layers 5 and context_window 0. */
__host__ __device__ inline bool full_supported(const DeviceArgs& a,bool tiny) {
    const RadDLoopConfig& c=a.c;
    const int64_t D=int64_t(c.num_kv_heads)*c.head_dim;
    if(a.capacity<1||c.num_layers!=5||c.context_window!=0)return false;
    if(c.world_size!=2||c.proposal_length!=8||c.conv_kernel_size!=2||c.conv_group_size!=16||
       c.pool_size!=64||c.width>12||c.budget>15||c.head_dim%2||c.head_dim>64||
       c.num_q_heads%c.num_kv_heads||int64_t(c.num_q_heads)*c.head_dim!=c.hidden_size||
       c.context_features_dim!=3*c.target_width||c.hidden_size%c.conv_group_size)return false;
    if(c.vocab_size_shard*2!=c.vocab_size||(c.vocab_lo!=0&&c.vocab_lo!=c.vocab_size_shard))return false;
    /* every index and arena product the kernels form must stay inside int64 */
    if(int64_t(c.max_seqs)*c.num_layers*a.capacity*2*D>INT64_MAX/4)return false;
    if(int64_t(c.n_seq)*D*a.capacity*2>INT64_MAX/8)return false;
    if(tiny)return c.hidden_size==64&&c.intermediate_size==128&&c.num_q_heads==4&&c.num_kv_heads==2&&
        c.head_dim==16&&c.context_features_dim==96&&c.target_width==32&&c.selector_rank==8&&
        c.vocab_size==512&&c.active_vocab==509&&c.vocab_size_shard==256;
    return c.hidden_size==768&&c.intermediate_size==2304&&c.num_q_heads==12&&c.num_kv_heads==4&&
        c.head_dim==64&&c.context_features_dim==7680&&c.target_width==2560&&c.selector_rank==256&&
        c.vocab_size==248320&&c.active_vocab==248077&&c.vocab_size_shard==124160;
}

__host__ inline int send(const void* kernel,dim3 grid,int nthr,void** args,hipStream_t stream) {
    return hipLaunchKernel(kernel,grid,dim3(nthr),args,0,stream)==hipSuccess?RAD_OK:RAD_E_DEVICE;
}

/* The source model's rotary frequency, bit for bit: `inv_freq = 1.0 / (theta ** (2j / dh))` in FP32
 * (research/block_drafter/model/drafter.py, RotaryEmbedding). The ring path's `powf(theta, -2j/dh)` is
 * a different function: it lands one ulp away from the source at about half of the pairs (15 of 32 at
 * dh 64, 4 of 8 at dh 16, measured on gfx1201), and an absolute position multiplies that error -- an
 * angle of 24000 rad at the far end of the cache. The pow therefore runs in double and only its
 * result is rounded, which reproduces the source for every pair of both head widths. Callers build
 * ONE table per CTA instead of calling this per element: a double pow is about a hundred FP64
 * operations, and FP64 runs at a small fraction of the FP32 rate. */
__device__ inline float rope_inv_freq(float theta,int pair,int dh) {
    return 1.0f/float(pow(double(theta),2.0*double(pair)/double(dh)));
}
/* ==================================================================== shared full kernels */

/* The metadata gate every full base/tree launch runs first: validate each sequence's slot and
 * verified length, mark the sequence valid/refused, and lay down the implicit root node. One thread
 * per sequence, so the result is a settled scalar before the next kernel reads it. A draft on a
 * verified prefix of length 0 is not a valid call (dloop_full.h), and the capacity bound is
 * enforced, never rounded. */
/* Position coefficients are identical across all layers and tree jobs of a sequence. */
__device__ inline void full_rope_table(DeviceArgs a,int s) {
    const auto& c=a.c;
    const int pairs=c.head_dim/2,b=c.proposal_length+1;
    __shared__ float inv[32];
    if(int(threadIdx.x)<pairs)inv[threadIdx.x]=rope_inv_freq(c.rope_theta,int(threadIdx.x),c.head_dim);
    __syncthreads();
    const int length=full_length(a)[full_slots(a)[s]];
    float* table=workspace(a).rope+int64_t(s)*b*c.head_dim;
    for(int i=threadIdx.x;i<b*pairs;i+=threads) {
        const int row=i/pairs,d=i%pairs;
        const float angle=float(length+row)*inv[d];
        table[2*i]=cosf(angle);table[2*i+1]=sinf(angle);
    }
}
template<int TAG>
__global__ __launch_bounds__(threads) void full_initialize(DeviceArgs a) {
    const bool leader=threadIdx.x==0;
    const auto& c=a.c;const int s=blockIdx.x;
    if(s>=c.n_seq)return;
    Work w=workspace(a);
    const int slot=full_slots(a)[s];
    bool valid=slot>=0&&slot<c.max_seqs;
    if(valid) {
        const int length=full_length(a)[slot];
        valid=length>=1&&length<=a.capacity;
    }
    if(leader) {
        w.valid[s]=valid?1:0;w.count[s]=valid?1:0;w.frontier[s*max_frontier]=0;
        const int root=s*max_nodes;
        w.node_score[root]=0;w.tok[root]=-1;w.parent[root]=-1;w.depth[root]=0;
        for(int i=0;i<c.proposal_length;++i)w.path[root*c.proposal_length+i]=-1;
    }
    if(valid&&c.head_dim==64)full_rope_table(a,s);
}

/* Layer-0 input norm with the block's masked suffix: rows up to and including `slot` carry the real
 * embeddings and every row after it is the mask token, substituted here so no separate fill pass
 * exists. Same arithmetic as the ring path's initialiser.
 *
 * One CTA owns one row. Its leader keeps the reference's ascending-d sum; all threads apply
 * that row's scale after the barrier. Rows are independent; the reduction order is unchanged. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_initial_norm(DeviceArgs a,int front,int slot) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    const int lo=blockIdx.y,hi=lo+1;
    if(!seq_valid(a,s))return;
    Work w=workspace(a);
    const uint16_t* mask=weight(a,DL_MASK_TOKEN);
    const uint16_t* norm=weight(a,DL_INPUT_NORM,0);
    float* block=w.block+int64_t(job)*b*h;
    float* y=w.norm+int64_t(job)*b*h;
    __shared__ float scale[32];
    const bool vec=(h&3)==0&&vec16(block)&&vec16(y)&&
        (reinterpret_cast<uintptr_t>(mask)&7)==0&&(reinterpret_cast<uintptr_t>(norm)&7)==0;
    for(int row=lo+threadIdx.x;row<hi;row+=threads) {
        float* br=block+row*h;
        float sum=0;int d=0;
        if(row>slot) {
            /* The mask token IS the row's content, so the substitution and the sum are one pass over
             * the same registers; the discarded read of the old value is gone. */
            #pragma unroll 4
            for(;d+4<=h;d+=4) {
                const uint32_t m01=*reinterpret_cast<const uint32_t*>(mask+d);
                const uint32_t m23=*reinterpret_cast<const uint32_t*>(mask+d+2);
                const float4 v=make_float4(bf(uint16_t(m01)),bf(uint16_t(m01>>16)),
                                           bf(uint16_t(m23)),bf(uint16_t(m23>>16)));
                st4(br+d,v);
                sum+=v.x*v.x;sum+=v.y*v.y;sum+=v.z*v.z;sum+=v.w*v.w;
            }
            for(;d<h;++d) {const float v=bf(mask[d]);br[d]=v;sum+=v*v;}
        } else {
            #pragma unroll 4
            for(;d+16<=h;d+=16) {
                const float4 v0=ld4(br+d),v1=ld4(br+d+4),v2=ld4(br+d+8),v3=ld4(br+d+12);
                sum+=v0.x*v0.x;sum+=v0.y*v0.y;sum+=v0.z*v0.z;sum+=v0.w*v0.w;
                sum+=v1.x*v1.x;sum+=v1.y*v1.y;sum+=v1.z*v1.z;sum+=v1.w*v1.w;
                sum+=v2.x*v2.x;sum+=v2.y*v2.y;sum+=v2.z*v2.z;sum+=v2.w*v2.w;
                sum+=v3.x*v3.x;sum+=v3.y*v3.y;sum+=v3.z*v3.z;sum+=v3.w*v3.w;
            }
            for(;d<h;++d)sum+=br[d]*br[d];
        }
        scale[row-lo]=1.0f/sqrtf(sum/float(h)+c.eps);
    }
    __syncthreads();
    for(int row=lo;row<hi;++row) {
        const float sc=scale[row-lo];
        const float* br=block+row*h;float* yr=y+row*h;
        if(vec) {
            for(int q=threadIdx.x;q<(h>>2);q+=threads) {
                const int d=q*4;
                const float4 v=ld4(br+d);
                const uint32_t w01=*reinterpret_cast<const uint32_t*>(norm+d);
                const uint32_t w23=*reinterpret_cast<const uint32_t*>(norm+d+2);
                st4(yr+d,make_float4(v.x*sc*bf(uint16_t(w01)),v.y*sc*bf(uint16_t(w01>>16)),
                                     v.z*sc*bf(uint16_t(w23)),v.w*sc*bf(uint16_t(w23>>16))));
            }
        } else {
            #pragma unroll 4
            for(int d=threadIdx.x;d<h;d+=threads)yr[d]=br[d]*sc*bf(norm[d]);
        }
    }
    __syncthreads();
}

template<int TAG>
__global__ __launch_bounds__(threads) void full_conv_pre(DeviceArgs a,int front,int layer,int mlp) {
    const auto& c=a.c;const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;
    Work w=workspace(a);
    convolve(w.norm+int64_t(job)*b*h,w.delta+int64_t(job)*b*4*(h/c.conv_group_size),
        weight(a,mlp?DL_MLP_BASE:DL_ATTN_BASE,layer),w.pre+int64_t(job)*b*h,b,h,c.conv_group_size,0);
}

/* Attention finish and residual, the post-attention norm, the MLP finish, then either the next
 * layer's input norm or the final norm, for ONE block row per CTA (grid jobs x b). Row r reads only
 * rows r and r-1 of w.out, which no CTA writes, and every norm is row-local, so the rows are
 * independent; each element and each row sum is computed exactly as by the one-CTA-per-job form.
 * `base_out` (>0) picks the final rows: the base op emits the M proposal rows 1..M, the tree op emits
 * the single row `slot+1`. The flag is an int because it crosses the kernel-argument ABI. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_finish_norm(DeviceArgs a,int front,int layer,int stage,int slot,int base_out) {
    const auto& c=a.c;const int job=blockIdx.x,row=blockIdx.y,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;
    Work w=workspace(a);const int64_t off=int64_t(job)*b*h,ro=off+int64_t(row)*h;
    convolve_column<true>(w.out+off,w.delta+int64_t(job)*b*4*(h/c.conv_group_size),
        weight(a,stage?DL_MLP_BASE:DL_ATTN_BASE,layer),w.pre+off,w.block+off,row,row+1,h,c.conv_group_size,1);
    if(!stage)rms(w.block+ro,weight(a,DL_POST_NORM,layer),w.norm+ro,1,h,c.eps);
    else if(layer+1<c.num_layers)rms(w.block+ro,weight(a,DL_INPUT_NORM,layer+1),w.norm+ro,1,h,c.eps);
    else if(base_out) {
        if(row)rms(w.block+ro,weight(a,DL_FINAL_NORM),w.final+(int64_t(job)*c.proposal_length+row-1)*h,1,h,c.eps);
    } else if(row==slot+1)rms(w.block+ro,weight(a,DL_FINAL_NORM),w.final+int64_t(job)*h,1,h,c.eps);
}


/* Row bound (B = 9, pinned by full_supported) and pair bound (dh/2 <= 32) of the rope's memo. */
constexpr int B_ROPE=9, ROPE_PAIRS=32;

/* Per-CTA cos/sin memo for the block rope's rotated half-pairs: one (cos,sin) per (row, in-head pair),
 * 2304 B of shared for the widest geometry (576 B for the tiny one). A half-pair's angle depends on the
 * ROW only -- every head rotates with the same ladder inv[d] -- while a thread's other coordinates are
 * (head, second pair), so the angle and its cosf/sinf were recomputed once per head: (h+D)/2 = 512
 * elements per row carrying 32 distinct angles at dh 64/dh 16.
 *
 * Bit-exact: each entry is written with the very expression the rotation used per element,
 * `float(base+row)*inv[pair]`, so this is the same two floats by the same two roundings and not merely
 * a close value. Caller fills it one thread per (row, pair), grid-strided, then hits the barrier inside
 * this call before reading. The table belongs to one CTA and one launch. */
__device__ inline void rope_cs_memo(float (*cs)[ROPE_PAIRS][2],const float* inv,int rows,int base,int dh) {
    const int pairs=dh/2,total=rows*pairs;
    __syncthreads();
    for(int i=threadIdx.x;i<total;i+=threads) {
        const int row=i/pairs,pair=i-row*pairs;
        const float angle=float(base+row)*inv[pair];
        cs[row][pair][0]=cosf(angle);cs[row][pair][1]=sinf(angle);
    }
    __syncthreads();
}

/* Block Q/K rotary at the block's ABSOLUTE positions length[slot]+row, one CTA per (sequence, job).
 * Q occupies [B][H] of the qkv row, then the block's own K and V in [B][D] each; only the rotated
 * half-pairs are touched, so V passes through unchanged. The frequency is the source model's own
 * (rope_inv_freq, one table per CTA) and cosf/sinf are exact. There is no `layer` argument because the rope is
 * position-based and never touches the per-layer cache. Runs BEFORE attention, which reads the
 * unrotated V. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_block_rope(DeviceArgs a,int front) {
    const auto& c=a.c;
    const int job=blockIdx.x,s=job/front,b=c.proposal_length+1,h=c.hidden_size;
    if(!seq_valid(a,s))return;
    const int D=c.num_kv_heads*c.head_dim,dh=c.head_dim,pitch=h+2*D;
    float* qkv=workspace(a).qkv+int64_t(job)*b*pitch;
    const int slot=full_slots(a)[s];
    const int length=full_length(a)[slot];
    /* One frequency table and one cos/sin memo for the whole block: the D heads of a row all rotate
     * with the same invariant ladder, so the transcendental work drops by that factor. Static shared
     * is 1536 floats = 6 KB, the same order the attention kernel's own `part` array uses. */
    __shared__ float inv[ROPE_PAIRS];
    __shared__ float cs[B_ROPE][ROPE_PAIRS][2];
    if(int(threadIdx.x)<dh/2)inv[threadIdx.x]=rope_inv_freq(c.rope_theta,int(threadIdx.x),dh);
    rope_cs_memo(cs,inv,b,length,dh);
    for(int i=threadIdx.x;i<b*((h+D)/2);i+=threads) {
        const int row=i/((h+D)/2),channel=i%((h+D)/2);
        const bool key=channel>=h/2;const int rel=key?channel-h/2:channel;
        const int head=rel/(dh/2),d=rel%(dh/2),at=row*pitch+(key?h:0)+head*dh+d;
        /* Bit-identical to `cosf(float(length+row)*inv[d])`: that is the expression the memo entry was
         * written with, at the block's own absolute base. */
        const float co=cs[row][d][0],sn=cs[row][d][1];
        const float lo=qkv[at],hi=qkv[at+dh/2];
        qkv[at]=lo*co-hi*sn;qkv[at+dh/2]=hi*co+lo*sn;
    }
}

/* Which variant a launch takes: the row `full_supported` pins (h 768, dh 64, 4 kv heads, 12 q heads,
 * M 8). A dispatch decision only; both variants compute the same expression. */
__host__ inline bool attn_prod(const DeviceArgs& a) {
    const auto& c=a.c;
    return c.hidden_size==768&&c.head_dim==64&&c.num_kv_heads==4&&c.num_q_heads==12&&
        c.proposal_length==8;
}

/* Production geometry fuses Q/K rotation into attention; generic geometry uses full_block_rope. */
constexpr bool PROD_ATTENTION=true;

/* Streaming full-context attention, SPLIT over the key range. Grid (n_seq*front, B*KV_heads): one
 * CTA per (block row, key/value head), 512 threads = 16 warps, and warp s of that CTA owns the
 * contiguous cache key range [length*s/attn_splits, length*(s+1)/attn_splits) of that pair. The b keys
 * of the block are spread over the warps (key m goes to warp m%attn_splits), so all length+b keys are
 * visited exactly once and no key twice. Lane
 * d holds head dimensions d and d+32, so head_dim <= 64 needs two registers per lane and nothing
 * else. The cached positions come straight from the cache operand -- the only per-pair state is
 * (q0,q1,m,l,acc0,acc1) per query head -- and the block's own B positions come from the qkv scratch.
 * No C-sized register array, no C-sized LDS array, no copy of the cache.
 *
 * WHY THE SPLIT: one warp per (row, query head) pair walking the positions serially costs 0.43-0.46
 * us per cached position per launch, because 108 pairs is 1.7 waves per CU and a single warp has
 * nothing to overlap the cache load's latency with. attn_splits warps of a CTA now work on the same
 * (row, key/value head) pair in parallel and the split state meets again in shared memory inside the
 * SAME launch (`full_backbone` still launches one grid and one kernel per layer, no second pass, no
 * partials in global memory).
 *
 * GQA REUSE: a warp owns the whole key/value head, so the `groups` query heads of that head share
 * the one K/V row it loads -- and one key's three score chains are independent, which is what gives
 * the warp something to overlap a load's latency with. The previous shape re-read the same K/V rows
 * from 108 pair-warps; this one reads each (row, key) K/V once per warp. A group slot past `groups`
 * has no query head at all, so it keeps q=0, never advances its state and is never written out.
 *
 * Mask and position rule, which is DIFFERENT from the ring path's in exactly one way: the ring path
 * right-pads its window (its valid rows sit at [C-length, C)), while the full cache stores the
 * verified prefix at [0, length) with nothing but unused tail beyond it. So the visited key range is
 * 0..length-1 for the cache and then the block's own b positions -- `length + b` keys, never C + b,
 * and there is no C-derived mask or clamp anywhere. A cached key therefore carries its own ascending
 * absolute index n, and the block's key n-length is at absolute position length+(n-length).
 *
 * Numerics: FP32 throughout, exact expf, no approximation, no BF16 activation. The reduction tree IS
 * what streaming changes, and it is TWO deviations from the two-pass reference, both written out:
 *   1. the dot product per key is a balanced butterfly over the lanes instead of the reference's
 *      ascending scalar sum -- unchanged from the shape this kernel replaced;
 *   2. the softmax runs online (running max; previously accumulated terms rescaled by
 *      exp(m_old-m_new)) instead of two passes, AND it is now evaluated once per key split and the
 *      splits are merged afterwards, which is the same associativity the online form already relies
 *      on applied to attn_splits partial runs instead of one:
 *          M   = max over splits of m_s
 *          l   = sum over splits of exp(m_s-M) * l_s
 *          acc = sum over splits of exp(m_s-M) * acc_s
 *          out = acc * (1/l)          (the reciprocal multiply the unsplit kernel also ends with)
 *      Every weight is still exp(score - max of its own run), FP32, and the result is normalised
 *      once by the total sum. Splits that own no key keep m=-INFINITY, l=0, acc=0: the merge reads
 *      exp(-INFINITY-M)=0 for them, so an empty split contributes exactly nothing.
 * dloop_full.h allows a reordered reduction for streaming; this is the exact deviation. */
constexpr int attn_threads=512;                 /* 16 warps per (block row, kv head) CTA; 32 measured slower */
constexpr int attn_splits=attn_threads/32;      /* warp s owns cache key split s of attn_splits */
constexpr int attn_groups_max=3;                /* the group count of the widest supported geometry */

/* PRODUCTION GEOMETRY SPECIALISATION. `PROD` folds the row `full_supported` pins -- D=256, dh=64,
 * groups=3, pitch h+512, dhscale 8, cache key stride 512, block K/V stride, the three loop divides --
 * into constants, so the inner loop reads immediates instead of the config. It also carries the
 * ROTATION of the Q it loads and of its own block K, which is what removes the per-layer rope launch
 * (and its whole-block scratch write) from the production path. The tiny row (h64, dh16, 2 kv, 4 q)
 * keeps the generic kernel and the memoised rope launch.
 *
 * No arithmetic and no memory access changes: every expression is the runtime one with its value
 * substituted, every lane guard stays, and the split boundaries, the online-softmax form and the
 * 16-way merge are untouched -- bit-identical by construction, so PROD is a dispatch decision and not
 * a numerical one. */
template<int TAG,bool PROD>
__global__ __launch_bounds__(attn_threads) void full_attention(DeviceArgs a,int front,int layer) {
    const auto& c=a.c;
    const int job=blockIdx.x,s=job/front,b=PROD?9:c.proposal_length+1,h=PROD?768:c.hidden_size;
    if(!seq_valid(a,s))return;
    const int D=PROD?256:c.num_kv_heads*c.head_dim,dh=PROD?64:c.head_dim;
    const int groups=PROD?3:c.num_q_heads/c.num_kv_heads;
    const int pitch=PROD?h+512:h+2*D;
    const float dhscale=PROD?8.0f:sqrtf(float(dh));
    const int slot=full_slots(a)[s];
    const int length=full_length(a)[slot];
    const float* cache=full_cache(a);
    const float* qkv=workspace(a).qkv+int64_t(job)*b*pitch;
    const int64_t cbase=kv_base(a,slot,layer);
    const int kvheads=PROD?4:c.num_kv_heads;
    const int unit=blockIdx.y,row=unit/kvheads,kh=unit%kvheads;
    const int warp=threadIdx.x/32,lane=threadIdx.x%32;
    const int d0=lane,d1=lane+32;
    const bool have0=d0<dh,have1=d1<dh;
    const float* rope=workspace(a).rope+int64_t(s)*b*dh;
    /* The query heads of this key/value head: they share every K/V row this warp loads. The loop
     * bound is the compile-time group maximum with a runtime guard, never a runtime trip count, so
     * the state arrays below stay in registers (both admitted geometries fit: 3 and 2). */
    float q0[attn_groups_max],q1[attn_groups_max],acc0[attn_groups_max],acc1[attn_groups_max];
    float mh[attn_groups_max],lh[attn_groups_max];
    float qco=0.0f,qsn=0.0f;
    if constexpr(PROD) {qco=rope[row*dh+2*lane];qsn=rope[row*dh+2*lane+1];}
    #pragma unroll
    for(int j=0;j<attn_groups_max;++j) {
        acc0[j]=0.0f;acc1[j]=0.0f;mh[j]=-INFINITY;lh[j]=0.0f;q0[j]=0.0f;q1[j]=0.0f;
        if(j<groups) {
            const float* qr=qkv+int64_t(row)*pitch+(kh*groups+j)*dh;
            if constexpr(PROD) {
                const float lo=qr[d0],hi=qr[d1];
                q0[j]=lo*qco-hi*qsn;q1[j]=hi*qco+lo*qsn;
            } else {
                q0[j]=have0?qr[d0]:0.0f;q1[j]=have1?qr[d1]:0.0f;
            }
        }
    }
    /* One key through the online softmax for every head of the group at once. The four values come in
     * ALREADY LOADED, because the caller hoists a whole group of keys' loads ahead of the arithmetic:
     * that is the only thing that puts more than one cache line in flight per warp, and this kernel is
     * memory-level-parallelism bound, not ALU bound (288 warps x 512 B in flight = 147 KB against the
     * ~440 KB the HBM latency at 545 GB/s needs). The head loop carries a guard and no `break` so all
     * three slots unroll; the runtime `groups` only masks the slots with no query head. */
    auto step=[&](float k0,float k1,float v0,float v1) {
        #pragma unroll
        for(int j=0;j<attn_groups_max;++j) {
            if(j<groups) {
                float sc=fmaf(q0[j],k0,q1[j]*k1);
                #pragma unroll
                for(int o=16;o;o>>=1)sc+=__shfl_xor(sc,o);
                sc/=dhscale;
                /* `sc` is warp-uniform after the butterfly, so the branch is uniform and exactly one
                 * expf is issued per key. */
                float p;
                if(sc>mh[j]) {
                    const float corr=expf(mh[j]-sc);
                    acc0[j]*=corr;acc1[j]*=corr;lh[j]*=corr;mh[j]=sc;p=1.0f;
                } else p=expf(sc-mh[j]);
                acc0[j]=fmaf(p,v0,acc0[j]);acc1[j]=fmaf(p,v1,acc1[j]);lh[j]+=p;
            }
        }
    };
    /* Every key of the warp's own cache range, ascending, four keys per turn: all sixteen lane loads
     * issue before any of the four butterflies runs, so the four keys' latencies overlap. The tail
     * runs one key at a time. Measured 2026-10-18: eight keys per turn changed nothing. */
    const int lo=int(int64_t(length)*warp/attn_splits),hi=int(int64_t(length)*(warp+1)/attn_splits);
    int n=lo;
    /* The cache's per-key stride is 2*D and the block's K/V stride is D, both literals in PROD. */
    const int kv2=PROD?512:2*D;
    for(;n+4<=hi;n+=4) {
        const float* p0=cache+cbase+int64_t(n+0)*kv2+kh*dh;
        const float* p1=cache+cbase+int64_t(n+1)*kv2+kh*dh;
        const float* p2=cache+cbase+int64_t(n+2)*kv2+kh*dh;
        const float* p3=cache+cbase+int64_t(n+3)*kv2+kh*dh;
        const float k0a=have0?p0[d0]:0.0f,k1a=have1?p0[d1]:0.0f,v0a=have0?p0[d0+D]:0.0f,v1a=have1?p0[d1+D]:0.0f;
        const float k0b=have0?p1[d0]:0.0f,k1b=have1?p1[d1]:0.0f,v0b=have0?p1[d0+D]:0.0f,v1b=have1?p1[d1+D]:0.0f;
        const float k0c=have0?p2[d0]:0.0f,k1c=have1?p2[d1]:0.0f,v0c=have0?p2[d0+D]:0.0f,v1c=have1?p2[d1+D]:0.0f;
        const float k0d=have0?p3[d0]:0.0f,k1d=have1?p3[d1]:0.0f,v0d=have0?p3[d0+D]:0.0f,v1d=have1?p3[d1+D]:0.0f;
        step(k0a,k1a,v0a,v1a);step(k0b,k1b,v0b,v1b);step(k0c,k1c,v0c,v1c);step(k0d,k1d,v0d,v1d);
    }
    for(;n<hi;++n) {
        const float* p=cache+cbase+int64_t(n)*kv2+kh*dh;
        step(have0?p[d0]:0.0f,have1?p[d1]:0.0f,have0?p[d0+D]:0.0f,have1?p[d1+D]:0.0f);
    }
    /* Preserve exactly which of the sixteen softmax splits receives each block key. */
    for(int m=warp;m<b;m+=attn_splits) {
        const float* p=qkv+int64_t(m)*pitch+h+kh*dh;
        if constexpr(PROD) {
            const float co=rope[m*dh+2*lane],sn=rope[m*dh+2*lane+1];
            const float lo=p[d0],hi=p[d1];
            step(lo*co-hi*sn,hi*co+lo*sn,p[d0+D],p[d1+D]);
        } else {
            step(have0?p[d0]:0.0f,have1?p[d1]:0.0f,have0?p[d0+D]:0.0f,have1?p[d1+D]:0.0f);
        }
    }
    /* The splits meet in shared memory: one merge per (query head, head dimension). */
    __shared__ float part[attn_splits][attn_groups_max][2+64];
    #pragma unroll
    for(int j=0;j<attn_groups_max;++j) {
        if(j<groups) {
            if(lane==0) {part[warp][j][0]=mh[j];part[warp][j][1]=lh[j];}
            if(have0)part[warp][j][2+d0]=acc0[j];
            if(have1)part[warp][j][2+d1]=acc1[j];
        }
    }
    __syncthreads();
    for(int t=threadIdx.x;t<groups*dh;t+=attn_threads) {
        const int j=PROD?(t>>6):t/dh,d=PROD?(t&63):t%dh,qh=kh*groups+j;
        float mx=-INFINITY;
        #pragma unroll
        for(int sp=0;sp<attn_splits;++sp)mx=fmaxf(mx,part[sp][j][0]);
        float lsum=0.0f,acc=0.0f;
        #pragma unroll
        for(int sp=0;sp<attn_splits;++sp) {
            const float corr=expf(part[sp][j][0]-mx);
            lsum=fmaf(corr,part[sp][j][1],lsum);
            acc=fmaf(corr,part[sp][j][2+d],acc);
        }
        if(mx>-INFINITY) {
            float* dst=workspace(a).norm+int64_t(job)*b*h+int64_t(row)*h+qh*dh;
            dst[d]=acc*(1.0f/lsum);
        }
    }
}

/* ============================================================== shared full backbone
 *
 * The stage sequence is written once and parameterised over the file that owns the GEMM. The policy
 * receives a GEMM request in a shape both files can serve: `kind` picks the weight mapping (0 one
 * matrix, 1 Q|K|V, 2 K|V, 3 gate/up with silu(gate)*up) and `source` picks the activation source
 * (0 a dense F32 matrix, 2 the anchor/path token embeddings). The policy object also supplies the
 * stream the elementwise stages are queued on, so one helper launches the whole layer stack.
 */
struct GemmReq {
    const float* x;
    const uint16_t *w0,*w1,*w2,*anchor,*embed;
    float* y;
    const int* valid;const int* path;const int* frontier;
    /* Optional uniform exit for a row tile with no live row: null for the ring and full base/tree paths,
     * set by the full append to its staged-chunk live count. */
    const int* live;
    int rows,in,out,xstride,rows_per_seq,kind,source,split,front,slot,m,k;
};

template<int TAG,class POLICY>
__host__ inline int full_backbone(DeviceArgs& a,const POLICY& p,int front,int slot,int base_out,float* final_dst) {
    const auto& c=a.c;
    const int h=c.hidden_size,D=c.num_kv_heads*c.head_dim,b=c.proposal_length+1;
    Work w=workspace(a);
    const hipStream_t stream=p.stream_of();
    const int jobs=c.n_seq*front,rows=jobs*b;
    auto el=[&](const void* kernel,dim3 grid,void** args) {
        return send(kernel,grid,threads,args,stream);
    };
    const dim3 flat(jobs), attn_grid(jobs,(int)(b*c.num_kv_heads));
    GemmReq embed{};
    embed.w0=full_weight(a,DL_TOKEN_PROJ);embed.y=w.block;embed.rows=rows;embed.in=c.target_width;
    embed.out=h;embed.xstride=c.target_width;embed.rows_per_seq=front*b;embed.source=2;
    embed.anchor=static_cast<const uint16_t*>(a.t[3]);
    embed.embed=a.op==FULL_TREE?static_cast<const uint16_t*>(a.t[9]):nullptr;
    embed.valid=w.valid;embed.path=w.path;embed.frontier=w.frontier;
    embed.front=front;embed.slot=slot;embed.m=c.proposal_length;embed.k=c.pool_size;
    int rc=p.gemm(embed);if(rc)return rc;
    void* initial[]={&a,&front,&slot};
    rc=el(reinterpret_cast<const void*>(full_initial_norm<TAG>),dim3(jobs,b),initial);if(rc)return rc;
    for(int layer=0;layer<c.num_layers;++layer) {
        for(int mlp=0;mlp<2;++mlp) {
            GemmReq dyn{};
            dyn.x=w.norm;dyn.w0=full_weight(a,mlp?DL_MLP_KERNEL:DL_ATTN_KERNEL,layer);dyn.y=w.delta;
            dyn.rows=rows;dyn.in=h;dyn.out=4*(h/c.conv_group_size);dyn.xstride=h;dyn.rows_per_seq=front*b;
            dyn.valid=w.valid;
            /* The dyn producer may fuse phase-0 conv (group-owned strided columns) or refuse the
             * shape; the refusal runs the unfused pair unchanged, every other error propagates. */
            rc=p.dyn(a,dyn,layer,mlp);
            if(rc==RAD_E_UNSUPPORTED) {
                rc=p.gemm(dyn);if(rc)return rc;
                void* conv[]={&a,&front,&layer,&mlp};
                rc=el(reinterpret_cast<const void*>(full_conv_pre<TAG>),flat,conv);if(rc)return rc;
            } else if(rc)return rc;
            if(!mlp) {
                GemmReq qkv{};
                qkv.x=w.pre;qkv.w0=full_weight(a,DL_Q_PROJ,layer);qkv.w1=full_weight(a,DL_K_PROJ,layer);
                qkv.w2=full_weight(a,DL_V_PROJ,layer);qkv.y=w.qkv;qkv.rows=rows;qkv.in=h;qkv.out=h+2*D;
                qkv.xstride=h;qkv.rows_per_seq=front*b;qkv.kind=1;qkv.valid=w.valid;qkv.split=h;
                rc=p.gemm(qkv);if(rc)return rc;
                /* The block's own Q and K are rotated at their ABSOLUTE positions (length[slot]+row)
                 * BEFORE scoring, exactly as the ring path does; V is untouched by the rope and is
                 * read unrotated by the attention kernel. The cached K was rotated when it was
                 * appended, so nothing here touches the cache.
                 *
                 * PROD does not launch this at all: full_attention rotates the Q *it loads* in
                 * registers and its own block K on the way in, which is the same two operands and the
                 * same expressions, so the qkv scratch keeps its unrotated projection and one launch
                 * per layer disappears. GENERIC (tiny) keeps the memoised launch below unchanged. */
                const bool prod=PROD_ATTENTION&&attn_prod(a);
                if(!prod) {
                    void* attn[]={&a,&front};
                    rc=el(reinterpret_cast<const void*>(full_block_rope<TAG>),flat,attn);if(rc)return rc;
                }
                void* score[]={&a,&front,&layer};
                const void* attn_kernel=prod?
                    reinterpret_cast<const void*>(full_attention<TAG,true>):
                    reinterpret_cast<const void*>(full_attention<TAG,false>);
                rc=send(attn_kernel,attn_grid,attn_threads,score,stream);
                if(rc)return rc;
                GemmReq o{};
                o.x=w.norm;o.w0=full_weight(a,DL_O_PROJ,layer);o.y=w.out;
                o.rows=rows;o.in=h;o.out=h;o.xstride=h;o.rows_per_seq=front*b;o.valid=w.valid;
                rc=p.gemm(o);if(rc)return rc;
            } else {
                GemmReq gate{};
                gate.x=w.pre;gate.w0=full_weight(a,DL_GATE_PROJ,layer);gate.w1=full_weight(a,DL_UP_PROJ,layer);
                gate.y=w.ff;gate.rows=rows;gate.in=h;gate.out=c.intermediate_size;
                gate.xstride=h;gate.rows_per_seq=front*b;gate.kind=3;gate.valid=w.valid;
                rc=p.gemm(gate);if(rc)return rc;
                GemmReq down{};
                down.x=w.ff;down.w0=full_weight(a,DL_DOWN_PROJ,layer);down.y=w.out;
                down.rows=rows;down.in=c.intermediate_size;down.out=h;
                down.xstride=c.intermediate_size;down.rows_per_seq=front*b;down.valid=w.valid;
                rc=p.gemm(down);if(rc)return rc;
            }
            int fin[6]={front,layer,mlp,slot,base_out?1:0,0};
            void* finish[]={&a,&fin[0],&fin[1],&fin[2],&fin[3],&fin[4]};
            rc=el(reinterpret_cast<const void*>(full_finish_norm<TAG>),dim3(jobs,b),finish);if(rc)return rc;
        }
    }
    GemmReq out{};
    out.x=w.final;out.w0=full_weight(a,DL_OUTPUT_PROJ);out.y=final_dst?final_dst:w.y;
    out.rows=base_out?c.n_seq*c.proposal_length:jobs;out.in=h;out.out=c.target_width;
    out.xstride=h;out.rows_per_seq=base_out?c.proposal_length:front;out.valid=w.valid;
    return p.gemm(out);
}

/* ============================================================ shared full append stage
 *
 * Four chunked steps, and the whole point of the op:
 *   1. full_append_scan     one CTA per sequence: validate slot/span/path/reset against the cache
 *                           capacity, publish len0/limit, then the batch refusal flag. Nothing else
 *                           in the launch writes anything once it is set.
 *   2. full_append_gather   stage the chunk's tap rows as dense F32 features, exactly once.
 *   3. projection           CONTEXT_PROJ + CONTEXT_NORM for the staged rows (the policy's GEMM).
 *   4. per layer            K_PROJ|V_PROJ on the projected rows, then the ABSOLUTE rotation at each
 *                           row's own position, then ONE write into the cache (full_append_store).
 *
 * K/V is projected ONCE per verified row here and never again: a decode step of one row costs L
 * projections, and a draft step costs none. The historical prefix is never reprojected.
 */

/* Two kernels, because the scan must settle `rowoff` and `bad` before any staging kernel reads them,
 * and a CTA boundary is the only ordering a launch gives cheaply. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_scan(DeviceArgs a) {
    if(threadIdx.x!=0)return;
    const auto& c=a.c;const int s=blockIdx.x;
    if(s>=c.n_seq)return;
    AppendCtl k=append_ctl(a);
    const int slot=static_cast<const int*>(a.t[5])[s];
    const int reset=static_cast<const int*>(a.t[6])[s];
    const int* span=static_cast<const int*>(a.t[4]);
    const int start=span[s*2],count=span[s*2+1];
    k.ylo[s]=k.yhi[s]=0;k.len0[s]=0;k.slot[s]=-1;k.limit[s]=0;k.ok[s]=0;k.rowoff[s]=0;
    if(s==0){k.rowoff[c.n_seq]=0;k.bad[0]=0;}
    bool valid=slot>=0&&slot<c.max_seqs&&(reset==0||reset==1);
    /* Slots are unique within one append: a repeated slot is a caller error and only the first owner
     * may touch the storage (the ring path's rule, kept). */
    if(valid)for(int prior=0;prior<s;++prior)if(static_cast<const int*>(a.t[5])[prior]==slot)valid=false;
    int nrows=0;
    if(valid) {
        if(count==-1) {
            bool ended=false;
            for(int i=0;i<16;++i) {
                const int row=static_cast<const int*>(a.t[3])[s*16+i];
                if(row==-1)ended=true;
                else if(ended||row<0||row>=c.tap_rows){valid=false;break;}
                else ++nrows;
            }
        } else if(count>=0) {
            /* A contiguous verified prefill chunk. An empty chunk is legal. */
            valid=start>=0&&int64_t(start)+count<=c.tap_rows;
            nrows=valid?count:0;
        } else valid=false;                        /* any other negative count is invalid */
    }
    const int length=valid?full_length(a)[slot]:0;
    const int len0=valid?(reset?0:length):0;
    /* The capacity bound is enforced, never truncated or wrapped: len0 + nrows <= C. */
    if(valid&&(len0<0||len0>a.capacity||nrows>a.capacity-len0))valid=false;
    if(!valid)return;                              /* limit stays 0: the whole batch becomes a no-op */
    k.len0[s]=len0;k.slot[s]=slot;k.limit[s]=nrows;k.ok[s]=1;
    const int64_t off=kv_base(a,slot,0)+int64_t(len0)*2*(c.num_kv_heads*c.head_dim);
    k.ylo[s]=int(uint32_t(uint64_t(off)));k.yhi[s]=int(uint32_t(uint64_t(off)>>32));
}
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_layout(DeviceArgs a) {
    AppendCtl k=append_ctl(a);const auto& c=a.c;
    if(threadIdx.x!=0)return;
    int sum=0,bad=0;
    for(int s=0;s<c.n_seq;++s) {
        if(!k.ok[s]||k.limit[s]<0)bad=1;
        k.rowoff[s]=sum;
        sum+=k.limit[s];
    }
    k.rowoff[c.n_seq]=sum;
    k.bad[0]=bad;                                  /* published last: everything above reads it */
}

/* One CTA per staged row of the current chunk. A chunk is a contiguous range [r0, r0+rows) of the
 * batch's flat new-row set, so it may straddle sequences; every row is independent (its sequence, its
 * index and its tap row are all looked up per row), which is what lets one arena of `chunk_rows`
 * rows serve a batch of any width.
 *
 * EVERY row of the chunk gets written, dead ones as zeros: the GEMM below reads the whole staged tile,
 * so leaving a dead row untouched would fold uninitialised scratch into an accumulator. The dead rows
 * cost their tile's arithmetic but can never reach the cache -- full_append_store re-tests the live
 * count per row before it addresses anything. The one-shot host-side bound on the chunk walk
 * (n_seq*max(tap_rows,16), see full_append_run) is what decides how many such rows exist; it is a
 * conservative bound, not a measurement, and no speedup is claimed from it.
 *
 * The three tap planes are laid out contiguously (plane 0 then 1 then 2, each W wide) in the same
 * order the ring's append concatenates them, so the projection below is one dense GEMM over F = 3*W. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_gather(DeviceArgs a,int r0,int rows) {
    AppendCtl k=append_ctl(a);
    const auto& c=a.c;
    const int l=blockIdx.x;
    if(l>=rows)return;
    const int F=c.context_features_dim,W=c.target_width;
    float* dst=workspace(a).stage+int64_t(l)*F;
    const int r=r0+l,total=k.rowoff[c.n_seq];
    /* bad: the batch is refused, so the staging is filled with zeros and nothing is stored. */
    const bool live=!k.bad[0]&&r<total;
    int s=-1,i=0,row=-1;
    if(live) {
        s=append_row_seq(k,c.n_seq,r);i=r-k.rowoff[s];
        if(i>=k.limit[s]||i<0)s=-1;                 /* past this sequence's rows: dead */
        else row=append_tap_row(a,s,i);
    }
    if(s>=0) {
        for(int d=threadIdx.x;d<F;d+=threads) {
            const int plane=d<W?0:(d<2*W?1:2),off=d-plane*W;
            dst[d]=bf(static_cast<const uint16_t*>(a.t[plane])[int64_t(row)*W+off]);
        }
    } else {
        for(int d=threadIdx.x;d<F;d+=threads)dst[d]=0.0f;
    }
}
/* Counts the live rows of the chunk [r0,r0+rows) and publishes TWO things the later stages use:
 *   k.live[0]           the live count (0 for a refused batch)
 *   k.rowseen[0..rows)  1 for a live row, 0 for a dead one
 *
 * The live rows of a chunk are always a PREFIX: rowoff is monotone, so every r < rowoff[n_seq] belongs
 * to some sequence's live range and every r >= it does not. That is what lets the GEMMs index validity
 * by the staged row itself (`rows_per_seq = 1`) instead of by a sequence, and it is why one table of
 * chunk_rows+1 entries is enough.
 *
 * A dead row is staged as zeros by full_append_gather and its accumulator is zeroed by this table, so
 * no uninitialised scratch can enter a result -- the count kernel is what makes that guarantee, not an
 * assumption about the caller's chunk sizes. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_count(DeviceArgs a,int r0,int rows) {
    AppendCtl k=append_ctl(a);
    const auto& c=a.c;
    int live=0;
    if(!k.bad[0]) {
        const int total=k.rowoff[c.n_seq];
        for(int l=0;l<rows;++l) {
            const int r=r0+l;
            if(r>=total)break;
            const int s=append_row_seq(k,c.n_seq,r);
            const int i=r-k.rowoff[s];
            if(i>=0&&i<k.limit[s])++live;
        }
    }
    for(int l=threadIdx.x;l<=chunk_rows;l+=threads)k.rowseen[l]=l<live?1:0;
    __syncthreads();
    if(threadIdx.x==0)k.live[0]=live;
}

/* CONTEXT_NORM over the live prefix of the staged chunk, in place: exact FP32 rms, the same expression
 * the ring path's context norm uses. Live rows form a prefix, so normalising [0, live) is exactly the
 * live set -- and it is the whole set, because the tile's dead rows are never read again (the GEMMs'
 * validity table returns a null source for them, so neither w.norm nor w.stage is touched for a row that
 * is not live). Nothing is read back to the host and no launch is added: the count comes from the same
 * device-side block the GEMMs use. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_norm(DeviceArgs a,int rows) {
    AppendCtl k=append_ctl(a);
    const int live=k.live[0];
    if(k.bad[0]||live<=0)return;
    if(rows>live)rows=live;
    Work w=workspace(a);
    rms(w.norm,full_weight(a,DL_CONTEXT_NORM),w.norm,rows,a.c.hidden_size,a.c.eps);
}

/* CONTEXT_PROJ then the norm above. The GEMM is the policy's. Its activation tile is validated per
 * STAGED ROW (`rows_per_seq = 1`, so validity is indexed by the row itself), which is what the count
 * kernel's rowseen table is for: a dead row's accumulator is zeroed instead of reading uninitialised
 * scratch, and the row is never stored because full_append_store re-tests the live range. Nothing is
 * read back to the host: the GEMMs read the device-side table directly. */
template<int TAG,class POLICY>
__host__ inline int full_append_project(DeviceArgs& a,const POLICY& p,int rows,int F) {
    const auto& c=a.c;
    Work w=workspace(a);
    AppendCtl k=append_ctl(a);
    GemmReq proj{};
    proj.x=w.stage;proj.w0=full_weight(a,DL_CONTEXT_PROJ);proj.y=w.norm;
    proj.rows=rows;proj.in=F;proj.out=c.hidden_size;proj.xstride=F;proj.rows_per_seq=1;
    proj.valid=k.rowseen;proj.live=k.live;
    int rc=p.gemm(proj);if(rc)return rc;
    int n=rows;
    void* args[]={&a,&n};
    const void* norm=reinterpret_cast<const void*>(&full_append_norm<TAG>);
    return send(norm,dim3(1),threads,args,p.stream_of());
}

/* K|V row store for ONE layer of the current chunk: the K half is rotated IN THE DESTINATION at the
 * position the row will occupy (len0+i), which is the contract's "K is already rotated at its absolute
 * position"; V is stored untouched. One CTA per staged row: the row's K|V is a contiguous 2D block for
 * the copy and a D/2 half-pair sweep for the rotation, both coalesced. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_store(DeviceArgs a,int r0,int rows,int layer) {
    AppendCtl k=append_ctl(a);
    const auto& c=a.c;
    if(k.bad[0])return;
    const int l=blockIdx.x;
    if(l>=rows)return;
    const int r=r0+l,total=k.rowoff[c.n_seq];
    if(r>=total)return;
    const int s=append_row_seq(k,c.n_seq,r),i=r-k.rowoff[s];
    if(i>=k.limit[s])return;
    const int D=c.num_kv_heads*c.head_dim,dh=c.head_dim;
    __shared__ float inv[32];                            /* published by the __syncthreads below */
    if(int(threadIdx.x)<dh/2)inv[threadIdx.x]=rope_inv_freq(c.rope_theta,int(threadIdx.x),dh);
    const int pos=k.len0[s]+i;
    const float* src=workspace(a).qkv+int64_t(l)*2*D;
    float* dst=full_cache_mut(a)+k.yoff(s)+int64_t(i)*2*D+kv_layer_stride(a)*layer;
    for(int d=threadIdx.x;d<2*D;d+=threads)dst[d]=src[d];
    __syncthreads();
    for(int j=threadIdx.x;j<D/2;j+=threads) {
        const int head=j/(dh/2),d=j%(dh/2),at=head*dh+d;
        const float angle=float(pos)*inv[d];
        const float co=cosf(angle),sn=sinf(angle),lo=dst[at],hi=dst[at+dh/2];
        dst[at]=lo*co-hi*sn;dst[at+dh/2]=hi*co+lo*sn;
    }
}

/* K|V for ONE layer of the current chunk: the policy's K|V GEMM over the normalised staging, then the
 * store kernel above. `rows_per_seq = 1` validates the K|V tile per staged row, exactly like the
 * projection, so a dead row contributes zeros and never reaches the cache. */
template<int TAG,class POLICY>
__host__ inline int full_append_kv(DeviceArgs& a,const POLICY& p,int r0,int rows,int layer) {
    const auto& c=a.c;
    Work w=workspace(a);
    AppendCtl k=append_ctl(a);
    GemmReq kv{};
    kv.x=w.norm;kv.w0=full_weight(a,DL_K_PROJ,layer);kv.w1=full_weight(a,DL_V_PROJ,layer);
    kv.y=w.qkv;kv.rows=rows;kv.in=c.hidden_size;kv.out=2*(c.num_kv_heads*c.head_dim);
    kv.xstride=c.hidden_size;kv.rows_per_seq=1;kv.kind=2;kv.valid=k.rowseen;kv.live=k.live;
    kv.split=c.num_kv_heads*c.head_dim;
    int rc=p.gemm(kv);if(rc)return rc;
    void* args[]={&a,&r0,&rows,&layer};
    const void* store=reinterpret_cast<const void*>(&full_append_store<TAG>);
    return send(store,dim3(rows>0?rows:1),threads,args,p.stream_of());
}

/* Publishes the new verified length once, after the last layer's rows are in the cache. A legal empty
 * chunk republishes the length it already had. */
template<int TAG>
__global__ __launch_bounds__(threads) void full_append_publish(DeviceArgs a) {
    if(threadIdx.x!=0)return;
    AppendCtl k=append_ctl(a);const int s=blockIdx.x;
    if(s>=a.c.n_seq||k.bad[0]||!k.ok[s])return;
    full_length_mut(a)[k.slot[s]]=k.len0[s]+k.limit[s];
}

/* The append's whole stage sequence: chunked over the batch's flat new-row set, so neither the batch
 * width nor `tap_rows` can make the arena grow, and one projection per verified row and layer. The
 * chunk boundaries come from the control block, which the scan has already published. */
template<int TAG,class POLICY>
__host__ inline int full_append_run(DeviceArgs& a,const POLICY& p) {
    const auto& c=a.c;
    const hipStream_t stream=p.stream_of();
    void* scan[]={&a};
    int rc=send(reinterpret_cast<const void*>(full_append_scan<TAG>),dim3(c.n_seq),threads,scan,stream);
    if(rc)return rc;
    rc=send(reinterpret_cast<const void*>(full_append_layout<TAG>),dim3(1),threads,scan,stream);
    if(rc)return rc;
    /* The batch's total new-row count lives in the control block, which only the device knows; the loop
     * bound is therefore the largest count any sequence can contribute. That is n_seq*max(tap_rows,16):
     * `tap_rows` bounds a contiguous span, 16 bounds a path, they may differ, and the schema permits
     * spans to overlap, so no tighter bound exists without inspecting the metadata on the host. Some
     * chunks in that range can therefore be empty. They cost the gather plus the one-CTA count kernel,
     * which publishes a zero live count and a zeroed validity table; the GEMMs then still run over the
     * staged tile with every row's accumulator zeroed, so an empty chunk cannot reach the cache and
     * cannot read uninitialised staging, but it is NOT free of GEMM work. No speedup is claimed from
     * the bound: it is a correctness-preserving upper bound, and the honest description is "empty
     * chunks cost their tile's arithmetic and nothing else".
     *
     * ponytail: tighten by publishing rowoff[n_seq] through a host-visible counter only if a profile
     * shows those empty chunks matter; the step path may not read device memory, so it is a measurement
     * question, not a design one. */
    const int64_t per_seq=c.tap_rows>16?c.tap_rows:16;
    const int64_t upper=int64_t(c.n_seq)*per_seq;
    for(int64_t off=0;off<upper;off+=chunk_rows) {
        int rows=int64_t(chunk_rows)<upper-off?chunk_rows:int(upper-off);
        int r0=int(off);          /* non-const: hipLaunchKernel's args array is void*[] */
        void* args[]={&a,&r0,&rows};
        rc=send(reinterpret_cast<const void*>(full_append_gather<TAG>),dim3(rows),threads,args,stream);
        if(rc)return rc;
        rc=send(reinterpret_cast<const void*>(full_append_count<TAG>),dim3(1),threads,args,stream);
        if(rc)return rc;
        rc=full_append_project<TAG>(a,p,rows,c.context_features_dim);if(rc)return rc;
        for(int layer=0;layer<c.num_layers;++layer) {
            rc=full_append_kv<TAG>(a,p,r0,rows,layer);if(rc)return rc;
        }
    }
    return send(reinterpret_cast<const void*>(full_append_publish<TAG>),dim3(c.n_seq),threads,scan,stream);
}

} /* namespace rad_dloop */
#endif
