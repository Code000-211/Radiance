#include <hip/hip_runtime.h>
#include "mtp_scored_tree.h"
#include <cmath>

namespace {
__device__ float hybrid_value(RadMtpHybridLseCall c,int64_t s,int j) {
    float z=__uint_as_float(static_cast<uint32_t>(c.coarse[s*c.coarse_ld+j])<<16);
    const int64_t id=static_cast<int64_t>(c.vocab_off)+j;
    for(int k=0;k<8;++k) if(c.idx[s*c.idx_ld+k]==id) { z=c.val[s*c.val_ld+k]; break; }
    return z;
}
__global__ void hybrid_lse(RadMtpHybridLseCall c) {
    const int64_t s=blockIdx.x; const int lane=threadIdx.x;
    __shared__ float work[256]; __shared__ int bad[256];
    float mx=-INFINITY; int invalid=0;
    for(int64_t j=lane;j<c.N && static_cast<int64_t>(c.vocab_off)+j<c.vocab;j+=256) {
        const float z=hybrid_value(c,s,static_cast<int>(j));
        if(isnan(z) || z==INFINITY) invalid=1;
        mx=fmaxf(mx,z);
    }
    work[lane]=mx; bad[lane]=invalid; __syncthreads();
    for(int d=128;d;d/=2) { if(lane<d) { work[lane]=fmaxf(work[lane],work[lane+d]); bad[lane]|=bad[lane+d]; } __syncthreads(); }
    mx=work[0]; invalid=bad[0]; float sum=0;
    __syncthreads();
    if(isfinite(mx) && !invalid) for(int64_t j=lane;j<c.N && static_cast<int64_t>(c.vocab_off)+j<c.vocab;j+=256)
        sum+=expf(hybrid_value(c,s,static_cast<int>(j))-mx);
    work[lane]=sum; __syncthreads();
    for(int d=128;d;d/=2) { if(lane<d) work[lane]+=work[lane+d]; __syncthreads(); }
    if(lane==0) { float* out=c.stats+s*c.stats_ld; out[0]=invalid ? NAN : mx; out[1]=invalid ? NAN : work[0]; out[2]=0; out[3]=0; }
}
__global__ void scored_merge(RadMtpScoredMergeTreeAccumCall c) {
    const int64_t s=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(s>=c.M) return;
    const uint32_t* input=c.gathered+s*c.gathered_ld;
    uint64_t keys[16];
    for(int k=0;k<16;++k) {
        const int32_t id=static_cast<int32_t>(input[2*k]); uint32_t bits=input[2*k+1];
        bits=bits&0x80000000u ? ~bits : bits|0x80000000u;
        keys[k]=id<0 || id>=c.vocab ? 0 : (static_cast<uint64_t>(bits)<<32)|(0xffffffffu-input[2*k]);
    }
    int ids[40]; float logs[40]; float values[8];
    for(int k=0;k<8;++k) {
        uint64_t winner=0; for(int j=0;j<16;++j) if(keys[j]>winner) winner=keys[j];
        uint32_t bits=static_cast<uint32_t>(winner>>32);
        bits=winner ? (bits&0x80000000u ? bits&0x7fffffffu : ~bits) : 0xff800000u;
        const int id=winner ? static_cast<int32_t>(0xffffffffu-static_cast<uint32_t>(winner)) : -1;
        c.idx[s*c.idx_ld+k]=id; c.val[s*c.val_ld+k]=bits;
        if(c.policy!=0) values[k]=__uint_as_float(bits);
        ids[(c.round-1)*8+k]=id;
        for(int j=0;j<16;++j) if(keys[j]==winner) keys[j]=0;
    }
    int32_t* out=c.packed+s*c.out_ld;
    int root=-1;
    if(c.round==1) { const int ix=c.ids[s*c.id_stride]; if(ix>=0 && ix<c.token_count) root=c.tokens[static_cast<int64_t>(ix)*c.token_stride]; }
    else if(out[0]>=1 && out[0]<=c.capacity) root=out[1];
    bool valid=root>=0 && root<c.vocab;
    if(c.policy==0) {
        for(int r=0;r<c.max_rounds;++r) for(int k=0;k<8;++k) {
            int32_t* hp=c.history+r*c.hist_round_stride+s*c.hist_row_stride+k;
            if(c.round==1) *hp=-1;
            if(r<c.round-1) ids[r*8+k]=*hp;
        }
        for(int k=0;k<8;++k)
            c.history[(c.round-1)*c.hist_round_stride+s*c.hist_row_stride+k]=ids[(c.round-1)*8+k];
    } else {
        const float* st=c.stats+s*c.stats_ld;
        for(int rank=0;rank<2;++rank) {
            const float m=st[rank*4], q=st[rank*4+1];
            valid=valid && ((isfinite(m) && isfinite(q) && q>0) || (m==-INFINITY && q==0));
        }
        const float mx=fmaxf(st[0],st[4]);
        const double sum=(st[1]>0 ? static_cast<double>(st[1])*exp(static_cast<double>(st[0])-mx) : 0.0)+
                         (st[5]>0 ? static_cast<double>(st[5])*exp(static_cast<double>(st[4])-mx) : 0.0);
        valid=valid && isfinite(mx) && isfinite(sum) && sum>0;
        for(int r=0;r<c.max_rounds;++r) for(int k=0;k<8;++k) {
            const int j=r*8+k; int32_t* hp=c.history+r*c.hist_round_stride+s*c.hist_row_stride+k;
            float* lp=c.log_history+r*c.log_round_stride+s*c.log_row_stride+k;
            if(c.round==1) { *hp=-1; *lp=-INFINITY; }
            if(r<c.round-1) { ids[j]=*hp; logs[j]=*lp; }
        }
        const double base=c.round==1 ? 0.0 : logs[(c.round-2)*8];
        if(c.round>1 && (ids[(c.round-2)*8]<0 || ids[(c.round-2)*8]>=c.vocab || isnan(base) || base>0)) valid=false;
        for(int k=0;k<8;++k) {
            const int j=(c.round-1)*8+k; const float z=values[k];
            double l=-INFINITY;
            if(ids[j]>=0 && ids[j]<c.vocab) {
                if(isnan(z) || z==INFINITY) valid=false;
                else if(valid) { l=(static_cast<double>(z)-mx)-log(sum); if(l>1e-5) valid=false; else l=fmin(l,0.0)+base; }
            }
            logs[j]=static_cast<float>(l);
            c.history[(c.round-1)*c.hist_round_stride+s*c.hist_row_stride+k]=ids[j];
            c.log_history[(c.round-1)*c.log_round_stride+s*c.log_row_stride+k]=logs[j];
        }
    }
    int tok[8], par[8];
    for(int k=0;k<c.capacity;++k) { tok[k]=-1; par[k]=-1; }
    int n=0;
    if(valid) {
        tok[0]=root; n=1;
        if(c.policy==0) {
            int spine=0;
            for(int r=0;r<c.round && n<c.capacity;++r) {
                if(ids[r*8]<0 || ids[r*8]>=c.vocab) break;
                tok[n]=ids[r*8]; par[n]=n-1; ++n; ++spine;
            }
            for(int r=0;r<spine && n<c.capacity;++r) for(int k=1;k<8 && n<c.capacity;++k) {
                const int id=ids[r*8+k]; if(id<0 || id>=c.vocab) continue;
                bool dup=false; for(int j=1;j<n;++j) if(par[j]==r && tok[j]==id) dup=true;
                if(!dup) { tok[n]=id; par[n]=r; ++n; }
            }
        } else {
            int source[8];
            bool used[40]{};
            while(n<c.capacity) {
                int best=-1, parent=-1;
                for(int j=0;j<c.round*8;++j) {
                    if(used[j] || ids[j]<0 || ids[j]>=c.vocab || isnan(logs[j]) || logs[j]>0) continue;
                    const int depth=j/8; int p=depth==0 ? 0 : -1;
                    for(int i=1;i<n && p<0;++i) if(source[i]/8==depth-1 && tok[i]==ids[(depth-1)*8]) p=i;
                    if(p<0) continue;
                    bool dup=false; for(int i=1;i<n;++i) if(par[i]==p && tok[i]==ids[j]) dup=true;
                    if(dup) { used[j]=true; continue; }
                    if(best<0 || logs[j]>logs[best] || (logs[j]==logs[best] &&
                       (depth<best/8 || (depth==best/8 && (ids[j]<ids[best] || (ids[j]==ids[best] && j<best)))))) { best=j; parent=p; }
                }
                if(best<0) break;
                used[best]=true; tok[n]=ids[best]; par[n]=parent; source[n]=best; ++n;
            }
            if(c.policy==2) {
                int keep=1; double mass=1.0, best=-log(c.costs[0]);
                for(int i=1;i<n;++i) { mass+=exp(static_cast<double>(logs[source[i]])); const double objective=log(mass)-log(c.costs[i]); if(objective>best) { best=objective; keep=i+1; } }
                n=keep;
            }
        }
    }
    out[0]=n;
    for(int k=0;k<c.capacity;++k) { out[1+k]=k<n ? tok[k] : -1; out[1+c.capacity+k]=k<n ? par[k] : -1; }
}
}
int rad_mine_mtp_hybrid_lse_launch(const RadArgs* a,RadStream stream) {
    RadMtpHybridLseCall c{}; const int rc=rad_mtp_hybrid_lse_args(a,&c); if(rc) return rc;
    void* args[]={&c}; return hipLaunchKernel((const void*)hybrid_lse,dim3(static_cast<unsigned>(c.M)),dim3(256),args,0,reinterpret_cast<hipStream_t>(stream))==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int rad_mine_mtp_scored_merge_tree_accum_launch(const RadArgs* a,RadStream stream) {
    RadMtpScoredMergeTreeAccumCall c{}; const int rc=rad_mtp_scored_merge_tree_accum_args(a,&c); if(rc) return rc;
    void* args[]={&c}; return hipLaunchKernel((const void*)scored_merge,dim3(static_cast<unsigned>((c.M+63)/64)),dim3(64),args,0,reinterpret_cast<hipStream_t>(stream))==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int rad_mine_mtp_scored_tree_count() { return 2; }
const RadKernelInfo* rad_mine_mtp_scored_tree_at(int i) {
    static const RadKernelInfo rows[]={rad_mtp_hybrid_lse_row(rad_mine_mtp_hybrid_lse_launch),rad_mtp_scored_merge_tree_accum_row(rad_mine_mtp_scored_merge_tree_accum_launch)};
    return i>=0 && i<2 ? rows+i : nullptr;
}
