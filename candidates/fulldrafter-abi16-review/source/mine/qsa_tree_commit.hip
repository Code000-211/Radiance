#include "qsa_tree_commit.h"
#include "../libr4d/r4d_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
static_assert(sizeof(RadQsaTreeCommitCall)<=768, "AQL kernarg slot bound");

namespace {
__global__ __launch_bounds__(256) void qsa_tree_commit_kernel(RadQsaTreeCommitCall c) {
    const int seq=int(blockIdx.x), t=int(threadIdx.x);
    __shared__ int valid;
    __shared__ uint16_t tail[32*256];
    __shared__ float norm[256], red[8];
    if (!t) valid=rad_qsa_tree_metadata(c);
    __syncthreads();
    if (!valid) return; // A bad LAST sequence must also stop every earlier sequence's writes.
    const int slot=c.sidx[int64_t(seq)*c.sidx_ld], count=rad_qsa_tree_length(c,seq);
    if (slot<0 || !count) return;
    if (t<c.n) for (int r=0; r<c.ring; ++r)
        tail[r*c.n+t]=c.tail[int64_t(slot)*c.tail_slot+int64_t(r)*c.tail_row+t];
    __syncthreads();
    for (int j=0; j<count; ++j) {
        const int tok=rad_qsa_tree_token(c,seq,j), p=c.pos[int64_t(tok)*c.pos_stride];
        const uint16_t key=t<c.n ? c.k[int64_t(tok)*c.k_ld+t] : uint16_t(0);
        const int page=(p+1)%c.ratio==0 ? rad_qsa_tree_page(c,seq,p) : -1;
        if (page>=0) {
            // Identical to qsa_work's single-token gather followed by qsa_block_key.
            float acc=0.0f;
            if (t<c.n) {
                const int first=p-c.ratio+1;
                for (int i=0; i<c.ratio; ++i) {
                    const int q=first+i;
                    const uint16_t raw=q==p ? key : tail[(q%c.ring)*c.n+t];
                    acc+=__bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(&raw));
                }
                acc=__bfloat162float(__float2bfloat16(acc/float(c.ratio)));
            }
            float ss=t<c.n ? acc*acc : 0.0f;
            ss=r4d_wave_sum(ss);
            if (!(t&31)) red[t>>5]=ss;
            __syncthreads();
            float sum=0.0f;
            const int nw=(int(blockDim.x)+31)/32;
            for (int i=0; i<nw; ++i) sum+=red[i];
            const float inv=rsqrtf(sum/float(c.n)+c.eps);
            if (t<c.n) norm[t]=__bfloat162float(__float2bfloat16(acc*inv*(c.wadd+c.weight[t])));
            __syncthreads();
            if (t<c.n) {
                float res=norm[t];
                if (t<c.rot) {
                    const int half=c.rot>>1;
                    const bool first=t<half;
                    const int pt=first ? t+half : t-half, ci=first ? t : t-half;
                    const float position=float(p-c.ratio+1)/(c.scale!=0.0f ? c.scale : 1.0f);
                    const float iv=1.0f/powf(c.theta,float(2*ci)/float(c.rot));
                    float sine, cosine;
                    sincosf(position*iv,&sine,&cosine);
                    const float xp=norm[pt];
                    res=first ? (res*cosine-xp*sine) : (xp*sine+res*cosine);
                }
                reinterpret_cast<__hip_bfloat16*>(c.out)[int64_t(page)*c.out_ld+t]=__float2bfloat16(res);
            }
            __syncthreads(); // No wave can replace norm[] before its rotary partner reads it.
        }
        // LDS, not a global prefix write: the next token sees exactly this token's old-chain store.
        if (t<c.n) tail[(p%c.ring)*c.n+t]=key;
        __syncthreads();
    }
    // Only the final owner of each touched ring row publishes it, including paths longer than ring.
    const int first=count>c.ring ? count-c.ring : 0;
    if (t<c.n) for (int j=first; j<count; ++j) {
        const int row=rad_qsa_tree_position(c,seq,j)%c.ring;
        c.tail[int64_t(slot)*c.tail_slot+int64_t(row)*c.tail_row+t]=tail[row*c.n+t];
    }
}
}

int rad_mine_qsa_tree_commit_launch(const RadArgs* a, RadStream stream) {
    RadQsaTreeCommitCall c{};
    const int rc=rad_qsa_tree_commit_args(a,&c);
    if (rc!=RAD_OK) return rc;
    void* args[]={&c};
    const unsigned threads=unsigned((c.n+31)/32*32);
    return hipLaunchKernel(reinterpret_cast<const void*>(qsa_tree_commit_kernel),dim3(c.seqs),
        dim3(threads),args,0,reinterpret_cast<hipStream_t>(stream))==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

int rad_mine_qsa_tree_commit_count(void) { return 1; }
const RadKernelInfo* rad_mine_qsa_tree_commit_at(int i) {
    static const RadKernelInfo row=rad_qsa_tree_commit_row(rad_mine_qsa_tree_commit_launch);
    return i==0 ? &row : nullptr;
}
