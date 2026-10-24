#include "qsa_tree_commit.h"

#include <cmath>
#include <cstring>

namespace {
/* The same three algorithms as the old GPU replay. The stage is caller scratch,
 * reused for one completed block; no vector or allocation exists on launch. */
void work(const RadQsaTreeCommitCall& c, int tok, int p, int slot, uint16_t* stage) {
    for (int i=0; i<c.ratio; ++i) {
        const int q=p-c.ratio+1+i;
        const uint16_t* src=q==p ? c.k+int64_t(tok)*c.k_ld :
            c.tail+int64_t(slot)*c.tail_slot+int64_t(q%c.ring)*c.tail_row;
        std::memcpy(stage+int64_t(i)*c.n,src,size_t(c.n)*2);
    }
}
void block_key(const RadQsaTreeCommitCall& c, int p, int page,
               const uint16_t* stage, float* pooled, float* norm) {
    const int threads=(c.n+31)/32*32;
    for (int d=0; d<threads; ++d) {
        float acc=0;
        if (d<c.n) {
            for (int i=0; i<c.ratio; ++i) acc+=rad_bf16_to_f32(stage[int64_t(i)*c.n+d]);
            acc=rad_bf16_to_f32(rad_f32_to_bf16(acc/float(c.ratio)));
        }
        pooled[d]=acc; norm[d]=acc*acc;
    }
    float sum=0;
    for (int wave=0; wave<threads; wave+=32) {
        // The lane-zero result of libr4d's XOR 16,8,4,2,1 butterfly, then wave-order addition.
        for (int offset=16; offset; offset>>=1)
            for (int d=0; d<offset; ++d) norm[wave+d]+=norm[wave+d+offset];
        sum+=norm[wave];
    }
    const float inv=1.0f/std::sqrt(sum/float(c.n)+c.eps);
    for (int d=0; d<c.n; ++d)
        norm[d]=rad_bf16_to_f32(rad_f32_to_bf16(pooled[d]*inv*(c.wadd+c.weight[d])));
    for (int d=0; d<c.n; ++d) {
        float res=norm[d];
        if (d<c.rot) {
            const int half=c.rot>>1;
            const bool first=d<half;
            const int pt=first ? d+half : d-half, ci=first ? d : d-half;
            const float position=float(p-c.ratio+1)/(c.scale!=0 ? c.scale : 1.0f);
            const float iv=1.0f/std::pow(c.theta,float(2*ci)/float(c.rot));
            const float sine=std::sin(position*iv), cosine=std::cos(position*iv);
            res=first ? (res*cosine-norm[pt]*sine) : (norm[pt]*sine+res*cosine);
        }
        c.out[int64_t(page)*c.out_ld+d]=rad_f32_to_bf16(res);
    }
}
void tail_store(const RadQsaTreeCommitCall& c, int tok, int p, int slot) {
    std::memcpy(c.tail+int64_t(slot)*c.tail_slot+int64_t(p%c.ring)*c.tail_row,
                c.k+int64_t(tok)*c.k_ld,size_t(c.n)*2);
}
}

int64_t rad_ref_qsa_tree_commit_scratch(const RadArgs* a) {
    if (!a || !a->p) return RAD_E_INVAL;
    const int64_t n=rad_param_geti(a->p,a->n_p,"n",0);
    const int64_t ratio=rad_param_geti(a->p,a->n_p,"ratio",0);
    const int64_t tokens=rad_param_getdim(a->p,a->n_p,"max_tokens",0);
    if (n<1 || n>256 || ratio<1 || ratio>32 || tokens<1 || tokens>INT32_MAX) return RAD_E_SHAPE;
    return ((ratio*n*2+3)&~int64_t(3))+2*((n+31)/32*32)*4;
}

int rad_ref_qsa_tree_commit(const RadArgs* a, RadStream) {
    RadQsaTreeCommitCall c{};
    int rc=rad_qsa_tree_commit_args(a,&c);
    if (rc!=RAD_OK) return rc;
    if (!rad_qsa_tree_metadata(c)) return RAD_E_INVAL;
    const int64_t need=rad_ref_qsa_tree_commit_scratch(a);
    if (need<0) return int(need);
    const uintptr_t scratch=reinterpret_cast<uintptr_t>(a->scratch);
    if (!scratch || (scratch&3) || a->scratch_bytes<need || scratch>UINTPTR_MAX-uint64_t(need))
        return RAD_E_SCRATCH;
    for (int i=0; i<10; ++i) {
        const auto& t=a->t[i];
        uint64_t span=1;
        for (unsigned d=0; d<t.rank; ++d) span+=uint64_t(t.shape[d]-1)*uint64_t(t.stride[d]);
        span*=t.dtype==RAD_BF16 ? 2 : 4;
        const uintptr_t p=reinterpret_cast<uintptr_t>(t.data);
        if (scratch<p+span && p<scratch+uint64_t(need)) return RAD_E_SCRATCH;
    }
    auto* stage=static_cast<uint16_t*>(a->scratch);
    auto* pooled=reinterpret_cast<float*>(static_cast<uint8_t*>(a->scratch)+((c.ratio*c.n*2+3)&~3));
    float* norm=pooled+(c.n+31)/32*32;
    for (int s=0; s<c.seqs; ++s) {
        const int slot=c.sidx[int64_t(s)*c.sidx_ld];
        if (slot<0) continue;
        for (int j=0; j<rad_qsa_tree_length(c,s); ++j) {
            const int tok=rad_qsa_tree_token(c,s,j), p=c.pos[int64_t(tok)*c.pos_stride];
            if ((p+1)%c.ratio==0) {
                const int page=rad_qsa_tree_page(c,s,p);
                if (page>=0) { work(c,tok,p,slot,stage); block_key(c,p,page,stage,pooled,norm); }
            }
            tail_store(c,tok,p,slot);
        }
    }
    return RAD_OK;
}

int rad_ref_qsa_tree_commit_count(void) { return 1; }
const RadKernelInfo* rad_ref_qsa_tree_commit_at(int i) {
    static const RadKernelInfo row=[] {
        RadKernelInfo r=rad_qsa_tree_commit_row(rad_ref_qsa_tree_commit);
        r.name="ref_qsa_tree_commit"; r.domain=RAD_DOMAIN_HOST; r.priority=0;
        r.scratch=rad_ref_qsa_tree_commit_scratch;
        return r;
    }();
    return i==0 ? &row : nullptr;
}
