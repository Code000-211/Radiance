#include "dloop_full.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

/* Independent host oracle for the three full-context ops.
 *
 * Independent of the RING oracle in the sense that matters: this path never builds a C-wide
 * activation plane (the ring oracle projects C+1 context rows per launch and allocates C*B*H of
 * scratch to hold them), it never rebuilds history, and its attention reads the F32 cache where
 * the ring path reads a resettable BF16 ring. The arithmetic itself -- the same projections, the
 * same grouped dynamic convolution, the same RMSNorm and SiLU -- is deliberately the same, because
 * the two model contracts differ only in how much history is visible and where K/V lives.
 *
 * Nothing here allocates: every row is projected once, in place, into the F32 cache at the tap
 * row's absolute position, and the whole tree works out of one arena per sequence. */
namespace {
float bf(uint16_t x) {return rad_bf16_to_f32(x);}
const uint16_t* bt(const RadArgs* a,int i) {return static_cast<const uint16_t*>(a->t[i].data);}
const float* ft(const RadArgs* a,int i) {return static_cast<const float*>(a->t[i].data);}
const int32_t* ixt(const RadArgs* a,int i) {return static_cast<const int32_t*>(a->t[i].data);}
float* fot(const RadArgs* a,int i) {return static_cast<float*>(a->t[i].data);}
int32_t* iot(const RadArgs* a,int i) {return static_cast<int32_t*>(a->t[i].data);}

float dot(const float* x,const uint16_t* w,int n) {
    float z=0;for(int k=0;k<n;++k)z+=x[k]*bf(w[k]);return z;
}
void linear(const float* x,const uint16_t* w,float* y,int rows,int in,int out) {
    for(int t=0;t<rows;++t)for(int j=0;j<out;++j)y[(int64_t)t*out+j]=dot(x+(int64_t)t*in,w+(int64_t)j*in,in);
}
void norm(const float* x,const uint16_t* w,float* y,int rows,int h,float eps) {
    for(int t=0;t<rows;++t) {float sum=0;for(int j=0;j<h;++j)sum+=x[(int64_t)t*h+j]*x[(int64_t)t*h+j];
        const float scale=1.0f/std::sqrt(sum/h+eps);
        for(int j=0;j<h;++j)y[(int64_t)t*h+j]=x[(int64_t)t*h+j]*scale*bf(w[j]);}
}
/* Grouped dynamic convolution over the block, identical to the ring oracle's: the kernel taps the
 * previous block row with a grouped multiplier, and cannot reach past row 0 into history. */
void conv(const RadDLoopConfig& c,const float* x,const uint16_t* base,const float* dyn,int branch,float* y) {
    const int H=c.hidden_size,G=H/c.conv_group_size,K=c.conv_kernel_size,B=c.proposal_length+1;
    for(int t=0;t<B;++t)for(int h=0;h<H;++h) {float sum=0;
        for(int o=0;o<K&&o<=t;++o) {const float v=x[(int64_t)(t-o)*H+h];
            sum+=bf(base[(int64_t)(branch*K+o)*H+h])*v;
            sum+=dyn[((int64_t)(t*2+branch)*K+o)*G+h/c.conv_group_size]*v;}
        y[(int64_t)t*H+h]=sum;}
}
/* Rotation at ABSOLUTE positions: `first` is the sequence position of row 0 -- length[slot] for
 * the block's own Q/K, the position itself for a freshly projected context row. */
void rope(float* x,int rows,int heads,int dim,int first,float theta) {
    for(int t=0;t<rows;++t)for(int h=0;h<heads;++h)for(int j=0;j<dim/2;++j) {
        float* p=x+((int64_t)t*heads+h)*dim;
        // The source's `1.0 / (theta ** (2j/dim))`, not powf(theta, -2j/dim): that is a different
        // function, one ulp away at many pairs, and an absolute position multiplies the error. The
        // device kernels use the same form (rope_inv_freq in dloop_full_device.h).
        const float angle=(first+t)*(1.0f/std::pow(theta,2.0f*j/dim));
        const float co=std::cos(angle),si=std::sin(angle),a=p[j],z=p[j+dim/2];
        p[j]=a*co-z*si;p[j+dim/2]=z*co+a*si;}
}
bool better(float score,int id,float old,int oid) {return score>old||(score==old&&(oid<0||id<oid));}
void insert(float score,int id,float* scores,int* ids,int k) {
    for(int j=0;j<k;++j)if(better(score,id,scores[j],ids[j])) {
        for(int z=k-1;z>j;--z) {scores[z]=scores[z-1];ids[z]=ids[z-1];}
        scores[j]=score;ids[j]=id;break;}
}

/* One tap row, all three planes, projected into every layer's K/V at `position`.
 *
 * `rowbuf` is the first F+H elements of the append arena: the concatenated tap row, then the
 * projected-and-normalised context row that BOTH K and V are read from -- which is what the ring
 * contract means by a context row being shared by K and V (ref_dloop.cpp: base projects ctxh and
 * feeds it to both projections at line ~"linear(ctxh,w(DL_K_PROJ,l)...)").
 *
 * The device may use the same region, which is why the arena names it. */
void project_row(const RadDLoopFullConfig& c,const uint16_t* weights,const uint16_t* tap0,
    const uint16_t* tap1,const uint16_t* tap2,int64_t row,float* cache,int slot,int position,
    float* rowbuf) {
    const RadDLoopConfig& m=c.model;
    const int H=m.hidden_size,F=m.context_features_dim,W=m.target_width;
    const int D=m.num_kv_heads*m.head_dim;
    for(int j=0;j<F;++j) {
        const uint16_t* plane=j<W?tap0:(j<2*W?tap1:tap2);
        const int off=j<W?j:(j<2*W?j-W:j-2*W);
        rowbuf[j]=bf(plane[row*W+off]);
    }
    const uint16_t* proj=weights+rad_dloop_weight_offset(m,DL_CONTEXT_PROJ);
    const uint16_t* nrm=weights+rad_dloop_weight_offset(m,DL_CONTEXT_NORM);
    float* ctx=rowbuf+F;
    linear(rowbuf,proj,ctx,1,F,H);
    norm(ctx,nrm,ctx,1,H,m.eps);
    for(int l=0;l<m.num_layers;++l) {
        float* k=cache+rad_dloop_full_kv_index(c,slot,l,position,0);
        float* v=k+D;
        linear(ctx,weights+rad_dloop_weight_offset(m,DL_K_PROJ,l),k,1,H,D);
        linear(ctx,weights+rad_dloop_weight_offset(m,DL_V_PROJ,l),v,1,H,D);
        rope(k,1,m.num_kv_heads,m.head_dim,position,m.rope_theta);
    }
}

/* Full-prefix attention for one block row of one layer: length[slot] cached positions (K already
 * roped at absolute positions when it was appended) plus the entire block. Reads the cache
 * operand directly -- `score` holds one row of exp weights and nothing else is materialised. */
void attend_row(const RadDLoopFullConfig& c,const float* cache,int slot,int layer,
    const float* q,float* att,const float* kblk,const float* vblk,float* score,int length,int t) {
    const RadDLoopConfig& m=c.model;
    const int H=m.hidden_size,D=m.num_kv_heads*m.head_dim,B=m.proposal_length+1;
    const int nkv=m.num_kv_heads,rep=m.num_q_heads/nkv,hd=m.head_dim;
    const int total=length+B;
    const float scale=1.0f/std::sqrt((float)hd);
    for(int h=0;h<m.num_q_heads;++h) {
        const int kh=h/rep;
        const float* qh=q+(int64_t)t*H+(int64_t)h*hd;
        float max=-INFINITY;
        for(int j=0;j<total;++j) {
            const float* row=j<length?cache+rad_dloop_full_kv_index(c,slot,layer,j,kh*hd)
                                     :kblk+((int64_t)(j-length)*nkv+kh)*hd;
            float s=0;for(int d=0;d<hd;++d)s+=qh[d]*row[d];
            s*=scale;score[j]=s;max=std::max(max,s);
        }
        float sum=0;for(int j=0;j<total;++j) {score[j]=std::exp(score[j]-max);sum+=score[j];}
        float* out=att+(int64_t)t*H+(int64_t)h*hd;
        for(int d=0;d<hd;++d) {
            float acc=0;
            for(int j=0;j<total;++j) {
                const float* row=j<length?cache+rad_dloop_full_kv_index(c,slot,layer,j,D+kh*hd)
                                         :vblk+((int64_t)(j-length)*nkv+kh)*hd;
                acc+=(score[j]/sum)*row[d];
            }
            out[d]=acc;
        }
    }
}

/* The five-layer backbone over the block plus the whole cached prefix. Writes only y[0..M*W).
 *
 * `plen` extra prefix rows (the path a tree node descends through) are fed AFTER the block, which
 * is how the ring oracle's forward places them: block row t<=plen takes prefix[t-1]. */
void forward_full(const RadDLoopFullConfig& c,const uint16_t* weights,const float* cache,int slot,
    int length,const uint16_t* anchor,const uint16_t* const* prefix,int plen,float* y,
    float* arena) {
    const RadDLoopConfig& m=c.model;
    const int H=m.hidden_size,B=m.proposal_length+1,D=m.num_kv_heads*m.head_dim;
    const int I=m.intermediate_size,G=H/m.conv_group_size;
    const RadDLoopFullScratch s=rad_dloop_full_scratch_layout(c,RAD_DLOOP_FULL_BASE);
    float* block=arena+s.block,*z=arena+s.z,*pre=arena+s.pre,*q=arena+s.q,*att=arena+s.att;
    float* out=arena+s.out,*kblk=arena+s.kblk,*vblk=arena+s.vblk,*dyn=arena+s.dyn;
    float* gate=arena+s.gate,*up=arena+s.up,*score=arena+s.score;
    auto w=[&](int id,int l=0) {return weights+rad_dloop_weight_offset(m,id,l);};
    for(int t=0;t<B;++t)for(int h=0;h<H;++h) {
        if(t==0||t<=plen) {const uint16_t* emb=t?prefix[t-1]:anchor;float sum=0;
            for(int j=0;j<m.target_width;++j)
                sum+=bf(emb[j])*bf(w(DL_TOKEN_PROJ)[(int64_t)h*m.target_width+j]);
            block[(int64_t)t*H+h]=sum;
        } else block[(int64_t)t*H+h]=bf(w(DL_MASK_TOKEN)[h]);
    }
    for(int l=0;l<m.num_layers;++l) {
        norm(block,w(DL_INPUT_NORM,l),z,B,H,m.eps);
        const float* x=z;
        if(m.conv_kernel_size) {linear(z,w(DL_ATTN_KERNEL,l),dyn,B,H,4*G);conv(m,z,w(DL_ATTN_BASE,l),dyn,0,pre);x=pre;}
        linear(x,w(DL_Q_PROJ,l),q,B,H,H);
        linear(x,w(DL_K_PROJ,l),kblk,B,H,D);
        linear(x,w(DL_V_PROJ,l),vblk,B,H,D);
        /* Absolute positions: the cache already holds [0,length), the block starts at length. */
        rope(q,B,m.num_q_heads,m.head_dim,length,m.rope_theta);
        rope(kblk,B,m.num_kv_heads,m.head_dim,length,m.rope_theta);
        for(int t=0;t<B;++t)
            /* attend_row applies the row offset itself for both q and att. */
            attend_row(c,cache,slot,l,q,att,kblk,vblk,score,length,t);
        linear(att,w(DL_O_PROJ,l),out,B,H,H);
        if(m.conv_kernel_size) {conv(m,out,w(DL_ATTN_BASE,l),dyn,1,pre);x=pre;} else x=out;
        for(int64_t j=0;j<(int64_t)B*H;++j)block[j]+=x[j];
        norm(block,w(DL_POST_NORM,l),z,B,H,m.eps);x=z;
        if(m.conv_kernel_size) {linear(z,w(DL_MLP_KERNEL,l),dyn,B,H,4*G);conv(m,z,w(DL_MLP_BASE,l),dyn,0,pre);x=pre;}
        linear(x,w(DL_GATE_PROJ,l),gate,B,H,I);linear(x,w(DL_UP_PROJ,l),up,B,H,I);
        for(int64_t j=0;j<(int64_t)B*I;++j)gate[j]=(gate[j]/(1.0f+std::exp(-gate[j])))*up[j];
        linear(gate,w(DL_DOWN_PROJ,l),out,B,I,H);
        if(m.conv_kernel_size) {conv(m,out,w(DL_MLP_BASE,l),dyn,1,pre);x=pre;} else x=out;
        for(int64_t j=0;j<(int64_t)B*H;++j)block[j]+=x[j];
    }
    norm(block+H,w(DL_FINAL_NORM),z,B-1,H,m.eps);
    linear(z,w(DL_OUTPUT_PROJ),y,B-1,H,m.target_width);
}

/* Deterministic active-vocabulary shard top-k of M rows. Slots no id fills keep -1/-infinity. */
int shard_topk(const RadDLoopFullConfig& c,const uint16_t* head,const float* y,int rows,
    int32_t* ids,float* scores) {
    const int K=c.model.pool_size,W=c.model.target_width;
    for(int t=0;t<rows;++t) {
        int* row_ids=ids+(int64_t)t*K;float* row_scores=scores+(int64_t)t*K;
        std::fill(row_ids,row_ids+K,-1);std::fill(row_scores,row_scores+K,-INFINITY);
        for(int j=0;j<c.model.vocab_size_shard&&c.model.vocab_lo+j<c.model.active_vocab;++j) {
            const float score=dot(y+(int64_t)t*W,head+(int64_t)j*W,W);
            if(!std::isfinite(score))return RAD_E_INVAL;
            insert(score,c.model.vocab_lo+j,row_scores,row_ids,K);
        }
    }
    return RAD_OK;
}

/* Everything that must fail BEFORE the first write. Returns the status the launch reports with
 * every operand untouched. */
int validate(const RadArgs* a,const RadDLoopFullConfig& c,int op) {
    const RadDLoopConfig& m=c.model;
    for(int s=0;s<m.n_seq;++s) {
        const int slot=ixt(a,2+(op==RAD_DLOOP_FULL_APPEND?3:0))[s];
        if(slot<0||slot>=m.max_seqs)return RAD_E_INVAL;
        if(op==RAD_DLOOP_FULL_APPEND) {
            /* A slot is a SCATTER destination here: two sequences appending into one slot would
             * race, and the plan makes the batch's slots unique. A read (base/tree) may name the
             * same slot twice -- it is the same prefix, which is merely unusual, not a fault. */
            for(int q=0;q<s;++q)if(ixt(a,5)[q]==slot)return RAD_E_INVAL;
            const int reset=ixt(a,6)[s];
            if(reset!=0&&reset!=1)return RAD_E_INVAL;
            const int32_t start=ixt(a,4)[2*s],count=ixt(a,4)[2*s+1];
            int64_t rows=0;
            if(count>=0) {
                if(start<0||count>m.tap_rows||(int64_t)start+count>m.tap_rows)return RAD_E_INVAL;
                rows=count;
            } else if(count==-1) {
                /* A -1 fills the UNUSED columns; a full sixteen needs no terminator (root plus
                 * fifteen draft nodes is a legal complete path). Every column is examined, so a
                 * stray positive behind the terminator is caught rather than silently ignored. */
                bool ended=false;
                for(int j=0;j<16;++j) {
                    const int32_t v=ixt(a,3)[s*16+j];
                    if(v==-1) {ended=true;continue;}
                    if(ended||v<0||v>=m.tap_rows)return RAD_E_INVAL;
                    ++rows;
                }
            } else return RAD_E_INVAL;
            const int64_t len=reset?0:(int64_t)ixt(a,9)[slot];
            /* Capacity is a hard bound: never truncated, never wrapped. */
            if(len<0||len>c.context_capacity||len+rows>c.context_capacity)return RAD_E_INVAL;
        } else {
            const int len=ixt(a,1)[slot];
            if(len<1||len>c.context_capacity)return RAD_E_INVAL;
        }
    }
    if(op==RAD_DLOOP_FULL_DARTREE) {
        const int64_t rows=(int64_t)m.n_seq*m.proposal_length*m.pool_size;
        const int64_t R=m.selector_rank,W=m.target_width;
        for(int64_t i=0;i<rows;++i) {
            const int id=ixt(a,6)[i];
            if(id<0||id>=m.active_vocab||(i%m.pool_size&&id<=ixt(a,6)[i-1])||!std::isfinite(ft(a,7)[i]))
                return RAD_E_INVAL;
            /* Every gathered teacher/codebook row this candidate scores against must be a number:
             * a BF16 NaN in the shard poisons the whole tree, and it is a caller bug, not a model. */
            for(int64_t j=0;j<W;++j)
                if(!std::isfinite(bf(bt(a,8)[i*W+j]))||!std::isfinite(bf(bt(a,9)[i*W+j])))return RAD_E_INVAL;
            for(int64_t j=0;j<R;++j)
                if(!std::isfinite(bf(bt(a,10)[i*R+j]))||!std::isfinite(bf(bt(a,11)[i*R+j])))return RAD_E_INVAL;
        }
        for(int s=0;s<m.n_seq;++s) {
            if(a->t[16].data&&(ixt(a,16)[s]<0||ixt(a,16)[s]>=m.active_vocab))return RAD_E_INVAL;
            for(int64_t j=0;j<R;++j)if(!std::isfinite(bf(bt(a,12)[(int64_t)s*R+j])))return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}

int ref_append(const RadArgs* a) {
    RadDLoopFullConfig c{};int rc=rad_dloop_full_args(a,RAD_DLOOP_FULL_APPEND,&c);if(rc)return rc;
    rc=validate(a,c,RAD_DLOOP_FULL_APPEND);if(rc)return rc;
    const RadDLoopConfig& m=c.model;
    float* cache=fot(a,8);int32_t* len=iot(a,9);
    for(int s=0;s<m.n_seq;++s) {
        float* arena=static_cast<float*>(a->scratch)+
            (int64_t)s*rad_dloop_full_scratch_elements(c,RAD_DLOOP_FULL_APPEND);
        const int slot=ixt(a,5)[s];
        if(ixt(a,6)[s])len[slot]=0;
        const int32_t start=ixt(a,4)[2*s],count=ixt(a,4)[2*s+1];
        int64_t rows=0;
        if(count>=0)rows=count;
        else for(int j=0;j<16;++j)if(ixt(a,3)[s*16+j]!=-1)++rows;
        for(int64_t r=0;r<rows;++r) {
            const int row=count>=0?(int)start+(int)r:ixt(a,3)[s*16+r];
            project_row(c,bt(a,7),bt(a,0),bt(a,1),bt(a,2),row,cache,slot,(int)len[slot],arena);
            ++len[slot];
        }
    }
    return RAD_OK;
}

int ref_base(const RadArgs* a) {
    RadDLoopFullConfig c{};int rc=rad_dloop_full_args(a,RAD_DLOOP_FULL_BASE,&c);if(rc)return rc;
    rc=validate(a,c,RAD_DLOOP_FULL_BASE);if(rc)return rc;
    const RadDLoopConfig& m=c.model;
    const int M=m.proposal_length,W=m.target_width,K=m.pool_size;
    for(int s=0;s<m.n_seq;++s) {
        float* arena=static_cast<float*>(a->scratch)+
            (int64_t)s*rad_dloop_full_scratch_elements(c,RAD_DLOOP_FULL_BASE);
        const int slot=ixt(a,2)[s],length=ixt(a,1)[slot];
        float* y=fot(a,6)+(int64_t)s*M*W;
        forward_full(c,bt(a,4),ft(a,0),slot,length,bt(a,3)+(int64_t)s*W,nullptr,0,y,arena);
        rc=shard_topk(c,bt(a,5),y,M,iot(a,7)+(int64_t)s*M*K,fot(a,8)+(int64_t)s*M*K);
        if(rc)return rc;
    }
    return RAD_OK;
}

int ref_dartree(const RadArgs* a) {
    RadDLoopFullConfig c{};int rc=rad_dloop_full_args(a,RAD_DLOOP_FULL_DARTREE,&c);if(rc)return rc;
    rc=validate(a,c,RAD_DLOOP_FULL_DARTREE);if(rc)return rc;
    const RadDLoopConfig& m=c.model;
    const int M=m.proposal_length,K=m.pool_size,W=m.target_width,R=m.selector_rank;
    const RadDLoopFullScratch lay=rad_dloop_full_scratch_layout(c,RAD_DLOOP_FULL_DARTREE);
    const int cap=rad_dloop_full_tree_capacity(c);       /* nodes before pruning */
    const int maxn=cap+1;                                /* root plus them */
    const int fmax=rad_dloop_full_frontier_max(c);
    for(int s=0;s<m.n_seq;++s) {
        float* arena=static_cast<float*>(a->scratch)+
            (int64_t)s*rad_dloop_full_scratch_elements(c,RAD_DLOOP_FULL_DARTREE);
        const int slot=ixt(a,2)[s],length=ixt(a,1)[slot];
        const int32_t* pool_ids=ixt(a,6)+(int64_t)s*M*K;
        const float* unary=ft(a,7)+(int64_t)s*M*K;
        const uint16_t* pool_head=bt(a,8)+(int64_t)s*M*K*W;
        const uint16_t* pool_embed=bt(a,9)+(int64_t)s*M*K*W;
        const uint16_t* pool_pred=bt(a,10)+(int64_t)s*M*K*R;
        const uint16_t* anchor_pred=bt(a,12)+(int64_t)s*R;
        int32_t* nodes=reinterpret_cast<int32_t*>(arena+lay.nodes);
        int32_t* ntok=nodes,*npar=nodes+maxn,*ndepth=nodes+2*maxn,*nsrc=nodes+3*maxn;
        int32_t* order=reinterpret_cast<int32_t*>(arena+lay.idx);
        int32_t* keep=order+maxn,*remap=order+2*maxn;
        int32_t* front=reinterpret_cast<int32_t*>(arena+lay.front);
        int32_t* next=front+fmax;
        float* node_score=arena+lay.node_score;
        float* cand=arena+lay.cand;
        float* rank=arena+lay.rank;
        /* y_base is an IN operand and must survive the call untouched: it is the ROOT's hidden
         * state (upstream scores the root with `base[:, 0]`), and it is also the input a caller may
         * reuse across variants. The per-depth backbone output therefore goes to the arena's own
         * `y` region, never back into the operand. */
        const float* base=ft(a,5)+(int64_t)s*M*W;
        float* y=arena+lay.y;
        int count=1,nfront=1;
        npar[0]=-1;ndepth[0]=0;nsrc[0]=-1;node_score[0]=0;front[0]=0;
        const int width=std::min(std::min(m.width,K),m.budget);
        for(int d=0;d<M&&count<=cap&&nfront;++d) {
            const int total=nfront*K;
            /* `cand` holds one depth's candidate scores, parent-major; `count` is bounded by the
             * loop condition plus quota <= cap+1-count, so the node arrays only need their bound. */
            if(nfront>fmax)return RAD_E_SHAPE;
            for(int p=0;p<nfront;++p) {
                const int parent=front[p];
                const float* hidden;
                if(!d)hidden=base;          /* the base forward's row 0, upstream's root state */
                else {
                    /* Root-first path prefix of this depth: the node's ancestors' embeddings.
                     * proposal_length is at most 8, so a fixed 16 is the schema's own bound. */
                    const uint16_t* prefix[16]{};
                    int node=parent;
                    for(int t=d-1;t>=0;--t) {prefix[t]=pool_embed+(int64_t)nsrc[node]*W;node=npar[node];}
                    forward_full(c,bt(a,4),ft(a,0),slot,length,bt(a,3)+(int64_t)s*W,prefix,d,y,arena);
                    hidden=y+(int64_t)d*W;
                }
                linear(hidden,bt(a,4)+rad_dloop_weight_offset(m,DL_SELECTOR_PROJ),rank,1,W,R);
                const uint16_t* pred=parent?pool_pred+(int64_t)nsrc[parent]*R:anchor_pred;
                for(int r=0;r<R;++r)rank[r]*=bf(pred[r]);
                /* The parent's prediction multiplies the rank vector above and the successor
                 * codebook then scores every candidate; `pool_succ_rows` is gathered per candidate
                 * row, so it is addressed exactly like `pool_head`. */
                const uint16_t* succ_rows=bt(a,11)+(int64_t)s*M*K*R;
                float max=-INFINITY;
                for(int j=0;j<K;++j) {
                    const int off=d*K+j;
                    float value=d?dot(hidden,pool_head+(int64_t)off*W,W):unary[off];
                    value+=dot(rank,succ_rows+(int64_t)off*R,R);
                    if(!std::isfinite(value))return RAD_E_INVAL;
                    cand[(int64_t)p*K+j]=value;max=std::max(max,value);
                }
                float sum=0;for(int j=0;j<K;++j)sum+=std::exp(cand[(int64_t)p*K+j]-max);
                const float logsum=std::log(sum);
                for(int j=0;j<K;++j)cand[(int64_t)p*K+j]=
                    node_score[parent]+((cand[(int64_t)p*K+j]-max)-logsum);
            }
            const int remaining=cap+1-count;
            int quota=m.pruned?width:(remaining+M-d-1)/(M-d);
            quota=std::min(std::min(quota,remaining),total);
            for(int k=0;k<quota;++k) {
                int best=0;for(int i=1;i<total;++i)if(cand[i]>cand[best])best=i;
                const int node=count++,parent=front[best/K];
                ntok[node]=pool_ids[d*K+best%K];npar[node]=parent;ndepth[node]=d+1;
                nsrc[node]=d*K+best%K;node_score[node]=cand[best];
                cand[best]=-INFINITY;next[k]=node;
            }
            nfront=quota;std::copy(next,next+quota,front);
        }
        for(int i=1;i<count;++i)order[i-1]=i;
        std::fill(keep,keep+maxn,0);std::fill(remap,remap+maxn,-1);
        if(m.pruned&&count-1>m.budget) {
            /* Stable insertion sort: descending priority, ascending depth, then index. */
            auto before=[&](int x,int z) {
                const float xa=node_score[x]+m.depth_bonus*ndepth[x];
                const float zb=node_score[z]+m.depth_bonus*ndepth[z];
                return xa>zb||(xa==zb&&(ndepth[x]<ndepth[z]||(ndepth[x]==ndepth[z]&&x<z)));};
            for(int j=1;j<count-1;++j) {int v=order[j],z=j;while(z&&before(v,order[z-1])) {order[z]=order[z-1];--z;}order[z]=v;}
            for(int j=0;j<m.budget;++j)keep[order[j]]=1;
        } else for(int i=1;i<count;++i)keep[i]=1;
        int n=0;
        std::fill(iot(a,13)+(int64_t)s*16,iot(a,13)+(int64_t)(s+1)*16,-1);
        std::fill(iot(a,14)+(int64_t)s*16,iot(a,14)+(int64_t)(s+1)*16,-1);
        for(int i=1;i<count;++i)if(keep[i]) {
            if(npar[i]&&remap[npar[i]]<0)return RAD_E_INVAL;
            remap[i]=n;
            iot(a,13)[(int64_t)s*16+n]=ntok[i];
            iot(a,14)[(int64_t)s*16+n]=remap[npar[i]];
            ++n;
        }
        iot(a,15)[s]=n;
        if(a->t[17].data) {
            int* packed=iot(a,17)+(int64_t)s*33;
            std::fill(packed,packed+33,-1);
            packed[0]=n+1;packed[1]=ixt(a,16)[s];
            for(int j=0;j<n;++j) {
                packed[2+j]=iot(a,13)[(int64_t)s*16+j];
                packed[18+j]=iot(a,14)[(int64_t)s*16+j]+1;
            }
        }
    }
    return RAD_OK;
}
}  /* namespace */

int rad_ref_dloop_full_append(const RadArgs* a,RadStream) {return ref_append(a);}
int rad_ref_dloop_full_base(const RadArgs* a,RadStream) {return ref_base(a);}
int rad_ref_dloop_full_dartree(const RadArgs* a,RadStream) {return ref_dartree(a);}
int rad_ref_dloop_full_count(void) {return RAD_DLOOP_FULL_OPS;}
const RadKernelInfo* rad_ref_dloop_full_at(int i) {
    static const RadKernelInfo rows[]={
        rad_dloop_full_row(RAD_DLOOP_FULL_APPEND,RAD_DOMAIN_HOST,rad_ref_dloop_full_append),
        rad_dloop_full_row(RAD_DLOOP_FULL_BASE,RAD_DOMAIN_HOST,rad_ref_dloop_full_base),
        rad_dloop_full_row(RAD_DLOOP_FULL_DARTREE,RAD_DOMAIN_HOST,rad_ref_dloop_full_dartree)};
    return i>=0&&i<RAD_DLOOP_FULL_OPS?rows+i:nullptr;
}
