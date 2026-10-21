#include "ref_dloop.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {
float bf(uint16_t x){return rad_bf16_to_f32(x);}
const uint16_t* b(const RadArgs* a,int i){return static_cast<const uint16_t*>(a->t[i].data);}
const float* f(const RadArgs* a,int i){return static_cast<const float*>(a->t[i].data);}
const int32_t* ix(const RadArgs* a,int i){return static_cast<const int32_t*>(a->t[i].data);}
float* fo(const RadArgs* a,int i){return static_cast<float*>(a->t[i].data);}
int32_t* io(const RadArgs* a,int i){return static_cast<int32_t*>(a->t[i].data);}
float dot(const float* x,const uint16_t* w,int n){float z=0;for(int k=0;k<n;++k)z+=x[k]*bf(w[k]);return z;}
void linear(const float* x,const uint16_t* w,float* y,int rows,int in,int out){
    for(int t=0;t<rows;++t)for(int j=0;j<out;++j)y[t*out+j]=dot(x+t*in,w+(int64_t)j*in,in);
}
void norm(const float* x,const uint16_t* w,float* y,int rows,int h,float eps){
    for(int t=0;t<rows;++t){float sum=0;for(int j=0;j<h;++j)sum+=x[t*h+j]*x[t*h+j];
        const float scale=1.0f/std::sqrt(sum/h+eps);
        for(int j=0;j<h;++j)y[t*h+j]=x[t*h+j]*scale*bf(w[j]);}
}
void conv(const RadDLoopConfig& c,const float* x,const uint16_t* base,const float* dyn,int branch,float* y){
    const int H=c.hidden_size,G=H/c.conv_group_size,K=c.conv_kernel_size,B=c.proposal_length+1;
    for(int t=0;t<B;++t)for(int h=0;h<H;++h){float sum=0;
        for(int o=0;o<K&&o<=t;++o){const float v=x[(t-o)*H+h];
            sum+=bf(base[(branch*K+o)*H+h])*v;
            sum+=dyn[((t*2+branch)*K+o)*G+h/c.conv_group_size]*v;}
        y[t*H+h]=sum;}
}
void rope(float* x,int rows,int heads,int dim,int position,float theta){
    for(int t=0;t<rows;++t)for(int h=0;h<heads;++h)for(int j=0;j<dim/2;++j){
        float* p=x+(t*heads+h)*dim;const float angle=(position+t)*std::pow(theta,-2.0f*j/dim);
        const float co=std::cos(angle),si=std::sin(angle),a=p[j],z=p[j+dim/2];
        p[j]=a*co-z*si;p[j+dim/2]=z*co+a*si;}
}
bool better(float score,int id,float old,int oid){return score>old||(score==old&&(oid<0||id<oid));}
void insert(float score,int id,float* scores,int* ids,int k){
    for(int j=0;j<k;++j)if(better(score,id,scores[j],ids[j])){
        for(int z=k-1;z>j;--z){scores[z]=scores[z-1];ids[z]=ids[z-1];}
        scores[j]=score;ids[j]=id;break;}
}
int metadata(const RadArgs* a,const RadDLoopConfig& c,int op){
    const int si=op==0?2:op==1?3:2,li=op==0?6:op==1?2:1,hi=op==0?5:1;
    if(op==2)return RAD_OK;
    for(int s=0;s<c.n_seq;++s){const int slot=ix(a,si)[s];if(slot<0||slot>=c.max_seqs)return RAD_E_INVAL;
        if(op==0){if(ix(a,3)[s]!=0&&ix(a,3)[s]!=1)return RAD_E_INVAL;
            for(int q=0;q<s;++q)if(ix(a,si)[q]==slot)return RAD_E_INVAL;
            bool ended=false;for(int j=0;j<16;++j){int v=ix(a,1)[s*16+j];if(v==-1)ended=true;
                else if(ended||v<0||v>=c.tap_rows)return RAD_E_INVAL;}}
        if(op!=0||!ix(a,3)[s]){if(ix(a,li)[slot]<(op==0?0:1)||ix(a,li)[slot]>c.context_window)return RAD_E_INVAL;
            if(op!=3&&(ix(a,hi)[slot]<0||ix(a,hi)[slot]>=c.context_window))return RAD_E_INVAL;}
        if(op==3&&a->t[16].data&&(ix(a,16)[s]<0||ix(a,16)[s]>=c.active_vocab))return RAD_E_INVAL;
    }
    return RAD_OK;
}
}

void rad_ref_dloop_forward(const RadDLoopConfig& c,const uint16_t* weights,const float* ctxh,int length,
    const uint16_t* anchor,const uint16_t* const* prefix,int plen,float* y,float* scratch){
    const int H=c.hidden_size,B=c.proposal_length+1,C=c.context_window,D=c.num_kv_heads*c.head_dim;
    const int I=c.intermediate_size,G=H/c.conv_group_size,N=C+B;
    float* p=scratch;auto take=[&](int64_t n){float* v=p;p+=n;return v;};
    float* block=take(B*H);float* z=take(B*H);float* pre=take(B*H);float* q=take(B*H);
    float* att=take(B*H);float* out=take(B*H);float* k=take(N*D);float* v=take(N*D);
    float* dyn=take(4*B*G);float* gate=take(B*I);float* up=take(B*I);
    auto w=[&](int id,int l=0){return weights+rad_dloop_weight_offset(c,id,l);};
    for(int t=0;t<B;++t)for(int h=0;h<H;++h){
        if(t==0||t<=plen){const auto* emb=t?prefix[t-1]:anchor;float sum=0;
            for(int j=0;j<c.target_width;++j)sum+=bf(emb[j])*bf(w(DL_TOKEN_PROJ)[(int64_t)h*c.target_width+j]);
            block[t*H+h]=sum;
        }else block[t*H+h]=bf(w(DL_MASK_TOKEN)[h]);}
    for(int l=0;l<c.num_layers;++l){
        norm(block,w(DL_INPUT_NORM,l),z,B,H,c.eps);
        const float* x=z;
        if(c.conv_kernel_size){linear(z,w(DL_ATTN_KERNEL,l),dyn,B,H,4*G);conv(c,z,w(DL_ATTN_BASE,l),dyn,0,pre);x=pre;}
        linear(x,w(DL_Q_PROJ,l),q,B,H,H);
        linear(ctxh,w(DL_K_PROJ,l),k,C,H,D);linear(ctxh,w(DL_V_PROJ,l),v,C,H,D);
        linear(x,w(DL_K_PROJ,l),k+C*D,B,H,D);linear(x,w(DL_V_PROJ,l),v+C*D,B,H,D);
        rope(q,B,c.num_q_heads,c.head_dim,C,c.rope_theta);rope(k,N,c.num_kv_heads,c.head_dim,0,c.rope_theta);
        for(int t=0;t<B;++t)for(int h=0;h<c.num_q_heads;++h){float scores[41];float max=-INFINITY;
            const int kh=h/(c.num_q_heads/c.num_kv_heads);
            for(int j=0;j<N;++j){float score=-INFINITY;
                if(j>=C-length){score=0;for(int d=0;d<c.head_dim;++d)score+=q[t*H+h*c.head_dim+d]*k[j*D+kh*c.head_dim+d];score/=std::sqrt((float)c.head_dim);}
                scores[j]=score;max=std::max(max,score);}
            float sum=0;for(int j=0;j<N;++j){scores[j]=std::exp(scores[j]-max);sum+=scores[j];}
            for(int d=0;d<c.head_dim;++d){float acc=0;for(int j=0;j<N;++j)acc+=(scores[j]/sum)*v[j*D+kh*c.head_dim+d];att[t*H+h*c.head_dim+d]=acc;}}
        linear(att,w(DL_O_PROJ,l),out,B,H,H);
        if(c.conv_kernel_size){conv(c,out,w(DL_ATTN_BASE,l),dyn,1,pre);x=pre;}else x=out;
        for(int j=0;j<B*H;++j)block[j]+=x[j];
        norm(block,w(DL_POST_NORM,l),z,B,H,c.eps);x=z;
        if(c.conv_kernel_size){linear(z,w(DL_MLP_KERNEL,l),dyn,B,H,4*G);conv(c,z,w(DL_MLP_BASE,l),dyn,0,pre);x=pre;}
        linear(x,w(DL_GATE_PROJ,l),gate,B,H,I);linear(x,w(DL_UP_PROJ,l),up,B,H,I);
        for(int j=0;j<B*I;++j)gate[j]=(gate[j]/(1.0f+std::exp(-gate[j])))*up[j];
        linear(gate,w(DL_DOWN_PROJ,l),out,B,I,H);
        if(c.conv_kernel_size){conv(c,out,w(DL_MLP_BASE,l),dyn,1,pre);x=pre;}else x=out;
        for(int j=0;j<B*H;++j)block[j]+=x[j];
    }
    norm(block+H,w(DL_FINAL_NORM),z,B-1,H,c.eps);linear(z,w(DL_OUTPUT_PROJ),y,B-1,H,c.target_width);
}
int rad_ref_dloop_ctx_append(const RadArgs* a,RadStream){
    RadDLoopConfig c{};int rc=rad_dloop_args(a,0,&c);if(rc)return rc;rc=metadata(a,c,0);if(rc)return rc;
    auto* ring=static_cast<uint16_t*>(a->t[4].data);auto* head=io(a,5);auto* len=io(a,6);
    for(int s=0;s<c.n_seq;++s){int slot=ix(a,2)[s];if(ix(a,3)[s])head[slot]=len[slot]=0;
        for(int j=0;j<16&&ix(a,1)[s*16+j]>=0;++j){int row=ix(a,1)[s*16+j];
            auto* dst=ring+((int64_t)slot*c.context_window+head[slot])*c.context_features_dim;
            if(a->t[7].data){for(int plane=0;plane<3;++plane)std::memcpy(dst+plane*c.target_width,
                b(a,plane?plane+6:0)+(int64_t)row*c.target_width,2*c.target_width);}
            else std::memcpy(dst,b(a,0)+(int64_t)row*c.context_features_dim,2*c.context_features_dim);
            head[slot]=(head[slot]+1)%c.context_window;len[slot]=std::min(len[slot]+1,c.context_window);}}
    return RAD_OK;
}
int rad_ref_dloop_base(const RadArgs* a,RadStream){
    RadDLoopConfig c{};int rc=rad_dloop_args(a,1,&c);if(rc)return rc;rc=metadata(a,c,1);if(rc)return rc;
    const int C=c.context_window,H=c.hidden_size,F=c.context_features_dim,M=c.proposal_length,W=c.target_width,K=c.pool_size;
    for(int s=0;s<c.n_seq;++s){float* scratch=static_cast<float*>(a->scratch)+s*rad_dloop_scratch_elements(c);
        const int slot=ix(a,3)[s],len=ix(a,2)[slot],head=ix(a,1)[slot];float* ctx=fo(a,7)+(int64_t)s*C*H;
        for(int t=0;t<C;++t){float* row=ctx+t*H;
            if(t<C-len){std::fill(row,row+H,0);continue;}
            int physical=(head-len+t-(C-len)+C)%C;const auto* input=b(a,0)+((int64_t)slot*C+physical)*F;
            const auto* w=b(a,5)+rad_dloop_weight_offset(c,DL_CONTEXT_PROJ);
            for(int h=0;h<H;++h){float sum=0;for(int j=0;j<F;++j)sum+=bf(input[j])*bf(w[(int64_t)h*F+j]);scratch[h]=sum;}
            norm(scratch,b(a,5)+rad_dloop_weight_offset(c,DL_CONTEXT_NORM),row,1,H,c.eps);}
        float* y=fo(a,8)+(int64_t)s*M*W;
        rad_ref_dloop_forward(c,b(a,5),ctx,len,b(a,4)+(int64_t)s*W,nullptr,0,y,scratch);
        for(int t=0;t<M;++t){int* ids=io(a,9)+(s*M+t)*K;float* scores=fo(a,10)+(s*M+t)*K;
            std::fill(ids,ids+K,-1);std::fill(scores,scores+K,-INFINITY);
            for(int j=0;j<c.vocab_size_shard&&c.vocab_lo+j<c.active_vocab;++j){float score=dot(y+t*W,b(a,6)+(int64_t)j*W,W);
                if(!std::isfinite(score))return RAD_E_INVAL;
                insert(score,c.vocab_lo+j,scores,ids,K);}}
    }
    return RAD_OK;
}
int rad_ref_dloop_pool(const RadArgs* a,RadStream){
    RadDLoopConfig c{};int rc=rad_dloop_args(a,2,&c);if(rc)return rc;
    const int M=c.proposal_length,K=c.pool_size,W=c.target_width,R=c.selector_rank,rows=c.n_seq*M*K;
    // Check producer packets before touching any output. Duplicate IDs may occur
    // only as repeated same-score packets (cold-shape tooling also uses these).
    for(int s=0;s<c.n_seq;++s)if(ix(a,2)[s]<0||ix(a,2)[s]>=c.active_vocab)return RAD_E_INVAL;
    for(int i=0;i<c.world_size*rows;++i){int id=ix(a,0)[i];float score=f(a,1)[i];
        if(id==-1){if(score!=-INFINITY)return RAD_E_INVAL;}else if(id<0||id>=c.active_vocab||!std::isfinite(score))return RAD_E_INVAL;}
    for(int row=0;row<c.n_seq*M;++row){int ids[64];float scores[64];std::fill(ids,ids+K,-1);std::fill(scores,scores+K,-INFINITY);
        for(int g=0;g<c.world_size;++g)for(int j=0;j<K;++j){int off=g*rows+row*K+j,id=ix(a,0)[off];if(id<0)continue;
            bool duplicate=false;for(int z=0;z<K;++z)if(ids[z]==id){if(scores[z]!=f(a,1)[off])return RAD_E_INVAL;duplicate=true;}
            if(!duplicate)insert(f(a,1)[off],id,scores,ids,K);}
        if(ids[K-1]<0)return RAD_E_INVAL;
        for(int j=1;j<K;++j){int id=ids[j];float score=scores[j];int z=j;while(z&&ids[z-1]>id){ids[z]=ids[z-1];scores[z]=scores[z-1];--z;}ids[z]=id;scores[z]=score;}
        for(int j=0;j<K;++j){const int index=row*K+j,id=ids[j],local=id-c.vocab_lo;io(a,7)[index]=id;fo(a,8)[index]=scores[j];
            for(int plane=0;plane<4;++plane){int width=plane<2?W:R;auto* dst=static_cast<uint16_t*>(a->t[9+plane].data)+(int64_t)index*width;
                if(local>=0&&local<c.vocab_size_shard)std::memcpy(dst,b(a,3+plane)+(int64_t)local*width,2*width);else std::fill(dst,dst+width,0);}}}
    for(int s=0;s<c.n_seq;++s){int id=ix(a,2)[s]-c.vocab_lo;auto* dst=static_cast<uint16_t*>(a->t[13].data)+(int64_t)s*R;
        if(id>=0&&id<c.vocab_size_shard)std::memcpy(dst,b(a,5)+(int64_t)id*R,2*R);else std::fill(dst,dst+R,0);}
    return RAD_OK;
}
int rad_ref_dloop_dartree(const RadArgs* a,RadStream){
    RadDLoopConfig c{};int rc=rad_dloop_args(a,3,&c);if(rc)return rc;rc=metadata(a,c,3);if(rc)return rc;
    const int M=c.proposal_length,K=c.pool_size,W=c.target_width,R=c.selector_rank,H=c.hidden_size,C=c.context_window;
    for(int s=0;s<c.n_seq;++s)for(int t=0;t<M;++t)for(int j=0;j<K;++j){int off=(s*M+t)*K+j,id=ix(a,6)[off];
        if(id<0||id>=c.active_vocab||(j&&id<=ix(a,6)[off-1])||!std::isfinite(f(a,7)[off]))return RAD_E_INVAL;}
    for(int s=0;s<c.n_seq;++s){
        int tok[121]{},par[121]{},depth[121]{},source[121]{},front[15]{},next[15]{},count=1,nfront=1;float score[121]{};
        par[0]=-1;source[0]=-1;front[0]=0;
        const int width=std::min({c.width,K,c.budget}),capacity=c.pruned?width*M:c.budget;
        float* scratch=static_cast<float*>(a->scratch)+s*rad_dloop_scratch_elements(c);
        float* y=scratch+rad_dloop_scratch_elements(c)-M*W-R;float* rank=y+M*W;
        for(int slot=0;slot<M&&count<=capacity&&nfront;++slot){
            float candidates[960];int total=nfront*K;
            for(int p=0;p<nfront;++p){const int parent=front[p];const float* hidden;
                if(!slot)hidden=f(a,5)+(int64_t)s*M*W;
                else {const uint16_t* prefix[8]{};int node=parent;
                    for(int t=slot-1;t>=0;--t){prefix[t]=b(a,9)+((int64_t)s*M*K+source[node])*W;node=par[node];}
                    rad_ref_dloop_forward(c,b(a,4),f(a,0)+(int64_t)s*C*H,ix(a,1)[ix(a,2)[s]],b(a,3)+(int64_t)s*W,prefix,slot,y,scratch);hidden=y+slot*W;}
                linear(hidden,b(a,4)+rad_dloop_weight_offset(c,DL_SELECTOR_PROJ),rank,1,W,R);
                const uint16_t* pred=parent?b(a,10)+((int64_t)s*M*K+source[parent])*R:b(a,12)+(int64_t)s*R;
                for(int r=0;r<R;++r)rank[r]*=bf(pred[r]);
                float max=-INFINITY;for(int j=0;j<K;++j){int off=(s*M+slot)*K+j;
                    float value=slot?dot(hidden,b(a,8)+(int64_t)off*W,W):f(a,7)[off];
                    value+=dot(rank,b(a,11)+(int64_t)off*R,R);if(!std::isfinite(value))return RAD_E_INVAL;
                    candidates[p*K+j]=value;max=std::max(max,value);}
                float sum=0;for(int j=0;j<K;++j)sum+=std::exp(candidates[p*K+j]-max);
                const float logsum=std::log(sum);
                for(int j=0;j<K;++j)candidates[p*K+j]=score[parent]+((candidates[p*K+j]-max)-logsum);
            }
            int remaining=capacity+1-count,quota=c.pruned?width:(remaining+M-slot-1)/(M-slot);quota=std::min({quota,remaining,total});
            for(int k=0;k<quota;++k){int best=0;for(int i=1;i<total;++i)if(candidates[i]>candidates[best])best=i;
                const int node=count++,parent=front[best/K];tok[node]=ix(a,6)[(s*M+slot)*K+best%K];par[node]=parent;
                depth[node]=slot+1;source[node]=slot*K+best%K;score[node]=candidates[best];candidates[best]=-INFINITY;next[k]=node;}
            nfront=quota;std::copy(next,next+quota,front);
        }
        int order[120],keep[121]{},remap[121];std::fill(remap,remap+121,-1);
        for(int i=1;i<count;++i)order[i-1]=i;
        if(c.pruned&&count-1>c.budget){
            // Stable insertion sort: descending priority, ascending depth, then index.
            auto before=[&](int x,int z){float a=score[x]+c.depth_bonus*depth[x],b=score[z]+c.depth_bonus*depth[z];return a>b||(a==b&&(depth[x]<depth[z]||(depth[x]==depth[z]&&x<z)));};
            for(int j=1;j<count-1;++j){int v=order[j],z=j;while(z&&before(v,order[z-1])){order[z]=order[z-1];--z;}order[z]=v;}
            for(int j=0;j<c.budget;++j)keep[order[j]]=1;
        }else for(int i=1;i<count;++i)keep[i]=1;
        int n=0;remap[0]=-1;std::fill(io(a,13)+s*16,io(a,13)+(s+1)*16,-1);std::fill(io(a,14)+s*16,io(a,14)+(s+1)*16,-1);
        for(int i=1;i<count;++i)if(keep[i]){if(par[i]&&remap[par[i]]<0)return RAD_E_INVAL;remap[i]=n;
            io(a,13)[s*16+n]=tok[i];io(a,14)[s*16+n]=remap[par[i]];++n;}
        io(a,15)[s]=n;
        if(a->t[17].data){int* packed=io(a,17)+s*33;std::fill(packed,packed+33,-1);packed[0]=n+1;packed[1]=ix(a,16)[s];
            for(int j=0;j<n;++j){packed[2+j]=io(a,13)[s*16+j];packed[18+j]=io(a,14)[s*16+j]+1;}}
    }
    return RAD_OK;
}
int rad_ref_dloop_count(){return 4;}
const RadKernelInfo* rad_ref_dloop_at(int i){
    static const RadKernelInfo rows[]={rad_dloop_row(0,RAD_DOMAIN_HOST,rad_ref_dloop_ctx_append),rad_dloop_row(1,RAD_DOMAIN_HOST,rad_ref_dloop_base),rad_dloop_row(2,RAD_DOMAIN_HOST,rad_ref_dloop_pool),rad_dloop_row(3,RAD_DOMAIN_HOST,rad_ref_dloop_dartree)};
    return i>=0&&i<4?rows+i:nullptr;
}
