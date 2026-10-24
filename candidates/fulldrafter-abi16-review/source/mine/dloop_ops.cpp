#include "dloop_ops.h"
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace {
#define PI(k) {k,RAD_P_INT,RAD_REQUIRED,0}
#define PF(k) {k,RAD_P_F64,RAD_REQUIRED,0}
static const RadParamSpec params[]={
    PI("n_seq"),PI("max_seqs"),PI("tap_rows"),PI("world_size"),
    PI("hidden_size"),PI("intermediate_size"),PI("num_layers"),PI("num_q_heads"),
    PI("num_kv_heads"),PI("head_dim"),PI("context_features_dim"),PI("target_width"),
    PI("proposal_length"),PI("context_window"),PI("active_vocab"),PI("vocab_size"),
    PI("selector_rank"),PI("conv_kernel_size"),PI("conv_group_size"),PI("vocab_lo"),
    PI("vocab_size_shard"),PI("pool_size"),PI("budget"),PI("width"),
    {"variant",RAD_P_STR,RAD_REQUIRED,0},PF("eps"),PF("rope_theta"),PF("depth_bonus"),
    {"dtype",RAD_P_STR,RAD_REQUIRED,0}};
#undef PI
#undef PF
#define IN(n) {n,RAD_OPD_IN,0}
#define OUT(n) {n,RAD_OPD_OUT,0}
#define IO(n) {n,RAD_OPD_INOUT,0}
/* The backbone blob and the teacher/codebook shards are the drafter's own device allocations, loaded
 * from the sidecar and RADIANCE_TEACHER_MATRICES, not container weights the builder names at declare.
 * They are IN operands passed at issue, as fulldrafter_two_stage passes its parameter block. */
static const RadOperandSpec append[]={IN("taps"),IN("path"),IN("slots"),IN("reset"),IO("ring"),IO("head"),IO("length"),{"tap1",RAD_OPD_IN,1},{"tap2",RAD_OPD_IN,1}};
static const RadOperandSpec base[]={IN("ring"),IN("head"),IN("length"),IN("slots"),IN("anchor_embed"),IN("weights"),IN("head_shard"),OUT("ctxh"),OUT("y_base"),OUT("local_ids"),OUT("local_logits")};
static const RadOperandSpec pool[]={IN("gathered_ids"),IN("gathered_logits"),IN("anchor"),IN("head_shard"),IN("embed_shard"),IN("pred_shard"),IN("succ_shard"),OUT("pool_ids"),OUT("unary"),OUT("pool_head"),OUT("pool_embed"),OUT("pool_pred"),OUT("pool_succ"),OUT("anchor_pred")};
static const RadOperandSpec tree[]={IN("ctxh"),IN("length"),IN("slots"),IN("anchor_embed"),IN("weights"),IN("y_base"),IN("pool_ids"),IN("unary"),IN("pool_head"),IN("pool_embed"),IN("pool_pred"),IN("pool_succ"),IN("anchor_pred"),OUT("tree_tok"),OUT("tree_parent"),OUT("n_nodes"),{"anchor",RAD_OPD_IN,1},{"packed_tree",RAD_OPD_OUT,1}};
#undef IN
#undef OUT
#undef IO

static const RadOpSchema schemas[]={
    {"dloop_ctx_append",params,29,append,9,"Append accepted absolute tap rows to a resettable BF16 context ring."},
    {"dloop_base",params,29,base,11,"Bidirectional DLoop backbone and deterministic active-vocabulary shard top-k."},
    {"dloop_pool",params,29,pool,14,"Merge token-sorted pools and gather zero-masked rank-local teacher/codebook rows."},
    {"dloop_dartree",params,29,tree,18,"Full-prefix DARTree fixed/pruned search; implicit anchor, at most fifteen draft nodes."}};
int shape(int op,const RadParam* p,int np,int operand,RadOpdDesc* out,const RadDLoopConfig* actual=nullptr) {
    if (!out || op<0 || op>=4 || operand<0 || operand>=schemas[op].n_operands) return RAD_E_INVAL;
    RadDLoopConfig c{};if(actual)c=*actual;else {int rc=rad_dloop_config(p,np,&c);if(rc)return rc;}
    const int S=c.n_seq,Q=c.max_seqs,C=c.context_window,H=c.hidden_size,F=c.context_features_dim;
    const int M=c.proposal_length,K=c.pool_size,W=c.target_width,R=c.selector_rank,Vs=c.vocab_size_shard;
    RadOpdDesc d[18]{};int at=0;
    auto add=[&](uint32_t dt,std::initializer_list<int64_t> dims) -> RadOpdDesc& {
        auto& v=d[at++];v.dtype=dt;v.idx_const=-1;for(auto n:dims)v.shape[v.rank++]=n;return v;};
    auto idx=[&](int n,int max,int fixed=-1,uint32_t flags=0) {
        auto& v=add(RAD_I32,{n});v.fill=RAD_FILL_INDEX;v.idx_max=max;v.idx_const=fixed;v.flags=flags;};
    auto slots=[&]() {idx(S,Q,-1,op==0 ? RAD_OPD_F_IDX_UNIQUE : 0);};
    auto ring=[&]() {add(RAD_BF16,{Q,C,F});};
    auto weights=[&]() {add(RAD_BF16,{rad_dloop_weight_elements(c)});};
    auto rows=[&](int width) {add(RAD_BF16,{S,M,K,width});};
    if(op==0) {
        add(RAD_BF16,{c.tap_rows,F});auto& path=add(RAD_I32,{S,16});path.fill=RAD_FILL_INDEX;path.idx_const=0;
        slots();idx(S,0,1);ring();idx(Q,0,0);idx(Q,0,0);
        add(RAD_BF16,{c.tap_rows,W}).flags=RAD_OPD_F_ABSENT;add(RAD_BF16,{c.tap_rows,W}).flags=RAD_OPD_F_ABSENT;
    } else if(op==1) {
        ring();idx(Q,0,0);idx(Q,0,C);slots();add(RAD_BF16,{S,W});weights();add(RAD_BF16,{Vs,W});
        add(RAD_F32,{S,C,H});add(RAD_F32,{S,M,W});
        auto& ids=add(RAD_I32,{S,M,K});ids.fill=RAD_FILL_INDEX;ids.idx_max=c.active_vocab;
        add(RAD_F32,{S,M,K});
    } else if(op==2) {
        auto& ids=add(RAD_I32,{c.world_size,S,M,K});ids.fill=RAD_FILL_INDEX;ids.idx_const=0;
        add(RAD_F32,{c.world_size,S,M,K});idx(S,c.active_vocab);
        add(RAD_BF16,{Vs,W});add(RAD_BF16,{Vs,W});add(RAD_BF16,{Vs,R});add(RAD_BF16,{Vs,R});
        add(RAD_I32,{S,M,K});add(RAD_F32,{S,M,K});rows(W);rows(W);rows(R);rows(R);add(RAD_BF16,{S,R});
    } else {
        add(RAD_F32,{S,C,H});idx(Q,0,C);slots();add(RAD_BF16,{S,W});weights();
        add(RAD_F32,{S,M,W});auto& ids=add(RAD_I32,{S,M,K});ids.fill=RAD_FILL_INDEX;ids.idx_const=0;
        add(RAD_F32,{S,M,K});rows(W);rows(W);rows(R);rows(R);add(RAD_BF16,{S,R});
        add(RAD_I32,{S,16});add(RAD_I32,{S,16});add(RAD_I32,{S});
        idx(S,c.active_vocab);d[at-1].flags=RAD_OPD_F_ABSENT;add(RAD_I32,{S,33}).flags=RAD_OPD_F_ABSENT;
    }
    if(at!=schemas[op].n_operands)return RAD_E_SHAPE;
    *out=d[operand];return RAD_OK;
}
}
int rad_dloop_config(const RadParam* p,int n,RadDLoopConfig* out) {
    if(!out||!p||n<0)return RAD_E_INVAL;
    RadDLoopConfig c{};
#define GET(k) {int64_t v=rad_param_getdim(p,n,#k,-1);if(v<0||v>1048576)return RAD_E_UNSUPPORTED;c.k=(int)v;}
    GET(n_seq);GET(max_seqs);GET(tap_rows);GET(world_size);GET(hidden_size);GET(intermediate_size);
    GET(num_layers);GET(num_q_heads);GET(num_kv_heads);GET(head_dim);GET(context_features_dim);GET(target_width);
    GET(proposal_length);GET(context_window);GET(active_vocab);GET(vocab_size);GET(selector_rank);
    GET(conv_kernel_size);GET(conv_group_size);GET(vocab_lo);GET(vocab_size_shard);GET(pool_size);GET(budget);GET(width);
#undef GET
    if(c.n_seq<1||c.max_seqs<c.n_seq||c.tap_rows<1||c.world_size<1||c.world_size>64||
       c.hidden_size<1||c.intermediate_size<1||c.num_layers<1||c.num_layers>64||c.head_dim<2||c.head_dim%2||
       c.num_q_heads<1||c.num_kv_heads<1||c.num_q_heads%c.num_kv_heads||
       (int64_t)c.num_q_heads*c.head_dim!=c.hidden_size||c.context_features_dim<1||c.target_width<1||
       c.proposal_length<1||c.proposal_length>8||c.context_window<1||c.context_window>32||
       c.active_vocab<1||c.active_vocab>c.vocab_size||c.selector_rank<1||
       (c.conv_kernel_size!=0&&c.conv_kernel_size!=2)||c.conv_group_size<1||c.hidden_size%c.conv_group_size||
       c.vocab_size_shard<1||c.vocab_lo>c.vocab_size-c.vocab_size_shard||
       c.pool_size<1||c.pool_size>64||c.pool_size>c.active_vocab||c.budget<1||c.budget>15||c.width<1)return RAD_E_UNSUPPORTED;
    if((int64_t)c.n_seq*c.world_size*c.proposal_length*c.pool_size>INT32_MAX)return RAD_E_SHAPE;
    const char* variant=rad_param_gets(p,n,"variant","");
    if(std::strcmp(variant,"fixed")&&std::strcmp(variant,"pruned"))return RAD_E_UNSUPPORTED;
    c.pruned=!std::strcmp(variant,"pruned");
    if(std::strcmp(rad_param_gets(p,n,"dtype",""),"bf16"))return RAD_E_DTYPE;
    c.eps=(float)rad_param_getf(p,n,"eps",0);c.rope_theta=(float)rad_param_getf(p,n,"rope_theta",0);
    c.depth_bonus=(float)rad_param_getf(p,n,"depth_bonus",1);
    if(!std::isfinite(c.eps)||c.eps<=0||!std::isfinite(c.rope_theta)||c.rope_theta<=0||
       !std::isfinite(c.depth_bonus)||c.depth_bonus>0)return RAD_E_INVAL;
    // Bound all packed/scratch products before converting them to byte counts.
    if(rad_dloop_weight_elements(c)>INT64_MAX/16||rad_dloop_scratch_elements(c)>INT64_MAX/(4LL*c.n_seq))return RAD_E_SHAPE;
    *out=c;return RAD_OK;
}
int rad_dloop_args(const RadArgs* a,int op,RadDLoopConfig* out) {
    if(!a||!a->t||!out||op<0||op>=4||a->n_t!=schemas[op].n_operands)return RAD_E_INVAL;
    RadDLoopConfig c{};int rc=rad_dloop_config(a->p,a->n_p,&c);if(rc)return rc;
    const int seq_operand=op==0?1:op==1?4:op==2?2:0;
    const auto& seq=a->t[seq_operand];
    if(seq.rank<1||seq.shape[0]<1||seq.shape[0]>c.n_seq)return RAD_E_SHAPE;
    c.n_seq=(int)seq.shape[0];
    if(op==0){if(a->t[0].rank!=2||a->t[0].shape[0]<1||a->t[0].shape[0]>c.tap_rows)return RAD_E_SHAPE;c.tap_rows=(int)a->t[0].shape[0];}
    for(int i=0;i<a->n_t;++i) {
        RadOpdDesc d{};rc=shape(op,a->p,a->n_p,i,&d,&c);if(rc)return rc;
        if(((op==3&&i>=16)||(op==0&&i>=7))&&!a->t[i].data)continue;
        if(op==0&&i==0&&a->t[7].data)d.shape[1]=c.target_width;
        if(op==2&&i<2&&a->t[i].rank==2){d.rank=2;d.shape[1]=(int64_t)c.n_seq*c.proposal_length*c.pool_size;}
        const auto& t=a->t[i];if(!t.data||t.dtype!=d.dtype||t.rank!=d.rank)return RAD_E_SHAPE;
        int64_t stride=1;
        for(int j=(int)t.rank-1;j>=0;--j){if(t.shape[j]!=d.shape[j]||t.stride[j]!=stride||t.shape[j]>INT64_MAX/stride)return RAD_E_SHAPE;stride*=t.shape[j];}
        if(stride>INT64_MAX/(t.dtype==RAD_BF16?2:4)||(uint64_t)stride>SIZE_MAX/(t.dtype==RAD_BF16?2u:4u))return RAD_E_SHAPE;
    }
    if(op==3&&bool(a->t[16].data)!=bool(a->t[17].data))return RAD_E_INVAL;
    if(op==0&&(bool(a->t[7].data)!=bool(a->t[8].data)||(a->t[7].data&&c.context_features_dim!=3*c.target_width)))return RAD_E_INVAL;
    if((op==1||op==3)&&(!a->scratch||a->scratch_bytes<4LL*c.n_seq*rad_dloop_scratch_elements(c)))return RAD_E_SHAPE;
    // Mutable operands must be disjoint from every other operand.
    for(int i=0;i<a->n_t;++i)if(a->t[i].data&&(schemas[op].operands[i].role==RAD_OPD_OUT||schemas[op].operands[i].role==RAD_OPD_INOUT))
        for(int j=0;j<a->n_t;++j)if(i!=j&&a->t[j].data){
            auto bytes=[](const RadTensor& t){int64_t n=1;for(unsigned k=0;k<t.rank;++k)n*=t.shape[k];return n*(t.dtype==RAD_BF16?2:4);};
            uintptr_t x=(uintptr_t)a->t[i].data,y=(uintptr_t)a->t[j].data;
            if((x<=y&&y-x<(uint64_t)bytes(a->t[i]))||(y<=x&&x-y<(uint64_t)bytes(a->t[j])))return RAD_E_INVAL;
        }
    *out=c;return RAD_OK;
}
int64_t rad_dloop_scratch_elements(const RadDLoopConfig& c) {
    const int64_t B=c.proposal_length+1,H=c.hidden_size,C=c.context_window,I=c.intermediate_size;
    // Backbone: block/norm/pre/post/q/attn/temp + KV + dynamic kernels + FFN;
    // tree: max 8*15 supertree nodes and parent-major candidate scores.
    return 16*B*H+2*(C+B)*c.num_kv_heads*c.head_dim+8*B*(H/c.conv_group_size)+2*B*I+
           B*c.target_width+c.selector_rank+4096;
}
int64_t rad_dloop_scratch(const RadArgs* a) {
    RadDLoopConfig c{};return a&&rad_dloop_config(a->p,a->n_p,&c)==RAD_OK?4LL*c.n_seq*rad_dloop_scratch_elements(c):0;
}
int rad_dloop_schema_count(){return 4;}
const RadOpSchema* rad_dloop_schema_at(int i){return i>=0&&i<4?schemas+i:nullptr;}
int rad_dloop_ctx_append_shape(const RadParam* p,int n,int i,RadOpdDesc* d){return shape(0,p,n,i,d);}
int rad_dloop_base_shape(const RadParam* p,int n,int i,RadOpdDesc* d){return shape(1,p,n,i,d);}
int rad_dloop_pool_shape(const RadParam* p,int n,int i,RadOpdDesc* d){return shape(2,p,n,i,d);}
int rad_dloop_dartree_shape(const RadParam* p,int n,int i,RadOpdDesc* d){return shape(3,p,n,i,d);}
RadKernelInfo rad_dloop_row(int op,int domain,RadLaunchFn launch) {
    RadKernelInfo r{};if(op<0||op>=4)return r;
    static const char* host[]={"ref_dloop_ctx_append","ref_dloop_base","ref_dloop_pool","ref_dloop_dartree"};
    static const char* device[]={"mine_dloop_ctx_append","mine_dloop_base","mine_dloop_pool","mine_dloop_dartree"};
    static const RadShapeFn shapes[]={rad_dloop_ctx_append_shape,rad_dloop_base_shape,rad_dloop_pool_shape,rad_dloop_dartree_shape};
    r.name=domain==RAD_DOMAIN_HOST?host[op]:device[op];r.op=schemas[op].op;r.family="dloop";r.computes=schemas[op].doc;
    r.shape="dense config-driven; M<=8 C<=32 K<=64 draft budget<=15";r.dtypes="bf16 weights/inputs, f32 intermediates, i32 indices";
    r.domain=domain;r.priority=domain==RAD_DOMAIN_HOST?0:100;r.launch=launch;r.opd_shape=shapes[op];
    if(op==1||op==3)r.scratch=rad_dloop_scratch;
    return r;
}
