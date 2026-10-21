#include "dloop_full.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

/* Schemas, validators, shape and scratch hooks for the three full-context ops.
 *
 * The geometry rules are the ring op's (dloop_ops.cpp) with the full-context deltas, RESTATED
 * here rather than shared: rad_dloop_config is private to another translation unit, and the ring
 * validators must keep accepting exactly what they accept today. Two places spelling one rule is
 * the price of not touching the old contract; the deviations are marked FULL below. */
namespace {
#define PI(k) {k,RAD_P_INT,RAD_REQUIRED,0}
#define PF(k) {k,RAD_P_F64,RAD_REQUIRED,0}
/* `tap_rows` is a CAPACITY: the launch narrows it to the tap planes' own extent (rad_dloop_full_args),
 * and the planes are sized by the step. The core refuses a fixed parameter that changes between the
 * real declare and a smaller sizing declare unless the schema says so, and without the role the
 * activation arena has no level below the largest step. */
static const RadParamSpec params[]={
    PI("n_seq"),PI("max_seqs"),{"tap_rows",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},PI("world_size"),
    PI("hidden_size"),PI("intermediate_size"),PI("num_layers"),PI("num_q_heads"),
    PI("num_kv_heads"),PI("head_dim"),PI("context_features_dim"),PI("target_width"),
    PI("proposal_length"),PI("context_window"),PI("active_vocab"),PI("vocab_size"),
    PI("selector_rank"),PI("conv_kernel_size"),PI("conv_group_size"),PI("vocab_lo"),
    PI("vocab_size_shard"),PI("pool_size"),PI("budget"),PI("width"),PI("context_capacity"),
    {"variant",RAD_P_STR,RAD_REQUIRED,0},PF("eps"),PF("rope_theta"),PF("depth_bonus"),
    {"dtype",RAD_P_STR,RAD_REQUIRED,0}};
#undef PI
#undef PF
#define IN(n) {n,RAD_OPD_IN,0}
#define OUT(n) {n,RAD_OPD_OUT,0}
#define IO(n) {n,RAD_OPD_INOUT,0}
/* The backbone blob and the teacher shards are the drafter's own allocations, exactly as in
 * dloop_ops.cpp: IN operands passed at issue, not container weights. */
static const RadOperandSpec append[]={IN("tap0"),IN("tap1"),IN("tap2"),IN("path"),IN("span"),
    IN("slots"),IN("reset"),IN("weights"),IO("cache"),IO("length")};
static const RadOperandSpec base[]={IN("cache"),IN("length"),IN("slots"),IN("anchor_embed"),
    IN("weights"),IN("head_shard"),OUT("y_base"),OUT("local_ids"),OUT("local_logits")};
static const RadOperandSpec tree[]={IN("cache"),IN("length"),IN("slots"),IN("anchor_embed"),
    IN("weights"),IN("y_base"),IN("pool_ids"),IN("unary"),IN("pool_head"),IN("pool_embed"),
    IN("pool_pred"),IN("pool_succ"),IN("anchor_pred"),OUT("tree_tok"),OUT("tree_parent"),
    OUT("n_nodes"),{"anchor",RAD_OPD_IN,1},{"packed_tree",RAD_OPD_OUT,1}};
#undef IN
#undef OUT
#undef IO

static const RadOpSchema schemas[]={
    {"dloop_full_append",params,30,append,10,
     "Project new verified tap rows into every layer's FP32 K/V cache at absolute positions."},
    {"dloop_full_base",params,30,base,9,
     "Bidirectional five-layer drafter backbone over the whole cached prefix; shard top-k."},
    {"dloop_full_dartree",params,30,tree,18,
     "DARTree over the persistent context cache; implicit anchor, at most fifteen draft nodes."}};

/* op-first shape, with an already-resolved config the caller may have narrowed (tap_rows). */
int full_shape(int op,const RadParam* p,int np,int operand,RadOpdDesc* out,
               const RadDLoopFullConfig* actual=nullptr) {
    if(!out||op<0||op>=RAD_DLOOP_FULL_OPS||operand<0||operand>=schemas[op].n_operands)return RAD_E_INVAL;
    RadDLoopFullConfig c{};if(actual)c=*actual;else {int rc=rad_dloop_full_config(p,np,&c);if(rc)return rc;}
    const int S=c.model.n_seq,Q=c.model.max_seqs,C=c.context_capacity;
    const int M=c.model.proposal_length,K=c.model.pool_size,W=c.model.target_width,R=c.model.selector_rank;
    const int Vs=c.model.vocab_size_shard,L=c.model.num_layers;
    const int64_t D=(int64_t)c.model.num_kv_heads*c.model.head_dim;
    RadOpdDesc d[18]{};int at=0;
    auto add=[&](uint32_t dt,std::initializer_list<int64_t> dims) -> RadOpdDesc& {
        auto& v=d[at++];v.dtype=dt;v.idx_const=-1;for(auto n:dims)v.shape[v.rank++]=n;return v;};
    auto idx=[&](int n,int max,int fixed=-1,uint32_t flags=0) {
        auto& v=add(RAD_I32,{n});v.fill=RAD_FILL_INDEX;v.idx_max=max;v.idx_const=fixed;v.flags=flags;};
    auto cache=[&]() {add(RAD_F32,{Q,L,C,2*D});};
    auto cache_len=[&]() {idx(Q,0,0);};
    /* A slot is a scatter destination only where a launch writes by slot: the append's uniqueness
     * is a contract this op enforces, and base/tree merely read it. */
    auto slots=[&]() {idx(S,Q,-1,op==RAD_DLOOP_FULL_APPEND?(uint32_t)RAD_OPD_F_IDX_UNIQUE:0u);};
    auto weights=[&]() {add(RAD_BF16,{rad_dloop_weight_elements(c.model)});};
    auto rows=[&](int width) {add(RAD_BF16,{S,M,K,width});};
    if(op==RAD_DLOOP_FULL_APPEND) {
        add(RAD_BF16,{c.model.tap_rows,W});add(RAD_BF16,{c.model.tap_rows,W});add(RAD_BF16,{c.model.tap_rows,W});
        auto& path=add(RAD_I32,{S,16});path.fill=RAD_FILL_INDEX;path.idx_const=0;
        /* span is [start,count] PER SEQUENCE: a rank-2 index pair, not a flat vector. */
        auto& span=add(RAD_I32,{S,2});span.fill=RAD_FILL_INDEX;span.idx_const=0;
        slots();idx(S,0,1);  /* reset */
        weights();cache();cache_len();
    } else if(op==RAD_DLOOP_FULL_BASE) {
        cache();idx(Q,0,0);slots();add(RAD_BF16,{S,W});weights();add(RAD_BF16,{Vs,W});
        add(RAD_F32,{S,M,W});
        auto& ids=add(RAD_I32,{S,M,K});ids.fill=RAD_FILL_INDEX;ids.idx_max=c.model.active_vocab;
        add(RAD_F32,{S,M,K});
    } else {
        cache();idx(Q,0,0);slots();add(RAD_BF16,{S,W});weights();add(RAD_F32,{S,M,W});
        auto& ids=add(RAD_I32,{S,M,K});ids.fill=RAD_FILL_INDEX;ids.idx_const=0;
        add(RAD_F32,{S,M,K});rows(W);rows(W);rows(R);rows(R);add(RAD_BF16,{S,R});
        add(RAD_I32,{S,16});add(RAD_I32,{S,16});add(RAD_I32,{S});
        idx(S,c.model.active_vocab);d[at-1].flags=RAD_OPD_F_ABSENT;
        add(RAD_I32,{S,33}).flags=RAD_OPD_F_ABSENT;
    }
    if(at!=schemas[op].n_operands)return RAD_E_SHAPE;
    *out=d[operand];return RAD_OK;
}
}  /* namespace */

RadDLoopFullScratch rad_dloop_full_scratch_layout(const RadDLoopFullConfig& c,int op) {
    const int64_t H=c.model.hidden_size,I=c.model.intermediate_size,B=c.model.proposal_length+1;
    const int64_t D=(int64_t)c.model.num_kv_heads*c.model.head_dim;
    const int64_t G=c.model.conv_group_size?H/c.model.conv_group_size:0;
    const int64_t F=c.model.context_features_dim,W=c.model.target_width,M=c.model.proposal_length;
    const int64_t K=c.model.pool_size,R=c.model.selector_rank;
    RadDLoopFullScratch s{};int64_t p=0;
    auto take=[&](int64_t n) {const int64_t a=p;p+=n;return a;};
    if(op==RAD_DLOOP_FULL_APPEND) {
        /* The host projects one row at a time and reuses this F+H buffer.
         * Device rows publish their independently chunk-bounded staging through their own hook. */
        s.stage=p;take(F+H);
        s.total=p;return s;
    }
    s.block=take(B*H);s.z=take(B*H);s.pre=take(B*H);s.q=take(B*H);s.att=take(B*H);s.out=take(B*H);
    s.kblk=take(B*D);s.vblk=take(B*D);s.dyn=take(4*B*G);s.gate=take(B*I);s.up=take(B*I);
    /* One attention row: C cached positions plus the block. The cache itself is never copied. */
    s.score=take((int64_t)c.context_capacity+B);s.y=take(M*W);
    if(op==RAD_DLOOP_FULL_DARTREE) {
        const int64_t cap=(int64_t)rad_dloop_full_tree_capacity(c)+2;
        const int64_t frontier=(int64_t)rad_dloop_full_frontier_max(c);
        s.rank=take(R);s.cand=take(frontier*K);s.node_score=take(cap);
        s.nodes=take(4*cap);s.idx=take(3*cap);s.front=take(2*frontier);
    }
    s.total=p;return s;
}
int64_t rad_dloop_full_scratch_elements(const RadDLoopFullConfig& c,int op) {
    if(op<0||op>=RAD_DLOOP_FULL_OPS)return 0;
    return rad_dloop_full_scratch_layout(c,op).total;
}
/* The same node bound the ring tree uses: width capped by the pool and the budget, times the
 * depths, or the budget outright when nothing is pruned. */
int rad_dloop_full_tree_capacity(const RadDLoopFullConfig& c) {
    if(!c.model.pruned)return c.model.budget;
    const int width=std::min(std::min(c.model.width,c.model.pool_size),c.model.budget);
    return width*c.model.proposal_length;
}
/* Candidates alive at one depth: a pruned level admits at most `width` successors of each
 * frontier node, an unpruned one at most the budget. */
int rad_dloop_full_frontier_max(const RadDLoopFullConfig& c) {
    return std::max(c.model.budget,c.model.pruned?std::min(c.model.width,c.model.pool_size):0);
}

int rad_dloop_full_config(const RadParam* p,int n,RadDLoopFullConfig* out) {
    if(!out||!p||n<0)return RAD_E_INVAL;
    RadDLoopFullConfig c{};RadDLoopConfig& m=c.model;
#define GET(k) {int64_t v=rad_param_getdim(p,n,#k,-1);if(v<0||v>1048576)return RAD_E_UNSUPPORTED;m.k=(int)v;}
    GET(n_seq);GET(max_seqs);GET(tap_rows);GET(world_size);GET(hidden_size);GET(intermediate_size);
    GET(num_layers);GET(num_q_heads);GET(num_kv_heads);GET(head_dim);GET(context_features_dim);GET(target_width);
    GET(proposal_length);GET(context_window);GET(active_vocab);GET(vocab_size);GET(selector_rank);
    GET(conv_kernel_size);GET(conv_group_size);GET(vocab_lo);GET(vocab_size_shard);GET(pool_size);GET(budget);GET(width);
#undef GET
    {int64_t v=rad_param_getdim(p,n,"context_capacity",-1);if(v<1||v>1048576)return RAD_E_UNSUPPORTED;c.context_capacity=(int)v;}
    if(m.n_seq<1||m.max_seqs<m.n_seq||m.tap_rows<1||m.world_size<1||m.world_size>64||
       m.hidden_size<1||m.intermediate_size<1||m.num_layers<1||m.num_layers>64||m.head_dim<2||m.head_dim%2||
       m.num_q_heads<1||m.num_kv_heads<1||m.num_q_heads%m.num_kv_heads||
       (int64_t)m.num_q_heads*m.head_dim!=m.hidden_size||m.context_features_dim<1||m.target_width<1||
       m.proposal_length<1||m.proposal_length>8||
       m.active_vocab<1||m.active_vocab>m.vocab_size||m.selector_rank<1||
       (m.conv_kernel_size!=0&&m.conv_kernel_size!=2)||m.conv_group_size<1||m.hidden_size%m.conv_group_size||
       m.vocab_size_shard<1||m.vocab_lo>m.vocab_size-m.vocab_size_shard||
       m.pool_size<1||m.pool_size>64||m.pool_size>m.active_vocab||m.budget<1||m.budget>15||m.width<1)
        return RAD_E_UNSUPPORTED;
    /* FULL: no window, a required capacity, and three concatenated tap planes. */
    if(m.context_window!=0)return RAD_E_UNSUPPORTED;
    if(m.context_features_dim!=3*m.target_width)return RAD_E_UNSUPPORTED;
    if((int64_t)m.n_seq*m.world_size*m.proposal_length*m.pool_size>INT32_MAX)return RAD_E_SHAPE;
    const char* variant=rad_param_gets(p,n,"variant","");
    if(std::strcmp(variant,"fixed")&&std::strcmp(variant,"pruned"))return RAD_E_UNSUPPORTED;
    m.pruned=!std::strcmp(variant,"pruned");
    if(std::strcmp(rad_param_gets(p,n,"dtype",""),"bf16"))return RAD_E_DTYPE;
    m.eps=(float)rad_param_getf(p,n,"eps",0);m.rope_theta=(float)rad_param_getf(p,n,"rope_theta",0);
    m.depth_bonus=(float)rad_param_getf(p,n,"depth_bonus",1);
    if(!std::isfinite(m.eps)||m.eps<=0||!std::isfinite(m.rope_theta)||m.rope_theta<=0||
       !std::isfinite(m.depth_bonus)||m.depth_bonus>0)return RAD_E_INVAL;
    if(rad_dloop_weight_elements(m)>INT64_MAX/16)return RAD_E_SHAPE;
    for(int op=0;op<RAD_DLOOP_FULL_OPS;++op)
        if(rad_dloop_full_scratch_elements(c,op)>INT64_MAX/(4LL*m.n_seq))return RAD_E_SHAPE;
    /* The cache must stay addressable: max_seqs*L*C*2D elements and its byte extent. */
    const int64_t kv=(int64_t)m.num_layers*c.context_capacity*2*m.num_kv_heads*m.head_dim;
    if(kv>INT64_MAX/(4LL*m.max_seqs))return RAD_E_SHAPE;
    *out=c;return RAD_OK;
}

int rad_dloop_full_args(const RadArgs* a,int op,RadDLoopFullConfig* out) {
    if(!a||!a->t||!out||op<0||op>=RAD_DLOOP_FULL_OPS||a->n_t!=schemas[op].n_operands)return RAD_E_INVAL;
    RadDLoopFullConfig c{};int rc=rad_dloop_full_config(a->p,a->n_p,&c);if(rc)return rc;
    const int seq_operand=op==RAD_DLOOP_FULL_APPEND?5:2;
    const auto& seq=a->t[seq_operand];
    if(seq.rank<1||seq.shape[0]<1||seq.shape[0]>c.model.n_seq)return RAD_E_SHAPE;
    c.model.n_seq=(int)seq.shape[0];
    /* The cache operand fixes capacity and layer count for this call. It is operand 0 for base and
     * tree, and operand 8 for append (whose first three operands are the tap planes). */
    const int cache_operand=op==RAD_DLOOP_FULL_APPEND?8:0;
    const int64_t D=(int64_t)c.model.num_kv_heads*c.model.head_dim;
    {
        const auto& t=a->t[cache_operand];
        if(t.rank!=4||t.dtype!=RAD_F32||t.shape[0]!=c.model.max_seqs||t.shape[1]!=c.model.num_layers||
           t.shape[2]!=c.context_capacity||t.shape[3]!=2*D)return RAD_E_SHAPE;
    }
    if(op==RAD_DLOOP_FULL_APPEND) {
        if(a->t[0].rank!=2||a->t[0].shape[0]<0||a->t[0].shape[0]>c.model.tap_rows)return RAD_E_SHAPE;
        c.model.tap_rows=(int)a->t[0].shape[0];
    }
    for(int i=0;i<a->n_t;++i) {
        RadOpdDesc d{};rc=full_shape(op,a->p,a->n_p,i,&d,&c);if(rc)return rc;
        if(op==RAD_DLOOP_FULL_DARTREE&&i>=16&&!a->t[i].data)continue;
        const auto& t=a->t[i];
        if(!t.data||t.dtype!=d.dtype||t.rank!=d.rank)return RAD_E_SHAPE;
        int64_t stride=1;
        for(int j=(int)t.rank-1;j>=0;--j) {
            if(t.shape[j]!=d.shape[j]||t.stride[j]!=stride||t.shape[j]>INT64_MAX/stride)return RAD_E_SHAPE;
            stride*=t.shape[j];
        }
        if(stride>INT64_MAX/(t.dtype==RAD_BF16?2:4)||(uint64_t)stride>SIZE_MAX/(t.dtype==RAD_BF16?2u:4u))
            return RAD_E_SHAPE;
    }
    if(op==RAD_DLOOP_FULL_DARTREE&&bool(a->t[16].data)!=bool(a->t[17].data))return RAD_E_INVAL;
    const int64_t per_seq=rad_dloop_full_scratch_elements(c,op);
    if(!a->scratch||per_seq<=0||a->scratch_bytes<4LL*c.model.n_seq*per_seq)return RAD_E_SCRATCH;
    /* Mutable operands must be disjoint from every other operand. */
    for(int i=0;i<a->n_t;++i)
        if(a->t[i].data&&(schemas[op].operands[i].role==RAD_OPD_OUT||schemas[op].operands[i].role==RAD_OPD_INOUT))
            for(int j=0;j<a->n_t;++j) if(i!=j&&a->t[j].data) {
                auto bytes=[](const RadTensor& t) {int64_t n=1;for(unsigned k=0;k<t.rank;++k)n*=t.shape[k];return n*(t.dtype==RAD_BF16?2:4);};
                uintptr_t x=(uintptr_t)a->t[i].data,y=(uintptr_t)a->t[j].data;
                if((x<=y&&y-x<(uint64_t)bytes(a->t[i]))||(y<=x&&x-y<(uint64_t)bytes(a->t[j])))return RAD_E_INVAL;
            }
    *out=c;return RAD_OK;
}

int rad_dloop_full_schema_count(void){return RAD_DLOOP_FULL_OPS;}
const RadOpSchema* rad_dloop_full_schema_at(int i) {
    return i>=0&&i<RAD_DLOOP_FULL_OPS?schemas+i:nullptr;
}
int rad_dloop_full_shape(int op,const RadParam* p,int np,int operand,RadOpdDesc* out) {
    return full_shape(op,p,np,operand,out);
}
int rad_dloop_full_append_shape(const RadParam* p,int n,int i,RadOpdDesc* d) {
    return full_shape(RAD_DLOOP_FULL_APPEND,p,n,i,d);
}
int rad_dloop_full_base_shape(const RadParam* p,int n,int i,RadOpdDesc* d) {
    return full_shape(RAD_DLOOP_FULL_BASE,p,n,i,d);
}
int rad_dloop_full_dartree_shape(const RadParam* p,int n,int i,RadOpdDesc* d) {
    return full_shape(RAD_DLOOP_FULL_DARTREE,p,n,i,d);
}
int64_t rad_dloop_full_scratch(const RadArgs* a) {
    RadDLoopFullConfig c{};
    if(!a||rad_dloop_full_config(a->p,a->n_p,&c))return 0;
    int64_t worst=0;
    for(int op=0;op<RAD_DLOOP_FULL_OPS;++op)
        worst=std::max(worst,rad_dloop_full_scratch_elements(c,op));
    return 4LL*c.model.n_seq*worst;
}
namespace {
int64_t scratch_for(const RadArgs* a,int op) {
    RadDLoopFullConfig c{};
    if(!a||op<0||op>=RAD_DLOOP_FULL_OPS||rad_dloop_full_config(a->p,a->n_p,&c))return 0;
    return 4LL*c.model.n_seq*rad_dloop_full_scratch_elements(c,op);
}
}
int64_t rad_dloop_full_append_scratch(const RadArgs* a) {return scratch_for(a,RAD_DLOOP_FULL_APPEND);}
int64_t rad_dloop_full_base_scratch(const RadArgs* a) {return scratch_for(a,RAD_DLOOP_FULL_BASE);}
int64_t rad_dloop_full_tree_scratch(const RadArgs* a) {return scratch_for(a,RAD_DLOOP_FULL_DARTREE);}

#ifdef RAD_DLOOP_FULL_DEVICE
/* The device arena. RadScratchFn carries no op, and the device arena differs per op (append stages
 * whole tap rows; base and tree stage the head's candidate keys), so bind the op here instead of
 * asking the device TU for three symbols. */
namespace {
int64_t dev_scratch(const RadArgs* a,int op) {
    RadDLoopFullConfig c{};
    if(!a||rad_dloop_full_config(a->p,a->n_p,&c))return 0;
    return rad_mine_dloop_full_scratch(a,op);
}
int64_t dev_append_scratch(const RadArgs* a) {return dev_scratch(a,RAD_DLOOP_FULL_APPEND);}
int64_t dev_base_scratch(const RadArgs* a) {return dev_scratch(a,RAD_DLOOP_FULL_BASE);}
int64_t dev_tree_scratch(const RadArgs* a) {return dev_scratch(a,RAD_DLOOP_FULL_DARTREE);}
}
#endif

RadKernelInfo rad_dloop_full_row(int op,int domain,RadLaunchFn launch) {
    RadKernelInfo r{};if(op<0||op>=RAD_DLOOP_FULL_OPS)return r;
    static const char* host[]={"ref_dloop_full_append","ref_dloop_full_base","ref_dloop_full_dartree"};
    static const char* device[]={"mine_dloop_full_append","mine_dloop_full_base","mine_dloop_full_dartree"};
    static const RadShapeFn shapes[]={rad_dloop_full_append_shape,rad_dloop_full_base_shape,
        rad_dloop_full_dartree_shape};
    static const RadScratchFn scratches[]={rad_dloop_full_append_scratch,rad_dloop_full_base_scratch,
        rad_dloop_full_tree_scratch};
    r.name=domain==RAD_DOMAIN_HOST?host[op]:device[op];r.op=schemas[op].op;r.family="dloop";
    r.computes=schemas[op].doc;
    r.shape="full-context; context_window=0, C=context_capacity, L<=64, M<=8, K<=64, budget<=15";
    r.dtypes="bf16 taps/weights, f32 cache/activations, i32 indices";
    r.domain=domain;r.priority=domain==RAD_DOMAIN_HOST?0:100;r.launch=launch;r.opd_shape=shapes[op];
    r.scratch=scratches[op];
    return r;
}

/* ============================================================ device rows (gfx1201)
 *
 * Compiled only into the device plugin, which is where the HIP entry points live. The host
 * build compiles this file too (mine_ref_ops carries the schemas), so every name below is
 * inside a guard the host never opens. */
#ifdef RAD_DLOOP_FULL_DEVICE
#define COMMON RAD_CEQ("world_size",2),RAD_CEQ("context_window",0), \
    RAD_CEQ("conv_kernel_size",2),RAD_CEQ("conv_group_size",16), \
    RAD_CEQ("pool_size",64),RAD_CLE("budget",15),RAD_CLE("width",12),RAD_CIN("dtype","bf16")
const RadConstraint production_full[]={COMMON,
    RAD_CEQ("num_layers",5),RAD_CEQ("hidden_size",768),RAD_CEQ("intermediate_size",2304),
    RAD_CEQ("num_q_heads",12),RAD_CEQ("num_kv_heads",4),RAD_CEQ("head_dim",64),
    RAD_CEQ("context_features_dim",7680),RAD_CEQ("target_width",2560),
    RAD_CEQ("selector_rank",256),RAD_CEQ("vocab_size",248320),
    RAD_CEQ("active_vocab",248077),RAD_CEQ("vocab_size_shard",124160)};
/* Five layers at a fixture width: the production layer count is the contract, the rest is small
 * enough for a CPU reference to follow. F = 3*32 = 96 taps per position. */
const RadConstraint tiny_full[]={COMMON,
    RAD_CEQ("num_layers",5),RAD_CEQ("hidden_size",64),RAD_CEQ("intermediate_size",128),
    RAD_CEQ("num_q_heads",4),RAD_CEQ("num_kv_heads",2),RAD_CEQ("head_dim",16),
    RAD_CEQ("context_features_dim",96),RAD_CEQ("target_width",32),
    RAD_CEQ("selector_rank",8),RAD_CEQ("vocab_size",512),
    RAD_CEQ("active_vocab",509),RAD_CEQ("vocab_size_shard",256)};
#undef COMMON
namespace {
struct FullRows {
    RadKernelInfo row[2*RAD_DLOOP_FULL_OPS];
    FullRows() {
        static const RadScratchFn scratch[]={dev_append_scratch,dev_base_scratch,dev_tree_scratch};
        const RadLaunchFn launch[2*RAD_DLOOP_FULL_OPS]={
            rad_mine_dloop_full_append_h768_l5_tp2,rad_mine_dloop_full_base_h768_l5_tp2,
            rad_mine_dloop_full_tree_h768_l5_tp2,rad_mine_dloop_full_append_h64_l5_tp2,
            rad_mine_dloop_full_base_h64_l5_tp2,rad_mine_dloop_full_tree_h64_l5_tp2};
        /* The row name IS the entry point, and it is unique within the plugin: two rows per op
         * (production and fixture) would otherwise collide in the loader's table. */
        const char* names[2*RAD_DLOOP_FULL_OPS]={
            "mine_dloop_full_append_h768_l5_tp2","mine_dloop_full_base_h768_l5_tp2",
            "mine_dloop_full_tree_h768_l5_tp2","mine_dloop_full_append_h64_l5_tp2",
            "mine_dloop_full_base_h64_l5_tp2","mine_dloop_full_tree_h64_l5_tp2"};
        /* Production first, then the fixture geometry: the ring rows put production first so a
         * selector meeting both constraints resolves the production kernel. */
        for(int i=0;i<2*RAD_DLOOP_FULL_OPS;++i) {
            row[i]=rad_dloop_full_row(i%RAD_DLOOP_FULL_OPS,RAD_DOMAIN_DEVICE,launch[i]);
            // Bind the device hook here: the shared factory can be interposed by a host executable.
            row[i].scratch=scratch[i%RAD_DLOOP_FULL_OPS];
            row[i].name=names[i];
            row[i].constraints=i<RAD_DLOOP_FULL_OPS?production_full:tiny_full;
            row[i].n_constraints=int(sizeof(production_full)/sizeof(production_full[0]));
            row[i].shape=i<RAD_DLOOP_FULL_OPS?
                "H768 FF2304 L5 Q12 KV4 Dh64 F7680 W2560 M8 R256 K64 TP2 full-context":
                "fixture H64 FF128 L5 Q4 KV2 Dh16 F96 W32 M8 R8 K64 TP2 full-context";
        }
    }
};
}  /* namespace */
extern "C" int rad_mine_dloop_full_count() {return 2*RAD_DLOOP_FULL_OPS;}
extern "C" const RadKernelInfo* rad_mine_dloop_full_at(int i) {
    static const FullRows rows;
    return i>=0&&i<2*RAD_DLOOP_FULL_OPS?rows.row+i:nullptr;
}
#endif
