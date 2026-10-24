#include "attn_paged_tree.h"
#include <climits>
#include <cstring>
#include <initializer_list>

namespace {
#define PI(k) {k, RAD_P_INT, RAD_REQUIRED, 0}
#define PS(k) {k, RAD_P_STR, RAD_REQUIRED, 0}
const RadParamSpec params[] = {
    PI("q_len"), {"max_tokens", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY},
    PI("max_seqs"), PI("max_ctx"), PI("head_dim"), PI("gqa"), PI("n_head"),
    PI("block_size"), PI("causal"), PI("window"), PS("q_dtype"), PS("kv_dtype")};
#undef PI
#undef PS
const RadOperandSpec operands[] = {
    {"q", RAD_OPD_IN, 0}, {"kv", RAD_OPD_IN, 0}, {"block_table", RAD_OPD_IN, 0},
    {"seqused", RAD_OPD_IN, 0}, {"cu_seqlens", RAD_OPD_IN, 0},
    {"spec_parent", RAD_OPD_IN, 0}, {"out", RAD_OPD_OUT, 0}};
const RadOpSchema schema = {"attn_paged_tree", params, 12, operands, RAD_ATP_OPERANDS,
    "Full-context paged attention over a prefix and a <=65-node ordered tree per sequence. "
    "Only self and ancestors are visible in the tree suffix. Causal sequences encode chains. "
    "FP32 dot/softmax/PV and split partials, BF16 output; invalid metadata zeroes the sequence. "
    "Fixed scale=1/16; no KV descales. Reordered, not bit-exact to F16/PF16 decode."};
struct Geometry { int64_t q, tokens, seqs, ctx; };
int geometry(const RadParam* p, int n, Geometry& g) {
    g = {rad_param_getdim(p,n,"q_len",0), rad_param_getdim(p,n,"max_tokens",0),
         rad_param_getdim(p,n,"max_seqs",0), rad_param_getdim(p,n,"max_ctx",0)};
    if(g.q<1 || g.q>65 || g.tokens<1 || g.tokens>INT32_MAX/12 ||
       g.seqs<1 || g.seqs>65535 || g.ctx<1 || g.ctx>INT32_MAX-3)
        return RAD_E_UNSUPPORTED;
    if(rad_param_getdim(p,n,"head_dim",0)!=256 || rad_param_getdim(p,n,"gqa",0)!=12 ||
       rad_param_getdim(p,n,"n_head",0)!=12 || rad_param_getdim(p,n,"block_size",0)!=4 ||
       rad_param_getdim(p,n,"causal",0)!=1 || rad_param_getdim(p,n,"window",-1)!=0)
        return RAD_E_UNSUPPORTED;
    if(std::strcmp(rad_param_gets(p,n,"q_dtype",""),"bf16") ||
       std::strcmp(rad_param_gets(p,n,"kv_dtype",""),"fp8_e4m3")) return RAD_E_DTYPE;
    return RAD_OK;
}
/* Validate positive non-overlapping strides and address arithmetic without reading data. */
int tensor(const RadTensor& t, uint32_t dtype, uint64_t& bytes) {
    if(!t.data) return RAD_E_INVAL;
    if(t.dtype!=dtype) return RAD_E_DTYPE;
    if(t.rank<1 || t.rank>4) return RAD_E_SHAPE;
    const uint64_t size = dtype==RAD_BF16 ? 2 : dtype==RAD_I32 ? 4 : 1;
    if(reinterpret_cast<uintptr_t>(t.data)%size) return RAD_E_ALIGN;
    uint64_t span=1;
    for(int i=int(t.rank)-1;i>=0;--i) {
        if(t.shape[i]<1 || t.shape[i]>INT32_MAX) return RAD_E_SHAPE;
        if(t.stride[i]<1 || (i<int(t.rank)-1 && uint64_t(t.stride[i])<span)) return RAD_E_STRIDE;
        const uint64_t extent=uint64_t(t.shape[i]-1);
        if(extent && uint64_t(t.stride[i])>(INT64_MAX/size-span)/extent) return RAD_E_SHAPE;
        span+=extent*uint64_t(t.stride[i]);
    }
    bytes=span*size;
    return reinterpret_cast<uintptr_t>(t.data)>UINTPTR_MAX-bytes ? RAD_E_INVAL : RAD_OK;
}
bool dense(const RadTensor& t) {
    int64_t stride=1;
    for(int i=int(t.rank)-1;i>=0;--i) {
        if(t.stride[i]!=stride) return false;
        stride*=t.shape[i];
    }
    return true;
}
const RadConstraint constraints[] = {
    RAD_CGE("q_len",1), RAD_CLE("q_len",65), RAD_CEQ("head_dim",256),
    RAD_CEQ("gqa",12), RAD_CEQ("n_head",12), RAD_CEQ("block_size",4),
    RAD_CEQ("causal",1), RAD_CEQ("window",0),
    RAD_CIN("q_dtype","bf16"), RAD_CIN("kv_dtype","fp8_e4m3")};
}
int rad_attn_paged_tree_schema_count() { return 1; }
const RadOpSchema* rad_attn_paged_tree_schema_at(int i) { return i==0 ? &schema : nullptr; }
int rad_attn_paged_tree_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if(!out || i<0 || i>=RAD_ATP_OPERANDS) return RAD_E_INVAL;
    Geometry g{}; const int rc=geometry(p,n,g); if(rc) return rc;
    RadOpdDesc d{}; d.dtype=RAD_I32; d.idx_const=-1;
    const int64_t blocks=(g.ctx+3)/4;
    if(i==RAD_ATP_Q || i==RAD_ATP_OUT) {
        d.dtype=RAD_BF16; d.rank=2; d.shape[0]=g.tokens; d.shape[1]=3072; d.fill=RAD_FILL_NORMAL;
    } else if(i==RAD_ATP_KV) {
        if(blocks>INT32_MAX/g.seqs) return RAD_E_UNSUPPORTED;
        d.dtype=RAD_F8E4M3; d.rank=4; d.shape[0]=blocks*g.seqs;
        d.shape[1]=1; d.shape[2]=4; d.shape[3]=512; d.fill=RAD_FILL_NORMAL;
    } else if(i==RAD_ATP_BT) {
        d.rank=2; d.shape[0]=g.seqs; d.shape[1]=blocks;
        d.fill=RAD_FILL_INDEX; d.idx_max=blocks*g.seqs;
    } else {
        d.rank=1; d.shape[0]=i==RAD_ATP_SEQ ? g.seqs : i==RAD_ATP_CU ? g.seqs+1 : g.tokens;
        d.fill=RAD_FILL_INDEX;
        d.idx_const=i==RAD_ATP_PARENT ? -1 : i==RAD_ATP_SEQ ? g.ctx : 0;
    }
    *out=d; return RAD_OK;
}
int rad_attn_paged_tree_args(const RadArgs* a,RadAttnPagedTreeCall* out) {
    if(!a || !out || !a->t || a->n_t!=RAD_ATP_OPERANDS) return RAD_E_INVAL;
    Geometry g{}; int rc=geometry(a->p,a->n_p,g); if(rc) return rc;
    uint64_t bytes[RAD_ATP_OPERANDS]{};
    for(int i=0;i<RAD_ATP_OPERANDS;++i) {
        const uint32_t dtype=i==RAD_ATP_Q || i==RAD_ATP_OUT ? uint32_t(RAD_BF16)
            : i==RAD_ATP_KV ? uint32_t(RAD_F8E4M3) : uint32_t(RAD_I32);
        if((rc=tensor(a->t[i],dtype,bytes[i]))) return rc;
    }
    const auto& q=a->t[RAD_ATP_Q]; const auto& o=a->t[RAD_ATP_OUT];
    const auto& kv=a->t[RAD_ATP_KV]; const auto& bt=a->t[RAD_ATP_BT];
    const int64_t rows=q.shape[0], seqs=bt.shape[0];
    if(rows>g.tokens || seqs>g.seqs || bt.rank!=2 || kv.rank!=4) return RAD_E_SHAPE;
    for(int i : {RAD_ATP_Q,RAD_ATP_OUT}) {
        const auto& t=a->t[i];
        if(t.shape[0]!=rows || (t.rank!=2 && t.rank!=3)) return RAD_E_SHAPE;
        if(t.rank==2 ? t.shape[1]!=3072 : (t.shape[1]!=12 || t.shape[2]!=256)) return RAD_E_SHAPE;
        if(!dense(t)) return RAD_E_STRIDE;
    }
    if(kv.shape[1]!=1 || kv.shape[2]!=4 || kv.shape[3]!=512) return RAD_E_SHAPE;
    if(kv.stride[3]!=1 || kv.stride[2]!=512 || bt.stride[1]!=1) return RAD_E_STRIDE;
    const int indices[]={RAD_ATP_SEQ,RAD_ATP_CU,RAD_ATP_PARENT};
    const int64_t lengths[]={seqs,seqs+1,rows};
    for(int j=0;j<3;++j) {
        const auto& t=a->t[indices[j]];
        if(t.rank!=1 || t.shape[0]!=lengths[j]) return RAD_E_SHAPE;
        if(!dense(t)) return RAD_E_STRIDE;
    }
    const uintptr_t begin=reinterpret_cast<uintptr_t>(o.data);
    for(int i=0;i<RAD_ATP_OUT;++i) {
        const uintptr_t x=reinterpret_cast<uintptr_t>(a->t[i].data);
        if(begin<x+bytes[i] && x<begin+bytes[RAD_ATP_OUT]) return RAD_E_INVAL;
    }
    *out={static_cast<const uint16_t*>(q.data),static_cast<const uint8_t*>(kv.data),
          static_cast<const int32_t*>(bt.data),static_cast<const int32_t*>(a->t[RAD_ATP_SEQ].data),
          static_cast<const int32_t*>(a->t[RAD_ATP_CU].data),
          static_cast<const int32_t*>(a->t[RAD_ATP_PARENT].data),static_cast<uint16_t*>(o.data),
          int(rows),int(seqs),int(kv.shape[0]),int(bt.shape[1]),int(g.ctx),int(g.q),
          bt.stride[0],kv.stride[0],kv.stride[1]};
    return RAD_OK;
}
int64_t rad_attn_paged_tree_scratch(const RadArgs* a) {
    Geometry g{};
    if(!a || geometry(a->p,a->n_p,g)) return 0;
    return g.tokens*12*RAD_ATP_SPLITS*RAD_ATP_PARTIAL*int64_t(sizeof(float));
}
RadKernelInfo rad_attn_paged_tree_row(RadLaunchFn launch) {
    RadKernelInfo r{}; r.name="mine_attn_paged_tree_h256_gqa12_b4"; r.op=schema.op;
    r.family="attention"; r.computes=schema.doc;
    r.shape="head_dim256, gqa12, n_head12, block4; ragged <=65 nodes/sequence; ordered I32 parents";
    r.dtypes="bf16 q/out, OCP fp8_e4m3 KV, i32 metadata";
    r.domain=RAD_DOMAIN_DEVICE; r.priority=100; r.launch=launch;
    r.constraints=constraints; r.n_constraints=int(sizeof(constraints)/sizeof(constraints[0]));
    r.opd_shape=rad_attn_paged_tree_shape; r.scratch=rad_attn_paged_tree_scratch;
    return r;
}
