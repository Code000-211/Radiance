#include "mtp_scored_tree.h"
#include <climits>
#include <cmath>
#include <cstdint>

namespace {
#define PI(k) {k, RAD_P_INT, RAD_REQUIRED, 0}
#define PF(k) {k, RAD_P_F64, RAD_OPTIONAL, 0}
const RadParamSpec lp[] = {{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY}, PI("N"), PI("n_vocab"), PI("vocab_off")};
const RadParamSpec mp[] = {{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY}, PI("pitch"), PI("capacity"), PI("max_rounds"),
    PI("n_vocab"), PI("round"), PI("world_size"), PI("policy"),
    PF("cost0"), PF("cost1"), PF("cost2"), PF("cost3"),
    PF("cost4"), PF("cost5"), PF("cost6"), PF("cost7")};
#undef PI
#undef PF
const RadOperandSpec lo[] = {{"coarse",RAD_OPD_IN,0},{"idx",RAD_OPD_IN,0},
    {"val",RAD_OPD_IN,0},{"stats",RAD_OPD_OUT,0}};
const RadOperandSpec mo[] = {{"gathered",RAD_OPD_IN,0},{"stats",RAD_OPD_IN,1},
    {"idx",RAD_OPD_OUT,0},{"val",RAD_OPD_OUT,0},{"token_ids",RAD_OPD_IN,1},
    {"draft_out_ids",RAD_OPD_IN,1},{"history",RAD_OPD_INOUT,0},
    {"log_history",RAD_OPD_INOUT,1},{"packed",RAD_OPD_INOUT,0}};
const RadOpSchema schemas[] = {
    {"mtp_hybrid_lse",lp,4,lo,4,"Stable max/scaled sum of BF16 coarse logits with exact local top-eight replacements; separate aligned metadata."},
    {"mtp_scored_merge_tree_accum",mp,16,mo,9,"TP2 bit-key merge and persistent ancestor-closed MTP log-probability tree selection."}};
int lgeom(const RadParam* p,int n,RadMtpHybridLseCall& c,bool wide) {
    c.M = wide ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
    c.N = rad_param_geti(p,n,"N",0);
    const int64_t v=rad_param_geti(p,n,"n_vocab",0), off=rad_param_geti(p,n,"vocab_off",-1);
    if(c.M<1 || c.M>INT32_MAX || c.N<1 || c.N>INT32_MAX || v<1 || v>INT32_MAX || off<0 || off>INT32_MAX)
        return RAD_E_UNSUPPORTED;
    c.vocab=static_cast<int>(v); c.vocab_off=static_cast<int>(off);
    return RAD_OK;
}
int mgeom(const RadParam* p,int n,RadMtpScoredMergeTreeAccumCall& c,bool wide) {
    c.M=wide ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
    const int64_t cap=rad_param_geti(p,n,"capacity",0), rounds=rad_param_geti(p,n,"max_rounds",0);
    const int64_t v=rad_param_geti(p,n,"n_vocab",0), r=rad_param_geti(p,n,"round",0), policy=rad_param_geti(p,n,"policy",-1);
    if(c.M<1 || c.M>INT32_MAX || cap<2 || cap>8 || (rounds!=3 && rounds!=5) ||
       v<1 || v>INT32_MAX || r<1 || r>rounds || policy<0 || policy>2 ||
       rad_param_geti(p,n,"pitch",0)!=8 || rad_param_geti(p,n,"world_size",0)!=2) return RAD_E_UNSUPPORTED;
    c.capacity=static_cast<int>(cap); c.max_rounds=static_cast<int>(rounds);
    c.vocab=static_cast<int>(v); c.round=static_cast<int>(r); c.policy=static_cast<int>(policy);
    if(policy!=2) return RAD_OK;
    for(int k=0;k<8;++k) {
        char key[]="cost0"; key[4]=static_cast<char>('0'+k);
        const RadParam* f=rad_param_find(p,n,key);
        if(!f || f->kind!=RAD_P_F64 || !std::isfinite(f->dval) || f->dval<=0) return RAD_E_INVAL;
        c.costs[k]=f->dval;
    }
    return RAD_OK;
}
int tensor(const RadTensor& t,int rank,const int64_t* dims,uint32_t dt,uint64_t& bytes) {
    if(!t.data) return RAD_E_INVAL;
    const uint64_t size=dt==RAD_BF16 ? 2 : 4;
    if(reinterpret_cast<uintptr_t>(t.data)%size) return RAD_E_INVAL;
    if(t.dtype!=dt) return RAD_E_DTYPE;
    if(t.rank!=static_cast<uint32_t>(rank)) return RAD_E_SHAPE;
    uint64_t span=1;
    for(int j=rank-1;j>=0;--j) {
        if(t.shape[j]<dims[j] || t.shape[j]>INT32_MAX) return RAD_E_SHAPE;
        if(t.stride[j]<1 || (j<rank-1 && static_cast<uint64_t>(t.stride[j])<span)) return RAD_E_STRIDE;
        const uint64_t extent=static_cast<uint64_t>(t.shape[j]-1);
        if(extent && static_cast<uint64_t>(t.stride[j])>(INT64_MAX/size-span)/extent) return RAD_E_SHAPE;
        span+=extent*static_cast<uint64_t>(t.stride[j]);
    }
    bytes=span*size;
    if(reinterpret_cast<uintptr_t>(t.data)>UINTPTR_MAX-bytes) return RAD_E_INVAL;
    return RAD_OK;
}
bool overlaps(const RadTensor& a,uint64_t an,const RadTensor& b,uint64_t bn) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a.data), y=reinterpret_cast<uintptr_t>(b.data);
    return an && bn && x<y+bn && y<x+an;
}
}
int rad_mtp_scored_tree_schema_count() { return 2; }
const RadOpSchema* rad_mtp_scored_tree_schema_at(int i) { return i>=0 && i<2 ? schemas+i : nullptr; }
int rad_mtp_hybrid_lse_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if(!out || i<0 || i>3) return RAD_E_INVAL;
    RadMtpHybridLseCall c{}; const int rc=lgeom(p,n,c,true); if(rc) return rc;
    RadOpdDesc d{}; d.idx_const=-1; d.rank=2; d.shape[0]=c.M;
    d.shape[1]=i==0 ? c.N : i==3 ? 4 : 8;
    d.dtype=i==0 ? RAD_BF16 : i==1 ? RAD_I32 : RAD_F32;
    if(i==1) { d.fill=RAD_FILL_INDEX; d.idx_max=c.vocab; }
    *out=d; return RAD_OK;
}
int rad_mtp_hybrid_lse_args(const RadArgs* a,RadMtpHybridLseCall* out) {
    if(!a || !out || !a->t || a->n_t!=4) return RAD_E_INVAL;
    RadMtpHybridLseCall c{}; int rc=lgeom(a->p,a->n_p,c,false); if(rc) return rc;
    if(a->t[0].rank!=2 || a->t[0].shape[0]<1 || a->t[0].shape[0]>c.M) return RAD_E_SHAPE;
    c.M=a->t[0].shape[0];
    const int64_t dims[][2]={{c.M,c.N},{c.M,8},{c.M,8},{c.M,4}};
    const uint32_t types[]={RAD_BF16,RAD_I32,RAD_F32,RAD_F32}; uint64_t bytes[4]{};
    for(int i=0;i<4;++i) {
        if((rc=tensor(a->t[i],2,dims[i],types[i],bytes[i]))) return rc;
        if(a->t[i].shape[0]!=c.M || a->t[i].shape[1]!=dims[i][1]) return RAD_E_SHAPE;
        if(a->t[i].stride[1]!=1) return RAD_E_STRIDE;
    }
    if(reinterpret_cast<uintptr_t>(a->t[3].data)%16 || a->t[3].stride[0]%4) return RAD_E_STRIDE;
    for(int i=0;i<3;++i) if(overlaps(a->t[i],bytes[i],a->t[3],bytes[3])) return RAD_E_INVAL;
    c.coarse=static_cast<const uint16_t*>(a->t[0].data); c.idx=static_cast<const int32_t*>(a->t[1].data);
    c.val=static_cast<const float*>(a->t[2].data); c.stats=static_cast<float*>(a->t[3].data);
    c.coarse_ld=a->t[0].stride[0]; c.idx_ld=a->t[1].stride[0]; c.val_ld=a->t[2].stride[0]; c.stats_ld=a->t[3].stride[0];
    *out=c; return RAD_OK;
}
int rad_mtp_scored_merge_tree_accum_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if(!out || i<0 || i>8) return RAD_E_INVAL;
    RadMtpScoredMergeTreeAccumCall c{}; const int rc=mgeom(p,n,c,true); if(rc) return rc;
    RadOpdDesc d{}; d.idx_const=-1; d.rank=2; d.shape[0]=c.M; d.shape[1]=8;
    d.dtype=(i==0 || i==1 || i==3 || i==7) ? RAD_F32 : RAD_I32;
    if(i==0) { d.rank=3; d.shape[1]=16; d.shape[2]=2; }
    if(i==4 || i==5) { d.rank=1; d.fill=RAD_FILL_INDEX; d.idx_max=i==4 ? c.vocab : c.M; if(c.round>1) d.flags=RAD_OPD_F_ABSENT; }
    if(i==6 || i==7) { d.rank=3; d.shape[0]=c.max_rounds; d.shape[1]=c.M; d.shape[2]=8; }
    if(i==8) d.shape[1]=1+2*c.capacity;
    if(c.policy==0 && (i==1 || i==7)) d.flags=RAD_OPD_F_ABSENT;
    *out=d; return RAD_OK;
}
int rad_mtp_scored_merge_tree_accum_args(const RadArgs* a,RadMtpScoredMergeTreeAccumCall* out) {
    if(!a || !out || !a->t || a->n_t!=9) return RAD_E_INVAL;
    RadMtpScoredMergeTreeAccumCall c{}; int rc=mgeom(a->p,a->n_p,c,false); if(rc) return rc;
    if(a->t[8].rank!=2 || a->t[8].shape[0]<1 || a->t[8].shape[0]>c.M) return RAD_E_SHAPE;
    c.M=a->t[8].shape[0];
    const int64_t dims[][3]={{c.M,16,2},{c.M,8},{c.M,8},{c.M,8},{1},{c.M},
        {c.max_rounds,c.M,8},{c.max_rounds,c.M,8},{c.M,1+2*c.capacity}};
    const int ranks[]={3,2,2,2,1,1,3,3,2}; uint64_t bytes[9]{};
    for(int i=0;i<9;++i) {
        if(c.round>1 && (i==4 || i==5)) continue;
        if(c.policy==0 && (i==1 || i==7)) continue;
        const uint32_t dt=(i==0 || i==1 || i==3 || i==7) ? RAD_F32 : RAD_I32;
        if((rc=tensor(a->t[i],ranks[i],dims[i],dt,bytes[i]))) return rc;
        if(i!=4 && i!=5 && a->t[i].stride[ranks[i]-1]!=1) return RAD_E_STRIDE;
        if(i!=4 && i!=5) {
            if(i==6 || i==7) { if(a->t[i].shape[0]!=c.max_rounds || a->t[i].shape[2]!=8) return RAD_E_SHAPE; }
            else for(int j=0;j<ranks[i];++j) if(a->t[i].shape[j]!=dims[i][j]) return RAD_E_SHAPE;
        }
    }
    if(a->t[0].stride[1]!=2) return RAD_E_STRIDE;
    if(c.policy!=0 && (reinterpret_cast<uintptr_t>(a->t[1].data)%16 || a->t[1].stride[0]%4)) return RAD_E_STRIDE;
    for(int i=0;i<9;++i) for(int j=i+1;j<9;++j) {
        if(c.policy==0 && (i==1 || i==7 || j==1 || j==7)) continue;
        if((i==2 || i==3 || i>=6 || j==2 || j==3 || j>=6) && overlaps(a->t[i],bytes[i],a->t[j],bytes[j])) return RAD_E_INVAL;
    }
    c.gathered=static_cast<const uint32_t*>(a->t[0].data);
    if(c.policy!=0) c.stats=static_cast<const float*>(a->t[1].data);
    c.idx=static_cast<int32_t*>(a->t[2].data); c.val=static_cast<uint32_t*>(a->t[3].data);
    c.tokens=static_cast<const int32_t*>(a->t[4].data); c.ids=static_cast<const int32_t*>(a->t[5].data);
    c.history=static_cast<int32_t*>(a->t[6].data); c.packed=static_cast<int32_t*>(a->t[8].data);
    if(c.policy!=0) c.log_history=static_cast<float*>(a->t[7].data);
    c.token_count=c.round==1 ? a->t[4].shape[0] : 0;
    c.gathered_ld=a->t[0].stride[0]; c.idx_ld=a->t[2].stride[0]; c.val_ld=a->t[3].stride[0];
    if(c.policy!=0) c.stats_ld=a->t[1].stride[0];
    c.token_stride=a->t[4].stride[0]; c.id_stride=a->t[5].stride[0];
    c.hist_round_stride=a->t[6].stride[0]; c.hist_row_stride=a->t[6].stride[1];
    c.out_ld=a->t[8].stride[0];
    if(c.policy!=0) { c.log_round_stride=a->t[7].stride[0]; c.log_row_stride=a->t[7].stride[1]; }
    *out=c; return RAD_OK;
}
RadKernelInfo rad_mtp_hybrid_lse_row(RadLaunchFn fn) {
    RadKernelInfo r{}; r.name="mine_mtp_hybrid_lse"; r.op=schemas[0].op; r.family="draft";
    r.computes=schemas[0].doc; r.shape="M,N,n_vocab,vocab_off; local8 exact replacement";
    r.dtypes="bf16 coarse, i32 idx, f32 val/stats"; r.domain=RAD_DOMAIN_DEVICE; r.priority=100;
    r.launch=fn; r.opd_shape=rad_mtp_hybrid_lse_shape; return r;
}
RadKernelInfo rad_mtp_scored_merge_tree_accum_row(RadLaunchFn fn) {
    RadKernelInfo r=rad_mtp_hybrid_lse_row(fn); r.name="mine_mtp_scored_merge_tree_accum"; r.op=schemas[1].op;
    r.computes=schemas[1].doc; r.shape="TP2, pitch8, capacity2..8, rounds3/5, policy0/1/2";
    r.dtypes="f32 pairs/stats/log_history, i32 ids/history/tree";
    r.opd_shape=rad_mtp_scored_merge_tree_accum_shape; return r;
}
