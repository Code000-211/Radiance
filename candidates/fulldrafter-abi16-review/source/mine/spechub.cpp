#include "spechub.h"
#include <climits>
#include <cstddef>
namespace {
#define PI(k) {k,RAD_P_INT,RAD_REQUIRED,0}
const RadParamSpec pp[]={PI("M"),PI("n_vocab"),PI("world_size")};
const RadParamSpec vp[]={PI("M"),PI("n_cand"),PI("n_vocab"),{"transport",RAD_P_INT,RAD_OPTIONAL,0}};
const RadParamSpec cp[]={PI("M"),PI("n_vocab"),PI("round"),PI("max_rounds")};
const RadParamSpec cvp[]={PI("M"),PI("n_cand"),PI("n_vocab")};
#undef PI
const RadOperandSpec co[]={
    {"idx",RAD_OPD_INOUT,0},{"val",RAD_OPD_IN,0},{"seq_ids",RAD_OPD_IN,0},
    {"draw",RAD_OPD_IN,0},{"state",RAD_OPD_INOUT,0}};
const RadOperandSpec po[]={
    {"gathered",RAD_OPD_IN,0},{"draw",RAD_OPD_IN,0},{"token_ids",RAD_OPD_IN,0},
    {"draft_out_ids",RAD_OPD_IN,0},{"state",RAD_OPD_OUT,0},{"packed",RAD_OPD_OUT,0},
    {"idx",RAD_OPD_OUT,0},{"val",RAD_OPD_OUT,0}};
const RadOperandSpec vo[]={
    {"cand_idx",RAD_OPD_IN,0},{"cand_val",RAD_OPD_IN,0},{"params",RAD_OPD_IN,0},
    {"state",RAD_OPD_IN,0},{"token",RAD_OPD_INOUT,0}};
const RadOpSchema schemas[]={
    {"spechub_propose",pp,3,po,8,"TP2 top8 FP64 SpecHub Q draw and GPU root-inclusive two-sibling tree; malformed rows reset to status=-1/count=0."},
    {"spechub_verify",vp,4,vo,5,"Appendix B sparse transport over all surviving target candidates; transport=0 uses ordinary PICK on the same FP64-normalized p. Default transport=1. FP64 normalization reorders the FP32 pick. Disabled rows preserve token; malformed rows emit -1."},
    {"mtp_couple_propose",cp,4,co,5,"Linear MTP round `round`: x~Q, Q = FP64 T1 softmax of the final top8 scores; x into idx column 0 and state[round-1][row]. mode0 rows keep column 0 (status 0); bad ids/scores or seq_ids != draw.card keep it too (status -1)."},
    {"mtp_couple_verify",cvp,3,vo,5,"Maximal coupling of a bound mtp_couple state after sample_pick over all surviving target candidates, FP64 P and 53-bit ACCEPT/RESID uniforms at params.pos; residual (P-Q)+ with r(x)=0. Rows whose state is not status 1, not bound to params seed/pos, malformed, or whose (P,Q) admit no coupling keep the ordinary token."}};
int geometry(const RadParam* p,int n,int op,bool range,int64_t& M,int64_t& vocab,int64_t& nc) {
    if (!p) return RAD_E_INVAL;
    M=range ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
    vocab=rad_param_geti(p,n,"n_vocab",0);
    nc=op ? rad_param_geti(p,n,"n_cand",0) : 8;
    if (M<1 || M>INT32_MAX || vocab<1 || vocab>INT32_MAX || nc<1 || nc>INT32_MAX) return RAD_E_UNSUPPORTED;
    if (!op && rad_param_geti(p,n,"world_size",0)!=2) return RAD_E_UNSUPPORTED;
    const int64_t transport=rad_param_geti(p,n,"transport",1);
    if (op && transport!=0 && transport!=1) return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
int span(const RadTensor& t,const RadOpdDesc& d,uint64_t& bytes) {
    if (!t.data || reinterpret_cast<uintptr_t>(t.data)%4) return RAD_E_INVAL;
    if (t.dtype!=d.dtype) return RAD_E_DTYPE;
    if (t.rank!=d.rank) return RAD_E_SHAPE;
    uint64_t words=1;
    for (int j=static_cast<int>(d.rank)-1;j>=0;--j) {
        if (t.shape[j]<1 || t.shape[j]>INT32_MAX || (d.shape[j]>0 && t.shape[j]!=d.shape[j])) return RAD_E_SHAPE;
        if (t.stride[j]<1 || (j==static_cast<int>(d.rank)-1 && t.stride[j]!=1) ||
            (j<static_cast<int>(d.rank)-1 && static_cast<uint64_t>(t.stride[j])<words)) return RAD_E_STRIDE;
        uint64_t extent=static_cast<uint64_t>(t.shape[j]-1);
        if (extent && static_cast<uint64_t>(t.stride[j])>(INT64_MAX/4-words)/extent) return RAD_E_SHAPE;
        words+=extent*static_cast<uint64_t>(t.stride[j]);
    }
    bytes=words*4;
    if (reinterpret_cast<uintptr_t>(t.data)>UINTPTR_MAX-bytes) return RAD_E_INVAL;
    return RAD_OK;
}
int validate(const RadArgs* a,int op,int64_t& M,int64_t& vocab,int64_t& nc) {
    if (!a || !a->t || a->n_t!=(op ? 5 : 8)) return RAD_E_INVAL;
    int rc=geometry(a->p,a->n_p,op,false,M,vocab,nc);if (rc!=RAD_OK) return rc;
    // Resolved M is the arena band ceiling; this issue may contain fewer live rows.
    const int64_t live=a->t[0].rank ? a->t[0].shape[0] : 0;
    if (live<1 || live>M) return RAD_E_SHAPE;
    M=live;
    uint64_t bytes[8]={};
    for (int i=0;i<a->n_t;++i) {
        RadOpdDesc d{};
        rc=op ? rad_spechub_verify_shape(a->p,a->n_p,i,&d) : rad_spechub_propose_shape(a->p,a->n_p,i,&d);
        if (rc!=RAD_OK) return rc;
        if (op || i!=2) d.shape[0]=M;
        // The sampler exposes both flat tokens and its production [M,1] plane.
        if (op && i==4 && a->t[i].rank==2) {
            d.rank=2;d.shape[1]=1;
        }
        if ((rc=span(a->t[i],d,bytes[i]))!=RAD_OK) return rc;
        const bool wire=op ? i==2 || i==3 : i==1 || i==4;
        if (wire && (reinterpret_cast<uintptr_t>(a->t[i].data)%8 || a->t[i].stride[0]%2)) return RAD_E_INVAL;
    }
    if (!op && a->t[0].stride[1]!=2) return RAD_E_STRIDE;
    // Separate storage is part of the wire contract, including input/input aliases.
    for (int i=0;i<a->n_t;++i) for (int j=i+1;j<a->n_t;++j) {
        uintptr_t x=reinterpret_cast<uintptr_t>(a->t[i].data),y=reinterpret_cast<uintptr_t>(a->t[j].data);
        if (x<y+bytes[j] && y<x+bytes[i]) return RAD_E_INVAL;
    }
    return RAD_OK;
}
int couple_geometry(const RadParam* p,int n,int op,bool range,int64_t& M,int64_t& vocab,int64_t& nc,
                    int64_t& round,int64_t& rounds) {
    if (!p) return RAD_E_INVAL;
    M=range ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
    vocab=rad_param_geti(p,n,"n_vocab",0);
    nc=op ? rad_param_geti(p,n,"n_cand",0) : 8;
    round=op ? 1 : rad_param_geti(p,n,"round",0);
    rounds=op ? 1 : rad_param_geti(p,n,"max_rounds",0);
    if (M<1 || M>INT32_MAX || vocab<1 || vocab>INT32_MAX || nc<1 || nc>INT32_MAX ||
        rounds<1 || rounds>RAD_MTP_COUPLE_MAX_ROUNDS || round<1 || round>rounds) return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
/* Live rows are operand 0's; seq_ids may cover more entries and the state plane more rows. */
int couple_validate(const RadArgs* a,int op,int64_t& M,int64_t& vocab,int64_t& nc,int64_t& round) {
    if (!a || !a->t || a->n_t!=5) return RAD_E_INVAL;
    int64_t rounds=0;
    int rc=couple_geometry(a->p,a->n_p,op,false,M,vocab,nc,round,rounds);if (rc!=RAD_OK) return rc;
    const int64_t live=a->t[0].rank ? a->t[0].shape[0] : 0;
    if (live<1 || live>M) return RAD_E_SHAPE;
    M=live;
    uint64_t bytes[5]={};
    for (int i=0;i<5;++i) {
        RadOpdDesc d{};
        rc=op ? rad_mtp_couple_verify_shape(a->p,a->n_p,i,&d) : rad_mtp_couple_propose_shape(a->p,a->n_p,i,&d);
        if (rc!=RAD_OK) return rc;
        d.shape[0]=M;
        if (!op && i==2) d.shape[0]=0;
        if (!op && i==4) {d.shape[0]=rounds;d.shape[1]=0;}
        if (op && i==4 && a->t[i].rank==2) {d.rank=2;d.shape[1]=1;}
        if ((rc=span(a->t[i],d,bytes[i]))!=RAD_OK) return rc;
        if (!op && i==2 && a->t[i].shape[0]<M) return RAD_E_SHAPE;
        if (!op && i==4 && a->t[i].shape[1]<M) return RAD_E_SHAPE;
        const bool wire=op ? i==2 || i==3 : i==3 || i==4;
        if (wire && (reinterpret_cast<uintptr_t>(a->t[i].data)%8 || a->t[i].stride[0]%2 ||
                     (a->t[i].rank==3 && a->t[i].stride[1]%2))) return RAD_E_INVAL;
    }
    for (int i=0;i<5;++i) for (int j=i+1;j<5;++j) {
        uintptr_t x=reinterpret_cast<uintptr_t>(a->t[i].data),y=reinterpret_cast<uintptr_t>(a->t[j].data);
        if (x<y+bytes[j] && y<x+bytes[i]) return RAD_E_INVAL;
    }
    return RAD_OK;
}
}
int rad_spechub_schema_count() {return 4;}
const RadOpSchema* rad_spechub_schema_at(int i) {return i>=0 && i<4 ? &schemas[i] : nullptr;}
int rad_mtp_couple_propose_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if (!out || i<0 || i>=5) return RAD_E_INVAL;
    int64_t M,v,nc,round,rounds;int rc=couple_geometry(p,n,0,true,M,v,nc,round,rounds);if (rc!=RAD_OK) return rc;
    RadOpdDesc d{};d.idx_const=-1;d.dtype=RAD_I32;d.rank=2;d.shape[0]=M;d.shape[1]=8;
    if (i==0) {d.fill=RAD_FILL_INDEX;d.idx_max=v;}
    if (i==1) d.dtype=RAD_F32;
    if (i==2) {d.rank=1;d.fill=RAD_FILL_INDEX;d.idx_max=M;}
    if (i==3) {d.dtype=RAD_U32;d.shape[1]=RAD_MTP_COUPLE_DRAW_WORDS;}
    if (i==4) {d.dtype=RAD_U32;d.rank=3;d.shape[0]=rounds;d.shape[1]=M;d.shape[2]=RAD_MTP_COUPLE_STATE_WORDS;}
    *out=d;return RAD_OK;
}
int rad_mtp_couple_verify_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if (!out || i<0 || i>=5) return RAD_E_INVAL;
    int64_t M,v,nc,round,rounds;int rc=couple_geometry(p,n,1,true,M,v,nc,round,rounds);if (rc!=RAD_OK) return rc;
    RadOpdDesc d{};d.idx_const=-1;d.dtype=RAD_I32;d.rank=2;d.shape[0]=M;d.shape[1]=nc;
    if (i==1) d.dtype=RAD_F32;
    if (i==2 || i==3) {d.dtype=RAD_U32;d.shape[1]=i==2 ? 24 : RAD_MTP_COUPLE_STATE_WORDS;}
    if (i==4) d.rank=1;
    *out=d;return RAD_OK;
}
int rad_mtp_couple_propose_args(const RadArgs* a,RadMtpCoupleProposeCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M,v,nc,round;int rc=couple_validate(a,0,M,v,nc,round);if (rc!=RAD_OK) return rc;
    RadMtpCoupleProposeCall c{};c.M=M;c.vocab=v;c.round=static_cast<int>(round);
    c.idx=static_cast<int32_t*>(a->t[0].data);c.val=static_cast<const float*>(a->t[1].data);
    c.card=static_cast<const int32_t*>(a->t[2].data);c.draw=static_cast<const uint32_t*>(a->t[3].data);
    c.state=static_cast<uint32_t*>(a->t[4].data);
    c.ld_idx=a->t[0].stride[0];c.ld_val=a->t[1].stride[0];c.ld_draw=a->t[3].stride[0];
    c.st_round=a->t[4].stride[0];c.st_row=a->t[4].stride[1];
    *out=c;return RAD_OK;
}
int rad_mtp_couple_verify_args(const RadArgs* a,RadMtpCoupleVerifyCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M,v,nc,round;int rc=couple_validate(a,1,M,v,nc,round);if (rc!=RAD_OK) return rc;
    RadMtpCoupleVerifyCall c{};c.M=M;c.vocab=v;c.n_cand=nc;
    c.idx=static_cast<const int32_t*>(a->t[0].data);c.val=static_cast<const float*>(a->t[1].data);
    c.params=static_cast<const uint32_t*>(a->t[2].data);c.state=static_cast<const uint32_t*>(a->t[3].data);
    c.token=static_cast<int32_t*>(a->t[4].data);
    for (int i=0;i<5;++i) c.ld[i]=a->t[i].stride[0];
    *out=c;return RAD_OK;
}
RadKernelInfo rad_mtp_couple_row(int i,RadLaunchFn launch) {
    RadKernelInfo r{};r.name=i ? "mine_mtp_couple_verify" : "mine_mtp_couple_propose";r.op=schemas[2+i].op;
    r.family="draft";r.computes=schemas[2+i].doc;r.shape=i ? "M,n_cand,n_vocab" : "M,n_vocab,round,max_rounds<=8";
    r.dtypes="f32,i32,u32";r.domain=RAD_DOMAIN_DEVICE;r.priority=100;r.launch=launch;
    r.opd_shape=i ? rad_mtp_couple_verify_shape : rad_mtp_couple_propose_shape;return r;
}
int rad_spechub_propose_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if (!out || i<0 || i>=8) return RAD_E_INVAL;
    int64_t M,v,nc;int rc=geometry(p,n,0,true,M,v,nc);if (rc!=RAD_OK) return rc;
    RadOpdDesc d{};d.idx_const=-1;d.dtype=RAD_I32;d.rank=2;d.shape[0]=M;d.shape[1]=8;
    if (i==0) {d.dtype=RAD_F32;d.rank=3;d.shape[1]=16;d.shape[2]=2;}
    if (i==1 || i==4) {d.dtype=RAD_U32;d.shape[1]=i==1 ? RAD_SPECHUB_DRAW_WORDS : RAD_SPECHUB_STATE_WORDS;}
    if (i==2 || i==3) {d.rank=1;d.shape[0]=i==2 ? 0 : M;d.fill=RAD_FILL_INDEX;d.idx_max=i==2 ? v : M;}
    if (i==5) d.shape[1]=7;
    if (i==7) d.dtype=RAD_F32;
    *out=d;return RAD_OK;
}
int rad_spechub_verify_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if (!out || i<0 || i>=5) return RAD_E_INVAL;
    int64_t M,v,nc;int rc=geometry(p,n,1,true,M,v,nc);if (rc!=RAD_OK) return rc;
    RadOpdDesc d{};d.idx_const=-1;d.dtype=RAD_I32;d.rank=2;d.shape[0]=M;d.shape[1]=nc;
    if (i==1) d.dtype=RAD_F32;
    if (i==2 || i==3) {d.dtype=RAD_U32;d.shape[1]=i==2 ? 24 : RAD_SPECHUB_STATE_WORDS;}
    if (i==4) d.rank=1;
    *out=d;return RAD_OK;
}
int rad_spechub_propose_args(const RadArgs* a,RadSpechubProposeCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M,v,nc;int rc=validate(a,0,M,v,nc);if (rc!=RAD_OK) return rc;
    RadSpechubProposeCall c{};c.M=M;c.vocab=static_cast<int>(v);c.token_count=a->t[2].shape[0];
    c.gathered=static_cast<const uint32_t*>(a->t[0].data);c.draw=static_cast<const uint32_t*>(a->t[1].data);
    c.tokens=static_cast<const int32_t*>(a->t[2].data);c.draft_ids=static_cast<const int32_t*>(a->t[3].data);
    c.state=static_cast<uint32_t*>(a->t[4].data);c.packed=static_cast<int32_t*>(a->t[5].data);
    c.idx=static_cast<int32_t*>(a->t[6].data);c.val=static_cast<float*>(a->t[7].data);
    for (int i=0;i<8;++i) c.ld[i]=a->t[i].stride[0];
    *out=c;return RAD_OK;
}
int rad_spechub_verify_args(const RadArgs* a,RadSpechubVerifyCall* out) {
    if (!out) return RAD_E_INVAL;
    int64_t M,v,nc;int rc=validate(a,1,M,v,nc);if (rc!=RAD_OK) return rc;
    RadSpechubVerifyCall c{};c.M=M;c.vocab=static_cast<int>(v);c.n_cand=nc;
    c.transport=static_cast<int>(rad_param_geti(a->p,a->n_p,"transport",1));
    c.idx=static_cast<const int32_t*>(a->t[0].data);c.val=static_cast<const float*>(a->t[1].data);
    c.params=static_cast<const uint32_t*>(a->t[2].data);c.state=static_cast<const uint32_t*>(a->t[3].data);
    c.token=static_cast<int32_t*>(a->t[4].data);
    for (int i=0;i<5;++i) c.ld[i]=a->t[i].stride[0];
    *out=c;return RAD_OK;
}
RadKernelInfo rad_spechub_row(int i,RadLaunchFn launch) {
    RadKernelInfo r{};r.name=i ? "mine_spechub_verify" : "mine_spechub_propose";r.op=schemas[i].op;
    r.family="draft";r.computes=schemas[i].doc;r.shape=i ? "M,n_cand,n_vocab" : "M,n_vocab,world_size=2; capacity=3";
    r.dtypes="f32,i32,u32";r.domain=RAD_DOMAIN_DEVICE;r.priority=100;r.launch=launch;
    r.opd_shape=i ? rad_spechub_verify_shape : rad_spechub_propose_shape;return r;
}
