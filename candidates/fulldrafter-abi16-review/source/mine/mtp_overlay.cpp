#include "mtp_overlay.h"
#include <climits>
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace {
#define PI(k) {k,RAD_P_INT,RAD_REQUIRED,0}
#define PS(k) {k,RAD_P_STR,RAD_REQUIRED,0}
const RadParamSpec params[]={{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},
    PI("head_dim"),PI("n_head"),PI("n_head_kv"),PI("block_size"),PI("depth"),
    PS("q_dtype"),PS("kv_dtype"),PI("max_ctx"),PI("N")};
#undef PI
#undef PS
const RadOperandSpec operands[]={{"q",RAD_OPD_IN,0},{"committed_kv",RAD_OPD_IN,0},
    {"committed_bt",RAD_OPD_IN,0},{"prefix",RAD_OPD_IN,0},
    {"provisional_kv",RAD_OPD_IN,0},{"ancestors",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0},
    {"meta",RAD_OPD_INOUT,0},{"request_ids",RAD_OPD_IN,0}};
const RadOpSchema schema={"mtp_overlay_attn",params,10,operands,9,
    "GQA over read-only committed paged prefix and chronological provisional ancestors. "
    "FP32 QK/softmax/PV, BF16 output; invalid metadata invalidates request status and zeroes output."};
int geometry(const RadParam* p,int n,RadMtpOverlayCall& c,bool shape) {
    c.M=shape ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
    c.N=shape ? rad_param_getdim(p,n,"N",0) : rad_param_geti(p,n,"N",0);
    if(c.N<1 || c.N>INT32_MAX) return RAD_E_UNSUPPORTED;
    const int64_t hd=rad_param_geti(p,n,"head_dim",0), nh=rad_param_geti(p,n,"n_head",0);
    const int64_t nk=rad_param_geti(p,n,"n_head_kv",0), bs=rad_param_geti(p,n,"block_size",0);
    const int64_t dep=rad_param_geti(p,n,"depth",0), ctx=rad_param_geti(p,n,"max_ctx",0);
    if(c.M<8 || c.M%8 || c.M>INT32_MAX/4 || hd<1 || hd>256 ||
       nh<1 || nh>65535 || nh>INT32_MAX/hd || nk<1 || nk>nh || nh%nk || bs!=4 ||
       dep<1 || dep>4 || ctx<1 || ctx>INT32_MAX-4) return RAD_E_UNSUPPORTED;
    if((ctx+3)/4*(c.M/8)>INT32_MAX) return RAD_E_UNSUPPORTED;
    if(rad_param_getdt(p,n,"q_dtype")!=RAD_BF16) return RAD_E_DTYPE;
    const char* kv=rad_param_gets(p,n,"kv_dtype","");
    c.kv_dtype=std::strcmp(kv,"fp8_e4m3")==0 ? static_cast<uint32_t>(RAD_F8E4M3) : rad_dtype_parse(kv);
    if(c.kv_dtype!=RAD_BF16 && c.kv_dtype!=RAD_F8E4M3) return RAD_E_DTYPE;
    c.head_dim=static_cast<int>(hd); c.n_head=static_cast<int>(nh); c.n_head_kv=static_cast<int>(nk);
    c.block_size=4; c.depth=static_cast<int>(dep); c.max_ctx=static_cast<int>(ctx);
    return RAD_OK;
}
int tensor(const RadTensor& t,int rank,uint32_t dt,uint64_t& bytes) {
    if(!t.data) return RAD_E_INVAL;
    if(t.dtype!=dt) return RAD_E_DTYPE;
    if(t.rank!=static_cast<uint32_t>(rank)) return RAD_E_SHAPE;
    const uint64_t size=dt==RAD_BF16 ? 2 : dt==RAD_I32 ? 4 : 1;
    if(reinterpret_cast<uintptr_t>(t.data)%size) return RAD_E_ALIGN;
    uint64_t span=1;
    for(int j=rank-1;j>=0;--j) {
        if(t.shape[j]<1 || t.shape[j]>INT32_MAX) return RAD_E_SHAPE;
        if(t.stride[j]<1 || (j<rank-1 && static_cast<uint64_t>(t.stride[j])<span)) return RAD_E_STRIDE;
        const uint64_t extent=static_cast<uint64_t>(t.shape[j]-1);
        if(extent && static_cast<uint64_t>(t.stride[j])>(INT64_MAX/size-span)/extent) return RAD_E_SHAPE;
        span+=extent*static_cast<uint64_t>(t.stride[j]);
    }
    bytes=span*size;
    if(reinterpret_cast<uintptr_t>(t.data)>UINTPTR_MAX-bytes) return RAD_E_INVAL;
    return RAD_OK;
}
}
int rad_mtp_overlay_schema_count() { return 1; }
const RadOpSchema* rad_mtp_overlay_schema_at(int i) { return i==0 ? &schema : nullptr; }
int rad_mtp_overlay_attn_shape(const RadParam* p,int n,int i,RadOpdDesc* out) {
    if(!out || i<0 || i>=9) return RAD_E_INVAL;
    RadMtpOverlayCall c{}; const int rc=geometry(p,n,c,true); if(rc) return rc;
    RadOpdDesc d{}; d.idx_const=-1; d.dtype=RAD_I32;
    if(i==7) {
        d.rank=2; d.shape[0]=c.N; d.shape[1]=85;
    } else if(i==8) {
        d.rank=1; d.shape[0]=c.M/8; d.fill=RAD_FILL_INDEX; d.idx_max=c.N;
    } else if(i==0 || i==6) {
        d.dtype=RAD_BF16; d.rank=2; d.shape[0]=c.M; d.shape[1]=int64_t(c.n_head)*c.head_dim;
    } else if(i==1 || i==4) {
        d.dtype=c.kv_dtype; d.rank=4;
        d.shape[0]=i==1 ? (int64_t(c.max_ctx)+3)/4*(c.M/8) : c.M;
        d.shape[1]=c.n_head_kv; d.shape[2]=4; d.shape[3]=2*c.head_dim;
    } else if(i==2) {
        d.rank=2; d.shape[0]=c.M/8; d.shape[1]=(int64_t(c.max_ctx)+3)/4;
        d.fill=RAD_FILL_INDEX; d.idx_max=(int64_t(c.max_ctx)+3)/4*(c.M/8);
    } else if(i==3) {
        d.rank=1; d.shape[0]=c.M; d.fill=RAD_FILL_INDEX; d.idx_const=c.max_ctx;
    } else {
        d.rank=2; d.shape[0]=c.M; d.shape[1]=4; d.fill=RAD_FILL_INDEX; d.idx_const=-1;
    }
    *out=d; return RAD_OK;
}
int rad_mtp_overlay_attn_args(const RadArgs* a,RadMtpOverlayCall* out) {
    if(!a || !out || !a->t || a->n_t!=9) return RAD_E_INVAL;
    RadMtpOverlayCall c{}; int rc=geometry(a->p,a->n_p,c,false); if(rc) return rc;
    if(a->t[0].rank!=2) return RAD_E_SHAPE;
    c.rows=a->t[0].shape[0];
    if(c.rows<8 || c.rows%8 || c.rows>c.M) return RAD_E_SHAPE;
    const int ranks[]={2,4,2,1,4,2,2,2,1}; uint64_t bytes[9]{};
    for(int i=0;i<9;++i) {
        const uint32_t dt=i==0 || i==6 ? static_cast<uint32_t>(RAD_BF16)
            : i==1 || i==4 ? c.kv_dtype : static_cast<uint32_t>(RAD_I32);
        if((rc=tensor(a->t[i],ranks[i],dt,bytes[i]))) return rc;
        if(ranks[i]>1 && a->t[i].stride[ranks[i]-1]!=1) return RAD_E_STRIDE;
    }
    const int64_t width=int64_t(c.n_head)*c.head_dim;
    for(int i : {0,6}) if(a->t[i].shape[0]!=c.rows || a->t[i].shape[1]!=width) return RAD_E_SHAPE;
    if(a->t[2].shape[0]!=c.rows/8 || a->t[3].shape[0]!=c.rows ||
       a->t[5].shape[0]!=c.rows || a->t[5].shape[1]!=4) return RAD_E_SHAPE;
    if(a->t[7].shape[0]!=c.N || a->t[7].shape[1]!=85 || a->t[8].shape[0]!=c.rows/8) return RAD_E_SHAPE;
    for(int i : {1,4}) {
        const RadTensor& t=a->t[i];
        if(t.shape[1]!=c.n_head_kv || t.shape[2]!=4 || t.shape[3]!=2*c.head_dim) return RAD_E_SHAPE;
        if(t.stride[2]!=2*c.head_dim) return RAD_E_STRIDE;
    }
    if(a->t[4].shape[0]<c.rows) return RAD_E_SHAPE;
    for(int w : {6,7}) for(int i=0;i<9;++i) {
        if(i==w) continue;
        const uintptr_t x=reinterpret_cast<uintptr_t>(a->t[i].data);
        const uintptr_t y=reinterpret_cast<uintptr_t>(a->t[w].data);
        if(x<y+bytes[w] && y<x+bytes[i]) return RAD_E_INVAL;
    }
    c.q=static_cast<const uint16_t*>(a->t[0].data); c.committed_kv=a->t[1].data;
    c.committed_bt=static_cast<const int32_t*>(a->t[2].data); c.prefix=static_cast<const int32_t*>(a->t[3].data);
    c.provisional_kv=a->t[4].data; c.ancestors=static_cast<const int32_t*>(a->t[5].data);
    c.out=static_cast<uint16_t*>(a->t[6].data);
    c.pages=a->t[1].shape[0]; c.provisional_pages=a->t[4].shape[0]; c.bt_pitch=a->t[2].shape[1];
    c.q_ld=a->t[0].stride[0]; c.out_ld=a->t[6].stride[0]; c.bt_ld=a->t[2].stride[0];
    c.prefix_stride=a->t[3].stride[0]; c.ancestor_ld=a->t[5].stride[0];
    for(int j=0;j<4;++j) { c.committed_stride[j]=a->t[1].stride[j]; c.provisional_stride[j]=a->t[4].stride[j]; }
    c.meta=static_cast<int32_t*>(a->t[7].data); c.meta_ld=a->t[7].stride[0];
    c.request_ids=static_cast<const int32_t*>(a->t[8].data); c.request_stride=a->t[8].stride[0];
    *out=c; return RAD_OK;
}
RadKernelInfo rad_mtp_overlay_attn_row(RadLaunchFn fn) {
    RadKernelInfo r{}; r.name="mine_mtp_overlay_attn"; r.op=schema.op; r.family="draft";
    r.computes=schema.doc; r.shape="R=8Q<=M, head_dim1..256, GQA, block4, depth1..4; strided paged KV";
    r.dtypes="bf16 q/out; bf16 or fp8_e4m3 KV; i32 metadata";
    r.domain=RAD_DOMAIN_DEVICE; r.priority=100; r.launch=fn; r.opd_shape=rad_mtp_overlay_attn_shape;
    return r;
}
