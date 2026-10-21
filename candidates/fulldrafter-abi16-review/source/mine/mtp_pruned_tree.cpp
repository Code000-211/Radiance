#include "mtp_pruned_tree.h"
#include <climits>
#include <cmath>
#include <cstdint>

namespace {
const RadParamSpec ep[]={{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"round",RAD_P_INT,RAD_REQUIRED,0},{"n_vocab",RAD_P_INT,RAD_REQUIRED,0}};
const RadParamSpec bp[]={{"M",RAD_P_INT,RAD_REQUIRED,RAD_PROLE_CAPACITY},{"beta",RAD_P_F64,RAD_REQUIRED,0}};
const RadOperandSpec eo[]={{"gathered",RAD_OPD_IN,0},{"request_ids",RAD_OPD_IN,1},{"tokens",RAD_OPD_IN,1},{"anchor_rows",RAD_OPD_IN,1},{"positions",RAD_OPD_IN,1},{"meta",RAD_OPD_INOUT,0},{"score",RAD_OPD_INOUT,0},{"packed",RAD_OPD_INOUT,0}};
const RadOperandSpec po[]={{"request_ids",RAD_OPD_IN,0},{"meta",RAD_OPD_IN,0},{"token_ids",RAD_OPD_OUT,0},{"positions",RAD_OPD_OUT,0},{"gather_ids",RAD_OPD_OUT,0},{"ancestors",RAD_OPD_OUT,0},{"slots",RAD_OPD_OUT,0},{"prefix",RAD_OPD_OUT,0}};
const RadOperandSpec bo[]={{"request_ids",RAD_OPD_IN,0},{"meta",RAD_OPD_IN,0},{"score",RAD_OPD_IN,0},{"packed",RAD_OPD_OUT,0}};
const RadOpSchema schemas[]={
 {"mtp_pruned_expand",ep,3,eo,8,"Exact TP2 top-eight candidate-set normalization and global width-eight frontier."},
 {"mtp_pruned_prepare",ep,3,po,8,"Prepare tiled real MTP rows and chronological provisional ancestry."},
 {"mtp_pruned_pack",bp,2,bo,4,"Ancestor-closed top-seven pruning by path score plus beta times depth."}};
int geometry(const RadParam* p,int n,RadMtpPrunedTreeCall& c,int op,bool wide) {
 c.M=wide ? rad_param_getdim(p,n,"M",0) : rad_param_geti(p,n,"M",0);
 if(c.M<1 || c.M>INT_MAX/32) return RAD_E_UNSUPPORTED;
 if(op==2) {
  const RadParam* b=rad_param_find(p,n,"beta");
  if(!b || b->kind!=RAD_P_F64 || !std::isfinite(b->dval) || b->dval>0) return RAD_E_INVAL;
  c.beta=b->dval; return RAD_OK;
 }
 const int64_t r=rad_param_geti(p,n,"round",0),v=rad_param_geti(p,n,"n_vocab",0);
 if(r<(op==1 ? 2 : 1) || r>5 || v<1 || v>INT_MAX) return RAD_E_UNSUPPORTED;
 c.round=static_cast<int>(r); c.vocab=static_cast<int>(v); return RAD_OK;
}
int shape(const RadParam* p,int n,int i,RadOpdDesc* out,int op) {
 if(!out || i<0 || i>=(op==2 ? 4 : 8)) return RAD_E_INVAL;
 RadMtpPrunedTreeCall c{}; int rc=geometry(p,n,c,op,true); if(rc) return rc;
 RadOpdDesc d{}; d.idx_const=-1; d.dtype=RAD_I32; d.rank=1; d.shape[0]=c.M;
 if(op==0) {
  if(i==0) { d.dtype=RAD_F32; d.rank=3; d.shape[0]=c.M*(c.round==1 ? 1 : 8); d.shape[1]=16; d.shape[2]=2; }
  if((i==1 && c.round==1) || (i>=2 && i<=4 && c.round!=1)) d.flags=RAD_OPD_F_ABSENT;
  if(i>=5) { d.rank=2; d.shape[1]=i==5 ? 85 : i==6 ? 41 : 17; if(i==6) d.dtype=RAD_F32; }
 } else if(op==1) {
  if(i==1) { d.rank=2; d.shape[1]=85; }
  if(i>=2) d.shape[0]=8*c.M;
  if(i==5) { d.rank=2; d.shape[1]=4; }
 } else if(i>=1) { d.rank=2; d.shape[1]=i==1 ? 85 : i==2 ? 41 : 17; if(i==2) d.dtype=RAD_F32; }
 *out=d; return RAD_OK;
}
int tensor(const RadTensor& t,int rank,int64_t rows,int64_t cols,uint32_t dtype,uint64_t& bytes) {
 if(!t.data || reinterpret_cast<uintptr_t>(t.data)%4) return RAD_E_INVAL;
 if(t.dtype!=dtype) return RAD_E_DTYPE;
 if(t.rank!=static_cast<uint32_t>(rank) || t.shape[0]!=rows || (rank>1 && t.shape[1]!=cols) || (rank==3 && t.shape[2]!=2)) return RAD_E_SHAPE;
 if(t.stride[rank-1]!=1 || (rank==3 && t.stride[1]!=2)) return RAD_E_STRIDE;
 uint64_t span=1;
 for(int j=rank-1;j>=0;--j) {
  if(t.shape[j]<1 || t.stride[j]<1 || (j<rank-1 && static_cast<uint64_t>(t.stride[j])<span)) return RAD_E_STRIDE;
  const uint64_t extent=static_cast<uint64_t>(t.shape[j]-1), stride=static_cast<uint64_t>(t.stride[j]);
  if(extent && stride>(static_cast<uint64_t>(INT64_MAX)/4-span)/extent) return RAD_E_SHAPE;
  span+=extent*stride;
 }
 bytes=span*4;
 if(reinterpret_cast<uintptr_t>(t.data)>UINTPTR_MAX-bytes) return RAD_E_INVAL;
 return RAD_OK;
}
int args(const RadArgs* a,RadMtpPrunedTreeCall* out,int op) {
 if(!a || !out || !a->t || a->n_t!=(op==2 ? 4 : 8)) return RAD_E_INVAL;
 RadMtpPrunedTreeCall c{}; int rc=geometry(a->p,a->n_p,c,op,false); if(rc) return rc;
 const bool root=op==0 && c.round==1;
 const RadTensor& qt=a->t[0];
 if(qt.rank!=(op==0 ? 3u : 1u) || qt.shape[0]<1) return RAD_E_SHAPE;
 c.Q=qt.shape[0]/(op==0 && !root ? 8 : 1);
 if(c.Q<1 || c.Q>c.M || (op==0 && !root && qt.shape[0]%8)) return RAD_E_SHAPE;
 uint64_t bytes[8]{}; bool writable[8]{};
 for(int i=0;i<a->n_t;++i) {
  bool absent=op==0 && ((i==1 && root) || (i>=2 && i<=4 && !root));
  if(absent) { if(a->t[i].data) return RAD_E_INVAL; continue; }
  int rank=1; int64_t rows=c.Q,cols=0; uint32_t dt=RAD_I32;
  if(op==0) {
   if(i==0) { rank=3; rows=c.Q*(root ? 1 : 8); cols=16; dt=RAD_F32; }
   if(i==2 || i==4) { rows=a->t[i].shape[0]; if(rows<1 || rows>INT_MAX) return RAD_E_SHAPE; }
   if(i>=5) { rank=2; rows=c.M; cols=i==5 ? 85 : i==6 ? 41 : 17; if(i==6) dt=RAD_F32; writable[i]=true; }
  } else if(op==1) {
   if(i==1) { rank=2; rows=c.M; cols=85; }
   if(i>=2) { rows=8*c.Q; writable[i]=true; }
   if(i==5) { rank=2; cols=4; }
  } else if(i>=1) { rank=2; rows=c.M; cols=i==1 ? 85 : i==2 ? 41 : 17; if(i==2) dt=RAD_F32; writable[i]=i==3; }
  if((rc=tensor(a->t[i],rank,rows,cols,dt,bytes[i]))) return rc;
 }
 if(root && a->t[2].shape[0]!=a->t[4].shape[0]) return RAD_E_SHAPE;
 for(int i=0;i<a->n_t;++i) for(int j=i+1;j<a->n_t;++j) if(bytes[i] && bytes[j] && (writable[i] || writable[j])) {
  const uintptr_t x=reinterpret_cast<uintptr_t>(a->t[i].data),y=reinterpret_cast<uintptr_t>(a->t[j].data);
  if(x<y+bytes[j] && y<x+bytes[i]) return RAD_E_INVAL;
 }
 if(op==0) {
  c.gathered=static_cast<const uint32_t*>(a->t[0].data); c.gathered_ld=a->t[0].stride[0];
  c.request_ids=static_cast<const int32_t*>(a->t[1].data); c.request_stride=a->t[1].stride[0];
  c.tokens=static_cast<const int32_t*>(a->t[2].data); c.token_stride=a->t[2].stride[0]; c.token_count=root ? a->t[2].shape[0] : 0;
  c.anchor_rows=static_cast<const int32_t*>(a->t[3].data); c.anchor_stride=a->t[3].stride[0];
  c.positions=static_cast<const int32_t*>(a->t[4].data); c.position_stride=a->t[4].stride[0];
 } else { c.request_ids=static_cast<const int32_t*>(a->t[0].data); c.request_stride=a->t[0].stride[0]; }
 int mi=op==0 ? 5 : 1;
 c.meta=static_cast<int32_t*>(a->t[mi].data); c.meta_ld=a->t[mi].stride[0];
 if(op!=1) {
  int si=op==0 ? 6 : 2,pi=op==0 ? 7 : 3;
  c.score=static_cast<float*>(a->t[si].data); c.score_ld=a->t[si].stride[0]; c.packed=static_cast<int32_t*>(a->t[pi].data); c.packed_ld=a->t[pi].stride[0];
 } else {
  c.token_ids=static_cast<int32_t*>(a->t[2].data); c.token_ids_stride=a->t[2].stride[0];
  c.out_positions=static_cast<int32_t*>(a->t[3].data); c.out_positions_stride=a->t[3].stride[0];
  c.gather_ids=static_cast<int32_t*>(a->t[4].data); c.gather_ids_stride=a->t[4].stride[0];
  c.ancestors=static_cast<int32_t*>(a->t[5].data); c.ancestors_ld=a->t[5].stride[0];
  c.slots=static_cast<int32_t*>(a->t[6].data); c.slots_stride=a->t[6].stride[0];
  c.prefix=static_cast<int32_t*>(a->t[7].data); c.prefix_stride=a->t[7].stride[0];
 }
 *out=c; return RAD_OK;
}
RadKernelInfo row(int op,RadLaunchFn fn) {
 RadKernelInfo r{}; static const char* names[]={"mine_mtp_pruned_expand","mine_mtp_pruned_prepare","mine_mtp_pruned_pack"};
 r.name=names[op]; r.op=schemas[op].op; r.family="draft"; r.computes=schemas[op].doc;
 r.shape="W8,K8,D5,B7; tiled request IDs; padded outer rows"; r.dtypes="f32 scores, i32 metadata"; r.domain=RAD_DOMAIN_DEVICE; r.priority=100; r.launch=fn;
 r.opd_shape=op==0 ? rad_mtp_pruned_expand_shape : op==1 ? rad_mtp_pruned_prepare_shape : rad_mtp_pruned_pack_shape;
 return r;
}
}
int rad_mtp_pruned_tree_schema_count() { return 3; }
const RadOpSchema* rad_mtp_pruned_tree_schema_at(int i) { return i>=0 && i<3 ? schemas+i : nullptr; }
#define DEFINE(op, index, Call) \
int rad_mtp_pruned_##op##_shape(const RadParam* p,int n,int i,RadOpdDesc* out) { return shape(p,n,i,out,index); } \
int rad_mtp_pruned_##op##_args(const RadArgs* a,Call* out) { return args(a,out,index); } \
RadKernelInfo rad_mtp_pruned_##op##_row(RadLaunchFn fn) { return row(index,fn); }
DEFINE(expand,0,RadMtpPrunedExpandCall)
DEFINE(prepare,1,RadMtpPrunedPrepareCall)
DEFINE(pack,2,RadMtpPrunedPackCall)
#undef DEFINE
