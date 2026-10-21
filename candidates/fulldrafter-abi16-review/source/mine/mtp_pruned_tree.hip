#include <hip/hip_runtime.h>
#include "mtp_pruned_tree.h"
#include <climits>
#include <cmath>

namespace {
__device__ bool metadata(const int32_t* m,const float* s,int levels,int vocab) {
 if(m[0]!=1 || m[1]<0 || m[1]>INT_MAX-5 || m[2]!=m[3] || m[3]<0 || (vocab && m[3]>=vocab) || m[44]!=-1) return false;
 if(s && s[0]!=0) return false;
 for(int n=1;n<=levels*8;++n) {
  const int d=(n-1)/8+1,p=m[44+n],lo=d==1 ? 0 : 1+(d-2)*8,hi=d==1 ? 1 : lo+8;
  if(m[3+n]<0 || (vocab && m[3+n]>=vocab) || p<lo || p>=hi) return false;
  if(s && (isnan(s[n]) || s[n]>0 || s[n]>s[p])) return false;
  for(int j=1+(d-1)*8;j<n;++j) if(m[44+j]==p && m[3+j]==m[3+n]) return false;
 }
 return true;
}
__device__ int request(RadMtpPrunedTreeCall c,int64_t q,bool& duplicate) {
 const int id=c.request_ids[q*c.request_stride]; duplicate=false;
 if(id<0 || id>=c.M) return -1;
 for(int64_t k=0;k<q;++k) if(c.request_ids[k*c.request_stride]==id) return -1;
 for(int64_t k=q+1;k<c.Q;++k) if(c.request_ids[k*c.request_stride]==id) duplicate=true;
 return id;
}
__device__ void clear(int32_t* out) { out[0]=0; for(int i=1;i<17;++i) out[i]=-1; }
__global__ void expand(RadMtpPrunedTreeCall c) {
 const int64_t q=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x; if(q>=c.Q) return;
 bool duplicate=false; const int id=c.round==1 ? static_cast<int>(q) : request(c,q,duplicate); if(id<0) return;
 int32_t* m=c.meta+id*c.meta_ld; float* s=c.score+id*c.score_ld; int32_t* out=c.packed+id*c.packed_ld;
 bool valid=!duplicate;
 if(c.round==1) {
  for(int i=0;i<85;++i) m[i]=-1;
  for(int i=0;i<41;++i) s[i]=-INFINITY;
  clear(out); m[0]=0;
  const int a=c.anchor_rows[q*c.anchor_stride];
  if(a<0 || a>=c.token_count) return;
  const int token=c.tokens[static_cast<int64_t>(a)*c.token_stride],p=c.positions[static_cast<int64_t>(a)*c.position_stride];
  if(token<0 || token>=c.vocab || p<0 || p>INT_MAX-5) return;
  m[1]=p; m[2]=token; m[3]=token; m[44]=-1; s[0]=0; out[1]=token;
 } else valid=valid && metadata(m,s,c.round-1,c.vocab);
 if(!valid) { m[0]=0; clear(out); return; }
 float edges[64]; int edge_token[64],edge_parent[64];
 const int parents=c.round==1 ? 1 : 8;
 for(int j=0;j<parents;++j) {
  const uint32_t* src=c.gathered+(q*parents+j)*c.gathered_ld;
  int order[16]; float z[16];
  for(int k=0;k<16;++k) {
   order[k]=k; z[k]=__uint_as_float(src[2*k+1]); const int token=static_cast<int32_t>(src[2*k]);
   if(token<0 || token>=c.vocab || !isfinite(z[k])) valid=false;
   for(int h=0;h<k;++h) if(src[2*h]==src[2*k]) valid=false;
  }
  if(!valid) { m[0]=0; clear(out); return; }
  for(int k=1;k<16;++k) {
   const int x=order[k]; int h=k;
   while(h>0 && (z[x]>z[order[h-1]] || (z[x]==z[order[h-1]] && src[2*x]<src[2*order[h-1]]))) { order[h]=order[h-1]; --h; }
   order[h]=x;
  }
  const double mx=z[order[0]]; double sum=0;
  for(int k=0;k<8;++k) sum+=exp(static_cast<double>(z[order[k]])-mx);
  const double lsum=log(sum); const int parent=c.round==1 ? 0 : 1+(c.round-2)*8+j;
  for(int k=0;k<8;++k) {
   const int e=j*8+k; edge_token[e]=static_cast<int32_t>(src[2*order[k]]); edge_parent[e]=parent;
   edges[e]=static_cast<float>(static_cast<double>(s[parent])+(static_cast<double>(z[order[k]])-mx-lsum));
  }
 }
 bool used[64]{};
 for(int k=0;k<8;++k) {
  int best=-1;
  for(int e=0;e<parents*8;++e) if(!used[e] && (best<0 || edges[e]>edges[best] || (edges[e]==edges[best] && (edge_parent[e]<edge_parent[best] || (edge_parent[e]==edge_parent[best] && e%8<best%8))))) best=e;
  used[best]=true; const int n=1+(c.round-1)*8+k;
  m[3+n]=edge_token[best]; m[44+n]=edge_parent[best]; s[n]=edges[best];
 }
 m[0]=1;
}
__global__ void prepare(RadMtpPrunedTreeCall c) {
 const int64_t q=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x; if(q>=c.Q) return;
 bool duplicate=false; const int id=request(c,q,duplicate),d=c.round-1;
 const int32_t* m=id<0 ? nullptr : c.meta+id*c.meta_ld;
 const bool valid=!duplicate && m && metadata(m,nullptr,d,c.vocab);
 for(int j=0;j<8;++j) {
  const int64_t r=q*8+j; int32_t* a=c.ancestors+r*c.ancestors_ld;
  for(int k=0;k<4;++k) a[k]=-1;
  c.token_ids[r*c.token_ids_stride]=0; c.out_positions[r*c.out_positions_stride]=0;
  c.gather_ids[r*c.gather_ids_stride]=0; c.slots[r*c.slots_stride]=-1; c.prefix[r*c.prefix_stride]=0;
  if(!valid) continue;
  const int n=1+(d-1)*8+j;
  c.token_ids[r*c.token_ids_stride]=m[3+n]; c.out_positions[r*c.out_positions_stride]=m[1]+d;
  c.gather_ids[r*c.gather_ids_stride]=d==1 ? id : static_cast<int>(q*8)+m[44+n]-(1+(d-2)*8);
  c.slots[r*c.slots_stride]=static_cast<int>(q*32)+n-1; c.prefix[r*c.prefix_stride]=m[1]+1;
  int node=n; for(int k=d-1;k>=0;--k) { a[k]=static_cast<int>(q*32)+node-1; node=m[44+node]; }
 }
}
__global__ void pack(RadMtpPrunedTreeCall c) {
 const int64_t q=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x; if(q>=c.Q) return;
 bool duplicate=false; const int id=request(c,q,duplicate); if(id<0) return;
 const int32_t* m=c.meta+id*c.meta_ld; const float* s=c.score+id*c.score_ld; int32_t* out=c.packed+id*c.packed_ld;
 clear(out); if(duplicate || !metadata(m,s,5,0)) return;
 bool selected[41]{}; selected[0]=true;
 for(int k=0;k<7;++k) {
  int best=-1;
  for(int n=1;n<=40;++n) if(!selected[n]) {
   const double v=static_cast<double>(s[n])+c.beta*((n-1)/8+1);
   const double b=best<0 ? -INFINITY : static_cast<double>(s[best])+c.beta*((best-1)/8+1);
   if(best<0 || v>b || (v==b && n<best)) best=n;
  }
  selected[best]=true;
 }
 for(int n=1;n<=40;++n) if(selected[n] && !selected[m[44+n]]) return;
 int map[41]; for(int n=0;n<=40;++n) map[n]=-1;
 out[1]=m[3]; map[0]=0; int count=1;
 for(int n=1;n<=40;++n) if(selected[n]) { map[n]=count; out[1+count]=m[3+n]; out[9+count]=map[m[44+n]]; ++count; }
 out[0]=8;
}
}
#define LAUNCH(op, kernel) \
int rad_mine_mtp_pruned_##op##_launch(const RadArgs* a,RadStream stream) { \
 RadMtpPrunedTreeCall c{}; const int rc=rad_mtp_pruned_##op##_args(a,&c); if(rc) return rc; \
 void* argv[]={&c}; return hipLaunchKernel((const void*)kernel,dim3(static_cast<unsigned>((c.Q+63)/64)),dim3(64),argv,0,reinterpret_cast<hipStream_t>(stream))==hipSuccess ? RAD_OK : RAD_E_DEVICE; }
LAUNCH(expand,expand)
LAUNCH(prepare,prepare)
LAUNCH(pack,pack)
#undef LAUNCH
int rad_mine_mtp_pruned_tree_count() { return 3; }
const RadKernelInfo* rad_mine_mtp_pruned_tree_at(int i) {
 static const RadKernelInfo rows[]={rad_mtp_pruned_expand_row(rad_mine_mtp_pruned_expand_launch),rad_mtp_pruned_prepare_row(rad_mine_mtp_pruned_prepare_launch),rad_mtp_pruned_pack_row(rad_mine_mtp_pruned_pack_launch)};
 return i>=0 && i<3 ? rows+i : nullptr;
}
