#include "mtp_pruned_tree.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {
constexpr float neginf=-std::numeric_limits<float>::infinity();
void reset(int32_t* p) { std::fill(p,p+17,-1); p[0]=0; }
float decode(uint32_t bits) { float x; std::memcpy(&x,&bits,sizeof x); return x; }
bool valid_tree(const int32_t* m,const float* scores,int levels,int vocab) {
 if(m[0]!=1 || m[1]<0 || m[1]>INT_MAX-5 || m[2]!=m[3] || m[3]<0 || (vocab>0 && m[3]>=vocab) || m[44]!=-1) return false;
 if(scores && scores[0]!=0) return false;
 for(int depth=1;depth<=levels;++depth) {
  const int first=1+(depth-1)*8;
  for(int n=first;n<first+8;++n) {
   const int parent=m[44+n];
   if(m[3+n]<0 || (vocab>0 && m[3+n]>=vocab)) return false;
   if(depth==1 ? parent!=0 : parent<first-8 || parent>=first) return false;
   if(scores && (std::isnan(scores[n]) || scores[n]>0 || scores[n]>scores[parent])) return false;
   for(int j=first;j<n;++j) if(m[44+j]==parent && m[3+j]==m[3+n]) return false;
  }
 }
 return true;
}
int global_id(const RadMtpPrunedTreeCall& c,int64_t q,bool& repeated) {
 const int id=c.request_ids[q*c.request_stride]; repeated=false;
 if(id<0 || id>=c.M) return -1;
 for(int64_t k=0;k<c.Q;++k) if(k!=q && c.request_ids[k*c.request_stride]==id) {
  if(k<q) return -1;
  repeated=true;
 }
 return id;
}
struct Offer { int token; float value; };
struct Edge { int token,parent,rank; float score; };
}
int rad_ref_mtp_pruned_expand(const RadArgs* a,RadStream) {
 RadMtpPrunedExpandCall c{}; int rc=rad_mtp_pruned_expand_args(a,&c); if(rc) return rc;
 for(int64_t q=0;q<c.Q;++q) {
  bool repeated=false; const int id=c.round==1 ? static_cast<int>(q) : global_id(c,q,repeated); if(id<0) continue;
  int32_t* m=c.meta+id*c.meta_ld; float* scores=c.score+id*c.score_ld; int32_t* packed=c.packed+id*c.packed_ld;
  if(c.round==1) {
   std::fill(m,m+85,-1); std::fill(scores,scores+41,neginf); reset(packed); m[0]=0;
   const int anchor=c.anchor_rows[q*c.anchor_stride];
   if(anchor<0 || anchor>=c.token_count) continue;
   const int token=c.tokens[static_cast<int64_t>(anchor)*c.token_stride],position=c.positions[static_cast<int64_t>(anchor)*c.position_stride];
   if(token<0 || token>=c.vocab || position<0 || position>INT_MAX-5) continue;
   m[1]=position; m[2]=token; m[3]=token; m[44]=-1; scores[0]=0; packed[1]=token;
  } else if(repeated || !valid_tree(m,scores,c.round-1,c.vocab)) { m[0]=0; reset(packed); continue; }
  std::vector<Edge> frontier;
  bool valid=true; const int parent_count=c.round==1 ? 1 : 8;
  for(int j=0;j<parent_count && valid;++j) {
   std::vector<Offer> offers;
   const uint32_t* row=c.gathered+(q*parent_count+j)*c.gathered_ld;
   for(int k=0;k<16;++k) {
    const int token=static_cast<int32_t>(row[2*k]); const float value=decode(row[2*k+1]);
    if(token<0 || token>=c.vocab || !std::isfinite(value)) valid=false;
    if(std::any_of(offers.begin(),offers.end(),[&](const Offer& o){return o.token==token;})) valid=false;
    offers.push_back({token,value});
   }
   if(!valid) break;
   std::sort(offers.begin(),offers.end(),[](const Offer& x,const Offer& y){return x.value!=y.value ? x.value>y.value : x.token<y.token;});
   offers.resize(8);
   const double maximum=offers.front().value; double total=0;
   for(const Offer& o:offers) total+=std::exp(static_cast<double>(o.value)-maximum);
   const double log_total=std::log(total); const int parent=c.round==1 ? 0 : 1+(c.round-2)*8+j;
   for(int rank=0;rank<8;++rank) frontier.push_back({offers[rank].token,parent,rank,static_cast<float>(static_cast<double>(scores[parent])+(static_cast<double>(offers[rank].value)-maximum-log_total))});
  }
  if(!valid) { m[0]=0; reset(packed); continue; }
  std::sort(frontier.begin(),frontier.end(),[](const Edge& x,const Edge& y) {
   if(x.score!=y.score) return x.score>y.score;
   return x.parent!=y.parent ? x.parent<y.parent : x.rank<y.rank;
  });
  for(int k=0;k<8;++k) {
   const int n=1+(c.round-1)*8+k;
   m[3+n]=frontier[k].token; m[44+n]=frontier[k].parent; scores[n]=frontier[k].score;
  }
  m[0]=1;
 }
 return RAD_OK;
}
int rad_ref_mtp_pruned_prepare(const RadArgs* a,RadStream) {
 RadMtpPrunedPrepareCall c{}; int rc=rad_mtp_pruned_prepare_args(a,&c); if(rc) return rc;
 for(int64_t q=0;q<c.Q;++q) {
  bool repeated=false; const int id=global_id(c,q,repeated),depth=c.round-1;
  const int32_t* m=id<0 ? nullptr : c.meta+id*c.meta_ld;
  const bool valid=m && !repeated && valid_tree(m,nullptr,depth,c.vocab);
  for(int j=0;j<8;++j) {
   const int64_t row=q*8+j;
   c.token_ids[row*c.token_ids_stride]=0; c.out_positions[row*c.out_positions_stride]=0;
   c.gather_ids[row*c.gather_ids_stride]=0; c.slots[row*c.slots_stride]=-1; c.prefix[row*c.prefix_stride]=0;
   int32_t* ancestors=c.ancestors+row*c.ancestors_ld; std::fill(ancestors,ancestors+4,-1);
   if(!valid) continue;
   const int node=1+(depth-1)*8+j;
   std::vector<int> path;
   for(int n=node;n!=0;n=m[44+n]) path.push_back(n);
   std::reverse(path.begin(),path.end());
   for(int k=0;k<depth;++k) ancestors[k]=static_cast<int>(q*32)+path[k]-1;
   c.token_ids[row*c.token_ids_stride]=m[3+node]; c.out_positions[row*c.out_positions_stride]=m[1]+depth;
   c.slots[row*c.slots_stride]=static_cast<int>(q*32)+node-1; c.prefix[row*c.prefix_stride]=m[1]+1;
   c.gather_ids[row*c.gather_ids_stride]=depth==1 ? id : static_cast<int>(q*8)+(m[44+node]-(1+(depth-2)*8));
  }
 }
 return RAD_OK;
}
int rad_ref_mtp_pruned_pack(const RadArgs* a,RadStream) {
 RadMtpPrunedPackCall c{}; int rc=rad_mtp_pruned_pack_args(a,&c); if(rc) return rc;
 for(int64_t q=0;q<c.Q;++q) {
  bool repeated=false; const int id=global_id(c,q,repeated); if(id<0) continue;
  const int32_t* m=c.meta+id*c.meta_ld; const float* scores=c.score+id*c.score_ld; int32_t* out=c.packed+id*c.packed_ld;
  reset(out); if(repeated || !valid_tree(m,scores,5,0)) continue;
  std::vector<int> order;
  for(int n=1;n<=40;++n) order.push_back(n);
  std::sort(order.begin(),order.end(),[&](int x,int y) {
   const double sx=static_cast<double>(scores[x])+c.beta*((x-1)/8+1),sy=static_cast<double>(scores[y])+c.beta*((y-1)/8+1);
   return sx!=sy ? sx>sy : x<y;
  });
  order.resize(7); std::sort(order.begin(),order.end());
  bool closed=true;
  for(int n:order) if(m[44+n]!=0 && !std::binary_search(order.begin(),order.end(),m[44+n])) closed=false;
  if(!closed) continue;
  out[1]=m[3];
  for(int k=0;k<7;++k) {
   const int n=order[k],parent=m[44+n]; out[2+k]=m[3+n];
   out[10+k]=parent==0 ? 0 : static_cast<int>(std::lower_bound(order.begin(),order.end(),parent)-order.begin())+1;
  }
  out[0]=8;
 }
 return RAD_OK;
}
int rad_ref_mtp_pruned_tree_count() { return 3; }
const RadKernelInfo* rad_ref_mtp_pruned_tree_at(int i) {
 struct Rows { RadKernelInfo data[3]; };
 static const Rows rows=[] {
  Rows r{{rad_mtp_pruned_expand_row(rad_ref_mtp_pruned_expand),rad_mtp_pruned_prepare_row(rad_ref_mtp_pruned_prepare),rad_mtp_pruned_pack_row(rad_ref_mtp_pruned_pack)}};
  const char* names[]={"ref_mtp_pruned_expand","ref_mtp_pruned_prepare","ref_mtp_pruned_pack"};
  for(int k=0;k<3;++k) { r.data[k].name=names[k]; r.data[k].domain=RAD_DOMAIN_HOST; r.data[k].priority=0; }
  return r;
 }();
 return i>=0 && i<3 ? rows.data+i : nullptr;
}
