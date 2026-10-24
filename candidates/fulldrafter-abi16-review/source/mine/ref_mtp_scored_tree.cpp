#include "mtp_scored_tree.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace {
constexpr double neginf=-std::numeric_limits<double>::infinity();
float from_bits(uint32_t bits) { float x; std::memcpy(&x,&bits,sizeof x); return x; }
struct Node { int token, depth, candidate; double score; };
}
int rad_ref_mtp_hybrid_lse(const RadArgs* a,RadStream) {
    RadMtpHybridLseCall c{}; const int rc=rad_mtp_hybrid_lse_args(a,&c); if(rc) return rc;
    for(int64_t s=0;s<c.M;++s) {
        auto value=[&](int64_t j) {
            double z=from_bits(static_cast<uint32_t>(c.coarse[s*c.coarse_ld+j])<<16);
            for(int k=0;k<8;++k) if(c.idx[s*c.idx_ld+k]==static_cast<int64_t>(c.vocab_off)+j) { z=c.val[s*c.val_ld+k]; break; }
            return z;
        };
        bool valid=true;
        double maximum=neginf;
        const int64_t count=std::max<int64_t>(0,std::min<int64_t>(c.N,static_cast<int64_t>(c.vocab)-c.vocab_off));
        for(int64_t j=0;j<count;++j) {
            const double z=value(j);
            valid=valid && !std::isnan(z) && z!=std::numeric_limits<double>::infinity();
            maximum=std::max(maximum,z);
        }
        double sum=0;
        if(std::isfinite(maximum)) for(int64_t j=0;j<count;++j) sum+=std::exp(value(j)-maximum);
        float* out=c.stats+s*c.stats_ld;
        out[0]=valid ? static_cast<float>(maximum) : std::numeric_limits<float>::quiet_NaN();
        out[1]=valid ? static_cast<float>(sum) : std::numeric_limits<float>::quiet_NaN(); out[2]=0; out[3]=0;
    }
    return RAD_OK;
}
int rad_ref_mtp_scored_merge_tree_accum(const RadArgs* a,RadStream) {
    RadMtpScoredMergeTreeAccumCall c{}; const int rc=rad_mtp_scored_merge_tree_accum_args(a,&c); if(rc) return rc;
    for(int64_t s=0;s<c.M;++s) {
        std::vector<uint64_t> keys;
        for(int j=0;j<16;++j) {
            const uint32_t* pair=c.gathered+s*c.gathered_ld+j*2;
            if(static_cast<int32_t>(pair[0])<0 || pair[0]>=static_cast<uint32_t>(c.vocab)) continue;
            const uint32_t image=pair[1]&0x80000000u ? ~pair[1] : pair[1]|0x80000000u;
            keys.push_back((static_cast<uint64_t>(image)<<32)|(0xffffffffu-pair[0]));
        }
        std::sort(keys.begin(),keys.end(),std::greater<uint64_t>());
        keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
        float z[8]; int merged[8];
        for(int k=0;k<8;++k) {
            const uint64_t key=static_cast<size_t>(k)<keys.size() ? keys[k] : 0;
            const uint32_t image=static_cast<uint32_t>(key>>32);
            const uint32_t bits=key ? (image&0x80000000u ? image&0x7fffffffu : ~image) : 0xff800000u;
            merged[k]=key ? static_cast<int32_t>(0xffffffffu-static_cast<uint32_t>(key)) : -1;
            c.idx[s*c.idx_ld+k]=merged[k]; c.val[s*c.val_ld+k]=bits; z[k]=from_bits(bits);
        }
        int32_t* out=c.packed+s*c.out_ld;
        int root=-1;
        if(c.round==1) { const int ix=c.ids[s*c.id_stride]; if(ix>=0 && ix<c.token_count) root=c.tokens[static_cast<int64_t>(ix)*c.token_stride]; }
        else if(out[0]>=1 && out[0]<=c.capacity) root=out[1];
        bool valid=root>=0 && root<c.vocab;
        if(c.policy==0) {
            if(c.round==1) for(int r=0;r<c.max_rounds;++r) for(int k=0;k<8;++k)
                c.history[r*c.hist_round_stride+s*c.hist_row_stride+k]=-1;
            for(int k=0;k<8;++k)
                c.history[(c.round-1)*c.hist_round_stride+s*c.hist_row_stride+k]=merged[k];
        } else {
            const float* st=c.stats+s*c.stats_ld;
            const double maximum=std::max(st[0],st[4]); double sum=0;
            for(int rank=0;rank<2;++rank) {
                const double m=st[rank*4], q=st[rank*4+1];
                valid=valid && ((std::isfinite(m) && std::isfinite(q) && q>0) || (m==neginf && q==0));
                if(q>0) sum+=q*std::exp(m-maximum);
            }
            valid=valid && std::isfinite(maximum) && std::isfinite(sum) && sum>0;
            if(c.round==1) for(int r=0;r<c.max_rounds;++r) for(int k=0;k<8;++k) {
                c.history[r*c.hist_round_stride+s*c.hist_row_stride+k]=-1;
                c.log_history[r*c.log_round_stride+s*c.log_row_stride+k]=static_cast<float>(neginf);
            }
            double base=0;
            if(c.round>1) {
                const int prev=c.history[(c.round-2)*c.hist_round_stride+s*c.hist_row_stride];
                base=c.log_history[(c.round-2)*c.log_round_stride+s*c.log_row_stride];
                if(prev<0 || prev>=c.vocab || std::isnan(base) || base>0) valid=false;
            }
            for(int k=0;k<8;++k) {
                double l=neginf;
                if(merged[k]>=0 && merged[k]<c.vocab) {
                    if(std::isnan(z[k]) || z[k]==std::numeric_limits<float>::infinity()) valid=false;
                    else if(valid) { l=(static_cast<double>(z[k])-maximum)-std::log(sum); if(l>1e-5) valid=false; else l=std::min(l,0.0)+base; }
                }
                c.history[(c.round-1)*c.hist_round_stride+s*c.hist_row_stride+k]=merged[k];
                c.log_history[(c.round-1)*c.log_round_stride+s*c.log_row_stride+k]=static_cast<float>(l);
            }
        }
        std::vector<Node> nodes;
        for(int r=0;r<c.round;++r) for(int k=0;k<8;++k) {
            const int id=c.history[r*c.hist_round_stride+s*c.hist_row_stride+k];
            const double l=c.policy==0 ? 0.0 : c.log_history[r*c.log_round_stride+s*c.log_row_stride+k];
            if(id>=0 && id<c.vocab && (c.policy==0 || (!std::isnan(l) && l<=0))) nodes.push_back({id,r,r*8+k,l});
        }
        std::vector<Node> chosen; std::vector<int> parents;
        if(valid) {
            auto append=[&](const Node& node,int parent) {
                for(size_t i=0;i<chosen.size();++i) if(parents[i]==parent && chosen[i].token==node.token) return false;
                chosen.push_back(node); parents.push_back(parent); return true;
            };
            if(c.policy==0) {
                int spine=0;
                for(int r=0;r<c.round && chosen.size()<static_cast<size_t>(c.capacity-1);++r) {
                    auto it=std::find_if(nodes.begin(),nodes.end(),[&](const Node& x){ return x.candidate==r*8; });
                    if(it==nodes.end()) break;
                    append(*it,r); ++spine;
                }
                for(int r=0;r<spine;++r) for(const auto& node:nodes)
                    if(node.depth==r && node.candidate%8 && chosen.size()<static_cast<size_t>(c.capacity-1)) append(node,r);
            } else {
                std::sort(nodes.begin(),nodes.end(),[](const Node& x,const Node& y) {
                    if(x.score!=y.score) return x.score>y.score;
                    if(x.depth!=y.depth) return x.depth<y.depth;
                    if(x.token!=y.token) return x.token<y.token;
                    return x.candidate<y.candidate;
                });
                std::vector<bool> consumed(nodes.size());
                while(chosen.size()<static_cast<size_t>(c.capacity-1)) {
                    bool found=false;
                    for(size_t j=0;j<nodes.size();++j) {
                        if(consumed[j]) continue;
                        const Node node=nodes[j]; int parent=node.depth==0 ? 0 : -1;
                        if(node.depth>0) {
                            const int greedy=c.history[(node.depth-1)*c.hist_round_stride+s*c.hist_row_stride];
                            for(size_t i=0;i<chosen.size();++i) if(chosen[i].depth==node.depth-1 && chosen[i].token==greedy) parent=static_cast<int>(i)+1;
                        }
                        if(parent<0) continue;
                        consumed[j]=true;
                        if(append(node,parent)) { found=true; break; }
                    }
                    if(!found) break;
                }
                if(c.policy==2) {
                    size_t keep=0; double mass=1, best=-std::log(c.costs[0]);
                    for(size_t i=0;i<chosen.size();++i) {
                        mass+=std::exp(chosen[i].score);
                        const double objective=std::log(mass)-std::log(c.costs[i+1]);
                        if(objective>best) { best=objective; keep=i+1; }
                    }
                    chosen.resize(keep); parents.resize(keep);
                }
            }
        }
        out[0]=valid ? static_cast<int>(chosen.size())+1 : 0;
        for(int k=0;k<c.capacity;++k) { out[1+k]=-1; out[1+c.capacity+k]=-1; }
        if(valid) {
            out[1]=root;
            for(size_t k=0;k<chosen.size();++k) { out[2+k]=chosen[k].token; out[2+c.capacity+k]=parents[k]; }
        }
    }
    return RAD_OK;
}
int rad_ref_mtp_scored_tree_count() { return 2; }
const RadKernelInfo* rad_ref_mtp_scored_tree_at(int i) {
    struct Rows { RadKernelInfo data[2]; };
    static const Rows rows=[] {
        Rows r{{rad_mtp_hybrid_lse_row(rad_ref_mtp_hybrid_lse),rad_mtp_scored_merge_tree_accum_row(rad_ref_mtp_scored_merge_tree_accum)}};
        r.data[0].name="ref_mtp_hybrid_lse"; r.data[1].name="ref_mtp_scored_merge_tree_accum";
        for(auto& row:r.data) { row.domain=RAD_DOMAIN_HOST; row.priority=0; }
        return r;
    }();
    return i>=0 && i<2 ? rows.data+i : nullptr;
}
