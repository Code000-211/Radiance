#ifndef RAD_MINE_SPECHUB_H
#define RAD_MINE_SPECHUB_H
#include "rad_plugin.h"
#include "rad_spechub.h"
#include "rad_sample.h"
#include <cmath>
#include <cstdint>
#if defined(__HIPCC__)
#define SH_HD __host__ __device__
#else
#define SH_HD
#endif

/* Public pure seam: joint rows (x,a) and (a,x), with matched nonhub and hub
 * columns. Unmatched row mass transports through the FULL target residual. */
struct RadSpechubTransport {
    int count;
    int32_t ids[8];
    double joint[2][8], nonhub[2][8], hub[2][8], used[8];
};
SH_HD inline double rad_spechub_min(double a,double b) { return a < b ? a : b; }
SH_HD inline double rad_spechub_positive(double a) { return a > 0 ? a : 0; }
SH_HD inline bool rad_spechub_finite(double x) { return x == x && x <= 1.7976931348623157e308 && x >= -1.7976931348623157e308; }
SH_HD inline double rad_spechub_uniform(uint64_t seed,uint64_t pos,uint64_t key) {
    uint64_t z = (seed ^ key)*0x9e3779b97f4a7c15ull + pos + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z>>30))*0xbf58476d1ce4e5b9ull;
    z = (z ^ (z>>27))*0x94d049bb133111ebull;
    z ^= z>>31;
    return static_cast<double>(z>>11)*0x1p-53;
}
SH_HD inline bool rad_spechub_state_valid(const RadSpechubState& s,int vocab) {
    if (s.status != 1 || s.root < 0 || s.root >= vocab || s.count < 1 || s.count > 8 || s.reserved) return false;
    double sum=0;
    bool first=false, second=false;
    for (int i=0;i<s.count;++i) {
        if (s.ids[i]<0 || s.ids[i]>=vocab || !rad_spechub_finite(s.q[i]) || s.q[i]<=0 || s.q[i]>s.q[0]) return false;
        for (int j=0;j<i;++j) if (s.ids[i]==s.ids[j]) return false;
        sum+=s.q[i]; first |= s.ids[i]==s.pair[0]; second |= s.ids[i]==s.pair[1];
    }
    if (::fabs(sum-1)>1e-12 || !first || !second) return false;
    if (s.count==1) return s.pair[0]==s.ids[0] && s.pair[1]==s.ids[0];
    return s.pair[0]!=s.pair[1] && (s.pair[0]==s.ids[0] || s.pair[1]==s.ids[0]);
}
SH_HD inline bool rad_spechub_transport(const RadSpechubState& s,const double* p,RadSpechubTransport* out) {
    RadSpechubTransport t{}; t.count=s.count;
    if (!out || !p || !rad_spechub_state_valid(s,INT32_MAX)) return false;
    double target_mass=0;
    for (int i=0;i<s.count;++i) {
        t.ids[i]=s.ids[i];
        if (!rad_spechub_finite(p[i]) || p[i]<0 || p[i]>1) return false;
        target_mass+=p[i];
    }
    if (target_mass>1+1e-12) return false;
    if (s.count==1) { t.joint[1][0]=1; t.hub[1][0]=p[0]; t.used[0]=p[0]; *out=t; return true; }
    double tail=0, capacity[2]={0,0};
    for (int i=1;i<s.count;++i) tail+=s.q[i];
    if (!(tail>0) || !rad_spechub_finite(tail)) return false;
    for (int i=1;i<s.count;++i) {
        t.joint[0][i]=s.q[i]; t.joint[1][i]=s.q[0]*(s.q[i]/tail);
        if (!(t.joint[1][i]>0)) return false;
        t.nonhub[0][i]=rad_spechub_min(p[i],t.joint[0][i]);
        t.nonhub[1][i]=rad_spechub_min(rad_spechub_positive(p[i]-t.nonhub[0][i]),t.joint[1][i]);
        t.used[i]=t.nonhub[0][i]+t.nonhub[1][i];
        for (int g=0;g<2;++g) capacity[g]+=rad_spechub_positive(t.joint[g][i]-t.nonhub[g][i]);
    }
    double remaining=p[0];
    // Appendix B: exhaust (a,x) hub capacity BEFORE (x,a), not reversed.
    for (int g=1;g>=0;--g) {
        double assigned=rad_spechub_min(remaining,capacity[g]);
        double fraction=capacity[g]>0 ? assigned/capacity[g] : 0;
        for (int i=1;i<s.count;++i) t.hub[g][i]=rad_spechub_positive(t.joint[g][i]-t.nonhub[g][i])*fraction;
        t.used[0]+=assigned; remaining=rad_spechub_positive(remaining-assigned);
    }
    if (remaining>1e-12) return false;
    *out=t; return true;
}
SH_HD inline double rad_spechub_residual(const RadSpechubTransport& t,int id,double p) {
    for (int i=0;i<t.count;++i) if (t.ids[i]==id) return rad_spechub_positive(p-t.used[i]);
    return p;
}
/* Returns -1 on malformed/zero-probability pair, -2 for full-target residual. */
SH_HD inline int rad_spechub_match(const RadSpechubTransport& t,int first,int second,double u) {
    if (!(u>=0 && u<1)) return -1;
    int g=-1,k=-1;
    if (t.count==1 && first==t.ids[0] && second==first) { g=1;k=0; }
    for (int i=1;i<t.count;++i) {
        if (first==t.ids[i] && second==t.ids[0]) {g=0;k=i;}
        if (first==t.ids[0] && second==t.ids[i]) {g=1;k=i;}
    }
    if (k<0 || !(t.joint[g][k]>0)) return -1;
    double v=u*t.joint[g][k];
    if (v<t.nonhub[g][k]) return t.ids[k];
    if (v<t.nonhub[g][k]+t.hub[g][k]) return t.ids[0];
    return -2;
}
struct RadSpechubProposeCall {
    const uint32_t* gathered; const uint32_t* draw;
    const int32_t* tokens; const int32_t* draft_ids;
    uint32_t* state; int32_t* packed; int32_t* idx; float* val;
    int64_t M,token_count,ld[8]; int vocab;
};
struct RadSpechubVerifyCall {
    const int32_t* idx; const float* val; const uint32_t* params; const uint32_t* state; int32_t* token;
    int64_t M,n_cand,ld[5]; int vocab,transport;
};

SH_HD inline void rad_spechub_propose_row(const RadSpechubProposeCall& c,int64_t row) {
    RadSpechubState s{}; s.draw=*reinterpret_cast<const RadSpechubDraw*>(c.draw+row*c.ld[1]); s.status=-1;
    int32_t* packed=c.packed+row*c.ld[5];
    for (int i=0;i<7;++i) packed[i]=i ? -1 : 0;
    for (int i=0;i<8;++i) {s.ids[i]=-1;c.idx[row*c.ld[6]+i]=-1;c.val[row*c.ld[7]+i]=-INFINITY;}
    bool valid=true; uint64_t keys[16];
    const uint32_t* gathered=c.gathered+row*c.ld[0];
    for (int j=0;j<16;++j) {
        int32_t id=static_cast<int32_t>(gathered[2*j]); uint32_t bits=gathered[2*j+1];
        if (id>=0 && (id>=c.vocab || (bits&0x7fffffffu)>0x7f800000u || bits==0x7f800000u)) valid=false;
        uint32_t ordered=bits&0x80000000u ? ~bits : bits|0x80000000u;
        keys[j]=id<0 ? 0 : (static_cast<uint64_t>(ordered)<<32)|(0xffffffffu-gathered[2*j]);
    }
    double logits[8];
    for (int i=0;i<8;++i) {
        uint64_t key=0; for (int j=0;j<16;++j) if (keys[j]>key) key=keys[j];
        for (int j=0;j<16;++j) if (keys[j]==key) keys[j]=0;
        uint32_t bits=static_cast<uint32_t>(key>>32);
        bits=key ? (bits&0x80000000u ? bits&0x7fffffffu : ~bits) : 0xff800000u;
        float value; __builtin_memcpy(&value,&bits,sizeof(value));
        int id=key ? static_cast<int32_t>(0xffffffffu-static_cast<uint32_t>(key)) : -1;
        c.idx[row*c.ld[6]+i]=id; c.val[row*c.ld[7]+i]=value;
        if (id<0 || bits==0xff800000u) continue;
        for (int j=0;j<s.count;++j) if (s.ids[j]==id) valid=false;
        s.ids[s.count]=id; logits[s.count++]=value;
    }
    int draft=c.draft_ids[row*c.ld[3]]; s.root=-1;
    if (draft>=0 && draft<c.token_count) s.root=c.tokens[static_cast<int64_t>(draft)*c.ld[2]];
    valid=valid && s.root>=0 && s.root<c.vocab && s.count>0;
    double sum=0;
    if (valid) for (int i=0;i<s.count;++i) {
        s.q[i]=::exp(logits[i]-logits[0]); if (!(s.q[i]>0) || !rad_spechub_finite(s.q[i])) valid=false; sum+=s.q[i];
    }
    if (valid) for (int i=0;i<s.count;++i) {
        s.q[i]/=sum;
        if (!(s.q[i]>0)) valid=false;
    }
    if (valid) {
        double u=rad_spechub_uniform(s.draw.seed,s.draw.pos,RAD_SPECHUB_KEY_FIRST);
        int first=s.count-1;
        for (int i=0;i<s.count;++i) {u-=s.q[i];if (u<0) {first=i;break;}}
        int second=0;
        if (first==0 && s.count>1) {
            double tail=0;for (int i=1;i<s.count;++i) tail+=s.q[i];
            u=rad_spechub_uniform(s.draw.seed,s.draw.pos,RAD_SPECHUB_KEY_TAIL)*tail;
            second=s.count-1;for (int i=1;i<s.count;++i) {u-=s.q[i];if (u<0) {second=i;break;}}
        }
        s.pair[0]=s.ids[first];s.pair[1]=s.ids[second];s.status=1;
        c.idx[row*c.ld[6]]=s.pair[0]; // ADV_TAKE uses the sampled first token, not the hub.
        packed[0]=s.count==1 ? 2 : 3;packed[1]=s.root;packed[2]=s.pair[0];packed[3]=s.count==1 ? -1 : s.pair[1];
        packed[4]=-1;packed[5]=0;packed[6]=s.count==1 ? -1 : 0;
    } else {s.count=0;}
    *reinterpret_cast<RadSpechubState*>(c.state+row*c.ld[4])=s;
}
SH_HD inline int rad_spechub_verify_row(const RadSpechubVerifyCall& c,int64_t row) {
    const auto& s=*reinterpret_cast<const RadSpechubState*>(c.state+row*c.ld[3]);
    if (s.status==0) return c.token[row*c.ld[4]];
    const auto& params=*reinterpret_cast<const RadSampleParams*>(c.params+row*c.ld[2]);
    if (!rad_spechub_state_valid(s,c.vocab) || s.draw.seed!=params.seed || s.draw.pos!=params.pos) return -1;
    const int32_t* ids=c.idx+row*c.ld[0];const float* val=c.val+row*c.ld[1];
    double maximum=-INFINITY;
    for (int64_t j=0;j<c.n_cand;++j) {
        if (ids[j]<0) continue;
        if (ids[j]>=c.vocab || (!rad_spechub_finite(val[j]) && val[j]!=-INFINITY)) return -1;
        if (val[j]==-INFINITY) continue;
        // ponytail: duplicate validation is O(n_cand^2); use scratch-backed ids if large top-k dominates.
        for (int64_t k=0;k<j;++k) if (ids[k]==ids[j] && val[k]!=-INFINITY) return -1;
        if (val[j]>maximum) maximum=val[j];
    }
    if (!rad_spechub_finite(maximum)) return -1;
    double sum=0,p[8]={};
    for (int64_t j=0;j<c.n_cand;++j) if (ids[j]>=0 && val[j]!=-INFINITY) {
        double w=::exp(static_cast<double>(val[j])-maximum);
        if (!(w>0) || !rad_spechub_finite(w)) return -1;
        sum+=w;for (int k=0;k<s.count;++k) if (ids[j]==s.ids[k]) p[k]=w;
    }
    if (!(sum>0) || !rad_spechub_finite(sum)) return -1;
    if (!c.transport) {
        // Same ordinary PICK key and 24-bit uniform, but the declared FP64 p plane.
        double raw=rad_spechub_uniform(params.seed,params.pos,RAD_SAMPLE_KEY_PICK);
        double u=(static_cast<uint32_t>(raw*16777216.0)/16777216.0)*sum;
        int last=-1;
        for (int64_t j=0;j<c.n_cand;++j) if (ids[j]>=0 && val[j]!=-INFINITY) {
            last=ids[j];u-=::exp(static_cast<double>(val[j])-maximum);
            if (u<0) return last;
        }
        return last;
    }
    for (int k=0;k<s.count;++k) p[k]/=sum;
    RadSpechubTransport t{};if (!rad_spechub_transport(s,p,&t)) return -1;
    int result=rad_spechub_match(t,s.pair[0],s.pair[1],rad_spechub_uniform(params.seed,params.pos,RAD_SPECHUB_KEY_MATCH));
    if (result!=-2) return result;
    double residual=0;
    for (int64_t j=0;j<c.n_cand;++j) if (ids[j]>=0 && val[j]!=-INFINITY)
        residual+=rad_spechub_residual(t,ids[j],::exp(static_cast<double>(val[j])-maximum)/sum);
    if (!(residual>0) || !rad_spechub_finite(residual)) return -1;
    double u=rad_spechub_uniform(params.seed,params.pos,RAD_SPECHUB_KEY_RESID)*residual;
    int last=-1;
    for (int64_t j=0;j<c.n_cand;++j) if (ids[j]>=0 && val[j]!=-INFINITY) {
        double mass=rad_spechub_residual(t,ids[j],::exp(static_cast<double>(val[j])-maximum)/sum);
        if (mass>0) {last=ids[j];u-=mass;if (u<0) return last;}
    }
    return last;
}
/* ---------------------------------------------------------------- linear MTP coupling
 * RADIANCE_MTP_COUPLED=1: x ~ Q over the FINAL reranked top-8 of every temporal round, then the
 * maximal coupling against the target P over the sampler's FINAL candidate set. Host reference,
 * GPU kernels, engine and sampler share these functions; nothing else re-derives them.
 *
 * Numerics: Q is the FP64 softmax of the unchanged FP32 shortlist scores at temperature 1; P is
 * the FP64 softmax of the unchanged FP32 candidate logits (same formula the SpecHub verifier
 * uses). Both are same-or-higher precision than the ordinary FP32/24-bit sample_pick, so a coupled
 * row realizes the ideal filtered target law to FP64 rounding, NOT the FP32 pick's finite law,
 * and is not seed-identical to it. Uniforms are 53-bit (rad_spechub_uniform). */
SH_HD inline bool rad_mtp_couple_state_valid(const RadMtpCoupleState& s,int64_t vocab) {
    if (s.status!=1 || s.draw.mode!=1 || s.count<1 || s.count>8 || s.round<1 ||
        s.round>RAD_MTP_COUPLE_MAX_ROUNDS) return false;
    double sum=0; bool has_x=false;
    for (int i=0;i<8;++i) {
        if (i>=s.count) {if (s.ids[i]!=-1 || s.q[i]!=0) return false; continue;}
        if (s.ids[i]<0 || s.ids[i]>=vocab || !rad_spechub_finite(s.q[i]) || !(s.q[i]>0) || s.q[i]>1) return false;
        for (int j=0;j<i;++j) if (s.ids[j]==s.ids[i]) return false;
        sum+=s.q[i]; has_x|=s.ids[i]==s.x;
    }
    return has_x && ::fabs(sum-1)<=1e-12;
}
/* Holes (id<0) and -inf scores carry no mass. An entry whose FP64 weight or normalized q
 * underflows to zero is DROPPED, so the stored q[] is exactly the law the draw walks. False on
 * an out-of-range id, NaN/+inf score, duplicate id or empty support (s->count = 0). */
SH_HD inline bool rad_mtp_couple_build_q(const int32_t* ids,const float* val,int64_t vocab,RadMtpCoupleState* s) {
    int32_t kid[8]; float kv[8]; int n=0;
    s->count=0;
    for (int i=0;i<8;++i) {s->ids[i]=-1;s->q[i]=0;}
    for (int i=0;i<8;++i) {
        if (ids[i]<0) continue;
        if (ids[i]>=vocab || val[i]!=val[i] || val[i]==INFINITY) return false;
        if (val[i]==-INFINITY) continue;
        for (int j=0;j<n;++j) if (kid[j]==ids[i]) return false;
        kid[n]=ids[i]; kv[n++]=val[i];
    }
    if (!n) return false;
    float top=kv[0];
    for (int j=1;j<n;++j) if (kv[j]>top) top=kv[j];
    double w[8],sum=0;
    for (int j=0;j<n;++j) {w[j]=::exp(static_cast<double>(kv[j])-top); sum+=w[j];}
    for (int j=0;j<n;++j) {
        const double q=w[j]/sum;
        if (q>0) {s->ids[s->count]=kid[j]; s->q[s->count++]=q;}
    }
    return s->count>0;
}
/* Inverse CDF over the stored q with a uniform in [0,1); index into ids. */
SH_HD inline int rad_mtp_couple_pick(const RadMtpCoupleState& s,double u) {
    double total=0;
    for (int i=0;i<s.count;++i) total+=s.q[i];
    double c=u*total;
    for (int i=0;i<s.count;++i) {c-=s.q[i]; if (c<0) return i;}
    return s.count-1;
}
/* Residual mass r(y)=max(0,P(y)-Q(y)) of one candidate. Q tokens use their P aggregated over
 * duplicate candidates, counted once at the first occurrence (`seen`). */
SH_HD inline double rad_mtp_couple_mass(const RadMtpCoupleState& s,const double* pq,int32_t id,float v,
                                        float top,double sum,unsigned* seen) {
    if (id<0 || v==-INFINITY) return 0;
    for (int k=0;k<s.count;++k) if (s.ids[k]==id) {
        if (*seen&(1u<<k)) return 0;
        *seen|=1u<<k; return rad_spechub_positive(pq[k]-s.q[k]);
    }
    return ::exp(static_cast<double>(v)-top)/sum;
}
/* Emits into *out and returns true, or returns false for an invalid/empty target. Needs
 * u_accept, u_resid in [0,1). If FP64 cancellation leaves R=0 while some Q token has P<Q,
 * sample P directly in FP64 with the independent residual uniform. The fallback decision reads
 * (P,Q) only -- never x or either uniform -- and must not restore sample_pick's FP32/24-bit law.
 *
 * R sums r over every candidate INCLUDING x. The residual walk runs only when P(x)<Q(x), where
 * r(x)=max(0,P(x)-Q(x)) is exactly +0 (distinct doubles never subtract to 0), so the walk --
 * which skips x explicitly, residual at x exactly zero -- has total R bit for bit. */
SH_HD inline bool rad_mtp_couple_decide(const RadMtpCoupleState& s,const int32_t* ids,const float* val,int64_t n,
                                        int64_t vocab,double u_accept,double u_resid,int32_t* out) {
    float top=-INFINITY;
    for (int64_t j=0;j<n;++j) {
        if (ids[j]<0) continue;
        if (ids[j]>=vocab || val[j]!=val[j] || val[j]==INFINITY) return false;
        if (val[j]>top) top=val[j];
    }
    if (top==-INFINITY) return false;
    double sum=0,pq[8]={};
    for (int64_t j=0;j<n;++j) {
        if (ids[j]<0 || val[j]==-INFINITY) continue;
        const double w=::exp(static_cast<double>(val[j])-top);
        sum+=w;
        for (int k=0;k<s.count;++k) if (ids[j]==s.ids[k]) {pq[k]+=w; break;}
    }
    if (!(sum>0) || !rad_spechub_finite(sum)) return false;
    int ix=-1; bool dominated=true;
    for (int k=0;k<s.count;++k) {
        pq[k]/=sum; dominated=dominated && pq[k]>=s.q[k];
        if (s.ids[k]==s.x) ix=k;
    }
    if (ix<0) return false;   /* unreachable for a rad_mtp_couple_state_valid state */
    unsigned seen=0; double resid=0;
    for (int64_t j=0;j<n;++j) resid+=rad_mtp_couple_mass(s,pq,ids[j],val[j],top,sum,&seen);
    if (!dominated && !(resid>0 && rad_spechub_finite(resid))) {
        double c=u_resid*sum; int32_t last=-1;
        for (int64_t j=0;j<n;++j) if (ids[j]>=0 && val[j]!=-INFINITY) {
            const double w=::exp(static_cast<double>(val[j])-top);
            if (w>0) {last=ids[j]; c-=w; if (c<0) {*out=last; return true;}}
        }
        *out=last; return true;
    }
    if (pq[ix]>=s.q[ix] || u_accept*s.q[ix]<pq[ix]) {*out=s.x; return true;}
    /* Rejected, so P(x)<Q(x): not dominated, resid>0, and some y!=x has mass>0 below. */
    double c=u_resid*resid; int32_t last=-1; seen=0;
    for (int64_t j=0;j<n;++j) {
        if (ids[j]==s.x) continue;
        const double m=rad_mtp_couple_mass(s,pq,ids[j],val[j],top,sum,&seen);
        if (m>0) {last=ids[j]; c-=m; if (c<0) {*out=last; return true;}}
    }
    *out=last; return true;
}
struct RadMtpCoupleProposeCall {
    int32_t* idx; const float* val; const int32_t* card; const uint32_t* draw; uint32_t* state;
    int64_t M,vocab,ld_idx,ld_val,ld_draw,st_round,st_row; int round;
};
struct RadMtpCoupleVerifyCall {
    const int32_t* idx; const float* val; const uint32_t* params; const uint32_t* state; int32_t* token;
    int64_t M,n_cand,vocab,ld[5];
};
/* Column 0 is what ADV_ROUND/ADV_TAKE take; it becomes x only for a valid sampled row. */
SH_HD inline void rad_mtp_couple_propose_row(const RadMtpCoupleProposeCall& c,int64_t row) {
    RadMtpCoupleState s{};
    s.draw=*reinterpret_cast<const RadMtpCoupleDraw*>(c.draw+row*c.ld_draw);
    int32_t* idx=c.idx+row*c.ld_idx;
    s.round=c.round; s.x=idx[0];
    for (int i=0;i<8;++i) s.ids[i]=-1;
    if (s.draw.mode==1) {
        s.status=-1;
        if (s.draw.card==c.card[row] && rad_mtp_couple_build_q(idx,c.val+row*c.ld_val,c.vocab,&s)) {
            const double u=rad_spechub_uniform(s.draw.seed,s.draw.base,
                                               RAD_MTP_COUPLE_KEY_DRAFT+static_cast<uint64_t>(c.round));
            s.x=s.ids[rad_mtp_couple_pick(s,u)]; s.status=1; idx[0]=s.x;
        } else {
            s.count=0;
            for (int i=0;i<8;++i) {s.ids[i]=-1;s.q[i]=0;}
        }
    }
    *reinterpret_cast<RadMtpCoupleState*>(c.state+(c.round-1)*c.st_round+row*c.st_row)=s;
}
/* Runs after sample_pick; rows without a valid bound state keep the ordinary token. */
SH_HD inline void rad_mtp_couple_verify_row(const RadMtpCoupleVerifyCall& c,int64_t row) {
    const auto& s=*reinterpret_cast<const RadMtpCoupleState*>(c.state+row*c.ld[3]);
    if (s.status!=1) return;
    const auto& p=*reinterpret_cast<const RadSampleParams*>(c.params+row*c.ld[2]);
    if ((p.flags&RAD_SP_GREEDY) || s.draw.seed!=p.seed || s.pos!=p.pos || !rad_mtp_couple_state_valid(s,c.vocab)) return;
    int32_t out=-1;
    if (rad_mtp_couple_decide(s,c.idx+row*c.ld[0],c.val+row*c.ld[1],c.n_cand,c.vocab,
            rad_spechub_uniform(p.seed,p.pos,RAD_SAMPLE_KEY_ACCEPT),
            rad_spechub_uniform(p.seed,p.pos,RAD_SAMPLE_KEY_RESID),&out))
        c.token[row*c.ld[4]]=out;
}
int rad_mtp_couple_propose_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_mtp_couple_verify_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_mtp_couple_propose_args(const RadArgs*,RadMtpCoupleProposeCall*);
int rad_mtp_couple_verify_args(const RadArgs*,RadMtpCoupleVerifyCall*);
RadKernelInfo rad_mtp_couple_row(int,RadLaunchFn);
int rad_mine_mtp_couple_propose_launch(const RadArgs*,RadStream);
int rad_mine_mtp_couple_verify_launch(const RadArgs*,RadStream);
int rad_ref_mtp_couple_propose(const RadArgs*,RadStream);
int rad_ref_mtp_couple_verify(const RadArgs*,RadStream);
int rad_spechub_schema_count();
const RadOpSchema* rad_spechub_schema_at(int);
int rad_spechub_propose_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_spechub_verify_shape(const RadParam*,int,int,RadOpdDesc*);
int rad_spechub_propose_args(const RadArgs*,RadSpechubProposeCall*);
int rad_spechub_verify_args(const RadArgs*,RadSpechubVerifyCall*);
RadKernelInfo rad_spechub_row(int,RadLaunchFn);
int rad_mine_spechub_propose_launch(const RadArgs*,RadStream);
int rad_mine_spechub_verify_launch(const RadArgs*,RadStream);
int rad_ref_spechub_propose(const RadArgs*,RadStream);
int rad_ref_spechub_verify(const RadArgs*,RadStream);
int rad_mine_spechub_count();
const RadKernelInfo* rad_mine_spechub_at(int);
int rad_ref_spechub_count();
const RadKernelInfo* rad_ref_spechub_at(int);
#undef SH_HD
#endif
