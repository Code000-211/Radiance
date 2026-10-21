#include "mine/spechub.h"
#include "core/plugin/registry.h"
#include <algorithm>
#include <initializer_list>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool b,const char* what) {if (!b) throw std::runtime_error(what);}
void check(int rc,const char* what) {require(rc==RAD_OK,what);}
void near(double a,double b,double tol=2e-12) {require(std::fabs(a-b)<tol,"probability mismatch");}
RadSpechubState state(std::initializer_list<double> q) {
    RadSpechubState s{};s.status=1;s.root=9;s.count=static_cast<int>(q.size());
    int i=0;for (double w:q) {s.ids[i]=i;s.q[i++]=w;}
    s.pair[0]=0;s.pair[1]=s.count==1 ? 0 : 1;return s;
}
/* Independent long-double allocation oracle, explicitly enumerating Q rows and
 * target columns. It does not call the transport, residual or match helpers. */
void oracle(const RadSpechubState& s,const std::vector<double>& p) {
    struct Row {int x,y;long double mass;std::vector<long double> cells;};
    std::vector<Row> rows;long double tail=0;
    for (int i=1;i<s.count;++i) tail+=s.q[i];
    if (s.count==1) rows.push_back({0,0,1,std::vector<long double>(p.size())});
    for (int i=1;i<s.count;++i) {
        long double first=std::min(static_cast<long double>(p[i]),static_cast<long double>(s.q[i]));
        Row r{i,0,s.q[i],std::vector<long double>(p.size())};r.cells[i]=first;rows.push_back(r);
        r={0,i,s.q[0]*(s.q[i]/tail),std::vector<long double>(p.size())};
        r.cells[i]=std::min(static_cast<long double>(p[i])-first,r.mass);rows.push_back(r);
    }
    long double hub=p[0];
    for (bool hub_first:{true,false}) {
        long double capacity=0;
        for (const auto& r:rows) if ((r.x==0)==hub_first) {
            long double used=0;for (auto v:r.cells) used+=v;capacity+=r.mass-used;
        }
        long double assign=std::min(hub,capacity);
        for (auto& r:rows) if ((r.x==0)==hub_first) {
            long double used=0;for (auto v:r.cells) used+=v;
            r.cells[0]=capacity>0 ? (r.mass-used)*assign/capacity : 0;
        }
        hub-=assign;
    }
    std::vector<long double> residual(p.begin(),p.end());
    for (const auto& r:rows) for (size_t i=0;i<p.size();++i) residual[i]-=r.cells[i];
    long double rem=0;for (auto v:residual) rem+=v;
    double onq[8]={};for (int i=0;i<s.count;++i) onq[i]=p[i];
    RadSpechubTransport t{};require(rad_spechub_transport(s,onq,&t),"transport failed");
    std::vector<long double> marginal(p.size(),0);
    for (const auto& r:rows) {
        int g=r.x==0 ? 1 : 0,k=r.x==0 ? r.y : r.x;
        near(t.joint[g][k],static_cast<double>(r.mass));
        near(t.hub[g][k],static_cast<double>(r.cells[0]));
        if (k) near(t.nonhub[g][k],static_cast<double>(r.cells[k]));
        long double matched=0;for (auto v:r.cells) matched+=v;
        long double row_total=0;
        for (size_t i=0;i<p.size();++i) {
            double rr=rad_spechub_residual(t,static_cast<int>(i),p[i]);near(rr,static_cast<double>(residual[i]));
            long double cell=r.cells[i]+(rem>1e-18L ? (r.mass-matched)*residual[i]/rem : 0);
            marginal[i]+=cell;row_total+=cell;
        }
        near(static_cast<double>(row_total),static_cast<double>(r.mass));
        // Supplied uniforms exercise both matched columns and residual without RNG coupling.
        if (k && t.nonhub[g][k]>0) require(rad_spechub_match(t,r.x,r.y,t.nonhub[g][k]/(2*t.joint[g][k]))==k,"nonhub interval");
        if (t.hub[g][k]>0) require(rad_spechub_match(t,r.x,r.y,(t.nonhub[g][k]+t.hub[g][k]/2)/t.joint[g][k])==0,"hub interval");
        if (r.mass-matched>1e-12L) require(rad_spechub_match(t,r.x,r.y,static_cast<double>((matched+(r.mass-matched)/2)/r.mass))==-2,"residual interval");
    }
    for (size_t i=0;i<p.size();++i) near(static_cast<double>(marginal[i]),p[i]);
}
RadTensor tensor(void* data,uint32_t dtype,std::initializer_list<int64_t> dims,int padding=0) {
    RadTensor t{};t.data=data;t.dtype=dtype;t.rank=static_cast<uint32_t>(dims.size());
    int i=0;for (auto d:dims) t.shape[i++]=d;
    int64_t stride=1;for (i=static_cast<int>(t.rank)-1;i>=0;--i) {t.stride[i]=stride;stride*=t.shape[i];}
    if (t.rank>1) t.stride[0]+=padding;return t;
}
struct Fixture {
    std::array<uint32_t,32> gathered{};
    RadSpechubDraw draw{11,3,77};RadSpechubState s{};
    int32_t root=9,draft=0,packed[7]{},idx[8]{};float val[8]{};
    int32_t target_ids[4]={0,1,2,7};float target_val[4]={-2,-1,0,1};
    RadSampleParams params{};int32_t token=123;
    Fixture() {
        params.seed=draw.seed;params.pos=draw.pos;
        for (int i=0;i<16;++i) {int32_t id=i<3 ? i : -1;float logit=i<3 ? -static_cast<float>(i) : -INFINITY;
            std::memcpy(&gathered[2*i],&id,4);std::memcpy(&gathered[2*i+1],&logit,4);}
    }
    std::array<RadTensor,8> proposal() {return {{tensor(gathered.data(),RAD_F32,{1,16,2}),tensor(&draw,RAD_U32,{1,6}),
        tensor(&root,RAD_I32,{1}),tensor(&draft,RAD_I32,{1}),tensor(&s,RAD_U32,{1,36}),tensor(packed,RAD_I32,{1,7}),
        tensor(idx,RAD_I32,{1,8}),tensor(val,RAD_F32,{1,8})}};}
    std::array<RadTensor,5> verifier() {return {{tensor(target_ids,RAD_I32,{1,4}),tensor(target_val,RAD_F32,{1,4}),
        tensor(&params,RAD_U32,{1,24}),tensor(&s,RAD_U32,{1,36}),tensor(&token,RAD_I32,{1})}};}
};
rad::Geometry geometry(bool verify,int M=1) {
    rad::Geometry g;g.set_i("M",M);g.set_i("n_vocab",10);g.set_i(verify ? "n_cand" : "world_size",verify ? 4 : 2);return g;
}
template<size_t N> RadArgs args(std::array<RadTensor,N>& t,rad::Geometry& g) {
    RadArgs a{};a.t=t.data();a.n_t=N;a.p=g.params();a.n_p=g.n_params();return a;
}
void host_boundaries(int transport) {
    Fixture f;auto p=f.proposal();auto gp=geometry(false);auto a=args(p,gp);
    check(rad_ref_spechub_propose(&a,nullptr),"proposal");require(f.s.status==1 && f.packed[0]==3,"proposal state");
    require(f.idx[0]==f.s.pair[0] && f.packed[2]==f.idx[0],"ADV_TAKE identity");
    auto v=f.verifier();auto gv=geometry(true);gv.set_i("transport",transport);auto b=args(v,gv);
    auto verify=[&] {check(rad_ref_spechub_verify(&b,nullptr),"verifier launch");return f.token;};
    require(verify()>=0,"valid target");
    const int flat_result=f.token;
    v[4]=tensor(&f.token,RAD_I32,{1,1});
    require(verify()==flat_result,"production [M,1] token layout");
    v[4].shape[1]=2;require(rad_ref_spechub_verify(&b,nullptr)==RAD_E_SHAPE,"token plane width");
    v[4]=tensor(&f.token,RAD_I32,{1,1});
    gv.set_i("M",4);b=args(v,gv);require(verify()==flat_result,"narrow live verifier rows in M band");
    gp.set_i("M",4);a=args(p,gp);check(rad_ref_spechub_propose(&a,nullptr),"narrow live proposal rows in M band");
    p[1].shape[0]=2;require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_SHAPE,"inconsistent live rows");p=f.proposal();a=args(p,gp);
    gp.set_i("M",1);a=args(p,gp);p[0].shape[0]=2;
    require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_SHAPE,"live rows exceed M band");p=f.proposal();a=args(p,gp);
    auto saved=f.s;
    ++f.s.draw.seed;require(verify()==-1,"wrong seed");f.s=saved;
    ++f.s.draw.pos;require(verify()==-1,"wrong position");f.s=saved;
    f.s.root=10;require(verify()==-1,"bad root");f.s=saved;
    f.s.pair[0]=8;require(verify()==-1,"bad pair");f.s=saved;
    f.s.pair[0]=f.s.pair[1]=f.s.ids[0];require(verify()==-1,"duplicate non-pointmass pair");f.s=saved;
    f.s.q[0]=NAN;require(verify()==-1,"nonfinite q");f.s=saved;
    f.s.q[0]=0;require(verify()==-1,"zero q");f.s=saved;
    f.s.count=9;require(verify()==-1,"oversize state");f.s=saved;
    f.s.ids[1]=f.s.ids[0];require(verify()==-1,"duplicate q ids");f.s=saved;
    f.s.status=-1;require(verify()==-1,"malformed status");f.s.status=0;f.token=456;require(verify()==456,"disabled row");f.s=saved;
    for (int i=0;i<4;++i) {f.target_ids[i]=6+i;f.target_val[i]=i==0 ? 0 : -INFINITY;}
    require(verify()==6,"target entirely outside q");
    f.target_val[0]=-INFINITY;require(verify()==-1,"zero target mass");
    f.target_val[0]=NAN;require(verify()==-1,"NaN target");
    f.target_val[0]=INFINITY;require(verify()==-1,"infinite target");
    f.target_val[0]=0;f.target_val[1]=-1000;require(verify()==-1,"finite target underflow");
    f.draft=-1;check(rad_ref_spechub_propose(&a,nullptr),"invalid root launch");require(f.s.status==-1 && f.s.count==0 && f.packed[0]==0,"reset invalid root");
    f.draft=0;float negative=-INFINITY;for (int i=1;i<3;++i) std::memcpy(&f.gathered[2*i+1],&negative,4);
    check(rad_ref_spechub_propose(&a,nullptr),"pointmass");require(f.s.count==1 && f.s.pair[0]==f.s.pair[1] && f.packed[0]==2,"dedup pointmass");
    f.target_val[1]=-INFINITY;require(verify()==6,"pointmass full-target residual");
    float tiny=-70;std::memcpy(&f.gathered[3],&tiny,4);
    check(rad_ref_spechub_propose(&a,nullptr),"near pointmass");
    require(f.s.status==1 && f.s.count==2 && f.s.q[0]==1 && f.s.q[1]>0 && f.s.pair[0]!=f.s.pair[1],"stable explicit tail");
    float logit=-1000;std::memcpy(&f.gathered[3],&logit,4);
    check(rad_ref_spechub_propose(&a,nullptr),"underflow");require(f.s.status==-1 && f.packed[0]==0,"proposal underflow not pointmass");
    float nan=NAN;std::memcpy(&f.gathered[3],&nan,4);
    check(rad_ref_spechub_propose(&a,nullptr),"NaN proposal");require(f.s.status==-1 && f.s.count==0,"NaN proposal rejected");
    for (int i=0;i<16;++i) {
        int32_t id=i<9 ? 8-i : 0;float score=0;
        std::memcpy(&f.gathered[2*i],&id,4);std::memcpy(&f.gathered[2*i+1],&score,4);
    }
    check(rad_ref_spechub_propose(&a,nullptr),"bit-key tie merge");require(f.s.count==8,"top8 count and identical-key removal");
    for (int i=0;i<8;++i) {require(f.s.ids[i]==i,"unsigned tie id order");near(f.s.q[i],0.125);}
    p[0].stride[1]=3;require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_STRIDE,"gather stride");p=f.proposal();a=args(p,gp);
    p[4].data=p[1].data;require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_INVAL,"overlap");p=f.proposal();a=args(p,gp);
    p[1].data=reinterpret_cast<uint32_t*>(&f.draw)+1;require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_INVAL,"wire alignment");
    p=f.proposal();a=args(p,gp);p[5].shape[1]=6;require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_SHAPE,"capacity");
    p=f.proposal();a=args(p,gp);p[4].dtype=RAD_I32;require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_DTYPE,"state dtype");
    p=f.proposal();gp.set_i("world_size",1);a=args(p,gp);require(rad_ref_spechub_propose(&a,nullptr)==RAD_E_UNSUPPORTED,"unsupported TP");
    gv.set_i("transport",2);b=args(v,gv);require(rad_ref_spechub_verify(&b,nullptr)==RAD_E_UNSUPPORTED,"unsupported transport");
}
void distribution_tests() {
    oracle(state({0.5,0.3,0.2}),{0.15,0.1,0.2,0.55});
    oracle(state({0.5,0.3,0.2}),{0,0,0,1});
    oracle(state({1}),{0.2,0.3,0.5});oracle(state({1}),{1,0});
    oracle(state({1,1e-30}),{0.1,0.2,0.7}); // q[a] rounds to 1; explicit tail remains real.
    // Counterexample: (a,x) must receive hub mass before the reverse pair.
    auto s=state({0.6,0.4});double p[8]={0.3,0.1};RadSpechubTransport t{};
    require(rad_spechub_transport(s,p,&t),"paper counterexample");near(t.hub[1][1],0.3);near(t.hub[0][1],0);
    oracle(s,{0.3,0.1,0.6});
    // Rational grid sweeps independently check marginals, including support outside q.
    for (int a=1;a<=8;++a) for (int b=1;b<=10-a;++b) {
        double qa=a/10.0,qb=b/10.0,qc=1-qa-qb;if (qc<=0 || qa<qb || qa<qc) continue;
        for (int x=0;x<=5;++x) for (int y=0;y<=5-x;++y)
            oracle(state({qa,qb,qc}),{x/5.0,y/5.0,0,(5-x-y)/5.0});
    }
    require(rad_spechub_match(t,0,8,0.5)==-1,"impossible pair");
    require(rad_spechub_match(t,0,1,1)==-1,"uniform upper boundary");
    constexpr uint64_t keys[]={RAD_SPECHUB_KEY_FIRST,RAD_SPECHUB_KEY_TAIL,RAD_SPECHUB_KEY_MATCH,RAD_SPECHUB_KEY_RESID};
    for (int i=0;i<4;++i) {
        require(keys[i]!=RAD_SAMPLE_KEY_PICK && keys[i]!=RAD_SAMPLE_KEY_ACCEPT && keys[i]!=RAD_SAMPLE_KEY_RESID,"ordinary stream collision");
        for (int j=0;j<i;++j) require(keys[i]!=keys[j],"SpecHub stream collision");
        require(rad_spechub_uniform(19,7,keys[i])!=rad_spechub_uniform(19,8,keys[i]),"position RNG coordinate");
    }
}
void proposal_statistics() {
    Fixture f;auto ts=f.proposal();auto g=geometry(false);auto a=args(ts,g);
    constexpr int N=40000;int first[3]={},pairs[2][3]={};
    int outputs[2][10]={};
    auto vt=f.verifier();auto vg0=geometry(true),vg1=geometry(true);vg0.set_i("transport",0);
    auto va0=args(vt,vg0),va1=args(vt,vg1);
    for (int i=0;i<N;++i) {
        f.draw.seed=static_cast<uint64_t>(i);check(rad_ref_spechub_propose(&a,nullptr),"statistical proposal");
        require(f.s.status==1,"proposal malformed");++first[f.s.pair[0]];
        if (f.s.pair[0]==0) ++pairs[1][f.s.pair[1]];else ++pairs[0][f.s.pair[0]];
        f.params.seed=f.draw.seed;
        check(rad_ref_spechub_verify(&va0,nullptr),"matched control");require(f.token>=0 && f.token<10,"control token");++outputs[0][f.token];
        double u=rad_sample_uniform01(f.params.seed,f.params.pos);
        double total=0;for (float v:f.target_val) total+=std::exp(static_cast<double>(v)-1);
        u*=total;int expected=f.target_ids[3];
        for (int k=0;k<4;++k) {u-=std::exp(static_cast<double>(f.target_val[k])-1);if (u<0) {expected=f.target_ids[k];break;}}
        require(f.token==expected,"control uses ordinary PICK stream");
        check(rad_ref_spechub_verify(&va1,nullptr),"transport sample");require(f.token>=0 && f.token<10,"transport token");++outputs[1][f.token];
    }
    double tail=f.s.q[1]+f.s.q[2];
    for (int i=0;i<3;++i) near(first[i]/double(N),f.s.q[i],0.012);
    for (int i=1;i<3;++i) {near(pairs[0][i]/double(N),f.s.q[i],0.012);near(pairs[1][i]/double(N),f.s.q[0]*f.s.q[i]/tail,0.012);}
    double total=0;for (float v:f.target_val) total+=std::exp(static_cast<double>(v)-1);
    for (int mode=0;mode<2;++mode) for (int k=0;k<4;++k)
        near(outputs[mode][f.target_ids[k]]/double(N),std::exp(static_cast<double>(f.target_val[k])-1)/total,0.012);
}
/* ------------------------------------------------------------------ linear MTP coupling */
RadMtpCoupleState cstate(std::initializer_list<int32_t> ids,std::initializer_list<double> q,int32_t x) {
    RadMtpCoupleState s{};s.status=1;s.round=1;s.draw.mode=1;s.x=x;s.count=static_cast<int>(ids.size());
    for (int i=0;i<8;++i) s.ids[i]=-1;
    int i=0;for (int32_t id:ids) s.ids[i++]=id;
    i=0;for (double w:q) s.q[i++]=w;
    return s;
}
struct Target {std::vector<int32_t> ids;std::vector<float> val;};
/* FP32 logits ln(p): their FP64 softmax is the dyadic law p within FP32 rounding (~1e-7). */
Target target(std::initializer_list<int32_t> ids,std::initializer_list<double> p) {
    Target t;t.ids.assign(ids.begin(),ids.end());
    for (double w:p) t.val.push_back(w>0 ? static_cast<float>(std::log(w)) : -INFINITY);
    return t;
}
/* Independent long-double law of the ordinary sampler's candidate multiset. */
std::vector<long double> target_law(const Target& t,int vocab) {
    long double top=-INFINITY,sum=0;std::vector<long double> law(static_cast<size_t>(vocab),0);
    for (size_t j=0;j<t.ids.size();++j) if (t.ids[j]>=0 && t.val[j]>top) top=t.val[j];
    for (size_t j=0;j<t.ids.size();++j) if (t.ids[j]>=0 && t.val[j]>-INFINITY) sum+=std::exp(t.val[j]-top);
    for (size_t j=0;j<t.ids.size();++j)
        if (t.ids[j]>=0 && t.val[j]>-INFINITY) law[static_cast<size_t>(t.ids[j])]+=std::exp(t.val[j]-top)/sum;
    return law;
}
int32_t decide(const RadMtpCoupleState& s,const Target& t,int vocab,double ua,double ur,bool* applicable=nullptr) {
    int32_t out=-99;
    const bool ok=rad_mtp_couple_decide(s,t.ids.data(),t.val.data(),static_cast<int64_t>(t.ids.size()),vocab,ua,ur,&out);
    if (applicable) *applicable=ok;else require(ok,"coupling applicable");
    return ok ? out : -99;
}
/* Midpoint integration of the REAL kernel over x~q and both uniforms. The dyadic fixtures put
 * every breakpoint -- P(x)/Q(x) and each residual CDF step over R -- on the 1/N grid, so the sum
 * is exact up to the logits' FP32 rounding, and a changed rule moves at least q/N^2 of mass. */
std::vector<long double> coupled_law(RadMtpCoupleState s,const Target& t,int vocab,int N=256) {
    std::vector<long double> law(static_cast<size_t>(vocab),0);
    for (int k=0;k<s.count;++k) {
        s.x=s.ids[k];
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
            const int32_t out=decide(s,t,vocab,(i+0.5)/N,(j+0.5)/N);
            require(out>=0 && out<vocab,"coupled token range");
            law[static_cast<size_t>(out)]+=static_cast<long double>(s.q[k])/(static_cast<long double>(N)*N);
        }
    }
    return law;
}
constexpr double kReject=0x1.fffffffffffffp-1;   // the largest uniform below one
void coupling_marginals() {
    constexpr int V=10;
    struct Case {RadMtpCoupleState q;Target p;};
    const Case cases[]={
        {cstate({0,1},{0.5,0.5},0),target({0,-1,1,2,3},{0.375,0,0.125,0.25,0.25})},   // supports differ, hole
        {cstate({0,1,5},{0.5,0.25,0.25},0),target({0,1,2},{0.25,0.25,0.5})},           // Q token outside P
        {cstate({0,1,2,3},{0.5,0.25,0.125,0.125},0),target({1},{1})},                  // point-mass P in Q
        {cstate({0,1},{0.5,0.5},0),target({7},{1})},                                    // point-mass P outside Q
        {cstate({0,1},{0.5,0.5},0),target({0,1},{0.875,0.125})},  // an x-excluded R would vanish at x=0
        {cstate({0,1},{0.5,0.5},0),target({0,0,1,2},{0.25,0.25,0.25,0.25})}};          // duplicates aggregate
    for (const auto& c:cases) {
        require(rad_mtp_couple_state_valid(c.q,V),"fixture state");
        const auto law=coupled_law(c.q,c.p,V),want=target_law(c.p,V);
        for (int y=0;y<V;++y) near(static_cast<double>(law[static_cast<size_t>(y)]),static_cast<double>(want[static_cast<size_t>(y)]),1e-6);
        // The residual at x is exactly zero: a rejected x never re-emerges.
        for (int k=0;k<c.q.count;++k) {
            auto s=c.q;s.x=s.ids[k];
            const long double px=want[static_cast<size_t>(s.x)];
            if (px>0) require(decide(s,c.p,V,0,0.5)==s.x,"u_accept=0 keeps a supported x");
            if (px<s.q[k]) for (int j=0;j<64;++j) require(decide(s,c.p,V,kReject,(j+0.5)/64)!=s.x,"residual excludes x");
        }
    }
    // P == Q bit for bit: build_q and the verifier normalize the same FP32 scores in one order.
    const int32_t sid[8]={3,1,4,-1,-1,-1,-1,-1};
    const float sval[8]={0,-0.5f,-1.25f,-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    RadMtpCoupleState pq{};pq.status=1;pq.round=1;pq.draw.mode=1;
    require(rad_mtp_couple_build_q(sid,sval,V,&pq) && pq.count==3,"P=Q shortlist");
    Target same;same.ids={3,1,4};same.val={0,-0.5f,-1.25f};
    auto nudged=pq;nudged.q[0]=std::nextafter(nudged.q[0],2.0);
    for (int k=0;k<3;++k) {
        auto s=pq;s.x=s.ids[k];require(rad_mtp_couple_state_valid(s,V),"P=Q state");
        for (double u:{0.0,0.5,kReject}) require(decide(s,same,V,u,0.5)==s.x,"P=Q always accepts");
        // R=0 while P(x0)<Q(x0): direct FP64 P sampling for EVERY x and acceptance uniform.
        auto t=nudged;t.x=t.ids[k];require(rad_mtp_couple_state_valid(t,V),"nudged state");
        for (double u:{0.0,kReject}) require(decide(t,same,V,u,0.5)==3,"R=0 fallback is x-independent");
    }
    auto s=cstate({0,1},{0.5,0.5},0);bool ok=true;
    for (int which=0;which<4;++which) {
        Target bad=which==2 ? target({0,10},{0.5,0.5}) : which==3 ? target({0,1},{0,0}) : target({0,1},{0.5,0.5});
        if (which==0) bad.val[1]=NAN;
        if (which==1) bad.val[1]=INFINITY;
        decide(s,bad,V,0.5,0.5,&ok);require(!ok,"invalid or empty target is inapplicable");
    }
    for (int which=0;which<10;++which) {
        auto m=cstate({0,1},{0.5,0.5},1);require(rad_mtp_couple_state_valid(m,V),"good state");
        if (which==0) m.status=0;
        if (which==1) m.draw.mode=0;
        if (which==2) m.round=0;
        if (which==3) m.round=RAD_MTP_COUPLE_MAX_ROUNDS+1;
        if (which==4) m.ids[1]=0;
        if (which==5) m.x=7;
        if (which==6) m.q[1]=0.5+1e-9;
        if (which==7) m.ids[2]=3;
        if (which==8) m.q[0]=NAN;
        if (which==9) m.ids[1]=V;
        require(!rad_mtp_couple_state_valid(m,V),"malformed coupling state");
    }
    RadMtpCoupleState q{};
    int32_t ids[8]={2,3,4,-1,-1,-1,-1,-1};
    float val[8]={0,-1000,-0.5f,-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    require(rad_mtp_couple_build_q(ids,val,V,&q) && q.count==2 && q.ids[0]==2 && q.ids[1]==4 &&
            q.ids[2]==-1 && q.q[2]==0,"underflowed Q entry is dropped");
    near(q.q[0],1/(1+std::exp(-0.5)),1e-15);near(q.q[0]+q.q[1],1,1e-15);
    for (int which=0;which<5;++which) {
        int32_t bi[8];float bv[8];std::copy(ids,ids+8,bi);std::copy(val,val+8,bv);
        if (which==0) bi[1]=2;
        if (which==1) bv[2]=NAN;
        if (which==2) bv[2]=INFINITY;
        if (which==3) bi[0]=V;
        if (which==4) std::fill(bi,bi+8,-1);
        require(!rad_mtp_couple_build_q(bi,bv,V,&q) && q.count==0,"malformed shortlist");
    }
}
struct CoupleFixture {
    static constexpr int M=2,R=2;
    int32_t idx[M][8];float val[M][8];int32_t card[M]={4,5};
    RadMtpCoupleDraw draw[M];RadMtpCoupleState state[R][M];
    CoupleFixture() {
        for (int r=0;r<M;++r) {
            for (int i=0;i<8;++i) {idx[r][i]=i<3 ? i : -1;val[r][i]=i<3 ? -static_cast<float>(i) : -INFINITY;}
            draw[r]={11,3,77u+static_cast<uint64_t>(r),card[r],1};
        }
        std::memset(state,0xab,sizeof state);
    }
    std::array<RadTensor,5> tensors() {return {{tensor(idx,RAD_I32,{M,8}),tensor(val,RAD_F32,{M,8}),tensor(card,RAD_I32,{M}),
        tensor(draw,RAD_U32,{M,8}),tensor(state,RAD_U32,{R,M,38})}};}
};
rad::Geometry couple_geometry(int round,int rounds=2,int M=2) {
    rad::Geometry g;g.set_i("M",M);g.set_i("n_vocab",10);g.set_i("round",round);g.set_i("max_rounds",rounds);return g;
}
void coupling_proposal() {
    CoupleFixture f;auto t=f.tensors();auto g=couple_geometry(2);auto a=args(t,g);
    f.draw[1].mode=0;
    check(rad_ref_mtp_couple_propose(&a,nullptr),"coupled proposal");
    const auto& s=f.state[1][0];
    require(s.status==1 && s.round==2 && s.count==3 && f.idx[0][0]==s.x && rad_mtp_couple_state_valid(s,10),"sampled row");
    require(!std::memcmp(&s.draw,&f.draw[0],sizeof s.draw) && s.pos==0,"state carries its draw, unbound");
    require(f.state[1][1].status==0 && f.idx[1][0]==0 && f.state[1][1].x==0 && f.state[1][1].count==0,"greedy row keeps column 0");
    uint8_t untouched[sizeof f.state[0]];std::memset(untouched,0xab,sizeof untouched);
    require(!std::memcmp(f.state[0],untouched,sizeof untouched),"other round untouched");
    for (int which=0;which<5;++which) {
        f=CoupleFixture{};
        if (which==0) f.card[0]=9;          // RadBatch::seq_ids names another card
        if (which==1) f.idx[0][2]=1;        // duplicate id
        if (which==2) f.val[0][1]=NAN;
        if (which==3) f.idx[0][1]=10;       // outside the vocabulary
        if (which==4) for (int i=0;i<8;++i) f.idx[0][i]=-1;
        check(rad_ref_mtp_couple_propose(&a,nullptr),"malformed proposal launch");
        const auto& m=f.state[1][0];
        require(m.status==-1 && m.count==0 && m.ids[0]==-1 && m.q[0]==0 && f.idx[0][0]==(which==4 ? -1 : 0),"malformed keeps greedy, status -1");
        require(f.state[1][1].status==1,"neighbouring row unaffected");
    }
    f=CoupleFixture{};f.val[0][1]=-1000;
    check(rad_ref_mtp_couple_propose(&a,nullptr),"underflow proposal");
    require(f.state[1][0].status==1 && f.state[1][0].count==2 && f.state[1][0].ids[1]==2,"underflow dropped");
    f=CoupleFixture{};
    auto g3=couple_geometry(2,3);a=args(t,g3);require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_SHAPE,"state rounds");
    auto gr=couple_geometry(3,2);a=args(t,gr);require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_UNSUPPORTED,"round beyond max_rounds");
    auto g9=couple_geometry(1,9);a=args(t,g9);require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_UNSUPPORTED,"more than eight rounds");
    a=args(t,g);t[2].shape[0]=1;require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_SHAPE,"seq_ids shorter than rows");
    t=f.tensors();t[4].dtype=RAD_I32;require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_DTYPE,"state dtype");
    t=f.tensors();t[3].data=reinterpret_cast<uint32_t*>(f.draw)+1;require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_INVAL,"draw alignment");
    t=f.tensors();t[4].data=t[3].data;require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_INVAL,"overlap");
    t=f.tensors();auto gm=couple_geometry(2,2,1);a=args(t,gm);require(rad_ref_mtp_couple_propose(&a,nullptr)==RAD_E_SHAPE,"live rows exceed M band");
}
void coupling_proposal_statistics() {
    CoupleFixture f;auto t=f.tensors();
    auto g1=couple_geometry(1),g2=couple_geometry(2);auto a1=args(t,g1),a2=args(t,g2);
    constexpr int N=40000;int hits[3]={},same_round=0,same_base=0;
    for (int i=0;i<N;++i) {
        f.draw[0].seed=f.draw[1].seed=static_cast<uint64_t>(i);f.draw[1].base=4;
        f.idx[0][0]=f.idx[1][0]=0;check(rad_ref_mtp_couple_propose(&a1,nullptr),"round 1");
        const int x=f.state[0][0].x;++hits[x];same_base+=x==f.state[0][1].x;
        f.idx[0][0]=f.idx[1][0]=0;check(rad_ref_mtp_couple_propose(&a2,nullptr),"round 2");
        same_round+=x==f.state[1][0].x;
    }
    const double z=1+std::exp(-1.0)+std::exp(-2.0),q[3]={1/z,std::exp(-1.0)/z,std::exp(-2.0)/z};
    for (int i=0;i<3;++i) near(hits[i]/double(N),q[i],0.012);
    const double collide=q[0]*q[0]+q[1]*q[1]+q[2]*q[2];
    near(same_round/double(N),collide,0.012);   // rounds are independent draft streams
    near(same_base/double(N),collide,0.012);    // so are two steps' pre-commit coordinates
    require(RAD_MTP_COUPLE_KEY_DRAFT+1!=RAD_SAMPLE_KEY_ACCEPT && RAD_MTP_COUPLE_KEY_DRAFT+1!=RAD_SAMPLE_KEY_RESID &&
            RAD_MTP_COUPLE_KEY_DRAFT+1!=RAD_SPECHUB_KEY_FIRST,"coupling key collision");
}
struct CoupleVerifyFixture {
    int32_t cand[4]={0,1,2,7};float val[4];RadSampleParams params{};RadMtpCoupleState s{};int32_t token=123;
    CoupleVerifyFixture() {
        const Target t=target({0,1,2,7},{0.375,0.125,0.25,0.25});std::copy(t.val.begin(),t.val.end(),val);
        s=cstate({0,1},{0.5,0.5},0);s.draw.seed=params.seed=11;s.pos=params.pos=5;
    }
    std::array<RadTensor,5> tensors() {return {{tensor(cand,RAD_I32,{1,4}),tensor(val,RAD_F32,{1,4}),tensor(&params,RAD_U32,{1,24}),
        tensor(&s,RAD_U32,{1,38}),tensor(&token,RAD_I32,{1})}};}
};
void coupling_verify() {
    CoupleVerifyFixture f;auto t=f.tensors();auto g=geometry(true);auto a=args(t,g);
    const Target tgt=target({0,1,2,7},{0.375,0.125,0.25,0.25});
    int accepted=0,residual=0;
    for (int seed=0;seed<400;++seed) {
        f.params.seed=f.s.draw.seed=static_cast<uint64_t>(seed);f.s.x=seed%2;f.token=123;
        check(rad_ref_mtp_couple_verify(&a,nullptr),"coupled verify");
        const int32_t want=decide(f.s,tgt,10,rad_spechub_uniform(f.params.seed,f.params.pos,RAD_SAMPLE_KEY_ACCEPT),
                                  rad_spechub_uniform(f.params.seed,f.params.pos,RAD_SAMPLE_KEY_RESID));
        require(f.token==want,"verify uses the ACCEPT and RESID streams at params.pos");
        (f.token==f.s.x ? accepted : residual)+=1;
    }
    require(accepted>0 && residual>0,"both coupling branches exercised");
    for (int which=0;which<6;++which) {
        CoupleVerifyFixture m;t=m.tensors();a=args(t,g);
        if (which==0) ++m.s.pos;
        if (which==1) ++m.s.draw.seed;
        if (which==2) m.s.status=0;
        if (which==3) m.s.ids[1]=m.s.ids[0];
        if (which==4) m.params.flags=RAD_SP_GREEDY;
        if (which==5) for (float& v:m.val) v=NAN;
        check(rad_ref_mtp_couple_verify(&a,nullptr),"unbound verify");
        require(m.token==123,"unbound, malformed or inapplicable row keeps the ordinary token");
    }
    CoupleVerifyFixture m;t=m.tensors();a=args(t,g);t[3].shape[1]=36;
    require(rad_ref_mtp_couple_verify(&a,nullptr)==RAD_E_SHAPE,"state width");
}
/* Real plugin GPU/reference check. Every launch uses separate device allocations,
 * includes padding guards and compares state/proposal/transport results. */
void gpu_compare(rad::Registry& reg,RadStream stream,const char* op,rad::Geometry& g,std::vector<RadTensor> host) {
    rad::SelectionTrace why;auto r=reg.resolve(op,g,RAD_DOMAIN_DEVICE,&why);require(r.row && r.row->info,"missing GPU row");
    require(r.row->plugin=="mine","wrong GPU plugin");check(reg.init_instance(r,0,1),"GPU init");
    std::vector<void*> allocations;std::vector<std::vector<uint32_t>> expected,got;
    auto device=host;
    try {
        for (size_t i=0;i<host.size();++i) {
            size_t words=1;for (uint32_t j=0;j<host[i].rank;++j) words+=static_cast<size_t>(host[i].shape[j]-1)*host[i].stride[j];
            expected.emplace_back(words+32,0xdeadbeefu);std::memcpy(expected.back().data()+16,host[i].data,words*4);got.emplace_back(words+32);
            void* ptr=rad_dev_alloc((words+32)*4,RAD_MEM_DEVICE);require(ptr!=nullptr,"GPU allocation");allocations.push_back(ptr);
            device[i].data=static_cast<uint32_t*>(ptr)+16;
            check(rad_memcpy_async(ptr,expected.back().data(),(words+32)*4,stream),"GPU upload");
        }
        RadArgs a{};a.t=device.data();a.n_t=static_cast<int>(device.size());a.p=r.geom.params();a.n_p=r.geom.n_params();a.instance=r.instance;
        check(r.row->info->launch(&a,stream),"GPU launch");
        RadArgs h=a;h.t=host.data();
        const RadLaunchFn ref=!std::strcmp(op,"spechub_propose") ? rad_ref_spechub_propose :
            !std::strcmp(op,"spechub_verify") ? rad_ref_spechub_verify :
            !std::strcmp(op,"mtp_couple_propose") ? rad_ref_mtp_couple_propose : rad_ref_mtp_couple_verify;
        check(ref(&h,nullptr),"reference launch");
        for (size_t i=0;i<host.size();++i) {
            std::memcpy(expected[i].data()+16,host[i].data,(expected[i].size()-32)*4);
            check(rad_memcpy_async(got[i].data(),allocations[i],got[i].size()*4,stream),"GPU download");
        }
        check(rad_stream_sync(stream),"GPU sync");
        for (size_t i=0;i<host.size();++i) {
            if (std::strcmp(op,"spechub_propose")==0 && i==4) {
                // Host/device libm need not return identical final FP64 bits.
                auto* x=reinterpret_cast<RadSpechubState*>(expected[i].data()+16);
                auto* y=reinterpret_cast<RadSpechubState*>(got[i].data()+16);
                for (int64_t row=0;row<host[i].shape[0];++row) {
                    require(x[row].status==y[row].status && x[row].count==y[row].count,"GPU state status");
                    for (int j=0;j<8;++j) {near(x[row].q[j],y[row].q[j]);y[row].q[j]=x[row].q[j];}
                }
            }
            if (std::strcmp(op,"mtp_couple_propose")==0 && i==4) {
                // Dense [rounds, rows] state plane; FP64 q may differ in its last bits only.
                auto* x=reinterpret_cast<RadMtpCoupleState*>(expected[i].data()+16);
                auto* y=reinterpret_cast<RadMtpCoupleState*>(got[i].data()+16);
                for (int64_t k=0;k<host[i].shape[0]*host[i].shape[1];++k) {
                    require(x[k].status==y[k].status && x[k].count==y[k].count && x[k].x==y[k].x,"GPU coupling state");
                    for (int j=0;j<8;++j) {near(x[k].q[j],y[k].q[j]);y[k].q[j]=x[k].q[j];}
                }
            }
            require(expected[i]==got[i],"GPU/reference/padding mismatch");
        }
    } catch (...) {for (void* ptr:allocations) rad_dev_free(ptr,RAD_MEM_DEVICE);throw;}
    for (void* ptr:allocations) rad_dev_free(ptr,RAD_MEM_DEVICE);
}
void coupling_fallback_verify(rad::Registry* reg=nullptr,RadStream stream=nullptr) {
    int32_t ids[8]={0,1,2,3,4,5,6,7};
    float values[8]={0,static_cast<float>(-std::log(2.0)),-37,-37,-37,-37,-37,-37};
    int32_t reordered[8]={2,3,4,5,6,7,0,1};float logits[8];
    for (int i=0;i<8;++i) logits[i]=values[reordered[i]];
    RadMtpCoupleState state{};state.status=1;state.round=1;state.draw.mode=1;
    require(rad_mtp_couple_build_q(ids,values,10,&state),"cancellation shortlist");
    RadSampleParams params{};params.pos=state.pos=5;
    // Inverted splitmix: RESID at position 5 is 0.666666675, above the FP64 CDF through id 0
    // (0.6666666670899233), but truncating to 24 bits puts the uniform below that boundary.
    params.seed=state.draw.seed=0xdfddea36362e095full;
    const double u=rad_spechub_uniform(params.seed,params.pos,RAD_SAMPLE_KEY_RESID);
    require(u>0.66666667 && u<0.66666668,"precision-sensitive residual uniform");
    int32_t token=0;
    std::array<RadTensor,5> ts={tensor(reordered,RAD_I32,{1,8}),tensor(logits,RAD_F32,{1,8}),
        tensor(&params,RAD_U32,{1,24}),tensor(&state,RAD_U32,{1,38}),tensor(&token,RAD_I32,{1})};
    auto g=geometry(true);g.set_i("n_cand",8);auto a=args(ts,g);
    for (int x:ids) {
        state.x=x;token=0;
        require(rad_mtp_couple_state_valid(state,10),"generated cancellation state");
        if (reg) gpu_compare(*reg,stream,"mtp_couple_verify",g,{ts.begin(),ts.end()});
        else check(rad_ref_mtp_couple_verify(&a,nullptr),"FP64 cancellation verify");
        require(token==1,"numerical fallback samples FP64 P, independent of x and ordinary token");
    }
}
/* Real GPU kernels against the host reference for both coupling ops. Every expected status and
 * token outcome is asserted on its own, so a kernel that writes status 0 everywhere or keeps every
 * ordinary token fails even though GPU and reference would agree. */
void couple_gpu_tests(rad::Registry& reg,RadStream stream) {
    {
        constexpr int M=6;
        std::vector<std::array<int32_t,8>> idx(M);std::vector<std::array<float,8>> val(M);
        std::vector<int32_t> card(M);std::vector<RadMtpCoupleDraw> draw(M);std::vector<RadMtpCoupleState> state(2*M);
        for (int r=0;r<M;++r) {
            for (int i=0;i<8;++i) {idx[r][i]=i<3 ? i : -1;val[r][i]=i<3 ? -static_cast<float>(i) : -INFINITY;}
            card[r]=r;draw[r]={static_cast<uint64_t>(31*r),9,100u+static_cast<uint64_t>(r),r,1};
        }
        draw[1].mode=0;card[2]=7;val[3][1]=NAN;val[4][1]=-1000;idx[5][2]=1;
        std::vector<RadTensor> ts={tensor(idx.data(),RAD_I32,{M,8}),tensor(val.data(),RAD_F32,{M,8}),tensor(card.data(),RAD_I32,{M}),
            tensor(draw.data(),RAD_U32,{M,8}),tensor(state.data(),RAD_U32,{2,M,38})};
        auto g=couple_geometry(2,2,M);gpu_compare(reg,stream,"mtp_couple_propose",g,ts);
        const int want[M]={1,0,-1,-1,1,-1};   // sampled, greedy, wrong card, NaN, underflow, duplicate
        for (int r=0;r<M;++r) {
            const auto& s=state[M+r];
            require(s.status==want[r] && s.round==2,"GPU coupling status per branch");
            require(idx[r][0]==(want[r]==1 ? s.x : 0),"column 0 changes only for sampled rows");
            require(state[r].status==0 && state[r].count==0,"other round untouched");
        }
        require(state[M+4].count==2,"GPU drops underflowed Q entries");
    }
    {
        constexpr int M=3;
        std::array<std::array<int32_t,8>,M> ids{};
        std::array<std::array<float,8>,M> val{};
        std::array<RadMtpCoupleState,M> state{};
        std::array<RadSampleParams,M> params{};
        std::array<int32_t,M> token{{73,73,73}};
        for (int r=0;r<M;++r) {
            ids[r]={{0,2,0,1,-1,-1,-1,-1}};
            val[r]={{-1000,-1,0,-3,-INFINITY,-INFINITY,-INFINITY,-INFINITY}};
            state[r]=cstate({0,1},{0.5,0.5},1);
            params[r].seed=state[r].draw.seed=26;
            params[r].pos=state[r].pos=5;
        }
        val[1][0]=-INFINITY;
        state[2].status=0;
        std::vector<RadTensor> ts={tensor(ids.data(),RAD_I32,{M,8}),tensor(val.data(),RAD_F32,{M,8}),
            tensor(params.data(),RAD_U32,{M,24}),tensor(state.data(),RAD_U32,{M,38}),
            tensor(token.data(),RAD_I32,{M})};
        auto g=geometry(true,M);g.set_i("n_cand",8);
        gpu_compare(reg,stream,"mtp_couple_verify",g,ts);
        require(token[0]==0 && token[1]==2 && token[2]==73,
                "first eligible Q duplicate includes finite underflow, skips -inf, and keeps unbound token");

        for (int n:{129,1024,1025}) {
            std::vector<int32_t> wide_ids(M*n);
            std::vector<float> wide_val(M*n,-1000);
            std::array<RadMtpCoupleState,M> wide_state{};
            std::array<RadSampleParams,M> wide_params{};
            std::array<int32_t,M> wide_token{{73,73,73}};
            for (int r=0;r<M;++r) {
                for (int j=0;j<n;++j) wide_ids[r*n+j]=j+1;
                wide_val[r*n+(r==1 ? 0 : n-1)]=0;
                wide_state[r]=cstate({0},{1},0);
                wide_params[r].seed=wide_state[r].draw.seed=11;
                wide_params[r].pos=wide_state[r].pos=5;
            }
            wide_state[2].status=0;
            std::vector<RadTensor> vt={tensor(wide_ids.data(),RAD_I32,{M,n}),
                tensor(wide_val.data(),RAD_F32,{M,n}),
                tensor(wide_params.data(),RAD_U32,{M,24}),
                tensor(wide_state.data(),RAD_U32,{M,38}),
                tensor(wide_token.data(),RAD_I32,{M})};
            auto vg=geometry(true,M);vg.set_i("n_cand",n);vg.set_i("n_vocab",2048);
            gpu_compare(reg,stream,"mtp_couple_verify",vg,vt);
            require(wide_token[0]==n && wide_token[1]==1 && wide_token[2]==73,
                    "candidate tail and wide geometry preserve residual tokens and unbound row");
        }
    }
    constexpr int N=8192;
    std::vector<std::array<int32_t,8>> idx(N);std::vector<std::array<float,8>> val(N);
    std::vector<int32_t> card(N);std::vector<RadMtpCoupleDraw> draw(N);std::vector<RadMtpCoupleState> state(N);
    for (int r=0;r<N;++r) {
        for (int i=0;i<8;++i) {idx[r][i]=i<3 ? i : -1;val[r][i]=i<3 ? -static_cast<float>(i) : -INFINITY;}
        card[r]=r;draw[r]={static_cast<uint64_t>(r),3,1000u+static_cast<uint64_t>(r),r,1};
    }
    auto propose=[&] {
        std::vector<RadTensor> ts={tensor(idx.data(),RAD_I32,{N,8}),tensor(val.data(),RAD_F32,{N,8}),tensor(card.data(),RAD_I32,{N}),
            tensor(draw.data(),RAD_U32,{N,8}),tensor(state.data(),RAD_U32,{1,N,38})};
        auto g=couple_geometry(1,1,N);gpu_compare(reg,stream,"mtp_couple_propose",g,ts);
    };
    std::vector<std::array<int32_t,4>> cand(N,{{0,1,2,7}});std::vector<std::array<float,4>> logit(N,{{-2,-1,0,1}});
    std::vector<RadSampleParams> params(N);std::vector<int32_t> token(N,-5);std::vector<RadMtpCoupleState> bound;
    auto verify=[&] {
        std::vector<RadTensor> vt={tensor(cand.data(),RAD_I32,{N,4}),tensor(logit.data(),RAD_F32,{N,4}),
            tensor(params.data(),RAD_U32,{N,24}),tensor(bound.data(),RAD_U32,{N,38}),tensor(token.data(),RAD_I32,{N})};
        auto g=geometry(true,N);gpu_compare(reg,stream,"mtp_couple_verify",g,vt);
    };
    propose();
    int hits[3]={};for (const auto& s:state) {require(s.status==1,"GPU sampled state");++hits[s.x];}
    for (int i=0;i<3;++i) near(hits[i]/double(N),state[0].q[i],0.025);
    // P != Q: the GPU marginal is P; every eighth row is unbound or malformed and keeps its token.
    bound=state;
    for (int r=0;r<N;++r) {
        params[r].seed=draw[r].seed;params[r].pos=11;bound[r].pos=11;
        if (r%8!=7) continue;
        const int w=(r/8)%4;
        if (w==0) bound[r].ids[1]=bound[r].ids[0];
        if (w==1) ++bound[r].pos;
        if (w==2) ++bound[r].draw.seed;
        if (w==3) bound[r].status=0;
    }
    verify();
    double sum=0;for (float v:logit[0]) sum+=std::exp(static_cast<double>(v)-1);
    int coupled[10]={},valid=0;
    for (int r=0;r<N;++r) {
        if (r%8==7) {require(token[r]==-5,"GPU keeps the ordinary token for an unbound or malformed state");continue;}
        require(token[r]>=0 && token[r]<10,"GPU coupled token");++coupled[token[r]];++valid;
    }
    for (int k=0;k<4;++k) near(coupled[cand[0][k]]/double(valid),std::exp(static_cast<double>(logit[0][k])-1)/sum,0.025);
    // P == Q (uniform, exact on any libm): the coupled verifier always emits x, where the
    // pre-existing independent ordinary sample (sample_pick's FP32 CDF at KEY_PICK) often differs.
    for (int r=0;r<N;++r) {
        for (int i=0;i<8;++i) {idx[r][i]=i<4 ? 2+2*i : -1;val[r][i]=i<4 ? 0.0f : -INFINITY;}
        draw[r].base=5;cand[r]={{2,4,6,8}};logit[r]={{0,0,0,0}};
    }
    propose();
    bound=state;int differs=0;
    for (int r=0;r<N;++r) {
        require(state[r].status==1 && state[r].count==4 && state[r].q[0]==0.25,"uniform Q");
        bound[r].pos=params[r].pos;
        const float u=rad_sample_uniform01(params[r].seed^RAD_SAMPLE_KEY_PICK,params[r].pos);
        float cum=0;int32_t pick=8;
        for (int k=0;k<4;++k) {cum+=0.25f;if (u<cum) {pick=2+2*k;break;}}
        token[r]=pick;differs+=pick!=state[r].x;
    }
    verify();
    for (int r=0;r<N;++r) require(token[r]==state[r].x,"P=Q: the coupled verifier emits the proposal");
    require(differs>N/2,"P=Q: the independent ordinary sample would differ");
}
void gpu_tests(int argc,char** argv) {
    require(argc==7 && std::string(argv[1])=="--mine" && std::string(argv[3])=="--ref" && std::string(argv[5])=="--libr4d","usage: spechub_test [--mine PATH --ref PATH --libr4d PATH]");
    check(rad_dev_set(0),"set GPU");RadDeviceProps props{};check(rad_dev_props(0,&props),"GPU props");require(!props.is_host_backend,"real GPU required");
    rad::Registry reg;reg.set_code_device(0);check(reg.load_plugin(argv[2],0),"load mine");check(reg.load_plugin(argv[4],1),"load ref");check(reg.load_plugin(argv[6],2),"load libr4d");
    RadStream stream=nullptr;check(rad_stream_create(&stream,0),"GPU stream");
    try {
        for (int transport:{0,1}) {
        Fixture f;
        for (int variant=0;variant<10;++variant) {
            f.draw.seed=variant*79;f.draft=variant==7 ? -1 : 0;
            f.params.seed=f.draw.seed;
            if (variant==8) {float value=-INFINITY;for (int i=1;i<3;++i) std::memcpy(&f.gathered[2*i+1],&value,4);}
            if (variant==9) {float value=-1000;std::memcpy(&f.gathered[3],&value,4);}
            auto p=f.proposal();auto gp=geometry(false);gpu_compare(reg,stream,"spechub_propose",gp,{p.begin(),p.end()});
            auto v=f.verifier();auto gv=geometry(true);gv.set_i("transport",transport);
            if (variant==1) ++f.s.draw.seed;
            if (variant==2) ++f.s.draw.pos;
            if (variant==3) f.s.root=-1;
            if (variant==4) f.s.pair[0]=8;
            if (variant==5) f.s.status=0;
            if (variant==6) for (int i=0;i<4;++i) {f.target_ids[i]=6+i;f.target_val[i]=i==0 ? 0 : -INFINITY;}
            gpu_compare(reg,stream,"spechub_verify",gv,{v.begin(),v.end()});
        }
        }
        constexpr int N=8192;Fixture base;
        std::vector<std::array<uint32_t,34>> gathered(N);
        for (auto& row:gathered) std::copy(base.gathered.begin(),base.gathered.end(),row.begin());
        std::vector<RadSpechubDraw> draw(N);std::vector<RadSpechubState> states(N);
        std::vector<int32_t> draft(N,0);std::vector<std::array<int32_t,9>> packed(N);
        std::vector<std::array<int32_t,10>> idx(N);std::vector<std::array<float,10>> val(N);
        for (int i=0;i<N;++i) draw[i]={static_cast<uint64_t>(i),3,77};
        std::vector<RadTensor> ts={tensor(gathered.data(),RAD_F32,{N,16,2},2),tensor(draw.data(),RAD_U32,{N,6}),tensor(&base.root,RAD_I32,{1}),
            tensor(draft.data(),RAD_I32,{N}),tensor(states.data(),RAD_U32,{N,36}),tensor(packed.data(),RAD_I32,{N,7},2),tensor(idx.data(),RAD_I32,{N,8},2),tensor(val.data(),RAD_F32,{N,8},2)};
        auto g=geometry(false,N);gpu_compare(reg,stream,"spechub_propose",g,ts);
        int counts[3]={};for (const auto& s:states) {require(s.status==1,"GPU statistical state");++counts[s.pair[0]];}
        for (int i=0;i<3;++i) near(counts[i]/double(N),states[0].q[i],0.025);
        std::vector<std::array<int32_t,4>> target_ids(N,{{0,1,2,7}});
        std::vector<std::array<float,4>> target_val(N,{{-2,-1,0,1}});
        std::vector<RadSampleParams> params(N);std::vector<int32_t> output(N,-99);
        for (int i=0;i<N;++i) {params[i].seed=draw[i].seed;params[i].pos=draw[i].pos;}
        std::vector<RadTensor> vt={tensor(target_ids.data(),RAD_I32,{N,4}),tensor(target_val.data(),RAD_F32,{N,4}),
            tensor(params.data(),RAD_U32,{N,24}),tensor(states.data(),RAD_U32,{N,36}),tensor(output.data(),RAD_I32,{N})};
        for (int transport:{0,1}) {
            auto vg=geometry(true,N);vg.set_i("transport",transport);gpu_compare(reg,stream,"spechub_verify",vg,vt);
            int counts_target[10]={};for (int tok:output) {require(tok>=0 && tok<10,"GPU target sample");++counts_target[tok];}
            double sum=0;for (float v:target_val[0]) sum+=std::exp(static_cast<double>(v)-1);
            for (int k=0;k<4;++k) near(counts_target[target_ids[0][k]]/double(N),std::exp(static_cast<double>(target_val[0][k])-1)/sum,0.025);
        }
        couple_gpu_tests(reg,stream);
        coupling_fallback_verify(&reg,stream);
    } catch (...) {rad_stream_destroy(stream);throw;}
    rad_stream_destroy(stream);
}
}
int main(int argc,char** argv) {
    try {
        distribution_tests();host_boundaries(0);host_boundaries(1);proposal_statistics();
        coupling_marginals();coupling_proposal();coupling_proposal_statistics();coupling_verify();
        coupling_fallback_verify();
        if (argc>1) gpu_tests(argc,argv);
        std::printf("ALL_PASS SpecHub independent_oracle boundaries proposal_statistics MTPcouple marginals "
                    "proposal verify gpu=%s\n",argc>1 ? "exercised" : "not_requested");return 0;
    } catch (const std::exception& e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
