/* Host-only build: use mine_ref's reference objects; C++20, -ffp-contract=off.
 * No HIP dependency.
 * Exact branch-vs-isolated-path checks are NOT a GPU parity claim. The unchanged
 * scalar oracle supplies an independent exact sparse recurrence witness below. */
#include "rad_test.h"
#include "../mine/tree_gdn.h"
#include "../mine/ref_gdn_tree.h"
#include "../mine/ref_ple_tree.h"
#include "../libref/ref_tree_state.h"
/* The C oracle's macro is not the C++ CandidateTree namespace constant. */
#undef RAD_TREE_MAX_NODES
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <sstream>
#include <any>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <memory>
#include <unordered_map>
#include "diag.h"
#define private public
#include "sched/scheduler.h"
#undef private
#include "sched/tree_state.h"
#include "sample/accept.h"
#include "../arch/common/rad_qsa.h"
#include "../arch/common/rad_block_gdn_fp8.h"
#include "../arch/common/rad_block_ple.h"

namespace {
uint16_t pack(float x) { uint32_t u;std::memcpy(&u,&x,4);return (uint16_t)((u+0x8000U)>>16); }
float unpack(uint16_t x) { uint32_t u=(uint32_t)x<<16;float f;std::memcpy(&f,&u,4);return f; }
RadTensor tensor(void* data,uint32_t dt,std::initializer_list<int64_t> shape) {
    RadTensor t{};t.data=data;t.dtype=dt;
    for (int64_t v:shape) t.shape[t.rank++]=v;
    int64_t stride=1;
    for (int i=(int)t.rank-1;i>=0;--i) { t.stride[i]=stride;stride*=t.shape[i]; }
    return t;
}
RadParam pi(const char* key,int64_t value) { RadParam p{};p.key=key;p.kind=RAD_P_INT;p.ival=value;return p; }
RadParam pf(const char* key,double value) { RadParam p{};p.key=key;p.kind=RAD_P_F64;p.dval=value;return p; }

/* Tokens/Cap default to the original 24-token, 16-node fixture; the wide tests
 * instantiate RAD_GDN_TREE_WIDE_CAP with a path pitch of Cap columns. */
template <int Tokens=24,int Cap=16>
struct FixtureT {
    static constexpr int H=2,D=512,T=Tokens,Slots=4,XP=D+17,SP=H*128*128+31,CAP=Cap;
    int width=4,ring=0,r=5,capacity=Cap;
    std::vector<uint16_t> x,w,bias,hist,q,k,v,z,ab,o,raw;
    std::vector<float> state,factors,alog,dt,norm;
    std::vector<int32_t> cu{3,10,15},parent,idx{2,91,0,92},ctx{4,9},path,len{4,1};
    FixtureT(int ring_=0,int width_=4,int r_=5):width(width_),ring(ring_),r(r_),
        x(T*XP),w(D*width),bias(D),hist(Slots*D*(r+1)),q(T*128,0x7fc1),k(q),
        v(T*H*128,0x7fc1),z(T*H*128),ab(T*2*H),o(v),raw(T*D,0x7fc1),
        state(Slots*SP),factors(T*H*257,-123.0f),alog(H,-2.0f),dt(H,-0.5f),norm(128,1),
        parent(T,-77),path(2*CAP,-9) {
        for (size_t i=0;i<x.size();++i) x[i]=pack((int(i%37)-18)*0.03125f);
        for (size_t i=0;i<w.size();++i) w[i]=pack((int(i%7)-3)*0.0625f);
        for (size_t i=0;i<hist.size();++i) hist[i]=pack((int(i%23)-11)*0.015625f);
        for (size_t i=0;i<state.size();++i) state[i]=(int(i%31)-15)*0.0009765625f;
        for (size_t i=0;i<ab.size();++i) ab[i]=pack((int(i%9)-4)*0.125f);
        for (size_t i=0;i<z.size();++i) z[i]=pack((int(i%13)-6)*0.125f);
        const int first[]={-1,0,0,1,2,1,4},second[]={-1,0,1,0,3};
        std::copy(first,first+7,parent.begin()+3);std::copy(second,second+5,parent.begin()+10);
        path[0]=0;path[1]=2;path[2]=4;path[3]=6;path[CAP]=0;
    }
    int history_offset(int slot,int ch,int pos) const { return (slot*D+ch)*(r+1)+pos; }
    int state_offset(int slot,int h,int row,int k_) const { return slot*SP+(h*128+row)*128+k_; }
    std::vector<RadParam> params() const {
        return {pi("q_len",capacity),pi("head_k",128),pi("head_v",128),pi("n_head_k",1),pi("n_head_v",H),
            pi("conv_width",width),pi("node_capacity",capacity),pi("history_ring",ring),pi("n_seq",2),
            pf("scale",0.125),pf("eps",1e-6),pf("l2_eps",1e-6),pf("softplus_thr",20)};
    }
    std::vector<RadTensor> operands(int op,int seq=0,int ns=2,bool linear=false) {
        RadTensor xx=tensor(x.data(),RAD_BF16,{T,D});xx.stride[0]=XP;
        RadTensor hh=tensor(hist.data(),RAD_BF16,{Slots,D,r});hh.stride[1]=r+1;hh.stride[0]=D*(r+1);
        RadTensor ss=tensor(state.data(),RAD_F32,{Slots,H,128,128});ss.stride[0]=SP;
        RadTensor ii=tensor(idx.data()+seq*2,RAD_I32,{ns,2});
        RadTensor cc=tensor(cu.data()+seq,RAD_I32,{ns+1});
        RadTensor pp=linear ? RadTensor{} : tensor(parent.data(),RAD_I32,{T});
        RadTensor cx=tensor(ctx.data()+seq,RAD_I32,{ns});
        RadTensor qq=tensor(q.data(),RAD_BF16,{T,128}),kk=tensor(k.data(),RAD_BF16,{T,1,128});
        RadTensor vv=tensor(v.data(),RAD_BF16,{T,H,128});
        RadTensor ff=tensor(factors.data(),RAD_F32,{T,H,257}),rr=tensor(raw.data(),RAD_BF16,{T,D});
        if (op==0) return {xx,tensor(w.data(),RAD_BF16,{D,width}),tensor(bias.data(),RAD_BF16,{D}),
            hh,ii,cc,pp,cx,qq,kk,vv,rr};
        if (op==1) {
            RadTensor aa=tensor(ab.data(),RAD_BF16,{T,H});aa.stride[0]=2*H;
            RadTensor bb=aa;bb.data=ab.data()+H;
            return {qq,kk,vv,aa,bb,tensor(alog.data(),RAD_F32,{H}),tensor(dt.data(),RAD_F32,{H}),
                ss,ii,cc,pp,tensor(z.data(),RAD_BF16,{T,H*128}),tensor(norm.data(),RAD_F32,{128}),
                tensor(o.data(),RAD_BF16,{T,H*128}),ff};
        }
        return {ss,ii,ff,hh,ii,rr,tensor(path.data()+seq*CAP,RAD_I32,{ns,CAP}),
            tensor(len.data()+seq,RAD_I32,{ns}),cx,cc,pp};
    }
    int run(int op,int seq=0,int ns=2,bool linear=false) {
        auto p=params();auto t=operands(op,seq,ns,linear);
        RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
        return rad_ref_gdn_tree_at(op)->launch(&a,nullptr);
    }
};
using Fixture=FixtureT<>;

template <class F>
std::vector<int> ancestors(const F& f,int seq,int node) {
    std::vector<int> p;
    for (;node>=0;node=f.parent[f.cu[seq]+node]) p.push_back(node);
    std::reverse(p.begin(),p.end());return p;
}
/* Independent execution schedule: reset to the committed anchor and issue ONE
 * token per call, committing it immediately. No tree replay factors are read by
 * recurrence and no sibling can occur in this schedule. */
template <class F>
F isolated_path(const F& original,int seq,const std::vector<int>& path) {
    F f=original;
    const int bos=original.cu[seq];
    for (int node:path) {
        const int tok=bos+node;
        f.cu[seq]=tok;f.cu[seq+1]=tok+1;f.parent[tok]=-1;
        f.path[seq*F::CAP]=0;f.len[seq]=1;
        CHECK(f.run(0,seq,1)==RAD_OK);CHECK(f.run(1,seq,1)==RAD_OK);CHECK(f.run(2,seq,1)==RAD_OK);
        ++f.ctx[seq];
    }
    return f;
}
template <class F>
void compare_rows(const F& a,const F& b,int tok) {
    CHECK(std::memcmp(a.q.data()+tok*128,b.q.data()+tok*128,128*2)==0);
    CHECK(std::memcmp(a.k.data()+tok*128,b.k.data()+tok*128,128*2)==0);
    CHECK(std::memcmp(a.v.data()+tok*256,b.v.data()+tok*256,256*2)==0);
    CHECK(std::memcmp(a.o.data()+tok*256,b.o.data()+tok*256,256*2)==0);
    CHECK(std::memcmp(a.factors.data()+tok*F::H*257,b.factors.data()+tok*F::H*257,
                      F::H*257*sizeof(float))==0);
}
void branching_case(int ring) {
    Fixture original(ring),tree=original;
    CHECK(tree.run(0)==RAD_OK);CHECK(tree.run(1)==RAD_OK);
    CHECK(tree.state==original.state);CHECK(tree.hist==original.hist);
    for (int seq=0;seq<2;++seq) for (int node=0;node<original.cu[seq+1]-original.cu[seq];++node) {
        Fixture serial=isolated_path(original,seq,ancestors(original,seq,node));
        compare_rows(tree,serial,original.cu[seq]+node);
    }
    CHECK(tree.run(2)==RAD_OK);
    for (int seq=0;seq<2;++seq) {
        std::vector<int> winner(original.path.begin()+seq*16,original.path.begin()+seq*16+original.len[seq]);
        Fixture serial=isolated_path(original,seq,winner);
        const int slot=original.idx[seq*2];
        CHECK(std::memcmp(tree.state.data()+slot*Fixture::SP,serial.state.data()+slot*Fixture::SP,
                          Fixture::SP*sizeof(float))==0);
        CHECK(std::memcmp(tree.hist.data()+slot*Fixture::D*(tree.r+1),serial.hist.data()+slot*Fixture::D*(tree.r+1),
                          Fixture::D*(tree.r+1)*sizeof(uint16_t))==0);
    }
    /* Unused state/history slots, token-prefix/suffix rows and padding survive. */
    for (int slot:{1,3}) {
        CHECK(std::memcmp(tree.state.data()+slot*Fixture::SP,original.state.data()+slot*Fixture::SP,
                          Fixture::SP*sizeof(float))==0);
        CHECK(std::memcmp(tree.hist.data()+slot*Fixture::D*(tree.r+1),original.hist.data()+slot*Fixture::D*(tree.r+1),
                          Fixture::D*(tree.r+1)*sizeof(uint16_t))==0);
    }
    for (int tok:{0,2,15,23}) compare_rows(tree,original,tok);
    for (int tok=3;tok<15;++tok) for (int ch=0;ch<Fixture::D;++ch)
        CHECK(tree.raw[tok*Fixture::D+ch]==original.x[tok*Fixture::XP+ch]);
}

/* Exercise the published host row with runtime-owned scratch, not its launch
 * function in isolation. The independent quantizer below reads the separately
 * executed unfused BF16 output; it does not call the fused quantization helper. */
template <class F=Fixture>
struct FusedHostCallT {
    static constexpr int QP=F::H*128+8,SC=F::H+3;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<uint64_t> scratch;
    std::vector<RadParam> params;
    std::vector<RadTensor> tensors;
    explicit FusedHostCallT(F& f,int ns=2)
        :codes(F::T*QP,0x5a),scales(F::T*SC,-123.0f),params(f.params()) {
        params.push_back(pi("max_tokens",F::T));
        auto conv=f.operands(0,0,ns),recurrent=f.operands(1,0,ns);
        auto q=tensor(codes.data(),RAD_I8,{F::T,F::H*128});q.stride[0]=QP;
        auto s=tensor(scales.data(),RAD_F32,{F::T,F::H});s.stride[0]=SC;
        tensors={conv[0],conv[1],conv[2],conv[3],conv[4],conv[5],conv[6],conv[7],
            recurrent[3],recurrent[4],recurrent[5],recurrent[6],recurrent[7],recurrent[8],
            recurrent[11],recurrent[12],conv[11],recurrent[14],q,s};
    }
    RadArgs args() {
        RadArgs a{};a.t=tensors.data();a.n_t=int(tensors.size());
        a.p=params.data();a.n_p=int(params.size());
        a.scratch=scratch.empty() ? nullptr : scratch.data();
        a.scratch_bytes=int64_t(scratch.size()*sizeof(uint64_t));a.world_size=1;
        return a;
    }
};
using FusedHostCall=FusedHostCallT<>;

void fused_inputs_unchanged(const Fixture& actual,const Fixture& initial) {
    CHECK(actual.x==initial.x);CHECK(actual.w==initial.w);CHECK(actual.bias==initial.bias);
    CHECK(actual.hist==initial.hist);CHECK(actual.state==initial.state);
    CHECK(actual.ab==initial.ab);CHECK(actual.z==initial.z);CHECK(actual.norm==initial.norm);
    CHECK(actual.alog==initial.alog);CHECK(actual.dt==initial.dt);
    CHECK(actual.cu==initial.cu);CHECK(actual.parent==initial.parent);
    CHECK(actual.idx==initial.idx);CHECK(actual.ctx==initial.ctx);
    CHECK(actual.q==initial.q);CHECK(actual.k==initial.k);CHECK(actual.v==initial.v);CHECK(actual.o==initial.o);
}

void fused_host_case(int ring,bool zero) {
    Fixture initial(ring);
    if (zero) std::fill(initial.z.begin(),initial.z.end(),uint16_t(0));
    // Copy the baseline before either arm writes; no tensor aliases cross arms.
    Fixture reference=initial,actual=initial;
    REQUIRE(reference.run(0)==RAD_OK);REQUIRE(reference.run(1)==RAD_OK);
    FusedHostCall call(actual);
    const auto* row=rad_ref_gdn_tree_at(RAD_GDN_TREE_FUSED_I8G);
    REQUIRE(row && row->launch && row->scratch && row->domain==RAD_DOMAIN_HOST);
    REQUIRE(std::strcmp(row->name,"ref_gdn_conv_recurrent_tree_i8g")==0);
    auto a=call.args();
    const int64_t need=row->scratch(&a);
    REQUIRE(need>0);
    CHECK(need==int64_t(Fixture::T)*512*(1+Fixture::H));
    call.scratch.resize(size_t((need+7)/8));a=call.args();
    REQUIRE(row->launch(&a,nullptr)==RAD_OK);
    CHECK(actual.raw==reference.raw);
    CHECK(std::memcmp(actual.factors.data(),reference.factors.data(),actual.factors.size()*sizeof(float))==0);
    fused_inputs_unchanged(actual,initial);
    std::vector<uint8_t> expected_codes(Fixture::T*FusedHostCall::QP,0x5a);
    std::vector<float> expected_scales(Fixture::T*FusedHostCall::SC,-123.0f);
    int groups=0,zero_groups=0;
    for (int seq=0;seq<2;++seq) for (int tok=initial.cu[seq];tok<initial.cu[seq+1];++tok) {
        for (int head=0;head<Fixture::H;++head) {
            float maximum=0;
            for (int d=0;d<128;++d)
                maximum=std::max(maximum,std::abs(unpack(reference.o[(tok*Fixture::H+head)*128+d])));
            const float scale=maximum>0 ? maximum*(1.0f/127.0f) : 1.0f;
            const float inverse=1.0f/scale;
            expected_scales[tok*FusedHostCall::SC+head]=scale;
            for (int d=0;d<128;++d) {
                const float value=unpack(reference.o[(tok*Fixture::H+head)*128+d])*inverse;
                const int code=int(std::nearbyint(std::clamp(value,-127.0f,127.0f)));
                expected_codes[tok*FusedHostCall::QP+head*128+d]=uint8_t(code);
            }
            ++groups;if (maximum==0) ++zero_groups;
        }
    }
    CHECK(groups==24);
    CHECK(!zero || zero_groups==groups);
    CHECK(zero || zero_groups<groups);
    CHECK(call.codes==expected_codes);
    CHECK(std::memcmp(call.scales.data(),expected_scales.data(),call.scales.size()*sizeof(float))==0);
}
} // namespace

TEST(gdn_tree_branching_canonical_history) { branching_case(0); }
TEST(gdn_tree_branching_ring_wrap) { branching_case(1); }

TEST(gdn_tree_fused_host_branching_ring_and_zero_groups) {
    for (int ring:{0,1}) for (bool zero:{false,true}) fused_host_case(ring,zero);
}

TEST(gdn_tree_fused_host_refuses_capacity_and_scratch_without_writes) {
    const auto* row=rad_ref_gdn_tree_at(RAD_GDN_TREE_FUSED_I8G);
    REQUIRE(row && row->launch && row->scratch);
    for (int fault=0;fault<5;++fault) {
        Fixture initial(1),actual=initial;
        FusedHostCall call(actual);
        auto a=call.args();
        const int64_t need=row->scratch(&a);
        REQUIRE(need>0);
        call.scratch.resize(size_t((need+7)/8));
        if (fault==0) call.params.pop_back(); // Required max_tokens absent.
        if (fault==1) call.params.back().ival=Fixture::T-1;
        if (fault==2) call.params.back().ival=0;
        a=call.args();
        if (fault==3) a.scratch_bytes=need-1;
        if (fault==4) {a.scratch=nullptr;a.scratch_bytes=0;}
        const int rc=row->launch(&a,nullptr);
        CHECK(rc==(fault<3 ? RAD_E_SHAPE : RAD_E_SCRATCH));
        CHECK(actual.raw==initial.raw);CHECK(actual.factors==initial.factors);
        CHECK(call.codes==std::vector<uint8_t>(Fixture::T*FusedHostCall::QP,0x5a));
        CHECK(call.scales==std::vector<float>(Fixture::T*FusedHostCall::SC,-123.0f));
        fused_inputs_unchanged(actual,initial);
    }
}

TEST(gdn_tree_sibling_poison_and_nonprefix_commit) {
    Fixture base(1),poison=base,flat=base;
    for (int node:{1,3,5}) {
        const int tok=3+node;
        for (int ch=0;ch<Fixture::D;++ch) poison.x[tok*Fixture::XP+ch]=pack(4.0f+ch*0.125f);
        for (int ch=0;ch<4;++ch) poison.ab[tok*4+ch]=pack(ch%2 ? 8.0f : -8.0f);
    }
    for (int i=1;i<7;++i) flat.parent[3+i]=i-1;
    CHECK(base.run(0)==RAD_OK);CHECK(base.run(1)==RAD_OK);
    CHECK(poison.run(0)==RAD_OK);CHECK(poison.run(1)==RAD_OK);
    CHECK(flat.run(0)==RAD_OK);CHECK(flat.run(1)==RAD_OK);
    for (int node:{0,2,4,6}) compare_rows(base,poison,3+node);
    for (int tok=10;tok<15;++tok) compare_rows(base,poison,tok);
    CHECK(base.o!=flat.o);CHECK(base.o!=poison.o);
    CHECK(base.run(2)==RAD_OK);CHECK(poison.run(2)==RAD_OK);
    CHECK(base.state==poison.state);CHECK(base.hist==poison.hist);
}

TEST(gdn_tree_mixed_absolute_subranges) {
    Fixture all,split=all;
    CHECK(all.run(0)==RAD_OK);CHECK(all.run(1)==RAD_OK);
    for (int s:{1,0}) { CHECK(split.run(0,s,1)==RAD_OK);CHECK(split.run(1,s,1)==RAD_OK); }
    CHECK(all.q==split.q);CHECK(all.k==split.k);CHECK(all.v==split.v);CHECK(all.o==split.o);
    CHECK(all.factors==split.factors);CHECK(all.raw==split.raw);
    CHECK(all.run(2)==RAD_OK);
    for (int s:{1,0}) CHECK(split.run(2,s,1)==RAD_OK);
    CHECK(all.state==split.state);CHECK(all.hist==split.hist);
}

TEST(gdn_tree_depth16_cold_width17_and_wrap) {
    for (int ring:{0,1}) {
        Fixture original(ring,17,16);original.cu={2,18,18};original.ctx[0]=0;
        for (int i=0;i<16;++i) original.parent[2+i]=i-1;
        original.len[0]=16;for (int i=0;i<16;++i) original.path[i]=i;
        Fixture tree=original;
        CHECK(tree.run(0,0,1)==RAD_OK);CHECK(tree.run(1,0,1)==RAD_OK);
        Fixture serial=isolated_path(original,0,ancestors(original,0,15));
        for (int tok=2;tok<18;++tok) compare_rows(tree,serial,tok);
        CHECK(tree.run(2,0,1)==RAD_OK);CHECK(tree.state==serial.state);CHECK(tree.hist==serial.hist);
    }
}

TEST(gdn_tree_invalid_metadata_is_transactional) {
    Fixture f;CHECK(f.run(0)==RAD_OK);CHECK(f.run(1)==RAD_OK);
    const Fixture good=f;
    f.len[1]=0;CHECK(f.run(2)==RAD_E_INVAL);CHECK(f.state==good.state);CHECK(f.hist==good.hist);
    f=good;f.path[1]=5;CHECK(f.run(2)==RAD_E_INVAL);CHECK(f.state==good.state);
    f=good;f.path[0]=2;CHECK(f.run(2)==RAD_E_INVAL);CHECK(f.hist==good.hist);
    f=good;f.idx[2]=f.idx[0];CHECK(f.run(2)==RAD_E_INVAL);CHECK(f.state==good.state);
    f=good;f.parent[7]=7;CHECK(f.run(0)==RAD_E_INVAL);CHECK(f.q==good.q);
    f=good;f.parent[3]=0;CHECK(f.run(1)==RAD_E_INVAL);CHECK(f.o==good.o);
    f=good;f.cu[2]=25;CHECK(f.run(0)==RAD_E_INVAL);CHECK(f.raw==good.raw);
    f=good;f.ctx[1]=INT32_MAX;CHECK(f.run(2)==RAD_E_INVAL);CHECK(f.state==good.state);
    f=good;f.idx[0]=-1;CHECK(f.run(1)==RAD_E_INVAL);
    f=good;f.cu[0]=-1;CHECK(f.run(0)==RAD_E_INVAL);
    auto p=good.params();auto t=f.operands(1);t[7].dtype=RAD_BF16;
    RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
    CHECK(rad_ref_gdn_recurrent_tree(&a,nullptr)==RAD_E_SHAPE);
}

TEST(gdn_tree_independent_scalar_root_witness) {
    Fixture f;f.cu={0,1,1};f.ctx[0]=0;f.idx[0]=0;f.parent[0]=-1;
    std::fill(f.state.begin(),f.state.end(),0);std::fill(f.q.begin(),f.q.end(),0);
    std::fill(f.k.begin(),f.k.end(),0);std::fill(f.v.begin(),f.v.end(),0);
    std::fill(f.ab.begin(),f.ab.end(),0);std::fill(f.z.begin(),f.z.end(),0);
    f.q[0]=pack(1);f.k[0]=pack(1);f.v[0]=pack(0.5f);f.v[128]=pack(-0.25f);
    auto p=f.params();RadParam act{};act.key="act";act.kind=RAD_P_STR;act.sval="sigmoid";p.push_back(act);
    auto t=f.operands(1,0,1);RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
    CHECK(rad_ref_gdn_recurrent_tree(&a,nullptr)==RAD_OK);
    RadTreeGdn cfg{128,128,1,2,0.125f,1e-6f,1e-6f,20.0f,1};
    std::vector<float> q(128),k(128),v(256),zero(256),ab(2),s0(32768),states(32768),out(256);
    q[0]=k[0]=1;v[0]=0.5f;v[128]=-0.25f;
    CHECK(rad_tree_gdn(&cfg,f.parent.data(),1,q.data(),k.data(),v.data(),ab.data(),ab.data(),
        f.alog.data(),f.dt.data(),s0.data(),zero.data(),f.norm.data(),out.data(),states.data())==RAD_OK);
    for (int i=0;i<256;++i) CHECK(f.o[i]==pack(out[i]));
    /* Independent full-state oracle vs compact parent-dependent delta replay. */
    f.path[0]=0;f.len[0]=1;CHECK(f.run(2,0,1)==RAD_OK);
    CHECK(std::memcmp(f.state.data(),states.data(),states.size()*sizeof(float))==0);
}

TEST(gdn_tree_independent_conv_oracle_and_width_one) {
    Fixture f(1,1,1);f.cu={0,7,7};f.ctx[0]=0;
    const int parents[]={-1,0,0,1,2,1,4};std::copy(parents,parents+7,f.parent.begin());
    std::fill(f.w.begin(),f.w.end(),pack(0.5f));
    std::vector<float> x(7*Fixture::D),weights(Fixture::D,0.5f),q(7*128),k(q),v(7*256);
    for (int i=0;i<7;++i) for (int ch=0;ch<Fixture::D;++ch) x[i*Fixture::D+ch]=unpack(f.x[i*Fixture::XP+ch]);
    RadTreeGdn cfg{128,128,1,2,0.125f,1e-6f,1e-6f,20.0f,0};
    CHECK(rad_tree_conv(&cfg,1,Fixture::D,parents,7,x.data(),weights.data(),nullptr,nullptr,
                        q.data(),k.data(),v.data(),nullptr)==RAD_OK);
    auto p=f.params();auto t=f.operands(0,0,1);t[2]={};
    RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
    CHECK(rad_ref_gdn_conv_tree(&a,nullptr)==RAD_OK);
    for (size_t i=0;i<q.size();++i) { CHECK(f.q[i]==pack(q[i]));CHECK(f.k[i]==pack(k[i])); }
    for (size_t i=0;i<v.size();++i) CHECK(f.v[i]==pack(v[i]));
}

TEST(gdn_tree_registered_shapes_are_callable) {
    Fixture f;auto p=f.params();
    for (int op=0;op<3;++op) {
        const auto* schema=rad_gdn_tree_schema_at(op);const auto* row=rad_ref_gdn_tree_at(op);
        CHECK(schema!=nullptr);CHECK(row!=nullptr);CHECK(std::strcmp(schema->op,row->op)==0);
        CHECK(row->launch!=nullptr);CHECK(row->opd_shape!=nullptr);CHECK(row->replaces==nullptr);
        std::vector<std::vector<uint32_t>> storage(schema->n_operands);
        std::vector<RadTensor> tensors(schema->n_operands);
        for (int i=0;i<schema->n_operands;++i) {
            RadOpdDesc d{};CHECK(row->opd_shape(p.data(),(int)p.size(),i,&d)==RAD_OK);
            if (d.flags&RAD_OPD_F_ABSENT) continue;
            int64_t count=1;for (unsigned j=0;j<d.rank;++j) count*=d.shape[j];
            storage[i].resize((count*(d.dtype==RAD_BF16 ? 2 : 4)+3)/4);
            auto& t=tensors[i];t.data=storage[i].data();t.dtype=d.dtype;t.rank=d.rank;
            int64_t stride=1;
            for (int j=(int)d.rank-1;j>=0;--j) { t.shape[j]=d.shape[j];t.stride[j]=stride;stride*=d.shape[j]; }
            if (d.fill==RAD_FILL_INDEX) {
                auto* data=static_cast<int32_t*>(t.data);
                for (int64_t j=0;j<count;++j) data[j]=d.flags&RAD_OPD_F_IDX_CU ? (int)(j*16) :
                    d.idx_const>=0 ? (int)d.idx_const : (int)j;
            } else if (d.dtype==RAD_BF16) {
                auto* data=static_cast<uint16_t*>(t.data);for (int64_t j=0;j<count;++j) data[j]=pack(0.125f);
            } else {
                auto* data=static_cast<float*>(t.data);for (int64_t j=0;j<count;++j) data[j]=0.125f;
            }
        }
        RadArgs a{};a.t=tensors.data();a.n_t=(int)tensors.size();a.p=p.data();a.n_p=(int)p.size();
        CHECK(row->launch(&a,nullptr)==RAD_OK);
        RadOpdDesc unused{};CHECK(row->opd_shape(p.data(),(int)p.size(),schema->n_operands,&unused)==RAD_E_INVAL);
    }
    CHECK(rad_gdn_tree_schema_at(-1)==nullptr);
    CHECK(rad_ref_gdn_tree_at(rad_ref_gdn_tree_count())==nullptr);
}
TEST(gdn_tree_independent_full_branch_states) {
    Fixture tree(1);CHECK(tree.run(0)==RAD_OK);CHECK(tree.run(1)==RAD_OK);
    RadTreeGdn cfg{128,128,1,2,0.125f,1e-6f,1e-6f,20.0f,0};
    for (int seq=0;seq<2;++seq) {
        const int bos=tree.cu[seq],n=tree.cu[seq+1]-bos,slot=tree.idx[seq*2];
        std::vector<float> q(n*128),k(q),v(n*256),z(v),a(n*2),b(a);
        std::vector<float> anchor(32768),states(n*32768),out(n*256);
        for (int i=0;i<n;++i) {
            for (int j=0;j<128;++j) { q[i*128+j]=unpack(tree.q[(bos+i)*128+j]);k[i*128+j]=unpack(tree.k[(bos+i)*128+j]); }
            for (int j=0;j<256;++j) { v[i*256+j]=unpack(tree.v[(bos+i)*256+j]);z[i*256+j]=unpack(tree.z[(bos+i)*256+j]); }
            for (int j=0;j<2;++j) { a[i*2+j]=unpack(tree.ab[(bos+i)*4+j]);b[i*2+j]=unpack(tree.ab[(bos+i)*4+2+j]); }
        }
        std::copy_n(tree.state.data()+slot*Fixture::SP,32768,anchor.data());
        CHECK(rad_tree_gdn(&cfg,tree.parent.data()+bos,n,q.data(),k.data(),v.data(),a.data(),b.data(),
            tree.alog.data(),tree.dt.data(),anchor.data(),z.data(),tree.norm.data(),out.data(),states.data())==RAD_OK);
        for (int node=0;node<n;++node) {
            Fixture committed=tree;const auto path=ancestors(tree,seq,node);
            std::copy(path.begin(),path.end(),committed.path.begin()+seq*16);committed.len[seq]=(int)path.size();
            CHECK(committed.run(2,seq,1)==RAD_OK);
            double error=0,energy=0;
            for (int j=0;j<32768;++j) {
                const double expected=states[node*32768+j];
                const double delta=committed.state[slot*Fixture::SP+j]-expected;
                error+=delta*delta;energy+=expected*expected;
            }
            /* Existing tree_state_oracle_test's libref state threshold, unchanged.
             * This is an independent scalar witness, not a bit-exact equivalence
             * claim between its serial sums and the selected two-half FMA arm. */
            CHECK(std::sqrt(error/(energy>0 ? energy : 1))<1e-6);
        }
    }
}

TEST(gdn_tree_raw_history_has_exact_logical_positions) {
    for (int ring:{0,1}) {
        Fixture f(ring),original=f;
        CHECK(f.run(0)==RAD_OK);CHECK(f.run(1)==RAD_OK);CHECK(f.run(2)==RAD_OK);
        for (int seq=0;seq<2;++seq) {
            const int slot=f.idx[seq*2],bos=f.cu[seq],count=f.len[seq];
            for (int ch=0;ch<Fixture::D;++ch) {
                if (ring) {
                    for (int j=0;j<count;++j)
                        CHECK(f.hist[f.history_offset(slot,ch,(f.ctx[seq]+j)%f.r)]==
                              original.x[(bos+f.path[seq*16+j])*Fixture::XP+ch]);
                } else {
                    for (int j=0;j<3;++j) {
                        const int from=count-3+j;
                        const uint16_t expected=from>=0
                            ? original.x[(bos+f.path[seq*16+from])*Fixture::XP+ch]
                            : original.hist[original.history_offset(slot,ch,count+j)];
                        CHECK(f.hist[f.history_offset(slot,ch,j)]==expected);
                    }
                }
            }
        }
    }
}

TEST(gdn_tree_nonlast_branch_commit_matches_linear_replay) {
    Fixture original, tree = original;
    tree.path[0] = 0; tree.path[1] = 1; tree.path[2] = 3; tree.len[0] = 3;
    CHECK(tree.run(0) == RAD_OK); CHECK(tree.run(1) == RAD_OK);
    CHECK(tree.run(2, 0, 1) == RAD_OK);
    const Fixture linear = isolated_path(original, 0, {0, 1, 3});
    const int slot = original.idx[0];
    CHECK(std::memcmp(tree.state.data() + slot * Fixture::SP,
                      linear.state.data() + slot * Fixture::SP,
                      Fixture::SP * sizeof(float)) == 0);
    CHECK(std::memcmp(tree.hist.data() + slot * Fixture::D * (tree.r + 1),
                      linear.hist.data() + slot * Fixture::D * (tree.r + 1),
                      Fixture::D * (tree.r + 1) * sizeof(uint16_t)) == 0);
}

TEST(gdn_tree_scheduler_requires_complete_bound_state_ops) {
    rad::Program p;
    p.kv_groups.resize(3);
    p.kv_groups[1].decl.kind = RAD_KV_LINEAR;
    p.kv_groups[2].decl.kind = RAD_KV_CONV;
    p.kv_groups[1].layers = p.kv_groups[2].layers = {2};
    rad::Scheduler sched;
    sched.prog_ = &p;
    auto req = std::make_unique<rad::Request>();
    req->id = 8101; req->prompt = {5}; req->max_tokens = 100; req->may_speculate = true;
    CHECK(sched.add(std::move(req)) == RAD_OK);
    radiance::CandidateTree tree;
    tree.add_node(0, -1, 5, 0, 0);
    tree.add_node(1, 0, 10, 1, 0);
    tree.add_node(2, 0, 20, 1, 0);
    CHECK(sched.set_draft_tree(8101, tree, 0) == RAD_E_UNSUPPORTED);
    static rad::KernelRow rows[3];
    for (int i = 0; i < 3; ++i) {
        rows[i].info = rad_ref_gdn_tree_at(i);
        rad::OpInfo op;
        op.op = rad_gdn_tree_schema_at(i)->op;
        op.base.set_i("state_group", 1); op.base.set_i("conv_group", 2);
        op.base.set_i("layer", 2);
        rad::Band band; band.hi = 16; band.dom[RAD_DOMAIN_HOST].row = &rows[i];
        op.bands.push_back(band);
        p.ops.push_back(op);
        CHECK(sched.set_draft_tree(8101, tree, 0) == (i == 2 ? RAD_OK : RAD_E_UNSUPPORTED));
    }
    CHECK(sched.find(8101)->tree_parent != nullptr);
    p.ops[1].bands.clear();
    CHECK(sched.set_draft_tree(8101, tree, 0) == RAD_E_UNSUPPORTED);
    p.ops[1].bands = p.ops[0].bands;
    /* Qwen's PLE groups need their own complete bindings; GDN rows alone
     * must never lift their veto, and a prefill-only row is not a verifier. */
    p.kv_groups.resize(5);
    p.kv_groups[3].name = "kv_ple_tok";
    p.kv_groups[4].name = "kv_ple_conv";
    p.kv_groups[3].decl.kind = p.kv_groups[4].decl.kind = RAD_KV_CONV;
    p.kv_groups[3].layers = p.kv_groups[4].layers = {2};
    CHECK(sched.set_draft_tree(8101, tree, 0) == RAD_E_UNSUPPORTED);
    const size_t ple_begin = p.ops.size();
    static rad::KernelRow ple_rows[RAD_PLE_TREE_OPS];
    for (int i = 0; i < RAD_PLE_TREE_OPS; ++i) {
        ple_rows[i].info = rad_ref_ple_tree_at(i);
        rad::OpInfo op;
        op.op = rad_ple_tree_schema_at(i)->op;
        op.base.set_i("tok_group", 3); op.base.set_i("conv_group", 4);
        op.base.set_i("layer", 2); op.base.set_s("mode", "verify");
        rad::Band band; band.hi = 16; band.dom[RAD_DOMAIN_HOST].row = &ple_rows[i];
        op.bands.push_back(band);
        p.ops.push_back(op);
        CHECK(sched.set_draft_tree(8101, tree, 0) ==
              (i == RAD_PLE_TREE_COMMIT ? RAD_OK : RAD_E_UNSUPPORTED));
    }
    p.ops[ple_begin + RAD_PLE_TREE_CONV].base.set_s("mode", "prefill");
    CHECK(sched.set_draft_tree(8101, tree, 0) == RAD_E_UNSUPPORTED);
    p.ops[ple_begin + RAD_PLE_TREE_CONV].base.set_s("mode", "verify");
    p.ops[ple_begin + RAD_PLE_TREE_COMMIT].base.set_i("tok_group", 2);
    CHECK(sched.set_draft_tree(8101, tree, 0) == RAD_E_UNSUPPORTED);
    p.ops[ple_begin + RAD_PLE_TREE_COMMIT].base.set_i("tok_group", 3);
    CHECK(sched.set_draft_tree(8101, tree, 0) == RAD_OK);
    rad::OpInfo qsa;
    qsa.op = "qsa_select"; qsa.base.set_i("topk", 512); qsa.base.set_i("ratio", 4);
    p.ops.push_back(qsa);
    CHECK(sched.set_draft_tree(8101, tree, 2050) == RAD_E_UNSUPPORTED);
    CHECK(sched.set_draft_tree(8101, tree, 2049) == RAD_OK);
    radiance::TreeRegistry::instance().clear(8101);
}

TEST(gdn_tree_batch_qsa_exactness_includes_other_prefill_rows) {
    rad::BatchBuilder builder;
    builder.max_ctx_ = 8192;
    rad::StepPlan plan;
    rad::StepEntry decode; decode.ctx_len = 100; decode.n_tokens = 16;
    plan.e.push_back(decode);
    CHECK(builder.qsa_dense_for(plan, 2051));
    rad::StepEntry prefill; prefill.is_prefill = true;
    prefill.ctx_len = 2048; prefill.n_tokens = 64;
    plan.e.push_back(prefill);
    CHECK(!builder.qsa_dense_for(plan, 2051));
}

TEST(gdn_tree_sample_and_match_walks_one_target_draw_per_visited_prefix) {
    radiance::CandidateTree tree;
    tree.add_node(0, -1, 5, 0, 0);
    tree.add_node(1, 0, 10, 1, 0); tree.add_node(2, 0, 20, 1, 0);
    tree.add_node(3, 1, 30, 2, 0); tree.add_node(4, 2, 50, 2, 0);
    for (int root : {10, 20, 99}) {
        const int32_t draws[] = {root, 30, 50, 40, 60};
        int32_t out[16]{}, nodes[16]{}, accepted = -1, bonus = -1;
        CHECK(rad::accept_tree_greedy(tree, draws, 5, out, &accepted, &bonus, nodes) == RAD_OK);
        CHECK(out[0] == root);
        CHECK(accepted == (root == 99 ? 0 : 2));
        CHECK(bonus == (root == 10 ? 40 : root == 20 ? 60 : 99));
        CHECK(nodes[0] == 0);
        if (accepted) CHECK(nodes[2] == (root == 10 ? 3 : 4));
        CHECK(rad::accept_tree_greedy(tree, draws, 4, out, &accepted, &bonus, nodes) == RAD_OK);
        CHECK(bonus == (root == 10 ? 40 : root == 20 ? 50 : 99));
    }
}

TEST(batch_teardown_retires_queued_metadata_before_freeing_storage) {
    rad::Program program;
    RadStream stream = nullptr;
    RadEvent done = nullptr;
    CHECK(rad_stream_create(&stream, 0) == RAD_OK);
    CHECK(rad_event_create(&done) == RAD_OK);
    constexpr int64_t bytes = 8 * 1024 * 1024;
    void* storage = rad_dev_alloc(bytes, RAD_MEM_DEVICE);
    REQUIRE(storage);
    rad::BatchBuilder builder;
    builder.ranks_.push_back({0, stream, &program});
    /* Keep the upload queue pending through fini; no sleep or timing assertion.
     * The event is the independent completion witness for all queued writes. */
    for (int i = 0; i < 64; ++i)
        CHECK(rad_memset_async(storage, i, bytes, stream) == RAD_OK);
    CHECK(rad_event_record(done, stream) == RAD_OK);
    builder.fini();
    CHECK(rad_event_query(done) == 1);
    CHECK(rad_stream_sync(stream) == RAD_OK); /* also makes red-test cleanup safe */
    rad_event_destroy(done);
    rad_dev_free(storage, RAD_MEM_DEVICE);
    rad_stream_destroy(stream);
}

TEST(qsa_winner_replay_only_follows_deferred_verification) {
    rad::arch::QsaIndexer indexer;
    indexer.c.tree_state = true;
    indexer.c.budget = 10;
    indexer.c.ratio = 4;
    indexer.c.topk = 2;
    CHECK(indexer.c.exact_to() == 11); /* budget is not divisible by ratio */
    rad::Program program;
    rad::OpInfo select;
    select.op = "qsa_select";
    select.base.set_i("topk", 2);
    select.base.set_i("ratio", 4);
    program.ops.push_back(select);
    CHECK(rad::tree_qsa_exact_to(program) == indexer.c.exact_to());
    RadBatch batch{};
    const int32_t parents[] = {-1, 0};
    batch.spec_parent = parents;
    batch.max_ctx_len = 9;
    batch.max_q_len = 2;
    CHECK(indexer.defer_decode(&batch));
    batch.max_ctx_len = 10;
    CHECK(!indexer.defer_decode(&batch)); /* sparse chain already materialized */
    batch.max_ctx_len = 0;
    batch.max_q_len = 1;
    batch.spec_parent = nullptr;
    CHECK(!indexer.defer_decode(&batch)); /* anchor-only already materialized */
    batch.spec_parent = parents;
    indexer.c.tree_state = false;
    CHECK(!indexer.defer_decode(&batch));
}

TEST(batch_failed_drain_keeps_mapped_winner_metadata_alive) {
    rad::Program program;
    rad::BatchBuilder builder;
    builder.tree_paths_ = static_cast<int32_t*>(rad_dev_alloc(64, RAD_MEM_HOST_MAPPED));
    REQUIRE(builder.tree_paths_);
    int32_t* const paths = builder.tree_paths_;
    builder.ranks_.push_back({INT32_MAX, nullptr, &program});
    builder.fini();
    CHECK(builder.tree_paths() == paths);
    builder.ranks_[0].device = 0;
    builder.fini();
    CHECK(builder.tree_paths() == nullptr);
}

namespace {
using WideFixture=FixtureT<72,RAD_GDN_TREE_WIDE_CAP>;
/* 64 proposals + root at tokens [3,68): a spine with a sibling at every fifth
 * node, so node IDs past 15/31/63 (root-inclusive node 64, depth 52) sit far
 * deeper than one 16-node tile. Only sequence 0 is issued. */
WideFixture wide_tree(int ring) {
    WideFixture f(ring);f.cu={3,68,70};
    for (int i=0;i<RAD_GDN_TREE_WIDE_CAP;++i) f.parent[3+i]=i==0 ? -1 : i%5==4 ? i-2 : i-1;
    f.path[0]=0;f.len[0]=1;
    return f;
}
long long row_node_limit(const RadKernelInfo& row) {
    long long most=-1;
    for (int i=0;i<row.n_constraints;++i)
        if (!std::strcmp(row.constraints[i].key,"node_capacity") && row.constraints[i].op==RAD_C_LE)
            most=row.constraints[i].ival;
    return most;
}
} // namespace

TEST(gdn_tree_wide65_branches_and_deep_commit_match_isolated_paths) {
    for (int ring:{0,1}) {
        const WideFixture original=wide_tree(ring);
        WideFixture tree=original;
        CHECK(tree.run(0,0,1)==RAD_OK);CHECK(tree.run(1,0,1)==RAD_OK);
        for (int node:{15,16,31,32,63,64})
            compare_rows(tree,isolated_path(original,0,ancestors(original,0,node)),3+node);
        const auto winner=ancestors(original,0,64);
        CHECK(winner.size()==52u);
        std::copy(winner.begin(),winner.end(),tree.path.begin());tree.len[0]=(int)winner.size();
        CHECK(tree.run(2,0,1)==RAD_OK);
        const WideFixture serial=isolated_path(original,0,winner);
        CHECK(tree.state==serial.state);CHECK(tree.hist==serial.hist);
    }
}

TEST(gdn_tree_wide65_chain_commits_every_node) {
    WideFixture original=wide_tree(1);
    std::vector<int> chain(RAD_GDN_TREE_WIDE_CAP);
    for (int i=0;i<RAD_GDN_TREE_WIDE_CAP;++i) { chain[i]=i;original.path[i]=i;original.parent[3+i]=i-1; }
    original.len[0]=RAD_GDN_TREE_WIDE_CAP;
    WideFixture tree=original;
    CHECK(tree.run(0,0,1)==RAD_OK);CHECK(tree.run(1,0,1)==RAD_OK);CHECK(tree.run(2,0,1)==RAD_OK);
    const WideFixture serial=isolated_path(original,0,chain);
    for (int node=0;node<RAD_GDN_TREE_WIDE_CAP;++node) compare_rows(tree,serial,3+node);
    CHECK(tree.state==serial.state);CHECK(tree.hist==serial.hist);
}

TEST(gdn_tree_wide65_fused_host_row_matches_unfused) {
    WideFixture initial=wide_tree(1);
    WideFixture reference=initial,actual=initial;
    REQUIRE(reference.run(0,0,1)==RAD_OK);REQUIRE(reference.run(1,0,1)==RAD_OK);
    FusedHostCallT<WideFixture> call(actual,1);
    const auto* row=rad_ref_gdn_tree_at(RAD_GDN_TREE_FUSED_I8G);
    REQUIRE(row && row->launch && row->scratch);
    auto a=call.args();
    const int64_t need=row->scratch(&a);
    REQUIRE(need>0);
    call.scratch.resize(size_t((need+7)/8));a=call.args();
    REQUIRE(row->launch(&a,nullptr)==RAD_OK);
    CHECK(actual.raw==reference.raw);
    CHECK(std::memcmp(actual.factors.data(),reference.factors.data(),actual.factors.size()*sizeof(float))==0);
    for (int head=0;head<WideFixture::H;++head)
        CHECK(call.scales[(3+64)*FusedHostCallT<WideFixture>::SC+head]!=-123.0f);
}

TEST(gdn_tree_wide_capacity_rows_refusal_and_path_pitch) {
    /* Host rows advertise the wide range; legacy device rows stay at sixteen. */
    for (int op=0;op<RAD_GDN_TREE_OPS;++op) {
        CHECK(row_node_limit(*rad_ref_gdn_tree_at(op))==RAD_GDN_TREE_WIDE_CAP);
        CHECK(row_node_limit(rad_gdn_tree_row(op,RAD_DOMAIN_DEVICE,nullptr,false))==RAD_GDN_TREE_CAP);
        CHECK(row_node_limit(rad_gdn_tree_row(op,RAD_DOMAIN_DEVICE,nullptr,true))==RAD_GDN_TREE_WIDE_CAP);
    }
    /* 66 total nodes: every op refuses its geometry before any write. */
    WideFixture f=wide_tree(0);f.capacity=RAD_GDN_TREE_WIDE_CAP+1;
    const WideFixture good=f;
    for (int op=0;op<3;++op) CHECK(f.run(op,0,1)==RAD_E_UNSUPPORTED);
    CHECK(f.q==good.q);CHECK(f.o==good.o);CHECK(f.raw==good.raw);CHECK(f.factors==good.factors);
    CHECK(f.state==good.state);CHECK(f.hist==good.hist);
    auto p=f.params();p.push_back(pi("max_tokens",2*(RAD_GDN_TREE_WIDE_CAP+1)));
    RadOpdDesc d{};
    CHECK(rad_gdn_tree_fused_shape(p.data(),(int)p.size(),0,&d)==RAD_E_UNSUPPORTED);
    CHECK(rad_gdn_tree_commit_shape(p.data(),(int)p.size(),6,&d)==RAD_E_UNSUPPORTED);
    /* 65: the declared commit pitch covers the capacity. */
    f.capacity=RAD_GDN_TREE_WIDE_CAP;p=f.params();
    CHECK(rad_gdn_tree_commit_shape(p.data(),(int)p.size(),6,&d)==RAD_OK);
    CHECK(d.rank==2 && d.shape[0]==2 && d.shape[1]==RAD_GDN_TREE_WIDE_CAP);
    /* A path pitch below the capacity is a shape error, not a truncated read. */
    {
        WideFixture g=wide_tree(0);
        CHECK(g.run(0,0,1)==RAD_OK);CHECK(g.run(1,0,1)==RAD_OK);
        const WideFixture ready=g;
        auto gp=g.params();auto t=g.operands(2,0,1);t[6].shape[1]=RAD_GDN_TREE_WIDE_CAP-1;
        RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=gp.data();a.n_p=(int)gp.size();
        CHECK(rad_ref_gdn_tree_commit(&a,nullptr)==RAD_E_SHAPE);
        CHECK(g.state==ready.state);CHECK(g.hist==ready.hist);
        CHECK(g.run(2,0,1)==RAD_OK);CHECK(g.state!=ready.state);
    }
    /* A 16-node geometry reads only path_len columns of a wider pitch: the same
     * bytes as pitch 16, with the trailing -9 columns ignored. */
    Fixture small;WideFixture pitched;pitched.capacity=RAD_GDN_TREE_CAP;
    for (int op=0;op<3;++op) { CHECK(small.run(op)==RAD_OK);CHECK(pitched.run(op)==RAD_OK); }
    CHECK(small.state==pitched.state);CHECK(small.hist==pitched.hist);
    CHECK(std::equal(small.factors.begin(),small.factors.end(),pitched.factors.begin()));
    CHECK(std::equal(small.raw.begin(),small.raw.end(),pitched.raw.begin()));
}

/* 64 draft nodes plus the root is the widest tree a GDN layer may declare. */
TEST(gdn_tree_rejects_a_declared_window_above_sixty_four_draft_nodes) {
    rad::arch::GdnFP8 block;
    rad::arch::Geom geometry{};
    geometry.max_spec = RAD_GDN_TREE_WIDE_CAP;
    rad::arch::GdnFP8::Config config{};
    config.tree_state = true;
    rad::arch::Names names;
    CHECK(block.declare(nullptr, names, geometry, config, 0, 0, 0, {}, {}) ==
          RAD_E_UNSUPPORTED);
}

TEST(ple_tree_rejects_an_unsupported_declared_window_or_dtype) {
    rad::arch::PleLayer block;
    rad::arch::Geom geometry{};
    geometry.max_spec = 65;
    geometry.act_dtype = RAD_BF16;
    rad::arch::PleLayer::Config config{};
    CHECK(!config.tree_state);  /* Legacy architectures do not opt in implicitly. */
    config.tree_state = true;
    rad::arch::Names names;
    CHECK(block.declare(nullptr, names, geometry, config, 1, {}, {}) ==
          RAD_E_UNSUPPORTED);
    geometry.max_spec = 15;
    geometry.act_dtype = RAD_F16;
    CHECK(block.declare(nullptr, names, geometry, config, 1, {}, {}) ==
          RAD_E_UNSUPPORTED);
}

RAD_TEST_MAIN()
