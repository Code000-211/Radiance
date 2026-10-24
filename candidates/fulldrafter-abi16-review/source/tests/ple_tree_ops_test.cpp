/* Host-only independent oracle: unchanged libref linear PLE operations replay
 * each root-to-node path. The tree implementation is never its own oracle. */
#include "rad_test.h"
#include "../mine/ref_ple_tree.h"
#include "../libref/ref_ops.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

namespace {
RadTensor tensor(void* data,uint32_t dtype,std::initializer_list<int64_t> shape) {
    RadTensor t{};t.data=data;t.dtype=dtype;
    for (int64_t d:shape) t.shape[t.rank++]=d;
    int64_t stride=1;
    for (int i=(int)t.rank-1;i>=0;--i) { t.stride[i]=stride;stride*=t.shape[i]; }
    return t;
}
RadParam pi(const char* key,int64_t value) {
    RadParam p{};p.key=key;p.kind=RAD_P_INT;p.ival=value;return p;
}
RadParam ps(const char* key,const char* value) {
    RadParam p{};p.key=key;p.kind=RAD_P_STR;p.sval=value;return p;
}
int launch(int op,std::vector<RadTensor>& t,std::vector<RadParam>& p) {
    RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
    return rad_ref_ple_tree_at(op)->launch(&a,nullptr);
}
struct Fixture {
    static constexpr int N=8,H=4,T=40,Slots=4,XP=N+3,RT=5,RP=RT+2;
    int ring,capacity,pitch,rows,width=4,dilation=3,rc=13,cp=15;
    std::vector<int32_t> tokens,tokhist,ids,cu{3,10,15},parent;
    std::vector<int32_t> ti{2,91,0,92},ci{1,93,3,94},ctx{4,17},path,len{3,2};
    std::vector<int64_t> mult{INT64_MAX,-9223372036854775783LL,0x123456789abcdefLL};
    std::vector<int64_t> vocab{97,101,103,107},offset{0,100,210,320};
    std::vector<uint16_t> x,w,history,residual,y,saved;
    explicit Fixture(int ring_=0,int capacity_=16,int pitch_=0):
        ring(ring_),capacity(capacity_),pitch(pitch_ ? pitch_ : capacity_),rows(capacity_>16 ? 144 : T),
        tokens(rows),tokhist(Slots*RP,-201),ids(rows*H,-901),
        parent(rows,-77),path(2*pitch,-88),x(rows*XP),w(N*width),history(Slots*N*cp),
        residual(rows*N),y(rows*N,0x7fc1),saved(rows*N,0x7fa1) {
        for (int i=0;i<rows;++i) tokens[i]=11+i*3;
        for (size_t i=0;i<x.size();++i) x[i]=rad_f32_to_bf16((int(i%37)-18)*0.03125f);
        for (size_t i=0;i<w.size();++i) w[i]=rad_f32_to_bf16((int(i%11)-5)*0.0625f);
        for (size_t i=0;i<history.size();++i) history[i]=rad_f32_to_bf16((int(i%29)-14)*0.015625f);
        for (size_t i=0;i<residual.size();++i) residual[i]=rad_f32_to_bf16((int(i%17)-8)*0.125f);
        for (int slot=0;slot<Slots;++slot) { tokhist[slot*RP]=31+slot;tokhist[slot*RP+1]=41+slot; }
        const int a[]={-1,0,0,1,2,1,4},b[]={-1,0,1,0,3};
        std::copy(a,a+7,parent.begin()+3);std::copy(b,b+5,parent.begin()+10);
        path[0]=0;path[1]=1;path[2]=3;path[pitch]=0;path[pitch+1]=3;
        tokens[4]=2;tokens[7]=2;tokens[12]=-123;
    }
    int hist() const { return (width-1)*dilation; }
    int ho(int slot,int ch,int pos) const { return (slot*N+ch)*cp+pos; }
    std::vector<RadParam> params(bool prefill=false,int q=0) const {
        return {pi("q_len",q ? q : capacity),pi("n",N),pi("heads",H),pi("ngram",3),pi("eos",2),
            pi("width",width),pi("dilation",dilation),pi("node_capacity",capacity),
            pi("history_ring",ring),pi("n_seq",2),ps("mode",prefill ? "prefill" : "verify")};
    }
    std::vector<RadTensor> operands(int op,int seq=0,int ns=2,bool linear=false) {
        RadTensor th=tensor(tokhist.data(),RAD_I32,{Slots,1,RT});th.stride[0]=RP;
        RadTensor ch=tensor(history.data(),RAD_BF16,{Slots,N,rc});ch.stride[1]=cp;ch.stride[0]=N*cp;
        RadTensor xx=tensor(x.data(),RAD_BF16,{rows,N});xx.stride[0]=XP;
        RadTensor tt=tensor(tokens.data(),RAD_I32,{rows});
        RadTensor ix=tensor(ti.data()+seq*2,RAD_I32,{ns,2});
        RadTensor cx=tensor(ci.data()+seq*2,RAD_I32,{ns,2});
        RadTensor cc=tensor(cu.data()+seq,RAD_I32,{ns+1});
        RadTensor pp=linear ? RadTensor{} : tensor(parent.data(),RAD_I32,{rows});
        RadTensor ctxb=tensor(ctx.data()+seq,RAD_I32,{ns});
        if (op==RAD_PLE_TREE_IDS) return {tt,th,ix,cc,pp,ctxb,tensor(mult.data(),RAD_I64,{3}),
            tensor(vocab.data(),RAD_I64,{H}),tensor(offset.data(),RAD_I64,{H}),tensor(ids.data(),RAD_I32,{rows,H})};
        if (op==RAD_PLE_TREE_CONV) return {xx,tensor(w.data(),RAD_BF16,{N,width}),ch,cx,cc,pp,ctxb,
            tensor(residual.data(),RAD_BF16,{rows,N}),tensor(y.data(),RAD_BF16,{rows,N}),tensor(saved.data(),RAD_BF16,{rows,N})};
        return {th,ix,ch,cx,tt,tensor(saved.data(),RAD_BF16,{rows,N}),cc,pp,ctxb,
            tensor(path.data()+seq*pitch,RAD_I32,{ns,pitch}),tensor(len.data()+seq,RAD_I32,{ns})};
    }
    int run(int op,int seq=0,int ns=2,bool linear=false,bool prefill=false,int q=0) {
        auto t=operands(op,seq,ns,linear);auto p=params(prefill,q);return launch(op,t,p);
    }
};
std::vector<int> ancestry(const Fixture& f,int s,int node) {
    std::vector<int> path;
    for (;node>=0;node=f.parent[f.cu[s]+node]) path.push_back(node);
    std::reverse(path.begin(),path.end());return path;
}
struct Linear {
    std::vector<int32_t> ids,tokhist;
    std::vector<uint16_t> y,history;
};
/* Convert the committed representation to libref's canonical linear state,
 * gather only this ancestry, and let the old operations compute and commit it. */
Linear oracle(const Fixture& f,int s,const std::vector<int>& nodes,bool residual=true) {
    const int length=(int)nodes.size(),bos=f.cu[s],ctx=f.ctx[s],hs=f.ci[2*s],ts=f.ti[2*s];
    Linear out;out.ids.resize(length*Fixture::H);out.y.resize(length*Fixture::N);
    out.tokhist.resize(2);out.history.resize(Fixture::N*f.hist());
    for (int j=0;j<2;++j) out.tokhist[j]=ctx-2+j<0 ? 2 : f.tokhist[ts*Fixture::RP+j];
    for (int ch=0;ch<Fixture::N;++ch) for (int j=0;j<f.hist();++j) {
        const int pos=ctx-f.hist()+j;
        out.history[ch*f.hist()+j]=pos<0 ? 0 : f.history[f.ho(hs,ch,f.ring ? pos%f.rc : j)];
    }
    std::vector<int32_t> tok(length),cu{0,length};
    std::vector<uint16_t> x(length*Fixture::N),r(x.size());
    for (int j=0;j<length;++j) {
        tok[j]=f.tokens[bos+nodes[j]];
        for (int ch=0;ch<Fixture::N;++ch) {
            x[j*Fixture::N+ch]=f.x[(bos+nodes[j])*Fixture::XP+ch];
            r[j*Fixture::N+ch]=f.residual[(bos+nodes[j])*Fixture::N+ch];
        }
    }
    std::vector<RadTensor> t={tensor(tok.data(),RAD_I32,{length}),tensor(out.tokhist.data(),RAD_I32,{1,2}),
        tensor(cu.data(),RAD_I32,{2}),tensor(const_cast<int64_t*>(f.mult.data()),RAD_I64,{3}),
        tensor(const_cast<int64_t*>(f.vocab.data()),RAD_I64,{Fixture::H}),
        tensor(const_cast<int64_t*>(f.offset.data()),RAD_I64,{Fixture::H}),{}, {}, {},
        tensor(out.ids.data(),RAD_I32,{length,Fixture::H})};
    auto p=f.params();RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
    CHECK_EQ(ref_ngram_ids(&a,nullptr),RAD_OK);
    t={tensor(x.data(),RAD_BF16,{length,Fixture::N}),
        tensor(const_cast<uint16_t*>(f.w.data()),RAD_BF16,{Fixture::N,f.width}),
        tensor(out.history.data(),RAD_BF16,{1,Fixture::N,f.hist()}),tensor(cu.data(),RAD_I32,{2}),
        residual ? tensor(r.data(),RAD_BF16,{length,Fixture::N}) : RadTensor{}, {}, {}, {},
        tensor(out.y.data(),RAD_BF16,{length,Fixture::N})};
    a.t=t.data();a.n_t=(int)t.size();CHECK_EQ(ref_ple_conv(&a,nullptr),RAD_OK);
    return out;
}
void compare_node(const Fixture& tree,const Fixture& anchor,int seq,int node) {
    const auto path=ancestry(anchor,seq,node);const auto expected=oracle(anchor,seq,path);
    const int row=anchor.cu[seq]+node,last=(int)path.size()-1;
    for (int h=0;h<Fixture::H;++h) CHECK_EQ(tree.ids[row*Fixture::H+h],expected.ids[last*Fixture::H+h]);
    for (int ch=0;ch<Fixture::N;++ch) {
        CHECK_EQ(tree.y[row*Fixture::N+ch],expected.y[last*Fixture::N+ch]);
        CHECK_EQ(tree.saved[row*Fixture::N+ch],anchor.x[row*Fixture::XP+ch]);
    }
}
void expect_commit(Fixture& expected,const Fixture& anchor,int seq,const std::vector<int>& path) {
    const auto linear=oracle(anchor,seq,path);const int ts=anchor.ti[2*seq],cs=anchor.ci[2*seq];
    for (int j=0;j<2;++j) expected.tokhist[ts*Fixture::RP+j]=linear.tokhist[j];
    for (int ch=0;ch<Fixture::N;++ch) {
        if (anchor.ring) {
            for (size_t j=0;j<path.size();++j)
                expected.history[expected.ho(cs,ch,(anchor.ctx[seq]+(int)j)%anchor.rc)]=
                    anchor.x[(anchor.cu[seq]+path[j])*Fixture::XP+ch];
        } else for (int j=0;j<anchor.hist();++j)
            expected.history[expected.ho(cs,ch,j)]=linear.history[ch*anchor.hist()+j];
    }
}
void branching_case(int ring,int context,bool root_only=false) {
    Fixture anchor(ring);anchor.ctx[0]=context;
    if (root_only) { anchor.tokens[3]=2;anchor.len[0]=1; }
    anchor.tokhist[anchor.ti[0]*Fixture::RP+1]=2;
    Fixture tree=anchor;
    REQUIRE_EQ(tree.run(RAD_PLE_TREE_IDS),RAD_OK);
    REQUIRE_EQ(tree.run(RAD_PLE_TREE_CONV),RAD_OK);
    CHECK(tree.tokhist==anchor.tokhist);CHECK(tree.history==anchor.history);
    for (int s=0;s<2;++s) for (int node=0;node<anchor.cu[s+1]-anchor.cu[s];++node)
        compare_node(tree,anchor,s,node);
    for (int row=0;row<Fixture::T;++row) if (row<anchor.cu[0] || row>=anchor.cu[2]) {
        for (int h=0;h<Fixture::H;++h) CHECK_EQ(tree.ids[row*Fixture::H+h],-901);
        for (int ch=0;ch<Fixture::N;++ch) {
            CHECK_EQ(tree.y[row*Fixture::N+ch],0x7fc1);CHECK_EQ(tree.saved[row*Fixture::N+ch],0x7fa1);
        }
    }
    Fixture expected=anchor;
    for (int s=0;s<2;++s) {
        const std::vector<int> path(anchor.path.begin()+16*s,anchor.path.begin()+16*s+anchor.len[s]);
        CHECK(path.back()!=anchor.cu[s+1]-anchor.cu[s]-1);
        expect_commit(expected,anchor,s,path);
    }
    REQUIRE_EQ(tree.run(RAD_PLE_TREE_COMMIT),RAD_OK);
    CHECK(tree.tokhist==expected.tokhist);CHECK(tree.history==expected.history);
    /* Consume the committed representation in a new step, not only inspect it. */
    tree.ctx[0]+=tree.len[0];tree.ctx[1]+=tree.len[1];tree.cu={30,31,32};
    tree.parent[30]=-1;tree.parent[31]=-1;
    Fixture next=tree;
    REQUIRE_EQ(tree.run(RAD_PLE_TREE_IDS),RAD_OK);REQUIRE_EQ(tree.run(RAD_PLE_TREE_CONV),RAD_OK);
    compare_node(tree,next,0,0);compare_node(tree,next,1,0);
}
} // namespace

TEST(ple_tree_branches_eos_cold_partial_wrap) {
    for (int ring:{0,1}) for (int ctx:{0,1,2,12,25}) branching_case(ring,ctx);
}

TEST(ple_tree_cold_eos_root_only_commit) {
    for (int ring:{0,1}) branching_case(ring,0,true);
}

TEST(ple_tree_mixed_absolute_subranges) {
    Fixture anchor(1),all=anchor,split=anchor;
    for (int op:{RAD_PLE_TREE_IDS,RAD_PLE_TREE_CONV}) {
        REQUIRE_EQ(all.run(op),RAD_OK);
        REQUIRE_EQ(split.run(op,1,1),RAD_OK);
        for (int row=0;row<anchor.cu[1];++row) {
            if (op==RAD_PLE_TREE_IDS) CHECK_EQ(split.ids[row*Fixture::H],-901);
            else CHECK_EQ(split.y[row*Fixture::N],0x7fc1);
        }
        REQUIRE_EQ(split.run(op,0,1),RAD_OK);
    }
    CHECK(all.ids==split.ids);CHECK(all.y==split.y);CHECK(all.saved==split.saved);
    REQUIRE_EQ(all.run(RAD_PLE_TREE_COMMIT),RAD_OK);
    REQUIRE_EQ(split.run(RAD_PLE_TREE_COMMIT,1,1),RAD_OK);
    REQUIRE_EQ(split.run(RAD_PLE_TREE_COMMIT,0,1),RAD_OK);
    CHECK(all.tokhist==split.tokhist);CHECK(all.history==split.history);
}

TEST(ple_tree_long_prefill_then_tree) {
    for (int capacity:{16,65}) for (int ring:{0,1}) for (int ctx:{0,3}) {
        Fixture f(ring,capacity);f.cu={2,30,37};f.ctx[0]=ctx;
        const Fixture anchor=f;std::vector<int> path(28);
        for (int j=0;j<28;++j) path[j]=j;
        const auto linear=oracle(anchor,0,path);
        auto t=f.operands(RAD_PLE_TREE_CONV,0,1,true);t[9]={};auto p=f.params(true,28);
        REQUIRE_EQ(launch(RAD_PLE_TREE_CONV,t,p),RAD_OK);
        for (int j=0;j<28;++j) for (int ch=0;ch<Fixture::N;++ch)
            CHECK_EQ(f.y[(2+j)*Fixture::N+ch],linear.y[j*Fixture::N+ch]);
        Fixture expected=anchor;expect_commit(expected,anchor,0,path);
        CHECK(f.history==expected.history);CHECK(f.tokhist==anchor.tokhist);
        /* The unchanged ngram prefill leaves the canonical history computed by
         * the independent oracle. Convolution prefill owns only its own state. */
        f.tokhist=expected.tokhist;f.ctx[0]+=28;f.cu={30,37,40};
        const int parent[]={-1,0,0,1,2,1,4};std::copy(parent,parent+7,f.parent.begin()+30);
        Fixture tree_anchor=f;
        REQUIRE_EQ(f.run(RAD_PLE_TREE_IDS,0,1),RAD_OK);REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV,0,1),RAD_OK);
        for (int node=0;node<7;++node) compare_node(f,tree_anchor,0,node);
        expected=tree_anchor;expect_commit(expected,tree_anchor,0,{0,1,3});
        REQUIRE_EQ(f.run(RAD_PLE_TREE_COMMIT,0,1),RAD_OK);
        CHECK(f.tokhist==expected.tokhist);CHECK(f.history==expected.history);
    }
}

TEST(ple_tree_linear_optional_operands_and_raw_bits) {
    Fixture f(0);f.cu={3,10,15};f.ctx[0]=0;
    const Fixture anchor=f;const std::vector<int> path{0,1,2,3,4,5,6};
    const auto expected=oracle(anchor,0,path,false);
    auto t=f.operands(RAD_PLE_TREE_CONV,0,1,true);t[7]={};auto p=f.params();
    REQUIRE_EQ(launch(RAD_PLE_TREE_CONV,t,p),RAD_OK);
    for (int j=0;j<7;++j) for (int ch=0;ch<Fixture::N;++ch)
        CHECK_EQ(f.y[(3+j)*Fixture::N+ch],expected.y[j*Fixture::N+ch]);
    /* These BF16 NaNs would change under a float roundtrip; saved_raw must not. */
    f.x[3*Fixture::XP]=0x7f81;f.x[3*Fixture::XP+1]=0xffa5;f.x[3*Fixture::XP+2]=0x8000;
    REQUIRE_EQ(launch(RAD_PLE_TREE_CONV,t,p),RAD_OK);
    CHECK_EQ(f.saved[3*Fixture::N],0x7f81);CHECK_EQ(f.saved[3*Fixture::N+1],0xffa5);
    CHECK_EQ(f.saved[3*Fixture::N+2],0x8000);
    f.len[0]=1;
    REQUIRE_EQ(f.run(RAD_PLE_TREE_COMMIT,0,1,true),RAD_OK);
    CHECK_EQ(f.history[f.ho(f.ci[0],0,f.hist()-1)],0x7f81);
    CHECK_EQ(f.history[f.ho(f.ci[0],1,f.hist()-1)],0xffa5);
    CHECK_EQ(f.history[f.ho(f.ci[0],2,f.hist()-1)],0x8000);
    CHECK_EQ(f.tokhist[f.ti[0]*Fixture::RP],2);
    CHECK_EQ(f.tokhist[f.ti[0]*Fixture::RP+1],f.tokens[3]);
    ++f.ctx[0];
    /* The accepted rank2 token-history and rank1 slot-index forms. */
    t=f.operands(RAD_PLE_TREE_IDS,0,1,true);t[1].rank=2;t[1].shape[1]=Fixture::RT;t[1].stride[1]=1;
    t[2].rank=1;t[2].stride[0]=2;
    REQUIRE_EQ(launch(RAD_PLE_TREE_IDS,t,p),RAD_OK);
}

TEST(ple_tree_metadata_rejection_is_atomic) {
    for (int fault=0;fault<10;++fault) {
        Fixture f(1);REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV),RAD_OK);
        const auto tok=f.tokhist;const auto hist=f.history;
        if (fault==0) f.path[17]=4;             // not a root child
        if (fault==1) f.path[0]=1;             // root omitted
        if (fault==2) f.len[1]=0;
        if (fault==3) f.len[1]=17;
        if (fault==4) f.ti[2]=f.ti[0];
        if (fault==5) f.ci[2]=f.ci[0];
        if (fault==6) f.parent[14]=4;           // self parent
        if (fault==7) f.ctx[1]=-1;
        if (fault==8) f.cu[2]=Fixture::T+1;
        if (fault==9) f.ci[2]=Fixture::Slots;
        CHECK(f.run(RAD_PLE_TREE_COMMIT)!=RAD_OK);
        CHECK(f.tokhist==tok);CHECK(f.history==hist);
    }
    Fixture f;const auto ids=f.ids;const auto y=f.y;const auto saved=f.saved;
    f.parent[14]=99;
    CHECK(f.run(RAD_PLE_TREE_IDS)!=RAD_OK);CHECK(f.ids==ids);
    CHECK(f.run(RAD_PLE_TREE_CONV)!=RAD_OK);CHECK(f.y==y);CHECK(f.saved==saved);
    f=Fixture();f.vocab.back()=0;CHECK(f.run(RAD_PLE_TREE_IDS)!=RAD_OK);CHECK(f.ids==ids);
    f=Fixture();f.offset.back()=INT32_MAX;CHECK(f.run(RAD_PLE_TREE_IDS)!=RAD_OK);CHECK(f.ids==ids);
    f=Fixture();f.ctx.back()=INT32_MAX;CHECK(f.run(RAD_PLE_TREE_CONV)!=RAD_OK);CHECK(f.y==y);
    f=Fixture();f.ci[2]=f.ci[0];const auto hist=f.history;
    CHECK(f.run(RAD_PLE_TREE_CONV,0,2,true,true)!=RAD_OK);CHECK(f.history==hist);
}

TEST(ple_tree_schema_shapes_and_argument_validation) {
    CHECK_EQ(rad_ple_tree_schema_count(),3);CHECK_EQ(rad_ref_ple_tree_count(),3);
    CHECK(rad_ple_tree_schema_at(-1)==nullptr);CHECK(rad_ref_ple_tree_at(3)==nullptr);
    Fixture f;auto p=f.params();p[0].kind=RAD_P_RANGE;p[0].ival=1;p[0].ihi=16;
    for (int op=0;op<3;++op) {
        const auto* schema=rad_ple_tree_schema_at(op);const auto* row=rad_ref_ple_tree_at(op);
        CHECK_EQ(schema->n_params,15);CHECK_EQ(schema->n_operands,op==2 ? 11 : 10);
        CHECK_EQ(row->domain,RAD_DOMAIN_HOST);
        for (int i=0;i<schema->n_operands;++i) {
            RadOpdDesc d{};CHECK_EQ(row->opd_shape(p.data(),(int)p.size(),i,&d),RAD_OK);
            CHECK(d.rank>0);
        }
        RadOpdDesc d{};CHECK_EQ(row->opd_shape(p.data(),(int)p.size(),schema->n_operands,&d),RAD_E_INVAL);
    }
    CHECK_EQ(rad_ple_tree_schema_at(0)->operands[1].role,RAD_OPD_IN);
    CHECK_EQ(rad_ple_tree_schema_at(1)->operands[2].role,RAD_OPD_INOUT);
    CHECK_EQ(rad_ple_tree_schema_at(1)->operands[9].optional,1);
    RadOpdDesc d{};REQUIRE_EQ(rad_ple_tree_ids_shape(p.data(),(int)p.size(),9,&d),RAD_OK);
    CHECK_EQ(d.shape[0],32);CHECK_EQ(d.shape[1],Fixture::H);
    p=f.params(true,128);REQUIRE_EQ(rad_ple_tree_conv_shape(p.data(),(int)p.size(),8,&d),RAD_OK);
    CHECK_EQ(d.shape[0],256);CHECK_EQ(rad_ple_tree_commit_shape(p.data(),(int)p.size(),0,&d),RAD_E_UNSUPPORTED);
    for (int fault=0;fault<12;++fault) {
        p=f.params();auto t=f.operands(RAD_PLE_TREE_CONV);RadPleTreeCall c{};
        if (fault==0) t[9]={};
        if (fault==1) t[0].stride[0]=Fixture::N-1;
        if (fault==2) t[2].shape[2]=8;
        if (fault==3) t[8].data=f.x.data()+1;
        if (fault==4) t[9].data=f.y.data()+1;
        if (fault==5) t[6].dtype=RAD_I64;
        if (fault==6) p.erase(p.begin()+4); // eos is required even when zero is valid
        if (fault==7) p[7].ival=15;
        if (fault==8) p[0].ival=17;
        if (fault==9) p[6].ival=22;
        if (fault==10) p.back().sval="other";
        if (fault==11) t[0].stride[1]=-1;
        RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
        CHECK(rad_ple_tree_args(&a,RAD_PLE_TREE_CONV,&c)!=RAD_OK);
    }
    p=f.params(true,28);auto t=f.operands(RAD_PLE_TREE_CONV);RadPleTreeCall c{};
    RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();
    CHECK_EQ(rad_ple_tree_args(&a,RAD_PLE_TREE_CONV,&c),RAD_E_INVAL); // prefill forbids parent
    p=f.params();p[1].ival=10240;p[2].ival=16;
    REQUIRE_EQ(rad_ple_tree_conv_shape(p.data(),(int)p.size(),8,&d),RAD_OK);CHECK_EQ(d.shape[1],10240);
}

TEST(ple_tree_full_capacity_and_history_extremes) {
    for (int capacity:{16,65}) for (int ring:{0,1}) for (int hist:{1,64}) {
        Fixture f(ring,capacity);f.width=hist==1 ? 2 : 5;f.dilation=hist==1 ? 1 : 16;
        f.rc=hist;f.cp=hist+2;
        f.w.assign(Fixture::N*f.width,rad_f32_to_bf16(0.125f));
        f.history.assign(Fixture::Slots*Fixture::N*f.cp,rad_f32_to_bf16(-0.25f));
        f.ctx[0]=hist-1;f.cu={2,2+capacity,7+capacity};
        for (int j=0;j<capacity;++j) f.parent[2+j]=j==0 ? -1 : j==2 ? 0 : j-1;
        f.path[0]=0;
        for (int j=1;j<capacity-2;++j) f.path[j]=j+1;
        f.len[0]=capacity-2;const Fixture anchor=f;
        REQUIRE_EQ(f.run(RAD_PLE_TREE_IDS,0,1),RAD_OK);
        REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV,0,1),RAD_OK);
        for (int node=0;node<capacity;++node) compare_node(f,anchor,0,node);
        Fixture expected=anchor;
        expect_commit(expected,anchor,0,std::vector<int>(f.path.begin(),f.path.begin()+f.len[0]));
        REQUIRE_EQ(f.run(RAD_PLE_TREE_COMMIT,0,1),RAD_OK);
        CHECK(f.tokhist==expected.tokhist);CHECK(f.history==expected.history);
    }
}

TEST(ple_tree_sixteen_node_fanout_and_sixteen_token_ancestry) {
    for (int ring:{0,1}) for (bool chain:{false,true}) for (int ctx:{0,8,12}) {
        Fixture f(ring);f.cu={2,18,23};f.ctx[0]=ctx;
        f.tokens[7]=2;f.tokens[13]=2;
        f.tokhist[f.ti[0]*Fixture::RP+1]=2;
        for (int j=0;j<16;++j) f.parent[2+j]=j==0 ? -1 : chain ? j-1 : 0;
        const auto accepted=chain ? ancestry(f,0,15) : ancestry(f,0,7);
        CHECK_EQ(accepted.size(),chain ? 16u : 2u);
        std::copy(accepted.begin(),accepted.end(),f.path.begin());f.len[0]=(int)accepted.size();
        const Fixture anchor=f;
        CHECK_EQ(f.hist(),9);CHECK_EQ(f.cu[1]-f.cu[0],16);
        REQUIRE_EQ(f.run(RAD_PLE_TREE_IDS,0,1),RAD_OK);
        REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV,0,1),RAD_OK);
        CHECK(f.tokhist==anchor.tokhist);CHECK(f.history==anchor.history);
        for (int node=0;node<16;++node) compare_node(f,anchor,0,node);
        Fixture expected=anchor;expect_commit(expected,anchor,0,accepted);
        REQUIRE_EQ(f.run(RAD_PLE_TREE_COMMIT,0,1),RAD_OK);
        CHECK(f.tokhist==expected.tokhist);CHECK(f.history==expected.history);
        f.ctx[0]+=f.len[0];f.cu={30,31,32};f.parent[30]=-1;
        const Fixture next=f;
        REQUIRE_EQ(f.run(RAD_PLE_TREE_IDS,0,1),RAD_OK);
        REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV,0,1),RAD_OK);
        compare_node(f,next,0,0);
    }
}

TEST(ple_tree_wide_ancestry_selected_history_and_next_request) {
    for (int capacity:{16,65}) for (int ring:{0,1}) for (bool chain:{false,true}) {
        Fixture f(ring,capacity);
        /* Two full sequences expose physical row pitch mistakes. The second
         * chain crosses the history/ring length; the first can skip siblings. */
        f.cu={2,2+capacity,2+2*capacity};f.ctx={1,25};
        for (int s=0;s<2;++s) {
            for (int node=0;node<capacity;++node)
                f.parent[f.cu[s]+node]=node==0 ? -1 : chain || s==1 ? node-1 : (node-1)/8;
            const auto accepted=ancestry(f,s,capacity-1-s);
            std::copy(accepted.begin(),accepted.end(),f.path.begin()+s*f.pitch);
            f.len[s]=(int)accepted.size();
        }
        const Fixture anchor=f;
        REQUIRE_EQ(f.run(RAD_PLE_TREE_IDS),RAD_OK);
        REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV),RAD_OK);
        CHECK(f.tokhist==anchor.tokhist);CHECK(f.history==anchor.history);
        for (int s=0;s<2;++s) for (int node=0;node<capacity;++node) compare_node(f,anchor,s,node);
        for (int row:{0,1,f.cu.back()}) {
            for (int h=0;h<Fixture::H;++h) CHECK_EQ(f.ids[row*Fixture::H+h],-901);
            for (int ch=0;ch<Fixture::N;++ch) {
                CHECK_EQ(f.y[row*Fixture::N+ch],0x7fc1);CHECK_EQ(f.saved[row*Fixture::N+ch],0x7fa1);
            }
        }
        Fixture expected=anchor;
        for (int s=0;s<2;++s) expect_commit(expected,anchor,s,
            std::vector<int>(f.path.begin()+s*f.pitch,f.path.begin()+s*f.pitch+f.len[s]));
        REQUIRE_EQ(f.run(RAD_PLE_TREE_COMMIT),RAD_OK);
        CHECK(f.tokhist==expected.tokhist);CHECK(f.history==expected.history);
        /* A restored/checkpointed request uses the same untouched padding and
         * canonical token/ring state, with no conversion or rejected sibling. */
        for (int s=0;s<2;++s) f.ctx[s]+=f.len[s];
        f.cu={f.rows-3,f.rows-2,f.rows-1};
        f.parent[f.cu[0]]=-1;f.parent[f.cu[1]]=-1;
        f.len={1,1};f.path[0]=0;f.path[f.pitch]=0;
        const Fixture next=f;
        REQUIRE_EQ(f.run(RAD_PLE_TREE_IDS),RAD_OK);REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV),RAD_OK);
        compare_node(f,next,0,0);compare_node(f,next,1,0);
        expected=next;
        for (int s=0;s<2;++s) expect_commit(expected,next,s,{0});
        REQUIRE_EQ(f.run(RAD_PLE_TREE_COMMIT),RAD_OK);
        CHECK(f.tokhist==expected.tokhist);CHECK(f.history==expected.history);
    }
}

TEST(ple_tree_runtime_path_pitch_is_independent_of_capacity) {
    for (int capacity:{16,65}) {
        Fixture compact(1,capacity,16),wide(1,capacity,65);
        for (int op:{RAD_PLE_TREE_IDS,RAD_PLE_TREE_CONV,RAD_PLE_TREE_COMMIT}) {
            REQUIRE_EQ(compact.run(op),RAD_OK);REQUIRE_EQ(wide.run(op),RAD_OK);
        }
        CHECK(compact.ids==wide.ids);CHECK(compact.y==wide.y);CHECK(compact.saved==wide.saved);
        CHECK(compact.tokhist==wide.tokhist);CHECK(compact.history==wide.history);
    }
}

TEST(ple_tree_wide_boundary_refusal_is_atomic) {
    for (int fault=0;fault<10;++fault) {
        Fixture f(1,65);f.cu={2,67,132};
        for (int s=0;s<2;++s) {
            for (int node=0;node<65;++node) f.parent[f.cu[s]+node]=node-1;
            for (int node=0;node<65;++node) f.path[s*f.pitch+node]=node;
        }
        f.len={65,65};REQUIRE_EQ(f.run(RAD_PLE_TREE_CONV),RAD_OK);
        auto t=f.operands(RAD_PLE_TREE_COMMIT);auto p=f.params();
        if (fault==0) p[7].ival=66;              // unsupported declared capacity
        if (fault==1) p[0].ival=66;              // unsupported query band
        if (fault==2) ++f.cu[2];                 // 66 nodes in second sequence
        if (fault==3) f.path[65+64]=65;          // node66, one beyond node64
        if (fault==4) f.path[65+63]=62;          // repeated node / wrong parent
        if (fault==5) f.path[65]=1;              // omitted root
        if (fault==6) f.parent[131]=64;          // node64 self-parent
        if (fault==7) f.len[1]=66;               // longer than physical path
        if (fault==8) t[9].shape[1]=66;          // unsupported physical pitch
        if (fault==9) {                         // legal P16, illegal 65-entry path
            t[9]=tensor(f.path.data(),RAD_I32,{2,16});
        }
        const auto tok=f.tokhist;const auto hist=f.history;
        CHECK(launch(RAD_PLE_TREE_COMMIT,t,p)!=RAD_OK);
        CHECK(f.tokhist==tok);CHECK(f.history==hist);
        if (fault==2 || fault==6) {
            const auto ids=f.ids;const auto y=f.y;const auto raw=f.saved;
            CHECK(f.run(RAD_PLE_TREE_IDS)!=RAD_OK);CHECK(f.ids==ids);
            CHECK(f.run(RAD_PLE_TREE_CONV)!=RAD_OK);CHECK(f.y==y);CHECK(f.saved==raw);
        }
    }
    Fixture f(0,65);auto p=f.params();RadOpdDesc d{};
    REQUIRE_EQ(rad_ple_tree_commit_shape(p.data(),(int)p.size(),9,&d),RAD_OK);
    CHECK_EQ(d.shape[0],2);CHECK_EQ(d.shape[1],65);
    REQUIRE_EQ(rad_ple_tree_ids_shape(p.data(),(int)p.size(),9,&d),RAD_OK);
    CHECK_EQ(d.shape[0],130);CHECK_EQ(d.shape[1],Fixture::H);
}

TEST(ple_tree_max_ngram_eos_matches_linear_oracle) {
    constexpr int T=10,H=14,K=7;
    std::vector<int32_t> tok{90,91,2,7,8,2,13,-5,92,93},parents(T,-99);
    const int pp[]={-1,0,0,1,2,4};std::copy(pp,pp+6,parents.begin()+2);
    std::vector<int32_t> cu{2,8},ctx{2},slot{0},history{44,45,46,47,48,2,19,901,902},out(T*H,-99);
    std::vector<int64_t> mult{INT64_MAX,INT64_MIN,-17,31,47,-59,71,-83},vocab(H,97),offset(H);
    auto p=Fixture().params();p[2].ival=H;p[3].ival=8;
    std::vector<RadTensor> t={tensor(tok.data(),RAD_I32,{T}),tensor(history.data(),RAD_I32,{1,9}),
        tensor(slot.data(),RAD_I32,{1}),tensor(cu.data(),RAD_I32,{2}),tensor(parents.data(),RAD_I32,{T}),
        tensor(ctx.data(),RAD_I32,{1}),tensor(mult.data(),RAD_I64,{8}),tensor(vocab.data(),RAD_I64,{H}),
        tensor(offset.data(),RAD_I64,{H}),tensor(out.data(),RAD_I32,{T,H})};
    const auto original=history;REQUIRE_EQ(launch(RAD_PLE_TREE_IDS,t,p),RAD_OK);CHECK(history==original);
    for (int node=0;node<6;++node) {
        std::vector<int32_t> linear;
        for (int j=node;j>=0;j=parents[2+j]) linear.push_back(tok[2+j]);
        std::reverse(linear.begin(),linear.end());
        std::vector<int32_t> h(K,2);h[6]=19;
        std::vector<int32_t> cc{0,(int32_t)linear.size()},ids(linear.size()*H);
        std::vector<RadTensor> old={tensor(linear.data(),RAD_I32,{(int64_t)linear.size()}),
            tensor(h.data(),RAD_I32,{1,K}),tensor(cc.data(),RAD_I32,{2}),t[6],t[7],t[8],{}, {}, {},
            tensor(ids.data(),RAD_I32,{(int64_t)linear.size(),H})};
        RadArgs a{};a.t=old.data();a.n_t=(int)old.size();a.p=p.data();a.n_p=(int)p.size();
        REQUIRE_EQ(ref_ngram_ids(&a,nullptr),RAD_OK);
        for (int head=0;head<H;++head) CHECK_EQ(out[(2+node)*H+head],ids[(linear.size()-1)*H+head]);
    }
    for (int row:{0,1,8,9}) for (int head=0;head<H;++head) CHECK_EQ(out[row*H+head],-99);
}

RAD_TEST_MAIN()
