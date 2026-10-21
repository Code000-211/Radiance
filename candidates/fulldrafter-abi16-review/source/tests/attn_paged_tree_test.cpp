#include "rad_test.h"
#include "../mine/attn_paged_tree.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

namespace {
RadParam pi(const char* key,int64_t value) {
    RadParam p{}; p.key=key; p.kind=RAD_P_INT; p.ival=value; return p;
}
RadParam ps(const char* key,const char* value) {
    RadParam p{}; p.key=key; p.kind=RAD_P_STR; p.sval=value; return p;
}
RadTensor tensor(void* data,uint32_t dtype,std::initializer_list<int64_t> shape) {
    RadTensor t{}; t.data=data; t.dtype=dtype;
    for(int64_t d:shape) t.shape[t.rank++]=d;
    int64_t stride=1;
    for(int j=int(t.rank)-1;j>=0;--j) { t.stride[j]=stride; stride*=t.shape[j]; }
    return t;
}
struct Fixture {
    static constexpr int T=68, Width=3072, Pages=40, PagePitch=2080, TablePitch=20;
    std::vector<uint16_t> q=std::vector<uint16_t>(T*Width), out=std::vector<uint16_t>(T*Width+16,0x7fc1);
    std::vector<uint8_t> kv=std::vector<uint8_t>(Pages*PagePitch,0x7f);
    std::vector<int32_t> bt=std::vector<int32_t>(2*TablePitch,-99), cu{0,65,68}, used{70,5};
    std::vector<int32_t> parent=std::vector<int32_t>(T,0);
    std::vector<RadParam> p{pi("q_len",65),pi("max_tokens",T),pi("max_seqs",2),pi("max_ctx",80),
        pi("head_dim",256),pi("gqa",12),pi("n_head",12),pi("block_size",4),
        pi("causal",1),pi("window",0),ps("q_dtype","bf16"),ps("kv_dtype","fp8_e4m3")};
    Fixture() {
        parent[0]=-1; parent[16]=0; parent[32]=16; parent[64]=32;
        parent[65]=-1; parent[66]=0; parent[67]=1; // mixed non-tree causal chain
        for(int s=0;s<2;++s) for(int b=0;b<(used[s]+3)/4;++b) {
            const int page=s==0 ? 35-b : b+1;
            bt[s*TablePitch+b]=page;
            for(int off=0;off<4;++off) for(int d=0;d<256;++d) {
                const int pos=b*4+off, base=page*PagePitch+off*512;
                kv[base+d]=rad_f32_to_fp8e4m3(float((pos*3+d)%13-6)*0.0625f);
                kv[base+256+d]=rad_f32_to_fp8e4m3(float((pos*7+d*3)%31-15)*0.125f);
            }
        }
    }
    std::vector<RadTensor> tensors(bool rank3=false) {
        auto cache=tensor(kv.data(),RAD_F8E4M3,{Pages,1,4,512}); cache.stride[0]=PagePitch;
        auto table=tensor(bt.data(),RAD_I32,{2,18}); table.stride[0]=TablePitch;
        auto query=rank3 ? tensor(q.data(),RAD_BF16,{T,12,256}) : tensor(q.data(),RAD_BF16,{T,Width});
        auto result=rank3 ? tensor(out.data(),RAD_BF16,{T,12,256}) : tensor(out.data(),RAD_BF16,{T,Width});
        return {query,cache,table,tensor(used.data(),RAD_I32,{2}),tensor(cu.data(),RAD_I32,{3}),
            tensor(parent.data(),RAD_I32,{T}),result};
    }
    int run(bool rank3=false) {
        auto t=tensors(rank3); RadArgs a{};
        a.t=t.data(); a.n_t=int(t.size()); a.p=p.data(); a.n_p=int(p.size());
        return rad_ref_attn_paged_tree_launch(&a,nullptr);
    }
    // Gather paths, then evaluate normalized weights directly in long double.
    // The oracle does not use the implementation's boolean visibility table.
    void check() const {
        for(int s=0;s<2;++s) for(int node=0;node<cu[s+1]-cu[s];++node) {
            const int prefix=used[s]-(cu[s+1]-cu[s]), row=cu[s]+node;
            std::vector<int> positions;
            for(int j=0;j<prefix;++j) positions.push_back(j);
            for(int j=node;j>=0;j=parent[cu[s]+j]) positions.push_back(prefix+j);
            std::sort(positions.begin(),positions.end());
            auto element=[&](int pos,int d) {
                return rad_fp8e4m3_to_f32(kv[bt[s*TablePitch+pos/4]*PagePitch+(pos%4)*512+d]);
            };
            for(int h=0;h<12;++h) {
                std::vector<long double> weights; long double sum=0;
                for(int pos:positions) {
                    long double dot=0;
                    for(int d=0;d<256;++d)
                        dot+=static_cast<long double>(rad_bf16_to_f32(q[row*Width+h*256+d]))*element(pos,d);
                    weights.push_back(std::exp(dot/16)); sum+=weights.back();
                }
                for(int d=0;d<256;++d) {
                    long double expected=0;
                    for(size_t j=0;j<positions.size();++j) expected+=weights[j]*element(positions[j],256+d);
                    CHECK_EQ(out[row*Width+h*256+d],rad_f32_to_bf16(float(expected/sum)));
                }
            }
        }
        for(size_t j=T*Width;j<out.size();++j) CHECK_EQ(out[j],0x7fc1);
    }
};
}
TEST(attn_paged_tree_wide_ancestors_and_mixed_causal_sequence) {
    Fixture f; REQUIRE_EQ(f.run(),RAD_OK); f.check();
    for(size_t j=0;j<f.q.size();++j) f.q[j]=rad_f32_to_bf16(float(int(j%17)-8)*0.03125f);
    REQUIRE_EQ(f.run(true),RAD_OK); f.check();
}
TEST(attn_paged_tree_65_node_chain_and_root_only_star) {
    Fixture f;
    for(int j=1;j<65;++j) f.parent[j]=j-1;
    REQUIRE_EQ(f.run(),RAD_OK); f.check();
    for(int j=1;j<65;++j) f.parent[j]=0;
    REQUIRE_EQ(f.run(),RAD_OK); f.check();
}
TEST(attn_paged_tree_invalid_metadata_zeroes_only_affected_sequence) {
    Fixture f; f.parent[64]=64;
    REQUIRE_EQ(f.run(),RAD_OK);
    for(int j=0;j<65*Fixture::Width;++j) CHECK_EQ(f.out[j],0);
    CHECK(std::any_of(f.out.begin()+65*Fixture::Width,f.out.begin()+68*Fixture::Width,
        [](uint16_t x){return x!=0;}));
    f.parent[64]=32; f.parent[0]=0;
    REQUIRE_EQ(f.run(),RAD_OK);
    for(int j=0;j<65*Fixture::Width;++j) CHECK_EQ(f.out[j],0);
    f.parent[0]=-1; f.bt[0]=Fixture::Pages;
    REQUIRE_EQ(f.run(),RAD_OK);
    for(int j=0;j<65*Fixture::Width;++j) CHECK_EQ(f.out[j],0);
}
TEST(attn_paged_tree_rejects_unsupported_geometry_and_aliasing) {
    Fixture f; RadAttnPagedTreeCall c{}; auto t=f.tensors(); RadArgs a{};
    a.t=t.data(); a.n_t=int(t.size()); a.p=f.p.data(); a.n_p=int(f.p.size());
    REQUIRE_EQ(rad_attn_paged_tree_args(&a,&c),RAD_OK);
    CHECK_EQ(rad_attn_paged_tree_scratch(&a),int64_t(Fixture::T)*12*16*258*4);
    f.p[0].ival=66; CHECK_EQ(rad_attn_paged_tree_args(&a,&c),RAD_E_UNSUPPORTED); f.p[0].ival=65;
    f.p[4].ival=128; CHECK_EQ(rad_attn_paged_tree_args(&a,&c),RAD_E_UNSUPPORTED); f.p[4].ival=256;
    f.p[9].ival=32; CHECK_EQ(rad_attn_paged_tree_args(&a,&c),RAD_E_UNSUPPORTED); f.p[9].ival=0;
    t[RAD_ATP_OUT].data=t[RAD_ATP_Q].data; CHECK_EQ(rad_attn_paged_tree_args(&a,&c),RAD_E_INVAL);
    t=f.tensors(); a.t=t.data(); t[RAD_ATP_CU].stride[0]=2;
    CHECK_EQ(rad_attn_paged_tree_args(&a,&c),RAD_E_STRIDE);
}
RAD_TEST_MAIN()
