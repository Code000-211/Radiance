#include "rad_test.h"
#include "mine/ref_dloop.h"
#include "core/format/safetensors.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace {
RadDLoopConfig tiny(){
    RadDLoopConfig c{};c.n_seq=4;c.max_seqs=4;c.tap_rows=128;c.world_size=2;
    c.hidden_size=64;c.intermediate_size=128;c.num_layers=2;c.num_q_heads=4;c.num_kv_heads=2;c.head_dim=16;
    c.context_features_dim=48;c.target_width=32;c.proposal_length=8;c.context_window=32;
    c.active_vocab=509;c.vocab_size=512;c.selector_rank=8;c.conv_kernel_size=2;c.conv_group_size=16;
    c.vocab_size_shard=256;c.pool_size=64;c.budget=15;c.width=12;c.eps=1e-6f;c.rope_theta=1e7f;c.depth_bonus=-.2f;return c;
}
std::vector<RadParam> params(const RadDLoopConfig& c){
    std::vector<RadParam> p;
#define P(k) {RadParam v{};v.key=#k;v.kind=RAD_P_INT;v.ival=c.k;p.push_back(v);}
    P(n_seq);P(max_seqs);P(tap_rows);P(world_size);P(hidden_size);P(intermediate_size);P(num_layers);
    P(num_q_heads);P(num_kv_heads);P(head_dim);P(context_features_dim);P(target_width);P(proposal_length);
    P(context_window);P(active_vocab);P(vocab_size);P(selector_rank);P(conv_kernel_size);P(conv_group_size);
    P(vocab_lo);P(vocab_size_shard);P(pool_size);P(budget);P(width);
#undef P
    RadParam variant{};variant.key="variant";variant.kind=RAD_P_STR;variant.sval=c.pruned?"pruned":"fixed";p.push_back(variant);
#define F(k) {RadParam v{};v.key=#k;v.kind=RAD_P_F64;v.dval=c.k;p.push_back(v);}
    F(eps);F(rope_theta);F(depth_bonus);
#undef F
    RadParam dtype{};dtype.key="dtype";dtype.kind=RAD_P_STR;dtype.sval="bf16";p.push_back(dtype);return p;
}
struct Call {
    RadDLoopConfig c;std::vector<RadParam> p;std::vector<RadTensor> t;
    std::vector<std::vector<uint64_t>> storage;std::vector<float> scratch;
    Call(int op,RadDLoopConfig config):c(config),p(params(c)){
        auto* schema=rad_dloop_schema_at(op);t.resize(schema->n_operands);storage.resize(t.size());
        const RadShapeFn shapes[]={rad_dloop_ctx_append_shape,rad_dloop_base_shape,rad_dloop_pool_shape,rad_dloop_dartree_shape};
        for(size_t i=0;i<t.size();++i){RadOpdDesc d{};if(shapes[op](p.data(),(int)p.size(),(int)i,&d))throw std::runtime_error("bad shape");
            if(d.flags&RAD_OPD_F_ABSENT)continue;
            int64_t n=1;t[i].dtype=d.dtype;t[i].rank=d.rank;
            for(int j=(int)d.rank-1;j>=0;--j){t[i].shape[j]=d.shape[j];t[i].stride[j]=n;n*=d.shape[j];}
            storage[i].resize((n*(d.dtype==RAD_BF16?2:4)+7)/8);t[i].data=storage[i].data();}
        scratch.resize(c.n_seq*rad_dloop_scratch_elements(c));
    }
    RadArgs args(){RadArgs a{};a.t=t.data();a.n_t=(int)t.size();a.p=p.data();a.n_p=(int)p.size();a.scratch=scratch.data();a.scratch_bytes=scratch.size()*4;return a;}
    int run(int op){auto a=args();const RadLaunchFn fn[]={rad_ref_dloop_ctx_append,rad_ref_dloop_base,rad_ref_dloop_pool,rad_ref_dloop_dartree};return fn[op](&a,nullptr);}
    template<class T>T* data(int i){return static_cast<T*>(t[i].data);}
    void copy(int i,const void* src,size_t bytes){std::memcpy(t[i].data,src,bytes);}
    void bind(int i,void* ptr,uint32_t dtype,std::initializer_list<int64_t> shape){
        RadTensor v{};v.data=ptr;v.dtype=dtype;v.rank=(uint32_t)shape.size();int j=0;for(auto n:shape)v.shape[j++]=n;
        int64_t stride=1;for(int k=(int)v.rank-1;k>=0;--k){v.stride[k]=stride;stride*=v.shape[k];}t[i]=v;
    }
};
struct Fixture {
    rad::SafeTensorsFile student,teacher,golden;
    std::vector<uint16_t> packed;
    Fixture(){
        const std::string root=RAD_DLOOP_FIXTURES;
        if(student.open(root+"/dloop_sidecar.safetensors")||teacher.open(root+"/teacher-matrices.safetensors")||golden.open(root+"/golden.safetensors"))throw std::runtime_error("required tiny fixture unavailable");
        auto c=tiny();packed.resize(rad_dloop_weight_elements(c));
        auto add=[&](int w,int layer=0){std::string name="dloop.";if(w>=DL_INPUT_NORM&&w<=DL_DOWN_PROJ)name+="layers."+std::to_string(layer)+".";
            name+=rad_dloop_weight_name(w);const auto* t=student.find(name);
            if(!t||t->dtype!=RAD_BF16||t->numel()!=rad_dloop_weight_size(c,w))throw std::runtime_error("bad packed weight: "+name);
            std::memcpy(packed.data()+rad_dloop_weight_offset(c,w,layer),t->data,t->bytes);};
        for(int w=0;w<=DL_MASK_TOKEN;++w)add(w);
        for(int l=0;l<c.num_layers;++l)for(int w=DL_INPUT_NORM;w<=DL_DOWN_PROJ;++w)add(w,l);
        for(int w=DL_FINAL_NORM;w<=DL_SELECTOR_PROJ;++w)add(w);
    }
    const rad::StTensor& get(const rad::SafeTensorsFile& file,const std::string& key){auto* t=file.find(key);if(!t)throw std::runtime_error("missing fixture tensor: "+key);return *t;}
    template<class T>const T* g(const std::string& key){return reinterpret_cast<const T*>(get(golden,key).data);}
    const uint16_t* teacher_rows(const char* key){return reinterpret_cast<const uint16_t*>(get(teacher,key).data);}
    void setup_base(Call& call){
        const auto c=call.c;const int slots[]={2,0,3,1};const auto* lengths=g<int32_t>("context_lengths");const auto* context=g<uint16_t>("context");
        const auto* emb=teacher_rows("model.language_model.embed_tokens.weight");
        call.copy(5,packed.data(),packed.size()*2);
        call.copy(6,teacher_rows("lm_head.weight")+(int64_t)c.vocab_lo*c.target_width,(size_t)c.vocab_size_shard*c.target_width*2);
        for(int s=0;s<c.n_seq;++s){const int slot=slots[s],len=lengths[s];call.data<int32_t>(3)[s]=slot;call.data<int32_t>(1)[slot]=len%c.context_window;call.data<int32_t>(2)[slot]=len;
            std::copy(context+(s*c.context_window+c.context_window-len)*c.context_features_dim,context+(s+1)*c.context_window*c.context_features_dim,
                call.data<uint16_t>(0)+(int64_t)slot*c.context_window*c.context_features_dim);
            std::copy(emb+g<int64_t>("anchors")[s]*c.target_width,emb+(g<int64_t>("anchors")[s]+1)*c.target_width,call.data<uint16_t>(4)+s*c.target_width);}
    }
    void setup_pool(Call& pool,const Call& base0,const Call& base1){
        const auto c=pool.c;const int rows=c.n_seq*c.proposal_length*c.pool_size;
        std::memcpy(pool.t[0].data,base0.t[9].data,rows*4);std::memcpy(pool.data<int32_t>(0)+rows,base1.t[9].data,rows*4);
        std::memcpy(pool.t[1].data,base0.t[10].data,rows*4);std::memcpy(pool.data<float>(1)+rows,base1.t[10].data,rows*4);
        for(int s=0;s<c.n_seq;++s)pool.data<int32_t>(2)[s]=(int32_t)g<int64_t>("anchors")[s];
        pool.copy(3,teacher_rows("lm_head.weight")+(int64_t)c.vocab_lo*c.target_width,(size_t)c.vocab_size_shard*c.target_width*2);
        pool.copy(4,teacher_rows("model.language_model.embed_tokens.weight")+(int64_t)c.vocab_lo*c.target_width,(size_t)c.vocab_size_shard*c.target_width*2);
        pool.copy(5,get(student,"dloop.selector.pred_codebook.weight").data+(int64_t)c.vocab_lo*c.selector_rank*2,(size_t)c.vocab_size_shard*c.selector_rank*2);
        pool.copy(6,get(student,"dloop.selector.succ_codebook.weight").data+(int64_t)c.vocab_lo*c.selector_rank*2,(size_t)c.vocab_size_shard*c.selector_rank*2);
    }
};
void near(const float* actual,const float* expected,int64_t n,float atol,float rtol){
    float worst=0;int64_t bad=0;for(int64_t i=0;i<n;++i){float delta=std::abs(actual[i]-expected[i]);worst=std::max(worst,delta);if(!std::isfinite(actual[i])||delta>atol+rtol*std::abs(expected[i]))++bad;}
    std::fprintf(stderr,"    float parity n=%lld worst_abs=%g atol=%g rtol=%g bad=%lld\n",(long long)n,worst,atol,rtol,(long long)bad);CHECK_EQ(bad,0);
}
}

TEST(context_append_wrap_reset_and_transaction){
    auto c=tiny();c.n_seq=1;c.max_seqs=2;c.context_window=3;c.tap_rows=8;Call call(0,c);
    auto* taps=call.data<uint16_t>(0);for(int row=0;row<8;++row)for(int j=0;j<c.context_features_dim;++j)taps[row*c.context_features_dim+j]=(uint16_t)(row*100+j);
    call.data<int32_t>(2)[0]=1;call.data<int32_t>(3)[0]=1;
    std::fill(call.data<int32_t>(1),call.data<int32_t>(1)+16,-1);
    const int path[]={4,1,6,2};std::copy(path,path+4,call.data<int32_t>(1));REQUIRE_EQ(call.run(0),RAD_OK);
    CHECK_EQ(call.data<int32_t>(5)[1],1);CHECK_EQ(call.data<int32_t>(6)[1],3);
    const int physical[]={2,1,6};for(int t=0;t<3;++t)for(int j=0;j<c.context_features_dim;++j)
        CHECK_EQ(call.data<uint16_t>(4)[(3+t)*c.context_features_dim+j],taps[physical[t]*c.context_features_dim+j]);
    call.data<int32_t>(3)[0]=0;std::fill(call.data<int32_t>(1),call.data<int32_t>(1)+16,-1);call.data<int32_t>(1)[0]=7;
    REQUIRE_EQ(call.run(0),RAD_OK);CHECK_EQ(call.data<int32_t>(5)[1],2);CHECK_EQ(call.data<int32_t>(6)[1],3);
    std::vector<uint16_t> before(call.data<uint16_t>(4),call.data<uint16_t>(4)+2*3*c.context_features_dim);
    call.data<int32_t>(1)[2]=0;CHECK_EQ(call.run(0),RAD_E_INVAL);CHECK(std::equal(before.begin(),before.end(),call.data<uint16_t>(4)));
    call.data<int32_t>(3)[0]=1;std::fill(call.data<int32_t>(1),call.data<int32_t>(1)+16,-1);call.data<int32_t>(1)[0]=0;
    REQUIRE_EQ(call.run(0),RAD_OK);CHECK_EQ(call.data<int32_t>(5)[1],1);CHECK_EQ(call.data<int32_t>(6)[1],1);
}
TEST(golden_pipeline_padding_bidirectionality_and_tree){
    Fixture fixture;auto c=tiny();Call base0(1,c);fixture.setup_base(base0);REQUIRE_EQ(base0.run(1),RAD_OK);
    near(base0.data<float>(7),fixture.g<float>("ctxh"),4*32*64,2e-5f,1e-5f);
    near(base0.data<float>(8),fixture.g<float>("y_base"),4*8*32,3e-5f,2e-5f);
    std::vector<float> original(base0.data<float>(8),base0.data<float>(8)+4*8*32);
    // Invalid physical ring cells may contain NaNs and must never enter projection.
    for(int slot=0;slot<4;++slot)for(int t=base0.data<int32_t>(2)[slot];t<32;++t)
        std::fill(base0.data<uint16_t>(0)+(slot*32+t)*48,base0.data<uint16_t>(0)+(slot*32+t+1)*48,0x7fc1);
    REQUIRE_EQ(base0.run(1),RAD_OK);CHECK(std::equal(original.begin(),original.end(),base0.data<float>(8)));
    // A later visible prefix token must change earlier rows: triangular attention is wrong.
    const auto* embeddings=fixture.teacher_rows("model.language_model.embed_tokens.weight");
    const uint16_t* prefix0[]={embeddings+2*32,embeddings+3*32,embeddings+4*32};
    const uint16_t* prefix1[]={embeddings+2*32,embeddings+3*32,embeddings+5*32};
    std::array<float,8*32> y0{},y1{};rad_ref_dloop_forward(c,fixture.packed.data(),base0.data<float>(7),1,base0.data<uint16_t>(4),prefix0,3,y0.data(),base0.scratch.data());
    rad_ref_dloop_forward(c,fixture.packed.data(),base0.data<float>(7),1,base0.data<uint16_t>(4),prefix1,3,y1.data(),base0.scratch.data());
    float delta=0;for(int j=0;j<32;++j)delta=std::max(delta,std::abs(y0[j]-y1[j]));CHECK(delta>1e-6f);
    c.vocab_lo=256;Call base1(1,c);fixture.setup_base(base1);REQUIRE_EQ(base1.run(1),RAD_OK);
    CHECK(std::equal(original.begin(),original.end(),base1.data<float>(8)));
    c.vocab_lo=0;Call pool0(2,c);fixture.setup_pool(pool0,base0,base1);REQUIRE_EQ(pool0.run(2),RAD_OK);
    c.vocab_lo=256;Call pool1(2,c);fixture.setup_pool(pool1,base0,base1);REQUIRE_EQ(pool1.run(2),RAD_OK);
    for(int i=0;i<4*8*64;++i){CHECK_EQ(pool0.data<int32_t>(7)[i],fixture.g<int64_t>("pool_ids")[i]);CHECK_EQ(pool1.data<int32_t>(7)[i],pool0.data<int32_t>(7)[i]);}
    near(pool0.data<float>(8),fixture.g<float>("pool_unary"),4*8*64,3e-5f,2e-5f);
    // Each rank contributes only owned BF16 rows: the exact all-reduce is bitwise OR
    // because one contributor is zero. Check ownership before reconstructing.
    for(int plane=9;plane<=13;++plane){int64_t n=1;for(unsigned d=0;d<pool0.t[plane].rank;++d)n*=pool0.t[plane].shape[d];
        for(int64_t j=0;j<n;++j){auto a=pool0.data<uint16_t>(plane)[j],b=pool1.data<uint16_t>(plane)[j];CHECK(a==0||b==0);pool0.data<uint16_t>(plane)[j]=a|b;}}
    for(int variant=0;variant<2;++variant){c.vocab_lo=0;c.pruned=variant;Call tree(3,c);
        tree.t[0]=base0.t[7];tree.t[1]=base0.t[2];tree.t[2]=base0.t[3];tree.t[3]=base0.t[4];tree.t[4]=base0.t[5];tree.t[5]=base0.t[8];
        for(int i=6;i<=12;++i)tree.t[i]=pool0.t[i+1];
        std::array<int32_t,4> anchors{};for(int s=0;s<4;++s)anchors[s]=(int32_t)fixture.g<int64_t>("anchors")[s];
        std::array<int32_t,4*33> packed{};tree.bind(16,anchors.data(),RAD_I32,{4});tree.bind(17,packed.data(),RAD_I32,{4,33});
        REQUIRE_EQ(tree.run(3),RAD_OK);const std::string name=variant?"pruned":"fixed";
        for(int s=0;s<4;++s){const int n=tree.data<int32_t>(15)[s];CHECK(n<=15);CHECK_EQ(n,fixture.g<int32_t>(name+".n_nodes")[s]);CHECK_EQ(packed[s*33],n+1);CHECK_EQ(packed[s*33+1],anchors[s]);CHECK_EQ(packed[s*33+17],-1);
            int depth[16]{},quotas[9]{},last_depth=0;
            for(int j=0;j<n;++j){int parent=tree.data<int32_t>(14)[s*16+j],token=tree.data<int32_t>(13)[s*16+j];REQUIRE(parent>=-1&&parent<j);
                depth[j]=parent<0?1:depth[parent]+1;CHECK(depth[j]>=last_depth);last_depth=depth[j];++quotas[depth[j]];
                for(int k=0;k<j;++k)CHECK(parent!=tree.data<int32_t>(14)[s*16+k]||token!=tree.data<int32_t>(13)[s*16+k]);
                CHECK_EQ(token,fixture.g<int64_t>(name+".tokens")[s*16+j+1]);CHECK_EQ(parent,fixture.g<int64_t>(name+".parents")[s*16+j+1]-1);
                CHECK_EQ(depth[j],fixture.g<int64_t>(name+".depths")[s*16+j+1]);CHECK_EQ(packed[s*33+j+2],token);CHECK_EQ(packed[s*33+j+18],parent+1);}
            if(!variant){int remaining=15,front=1;for(int d=1;d<=8;++d){int quota=std::min({(remaining+8-d)/(9-d),remaining,front*64});CHECK_EQ(quotas[d],quota);remaining-=quota;front=quota;}}
            for(int j=n;j<16;++j){CHECK_EQ(tree.data<int32_t>(13)[s*16+j],-1);CHECK_EQ(tree.data<int32_t>(14)[s*16+j],-1);}}
    }
}
TEST(pool_cutoff_ties_and_small_gap){
    rad::SafeTensorsFile golden;REQUIRE_EQ(golden.open(std::string(RAD_DLOOP_FIXTURES)+"/near_tie.safetensors"),RAD_OK);
    auto* logits=golden.find("logits");auto* expected=golden.find("pool_ids");auto* unary=golden.find("pool_unary");REQUIRE(logits&&expected&&unary);
    auto c=tiny();c.n_seq=1;c.max_seqs=1;Call pool(2,c);pool.data<int32_t>(2)[0]=0;
    const auto* values=reinterpret_cast<const float*>(logits->data);
    for(int g=0;g<2;++g)for(int t=0;t<8;++t){std::vector<int> ids;
        for(int id=g*256;id<std::min((g+1)*256,509);++id)ids.push_back(id);
        std::stable_sort(ids.begin(),ids.end(),[&](int x,int y){return values[t*509+x]>values[t*509+y];});
        for(int j=0;j<64;++j){int off=(g*8+t)*64+j;pool.data<int32_t>(0)[off]=ids[j];pool.data<float>(1)[off]=values[t*509+ids[j]];}}
    REQUIRE_EQ(pool.run(2),RAD_OK);for(int i=0;i<8*64;++i){CHECK_EQ(pool.data<int32_t>(7)[i],reinterpret_cast<const int64_t*>(expected->data)[i]);CHECK_EQ(pool.data<float>(8)[i],reinterpret_cast<const float*>(unary->data)[i]);}
}
TEST(base_ties_exclude_inactive_vocab_and_validate_shapes){
    auto c=tiny();c.n_seq=1;c.max_seqs=1;c.vocab_lo=0;c.vocab_size_shard=512;Call base(1,c);
    base.data<int32_t>(1)[0]=0;base.data<int32_t>(2)[0]=1;base.data<int32_t>(3)[0]=0;
    // Zero output yields an exact all-vocabulary tie; deterministic local top-k
    // must retain low IDs, never inactive padding IDs 509..511.
    REQUIRE_EQ(base.run(1),RAD_OK);for(int t=0;t<8;++t)for(int j=0;j<64;++j)CHECK_EQ(base.data<int32_t>(9)[t*64+j],j);
    auto args=base.args();RadDLoopConfig out{};args.scratch_bytes=0;CHECK_EQ(rad_dloop_args(&args,1,&out),RAD_E_SHAPE);
    args=base.args();base.t[0].stride[1]++;CHECK_EQ(rad_dloop_args(&args,1,&out),RAD_E_SHAPE);base.t[0].stride[1]--;
    base.data<int32_t>(2)[0]=33;CHECK_EQ(base.run(1),RAD_E_INVAL);
}
TEST(separate_tap_planes_and_narrowed_batch){
    auto c=tiny();c.context_features_dim=3*c.target_width;Call call(0,c);
    std::array<std::array<uint16_t,5*32>,3> planes{};
    for(int p=0;p<3;++p)for(int r=0;r<5;++r)for(int j=0;j<32;++j)planes[p][r*32+j]=(uint16_t)(p*1000+r*100+j);
    std::array<int32_t,16> path{};path.fill(-1);path[0]=3;path[1]=1;
    int32_t slot=2,reset=1;call.bind(0,planes[0].data(),RAD_BF16,{5,32});
    call.bind(1,path.data(),RAD_I32,{1,16});call.bind(2,&slot,RAD_I32,{1});call.bind(3,&reset,RAD_I32,{1});
    call.bind(7,planes[1].data(),RAD_BF16,{5,32});call.bind(8,planes[2].data(),RAD_BF16,{5,32});
    REQUIRE_EQ(call.run(0),RAD_OK);CHECK_EQ(call.data<int32_t>(5)[2],2);CHECK_EQ(call.data<int32_t>(6)[2],2);
    for(int r=0;r<2;++r)for(int p=0;p<3;++p)for(int j=0;j<32;++j)
        CHECK_EQ(call.data<uint16_t>(4)[(2*32+r)*96+p*32+j],planes[p][path[r]*32+j]);
    call.t[8].data=nullptr;CHECK_EQ(call.run(0),RAD_E_INVAL);
}
TEST(pruned_zero_transition_ancestor_ties_and_budget_boundaries){
    auto c=tiny();c.n_seq=1;c.max_seqs=1;c.hidden_size=2;c.intermediate_size=2;c.num_layers=1;
    c.num_q_heads=1;c.num_kv_heads=1;c.head_dim=2;c.context_features_dim=2;c.target_width=2;
    c.selector_rank=1;c.conv_group_size=1;c.conv_kernel_size=0;c.context_window=1;c.pool_size=1;c.depth_bonus=0;
    for(int budget=1;budget<=15;++budget)for(int variant=0;variant<2;++variant){
        c.budget=budget;c.pruned=variant;Call tree(3,c);tree.data<int32_t>(1)[0]=1;tree.data<int32_t>(2)[0]=0;
        for(int t=0;t<8;++t)tree.data<int32_t>(6)[t]=10+t;
        REQUIRE_EQ(tree.run(3),RAD_OK);const int n=std::min(budget,8);CHECK_EQ(tree.data<int32_t>(15)[0],n);
        for(int j=0;j<n;++j){CHECK_EQ(tree.data<int32_t>(13)[j],10+j);CHECK_EQ(tree.data<int32_t>(14)[j],j-1);}
    }
}
RAD_TEST_MAIN()
